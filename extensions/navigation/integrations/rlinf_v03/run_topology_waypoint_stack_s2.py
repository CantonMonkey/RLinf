"""Bounded train-only S2 closure for the topology/metric-waypoint stack plugin.

The runner opens exactly one fixed R2R ``train`` episode, executes two hybrid
high-level decisions through the real Habitat service, performs one PPO update,
and verifies an independent checkpoint restore.  It is a stack/plugin closure
test, not a navigation-quality benchmark.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from navrl.stack import (
    DecisionCollector,
    compute_duration_aware_gae,
    load_checkpoint_verified,
    save_checkpoint_atomic,
)
from integrations.rlinf_v03.topology_waypoint_stack import (
    EpisodeTopologyState,
    RLinfVLNTaskAdapter,
    STOP_CANDIDATE_ID,
    TopologyWaypointActionExecutor,
    TopologyWaypointCheckpointAdapter,
    TopologyWaypointPolicy,
    TopologyWaypointPolicyAdapter,
)
from integrations.rlinf_v03.vln_stack_ppo import compute_torch_ppo_objective


class Config(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_tensor(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value.detach()).all().item())


def _make_policy(device: torch.device) -> TopologyWaypointPolicy:
    return TopologyWaypointPolicy(
        vocab_size=2504,
        max_instruction_tokens=200,
        image_size=64,
        hidden_dim=64,
        topology_max_nodes=16,
        topology_model_dim=32,
        max_candidates=5,
        residual_limit_m=0.2,
        initial_log_std=-0.5,
    ).to(device)


def _make_state() -> EpisodeTopologyState:
    return EpisodeTopologyState(
        max_decisions=2,
        topology_max_nodes=16,
        frontier_distance_m=0.75,
        frontier_lateral_m=0.5,
        residual_limit_m=0.2,
    )


def _json_scalar(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        return _json_scalar(value.item())
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-index", type=int, default=4852)
    parser.add_argument("--max-decisions", type=int, default=2)
    parser.add_argument("--max-episode-steps", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.episode_index != 4852:
        raise ValueError("S2 receipt is pinned to R2R train episode index 4852")
    if args.max_decisions != 2 or args.max_episode_steps != 8:
        raise ValueError("S2 receipt requires exactly two decisions and eight primitive steps")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    from rlinf.envs.vln.vln_env import VLNEnv

    checkout = Path(__file__).resolve().parents[2]
    project = Path("/srv/navigation")
    train_dataset = Path("/cache/navigation-example/datasets/r2r/train/train.json.gz")
    if not train_dataset.is_file():
        raise FileNotFoundError(train_dataset)
    cfg = Config(
        seed=17,
        auto_reset=False,
        max_episode_steps=args.max_episode_steps,
        python_bin=str(project / "envs/habitat-cpu-egl/bin/python"),
        server_script=str(checkout / "scripts/habitat_r2r_service.py"),
        asset_root="/cache/navigation-example",
        dataset_split="train",
        episode_index=args.episode_index,
        rgb_size=64,
        max_instruction_tokens=200,
        success_distance_m=3.0,
        habitat_sim_python=str(project / "repos/habitat-sim-cpu-egl/src_python"),
        habitat_env_lib=str(project / "envs/habitat-cpu-egl/lib"),
        egl_vendor_json="/usr/share/glvnd/egl_vendor.d/50_mesa.json",
        startup_timeout_s=60.0,
        video_cfg=Config(save_video=False),
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    torch.manual_seed(20260930)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(20260930)

    env = VLNEnv(cfg, 1, 0, 1, None)
    task = RLinfVLNTaskAdapter(env)
    policy = _make_policy(device)
    # Keep STOP in the declared distribution but make it effectively impossible
    # in this two-decision closure.  Both behavior and replay use these logits.
    with torch.no_grad():
        policy.candidate_slot_bias.zero_()
        policy.candidate_slot_bias[-1] = -100.0
    state = _make_state()
    adapter = TopologyWaypointPolicyAdapter(
        policy,
        state,
        mode="train",
        root_forward=policy,
    )
    executor = TopologyWaypointActionExecutor(state)
    service_process = env.process
    try:
        frozen_before = {
            name: parameter.detach().cpu().clone()
            for name, parameter in policy.named_parameters()
            if not parameter.requires_grad
        }
        trainable_names = adapter.trainable_parameter_names()
        named = dict(policy.named_parameters())
        trainable_before = {
            name: named[name].detach().cpu().clone() for name in trainable_names
        }

        batch = DecisionCollector().collect_episode(
            task,
            adapter,
            executor,
            max_decisions=args.max_decisions,
            episode=args.episode_index,
            seed=17,
        )
        if len(batch.records) != 2:
            raise RuntimeError("real Habitat S2 closure did not collect two decisions")
        if any(row.sample.action_payload[3] == STOP_CANDIDATE_ID for row in batch.records):
            raise RuntimeError("bounded S2 closure unexpectedly sampled STOP")
        first_context = batch.records[0].sample.replay_context["forward_inputs"]
        second_context = batch.records[1].sample.replay_context["forward_inputs"]
        first_fraction = float(first_context["topology_global_features"][0, 9])
        second_fraction = float(second_context["topology_global_features"][0, 9])
        if first_fraction != 0.0 or second_fraction <= first_fraction:
            raise RuntimeError("second decision did not observe the first execution receipt")

        evaluations = adapter.evaluate(batch)
        gae = compute_duration_aware_gae(batch, gamma=0.97, gae_lambda=0.91)
        objective = compute_torch_ppo_objective(
            batch,
            evaluations,
            gae,
            entropy_coef=0.01,
            pre_update_tolerance=1e-6,
        )
        if not _finite_tensor(objective.total_loss):
            raise RuntimeError("S2 PPO objective is non-finite")

        optimizer = torch.optim.Adam((named[name] for name in trainable_names), lr=1e-3)
        optimizer.zero_grad(set_to_none=True)
        objective.total_loss.backward()
        gradient_sums = {}
        for name in trainable_names:
            gradient = named[name].grad
            if gradient is not None:
                if not _finite_tensor(gradient):
                    raise RuntimeError(f"non-finite S2 gradient: {name}")
                gradient_sums[name] = float(gradient.detach().abs().sum().cpu())
        required_gradient_groups = (
            "candidate_scorer",
            "residual_mean_head",
            "residual_log_std",
            "value_head",
        )
        for group in required_gradient_groups:
            if not any(group in name and value > 0.0 for name, value in gradient_sums.items()):
                raise RuntimeError(f"S2 required gradient group is zero: {group}")
        optimizer.step()
        adapter.mark_updated()

        changed = tuple(
            name
            for name in trainable_names
            if not torch.equal(trainable_before[name], named[name].detach().cpu())
        )
        for group in required_gradient_groups:
            if not any(group in name for name in changed):
                raise RuntimeError(f"S2 required parameter group did not change: {group}")
        for name, expected in frozen_before.items():
            if not torch.equal(expected, named[name].detach().cpu()):
                raise RuntimeError(f"frozen observation encoder changed: {name}")

        checkpoint = args.output_dir / "checkpoint"
        saved = save_checkpoint_atomic(
            TopologyWaypointCheckpointAdapter(adapter), checkpoint
        )
        restored_policy = _make_policy(device)
        restored_state = _make_state()
        restored_adapter = TopologyWaypointPolicyAdapter(
            restored_policy,
            restored_state,
            mode="train",
            root_forward=restored_policy,
        )
        loaded = load_checkpoint_verified(
            TopologyWaypointCheckpointAdapter(restored_adapter), checkpoint
        )
        if saved.files_sha256 != loaded.files_sha256:
            raise RuntimeError("independent S2 checkpoint receipt differs")
        if restored_adapter.policy_version != adapter.policy_version:
            raise RuntimeError("restored S2 policy version differs")
        for name, expected in policy.state_dict().items():
            if not torch.equal(expected.detach().cpu(), restored_policy.state_dict()[name].detach().cpu()):
                raise RuntimeError(f"restored S2 tensor differs: {name}")

        metrics = task.get_metrics()
        route_counts = Counter(
            row.sample.action_payload[3] for row in batch.records
        )
        primitive_counts = Counter(
            int(action)
            for row in batch.records
            for action in row.execution.active_primitives
        )
        result = {
            "schema_version": 1,
            "status": "s2_real_habitat_stack_plugin_closure_passed",
            "claim_boundary": "stack_plugin_closure_not_navigation_quality",
            "dataset_split": "train",
            "dataset_sha256": _sha256(train_dataset),
            "episode_index": args.episode_index,
            "episode_id": batch.identity.episode_id,
            "scene_id": env.identity["scene_id"],
            "decisions": len(batch.records),
            "physical_steps": sum(row.execution.duration for row in batch.records),
            "durations": [row.execution.duration for row in batch.records],
            "candidate_ids": [row.sample.action_payload[3] for row in batch.records],
            "route_counts": dict(sorted(route_counts.items())),
            "primitive_action_counts": {
                str(key): value for key, value in sorted(primitive_counts.items())
            },
            "topology_fraction_before": first_fraction,
            "topology_fraction_second": second_fraction,
            "topology_summary": state.topology.summary(),
            "ratio_max_abs_deviation": objective.diagnostics[
                "ratio_max_abs_deviation"
            ],
            "total_loss": objective.diagnostics["total_loss"],
            "gradient_nonzero_names": sorted(
                name for name, value in gradient_sums.items() if value > 0.0
            ),
            "changed_trainable_names": sorted(changed),
            "frozen_parameter_count": len(frozen_before),
            "frozen_parameters_unchanged": True,
            "policy_version": adapter.policy_version,
            "forward_path": adapter.trainable_manifest()["forward_path"],
            "rlinf_manager_used": False,
            "checkpoint_files_sha256": dict(saved.files_sha256),
            "checkpoint_restore_exact": True,
            "success": metrics.success,
            "spl": metrics.spl,
            "navigation_error_m": metrics.navigation_error_m,
            "path_length_m": metrics.path_length_m,
            "collisions": metrics.collisions,
            "stop_called": metrics.stop_called,
            "timed_out": metrics.timed_out,
            "metrics_extras": dict(metrics.extras),
        }
        result = {key: _json_scalar(value) for key, value in result.items()}
        result_path = args.output_dir / "result.json"
        result_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, sort_keys=True))
    finally:
        task.close()
    if service_process.poll() is None:
        raise RuntimeError("Habitat service remained alive after task close")


if __name__ == "__main__":
    main()
