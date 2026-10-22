"""Bounded train-only engineering gate for executor-aligned StreamVLN G8."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import time

import torch
from omegaconf import OmegaConf

from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager

from run_streamvln_ppo_g4 import make_env
from run_streamvln_topology_g5 import (
    ORIGINAL_STREAM_COMMIT,
    checkpoint_load_audit,
    cleanup_manager,
    gradient_audit,
    make_manager_cfg,
    model_device,
    parameter_changes,
    parameter_snapshot,
    save_head_checkpoint,
    sha256_file,
    stable_seed,
    state_sha256,
    topology_tensor,
)
from run_streamvln_plumbing_g717 import (
    frozen_parameter_sentinels,
    optimizer_parameter_scope,
    small_file_inventory,
)
from streamvln_chunk_ppo import collapse_chunk_rewards, variable_duration_gae
from streamvln_executor_contract import (
    HighLevelCommand,
    PRIVILEGED_POLICY_KEYS,
    assert_history_only_keys,
    feedback_from_dict,
)
from streamvln_g8_action import (
    ACTION_COUNT,
    NAVIGATE_ACTION_COUNT,
    CANDIDATE_ORDER_SHA256,
    action_mode,
    action_waypoint_local_m,
    assert_g8_action_contract,
    compile_g8_action,
)
from streamvln_g8_policy import get_model as get_g8_model
from streamvln_topology import (
    FEATURE_NAMES,
    MODE_NAMES,
    NAVIGATE_MODE,
    OnlineTopology,
    RECOVER_MODE,
    STOP_MODE,
)


RLINF_COMMIT = "0505431899574619da86f551bad70b71e0ea2177"
MANIFEST_SHA256 = "5308c9609e302be81801870cf391c8978a046c09ee91c26cfab1c2932751863e"


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def git_head(path: Path) -> str:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={path}", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(f"cannot resolve git head for {path}: {result.stderr}")
    return result.stdout.strip()


def make_policy_cfg(run_cfg: dict, *, trainable: bool, head_checkpoint=None):
    return OmegaConf.create({
        "model_type": "streamvln_ppo",
        "model_path": "",
        "stream_source": run_cfg["stream_source"],
        "base_model": run_cfg["base_model"],
        "tokenizer": run_cfg["tokenizer"],
        "vision_tower": run_cfg["vision_tower"],
        "expected_source_commit": run_cfg["stream_commit"],
        "head_checkpoint": head_checkpoint,
        "trainable": trainable,
        "context_tokens": run_cfg["context_tokens"],
        "seed": run_cfg["seed"],
        "topology_dim": len(FEATURE_NAMES),
        "head_hidden": run_cfg["head_hidden"],
        "alignment_coefficient": 4.0,
        "precision": "bf16",
        "add_value_head": True,
        "is_lora": False,
        "load_to_device": False,
    })


class G8FSDPManager(FSDPModelManager):
    def model_provider_func(self):
        return get_g8_model(self._cfg.model)


def setup_manager(run_cfg: dict):
    from run_streamvln_ppo_g3 import init_distributed

    torch.manual_seed(run_cfg["seed"])
    torch.cuda.manual_seed_all(run_cfg["seed"])
    init_distributed(run_cfg["master_port"])
    cfg = make_manager_cfg(run_cfg, total_training_steps=1)
    cfg.model = make_policy_cfg(run_cfg, trainable=True)
    manager = G8FSDPManager(cfg, world_size=1, rank=0)
    manager.setup_model_and_optimizer()
    if type(manager.model).__name__ != "FullyShardedDataParallel":
        raise RuntimeError("G8 actor is not wrapped by RLinf FSDP")
    core = manager.model.module
    manifest = core.trainable_manifest()
    if (
        manifest["source_kind"] != "upstream-original"
        or manifest["memory_adapter"]
        or manifest["secondary_post_training_used"]
        or manifest["backbone_trainable"]
        or manifest["gaussian_waypoint_likelihood"]
        or manifest["waypoint_log_std_trainable"]
        or manifest["action_count"] != ACTION_COUNT
        or not 500_000 <= manifest["trainable_parameter_count"] <= 2_000_000
    ):
        raise RuntimeError("G8 source or trainable parameter scope is invalid")
    return manager, core, manifest


def validate_manifest(manifest: dict) -> None:
    if manifest["schema_version"] != 1 or manifest["stage"] != "G8 bounded executor-aligned engineering gate":
        raise ValueError("G8 engineering manifest identity changed")
    if manifest["source_contract"] != {
        "stream_commit": ORIGINAL_STREAM_COMMIT,
        "rlinf_commit": RLINF_COMMIT,
        "original_stream_only": True,
        "ttt_or_secondary_post_training": False,
    }:
        raise ValueError("G8 source contract changed")
    assert_history_only_keys(manifest["policy_observation_keys"])
    specs = manifest["episodes"]
    if (
        [(row["target_mode"], row["target_mode_id"]) for row in specs]
        != [("navigate", 0), ("stop", 1), ("recover", 2)]
        or manifest["expected_executed_lengths"] != [4, 1, 4]
        or manifest["expected_end_boundaries"] != ["truncated", "terminated", "truncated"]
        or manifest["max_episode_steps"] != 4
        or manifest["optimizer_update_limit"] != 1
        or manifest["model_load_limit"] != 1
        or manifest["val_unseen_episode_count"] != 0
        or manifest["claim"] != "engineering_correctness_only"
    ):
        raise ValueError("G8 engineering protocol changed")


def validate_identity(identity: dict, spec: dict) -> None:
    if (
        str(identity.get("episode_id")) != str(spec["episode_id"])
        or Path(identity.get("scene_id", "")).stem != spec["scene"]
    ):
        raise RuntimeError("G8 Habitat episode identity changed")


def sample_target_mode(manager, state, top, run_cfg, spec):
    maximum = run_cfg["manifest"]["sampling"]["maximum_attempts_per_episode"]
    for attempt in range(maximum):
        seed = stable_seed(
            run_cfg["seed"], "g8_engineering_mode", spec["episode_id"],
            spec["target_mode"], attempt,
        )
        sampled = manager.model(
            operation="sample_g8",
            state_features=state,
            topology_features=top,
            seed=seed,
            deterministic=False,
        )
        action_id = int(sampled["action_ids"].reshape(-1)[0])
        if action_mode(action_id) == spec["target_mode_id"]:
            return sampled, action_id, {
                "seed": seed,
                "attempt_index": attempt,
                "attempt_count": attempt + 1,
            }
    raise RuntimeError(f"G8 could not sample target mode {spec['target_mode']}")


def execute_action(env, topology, action_id: int) -> dict:
    mode = action_mode(action_id)
    waypoint = action_waypoint_local_m(action_id, topology)
    command = HighLevelCommand(
        decision_index=0,
        mode=MODE_NAMES[mode],
        waypoint_forward_m=float(waypoint[0]),
        waypoint_lateral_m=float(waypoint[1]),
    )
    proposed = compile_g8_action(action_id, topology)
    obs_list, rewards, terminations, truncations, infos = env.chunk_step(
        torch.tensor(proposed, dtype=torch.int64).reshape(1, 4, 1)
    )
    feedback = feedback_from_dict(infos[0].get("executor_feedback"))
    if list(feedback.proposed_actions) != proposed:
        raise RuntimeError("G8 executor proposal differs from action-ID compiler")
    topology.update_from_feedback(feedback.to_dict())
    return {
        "obs": obs_list[0],
        "command": command.to_dict(),
        "proposed_actions": proposed,
        "feedback": feedback,
        "primitive_rewards": [float(value) for value in rewards[0].detach().cpu().tolist()],
        "terminated": bool(terminations.any()),
        "truncated": bool(truncations.any()),
    }


def collect_episode(manager, core, run_cfg: dict, spec: dict, slot: int) -> dict:
    device = model_device(core)
    env = make_env(
        run_cfg,
        split="train",
        episode_index=spec["index"],
        max_steps=run_cfg["manifest"]["max_episode_steps"],
    )
    topology = OnlineTopology(max_decisions=1)
    try:
        obs, identity = env.reset()
        validate_identity(identity, spec)
        if set(obs) != set(run_cfg["manifest"]["policy_observation_keys"]):
            raise RuntimeError("G8 policy observation schema changed")
        assert_history_only_keys(list(obs))
        with torch.no_grad():
            state = manager.model(operation="encode_state", obs=obs)
            top = topology_tensor(topology, device)
            sampled, action_id, search = sample_target_mode(
                manager, state, top, run_cfg, spec
            )
            immediate_replay = manager.model(
                operation="replay_g8",
                state_features=state,
                topology_features=top,
                action_ids=sampled["action_ids"],
            )
        sample_replay_error = float((
            immediate_replay["log_probs"].float()
            - sampled["behavior_log_probs"].float()
        ).abs().max())
        step = execute_action(env, topology, action_id)
        feedback = step["feedback"]
        duration = len(feedback.executed_actions)
        reward_row = step["primitive_rewards"]
        chunk_rewards, durations = collapse_chunk_rewards(
            [reward_row], [list(feedback.executed_mask)], run_cfg["manifest"]["gamma"]
        )
        reward_error = max(
            abs(chunk_rewards[0] - feedback.discounted_environment_reward(run_cfg["manifest"]["gamma"])),
            abs(feedback.environment_reward_sum - sum(reward_row[:duration])),
        )
        if durations != [duration]:
            raise RuntimeError("G8 reward duration differs from feedback")

        if step["terminated"]:
            bootstrap = 0.0
            bootstrap_source = "natural_termination_zero"
        else:
            if not bool(step["obs"]["terminal_frozen"].item()):
                raise RuntimeError("G8 truncated final observation is not frozen")
            with torch.no_grad():
                bootstrap_state = manager.model(
                    operation="encode_bootstrap_state", obs=step["obs"]
                )
                bootstrap_top = topology_tensor(topology, device)
                bootstrap_output = manager.model(
                    operation="sample_g8",
                    state_features=bootstrap_state,
                    topology_features=bootstrap_top,
                    seed=stable_seed(run_cfg["seed"], "g8_bootstrap", spec["episode_id"]),
                    deterministic=True,
                )
                bootstrap = float(bootstrap_output["values"].reshape(-1)[0])
            bootstrap_source = "terminal_frozen_observation_value"

        value = float(sampled["values"].reshape(-1)[0])
        advantages, returns, deltas = variable_duration_gae(
            chunk_rewards,
            [value],
            [bootstrap],
            [duration],
            [step["terminated"]],
            run_cfg["manifest"]["gamma"],
            run_cfg["manifest"]["gae_lambda"],
        )
        return {
            "slot": slot,
            "episode_index": spec["index"],
            "episode_id": str(spec["episode_id"]),
            "scene": spec["scene"],
            "target_mode": spec["target_mode"],
            "state": state.detach().cpu().float(),
            "topology_features": top.detach().cpu().float(),
            "action_id": action_id,
            "mode": action_mode(action_id),
            "behavior_log_prob": float(sampled["behavior_log_probs"].reshape(-1)[0]),
            "value": value,
            "advantage": float(advantages[0]),
            "return": float(returns[0]),
            "delta": float(deltas[0]),
            "duration": duration,
            "terminated": step["terminated"],
            "truncated": step["truncated"],
            "bootstrap": bootstrap,
            "bootstrap_source": bootstrap_source,
            "sample_replay_error": sample_replay_error,
            "seed_search": search,
            "command": step["command"],
            "proposed_actions": step["proposed_actions"],
            "executor_feedback": feedback.to_dict(),
            "reward_accounting_error": reward_error,
            "topology": topology.summary(),
        }
    finally:
        env.close()


def run_update(manager, core, run_cfg: dict):
    rows = [
        collect_episode(manager, core, run_cfg, spec, slot)
        for slot, spec in enumerate(run_cfg["manifest"]["episodes"])
    ]
    device = model_device(core)
    states = torch.stack([row["state"] for row in rows]).to(device)
    tops = torch.stack([row["topology_features"] for row in rows]).to(device)
    action_ids = torch.tensor([row["action_id"] for row in rows], dtype=torch.int64, device=device)
    target_waypoints = torch.tensor([
        [row["command"]["waypoint_forward_m"], row["command"]["waypoint_lateral_m"]]
        for row in rows
    ], dtype=torch.float32, device=device)
    behavior = torch.tensor(
        [row["behavior_log_prob"] for row in rows], dtype=torch.float32, device=device
    )
    advantages = torch.tensor(
        [row["advantage"] for row in rows], dtype=torch.float32, device=device
    )
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
    returns = torch.tensor([row["return"] for row in rows], dtype=torch.float32, device=device)

    before = parameter_snapshot(core)
    frozen_before, sentinel_names = frozen_parameter_sentinels(core)
    input_bundle = {
        "states": states.detach().cpu(),
        "topology": tops.detach().cpu(),
        "action_ids": action_ids.detach().cpu(),
        "target_waypoints": target_waypoints.detach().cpu(),
    }
    input_sha_before = state_sha256(input_bundle)
    with torch.no_grad():
        preupdate = manager.model(
            operation="replay_g8",
            state_features=states,
            topology_features=tops,
            action_ids=action_ids,
        )
        ratios = torch.exp(preupdate["log_probs"].float() - behavior)
        max_ratio_deviation = float((ratios - 1.0).abs().max())

    replay = manager.model(
        operation="replay_g8",
        state_features=states,
        topology_features=tops,
        action_ids=action_ids,
    )
    ratios = torch.exp(replay["log_probs"].float() - behavior)
    clipped = ratios.clamp(
        1.0 - run_cfg["manifest"]["clip_ratio"],
        1.0 + run_cfg["manifest"]["clip_ratio"],
    )
    actor_loss = -torch.minimum(ratios * advantages, clipped * advantages).mean()
    value_loss = 0.5 * (replay["values"].float() - returns).square().mean()
    entropy = replay["entropy"].float().mean()
    bc = manager.model(
        operation="bc_loss_g8",
        state_features=states,
        topology_features=tops,
        target_action_ids=action_ids,
        target_waypoints=target_waypoints,
    )
    loss = (
        actor_loss
        + run_cfg["manifest"]["value_coef"] * value_loss
        - run_cfg["manifest"]["entropy_coef"] * entropy
        + run_cfg["manifest"]["bc_auxiliary_coefficient"] * bc["loss"]
    )
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("G8 engineering loss is nonfinite")
    scope = optimizer_parameter_scope(manager, core)
    manager.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    audit = gradient_audit(core)
    if (
        not audit["finite"]
        or audit["group_norm_sums"]["high_level_policy"] <= 0
        or audit["group_norm_sums"]["value_head"] <= 0
        or audit["frozen_with_grad"]
    ):
        raise RuntimeError("G8 gradient isolation failed")
    actor_version_before = core.actor_version
    grad_norm, learning_rates = manager.optimizer_step()
    core.mark_updated()
    changes = parameter_changes(core, before)
    frozen_after = {
        name: parameter.detach().cpu().clone()
        for raw_name, parameter in core.named_parameters()
        for name in [raw_name.replace("_fsdp_wrapped_module.", "")]
        if name in frozen_before
    }
    frozen_unchanged = state_sha256(frozen_before) == state_sha256(frozen_after)
    input_sha_after = state_sha256(input_bundle)
    update = {
        "loss": float(loss.detach()),
        "actor_loss": float(actor_loss.detach()),
        "value_loss": float(value_loss.detach()),
        "entropy": float(entropy.detach()),
        "bc_loss": float(bc["loss"].detach()),
        "preupdate_ratio_max_deviation": max_ratio_deviation,
        "gradient_audit": audit,
        "optimizer_scope": scope,
        "grad_norm": float(torch.as_tensor(grad_norm)),
        "learning_rates": [float(value) for value in learning_rates],
        "actor_version_before": actor_version_before,
        "actor_version_after": core.actor_version,
        "parameter_changes": changes,
        "frozen_sentinel_names": sentinel_names,
        "frozen_sentinels_unchanged": frozen_unchanged,
        "rollout_inputs_unchanged": input_sha_before == input_sha_after,
        "batch_shapes": {
            "log_probs": list(replay["log_probs"].shape),
            "values": list(replay["values"].shape),
            "action_logits": list(replay["action_logits"].shape),
            "waypoint_mean": list(replay["waypoint_mean"].shape),
        },
    }
    return rows, update


def serialized_row(row: dict) -> dict:
    return {
        key: value for key, value in row.items()
        if key not in ("state", "topology_features")
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rlinf-root", type=Path, required=True)
    parser.add_argument("--stream-source", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--vision-tower", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--habitat-python", type=Path, required=True)
    parser.add_argument("--habitat-server", type=Path, required=True)
    parser.add_argument("--habitat-sim-python", type=Path, required=True)
    parser.add_argument("--habitat-env-lib", type=Path, required=True)
    parser.add_argument("--egl-vendor-json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026092808)
    parser.add_argument("--master-port", type=int, default=29608)
    parser.add_argument("--context-tokens", type=int, default=4096)
    parser.add_argument("--head-hidden", type=int, default=256)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    started = time.time()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = args.manifest.expanduser().resolve(strict=True)
    if sha256_file(manifest_path) != MANIFEST_SHA256:
        raise RuntimeError("G8 engineering manifest hash changed")
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest)
    action_contract = assert_g8_action_contract()
    if action_contract["candidate_order_sha256"] != CANDIDATE_ORDER_SHA256:
        raise RuntimeError("G8 action contract hash changed")

    path_names = (
        "rlinf_root", "stream_source", "base_model", "tokenizer", "vision_tower",
        "asset_root", "habitat_python", "habitat_server", "habitat_sim_python",
        "habitat_env_lib", "egl_vendor_json",
    )
    paths = {
        name: str(Path(getattr(args, name)).expanduser().resolve(strict=True))
        for name in path_names
    }
    source_heads = {
        "stream": git_head(Path(paths["stream_source"])),
        "rlinf": git_head(Path(paths["rlinf_root"])),
    }
    if source_heads != {"stream": ORIGINAL_STREAM_COMMIT, "rlinf": RLINF_COMMIT}:
        raise RuntimeError("G8 source commit changed")
    train_path = Path(paths["asset_root"]) / "datasets" / "r2r" / "train" / "train.json.gz"
    train_sha = sha256_file(train_path.resolve(strict=True))
    if train_sha != manifest["train_sha256"]:
        raise RuntimeError("G8 train dataset hash changed")

    run_cfg = {
        **paths,
        "output_dir": str(output),
        "stream_commit": ORIGINAL_STREAM_COMMIT,
        "rlinf_commit": RLINF_COMMIT,
        "manifest": manifest,
        "context_tokens": args.context_tokens,
        "head_hidden": args.head_hidden,
        "seed": args.seed,
        "master_port": args.master_port,
        "learning_rate": manifest["learning_rate"],
        "value_learning_rate": manifest["value_learning_rate"],
    }
    script_root = Path(__file__).resolve().parent
    implementation_names = (
        "run_streamvln_g8_engineering.py",
        "streamvln_g8_action.py",
        "streamvln_g8_policy.py",
        "streamvln_topological_policy.py",
        "streamvln_candidates_g73.py",
        "streamvln_executor_contract.py",
        "streamvln_topology.py",
    )
    config = {
        "kind": "streamvln-g8-executor-aligned-engineering-gate",
        "manifest": str(manifest_path),
        "manifest_sha256": MANIFEST_SHA256,
        "source_heads": source_heads,
        "train_dataset_sha256": train_sha,
        "action_contract": action_contract,
        "implementation_sha256": {
            name: sha256_file((script_root / name).resolve(strict=True))
            for name in implementation_names
        },
        "model_input_inventory": {
            "base_model": small_file_inventory(Path(paths["base_model"])),
            "tokenizer": small_file_inventory(Path(paths["tokenizer"])),
            "vision_tower": small_file_inventory(Path(paths["vision_tower"])),
        },
        "run": run_cfg,
        "model_load_count": 1,
        "optimizer_update_count": 1,
        "val_unseen_episode_count": 0,
    }
    config_path = output / "run_config.json"
    write_json(config_path, config)

    manager = None
    try:
        manager, core, trainable_manifest = setup_manager(run_cfg)
        rows, update = run_update(manager, core, run_cfg)
        rows_path = output / "rollout_records.jsonl"
        rows_path.write_text("".join(
            json.dumps(serialized_row(row), sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ))
        checkpoint = save_head_checkpoint(
            core,
            output / "checkpoint_engineering",
            {
                "schema": "streamvln_g8_executor_aligned_v1",
                "variant": "engineering_one_update",
                "stream_commit": ORIGINAL_STREAM_COMMIT,
                "rlinf_commit": RLINF_COMMIT,
                "val_unseen_used": False,
                "memory_adapter": False,
                "secondary_post_training_used": False,
            },
        )
        load_audit = checkpoint_load_audit(core, Path(checkpoint["policy"]))
        modes = [row["mode"] for row in rows]
        durations = [row["duration"] for row in rows]
        boundaries = ["terminated" if row["terminated"] else "truncated" for row in rows]
        guards = {
            "source_pins": source_heads == {"stream": ORIGINAL_STREAM_COMMIT, "rlinf": RLINF_COMMIT},
            "manifest_hash": sha256_file(manifest_path) == MANIFEST_SHA256,
            "train_dataset_hash": train_sha == manifest["train_sha256"],
            "action_contract": action_contract == {
                "action_count": 28,
                "navigate_candidate_count": 26,
                "candidate_order_sha256": CANDIDATE_ORDER_SHA256,
                "unique_navigate_macros": 26,
            },
            "original_stream_frozen_scope": (
                trainable_manifest["source_kind"] == "upstream-original"
                and trainable_manifest["memory_adapter"] is False
                and trainable_manifest["secondary_post_training_used"] is False
                and trainable_manifest["backbone_trainable"] is False
                and trainable_manifest["gaussian_waypoint_likelihood"] is False
                and trainable_manifest["waypoint_log_std_trainable"] is False
            ),
            "optimizer_scope": (
                update["optimizer_scope"]["optimizer_equals_trainable"]
                and update["optimizer_scope"]["optimizer_frozen_overlap_count"] == 0
            ),
            "sample_replay_match": max(row["sample_replay_error"] for row in rows) <= 1e-6,
            "mode_coverage": modes == [NAVIGATE_MODE, STOP_MODE, RECOVER_MODE],
            "executed_prefixes": durations == manifest["expected_executed_lengths"],
            "end_boundaries": boundaries == manifest["expected_end_boundaries"],
            "causal_feedback": all(
                row["executor_feedback"]["proposed_actions"] == row["proposed_actions"]
                for row in rows
            ),
            "history_only_observation": not (
                set(manifest["policy_observation_keys"]) & PRIVILEGED_POLICY_KEYS
            ),
            "reward_accounting": max(row["reward_accounting_error"] for row in rows) <= 1e-6,
            "bootstrap_contract": (
                rows[1]["bootstrap"] == 0.0
                and rows[1]["bootstrap_source"] == "natural_termination_zero"
                and all(
                    math.isfinite(rows[index]["bootstrap"])
                    and rows[index]["bootstrap_source"] == "terminal_frozen_observation_value"
                    for index in (0, 2)
                )
            ),
            "preupdate_ratio": update["preupdate_ratio_max_deviation"] <= manifest["preupdate_ratio_max_deviation"],
            "batch_shapes": update["batch_shapes"] == {
                "log_probs": [3],
                "values": [3],
                "action_logits": [3, ACTION_COUNT],
                "waypoint_mean": [3, 2],
            },
            "finite_scoped_gradients": (
                all(math.isfinite(update[key]) for key in ("loss", "actor_loss", "value_loss", "entropy", "bc_loss", "grad_norm"))
                and update["gradient_audit"]["finite"]
                and update["gradient_audit"]["group_norm_sums"]["high_level_policy"] > 0
                and update["gradient_audit"]["group_norm_sums"]["value_head"] > 0
                and not update["gradient_audit"]["frozen_with_grad"]
            ),
            "exactly_one_update": update["actor_version_after"] - update["actor_version_before"] == 1,
            "parameters_changed": update["parameter_changes"]["changed_tensor_count"] > 0,
            "frozen_sentinels_unchanged": update["frozen_sentinels_unchanged"],
            "rollout_inputs_unchanged": update["rollout_inputs_unchanged"],
            "checkpoint_roundtrip": load_audit["passed"] and checkpoint["max_persist_error"] == 0.0,
            "zero_val_unseen": config["val_unseen_episode_count"] == 0,
        }
        all_passed = all(guards.values())
        result = {
            "kind": config["kind"],
            "all_engineering_guards_passed": all_passed,
            "algorithm_claim": False,
            "navigation_evaluation": False,
            "trainable_manifest": trainable_manifest,
            "counts": {
                "model_loads": 1,
                "train_episodes": len(rows),
                "optimizer_updates": 1,
                "val_unseen_episodes": 0,
            },
            "action_ids": [row["action_id"] for row in rows],
            "modes": modes,
            "durations": durations,
            "boundaries": boundaries,
            "update": update,
            "checkpoint": checkpoint,
            "checkpoint_load_audit": load_audit,
            "guards": guards,
            "elapsed_seconds": time.time() - started,
            "evidence": {
                "manifest_sha256": MANIFEST_SHA256,
                "run_config_sha256": sha256_file(config_path),
                "rollout_records_sha256": sha256_file(rows_path),
            },
        }
        result_path = output / "result.json"
        write_json(result_path, result)
        print(json.dumps({
            "all_engineering_guards_passed": all_passed,
            "guards": guards,
            "counts": result["counts"],
            "result": str(result_path),
        }, indent=2, sort_keys=True))
        if not all_passed:
            raise SystemExit(2)
    finally:
        if manager is not None:
            cleanup_manager(manager)


if __name__ == "__main__":
    main()
