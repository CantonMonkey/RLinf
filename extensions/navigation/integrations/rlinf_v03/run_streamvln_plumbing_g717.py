"""Run the preregistered G7.17 bounded high-level RL plumbing smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import torch

from run_streamvln_ppo_g4 import make_env
from run_streamvln_recovery_g6 import replay_individual, setup_manager
from run_streamvln_topology_g5 import (
    cleanup_manager,
    gradient_audit,
    jsonable,
    model_device,
    parameter_changes,
    parameter_snapshot,
    stable_seed,
    state_sha256,
    topology_tensor,
)
from streamvln_chunk_ppo import collapse_chunk_rewards, variable_duration_gae
from streamvln_executor_contract import (
    HighLevelCommand,
    PRIVILEGED_POLICY_KEYS,
    assert_history_only_keys,
    feedback_from_dict,
)
from streamvln_recovery import compile_recovery_aware, recovery_target
from streamvln_topological_policy import canonical_parameter_name
from streamvln_topology import (
    MODE_NAMES,
    NAVIGATE_MODE,
    OnlineTopology,
    RECOVER_MODE,
    STOP_MODE,
)


ORIGINAL_STREAM_COMMIT = "e48f6ff7e9201d93aae003e8f64b04c00cec13bc"
RLINF_COMMIT = "0505431899574619da86f551bad70b71e0ea2177"
G716_RESULT_SHA256 = "e8541234dae71b751b1aa7fd84de89eb3c4f95ab9e63e6762a86fecb869195f6"
MANIFEST_SHA256 = "f8937a8845dad1488b714c4862210b1e4b0eb460dbc87726f5c6bf9f2f1f0838"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def git_head(path: Path) -> str:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={path}", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=20,
    )
    if result.returncode:
        raise RuntimeError(f"cannot resolve git head for {path}: {result.stderr}")
    return result.stdout.strip()


def validate_manifest(manifest: dict) -> dict:
    expected = {
        "schema_version", "stage", "source_contract", "source", "predecessor",
        "policy_observation_keys", "privileged_policy_blacklist", "plumbing_smoke",
        "gates", "model_load", "rl_update", "navigation_evaluation",
        "controller_comparison", "checkpoint_selection", "g8_ready",
    }
    if set(manifest) != expected or manifest["schema_version"] != 1:
        raise ValueError("G7.17 manifest schema changed")
    source = manifest["source_contract"]
    if source != {
        "stream_commit": ORIGINAL_STREAM_COMMIT,
        "rlinf_commit": RLINF_COMMIT,
        "original_stream_only": True,
        "ttt_or_secondary_post_training": False,
    }:
        raise ValueError("G7.17 source contract changed")
    if manifest["predecessor"]["accepted_result_sha256"] != G716_RESULT_SHA256:
        raise ValueError("G7.17 predecessor anchor changed")
    if set(manifest["privileged_policy_blacklist"]) != PRIVILEGED_POLICY_KEYS:
        raise ValueError("G7.17 privileged-data blacklist changed")
    assert_history_only_keys(manifest["policy_observation_keys"])
    smoke = manifest["plumbing_smoke"]
    specs = smoke["scene_specs"]
    if (
        smoke["split"] != "train"
        or [(row["target_mode"], row["target_mode_id"]) for row in specs]
        != [("navigate", 0), ("stop", 1), ("recover", 2)]
        or smoke["expected_executed_lengths"] != [4, 1, 4]
        or smoke["expected_end_boundaries"] != ["truncated", "terminated", "truncated"]
        or smoke["max_episode_steps"] != 4
        or smoke["optimizer_update_limit"] != 1
        or smoke["val_unseen_episode_count"] != 0
        or smoke["mode_coverage_sampling"]["maximum_attempts_per_episode"] != 4096
        or smoke["mode_coverage_sampling"]["coverage_conditioned_not_unbiased_learning_data"]
        is not True
    ):
        raise ValueError("G7.17 plumbing protocol changed")
    if (
        manifest["model_load"] is not True
        or manifest["rl_update"] is not True
        or manifest["navigation_evaluation"] is not False
        or manifest["controller_comparison"] is not False
        or manifest["checkpoint_selection"] is not False
        or manifest["g8_ready"] is not False
    ):
        raise ValueError("G7.17 claim boundary changed")
    return smoke


def dataset_hashes(asset_root: Path, manifest: dict) -> dict:
    rows = {}
    for split in ("train", "val_unseen"):
        path = asset_root / "datasets" / "r2r" / split / f"{split}.json.gz"
        digest = sha256_file(path.resolve(strict=True))
        expected = manifest["source"][f"{split}_sha256"]
        if digest != expected:
            raise RuntimeError(f"G7.17 {split} dataset hash changed")
        rows[split] = {"path": str(path), "sha256": digest}
    return rows


def small_file_inventory(root: Path) -> dict:
    """Bind lightweight model configuration inputs without rereading 8B weights."""
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        size = path.stat().st_size
        if size <= 2 * 1024 * 1024:
            rows.append({
                "relative_path": path.relative_to(root).as_posix(),
                "size": size,
                "sha256": sha256_file(path),
            })
    digest = hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {"root": str(root), "small_file_count": len(rows), "sha256": digest, "files": rows}


def frozen_parameter_sentinels(core) -> tuple[dict[str, torch.Tensor], list[str]]:
    candidates = []
    for raw_name, parameter in core.named_parameters():
        if not parameter.requires_grad and 0 < parameter.numel() <= 65536:
            candidates.append((canonical_parameter_name(raw_name), parameter))
    candidates.sort(key=lambda item: item[0])
    if len(candidates) < 8:
        raise RuntimeError("G7.17 found too few frozen parameter sentinels")
    indices = sorted({
        round(index * (len(candidates) - 1) / 11)
        for index in range(12)
    })
    selected = {name: parameter.detach().cpu().clone() for index in indices
                for name, parameter in [candidates[index]]}
    if len(selected) < 8 or not any("backbone" in name for name in selected):
        raise RuntimeError("G7.17 frozen sentinels do not cover the backbone")
    return selected, sorted(selected)


def validate_identity(identity: dict, spec: dict) -> None:
    if (
        str(identity.get("episode_id")) != str(spec["episode_id"])
        or Path(identity.get("scene_id", "")).stem != spec["scene"]
    ):
        raise RuntimeError("G7.17 Habitat episode identity changed")


def sample_target_mode(manager, state, topology_features, run_cfg, spec):
    maximum = run_cfg["smoke"]["mode_coverage_sampling"]["maximum_attempts_per_episode"]
    for attempt in range(maximum):
        seed = stable_seed(
            run_cfg["seed"], "g717_mode_coverage", spec["episode_id"],
            spec["target_mode"], attempt,
        )
        sampled = manager.model(
            operation="sample_head",
            state_features=state,
            topology_features=topology_features,
            seed=seed,
            deterministic=False,
        )
        if int(sampled["modes"].reshape(-1)[0]) == spec["target_mode_id"]:
            return sampled, {"attempt_index": attempt, "attempt_count": attempt + 1, "seed": seed}
    raise RuntimeError(f"G7.17 could not sample target mode {spec['target_mode']}")


def execute_choice(env, topology, mode: int, sampled_waypoint: list[float]) -> dict:
    if mode == STOP_MODE:
        command_waypoint = [0.0, 0.0]
    elif mode == RECOVER_MODE:
        command_waypoint = list(recovery_target(topology))
    else:
        command_waypoint = [float(value) for value in sampled_waypoint]
    command = HighLevelCommand(
        decision_index=0,
        mode=MODE_NAMES[mode],
        waypoint_forward_m=command_waypoint[0],
        waypoint_lateral_m=command_waypoint[1],
    )
    proposed = compile_recovery_aware(command_waypoint, mode, topology)
    obs_list, rewards, terminations, truncations, infos = env.chunk_step(
        torch.tensor(proposed, dtype=torch.int64).reshape(1, 4, 1)
    )
    feedback = feedback_from_dict(infos[0].get("executor_feedback"))
    if list(feedback.proposed_actions) != proposed:
        raise RuntimeError("G7.17 executor proposal differs from compiler output")
    topology.update_from_feedback(feedback.to_dict())
    return {
        "obs": obs_list[0],
        "command": command.to_dict(),
        "feedback": feedback,
        "primitive_rewards": [float(value) for value in rewards[0].detach().cpu().tolist()],
        "terminated": bool(terminations.any()),
        "truncated": bool(truncations.any()),
        "termination_row": [bool(value) for value in terminations[0].detach().cpu().tolist()],
        "truncation_row": [bool(value) for value in truncations[0].detach().cpu().tolist()],
    }


def collect_episode(manager, core, run_cfg: dict, spec: dict, slot: int) -> dict:
    device = model_device(core)
    env = make_env(
        run_cfg,
        split="train",
        episode_index=spec["index"],
        max_steps=run_cfg["smoke"]["max_episode_steps"],
    )
    topology = OnlineTopology(max_decisions=1)
    try:
        obs, identity = env.reset()
        validate_identity(identity, spec)
        if set(obs) != set(run_cfg["policy_observation_keys"]):
            raise RuntimeError("G7.17 policy observation schema changed")
        assert_history_only_keys(list(obs))
        with torch.no_grad():
            state = manager.model(operation="encode_state", obs=obs)
            top = topology_tensor(topology, device)
            sampled, search = sample_target_mode(manager, state, top, run_cfg, spec)
        mode = int(sampled["modes"].reshape(-1)[0])
        waypoint = sampled["waypoints"].reshape(-1, 2)[0].detach().cpu().tolist()
        step = execute_choice(env, topology, mode, waypoint)
        feedback = step["feedback"]
        duration = len(feedback.executed_actions)
        expected_duration = run_cfg["smoke"]["expected_executed_lengths"][slot]
        if duration != expected_duration:
            raise RuntimeError("G7.17 executed prefix length changed")
        if step["terminated"] == step["truncated"]:
            raise RuntimeError("G7.17 episode must end by exactly one boundary")
        boundary = "terminated" if step["terminated"] else "truncated"
        if boundary != run_cfg["smoke"]["expected_end_boundaries"][slot]:
            raise RuntimeError("G7.17 episode boundary changed")
        reward_row = step["primitive_rewards"]
        if any(abs(value) > 1e-8 for value in reward_row[duration:]):
            raise RuntimeError("G7.17 padded reward is nonzero")
        chunk_rewards, durations = collapse_chunk_rewards(
            [reward_row], [list(feedback.executed_mask)], run_cfg["gamma"]
        )
        reward_error = abs(
            chunk_rewards[0] - feedback.discounted_environment_reward(run_cfg["gamma"])
        )
        reward_sum_error = abs(
            feedback.environment_reward_sum - sum(reward_row[:duration])
        )
        if durations != [duration] or max(reward_error, reward_sum_error) > 1e-6:
            raise RuntimeError("G7.17 reward collapse differs from executor feedback")

        if step["terminated"]:
            bootstrap = 0.0
            bootstrap_source = "natural_termination_zero"
        else:
            if not bool(step["obs"]["terminal_frozen"].item()):
                raise RuntimeError("G7.17 truncated final observation is not frozen")
            with torch.no_grad():
                bootstrap_state = manager.model(
                    operation="encode_bootstrap_state", obs=step["obs"]
                )
                bootstrap_topology = topology_tensor(topology, device)
                bootstrap_dist = manager.model(
                    operation="head_distribution",
                    state_features=bootstrap_state,
                    topology_features=bootstrap_topology,
                )
                bootstrap = float(bootstrap_dist["values"].reshape(-1)[0])
            bootstrap_source = "terminal_frozen_observation_value"
        if not math.isfinite(bootstrap):
            raise RuntimeError("G7.17 bootstrap value is nonfinite")

        value = float(sampled["values"].reshape(-1)[0])
        advantages, returns, deltas = variable_duration_gae(
            chunk_rewards,
            [value],
            [bootstrap],
            durations,
            [step["terminated"]],
            run_cfg["gamma"],
            run_cfg["gae_lambda"],
        )
        return {
            "episode_slot": slot,
            "episode_index": spec["index"],
            "episode_id": str(spec["episode_id"]),
            "scene": spec["scene"],
            "target_mode": spec["target_mode"],
            "policy_mode": mode,
            "state": state.detach().cpu().float(),
            "topology_features": top.detach().cpu().float(),
            "waypoint": [float(value) for value in waypoint],
            "behavior_log_prob": float(sampled["behavior_log_probs"].reshape(-1)[0]),
            "value": value,
            "advantage": float(advantages[0]),
            "return": float(returns[0]),
            "delta": float(deltas[0]),
            "chunk_reward": float(chunk_rewards[0]),
            "duration": duration,
            "terminated": step["terminated"],
            "truncated": step["truncated"],
            "bootstrap": bootstrap,
            "bootstrap_source": bootstrap_source,
            "seed_search": search,
            "command": step["command"],
            "executor_feedback": feedback.to_dict(),
            "primitive_rewards": reward_row,
            "termination_row": step["termination_row"],
            "truncation_row": step["truncation_row"],
            "reward_accounting_error": max(reward_error, reward_sum_error),
            "topology": topology.summary(),
        }
    finally:
        env.close()


def optimizer_parameter_scope(manager, core) -> dict:
    trainable = {id(parameter) for parameter in core.parameters() if parameter.requires_grad}
    frozen = {id(parameter) for parameter in core.parameters() if not parameter.requires_grad}
    optimized = {
        id(parameter)
        for group in manager.optimizer.param_groups
        for parameter in group["params"]
    }
    return {
        "trainable_parameter_objects": len(trainable),
        "optimizer_parameter_objects": len(optimized),
        "optimizer_equals_trainable": optimized == trainable,
        "optimizer_frozen_overlap_count": len(optimized & frozen),
    }


def serialized_row(row: dict) -> dict:
    return {
        key: jsonable(value)
        for key, value in row.items()
        if key not in ("state", "topology_features")
    }


def run_update(manager, core, run_cfg: dict) -> tuple[dict, list[dict]]:
    actor_version_before = core.actor_version
    before_parameters = parameter_snapshot(core)
    frozen_before, sentinel_names = frozen_parameter_sentinels(core)
    optimizer_scope = optimizer_parameter_scope(manager, core)
    if not optimizer_scope["optimizer_equals_trainable"] or optimizer_scope["optimizer_frozen_overlap_count"]:
        raise RuntimeError("G7.17 optimizer scope includes frozen or misses trainable parameters")

    rows = [
        collect_episode(manager, core, run_cfg, spec, slot)
        for slot, spec in enumerate(run_cfg["smoke"]["scene_specs"])
    ]
    if core.actor_version != actor_version_before:
        raise RuntimeError("G7.17 policy changed during collection")
    device = model_device(core)
    indices = list(range(len(rows)))
    replay_inputs = {
        **{f"state_{index}": row["state"] for index, row in enumerate(rows)},
        **{f"topology_{index}": row["topology_features"] for index, row in enumerate(rows)},
        "modes": torch.tensor([row["policy_mode"] for row in rows], dtype=torch.int64),
        "waypoints": torch.tensor([row["waypoint"] for row in rows], dtype=torch.float32),
    }
    input_sha_before = state_sha256(replay_inputs)
    with torch.no_grad():
        initial = replay_individual(manager, rows, indices, device)
        behavior = torch.tensor(
            [row["behavior_log_prob"] for row in rows], dtype=torch.float32, device=device
        )
        preupdate_ratios = torch.exp(initial["log_probs"].float() - behavior)
        ratio_deviation = float((preupdate_ratios - 1.0).abs().max())
    if ratio_deviation > run_cfg["smoke"]["preupdate_ratio_max_deviation"]:
        raise RuntimeError("G7.17 pre-update replay ratio differs from one")

    advantages_raw = torch.tensor([row["advantage"] for row in rows], dtype=torch.float32)
    advantage_std = advantages_raw.std(unbiased=False)
    if not bool(torch.isfinite(advantage_std)) or float(advantage_std) <= 1e-8:
        raise RuntimeError("G7.17 coverage batch has degenerate advantages")
    advantages = (advantages_raw - advantages_raw.mean()) / advantage_std
    returns = torch.tensor([row["return"] for row in rows], dtype=torch.float32, device=device)
    replay = replay_individual(manager, rows, indices, device)
    old_log_probs = torch.tensor(
        [row["behavior_log_prob"] for row in rows], dtype=torch.float32, device=device
    )
    ratios = torch.exp(replay["log_probs"].float() - old_log_probs)
    clipped = ratios.clamp(
        1.0 - run_cfg["clip_ratio"], 1.0 + run_cfg["clip_ratio"]
    )
    actor_loss = -torch.minimum(
        ratios * advantages.to(device), clipped * advantages.to(device)
    ).mean()
    value_loss = 0.5 * (replay["values"].float() - returns).square().mean()
    entropy = replay["entropy"].float().mean()
    loss = (
        actor_loss
        + run_cfg["value_coef"] * value_loss
        - run_cfg["entropy_coef"] * entropy
    )
    finite_loss = all(bool(torch.isfinite(value)) for value in (
        loss, actor_loss, value_loss, entropy
    ))
    if not finite_loss:
        raise RuntimeError("G7.17 PPO loss is nonfinite")
    manager.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    audit = gradient_audit(core)
    if (
        not audit["finite"]
        or audit["group_norm_sums"]["high_level_policy"] <= 0
        or audit["group_norm_sums"]["value_head"] <= 0
        or audit["frozen_with_grad"]
    ):
        raise RuntimeError("G7.17 gradient isolation failed")
    grad_norm, learning_rates = manager.optimizer_step()
    core.mark_updated()

    changes = parameter_changes(core, before_parameters)
    frozen_after, after_names = frozen_parameter_sentinels(core)
    frozen_unchanged = (
        after_names == sentinel_names
        and state_sha256(frozen_after) == state_sha256(frozen_before)
    )
    input_sha_after = state_sha256(replay_inputs)
    update = {
        "sample_count": len(rows),
        "mode_ids": [row["policy_mode"] for row in rows],
        "episode_slots": [row["episode_slot"] for row in rows],
        "batch_shapes": {
            "log_probs": list(replay["log_probs"].shape),
            "values": list(replay["values"].shape),
            "mode_logits": list(replay["mode_logits"].shape),
            "waypoint_mean": list(replay["waypoint_mean"].shape),
        },
        "preupdate_ratio_max_deviation": ratio_deviation,
        "ratio_mean_at_update": float(ratios.detach().mean()),
        "loss": float(loss.detach()),
        "actor_loss": float(actor_loss.detach()),
        "value_loss": float(value_loss.detach()),
        "entropy": float(entropy.detach()),
        "advantage_raw_mean": float(advantages_raw.mean()),
        "advantage_raw_std": float(advantage_std),
        "grad_norm": float(torch.as_tensor(grad_norm)),
        "learning_rates": [float(value) for value in learning_rates],
        "gradient_audit": audit,
        "optimizer_scope": optimizer_scope,
        "actor_version_before": actor_version_before,
        "actor_version_after": core.actor_version,
        "parameter_changes": changes,
        "frozen_sentinel_names": sentinel_names,
        "frozen_sentinel_sha256_before": state_sha256(frozen_before),
        "frozen_sentinel_sha256_after": state_sha256(frozen_after),
        "frozen_sentinels_unchanged": frozen_unchanged,
        "rollout_input_sha256_before": input_sha_before,
        "rollout_input_sha256_after": input_sha_after,
        "rollout_inputs_unchanged": input_sha_before == input_sha_after,
    }
    return update, rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--g716-result", type=Path, required=True)
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
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--master-port", type=int, default=29597)
    parser.add_argument("--context-tokens", type=int, default=4096)
    parser.add_argument("--head-hidden", type=int, default=256)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    started = time.time()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = args.manifest.resolve(strict=True)
    if sha256_file(manifest_path) != MANIFEST_SHA256:
        raise RuntimeError("G7.17 frozen manifest hash changed")
    manifest = json.loads(manifest_path.read_text())
    smoke = validate_manifest(manifest)
    g716_result = args.g716_result.resolve(strict=True)
    if sha256_file(g716_result) != G716_RESULT_SHA256:
        raise RuntimeError("G7.17 G7.16 accepted-result anchor changed")

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
        raise RuntimeError("G7.17 source commit changed")
    datasets = dataset_hashes(Path(paths["asset_root"]), manifest)
    script_root = Path(__file__).resolve().parent
    implementation_paths = {
        name: script_root / name
        for name in (
            "run_streamvln_plumbing_g717.py", "streamvln_topological_policy.py",
            "streamvln_recovery_policy.py", "streamvln_executor_contract.py",
            "streamvln_topology.py", "streamvln_chunk_ppo.py",
        )
    }
    implementation_paths["rlinf/envs/streamvln/streamvln_env.py"] = (
        Path(paths["rlinf_root"]) / "rlinf" / "envs" / "streamvln" / "streamvln_env.py"
    )
    implementation_paths["rlinf/envs/streamvln/streamvln_executor_contract.py"] = (
        Path(paths["rlinf_root"]) / "rlinf" / "envs" / "streamvln"
        / "streamvln_executor_contract.py"
    )
    implementation_paths["rlinf/models/embodiment/streamvln_ppo_policy.py"] = (
        Path(paths["rlinf_root"]) / "rlinf" / "models" / "embodiment"
        / "streamvln_ppo_policy.py"
    )
    implementation = {
        name: sha256_file(path.resolve(strict=True))
        for name, path in implementation_paths.items()
    }
    run_cfg = {
        **paths,
        "output_dir": str(output),
        "stream_commit": ORIGINAL_STREAM_COMMIT,
        "rlinf_commit": RLINF_COMMIT,
        "manifest": {"source": manifest["source"], "train": smoke["scene_specs"], "val_unseen": []},
        "smoke": smoke,
        "policy_observation_keys": manifest["policy_observation_keys"],
        "context_tokens": args.context_tokens,
        "head_hidden": args.head_hidden,
        "seed": args.seed,
        "master_port": args.master_port,
        "learning_rate": smoke["learning_rate"],
        "value_learning_rate": smoke["value_learning_rate"],
        "gamma": smoke["gamma"],
        "gae_lambda": smoke["gae_lambda"],
        "clip_ratio": smoke["clip_ratio"],
        "value_coef": smoke["value_coef"],
        "entropy_coef": smoke["entropy_coef"],
    }
    config = {
        "kind": "streamvln-g7.17-bounded-high-level-rl-plumbing",
        "manifest": str(manifest_path),
        "manifest_sha256": MANIFEST_SHA256,
        "g716_result": str(g716_result),
        "g716_result_sha256": G716_RESULT_SHA256,
        "source_heads": source_heads,
        "datasets": datasets,
        "implementation_sha256": implementation,
        "model_input_inventory": {
            "base_model": small_file_inventory(Path(paths["base_model"])),
            "tokenizer": small_file_inventory(Path(paths["tokenizer"])),
            "vision_tower": small_file_inventory(Path(paths["vision_tower"])),
        },
        "run": run_cfg,
        "model_load_count": 1,
        "optimizer_update_limit": 1,
        "val_unseen_episode_count": 0,
    }
    config_path = output / "run_config.json"
    write_json(config_path, config)

    manager = core = None
    try:
        manager, core, trainable_manifest = setup_manager(run_cfg, total_steps=1)
        update, rows = run_update(manager, core, run_cfg)
        rows_path = output / "rollout_records.jsonl"
        rows_path.write_text("".join(
            json.dumps(serialized_row(row), sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ))
        mode_ids = [row["policy_mode"] for row in rows]
        durations = [row["duration"] for row in rows]
        boundaries = ["terminated" if row["terminated"] else "truncated" for row in rows]
        bootstraps = [row["bootstrap"] for row in rows]
        maximum_reward_error = max(row["reward_accounting_error"] for row in rows)
        guards = {
            "source_pins": source_heads == {"stream": ORIGINAL_STREAM_COMMIT, "rlinf": RLINF_COMMIT},
            "manifest_hash": sha256_file(manifest_path) == MANIFEST_SHA256,
            "predecessor_hash": sha256_file(g716_result) == G716_RESULT_SHA256,
            "dataset_hashes": all(
                datasets[split]["sha256"] == manifest["source"][f"{split}_sha256"]
                for split in datasets
            ),
            "original_stream_frozen_scope": (
                trainable_manifest["source_kind"] == "upstream-original"
                and trainable_manifest["memory_adapter"] is False
                and trainable_manifest["secondary_post_training_used"] is False
                and trainable_manifest["backbone_trainable"] is False
            ),
            "optimizer_scope": (
                update["optimizer_scope"]["optimizer_equals_trainable"]
                and update["optimizer_scope"]["optimizer_frozen_overlap_count"] == 0
            ),
            "mode_coverage": mode_ids == [NAVIGATE_MODE, STOP_MODE, RECOVER_MODE],
            "executed_prefixes": durations == smoke["expected_executed_lengths"],
            "end_boundaries": boundaries == smoke["expected_end_boundaries"],
            "causal_feedback": all(
                set(row["executor_feedback"]) == {
                    "schema_version", "proposed_actions", "executed_actions", "executed_mask",
                    "primitives", "commanded_forward_m", "realized_path_m", "collision_count",
                    "blocked_forward_count", "environment_reward_sum", "end_reason",
                }
                for row in rows
            ),
            "history_only_observation": not (
                set(manifest["policy_observation_keys"]) & PRIVILEGED_POLICY_KEYS
            ),
            "reward_accounting": maximum_reward_error <= 1e-6,
            "termination_bootstrap_zero": bootstraps[1] == 0.0
            and rows[1]["bootstrap_source"] == "natural_termination_zero",
            "truncation_bootstrap_observation": all(
                math.isfinite(rows[index]["bootstrap"])
                and rows[index]["bootstrap_source"] == "terminal_frozen_observation_value"
                for index in (0, 2)
            ),
            "episode_local_variable_duration_gae": durations == [4, 1, 4]
            and all(math.isfinite(row["advantage"]) and math.isfinite(row["return"]) for row in rows),
            "preupdate_ratio": update["preupdate_ratio_max_deviation"]
            <= smoke["preupdate_ratio_max_deviation"],
            "batch_shapes": update["batch_shapes"] == {
                "log_probs": [3], "values": [3], "mode_logits": [3, 3], "waypoint_mean": [3, 2]
            },
            "finite_loss_and_gradients": (
                all(math.isfinite(update[key]) for key in ("loss", "actor_loss", "value_loss", "entropy", "grad_norm"))
                and update["gradient_audit"]["finite"]
                and update["gradient_audit"]["group_norm_sums"]["high_level_policy"] > 0
                and update["gradient_audit"]["group_norm_sums"]["value_head"] > 0
                and not update["gradient_audit"]["frozen_with_grad"]
            ),
            "exactly_one_update": update["actor_version_after"] - update["actor_version_before"] == 1,
            "high_level_parameters_changed": update["parameter_changes"]["changed_tensor_count"] > 0,
            "frozen_sentinels_unchanged": update["frozen_sentinels_unchanged"],
            "rollout_inputs_unchanged": update["rollout_inputs_unchanged"],
            "no_checkpoint_or_navigation_evaluation": (
                manifest["checkpoint_selection"] is False
                and manifest["navigation_evaluation"] is False
            ),
            "zero_val_unseen": config["val_unseen_episode_count"] == 0,
        }
        all_passed = all(guards.values())
        result = {
            "kind": config["kind"],
            "all_protocol_guards_passed": all_passed,
            "high_level_rl_plumbing_ready": all_passed,
            "coverage_conditioned_batch": True,
            "unbiased_learning_or_navigation_claim": False,
            "source_contract": manifest["source_contract"],
            "trainable_manifest": trainable_manifest,
            "counts": {
                "model_loads": 1,
                "train_episodes": len(rows),
                "rollout_rows": len(rows),
                "executed_primitives": sum(durations),
                "optimizer_updates": 1,
                "val_unseen_episodes": 0,
            },
            "mode_ids": mode_ids,
            "durations": durations,
            "end_boundaries": boundaries,
            "maximum_reward_accounting_error": maximum_reward_error,
            "update": update,
            "protocol_guards": guards,
            "g8_ready": False,
            "permitted_next_stage": (
                "G7.18 train-only unbiased learning-signal and short behavior smoke with no-update control"
                if all_passed else "G7.17 plumbing repair only"
            ),
            "elapsed_seconds": time.time() - started,
            "evidence": {
                "manifest_sha256": MANIFEST_SHA256,
                "run_config_sha256": sha256_file(config_path),
                "rollout_records_sha256": sha256_file(rows_path),
            },
        }
        result_path = output / "result.json"
        write_json(result_path, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        if not all_passed:
            raise SystemExit(2)
    finally:
        if manager is not None:
            cleanup_manager(manager)


if __name__ == "__main__":
    main()
