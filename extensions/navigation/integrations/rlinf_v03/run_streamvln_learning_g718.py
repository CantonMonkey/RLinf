"""Run the preregistered G7.18 unbiased learning-signal and behavior smoke."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from run_streamvln_plumbing_g717 import (
    dataset_hashes,
    frozen_parameter_sentinels,
    git_head,
    optimizer_parameter_scope,
    sha256_file,
    small_file_inventory,
    validate_identity,
    write_json,
)
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
from streamvln_topology import (
    MODE_NAMES,
    NAVIGATE_MODE,
    OnlineTopology,
    RECOVER_MODE,
    STOP_MODE,
)


ORIGINAL_STREAM_COMMIT = "e48f6ff7e9201d93aae003e8f64b04c00cec13bc"
RLINF_COMMIT = "0505431899574619da86f551bad70b71e0ea2177"
G717_RESULT_SHA256 = "143303650dca8c2832386a516c613e208638f6967a87084654bb1a9b55ea09fd"
MANIFEST_SHA256 = "a985243b5aadf19ee8ab37ac534aa1292d095588f8a49427c984e81656864aab"
FEEDBACK_KEYS = {
    "schema_version", "proposed_actions", "executed_actions", "executed_mask",
    "primitives", "commanded_forward_m", "realized_path_m", "collision_count",
    "blocked_forward_count", "environment_reward_sum", "end_reason",
}


def validate_manifest(manifest: dict) -> dict:
    expected = {
        "schema_version", "stage", "source_contract", "source", "predecessor",
        "policy_observation_keys", "privileged_policy_blacklist", "protocol",
        "required_reports", "gates", "model_load", "rl_update",
        "navigation_evaluation", "controller_comparison", "robot_execution", "g8_ready",
    }
    if set(manifest) != expected or manifest["schema_version"] != 1:
        raise ValueError("G7.18 manifest schema changed")
    if manifest["source_contract"] != {
        "stream_commit": ORIGINAL_STREAM_COMMIT,
        "rlinf_commit": RLINF_COMMIT,
        "original_stream_only": True,
        "ttt_or_secondary_post_training": False,
    }:
        raise ValueError("G7.18 source contract changed")
    if manifest["predecessor"]["accepted_result_sha256"] != G717_RESULT_SHA256:
        raise ValueError("G7.18 predecessor anchor changed")
    if set(manifest["privileged_policy_blacklist"]) != PRIVILEGED_POLICY_KEYS:
        raise ValueError("G7.18 privileged-data blacklist changed")
    assert_history_only_keys(manifest["policy_observation_keys"])
    protocol = manifest["protocol"]
    collection = protocol["collection_scene_specs"]
    behavior = protocol["behavior_scene_specs"]
    if (
        protocol["split"] != "train"
        or len(collection) != 4
        or len(behavior) != 4
        or set(row["scene"] for row in collection) & set(row["scene"] for row in behavior)
        or protocol["sampling"] != {
            "method": "one ordinary stochastic policy sample per reached decision",
            "seed_domain": "sha256(base_seed, g718, phase, episode_id, decision_index)",
            "sample_calls_per_decision": 1,
            "target_mode_scan": False,
            "mode_filter_or_resample": False,
            "recovery_override": False,
        }
        or protocol["max_decisions_per_episode"] != 3
        or protocol["max_primitives_per_episode"] != 12
        or protocol["ppo_full_batch_updates"] != 2
        or protocol["behavior_improvement_required"] is not False
        or protocol["checkpoint_selection"] is not False
        or protocol["val_unseen_episode_count"] != 0
    ):
        raise ValueError("G7.18 protocol changed")
    if (
        manifest["model_load"] is not True
        or manifest["rl_update"] is not True
        or manifest["navigation_evaluation"] is not False
        or manifest["controller_comparison"] is not False
        or manifest["robot_execution"] is not False
        or manifest["g8_ready"] is not False
    ):
        raise ValueError("G7.18 claim boundary changed")
    return protocol


def observation_sha256(obs: dict, expected_keys: set[str]) -> str:
    if set(obs) != expected_keys:
        raise RuntimeError("G7.18 policy observation schema changed")
    assert_history_only_keys(list(obs))
    state = {name: value.detach().cpu().contiguous() for name, value in obs.items()}
    return state_sha256(state)


def execute_sampled(env, topology, mode: int, sampled_waypoint, decision_index: int):
    if mode == STOP_MODE:
        command_waypoint = [0.0, 0.0]
    elif mode == RECOVER_MODE:
        command_waypoint = list(recovery_target(topology))
    else:
        command_waypoint = [float(value) for value in sampled_waypoint]
    command = HighLevelCommand(
        decision_index=decision_index,
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
        raise RuntimeError("G7.18 executor proposal differs from sampled command")
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


def seed_schedule(run_cfg: dict, spec: dict, seed_phase: str) -> list[int]:
    return [
        stable_seed(run_cfg["seed"], "g718", seed_phase, spec["episode_id"], index)
        for index in range(run_cfg["protocol"]["max_decisions_per_episode"])
    ]


def collect_episode(manager, core, run_cfg, spec, *, phase: str, seed_phase: str, slot: int):
    device = model_device(core)
    max_decisions = run_cfg["protocol"]["max_decisions_per_episode"]
    env = make_env(
        run_cfg,
        split="train",
        episode_index=spec["index"],
        max_steps=run_cfg["protocol"]["max_primitives_per_episode"],
    )
    topology = OnlineTopology(max_decisions=max_decisions)
    schedule = seed_schedule(run_cfg, spec, seed_phase)
    rows = []
    try:
        obs, identity = env.reset()
        validate_identity(identity, spec)
        initial_observation = observation_sha256(
            obs, set(run_cfg["policy_observation_keys"])
        )
        for decision_index in range(max_decisions):
            with torch.no_grad():
                state = manager.model(operation="encode_state", obs=obs)
                top = topology_tensor(topology, device)
                sampled = manager.model(
                    operation="sample_head",
                    state_features=state,
                    topology_features=top,
                    seed=schedule[decision_index],
                    deterministic=False,
                )
                reference = manager.model(
                    operation="head_distribution",
                    state_features=state,
                    topology_features=top,
                )
            mode = int(sampled["modes"].reshape(-1)[0])
            waypoint = sampled["waypoints"].reshape(-1, 2)[0].detach().cpu().tolist()
            step = execute_sampled(env, topology, mode, waypoint, decision_index)
            feedback = step["feedback"]
            duration = len(feedback.executed_actions)
            reward_row = step["primitive_rewards"]
            if any(abs(value) > 1e-8 for value in reward_row[duration:]):
                raise RuntimeError("G7.18 padded reward is nonzero")
            chunk_rewards, durations = collapse_chunk_rewards(
                [reward_row], [list(feedback.executed_mask)], run_cfg["gamma"]
            )
            reward_error = max(
                abs(chunk_rewards[0] - feedback.discounted_environment_reward(run_cfg["gamma"])),
                abs(feedback.environment_reward_sum - sum(reward_row[:duration])),
            )
            if durations != [duration] or reward_error > 1e-6:
                raise RuntimeError("G7.18 reward collapse differs from executor feedback")
            rows.append({
                "episode_slot": slot,
                "decision_index": decision_index,
                "phase": phase,
                "sample_call_count": 1,
                "sampling_seed": schedule[decision_index],
                "policy_mode": mode,
                "state": state.detach().cpu().float(),
                "topology_features": top.detach().cpu().float(),
                "waypoint": [float(value) for value in waypoint],
                "behavior_log_prob": float(sampled["behavior_log_probs"].reshape(-1)[0]),
                "value": float(sampled["values"].reshape(-1)[0]),
                "reference_mode_logits": reference["mode_logits"].reshape(-1, 3)[0].detach().cpu().tolist(),
                "reference_waypoint_mean": reference["waypoint_mean"].reshape(-1, 2)[0].detach().cpu().tolist(),
                "reference_waypoint_log_std": reference["waypoint_log_std"].reshape(-1, 2)[0].detach().cpu().tolist(),
                "command": step["command"],
                "executor_feedback": feedback.to_dict(),
                "primitive_rewards": reward_row,
                "chunk_reward": float(chunk_rewards[0]),
                "duration": duration,
                "terminated": step["terminated"],
                "truncated": step["truncated"],
                "termination_row": step["termination_row"],
                "truncation_row": step["truncation_row"],
                "reward_accounting_error": reward_error,
                "recovery_override": False,
            })
            obs = step["obs"]
            if step["terminated"] or step["truncated"]:
                break
        if not rows:
            raise RuntimeError("G7.18 episode produced no decision")
        final = rows[-1]
        if final["terminated"]:
            bootstrap = 0.0
            bootstrap_source = "natural_termination_zero"
        elif final["truncated"]:
            if not bool(obs["terminal_frozen"].item()):
                raise RuntimeError("G7.18 truncation did not expose a frozen final observation")
            with torch.no_grad():
                bootstrap_state = manager.model(operation="encode_bootstrap_state", obs=obs)
                bootstrap_topology = topology_tensor(topology, device)
                bootstrap_dist = manager.model(
                    operation="head_distribution",
                    state_features=bootstrap_state,
                    topology_features=bootstrap_topology,
                )
            bootstrap = float(bootstrap_dist["values"].reshape(-1)[0])
            bootstrap_source = "terminal_frozen_observation_value"
        else:
            raise RuntimeError("G7.18 fixed episode did not reach STOP or time limit")
        if not math.isfinite(bootstrap):
            raise RuntimeError("G7.18 bootstrap is nonfinite")
        rewards = [row["chunk_reward"] for row in rows]
        values = [row["value"] for row in rows]
        durations = [row["duration"] for row in rows]
        dones = [row["terminated"] for row in rows]
        advantages, returns, deltas = variable_duration_gae(
            rewards,
            values,
            values[1:] + [bootstrap],
            durations,
            dones,
            run_cfg["gamma"],
            run_cfg["gae_lambda"],
        )
        for row, advantage, target_return, delta in zip(rows, advantages, returns, deltas):
            row["advantage"] = float(advantage)
            row["return"] = float(target_return)
            row["delta"] = float(delta)
        return {
            "phase": phase,
            "episode_slot": slot,
            "episode_index": spec["index"],
            "episode_id": str(spec["episode_id"]),
            "scene": spec["scene"],
            "initial_observation_sha256": initial_observation,
            "sampling_seed_schedule": schedule,
            "decision_count": len(rows),
            "primitive_count": sum(row["duration"] for row in rows),
            "bootstrap": bootstrap,
            "bootstrap_source": bootstrap_source,
            "terminated": final["terminated"],
            "truncated": final["truncated"],
            "rows": rows,
            "topology": topology.summary(),
        }
    finally:
        env.close()


def collect_phase(manager, core, run_cfg, specs, *, phase: str, seed_phase: str):
    return [
        collect_episode(
            manager, core, run_cfg, spec, phase=phase, seed_phase=seed_phase, slot=slot
        )
        for slot, spec in enumerate(specs)
    ]


def flat_rows(episodes):
    return [row for episode in episodes for row in episode["rows"]]


def scalar_distribution(values) -> dict:
    rows = [float(value) for value in values]
    if not rows:
        raise ValueError("distribution cannot be empty")
    return {
        "count": len(rows),
        "min": min(rows),
        "mean": sum(rows) / len(rows),
        "max": max(rows),
    }


def aggregate_behavior(episodes) -> dict:
    rows = flat_rows(episodes)
    modes = Counter(MODE_NAMES[row["policy_mode"]] for row in rows)
    actions = Counter()
    for row in rows:
        actions.update(str(action) for action in row["executor_feedback"]["executed_actions"])
    return {
        "episode_count": len(episodes),
        "decision_count": len(rows),
        "primitive_count": sum(row["duration"] for row in rows),
        "environment_reward_sum": sum(
            row["executor_feedback"]["environment_reward_sum"] for row in rows
        ),
        "discounted_chunk_reward_sum": sum(row["chunk_reward"] for row in rows),
        "realized_path_m": sum(row["executor_feedback"]["realized_path_m"] for row in rows),
        "collision_count": sum(row["executor_feedback"]["collision_count"] for row in rows),
        "blocked_forward_count": sum(
            row["executor_feedback"]["blocked_forward_count"] for row in rows
        ),
        "stop_decision_count": modes["stop"],
        "recover_decision_count": modes["recover"],
        "navigate_decision_count": modes["navigate"],
        "terminated_episode_count": sum(episode["terminated"] for episode in episodes),
        "timeout_episode_count": sum(episode["truncated"] for episode in episodes),
        "mode_distribution": {name: modes[name] for name in MODE_NAMES},
        "primitive_action_distribution": {
            name: actions[str(index)] for index, name in enumerate(("stop", "forward", "left", "right"))
        },
        "reward_distribution": scalar_distribution(row["chunk_reward"] for row in rows),
        "advantage_distribution": scalar_distribution(row["advantage"] for row in rows),
        "return_distribution": scalar_distribution(row["return"] for row in rows),
    }


def paired_behavior(control, updated) -> dict:
    if len(control) != len(updated):
        raise RuntimeError("G7.18 behavior episode count changed")
    pairs = []
    for before, after in zip(control, updated):
        if before["episode_id"] != after["episode_id"] or before["scene"] != after["scene"]:
            raise RuntimeError("G7.18 behavior identity changed")
        before_summary = aggregate_behavior([before])
        after_summary = aggregate_behavior([after])
        pairs.append({
            "episode_id": before["episode_id"],
            "scene": before["scene"],
            "initial_observation_match": (
                before["initial_observation_sha256"] == after["initial_observation_sha256"]
            ),
            "seed_schedule_match": (
                before["sampling_seed_schedule"] == after["sampling_seed_schedule"]
            ),
            "control": before_summary,
            "updated": after_summary,
            "delta": {
                key: after_summary[key] - before_summary[key]
                for key in (
                    "decision_count", "primitive_count", "environment_reward_sum",
                    "discounted_chunk_reward_sum", "realized_path_m", "collision_count",
                    "blocked_forward_count", "stop_decision_count", "recover_decision_count",
                    "navigate_decision_count", "terminated_episode_count", "timeout_episode_count",
                )
            },
        })
    return {
        "all_initial_observations_match": all(row["initial_observation_match"] for row in pairs),
        "all_seed_schedules_match": all(row["seed_schedule_match"] for row in pairs),
        "pairs": pairs,
    }


def structured_distribution_kl(initial, current, reference_log_std, current_log_std):
    reference_logits = initial["mode_logits"].float()
    current_logits = current["mode_logits"].float()
    reference_log_probs = reference_logits.log_softmax(-1)
    reference_probs = reference_log_probs.exp()
    categorical = (
        reference_probs * (reference_log_probs - current_logits.log_softmax(-1))
    ).sum(-1)
    reference_mean = initial["waypoint_mean"].float()
    current_mean = current["waypoint_mean"].float()
    ref_log_std = reference_log_std.float().reshape(1, 2).expand_as(reference_mean)
    cur_log_std = current_log_std.float().reshape(1, 2).expand_as(current_mean)
    gaussian = (
        cur_log_std
        - ref_log_std
        + (ref_log_std.mul(2).exp() + (reference_mean - current_mean).square())
        / (2.0 * cur_log_std.mul(2).exp())
        - 0.5
    ).sum(-1)
    combined = (categorical + reference_probs[:, NAVIGATE_MODE] * gaussian).clamp_min(0.0)
    return combined, categorical.clamp_min(0.0), gaussian.clamp_min(0.0)


def train_updates(manager, core, run_cfg, episodes):
    rows = flat_rows(episodes)
    if len(rows) < 2:
        raise RuntimeError("G7.18 collection batch is too small")
    device = model_device(core)
    indices = list(range(len(rows)))
    replay_inputs = {
        **{f"state_{index}": row["state"] for index, row in enumerate(rows)},
        **{f"topology_{index}": row["topology_features"] for index, row in enumerate(rows)},
        "modes": torch.tensor([row["policy_mode"] for row in rows], dtype=torch.int64),
        "waypoints": torch.tensor([row["waypoint"] for row in rows], dtype=torch.float32),
    }
    input_sha_before = state_sha256(replay_inputs)
    before_parameters = parameter_snapshot(core)
    actor_version_before = core.actor_version
    with torch.no_grad():
        initial = replay_individual(manager, rows, indices, device)
        behavior_log_probs = torch.tensor(
            [row["behavior_log_prob"] for row in rows], dtype=torch.float32, device=device
        )
        initial_ratios = torch.exp(initial["log_probs"].float() - behavior_log_probs)
        preupdate_deviation = float((initial_ratios - 1.0).abs().max())
        reference_log_std = core.waypoint_log_std.detach().float().clamp(-3.0, -0.25).clone()
    if preupdate_deviation > run_cfg["protocol"]["preupdate_ratio_max_deviation"]:
        raise RuntimeError("G7.18 pre-update ratio differs from one")
    advantages_raw = torch.tensor([row["advantage"] for row in rows], dtype=torch.float32)
    advantage_std = advantages_raw.std(unbiased=False)
    if not bool(torch.isfinite(advantage_std)) or float(advantage_std) <= 1e-8:
        raise RuntimeError("G7.18 ordinary collection advantages are degenerate")
    advantages = (advantages_raw - advantages_raw.mean()) / advantage_std
    returns = torch.tensor([row["return"] for row in rows], dtype=torch.float32, device=device)
    updates = []
    for update_index in range(run_cfg["protocol"]["ppo_full_batch_updates"]):
        replay = replay_individual(manager, rows, indices, device)
        ratios = torch.exp(replay["log_probs"].float() - behavior_log_probs)
        clipped = ratios.clamp(
            1.0 - run_cfg["clip_ratio"], 1.0 + run_cfg["clip_ratio"]
        )
        actor_loss = -torch.minimum(
            ratios * advantages.to(device), clipped * advantages.to(device)
        ).mean()
        value_loss = 0.5 * (replay["values"].float() - returns).square().mean()
        entropy = replay["entropy"].float().mean()
        loss = actor_loss + run_cfg["value_coef"] * value_loss - run_cfg["entropy_coef"] * entropy
        if not all(bool(torch.isfinite(value)) for value in (loss, actor_loss, value_loss, entropy)):
            raise RuntimeError("G7.18 update loss is nonfinite")
        manager.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        audit = gradient_audit(core)
        if (
            not audit["finite"]
            or audit["group_norm_sums"]["high_level_policy"] <= 0
            or audit["group_norm_sums"]["value_head"] <= 0
            or audit["frozen_with_grad"]
        ):
            raise RuntimeError("G7.18 gradient isolation failed")
        grad_norm, learning_rates = manager.optimizer_step()
        core.mark_updated()
        updates.append({
            "update_index": update_index,
            "ratio_mean": float(ratios.detach().mean()),
            "ratio_max_deviation": float((ratios.detach() - 1.0).abs().max()),
            "loss": float(loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "value_loss": float(value_loss.detach()),
            "entropy": float(entropy.detach()),
            "grad_norm": float(torch.as_tensor(grad_norm)),
            "learning_rates": [float(value) for value in learning_rates],
            "gradient_audit": audit,
            "actor_version_after": core.actor_version,
        })
    with torch.no_grad():
        final = replay_individual(manager, rows, indices, device)
        final_ratios = torch.exp(final["log_probs"].float() - behavior_log_probs)
        current_log_std = core.waypoint_log_std.detach().float().clamp(-3.0, -0.25)
        combined_kl, categorical_kl, gaussian_kl = structured_distribution_kl(
            initial, final, reference_log_std, current_log_std
        )
    input_sha_after = state_sha256(replay_inputs)
    return {
        "row_count": len(rows),
        "episode_count": len(episodes),
        "preupdate_ratio_max_deviation": preupdate_deviation,
        "advantage_raw": scalar_distribution(advantages_raw.tolist()),
        "return_raw": scalar_distribution(row["return"] for row in rows),
        "updates": updates,
        "update_count": len(updates),
        "actor_version_before": actor_version_before,
        "actor_version_after": core.actor_version,
        "postupdate_ratio_mean": float(final_ratios.mean()),
        "postupdate_ratio_max_deviation": float((final_ratios - 1.0).abs().max()),
        "postupdate_distribution_kl": {
            "mean": float(combined_kl.mean()),
            "max": float(combined_kl.max()),
            "categorical_mean": float(categorical_kl.mean()),
            "gaussian_mean": float(gaussian_kl.mean()),
        },
        "parameter_changes": parameter_changes(core, before_parameters),
        "collection_input_sha256_before": input_sha_before,
        "collection_input_sha256_after": input_sha_after,
        "collection_inputs_unchanged": input_sha_before == input_sha_after,
    }, rows


def serialized_episode(episode):
    result = {key: jsonable(value) for key, value in episode.items() if key != "rows"}
    result["rows"] = [
        {key: jsonable(value) for key, value in row.items()
         if key not in ("state", "topology_features")}
        for row in episode["rows"]
    ]
    return result


def write_jsonl(path: Path, episodes) -> None:
    path.write_text("".join(
        json.dumps(serialized_episode(episode), sort_keys=True, separators=(",", ":")) + "\n"
        for episode in episodes
    ))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--g717-result", type=Path, required=True)
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
    parser.add_argument("--master-port", type=int, default=29608)
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
        raise RuntimeError("G7.18 frozen manifest hash changed")
    manifest = json.loads(manifest_path.read_text())
    protocol = validate_manifest(manifest)
    predecessor = args.g717_result.resolve(strict=True)
    if sha256_file(predecessor) != G717_RESULT_SHA256:
        raise RuntimeError("G7.18 G7.17 accepted-result anchor changed")
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
        raise RuntimeError("G7.18 source commit changed")
    datasets = dataset_hashes(Path(paths["asset_root"]), manifest)
    script_root = Path(__file__).resolve().parent
    implementation_paths = {
        name: script_root / name
        for name in (
            "run_streamvln_learning_g718.py", "run_streamvln_plumbing_g717.py",
            "streamvln_topological_policy.py", "streamvln_recovery_policy.py",
            "streamvln_executor_contract.py", "streamvln_topology.py",
            "streamvln_chunk_ppo.py",
        )
    }
    implementation_paths.update({
        "rlinf/envs/streamvln/streamvln_env.py": (
            Path(paths["rlinf_root"]) / "rlinf" / "envs" / "streamvln" / "streamvln_env.py"
        ),
        "rlinf/envs/streamvln/streamvln_executor_contract.py": (
            Path(paths["rlinf_root"]) / "rlinf" / "envs" / "streamvln"
            / "streamvln_executor_contract.py"
        ),
        "rlinf/models/embodiment/streamvln_ppo_policy.py": (
            Path(paths["rlinf_root"]) / "rlinf" / "models" / "embodiment"
            / "streamvln_ppo_policy.py"
        ),
    })
    implementation = {
        name: sha256_file(path.resolve(strict=True))
        for name, path in implementation_paths.items()
    }
    all_specs = protocol["collection_scene_specs"] + protocol["behavior_scene_specs"]
    run_cfg = {
        **paths,
        "output_dir": str(output),
        "stream_commit": ORIGINAL_STREAM_COMMIT,
        "rlinf_commit": RLINF_COMMIT,
        "manifest": {"source": manifest["source"], "train": all_specs, "val_unseen": []},
        "protocol": protocol,
        "policy_observation_keys": manifest["policy_observation_keys"],
        "context_tokens": args.context_tokens,
        "head_hidden": args.head_hidden,
        "seed": args.seed,
        "master_port": args.master_port,
        "learning_rate": protocol["learning_rate"],
        "value_learning_rate": protocol["value_learning_rate"],
        "gamma": protocol["gamma"],
        "gae_lambda": protocol["gae_lambda"],
        "clip_ratio": protocol["clip_ratio"],
        "value_coef": protocol["value_coef"],
        "entropy_coef": protocol["entropy_coef"],
    }
    config = {
        "kind": "streamvln-g7.18-unbiased-learning-and-behavior-smoke",
        "manifest": str(manifest_path),
        "manifest_sha256": MANIFEST_SHA256,
        "g717_result": str(predecessor),
        "g717_result_sha256": G717_RESULT_SHA256,
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
        "optimizer_update_limit": protocol["ppo_full_batch_updates"],
        "checkpoint_selection": False,
        "val_unseen_episode_count": 0,
    }
    config_path = output / "run_config.json"
    write_json(config_path, config)

    manager = core = None
    try:
        manager, core, trainable_manifest = setup_manager(
            run_cfg, total_steps=protocol["ppo_full_batch_updates"]
        )
        optimizer_scope = optimizer_parameter_scope(manager, core)
        frozen_before, sentinel_names = frozen_parameter_sentinels(core)
        initial_high_level = parameter_snapshot(core)
        initial_high_level_sha = state_sha256(initial_high_level)
        initial_actor_version = core.actor_version

        behavior_control = collect_phase(
            manager, core, run_cfg, protocol["behavior_scene_specs"],
            phase="behavior_control", seed_phase="behavior",
        )
        collection = collect_phase(
            manager, core, run_cfg, protocol["collection_scene_specs"],
            phase="collection", seed_phase="collection",
        )
        no_update_sha = state_sha256(parameter_snapshot(core))
        no_update_control_unchanged = (
            no_update_sha == initial_high_level_sha
            and core.actor_version == initial_actor_version
        )
        training, collection_rows = train_updates(manager, core, run_cfg, collection)
        behavior_updated = collect_phase(
            manager, core, run_cfg, protocol["behavior_scene_specs"],
            phase="behavior_updated", seed_phase="behavior",
        )
        frozen_after, after_names = frozen_parameter_sentinels(core)
        frozen_unchanged = (
            sentinel_names == after_names
            and state_sha256(frozen_before) == state_sha256(frozen_after)
        )

        collection_path = output / "collection_records.jsonl"
        control_path = output / "behavior_control.jsonl"
        updated_path = output / "behavior_updated.jsonl"
        paired_path = output / "paired_behavior.json"
        write_jsonl(collection_path, collection)
        write_jsonl(control_path, behavior_control)
        write_jsonl(updated_path, behavior_updated)
        control_summary = aggregate_behavior(behavior_control)
        updated_summary = aggregate_behavior(behavior_updated)
        collection_summary = aggregate_behavior(collection)
        paired = paired_behavior(behavior_control, behavior_updated)
        write_json(paired_path, paired)

        collection_scenes = {row["scene"] for row in protocol["collection_scene_specs"]}
        behavior_scenes = {row["scene"] for row in protocol["behavior_scene_specs"]}
        all_rows = flat_rows(collection + behavior_control + behavior_updated)
        max_reward_error = max(row["reward_accounting_error"] for row in all_rows)
        all_feedback_valid = all(
            set(row["executor_feedback"]) == FEEDBACK_KEYS for row in all_rows
        )
        all_single_sample = all(row["sample_call_count"] == 1 for row in all_rows)
        post_ratio_deviation = training["postupdate_ratio_max_deviation"]
        post_kl = training["postupdate_distribution_kl"]["max"]
        functional_change = max(
            post_ratio_deviation,
            training["postupdate_distribution_kl"]["mean"],
        )
        updates_finite = all(
            all(math.isfinite(update[key]) for key in (
                "loss", "actor_loss", "value_loss", "entropy", "grad_norm"
            ))
            and update["gradient_audit"]["finite"]
            and update["gradient_audit"]["group_norm_sums"]["high_level_policy"] > 0
            and update["gradient_audit"]["group_norm_sums"]["value_head"] > 0
            and not update["gradient_audit"]["frozen_with_grad"]
            for update in training["updates"]
        )
        required_reports_present = all(
            key in control_summary and key in updated_summary
            for key in (
                "reward_distribution", "advantage_distribution", "return_distribution",
                "realized_path_m", "collision_count", "blocked_forward_count",
                "stop_decision_count", "timeout_episode_count", "recover_decision_count",
                "primitive_action_distribution", "mode_distribution",
            )
        ) and bool(paired["pairs"])
        guards = {
            "source_pins": source_heads == {"stream": ORIGINAL_STREAM_COMMIT, "rlinf": RLINF_COMMIT},
            "manifest_hash": sha256_file(manifest_path) == MANIFEST_SHA256,
            "predecessor_hash": sha256_file(predecessor) == G717_RESULT_SHA256,
            "dataset_hashes": all(
                datasets[split]["sha256"] == manifest["source"][f"{split}_sha256"]
                for split in datasets
            ),
            "scene_disjoint_train_only": not (collection_scenes & behavior_scenes)
            and len(collection_scenes) == len(behavior_scenes) == 4,
            "ordinary_unconditioned_sampling": all_single_sample
            and all(not row["recovery_override"] for row in all_rows)
            and protocol["sampling"]["target_mode_scan"] is False,
            "no_update_control_unchanged": no_update_control_unchanged,
            "matched_behavior_contract": paired["all_initial_observations_match"]
            and paired["all_seed_schedules_match"],
            "causal_feedback_and_history_only": all_feedback_valid
            and not (set(manifest["policy_observation_keys"]) & PRIVILEGED_POLICY_KEYS),
            "reward_bootstrap_and_gae": max_reward_error <= 1e-6
            and all(math.isfinite(row["advantage"]) and math.isfinite(row["return"]) for row in all_rows),
            "preupdate_ratio": training["preupdate_ratio_max_deviation"]
            <= protocol["preupdate_ratio_max_deviation"],
            "finite_nonzero_gradients": updates_finite,
            "exact_update_count": training["update_count"] == protocol["ppo_full_batch_updates"]
            and training["actor_version_after"] - training["actor_version_before"]
            == protocol["ppo_full_batch_updates"],
            "postupdate_ratio_and_kl_bounded": post_ratio_deviation
            <= protocol["postupdate_ratio_max_deviation"]
            and post_kl <= protocol["postupdate_distribution_kl_max"],
            "postupdate_distribution_change_measurable": functional_change
            >= protocol["postupdate_functional_change_min"],
            "high_level_only_mutation": optimizer_scope["optimizer_equals_trainable"]
            and optimizer_scope["optimizer_frozen_overlap_count"] == 0
            and training["parameter_changes"]["changed_tensor_count"] > 0,
            "frozen_and_collection_inputs_unchanged": frozen_unchanged
            and training["collection_inputs_unchanged"],
            "required_behavior_reports": required_reports_present,
            "no_checkpoint_selection_or_navigation": config["checkpoint_selection"] is False
            and manifest["navigation_evaluation"] is False,
            "zero_val_unseen": config["val_unseen_episode_count"] == 0,
        }
        all_passed = all(guards.values())
        behavior_delta_observed = control_summary != updated_summary
        result = {
            "kind": config["kind"],
            "all_protocol_guards_passed": all_passed,
            "unbiased_train_learning_signal_ready": all_passed,
            "behavior_improvement_required": False,
            "behavior_delta_observed": behavior_delta_observed,
            "navigation_or_generalization_claim": False,
            "source_contract": manifest["source_contract"],
            "trainable_manifest": trainable_manifest,
            "counts": {
                "model_loads": 1,
                "collection_episodes": len(collection),
                "collection_rows": len(collection_rows),
                "behavior_control_episodes": len(behavior_control),
                "behavior_updated_episodes": len(behavior_updated),
                "optimizer_updates": training["update_count"],
                "val_unseen_episodes": 0,
            },
            "no_update_control": {
                "initial_high_level_sha256": initial_high_level_sha,
                "after_control_and_collection_sha256": no_update_sha,
                "actor_version_before": initial_actor_version,
                "actor_version_after": training["actor_version_before"],
                "unchanged": no_update_control_unchanged,
            },
            "optimizer_scope": optimizer_scope,
            "collection": collection_summary,
            "training": training,
            "behavior_control": control_summary,
            "behavior_updated": updated_summary,
            "paired_behavior": paired,
            "frozen_sentinel_names": sentinel_names,
            "frozen_sentinel_sha256_before": state_sha256(frozen_before),
            "frozen_sentinel_sha256_after": state_sha256(frozen_after),
            "maximum_reward_accounting_error": max_reward_error,
            "protocol_guards": guards,
            "g8_ready": False,
            "permitted_next_stage": (
                "independent result-to-claim decides bounded next train-only experiment"
                if all_passed else "G7.18 plumbing repair only"
            ),
            "elapsed_seconds": time.time() - started,
            "evidence": {
                "manifest_sha256": MANIFEST_SHA256,
                "run_config_sha256": sha256_file(config_path),
                "collection_records_sha256": sha256_file(collection_path),
                "behavior_control_sha256": sha256_file(control_path),
                "behavior_updated_sha256": sha256_file(updated_path),
                "paired_behavior_sha256": sha256_file(paired_path),
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
