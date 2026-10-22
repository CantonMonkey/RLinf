"""Stdlib-only independent recomputation for G7.23 executor-boundary evidence."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path


MANIFEST_SHA256 = "59af15397be0d6642474459ad74e21da4fb7896a28c76ea4e9f6da7326cc8297"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stable_seed(*parts) -> int:
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def exact_restore(evidence) -> bool:
    return bool(
        evidence
        and evidence.get("exact") is True
        and evidence.get("before") == evidence.get("after")
        and all(evidence.get("component_equal", {}).values())
    )


def identities(episodes):
    return [
        {"index": episode["episode_index"], "episode_id": str(episode["episode_id"]), "scene": episode["scene"]}
        for episode in episodes
    ]


def collection_check(episodes, specs, *, seed: int, phase: str, seed_phase: str, max_decisions: int, require_hashes=False):
    hashes_valid = all(
        all(
            isinstance(row.get(key), str)
            and len(row[key]) == 64
            and all(character in "0123456789abcdef" for character in row[key])
            for key in ("state_features_sha256", "topology_features_sha256", "policy_input_sha256")
        )
        for episode in episodes for row in episode["rows"]
    ) if require_hashes else True
    return {
        "identities_exact": identities(episodes) == specs,
        "phase_exact": all(
            episode["phase"] == phase and all(row["phase"] == phase for row in episode["rows"])
            for episode in episodes
        ),
        "schedules_exact": all(
            episode["sampling_seed_schedule"] == [
                stable_seed(seed, "g718", seed_phase, spec["episode_id"], index)
                for index in range(max_decisions)
            ]
            for episode, spec in zip(episodes, specs)
        ),
        "single_sample_exact": all(
            row["sampling_seed"] == stable_seed(seed, "g718", seed_phase, episode["episode_id"], row["decision_index"])
            and row["sample_call_count"] == 1
            and row["recovery_override"] is False
            for episode in episodes for row in episode["rows"]
        ),
        "policy_input_hashes_present": hashes_valid,
        "episode_count": len(episodes),
        "row_count": sum(len(episode["rows"]) for episode in episodes),
        "scenes": [episode["scene"] for episode in episodes],
    }


def behavior_metrics(episodes):
    rows = [row for episode in episodes for row in episode["rows"]]
    modes = Counter(row["policy_mode"] for row in rows)
    return {
        "episode_count": len(episodes),
        "decision_count": len(rows),
        "primitive_count": sum(row["duration"] for row in rows),
        "environment_reward_sum": sum(row["executor_feedback"]["environment_reward_sum"] for row in rows),
        "discounted_chunk_reward_sum": sum(row["chunk_reward"] for row in rows),
        "realized_path_m": sum(row["executor_feedback"]["realized_path_m"] for row in rows),
        "collision_count": sum(row["executor_feedback"]["collision_count"] for row in rows),
        "blocked_forward_count": sum(row["executor_feedback"]["blocked_forward_count"] for row in rows),
        "stop_decision_count": modes[1],
        "recover_decision_count": modes[2],
        "navigate_decision_count": modes[0],
        "terminated_episode_count": sum(bool(episode["terminated"]) for episode in episodes),
        "timeout_episode_count": sum(bool(episode["truncated"]) for episode in episodes),
    }


def approximately_equal(left, right, tolerance: float) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isfinite(float(left)) and math.isfinite(float(right)) and abs(float(left) - float(right)) <= tolerance
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(approximately_equal(left[key], right[key], tolerance) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(approximately_equal(a, b, tolerance) for a, b in zip(left, right))
    return left == right


def compiler_metadata(waypoint, compiler: dict) -> dict:
    forward, lateral = (float(value) for value in waypoint)
    bearing = math.degrees(math.atan2(lateral, forward))
    absolute = abs(bearing)
    turn_steps_uncapped = int(round(absolute / compiler["turn_degrees"]))
    turn_steps = min(4, turn_steps_uncapped)
    nearest = min(compiler["turn_step_boundaries_degrees"], key=lambda value: abs(absolute - value))
    turn_action = 2 if bearing > 0 else 3
    return {
        "bearing_degrees": bearing,
        "absolute_bearing_degrees": absolute,
        "nearest_half_step_boundary_degrees": nearest,
        "signed_margin_to_nearest_boundary_degrees": absolute - nearest,
        "absolute_margin_to_nearest_boundary_degrees": abs(absolute - nearest),
        "turn_steps_uncapped": turn_steps_uncapped,
        "turn_steps": turn_steps,
        "compiled_actions": [turn_action] * turn_steps + [1] * (4 - turn_steps),
    }


def feedback_delta(before: dict, after: dict) -> dict:
    return {
        key: after[key] - before[key]
        for key in ("environment_reward_sum", "collision_count", "blocked_forward_count", "realized_path_m", "commanded_forward_m")
    }


def analyze_boundary(initial, final, compiler: dict) -> dict:
    if len(initial) != len(final):
        raise ValueError("boundary episode count changed")
    rows = []
    episodes = []
    controls_match = True
    compiler_holds = True
    for before_episode, after_episode in zip(initial, final):
        if before_episode["episode_id"] != after_episode["episode_id"] or before_episode["scene"] != after_episode["scene"]:
            raise ValueError("boundary identity changed")
        identity_match = before_episode["initial_observation_sha256"] == after_episode["initial_observation_sha256"]
        schedule_match = before_episode["sampling_seed_schedule"] == after_episode["sampling_seed_schedule"]
        prior_equal = True
        episode_rows = []
        for decision_index in range(max(len(before_episode["rows"]), len(after_episode["rows"]))):
            if decision_index >= len(before_episode["rows"]) or decision_index >= len(after_episode["rows"]):
                record = {
                    "episode_id": before_episode["episode_id"], "scene": before_episode["scene"],
                    "decision_index": decision_index, "paired_row_present": False,
                    "causal_prefix": False, "descriptive_only": True,
                }
                rows.append(record)
                episode_rows.append(record)
                prior_equal = False
                continue
            before = before_episode["rows"][decision_index]
            after = after_episode["rows"][decision_index]
            input_equal = (
                before["state_features_sha256"] == after["state_features_sha256"]
                and before["topology_features_sha256"] == after["topology_features_sha256"]
                and before["policy_input_sha256"] == after["policy_input_sha256"]
            )
            causal = bool(identity_match and schedule_match and prior_equal and input_equal)
            same_mode = before["policy_mode"] == after["policy_mode"]
            navigate = same_mode and before["policy_mode"] == 0
            waypoint_l2 = math.dist(before["waypoint"], after["waypoint"])
            small = navigate and waypoint_l2 <= compiler["navigate_small_waypoint_shift_l2_max_m"]
            before_proposed = list(before["executor_feedback"]["proposed_actions"])
            after_proposed = list(after["executor_feedback"]["proposed_actions"])
            before_executed = list(before["executor_feedback"]["executed_actions"])
            after_executed = list(after["executor_feedback"]["executed_actions"])
            before_compiler = compiler_metadata(before["waypoint"], compiler) if before["policy_mode"] == 0 else None
            after_compiler = compiler_metadata(after["waypoint"], compiler) if after["policy_mode"] == 0 else None
            row_compiler_holds = (
                (before_compiler is None or before_compiler["compiled_actions"] == before_proposed)
                and (after_compiler is None or after_compiler["compiled_actions"] == after_proposed)
            )
            compiler_holds = compiler_holds and row_compiler_holds
            delta = feedback_delta(before["executor_feedback"], after["executor_feedback"])
            proposed_flip = before_proposed != after_proposed
            executed_flip = before_executed != after_executed
            boundary_flip = bool(causal and navigate and small and proposed_flip)
            harmful = bool(boundary_flip and (
                delta["environment_reward_sum"] < -1e-6
                or delta["collision_count"] > 0
                or delta["blocked_forward_count"] > 0
            ))
            same_sequence_control = bool(causal and before_executed == after_executed)
            feedback_match = (
                approximately_equal(before["executor_feedback"], after["executor_feedback"], compiler["feedback_equality_tolerance"])
                if same_sequence_control else None
            )
            if same_sequence_control:
                controls_match = controls_match and feedback_match
            record = {
                "episode_id": before_episode["episode_id"], "scene": before_episode["scene"],
                "decision_index": decision_index, "paired_row_present": True,
                "initial_observation_match": identity_match, "seed_schedule_match": schedule_match,
                "all_prior_executed_sequences_equal": prior_equal,
                "state_features_match": before["state_features_sha256"] == after["state_features_sha256"],
                "topology_features_match": before["topology_features_sha256"] == after["topology_features_sha256"],
                "policy_input_match": input_equal, "causal_prefix": causal,
                "descriptive_only": not causal, "state_zero_mode": before["policy_mode"],
                "final_mode": after["policy_mode"], "same_mode": same_mode,
                "same_mode_navigate": navigate, "state_zero_waypoint": before["waypoint"],
                "final_waypoint": after["waypoint"], "waypoint_l2_m": waypoint_l2,
                "small_waypoint_shift": small, "state_zero_compiler": before_compiler,
                "final_compiler": after_compiler, "compiler_contract_holds": row_compiler_holds,
                "state_zero_proposed_actions": before_proposed, "final_proposed_actions": after_proposed,
                "state_zero_executed_actions": before_executed, "final_executed_actions": after_executed,
                "proposed_sequence_flip": proposed_flip, "executed_sequence_flip": executed_flip,
                "causal_boundary_flip": boundary_flip, "feedback_delta": delta,
                "harmful_boundary_flip": harmful, "same_sequence_causal_control": same_sequence_control,
                "same_sequence_feedback_match": feedback_match,
            }
            rows.append(record)
            episode_rows.append(record)
            if executed_flip:
                prior_equal = False
        episodes.append({
            "episode_id": before_episode["episode_id"], "scene": before_episode["scene"],
            "initial_observation_match": identity_match, "seed_schedule_match": schedule_match,
            "causal_row_count": sum(row.get("causal_prefix", False) for row in episode_rows),
            "boundary_flip_count": sum(row.get("causal_boundary_flip", False) for row in episode_rows),
            "harmful_boundary_flip_count": sum(row.get("harmful_boundary_flip", False) for row in episode_rows),
        })
    flips = [row for row in rows if row.get("causal_boundary_flip")]
    harmful = [row for row in flips if row["harmful_boundary_flip"]]
    controls = [row for row in rows if row.get("same_sequence_causal_control")]
    return {
        "episodes": episodes,
        "decision_rows": rows,
        "summary": {
            "paired_episode_count": len(episodes), "decision_row_count": len(rows),
            "causal_prefix_row_count": sum(row.get("causal_prefix", False) for row in rows),
            "descriptive_only_row_count": sum(row.get("descriptive_only", False) for row in rows),
            "same_mode_navigate_causal_row_count": sum(row.get("causal_prefix", False) and row.get("same_mode_navigate", False) for row in rows),
            "small_shift_causal_navigate_row_count": sum(row.get("causal_prefix", False) and row.get("small_waypoint_shift", False) for row in rows),
            "causal_boundary_flip_count": len(flips),
            "causal_boundary_flip_scenes": sorted({row["scene"] for row in flips}),
            "harmful_boundary_flip_count": len(harmful),
            "harmful_boundary_flip_scenes": sorted({row["scene"] for row in harmful}),
            "pooled_harmful_flip_reward_delta": sum(row["feedback_delta"]["environment_reward_sum"] for row in harmful),
            "pooled_harmful_flip_collision_delta": sum(row["feedback_delta"]["collision_count"] for row in harmful),
            "pooled_harmful_flip_blocked_forward_delta": sum(row["feedback_delta"]["blocked_forward_count"] for row in harmful),
            "mode_flip_count_all_rows": sum(row.get("paired_row_present", False) and not row.get("same_mode", True) for row in rows),
            "executed_sequence_flip_count_all_rows": sum(row.get("executed_sequence_flip", False) for row in rows),
            "same_sequence_causal_control_count": len(controls),
            "same_sequence_causal_control_feedback_match": controls_match,
            "compiler_contract_holds": compiler_holds,
        },
    }


def numeric_mapping_close(left: dict, right: dict, tolerance=1e-9) -> bool:
    return all(
        key in left and abs(float(left[key]) - float(value)) <= tolerance
        for key, value in right.items()
    )


def recompute_arm(manifest: dict, run_dir: Path) -> dict:
    paths = {
        "config": run_dir / "run_config.json",
        "decision": run_dir / "reconstructed_decision_collection_records.jsonl",
        "audit": run_dir / "reconstructed_audit_collection_records.jsonl",
        "initial": run_dir / "state_zero_boundary_records.jsonl",
        "final": run_dir / "final_boundary_records.jsonl",
        "result": run_dir / "result.json",
    }
    config = read_json(paths["config"])
    result = read_json(paths["result"])
    decision = read_jsonl(paths["decision"])
    audit = read_jsonl(paths["audit"])
    initial = read_jsonl(paths["initial"])
    final = read_jsonl(paths["final"])
    seed = int(result["seed"])
    seed_key = str(seed)
    protocol = manifest["protocol"]
    g721 = protocol["g721_arms"][seed_key]
    boundary = protocol["boundary_arms"][seed_key]
    anchor = protocol["g721_reconstruction_anchors"][seed_key]
    max_decisions = protocol["max_decisions_per_episode"]
    decision_check = collection_check(decision, g721["decision"], seed=seed, phase=f"g721_seed_{seed}_decision", seed_phase="g721_decision", max_decisions=max_decisions)
    audit_check = collection_check(audit, g721["audit"], seed=seed, phase=f"g721_seed_{seed}_audit", seed_phase="g721_audit", max_decisions=max_decisions)
    initial_check = collection_check(initial, boundary, seed=seed, phase=f"g723_seed_{seed}_state_zero_boundary", seed_phase="g723_boundary", max_decisions=max_decisions, require_hashes=True)
    final_check = collection_check(final, boundary, seed=seed, phase=f"g723_seed_{seed}_final_boundary", seed_phase="g723_boundary", max_decisions=max_decisions, require_hashes=True)
    initial_metrics = behavior_metrics(initial)
    final_metrics = behavior_metrics(final)
    delta = {
        key: final_metrics[key] - initial_metrics[key]
        for key in (
            "decision_count", "primitive_count", "environment_reward_sum", "discounted_chunk_reward_sum",
            "realized_path_m", "collision_count", "blocked_forward_count", "stop_decision_count",
            "recover_decision_count", "navigate_decision_count", "terminated_episode_count", "timeout_episode_count",
        )
    }
    delta["realized_path_retention"] = final_metrics["realized_path_m"] / initial_metrics["realized_path_m"] if initial_metrics["realized_path_m"] > 0 else 1.0
    analysis = analyze_boundary(initial, final, protocol["compiler_contract"])
    candidates = result["candidates"]
    candidate_exact = len(candidates) == len(anchor["candidates"]) and all(
        row["candidate_index"] == expected["candidate_index"]
        and row["accepted"] == expected["accepted"]
        and row["failed_gates"] == expected["failed_gates"]
        and row["training_state_hashes_before"] == expected["training_state_hashes_before"]
        and row["training_state_hashes_candidate"] == expected["training_state_hashes_candidate"]
        and row["reference_probe"] == expected["reference_probe"]
        and all(row["g721_reconstruction"].values())
        and (expected["rejected_continuation_probe"] is None if row["accepted"] else row["continuation_equivalence"]["restored"] == expected["rejected_continuation_probe"])
        for row, expected in zip(candidates, anchor["candidates"])
    )
    paired = result["boundary"]["paired_episodes"]
    paired_raw_exact = len(initial) == len(final) == 8 and all(
        before["episode_id"] == after["episode_id"]
        and before["scene"] == after["scene"]
        and before["initial_observation_sha256"] == after["initial_observation_sha256"]
        and before["sampling_seed_schedule"] == after["sampling_seed_schedule"]
        for before, after in zip(initial, final)
    )
    checks = {
        "manifest_sha": config["manifest_sha256"] == MANIFEST_SHA256 and result["evidence"]["manifest_sha256"] == MANIFEST_SHA256,
        "source_contract": config["source_heads"] == {"stream": manifest["source_contract"]["stream_commit"], "rlinf": manifest["source_contract"]["rlinf_commit"]} and result["source_contract"] == manifest["source_contract"],
        "predecessor_hashes": {int(k): v for k, v in config["predecessor_sha256"].items()} == {int(k): v for k, v in manifest["predecessor"]["result_sha256"].items()},
        "dataset_hashes": all(config["datasets"][split]["sha256"] == manifest["source"][f"{split}_sha256"] for split in config["datasets"]),
        "frozen_specs": config["g721_scene_specs"] == g721 and config["boundary_scene_specs"] == boundary,
        "file_hashes": sha256_file(paths["config"]) == result["evidence"]["run_config_sha256"]
        and sha256_file(paths["decision"]) == result["evidence"]["reconstructed_decision_records_sha256"]
        and sha256_file(paths["audit"]) == result["evidence"]["reconstructed_audit_records_sha256"]
        and sha256_file(paths["initial"]) == result["evidence"]["state_zero_boundary_records_sha256"]
        and sha256_file(paths["final"]) == result["evidence"]["final_boundary_records_sha256"],
        "decision_collection": all(value is True for key, value in decision_check.items() if key not in {"episode_count", "row_count", "scenes"}),
        "audit_collection": all(value is True for key, value in audit_check.items() if key not in {"episode_count", "row_count", "scenes"}),
        "initial_boundary_collection": all(value is True for key, value in initial_check.items() if key not in {"episode_count", "row_count", "scenes"}),
        "final_boundary_collection": all(value is True for key, value in final_check.items() if key not in {"episode_count", "row_count", "scenes"}),
        "g721_baseline_exact": result["baseline_training_state_hashes"] == anchor["baseline_training_state_hashes"],
        "g721_candidates_exact": candidate_exact,
        "g721_final_exact": result["final_training_state_hashes"] == anchor["final_training_state_hashes"],
        "boundary_restores_exact": exact_restore(result["snapshot_restores"]["after_state_zero_boundary"]) and exact_restore(result["snapshot_restores"]["after_final_boundary"]),
        "paired_raw_identity": paired_raw_exact and paired["all_initial_observations_match"] is True and paired["all_seed_schedules_match"] is True,
        "metrics_recomputed": numeric_mapping_close(result["boundary"]["state_zero"]["metrics"], initial_metrics) and numeric_mapping_close(result["boundary"]["final"]["metrics"], final_metrics),
        "delta_recomputed": numeric_mapping_close(result["boundary"]["delta_final_minus_state_zero"], delta),
        "boundary_analysis_recomputed": result["boundary"]["analysis"] == analysis,
        "compiler_contract": analysis["summary"]["compiler_contract_holds"] is True,
        "causal_control_feedback": analysis["summary"]["same_sequence_causal_control_feedback_match"] is True,
        "runtime_guards": all(result["protocol_guards"].values()) and result["all_protocol_guards_passed"] is True,
        "claim_boundary": result["navigation_or_generalization_claim"] is False and result["g8_ready"] is False and result["counts"]["val_unseen_episodes"] == 0,
    }
    return {
        "seed": seed, "run_dir": str(run_dir),
        "file_sha256": {name: sha256_file(path) for name, path in paths.items()},
        "implementation_sha256": config["implementation_sha256"],
        "collection_checks": {"decision": decision_check, "audit": audit_check, "state_zero_boundary": initial_check, "final_boundary": final_check},
        "state_zero_metrics": initial_metrics, "final_metrics": final_metrics,
        "delta_final_minus_state_zero": delta, "boundary_analysis": analysis,
        "protocol_valid": all(result["protocol_guards"].values()),
        "checks": checks, "all_checks_passed": all(checks.values()),
    }


def clustered_scene_summary(rows):
    scenes = defaultdict(lambda: {
        "causal_prefix_rows": 0, "causal_boundary_flips": 0, "harmful_boundary_flips": 0,
        "harmful_reward_delta": 0.0, "harmful_collision_delta": 0, "harmful_blocked_forward_delta": 0,
    })
    for row in rows:
        scene = scenes[row["scene"]]
        scene["causal_prefix_rows"] += int(row.get("causal_prefix", False))
        scene["causal_boundary_flips"] += int(row.get("causal_boundary_flip", False))
        scene["harmful_boundary_flips"] += int(row.get("harmful_boundary_flip", False))
        if row.get("harmful_boundary_flip"):
            scene["harmful_reward_delta"] += row["feedback_delta"]["environment_reward_sum"]
            scene["harmful_collision_delta"] += row["feedback_delta"]["collision_count"]
            scene["harmful_blocked_forward_delta"] += row["feedback_delta"]["blocked_forward_count"]
    return dict(sorted(scenes.items()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-dirs", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if sha256_file(args.manifest) != MANIFEST_SHA256:
        raise SystemExit("manifest hash mismatch")
    manifest = read_json(args.manifest)
    arms = sorted((recompute_arm(manifest, path) for path in args.run_dirs), key=lambda row: row["seed"])
    rows = [row for arm in arms for row in arm["boundary_analysis"]["decision_rows"] if row.get("paired_row_present", False)]
    flips = [row for row in rows if row.get("causal_boundary_flip")]
    harmful = [row for row in flips if row["harmful_boundary_flip"]]
    pooled = {
        "causal_boundary_flip_count": len(flips),
        "causal_boundary_flip_scene_count": len({row["scene"] for row in flips}),
        "causal_boundary_flip_scenes": sorted({row["scene"] for row in flips}),
        "harmful_boundary_flip_count": len(harmful),
        "harmful_boundary_flip_scene_count": len({row["scene"] for row in harmful}),
        "harmful_boundary_flip_scenes": sorted({row["scene"] for row in harmful}),
        "pooled_harmful_flip_reward_delta": sum(row["feedback_delta"]["environment_reward_sum"] for row in harmful),
        "pooled_harmful_flip_collision_delta": sum(row["feedback_delta"]["collision_count"] for row in harmful),
        "pooled_harmful_flip_blocked_forward_delta": sum(row["feedback_delta"]["blocked_forward_count"] for row in harmful),
        "causal_prefix_row_count": sum(row.get("causal_prefix", False) for row in rows),
        "descriptive_only_row_count": sum(row.get("descriptive_only", False) for row in rows),
        "same_sequence_causal_control_count": sum(row.get("same_sequence_causal_control", False) for row in rows),
        "mode_flip_count_all_rows": sum(not row.get("same_mode", True) for row in rows),
        "executed_sequence_flip_count_all_rows": sum(row.get("executed_sequence_flip", False) for row in rows),
    }
    rule = manifest["decision_rule"]
    recurrence_checks = {
        "minimum_flip_count": pooled["causal_boundary_flip_count"] >= rule["minimum_flip_count"],
        "minimum_flip_scene_count": pooled["causal_boundary_flip_scene_count"] >= rule["minimum_flip_scene_count"],
    }
    recurrent = all(recurrence_checks.values())
    harmful_checks = {
        "recurrent_boundary_sensitivity": recurrent,
        "minimum_harmful_flip_count": pooled["harmful_boundary_flip_count"] >= rule["minimum_harmful_flip_count"],
        "minimum_harmful_flip_scene_count": pooled["harmful_boundary_flip_scene_count"] >= rule["minimum_harmful_flip_scene_count"],
        "pooled_harmful_reward_negative": pooled["pooled_harmful_flip_reward_delta"] < 0,
        "pooled_harmful_collision_or_blocked_positive": pooled["pooled_harmful_flip_collision_delta"] > 0 or pooled["pooled_harmful_flip_blocked_forward_delta"] > 0,
    }
    specs = [row for seed in manifest["selection"]["seeds"] for row in manifest["protocol"]["boundary_arms"][str(seed)]]
    scene_counts = Counter((seed, row["scene"]) for seed in manifest["selection"]["seeds"] for row in manifest["protocol"]["boundary_arms"][str(seed)])
    implementations = {json.dumps(arm["implementation_sha256"], sort_keys=True) for arm in arms}
    stage_checks = {
        "exact_seed_set": [arm["seed"] for arm in arms] == manifest["selection"]["seeds"],
        "all_arm_checks": all(arm["all_checks_passed"] for arm in arms),
        "all_protocol_valid": all(arm["protocol_valid"] for arm in arms),
        "same_implementation": len(implementations) == 1,
        "twenty_four_unique_episode_identities": len(specs) == len({row["episode_id"] for row in specs}) == 24,
        "only_five_eligible_scenes": {row["scene"] for row in specs} == set(manifest["selection"]["eligible_new_scenes"]),
        "all_prior_scenes_excluded": not ({row["scene"] for row in specs} & set(manifest["selection"]["excluded_scenes"])),
        "per_seed_scene_cap": max(scene_counts.values()) <= manifest["selection"]["max_episodes_per_scene_per_seed"],
        "same_sequence_controls_match": all(arm["boundary_analysis"]["summary"]["same_sequence_causal_control_feedback_match"] for arm in arms),
        "compiler_contract_holds": all(arm["boundary_analysis"]["summary"]["compiler_contract_holds"] for arm in arms),
    }
    evidence = {
        "kind": "streamvln-g7.23-independent-stdlib-recompute", "manifest_sha256": MANIFEST_SHA256,
        "arms": arms, "pooled_endpoints": pooled, "scene_clustered_endpoints": clustered_scene_summary(rows),
        "recurrence_checks": recurrence_checks, "harmful_recurrence_checks": harmful_checks,
        "stage_checks": stage_checks,
        "recurrent_executor_boundary_sensitivity_supported": all(stage_checks.values()) and recurrent,
        "harmful_executor_boundary_sensitivity_supported": all(stage_checks.values()) and all(harmful_checks.values()),
        "navigation_or_generalization_claim": False, "g8_ready": False,
    }
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output), "output_sha256": sha256_file(args.output),
        "all_checks_passed": all(stage_checks.values()),
        "recurrence_supported": evidence["recurrent_executor_boundary_sensitivity_supported"],
        "harmful_recurrence_supported": evidence["harmful_executor_boundary_sensitivity_supported"],
        "pooled": pooled,
    }, sort_keys=True))
    if not all(stage_checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
