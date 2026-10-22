"""Stdlib-only independent recomputation for G7.22 evidence."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


MANIFEST_SHA256 = "3fe63b7afc3f40ed8086c6da6c38496df4a3780360387f324491f8f04751efdd"


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
        {
            "index": episode["episode_index"],
            "episode_id": str(episode["episode_id"]),
            "scene": episode["scene"],
        }
        for episode in episodes
    ]


def collection_check(episodes, specs, *, seed: int, phase: str, seed_phase: str, max_decisions: int):
    return {
        "identities_exact": identities(episodes) == specs,
        "phase_exact": all(
            episode["phase"] == phase
            and all(row["phase"] == phase for row in episode["rows"])
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
            row["sampling_seed"]
            == stable_seed(seed, "g718", seed_phase, episode["episode_id"], row["decision_index"])
            and row["sample_call_count"] == 1
            and row["recovery_override"] is False
            for episode in episodes for row in episode["rows"]
        ),
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


def recompute_arm(manifest: dict, run_dir: Path) -> dict:
    paths = {
        "config": run_dir / "run_config.json",
        "decision": run_dir / "reconstructed_decision_collection_records.jsonl",
        "audit": run_dir / "reconstructed_audit_collection_records.jsonl",
        "initial": run_dir / "state_zero_behavior_records.jsonl",
        "final": run_dir / "final_behavior_records.jsonl",
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
    behavior = protocol["behavior_arms"][seed_key]
    anchor = protocol["g721_reconstruction_anchors"][seed_key]
    max_decisions = protocol["max_decisions_per_episode"]
    decision_check = collection_check(
        decision, g721["decision"], seed=seed,
        phase=f"g721_seed_{seed}_decision", seed_phase="g721_decision",
        max_decisions=max_decisions,
    )
    audit_check = collection_check(
        audit, g721["audit"], seed=seed,
        phase=f"g721_seed_{seed}_audit", seed_phase="g721_audit",
        max_decisions=max_decisions,
    )
    initial_check = collection_check(
        initial, behavior, seed=seed,
        phase=f"g722_seed_{seed}_state_zero_behavior", seed_phase="g722_behavior",
        max_decisions=max_decisions,
    )
    final_check = collection_check(
        final, behavior, seed=seed,
        phase=f"g722_seed_{seed}_final_behavior", seed_phase="g722_behavior",
        max_decisions=max_decisions,
    )
    initial_metrics = behavior_metrics(initial)
    final_metrics = behavior_metrics(final)
    delta_keys = (
        "decision_count", "primitive_count", "environment_reward_sum",
        "discounted_chunk_reward_sum", "realized_path_m", "collision_count",
        "blocked_forward_count", "stop_decision_count", "recover_decision_count",
        "navigate_decision_count", "terminated_episode_count", "timeout_episode_count",
    )
    delta = {key: final_metrics[key] - initial_metrics[key] for key in delta_keys}
    path_retention = (
        final_metrics["realized_path_m"] / initial_metrics["realized_path_m"]
        if initial_metrics["realized_path_m"] > 0 else 1.0
    )
    delta["realized_path_retention"] = path_retention
    candidates = result["candidates"]
    candidate_exact = len(candidates) == len(anchor["candidates"]) and all(
        row["candidate_index"] == expected["candidate_index"]
        and row["accepted"] == expected["accepted"]
        and row["failed_gates"] == expected["failed_gates"]
        and row["training_state_hashes_before"] == expected["training_state_hashes_before"]
        and row["training_state_hashes_candidate"] == expected["training_state_hashes_candidate"]
        and row["reference_probe"] == expected["reference_probe"]
        and all(row["g721_reconstruction"].values())
        and (
            expected["rejected_continuation_probe"] is None
            if row["accepted"]
            else row["continuation_equivalence"]["restored"]
            == expected["rejected_continuation_probe"]
        )
        for row, expected in zip(candidates, anchor["candidates"])
    )
    paired = result["behavior"]["paired_episodes"]
    paired_raw_exact = len(initial) == len(final) == 4 and all(
        before["episode_id"] == after["episode_id"]
        and before["scene"] == after["scene"]
        and before["initial_observation_sha256"] == after["initial_observation_sha256"]
        and before["sampling_seed_schedule"] == after["sampling_seed_schedule"]
        for before, after in zip(initial, final)
    )
    checks = {
        "manifest_sha": config["manifest_sha256"] == MANIFEST_SHA256
        and result["evidence"]["manifest_sha256"] == MANIFEST_SHA256,
        "source_contract": config["source_heads"] == {
            "stream": manifest["source_contract"]["stream_commit"],
            "rlinf": manifest["source_contract"]["rlinf_commit"],
        } and result["source_contract"] == manifest["source_contract"],
        "predecessor_hashes": {
            int(k): v for k, v in config["predecessor_sha256"].items()
        } == {int(k): v for k, v in manifest["predecessor"]["result_sha256"].items()},
        "dataset_hashes": all(
            config["datasets"][split]["sha256"] == manifest["source"][f"{split}_sha256"]
            for split in config["datasets"]
        ),
        "frozen_specs": config["g721_scene_specs"] == g721
        and config["behavior_scene_specs"] == behavior,
        "file_hashes": sha256_file(paths["config"]) == result["evidence"]["run_config_sha256"]
        and sha256_file(paths["decision"]) == result["evidence"]["reconstructed_decision_records_sha256"]
        and sha256_file(paths["audit"]) == result["evidence"]["reconstructed_audit_records_sha256"]
        and sha256_file(paths["initial"]) == result["evidence"]["state_zero_behavior_records_sha256"]
        and sha256_file(paths["final"]) == result["evidence"]["final_behavior_records_sha256"],
        "decision_collection": all(
            value is True for key, value in decision_check.items()
            if key not in {"episode_count", "row_count", "scenes"}
        ),
        "audit_collection": all(
            value is True for key, value in audit_check.items()
            if key not in {"episode_count", "row_count", "scenes"}
        ),
        "initial_behavior_collection": all(
            value is True for key, value in initial_check.items()
            if key not in {"episode_count", "row_count", "scenes"}
        ),
        "final_behavior_collection": all(
            value is True for key, value in final_check.items()
            if key not in {"episode_count", "row_count", "scenes"}
        ),
        "g721_baseline_exact": result["baseline_training_state_hashes"]
        == anchor["baseline_training_state_hashes"],
        "g721_candidates_exact": candidate_exact,
        "g721_final_exact": result["final_training_state_hashes"]
        == anchor["final_training_state_hashes"],
        "behavior_restores_exact": exact_restore(result["snapshot_restores"]["after_state_zero_behavior"])
        and exact_restore(result["snapshot_restores"]["after_final_behavior"]),
        "paired_raw_identity": paired_raw_exact
        and paired["all_initial_observations_match"] is True
        and paired["all_seed_schedules_match"] is True,
        "metrics_recomputed": all(
            abs(result["behavior"]["state_zero"]["metrics"][key] - value) <= 1e-9
            for key, value in initial_metrics.items()
        ) and all(
            abs(result["behavior"]["final"]["metrics"][key] - value) <= 1e-9
            for key, value in final_metrics.items()
        ),
        "delta_recomputed": all(
            abs(result["behavior"]["delta_final_minus_state_zero"][key] - value) <= 1e-9
            for key, value in delta.items()
        ),
        "runtime_guards": all(result["protocol_guards"].values())
        and result["all_protocol_guards_passed"] is True,
        "claim_boundary": result["navigation_or_generalization_claim"] is False
        and result["g8_ready"] is False and result["counts"]["val_unseen_episodes"] == 0,
    }
    return {
        "seed": seed,
        "run_dir": str(run_dir),
        "file_sha256": {name: sha256_file(path) for name, path in paths.items()},
        "implementation_sha256": config["implementation_sha256"],
        "collection_checks": {
            "decision": decision_check, "audit": audit_check,
            "state_zero_behavior": initial_check, "final_behavior": final_check,
        },
        "state_zero_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "delta_final_minus_state_zero": delta,
        "reward_delta_nonnegative": delta["environment_reward_sum"] >= 0,
        "protocol_valid": all(result["protocol_guards"].values()),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-dirs", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if sha256_file(args.manifest) != MANIFEST_SHA256:
        raise SystemExit("manifest hash mismatch")
    manifest = read_json(args.manifest)
    arms = [recompute_arm(manifest, path) for path in args.run_dirs]
    arms.sort(key=lambda row: row["seed"])
    initial_reward = sum(arm["state_zero_metrics"]["environment_reward_sum"] for arm in arms)
    final_reward = sum(arm["final_metrics"]["environment_reward_sum"] for arm in arms)
    initial_path = sum(arm["state_zero_metrics"]["realized_path_m"] for arm in arms)
    final_path = sum(arm["final_metrics"]["realized_path_m"] for arm in arms)
    pooled = {
        "environment_reward_delta": final_reward - initial_reward,
        "collision_delta": sum(arm["delta_final_minus_state_zero"]["collision_count"] for arm in arms),
        "blocked_forward_delta": sum(arm["delta_final_minus_state_zero"]["blocked_forward_count"] for arm in arms),
        "realized_path_retention": final_path / initial_path if initial_path > 0 else 1.0,
        "nonnegative_seed_reward_delta_count": sum(arm["reward_delta_nonnegative"] for arm in arms),
    }
    rule = manifest["decision_rule"]
    viability_checks = {
        "pooled_reward_nonnegative": pooled["environment_reward_delta"]
        >= rule["pooled_environment_reward_delta_min"],
        "seed_reward_majority_nonnegative": pooled["nonnegative_seed_reward_delta_count"]
        >= rule["minimum_nonnegative_seed_reward_deltas"],
        "pooled_collisions_nonincreasing": pooled["collision_delta"]
        <= rule["pooled_collision_delta_max"],
        "pooled_blocked_forward_nonincreasing": pooled["blocked_forward_delta"]
        <= rule["pooled_blocked_forward_delta_max"],
        "pooled_path_retained": pooled["realized_path_retention"]
        >= rule["pooled_realized_path_retention_min"],
    }
    implementations = {json.dumps(arm["implementation_sha256"], sort_keys=True) for arm in arms}
    behavior_scenes = [
        scene for arm in arms for scene in arm["collection_checks"]["state_zero_behavior"]["scenes"]
    ]
    stage_checks = {
        "exact_seed_set": [arm["seed"] for arm in arms] == manifest["selection"]["seeds"],
        "all_arm_checks": all(arm["all_checks_passed"] for arm in arms),
        "all_protocol_valid": all(arm["protocol_valid"] for arm in arms),
        "same_implementation": len(implementations) == 1,
        "global_behavior_scene_disjoint": len(behavior_scenes) == len(set(behavior_scenes)) == 12,
    }
    supported = all(stage_checks.values()) and all(viability_checks.values())
    evidence = {
        "kind": "streamvln-g7.22-independent-stdlib-recompute",
        "manifest_sha256": MANIFEST_SHA256,
        "arms": arms,
        "pooled_endpoints": pooled,
        "viability_checks": viability_checks,
        "stage_checks": stage_checks,
        "short_horizon_train_behavioral_viability_supported": supported,
        "navigation_or_generalization_claim": False,
        "g8_ready": False,
    }
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "all_checks_passed": all(stage_checks.values()),
        "viability_supported": supported,
        "pooled": pooled,
    }, sort_keys=True))
    if not all(stage_checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
