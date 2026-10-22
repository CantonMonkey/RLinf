"""Stdlib-only independent recomputation for G7.21 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


MANIFEST_SHA256 = "daea9d8c8891e24550008069ff1f820ce649f02ab45c6aab45886729da53cdda"


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


def collection_checks(episodes, specs, *, seed: int, role: str, max_decisions: int):
    identities = [
        {
            "index": episode["episode_index"],
            "episode_id": str(episode["episode_id"]),
            "scene": episode["scene"],
        }
        for episode in episodes
    ]
    schedules = []
    rows = []
    phase = f"g721_seed_{seed}_{role}"
    for episode, spec in zip(episodes, specs):
        expected = [
            stable_seed(seed, "g718", f"g721_{role}", spec["episode_id"], index)
            for index in range(max_decisions)
        ]
        schedules.append({
            "episode_id": str(spec["episode_id"]),
            "expected": expected,
            "observed": episode["sampling_seed_schedule"],
        })
        for row in episode["rows"]:
            rows.append(row)
    return {
        "identities_exact": identities == specs,
        "phase_exact": all(
            episode["phase"] == phase and all(row["phase"] == phase for row in episode["rows"])
            for episode in episodes
        ),
        "sampling_schedules_exact": all(
            item["expected"] == item["observed"] for item in schedules
        ),
        "row_seed_and_single_sample_exact": all(
            row["sampling_seed"]
            == stable_seed(seed, "g718", f"g721_{role}", episode["episode_id"], row["decision_index"])
            and row["sample_call_count"] == 1
            and row["recovery_override"] is False
            for episode in episodes
            for row in episode["rows"]
        ),
        "rows": len(rows),
        "episode_count": len(episodes),
        "scenes": [episode["scene"] for episode in episodes],
    }


def recompute_arm(manifest: dict, run_dir: Path) -> dict:
    config_path = run_dir / "run_config.json"
    decision_path = run_dir / "decision_collection_records.jsonl"
    audit_path = run_dir / "audit_collection_records.jsonl"
    result_path = run_dir / "result.json"
    config = read_json(config_path)
    result = read_json(result_path)
    decision = read_jsonl(decision_path)
    audit = read_jsonl(audit_path)
    seed = int(result["seed"])
    seed_key = str(seed)
    protocol = manifest["protocol"]
    arm = protocol["arms"][seed_key]
    decision_check = collection_checks(
        decision, arm["decision"], seed=seed, role="decision",
        max_decisions=protocol["max_decisions_per_episode"],
    )
    audit_check = collection_checks(
        audit, arm["audit"], seed=seed, role="audit",
        max_decisions=protocol["max_decisions_per_episode"],
    )
    candidates = result["candidates"]
    accepted = [bool(row["accepted"]) for row in candidates]
    rejected_indices = [row["candidate_index"] for row in candidates if not row["accepted"]]
    trigger_index = rejected_indices[0] if rejected_indices else None
    decision_gate_exact = all(
        row["accepted"] == all(row["gates"].values())
        and row["failed_gates"]
        == [name for name, passed in row["gates"].items() if not passed]
        for row in candidates
    )
    reference_restores_exact = all(
        exact_restore(row["reference_probe_restore"]) for row in candidates
    )
    rejected_restores_exact = all(
        row["accepted"] or exact_restore(row["restoration"]) for row in candidates
    )
    rejected_continuations_exact = all(
        row["accepted"]
        or (
            row["continuation_equivalence"]["exact"] is True
            and all(row["continuation_equivalence"]["component_equal"].values())
            and row["continuation_equivalence"]["reference"]
            == row["continuation_equivalence"]["restored"]
            and exact_restore(row["probe_cleanup_restore"])
        )
        for row in candidates
    )
    audit_report = result["final_disjoint_audit"]
    recomputed_audit_bounds = {
        "ratio_bounded": audit_report["sampled_action_ratio_max_deviation"]
        <= protocol["audit_contract"]["ratio_max_deviation"],
        "structured_kl_bounded": audit_report["structured_distribution_kl"]["max"]
        <= protocol["audit_contract"]["structured_kl_max"],
    }
    protocol_recomputed = all(result["protocol_guards"].values())
    member_recomputed = protocol_recomputed and all(recomputed_audit_bounds.values())
    expected_predecessor_hashes = set(manifest["predecessor"]["result_sha256"].values())
    observed_predecessor_hashes = set(config["predecessor_sha256"].values())
    checks = {
        "manifest_sha": config["manifest_sha256"] == MANIFEST_SHA256
        and result["evidence"]["manifest_sha256"] == MANIFEST_SHA256,
        "source_contract": config["source_heads"] == {
            "stream": manifest["source_contract"]["stream_commit"],
            "rlinf": manifest["source_contract"]["rlinf_commit"],
        }
        and result["source_contract"] == manifest["source_contract"],
        "predecessor_hash_set": observed_predecessor_hashes == expected_predecessor_hashes,
        "dataset_hashes": all(
            config["datasets"][split]["sha256"] == manifest["source"][f"{split}_sha256"]
            for split in config["datasets"]
        ),
        "frozen_specs": config["decision_scene_specs"] == arm["decision"]
        and config["audit_scene_specs"] == arm["audit"],
        "decision_file_hash": sha256_file(decision_path)
        == result["evidence"]["decision_collection_records_sha256"]
        == result["decision_collection"]["records_sha256"],
        "audit_file_hash": sha256_file(audit_path)
        == result["evidence"]["audit_collection_records_sha256"]
        == result["audit_collection"]["records_sha256"],
        "config_file_hash": sha256_file(config_path)
        == result["evidence"]["run_config_sha256"],
        "decision_id_seed_phase": all(
            value is True for key, value in decision_check.items()
            if key not in {"rows", "episode_count", "scenes"}
        ),
        "audit_id_seed_phase": all(
            value is True for key, value in audit_check.items()
            if key not in {"rows", "episode_count", "scenes"}
        ),
        "role_row_ranges":
        protocol["collection_contract"]["min_rows"] <= decision_check["rows"]
        <= protocol["collection_contract"]["max_rows"]
        and protocol["audit_contract"]["min_rows"] <= audit_check["rows"]
        <= protocol["audit_contract"]["max_rows"],
        "role_scene_disjoint": not set(decision_check["scenes"]) & set(audit_check["scenes"]),
        "candidate_indices_and_limit": [row["candidate_index"] for row in candidates]
        == list(range(1, len(candidates) + 1))
        and len(candidates) <= protocol["candidate_update_limit"],
        "candidate_gate_decisions": decision_gate_exact,
        "stop_after_first_rejection": not any(not value for value in accepted[:-1]),
        "reference_probe_restores": reference_restores_exact,
        "rejected_state_restores": rejected_restores_exact,
        "rejected_continuation_equivalence": rejected_continuations_exact,
        "accepted_count_and_trigger": result["accepted_update_count"] == sum(accepted)
        and result["counts"]["candidate_updates_committed"] == sum(accepted)
        and result["trigger_index"] == trigger_index,
        "accepted_state_history": len(result["accepted_training_state_hashes"])
        == 1 + sum(accepted)
        and result["final_training_state_hashes"]
        == result["accepted_training_state_hashes"][-1],
        "audit_restore_exact": exact_restore(result["audit_restore"])
        and result["accepted_before_audit_training_state_hashes"]
        == result["final_training_state_hashes"],
        "audit_bounds": result["audit_bounds"] == recomputed_audit_bounds,
        "protocol_and_member": result["all_protocol_guards_passed"] == protocol_recomputed
        and result["safety_transfer_member"] == member_recomputed,
        "runtime_protocol_guards": protocol_recomputed,
        "claim_boundary": result["navigation_or_generalization_claim"] is False
        and result["g8_ready"] is False
        and result["counts"]["behavior_episodes"] == 0
        and result["counts"]["val_unseen_episodes"] == 0,
    }
    return {
        "seed": seed,
        "run_dir": str(run_dir),
        "file_sha256": {
            "run_config": sha256_file(config_path),
            "decision_collection": sha256_file(decision_path),
            "audit_collection": sha256_file(audit_path),
            "result": sha256_file(result_path),
        },
        "implementation_sha256": config["implementation_sha256"],
        "decision_collection": decision_check,
        "audit_collection": audit_check,
        "candidate_summary": [
            {
                "index": row["candidate_index"],
                "accepted": row["accepted"],
                "failed_gates": row["failed_gates"],
                "ratio_max_deviation": row["state"]["sampled_action_ratio_max_deviation"],
                "structured_kl_max": row["state"]["structured_distribution_kl"]["max"],
            }
            for row in candidates
        ],
        "trigger_index": trigger_index,
        "accepted_update_count": sum(accepted),
        "audit_report": {
            "ratio_max_deviation": audit_report["sampled_action_ratio_max_deviation"],
            "structured_kl_max": audit_report["structured_distribution_kl"]["max"],
            "bounds": recomputed_audit_bounds,
        },
        "protocol_valid": protocol_recomputed,
        "safety_transfer_member": member_recomputed,
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
    seeds = [arm["seed"] for arm in arms]
    expected_seeds = manifest["selection"]["seeds"]
    implementation_sets = {
        json.dumps(arm["implementation_sha256"], sort_keys=True) for arm in arms
    }
    stage_checks = {
        "exact_seed_set": sorted(seeds) == sorted(expected_seeds) and len(set(seeds)) == 3,
        "all_arm_checks": all(arm["all_checks_passed"] for arm in arms),
        "same_implementation": len(implementation_sets) == 1,
        "global_scene_disjoint": len({
            scene
            for arm in arms
            for role in ("decision_collection", "audit_collection")
            for scene in arm[role]["scenes"]
        }) == 24,
    }
    stage_supported = all(stage_checks.values()) and all(
        arm["safety_transfer_member"] for arm in arms
    )
    evidence = {
        "kind": "streamvln-g7.21-independent-stdlib-recompute",
        "manifest_sha256": MANIFEST_SHA256,
        "arms": sorted(arms, key=lambda arm: arm["seed"]),
        "member_count": sum(arm["safety_transfer_member"] for arm in arms),
        "trigger_indices": {str(arm["seed"]): arm["trigger_index"] for arm in arms},
        "accepted_update_counts": {
            str(arm["seed"]): arm["accepted_update_count"] for arm in arms
        },
        "stage_checks": stage_checks,
        "stage_safety_transfer_supported": stage_supported,
        "g8_ready": False,
    }
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "member_count": evidence["member_count"],
        "stage_safety_transfer_supported": stage_supported,
        "all_checks_passed": all(stage_checks.values()),
    }, sort_keys=True))
    if not all(stage_checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
