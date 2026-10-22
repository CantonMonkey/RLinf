"""Run one preregistered G7.22 fixed-checkpoint matched train-behavior arm."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import time

import numpy as np

from run_streamvln_atomic_guard_g720 import candidate_gates, evaluate_state, one_update, prepare_batch
from run_streamvln_learning_g718 import (
    ORIGINAL_STREAM_COMMIT,
    RLINF_COMMIT,
    aggregate_behavior,
    collect_phase,
    flat_rows,
    paired_behavior,
    write_jsonl,
)
from run_streamvln_plumbing_g717 import (
    dataset_hashes,
    frozen_parameter_sentinels,
    git_head,
    optimizer_parameter_scope,
    sha256_file,
    small_file_inventory,
    write_json,
)
from run_streamvln_recovery_g6 import setup_manager
from run_streamvln_safety_transfer_g721 import (
    collection_semantics,
    compare_probes,
    fixed_continuation_probe,
    prepare_evaluation_batch,
)
from run_streamvln_stability_g719_v2 import optimizer_state_contract
from run_streamvln_topology_g5 import cleanup_manager, parameter_snapshot, state_sha256
from streamvln_atomic_state import capture_training_state, restore_training_state, training_state_hashes
from streamvln_executor_contract import PRIVILEGED_POLICY_KEYS, assert_history_only_keys


MANIFEST_SHA256 = "3fe63b7afc3f40ed8086c6da6c38496df4a3780360387f324491f8f04751efdd"
G721_LOCAL_COMMIT = "4c4f139cafef4d3165402779e6bd0f392b1d90ba"
G721_MANIFEST_SHA256 = "daea9d8c8891e24550008069ff1f820ce649f02ab45c6aab45886729da53cdda"
G721_RECOMPUTE_SHA256 = "b740edc45e21f788f95a85d45229a8538d0b345c858415bdfa739b8dab20913b"
G721_RESULT_SHA256 = {
    2026092811: "2036cca5ef43eb405d95ec3d45585e528fc8f6690fd26e46684f901b52b9fc50",
    2026092812: "27e17467ae164796a8a1b494517337182b86d8b01dd8c57f644c614a777566eb",
    2026092813: "4341ad6bda4765919a4170c632af3a4a4ac7f15b2d6c385c462596a8eb4894b9",
}
FROZEN_BEHAVIOR_ARMS = {
    "2026092811": [
        {"index": 5792, "episode_id": "5802", "scene": "JeFG25nYj2p"},
        {"index": 6617, "episode_id": "6627", "scene": "5LpN3gDmAk7"},
        {"index": 9746, "episode_id": "9765", "scene": "Vvot9Ly1tCj"},
        {"index": 10507, "episode_id": "10526", "scene": "1LXtFkjw3qL"},
    ],
    "2026092812": [
        {"index": 7301, "episode_id": "7311", "scene": "jh4fc5c5qoQ"},
        {"index": 4466, "episode_id": "4476", "scene": "s8pcmisQ38h"},
        {"index": 1983, "episode_id": "1987", "scene": "SN83YJsR3w2"},
        {"index": 614, "episode_id": "615", "scene": "8WUmhLawc2A"},
    ],
    "2026092813": [
        {"index": 4127, "episode_id": "4137", "scene": "VLzqgDo317F"},
        {"index": 4345, "episode_id": "4355", "scene": "29hnd4uzFmX"},
        {"index": 9435, "episode_id": "9451", "scene": "XcA2TqTSSAj"},
        {"index": 10546, "episode_id": "10565", "scene": "Pm6F8kyY3z2"},
    ],
}


def validate_manifest(manifest: dict) -> dict:
    expected = {
        "schema_version", "stage", "source_contract", "source", "predecessor",
        "selection", "policy_observation_keys", "privileged_policy_blacklist",
        "protocol", "decision_rule", "required_reports", "model_load", "rl_update",
        "navigation_evaluation", "controller_comparison", "robot_execution", "g8_ready",
    }
    if set(manifest) != expected or manifest["schema_version"] != 1:
        raise ValueError("G7.22 manifest schema changed")
    if manifest["source_contract"] != {
        "stream_commit": ORIGINAL_STREAM_COMMIT,
        "rlinf_commit": RLINF_COMMIT,
        "original_stream_only": True,
        "ttt_or_secondary_post_training": False,
    }:
        raise ValueError("G7.22 source contract changed")
    predecessor = manifest["predecessor"]
    if (
        predecessor["accepted_local_commit"] != G721_LOCAL_COMMIT
        or predecessor["manifest_sha256"] != G721_MANIFEST_SHA256
        or {int(k): v for k, v in predecessor["result_sha256"].items()} != G721_RESULT_SHA256
        or predecessor["evidence_recompute_sha256"] != G721_RECOMPUTE_SHA256
    ):
        raise ValueError("G7.22 predecessor anchor changed")
    assert_history_only_keys(manifest["policy_observation_keys"])
    if set(manifest["privileged_policy_blacklist"]) != PRIVILEGED_POLICY_KEYS:
        raise ValueError("G7.22 privileged-data blacklist changed")
    protocol = manifest["protocol"]
    selection = manifest["selection"]
    all_behavior = [row for rows in FROZEN_BEHAVIOR_ARMS.values() for row in rows]
    g721_scenes = {
        row["scene"]
        for arm in protocol["g721_arms"].values()
        for role in ("decision", "audit")
        for row in arm[role]
    }
    if (
        protocol["behavior_arms"] != FROZEN_BEHAVIOR_ARMS
        or selection["seeds"] != [int(seed) for seed in FROZEN_BEHAVIOR_ARMS]
        or selection["episodes_per_seed"] != 4
        or selection["outcome_or_policy_fields_used"] is not False
        or selection["global_scene_disjoint"] is not True
        or selection["replacement_after_collection"] is not False
        or len({row["scene"] for row in all_behavior}) != 12
        or {row["scene"] for row in all_behavior} & g721_scenes
        or {row["scene"] for row in all_behavior} & set(selection["excluded_scenes"])
        or protocol["split"] != "train"
        or protocol["sampling"]["sample_calls_per_decision"] != 1
        or protocol["sampling"]["target_mode_scan"] is not False
        or protocol["sampling"]["mode_filter_or_resample"] is not False
        or protocol["sampling"]["recovery_override"] is not False
        or protocol["max_decisions_per_episode"] != 3
        or protocol["max_primitives_per_episode"] != 12
        or protocol["candidate_update_limit"] != 3
        or protocol["stop_candidate_loop_after_first_rejection"] is not True
        or protocol["state_zero_and_final_must_exactly_match_g721"] is not True
        or protocol["behavior_is_nonselective"] is not True
        or protocol["checkpoint_selection"] is not False
        or protocol["val_unseen_episode_count"] != 0
    ):
        raise ValueError("G7.22 protocol changed")
    rule = manifest["decision_rule"]
    if (
        rule["pooled_environment_reward_delta_min"] != 0.0
        or rule["minimum_nonnegative_seed_reward_deltas"] != 2
        or rule["pooled_collision_delta_max"] != 0
        or rule["pooled_blocked_forward_delta_max"] != 0
        or rule["pooled_realized_path_retention_min"] != 0.8
        or rule["posthoc_selection_or_rerun"] is not False
    ):
        raise ValueError("G7.22 decision rule changed")
    if any(manifest[key] is not False for key in (
        "navigation_evaluation", "controller_comparison", "robot_execution", "g8_ready"
    )) or manifest["model_load"] is not True or manifest["rl_update"] is not True:
        raise ValueError("G7.22 claim boundary changed")
    return protocol


def exact_restore(evidence: dict) -> bool:
    return bool(
        evidence
        and evidence["exact"] is True
        and evidence["before"] == evidence["after"]
        and all(evidence["component_equal"].values())
    )


def behavior_delta(initial: dict, final: dict) -> dict:
    keys = (
        "decision_count", "primitive_count", "environment_reward_sum",
        "discounted_chunk_reward_sum", "realized_path_m", "collision_count",
        "blocked_forward_count", "stop_decision_count", "recover_decision_count",
        "navigate_decision_count", "terminated_episode_count", "timeout_episode_count",
    )
    delta = {key: final[key] - initial[key] for key in keys}
    delta["realized_path_retention"] = (
        final["realized_path_m"] / initial["realized_path_m"]
        if initial["realized_path_m"] > 0 else 1.0
    )
    return delta


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--g721-results", type=Path, nargs=3, required=True)
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
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--master-port", type=int, required=True)
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
        raise RuntimeError("G7.22 frozen manifest hash changed")
    manifest = json.loads(manifest_path.read_text())
    protocol = validate_manifest(manifest)
    seed_key = str(args.seed)
    if seed_key not in FROZEN_BEHAVIOR_ARMS:
        raise RuntimeError("G7.22 seed is not preregistered")
    g721_specs = protocol["g721_arms"][seed_key]
    behavior_specs = protocol["behavior_arms"][seed_key]
    anchor = protocol["g721_reconstruction_anchors"][seed_key]

    predecessor_results = {}
    for path in [item.resolve(strict=True) for item in args.g721_results]:
        result = json.loads(path.read_text())
        seed = int(result["seed"])
        if seed not in G721_RESULT_SHA256 or sha256_file(path) != G721_RESULT_SHA256[seed]:
            raise RuntimeError("G7.22 G7.21 result anchor changed")
        if result["all_protocol_guards_passed"] is not True or result["safety_transfer_member"] is not True:
            raise RuntimeError("G7.22 predecessor protocol is invalid")
        predecessor_results[seed] = str(path)
    if set(predecessor_results) != set(G721_RESULT_SHA256):
        raise RuntimeError("G7.22 predecessor result set changed")

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
        raise RuntimeError("G7.22 source commit changed")
    datasets = dataset_hashes(Path(paths["asset_root"]), manifest)
    script_root = Path(__file__).resolve().parent
    implementation_names = (
        "run_streamvln_matched_behavior_g722.py", "run_streamvln_safety_transfer_g721.py",
        "run_streamvln_atomic_guard_g720.py", "streamvln_atomic_state.py",
        "run_streamvln_learning_g718.py", "run_streamvln_plumbing_g717.py",
        "streamvln_topological_policy.py", "streamvln_recovery_policy.py",
        "streamvln_executor_contract.py", "streamvln_topology.py", "streamvln_chunk_ppo.py",
    )
    implementation_paths = {name: script_root / name for name in implementation_names}
    implementation_paths.update({
        "rlinf/envs/streamvln/streamvln_env.py": Path(paths["rlinf_root"]) / "rlinf/envs/streamvln/streamvln_env.py",
        "rlinf/envs/streamvln/streamvln_executor_contract.py": Path(paths["rlinf_root"]) / "rlinf/envs/streamvln/streamvln_executor_contract.py",
        "rlinf/models/embodiment/streamvln_ppo_policy.py": Path(paths["rlinf_root"]) / "rlinf/models/embodiment/streamvln_ppo_policy.py",
    })
    implementation = {
        name: sha256_file(path.resolve(strict=True)) for name, path in implementation_paths.items()
    }
    run_cfg = {
        **paths,
        "output_dir": str(output),
        "stream_commit": ORIGINAL_STREAM_COMMIT,
        "rlinf_commit": RLINF_COMMIT,
        "manifest": {
            "source": manifest["source"],
            "train": g721_specs["decision"] + g721_specs["audit"] + behavior_specs,
            "val_unseen": [],
        },
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
        "kind": "streamvln-g7.22-fixed-checkpoint-matched-train-behavior",
        "manifest": str(manifest_path),
        "manifest_sha256": MANIFEST_SHA256,
        "predecessor_results": predecessor_results,
        "predecessor_sha256": G721_RESULT_SHA256,
        "source_heads": source_heads,
        "datasets": datasets,
        "implementation_sha256": implementation,
        "model_input_inventory": {
            "base_model": small_file_inventory(Path(paths["base_model"])),
            "tokenizer": small_file_inventory(Path(paths["tokenizer"])),
            "vision_tower": small_file_inventory(Path(paths["vision_tower"])),
        },
        "run": run_cfg,
        "seed": args.seed,
        "g721_scene_specs": g721_specs,
        "behavior_scene_specs": behavior_specs,
        "model_load_count": 1,
        "checkpoint_selection": False,
        "outcome_based_selection": False,
        "val_unseen_episode_count": 0,
    }
    config_path = output / "run_config.json"
    write_json(config_path, config)

    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    manager = core = None
    try:
        manager, core, trainable_manifest = setup_manager(
            run_cfg, total_steps=protocol["candidate_update_limit"]
        )
        optimizer_scope = optimizer_parameter_scope(manager, core)
        optimizer_initial = optimizer_state_contract(manager.optimizer)
        if optimizer_initial != protocol["optimizer_initial_contract"]:
            raise RuntimeError("G7.22 initial optimizer differs from eager-zero contract")
        initial_parameters = parameter_snapshot(core)
        initial_parameter_hash = state_sha256(initial_parameters)
        frozen_initial, frozen_names = frozen_parameter_sentinels(core)
        frozen_initial_hash = state_sha256(frozen_initial)

        decision_collection = collect_phase(
            manager, core, run_cfg, g721_specs["decision"],
            phase=f"g721_seed_{args.seed}_decision", seed_phase="g721_decision",
        )
        audit_collection = collect_phase(
            manager, core, run_cfg, g721_specs["audit"],
            phase=f"g721_seed_{args.seed}_audit", seed_phase="g721_audit",
        )
        decision_path = output / "reconstructed_decision_collection_records.jsonl"
        audit_path = output / "reconstructed_audit_collection_records.jsonl"
        write_jsonl(decision_path, decision_collection)
        write_jsonl(audit_path, audit_collection)
        if state_sha256(parameter_snapshot(core)) != initial_parameter_hash:
            raise RuntimeError("G7.22 G7.21 collection changed parameters")
        decision_batch = prepare_batch(manager, core, run_cfg, decision_collection)
        audit_batch = prepare_evaluation_batch(
            manager, core, audit_collection, protocol["audit_contract"], "audit"
        )
        if (
            decision_batch["preupdate_ratio_max_deviation"] > protocol["preupdate_ratio_max_deviation"]
            or audit_batch["preupdate_ratio_max_deviation"] > protocol["preupdate_ratio_max_deviation"]
        ):
            raise RuntimeError("G7.22 reconstructed state-zero ratio differs from one")
        state_zero = evaluate_state(
            manager, core, decision_batch, initial_parameters, "decision_state_zero"
        )
        baseline_snapshot = capture_training_state(manager, core)
        baseline_hashes = training_state_hashes(baseline_snapshot)
        if baseline_hashes != anchor["baseline_training_state_hashes"]:
            raise RuntimeError("G7.22 baseline state does not exactly reconstruct G7.21")

        behavior_initial = collect_phase(
            manager, core, run_cfg, behavior_specs,
            phase=f"g722_seed_{args.seed}_state_zero_behavior", seed_phase="g722_behavior",
        )
        behavior_initial_path = output / "state_zero_behavior_records.jsonl"
        write_jsonl(behavior_initial_path, behavior_initial)
        initial_behavior_restore = restore_training_state(manager, core, baseline_snapshot)
        if not initial_behavior_restore["exact"]:
            raise RuntimeError("G7.22 state-zero behavior did not restore exactly")

        previous_report = state_zero
        accepted_snapshot = baseline_snapshot
        candidates = []
        for candidate_index in range(1, protocol["candidate_update_limit"] + 1):
            before_snapshot = capture_training_state(manager, core)
            before_hashes = training_state_hashes(before_snapshot)
            reference_probe = fixed_continuation_probe(manager, decision_batch)
            reference_probe_restore = restore_training_state(manager, core, before_snapshot)
            if not reference_probe_restore["exact"]:
                raise RuntimeError("G7.22 reference probe did not restore exactly")
            before_parameters = parameter_snapshot(core)
            frozen_before, names_before = frozen_parameter_sentinels(core)
            if names_before != frozen_names:
                raise RuntimeError("G7.22 frozen sentinel schema changed")
            update = one_update(manager, core, run_cfg, decision_batch, candidate_index)
            report = evaluate_state(
                manager, core, decision_batch, before_parameters, f"candidate_{candidate_index}"
            )
            frozen_after, names_after = frozen_parameter_sentinels(core)
            if names_after != frozen_names:
                raise RuntimeError("G7.22 frozen sentinel schema changed after candidate")
            after_snapshot = capture_training_state(manager, core)
            after_hashes = training_state_hashes(after_snapshot)
            gates = candidate_gates(
                report, previous_report, update, protocol,
                actor_version_before=before_snapshot["actor_version"],
                optimizer_steps_before=before_snapshot["optimizer_steps"],
                frozen_before=state_sha256(frozen_before),
                frozen_after=state_sha256(frozen_after),
                input_sha256=decision_batch["input_sha256"],
            )
            accepted = all(gates.values())
            restoration = continuation = probe_cleanup_restore = None
            if accepted:
                accepted_snapshot = after_snapshot
                previous_report = report
                disposition = "committed_all_decision_gates_passed"
            else:
                restoration = restore_training_state(manager, core, before_snapshot)
                if not restoration["exact"]:
                    raise RuntimeError("G7.22 rejected candidate did not restore exactly")
                restored_probe = fixed_continuation_probe(manager, decision_batch)
                continuation = compare_probes(reference_probe, restored_probe)
                if not continuation["exact"]:
                    raise RuntimeError("G7.22 rejected candidate changed continuation")
                probe_cleanup_restore = restore_training_state(manager, core, before_snapshot)
                if not probe_cleanup_restore["exact"]:
                    raise RuntimeError("G7.22 continuation cleanup was not exact")
                accepted_snapshot = before_snapshot
                disposition = "rejected_restored_continuation_equal"
            row = {
                "candidate_index": candidate_index,
                "accepted": accepted,
                "disposition": disposition,
                "failed_gates": [name for name, passed in gates.items() if not passed],
                "gates": gates,
                "update": update,
                "state": report,
                "training_state_hashes_before": before_hashes,
                "training_state_hashes_candidate": after_hashes,
                "reference_probe": reference_probe,
                "reference_probe_restore": reference_probe_restore,
                "restoration": restoration,
                "continuation_equivalence": continuation,
                "probe_cleanup_restore": probe_cleanup_restore,
            }
            expected = anchor["candidates"][candidate_index - 1]
            reconstruction = {
                "candidate_index": candidate_index == expected["candidate_index"],
                "accepted": accepted == expected["accepted"],
                "failed_gates": row["failed_gates"] == expected["failed_gates"],
                "before_state": before_hashes == expected["training_state_hashes_before"],
                "candidate_state": after_hashes == expected["training_state_hashes_candidate"],
                "reference_probe": reference_probe == expected["reference_probe"],
                "rejected_continuation_probe": (
                    expected["rejected_continuation_probe"] is None
                    if accepted else continuation["restored"] == expected["rejected_continuation_probe"]
                ),
            }
            row["g721_reconstruction"] = reconstruction
            if not all(reconstruction.values()):
                raise RuntimeError(f"G7.22 candidate {candidate_index} differs from G7.21")
            candidates.append(row)
            if not accepted:
                break

        final_snapshot = capture_training_state(manager, core)
        final_hashes = training_state_hashes(final_snapshot)
        if final_hashes != anchor["final_training_state_hashes"]:
            raise RuntimeError("G7.22 final state does not exactly reconstruct G7.21")
        if final_hashes != training_state_hashes(accepted_snapshot):
            raise RuntimeError("G7.22 final state differs from last accepted state")

        behavior_final = collect_phase(
            manager, core, run_cfg, behavior_specs,
            phase=f"g722_seed_{args.seed}_final_behavior", seed_phase="g722_behavior",
        )
        behavior_final_path = output / "final_behavior_records.jsonl"
        write_jsonl(behavior_final_path, behavior_final)
        final_behavior_restore = restore_training_state(manager, core, final_snapshot)
        if not final_behavior_restore["exact"]:
            raise RuntimeError("G7.22 final behavior did not restore exactly")
        final_after_behavior_hashes = training_state_hashes(capture_training_state(manager, core))
        frozen_final, final_names = frozen_parameter_sentinels(core)

        initial_metrics = aggregate_behavior(behavior_initial)
        final_metrics = aggregate_behavior(behavior_final)
        paired = paired_behavior(behavior_initial, behavior_final)
        delta = behavior_delta(initial_metrics, final_metrics)
        guards = {
            "source_and_predecessors_anchored": source_heads == {
                "stream": ORIGINAL_STREAM_COMMIT, "rlinf": RLINF_COMMIT
            } and set(predecessor_results) == set(G721_RESULT_SHA256),
            "dataset_hashes_fixed": all(
                datasets[split]["sha256"] == manifest["source"][f"{split}_sha256"]
                for split in datasets
            ),
            "preregistered_seed_and_scene_specs": behavior_specs == FROZEN_BEHAVIOR_ARMS[seed_key],
            "all_three_roles_scene_disjoint": len({
                row["scene"] for role in ("decision", "audit") for row in g721_specs[role]
            } | {row["scene"] for row in behavior_specs}) == 12,
            "ordinary_reconstruction_collection_semantics": collection_semantics(decision_collection)
            and collection_semantics(audit_collection),
            "ordinary_behavior_collection_semantics": collection_semantics(behavior_initial)
            and collection_semantics(behavior_final),
            "initial_optimizer_eager_zero_contract": optimizer_initial == protocol["optimizer_initial_contract"],
            "optimizer_scope_is_high_level_only": optimizer_scope["optimizer_equals_trainable"]
            and optimizer_scope["optimizer_frozen_overlap_count"] == 0,
            "baseline_exactly_reconstructed": baseline_hashes == anchor["baseline_training_state_hashes"],
            "every_candidate_exactly_reconstructed": all(
                all(row["g721_reconstruction"].values()) for row in candidates
            ) and len(candidates) == len(anchor["candidates"]),
            "final_exactly_reconstructed": final_hashes == anchor["final_training_state_hashes"],
            "behavior_snapshots_restored_exactly": exact_restore(initial_behavior_restore)
            and exact_restore(final_behavior_restore)
            and final_after_behavior_hashes == final_hashes,
            "paired_identity_observation_and_seed_schedule": paired["all_initial_observations_match"]
            and paired["all_seed_schedules_match"],
            "frozen_parameters_unchanged": final_names == frozen_names
            and state_sha256(frozen_final) == frozen_initial_hash,
            "behavior_nonselective": protocol["behavior_is_nonselective"]
            and config["checkpoint_selection"] is False and config["outcome_based_selection"] is False,
            "zero_val_unseen": protocol["val_unseen_episode_count"] == 0,
        }
        protocol_valid = all(guards.values())
        result = {
            "kind": config["kind"],
            "seed": args.seed,
            "all_protocol_guards_passed": protocol_valid,
            "outcome": "matched_train_behavior_measured" if protocol_valid else "protocol_invalid",
            "navigation_or_generalization_claim": False,
            "source_contract": manifest["source_contract"],
            "trainable_manifest": trainable_manifest,
            "optimizer_scope": optimizer_scope,
            "optimizer_initial_contract": optimizer_initial,
            "counts": {
                "model_loads": 1,
                "reconstruction_decision_episodes": len(decision_collection),
                "reconstruction_audit_episodes": len(audit_collection),
                "candidate_updates_attempted": len(candidates),
                "candidate_updates_committed": sum(row["accepted"] for row in candidates),
                "state_zero_behavior_episodes": len(behavior_initial),
                "final_behavior_episodes": len(behavior_final),
                "val_unseen_episodes": 0,
            },
            "initial_high_level_sha256": initial_parameter_hash,
            "decision_state_zero": state_zero,
            "baseline_training_state_hashes": baseline_hashes,
            "candidates": candidates,
            "final_training_state_hashes": final_hashes,
            "behavior": {
                "state_zero": {
                    "records_sha256": sha256_file(behavior_initial_path),
                    "metrics": initial_metrics,
                },
                "final": {
                    "records_sha256": sha256_file(behavior_final_path),
                    "metrics": final_metrics,
                },
                "delta_final_minus_state_zero": delta,
                "paired_episodes": paired,
                "seed_reward_delta_nonnegative": delta["environment_reward_sum"] >= 0,
            },
            "snapshot_restores": {
                "after_state_zero_behavior": initial_behavior_restore,
                "after_final_behavior": final_behavior_restore,
            },
            "protocol_guards": guards,
            "g8_ready": False,
            "elapsed_seconds": time.time() - started,
            "evidence": {
                "manifest_sha256": MANIFEST_SHA256,
                "run_config_sha256": sha256_file(config_path),
                "reconstructed_decision_records_sha256": sha256_file(decision_path),
                "reconstructed_audit_records_sha256": sha256_file(audit_path),
                "state_zero_behavior_records_sha256": sha256_file(behavior_initial_path),
                "final_behavior_records_sha256": sha256_file(behavior_final_path),
            },
        }
        result_path = output / "result.json"
        write_json(result_path, result)
        print(json.dumps({
            "result": str(result_path),
            "result_sha256": sha256_file(result_path),
            "seed": args.seed,
            "protocol_valid": protocol_valid,
            "reward_delta": delta["environment_reward_sum"],
            "collision_delta": delta["collision_count"],
            "blocked_forward_delta": delta["blocked_forward_count"],
            "path_retention": delta["realized_path_retention"],
        }, sort_keys=True))
    finally:
        if manager is not None:
            cleanup_manager(manager)


if __name__ == "__main__":
    main()
