"""Run one preregistered G7.21 atomic-guard safety-transfer arm."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from run_streamvln_atomic_guard_g720 import (
    candidate_gates,
    evaluate_state,
    one_update,
    prepare_batch,
    replay_input_bundle,
)
from run_streamvln_learning_g718 import (
    FEEDBACK_KEYS,
    ORIGINAL_STREAM_COMMIT,
    RLINF_COMMIT,
    aggregate_behavior,
    collect_phase,
    flat_rows,
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
from run_streamvln_recovery_g6 import replay_individual, setup_manager
from run_streamvln_stability_g719_v2 import optimizer_state_contract
from run_streamvln_topology_g5 import (
    cleanup_manager,
    model_device,
    parameter_snapshot,
    state_sha256,
)
from streamvln_atomic_state import (
    canonical_sha256,
    capture_training_state,
    restore_training_state,
    training_state_hashes,
)
from streamvln_executor_contract import PRIVILEGED_POLICY_KEYS, assert_history_only_keys


MANIFEST_SHA256 = "daea9d8c8891e24550008069ff1f820ce649f02ab45c6aab45886729da53cdda"
G720_LOCAL_COMMIT = "8ee7e9101250d802f5c310597aab22bd8f8a786b"
G720_MANIFEST_SHA256 = "0286ffdcb1a42da579c76eeb6600d4123bb6edb242c1d7ec506836fb7d1a311d"
G720_RECOMPUTE_SHA256 = "e62a3522779f17fb013914a1affb45c557bb64240408d2149a1f5666df382ca8"
G720_RESULT_SHA256 = {
    2026092801: "6e2a14e1cfe1e063ce457974b0bafc78f053dccaa210c2e9cde2e163bd22cc6f",
    2026092802: "8613961e2f6475ce4bafbca007796a2ba8f4f512f5bb912d0b671b457f83974b",
    2026092803: "e22260cb3b103bc5812832cbeabd41a59efb49428bb944bf7d5b26c26b984a8e",
}
FROZEN_ARMS = {
    "2026092811": {
        "decision": [
            {"index": 5398, "episode_id": "5408", "scene": "Uxmj2M2itWa"},
            {"index": 5483, "episode_id": "5493", "scene": "EDJbREhghzL"},
            {"index": 1123, "episode_id": "1124", "scene": "1pXnuDYAj8r"},
            {"index": 7950, "episode_id": "7963", "scene": "D7N2EKCX4Sj"},
        ],
        "audit": [
            {"index": 1546, "episode_id": "1550", "scene": "PuKPg4mmafe"},
            {"index": 10147, "episode_id": "10166", "scene": "ac26ZMwG7aT"},
            {"index": 4266, "episode_id": "4276", "scene": "qoiz87JEwZ2"},
            {"index": 774, "episode_id": "775", "scene": "V2XKFyX4ASd"},
        ],
    },
    "2026092812": {
        "decision": [
            {"index": 4628, "episode_id": "4638", "scene": "b8cTxDM8gDG"},
            {"index": 8953, "episode_id": "8966", "scene": "pRbA3pwrgk9"},
            {"index": 2800, "episode_id": "2804", "scene": "p5wJjkQkbXX"},
            {"index": 5217, "episode_id": "5227", "scene": "E9uDoFAP3SH"},
        ],
        "audit": [
            {"index": 169, "episode_id": "170", "scene": "5q7pvUzZiYa"},
            {"index": 2627, "episode_id": "2631", "scene": "JmbYfDe2QKZ"},
            {"index": 5487, "episode_id": "5497", "scene": "759xd9YjKW5"},
            {"index": 8526, "episode_id": "8539", "scene": "sKLMLpTHeUy"},
        ],
    },
    "2026092813": {
        "decision": [
            {"index": 8029, "episode_id": "8042", "scene": "S9hNv5qa7GM"},
            {"index": 1810, "episode_id": "1814", "scene": "r47D5H71a5s"},
            {"index": 1657, "episode_id": "1661", "scene": "2n8kARJN3HM"},
            {"index": 7722, "episode_id": "7732", "scene": "uNb9QFRL6hY"},
        ],
        "audit": [
            {"index": 10099, "episode_id": "10118", "scene": "sT4fr6TAbpF"},
            {"index": 5603, "episode_id": "5613", "scene": "ZMojNkEp431"},
            {"index": 8827, "episode_id": "8840", "scene": "gTV8FGcVJC9"},
            {"index": 7, "episode_id": "8", "scene": "7y3sRwLe3Va"},
        ],
    },
}


def validate_manifest(manifest: dict) -> dict:
    expected = {
        "schema_version", "stage", "source_contract", "source", "predecessor",
        "selection", "policy_observation_keys", "privileged_policy_blacklist",
        "protocol", "decision_rule", "required_reports", "model_load", "rl_update",
        "navigation_evaluation", "controller_comparison", "robot_execution", "g8_ready",
    }
    if set(manifest) != expected or manifest["schema_version"] != 1:
        raise ValueError("G7.21 manifest schema changed")
    if manifest["source_contract"] != {
        "stream_commit": ORIGINAL_STREAM_COMMIT,
        "rlinf_commit": RLINF_COMMIT,
        "original_stream_only": True,
        "ttt_or_secondary_post_training": False,
    }:
        raise ValueError("G7.21 source contract changed")
    predecessor = manifest["predecessor"]
    if (
        predecessor["accepted_local_commit"] != G720_LOCAL_COMMIT
        or predecessor["manifest_sha256"] != G720_MANIFEST_SHA256
        or predecessor["result_sha256"] != {
            "s1": G720_RESULT_SHA256[2026092801],
            "s2": G720_RESULT_SHA256[2026092802],
            "s3": G720_RESULT_SHA256[2026092803],
        }
        or predecessor["evidence_recompute_sha256"] != G720_RECOMPUTE_SHA256
        or predecessor["claim"]
        != "atomic guard correctness accepted; second-update trigger repeatability rejected"
    ):
        raise ValueError("G7.21 predecessor anchor changed")
    assert_history_only_keys(manifest["policy_observation_keys"])
    if set(manifest["privileged_policy_blacklist"]) != PRIVILEGED_POLICY_KEYS:
        raise ValueError("G7.21 privileged-data blacklist changed")
    selection = manifest["selection"]
    protocol = manifest["protocol"]
    if protocol["arms"] != FROZEN_ARMS:
        raise ValueError("G7.21 frozen arms changed")
    all_rows = [
        row
        for arm in FROZEN_ARMS.values()
        for role in ("decision", "audit")
        for row in arm[role]
    ]
    all_scenes = [row["scene"] for row in all_rows]
    if (
        selection["seeds"] != [int(seed) for seed in FROZEN_ARMS]
        or selection["episodes_per_role"] != 4
        or selection["role_order"] != ["decision", "audit"]
        or selection["outcome_or_policy_fields_used"] is not False
        or selection["global_scene_disjoint"] is not True
        or selection["replacement_after_collection"] is not False
        or len(all_scenes) != len(set(all_scenes))
        or set(all_scenes) & set(selection["excluded_scenes"])
        or protocol["split"] != "train"
        or protocol["sampling"]["sample_calls_per_decision"] != 1
        or protocol["sampling"]["target_mode_scan"] is not False
        or protocol["sampling"]["mode_filter_or_resample"] is not False
        or protocol["sampling"]["recovery_override"] is not False
        or protocol["collection_contract"] != {
            "max_rows": 12,
            "min_rows": 8,
            "insufficient_rows": "protocol_invalid_without_replacement",
        }
        or protocol["audit_contract"] != {
            "max_rows": 12,
            "min_rows": 8,
            "collected_at_state_zero": True,
            "evaluated_once_at_final_accepted_state": True,
            "may_affect_commit_stop_or_checkpoint": False,
            "ratio_max_deviation": 0.5,
            "structured_kl_max": 0.1,
        }
        or protocol["max_decisions_per_episode"] != 3
        or protocol["max_primitives_per_episode"] != 12
        or protocol["candidate_update_limit"] != 3
        or protocol["stop_candidate_loop_after_first_rejection"] is not True
        or protocol["postupdate_ratio_max_deviation"] != 0.5
        or protocol["postupdate_distribution_kl_max"] != 0.1
        or protocol["postupdate_functional_change_min"] != 1e-8
        or protocol["reward_path_collision_and_collection_outcomes_are_descriptive_only"]
        is not True
        or protocol["outcome_based_seed_update_threshold_or_checkpoint_selection"]
        is not False
        or protocol["behavior_evaluation_episode_count"] != 0
        or protocol["val_unseen_episode_count"] != 0
    ):
        raise ValueError("G7.21 protocol changed")
    if protocol["continuation_probe"] != {
        "generated_before_each_candidate_from_previous_accepted_state": True,
        "restore_before_candidate_must_be_exact": True,
        "components": [
            "python_rng", "numpy_rng", "torch_cpu_rng", "torch_cuda_rng",
            "fixed_first_decision_replay",
        ],
        "rejected_candidate_probe_must_equal_reference": True,
        "restore_after_probe_must_be_exact": True,
    }:
        raise ValueError("G7.21 continuation probe changed")
    if (
        manifest["model_load"] is not True
        or manifest["rl_update"] is not True
        or manifest["navigation_evaluation"] is not False
        or manifest["controller_comparison"] is not False
        or manifest["robot_execution"] is not False
        or manifest["g8_ready"] is not False
    ):
        raise ValueError("G7.21 claim boundary changed")
    return protocol


def prepare_evaluation_batch(manager, core, episodes, contract: dict, label: str) -> dict:
    rows = flat_rows(episodes)
    if not contract["min_rows"] <= len(rows) <= contract["max_rows"]:
        raise RuntimeError(f"G7.21 {label} row count is outside preregistered range")
    device = model_device(core)
    indices = list(range(len(rows)))
    replay_inputs = replay_input_bundle(rows)
    input_sha = state_sha256(replay_inputs)
    with torch.no_grad():
        reference = replay_individual(manager, rows, indices, device)
        behavior_log_probs = torch.tensor(
            [row["behavior_log_prob"] for row in rows], dtype=torch.float32, device=device
        )
        pre_ratios = torch.exp(reference["log_probs"].float() - behavior_log_probs)
        reference_log_std = core.waypoint_log_std.detach().float().clamp(-3.0, -0.25).clone()
    return {
        "rows": rows,
        "indices": indices,
        "device": device,
        "replay_inputs": replay_inputs,
        "input_sha256": input_sha,
        "reference": reference,
        "reference_log_std": reference_log_std,
        "behavior_log_probs": behavior_log_probs,
        "preupdate_ratio_max_deviation": float((pre_ratios - 1.0).abs().max()),
    }


def fixed_continuation_probe(manager, batch: dict) -> dict:
    with torch.no_grad():
        replay = replay_individual(manager, [batch["rows"][0]], [0], batch["device"])
    components = {
        "python_rng": random.getrandbits(64),
        "numpy_rng": np.random.randint(0, 2**31, size=4, dtype=np.int64),
        "torch_cpu_rng": torch.rand(4, device="cpu"),
        "torch_cuda_rng": torch.rand(4, device=batch["device"]).detach().cpu(),
        "fixed_first_decision_replay": {
            key: value.detach().cpu() if torch.is_tensor(value) else value
            for key, value in replay.items()
        },
    }
    hashes = {name: canonical_sha256(value) for name, value in components.items()}
    return {"component_sha256": hashes, "complete_sha256": canonical_sha256(hashes)}


def compare_probes(reference: dict, restored: dict) -> dict:
    names = sorted(set(reference["component_sha256"]) | set(restored["component_sha256"]))
    component_equal = {
        name: reference["component_sha256"].get(name)
        == restored["component_sha256"].get(name)
        for name in names
    }
    return {
        "reference": reference,
        "restored": restored,
        "component_equal": component_equal,
        "exact": all(component_equal.values())
        and reference["complete_sha256"] == restored["complete_sha256"],
    }


def collection_semantics(episodes) -> bool:
    return all(
        set(row["executor_feedback"]) == FEEDBACK_KEYS
        and row["sample_call_count"] == 1
        and row["recovery_override"] is False
        and row["reward_accounting_error"] <= 1e-6
        and math.isfinite(row["advantage"])
        and math.isfinite(row["return"])
        for row in flat_rows(episodes)
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--g720-results", type=Path, nargs=3, required=True)
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
        raise RuntimeError("G7.21 frozen manifest hash changed")
    manifest = json.loads(manifest_path.read_text())
    protocol = validate_manifest(manifest)
    seed_key = str(args.seed)
    if seed_key not in FROZEN_ARMS:
        raise RuntimeError("G7.21 seed is not preregistered")
    specs = FROZEN_ARMS[seed_key]

    predecessor_paths = [path.resolve(strict=True) for path in args.g720_results]
    predecessor_results = {}
    for path in predecessor_paths:
        predecessor = json.loads(path.read_text())
        seed = int(predecessor["seed"])
        if seed not in G720_RESULT_SHA256 or sha256_file(path) != G720_RESULT_SHA256[seed]:
            raise RuntimeError("G7.21 G7.20 result anchor changed")
        if predecessor["all_protocol_guards_passed"] is not True:
            raise RuntimeError("G7.21 predecessor protocol is invalid")
        predecessor_results[seed] = str(path)
    if set(predecessor_results) != set(G720_RESULT_SHA256):
        raise RuntimeError("G7.21 predecessor result set changed")

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
        raise RuntimeError("G7.21 source commit changed")
    datasets = dataset_hashes(Path(paths["asset_root"]), manifest)
    script_root = Path(__file__).resolve().parent
    implementation_names = (
        "run_streamvln_safety_transfer_g721.py", "run_streamvln_atomic_guard_g720.py",
        "streamvln_atomic_state.py", "run_streamvln_learning_g718.py",
        "run_streamvln_plumbing_g717.py", "streamvln_topological_policy.py",
        "streamvln_recovery_policy.py", "streamvln_executor_contract.py",
        "streamvln_topology.py", "streamvln_chunk_ppo.py",
    )
    implementation_paths = {name: script_root / name for name in implementation_names}
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
        name: sha256_file(path.resolve(strict=True)) for name, path in implementation_paths.items()
    }
    run_cfg = {
        **paths,
        "output_dir": str(output),
        "stream_commit": ORIGINAL_STREAM_COMMIT,
        "rlinf_commit": RLINF_COMMIT,
        "manifest": {
            "source": manifest["source"],
            "train": specs["decision"] + specs["audit"],
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
        "kind": "streamvln-g7.21-atomic-guard-safety-transfer",
        "manifest": str(manifest_path),
        "manifest_sha256": MANIFEST_SHA256,
        "predecessor_results": predecessor_results,
        "predecessor_sha256": G720_RESULT_SHA256,
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
        "decision_scene_specs": specs["decision"],
        "audit_scene_specs": specs["audit"],
        "model_load_count": 1,
        "candidate_update_limit": protocol["candidate_update_limit"],
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
            raise RuntimeError("G7.21 initial optimizer differs from eager-zero contract")
        initial_parameters = parameter_snapshot(core)
        initial_parameter_hash = state_sha256(initial_parameters)
        frozen_initial, frozen_names = frozen_parameter_sentinels(core)
        frozen_initial_hash = state_sha256(frozen_initial)

        decision_collection = collect_phase(
            manager, core, run_cfg, specs["decision"],
            phase=f"g721_seed_{args.seed}_decision", seed_phase="g721_decision",
        )
        audit_collection = collect_phase(
            manager, core, run_cfg, specs["audit"],
            phase=f"g721_seed_{args.seed}_audit", seed_phase="g721_audit",
        )
        decision_path = output / "decision_collection_records.jsonl"
        audit_path = output / "audit_collection_records.jsonl"
        write_jsonl(decision_path, decision_collection)
        write_jsonl(audit_path, audit_collection)
        if state_sha256(parameter_snapshot(core)) != initial_parameter_hash:
            raise RuntimeError("G7.21 collection changed high-level parameters")
        decision_batch = prepare_batch(manager, core, run_cfg, decision_collection)
        audit_batch = prepare_evaluation_batch(
            manager, core, audit_collection, protocol["audit_contract"], "audit"
        )
        if (
            decision_batch["preupdate_ratio_max_deviation"]
            > protocol["preupdate_ratio_max_deviation"]
            or audit_batch["preupdate_ratio_max_deviation"]
            > protocol["preupdate_ratio_max_deviation"]
        ):
            raise RuntimeError("G7.21 state-zero ratio differs from one")
        state_zero = evaluate_state(
            manager, core, decision_batch, initial_parameters, "decision_state_zero"
        )
        baseline_snapshot = capture_training_state(manager, core)
        baseline_hashes = training_state_hashes(baseline_snapshot)
        previous_report = state_zero
        accepted_snapshot = baseline_snapshot
        accepted_hashes = [baseline_hashes]
        candidates = []

        for candidate_index in range(1, protocol["candidate_update_limit"] + 1):
            before_snapshot = capture_training_state(manager, core)
            before_hashes = training_state_hashes(before_snapshot)
            reference_probe = fixed_continuation_probe(manager, decision_batch)
            reference_probe_restore = restore_training_state(manager, core, before_snapshot)
            if not reference_probe_restore["exact"]:
                raise RuntimeError("G7.21 reference probe did not restore exactly")
            before_parameters = parameter_snapshot(core)
            frozen_before, names_before = frozen_parameter_sentinels(core)
            if names_before != frozen_names:
                raise RuntimeError("G7.21 frozen sentinel schema changed")
            update = one_update(manager, core, run_cfg, decision_batch, candidate_index)
            report = evaluate_state(
                manager, core, decision_batch, before_parameters,
                f"candidate_{candidate_index}",
            )
            frozen_after, names_after = frozen_parameter_sentinels(core)
            if names_after != frozen_names:
                raise RuntimeError("G7.21 frozen sentinel schema changed after candidate")
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
                accepted_hashes.append(after_hashes)
                previous_report = report
                disposition = "committed_all_decision_gates_passed"
            else:
                restoration = restore_training_state(manager, core, before_snapshot)
                if not restoration["exact"]:
                    raise RuntimeError("G7.21 rejected candidate did not restore exactly")
                restored_probe = fixed_continuation_probe(manager, decision_batch)
                continuation = compare_probes(reference_probe, restored_probe)
                if not continuation["exact"]:
                    raise RuntimeError("G7.21 rejected candidate changed continuation")
                probe_cleanup_restore = restore_training_state(manager, core, before_snapshot)
                if not probe_cleanup_restore["exact"]:
                    raise RuntimeError("G7.21 continuation probe cleanup was not exact")
                accepted_snapshot = before_snapshot
                disposition = "rejected_restored_continuation_equal"
            candidates.append({
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
            })
            if not accepted:
                break

        accepted_before_audit = capture_training_state(manager, core)
        accepted_before_audit_hashes = training_state_hashes(accepted_before_audit)
        audit_state = evaluate_state(
            manager, core, audit_batch, initial_parameters, "final_disjoint_audit"
        )
        audit_restore = restore_training_state(manager, core, accepted_before_audit)
        if not audit_restore["exact"]:
            raise RuntimeError("G7.21 final audit changed committed state")
        final_snapshot = capture_training_state(manager, core)
        final_hashes = training_state_hashes(final_snapshot)
        expected_final_hashes = training_state_hashes(accepted_snapshot)
        frozen_final, final_names = frozen_parameter_sentinels(core)

        decision_rows = flat_rows(decision_collection)
        audit_rows = flat_rows(audit_collection)
        all_specs = specs["decision"] + specs["audit"]
        guards = {
            "source_and_predecessors_anchored": source_heads
            == {"stream": ORIGINAL_STREAM_COMMIT, "rlinf": RLINF_COMMIT}
            and set(predecessor_results) == set(G720_RESULT_SHA256),
            "dataset_hashes_fixed": all(
                datasets[split]["sha256"] == manifest["source"][f"{split}_sha256"]
                for split in datasets
            ),
            "preregistered_seed_and_scene_specs": specs == protocol["arms"][seed_key]
            and len({row["scene"] for row in all_specs}) == len(all_specs),
            "decision_collection_row_count_in_range":
            protocol["collection_contract"]["min_rows"] <= len(decision_rows)
            <= protocol["collection_contract"]["max_rows"],
            "audit_collection_row_count_in_range":
            protocol["audit_contract"]["min_rows"] <= len(audit_rows)
            <= protocol["audit_contract"]["max_rows"],
            "ordinary_decision_collection_semantics": collection_semantics(decision_collection),
            "ordinary_audit_collection_semantics": collection_semantics(audit_collection),
            "decision_and_audit_disjoint": not (
                {episode["scene"] for episode in decision_collection}
                & {episode["scene"] for episode in audit_collection}
            ),
            "state_zero_ratios_near_one":
            decision_batch["preupdate_ratio_max_deviation"]
            <= protocol["preupdate_ratio_max_deviation"]
            and audit_batch["preupdate_ratio_max_deviation"]
            <= protocol["preupdate_ratio_max_deviation"],
            "initial_optimizer_eager_zero_contract": optimizer_initial
            == protocol["optimizer_initial_contract"],
            "optimizer_scope_is_high_level_only":
            optimizer_scope["optimizer_equals_trainable"]
            and optimizer_scope["optimizer_frozen_overlap_count"] == 0,
            "all_accepted_candidates_pass_every_gate": all(
                not row["accepted"] or all(row["gates"].values()) for row in candidates
            ),
            "all_candidate_reference_probes_restore_exactly": all(
                row["reference_probe_restore"]["exact"] for row in candidates
            ),
            "all_rejected_candidates_restore_exactly": all(
                row["accepted"] or (row["restoration"] and row["restoration"]["exact"])
                for row in candidates
            ),
            "all_rejected_continuations_equal_reference": all(
                row["accepted"]
                or (
                    row["continuation_equivalence"]
                    and row["continuation_equivalence"]["exact"]
                    and row["probe_cleanup_restore"]
                    and row["probe_cleanup_restore"]["exact"]
                )
                for row in candidates
            ),
            "stopped_after_first_rejection": not any(
                not row["accepted"] for row in candidates[:-1]
            ),
            "candidate_limit_respected": len(candidates) <= protocol["candidate_update_limit"],
            "audit_evaluated_once_and_restored": audit_restore["exact"]
            and accepted_before_audit_hashes == final_hashes,
            "final_state_equals_last_accepted_state": final_hashes == expected_final_hashes,
            "frozen_parameters_unchanged": final_names == frozen_names
            and state_sha256(frozen_final) == frozen_initial_hash,
            "collection_inputs_unchanged":
            state_sha256(decision_batch["replay_inputs"])
            == decision_batch["input_sha256"]
            and state_sha256(audit_batch["replay_inputs"]) == audit_batch["input_sha256"],
            "audit_nonselective": protocol["audit_contract"][
                "may_affect_commit_stop_or_checkpoint"
            ] is False,
            "reward_and_outcomes_nonselective":
            protocol["reward_path_collision_and_collection_outcomes_are_descriptive_only"]
            and protocol["outcome_based_seed_update_threshold_or_checkpoint_selection"]
            is False,
            "zero_behavior_val_unseen_and_checkpoint_selection":
            protocol["behavior_evaluation_episode_count"] == 0
            and protocol["val_unseen_episode_count"] == 0
            and config["checkpoint_selection"] is False,
        }
        protocol_valid = all(guards.values())
        audit_bounds = {
            "ratio_bounded": audit_state["sampled_action_ratio_max_deviation"]
            <= protocol["audit_contract"]["ratio_max_deviation"],
            "structured_kl_bounded": audit_state["structured_distribution_kl"]["max"]
            <= protocol["audit_contract"]["structured_kl_max"],
        }
        safety_transfer_member = protocol_valid and all(audit_bounds.values())
        trigger_index = next(
            (row["candidate_index"] for row in candidates if not row["accepted"]), None
        )
        accepted_update_count = sum(row["accepted"] for row in candidates)
        if not protocol_valid:
            outcome = "protocol_invalid"
        elif safety_transfer_member:
            outcome = "decision_selected_state_bounded_on_disjoint_audit"
        else:
            outcome = "decision_selected_state_exceeded_disjoint_audit_bound"
        result = {
            "kind": config["kind"],
            "seed": args.seed,
            "all_protocol_guards_passed": protocol_valid,
            "outcome": outcome,
            "safety_transfer_member": safety_transfer_member,
            "navigation_or_generalization_claim": False,
            "source_contract": manifest["source_contract"],
            "trainable_manifest": trainable_manifest,
            "optimizer_scope": optimizer_scope,
            "optimizer_initial_contract": optimizer_initial,
            "counts": {
                "model_loads": 1,
                "decision_episodes": len(decision_collection),
                "decision_rows": len(decision_rows),
                "audit_episodes": len(audit_collection),
                "audit_rows": len(audit_rows),
                "candidate_updates_attempted": len(candidates),
                "candidate_updates_committed": accepted_update_count,
                "behavior_episodes": 0,
                "val_unseen_episodes": 0,
            },
            "trigger_index": trigger_index,
            "accepted_update_count": accepted_update_count,
            "decision_collection": {
                "records_sha256": sha256_file(decision_path),
                "input_sha256": decision_batch["input_sha256"],
                "preupdate_ratio_max_deviation":
                decision_batch["preupdate_ratio_max_deviation"],
                "descriptive_metrics": aggregate_behavior(decision_collection),
            },
            "audit_collection": {
                "records_sha256": sha256_file(audit_path),
                "input_sha256": audit_batch["input_sha256"],
                "preupdate_ratio_max_deviation": audit_batch["preupdate_ratio_max_deviation"],
                "descriptive_metrics": aggregate_behavior(audit_collection),
            },
            "initial_high_level_sha256": initial_parameter_hash,
            "decision_state_zero": state_zero,
            "baseline_training_state_hashes": baseline_hashes,
            "candidates": candidates,
            "accepted_training_state_hashes": accepted_hashes,
            "accepted_before_audit_training_state_hashes": accepted_before_audit_hashes,
            "final_training_state_hashes": final_hashes,
            "final_disjoint_audit": audit_state,
            "audit_bounds": audit_bounds,
            "audit_restore": audit_restore,
            "protocol_guards": guards,
            "g8_ready": False,
            "elapsed_seconds": time.time() - started,
            "evidence": {
                "manifest_sha256": MANIFEST_SHA256,
                "run_config_sha256": sha256_file(config_path),
                "decision_collection_records_sha256": sha256_file(decision_path),
                "audit_collection_records_sha256": sha256_file(audit_path),
            },
        }
        result_path = output / "result.json"
        write_json(result_path, result)
        print(json.dumps({
            "result": str(result_path),
            "result_sha256": sha256_file(result_path),
            "seed": args.seed,
            "outcome": outcome,
            "safety_transfer_member": safety_transfer_member,
            "protocol_valid": protocol_valid,
            "trigger_index": trigger_index,
            "accepted_update_count": accepted_update_count,
        }, sort_keys=True))
    finally:
        if manager is not None:
            cleanup_manager(manager)


if __name__ == "__main__":
    main()
