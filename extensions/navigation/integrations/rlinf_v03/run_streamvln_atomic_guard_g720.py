"""Run one preregistered G7.20 atomic candidate-update confirmation arm."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from run_streamvln_learning_g718 import (
    FEEDBACK_KEYS,
    ORIGINAL_STREAM_COMMIT,
    RLINF_COMMIT,
    aggregate_behavior,
    collect_phase,
    flat_rows,
    structured_distribution_kl,
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
    gradient_audit,
    model_device,
    parameter_changes,
    parameter_snapshot,
    state_sha256,
)
from streamvln_atomic_state import (
    capture_training_state,
    restore_training_state,
    training_state_hashes,
)
from streamvln_executor_contract import PRIVILEGED_POLICY_KEYS, assert_history_only_keys


MANIFEST_SHA256 = "0286ffdcb1a42da579c76eeb6600d4123bb6edb242c1d7ec506836fb7d1a311d"
G719_LOCAL_COMMIT = "eef48f1894c8ce934b2c9aac5868578cae2106a2"
G719_MANIFEST_SHA256 = "bd86c0c6ebd3597f1d4a2bae7b0a546f3e5dc4ff45676e4cc002f44c8448d423"
G719_RESULT_SHA256 = "1b846ab7ba9d0948633f4264f363dcc3169b01cb3307632c817ddd92a4d8242a"
G719_RUN_CONFIG_SHA256 = "2e6869c1764bb5837bce32a12ad0a820b09a47e1bf0aab10a3317eed6d2eb4a5"
FROZEN_ARMS = {
    "2026092801": [
        {"index": 4074, "episode_id": "4084", "scene": "ur6pFq6Qu1A"},
        {"index": 10789, "episode_id": "10808", "scene": "cV4RVeZvu5T"},
        {"index": 4452, "episode_id": "4462", "scene": "VVfe2KiqLaN"},
        {"index": 7729, "episode_id": "7739", "scene": "ULsKaCPVFJR"},
    ],
    "2026092802": [
        {"index": 334, "episode_id": "335", "scene": "aayBHfsNo7d"},
        {"index": 7819, "episode_id": "7829", "scene": "82sE5b5pLXE"},
        {"index": 5297, "episode_id": "5307", "scene": "PX4nDJXEHrG"},
        {"index": 7109, "episode_id": "7119", "scene": "rPc6DW4iMge"},
    ],
    "2026092803": [
        {"index": 928, "episode_id": "929", "scene": "mJXqzFtmKg4"},
        {"index": 8414, "episode_id": "8427", "scene": "r1Q1Z4BcV1o"},
        {"index": 6708, "episode_id": "6718", "scene": "VzqfbhrpDEA"},
        {"index": 6406, "episode_id": "6416", "scene": "B6ByNegPMKs"},
    ],
}


def validate_manifest(manifest: dict) -> dict:
    expected = {
        "schema_version", "stage", "source_contract", "source", "predecessor",
        "selection", "policy_observation_keys", "privileged_policy_blacklist",
        "protocol", "decision_rule", "required_reports", "gates", "model_load",
        "rl_update", "navigation_evaluation", "controller_comparison",
        "robot_execution", "g8_ready",
    }
    if set(manifest) != expected or manifest["schema_version"] != 1:
        raise ValueError("G7.20 manifest schema changed")
    if manifest["source_contract"] != {
        "stream_commit": ORIGINAL_STREAM_COMMIT,
        "rlinf_commit": RLINF_COMMIT,
        "original_stream_only": True,
        "ttt_or_secondary_post_training": False,
    }:
        raise ValueError("G7.20 source contract changed")
    predecessor = manifest["predecessor"]
    if (
        predecessor["accepted_local_commit"] != G719_LOCAL_COMMIT
        or predecessor["manifest_sha256"] != G719_MANIFEST_SHA256
        or predecessor["result_sha256"] != G719_RESULT_SHA256
        or predecessor["run_config_sha256"] != G719_RUN_CONFIG_SHA256
        or predecessor["causal_outcome"] != "second_update_trigger_supported"
    ):
        raise ValueError("G7.20 predecessor anchor changed")
    assert_history_only_keys(manifest["policy_observation_keys"])
    if set(manifest["privileged_policy_blacklist"]) != PRIVILEGED_POLICY_KEYS:
        raise ValueError("G7.20 privileged-data blacklist changed")
    selection = manifest["selection"]
    protocol = manifest["protocol"]
    if protocol["arms"] != FROZEN_ARMS:
        raise ValueError("G7.20 frozen arms changed")
    all_scenes = [row["scene"] for specs in FROZEN_ARMS.values() for row in specs]
    if (
        selection["seeds"] != [int(seed) for seed in FROZEN_ARMS]
        or selection["episodes_per_seed"] != 4
        or selection["outcome_or_policy_fields_used"] is not False
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
            "independent_collection_per_seed": True,
        }
        or protocol["max_decisions_per_episode"] != 3
        or protocol["max_primitives_per_episode"] != 12
        or protocol["candidate_update_limit"] != 2
        or protocol["candidate_two_requires_candidate_one_accepted"] is not True
        or protocol["postupdate_ratio_max_deviation"] != 0.5
        or protocol["postupdate_distribution_kl_max"] != 0.1
        or protocol["postupdate_functional_change_min"] != 1e-8
        or protocol["reward_path_and_collection_outcomes_are_descriptive_only"] is not True
        or protocol["outcome_based_seed_update_or_threshold_selection"] is not False
        or protocol["behavior_evaluation_episode_count"] != 0
        or protocol["off_batch_audit_episode_count"] != 0
        or protocol["checkpoint_selection"] is not False
        or protocol["val_unseen_episode_count"] != 0
    ):
        raise ValueError("G7.20 protocol changed")
    if manifest["decision_rule"] != {
        "per_seed_member": "candidate one accepted and candidate two rejected with exact rollback",
        "repeatability_supported": "at least two of three preregistered seeds are members and every run passes protocol guards",
        "otherwise": "retain only atomic-guard implementation evidence and do not claim repeatability",
        "g719_seed_counts_as_confirmation": False,
        "outcomes_may_change_seed_threshold_learning_rate_data_or_batch": False,
    }:
        raise ValueError("G7.20 decision rule changed")
    if (
        manifest["model_load"] is not True
        or manifest["rl_update"] is not True
        or manifest["navigation_evaluation"] is not False
        or manifest["controller_comparison"] is not False
        or manifest["robot_execution"] is not False
        or manifest["g8_ready"] is not False
    ):
        raise ValueError("G7.20 claim boundary changed")
    return protocol


def replay_input_bundle(rows) -> dict:
    return {
        **{f"state_{index}": row["state"] for index, row in enumerate(rows)},
        **{f"topology_{index}": row["topology_features"] for index, row in enumerate(rows)},
        "modes": torch.tensor([row["policy_mode"] for row in rows], dtype=torch.int64),
        "waypoints": torch.tensor([row["waypoint"] for row in rows], dtype=torch.float32),
    }


def prepare_batch(manager, core, run_cfg, episodes) -> dict:
    rows = flat_rows(episodes)
    contract = run_cfg["protocol"]["collection_contract"]
    if not contract["min_rows"] <= len(rows) <= contract["max_rows"]:
        raise RuntimeError("G7.20 frozen collection row count is outside preregistered range")
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
    pre_deviation = float((pre_ratios - 1.0).abs().max())
    if pre_deviation > run_cfg["protocol"]["preupdate_ratio_max_deviation"]:
        raise RuntimeError("G7.20 pre-update ratio differs from one")
    advantages_raw = torch.tensor([row["advantage"] for row in rows], dtype=torch.float32)
    advantage_std = advantages_raw.std(unbiased=False)
    if not bool(torch.isfinite(advantage_std)) or float(advantage_std) <= 1e-8:
        raise RuntimeError("G7.20 collection advantages are degenerate")
    return {
        "rows": rows,
        "indices": indices,
        "device": device,
        "replay_inputs": replay_inputs,
        "input_sha256": input_sha,
        "reference": reference,
        "reference_log_std": reference_log_std,
        "behavior_log_probs": behavior_log_probs,
        "advantages": ((advantages_raw - advantages_raw.mean()) / advantage_std).to(device),
        "returns": torch.tensor(
            [row["return"] for row in rows], dtype=torch.float32, device=device
        ),
        "preupdate_ratio_max_deviation": pre_deviation,
    }


def evaluate_state(manager, core, batch, previous_parameters, label: str) -> dict:
    with torch.no_grad():
        replay = replay_individual(
            manager, batch["rows"], batch["indices"], batch["device"]
        )
        ratios = torch.exp(replay["log_probs"].float() - batch["behavior_log_probs"])
        current_log_std = core.waypoint_log_std.detach().float().clamp(-3.0, -0.25)
        combined, categorical, gaussian = structured_distribution_kl(
            batch["reference"], replay, batch["reference_log_std"], current_log_std
        )
    return {
        "label": label,
        "actor_version": int(core.actor_version),
        "sampled_action_ratio_mean": float(ratios.mean()),
        "sampled_action_ratio_max_deviation": float((ratios - 1.0).abs().max()),
        "structured_distribution_kl": {
            "mean": float(combined.mean()),
            "max": float(combined.max()),
            "categorical_mean": float(categorical.mean()),
            "gaussian_mean": float(gaussian.mean()),
        },
        "parameter_changes_from_previous": parameter_changes(core, previous_parameters),
        "collection_input_sha256": state_sha256(batch["replay_inputs"]),
        "high_level_parameters_finite": all(
            bool(torch.isfinite(value.detach().float()).all())
            for value in core.high_level_state().values()
        ),
    }


def one_update(manager, core, run_cfg, batch, update_index: int) -> dict:
    replay = replay_individual(manager, batch["rows"], batch["indices"], batch["device"])
    ratios = torch.exp(replay["log_probs"].float() - batch["behavior_log_probs"])
    clipped = ratios.clamp(1.0 - run_cfg["clip_ratio"], 1.0 + run_cfg["clip_ratio"])
    actor_loss = -torch.minimum(
        ratios * batch["advantages"], clipped * batch["advantages"]
    ).mean()
    value_loss = 0.5 * (replay["values"].float() - batch["returns"]).square().mean()
    entropy = replay["entropy"].float().mean()
    loss = actor_loss + run_cfg["value_coef"] * value_loss - run_cfg["entropy_coef"] * entropy
    if not all(bool(torch.isfinite(value)) for value in (loss, actor_loss, value_loss, entropy)):
        raise RuntimeError("G7.20 candidate loss is nonfinite")
    manager.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    audit = gradient_audit(core)
    if (
        not audit["finite"]
        or audit["group_norm_sums"]["high_level_policy"] <= 0
        or audit["group_norm_sums"]["value_head"] <= 0
        or audit["frozen_with_grad"]
    ):
        raise RuntimeError("G7.20 candidate gradient isolation failed")
    grad_norm, learning_rates = manager.optimizer_step()
    core.mark_updated()
    return {
        "candidate_index": update_index,
        "ratio_before_update_mean": float(ratios.detach().mean()),
        "ratio_before_update_max_deviation": float((ratios.detach() - 1.0).abs().max()),
        "loss": float(loss.detach()),
        "actor_loss": float(actor_loss.detach()),
        "value_loss": float(value_loss.detach()),
        "entropy": float(entropy.detach()),
        "grad_norm": float(torch.as_tensor(grad_norm)),
        "learning_rates": [float(value) for value in learning_rates],
        "gradient_audit": audit,
        "actor_version_after": int(core.actor_version),
        "optimizer_steps_after": int(manager.optimizer_steps),
    }


def candidate_gates(
    report: dict,
    previous_report: dict,
    update: dict,
    protocol: dict,
    *,
    actor_version_before: int,
    optimizer_steps_before: int,
    frozen_before: str,
    frozen_after: str,
    input_sha256: str,
) -> dict:
    functional_delta = max(
        abs(
            report["sampled_action_ratio_max_deviation"]
            - previous_report["sampled_action_ratio_max_deviation"]
        ),
        abs(
            report["structured_distribution_kl"]["mean"]
            - previous_report["structured_distribution_kl"]["mean"]
        ),
        report["parameter_changes_from_previous"]["max_abs"],
    )
    finite_update = all(math.isfinite(update[key]) for key in (
        "loss", "actor_loss", "value_loss", "entropy", "grad_norm"
    ))
    return {
        "ratio_bounded": report["sampled_action_ratio_max_deviation"]
        <= protocol["postupdate_ratio_max_deviation"],
        "structured_kl_bounded": report["structured_distribution_kl"]["max"]
        <= protocol["postupdate_distribution_kl_max"],
        "functional_change_from_previous": functional_delta
        >= protocol["postupdate_functional_change_min"],
        "finite_loss_gradients_and_parameters": finite_update
        and update["gradient_audit"]["finite"]
        and report["high_level_parameters_finite"],
        "nonzero_actor_and_value_gradients":
        update["gradient_audit"]["group_norm_sums"]["high_level_policy"] > 0
        and update["gradient_audit"]["group_norm_sums"]["value_head"] > 0,
        "no_frozen_gradients": not update["gradient_audit"]["frozen_with_grad"],
        "high_level_parameters_changed":
        report["parameter_changes_from_previous"]["changed_tensor_count"] > 0,
        "frozen_sentinels_unchanged": frozen_before == frozen_after,
        "collection_inputs_unchanged": report["collection_input_sha256"] == input_sha256,
        "actor_version_incremented_once": report["actor_version"] == actor_version_before + 1,
        "optimizer_steps_incremented_once":
        update["optimizer_steps_after"] == optimizer_steps_before + 1,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--g719-result", type=Path, required=True)
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
        raise RuntimeError("G7.20 frozen manifest hash changed")
    manifest = json.loads(manifest_path.read_text())
    protocol = validate_manifest(manifest)
    seed_key = str(args.seed)
    if seed_key not in FROZEN_ARMS:
        raise RuntimeError("G7.20 seed is not preregistered")
    specs = FROZEN_ARMS[seed_key]

    predecessor_path = args.g719_result.resolve(strict=True)
    if sha256_file(predecessor_path) != G719_RESULT_SHA256:
        raise RuntimeError("G7.20 G7.19 result anchor changed")
    predecessor = json.loads(predecessor_path.read_text())
    if (
        predecessor["all_protocol_guards_passed"] is not True
        or predecessor["causal_outcome"] != "second_update_trigger_supported"
    ):
        raise RuntimeError("G7.20 predecessor is not the valid G7.19 result")

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
        raise RuntimeError("G7.20 source commit changed")
    datasets = dataset_hashes(Path(paths["asset_root"]), manifest)
    script_root = Path(__file__).resolve().parent
    implementation_names = (
        "run_streamvln_atomic_guard_g720.py", "streamvln_atomic_state.py",
        "run_streamvln_learning_g718.py", "run_streamvln_plumbing_g717.py",
        "streamvln_topological_policy.py", "streamvln_recovery_policy.py",
        "streamvln_executor_contract.py", "streamvln_topology.py",
        "streamvln_chunk_ppo.py",
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
        "manifest": {"source": manifest["source"], "train": specs, "val_unseen": []},
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
        "kind": "streamvln-g7.20-atomic-candidate-guard",
        "manifest": str(manifest_path),
        "manifest_sha256": MANIFEST_SHA256,
        "predecessor_path": str(predecessor_path),
        "predecessor_sha256": G719_RESULT_SHA256,
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
        "collection_scene_specs": specs,
        "model_load_count": 1,
        "candidate_update_limit": 2,
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
        manager, core, trainable_manifest = setup_manager(run_cfg, total_steps=2)
        optimizer_scope = optimizer_parameter_scope(manager, core)
        optimizer_initial = optimizer_state_contract(manager.optimizer)
        if optimizer_initial != protocol["optimizer_initial_contract"]:
            raise RuntimeError("G7.20 initial optimizer differs from eager-zero contract")
        initial_parameters = parameter_snapshot(core)
        initial_parameter_hash = state_sha256(initial_parameters)
        frozen_initial, frozen_names = frozen_parameter_sentinels(core)
        frozen_initial_hash = state_sha256(frozen_initial)

        collection = collect_phase(
            manager, core, run_cfg, specs,
            phase=f"g720_seed_{args.seed}_collection", seed_phase="g720_collection",
        )
        collection_path = output / "collection_records.jsonl"
        write_jsonl(collection_path, collection)
        if state_sha256(parameter_snapshot(core)) != initial_parameter_hash:
            raise RuntimeError("G7.20 collection changed high-level parameters")
        batch = prepare_batch(manager, core, run_cfg, collection)
        state_zero = evaluate_state(manager, core, batch, initial_parameters, "state_zero")
        baseline_snapshot = capture_training_state(manager, core)
        baseline_hashes = training_state_hashes(baseline_snapshot)
        previous_report = state_zero
        accepted_snapshot = baseline_snapshot
        accepted_hashes = [baseline_hashes]
        candidates = []

        for candidate_index in range(1, protocol["candidate_update_limit"] + 1):
            if candidate_index == 2 and not candidates[0]["accepted"]:
                break
            before_snapshot = capture_training_state(manager, core)
            before_hashes = training_state_hashes(before_snapshot)
            before_parameters = parameter_snapshot(core)
            frozen_before, names_before = frozen_parameter_sentinels(core)
            if names_before != frozen_names:
                raise RuntimeError("G7.20 frozen sentinel schema changed")
            update = one_update(manager, core, run_cfg, batch, candidate_index)
            report = evaluate_state(
                manager, core, batch, before_parameters, f"candidate_{candidate_index}"
            )
            frozen_after, names_after = frozen_parameter_sentinels(core)
            if names_after != frozen_names:
                raise RuntimeError("G7.20 frozen sentinel schema changed after candidate")
            after_snapshot = capture_training_state(manager, core)
            after_hashes = training_state_hashes(after_snapshot)
            gates = candidate_gates(
                report, previous_report, update, protocol,
                actor_version_before=before_snapshot["actor_version"],
                optimizer_steps_before=before_snapshot["optimizer_steps"],
                frozen_before=state_sha256(frozen_before),
                frozen_after=state_sha256(frozen_after),
                input_sha256=batch["input_sha256"],
            )
            accepted = all(gates.values())
            restoration = None
            if accepted:
                accepted_snapshot = after_snapshot
                accepted_hashes.append(after_hashes)
                previous_report = report
                disposition = "committed_all_gates_passed"
            else:
                restoration = restore_training_state(manager, core, before_snapshot)
                if not restoration["exact"]:
                    raise RuntimeError("G7.20 rejected candidate did not restore exactly")
                accepted_snapshot = before_snapshot
                previous_report = previous_report
                disposition = "rejected_and_exactly_restored"
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
                "restoration": restoration,
            })
            if not accepted:
                break

        final_snapshot = capture_training_state(manager, core)
        final_hashes = training_state_hashes(final_snapshot)
        expected_final_hashes = training_state_hashes(accepted_snapshot)
        frozen_final, final_names = frozen_parameter_sentinels(core)
        all_rows = flat_rows(collection)
        collection_semantics = all(
            set(row["executor_feedback"]) == FEEDBACK_KEYS
            and row["sample_call_count"] == 1
            and row["recovery_override"] is False
            and row["reward_accounting_error"] <= 1e-6
            and math.isfinite(row["advantage"])
            and math.isfinite(row["return"])
            for row in all_rows
        )
        guards = {
            "source_and_predecessor_anchored": source_heads
            == {"stream": ORIGINAL_STREAM_COMMIT, "rlinf": RLINF_COMMIT}
            and sha256_file(predecessor_path) == G719_RESULT_SHA256,
            "dataset_hashes_fixed": all(
                datasets[split]["sha256"] == manifest["source"][f"{split}_sha256"]
                for split in datasets
            ),
            "preregistered_seed_and_scene_specs": specs == protocol["arms"][seed_key],
            "collection_row_count_in_range":
            protocol["collection_contract"]["min_rows"] <= len(all_rows)
            <= protocol["collection_contract"]["max_rows"],
            "ordinary_collection_semantics": collection_semantics,
            "preupdate_ratio_near_one": batch["preupdate_ratio_max_deviation"]
            <= protocol["preupdate_ratio_max_deviation"],
            "initial_optimizer_eager_zero_contract": optimizer_initial
            == protocol["optimizer_initial_contract"],
            "optimizer_scope_is_high_level_only":
            optimizer_scope["optimizer_equals_trainable"]
            and optimizer_scope["optimizer_frozen_overlap_count"] == 0,
            "all_accepted_candidates_pass_every_gate": all(
                not row["accepted"] or all(row["gates"].values()) for row in candidates
            ),
            "all_rejected_candidates_exactly_restored": all(
                row["accepted"] or (row["restoration"] and row["restoration"]["exact"])
                for row in candidates
            ),
            "candidate_two_only_after_candidate_one_acceptance":
            len(candidates) < 2 or candidates[0]["accepted"],
            "final_state_equals_last_accepted_state": final_hashes == expected_final_hashes,
            "frozen_parameters_unchanged": final_names == frozen_names
            and state_sha256(frozen_final) == frozen_initial_hash,
            "collection_inputs_unchanged": state_sha256(batch["replay_inputs"])
            == batch["input_sha256"],
            "reward_and_outcomes_nonselective":
            protocol["reward_path_and_collection_outcomes_are_descriptive_only"]
            and protocol["outcome_based_seed_update_or_threshold_selection"] is False,
            "zero_behavior_checkpoint_val_unseen_selection":
            protocol["behavior_evaluation_episode_count"] == 0
            and protocol["checkpoint_selection"] is False
            and protocol["val_unseen_episode_count"] == 0,
        }
        protocol_valid = all(guards.values())
        repeatability_member = (
            protocol_valid
            and len(candidates) == 2
            and candidates[0]["accepted"]
            and not candidates[1]["accepted"]
            and candidates[1]["restoration"]["exact"]
        )
        if not protocol_valid:
            outcome = "protocol_invalid"
        elif repeatability_member:
            outcome = "candidate_one_committed_candidate_two_rejected_restored"
        elif candidates and not candidates[0]["accepted"]:
            outcome = "candidate_one_rejected_restored"
        elif len(candidates) == 2 and candidates[1]["accepted"]:
            outcome = "two_candidates_committed_within_bounds"
        else:
            outcome = "valid_atomic_guard_other"
        result = {
            "kind": config["kind"],
            "seed": args.seed,
            "all_protocol_guards_passed": protocol_valid,
            "outcome": outcome,
            "repeatability_member": repeatability_member,
            "navigation_or_generalization_claim": False,
            "source_contract": manifest["source_contract"],
            "trainable_manifest": trainable_manifest,
            "optimizer_scope": optimizer_scope,
            "optimizer_initial_contract": optimizer_initial,
            "counts": {
                "model_loads": 1,
                "collection_episodes": len(collection),
                "collection_rows": len(all_rows),
                "candidate_updates_attempted": len(candidates),
                "candidate_updates_committed": sum(row["accepted"] for row in candidates),
                "behavior_episodes": 0,
                "off_batch_audit_episodes": 0,
                "val_unseen_episodes": 0,
            },
            "collection": {
                "records_sha256": sha256_file(collection_path),
                "input_sha256": batch["input_sha256"],
                "preupdate_ratio_max_deviation": batch["preupdate_ratio_max_deviation"],
                "descriptive_metrics": aggregate_behavior(collection),
            },
            "initial_high_level_sha256": initial_parameter_hash,
            "state_zero": state_zero,
            "baseline_training_state_hashes": baseline_hashes,
            "candidates": candidates,
            "accepted_training_state_hashes": accepted_hashes,
            "final_training_state_hashes": final_hashes,
            "protocol_guards": guards,
            "g8_ready": False,
            "elapsed_seconds": time.time() - started,
            "evidence": {
                "manifest_sha256": MANIFEST_SHA256,
                "run_config_sha256": sha256_file(config_path),
                "collection_records_sha256": sha256_file(collection_path),
            },
        }
        result_path = output / "result.json"
        write_json(result_path, result)
        print(json.dumps({
            "result": str(result_path),
            "result_sha256": sha256_file(result_path),
            "seed": args.seed,
            "outcome": outcome,
            "repeatability_member": repeatability_member,
            "protocol_valid": protocol_valid,
        }, sort_keys=True))
    finally:
        if manager is not None:
            cleanup_manager(manager)


if __name__ == "__main__":
    main()
