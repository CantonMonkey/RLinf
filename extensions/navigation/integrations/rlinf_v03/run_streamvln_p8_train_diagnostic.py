"""Fail-closed state machine and hard gates for the StreamVLN P8 diagnostic.

The live collector calls :func:`run_diagnostic` with three gate providers.  This
module deliberately contains no PPO entry point and accepts only a frozen
train-only manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import random
import re
import subprocess
import sys
from typing import Callable


MANIFEST_SHA256 = "499c6c25e3f44d3947d600cf87905d58a545e029bbe59d4c09087e09675e3f07"
RESULT_KIND = "streamvln-p8-train-only-diagnostic"
PRELOAD_PROBE_KIND = "streamvln-p8-preload-env-probe"
PRELOAD_OBSERVATION_KEYS = sorted({
    "rgb",
    "instruction_utf8",
    "instruction_utf8_mask",
    "scene_utf8",
    "scene_utf8_mask",
    "episode_utf8",
    "episode_utf8_mask",
    "env_id",
    "episode_serial",
    "episode_step",
    "episode_start",
    "previous_actions",
    "previous_action_mask",
    "terminal_frozen",
})
RLINF_COMMIT = "0505431899574619da86f551bad70b71e0ea2177"
STREAM_COMMIT = "e48f6ff7e9201d93aae003e8f64b04c00cec13bc"
HABITAT_SIM_COMMIT = "f179b584bcd713c5a2a998132211e2cae881d6d1"
MANIFEST_LINE_NORMALIZATION = 'MANIFEST_SHA256 = "<NORMALIZED>"'


class ProtocolError(RuntimeError):
    """The run is invalid and must not receive a completion marker."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_implementation_sha256(path: Path) -> str:
    data = path.read_bytes()
    if path.name == "run_streamvln_p8_train_diagnostic.py":
        text, count = re.subn(
            r'^MANIFEST_SHA256 = [^\r\n]+$',
            MANIFEST_LINE_NORMALIZATION,
            data.decode("utf-8"),
            count=1,
            flags=re.MULTILINE,
        )
        if count != 1:
            raise ProtocolError("P8 runner has no unique MANIFEST_SHA256 line")
        data = text.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def git_head(path: Path) -> str:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={path}", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if result.returncode:
        raise ProtocolError(f"cannot resolve git head for {path}: {result.stderr.strip()}")
    return result.stdout.strip()


def _identity_key(row: dict) -> tuple[int, str, str]:
    return int(row["index"]), str(row["episode_id"]), str(row["scene"])


def validate_manifest(manifest: dict) -> None:
    if manifest.get("schema_version") != 2:
        raise ProtocolError("unsupported P8 manifest schema")
    if manifest.get("stage") != "P8/G8 train-only causal diagnostic":
        raise ProtocolError("unexpected P8 manifest stage")
    if manifest.get("seeds") != {
        "selection": 2026092811,
        "base_wrapper_init": 2026092811,
        "residual_init": 2026092812,
    }:
        raise ProtocolError("unexpected P8 diagnostic seeds")

    source = manifest.get("source", {})
    if source.get("allowed_splits") != ["train"]:
        raise ProtocolError("P8 diagnostic allows only the train split")
    if source.get("val_seen_read") is not False:
        raise ProtocolError("P8 diagnostic manifest permits val_seen access")
    if source.get("val_unseen_read") is not False:
        raise ProtocolError("P8 diagnostic manifest permits val_unseen access")

    episodes = manifest.get("episodes", {})
    tiny = episodes.get("tiny_train", [])
    equivalence = episodes.get("equivalence", [])
    development = episodes.get("train_development", [])
    if len(tiny) != 8 or len(equivalence) != 4 or len(development) != 8:
        raise ProtocolError("P8 diagnostic requires 8/4/8 frozen identities")
    if [_identity_key(row) for row in equivalence] != [
        _identity_key(row) for row in tiny[:4]
    ]:
        raise ProtocolError("equivalence identities must be tiny_train[:4]")
    tiny_scenes = {str(row["scene"]) for row in tiny}
    development_scenes = {str(row["scene"]) for row in development}
    if len(tiny_scenes) != 8 or len(development_scenes) != 8:
        raise ProtocolError("P8 diagnostic identities must use distinct scenes")
    if tiny_scenes.intersection(development_scenes):
        raise ProtocolError("tiny-train and train-development scenes overlap")

    if manifest.get("training", {}).get("ppo_updates") != 0:
        raise ProtocolError("P8 diagnostic must not run PPO")
    expected_runtime = {
        "model_load_count": 1,
        "visible_gpu_count": 1,
        "device": "cuda:0",
        "deterministic_eval": False,
        "context_tokens": 4096,
        "temperature": 1.0,
        "max_primitive_steps": 160,
        "habitat_startup_timeout_s": 180.0,
        "determinism": {
            "seed_python": True,
            "seed_numpy": True,
            "seed_torch_cpu": True,
            "seed_torch_cuda": True,
            "cublas_workspace_config": ":4096:8",
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "torch_use_deterministic_algorithms": False,
            "deterministic_sampling": True,
        },
    }
    if manifest.get("runtime") != expected_runtime:
        raise ProtocolError("P8 runtime contract changed")
    if manifest.get("architecture") != {
        "topology_model_dim": 128,
        "node_feature_dim": 8,
        "edge_feature_dim": 4,
        "max_waypoint_m": 1.25,
        "value_head_trainable": False,
    }:
        raise ProtocolError("P8 architecture contract changed")
    if manifest.get("executor") != {
        "action_chunk_size": 4,
        "action_names": ["STOP", "↑", "←", "→"],
        "action_token_ids": [50669, 76286, 71858, 51018],
        "forward_step_m": 0.25,
        "turn_angle_deg": 15.0,
        "topology_node_radius_m": 0.5,
        "success_radius_m": 3.0,
        "expert_lookahead_m": 1.0,
        "blocked_forward_threshold": 2,
        "recover_target": "parent",
    }:
        raise ProtocolError("P8 executor contract changed")
    training = manifest.get("training", {})
    fixed_training = {
        "expert_max_decisions_per_episode": 32,
        "warmup_updates": 500,
        "dagger_max_decisions_per_episode": 20,
        "final_updates": 1500,
        "batch_size": 32,
        "learning_rate": 0.0001,
        "weight_decay": 0.0,
        "metric_interval_updates": 25,
        "consecutive_gate_b_passes": 2,
        "duplicate_rows_by_state_hash": False,
        "loss": {
            "token": {"coefficient": 1.0},
            "intent": {"coefficient": 1.0},
            "waypoint": {"coefficient": 1.0, "type": "SmoothL1Loss", "beta": 1.0, "reduction": "mean"},
        },
        "optimizer": {"type": "AdamW", "betas": [0.9, 0.999], "eps": 1e-8, "amsgrad": False, "gradient_clip": None, "scheduler": None},
        "sampling": {"method": "torch.randint", "replacement": True},
        "class_weights": {
            "method": "inverse_frequency",
            "normalization": "total_rows_divided_by_present_class_count_times_class_count",
            "phase_sources": {"warmup": "deduplicated_expert_rows", "final": "deduplicated_expert_plus_dagger_rows"},
            "computed_once_per_phase": True,
            "row_duplication": False,
        },
        "stopping": {"rule": "first_two_consecutive_passing_checks", "check_interval_updates": 25, "only_stopping_point": True},
        "value_head_trainable": False,
        "trainable_precision": "fp32",
        "frozen_backbone_precision": "bf16",
        "ppo_updates": 0,
    }
    if training != fixed_training:
        raise ProtocolError("P8 training contract changed")
    if manifest.get("released_inputs", {}).get("adapter_checkpoint", "missing") is not None:
        raise ProtocolError("P8 adapter checkpoint must be null")
    for name, expected_count in (("base_model", 10), ("tokenizer", 7), ("vision_tower", 3)):
        inventory = manifest.get("released_inputs", {}).get(name, {})
        if inventory.get("root") != manifest.get("paths", {}).get(name):
            raise ProtocolError(f"P8 {name} root differs from path contract")
        files = inventory.get("files", [])
        if len(files) != expected_count or len({row.get("relative_path") for row in files}) != expected_count:
            raise ProtocolError(f"P8 {name} file inventory is incomplete")
        if any(int(row.get("bytes", -1)) < 0 or len(str(row.get("sha256", ""))) != 64 for row in files):
            raise ProtocolError(f"P8 {name} file identity is invalid")
    if manifest.get("irreversibility") != {
        "protocol_changes_allowed_until": "before_first_base_model_load",
        "scientific_failure_rerun_allowed": False,
        "unique_live_diagnostic": True,
    }:
        raise ProtocolError("P8 irreversibility contract changed")
    commits = manifest.get("source", {}).get("commits", {})
    if commits != {"rlinf": RLINF_COMMIT, "streamvln": STREAM_COMMIT, "habitat_sim": HABITAT_SIM_COMMIT}:
        raise ProtocolError("P8 source commits changed")
    scene_assets = episodes.get("selected_scene_assets", [])
    if len(scene_assets) != 16 or {row["scene"] for row in scene_assets} != tiny_scenes | development_scenes:
        raise ProtocolError("P8 selected scene asset inventory changed")
    for row in scene_assets:
        for kind in ("glb", "navmesh"):
            record = row.get(kind, {})
            if not record.get("path") or int(record.get("bytes", -1)) < 0 or len(str(record.get("sha256", ""))) != 64:
                raise ProtocolError(f"invalid P8 {kind} asset identity")
    implementation = manifest.get("implementation", {})
    if implementation.get("manifest_sha_line_replacement") != MANIFEST_LINE_NORMALIZATION or len(implementation.get("files", [])) != 15:
        raise ProtocolError("P8 implementation inventory is incomplete")
    boundary = manifest.get("claim_boundary", {})
    required_false = (
        "val_seen_access",
        "val_unseen_access",
        "ppo",
        "ttt_or_secondary_post_training",
        "robot_execution",
        "remote_push",
    )
    if any(boundary.get(key) is not False for key in required_false):
        raise ProtocolError("P8 diagnostic claim boundary changed")
    if boundary.get("train_only") is not True:
        raise ProtocolError("P8 diagnostic is not marked train-only")


def validate_split_counts(payload: dict) -> dict[str, int]:
    counts = payload.get("split_read_counts")
    if not isinstance(counts, dict):
        raise ProtocolError("gate evidence lacks split_read_counts")
    expected_splits = {"train", "val_seen", "val_unseen"}
    if set(counts) != expected_splits:
        raise ProtocolError("gate split_read_counts keys changed")
    normalized = {
        name: _strict_int(counts[name], f"split_read_counts.{name}")
        for name in sorted(expected_splits)
    }
    if any(value < 0 for value in normalized.values()):
        raise ProtocolError("negative split read count")
    if normalized["val_seen"] != 0 or normalized["val_unseen"] != 0:
        raise ProtocolError("P8 diagnostic accessed a validation split")
    return normalized


def _strict_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ProtocolError(f"gate field must be an integer: {label}")
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ProtocolError(f"invalid integer gate field: {label}") from error


def _strict_finite_real(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ProtocolError(f"gate field must be numeric: {label}")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ProtocolError(f"invalid numeric gate field: {label}") from error
    if not math.isfinite(number):
        raise ProtocolError(f"non-finite gate metric: {label}")
    return number


def _finite_number(payload: dict, key: str) -> float:
    if key not in payload:
        raise ProtocolError(f"gate evidence lacks metric: {key}")
    return _strict_finite_real(payload[key], key)


def adjudicate_gate_a(payload: dict, thresholds: dict) -> dict:
    split_counts = validate_split_counts(payload)
    required_mismatch_fields = (
        "base_logits",
        "sampled_tokens",
        "primitive_actions",
        "proposed_prefix",
        "executed_prefix",
        "stop_position",
        "termination",
        "terminal_metrics",
    )
    mismatch_counts = payload.get("mismatch_counts", {})
    if not isinstance(mismatch_counts, dict):
        raise ProtocolError("Gate A mismatch_counts must be an object")
    missing = [key for key in required_mismatch_fields if key not in mismatch_counts]
    if missing:
        raise ProtocolError(f"Gate A lacks mismatch fields: {missing}")
    normalized_mismatches = {
        key: _strict_int(mismatch_counts[key], f"mismatch_counts.{key}")
        for key in required_mismatch_fields
    }
    zero_mismatch = all(value == 0 for value in normalized_mismatches.values())
    hash_scope = payload.get("parameter_hash_scope", {})
    backbone_before = str(payload.get("backbone_full_hash_before", ""))
    backbone_after = str(payload.get("backbone_full_hash_after", ""))
    residual_before = str(payload.get("residual_hash_before", ""))
    residual_after = str(payload.get("residual_hash_after", ""))
    checks = {
        "episode_count": _strict_int(payload.get("episode_count"), "episode_count") == 4,
        "all_required_mismatches_zero": zero_mismatch,
        "optimizer_steps_zero": _strict_int(payload.get("optimizer_steps"), "optimizer_steps")
        == int(thresholds["optimizer_steps"]),
        "parameter_hash_unchanged": payload.get("parameter_hash_unchanged")
        is thresholds["parameter_hash_unchanged"],
        "model_load_count_one": _strict_int(payload.get("model_load_count"), "model_load_count") == 1,
        "independent_release_reference": payload.get("release_reference")
        == "pre_refactor_sampler_at_5434ab0",
        "complete_backbone_hash_scope": hash_scope.get("backbone")
        == "complete_named_parameters_and_buffers",
        "backbone_full_hash_unchanged": len(backbone_before) == 64
        and backbone_before == backbone_after,
        "residual_full_hash_unchanged": len(residual_before) == 64
        and residual_before == residual_after,
        "train_only": split_counts["val_seen"] == 0
        and split_counts["val_unseen"] == 0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "mismatch_counts": normalized_mismatches,
        "split_read_counts": split_counts,
    }


def adjudicate_gate_b(payload: dict, thresholds: dict) -> dict:
    split_counts = validate_split_counts(payload)
    token_accuracy = _finite_number(payload, "token_top1_accuracy")
    navigate_intent_recall = _finite_number(payload, "navigate_intent_recall")
    stop_intent_recall = _finite_number(payload, "stop_intent_recall")
    stop_token_recall = _finite_number(payload, "stop_token_recall")
    waypoint_error = _finite_number(payload, "waypoint_mean_l2_m")
    ratio_deviation = _finite_number(payload, "sample_replay_ratio_deviation")
    recover_positive_count = _strict_int(
        payload.get("recover_positive_count"), "recover_positive_count"
    )
    consecutive_passes = _strict_int(
        payload.get("consecutive_passes"), "consecutive_passes"
    )
    recover_recall = payload.get("recover_recall")
    if recover_positive_count > 0:
        if recover_recall is None:
            raise ProtocolError("Gate B RECOVER positives lack finite recall")
        recover_recall = _strict_finite_real(recover_recall, "recover_recall")
        recover_reporting_valid = True
    else:
        recover_reporting_valid = recover_recall in (None, "not_applicable")
    expected_names = payload.get("expected_trainable_parameter_names", [])
    optimizer_names = payload.get("optimizer_parameter_names", [])
    finite_gradient_names = payload.get("finite_gradient_names", [])
    nonzero_gradient_names = payload.get("nonzero_gradient_names", [])
    parameter_change_checks = payload.get("parameter_change_checks", {})
    name_lists = (expected_names, optimizer_names, finite_gradient_names, nonzero_gradient_names)
    if (
        any(not isinstance(names, list) for names in name_lists)
        or any(any(not isinstance(name, str) or not name for name in names) for names in name_lists)
        or not expected_names
        or len(expected_names) != len(set(expected_names))
        or not isinstance(parameter_change_checks, dict)
        or any(not isinstance(name, str) or not name for name in parameter_change_checks)
    ):
        raise ProtocolError("Gate B trainable parameter inventory is malformed")
    exact_trainable_evidence = (
        sorted(optimizer_names) == sorted(expected_names)
        and sorted(finite_gradient_names) == sorted(expected_names)
        and sorted(nonzero_gradient_names) == sorted(expected_names)
        and sorted(parameter_change_checks) == sorted(expected_names)
        and all(value is True for value in parameter_change_checks.values())
    )

    checks = {
        "token_top1_accuracy": token_accuracy >= float(thresholds["token_top1_accuracy_min"]),
        "navigate_intent_recall": navigate_intent_recall >= float(thresholds["navigate_intent_recall_min"]),
        "stop_intent_recall": stop_intent_recall >= float(thresholds["stop_intent_recall_min"]),
        "stop_token_recall": stop_token_recall >= float(thresholds["stop_token_recall_min"]),
        "waypoint_mean_l2_m": waypoint_error <= float(thresholds["waypoint_mean_l2_m_max"]),
        "sample_replay_ratio": ratio_deviation
        <= float(thresholds["sample_replay_ratio_deviation_max"]),
        "consecutive_passes": consecutive_passes
        >= int(thresholds["consecutive_passes"]),
        "trainable_tensors_changed": payload.get("trainable_tensors_changed") is True,
        "exact_trainable_evidence": exact_trainable_evidence,
        "optimizer_scope_exact": payload.get("optimizer_scope_exact") is True,
        "gradient_scope_exact": payload.get("gradient_scope_exact") is True,
        "all_expected_trainables_changed": payload.get("all_expected_trainables_changed") is True,
        "required_trainable_groups_changed": payload.get("required_trainable_groups_changed") is True,
        "value_head_frozen_unchanged": payload.get("value_head_frozen_unchanged") is True,
        "undeclared_tensors_unchanged": payload.get("undeclared_tensors_unchanged") is True,
        "frozen_backbone_hash_unchanged": payload.get("frozen_backbone_hash_unchanged") is True,
        "complete_backbone_hash_scope": payload.get("backbone_hash_scope")
        == "complete_named_parameters_and_buffers",
        "all_finite": payload.get("all_finite") is True,
        "trace_fields_complete": payload.get("trace_fields_complete") is True,
        "recover_reporting_valid": recover_reporting_valid,
        "train_only": split_counts["val_seen"] == 0
        and split_counts["val_unseen"] == 0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "token_top1_accuracy": token_accuracy,
            "navigate_intent_recall": navigate_intent_recall,
            "stop_intent_recall": stop_intent_recall,
            "stop_token_recall": stop_token_recall,
            "waypoint_mean_l2_m": waypoint_error,
            "sample_replay_ratio_deviation": ratio_deviation,
            "consecutive_passes": consecutive_passes,
            "recover_positive_count": recover_positive_count,
            "recover_recall": recover_recall,
        },
        "split_read_counts": split_counts,
    }


def _positive_category_count(counts: dict, label: str) -> int:
    return sum(
        _strict_int(value, f"{label}.{name}") > 0
        for name, value in counts.items()
    )


def _recover_outcome_is_valid(row: object, index: int) -> bool:
    if not isinstance(row, dict):
        raise ProtocolError(f"Gate C recover_outcomes[{index}] must be an object")
    episode_id = row.get("episode_id")
    if not isinstance(episode_id, str) or not episode_id:
        raise ProtocolError(f"Gate C recover_outcomes[{index}] has invalid episode_id")
    decision_index = _strict_int(row.get("decision_index"), f"recover_outcomes[{index}].decision_index")
    blocked_before = _strict_int(
        row.get("consecutive_blocked_before"),
        f"recover_outcomes[{index}].consecutive_blocked_before",
    )
    blocked_after = _strict_int(
        row.get("consecutive_blocked_after"),
        f"recover_outcomes[{index}].consecutive_blocked_after",
    )
    realized_path_m = _strict_finite_real(
        row.get("realized_path_m"), f"recover_outcomes[{index}].realized_path_m"
    )
    return (
        decision_index >= 0
        and row.get("actionable_blocked_before") is True
        and blocked_before >= 2
        and 0 <= blocked_after <= blocked_before
        and math.isfinite(realized_path_m)
        and realized_path_m >= 0.0
    )


def adjudicate_gate_c(payload: dict, thresholds: dict) -> dict:
    split_counts = validate_split_counts(payload)
    learned_successes = _strict_int(payload.get("learned_success_count"), "learned_success_count")
    keep_successes = _strict_int(payload.get("keep_original_success_count"), "keep_original_success_count")
    successful_stops = _strict_int(payload.get("successful_stop_count"), "successful_stop_count")
    stop_count = _strict_int(payload.get("stop_count"), "stop_count")
    left = _strict_int(payload.get("left_turn_count"), "left_turn_count")
    right = _strict_int(payload.get("right_turn_count"), "right_turn_count")
    blocked = _strict_int(payload.get("blocked_state_count"), "blocked_state_count")
    recover = _strict_int(payload.get("recover_trigger_count"), "recover_trigger_count")
    alignment_miss = _strict_int(
        payload.get("recover_alignment_miss_count"), "recover_alignment_miss_count"
    )
    primitive_counts = payload.get("primitive_counts", {})
    mode_counts = payload.get("mode_counts", {})
    if not isinstance(primitive_counts, dict) or not isinstance(mode_counts, dict):
        raise ProtocolError("Gate C primitive_counts and mode_counts must be objects")
    primitive_category_count = _positive_category_count(primitive_counts, "primitive_counts")
    mode_category_count = _positive_category_count(mode_counts, "mode_counts")
    turns = left + right
    recover_outcomes = payload.get("recover_outcomes", [])
    if not isinstance(recover_outcomes, list):
        raise ProtocolError("Gate C recover_outcomes must be a list")
    recover_outcome_validity = [
        _recover_outcome_is_valid(row, index)
        for index, row in enumerate(recover_outcomes)
    ]
    recover_outcomes_valid = all(recover_outcome_validity)
    checkpoint_scope = payload.get("checkpoint_hash_scope", {})
    backbone_before = str(payload.get("backbone_full_hash_before", ""))
    backbone_after = str(payload.get("backbone_full_hash_after", ""))
    residual_before = str(payload.get("residual_full_hash_before", ""))
    residual_after = str(payload.get("residual_full_hash_after", ""))
    handedness_required = turns >= int(thresholds["turn_count_for_handedness_check"])
    handedness_fraction = max(left, right) / turns if turns else 0.0
    handedness_passed = (
        not handedness_required
        or handedness_fraction <= float(thresholds["single_turn_direction_fraction_max"])
    )
    recover_passed = (
        blocked == 0
        or not bool(thresholds["recover_required_when_blocked"])
        or recover >= 1
    )

    checks = {
        "episode_count": _strict_int(payload.get("episode_count"), "episode_count") == 8,
        "nonzero_success": learned_successes
        >= int(thresholds["learned_success_count_min"]),
        "successful_stop": successful_stops
        >= int(thresholds["successful_stop_count_min"]),
        "stop_precision_positive": stop_count > 0 and successful_stops / stop_count > 0,
        "keep_original_noninferiority": learned_successes
        >= keep_successes - int(thresholds["max_success_deficit_vs_keep_original"]),
        "primitive_diversity": primitive_category_count
        >= int(thresholds["primitive_category_count_min"]),
        "mode_diversity": mode_category_count
        >= int(thresholds["decision_mode_count_min"]),
        "handedness": handedness_passed,
        "recover_when_blocked": recover_passed,
        "recover_outcomes_aligned": len(recover_outcomes) == recover
        and recover_outcomes_valid,
        "recover_alignment_miss": alignment_miss
        <= int(thresholds["recover_alignment_miss_max"]),
        "checkpoint_frozen": payload.get("checkpoint_hash_before")
        == payload.get("checkpoint_hash_after")
        and bool(payload.get("checkpoint_hash_before")),
        "complete_checkpoint_hash_scope": checkpoint_scope.get("backbone")
        == "complete_named_parameters_and_buffers"
        and checkpoint_scope.get("residual") == "complete_state_dict",
        "backbone_full_hash_frozen": len(backbone_before) == 64
        and backbone_before == backbone_after,
        "residual_full_hash_frozen": len(residual_before) == 64
        and residual_before == residual_after,
        "train_only": split_counts["val_seen"] == 0
        and split_counts["val_unseen"] == 0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "learned_success_count": learned_successes,
            "keep_original_success_count": keep_successes,
            "successful_stop_count": successful_stops,
            "stop_count": stop_count,
            "stop_precision": successful_stops / stop_count if stop_count > 0 else 0.0,
            "primitive_counts": primitive_counts,
            "mode_counts": mode_counts,
            "left_turn_count": left,
            "right_turn_count": right,
            "single_turn_direction_fraction": handedness_fraction,
            "blocked_state_count": blocked,
            "recover_trigger_count": recover,
            "recover_outcomes": recover_outcomes,
            "recover_alignment_miss_count": alignment_miss,
        },
        "split_read_counts": split_counts,
    }


def validate_preload_probe(payload: object, manifest: dict) -> dict:
    """Validate the zero-model Habitat probe before Gate A may load the 7B model."""

    required_keys = {
        "schema_version",
        "kind",
        "episode_count",
        "scene_count",
        "identities",
        "observation_keys",
        "rgb_shape",
        "rgb_dtype",
        "observation_valid",
        "expert_waypoint_valid",
        "model_load_count",
        "probe_split_read_counts",
        "scientific_split_read_counts",
    }
    if not isinstance(payload, dict) or set(payload) != required_keys:
        raise ProtocolError("P8 pre-load environment probe receipt schema changed")
    if (
        _strict_int(payload["schema_version"], "preload.schema_version") != 1
        or payload["kind"] != PRELOAD_PROBE_KIND
    ):
        raise ProtocolError("P8 pre-load environment probe identity changed")

    expected_rows = list(manifest["episodes"]["tiny_train"]) + list(
        manifest["episodes"]["train_development"]
    )
    expected_identities = [
        {
            "index": int(row["index"]),
            "episode_id": str(row["episode_id"]),
            "scene": str(row["scene"]),
        }
        for row in expected_rows
    ]
    identities = payload["identities"]
    if not isinstance(identities, list) or len(identities) != len(expected_identities):
        raise ProtocolError("P8 pre-load environment probe identity count changed")
    normalized_identities = []
    for index, row in enumerate(identities):
        if not isinstance(row, dict) or set(row) != {"index", "episode_id", "scene"}:
            raise ProtocolError(f"P8 pre-load identity {index} schema changed")
        if not isinstance(row["episode_id"], str) or not row["episode_id"]:
            raise ProtocolError(f"P8 pre-load identity {index} episode is invalid")
        if not isinstance(row["scene"], str) or not row["scene"]:
            raise ProtocolError(f"P8 pre-load identity {index} scene is invalid")
        normalized_identities.append({
            "index": _strict_int(row["index"], f"preload.identities[{index}].index"),
            "episode_id": row["episode_id"],
            "scene": row["scene"],
        })
    if normalized_identities != expected_identities:
        raise ProtocolError("P8 pre-load environment identities differ from the manifest")

    if (
        _strict_int(payload["episode_count"], "preload.episode_count") != 16
        or _strict_int(payload["scene_count"], "preload.scene_count") != 16
        or _strict_int(payload["model_load_count"], "preload.model_load_count") != 0
        or payload["observation_keys"] != PRELOAD_OBSERVATION_KEYS
        or payload["rgb_dtype"] != "torch.uint8"
        or payload["observation_valid"] is not True
        or payload["expert_waypoint_valid"] is not True
    ):
        raise ProtocolError("P8 pre-load environment probe contract failed")
    rgb_shape = payload["rgb_shape"]
    if (
        not isinstance(rgb_shape, list)
        or [_strict_int(value, "preload.rgb_shape") for value in rgb_shape]
        != [1, 64, 64, 3]
    ):
        raise ProtocolError("P8 pre-load RGB shape changed")

    expected_counts = {
        "probe_split_read_counts": {"train": 16, "val_seen": 0, "val_unseen": 0},
        "scientific_split_read_counts": {"train": 0, "val_seen": 0, "val_unseen": 0},
    }
    for field, expected in expected_counts.items():
        counts = payload[field]
        if not isinstance(counts, dict) or set(counts) != set(expected):
            raise ProtocolError(f"P8 {field} schema changed")
        normalized = {
            name: _strict_int(counts[name], f"preload.{field}.{name}")
            for name in expected
        }
        if normalized != expected:
            raise ProtocolError(f"P8 {field} changed")
    return dict(payload)


def _artifact_inventory(output_dir: Path) -> dict:
    files = []
    excluded = {"artifact_inventory.json", "COMPLETED.json"}
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name not in excluded and not path.name.endswith(".tmp"):
            files.append({
                "path": path.relative_to(output_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    return {"schema_version": 1, "files": files}


GateProvider = Callable[[], dict]
PreloadProvider = Callable[[], dict]


def run_diagnostic(
    *,
    manifest_path: Path,
    output_dir: Path,
    expected_manifest_sha256: str,
    gate_a_provider: GateProvider,
    gate_b_provider: GateProvider,
    gate_c_provider: GateProvider,
    preload_provider: PreloadProvider | None = None,
    source_heads: dict | None = None,
    identity: dict | None = None,
) -> dict:
    """Execute the frozen gate ladder and always stop at the first scientific failure."""

    manifest_hash = sha256_file(manifest_path)
    if not expected_manifest_sha256 or expected_manifest_sha256 == "UNPINNED":
        raise ProtocolError("formal P8 manifest hash is not pinned")
    if manifest_hash != expected_manifest_sha256:
        raise ProtocolError("P8 manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    output_dir.mkdir(parents=True, exist_ok=False)

    result = {
        "schema_version": 1,
        "kind": RESULT_KIND,
        "manifest_sha256": manifest_hash,
        "source_heads": source_heads or {},
        "identity": identity or {},
        "model_load_count": 0,
        "preload_probe_split_read_counts": {
            "train": 0,
            "val_seen": 0,
            "val_unseen": 0,
        },
        "split_read_counts": {"train": 0, "val_seen": 0, "val_unseen": 0},
        "completed_gates": [],
        "first_failed_gate": None,
        "gate_a": None,
        "gate_b": None,
        "gate_c": None,
        "eligible_for_full_p8": False,
        "guards": {
            "train_only": True,
            "ppo_updates": 0,
            "val_seen_read": False,
            "val_unseen_read": False,
        },
    }
    write_json_atomic(output_dir / "RUN_STARTED.json", {
        "schema_version": 1,
        "kind": "streamvln-p8-run-started",
        "manifest_sha256": manifest_hash,
        "source_heads": result["source_heads"],
        "identity": result["identity"],
        "model_load_count": 0,
        "preload_env_probe_required": preload_provider is not None,
        "scientific_split_read_counts": dict(result["split_read_counts"]),
    })
    if preload_provider is not None:
        preload_payload = validate_preload_probe(preload_provider(), manifest)
        write_json_atomic(output_dir / "PRELOAD_ENV_PROBE.json", preload_payload)
        result["preload_probe_split_read_counts"] = dict(
            preload_payload["probe_split_read_counts"]
        )

    def finish(decision: str) -> dict:
        result["decision"] = decision
        write_json_atomic(output_dir / "result.json", result)
        write_json_atomic(output_dir / "artifact_inventory.json", _artifact_inventory(output_dir))
        write_json_atomic(output_dir / "COMPLETED.json", {
            "schema_version": 1,
            "kind": RESULT_KIND,
            "manifest_sha256": manifest_hash,
            "status": decision,
            "eligible_for_full_p8": result["eligible_for_full_p8"],
        })
        return result

    gate_a_payload = gate_a_provider()
    write_json_atomic(output_dir / "gate_a_raw.json", gate_a_payload)
    gate_a = adjudicate_gate_a(gate_a_payload, manifest["gates"]["a"])
    result["model_load_count"] = int(gate_a_payload.get("model_load_count", 0))
    result["gate_a"] = gate_a
    result["split_read_counts"] = dict(gate_a["split_read_counts"])
    write_json_atomic(output_dir / "gate_a_equivalence.json", gate_a)
    result["completed_gates"].append("a")
    if not gate_a["passed"]:
        result["first_failed_gate"] = "a"
        return finish("stopped_at_a")

    gate_b_payload = gate_b_provider()
    write_json_atomic(output_dir / "gate_b_raw.json", gate_b_payload)
    gate_b = adjudicate_gate_b(gate_b_payload, manifest["gates"]["b"])
    result["gate_b"] = gate_b
    for split, count in gate_b["split_read_counts"].items():
        result["split_read_counts"][split] += count
    write_json_atomic(output_dir / "gate_b_bc_learnability.json", gate_b)
    result["completed_gates"].append("b")
    if not gate_b["passed"]:
        result["first_failed_gate"] = "b"
        return finish("stopped_at_b")

    gate_c_payload = gate_c_provider()
    write_json_atomic(output_dir / "gate_c_raw.json", gate_c_payload)
    gate_c = adjudicate_gate_c(gate_c_payload, manifest["gates"]["c"])
    result["gate_c"] = gate_c
    for split, count in gate_c["split_read_counts"].items():
        result["split_read_counts"][split] += count
    write_json_atomic(output_dir / "gate_c_train_development.json", gate_c)
    result["completed_gates"].append("c")
    if not gate_c["passed"]:
        result["first_failed_gate"] = "c"
        return finish("stopped_at_c")

    result["eligible_for_full_p8"] = True
    return finish("pass")


def _metadata_identity(python_path: Path, package_names: list[str]) -> dict:
    program = (
        "import importlib.metadata,json,platform;"
        f"names={package_names!r};"
        "print(json.dumps({'python_version':platform.python_version(),"
        "'packages':{name:importlib.metadata.version(name) for name in names}},sort_keys=True))"
    )
    completed = subprocess.run(
        [str(python_path), "-c", program],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if completed.returncode:
        raise ProtocolError(
            f"runtime metadata probe failed for {python_path}: {completed.stderr.strip()}"
        )
    return json.loads(completed.stdout)


def _absolute_existing(path: Path) -> Path:
    expanded = path.expanduser()
    absolute = expanded if expanded.is_absolute() else expanded.resolve()
    if not absolute.exists():
        raise ProtocolError(f"required P8 path does not exist: {absolute}")
    return absolute


def _verify_file_record(path: Path, record: dict, label: str) -> None:
    if path.stat().st_size != int(record["bytes"]):
        raise ProtocolError(f"P8 {label} size changed")
    if sha256_file(path) != str(record["sha256"]):
        raise ProtocolError(f"P8 {label} SHA-256 changed")


def verify_preload_contract(args, manifest: dict) -> dict:
    """Verify all frozen identities without importing torch or loading a model."""

    path_arg_names = (
        "rlinf_root", "stream_source", "base_model", "tokenizer", "vision_tower",
        "asset_root", "rlinf_python", "habitat_python", "habitat_server",
        "habitat_sim_root", "habitat_sim_python", "habitat_env_lib", "egl_vendor_json",
    )
    paths = {}
    for name in path_arg_names:
        value = getattr(args, name, None)
        if value is None:
            raise ProtocolError(f"--{name.replace('_', '-')} is required in live/preflight mode")
        paths[name] = _absolute_existing(value)
        if str(paths[name]) != str(manifest["paths"][name]):
            raise ProtocolError(f"P8 CLI path differs from manifest: {name}")

    cli_values = {
        "device": args.device,
        "context_tokens": args.context_tokens,
        "temperature": args.temperature,
        "deterministic_eval": args.deterministic_eval,
        "habitat_startup_timeout_s": args.habitat_startup_timeout_s,
        "topology_model_dim": args.topology_model_dim,
        "node_feature_dim": args.node_feature_dim,
        "edge_feature_dim": args.edge_feature_dim,
        "max_waypoint_m": args.max_waypoint_m,
        "expert_lookahead_m": args.expert_lookahead_m,
        "base_wrapper_init_seed": args.base_wrapper_init_seed,
        "residual_init_seed": args.residual_init_seed,
    }
    expected_cli = {
        "device": manifest["runtime"]["device"],
        "context_tokens": manifest["runtime"]["context_tokens"],
        "temperature": manifest["runtime"]["temperature"],
        "deterministic_eval": manifest["runtime"]["deterministic_eval"],
        "habitat_startup_timeout_s": manifest["runtime"]["habitat_startup_timeout_s"],
        "topology_model_dim": manifest["architecture"]["topology_model_dim"],
        "node_feature_dim": manifest["architecture"]["node_feature_dim"],
        "edge_feature_dim": manifest["architecture"]["edge_feature_dim"],
        "max_waypoint_m": manifest["architecture"]["max_waypoint_m"],
        "expert_lookahead_m": manifest["executor"]["expert_lookahead_m"],
        "base_wrapper_init_seed": manifest["seeds"]["base_wrapper_init"],
        "residual_init_seed": manifest["seeds"]["residual_init"],
    }
    if cli_values != expected_cli:
        changed = sorted(key for key in cli_values if cli_values[key] != expected_cli[key])
        raise ProtocolError(f"P8 live CLI differs from manifest: {changed}")

    visible = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
    if len(visible) != int(manifest["runtime"]["visible_gpu_count"]):
        raise ProtocolError("P8 requires exactly one CUDA_VISIBLE_DEVICES entry")
    source_heads = {
        "rlinf": git_head(paths["rlinf_root"]),
        "stream": git_head(paths["stream_source"]),
        "habitat_sim": git_head(paths["habitat_sim_root"]),
    }
    if source_heads != {
        "rlinf": RLINF_COMMIT,
        "stream": STREAM_COMMIT,
        "habitat_sim": HABITAT_SIM_COMMIT,
    }:
        raise ProtocolError(f"P8 source head changed: {source_heads}")

    train = Path(manifest["source"]["train_path"])
    _verify_file_record(
        train,
        {"bytes": train.stat().st_size, "sha256": manifest["source"]["train_sha256"]},
        "train dataset",
    )
    checked_model_files = 0
    for name in ("base_model", "tokenizer", "vision_tower"):
        inventory = manifest["released_inputs"][name]
        root = paths[name]
        expected_names = sorted(row["relative_path"] for row in inventory["files"])
        actual_names = sorted(path.name for path in root.iterdir() if path.is_file())
        if actual_names != expected_names:
            raise ProtocolError(f"P8 {name} file set changed")
        for record in inventory["files"]:
            _verify_file_record(root / record["relative_path"], record, f"{name}/{record['relative_path']}")
            checked_model_files += 1

    checked_scene_files = 0
    for scene in manifest["episodes"]["selected_scene_assets"]:
        for kind in ("glb", "navmesh"):
            record = scene[kind]
            _verify_file_record(Path(record["path"]), record, f"scene {scene['scene']} {kind}")
            checked_scene_files += 1

    for name in ("rlinf", "habitat"):
        expected = manifest["runtime_identity"][name]
        python_path = paths[f"{name}_python"]
        if sha256_file(python_path) != expected["python_sha256"]:
            raise ProtocolError(f"P8 {name} Python SHA-256 changed")
        actual = _metadata_identity(python_path, list(expected["packages"]))
        wanted = {"python_version": expected["python_version"], "packages": expected["packages"]}
        if actual != wanted:
            raise ProtocolError(f"P8 {name} runtime versions changed: {actual}")
    if os.path.abspath(sys.executable) != str(paths["rlinf_python"]):
        raise ProtocolError("P8 runner is not executing under the frozen RLinf Python")
    if sha256_file(paths["egl_vendor_json"]) != manifest["runtime_identity"]["egl_vendor_json_sha256"]:
        raise ProtocolError("P8 EGL vendor JSON changed")

    implementation = []
    for record in manifest["implementation"]["files"]:
        path = _absolute_existing(Path(record["path"]))
        actual = normalized_implementation_sha256(path)
        if actual != record["sha256"]:
            raise ProtocolError(f"P8 implementation changed: {record['logical_name']}")
        implementation.append({"logical_name": record["logical_name"], "sha256": actual})
    if "torch" in sys.modules:
        raise ProtocolError("pre-load verification imported torch")
    return {
        "source_heads": source_heads,
        "identity": {
            "cli": cli_values,
            "cuda_visible_devices": visible,
            "runtime_identity": manifest["runtime_identity"],
            "released_input_file_count": checked_model_files,
            "selected_scene_file_count": checked_scene_files,
            "implementation": implementation,
            "torch_imported": "torch" in sys.modules,
            "model_constructed": False,
        },
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true")
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--adjudicate", action="store_true")
    parser.add_argument("--gate-a-input", type=Path)
    parser.add_argument("--gate-b-input", type=Path)
    parser.add_argument("--gate-c-input", type=Path)
    parser.add_argument("--rlinf-root", type=Path)
    parser.add_argument("--stream-source", type=Path)
    parser.add_argument("--base-model", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--vision-tower", type=Path)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--rlinf-python", type=Path)
    parser.add_argument("--habitat-python", type=Path)
    parser.add_argument("--habitat-server", type=Path)
    parser.add_argument("--habitat-sim-python", type=Path)
    parser.add_argument("--habitat-env-lib", type=Path)
    parser.add_argument("--habitat-sim-root", type=Path)
    parser.add_argument("--egl-vendor-json", type=Path)
    parser.add_argument("--habitat-startup-timeout-s", type=float, default=180.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--context-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--topology-model-dim", type=int, default=128)
    parser.add_argument("--node-feature-dim", type=int, default=8)
    parser.add_argument("--edge-feature-dim", type=int, default=4)
    parser.add_argument("--max-waypoint-m", type=float, default=1.25)
    parser.add_argument("--expert-lookahead-m", type=float, default=1.0)
    parser.add_argument("--base-wrapper-init-seed", type=int, default=2026092811)
    parser.add_argument("--residual-init-seed", type=int, default=2026092812)
    parser.add_argument("--deterministic-eval", action="store_true", default=False)
    return parser.parse_args(argv)


def _json_provider(path: Path) -> GateProvider:
    return lambda: json.loads(path.read_text(encoding="utf-8"))


def _required_paths(args, names: tuple[str, ...]) -> dict[str, Path]:
    resolved = {}
    for name in names:
        value = getattr(args, name)
        if value is None:
            raise ProtocolError(f"--{name.replace('_', '-')} is required in live mode")
        # Preserve venv interpreter entrypoints.  Resolving their symlinks
        # launches the base interpreter and discards the venv package context.
        resolved[name] = _absolute_existing(value)
    return resolved


def _live_providers(args, manifest: dict, output_dir: Path):
    determinism = manifest["runtime"]["determinism"]
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = determinism["cublas_workspace_config"]
    base_seed = int(manifest["seeds"]["base_wrapper_init"])
    random.seed(base_seed)
    import numpy as np
    np.random.seed(base_seed % (2 ** 32))
    import torch
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed_all(base_seed)
    torch.backends.cudnn.deterministic = bool(determinism["cudnn_deterministic"])
    torch.backends.cudnn.benchmark = bool(determinism["cudnn_benchmark"])
    torch.use_deterministic_algorithms(bool(determinism["torch_use_deterministic_algorithms"]))

    try:
        from .streamvln_p8_live_diagnostic import LiveP8DiagnosticProviders
        from .streamvln_p8_residual_policy import StreamVLNPolicyPreservingResidual
        from .streamvln_ppo_policy import StreamVLNPPOPolicy
    except ImportError:
        from streamvln_p8_live_diagnostic import LiveP8DiagnosticProviders
        from streamvln_p8_residual_policy import StreamVLNPolicyPreservingResidual
        from streamvln_ppo_policy import StreamVLNPPOPolicy

    paths = _required_paths(args, (
        "rlinf_root",
        "stream_source",
        "base_model",
        "tokenizer",
        "vision_tower",
        "asset_root",
        "habitat_python",
        "habitat_server",
        "habitat_sim_python",
        "habitat_env_lib",
        "egl_vendor_json",
    ))
    if git_head(paths["rlinf_root"]) != RLINF_COMMIT:
        raise ProtocolError("live P8 RLinf source commit changed")
    if git_head(paths["stream_source"]) != STREAM_COMMIT:
        raise ProtocolError("live P8 StreamVLN source commit changed")
    if not torch.cuda.is_available():
        raise ProtocolError("live P8 requires one CUDA device")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ProtocolError("live P8 base model must run on CUDA")

    load_counter = {"base": 0}

    def load_base_policy():
        load_counter["base"] += 1
        if load_counter["base"] != 1:
            raise ProtocolError("live P8 attempted a second base-model load")
        policy = StreamVLNPPOPolicy(
            stream_source=str(paths["stream_source"]),
            base_model=str(paths["base_model"]),
            tokenizer=str(paths["tokenizer"]),
            vision_tower=str(paths["vision_tower"]),
            expected_source_commit=STREAM_COMMIT,
            adapter_checkpoint=None,
            trainable=False,
            temperature=float(args.temperature),
            context_tokens=int(args.context_tokens),
            seed=base_seed,
            deterministic_eval=bool(manifest["runtime"]["deterministic_eval"]),
        )
        policy.to(device)
        policy.eval()
        policy.requires_grad_(False)
        if getattr(policy, "source_kind", None) != "upstream-original":
            raise ProtocolError("live P8 base policy is not the original released model")
        if any(parameter.requires_grad for parameter in policy.parameters()):
            raise ProtocolError("live P8 base policy is not fully frozen")
        return policy

    def load_residual_policy(base_policy):
        session = base_policy._ensure_session()
        if session.model is not base_policy.backbone:
            raise ProtocolError("live P8 session does not share the unique base model")
        hidden_size = int(session.model.lm_head.weight.shape[1])
        residual_seed = int(manifest["seeds"]["residual_init"])
        random.seed(residual_seed)
        np.random.seed(residual_seed % (2 ** 32))
        torch.manual_seed(residual_seed)
        torch.cuda.manual_seed_all(residual_seed)
        residual = StreamVLNPolicyPreservingResidual(
            token_hidden_dim=hidden_size,
            node_hidden_dim=hidden_size,
            node_feature_dim=int(manifest["architecture"]["node_feature_dim"]),
            edge_feature_dim=int(manifest["architecture"]["edge_feature_dim"]),
            model_dim=int(manifest["architecture"]["topology_model_dim"]),
            max_waypoint_m=float(manifest["architecture"]["max_waypoint_m"]),
        ).to(device=device, dtype=torch.float32)
        if load_counter["base"] != 1:
            raise ProtocolError("residual construction changed the base-model load count")
        return residual

    run_cfg = {
        "seed": int(manifest["seeds"]["selection"]),
        "asset_root": str(paths["asset_root"]),
        "habitat_python": str(paths["habitat_python"]),
        "habitat_server": str(paths["habitat_server"]),
        "habitat_sim_python": str(paths["habitat_sim_python"]),
        "habitat_env_lib": str(paths["habitat_env_lib"]),
        "egl_vendor_json": str(paths["egl_vendor_json"]),
        "habitat_startup_timeout_s": float(args.habitat_startup_timeout_s),
        "output_dir": str(output_dir),
    }
    providers = LiveP8DiagnosticProviders(
        manifest=manifest,
        run_cfg=run_cfg,
        base_policy_loader=load_base_policy,
        residual_policy_loader=load_residual_policy,
    )
    return providers, {
        "stream": STREAM_COMMIT,
        "rlinf": RLINF_COMMIT,
        "base_model_load_contract": "exactly_one",
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        manifest_path = args.manifest.resolve(strict=True)
        output_dir = args.output_dir.resolve()
        manifest_hash = sha256_file(manifest_path)
        if not MANIFEST_SHA256 or MANIFEST_SHA256 == "UNPINNED":
            raise ProtocolError("formal P8 manifest hash is not pinned")
        if manifest_hash != MANIFEST_SHA256:
            raise ProtocolError("P8 manifest hash mismatch")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_manifest(manifest)
        if args.live or args.preflight_only:
            preload = verify_preload_contract(args, manifest)
        if args.preflight_only:
            output_dir.mkdir(parents=True, exist_ok=False)
            receipt = {
                "schema_version": 1,
                "kind": "streamvln-p8-preload-preflight",
                "manifest_sha256": manifest_hash,
                **preload,
                "torch_imported": False,
                "provider_constructed": False,
                "model_load_count": 0,
            }
            write_json_atomic(output_dir / "PRELOAD_PREFLIGHT.json", receipt)
            print(json.dumps(receipt, indent=2, sort_keys=True))
            return 0
        if args.live:
            providers, source_heads = _live_providers(args, manifest, output_dir)
            source_heads = preload["source_heads"]
            identity = preload["identity"]
            preload_provider = providers.preload_env_probe
            gate_a_provider = providers.gate_a_provider
            gate_b_provider = providers.gate_b_provider
            gate_c_provider = providers.gate_c_provider
        else:
            missing = [
                name
                for name in ("gate_a_input", "gate_b_input", "gate_c_input")
                if getattr(args, name) is None
            ]
            if missing:
                raise ProtocolError(
                    "adjudicate mode requires all three gate input files: "
                    + ", ".join(missing)
                )
            source_heads = {}
            identity = {}
            preload_provider = None
            gate_a_provider = _json_provider(args.gate_a_input.resolve(strict=True))
            gate_b_provider = _json_provider(args.gate_b_input.resolve(strict=True))
            gate_c_provider = _json_provider(args.gate_c_input.resolve(strict=True))
        result = run_diagnostic(
            manifest_path=manifest_path,
            output_dir=output_dir,
            expected_manifest_sha256=MANIFEST_SHA256,
            gate_a_provider=gate_a_provider,
            gate_b_provider=gate_b_provider,
            gate_c_provider=gate_c_provider,
            preload_provider=preload_provider,
            source_heads=source_heads,
            identity=identity,
        )
    except ProtocolError as error:
        print(json.dumps({"protocol_error": str(error)}, sort_keys=True))
        return 2
    print(json.dumps({
        "decision": result["decision"],
        "eligible_for_full_p8": result["eligible_for_full_p8"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
