"""Read-only train-development causal probe for the failed P8 BC candidate.

The probe reuses the frozen BC-readiness provider and changes only which half
of a deterministic decision is executed.  It never trains, promotes, imports
validation data, or starts a CUDA worker/PPO path.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Callable, Mapping, Protocol, Sequence


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import run_streamvln_p8_successor_bc_readiness as readiness


SCHEMA_NAME = "streamvln_p8_successor_causal_probe"
SCHEMA_VERSION = 1
MANIFEST_SCHEMA_NAME = "streamvln_p8_successor_causal_probe_manifest"
TRAIN_SPLIT = "train"
ARM_ROUTE = "learned_route_stop_oracle_waypoint"
ARM_WAYPOINT = "oracle_route_stop_learned_waypoint"
ARMS = (ARM_ROUTE, ARM_WAYPOINT)
EXPECTED_EPISODES = 8
EXPECTED_TRAIN_READS = EXPECTED_EPISODES * (len(ARMS) + 1)
EXPECTED_READINESS_MANIFEST_RAW_SHA256 = (
    "61396c099763a3c340ec66056e9fce5244baffc5e6c7ee3fb4f852b984f05ae2"
)
EXPECTED_READINESS_MANIFEST_CANONICAL_SHA256 = (
    "d340a3b9c1db5eb80465dc996f93561ecde3ec3ff2f8648fddd0367966421e80"
)
EXPECTED_READINESS_RECEIPT_FILE_SHA256 = (
    "3dcd3c10935e7d15d45858d6a73f5d2d32cb2aab95002818137de05e1c799fa3"
)
EXPECTED_READINESS_RECEIPT_SHA256 = (
    "1f90168ec22a24e29797e259ccbfa20d5f92dee55bc1e6ffb1d60a328641f25b"
)
EXPECTED_CANDIDATE_SHA256 = (
    "6f8826594bdee7029c20efd1fa393cbf411b0d075fa5b2c0a15f243d0e4291f7"
)
EXPECTED_POLICY_STATE_SHA256 = (
    "a324dbbf7d895e41c85704780f981eaaf99d335e9a5d75d039b5bfb179d94f48"
)


ProtocolError = readiness.ProtocolError
_ORIGINAL_LABEL_SUCCESSOR_ROUTE = readiness.label_successor_route


def _strict_keys(value: Mapping, expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ProtocolError(
            f"{name} keys drifted: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def _identity(value: object, name: str) -> dict:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{name} must be an object")
    row = dict(value)
    _strict_keys(row, {"episode_id", "index", "scene"}, name)
    if not isinstance(row["episode_id"], str) or not row["episode_id"]:
        raise ProtocolError(f"{name}.episode_id must be non-empty")
    if type(row["index"]) is not int or row["index"] < 0:
        raise ProtocolError(f"{name}.index must be a nonnegative integer")
    if not isinstance(row["scene"], str) or not row["scene"]:
        raise ProtocolError(f"{name}.scene must be non-empty")
    return row


def validate_probe_manifest(value: object, readiness_manifest: Mapping) -> dict:
    """Bind the diagnostic exactly to the negative v5 readiness lineage."""

    if not isinstance(value, Mapping):
        raise ProtocolError("causal probe manifest root must be an object")
    manifest = dict(value)
    _strict_keys(
        manifest,
        {
            "arms",
            "attribution_rule",
            "claim_boundary",
            "created",
            "development_episodes",
            "diagnostics",
            "implementation",
            "protocol",
            "readiness_binding",
            "schema_name",
            "schema_version",
            "source_contract",
            "split_contract",
            "stage",
            "thresholds",
        },
        "causal probe manifest",
    )
    if (
        manifest["schema_name"] != MANIFEST_SCHEMA_NAME
        or manifest["schema_version"] != 1
        or manifest["stage"] != "P8 successor train-only causal feature probe"
    ):
        raise ProtocolError("unsupported causal probe manifest")

    split = manifest["split_contract"]
    if split != {
        "allowed_splits": [TRAIN_SPLIT],
        "train_only": True,
        "val_seen_reads": 0,
        "val_unseen_reads": 0,
    }:
        raise ProtocolError("causal probe split contract drifted")
    expected_boundary = {
        "bc_promotion": False,
        "causal_diagnostic_only": True,
        "cuda_worker_smoke": False,
        "g9": False,
        "ppo": False,
        "remote_push": False,
        "training": False,
        "validation_split_access": False,
    }
    if manifest["claim_boundary"] != expected_boundary:
        raise ProtocolError("causal probe claim boundary drifted")
    expected_arms = {
        ARM_ROUTE: {
            "route_and_stop_source": "deterministic_candidate_argmax",
            "true_stop_predicted_waypoint_fallback": (
                "candidate_deterministic_mean; counted explicitly and never "
                "described as oracle"
            ),
            "waypoint_source_when_waypoint": (
                "train_shortest_path_expert_when_available"
            ),
        },
        ARM_WAYPOINT: {
            "route_and_stop_source": "train_shortest_path_expert_route_label",
            "waypoint_source_when_waypoint": "deterministic_candidate_mean",
        },
    }
    if manifest["arms"] != expected_arms:
        raise ProtocolError("causal probe arms drifted")
    protocol = manifest["protocol"]
    if protocol != {
        "decision_budget_per_episode": 160,
        "deterministic": True,
        "episode_count_per_arm": EXPECTED_EPISODES,
        "physical_budget_per_episode": 160,
        "reuse_readiness_provider": True,
        "single_model_load": True,
        "teacher_forced_decision_budget_per_episode": 160,
        "teacher_forced_episode_count": EXPECTED_EPISODES,
        "teacher_forced_physical_budget_per_episode": 160,
    }:
        raise ProtocolError("causal probe budget or execution protocol drifted")
    expected_diagnostics = {
        "compiled_first_action": True,
        "counterfactual_at_each_hybrid_state": True,
        "route_confusion": True,
        "route_arm_true_stop_predicted_waypoint_fallback": {
            "count_field": (
                "true_stop_predicted_waypoint_candidate_fallback_count"
            ),
            "execution_source": "candidate_deterministic_mean",
            "is_oracle": False,
        },
        "stop_precision_recall": True,
        "teacher_forced_on_expert_rollout_states": True,
        "waypoint_bearing_and_distance_error": True,
        "zero_forward_macro": True,
    }
    if manifest["diagnostics"] != expected_diagnostics:
        raise ProtocolError("causal probe diagnostics drifted")
    expected_thresholds = {
        "arm_recovered": {
            "episode_count": EXPECTED_EPISODES,
            "forward_nonstop_fraction_min": 0.295,
            "root_recovery_unavailable_count_max": 0,
            "success_count_min": 4,
            "successful_stop_count_min": 1,
        },
        "teacher_route": {
            "route_nonstop_bal_acc_min": 0.65,
            "premature_stop_count_max": 1,
            "stop_recall_min": 0.5,
        },
        "teacher_sample_min": {
            "keep_original": 8,
            "stop": 4,
            "waypoint": 8,
        },
        "teacher_waypoint": {
            "bearing_abs_error_median_deg_max": 15.0,
            "bearing_abs_error_p90_deg_max": 45.0,
            "first_action_agreement_min": 0.65,
            "percentile_method": "nearest_rank",
            "vector_l2_median_m_max": 0.25,
        },
    }
    if manifest["thresholds"] != expected_thresholds:
        raise ProtocolError("causal probe thresholds drifted")
    expected_attribution_rule = {
        "authorize_feature_v2_when": (
            "sample_sufficient and not "
            "learned_route_stop_oracle_waypoint.recovered and not "
            "oracle_route_stop_learned_waypoint.recovered and not "
            "route_group_pass and not waypoint_group_pass"
        ),
        "labels": {
            "both_arms_recovered": "both_component_substitutions_recover",
            "feature_v2": "joint_teacher_failure_authorizes_feature_v2",
            "insufficient": "insufficient_teacher_samples",
            "neither_group_failed": "closed_loop_interaction_or_distribution_shift",
            "route_arm_only": "waypoint_branch_failure_supported",
            "route_group_only": (
                "route_stop_failure_without_closed_loop_recovery"
            ),
            "waypoint_arm_only": "route_stop_branch_failure_supported",
            "waypoint_group_only": (
                "waypoint_failure_without_closed_loop_recovery"
            ),
        },
        "otherwise_authorize_feature_v2": False,
    }
    if manifest["attribution_rule"] != expected_attribution_rule:
        raise ProtocolError("causal probe attribution rule drifted")
    implementation = manifest["implementation"]
    if not isinstance(implementation, Mapping):
        raise ProtocolError("causal probe implementation binding is missing")
    _strict_keys(
        implementation,
        {"runner_bytes", "runner_relative_path", "runner_sha256"},
        "causal probe implementation",
    )
    if (
        implementation["runner_relative_path"]
        != "integrations/rlinf_v03/run_streamvln_p8_successor_causal_probe.py"
        or type(implementation["runner_bytes"]) is not int
        or implementation["runner_bytes"] < 1
    ):
        raise ProtocolError("causal probe runner path/size binding drifted")
    readiness._sha(
        implementation["runner_sha256"], "implementation.runner_sha256"
    )

    readiness_contract = readiness.validate_protocol_manifest(readiness_manifest)
    development = [
        _identity(row, f"development_episodes[{index}]")
        for index, row in enumerate(manifest["development_episodes"])
    ]
    if len(development) != EXPECTED_EPISODES:
        raise ProtocolError("causal probe requires exactly eight development episodes")
    if development != readiness_contract["development"]:
        raise ProtocolError("causal probe episodes differ from v5 train-development")
    if len({row["episode_id"] for row in development}) != EXPECTED_EPISODES:
        raise ProtocolError("causal probe episode identities are not unique")
    if len({row["scene"] for row in development}) != EXPECTED_EPISODES:
        raise ProtocolError("causal probe scenes are not separated")

    binding = manifest["readiness_binding"]
    expected_binding = {
        "candidate_bc_ready": 0,
        "candidate_checkpoint_sha256": EXPECTED_CANDIDATE_SHA256,
        "candidate_policy_state_sha256": EXPECTED_POLICY_STATE_SHA256,
        "readiness_gate_pass": False,
        "readiness_manifest_canonical_sha256": (
            EXPECTED_READINESS_MANIFEST_CANONICAL_SHA256
        ),
        "readiness_manifest_raw_sha256": EXPECTED_READINESS_MANIFEST_RAW_SHA256,
        "readiness_manifest_relative_path": (
            "integrations/rlinf_v03/streamvln_p8_successor_bc_readiness_manifest.json"
        ),
        "readiness_receipt_file_sha256": EXPECTED_READINESS_RECEIPT_FILE_SHA256,
        "readiness_receipt_sha256": EXPECTED_READINESS_RECEIPT_SHA256,
    }
    if binding != expected_binding:
        raise ProtocolError("causal probe readiness binding drifted")
    if readiness._json_hash(readiness_manifest) != EXPECTED_READINESS_MANIFEST_CANONICAL_SHA256:
        raise ProtocolError("v5 readiness manifest canonical digest drifted")
    if manifest["source_contract"] != {
        "commits": {
            "habitat_sim": "f179b584bcd713c5a2a998132211e2cae881d6d1",
            "rlinf": "0505431899574619da86f551bad70b71e0ea2177",
            "streamvln": "e48f6ff7e9201d93aae003e8f64b04c00cec13bc",
        },
        "original_streamvln_only": True,
        "ttt_or_secondary_post_training": False,
    }:
        raise ProtocolError("causal probe source contract drifted")
    if manifest["source_contract"] != {
        key: readiness_manifest["source_contract"][key]
        for key in (
            "commits",
            "original_streamvln_only",
            "ttt_or_secondary_post_training",
        )
    }:
        raise ProtocolError("causal probe source differs from v5 readiness")
    return {
        "attribution_rule": dict(manifest["attribution_rule"]),
        "development": development,
        "implementation": dict(implementation),
        "protocol": dict(protocol),
        "readiness": readiness_contract,
        "thresholds": dict(manifest["thresholds"]),
    }


def validate_readiness_receipt(
    value: object,
    *,
    readiness_manifest_path: Path,
    candidate_path: Path,
) -> dict:
    if not isinstance(value, Mapping):
        raise ProtocolError("v5 readiness receipt must be an object")
    receipt = dict(value)
    embedded = receipt.get("receipt_sha256")
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256", None)
    if embedded != EXPECTED_READINESS_RECEIPT_SHA256:
        raise ProtocolError("v5 readiness receipt embedded digest drifted")
    if readiness._json_hash(unsigned) != embedded:
        raise ProtocolError("v5 readiness receipt self-hash is invalid")
    if (
        receipt.get("schema_name") != readiness.SCHEMA_NAME
        or receipt.get("stage") != "readiness"
        or receipt.get("split") != TRAIN_SPLIT
        or receipt.get("gate_pass") is not False
        or receipt.get("promotion") is not None
    ):
        raise ProtocolError("v5 readiness receipt outcome or scope drifted")
    manifest_record = receipt.get("manifest")
    if not isinstance(manifest_record, Mapping) or (
        manifest_record.get("file_sha256") != EXPECTED_READINESS_MANIFEST_RAW_SHA256
        or manifest_record.get("canonical_sha256")
        != EXPECTED_READINESS_MANIFEST_CANONICAL_SHA256
    ):
        raise ProtocolError("v5 receipt does not bind the frozen readiness manifest")
    result = receipt.get("result")
    if not isinstance(result, Mapping) or (
        result.get("candidate_bc_ready") not in (0, 0.0, False)
        or result.get("candidate_checkpoint_sha256") != EXPECTED_CANDIDATE_SHA256
        or result.get("candidate_policy_state_sha256") != EXPECTED_POLICY_STATE_SHA256
    ):
        raise ProtocolError("v5 receipt candidate identity drifted")
    recorded_candidate = Path(str(result.get("candidate_checkpoint", ""))).expanduser()
    if recorded_candidate.resolve(strict=False) != candidate_path.resolve(strict=True):
        raise ProtocolError("candidate path differs from the v5 readiness receipt")
    if readiness._file_sha256(readiness_manifest_path) != EXPECTED_READINESS_MANIFEST_RAW_SHA256:
        raise ProtocolError("v5 readiness manifest bytes drifted")
    if readiness._file_sha256(candidate_path) != EXPECTED_CANDIDATE_SHA256:
        raise ProtocolError("v5 candidate checkpoint bytes drifted")
    return {
        "receipt_sha256": embedded,
        "candidate_checkpoint_sha256": EXPECTED_CANDIDATE_SHA256,
        "candidate_policy_state_sha256": EXPECTED_POLICY_STATE_SHA256,
        "gate_pass": False,
        "promotion": None,
    }


def verify_probe_implementation(
    manifest: Mapping,
    *,
    runner_path: Path | None = None,
) -> dict:
    """Verify this runner without placing its digest inside its own bytes."""

    implementation = manifest.get("implementation")
    if not isinstance(implementation, Mapping):
        raise ProtocolError("causal probe implementation binding is missing")
    path = (runner_path or Path(__file__)).expanduser().resolve(strict=True)
    relative = str(implementation.get("runner_relative_path", "")).replace(
        "\\", "/"
    )
    if not path.as_posix().endswith(relative):
        raise ProtocolError("loaded causal probe runner escaped its relative path")
    observed_bytes = path.stat().st_size
    observed_sha = readiness._file_sha256(path)
    if (
        observed_bytes != implementation.get("runner_bytes")
        or observed_sha != implementation.get("runner_sha256")
    ):
        raise ProtocolError("causal probe runner bytes/hash drifted")
    return {
        "runner_relative_path": relative,
        "runner_bytes": observed_bytes,
        "runner_sha256": observed_sha,
    }


def production_lineage(
    args: argparse.Namespace,
    readiness_manifest: Mapping,
    readiness_contract: Mapping,
) -> dict:
    readiness_path = Path(args.readiness_manifest).expanduser().resolve(strict=True)
    receipt_path = Path(args.readiness_receipt).expanduser().resolve(strict=True)
    candidate_path = Path(args.candidate).expanduser().resolve(strict=True)
    if readiness._file_sha256(receipt_path) != EXPECTED_READINESS_RECEIPT_FILE_SHA256:
        raise ProtocolError("v5 readiness receipt file bytes drifted")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    binding = validate_readiness_receipt(
        receipt,
        readiness_manifest_path=readiness_path,
        candidate_path=candidate_path,
    )
    production = readiness.production_preflight(args, readiness_manifest, readiness_contract)
    if production.get("status") != "pass":
        raise ProtocolError("reused v5 production preflight did not pass")
    return {
        "status": "pass",
        "readiness_binding": binding,
        "production": production,
        "paths": dict(production["paths"]),
    }


def _angle_error_radians(predicted: float, target: float) -> float:
    return abs(math.atan2(math.sin(predicted - target), math.cos(predicted - target)))


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _nearest_rank(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    if not 0.0 < percentile <= 1.0:
        raise ValueError("nearest-rank percentile must be in (0,1]")
    ordered = sorted(float(value) for value in values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def summarize_teacher_forced(records: Sequence[Mapping]) -> dict:
    confusion = {
        truth: {prediction: 0 for prediction in readiness.ROUTES}
        for truth in readiness.ROUTES
    }
    first_action_confusion = {
        truth: {prediction: 0 for prediction in readiness.ACTION_NAMES}
        for truth in readiness.ACTION_NAMES
    }
    stop_tp = stop_fp = stop_fn = 0
    route_correct = 0
    vector_errors = []
    bearing_errors = []
    distance_errors = []
    predicted_zero_forward = 0
    selected_zero_forward = 0
    selected_waypoint_count = 0
    first_action_correct = 0
    true_stop_waypoint_fallback_count = 0
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ProtocolError(f"teacher record {index} must be an object")
        truth = str(record["truth_route"])
        prediction = str(record["predicted_route"])
        if truth not in readiness.ROUTES or prediction not in readiness.ROUTES:
            raise ProtocolError("teacher route record is outside the frozen route set")
        confusion[truth][prediction] += 1
        route_correct += int(truth == prediction)
        truth_stop = truth == "stop"
        predicted_stop = prediction == "stop"
        stop_tp += int(truth_stop and predicted_stop)
        stop_fp += int(not truth_stop and predicted_stop)
        stop_fn += int(truth_stop and not predicted_stop)
        if record.get("waypoint_vector_l2_m") is not None:
            vector_errors.append(float(record["waypoint_vector_l2_m"]))
            bearing_errors.append(float(record["waypoint_bearing_abs_error_rad"]))
            distance_errors.append(float(record["waypoint_distance_abs_error_m"]))
            truth_action = readiness.ACTION_NAMES[int(record["oracle_waypoint_first_action"])]
            predicted_action = readiness.ACTION_NAMES[
                int(record["predicted_waypoint_first_action"])
            ]
            first_action_confusion[truth_action][predicted_action] += 1
            first_action_correct += int(truth_action == predicted_action)
            predicted_zero_forward += int(record["predicted_waypoint_zero_forward"])
        if record.get("selected_waypoint_zero_forward") is not None:
            selected_waypoint_count += 1
            selected_zero_forward += int(record["selected_waypoint_zero_forward"])
        fallback = record.get(
            "true_stop_predicted_waypoint_candidate_fallback", False
        )
        if type(fallback) is not bool:
            raise ProtocolError("true-STOP waypoint fallback flag must be boolean")
        true_stop_waypoint_fallback_count += int(fallback)
    count = len(records)
    predicted_stops = stop_tp + stop_fp
    true_stops = stop_tp + stop_fn
    waypoint_count = len(vector_errors)
    route_support = {
        truth: sum(confusion[truth].values()) for truth in readiness.ROUTES
    }
    keep_recall = (
        confusion["keep_original"]["keep_original"]
        / route_support["keep_original"]
        if route_support["keep_original"]
        else None
    )
    waypoint_recall = (
        confusion["waypoint"]["waypoint"] / route_support["waypoint"]
        if route_support["waypoint"]
        else None
    )
    nonstop_balanced_accuracy = (
        (keep_recall + waypoint_recall) / 2.0
        if keep_recall is not None and waypoint_recall is not None
        else None
    )
    median_bearing = (
        statistics.median(bearing_errors) if bearing_errors else None
    )
    return {
        "decision_count": count,
        "route_accuracy": route_correct / count if count else None,
        "route_confusion": confusion,
        "route_support": route_support,
        "route_nonstop_bal_acc": nonstop_balanced_accuracy,
        "stop": {
            "true_positive": stop_tp,
            "false_positive": stop_fp,
            "false_negative": stop_fn,
            "predicted_positive": predicted_stops,
            "actual_positive": true_stops,
            "precision": stop_tp / predicted_stops if predicted_stops else None,
            "recall": stop_tp / true_stops if true_stops else None,
        },
        "waypoint": {
            "comparison_count": waypoint_count,
            "mean_vector_l2_m": _mean(vector_errors),
            "mean_bearing_abs_error_rad": _mean(bearing_errors),
            "mean_bearing_abs_error_deg": (
                math.degrees(_mean(bearing_errors)) if bearing_errors else None
            ),
            "mean_distance_abs_error_m": _mean(distance_errors),
            "first_action_agreement": (
                first_action_correct / waypoint_count if waypoint_count else None
            ),
            "median_bearing_abs_error_rad": median_bearing,
            "median_bearing_abs_error_deg": (
                math.degrees(median_bearing)
                if median_bearing is not None
                else None
            ),
            "p90_bearing_abs_error_rad": _nearest_rank(bearing_errors, 0.9),
            "p90_bearing_abs_error_deg": (
                math.degrees(_nearest_rank(bearing_errors, 0.9))
                if bearing_errors
                else None
            ),
            "median_vector_l2_m": (
                statistics.median(vector_errors) if vector_errors else None
            ),
            "compiled_first_action_confusion": first_action_confusion,
            "predicted_zero_forward_macro_count": predicted_zero_forward,
            "predicted_zero_forward_macro_fraction": (
                predicted_zero_forward / waypoint_count if waypoint_count else None
            ),
        },
        "selected_waypoint_macro": {
            "count": selected_waypoint_count,
            "zero_forward_count": selected_zero_forward,
            "zero_forward_fraction": (
                selected_zero_forward / selected_waypoint_count
                if selected_waypoint_count
                else None
            ),
        },
        "route_arm_true_stop_predicted_waypoint_fallback": {
            "true_stop_predicted_waypoint_candidate_fallback_count": (
                true_stop_waypoint_fallback_count
            ),
            "execution_source": "candidate_deterministic_mean",
            "is_oracle": False,
        },
    }


def _threshold_number(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
) -> float:
    return readiness._finite(value, name, minimum=minimum)


def adjudicate_frozen_attribution(
    payload: Mapping,
    *,
    thresholds: Mapping,
    attribution_rule: Mapping,
) -> dict:
    """Apply the frozen causal authorization rule to completed probe evidence."""

    arms = payload["arms"]
    arm_threshold = thresholds["arm_recovered"]
    arm_results = {}
    for arm in ARMS:
        navigation = arms[arm]["navigation"]
        actions = navigation.get("action_counts")
        if not isinstance(actions, Mapping) or set(actions) != set(
            readiness.ACTION_NAMES
        ):
            raise ProtocolError(f"{arm} action counts drifted")
        action_counts = {
            name: readiness._strict_int(
                actions[name], f"{arm}.action_counts.{name}"
            )
            for name in readiness.ACTION_NAMES
        }
        nonstop = sum(
            action_counts[name] for name in ("FORWARD", "LEFT", "RIGHT")
        )
        forward_fraction = (
            action_counts["FORWARD"] / nonstop if nonstop else None
        )
        episode_count = readiness._strict_int(
            navigation.get("episode_count"), f"{arm}.episode_count"
        )
        success_count = readiness._strict_int(
            navigation.get("success_count"), f"{arm}.success_count"
        )
        successful_stop_count = readiness._strict_int(
            navigation.get("successful_stop_count"),
            f"{arm}.successful_stop_count",
        )
        root_unavailable = readiness._strict_int(
            navigation.get("root_recovery_unavailable_count"),
            f"{arm}.root_recovery_unavailable_count",
        )
        checks = {
            "episode_count": episode_count
            == int(arm_threshold["episode_count"]),
            "success_count": success_count
            >= int(arm_threshold["success_count_min"]),
            "successful_stop_count": successful_stop_count
            >= int(arm_threshold["successful_stop_count_min"]),
            "root_recovery_available": root_unavailable
            <= int(arm_threshold["root_recovery_unavailable_count_max"]),
            "forward_nonstop_fraction": (
                forward_fraction is not None
                and forward_fraction
                >= float(arm_threshold["forward_nonstop_fraction_min"])
            ),
        }
        arm_results[arm] = {
            "recovered": all(checks.values()),
            "checks": checks,
            "observed": {
                "episode_count": episode_count,
                "success_count": success_count,
                "successful_stop_count": successful_stop_count,
                "root_recovery_unavailable_count": root_unavailable,
                "forward_nonstop_fraction": forward_fraction,
            },
        }

    teacher = payload["teacher_forced"]["metrics"]
    supports = teacher.get("route_support")
    if not isinstance(supports, Mapping) or set(supports) != set(
        readiness.ROUTES
    ):
        raise ProtocolError("teacher route support drifted")
    support_counts = {
        route: readiness._strict_int(
            supports[route], f"teacher.route_support.{route}"
        )
        for route in readiness.ROUTES
    }
    sample_min = thresholds["teacher_sample_min"]
    sample_checks = {
        route: support_counts[route] >= int(sample_min[route])
        for route in readiness.ROUTES
    }
    sample_sufficient = all(sample_checks.values())

    route_threshold = thresholds["teacher_route"]
    stop = teacher.get("stop")
    if not isinstance(stop, Mapping):
        raise ProtocolError("teacher STOP metrics are missing")
    balanced_accuracy = teacher.get("route_nonstop_bal_acc")
    stop_recall = stop.get("recall")
    premature_stop_count = readiness._strict_int(
        stop.get("false_positive"), "teacher.stop.false_positive"
    )
    route_checks = {
        "route_nonstop_bal_acc": (
            balanced_accuracy is not None
            and _threshold_number(
                balanced_accuracy,
                "teacher.route_nonstop_bal_acc",
                minimum=0.0,
            )
            >= float(route_threshold["route_nonstop_bal_acc_min"])
        ),
        "stop_recall": (
            stop_recall is not None
            and _threshold_number(
                stop_recall, "teacher.stop.recall", minimum=0.0
            )
            >= float(route_threshold["stop_recall_min"])
        ),
        "premature_stop_count": premature_stop_count
        <= int(route_threshold["premature_stop_count_max"]),
    }
    route_group_pass = sample_sufficient and all(route_checks.values())

    waypoint = teacher.get("waypoint")
    if not isinstance(waypoint, Mapping):
        raise ProtocolError("teacher waypoint metrics are missing")
    waypoint_threshold = thresholds["teacher_waypoint"]
    first_action_agreement = waypoint.get("first_action_agreement")
    median_bearing = waypoint.get("median_bearing_abs_error_deg")
    p90_bearing = waypoint.get("p90_bearing_abs_error_deg")
    median_l2 = waypoint.get("median_vector_l2_m")
    waypoint_checks = {
        "first_action_agreement": (
            first_action_agreement is not None
            and _threshold_number(
                first_action_agreement,
                "teacher.waypoint.first_action_agreement",
                minimum=0.0,
            )
            >= float(waypoint_threshold["first_action_agreement_min"])
        ),
        "median_bearing_abs_error_deg": (
            median_bearing is not None
            and _threshold_number(
                median_bearing,
                "teacher.waypoint.median_bearing_abs_error_deg",
                minimum=0.0,
            )
            <= float(waypoint_threshold["bearing_abs_error_median_deg_max"])
        ),
        "p90_bearing_abs_error_deg": (
            p90_bearing is not None
            and _threshold_number(
                p90_bearing,
                "teacher.waypoint.p90_bearing_abs_error_deg",
                minimum=0.0,
            )
            <= float(waypoint_threshold["bearing_abs_error_p90_deg_max"])
        ),
        "median_vector_l2_m": (
            median_l2 is not None
            and _threshold_number(
                median_l2,
                "teacher.waypoint.median_vector_l2_m",
                minimum=0.0,
            )
            <= float(waypoint_threshold["vector_l2_median_m_max"])
        ),
    }
    waypoint_group_pass = sample_sufficient and all(waypoint_checks.values())

    route_arm_recovered = arm_results[ARM_ROUTE]["recovered"]
    waypoint_arm_recovered = arm_results[ARM_WAYPOINT]["recovered"]
    authorize = bool(
        sample_sufficient
        and not route_arm_recovered
        and not waypoint_arm_recovered
        and not route_group_pass
        and not waypoint_group_pass
    )
    labels = attribution_rule["labels"]
    if not sample_sufficient:
        attribution = labels["insufficient"]
    elif route_arm_recovered and waypoint_arm_recovered:
        attribution = labels["both_arms_recovered"]
    elif route_arm_recovered:
        attribution = labels["route_arm_only"]
    elif waypoint_arm_recovered:
        attribution = labels["waypoint_arm_only"]
    elif authorize:
        attribution = labels["feature_v2"]
    elif not route_group_pass and waypoint_group_pass:
        attribution = labels["route_group_only"]
    elif route_group_pass and not waypoint_group_pass:
        attribution = labels["waypoint_group_only"]
    else:
        attribution = labels["neither_group_failed"]

    return {
        "arms": arm_results,
        "teacher": {
            "sample_sufficient": sample_sufficient,
            "sample_checks": sample_checks,
            "route_group_pass": route_group_pass,
            "route_checks": route_checks,
            "waypoint_group_pass": waypoint_group_pass,
            "waypoint_checks": waypoint_checks,
        },
        "authorize_feature_v2": authorize,
        "attribution": attribution,
        "rule": attribution_rule["authorize_feature_v2_when"],
    }


@contextmanager
def _expose_oracle_waypoint_to_probe():
    """Pass all non-STOP expert geometry through the existing provider hook."""

    original = readiness.label_successor_route

    def probe_label(*, expert_stop, official_shadow_action, expert_first_primitive):
        del official_shadow_action, expert_first_primitive
        return "stop" if expert_stop else "waypoint"

    readiness.label_successor_route = probe_label
    try:
        yield
    finally:
        readiness.label_successor_route = original


def _select_hybrid_execution(
    *,
    arm: str,
    predicted_route: str,
    truth_route: str,
    predicted_waypoint: tuple[float, float],
    oracle_waypoint: tuple[float, float] | None,
) -> dict:
    """Select the intervened route/waypoint without inventing STOP geometry."""

    if arm not in ARMS:
        raise ValueError("unknown causal probe arm")
    if predicted_route not in readiness.ROUTES or truth_route not in readiness.ROUTES:
        raise ProtocolError("hybrid route is outside the frozen route set")
    fallback = False
    waypoint_source = "none"
    if arm == ARM_ROUTE:
        selected_route = predicted_route
        if selected_route == "waypoint":
            if oracle_waypoint is not None:
                selected_waypoint = oracle_waypoint
                waypoint_source = "train_shortest_path_expert"
            else:
                if truth_route != "stop":
                    raise ProtocolError(
                        "non-STOP route-arm waypoint lacks oracle geometry"
                    )
                # A true STOP has no metric oracle waypoint.  Continuing with
                # the deterministic candidate mean preserves the route-error
                # intervention without fabricating or mislabeling an oracle.
                selected_waypoint = predicted_waypoint
                waypoint_source = (
                    "candidate_deterministic_mean_true_stop_fallback"
                )
                fallback = True
        else:
            selected_waypoint = (0.0, 0.0)
    else:
        selected_route = truth_route
        if selected_route == "waypoint":
            selected_waypoint = predicted_waypoint
            waypoint_source = "candidate_deterministic_mean"
        else:
            selected_waypoint = (0.0, 0.0)
    return {
        "selected_route": selected_route,
        "selected_waypoint": tuple(float(value) for value in selected_waypoint),
        "selected_waypoint_source": waypoint_source,
        "true_stop_predicted_waypoint_candidate_fallback": fallback,
    }


class ProbeProvider(Protocol):
    def execute(
        self,
        *,
        manifest: Mapping,
        contract: Mapping,
        preflight: Mapping,
        args: argparse.Namespace,
    ) -> Mapping:
        ...


class CanonicalCausalProbeProvider(readiness.CanonicalReadinessProvider):
    """Two-arm decision intervention over the frozen v5 candidate."""

    def __init__(
        self,
        *,
        args: argparse.Namespace,
        readiness_manifest: Mapping,
        readiness_contract: Mapping,
        preflight: Mapping,
        probe_contract: Mapping,
    ) -> None:
        super().__init__(
            args=args,
            manifest=readiness_manifest,
            contract=readiness_contract,
            preflight=preflight["production"],
        )
        self.probe_contract = probe_contract
        self.candidate_path = Path(args.candidate).expanduser().resolve(strict=True)
        self._candidate_loaded = False
        self._candidate_load_count = 0
        self._active_arm: str | None = None
        self._active_episode: Mapping | None = None
        self._teacher_records: list[dict] = []

    def _ensure_policy(self, feature, topology_snapshot) -> None:
        super()._ensure_policy(feature, topology_snapshot)
        if self._candidate_loaded:
            return
        import torch

        try:
            checkpoint = torch.load(
                self.candidate_path, map_location="cpu", weights_only=True
            )
        except TypeError:  # pragma: no cover - older pinned torch fallback
            checkpoint = torch.load(self.candidate_path, map_location="cpu")
        if not isinstance(checkpoint, Mapping):
            raise ProtocolError("v5 candidate checkpoint root must be an object")
        required = {
            "schema_name",
            "schema_version",
            "bc_ready",
            "policy_state_dict",
            "manifest_sha256",
            "expert_dataset_sha256",
            "combined_dataset_sha256",
        }
        if set(checkpoint) != required:
            raise ProtocolError("v5 candidate checkpoint schema drifted")
        if (
            checkpoint["schema_name"] != readiness.SCHEMA_NAME
            or checkpoint["schema_version"] != readiness.SCHEMA_VERSION
            or checkpoint["bc_ready"] not in (0, 0.0, False)
            or checkpoint["manifest_sha256"]
            != EXPECTED_READINESS_MANIFEST_CANONICAL_SHA256
        ):
            raise ProtocolError("v5 candidate checkpoint header drifted")
        state = checkpoint["policy_state_dict"]
        if not isinstance(state, Mapping):
            raise ProtocolError("v5 candidate policy state is missing")
        if readiness._tensor_mapping_hash(state) != EXPECTED_POLICY_STATE_SHA256:
            raise ProtocolError("v5 candidate policy state digest drifted")
        self.policy.load_state_dict(state, strict=True)
        self.policy.eval()
        if self.policy.is_bc_ready():
            raise ProtocolError("causal probe refuses a promoted candidate")
        if readiness._module_state_hash(self.policy) != EXPECTED_POLICY_STATE_SHA256:
            raise ProtocolError("loaded v5 policy state digest drifted")
        self._candidate_loaded = True
        self._candidate_load_count = 1

    def _learned_sample(
        self,
        frame,
        shadow_action: int,
        context: Mapping,
        *,
        deterministic: bool,
        expert_route,
        expert_waypoint,
    ):
        del deterministic
        import torch

        if self._active_arm not in ARMS or self._active_episode is None:
            raise ProtocolError("causal probe decision has no active arm/episode")
        with torch.inference_mode():
            *_, route_logits, raw_mean, _raw_log_std, values = (
                self.policy._distribution_terms(context)
            )
            route_index = route_logits.argmax(dim=-1)
            predicted_route = self.policy_module.POLICY_ROUTES[int(route_index.item())]
            predicted_waypoint_tensor = self.policy._waypoint_from_raw(raw_mean)
        predicted_waypoint = tuple(
            float(value) for value in predicted_waypoint_tensor.reshape(-1).tolist()
        )
        predicted_actions = tuple(
            self.action.compile_metric_waypoint(predicted_waypoint)
        )

        is_stop = expert_route is self.action.RouteName.STOP
        oracle_waypoint = None if is_stop else tuple(float(value) for value in expert_waypoint)
        oracle_actions = (
            (self.action.STOP_ACTION,)
            if is_stop
            else tuple(self.action.compile_metric_waypoint(oracle_waypoint))
        )
        truth_label = _ORIGINAL_LABEL_SUCCESSOR_ROUTE(
            expert_stop=is_stop,
            official_shadow_action=int(shadow_action),
            expert_first_primitive=int(oracle_actions[0]),
        )
        truth_route = self.action.RouteName(truth_label)

        selection = _select_hybrid_execution(
            arm=self._active_arm,
            predicted_route=predicted_route.value,
            truth_route=truth_route.value,
            predicted_waypoint=predicted_waypoint,
            oracle_waypoint=oracle_waypoint,
        )
        selected_route = self.action.RouteName(selection["selected_route"])
        selected_waypoint = selection["selected_waypoint"]
        if selected_route is self.action.RouteName.WAYPOINT:
            selected_actions = tuple(
                self.action.compile_metric_waypoint(selected_waypoint)
            )
            official_action = None
        elif selected_route is self.action.RouteName.KEEP_ORIGINAL:
            selected_actions = (int(shadow_action),)
            official_action = int(shadow_action)
        else:
            selected_actions = (self.action.STOP_ACTION,)
            official_action = None

        record = {
            "episode_id": str(self._active_episode["episode_id"]),
            "scene": str(self._active_episode["scene"]),
            "decision_index": int(frame.decision_index),
            "truth_route": truth_route.value,
            "predicted_route": predicted_route.value,
            "selected_route": selected_route.value,
            "selected_first_action": int(selected_actions[0]),
            "selected_waypoint_source": selection["selected_waypoint_source"],
            "true_stop_predicted_waypoint_candidate_fallback": selection[
                "true_stop_predicted_waypoint_candidate_fallback"
            ],
            "predicted_waypoint": list(predicted_waypoint),
            "predicted_waypoint_first_action": int(predicted_actions[0]),
            "predicted_waypoint_zero_forward": (
                self.action.FORWARD_ACTION not in predicted_actions
            ),
            "oracle_waypoint": None,
            "oracle_waypoint_first_action": None,
            "waypoint_vector_l2_m": None,
            "waypoint_bearing_abs_error_rad": None,
            "waypoint_distance_abs_error_m": None,
            "selected_waypoint_zero_forward": (
                self.action.FORWARD_ACTION not in selected_actions
                if selected_route is self.action.RouteName.WAYPOINT
                else None
            ),
        }
        if oracle_waypoint is not None:
            predicted_distance = math.hypot(*predicted_waypoint)
            oracle_distance = math.hypot(*oracle_waypoint)
            predicted_bearing = math.atan2(
                predicted_waypoint[1], predicted_waypoint[0]
            )
            oracle_bearing = math.atan2(oracle_waypoint[1], oracle_waypoint[0])
            record.update({
                "oracle_waypoint": list(oracle_waypoint),
                "oracle_waypoint_first_action": int(oracle_actions[0]),
                "waypoint_vector_l2_m": math.hypot(
                    predicted_waypoint[0] - oracle_waypoint[0],
                    predicted_waypoint[1] - oracle_waypoint[1],
                ),
                "waypoint_bearing_abs_error_rad": _angle_error_radians(
                    predicted_bearing, oracle_bearing
                ),
                "waypoint_distance_abs_error_m": abs(
                    predicted_distance - oracle_distance
                ),
            })
        self._teacher_records.append(record)

        decision = self.action.RouteDecision(
            decision_index=int(frame.decision_index),
            route=selected_route,
            waypoint_local_m=selected_waypoint,
            official_action=official_action,
            episode_token=frame.episode_token,
            physical_step=int(frame.physical_step),
        )
        return self.protocol.SuccessorDecisionSample(
            decision=decision,
            proposed_actions=selected_actions,
            logprob=0.0,
            value=float(values.item()),
            official_shadow_action=int(shadow_action),
            forward_context={
                key: value.detach().cpu().clone() for key, value in context.items()
            },
            expert_route_target=truth_route,
            expert_waypoint_target=(
                oracle_waypoint
                if truth_route is self.action.RouteName.WAYPOINT
                else None
            ),
        )

    def _score_teacher_rows(self, rows: Sequence[Mapping]) -> list[dict]:
        import torch

        records = []
        for row in rows:
            context = {
                key: value.to(device=self.args.device)
                for key, value in row["forward_context"].items()
            }
            with torch.inference_mode():
                *_, route_logits, raw_mean, _raw_log_std, _values = (
                    self.policy._distribution_terms(context)
                )
                route_index = route_logits.argmax(dim=-1)
                predicted_route = self.policy_module.POLICY_ROUTES[
                    int(route_index.item())
                ]
                waypoint_tensor = self.policy._waypoint_from_raw(raw_mean)
            predicted_waypoint = tuple(
                float(value) for value in waypoint_tensor.reshape(-1).tolist()
            )
            predicted_actions = tuple(
                self.action.compile_metric_waypoint(predicted_waypoint)
            )
            truth_route = self.action.RouteName(str(row["route_label"]))
            record = {
                "truth_route": truth_route.value,
                "predicted_route": predicted_route.value,
                "predicted_waypoint_first_action": int(predicted_actions[0]),
                "predicted_waypoint_zero_forward": (
                    self.action.FORWARD_ACTION not in predicted_actions
                ),
                "oracle_waypoint_first_action": None,
                "waypoint_vector_l2_m": None,
                "waypoint_bearing_abs_error_rad": None,
                "waypoint_distance_abs_error_m": None,
                "selected_waypoint_zero_forward": None,
            }
            if truth_route is self.action.RouteName.WAYPOINT:
                target = tuple(float(value) for value in row["waypoint_target"])
                target_actions = tuple(
                    self.action.compile_metric_waypoint(target)
                )
                predicted_distance = math.hypot(*predicted_waypoint)
                target_distance = math.hypot(*target)
                predicted_bearing = math.atan2(
                    predicted_waypoint[1], predicted_waypoint[0]
                )
                target_bearing = math.atan2(target[1], target[0])
                if int(target_actions[0]) != int(row["expert_first_primitive"]):
                    raise ProtocolError(
                        "teacher waypoint compilation differs from collected label"
                    )
                record.update({
                    "oracle_waypoint_first_action": int(target_actions[0]),
                    "waypoint_vector_l2_m": math.hypot(
                        predicted_waypoint[0] - target[0],
                        predicted_waypoint[1] - target[1],
                    ),
                    "waypoint_bearing_abs_error_rad": _angle_error_radians(
                        predicted_bearing, target_bearing
                    ),
                    "waypoint_distance_abs_error_m": abs(
                        predicted_distance - target_distance
                    ),
                })
            records.append(record)
        return records

    def _run_teacher_forced(self) -> dict:
        protocol = self.probe_contract["protocol"]
        all_records = []
        episodes = []
        for spec in self.probe_contract["development"]:
            result = self._run_episode(
                spec=spec,
                mode="expert",
                decision_budget=int(
                    protocol["teacher_forced_decision_budget_per_episode"]
                ),
                physical_budget=int(
                    protocol["teacher_forced_physical_budget_per_episode"]
                ),
                collect_rows=True,
            )
            records = self._score_teacher_rows(result["rows"])
            all_records.extend(records)
            episodes.append({
                "episode_id": str(spec["episode_id"]),
                "scene": str(spec["scene"]),
                "decision_count": int(result["decision_count"]),
                "physical_steps": int(result["physical_steps"]),
                "exhausted": bool(result["exhausted"]),
                "unique_state_count": len(result["rows"]),
                "metrics": summarize_teacher_forced(records),
            })
        return {
            "metrics": summarize_teacher_forced(all_records),
            "episodes": episodes,
        }

    def _run_arm(self, arm: str) -> dict:
        if arm not in ARMS:
            raise ValueError("unknown causal probe arm")
        protocol = self.probe_contract["protocol"]
        episodes = []
        arm_records = []
        self._active_arm = arm
        try:
            for spec in self.probe_contract["development"]:
                self._active_episode = spec
                start = len(self._teacher_records)
                with _expose_oracle_waypoint_to_probe():
                    episode = self._run_episode(
                        spec=spec,
                        mode="learned",
                        decision_budget=int(protocol["decision_budget_per_episode"]),
                        physical_budget=int(protocol["physical_budget_per_episode"]),
                        collect_rows=False,
                    )
                decision_records = self._teacher_records[start:]
                episode["counterfactual_diagnostics"] = summarize_teacher_forced(
                    decision_records
                )
                episode[
                    "true_stop_predicted_waypoint_candidate_fallback_records"
                ] = [
                    {
                        "decision_index": int(record["decision_index"]),
                        "selected_waypoint_source": record[
                            "selected_waypoint_source"
                        ],
                        "selected_first_action": int(
                            record["selected_first_action"]
                        ),
                        "predicted_waypoint": list(record["predicted_waypoint"]),
                        "is_oracle": False,
                    }
                    for record in decision_records
                    if record[
                        "true_stop_predicted_waypoint_candidate_fallback"
                    ]
                ]
                arm_records.extend(decision_records)
                episodes.append(episode)
        finally:
            self._active_episode = None
            self._active_arm = None
        return {
            "navigation": self._aggregate(episodes),
            "counterfactual_diagnostics": summarize_teacher_forced(arm_records),
            "episodes": episodes,
        }

    def execute(
        self,
        *,
        manifest: Mapping,
        contract: Mapping,
        preflight: Mapping,
        args: argparse.Namespace,
    ) -> Mapping:
        del manifest, contract, preflight, args
        self._load_once()
        teacher_forced = self._run_teacher_forced()
        results = {arm: self._run_arm(arm) for arm in ARMS}
        if not self._candidate_loaded or self.policy is None:
            raise ProtocolError("causal probe did not load the frozen v5 candidate")
        return {
            "candidate_checkpoint_sha256": readiness._file_sha256(
                self.candidate_path
            ),
            "candidate_policy_state_sha256": readiness._module_state_hash(
                self.policy
            ),
            "teacher_forced": teacher_forced,
            "arms": results,
            "split_read_counts": {
                "train": int(self._split_reads["train"]),
                "val_seen": 0,
                "val_unseen": 0,
            },
            "guards": {
                "model_load_count": int(self._model_load_count),
                "candidate_load_count": int(self._candidate_load_count),
                "training_updates": 0,
                "ppo_updates": 0,
                "promotion_attempts": 0,
                "single_process": True,
                "train_only": True,
            },
        }


def validate_probe_payload(payload: object, contract: Mapping) -> dict:
    if not isinstance(payload, Mapping):
        raise ProtocolError("causal probe provider returned a non-object")
    result = dict(payload)
    if (
        result.get("candidate_checkpoint_sha256") != EXPECTED_CANDIDATE_SHA256
        or result.get("candidate_policy_state_sha256") != EXPECTED_POLICY_STATE_SHA256
    ):
        raise ProtocolError("causal probe provider used the wrong candidate")
    guards = result.get("guards")
    if guards != {
        "model_load_count": 1,
        "candidate_load_count": 1,
        "training_updates": 0,
        "ppo_updates": 0,
        "promotion_attempts": 0,
        "single_process": True,
        "train_only": True,
    }:
        raise ProtocolError("causal probe execution guards drifted")
    if result.get("split_read_counts") != {
        "train": EXPECTED_TRAIN_READS,
        "val_seen": 0,
        "val_unseen": 0,
    }:
        raise ProtocolError("causal probe split-read accounting drifted")
    teacher_forced = result.get("teacher_forced")
    if not isinstance(teacher_forced, Mapping):
        raise ProtocolError("causal probe lacks teacher-forced evidence")
    teacher_metrics = teacher_forced.get("metrics")
    teacher_episodes = teacher_forced.get("episodes")
    expected_ids = [row["episode_id"] for row in contract["development"]]
    if (
        not isinstance(teacher_metrics, Mapping)
        or type(teacher_metrics.get("decision_count")) is not int
        or teacher_metrics["decision_count"] < 1
        or not isinstance(teacher_episodes, list)
        or [str(row.get("episode_id")) for row in teacher_episodes] != expected_ids
    ):
        raise ProtocolError("causal probe teacher-forced evidence is incomplete")
    arms = result.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != set(ARMS):
        raise ProtocolError("causal probe result lacks the exact two arms")
    for arm in ARMS:
        value = arms[arm]
        if not isinstance(value, Mapping):
            raise ProtocolError(f"causal probe arm {arm} is invalid")
        navigation = value.get("navigation")
        counterfactual = value.get("counterfactual_diagnostics")
        episodes = value.get("episodes")
        if (
            not isinstance(navigation, Mapping)
            or navigation.get("episode_count") != EXPECTED_EPISODES
            or not isinstance(counterfactual, Mapping)
            or type(counterfactual.get("decision_count")) is not int
            or counterfactual["decision_count"] < 1
            or not isinstance(episodes, list)
            or [str(row.get("episode_id")) for row in episodes] != expected_ids
        ):
            raise ProtocolError(f"causal probe arm {arm} evidence is incomplete")
    return result


def _provider_execute(provider, **kwargs) -> Mapping:
    function = getattr(provider, "execute", provider)
    result = function(**kwargs)
    if not isinstance(result, Mapping):
        raise ProtocolError("causal probe provider returned a non-object")
    return result


def run(
    args: argparse.Namespace,
    *,
    lineage_provider: Callable[[argparse.Namespace, Mapping, Mapping], Mapping]
    | None = None,
    probe_provider: ProbeProvider | Callable | None = None,
) -> dict:
    # Literal split rejection must precede all path reads and runtime imports.
    readiness.require_train_split(args.split)
    output_path = Path(args.output).expanduser().absolute()
    if output_path.exists():
        raise FileExistsError(f"refusing to replace existing receipt: {output_path}")
    probe_manifest_path = Path(args.probe_manifest).expanduser().resolve(strict=True)
    readiness_manifest_path = Path(args.readiness_manifest).expanduser().resolve(
        strict=True
    )
    probe_manifest = json.loads(probe_manifest_path.read_text(encoding="utf-8"))
    readiness_manifest = json.loads(
        readiness_manifest_path.read_text(encoding="utf-8")
    )
    if readiness._file_sha256(readiness_manifest_path) != (
        EXPECTED_READINESS_MANIFEST_RAW_SHA256
    ):
        raise ProtocolError("v5 readiness manifest file bytes drifted")
    contract = validate_probe_manifest(probe_manifest, readiness_manifest)
    implementation_receipt = verify_probe_implementation(probe_manifest)
    lineage_fn = lineage_provider or production_lineage
    preflight = dict(
        lineage_fn(args, readiness_manifest, contract["readiness"])
    )
    if preflight.get("status") != "pass":
        raise ProtocolError("causal probe lineage/preflight did not pass")
    preflight["probe_implementation"] = implementation_receipt
    if probe_provider is None:
        probe_provider = CanonicalCausalProbeProvider(
            args=args,
            readiness_manifest=readiness_manifest,
            readiness_contract=contract["readiness"],
            preflight=preflight,
            probe_contract=contract,
        )
    payload = validate_probe_payload(
        _provider_execute(
            probe_provider,
            manifest=probe_manifest,
            contract=contract,
            preflight=preflight,
            args=args,
        ),
        contract,
    )
    frozen = adjudicate_frozen_attribution(
        payload,
        thresholds=contract["thresholds"],
        attribution_rule=contract["attribution_rule"],
    )
    for arm in ARMS:
        payload["arms"][arm]["recovered"] = frozen["arms"][arm]["recovered"]
        payload["arms"][arm]["recovery_checks"] = frozen["arms"][arm]
    payload["teacher_forced"]["judgement"] = frozen["teacher"]
    receipt = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "stage": "causal_probe",
        "split": TRAIN_SPLIT,
        "manifest": {
            "path": str(probe_manifest_path),
            "file_sha256": readiness._file_sha256(probe_manifest_path),
            "canonical_sha256": readiness._json_hash(probe_manifest),
        },
        "readiness_lineage": {
            "manifest_file_sha256": EXPECTED_READINESS_MANIFEST_RAW_SHA256,
            "manifest_canonical_sha256": (
                EXPECTED_READINESS_MANIFEST_CANONICAL_SHA256
            ),
            "receipt_file_sha256": EXPECTED_READINESS_RECEIPT_FILE_SHA256,
            "receipt_sha256": EXPECTED_READINESS_RECEIPT_SHA256,
            "candidate_checkpoint_sha256": EXPECTED_CANDIDATE_SHA256,
            "candidate_policy_state_sha256": EXPECTED_POLICY_STATE_SHA256,
            "gate_pass": False,
            "promotion": None,
        },
        "preflight": preflight,
        "result": payload,
        "frozen_attribution": frozen,
        "authorize_feature_v2": frozen["authorize_feature_v2"],
        "claim_boundary": dict(probe_manifest["claim_boundary"]),
        "promotion": None,
    }
    receipt["receipt_sha256"] = readiness._json_hash(receipt)
    readiness._atomic_json(output_path, receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True)
    parser.add_argument("--probe-manifest", type=Path, required=True)
    parser.add_argument("--readiness-manifest", type=Path, required=True)
    parser.add_argument("--readiness-receipt", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--rlinf-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--vision", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--expert-receipt", type=Path, required=True)
    parser.add_argument("--keep-receipt", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
