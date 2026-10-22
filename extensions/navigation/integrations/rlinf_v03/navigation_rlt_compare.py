#!/usr/bin/env python3
"""Strict matched comparison of Native, A3, and Navigation-RLT evaluations.

Each input is a completed ``navigation_rlt_matched_train_development_eval``
receipt produced by ``navigation_rlt_eval.py``.  The comparator recomputes
navigation metrics from the 32 episode rows and fails closed when run, split,
protocol, model, checkpoint, or ordered episode identities differ.

The result is descriptive, single-seed, and train-only.  Passing the operational
gate does not establish validation generalization, statistical reliability,
component-level causality, or real-robot safety.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Mapping, Sequence


INPUT_SCHEMA = "navigation_rlt_matched_train_development_eval"
INPUT_VERSION = 1
OUTPUT_SCHEMA = "navigation_rlt_matched_comparison"
OUTPUT_VERSION = 1
EVALUATION_INVENTORY_SCHEMA = "navigation_rlt_evaluation_code_inventory"
EVALUATION_INVENTORY_VERSION = 1
EXPECTED_EPISODES = 32
EVALUATION_CODE_FILES = (
    "build_navigation_rlt_train_split.py",
    "lightnav_a3_policy.py",
    "lightnav_a3_rollout.py",
    "lightnav_l0_native_probe.py",
    "navigation_rlt_action.py",
    "navigation_rlt_compare.py",
    "navigation_rlt_eval.py",
    "navigation_rlt_obs_archive.py",
    "navigation_rlt_parallel.py",
    "navigation_rlt_policy.py",
    "navigation_rlt_post_training.py",
    "navigation_rlt_replay.py",
    "navigation_rlt_rollout.py",
    "streamvln_executor_contract.py",
    "streamvln_topology.py",
    "navrl/__init__.py",
    "navrl/contracts.py",
    "navrl/envs/__init__.py",
    "navrl/envs/base.py",
    "navrl/envs/grid_mock.py",
    "navrl/episode.py",
    "navrl/metrics.py",
    "navrl/policies/__init__.py",
    "navrl/policies/base.py",
    "navrl/rewards.py",
    "navrl/stack/__init__.py",
    "navrl/stack/checkpoint.py",
    "navrl/stack/collector.py",
    "navrl/stack/contracts.py",
    "navrl/stack/execution.py",
    "navrl/stack/interfaces.py",
    "navrl/stack/legacy.py",
    "navrl/stack/ppo.py",
    "navrl/stack/returns.py",
    "run_navigation_rlt_ebk_eval.sh",
)

ARM_NATIVE = "native_lightnav"
ARM_A3 = "a3_ppo"
ARM_RLT = "navigation_rlt"
ARM_ORDER = (ARM_NATIVE, ARM_A3, ARM_RLT)
BEHAVIOR_BY_ARM = {
    ARM_NATIVE: "native",
    ARM_A3: "a3",
    ARM_RLT: "navigation-rlt",
}
SOURCE_KIND_BY_ARM = dict(BEHAVIOR_BY_ARM)
BEHAVIOR_POLICY_ID = "navigation_rlt_warmstart_lineage_v1"
MODE_NAMES = ("KEEP", "ADJUST", "STOP", "RECOVER")
STOP_SOURCES = (
    "native_lightnav_stop",
    "learned_stop",
    "budget_stop",
    "none",
)

SUCCESS_GAIN_MIN = 2
SPL_DELTA_MIN = -0.01
FINAL_NE_DELTA_MAX_M = 0.10
ACTION_COLLAPSE_STRICTLY_ABOVE = 0.98
RECOVER_LOOP_MIN_STREAK = 2

SHA256_RE = re.compile(r"[0-9a-f]{64}")
SHA1_RE = re.compile(r"[0-9a-f]{40}")


class ComparisonError(RuntimeError):
    """Raised when the receipts do not prove a strict matched comparison."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ComparisonError(f"value is not canonical JSON: {exc}") from exc


def _object_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ComparisonError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _sha1(value: object, name: str) -> str:
    if not isinstance(value, str) or SHA1_RE.fullmatch(value) is None:
        raise ComparisonError(f"{name} must be a lowercase full Git SHA-1")
    return value


def _evaluation_source(filename: str) -> Path:
    relative = PurePosixPath(filename)
    code_root = Path(__file__).resolve(strict=True).parent
    candidates = [code_root.joinpath(*relative.parts)]
    if relative.parts and relative.parts[0] == "navrl":
        candidates.append(code_root.parents[1].joinpath(*relative.parts))
    found: list[Path] = []
    for candidate in candidates:
        if candidate.is_symlink():
            continue
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        if resolved.is_file():
            found.append(resolved)
    if len(found) != 1:
        raise ComparisonError(
            f"evaluation source {filename} is absent or ambiguous in the local layout"
        )
    return found[0]


def _load_evaluation_inventory(
    path: Path, *, expected_sha256: str
) -> dict[str, object]:
    raw = Path(path).expanduser()
    if raw.is_symlink():
        raise ComparisonError("evaluation code inventory must not be a symbolic link")
    resolved = raw.resolve(strict=True)
    if not resolved.is_file():
        raise ComparisonError("evaluation code inventory is not a regular file")
    observed_sha = sha256_file(resolved)
    if observed_sha != _sha256(
        expected_sha256, "expected evaluation code inventory SHA-256"
    ):
        raise ComparisonError("evaluation code inventory SHA-256 differs")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ComparisonError("evaluation code inventory is invalid JSON") from exc
    if (
        not isinstance(value, dict)
        or set(value)
        != {"schema_name", "schema_version", "navigation_code_commit", "files"}
        or value.get("schema_name") != EVALUATION_INVENTORY_SCHEMA
        or value.get("schema_version") != EVALUATION_INVENTORY_VERSION
    ):
        raise ComparisonError("evaluation code inventory schema differs")
    commit = _sha1(value.get("navigation_code_commit"), "evaluation code commit")
    files = _mapping(value.get("files"), "evaluation code inventory files")
    if set(files) != set(EVALUATION_CODE_FILES):
        raise ComparisonError("evaluation code inventory dependency set differs")
    normalized: dict[str, str] = {}
    for filename, raw_sha in files.items():
        if not isinstance(filename, str):
            raise ComparisonError("evaluation code inventory filename is invalid")
        relative = PurePosixPath(filename)
        if (
            not relative.parts
            or relative.is_absolute()
            or relative.as_posix() != filename
            or any(part in ("", ".", "..") for part in relative.parts)
        ):
            raise ComparisonError(f"evaluation source path is not canonical: {filename!r}")
        expected = _sha256(raw_sha, f"evaluation source {filename}")
        source = _evaluation_source(filename)
        if sha256_file(source) != expected:
            raise ComparisonError(f"evaluation source {filename} SHA-256 differs")
        normalized[filename] = expected
    return {
        "commit": commit,
        "inventory_sha256": observed_sha,
        "inventory_path": str(resolved),
        "files": normalized,
    }


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ComparisonError(f"{name} must be an object")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ComparisonError(f"{name} must be a non-empty string")
    return value


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ComparisonError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: object, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ComparisonError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ComparisonError(f"{name} must be a finite number")
    if minimum is not None and result < minimum:
        raise ComparisonError(f"{name} must be >= {minimum}")
    return result


def _boolean(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ComparisonError(f"{name} must be boolean")
    return value


def _load(path: Path, arm: str) -> dict[str, object]:
    raw = Path(path).expanduser()
    if raw.is_symlink():
        raise ComparisonError(f"{arm} input must not be a symbolic link")
    resolved = raw.resolve(strict=True)
    if not resolved.is_file():
        raise ComparisonError(f"{arm} input is not a regular file")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ComparisonError(f"{arm} input is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ComparisonError(f"{arm} input must contain an object")
    return value


def _run_manifest_sha(receipt: Mapping[str, object], arm: str) -> str:
    direct = receipt.get("run_manifest_sha256")
    nested_value = receipt.get("run_manifest")
    nested = None
    if nested_value is not None:
        nested = _mapping(nested_value, f"{arm}.run_manifest").get("sha256")
    if direct is None and nested is None:
        raise ComparisonError(f"{arm} lacks run_manifest_sha256")
    if direct is not None and nested is not None and direct != nested:
        raise ComparisonError(f"{arm} run-manifest identities disagree")
    return _sha256(direct if direct is not None else nested, f"{arm}.run_manifest_sha256")


def _validate_model(
    receipt: Mapping[str, object],
    *,
    arm: str,
    expected_model_sha256: str,
    expected_lightnav_weight_sha256: str,
) -> dict[str, object]:
    model = dict(_mapping(receipt.get("model_identity"), f"{arm}.model_identity"))
    expected_keys = {
        "lightnav_source_revision",
        "lightnav_hook_sha256",
        "lightnav_weight_sha256",
        "tokenizer_sha256",
        "processor_sha256",
        "history_config_sha256",
        "original_release_weights",
    }
    if set(model) != expected_keys:
        raise ComparisonError(f"{arm}.model_identity members differ from the eval contract")
    _sha1(model["lightnav_source_revision"], f"{arm}.model.lightnav_source_revision")
    for key in (
        "lightnav_hook_sha256",
        "lightnav_weight_sha256",
        "tokenizer_sha256",
        "processor_sha256",
        "history_config_sha256",
    ):
        _sha256(model[key], f"{arm}.model.{key}")
    if model["lightnav_weight_sha256"] != expected_lightnav_weight_sha256:
        raise ComparisonError(f"{arm} LightNav weight SHA-256 differs")
    if _boolean(model["original_release_weights"], f"{arm}.model.original_release_weights") is not True:
        raise ComparisonError(f"{arm} did not use original release weights")
    if _object_sha256(model) != expected_model_sha256:
        raise ComparisonError(f"{arm} model identity SHA-256 differs")
    return model


def _validate_checkpoint(
    receipt: Mapping[str, object],
    *,
    arm: str,
    expected_lightnav_weight_sha256: str,
    expected_checkpoint_sha256: str | None,
) -> dict[str, object]:
    behavior = BEHAVIOR_BY_ARM[arm]
    if receipt.get("behavior") != behavior:
        raise ComparisonError(f"{arm} behavior label differs")
    version = _integer(
        receipt.get("behavior_policy_version"), f"{arm}.behavior_policy_version"
    )
    source = _mapping(receipt.get("behavior_source"), f"{arm}.behavior_source")
    if source.get("kind") != SOURCE_KIND_BY_ARM[arm]:
        raise ComparisonError(f"{arm} behavior source kind differs")
    if arm == ARM_NATIVE:
        if version != 0:
            raise ComparisonError("native LightNav policy version must be zero")
        _sha1(source.get("lightnav_revision"), "native LightNav revision")
        return {
            "kind": "original_lightnav",
            "sha256": expected_lightnav_weight_sha256,
            "policy_version": None,
        }
    checkpoint_sha = _sha256(
        source.get("checkpoint_sha256"), f"{arm}.behavior_source.checkpoint_sha256"
    )
    if checkpoint_sha != expected_checkpoint_sha256:
        raise ComparisonError(f"{arm} checkpoint SHA-256 differs")
    checkpoint_version = _integer(
        source.get("checkpoint_policy_version"),
        f"{arm}.behavior_source.checkpoint_policy_version",
    )
    expected_behavior_version = (
        100_000 + checkpoint_version
        if arm == ARM_A3
        else 1_000_000 + checkpoint_version
    )
    if version != expected_behavior_version:
        raise ComparisonError(f"{arm} behavior and checkpoint policy versions differ")
    return {
        "kind": arm,
        "sha256": checkpoint_sha,
        "policy_version": checkpoint_version,
    }


def _validate_latency(value: object, name: str, *, expected_count: int) -> dict[str, object]:
    latency = _mapping(value, name)
    expected_keys = {"count", "mean_ms", "p50_ms", "p95_ms", "max_ms"}
    if set(latency) != expected_keys:
        raise ComparisonError(f"{name} members differ")
    count = _integer(latency["count"], f"{name}.count", minimum=1)
    if count != expected_count:
        raise ComparisonError(f"{name}.count differs from actor decisions")
    result: dict[str, object] = {"count": count}
    for key in ("mean_ms", "p50_ms", "p95_ms", "max_ms"):
        result[key] = _number(latency[key], f"{name}.{key}", minimum=0.0)
    if not (
        float(result["p50_ms"])
        <= float(result["p95_ms"])
        <= float(result["max_ms"])
        and float(result["mean_ms"]) <= float(result["max_ms"])
    ):
        raise ComparisonError(f"{name} percentile ordering is invalid")
    return result


def _validate_modes(value: object, name: str, *, expected_total: int) -> dict[str, int]:
    modes = _mapping(value, name)
    if set(modes) != set(MODE_NAMES):
        raise ComparisonError(f"{name} must contain exactly {MODE_NAMES}")
    result = {mode: _integer(modes[mode], f"{name}.{mode}") for mode in MODE_NAMES}
    if sum(result.values()) != expected_total:
        raise ComparisonError(f"{name} total differs from actor_decision_count")
    return result


def _recover_streak(row: Mapping[str, object], executed_recover: int, name: str) -> int:
    value = row.get("max_consecutive_recover")
    if value is not None:
        streak = _integer(value, f"{name}.max_consecutive_recover")
        if streak > executed_recover:
            raise ComparisonError(f"{name}.max_consecutive_recover exceeds RECOVER count")
        return streak
    # With zero or one executed RECOVER, a loop is mathematically impossible.
    # Two or more require sequence evidence; totals alone cannot prove separation.
    if executed_recover <= 1:
        return executed_recover
    raise ComparisonError(
        f"{name} has multiple RECOVER actions but no max_consecutive_recover evidence"
    )


def _validate_episode(
    value: object,
    *,
    arm: str,
    expected_order: int,
    expected_behavior_version: int,
) -> dict[str, object]:
    row = _mapping(value, f"{arm}.episodes[{expected_order}]")
    name = f"{arm}.episodes[{expected_order}]"
    order = _integer(row.get("episode_order"), f"{name}.episode_order")
    if order != expected_order:
        raise ComparisonError(f"{arm} episode order is not 0..31")
    if row.get("behavior") != BEHAVIOR_BY_ARM[arm]:
        raise ComparisonError(f"{name}.behavior differs")
    if (
        _integer(row.get("behavior_policy_version"), f"{name}.behavior_policy_version")
        != expected_behavior_version
    ):
        raise ComparisonError(f"{name}.behavior policy version differs")
    success = _boolean(row.get("success"), f"{name}.success")
    timeout = _boolean(row.get("timeout"), f"{name}.timeout")
    if success and timeout:
        raise ComparisonError(f"{name} cannot be both success and timeout")
    spl = _number(row.get("spl"), f"{name}.spl", minimum=0.0)
    if spl > 1.0:
        raise ComparisonError(f"{name}.spl exceeds one")
    decisions = _integer(
        row.get("actor_decision_count"), f"{name}.actor_decision_count", minimum=1
    )
    requested = _validate_modes(
        row.get("requested_mode_distribution"),
        f"{name}.requested_mode_distribution",
        expected_total=decisions,
    )
    executed = _validate_modes(
        row.get("executed_mode_distribution"),
        f"{name}.executed_mode_distribution",
        expected_total=decisions,
    )
    stop_source = _string(row.get("stop_source"), f"{name}.stop_source")
    if stop_source not in STOP_SOURCES:
        raise ComparisonError(f"{name}.stop_source is unsupported")
    context_requests = _integer(
        row.get("context_request_count"), f"{name}.context_request_count"
    )
    if arm == ARM_RLT:
        if context_requests != decisions:
            raise ComparisonError(f"{name} lacks one context request per decision")
    elif context_requests != 0:
        raise ComparisonError(f"{name} unexpectedly requested contextual features")
    latency = _validate_latency(
        row.get("model_latency"), f"{name}.model_latency", expected_count=decisions
    )
    max_recover_streak = _recover_streak(row, executed["RECOVER"], name)
    return {
        "episode_order": order,
        "episode_id": _string(row.get("episode_id"), f"{name}.episode_id"),
        "scene": _string(row.get("scene"), f"{name}.scene"),
        "instruction_text_sha256": _sha256(
            row.get("instruction_text_sha256"), f"{name}.instruction_text_sha256"
        ),
        "success": success,
        "spl": spl,
        "final_navigation_error_m": _number(
            row.get("final_navigation_error_m"),
            f"{name}.final_navigation_error_m",
            minimum=0.0,
        ),
        "path_length_m": _number(
            row.get("path_length_m"), f"{name}.path_length_m", minimum=0.0
        ),
        "collision_count": _number(
            row.get("collision_count"), f"{name}.collision_count", minimum=0.0
        ),
        "blocked_step_count": _integer(
            row.get("blocked_step_count"), f"{name}.blocked_step_count"
        ),
        "stop_source": stop_source,
        "timeout": timeout,
        "actor_decision_count": decisions,
        "requested_mode_distribution": requested,
        "executed_mode_distribution": executed,
        "safety_override_count": _integer(
            row.get("safety_override_count"), f"{name}.safety_override_count"
        ),
        "context_request_count": context_requests,
        "model_latency": latency,
        "max_consecutive_recover": max_recover_streak,
    }


def _episode_identity(row: Mapping[str, object]) -> tuple[object, ...]:
    return (
        row["episode_order"],
        row["scene"],
        row["episode_id"],
        row["instruction_text_sha256"],
    )


def _protocol_contract(
    receipt: Mapping[str, object],
    *,
    arm: str,
    expected_run_id: str,
    expected_eval_plan_sha256: str,
    expected_split_manifest_sha256: str,
    expected_seed: int,
) -> dict[str, object]:
    if receipt.get("run_id") != expected_run_id:
        raise ComparisonError(f"{arm} run id differs")
    if receipt.get("behavior_policy_id") != BEHAVIOR_POLICY_ID:
        raise ComparisonError(f"{arm} behavior policy id differs")
    if (
        receipt.get("split_label") != "train_development"
        or receipt.get("upstream_split_read") != "train"
        or receipt.get("validation_split_reads") != 0
    ):
        raise ComparisonError(f"{arm} is not train-only train_development")
    if _boolean(receipt.get("deterministic_policy"), f"{arm}.deterministic_policy") is not True:
        raise ComparisonError(f"{arm} evaluation is not deterministic")
    if _boolean(receipt.get("training_replay_written"), f"{arm}.training_replay_written") is not False:
        raise ComparisonError(f"{arm} evaluation wrote training replay")
    if _integer(receipt.get("seed"), f"{arm}.seed") != expected_seed:
        raise ComparisonError(f"{arm} evaluation seed differs")

    plan = _mapping(receipt.get("evaluation_plan"), f"{arm}.evaluation_plan")
    plan_sha = _sha256(plan.get("sha256"), f"{arm}.evaluation_plan.sha256")
    split_sha = _sha256(
        plan.get("split_manifest_sha256"),
        f"{arm}.evaluation_plan.split_manifest_sha256",
    )
    if plan_sha != expected_eval_plan_sha256:
        raise ComparisonError(f"{arm} evaluation plan SHA-256 differs")
    if split_sha != expected_split_manifest_sha256:
        raise ComparisonError(f"{arm} split manifest SHA-256 differs")

    executor = dict(_mapping(receipt.get("executor"), f"{arm}.executor"))
    if set(executor) != {
        "schema",
        "max_physical_steps",
        "jpeg_quality",
        "recover_loop_definition",
    }:
        raise ComparisonError(f"{arm}.executor members differ")
    _string(executor["schema"], f"{arm}.executor.schema")
    if _integer(executor["max_physical_steps"], f"{arm}.executor.max_physical_steps") != 160:
        raise ComparisonError(f"{arm} max physical steps is not 160")
    quality = _integer(executor["jpeg_quality"], f"{arm}.executor.jpeg_quality", minimum=1)
    if quality > 100:
        raise ComparisonError(f"{arm} jpeg quality exceeds 100")
    if executor["recover_loop_definition"] != "executed consecutive RECOVER >= 2":
        raise ComparisonError(f"{arm} recover-loop definition differs")

    schemas = dict(_mapping(receipt.get("schemas"), f"{arm}.schemas"))
    if set(schemas) != {"reference", "topology", "topology_node_dim", "action"}:
        raise ComparisonError(f"{arm}.schemas members differ")
    for key in ("reference", "topology", "action"):
        _string(schemas[key], f"{arm}.schemas.{key}")
    _integer(schemas["topology_node_dim"], f"{arm}.schemas.topology_node_dim", minimum=1)
    return {
        "run_id": expected_run_id,
        "behavior_policy_id": BEHAVIOR_POLICY_ID,
        "evaluation_plan_sha256": plan_sha,
        "split_manifest_sha256": split_sha,
        "split_label": "train_development",
        "upstream_split_read": "train",
        "validation_split_reads": 0,
        "deterministic_policy": True,
        "training_replay_written": False,
        "seed": expected_seed,
        "executor": executor,
        "schemas": schemas,
    }


def _validate_context(
    receipt: Mapping[str, object], *, arm: str, decision_count: int
) -> dict[str, object]:
    context = dict(_mapping(receipt.get("context"), f"{arm}.context"))
    expected_keys = {
        "requested",
        "request_count",
        "source_sha256",
        "artifact_capture_rule",
        "prefix_feature_schema",
        "prefix_selection_schema",
        "prefix_selection_rule",
        "full_artifacts_cleaned",
    }
    if set(context) != expected_keys:
        raise ComparisonError(f"{arm}.context members differ")
    requested = _boolean(context["requested"], f"{arm}.context.requested")
    request_count = _integer(context["request_count"], f"{arm}.context.request_count")
    cleaned = _boolean(
        context["full_artifacts_cleaned"], f"{arm}.context.full_artifacts_cleaned"
    )
    feature_keys = (
        "source_sha256",
        "artifact_capture_rule",
        "prefix_feature_schema",
        "prefix_selection_schema",
        "prefix_selection_rule",
    )
    if arm == ARM_RLT:
        if not requested or request_count != decision_count or not cleaned:
            raise ComparisonError("Navigation-RLT contextual request/cleanup counts differ")
        _sha256(context["source_sha256"], "Navigation-RLT context source SHA")
        for key in feature_keys[1:]:
            if context[key] is None:
                raise ComparisonError(f"Navigation-RLT context lacks {key}")
    else:
        if requested or request_count != 0 or cleaned:
            raise ComparisonError(f"{arm} unexpectedly used contextual artifacts")
        if any(context[key] is not None for key in feature_keys):
            raise ComparisonError(f"{arm} contains contextual feature metadata")
    return context


def _assert_close(observed: object, expected: float, name: str) -> None:
    value = _number(observed, name)
    if not math.isclose(value, expected, rel_tol=1.0e-9, abs_tol=1.0e-9):
        raise ComparisonError(f"{name} differs from recomputed episodes")


def _mode_summary(rows: Sequence[Mapping[str, object]], key: str) -> dict[str, object]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(dict(row[key]))
    normalized = {mode: int(counts[mode]) for mode in MODE_NAMES}
    total = sum(normalized.values())
    if total <= 0:
        raise ComparisonError(f"{key} has no actions")
    ratios = {mode: normalized[mode] / total for mode in MODE_NAMES}
    max_mode = max(MODE_NAMES, key=lambda mode: (ratios[mode], -MODE_NAMES.index(mode)))
    return {
        "total": total,
        "counts": normalized,
        "ratios": ratios,
        "max_mode": max_mode,
        "max_mode_share": ratios[max_mode],
        "collapsed_strictly_above_0_98": ratios[max_mode] > ACTION_COLLAPSE_STRICTLY_ABOVE,
    }


def _mean(rows: Sequence[Mapping[str, object]], key: str) -> float:
    return math.fsum(float(row[key]) for row in rows) / len(rows)


def _summary(rows: Sequence[Mapping[str, object]], latency: Mapping[str, object]) -> dict[str, object]:
    count = len(rows)
    success_count = sum(bool(row["success"]) for row in rows)
    collisions = math.fsum(float(row["collision_count"]) for row in rows)
    blocked = sum(int(row["blocked_step_count"]) for row in rows)
    stop_counts = Counter(str(row["stop_source"]) for row in rows)
    timeouts = sum(bool(row["timeout"]) for row in rows)
    decisions = sum(int(row["actor_decision_count"]) for row in rows)
    max_recover = max(int(row["max_consecutive_recover"]) for row in rows)
    loop_episodes = sum(
        int(row["max_consecutive_recover"]) >= RECOVER_LOOP_MIN_STREAK for row in rows
    )
    return {
        "episode_count": count,
        "success_count": success_count,
        "success_rate": success_count / count,
        "mean_spl": _mean(rows, "spl"),
        "mean_final_navigation_error_m": _mean(rows, "final_navigation_error_m"),
        "mean_path_length_m": _mean(rows, "path_length_m"),
        "collision_count": collisions,
        "mean_collision_count": collisions / count,
        "blocked_step_count": blocked,
        "mean_blocked_step_count": blocked / count,
        "native_stop_count": int(stop_counts["native_lightnav_stop"]),
        "learned_stop_count": int(stop_counts["learned_stop"]),
        "budget_stop_count": int(stop_counts["budget_stop"]),
        "no_stop_count": int(stop_counts["none"]),
        "native_stop_rate": stop_counts["native_lightnav_stop"] / count,
        "learned_stop_rate": stop_counts["learned_stop"] / count,
        "budget_stop_rate": stop_counts["budget_stop"] / count,
        "timeout_count": timeouts,
        "timeout_rate": timeouts / count,
        "actor_decision_count": decisions,
        "requested_modes": _mode_summary(rows, "requested_mode_distribution"),
        "executed_modes": _mode_summary(rows, "executed_mode_distribution"),
        "safety_override_count": sum(int(row["safety_override_count"]) for row in rows),
        "max_consecutive_recover": max_recover,
        "recover_loop_episode_count": loop_episodes,
        "model_latency": dict(latency),
    }


def _validate_aggregate(
    raw: object,
    *,
    arm: str,
    summary: Mapping[str, object],
) -> None:
    aggregate = _mapping(raw, f"{arm}.aggregate")
    numeric = {
        "episode_count": summary["episode_count"],
        "success_count": summary["success_count"],
        "success_rate": summary["success_rate"],
        "SR": summary["success_rate"],
        "mean_spl": summary["mean_spl"],
        "SPL": summary["mean_spl"],
        "mean_final_navigation_error_m": summary["mean_final_navigation_error_m"],
        "final_NE_m": summary["mean_final_navigation_error_m"],
        "mean_path_length_m": summary["mean_path_length_m"],
        "path_m": summary["mean_path_length_m"],
        "collision_count": summary["collision_count"],
        "collisions": summary["collision_count"],
        "blocked_step_count": summary["blocked_step_count"],
        "blocked": summary["blocked_step_count"],
        "native_stop_count": summary["native_stop_count"],
        "learned_stop_count": summary["learned_stop_count"],
        "timeout_count": summary["timeout_count"],
        "timeouts": summary["timeout_count"],
        "actor_decision_count": summary["actor_decision_count"],
        "safety_override_count": summary["safety_override_count"],
    }
    for key, expected in numeric.items():
        if key not in aggregate:
            raise ComparisonError(f"{arm}.aggregate lacks {key}")
        _assert_close(aggregate[key], float(expected), f"{arm}.aggregate.{key}")
    for key, summary_key in (
        ("requested_mode_distribution", "requested_modes"),
        ("executed_mode_distribution", "executed_modes"),
    ):
        observed = _validate_modes(
            aggregate.get(key),
            f"{arm}.aggregate.{key}",
            expected_total=int(summary["actor_decision_count"]),
        )
        if observed != summary[summary_key]["counts"]:
            raise ComparisonError(f"{arm}.aggregate.{key} differs from episodes")
    aggregate_latency = _validate_latency(
        aggregate.get("latency"),
        f"{arm}.aggregate.latency",
        expected_count=int(summary["actor_decision_count"]),
    )
    if aggregate_latency != summary["model_latency"]:
        raise ComparisonError(f"{arm}.aggregate.latency normalization differs")
    if "model_latency" in aggregate and aggregate["model_latency"] != aggregate["latency"]:
        raise ComparisonError(f"{arm}.aggregate latency aliases disagree")


def _validate_arm(
    receipt: Mapping[str, object],
    *,
    arm: str,
    expected_run_id: str,
    expected_run_manifest_sha256: str,
    expected_eval_plan_sha256: str,
    expected_split_manifest_sha256: str,
    expected_protocol_sha256: str,
    expected_model_sha256: str,
    expected_lightnav_weight_sha256: str,
    expected_checkpoint_sha256: str | None,
    expected_seed: int,
    expected_evaluation_code_commit: str,
    expected_evaluation_inventory_sha256: str,
) -> dict[str, object]:
    if (
        receipt.get("schema_name") != INPUT_SCHEMA
        or receipt.get("schema_version") != INPUT_VERSION
        or receipt.get("status") != "completed"
    ):
        raise ComparisonError(f"{arm} is not a completed {INPUT_SCHEMA} v1 receipt")
    if _run_manifest_sha(receipt, arm) != expected_run_manifest_sha256:
        raise ComparisonError(f"{arm} run manifest SHA-256 differs")
    evaluation_code = _mapping(receipt.get("evaluation_code"), f"{arm}.evaluation_code")
    if set(evaluation_code) != {"commit", "inventory_sha256"}:
        raise ComparisonError(f"{arm} evaluation code identity members differ")
    if (
        _sha1(evaluation_code.get("commit"), f"{arm} evaluation code commit")
        != expected_evaluation_code_commit
        or _sha256(
            evaluation_code.get("inventory_sha256"),
            f"{arm} evaluation code inventory SHA-256",
        )
        != expected_evaluation_inventory_sha256
    ):
        raise ComparisonError(f"{arm} evaluation code identity differs")
    protocol = _protocol_contract(
        receipt,
        arm=arm,
        expected_run_id=expected_run_id,
        expected_eval_plan_sha256=expected_eval_plan_sha256,
        expected_split_manifest_sha256=expected_split_manifest_sha256,
        expected_seed=expected_seed,
    )
    if _object_sha256(protocol) != expected_protocol_sha256:
        raise ComparisonError(f"{arm} protocol SHA-256 differs")
    model = _validate_model(
        receipt,
        arm=arm,
        expected_model_sha256=expected_model_sha256,
        expected_lightnav_weight_sha256=expected_lightnav_weight_sha256,
    )
    checkpoint = _validate_checkpoint(
        receipt,
        arm=arm,
        expected_lightnav_weight_sha256=expected_lightnav_weight_sha256,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
    )
    raw_rows = receipt.get("episodes")
    if not isinstance(raw_rows, list) or len(raw_rows) != EXPECTED_EPISODES:
        raise ComparisonError(f"{arm} must contain exactly 32 episodes")
    behavior_version = _integer(
        receipt.get("behavior_policy_version"), f"{arm}.behavior_policy_version"
    )
    rows = [
        _validate_episode(
            row,
            arm=arm,
            expected_order=index,
            expected_behavior_version=behavior_version,
        )
        for index, row in enumerate(raw_rows)
    ]
    identities = [_episode_identity(row) for row in rows]
    if len(set(identities)) != EXPECTED_EPISODES:
        raise ComparisonError(f"{arm} contains duplicate episode identities")
    decisions = sum(int(row["actor_decision_count"]) for row in rows)
    aggregate = _mapping(receipt.get("aggregate"), f"{arm}.aggregate")
    latency = _validate_latency(
        aggregate.get("latency"),
        f"{arm}.aggregate.latency",
        expected_count=decisions,
    )
    summary = _summary(rows, latency)
    _validate_aggregate(aggregate, arm=arm, summary=summary)
    context = _validate_context(receipt, arm=arm, decision_count=decisions)
    return {
        "run_id": receipt["run_id"],
        "collection_manifest_sha256": _run_manifest_sha(receipt, arm),
        "evaluation_code": {
            "commit": expected_evaluation_code_commit,
            "inventory_sha256": expected_evaluation_inventory_sha256,
        },
        "protocol": protocol,
        "model": model,
        "checkpoint": checkpoint,
        "episodes": rows,
        "summary": summary,
        "context": context,
    }


def _numeric_delta(after: Mapping[str, object], before: Mapping[str, object]) -> dict[str, object]:
    keys = (
        "success_count",
        "success_rate",
        "mean_spl",
        "mean_final_navigation_error_m",
        "mean_path_length_m",
        "collision_count",
        "mean_collision_count",
        "blocked_step_count",
        "mean_blocked_step_count",
        "native_stop_count",
        "learned_stop_count",
        "budget_stop_count",
        "timeout_count",
        "timeout_rate",
        "actor_decision_count",
        "safety_override_count",
        "recover_loop_episode_count",
    )
    result: dict[str, object] = {
        key: float(after[key]) - float(before[key]) for key in keys
    }
    result["requested_mode_counts"] = {
        mode: int(after["requested_modes"]["counts"][mode])
        - int(before["requested_modes"]["counts"][mode])
        for mode in MODE_NAMES
    }
    result["executed_mode_counts"] = {
        mode: int(after["executed_modes"]["counts"][mode])
        - int(before["executed_modes"]["counts"][mode])
        for mode in MODE_NAMES
    }
    result["model_latency_ms"] = {
        key: float(after["model_latency"][key]) - float(before["model_latency"][key])
        for key in ("mean_ms", "p50_ms", "p95_ms", "max_ms")
    }
    return result


def _paired_rows(arms: Mapping[str, Mapping[str, object]]) -> list[dict[str, object]]:
    native_rows = arms[ARM_NATIVE]["episodes"]
    a3_rows = arms[ARM_A3]["episodes"]
    rlt_rows = arms[ARM_RLT]["episodes"]
    assert isinstance(native_rows, list) and isinstance(a3_rows, list) and isinstance(rlt_rows, list)
    result: list[dict[str, object]] = []
    for native, a3, rlt in zip(native_rows, a3_rows, rlt_rows):
        identity = _episode_identity(native)
        row: dict[str, object] = {
            "episode_order": identity[0],
            "scene": identity[1],
            "episode_id": identity[2],
            "instruction_text_sha256": identity[3],
            "success": {
                ARM_NATIVE: native["success"],
                ARM_A3: a3["success"],
                ARM_RLT: rlt["success"],
            },
            "stop_source": {
                ARM_NATIVE: native["stop_source"],
                ARM_A3: a3["stop_source"],
                ARM_RLT: rlt["stop_source"],
            },
            "timeout": {
                ARM_NATIVE: native["timeout"],
                ARM_A3: a3["timeout"],
                ARM_RLT: rlt["timeout"],
            },
        }
        for name, first, second in (
            ("a3_minus_native", native, a3),
            ("rlt_minus_a3", a3, rlt),
            ("rlt_minus_native", native, rlt),
        ):
            row[name] = {
                "success": int(bool(second["success"])) - int(bool(first["success"])),
                "spl": float(second["spl"]) - float(first["spl"]),
                "final_navigation_error_m": float(second["final_navigation_error_m"])
                - float(first["final_navigation_error_m"]),
                "path_length_m": float(second["path_length_m"])
                - float(first["path_length_m"]),
                "collision_count": float(second["collision_count"])
                - float(first["collision_count"]),
                "blocked_step_count": int(second["blocked_step_count"])
                - int(first["blocked_step_count"]),
            }
        result.append(row)
    return result


def _gate(
    summaries: Mapping[str, Mapping[str, object]],
    arms: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    a3 = summaries[ARM_A3]
    rlt = summaries[ARM_RLT]
    success_delta = int(rlt["success_count"]) - int(a3["success_count"])
    spl_delta = float(rlt["mean_spl"]) - float(a3["mean_spl"])
    ne_delta = float(rlt["mean_final_navigation_error_m"]) - float(
        a3["mean_final_navigation_error_m"]
    )
    spl_within_threshold = spl_delta > SPL_DELTA_MIN or math.isclose(
        spl_delta, SPL_DELTA_MIN, rel_tol=1.0e-12, abs_tol=1.0e-12
    )
    ne_within_threshold = ne_delta < FINAL_NE_DELTA_MAX_M or math.isclose(
        ne_delta, FINAL_NE_DELTA_MAX_M, rel_tol=1.0e-12, abs_tol=1.0e-12
    )
    a3_rows = arms[ARM_A3]["episodes"]
    rlt_rows = arms[ARM_RLT]["episodes"]
    assert isinstance(a3_rows, list) and isinstance(rlt_rows, list)
    new_timeout_orders = [
        int(rlt_row["episode_order"])
        for a3_row, rlt_row in zip(a3_rows, rlt_rows)
        if not bool(a3_row["timeout"]) and bool(rlt_row["timeout"])
    ]
    new_budget_stop_orders = [
        int(rlt_row["episode_order"])
        for a3_row, rlt_row in zip(a3_rows, rlt_rows)
        if a3_row["stop_source"] != "budget_stop"
        and rlt_row["stop_source"] == "budget_stop"
    ]
    checks = {
        "success_gain_at_least_2": success_delta >= SUCCESS_GAIN_MIN,
        "spl_delta_at_least_minus_0_01": spl_within_threshold,
        "final_ne_delta_at_most_plus_0_10_m": ne_within_threshold,
        "requested_action_max_share_not_above_0_98": not bool(
            rlt["requested_modes"]["collapsed_strictly_above_0_98"]
        ),
        "executed_action_max_share_not_above_0_98": not bool(
            rlt["executed_modes"]["collapsed_strictly_above_0_98"]
        ),
        "no_recover_loop": int(rlt["recover_loop_episode_count"]) == 0,
        "timeout_count_not_increased": int(rlt["timeout_count"])
        <= int(a3["timeout_count"]),
        "budget_stop_count_not_increased": int(rlt["budget_stop_count"])
        <= int(a3["budget_stop_count"]),
        "no_new_timeout_episode": not new_timeout_orders,
        "no_new_budget_stop_episode": not new_budget_stop_orders,
    }
    return {
        "name": "a3_to_navigation_rlt_train_development_gate_v1",
        "thresholds": {
            "success_count_delta_min": SUCCESS_GAIN_MIN,
            "mean_spl_delta_min": SPL_DELTA_MIN,
            "mean_final_navigation_error_delta_max_m": FINAL_NE_DELTA_MAX_M,
            "single_mode_share_fails_if_strictly_above": ACTION_COLLAPSE_STRICTLY_ABOVE,
            "recover_loop_min_consecutive_recover": RECOVER_LOOP_MIN_STREAK,
            "recover_loop_episode_count_max": 0,
            "timeout_and_budget_stop_counts_may_not_increase": True,
            "new_timeout_or_budget_stop_episode_count_max": 0,
        },
        "observed": {
            "success_count_delta": success_delta,
            "mean_spl_delta": spl_delta,
            "mean_final_navigation_error_delta_m": ne_delta,
            "requested_action_max_mode": rlt["requested_modes"]["max_mode"],
            "requested_action_max_share": rlt["requested_modes"]["max_mode_share"],
            "executed_action_max_mode": rlt["executed_modes"]["max_mode"],
            "executed_action_max_share": rlt["executed_modes"]["max_mode_share"],
            "max_consecutive_recover": rlt["max_consecutive_recover"],
            "recover_loop_episode_count": rlt["recover_loop_episode_count"],
            "a3_timeout_count": a3["timeout_count"],
            "rlt_timeout_count": rlt["timeout_count"],
            "a3_budget_stop_count": a3["budget_stop_count"],
            "rlt_budget_stop_count": rlt["budget_stop_count"],
            "new_timeout_episode_orders": new_timeout_orders,
            "new_budget_stop_episode_orders": new_budget_stop_orders,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def _claim_boundary(passed: bool) -> dict[str, object]:
    claim = (
        "Navigation-RLT passed the specified operational gate against A3 on the "
        "single fixed 32-episode R2R train_development protocol."
        if passed
        else "The fixed 32-episode train_development result does not support the "
        "specified Navigation-RLT improvement claim."
    )
    return {
        "verdict": "partial" if passed else "not_supported",
        "supported_claim": claim,
        "scope": "single-seed, deterministic, matched, R2R train-only development episodes",
        "unsupported_claims": [
            "val_seen or val_unseen generalization",
            "multi-seed statistical reliability",
            "real-robot safety or sim-to-real transfer",
            "causal attribution to the multimodal token, topology, or RL objective alone",
            "superiority outside the frozen model, executor, budget, and episode identities",
        ],
        "selection_rule": "the fixed gate is evaluated once; results do not retune or select a checkpoint",
    }


def _markdown(output: Mapping[str, object]) -> str:
    summaries = output["summaries"]
    gate = output["gate"]
    assert isinstance(summaries, Mapping) and isinstance(gate, Mapping)
    metrics = (
        ("success", "success_count", "count / 32"),
        ("SR", "success_rate", "rate"),
        ("SPL", "mean_spl", "mean"),
        ("final NE", "mean_final_navigation_error_m", "m"),
        ("path", "mean_path_length_m", "m"),
        ("collisions", "collision_count", "total"),
        ("blocked", "blocked_step_count", "total"),
        ("native STOP", "native_stop_count", "episodes"),
        ("learned STOP", "learned_stop_count", "episodes"),
        ("budget STOP", "budget_stop_count", "episodes"),
        ("timeout", "timeout_count", "episodes"),
    )
    lines = [
        "# Navigation-RLT matched train-only comparison",
        "",
        "| metric | unit | Native | A3 | Navigation-RLT | RLT - A3 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for label, key, unit in metrics:
        native = float(summaries[ARM_NATIVE][key])
        a3 = float(summaries[ARM_A3][key])
        rlt = float(summaries[ARM_RLT][key])
        lines.append(
            f"| {label} | {unit} | {native:.6g} | {a3:.6g} | {rlt:.6g} | {rlt - a3:+.6g} |"
        )
    for label, latency_key in (("latency mean", "mean_ms"), ("latency p95", "p95_ms")):
        native = float(summaries[ARM_NATIVE]["model_latency"][latency_key])
        a3 = float(summaries[ARM_A3]["model_latency"][latency_key])
        rlt = float(summaries[ARM_RLT]["model_latency"][latency_key])
        lines.append(
            f"| {label} | ms | {native:.6g} | {a3:.6g} | {rlt:.6g} | {rlt - a3:+.6g} |"
        )
    lines.extend(("", "## Action distributions", ""))
    lines.extend(("| arm | route | mode | count | ratio |", "|---|---|---|---:|---:|"))
    for arm in ARM_ORDER:
        for route in ("requested_modes", "executed_modes"):
            for mode in MODE_NAMES:
                lines.append(
                    f"| {arm} | {route.removesuffix('_modes')} | {mode} | "
                    f"{summaries[arm][route]['counts'][mode]} | "
                    f"{summaries[arm][route]['ratios'][mode]:.6f} |"
                )
    lines.extend(("", "## A3 to Navigation-RLT gate", ""))
    lines.append(f"**{'PASS' if gate['passed'] else 'FAIL'}**")
    lines.extend(("", "| check | result |", "|---|---|"))
    for name, passed in gate["checks"].items():
        lines.append(f"| {name} | {'pass' if passed else 'fail'} |")
    boundary = output["result_to_claim_boundary"]
    lines.extend(
        (
            "",
            "## Result-to-claim boundary",
            "",
            str(boundary["supported_claim"]),
            "",
            "This is one deterministic seed on the fixed 32 train-only development "
            "episodes. It does not establish validation generalization, multi-seed "
            "reliability, real-robot safety, or component-level causality.",
            "",
        )
    )
    return "\n".join(lines)


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".staging-", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    temporary_identity = _file_identity(os.fstat(descriptor))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        _publish_file_new(temporary, path, temporary_identity)
    except BaseException:
        _unlink_if_same(temporary, temporary_identity)
        raise


def _fsync_parent(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path.parent, flags)
    except OSError:
        if os.name == "nt":
            return
        raise
    try:
        os.fsync(descriptor)
    except OSError:
        if os.name != "nt":
            raise
    finally:
        os.close(descriptor)


def _file_identity(status: os.stat_result) -> tuple[int, int]:
    return int(status.st_dev), int(status.st_ino)


def _unlink_if_same(path: Path, identity: tuple[int, int]) -> None:
    try:
        status = path.lstat()
    except FileNotFoundError:
        return
    if _file_identity(status) != identity:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _publish_file_new(
    temporary: Path, destination: Path, identity: tuple[int, int]
) -> None:
    if temporary.parent.resolve(strict=True) != destination.parent.resolve(strict=True):
        raise ComparisonError("atomic publication requires one destination directory")
    os.link(temporary, destination, follow_symlinks=False)
    _fsync_parent(destination)
    _unlink_if_same(temporary, identity)
    _fsync_parent(destination)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--native", required=True, type=Path)
    result.add_argument("--a3", required=True, type=Path)
    result.add_argument("--navigation-rlt", required=True, type=Path)
    result.add_argument("--expected-run-id", required=True)
    result.add_argument("--expected-run-manifest-sha256", required=True)
    result.add_argument("--expected-eval-plan-sha256", required=True)
    result.add_argument("--expected-split-manifest-sha256", required=True)
    result.add_argument("--expected-protocol-sha256", required=True)
    result.add_argument("--expected-model-sha256", required=True)
    result.add_argument("--expected-lightnav-weight-sha256", required=True)
    result.add_argument("--expected-a3-checkpoint-sha256", required=True)
    result.add_argument("--expected-rlt-checkpoint-sha256", required=True)
    result.add_argument("--expected-seed", required=True, type=int)
    result.add_argument("--evaluation-code-inventory", required=True, type=Path)
    result.add_argument("--expected-evaluation-code-inventory-sha256", required=True)
    result.add_argument("--output-json", required=True, type=Path)
    result.add_argument("--output-markdown", required=True, type=Path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    expected_run_id = _string(args.expected_run_id, "expected run id")
    expected_run_manifest_sha = _sha256(
        args.expected_run_manifest_sha256, "expected run manifest SHA-256"
    )
    expected_eval_plan_sha = _sha256(
        args.expected_eval_plan_sha256, "expected evaluation plan SHA-256"
    )
    expected_split_sha = _sha256(
        args.expected_split_manifest_sha256, "expected split manifest SHA-256"
    )
    expected_protocol_sha = _sha256(
        args.expected_protocol_sha256, "expected protocol SHA-256"
    )
    expected_model_sha = _sha256(args.expected_model_sha256, "expected model SHA-256")
    expected_weight_sha = _sha256(
        args.expected_lightnav_weight_sha256, "expected LightNav weight SHA-256"
    )
    expected_a3_sha = _sha256(
        args.expected_a3_checkpoint_sha256, "expected A3 checkpoint SHA-256"
    )
    expected_rlt_sha = _sha256(
        args.expected_rlt_checkpoint_sha256, "expected Navigation-RLT checkpoint SHA-256"
    )
    expected_seed = _integer(args.expected_seed, "expected seed")
    evaluation_code = _load_evaluation_inventory(
        args.evaluation_code_inventory,
        expected_sha256=args.expected_evaluation_code_inventory_sha256,
    )
    if args.output_json.resolve() == args.output_markdown.resolve():
        raise ComparisonError("JSON and Markdown outputs must be different paths")
    if os.path.lexists(args.output_json) or os.path.lexists(args.output_markdown):
        raise ComparisonError("comparison output already exists")

    paths = {
        ARM_NATIVE: args.native,
        ARM_A3: args.a3,
        ARM_RLT: args.navigation_rlt,
    }
    expected_checkpoints = {
        ARM_NATIVE: None,
        ARM_A3: expected_a3_sha,
        ARM_RLT: expected_rlt_sha,
    }
    raw = {arm: _load(path, arm) for arm, path in paths.items()}
    arms = {
        arm: _validate_arm(
            raw[arm],
            arm=arm,
            expected_run_id=expected_run_id,
            expected_run_manifest_sha256=expected_run_manifest_sha,
            expected_eval_plan_sha256=expected_eval_plan_sha,
            expected_split_manifest_sha256=expected_split_sha,
            expected_protocol_sha256=expected_protocol_sha,
            expected_model_sha256=expected_model_sha,
            expected_lightnav_weight_sha256=expected_weight_sha,
            expected_checkpoint_sha256=expected_checkpoints[arm],
            expected_seed=expected_seed,
            expected_evaluation_code_commit=str(evaluation_code["commit"]),
            expected_evaluation_inventory_sha256=str(
                evaluation_code["inventory_sha256"]
            ),
        )
        for arm in ARM_ORDER
    }
    for key in ("protocol", "model"):
        reference = arms[ARM_NATIVE][key]
        for arm in (ARM_A3, ARM_RLT):
            if arms[arm][key] != reference:
                raise ComparisonError(f"{arm} {key} contract differs from native LightNav")
    identities = {
        arm: [_episode_identity(row) for row in arms[arm]["episodes"]]
        for arm in ARM_ORDER
    }
    if identities[ARM_NATIVE] != identities[ARM_A3] or identities[ARM_NATIVE] != identities[ARM_RLT]:
        raise ComparisonError("the three arms do not contain the same ordered episode identities")

    summaries = {arm: arms[arm]["summary"] for arm in ARM_ORDER}
    deltas = {
        "a3_minus_native": _numeric_delta(summaries[ARM_A3], summaries[ARM_NATIVE]),
        "rlt_minus_a3": _numeric_delta(summaries[ARM_RLT], summaries[ARM_A3]),
        "rlt_minus_native": _numeric_delta(summaries[ARM_RLT], summaries[ARM_NATIVE]),
    }
    gate = _gate(summaries, arms)
    verified_run_ids = {str(arms[arm]["run_id"]) for arm in ARM_ORDER}
    verified_collection_shas = {
        str(arms[arm]["collection_manifest_sha256"]) for arm in ARM_ORDER
    }
    verified_evaluation_code = {
        (
            str(_mapping(arms[arm]["evaluation_code"], f"validated {arm} evaluation code")["commit"]),
            str(
                _mapping(
                    arms[arm]["evaluation_code"], f"validated {arm} evaluation code"
                )["inventory_sha256"]
            ),
        )
        for arm in ARM_ORDER
    }
    if (
        len(verified_run_ids) != 1
        or len(verified_collection_shas) != 1
        or len(verified_evaluation_code) != 1
    ):
        raise ComparisonError("validated arm provenance identities disagree")
    verified_run_id = verified_run_ids.pop()
    verified_collection_sha = verified_collection_shas.pop()
    verified_evaluation_commit, verified_evaluation_inventory_sha = (
        verified_evaluation_code.pop()
    )
    rlt_checkpoint_sha = _sha256(
        _mapping(arms[ARM_RLT]["checkpoint"], "validated Navigation-RLT checkpoint").get(
            "sha256"
        ),
        "validated Navigation-RLT checkpoint SHA-256",
    )
    output = {
        "schema_name": OUTPUT_SCHEMA,
        "schema_version": OUTPUT_VERSION,
        "status": "completed",
        "run_id": verified_run_id,
        "collection_manifest_sha256": verified_collection_sha,
        "evaluation_code_commit": verified_evaluation_commit,
        "evaluation_code_inventory_sha256": verified_evaluation_inventory_sha,
        "navigation_rlt_checkpoint_sha256": rlt_checkpoint_sha,
        "inputs": {
            arm: {
                "path": str(Path(paths[arm]).resolve()),
                "sha256": sha256_file(Path(paths[arm]).resolve()),
            }
            for arm in ARM_ORDER
        },
        "matched_contract": {
            "run_manifest_sha256": verified_collection_sha,
            "protocol": arms[ARM_NATIVE]["protocol"],
            "protocol_sha256": expected_protocol_sha,
            "model": arms[ARM_NATIVE]["model"],
            "model_sha256": expected_model_sha,
            "ordered_episode_identity_sha256": _object_sha256(identities[ARM_NATIVE]),
            "episode_identity_fields": [
                "episode_order",
                "scene",
                "episode_id",
                "instruction_text_sha256",
            ],
        },
        "checkpoints": {arm: arms[arm]["checkpoint"] for arm in ARM_ORDER},
        "contexts": {arm: arms[arm]["context"] for arm in ARM_ORDER},
        "summaries": summaries,
        "deltas": deltas,
        "paired_episodes": _paired_rows(arms),
        "gate": gate,
        "result_to_claim_boundary": _claim_boundary(bool(gate["passed"])),
    }
    json_text = json.dumps(
        output, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ) + "\n"
    markdown_text = _markdown(output)
    _atomic_write(args.output_json, json_text)
    _atomic_write(args.output_markdown, markdown_text)
    print(
        json.dumps(
            {
                "gate_passed": gate["passed"],
                "output_json": str(args.output_json.resolve()),
                "output_json_sha256": sha256_file(args.output_json),
                "output_markdown": str(args.output_markdown.resolve()),
                "output_markdown_sha256": sha256_file(args.output_markdown),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ComparisonError as error:
        print(f"comparison invalid: {error}")
        raise SystemExit(2)
