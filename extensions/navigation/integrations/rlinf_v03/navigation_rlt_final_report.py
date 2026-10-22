#!/usr/bin/env python3
"""Create a deterministic final report from a verified Navigation-RLT OBS archive.

The report is deliberately post-completion and read-only with respect to the OBS
tree.  It summarizes the fixed training budgets, eight-GPU evidence, and the
strict matched train-development comparison.  It does not turn the single
fixed-budget result into a conventional sample-efficiency curve.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
from typing import Mapping, Sequence


ARCHIVE_SCHEMA = "navigation_rlt_obs_archive"
ARCHIVE_COMPLETION_SCHEMA = "navigation_rlt_obs_completion"
TRAINING_SCHEMA = "navigation_rlt_rlinf_fsdp_training"
TELEMETRY_SCHEMA = "lightnav_a3_gpu_telemetry"
MATCHED_SCHEMA = "navigation_rlt_matched_comparison"
REPLAY_SCHEMA = "navigation_rlt_replay_manifest"
REPORT_SCHEMA = "navigation_rlt_final_experiment_report"
REPORT_VERSION = 1
REPORT_COMPLETION_SCHEMA = "navigation_rlt_final_report_completion"
A3_BUDGET_RECEIPT_SCHEMA = "navigation_rlt_a3_training_budget_receipt"
A3_BUDGET_RECEIPT_FILENAME = "A3_TRAINING_BUDGET_RECEIPT.json"
A3_RUN_ID = "lightnav-a3-20260930-v1"
A3_EXPECTED_PROVENANCE = {
    "run_manifest_sha256": "85169e0495e7e26044223b2b66d9ac14c654b1da43da389fbfce489bb310df51",
    "split_manifest_sha256": "beaf5ebc44dca6729dd5c7431eba0675dd7c8ddb3c4ebdae3ad410de9fa73a38",
    "final_manifest_sha256": "1a485f6ef877d987073bfdebee5766f21747919eac897400c5f0384a237570bb",
    "replay_sha256": "e1e6b4461c690f7e7c58cda7fa5db65eeb0ff9064e186db006eb76d5e5cd6be4",
    "checkpoint_sha256": "f9a65a6b54efa53422e72a6918a8d6076e5893f821d68b2864b0e420e81c16e1",
}
A3_ACTOR_ELIGIBILITY = {
    "actor_eligible_required": True,
    "policy_version": 0,
    "native_stop_excluded": True,
    "external_forced_stop_excluded": True,
}
ARMS = ("native_lightnav", "a3_ppo", "navigation_rlt")
MODES = ("KEEP", "ADJUST", "STOP", "RECOVER")
PHASES = ("stage1", "stage2")
STAGE2_ELIGIBILITY_SCHEMA = (
    "credit_mask_true_and_requested_gate_equals_executed_gate_v1"
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
A800_MODEL_PATTERN = re.compile(r"^NVIDIA A800\b", re.IGNORECASE)
REQUIRED_RESULTS = {
    "stage1_training": "stage1/training/training_result.json",
    "stage1_telemetry": "stage1/gpu_telemetry_summary.json",
    "stage1_replay": "stage1/provenance/replay_manifest.json",
    "stage2_training": "stage2/training/training_result.json",
    "stage2_telemetry": "stage2/gpu_telemetry_summary.json",
    "stage2_replay": "stage2/provenance/replay_manifest.json",
    "matched": "matched/matched_result.json",
}


class FinalReportError(RuntimeError):
    """Raised when an OBS result tree cannot support the final report."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    raise FinalReportError(f"non-finite JSON constant is forbidden: {value}")


def _load_json(path: Path, name: str) -> dict[str, object]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=_reject_constant
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FinalReportError(f"{name} is not strict JSON") from error
    if not isinstance(value, dict):
        raise FinalReportError(f"{name} must contain an object")
    return value


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise FinalReportError(f"{name} is not a lowercase SHA-256")
    return value


def _finite(value: object, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FinalReportError(f"{name} is not numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise FinalReportError(f"{name} is outside its valid range")
    return result


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise FinalReportError(f"{name} is not an integer >= {minimum}")
    return value


def _normalized_gpu_model_name(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise FinalReportError(f"{name} is absent")
    normalized = " ".join(value.split())
    if not normalized:
        raise FinalReportError(f"{name} is absent")
    return normalized


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise FinalReportError(f"{name} is not an object")
    return value


def _regular_root(path: Path) -> Path:
    raw = Path(path).expanduser().absolute()
    if raw.is_symlink():
        raise FinalReportError("OBS result root must not be a symlink")
    try:
        root = raw.resolve(strict=True)
    except OSError as error:
        raise FinalReportError("OBS result root is missing") from error
    if not root.is_dir():
        raise FinalReportError("OBS result root is not a directory")
    return root


def _regular_file(root: Path, relative: str, name: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or any(part in ("", ".", "..") for part in pure.parts):
        raise FinalReportError(f"{name} relative path is unsafe")
    raw = root.joinpath(*pure.parts)
    if raw.is_symlink():
        raise FinalReportError(f"{name} must not be a symlink")
    try:
        path = raw.resolve(strict=True)
        path.relative_to(root)
    except (OSError, ValueError) as error:
        raise FinalReportError(f"{name} is missing or escapes the result root") from error
    if not path.is_file():
        raise FinalReportError(f"{name} is not a regular file")
    return path


def _path_field(value: object, expected: Path, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise FinalReportError(f"{name} path is absent")
    try:
        actual = Path(value).resolve(strict=True)
    except OSError as error:
        raise FinalReportError(f"{name} path is missing") from error
    if actual != expected:
        raise FinalReportError(f"{name} path differs")


def _exact_keys(value: Mapping[str, object], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise FinalReportError(f"{name} field inventory differs")


def _validate_a3_training_budget_receipt(path: Path) -> dict[str, object]:
    raw = Path(path).expanduser().absolute()
    if raw.name != A3_BUDGET_RECEIPT_FILENAME:
        raise FinalReportError(
            "A3 training-budget receipt must be named "
            f"{A3_BUDGET_RECEIPT_FILENAME}"
        )
    if raw.is_symlink():
        raise FinalReportError("A3 training-budget receipt must not be a symlink")
    try:
        resolved = raw.resolve(strict=True)
    except OSError as error:
        raise FinalReportError("A3 training-budget receipt is missing") from error
    if not resolved.is_file():
        raise FinalReportError("A3 training-budget receipt is not a regular file")
    value = _load_json(resolved, "A3 training-budget receipt")
    _exact_keys(
        value,
        {"schema_name", "schema_version", "status", "run_id", "scope", "provenance", "budget"},
        "A3 training-budget receipt",
    )
    if (
        value.get("schema_name") != A3_BUDGET_RECEIPT_SCHEMA
        or type(value.get("schema_version")) is not int
        or value.get("schema_version") != 1
        or value.get("status") != "verified"
        or value.get("run_id") != A3_RUN_ID
    ):
        raise FinalReportError("A3 training-budget receipt schema/status/run differs")
    scope = _mapping(value.get("scope"), "A3 training-budget scope")
    _exact_keys(
        scope,
        {"upstream_split_read", "validation_split_reads", "train_only"},
        "A3 training-budget scope",
    )
    if (
        scope.get("upstream_split_read") != "train"
        or type(scope.get("validation_split_reads")) is not int
        or scope.get("validation_split_reads") != 0
        or scope.get("train_only") is not True
    ):
        raise FinalReportError("A3 training-budget receipt violates the train-only contract")
    provenance = _mapping(value.get("provenance"), "A3 training-budget provenance")
    _exact_keys(provenance, set(A3_EXPECTED_PROVENANCE), "A3 training-budget provenance")
    normalized_provenance = {
        name: _sha256(provenance.get(name), f"A3 {name}")
        for name in A3_EXPECTED_PROVENANCE
    }
    if normalized_provenance != A3_EXPECTED_PROVENANCE:
        raise FinalReportError("A3 training-budget provenance differs")
    budget = _mapping(value.get("budget"), "A3 training budget")
    _exact_keys(
        budget,
        {
            "rollout_episode_count",
            "unique_actor_decisions",
            "actor_eligibility",
            "optimizer_epochs",
            "optimizer_presentations",
            "padding_excluded",
        },
        "A3 training budget",
    )
    actor_eligibility = _mapping(
        budget.get("actor_eligibility"), "A3 actor eligibility"
    )
    _exact_keys(
        actor_eligibility, set(A3_ACTOR_ELIGIBILITY), "A3 actor eligibility"
    )
    if (
        actor_eligibility.get("actor_eligible_required") is not True
        or type(actor_eligibility.get("policy_version")) is not int
        or actor_eligibility.get("policy_version") != 0
        or actor_eligibility.get("native_stop_excluded") is not True
        or actor_eligibility.get("external_forced_stop_excluded") is not True
    ):
        raise FinalReportError("A3 actor eligibility semantics differ")
    rollout_episodes = _integer(
        budget.get("rollout_episode_count"), "A3 rollout episode count", minimum=1
    )
    unique_decisions = _integer(
        budget.get("unique_actor_decisions"), "A3 unique actor decisions", minimum=1
    )
    optimizer_epochs = _integer(
        budget.get("optimizer_epochs"), "A3 optimizer epochs", minimum=1
    )
    presentations = _integer(
        budget.get("optimizer_presentations"), "A3 optimizer presentations", minimum=1
    )
    if (
        rollout_episodes != 64
        or unique_decisions != 4642
        or optimizer_epochs != 4
        or presentations != 18568
        or presentations != unique_decisions * optimizer_epochs
        or budget.get("padding_excluded") is not True
    ):
        raise FinalReportError("A3 training budget differs")
    return {
        "receipt": {
            "path": str(resolved),
            "sha256": sha256_file(resolved),
        },
        "run_id": A3_RUN_ID,
        "scope": dict(scope),
        "provenance": normalized_provenance,
        "rollout_episode_count": rollout_episodes,
        "unique_actor_decisions": unique_decisions,
        "actor_eligibility": dict(actor_eligibility),
        "optimizer_epochs": optimizer_epochs,
        "optimizer_presentations": presentations,
        "padding_excluded": True,
    }


def _validate_archive(root: Path) -> tuple[dict[str, object], dict[str, object], dict[str, Path], dict[str, object]]:
    manifest_path = _regular_file(root, "archive_manifest.json", "archive manifest")
    completion_path = _regular_file(root, "COMPLETED.json", "archive completion")
    manifest = _load_json(manifest_path, "archive manifest")
    completion = _load_json(completion_path, "archive completion")
    if (
        manifest.get("schema_name") != ARCHIVE_SCHEMA
        or manifest.get("schema_version") != 1
        or manifest.get("status") != "readback_verified"
    ):
        raise FinalReportError("archive manifest schema/status differs")
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise FinalReportError("archive run_id is absent")
    readback_sha = _sha256(
        manifest.get("independent_readback_sha256"),
        "archive independent readback fingerprint",
    )
    if (
        completion.get("schema_name") != ARCHIVE_COMPLETION_SCHEMA
        or completion.get("schema_version") != 1
        or completion.get("status") != "completed"
        or completion.get("run_id") != run_id
        or completion.get("independent_readback_sha256") != readback_sha
    ):
        raise FinalReportError("archive completion schema/status/readback binding differs")
    manifest_sha = sha256_file(manifest_path)
    if completion.get("archive_manifest_sha256") != manifest_sha:
        raise FinalReportError("archive completion manifest SHA-256 differs")
    _path_field(completion.get("archive_manifest"), manifest_path, "archive manifest")
    _path_field(completion.get("result_root"), root, "archive result root")
    destinations = _mapping(manifest.get("destinations"), "archive destinations")
    _path_field(destinations.get("results"), root, "manifest result root")

    raw_records = manifest.get("files")
    if not isinstance(raw_records, list) or not raw_records:
        raise FinalReportError("archive file inventory is absent")
    records: dict[tuple[str, str], Mapping[str, object]] = {}
    for index, raw in enumerate(raw_records):
        record = _mapping(raw, f"archive file record {index}")
        group = record.get("group")
        relative = record.get("relative_path")
        if not isinstance(group, str) or not isinstance(relative, str):
            raise FinalReportError(f"archive file record {index} identity is invalid")
        identity = (group, relative)
        if identity in records:
            raise FinalReportError("archive file inventory contains a duplicate")
        records[identity] = record

    paths: dict[str, Path] = {}
    evidence: dict[str, object] = {}
    for key, relative in REQUIRED_RESULTS.items():
        record = records.get(("results", relative))
        if record is None:
            raise FinalReportError(f"archive lacks required result {relative}")
        path = _regular_file(root, relative, key)
        expected_sha = _sha256(record.get("sha256"), f"{key} SHA-256")
        if record.get("readback_sha256") != expected_sha:
            raise FinalReportError(f"{key} independent readback SHA-256 differs")
        if record.get("size_bytes") != path.stat().st_size or sha256_file(path) != expected_sha:
            raise FinalReportError(f"{key} current readback differs")
        _path_field(record.get("destination"), path, f"{key} destination")
        paths[key] = path
        evidence[key] = {
            "path": str(path),
            "sha256": expected_sha,
            "size_bytes": path.stat().st_size,
            "readback_sha256": expected_sha,
        }

    matched_receipt = _mapping(manifest.get("matched_result"), "archive matched result")
    if matched_receipt.get("sha256") != evidence["matched"]["sha256"]:
        raise FinalReportError("archive matched-result SHA-256 binding differs")
    evidence["archive_manifest"] = {
        "path": str(manifest_path),
        "sha256": manifest_sha,
    }
    evidence["archive_completion"] = {
        "path": str(completion_path),
        "sha256": sha256_file(completion_path),
    }
    return manifest, completion, paths, evidence


def _validate_replay(
    value: Mapping[str, object], *, phase: str, run_id: str
) -> dict[str, object]:
    expected_representation = "prefix_hidden" if phase == "stage1" else "frozen_z_rl"
    if (
        value.get("schema_name") != REPLAY_SCHEMA
        or value.get("schema_version") != 3
        or value.get("run_id") != run_id
        or value.get("representation_kind") != expected_representation
        or value.get("split_label") != "train_update"
        or value.get("validation_split_reads") != 0
    ):
        raise FinalReportError(f"{phase} replay violates the train-only contract")
    episodes = _integer(value.get("completed_episode_count"), f"{phase} episodes", minimum=1)
    if value.get("expected_episode_count") != episodes:
        raise FinalReportError(f"{phase} replay episode budget is incomplete")
    transitions = _integer(value.get("transition_count"), f"{phase} transitions", minimum=1)
    return {
        "episode_count": episodes,
        "transition_count": transitions,
        "representation_kind": expected_representation,
        "source_fingerprint": _sha256(
            value.get("source_fingerprint"), f"{phase} source fingerprint"
        ),
    }


def _summary_stats(values: Sequence[float]) -> dict[str, float]:
    return {
        "minimum": min(values),
        "mean": math.fsum(values) / len(values),
        "maximum": max(values),
    }


def _validate_telemetry(
    value: Mapping[str, object], *, phase: str, expected_uuids: set[str]
) -> dict[str, object]:
    gpus = value.get("gpus")
    if (
        value.get("schema_name") != TELEMETRY_SCHEMA
        or value.get("schema_version") != 1
        or value.get("gpu_count") != 8
        or not isinstance(gpus, list)
        or len(gpus) != 8
    ):
        raise FinalReportError(f"{phase} telemetry schema/GPU count differs")
    sample_interval_s = _finite(
        value.get("sample_interval_s"), f"{phase} telemetry sample interval", minimum=1.0e-12
    )
    normalized = []
    for index, raw in enumerate(gpus):
        gpu = _mapping(raw, f"{phase} telemetry GPU {index}")
        gpu_index = _integer(gpu.get("index"), f"{phase} GPU index")
        uuid = gpu.get("uuid")
        if not isinstance(uuid, str) or not uuid:
            raise FinalReportError(f"{phase} telemetry GPU UUID is absent")
        sample_count = _integer(
            gpu.get("sample_count"), f"{phase} GPU sample count", minimum=2
        )
        memory_total = _finite(
            gpu.get("memory_total_mib"), f"{phase} GPU memory total", minimum=0.0
        )
        memory_mean = _finite(
            gpu.get("memory_used_mean_mib"), f"{phase} GPU memory mean", minimum=0.0
        )
        memory_max = _finite(
            gpu.get("memory_used_max_mib"), f"{phase} GPU memory max", minimum=0.0
        )
        util_mean = _finite(
            gpu.get("gpu_util_mean_percent"), f"{phase} GPU utilization mean", minimum=0.0
        )
        util_max = _finite(
            gpu.get("gpu_util_max_percent"), f"{phase} GPU utilization max", minimum=0.0
        )
        power_mean = _finite(
            gpu.get("power_mean_w"), f"{phase} GPU power mean", minimum=0.0
        )
        power_max = _finite(
            gpu.get("power_max_w"), f"{phase} GPU power max", minimum=0.0
        )
        if (
            util_mean > util_max
            or util_max > 100.0
            or memory_mean > memory_max
            or memory_max > memory_total
            or power_mean > power_max
        ):
            raise FinalReportError(f"{phase} telemetry mean/max bounds differ")
        normalized.append(
            {
                "index": gpu_index,
                "uuid": uuid,
                "sample_count": sample_count,
                "gpu_util_mean_percent": util_mean,
                "gpu_util_max_percent": util_max,
                "memory_used_mean_mib": memory_mean,
                "memory_used_max_mib": memory_max,
                "memory_total_mib": memory_total,
                "power_mean_w": power_mean,
                "power_max_w": power_max,
            }
        )
    if {row["index"] for row in normalized} != set(range(8)):
        raise FinalReportError(f"{phase} telemetry GPU indices differ")
    if {row["uuid"] for row in normalized} != expected_uuids:
        raise FinalReportError(f"{phase} telemetry/training GPU UUIDs differ")
    normalized.sort(key=lambda row: int(row["index"]))
    util_means = [float(row["gpu_util_mean_percent"]) for row in normalized]
    sample_counts = [int(row["sample_count"]) for row in normalized]
    if len(set(sample_counts)) != 1:
        raise FinalReportError(f"{phase} telemetry GPU sample counts differ")
    sample_count = sample_counts[0]
    return {
        "gpu_count": 8,
        "sample_interval_s": sample_interval_s,
        "sample_count_per_gpu": sample_count,
        "sampled_coverage_lower_bound_s": (sample_count - 1) * sample_interval_s,
        "per_gpu": normalized,
        "gpu_util_mean_percent_across_gpus": math.fsum(util_means) / 8,
        "gpu_util_per_gpu_mean_range_percent": [min(util_means), max(util_means)],
        "gpu_util_observed_max_percent": max(
            float(row["gpu_util_max_percent"]) for row in normalized
        ),
        "memory_used_mean_mib_across_gpus": math.fsum(
            float(row["memory_used_mean_mib"]) for row in normalized
        )
        / 8,
        "memory_used_observed_max_mib": max(
            float(row["memory_used_max_mib"]) for row in normalized
        ),
        "power_mean_w_across_gpus": math.fsum(
            float(row["power_mean_w"]) for row in normalized
        )
        / 8,
        "power_observed_max_w": max(float(row["power_max_w"]) for row in normalized),
    }


def _validate_training(
    value: Mapping[str, object],
    telemetry: Mapping[str, object],
    replay: Mapping[str, object],
    *,
    phase: str,
    run_id: str,
    replay_sha256: str,
) -> dict[str, object]:
    if (
        value.get("schema_name") != TRAINING_SCHEMA
        or value.get("schema_version") != 1
        or value.get("status") != "completed"
        or value.get("phase") != phase
        or value.get("run_id") != run_id
        or value.get("world_size") != 8
        or value.get("unique_gpu_uuid_count") != 8
    ):
        raise FinalReportError(f"{phase} training result schema/status differs")
    replay_receipt = _mapping(value.get("replay_manifest"), f"{phase} replay receipt")
    if (
        replay_receipt.get("sha256") != replay_sha256
        or replay_receipt.get("episode_count") != replay["episode_count"]
        or replay_receipt.get("transition_count") != replay["transition_count"]
        or replay_receipt.get("representation_kind") != replay["representation_kind"]
        or replay_receipt.get("source_fingerprint") != replay["source_fingerprint"]
    ):
        raise FinalReportError(f"{phase} training/replay binding differs")
    training = _mapping(value.get("training"), f"{phase} training configuration")
    epochs = _integer(training.get("epochs"), f"{phase} epochs", minimum=1)
    batch_size = _integer(
        training.get("batch_size_per_rank"), f"{phase} batch size", minimum=1
    )
    optimizer_steps = _integer(
        value.get("optimizer_steps_per_rank"), f"{phase} optimizer steps", minimum=1
    )
    ranks = value.get("rank_receipts")
    if not isinstance(ranks, list) or len(ranks) != 8:
        raise FinalReportError(f"{phase} rank receipt count differs")
    normalized_ranks = []
    for raw in ranks:
        rank = _mapping(raw, f"{phase} rank receipt")
        rank_index = _integer(rank.get("rank"), f"{phase} rank")
        if (
            rank.get("phase") != phase
            or rank.get("world_size") != 8
            or rank.get("gradient_steps") != optimizer_steps
        ):
            raise FinalReportError(f"{phase} rank training binding differs")
        data = _mapping(rank.get("data"), f"{phase} rank data")
        start = _mapping(rank.get("gpu_start"), f"{phase} rank GPU start")
        end = _mapping(rank.get("gpu_end"), f"{phase} rank GPU end")
        uuid = start.get("gpu_uuid")
        if not isinstance(uuid, str) or uuid != end.get("gpu_uuid"):
            raise FinalReportError(f"{phase} rank GPU identity differs")
        start_model = _normalized_gpu_model_name(
            start.get("gpu_name"), f"{phase} rank GPU start model"
        )
        end_model = _normalized_gpu_model_name(
            end.get("gpu_name"), f"{phase} rank GPU end model"
        )
        if start_model != end_model:
            raise FinalReportError(f"{phase} rank GPU model differs between start and end")
        if A800_MODEL_PATTERN.match(start_model) is None:
            raise FinalReportError(f"{phase} rank GPU model is not an NVIDIA A800")
        normalized_rank = {
            "rank": rank_index,
            "gpu_uuid": uuid,
            "gpu_model_name_normalized": start_model,
            "global_real_items": _integer(
                data.get("global_real_items"),
                f"{phase} global real items",
                minimum=1,
            ),
            "local_real_items": _integer(
                data.get("local_real_items"),
                f"{phase} local real items",
                minimum=1,
            ),
            "gradient_steps": optimizer_steps,
            "training_wall_time_s": _finite(
                rank.get("training_wall_time_s"),
                f"{phase} rank wall time",
                minimum=1.0e-12,
            ),
            "optimizer_steps_per_second": _finite(
                rank.get("optimizer_steps_per_second"),
                f"{phase} optimizer throughput",
                minimum=1.0e-12,
            ),
            "real_item_epochs_per_second": _finite(
                rank.get("real_item_epochs_per_second"),
                f"{phase} item throughput",
                minimum=1.0e-12,
            ),
            "cuda_peak_allocated_bytes": _integer(
                rank.get("cuda_peak_allocated_bytes"),
                f"{phase} peak CUDA allocated bytes",
            ),
            "cuda_peak_reserved_bytes": _integer(
                rank.get("cuda_peak_reserved_bytes"),
                f"{phase} peak CUDA reserved bytes",
            ),
        }
        if phase == "stage2":
            gate_names = data.get("gate_names")
            gate_counts = data.get("gate_counts")
            gate_support_mask = data.get("gate_support_mask")
            if not isinstance(gate_names, list):
                raise FinalReportError("stage2 rank gate names are invalid")
            if not isinstance(gate_counts, list):
                raise FinalReportError("stage2 rank gate counts are invalid")
            if not isinstance(gate_support_mask, list) or not all(
                isinstance(item, bool) for item in gate_support_mask
            ):
                raise FinalReportError("stage2 rank gate support mask is invalid")
            normalized_rank.update(
                {
                    "global_padding_items": _integer(
                        data.get("global_padding_items"),
                        "stage2 global padding items",
                    ),
                    "global_credit_ineligible_items_excluded": _integer(
                        data.get("global_credit_ineligible_items_excluded"),
                        "stage2 credit-ineligible items excluded",
                    ),
                    "global_gate_mismatch_items_excluded": _integer(
                        data.get("global_gate_mismatch_items_excluded"),
                        "stage2 gate-mismatch items excluded",
                    ),
                    "eligibility_schema": data.get("eligibility_schema"),
                    "gate_names": list(gate_names),
                    "gate_counts": [
                        _integer(item, "stage2 rank gate count")
                        for item in gate_counts
                    ],
                    "gate_support_mask": list(gate_support_mask),
                    "minimum_behavior_gate_count": _integer(
                        data.get("minimum_behavior_gate_count"),
                        "stage2 minimum behavior gate count",
                        minimum=1,
                    ),
                    "local_sampler_items": _integer(
                        data.get("local_sampler_items"),
                        "stage2 local sampler items",
                        minimum=1,
                    ),
                    "local_padding_items": _integer(
                        data.get("local_padding_items"),
                        "stage2 local padding items",
                    ),
                    "sampler_indices_sha256": _sha256(
                        data.get("sampler_indices_sha256"),
                        "stage2 sampler indices SHA-256",
                    ),
                    "sampler": data.get("sampler"),
                    "sampler_real_row_duplication": _integer(
                        data.get("sampler_real_row_duplication"),
                        "stage2 sampler real-row duplication",
                    ),
                }
            )
        normalized_ranks.append(normalized_rank)
    if {row["rank"] for row in normalized_ranks} != set(range(8)):
        raise FinalReportError(f"{phase} rank identities differ")
    uuids = {str(row["gpu_uuid"]) for row in normalized_ranks}
    if len(uuids) != 8:
        raise FinalReportError(f"{phase} rank GPU UUIDs are not unique")
    gpu_models = {
        str(row["gpu_model_name_normalized"]) for row in normalized_ranks
    }
    if len(gpu_models) != 1:
        raise FinalReportError(f"{phase} rank GPU models differ")
    gpu_model = gpu_models.pop()
    global_items = {int(row["global_real_items"]) for row in normalized_ranks}
    if len(global_items) != 1:
        raise FinalReportError(f"{phase} global real-item counts differ by rank")
    global_real_items = global_items.pop()
    if sum(int(row["local_real_items"]) for row in normalized_ranks) != global_real_items:
        raise FinalReportError(f"{phase} local real-item partitions do not cover the global set")
    stage2_budget_evidence: dict[str, object] = {}
    if phase == "stage2":
        support = _mapping(value.get("stage2_support"), "stage2 support evidence")
        support_keys = {
            "eligibility_schema",
            "gate_names",
            "gate_counts",
            "gate_support_mask",
            "minimum_behavior_gate_count",
        }
        if set(support) != support_keys:
            raise FinalReportError("stage2 support evidence contract differs")
        if support.get("eligibility_schema") != STAGE2_ELIGIBILITY_SCHEMA:
            raise FinalReportError("stage2 eligibility schema differs")
        if support.get("gate_names") != list(MODES):
            raise FinalReportError("stage2 gate names differ")
        gate_counts = support.get("gate_counts")
        gate_support_mask = support.get("gate_support_mask")
        if (
            not isinstance(gate_counts, list)
            or len(gate_counts) != len(MODES)
            or any(type(item) is not int or item < 0 for item in gate_counts)
        ):
            raise FinalReportError("stage2 gate counts are invalid")
        if (
            not isinstance(gate_support_mask, list)
            or len(gate_support_mask) != len(MODES)
            or any(type(item) is not bool for item in gate_support_mask)
        ):
            raise FinalReportError("stage2 gate support mask is invalid")
        minimum_gate_count = _integer(
            support.get("minimum_behavior_gate_count"),
            "stage2 minimum behavior gate count",
            minimum=1,
        )
        expected_support_mask = [
            count >= minimum_gate_count for count in gate_counts
        ]
        if gate_support_mask != expected_support_mask or not any(gate_support_mask):
            raise FinalReportError("stage2 gate support mask differs from counts")
        if sum(gate_counts) != global_real_items:
            raise FinalReportError("stage2 gate counts do not cover the eligible real rows")
        normalized_support = {
            "eligibility_schema": STAGE2_ELIGIBILITY_SCHEMA,
            "gate_names": list(MODES),
            "gate_counts": list(gate_counts),
            "gate_support_mask": list(gate_support_mask),
            "minimum_behavior_gate_count": minimum_gate_count,
        }
        for row in normalized_ranks:
            rank_support = {
                name: row[name]
                for name in support_keys
            }
            if rank_support != normalized_support:
                raise FinalReportError("stage2 rank support evidence differs from summary")
            if row["sampler"] != "torch.utils.data.DistributedSampler":
                raise FinalReportError("stage2 sampler implementation differs")
            if int(row["sampler_real_row_duplication"]) != 0:
                raise FinalReportError("stage2 sampler duplicates real rows")
            if (
                int(row["local_real_items"])
                + int(row["local_padding_items"])
                != int(row["local_sampler_items"])
            ):
                raise FinalReportError("stage2 local sampler coverage differs")
        sampler_hashes = {
            str(row["sampler_indices_sha256"]) for row in normalized_ranks
        }
        if len(sampler_hashes) != 8:
            raise FinalReportError("stage2 rank sampler partitions are not unique")
        global_padding_values = {
            int(row["global_padding_items"]) for row in normalized_ranks
        }
        credit_excluded_values = {
            int(row["global_credit_ineligible_items_excluded"])
            for row in normalized_ranks
        }
        gate_mismatch_values = {
            int(row["global_gate_mismatch_items_excluded"])
            for row in normalized_ranks
        }
        if (
            len(global_padding_values) != 1
            or len(credit_excluded_values) != 1
            or len(gate_mismatch_values) != 1
        ):
            raise FinalReportError("stage2 global eligibility evidence differs by rank")
        global_padding_items = global_padding_values.pop()
        credit_excluded = credit_excluded_values.pop()
        gate_mismatch_excluded = gate_mismatch_values.pop()
        if global_padding_items != (-global_real_items) % 8:
            raise FinalReportError("stage2 global padding count differs")
        expected_local_sampler_items = (global_real_items + global_padding_items) // 8
        if any(
            int(row["local_sampler_items"]) != expected_local_sampler_items
            for row in normalized_ranks
        ):
            raise FinalReportError("stage2 local sampler sizes differ")
        if (
            sum(int(row["local_padding_items"]) for row in normalized_ranks)
            != global_padding_items
        ):
            raise FinalReportError("stage2 local padding partitions do not cover padding")
        if (
            global_real_items + credit_excluded + gate_mismatch_excluded
            != int(replay["transition_count"])
        ):
            raise FinalReportError("stage2 eligibility exclusions do not partition replay rows")
        stage2_budget_evidence = {
            **normalized_support,
            "credit_ineligible_items_excluded": credit_excluded,
            "gate_mismatch_items_excluded": gate_mismatch_excluded,
            "global_padding_items": global_padding_items,
            "sampler_real_row_duplication": 0,
            "unique_real_rows_verified": True,
            "padding_excluded": True,
        }
    wall_times = [float(row["training_wall_time_s"]) for row in normalized_ranks]
    step_rates = [float(row["optimizer_steps_per_second"]) for row in normalized_ranks]
    item_rates = [float(row["real_item_epochs_per_second"]) for row in normalized_ranks]
    total_item_epochs = global_real_items * epochs
    telemetry_summary = {
        **_validate_telemetry(telemetry, phase=phase, expected_uuids=uuids),
        "gpu_model_name_normalized": gpu_model,
        "a800_identity_verified": True,
    }
    action_probabilities = value.get("action_probability_epochs")
    if not isinstance(action_probabilities, list):
        raise FinalReportError(f"{phase} action-probability history is invalid")
    if phase == "stage2" and len(action_probabilities) != epochs:
        raise FinalReportError("stage2 action-probability epoch count differs")
    if phase == "stage1" and action_probabilities:
        raise FinalReportError("stage1 unexpectedly contains actor probabilities")
    normalized_ranks.sort(key=lambda row: int(row["rank"]))
    return {
        "budget": {
            "episode_count": replay["episode_count"],
            "transition_count": replay["transition_count"],
            "global_real_items": global_real_items,
            "epochs": epochs,
            "batch_size_per_rank": batch_size,
            "world_size": 8,
            "optimizer_steps_per_rank": optimizer_steps,
            "total_real_item_epochs": total_item_epochs,
            **stage2_budget_evidence,
        },
        "throughput": {
            "global_real_item_epochs_per_second": total_item_epochs / max(wall_times),
            "rank_optimizer_steps_per_second": _summary_stats(step_rates),
            "rank_real_item_epochs_per_second": _summary_stats(item_rates),
            "rank_training_wall_time_s": _summary_stats(wall_times),
            "rank_wall_time_max_over_min_proxy": max(wall_times) / min(wall_times),
            "direct_nccl_wait_measurement_available": False,
        },
        "gpu": telemetry_summary,
        "rank_receipts": normalized_ranks,
        "action_probability_epochs": action_probabilities,
    }


def _validate_modes(value: object, name: str, decisions: int) -> dict[str, object]:
    modes = _mapping(value, name)
    counts = _mapping(modes.get("counts"), f"{name} counts")
    ratios = _mapping(modes.get("ratios"), f"{name} ratios")
    if set(counts) != set(MODES) or set(ratios) != set(MODES):
        raise FinalReportError(f"{name} mode inventory differs")
    normalized_counts = {
        mode: _integer(counts.get(mode), f"{name} {mode} count") for mode in MODES
    }
    total = sum(normalized_counts.values())
    if total != decisions or modes.get("total") != decisions or total < 1:
        raise FinalReportError(f"{name} action total differs")
    normalized_ratios = {}
    for mode in MODES:
        ratio = _finite(ratios.get(mode), f"{name} {mode} ratio", minimum=0.0)
        expected = normalized_counts[mode] / total
        if ratio > 1.0 or not math.isclose(ratio, expected, rel_tol=1.0e-9, abs_tol=1.0e-12):
            raise FinalReportError(f"{name} {mode} ratio differs")
        normalized_ratios[mode] = ratio
    return {"total": total, "counts": normalized_counts, "ratios": normalized_ratios}


def _validate_matched(value: Mapping[str, object], *, run_id: str) -> dict[str, object]:
    if (
        value.get("schema_name") != MATCHED_SCHEMA
        or value.get("schema_version") != 1
        or value.get("status") != "completed"
        or value.get("run_id") != run_id
    ):
        raise FinalReportError("matched result schema/status differs")
    contract = _mapping(value.get("matched_contract"), "matched contract")
    protocol = _mapping(contract.get("protocol"), "matched protocol")
    if (
        protocol.get("split_label") != "train_development"
        or protocol.get("upstream_split_read") != "train"
        or protocol.get("validation_split_reads") != 0
        or protocol.get("training_replay_written") is not False
    ):
        raise FinalReportError("matched result violates the train-only contract")
    raw_checkpoints = _mapping(value.get("checkpoints"), "matched checkpoints")
    if set(raw_checkpoints) != set(ARMS):
        raise FinalReportError("matched checkpoint arm inventory differs")
    expected_checkpoint_kinds = {
        "native_lightnav": "original_lightnav",
        "a3_ppo": "a3_ppo",
        "navigation_rlt": "navigation_rlt",
    }
    checkpoints: dict[str, object] = {}
    for arm in ARMS:
        checkpoint = _mapping(raw_checkpoints.get(arm), f"{arm} checkpoint")
        if set(checkpoint) != {"kind", "sha256", "policy_version"}:
            raise FinalReportError(f"{arm} checkpoint contract differs")
        if checkpoint.get("kind") != expected_checkpoint_kinds[arm]:
            raise FinalReportError(f"{arm} checkpoint kind differs")
        if arm == "native_lightnav":
            if checkpoint.get("policy_version") is not None:
                raise FinalReportError("native LightNav checkpoint policy version differs")
            policy_version = None
        else:
            policy_version = _integer(
                checkpoint.get("policy_version"), f"{arm} checkpoint policy version"
            )
        checkpoints[arm] = {
            "kind": expected_checkpoint_kinds[arm],
            "sha256": _sha256(checkpoint.get("sha256"), f"{arm} checkpoint SHA-256"),
            "policy_version": policy_version,
        }
    if value.get("navigation_rlt_checkpoint_sha256") != _mapping(
        checkpoints["navigation_rlt"], "Navigation-RLT checkpoint"
    )["sha256"]:
        raise FinalReportError("matched Navigation-RLT checkpoint binding differs")
    raw_summaries = _mapping(value.get("summaries"), "matched summaries")
    if set(raw_summaries) != set(ARMS):
        raise FinalReportError("matched arm inventory differs")
    summaries: dict[str, object] = {}
    metric_names = (
        "success_rate",
        "mean_spl",
        "mean_final_navigation_error_m",
        "mean_path_length_m",
        "collision_count",
        "blocked_step_count",
    )
    stop_names = (
        "native_stop_count",
        "learned_stop_count",
        "budget_stop_count",
        "no_stop_count",
    )
    for arm in ARMS:
        summary = _mapping(raw_summaries.get(arm), f"{arm} summary")
        episodes = _integer(summary.get("episode_count"), f"{arm} episodes", minimum=1)
        if episodes != 32:
            raise FinalReportError(f"{arm} matched episode count differs")
        successes = _integer(summary.get("success_count"), f"{arm} successes")
        if successes > episodes:
            raise FinalReportError(f"{arm} successes exceed episodes")
        metrics = {
            name: _finite(summary.get(name), f"{arm} {name}", minimum=0.0)
            for name in metric_names
        }
        if not math.isclose(
            metrics["success_rate"], successes / episodes, rel_tol=1.0e-9, abs_tol=1.0e-12
        ):
            raise FinalReportError(f"{arm} success rate differs")
        stops = {name: _integer(summary.get(name), f"{arm} {name}") for name in stop_names}
        if sum(stops.values()) != episodes:
            raise FinalReportError(f"{arm} STOP-source counts differ")
        timeouts = _integer(summary.get("timeout_count"), f"{arm} timeouts")
        decisions = _integer(
            summary.get("actor_decision_count"), f"{arm} actor decisions", minimum=1
        )
        raw_latency = _mapping(summary.get("model_latency"), f"{arm} model latency")
        latency_names = ("count", "mean_ms", "p50_ms", "p95_ms", "max_ms")
        if set(raw_latency) != set(latency_names):
            raise FinalReportError(f"{arm} model latency inventory differs")
        latency_count = _integer(
            raw_latency.get("count"), f"{arm} model latency count", minimum=1
        )
        if latency_count != decisions:
            raise FinalReportError(f"{arm} model latency count differs")
        model_latency = {
            "count": latency_count,
            **{
                name: _finite(
                    raw_latency.get(name), f"{arm} model latency {name}", minimum=0.0
                )
                for name in latency_names[1:]
            },
        }
        if not (
            model_latency["p50_ms"]
            <= model_latency["p95_ms"]
            <= model_latency["max_ms"]
            and model_latency["mean_ms"] <= model_latency["max_ms"]
        ):
            raise FinalReportError(f"{arm} model latency ordering differs")
        summaries[arm] = {
            "episode_count": episodes,
            "success_count": successes,
            **metrics,
            **stops,
            "timeout_count": timeouts,
            "safety_override_count": _integer(
                summary.get("safety_override_count"), f"{arm} safety overrides"
            ),
            "model_latency": model_latency,
            "requested_modes": _validate_modes(
                summary.get("requested_modes"), f"{arm} requested modes", decisions
            ),
            "executed_modes": _validate_modes(
                summary.get("executed_modes"), f"{arm} executed modes", decisions
            ),
        }
    a3 = _mapping(summaries["a3_ppo"], "A3 summary")
    rlt = _mapping(summaries["navigation_rlt"], "Navigation-RLT summary")
    deltas = {
        "success_count": int(rlt["success_count"]) - int(a3["success_count"]),
        "success_rate": float(rlt["success_rate"]) - float(a3["success_rate"]),
        "mean_spl": float(rlt["mean_spl"]) - float(a3["mean_spl"]),
    }
    recorded = _mapping(
        _mapping(value.get("deltas"), "matched deltas").get("rlt_minus_a3"),
        "RLT minus A3 deltas",
    )
    for name, expected in deltas.items():
        observed = _finite(recorded.get(name), f"matched delta {name}")
        if not math.isclose(observed, float(expected), rel_tol=1.0e-9, abs_tol=1.0e-12):
            raise FinalReportError(f"matched delta {name} differs")
    return {
        "protocol": {
            "split_label": "train_development",
            "upstream_split_read": "train",
            "validation_split_reads": 0,
            "training_replay_written": False,
        },
        "checkpoints": checkpoints,
        "summaries": summaries,
        "rlt_minus_a3": deltas,
        "gate": value.get("gate"),
        "result_to_claim_boundary": value.get("result_to_claim_boundary"),
    }


def _matched_delta_vs_native(
    summaries: Mapping[str, object], arm: str
) -> dict[str, float | int]:
    native = _mapping(summaries["native_lightnav"], "Native summary")
    candidate = _mapping(summaries[arm], f"{arm} summary")
    return {
        "success_count": int(candidate["success_count"]) - int(native["success_count"]),
        "SR": float(candidate["success_rate"]) - float(native["success_rate"]),
        "SPL": float(candidate["mean_spl"]) - float(native["mean_spl"]),
    }


def _per_thousand(
    values: Mapping[str, object], denominator: int
) -> dict[str, float]:
    return {
        name: float(value) * 1000.0 / denominator
        for name, value in values.items()
    }


def build_report(
    obs_result_root: Path,
    a3_training_budget_receipt: Path | None = None,
) -> dict[str, object]:
    root = _regular_root(obs_result_root)
    manifest, _, paths, evidence = _validate_archive(root)
    run_id = str(manifest["run_id"])
    training: dict[str, object] = {}
    for phase in PHASES:
        replay = _validate_replay(
            _load_json(paths[f"{phase}_replay"], f"{phase} replay manifest"),
            phase=phase,
            run_id=run_id,
        )
        archived_replay = _mapping(
            _mapping(manifest.get("replay"), "archive replay summaries").get(phase),
            f"archive {phase} replay summary",
        )
        if (
            archived_replay.get("episode_count") != replay["episode_count"]
            or archived_replay.get("transition_count") != replay["transition_count"]
            or archived_replay.get("representation_kind") != replay["representation_kind"]
            or archived_replay.get("source_fingerprint") != replay["source_fingerprint"]
        ):
            raise FinalReportError(f"archive {phase} replay summary differs")
        training[phase] = _validate_training(
            _load_json(paths[f"{phase}_training"], f"{phase} training result"),
            _load_json(paths[f"{phase}_telemetry"], f"{phase} telemetry summary"),
            replay,
            phase=phase,
            run_id=run_id,
            replay_sha256=str(evidence[f"{phase}_replay"]["sha256"]),
        )
    matched = _validate_matched(
        _load_json(paths["matched"], "matched result"), run_id=run_id
    )
    a3_budget = (
        None
        if a3_training_budget_receipt is None
        else _validate_a3_training_budget_receipt(a3_training_budget_receipt)
    )
    if a3_budget is not None:
        matched_a3_checkpoint = _mapping(
            _mapping(matched["checkpoints"], "matched checkpoints").get("a3_ppo"),
            "matched A3 checkpoint",
        )
        a3_provenance = _mapping(a3_budget["provenance"], "A3 budget provenance")
        if matched_a3_checkpoint.get("sha256") != a3_provenance.get(
            "checkpoint_sha256"
        ):
            raise FinalReportError(
                "A3 training-budget checkpoint does not match matched A3 arm"
            )
    report_inputs = dict(evidence)
    if a3_budget is not None:
        report_inputs["a3_training_budget_receipt"] = dict(
            _mapping(a3_budget["receipt"], "A3 receipt identity")
        )
    stage2_budget = _mapping(training["stage2"]["budget"], "stage2 budget")
    transitions = int(stage2_budget["transition_count"])
    rlt_unique_decisions = int(stage2_budget["global_real_items"])
    rlt_presentations = int(stage2_budget["total_real_item_epochs"])
    deltas = _mapping(matched["rlt_minus_a3"], "RLT minus A3 deltas")
    raw_proxy = {
        "success_count": deltas["success_count"],
        "SR": deltas["success_rate"],
        "SPL": deltas["mean_spl"],
    }
    summaries = _mapping(matched["summaries"], "matched summaries")
    a3_vs_native = _matched_delta_vs_native(summaries, "a3_ppo")
    rlt_vs_native = _matched_delta_vs_native(summaries, "navigation_rlt")
    one_point_vs_native: dict[str, object] = {
        "a3_ppo": {
            "raw_matched_delta": a3_vs_native,
            "per_1000_unique_actor_decisions": (
                None
                if a3_budget is None
                else _per_thousand(
                    a3_vs_native, int(a3_budget["unique_actor_decisions"])
                )
            ),
        },
        "navigation_rlt": {
            "raw_matched_delta": rlt_vs_native,
            "per_1000_unique_actor_decisions": _per_thousand(
                rlt_vs_native, rlt_unique_decisions
            ),
        },
    }
    return {
        "schema_name": REPORT_SCHEMA,
        "schema_version": REPORT_VERSION,
        "status": "completed",
        "run_id": run_id,
        "scope": {
            "training_split_label": "train_update",
            "matched_split_label": "train_development",
            "upstream_split_read": "train",
            "validation_split_reads": 0,
            "train_only": True,
        },
        "inputs": report_inputs,
        "training": training,
        "matched": matched,
        "fixed_budget_normalized_effect_proxy": {
            "label": "one_point_matched_effect_vs_native_per_1000_unique_actor_decisions",
            "denominator_scope": "method_specific_unique_actor_decisions",
            "a3_training_budget_available": a3_budget is not None,
            "matched_evaluation_episode_count": 32,
            "matched_evaluation_seed_count": 1,
            "not_a_sample_efficiency_estimate": True,
            "raw_rlt_replay_transition_count": transitions,
            "training_budgets": {
                "a3_ppo": a3_budget,
                "navigation_rlt": {
                    "unique_actor_decisions": rlt_unique_decisions,
                    "optimizer_presentations": rlt_presentations,
                    "eligibility_schema": stage2_budget["eligibility_schema"],
                    "credit_ineligible_items_excluded": stage2_budget[
                        "credit_ineligible_items_excluded"
                    ],
                    "gate_mismatch_items_excluded": stage2_budget[
                        "gate_mismatch_items_excluded"
                    ],
                    "global_padding_items": stage2_budget["global_padding_items"],
                    "sampler_real_row_duplication": stage2_budget[
                        "sampler_real_row_duplication"
                    ],
                    "unique_real_rows_verified": stage2_budget[
                        "unique_real_rows_verified"
                    ],
                    "padding_excluded": stage2_budget["padding_excluded"],
                    "raw_replay_transition_count": transitions,
                },
            },
            "raw_rlt_minus_a3": raw_proxy,
            "one_point_vs_native": one_point_vs_native,
            "conventional_sample_efficiency_curve_available": False,
        },
        "limitations": {
            "sample_efficiency": (
                "One fixed training budget yields a one-point proxy, not a conventional "
                "sample-efficiency curve or area-under-curve comparison."
            ),
            "communication_wait": (
                "No direct NCCL or communication-wait field was recorded; rank wall-time "
                "imbalance is reported only as a proxy."
            ),
        },
        "obs_readback": {
            "status": "readback_verified",
            "independent_readback_sha256": manifest["independent_readback_sha256"],
            "result_root": str(root),
        },
    }


def _format_number(value: object) -> str:
    if type(value) is int:
        return str(value)
    return f"{float(value):.6g}"


def render_markdown(report: Mapping[str, object]) -> str:
    training = _mapping(report["training"], "report training")
    matched = _mapping(report["matched"], "report matched")
    lines = [
        "# Navigation-RLT first train-only experiment",
        "",
        f"Run: `{report['run_id']}`",
        "",
        "This receipt covers the fixed train-only run and the matched 32-episode "
        "train-development comparison. Validation split reads are zero.",
        "",
        "## Training budget and throughput",
        "",
        "| stage | episodes | transitions | global items | epochs | batch/rank | steps/rank | item-epochs/s | rank steps/s mean | wall imbalance proxy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for phase in PHASES:
        row = _mapping(training[phase], phase)
        budget = _mapping(row["budget"], f"{phase} budget")
        throughput = _mapping(row["throughput"], f"{phase} throughput")
        lines.append(
            "| {phase} | {episodes} | {transitions} | {items} | {epochs} | {batch} | "
            "{steps} | {item_rate} | {step_rate} | {imbalance} |".format(
                phase=phase,
                episodes=budget["episode_count"],
                transitions=budget["transition_count"],
                items=budget["global_real_items"],
                epochs=budget["epochs"],
                batch=budget["batch_size_per_rank"],
                steps=budget["optimizer_steps_per_rank"],
                item_rate=_format_number(throughput["global_real_item_epochs_per_second"]),
                step_rate=_format_number(
                    throughput["rank_optimizer_steps_per_second"]["mean"]
                ),
                imbalance=_format_number(throughput["rank_wall_time_max_over_min_proxy"]),
            )
        )
    lines.extend(
        (
            "",
            "## Sampled 8×A800 utilization",
            "",
            "| stage | verified model | interval s | samples/GPU | sampled GPU util mean | per-GPU mean range | observed max | memory mean MiB | memory max MiB | power mean W | power max W |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        )
    )
    for phase in PHASES:
        gpu = _mapping(_mapping(training[phase], phase)["gpu"], f"{phase} GPU")
        util_range = gpu["gpu_util_per_gpu_mean_range_percent"]
        lines.append(
            "| {phase} | {model} | {interval} | {samples} | {mean} | {low}–{high} | {peak} | {memory_mean} | "
            "{memory_peak} | {power_mean} | {power_peak} |".format(
                phase=phase,
                model=gpu["gpu_model_name_normalized"],
                interval=_format_number(gpu["sample_interval_s"]),
                samples=gpu["sample_count_per_gpu"],
                mean=_format_number(gpu["gpu_util_mean_percent_across_gpus"]),
                low=_format_number(util_range[0]),
                high=_format_number(util_range[1]),
                peak=_format_number(gpu["gpu_util_observed_max_percent"]),
                memory_mean=_format_number(gpu["memory_used_mean_mib_across_gpus"]),
                memory_peak=_format_number(gpu["memory_used_observed_max_mib"]),
                power_mean=_format_number(gpu["power_mean_w_across_gpus"]),
                power_peak=_format_number(gpu["power_observed_max_w"]),
            )
        )
    summaries = _mapping(matched["summaries"], "matched summaries")
    lines.extend(
        (
            "",
            "## Matched navigation quality",
            "",
            "| arm | success count / 32 | SR | SPL | final NE m | path m | collisions | blocked | native STOP | learned STOP | budget STOP | timeout | latency mean ms | latency p95 ms |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        )
    )
    for arm in ARMS:
        row = _mapping(summaries[arm], arm)
        latency = _mapping(row["model_latency"], f"{arm} model latency")
        lines.append(
            "| {arm} | {successes} / {episodes} | {sr} | {spl} | {ne} | {path} | "
            "{collisions} | {blocked} | {native_stop} | {learned_stop} | "
            "{budget_stop} | {timeout} | {latency_mean} | {latency_p95} |".format(
                arm=arm,
                successes=row["success_count"],
                episodes=row["episode_count"],
                sr=_format_number(row["success_rate"]),
                spl=_format_number(row["mean_spl"]),
                ne=_format_number(row["mean_final_navigation_error_m"]),
                path=_format_number(row["mean_path_length_m"]),
                collisions=_format_number(row["collision_count"]),
                blocked=_format_number(row["blocked_step_count"]),
                native_stop=row["native_stop_count"],
                learned_stop=row["learned_stop_count"],
                budget_stop=row["budget_stop_count"],
                timeout=row["timeout_count"],
                latency_mean=_format_number(latency["mean_ms"]),
                latency_p95=_format_number(latency["p95_ms"]),
            )
        )
    lines.extend(
        (
            "",
            "## Action distributions",
            "",
            "| arm | route | mode | count | ratio |",
            "|---|---|---|---:|---:|",
        )
    )
    for arm in ARMS:
        row = _mapping(summaries[arm], arm)
        for route in ("requested_modes", "executed_modes"):
            modes = _mapping(row[route], f"{arm} {route}")
            for mode in MODES:
                lines.append(
                    f"| {arm} | {route.removesuffix('_modes')} | {mode} | "
                    f"{modes['counts'][mode]} | {_format_number(modes['ratios'][mode])} |"
                )
    proxy = _mapping(
        report["fixed_budget_normalized_effect_proxy"], "normalized-effect proxy"
    )
    raw = _mapping(proxy["raw_rlt_minus_a3"], "raw proxy")
    budgets = _mapping(proxy["training_budgets"], "training budgets")
    rlt_budget = _mapping(budgets["navigation_rlt"], "Navigation-RLT budget")
    outcomes = _mapping(proxy["one_point_vs_native"], "one-point outcomes")
    lines.extend(
        (
            "",
            "## One-point fixed-budget normalized effect proxy",
            "",
            f"Navigation-RLT Stage 2 contains **{rlt_budget['unique_actor_decisions']}** "
            "unique actor-eligible training decisions. Its raw replay contains "
            f"**{proxy['raw_rlt_replay_transition_count']}** transitions; that broader count "
            "is reported separately and is not the normalization denominator.",
        )
    )
    a3_budget_value = budgets["a3_ppo"]
    if proxy["a3_training_budget_available"]:
        a3_budget = _mapping(a3_budget_value, "A3 training budget")
        lines.append(
            f"The validated A3 receipt contains **{a3_budget['unique_actor_decisions']}** "
            "unique actor-eligible decisions and "
            f"**{a3_budget['optimizer_presentations']}** non-padding optimizer presentations."
        )
    else:
        lines.append(
            "The A3 training budget is unavailable because no validated "
            "A3_TRAINING_BUDGET_RECEIPT.json was supplied."
        )
    lines.extend(
        (
            "The matched outcome uses 32 episodes from one deterministic seed.",
            "",
            "| method | metric vs Native | raw delta | delta per 1000 unique actor decisions |",
            "|---|---|---:|---:|",
        )
    )
    for method, label in (("a3_ppo", "A3 PPO"), ("navigation_rlt", "Navigation-RLT")):
        outcome = _mapping(outcomes[method], f"{label} one-point outcome")
        method_raw = _mapping(outcome["raw_matched_delta"], f"{label} raw outcome")
        normalized_value = outcome["per_1000_unique_actor_decisions"]
        normalized = (
            None
            if normalized_value is None
            else _mapping(normalized_value, f"{label} normalized outcome")
        )
        for name in ("success_count", "SR", "SPL"):
            scaled = "unavailable" if normalized is None else _format_number(normalized[name])
            lines.append(
                f"| {label} | {name} | {_format_number(method_raw[name])} | {scaled} |"
            )
    lines.extend(
        (
            "",
            "| direct RLT - A3 metric | raw delta |",
            "|---|---:|",
        )
    )
    for name in ("success_count", "SR", "SPL"):
        lines.append(f"| {name} | {_format_number(raw[name])} |")
    lines.extend(
        (
            "",
            "These normalized outcomes are not a sample-efficiency curve or estimate. "
            "There is one training-budget point and one evaluation seed, so no "
            "area-under-curve comparison or uncertainty estimate is available.",
            "",
            "No direct NCCL or communication-wait field was recorded. The rank wall-time "
            "ratio above is only an imbalance proxy.",
            "",
            "## Archive receipt",
            "",
            f"OBS status: `{report['obs_readback']['status']}`  ",
            f"Independent readback SHA-256: `{report['obs_readback']['independent_readback_sha256']}`",
            "",
        )
    )
    return "\n".join(lines)


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDWR if os.name == "nt" else os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_parent(path: Path) -> None:
    """Persist directory metadata after a create-only publication.

    The report is produced on Linux in the experiment environment.  Windows
    does not expose directory descriptors through ``os.open``; FlushFileBuffers
    on a directory handle is the corresponding local test/development path.
    """

    parent = path.parent
    if os.name != "nt":
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        descriptor = os.open(parent, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return

    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    flush_file_buffers = kernel32.FlushFileBuffers
    flush_file_buffers.argtypes = (wintypes.HANDLE,)
    flush_file_buffers.restype = wintypes.BOOL
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL
    invalid_handle = wintypes.HANDLE(-1).value
    handle = create_file(
        str(parent),
        0x80000000 | 0x40000000,
        0x00000001 | 0x00000002 | 0x00000004,
        None,
        3,
        0x02000000,
        None,
    )
    if handle == invalid_handle:
        raise OSError(ctypes.get_last_error(), f"cannot open directory for fsync: {parent}")
    try:
        if not flush_file_buffers(handle):
            raise OSError(
                ctypes.get_last_error(), f"cannot fsync output directory: {parent}"
            )
    finally:
        close_handle(handle)


def _existing_exact(path: Path, payload: bytes, name: str) -> bool:
    if not os.path.lexists(path):
        return False
    try:
        status = path.lstat()
    except OSError as error:
        raise FinalReportError(f"cannot inspect existing {name}") from error
    if path.is_symlink() or not stat.S_ISREG(status.st_mode):
        raise FinalReportError(f"existing {name} is not a regular file")
    try:
        existing = path.read_bytes()
    except OSError as error:
        raise FinalReportError(f"cannot read existing {name}") from error
    if existing != payload:
        raise FinalReportError(f"existing {name} bytes differ from deterministic output")
    return True


def _publish_or_reuse(path: Path, payload: bytes, name: str) -> str:
    destination = Path(path).expanduser().absolute()
    if _existing_exact(destination, payload, name):
        _fsync_file(destination)
        _fsync_parent(destination)
        return "reused"
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".staging-", dir=destination.parent
    )
    temporary = Path(temporary_name)
    published = False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination, follow_symlinks=False)
            published = True
        except FileExistsError:
            if not _existing_exact(destination, payload, name):
                raise AssertionError("existing-path validation did not return or raise")
            _fsync_file(destination)
            return "reused"
        _fsync_file(destination)
        return "created"
    finally:
        temporary.unlink(missing_ok=True)
        if published or destination.parent.exists():
            _fsync_parent(destination)


def _report_payloads(
    report: Mapping[str, object],
    output_json: Path,
    output_markdown: Path,
) -> tuple[bytes, bytes, dict[str, object], bytes]:
    json_path = Path(output_json).expanduser().absolute()
    markdown_path = Path(output_markdown).expanduser().absolute()
    json_payload = (
        json.dumps(
            report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")
    markdown_payload = render_markdown(report).encode("utf-8")
    completion = {
        "schema_name": REPORT_COMPLETION_SCHEMA,
        "schema_version": 1,
        "status": "completed",
        "run_id": report["run_id"],
        "validation_split_reads": 0,
        "outputs": {
            "json": {
                "path": str(json_path),
                "sha256": hashlib.sha256(json_payload).hexdigest(),
                "size_bytes": len(json_payload),
            },
            "markdown": {
                "path": str(markdown_path),
                "sha256": hashlib.sha256(markdown_payload).hexdigest(),
                "size_bytes": len(markdown_payload),
            },
        },
    }
    completion_payload = (
        json.dumps(completion, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    return json_payload, markdown_payload, completion, completion_payload


def write_report_bundle(
    report: Mapping[str, object],
    output_json: Path,
    output_markdown: Path,
    output_completion: Path,
) -> dict[str, object]:
    proxy = _mapping(
        report.get("fixed_budget_normalized_effect_proxy"),
        "normalized-effect proxy",
    )
    if proxy.get("a3_training_budget_available") is not True:
        raise FinalReportError(
            "validated A3 training-budget receipt is required for authoritative "
            "report publication"
        )
    json_path = Path(output_json).expanduser().absolute()
    markdown_path = Path(output_markdown).expanduser().absolute()
    completion_path = Path(output_completion).expanduser().absolute()
    if len({json_path, markdown_path, completion_path}) != 3:
        raise FinalReportError("JSON, Markdown, and completion outputs must differ")
    if report.get("scope", {}).get("validation_split_reads") != 0:
        raise FinalReportError("final report does not preserve zero validation reads")
    json_payload, markdown_payload, completion, completion_payload = _report_payloads(
        report, json_path, markdown_path
    )

    json_exists = _existing_exact(json_path, json_payload, "report JSON")
    markdown_exists = _existing_exact(
        markdown_path, markdown_payload, "report Markdown"
    )
    completion_exists = _existing_exact(
        completion_path, completion_payload, "report completion marker"
    )
    if completion_exists and not (json_exists and markdown_exists):
        raise FinalReportError("completion marker exists without both exact report outputs")
    if completion_exists:
        for path in (json_path, markdown_path, completion_path):
            _fsync_file(path)
            _fsync_parent(path)
        return {
            "completion": completion,
            "publication": {
                "json": "reused",
                "markdown": "reused",
                "completion": "reused",
            },
        }

    publication = {
        "json": _publish_or_reuse(json_path, json_payload, "report JSON"),
        "markdown": _publish_or_reuse(
            markdown_path, markdown_payload, "report Markdown"
        ),
    }
    publication["completion"] = _publish_or_reuse(
        completion_path, completion_payload, "report completion marker"
    )
    return {"completion": completion, "publication": publication}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--obs-result-root", required=True, type=Path)
    result.add_argument("--a3-training-budget-receipt", required=True, type=Path)
    result.add_argument("--output-json", required=True, type=Path)
    result.add_argument("--output-markdown", required=True, type=Path)
    result.add_argument("--output-completion", required=True, type=Path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    report = build_report(
        args.obs_result_root,
        a3_training_budget_receipt=args.a3_training_budget_receipt,
    )
    publication = write_report_bundle(
        report, args.output_json, args.output_markdown, args.output_completion
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "run_id": report["run_id"],
                "output_json": str(Path(args.output_json).expanduser().absolute()),
                "output_json_sha256": sha256_file(Path(args.output_json).expanduser().absolute()),
                "output_markdown": str(Path(args.output_markdown).expanduser().absolute()),
                "output_markdown_sha256": sha256_file(
                    Path(args.output_markdown).expanduser().absolute()
                ),
                "output_completion": str(
                    Path(args.output_completion).expanduser().absolute()
                ),
                "output_completion_sha256": sha256_file(
                    Path(args.output_completion).expanduser().absolute()
                ),
                "publication": publication["publication"],
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
