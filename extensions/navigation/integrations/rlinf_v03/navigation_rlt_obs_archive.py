#!/usr/bin/env python3
"""Archive a completed two-stage Navigation-RLT run with OBS readback.

Only bounded provenance, training evidence, telemetry, source files, matched
results, and the two final checkpoints are copied.  Replay manifests and their
completion receipts are archived, while replay shard payloads remain in their
declared storage locations.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import errno
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Mapping, Sequence

import navigation_rlt_post_training as post_training


RUNS_ROOT = Path("/srv/navigation/runs")
OBS_MOUNT = Path("/obs")
DEFAULT_OBS_ROOT = OBS_MOUNT / "navigation-example"
CHECKPOINT_NAMESPACE = Path("checkpoints/navigation-rlt")
RESULT_NAMESPACE = Path("results/navigation-rlt")
SCHEMA_NAME = "navigation_rlt_obs_archive"
SCHEMA_VERSION = 1
COMPLETION_SCHEMA = "navigation_rlt_obs_completion"
MATCHED_SCHEMA = "navigation_rlt_matched_comparison"
MATCHED_VERSION = 1
COLLECTION_MANIFEST_SCHEMA = "navigation_rlt_experiment_manifest"
EVALUATION_INVENTORY_SCHEMA = "navigation_rlt_evaluation_code_inventory"
TRAINING_CODE_VERIFICATION_SCHEMA = "navigation_rlt_a800_code_verification"
TRAINING_PROTOCOL_SCHEMA = "navigation_rlt_training_protocol"
STAGE2_REPAIR_PROTOCOL_SCHEMA = "navigation_rlt_stage2_repair_protocol"
STAGE2_REPAIR_OBJECTIVE_SCHEMA = "proposal_conditioned_finite_candidate_cql_v1"
STAGE2_BEHAVIOR_KL_OBJECTIVE_SCHEMA = (
    "proposal_conditioned_finite_candidate_cql_behavior_kl_v2"
)
TRAINING_INVENTORY_SCHEMA = "navigation_rlt_code_archive_inventory"
VERIFIED_COPY_SCHEMA = "navigation_rlt_verified_checkpoint_copy"
EVALUATION_HANDOFF_SCHEMA = "navigation_rlt_remote_evaluation_handoff"
MAX_CHECKPOINT_BYTES = 2 * 1024**3
MAX_RESULTS_BYTES = 4 * 1024**3
MAX_SOURCE_BYTES = 64 * 1024**2
MAX_RESULT_FILES = 512
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
SHA1_PATTERN = re.compile(r"[0-9a-f]{40}")
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
EVALUATION_ARMS = ("native_lightnav", "a3_ppo", "navigation_rlt")
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
TRAINING_PROTOCOL_FIELDS = {
    "schema_name",
    "schema_version",
    "run_id",
    "preregistration_commit",
    "navigation_training_code_commit",
    "boundaries",
    "rollout",
    "model",
    "stage1",
    "heldout_gate",
    "stage2",
    "stage1_checkpoint_provenance_gate",
}
STAGE1_PROTOCOL_FIELDS = (
    "seed",
    "epochs",
    "batch_size_per_rank",
    "learning_rate",
    "weight_decay",
    "clip_grad",
    "prefix_mask_ratio",
    "topology_mask_ratio",
    "prefix_weight",
    "topology_weight",
    "required_world_size",
    "expected_episode_count",
)
STAGE2_PROTOCOL_FIELDS = (
    "seed",
    "epochs",
    "batch_size_per_rank",
    "learning_rate",
    "weight_decay",
    "clip_grad",
    "reference_gate_weight",
    "reference_residual_weight",
    "reference_dropout_probability",
    "target_tau",
    "policy_version",
    "required_world_size",
    "expected_episode_count",
)
STAGE2_REPAIR_TRAINING_FIELDS = (
    "objective_schema",
    "cql_alpha",
    "cql_temperature",
    "minimum_behavior_gate_count",
)
STAGE2_BEHAVIOR_KL_TRAINING_FIELDS = (
    *STAGE2_REPAIR_TRAINING_FIELDS,
    "behavior_kl_weight",
)
STAGE2_REPAIR_PROTOCOL_FIELDS = {
    "schema_name",
    "schema_version",
    "run_id",
    "base_training_protocol_sha256",
    "stage1_code_verification_sha256",
    "stage1_checkpoint_sha256",
    "stage2_repair_code_verification_sha256",
    "stage2_replay_manifest_sha256",
    "stage2_run_manifest_sha256",
    "navigation_training_code_commit",
    "validation_split_reads",
    "stage2_training_additions",
    "stage2_support",
}
STAGE2_GATE_NAMES = ("KEEP", "ADJUST", "STOP", "RECOVER")
STAGE2_SUPPORT_FIELDS = {
    "eligibility_schema",
    "gate_names",
    "gate_counts",
    "gate_support_mask",
    "minimum_behavior_gate_count",
}
STAGE2_ELIGIBILITY_SCHEMA = (
    "credit_mask_true_and_requested_gate_equals_executed_gate_v1"
)
STAGE2_CQL_ALPHA = 1.0
STAGE2_CQL_TEMPERATURE = 1.0
STAGE2_MINIMUM_BEHAVIOR_GATE_COUNT = 1
HELDOUT_GATE_PROTOCOL_FIELDS = (
    "seed",
    "batch_size",
    "prefix_mask_ratio",
    "topology_mask_ratio",
    "minimum_relative_improvement",
    "minimum_z_rl_variance",
)
MODEL_PROTOCOL_FIELDS = (
    "attention_heads",
    "critic_hidden_dim",
    "dropout",
    "feedforward_dim",
    "initial_log_std",
    "max_log_std",
    "max_prefix_tokens",
    "min_log_std",
    "residual_forward_limit_m",
    "residual_yaw_limit_rad",
    "transformer_layers",
)
PROTOCOL_BOUNDARIES = {
    "lightnav_frozen": True,
    "remote_push_allowed": False,
    "ttt_allowed": False,
    "validation_splits_allowed": False,
}
PROVENANCE_GATE_PROTOCOL = {
    "required": True,
    "run_manifest_sha256_source": "navigation_rlt_stage1/STAGE.json:run_manifest.sha256",
    "replay_manifest_sha256_source": "navigation_rlt_stage1/STAGE.json:replay_manifest.sha256",
    "required_cli_arguments": [
        "--expected-stage1-run-manifest-sha256",
        "--expected-stage1-replay-manifest-sha256",
    ],
}
SOURCE_FILES = {
    "trainer_sha256": "navigation_rlt_fsdp_train.py",
    "policy_sha256": "navigation_rlt_policy.py",
    "replay_sha256": "navigation_rlt_replay.py",
    "replay_relocator_sha256": "navigation_rlt_relocate_replay.py",
    "stage1_gate_sha256": "navigation_rlt_stage1_gate.py",
    "preparer_sha256": "navigation_rlt_prepare_training.py",
    "promoter_sha256": "navigation_rlt_promote_replay.py",
    "wrapper_sha256": "run_navigation_rlt_a800.sh",
    "gpu_monitor_sha256": "lightnav_a3_gpu_monitor.py",
}
STAGE2_REPAIR_REQUIRED_SOURCE_CLAIMS = frozenset(
    {
        "trainer_sha256",
        "policy_sha256",
        "replay_sha256",
        "preparer_sha256",
        "promoter_sha256",
        "wrapper_sha256",
        "gpu_monitor_sha256",
    }
)


class ArchiveError(RuntimeError):
    """Raised when an archive input, destination, or readback is unsafe."""


@dataclass(frozen=True)
class Artifact:
    source: Path
    group: str
    relative: Path
    kind: str
    sha256: str
    size_bytes: int


@dataclass
class StageEvidence:
    phase: str
    stage_root: Path
    checkpoint: Path
    checkpoint_sha256: str
    training_result: dict[str, object]
    run_manifest: dict[str, object]
    replay_manifest: dict[str, object]
    replay_summary: dict[str, object]
    run_manifest_sha256: str
    replay_manifest_sha256: str
    run_package_receipt: dict[str, object]
    replay_package_receipt: dict[str, object]
    source_claims: list[dict[str, str]]
    artifacts: list[Artifact]
    successor: dict[str, object] | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ArchiveError(f"{name} must be a lowercase SHA-256")
    return value


def _validate_sha1(value: object, name: str) -> str:
    if not isinstance(value, str) or SHA1_PATTERN.fullmatch(value) is None:
        raise ArchiveError(f"{name} must be a lowercase full Git SHA-1")
    return value


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _regular_file(path: Path, root: Path, name: str) -> Path:
    raw = Path(path)
    if raw.is_symlink():
        raise ArchiveError(f"{name} must not be a symbolic link")
    try:
        resolved = raw.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise ArchiveError(f"{name} is missing or escapes {root}") from error
    if not resolved.is_file():
        raise ArchiveError(f"{name} must be a regular file")
    return resolved


def _regular_directory(path: Path, root: Path, name: str) -> Path:
    raw = Path(path)
    if raw.is_symlink():
        raise ArchiveError(f"{name} must not be a symbolic link")
    try:
        resolved = raw.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise ArchiveError(f"{name} is missing or escapes {root}") from error
    if not resolved.is_dir():
        raise ArchiveError(f"{name} must be a directory")
    return resolved


def _reject_constant(value: str) -> object:
    raise ArchiveError(f"non-finite JSON constant is forbidden: {value}")


def _load_json(path: Path, name: str) -> dict[str, object]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=_reject_constant
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArchiveError(f"{name} is not strict JSON") from error
    if not isinstance(value, dict):
        raise ArchiveError(f"{name} must contain a JSON object")
    return value


def _matched_checkpoint_sha256(
    matched: Mapping[str, object], *, run_id: str
) -> str:
    if (
        matched.get("schema_name") != MATCHED_SCHEMA
        or matched.get("schema_version") != MATCHED_VERSION
        or matched.get("status") != "completed"
        or matched.get("run_id") != run_id
    ):
        raise ArchiveError("matched result schema/status/run_id differs")
    direct = _validate_sha256(
        matched.get("navigation_rlt_checkpoint_sha256"),
        "matched result Navigation-RLT checkpoint SHA-256",
    )
    checkpoints = matched.get("checkpoints")
    if not isinstance(checkpoints, Mapping):
        raise ArchiveError("matched result checkpoint inventory is absent")
    navigation_rlt = checkpoints.get("navigation_rlt")
    if not isinstance(navigation_rlt, Mapping):
        raise ArchiveError("matched result Navigation-RLT checkpoint is absent")
    nested = _validate_sha256(
        navigation_rlt.get("sha256"),
        "matched result nested Navigation-RLT checkpoint SHA-256",
    )
    if direct != nested:
        raise ArchiveError("matched result Navigation-RLT checkpoint identities disagree")
    return direct


def _safe_relative(value: object, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ArchiveError(f"{name} must be a non-empty relative path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise ArchiveError(f"{name} is not a safe relative path")
    return Path(*pure.parts)


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


def _renameat2_noreplace(source: Path, destination: Path) -> None:
    if not sys.platform.startswith("linux"):
        raise ArchiveError("archive directory publication requires Linux renameat2")
    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as error:
        raise ArchiveError("Linux renameat2 is unavailable") from error
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = renameat2(
        -100,  # AT_FDCWD
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,  # RENAME_NOREPLACE
    )
    if result != 0:
        error_number = ctypes.get_errno() or errno.EIO
        raise OSError(error_number, os.strerror(error_number), str(destination))


def _publish_directory_new(staging: Path, destination: Path) -> None:
    staging_parent = staging.parent.resolve(strict=True)
    destination_parent = destination.parent.resolve(strict=True)
    if staging_parent != destination_parent:
        raise ArchiveError("archive directory publication requires one parent")
    if staging.is_symlink() or not staging.is_dir():
        raise ArchiveError(f"archive staging directory is unsafe: {staging}")
    _renameat2_noreplace(staging, destination)


def _publish_archive_pair(
    checkpoint_staging: Path,
    checkpoint_final: Path,
    result_staging: Path,
    result_final: Path,
) -> None:
    _publish_directory_new(checkpoint_staging, checkpoint_final)
    try:
        _publish_directory_new(result_staging, result_final)
    except BaseException as publish_error:
        try:
            _publish_directory_new(checkpoint_final, checkpoint_staging)
        except BaseException as rollback_error:
            raise ArchiveError(
                "result archive publication failed and checkpoint rollback failed; "
                "retained attributable archive state at "
                f"{checkpoint_final}, {checkpoint_staging}, {result_staging}, "
                f"and {result_final}; publication error: {publish_error!r}"
            ) from rollback_error
        raise


def _publish_file_new(
    temporary: Path, destination: Path, identity: tuple[int, int]
) -> None:
    if temporary.parent.resolve(strict=True) != destination.parent.resolve(strict=True):
        raise ArchiveError("atomic publication requires one destination directory")
    os.link(temporary, destination, follow_symlinks=False)
    _fsync_parent(destination)
    _unlink_if_same(temporary, identity)
    _fsync_parent(destination)


def _atomic_json_new(path: Path, value: object) -> str:
    try:
        payload = (
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ArchiveError("archive output is not strict JSON") from error
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".staging-", dir=path.parent
    )
    temporary = Path(temporary_name)
    temporary_identity = _file_identity(os.fstat(descriptor))
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _publish_file_new(temporary, path, temporary_identity)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        _unlink_if_same(temporary, temporary_identity)
        raise
    return hashlib.sha256(payload).hexdigest()


def _artifact(
    source: Path,
    *,
    source_root: Path,
    group: str,
    relative: Path,
    kind: str,
) -> Artifact:
    path = _regular_file(source, source_root, kind)
    safe_relative = _safe_relative(relative.as_posix(), f"{kind} destination")
    return Artifact(
        source=path,
        group=group,
        relative=safe_relative,
        kind=kind,
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
    )


def _artifact_from_reference(
    value: object,
    *,
    source_root: Path,
    group: str,
    relative: Path,
    kind: str,
) -> Artifact:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ArchiveError(f"{kind} reference differs")
    artifact = _artifact(
        Path(str(value.get("path"))),
        source_root=source_root,
        group=group,
        relative=relative,
        kind=kind,
    )
    expected = _validate_sha256(value.get("sha256"), f"{kind} SHA-256")
    if artifact.sha256 != expected:
        raise ArchiveError(f"{kind} SHA-256 differs")
    return artifact


def _successor_provenance_artifacts(
    successor: Mapping[str, object], *, run_root: Path
) -> list[Artifact]:
    identity = successor.get("successor_source_identity")
    if not isinstance(identity, Mapping):
        raise ArchiveError("Stage2 successor source identity is absent")
    references = (
        (
            successor.get("formal_completion"),
            Path("stage2/provenance/successor/FORMAL_COMPLETED.json"),
            "Stage2 successor formal completion",
        ),
        (
            identity.get("derivation"),
            Path("stage2/provenance/successor/prepare/DERIVATION.json"),
            "Stage2 successor derivation",
        ),
        (
            identity.get("prepare_completion"),
            Path("stage2/provenance/successor/prepare/COMPLETED.json"),
            "Stage2 successor prepare completion",
        ),
        (
            identity.get("base_run_manifest"),
            Path("stage2/provenance/successor/base/run_manifest.json"),
            "Stage2 successor base run manifest",
        ),
        (
            identity.get("base_prepare_receipt"),
            Path("stage2/provenance/successor/base/prepare_receipt.json"),
            "Stage2 successor base prepare receipt",
        ),
        (
            identity.get("base_prepare_completion"),
            Path("stage2/provenance/successor/base/COMPLETED.json"),
            "Stage2 successor base prepare completion",
        ),
    )
    artifacts = [
        _artifact_from_reference(
            reference,
            source_root=run_root,
            group="results",
            relative=relative,
            kind=kind,
        )
        for reference, relative, kind in references
    ]
    source_files = identity.get("source_files")
    if not isinstance(source_files, Mapping) or set(source_files) != {
        "trainer_sha256",
        "policy_sha256",
        "replay_sha256",
        "wrapper_sha256",
        "gpu_monitor_sha256",
    }:
        raise ArchiveError("Stage2 successor source file inventory differs")
    source_root = _regular_directory(
        Path(str(identity.get("source_root"))),
        run_root,
        "Stage2 successor source root",
    )
    for claim, reference in sorted(source_files.items()):
        if not isinstance(reference, Mapping):
            raise ArchiveError("Stage2 successor source reference differs")
        source = Path(str(reference.get("path")))
        try:
            relative = source.resolve(strict=True).relative_to(source_root)
        except (OSError, ValueError) as error:
            raise ArchiveError("Stage2 successor source escapes its root") from error
        artifacts.append(
            _artifact_from_reference(
                reference,
                source_root=source_root,
                group="results",
                relative=Path("source/stage2-successor") / relative,
                kind=f"Stage2 successor source {claim}",
            )
        )
    return artifacts


def _package_evidence(
    manifest_path: Path,
    manifest_sha256: str,
    *,
    phase: str,
    label: str,
    run_id: str,
    runs_root: Path,
) -> tuple[list[Artifact], dict[str, object], dict[str, str]]:
    candidates = [manifest_path.parent, *manifest_path.parents]
    package_root: Path | None = None
    marker: dict[str, object] | None = None
    marker_path: Path | None = None
    for candidate in candidates:
        try:
            candidate.relative_to(runs_root)
        except ValueError:
            break
        possible = candidate / "COMPLETED.json"
        if possible.is_file() and not possible.is_symlink():
            possible_marker = _load_json(possible, f"{phase} {label} completion")
            schema = possible_marker.get("schema_name")
            if schema in (
                "navigation_rlt_prepare_training",
                "navigation_rlt_replay_promotion",
            ):
                package_root = candidate
                marker = possible_marker
                marker_path = possible.resolve(strict=True)
                break
    if package_root is None or marker is None or marker_path is None:
        raise ArchiveError(f"{phase} {label} has no completed provenance package")
    if marker.get("status") != "completed":
        raise ArchiveError(f"{phase} {label} package is not completed")

    schema = marker["schema_name"]
    if schema == "navigation_rlt_prepare_training":
        if marker.get("schema_version") != 1 or marker.get("artifact_sha256") != manifest_sha256:
            raise ArchiveError(f"{phase} {label} prepare marker differs")
        receipt_path = _regular_file(
            package_root / "prepare_receipt.json", runs_root, f"{phase} {label} receipt"
        )
        receipt_sha = _validate_sha256(
            marker.get("receipt_sha256"), f"{phase} {label} receipt SHA-256"
        )
        receipt_name = "prepare_receipt.json"
    else:
        if (
            marker.get("schema_version") != 1
            or marker.get("output_manifest_sha256") != manifest_sha256
        ):
            raise ArchiveError(f"{phase} {label} promotion marker differs")
        receipt_path = _regular_file(
            package_root / "promotion_receipt.json",
            runs_root,
            f"{phase} {label} receipt",
        )
        receipt_sha = _validate_sha256(
            marker.get("promotion_receipt_sha256"),
            f"{phase} {label} receipt SHA-256",
        )
        receipt_name = "promotion_receipt.json"
    if sha256_file(receipt_path) != receipt_sha:
        raise ArchiveError(f"{phase} {label} receipt SHA-256 differs")
    receipt = _load_json(receipt_path, f"{phase} {label} receipt")
    if receipt.get("status") != "completed":
        raise ArchiveError(f"{phase} {label} receipt is not completed")
    receipt_run_id = receipt.get("run_id")
    if receipt_run_id is not None and receipt_run_id != run_id:
        raise ArchiveError(f"{phase} {label} receipt run_id differs")
    implementation = receipt.get("implementation")
    if not isinstance(implementation, dict):
        raise ArchiveError(f"{phase} {label} receipt lacks implementation hashes")
    claims = {
        str(key): _validate_sha256(value, f"{phase} {label} {key}")
        for key, value in implementation.items()
    }

    prefix = Path(phase) / "provenance" / label
    artifacts = [
        _artifact(
            marker_path,
            source_root=runs_root,
            group="results",
            relative=prefix / "COMPLETED.json",
            kind=f"{phase} {label} package marker",
        ),
        _artifact(
            receipt_path,
            source_root=runs_root,
            group="results",
            relative=prefix / receipt_name,
            kind=f"{phase} {label} package receipt",
        ),
    ]
    return artifacts, receipt, claims


def _validate_replay_manifest(
    value: Mapping[str, object], *, phase: str, run_id: str
) -> dict[str, object]:
    representation = "prefix_hidden" if phase == "stage1" else "frozen_z_rl"
    if (
        value.get("schema_name") != "navigation_rlt_replay_manifest"
        or value.get("schema_version") != 3
        or value.get("run_id") != run_id
        or value.get("representation_kind") != representation
        or value.get("split_label") != "train_update"
        or type(value.get("validation_split_reads")) is not int
        or value.get("validation_split_reads") != 0
    ):
        raise ArchiveError(f"{phase} replay manifest contract differs")
    expected = value.get("expected_episode_count")
    completed = value.get("completed_episode_count")
    transitions = value.get("transition_count")
    shards = value.get("shards")
    if (
        type(expected) is not int
        or expected < 1
        or completed != expected
        or type(transitions) is not int
        or transitions < 1
        or not isinstance(shards, list)
        or len(shards) != expected
    ):
        raise ArchiveError(f"{phase} replay manifest fixed budget differs")
    identities: set[tuple[str, str]] = set()
    paths: set[str] = set()
    hashes: set[str] = set()
    declared_bytes = 0
    declared_rows = 0
    for index, receipt in enumerate(shards):
        if not isinstance(receipt, dict):
            raise ArchiveError(f"{phase} replay shard {index} receipt is invalid")
        path_value = receipt.get("path")
        if not isinstance(path_value, str) or not path_value:
            raise ArchiveError(f"{phase} replay shard {index} path is absent")
        shard_path = Path(path_value)
        if not shard_path.is_absolute():
            raise ArchiveError(f"{phase} replay shard {index} path is not absolute")
        try:
            shard_path.resolve(strict=False).relative_to(RUNS_ROOT)
        except ValueError as error:
            raise ArchiveError(f"{phase} replay shard {index} path escapes runs") from error
        shard_sha = _validate_sha256(
            receipt.get("sha256"), f"{phase} replay shard {index} SHA-256"
        )
        identity = (str(receipt.get("episode_id")), str(receipt.get("reset_token")))
        canonical_path = str(shard_path.resolve(strict=False))
        if identity in identities or canonical_path in paths or shard_sha in hashes:
            raise ArchiveError(f"{phase} replay manifest contains a duplicate shard")
        identities.add(identity)
        paths.add(canonical_path)
        hashes.add(shard_sha)
        size = receipt.get("size_bytes")
        rows = receipt.get("row_count")
        if type(size) is not int or size < 1 or type(rows) is not int or rows < 1:
            raise ArchiveError(f"{phase} replay shard {index} size/rows are invalid")
        declared_bytes += size
        declared_rows += rows
    if declared_rows != transitions:
        raise ArchiveError(f"{phase} replay transition count differs from shards")
    return {
        "representation_kind": representation,
        "source_fingerprint": _validate_sha256(
            value.get("source_fingerprint"), f"{phase} replay source fingerprint"
        ),
        "episode_count": completed,
        "transition_count": transitions,
        "shard_count": len(shards),
        "declared_shard_bytes": declared_bytes,
        "shard_payloads_archived": False,
    }


def _validate_inventory(
    training_root: Path, inventory: Mapping[str, object], *, phase: str
) -> dict[str, Path]:
    if not inventory or len(inventory) > MAX_RESULT_FILES:
        raise ArchiveError(f"{phase} training inventory size is invalid")
    files: dict[str, Path] = {}
    for raw_relative, raw_receipt in inventory.items():
        relative = _safe_relative(raw_relative, f"{phase} inventory path")
        if not isinstance(raw_receipt, dict):
            raise ArchiveError(f"{phase} inventory receipt is invalid")
        path = _regular_file(
            training_root / relative, training_root, f"{phase} training artifact"
        )
        expected_sha = _validate_sha256(
            raw_receipt.get("sha256"), f"{phase} inventory SHA-256"
        )
        if sha256_file(path) != expected_sha or raw_receipt.get("size_bytes") != path.stat().st_size:
            raise ArchiveError(f"{phase} training inventory readback differs: {relative}")
        files[relative.as_posix()] = path
    required = {
        "training_result.json",
        "checkpoint_restore_receipt.json",
        "checkpoints/latest/navigation_rlt_policy.pt",
        *(f"ranks/rank_{index:03d}.json" for index in range(8)),
    }
    if not required.issubset(files):
        raise ArchiveError(f"{phase} training inventory lacks required receipts")
    return files


def _validate_stage(
    run_root: Path,
    *,
    phase: str,
    run_id: str,
    runs_root: Path,
    stage_run_root: Path | None = None,
    successor: Mapping[str, object] | None = None,
) -> StageEvidence:
    effective_run_root = run_root
    if successor is not None:
        if phase != "stage2" or stage_run_root is None:
            raise ArchiveError("successor evidence is only valid for Stage2")
        effective_run_root = _regular_directory(
            stage_run_root, run_root, "Stage2 successor run root"
        )
        if successor.get("stage_run_root") != str(effective_run_root):
            raise ArchiveError("Stage2 successor run root differs")
    elif stage_run_root is not None:
        raise ArchiveError("legacy stage must not override its run root")
    stage_root = _regular_directory(
        effective_run_root / f"navigation_rlt_{phase}",
        effective_run_root,
        f"{phase} stage root",
    )
    marker_path = _regular_file(stage_root / "STAGE.json", stage_root, f"{phase} marker")
    marker = _load_json(marker_path, f"{phase} marker")
    if successor is not None and successor.get("marker") != {
        "path": str(marker_path),
        "sha256": sha256_file(marker_path),
    }:
        raise ArchiveError("Stage2 successor marker reference differs")
    if (
        marker.get("schema_name") != "navigation_rlt_a800_stage"
        or marker.get("schema_version") != 1
        or marker.get("status") != "completed"
        or marker.get("phase") != phase
        or marker.get("run_root") != str(effective_run_root)
        or marker.get("exit_code") != 0
    ):
        raise ArchiveError(f"{phase} A800 stage marker differs")
    exit_codes = marker.get("process_exit_codes")
    if exit_codes != {"torchrun": 0, "gpu_monitor": 0, "telemetry_summary": 0}:
        raise ArchiveError(f"{phase} process receipts are not all successful")

    output = marker.get("output")
    if not isinstance(output, dict):
        raise ArchiveError(f"{phase} output receipt is absent")
    training_root = _regular_directory(
        Path(str(output.get("training"))), stage_root, f"{phase} training root"
    )
    if training_root != (stage_root / "training").resolve(strict=True):
        raise ArchiveError(f"{phase} training root is not canonical")
    checkpoint_receipt = output.get("checkpoint")
    if not isinstance(checkpoint_receipt, dict):
        raise ArchiveError(f"{phase} checkpoint receipt is absent")
    checkpoint = _regular_file(
        Path(str(checkpoint_receipt.get("path"))), training_root, f"{phase} checkpoint"
    )
    expected_checkpoint_sha = _validate_sha256(
        checkpoint_receipt.get("sha256"), f"{phase} checkpoint SHA-256"
    )
    if checkpoint.stat().st_size > MAX_CHECKPOINT_BYTES:
        raise ArchiveError(f"{phase} checkpoint exceeds the 2 GiB hard cap")
    if sha256_file(checkpoint) != expected_checkpoint_sha:
        raise ArchiveError(f"{phase} checkpoint SHA-256 differs")
    if successor is not None:
        successor_checkpoint = successor.get("checkpoint")
        if (
            not isinstance(successor_checkpoint, Mapping)
            or successor_checkpoint.get("path") != str(checkpoint)
            or successor_checkpoint.get("sha256") != expected_checkpoint_sha
            or successor_checkpoint.get("size_bytes") != checkpoint.stat().st_size
        ):
            raise ArchiveError("Stage2 successor checkpoint reference differs")

    completion_path = _regular_file(
        training_root / "COMPLETED.json", training_root, f"{phase} training completion"
    )
    result_path = _regular_file(
        training_root / "training_result.json", training_root, f"{phase} training result"
    )
    inventory_path = _regular_file(
        training_root / "artifact_inventory.json", training_root, f"{phase} inventory"
    )
    completion = _load_json(completion_path, f"{phase} training completion")
    result = _load_json(result_path, f"{phase} training result")
    inventory = _load_json(inventory_path, f"{phase} training inventory")
    expected_completion = {
        "schema_name": "navigation_rlt_training_completion",
        "schema_version": 1,
        "status": "completed",
        "phase": phase,
        "training_result_sha256": sha256_file(result_path),
        "artifact_inventory_sha256": sha256_file(inventory_path),
    }
    if completion != expected_completion:
        raise ArchiveError(f"{phase} training completion marker differs")
    if successor is not None and successor.get("training_result") != {
        "path": str(result_path),
        "sha256": sha256_file(result_path),
    }:
        raise ArchiveError("Stage2 successor training result reference differs")
    if (
        result.get("schema_name") != "navigation_rlt_rlinf_fsdp_training"
        or result.get("schema_version") != 1
        or result.get("status") != "completed"
        or result.get("phase") != phase
        or result.get("run_id") != run_id
        or result.get("world_size") != 8
        or result.get("unique_gpu_uuid_count") != 8
    ):
        raise ArchiveError(f"{phase} training result contract differs")
    final_checkpoint = result.get("final_checkpoint")
    restore = result.get("independent_restore")
    if (
        not isinstance(final_checkpoint, dict)
        or final_checkpoint.get("sha256") != expected_checkpoint_sha
        or not isinstance(restore, dict)
        or restore.get("status") != "passed"
        or restore.get("checkpoint_sha256") != expected_checkpoint_sha
    ):
        raise ArchiveError(f"{phase} checkpoint/restore evidence differs")
    ranks = result.get("rank_receipts")
    if not isinstance(ranks, list) or len(ranks) != 8:
        raise ArchiveError(f"{phase} rank receipt count differs")
    if {row.get("rank") for row in ranks if isinstance(row, dict)} != set(range(8)):
        raise ArchiveError(f"{phase} rank receipt identities differ")
    uuids = {
        row.get("gpu_start", {}).get("gpu_uuid")
        for row in ranks
        if isinstance(row, dict) and isinstance(row.get("gpu_start"), dict)
    }
    if len(uuids) != 8 or None in uuids:
        raise ArchiveError(f"{phase} rank GPU UUIDs are not unique")
    inventory_files = _validate_inventory(training_root, inventory, phase=phase)
    if inventory_files["checkpoints/latest/navigation_rlt_policy.pt"] != checkpoint:
        raise ArchiveError(f"{phase} inventory checkpoint path differs")

    verification_receipt = marker.get("verification")
    preflight_receipt = marker.get("preflight")
    if not isinstance(verification_receipt, dict) or not isinstance(preflight_receipt, dict):
        raise ArchiveError(f"{phase} wrapper verification/preflight is absent")
    verification_path = _regular_file(
        Path(str(verification_receipt.get("path"))), stage_root, f"{phase} verification"
    )
    preflight_path = _regular_file(
        Path(str(preflight_receipt.get("path"))), stage_root, f"{phase} preflight"
    )
    for path, receipt, name in (
        (verification_path, verification_receipt, "verification"),
        (preflight_path, preflight_receipt, "preflight"),
    ):
        expected = _validate_sha256(receipt.get("sha256"), f"{phase} {name} SHA-256")
        if sha256_file(path) != expected:
            raise ArchiveError(f"{phase} {name} SHA-256 differs")
    verification = _load_json(verification_path, f"{phase} verification")
    if successor is not None and successor.get("verification") != {
        "path": str(verification_path),
        "sha256": sha256_file(verification_path),
    }:
        raise ArchiveError("Stage2 successor verification reference differs")
    if (
        verification.get("status") != "passed"
        or verification.get("phase") != phase
        or verification.get("world_size") != 8
        or verification.get("unique_gpu_uuid_count") != 8
        or verification.get("checkpoint", {}).get("sha256") != expected_checkpoint_sha
    ):
        raise ArchiveError(f"{phase} wrapper verification differs")

    telemetry_path = _regular_file(
        Path(str(output.get("telemetry"))), stage_root, f"{phase} telemetry"
    )
    telemetry_summary_path = _regular_file(
        Path(str(output.get("telemetry_summary"))),
        stage_root,
        f"{phase} telemetry summary",
    )
    telemetry_summary = _load_json(telemetry_summary_path, f"{phase} telemetry summary")
    gpus = telemetry_summary.get("gpus")
    if telemetry_summary.get("gpu_count") != 8 or not isinstance(gpus, list) or len(gpus) != 8:
        raise ArchiveError(f"{phase} telemetry does not contain eight GPUs")
    for gpu in gpus:
        if not isinstance(gpu, dict) or type(gpu.get("sample_count")) is not int or gpu["sample_count"] < 1:
            raise ArchiveError(f"{phase} telemetry sample count is invalid")
        for name in (
            "memory_used_mean_mib",
            "memory_used_max_mib",
            "gpu_util_mean_percent",
            "gpu_util_max_percent",
            "power_mean_w",
            "power_max_w",
        ):
            value = gpu.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ArchiveError(f"{phase} telemetry {name} is non-finite")

    run_receipt = marker.get("run_manifest")
    replay_receipt = marker.get("replay_manifest")
    if not isinstance(run_receipt, dict) or not isinstance(replay_receipt, dict):
        raise ArchiveError(f"{phase} manifest receipts are absent")
    run_manifest_path = _regular_file(
        Path(str(run_receipt.get("path"))), runs_root, f"{phase} run manifest"
    )
    replay_manifest_path = _regular_file(
        Path(str(replay_receipt.get("path"))), runs_root, f"{phase} replay manifest"
    )
    run_manifest_sha = _validate_sha256(
        run_receipt.get("sha256"), f"{phase} run manifest SHA-256"
    )
    replay_manifest_sha = _validate_sha256(
        replay_receipt.get("sha256"), f"{phase} replay manifest SHA-256"
    )
    if sha256_file(run_manifest_path) != run_manifest_sha:
        raise ArchiveError(f"{phase} run manifest SHA-256 differs")
    if sha256_file(replay_manifest_path) != replay_manifest_sha:
        raise ArchiveError(f"{phase} replay manifest SHA-256 differs")
    run_manifest = _load_json(run_manifest_path, f"{phase} run manifest")
    replay_manifest = _load_json(replay_manifest_path, f"{phase} replay manifest")
    if successor is not None and successor.get("replay_manifest") != {
        "path": str(replay_manifest_path),
        "sha256": replay_manifest_sha,
    }:
        raise ArchiveError("Stage2 successor replay reference differs")
    result_run = result.get("run_manifest")
    result_replay = result.get("replay_manifest")
    if (
        run_manifest.get("schema_name") != "navigation_rlt_training_run_manifest"
        or run_manifest.get("schema_version") != 1
        or run_manifest.get("run_id") != run_id
        or run_manifest.get("phase") != phase
        or run_manifest.get("required_world_size") != 8
        or not isinstance(result_run, dict)
        or result_run.get("sha256") != run_manifest_sha
    ):
        raise ArchiveError(f"{phase} run manifest/result binding differs")
    manifest_replay = run_manifest.get("replay")
    if (
        not isinstance(manifest_replay, dict)
        or manifest_replay.get("manifest_sha256") != replay_manifest_sha
        or not isinstance(result_replay, dict)
        or result_replay.get("sha256") != replay_manifest_sha
    ):
        raise ArchiveError(f"{phase} replay manifest/result binding differs")
    replay_summary = _validate_replay_manifest(replay_manifest, phase=phase, run_id=run_id)
    if manifest_replay.get("source_fingerprint") != replay_summary["source_fingerprint"]:
        raise ArchiveError(f"{phase} replay source binding differs")
    checkpoint_metadata = final_checkpoint.get("metadata")
    checkpoint_extra = (
        checkpoint_metadata.get("extra")
        if isinstance(checkpoint_metadata, dict)
        else None
    )
    if (
        not isinstance(checkpoint_extra, dict)
        or checkpoint_extra.get("training_phase") != phase
        or checkpoint_extra.get("run_manifest_sha256") != run_manifest_sha
        or checkpoint_extra.get("replay_manifest_sha256") != replay_manifest_sha
    ):
        raise ArchiveError(f"{phase} final checkpoint run/replay provenance differs")

    successor_artifacts: list[Artifact] = []
    if successor is None:
        run_package_artifacts, run_package_receipt, run_claims = _package_evidence(
            run_manifest_path,
            run_manifest_sha,
            phase=phase,
            label="run_manifest",
            run_id=run_id,
            runs_root=runs_root,
        )
    else:
        training_ref = successor.get("training_run_manifest")
        identity = successor.get("successor_source_identity")
        if (
            not isinstance(training_ref, Mapping)
            or training_ref.get("path") != str(run_manifest_path)
            or training_ref.get("sha256") != run_manifest_sha
            or not isinstance(identity, Mapping)
            or not isinstance(identity.get("base_prepare_receipt_value"), Mapping)
            or not isinstance(identity.get("base_preparation_claims"), Mapping)
        ):
            raise ArchiveError("Stage2 successor run/base preparation differs")
        run_package_artifacts = []
        run_package_receipt = dict(identity["base_prepare_receipt_value"])
        run_claims = {
            str(key): _validate_sha256(value, f"Stage2 base {key}")
            for key, value in identity["base_preparation_claims"].items()
        }
        successor_artifacts = _successor_provenance_artifacts(
            successor, run_root=run_root
        )
    replay_package_artifacts, replay_package_receipt, replay_claims = _package_evidence(
        replay_manifest_path,
        replay_manifest_sha,
        phase=phase,
        label="replay_manifest",
        run_id=run_id,
        runs_root=runs_root,
    )
    if successor is not None:
        identity = successor["successor_source_identity"]
        if identity.get("promotion_receipt_value") != replay_package_receipt:
            raise ArchiveError("Stage2 successor replay promotion identity differs")

    artifacts = [
        _artifact(
            marker_path,
            source_root=stage_root,
            group="results",
            relative=Path(phase) / "STAGE.json",
            kind=f"{phase} stage marker",
        ),
        _artifact(
            preflight_path,
            source_root=stage_root,
            group="results",
            relative=Path(phase) / "preflight.json",
            kind=f"{phase} preflight",
        ),
        _artifact(
            verification_path,
            source_root=stage_root,
            group="results",
            relative=Path(phase) / "verification.json",
            kind=f"{phase} verification",
        ),
        _artifact(
            telemetry_path,
            source_root=stage_root,
            group="results",
            relative=Path(phase) / "gpu_telemetry.jsonl",
            kind=f"{phase} GPU telemetry",
        ),
        _artifact(
            telemetry_summary_path,
            source_root=stage_root,
            group="results",
            relative=Path(phase) / "gpu_telemetry_summary.json",
            kind=f"{phase} GPU telemetry summary",
        ),
        _artifact(
            run_manifest_path,
            source_root=runs_root,
            group="results",
            relative=Path(phase) / "provenance/run_manifest.json",
            kind=f"{phase} run manifest",
        ),
        _artifact(
            replay_manifest_path,
            source_root=runs_root,
            group="results",
            relative=Path(phase) / "provenance/replay_manifest.json",
            kind=f"{phase} replay manifest",
        ),
        _artifact(
            completion_path,
            source_root=training_root,
            group="results",
            relative=Path(phase) / "training/COMPLETED.json",
            kind=f"{phase} training completion",
        ),
        _artifact(
            inventory_path,
            source_root=training_root,
            group="results",
            relative=Path(phase) / "training/artifact_inventory.json",
            kind=f"{phase} training inventory",
        ),
        _artifact(
            checkpoint,
            source_root=training_root,
            group="checkpoints",
            relative=Path(phase) / "navigation_rlt_policy.pt",
            kind=f"{phase} final checkpoint",
        ),
        *run_package_artifacts,
        *replay_package_artifacts,
        *successor_artifacts,
    ]
    for relative, path in sorted(inventory_files.items()):
        if relative == "checkpoints/latest/navigation_rlt_policy.pt":
            continue
        artifacts.append(
            _artifact(
                path,
                source_root=training_root,
                group="results",
                relative=Path(phase) / "training" / Path(relative),
                kind=f"{phase} training artifact",
            )
        )
    for name in ("log",):
        log_path = _regular_file(Path(str(output.get(name))), stage_root, f"{phase} log")
        artifacts.append(
            _artifact(
                log_path,
                source_root=stage_root,
                group="results",
                relative=Path(phase) / "torchrun.log",
                kind=f"{phase} torchrun log",
            )
        )
    monitor_log = _regular_file(
        stage_root / "gpu_monitor.log", stage_root, f"{phase} monitor log"
    )
    artifacts.append(
        _artifact(
            monitor_log,
            source_root=stage_root,
            group="results",
            relative=Path(phase) / "gpu_monitor.log",
            kind=f"{phase} GPU monitor log",
        )
    )

    implementation = run_manifest.get("implementation")
    marker_implementation = marker.get("implementation")
    if not isinstance(implementation, dict) or not isinstance(marker_implementation, dict):
        raise ArchiveError(f"{phase} source claims are absent")
    if successor is None:
        claims = [
            {
                str(key): _validate_sha256(value, f"{phase} run {key}")
                for key, value in implementation.items()
            },
            {
                str(key): _validate_sha256(value, f"{phase} wrapper {key}")
                for key, value in marker_implementation.items()
            },
            run_claims,
            replay_claims,
        ]
    else:
        identity = successor["successor_source_identity"]
        successor_claims = identity.get("source_claims")
        if not isinstance(successor_claims, Mapping):
            raise ArchiveError("Stage2 successor source claims are absent")
        base_claims = identity.get("base_preparation_claims")
        replay_producer_claims = identity.get("replay_producer_claims")
        if not isinstance(base_claims, Mapping) or not isinstance(
            replay_producer_claims, Mapping
        ):
            raise ArchiveError("Stage2 successor historical source claims are absent")
        claims = [
            {
                str(key): _validate_sha256(value, f"Stage2 successor {key}")
                for key, value in successor_claims.items()
            },
            {
                str(key): _validate_sha256(value, f"Stage2 base {key}")
                for key, value in base_claims.items()
            },
            {
                str(key): _validate_sha256(value, f"Stage2 replay producer {key}")
                for key, value in replay_producer_claims.items()
            },
        ]
    return StageEvidence(
        phase=phase,
        stage_root=stage_root,
        checkpoint=checkpoint,
        checkpoint_sha256=expected_checkpoint_sha,
        training_result=result,
        run_manifest=run_manifest,
        replay_manifest=replay_manifest,
        replay_summary=replay_summary,
        run_manifest_sha256=run_manifest_sha,
        replay_manifest_sha256=replay_manifest_sha,
        run_package_receipt=run_package_receipt,
        replay_package_receipt=replay_package_receipt,
        source_claims=claims,
        artifacts=artifacts,
        successor=(dict(successor) if successor is not None else None),
    )


def _cross_stage_provenance(
    stage1: StageEvidence,
    stage2: StageEvidence,
    *,
    run_root: Path,
) -> list[Artifact]:
    promotion = stage2.replay_package_receipt
    gate = promotion.get("stage1_gate")
    promoted_checkpoint = promotion.get("stage1_checkpoint")
    if (
        promotion.get("schema_name") != "navigation_rlt_replay_promotion"
        or promotion.get("schema_version") != 1
        or not isinstance(gate, dict)
        or not isinstance(promoted_checkpoint, dict)
        or gate.get("gate_passed") is not True
        or gate.get("checkpoint_sha256") != stage1.checkpoint_sha256
        or gate.get("run_manifest_sha256") != stage1.run_manifest_sha256
        or gate.get("replay_manifest_sha256") != stage1.replay_manifest_sha256
        or promoted_checkpoint.get("sha256") != stage1.checkpoint_sha256
    ):
        raise ArchiveError("Stage2 replay promotion does not bind the Stage1 checkpoint/run/replay")

    prepare_replay = stage2.run_package_receipt.get("replay")
    completion_marker = (
        prepare_replay.get("completion_marker")
        if isinstance(prepare_replay, dict)
        else None
    )
    if (
        not isinstance(completion_marker, dict)
        or completion_marker.get("stage1_checkpoint_sha256")
        != stage1.checkpoint_sha256
        or completion_marker.get("stage1_run_manifest_sha256")
        != stage1.run_manifest_sha256
        or completion_marker.get("stage1_replay_manifest_sha256")
        != stage1.replay_manifest_sha256
        or completion_marker.get("stage1_gate_marker_sha256")
        != gate.get("marker_sha256")
        or completion_marker.get("stage1_gate_receipt_sha256")
        != gate.get("receipt_sha256")
    ):
        raise ArchiveError("Stage2 prepare completion does not bind the Stage1 gate/run/replay")

    marker_path = _regular_file(
        Path(str(gate.get("marker_path"))), run_root, "Stage1 gate marker"
    )
    receipt_path = _regular_file(
        Path(str(gate.get("receipt_path"))), run_root, "Stage1 gate receipt"
    )
    marker_sha = _validate_sha256(
        gate.get("marker_sha256"), "Stage1 gate marker SHA-256"
    )
    receipt_sha = _validate_sha256(
        gate.get("receipt_sha256"), "Stage1 gate receipt SHA-256"
    )
    if sha256_file(marker_path) != marker_sha or sha256_file(receipt_path) != receipt_sha:
        raise ArchiveError("Stage1 gate marker/receipt readback differs")
    marker = _load_json(marker_path, "Stage1 gate marker")
    receipt = _load_json(receipt_path, "Stage1 gate receipt")
    checkpoint = receipt.get("checkpoint")
    split = receipt.get("split")
    if (
        marker.get("schema_name") != "navigation_rlt_stage1_gate"
        or marker.get("status") != "completed"
        or marker.get("gate_passed") is not True
        or marker.get("checkpoint_sha256") != stage1.checkpoint_sha256
        or marker.get("receipt_sha256") != receipt_sha
        or receipt.get("schema_name") != "navigation_rlt_stage1_gate"
        or receipt.get("status") != "passed"
        or not isinstance(checkpoint, dict)
        or checkpoint.get("sha256") != stage1.checkpoint_sha256
        or checkpoint.get("run_manifest_sha256") != stage1.run_manifest_sha256
        or checkpoint.get("replay_manifest_sha256") != stage1.replay_manifest_sha256
        or not isinstance(split, dict)
        or split.get("train_manifest_sha256") != stage1.replay_manifest_sha256
    ):
        raise ArchiveError("Stage1 gate receipt does not directly bind its checkpoint/run/replay")
    return [
        _artifact(
            marker_path,
            source_root=run_root,
            group="results",
            relative=Path("stage1/provenance/gate/COMPLETED.json"),
            kind="Stage1 gate marker",
        ),
        _artifact(
            receipt_path,
            source_root=run_root,
            group="results",
            relative=Path("stage1/provenance/gate/gate_receipt.json"),
            kind="Stage1 gate receipt",
        ),
    ]


def _source_inventory(
    run_root: Path, stages: Sequence[StageEvidence]
) -> tuple[list[Artifact], dict[str, object]]:
    code_root = _regular_directory(run_root / "code", run_root, "source code root")
    artifacts: list[Artifact] = []
    hashes: dict[str, str] = {}
    total = 0
    for claim_name, filename in SOURCE_FILES.items():
        path = _regular_file(code_root / filename, code_root, f"source {filename}")
        size = path.stat().st_size
        total += size
        if total > MAX_SOURCE_BYTES:
            raise ArchiveError("source inventory exceeds the 64 MiB hard cap")
        digest = sha256_file(path)
        hashes[claim_name] = digest
        artifacts.append(
            Artifact(
                source=path,
                group="results",
                relative=Path("source") / filename,
                kind="source",
                sha256=digest,
                size_bytes=size,
            )
        )
    known_claims = set(SOURCE_FILES)
    for stage in stages:
        for claims in stage.source_claims:
            unknown = set(claims) - known_claims
            if unknown:
                raise ArchiveError(f"unknown source hash claims: {sorted(unknown)}")
            for name, expected in claims.items():
                if hashes[name] != expected:
                    raise ArchiveError(f"source {name} differs from recorded provenance")
    inventory = {
        "schema_name": "navigation_rlt_training_source_inventory",
        "schema_version": 1,
        "source_root": str(code_root),
        "total_bytes": total,
        "files": {
            SOURCE_FILES[name]: {"sha256": digest, "size_bytes": (code_root / SOURCE_FILES[name]).stat().st_size}
            for name, digest in sorted(hashes.items())
        },
    }
    return artifacts, inventory


def _external_regular_file(path: Path, name: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink():
        raise ArchiveError(f"{name} must not be a symbolic link")
    try:
        resolved = raw.resolve(strict=True)
    except OSError as error:
        raise ArchiveError(f"{name} is missing") from error
    if not resolved.is_file():
        raise ArchiveError(f"{name} must be a regular file")
    return resolved


def _evaluation_source(
    code_root: Path,
    relative: Path,
    name: str,
    *,
    allow_repository_fallback: bool = True,
) -> tuple[Path, Path]:
    candidates = [(code_root, code_root / relative)]
    if (
        allow_repository_fallback
        and relative.parts
        and relative.parts[0] == "navrl"
    ):
        repository_root = code_root.parents[1]
        candidates.append((repository_root, repository_root / relative))
    regular: list[tuple[Path, Path]] = []
    for source_root, candidate in candidates:
        if candidate.is_symlink():
            continue
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        if resolved.is_file():
            regular.append((resolved, source_root.resolve(strict=True)))
    if len(regular) != 1:
        raise ArchiveError(f"{name} is missing or ambiguous")
    return regular[0]


def _evaluation_code_evidence(
    inventory_path: Path,
    expected_sha256: str,
    *,
    source_root: Path | None = None,
) -> tuple[dict[str, object], list[Artifact]]:
    path = _external_regular_file(inventory_path, "evaluation code inventory")
    expected = _validate_sha256(
        expected_sha256, "expected evaluation code inventory SHA-256"
    )
    if sha256_file(path) != expected:
        raise ArchiveError("evaluation code inventory SHA-256 differs")
    inventory = _load_json(path, "evaluation code inventory")
    if (
        set(inventory)
        != {"schema_name", "schema_version", "navigation_code_commit", "files"}
        or inventory.get("schema_name") != EVALUATION_INVENTORY_SCHEMA
        or inventory.get("schema_version") != 1
    ):
        raise ArchiveError("evaluation code inventory schema differs")
    commit = _validate_sha1(
        inventory.get("navigation_code_commit"), "evaluation code commit"
    )
    files = inventory.get("files")
    if not isinstance(files, dict) or set(files) != set(EVALUATION_CODE_FILES):
        raise ArchiveError("evaluation code inventory dependency set differs")
    explicit_source_root = source_root is not None
    if source_root is None:
        code_root = Path(__file__).resolve(strict=True).parent
    else:
        raw_source_root = Path(source_root)
        if raw_source_root.is_symlink():
            raise ArchiveError("evaluation source root must not be a symbolic link")
        try:
            code_root = raw_source_root.resolve(strict=True)
        except OSError as error:
            raise ArchiveError("evaluation source root is missing") from error
        if not code_root.is_dir():
            raise ArchiveError("evaluation source root must be a directory")
    normalized: dict[str, str] = {}
    artifacts = [
        _artifact(
            path,
            source_root=path.parent,
            group="results",
            relative=Path("provenance/evaluation/code_inventory.json"),
            kind="evaluation code inventory",
        )
    ]
    for filename in EVALUATION_CODE_FILES:
        raw_sha = files[filename]
        relative = _safe_relative(filename, "evaluation code inventory path")
        if relative.as_posix() != filename:
            raise ArchiveError("evaluation code inventory path is not canonical")
        digest = _validate_sha256(raw_sha, f"evaluation source {filename}")
        source, source_root = _evaluation_source(
            code_root,
            relative,
            f"evaluation source {filename}",
            allow_repository_fallback=not explicit_source_root,
        )
        if sha256_file(source) != digest:
            raise ArchiveError(f"evaluation source {filename} SHA-256 differs")
        if explicit_source_root and os.name != "nt" and source.stat().st_mode & 0o222:
            raise ArchiveError(
                f"explicit evaluation source {filename} must be read-only"
            )
        normalized[filename] = digest
        artifacts.append(
            _artifact(
                source,
                source_root=source_root,
                group="results",
                relative=Path("provenance/evaluation/source") / relative,
                kind=f"evaluation source {filename}",
            )
        )
    return {
        "commit": commit,
        "inventory_sha256": expected,
        **({"source_root": str(code_root)} if explicit_source_root else {}),
        "files": normalized,
    }, artifacts


def _exact_mapping(
    value: object, expected_fields: Sequence[str] | set[str], name: str
) -> Mapping[str, object]:
    if not isinstance(value, dict) or set(value) != set(expected_fields):
        raise ArchiveError(f"{name} fields differ")
    return value


def _stage2_training_additions(
    value: object, *, name: str
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ArchiveError(f"{name} fields differ")
    schema = value.get("objective_schema")
    fields = (
        STAGE2_BEHAVIOR_KL_TRAINING_FIELDS
        if schema == STAGE2_BEHAVIOR_KL_OBJECTIVE_SCHEMA
        else STAGE2_REPAIR_TRAINING_FIELDS
    )
    additions = _exact_mapping(value, fields, name)
    alpha = additions.get("cql_alpha")
    temperature = additions.get("cql_temperature")
    minimum = additions.get("minimum_behavior_gate_count")
    if schema not in (
        STAGE2_REPAIR_OBJECTIVE_SCHEMA,
        STAGE2_BEHAVIOR_KL_OBJECTIVE_SCHEMA,
    ):
        raise ArchiveError("Stage2 repair objective schema differs")
    if schema == STAGE2_BEHAVIOR_KL_OBJECTIVE_SCHEMA:
        weight = additions.get("behavior_kl_weight")
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(float(weight))
            or float(weight) != 1.0
        ):
            raise ArchiveError("Stage2 behavior KL weight differs")
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not math.isfinite(float(alpha))
        or float(alpha) != STAGE2_CQL_ALPHA
    ):
        raise ArchiveError("Stage2 repair cql_alpha differs")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(float(temperature))
        or float(temperature) != STAGE2_CQL_TEMPERATURE
    ):
        raise ArchiveError("Stage2 repair cql_temperature differs")
    if type(minimum) is not int or minimum != STAGE2_MINIMUM_BEHAVIOR_GATE_COUNT:
        raise ArchiveError("Stage2 repair minimum behavior gate count differs")
    return dict(additions)


def _stage2_support_contract(value: object, *, name: str) -> dict[str, object]:
    support = _exact_mapping(value, STAGE2_SUPPORT_FIELDS, name)
    if support.get("eligibility_schema") != STAGE2_ELIGIBILITY_SCHEMA:
        raise ArchiveError("Stage2 repair eligibility schema differs")
    if support.get("gate_names") != list(STAGE2_GATE_NAMES):
        raise ArchiveError("Stage2 repair gate names differ")
    minimum = support.get("minimum_behavior_gate_count")
    if type(minimum) is not int or minimum != STAGE2_MINIMUM_BEHAVIOR_GATE_COUNT:
        raise ArchiveError("Stage2 repair support minimum gate count differs")
    counts = support.get("gate_counts")
    mask = support.get("gate_support_mask")
    if (
        not isinstance(counts, list)
        or len(counts) != len(STAGE2_GATE_NAMES)
        or any(type(count) is not int or count < 0 for count in counts)
    ):
        raise ArchiveError("Stage2 repair gate counts are invalid")
    if (
        not isinstance(mask, list)
        or len(mask) != len(STAGE2_GATE_NAMES)
        or any(type(enabled) is not bool for enabled in mask)
    ):
        raise ArchiveError("Stage2 repair gate support mask is invalid")
    expected_mask = [count >= minimum for count in counts]
    if mask != expected_mask or not any(mask):
        raise ArchiveError("Stage2 repair gate support mask differs from counts")
    if mask[STAGE2_GATE_NAMES.index("RECOVER")] is not False:
        raise ArchiveError("Stage2 repair requires RECOVER support to be false")
    return dict(support)


def _stage_protocol_contract(
    stage: StageEvidence,
    protocol: Mapping[str, object],
    *,
    phase: str,
    fields: Sequence[str],
    expected_episode_count: int,
    run_id: str,
    training_additions: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    declared = _exact_mapping(protocol, fields, f"{phase} training protocol")
    if (
        stage.phase != phase
        or stage.run_manifest.get("run_id") != run_id
        or stage.run_manifest.get("phase") != phase
        or declared.get("expected_episode_count") != expected_episode_count
        or stage.run_manifest.get("seed") != declared.get("seed")
        or stage.run_manifest.get("required_world_size")
        != declared.get("required_world_size")
        or stage.training_result.get("world_size")
        != declared.get("required_world_size")
        or stage.replay_manifest.get("expected_episode_count")
        != expected_episode_count
        or stage.replay_manifest.get("completed_episode_count")
        != expected_episode_count
    ):
        raise ArchiveError(f"{phase} seed/world-size/episode protocol differs")
    training_fields = tuple(
        field
        for field in fields
        if field not in {"seed", "required_world_size", "expected_episode_count"}
    )
    expected_training = {field: declared[field] for field in training_fields}
    if training_additions is not None:
        overlap = set(expected_training).intersection(training_additions)
        if overlap:
            raise ArchiveError(
                f"{phase} repair training fields overlap the base protocol"
            )
        expected_training.update(training_additions)
    if (
        stage.run_manifest.get("training") != expected_training
        or stage.training_result.get("training") != expected_training
    ):
        raise ArchiveError(f"{phase} actual training parameters differ from protocol")
    return declared


def _training_protocol_contract(
    *,
    protocol: Mapping[str, object],
    collection: Mapping[str, object],
    collection_sha256: str,
    run_root: Path,
    run_id: str,
    stages: Sequence[StageEvidence],
    stage2_training_additions: Mapping[str, object] | None = None,
) -> None:
    if set(protocol) != TRAINING_PROTOCOL_FIELDS:
        raise ArchiveError("training protocol top-level fields differ")
    _validate_sha1(protocol.get("preregistration_commit"), "preregistration commit")
    if protocol.get("boundaries") != PROTOCOL_BOUNDARIES:
        raise ArchiveError("training protocol boundaries differ")
    if protocol.get("stage1_checkpoint_provenance_gate") != PROVENANCE_GATE_PROTOCOL:
        raise ArchiveError("training protocol Stage1 provenance gate differs")
    if len(stages) != 2 or [stage.phase for stage in stages] != ["stage1", "stage2"]:
        raise ArchiveError("training protocol requires ordered Stage1 and Stage2 evidence")
    stage1, stage2 = stages
    stage1_protocol = _stage_protocol_contract(
        stage1,
        _exact_mapping(
            protocol.get("stage1"), STAGE1_PROTOCOL_FIELDS, "Stage1 training protocol"
        ),
        phase="stage1",
        fields=STAGE1_PROTOCOL_FIELDS,
        expected_episode_count=224,
        run_id=run_id,
    )
    stage2_protocol = _stage_protocol_contract(
        stage2,
        _exact_mapping(
            protocol.get("stage2"), STAGE2_PROTOCOL_FIELDS, "Stage2 training protocol"
        ),
        phase="stage2",
        fields=STAGE2_PROTOCOL_FIELDS,
        expected_episode_count=256,
        run_id=run_id,
        training_additions=stage2_training_additions,
    )
    if (
        stage1_protocol["required_world_size"] != 8
        or stage2_protocol["required_world_size"] != 8
    ):
        raise ArchiveError("training protocol requires exactly eight ranks")

    protocol_model = _exact_mapping(
        protocol.get("model"), MODEL_PROTOCOL_FIELDS, "training protocol model"
    )
    stage1_model = stage1.run_manifest.get("model")
    stage2_model = stage2.run_manifest.get("model")
    if not isinstance(stage1_model, dict) or stage1_model != stage2_model:
        raise ArchiveError("Stage1 and Stage2 model contracts differ")
    stage1_config = stage1_model.get("config")
    if not isinstance(stage1_config, dict):
        raise ArchiveError("training run model config is absent")
    if {field: stage1_config.get(field) for field in MODEL_PROTOCOL_FIELDS} != dict(
        protocol_model
    ):
        raise ArchiveError("actual model config differs from training protocol")

    rollout = _exact_mapping(
        protocol.get("rollout"),
        (
            "behavior",
            "episode_count",
            "outer_manifest_sha256",
            "parallel_plan_sha256",
            "split",
        ),
        "training protocol rollout",
    )
    collection_boundaries = collection.get("boundaries")
    collection_data = collection.get("data")
    collection_behavior = collection.get("behavior")
    collection_rlinf = collection.get("rlinf")
    if (
        not isinstance(collection_boundaries, dict)
        or collection_boundaries.get("upstream_split_read") != "train"
        or collection_boundaries.get("validation_split_reads") != 0
        or collection_boundaries.get("lightnav_frozen") is not True
        or collection_boundaries.get("ttt_or_secondary_post_training") is not False
        or collection_boundaries.get("remote_push") is not False
        or not isinstance(collection_data, dict)
        or collection_data.get("update_episode_count") != 256
        or collection_data.get("development_episode_count") != 32
        or collection_data.get("worker_count") != 8
        or collection_data.get("episodes_per_worker") != 32
        or not isinstance(collection_behavior, dict)
        or collection_behavior.get("name") != "a3_stochastic"
        or not isinstance(collection_rlinf, dict)
        or collection_rlinf.get("required_world_size") != 8
    ):
        raise ArchiveError("collection outer manifest training contract differs")
    if dict(rollout) != {
        "behavior": "a3",
        "episode_count": collection_data["update_episode_count"],
        "outer_manifest_sha256": collection_sha256,
        "parallel_plan_sha256": collection_data.get("parallel_plan_sha256"),
        "split": collection_boundaries["upstream_split_read"],
    }:
        raise ArchiveError("training protocol rollout differs from collection manifest")

    gate_protocol = _exact_mapping(
        protocol.get("heldout_gate"),
        HELDOUT_GATE_PROTOCOL_FIELDS,
        "heldout gate protocol",
    )
    gate_reference = stage2.replay_package_receipt.get("stage1_gate")
    if not isinstance(gate_reference, dict):
        raise ArchiveError("Stage2 replay promotion lacks Stage1 gate evidence")
    gate_path = _regular_file(
        Path(str(gate_reference.get("receipt_path"))),
        run_root,
        "Stage1 heldout gate receipt",
    )
    gate_sha = _validate_sha256(
        gate_reference.get("receipt_sha256"), "Stage1 heldout gate receipt SHA-256"
    )
    if sha256_file(gate_path) != gate_sha:
        raise ArchiveError("Stage1 heldout gate receipt SHA-256 differs")
    gate = _load_json(gate_path, "Stage1 heldout gate receipt")
    gate_split = gate.get("split")
    gate_checkpoint = gate.get("checkpoint")
    gate_evaluation = gate.get("evaluation_contract")
    if (
        gate.get("schema_name") != "navigation_rlt_stage1_gate"
        or gate.get("schema_version") != 1
        or gate.get("status") != "passed"
        or gate.get("operation") != "evaluate"
        or not isinstance(gate_split, dict)
        or gate_split.get("train_episode_count") != 224
        or gate_split.get("dev_episode_count") != 32
        or gate_split.get("train_manifest_sha256")
        != stage1.replay_manifest_sha256
        or not isinstance(gate_checkpoint, dict)
        or gate_checkpoint.get("sha256") != stage1.checkpoint_sha256
        or gate_checkpoint.get("run_manifest_sha256") != stage1.run_manifest_sha256
        or gate_checkpoint.get("replay_manifest_sha256")
        != stage1.replay_manifest_sha256
        or gate_checkpoint.get("config") != stage1_config
        or not isinstance(gate_evaluation, dict)
        or {field: gate_evaluation.get(field) for field in HELDOUT_GATE_PROTOCOL_FIELDS}
        != dict(gate_protocol)
    ):
        raise ArchiveError("heldout gate receipt differs from training protocol")


def _verified_training_code_bundle(
    *,
    run_root: Path,
    verification_path: Path,
    expected_verification_sha256: str,
    expected_superseded_commit: str,
    stages: Sequence[StageEvidence],
    label: str,
    artifact_prefix: Path,
    expected_code_root: Path | None = None,
    forbidden_code_root: Path | None = None,
    source_inventory: Mapping[str, object] | None = None,
    required_inventory_files: Sequence[str] = (),
    required_source_claims: Sequence[str] = (),
) -> tuple[dict[str, object], list[Artifact]]:
    checked_verification_path = _regular_file(
        verification_path, run_root, f"{label} code verification"
    )
    verification_sha = _validate_sha256(
        expected_verification_sha256,
        f"expected {label} code verification SHA-256",
    )
    if sha256_file(checked_verification_path) != verification_sha:
        raise ArchiveError(f"{label} code verification SHA-256 differs")
    verification = _load_json(
        checked_verification_path, f"{label} code verification"
    )
    if (
        verification.get("schema_name") != TRAINING_CODE_VERIFICATION_SCHEMA
        or verification.get("schema_version") != 1
        or verification.get("status") != "passed"
    ):
        raise ArchiveError(f"{label} code verification schema/status differs")
    code_commit = _validate_sha1(
        verification.get("navigation_code_commit"), f"{label} code commit"
    )
    superseded = _validate_sha1(
        verification.get("supersedes_navigation_code_commit"),
        f"{label} superseded code commit",
    )
    if superseded != expected_superseded_commit:
        raise ArchiveError(f"{label} code verification superseded commit differs")
    code_root = _regular_directory(
        Path(str(verification.get("code_root"))), run_root, f"{label} code root"
    )
    if expected_code_root is not None:
        checked_expected_root = _regular_directory(
            expected_code_root, run_root, f"expected {label} code root"
        )
        if code_root != checked_expected_root:
            raise ArchiveError(f"{label} code verification code root differs")
    if forbidden_code_root is not None:
        checked_forbidden_root = _regular_directory(
            forbidden_code_root, run_root, f"forbidden {label} code root"
        )
        roots_overlap = code_root == checked_forbidden_root
        for candidate, parent in (
            (code_root, checked_forbidden_root),
            (checked_forbidden_root, code_root),
        ):
            try:
                candidate.relative_to(parent)
            except ValueError:
                continue
            roots_overlap = True
        if roots_overlap:
            raise ArchiveError(f"{label} code root is not independent")

    inventory_path = _regular_file(
        Path(str(verification.get("inventory_path"))),
        run_root,
        f"{label} code inventory",
    )
    inventory_sha = _validate_sha256(
        verification.get("inventory_sha256"), f"{label} code inventory SHA-256"
    )
    if sha256_file(inventory_path) != inventory_sha:
        raise ArchiveError(f"{label} code inventory SHA-256 differs")
    inventory = _load_json(inventory_path, f"{label} code inventory")
    raw_files = inventory.get("files")
    if (
        inventory.get("schema_name") != TRAINING_INVENTORY_SCHEMA
        or inventory.get("schema_version") != 1
        or inventory.get("navigation_code_commit") != code_commit
        or not isinstance(raw_files, list)
        or inventory.get("file_count") != len(raw_files)
        or verification.get("file_count") != len(raw_files)
    ):
        raise ArchiveError(f"{label} code inventory identity/count differs")
    indexed: dict[str, dict[str, object]] = {}
    for raw in raw_files:
        if not isinstance(raw, dict):
            raise ArchiveError(f"{label} code inventory row is invalid")
        relative = _safe_relative(
            raw.get("path"), f"{label} code inventory path"
        ).as_posix()
        if relative in indexed:
            raise ArchiveError(f"{label} code inventory path repeats")
        digest = _validate_sha256(raw.get("sha256"), f"{label} source {relative}")
        size = raw.get("size_bytes")
        if type(size) is not int or size < 0:
            raise ArchiveError(f"{label} code inventory size is invalid")
        indexed[relative] = {"sha256": digest, "size_bytes": size}

    required_files = set(required_inventory_files)
    missing_files = required_files - set(indexed)
    if missing_files:
        raise ArchiveError(
            f"{label} code inventory lacks required sources: {sorted(missing_files)}"
        )
    for relative in sorted(required_files):
        live_source = _regular_file(
            code_root / relative, code_root, f"{label} live source {relative}"
        )
        if (
            sha256_file(live_source) != indexed[relative]["sha256"]
            or live_source.stat().st_size != indexed[relative]["size_bytes"]
        ):
            raise ArchiveError(f"{label} live source differs from code inventory")

    archive_path = _regular_file(
        Path(str(verification.get("archive_path"))),
        run_root,
        f"{label} code archive",
    )
    archive_sha = _validate_sha256(
        verification.get("archive_sha256"), f"{label} code archive SHA-256"
    )
    if (
        sha256_file(archive_path) != archive_sha
        or inventory.get("archive_name") != archive_path.name
        or inventory.get("archive_sha256") != archive_sha
        or inventory.get("archive_size_bytes") != archive_path.stat().st_size
    ):
        raise ArchiveError(f"{label} code archive/inventory identity differs")

    if source_inventory is not None:
        source_files = source_inventory.get("files")
        if not isinstance(source_files, dict):
            raise ArchiveError(f"{label} source inventory files are absent")
        for filename, evidence in source_files.items():
            if not isinstance(evidence, dict) or filename not in indexed:
                raise ArchiveError(
                    f"{label} source is absent from verified code inventory"
                )
            if (
                evidence.get("sha256") != indexed[filename]["sha256"]
                or evidence.get("size_bytes") != indexed[filename]["size_bytes"]
            ):
                raise ArchiveError(
                    f"{label} source differs from verified code inventory"
                )
    observed_claims: set[str] = set()
    for stage in stages:
        for claims in stage.source_claims:
            for claim_name, digest in claims.items():
                observed_claims.add(claim_name)
                filename = SOURCE_FILES.get(claim_name)
                if filename is None or filename not in indexed:
                    raise ArchiveError(
                        f"{label} stage source claim is not inventoried"
                    )
                if indexed[filename]["sha256"] != digest:
                    raise ArchiveError(
                        f"{label} stage source claim differs from code inventory"
                    )
    missing_claims = set(required_source_claims) - observed_claims
    if missing_claims:
        raise ArchiveError(
            f"{label} stage provenance lacks required source claims: "
            f"{sorted(missing_claims)}"
        )

    artifacts = [
        _artifact(
            checked_verification_path,
            source_root=run_root,
            group="results",
            relative=artifact_prefix / "code_verification.json",
            kind=f"{label} code verification",
        ),
        _artifact(
            inventory_path,
            source_root=run_root,
            group="results",
            relative=artifact_prefix / "code_inventory.json",
            kind=f"{label} code inventory",
        ),
        _artifact(
            archive_path,
            source_root=run_root,
            group="results",
            relative=artifact_prefix / archive_path.name,
            kind=f"{label} code archive",
        ),
    ]
    return {
        "code_commit": code_commit,
        "code_root": str(code_root),
        "code_verification_sha256": verification_sha,
        "code_inventory_sha256": inventory_sha,
        "code_archive_sha256": archive_sha,
    }, artifacts


def _training_provenance_evidence(
    *,
    run_root: Path,
    run_id: str,
    collection_manifest_path: Path,
    expected_collection_sha256: str,
    code_verification_path: Path,
    expected_code_verification_sha256: str,
    training_protocol_path: Path,
    expected_training_protocol_sha256: str,
    stages: Sequence[StageEvidence],
    training_source_inventory: Mapping[str, object],
    stage2_repair_code_verification_path: Path | None = None,
    expected_stage2_repair_code_verification_sha256: str | None = None,
    stage2_repair_protocol_path: Path | None = None,
    expected_stage2_repair_protocol_sha256: str | None = None,
) -> tuple[dict[str, object], list[Artifact]]:
    if len(stages) != 2 or [stage.phase for stage in stages] != ["stage1", "stage2"]:
        raise ArchiveError("training provenance requires ordered Stage1 and Stage2 evidence")
    stage1, stage2 = stages
    collection_path = _regular_file(
        collection_manifest_path, run_root, "collection outer manifest"
    )
    collection_sha = _validate_sha256(
        expected_collection_sha256, "expected collection outer manifest SHA-256"
    )
    if sha256_file(collection_path) != collection_sha:
        raise ArchiveError("collection outer manifest SHA-256 differs")
    collection = _load_json(collection_path, "collection outer manifest")
    if (
        collection.get("schema_name") != COLLECTION_MANIFEST_SCHEMA
        or collection.get("schema_version") != 1
        or collection.get("run_id") != run_id
    ):
        raise ArchiveError("collection outer manifest schema/run_id differs")
    collection_code = collection.get("code")
    if not isinstance(collection_code, dict):
        raise ArchiveError("collection outer manifest code identity is absent")
    collection_commit = _validate_sha1(
        collection_code.get("git_commit"), "collection navigation code commit"
    )
    collection_files = collection_code.get("files")
    if not isinstance(collection_files, dict) or not collection_files:
        raise ArchiveError("collection outer manifest code file hashes are absent")
    for filename, digest in collection_files.items():
        _safe_relative(filename, "collection code file")
        _validate_sha256(digest, f"collection source {filename}")
    collection_data = collection.get("data")
    if not isinstance(collection_data, dict):
        raise ArchiveError("collection outer manifest data contract is absent")
    collection_split_sha = _validate_sha256(
        collection_data.get("split_manifest_sha256"),
        "collection train split manifest SHA-256",
    )

    stage1_identity, stage1_code_artifacts = _verified_training_code_bundle(
        run_root=run_root,
        verification_path=code_verification_path,
        expected_verification_sha256=expected_code_verification_sha256,
        expected_superseded_commit=collection_commit,
        stages=[stage1],
        label="Stage1 training",
        artifact_prefix=Path("provenance/training"),
        expected_code_root=run_root / "code",
        source_inventory=training_source_inventory,
    )
    training_commit = str(stage1_identity["code_commit"])

    protocol_path = _regular_file(
        training_protocol_path, run_root, "training protocol"
    )
    protocol_sha = _validate_sha256(
        expected_training_protocol_sha256, "expected training protocol SHA-256"
    )
    if sha256_file(protocol_path) != protocol_sha:
        raise ArchiveError("training protocol SHA-256 differs")
    protocol = _load_json(protocol_path, "training protocol")
    rollout = protocol.get("rollout")
    if (
        protocol.get("schema_name") != TRAINING_PROTOCOL_SCHEMA
        or protocol.get("schema_version") != 1
        or protocol.get("run_id") != run_id
        or protocol.get("navigation_training_code_commit") != training_commit
        or not isinstance(rollout, dict)
        or rollout.get("outer_manifest_sha256") != collection_sha
    ):
        raise ArchiveError("training protocol code/collection identity differs")

    repair_values = (
        stage2_repair_code_verification_path,
        expected_stage2_repair_code_verification_sha256,
        stage2_repair_protocol_path,
        expected_stage2_repair_protocol_sha256,
    )
    repair_supplied = all(value is not None for value in repair_values)
    if any(value is not None for value in repair_values) and not repair_supplied:
        raise ArchiveError("Stage2 repair provenance arguments must be all present")
    stage2_training = stage2.run_manifest.get("training")
    successor_identity = (
        stage2.successor.get("successor_source_identity")
        if isinstance(stage2.successor, Mapping)
        else None
    )
    if stage2.successor is not None and not isinstance(successor_identity, Mapping):
        raise ArchiveError("Stage2 successor source identity is absent")
    objective_schema = (
        stage2_training.get("objective_schema")
        if isinstance(stage2_training, dict)
        else None
    )
    actual_repair = (
        isinstance(stage2_training, dict)
        and objective_schema
        in (
            STAGE2_REPAIR_OBJECTIVE_SCHEMA,
            STAGE2_BEHAVIOR_KL_OBJECTIVE_SCHEMA,
        )
    )
    if (successor_identity is not None) != (
        objective_schema == STAGE2_BEHAVIOR_KL_OBJECTIVE_SCHEMA
    ):
        raise ArchiveError("Stage2 successor identity/objective differs")
    if actual_repair != repair_supplied:
        raise ArchiveError("Stage2 repair provenance does not match the actual objective")

    repair_identity: dict[str, object] | None = None
    repair_protocol_sha: str | None = None
    repair_artifacts: list[Artifact] = []
    stage2_additions: dict[str, object] | None = None
    if repair_supplied:
        assert stage2_repair_code_verification_path is not None
        assert expected_stage2_repair_code_verification_sha256 is not None
        assert stage2_repair_protocol_path is not None
        assert expected_stage2_repair_protocol_sha256 is not None
        repair_identity, repair_code_artifacts = _verified_training_code_bundle(
            run_root=run_root,
            verification_path=stage2_repair_code_verification_path,
            expected_verification_sha256=(
                expected_stage2_repair_code_verification_sha256
            ),
            expected_superseded_commit=training_commit,
            stages=[] if successor_identity is not None else [stage2],
            label="Stage2 repair training",
            artifact_prefix=Path("provenance/training/stage2-repair"),
            forbidden_code_root=Path(str(stage1_identity["code_root"])),
            required_inventory_files=tuple(SOURCE_FILES.values()),
            required_source_claims=(
                ()
                if successor_identity is not None
                else tuple(STAGE2_REPAIR_REQUIRED_SOURCE_CLAIMS)
            ),
        )
        if repair_identity["code_commit"] == training_commit:
            raise ArchiveError("Stage2 repair code commit must differ from Stage1")
        historical_source_claims: dict[str, str] | None = None
        base_run_manifest_sha = stage2.run_manifest_sha256
        if successor_identity is not None:
            base_claims = successor_identity.get("base_preparation_claims")
            replay_claims = successor_identity.get("replay_producer_claims")
            base_manifest_ref = successor_identity.get("base_run_manifest")
            if (
                not isinstance(base_claims, Mapping)
                or not isinstance(replay_claims, Mapping)
                or not isinstance(base_manifest_ref, Mapping)
            ):
                raise ArchiveError("Stage2 successor base lineage is incomplete")
            historical_source_claims = {}
            for group in (base_claims, replay_claims):
                for claim, raw_digest in group.items():
                    digest = _validate_sha256(
                        raw_digest, f"Stage2 historical {claim}"
                    )
                    previous = historical_source_claims.setdefault(str(claim), digest)
                    if previous != digest:
                        raise ArchiveError(
                            f"Stage2 historical source claim {claim} differs"
                        )
            repair_code_root = _regular_directory(
                Path(str(repair_identity["code_root"])),
                run_root,
                "Stage2 base repair code root",
            )
            for claim in ("wrapper_sha256", "gpu_monitor_sha256"):
                source = _regular_file(
                    repair_code_root / SOURCE_FILES[claim],
                    repair_code_root,
                    f"Stage2 historical {claim}",
                )
                historical_source_claims[claim] = sha256_file(source)
            if set(historical_source_claims) != STAGE2_REPAIR_REQUIRED_SOURCE_CLAIMS:
                raise ArchiveError("Stage2 historical source claim coverage differs")
            for claim, digest in historical_source_claims.items():
                source = _regular_file(
                    repair_code_root / SOURCE_FILES[claim],
                    repair_code_root,
                    f"Stage2 historical {claim}",
                )
                if sha256_file(source) != digest:
                    raise ArchiveError(
                        f"Stage2 historical source {claim} differs from code inventory"
                    )
            base_run_manifest_sha = _validate_sha256(
                base_manifest_ref.get("sha256"),
                "Stage2 successor base run manifest SHA-256",
            )
        checked_repair_protocol_path = _regular_file(
            stage2_repair_protocol_path, run_root, "Stage2 repair protocol"
        )
        repair_protocol_sha = _validate_sha256(
            expected_stage2_repair_protocol_sha256,
            "expected Stage2 repair protocol SHA-256",
        )
        if sha256_file(checked_repair_protocol_path) != repair_protocol_sha:
            raise ArchiveError("Stage2 repair protocol SHA-256 differs")
        repair_protocol = _load_json(
            checked_repair_protocol_path, "Stage2 repair protocol"
        )
        _exact_mapping(
            repair_protocol,
            STAGE2_REPAIR_PROTOCOL_FIELDS,
            "Stage2 repair protocol",
        )
        if (
            repair_protocol.get("schema_name") != STAGE2_REPAIR_PROTOCOL_SCHEMA
            or repair_protocol.get("schema_version") != 1
            or repair_protocol.get("run_id") != run_id
            or repair_protocol.get("base_training_protocol_sha256") != protocol_sha
            or repair_protocol.get("stage1_code_verification_sha256")
            != stage1_identity["code_verification_sha256"]
            or repair_protocol.get("stage1_checkpoint_sha256")
            != stage1.checkpoint_sha256
            or repair_protocol.get("stage2_repair_code_verification_sha256")
            != repair_identity["code_verification_sha256"]
            or repair_protocol.get("stage2_replay_manifest_sha256")
            != stage2.replay_manifest_sha256
            or repair_protocol.get("stage2_run_manifest_sha256")
            != base_run_manifest_sha
            or repair_protocol.get("navigation_training_code_commit")
            != repair_identity["code_commit"]
            or type(repair_protocol.get("validation_split_reads")) is not int
            or repair_protocol.get("validation_split_reads") != 0
        ):
            raise ArchiveError("Stage2 repair protocol source/boundary identity differs")
        protocol_additions = _stage2_training_additions(
            repair_protocol.get("stage2_training_additions"),
            name="Stage2 repair training additions",
        )
        if protocol_additions["objective_schema"] != STAGE2_REPAIR_OBJECTIVE_SCHEMA:
            raise ArchiveError("Stage2 base repair objective schema differs")
        stage2_additions = (
            _stage2_training_additions(
                {
                    field: stage2_training.get(field)
                    for field in STAGE2_BEHAVIOR_KL_TRAINING_FIELDS
                },
                name="Stage2 successor training additions",
            )
            if successor_identity is not None
            else protocol_additions
        )
        if successor_identity is not None:
            repair_contract = stage2.successor.get("repair_contract")
            if (
                not isinstance(repair_contract, Mapping)
                or repair_contract.get("training_additions") != stage2_additions
            ):
                raise ArchiveError("Stage2 successor repair contract differs")
        support = _stage2_support_contract(
            repair_protocol.get("stage2_support"),
            name="Stage2 repair support",
        )
        if (
            protocol_additions["minimum_behavior_gate_count"]
            != support["minimum_behavior_gate_count"]
            or stage2.run_manifest.get("stage2_support") != support
            or stage2.training_result.get("stage2_support") != support
        ):
            raise ArchiveError("Stage2 repair support differs from actual training")
        replay_validation_reads = stage2.replay_manifest.get(
            "validation_split_reads"
        )
        if replay_validation_reads is not None and (
            type(replay_validation_reads) is not int
            or replay_validation_reads != 0
        ):
            raise ArchiveError("Stage2 repair replay reports validation split reads")
        for receipt in stage2.training_result.get("rank_receipts", []):
            data = receipt.get("data") if isinstance(receipt, dict) else None
            if not isinstance(data, dict) or any(
                data.get(field) != support[field] for field in STAGE2_SUPPORT_FIELDS
            ):
                raise ArchiveError("Stage2 rank support evidence differs")
        final_checkpoint = stage2.training_result.get("final_checkpoint")
        metadata = (
            final_checkpoint.get("metadata")
            if isinstance(final_checkpoint, dict)
            else None
        )
        expected_supported_names = [
            name
            for name, enabled in zip(
                STAGE2_GATE_NAMES, support["gate_support_mask"], strict=True
            )
            if enabled
        ]
        if (
            not isinstance(metadata, dict)
            or metadata.get("stage2_objective")
            != (
                "reference_conditioned_min_target_twin_q_with_finite_gate_cql_"
                "and_native_reference_actor_anchor_dropout"
            )
            or metadata.get("cql_enabled") is not True
            or metadata.get("cql_candidate_count") != len(STAGE2_GATE_NAMES)
            or metadata.get("entropy_target_enabled") is not False
            or metadata.get("gate_support_mask") != support["gate_support_mask"]
            or metadata.get("supported_gate_names") != expected_supported_names
        ):
            raise ArchiveError("Stage2 checkpoint repair metadata differs")
        repair_artifacts = [
            *repair_code_artifacts,
            _artifact(
                checked_repair_protocol_path,
                source_root=run_root,
                group="results",
                relative=Path(
                    "provenance/training/stage2-repair/stage2_repair_protocol.json"
                ),
                kind="Stage2 repair protocol",
            ),
        ]

    _training_protocol_contract(
        protocol=protocol,
        collection=collection,
        collection_sha256=collection_sha,
        run_root=run_root,
        run_id=run_id,
        stages=stages,
        stage2_training_additions=stage2_additions,
    )

    artifacts = [
        _artifact(
            collection_path,
            source_root=run_root,
            group="results",
            relative=Path("provenance/collection/run_manifest.json"),
            kind="collection outer manifest",
        ),
        *stage1_code_artifacts[:1],
        _artifact(
            protocol_path,
            source_root=run_root,
            group="results",
            relative=Path("provenance/training/training_protocol.json"),
            kind="training protocol",
        ),
        *stage1_code_artifacts[1:],
        *repair_artifacts,
    ]
    source_identities: dict[str, object] = {
        "stage1": {
            **stage1_identity,
            "training_protocol_sha256": protocol_sha,
        }
    }
    result: dict[str, object] = {
        "collection_manifest_sha256": collection_sha,
        "collection_split_manifest_sha256": collection_split_sha,
        "collection_code_commit": collection_commit,
        "training_code_commit": training_commit,
        "training_code_verification_sha256": stage1_identity[
            "code_verification_sha256"
        ],
        "training_protocol_sha256": protocol_sha,
        "training_code_inventory_sha256": stage1_identity[
            "code_inventory_sha256"
        ],
        "training_code_archive_sha256": stage1_identity["code_archive_sha256"],
    }
    if repair_identity is not None and repair_protocol_sha is not None:
        source_identities["stage2_repair"] = {
            **repair_identity,
            "stage2_repair_protocol_sha256": repair_protocol_sha,
            "base_training_protocol_sha256": protocol_sha,
            "stage1_checkpoint_sha256": stage1.checkpoint_sha256,
            "stage2_replay_manifest_sha256": stage2.replay_manifest_sha256,
            "stage2_run_manifest_sha256": base_run_manifest_sha,
        }
        if successor_identity is not None:
            source_identities["stage2_successor"] = dict(successor_identity)
        result.update(
            {
                "training_source_identities": source_identities,
                "stage2_repair_code_commit": repair_identity["code_commit"],
                "stage2_repair_code_verification_sha256": repair_identity[
                    "code_verification_sha256"
                ],
                "stage2_repair_protocol_sha256": repair_protocol_sha,
                "stage2_repair_code_inventory_sha256": repair_identity[
                    "code_inventory_sha256"
                ],
                "stage2_repair_code_archive_sha256": repair_identity[
                    "code_archive_sha256"
                ],
            }
        )
    return result, artifacts

def _matched_input_evidence(
    matched: Mapping[str, object],
    *,
    run_root: Path,
    run_id: str,
    collection_manifest_sha256: str,
    evaluation_code: Mapping[str, object],
) -> list[Artifact]:
    if matched.get("collection_manifest_sha256") != collection_manifest_sha256:
        raise ArchiveError("matched result collection manifest SHA-256 differs")
    if (
        matched.get("evaluation_code_commit") != evaluation_code.get("commit")
        or matched.get("evaluation_code_inventory_sha256")
        != evaluation_code.get("inventory_sha256")
    ):
        raise ArchiveError("matched result evaluation code identity differs")
    contract = matched.get("matched_contract")
    if (
        not isinstance(contract, dict)
        or contract.get("run_manifest_sha256") != collection_manifest_sha256
    ):
        raise ArchiveError("matched result collection contract differs")
    inputs = matched.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != set(EVALUATION_ARMS):
        raise ArchiveError("matched result evaluation input inventory differs")
    artifacts: list[Artifact] = []
    for arm in EVALUATION_ARMS:
        reference = inputs.get(arm)
        if not isinstance(reference, dict):
            raise ArchiveError(f"matched result {arm} input is invalid")
        path = _regular_file(Path(str(reference.get("path"))), run_root, f"{arm} evaluation")
        expected = _validate_sha256(reference.get("sha256"), f"{arm} evaluation SHA-256")
        if sha256_file(path) != expected:
            raise ArchiveError(f"{arm} evaluation SHA-256 differs")
        receipt = _load_json(path, f"{arm} evaluation")
        receipt_code = receipt.get("evaluation_code")
        if (
            receipt.get("schema_name") != "navigation_rlt_matched_train_development_eval"
            or receipt.get("schema_version") != 1
            or receipt.get("status") != "completed"
            or receipt.get("run_id") != run_id
            or receipt.get("collection_manifest_sha256") != collection_manifest_sha256
            or receipt.get("run_manifest_sha256") != collection_manifest_sha256
            or not isinstance(receipt_code, dict)
            or receipt_code.get("commit") != evaluation_code.get("commit")
            or receipt_code.get("inventory_sha256")
            != evaluation_code.get("inventory_sha256")
        ):
            raise ArchiveError(f"{arm} evaluation provenance differs")
        artifacts.append(
            _artifact(
                path,
                source_root=run_root,
                group="results",
                relative=Path("matched/inputs") / f"{arm}.json",
                kind=f"{arm} evaluation",
            )
        )
    return artifacts


def _reviewed_post_training_inputs(
    *,
    post_training_plan_path: Path,
    expected_post_training_plan_sha256: str,
    evaluation_plan_path: Path,
    expected_evaluation_plan_sha256: str,
    split_manifest_path: Path,
    expected_split_manifest_sha256: str,
    matched_markdown_path: Path,
    expected_matched_markdown_sha256: str,
    matched_result_path: Path,
    run_root: Path,
    run_id: str,
    stage2_checkpoint_sha256: str,
    collection_manifest_sha256: str,
    collection_split_manifest_sha256: str,
    training_protocol_sha256: str,
    evaluation_code: Mapping[str, object],
    stage2_repair_code_verification_sha256: str | None = None,
    stage2_repair_protocol_sha256: str | None = None,
    training_source_identities: Mapping[str, object] | None = None,
    stage2_successor_evidence: Mapping[str, object] | None = None,
) -> tuple[dict[str, str], list[Artifact]]:
    paths_and_hashes = (
        (
            "post-training plan",
            post_training_plan_path,
            expected_post_training_plan_sha256,
        ),
        ("evaluation plan", evaluation_plan_path, expected_evaluation_plan_sha256),
        ("train split manifest", split_manifest_path, expected_split_manifest_sha256),
        ("matched Markdown", matched_markdown_path, expected_matched_markdown_sha256),
    )
    checked: dict[str, tuple[Path, str]] = {}
    for name, raw_path, raw_sha in paths_and_hashes:
        path = _regular_file(raw_path, run_root, name)
        expected = _validate_sha256(raw_sha, f"expected {name} SHA-256")
        if sha256_file(path) != expected:
            raise ArchiveError(f"{name} SHA-256 differs")
        checked[name] = (path, expected)
    plan_path, plan_sha = checked["post-training plan"]
    evaluation_path, evaluation_sha = checked["evaluation plan"]
    split_path, split_sha = checked["train split manifest"]
    markdown_path, markdown_sha = checked["matched Markdown"]

    split = _load_json(split_path, "train split manifest")
    split_rows = split.get("splits")
    update = split_rows.get("train_update") if isinstance(split_rows, dict) else None
    development = (
        split_rows.get("train_development") if isinstance(split_rows, dict) else None
    )
    if (
        split.get("schema_name") != "navigation_rlt_train_scene_split"
        or split.get("schema_version") != 1
        or split.get("upstream_split_read") != "train"
        or split.get("validation_split_reads") != 0
        or not isinstance(update, dict)
        or update.get("episode_count") != 256
        or not isinstance(update.get("episodes"), list)
        or len(update["episodes"]) != 256
        or not isinstance(development, dict)
        or development.get("episode_count") != 32
        or not isinstance(development.get("episodes"), list)
        or len(development["episodes"]) != 32
    ):
        raise ArchiveError("train split manifest fixed contract differs")
    _validate_sha256(development.get("dataset_sha256"), "development dataset SHA-256")
    _validate_sha256(development.get("gt_sha256"), "development ground-truth SHA-256")

    evaluation_plan = _load_json(evaluation_path, "evaluation plan")
    split_reference = evaluation_plan.get("split_manifest")
    workers = evaluation_plan.get("workers")
    if (
        evaluation_plan.get("schema_name") != "navigation_rlt_parallel_rollout_plan"
        or evaluation_plan.get("schema_version") != 1
        or evaluation_plan.get("run_id") != run_id
        or evaluation_plan.get("split_label") != "train_development"
        or evaluation_plan.get("validation_split_reads") != 0
        or evaluation_plan.get("episode_count") != 32
        or not isinstance(split_reference, dict)
        or Path(str(split_reference.get("path"))).resolve(strict=True) != split_path
        or split_reference.get("sha256") != split_sha
        or not isinstance(workers, list)
        or evaluation_plan.get("worker_count") != 1
        or len(workers) != 1
        or not isinstance(workers[0], dict)
        or workers[0].get("worker_id") != 0
        or not isinstance(workers[0].get("episodes"), list)
        or len(workers[0]["episodes"]) != 32
    ):
        raise ArchiveError("evaluation plan fixed contract differs")
    planned_identities = [
        [
            row.get("episode_order"),
            row.get("scene"),
            row.get("episode_id"),
            row.get("instruction_text_sha256"),
        ]
        for row in workers[0]["episodes"]
        if isinstance(row, dict)
    ]
    split_identities = [
        [
            index,
            row.get("scene"),
            str(row.get("episode_id")),
            row.get("instruction_text_sha256"),
        ]
        for index, row in enumerate(development["episodes"])
        if isinstance(row, dict)
    ]
    if (
        len(planned_identities) != 32
        or planned_identities != split_identities
        or any(
            row[0] != index
            or not isinstance(row[1], str)
            or not row[1]
            or not isinstance(row[2], str)
            or not row[2]
            or SHA256_PATTERN.fullmatch(str(row[3])) is None
            for index, row in enumerate(planned_identities)
        )
    ):
        raise ArchiveError("evaluation plan episode identities differ from fixed split")

    plan = _load_json(plan_path, "post-training plan")
    stage2 = plan.get("stage2")
    collection_reference = plan.get("collection_manifest")
    plan_split = plan.get("split_manifest")
    plan_evaluation = plan.get("evaluation_plan")
    plan_code = plan.get("evaluation_code")
    plan_protocol = plan.get("training_protocol")
    plan_repair_verification = plan.get("stage2_repair_code_verification")
    plan_repair_protocol = plan.get("stage2_repair_protocol")
    plan_source_identities = plan.get("training_source_identities")
    outputs = plan.get("outputs")
    if (
        plan.get("schema_name") != "navigation_rlt_post_training_plan"
        or plan.get("schema_version") != 1
        or plan.get("status") != "planned"
        or plan.get("run_id") != run_id
        or Path(str(plan.get("a800_run_root"))).resolve(strict=True) != run_root
        or not isinstance(stage2, dict)
        or not isinstance(stage2.get("checkpoint"), dict)
        or stage2["checkpoint"].get("sha256") != stage2_checkpoint_sha256
        or not isinstance(collection_reference, dict)
        or collection_reference.get("sha256") != collection_manifest_sha256
        or not isinstance(plan_split, dict)
        or Path(str(plan_split.get("path"))).resolve(strict=True) != split_path
        or plan_split.get("sha256") != split_sha
        or split_sha != collection_split_manifest_sha256
        or not isinstance(plan_evaluation, dict)
        or Path(str(plan_evaluation.get("path"))).resolve(strict=True) != evaluation_path
        or plan_evaluation.get("sha256") != evaluation_sha
        or plan_evaluation.get("episode_count") != 32
        or not isinstance(plan_code, dict)
        or plan_code.get("commit") != evaluation_code.get("commit")
        or plan_code.get("inventory_sha256")
        != evaluation_code.get("inventory_sha256")
        or plan_code.get("source_root") != evaluation_code.get("source_root")
        or not isinstance(plan_protocol, dict)
        or plan_protocol.get("sha256") != training_protocol_sha256
        or not isinstance(outputs, dict)
        or Path(str(outputs.get("matched_result"))).resolve(strict=True)
        != matched_result_path
        or Path(str(outputs.get("matched_markdown"))).resolve(strict=True)
        != markdown_path
    ):
        raise ArchiveError("post-training plan reviewed identity differs")
    if stage2_successor_evidence is not None and stage2 != dict(
        stage2_successor_evidence
    ):
        raise ArchiveError("post-training plan Stage2 successor evidence differs")
    repair_expected = (
        stage2_repair_code_verification_sha256 is not None
        or stage2_repair_protocol_sha256 is not None
        or training_source_identities is not None
    )
    if repair_expected:
        if (
            stage2_repair_code_verification_sha256 is None
            or stage2_repair_protocol_sha256 is None
            or training_source_identities is None
            or not isinstance(plan_repair_verification, dict)
            or plan_repair_verification.get("sha256")
            != stage2_repair_code_verification_sha256
            or not isinstance(plan_repair_protocol, dict)
            or plan_repair_protocol.get("sha256") != stage2_repair_protocol_sha256
            or plan_source_identities != dict(training_source_identities)
        ):
            raise ArchiveError("post-training plan Stage2 repair identity differs")
    elif any(
        value is not None
        for value in (
            plan_repair_verification,
            plan_repair_protocol,
            plan_source_identities,
        )
    ):
        raise ArchiveError("legacy post-training plan unexpectedly claims Stage2 repair")

    identity = {
        "post_training_plan_sha256": plan_sha,
        "evaluation_plan_sha256": evaluation_sha,
        "split_manifest_sha256": split_sha,
        "matched_markdown_sha256": markdown_sha,
    }
    artifacts = [
        _artifact(
            plan_path,
            source_root=run_root,
            group="results",
            relative=Path("provenance/post_training/post_training_plan.json"),
            kind="reviewed post-training plan",
        ),
        _artifact(
            evaluation_path,
            source_root=run_root,
            group="results",
            relative=Path("provenance/evaluation/evaluation_plan.json"),
            kind="fixed evaluation plan",
        ),
        _artifact(
            split_path,
            source_root=run_root,
            group="results",
            relative=Path("provenance/evaluation/split_manifest.json"),
            kind="fixed train split manifest",
        ),
        _artifact(
            markdown_path,
            source_root=run_root,
            group="results",
            relative=Path("matched/matched_comparison.md"),
            kind="matched comparison Markdown",
        ),
    ]
    return identity, artifacts


def _post_training_handoff_evidence(
    *,
    copy_receipt_path: Path,
    expected_copy_receipt_sha256: str,
    handoff_receipt_path: Path,
    expected_handoff_receipt_sha256: str,
    run_root: Path,
    run_id: str,
    stage2: StageEvidence,
    matched: Mapping[str, object],
    collection_manifest_sha256: str,
    evaluation_code: Mapping[str, object],
    reviewed_inputs: Mapping[str, str],
) -> tuple[dict[str, object], list[Artifact]]:
    copy_path = _regular_file(
        copy_receipt_path, run_root, "verified checkpoint copy receipt"
    )
    copy_sha = _validate_sha256(
        expected_copy_receipt_sha256,
        "expected verified checkpoint copy receipt SHA-256",
    )
    if sha256_file(copy_path) != copy_sha:
        raise ArchiveError("verified checkpoint copy receipt SHA-256 differs")
    copy = _load_json(copy_path, "verified checkpoint copy receipt")
    source = copy.get("source")
    destination = copy.get("destination")
    if (
        copy.get("schema_name") != VERIFIED_COPY_SCHEMA
        or copy.get("schema_version") != 1
        or copy.get("status") != "verified"
        or copy.get("run_id") != run_id
        or copy.get("atomic_publish") is not True
        or copy.get("temporary_absent_after_publish") is not True
        or not isinstance(source, dict)
        or source.get("path") != str(stage2.checkpoint)
        or source.get("sha256") != stage2.checkpoint_sha256
        or source.get("size_bytes") != stage2.checkpoint.stat().st_size
        or not isinstance(destination, dict)
        or destination.get("sha256") != stage2.checkpoint_sha256
        or destination.get("size_bytes") != stage2.checkpoint.stat().st_size
        or not isinstance(destination.get("path"), str)
        or not PurePosixPath(str(destination["path"])).is_absolute()
    ):
        raise ArchiveError("verified checkpoint copy does not bind the Stage2 checkpoint")

    handoff_path = _regular_file(
        handoff_receipt_path, run_root, "evaluation handoff receipt"
    )
    handoff_sha = _validate_sha256(
        expected_handoff_receipt_sha256,
        "expected evaluation handoff receipt SHA-256",
    )
    if sha256_file(handoff_path) != handoff_sha:
        raise ArchiveError("evaluation handoff receipt SHA-256 differs")
    handoff = _load_json(handoff_path, "evaluation handoff receipt")
    configs = handoff.get("arm_configs")
    arm_rows = handoff.get("arms")
    if (
        handoff.get("schema_name") != EVALUATION_HANDOFF_SCHEMA
        or handoff.get("schema_version") != 1
        or handoff.get("status") != "verified"
        or handoff.get("run_id") != run_id
        or handoff.get("collection_manifest_sha256") != collection_manifest_sha256
        or handoff.get("split_manifest_sha256")
        != reviewed_inputs.get("split_manifest_sha256")
        or handoff.get("evaluation_plan_sha256")
        != reviewed_inputs.get("evaluation_plan_sha256")
        or handoff.get("evaluation_code_commit") != evaluation_code.get("commit")
        or handoff.get("evaluation_code_inventory_sha256")
        != evaluation_code.get("inventory_sha256")
        or handoff.get("navigation_rlt_checkpoint_sha256")
        != stage2.checkpoint_sha256
        or handoff.get("max_physical_steps") != 160
        or handoff.get("seed") != 2026100202
        or handoff.get("arm_order") != list(EVALUATION_ARMS)
        or not isinstance(configs, dict)
        or set(configs) != set(EVALUATION_ARMS)
        or not isinstance(arm_rows, list)
        or len(arm_rows) != len(EVALUATION_ARMS)
    ):
        raise ArchiveError("evaluation handoff identity/contract differs")
    for arm, raw_config in configs.items():
        if not isinstance(raw_config, dict):
            raise ArchiveError(f"{arm} evaluation config is invalid")
        config = dict(raw_config)
        declared = _validate_sha256(
            config.pop("sha256", None), f"{arm} evaluation config SHA-256"
        )
        if _object_sha256(config) != declared:
            raise ArchiveError(f"{arm} evaluation config SHA-256 differs")

    matched_inputs = matched.get("inputs")
    if not isinstance(matched_inputs, dict):
        raise ArchiveError("matched input inventory is absent")
    artifacts: list[Artifact] = [
        _artifact(
            copy_path,
            source_root=run_root,
            group="results",
            relative=Path("provenance/post_training/VERIFIED_COPY.json"),
            kind="verified checkpoint copy receipt",
        ),
        _artifact(
            handoff_path,
            source_root=run_root,
            group="results",
            relative=Path("provenance/post_training/REMOTE_EVAL_COMPLETED.json"),
            kind="evaluation handoff receipt",
        ),
    ]
    for expected_arm, row in zip(EVALUATION_ARMS, arm_rows):
        if not isinstance(row, dict) or row.get("arm") != expected_arm:
            raise ArchiveError("evaluation handoff arm order differs")
        config = configs[expected_arm]
        if row.get("arm_config_sha256") != config.get("sha256"):
            raise ArchiveError(f"{expected_arm} handoff config identity differs")
        evaluation_ref = row.get("evaluation")
        completion_ref = row.get("completion")
        if not isinstance(evaluation_ref, dict) or not isinstance(completion_ref, dict):
            raise ArchiveError(f"{expected_arm} handoff files are absent")
        evaluation = _regular_file(
            Path(str(evaluation_ref.get("path"))),
            run_root,
            f"{expected_arm} returned evaluation",
        )
        completion = _regular_file(
            Path(str(completion_ref.get("path"))),
            run_root,
            f"{expected_arm} returned completion",
        )
        evaluation_sha = _validate_sha256(
            evaluation_ref.get("sha256"), f"{expected_arm} returned evaluation SHA-256"
        )
        completion_sha = _validate_sha256(
            completion_ref.get("sha256"), f"{expected_arm} returned completion SHA-256"
        )
        if (
            sha256_file(evaluation) != evaluation_sha
            or evaluation_ref.get("size_bytes") != evaluation.stat().st_size
            or sha256_file(completion) != completion_sha
            or completion_ref.get("size_bytes") != completion.stat().st_size
        ):
            raise ArchiveError(f"{expected_arm} returned file readback differs")
        matched_ref = matched_inputs.get(expected_arm)
        if (
            not isinstance(matched_ref, dict)
            or Path(str(matched_ref.get("path"))).resolve(strict=True) != evaluation
            or matched_ref.get("sha256") != evaluation_sha
        ):
            raise ArchiveError(f"{expected_arm} handoff and comparator inputs differ")
        completion_value = _load_json(completion, f"{expected_arm} completion")
        evaluation_value = _load_json(evaluation, f"{expected_arm} evaluation")
        if (
            completion_value.get("schema_name")
            != "navigation_rlt_matched_train_development_eval"
            or completion_value.get("status") != "completed"
            or completion_value.get("evaluation_sha256") != evaluation_sha
            or completion_value.get("collection_manifest_sha256")
            != collection_manifest_sha256
            or evaluation_value.get("collection_manifest_sha256")
            != collection_manifest_sha256
            or evaluation_value.get("evaluation_code", {}).get("inventory_sha256")
            != evaluation_code.get("inventory_sha256")
        ):
            raise ArchiveError(f"{expected_arm} returned completion/evaluation differs")
        artifacts.append(
            _artifact(
                completion,
                source_root=run_root,
                group="results",
                relative=Path("matched/completions") / f"{expected_arm}.json",
                kind=f"{expected_arm} evaluation completion",
            )
        )
    return {
        "verified_copy_receipt_sha256": copy_sha,
        "evaluation_handoff_receipt_sha256": handoff_sha,
        **dict(reviewed_inputs),
    }, artifacts


def _deduplicate_artifacts(artifacts: Sequence[Artifact]) -> list[Artifact]:
    destinations: set[tuple[str, str]] = set()
    sources: set[Path] = set()
    result: list[Artifact] = []
    for artifact in artifacts:
        key = (artifact.group, artifact.relative.as_posix())
        if key in destinations:
            raise ArchiveError(f"duplicate archive destination: {key}")
        destinations.add(key)
        if artifact.source in sources and artifact.kind != "source":
            raise ArchiveError(f"duplicate archive source: {artifact.source}")
        sources.add(artifact.source)
        result.append(artifact)
    if len(result) > MAX_RESULT_FILES:
        raise ArchiveError("archive file count exceeds the bounded limit")
    return result


def _ensure_destination_parents(obs_root: Path) -> None:
    for relative in (CHECKPOINT_NAMESPACE, RESULT_NAMESPACE):
        current = obs_root
        for part in relative.parts:
            current = current / part
            if _lexists(current):
                if current.is_symlink() or not current.is_dir():
                    raise ArchiveError(f"OBS destination parent is unsafe: {current}")
            else:
                current.mkdir()


def _independent_readback(records: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    code = r'''
import hashlib, json, os, pathlib, sys
plan=json.load(sys.stdin)
out=[]
for row in plan:
    path=pathlib.Path(row["destination"])
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"readback is not a regular file: {path}")
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    size=path.stat().st_size
    if digest.hexdigest()!=row["sha256"] or size!=row["size_bytes"]:
        raise SystemExit(f"readback differs: {path}")
    out.append({"destination":str(path),"sha256":digest.hexdigest(),"size_bytes":size})
print(json.dumps(out,sort_keys=True,separators=(",",":")))
'''
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        input=json.dumps(list(records), sort_keys=True, allow_nan=False),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=3600,
    )
    if completed.returncode != 0:
        raise ArchiveError(f"independent OBS readback failed: {completed.stderr.strip()}")
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ArchiveError("independent OBS readback returned invalid JSON") from error
    if not isinstance(value, list) or len(value) != len(records):
        raise ArchiveError("independent OBS readback inventory differs")
    return value


def _copy_to_staging(
    artifacts: Sequence[Artifact], checkpoint_staging: Path, result_staging: Path
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for artifact in artifacts:
        root = checkpoint_staging if artifact.group == "checkpoints" else result_staging
        destination = root / artifact.relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if _lexists(destination):
            raise FileExistsError(destination)
        shutil.copyfile(artifact.source, destination)
        copied_sha = sha256_file(destination)
        copied_size = destination.stat().st_size
        if copied_sha != artifact.sha256 or copied_size != artifact.size_bytes:
            raise ArchiveError(f"staging readback differs: {artifact.relative}")
        records.append(
            {
                "kind": artifact.kind,
                "source": str(artifact.source),
                "group": artifact.group,
                "relative_path": artifact.relative.as_posix(),
                "size_bytes": artifact.size_bytes,
                "sha256": artifact.sha256,
            }
        )
    return records


def _stage2_successor_evidence(
    args: argparse.Namespace, *, run_root: Path
) -> dict[str, object] | None:
    values = (
        getattr(args, "stage2_formal_completion", None),
        getattr(args, "expected_stage2_formal_completion_sha256", None),
        getattr(args, "stage2_source_root", None),
        getattr(args, "evaluation_source_root", None),
    )
    supplied = all(value is not None for value in values)
    if any(value is not None for value in values) and not supplied:
        raise ArchiveError("successor archive arguments must be all present")
    if not supplied:
        return None
    formal_path = _regular_file(
        Path(values[0]), run_root, "Stage2 successor formal completion"
    )
    formal_sha = _validate_sha256(
        values[1], "expected Stage2 successor formal completion SHA-256"
    )
    if sha256_file(formal_path) != formal_sha:
        raise ArchiveError("Stage2 successor formal completion SHA-256 differs")
    formal = _load_json(formal_path, "Stage2 successor formal completion")
    marker_ref = formal.get("stage_marker")
    stage_run_root = formal_path.parent
    marker_path = stage_run_root / "navigation_rlt_stage2/STAGE.json"
    if (
        stage_run_root.parent != run_root / "stage2-successors"
        or formal_path.name != "FORMAL_COMPLETED.json"
        or not isinstance(marker_ref, Mapping)
        or marker_ref.get("path") != str(marker_path)
    ):
        raise ArchiveError("Stage2 successor formal root/marker is not canonical")
    marker_sha = _validate_sha256(
        marker_ref.get("sha256"), "Stage2 successor marker SHA-256"
    )
    try:
        evidence = post_training._stage2_evidence(
            marker_path,
            expected_marker_sha256=marker_sha,
            run_root=run_root,
            run_id=args.run_id,
            formal_completion_path=formal_path,
            expected_formal_completion_sha256=formal_sha,
            source_root=Path(values[2]),
        )
    except post_training.PostTrainingError as error:
        raise ArchiveError(f"Stage2 successor evidence differs: {error}") from error
    if (
        not isinstance(evidence, dict)
        or evidence.get("stage_run_root") != str(stage_run_root)
        or evidence.get("formal_completion")
        != {"path": str(formal_path), "sha256": formal_sha}
    ):
        raise ArchiveError("Stage2 successor helper result differs")
    return evidence


def _archive(args: argparse.Namespace) -> dict[str, object]:
    if RUN_ID_PATTERN.fullmatch(args.run_id) is None or args.run_id in (".", ".."):
        raise ArchiveError("run-id must be one safe path component")
    runs_root = _regular_directory(RUNS_ROOT, RUNS_ROOT, "fixed runs root")
    run_root = _regular_directory(args.run_root, runs_root, "run root")
    if run_root.name != args.run_id:
        raise ArchiveError("run-id must equal the run-root basename")
    obs_mount = _regular_directory(OBS_MOUNT, OBS_MOUNT, "OBS mount")
    obs_root = _regular_directory(args.obs_root, obs_mount, "OBS root")
    matched_path = _regular_file(args.matched_result, run_root, "matched result")
    expected_matched_sha = _validate_sha256(
        args.expected_matched_result_sha256, "matched result SHA-256"
    )
    if sha256_file(matched_path) != expected_matched_sha:
        raise ArchiveError("matched result SHA-256 differs")
    matched = _load_json(matched_path, "matched result")

    successor = _stage2_successor_evidence(args, run_root=run_root)
    stage1 = _validate_stage(
        run_root, phase="stage1", run_id=args.run_id, runs_root=runs_root
    )
    stage2 = _validate_stage(
        run_root,
        phase="stage2",
        run_id=args.run_id,
        runs_root=runs_root,
        stage_run_root=(
            Path(str(successor["stage_run_root"]))
            if successor is not None
            else None
        ),
        successor=successor,
    )
    stages = [stage1, stage2]
    cross_stage_artifacts = _cross_stage_provenance(
        stage1, stage2, run_root=run_root
    )
    matched_checkpoint_sha = _matched_checkpoint_sha256(
        matched,
        run_id=args.run_id,
    )
    if matched_checkpoint_sha != stage2.checkpoint_sha256:
        raise ArchiveError("matched result does not bind the Stage2 checkpoint")
    if stage1.run_manifest.get("model") != stage2.run_manifest.get("model"):
        raise ArchiveError("Stage1 and Stage2 model contracts differ")
    if stage2.run_manifest.get("stage1_checkpoint") != {
        "sha256": stage1.checkpoint_sha256
    }:
        raise ArchiveError("Stage2 run manifest does not bind the Stage1 checkpoint")
    stage2_checkpoint_input = stage2.training_result.get("stage1_checkpoint_input")
    if (
        not isinstance(stage2_checkpoint_input, dict)
        or stage2_checkpoint_input.get("sha256") != stage1.checkpoint_sha256
    ):
        raise ArchiveError("Stage2 training result does not bind the Stage1 checkpoint")
    promotion_receipts = [
        claims
        for evidence in stages
        for claims in evidence.source_claims
        if "promoter_sha256" in claims
    ]
    if len(promotion_receipts) != 1:
        raise ArchiveError("exactly one frozen replay promotion receipt is required")
    # The promotion receipt itself is archived and its checkpoint binding is
    # checked by reloading the receipt selected through the package artifacts.
    promotion_artifacts = [
        artifact
        for artifact in stage2.artifacts
        if artifact.relative.as_posix().endswith(
            "replay_manifest/promotion_receipt.json"
        )
    ]
    if len(promotion_artifacts) != 1:
        raise ArchiveError("Stage2 promotion receipt inventory differs")
    promotion_artifact = promotion_artifacts[0]
    promotion_receipt = _load_json(promotion_artifact.source, "Stage2 promotion receipt")
    promoted_checkpoint = promotion_receipt.get("stage1_checkpoint")
    if (
        not isinstance(promoted_checkpoint, dict)
        or promoted_checkpoint.get("sha256") != stage1.checkpoint_sha256
    ):
        raise ArchiveError("frozen replay promotion does not bind the Stage1 checkpoint")

    stage2_training = stage2.run_manifest.get("training")
    repair_objective = (
        isinstance(stage2_training, dict)
        and stage2_training.get("objective_schema")
        in (
            STAGE2_REPAIR_OBJECTIVE_SCHEMA,
            STAGE2_BEHAVIOR_KL_OBJECTIVE_SCHEMA,
        )
    )
    source_artifacts, source_inventory = _source_inventory(
        run_root, [stage1] if repair_objective else stages
    )
    evaluation_code, evaluation_artifacts = _evaluation_code_evidence(
        args.evaluation_code_inventory,
        args.expected_evaluation_code_inventory_sha256,
        source_root=getattr(args, "evaluation_source_root", None),
    )
    provenance, provenance_artifacts = _training_provenance_evidence(
        run_root=run_root,
        run_id=args.run_id,
        collection_manifest_path=args.collection_manifest,
        expected_collection_sha256=args.expected_collection_manifest_sha256,
        code_verification_path=args.training_code_verification,
        expected_code_verification_sha256=(
            args.expected_training_code_verification_sha256
        ),
        training_protocol_path=args.training_protocol,
        expected_training_protocol_sha256=args.expected_training_protocol_sha256,
        stages=stages,
        training_source_inventory=source_inventory,
        stage2_repair_code_verification_path=getattr(
            args, "stage2_repair_code_verification", None
        ),
        expected_stage2_repair_code_verification_sha256=getattr(
            args, "expected_stage2_repair_code_verification_sha256", None
        ),
        stage2_repair_protocol_path=getattr(
            args, "stage2_repair_protocol", None
        ),
        expected_stage2_repair_protocol_sha256=getattr(
            args, "expected_stage2_repair_protocol_sha256", None
        ),
    )
    matched_input_artifacts = _matched_input_evidence(
        matched,
        run_root=run_root,
        run_id=args.run_id,
        collection_manifest_sha256=str(provenance["collection_manifest_sha256"]),
        evaluation_code=evaluation_code,
    )
    reviewed_inputs, reviewed_input_artifacts = _reviewed_post_training_inputs(
        post_training_plan_path=args.post_training_plan,
        expected_post_training_plan_sha256=(
            args.expected_post_training_plan_sha256
        ),
        evaluation_plan_path=args.evaluation_plan,
        expected_evaluation_plan_sha256=args.expected_evaluation_plan_sha256,
        split_manifest_path=args.split_manifest,
        expected_split_manifest_sha256=args.expected_split_manifest_sha256,
        matched_markdown_path=args.matched_markdown,
        expected_matched_markdown_sha256=args.expected_matched_markdown_sha256,
        matched_result_path=matched_path,
        run_root=run_root,
        run_id=args.run_id,
        stage2_checkpoint_sha256=stage2.checkpoint_sha256,
        collection_manifest_sha256=str(provenance["collection_manifest_sha256"]),
        collection_split_manifest_sha256=str(
            provenance["collection_split_manifest_sha256"]
        ),
        training_protocol_sha256=str(provenance["training_protocol_sha256"]),
        evaluation_code=evaluation_code,
        stage2_repair_code_verification_sha256=(
            str(provenance["stage2_repair_code_verification_sha256"])
            if "stage2_repair_code_verification_sha256" in provenance
            else None
        ),
        stage2_repair_protocol_sha256=(
            str(provenance["stage2_repair_protocol_sha256"])
            if "stage2_repair_protocol_sha256" in provenance
            else None
        ),
        training_source_identities=(
            provenance.get("training_source_identities")
            if "stage2_repair_protocol_sha256" in provenance
            else None
        ),
        stage2_successor_evidence=successor,
    )
    post_training, post_training_artifacts = _post_training_handoff_evidence(
        copy_receipt_path=args.verified_copy_receipt,
        expected_copy_receipt_sha256=args.expected_verified_copy_receipt_sha256,
        handoff_receipt_path=args.evaluation_handoff_receipt,
        expected_handoff_receipt_sha256=(
            args.expected_evaluation_handoff_receipt_sha256
        ),
        run_root=run_root,
        run_id=args.run_id,
        stage2=stage2,
        matched=matched,
        collection_manifest_sha256=str(provenance["collection_manifest_sha256"]),
        evaluation_code=evaluation_code,
        reviewed_inputs=reviewed_inputs,
    )
    artifacts = [
        *(artifact for stage in stages for artifact in stage.artifacts),
        *cross_stage_artifacts,
        *source_artifacts,
        *evaluation_artifacts,
        *provenance_artifacts,
        *matched_input_artifacts,
        *reviewed_input_artifacts,
        *post_training_artifacts,
        _artifact(
            matched_path,
            source_root=run_root,
            group="results",
            relative=Path("matched/matched_result.json"),
            kind="matched result",
        ),
    ]
    artifacts = _deduplicate_artifacts(artifacts)
    result_bytes = sum(
        artifact.size_bytes for artifact in artifacts if artifact.group == "results"
    )
    if result_bytes > MAX_RESULTS_BYTES:
        raise ArchiveError("bounded result archive exceeds the 4 GiB hard cap")
    if any(
        artifact.relative.suffix.lower() in (".npz", ".npy") for artifact in artifacts
    ):
        raise ArchiveError("replay payload arrays must not enter the OBS archive")

    _ensure_destination_parents(obs_root)
    checkpoint_final = obs_root / CHECKPOINT_NAMESPACE / args.run_id
    result_final = obs_root / RESULT_NAMESPACE / args.run_id
    checkpoint_staging = checkpoint_final.with_name("." + args.run_id + ".partial")
    result_staging = result_final.with_name("." + args.run_id + ".partial")
    for path in (
        checkpoint_final,
        result_final,
        checkpoint_staging,
        result_staging,
    ):
        if _lexists(path):
            raise FileExistsError(path)
    checkpoint_staging.mkdir()
    result_staging.mkdir()

    records = _copy_to_staging(artifacts, checkpoint_staging, result_staging)
    source_inventory_path = result_staging / "source/source_inventory.json"
    source_inventory_sha = _atomic_json_new(source_inventory_path, source_inventory)
    records.append(
        {
            "kind": "source inventory",
            "source": None,
            "group": "results",
            "relative_path": "source/source_inventory.json",
            "size_bytes": source_inventory_path.stat().st_size,
            "sha256": source_inventory_sha,
        }
    )

    _publish_archive_pair(
        checkpoint_staging,
        checkpoint_final,
        result_staging,
        result_final,
    )
    readback_plan = []
    for record in records:
        root = checkpoint_final if record["group"] == "checkpoints" else result_final
        readback_plan.append(
            {
                "destination": str(root / str(record["relative_path"])),
                "sha256": record["sha256"],
                "size_bytes": record["size_bytes"],
            }
        )
    readback = _independent_readback(readback_plan)
    readback_by_destination = {row["destination"]: row for row in readback}
    for record, plan in zip(records, readback_plan):
        row = readback_by_destination.get(plan["destination"])
        if row is None or row["sha256"] != record["sha256"]:
            raise ArchiveError("independent readback receipt differs")
        record["destination"] = plan["destination"]
        record["readback_sha256"] = row["sha256"]

    readback_fingerprint = hashlib.sha256(
        json.dumps(readback, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "status": "readback_verified",
        "run_id": args.run_id,
        "source_run_root": str(run_root),
        "destinations": {
            "checkpoints": str(checkpoint_final),
            "results": str(result_final),
        },
        "checkpoints": {
            stage.phase: {
                "source": str(stage.checkpoint),
                "sha256": stage.checkpoint_sha256,
                "size_bytes": stage.checkpoint.stat().st_size,
            }
            for stage in stages
        },
        "replay": {stage.phase: stage.replay_summary for stage in stages},
        "matched_result": {
            "source": str(matched_path),
            "sha256": expected_matched_sha,
            "navigation_rlt_checkpoint_sha256": matched_checkpoint_sha,
        },
        "code_identity": {
            **provenance,
            "evaluation_code_commit": evaluation_code["commit"],
            "evaluation_code_inventory_sha256": evaluation_code[
                "inventory_sha256"
            ],
            **(
                {
                    "evaluation_code_source_root": evaluation_code[
                        "source_root"
                    ]
                }
                if "source_root" in evaluation_code
                else {}
            ),
        },
        "post_training_handoff": post_training,
        "source_inventory": source_inventory,
        "result_bytes": result_bytes,
        "checkpoint_hard_cap_bytes": MAX_CHECKPOINT_BYTES,
        "results_hard_cap_bytes": MAX_RESULTS_BYTES,
        "file_count": len(records),
        "files": records,
        "independent_readback_sha256": readback_fingerprint,
        "readback_completed_unix_s": time.time(),
    }
    manifest_path = result_final / "archive_manifest.json"
    manifest_sha = _atomic_json_new(manifest_path, manifest)
    manifest_readback = _independent_readback(
        [
            {
                "destination": str(manifest_path),
                "sha256": manifest_sha,
                "size_bytes": manifest_path.stat().st_size,
            }
        ]
    )
    if manifest_readback[0]["sha256"] != manifest_sha:
        raise ArchiveError("archive manifest independent readback differs")

    completion_path = result_final / "COMPLETED.json"
    completion_sha = _atomic_json_new(
        completion_path,
        {
            "schema_name": COMPLETION_SCHEMA,
            "schema_version": 1,
            "status": "completed",
            "run_id": args.run_id,
            "archive_manifest": str(manifest_path),
            "archive_manifest_sha256": manifest_sha,
            "independent_readback_sha256": readback_fingerprint,
            "checkpoint_root": str(checkpoint_final),
            "result_root": str(result_final),
            "post_training_handoff": post_training,
            "completed_unix_s": time.time(),
        },
    )
    completion_readback = _independent_readback(
        [
            {
                "destination": str(completion_path),
                "sha256": completion_sha,
                "size_bytes": completion_path.stat().st_size,
            }
        ]
    )
    if completion_readback[0]["sha256"] != completion_sha:
        raise ArchiveError("OBS completion marker independent readback differs")
    return {
        "status": "completed",
        "run_id": args.run_id,
        "checkpoint_root": str(checkpoint_final),
        "result_root": str(result_final),
        "archive_manifest": str(manifest_path),
        "archive_manifest_sha256": manifest_sha,
        "completion_marker": str(completion_path),
        "completion_marker_sha256": completion_sha,
        "independent_readback_sha256": readback_fingerprint,
        "post_training_handoff": post_training,
        "replay_shard_payloads_archived": False,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run-root", required=True, type=Path)
    result.add_argument("--run-id", required=True)
    result.add_argument("--matched-result", required=True, type=Path)
    result.add_argument("--expected-matched-result-sha256", required=True)
    result.add_argument("--collection-manifest", required=True, type=Path)
    result.add_argument("--expected-collection-manifest-sha256", required=True)
    result.add_argument("--evaluation-code-inventory", required=True, type=Path)
    result.add_argument("--expected-evaluation-code-inventory-sha256", required=True)
    result.add_argument("--training-code-verification", required=True, type=Path)
    result.add_argument("--expected-training-code-verification-sha256", required=True)
    result.add_argument("--training-protocol", required=True, type=Path)
    result.add_argument("--expected-training-protocol-sha256", required=True)
    result.add_argument("--stage2-repair-code-verification", type=Path)
    result.add_argument("--expected-stage2-repair-code-verification-sha256")
    result.add_argument("--stage2-repair-protocol", type=Path)
    result.add_argument("--expected-stage2-repair-protocol-sha256")
    result.add_argument("--stage2-formal-completion", type=Path)
    result.add_argument("--expected-stage2-formal-completion-sha256")
    result.add_argument("--stage2-source-root", type=Path)
    result.add_argument("--evaluation-source-root", type=Path)
    result.add_argument("--post-training-plan", required=True, type=Path)
    result.add_argument("--expected-post-training-plan-sha256", required=True)
    result.add_argument("--evaluation-plan", required=True, type=Path)
    result.add_argument("--expected-evaluation-plan-sha256", required=True)
    result.add_argument("--split-manifest", required=True, type=Path)
    result.add_argument("--expected-split-manifest-sha256", required=True)
    result.add_argument("--matched-markdown", required=True, type=Path)
    result.add_argument("--expected-matched-markdown-sha256", required=True)
    result.add_argument("--verified-copy-receipt", required=True, type=Path)
    result.add_argument("--expected-verified-copy-receipt-sha256", required=True)
    result.add_argument("--evaluation-handoff-receipt", required=True, type=Path)
    result.add_argument(
        "--expected-evaluation-handoff-receipt-sha256", required=True
    )
    result.add_argument("--obs-root", type=Path, default=DEFAULT_OBS_ROOT)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    result = _archive(args)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ArchiveError, FileExistsError, OSError, subprocess.SubprocessError) as error:
        print(
            json.dumps(
                {
                    "schema_name": SCHEMA_NAME,
                    "schema_version": SCHEMA_VERSION,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(2)
