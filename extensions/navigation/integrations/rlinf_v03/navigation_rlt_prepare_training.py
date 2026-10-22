#!/usr/bin/env python3
"""Prepare content-addressed Navigation-RLT replay and trainer manifests.

The ``aggregate-prefix`` command verifies completed rollout worker packages and
builds one global prefix replay manifest without copying episode arrays.
``stage1`` emits the exact run-manifest schema consumed by
``navigation_rlt_fsdp_train.py stage1``.  After the separate replay promotion
step, ``stage2`` verifies the frozen-token package and emits the corresponding
Stage-2 run manifest.  Every command writes an immutable receipt and completion
marker inside an atomically published output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
from typing import Mapping, Sequence

import numpy as np

import navigation_rlt_policy as policy_module
import navigation_rlt_replay as replay_module
from navigation_rlt_policy import (
    CANONICAL_ACTION_DIM,
    CANONICAL_ACTION_SCHEMA,
    GATE_COUNT,
    GATE_NAMES,
    PREFIX_SELECTION_SCHEMA,
    TOPOLOGY_PACKING_SCHEMA,
    NavigationRLTConfig,
    NavigationRLTPolicy,
    Stage1LossConfig,
    Stage2LossConfig,
    Z_RL_DIM,
)
from navigation_rlt_replay import (
    FEATURE_STAGE_FROZEN_Z_RL,
    FEATURE_STAGE_PREFIX_HIDDEN,
    LIGHTNAV_SOURCE_REVISION,
    REFERENCE_DIM,
    REFERENCE_SCHEMA,
    REPRESENTATION_FROZEN_Z_RL,
    REPRESENTATION_PREFIX_HIDDEN,
    build_replay_manifest,
    load_episode_shard,
    load_replay_manifest,
    save_replay_manifest,
    sha256_file,
)


SCHEMA_NAME = "navigation_rlt_prepare_training"
SCHEMA_VERSION = 1
RUN_MANIFEST_SCHEMA = "navigation_rlt_training_run_manifest"
RUN_MANIFEST_VERSION = 1
WORKER_COMPLETION_SCHEMA = "navigation_rlt_worker_completion"
WORKER_COMPLETION_VERSION = 1
ROLLOUT_MARKER_SCHEMA = "navigation_rlt_rollout_worker"
ROLLOUT_MARKER_VERSION = 1
PROMOTION_MARKER_SCHEMA = "navigation_rlt_replay_promotion"
PROMOTION_MARKER_VERSION = 1
RECEIPT_NAME = "prepare_receipt.json"
COMPLETED_NAME = "COMPLETED.json"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
STAGE2_OBJECTIVE_SCHEMA = "proposal_conditioned_finite_candidate_cql_behavior_kl_v2"
STAGE2_BEHAVIOR_KL_WEIGHT = 1.0
STAGE2_ELIGIBILITY_SCHEMA = (
    "credit_mask_true_and_requested_gate_equals_executed_gate_v1"
)
STAGE2_CQL_ALPHA = 1.0
STAGE2_CQL_TEMPERATURE = 1.0
MINIMUM_BEHAVIOR_GATE_COUNT = 1


class PrepareTrainingError(RuntimeError):
    """Raised when a worker, replay, or trainer manifest is not provable."""


def _validate_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise PrepareTrainingError(f"{name} must be a lowercase SHA-256")
    return value


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value < 1:
        raise PrepareTrainingError(f"{name} must be a positive integer")
    return int(value)


def _finite(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PrepareTrainingError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise PrepareTrainingError(f"{name} must be {qualifier}")
    return result


def _regular_file(path: Path, name: str) -> Path:
    raw = Path(path)
    if raw.is_symlink():
        raise PrepareTrainingError(f"{name} must not be a symbolic link")
    try:
        result = raw.resolve(strict=True)
    except OSError as error:
        raise PrepareTrainingError(f"{name} is missing or inaccessible") from error
    if not result.is_file():
        raise PrepareTrainingError(f"{name} must be a regular file")
    return result


def _regular_directory(path: Path, name: str) -> Path:
    raw = Path(path)
    if raw.is_symlink():
        raise PrepareTrainingError(f"{name} must not be a symbolic link")
    try:
        result = raw.resolve(strict=True)
    except OSError as error:
        raise PrepareTrainingError(f"{name} is missing or inaccessible") from error
    if not result.is_dir():
        raise PrepareTrainingError(f"{name} must be a directory")
    return result


def _load_json(path: Path, name: str) -> dict[str, object]:
    path = _regular_file(path, name)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PrepareTrainingError(f"{name} is not valid JSON") from error
    if not isinstance(value, dict):
        raise PrepareTrainingError(f"{name} must contain a JSON object")
    return value


def _atomic_json_new(path: Path, value: object) -> None:
    path = Path(path)
    if os.path.lexists(path):
        raise FileExistsError(path)
    try:
        payload = json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8") + b"\n"
    except (TypeError, ValueError) as error:
        raise PrepareTrainingError("output is not strict JSON") from error
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".staging-", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if os.path.lexists(path):
            raise FileExistsError(path)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _output_layout(
    output_dir_value: Path, artifact_value: Path
) -> tuple[Path, Path, Path, Path]:
    output_dir = Path(output_dir_value).resolve(strict=False)
    if os.path.lexists(output_dir):
        raise FileExistsError(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if not output_dir.parent.is_dir():
        raise PrepareTrainingError("output parent is not a directory")
    raw_artifact = Path(artifact_value)
    final_artifact = (
        raw_artifact.resolve(strict=False)
        if raw_artifact.is_absolute()
        else (output_dir / raw_artifact).resolve(strict=False)
    )
    try:
        relative_artifact = final_artifact.relative_to(output_dir)
    except ValueError as error:
        raise PrepareTrainingError("output artifact escapes output-dir") from error
    if relative_artifact == Path(".") or final_artifact.suffix.lower() != ".json":
        raise PrepareTrainingError("output artifact must be a JSON file")
    staging = Path(
        tempfile.mkdtemp(
            prefix=output_dir.name + ".staging-", dir=str(output_dir.parent)
        )
    )
    return output_dir, final_artifact, relative_artifact, staging


def _publish_package(
    *,
    staging: Path,
    output_dir: Path,
    operation: str,
    artifact_kind: str,
    artifact_sha256: str,
    receipt: Mapping[str, object],
) -> tuple[str, str]:
    receipt_path = staging / RECEIPT_NAME
    _atomic_json_new(receipt_path, dict(receipt))
    receipt_sha = sha256_file(receipt_path)
    _atomic_json_new(
        staging / COMPLETED_NAME,
        {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "operation": operation,
            "artifact_kind": artifact_kind,
            "artifact_sha256": artifact_sha256,
            "receipt_sha256": receipt_sha,
        },
    )
    if os.path.lexists(output_dir):
        raise FileExistsError(output_dir)
    os.replace(staging, output_dir)
    return str(output_dir / RECEIPT_NAME), receipt_sha


def _implementation_paths() -> dict[str, Path]:
    root = Path(__file__).resolve().parent
    paths = {
        "trainer_sha256": root / "navigation_rlt_fsdp_train.py",
        "policy_sha256": root / "navigation_rlt_policy.py",
        "replay_sha256": root / "navigation_rlt_replay.py",
    }
    for name, path in paths.items():
        _regular_file(path, name)
    return paths


def _implementation_hashes(*, include_preparer: bool) -> dict[str, str]:
    result = {name: sha256_file(path) for name, path in _implementation_paths().items()}
    if include_preparer:
        result["preparer_sha256"] = sha256_file(Path(__file__).resolve())
    return result


def _contained_file(path_value: object, root: Path, name: str) -> Path:
    if not isinstance(path_value, str) or not path_value:
        raise PrepareTrainingError(f"{name} path is absent")
    path = _regular_file(Path(path_value), name)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise PrepareTrainingError(f"{name} path escapes its worker package") from error
    return path


def _worker_receipts(
    worker_dirs: Sequence[Path],
    *,
    run_id: str,
    expected_plan_sha256: str,
    expected_worker_count: int,
    expected_episode_count: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if len(worker_dirs) != expected_worker_count:
        raise PrepareTrainingError("worker directory count differs from the fixed budget")
    observed_workers: set[int] = set()
    observed_paths: set[Path] = set()
    observed_shas: set[str] = set()
    shard_receipts: list[dict[str, object]] = []
    worker_evidence: list[dict[str, object]] = []
    for raw_worker_dir in worker_dirs:
        worker_dir = _regular_directory(raw_worker_dir, "worker output directory")
        completion_path = _regular_file(
            worker_dir / "worker_completion.json", "worker completion"
        )
        marker_path = _regular_file(worker_dir / COMPLETED_NAME, "worker marker")
        marker = _load_json(marker_path, "worker marker")
        expected_marker = {
            "schema_name": ROLLOUT_MARKER_SCHEMA,
            "schema_version": ROLLOUT_MARKER_VERSION,
            "status": "completed",
            "worker_completion_sha256": sha256_file(completion_path),
        }
        if marker != expected_marker:
            raise PrepareTrainingError("worker completion marker differs")
        completion = _load_json(completion_path, "worker completion")
        required = {
            "schema_name",
            "schema_version",
            "status",
            "run_id",
            "worker_id",
            "plan_sha256",
            "split_label",
            "validation_split_reads",
            "behavior_policy_id",
            "behavior_policy_version",
            "episodes",
            "replay_shards",
        }
        if not required.issubset(completion):
            raise PrepareTrainingError("worker completion members are incomplete")
        if (
            completion["schema_name"] != WORKER_COMPLETION_SCHEMA
            or completion["schema_version"] != WORKER_COMPLETION_VERSION
            or completion["status"] != "completed"
            or completion["run_id"] != run_id
            or completion["plan_sha256"] != expected_plan_sha256
            or completion["split_label"] != "train_update"
            or completion["validation_split_reads"] != 0
        ):
            raise PrepareTrainingError("worker completion provenance differs")
        worker_id = completion["worker_id"]
        if type(worker_id) is not int or worker_id < 0 or worker_id in observed_workers:
            raise PrepareTrainingError("worker id is invalid or duplicated")
        observed_workers.add(worker_id)
        behavior_version = completion["behavior_policy_version"]
        if type(behavior_version) is not int or behavior_version < 0:
            raise PrepareTrainingError("worker behavior policy version is invalid")
        episodes = completion["episodes"]
        receipts = completion["replay_shards"]
        if (
            not isinstance(episodes, list)
            or not isinstance(receipts, list)
            or len(episodes) != len(receipts)
            or not receipts
        ):
            raise PrepareTrainingError("worker episode/shard inventory differs")
        replay_root = _regular_directory(worker_dir / "replay", "worker replay directory")
        for episode, receipt_value in zip(episodes, receipts):
            if not isinstance(episode, dict) or not isinstance(receipt_value, dict):
                raise PrepareTrainingError("worker episode or shard row is invalid")
            receipt = dict(receipt_value)
            path = _contained_file(receipt.get("path"), replay_root, "replay shard")
            shard_sha = _validate_sha256(receipt.get("sha256"), "replay shard SHA-256")
            if path in observed_paths or shard_sha in observed_shas:
                raise PrepareTrainingError("replay shard path or SHA-256 is duplicated")
            observed_paths.add(path)
            observed_shas.add(shard_sha)
            arrays, metadata = load_episode_shard(
                path,
                expected_sha256=shard_sha,
                expected_representation_kind=REPRESENTATION_PREFIX_HIDDEN,
            )
            del arrays
            if (
                metadata["run_id"] != run_id
                or metadata["episode_id"] != receipt.get("episode_id")
                or metadata["reset_token"] != receipt.get("reset_token")
                or metadata["row_count"] != receipt.get("row_count")
                or metadata["source_fingerprint"] != receipt.get("source_fingerprint")
                or metadata["behavior_policy_id"]
                != completion["behavior_policy_id"]
                or metadata["behavior_policy_version"] != behavior_version
                or metadata["feature_stage"] != FEATURE_STAGE_PREFIX_HIDDEN
            ):
                raise PrepareTrainingError("worker replay shard receipt differs")
            if episode.get("episode_id") != metadata["episode_id"]:
                raise PrepareTrainingError("worker episode identity differs from shard")
            if episode.get("reset_token") != metadata["reset_token"]:
                raise PrepareTrainingError("worker reset token differs from shard")
            receipt["path"] = str(path)
            shard_receipts.append(receipt)
        worker_evidence.append(
            {
                "worker_id": worker_id,
                "output_dir": str(worker_dir),
                "completion_path": str(completion_path),
                "completion_sha256": sha256_file(completion_path),
                "marker_path": str(marker_path),
                "marker_sha256": sha256_file(marker_path),
                "episode_count": len(receipts),
                "behavior_policy_version": behavior_version,
            }
        )
    if observed_workers != set(range(expected_worker_count)):
        raise PrepareTrainingError("worker ids do not cover the fixed rank range")
    if len(shard_receipts) != expected_episode_count:
        raise PrepareTrainingError("global episode count differs from the fixed budget")
    worker_evidence.sort(key=lambda row: int(row["worker_id"]))
    return shard_receipts, worker_evidence


def aggregate_prefix(args: argparse.Namespace) -> dict[str, object]:
    started = time.time()
    expected_plan_sha = _validate_sha256(
        args.expected_plan_sha256, "expected rollout plan SHA-256"
    )
    worker_count = _positive_int(args.expected_worker_count, "expected worker count")
    episode_count = _positive_int(args.expected_episode_count, "expected episode count")
    if not isinstance(args.run_id, str) or not args.run_id:
        raise PrepareTrainingError("run-id must be non-empty")
    output_dir, final_manifest, relative_manifest, staging = _output_layout(
        args.output_dir, args.output_manifest
    )
    try:
        receipts, workers = _worker_receipts(
            args.worker_output_dir,
            run_id=args.run_id,
            expected_plan_sha256=expected_plan_sha,
            expected_worker_count=worker_count,
            expected_episode_count=episode_count,
        )
        manifest = build_replay_manifest(
            receipts,
            run_id=args.run_id,
            expected_episode_count=episode_count,
        )
        if (
            manifest["representation_kind"] != REPRESENTATION_PREFIX_HIDDEN
            or manifest["feature_stage"] != FEATURE_STAGE_PREFIX_HIDDEN
        ):
            raise PrepareTrainingError("global replay is not Stage-1 prefix replay")
        staged_manifest = staging / relative_manifest
        artifact = save_replay_manifest(staged_manifest, manifest)
        manifest_sha = str(artifact["sha256"])
        receipt = {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "operation": "aggregate-prefix",
            "run_id": args.run_id,
            "rollout_plan_sha256": expected_plan_sha,
            "worker_count": worker_count,
            "episode_count": manifest["completed_episode_count"],
            "transition_count": manifest["transition_count"],
            "behavior_policy_versions": manifest["behavior_policy_versions"],
            "workers": workers,
            "output": {
                "manifest_path": str(final_manifest),
                "manifest_sha256": manifest_sha,
                "source_fingerprint": manifest["source_fingerprint"],
                "representation_kind": manifest["representation_kind"],
            },
            "implementation": _implementation_hashes(include_preparer=True),
            "large_arrays_copied": False,
            "started_unix_s": started,
            "completed_unix_s": time.time(),
        }
        receipt_path, receipt_sha = _publish_package(
            staging=staging,
            output_dir=output_dir,
            operation="aggregate-prefix",
            artifact_kind="replay_manifest",
            artifact_sha256=manifest_sha,
            receipt=receipt,
        )
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    verified = load_replay_manifest(
        final_manifest,
        expected_sha256=manifest_sha,
        expected_source_fingerprint=str(manifest["source_fingerprint"]),
        verify_shards=True,
    )
    return {
        "status": "completed",
        "operation": "aggregate-prefix",
        "replay_manifest": str(final_manifest),
        "replay_manifest_sha256": manifest_sha,
        "source_fingerprint": verified["source_fingerprint"],
        "episode_count": verified["completed_episode_count"],
        "transition_count": verified["transition_count"],
        "receipt": receipt_path,
        "receipt_sha256": receipt_sha,
    }


def _verify_replay_package(
    manifest_value: Path,
    expected_sha256: str,
    completion_marker_value: Path,
    *,
    representation: str,
) -> tuple[Path, dict[str, object], dict[str, object]]:
    manifest_path = _regular_file(manifest_value, "replay manifest")
    marker_path = _regular_file(completion_marker_value, "replay completion marker")
    try:
        manifest_path.relative_to(marker_path.parent)
    except ValueError as error:
        raise PrepareTrainingError(
            "replay manifest escapes its completed package"
        ) from error
    marker = _load_json(marker_path, "replay completion marker")
    if marker.get("status") != "completed":
        raise PrepareTrainingError("replay package is not completed")
    marker_schema = marker.get("schema_name")
    if representation == REPRESENTATION_PREFIX_HIDDEN:
        expected_marker = {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "operation": "aggregate-prefix",
            "artifact_kind": "replay_manifest",
            "artifact_sha256": expected_sha256,
            "receipt_sha256": marker.get("receipt_sha256"),
        }
        if marker != expected_marker:
            raise PrepareTrainingError("prefix replay completion marker differs")
        receipt_path = _regular_file(marker_path.parent / RECEIPT_NAME, "prepare receipt")
        if sha256_file(receipt_path) != _validate_sha256(
            marker["receipt_sha256"], "prepare receipt SHA-256"
        ):
            raise PrepareTrainingError("prefix replay receipt SHA-256 differs")
    elif representation == REPRESENTATION_FROZEN_Z_RL:
        gate_marker_sha = _validate_sha256(
            marker.get("stage1_gate_marker_sha256"),
            "Stage-1 gate marker SHA-256",
        )
        gate_receipt_sha = _validate_sha256(
            marker.get("stage1_gate_receipt_sha256"),
            "Stage-1 gate receipt SHA-256",
        )
        stage1_checkpoint_sha = _validate_sha256(
            marker.get("stage1_checkpoint_sha256"),
            "Stage-1 checkpoint SHA-256",
        )
        if (
            marker_schema != PROMOTION_MARKER_SCHEMA
            or marker.get("schema_version") != PROMOTION_MARKER_VERSION
            or marker.get("output_manifest_sha256") != expected_sha256
        ):
            raise PrepareTrainingError("frozen-z promotion completion marker differs")
        receipt_sha = _validate_sha256(
            marker.get("promotion_receipt_sha256"), "promotion receipt SHA-256"
        )
        receipt_path = _regular_file(
            marker_path.parent / "promotion_receipt.json", "promotion receipt"
        )
        if sha256_file(receipt_path) != receipt_sha:
            raise PrepareTrainingError("promotion receipt SHA-256 differs")
        promotion_receipt = _load_json(receipt_path, "promotion receipt")
        gate = promotion_receipt.get("stage1_gate")
        checkpoint = promotion_receipt.get("stage1_checkpoint")
        promotion_input = promotion_receipt.get("input")
        gate_run_manifest_sha = _validate_sha256(
            gate.get("run_manifest_sha256") if isinstance(gate, dict) else None,
            "Stage-1 gate run manifest SHA-256",
        )
        gate_replay_manifest_sha = _validate_sha256(
            gate.get("replay_manifest_sha256") if isinstance(gate, dict) else None,
            "Stage-1 gate replay manifest SHA-256",
        )
        if (
            promotion_receipt.get("schema_name") != PROMOTION_MARKER_SCHEMA
            or promotion_receipt.get("schema_version") != PROMOTION_MARKER_VERSION
            or promotion_receipt.get("status") != "completed"
            or not isinstance(gate, dict)
            or gate.get("gate_passed") is not True
            or gate.get("marker_sha256") != gate_marker_sha
            or gate.get("receipt_sha256") != gate_receipt_sha
            or gate.get("checkpoint_sha256") != stage1_checkpoint_sha
            or not isinstance(checkpoint, dict)
            or checkpoint.get("sha256") != stage1_checkpoint_sha
            or not isinstance(promotion_input, dict)
            or gate.get("parent_replay_manifest_sha256")
            != promotion_input.get("manifest_sha256")
            or gate.get("source_fingerprint")
            != promotion_input.get("source_fingerprint")
        ):
            raise PrepareTrainingError("promotion Stage-1 gate binding differs")
    else:
        raise PrepareTrainingError("unsupported replay representation")
    manifest = load_replay_manifest(
        manifest_path,
        expected_sha256=expected_sha256,
        verify_shards=True,
    )
    if manifest["representation_kind"] != representation:
        raise PrepareTrainingError("replay representation differs from phase")
    marker_evidence = {
        "path": str(marker_path),
        "sha256": sha256_file(marker_path),
        "schema_name": marker_schema,
    }
    if representation == REPRESENTATION_FROZEN_Z_RL:
        marker_evidence["stage1_checkpoint_sha256"] = stage1_checkpoint_sha
        marker_evidence["stage1_gate_marker_sha256"] = gate_marker_sha
        marker_evidence["stage1_gate_receipt_sha256"] = gate_receipt_sha
        marker_evidence["stage1_run_manifest_sha256"] = gate_run_manifest_sha
        marker_evidence["stage1_replay_manifest_sha256"] = gate_replay_manifest_sha
    return manifest_path, manifest, marker_evidence


def _first_metadata(manifest: Mapping[str, object], representation: str) -> dict[str, object]:
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards or not isinstance(shards[0], dict):
        raise PrepareTrainingError("replay manifest has no shard inventory")
    receipt = shards[0]
    _, metadata = load_episode_shard(
        Path(str(receipt["path"])),
        expected_sha256=str(receipt["sha256"]),
        expected_source_fingerprint=str(manifest["source_fingerprint"]),
        expected_representation_kind=representation,
    )
    return metadata


def _stage1_model(args: argparse.Namespace, metadata: Mapping[str, object]) -> NavigationRLTConfig:
    topology_dim = (
        max(int(metadata["topology_node_dim"]), int(metadata["topology_global_dim"]))
        if args.topology_dim is None
        else args.topology_dim
    )
    return NavigationRLTConfig(
        prefix_hidden_dim=int(metadata["prefix_hidden_dim"]),
        topology_dim=topology_dim,
        reference_dim=REFERENCE_DIM,
        canonical_action_dim=CANONICAL_ACTION_DIM,
        topology_node_dim=int(metadata["topology_node_dim"]),
        topology_global_dim=int(metadata["topology_global_dim"]),
        topology_packing_schema=TOPOLOGY_PACKING_SCHEMA,
        attention_heads=args.attention_heads,
        transformer_layers=args.transformer_layers,
        feedforward_dim=args.feedforward_dim,
        critic_hidden_dim=args.critic_hidden_dim,
        dropout=args.dropout,
        max_prefix_tokens=args.max_prefix_tokens,
        max_artifact_prefix_tokens=int(metadata["prefix_max_tokens"]),
        prefix_selection_schema=PREFIX_SELECTION_SCHEMA,
        max_topology_tokens=int(metadata["topology_max_nodes"]) + 1,
        residual_forward_limit_m=args.residual_forward_limit_m,
        residual_yaw_limit_rad=args.residual_yaw_limit_rad,
        initial_log_std=args.initial_log_std,
        min_log_std=args.min_log_std,
        max_log_std=args.max_log_std,
        strict_frozen_inputs=True,
    )


def _common_training(args: argparse.Namespace) -> dict[str, object]:
    _positive_int(args.epochs, "epochs")
    _positive_int(args.batch_size_per_rank, "batch size per rank")
    _finite(args.learning_rate, "learning rate", positive=True)
    weight_decay = _finite(args.weight_decay, "weight decay")
    if weight_decay < 0.0:
        raise PrepareTrainingError("weight decay must be non-negative")
    _finite(args.clip_grad, "clip grad", positive=True)
    return {
        "epochs": args.epochs,
        "batch_size_per_rank": args.batch_size_per_rank,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "clip_grad": args.clip_grad,
    }


def _stage1_training(args: argparse.Namespace) -> dict[str, object]:
    Stage1LossConfig(
        prefix_mask_ratio=args.prefix_mask_ratio,
        topology_mask_ratio=args.topology_mask_ratio,
        prefix_weight=args.prefix_weight,
        topology_weight=args.topology_weight,
    )
    return {
        **_common_training(args),
        "prefix_mask_ratio": args.prefix_mask_ratio,
        "topology_mask_ratio": args.topology_mask_ratio,
        "prefix_weight": args.prefix_weight,
        "topology_weight": args.topology_weight,
    }


def _stage2_training(args: argparse.Namespace) -> dict[str, object]:
    Stage2LossConfig(
        reference_gate_weight=args.reference_gate_weight,
        reference_residual_weight=args.reference_residual_weight,
        reference_dropout_probability=args.reference_dropout_probability,
        target_tau=args.target_tau,
        cql_alpha=STAGE2_CQL_ALPHA,
        cql_temperature=STAGE2_CQL_TEMPERATURE,
    )
    if args.reference_gate_weight <= 0.0 or args.reference_residual_weight <= 0.0:
        raise PrepareTrainingError("Stage 2 Native-reference anchors must be positive")
    if not 0.0 < args.reference_dropout_probability < 1.0:
        raise PrepareTrainingError("Stage 2 reference dropout must be inside (0,1)")
    _positive_int(args.policy_version, "policy version")
    return {
        **_common_training(args),
        "objective_schema": STAGE2_OBJECTIVE_SCHEMA,
        "behavior_kl_weight": STAGE2_BEHAVIOR_KL_WEIGHT,
        "cql_alpha": STAGE2_CQL_ALPHA,
        "cql_temperature": STAGE2_CQL_TEMPERATURE,
        "minimum_behavior_gate_count": MINIMUM_BEHAVIOR_GATE_COUNT,
        "reference_gate_weight": args.reference_gate_weight,
        "reference_residual_weight": args.reference_residual_weight,
        "reference_dropout_probability": args.reference_dropout_probability,
        "target_tau": args.target_tau,
        "policy_version": args.policy_version,
    }


def _run_manifest(
    *,
    phase: str,
    run_id: str,
    seed: int,
    world_size: int,
    replay_sha256: str,
    replay: Mapping[str, object],
    model: NavigationRLTConfig,
    training: Mapping[str, object],
    checkpoint_sha256: str | None,
    stage2_support: Mapping[str, object] | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "schema_name": RUN_MANIFEST_SCHEMA,
        "schema_version": RUN_MANIFEST_VERSION,
        "run_id": run_id,
        "phase": phase,
        "seed": seed,
        "required_world_size": world_size,
        "boundaries": {
            "upstream_split_read": "train",
            "split_label": "train_update",
            "validation_split_reads": 0,
            "lightnav_source_revision": LIGHTNAV_SOURCE_REVISION,
            "lightnav_frozen": True,
            "ttt_enabled": False,
        },
        "implementation": _implementation_hashes(include_preparer=False),
        "model": {
            "config": model.to_dict(),
            "reference_embedding_schema": REFERENCE_SCHEMA,
            "topology_packing_schema": TOPOLOGY_PACKING_SCHEMA,
            "canonical_action_schema": CANONICAL_ACTION_SCHEMA,
        },
        "replay": {
            "manifest_sha256": replay_sha256,
            "source_fingerprint": replay["source_fingerprint"],
            "representation_kind": replay["representation_kind"],
        },
        "training": dict(training),
    }
    if phase == "stage2":
        if checkpoint_sha256 is None:
            raise PrepareTrainingError("Stage 2 requires a checkpoint SHA-256")
        if stage2_support is None:
            raise PrepareTrainingError("Stage 2 requires behavior gate support evidence")
        result["stage1_checkpoint"] = {"sha256": checkpoint_sha256}
        result["stage2_support"] = dict(stage2_support)
    elif stage2_support is not None:
        raise PrepareTrainingError("Stage 1 cannot carry Stage-2 support evidence")
    return result


def _stage2_checkpoint(
    args: argparse.Namespace,
) -> tuple[Path, str, NavigationRLTPolicy, dict[str, object]]:
    path = _regular_file(args.stage1_checkpoint, "Stage-1 checkpoint")
    expected = _validate_sha256(
        args.expected_stage1_checkpoint_sha256,
        "expected Stage-1 checkpoint SHA-256",
    )
    if sha256_file(path) != expected:
        raise PrepareTrainingError("Stage-1 checkpoint SHA-256 differs")
    policy, metadata = NavigationRLTPolicy.load_checkpoint(path, map_location="cpu")
    extra = metadata.get("extra")
    if (
        not bool(policy.stage1_frozen.item())
        or metadata.get("stage1_frozen") is not True
        or metadata.get("lightnav_backbone_trainable") is not False
        or metadata.get("ttt_enabled") is not False
        or type(metadata.get("policy_version")) is not int
        or metadata.get("policy_version") != 0
        or not isinstance(extra, Mapping)
        or extra.get("training_phase") != "stage1"
    ):
        raise PrepareTrainingError("checkpoint is not an eligible frozen Stage 1")
    return path, expected, policy, metadata


def _stage2_eligible(
    replay: Mapping[str, object], checkpoint_sha256: str, config: NavigationRLTConfig
) -> dict[str, object]:
    eligible = 0
    credit_ineligible = 0
    gate_mismatch = 0
    gate_counts = np.zeros((GATE_COUNT,), dtype=np.int64)
    for receipt in replay["shards"]:
        arrays, metadata = load_episode_shard(
            Path(str(receipt["path"])),
            expected_sha256=str(receipt["sha256"]),
            expected_source_fingerprint=str(replay["source_fingerprint"]),
            expected_representation_kind=REPRESENTATION_FROZEN_Z_RL,
        )
        if metadata["rlt_checkpoint_sha256"] != checkpoint_sha256:
            raise PrepareTrainingError("frozen replay checkpoint binding differs")
        if (
            int(metadata["z_rl_dim"]) != Z_RL_DIM
            or int(metadata["reference_dim"]) != config.reference_dim
            or int(metadata["topology_node_dim"]) != config.topology_node_dim
            or int(metadata["topology_global_dim"]) != config.topology_global_dim
            or int(metadata["topology_max_nodes"]) + 1 != config.max_topology_tokens
        ):
            raise PrepareTrainingError("frozen replay dimensions differ from checkpoint")
        credit = np.asarray(arrays["credit_mask"], dtype=np.bool_)
        requested = np.asarray(arrays["requested_mode"])
        executed = np.asarray(arrays["executed_mode"])
        if (
            requested.ndim != 1
            or executed.shape != requested.shape
            or credit.shape != requested.shape
        ):
            raise PrepareTrainingError("Stage-2 gate/credit arrays have incompatible shapes")
        if bool(
            ((requested < 0) | (requested >= GATE_COUNT)).any()
            or ((executed < 0) | (executed >= GATE_COUNT)).any()
        ):
            raise PrepareTrainingError("Stage-2 gate arrays contain an invalid gate index")
        matched = requested == executed
        eligible_mask = credit & matched
        eligible += int(eligible_mask.sum())
        credit_ineligible += int((~credit).sum())
        gate_mismatch += int((credit & ~matched).sum())
        gate_counts += np.bincount(
            requested[eligible_mask].astype(np.int64), minlength=GATE_COUNT
        )[:GATE_COUNT]
    if eligible < 1:
        raise PrepareTrainingError("Stage-2 replay contains no credit-eligible row")
    support_mask = gate_counts >= MINIMUM_BEHAVIOR_GATE_COUNT
    return {
        "eligibility_schema": STAGE2_ELIGIBILITY_SCHEMA,
        "global_training_items": eligible,
        "credit_ineligible_items_excluded": credit_ineligible,
        "gate_mismatch_items_excluded": gate_mismatch,
        "gate_names": list(GATE_NAMES),
        "gate_counts": [int(value) for value in gate_counts.tolist()],
        "gate_support_mask": [bool(value) for value in support_mask.tolist()],
        "minimum_behavior_gate_count": MINIMUM_BEHAVIOR_GATE_COUNT,
    }


def prepare_run(args: argparse.Namespace) -> dict[str, object]:
    started = time.time()
    phase = args.command
    replay_sha = _validate_sha256(
        args.expected_replay_manifest_sha256,
        "expected replay manifest SHA-256",
    )
    representation = (
        REPRESENTATION_PREFIX_HIDDEN
        if phase == "stage1"
        else REPRESENTATION_FROZEN_Z_RL
    )
    replay_path, replay, marker = _verify_replay_package(
        args.replay_manifest,
        replay_sha,
        args.replay_completion_marker,
        representation=representation,
    )
    if replay["run_id"] != args.expected_run_id:
        raise PrepareTrainingError("replay run_id differs from expected-run-id")
    expected_episode_count = _positive_int(
        args.expected_episode_count, "expected episode count"
    )
    if (
        int(replay["expected_episode_count"]) != expected_episode_count
        or int(replay["completed_episode_count"]) != expected_episode_count
    ):
        raise PrepareTrainingError(
            "replay expected/completed episode count differs from expected-episode-count"
        )
    seed = _positive_int(args.seed, "seed")
    world_size = _positive_int(args.required_world_size, "required world size")
    checkpoint_evidence: dict[str, object] | None = None
    stage2_support: dict[str, object] | None = None
    if phase == "stage1":
        metadata = _first_metadata(replay, REPRESENTATION_PREFIX_HIDDEN)
        model = _stage1_model(args, metadata)
        training = _stage1_training(args)
        if 2 * int(replay["transition_count"]) < world_size:
            raise PrepareTrainingError("Stage-1 items are fewer than distributed ranks")
        data_evidence = {
            "global_training_items": 2 * int(replay["transition_count"]),
            "prefix_max_tokens": metadata["prefix_max_tokens"],
            "prefix_hidden_dim": metadata["prefix_hidden_dim"],
        }
        checkpoint_sha = None
    else:
        checkpoint_path, checkpoint_sha, policy, checkpoint_metadata = _stage2_checkpoint(args)
        model = policy.config
        training = _stage2_training(args)
        stage2_data = _stage2_eligible(replay, checkpoint_sha, model)
        eligible = int(stage2_data["global_training_items"])
        if eligible < world_size:
            raise PrepareTrainingError("Stage-2 eligible rows are fewer than distributed ranks")
        data_evidence = dict(stage2_data)
        stage2_support = {
            "eligibility_schema": stage2_data["eligibility_schema"],
            "gate_names": stage2_data["gate_names"],
            "gate_counts": stage2_data["gate_counts"],
            "gate_support_mask": stage2_data["gate_support_mask"],
            "minimum_behavior_gate_count": stage2_data[
                "minimum_behavior_gate_count"
            ],
        }
        checkpoint_evidence = {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha,
            "training_step": checkpoint_metadata["training_step"],
            "policy_version": checkpoint_metadata["policy_version"],
            "stage1_frozen": True,
        }
    run_manifest = _run_manifest(
        phase=phase,
        run_id=str(replay["run_id"]),
        seed=seed,
        world_size=world_size,
        replay_sha256=replay_sha,
        replay=replay,
        model=model,
        training=training,
        checkpoint_sha256=checkpoint_sha,
        stage2_support=stage2_support,
    )
    output_dir, final_manifest, relative_manifest, staging = _output_layout(
        args.output_dir, args.output_manifest
    )
    try:
        staged_manifest = staging / relative_manifest
        _atomic_json_new(staged_manifest, run_manifest)
        run_sha = sha256_file(staged_manifest)
        receipt = {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "operation": phase,
            "run_id": replay["run_id"],
            "phase": phase,
            "replay": {
                "manifest_path": str(replay_path),
                "manifest_sha256": replay_sha,
                "source_fingerprint": replay["source_fingerprint"],
                "representation_kind": replay["representation_kind"],
                "completion_marker": marker,
                "expected_episode_count": expected_episode_count,
                "episode_count": replay["completed_episode_count"],
                "transition_count": replay["transition_count"],
            },
            "stage1_checkpoint": checkpoint_evidence,
            "model_config": model.to_dict(),
            "training": training,
            "data": data_evidence,
            "output": {
                "run_manifest_path": str(final_manifest),
                "run_manifest_sha256": run_sha,
            },
            "implementation": _implementation_hashes(include_preparer=True),
            "started_unix_s": started,
            "completed_unix_s": time.time(),
        }
        receipt_path, receipt_sha = _publish_package(
            staging=staging,
            output_dir=output_dir,
            operation=phase,
            artifact_kind="run_manifest",
            artifact_sha256=run_sha,
            receipt=receipt,
        )
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    if sha256_file(final_manifest) != run_sha:
        raise PrepareTrainingError("published run manifest SHA-256 differs")
    if _load_json(final_manifest, "published run manifest") != run_manifest:
        raise PrepareTrainingError("published run manifest content differs")
    return {
        "status": "completed",
        "operation": phase,
        "run_manifest": str(final_manifest),
        "run_manifest_sha256": run_sha,
        "replay_manifest_sha256": replay_sha,
        "expected_episode_count": expected_episode_count,
        "receipt": receipt_path,
        "receipt_sha256": receipt_sha,
    }


def _add_output(parser: argparse.ArgumentParser, default_name: str) -> None:
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path(default_name),
        help="JSON path relative to output-dir, or an absolute path inside it",
    )


def _add_replay(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--replay-manifest", required=True, type=Path)
    parser.add_argument("--expected-replay-manifest-sha256", required=True)
    parser.add_argument("--replay-completion-marker", required=True, type=Path)
    parser.add_argument("--expected-run-id", required=True)
    parser.add_argument("--expected-episode-count", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--required-world-size", required=True, type=int)
    parser.add_argument("--epochs", required=True, type=int)
    parser.add_argument("--batch-size-per-rank", required=True, type=int)
    parser.add_argument("--learning-rate", required=True, type=float)
    parser.add_argument("--weight-decay", required=True, type=float)
    parser.add_argument("--clip-grad", required=True, type=float)
    _add_output(parser, "run_manifest.json")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)

    aggregate = commands.add_parser("aggregate-prefix")
    aggregate.add_argument("--worker-output-dir", required=True, type=Path, action="append")
    aggregate.add_argument("--run-id", required=True)
    aggregate.add_argument("--expected-plan-sha256", required=True)
    aggregate.add_argument("--expected-worker-count", required=True, type=int)
    aggregate.add_argument("--expected-episode-count", required=True, type=int)
    _add_output(aggregate, "replay_manifest.json")

    stage1 = commands.add_parser("stage1")
    _add_replay(stage1)
    stage1.add_argument("--max-prefix-tokens", required=True, type=int)
    stage1.add_argument("--topology-dim", type=int)
    stage1.add_argument("--attention-heads", type=int, default=8)
    stage1.add_argument("--transformer-layers", type=int, default=2)
    stage1.add_argument("--feedforward-dim", type=int, default=768)
    stage1.add_argument("--critic-hidden-dim", type=int, default=256)
    stage1.add_argument("--dropout", type=float, default=0.0)
    stage1.add_argument("--residual-forward-limit-m", type=float, default=0.05)
    stage1.add_argument(
        "--residual-yaw-limit-rad", type=float, default=math.radians(10.0)
    )
    stage1.add_argument("--initial-log-std", type=float, default=-1.5)
    stage1.add_argument("--min-log-std", type=float, default=-5.0)
    stage1.add_argument("--max-log-std", type=float, default=1.0)
    stage1.add_argument("--prefix-mask-ratio", required=True, type=float)
    stage1.add_argument("--topology-mask-ratio", required=True, type=float)
    stage1.add_argument("--prefix-weight", required=True, type=float)
    stage1.add_argument("--topology-weight", required=True, type=float)

    stage2 = commands.add_parser("stage2")
    _add_replay(stage2)
    stage2.add_argument("--stage1-checkpoint", required=True, type=Path)
    stage2.add_argument("--expected-stage1-checkpoint-sha256", required=True)
    stage2.add_argument("--reference-gate-weight", required=True, type=float)
    stage2.add_argument("--reference-residual-weight", required=True, type=float)
    stage2.add_argument(
        "--reference-dropout-probability", required=True, type=float
    )
    stage2.add_argument("--target-tau", required=True, type=float)
    stage2.add_argument("--policy-version", required=True, type=int)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    result = aggregate_prefix(args) if args.command == "aggregate-prefix" else prepare_run(args)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        PrepareTrainingError,
        FileExistsError,
        replay_module.ReplayContractError,
        policy_module.NavigationRLTContractError,
    ) as error:
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
