#!/usr/bin/env python3
"""Build and evaluate the episode-held-out Navigation-RLT Stage-1 gate.

``split`` verifies the fixed 256-episode prefix replay, ranks whole episode
identities with a content-bound hash, and publishes exact 224/32 train/dev
subset manifests.  Shards are referenced in place, so no replay arrays are
copied.  Each subset also carries the strict completion marker accepted by
``navigation_rlt_prepare_training.py``.

``evaluate`` verifies the split and a frozen Stage-1 checkpoint, derives a
feature-wise constant-mean baseline only from representation-train episodes,
and measures deterministic masked reconstruction on representation-dev.
Neither command reads a navigation evaluation split.
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
from typing import Iterator, Mapping, Sequence

import numpy as np
import torch

import navigation_rlt_policy as policy_module
import navigation_rlt_replay as replay_module
from navigation_rlt_policy import (
    NavigationRLTConfig,
    NavigationRLTContext,
    NavigationRLTPolicy,
    Z_RL_DIM,
    pack_topology_tokens,
    select_prefix_tokens,
)
from navigation_rlt_replay import (
    FEATURE_STAGE_PREFIX_HIDDEN,
    REFERENCE_DIM,
    REFERENCE_SCHEMA,
    REPRESENTATION_PREFIX_HIDDEN,
    build_replay_manifest,
    load_episode_shard,
    load_replay_manifest,
    save_replay_manifest,
    sha256_file,
)


SCHEMA_NAME = "navigation_rlt_stage1_gate"
SCHEMA_VERSION = 1
PREPARE_SCHEMA_NAME = "navigation_rlt_prepare_training"
PREPARE_SCHEMA_VERSION = 1
EXPECTED_EPISODES = 256
TRAIN_EPISODES = 224
DEV_EPISODES = 32
SPLIT_ALGORITHM = "sha256_parent_manifest_episode_identity_rank_v1"
MASK_ALGORITHM = "sha256_state_identity_uniform_without_replacement_v1"
RECEIPT_NAME = "stage1_gate_receipt.json"
COMPLETED_NAME = "COMPLETED.json"
PREPARE_RECEIPT_NAME = "prepare_receipt.json"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class Stage1GateError(RuntimeError):
    """Raised when Stage-1 evidence violates the fixed gate contract."""


def _validate_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise Stage1GateError(f"{name} must be a lowercase SHA-256")
    return value


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value < 1:
        raise Stage1GateError(f"{name} must be a positive integer")
    return int(value)


def _ratio(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Stage1GateError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result <= 1.0:
        raise Stage1GateError(f"{name} must be finite and in (0, 1]")
    return result


def _nonnegative_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Stage1GateError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise Stage1GateError(f"{name} must be finite and non-negative")
    return result


def _regular_file(path: Path, name: str) -> Path:
    raw = Path(path)
    if raw.is_symlink():
        raise Stage1GateError(f"{name} must not be a symbolic link")
    try:
        result = raw.resolve(strict=True)
    except OSError as error:
        raise Stage1GateError(f"{name} is missing or inaccessible") from error
    if not result.is_file():
        raise Stage1GateError(f"{name} must be a regular file")
    return result


def _regular_directory(path: Path, name: str) -> Path:
    raw = Path(path)
    if raw.is_symlink():
        raise Stage1GateError(f"{name} must not be a symbolic link")
    try:
        result = raw.resolve(strict=True)
    except OSError as error:
        raise Stage1GateError(f"{name} is missing or inaccessible") from error
    if not result.is_dir():
        raise Stage1GateError(f"{name} must be a directory")
    return result


def _contained_file(value: object, root: Path, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise Stage1GateError(f"{name} path must be non-empty")
    path = _regular_file(Path(value), name)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise Stage1GateError(f"{name} escapes the split package") from error
    return path


def _load_json(path: Path, name: str) -> dict[str, object]:
    path = _regular_file(path, name)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Stage1GateError(f"{name} is not valid JSON") from error
    if not isinstance(value, dict):
        raise Stage1GateError(f"{name} must contain a JSON object")
    return value


def _json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError) as error:
        raise Stage1GateError("output is not strict JSON") from error


def _atomic_json_new(path: Path, value: object) -> None:
    path = Path(path)
    if os.path.lexists(path):
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".staging-", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        if os.path.lexists(path):
            raise FileExistsError(path)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _new_staging(output_dir_value: Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir_value).resolve(strict=False)
    if os.path.lexists(output_dir):
        raise FileExistsError(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=output_dir.name + ".staging-", dir=output_dir.parent)
    )
    return output_dir, staging


def _publish_directory(staging: Path, output_dir: Path) -> None:
    if os.path.lexists(output_dir):
        raise FileExistsError(output_dir)
    os.replace(staging, output_dir)


def _implementation_hashes() -> dict[str, str]:
    return {
        "stage1_gate_sha256": sha256_file(Path(__file__).resolve()),
        "policy_sha256": sha256_file(Path(policy_module.__file__).resolve()),
        "replay_sha256": sha256_file(Path(replay_module.__file__).resolve()),
    }


def _identity(receipt: Mapping[str, object]) -> tuple[str, str]:
    episode = receipt.get("episode_id")
    reset = receipt.get("reset_token")
    if not isinstance(episode, str) or not episode or not isinstance(reset, str) or not reset:
        raise Stage1GateError("episode identity is invalid")
    return episode, reset


def _identity_digest(receipts: Sequence[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for receipt in sorted(receipts, key=_identity):
        episode, reset = _identity(receipt)
        digest.update(episode.encode("utf-8"))
        digest.update(b"\0")
        digest.update(reset.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _verify_prefix_manifest(
    path_value: Path,
    expected_sha256: str,
    expected_source_fingerprint: str,
    *,
    expected_episodes: int,
) -> tuple[Path, dict[str, object]]:
    path = _regular_file(path_value, "prefix replay manifest")
    expected_sha256 = _validate_sha256(expected_sha256, "replay manifest SHA-256")
    source = _validate_sha256(expected_source_fingerprint, "source fingerprint")
    manifest = load_replay_manifest(
        path,
        expected_sha256=expected_sha256,
        expected_source_fingerprint=source,
        verify_shards=True,
    )
    if (
        manifest["representation_kind"] != REPRESENTATION_PREFIX_HIDDEN
        or manifest["feature_stage"] != FEATURE_STAGE_PREFIX_HIDDEN
        or manifest["split_label"] != "train_update"
        or manifest["validation_split_reads"] != 0
    ):
        raise Stage1GateError("replay is not an eligible train-update prefix replay")
    if (
        int(manifest["expected_episode_count"]) != expected_episodes
        or int(manifest["completed_episode_count"]) != expected_episodes
        or len(manifest["shards"]) != expected_episodes
    ):
        raise Stage1GateError("replay episode count differs from the fixed gate budget")
    return path, manifest


def _split_rank(
    receipt: Mapping[str, object], parent_manifest_sha256: str
) -> tuple[str, str, str]:
    episode, reset = _identity(receipt)
    payload = "\0".join(
        (SPLIT_ALGORITHM, parent_manifest_sha256, episode, reset)
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), episode, reset


def _write_subset_package(
    *,
    staged_root: Path,
    final_root: Path,
    label: str,
    manifest: Mapping[str, object],
    parent_manifest_path: Path,
    parent_manifest_sha256: str,
    identity_digest: str,
) -> dict[str, object]:
    staged_package = staged_root / label
    final_package = final_root / label
    temporary_manifest = staged_package / "manifests" / "replay_manifest.json"
    artifact = save_replay_manifest(temporary_manifest, manifest)
    manifest_sha = str(artifact["sha256"])
    final_name = f"replay_manifest.{manifest_sha}.json"
    staged_manifest = temporary_manifest.with_name(final_name)
    temporary_manifest.replace(staged_manifest)
    final_manifest = final_package / "manifests" / final_name
    compatibility_receipt = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "operation": "episode-heldout-subset",
        "subset": label,
        "parent_replay_manifest": {
            "path": str(parent_manifest_path),
            "sha256": parent_manifest_sha256,
        },
        "episode_identity_sha256": identity_digest,
        "episode_count": manifest["completed_episode_count"],
        "transition_count": manifest["transition_count"],
        "source_fingerprint": manifest["source_fingerprint"],
        "output": {
            "manifest_path": str(final_manifest),
            "manifest_sha256": manifest_sha,
        },
        "large_arrays_copied": False,
        "prepare_compatibility": {
            "schema_name": PREPARE_SCHEMA_NAME,
            "schema_version": PREPARE_SCHEMA_VERSION,
            "accepted_operation": "aggregate-prefix",
        },
    }
    prepare_receipt = staged_package / PREPARE_RECEIPT_NAME
    _atomic_json_new(prepare_receipt, compatibility_receipt)
    prepare_receipt_sha = sha256_file(prepare_receipt)
    marker = {
        "schema_name": PREPARE_SCHEMA_NAME,
        "schema_version": PREPARE_SCHEMA_VERSION,
        "status": "completed",
        "operation": "aggregate-prefix",
        "artifact_kind": "replay_manifest",
        "artifact_sha256": manifest_sha,
        "receipt_sha256": prepare_receipt_sha,
    }
    marker_path = staged_package / COMPLETED_NAME
    _atomic_json_new(marker_path, marker)
    return {
        "label": label,
        "manifest_path": str(final_manifest),
        "manifest_sha256": manifest_sha,
        "completion_marker_path": str(final_package / COMPLETED_NAME),
        "completion_marker_sha256": sha256_file(marker_path),
        "prepare_receipt_path": str(final_package / PREPARE_RECEIPT_NAME),
        "prepare_receipt_sha256": prepare_receipt_sha,
        "episode_count": manifest["completed_episode_count"],
        "transition_count": manifest["transition_count"],
        "episode_identity_sha256": identity_digest,
    }


def split_replay(args: argparse.Namespace) -> dict[str, object]:
    started = time.time()
    parent_sha = _validate_sha256(
        args.expected_replay_manifest_sha256, "replay manifest SHA-256"
    )
    source = _validate_sha256(args.expected_source_fingerprint, "source fingerprint")
    parent_path, parent = _verify_prefix_manifest(
        args.replay_manifest,
        parent_sha,
        source,
        expected_episodes=EXPECTED_EPISODES,
    )
    ranked = sorted(parent["shards"], key=lambda row: _split_rank(row, parent_sha))
    dev_receipts = [dict(row) for row in ranked[:DEV_EPISODES]]
    train_receipts = [dict(row) for row in ranked[DEV_EPISODES:]]
    if len(train_receipts) != TRAIN_EPISODES or len(dev_receipts) != DEV_EPISODES:
        raise Stage1GateError("deterministic episode split cardinality differs")
    train_ids = {_identity(row) for row in train_receipts}
    dev_ids = {_identity(row) for row in dev_receipts}
    parent_ids = {_identity(row) for row in parent["shards"]}
    if train_ids & dev_ids or train_ids | dev_ids != parent_ids:
        raise Stage1GateError("episode split leaks or omits an episode")

    train_manifest = build_replay_manifest(
        train_receipts,
        run_id=str(parent["run_id"]),
        expected_episode_count=TRAIN_EPISODES,
        expected_source_fingerprint=source,
    )
    dev_manifest = build_replay_manifest(
        dev_receipts,
        run_id=str(parent["run_id"]),
        expected_episode_count=DEV_EPISODES,
        expected_source_fingerprint=source,
    )
    output_dir, staging = _new_staging(args.output_dir)
    try:
        train_output = _write_subset_package(
            staged_root=staging,
            final_root=output_dir,
            label="representation-train",
            manifest=train_manifest,
            parent_manifest_path=parent_path,
            parent_manifest_sha256=parent_sha,
            identity_digest=_identity_digest(train_receipts),
        )
        dev_output = _write_subset_package(
            staged_root=staging,
            final_root=output_dir,
            label="representation-dev",
            manifest=dev_manifest,
            parent_manifest_path=parent_path,
            parent_manifest_sha256=parent_sha,
            identity_digest=_identity_digest(dev_receipts),
        )
        receipt = {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "operation": "split",
            "split_algorithm": SPLIT_ALGORITHM,
            "parent_replay_manifest": {
                "path": str(parent_path),
                "sha256": parent_sha,
                "run_id": parent["run_id"],
                "source_fingerprint": source,
                "episode_count": parent["completed_episode_count"],
                "transition_count": parent["transition_count"],
                "episode_identity_sha256": _identity_digest(parent["shards"]),
            },
            "subsets": {
                "representation_train": train_output,
                "representation_dev": dev_output,
            },
            "episode_overlap_count": 0,
            "episode_union_count": len(train_ids | dev_ids),
            "large_arrays_copied": False,
            "implementation": _implementation_hashes(),
            "started_unix_s": started,
            "completed_unix_s": time.time(),
        }
        receipt_path = staging / RECEIPT_NAME
        _atomic_json_new(receipt_path, receipt)
        receipt_sha = sha256_file(receipt_path)
        marker = {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "operation": "split",
            "receipt_sha256": receipt_sha,
            "train_manifest_sha256": train_output["manifest_sha256"],
            "dev_manifest_sha256": dev_output["manifest_sha256"],
        }
        _atomic_json_new(staging / COMPLETED_NAME, marker)
        _publish_directory(staging, output_dir)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    published_receipt = _regular_file(output_dir / RECEIPT_NAME, "split receipt")
    if sha256_file(published_receipt) != receipt_sha:
        raise Stage1GateError("published split receipt SHA-256 differs")
    for subset, expected_count in (
        (train_output, TRAIN_EPISODES),
        (dev_output, DEV_EPISODES),
    ):
        verified = load_replay_manifest(
            Path(str(subset["manifest_path"])),
            expected_sha256=str(subset["manifest_sha256"]),
            expected_source_fingerprint=source,
            verify_shards=True,
        )
        if int(verified["completed_episode_count"]) != expected_count:
            raise Stage1GateError("published subset episode count differs")
    return {
        "status": "completed",
        "operation": "split",
        "output_dir": str(output_dir),
        "receipt": str(published_receipt),
        "receipt_sha256": receipt_sha,
        "train_manifest": train_output["manifest_path"],
        "train_manifest_sha256": train_output["manifest_sha256"],
        "dev_manifest": dev_output["manifest_path"],
        "dev_manifest_sha256": dev_output["manifest_sha256"],
        "train_episodes": TRAIN_EPISODES,
        "dev_episodes": DEV_EPISODES,
    }


def _verify_split_package(
    split_dir_value: Path,
    expected_receipt_sha256: str,
    expected_parent_sha256: str,
    expected_source_fingerprint: str,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    root = _regular_directory(split_dir_value, "split output directory")
    receipt_path = _contained_file(str(root / RECEIPT_NAME), root, "split receipt")
    expected_receipt_sha256 = _validate_sha256(
        expected_receipt_sha256, "split receipt SHA-256"
    )
    if sha256_file(receipt_path) != expected_receipt_sha256:
        raise Stage1GateError("split receipt SHA-256 differs")
    receipt = _load_json(receipt_path, "split receipt")
    if (
        receipt.get("schema_name") != SCHEMA_NAME
        or receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("status") != "completed"
        or receipt.get("operation") != "split"
        or receipt.get("split_algorithm") != SPLIT_ALGORITHM
        or receipt.get("episode_overlap_count") != 0
        or receipt.get("episode_union_count") != EXPECTED_EPISODES
        or receipt.get("large_arrays_copied") is not False
        or receipt.get("implementation") != _implementation_hashes()
    ):
        raise Stage1GateError("split receipt contract differs")
    marker_path = _contained_file(str(root / COMPLETED_NAME), root, "split marker")
    marker = _load_json(marker_path, "split marker")
    if (
        marker.get("schema_name") != SCHEMA_NAME
        or marker.get("schema_version") != SCHEMA_VERSION
        or marker.get("status") != "completed"
        or marker.get("operation") != "split"
        or marker.get("receipt_sha256") != expected_receipt_sha256
    ):
        raise Stage1GateError("split completion marker differs")
    parent = receipt.get("parent_replay_manifest")
    if not isinstance(parent, dict):
        raise Stage1GateError("split parent evidence is missing")
    parent_sha = _validate_sha256(
        expected_parent_sha256, "parent replay manifest SHA-256"
    )
    source = _validate_sha256(expected_source_fingerprint, "source fingerprint")
    if (
        parent.get("sha256") != parent_sha
        or parent.get("source_fingerprint") != source
        or parent.get("episode_count") != EXPECTED_EPISODES
    ):
        raise Stage1GateError("split parent source/hash evidence differs")
    parent_path = _regular_file(
        Path(str(parent.get("path"))), "parent replay manifest"
    )
    parent_manifest = load_replay_manifest(
        parent_path,
        expected_sha256=parent_sha,
        expected_source_fingerprint=source,
        verify_shards=True,
    )
    if (
        int(parent_manifest["completed_episode_count"]) != EXPECTED_EPISODES
        or _identity_digest(parent_manifest["shards"])
        != parent.get("episode_identity_sha256")
    ):
        raise Stage1GateError("split parent episode inventory differs")
    subsets = receipt.get("subsets")
    if not isinstance(subsets, dict) or set(subsets) != {
        "representation_train",
        "representation_dev",
    }:
        raise Stage1GateError("split subset inventory differs")

    loaded: dict[str, dict[str, object]] = {}
    for key, expected_count in (
        ("representation_train", TRAIN_EPISODES),
        ("representation_dev", DEV_EPISODES),
    ):
        evidence = subsets.get(key)
        if not isinstance(evidence, dict):
            raise Stage1GateError(f"{key} evidence is missing")
        path = _contained_file(evidence.get("manifest_path"), root, f"{key} manifest")
        manifest_sha = _validate_sha256(
            evidence.get("manifest_sha256"), f"{key} manifest SHA-256"
        )
        manifest = load_replay_manifest(
            path,
            expected_sha256=manifest_sha,
            expected_source_fingerprint=source,
            verify_shards=True,
        )
        if (
            manifest["representation_kind"] != REPRESENTATION_PREFIX_HIDDEN
            or manifest["feature_stage"] != FEATURE_STAGE_PREFIX_HIDDEN
            or manifest["split_label"] != "train_update"
            or manifest["validation_split_reads"] != 0
            or int(manifest["completed_episode_count"]) != expected_count
            or _identity_digest(manifest["shards"])
            != evidence.get("episode_identity_sha256")
        ):
            raise Stage1GateError(f"{key} manifest contract differs")
        loaded[key] = manifest
    train_ids = {_identity(row) for row in loaded["representation_train"]["shards"]}
    dev_ids = {_identity(row) for row in loaded["representation_dev"]["shards"]}
    ranked = sorted(
        parent_manifest["shards"], key=lambda row: _split_rank(row, parent_sha)
    )
    expected_dev = {_identity(row) for row in ranked[:DEV_EPISODES]}
    expected_train = {_identity(row) for row in ranked[DEV_EPISODES:]}
    if (
        train_ids & dev_ids
        or len(train_ids | dev_ids) != EXPECTED_EPISODES
        or train_ids != expected_train
        or dev_ids != expected_dev
    ):
        raise Stage1GateError("published split has episode leakage or omission")
    return receipt, loaded["representation_train"], loaded["representation_dev"]


def _check_shard_metadata(
    metadata: Mapping[str, object], config: NavigationRLTConfig
) -> None:
    if (
        int(metadata["prefix_hidden_dim"]) != config.prefix_hidden_dim
        or int(metadata["prefix_max_tokens"]) > config.max_artifact_prefix_tokens
        or int(metadata["topology_node_dim"]) != config.topology_node_dim
        or int(metadata["topology_global_dim"]) != config.topology_global_dim
        or int(metadata["topology_max_nodes"]) + 1 != config.max_topology_tokens
        or metadata["reference_schema"] != REFERENCE_SCHEMA
        or int(metadata["reference_dim"]) != REFERENCE_DIM
        or config.reference_dim != REFERENCE_DIM
    ):
        raise Stage1GateError("replay dimensions/schema differ from the checkpoint")


def _state_batches(
    manifest: Mapping[str, object],
    config: NavigationRLTConfig,
    batch_size: int,
) -> Iterator[tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    source = str(manifest["source_fingerprint"])
    for receipt in manifest["shards"]:
        arrays, metadata = load_episode_shard(
            Path(str(receipt["path"])),
            expected_sha256=str(receipt["sha256"]),
            expected_source_fingerprint=source,
            expected_representation_kind=REPRESENTATION_PREFIX_HIDDEN,
        )
        _check_shard_metadata(metadata, config)
        episode, reset = _identity(receipt)
        rows = int(metadata["row_count"])
        for array_prefix, state_name in (("", "current"), ("next_", "next")):
            hidden = torch.from_numpy(
                np.asarray(arrays[array_prefix + "prefix_hidden"]).astype(
                    np.float32, copy=False
                )
            )
            hidden_mask = torch.from_numpy(
                np.asarray(arrays[array_prefix + "prefix_mask"]).astype(
                    np.bool_, copy=False
                )
            )
            selected = select_prefix_tokens(hidden, hidden_mask, config=config)
            topology = pack_topology_tokens(
                torch.from_numpy(
                    np.asarray(arrays[array_prefix + "topology_node_targets"]).astype(
                        np.float32, copy=False
                    )
                ),
                torch.from_numpy(
                    np.asarray(arrays[array_prefix + "topology_node_mask"]).astype(
                        np.bool_, copy=False
                    )
                ),
                torch.from_numpy(
                    np.asarray(arrays[array_prefix + "topology_global_targets"]).astype(
                        np.float32, copy=False
                    )
                ),
                torch.from_numpy(
                    np.asarray(arrays[array_prefix + "topology_global_mask"]).astype(
                        np.bool_, copy=False
                    )
                ),
                config=config,
            )
            for start in range(0, rows, batch_size):
                stop = min(start + batch_size, rows)
                keys = [
                    f"{episode}\0{reset}\0{state_name}\0{index}"
                    for index in range(start, stop)
                ]
                yield (
                    keys,
                    selected.contextual_prefix_hidden[start:stop],
                    selected.prefix_mask[start:stop],
                    topology.tokens[start:stop],
                    topology.token_mask[start:stop],
                )


def _train_means(
    manifest: Mapping[str, object], config: NavigationRLTConfig, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    prefix_sum = torch.zeros(config.prefix_hidden_dim, dtype=torch.float64)
    topology_sum = torch.zeros(config.topology_dim, dtype=torch.float64)
    prefix_count = 0
    topology_count = 0
    state_count = 0
    for _, prefix, prefix_mask, topology, topology_mask in _state_batches(
        manifest, config, batch_size
    ):
        prefix_sum += prefix[prefix_mask].to(torch.float64).sum(dim=0)
        topology_sum += topology[topology_mask].to(torch.float64).sum(dim=0)
        prefix_count += int(prefix_mask.sum().item())
        topology_count += int(topology_mask.sum().item())
        state_count += int(prefix.shape[0])
    if prefix_count < 1 or topology_count < 1 or state_count < 1:
        raise Stage1GateError("representation-train contains no valid token state")
    prefix_mean = (prefix_sum / prefix_count).to(torch.float32)
    topology_mean = (topology_sum / topology_count).to(torch.float32)
    if not bool(torch.isfinite(prefix_mean).all()) or not bool(
        torch.isfinite(topology_mean).all()
    ):
        raise Stage1GateError("train-set constant means are non-finite")
    return prefix_mean, topology_mean, {
        "state_count": state_count,
        "prefix_token_count": prefix_count,
        "topology_token_count": topology_count,
    }


def _reconstruction_mask(
    valid: torch.Tensor,
    ratio: float,
    keys: Sequence[str],
    *,
    seed: int,
    modality: str,
) -> torch.Tensor:
    if valid.device.type != "cpu" or valid.dtype != torch.bool or valid.ndim != 2:
        raise Stage1GateError("mask construction requires a CPU boolean matrix")
    if len(keys) != valid.shape[0]:
        raise Stage1GateError("mask identity count differs from the batch")
    result = torch.zeros_like(valid)
    for row, key in enumerate(keys):
        indices = torch.nonzero(valid[row], as_tuple=False).flatten().numpy()
        count = int(indices.size)
        if count < 1:
            raise Stage1GateError("reconstruction mask encountered an empty row")
        masked = max(1, int(math.ceil(count * ratio)))
        if count > 1:
            masked = min(masked, count - 1)
        digest = hashlib.sha256(
            "\0".join((MASK_ALGORITHM, str(seed), modality, key)).encode("utf-8")
        ).digest()
        generator = np.random.default_rng(int.from_bytes(digest[:16], "big"))
        selected = generator.choice(indices, size=masked, replace=False)
        result[row, torch.from_numpy(np.asarray(selected, dtype=np.int64))] = True
    return result


def _device(value: str) -> torch.device:
    try:
        result = torch.device(value)
    except (TypeError, RuntimeError) as error:
        raise Stage1GateError("device is invalid") from error
    if result.type not in {"cpu", "cuda"}:
        raise Stage1GateError("device must be cpu or cuda")
    if result.type == "cuda" and not torch.cuda.is_available():
        raise Stage1GateError("CUDA was requested but is unavailable")
    return result


def _evaluate_dev(
    manifest: Mapping[str, object],
    policy: NavigationRLTPolicy,
    prefix_mean: torch.Tensor,
    topology_mean: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    prefix_mask_ratio: float,
    topology_mask_ratio: float,
    seed: int,
) -> dict[str, object]:
    model_prefix_sum = 0.0
    model_topology_sum = 0.0
    baseline_prefix_sum = 0.0
    baseline_topology_sum = 0.0
    prefix_masked = 0
    topology_masked = 0
    state_count = 0
    z_sum = torch.zeros(Z_RL_DIM, dtype=torch.float64)
    z_square_sum = torch.zeros(Z_RL_DIM, dtype=torch.float64)
    prefix_mean_device = prefix_mean.to(device=device)
    topology_mean_device = topology_mean.to(device=device)
    policy.eval()
    with torch.no_grad():
        for keys, prefix, prefix_valid, topology, topology_valid in _state_batches(
            manifest, policy.config, batch_size
        ):
            prefix_reconstruction_mask = _reconstruction_mask(
                prefix_valid,
                prefix_mask_ratio,
                keys,
                seed=seed,
                modality="prefix",
            )
            topology_reconstruction_mask = _reconstruction_mask(
                topology_valid,
                topology_mask_ratio,
                keys,
                seed=seed,
                modality="topology",
            )
            context = NavigationRLTContext(
                contextual_prefix_hidden=prefix.to(device=device),
                prefix_mask=prefix_valid.to(device=device),
                topology_tokens=topology.to(device=device),
                topology_mask=topology_valid.to(device=device),
            )
            unmasked_z = policy.encode(context).float()
            representation = policy.stage1(
                context,
                prefix_reconstruction_mask=prefix_reconstruction_mask.to(device=device),
                topology_reconstruction_mask=topology_reconstruction_mask.to(
                    device=device
                ),
            )
            prefix_prediction = representation.prefix_reconstruction
            topology_prediction = representation.topology_reconstruction
            if prefix_prediction is None or topology_prediction is None:
                raise Stage1GateError("checkpoint returned no reconstruction")
            prefix_target = context.contextual_prefix_hidden
            topology_target = context.topology_tokens
            prefix_rows = (prefix_prediction - prefix_target).square().mean(dim=-1)
            topology_rows = (
                topology_prediction - topology_target
            ).square().mean(dim=-1)
            baseline_prefix_rows = (
                prefix_target - prefix_mean_device.reshape(1, 1, -1)
            ).square().mean(dim=-1)
            baseline_topology_rows = (
                topology_target - topology_mean_device.reshape(1, 1, -1)
            ).square().mean(dim=-1)
            prefix_selected = prefix_reconstruction_mask.to(device=device)
            topology_selected = topology_reconstruction_mask.to(device=device)
            model_prefix_sum += float(prefix_rows[prefix_selected].double().sum().item())
            model_topology_sum += float(
                topology_rows[topology_selected].double().sum().item()
            )
            baseline_prefix_sum += float(
                baseline_prefix_rows[prefix_selected].double().sum().item()
            )
            baseline_topology_sum += float(
                baseline_topology_rows[topology_selected].double().sum().item()
            )
            prefix_masked += int(prefix_selected.sum().item())
            topology_masked += int(topology_selected.sum().item())
            z_cpu = unmasked_z.to(device="cpu", dtype=torch.float64)
            z_sum += z_cpu.sum(dim=0)
            z_square_sum += z_cpu.square().sum(dim=0)
            state_count += int(z_cpu.shape[0])
    if min(prefix_masked, topology_masked, state_count) < 1:
        raise Stage1GateError("representation-dev produced no evaluable state")
    variance = z_square_sum / state_count - (z_sum / state_count).square()
    if float(variance.min().item()) < -1.0e-10:
        raise Stage1GateError("z_rl variance accumulator became invalid")
    variance = variance.clamp_min(0.0)
    model_prefix = model_prefix_sum / prefix_masked
    model_topology = model_topology_sum / topology_masked
    baseline_prefix = baseline_prefix_sum / prefix_masked
    baseline_topology = baseline_topology_sum / topology_masked
    total_masked = prefix_masked + topology_masked
    model_combined = (model_prefix_sum + model_topology_sum) / total_masked
    baseline_combined = (baseline_prefix_sum + baseline_topology_sum) / total_masked
    values = (
        model_prefix,
        model_topology,
        model_combined,
        baseline_prefix,
        baseline_topology,
        baseline_combined,
        float(variance.mean().item()),
    )
    if not all(math.isfinite(value) for value in values):
        raise Stage1GateError("Stage-1 gate produced NaN or infinity")
    return {
        "state_count": state_count,
        "masked_token_count": {
            "prefix": prefix_masked,
            "topology": topology_masked,
            "combined": total_masked,
        },
        "checkpoint_masked_mse": {
            "prefix": model_prefix,
            "topology": model_topology,
            "combined": model_combined,
        },
        "train_constant_mean_masked_mse": {
            "prefix": baseline_prefix,
            "topology": baseline_topology,
            "combined": baseline_combined,
        },
        "combined_relative_improvement": (
            (baseline_combined - model_combined) / baseline_combined
            if baseline_combined > 0.0
            else -1.0
        ),
        "z_rl_variance": {
            "mean": float(variance.mean().item()),
            "minimum": float(variance.min().item()),
            "maximum": float(variance.max().item()),
            "finite": bool(torch.isfinite(variance).all()),
        },
    }


def evaluate_checkpoint(args: argparse.Namespace) -> dict[str, object]:
    started = time.time()
    source = _validate_sha256(args.expected_source_fingerprint, "source fingerprint")
    parent_sha = _validate_sha256(
        args.expected_parent_replay_manifest_sha256,
        "parent replay manifest SHA-256",
    )
    split_receipt, train_manifest, dev_manifest = _verify_split_package(
        args.split_output_dir,
        args.expected_split_receipt_sha256,
        parent_sha,
        source,
    )
    stage1_run_sha = _validate_sha256(
        args.expected_stage1_run_manifest_sha256,
        "Stage-1 run manifest SHA-256",
    )
    stage1_replay_sha = _validate_sha256(
        args.expected_stage1_replay_manifest_sha256,
        "Stage-1 replay manifest SHA-256",
    )
    if (
        split_receipt["subsets"]["representation_train"]["manifest_sha256"]
        != stage1_replay_sha
    ):
        raise Stage1GateError(
            "Stage-1 replay manifest is not the fixed representation-train subset"
        )
    checkpoint_path = _regular_file(args.stage1_checkpoint, "Stage-1 checkpoint")
    checkpoint_sha = _validate_sha256(
        args.expected_stage1_checkpoint_sha256, "Stage-1 checkpoint SHA-256"
    )
    if sha256_file(checkpoint_path) != checkpoint_sha:
        raise Stage1GateError("Stage-1 checkpoint SHA-256 differs")
    device = _device(args.device)
    policy, checkpoint_metadata = NavigationRLTPolicy.load_checkpoint(
        checkpoint_path, map_location=device
    )
    extra = checkpoint_metadata.get("extra")
    if (
        not bool(policy.stage1_frozen.item())
        or checkpoint_metadata.get("stage1_frozen") is not True
        or checkpoint_metadata.get("lightnav_backbone_trainable") is not False
        or checkpoint_metadata.get("ttt_enabled") is not False
        or type(checkpoint_metadata.get("training_step")) is not int
        or int(checkpoint_metadata["training_step"]) < 1
        or not isinstance(extra, dict)
        or extra.get("training_phase") != "stage1"
        or extra.get("run_manifest_sha256") != stage1_run_sha
        or extra.get("replay_manifest_sha256") != stage1_replay_sha
    ):
        raise Stage1GateError(
            "checkpoint is not the completed frozen Stage 1 for this run/replay"
        )
    policy.to(device)
    batch_size = _positive_int(args.batch_size, "batch size")
    prefix_ratio = _ratio(args.prefix_mask_ratio, "prefix mask ratio")
    topology_ratio = _ratio(args.topology_mask_ratio, "topology mask ratio")
    minimum_improvement = _nonnegative_finite(
        args.minimum_relative_improvement, "minimum relative improvement"
    )
    minimum_variance = _nonnegative_finite(
        args.minimum_z_variance, "minimum z_rl variance"
    )
    if type(args.seed) is not int or args.seed < 0:
        raise Stage1GateError("seed must be a non-negative integer")

    prefix_mean, topology_mean, train_counts = _train_means(
        train_manifest, policy.config, batch_size
    )
    evaluation = _evaluate_dev(
        dev_manifest,
        policy,
        prefix_mean,
        topology_mean,
        device=device,
        batch_size=batch_size,
        prefix_mask_ratio=prefix_ratio,
        topology_mask_ratio=topology_ratio,
        seed=args.seed,
    )
    baseline_combined = float(
        evaluation["train_constant_mean_masked_mse"]["combined"]
    )
    checkpoint_combined = float(evaluation["checkpoint_masked_mse"]["combined"])
    relative_improvement = float(evaluation["combined_relative_improvement"])
    variance_mean = float(evaluation["z_rl_variance"]["mean"])
    reconstruction_passed = (
        baseline_combined > 0.0
        and checkpoint_combined < baseline_combined
        and relative_improvement >= minimum_improvement
    )
    variance_passed = (
        bool(evaluation["z_rl_variance"]["finite"])
        and variance_mean > minimum_variance
    )
    passed = reconstruction_passed and variance_passed

    output_dir, staging = _new_staging(args.output_dir)
    try:
        receipt = {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "status": "passed" if passed else "rejected",
            "operation": "evaluate",
            "gate": "episode-heldout-stage1-validity",
            "split": {
                "receipt_path": str(
                    Path(args.split_output_dir).resolve() / RECEIPT_NAME
                ),
                "receipt_sha256": _validate_sha256(
                    args.expected_split_receipt_sha256, "split receipt SHA-256"
                ),
                "algorithm": split_receipt["split_algorithm"],
                "parent_replay_manifest_sha256": parent_sha,
                "source_fingerprint": source,
                "train_episode_count": TRAIN_EPISODES,
                "dev_episode_count": DEV_EPISODES,
                "episode_overlap_count": 0,
                "train_manifest_sha256": split_receipt["subsets"][
                    "representation_train"
                ]["manifest_sha256"],
                "dev_manifest_sha256": split_receipt["subsets"][
                    "representation_dev"
                ]["manifest_sha256"],
            },
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": checkpoint_sha,
                "training_step": checkpoint_metadata["training_step"],
                "policy_version": checkpoint_metadata["policy_version"],
                "stage1_frozen": True,
                "run_manifest_sha256": stage1_run_sha,
                "replay_manifest_sha256": stage1_replay_sha,
                "config": policy.config.to_dict(),
            },
            "evaluation_contract": {
                "state_scope": "current_and_next_transition_states",
                "baseline": "representation_train_featurewise_constant_mean",
                "mask_algorithm": MASK_ALGORITHM,
                "seed": args.seed,
                "prefix_mask_ratio": prefix_ratio,
                "topology_mask_ratio": topology_ratio,
                "batch_size": batch_size,
                "device": str(device),
                "minimum_relative_improvement": minimum_improvement,
                "minimum_z_rl_variance": minimum_variance,
            },
            "train_baseline": {
                **train_counts,
                "prefix_mean_finite": bool(torch.isfinite(prefix_mean).all()),
                "topology_mean_finite": bool(torch.isfinite(topology_mean).all()),
            },
            "heldout": evaluation,
            "checks": {
                "episode_disjoint": True,
                "reconstruction_better_than_train_constant_mean": reconstruction_passed,
                "finite_nonzero_z_rl_variance": variance_passed,
                "no_nan_or_inf": True,
            },
            "implementation": _implementation_hashes(),
            "started_unix_s": started,
            "completed_unix_s": time.time(),
        }
        receipt_path = staging / RECEIPT_NAME
        _atomic_json_new(receipt_path, receipt)
        receipt_sha = sha256_file(receipt_path)
        marker = {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "status": "completed" if passed else "rejected",
            "operation": "evaluate",
            "gate_passed": passed,
            "receipt_sha256": receipt_sha,
            "checkpoint_sha256": checkpoint_sha,
        }
        marker_name = COMPLETED_NAME if passed else "REJECTED.json"
        _atomic_json_new(staging / marker_name, marker)
        _publish_directory(staging, output_dir)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    result = {
        "status": "passed" if passed else "rejected",
        "operation": "evaluate",
        "gate_passed": passed,
        "output_dir": str(output_dir),
        "receipt": str(output_dir / RECEIPT_NAME),
        "receipt_sha256": receipt_sha,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_combined_mse": checkpoint_combined,
        "baseline_combined_mse": baseline_combined,
        "combined_relative_improvement": relative_improvement,
        "z_rl_variance_mean": variance_mean,
    }
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    split = commands.add_parser("split")
    split.add_argument("--replay-manifest", required=True, type=Path)
    split.add_argument("--expected-replay-manifest-sha256", required=True)
    split.add_argument("--expected-source-fingerprint", required=True)
    split.add_argument("--output-dir", required=True, type=Path)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--split-output-dir", required=True, type=Path)
    evaluate.add_argument("--expected-split-receipt-sha256", required=True)
    evaluate.add_argument("--expected-parent-replay-manifest-sha256", required=True)
    evaluate.add_argument("--expected-source-fingerprint", required=True)
    evaluate.add_argument("--stage1-checkpoint", required=True, type=Path)
    evaluate.add_argument("--expected-stage1-checkpoint-sha256", required=True)
    evaluate.add_argument("--expected-stage1-run-manifest-sha256", required=True)
    evaluate.add_argument("--expected-stage1-replay-manifest-sha256", required=True)
    evaluate.add_argument("--output-dir", required=True, type=Path)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--batch-size", type=int, default=64)
    evaluate.add_argument("--prefix-mask-ratio", type=float, default=0.25)
    evaluate.add_argument("--topology-mask-ratio", type=float, default=0.25)
    evaluate.add_argument("--seed", type=int, default=20261002)
    evaluate.add_argument("--minimum-relative-improvement", type=float, default=0.0)
    evaluate.add_argument("--minimum-z-variance", type=float, default=1.0e-8)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    output = split_replay(args) if args.command == "split" else evaluate_checkpoint(args)
    print(json.dumps(output, sort_keys=True, allow_nan=False))
    return 0 if output.get("status") != "rejected" else 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        Stage1GateError,
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
