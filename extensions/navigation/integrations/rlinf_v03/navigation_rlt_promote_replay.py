#!/usr/bin/env python3
"""Promote immutable Navigation-RLT prefix replay into frozen Stage-1 tokens.

This command is the only bridge between the two Navigation-RLT training
stages.  It verifies a complete ``prefix_hidden`` replay manifest and a
content-addressed frozen Stage-1 checkpoint, deterministically encodes current
and next states, and writes a new ``frozen_z_rl`` replay package.  Stage 2 can
therefore remain fail-closed and never recompute representations while it is
optimizing the actor and critics.
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
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import navigation_rlt_policy as policy_module
import navigation_rlt_replay as replay_module
from navigation_rlt_policy import (
    NavigationRLTContext,
    NavigationRLTPolicy,
    Z_RL_DIM,
    pack_topology_tokens,
    select_prefix_tokens,
)
from navigation_rlt_replay import (
    FEATURE_STAGE_FROZEN_Z_RL,
    FEATURE_STAGE_PREFIX_HIDDEN,
    PREFIX_REPRESENTATION_ARRAYS,
    REFERENCE_DIM,
    REFERENCE_SCHEMA,
    REPRESENTATION_FROZEN_Z_RL,
    REPRESENTATION_PREFIX_HIDDEN,
    build_replay_manifest,
    load_episode_shard,
    load_replay_manifest,
    save_episode_shard,
    save_replay_manifest,
    sha256_file,
)


SCHEMA_NAME = "navigation_rlt_replay_promotion"
SCHEMA_VERSION = 1
DEFAULT_MANIFEST_NAME = "replay_manifest.json"
RECEIPT_NAME = "promotion_receipt.json"
COMPLETED_NAME = "COMPLETED.json"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
STAGE1_GATE_SCHEMA_NAME = "navigation_rlt_stage1_gate"
STAGE1_GATE_SCHEMA_VERSION = 1
STAGE1_GATE_RECEIPT_NAME = "stage1_gate_receipt.json"
STAGE1_GATE_MARKER_FIELDS = {
    "schema_name",
    "schema_version",
    "status",
    "operation",
    "gate_passed",
    "receipt_sha256",
    "checkpoint_sha256",
}


class ReplayPromotionError(RuntimeError):
    """Raised when promotion would violate provenance or representation gates."""


def _validate_sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ReplayPromotionError(f"{name} must be a lowercase SHA-256")
    return value


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ReplayPromotionError("promotion evidence is not canonical JSON") from error


def _canonical_json_file(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8") + b"\n"
    except (TypeError, ValueError) as error:
        raise ReplayPromotionError("gate evidence is not canonical JSON") from error


def _load_canonical_json(path: Path, name: str) -> dict[str, object]:
    raw = Path(path).read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReplayPromotionError(f"{name} is invalid JSON") from error
    if not isinstance(value, dict):
        raise ReplayPromotionError(f"{name} must contain an object")
    if raw != _canonical_json_file(value):
        raise ReplayPromotionError(f"{name} is not canonical JSON")
    return value


def _sha256_payload(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _atomic_json_new(path: Path, value: object) -> None:
    path = Path(path)
    if os.path.lexists(path):
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
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


def _regular_input(path: Path, name: str) -> Path:
    raw = Path(path)
    if raw.is_symlink():
        raise ReplayPromotionError(f"{name} must not be a symbolic link")
    resolved = raw.resolve(strict=True)
    if not resolved.is_file():
        raise ReplayPromotionError(f"{name} must be a regular file")
    return resolved


def _verify_stage1_gate(
    marker_value: Path,
    expected_marker_sha256: str,
    *,
    expected_checkpoint_sha256: str,
    expected_parent_manifest_sha256: str,
    expected_source_fingerprint: str,
) -> dict[str, object]:
    marker_path = _regular_input(marker_value, "Stage-1 gate completion marker")
    if marker_path.name != COMPLETED_NAME:
        raise ReplayPromotionError("Stage-1 gate marker must be COMPLETED.json")
    marker_sha = _validate_sha256(
        expected_marker_sha256, "expected Stage-1 gate marker SHA-256"
    )
    if sha256_file(marker_path) != marker_sha:
        raise ReplayPromotionError("Stage-1 gate marker SHA-256 differs")
    marker = _load_canonical_json(marker_path, "Stage-1 gate completion marker")
    if set(marker) != STAGE1_GATE_MARKER_FIELDS:
        raise ReplayPromotionError("Stage-1 gate marker members differ")
    if (
        marker.get("schema_name") != STAGE1_GATE_SCHEMA_NAME
        or marker.get("schema_version") != STAGE1_GATE_SCHEMA_VERSION
        or marker.get("status") != "completed"
        or marker.get("operation") != "evaluate"
        or marker.get("gate_passed") is not True
        or marker.get("checkpoint_sha256") != expected_checkpoint_sha256
    ):
        raise ReplayPromotionError("Stage-1 gate marker did not pass this checkpoint")

    receipt_sha = _validate_sha256(
        marker.get("receipt_sha256"), "Stage-1 gate receipt SHA-256"
    )
    receipt_path = _regular_input(
        marker_path.parent / STAGE1_GATE_RECEIPT_NAME,
        "Stage-1 gate receipt",
    )
    if sha256_file(receipt_path) != receipt_sha:
        raise ReplayPromotionError("Stage-1 gate receipt SHA-256 differs")
    receipt = _load_canonical_json(receipt_path, "Stage-1 gate receipt")
    if (
        receipt.get("schema_name") != STAGE1_GATE_SCHEMA_NAME
        or receipt.get("schema_version") != STAGE1_GATE_SCHEMA_VERSION
        or receipt.get("status") != "passed"
        or receipt.get("operation") != "evaluate"
        or receipt.get("gate") != "episode-heldout-stage1-validity"
    ):
        raise ReplayPromotionError("Stage-1 gate receipt contract differs")
    checkpoint = receipt.get("checkpoint")
    split = receipt.get("split")
    if not isinstance(checkpoint, dict) or not isinstance(split, dict):
        raise ReplayPromotionError("Stage-1 gate identity evidence is missing")
    if (
        checkpoint.get("sha256") != expected_checkpoint_sha256
        or checkpoint.get("stage1_frozen") is not True
        or split.get("parent_replay_manifest_sha256")
        != expected_parent_manifest_sha256
        or split.get("source_fingerprint") != expected_source_fingerprint
    ):
        raise ReplayPromotionError("Stage-1 gate input identity differs")
    stage1_run_manifest_sha = _validate_sha256(
        checkpoint.get("run_manifest_sha256"),
        "Stage-1 training run manifest SHA-256",
    )
    stage1_replay_manifest_sha = _validate_sha256(
        checkpoint.get("replay_manifest_sha256"),
        "Stage-1 training replay manifest SHA-256",
    )
    if split.get("train_manifest_sha256") != stage1_replay_manifest_sha:
        raise ReplayPromotionError(
            "Stage-1 checkpoint replay binding differs from the train subset"
        )
    return {
        "schema_name": STAGE1_GATE_SCHEMA_NAME,
        "schema_version": STAGE1_GATE_SCHEMA_VERSION,
        "gate_passed": True,
        "marker_path": str(marker_path),
        "marker_sha256": marker_sha,
        "receipt_path": str(receipt_path),
        "receipt_sha256": receipt_sha,
        "checkpoint_sha256": expected_checkpoint_sha256,
        "run_manifest_sha256": stage1_run_manifest_sha,
        "replay_manifest_sha256": stage1_replay_manifest_sha,
        "parent_replay_manifest_sha256": expected_parent_manifest_sha256,
        "source_fingerprint": expected_source_fingerprint,
    }


def _device(value: str) -> torch.device:
    try:
        result = torch.device(value)
    except (RuntimeError, ValueError) as error:
        raise ReplayPromotionError(f"invalid torch device {value!r}") from error
    if result.type == "cuda":
        if not torch.cuda.is_available():
            raise ReplayPromotionError("CUDA promotion was requested but CUDA is unavailable")
        index = torch.cuda.current_device() if result.index is None else result.index
        if index < 0 or index >= torch.cuda.device_count():
            raise ReplayPromotionError("CUDA device index is outside the visible devices")
        result = torch.device("cuda", index)
    return result


def _output_paths(
    output_dir_value: Path, output_manifest_value: Path
) -> tuple[Path, Path, Path]:
    output_dir = Path(output_dir_value).resolve(strict=False)
    if os.path.lexists(output_dir):
        raise FileExistsError(output_dir)
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise ReplayPromotionError("output parent must be a regular directory")

    raw_manifest = Path(output_manifest_value)
    final_manifest = (
        raw_manifest.resolve(strict=False)
        if raw_manifest.is_absolute()
        else (output_dir / raw_manifest).resolve(strict=False)
    )
    try:
        relative_manifest = final_manifest.relative_to(output_dir)
    except ValueError as error:
        raise ReplayPromotionError(
            "output manifest must be contained inside output-dir"
        ) from error
    if relative_manifest == Path(".") or final_manifest.suffix.lower() != ".json":
        raise ReplayPromotionError("output manifest must name a JSON file")
    return output_dir, final_manifest, relative_manifest


def _safe_stem(index: int, episode_id: object, input_sha256: object) -> str:
    safe_episode = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(episode_id)).strip("._")
    if not safe_episode:
        safe_episode = "episode"
    return f"{index:06d}_{safe_episode[:96]}_{str(input_sha256)[:12]}"


class _RangeStats:
    def __init__(self) -> None:
        self.count = 0
        self.minimum: int | None = None
        self.maximum: int | None = None

    def update(self, values: torch.Tensor) -> None:
        value = values.detach().to(device="cpu", dtype=torch.int64).numpy()
        if value.ndim != 1 or value.size < 1:
            raise ReplayPromotionError("prefix length statistic is empty")
        current_minimum = int(value.min())
        current_maximum = int(value.max())
        self.count += int(value.size)
        self.minimum = (
            current_minimum
            if self.minimum is None
            else min(self.minimum, current_minimum)
        )
        self.maximum = (
            current_maximum
            if self.maximum is None
            else max(self.maximum, current_maximum)
        )

    def receipt(self) -> dict[str, int]:
        if self.count < 1 or self.minimum is None or self.maximum is None:
            raise ReplayPromotionError("prefix length statistic was never populated")
        return {
            "count": self.count,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }


class _ZStats:
    def __init__(self) -> None:
        self.rows = 0
        self.values = 0
        self.total = 0.0
        self.total_squared = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf
        self.l2_minimum = math.inf
        self.l2_maximum = -math.inf

    def update(self, value: np.ndarray) -> None:
        array = np.asarray(value, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != Z_RL_DIM or array.shape[0] < 1:
            raise ReplayPromotionError("encoded z_rl shape differs from [B,256]")
        if not bool(np.isfinite(array).all()):
            raise ReplayPromotionError("encoded z_rl contains non-finite values")
        as64 = array.astype(np.float64, copy=False)
        norms = np.linalg.norm(as64, axis=1)
        self.rows += int(array.shape[0])
        self.values += int(array.size)
        self.total += float(as64.sum(dtype=np.float64))
        self.total_squared += float(np.square(as64).sum(dtype=np.float64))
        self.minimum = min(self.minimum, float(as64.min()))
        self.maximum = max(self.maximum, float(as64.max()))
        self.l2_minimum = min(self.l2_minimum, float(norms.min()))
        self.l2_maximum = max(self.l2_maximum, float(norms.max()))

    def receipt(self) -> dict[str, object]:
        if self.rows < 1 or self.values < 1:
            raise ReplayPromotionError("z_rl statistic was never populated")
        mean = self.total / self.values
        variance = max(0.0, self.total_squared / self.values - mean * mean)
        result: dict[str, object] = {
            "finite": True,
            "rows": self.rows,
            "dimension": Z_RL_DIM,
            "value_count": self.values,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": mean,
            "standard_deviation": math.sqrt(variance),
            "l2_norm_minimum": self.l2_minimum,
            "l2_norm_maximum": self.l2_maximum,
        }
        if not all(
            math.isfinite(float(value))
            for value in result.values()
            if isinstance(value, float)
        ):
            raise ReplayPromotionError("z_rl statistics are non-finite")
        return result


def _common_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    representation_fields = {
        "source_fingerprint",
        "representation_kind",
        "feature_stage",
        "prefix_max_tokens",
        "prefix_hidden_dim",
        "z_rl_dim",
        "rlt_checkpoint_sha256",
    }
    return {
        name: value
        for name, value in metadata.items()
        if name not in representation_fields
    }


def _promoted_metadata(
    metadata: Mapping[str, object], checkpoint_sha256: str
) -> dict[str, object]:
    result = _common_metadata(metadata)
    result.update(
        representation_kind=REPRESENTATION_FROZEN_Z_RL,
        feature_stage=FEATURE_STAGE_FROZEN_Z_RL,
        z_rl_dim=Z_RL_DIM,
        rlt_checkpoint_sha256=checkpoint_sha256,
    )
    return result


def _validate_checkpoint_contract(
    policy: NavigationRLTPolicy,
    checkpoint_metadata: Mapping[str, object],
    replay_metadata: Mapping[str, object],
) -> None:
    config = policy.config
    if not bool(policy.stage1_frozen.item()) or checkpoint_metadata.get(
        "stage1_frozen"
    ) is not True:
        raise ReplayPromotionError("checkpoint does not contain a frozen Stage 1")
    if any(parameter.requires_grad for parameter in policy.stage1.parameters()):
        raise ReplayPromotionError("checkpoint Stage-1 parameters require gradients")
    if checkpoint_metadata.get("lightnav_backbone_trainable") is not False:
        raise ReplayPromotionError("checkpoint does not keep LightNav external and frozen")
    if checkpoint_metadata.get("ttt_enabled") is not False:
        raise ReplayPromotionError("TTT checkpoints are forbidden for promotion")
    if not bool(config.strict_frozen_inputs):
        raise ReplayPromotionError("checkpoint permits trainable external inputs")
    if config.reference_dim != REFERENCE_DIM:
        raise ReplayPromotionError("checkpoint reference width differs from replay v3")
    if replay_metadata.get("reference_schema") != REFERENCE_SCHEMA:
        raise ReplayPromotionError("input replay reference schema differs")
    exact_dimensions = {
        "prefix_hidden_dim": config.prefix_hidden_dim,
        "prefix_max_tokens": config.max_artifact_prefix_tokens,
        "topology_node_dim": config.topology_node_dim,
        "topology_global_dim": config.topology_global_dim,
    }
    for name, expected in exact_dimensions.items():
        if replay_metadata.get(name) != expected:
            raise ReplayPromotionError(
                f"input replay {name} differs from checkpoint config"
            )
    if int(replay_metadata["topology_max_nodes"]) + 1 != config.max_topology_tokens:
        raise ReplayPromotionError(
            "input topology extent differs from checkpoint packing width"
        )
    if replay_metadata.get("representation_kind") != REPRESENTATION_PREFIX_HIDDEN:
        raise ReplayPromotionError("input shard is not prefix_hidden replay")
    if replay_metadata.get("feature_stage") != FEATURE_STAGE_PREFIX_HIDDEN:
        raise ReplayPromotionError("input shard is not Stage-1 prefix replay")


@torch.inference_mode()
def _encode_state(
    policy: NavigationRLTPolicy,
    arrays: Mapping[str, np.ndarray],
    *,
    prefix: str,
    device: torch.device,
    batch_size: int,
    original_lengths: _RangeStats,
    selected_lengths: _RangeStats,
    z_stats: _ZStats,
) -> np.ndarray:
    rows = int(np.asarray(arrays[prefix + "prefix_hidden"]).shape[0])
    output = np.empty((rows, Z_RL_DIM), dtype=np.float32)
    config = policy.config
    for start in range(0, rows, batch_size):
        stop = min(rows, start + batch_size)
        hidden = torch.as_tensor(
            np.asarray(arrays[prefix + "prefix_hidden"])[start:stop],
            device=device,
            dtype=torch.float32,
        )
        mask = torch.as_tensor(
            np.asarray(arrays[prefix + "prefix_mask"])[start:stop],
            device=device,
            dtype=torch.bool,
        )
        selected = select_prefix_tokens(hidden, mask, config=config)
        packed = pack_topology_tokens(
            torch.as_tensor(
                np.asarray(arrays[prefix + "topology_node_targets"])[start:stop],
                device=device,
                dtype=torch.float32,
            ),
            torch.as_tensor(
                np.asarray(arrays[prefix + "topology_node_mask"])[start:stop],
                device=device,
                dtype=torch.bool,
            ),
            torch.as_tensor(
                np.asarray(arrays[prefix + "topology_global_targets"])[start:stop],
                device=device,
                dtype=torch.float32,
            ),
            torch.as_tensor(
                np.asarray(arrays[prefix + "topology_global_mask"])[start:stop],
                device=device,
                dtype=torch.bool,
            ),
            config=config,
        )
        context = NavigationRLTContext(
            contextual_prefix_hidden=selected.contextual_prefix_hidden,
            prefix_mask=selected.prefix_mask,
            topology_tokens=packed.tokens,
            topology_mask=packed.token_mask,
        )
        encoded = policy.encode(context)
        if encoded.requires_grad or encoded.shape != (stop - start, Z_RL_DIM):
            raise ReplayPromotionError("policy returned an invalid frozen z_rl tensor")
        encoded_numpy = encoded.detach().to(device="cpu", dtype=torch.float32).numpy()
        if not bool(np.isfinite(encoded_numpy).all()):
            raise ReplayPromotionError("policy returned non-finite z_rl")
        output[start:stop] = encoded_numpy
        original_lengths.update(selected.original_valid_lengths)
        selected_lengths.update(selected.selected_valid_lengths)
        z_stats.update(encoded_numpy)
    return output


def _verify_preserved_arrays(
    source: Mapping[str, np.ndarray], promoted: Mapping[str, np.ndarray]
) -> None:
    representation_names = set(PREFIX_REPRESENTATION_ARRAYS) | {
        "z_rl",
        "next_z_rl",
    }
    source_common = set(source) - representation_names
    promoted_common = set(promoted) - representation_names
    if source_common != promoted_common:
        raise ReplayPromotionError("promotion changed non-representation array members")
    for name in sorted(source_common):
        before = np.asarray(source[name])
        after = np.asarray(promoted[name])
        if (
            before.dtype != after.dtype
            or before.shape != after.shape
            or not np.array_equal(before, after)
        ):
            raise ReplayPromotionError(
                f"promotion changed non-representation array {name}"
            )


def _implementation_hashes() -> dict[str, str]:
    paths = {
        "promoter_sha256": Path(__file__).resolve(),
        "policy_sha256": Path(str(policy_module.__file__)).resolve(strict=True),
        "replay_sha256": Path(str(replay_module.__file__)).resolve(strict=True),
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def promote(args: argparse.Namespace) -> dict[str, object]:
    started = time.time()
    if args.batch_size < 1:
        raise ReplayPromotionError("batch-size must be positive")
    expected_manifest_sha = _validate_sha256(
        args.expected_input_manifest_sha256,
        "expected input manifest SHA-256",
    )
    expected_checkpoint_sha = _validate_sha256(
        args.expected_stage1_checkpoint_sha256,
        "expected Stage-1 checkpoint SHA-256",
    )
    input_manifest_path = _regular_input(args.input_manifest, "input manifest")
    checkpoint_path = _regular_input(args.stage1_checkpoint, "Stage-1 checkpoint")
    output_dir, final_manifest_path, relative_manifest_path = _output_paths(
        args.output_dir, args.output_manifest
    )

    if sha256_file(checkpoint_path) != expected_checkpoint_sha:
        raise ReplayPromotionError("Stage-1 checkpoint SHA-256 differs")
    input_manifest = load_replay_manifest(
        input_manifest_path,
        expected_sha256=expected_manifest_sha,
        verify_shards=True,
    )
    if input_manifest["representation_kind"] != REPRESENTATION_PREFIX_HIDDEN:
        raise ReplayPromotionError("input manifest is not prefix_hidden replay")
    if input_manifest["feature_stage"] != FEATURE_STAGE_PREFIX_HIDDEN:
        raise ReplayPromotionError("input manifest is not Stage-1 prefix replay")
    stage1_gate = _verify_stage1_gate(
        args.stage1_gate_completion_marker,
        args.expected_stage1_gate_completion_marker_sha256,
        expected_checkpoint_sha256=expected_checkpoint_sha,
        expected_parent_manifest_sha256=expected_manifest_sha,
        expected_source_fingerprint=str(input_manifest["source_fingerprint"]),
    )

    device = _device(args.device)
    policy, checkpoint_metadata = NavigationRLTPolicy.load_checkpoint(
        checkpoint_path, map_location="cpu"
    )
    policy.to(device).eval()
    if device.type == "cuda":
        torch.cuda.set_device(device)

    first_receipt = input_manifest["shards"][0]
    first_path = _regular_input(Path(str(first_receipt["path"])), "input shard")
    _, first_metadata = load_episode_shard(
        first_path,
        expected_sha256=str(first_receipt["sha256"]),
        expected_source_fingerprint=str(input_manifest["source_fingerprint"]),
        expected_representation_kind=REPRESENTATION_PREFIX_HIDDEN,
    )
    _validate_checkpoint_contract(policy, checkpoint_metadata, first_metadata)

    provenance = {
        "input_manifest_sha256": expected_manifest_sha,
        "input_source_fingerprint": input_manifest["source_fingerprint"],
        "stage1_checkpoint_sha256": expected_checkpoint_sha,
        "stage1_gate_marker_sha256": stage1_gate["marker_sha256"],
        "stage1_gate_receipt_sha256": stage1_gate["receipt_sha256"],
        "policy_config": policy.config.to_dict(),
        "representation_from": REPRESENTATION_PREFIX_HIDDEN,
        "representation_to": REPRESENTATION_FROZEN_Z_RL,
    }
    promotion_fingerprint = _sha256_payload(provenance)
    original_current = _RangeStats()
    original_next = _RangeStats()
    selected_current = _RangeStats()
    selected_next = _RangeStats()
    z_current_stats = _ZStats()
    z_next_stats = _ZStats()
    output_receipts: list[dict[str, object]] = []
    shard_mappings: list[dict[str, object]] = []

    staging_dir = Path(
        tempfile.mkdtemp(prefix=output_dir.name + ".staging-", dir=str(output_dir.parent))
    )
    try:
        for index, input_receipt in enumerate(input_manifest["shards"]):
            input_path = _regular_input(
                Path(str(input_receipt["path"])), "input shard"
            )
            arrays, metadata = load_episode_shard(
                input_path,
                expected_sha256=str(input_receipt["sha256"]),
                expected_source_fingerprint=str(input_manifest["source_fingerprint"]),
                expected_representation_kind=REPRESENTATION_PREFIX_HIDDEN,
            )
            _validate_checkpoint_contract(policy, checkpoint_metadata, metadata)
            current_z = _encode_state(
                policy,
                arrays,
                prefix="",
                device=device,
                batch_size=args.batch_size,
                original_lengths=original_current,
                selected_lengths=selected_current,
                z_stats=z_current_stats,
            )
            next_z = _encode_state(
                policy,
                arrays,
                prefix="next_",
                device=device,
                batch_size=args.batch_size,
                original_lengths=original_next,
                selected_lengths=selected_next,
                z_stats=z_next_stats,
            )
            promoted_arrays = {
                name: np.asarray(value).copy()
                for name, value in arrays.items()
                if name not in PREFIX_REPRESENTATION_ARRAYS
            }
            promoted_arrays["z_rl"] = current_z
            promoted_arrays["next_z_rl"] = next_z
            promoted_metadata = _promoted_metadata(metadata, expected_checkpoint_sha)
            stem = _safe_stem(
                index,
                metadata["episode_id"],
                input_receipt["sha256"],
            )
            staged_shard = staging_dir / "shards" / f"{stem}.npz"
            final_shard = output_dir / "shards" / f"{stem}.npz"
            staged_receipt = save_episode_shard(
                staged_shard, promoted_arrays, promoted_metadata
            )
            loaded_arrays, loaded_metadata = load_episode_shard(
                staged_shard,
                expected_sha256=str(staged_receipt["sha256"]),
                expected_representation_kind=REPRESENTATION_FROZEN_Z_RL,
            )
            _verify_preserved_arrays(arrays, loaded_arrays)
            if _common_metadata(metadata) != _common_metadata(loaded_metadata):
                raise ReplayPromotionError("promotion changed common replay metadata")
            if not np.array_equal(loaded_arrays["z_rl"], current_z) or not np.array_equal(
                loaded_arrays["next_z_rl"], next_z
            ):
                raise ReplayPromotionError("saved frozen tokens differ from encoder output")
            final_receipt = dict(staged_receipt)
            final_receipt["path"] = str(final_shard.resolve(strict=False))
            output_receipts.append(final_receipt)
            shard_mappings.append(
                {
                    "episode_id": metadata["episode_id"],
                    "reset_token": metadata["reset_token"],
                    "row_count": metadata["row_count"],
                    "behavior_policy_version": metadata["behavior_policy_version"],
                    "input_path": str(input_path),
                    "input_sha256": input_receipt["sha256"],
                    "output_path": str(final_shard.resolve(strict=False)),
                    "output_sha256": staged_receipt["sha256"],
                }
            )

        output_manifest = build_replay_manifest(
            output_receipts,
            run_id=str(input_manifest["run_id"]),
            expected_episode_count=int(input_manifest["expected_episode_count"]),
        )
        staged_manifest_path = staging_dir / relative_manifest_path
        manifest_receipt = save_replay_manifest(
            staged_manifest_path, output_manifest
        )
        output_manifest_sha = str(manifest_receipt["sha256"])
        receipt = {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "promotion_fingerprint": promotion_fingerprint,
            "input": {
                "manifest_path": str(input_manifest_path),
                "manifest_sha256": expected_manifest_sha,
                "source_fingerprint": input_manifest["source_fingerprint"],
                "representation_kind": input_manifest["representation_kind"],
                "episode_count": input_manifest["completed_episode_count"],
                "transition_count": input_manifest["transition_count"],
                "behavior_policy_versions": input_manifest[
                    "behavior_policy_versions"
                ],
            },
            "stage1_checkpoint": {
                "path": str(checkpoint_path),
                "sha256": expected_checkpoint_sha,
                "training_step": checkpoint_metadata["training_step"],
                "policy_version": checkpoint_metadata["policy_version"],
                "stage1_frozen": True,
            },
            "stage1_gate": stage1_gate,
            "output": {
                "manifest_path": str(final_manifest_path),
                "manifest_sha256": output_manifest_sha,
                "source_fingerprint": output_manifest["source_fingerprint"],
                "representation_kind": output_manifest["representation_kind"],
                "feature_stage": output_manifest["feature_stage"],
                "episode_count": output_manifest["completed_episode_count"],
                "transition_count": output_manifest["transition_count"],
                "behavior_policy_versions": output_manifest[
                    "behavior_policy_versions"
                ],
            },
            "model_contract": {
                "z_rl_dim": Z_RL_DIM,
                "reference_schema": REFERENCE_SCHEMA,
                "reference_dim": REFERENCE_DIM,
                "prefix_selection_schema": policy.config.prefix_selection_schema,
                "topology_packing_schema": policy.config.topology_packing_schema,
                "max_artifact_prefix_tokens": policy.config.max_artifact_prefix_tokens,
                "max_selected_prefix_tokens": policy.config.max_prefix_tokens,
                "max_topology_tokens": policy.config.max_topology_tokens,
            },
            "prefix_lengths": {
                "current_original": original_current.receipt(),
                "current_selected": selected_current.receipt(),
                "next_original": original_next.receipt(),
                "next_selected": selected_next.receipt(),
            },
            "z_rl_statistics": {
                "current": z_current_stats.receipt(),
                "next": z_next_stats.receipt(),
            },
            "implementation": _implementation_hashes(),
            "shards": shard_mappings,
            "device": str(device),
            "batch_size": int(args.batch_size),
            "started_unix_s": started,
            "completed_unix_s": time.time(),
        }
        staged_receipt_path = staging_dir / RECEIPT_NAME
        _atomic_json_new(staged_receipt_path, receipt)
        receipt_sha = sha256_file(staged_receipt_path)
        _atomic_json_new(
            staging_dir / COMPLETED_NAME,
            {
                "schema_name": SCHEMA_NAME,
                "schema_version": SCHEMA_VERSION,
                "status": "completed",
                "promotion_receipt_sha256": receipt_sha,
                "output_manifest_sha256": output_manifest_sha,
                "stage1_checkpoint_sha256": expected_checkpoint_sha,
                "stage1_gate_marker_sha256": stage1_gate["marker_sha256"],
                "stage1_gate_receipt_sha256": stage1_gate["receipt_sha256"],
            },
        )
        if os.path.lexists(output_dir):
            raise FileExistsError(output_dir)
        os.replace(staging_dir, output_dir)
    except BaseException:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise

    verified = load_replay_manifest(
        final_manifest_path,
        expected_sha256=output_manifest_sha,
        expected_source_fingerprint=str(output_manifest["source_fingerprint"]),
        verify_shards=True,
    )
    if (
        verified["completed_episode_count"]
        != input_manifest["completed_episode_count"]
        or verified["transition_count"] != input_manifest["transition_count"]
        or verified["behavior_policy_versions"]
        != input_manifest["behavior_policy_versions"]
    ):
        raise ReplayPromotionError("final promoted manifest inventory differs")
    return {
        "status": "completed",
        "output_dir": str(output_dir),
        "output_manifest": str(final_manifest_path),
        "output_manifest_sha256": output_manifest_sha,
        "output_source_fingerprint": output_manifest["source_fingerprint"],
        "promotion_receipt": str(output_dir / RECEIPT_NAME),
        "promotion_receipt_sha256": receipt_sha,
        "stage1_gate_marker_sha256": stage1_gate["marker_sha256"],
        "stage1_gate_receipt_sha256": stage1_gate["receipt_sha256"],
        "episode_count": verified["completed_episode_count"],
        "transition_count": verified["transition_count"],
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input-manifest", required=True, type=Path)
    result.add_argument("--expected-input-manifest-sha256", required=True)
    result.add_argument("--stage1-checkpoint", required=True, type=Path)
    result.add_argument("--expected-stage1-checkpoint-sha256", required=True)
    result.add_argument(
        "--stage1-gate-completion-marker",
        required=True,
        type=Path,
        help="Passed Stage-1 gate COMPLETED.json",
    )
    result.add_argument(
        "--expected-stage1-gate-completion-marker-sha256",
        required=True,
    )
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument(
        "--output-manifest",
        type=Path,
        default=Path(DEFAULT_MANIFEST_NAME),
        help="JSON path relative to output-dir, or an absolute path inside it",
    )
    result.add_argument("--device", default="cpu")
    result.add_argument("--batch-size", type=int, default=128)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    result = promote(args)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ReplayPromotionError, FileExistsError, replay_module.ReplayContractError) as error:
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
