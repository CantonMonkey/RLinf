#!/usr/bin/env python3
"""Train the two-stage Navigation-RLT policy through RLinf FSDP.

The command deliberately has two separate phases.  ``stage1`` learns the
masked-reconstruction token from frozen LightNav prefix hidden states and
causal topology targets, then saves a checkpoint with Stage 1 frozen.
``stage2`` accepts only replay containing tokens exported by that exact frozen
checkpoint and performs proposal-conditioned finite-candidate CQL/twin-Q
updates with a behavior-derived gate support mask and Native anchors.

Both phases are train-only.  Their immutable run manifest and replay manifest
are content-addressed on the command line so a torchrun cannot silently change
source identity or hyperparameters between ranks.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel,
    StateDictType,
)
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from omegaconf import OmegaConf

from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager

from navigation_rlt_policy import (
    ADJUST_GATE,
    CANONICAL_ACTION_DIM,
    CANONICAL_ACTION_SCHEMA,
    GATE_COUNT,
    GATE_NAMES,
    KEEP_GATE,
    NavigationRLTCompilerFields,
    NavigationRLTConfig,
    NavigationRLTContext,
    NavigationRLTPolicy,
    NavigationRLTZTransitionBatch,
    RECOVER_GATE,
    STAGE2_BEHAVIOR_PRIOR_UNIFORM_MIX,
    Stage1LossConfig,
    Stage2LossConfig,
    STOP_GATE,
    TOPOLOGY_PACKING_SCHEMA,
    Z_RL_DIM,
    pack_topology_tokens,
    select_prefix_tokens,
)
from navigation_rlt_replay import (
    LIGHTNAV_SOURCE_REVISION,
    REFERENCE_DIM,
    REFERENCE_SCHEMA,
    REPRESENTATION_FROZEN_Z_RL,
    REPRESENTATION_PREFIX_HIDDEN,
    load_episode_shard,
    load_replay_manifest,
    sha256_file,
)


SCHEMA_NAME = "navigation_rlt_rlinf_fsdp_training"
SCHEMA_VERSION = 1
RUN_MANIFEST_SCHEMA = "navigation_rlt_training_run_manifest"
RUN_MANIFEST_VERSION = 1
MAX_CHECKPOINT_BYTES = 2 * 1024**3
MAX_RUN_BYTES = 16 * 1024**3
MAX_LOSS_KEYS = 64
MAX_LOSS_SNAPSHOTS = 16
STAGE2_LEGACY_OBJECTIVE_SCHEMA = "proposal_conditioned_finite_candidate_cql_v1"
STAGE2_OBJECTIVE_SCHEMA = "proposal_conditioned_finite_candidate_cql_behavior_kl_v2"
STAGE2_BEHAVIOR_KL_WEIGHT = 1.0
STAGE2_ELIGIBILITY_SCHEMA = (
    "credit_mask_true_and_requested_gate_equals_executed_gate_v1"
)
STAGE2_CQL_ALPHA = 1.0
STAGE2_CQL_TEMPERATURE = 1.0
MINIMUM_BEHAVIOR_GATE_COUNT = 1


class NavigationRLTTrainingError(RuntimeError):
    """Raised when a fixed training or evidence contract is violated."""


def _atomic_json(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".staging")
    staging.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    staging.replace(path)


def _tree_size(path: Path) -> int:
    return sum(item.stat().st_size for item in Path(path).rglob("*") if item.is_file())


def _tensor_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _component_hash(state: Mapping[str, torch.Tensor], prefixes: Sequence[str]) -> str:
    selected = {
        name: value
        for name, value in state.items()
        if any(name.startswith(prefix) for prefix in prefixes)
    }
    if not selected:
        raise NavigationRLTTrainingError(
            f"state has no tensors under component prefixes {tuple(prefixes)!r}"
        )
    return _tensor_hash(selected)


def _finite_state(state: Mapping[str, torch.Tensor]) -> None:
    for name, value in state.items():
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise NavigationRLTTrainingError(f"state tensor {name} is non-finite")


def _gpu_snapshot(local_rank: int) -> dict[str, object]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(local_rank),
            "--query-gpu=uuid,name,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=20,
    )
    fields = [item.strip() for item in completed.stdout.strip().split(",")]
    if len(fields) != 5 or not fields[0].startswith("GPU-"):
        raise NavigationRLTTrainingError("nvidia-smi GPU identity response differs")
    return {
        "gpu_uuid": fields[0],
        "gpu_name": fields[1],
        "utilization_gpu_percent": float(fields[2]),
        "memory_used_mib": float(fields[3]),
        "memory_total_mib": float(fields[4]),
        "sampled_unix_s": time.time(),
    }


def _canonical_json_hash(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path, name: str) -> dict[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NavigationRLTTrainingError(f"{name} is not readable JSON") from error
    if not isinstance(value, dict):
        raise NavigationRLTTrainingError(f"{name} must contain a JSON object")
    return value


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise NavigationRLTTrainingError(f"{name} must be lowercase SHA-256")
    return value


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value < 1:
        raise NavigationRLTTrainingError(f"{name} must be a positive integer")
    return int(value)


def _finite_number(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NavigationRLTTrainingError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise NavigationRLTTrainingError(f"{name} must be {qualifier}")
    return result


def _validate_stage2_support(
    value: object, *, minimum_behavior_gate_count: int
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != {
        "eligibility_schema",
        "gate_names",
        "gate_counts",
        "gate_support_mask",
        "minimum_behavior_gate_count",
    }:
        raise NavigationRLTTrainingError("Stage-2 support evidence contract differs")
    if value.get("eligibility_schema") != STAGE2_ELIGIBILITY_SCHEMA:
        raise NavigationRLTTrainingError("Stage-2 eligibility schema differs")
    if value.get("gate_names") != list(GATE_NAMES):
        raise NavigationRLTTrainingError("Stage-2 gate names differ")
    if value.get("minimum_behavior_gate_count") != minimum_behavior_gate_count:
        raise NavigationRLTTrainingError("Stage-2 minimum gate count differs")
    counts = value.get("gate_counts")
    mask = value.get("gate_support_mask")
    if (
        not isinstance(counts, list)
        or len(counts) != GATE_COUNT
        or any(type(item) is not int or item < 0 for item in counts)
    ):
        raise NavigationRLTTrainingError("Stage-2 gate counts are invalid")
    if (
        not isinstance(mask, list)
        or len(mask) != GATE_COUNT
        or any(type(item) is not bool for item in mask)
    ):
        raise NavigationRLTTrainingError("Stage-2 gate support mask is invalid")
    expected_mask = [count >= minimum_behavior_gate_count for count in counts]
    if mask != expected_mask or not any(mask):
        raise NavigationRLTTrainingError("Stage-2 gate support mask differs from counts")
    return dict(value)


def _validate_run_manifest(
    value: Mapping[str, object],
    *,
    phase: str,
    replay_manifest_sha256: str,
    stage1_checkpoint_sha256: str | None,
) -> tuple[dict[str, object], NavigationRLTConfig]:
    required = {
        "schema_name",
        "schema_version",
        "run_id",
        "phase",
        "seed",
        "required_world_size",
        "boundaries",
        "implementation",
        "model",
        "replay",
        "training",
    }
    if phase == "stage2":
        required.update({"stage1_checkpoint", "stage2_support"})
    if set(value) != required:
        raise NavigationRLTTrainingError(
            "run manifest members differ; "
            f"missing={sorted(required - set(value))}, extra={sorted(set(value) - required)}"
        )
    if value.get("schema_name") != RUN_MANIFEST_SCHEMA:
        raise NavigationRLTTrainingError("run manifest schema name differs")
    if value.get("schema_version") != RUN_MANIFEST_VERSION:
        raise NavigationRLTTrainingError("run manifest schema version differs")
    if value.get("phase") != phase:
        raise NavigationRLTTrainingError("run manifest phase differs from the CLI")
    if not isinstance(value.get("run_id"), str) or not value["run_id"]:
        raise NavigationRLTTrainingError("run manifest run_id must be non-empty")
    _positive_int(value.get("seed"), "run manifest seed")
    _positive_int(value.get("required_world_size"), "required world size")

    boundaries = value.get("boundaries")
    expected_boundaries = {
        "upstream_split_read": "train",
        "split_label": "train_update",
        "validation_split_reads": 0,
        "lightnav_source_revision": LIGHTNAV_SOURCE_REVISION,
        "lightnav_frozen": True,
        "ttt_enabled": False,
    }
    if boundaries != expected_boundaries:
        raise NavigationRLTTrainingError("run manifest train-only boundary differs")

    implementation = value.get("implementation")
    implementation_paths = {
        "trainer_sha256": Path(__file__).resolve(),
        "policy_sha256": Path(__file__).resolve().with_name("navigation_rlt_policy.py"),
        "replay_sha256": Path(__file__).resolve().with_name("navigation_rlt_replay.py"),
    }
    if not isinstance(implementation, dict) or set(implementation) != set(
        implementation_paths
    ):
        raise NavigationRLTTrainingError("run manifest implementation contract differs")
    for name, path in implementation_paths.items():
        expected = _require_sha256(implementation.get(name), name)
        if sha256_file(path) != expected:
            raise NavigationRLTTrainingError(f"run manifest {name} differs")

    replay = value.get("replay")
    expected_representation = (
        REPRESENTATION_PREFIX_HIDDEN
        if phase == "stage1"
        else REPRESENTATION_FROZEN_Z_RL
    )
    if not isinstance(replay, dict):
        raise NavigationRLTTrainingError("run manifest replay must be an object")
    if set(replay) != {
        "manifest_sha256",
        "source_fingerprint",
        "representation_kind",
    }:
        raise NavigationRLTTrainingError("run manifest replay members differ")
    if _require_sha256(replay.get("manifest_sha256"), "replay manifest SHA-256") != replay_manifest_sha256:
        raise NavigationRLTTrainingError("run and CLI replay manifest hashes differ")
    _require_sha256(replay.get("source_fingerprint"), "replay source fingerprint")
    if replay.get("representation_kind") != expected_representation:
        raise NavigationRLTTrainingError("run manifest replay representation differs")

    model = value.get("model")
    if not isinstance(model, dict) or set(model) != {
        "config",
        "reference_embedding_schema",
        "topology_packing_schema",
        "canonical_action_schema",
    }:
        raise NavigationRLTTrainingError("run manifest model contract differs")
    if model.get("reference_embedding_schema") != REFERENCE_SCHEMA:
        raise NavigationRLTTrainingError("reference embedding schema differs")
    if model.get("topology_packing_schema") != TOPOLOGY_PACKING_SCHEMA:
        raise NavigationRLTTrainingError("topology packing schema differs")
    if model.get("canonical_action_schema") != CANONICAL_ACTION_SCHEMA:
        raise NavigationRLTTrainingError("canonical action schema differs")
    try:
        model_config = NavigationRLTConfig(**model["config"])
    except (KeyError, TypeError, ValueError) as error:
        raise NavigationRLTTrainingError("Navigation-RLT model config is invalid") from error
    if model_config.reference_dim != REFERENCE_DIM:
        raise NavigationRLTTrainingError("model reference_dim differs from replay")
    if model_config.canonical_action_dim != CANONICAL_ACTION_DIM:
        raise NavigationRLTTrainingError("model canonical_action_dim differs")

    training = value.get("training")
    if not isinstance(training, dict):
        raise NavigationRLTTrainingError("run manifest training must be an object")
    common = {
        "epochs",
        "batch_size_per_rank",
        "learning_rate",
        "weight_decay",
        "clip_grad",
    }
    phase_fields = (
        {
            "prefix_mask_ratio",
            "topology_mask_ratio",
            "prefix_weight",
            "topology_weight",
        }
        if phase == "stage1"
        else {
            "objective_schema",
            "cql_alpha",
            "cql_temperature",
            "minimum_behavior_gate_count",
            "reference_gate_weight",
            "reference_residual_weight",
            "reference_dropout_probability",
            "target_tau",
            "policy_version",
        }
    )
    regularized_stage2 = (
        phase == "stage2" and training.get("objective_schema") == STAGE2_OBJECTIVE_SCHEMA
    )
    if regularized_stage2:
        phase_fields.add("behavior_kl_weight")
    if set(training) != common | phase_fields:
        raise NavigationRLTTrainingError("run manifest training members differ")
    _positive_int(training.get("epochs"), "epochs")
    _positive_int(training.get("batch_size_per_rank"), "batch_size_per_rank")
    for name in ("learning_rate", "clip_grad"):
        _finite_number(training.get(name), name, positive=True)
    weight_decay = _finite_number(training.get("weight_decay"), "weight_decay")
    if weight_decay < 0.0:
        raise NavigationRLTTrainingError("weight_decay must be non-negative")
    if phase == "stage1":
        Stage1LossConfig(
            prefix_mask_ratio=training["prefix_mask_ratio"],
            topology_mask_ratio=training["topology_mask_ratio"],
            prefix_weight=training["prefix_weight"],
            topology_weight=training["topology_weight"],
        )
    else:
        if training.get("objective_schema") not in {
            STAGE2_OBJECTIVE_SCHEMA, STAGE2_LEGACY_OBJECTIVE_SCHEMA
        }:
            raise NavigationRLTTrainingError("Stage-2 objective schema differs")
        if regularized_stage2 and _finite_number(
            training.get("behavior_kl_weight"), "behavior_kl_weight", positive=True
        ) != STAGE2_BEHAVIOR_KL_WEIGHT:
            raise NavigationRLTTrainingError("Stage-2 fixed behavior KL weight differs")
        if (
            _finite_number(training.get("cql_alpha"), "cql_alpha", positive=True)
            != STAGE2_CQL_ALPHA
        ):
            raise NavigationRLTTrainingError("Stage-2 cql_alpha differs")
        if (
            _finite_number(
                training.get("cql_temperature"), "cql_temperature", positive=True
            )
            != STAGE2_CQL_TEMPERATURE
        ):
            raise NavigationRLTTrainingError("Stage-2 cql_temperature differs")
        minimum_gate_count = _positive_int(
            training.get("minimum_behavior_gate_count"),
            "minimum_behavior_gate_count",
        )
        if minimum_gate_count != MINIMUM_BEHAVIOR_GATE_COUNT:
            raise NavigationRLTTrainingError(
                "Stage-2 minimum_behavior_gate_count differs"
            )
        Stage2LossConfig(
            reference_gate_weight=training["reference_gate_weight"],
            reference_residual_weight=training["reference_residual_weight"],
            reference_dropout_probability=training["reference_dropout_probability"],
            target_tau=training["target_tau"],
            cql_alpha=training["cql_alpha"],
            cql_temperature=training["cql_temperature"],
        )
        if float(training["reference_gate_weight"]) <= 0.0:
            raise NavigationRLTTrainingError(
                "Stage 2 requires a positive Native gate anchor"
            )
        if float(training["reference_residual_weight"]) <= 0.0:
            raise NavigationRLTTrainingError(
                "Stage 2 requires a positive conservative residual anchor"
            )
        dropout = float(training["reference_dropout_probability"])
        if not 0.0 < dropout < 1.0:
            raise NavigationRLTTrainingError(
                "Stage 2 requires reference dropout strictly inside (0, 1)"
            )
        if type(training.get("policy_version")) is not int or training["policy_version"] < 1:
            raise NavigationRLTTrainingError("stage2 policy_version must be positive")
        checkpoint = value.get("stage1_checkpoint")
        if not isinstance(checkpoint, dict) or set(checkpoint) != {"sha256"}:
            raise NavigationRLTTrainingError("stage1 checkpoint contract differs")
        manifest_checkpoint_sha = _require_sha256(
            checkpoint.get("sha256"), "stage1 checkpoint SHA-256"
        )
        if manifest_checkpoint_sha != stage1_checkpoint_sha256:
            raise NavigationRLTTrainingError("run and CLI stage1 checkpoint hashes differ")
        _validate_stage2_support(
            value.get("stage2_support"),
            minimum_behavior_gate_count=minimum_gate_count,
        )
    return dict(value), model_config


def _full_state(
    model: FullyShardedDataParallel, *, rank0_only: bool
) -> dict[str, torch.Tensor]:
    config = FullStateDictConfig(offload_to_cpu=True, rank0_only=rank0_only)
    with FullyShardedDataParallel.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, config
    ):
        state = model.state_dict()
    normalized: dict[str, torch.Tensor] = {}
    for name, value in state.items():
        key = name
        for prefix in ("module.", "_fsdp_wrapped_module."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        normalized[key] = value.detach().cpu()
    return normalized


def _policy_state(wrapper_state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefix = "policy."
    result = {
        name[len(prefix) :]: value
        for name, value in wrapper_state.items()
        if name.startswith(prefix)
    }
    if len(result) != len(wrapper_state):
        unexpected = sorted(name for name in wrapper_state if not name.startswith(prefix))
        raise NavigationRLTTrainingError(
            f"FSDP wrapper state contains unexpected members {unexpected}"
        )
    return result


def _topology_tokens(
    arrays: Mapping[str, np.ndarray], prefix: str, config: NavigationRLTConfig
) -> tuple[np.ndarray, np.ndarray]:
    nodes = np.asarray(arrays[prefix + "topology_node_targets"])
    node_mask = np.asarray(arrays[prefix + "topology_node_mask"])
    global_values = np.asarray(arrays[prefix + "topology_global_targets"])
    global_mask = np.asarray(arrays[prefix + "topology_global_mask"])
    if nodes.shape[-1] != config.topology_node_dim:
        raise NavigationRLTTrainingError("replay topology node width differs")
    if global_values.shape[-1] != config.topology_global_dim:
        raise NavigationRLTTrainingError("replay topology global width differs")
    packed = pack_topology_tokens(
        torch.from_numpy(nodes.astype(np.float32, copy=False)),
        torch.from_numpy(node_mask.astype(np.bool_, copy=False)),
        torch.from_numpy(global_values.astype(np.float32, copy=False)),
        torch.from_numpy(global_mask.astype(np.bool_, copy=False)),
        config=config,
    )
    return packed.tokens.numpy(), packed.token_mask.numpy()


def _reference_embedding(arrays: Mapping[str, np.ndarray], prefix: str) -> np.ndarray:
    result = np.asarray(arrays[prefix + "reference_embedding"])
    if result.ndim != 2 or result.shape[1] != REFERENCE_DIM:
        raise NavigationRLTTrainingError("replay reference embedding width differs")
    if not np.issubdtype(result.dtype, np.floating) or not np.isfinite(result).all():
        raise NavigationRLTTrainingError("replay reference embedding is invalid")
    return result


def _canonical_actions(
    modes: np.ndarray, resolved_se2: np.ndarray, *, dtype: np.dtype
) -> np.ndarray:
    modes = np.asarray(modes, dtype=np.int64)
    resolved = np.asarray(resolved_se2, dtype=dtype)
    rows = int(modes.shape[0])
    result = np.zeros((rows, CANONICAL_ACTION_DIM), dtype=dtype)
    result[np.arange(rows, dtype=np.int64), modes] = 1.0
    result[:, GATE_COUNT:] = resolved
    return result


def _compiler_arrays(
    arrays: Mapping[str, np.ndarray], prefix: str
) -> dict[str, np.ndarray]:
    trajectory = np.asarray(arrays[prefix + "reference_trajectory"])
    dtype = trajectory.dtype
    rows = int(trajectory.shape[0])
    selected_index = np.asarray(arrays[prefix + "reference_selected_index"], dtype=np.int64)
    selected = trajectory[np.arange(rows, dtype=np.int64), selected_index]
    executable_reference = selected.copy()
    executable_reference[:, 0] = np.clip(executable_reference[:, 0], 0.0, 0.25)
    executable_reference[:, 2] = np.clip(
        executable_reference[:, 2], -math.radians(30.0), math.radians(30.0)
    )
    native_stop = np.asarray(arrays[prefix + "native_stop"])
    reference_gate = np.where(native_stop, STOP_GATE, KEEP_GATE).astype(np.int64)
    reference_action = _canonical_actions(
        np.full((rows,), KEEP_GATE, dtype=np.int64), executable_reference, dtype=dtype
    )
    adjust_base_action = _canonical_actions(
        np.full((rows,), ADJUST_GATE, dtype=np.int64), executable_reference, dtype=dtype
    )
    stop_action = _canonical_actions(
        np.full((rows,), STOP_GATE, dtype=np.int64),
        np.zeros_like(selected),
        dtype=dtype,
    )
    parent_index = np.asarray(arrays[prefix + "recovery_parent_index"], dtype=np.int64)
    parent_valid = parent_index >= 0
    nodes = np.asarray(arrays[prefix + "topology_node_targets"])
    parent_se2 = np.zeros((rows, 3), dtype=dtype)
    available = np.flatnonzero(parent_valid)
    if available.size:
        parent_features = nodes[available, parent_index[available]]
        forward = parent_features[:, 0]
        lateral = parent_features[:, 1]
        parent_se2[available, 0] = forward
        parent_se2[available, 1] = lateral
        nonzero = np.hypot(forward, lateral) > 1.0e-8
        parent_se2[available, 2] = np.where(
            nonzero, np.arctan2(lateral, forward), 0.0
        )
    recover_action = _canonical_actions(
        np.full((rows,), RECOVER_GATE, dtype=np.int64), parent_se2, dtype=dtype
    )
    residual_basis = np.zeros((rows, 2, CANONICAL_ACTION_DIM), dtype=dtype)
    residual_basis[:, 0, GATE_COUNT] = 1.0
    residual_basis[:, 1, GATE_COUNT + 2] = 1.0
    return {
        "valid_gate_mask": np.asarray(arrays[prefix + "valid_mode_mask"]),
        "reference_action": reference_action,
        "adjust_base_action": adjust_base_action,
        "adjust_residual_basis": residual_basis,
        "stop_action": stop_action,
        "recover_parent_action": recover_action,
        "recover_parent_valid": parent_valid,
        "recover_parent_index": parent_index,
        "reference_gate": reference_gate,
    }


def _assert_compiler_replay_parity(
    arrays: Mapping[str, np.ndarray], compiler: Mapping[str, np.ndarray]
) -> None:
    """Prove the offline compiler reproduces credited online canonical actions."""

    modes = np.asarray(arrays["executed_mode"], dtype=np.int64)
    actual = _canonical_actions(
        modes,
        np.asarray(arrays["resolved_se2"]),
        dtype=np.asarray(arrays["resolved_se2"]).dtype,
    )
    candidates = np.stack(
        (
            compiler["reference_action"],
            compiler["adjust_base_action"],
            compiler["stop_action"],
            compiler["recover_parent_action"],
        ),
        axis=1,
    )
    expected = candidates[np.arange(modes.shape[0], dtype=np.int64), modes].copy()
    adjust = modes == ADJUST_GATE
    if bool(adjust.any()):
        expected[adjust] += np.einsum(
            "br,bra->ba",
            np.asarray(arrays["bounded_residual"])[adjust],
            compiler["adjust_residual_basis"][adjust],
        )
        expected[adjust, GATE_COUNT] = np.clip(
            expected[adjust, GATE_COUNT], 0.0, 0.25
        )
        expected[adjust, GATE_COUNT + 2] = np.clip(
            expected[adjust, GATE_COUNT + 2],
            -math.radians(30.0),
            math.radians(30.0),
        )
    credited = np.asarray(arrays["credit_mask"])
    if not np.allclose(
        actual[credited], expected[credited], rtol=1.0e-5, atol=1.0e-6
    ):
        difference = float(np.max(np.abs(actual[credited] - expected[credited])))
        raise NavigationRLTTrainingError(
            f"offline compiler differs from credited online canonical action by {difference}"
        )


class _LocalTensorDataset(Dataset):
    def __init__(self, tensors: Mapping[str, torch.Tensor]):
        if not tensors:
            raise NavigationRLTTrainingError("local training dataset is empty")
        lengths = {int(value.shape[0]) for value in tensors.values()}
        if len(lengths) != 1:
            raise NavigationRLTTrainingError("local tensor lengths differ")
        self.tensors = dict(tensors)
        self.rows = lengths.pop()

    def __len__(self) -> int:
        return self.rows

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {name: value[index] for name, value in self.tensors.items()}


def _partition_indices(
    item_count: int,
    *,
    world_size: int,
    rank: int,
    seed: int,
) -> tuple[list[int], int, str]:
    if item_count < world_size:
        raise NavigationRLTTrainingError("training items are fewer than distributed ranks")
    padding = (-item_count) % world_size
    padded_count = item_count + padding
    index_dataset = list(range(padded_count))
    audit_sets: list[set[int]] = []
    selected: list[int] | None = None
    for audit_rank in range(world_size):
        sampler = DistributedSampler(
            index_dataset,
            num_replicas=world_size,
            rank=audit_rank,
            shuffle=True,
            seed=seed,
            drop_last=False,
        )
        sampler.set_epoch(0)
        values = list(iter(sampler))
        if len(values) != len(set(values)):
            raise NavigationRLTTrainingError("DistributedSampler duplicated a padded index")
        audit_sets.append(set(values))
        if audit_rank == rank:
            selected = values
    union = set().union(*audit_sets)
    if union != set(range(padded_count)):
        raise NavigationRLTTrainingError("DistributedSampler partition is incomplete")
    if sum(len(values) for values in audit_sets) != len(union):
        raise NavigationRLTTrainingError("DistributedSampler partitions overlap")
    if selected is None:
        raise NavigationRLTTrainingError("rank has no sampler partition")
    digest = hashlib.sha256(
        np.asarray(selected, dtype=np.int64).tobytes()
    ).hexdigest()
    return selected, padding, digest


def _manifest_layout(manifest: Mapping[str, object]) -> tuple[list[dict[str, object]], list[int]]:
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise NavigationRLTTrainingError("replay manifest has no shards")
    normalized: list[dict[str, object]] = []
    offsets = [0]
    for value in shards:
        if not isinstance(value, dict):
            raise NavigationRLTTrainingError("replay shard receipt is not an object")
        row_count = _positive_int(value.get("row_count"), "replay shard row_count")
        normalized.append(value)
        offsets.append(offsets[-1] + row_count)
    if offsets[-1] != manifest.get("transition_count"):
        raise NavigationRLTTrainingError("replay manifest row layout differs")
    return normalized, offsets


def _load_shard(
    receipt: Mapping[str, object],
    *,
    source_fingerprint: str,
    representation: str,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    return load_episode_shard(
        Path(str(receipt["path"])),
        expected_sha256=str(receipt["sha256"]),
        expected_source_fingerprint=source_fingerprint,
        expected_representation_kind=representation,
    )


def _append_piece(
    pieces: dict[str, list[np.ndarray]], name: str, value: np.ndarray
) -> None:
    pieces[name].append(np.asarray(value))


def _finish_pieces(
    pieces: Mapping[str, Sequence[np.ndarray]], positions: Sequence[np.ndarray]
) -> dict[str, torch.Tensor]:
    if not positions:
        raise NavigationRLTTrainingError("rank materialized no replay item")
    joined_positions = np.concatenate(positions, axis=0)
    order = np.argsort(joined_positions, kind="stable")
    result: dict[str, torch.Tensor] = {}
    for name, values in pieces.items():
        joined = np.concatenate(values, axis=0)[order]
        if joined.dtype == np.bool_:
            tensor = torch.from_numpy(joined.astype(np.bool_, copy=False))
        elif np.issubdtype(joined.dtype, np.integer):
            tensor = torch.from_numpy(joined.astype(np.int64, copy=False))
        else:
            tensor = torch.from_numpy(joined.astype(np.float32, copy=False))
        result[name] = tensor
    return result


def _materialize_stage1(
    manifest: Mapping[str, object],
    config: NavigationRLTConfig,
    *,
    world_size: int,
    rank: int,
    seed: int,
) -> tuple[_LocalTensorDataset, dict[str, object]]:
    shards, offsets = _manifest_layout(manifest)
    transition_count = offsets[-1]
    item_count = 2 * transition_count
    selected, padding, sampler_hash = _partition_indices(
        item_count, world_size=world_size, rank=rank, seed=seed
    )
    real_assignments: dict[int, list[tuple[int, int, bool]]] = defaultdict(list)
    dummy_positions: list[int] = []
    for local_position, item_index in enumerate(selected):
        if item_index >= item_count:
            dummy_positions.append(local_position)
            continue
        next_state = item_index >= transition_count
        row_index = item_index - transition_count if next_state else item_index
        shard_index = bisect_right(offsets, row_index) - 1
        real_assignments[shard_index].append(
            (local_position, row_index - offsets[shard_index], next_state)
        )
    pieces: dict[str, list[np.ndarray]] = defaultdict(list)
    positions: list[np.ndarray] = []
    first_example: dict[str, np.ndarray] | None = None
    original_prefix_lengths: list[int] = []
    selected_prefix_lengths: list[int] = []
    source = str(manifest["source_fingerprint"])
    for shard_index in sorted(real_assignments):
        arrays, metadata = _load_shard(
            shards[shard_index],
            source_fingerprint=source,
            representation=REPRESENTATION_PREFIX_HIDDEN,
        )
        if int(metadata["prefix_hidden_dim"]) != config.prefix_hidden_dim:
            raise NavigationRLTTrainingError("replay prefix hidden width differs")
        if int(metadata["prefix_max_tokens"]) > config.max_artifact_prefix_tokens:
            raise NavigationRLTTrainingError(
                "replay prefix artifact exceeds max_artifact_prefix_tokens"
            )
        rows = real_assignments[shard_index]
        current = [value for value in rows if not value[2]]
        future = [value for value in rows if value[2]]
        for group, prefix in ((current, ""), (future, "next_")):
            if not group:
                continue
            local_positions = np.asarray([value[0] for value in group], dtype=np.int64)
            row_indices = np.asarray([value[1] for value in group], dtype=np.int64)
            selected_prefix = select_prefix_tokens(
                torch.from_numpy(
                    np.asarray(arrays[prefix + "prefix_hidden"]).astype(
                        np.float32, copy=False
                    )
                ),
                torch.from_numpy(
                    np.asarray(arrays[prefix + "prefix_mask"]).astype(
                        np.bool_, copy=False
                    )
                ),
                config=config,
            )
            topology, topology_mask = _topology_tokens(arrays, prefix, config)
            example = {
                "contextual_prefix_hidden": selected_prefix.contextual_prefix_hidden.numpy()[
                    row_indices
                ],
                "prefix_mask": selected_prefix.prefix_mask.numpy()[row_indices],
                "topology_tokens": topology[row_indices],
                "topology_mask": topology_mask[row_indices],
                "valid": np.ones((len(group),), dtype=np.bool_),
            }
            original_prefix_lengths.extend(
                selected_prefix.original_valid_lengths.numpy()[row_indices].tolist()
            )
            selected_prefix_lengths.extend(
                selected_prefix.selected_valid_lengths.numpy()[row_indices].tolist()
            )
            positions.append(local_positions)
            for name, value in example.items():
                _append_piece(pieces, name, value)
            if first_example is None:
                first_example = {name: value[:1].copy() for name, value in example.items()}
    if first_example is None:
        raise NavigationRLTTrainingError("rank partition contains no real Stage-1 row")
    for local_position in dummy_positions:
        positions.append(np.asarray([local_position], dtype=np.int64))
        for name, value in first_example.items():
            dummy = value.copy()
            if name == "valid":
                dummy.fill(False)
            _append_piece(pieces, name, dummy)
    tensors = _finish_pieces(pieces, positions)
    return _LocalTensorDataset(tensors), {
        "global_real_items": item_count,
        "global_padding_items": padding,
        "local_sampler_items": len(selected),
        "local_real_items": int(tensors["valid"].sum().item()),
        "local_padding_items": int((~tensors["valid"]).sum().item()),
        "sampler_indices_sha256": sampler_hash,
        "sampler": "torch.utils.data.DistributedSampler",
        "sampler_real_row_duplication": 0,
        "prefix_selection_schema": config.prefix_selection_schema,
        "prefix_source_width_cap": config.max_artifact_prefix_tokens,
        "prefix_policy_width": config.max_prefix_tokens,
        "original_prefix_valid_length_min": min(original_prefix_lengths),
        "original_prefix_valid_length_max": max(original_prefix_lengths),
        "selected_prefix_valid_length_min": min(selected_prefix_lengths),
        "selected_prefix_valid_length_max": max(selected_prefix_lengths),
    }


def _eligible_stage2_rows(
    manifest: Mapping[str, object],
    *,
    checkpoint_sha256: str,
    minimum_behavior_gate_count: int,
) -> tuple[list[int], dict[str, object]]:
    shards, offsets = _manifest_layout(manifest)
    source = str(manifest["source_fingerprint"])
    eligible: list[int] = []
    credit_ineligible = 0
    gate_mismatch = 0
    gate_counts = np.zeros((GATE_COUNT,), dtype=np.int64)
    for shard_index, receipt in enumerate(shards):
        arrays, metadata = _load_shard(
            receipt,
            source_fingerprint=source,
            representation=REPRESENTATION_FROZEN_Z_RL,
        )
        if metadata.get("rlt_checkpoint_sha256") != checkpoint_sha256:
            raise NavigationRLTTrainingError(
                "Stage-2 replay token checkpoint differs from the frozen Stage-1 checkpoint"
            )
        credit = np.asarray(arrays["credit_mask"], dtype=np.bool_)
        requested = np.asarray(arrays["requested_mode"])
        executed = np.asarray(arrays["executed_mode"])
        if (
            requested.ndim != 1
            or executed.shape != requested.shape
            or credit.shape != requested.shape
        ):
            raise NavigationRLTTrainingError(
                "Stage-2 gate/credit arrays have incompatible shapes"
            )
        if bool(
            ((requested < 0) | (requested >= GATE_COUNT)).any()
            or ((executed < 0) | (executed >= GATE_COUNT)).any()
        ):
            raise NavigationRLTTrainingError(
                "Stage-2 gate arrays contain an invalid gate index"
            )
        matched = requested == executed
        eligible_mask = credit & matched
        rows = np.flatnonzero(eligible_mask).astype(np.int64)
        eligible.extend((rows + offsets[shard_index]).tolist())
        credit_ineligible += int((~credit).sum())
        gate_mismatch += int((credit & ~matched).sum())
        gate_counts += np.bincount(
            requested[eligible_mask].astype(np.int64), minlength=GATE_COUNT
        )[:GATE_COUNT]
    if not eligible:
        raise NavigationRLTTrainingError("Stage-2 replay contains no credit-eligible row")
    support_mask = gate_counts >= minimum_behavior_gate_count
    evidence = {
        "eligibility_schema": STAGE2_ELIGIBILITY_SCHEMA,
        "global_training_items": len(eligible),
        "credit_ineligible_items_excluded": credit_ineligible,
        "gate_mismatch_items_excluded": gate_mismatch,
        "gate_names": list(GATE_NAMES),
        "gate_counts": [int(value) for value in gate_counts.tolist()],
        "gate_support_mask": [bool(value) for value in support_mask.tolist()],
        "minimum_behavior_gate_count": minimum_behavior_gate_count,
    }
    return eligible, evidence


def _distributed_stage2_eligibility(
    manifest: Mapping[str, object],
    *,
    checkpoint_sha256: str,
    rank: int,
    minimum_behavior_gate_count: int,
) -> tuple[list[int], dict[str, object]]:
    payload: list[object | None] = [None]
    if rank == 0:
        try:
            eligible, evidence = _eligible_stage2_rows(
                manifest,
                checkpoint_sha256=checkpoint_sha256,
                minimum_behavior_gate_count=minimum_behavior_gate_count,
            )
            payload[0] = {
                "status": "passed",
                "eligible": eligible,
                "evidence": evidence,
            }
        except Exception as error:  # broadcast the fixed-data failure to every rank
            payload[0] = {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
    dist.broadcast_object_list(payload, src=0)
    result = payload[0]
    if not isinstance(result, dict) or result.get("status") != "passed":
        detail = result.get("error") if isinstance(result, dict) else "missing receipt"
        raise NavigationRLTTrainingError(
            f"distributed Stage-2 eligibility scan failed: {detail}"
        )
    eligible = result.get("eligible")
    evidence = result.get("evidence")
    if not isinstance(eligible, list) or not isinstance(evidence, dict):
        raise NavigationRLTTrainingError("Stage-2 eligibility receipt is invalid")
    validated_support = _validate_stage2_support(
        {
            name: evidence.get(name)
            for name in (
                "eligibility_schema",
                "gate_names",
                "gate_counts",
                "gate_support_mask",
                "minimum_behavior_gate_count",
            )
        },
        minimum_behavior_gate_count=minimum_behavior_gate_count,
    )
    normalized_evidence = dict(evidence)
    normalized_evidence.update(validated_support)
    return [int(value) for value in eligible], normalized_evidence


def _materialize_stage2(
    manifest: Mapping[str, object],
    config: NavigationRLTConfig,
    *,
    checkpoint_sha256: str,
    world_size: int,
    rank: int,
    seed: int,
    minimum_behavior_gate_count: int,
    expected_stage2_support: Mapping[str, object],
) -> tuple[_LocalTensorDataset, dict[str, object]]:
    shards, offsets = _manifest_layout(manifest)
    eligible, gate_evidence = _distributed_stage2_eligibility(
        manifest,
        checkpoint_sha256=checkpoint_sha256,
        rank=rank,
        minimum_behavior_gate_count=minimum_behavior_gate_count,
    )
    observed_support = {
        name: gate_evidence[name]
        for name in (
            "eligibility_schema",
            "gate_names",
            "gate_counts",
            "gate_support_mask",
            "minimum_behavior_gate_count",
        )
    }
    if observed_support != dict(expected_stage2_support):
        raise NavigationRLTTrainingError(
            "Stage-2 replay gate support differs from the run manifest"
        )
    selected_items, padding, sampler_hash = _partition_indices(
        len(eligible), world_size=world_size, rank=rank, seed=seed
    )
    assignments: dict[int, list[tuple[int, int]]] = defaultdict(list)
    dummy_positions: list[int] = []
    for local_position, item_index in enumerate(selected_items):
        if item_index >= len(eligible):
            dummy_positions.append(local_position)
            continue
        global_row = eligible[item_index]
        shard_index = bisect_right(offsets, global_row) - 1
        assignments[shard_index].append(
            (local_position, global_row - offsets[shard_index])
        )
    source = str(manifest["source_fingerprint"])
    pieces: dict[str, list[np.ndarray]] = defaultdict(list)
    positions: list[np.ndarray] = []
    first_example: dict[str, np.ndarray] | None = None
    discount_gamma: float | None = None
    for shard_index in sorted(assignments):
        arrays, metadata = _load_shard(
            shards[shard_index],
            source_fingerprint=source,
            representation=REPRESENTATION_FROZEN_Z_RL,
        )
        if metadata.get("rlt_checkpoint_sha256") != checkpoint_sha256:
            raise NavigationRLTTrainingError("Stage-2 replay checkpoint binding differs")
        gamma = float(metadata["discount_gamma"])
        if discount_gamma is None:
            discount_gamma = gamma
        elif gamma != discount_gamma:
            raise NavigationRLTTrainingError("replay discount gamma differs across shards")
        if int(metadata["z_rl_dim"]) != Z_RL_DIM:
            raise NavigationRLTTrainingError("Stage-2 replay z_rl width differs")
        group = assignments[shard_index]
        local_positions = np.asarray([value[0] for value in group], dtype=np.int64)
        row_indices = np.asarray([value[1] for value in group], dtype=np.int64)
        current_compiler = _compiler_arrays(arrays, "")
        next_compiler = _compiler_arrays(arrays, "next_")
        _assert_compiler_replay_parity(arrays, current_compiler)
        reference = _reference_embedding(arrays, "")
        next_reference = _reference_embedding(arrays, "next_")
        executed = _canonical_actions(
            np.asarray(arrays["executed_mode"], dtype=np.int64),
            np.asarray(arrays["resolved_se2"]),
            dtype=np.asarray(arrays["resolved_se2"]).dtype,
        )
        done = np.asarray(arrays["done"])
        duration = np.asarray(arrays["duration"], dtype=np.float64)
        discount = np.where(done, 0.0, np.power(gamma, duration)).astype(
            np.asarray(arrays["discounted_reward"]).dtype
        )
        example: dict[str, np.ndarray] = {
            "z_rl": np.asarray(arrays["z_rl"])[row_indices],
            "next_z_rl": np.asarray(arrays["next_z_rl"])[row_indices],
            "reference_embedding": reference[row_indices],
            "next_reference_embedding": next_reference[row_indices],
            "canonical_executed_action": executed[row_indices],
            "reward": np.asarray(arrays["discounted_reward"])[row_indices],
            "discount": discount[row_indices],
            "reference_gate": current_compiler["reference_gate"][row_indices],
            "behavior_gate": np.asarray(arrays["requested_mode"])[row_indices],
            "behavior_raw_residual": np.asarray(arrays["raw_residual"])[row_indices],
            "behavior_policy_version": np.asarray(arrays["behavior_policy_version"])[row_indices],
            "sample_weight": np.ones((len(group),), dtype=np.float32),
            "valid": np.ones((len(group),), dtype=np.bool_),
        }
        for field, value in current_compiler.items():
            if field != "reference_gate":
                example["compiler_" + field] = value[row_indices]
        for field, value in next_compiler.items():
            if field != "reference_gate":
                example["next_compiler_" + field] = value[row_indices]
        positions.append(local_positions)
        for name, value in example.items():
            _append_piece(pieces, name, value)
        if first_example is None:
            first_example = {name: value[:1].copy() for name, value in example.items()}
    if first_example is None or discount_gamma is None:
        raise NavigationRLTTrainingError("rank partition contains no real Stage-2 row")
    for local_position in dummy_positions:
        positions.append(np.asarray([local_position], dtype=np.int64))
        for name, value in first_example.items():
            dummy = value.copy()
            if name == "valid":
                dummy.fill(False)
            _append_piece(pieces, name, dummy)
    tensors = _finish_pieces(pieces, positions)
    return _LocalTensorDataset(tensors), {
        "global_real_items": len(eligible),
        "global_padding_items": padding,
        "global_credit_ineligible_items_excluded": gate_evidence[
            "credit_ineligible_items_excluded"
        ],
        "global_gate_mismatch_items_excluded": gate_evidence[
            "gate_mismatch_items_excluded"
        ],
        "eligibility_schema": gate_evidence["eligibility_schema"],
        "gate_names": gate_evidence["gate_names"],
        "gate_counts": gate_evidence["gate_counts"],
        "gate_support_mask": gate_evidence["gate_support_mask"],
        "minimum_behavior_gate_count": gate_evidence[
            "minimum_behavior_gate_count"
        ],
        "local_sampler_items": len(selected_items),
        "local_real_items": int(tensors["valid"].sum().item()),
        "local_padding_items": int((~tensors["valid"]).sum().item()),
        "sampler_indices_sha256": sampler_hash,
        "sampler": "torch.utils.data.DistributedSampler",
        "sampler_real_row_duplication": 0,
        "discount_gamma": discount_gamma,
    }


def _move_batch(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        name: value.to(device, non_blocking=True)
        for name, value in batch.items()
    }


def _select_batch(
    batch: Mapping[str, torch.Tensor], mask: torch.Tensor
) -> dict[str, torch.Tensor]:
    if bool(mask.any().item()):
        return {name: value[mask] for name, value in batch.items() if name != "valid"}
    # A padded-only rank still executes the FSDP forward so collective order is
    # identical.  Its loss is multiplied by zero after the forward.
    return {name: value[:1] for name, value in batch.items() if name != "valid"}


def _context(batch: Mapping[str, torch.Tensor]) -> NavigationRLTContext:
    return NavigationRLTContext(
        contextual_prefix_hidden=batch["contextual_prefix_hidden"],
        prefix_mask=batch["prefix_mask"],
        topology_tokens=batch["topology_tokens"],
        topology_mask=batch["topology_mask"],
    )


def _compiler(batch: Mapping[str, torch.Tensor], prefix: str) -> NavigationRLTCompilerFields:
    fields: dict[str, torch.Tensor] = {
        "valid_gate_mask": batch[prefix + "compiler_valid_gate_mask"],
        "reference_action": batch[prefix + "compiler_reference_action"],
        "adjust_base_action": batch[prefix + "compiler_adjust_base_action"],
        "adjust_residual_basis": batch[prefix + "compiler_adjust_residual_basis"],
        "stop_action": batch[prefix + "compiler_stop_action"],
        "recover_parent_action": batch[prefix + "compiler_recover_parent_action"],
        "recover_parent_valid": batch[prefix + "compiler_recover_parent_valid"],
        "recover_parent_index": batch[prefix + "compiler_recover_parent_index"],
    }
    return NavigationRLTCompilerFields(**fields)


class NavigationRLTFSDPRoot(nn.Module):
    """Root-forward adapter; all train operations cross the FSDP root."""

    def __init__(self, policy: NavigationRLTPolicy, phase: str):
        super().__init__()
        self.policy = policy
        self.phase = phase

    def forward(
        self,
        *,
        operation: str,
        batch: Mapping[str, torch.Tensor] | None = None,
        loss_config: Stage1LossConfig | Stage2LossConfig | None = None,
        generator: torch.Generator | None = None,
    ) -> Mapping[str, torch.Tensor]:
        if operation == "stage1_loss":
            if self.phase != "stage1" or batch is None or not isinstance(loss_config, Stage1LossConfig):
                raise NavigationRLTTrainingError("invalid Stage-1 root-forward request")
            return self.policy.stage1_loss(
                _context(batch), loss_config, generator=generator
            )
        if operation == "stage2_loss":
            if self.phase != "stage2" or batch is None or not isinstance(loss_config, Stage2LossConfig):
                raise NavigationRLTTrainingError("invalid Stage-2 root-forward request")
            replay = NavigationRLTZTransitionBatch(
                z_rl=batch["z_rl"],
                next_z_rl=batch["next_z_rl"],
                reference_embedding=batch["reference_embedding"],
                next_reference_embedding=batch["next_reference_embedding"],
                compiler=_compiler(batch, ""),
                next_compiler=_compiler(batch, "next_"),
                canonical_executed_action=batch["canonical_executed_action"],
                reward=batch["reward"],
                discount=batch["discount"],
                reference_gate=batch["reference_gate"],
                behavior_gate=batch["behavior_gate"],
                behavior_raw_residual=batch["behavior_raw_residual"],
                behavior_policy_version=batch["behavior_policy_version"],
                sample_weight=batch["sample_weight"],
            )
            return self.policy.stage2_loss_from_z(
                replay, loss_config, generator=generator
            )
        raise NavigationRLTTrainingError(f"unknown FSDP root operation {operation!r}")


def _fsdp_soft_update_targets(
    model: FullyShardedDataParallel, target_tau: float
) -> None:
    """Run the policy-owned target update while original parameters are materialized.

    FSDP exposes non-leaf parameter views during ``forward``.  The public policy
    helper reasserts that target critics are frozen, which is valid only after
    the forward has returned and full original parameters are available.  All
    ranks enter this context so corresponding online/target tensors have the
    same complete layout before the policy performs its update.
    """

    with FullyShardedDataParallel.summon_full_params(
        model,
        recurse=True,
        writeback=True,
        rank0_only=False,
        offload_to_cpu=False,
    ):
        root = model.module
        if not isinstance(root, NavigationRLTFSDPRoot) or root.phase != "stage2":
            raise NavigationRLTTrainingError("invalid FSDP target-update root")
        root.policy.soft_update_targets(float(target_tau))


class NavigationRLTFSDPManager(FSDPModelManager):
    def __init__(
        self,
        cfg,
        *,
        world_size: int,
        rank: int,
        phase: str,
        model_config: NavigationRLTConfig,
        stage1_checkpoint: Path | None,
        expected_stage1_sha256: str | None,
        stage2_gate_support_mask: Sequence[bool] | None,
        stage2_gate_counts: Sequence[int] | None,
    ):
        super().__init__(cfg, world_size=world_size, rank=rank)
        self.phase = phase
        self.model_config = model_config
        self.stage1_checkpoint = stage1_checkpoint
        self.expected_stage1_sha256 = expected_stage1_sha256
        self.stage2_gate_support_mask = (
            None
            if stage2_gate_support_mask is None
            else tuple(bool(value) for value in stage2_gate_support_mask)
        )
        self.stage2_gate_counts = (
            None
            if stage2_gate_counts is None
            else tuple(int(value) for value in stage2_gate_counts)
        )
        self.stage2_initial_gate_prior: tuple[float, ...] | None = None
        self.initial_component_hashes: dict[str, str] | None = None
        self.initial_checkpoint_metadata: dict[str, object] | None = None

    def model_provider_func(self):
        if self.phase == "stage1":
            policy = NavigationRLTPolicy(self.model_config)
            for module in (policy.actor, policy.q1, policy.q2):
                module.requires_grad_(False)
        else:
            if self.stage1_checkpoint is None or self.expected_stage1_sha256 is None:
                raise NavigationRLTTrainingError("Stage 2 has no frozen Stage-1 checkpoint")
            if sha256_file(self.stage1_checkpoint) != self.expected_stage1_sha256:
                raise NavigationRLTTrainingError("Stage-1 checkpoint SHA-256 differs")
            policy, metadata = NavigationRLTPolicy.load_checkpoint(
                self.stage1_checkpoint, map_location="cpu"
            )
            if policy.config != self.model_config:
                raise NavigationRLTTrainingError("checkpoint model config differs")
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
                raise NavigationRLTTrainingError(
                    "checkpoint is not an eligible frozen Stage 1"
                )
            expected_hash = extra.get("stage1_parameter_sha256")
            actual_hash = _component_hash(policy.state_dict(), ("stage1.",))
            if expected_hash != actual_hash:
                raise NavigationRLTTrainingError("checkpoint Stage-1 freeze receipt differs")
            if (
                self.stage2_gate_support_mask is None
                or len(self.stage2_gate_support_mask) != GATE_COUNT
            ):
                raise NavigationRLTTrainingError(
                    "Stage 2 has no valid replay gate support mask"
                )
            policy.set_gate_support_mask(
                torch.tensor(self.stage2_gate_support_mask, dtype=torch.bool)
            )
            if (
                self.stage2_gate_counts is None
                or len(self.stage2_gate_counts) != GATE_COUNT
            ):
                raise NavigationRLTTrainingError(
                    "Stage 2 has no valid replay gate counts"
                )
            initial_prior = policy.initialize_stage2_gate_prior_from_behavior_counts(
                self.stage2_gate_counts
            )
            self.stage2_initial_gate_prior = tuple(
                float(value) for value in initial_prior.cpu().tolist()
            )
            self.initial_checkpoint_metadata = metadata
        state = policy.state_dict()
        self.initial_component_hashes = {
            "stage1": _component_hash(state, ("stage1.",)),
            "actor": _component_hash(state, ("actor.",)),
            "q1": _component_hash(state, ("q1.",)),
            "q2": _component_hash(state, ("q2.",)),
            "target_q1": _component_hash(state, ("target_q1.",)),
            "target_q2": _component_hash(state, ("target_q2.",)),
        }
        return NavigationRLTFSDPRoot(policy, self.phase)


def _manager_config(
    model_config: NavigationRLTConfig,
    training: Mapping[str, object],
    *,
    total_steps: int,
    phase: str,
):
    return OmegaConf.create(
        {
            "model": {
                "model_type": "navigation_rlt_" + phase,
                "model_path": "",
                "precision": "fp32",
                "is_lora": False,
                "navigation_rlt_config": model_config.to_dict(),
            },
            "optim": {
                "lr": float(training["learning_rate"]),
                "value_lr": float(training["learning_rate"]),
                "adam_beta1": 0.9,
                "adam_beta2": 0.999,
                "adam_eps": 1.0e-8,
                "weight_decay": float(training["weight_decay"]),
                "clip_grad": float(training["clip_grad"]),
                "total_training_steps": total_steps,
                "lr_warmup_steps": 0,
                "lr_scheduler": "constant",
            },
            "fsdp_config": {
                "disable": True,
                "strategy": "fsdp",
                "sharding_strategy": "full_shard",
                "gradient_checkpointing": False,
                "cpu_offload": False,
                "offload_pin_memory": False,
                "reshard_after_forward": True,
                "enable_gradient_accumulation": False,
                "forward_prefetch": False,
                "limit_all_gathers": True,
                "backward_prefetch": None,
                "use_orig_params": True,
                "use_liger_kernel": False,
                "wrap_policy": {"disable": True},
                "mixed_precision": {
                    "param_dtype": None,
                    "reduce_dtype": None,
                    "buffer_dtype": None,
                },
                "amp_autocast": {"enabled": False, "precision": "bf16"},
                "grad_scaler": {"enabled": False},
            },
        }
    )


class _LossSummary:
    def __init__(self) -> None:
        self.steps = 0
        self.sums: dict[str, float] = defaultdict(float)
        self.minimum: dict[str, float] = {}
        self.maximum: dict[str, float] = {}
        self.first: dict[str, float] | None = None
        self.last: dict[str, float] | None = None
        self.snapshots: list[dict[str, float]] = []

    def add(self, values: Mapping[str, float]) -> None:
        if len(values) > MAX_LOSS_KEYS:
            raise NavigationRLTTrainingError("loss diagnostic key count exceeds cap")
        normalized = {name: float(value) for name, value in values.items()}
        if not all(math.isfinite(value) for value in normalized.values()):
            raise NavigationRLTTrainingError("loss diagnostics are non-finite")
        self.steps += 1
        if self.first is None:
            self.first = normalized
        self.last = normalized
        for name, value in normalized.items():
            self.sums[name] += value
            self.minimum[name] = min(value, self.minimum.get(name, value))
            self.maximum[name] = max(value, self.maximum.get(name, value))
        if len(self.snapshots) < MAX_LOSS_SNAPSHOTS:
            self.snapshots.append(normalized)

    def result(self) -> dict[str, object]:
        if self.steps < 1 or self.first is None or self.last is None:
            raise NavigationRLTTrainingError("training produced no loss diagnostic")
        return {
            "steps": self.steps,
            "first": self.first,
            "last": self.last,
            "mean": {name: value / self.steps for name, value in self.sums.items()},
            "minimum": self.minimum,
            "maximum": self.maximum,
            "bounded_snapshots": self.snapshots,
        }


def _global_diagnostic(
    values: Mapping[str, torch.Tensor],
    *,
    local_valid: torch.Tensor,
    global_valid: torch.Tensor,
) -> dict[str, float]:
    names = sorted(values)
    if len(names) > MAX_LOSS_KEYS:
        raise NavigationRLTTrainingError("policy returned too many diagnostics")
    stacked = torch.stack(
        [values[name].detach().float().reshape(()) * local_valid for name in names]
    )
    dist.all_reduce(stacked, op=dist.ReduceOp.SUM)
    stacked = stacked / global_valid
    return {name: float(stacked[index].item()) for index, name in enumerate(names)}


def _save_checkpoint_from_state(
    state: Mapping[str, torch.Tensor],
    config: NavigationRLTConfig,
    path: Path,
    *,
    phase: str,
    training_step: int,
    policy_version: int,
    initial_hashes: Mapping[str, str],
    run_manifest_sha256: str,
    replay_manifest_sha256: str,
    stage2_gate_support_mask: Sequence[bool] | None,
    stage2_actor_regularization: Mapping[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, str]]:
    _finite_state(state)
    policy = NavigationRLTPolicy(config)
    if set(state) != set(policy.state_dict()):
        raise NavigationRLTTrainingError(
            f"FSDP policy state differs: {sorted(set(state) ^ set(policy.state_dict()))}"
        )
    policy.load_state_dict(state, strict=True)
    if phase == "stage2":
        if (
            stage2_gate_support_mask is None
            or len(stage2_gate_support_mask) != GATE_COUNT
        ):
            raise NavigationRLTTrainingError(
                "Stage-2 checkpoint has no valid replay gate support mask"
            )
        policy.set_gate_support_mask(
            torch.tensor(stage2_gate_support_mask, dtype=torch.bool)
        )
    elif stage2_gate_support_mask is not None:
        raise NavigationRLTTrainingError(
            "Stage-1 checkpoint cannot carry Stage-2 gate support"
        )
    component_hashes = {
        "stage1": _component_hash(policy.state_dict(), ("stage1.",)),
        "actor": _component_hash(policy.state_dict(), ("actor.",)),
        "q1": _component_hash(policy.state_dict(), ("q1.",)),
        "q2": _component_hash(policy.state_dict(), ("q2.",)),
        "target_q1": _component_hash(policy.state_dict(), ("target_q1.",)),
        "target_q2": _component_hash(policy.state_dict(), ("target_q2.",)),
    }
    if phase == "stage1":
        if component_hashes["stage1"] == initial_hashes["stage1"]:
            raise NavigationRLTTrainingError("Stage-1 optimizer changed no representation parameter")
        for name in ("actor", "q1", "q2", "target_q1", "target_q2"):
            if component_hashes[name] != initial_hashes[name]:
                raise NavigationRLTTrainingError(
                    f"Stage-1 optimizer changed isolated {name} parameters"
                )
        policy.freeze_stage1()
    else:
        if component_hashes["stage1"] != initial_hashes["stage1"]:
            raise NavigationRLTTrainingError("Stage-2 optimizer changed frozen Stage-1")
        for name in ("actor", "q1", "q2"):
            if component_hashes[name] == initial_hashes[name]:
                raise NavigationRLTTrainingError(
                    f"Stage-2 optimizer changed no {name} parameter"
                )
        for name in ("target_q1", "target_q2"):
            if component_hashes[name] == initial_hashes[name]:
                raise NavigationRLTTrainingError(
                    f"Stage-2 {name} was never soft-updated"
                )
        policy.freeze_stage1()
    receipt = policy.save_checkpoint(
        path,
        training_step=training_step,
        policy_version=policy_version,
        extra_metadata={
            "training_phase": phase,
            "run_manifest_sha256": _require_sha256(
                run_manifest_sha256, "checkpoint run manifest SHA-256"
            ),
            "replay_manifest_sha256": _require_sha256(
                replay_manifest_sha256, "checkpoint replay manifest SHA-256"
            ),
            "stage1_parameter_sha256": component_hashes["stage1"],
            "actor_parameter_sha256": component_hashes["actor"],
            "q1_parameter_sha256": component_hashes["q1"],
            "q2_parameter_sha256": component_hashes["q2"],
            "target_q1_parameter_sha256": component_hashes["target_q1"],
            "target_q2_parameter_sha256": component_hashes["target_q2"],
            "initial_component_sha256": dict(initial_hashes),
            "stage2_actor_regularization": (
                None if stage2_actor_regularization is None
                else dict(stage2_actor_regularization)
            ),
        },
    )
    if int(receipt["size_bytes"]) > MAX_CHECKPOINT_BYTES:
        raise NavigationRLTTrainingError("checkpoint exceeds the 2 GiB hard cap")
    return receipt, component_hashes


def _independent_restore_receipt(
    checkpoint: Path, expected_sha256: str, expected_state_hash: str
) -> dict[str, object]:
    code = r'''import hashlib, json, sys
from pathlib import Path
import torch
from navigation_rlt_policy import NavigationRLTPolicy

def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def state_sha(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()

path = Path(sys.argv[1]).resolve(strict=True)
expected_file = sys.argv[2]
expected_state = sys.argv[3]
actual_file = file_sha(path)
if actual_file != expected_file:
    raise SystemExit("checkpoint file hash differs")
policy, metadata = NavigationRLTPolicy.load_checkpoint(path, map_location="cpu")
state = policy.state_dict()
actual_state = state_sha(state)
if actual_state != expected_state:
    raise SystemExit("checkpoint state hash differs")
if not bool(policy.stage1_frozen.item()):
    raise SystemExit("restored Stage 1 is not frozen")
if any(value.is_floating_point() and not bool(torch.isfinite(value).all()) for value in state.values()):
    raise SystemExit("restored state is non-finite")
print(json.dumps({
    "schema_name": "navigation_rlt_independent_restore",
    "schema_version": 1,
    "status": "passed",
    "checkpoint_sha256": actual_file,
    "state_sha256": actual_state,
    "training_step": metadata["training_step"],
    "policy_version": metadata["policy_version"],
    "stage1_frozen": bool(policy.stage1_frozen.item()),
    "state_tensor_count": len(state),
    "parameter_count": sum(value.numel() for value in state.values()),
    "stage2_actor_regularization": metadata.get("extra", {}).get("stage2_actor_regularization"),
}, sort_keys=True))
'''
    environment = dict(os.environ)
    module_dir = str(Path(__file__).resolve().parent)
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = module_dir if not existing else module_dir + os.pathsep + existing
    completed = subprocess.run(
        [sys.executable, "-c", code, str(checkpoint), expected_sha256, expected_state_hash],
        cwd=module_dir,
        env=environment,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=180,
    )
    try:
        receipt = json.loads(completed.stdout.strip())
    except json.JSONDecodeError as error:
        raise NavigationRLTTrainingError("independent restore returned invalid JSON") from error
    if not isinstance(receipt, dict) or receipt.get("status") != "passed":
        raise NavigationRLTTrainingError("independent checkpoint restore did not pass")
    return receipt


def _distributed_verify_replay(
    replay_manifest_path: Path,
    *,
    expected_sha256: str,
    expected_source_fingerprint: str,
    rank: int,
) -> None:
    payload: list[object | None] = [None]
    if rank == 0:
        try:
            verified = load_replay_manifest(
                replay_manifest_path,
                expected_sha256=expected_sha256,
                expected_source_fingerprint=expected_source_fingerprint,
                verify_shards=True,
            )
            payload[0] = {
                "status": "passed",
                "episode_count": verified["completed_episode_count"],
                "transition_count": verified["transition_count"],
            }
        except Exception as error:  # keep all ranks on the same failure path
            payload[0] = {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
    dist.broadcast_object_list(payload, src=0)
    result = payload[0]
    if not isinstance(result, dict) or result.get("status") != "passed":
        detail = result.get("error") if isinstance(result, dict) else "missing receipt"
        raise NavigationRLTTrainingError(
            f"rank-zero replay verification failed: {detail}"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)
    for phase in ("stage1", "stage2"):
        command = subparsers.add_parser(phase)
        command.add_argument("--run-manifest", required=True, type=Path)
        command.add_argument("--expected-run-manifest-sha256", required=True)
        command.add_argument("--replay-manifest", required=True, type=Path)
        command.add_argument("--expected-replay-manifest-sha256", required=True)
        command.add_argument("--output-dir", required=True, type=Path)
        command.add_argument("--required-world-size", type=int, default=8)
        if phase == "stage2":
            command.add_argument("--stage1-checkpoint", required=True, type=Path)
            command.add_argument("--expected-stage1-checkpoint-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    phase = str(args.phase)
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "-1"))
    if rank < 0 or local_rank < 0 or world_size != args.required_world_size:
        raise NavigationRLTTrainingError(
            f"Navigation-RLT requires torchrun world_size={args.required_world_size}; "
            f"received rank={rank}, local_rank={local_rank}, world_size={world_size}"
        )
    expected_run_sha = _require_sha256(
        args.expected_run_manifest_sha256, "expected run manifest SHA-256"
    )
    expected_replay_sha = _require_sha256(
        args.expected_replay_manifest_sha256,
        "expected replay manifest SHA-256",
    )
    run_manifest_path = args.run_manifest.resolve(strict=True)
    replay_manifest_path = args.replay_manifest.resolve(strict=True)
    if sha256_file(run_manifest_path) != expected_run_sha:
        raise NavigationRLTTrainingError("run manifest SHA-256 differs")
    if sha256_file(replay_manifest_path) != expected_replay_sha:
        raise NavigationRLTTrainingError("replay manifest SHA-256 differs")
    stage1_checkpoint = (
        args.stage1_checkpoint.resolve(strict=True) if phase == "stage2" else None
    )
    expected_stage1_sha = (
        _require_sha256(
            args.expected_stage1_checkpoint_sha256,
            "expected Stage-1 checkpoint SHA-256",
        )
        if phase == "stage2"
        else None
    )
    if stage1_checkpoint is not None and sha256_file(stage1_checkpoint) != expected_stage1_sha:
        raise NavigationRLTTrainingError("Stage-1 checkpoint SHA-256 differs")

    raw_run_manifest = _load_json(run_manifest_path, "run manifest")
    run_manifest, model_config = _validate_run_manifest(
        raw_run_manifest,
        phase=phase,
        replay_manifest_sha256=expected_replay_sha,
        stage1_checkpoint_sha256=expected_stage1_sha,
    )
    if int(run_manifest["required_world_size"]) != world_size:
        raise NavigationRLTTrainingError("run manifest world size differs")
    replay_contract = run_manifest["replay"]
    replay_manifest = load_replay_manifest(
        replay_manifest_path,
        expected_sha256=expected_replay_sha,
        expected_source_fingerprint=str(replay_contract["source_fingerprint"]),
        verify_shards=False,
    )
    if replay_manifest["run_id"] != run_manifest["run_id"]:
        raise NavigationRLTTrainingError("run and replay manifest run_id differ")
    if replay_manifest["representation_kind"] != replay_contract["representation_kind"]:
        raise NavigationRLTTrainingError("run and replay representations differ")

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)
    seed = int(run_manifest["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    start_gpu = _gpu_snapshot(local_rank)

    collective = torch.tensor(float(rank), device=device)
    dist.all_reduce(collective, op=dist.ReduceOp.SUM)
    expected_collective = world_size * (world_size - 1) / 2
    if float(collective.item()) != expected_collective:
        raise NavigationRLTTrainingError("NCCL rank collective differs")
    _distributed_verify_replay(
        replay_manifest_path,
        expected_sha256=expected_replay_sha,
        expected_source_fingerprint=str(replay_contract["source_fingerprint"]),
        rank=rank,
    )

    output_dir = args.output_dir.resolve()
    if rank == 0:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise NavigationRLTTrainingError("output directory must be absent or empty")
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "ranks").mkdir(parents=True, exist_ok=True)
    dist.barrier()

    training = run_manifest["training"]
    stage2_support = (
        None if phase == "stage1" else dict(run_manifest["stage2_support"])
    )
    if phase == "stage1":
        dataset, data_receipt = _materialize_stage1(
            replay_manifest,
            model_config,
            world_size=world_size,
            rank=rank,
            seed=seed,
        )
    else:
        assert expected_stage1_sha is not None
        assert stage2_support is not None
        dataset, data_receipt = _materialize_stage2(
            replay_manifest,
            model_config,
            checkpoint_sha256=expected_stage1_sha,
            world_size=world_size,
            rank=rank,
            seed=seed,
            minimum_behavior_gate_count=int(
                training["minimum_behavior_gate_count"]
            ),
            expected_stage2_support=stage2_support,
        )
    batch_size = int(training["batch_size_per_rank"])
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    epochs = int(training["epochs"])
    total_steps = len(loader) * epochs
    if total_steps < 1:
        raise NavigationRLTTrainingError("training has zero optimizer steps")
    manager = NavigationRLTFSDPManager(
        _manager_config(model_config, training, total_steps=total_steps, phase=phase),
        world_size=world_size,
        rank=rank,
        phase=phase,
        model_config=model_config,
        stage1_checkpoint=stage1_checkpoint,
        expected_stage1_sha256=expected_stage1_sha,
        stage2_gate_support_mask=(
            None if stage2_support is None else stage2_support["gate_support_mask"]
        ),
        stage2_gate_counts=(
            None if stage2_support is None else stage2_support["gate_counts"]
        ),
    )
    manager.setup_model_and_optimizer()
    if not isinstance(manager.model, FullyShardedDataParallel):
        raise NavigationRLTTrainingError("RLinf did not wrap Navigation-RLT in FSDP")
    if manager.initial_component_hashes is None:
        raise NavigationRLTTrainingError("model provider did not record initial state")

    if phase == "stage1":
        loss_config: Stage1LossConfig | Stage2LossConfig = Stage1LossConfig(
            prefix_mask_ratio=training["prefix_mask_ratio"],
            topology_mask_ratio=training["topology_mask_ratio"],
            prefix_weight=training["prefix_weight"],
            topology_weight=training["topology_weight"],
        )
    else:
        loss_config = Stage2LossConfig(
            reference_gate_weight=training["reference_gate_weight"],
            reference_residual_weight=training["reference_residual_weight"],
            reference_dropout_probability=training["reference_dropout_probability"],
            target_tau=training["target_tau"],
            cql_alpha=training["cql_alpha"],
            cql_temperature=training["cql_temperature"],
            behavior_kl_weight=float(training.get("behavior_kl_weight", 0.0)),
            behavior_gate_prior=manager.stage2_initial_gate_prior,
        )
    stage2_actor_regularization = None
    if phase == "stage2" and float(training.get("behavior_kl_weight", 0.0)) > 0.0:
        assert stage2_support is not None
        stage2_actor_regularization = {
            "objective_schema": training["objective_schema"],
            "direction": "behavior_to_actor_forward_kl",
            "behavior_kl_weight": float(training["behavior_kl_weight"]),
            "behavior_gate_prior": list(manager.stage2_initial_gate_prior),
            "gate_counts": list(stage2_support["gate_counts"]),
            "supported_uniform_mix": STAGE2_BEHAVIOR_PRIOR_UNIFORM_MIX,
        }
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + rank * 104729)
    losses = _LossSummary()
    gradient_steps = 0
    action_probability_epochs: list[dict[str, float]] = []
    collapsed_epoch_streak = 0
    probability_names = (
        "keep_probability_mean",
        "adjust_probability_mean",
        "stop_probability_mean",
        "recover_probability_mean",
    )
    training_started = time.time()
    for _epoch in range(epochs):
        epoch_losses = _LossSummary()
        epoch_probability_sums = {name: 0.0 for name in probability_names}
        epoch_probability_steps = 0
        for raw_batch in loader:
            batch = _move_batch(raw_batch, device)
            valid = batch["valid"]
            local_valid = valid.float().sum()
            global_valid = local_valid.clone()
            dist.all_reduce(global_valid, op=dist.ReduceOp.SUM)
            if float(global_valid.item()) <= 0.0:
                raise NavigationRLTTrainingError("distributed minibatch has no real row")
            selected_batch = _select_batch(batch, valid)
            manager.optimizer.zero_grad(set_to_none=True)
            output = manager.model(
                operation=phase + "_loss",
                batch=selected_batch,
                loss_config=loss_config,
                generator=generator,
            )
            if "loss" not in output or not bool(torch.isfinite(output["loss"]).item()):
                raise NavigationRLTTrainingError("training objective is absent or non-finite")
            gradient_scale = world_size * local_valid / global_valid
            loss = output["loss"] * gradient_scale
            loss.backward()
            grad_norm, learning_rates = manager.optimizer_step()
            gradient_steps += 1
            if phase == "stage2":
                with torch.no_grad():
                    _fsdp_soft_update_targets(
                        manager.model, float(training["target_tau"])
                    )
            diagnostic = _global_diagnostic(
                output, local_valid=local_valid, global_valid=global_valid
            )
            diagnostic["grad_norm"] = float(grad_norm)
            diagnostic["learning_rate"] = float(learning_rates[0])
            losses.add(diagnostic)
            epoch_losses.add(diagnostic)
            if phase == "stage2":
                for name in probability_names:
                    epoch_probability_sums[name] += diagnostic[name]
                epoch_probability_steps += 1
        if phase == "stage2":
            if epoch_probability_steps < 1:
                raise NavigationRLTTrainingError("Stage-2 epoch has no probability diagnostic")
            epoch_probabilities = {
                name: value / epoch_probability_steps
                for name, value in epoch_probability_sums.items()
            }
            action_probability_epochs.append(epoch_probabilities)
            collapsed_epoch_streak = (
                collapsed_epoch_streak + 1
                if max(epoch_probabilities.values()) > 0.98
                else 0
            )
            if rank == 0:
                epoch_loss_means = epoch_losses.result()["mean"]
                print(
                    json.dumps(
                        {
                            "event": "navigation_rlt_stage2_epoch",
                            "epoch_index": _epoch,
                            "optimizer_steps": gradient_steps,
                            "global_gate_probabilities": epoch_probabilities,
                            "collapsed_epoch_streak": collapsed_epoch_streak,
                            "actor_rl_loss": epoch_loss_means["actor_rl_loss"],
                            "reference_anchor_loss": epoch_loss_means[
                                "reference_anchor_loss"
                            ],
                            "critic_bellman_loss": epoch_loss_means[
                                "critic_bellman_loss"
                            ],
                            "cql_loss": epoch_loss_means["cql_loss"],
                            "losses_mean": epoch_loss_means,
                            "finite": True,
                        },
                        sort_keys=True,
                        allow_nan=False,
                    ),
                    flush=True,
                )
            if collapsed_epoch_streak >= 2:
                raise NavigationRLTTrainingError(
                    "Stage-2 actor probability exceeded 98% for two consecutive epochs"
                )
    training_completed = time.time()

    if gradient_steps != total_steps:
        raise NavigationRLTTrainingError("optimizer step count differs")
    dist.barrier()
    wrapper_state = _full_state(manager.model, rank0_only=True)
    checkpoint_receipt: dict[str, object] | None = None
    component_hashes: dict[str, str] | None = None
    restore_receipt: dict[str, object] | None = None
    if rank == 0:
        state = _policy_state(wrapper_state)
        checkpoint_path = output_dir / "checkpoints" / "latest" / "navigation_rlt_policy.pt"
        checkpoint_receipt, component_hashes = _save_checkpoint_from_state(
            state,
            model_config,
            checkpoint_path,
            phase=phase,
            training_step=total_steps,
            policy_version=0 if phase == "stage1" else int(training["policy_version"]),
            initial_hashes=manager.initial_component_hashes,
            run_manifest_sha256=expected_run_sha,
            replay_manifest_sha256=expected_replay_sha,
            stage2_gate_support_mask=(
                None
                if stage2_support is None
                else stage2_support["gate_support_mask"]
            ),
            stage2_actor_regularization=stage2_actor_regularization,
        )
        expected_state_hash = _tensor_hash(
            NavigationRLTPolicy.load_checkpoint(checkpoint_path, map_location="cpu")[0].state_dict()
        )
        restore_receipt = _independent_restore_receipt(
            checkpoint_path,
            str(checkpoint_receipt["sha256"]),
            expected_state_hash,
        )
        if restore_receipt.get("stage2_actor_regularization") != stage2_actor_regularization:
            raise NavigationRLTTrainingError("restored Stage-2 actor regularization differs")
        _atomic_json(output_dir / "checkpoint_restore_receipt.json", restore_receipt)

    end_gpu = _gpu_snapshot(local_rank)
    rank_receipt = {
        "schema_name": "navigation_rlt_rank_receipt",
        "schema_version": 1,
        "phase": phase,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "pid": os.getpid(),
        "collective_sum": float(collective.item()),
        "expected_collective_sum": expected_collective,
        "gradient_steps": gradient_steps,
        "data": data_receipt,
        "gpu_start": start_gpu,
        "gpu_end": end_gpu,
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "training_wall_time_s": training_completed - training_started,
        "optimizer_steps_per_second": gradient_steps
        / max(training_completed - training_started, 1.0e-9),
        "real_item_epochs_per_second": (
            int(data_receipt["local_real_items"]) * epochs
        )
        / max(training_completed - training_started, 1.0e-9),
        "completed_unix_s": time.time(),
    }
    _atomic_json(output_dir / "ranks" / f"rank_{rank:03d}.json", rank_receipt)
    dist.barrier()

    if rank == 0:
        rank_receipts = [
            _load_json(output_dir / "ranks" / f"rank_{index:03d}.json", "rank receipt")
            for index in range(world_size)
        ]
        uuids = {str(row["gpu_start"]["gpu_uuid"]) for row in rank_receipts}
        if len(uuids) != world_size:
            raise NavigationRLTTrainingError("rank receipts do not prove unique GPUs")
        if any(int(row["gradient_steps"]) != total_steps for row in rank_receipts):
            raise NavigationRLTTrainingError("one or more ranks missed optimizer steps")
        sampler_hashes = {str(row["data"]["sampler_indices_sha256"]) for row in rank_receipts}
        if len(sampler_hashes) != world_size:
            raise NavigationRLTTrainingError("rank sampler partitions are not distinct")
        if phase == "stage2":
            assert stage2_support is not None
            support_fields = tuple(stage2_support)
            for row in rank_receipts:
                rank_support = {
                    name: row["data"].get(name) for name in support_fields
                }
                if rank_support != stage2_support:
                    raise NavigationRLTTrainingError(
                        "rank Stage-2 gate support evidence differs"
                    )
        summary = {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "phase": phase,
            "run_id": run_manifest["run_id"],
            "rlinf_backend": "FSDPModelManager",
            "fsdp_wrapper": type(manager.model).__name__,
            "world_size": world_size,
            "unique_gpu_uuid_count": len(uuids),
            "rank_receipts": rank_receipts,
            "run_manifest": {
                "path": str(run_manifest_path),
                "sha256": expected_run_sha,
                "canonical_content_sha256": _canonical_json_hash(run_manifest),
            },
            "replay_manifest": {
                "path": str(replay_manifest_path),
                "sha256": expected_replay_sha,
                "source_fingerprint": replay_manifest["source_fingerprint"],
                "representation_kind": replay_manifest["representation_kind"],
                "episode_count": replay_manifest["completed_episode_count"],
                "transition_count": replay_manifest["transition_count"],
                "behavior_policy_versions": replay_manifest["behavior_policy_versions"],
            },
            "stage1_checkpoint_input": (
                None
                if stage1_checkpoint is None
                else {"path": str(stage1_checkpoint), "sha256": expected_stage1_sha}
            ),
            "final_checkpoint": checkpoint_receipt,
            "component_parameter_sha256": component_hashes,
            "independent_restore": restore_receipt,
            "training": dict(training),
            "optimizer_steps_per_rank": total_steps,
            "losses": losses.result(),
            "action_probability_epochs": action_probability_epochs,
            "collapsed_epoch_streak": collapsed_epoch_streak,
            "checkpoint_hard_cap_bytes": MAX_CHECKPOINT_BYTES,
            "run_hard_cap_bytes": MAX_RUN_BYTES,
            "started_unix_s": started,
            "completed_unix_s": time.time(),
        }
        if phase == "stage2":
            assert stage2_support is not None
            summary["stage2_support"] = dict(stage2_support)
            if manager.stage2_initial_gate_prior is None:
                raise NavigationRLTTrainingError(
                    "Stage-2 behavior gate prior evidence is absent"
                )
            summary["stage2_actor_initialization"] = {
                "schema_name": "navigation_rlt_support_aware_behavior_gate_prior",
                "schema_version": 1,
                "source": "global_credit_eligible_matching_behavior_gate_counts",
                "gate_names": list(GATE_NAMES),
                "gate_counts": list(stage2_support["gate_counts"]),
                "gate_support_mask": list(stage2_support["gate_support_mask"]),
                "supported_uniform_mix": STAGE2_BEHAVIOR_PRIOR_UNIFORM_MIX,
                "gate_probabilities": list(manager.stage2_initial_gate_prior),
                "maximum_gate_probability": max(manager.stage2_initial_gate_prior),
            }
        _atomic_json(output_dir / "training_result.json", summary)
        inventory = {
            str(path.relative_to(output_dir)): {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(output_dir.rglob("*"))
            if path.is_file()
            and path.name not in {"artifact_inventory.json", "COMPLETED.json"}
        }
        _atomic_json(output_dir / "artifact_inventory.json", inventory)
        if _tree_size(output_dir) > MAX_RUN_BYTES:
            raise NavigationRLTTrainingError("training output exceeds the 16 GiB hard cap")
        _atomic_json(
            output_dir / "COMPLETED.json",
            {
                "schema_name": "navigation_rlt_training_completion",
                "schema_version": 1,
                "status": "completed",
                "phase": phase,
                "training_result_sha256": sha256_file(output_dir / "training_result.json"),
                "artifact_inventory_sha256": sha256_file(output_dir / "artifact_inventory.json"),
            },
        )
        print(json.dumps(summary, sort_keys=True, allow_nan=False))

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
