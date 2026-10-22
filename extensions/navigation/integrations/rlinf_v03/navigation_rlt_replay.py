"""Faithful Navigation-RLT episode replay shards.

One ``.npz`` shard is one complete episode collected with one fixed behavior
policy version. Different shards may contain different behavior versions: the
replay is deliberately off-policy. Current and next state are stored directly,
so TD bootstrapping never joins rows across episodes.

The representation contract has two explicit modes. ``prefix_hidden`` stores
frozen LightNav final-prefix hidden states and masks for Stage 1.
``frozen_z_rl`` stores tokens produced by a frozen Stage-1 encoder for Stage 2
and binds them to its checkpoint SHA-256. Both store the current and next
64-D reference embeddings directly, while retaining causal topology targets,
the native trajectory, executed action, safety intervention, primitive rewards,
and episode termination as separate evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Iterable, Mapping, Sequence

import numpy as np


SCHEMA_NAME = "navigation_rlt_episode_replay"
SCHEMA_VERSION = 3
MANIFEST_SCHEMA_NAME = "navigation_rlt_replay_manifest"
MANIFEST_SCHEMA_VERSION = 3
LIGHTNAV_SOURCE_REVISION = "a645828d81a8439651172197ca80a75dc1377977"

REPRESENTATION_PREFIX_HIDDEN = "prefix_hidden"
REPRESENTATION_FROZEN_Z_RL = "frozen_z_rl"
FEATURE_STAGE_PREFIX_HIDDEN = "stage1_prefix_hidden"
FEATURE_STAGE_FROZEN_Z_RL = "stage2_frozen_z_rl"
REPRESENTATION_STAGES = {
    REPRESENTATION_PREFIX_HIDDEN: FEATURE_STAGE_PREFIX_HIDDEN,
    REPRESENTATION_FROZEN_Z_RL: FEATURE_STAGE_FROZEN_Z_RL,
}

REFERENCE_TRAJECTORY_SHAPE = (10, 3)
REFERENCE_SCHEMA = "trajectory10_selected_topology_reference64_v1"
REFERENCE_DIM = 64
MODE_NAMES = ("KEEP", "ADJUST", "STOP", "RECOVER")
KEEP_MODE = 0
ADJUST_MODE = 1
STOP_MODE = 2
RECOVER_MODE = 3
MODE_COUNT = len(MODE_NAMES)
RESIDUAL_DIM = 2
RESOLVED_SE2_DIM = 3
RESIDUAL_LIMITS = np.asarray((0.05, math.radians(10.0)), dtype=np.float64)
EXECUTOR_FORWARD_RANGE_M = (0.0, 0.25)
EXECUTOR_YAW_RANGE_RAD = (-math.radians(30.0), math.radians(30.0))

END_REASON_NAMES = (
    "running",
    "success",
    "policy_stop_success",
    "policy_stop_failure",
    "timeout",
    "max_steps",
    "collision",
    "safety_abort",
    "environment_terminal",
    "environment_truncated",
    "collector_abort",
)
END_REASON_RUNNING = 0

_STATE_ARRAY_BASES = (
    "topology_node_targets",
    "topology_node_mask",
    "topology_global_targets",
    "topology_global_mask",
    "reference_embedding",
    "reference_trajectory",
    "reference_selected_index",
    "native_stop",
    "valid_mode_mask",
    "recovery_parent_index",
)
STATE_ARRAYS = _STATE_ARRAY_BASES + tuple(f"next_{name}" for name in _STATE_ARRAY_BASES)
PREFIX_REPRESENTATION_ARRAYS = (
    "prefix_hidden",
    "prefix_mask",
    "next_prefix_hidden",
    "next_prefix_mask",
)
Z_REPRESENTATION_ARRAYS = ("z_rl", "next_z_rl")
TRANSITION_ARRAYS = (
    "requested_mode",
    "executed_mode",
    "raw_residual",
    "bounded_residual",
    "resolved_se2",
    "safety_override",
    "credit_mask",
    "primitive_rewards",
    "executed_mask",
    "duration",
    "discounted_reward",
    "collision",
    "blocked",
    "terminated",
    "truncated",
    "done",
    "end_reason",
    "decision_index",
    "physical_step",
    "behavior_policy_version",
)
COMMON_ARRAYS = STATE_ARRAYS + TRANSITION_ARRAYS

_COMMON_METADATA_FIELDS = {
    "schema_name",
    "schema_version",
    "run_id",
    "episode_id",
    "reset_token",
    "scene_id",
    "instruction_sha256",
    "split_label",
    "env_slot",
    "row_count",
    "lightnav_source_revision",
    "lightnav_hook_sha256",
    "lightnav_weight_sha256",
    "tokenizer_sha256",
    "processor_sha256",
    "history_config_sha256",
    "prefix_feature_schema",
    "prefix_feature_source_sha256",
    "topology_schema",
    "reference_schema",
    "reference_dim",
    "action_schema",
    "executor_schema",
    "representation_kind",
    "feature_stage",
    "topology_max_nodes",
    "topology_node_dim",
    "topology_global_dim",
    "max_primitive_steps",
    "discount_gamma",
    "behavior_policy_id",
    "behavior_policy_version",
}
_PREFIX_METADATA_FIELDS = {"prefix_max_tokens", "prefix_hidden_dim"}
_Z_METADATA_FIELDS = {"z_rl_dim", "rlt_checkpoint_sha256"}
_SOURCE_SHA_FIELDS = (
    "lightnav_hook_sha256",
    "lightnav_weight_sha256",
    "tokenizer_sha256",
    "processor_sha256",
    "history_config_sha256",
    "prefix_feature_source_sha256",
)
_SOURCE_IDENTITY_FIELDS = (
    "schema_name",
    "schema_version",
    "lightnav_source_revision",
    "lightnav_hook_sha256",
    "lightnav_weight_sha256",
    "tokenizer_sha256",
    "processor_sha256",
    "history_config_sha256",
    "prefix_feature_schema",
    "prefix_feature_source_sha256",
    "topology_schema",
    "reference_schema",
    "reference_dim",
    "action_schema",
    "executor_schema",
    "representation_kind",
    "feature_stage",
    "topology_max_nodes",
    "topology_node_dim",
    "topology_global_dim",
    "max_primitive_steps",
    "discount_gamma",
)


class ReplayContractError(ValueError):
    """Raised when replay cannot prove the Navigation-RLT contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object, name: str) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ReplayContractError(f"{name} is not canonical JSON: {exc}") from exc


def _json_mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ReplayContractError(f"{name} must be an object")
    decoded = json.loads(_canonical_json(value, name).decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ReplayContractError(f"{name} must remain an object")
    return decoded


def _required_string(metadata: Mapping[str, object], name: str) -> str:
    value = metadata.get(name)
    if not isinstance(value, str) or not value:
        raise ReplayContractError(f"metadata.{name} must be a non-empty string")
    return value


def _validate_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ReplayContractError(f"{name} must be lowercase SHA-256")
    if any(character not in "0123456789abcdef" for character in value):
        raise ReplayContractError(f"{name} must be lowercase SHA-256")
    return value


def _positive_integer(value: object, name: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ReplayContractError(f"{name} must be an integer >= {minimum}")
    return value


def _nonnegative_integer(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ReplayContractError(f"{name} must be a non-negative integer")
    return value


def _source_payload(metadata: Mapping[str, object]) -> dict[str, object]:
    payload = {name: metadata[name] for name in _SOURCE_IDENTITY_FIELDS}
    representation = str(metadata["representation_kind"])
    if representation == REPRESENTATION_PREFIX_HIDDEN:
        payload.update(
            prefix_max_tokens=metadata["prefix_max_tokens"],
            prefix_hidden_dim=metadata["prefix_hidden_dim"],
        )
    elif representation == REPRESENTATION_FROZEN_Z_RL:
        payload.update(
            z_rl_dim=metadata["z_rl_dim"],
            rlt_checkpoint_sha256=metadata["rlt_checkpoint_sha256"],
        )
    else:
        raise ReplayContractError("metadata.representation_kind is unknown")
    return payload


def source_fingerprint(metadata: Mapping[str, object]) -> str:
    """Hash immutable source, schema, shape, and discount choices."""

    return hashlib.sha256(_canonical_json(_source_payload(metadata), "source identity")).hexdigest()


def validate_metadata(
    metadata: Mapping[str, object], *, rows: int | None = None
) -> dict[str, object]:
    result = _json_mapping(metadata, "metadata")
    if result.get("schema_name") != SCHEMA_NAME:
        raise ReplayContractError("replay schema name differs")
    if result.get("schema_version") != SCHEMA_VERSION:
        raise ReplayContractError("replay schema version differs")

    representation = result.get("representation_kind")
    if representation not in REPRESENTATION_STAGES:
        raise ReplayContractError("metadata.representation_kind is unknown")
    expected_fields = set(_COMMON_METADATA_FIELDS)
    expected_fields.update(
        _PREFIX_METADATA_FIELDS
        if representation == REPRESENTATION_PREFIX_HIDDEN
        else _Z_METADATA_FIELDS
    )
    supplied_fingerprint = result.get("source_fingerprint")
    if supplied_fingerprint is not None:
        expected_fields.add("source_fingerprint")
    if set(result) != expected_fields:
        missing = sorted(expected_fields - set(result))
        extra = sorted(set(result) - expected_fields)
        raise ReplayContractError(
            f"metadata members differ; missing={missing}, extra={extra}"
        )

    for name in (
        "run_id",
        "episode_id",
        "reset_token",
        "scene_id",
        "instruction_sha256",
        "prefix_feature_schema",
        "topology_schema",
        "reference_schema",
        "action_schema",
        "executor_schema",
        "behavior_policy_id",
        "split_label",
    ):
        _required_string(result, name)
    if result["split_label"] != "train_update":
        raise ReplayContractError("Navigation-RLT replay must come from train_update")
    if result.get("lightnav_source_revision") != LIGHTNAV_SOURCE_REVISION:
        raise ReplayContractError("LightNav source revision differs from the fixed contract")
    _validate_sha256(result["instruction_sha256"], "metadata.instruction_sha256")
    for name in _SOURCE_SHA_FIELDS:
        _validate_sha256(result[name], f"metadata.{name}")

    if result.get("feature_stage") != REPRESENTATION_STAGES[representation]:
        raise ReplayContractError("feature_stage and representation_kind disagree")
    if result.get("reference_schema") != REFERENCE_SCHEMA:
        raise ReplayContractError("metadata.reference_schema differs from the fixed contract")
    if (
        type(result.get("reference_dim")) is not int
        or result["reference_dim"] != REFERENCE_DIM
    ):
        raise ReplayContractError(
            f"metadata.reference_dim must be the fixed width {REFERENCE_DIM}"
        )
    if representation == REPRESENTATION_PREFIX_HIDDEN:
        _positive_integer(result["prefix_max_tokens"], "metadata.prefix_max_tokens")
        _positive_integer(result["prefix_hidden_dim"], "metadata.prefix_hidden_dim")
    else:
        _positive_integer(result["z_rl_dim"], "metadata.z_rl_dim")
        _validate_sha256(
            result["rlt_checkpoint_sha256"], "metadata.rlt_checkpoint_sha256"
        )

    _positive_integer(result["topology_max_nodes"], "metadata.topology_max_nodes")
    _positive_integer(result["topology_node_dim"], "metadata.topology_node_dim", minimum=2)
    _positive_integer(result["topology_global_dim"], "metadata.topology_global_dim")
    _positive_integer(result["max_primitive_steps"], "metadata.max_primitive_steps")
    row_count = _positive_integer(result["row_count"], "metadata.row_count")
    if rows is not None and row_count != rows:
        raise ReplayContractError("metadata.row_count differs from replay arrays")
    _nonnegative_integer(result["env_slot"], "metadata.env_slot")
    _nonnegative_integer(
        result["behavior_policy_version"], "metadata.behavior_policy_version"
    )
    gamma = result.get("discount_gamma")
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)):
        raise ReplayContractError("metadata.discount_gamma must be numeric")
    gamma = float(gamma)
    if not math.isfinite(gamma) or not 0.0 < gamma <= 1.0:
        raise ReplayContractError("metadata.discount_gamma must be finite in (0,1]")
    result["discount_gamma"] = gamma

    computed_fingerprint = source_fingerprint(result)
    if supplied_fingerprint is not None:
        _validate_sha256(supplied_fingerprint, "metadata.source_fingerprint")
        if supplied_fingerprint != computed_fingerprint:
            raise ReplayContractError("metadata.source_fingerprint differs from its sources")
    result["source_fingerprint"] = computed_fingerprint
    return result


def required_arrays(metadata_or_representation: Mapping[str, object] | str) -> tuple[str, ...]:
    if isinstance(metadata_or_representation, Mapping):
        representation = metadata_or_representation.get("representation_kind")
    else:
        representation = metadata_or_representation
    if representation == REPRESENTATION_PREFIX_HIDDEN:
        representation_arrays = PREFIX_REPRESENTATION_ARRAYS
    elif representation == REPRESENTATION_FROZEN_Z_RL:
        representation_arrays = Z_REPRESENTATION_ARRAYS
    else:
        raise ReplayContractError("unknown replay representation")
    return representation_arrays + COMMON_ARRAYS


def _floating(array: np.ndarray, shape: tuple[int, ...], name: str) -> None:
    if array.shape != shape or not np.issubdtype(array.dtype, np.floating):
        raise ReplayContractError(f"{name} must be floating with shape {shape}")
    if not np.isfinite(array).all():
        raise ReplayContractError(f"{name} must contain only finite values")


def _bool_array(array: np.ndarray, shape: tuple[int, ...], name: str) -> None:
    if array.shape != shape or array.dtype != np.bool_:
        raise ReplayContractError(f"{name} must be bool with shape {shape}")


def _integer_vector(
    array: np.ndarray, rows: int, name: str, *, signed: bool = False
) -> None:
    if array.shape != (rows,) or not np.issubdtype(array.dtype, np.integer):
        raise ReplayContractError(f"{name} must be an integer vector with {rows} rows")
    if signed and not np.issubdtype(array.dtype, np.signedinteger):
        raise ReplayContractError(f"{name} must use a signed integer dtype")


def _prefix_mask(mask: np.ndarray, name: str) -> np.ndarray:
    lengths = mask.sum(axis=1, dtype=np.int64)
    if bool((lengths < 1).any()):
        raise ReplayContractError(f"{name} contains an empty row")
    expected = np.arange(mask.shape[1], dtype=np.int64)[None, :] < lengths[:, None]
    if not np.array_equal(mask, expected):
        raise ReplayContractError(f"{name} must describe right-padded prefixes")
    return lengths


def _masked_padding_is_zero(values: np.ndarray, mask: np.ndarray, name: str) -> None:
    if values.ndim not in (mask.ndim, mask.ndim + 1):
        raise ReplayContractError(f"{name} and its mask have incompatible ranks")
    padded = values[~mask]
    if padded.size and bool((padded != 0).any()):
        raise ReplayContractError(f"{name} padding must be exactly zero")


def _validate_state(
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, object],
    *,
    prefix: str,
    rows: int,
) -> None:
    representation = str(metadata["representation_kind"])
    if representation == REPRESENTATION_PREFIX_HIDDEN:
        hidden = arrays[prefix + "prefix_hidden"]
        mask = arrays[prefix + "prefix_mask"]
        hidden_shape = (
            rows,
            int(metadata["prefix_max_tokens"]),
            int(metadata["prefix_hidden_dim"]),
        )
        _floating(hidden, hidden_shape, prefix + "prefix_hidden")
        _bool_array(mask, hidden_shape[:2], prefix + "prefix_mask")
        _prefix_mask(mask, prefix + "prefix_mask")
        _masked_padding_is_zero(hidden, mask, prefix + "prefix_hidden")
    else:
        _floating(
            arrays[prefix + "z_rl"],
            (rows, int(metadata["z_rl_dim"])),
            prefix + "z_rl",
        )

    max_nodes = int(metadata["topology_max_nodes"])
    node_dim = int(metadata["topology_node_dim"])
    global_dim = int(metadata["topology_global_dim"])
    node_targets = arrays[prefix + "topology_node_targets"]
    node_mask = arrays[prefix + "topology_node_mask"]
    global_targets = arrays[prefix + "topology_global_targets"]
    global_mask = arrays[prefix + "topology_global_mask"]
    _floating(node_targets, (rows, max_nodes, node_dim), prefix + "topology_node_targets")
    _bool_array(node_mask, (rows, max_nodes), prefix + "topology_node_mask")
    _prefix_mask(node_mask, prefix + "topology_node_mask")
    _masked_padding_is_zero(node_targets, node_mask, prefix + "topology_node_targets")
    _floating(global_targets, (rows, global_dim), prefix + "topology_global_targets")
    _bool_array(global_mask, (rows, global_dim), prefix + "topology_global_mask")
    if bool((global_mask.sum(axis=1) < 1).any()):
        raise ReplayContractError(f"{prefix}topology_global_mask contains an empty row")
    _masked_padding_is_zero(global_targets, global_mask, prefix + "topology_global_targets")

    trajectory = arrays[prefix + "reference_trajectory"]
    reference_embedding = arrays[prefix + "reference_embedding"]
    selected_index = arrays[prefix + "reference_selected_index"]
    native_stop = arrays[prefix + "native_stop"]
    valid_modes = arrays[prefix + "valid_mode_mask"]
    recovery_parent = arrays[prefix + "recovery_parent_index"]
    _floating(
        reference_embedding,
        (rows, REFERENCE_DIM),
        prefix + "reference_embedding",
    )
    _floating(
        trajectory,
        (rows, *REFERENCE_TRAJECTORY_SHAPE),
        prefix + "reference_trajectory",
    )
    _integer_vector(selected_index, rows, prefix + "reference_selected_index")
    if bool(((selected_index < 0) | (selected_index >= REFERENCE_TRAJECTORY_SHAPE[0])).any()):
        raise ReplayContractError(f"{prefix}reference_selected_index is out of range")
    _bool_array(native_stop, (rows,), prefix + "native_stop")
    _bool_array(valid_modes, (rows, MODE_COUNT), prefix + "valid_mode_mask")
    if bool((valid_modes.sum(axis=1) < 1).any()):
        raise ReplayContractError(f"{prefix}valid_mode_mask contains an empty row")
    if not bool(valid_modes[:, STOP_MODE].all()):
        raise ReplayContractError(f"{prefix}STOP must always be valid")
    expected_motion_valid = ~native_stop
    if not np.array_equal(valid_modes[:, KEEP_MODE], expected_motion_valid):
        raise ReplayContractError(f"{prefix}KEEP validity disagrees with native_stop")
    if not np.array_equal(valid_modes[:, ADJUST_MODE], expected_motion_valid):
        raise ReplayContractError(f"{prefix}ADJUST validity disagrees with native_stop")

    _integer_vector(recovery_parent, rows, prefix + "recovery_parent_index", signed=True)
    if bool(((recovery_parent < -1) | (recovery_parent >= max_nodes)).any()):
        raise ReplayContractError(f"{prefix}recovery_parent_index is out of range")
    parent_available = recovery_parent >= 0
    if not np.array_equal(valid_modes[:, RECOVER_MODE], parent_available):
        raise ReplayContractError(
            f"{prefix}RECOVER validity disagrees with recovery parent availability"
        )
    available_rows = np.flatnonzero(parent_available)
    if available_rows.size:
        parent_is_valid = node_mask[available_rows, recovery_parent[available_rows]]
        if not bool(parent_is_valid.all()):
            raise ReplayContractError(
                f"{prefix}recovery_parent_index points to a padded topology node"
            )


def _selected_reference(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    rows = arrays["reference_selected_index"].shape[0]
    return arrays["reference_trajectory"][
        np.arange(rows, dtype=np.int64),
        arrays["reference_selected_index"].astype(np.int64),
    ]


def _validate_actions(
    arrays: Mapping[str, np.ndarray], metadata: Mapping[str, object], rows: int
) -> None:
    del metadata
    requested = arrays["requested_mode"]
    executed = arrays["executed_mode"]
    _integer_vector(requested, rows, "requested_mode")
    _integer_vector(executed, rows, "executed_mode")
    for name, values in (("requested_mode", requested), ("executed_mode", executed)):
        if bool(((values < 0) | (values >= MODE_COUNT)).any()):
            raise ReplayContractError(f"{name} is outside {MODE_NAMES}")

    valid = arrays["valid_mode_mask"]
    row_indices = np.arange(rows, dtype=np.int64)
    if not bool(valid[row_indices, requested.astype(np.int64)].all()):
        raise ReplayContractError("requested_mode contains an invalid policy action")
    safety_override = arrays["safety_override"]
    credit_mask = arrays["credit_mask"]
    _bool_array(safety_override, (rows,), "safety_override")
    _bool_array(credit_mask, (rows,), "credit_mask")
    if not np.array_equal(credit_mask, ~safety_override):
        raise ReplayContractError("credit_mask must be false exactly on safety overrides")
    if bool(((requested != executed) & ~safety_override).any()):
        raise ReplayContractError("requested/executed mode differs without safety override")
    if bool((~valid[row_indices, executed.astype(np.int64)] & ~safety_override).any()):
        raise ReplayContractError("executed_mode is invalid without a safety override")

    recovery_parent = arrays["recovery_parent_index"]
    recovery_rows = (requested == RECOVER_MODE) | (executed == RECOVER_MODE)
    if bool((recovery_rows & (recovery_parent < 0)).any()):
        raise ReplayContractError("RECOVER action has no causal parent")

    raw = arrays["raw_residual"]
    bounded = arrays["bounded_residual"]
    resolved = arrays["resolved_se2"]
    _floating(raw, (rows, RESIDUAL_DIM), "raw_residual")
    _floating(bounded, (rows, RESIDUAL_DIM), "bounded_residual")
    _floating(resolved, (rows, RESOLVED_SE2_DIM), "resolved_se2")
    adjust_rows = requested == ADJUST_MODE
    if bool((np.abs(raw[~adjust_rows]) > 1.0e-7).any()):
        raise ReplayContractError("raw_residual must be zero outside requested ADJUST")
    if bool((np.abs(bounded[~adjust_rows]) > 1.0e-7).any()):
        raise ReplayContractError("bounded_residual must be zero outside requested ADJUST")
    expected_bounded = np.tanh(raw[adjust_rows].astype(np.float64)) * RESIDUAL_LIMITS
    if not np.allclose(bounded[adjust_rows], expected_bounded, rtol=1.0e-5, atol=1.0e-6):
        raise ReplayContractError("bounded_residual differs from tanh(raw)*limits")
    if bool((np.abs(bounded) > RESIDUAL_LIMITS[None, :] + 1.0e-7).any()):
        raise ReplayContractError("bounded_residual exceeds the executor contract")

    stop_rows = executed == STOP_MODE
    if bool((np.abs(resolved[stop_rows]) > 1.0e-7).any()):
        raise ReplayContractError("executed STOP must resolve to zero SE(2)")
    selected = _selected_reference(arrays)
    executable_reference = selected.astype(np.float64).copy()
    executable_reference[:, 0] = np.clip(
        executable_reference[:, 0], *EXECUTOR_FORWARD_RANGE_M
    )
    executable_reference[:, 2] = np.clip(
        executable_reference[:, 2], *EXECUTOR_YAW_RANGE_RAD
    )
    no_override = ~safety_override
    keep_rows = no_override & (executed == KEEP_MODE)
    if not np.allclose(
        resolved[keep_rows], executable_reference[keep_rows], rtol=1.0e-6, atol=1.0e-6
    ):
        raise ReplayContractError(
            "KEEP resolved SE(2) differs from the executor-bounded native reference"
        )
    adjust_executed = no_override & (executed == ADJUST_MODE)
    if bool(adjust_executed.any()):
        expected = executable_reference[adjust_executed].copy()
        expected[:, 0] = np.clip(
            expected[:, 0] + bounded[adjust_executed, 0], *EXECUTOR_FORWARD_RANGE_M
        )
        expected[:, 2] = np.clip(
            expected[:, 2] + bounded[adjust_executed, 1], *EXECUTOR_YAW_RANGE_RAD
        )
        if not np.allclose(resolved[adjust_executed], expected, rtol=1.0e-5, atol=1.0e-6):
            raise ReplayContractError("ADJUST resolved SE(2) differs from its bounded residual")

    recover_executed = no_override & (executed == RECOVER_MODE)
    recover_indices = np.flatnonzero(recover_executed)
    if recover_indices.size:
        parent = recovery_parent[recover_indices].astype(np.int64)
        parent_xy = arrays["topology_node_targets"][recover_indices, parent, :2]
        if not np.allclose(
            resolved[recover_indices, :2], parent_xy, rtol=1.0e-5, atol=1.0e-6
        ):
            raise ReplayContractError("RECOVER resolved waypoint differs from its causal parent")


def _validate_execution(
    arrays: Mapping[str, np.ndarray], metadata: Mapping[str, object], rows: int
) -> None:
    primitive_steps = int(metadata["max_primitive_steps"])
    rewards = arrays["primitive_rewards"]
    executed_mask = arrays["executed_mask"]
    duration = arrays["duration"]
    discounted = arrays["discounted_reward"]
    _floating(rewards, (rows, primitive_steps), "primitive_rewards")
    _bool_array(executed_mask, (rows, primitive_steps), "executed_mask")
    _integer_vector(duration, rows, "duration")
    if bool(((duration < 1) | (duration > primitive_steps)).any()):
        raise ReplayContractError("duration is outside the primitive horizon")
    expected_mask = (
        np.arange(primitive_steps, dtype=np.int64)[None, :]
        < duration.astype(np.int64)[:, None]
    )
    if not np.array_equal(executed_mask, expected_mask):
        raise ReplayContractError("executed_mask must be a prefix matching duration")
    _masked_padding_is_zero(rewards, executed_mask, "primitive_rewards")
    _floating(discounted, (rows,), "discounted_reward")
    gamma = float(metadata["discount_gamma"])
    weights = np.power(gamma, np.arange(primitive_steps, dtype=np.float64))[None, :]
    expected_reward = np.sum(rewards.astype(np.float64) * weights * executed_mask, axis=1)
    if not np.allclose(discounted, expected_reward, rtol=1.0e-5, atol=1.0e-6):
        raise ReplayContractError("discounted_reward differs from primitive rewards")

    for name in ("collision", "blocked"):
        _bool_array(arrays[name], (rows,), name)
    terminated = arrays["terminated"]
    truncated = arrays["truncated"]
    done = arrays["done"]
    for name, value in (
        ("terminated", terminated),
        ("truncated", truncated),
        ("done", done),
    ):
        _bool_array(value, (rows,), name)
    if bool((terminated & truncated).any()):
        raise ReplayContractError("a transition cannot both terminate and truncate")
    if not np.array_equal(done, terminated | truncated):
        raise ReplayContractError("done must equal terminated OR truncated")
    done_positions = np.flatnonzero(done)
    if not np.array_equal(done_positions, np.asarray([rows - 1], dtype=np.int64)):
        raise ReplayContractError("a complete episode shard must end exactly on its last row")

    end_reason = arrays["end_reason"]
    _integer_vector(end_reason, rows, "end_reason")
    if bool(((end_reason < 0) | (end_reason >= len(END_REASON_NAMES))).any()):
        raise ReplayContractError("end_reason is outside the fixed vocabulary")
    if bool((end_reason[~done] != END_REASON_RUNNING).any()):
        raise ReplayContractError("non-final rows must have end_reason=running")
    if bool((end_reason[done] == END_REASON_RUNNING).any()):
        raise ReplayContractError("the final row must record a non-running end reason")
    if bool(((arrays["executed_mode"] == STOP_MODE) & ~done).any()):
        raise ReplayContractError("an executed STOP must end the episode")

    decision_index = arrays["decision_index"]
    physical_step = arrays["physical_step"]
    behavior_version = arrays["behavior_policy_version"]
    for name, value in (
        ("decision_index", decision_index),
        ("physical_step", physical_step),
        ("behavior_policy_version", behavior_version),
    ):
        _integer_vector(value, rows, name)
    if not np.array_equal(decision_index.astype(np.int64), np.arange(rows, dtype=np.int64)):
        raise ReplayContractError("decision_index must be contiguous from zero")
    expected_steps = np.concatenate(
        (np.zeros((1,), dtype=np.int64), np.cumsum(duration[:-1], dtype=np.int64))
    )
    if not np.array_equal(physical_step.astype(np.int64), expected_steps):
        raise ReplayContractError("physical_step differs from cumulative duration")
    if bool((behavior_version < 0).any()):
        raise ReplayContractError("behavior_policy_version must be non-negative")
    unique_versions = np.unique(behavior_version)
    if unique_versions.size != 1:
        raise ReplayContractError("one episode shard cannot mix behavior versions")
    if int(unique_versions[0]) != int(metadata["behavior_policy_version"]):
        raise ReplayContractError("array and metadata behavior versions differ")


def validate_arrays(
    arrays: Mapping[str, np.ndarray], metadata: Mapping[str, object]
) -> dict[str, np.ndarray]:
    normalized_metadata = validate_metadata(metadata)
    expected_names = set(required_arrays(normalized_metadata))
    if set(arrays) != expected_names:
        missing = sorted(expected_names - set(arrays))
        extra = sorted(set(arrays) - expected_names)
        raise ReplayContractError(
            f"replay array members differ; missing={missing}, extra={extra}"
        )
    result = {name: np.asarray(value) for name, value in arrays.items()}
    requested = result["requested_mode"]
    rows = int(requested.shape[0]) if requested.ndim == 1 else -1
    if rows < 1:
        raise ReplayContractError("replay must contain at least one transition")
    normalized_metadata = validate_metadata(normalized_metadata, rows=rows)
    _validate_state(result, normalized_metadata, prefix="", rows=rows)
    _validate_state(result, normalized_metadata, prefix="next_", rows=rows)
    _validate_actions(result, normalized_metadata, rows)
    _validate_execution(result, normalized_metadata, rows)
    return result


def _metadata_bytes(metadata: Mapping[str, object]) -> np.ndarray:
    return np.frombuffer(_canonical_json(metadata, "metadata"), dtype=np.uint8).copy()


def _atomic_replace_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".staging-", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise FileExistsError(path)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def save_episode_shard(
    path: Path, arrays: Mapping[str, np.ndarray], metadata: Mapping[str, object]
) -> dict[str, object]:
    """Validate and atomically save one immutable compressed episode shard."""

    path = Path(path)
    if path.suffix.lower() != ".npz":
        raise ReplayContractError("episode replay path must end in .npz")
    if path.exists():
        raise FileExistsError(path)
    normalized_metadata = validate_metadata(metadata)
    normalized_arrays = validate_arrays(arrays, normalized_metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".staging-", dir=str(path.parent)
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                metadata_json=_metadata_bytes(normalized_metadata),
                **normalized_arrays,
            )
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise FileExistsError(path)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "archive_schema_name": SCHEMA_NAME,
        "archive_schema_version": SCHEMA_VERSION,
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "run_id": normalized_metadata["run_id"],
        "episode_id": normalized_metadata["episode_id"],
        "reset_token": normalized_metadata["reset_token"],
        "row_count": normalized_metadata["row_count"],
        "source_fingerprint": normalized_metadata["source_fingerprint"],
        "representation_kind": normalized_metadata["representation_kind"],
        "feature_stage": normalized_metadata["feature_stage"],
        "behavior_policy_id": normalized_metadata["behavior_policy_id"],
        "behavior_policy_version": normalized_metadata["behavior_policy_version"],
    }


def load_episode_shard(
    path: Path,
    *,
    expected_sha256: str | None = None,
    expected_source_fingerprint: str | None = None,
    expected_representation_kind: str | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    path = Path(path).resolve(strict=True)
    actual_sha = sha256_file(path)
    if expected_sha256 is not None:
        _validate_sha256(expected_sha256, "expected replay SHA-256")
        if actual_sha != expected_sha256:
            raise ReplayContractError("episode replay SHA-256 differs")
    with np.load(path, allow_pickle=False) as archive:
        members = set(archive.files)
        if "metadata_json" not in members:
            raise ReplayContractError("episode replay has no metadata_json")
        raw_metadata = np.asarray(archive["metadata_json"], dtype=np.uint8)
        if raw_metadata.ndim != 1:
            raise ReplayContractError("metadata_json must be a byte vector")
        try:
            metadata_value = json.loads(raw_metadata.tobytes().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ReplayContractError(f"metadata_json is invalid: {exc}") from exc
        metadata = validate_metadata(metadata_value)
        expected_members = set(required_arrays(metadata)) | {"metadata_json"}
        if members != expected_members:
            missing = sorted(expected_members - members)
            extra = sorted(members - expected_members)
            raise ReplayContractError(
                f"episode archive members differ; missing={missing}, extra={extra}"
            )
        arrays = {
            name: np.asarray(archive[name]).copy() for name in required_arrays(metadata)
        }
    normalized_arrays = validate_arrays(arrays, metadata)
    if expected_source_fingerprint is not None:
        _validate_sha256(expected_source_fingerprint, "expected source fingerprint")
        if metadata["source_fingerprint"] != expected_source_fingerprint:
            raise ReplayContractError("episode replay source identity differs")
    if (
        expected_representation_kind is not None
        and metadata["representation_kind"] != expected_representation_kind
    ):
        raise ReplayContractError("episode replay representation differs")
    return normalized_arrays, metadata


_SHARD_RECEIPT_FIELDS = {
    "archive_schema_name",
    "archive_schema_version",
    "path",
    "sha256",
    "size_bytes",
    "run_id",
    "episode_id",
    "reset_token",
    "row_count",
    "source_fingerprint",
    "representation_kind",
    "feature_stage",
    "behavior_policy_id",
    "behavior_policy_version",
}


def _validate_shard_receipt(value: object) -> dict[str, object]:
    receipt = _json_mapping(value, "shard receipt")
    if set(receipt) != _SHARD_RECEIPT_FIELDS:
        missing = sorted(_SHARD_RECEIPT_FIELDS - set(receipt))
        extra = sorted(set(receipt) - _SHARD_RECEIPT_FIELDS)
        raise ReplayContractError(
            f"shard receipt members differ; missing={missing}, extra={extra}"
        )
    if receipt.get("archive_schema_name") != SCHEMA_NAME:
        raise ReplayContractError("shard receipt schema name differs")
    if receipt.get("archive_schema_version") != SCHEMA_VERSION:
        raise ReplayContractError("shard receipt schema version differs")
    for name in (
        "path",
        "run_id",
        "episode_id",
        "reset_token",
        "representation_kind",
        "feature_stage",
        "behavior_policy_id",
    ):
        item = receipt.get(name)
        if not isinstance(item, str) or not item:
            raise ReplayContractError(f"shard receipt {name} must be non-empty")
    _validate_sha256(receipt["sha256"], "shard receipt SHA-256")
    _validate_sha256(receipt["source_fingerprint"], "shard source fingerprint")
    _positive_integer(receipt["size_bytes"], "shard receipt size_bytes")
    _positive_integer(receipt["row_count"], "shard receipt row_count")
    _nonnegative_integer(
        receipt["behavior_policy_version"], "shard behavior_policy_version"
    )
    representation = receipt["representation_kind"]
    if representation not in REPRESENTATION_STAGES:
        raise ReplayContractError("shard receipt representation is unknown")
    if receipt["feature_stage"] != REPRESENTATION_STAGES[representation]:
        raise ReplayContractError("shard receipt feature stage differs")
    return receipt


def build_replay_manifest(
    shard_receipts: Sequence[Mapping[str, object]],
    *,
    run_id: str,
    expected_episode_count: int,
    expected_source_fingerprint: str | None = None,
) -> dict[str, object]:
    """Build a duplicate-free manifest while allowing cross-shard versions."""

    if not isinstance(run_id, str) or not run_id:
        raise ReplayContractError("run_id must be non-empty")
    _positive_integer(expected_episode_count, "expected_episode_count")
    if len(shard_receipts) != expected_episode_count:
        raise ReplayContractError("replay shard count differs from the fixed budget")
    rows = [_validate_shard_receipt(receipt) for receipt in shard_receipts]
    if not rows:
        raise ReplayContractError("manifest requires at least one shard")
    first = rows[0]
    source = str(first["source_fingerprint"])
    if expected_source_fingerprint is not None:
        _validate_sha256(expected_source_fingerprint, "expected source fingerprint")
        if source != expected_source_fingerprint:
            raise ReplayContractError("manifest source fingerprint differs")
    representation = str(first["representation_kind"])
    feature_stage = str(first["feature_stage"])
    behavior_policy_id = str(first["behavior_policy_id"])
    identities: set[tuple[str, str]] = set()
    total_rows = 0
    versions: set[int] = set()
    for row in rows:
        identity = (str(row["episode_id"]), str(row["reset_token"]))
        if identity in identities:
            raise ReplayContractError("replay contains a duplicate episode identity")
        identities.add(identity)
        if row["run_id"] != run_id:
            raise ReplayContractError("shard receipt run_id differs")
        if row["source_fingerprint"] != source:
            raise ReplayContractError("replay shards mix immutable source identities")
        if row["representation_kind"] != representation:
            raise ReplayContractError("replay shards mix representation kinds")
        if row["feature_stage"] != feature_stage:
            raise ReplayContractError("replay shards mix feature stages")
        if row["behavior_policy_id"] != behavior_policy_id:
            raise ReplayContractError("replay shards mix behavior policy identities")
        total_rows += int(row["row_count"])
        versions.add(int(row["behavior_policy_version"]))
    rows.sort(key=lambda row: (str(row["episode_id"]), str(row["reset_token"])))
    manifest = {
        "schema_name": MANIFEST_SCHEMA_NAME,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "split_label": "train_update",
        "validation_split_reads": 0,
        "expected_episode_count": expected_episode_count,
        "completed_episode_count": len(rows),
        "transition_count": total_rows,
        "source_fingerprint": source,
        "representation_kind": representation,
        "feature_stage": feature_stage,
        "behavior_policy_id": behavior_policy_id,
        "behavior_policy_versions": sorted(versions),
        "shards": rows,
    }
    return validate_replay_manifest(manifest)


_MANIFEST_FIELDS = {
    "schema_name",
    "schema_version",
    "run_id",
    "split_label",
    "validation_split_reads",
    "expected_episode_count",
    "completed_episode_count",
    "transition_count",
    "source_fingerprint",
    "representation_kind",
    "feature_stage",
    "behavior_policy_id",
    "behavior_policy_versions",
    "shards",
}


def validate_replay_manifest(manifest: Mapping[str, object]) -> dict[str, object]:
    result = _json_mapping(manifest, "replay manifest")
    if set(result) != _MANIFEST_FIELDS:
        missing = sorted(_MANIFEST_FIELDS - set(result))
        extra = sorted(set(result) - _MANIFEST_FIELDS)
        raise ReplayContractError(
            f"manifest members differ; missing={missing}, extra={extra}"
        )
    if result.get("schema_name") != MANIFEST_SCHEMA_NAME:
        raise ReplayContractError("manifest schema name differs")
    if result.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ReplayContractError("manifest schema version differs")
    for name in ("run_id", "representation_kind", "feature_stage", "behavior_policy_id"):
        item = result.get(name)
        if not isinstance(item, str) or not item:
            raise ReplayContractError(f"manifest.{name} must be non-empty")
    if result.get("split_label") != "train_update":
        raise ReplayContractError("manifest split label differs")
    if result.get("validation_split_reads") != 0:
        raise ReplayContractError("manifest records validation split access")
    _validate_sha256(result["source_fingerprint"], "manifest source fingerprint")
    representation = result["representation_kind"]
    if representation not in REPRESENTATION_STAGES:
        raise ReplayContractError("manifest representation is unknown")
    if result["feature_stage"] != REPRESENTATION_STAGES[representation]:
        raise ReplayContractError("manifest feature stage differs")
    expected = _positive_integer(
        result["expected_episode_count"], "manifest.expected_episode_count"
    )
    completed = _positive_integer(
        result["completed_episode_count"], "manifest.completed_episode_count"
    )
    transitions = _positive_integer(
        result["transition_count"], "manifest.transition_count"
    )
    if expected != completed:
        raise ReplayContractError("manifest is not a complete fixed-budget replay")
    raw_versions = result.get("behavior_policy_versions")
    if not isinstance(raw_versions, list) or not raw_versions:
        raise ReplayContractError("manifest behavior_policy_versions must be non-empty")
    versions = [
        _nonnegative_integer(value, "manifest behavior policy version")
        for value in raw_versions
    ]
    if versions != sorted(set(versions)):
        raise ReplayContractError("manifest behavior policy versions are not canonical")

    raw_shards = result.get("shards")
    if not isinstance(raw_shards, list) or len(raw_shards) != completed:
        raise ReplayContractError("manifest shard inventory is incomplete")
    shards = [_validate_shard_receipt(value) for value in raw_shards]
    identities: set[tuple[str, str]] = set()
    observed_versions: set[int] = set()
    total_rows = 0
    for shard in shards:
        identity = (str(shard["episode_id"]), str(shard["reset_token"]))
        if identity in identities:
            raise ReplayContractError("manifest contains a duplicate episode identity")
        identities.add(identity)
        if shard["run_id"] != result["run_id"]:
            raise ReplayContractError("manifest shard run_id differs")
        if shard["source_fingerprint"] != result["source_fingerprint"]:
            raise ReplayContractError("manifest shard source identity differs")
        if shard["representation_kind"] != representation:
            raise ReplayContractError("manifest shard representation differs")
        if shard["feature_stage"] != result["feature_stage"]:
            raise ReplayContractError("manifest shard feature stage differs")
        if shard["behavior_policy_id"] != result["behavior_policy_id"]:
            raise ReplayContractError("manifest shard behavior policy differs")
        total_rows += int(shard["row_count"])
        observed_versions.add(int(shard["behavior_policy_version"]))
    canonical_order = sorted(
        shards, key=lambda row: (str(row["episode_id"]), str(row["reset_token"]))
    )
    if shards != canonical_order:
        raise ReplayContractError("manifest shard inventory is not canonically sorted")
    if total_rows != transitions:
        raise ReplayContractError("manifest transition count differs from its shards")
    if sorted(observed_versions) != versions:
        raise ReplayContractError("manifest behavior version inventory differs")
    result["shards"] = shards
    return result


def save_replay_manifest(path: Path, manifest: Mapping[str, object]) -> dict[str, object]:
    """Validate and atomically save the canonical JSON shard inventory."""

    path = Path(path)
    normalized = validate_replay_manifest(manifest)
    _atomic_replace_bytes(path, _canonical_json(normalized, "replay manifest") + b"\n")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "run_id": normalized["run_id"],
        "episode_count": normalized["completed_episode_count"],
        "transition_count": normalized["transition_count"],
        "source_fingerprint": normalized["source_fingerprint"],
    }


def load_replay_manifest(
    path: Path,
    *,
    expected_sha256: str | None = None,
    expected_source_fingerprint: str | None = None,
    verify_shards: bool = False,
) -> dict[str, object]:
    path = Path(path).resolve(strict=True)
    actual_sha = sha256_file(path)
    if expected_sha256 is not None:
        _validate_sha256(expected_sha256, "expected manifest SHA-256")
        if actual_sha != expected_sha256:
            raise ReplayContractError("replay manifest SHA-256 differs")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReplayContractError(f"replay manifest JSON is invalid: {exc}") from exc
    manifest = validate_replay_manifest(value)
    if expected_source_fingerprint is not None:
        _validate_sha256(expected_source_fingerprint, "expected source fingerprint")
        if manifest["source_fingerprint"] != expected_source_fingerprint:
            raise ReplayContractError("replay manifest source identity differs")
    if verify_shards:
        for receipt_value in manifest["shards"]:
            receipt = _validate_shard_receipt(receipt_value)
            _, metadata = load_episode_shard(
                Path(str(receipt["path"])),
                expected_sha256=str(receipt["sha256"]),
                expected_source_fingerprint=str(manifest["source_fingerprint"]),
                expected_representation_kind=str(manifest["representation_kind"]),
            )
            for name in (
                "run_id",
                "episode_id",
                "reset_token",
                "row_count",
                "behavior_policy_id",
                "behavior_policy_version",
            ):
                if metadata[name] != receipt[name]:
                    raise ReplayContractError(
                        f"manifest receipt {name} differs from its episode shard"
                    )
    return manifest


@dataclass(frozen=True)
class ReplayBatch:
    arrays: Mapping[str, np.ndarray]
    source_shards: tuple[str, ...]
    episode_offsets: tuple[int, ...]
    source_fingerprint: str
    representation_kind: str
    feature_stage: str
    behavior_policy_versions: tuple[int, ...]

    @property
    def rows(self) -> int:
        return int(self.arrays["requested_mode"].shape[0])


def concatenate_episode_shards(
    paths: Iterable[Path], *, expected_source_fingerprint: str
) -> ReplayBatch:
    """Load compatible episodes; behavior versions may differ between shards."""

    _validate_sha256(expected_source_fingerprint, "expected source fingerprint")
    loaded: list[dict[str, np.ndarray]] = []
    names: list[str] = []
    offsets = [0]
    feature_shapes: dict[str, tuple[int, ...]] | None = None
    expected_array_names: tuple[str, ...] | None = None
    seen: set[tuple[str, str]] = set()
    representation: str | None = None
    feature_stage: str | None = None
    behavior_policy_id: str | None = None
    versions: set[int] = set()
    for raw_path in paths:
        path = Path(raw_path)
        arrays, metadata = load_episode_shard(
            path, expected_source_fingerprint=expected_source_fingerprint
        )
        identity = (str(metadata["episode_id"]), str(metadata["reset_token"]))
        if identity in seen:
            raise ReplayContractError("duplicate episode identity during concatenation")
        seen.add(identity)
        if representation is None:
            representation = str(metadata["representation_kind"])
            feature_stage = str(metadata["feature_stage"])
            behavior_policy_id = str(metadata["behavior_policy_id"])
            expected_array_names = required_arrays(metadata)
        elif metadata["representation_kind"] != representation:
            raise ReplayContractError("episode replay representations differ")
        elif metadata["feature_stage"] != feature_stage:
            raise ReplayContractError("episode replay feature stages differ")
        elif metadata["behavior_policy_id"] != behavior_policy_id:
            raise ReplayContractError("episode replay behavior policy identities differ")
        shapes = {name: tuple(value.shape[1:]) for name, value in arrays.items()}
        if feature_shapes is None:
            feature_shapes = shapes
        elif shapes != feature_shapes:
            raise ReplayContractError("episode replay tensor shapes differ")
        loaded.append(arrays)
        names.append(str(path.resolve()))
        offsets.append(offsets[-1] + int(metadata["row_count"]))
        versions.add(int(metadata["behavior_policy_version"]))
    if not loaded or expected_array_names is None or representation is None or feature_stage is None:
        raise ReplayContractError("no replay shards were provided")
    concatenated = {
        name: np.concatenate([episode[name] for episode in loaded], axis=0)
        for name in expected_array_names
    }
    return ReplayBatch(
        arrays=concatenated,
        source_shards=tuple(names),
        episode_offsets=tuple(offsets),
        source_fingerprint=expected_source_fingerprint,
        representation_kind=representation,
        feature_stage=feature_stage,
        behavior_policy_versions=tuple(sorted(versions)),
    )


__all__ = [
    "ADJUST_MODE",
    "COMMON_ARRAYS",
    "END_REASON_NAMES",
    "END_REASON_RUNNING",
    "FEATURE_STAGE_FROZEN_Z_RL",
    "FEATURE_STAGE_PREFIX_HIDDEN",
    "KEEP_MODE",
    "LIGHTNAV_SOURCE_REVISION",
    "MANIFEST_SCHEMA_NAME",
    "MANIFEST_SCHEMA_VERSION",
    "MODE_COUNT",
    "MODE_NAMES",
    "PREFIX_REPRESENTATION_ARRAYS",
    "RECOVER_MODE",
    "REFERENCE_DIM",
    "REFERENCE_SCHEMA",
    "REFERENCE_TRAJECTORY_SHAPE",
    "REPRESENTATION_FROZEN_Z_RL",
    "REPRESENTATION_PREFIX_HIDDEN",
    "RESIDUAL_DIM",
    "RESIDUAL_LIMITS",
    "ReplayBatch",
    "ReplayContractError",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "STATE_ARRAYS",
    "STOP_MODE",
    "TRANSITION_ARRAYS",
    "Z_REPRESENTATION_ARRAYS",
    "build_replay_manifest",
    "concatenate_episode_shards",
    "load_episode_shard",
    "load_replay_manifest",
    "required_arrays",
    "save_episode_shard",
    "save_replay_manifest",
    "sha256_file",
    "source_fingerprint",
    "validate_arrays",
    "validate_metadata",
    "validate_replay_manifest",
]
