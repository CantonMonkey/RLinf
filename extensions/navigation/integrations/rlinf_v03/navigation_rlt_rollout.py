#!/usr/bin/env python3
"""Collect immutable train-update Navigation-RLT episode replay shards.

Each worker owns one Habitat environment, one patched frozen-LightNav session,
and one causal topology.  The worker consumes a preassigned parallel plan,
requests the opt-in ``contextual_prefix_hidden_v1`` artifact, verifies every
artifact before use, and writes one replay shard per complete episode through
``navigation_rlt_replay.save_episode_shard``.

The artifact boundary is deliberately strict: every attention-valid token is
authenticated in original order and an artifact over its configured capture
cap makes the worker invalid.  The authenticated sequence is then compressed
once through the policy's public deterministic selector and right-padded to the
fixed replay width.  Optional artifact consumption is allowed only after both
the replay shard and its feature evidence are durable.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import stat
import time
from typing import Any, Mapping, Sequence

import numpy as np

from lightnav_a3_policy import (
    CausalTopologyState,
    NumpyLightNavPolicy,
    build_features as build_a3_features,
    sha256_file,
)
from lightnav_a3_rollout import (
    _bool_success,
    _collision_event,
    _episode_metrics,
    _physical_velocity,
    _normalized_command,
    _reward,
    _scene_name,
    _validate_velocity_contract,
    aggregate as aggregate_a3,
)
from lightnav_l0_native_probe import (
    HabitatProtocolError,
    LightNavPrediction,
    LightNavProtocolError,
    LightNavWebSocketClient,
    VelocitySpec,
    ZmqHabitatClient,
    _collision_observation,
    _mapping,
    _metric,
    _optional_metric,
    atomic_write_json,
    classify_velocity_action,
    encode_rgb_jpeg_base64,
    extract_instruction,
    login_request,
    next_request,
    parse_lightnav_response,
    redact_endpoint,
    waypoint_to_velocity_action,
)
from navigation_rlt_action import (
    ADJUST,
    CANONICAL_ACTION_DIM,
    KEEP,
    MAX_RECOVERY_PRIMITIVES,
    MODE_NAMES,
    RECOVER,
    STOP,
    CompiledAction,
    clip_reference_waypoint,
    compile_action,
    valid_mode_mask,
)
from navigation_rlt_parallel import (
    PLAN_SCHEMA,
    PLAN_VERSION,
    WORKER_SCHEMA,
    WORKER_VERSION,
    atomic_json,
)
from navigation_rlt_replay import (
    END_REASON_NAMES,
    FEATURE_STAGE_PREFIX_HIDDEN,
    LIGHTNAV_SOURCE_REVISION,
    REFERENCE_DIM as REPLAY_REFERENCE_DIM,
    REFERENCE_SCHEMA as REPLAY_REFERENCE_SCHEMA,
    REPRESENTATION_PREFIX_HIDDEN,
    SCHEMA_NAME as REPLAY_SCHEMA_NAME,
    SCHEMA_VERSION as REPLAY_SCHEMA_VERSION,
    save_episode_shard,
)
from streamvln_topology import (
    CAUSAL_GLOBAL_FEATURE_NAMES,
    CAUSAL_NODE_FEATURE_DIM,
    OnlineTopology,
    causal_topology_payload,
)


SCHEMA_NAME = "navigation_rlt_rollout_worker"
SCHEMA_VERSION = 1
CONTEXT_REQUEST = "contextual_prefix_hidden_v1"
CONTEXT_ARTIFACT_SCHEMA = "lightnav.contextual_prefix_hidden.v1"
ARTIFACT_PREFIX_SCHEMA = (
    "lightnav.contextual_prefix_hidden.v1/all_attention_valid_original_order_float16"
)
POLICY_PREFIX_SELECTION_SCHEMA = "keep_endpoints_uniform_middle_v1"
PREFIX_FEATURE_SCHEMA = (
    f"{ARTIFACT_PREFIX_SCHEMA}/{POLICY_PREFIX_SELECTION_SCHEMA}/right_padded"
)
ARTIFACT_CAPTURE_RULE = {
    "selection": "all_attention_valid_tokens",
    "order": "original_prefix_order",
    "pooling": "none",
    "artifact_truncation": "forbidden_fail_fast",
    "artifact_storage_dtype": "float16",
}
POLICY_PREFIX_SELECTION_RULE = {
    "schema": POLICY_PREFIX_SELECTION_SCHEMA,
    "when_original_le_policy_cap": "keep_all_valid_tokens",
    "when_original_gt_policy_cap": (
        "keep first/last and use floor(j*(L-1)/(K-1)) valid-token ordinals"
    ),
    "order": "source_order",
    "replay_layout": "right_padded_to_policy_cap",
}
TOPOLOGY_SCHEMA = "causal_online_topology_nodes8_global14_v1"
ACTION_SCHEMA = "navigation_rlt_keep_adjust_stop_recover_canonical7_v1"
EXECUTOR_SCHEMA = "lightnav_habitat_velocity_se2_prefix_v1"
REFERENCE_SCHEMA = REPLAY_REFERENCE_SCHEMA
REFERENCE_DIM = REPLAY_REFERENCE_DIM
BEHAVIOR_POLICY_ID = "navigation_rlt_warmstart_lineage_v1"
ARTIFACT_CLEANUP_SCHEMA = "navigation_rlt_feature_artifact_cleanup"
ARTIFACT_CLEANUP_VERSION = 1

BEHAVIOR_NATIVE = "native"
BEHAVIOR_A3 = "a3"
BEHAVIOR_RLT = "navigation-rlt"
BEHAVIOR_NAMES = (BEHAVIOR_NATIVE, BEHAVIOR_A3, BEHAVIOR_RLT)


class NavigationRLTRolloutError(RuntimeError):
    """Raised when a worker cannot prove a complete train-update replay."""


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise NavigationRLTRolloutError(f"value is not canonical JSON: {exc}") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validate_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise NavigationRLTRolloutError(f"{name} must be a lowercase SHA-256")
    return value


def _array_digest(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    header = _canonical_json(
        {"dtype": contiguous.dtype.str, "shape": list(contiguous.shape)}
    )
    return _sha256_bytes(header + b"\0" + contiguous.tobytes(order="C"))


def _true_ranges(mask: np.ndarray) -> list[list[int]]:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []
    splits = np.flatnonzero(np.diff(indices) != 1) + 1
    return [
        [int(group[0]), int(group[-1] - group[0] + 1)]
        for group in np.split(indices, splits)
    ]


def _path_entry_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _regular_file_no_reparse(path: Path, name: str) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        raise NavigationRLTRolloutError(f"{name} is missing")
    attributes = getattr(info, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if stat.S_ISLNK(info.st_mode) or attributes & reparse_flag or not stat.S_ISREG(
        info.st_mode
    ):
        raise NavigationRLTRolloutError(
            f"{name} must be a regular non-link, non-reparse file"
        )


def _directory_no_reparse(path: Path, name: str) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        raise NavigationRLTRolloutError(f"{name} is missing")
    attributes = getattr(info, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if stat.S_ISLNK(info.st_mode) or attributes & reparse_flag or not stat.S_ISDIR(
        info.st_mode
    ):
        raise NavigationRLTRolloutError(
            f"{name} must be a non-link, non-reparse directory"
        )


def _resolve_artifact_root(path: Path) -> Path:
    raw = Path(path).expanduser()
    _directory_no_reparse(raw, "feature artifact root")
    resolved = raw.resolve(strict=True)
    _directory_no_reparse(resolved, "resolved feature artifact root")
    return resolved


def _strict_child(root: Path, relative: object, name: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise NavigationRLTRolloutError(f"{name} path is missing")
    relative_path = Path(relative)
    if relative_path.is_absolute() or relative_path.name != relative:
        raise NavigationRLTRolloutError(f"{name} must be a direct artifact filename")
    root = root.resolve(strict=True)
    raw_candidate = root / relative
    _regular_file_no_reparse(raw_candidate, name)
    candidate = raw_candidate.resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise NavigationRLTRolloutError(f"{name} escapes the artifact directory") from exc
    _regular_file_no_reparse(candidate, name)
    return candidate


@dataclass(frozen=True)
class ArtifactFile:
    path: Path
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class ContextArtifact:
    hidden: np.ndarray
    identity: str
    manifest_sha256: str
    prompt_identity: str
    source_sha256: str
    valid_count: int
    hidden_dim: int
    manifest_path: Path
    manifest_size_bytes: int
    files: tuple[ArtifactFile, ...]
    content_root: Path | None

    def evidence(self) -> dict[str, object]:
        return {
            "identity": self.identity,
            "manifest_sha256": self.manifest_sha256,
            "prompt_identity": self.prompt_identity,
            "source_sha256": self.source_sha256,
            "valid_count": self.valid_count,
            "hidden_dim": self.hidden_dim,
            "artifact_capture_rule": ARTIFACT_CAPTURE_RULE,
        }


def load_context_artifact(
    feature_ref: Mapping[str, object],
    *,
    artifact_prefix_max_tokens: int,
    prefix_hidden_dim: int,
    expected_source_sha256: str | None,
    feature_artifact_root: Path | None,
) -> ContextArtifact:
    """Load and fully authenticate one LightNav contextual-prefix artifact."""

    ref = _mapping(feature_ref, "LightNav feature_ref")
    expected_ref_fields = {
        "schema",
        "request",
        "ref",
        "shape",
        "mask",
        "hash",
        "identity",
        "model_identity",
        "prompt_identity",
    }
    if set(ref) != expected_ref_fields:
        raise NavigationRLTRolloutError("feature_ref members differ from the patch contract")
    if ref.get("schema") != CONTEXT_ARTIFACT_SCHEMA or ref.get("request") != CONTEXT_REQUEST:
        raise NavigationRLTRolloutError("feature_ref schema/request differs")
    identity = _validate_sha256(ref.get("identity"), "feature_ref.identity")
    prompt_identity = _validate_sha256(
        ref.get("prompt_identity"), "feature_ref.prompt_identity"
    )
    reference = _mapping(ref.get("ref"), "feature_ref.ref")
    if set(reference) != {"kind", "path"} or reference.get("kind") != "local_manifest":
        raise NavigationRLTRolloutError("feature_ref must use a local_manifest reference")
    raw_manifest_path = reference.get("path")
    if not isinstance(raw_manifest_path, str) or not raw_manifest_path:
        raise NavigationRLTRolloutError("feature_ref manifest path is missing")
    raw_manifest = Path(raw_manifest_path).expanduser()
    _regular_file_no_reparse(raw_manifest, "feature manifest")
    manifest_path = raw_manifest.resolve(strict=True)
    _regular_file_no_reparse(manifest_path, "resolved feature manifest")
    content_root = None
    if feature_artifact_root is not None:
        content_root = _resolve_artifact_root(feature_artifact_root)
        expected_manifest_path = (
            content_root / identity[:2] / identity / "manifest.json"
        )
        expected_prefix_dir = content_root / identity[:2]
        expected_identity_dir = expected_prefix_dir / identity
        _directory_no_reparse(expected_prefix_dir, "artifact prefix directory")
        _directory_no_reparse(expected_identity_dir, "artifact identity directory")
        expected_manifest_resolved = expected_manifest_path.resolve(strict=True)
        if manifest_path != expected_manifest_resolved:
            raise NavigationRLTRolloutError(
                "feature manifest is not at its exact content-addressed path"
            )
    hashes = _mapping(ref.get("hash"), "feature_ref.hash")
    if set(hashes) != {"algorithm", "manifest", "hidden"} or hashes.get("algorithm") != "sha256":
        raise NavigationRLTRolloutError("feature_ref hash contract differs")
    expected_manifest_sha = _validate_sha256(hashes.get("manifest"), "manifest hash")
    observed_manifest_sha = sha256_file(manifest_path)
    if observed_manifest_sha != expected_manifest_sha:
        raise NavigationRLTRolloutError("context manifest SHA-256 differs")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NavigationRLTRolloutError(f"context manifest is invalid JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise NavigationRLTRolloutError("context manifest must contain an object")
    expected_manifest_fields = {
        "schema",
        "request",
        "model_identity",
        "prompt_identity",
        "arrays",
        "identity",
        "source_dtype",
        "storage_dtype",
        "payload_bytes",
        "files",
    }
    if set(manifest) != expected_manifest_fields:
        raise NavigationRLTRolloutError("context manifest members differ")
    if (
        manifest.get("schema") != CONTEXT_ARTIFACT_SCHEMA
        or manifest.get("request") != CONTEXT_REQUEST
        or manifest.get("identity") != identity
        or manifest.get("prompt_identity") != prompt_identity
        or manifest.get("model_identity") != ref.get("model_identity")
    ):
        raise NavigationRLTRolloutError("context manifest identity differs from feature_ref")
    if manifest.get("storage_dtype") != "float16":
        raise NavigationRLTRolloutError("context artifact storage dtype is not float16")

    files = _mapping(manifest.get("files"), "context manifest files")
    arrays_claim = _mapping(manifest.get("arrays"), "context manifest arrays")
    required_names = {"hidden", "attention_mask", "valid_token_indices", "input_ids"}
    allowed_names = required_names | {"video_grid_thw"}
    if not required_names.issubset(files) or set(files) - allowed_names:
        raise NavigationRLTRolloutError("context manifest file inventory differs")
    if set(arrays_claim) != set(files):
        raise NavigationRLTRolloutError("context manifest array/file inventories differ")

    arrays: dict[str, np.ndarray] = {}
    artifact_files: list[ArtifactFile] = []
    payload_bytes = 0
    for name in sorted(files):
        file_row = _mapping(files[name], f"context file {name}")
        if set(file_row) != {"path", "dtype", "shape", "sha256"}:
            raise NavigationRLTRolloutError(f"context file {name} metadata differs")
        path = _strict_child(manifest_path.parent, file_row["path"], name)
        expected_file_sha = _validate_sha256(file_row.get("sha256"), f"{name} file SHA")
        if sha256_file(path) != expected_file_sha:
            raise NavigationRLTRolloutError(f"context file {name} SHA-256 differs")
        artifact_files.append(
            ArtifactFile(
                path=path,
                sha256=expected_file_sha,
                size_bytes=int(path.stat().st_size),
            )
        )
        try:
            array = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise NavigationRLTRolloutError(f"cannot load context array {name}: {exc}") from exc
        if array.dtype.str != file_row.get("dtype") or list(array.shape) != file_row.get("shape"):
            raise NavigationRLTRolloutError(f"context array {name} dtype/shape differs")
        digest = _array_digest(array)
        if digest != _validate_sha256(arrays_claim.get(name), f"{name} array digest"):
            raise NavigationRLTRolloutError(f"context array {name} content digest differs")
        arrays[name] = np.asarray(array)
        payload_bytes += int(array.nbytes)
    if manifest.get("payload_bytes") != payload_bytes:
        raise NavigationRLTRolloutError("context manifest payload_bytes differs")

    hidden = arrays["hidden"]
    attention = arrays["attention_mask"]
    indices = arrays["valid_token_indices"]
    input_ids = arrays["input_ids"]
    if hidden.dtype != np.dtype("<f2") or hidden.ndim != 2 or not np.isfinite(hidden).all():
        raise NavigationRLTRolloutError("context hidden must be finite little-endian float16 [T,H]")
    if attention.ndim != 1 or attention.dtype != np.uint8 or not np.isin(attention, (0, 1)).all():
        raise NavigationRLTRolloutError("context attention_mask must be uint8 binary [prefix]")
    if input_ids.ndim != 1 or input_ids.shape != attention.shape or not np.issubdtype(
        input_ids.dtype, np.signedinteger
    ):
        raise NavigationRLTRolloutError("context input_ids/attention_mask shapes differ")
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.signedinteger):
        raise NavigationRLTRolloutError("context valid_token_indices must be an integer vector")
    expected_indices = np.flatnonzero(attention).astype(indices.dtype, copy=False)
    if not np.array_equal(indices, expected_indices):
        raise NavigationRLTRolloutError("context valid_token_indices differ from attention_mask")
    valid_count = int(attention.sum(dtype=np.int64))
    if hidden.shape != (valid_count, prefix_hidden_dim):
        raise NavigationRLTRolloutError(
            "context hidden shape differs from valid mask count/configured hidden width"
        )
    if valid_count > artifact_prefix_max_tokens:
        raise NavigationRLTRolloutError(
            f"context has {valid_count} valid tokens over "
            f"artifact_prefix_max_tokens={artifact_prefix_max_tokens}"
        )

    shape_claim = ref.get("shape")
    mask_claim = _mapping(ref.get("mask"), "feature_ref.mask")
    if shape_claim != list(hidden.shape):
        raise NavigationRLTRolloutError("feature_ref hidden shape differs")
    if set(mask_claim) != {
        "encoding",
        "shape",
        "valid_count",
        "valid_ranges",
        "sha256",
    }:
        raise NavigationRLTRolloutError("feature_ref mask members differ")
    if (
        mask_claim.get("encoding") != "valid_ranges_start_length"
        or mask_claim.get("shape") != list(attention.shape)
        or mask_claim.get("valid_count") != valid_count
        or mask_claim.get("valid_ranges") != _true_ranges(attention)
        or mask_claim.get("sha256") != _array_digest(attention)
    ):
        raise NavigationRLTRolloutError("feature_ref mask identity differs")
    if hashes.get("hidden") != _array_digest(hidden):
        raise NavigationRLTRolloutError("feature_ref hidden digest differs")

    arrays_identity = {name: _array_digest(arrays[name]) for name in arrays}
    expected_prompt_identity = _sha256_bytes(
        _canonical_json(
            {
                "input_ids": arrays_identity["input_ids"],
                "attention_mask": arrays_identity["attention_mask"],
                "video_grid_thw": arrays_identity.get("video_grid_thw"),
            }
        )
    )
    identity_basis = {
        "schema": CONTEXT_ARTIFACT_SCHEMA,
        "request": CONTEXT_REQUEST,
        "model_identity": manifest["model_identity"],
        "prompt_identity": expected_prompt_identity,
        "arrays": arrays_identity,
    }
    if expected_prompt_identity != prompt_identity or _sha256_bytes(
        _canonical_json(identity_basis)
    ) != identity:
        raise NavigationRLTRolloutError("context content-addressed identity differs")

    source_descriptor = {
        "schema": CONTEXT_ARTIFACT_SCHEMA,
        "request": CONTEXT_REQUEST,
        "model_identity": manifest["model_identity"],
        "source_dtype": manifest["source_dtype"],
        "storage_dtype": manifest["storage_dtype"],
        "hidden_dim": prefix_hidden_dim,
        "artifact_capture_rule": ARTIFACT_CAPTURE_RULE,
        "lightnav_source_revision": LIGHTNAV_SOURCE_REVISION,
    }
    source_sha = _sha256_bytes(_canonical_json(source_descriptor))
    if expected_source_sha256 is not None and source_sha != expected_source_sha256:
        raise NavigationRLTRolloutError("context feature source identity differs")
    return ContextArtifact(
        hidden=np.ascontiguousarray(hidden),
        identity=identity,
        manifest_sha256=observed_manifest_sha,
        prompt_identity=prompt_identity,
        source_sha256=source_sha,
        valid_count=valid_count,
        hidden_dim=int(hidden.shape[1]),
        manifest_path=manifest_path,
        manifest_size_bytes=int(manifest_path.stat().st_size),
        files=tuple(artifact_files),
        content_root=content_root,
    )


@dataclass(frozen=True)
class ContextualPrediction:
    prediction: LightNavPrediction
    artifact: ContextArtifact


class ContextLightNavClient(LightNavWebSocketClient):
    """Official synchronous client plus the explicit feature-request extension."""

    def login_for_context(self, client_id: str) -> None:
        response = self._rpc(login_request(client_id))
        self._ack(response, "login")
        data = _mapping(response.get("data"), "LightNav login data")
        capabilities = _mapping(data.get("capabilities"), "LightNav capabilities")
        requests = capabilities.get("feature_request")
        if not isinstance(requests, list) or CONTEXT_REQUEST not in requests:
            raise LightNavProtocolError(
                f"LightNav did not advertise required feature request {CONTEXT_REQUEST}"
            )

    def predict_with_context(
        self,
        *,
        seq: int,
        image_b64: str,
        instruction: str,
        artifact_prefix_max_tokens: int,
        prefix_hidden_dim: int,
        expected_source_sha256: str | None,
        feature_artifact_root: Path | None,
    ) -> ContextualPrediction:
        request = next_request(seq=seq, image_b64=image_b64, instruction=instruction)
        data = _mapping(request["data"], "LightNav next request data")
        data["feature_request"] = CONTEXT_REQUEST
        response = self._rpc(request)
        prediction = parse_lightnav_response(response, expected_seq=seq)
        response_data = _mapping(response.get("data"), "LightNav next response data")
        artifact = load_context_artifact(
            _mapping(response_data.get("feature_ref"), "LightNav feature_ref"),
            artifact_prefix_max_tokens=artifact_prefix_max_tokens,
            prefix_hidden_dim=prefix_hidden_dim,
            expected_source_sha256=expected_source_sha256,
            feature_artifact_root=feature_artifact_root,
        )
        return ContextualPrediction(prediction=prediction, artifact=artifact)


class ContinuousOnlineTopology(OnlineTopology):
    """OnlineTopology graph updated from executed SE(2) and realized path only."""

    def update_executed(
        self,
        primitives: Sequence[tuple[float, float, bool, bool]],
    ) -> None:
        """Update from ``(forward, yaw, collision, blocked)`` feedback.

        ``forward`` is realized path distance and ``yaw`` is the executed yaw
        delta.  Lateral simulator pose and goal state are intentionally absent.
        """

        if not primitives or len(primitives) > MAX_RECOVERY_PRIMITIVES:
            raise NavigationRLTRolloutError("topology update primitive count differs")
        for realized, yaw, collision, blocked in primitives:
            if not all(math.isfinite(item) for item in (realized, yaw)) or realized < 0.0:
                raise NavigationRLTRolloutError("topology feedback is non-finite")
            if type(collision) is not bool or type(blocked) is not bool:
                raise NavigationRLTRolloutError("topology feedback flags must be bool")
            self.primitive_count += 1
            attempted_forward = realized > 0.0 or blocked
            if attempted_forward:
                self.forward_attempts += 1
                if blocked:
                    self.blocked_forwards += 1
                    self.consecutive_blocked_forwards += 1
                    self.last_forward_succeeded = 0.0
                else:
                    self.consecutive_blocked_forwards = 0
                    self.last_forward_succeeded = 1.0
            if abs(yaw) > 1.0e-8:
                self.turn_count += 1
            middle = self.heading + 0.5 * yaw
            if realized > 0.0:
                self.x += math.sin(middle) * realized
                self.z += math.cos(middle) * realized
            self.heading = math.atan2(
                math.sin(self.heading + yaw), math.cos(self.heading + yaw)
            )
            del collision  # retained in replay; blocked/path drive causal odometry
        self.decision_count += 1
        self._update_graph()


@dataclass(frozen=True)
class DecisionState:
    prediction: LightNavPrediction
    artifact: ContextArtifact
    prefix_hidden: np.ndarray
    prefix_mask: np.ndarray
    topology_node_targets: np.ndarray
    topology_node_mask: np.ndarray
    topology_global_targets: np.ndarray
    topology_global_mask: np.ndarray
    reference_trajectory: np.ndarray
    reference_selected_index: int
    native_stop: bool
    valid_mode_mask: np.ndarray
    recovery_parent_index: int
    recovery_parent_waypoint: tuple[float, float, float] | None
    reference_embedding: np.ndarray
    prefix_original_valid_count: int
    prefix_selected_valid_count: int
    prefix_selected_source_indices: tuple[int, ...]
    prefix_selection_schema: str

    def feature_evidence(self) -> dict[str, object]:
        result = self.artifact.evidence()
        result.update(
            {
                "original_valid_count": self.prefix_original_valid_count,
                "selected_valid_count": self.prefix_selected_valid_count,
                "selected_source_indices": list(self.prefix_selected_source_indices),
                "selection_schema": self.prefix_selection_schema,
                "selection_rule": POLICY_PREFIX_SELECTION_RULE,
            }
        )
        return result


def _build_decision_state(
    contextual: ContextualPrediction,
    topology: ContinuousOnlineTopology,
    *,
    artifact_prefix_max_tokens: int,
    policy_prefix_max_tokens: int,
    prefix_hidden_dim: int,
    topology_max_nodes: int,
    decision_index: int,
    max_steps: int,
) -> DecisionState:
    prediction = contextual.prediction
    trajectory = np.asarray(prediction.waypoints, dtype=np.float32)
    if trajectory.shape != (10, 3) or not np.isfinite(trajectory).all():
        raise NavigationRLTRolloutError("LightNav reference trajectory must be finite [10,3]")
    if not 0 <= prediction.selected_index < 10 or not np.allclose(
        trajectory[prediction.selected_index],
        np.asarray(prediction.selected_waypoint, dtype=np.float32),
        rtol=1.0e-6,
        atol=1.0e-6,
    ):
        raise NavigationRLTRolloutError("LightNav selected reference differs from trajectory")

    import torch

    from navigation_rlt_policy import (
        PREFIX_SELECTION_SCHEMA,
        NavigationRLTConfig,
        select_prefix_tokens,
    )

    selection_config = NavigationRLTConfig(
        prefix_hidden_dim=prefix_hidden_dim,
        topology_dim=len(CAUSAL_GLOBAL_FEATURE_NAMES),
        reference_dim=REFERENCE_DIM,
        topology_node_dim=CAUSAL_NODE_FEATURE_DIM,
        topology_global_dim=len(CAUSAL_GLOBAL_FEATURE_NAMES),
        max_prefix_tokens=policy_prefix_max_tokens,
        max_artifact_prefix_tokens=artifact_prefix_max_tokens,
        max_topology_tokens=topology_max_nodes + 1,
    )
    selected_prefix = select_prefix_tokens(
        torch.from_numpy(contextual.artifact.hidden)[None, :, :],
        torch.ones(
            (1, contextual.artifact.valid_count), dtype=torch.bool
        ),
        config=selection_config,
    )
    if PREFIX_SELECTION_SCHEMA != POLICY_PREFIX_SELECTION_SCHEMA:
        raise NavigationRLTRolloutError("policy prefix selection schema drifted")
    prefix = (
        selected_prefix.contextual_prefix_hidden[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float16, copy=False)
    )
    prefix_mask = (
        selected_prefix.prefix_mask[0].detach().cpu().numpy().astype(np.bool_, copy=False)
    )
    original_valid_count = int(
        selected_prefix.original_valid_lengths[0].detach().cpu().item()
    )
    selected_valid_count = int(
        selected_prefix.selected_valid_lengths[0].detach().cpu().item()
    )
    selected_indices_array = (
        selected_prefix.selected_source_indices[0, :selected_valid_count]
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64, copy=False)
    )
    selected_source_indices = tuple(int(value) for value in selected_indices_array)
    if (
        prefix.shape != (policy_prefix_max_tokens, prefix_hidden_dim)
        or prefix_mask.shape != (policy_prefix_max_tokens,)
        or original_valid_count != contextual.artifact.valid_count
        or selected_valid_count != min(original_valid_count, policy_prefix_max_tokens)
        or len(selected_source_indices) != selected_valid_count
    ):
        raise NavigationRLTRolloutError("policy prefix selection output differs")

    payload = causal_topology_payload(topology, max_nodes=topology_max_nodes)
    node_count = len(payload.node_features)
    nodes = np.zeros((topology_max_nodes, CAUSAL_NODE_FEATURE_DIM), dtype=np.float32)
    node_mask = np.zeros((topology_max_nodes,), dtype=np.bool_)
    nodes[:node_count] = np.asarray(payload.node_features, dtype=np.float32)
    node_mask[:node_count] = True
    globals_ = np.asarray(payload.global_features, dtype=np.float32)
    global_mask = np.ones((len(CAUSAL_GLOBAL_FEATURE_NAMES),), dtype=np.bool_)
    parent_index = int(payload.parent_index[payload.current_index])
    parent_waypoint = None
    if parent_index >= 0:
        forward = float(nodes[parent_index, 0])
        lateral = float(nodes[parent_index, 1])
        parent_waypoint = (
            forward,
            lateral,
            math.atan2(lateral, forward) if math.hypot(forward, lateral) > 1.0e-8 else 0.0,
        )
    mode_mask = np.asarray(
        valid_mode_mask(
            native_stop=bool(prediction.stop), has_recovery_parent=parent_index >= 0
        ),
        dtype=np.bool_,
    )

    normalized_trajectory = trajectory.copy()
    normalized_trajectory[:, :2] = np.clip(normalized_trajectory[:, :2], -1.0, 1.0)
    normalized_trajectory[:, 2] = np.clip(normalized_trajectory[:, 2] / math.pi, -1.0, 1.0)
    selected = normalized_trajectory[prediction.selected_index]
    selected_one_hot = np.zeros((10,), dtype=np.float32)
    selected_one_hot[prediction.selected_index] = 1.0
    reference = np.concatenate(
        (
            normalized_trajectory.reshape(-1),
            selected_one_hot,
            selected,
            mode_mask.astype(np.float32),
            globals_,
            np.asarray(
                (
                    float(prediction.stop),
                    min(decision_index, max_steps) / max_steps,
                    prediction.selected_index / 9.0,
                ),
                dtype=np.float32,
            ),
        )
    )
    if reference.shape != (REFERENCE_DIM,) or not np.isfinite(reference).all():
        raise NavigationRLTRolloutError("reference embedding compiler produced wrong shape")
    return DecisionState(
        prediction=prediction,
        artifact=contextual.artifact,
        prefix_hidden=prefix,
        prefix_mask=prefix_mask,
        topology_node_targets=nodes,
        topology_node_mask=node_mask,
        topology_global_targets=globals_,
        topology_global_mask=global_mask,
        reference_trajectory=trajectory,
        reference_selected_index=int(prediction.selected_index),
        native_stop=bool(prediction.stop),
        valid_mode_mask=mode_mask,
        recovery_parent_index=parent_index,
        recovery_parent_waypoint=parent_waypoint,
        reference_embedding=reference,
        prefix_original_valid_count=original_valid_count,
        prefix_selected_valid_count=selected_valid_count,
        prefix_selected_source_indices=selected_source_indices,
        prefix_selection_schema=PREFIX_SELECTION_SCHEMA,
    )


def _state_arrays(state: DecisionState, prefix: str = "") -> dict[str, np.ndarray]:
    return {
        prefix + "prefix_hidden": state.prefix_hidden,
        prefix + "prefix_mask": state.prefix_mask,
        prefix + "topology_node_targets": state.topology_node_targets,
        prefix + "topology_node_mask": state.topology_node_mask,
        prefix + "topology_global_targets": state.topology_global_targets,
        prefix + "topology_global_mask": state.topology_global_mask,
        prefix + "reference_embedding": state.reference_embedding,
        prefix + "reference_trajectory": state.reference_trajectory,
        prefix + "reference_selected_index": np.asarray(
            state.reference_selected_index, dtype=np.int64
        ),
        prefix + "native_stop": np.asarray(state.native_stop, dtype=np.bool_),
        prefix + "valid_mode_mask": state.valid_mode_mask,
        prefix + "recovery_parent_index": np.asarray(
            state.recovery_parent_index, dtype=np.int64
        ),
    }


def _compiler_fields(state: DecisionState):
    import torch

    from navigation_rlt_policy import NavigationRLTCompilerFields

    selected = np.asarray(
        clip_reference_waypoint(state.prediction.selected_waypoint), dtype=np.float32
    )
    reference_action = np.concatenate(
        (np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32), selected)
    )
    adjust_base_action = np.concatenate(
        (np.asarray((0.0, 1.0, 0.0, 0.0), dtype=np.float32), selected)
    )
    stop = compile_action(
        mode=STOP,
        valid_modes=state.valid_mode_mask,
        reference_waypoint=state.prediction.selected_waypoint,
    )
    if state.recovery_parent_waypoint is None:
        recover_action = np.asarray(
            (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0), dtype=np.float32
        )
        parent_valid = False
    else:
        recover = compile_action(
            mode=RECOVER,
            valid_modes=state.valid_mode_mask,
            reference_waypoint=state.prediction.selected_waypoint,
            recovery_parent_waypoint=state.recovery_parent_waypoint,
            recovery_parent_index=state.recovery_parent_index,
        )
        recover_action = np.asarray(recover.canonical_action, dtype=np.float32)
        parent_valid = True
    basis = np.zeros((2, CANONICAL_ACTION_DIM), dtype=np.float32)
    basis[0, 4] = 1.0
    basis[1, 6] = 1.0
    return NavigationRLTCompilerFields(
        valid_gate_mask=torch.from_numpy(state.valid_mode_mask[None, :]),
        reference_action=torch.from_numpy(reference_action[None, :]),
        adjust_base_action=torch.from_numpy(adjust_base_action[None, :]),
        adjust_residual_basis=torch.from_numpy(basis[None, :, :]),
        stop_action=torch.from_numpy(
            np.asarray(stop.canonical_action, dtype=np.float32)[None, :]
        ),
        recover_parent_action=torch.from_numpy(recover_action[None, :]),
        recover_parent_valid=torch.tensor([parent_valid], dtype=torch.bool),
        recover_parent_index=torch.tensor(
            [state.recovery_parent_index if parent_valid else -1], dtype=torch.int64
        ),
    )


def _policy_context(state: DecisionState, config: Any):
    import torch

    from navigation_rlt_policy import (
        NavigationRLTContext,
        pack_topology_tokens,
    )

    if state.prefix_hidden.shape != (
        config.max_prefix_tokens,
        config.prefix_hidden_dim,
    ) or state.prefix_mask.shape != (config.max_prefix_tokens,):
        raise NavigationRLTRolloutError(
            "selected replay prefix differs from the Navigation-RLT policy width"
        )
    topology = pack_topology_tokens(
        torch.from_numpy(state.topology_node_targets)[None, :, :],
        torch.from_numpy(state.topology_node_mask)[None, :],
        torch.from_numpy(state.topology_global_targets)[None, :],
        torch.from_numpy(state.topology_global_mask)[None, :],
        config=config,
    )
    return NavigationRLTContext(
        contextual_prefix_hidden=torch.from_numpy(
            state.prefix_hidden.astype(np.float32, copy=False)
        )[None, :, :],
        prefix_mask=torch.from_numpy(state.prefix_mask)[None, :],
        topology_tokens=topology.tokens,
        topology_mask=topology.token_mask,
    )


@dataclass
class Behavior:
    name: str
    version: int
    source: dict[str, object]
    a3_policy: NumpyLightNavPolicy | None = None
    rlt_policy: Any | None = None
    torch_generator: Any | None = None
    deterministic: bool = False

    def request_action(
        self,
        state: DecisionState,
        *,
        a3_state: CausalTopologyState,
        decision_index: int,
        max_steps: int,
        rng: np.random.Generator,
    ) -> CompiledAction:
        if self.name == BEHAVIOR_NATIVE:
            mode = STOP if state.native_stop else KEEP
            raw = (0.0, 0.0)
        elif self.name == BEHAVIOR_A3:
            if state.native_stop:
                mode = STOP
                raw = (0.0, 0.0)
            else:
                if self.a3_policy is None:
                    raise NavigationRLTRolloutError("A3 behavior policy is missing")
                features = build_a3_features(
                    state.prediction.waypoints,
                    selected_index=state.prediction.selected_index,
                    selected_waypoint=state.prediction.selected_waypoint,
                    model_stop=state.prediction.stop,
                    step_index=min(decision_index, max_steps - 1),
                    max_steps=max_steps,
                    state=a3_state,
                )
                action = self.a3_policy.act(
                    features, rng=rng, deterministic=self.deterministic
                )
                mode = int(action.gate)
                raw = action.raw_residual if mode == ADJUST else (0.0, 0.0)
        elif self.name == BEHAVIOR_RLT:
            if self.rlt_policy is None:
                raise NavigationRLTRolloutError("Navigation-RLT behavior policy is missing")
            import torch

            context = _policy_context(state, self.rlt_policy.config)
            reference = torch.from_numpy(state.reference_embedding[None, :])
            compiler = _compiler_fields(state)
            if self.deterministic:
                action = self.rlt_policy.act_deterministic(context, reference, compiler)
            else:
                action = self.rlt_policy.act_stochastic(
                    context,
                    reference,
                    compiler,
                    generator=self.torch_generator,
                )
            mode = int(action.gate.detach().cpu().item())
            raw_values = action.raw_residual.detach().cpu().numpy()[0]
            raw = (
                (float(raw_values[0]), float(raw_values[1]))
                if mode == ADJUST
                else (0.0, 0.0)
            )
        else:  # pragma: no cover - parser and loader reject this first
            raise NavigationRLTRolloutError(f"unknown behavior {self.name}")
        return compile_action(
            mode=mode,
            valid_modes=state.valid_mode_mask,
            reference_waypoint=state.prediction.selected_waypoint,
            raw_residual=raw,
            recovery_parent_waypoint=state.recovery_parent_waypoint,
            recovery_parent_index=state.recovery_parent_index,
        )


def _load_behavior(args: argparse.Namespace) -> Behavior:
    if args.behavior == BEHAVIOR_NATIVE:
        if args.a3_checkpoint is not None or args.rlt_checkpoint is not None:
            raise NavigationRLTRolloutError("native behavior does not accept a policy checkpoint")
        return Behavior(
            name=BEHAVIOR_NATIVE,
            version=0,
            source={"kind": BEHAVIOR_NATIVE, "lightnav_revision": LIGHTNAV_SOURCE_REVISION},
            deterministic=True,
        )
    if args.behavior == BEHAVIOR_A3:
        if args.a3_checkpoint is None or args.expected_a3_checkpoint_sha256 is None:
            raise NavigationRLTRolloutError("A3 behavior requires its checkpoint and SHA-256")
        if args.rlt_checkpoint is not None:
            raise NavigationRLTRolloutError("A3 behavior cannot load a Navigation-RLT checkpoint")
        observed = sha256_file(args.a3_checkpoint)
        if observed != _validate_sha256(
            args.expected_a3_checkpoint_sha256, "expected A3 checkpoint SHA"
        ):
            raise NavigationRLTRolloutError("A3 checkpoint SHA-256 differs")
        policy = NumpyLightNavPolicy.from_checkpoint(args.a3_checkpoint)
        policy_version = int(policy.metadata["policy_version"])
        return Behavior(
            name=BEHAVIOR_A3,
            version=100_000 + policy_version,
            source={
                "kind": BEHAVIOR_A3,
                "checkpoint_sha256": observed,
                "checkpoint_policy_version": policy_version,
            },
            a3_policy=policy,
            deterministic=bool(args.deterministic_policy),
        )

    if args.rlt_checkpoint is None or args.expected_rlt_checkpoint_sha256 is None:
        raise NavigationRLTRolloutError(
            "Navigation-RLT behavior requires its checkpoint and SHA-256"
        )
    if args.a3_checkpoint is not None:
        raise NavigationRLTRolloutError("Navigation-RLT behavior cannot load an A3 checkpoint")
    observed = sha256_file(args.rlt_checkpoint)
    if observed != _validate_sha256(
        args.expected_rlt_checkpoint_sha256, "expected Navigation-RLT checkpoint SHA"
    ):
        raise NavigationRLTRolloutError("Navigation-RLT checkpoint SHA-256 differs")
    import torch

    from navigation_rlt_policy import NavigationRLTPolicy

    policy, metadata = NavigationRLTPolicy.load_checkpoint(
        args.rlt_checkpoint, map_location=args.policy_device
    )
    policy.to(args.policy_device).eval()
    config = policy.config
    if (
        config.prefix_hidden_dim != args.prefix_hidden_dim
        or config.reference_dim != REFERENCE_DIM
        or config.canonical_action_dim != CANONICAL_ACTION_DIM
        or config.topology_node_dim != CAUSAL_NODE_FEATURE_DIM
        or config.topology_global_dim != len(CAUSAL_GLOBAL_FEATURE_NAMES)
        or config.max_prefix_tokens != args.policy_prefix_max_tokens
        or config.max_artifact_prefix_tokens < args.artifact_prefix_max_tokens
        or config.max_topology_tokens < args.topology_max_nodes + 1
    ):
        raise NavigationRLTRolloutError("Navigation-RLT checkpoint feature/action config differs")
    if not bool(policy.stage1_frozen.item()):
        raise NavigationRLTRolloutError("Navigation-RLT behavior requires a frozen Stage 1")
    policy_version = int(metadata["policy_version"])
    policy_device = next(policy.parameters()).device
    generator = torch.Generator(device=policy_device)
    generator.manual_seed(int(args.seed) + int(args.worker_id))
    return Behavior(
        name=BEHAVIOR_RLT,
        version=1_000_000 + policy_version,
        source={
            "kind": BEHAVIOR_RLT,
            "checkpoint_sha256": observed,
            "checkpoint_policy_version": policy_version,
            "reference_schema": REFERENCE_SCHEMA,
            "prefix_selection": {
                "schema": config.prefix_selection_schema,
                "policy_max_prefix_tokens": config.max_prefix_tokens,
                "rule": "keep endpoints and uniformly sample valid-token ordinals in between",
            },
            "topology_packing_schema": config.topology_packing_schema,
        },
        rlt_policy=policy,
        torch_generator=generator,
        deterministic=bool(args.deterministic_policy),
    )


def _load_worker_plan(
    path: Path, *, expected_sha256: str, worker_id: int
) -> tuple[dict[str, object], dict[str, object], str]:
    path = Path(path).resolve(strict=True)
    observed = sha256_file(path)
    if observed != _validate_sha256(expected_sha256, "worker plan SHA"):
        raise NavigationRLTRolloutError("parallel worker plan SHA-256 differs")
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NavigationRLTRolloutError(f"worker plan is invalid JSON: {exc}") from exc
    if (
        not isinstance(plan, dict)
        or plan.get("schema_name") != PLAN_SCHEMA
        or plan.get("schema_version") != PLAN_VERSION
        or plan.get("split_label") != "train_update"
        or plan.get("validation_split_reads") != 0
    ):
        raise NavigationRLTRolloutError("parallel worker plan is not fixed train_update")
    workers = plan.get("workers")
    if not isinstance(workers, list):
        raise NavigationRLTRolloutError("parallel worker plan has no worker inventory")
    matches = [row for row in workers if isinstance(row, dict) and row.get("worker_id") == worker_id]
    if len(matches) != 1:
        raise NavigationRLTRolloutError("worker id is absent or duplicated in the plan")
    worker = matches[0]
    episodes = worker.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != worker.get("episode_count") or not episodes:
        raise NavigationRLTRolloutError("worker episode inventory differs")
    return plan, worker, observed


def _episode_reset_token(
    *, run_id: str, worker_id: int, episode: Mapping[str, object]
) -> str:
    basis = {
        "run_id": run_id,
        "worker_id": worker_id,
        "episode_order": episode["episode_order"],
        "episode_id": episode["episode_id"],
        "scene": episode["scene"],
        "instruction_text_sha256": episode["instruction_text_sha256"],
    }
    return _sha256_bytes(_canonical_json(basis))


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".staging")
    with staging.open("wb") as stream:
        for row in rows:
            stream.write(_canonical_json(row) + b"\n")
        stream.flush()
    staging.replace(path)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "row_count": len(rows),
    }


def _verified_output_receipt(
    receipt: Mapping[str, object], name: str
) -> dict[str, object]:
    raw_path = receipt.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise NavigationRLTRolloutError(f"{name} receipt path is missing")
    path = Path(raw_path)
    _regular_file_no_reparse(path, f"{name} output")
    resolved = path.resolve(strict=True)
    expected_sha = _validate_sha256(receipt.get("sha256"), f"{name} receipt SHA")
    observed_sha = sha256_file(resolved)
    if observed_sha != expected_sha:
        raise NavigationRLTRolloutError(f"{name} output SHA-256 differs from its receipt")
    size = int(resolved.stat().st_size)
    if receipt.get("size_bytes") != size:
        raise NavigationRLTRolloutError(f"{name} output size differs from its receipt")
    return {"path": str(resolved), "sha256": observed_sha, "size_bytes": size}


def _artifact_relative_path(path: Path, root: Path, name: str) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise NavigationRLTRolloutError(f"{name} escapes the feature artifact root") from exc
    if len(relative.parts) != 3:
        raise NavigationRLTRolloutError(
            f"{name} is not an exact content-addressed artifact file"
        )
    return relative.as_posix()


def consume_feature_artifacts(
    artifacts: Sequence[ContextArtifact],
    *,
    root: Path,
    receipt_path: Path,
    replay_receipt: Mapping[str, object],
    feature_receipt: Mapping[str, object],
) -> dict[str, object]:
    """Delete authenticated content-addressed inputs after durable episode outputs.

    The caller invokes this only after one complete replay shard and its feature
    evidence are durable.  This function re-verifies those receipts, performs a
    mutation-free preflight of every exact manifest-listed file, writes a ready
    receipt, and then uses only ``unlink`` and non-recursive ``rmdir``.
    """

    root = _resolve_artifact_root(root)
    replay_output = _verified_output_receipt(replay_receipt, "replay")
    feature_output = _verified_output_receipt(feature_receipt, "feature evidence")
    if not artifacts:
        raise NavigationRLTRolloutError("episode has no validated feature artifacts")

    deduplicated: dict[tuple[str, str], ContextArtifact] = {}
    identity_paths: dict[str, str] = {}
    manifest_identities: dict[str, str] = {}
    for artifact in artifacts:
        path_key = str(artifact.manifest_path)
        prior_path = identity_paths.setdefault(artifact.identity, path_key)
        if prior_path != path_key:
            raise NavigationRLTRolloutError(
                "one artifact identity maps to multiple manifest paths"
            )
        prior_identity = manifest_identities.setdefault(path_key, artifact.identity)
        if prior_identity != artifact.identity:
            raise NavigationRLTRolloutError(
                "one artifact manifest path maps to multiple identities"
            )
        deduplicated.setdefault((artifact.identity, path_key), artifact)

    receipt: dict[str, object] = {
        "schema_name": ARTIFACT_CLEANUP_SCHEMA,
        "schema_version": ARTIFACT_CLEANUP_VERSION,
        "status": "preflight",
        "root": str(root),
        "replay_output": replay_output,
        "feature_evidence_output": feature_output,
        "artifact_count": len(deduplicated),
        "artifacts": [],
        "started_unix_s": time.time(),
    }
    prepared: list[tuple[ContextArtifact, Path, Path, list[ArtifactFile], dict[str, object]]] = []
    try:
        for artifact in deduplicated.values():
            if artifact.content_root != root:
                raise NavigationRLTRolloutError(
                    "artifact was not validated against this exact artifact root"
                )
            prefix_dir = root / artifact.identity[:2]
            identity_dir = prefix_dir / artifact.identity
            exact_manifest = identity_dir / "manifest.json"
            if artifact.manifest_path != exact_manifest:
                raise NavigationRLTRolloutError(
                    "recorded manifest differs from the exact content-addressed path"
                )
            expected_files = list(artifact.files) + [
                ArtifactFile(
                    path=artifact.manifest_path,
                    sha256=artifact.manifest_sha256,
                    size_bytes=artifact.manifest_size_bytes,
                )
            ]
            expected_paths = {item.path for item in expected_files}
            if len(expected_paths) != len(expected_files):
                raise NavigationRLTRolloutError("artifact file inventory contains duplicates")
            for item in expected_files:
                if item.path.parent != identity_dir:
                    raise NavigationRLTRolloutError(
                        "artifact inventory contains a file outside its identity directory"
                    )
                _artifact_relative_path(item.path, root, "artifact file")

            exists = [_path_entry_exists(item.path) for item in expected_files]
            if any(exists) and not all(exists):
                raise NavigationRLTRolloutError(
                    "artifact is partially missing before cleanup; refusing mutation"
                )
            initially_consumed = not any(exists)
            if initially_consumed:
                if _path_entry_exists(identity_dir):
                    _directory_no_reparse(identity_dir, "empty artifact identity directory")
                    if any(identity_dir.iterdir()):
                        raise NavigationRLTRolloutError(
                            "missing artifact files leave unexpected identity-directory entries"
                        )
                if _path_entry_exists(prefix_dir):
                    _directory_no_reparse(prefix_dir, "artifact prefix directory")
            else:
                _directory_no_reparse(prefix_dir, "artifact prefix directory")
                _directory_no_reparse(identity_dir, "artifact identity directory")
                directory_entries = set(identity_dir.iterdir())
                if directory_entries != expected_paths:
                    raise NavigationRLTRolloutError(
                        "artifact identity directory differs from the manifest inventory"
                    )
                for item in expected_files:
                    _regular_file_no_reparse(item.path, "artifact cleanup file")
                    if int(item.path.stat().st_size) != item.size_bytes:
                        raise NavigationRLTRolloutError(
                            "artifact cleanup file size changed after validation"
                        )
                    if sha256_file(item.path) != item.sha256:
                        raise NavigationRLTRolloutError(
                            "artifact cleanup file SHA-256 changed after validation"
                        )

            artifact_row: dict[str, object] = {
                "identity": artifact.identity,
                "manifest": _artifact_relative_path(
                    artifact.manifest_path, root, "feature manifest"
                ),
                "status": "already_consumed" if initially_consumed else "ready",
                "files": [
                    {
                        "path": _artifact_relative_path(item.path, root, "artifact file"),
                        "sha256": item.sha256,
                        "size_bytes": item.size_bytes,
                        "status": "already_consumed" if initially_consumed else "verified",
                    }
                    for item in expected_files
                ],
                "identity_directory": "present_empty" if initially_consumed and _path_entry_exists(identity_dir) else (
                    "absent" if initially_consumed else "verified"
                ),
            }
            cast_rows = receipt["artifacts"]
            if not isinstance(cast_rows, list):  # internal receipt invariant
                raise NavigationRLTRolloutError("cleanup receipt artifact rows drifted")
            cast_rows.append(artifact_row)
            prepared.append(
                (artifact, prefix_dir, identity_dir, expected_files, artifact_row)
            )

        receipt["status"] = "ready"
        atomic_json(receipt_path, receipt)

        for artifact, prefix_dir, identity_dir, expected_files, artifact_row in prepared:
            if artifact_row["status"] != "already_consumed":
                concurrent_missing = False
                file_rows = artifact_row.get("files")
                if not isinstance(file_rows, list) or len(file_rows) != len(expected_files):
                    raise NavigationRLTRolloutError("cleanup receipt file rows drifted")
                for item, file_row in zip(expected_files, file_rows):
                    if not isinstance(file_row, dict):
                        raise NavigationRLTRolloutError("cleanup receipt file row drifted")
                    try:
                        prefix_dir.lstat()
                        identity_dir.lstat()
                        item.path.lstat()
                        _directory_no_reparse(prefix_dir, "artifact prefix directory")
                        _directory_no_reparse(identity_dir, "artifact identity directory")
                        _regular_file_no_reparse(item.path, "artifact cleanup file")
                        if item.path.resolve(strict=True) != item.path:
                            raise NavigationRLTRolloutError(
                                "artifact cleanup file resolution changed"
                            )
                        item.path.unlink()
                        file_row["status"] = "deleted"
                    except FileNotFoundError:
                        concurrent_missing = True
                        break
                    except NavigationRLTRolloutError:
                        if not any(_path_entry_exists(row.path) for row in expected_files):
                            concurrent_missing = True
                            break
                        raise
                if concurrent_missing:
                    if any(_path_entry_exists(item.path) for item in expected_files):
                        raise NavigationRLTRolloutError(
                            "artifact disappeared concurrently but still has partial files"
                        )
                    artifact_row["status"] = "already_consumed"
                    for file_row in artifact_row["files"]:
                        if isinstance(file_row, dict) and file_row["status"] == "verified":
                            file_row["status"] = "already_consumed"
                else:
                    artifact_row["status"] = "deleted"

            if any(_path_entry_exists(item.path) for item in expected_files):
                raise NavigationRLTRolloutError(
                    "artifact cleanup left a manifest-listed file behind"
                )
            if _path_entry_exists(identity_dir):
                _directory_no_reparse(identity_dir, "artifact identity directory")
                if any(identity_dir.iterdir()):
                    raise NavigationRLTRolloutError(
                        "artifact identity directory is not empty after exact deletion"
                    )
                try:
                    identity_dir.rmdir()
                    artifact_row["identity_directory"] = "deleted"
                except FileNotFoundError:
                    artifact_row["identity_directory"] = "already_consumed"
            else:
                artifact_row["identity_directory"] = "already_consumed"
            if _path_entry_exists(prefix_dir):
                _directory_no_reparse(prefix_dir, "artifact prefix directory")
                if not any(prefix_dir.iterdir()):
                    try:
                        prefix_dir.rmdir()
                        artifact_row["prefix_directory"] = "deleted"
                    except FileNotFoundError:
                        artifact_row["prefix_directory"] = "already_consumed"
                else:
                    artifact_row["prefix_directory"] = "retained_nonempty"
            else:
                artifact_row["prefix_directory"] = "already_consumed"

        receipt["status"] = "completed"
        receipt["completed_unix_s"] = time.time()
        atomic_json(receipt_path, receipt)
        return {
            "path": str(receipt_path.resolve(strict=True)),
            "sha256": sha256_file(receipt_path),
            "size_bytes": int(receipt_path.stat().st_size),
            "artifact_count": len(prepared),
            "status": "completed",
        }
    except Exception as exc:
        receipt["status"] = "failed"
        receipt["completed_unix_s"] = time.time()
        receipt["error"] = {"type": type(exc).__name__, "message": str(exc)}
        atomic_json(receipt_path, receipt)
        raise


def _query_state(
    model: ContextLightNavClient,
    observation: Mapping[str, object],
    instruction: str,
    topology: ContinuousOnlineTopology,
    *,
    seq: int,
    jpeg_quality: int,
    artifact_prefix_max_tokens: int,
    policy_prefix_max_tokens: int,
    prefix_hidden_dim: int,
    topology_max_nodes: int,
    decision_index: int,
    max_steps: int,
    expected_source_sha256: str | None,
    feature_artifact_root: Path | None,
) -> DecisionState:
    if "rgb" not in observation:
        raise HabitatProtocolError("Habitat observation has no RGB frame")
    contextual = model.predict_with_context(
        seq=seq,
        image_b64=encode_rgb_jpeg_base64(observation["rgb"], quality=jpeg_quality),
        instruction=instruction,
        artifact_prefix_max_tokens=artifact_prefix_max_tokens,
        prefix_hidden_dim=prefix_hidden_dim,
        expected_source_sha256=expected_source_sha256,
        feature_artifact_root=feature_artifact_root,
    )
    return _build_decision_state(
        contextual,
        topology,
        artifact_prefix_max_tokens=artifact_prefix_max_tokens,
        policy_prefix_max_tokens=policy_prefix_max_tokens,
        prefix_hidden_dim=prefix_hidden_dim,
        topology_max_nodes=topology_max_nodes,
        decision_index=decision_index,
        max_steps=max_steps,
    )


def _end_reason(
    *, done: bool, truncated: bool, stopped: bool, success: bool, collision: bool, budget: bool
) -> int:
    if not done:
        name = "running"
    elif budget:
        name = "max_steps"
    elif truncated:
        name = "environment_truncated"
    elif stopped:
        name = "policy_stop_success" if success else "policy_stop_failure"
    elif success:
        name = "success"
    elif collision:
        name = "collision"
    else:
        name = "environment_terminal"
    return END_REASON_NAMES.index(name)


def _stack_rows(rows: Sequence[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not rows:
        raise NavigationRLTRolloutError("episode produced no transition rows")
    keys = set(rows[0])
    if any(set(row) != keys for row in rows):
        raise NavigationRLTRolloutError("transition row array members differ")
    return {name: np.stack([np.asarray(row[name]) for row in rows], axis=0) for name in sorted(keys)}


def run_episode(
    env: ZmqHabitatClient,
    model: ContextLightNavClient,
    behavior: Behavior,
    expected_episode: Mapping[str, object],
    *,
    run_id: str,
    worker_id: int,
    rng: np.random.Generator,
    max_steps: int,
    jpeg_quality: int,
    artifact_prefix_max_tokens: int,
    policy_prefix_max_tokens: int,
    prefix_hidden_dim: int,
    topology_max_nodes: int,
    expected_context_source_sha256: str | None,
    feature_artifact_root: Path | None,
    require_collisions: bool,
    gamma: float,
) -> tuple[
    dict[str, object],
    dict[str, np.ndarray],
    dict[str, object],
    list[dict[str, object]],
    list[float],
    tuple[ContextArtifact, ...],
]:
    observation, reset_info = env.reset()
    model.reset()
    instruction = extract_instruction(observation)
    instruction_sha = _sha256_bytes(instruction.encode("utf-8"))
    expected_instruction = _validate_sha256(
        expected_episode.get("instruction_text_sha256"), "planned instruction hash"
    )
    actual_episode_id = str(reset_info.get("episode_id", ""))
    actual_scene = _scene_name(reset_info.get("scene_id", ""))
    if (
        actual_episode_id != str(expected_episode.get("episode_id"))
        or actual_scene != str(expected_episode.get("scene"))
        or instruction_sha != expected_instruction
    ):
        raise NavigationRLTRolloutError("Habitat reset identity differs from worker plan")

    velocity = VelocitySpec.from_info(reset_info)
    _validate_velocity_contract(velocity)
    initial_distance = _metric(
        reset_info,
        ("distance_to_goal", "goal_distance", "navigation_error"),
        "navigation error",
    )
    previous_distance = initial_distance
    minimum_distance = initial_distance
    initial_path = _optional_metric(reset_info, ("path_length",)) or 0.0
    previous_path = initial_path
    collision_cumulative, _, reset_collision_source = _collision_observation(reset_info)
    if require_collisions and reset_collision_source == "unavailable":
        raise HabitatProtocolError("formal Navigation-RLT replay requires collision feedback")

    topology = ContinuousOnlineTopology(
        max_decisions=max_steps,
        forward_step_m=0.25,
        turn_degrees=15.0,
        node_radius_m=0.5,
    )
    a3_state = CausalTopologyState(previous_path_length_m=initial_path)
    seq = 0
    current = _query_state(
        model,
        observation,
        instruction,
        topology,
        seq=seq,
        jpeg_quality=jpeg_quality,
        artifact_prefix_max_tokens=artifact_prefix_max_tokens,
        policy_prefix_max_tokens=policy_prefix_max_tokens,
        prefix_hidden_dim=prefix_hidden_dim,
        topology_max_nodes=topology_max_nodes,
        decision_index=0,
        max_steps=max_steps,
        expected_source_sha256=expected_context_source_sha256,
        feature_artifact_root=feature_artifact_root,
    )
    seq += 1
    source_sha = current.artifact.source_sha256
    if expected_context_source_sha256 is not None and source_sha != expected_context_source_sha256:
        raise NavigationRLTRolloutError("initial context source identity differs")

    rows: list[dict[str, np.ndarray]] = []
    feature_trace: list[dict[str, object]] = []
    artifacts: dict[tuple[str, str], ContextArtifact] = {}

    def remember_artifact(artifact: ContextArtifact) -> None:
        path_key = str(artifact.manifest_path)
        for (known_identity, known_path) in artifacts:
            if known_identity == artifact.identity and known_path != path_key:
                raise NavigationRLTRolloutError(
                    "one content identity resolved to multiple artifact paths"
                )
        artifacts.setdefault((artifact.identity, path_key), artifact)

    remember_artifact(current.artifact)
    latencies: list[float] = [current.prediction.latency_ms]
    action_counts: Counter[str] = Counter()
    requested_counts: Counter[str] = Counter()
    executed_counts: Counter[str] = Counter()
    blocked_steps = 0
    collision_events = 0
    override_count = 0
    physical_step = 0
    decision_index = 0
    final_info: Mapping[str, object] = reset_info
    stop_source = "none"
    timed_out = False

    while True:
        requested = behavior.request_action(
            current,
            a3_state=a3_state,
            decision_index=decision_index,
            max_steps=max_steps,
            rng=rng,
        )
        requested_counts[MODE_NAMES[requested.requested_mode]] += 1
        remaining = max_steps - physical_step
        if remaining < 1:
            raise NavigationRLTRolloutError("physical budget exhausted before terminal transition")
        budget_override = requested.requested_mode != STOP and len(requested.primitives) >= remaining
        if budget_override:
            executed = compile_action(
                mode=STOP,
                valid_modes=current.valid_mode_mask,
                reference_waypoint=current.prediction.selected_waypoint,
            )
            override_count += 1
            stop_source = "budget_stop"
        else:
            executed = requested
            if executed.executed_mode == STOP:
                if behavior.name in (BEHAVIOR_NATIVE, BEHAVIOR_A3) and current.native_stop:
                    stop_source = "native_lightnav_stop"
                else:
                    stop_source = "learned_stop"
        executed_counts[MODE_NAMES[executed.executed_mode]] += 1

        primitive_rewards = np.zeros((MAX_RECOVERY_PRIMITIVES,), dtype=np.float32)
        executed_mask = np.zeros((MAX_RECOVERY_PRIMITIVES,), dtype=np.bool_)
        topology_feedback: list[tuple[float, float, bool, bool]] = []
        transition_collision = False
        transition_blocked = False
        terminated_flag = False
        truncated_flag = False
        success = False

        for primitive_index, primitive in enumerate(executed.primitives):
            action = waypoint_to_velocity_action(primitive, velocity)
            normalized_linear, normalized_angular = _normalized_command(action)
            physical_linear, physical_angular = _physical_velocity(action, velocity)
            category = classify_velocity_action(action, velocity)
            action_counts[category] += 1
            observation, _, habitat_terminated, habitat_truncated, final_info = env.step(action)
            if not isinstance(observation, Mapping) or not isinstance(final_info, Mapping):
                raise HabitatProtocolError("Habitat step returned invalid observation/info")
            physical_step += 1
            executed_mask[primitive_index] = True
            current_distance = _metric(
                final_info,
                ("distance_to_goal", "goal_distance", "navigation_error"),
                "navigation error",
            )
            minimum_distance = min(minimum_distance, current_distance)
            path_length = _optional_metric(final_info, ("path_length",))
            if path_length is None:
                raise HabitatProtocolError("Navigation-RLT requires Habitat path_length")
            path_delta = max(0.0, path_length - previous_path)
            collided, collision_cumulative, collision_source = _collision_event(
                final_info, collision_cumulative
            )
            if require_collisions and collision_source == "unavailable":
                raise HabitatProtocolError("Navigation-RLT lost collision feedback")
            collision_events += int(collided)
            transition_collision = transition_collision or collided
            commanded_distance = abs(physical_linear * velocity.time_step_s)
            blocked = commanded_distance >= 0.03 and path_delta < 0.01
            blocked_steps += int(blocked)
            transition_blocked = transition_blocked or blocked
            stopped = category == "STOP"
            success = _bool_success(final_info.get("success", False))
            primitive_timeout = bool(budget_override or habitat_truncated)
            reward, _ = _reward(
                previous_distance=previous_distance,
                current_distance=current_distance,
                success=success,
                collision_event=collided,
                stopped=stopped,
                timeout=primitive_timeout,
            )
            primitive_rewards[primitive_index] = reward
            yaw_delta = math.radians(physical_angular) * velocity.time_step_s
            topology_feedback.append((path_delta, yaw_delta, bool(collided), bool(blocked)))
            if behavior.name == BEHAVIOR_A3:
                a3_state.update(
                    normalized_linear=normalized_linear,
                    normalized_angular=normalized_angular,
                    physical_linear_mps=physical_linear,
                    physical_angular_dps=physical_angular,
                    time_step_s=velocity.time_step_s,
                    gate=executed.executed_mode,
                    selected_waypoint=current.prediction.selected_waypoint,
                    path_length_m=path_length,
                )
            previous_distance = current_distance
            previous_path = path_length
            timed_out = timed_out or primitive_timeout
            truncated_flag = bool(primitive_timeout)
            terminated_flag = bool(habitat_terminated and not truncated_flag)
            if stopped and not (habitat_terminated or habitat_truncated):
                raise HabitatProtocolError("physical STOP did not terminate Habitat")
            if habitat_terminated or habitat_truncated or stopped or primitive_timeout:
                break

        duration = int(executed_mask.sum())
        if duration < 1:
            raise NavigationRLTRolloutError("compiled action executed no physical primitive")
        topology.update_executed(topology_feedback)
        done = bool(terminated_flag or truncated_flag)
        if not done and physical_step >= max_steps:
            raise NavigationRLTRolloutError("nonterminal transition consumed the physical budget")

        if done:
            next_state = _build_decision_state(
                ContextualPrediction(current.prediction, current.artifact),
                topology,
                artifact_prefix_max_tokens=artifact_prefix_max_tokens,
                policy_prefix_max_tokens=policy_prefix_max_tokens,
                prefix_hidden_dim=prefix_hidden_dim,
                topology_max_nodes=topology_max_nodes,
                decision_index=decision_index + 1,
                max_steps=max_steps,
            )
        else:
            next_state = _query_state(
                model,
                observation,
                instruction,
                topology,
                seq=seq,
                jpeg_quality=jpeg_quality,
                artifact_prefix_max_tokens=artifact_prefix_max_tokens,
                policy_prefix_max_tokens=policy_prefix_max_tokens,
                prefix_hidden_dim=prefix_hidden_dim,
                topology_max_nodes=topology_max_nodes,
                decision_index=decision_index + 1,
                max_steps=max_steps,
                expected_source_sha256=source_sha,
                feature_artifact_root=feature_artifact_root,
            )
            seq += 1
            latencies.append(next_state.prediction.latency_ms)
        remember_artifact(next_state.artifact)

        discounted_reward = float(
            np.sum(
                primitive_rewards.astype(np.float64)
                * np.power(gamma, np.arange(MAX_RECOVERY_PRIMITIVES, dtype=np.float64))
                * executed_mask
            )
        )
        transition: dict[str, np.ndarray] = {}
        transition.update(_state_arrays(current))
        transition.update(_state_arrays(next_state, "next_"))
        transition.update(
            {
                "requested_mode": np.asarray(requested.requested_mode, dtype=np.int64),
                "executed_mode": np.asarray(executed.executed_mode, dtype=np.int64),
                "raw_residual": np.asarray(requested.raw_residual, dtype=np.float32),
                "bounded_residual": np.asarray(requested.bounded_residual, dtype=np.float32),
                "resolved_se2": np.asarray(executed.resolved_se2, dtype=np.float32),
                "safety_override": np.asarray(budget_override, dtype=np.bool_),
                "credit_mask": np.asarray(not budget_override, dtype=np.bool_),
                "primitive_rewards": primitive_rewards,
                "executed_mask": executed_mask,
                "duration": np.asarray(duration, dtype=np.int64),
                "discounted_reward": np.asarray(discounted_reward, dtype=np.float32),
                "collision": np.asarray(transition_collision, dtype=np.bool_),
                "blocked": np.asarray(transition_blocked, dtype=np.bool_),
                "terminated": np.asarray(terminated_flag, dtype=np.bool_),
                "truncated": np.asarray(truncated_flag, dtype=np.bool_),
                "done": np.asarray(done, dtype=np.bool_),
                "end_reason": np.asarray(
                    _end_reason(
                        done=done,
                        truncated=truncated_flag,
                        stopped=executed.executed_mode == STOP,
                        success=success,
                        collision=transition_collision,
                        budget=budget_override,
                    ),
                    dtype=np.int64,
                ),
                "decision_index": np.asarray(decision_index, dtype=np.int64),
                "physical_step": np.asarray(physical_step - duration, dtype=np.int64),
                "behavior_policy_version": np.asarray(behavior.version, dtype=np.int64),
            }
        )
        rows.append(transition)
        feature_trace.append(
            {
                "decision_index": decision_index,
                "physical_step": physical_step - duration,
                "current": current.feature_evidence(),
                "next": next_state.feature_evidence(),
            }
        )
        decision_index += 1
        current = next_state
        if done:
            break

    arrays = _stack_rows(rows)
    collision_metric = (
        float(collision_cumulative)
        if collision_cumulative is not None
        else float(collision_events)
    )
    metrics = _episode_metrics(
        reset_info=reset_info,
        final_info=final_info,
        initial_distance=initial_distance,
        minimum_distance=minimum_distance,
        initial_path=initial_path,
        step_count=physical_step,
        collision_count=collision_metric,
        blocked_steps=blocked_steps,
        stop_source=stop_source,
        timeout=timed_out,
        action_counts=action_counts,
        gate_counts=requested_counts,
    )
    metrics.update(
        {
            "instruction_sha256": instruction_sha,
            "actor_decision_count": len(rows),
            "requested_mode_distribution": dict(sorted(requested_counts.items())),
            "executed_mode_distribution": dict(sorted(executed_counts.items())),
            "safety_override_count": override_count,
            "behavior": behavior.name,
            "behavior_policy_version": behavior.version,
            "context_source_sha256": source_sha,
        }
    )
    reset_token = _episode_reset_token(
        run_id=run_id, worker_id=worker_id, episode=expected_episode
    )
    metadata = {
        "schema_name": REPLAY_SCHEMA_NAME,
        "schema_version": REPLAY_SCHEMA_VERSION,
        "run_id": run_id,
        "episode_id": actual_episode_id,
        "reset_token": reset_token,
        "scene_id": str(reset_info.get("scene_id")),
        "instruction_sha256": instruction_sha,
        "split_label": "train_update",
        "env_slot": worker_id,
        "row_count": len(rows),
        "lightnav_source_revision": LIGHTNAV_SOURCE_REVISION,
        "lightnav_hook_sha256": None,
        "lightnav_weight_sha256": None,
        "tokenizer_sha256": None,
        "processor_sha256": None,
        "history_config_sha256": None,
        "prefix_feature_schema": PREFIX_FEATURE_SCHEMA,
        "prefix_feature_source_sha256": source_sha,
        "topology_schema": TOPOLOGY_SCHEMA,
        "reference_schema": REFERENCE_SCHEMA,
        "reference_dim": REFERENCE_DIM,
        "action_schema": ACTION_SCHEMA,
        "executor_schema": EXECUTOR_SCHEMA,
        "representation_kind": REPRESENTATION_PREFIX_HIDDEN,
        "feature_stage": FEATURE_STAGE_PREFIX_HIDDEN,
        "topology_max_nodes": topology_max_nodes,
        "topology_node_dim": CAUSAL_NODE_FEATURE_DIM,
        "topology_global_dim": len(CAUSAL_GLOBAL_FEATURE_NAMES),
        "max_primitive_steps": MAX_RECOVERY_PRIMITIVES,
        "discount_gamma": gamma,
        "behavior_policy_id": BEHAVIOR_POLICY_ID,
        "behavior_policy_version": behavior.version,
        "prefix_max_tokens": policy_prefix_max_tokens,
        "prefix_hidden_dim": prefix_hidden_dim,
    }
    return metrics, arrays, metadata, feature_trace, latencies, tuple(artifacts.values())


def _validate_args(args: argparse.Namespace) -> None:
    if args.worker_id < 0 or args.max_steps < 2:
        raise NavigationRLTRolloutError("worker_id must be nonnegative and max_steps at least two")
    if not 1 <= args.jpeg_quality <= 100:
        raise NavigationRLTRolloutError("jpeg_quality must be in [1,100]")
    if (
        args.artifact_prefix_max_tokens < 2
        or args.policy_prefix_max_tokens < 2
        or args.prefix_hidden_dim < 1
        or args.topology_max_nodes < 1
    ):
        raise NavigationRLTRolloutError("feature extents must be positive")
    if args.policy_prefix_max_tokens > args.artifact_prefix_max_tokens:
        raise NavigationRLTRolloutError(
            "policy prefix cap cannot exceed the authenticated artifact cap"
        )
    if args.consume_feature_artifacts and args.feature_artifact_root is None:
        raise NavigationRLTRolloutError(
            "artifact consumption requires --feature-artifact-root"
        )
    if args.feature_artifact_root is not None:
        _resolve_artifact_root(args.feature_artifact_root)
    if not 0.0 < args.gamma <= 1.0 or not math.isfinite(args.gamma):
        raise NavigationRLTRolloutError("gamma must be finite in (0,1]")
    for name in (
        "lightnav_hook_sha256",
        "lightnav_weight_sha256",
        "tokenizer_sha256",
        "processor_sha256",
        "history_config_sha256",
    ):
        _validate_sha256(getattr(args, name), name)
    if args.expected_context_source_sha256 is not None:
        _validate_sha256(
            args.expected_context_source_sha256, "expected context source SHA"
        )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--worker-plan", required=True, type=Path)
    result.add_argument("--expected-worker-plan-sha256", required=True)
    result.add_argument("--worker-id", required=True, type=int)
    result.add_argument("--habitat-server", required=True)
    result.add_argument("--lightnav-server", required=True)
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument("--behavior", choices=BEHAVIOR_NAMES, required=True)
    result.add_argument("--a3-checkpoint", type=Path)
    result.add_argument("--expected-a3-checkpoint-sha256")
    result.add_argument("--rlt-checkpoint", type=Path)
    result.add_argument("--expected-rlt-checkpoint-sha256")
    result.add_argument("--policy-device", default="cpu")
    result.add_argument("--deterministic-policy", action="store_true")
    result.add_argument("--max-steps", type=int, default=160)
    result.add_argument("--jpeg-quality", type=int, default=90)
    result.add_argument("--seed", type=int, default=2026100201)
    result.add_argument("--gamma", type=float, default=0.99)
    result.add_argument("--client-id", default="rlinf-navigation-rlt")
    result.add_argument("--artifact-prefix-max-tokens", type=int, required=True)
    result.add_argument("--policy-prefix-max-tokens", type=int, required=True)
    result.add_argument("--prefix-hidden-dim", type=int, required=True)
    result.add_argument("--topology-max-nodes", type=int, default=128)
    result.add_argument("--expected-context-source-sha256")
    result.add_argument("--lightnav-hook-sha256", required=True)
    result.add_argument("--lightnav-weight-sha256", required=True)
    result.add_argument("--tokenizer-sha256", required=True)
    result.add_argument("--processor-sha256", required=True)
    result.add_argument("--history-config-sha256", required=True)
    result.add_argument("--allow-missing-collisions", action="store_true")
    result.add_argument("--consume-feature-artifacts", action="store_true")
    result.add_argument("--feature-artifact-root", type=Path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    started = time.time()
    output_created = False
    try:
        _validate_args(args)
        plan, worker, plan_sha = _load_worker_plan(
            args.worker_plan,
            expected_sha256=args.expected_worker_plan_sha256,
            worker_id=args.worker_id,
        )
        # The behavior loader uses worker_id when seeding the torch generator.
        behavior = _load_behavior(args)
        feature_artifact_root = (
            _resolve_artifact_root(args.feature_artifact_root)
            if args.feature_artifact_root is not None
            else None
        )
        if args.output_dir.exists():
            raise FileExistsError(args.output_dir)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        output_created = True
        replay_dir = args.output_dir / "replay"
        feature_dir = args.output_dir / "feature_evidence"
        cleanup_dir = args.output_dir / "artifact_cleanup"
        replay_dir.mkdir()
        feature_dir.mkdir()
        if args.consume_feature_artifacts:
            cleanup_dir.mkdir()

        env = ZmqHabitatClient(args.habitat_server)
        model = ContextLightNavClient(args.lightnav_server, timeout_s=600.0)
        model.open()
        try:
            model.login_for_context(f"{args.client_id}-w{args.worker_id}")
            rng = np.random.default_rng(args.seed + args.worker_id)
            episode_receipts: list[dict[str, object]] = []
            replay_receipts: list[dict[str, object]] = []
            feature_receipts: list[dict[str, object]] = []
            cleanup_receipts: list[dict[str, object]] = []
            all_latencies: list[float] = []
            observed_source: str | None = args.expected_context_source_sha256
            for local_index, expected in enumerate(worker["episodes"]):
                if not isinstance(expected, dict):
                    raise NavigationRLTRolloutError("planned episode row must be an object")
                (
                    metrics,
                    arrays,
                    metadata,
                    feature_trace,
                    latencies,
                    episode_artifacts,
                ) = run_episode(
                    env,
                    model,
                    behavior,
                    expected,
                    run_id=str(plan["run_id"]),
                    worker_id=args.worker_id,
                    rng=rng,
                    max_steps=args.max_steps,
                    jpeg_quality=args.jpeg_quality,
                    artifact_prefix_max_tokens=args.artifact_prefix_max_tokens,
                    policy_prefix_max_tokens=args.policy_prefix_max_tokens,
                    prefix_hidden_dim=args.prefix_hidden_dim,
                    topology_max_nodes=args.topology_max_nodes,
                    expected_context_source_sha256=observed_source,
                    feature_artifact_root=feature_artifact_root,
                    require_collisions=not args.allow_missing_collisions,
                    gamma=args.gamma,
                )
                if observed_source is None:
                    observed_source = str(metadata["prefix_feature_source_sha256"])
                metadata.update(
                    {
                        "lightnav_hook_sha256": args.lightnav_hook_sha256,
                        "lightnav_weight_sha256": args.lightnav_weight_sha256,
                        "tokenizer_sha256": args.tokenizer_sha256,
                        "processor_sha256": args.processor_sha256,
                        "history_config_sha256": args.history_config_sha256,
                    }
                )
                safe_episode = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(expected["episode_id"]))
                stem = f"{int(expected['episode_order']):06d}_{safe_episode}"
                replay_receipt = save_episode_shard(
                    replay_dir / f"{stem}.npz", arrays, metadata
                )
                feature_receipt = _write_jsonl(
                    feature_dir / f"{stem}.jsonl", feature_trace
                )
                feature_receipt["episode_id"] = str(expected["episode_id"])
                if args.consume_feature_artifacts:
                    if feature_artifact_root is None:  # guarded by _validate_args
                        raise NavigationRLTRolloutError(
                            "feature artifact root disappeared from worker configuration"
                        )
                    cleanup_receipt = consume_feature_artifacts(
                        episode_artifacts,
                        root=feature_artifact_root,
                        receipt_path=cleanup_dir / f"{stem}.json",
                        replay_receipt=replay_receipt,
                        feature_receipt=feature_receipt,
                    )
                    cleanup_receipt["episode_id"] = str(expected["episode_id"])
                    cleanup_receipts.append(cleanup_receipt)
                metrics.update(
                    {
                        "episode_order": int(expected["episode_order"]),
                        "scene": str(expected["scene"]),
                        "instruction_text_sha256": str(
                            expected["instruction_text_sha256"]
                        ),
                        "reset_token": metadata["reset_token"],
                    }
                )
                episode_receipts.append(metrics)
                replay_receipts.append(replay_receipt)
                feature_receipts.append(feature_receipt)
                all_latencies.extend(latencies)
                atomic_json(
                    args.output_dir / "progress.json",
                    {
                        "schema_name": SCHEMA_NAME,
                        "schema_version": SCHEMA_VERSION,
                        "status": "running",
                        "run_id": plan["run_id"],
                        "worker_id": args.worker_id,
                        "completed_episode_count": local_index + 1,
                        "planned_episode_count": worker["episode_count"],
                    },
                )
        finally:
            model.close()
            env.close()

        aggregate = aggregate_a3(episode_receipts, all_latencies)
        requested_total: Counter[str] = Counter()
        executed_total: Counter[str] = Counter()
        for episode in episode_receipts:
            requested_total.update(episode["requested_mode_distribution"])
            executed_total.update(episode["executed_mode_distribution"])
        aggregate.update(
            {
                "requested_mode_distribution": dict(sorted(requested_total.items())),
                "executed_mode_distribution": dict(sorted(executed_total.items())),
                "safety_override_count": sum(
                    int(row["safety_override_count"]) for row in episode_receipts
                ),
                "context_source_sha256": observed_source,
            }
        )
        completion = {
            "schema_name": WORKER_SCHEMA,
            "schema_version": WORKER_VERSION,
            "status": "completed",
            "run_id": plan["run_id"],
            "worker_id": args.worker_id,
            "plan_sha256": plan_sha,
            "split_label": "train_update",
            "validation_split_reads": 0,
            "behavior_policy_id": BEHAVIOR_POLICY_ID,
            "behavior_policy_version": behavior.version,
            "behavior": behavior.source,
            "deterministic_policy": behavior.deterministic,
            "seed": args.seed,
            "max_physical_steps": args.max_steps,
            "discount_gamma": args.gamma,
            "jpeg_quality": args.jpeg_quality,
            "artifact_prefix_max_tokens": args.artifact_prefix_max_tokens,
            "policy_prefix_max_tokens": args.policy_prefix_max_tokens,
            "prefix_max_tokens": args.policy_prefix_max_tokens,
            "prefix_hidden_dim": args.prefix_hidden_dim,
            "topology_max_nodes": args.topology_max_nodes,
            "artifact_capture_rule": ARTIFACT_CAPTURE_RULE,
            "prefix_selection_schema": POLICY_PREFIX_SELECTION_SCHEMA,
            "prefix_selection_rule": POLICY_PREFIX_SELECTION_RULE,
            "prefix_feature_schema": PREFIX_FEATURE_SCHEMA,
            "prefix_feature_source_sha256": observed_source,
            "consume_feature_artifacts": bool(args.consume_feature_artifacts),
            "feature_artifact_root": (
                str(feature_artifact_root) if feature_artifact_root is not None else None
            ),
            "reference_schema": REFERENCE_SCHEMA,
            "topology_schema": TOPOLOGY_SCHEMA,
            "action_schema": ACTION_SCHEMA,
            "executor_schema": EXECUTOR_SCHEMA,
            "habitat_server": redact_endpoint(args.habitat_server),
            "lightnav_server": redact_endpoint(args.lightnav_server),
            "episodes": episode_receipts,
            "replay_shards": replay_receipts,
            "feature_artifact_traces": feature_receipts,
            "feature_artifact_cleanup": cleanup_receipts,
            "aggregate": aggregate,
            "started_unix_s": started,
            "completed_unix_s": time.time(),
        }
        atomic_json(args.output_dir / "worker_completion.json", completion)
        atomic_json(
            args.output_dir / "COMPLETED.json",
            {
                "schema_name": SCHEMA_NAME,
                "schema_version": SCHEMA_VERSION,
                "status": "completed",
                "worker_completion_sha256": sha256_file(
                    args.output_dir / "worker_completion.json"
                ),
            },
        )
        print(json.dumps(aggregate, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001 - INVALID is the fail-closed boundary
        if output_created:
            atomic_write_json(
                args.output_dir / "INVALID.json",
                {
                    "schema_name": SCHEMA_NAME,
                    "schema_version": SCHEMA_VERSION,
                    "status": "invalid",
                    "worker_id": args.worker_id,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                    "started_unix_s": started,
                    "completed_unix_s": time.time(),
                },
            )
        print(f"Navigation-RLT rollout invalid: {type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
