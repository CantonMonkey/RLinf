"""Original-StreamVLN contextual features for the S2 topology/waypoint policy.

This integration keeps the released StreamVLN model outside the trainable
policy root.  A provider advances the official generation session once per
physical primitive and freezes the resulting Feature-v2 receipt.  The small
policy consumes that receipt together with the existing causal topology and
retains S2's masked categorical plus conditional Gaussian action density.

The generic :mod:`navrl.stack` core is intentionally not involved here.  This
module is an integration-layer adapter around its existing public contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Callable, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from navrl.stack import CheckpointAdapter, PolicySample

try:
    from .streamvln_p8_successor_action import STOP_ACTION
    from .streamvln_p8_successor_policy import (
        SUCCESSOR_FEATURE_V2_DIM,
        SUCCESSOR_FEATURE_V2_SCHEMA_VERSION,
        audit_successor_features_v2,
        compose_successor_features_v2,
        successor_feature_v2_schema_digest,
    )
    from .topology_waypoint_stack import (
        ACTION_SCHEMA,
        MAX_CANDIDATES,
        EpisodeTopologyState,
        TopologyDecisionSnapshot,
        TopologyWaypointPolicy,
        TopologyWaypointPolicyAdapter,
        _tensor_context_digest,
    )
except ImportError:  # Direct staged-file mode used by bounded remote tests.
    from streamvln_p8_successor_action import STOP_ACTION
    from streamvln_p8_successor_policy import (
        SUCCESSOR_FEATURE_V2_DIM,
        SUCCESSOR_FEATURE_V2_SCHEMA_VERSION,
        audit_successor_features_v2,
        compose_successor_features_v2,
        successor_feature_v2_schema_digest,
    )
    from topology_waypoint_stack import (
        ACTION_SCHEMA,
        MAX_CANDIDATES,
        EpisodeTopologyState,
        TopologyDecisionSnapshot,
        TopologyWaypointPolicy,
        TopologyWaypointPolicyAdapter,
        _tensor_context_digest,
    )


CONTEXTUAL_ACTION_SCHEMA = "streamvln_contextual_topology_waypoint_v1"
CONTEXTUAL_CHECKPOINT_KIND = "streamvln_contextual_topology_head_v1"
CONTEXTUAL_FEATURE_DIM = SUCCESSOR_FEATURE_V2_DIM
CONTEXTUAL_FEATURE_SCHEMA_VERSION = SUCCESSOR_FEATURE_V2_SCHEMA_VERSION
CONTEXTUAL_FEATURE_SCHEMA_SHA256 = successor_feature_v2_schema_digest()

_FEATURES = "streamvln_features"
_SCHEMA_SHA256 = "streamvln_feature_schema_sha256"
_FEATURE_SHA256 = "streamvln_feature_sha256"
_CONTEXT_FRAME_SHA256 = "streamvln_context_frame_sha256"
_CURRENT_FRAME_SHA256 = "streamvln_current_frame_sha256"
_DEPTH_SHA256 = "streamvln_current_depth_sha256"
_EPISODE_GENERATION = "streamvln_episode_generation"
_PAST_KEY_VALUES_REF = "streamvln_past_key_values_ref"
_INPUTS_EMBEDS_REF = "streamvln_inputs_embeds_ref"
_RESET_ORDINAL = "streamvln_reset_ordinal"
_GENERATION_ORDINAL = "streamvln_generation_ordinal"
_PREFILL_LENGTH = "streamvln_prefill_length"
_CONTEXT_STEP = "streamvln_context_physical_step"
_CURRENT_STEP = "streamvln_current_physical_step"
_CONTEXT_AGE = "streamvln_context_age"
_SHADOW_ACTION = "streamvln_official_shadow_action"
_FORCE_STOP = "streamvln_force_stop"

CONTEXTUAL_OBSERVATION_KEYS = frozenset(
    {
        _FEATURES,
        _SCHEMA_SHA256,
        _FEATURE_SHA256,
        _CONTEXT_FRAME_SHA256,
        _CURRENT_FRAME_SHA256,
        _DEPTH_SHA256,
        _EPISODE_GENERATION,
        _PAST_KEY_VALUES_REF,
        _INPUTS_EMBEDS_REF,
        _RESET_ORDINAL,
        _GENERATION_ORDINAL,
        _PREFILL_LENGTH,
        _CONTEXT_STEP,
        _CURRENT_STEP,
        _CONTEXT_AGE,
        _SHADOW_ACTION,
        _FORCE_STOP,
    }
)

_PROVENANCE_KEYS = tuple(
    sorted(
        CONTEXTUAL_OBSERVATION_KEYS
        - {
            _FEATURES,
        }
    )
)


class ExternalShadowStopRequired(RuntimeError):
    """Signal that official shadow STOP must bypass the trainable actor.

    A live collector catches this signal, cancels any unexecuted macro suffix,
    executes STOP at the current physical boundary, and acknowledges the
    provider as terminated.  No :class:`PolicySample` is created, so the event
    cannot enter an actor replay batch or acquire categorical log-prob credit.
    """


def _valid_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest_payload(value: str, *, name: str, bytes_length: int = 32) -> torch.Tensor:
    if not isinstance(value, str) or len(value) != bytes_length * 2:
        raise ValueError(f"{name} must be {bytes_length * 2} lowercase hex characters")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be lowercase hexadecimal")
    return torch.tensor([list(bytes.fromhex(value))], dtype=torch.uint8)


def _frame_sha256(frame_ref: str, *, name: str) -> str:
    if not isinstance(frame_ref, str) or not frame_ref.startswith("sha256:"):
        raise ValueError(f"{name} must be a sha256: provenance reference")
    digest = frame_ref.removeprefix("sha256:")
    if not _valid_sha256(digest):
        raise ValueError(f"{name} contains an invalid SHA-256")
    return digest


def _single_int(value: torch.Tensor, *, name: str) -> int:
    if not isinstance(value, torch.Tensor) or value.shape != (1,) or value.dtype != torch.int64:
        raise ValueError(f"{name} must be an int64 tensor with shape [1]")
    return int(value.item())


def _single_bool(value: torch.Tensor, *, name: str) -> bool:
    if not isinstance(value, torch.Tensor) or value.shape != (1,) or value.dtype != torch.bool:
        raise ValueError(f"{name} must be a bool tensor with shape [1]")
    return bool(value.item())


def _digest_from_tensor(value: torch.Tensor, *, name: str, bytes_length: int = 32) -> str:
    if (
        not isinstance(value, torch.Tensor)
        or value.shape != (1, bytes_length)
        or value.dtype != torch.uint8
        or value.device.type != "cpu"
    ):
        raise ValueError(f"{name} must be a CPU uint8 tensor with shape [1,{bytes_length}]")
    return bytes(int(item) for item in value.reshape(-1).tolist()).hex()


def _feature_provenance_digest(forward_inputs: Mapping[str, object]) -> str:
    selected = {}
    for key in _PROVENANCE_KEYS:
        value = forward_inputs.get(key)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"contextual provenance field {key} must be a tensor")
        selected[key] = value
    return _tensor_context_digest(selected)


@dataclass(frozen=True)
class ContextualFeatureObservation:
    """One immutable, replayable original-StreamVLN Feature-v2 receipt."""

    features: torch.Tensor
    schema_sha256: str
    feature_sha256: str
    context_frame_ref: str
    current_frame_ref: str
    current_depth_sha256: str
    episode_generation: str
    past_key_values_ref: str
    inputs_embeds_ref: str
    reset_ordinal: int
    generation_ordinal: int
    prefill_length: int
    context_physical_step: int
    current_physical_step: int
    context_age: int
    official_shadow_action: int
    force_stop: bool

    def __post_init__(self) -> None:
        audit = audit_successor_features_v2(self.features)
        if self.schema_sha256 != CONTEXTUAL_FEATURE_SCHEMA_SHA256:
            raise ValueError("contextual feature schema digest drifted")
        if audit.get("schema_sha256") != self.schema_sha256:
            raise ValueError("contextual feature audit schema differs")
        if audit.get("feature_sha256") != self.feature_sha256:
            raise ValueError("contextual feature SHA-256 differs from its tensor")
        _frame_sha256(self.context_frame_ref, name="context frame_ref")
        _frame_sha256(self.current_frame_ref, name="current frame_ref")
        for value, name in (
            (self.current_depth_sha256, "current depth SHA-256"),
            (self.past_key_values_ref, "past-key-values reference"),
            (self.inputs_embeds_ref, "inputs-embeds reference"),
        ):
            if not _valid_sha256(value):
                raise ValueError(f"{name} is invalid")
        if (
            not isinstance(self.episode_generation, str)
            or len(self.episode_generation) != 32
            or any(character not in "0123456789abcdef" for character in self.episode_generation)
        ):
            raise ValueError("contextual episode generation is invalid")
        for value, name in (
            (self.reset_ordinal, "reset_ordinal"),
            (self.generation_ordinal, "generation_ordinal"),
            (self.context_physical_step, "context_physical_step"),
            (self.current_physical_step, "current_physical_step"),
            (self.context_age, "context_age"),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if type(self.prefill_length) is not int or self.prefill_length < 1:
            raise ValueError("prefill_length must be a positive integer")
        if self.current_physical_step - self.context_physical_step != self.context_age:
            raise ValueError("contextual feature age does not match its physical steps")
        if audit.get("context_age") != self.context_age:
            raise ValueError("composed Feature-v2 age differs from provenance")
        if type(self.official_shadow_action) is not int or not 0 <= self.official_shadow_action < 4:
            raise ValueError("official shadow action must be an integer in 0..3")
        if audit.get("official_shadow_action") != self.official_shadow_action:
            raise ValueError("composed Feature-v2 action differs from provenance")
        if type(self.force_stop) is not bool:
            raise TypeError("force_stop must be boolean")
        if self.force_stop != (self.official_shadow_action == STOP_ACTION):
            raise ValueError("force_stop must exactly follow official shadow STOP")

    @property
    def causal_current(self) -> bool:
        return self.context_age == 0

    def to_policy_observation(self) -> dict[str, torch.Tensor]:
        """Return a detached tensor-only mapping suitable for stack replay."""

        feature = self.features.detach().cpu().to(dtype=torch.float32).contiguous()
        return {
            _FEATURES: feature.unsqueeze(0).clone(),
            _SCHEMA_SHA256: _digest_payload(
                self.schema_sha256, name="feature schema SHA-256"
            ),
            _FEATURE_SHA256: _digest_payload(
                self.feature_sha256, name="feature SHA-256"
            ),
            _CONTEXT_FRAME_SHA256: _digest_payload(
                _frame_sha256(self.context_frame_ref, name="context frame_ref"),
                name="context frame SHA-256",
            ),
            _CURRENT_FRAME_SHA256: _digest_payload(
                _frame_sha256(self.current_frame_ref, name="current frame_ref"),
                name="current frame SHA-256",
            ),
            _DEPTH_SHA256: _digest_payload(
                self.current_depth_sha256, name="current depth SHA-256"
            ),
            _EPISODE_GENERATION: _digest_payload(
                self.episode_generation,
                name="episode generation",
                bytes_length=16,
            ),
            _PAST_KEY_VALUES_REF: _digest_payload(
                self.past_key_values_ref, name="past-key-values reference"
            ),
            _INPUTS_EMBEDS_REF: _digest_payload(
                self.inputs_embeds_ref, name="inputs-embeds reference"
            ),
            _RESET_ORDINAL: torch.tensor([self.reset_ordinal], dtype=torch.int64),
            _GENERATION_ORDINAL: torch.tensor(
                [self.generation_ordinal], dtype=torch.int64
            ),
            _PREFILL_LENGTH: torch.tensor([self.prefill_length], dtype=torch.int64),
            _CONTEXT_STEP: torch.tensor(
                [self.context_physical_step], dtype=torch.int64
            ),
            _CURRENT_STEP: torch.tensor(
                [self.current_physical_step], dtype=torch.int64
            ),
            _CONTEXT_AGE: torch.tensor([self.context_age], dtype=torch.int64),
            _SHADOW_ACTION: torch.tensor(
                [self.official_shadow_action], dtype=torch.int64
            ),
            _FORCE_STOP: torch.tensor([self.force_stop], dtype=torch.bool),
        }


class StreamVLNContextualFeatureProvider:
    """Advance an official frozen StreamVLN session and freeze Feature-v2 rows.

    ``prepare`` must run once before each physical primitive, and
    ``acknowledge`` must run after that primitive.  A generated shadow STOP is
    part of the feature prior and therefore forces the learned action to STOP;
    the provider rejects an acknowledgement that would let execution continue.
    """

    def __init__(
        self,
        generation_adapter: object,
        official_driver: object,
        *,
        depth_summary_builder: Callable[[object], object] | None = None,
        feature_composer: Callable[..., torch.Tensor] | None = None,
        feature_auditor: Callable[[torch.Tensor], Mapping[str, object]] | None = None,
    ) -> None:
        self.generation_adapter = generation_adapter
        self.official_driver = official_driver
        self.depth_summary_builder = depth_summary_builder
        self.feature_composer = feature_composer or compose_successor_features_v2
        self.feature_auditor = feature_auditor or audit_successor_features_v2
        self._pending: ContextualFeatureObservation | None = None
        self._provider_generation = 0
        self._validate_frozen_backbone()
        self.reset()

    def _validate_frozen_backbone(self) -> None:
        model = getattr(self.generation_adapter, "model", None)
        parameters = getattr(model, "parameters", None)
        if not callable(parameters):
            raise TypeError("contextual feature provider requires the frozen model object")
        values = list(parameters())
        if not values:
            raise RuntimeError("contextual feature provider found no backbone parameters")
        if any(getattr(parameter, "requires_grad", None) is not False for parameter in values):
            raise RuntimeError("original StreamVLN backbone must be completely frozen")
        self.backbone_parameter_count = int(
            sum(int(parameter.numel()) for parameter in values)
        )

    @property
    def pending(self) -> ContextualFeatureObservation | None:
        return self._pending

    @property
    def provider_generation(self) -> int:
        return self._provider_generation

    def backbone_manifest(self) -> Mapping[str, object]:
        return {
            "external_backbone": True,
            "backbone_trainable": False,
            "backbone_parameter_count": self.backbone_parameter_count,
            "feature_schema_version": CONTEXTUAL_FEATURE_SCHEMA_VERSION,
            "feature_schema_sha256": CONTEXTUAL_FEATURE_SCHEMA_SHA256,
            "feature_dim": CONTEXTUAL_FEATURE_DIM,
        }

    def reset(self, official_driver: object | None = None) -> None:
        if self._pending is not None:
            raise RuntimeError("cannot reset contextual provider with an unacknowledged frame")
        if official_driver is not None:
            self.official_driver = official_driver
        session = getattr(self.official_driver, "session", None)
        if session is None or getattr(session, "physical_step", None) != 0:
            raise RuntimeError("contextual provider reset requires a fresh step-zero driver")
        reset_ordinal = getattr(self.generation_adapter, "reset_ordinal", None)
        if type(reset_ordinal) is not int or reset_ordinal < 0:
            raise RuntimeError("generation adapter must be reset before provider binding")
        self._provider_generation += 1

    def _depth_summary(self, frame: object):
        if self.depth_summary_builder is None:
            # Keep the Torch-2.6 RLinf/FSDP model process independent of the
            # Torch-2.1 official 8B environment.  Only the live feature
            # provider imports this official-frame helper.
            try:
                from .streamvln_p8_official_adapter import (
                    build_feature_v2_depth_summary,
                )
            except ImportError:  # Direct staged-file live-provider mode.
                from streamvln_p8_official_adapter import (
                    build_feature_v2_depth_summary,
                )
            result = build_feature_v2_depth_summary(
                frame,
                torch_module=getattr(self.generation_adapter, "torch"),
                tensor_auditor=getattr(self.generation_adapter, "tensor_auditor"),
            )
        else:
            result = self.depth_summary_builder(frame)
        if not isinstance(result, tuple) or len(result) != 3:
            raise TypeError(
                "depth summary builder must return (tensor, summary_audit, source_audit)"
            )
        depth, summary_audit, source_audit = result
        if not isinstance(depth, torch.Tensor):
            raise TypeError("depth summary builder returned no tensor")
        if not isinstance(summary_audit, Mapping) or not isinstance(source_audit, Mapping):
            raise TypeError("depth summary audits must be mappings")
        return depth, dict(summary_audit), dict(source_audit)

    def prepare(self, frame: object) -> ContextualFeatureObservation:
        if self._pending is not None:
            raise RuntimeError("contextual frame must be acknowledged before preparing another")
        session = getattr(self.official_driver, "session", None)
        current_step = getattr(frame, "physical_step", None)
        if type(current_step) is not int or current_step < 0:
            raise ValueError("contextual frame physical_step is invalid")
        if session is None or getattr(session, "physical_step", None) != current_step:
            raise RuntimeError("official driver and contextual frame physical steps differ")
        action = self.official_driver.prepare_shadow_action(frame)
        if type(action) is not int or not 0 <= action < 4:
            raise RuntimeError("official driver returned an invalid shadow action")
        snapshot = self.generation_adapter.latest_feature_v2_snapshot()
        if snapshot.reset_ordinal != self.generation_adapter.reset_ordinal:
            raise RuntimeError("contextual snapshot escaped the active reset")
        if snapshot.episode_generation != getattr(frame, "episode_generation", None):
            raise RuntimeError("contextual snapshot escaped the active episode generation")
        if snapshot.physical_step > current_step:
            raise RuntimeError("contextual snapshot comes from a future physical step")
        if self.generation_adapter.generation_ordinal != snapshot.generation_ordinal + 1:
            raise RuntimeError("contextual generation latch ordinal drifted")

        depth, depth_audit, depth_source_audit = self._depth_summary(frame)
        if (
            depth_source_audit.get("frame_ref") != getattr(frame, "frame_ref", None)
            or depth_source_audit.get("depth_sha256")
            != getattr(frame, "depth_sha256", None)
            or depth_source_audit.get("physical_step") != current_step
        ):
            raise RuntimeError("current depth provenance differs from the official frame")
        observed_depth_audit = getattr(self.generation_adapter, "tensor_auditor")(
            depth
        )
        if observed_depth_audit.get("sha256") != depth_audit.get("sha256"):
            raise RuntimeError("current depth tensor differs from its builder audit")

        features = self.feature_composer(
            contextual_hidden=snapshot.contextual_hidden,
            depth_summary=depth,
            official_shadow_action=action,
            physical_step=current_step,
            context_physical_step=snapshot.physical_step,
        )
        if not isinstance(features, torch.Tensor):
            raise TypeError("Feature-v2 composer returned no tensor")
        features = features.detach().cpu().to(dtype=torch.float32).contiguous()
        audit = dict(self.feature_auditor(features))
        age = current_step - snapshot.physical_step
        if (
            audit.get("schema_version") != CONTEXTUAL_FEATURE_SCHEMA_VERSION
            or audit.get("schema_sha256") != CONTEXTUAL_FEATURE_SCHEMA_SHA256
            or audit.get("official_shadow_action") != action
            or audit.get("context_age") != age
            or not _valid_sha256(audit.get("feature_sha256"))
        ):
            raise RuntimeError("Feature-v2 composition audit differs from live provenance")
        observation = ContextualFeatureObservation(
            features=features,
            schema_sha256=CONTEXTUAL_FEATURE_SCHEMA_SHA256,
            feature_sha256=str(audit["feature_sha256"]),
            context_frame_ref=snapshot.frame_ref,
            current_frame_ref=getattr(frame, "frame_ref"),
            current_depth_sha256=getattr(frame, "depth_sha256"),
            episode_generation=snapshot.episode_generation,
            past_key_values_ref=snapshot.past_key_values_ref,
            inputs_embeds_ref=snapshot.inputs_embeds_ref,
            reset_ordinal=snapshot.reset_ordinal,
            generation_ordinal=snapshot.generation_ordinal,
            prefill_length=snapshot.prefill_length,
            context_physical_step=snapshot.physical_step,
            current_physical_step=current_step,
            context_age=age,
            official_shadow_action=action,
            force_stop=action == STOP_ACTION,
        )
        self._pending = observation
        return observation

    def acknowledge(self, *, terminated: bool, truncated: bool) -> Mapping[str, object]:
        if type(terminated) is not bool or type(truncated) is not bool:
            raise TypeError("terminated and truncated must be booleans")
        if terminated and truncated:
            raise ValueError("one primitive cannot both terminate and truncate")
        pending = self._pending
        if pending is None:
            raise RuntimeError("no prepared contextual frame is awaiting acknowledgement")
        if pending.force_stop and not terminated:
            raise RuntimeError("official shadow STOP must execute synchronized STOP")
        event = self.official_driver.acknowledge_physical_step(
            terminated=terminated,
            truncated=truncated,
        )
        if not isinstance(event, Mapping):
            raise TypeError("official driver acknowledgement must return a mapping")
        if event.get("completed_step") != pending.current_physical_step:
            raise RuntimeError("official acknowledgement completed another physical step")
        self._pending = None
        return dict(event)


def _validate_contextual_observation(
    observation: Mapping[str, object],
) -> dict[str, torch.Tensor]:
    if not isinstance(observation, Mapping) or set(observation) != CONTEXTUAL_OBSERVATION_KEYS:
        raise ValueError("contextual policy observation schema drifted")
    values: dict[str, torch.Tensor] = {}
    for key in CONTEXTUAL_OBSERVATION_KEYS:
        value = observation[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"contextual policy field {key} must be a tensor")
        if value.requires_grad:
            raise ValueError(f"contextual policy field {key} must be frozen")
        values[key] = value.detach().cpu().clone().contiguous()

    features = values[_FEATURES]
    if features.shape != (1, CONTEXTUAL_FEATURE_DIM) or features.dtype != torch.float32:
        raise ValueError("contextual Feature-v2 tensor must have shape [1,3604] FP32")
    audit = audit_successor_features_v2(features[0])
    if _digest_from_tensor(values[_SCHEMA_SHA256], name=_SCHEMA_SHA256) != (
        CONTEXTUAL_FEATURE_SCHEMA_SHA256
    ):
        raise ValueError("contextual feature schema tensor drifted")
    if _digest_from_tensor(values[_FEATURE_SHA256], name=_FEATURE_SHA256) != audit.get(
        "feature_sha256"
    ):
        raise ValueError("contextual feature tensor differs from its SHA-256 provenance")
    for key in (
        _CONTEXT_FRAME_SHA256,
        _CURRENT_FRAME_SHA256,
        _DEPTH_SHA256,
        _PAST_KEY_VALUES_REF,
        _INPUTS_EMBEDS_REF,
    ):
        _digest_from_tensor(values[key], name=key)
    _digest_from_tensor(
        values[_EPISODE_GENERATION], name=_EPISODE_GENERATION, bytes_length=16
    )
    reset_ordinal = _single_int(values[_RESET_ORDINAL], name=_RESET_ORDINAL)
    generation_ordinal = _single_int(
        values[_GENERATION_ORDINAL], name=_GENERATION_ORDINAL
    )
    prefill_length = _single_int(values[_PREFILL_LENGTH], name=_PREFILL_LENGTH)
    context_step = _single_int(values[_CONTEXT_STEP], name=_CONTEXT_STEP)
    current_step = _single_int(values[_CURRENT_STEP], name=_CURRENT_STEP)
    age = _single_int(values[_CONTEXT_AGE], name=_CONTEXT_AGE)
    action = _single_int(values[_SHADOW_ACTION], name=_SHADOW_ACTION)
    force_stop = _single_bool(values[_FORCE_STOP], name=_FORCE_STOP)
    if min(reset_ordinal, generation_ordinal, context_step, current_step, age) < 0:
        raise ValueError("contextual provenance ordinals must be non-negative")
    if prefill_length < 1:
        raise ValueError("contextual prefill length must be positive")
    if current_step - context_step != age or audit.get("context_age") != age:
        raise ValueError("contextual tensor age differs from its feature receipt")
    if not 0 <= action < 4 or audit.get("official_shadow_action") != action:
        raise ValueError("contextual shadow action differs from its feature receipt")
    if force_stop != (action == STOP_ACTION):
        raise ValueError("contextual force-stop flag differs from shadow action")
    return values


class _ContextFeatureProjection(nn.Module):
    """Small trainable projection over detached 3604-wide frozen receipts."""

    output_dim = 128

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, self.output_dim),
            nn.Tanh(),
        )

    def forward(self, obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        # The parent policy accepts a fixed three-key encoder interface.  These
        # keys are constructed only inside `_distribution`; replay itself keeps
        # explicit StreamVLN field names.
        if set(obs) != {"rgb", "instruction_ids", "instruction_mask"}:
            raise ValueError("context projection shim keys drifted")
        features = obs["rgb"]
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError("context projection feature shape drifted")
        if features.dtype != torch.float32:
            raise TypeError("context projection requires FP32 features")
        return self.projection(features)


class StreamVLNContextualTopologyWaypointPolicy(TopologyWaypointPolicy):
    """S2 hybrid head conditioned on frozen official StreamVLN Feature-v2."""

    def __init__(
        self,
        *,
        release_manifest_sha256: str,
        hidden_dim: int = 128,
        topology_max_nodes: int = 16,
        topology_model_dim: int = 32,
        max_candidates: int = MAX_CANDIDATES,
        residual_limit_m: float = 0.2,
        initial_log_std: float = -0.5,
        feature_dim: int = CONTEXTUAL_FEATURE_DIM,
    ) -> None:
        if not _valid_sha256(release_manifest_sha256):
            raise ValueError("released StreamVLN manifest SHA-256 is invalid")
        if int(feature_dim) != CONTEXTUAL_FEATURE_DIM:
            raise ValueError("contextual policy requires the frozen 3604-wide schema")
        # The parent builds the already-accepted graph/action heads.  Its small
        # toy observation encoder is immediately removed and never enters the
        # state dict or optimizer.
        super().__init__(
            vocab_size=2,
            max_instruction_tokens=1,
            image_size=16,
            hidden_dim=hidden_dim,
            topology_max_nodes=topology_max_nodes,
            topology_model_dim=topology_model_dim,
            max_candidates=max_candidates,
            residual_limit_m=residual_limit_m,
            initial_log_std=initial_log_std,
        )
        del self.observation_encoder
        self.observation_encoder = _ContextFeatureProjection(int(feature_dim))
        self.feature_dim = int(feature_dim)
        self.release_manifest_sha256 = release_manifest_sha256
        self.float()

    def prepare_forward_inputs(
        self,
        observation: Mapping[str, object],
        snapshot: TopologyDecisionSnapshot,
        *,
        policy_version: int,
    ) -> dict[str, torch.Tensor]:
        contextual = _validate_contextual_observation(observation)
        dummy = {
            "rgb": torch.zeros(1, 16, 16, 3, dtype=torch.uint8),
            "instruction_ids": torch.zeros(1, 1, dtype=torch.int64),
            "instruction_mask": torch.ones(1, 1, dtype=torch.bool),
        }
        result = super().prepare_forward_inputs(
            dummy, snapshot, policy_version=policy_version
        )
        for key in tuple(dummy):
            del result[key]
        result.update(contextual)
        return result

    def _distribution(self, forward_inputs: Mapping[str, object]):
        expected_context = CONTEXTUAL_OBSERVATION_KEYS
        actual_context = set(forward_inputs) & CONTEXTUAL_OBSERVATION_KEYS
        if actual_context != expected_context:
            missing = sorted(expected_context - actual_context)
            raise ValueError(f"contextual forward inputs are incomplete: {missing}")
        contextual = _validate_contextual_observation(
            {key: forward_inputs[key] for key in CONTEXTUAL_OBSERVATION_KEYS}
        )
        force_stop = bool(contextual[_FORCE_STOP].item())
        if force_stop:
            raise ExternalShadowStopRequired(
                "official shadow STOP must bypass actor forward and replay credit"
            )

        parent_inputs = {
            key: value
            for key, value in forward_inputs.items()
            if key not in CONTEXTUAL_OBSERVATION_KEYS
        }
        batch = contextual[_FEATURES].shape[0]
        parent_inputs.update(
            {
                # Internal shim only; serialized replay uses explicit contextual keys.
                "rgb": contextual[_FEATURES],
                "instruction_ids": torch.zeros(batch, 1, dtype=torch.int64),
                "instruction_mask": torch.ones(batch, 1, dtype=torch.bool),
            }
        )
        return super()._distribution(parent_inputs)

    def bc_loss(
        self,
        forward_inputs: Mapping[str, object],
        target_candidate: torch.Tensor,
        target_waypoint: torch.Tensor,
        *,
        waypoint_coef: float = 2.0,
    ) -> Mapping[str, torch.Tensor]:
        if not isinstance(waypoint_coef, (int, float)) or float(waypoint_coef) < 0.0:
            raise ValueError("waypoint_coef must be a non-negative scalar")
        logits, means, log_std, _, continuous_mask, anchors = self._distribution(
            forward_inputs
        )
        batch = logits.shape[0]
        if not isinstance(target_candidate, torch.Tensor) or target_candidate.shape != (
            batch,
        ):
            raise ValueError("contextual BC candidate target shape drifted")
        if not isinstance(target_waypoint, torch.Tensor) or target_waypoint.shape != (
            batch,
            2,
        ):
            raise ValueError("contextual BC target batch shape drifted")
        targets = target_candidate.to(device=logits.device, dtype=torch.int64)
        waypoint_targets = target_waypoint.to(
            device=logits.device, dtype=torch.float32
        )
        if not bool(torch.isfinite(waypoint_targets).all()):
            raise ValueError("contextual BC waypoint targets must be finite")
        if bool(((targets < 0) | (targets >= self.max_candidates)).any()):
            raise ValueError("contextual BC candidate target is outside the fixed extent")
        rows = torch.arange(logits.shape[0], device=logits.device)
        if not bool(torch.isfinite(logits[rows, targets]).all()):
            raise ValueError("contextual BC selected a masked candidate")
        route_loss = F.cross_entropy(logits, targets)
        selected_mean = means[rows, targets]
        selected_log_std = log_std[rows, targets]
        selected_anchor = anchors[rows, targets]
        predicted_waypoint = selected_anchor + self.residual_limit_m * torch.tanh(
            selected_mean
        )
        active = continuous_mask[rows, targets]
        if bool(active.any()):
            # The PPO density lives in unbounded raw residual space.  Convert
            # metric BC targets through the exact inverse of the bounded tanh
            # transform so BC initializes both mean and diagonal log std in
            # the same action space later used by replay/PPO.
            normalized_residual = (
                waypoint_targets[active] - selected_anchor[active]
            ) / self.residual_limit_m
            epsilon = torch.finfo(normalized_residual.dtype).eps
            bounded_residual = normalized_residual.clamp(
                min=-1.0 + epsilon, max=1.0 - epsilon
            )
            raw_targets = torch.atanh(bounded_residual)
            if not bool(torch.isfinite(raw_targets).all()):
                raise ValueError("contextual BC raw residual targets must be finite")
            gaussian_nll = -Normal(
                selected_mean[active], selected_log_std[active].exp()
            ).log_prob(raw_targets).sum(dim=-1).mean()
            waypoint_loss = F.smooth_l1_loss(
                predicted_waypoint[active], waypoint_targets[active]
            )
        else:
            # Preserve a differentiable zero for the waypoint mean branch;
            # residual_log_std remains entirely absent from STOP supervision.
            gaussian_nll = selected_mean.sum() * 0.0
            waypoint_loss = predicted_waypoint.sum() * 0.0
            raw_targets = selected_mean.new_empty((0, 2))
        loss = route_loss + float(waypoint_coef) * (
            gaussian_nll + waypoint_loss
        )
        return {
            "loss": loss,
            "route_loss": route_loss,
            "gaussian_nll": gaussian_nll,
            "waypoint_loss": waypoint_loss,
            "route_accuracy": (logits.argmax(dim=-1) == targets).float().mean(),
            "predicted_waypoint": predicted_waypoint,
            "raw_waypoint_target": raw_targets,
        }

    def forward(self, *, operation: str, **kwargs):
        if operation == "bc_loss":
            return self.bc_loss(
                kwargs["forward_inputs"],
                kwargs["target_candidate"],
                kwargs["target_waypoint"],
                waypoint_coef=float(kwargs.get("waypoint_coef", 2.0)),
            )
        return super().forward(operation=operation, **kwargs)


class StreamVLNContextualTopologyWaypointPolicyAdapter(TopologyWaypointPolicyAdapter):
    """S2 adapter with provenance-bound replay and synchronized shadow STOP."""

    def __init__(
        self,
        policy: StreamVLNContextualTopologyWaypointPolicy,
        topology_state: EpisodeTopologyState,
        *,
        mode: str = "train",
        deterministic: bool | None = None,
        policy_version: int = 0,
        root_forward: Callable[..., object] | None = None,
    ) -> None:
        if not isinstance(policy, StreamVLNContextualTopologyWaypointPolicy):
            raise TypeError("contextual adapter requires its contextual hybrid policy")
        super().__init__(
            policy,
            topology_state,
            mode=mode,
            deterministic=deterministic,
            policy_version=policy_version,
            root_forward=root_forward,
        )

    @staticmethod
    def _force_stop(observation: object) -> bool:
        if not isinstance(observation, Mapping):
            raise ValueError("contextual observation is not a mapping")
        validated = _validate_contextual_observation(observation)
        return _single_bool(validated[_FORCE_STOP], name=_FORCE_STOP)

    def _sample_from_snapshot(
        self,
        observation: Mapping[str, object],
        snapshot: TopologyDecisionSnapshot,
    ) -> PolicySample:
        forward_inputs = self.policy.prepare_forward_inputs(
            observation, snapshot, policy_version=self.policy_version
        )
        output = self._forward(
            "sample",
            forward_inputs=forward_inputs,
            deterministic=self.deterministic,
        )
        slot = int(output["candidate"].reshape(-1)[0].item())
        candidate = snapshot.candidates[slot]
        raw = tuple(float(value) for value in output["raw"].reshape(-1).tolist())
        waypoint = tuple(
            float(value) for value in output["waypoint"].reshape(-1).tolist()
        )
        terms = (output["route_logprob"].reshape(-1)[0],)
        if candidate.continuous:
            terms += (output["raw_logprob"].reshape(-1)[0],)
        replay_context = {
            "adapter_kind": ACTION_SCHEMA,
            "generation": self._generation,
            "candidate_digest": snapshot.digest,
            "candidates": tuple(row.canonical() for row in snapshot.candidates),
            "candidate_ids": tuple(row.candidate_id for row in snapshot.candidates),
            "candidate_kinds": tuple(row.kind for row in snapshot.candidates),
            "candidate_mask": tuple(row.valid for row in snapshot.candidates),
            "candidate_continuous_mask": tuple(
                row.continuous for row in snapshot.candidates
            ),
            "candidate_anchors_m": tuple(
                (row.forward_m, row.lateral_m) for row in snapshot.candidates
            ),
            "forward_inputs": forward_inputs,
            "forward_inputs_digest": _tensor_context_digest(forward_inputs),
            "streamvln_feature_provenance_digest": _feature_provenance_digest(
                forward_inputs
            ),
        }
        return PolicySample(
            identity=snapshot.identity,
            decision_index=snapshot.decision_index,
            physical_step=snapshot.physical_step,
            action_payload=(
                ACTION_SCHEMA,
                snapshot.digest,
                slot,
                candidate.candidate_id,
                raw[0],
                raw[1],
                waypoint[0],
                waypoint[1],
            ),
            replay_context=replay_context,
            behavior_log_prob_terms=terms,
            value=output["value"].reshape(-1)[0],
            policy_version=self.policy_version,
            action_distribution=self.action_distribution,
        )

    def sample(
        self,
        observation: object,
        identity,
        decision_index: int,
        physical_step: int,
    ) -> PolicySample:
        self._require_identity(identity)
        if self._force_stop(observation):
            raise ExternalShadowStopRequired(
                "official shadow STOP must execute outside the actor at this physical boundary"
            )
        snapshot = self.topology_state.snapshot(identity, decision_index, physical_step)
        return self._sample_from_snapshot(observation, snapshot)

    def _validate_bound_sample(self, sample: PolicySample):
        context = sample.replay_context
        forward_inputs = context.get("forward_inputs")
        if not isinstance(forward_inputs, Mapping):
            raise ValueError("contextual replay has no forward inputs")
        force_stop = forward_inputs.get(_FORCE_STOP)
        if not isinstance(force_stop, torch.Tensor):
            raise ValueError("contextual replay has no force-stop provenance")
        if _single_bool(force_stop.detach().cpu(), name=_FORCE_STOP):
            raise ValueError("external shadow STOP may not enter actor replay")
        expected = _feature_provenance_digest(forward_inputs)
        if context.get("streamvln_feature_provenance_digest") != expected:
            raise ValueError("contextual replay feature provenance was tampered")
        return super()._validate_bound_sample(sample)

    def bootstrap(
        self,
        observation: object,
        identity,
        physical_step: int,
    ):
        self._require_identity(identity)
        if self._force_stop(observation):
            raise ExternalShadowStopRequired(
                "official shadow STOP is terminal and may not be bootstrapped"
            )
        return super().bootstrap(observation, identity, physical_step)

    def trainable_manifest(self) -> Mapping[str, object]:
        manifest = dict(super().trainable_manifest())
        manifest.update(
            {
                "source_kind": "original_streamvln_frozen_feature_v2",
                "feature_dim": self.policy.feature_dim,
                "feature_schema_version": CONTEXTUAL_FEATURE_SCHEMA_VERSION,
                "feature_schema_sha256": CONTEXTUAL_FEATURE_SCHEMA_SHA256,
                "release_manifest_sha256": self.policy.release_manifest_sha256,
                "external_backbone": True,
                "backbone_trainable": False,
                "shadow_stop_semantics": "only_stop_candidate_and_synchronized_execution",
            }
        )
        return manifest


class StreamVLNContextualHeadCheckpointAdapter(CheckpointAdapter):
    """Head-only codec bound to the frozen release and Feature-v2 semantics."""

    filename = "streamvln_contextual_topology_head.pt"

    def __init__(self, adapter: StreamVLNContextualTopologyWaypointPolicyAdapter):
        if not isinstance(adapter, StreamVLNContextualTopologyWaypointPolicyAdapter):
            raise TypeError("contextual checkpoint requires its policy adapter")
        self.adapter = adapter

    @staticmethod
    def _state_schema(state: Mapping[str, object]) -> Mapping[str, object]:
        return {
            str(name): {
                "shape": [int(item) for item in value.shape],
                "dtype": str(value.dtype),
            }
            for name, value in sorted(state.items())
        }

    def checkpoint_compatibility(self) -> Mapping[str, object]:
        policy = self.adapter.policy
        return {
            "checkpoint_kind": CONTEXTUAL_CHECKPOINT_KIND,
            "external_backbone_in_checkpoint": False,
            "release_manifest_sha256": policy.release_manifest_sha256,
            "feature_schema_version": CONTEXTUAL_FEATURE_SCHEMA_VERSION,
            "feature_schema_sha256": CONTEXTUAL_FEATURE_SCHEMA_SHA256,
            "feature_dim": policy.feature_dim,
            "policy_semantics": {
                "policy_class": type(policy).__qualname__,
                "hidden_dim": policy.hidden_dim,
                "topology_max_nodes": policy.topology_max_nodes,
                "topology_model_dim": policy.topology_model_dim,
                "max_candidates": policy.max_candidates,
                "residual_limit_m": policy.residual_limit_m,
                "action_distribution": self.adapter.action_distribution,
                "shadow_stop_semantics": "only_stop_candidate_and_synchronized_execution",
            },
            "topology_action_contract": self.adapter.topology_state.compatibility(),
            "state_schema": self._state_schema(policy.state_dict()),
            "trainable_names": list(self.adapter.trainable_parameter_names()),
        }

    def checkpoint_metadata(self) -> Mapping[str, object]:
        return {
            "checkpoint_kind": CONTEXTUAL_CHECKPOINT_KIND,
            "policy_version": self.adapter.policy_version,
            "trainable_scope": "context_projection_topology_route_waypoint_value_heads",
        }

    def save_checkpoint(self, staging_directory: Path) -> Mapping[str, str]:
        path = Path(staging_directory) / self.filename
        torch.save(
            {
                "state_dict": self.adapter.policy.state_dict(),
                "policy_version": self.adapter.policy_version,
            },
            path,
        )
        return {path.name: _sha256_file(path)}

    def _read_payload(self, directory: Path) -> Mapping[str, object]:
        path = Path(directory) / self.filename
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, Mapping) or set(payload) != {
            "state_dict",
            "policy_version",
        }:
            raise ValueError("contextual head checkpoint payload is incomplete")
        if not isinstance(payload["state_dict"], Mapping):
            raise ValueError("contextual head checkpoint state_dict is invalid")
        version = payload["policy_version"]
        if type(version) is not int or version < 0:
            raise ValueError("contextual head checkpoint policy_version is invalid")
        return payload

    def inspect_checkpoint_metadata(self, directory: Path) -> Mapping[str, object]:
        payload = self._read_payload(directory)
        return {
            "checkpoint_kind": CONTEXTUAL_CHECKPOINT_KIND,
            "policy_version": payload["policy_version"],
            "trainable_scope": "context_projection_topology_route_waypoint_value_heads",
        }

    def load_checkpoint(self, directory: Path) -> None:
        payload = self._read_payload(directory)
        saved_state = payload["state_dict"]
        if self._state_schema(saved_state) != self._state_schema(
            self.adapter.policy.state_dict()
        ):
            raise RuntimeError("contextual head checkpoint state schema differs")
        self.adapter.policy.load_state_dict(saved_state, strict=True)
        self.adapter._policy_version = int(payload["policy_version"])
        if self.adapter._identity is not None:
            self.adapter.reset_state(self.adapter._identity)


__all__ = [
    "CONTEXTUAL_ACTION_SCHEMA",
    "CONTEXTUAL_CHECKPOINT_KIND",
    "CONTEXTUAL_FEATURE_DIM",
    "CONTEXTUAL_FEATURE_SCHEMA_SHA256",
    "CONTEXTUAL_FEATURE_SCHEMA_VERSION",
    "CONTEXTUAL_OBSERVATION_KEYS",
    "ContextualFeatureObservation",
    "ExternalShadowStopRequired",
    "StreamVLNContextualFeatureProvider",
    "StreamVLNContextualHeadCheckpointAdapter",
    "StreamVLNContextualTopologyWaypointPolicy",
    "StreamVLNContextualTopologyWaypointPolicyAdapter",
]
