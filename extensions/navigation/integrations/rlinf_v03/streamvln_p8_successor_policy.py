"""Shared route/waypoint policy for the production P8 successor path.

The rollout and actor workers use the same module and the same versioned,
tensor-only context.  The policy action space is exactly KEEP_ORIGINAL,
WAYPOINT, and STOP; deterministic recovery remains below the policy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import struct
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

try:
    from rlinf.models.embodiment.base_policy import BasePolicy
except ModuleNotFoundError:  # Lightweight local contract tests.
    class BasePolicy:  # type: ignore[no-redef]
        pass

try:
    from .streamvln_p8_successor_action import (
        STOP_ACTION,
        RouteDecision,
        RouteName,
        compile_metric_waypoint,
    )
    from .streamvln_p8_successor_protocol import (
        SUCCESSOR_REPLAY_SCHEMA,
        SuccessorDecisionSample,
    )
    from .streamvln_p8_topology import (
        CAUSAL_EDGE_FEATURE_DIM,
        CAUSAL_GLOBAL_FEATURE_DIM,
        CAUSAL_NODE_FEATURE_DIM,
        CAUSAL_NODE_HIDDEN_DIM,
        P8CausalTopologySnapshot,
        P8TopologyEncoder,
        P8TopologyTensors,
    )
except ImportError:  # Direct staged-file tests.
    from streamvln_p8_successor_action import (
        STOP_ACTION,
        RouteDecision,
        RouteName,
        compile_metric_waypoint,
    )
    from streamvln_p8_successor_protocol import (
        SUCCESSOR_REPLAY_SCHEMA,
        SuccessorDecisionSample,
    )
    from streamvln_p8_topology import (
        CAUSAL_EDGE_FEATURE_DIM,
        CAUSAL_GLOBAL_FEATURE_DIM,
        CAUSAL_NODE_FEATURE_DIM,
        CAUSAL_NODE_HIDDEN_DIM,
        P8CausalTopologySnapshot,
        P8TopologyEncoder,
        P8TopologyTensors,
    )


SUCCESSOR_POLICY_MODEL_TYPE = "streamvln_p8_successor"
SUCCESSOR_POLICY_CONTEXT_VERSION = 4
SUCCESSOR_ACTION_FEATURE_DIM = 4
SUCCESSOR_FEATURE_AGE_DIM = 1
SUCCESSOR_COMPOSED_AUX_DIM = SUCCESSOR_ACTION_FEATURE_DIM + SUCCESSOR_FEATURE_AGE_DIM
SUCCESSOR_FEATURE_V1_SCHEMA_VERSION = 1
SUCCESSOR_FEATURE_V1_FROZEN_DIM = 7_168
SUCCESSOR_FEATURE_V1_DIM = SUCCESSOR_FEATURE_V1_FROZEN_DIM + SUCCESSOR_COMPOSED_AUX_DIM
SUCCESSOR_FEATURE_V2_SCHEMA_VERSION = 2
SUCCESSOR_FEATURE_V2_CONTEXT_DIM = 3_584
SUCCESSOR_FEATURE_V2_DEPTH_DIM = 15
SUCCESSOR_FEATURE_V2_DIM = (
    SUCCESSOR_FEATURE_V2_CONTEXT_DIM
    + SUCCESSOR_FEATURE_V2_DEPTH_DIM
    + SUCCESSOR_COMPOSED_AUX_DIM
)
SUCCESSOR_FEATURE_V2_DEPTH_SECTORS = 5
SUCCESSOR_FEATURE_V2_DEPTH_MAX_M = 10.0
SUCCESSOR_FEATURE_MAX_AGE = 1_000_000
SUCCESSOR_FEATURE_CHECKPOINT_BINDING_KEYS = (
    "successor_feature_schema_version",
    "successor_feature_schema_sha256",
)
SUCCESSOR_FEATURE_V2_WAYPOINT_MIN_M = 0.1001
SUCCESSOR_FEATURE_V2_WAYPOINT_MAX_M = 1.25
SUCCESSOR_FEATURE_V2_WAYPOINT_MIN_PROBABILITY = 0.80
SUCCESSOR_POLICY_CONTEXT_KEYS = (
    "schema_version",
    "policy_version",
    "features",
    "route_candidate_mask",
    "expert_route_candidate_mask",
    "topology_global_features",
    "topology_node_hidden",
    "topology_node_features",
    "topology_node_mask",
    "topology_adjacency",
    "topology_edge_features",
    "topology_current_index",
    "topology_parent_index",
)
POLICY_ROUTES = (
    RouteName.KEEP_ORIGINAL,
    RouteName.WAYPOINT,
    RouteName.STOP,
)
ROUTE_TO_INDEX = {route: index for index, route in enumerate(POLICY_ROUTES)}


def _feature_component(*, name: str, start: int, stop: int) -> dict[str, object]:
    return {"name": name, "start": start, "stop": stop, "dim": stop - start}


def _successor_feature_schema(schema_version: int) -> dict[str, object]:
    if type(schema_version) is not int:
        raise TypeError("successor feature schema version must be an integer")
    if schema_version == SUCCESSOR_FEATURE_V1_SCHEMA_VERSION:
        action_start = SUCCESSOR_FEATURE_V1_FROZEN_DIM
        age_start = action_start + SUCCESSOR_ACTION_FEATURE_DIM
        return {
            "name": "streamvln_p8_successor_feature_v1",
            "version": SUCCESSOR_FEATURE_V1_SCHEMA_VERSION,
            "dtype": "torch.float32",
            "rank": 1,
            "total_dim": SUCCESSOR_FEATURE_V1_DIM,
            "components": [
                _feature_component(
                    name="frozen_feature",
                    start=0,
                    stop=SUCCESSOR_FEATURE_V1_FROZEN_DIM,
                ),
                _feature_component(
                    name="official_action_onehot",
                    start=action_start,
                    stop=age_start,
                ),
                _feature_component(
                    name="context_age",
                    start=age_start,
                    stop=SUCCESSOR_FEATURE_V1_DIM,
                ),
            ],
            "composer": "compose_successor_features",
        }
    if schema_version == SUCCESSOR_FEATURE_V2_SCHEMA_VERSION:
        depth_start = SUCCESSOR_FEATURE_V2_CONTEXT_DIM
        action_start = depth_start + SUCCESSOR_FEATURE_V2_DEPTH_DIM
        age_start = action_start + SUCCESSOR_ACTION_FEATURE_DIM
        return {
            "name": "streamvln_p8_successor_feature_v2",
            "version": SUCCESSOR_FEATURE_V2_SCHEMA_VERSION,
            "dtype": "torch.float32",
            "device": "cpu",
            "contiguous": True,
            "requires_grad": False,
            "rank": 1,
            "total_dim": SUCCESSOR_FEATURE_V2_DIM,
            "components": [
                _feature_component(
                    name="contextual_hidden",
                    start=0,
                    stop=SUCCESSOR_FEATURE_V2_CONTEXT_DIM,
                ),
                _feature_component(
                    name="depth_summary",
                    start=depth_start,
                    stop=action_start,
                ),
                _feature_component(
                    name="official_action_onehot",
                    start=action_start,
                    stop=age_start,
                ),
                _feature_component(
                    name="context_age",
                    start=age_start,
                    stop=SUCCESSOR_FEATURE_V2_DIM,
                ),
            ],
            "depth_layout": {
                "sector_count": SUCCESSOR_FEATURE_V2_DEPTH_SECTORS,
                "sector_order": "left_to_right",
                "values_per_sector": ["q10_m", "q50_m", "valid_fraction"],
                "max_depth_m": SUCCESSOR_FEATURE_V2_DEPTH_MAX_M,
                "empty_sector": [0.0, 0.0, 0.0],
            },
            "context_age": {
                "minimum": 0,
                "maximum": SUCCESSOR_FEATURE_MAX_AGE,
                "gate_requires": 0,
            },
            "causal_binding": {
                "age_expression": "current_physical_step - snapshot_physical_step",
                "component_sources": {
                    "contextual_hidden": "generation_latched_contextual_hidden",
                    "depth_summary": "current_decision_official_frame_metric_depth",
                },
                "same_frame_when": "context_age == 0",
                "runner_audit_requirements": [
                    "reset_ordinal_matches_active_session",
                    "contextual_hidden_generation_ordinal_matches_latch",
                    "depth_frame_ref_matches_current_official_frame",
                    "depth_physical_step_matches_current_frame",
                    "depth_sha256_matches_depth_source_audit",
                ],
            },
            "audit": {
                "required_fields": [
                    "schema_version",
                    "schema_sha256",
                    "feature_sha256",
                    "component_sha256",
                    "shape",
                    "dtype",
                    "device",
                    "contiguous",
                    "finite",
                    "official_shadow_action",
                    "context_age",
                    "causal_current",
                ],
                "digest_encoding": "little_endian_float32",
            },
            "composer": "compose_successor_features_v2",
        }
    raise ValueError(f"unsupported successor feature schema version {schema_version}")


def successor_feature_v2_schema() -> dict[str, object]:
    """Return a detached JSON-safe copy of the frozen feature-v2 schema."""

    return json.loads(
        json.dumps(
            _successor_feature_schema(SUCCESSOR_FEATURE_V2_SCHEMA_VERSION),
            sort_keys=True,
        )
    )


def successor_feature_schema_digest(schema_version: int) -> str:
    """Hash one canonical schema so external checkpoints can bind to it."""

    payload = json.dumps(
        _successor_feature_schema(schema_version),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def successor_feature_v2_schema_digest() -> str:
    return successor_feature_schema_digest(SUCCESSOR_FEATURE_V2_SCHEMA_VERSION)


def successor_feature_checkpoint_binding(schema_version: int) -> dict[str, object]:
    """Build the metadata fragment required beside a new successor checkpoint."""

    return {
        SUCCESSOR_FEATURE_CHECKPOINT_BINDING_KEYS[0]: schema_version,
        SUCCESSOR_FEATURE_CHECKPOINT_BINDING_KEYS[1]: successor_feature_schema_digest(
            schema_version
        ),
    }


def require_successor_feature_checkpoint_binding(
    binding: Mapping[str, object], *, expected_schema_version: int
) -> dict[str, object]:
    """Reject absent, malformed, tampered, or cross-schema checkpoint metadata."""

    if not isinstance(binding, Mapping):
        raise TypeError("successor feature checkpoint binding must be a mapping")
    if len(binding) != len(SUCCESSOR_FEATURE_CHECKPOINT_BINDING_KEYS) or set(
        binding
    ) != set(SUCCESSOR_FEATURE_CHECKPOINT_BINDING_KEYS):
        raise ValueError("successor feature checkpoint binding schema drifted")
    actual_version = binding[SUCCESSOR_FEATURE_CHECKPOINT_BINDING_KEYS[0]]
    actual_digest = binding[SUCCESSOR_FEATURE_CHECKPOINT_BINDING_KEYS[1]]
    if type(actual_version) is not int:
        raise TypeError("successor checkpoint feature schema version must be an integer")
    if not isinstance(actual_digest, str) or len(actual_digest) != 64 or any(
        character not in "0123456789abcdef" for character in actual_digest
    ):
        raise ValueError("successor checkpoint feature schema digest must be lowercase SHA-256")
    expected_digest = successor_feature_schema_digest(expected_schema_version)
    if actual_version != expected_schema_version or actual_digest != expected_digest:
        raise RuntimeError("successor checkpoint feature schema binding mismatch")
    return dict(binding)


def _require_feature_v2_vector(
    value: torch.Tensor, *, name: str, dimension: int
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"feature-v2 {name} must be a torch tensor")
    if tuple(value.shape) != (dimension,):
        raise ValueError(f"feature-v2 {name} must have shape [{dimension}]")
    if value.dtype != torch.float32:
        raise TypeError(f"feature-v2 {name} must use torch.float32")
    if value.device.type != "cpu":
        raise ValueError(f"feature-v2 {name} must be on CPU")
    if not value.is_contiguous():
        raise ValueError(f"feature-v2 {name} must be contiguous")
    if value.requires_grad:
        raise ValueError(f"feature-v2 {name} must be frozen")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"feature-v2 {name} must be finite")
    return value.detach().clone()


def _validate_feature_v2_depth_summary(depth_summary: torch.Tensor) -> None:
    sectors = depth_summary.reshape(SUCCESSOR_FEATURE_V2_DEPTH_SECTORS, 3)
    q10_m = sectors[:, 0]
    q50_m = sectors[:, 1]
    valid_fraction = sectors[:, 2]
    if bool((q10_m < 0.0).any()) or bool((q50_m < q10_m).any()):
        raise ValueError("feature-v2 depth quantiles are invalid")
    if bool((q50_m > SUCCESSOR_FEATURE_V2_DEPTH_MAX_M).any()):
        raise ValueError("feature-v2 depth quantiles exceed the metric range")
    if bool(((valid_fraction < 0.0) | (valid_fraction > 1.0)).any()):
        raise ValueError("feature-v2 depth valid fractions must be in [0, 1]")
    empty = valid_fraction == 0.0
    if bool(((q10_m != 0.0) | (q50_m != 0.0))[empty].any()):
        raise ValueError("feature-v2 empty depth sectors must be all zero")
    nonempty = ~empty
    if bool(((q10_m <= 0.0) | (q50_m <= 0.0))[nonempty].any()):
        raise ValueError("feature-v2 nonempty depth sectors require positive quantiles")


def _float32_tensor_sha256(value: torch.Tensor) -> str:
    values = tuple(float(item) for item in value.detach().cpu().reshape(-1).tolist())
    payload = struct.pack(f"<{len(values)}f", *values)
    return hashlib.sha256(payload).hexdigest()


def audit_successor_features_v2(features: torch.Tensor) -> dict[str, object]:
    """Validate and summarize a complete feature-v2 causal receipt."""

    checked = _require_feature_v2_vector(
        features, name="composed feature", dimension=SUCCESSOR_FEATURE_V2_DIM
    )
    schema = successor_feature_v2_schema()
    components = {
        component["name"]: checked[component["start"] : component["stop"]]
        for component in schema["components"]
    }
    _validate_feature_v2_depth_summary(components["depth_summary"])
    action = components["official_action_onehot"]
    if not bool(((action == 0.0) | (action == 1.0)).all()) or float(action.sum()) != 1.0:
        raise ValueError("feature-v2 official action must be exactly one-hot")
    age_value = float(components["context_age"].item())
    if (
        not math.isfinite(age_value)
        or age_value != int(age_value)
        or not 0 <= int(age_value) <= SUCCESSOR_FEATURE_MAX_AGE
    ):
        raise ValueError("feature-v2 context age is invalid")
    context_age = int(age_value)
    official_shadow_action = int(action.argmax().item())
    return {
        "schema_version": SUCCESSOR_FEATURE_V2_SCHEMA_VERSION,
        "schema_sha256": successor_feature_v2_schema_digest(),
        "feature_sha256": _float32_tensor_sha256(checked),
        "component_sha256": {
            name: _float32_tensor_sha256(component)
            for name, component in components.items()
        },
        "shape": [SUCCESSOR_FEATURE_V2_DIM],
        "dtype": "torch.float32",
        "device": "cpu",
        "contiguous": True,
        "finite": True,
        "official_shadow_action": official_shadow_action,
        "context_age": context_age,
        "causal_current": context_age == 0,
    }


def compose_successor_features_v2(
    *,
    contextual_hidden: torch.Tensor,
    depth_summary: torch.Tensor,
    official_shadow_action: int,
    physical_step: int,
    context_physical_step: int,
) -> torch.Tensor:
    """Compose the frozen 3,604-wide feature-v2 causal receipt."""

    hidden = _require_feature_v2_vector(
        contextual_hidden,
        name="contextual_hidden",
        dimension=SUCCESSOR_FEATURE_V2_CONTEXT_DIM,
    )
    depth = _require_feature_v2_vector(
        depth_summary,
        name="depth_summary",
        dimension=SUCCESSOR_FEATURE_V2_DEPTH_DIM,
    )
    _validate_feature_v2_depth_summary(depth)
    if (
        type(official_shadow_action) is not int
        or not 0 <= official_shadow_action < SUCCESSOR_ACTION_FEATURE_DIM
    ):
        raise ValueError("feature-v2 official shadow action must be an integer in 0..3")
    if (
        type(physical_step) is not int
        or type(context_physical_step) is not int
        or context_physical_step < 0
        or physical_step < context_physical_step
    ):
        raise ValueError("feature-v2 context age is non-causal")
    context_age = physical_step - context_physical_step
    if context_age > SUCCESSOR_FEATURE_MAX_AGE:
        raise ValueError("feature-v2 context age exceeds the bounded contract")
    action = F.one_hot(
        torch.tensor(official_shadow_action, dtype=torch.int64),
        num_classes=SUCCESSOR_ACTION_FEATURE_DIM,
    ).to(dtype=torch.float32)
    result = torch.cat(
        (hidden, depth, action, torch.tensor([float(context_age)], dtype=torch.float32)),
        dim=0,
    ).contiguous()
    audit_successor_features_v2(result)
    return result


@dataclass(frozen=True)
class SuccessorFeatureV2GateAudit:
    """One deterministic conservative-gate outcome for per-decision counting."""

    selected_route: RouteName
    selected_waypoint_local_m: tuple[float, float]
    accepted: bool
    reason: str
    p_waypoint: float | None
    compiled_first_action: int | None
    compiled_actions: tuple[int, ...] | None
    root_fallback: bool = False

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["selected_route"] = self.selected_route.value
        result["selected_waypoint_local_m"] = list(self.selected_waypoint_local_m)
        if self.compiled_actions is not None:
            result["compiled_actions"] = list(self.compiled_actions)
        return result


def _feature_v2_gate_audit(
    *,
    accepted: bool,
    reason: str,
    p_waypoint: float | None,
    compiled_actions: tuple[int, ...] | None = None,
    waypoint_local_m: tuple[float, float] = (0.0, 0.0),
    root_fallback: bool = False,
) -> SuccessorFeatureV2GateAudit:
    return SuccessorFeatureV2GateAudit(
        selected_route=(RouteName.WAYPOINT if accepted else RouteName.KEEP_ORIGINAL),
        selected_waypoint_local_m=(waypoint_local_m if accepted else (0.0, 0.0)),
        accepted=accepted,
        reason=reason,
        p_waypoint=p_waypoint,
        compiled_first_action=(compiled_actions[0] if compiled_actions else None),
        compiled_actions=compiled_actions,
        root_fallback=root_fallback,
    )


def _feature_v2_route_probabilities(
    values: Sequence[float] | torch.Tensor | None,
) -> tuple[float, float, float] | None:
    if values is None:
        return None
    if isinstance(values, torch.Tensor):
        if values.ndim != 1 or tuple(values.shape) != (len(POLICY_ROUTES),):
            return None
        if not values.is_floating_point() or not bool(torch.isfinite(values).all()):
            return None
        raw_values = values.detach().cpu().tolist()
    else:
        if isinstance(values, (str, bytes)):
            return None
        try:
            raw_values = list(values)
        except TypeError:
            return None
        if len(raw_values) != len(POLICY_ROUTES):
            return None
    if any(type(value) is bool for value in raw_values):
        return None
    try:
        probabilities = tuple(float(value) for value in raw_values)
    except (TypeError, ValueError):
        return None
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities):
        return None
    if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-6):
        return None
    return probabilities  # type: ignore[return-value]


def _feature_v2_waypoint(
    value: Sequence[float] | torch.Tensor | None,
) -> tuple[float, float] | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.ndim != 1 or tuple(value.shape) != (2,):
            return None
        raw_values = value.detach().cpu().tolist()
    else:
        if isinstance(value, (str, bytes)):
            return None
        try:
            raw_values = list(value)
        except TypeError:
            return None
        if len(raw_values) != 2:
            return None
    if any(type(item) is bool for item in raw_values):
        return None
    try:
        waypoint = float(raw_values[0]), float(raw_values[1])
    except (TypeError, ValueError):
        return None
    return waypoint if all(math.isfinite(item) for item in waypoint) else None


def apply_successor_feature_v2_keep_gate(
    *,
    official_shadow_action: int,
    route_probabilities: Sequence[float] | torch.Tensor | None,
    waypoint_local_m: Sequence[float] | torch.Tensor | None,
    context_age: int | None,
    recovery_active: bool = False,
    blocked_parent: bool = False,
    blocked_root: bool = False,
) -> SuccessorFeatureV2GateAudit:
    """Fail closed for either matched arm unless a current waypoint is actionable."""

    if (
        type(official_shadow_action) is not int
        or not 0 <= official_shadow_action < SUCCESSOR_ACTION_FEATURE_DIM
    ):
        raise ValueError("feature-v2 gate official shadow action must be an integer in 0..3")
    if any(type(value) is not bool for value in (recovery_active, blocked_parent, blocked_root)):
        raise TypeError("feature-v2 gate recovery and blocked flags must be boolean")
    if official_shadow_action == STOP_ACTION:
        return _feature_v2_gate_audit(
            accepted=False,
            reason="official_stop_keep",
            p_waypoint=None,
        )
    if recovery_active or blocked_parent:
        return _feature_v2_gate_audit(
            accepted=False,
            reason="recovery_or_blocked_parent_keep",
            p_waypoint=None,
        )
    if blocked_root:
        return _feature_v2_gate_audit(
            accepted=False,
            reason="blocked_root_keep_fallback",
            p_waypoint=None,
            root_fallback=True,
        )
    if context_age is None:
        return _feature_v2_gate_audit(
            accepted=False,
            reason="missing_context_keep",
            p_waypoint=None,
        )
    if type(context_age) is not int or not 0 <= context_age <= SUCCESSOR_FEATURE_MAX_AGE:
        return _feature_v2_gate_audit(
            accepted=False,
            reason="invalid_context_age_keep",
            p_waypoint=None,
        )
    if context_age != 0:
        return _feature_v2_gate_audit(
            accepted=False,
            reason="stale_context_keep",
            p_waypoint=None,
        )
    probabilities = _feature_v2_route_probabilities(route_probabilities)
    if probabilities is None:
        return _feature_v2_gate_audit(
            accepted=False,
            reason="invalid_route_probabilities_keep",
            p_waypoint=None,
        )
    waypoint_index = ROUTE_TO_INDEX[RouteName.WAYPOINT]
    p_waypoint = probabilities[waypoint_index]
    maximum = max(probabilities)
    unique_waypoint_argmax = (
        p_waypoint == maximum and sum(value == maximum for value in probabilities) == 1
    )
    if not unique_waypoint_argmax:
        return _feature_v2_gate_audit(
            accepted=False,
            reason="learned_argmax_not_waypoint_keep",
            p_waypoint=p_waypoint,
        )
    if p_waypoint < SUCCESSOR_FEATURE_V2_WAYPOINT_MIN_PROBABILITY:
        return _feature_v2_gate_audit(
            accepted=False,
            reason="waypoint_probability_below_threshold_keep",
            p_waypoint=p_waypoint,
        )
    waypoint = _feature_v2_waypoint(waypoint_local_m)
    if waypoint is None:
        return _feature_v2_gate_audit(
            accepted=False,
            reason="invalid_waypoint_keep",
            p_waypoint=p_waypoint,
        )
    distance = math.hypot(*waypoint)
    if not (
        SUCCESSOR_FEATURE_V2_WAYPOINT_MIN_M
        < distance
        <= SUCCESSOR_FEATURE_V2_WAYPOINT_MAX_M
    ):
        return _feature_v2_gate_audit(
            accepted=False,
            reason="waypoint_distance_out_of_range_keep",
            p_waypoint=p_waypoint,
        )
    try:
        compiled_actions = tuple(compile_metric_waypoint(waypoint))
    except (TypeError, ValueError, RuntimeError):
        return _feature_v2_gate_audit(
            accepted=False,
            reason="invalid_waypoint_keep",
            p_waypoint=p_waypoint,
        )
    if compiled_actions[0] == official_shadow_action:
        return _feature_v2_gate_audit(
            accepted=False,
            reason="compiled_first_action_matches_official_keep",
            p_waypoint=p_waypoint,
            compiled_actions=compiled_actions,
        )
    return _feature_v2_gate_audit(
        accepted=True,
        reason="accepted_waypoint",
        p_waypoint=p_waypoint,
        compiled_actions=compiled_actions,
        waypoint_local_m=waypoint,
    )


def compose_successor_features(
    *,
    frozen_feature: torch.Tensor,
    official_shadow_action: int,
    physical_step: int,
    feature_physical_step: int,
) -> torch.Tensor:
    """Compose the sole production feature contract at a causal boundary."""

    if not isinstance(frozen_feature, torch.Tensor) or frozen_feature.ndim != 1:
        raise ValueError("frozen StreamVLN feature must have shape [feature_dim]")
    feature = frozen_feature.detach().cpu().to(dtype=torch.float32).contiguous()
    if feature.numel() < 2 or not bool(torch.isfinite(feature).all()):
        raise ValueError("frozen StreamVLN feature must be finite and non-empty")
    if (
        type(official_shadow_action) is not int
        or official_shadow_action < 0
        or official_shadow_action >= SUCCESSOR_ACTION_FEATURE_DIM
    ):
        raise ValueError("official shadow action must be an integer in 0..3")
    if (
        type(physical_step) is not int
        or type(feature_physical_step) is not int
        or feature_physical_step < 0
        or physical_step < feature_physical_step
    ):
        raise ValueError("successor feature age is non-causal")
    age = physical_step - feature_physical_step
    if age > 1_000_000:
        raise ValueError("successor feature age exceeds the bounded contract")
    action = F.one_hot(
        torch.tensor(official_shadow_action, dtype=torch.int64),
        num_classes=SUCCESSOR_ACTION_FEATURE_DIM,
    ).to(torch.float32)
    result = torch.cat((feature, action, torch.tensor([float(age)])), dim=0)
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("composed successor feature contains non-finite values")
    return result.contiguous()


def _cfg_get(value, key, default=None):
    getter = getattr(value, "get", None)
    return getter(key, default) if callable(getter) else getattr(value, key, default)


def _decision_axis(value: torch.Tensor, *, unbatched_ndim: int, name: str) -> torch.Tensor:
    if value.ndim == unbatched_ndim:
        return value.unsqueeze(0).unsqueeze(1)
    if value.ndim == unbatched_ndim + 2 and value.shape[1] == 1:
        return value
    raise ValueError(f"successor context {name} decision-axis shape drifted")


def _require_tensor_context(
    context: Mapping[str, object], *, feature_dim: int, topology_max_nodes: int
) -> dict[str, torch.Tensor]:
    if not isinstance(context, Mapping) or tuple(context) != SUCCESSOR_POLICY_CONTEXT_KEYS:
        raise ValueError("successor forward_context schema or key order drifted")
    if any(not isinstance(value, torch.Tensor) for value in context.values()):
        raise TypeError("successor forward_context must contain tensors only")
    normalized = {
        "schema_version": _decision_axis(
            context["schema_version"], unbatched_ndim=0, name="schema_version"
        ),
        "policy_version": _decision_axis(
            context["policy_version"], unbatched_ndim=0, name="policy_version"
        ),
        "features": _decision_axis(
            context["features"], unbatched_ndim=1, name="features"
        ),
        "route_candidate_mask": _decision_axis(
            context["route_candidate_mask"],
            unbatched_ndim=1,
            name="route_candidate_mask",
        ),
        "expert_route_candidate_mask": _decision_axis(
            context["expert_route_candidate_mask"],
            unbatched_ndim=1,
            name="expert_route_candidate_mask",
        ),
        "topology_global_features": _decision_axis(
            context["topology_global_features"],
            unbatched_ndim=1,
            name="topology_global_features",
        ),
        "topology_node_hidden": _decision_axis(
            context["topology_node_hidden"],
            unbatched_ndim=2,
            name="topology_node_hidden",
        ),
        "topology_node_features": _decision_axis(
            context["topology_node_features"],
            unbatched_ndim=2,
            name="topology_node_features",
        ),
        "topology_node_mask": _decision_axis(
            context["topology_node_mask"],
            unbatched_ndim=1,
            name="topology_node_mask",
        ),
        "topology_adjacency": _decision_axis(
            context["topology_adjacency"],
            unbatched_ndim=2,
            name="topology_adjacency",
        ),
        "topology_edge_features": _decision_axis(
            context["topology_edge_features"],
            unbatched_ndim=3,
            name="topology_edge_features",
        ),
        "topology_current_index": _decision_axis(
            context["topology_current_index"],
            unbatched_ndim=0,
            name="topology_current_index",
        ),
        "topology_parent_index": _decision_axis(
            context["topology_parent_index"],
            unbatched_ndim=1,
            name="topology_parent_index",
        ),
    }
    features = normalized["features"]
    candidate_mask = normalized["route_candidate_mask"]
    expert_candidate_mask = normalized["expert_route_candidate_mask"]
    schema = normalized["schema_version"]
    policy_version = normalized["policy_version"]
    if features.ndim != 3 or features.shape[-1] != feature_dim:
        raise ValueError("successor context features must have shape [B, 1, feature_dim]")
    expected_prefix = features.shape[:2]
    if expected_prefix[1] != 1:
        raise ValueError("successor context represents one decision per row")
    if tuple(candidate_mask.shape) != (*expected_prefix, len(POLICY_ROUTES)):
        raise ValueError("successor route candidate mask shape drifted")
    if tuple(expert_candidate_mask.shape) != (*expected_prefix, len(POLICY_ROUTES)):
        raise ValueError("successor expert route candidate mask shape drifted")
    if tuple(schema.shape) != tuple(expected_prefix) or tuple(policy_version.shape) != tuple(expected_prefix):
        raise ValueError("successor context version tensor shape drifted")
    if schema.dtype != torch.int64 or policy_version.dtype != torch.int64:
        raise TypeError("successor context versions must be int64 tensors")
    if candidate_mask.dtype != torch.bool or expert_candidate_mask.dtype != torch.bool:
        raise TypeError("successor route candidate masks must be bool")
    if bool((schema != SUCCESSOR_POLICY_CONTEXT_VERSION).any()):
        raise RuntimeError("successor forward_context version mismatch")
    if bool((~candidate_mask.any(dim=-1)).any()):
        raise ValueError("successor route candidate mask cannot be all false")
    if bool((~expert_candidate_mask.any(dim=-1)).any()):
        raise ValueError("successor expert route candidate mask cannot be all false")
    if bool((candidate_mask & ~expert_candidate_mask).any()):
        raise ValueError("successor behavior candidates must be a subset of expert candidates")
    if not features.is_floating_point() or not bool(torch.isfinite(features).all()):
        raise ValueError("successor context features must be finite floating tensors")
    global_features = normalized["topology_global_features"]
    if tuple(global_features.shape) != (*expected_prefix, CAUSAL_GLOBAL_FEATURE_DIM):
        raise ValueError("successor topology global feature shape drifted")
    if not global_features.is_floating_point() or not bool(
        torch.isfinite(global_features).all()
    ):
        raise ValueError("successor topology global features must be finite")
    topology = P8TopologyTensors(
        node_hidden=normalized["topology_node_hidden"].squeeze(1),
        node_features=normalized["topology_node_features"].squeeze(1),
        node_mask=normalized["topology_node_mask"].squeeze(1),
        adjacency=normalized["topology_adjacency"].squeeze(1),
        edge_features=normalized["topology_edge_features"].squeeze(1),
        current_index=normalized["topology_current_index"].squeeze(1),
        parent_index=normalized["topology_parent_index"].squeeze(1),
    ).validate()
    if topology.max_nodes != topology_max_nodes:
        raise ValueError("successor topology fixed node extent drifted")
    if topology.node_hidden.shape[-1] != CAUSAL_NODE_HIDDEN_DIM:
        raise ValueError("successor causal node hidden width drifted")
    if topology.node_features.shape[-1] != CAUSAL_NODE_FEATURE_DIM:
        raise ValueError("successor causal node feature width drifted")
    if topology.edge_features.shape[-1] != CAUSAL_EDGE_FEATURE_DIM:
        raise ValueError("successor causal edge feature width drifted")
    return normalized


def make_successor_forward_context(
    *,
    features: torch.Tensor,
    topology_snapshot: P8CausalTopologySnapshot,
    policy_version: int,
) -> dict[str, torch.Tensor]:
    """Create the only replayable context accepted by rollout and actor."""

    if type(policy_version) is not int or policy_version < 0:
        raise ValueError("policy_version must be a non-negative integer")
    if not isinstance(features, torch.Tensor) or features.ndim != 1:
        raise ValueError("rollout successor features must have shape [feature_dim]")
    if not isinstance(topology_snapshot, P8CausalTopologySnapshot):
        raise TypeError("rollout requires a typed causal topology snapshot")
    topology = topology_snapshot.tensors.canonical()
    if topology.batch_size != 1:
        raise ValueError("one rollout frame requires one causal topology row")
    route_candidate_mask = torch.tensor(
        topology_snapshot.route_candidate_mask, dtype=torch.bool
    )
    context = {
        "schema_version": torch.tensor(SUCCESSOR_POLICY_CONTEXT_VERSION, dtype=torch.int64),
        "policy_version": torch.tensor(policy_version, dtype=torch.int64),
        "features": features.detach().to(dtype=torch.float32),
        "route_candidate_mask": route_candidate_mask.detach().to(dtype=torch.bool),
        "expert_route_candidate_mask": route_candidate_mask.detach().to(dtype=torch.bool).clone(),
        "topology_global_features": torch.tensor(
            topology_snapshot.global_features, dtype=torch.float32
        ),
        "topology_node_hidden": topology.node_hidden[0].detach().clone(),
        "topology_node_features": topology.node_features[0].detach().clone(),
        "topology_node_mask": topology.node_mask[0].detach().clone(),
        "topology_adjacency": topology.adjacency[0].detach().clone(),
        "topology_edge_features": topology.edge_features[0].detach().clone(),
        "topology_current_index": topology.current_index[0].detach().clone(),
        "topology_parent_index": topology.parent_index[0].detach().clone(),
    }
    _require_tensor_context(
        context,
        feature_dim=int(features.numel()),
        topology_max_nodes=topology.max_nodes,
    )
    return context


class P8SuccessorPolicy(nn.Module, BasePolicy):
    """Small synchronized head for a conditional metric-waypoint action."""

    def __init__(
        self,
        *,
        feature_dim: int,
        hidden_dim: int = 256,
        topology_max_nodes: int = 41,
        topology_model_dim: int = 64,
        max_waypoint_m: float = 1.25,
        initial_log_std: float = -0.5,
        expected_bc_readiness_digest: str | None = None,
    ) -> None:
        super().__init__()
        if type(feature_dim) is not int or feature_dim < 1:
            raise ValueError("successor feature_dim must be positive")
        if type(hidden_dim) is not int or hidden_dim < 1:
            raise ValueError("successor hidden_dim must be positive")
        if type(topology_max_nodes) is not int or topology_max_nodes < 2:
            raise ValueError("successor topology_max_nodes must be at least two")
        if type(topology_model_dim) is not int or topology_model_dim < 1:
            raise ValueError("successor topology_model_dim must be positive")
        if not math.isfinite(float(max_waypoint_m)) or float(max_waypoint_m) <= 0:
            raise ValueError("max_waypoint_m must be finite and positive")
        self.feature_dim = feature_dim
        self.topology_max_nodes = topology_max_nodes
        self.topology_encoder = P8TopologyEncoder(
            node_hidden_dim=CAUSAL_NODE_HIDDEN_DIM,
            node_feature_dim=CAUSAL_NODE_FEATURE_DIM,
            edge_feature_dim=CAUSAL_EDGE_FEATURE_DIM,
            model_dim=topology_model_dim,
        )
        self.max_waypoint_m = float(max_waypoint_m)
        self._expected_bc_readiness_digest: str | None = None
        if expected_bc_readiness_digest is not None:
            self.set_expected_bc_readiness_digest(expected_bc_readiness_digest)
        self.min_waypoint_m = 0.1001
        if self.max_waypoint_m <= self.min_waypoint_m:
            raise ValueError("max_waypoint_m must exceed the compiler arrival tolerance")
        self.backbone = nn.Sequential(
            nn.Linear(
                feature_dim + CAUSAL_GLOBAL_FEATURE_DIM + topology_model_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.route_head = nn.Linear(hidden_dim, len(POLICY_ROUTES))
        self.waypoint_mean_head = nn.Linear(hidden_dim, 2)
        self.waypoint_log_std = nn.Parameter(torch.full((2,), float(initial_log_std)))
        self.value_head = nn.Linear(hidden_dim, 1)
        self.register_buffer("policy_version", torch.zeros((), dtype=torch.int64))
        # Persistent FP32 state participates in RLinf's selective parameter/buffer
        # sync and remains compatible with delta-based sync backends.
        self.register_buffer("bc_ready", torch.zeros((), dtype=torch.float32), persistent=True)
        self.register_buffer(
            "bc_readiness_digest", torch.zeros(32, dtype=torch.float32), persistent=True
        )
        with torch.no_grad():
            nn.init.zeros_(self.route_head.weight)
            nn.init.zeros_(self.route_head.bias)

    def set_global_step(self, version: int) -> None:
        if type(version) is not int or version < 0:
            raise ValueError("successor policy version must be non-negative")
        self.policy_version.fill_(version)

    @staticmethod
    def _validate_readiness_digest(receipt_sha256: str) -> None:
        if (
            not isinstance(receipt_sha256, str)
            or len(receipt_sha256) != 64
            or receipt_sha256 == "0" * 64
            or any(character not in "0123456789abcdef" for character in receipt_sha256)
        ):
            raise ValueError("BC readiness requires a lowercase SHA-256 receipt digest")

    def set_expected_bc_readiness_digest(self, receipt_sha256: str) -> None:
        self._validate_readiness_digest(receipt_sha256)
        if (
            self._expected_bc_readiness_digest is not None
            and self._expected_bc_readiness_digest != receipt_sha256
        ):
            raise RuntimeError("BC readiness expected digest is immutable")
        self._expected_bc_readiness_digest = receipt_sha256

    def _raw_bc_readiness(self) -> tuple[bool, str | None]:
        value = float(self.bc_ready.detach().cpu().item())
        if not math.isfinite(value) or value not in (0.0, 1.0):
            raise RuntimeError("successor BC readiness state is invalid")
        digest_values = self.bc_readiness_digest.detach().cpu().tolist()
        if any(
            not math.isfinite(float(item))
            or float(item) != int(item)
            or not 0 <= int(item) <= 255
            for item in digest_values
        ):
            raise RuntimeError("successor BC readiness digest bytes are invalid")
        digest_bytes = bytes(int(item) for item in digest_values)
        digest_nonzero = any(digest_bytes)
        if (value == 1.0) != digest_nonzero:
            raise RuntimeError("successor BC readiness flag/digest binding is invalid")
        return value == 1.0, digest_bytes.hex() if digest_nonzero else None

    def promote_bc_readiness(self, receipt_sha256: str) -> None:
        """Bind readiness to one immutable external evaluation receipt digest."""

        self._validate_readiness_digest(receipt_sha256)
        digest = torch.tensor(
            list(bytes.fromhex(receipt_sha256)),
            dtype=torch.float32,
            device=self.bc_readiness_digest.device,
        )
        ready, current_digest = self._raw_bc_readiness()
        if ready:
            if current_digest != receipt_sha256:
                raise RuntimeError("BC readiness is already bound to another receipt")
            self.set_expected_bc_readiness_digest(receipt_sha256)
            return
        self.set_expected_bc_readiness_digest(receipt_sha256)
        self.bc_readiness_digest.copy_(digest)
        self.bc_ready.fill_(1.0)

    def bc_readiness_receipt_sha256(self) -> str | None:
        _, digest = self._raw_bc_readiness()
        return digest

    def is_bc_ready(self, expected_digest: str | None = None) -> bool:
        ready, actual_digest = self._raw_bc_readiness()
        if not ready:
            return False
        expected = expected_digest or self._expected_bc_readiness_digest
        if expected is None:
            raise RuntimeError("ready successor policy lacks a pinned receipt digest")
        self._validate_readiness_digest(expected)
        if actual_digest != expected:
            raise RuntimeError("successor BC readiness receipt digest mismatched")
        return True

    def _distribution_terms(self, forward_inputs: Mapping[str, object]):
        context = _require_tensor_context(
            forward_inputs,
            feature_dim=self.feature_dim,
            topology_max_nodes=self.topology_max_nodes,
        )
        expected_version = int(self.policy_version.detach().cpu().item())
        if bool((context["policy_version"].detach().cpu() != expected_version).any()):
            raise RuntimeError("successor rollout/actor policy version mismatch")
        features = context["features"].to(device=self.route_head.weight.device, dtype=torch.float32)
        candidate_mask = context["route_candidate_mask"].to(device=features.device)
        expert_candidate_mask = context["expert_route_candidate_mask"].to(
            device=features.device
        )
        topology = P8TopologyTensors(
            node_hidden=context["topology_node_hidden"].squeeze(1),
            node_features=context["topology_node_features"].squeeze(1),
            node_mask=context["topology_node_mask"].squeeze(1),
            adjacency=context["topology_adjacency"].squeeze(1),
            edge_features=context["topology_edge_features"].squeeze(1),
            current_index=context["topology_current_index"].squeeze(1),
            parent_index=context["topology_parent_index"].squeeze(1),
        )
        topology_context = self.topology_encoder(topology)["context"].unsqueeze(1)
        global_features = context["topology_global_features"].to(
            device=features.device, dtype=torch.float32
        )
        hidden = self.backbone(
            torch.cat((features, global_features, topology_context), dim=-1)
        )
        raw_route_logits = self.route_head(hidden)
        masked_logits = raw_route_logits.masked_fill(~candidate_mask, -torch.inf)
        raw_mean = self.waypoint_mean_head(hidden)
        raw_log_std = self.waypoint_log_std.clamp(-5.0, 2.0).reshape(1, 1, 2).expand_as(raw_mean)
        values = self.value_head(hidden).squeeze(-1)
        return (
            context,
            candidate_mask,
            expert_candidate_mask,
            raw_route_logits,
            masked_logits,
            raw_mean,
            raw_log_std,
            values,
        )

    def _waypoint_from_raw(self, raw: torch.Tensor) -> torch.Tensor:
        distance_fraction = torch.sigmoid(raw[..., 0])
        distance = self.min_waypoint_m + (
            self.max_waypoint_m - self.min_waypoint_m
        ) * distance_fraction
        bearing = math.pi * torch.tanh(raw[..., 1])
        return torch.stack((distance * torch.cos(bearing), distance * torch.sin(bearing)), dim=-1)

    def _waypoint_logprob(
        self, raw: torch.Tensor, raw_mean: torch.Tensor, raw_log_std: torch.Tensor
    ) -> torch.Tensor:
        base = Normal(raw_mean, raw_log_std.exp()).log_prob(raw).sum(dim=-1)
        distance_fraction = torch.sigmoid(raw[..., 0]).clamp(1e-6, 1.0 - 1e-6)
        distance = self.min_waypoint_m + (
            self.max_waypoint_m - self.min_waypoint_m
        ) * distance_fraction
        bearing_squash = torch.tanh(raw[..., 1]).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        # |d(x,y)/d(raw_distance,raw_bearing)| = r * dr * dbearing.
        log_jacobian = (
            torch.log(distance)
            + math.log(self.max_waypoint_m - self.min_waypoint_m)
            + torch.log(distance_fraction)
            + torch.log1p(-distance_fraction)
            + math.log(math.pi)
            + torch.log1p(-bearing_squash.square())
        )
        return base - log_jacobian

    def _raw_from_waypoint(self, waypoint: torch.Tensor) -> torch.Tensor:
        distance = waypoint.square().sum(dim=-1).sqrt()
        distance_fraction = (
            (distance - self.min_waypoint_m)
            / (self.max_waypoint_m - self.min_waypoint_m)
        ).clamp(1e-6, 1.0 - 1e-6)
        raw_distance = torch.log(distance_fraction) - torch.log1p(-distance_fraction)
        bearing_fraction = (
            torch.atan2(waypoint[..., 1], waypoint[..., 0]) / math.pi
        ).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        return torch.stack((raw_distance, torch.atanh(bearing_fraction)), dim=-1)

    def forward(
        self,
        *,
        forward_inputs: Mapping[str, object],
        route_actions: torch.Tensor,
        waypoint_targets: torch.Tensor,
        expert_waypoint_targets: torch.Tensor | None = None,
        compute_logprobs: bool = True,
        compute_values: bool = True,
        use_cache: bool = False,
        **_kwargs,
    ) -> dict[str, torch.Tensor]:
        if use_cache:
            raise ValueError("successor policy context is replayed without cache")
        (
            _,
            candidate_mask,
            expert_candidate_mask,
            raw_route_logits,
            route_logits,
            raw_mean,
            raw_log_std,
            values,
        ) = self._distribution_terms(forward_inputs)
        device = route_logits.device
        route_actions = route_actions.to(device=device, dtype=torch.int64)
        waypoint_targets = waypoint_targets.to(device=device, dtype=torch.float32)
        if route_actions.shape != route_logits.shape[:-1]:
            raise ValueError("successor route action shape mismatch")
        if waypoint_targets.shape != raw_mean.shape:
            raise ValueError("successor waypoint target shape mismatch")
        if bool(((route_actions < 0) | (route_actions >= len(POLICY_ROUTES))).any()):
            raise ValueError("successor route action is out of range")
        if bool((~candidate_mask.gather(-1, route_actions.unsqueeze(-1)).squeeze(-1)).any()):
            raise ValueError("successor replay selected a masked route")
        route_logprobs = Categorical(logits=route_logits).log_prob(route_actions)
        raw_targets = self._raw_from_waypoint(waypoint_targets)
        waypoint_logprobs = self._waypoint_logprob(raw_targets, raw_mean, raw_log_std)
        result = {
            "raw_route_logits": raw_route_logits,
            "route_candidate_mask": candidate_mask,
            "expert_route_candidate_mask": expert_candidate_mask,
            "route_logprobs": route_logprobs,
            "waypoint_logprobs": waypoint_logprobs,
            "waypoint_predictions": self._waypoint_from_raw(raw_mean),
            "waypoint_raw_mean": raw_mean,
            "waypoint_raw_log_std": raw_log_std,
            "values": values,
        }
        if expert_waypoint_targets is not None:
            expert_waypoint_targets = expert_waypoint_targets.to(
                device=device, dtype=torch.float32
            )
            if expert_waypoint_targets.shape != raw_mean.shape:
                raise ValueError("successor expert waypoint target shape mismatch")
            result["expert_waypoint_logprobs"] = self._waypoint_logprob(
                self._raw_from_waypoint(expert_waypoint_targets),
                raw_mean,
                raw_log_std,
            )
        if not compute_logprobs:
            result.pop("route_logprobs")
            result.pop("waypoint_logprobs")
        if not compute_values:
            result.pop("values")
        return result

    def default_forward(self, **kwargs) -> dict[str, torch.Tensor]:
        """Satisfy RLinf's BasePolicy contract with the successor replay path."""

        return self.forward(**kwargs)

    def predict_action_batch(self, **_kwargs):
        """Successor rollout sampling is owned by the decision-boundary controller."""

        raise NotImplementedError(
            "P8 successor sampling requires build_successor_rollout_controller"
        )

    @torch.no_grad()
    def sample_successor_decision(
        self,
        *,
        forward_context: Mapping[str, object],
        episode_token: str,
        physical_step: int,
        decision_index: int,
        official_shadow_action: int,
        deterministic: bool,
        expert_route_target: RouteName | None = None,
        expert_waypoint_target: tuple[float, float] | None = None,
    ) -> SuccessorDecisionSample:
        normalized = _require_tensor_context(
            forward_context,
            feature_dim=self.feature_dim,
            topology_max_nodes=self.topology_max_nodes,
        )
        behavior_context = dict(forward_context)
        keep_index = ROUTE_TO_INDEX[RouteName.KEEP_ORIGINAL]
        if not self.is_bc_ready():
            if not bool(normalized["route_candidate_mask"][0, 0, keep_index]):
                raise RuntimeError("cold successor head requires KEEP_ORIGINAL to be available")
            keep_only = torch.zeros_like(forward_context["route_candidate_mask"], dtype=torch.bool)
            keep_only[..., keep_index] = True
            behavior_context["route_candidate_mask"] = keep_only
        (
            _,
            candidate_mask,
            _,
            _,
            route_logits,
            raw_mean,
            raw_log_std,
            values,
        ) = self._distribution_terms(behavior_context)
        if route_logits.shape[:2] != (1, 1):
            raise ValueError("rollout sampling requires exactly one successor decision")
        if not self.is_bc_ready():
            route_index = torch.tensor([[keep_index]], device=route_logits.device)
        else:
            route_distribution = Categorical(logits=route_logits)
            route_index = route_logits.argmax(dim=-1) if deterministic else route_distribution.sample()
        route_distribution = Categorical(logits=route_logits)
        route_logprob = route_distribution.log_prob(route_index)
        route = POLICY_ROUTES[int(route_index.item())]
        if route is RouteName.WAYPOINT:
            raw = raw_mean if deterministic else Normal(raw_mean, raw_log_std.exp()).sample()
            waypoint_tensor = self._waypoint_from_raw(raw)
            waypoint_logprob = self._waypoint_logprob(raw, raw_mean, raw_log_std)
            waypoint = tuple(float(value) for value in waypoint_tensor.reshape(-1).tolist())
            proposed_actions = compile_metric_waypoint(waypoint)
            joint_logprob = route_logprob + waypoint_logprob
        elif route is RouteName.KEEP_ORIGINAL:
            waypoint = (0.0, 0.0)
            proposed_actions = (int(official_shadow_action),)
            joint_logprob = route_logprob
        else:
            waypoint = (0.0, 0.0)
            proposed_actions = (STOP_ACTION,)
            joint_logprob = route_logprob
        decision = RouteDecision(
            decision_index=decision_index,
            route=route,
            waypoint_local_m=waypoint,
            official_action=int(official_shadow_action) if route is RouteName.KEEP_ORIGINAL else None,
            episode_token=episode_token,
            physical_step=physical_step,
        )
        return SuccessorDecisionSample(
            decision=decision,
            proposed_actions=proposed_actions,
            logprob=float(joint_logprob.item()),
            value=float(values.item()),
            official_shadow_action=int(official_shadow_action),
            forward_context={
                key: value.detach().cpu().clone() for key, value in behavior_context.items()
            },
            expert_route_target=expert_route_target,
            expert_waypoint_target=expert_waypoint_target,
        )

    @torch.no_grad()
    def evaluate_successor_value(self, forward_context: Mapping[str, object]) -> float:
        *_, values = self._distribution_terms(forward_context)
        if values.shape != (1, 1):
            raise ValueError("successor bootstrap requires exactly one decision state")
        return float(values.item())

def get_model(cfg, torch_dtype=None):
    model = P8SuccessorPolicy(
        feature_dim=int(_cfg_get(cfg, "successor_feature_dim")),
        hidden_dim=int(_cfg_get(cfg, "successor_hidden_dim", 256)),
        topology_max_nodes=int(_cfg_get(cfg, "successor_topology_max_nodes")),
        topology_model_dim=int(_cfg_get(cfg, "successor_topology_model_dim", 64)),
        max_waypoint_m=float(_cfg_get(cfg, "successor_max_waypoint_m", 1.25)),
        initial_log_std=float(_cfg_get(cfg, "successor_initial_log_std", -0.5)),
        expected_bc_readiness_digest=_cfg_get(
            cfg, "successor_bc_readiness_receipt_sha256", None
        ),
    )
    # This small head stays FP32 on rollout and actor.  Casting observations to
    # FP32 is then deterministic and cannot mismatch BF16 module weights.
    return model.float()


def _single_observation_tensor(observation: Mapping[str, object], key: str) -> torch.Tensor:
    if key not in observation:
        raise KeyError(f"successor observation is missing {key}")
    value = observation[key]
    tensor = value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    return tensor.reshape(-1)


def _instruction_from_observation(observation: Mapping[str, object]) -> str:
    values = _single_observation_tensor(observation, "instruction_utf8").to(torch.uint8)
    mask = _single_observation_tensor(observation, "instruction_utf8_mask").to(torch.bool)
    if values.shape != mask.shape or not bool(mask.any()):
        raise ValueError("successor instruction encoding is invalid")
    try:
        return bytes(int(value) for value in values[mask]).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("successor instruction is not valid UTF-8") from exc


def build_successor_env_controller(*, worker, env, stage_id: int, env_slot: int):
    if stage_id != 0 or env_slot != 0:
        raise ValueError("production successor env factory supports only stage/slot zero")
    try:
        from .streamvln_p8_successor_action import SuccessorPlanner
        from .streamvln_p8_successor_workers import SuccessorEnvController
    except ImportError:
        from streamvln_p8_successor_action import SuccessorPlanner
        from streamvln_p8_successor_workers import SuccessorEnvController
    train_cfg = _cfg_get(_cfg_get(worker.cfg, "env", {}), "train", {})
    model_cfg = _cfg_get(_cfg_get(worker.cfg, "actor", {}), "model", {})
    max_decisions = int(_cfg_get(train_cfg, "max_episode_steps"))
    topology_max_nodes = int(_cfg_get(model_cfg, "successor_topology_max_nodes"))
    blocked_threshold = int(_cfg_get(model_cfg, "successor_blocked_threshold", 2))
    bc_cfg = _cfg_get(_cfg_get(worker.cfg, "algorithm", {}), "successor_bc", {})
    bc_enabled = bool(_cfg_get(bc_cfg, "enabled", False))
    expert_source = str(_cfg_get(bc_cfg, "expert_source", "none"))
    if bc_enabled and expert_source != "train_shortest_path":
        raise RuntimeError("BC env controller requires train_shortest_path expert data")
    if not bc_enabled and expert_source not in {"none", "None", ""}:
        raise RuntimeError("expert data source may be configured only for explicit BC")
    return SuccessorEnvController(
        env=env,
        env_slot=env_slot,
        max_decisions=max_decisions,
        topology_max_nodes=topology_max_nodes,
        planner=SuccessorPlanner(blocked_threshold=blocked_threshold),
        bc_expert_enabled=bc_enabled,
        expert_lookahead_m=float(_cfg_get(bc_cfg, "expert_lookahead_m", 1.0)),
    )


def _load_official_adapter(worker):
    runner = _cfg_get(worker.cfg, "runner", {})
    official = _cfg_get(runner, "p8_successor_official", None)
    if official is None:
        raise RuntimeError("runner.p8_successor_official is required")
    manifest_path = Path(str(_cfg_get(official, "artifact_manifest"))).expanduser().resolve(strict=True)
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    try:
        from .streamvln_p8_official_adapter import PinnedStreamVLNGenerationAdapter
    except ImportError:
        from streamvln_p8_official_adapter import PinnedStreamVLNGenerationAdapter
    local_rank = int(worker.device)
    return PinnedStreamVLNGenerationAdapter.from_released_checkpoint(
        model_path=str(_cfg_get(official, "model_path")),
        tokenizer_path=str(_cfg_get(official, "tokenizer_path")),
        vision_tower_path=str(_cfg_get(official, "vision_tower_path")),
        source_root=str(_cfg_get(official, "source_root")),
        artifact_manifest=manifest,
        device=f"cuda:{local_rank}",
        local_rank=local_rank,
        env_count=1,
        model_env_id=0,
    )


def build_successor_rollout_controller(*, worker, env_slot: int):
    """Bind the RLinf-synchronized worker.hf_model; never construct a policy here."""

    if env_slot != 0:
        raise ValueError("production successor rollout factory supports only slot zero")
    policy = getattr(worker, "hf_model", None)
    if not isinstance(policy, P8SuccessorPolicy):
        raise TypeError("successor rollout factory must bind worker.hf_model")
    official_adapter = _load_official_adapter(worker)
    try:
        from .streamvln_p8_official_keep import OfficialEvalSession
        from .streamvln_p8_successor_workers import SuccessorRolloutController
    except ImportError:
        from streamvln_p8_official_keep import OfficialEvalSession
        from streamvln_p8_successor_workers import SuccessorRolloutController
    runner = _cfg_get(worker.cfg, "runner", {})
    bc_cfg = _cfg_get(_cfg_get(worker.cfg, "algorithm", {}), "successor_bc", {})
    deterministic = bool(_cfg_get(runner, "only_eval", False))
    seed = int(_cfg_get(worker.cfg, "seed", 0))
    latest_feature = {"snapshot": None, "official_action": None}

    def context_for(frame, official_action: int, *, extract_current: bool):
        if not isinstance(frame.observation, Mapping):
            raise TypeError("successor policy requires a mapping observation")
        if extract_current:
            if frame.official_frame is None:
                raise RuntimeError("decision feature extraction requires an OfficialFrame")
            snapshot = official_adapter.extract_frozen_feature_snapshot(
                frame=frame.official_frame,
                instruction=_instruction_from_observation(frame.observation),
            )
            latest_feature["snapshot"] = snapshot
            latest_feature["official_action"] = int(official_action)
        else:
            snapshot = latest_feature["snapshot"]
            official_action = latest_feature["official_action"]
            if snapshot is None or official_action is None:
                raise RuntimeError("successor bootstrap has no causal frozen feature")
        features = compose_successor_features(
            frozen_feature=snapshot.feature,
            official_shadow_action=int(official_action),
            physical_step=frame.physical_step,
            feature_physical_step=snapshot.physical_step,
        )
        if features.numel() != policy.feature_dim:
            raise ValueError("frozen successor feature dimension disagrees with policy")
        return make_successor_forward_context(
            features=features,
            topology_snapshot=frame.topology_snapshot,
            policy_version=int(policy.policy_version.item()),
        )

    def official_driver_factory(frame):
        official_frame = frame.official_frame
        instruction = _instruction_from_observation(frame.observation)
        identity = {
            "instruction_text": instruction,
            "episode_id": official_frame.episode_id,
            "scene_id": official_frame.scene_id,
            "instruction_sha256": official_frame.instruction_sha256,
        }
        session = OfficialEvalSession(
            env_id=official_frame.env_id,
            instruction=instruction,
            conjunction_rng=random.Random(seed + official_frame.episode_serial),
        )
        return official_adapter.make_driver(
            session=session,
            episode_serial=official_frame.episode_serial,
            episode_identity=identity,
        )

    def decision_sampler(frame, shadow_action):
        expert_source = str(_cfg_get(bc_cfg, "expert_source", "none"))
        if bool(_cfg_get(bc_cfg, "enabled", False)):
            if expert_source != "train_shortest_path":
                raise RuntimeError(
                    "production BC requires expert_source=train_shortest_path"
                )
            expert = frame.expert_target
            if expert is None:
                raise RuntimeError("production BC decision lacks a typed expert target")
            if expert.stop:
                expert_route = RouteName.STOP
                expert_waypoint = None
            else:
                waypoint = tuple(expert.waypoint_local_m)
                expert_first = compile_metric_waypoint(waypoint)[0]
                if int(shadow_action) == expert_first:
                    expert_route = RouteName.KEEP_ORIGINAL
                    expert_waypoint = None
                else:
                    expert_route = RouteName.WAYPOINT
                    expert_waypoint = waypoint
        else:
            if frame.expert_target is not None:
                raise RuntimeError("non-BC rollout received private expert data")
            expert_route = None
            expert_waypoint = None
        return policy.sample_successor_decision(
            forward_context=context_for(frame, shadow_action, extract_current=True),
            episode_token=frame.episode_token,
            physical_step=frame.physical_step,
            decision_index=frame.decision_index,
            official_shadow_action=shadow_action,
            deterministic=deterministic,
            expert_route_target=expert_route,
            expert_waypoint_target=expert_waypoint,
        )

    def bootstrap_value(frame):
        return policy.evaluate_successor_value(
            context_for(frame, 0, extract_current=False)
        )

    controller = SuccessorRolloutController(
        official_driver_factory=official_driver_factory,
        decision_sampler=decision_sampler,
        bootstrap_value_evaluator=bootstrap_value,
    )
    controller.synchronized_policy = policy
    return controller


build_successor_env_controller.successor_factory_version = SUCCESSOR_POLICY_CONTEXT_VERSION
build_successor_rollout_controller.successor_factory_version = SUCCESSOR_POLICY_CONTEXT_VERSION
build_successor_env_controller.successor_replay_schema = SUCCESSOR_REPLAY_SCHEMA
build_successor_rollout_controller.successor_replay_schema = SUCCESSOR_REPLAY_SCHEMA


__all__ = [
    "POLICY_ROUTES",
    "ROUTE_TO_INDEX",
    "SUCCESSOR_FEATURE_CHECKPOINT_BINDING_KEYS",
    "SUCCESSOR_FEATURE_MAX_AGE",
    "SUCCESSOR_FEATURE_V1_DIM",
    "SUCCESSOR_FEATURE_V1_FROZEN_DIM",
    "SUCCESSOR_FEATURE_V1_SCHEMA_VERSION",
    "SUCCESSOR_FEATURE_V2_CONTEXT_DIM",
    "SUCCESSOR_FEATURE_V2_DEPTH_DIM",
    "SUCCESSOR_FEATURE_V2_DIM",
    "SUCCESSOR_FEATURE_V2_SCHEMA_VERSION",
    "SUCCESSOR_FEATURE_V2_WAYPOINT_MAX_M",
    "SUCCESSOR_FEATURE_V2_WAYPOINT_MIN_M",
    "SUCCESSOR_FEATURE_V2_WAYPOINT_MIN_PROBABILITY",
    "SUCCESSOR_POLICY_CONTEXT_KEYS",
    "SUCCESSOR_POLICY_CONTEXT_VERSION",
    "SUCCESSOR_POLICY_MODEL_TYPE",
    "SUCCESSOR_ACTION_FEATURE_DIM",
    "SUCCESSOR_COMPOSED_AUX_DIM",
    "P8SuccessorPolicy",
    "SuccessorFeatureV2GateAudit",
    "apply_successor_feature_v2_keep_gate",
    "audit_successor_features_v2",
    "build_successor_env_controller",
    "build_successor_rollout_controller",
    "compose_successor_features_v2",
    "get_model",
    "compose_successor_features",
    "make_successor_forward_context",
    "require_successor_feature_checkpoint_binding",
    "successor_feature_checkpoint_binding",
    "successor_feature_schema_digest",
    "successor_feature_v2_schema",
    "successor_feature_v2_schema_digest",
]
