"""Canonical proposal-conditioned actions for Navigation-RLT.

The actor chooses a high-level mode.  This module deterministically compiles it
to bounded SE(2) primitives and returns the exact canonical action consumed by
the critics.  Safety overrides are recorded by rollout code and never rewritten
into a learned mode here.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


KEEP = 0
ADJUST = 1
STOP = 2
RECOVER = 3
MODE_NAMES = ("KEEP", "ADJUST", "STOP", "RECOVER")
MODE_COUNT = len(MODE_NAMES)
RESIDUAL_LIMITS = np.asarray((0.05, math.radians(10.0)), dtype=np.float32)
FORWARD_RANGE_M = (0.0, 0.25)
YAW_RANGE_RAD = (-math.radians(30.0), math.radians(30.0))
MAX_RECOVERY_PRIMITIVES = 4
CANONICAL_ACTION_DIM = MODE_COUNT + 3


class ActionContractError(ValueError):
    """Raised when a requested action cannot be executed unambiguously."""


def _se2(value: Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ActionContractError(f"{name} must be finite SE(2)")
    return result


def _mode_mask(value: Sequence[bool]) -> np.ndarray:
    result = np.asarray(value)
    if result.shape != (MODE_COUNT,) or result.dtype != np.bool_:
        raise ActionContractError("valid_mode_mask must be bool[4]")
    if not bool(result.any()):
        raise ActionContractError("at least one mode must be valid")
    return result


def valid_mode_mask(*, native_stop: bool, has_recovery_parent: bool) -> tuple[bool, bool, bool, bool]:
    """Return the alias-free mode set for this state.

    KEEP and ADJUST are invalid after a native STOP because both would otherwise
    alias an all-zero command.  RECOVER is available only with a causal parent.
    """

    if type(native_stop) is not bool or type(has_recovery_parent) is not bool:
        raise TypeError("native_stop and has_recovery_parent must be bool")
    if native_stop:
        return (False, False, True, has_recovery_parent)
    return (True, True, True, has_recovery_parent)


def bounded_adjustment(raw_residual: Sequence[float]) -> np.ndarray:
    value = np.asarray(raw_residual, dtype=np.float32)
    if value.shape != (2,) or not np.isfinite(value).all():
        raise ActionContractError("raw_residual must be finite [2]")
    return np.tanh(value) * RESIDUAL_LIMITS


def _clip_primitive(forward: float, lateral: float, yaw: float) -> tuple[float, float, float]:
    if not all(math.isfinite(item) for item in (forward, lateral, yaw)):
        raise ActionContractError("primitive contains non-finite values")
    return (
        min(FORWARD_RANGE_M[1], max(FORWARD_RANGE_M[0], float(forward))),
        float(lateral),
        min(YAW_RANGE_RAD[1], max(YAW_RANGE_RAD[0], float(yaw))),
    )


def clip_reference_waypoint(reference_waypoint: Sequence[float]) -> tuple[float, float, float]:
    """Return the exact executor-bounded native waypoint used by KEEP.

    The raw LightNav trajectory remains an observation.  Actor actions and
    critic actions are anchored to this executable projection so KEEP and an
    ADJUST residual share one unambiguous base action.
    """

    return _clip_primitive(*_se2(reference_waypoint, "reference_waypoint"))


def _turn_prefix(angle: float, budget: int) -> list[tuple[float, float, float]]:
    result: list[tuple[float, float, float]] = []
    remaining = math.atan2(math.sin(angle), math.cos(angle))
    while abs(remaining) > 1.0e-6 and len(result) < budget:
        step = min(YAW_RANGE_RAD[1], max(YAW_RANGE_RAD[0], remaining))
        result.append((0.0, 0.0, step))
        remaining -= step
    return result


def compile_recovery_parent(
    recovery_parent_waypoint: Sequence[float],
    *,
    max_primitives: int = MAX_RECOVERY_PRIMITIVES,
) -> tuple[tuple[float, float, float], ...]:
    """Compile a causal parent waypoint into turn/forward/turn primitives.

    ``recovery_parent_waypoint`` is relative robot-frame ``(forward, lateral,
    yaw)``.  The prefix first faces the parent, moves at most one executor step,
    then uses remaining budget to approach the parent's requested orientation.
    Long recovery paths require another actor decision and fresh observation.
    """

    if type(max_primitives) is not int or not 1 <= max_primitives <= MAX_RECOVERY_PRIMITIVES:
        raise ActionContractError("max_primitives is outside the recovery contract")
    parent = _se2(recovery_parent_waypoint, "recovery_parent_waypoint")
    distance = math.hypot(float(parent[0]), float(parent[1]))
    heading = math.atan2(float(parent[1]), float(parent[0])) if distance > 1.0e-6 else 0.0
    primitives = _turn_prefix(heading, max_primitives)
    if len(primitives) < max_primitives and distance > 1.0e-6:
        primitives.append((min(distance, FORWARD_RANGE_M[1]), 0.0, 0.0))
    if len(primitives) < max_primitives:
        remaining_yaw = math.atan2(
            math.sin(float(parent[2]) - heading), math.cos(float(parent[2]) - heading)
        )
        primitives.extend(_turn_prefix(remaining_yaw, max_primitives - len(primitives)))
    if not primitives:
        # A zero parent would alias STOP and must never receive RECOVER credit.
        raise ActionContractError("recovery parent compiles to an empty/STOP prefix")
    return tuple(_clip_primitive(*primitive) for primitive in primitives)


@dataclass(frozen=True)
class CompiledAction:
    requested_mode: int
    executed_mode: int
    raw_residual: tuple[float, float]
    bounded_residual: tuple[float, float]
    resolved_se2: tuple[float, float, float]
    primitives: tuple[tuple[float, float, float], ...]
    canonical_action: tuple[float, ...]
    recovery_parent_index: int

    def __post_init__(self) -> None:
        if self.requested_mode != self.executed_mode:
            raise ActionContractError("compiler cannot silently change a requested mode")
        if self.executed_mode not in range(MODE_COUNT):
            raise ActionContractError("executed mode is invalid")
        if not 1 <= len(self.primitives) <= MAX_RECOVERY_PRIMITIVES:
            raise ActionContractError("compiled action has an invalid primitive count")
        if len(self.canonical_action) != CANONICAL_ACTION_DIM:
            raise ActionContractError("canonical action has the wrong width")


def compile_action(
    *,
    mode: int,
    valid_modes: Sequence[bool],
    reference_waypoint: Sequence[float],
    raw_residual: Sequence[float] = (0.0, 0.0),
    recovery_parent_waypoint: Sequence[float] | None = None,
    recovery_parent_index: int = -1,
) -> CompiledAction:
    if type(mode) is not int or mode not in range(MODE_COUNT):
        raise ActionContractError("mode is outside the action contract")
    mask = _mode_mask(valid_modes)
    if not bool(mask[mode]):
        raise ActionContractError(f"requested mode {MODE_NAMES[mode]} is masked")
    reference = _se2(reference_waypoint, "reference_waypoint")
    raw = np.asarray(raw_residual, dtype=np.float32)
    if raw.shape != (2,) or not np.isfinite(raw).all():
        raise ActionContractError("raw_residual must be finite [2]")
    bounded = bounded_adjustment(raw) if mode == ADJUST else np.zeros((2,), dtype=np.float32)

    executable_reference = clip_reference_waypoint(reference)
    if mode == KEEP:
        resolved = executable_reference
        primitives = (resolved,)
        parent_index = -1
    elif mode == ADJUST:
        resolved = _clip_primitive(
            float(executable_reference[0] + bounded[0]),
            float(executable_reference[1]),
            float(executable_reference[2] + bounded[1]),
        )
        primitives = (resolved,)
        parent_index = -1
    elif mode == STOP:
        resolved = (0.0, 0.0, 0.0)
        primitives = (resolved,)
        parent_index = -1
    else:
        if recovery_parent_waypoint is None:
            raise ActionContractError("RECOVER requires a causal parent waypoint")
        if type(recovery_parent_index) is not int or recovery_parent_index < 0:
            raise ActionContractError("RECOVER requires a non-negative parent index")
        primitives = compile_recovery_parent(recovery_parent_waypoint)
        resolved = tuple(float(item) for item in _se2(recovery_parent_waypoint, "parent"))
        parent_index = recovery_parent_index

    one_hot = [0.0] * MODE_COUNT
    one_hot[mode] = 1.0
    canonical = tuple(one_hot + [float(item) for item in resolved])
    return CompiledAction(
        requested_mode=mode,
        executed_mode=mode,
        raw_residual=(float(raw[0]), float(raw[1])),
        bounded_residual=(float(bounded[0]), float(bounded[1])),
        resolved_se2=tuple(float(item) for item in resolved),
        primitives=primitives,
        canonical_action=canonical,
        recovery_parent_index=parent_index,
    )


__all__ = [
    "ADJUST",
    "CANONICAL_ACTION_DIM",
    "CompiledAction",
    "FORWARD_RANGE_M",
    "KEEP",
    "MAX_RECOVERY_PRIMITIVES",
    "MODE_COUNT",
    "MODE_NAMES",
    "RECOVER",
    "RESIDUAL_LIMITS",
    "STOP",
    "YAW_RANGE_RAD",
    "ActionContractError",
    "bounded_adjustment",
    "clip_reference_waypoint",
    "compile_action",
    "compile_recovery_parent",
    "valid_mode_mask",
]
