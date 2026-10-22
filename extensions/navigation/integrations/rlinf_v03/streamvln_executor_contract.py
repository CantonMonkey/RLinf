"""Causal lower-layer execution feedback contract for StreamVLN G7.16.

The high-level policy proposes a local waypoint and one of NAVIGATE/STOP/RECOVER.
The lower layer compiles that command to Habitat primitives and reports only
measurements available after each primitive executes.  Simulator oracle state,
navmesh queries and future outcomes are deliberately absent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping, Sequence


SCHEMA_VERSION = 1
ACTION_CHUNK = 4
STOP_ACTION = 0
FORWARD_ACTION = 1
LEFT_ACTION = 2
RIGHT_ACTION = 3
ACTION_NAMES = ("stop", "forward", "left", "right")
MODE_NAMES = ("navigate", "stop", "recover")
END_REASONS = frozenset(("running", "stop", "terminated", "truncated"))
FORWARD_STEP_M = 0.25
TURN_DEGREES = 15.0
BLOCKED_RATIO_THRESHOLD = 0.5

# These values can exist inside a simulator or reward implementation, but they
# must never enter the high-level policy observation or causal execution state.
PRIVILEGED_POLICY_KEYS = frozenset({
    "navmesh",
    "navigable",
    "oracle_action",
    "oracle_waypoint",
    "shortest_path",
    "geodesic_distance",
    "distance_to_goal",
    "future_collision",
    "future_outcome",
    "severity_label",
    "success",
    "spl",
})


def _finite(value, name: str, *, lower: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or lower is not None and result < lower:
        raise ValueError(f"{name} is outside its contract")
    return result


def assert_history_only_keys(keys: Sequence[str]) -> None:
    overlap = set(keys) & PRIVILEGED_POLICY_KEYS
    if overlap:
        raise ValueError(f"privileged policy keys are forbidden: {sorted(overlap)}")


@dataclass(frozen=True)
class HighLevelCommand:
    decision_index: int
    mode: str
    waypoint_forward_m: float
    waypoint_lateral_m: float

    def __post_init__(self) -> None:
        if type(self.decision_index) is not int or self.decision_index < 0:
            raise ValueError("decision_index must be a nonnegative integer")
        if self.mode not in MODE_NAMES:
            raise ValueError("mode must be navigate, stop or recover")
        _finite(self.waypoint_forward_m, "waypoint_forward_m")
        _finite(self.waypoint_lateral_m, "waypoint_lateral_m")
        if self.mode == "stop" and (
            abs(self.waypoint_forward_m) > 1e-8
            or abs(self.waypoint_lateral_m) > 1e-8
        ):
            raise ValueError("STOP requires a zero local waypoint")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PrimitiveExecution:
    primitive_index: int
    action: int
    commanded_forward_m: float
    commanded_yaw_degrees: float
    realized_path_m: float
    realized_forward_ratio: float | None
    collision: bool
    blocked: bool
    environment_reward: float
    terminated: bool
    truncated: bool

    def __post_init__(self) -> None:
        if type(self.primitive_index) is not int or not 0 <= self.primitive_index < ACTION_CHUNK:
            raise ValueError("primitive_index must be in 0..3")
        if type(self.action) is not int or not 0 <= self.action < len(ACTION_NAMES):
            raise ValueError("action must be an integer in 0..3")
        for name in ("collision", "blocked", "terminated", "truncated"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be boolean")
        if self.terminated and self.truncated:
            raise ValueError("a primitive cannot terminate and truncate together")
        _finite(self.commanded_forward_m, "commanded_forward_m", lower=0.0)
        _finite(self.commanded_yaw_degrees, "commanded_yaw_degrees")
        _finite(self.realized_path_m, "realized_path_m", lower=0.0)
        _finite(self.environment_reward, "environment_reward")
        if self.action == FORWARD_ACTION:
            ratio = _finite(self.realized_forward_ratio, "realized_forward_ratio", lower=0.0)
            expected = self.realized_path_m / FORWARD_STEP_M
            if not math.isclose(ratio, expected, rel_tol=0.0, abs_tol=1e-8):
                raise ValueError("realized forward ratio disagrees with path increment")
            if self.blocked != (ratio < BLOCKED_RATIO_THRESHOLD):
                raise ValueError("blocked must use the fixed post-action progress rule")
        elif self.realized_forward_ratio is not None or self.blocked:
            raise ValueError("only forward primitives can carry forward blockage")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ChunkExecutionFeedback:
    schema_version: int
    proposed_actions: tuple[int, ...]
    executed_actions: tuple[int, ...]
    executed_mask: tuple[bool, ...]
    primitives: tuple[PrimitiveExecution, ...]
    commanded_forward_m: float
    realized_path_m: float
    collision_count: int
    blocked_forward_count: int
    environment_reward_sum: float
    end_reason: str

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported executor feedback schema")
        if not 1 <= len(self.proposed_actions) <= ACTION_CHUNK:
            raise ValueError("proposed action count must be in 1..4")
        if any(type(action) is not int or not 0 <= action < len(ACTION_NAMES)
               for action in self.proposed_actions):
            raise ValueError("proposed actions must be integer categories in 0..3")
        if not 1 <= len(self.executed_actions) <= len(self.proposed_actions):
            raise ValueError("executed actions must be a nonempty proposal prefix")
        if tuple(self.proposed_actions[: len(self.executed_actions)]) != self.executed_actions:
            raise ValueError("executed actions differ from the proposal prefix")
        if len(self.executed_mask) != ACTION_CHUNK or self.executed_mask != tuple(
            index < len(self.executed_actions) for index in range(ACTION_CHUNK)
        ):
            raise ValueError("executed mask must be a four-wide contiguous prefix")
        if tuple(item.action for item in self.primitives) != self.executed_actions:
            raise ValueError("primitive records differ from executed actions")
        if self.end_reason not in END_REASONS:
            raise ValueError("unknown executor end reason")
        if self.end_reason == "stop" and self.executed_actions[-1] != STOP_ACTION:
            raise ValueError("stop end reason requires an executed STOP")
        if self.end_reason == "terminated" and not self.primitives[-1].terminated:
            raise ValueError("terminated reason requires terminal feedback")
        if self.end_reason == "truncated" and not self.primitives[-1].truncated:
            raise ValueError("truncated reason requires truncation feedback")
        _finite(self.commanded_forward_m, "commanded_forward_m", lower=0.0)
        _finite(self.realized_path_m, "realized_path_m", lower=0.0)
        _finite(self.environment_reward_sum, "environment_reward_sum")
        if type(self.collision_count) is not int or self.collision_count < 0:
            raise ValueError("collision_count must be a nonnegative integer")
        if type(self.blocked_forward_count) is not int or self.blocked_forward_count < 0:
            raise ValueError("blocked_forward_count must be a nonnegative integer")
        if not math.isclose(
            self.commanded_forward_m,
            sum(row.commanded_forward_m for row in self.primitives),
            rel_tol=0.0,
            abs_tol=1e-8,
        ):
            raise ValueError("chunk commanded distance aggregate mismatch")
        if not math.isclose(
            self.realized_path_m,
            sum(row.realized_path_m for row in self.primitives),
            rel_tol=0.0,
            abs_tol=1e-8,
        ):
            raise ValueError("chunk realized distance aggregate mismatch")
        if self.collision_count != sum(row.collision for row in self.primitives):
            raise ValueError("chunk collision aggregate mismatch")
        if self.blocked_forward_count != sum(row.blocked for row in self.primitives):
            raise ValueError("chunk blockage aggregate mismatch")
        if not math.isclose(
            self.environment_reward_sum,
            sum(row.environment_reward for row in self.primitives),
            rel_tol=0.0,
            abs_tol=1e-7,
        ):
            raise ValueError("chunk reward aggregate mismatch")

    def discounted_environment_reward(self, gamma: float) -> float:
        gamma = _finite(gamma, "gamma", lower=0.0)
        if gamma > 1.0:
            raise ValueError("gamma must be in [0, 1]")
        return sum(
            gamma**index * row.environment_reward
            for index, row in enumerate(self.primitives)
        )

    def to_dict(self) -> dict:
        result = asdict(self)
        result["proposed_actions"] = list(self.proposed_actions)
        result["executed_actions"] = list(self.executed_actions)
        result["executed_mask"] = list(self.executed_mask)
        result["primitives"] = [row.to_dict() for row in self.primitives]
        return result


def primitive_execution(
    *,
    primitive_index: int,
    action: int,
    previous_path_length_m: float,
    current_path_length_m: float,
    collision: bool,
    environment_reward: float,
    terminated: bool,
    truncated: bool,
) -> PrimitiveExecution:
    previous = _finite(previous_path_length_m, "previous_path_length_m", lower=0.0)
    current = _finite(current_path_length_m, "current_path_length_m", lower=0.0)
    delta = current - previous
    if delta < -1e-8:
        raise ValueError("path length decreased inside an episode")
    delta = max(0.0, delta)
    if type(action) is not int or not 0 <= action < len(ACTION_NAMES):
        raise ValueError("action must be an integer in 0..3")
    commanded_forward = FORWARD_STEP_M if action == FORWARD_ACTION else 0.0
    commanded_yaw = (
        TURN_DEGREES if action == LEFT_ACTION
        else -TURN_DEGREES if action == RIGHT_ACTION
        else 0.0
    )
    ratio = delta / FORWARD_STEP_M if action == FORWARD_ACTION else None
    blocked = bool(action == FORWARD_ACTION and ratio < BLOCKED_RATIO_THRESHOLD)
    return PrimitiveExecution(
        primitive_index=primitive_index,
        action=action,
        commanded_forward_m=commanded_forward,
        commanded_yaw_degrees=commanded_yaw,
        realized_path_m=delta,
        realized_forward_ratio=ratio,
        collision=collision,
        blocked=blocked,
        environment_reward=environment_reward,
        terminated=terminated,
        truncated=truncated,
    )


def chunk_feedback(
    proposed_actions: Sequence[int],
    primitives: Sequence[PrimitiveExecution],
) -> ChunkExecutionFeedback:
    proposed = tuple(int(value) for value in proposed_actions)
    rows = tuple(primitives)
    if not rows:
        raise ValueError("chunk feedback requires at least one executed primitive")
    executed = tuple(row.action for row in rows)
    final = rows[-1]
    if final.truncated:
        reason = "truncated"
    elif final.terminated and final.action != STOP_ACTION:
        reason = "terminated"
    elif final.action == STOP_ACTION:
        reason = "stop"
    else:
        reason = "running"
    return ChunkExecutionFeedback(
        schema_version=SCHEMA_VERSION,
        proposed_actions=proposed,
        executed_actions=executed,
        executed_mask=tuple(index < len(executed) for index in range(ACTION_CHUNK)),
        primitives=rows,
        commanded_forward_m=sum(row.commanded_forward_m for row in rows),
        realized_path_m=sum(row.realized_path_m for row in rows),
        collision_count=sum(row.collision for row in rows),
        blocked_forward_count=sum(row.blocked for row in rows),
        environment_reward_sum=sum(row.environment_reward for row in rows),
        end_reason=reason,
    )


def feedback_from_dict(raw: Mapping) -> ChunkExecutionFeedback:
    if not isinstance(raw, Mapping):
        raise TypeError("executor feedback must be a mapping")
    expected = {
        "schema_version", "proposed_actions", "executed_actions", "executed_mask",
        "primitives", "commanded_forward_m", "realized_path_m", "collision_count",
        "blocked_forward_count", "environment_reward_sum", "end_reason",
    }
    if set(raw) != expected or not isinstance(raw["primitives"], list):
        raise ValueError("executor feedback schema mismatch")
    primitive_fields = set(PrimitiveExecution.__dataclass_fields__)
    rows = []
    for item in raw["primitives"]:
        if not isinstance(item, Mapping) or set(item) != primitive_fields:
            raise ValueError("primitive feedback schema mismatch")
        rows.append(PrimitiveExecution(**item))
    return ChunkExecutionFeedback(
        schema_version=raw["schema_version"],
        proposed_actions=tuple(raw["proposed_actions"]),
        executed_actions=tuple(raw["executed_actions"]),
        executed_mask=tuple(raw["executed_mask"]),
        primitives=tuple(rows),
        commanded_forward_m=raw["commanded_forward_m"],
        realized_path_m=raw["realized_path_m"],
        collision_count=raw["collision_count"],
        blocked_forward_count=raw["blocked_forward_count"],
        environment_reward_sum=raw["environment_reward_sum"],
        end_reason=raw["end_reason"],
    )
