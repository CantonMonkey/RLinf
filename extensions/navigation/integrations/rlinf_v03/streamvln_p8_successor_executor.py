"""Feedback-aware one-primitive executor for the P8 successor."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Callable

if __package__:
    from .streamvln_p8_successor_action import (
        ACTION_CHUNK,
        FORWARD_ACTION,
        STOP_ACTION,
        CompiledRoute,
    )
else:  # Explicit direct-file mode for local contract tests only.
    from streamvln_p8_successor_action import (
        ACTION_CHUNK,
        FORWARD_ACTION,
        STOP_ACTION,
        CompiledRoute,
    )


PRODUCTION_SUCCESSOR_EXECUTION_SCHEMA_NAME = "streamvln_p8_successor_execution"
PURE_SUCCESSOR_EXECUTION_SCHEMA_NAME = "streamvln_p8_successor_pure_execution"
SUCCESSOR_EXECUTION_SCHEMA_VERSION = 1


def successor_end_reason(
    *,
    action: int,
    collision: bool,
    blocked: bool,
    terminated: bool,
    truncated: bool,
    plan_complete: bool,
) -> tuple[str | None, bool]:
    """Apply one shared terminal/replan precedence for tests and Habitat."""

    if truncated:
        return "truncated", False
    if terminated and action != STOP_ACTION:
        return "terminated", False
    if action == STOP_ACTION:
        return "stop", False
    if collision:
        return "collision", True
    if blocked:
        return "blocked", True
    if plan_complete:
        return "plan_complete", True
    return None, False


@dataclass(frozen=True)
class PrimitiveOutcome:
    action: int
    collision: bool = False
    blocked: bool = False
    terminated: bool = False
    truncated: bool = False
    realized_path_m: float = 0.0
    environment_reward: float = 0.0

    def __post_init__(self) -> None:
        if type(self.action) is not int or not 0 <= self.action < 4:
            raise ValueError("primitive outcome action must be in 0..3")
        for name in ("collision", "blocked", "terminated", "truncated"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be boolean")
        if self.terminated and self.truncated:
            raise ValueError("one primitive cannot terminate and truncate together")
        if self.blocked and self.action != FORWARD_ACTION:
            raise ValueError("only a forward primitive can be blocked")
        if not math.isfinite(self.realized_path_m) or self.realized_path_m < 0.0:
            raise ValueError("realized_path_m must be finite and nonnegative")
        if not math.isfinite(self.environment_reward):
            raise ValueError("environment_reward must be finite")


@dataclass(frozen=True)
class CausalExecution:
    plan: CompiledRoute
    executed_actions: tuple[int, ...]
    executed_mask: tuple[bool, ...]
    outcomes: tuple[PrimitiveOutcome, ...]
    end_reason: str
    replan_required: bool

    def __post_init__(self) -> None:
        if not self.executed_actions:
            raise ValueError("successor executor must execute one primitive")
        if not 1 <= len(self.executed_actions) <= len(self.plan.proposed_actions):
            raise ValueError("successor execution must be a nonempty proposal prefix")
        if len(self.executed_actions) != len(self.outcomes):
            raise ValueError("successor execution and outcome lengths differ")
        if self.executed_actions != tuple(outcome.action for outcome in self.outcomes):
            raise ValueError("execution outcome does not match the executed prefix")
        if self.executed_actions != self.plan.proposed_actions[: len(self.executed_actions)]:
            raise ValueError("executed action must be a prefix of the compiled plan")
        expected_mask = tuple(index < len(self.executed_actions) for index in range(ACTION_CHUNK))
        if self.executed_mask != expected_mask:
            raise ValueError("successor executed mask must contain one causal primitive")

    def to_trace(self) -> dict:
        return {
            "schema_name": PURE_SUCCESSOR_EXECUTION_SCHEMA_NAME,
            "schema_version": SUCCESSOR_EXECUTION_SCHEMA_VERSION,
            "decision_index": self.plan.decision_index,
            "requested_route": self.plan.requested_route.value,
            "executed_route": self.plan.executed_route.value,
            "source": self.plan.source,
            "recovery_override": self.plan.recovery_override,
            "recovery_parent_index": self.plan.recovery_parent_index,
            "waypoint_local_m": list(self.plan.waypoint_local_m),
            "target_distance_m": self.plan.target_distance_m,
            "target_bearing_rad": self.plan.target_bearing_rad,
            "compiled_actions": list(self.plan.proposed_actions),
            "executed_actions": list(self.executed_actions),
            "executed_mask": list(self.executed_mask),
            "primitive_outcomes": [asdict(outcome) for outcome in self.outcomes],
            "end_reason": self.end_reason,
            "replan_required": self.replan_required,
        }


def execute_feedback_aware(
    plan: CompiledRoute,
    primitive_step: Callable[[int, int], PrimitiveOutcome],
) -> CausalExecution:
    """Execute a variable prefix with feedback after each primitive.

    Collision, blockage, termination or truncation discards the unexecuted tail.
    A fully completed proposal returns to the policy for a fresh metric waypoint.
    """

    outcomes = []
    end_reason = "plan_complete"
    replan = True
    for index, raw_action in enumerate(plan.proposed_actions):
        action = raw_action
        outcome = primitive_step(action, index)
        if not isinstance(outcome, PrimitiveOutcome):
            raise TypeError("primitive_step must return PrimitiveOutcome")
        if outcome.action != action:
            raise ValueError("primitive_step reported a different action")
        outcomes.append(outcome)
        reason, should_replan = successor_end_reason(
            action=action,
            collision=outcome.collision,
            blocked=outcome.blocked,
            terminated=outcome.terminated,
            truncated=outcome.truncated,
            plan_complete=index == len(plan.proposed_actions) - 1,
        )
        if reason is not None:
            end_reason, replan = reason, should_replan
            break
    executed = tuple(outcome.action for outcome in outcomes)
    return CausalExecution(
        plan=plan,
        executed_actions=executed,
        executed_mask=tuple(index < len(executed) for index in range(ACTION_CHUNK)),
        outcomes=tuple(outcomes),
        end_reason=end_reason,
        replan_required=replan,
    )
