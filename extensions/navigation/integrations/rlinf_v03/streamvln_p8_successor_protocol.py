"""Typed worker protocol for one-primitive P8 successor execution.

The protocol is deliberately isolated from RLinf's generic rollout dataclasses.
The dedicated successor worker pair and actor replay import these records while
leaving baseline rollout serialization unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence


SUCCESSOR_REPLAY_SCHEMA = "streamvln-p8-successor-decision-v2-topology"

try:
    from .streamvln_chunk_ppo import variable_duration_gae
    from .streamvln_p8_successor_action import (
        ACTION_CHUNK,
        ACTION_COUNT,
        CompiledRoute,
        FORWARD_ACTION,
        STOP_ACTION,
        RouteDecision,
        RouteName,
    )
    from .streamvln_p8_successor_executor import successor_end_reason
except ImportError:  # Direct staged-file tests.
    from streamvln_chunk_ppo import variable_duration_gae
    from streamvln_p8_successor_action import (
        ACTION_CHUNK,
        ACTION_COUNT,
        CompiledRoute,
        FORWARD_ACTION,
        STOP_ACTION,
        RouteDecision,
        RouteName,
    )
    from streamvln_p8_successor_executor import successor_end_reason


def _validate_identity(episode_token: str, physical_step: int, decision_index: int) -> None:
    if not isinstance(episode_token, str) or not episode_token:
        raise ValueError("episode_token must be a non-empty string")
    for name, value in (
        ("physical_step", physical_step),
        ("decision_index", decision_index),
    ):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")


def _validate_action(name: str, action: int) -> None:
    if type(action) is not int or not 0 <= action < ACTION_COUNT:
        raise ValueError(f"{name} must be a Habitat action in [0, {ACTION_COUNT - 1}]")


@dataclass(frozen=True)
class SuccessorDecisionSample:
    """Immutable actor sample retained until its complete macro row closes."""

    decision: RouteDecision
    proposed_actions: tuple[int, ...]
    logprob: float
    value: float
    official_shadow_action: int
    forward_context: object | None = None
    expert_route_target: RouteName | None = None
    expert_waypoint_target: tuple[float, float] | None = None
    compiled_route: CompiledRoute | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.decision, RouteDecision):
            raise ValueError("decision must be a RouteDecision")
        if self.decision.episode_token is None or self.decision.physical_step is None:
            raise ValueError("decision must carry episode_token and physical_step")
        if not isinstance(self.proposed_actions, tuple) or not 1 <= len(
            self.proposed_actions
        ) <= ACTION_CHUNK:
            raise ValueError(f"proposed_actions must be a tuple of 1..{ACTION_CHUNK} actions")
        for action in self.proposed_actions:
            _validate_action("proposed action", action)
        if STOP_ACTION in self.proposed_actions and self.proposed_actions != (STOP_ACTION,):
            raise ValueError("STOP must be the only proposed action")
        if not math.isfinite(float(self.logprob)) or not math.isfinite(float(self.value)):
            raise ValueError("logprob and value must be finite")
        _validate_action("official_shadow_action", self.official_shadow_action)
        if self.compiled_route is not None:
            if not isinstance(self.compiled_route, CompiledRoute):
                raise TypeError("compiled_route must be a CompiledRoute")
            if (
                self.compiled_route.decision_index != self.decision.decision_index
                or self.compiled_route.requested_route is not self.decision.route
                or self.compiled_route.proposed_actions != self.proposed_actions
            ):
                raise ValueError("compiled route provenance disagrees with policy sample")
        if self.expert_route_target is None:
            if self.expert_waypoint_target is not None:
                raise ValueError("expert waypoint requires an explicit expert route")
        else:
            expert_route = RouteName(self.expert_route_target)
            if expert_route is RouteName.RECOVER_PARENT:
                raise ValueError("RECOVER_PARENT cannot be a policy expert target")
            object.__setattr__(self, "expert_route_target", expert_route)
            if expert_route is RouteName.WAYPOINT:
                if (
                    not isinstance(self.expert_waypoint_target, tuple)
                    or len(self.expert_waypoint_target) != 2
                    or any(not math.isfinite(float(value)) for value in self.expert_waypoint_target)
                ):
                    raise ValueError("WAYPOINT expert requires a finite 2D target")
            elif self.expert_waypoint_target not in (None, (0.0, 0.0)):
                raise ValueError("KEEP/STOP expert targets cannot carry a waypoint")


@dataclass(frozen=True)
class SuccessorStepRequest:
    """One rollout-to-environment request for exactly one physical primitive."""

    episode_token: str
    physical_step: int
    decision_index: int
    needs_decision: bool
    decision_mask: bool
    official_shadow_action: int
    decision_sample: SuccessorDecisionSample | None = None

    def __post_init__(self) -> None:
        _validate_identity(self.episode_token, self.physical_step, self.decision_index)
        if type(self.needs_decision) is not bool or type(self.decision_mask) is not bool:
            raise ValueError("needs_decision and decision_mask must be bools")
        if self.needs_decision != self.decision_mask:
            raise ValueError("needs_decision and decision_mask must agree")
        _validate_action("official_shadow_action", self.official_shadow_action)
        if self.needs_decision != (self.decision_sample is not None):
            raise ValueError("decision_sample must be present exactly at decision boundaries")
        if self.decision_sample is not None:
            decision = self.decision_sample.decision
            if (
                decision.episode_token != self.episode_token
                or decision.physical_step != self.physical_step
                or decision.decision_index != self.decision_index
            ):
                raise ValueError("decision sample identity does not match request identity")
            if self.decision_sample.official_shadow_action != self.official_shadow_action:
                raise ValueError("decision sample must retain the boundary official shadow action")


@dataclass(frozen=True)
class SuccessorStepResult:
    """Environment feedback for the single primitive named by a request."""

    episode_token: str
    physical_step: int
    decision_index: int
    needs_decision: bool
    decision_mask: bool
    executed_action: int
    reward: float
    collision: bool = False
    blocked: bool = False
    terminated: bool = False
    truncated: bool = False
    realized_path_m: float = 0.0

    def __post_init__(self) -> None:
        _validate_identity(self.episode_token, self.physical_step, self.decision_index)
        if type(self.needs_decision) is not bool or type(self.decision_mask) is not bool:
            raise ValueError("needs_decision and decision_mask must be bools")
        _validate_action("executed_action", self.executed_action)
        if not math.isfinite(float(self.reward)):
            raise ValueError("reward must be finite")
        if not math.isfinite(float(self.realized_path_m)) or self.realized_path_m < 0.0:
            raise ValueError("realized_path_m must be finite and non-negative")
        for name in ("collision", "blocked", "terminated", "truncated"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a bool")
        if self.terminated and self.truncated:
            raise ValueError("a primitive cannot be both terminated and truncated")
        if self.executed_action == STOP_ACTION and not (self.terminated or self.truncated):
            raise ValueError("STOP feedback must terminate or truncate the episode")


@dataclass(frozen=True)
class PrimitiveCommand:
    """Validated physical command emitted by the EnvWorker accumulator."""

    episode_token: str
    physical_step: int
    decision_index: int
    needs_decision: bool
    decision_mask: bool
    executed_action: int
    official_stop_override: bool


@dataclass(frozen=True)
class MacroTransition:
    """One replay row at a learned decision boundary."""

    episode_token: str
    physical_step: int
    decision_index: int
    decision_mask: bool
    decision_sample: SuccessorDecisionSample
    compiled_route: CompiledRoute
    executed_actions: tuple[int, ...]
    executed_mask: tuple[bool, ...]
    primitive_rewards: tuple[float, ...]
    primitive_collisions: tuple[bool, ...]
    primitive_blocked: tuple[bool, ...]
    realized_path_m: tuple[float, ...]
    duration: int
    terminated: bool
    truncated: bool
    end_reason: str
    replan_required: bool

    def __post_init__(self) -> None:
        _validate_identity(self.episode_token, self.physical_step, self.decision_index)
        if self.decision_mask is not True:
            raise ValueError("a macro transition must represent one decision boundary")
        if not isinstance(self.decision_sample, SuccessorDecisionSample):
            raise ValueError("macro transition must retain its decision sample")
        if not isinstance(self.compiled_route, CompiledRoute):
            raise ValueError("macro transition must retain its compiled route")
        decision = self.decision_sample.decision
        if (
            decision.episode_token != self.episode_token
            or decision.physical_step != self.physical_step
            or decision.decision_index != self.decision_index
        ):
            raise ValueError("macro transition and decision sample identities differ")
        if (
            self.decision_sample.compiled_route != self.compiled_route
            or self.compiled_route.decision_index != self.decision_index
            or self.compiled_route.requested_route is not decision.route
        ):
            raise ValueError("macro transition compiled route provenance drifted")
        if len(self.executed_actions) != ACTION_CHUNK:
            raise ValueError("executed_actions must use the fixed action-chunk width")
        if len(self.executed_mask) != ACTION_CHUNK:
            raise ValueError("executed_mask must use the fixed action-chunk width")
        if len(self.primitive_rewards) != ACTION_CHUNK:
            raise ValueError("primitive_rewards must use the fixed action-chunk width")
        if len(self.primitive_collisions) != ACTION_CHUNK:
            raise ValueError("primitive_collisions must use the fixed action-chunk width")
        if len(self.primitive_blocked) != ACTION_CHUNK:
            raise ValueError("primitive_blocked must use the fixed action-chunk width")
        if len(self.realized_path_m) != ACTION_CHUNK:
            raise ValueError("realized_path_m must use the fixed action-chunk width")
        if not 1 <= self.duration <= ACTION_CHUNK:
            raise ValueError("duration must be within the action-chunk width")
        if self.duration != sum(bool(value) for value in self.executed_mask):
            raise ValueError("duration must equal the executed-mask prefix length")
        if self.executed_mask != (True,) * self.duration + (False,) * (
            ACTION_CHUNK - self.duration
        ):
            raise ValueError("executed_mask must be prefix-contiguous")
        for action in self.executed_actions:
            _validate_action("executed action", action)
        if any(not math.isfinite(float(value)) for value in self.primitive_rewards):
            raise ValueError("primitive_rewards must be finite")
        if any(type(value) is not bool for value in self.primitive_collisions):
            raise ValueError("primitive_collisions must contain bools")
        if any(type(value) is not bool for value in self.primitive_blocked):
            raise ValueError("primitive_blocked must contain bools")
        if any(
            not math.isfinite(float(value)) or value < 0.0
            for value in self.realized_path_m
        ):
            raise ValueError("realized_path_m must be finite and non-negative")
        if any(action != STOP_ACTION for action in self.executed_actions[self.duration :]):
            raise ValueError("padded actions must use STOP")
        if any(value != 0.0 for value in self.primitive_rewards[self.duration :]):
            raise ValueError("padded primitive rewards must be zero")
        if any(self.primitive_collisions[self.duration :]) or any(
            self.primitive_blocked[self.duration :]
        ):
            raise ValueError("padded primitive feedback flags must be false")
        if any(value != 0.0 for value in self.realized_path_m[self.duration :]):
            raise ValueError("padded realized path increments must be zero")
        if any(
            blocked and action != FORWARD_ACTION
            for action, blocked in zip(
                self.executed_actions[: self.duration],
                self.primitive_blocked[: self.duration],
            )
        ):
            raise ValueError("blocked feedback applies only to forward primitives")
        proposed = self.decision_sample.proposed_actions
        if self.duration > len(proposed):
            raise ValueError("duration cannot exceed the sampled action prefix")
        active = self.executed_actions[: self.duration]
        if self.end_reason == "official_stop_override":
            if active[-1] != STOP_ACTION or active[:-1] != proposed[: self.duration - 1]:
                raise ValueError("official STOP may only replace the final executed primitive")
        elif active != proposed[: self.duration]:
            raise ValueError("executed prefix must match the sampled action prefix")
        if type(self.terminated) is not bool or type(self.truncated) is not bool:
            raise ValueError("terminal flags must be bools")
        if self.terminated and self.truncated:
            raise ValueError("a macro cannot be both terminated and truncated")
        if not isinstance(self.end_reason, str) or not self.end_reason:
            raise ValueError("end_reason must be a non-empty string")
        if type(self.replan_required) is not bool:
            raise ValueError("replan_required must be a bool")

    def discounted_reward(self, gamma: float) -> float:
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        return sum((gamma**index) * self.primitive_rewards[index] for index in range(self.duration))


def _split_items(items: tuple, sizes: Sequence[int]) -> tuple[tuple, ...]:
    if not sizes or any(type(size) is not int or size < 1 for size in sizes):
        raise ValueError("split sizes must be positive integers")
    if sum(sizes) != len(items):
        raise ValueError("split sizes must sum to the batch size")
    shards = []
    offset = 0
    for size in sizes:
        shards.append(items[offset : offset + size])
        offset += size
    return tuple(shards)


@dataclass(frozen=True)
class SuccessorRequestBatch:
    """Transport batch with lossless shard split/merge operations."""

    items: tuple[SuccessorStepRequest, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.items, tuple) or not self.items or any(
            not isinstance(item, SuccessorStepRequest) for item in self.items
        ):
            raise ValueError("request batch must contain typed request items")

    def split_by_sizes(self, sizes: Sequence[int]) -> tuple["SuccessorRequestBatch", ...]:
        return tuple(SuccessorRequestBatch(items) for items in _split_items(self.items, sizes))

    @classmethod
    def merge_shards(cls, shards: Iterable["SuccessorRequestBatch"]) -> "SuccessorRequestBatch":
        shard_list = tuple(shards)
        if not shard_list or any(not isinstance(shard, cls) for shard in shard_list):
            raise ValueError("request shards must be non-empty typed batches")
        return cls(tuple(item for shard in shard_list for item in shard.items))


@dataclass(frozen=True)
class SuccessorResultBatch:
    """Feedback batch with lossless shard split/merge operations."""

    items: tuple[SuccessorStepResult, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.items, tuple) or not self.items or any(
            not isinstance(item, SuccessorStepResult) for item in self.items
        ):
            raise ValueError("result batch must contain typed result items")

    def split_by_sizes(self, sizes: Sequence[int]) -> tuple["SuccessorResultBatch", ...]:
        return tuple(SuccessorResultBatch(items) for items in _split_items(self.items, sizes))

    @classmethod
    def merge_shards(cls, shards: Iterable["SuccessorResultBatch"]) -> "SuccessorResultBatch":
        shard_list = tuple(shards)
        if not shard_list or any(not isinstance(shard, cls) for shard in shard_list):
            raise ValueError("result shards must be non-empty typed batches")
        return cls(tuple(item for shard in shard_list for item in shard.items))


class SuccessorMacroAccumulator:
    """EnvWorker-owned pending-prefix state with one feedback per round trip."""

    def __init__(
        self,
        episode_token: str,
        *,
        physical_step: int = 0,
        decision_index: int = 0,
    ) -> None:
        _validate_identity(episode_token, physical_step, decision_index)
        self._episode_token = episode_token
        self._physical_step = physical_step
        self._decision_index = decision_index
        self._clear_macro()

    @property
    def episode_token(self) -> str:
        return self._episode_token

    @property
    def physical_step(self) -> int:
        return self._physical_step

    @property
    def decision_index(self) -> int:
        return self._decision_index

    @property
    def needs_decision(self) -> bool:
        return self._pending_actions is None

    @property
    def decision_mask(self) -> bool:
        return self.needs_decision

    def reset(
        self,
        episode_token: str,
        *,
        physical_step: int = 0,
        decision_index: int = 0,
    ) -> None:
        _validate_identity(episode_token, physical_step, decision_index)
        if episode_token == self._episode_token:
            raise ValueError("reset must rotate episode_token")
        self._episode_token = episode_token
        self._physical_step = physical_step
        self._decision_index = decision_index
        self._clear_macro()

    def _clear_macro(self) -> None:
        self._pending_actions: tuple[int, ...] | None = None
        self._next_action_index = 0
        self._executed_actions: list[int] = []
        self._primitive_rewards: list[float] = []
        self._primitive_collisions: list[bool] = []
        self._primitive_blocked: list[bool] = []
        self._realized_path_m: list[float] = []
        self._decision_sample: SuccessorDecisionSample | None = None
        self._macro_start_step = 0
        self._in_flight: PrimitiveCommand | None = None

    def _check_current_identity(self, record: object) -> None:
        for name, expected in (
            ("episode_token", self._episode_token),
            ("physical_step", self._physical_step),
            ("decision_index", self._decision_index),
        ):
            actual = getattr(record, name)
            if actual != expected:
                raise ValueError(f"stale {name}: expected {expected!r}, got {actual!r}")

    def prepare_step(self, request: SuccessorStepRequest) -> PrimitiveCommand:
        """Validate one request and return the only primitive that may execute."""

        if not isinstance(request, SuccessorStepRequest):
            raise TypeError("request must be SuccessorStepRequest")
        if self._in_flight is not None:
            raise RuntimeError("one primitive is already awaiting feedback")
        self._check_current_identity(request)
        if self.needs_decision:
            if not request.needs_decision or request.decision_sample is None:
                raise ValueError("a new decision requires a typed decision sample")
            self._decision_sample = request.decision_sample
            self._pending_actions = request.decision_sample.proposed_actions
            self._macro_start_step = self._physical_step
        elif request.needs_decision or request.decision_sample is not None:
            raise ValueError("a continuation must not replace the pending decision sample")

        assert self._pending_actions is not None
        proposed_action = self._pending_actions[self._next_action_index]
        official_stop_override = (
            request.official_shadow_action == STOP_ACTION and proposed_action != STOP_ACTION
        )
        executed_action = STOP_ACTION if official_stop_override else proposed_action
        self._in_flight = PrimitiveCommand(
            episode_token=self._episode_token,
            physical_step=self._physical_step,
            decision_index=self._decision_index,
            needs_decision=request.needs_decision,
            decision_mask=request.decision_mask,
            executed_action=executed_action,
            official_stop_override=official_stop_override,
        )
        return self._in_flight

    def feedback_will_close_macro(self, result: SuccessorStepResult) -> bool:
        """Return the post-feedback decision state without mutating the prefix."""

        if not isinstance(result, SuccessorStepResult):
            raise TypeError("result must be SuccessorStepResult")
        if self._in_flight is None or self._pending_actions is None:
            raise RuntimeError("feedback has no matching in-flight primitive")
        self._check_current_identity(result)
        if result.executed_action != self._in_flight.executed_action:
            raise ValueError("feedback action does not match the in-flight primitive")
        reason, _ = successor_end_reason(
            action=result.executed_action,
            collision=result.collision,
            blocked=result.blocked,
            terminated=result.terminated,
            truncated=result.truncated,
            plan_complete=self._next_action_index == len(self._pending_actions) - 1,
        )
        return reason is not None

    def accept_feedback(self, result: SuccessorStepResult) -> MacroTransition | None:
        """Consume exactly one real primitive result and optionally close a macro row."""

        if not isinstance(result, SuccessorStepResult):
            raise TypeError("result must be SuccessorStepResult")
        if self._in_flight is None:
            raise RuntimeError("feedback has no matching in-flight primitive")
        self._check_current_identity(result)
        command = self._in_flight
        if result.decision_mask != command.decision_mask:
            raise ValueError("feedback decision_mask does not match the in-flight primitive")
        if result.executed_action != command.executed_action:
            raise ValueError("feedback action does not match the in-flight primitive")

        assert self._pending_actions is not None and self._decision_sample is not None
        self._executed_actions.append(result.executed_action)
        self._primitive_rewards.append(float(result.reward))
        self._primitive_collisions.append(result.collision)
        self._primitive_blocked.append(result.blocked)
        self._realized_path_m.append(float(result.realized_path_m))
        plan_complete = self._next_action_index == len(self._pending_actions) - 1
        end_reason, replan_required = successor_end_reason(
            action=result.executed_action,
            collision=result.collision,
            blocked=result.blocked,
            terminated=result.terminated,
            truncated=result.truncated,
            plan_complete=plan_complete,
        )
        if end_reason is not None and command.official_stop_override:
            end_reason, replan_required = "official_stop_override", False
        expected_next_decision = end_reason is not None
        if result.needs_decision != expected_next_decision:
            raise ValueError(
                "feedback needs_decision does not match the post-feedback pending state"
            )

        self._in_flight = None
        self._physical_step += 1
        if end_reason is None:
            self._next_action_index += 1
            return None

        duration = len(self._executed_actions)
        if self._decision_sample.compiled_route is None:
            raise RuntimeError("successor macro lacks executor compiled-route provenance")
        transition = MacroTransition(
            episode_token=self._episode_token,
            physical_step=self._macro_start_step,
            decision_index=self._decision_index,
            decision_mask=True,
            decision_sample=self._decision_sample,
            compiled_route=self._decision_sample.compiled_route,
            executed_actions=tuple(self._executed_actions)
            + (STOP_ACTION,) * (ACTION_CHUNK - duration),
            executed_mask=(True,) * duration + (False,) * (ACTION_CHUNK - duration),
            primitive_rewards=tuple(self._primitive_rewards)
            + (0.0,) * (ACTION_CHUNK - duration),
            primitive_collisions=tuple(self._primitive_collisions)
            + (False,) * (ACTION_CHUNK - duration),
            primitive_blocked=tuple(self._primitive_blocked)
            + (False,) * (ACTION_CHUNK - duration),
            realized_path_m=tuple(self._realized_path_m)
            + (0.0,) * (ACTION_CHUNK - duration),
            duration=duration,
            terminated=result.terminated,
            truncated=result.truncated,
            end_reason=end_reason,
            replan_required=replan_required,
        )
        self._decision_index += 1
        self._clear_macro()
        return transition


def compute_successor_gae(
    chunk_rewards: Sequence[float],
    values: Sequence[float],
    next_values: Sequence[float],
    durations: Sequence[int],
    terminations: Sequence[bool],
    truncations: Sequence[bool],
    gamma: float,
    gae_lambda: float,
    *,
    duration_aware: bool = False,
) -> tuple[list[float], list[float], list[float]]:
    """Opt into physical-duration discount powers without changing baseline math."""

    if len(durations) != len(chunk_rewards):
        raise ValueError("durations and chunk_rewards must have the same length")
    if any(type(duration) is not int or duration < 1 for duration in durations):
        raise ValueError("durations must be positive integers")
    fields = (values, next_values, terminations, truncations)
    if any(len(field) != len(chunk_rewards) for field in fields):
        raise ValueError("all GAE fields must have the same length")
    if any(type(flag) is not bool for flag in (*terminations, *truncations)):
        raise ValueError("termination and truncation flags must be bools")
    if any(terminated and truncated for terminated, truncated in zip(terminations, truncations)):
        raise ValueError("a row cannot be both terminated and truncated")
    effective_durations = durations if duration_aware else [1] * len(chunk_rewards)
    if not any(truncations):
        return variable_duration_gae(
            chunk_rewards,
            values,
            next_values,
            effective_durations,
            terminations,
            gamma,
            gae_lambda,
        )
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gamma and lambda must be in [0, 1]")
    if any(not math.isfinite(float(value)) for value in (*chunk_rewards, *values, *next_values)):
        raise ValueError("GAE inputs must be finite")

    count = len(chunk_rewards)
    advantages = [0.0] * count
    deltas = [0.0] * count
    running = 0.0
    for index in range(count - 1, -1, -1):
        duration = effective_durations[index]
        bootstrap = 0.0 if terminations[index] else 1.0
        trace = 0.0 if terminations[index] or truncations[index] else 1.0
        delta = (
            float(chunk_rewards[index])
            + (gamma**duration) * bootstrap * float(next_values[index])
            - float(values[index])
        )
        running = delta + ((gamma * gae_lambda) ** duration) * trace * running
        deltas[index] = delta
        advantages[index] = running
    returns = [advantage + float(value) for advantage, value in zip(advantages, values)]
    return advantages, returns, deltas
