"""Causal route and metric-waypoint interface for the P8 successor.

The failed P8 residual remains untouched.  This versioned interface makes the
learned waypoint an executed command, preserves an opaque one-primitive
official KEEP branch, and keeps deterministic parent recovery below the policy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Sequence


STOP_ACTION = 0
FORWARD_ACTION = 1
LEFT_ACTION = 2
RIGHT_ACTION = 3
ACTION_COUNT = 4
ACTION_CHUNK = 4
HABITAT_FORWARD_STEP_M = 0.25
HABITAT_TURN_DEGREES = 15.0


class RouteName(str, Enum):
    KEEP_ORIGINAL = "keep_original"
    WAYPOINT = "waypoint"
    STOP = "stop"
    RECOVER_PARENT = "recover_parent"


class StopBranchRequired(ValueError):
    """Raised when a learned waypoint is close enough that STOP must decide."""


class ParentRecoveryUnavailable(RuntimeError):
    """Raised when deterministic recovery has no causal topology parent."""


def _finite_pair(values: Sequence[float], name: str) -> tuple[float, float]:
    if len(values) != 2:
        raise ValueError(f"{name} must contain forward and lateral meters")
    if any(type(value) not in (int, float) for value in values):
        raise TypeError(f"{name} values must be numeric scalars")
    pair = float(values[0]), float(values[1])
    if not all(math.isfinite(value) for value in pair):
        raise ValueError(f"{name} must be finite")
    return pair


@dataclass(frozen=True)
class WaypointCompilerConfig:
    forward_step_m: float = 0.25
    turn_degrees: float = 15.0
    arrival_tolerance_m: float = 0.10
    max_primitives: int = ACTION_CHUNK

    def __post_init__(self) -> None:
        if not math.isclose(
            self.forward_step_m,
            HABITAT_FORWARD_STEP_M,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("P8 successor forward_step_m is fixed to Habitat 0.25 m")
        if not math.isclose(
            self.turn_degrees,
            HABITAT_TURN_DEGREES,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("P8 successor turn_degrees is fixed to Habitat 15 degrees")
        if not math.isclose(
            self.arrival_tolerance_m, 0.10, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("P8 successor arrival_tolerance_m is fixed to 0.10 m")
        if type(self.max_primitives) is not int or not 1 <= self.max_primitives <= ACTION_CHUNK:
            raise ValueError("max_primitives must be an integer in 1..4")

    @property
    def turn_radians(self) -> float:
        return math.radians(self.turn_degrees)


@dataclass(frozen=True)
class RouteDecision:
    decision_index: int
    route: RouteName
    waypoint_local_m: tuple[float, float] = (0.0, 0.0)
    official_action: int | None = None
    episode_token: str | None = None
    physical_step: int | None = None

    def __post_init__(self) -> None:
        if type(self.decision_index) is not int or self.decision_index < 0:
            raise ValueError("decision_index must be a nonnegative integer")
        route = RouteName(self.route)
        waypoint = _finite_pair(self.waypoint_local_m, "route waypoint")
        object.__setattr__(self, "route", route)
        object.__setattr__(self, "waypoint_local_m", waypoint)
        if route is RouteName.KEEP_ORIGINAL:
            if self.official_action is not None and (
                type(self.official_action) is not int
                or not 0 <= self.official_action < ACTION_COUNT
            ):
                raise ValueError("KEEP_ORIGINAL requires one official primitive action")
            if waypoint != (0.0, 0.0):
                raise ValueError("KEEP_ORIGINAL is an opaque primitive delegation")
        elif self.official_action is not None:
            raise ValueError("only KEEP_ORIGINAL can carry an official action")
        if route is RouteName.STOP and waypoint != (0.0, 0.0):
            raise ValueError("STOP requires a zero waypoint")
        if route is RouteName.RECOVER_PARENT:
            raise ValueError("RECOVER_PARENT is lower-layer state, not a policy decision")
        if (self.episode_token is None) != (self.physical_step is None):
            raise ValueError("episode_token and physical_step must be supplied together")
        if self.episode_token is not None and (
            not isinstance(self.episode_token, str) or not self.episode_token
        ):
            raise ValueError("episode_token must be a non-empty string")
        if self.physical_step is not None and (
            type(self.physical_step) is not int or self.physical_step < 0
        ):
            raise ValueError("physical_step must be a nonnegative integer")


@dataclass(frozen=True)
class CompiledRoute:
    decision_index: int
    requested_route: RouteName
    executed_route: RouteName
    waypoint_local_m: tuple[float, float]
    proposed_actions: tuple[int, ...]
    source: str
    recovery_override: bool
    recovery_parent_index: int | None
    target_distance_m: float | None
    target_bearing_rad: float | None

    def __post_init__(self) -> None:
        requested = RouteName(self.requested_route)
        executed = RouteName(self.executed_route)
        actions = tuple(self.proposed_actions)
        waypoint = _finite_pair(self.waypoint_local_m, "compiled waypoint")
        if type(self.decision_index) is not int or self.decision_index < 0:
            raise ValueError("compiled decision_index must be a nonnegative integer")
        if requested is RouteName.RECOVER_PARENT:
            raise ValueError("RECOVER_PARENT cannot be requested by the policy")
        allowed_pairs = {
            (RouteName.STOP, RouteName.STOP),
            (RouteName.KEEP_ORIGINAL, RouteName.KEEP_ORIGINAL),
            (RouteName.KEEP_ORIGINAL, RouteName.RECOVER_PARENT),
            (RouteName.WAYPOINT, RouteName.WAYPOINT),
            (RouteName.WAYPOINT, RouteName.RECOVER_PARENT),
        }
        if (requested, executed) not in allowed_pairs:
            raise ValueError("requested and executed route combination is invalid")
        if not 1 <= len(actions) <= ACTION_CHUNK:
            raise ValueError("compiled route must contain one to four primitives")
        if any(type(action) is not int or not 0 <= action < ACTION_COUNT for action in actions):
            raise ValueError("compiled route contains an invalid primitive")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("compiled route source must be non-empty")
        if executed is RouteName.STOP and actions != (STOP_ACTION,):
            raise ValueError("STOP route must compile to one STOP primitive")
        if executed in (RouteName.WAYPOINT, RouteName.RECOVER_PARENT) and STOP_ACTION in actions:
            raise ValueError("waypoint and recovery routes may not synthesize STOP")
        if executed is RouteName.KEEP_ORIGINAL and len(actions) != 1:
            raise ValueError("official KEEP executes exactly one queued primitive per decision")
        if self.recovery_override != (executed is RouteName.RECOVER_PARENT):
            raise ValueError("recovery_override disagrees with executed route")
        if executed is RouteName.RECOVER_PARENT:
            if type(self.recovery_parent_index) is not int or self.recovery_parent_index < 0:
                raise ValueError("parent recovery requires a stable parent index")
        elif self.recovery_parent_index is not None:
            raise ValueError("only parent recovery can carry a recovery parent index")
        if executed is RouteName.KEEP_ORIGINAL:
            if waypoint != (0.0, 0.0):
                raise ValueError("official KEEP cannot carry waypoint geometry")
            if self.target_distance_m is not None or self.target_bearing_rad is not None:
                raise ValueError("official KEEP cannot carry target geometry")
        elif executed is RouteName.STOP:
            if waypoint != (0.0, 0.0):
                raise ValueError("STOP cannot carry waypoint geometry")
            if self.target_distance_m != 0.0 or self.target_bearing_rad != 0.0:
                raise ValueError("STOP target geometry must be exactly zero")
        else:
            distance, bearing = _target_geometry(waypoint)
            if type(self.target_distance_m) not in (int, float) or not math.isfinite(
                float(self.target_distance_m)
            ) or not math.isclose(
                float(self.target_distance_m), distance, rel_tol=0.0, abs_tol=1e-8
            ):
                raise ValueError("compiled target distance disagrees with waypoint")
            if type(self.target_bearing_rad) not in (int, float) or not math.isfinite(
                float(self.target_bearing_rad)
            ) or not math.isclose(
                float(self.target_bearing_rad), bearing, rel_tol=0.0, abs_tol=1e-8
            ):
                raise ValueError("compiled target bearing disagrees with waypoint")
        object.__setattr__(self, "requested_route", requested)
        object.__setattr__(self, "executed_route", executed)
        object.__setattr__(self, "waypoint_local_m", waypoint)
        object.__setattr__(self, "proposed_actions", actions)

    def to_dict(self) -> dict:
        result = asdict(self)
        result["requested_route"] = self.requested_route.value
        result["executed_route"] = self.executed_route.value
        result["waypoint_local_m"] = list(self.waypoint_local_m)
        result["proposed_actions"] = list(self.proposed_actions)
        return result


def _round_half_up_nonnegative(value: float) -> int:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("rounding input must be finite and nonnegative")
    return int(math.floor(value + 0.5))


def compile_metric_waypoint(
    waypoint_local_m: Sequence[float],
    *,
    config: WaypointCompilerConfig = WaypointCompilerConfig(),
) -> tuple[int, ...]:
    """Compile bearing and distance to a variable one-to-four action proposal."""

    forward, lateral = _finite_pair(waypoint_local_m, "metric waypoint")
    distance = math.hypot(forward, lateral)
    if distance <= config.arrival_tolerance_m:
        raise StopBranchRequired(
            "a near-zero learned waypoint cannot emit STOP; the STOP branch must decide"
        )
    bearing = math.atan2(lateral, forward)
    if forward < 0.0 and abs(lateral) <= 1e-12:
        bearing = math.pi
    turn_steps = min(
        config.max_primitives,
        _round_half_up_nonnegative(abs(bearing) / config.turn_radians),
    )
    turn_action = LEFT_ACTION if bearing > 0.0 else RIGHT_ACTION
    actions = [turn_action] * turn_steps
    remaining = config.max_primitives - len(actions)
    if remaining:
        forward_steps = max(
            1,
            min(
                remaining,
                int(math.ceil(distance / config.forward_step_m - 1e-12)),
            ),
        )
        actions.extend([FORWARD_ACTION] * forward_steps)
    if not actions or STOP_ACTION in actions:
        raise RuntimeError("metric waypoint compilation produced an invalid route")
    return tuple(actions)


def _target_geometry(waypoint: tuple[float, float]) -> tuple[float, float]:
    return math.hypot(*waypoint), math.atan2(waypoint[1], waypoint[0])


def compile_route(
    decision: RouteDecision,
    topology,
    *,
    config: WaypointCompilerConfig = WaypointCompilerConfig(),
    blocked_threshold: int = 2,
) -> CompiledRoute:
    """One-shot helper; live recovery should use the stateful SuccessorPlanner."""

    return SuccessorPlanner(
        config=config,
        blocked_threshold=blocked_threshold,
    ).plan(decision, topology)


def _waypoint_to_node(topology, node_index: int) -> tuple[float, float]:
    nodes = getattr(topology, "nodes", None)
    if nodes is None or not 0 <= node_index < len(nodes):
        raise ParentRecoveryUnavailable("latched recovery parent is absent from topology")
    target = nodes[node_index]
    dx = float(target.x) - float(topology.x)
    dz = float(target.z) - float(topology.z)
    heading = float(topology.heading)
    forward = dx * math.sin(heading) + dz * math.cos(heading)
    lateral = dx * math.cos(heading) - dz * math.sin(heading)
    return _finite_pair((forward, lateral), "latched topology parent waypoint")


def _distance_to_node(topology, node_index: int) -> float:
    forward, lateral = _waypoint_to_node(topology, node_index)
    return math.hypot(forward, lateral)


def _validate_motion_contract(topology, config: WaypointCompilerConfig) -> None:
    forward_step = float(getattr(topology, "forward_step_m"))
    turn_radians = float(getattr(topology, "turn_radians"))
    if not math.isclose(
        forward_step, config.forward_step_m, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("topology and successor forward-step contracts differ")
    if not math.isclose(
        turn_radians, config.turn_radians, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("topology and successor turn-angle contracts differ")


@dataclass
class SuccessorPlanner:
    """Resolve policy routes while latching a real topology parent for recovery."""

    config: WaypointCompilerConfig = WaypointCompilerConfig()
    blocked_threshold: int = 2
    recovery_enabled: bool = True
    recovery_parent_index: int | None = None
    root_recovery_policy: str = "raise"

    def __post_init__(self) -> None:
        if type(self.blocked_threshold) is not int or self.blocked_threshold < 1:
            raise ValueError("blocked_threshold must be a positive integer")
        if type(self.recovery_enabled) is not bool:
            raise TypeError("recovery_enabled must be boolean")
        if self.root_recovery_policy not in ("raise", "keep_original"):
            raise ValueError(
                "root_recovery_policy must be 'raise' or 'keep_original'"
            )
        if self.recovery_parent_index is not None and (
            type(self.recovery_parent_index) is not int or self.recovery_parent_index < 0
        ):
            raise ValueError("recovery_parent_index must be a nonnegative integer")

    def reset_episode(self) -> None:
        """Clear lower-layer state before attaching a new episode topology."""

        self.recovery_parent_index = None

    def _maybe_latch_parent(self, topology) -> None:
        if self.recovery_parent_index is not None:
            return
        blocked = int(getattr(topology, "consecutive_blocked_forwards", 0))
        if blocked < self.blocked_threshold:
            return
        current = int(getattr(topology, "current_node"))
        nodes = getattr(topology, "nodes")
        parent = nodes[current].parent
        if parent is None:
            raise ParentRecoveryUnavailable(
                "blocked recovery requires a causal topology parent"
            )
        self.recovery_parent_index = int(parent)

    def _use_root_keep_fallback(self, decision: RouteDecision, topology) -> bool:
        if (
            self.root_recovery_policy != "keep_original"
            or not self.recovery_enabled
            or self.recovery_parent_index is not None
            or decision.route is not RouteName.KEEP_ORIGINAL
        ):
            return False
        blocked = int(getattr(topology, "consecutive_blocked_forwards", 0))
        if blocked < self.blocked_threshold:
            return False
        current = int(getattr(topology, "current_node"))
        nodes = getattr(topology, "nodes")
        return nodes[current].parent is None

    @staticmethod
    def _compile_keep(decision: RouteDecision, *, source: str) -> CompiledRoute:
        return CompiledRoute(
            decision.decision_index,
            decision.route,
            RouteName.KEEP_ORIGINAL,
            (0.0, 0.0),
            (int(decision.official_action),),
            source,
            False,
            None,
            None,
            None,
        )

    def plan(self, decision: RouteDecision, topology) -> CompiledRoute:
        _validate_motion_contract(topology, self.config)
        if (
            decision.route is RouteName.KEEP_ORIGINAL
            and decision.official_action is None
        ):
            raise ValueError(
                "KEEP_ORIGINAL must be bound to the official primitive before planning"
            )
        if decision.route is RouteName.STOP:
            return CompiledRoute(
                decision.decision_index,
                decision.route,
                RouteName.STOP,
                (0.0, 0.0),
                (STOP_ACTION,),
                "learned_stop_branch",
                False,
                None,
                0.0,
                0.0,
            )
        if decision.route not in (RouteName.KEEP_ORIGINAL, RouteName.WAYPOINT):
            raise ValueError("policy route must be KEEP_ORIGINAL, WAYPOINT or STOP")

        root_keep_fallback = self._use_root_keep_fallback(decision, topology)
        if (
            decision.route is RouteName.KEEP_ORIGINAL
            and decision.official_action == STOP_ACTION
        ):
            return self._compile_keep(
                decision,
                source=(
                    "root_keep_fallback"
                    if root_keep_fallback
                    else "official_upstream_queue"
                ),
            )
        if root_keep_fallback:
            return self._compile_keep(decision, source="root_keep_fallback")

        reached_latched_parent = self.recovery_enabled and self.recovery_parent_index is not None and (
            _distance_to_node(topology, self.recovery_parent_index)
            <= self.config.arrival_tolerance_m
        )
        if reached_latched_parent:
            self.recovery_parent_index = None
        if self.recovery_enabled and not reached_latched_parent:
            self._maybe_latch_parent(topology)
        recovery_override = self.recovery_enabled and self.recovery_parent_index is not None
        if recovery_override:
            waypoint = _waypoint_to_node(topology, self.recovery_parent_index)
            executed_route = RouteName.RECOVER_PARENT
            source = "deterministic_topology_parent"
        elif decision.route is RouteName.KEEP_ORIGINAL:
            return self._compile_keep(
                decision,
                source="official_upstream_queue",
            )
        else:
            waypoint = decision.waypoint_local_m
            executed_route = RouteName.WAYPOINT
            source = "learned_metric_waypoint"
        actions = compile_metric_waypoint(waypoint, config=self.config)
        distance, bearing = _target_geometry(waypoint)
        return CompiledRoute(
            decision.decision_index,
            decision.route,
            executed_route,
            waypoint,
            actions,
            source,
            recovery_override,
            self.recovery_parent_index,
            distance,
            bearing,
        )


def pack_fixed_action_chunk(plan: CompiledRoute) -> tuple[tuple[int, ...], int]:
    """Pad a variable proposal for the fixed RLinf transport without executing padding."""

    count = len(plan.proposed_actions)
    padded = plan.proposed_actions + (STOP_ACTION,) * (ACTION_CHUNK - count)
    return padded, count
