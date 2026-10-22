"""Executor-aligned fixed-four primitive waypoint candidates for G7.3."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Sequence

try:
    from .streamvln_topology import (
        FORWARD_ACTION,
        LEFT_ACTION,
        OnlineTopology,
        RIGHT_ACTION,
        STOP_ACTION,
    )
except ImportError:
    from streamvln_topology import (
        FORWARD_ACTION,
        LEFT_ACTION,
        OnlineTopology,
        RIGHT_ACTION,
        STOP_ACTION,
    )


_EXECUTOR_ACTIONS = (FORWARD_ACTION, LEFT_ACTION, RIGHT_ACTION)


@dataclass(frozen=True)
class ExecutableWaypointCandidate:
    name: str
    forward_m: float
    lateral_m: float
    actions: tuple[int, int, int, int]
    final_heading_degrees: float
    source: str = "fixed4_action_lattice"

    def vector(self) -> tuple[float, float]:
        return float(self.forward_m), float(self.lateral_m)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "waypoint_local_m": [float(self.forward_m), float(self.lateral_m)],
            "actions": list(self.actions),
            "final_heading_degrees": float(self.final_heading_degrees),
            "source": self.source,
        }


def simulate_fixed4_endpoint(
    actions: Sequence[int],
    *,
    forward_step_m: float,
    turn_radians: float,
) -> tuple[float, float, float]:
    """Return forward, lateral and final heading for a collision-free action chunk."""
    if len(actions) != 4 or any(action not in _EXECUTOR_ACTIONS for action in actions):
        raise ValueError("executor candidate requires four non-stop primitives")
    if forward_step_m <= 0.0 or turn_radians <= 0.0:
        raise ValueError("executor motion constants must be positive")
    forward = 0.0
    lateral = 0.0
    heading = 0.0
    for action in actions:
        if action == LEFT_ACTION:
            heading += turn_radians
        elif action == RIGHT_ACTION:
            heading -= turn_radians
        else:
            lateral += math.sin(heading) * forward_step_m
            forward += math.cos(heading) * forward_step_m
    heading = math.atan2(math.sin(heading), math.cos(heading))
    return float(forward), float(lateral), float(heading)


def _canonical_rank(
    actions: tuple[int, int, int, int],
    forward: float,
    lateral: float,
    final_heading: float,
) -> tuple:
    bearing = math.atan2(lateral, forward)
    turn_count = sum(action != FORWARD_ACTION for action in actions)
    heading_error = abs(math.atan2(
        math.sin(final_heading - bearing), math.cos(final_heading - bearing)
    ))
    return turn_count, heading_error, actions


def generate_executable_candidates(
    topology: OnlineTopology,
) -> tuple[ExecutableWaypointCandidate, ...]:
    """Enumerate unique realized endpoints of the exact fixed-four executor."""
    by_endpoint: dict[tuple[float, float], tuple] = {}
    for raw_actions in itertools.product(_EXECUTOR_ACTIONS, repeat=4):
        actions = tuple(int(action) for action in raw_actions)
        if FORWARD_ACTION not in actions:
            continue
        forward, lateral, final_heading = simulate_fixed4_endpoint(
            actions,
            forward_step_m=topology.forward_step_m,
            turn_radians=topology.turn_radians,
        )
        endpoint_key = (round(forward, 9), round(lateral, 9))
        rank = _canonical_rank(actions, forward, lateral, final_heading)
        previous = by_endpoint.get(endpoint_key)
        if previous is None or rank < previous[0]:
            by_endpoint[endpoint_key] = (
                rank,
                actions,
                forward,
                lateral,
                final_heading,
            )
    candidates = []
    for _, actions, forward, lateral, final_heading in by_endpoint.values():
        action_name = "".join(str(action) for action in actions)
        candidates.append(
            ExecutableWaypointCandidate(
                name=f"fixed4_{action_name}",
                forward_m=forward,
                lateral_m=lateral,
                actions=actions,
                final_heading_degrees=math.degrees(final_heading),
            )
        )
    candidates.sort(key=lambda item: item.name)
    if not candidates:
        raise RuntimeError("executor candidate lattice is empty")
    endpoint_keys = {
        (round(item.forward_m, 9), round(item.lateral_m, 9)) for item in candidates
    }
    if len(endpoint_keys) != len(candidates):
        raise RuntimeError("executor candidate endpoints must be unique")
    if any(
        len(item.actions) != 4
        or STOP_ACTION in item.actions
        or not math.isfinite(item.forward_m)
        or not math.isfinite(item.lateral_m)
        for item in candidates
    ):
        raise RuntimeError("executor candidate contract failed")
    return tuple(candidates)


def select_oracle_executable_candidate(
    candidates: Sequence[ExecutableWaypointCandidate],
    expert_waypoint_local_m: Sequence[float],
) -> tuple[ExecutableWaypointCandidate, float]:
    if not candidates:
        raise ValueError("executor oracle selection requires candidates")
    if len(expert_waypoint_local_m) != 2:
        raise ValueError("expert waypoint must contain two components")
    target = tuple(float(value) for value in expert_waypoint_local_m)
    if any(not math.isfinite(value) for value in target):
        raise ValueError("expert waypoint must be finite")

    def error(candidate: ExecutableWaypointCandidate) -> float:
        return math.hypot(
            candidate.forward_m - target[0], candidate.lateral_m - target[1]
        )

    selected = min(candidates, key=lambda item: (error(item), item.name))
    return selected, float(error(selected))


def compile_executable_candidate(
    candidate: ExecutableWaypointCandidate,
    topology: OnlineTopology,
) -> tuple[list[int], int]:
    del topology
    return list(candidate.actions), 4
