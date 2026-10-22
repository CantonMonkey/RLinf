"""Dense, adaptive-duration, collision-aware waypoint candidates for G7.1."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

try:
    from .streamvln_topology import (
        FORWARD_ACTION,
        LEFT_ACTION,
        NAVIGATE_MODE,
        OnlineTopology,
        RIGHT_ACTION,
        STOP_ACTION,
        compile_waypoint,
    )
except ImportError:
    from streamvln_topology import (
        FORWARD_ACTION,
        LEFT_ACTION,
        NAVIGATE_MODE,
        OnlineTopology,
        RIGHT_ACTION,
        STOP_ACTION,
        compile_waypoint,
    )


@dataclass(frozen=True)
class DenseWaypointCandidate:
    name: str
    forward_m: float
    lateral_m: float
    distance_m: float
    heading_degrees: float
    source: str

    def vector(self) -> tuple[float, float]:
        return float(self.forward_m), float(self.lateral_m)

    def direction_key(self) -> int:
        return int(round(self.heading_degrees / 15.0))

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "waypoint_local_m": [float(self.forward_m), float(self.lateral_m)],
            "distance_m": float(self.distance_m),
            "heading_degrees": float(self.heading_degrees),
            "direction_key": self.direction_key(),
            "source": self.source,
        }


def generate_dense_candidates(
    topology: OnlineTopology,
    *,
    distances_m: Sequence[float] = (0.5, 1.0),
    headings_degrees: Sequence[float] = (0.0, 15.0, -15.0, 30.0, -30.0),
    include_parent: bool = True,
) -> tuple[DenseWaypointCandidate, ...]:
    """Generate candidates from local templates and the online parent only."""
    if (
        not distances_m
        or not headings_degrees
        or any(float(distance) <= 0.0 for distance in distances_m)
        or any(abs(float(angle)) > 60.0 for angle in headings_degrees)
    ):
        raise ValueError("dense candidate geometry is invalid")
    candidates = []
    for distance in distances_m:
        for angle_degrees in headings_degrees:
            distance = float(distance)
            angle_degrees = float(angle_degrees)
            angle = math.radians(angle_degrees)
            angle_name = f"p{int(angle_degrees)}" if angle_degrees >= 0 else f"m{abs(int(angle_degrees))}"
            distance_name = str(distance).replace(".", "p")
            candidates.append(
                DenseWaypointCandidate(
                    f"local_{angle_name}_d{distance_name}",
                    distance * math.cos(angle),
                    distance * math.sin(angle),
                    distance,
                    angle_degrees,
                    "local_template",
                )
            )
    parent = topology.backtrack_waypoint() if include_parent else None
    if parent is not None and math.hypot(*parent) > 1e-6:
        distance = math.hypot(*parent)
        angle = math.degrees(math.atan2(parent[1], parent[0]))
        candidates.append(
            DenseWaypointCandidate(
                "parent",
                float(parent[0]),
                float(parent[1]),
                float(distance),
                float(angle),
                "online_topology",
            )
        )
    if len({candidate.name for candidate in candidates}) != len(candidates):
        raise RuntimeError("dense candidate names must be unique")
    return tuple(candidates)


def select_oracle_dense_candidate(
    candidates: Sequence[DenseWaypointCandidate],
    expert_waypoint_local_m: Sequence[float],
) -> tuple[DenseWaypointCandidate, float]:
    if not candidates:
        raise ValueError("dense oracle selection requires candidates")
    if len(expert_waypoint_local_m) != 2:
        raise ValueError("expert waypoint must contain two components")
    target = tuple(float(value) for value in expert_waypoint_local_m)

    def error(candidate: DenseWaypointCandidate) -> float:
        return math.hypot(
            candidate.forward_m - target[0], candidate.lateral_m - target[1]
        )

    selected = min(candidates, key=lambda item: (error(item), item.name))
    return selected, float(error(selected))


def compile_dense_fixed(
    candidate: DenseWaypointCandidate,
    topology: OnlineTopology,
) -> tuple[list[int], int]:
    actions = compile_waypoint(candidate.vector(), NAVIGATE_MODE, topology)
    return actions, 4


def compile_dense_adaptive(
    candidate: DenseWaypointCandidate,
    topology: OnlineTopology,
) -> tuple[list[int], int]:
    """Compile a candidate to a fixed tensor plus an explicit executed prefix."""
    bearing = math.atan2(candidate.lateral_m, candidate.forward_m)
    turn_steps = min(3, int(round(abs(bearing) / topology.turn_radians)))
    turn_action = LEFT_ACTION if bearing >= 0.0 else RIGHT_ACTION
    actions = [turn_action] * turn_steps
    requested_forward = max(1, int(math.ceil(candidate.distance_m / topology.forward_step_m)))
    forward_steps = min(requested_forward, 4 - len(actions))
    actions.extend([FORWARD_ACTION] * forward_steps)
    execute_count = len(actions)
    if not 1 <= execute_count <= 4 or FORWARD_ACTION not in actions:
        raise RuntimeError("adaptive candidate must execute 1..4 actions with translation")
    actions.extend([STOP_ACTION] * (4 - len(actions)))
    return actions, execute_count


@dataclass
class DirectionFeasibility:
    cooldown_decisions: int = 2
    blocked: dict[int, int] = field(default_factory=dict)

    def __post_init__(self):
        if self.cooldown_decisions < 1:
            raise ValueError("cooldown_decisions must be positive")

    def begin_decision(self) -> None:
        self.blocked = {
            key: remaining - 1
            for key, remaining in self.blocked.items()
            if remaining > 1
        }

    def available(
        self, candidates: Sequence[DenseWaypointCandidate]
    ) -> tuple[DenseWaypointCandidate, ...]:
        allowed = tuple(
            candidate
            for candidate in candidates
            if self.blocked.get(candidate.direction_key(), 0) == 0
        )
        return allowed if allowed else tuple(candidates)

    def update(
        self,
        candidate: DenseWaypointCandidate,
        actions: Sequence[int],
        collisions: Sequence[bool],
    ) -> bool:
        if len(actions) != len(collisions):
            raise ValueError("feasibility update needs one collision per action")
        blocked_forward = any(
            action == FORWARD_ACTION and collision
            for action, collision in zip(actions, collisions)
        )
        if blocked_forward:
            self.blocked[candidate.direction_key()] = self.cooldown_decisions
        return blocked_forward

    def summary(self) -> dict:
        return {
            "cooldown_decisions": self.cooldown_decisions,
            "blocked_direction_bins": {
                str(key): remaining for key, remaining in sorted(self.blocked.items())
            },
        }
