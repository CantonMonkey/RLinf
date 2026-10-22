"""Exact phased macro candidates and collision-heading shield for G7.4."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence, TypeVar

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


MACRO_DISTANCES_M = (0.5, 0.75, 1.0)
MACRO_HEADINGS_DEGREES = (
    0.0,
    15.0,
    -15.0,
    30.0,
    -30.0,
    45.0,
    -45.0,
    60.0,
    -60.0,
    75.0,
    -75.0,
    90.0,
    -90.0,
)
CandidateT = TypeVar("CandidateT")


@dataclass(frozen=True)
class MacroWaypointCandidate:
    name: str
    forward_m: float
    lateral_m: float
    actions: tuple[int, ...]
    heading_degrees: float
    distance_m: float
    first_forward_heading_degrees: float
    source: str = "exact_align_then_translate_macro"

    def vector(self) -> tuple[float, float]:
        return float(self.forward_m), float(self.lateral_m)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "waypoint_local_m": [float(self.forward_m), float(self.lateral_m)],
            "actions": list(self.actions),
            "heading_degrees": float(self.heading_degrees),
            "distance_m": float(self.distance_m),
            "first_forward_heading_degrees": float(
                self.first_forward_heading_degrees
            ),
            "source": self.source,
        }


def simulate_action_endpoint(
    actions: Sequence[int],
    *,
    forward_step_m: float,
    turn_radians: float,
) -> tuple[float, float, float]:
    if not actions or len(actions) > 8:
        raise ValueError("macro action sequence must contain 1..8 primitives")
    if any(action not in (FORWARD_ACTION, LEFT_ACTION, RIGHT_ACTION) for action in actions):
        raise ValueError("macro action sequence contains an invalid primitive")
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


def first_forward_heading_degrees(
    actions: Sequence[int], *, turn_degrees: float = 15.0
) -> float:
    heading = 0.0
    for action in actions:
        if action == LEFT_ACTION:
            heading += turn_degrees
        elif action == RIGHT_ACTION:
            heading -= turn_degrees
        elif action == FORWARD_ACTION:
            return float(heading)
        elif action == STOP_ACTION:
            break
        else:
            raise ValueError("unknown primitive action")
    raise ValueError("candidate action sequence has no forward primitive")


def generate_macro_candidates(
    topology: OnlineTopology,
) -> tuple[MacroWaypointCandidate, ...]:
    """Generate exact turn-then-translate sequences spanning at most two chunks."""
    turn_degrees = math.degrees(topology.turn_radians)
    candidates = []
    for heading_degrees in MACRO_HEADINGS_DEGREES:
        signed_turns = int(round(heading_degrees / turn_degrees))
        turn_action = LEFT_ACTION if signed_turns >= 0 else RIGHT_ACTION
        turn_actions = [turn_action] * abs(signed_turns)
        for distance_m in MACRO_DISTANCES_M:
            forward_steps = int(round(distance_m / topology.forward_step_m))
            actions = tuple(turn_actions + [FORWARD_ACTION] * forward_steps)
            if len(actions) > 8:
                continue
            forward, lateral, final_heading = simulate_action_endpoint(
                actions,
                forward_step_m=topology.forward_step_m,
                turn_radians=topology.turn_radians,
            )
            heading_name = (
                f"p{int(heading_degrees)}"
                if heading_degrees >= 0
                else f"m{abs(int(heading_degrees))}"
            )
            distance_name = str(distance_m).replace(".", "p")
            candidates.append(
                MacroWaypointCandidate(
                    name=f"macro_{heading_name}_d{distance_name}",
                    forward_m=forward,
                    lateral_m=lateral,
                    actions=actions,
                    heading_degrees=float(heading_degrees),
                    distance_m=float(distance_m),
                    first_forward_heading_degrees=float(heading_degrees),
                )
            )
    candidates.sort(key=lambda item: item.name)
    endpoints = {
        (round(item.forward_m, 9), round(item.lateral_m, 9)) for item in candidates
    }
    if len(candidates) != 33 or len(endpoints) != len(candidates):
        raise RuntimeError("macro endpoint bank must contain 33 unique candidates")
    if any(
        not 1 <= len(item.actions) <= 8
        or FORWARD_ACTION not in item.actions
        or STOP_ACTION in item.actions
        for item in candidates
    ):
        raise RuntimeError("macro candidate action contract failed")
    return tuple(candidates)


def select_oracle_candidate(
    candidates: Sequence[CandidateT],
    expert_waypoint_local_m: Sequence[float],
) -> tuple[CandidateT, float]:
    if not candidates:
        raise ValueError("oracle selection requires candidates")
    if len(expert_waypoint_local_m) != 2:
        raise ValueError("expert waypoint must contain two components")
    target = tuple(float(value) for value in expert_waypoint_local_m)
    if any(not math.isfinite(value) for value in target):
        raise ValueError("expert waypoint must be finite")

    def error(candidate) -> float:
        forward, lateral = candidate.vector()
        return math.hypot(forward - target[0], lateral - target[1])

    selected = min(candidates, key=lambda item: (error(item), item.name))
    return selected, float(error(selected))


def trace_collision_world_headings(
    actions: Sequence[int],
    collisions: Sequence[bool],
    *,
    start_world_heading: float,
    turn_radians: float,
) -> tuple[float, ...]:
    if len(actions) != len(collisions):
        raise ValueError("collision heading trace requires aligned inputs")
    heading = float(start_world_heading)
    result = []
    for action, collision in zip(actions, collisions):
        if action == LEFT_ACTION:
            heading += turn_radians
        elif action == RIGHT_ACTION:
            heading -= turn_radians
        elif action == FORWARD_ACTION and collision:
            result.append(math.atan2(math.sin(heading), math.cos(heading)))
    return tuple(result)

@dataclass
class WorldHeadingShield:
    turn_radians: float
    cooldown_decisions: int = 2
    blocked_bins: dict[int, int] = field(default_factory=dict)

    def __post_init__(self):
        if self.turn_radians <= 0.0 or self.cooldown_decisions < 1:
            raise ValueError("shield configuration is invalid")
        self.bin_count = int(round(2.0 * math.pi / self.turn_radians))

    def _bin(self, heading_radians: float) -> int:
        return int(round(heading_radians / self.turn_radians)) % self.bin_count

    def begin_decision(self) -> None:
        self.blocked_bins = {
            key: remaining - 1
            for key, remaining in self.blocked_bins.items()
            if remaining > 1
        }

    def block_world_heading(self, heading_radians: float) -> None:
        self.blocked_bins[self._bin(heading_radians)] = self.cooldown_decisions + 1

    def available(
        self,
        candidates: Sequence[CandidateT],
        *,
        current_world_heading: float,
    ) -> tuple[CandidateT, ...]:
        allowed = []
        for candidate in candidates:
            relative = math.radians(
                first_forward_heading_degrees(candidate.actions)
            )
            world_bin = self._bin(current_world_heading + relative)
            if self.blocked_bins.get(world_bin, 0) == 0:
                allowed.append(candidate)
        return tuple(allowed) if allowed else tuple(candidates)

    def summary(self) -> dict:
        return {
            "cooldown_decisions": self.cooldown_decisions,
            "blocked_world_heading_bins": {
                str(key): remaining
                for key, remaining in sorted(self.blocked_bins.items())
            },
        }
