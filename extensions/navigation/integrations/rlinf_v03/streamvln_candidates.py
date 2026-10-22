"""Goal-free waypoint candidates and phased collision recovery for G7."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

try:
    from .streamvln_topology import (
        FORWARD_ACTION,
        LEFT_ACTION,
        NAVIGATE_MODE,
        OnlineTopology,
        RECOVER_MODE,
        RIGHT_ACTION,
        STOP_ACTION,
        STOP_MODE,
        compile_waypoint,
    )
except ImportError:  # Direct staged-file execution on the A800 host.
    from streamvln_topology import (
        FORWARD_ACTION,
        LEFT_ACTION,
        NAVIGATE_MODE,
        OnlineTopology,
        RECOVER_MODE,
        RIGHT_ACTION,
        STOP_ACTION,
        STOP_MODE,
        compile_waypoint,
    )


@dataclass(frozen=True)
class WaypointCandidate:
    name: str
    forward_m: float
    lateral_m: float
    source: str

    def vector(self) -> tuple[float, float]:
        return float(self.forward_m), float(self.lateral_m)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "waypoint_local_m": [float(self.forward_m), float(self.lateral_m)],
            "source": self.source,
        }


def generate_waypoint_candidates(
    topology: OnlineTopology,
    *,
    distance_m: float = 1.0,
    branch_degrees: float = 30.0,
    include_parent: bool = True,
) -> tuple[WaypointCandidate, ...]:
    """Generate a fixed local candidate set without a goal or expert input."""
    if distance_m <= 0.0 or not 0.0 < branch_degrees < 90.0:
        raise ValueError("candidate geometry must be positive and forward-facing")
    angle = math.radians(float(branch_degrees))
    forward = float(distance_m) * math.cos(angle)
    lateral = float(distance_m) * math.sin(angle)
    candidates = [
        WaypointCandidate("forward", float(distance_m), 0.0, "local_template"),
        WaypointCandidate("left_front", forward, lateral, "local_template"),
        WaypointCandidate("right_front", forward, -lateral, "local_template"),
    ]
    parent = topology.backtrack_waypoint() if include_parent else None
    if parent is not None and math.hypot(*parent) > 1e-6:
        candidates.append(
            WaypointCandidate("parent", float(parent[0]), float(parent[1]), "online_topology")
        )
    if len({candidate.name for candidate in candidates}) != len(candidates):
        raise RuntimeError("candidate names must be unique")
    if any(
        not math.isfinite(value)
        for candidate in candidates
        for value in candidate.vector()
    ):
        raise RuntimeError("candidate geometry must be finite")
    return tuple(candidates)


def select_oracle_candidate(
    candidates: Sequence[WaypointCandidate],
    expert_waypoint_local_m: Sequence[float],
) -> tuple[WaypointCandidate, float]:
    """Select the nearest candidate for a train-only privileged diagnostic."""
    if not candidates:
        raise ValueError("oracle selection requires candidates")
    if len(expert_waypoint_local_m) != 2:
        raise ValueError("expert waypoint must contain forward and lateral meters")
    target = tuple(float(value) for value in expert_waypoint_local_m)
    if any(not math.isfinite(value) for value in target):
        raise ValueError("expert waypoint must be finite")

    def error(candidate: WaypointCandidate) -> float:
        return math.hypot(
            candidate.forward_m - target[0], candidate.lateral_m - target[1]
        )

    selected = min(candidates, key=lambda candidate: (error(candidate), candidate.name))
    return selected, float(error(selected))


def compile_candidate(
    candidate: WaypointCandidate,
    topology: OnlineTopology,
) -> list[int]:
    return compile_waypoint(candidate.vector(), NAVIGATE_MODE, topology)


def compile_phased_recovery(topology: OnlineTopology) -> tuple[list[int], dict]:
    """Align to a parent in turn-only chunks, then translate and exit on success."""
    parent = topology.backtrack_waypoint()
    if parent is None:
        actions = [LEFT_ACTION, LEFT_ACTION, FORWARD_ACTION, FORWARD_ACTION]
        return actions, {
            "phase": "root_probe",
            "target_local_m": [-0.5, 0.0],
            "bearing_rad": math.pi,
        }

    forward, lateral = (float(value) for value in parent)
    bearing = math.atan2(lateral, forward)
    turn_steps = int(round(abs(bearing) / topology.turn_radians))
    if turn_steps > 1:
        turn_action = LEFT_ACTION if bearing > 0.0 else RIGHT_ACTION
        applied_turns = 4 if turn_steps >= 4 else 2
        actions = [turn_action] * applied_turns
        if applied_turns == 2:
            opposite = RIGHT_ACTION if turn_action == LEFT_ACTION else LEFT_ACTION
            actions.extend([opposite, turn_action])
        phase = "align"
    else:
        distance = math.hypot(forward, lateral)
        forward_steps = 4 if distance > 0.75 else 2
        actions = [FORWARD_ACTION] * forward_steps
        if forward_steps == 2:
            actions.extend([LEFT_ACTION, RIGHT_ACTION])
        phase = "translate"
    if len(actions) != 4 or STOP_ACTION in actions:
        raise RuntimeError("phased recovery must compile four non-stop primitives")
    if phase == "align" and FORWARD_ACTION in actions:
        raise RuntimeError("alignment phase must not probe before alignment")
    if phase == "translate" and FORWARD_ACTION not in actions:
        raise RuntimeError("translation phase requires forward progress")
    return actions, {
        "phase": phase,
        "target_local_m": [forward, lateral],
        "bearing_rad": float(bearing),
    }


def compile_mode(
    mode: int,
    waypoint_local_m: Sequence[float],
    topology: OnlineTopology,
) -> tuple[list[int], dict]:
    if mode == STOP_MODE:
        return [STOP_ACTION] * 4, {"phase": "stop"}
    if mode == RECOVER_MODE:
        return compile_phased_recovery(topology)
    if mode != NAVIGATE_MODE:
        raise ValueError("unknown high-level mode")
    actions = compile_waypoint(waypoint_local_m, NAVIGATE_MODE, topology)
    return actions, {"phase": "navigate"}
