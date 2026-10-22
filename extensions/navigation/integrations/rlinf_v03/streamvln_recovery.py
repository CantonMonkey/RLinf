"""Goal-free collision recovery and waypoint execution diagnostics for G6."""

from __future__ import annotations

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


def recovery_target(topology: OnlineTopology) -> tuple[float, float]:
    """Return a local backtrack target using only the online execution graph."""
    target = topology.backtrack_waypoint()
    return (-0.5, 0.0) if target is None else target


def recovery_required(
    topology: OnlineTopology,
    *,
    blocked_threshold: int = 2,
) -> bool:
    if blocked_threshold < 1:
        raise ValueError("blocked_threshold must be positive")
    return topology.consecutive_blocked_forwards >= blocked_threshold


def resolved_choice(
    policy_mode: int,
    policy_waypoint: Sequence[float],
    topology: OnlineTopology,
    *,
    recovery_enabled: bool,
    blocked_threshold: int = 2,
) -> dict:
    """Apply the deterministic execution shield without goal or oracle inputs."""
    if policy_mode not in (NAVIGATE_MODE, STOP_MODE, RECOVER_MODE):
        raise ValueError("unknown high-level mode")
    if len(policy_waypoint) != 2:
        raise ValueError("policy waypoint must have two components")
    override = bool(
        recovery_enabled
        and policy_mode != STOP_MODE
        and recovery_required(topology, blocked_threshold=blocked_threshold)
    )
    if override or policy_mode == RECOVER_MODE:
        executed_mode = RECOVER_MODE
        waypoint = recovery_target(topology)
        source = "collision_override" if override else "policy_recover"
    else:
        executed_mode = int(policy_mode)
        waypoint = (float(policy_waypoint[0]), float(policy_waypoint[1]))
        source = "policy"
    return {
        "policy_mode": int(policy_mode),
        "executed_mode": executed_mode,
        "waypoint_local_m": [float(waypoint[0]), float(waypoint[1])],
        "recovery_override": override,
        "choice_source": source,
    }


def compile_recovery_aware(
    waypoint_local_m: Sequence[float],
    mode: int,
    topology: OnlineTopology,
) -> list[int]:
    """Compile RECOVER with a reserved translation step; defer other modes to G5."""
    if mode != RECOVER_MODE:
        return compile_waypoint(waypoint_local_m, mode, topology)
    forward, lateral = recovery_target(topology)
    if not math.isfinite(forward) or not math.isfinite(lateral):
        raise ValueError("recovery waypoint must be finite")
    bearing = math.atan2(lateral, forward)
    # At most three turns guarantees every recovery decision probes one new
    # translation, instead of spending the entire four-action chunk rotating.
    turn_steps = min(3, int(round(abs(bearing) / topology.turn_radians)))
    action = LEFT_ACTION if bearing >= 0.0 else RIGHT_ACTION
    actions = [action] * turn_steps + [FORWARD_ACTION] * (4 - turn_steps)
    if len(actions) != 4 or STOP_ACTION in actions or FORWARD_ACTION not in actions:
        raise RuntimeError("recovery compilation must contain four non-stop actions and a forward probe")
    return actions


def dead_reckoned_local_displacement(
    start_pose: Sequence[float],
    topology: OnlineTopology,
) -> tuple[float, float]:
    if len(start_pose) != 3:
        raise ValueError("start pose must be x, z, heading")
    start_x, start_z, start_heading = (float(value) for value in start_pose)
    dx, dz = topology.x - start_x, topology.z - start_z
    forward = dx * math.sin(start_heading) + dz * math.cos(start_heading)
    lateral = dx * math.cos(start_heading) - dz * math.sin(start_heading)
    return float(forward), float(lateral)


def waypoint_execution_error(
    target_local_m: Sequence[float],
    start_pose: Sequence[float],
    topology: OnlineTopology,
) -> dict:
    if len(target_local_m) != 2:
        raise ValueError("target waypoint must have two components")
    target = (float(target_local_m[0]), float(target_local_m[1]))
    realized = dead_reckoned_local_displacement(start_pose, topology)
    error = math.hypot(target[0] - realized[0], target[1] - realized[1])
    return {
        "target_local_m": list(target),
        "realized_local_m": list(realized),
        "l2_m": float(error),
    }
