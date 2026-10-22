"""Executor-aligned high-level action contract for StreamVLN G8."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Sequence

try:
    from .streamvln_candidates_g73 import (
        ExecutableWaypointCandidate,
        generate_executable_candidates,
        select_oracle_executable_candidate,
    )
    from .streamvln_topology import (
        NAVIGATE_MODE,
        RECOVER_MODE,
        STOP_ACTION,
        STOP_MODE,
        OnlineTopology,
        compile_waypoint,
    )
except ImportError:  # Direct staged-file tests.
    from streamvln_candidates_g73 import (
        ExecutableWaypointCandidate,
        generate_executable_candidates,
        select_oracle_executable_candidate,
    )
    from streamvln_topology import (
        NAVIGATE_MODE,
        RECOVER_MODE,
        STOP_ACTION,
        STOP_MODE,
        OnlineTopology,
        compile_waypoint,
    )


NAVIGATE_ACTION_COUNT = 26
STOP_ACTION_ID = 26
RECOVER_ACTION_ID = 27
ACTION_COUNT = 28
ACTION_CHUNK = 4
CANDIDATE_ORDER_SHA256 = (
    "1aa80fd3ada6d18fb042b27403bd1b6c74d760273901bcaebfa5c96d0acdf2cd"
)


def candidate_order_sha256(
    candidates: Sequence[ExecutableWaypointCandidate],
) -> str:
    # Hash integerized semantics instead of raw JSON floats.  CPython/libm
    # versions can differ in the final bit of trig-derived endpoints even when
    # the executor lattice is identical.
    canonical = [
        {
            "name": candidate.name,
            "endpoint_nanometers": [
                int(round(candidate.forward_m * 1_000_000_000)),
                int(round(candidate.lateral_m * 1_000_000_000)),
            ],
            "actions": list(candidate.actions),
            "source": candidate.source,
        }
        for candidate in candidates
    ]
    payload = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def executable_candidates(
    topology: OnlineTopology,
) -> tuple[ExecutableWaypointCandidate, ...]:
    candidates = generate_executable_candidates(topology)
    if len(candidates) != NAVIGATE_ACTION_COUNT:
        raise RuntimeError(
            f"G8 expected {NAVIGATE_ACTION_COUNT} navigation candidates, "
            f"received {len(candidates)}"
        )
    digest = candidate_order_sha256(candidates)
    if digest != CANDIDATE_ORDER_SHA256:
        raise RuntimeError(
            "G8 executable candidate ordering or motion constants changed: " + digest
        )
    return candidates


def action_mode(action_id: int) -> int:
    if type(action_id) is not int or not 0 <= action_id < ACTION_COUNT:
        raise ValueError(f"G8 action id must be an integer in 0..{ACTION_COUNT - 1}")
    if action_id < NAVIGATE_ACTION_COUNT:
        return NAVIGATE_MODE
    return STOP_MODE if action_id == STOP_ACTION_ID else RECOVER_MODE


def action_waypoint_local_m(
    action_id: int,
    topology: OnlineTopology,
) -> tuple[float, float]:
    mode = action_mode(action_id)
    if mode == NAVIGATE_MODE:
        return executable_candidates(topology)[action_id].vector()
    if mode == STOP_MODE:
        return 0.0, 0.0
    recovery = topology.backtrack_waypoint()
    return recovery if recovery is not None else (-0.5, 0.0)


def compile_g8_action(
    action_id: int,
    topology: OnlineTopology,
) -> list[int]:
    """Map one replayable high-level action ID to the exact proposed macro."""
    mode = action_mode(action_id)
    if mode == NAVIGATE_MODE:
        actions = list(executable_candidates(topology)[action_id].actions)
    elif mode == STOP_MODE:
        actions = [STOP_ACTION] * ACTION_CHUNK
    else:
        actions = compile_waypoint(
            action_waypoint_local_m(action_id, topology),
            RECOVER_MODE,
            topology,
        )
    if len(actions) != ACTION_CHUNK:
        raise RuntimeError("G8 actions must propose exactly four primitives")
    return [int(action) for action in actions]


def target_action_id(
    target_mode: int,
    target_waypoint_local_m: Sequence[float],
    topology: OnlineTopology,
) -> int:
    """Convert a BC mode/metric-waypoint target to the nearest executable ID."""
    if target_mode == STOP_MODE:
        return STOP_ACTION_ID
    if target_mode == RECOVER_MODE:
        return RECOVER_ACTION_ID
    if target_mode != NAVIGATE_MODE:
        raise ValueError("unknown G8 target mode")
    if len(target_waypoint_local_m) != 2:
        raise ValueError("G8 target waypoint must contain forward and lateral meters")
    target = tuple(float(value) for value in target_waypoint_local_m)
    if any(not math.isfinite(value) for value in target):
        raise ValueError("G8 target waypoint must be finite")
    candidates = executable_candidates(topology)
    selected, _ = select_oracle_executable_candidate(candidates, target)
    return candidates.index(selected)


def candidate_endpoint_matrix(
    topology: OnlineTopology | None = None,
) -> tuple[tuple[float, float], ...]:
    if topology is None:
        topology = OnlineTopology(max_decisions=1)
    return tuple(candidate.vector() for candidate in executable_candidates(topology))


def assert_g8_action_contract() -> dict:
    topology = OnlineTopology(max_decisions=2)
    candidates = executable_candidates(topology)
    compiled = tuple(tuple(compile_g8_action(index, topology)) for index in range(ACTION_COUNT))
    if len(set(compiled[:NAVIGATE_ACTION_COUNT])) != NAVIGATE_ACTION_COUNT:
        raise RuntimeError("G8 navigation action IDs must map to unique macros")
    if compiled[STOP_ACTION_ID] != (STOP_ACTION,) * ACTION_CHUNK:
        raise RuntimeError("G8 STOP action contract failed")
    return {
        "action_count": ACTION_COUNT,
        "navigate_candidate_count": len(candidates),
        "candidate_order_sha256": candidate_order_sha256(candidates),
        "unique_navigate_macros": len(set(compiled[:NAVIGATE_ACTION_COUNT])),
    }
