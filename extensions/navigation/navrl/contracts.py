"""Backend-neutral navigation data shapes.

An RGB frame is H x W x 3 uint8. The standard-library mock uses immutable nested
tuples; a real simulator may return a NumPy array with the same shape and dtype.
``pose`` is privileged and must be hidden from a visual policy by default.
"""

from __future__ import annotations

from enum import Enum
from typing import TypedDict


class NavAction(str, Enum):
    MOVE_FORWARD = "MOVE_FORWARD"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    STOP = "STOP"


class Observation(TypedDict):
    rgb: object
    depth: object | None
    instruction: str
    history: list[object]
    pose: tuple[float, float, float] | None
    episode_id: str
    step_id: int


class NavInfo(TypedDict):
    success: bool
    distance_to_goal: float  # Euclidean; diagnostic only
    geodesic_distance: float  # traversable-path distance; reward uses this
    path_length: float
    spl: float
    collision: bool  # this step
    collision_count: int  # episode total
    stop_called: bool
    timeout: bool
    invalid_action: bool  # this step
    termination_reason: str | None


class PolicyOutput(TypedDict):
    action: object
    stop: bool
    log_prob: object | None
    value: object | None
    policy_state: object | None
