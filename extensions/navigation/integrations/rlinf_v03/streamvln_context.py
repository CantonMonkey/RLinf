"""State and wire contracts for isolated StreamVLN categorical sessions."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import hashlib


ACTION_CHUNK = 4
BLOCK_ACTIONS = 32
BLOCK_OBSERVATIONS = BLOCK_ACTIONS // ACTION_CHUNK
HISTORY_OBSERVATIONS = 8
ACTION_COUNT = 4


def encode_fixed_utf8(text: str, width: int) -> tuple[list[int], list[bool]]:
    if not isinstance(text, str) or not text:
        raise ValueError("text must be a non-empty string")
    if width < 1:
        raise ValueError("UTF-8 width must be positive")
    payload = text.encode("utf-8")
    if len(payload) > width:
        raise ValueError(f"UTF-8 text requires {len(payload)} bytes, limit is {width}")
    values = list(payload) + [0] * (width - len(payload))
    mask = [index < len(payload) for index in range(width)]
    return values, mask


def decode_fixed_utf8(values, mask) -> str:
    if len(values) != len(mask) or not values:
        raise ValueError("UTF-8 values and mask must have one fixed width")
    seen_padding = False
    payload = bytearray()
    for value, valid in zip(values, mask):
        if type(valid) is not bool:
            raise TypeError("UTF-8 mask must contain booleans")
        if type(value) is not int or not 0 <= value <= 255:
            raise TypeError("UTF-8 values must contain bytes")
        if valid:
            if seen_padding:
                raise ValueError("UTF-8 mask must be prefix-contiguous")
            payload.append(value)
        else:
            seen_padding = True
            if value != 0:
                raise ValueError("UTF-8 padding bytes must be zero")
    if not payload:
        raise ValueError("UTF-8 payload cannot be empty")
    return bytes(payload).decode("utf-8")


def frame_sha256(frame: bytes) -> str:
    return hashlib.sha256(frame).hexdigest()


@dataclass(frozen=True)
class SessionKey:
    env_id: int
    episode_serial: int
    scene: str
    episode_id: str

    def __post_init__(self) -> None:
        if type(self.env_id) is not int or self.env_id < 0:
            raise ValueError("env_id must be a nonnegative integer")
        if type(self.episode_serial) is not int or self.episode_serial < 0:
            raise ValueError("episode_serial must be a nonnegative integer")
        if not self.scene or not self.episode_id:
            raise ValueError("scene and episode_id are required")


@dataclass
class ContextLedger:
    """Per-environment chronology independent from model implementation."""

    key: SessionKey
    instruction: str
    image_shape: tuple[int, int, int]
    frames: dict[int, bytes] = field(default_factory=dict)
    actions: list[int] = field(default_factory=list)
    pending_actions: list[int] | None = None
    last_response_key: tuple[int, str] | None = None
    last_response: dict | None = None

    def __post_init__(self) -> None:
        if not self.instruction:
            raise ValueError("instruction is required")
        if (
            len(self.image_shape) != 3
            or any(type(value) is not int or value < 1 for value in self.image_shape)
            or self.image_shape[2] != 3
        ):
            raise ValueError("image_shape must be positive HWC RGB")

    @property
    def step(self) -> int:
        return len(self.actions)

    def acknowledge(self, executed_actions) -> None:
        if self.pending_actions is None:
            if list(executed_actions):
                raise ValueError("cannot acknowledge without a pending decision")
            return
        executed = list(executed_actions)
        if not 1 <= len(executed) <= ACTION_CHUNK:
            raise ValueError("executed action count must be in 1..4")
        if executed != self.pending_actions[: len(executed)]:
            raise ValueError("executed actions disagree with the pending proposal")
        self.actions.extend(executed)
        self.pending_actions = None
        self.last_response_key = None
        self.last_response = None

    def record_observation(
        self,
        step: int,
        frame: bytes,
        image_shape: tuple[int, int, int],
    ) -> None:
        if self.pending_actions is not None:
            raise ValueError("previous decision must be acknowledged first")
        if step != self.step or step % ACTION_CHUNK:
            raise ValueError("observation step must match four-action chronology")
        if image_shape != self.image_shape:
            raise ValueError("RGB shape changed inside an episode")
        expected = self.image_shape[0] * self.image_shape[1] * self.image_shape[2]
        if not isinstance(frame, bytes) or len(frame) != expected:
            raise ValueError("RGB byte count does not match image_shape")
        if step in self.frames:
            if self.frames[step] != frame:
                raise ValueError("repeated observation changed RGB content")
            return
        self.frames[step] = frame

    def set_proposal(self, proposed_actions) -> None:
        proposed = list(proposed_actions)
        if self.pending_actions is not None:
            raise ValueError("a decision is already pending")
        if (
            not 1 <= len(proposed) <= ACTION_CHUNK
            or any(type(action) is not int or not 0 <= action < ACTION_COUNT for action in proposed)
        ):
            raise ValueError("proposal must contain one to four categorical actions")
        if 0 in proposed[:-1]:
            raise ValueError("STOP can only be the final proposed action")
        self.pending_actions = proposed

    def response_key(self, step: int, frame: bytes) -> tuple[int, str]:
        return step, frame_sha256(frame)

    def snapshot(self) -> dict:
        if self.pending_actions is None:
            raise ValueError("snapshot requires a pending decision")
        step = self.step
        if step not in self.frames:
            raise ValueError("current observation is not recorded")
        block_start = (step // BLOCK_ACTIONS) * BLOCK_ACTIONS
        if block_start:
            stride = block_start // HISTORY_OBSERVATIONS
            history_steps = list(range(0, block_start, stride))
            if len(history_steps) != HISTORY_OBSERVATIONS:
                raise ValueError("history sampling must produce eight observations")
        else:
            history_steps = []
        block_steps = list(range(block_start, step + 1, ACTION_CHUNK))
        missing = [
            value for value in (*history_steps, *block_steps)
            if value not in self.frames
        ]
        if missing:
            raise ValueError(f"missing replay observations at steps {missing}")
        block_actions = self.actions[block_start:step]
        if len(block_actions) != step - block_start:
            raise ValueError("block action chronology is incomplete")
        return {
            "env_id": self.key.env_id,
            "episode_serial": self.key.episode_serial,
            "scene": self.key.scene,
            "episode_id": self.key.episode_id,
            "instruction": self.instruction,
            "image_shape": list(self.image_shape),
            "step": step,
            "block_start": block_start,
            "history_steps": history_steps,
            "history_frames": [self.frames[value] for value in history_steps],
            "block_steps": block_steps,
            "block_frames": [self.frames[value] for value in block_steps],
            "block_actions": block_actions,
            "current_actions": list(self.pending_actions),
        }


def snapshot_to_wire(snapshot: dict) -> dict:
    result = dict(snapshot)
    result["history_frames"] = [
        base64.b64encode(value).decode("ascii")
        for value in snapshot["history_frames"]
    ]
    result["block_frames"] = [
        base64.b64encode(value).decode("ascii")
        for value in snapshot["block_frames"]
    ]
    return result


def snapshot_from_wire(wire: dict) -> dict:
    if not isinstance(wire, dict):
        raise TypeError("snapshot wire value must be a mapping")
    result = dict(wire)
    try:
        result["history_frames"] = [
            base64.b64decode(value, validate=True)
            for value in wire["history_frames"]
        ]
        result["block_frames"] = [
            base64.b64decode(value, validate=True)
            for value in wire["block_frames"]
        ]
    except Exception as error:
        raise ValueError("snapshot contains invalid base64 RGB") from error
    return result