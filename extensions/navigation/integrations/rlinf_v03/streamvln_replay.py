"""Fixed-shape replay rows for shuffled StreamVLN categorical batches."""

from __future__ import annotations

from dataclasses import dataclass

try:
    from .streamvln_context import (
        ACTION_CHUNK,
        BLOCK_ACTIONS,
        BLOCK_OBSERVATIONS,
        HISTORY_OBSERVATIONS,
        decode_fixed_utf8,
        encode_fixed_utf8,
    )
except ImportError:  # Direct staged-file contract tests and actor server.
    from streamvln_context import (
        ACTION_CHUNK,
        BLOCK_ACTIONS,
        BLOCK_OBSERVATIONS,
        HISTORY_OBSERVATIONS,
        decode_fixed_utf8,
        encode_fixed_utf8,
    )


SCALAR_KEYS = (
    "env_id",
    "episode_serial",
    "step",
    "block_start",
)
TEXT_FIELDS = (
    ("instruction", "instruction_utf8"),
    ("scene", "scene_utf8"),
    ("episode_id", "episode_utf8"),
)


@dataclass(frozen=True)
class ReplaySpec:
    height: int
    width: int
    instruction_bytes: int = 512
    scene_bytes: int = 384
    episode_bytes: int = 64

    def __post_init__(self) -> None:
        if self.height < 1 or self.width < 1:
            raise ValueError("replay image shape must be positive")
        if min(self.instruction_bytes, self.scene_bytes, self.episode_bytes) < 1:
            raise ValueError("replay text widths must be positive")

    @property
    def image_bytes(self) -> int:
        return self.height * self.width * 3


def _pad_frames(frames, count: int, image_bytes: int):
    if len(frames) > count:
        raise ValueError("too many replay frames")
    values = []
    mask = []
    for frame in frames:
        if not isinstance(frame, bytes) or len(frame) != image_bytes:
            raise ValueError("replay RGB byte count mismatch")
        values.append(list(frame))
        mask.append(True)
    while len(values) < count:
        values.append([0] * image_bytes)
        mask.append(False)
    return values, mask


def _pad_actions(actions, count: int):
    if len(actions) > count:
        raise ValueError("too many replay actions")
    if any(type(value) is not int or not 0 <= value < 4 for value in actions):
        raise ValueError("replay action outside 0..3")
    return list(actions) + [0] * (count - len(actions)), [
        index < len(actions) for index in range(count)
    ]


def pack_snapshot(snapshot: dict, spec: ReplaySpec) -> dict:
    if snapshot.get("image_shape") != [spec.height, spec.width, 3]:
        raise ValueError("snapshot image shape differs from replay spec")
    step = snapshot.get("step")
    block_start = snapshot.get("block_start")
    if (
        type(step) is not int
        or type(block_start) is not int
        or step < 0
        or step % ACTION_CHUNK
        or block_start != (step // BLOCK_ACTIONS) * BLOCK_ACTIONS
    ):
        raise ValueError("snapshot step/block_start contract mismatch")
    expected_history = 0 if block_start == 0 else HISTORY_OBSERVATIONS
    expected_block = (step - block_start) // ACTION_CHUNK + 1
    if len(snapshot["history_frames"]) != expected_history:
        raise ValueError("snapshot history frame count mismatch")
    if len(snapshot["block_frames"]) != expected_block:
        raise ValueError("snapshot block frame count mismatch")
    if len(snapshot["block_actions"]) != step - block_start:
        raise ValueError("snapshot block action count mismatch")
    if not 1 <= len(snapshot["current_actions"]) <= ACTION_CHUNK:
        raise ValueError("snapshot current action count mismatch")

    row = {key: int(snapshot[key]) for key in SCALAR_KEYS}
    for source, target in TEXT_FIELDS:
        width = getattr(spec, target.replace("_utf8", "_bytes"))
        values, mask = encode_fixed_utf8(str(snapshot[source]), width)
        row[target] = values
        row[target + "_mask"] = mask
    row["history_rgb"], row["history_mask"] = _pad_frames(
        snapshot["history_frames"],
        HISTORY_OBSERVATIONS,
        spec.image_bytes,
    )
    row["block_rgb"], row["block_mask"] = _pad_frames(
        snapshot["block_frames"],
        BLOCK_OBSERVATIONS,
        spec.image_bytes,
    )
    row["block_actions"], row["block_action_mask"] = _pad_actions(
        snapshot["block_actions"],
        BLOCK_ACTIONS,
    )
    row["current_actions"], row["current_action_mask"] = _pad_actions(
        snapshot["current_actions"],
        ACTION_CHUNK,
    )
    return row


def _unpad_frames(values, mask, image_bytes):
    if len(values) != len(mask):
        raise ValueError("replay frame values/mask length mismatch")
    frames = []
    seen_padding = False
    for frame, valid in zip(values, mask):
        if type(valid) is not bool:
            raise TypeError("replay frame mask must be boolean")
        if len(frame) != image_bytes:
            raise ValueError("replay fixed RGB width mismatch")
        if valid:
            if seen_padding:
                raise ValueError("replay frame mask must be prefix-contiguous")
            if any(type(value) is not int or not 0 <= value <= 255 for value in frame):
                raise TypeError("replay RGB tensor must contain bytes")
            frames.append(bytes(frame))
        else:
            seen_padding = True
            if any(frame):
                raise ValueError("replay RGB padding must be zero")
    return frames


def _unpad_actions(values, mask):
    if len(values) != len(mask):
        raise ValueError("replay action values/mask length mismatch")
    actions = []
    seen_padding = False
    for value, valid in zip(values, mask):
        if type(valid) is not bool:
            raise TypeError("replay action mask must be boolean")
        if type(value) is not int or not 0 <= value < 4:
            raise TypeError("replay action tensor must contain categories")
        if valid:
            if seen_padding:
                raise ValueError("replay action mask must be prefix-contiguous")
            actions.append(value)
        else:
            seen_padding = True
            if value != 0:
                raise ValueError("replay action padding must be STOP/zero")
    return actions


def unpack_row(row: dict, spec: ReplaySpec) -> dict:
    result = {key: int(row[key]) for key in SCALAR_KEYS}
    for source, target in TEXT_FIELDS:
        result[source] = decode_fixed_utf8(
            row[target],
            row[target + "_mask"],
        )
    result["image_shape"] = [spec.height, spec.width, 3]
    result["history_frames"] = _unpad_frames(
        row["history_rgb"],
        row["history_mask"],
        spec.image_bytes,
    )
    result["block_frames"] = _unpad_frames(
        row["block_rgb"],
        row["block_mask"],
        spec.image_bytes,
    )
    result["block_actions"] = _unpad_actions(
        row["block_actions"],
        row["block_action_mask"],
    )
    result["current_actions"] = _unpad_actions(
        row["current_actions"],
        row["current_action_mask"],
    )
    block_start = result["block_start"]
    step = result["step"]
    stride = block_start // HISTORY_OBSERVATIONS if block_start else 0
    result["history_steps"] = (
        list(range(0, block_start, stride)) if block_start else []
    )
    result["block_steps"] = list(
        range(block_start, step + 1, ACTION_CHUNK)
    )
    # Repack validation makes every restored/shuffled row independent.
    pack_snapshot(result, spec)
    return result


def stack_rows(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("cannot stack an empty replay batch")
    keys = set(rows[0])
    if any(set(row) != keys for row in rows):
        raise ValueError("replay rows have different schemas")
    return {key: [row[key] for row in rows] for key in rows[0]}


def unstack_row(batch: dict, index: int) -> dict:
    if not batch:
        raise ValueError("cannot unstack an empty replay batch")
    sizes = {len(values) for values in batch.values()}
    if len(sizes) != 1:
        raise ValueError("replay batch fields have different batch sizes")
    size = sizes.pop()
    if not 0 <= index < size:
        raise IndexError(index)
    return {key: values[index] for key, values in batch.items()}