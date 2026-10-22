"""Variable-duration four-action chunk math for StreamVLN PPO."""

from __future__ import annotations

import math
from collections.abc import Sequence


def _prefix_length(mask: Sequence[bool]) -> int:
    if not mask:
        raise ValueError("executed mask must not be empty")
    length = 0
    padded = False
    for item in mask:
        valid = bool(item)
        if valid:
            if padded:
                raise ValueError("executed mask must be prefix-contiguous")
            length += 1
        else:
            padded = True
    if length < 1:
        raise ValueError("each decision must execute at least one action")
    return length


def collapse_chunk_rewards(
    rewards: Sequence[Sequence[float]],
    executed_masks: Sequence[Sequence[bool]],
    gamma: float,
) -> tuple[list[float], list[int]]:
    """Collapse primitive rewards onto decision boundaries.

    For a decision that executes k primitive actions:
      R_chunk = sum_{j=0}^{k-1} gamma**j * r_j.
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if len(rewards) != len(executed_masks):
        raise ValueError("rewards and masks must have the same decision count")
    collapsed, durations = [], []
    for row, mask in zip(rewards, executed_masks):
        if len(row) != len(mask):
            raise ValueError("reward and mask widths differ")
        k = _prefix_length(mask)
        collapsed.append(sum((gamma**j) * float(row[j]) for j in range(k)))
        durations.append(k)
    return collapsed, durations


def variable_duration_gae(
    chunk_rewards: Sequence[float],
    values: Sequence[float],
    next_values: Sequence[float],
    durations: Sequence[int],
    dones: Sequence[bool],
    gamma: float,
    gae_lambda: float,
) -> tuple[list[float], list[float], list[float]]:
    """GAE at chunk boundaries with physical-step discount semantics.

    delta_t = R_chunk_t + gamma**k_t * (1-done_t) * V_{t+1} - V_t
    A_t = delta_t + (gamma*lambda)**k_t * (1-done_t) * A_{t+1}
    """
    fields = (chunk_rewards, values, next_values, durations, dones)
    n = len(chunk_rewards)
    if any(len(field) != n for field in fields):
        raise ValueError("all GAE fields must have the same length")
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gamma and lambda must be in [0, 1]")
    if any(int(k) < 1 for k in durations):
        raise ValueError("durations must be positive")
    numeric = (*chunk_rewards, *values, *next_values)
    if any(not math.isfinite(float(item)) for item in numeric):
        raise ValueError("GAE inputs must be finite")

    advantages = [0.0] * n
    deltas = [0.0] * n
    running = 0.0
    for index in range(n - 1, -1, -1):
        k = int(durations[index])
        continuation = 0.0 if bool(dones[index]) else 1.0
        delta = (
            float(chunk_rewards[index])
            + (gamma**k) * continuation * float(next_values[index])
            - float(values[index])
        )
        running = delta + ((gamma * gae_lambda) ** k) * continuation * running
        deltas[index] = delta
        advantages[index] = running
    returns = [advantage + float(value) for advantage, value in zip(advantages, values)]
    return advantages, returns, deltas
