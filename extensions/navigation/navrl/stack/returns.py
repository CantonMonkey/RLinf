"""Physical-step reward collapse and duration-aware GAE."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

from .contracts import DecisionBatch, ExecutionReceipt


def _validate_discount(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return value


def _prefix_length(mask: Sequence[bool]) -> int:
    if not mask:
        raise ValueError("executed mask must not be empty")
    duration = 0
    padding = False
    for item in mask:
        if type(item) is not bool:
            raise TypeError("executed mask must contain bools")
        if item:
            if padding:
                raise ValueError("executed mask must be prefix-contiguous")
            duration += 1
        else:
            padding = True
    if duration < 1:
        raise ValueError("each decision must execute at least one primitive")
    return duration


@dataclass(frozen=True)
class GAEOutput:
    discounted_rewards: tuple[float, ...]
    deltas: tuple[float, ...]
    advantages: tuple[float, ...]
    returns: tuple[float, ...]


def collapse_primitive_rewards(
    rewards: Sequence[Sequence[float]],
    executed_masks: Sequence[Sequence[bool]],
    gamma: float,
) -> tuple[list[float], list[int]]:
    gamma = _validate_discount(gamma, "gamma")
    if len(rewards) != len(executed_masks):
        raise ValueError("rewards and masks must have the same decision count")
    collapsed, durations = [], []
    for row, mask in zip(rewards, executed_masks):
        if len(row) != len(mask):
            raise ValueError("reward and mask widths differ")
        duration = _prefix_length(mask)
        active = [float(value) for value in row[:duration]]
        if any(not math.isfinite(value) for value in active):
            raise ValueError("primitive rewards must be finite")
        collapsed.append(sum((gamma**index) * value for index, value in enumerate(active)))
        durations.append(duration)
    return collapsed, durations


def duration_aware_gae(
    chunk_rewards: Sequence[float],
    values: Sequence[float],
    next_values: Sequence[float],
    durations: Sequence[int],
    terminated: Sequence[bool],
    truncated: Sequence[bool],
    gamma: float,
    gae_lambda: float,
) -> GAEOutput:
    gamma = _validate_discount(gamma, "gamma")
    gae_lambda = _validate_discount(gae_lambda, "gae_lambda")
    fields = (values, next_values, durations, terminated, truncated)
    count = len(chunk_rewards)
    if count < 1 or any(len(field) != count for field in fields):
        raise ValueError("all GAE fields must have the same non-zero length")
    numeric = [float(value) for value in (*chunk_rewards, *values, *next_values)]
    if any(not math.isfinite(value) for value in numeric):
        raise ValueError("GAE inputs must be finite")
    if any(type(value) is not int or value < 1 for value in durations):
        raise ValueError("durations must be positive integers")
    if any(type(value) is not bool for value in (*terminated, *truncated)):
        raise TypeError("termination fields must contain bools")
    if any(end and cut for end, cut in zip(terminated, truncated)):
        raise ValueError("a decision cannot terminate and truncate")

    advantages = [0.0] * count
    deltas = [0.0] * count
    running = 0.0
    for index in range(count - 1, -1, -1):
        duration = durations[index]
        bootstrap = 0.0 if terminated[index] else 1.0
        trace = 0.0 if terminated[index] or truncated[index] else 1.0
        delta = (
            float(chunk_rewards[index])
            + (gamma**duration) * bootstrap * float(next_values[index])
            - float(values[index])
        )
        running = delta + ((gamma * gae_lambda) ** duration) * trace * running
        deltas[index] = delta
        advantages[index] = running
    returns = [advantage + float(value) for advantage, value in zip(advantages, values)]
    return GAEOutput(
        discounted_rewards=tuple(float(value) for value in chunk_rewards),
        deltas=tuple(deltas),
        advantages=tuple(advantages),
        returns=tuple(returns),
    )


def discounted_prefix_reward(receipt: ExecutionReceipt, gamma: float) -> float:
    rewards, _ = collapse_primitive_rewards(
        [receipt.primitive_rewards], [receipt.executed_mask], gamma
    )
    return rewards[0]


def compute_duration_aware_gae(
    batch: DecisionBatch,
    gamma: float,
    gae_lambda: float,
) -> GAEOutput:
    rewards = tuple(discounted_prefix_reward(row.execution, gamma) for row in batch.records)
    values = tuple(float(row.sample.value) for row in batch.records)
    next_values = tuple(float(row.next_value) for row in batch.records)
    durations = tuple(row.execution.duration for row in batch.records)
    terminated = tuple(row.execution.terminated for row in batch.records)
    truncated = tuple(row.execution.truncated for row in batch.records)
    return duration_aware_gae(
        rewards,
        values,
        next_values,
        durations,
        terminated,
        truncated,
        gamma,
        gae_lambda,
    )
