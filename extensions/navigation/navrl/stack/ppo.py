"""Dependency-free PPO objective and on-policy ratio diagnostics.

All log probabilities passed here are already joint log probabilities for one
policy decision.  The helpers intentionally return ordinary Python numbers;
they validate PPO arithmetic but do not provide automatic differentiation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from numbers import Real


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must contain real numbers")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must contain finite numbers")
    return result


def _nonnegative_number(value: object, name: str) -> float:
    result = _finite_number(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _active_rows(
    fields: dict[str, Sequence[object]],
    mask: Sequence[bool] | None,
) -> dict[str, tuple[float, ...]]:
    lengths = {len(values) for values in fields.values()}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) < 1:
        raise ValueError("PPO fields must have the same non-zero length")
    count = next(iter(lengths))
    if mask is None:
        active = (True,) * count
    else:
        if len(mask) != count or any(type(item) is not bool for item in mask):
            raise ValueError("PPO mask must be a bool sequence matching the fields")
        active = tuple(mask)
    if not any(active):
        raise ValueError("PPO mask must retain at least one decision")
    return {
        name: tuple(
            _finite_number(value, name)
            for value, keep in zip(values, active)
            if keep
        )
        for name, values in fields.items()
    }


def _ratios(old_log_probs: Sequence[float], new_log_probs: Sequence[float]) -> tuple[float, ...]:
    result = []
    for old, new in zip(old_log_probs, new_log_probs):
        try:
            ratio = math.exp(new - old)
        except OverflowError as exc:
            raise ValueError("PPO log-probability difference overflows the ratio") from exc
        if not math.isfinite(ratio) or ratio <= 0.0:
            raise ValueError("PPO ratio must be finite and positive")
        result.append(ratio)
    return tuple(result)


@dataclass(frozen=True)
class RatioMetrics:
    """Decision-level likelihood ratios after an optional loss mask."""

    ratios: tuple[float, ...]
    mean: float
    minimum: float
    maximum: float
    max_abs_deviation: float
    clip_fraction: float


@dataclass(frozen=True)
class PPOObjective:
    """Numerical PPO objective components and their ratio diagnostics."""

    total_loss: float
    policy_loss: float
    value_loss: float
    entropy_loss: float
    mean_entropy: float
    ratio_metrics: RatioMetrics
    active_count: int


def compute_ratio_metrics(
    old_joint_log_probs: Sequence[object],
    new_joint_log_probs: Sequence[object],
    *,
    clip_ratio_low: float = 0.2,
    clip_ratio_high: float = 0.2,
    mask: Sequence[bool] | None = None,
) -> RatioMetrics:
    """Compute ratios for already-joint decision log probabilities."""

    clip_low = _nonnegative_number(clip_ratio_low, "clip_ratio_low")
    clip_high = _nonnegative_number(clip_ratio_high, "clip_ratio_high")
    if clip_low > 1.0:
        raise ValueError("clip_ratio_low must not exceed 1")
    rows = _active_rows(
        {
            "old_joint_log_probs": old_joint_log_probs,
            "new_joint_log_probs": new_joint_log_probs,
        },
        mask,
    )
    ratios = _ratios(rows["old_joint_log_probs"], rows["new_joint_log_probs"])
    lower, upper = 1.0 - clip_low, 1.0 + clip_high
    return RatioMetrics(
        ratios=ratios,
        mean=sum(ratios) / len(ratios),
        minimum=min(ratios),
        maximum=max(ratios),
        max_abs_deviation=max(abs(value - 1.0) for value in ratios),
        clip_fraction=sum(value < lower or value > upper for value in ratios)
        / len(ratios),
    )


def assert_pre_update_ratio(
    old_joint_log_probs: Sequence[object],
    new_joint_log_probs: Sequence[object],
    *,
    tolerance: float = 1e-6,
    mask: Sequence[bool] | None = None,
) -> RatioMetrics:
    """Fail when replay before an update does not reproduce behavior likelihoods."""

    tolerance = _nonnegative_number(tolerance, "tolerance")
    metrics = compute_ratio_metrics(
        old_joint_log_probs,
        new_joint_log_probs,
        clip_ratio_low=min(tolerance, 1.0),
        clip_ratio_high=tolerance,
        mask=mask,
    )
    if metrics.max_abs_deviation > tolerance:
        raise ValueError(
            "pre-update PPO ratio deviates from 1 by "
            f"{metrics.max_abs_deviation:.12g}, tolerance={tolerance:.12g}"
        )
    return metrics


def compute_ppo_objective(
    *,
    old_joint_log_probs: Sequence[object],
    new_joint_log_probs: Sequence[object],
    old_values: Sequence[object],
    new_values: Sequence[object],
    returns: Sequence[object],
    advantages: Sequence[object],
    entropies: Sequence[object] | None = None,
    clip_ratio_low: float = 0.2,
    clip_ratio_high: float = 0.2,
    value_clip: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.0,
    mask: Sequence[bool] | None = None,
) -> PPOObjective:
    """Evaluate the clipped PPO actor/value/entropy objective in Python.

    This function is a reference and validation helper.  Integrations should
    implement the same formula with their tensor library for backpropagation.
    """

    clip_low = _nonnegative_number(clip_ratio_low, "clip_ratio_low")
    clip_high = _nonnegative_number(clip_ratio_high, "clip_ratio_high")
    if clip_low > 1.0:
        raise ValueError("clip_ratio_low must not exceed 1")
    value_clip = _nonnegative_number(value_clip, "value_clip")
    value_coef = _nonnegative_number(value_coef, "value_coef")
    entropy_coef = _nonnegative_number(entropy_coef, "entropy_coef")

    if entropies is None:
        entropies = (0.0,) * len(old_joint_log_probs)
    rows = _active_rows(
        {
            "old_joint_log_probs": old_joint_log_probs,
            "new_joint_log_probs": new_joint_log_probs,
            "old_values": old_values,
            "new_values": new_values,
            "returns": returns,
            "advantages": advantages,
            "entropies": entropies,
        },
        mask,
    )
    ratios = _ratios(rows["old_joint_log_probs"], rows["new_joint_log_probs"])
    lower, upper = 1.0 - clip_low, 1.0 + clip_high

    policy_terms = []
    value_errors = []
    for ratio, old_value, new_value, target, advantage in zip(
        ratios,
        rows["old_values"],
        rows["new_values"],
        rows["returns"],
        rows["advantages"],
    ):
        clipped_ratio = min(max(ratio, lower), upper)
        policy_terms.append(-min(ratio * advantage, clipped_ratio * advantage))
        clipped_value = old_value + min(max(new_value - old_value, -value_clip), value_clip)
        value_errors.append(
            max((new_value - target) ** 2, (clipped_value - target) ** 2)
        )

    active_count = len(ratios)
    policy_loss = sum(policy_terms) / active_count
    value_loss = 0.5 * sum(value_errors) / active_count
    mean_entropy = sum(rows["entropies"]) / active_count
    entropy_loss = -mean_entropy
    total_loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
    if not all(
        math.isfinite(value)
        for value in (policy_loss, value_loss, mean_entropy, entropy_loss, total_loss)
    ):
        raise ValueError("PPO objective is non-finite")

    ratio_metrics = RatioMetrics(
        ratios=ratios,
        mean=sum(ratios) / active_count,
        minimum=min(ratios),
        maximum=max(ratios),
        max_abs_deviation=max(abs(value - 1.0) for value in ratios),
        clip_fraction=sum(value < lower or value > upper for value in ratios)
        / active_count,
    )
    return PPOObjective(
        total_loss=total_loss,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy_loss=entropy_loss,
        mean_entropy=mean_entropy,
        ratio_metrics=ratio_metrics,
        active_count=active_count,
    )
