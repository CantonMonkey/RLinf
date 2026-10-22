"""Differentiable Torch PPO objective for validated ``navrl.stack`` batches.

This bridge owns tensor assembly and PPO loss arithmetic only.  It deliberately
does not create an optimizer, call ``backward()``, or interact with an
environment.  Torch is imported lazily so the generic stack remains usable in
dependency-free processes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Real

from navrl.stack import DecisionBatch, GAEOutput, PolicyEvaluation


@dataclass(frozen=True)
class TorchPPOObjective:
    """Differentiable loss components plus detached numerical diagnostics."""

    total_loss: object
    policy_loss: object
    value_loss: object
    entropy_loss: object
    diagnostics: Mapping[str, float | int | bool]


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the runtime image
        raise RuntimeError(
            "compute_torch_ppo_objective requires PyTorch in the integration runtime"
        ) from exc
    return torch


def _nonnegative_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _infer_device(torch, values: Sequence[object], requested: object | None):
    if requested is not None:
        return torch.device(requested)
    for value in values:
        if torch.is_tensor(value):
            return value.device
    return torch.device("cpu")


def _scalar_tensor(torch, value: object, name: str, device, *, detach: bool):
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(f"{name} entries must be scalar tensors")
        if value.is_complex():
            raise TypeError(f"{name} entries must be real-valued")
        tensor = value.reshape(())
        if detach:
            tensor = tensor.detach()
        tensor = tensor.to(device=device, dtype=torch.float32)
    else:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} entries must be real numbers or scalar tensors")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{name} entries must be finite")
        tensor = torch.tensor(numeric, device=device, dtype=torch.float32)
    if not bool(torch.isfinite(tensor.detach()).item()):
        raise ValueError(f"{name} entries must be finite")
    return tensor


def _stack_scalars(
    torch,
    values: Sequence[object],
    name: str,
    device,
    *,
    detach: bool,
):
    return torch.stack(
        [
            _scalar_tensor(torch, value, name, device, detach=detach)
            for value in values
        ]
    )


def _validate_alignment(
    batch: DecisionBatch,
    evaluations: tuple[PolicyEvaluation, ...],
    gae: GAEOutput,
) -> None:
    if not isinstance(batch, DecisionBatch):
        raise TypeError("batch must be a DecisionBatch")
    if not isinstance(gae, GAEOutput):
        raise TypeError("gae must be a GAEOutput")
    count = len(batch.records)
    if len(evaluations) != count:
        raise ValueError("evaluations must contain one row per decision")
    gae_fields = {
        "discounted_rewards": gae.discounted_rewards,
        "deltas": gae.deltas,
        "advantages": gae.advantages,
        "returns": gae.returns,
    }
    if any(len(values) != count for values in gae_fields.values()):
        raise ValueError("every GAE field must align with the DecisionBatch")

    for position, (record, evaluation) in enumerate(zip(batch.records, evaluations)):
        if not isinstance(evaluation, PolicyEvaluation):
            raise TypeError("evaluations must contain PolicyEvaluation rows")
        if evaluation.identity != record.sample.identity:
            raise ValueError(f"evaluation identity mismatch at decision {position}")
        if evaluation.decision_index != record.sample.decision_index:
            raise ValueError(f"evaluation index mismatch at decision {position}")
        if evaluation.policy_version != record.sample.policy_version:
            raise ValueError(f"evaluation policy version mismatch at decision {position}")


def compute_torch_ppo_objective(
    batch: DecisionBatch,
    evaluations: Sequence[PolicyEvaluation],
    gae: GAEOutput,
    *,
    clip_ratio_low: float = 0.2,
    clip_ratio_high: float = 0.2,
    value_clip: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.0,
    pre_update_tolerance: float | None = 0.03,
    device: object | None = None,
) -> TorchPPOObjective:
    """Build the clipped PPO loss from one identity-aligned replay batch.

    ``pre_update_tolerance`` checks that replay under the behavior policy gives
    ratios near one.  Set it to ``None`` only for later optimization epochs over
    the same behavior batch, after parameters have already changed.
    """

    torch = _torch()
    clip_low = _nonnegative_number(clip_ratio_low, "clip_ratio_low")
    clip_high = _nonnegative_number(clip_ratio_high, "clip_ratio_high")
    if clip_low > 1.0:
        raise ValueError("clip_ratio_low must not exceed 1")
    value_clip = _nonnegative_number(value_clip, "value_clip")
    value_coef = _nonnegative_number(value_coef, "value_coef")
    entropy_coef = _nonnegative_number(entropy_coef, "entropy_coef")
    tolerance = (
        None
        if pre_update_tolerance is None
        else _nonnegative_number(pre_update_tolerance, "pre_update_tolerance")
    )

    try:
        evaluation_rows = tuple(evaluations)
    except TypeError as exc:
        raise TypeError("evaluations must be a sequence") from exc
    _validate_alignment(batch, evaluation_rows, gae)

    device_inputs = [
        value
        for evaluation in evaluation_rows
        for value in (evaluation.joint_log_prob, evaluation.value, evaluation.entropy)
        if value is not None
    ]
    tensor_device = _infer_device(torch, device_inputs, device)

    old_joint_log_probs = _stack_scalars(
        torch,
        tuple(record.old_joint_log_prob for record in batch.records),
        "old_joint_log_probs",
        tensor_device,
        detach=True,
    )
    new_joint_log_probs = _stack_scalars(
        torch,
        tuple(evaluation.joint_log_prob for evaluation in evaluation_rows),
        "new_joint_log_probs",
        tensor_device,
        detach=False,
    )
    old_values = _stack_scalars(
        torch,
        tuple(record.sample.value for record in batch.records),
        "old_values",
        tensor_device,
        detach=True,
    )
    new_values = _stack_scalars(
        torch,
        tuple(evaluation.value for evaluation in evaluation_rows),
        "new_values",
        tensor_device,
        detach=False,
    )
    advantages = _stack_scalars(
        torch,
        gae.advantages,
        "advantages",
        tensor_device,
        detach=True,
    )
    returns = _stack_scalars(
        torch,
        gae.returns,
        "returns",
        tensor_device,
        detach=True,
    )

    entropy_inputs = tuple(evaluation.entropy for evaluation in evaluation_rows)
    if any(value is None for value in entropy_inputs):
        if not all(value is None for value in entropy_inputs):
            raise ValueError("entropy must be provided for every evaluation or none")
        if entropy_coef != 0.0:
            raise ValueError("entropy is required when entropy_coef is non-zero")
        entropies = new_joint_log_probs * 0.0
    else:
        entropies = _stack_scalars(
            torch,
            entropy_inputs,
            "entropies",
            tensor_device,
            detach=False,
        )

    ratios = torch.exp(new_joint_log_probs - old_joint_log_probs)
    detached_ratios = ratios.detach()
    if not bool(torch.isfinite(detached_ratios).all().item()) or bool(
        (detached_ratios <= 0.0).any().item()
    ):
        raise ValueError("PPO ratios must be finite and positive")
    ratio_deviations = torch.abs(detached_ratios - 1.0)
    max_ratio_deviation = float(ratio_deviations.max().item())
    if tolerance is not None and max_ratio_deviation > tolerance:
        raise ValueError(
            "pre-update PPO ratio deviates from 1 by "
            f"{max_ratio_deviation:.12g}, tolerance={tolerance:.12g}"
        )

    lower, upper = 1.0 - clip_low, 1.0 + clip_high
    clipped_ratios = torch.clamp(ratios, min=lower, max=upper)
    policy_loss = -torch.minimum(
        ratios * advantages, clipped_ratios * advantages
    ).mean()

    clipped_values = old_values + torch.clamp(
        new_values - old_values, min=-value_clip, max=value_clip
    )
    value_loss = 0.5 * torch.maximum(
        torch.square(new_values - returns),
        torch.square(clipped_values - returns),
    ).mean()
    mean_entropy = entropies.mean()
    entropy_loss = -mean_entropy
    total_loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

    loss_values = torch.stack(
        (total_loss.detach(), policy_loss.detach(), value_loss.detach(), entropy_loss.detach())
    )
    if not bool(torch.isfinite(loss_values).all().item()):
        raise ValueError("PPO objective is non-finite")

    clipped = (detached_ratios < lower) | (detached_ratios > upper)
    diagnostics: dict[str, float | int | bool] = {
        "active_count": len(batch.records),
        "policy_version": batch.policy_version,
        "pre_update_guard_applied": tolerance is not None,
        "ratio_mean": float(detached_ratios.mean().item()),
        "ratio_min": float(detached_ratios.min().item()),
        "ratio_max": float(detached_ratios.max().item()),
        "ratio_max_abs_deviation": max_ratio_deviation,
        "ratio_clip_fraction": float(clipped.float().mean().item()),
        "policy_loss": float(policy_loss.detach().item()),
        "value_loss": float(value_loss.detach().item()),
        "entropy_loss": float(entropy_loss.detach().item()),
        "mean_entropy": float(mean_entropy.detach().item()),
        "total_loss": float(total_loss.detach().item()),
    }
    return TorchPPOObjective(
        total_loss=total_loss,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy_loss=entropy_loss,
        diagnostics=diagnostics,
    )


__all__ = ["TorchPPOObjective", "compute_torch_ppo_objective"]
