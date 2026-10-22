"""Decision-row replay and PPO adapter for the P8 successor worker trio."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:  # Lightweight schema/math tests do not require Torch.
    torch = None
    F = None

from rlinf.algorithms.registry import (
    ADV_REGISTRY,
    LOSS_REGISTRY,
    get_adv_and_returns,
    get_policy_loss,
    register_advantage,
    register_policy_loss,
)
from rlinf.utils.metric_utils import append_to_dict, compute_split_num
from rlinf.utils.nested_dict_process import put_tensor_device
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor

try:
    from .streamvln_p8_successor_action import ACTION_CHUNK, RouteName, STOP_ACTION
    from .streamvln_p8_successor_protocol import (
        SUCCESSOR_REPLAY_SCHEMA,
        MacroTransition,
        compute_successor_gae,
    )
except ImportError:  # Direct staged-file tests.
    from streamvln_p8_successor_action import ACTION_CHUNK, RouteName, STOP_ACTION
    from streamvln_p8_successor_protocol import (
        SUCCESSOR_REPLAY_SCHEMA,
        MacroTransition,
        compute_successor_gae,
    )


SUCCESSOR_ADVANTAGE_NAME = "p8_successor_duration_gae"
SUCCESSOR_POLICY_LOSS_NAME = "p8_successor_joint_ppo"


def _cfg_get(value, key, default=None):
    getter = getattr(value, "get", None)
    return getter(key, default) if callable(getter) else getattr(value, key, default)


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _validate_context(context: object) -> Mapping[str, object]:
    if not isinstance(context, Mapping) or not context:
        raise ValueError("successor replay requires a non-empty forward_context mapping")

    def check(value: object, path: str) -> None:
        if isinstance(value, Mapping):
            if not value:
                raise ValueError(f"empty successor forward_context mapping at {path}")
            for key, child in value.items():
                if not isinstance(key, str) or not key:
                    raise ValueError("successor forward_context keys must be non-empty strings")
                check(child, f"{path}.{key}")
            return
        if torch is not None and not isinstance(value, torch.Tensor):
            raise TypeError(f"successor forward_context leaf {path} must be a tensor")

    check(context, "forward_context")
    return context


def _stack_context(values: Sequence[object], path: str = "forward_context"):
    if torch is None:
        raise RuntimeError("Torch is required to tensorize successor replay")
    first = values[0]
    if isinstance(first, Mapping):
        keys = tuple(first)
        if any(not isinstance(value, Mapping) or tuple(value) != keys for value in values):
            raise ValueError(f"successor replay context schemas differ at {path}")
        return {
            key: _stack_context([value[key] for value in values], f"{path}.{key}")
            for key in keys
        }
    if any(not isinstance(value, torch.Tensor) for value in values):
        raise TypeError(f"successor replay context leaves differ at {path}")
    try:
        return torch.stack([value.detach().cpu() for value in values], dim=0).unsqueeze(1)
    except RuntimeError as exc:
        raise ValueError(f"successor replay context tensor shapes differ at {path}") from exc


@dataclass(frozen=True)
class SuccessorReplayRow:
    """One fully bootstrapped PPO row for one high-level decision."""

    episode_token: str
    physical_step: int
    decision_index: int
    route: RouteName
    waypoint_local_m: tuple[float, float]
    proposed_actions: tuple[int, ...]
    old_joint_logprob: float
    old_value: float
    official_shadow_action: int
    executed_actions: tuple[int, ...]
    executed_mask: tuple[bool, ...]
    primitive_rewards: tuple[float, ...]
    aggregated_reward: float
    duration: int
    terminated: bool
    truncated: bool
    next_value: float
    forward_context: Mapping[str, object]
    expert_route_target: RouteName | None = None
    expert_waypoint_target: tuple[float, float] | None = None
    truncation_final_observation: object | None = None
    decision_mask: bool = True
    valid_padding: bool = True

    def __post_init__(self) -> None:
        if not self.episode_token or self.physical_step < 0 or self.decision_index < 0:
            raise ValueError("successor replay identity is invalid")
        object.__setattr__(self, "route", RouteName(self.route))
        if len(self.waypoint_local_m) != 2:
            raise ValueError("waypoint target must contain two coordinates")
        if not 1 <= len(self.proposed_actions) <= ACTION_CHUNK:
            raise ValueError("proposed prefix must contain one to four primitives")
        if len(self.executed_actions) != ACTION_CHUNK:
            raise ValueError("executed actions must use the fixed width")
        if len(self.executed_mask) != ACTION_CHUNK:
            raise ValueError("executed mask must use the fixed width")
        if len(self.primitive_rewards) != ACTION_CHUNK:
            raise ValueError("primitive rewards must use the fixed width")
        if self.duration != sum(self.executed_mask):
            raise ValueError("duration and executed mask disagree")
        for name in ("old_joint_logprob", "old_value", "aggregated_reward", "next_value"):
            _finite(getattr(self, name), name)
        if type(self.terminated) is not bool or type(self.truncated) is not bool:
            raise ValueError("termination flags must be bools")
        if self.terminated and self.truncated:
            raise ValueError("a successor row cannot terminate and truncate")
        if self.terminated and self.next_value != 0.0:
            raise ValueError("terminated successor rows require next_value=0")
        if self.truncated != (self.truncation_final_observation is not None):
            raise ValueError(
                "truncation rows must exclusively retain their final observation"
            )
        if self.decision_mask is not True or self.valid_padding is not True:
            raise ValueError("emitted successor rows must be valid decision rows")
        if self.expert_route_target is None:
            if self.expert_waypoint_target is not None:
                raise ValueError("expert waypoint requires an expert route")
        else:
            object.__setattr__(self, "expert_route_target", RouteName(self.expert_route_target))
            if self.expert_route_target is RouteName.RECOVER_PARENT:
                raise ValueError("RECOVER_PARENT cannot be an expert policy target")
            if self.expert_route_target is RouteName.WAYPOINT:
                if self.expert_waypoint_target is None or len(self.expert_waypoint_target) != 2:
                    raise ValueError("WAYPOINT expert requires a 2D waypoint")
            elif self.expert_waypoint_target not in (None, (0.0, 0.0)):
                raise ValueError("KEEP/STOP expert cannot carry a waypoint")
        _validate_context(self.forward_context)


@dataclass(frozen=True)
class SuccessorReplayBatch:
    """One env-stage epoch payload consumed by the successor actor worker."""

    rows: tuple[SuccessorReplayRow, ...]
    schema: str = SUCCESSOR_REPLAY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SUCCESSOR_REPLAY_SCHEMA:
            raise ValueError("successor replay schema mismatch")
        if not self.rows or any(not isinstance(row, SuccessorReplayRow) for row in self.rows):
            raise ValueError("successor replay batch must contain typed decision rows")

    @classmethod
    def merge(cls, batches: Sequence["SuccessorReplayBatch"]):
        batches = tuple(batches)
        if not batches or any(not isinstance(batch, cls) for batch in batches):
            raise TypeError("successor actor received an untyped replay payload")
        return cls(tuple(row for batch in batches for row in batch.rows))

    def to_actor_batch(self) -> dict[str, object]:
        rows = self.rows
        route_index = {
            RouteName.KEEP_ORIGINAL: 0,
            RouteName.WAYPOINT: 1,
            RouteName.STOP: 2,
        }
        proposed_actions = []
        proposed_mask = []
        for row in rows:
            width = len(row.proposed_actions)
            proposed_actions.append(row.proposed_actions + (STOP_ACTION,) * (ACTION_CHUNK - width))
            proposed_mask.append((True,) * width + (False,) * (ACTION_CHUNK - width))
        decision_mask = torch.tensor([[row.decision_mask] for row in rows], dtype=torch.bool)
        valid_padding = torch.tensor([[row.valid_padding] for row in rows], dtype=torch.bool)
        expert_target_mask = torch.tensor(
            [[row.expert_route_target is not None] for row in rows], dtype=torch.bool
        )
        return {
            "prev_logprobs": torch.tensor(
                [[row.old_joint_logprob] for row in rows], dtype=torch.float32
            ),
            "old_values": torch.tensor([[row.old_value] for row in rows], dtype=torch.float32),
            "next_values": torch.tensor([[row.next_value] for row in rows], dtype=torch.float32),
            "rewards": torch.tensor(
                [[row.aggregated_reward] for row in rows], dtype=torch.float32
            ),
            "durations": torch.tensor([[row.duration] for row in rows], dtype=torch.int64),
            "terminated": torch.tensor([[row.terminated] for row in rows], dtype=torch.bool),
            "truncated": torch.tensor([[row.truncated] for row in rows], dtype=torch.bool),
            "decision_mask": decision_mask,
            "valid_padding": valid_padding,
            "loss_mask": decision_mask & valid_padding,
            "route_actions": torch.tensor(
                [[route_index[row.route]] for row in rows], dtype=torch.int64
            ),
            "waypoint_targets": torch.tensor(
                [[row.waypoint_local_m] for row in rows], dtype=torch.float32
            ),
            "waypoint_mask": torch.tensor(
                [[row.route is RouteName.WAYPOINT] for row in rows], dtype=torch.bool
            ),
            "expert_target_mask": expert_target_mask,
            "expert_route_targets": torch.tensor(
                [[route_index.get(row.expert_route_target, 0)] for row in rows],
                dtype=torch.int64,
            ),
            "expert_waypoint_targets": torch.tensor(
                [[row.expert_waypoint_target or (0.0, 0.0)] for row in rows],
                dtype=torch.float32,
            ),
            "expert_waypoint_mask": torch.tensor(
                [[row.expert_route_target is RouteName.WAYPOINT] for row in rows],
                dtype=torch.bool,
            ),
            "proposed_actions": torch.tensor(proposed_actions, dtype=torch.int64).unsqueeze(1),
            "proposed_action_mask": torch.tensor(proposed_mask, dtype=torch.bool).unsqueeze(1),
            "forward_context": _stack_context([row.forward_context for row in rows]),
        }


class SuccessorActorReplayAdapter:
    successor_replay_schema = SUCCESSOR_REPLAY_SCHEMA

    def __init__(self, gamma: float) -> None:
        if not 0.0 <= float(gamma) <= 1.0:
            raise ValueError("successor replay gamma must be in [0, 1]")
        self.gamma = float(gamma)

    def convert(
        self,
        *,
        macro: MacroTransition,
        next_value: float,
        stage_id: int,
        env_slot: int,
        final_observation: object | None = None,
    ) -> SuccessorReplayRow:
        if not isinstance(macro, MacroTransition):
            raise TypeError("successor adapter requires MacroTransition")
        if stage_id != 0 or env_slot != 0:
            raise ValueError("successor replay currently supports only stage 0, env slot 0")
        sample = macro.decision_sample
        context = _validate_context(sample.forward_context)
        aggregated = sum(
            (self.gamma**offset) * macro.primitive_rewards[offset]
            for offset in range(macro.duration)
        )
        return SuccessorReplayRow(
            episode_token=macro.episode_token,
            physical_step=macro.physical_step,
            decision_index=macro.decision_index,
            route=sample.decision.route,
            waypoint_local_m=sample.decision.waypoint_local_m,
            proposed_actions=sample.proposed_actions,
            old_joint_logprob=sample.logprob,
            old_value=sample.value,
            official_shadow_action=sample.official_shadow_action,
            executed_actions=macro.executed_actions,
            executed_mask=macro.executed_mask,
            primitive_rewards=macro.primitive_rewards,
            aggregated_reward=aggregated,
            duration=macro.duration,
            terminated=macro.terminated,
            truncated=macro.truncated,
            next_value=_finite(next_value, "next_value"),
            forward_context=context,
            expert_route_target=sample.expert_route_target,
            expert_waypoint_target=sample.expert_waypoint_target,
            truncation_final_observation=final_observation,
        )

    def build_batch(self, rows: Sequence[SuccessorReplayRow]) -> SuccessorReplayBatch:
        return SuccessorReplayBatch(tuple(rows))


def build_successor_actor_replay_adapter(*, worker) -> SuccessorActorReplayAdapter:
    gamma = worker.cfg.algorithm.get("gamma", 1.0)
    return SuccessorActorReplayAdapter(gamma)


build_successor_actor_replay_adapter.successor_replay_schema = SUCCESSOR_REPLAY_SCHEMA


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    count = mask.count_nonzero().clamp(min=1)
    return torch.where(mask, value, torch.zeros_like(value)).sum() / count


def successor_bc_waypoint_terms(
    *,
    expert_waypoint_logprobs: torch.Tensor,
    waypoint_predictions: torch.Tensor,
    expert_waypoint_targets: torch.Tensor,
    expert_waypoint_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute BC waypoint terms without cross-row broadcasting."""

    if waypoint_predictions.shape != expert_waypoint_targets.shape:
        raise ValueError("successor BC waypoint prediction/target shape mismatch")
    if waypoint_predictions.ndim != 3 or waypoint_predictions.shape[-1] != 2:
        raise ValueError("successor BC waypoint tensors must have shape [B, 1, 2]")
    expected_scalar_shape = waypoint_predictions.shape[:-1]
    if (
        expert_waypoint_logprobs.shape != expected_scalar_shape
        or expert_waypoint_mask.shape != expected_scalar_shape
    ):
        raise ValueError("successor BC waypoint scalar shape mismatch")
    expert_waypoint_mask = expert_waypoint_mask.bool()
    waypoint_count = expert_waypoint_mask.count_nonzero().clamp(min=1)
    waypoint_nll = torch.where(
        expert_waypoint_mask,
        -expert_waypoint_logprobs,
        torch.zeros_like(expert_waypoint_logprobs),
    ).sum() / waypoint_count
    waypoint_regression_rows = (
        waypoint_predictions - expert_waypoint_targets
    ).square().mean(dim=-1)
    waypoint_regression = torch.where(
        expert_waypoint_mask,
        waypoint_regression_rows,
        torch.zeros_like(waypoint_regression_rows),
    ).sum() / waypoint_count
    return waypoint_nll, waypoint_regression


def successor_duration_gae_core(
    rewards: Sequence[float],
    values: Sequence[float],
    next_values: Sequence[float],
    durations: Sequence[int],
    terminated: Sequence[bool],
    truncated: Sequence[bool],
    gamma: float,
    gae_lambda: float,
) -> tuple[list[float], list[float]]:
    advantages, returns, _ = compute_successor_gae(
        rewards,
        values,
        next_values,
        durations,
        terminated,
        truncated,
        gamma,
        gae_lambda,
        duration_aware=True,
    )
    return advantages, returns


def successor_joint_policy_core(
    route_logprob,
    waypoint_logprob,
    old_logprob,
    advantage,
    clip_ratio_low: float,
    clip_ratio_high: float,
    *,
    waypoint_mask=True,
):
    if torch is not None and isinstance(route_logprob, torch.Tensor):
        active_waypoint_logprob = torch.where(
            waypoint_mask.bool(), waypoint_logprob, torch.zeros_like(waypoint_logprob)
        )
    else:
        active_waypoint_logprob = waypoint_logprob if bool(waypoint_mask) else 0.0
    log_ratio = route_logprob + active_waypoint_logprob - old_logprob
    if torch is not None and isinstance(log_ratio, torch.Tensor):
        ratio = torch.exp(log_ratio)
        clipped = torch.clamp(
            ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high
        )
        policy = torch.maximum(-advantage * ratio, -advantage * clipped)
    else:
        ratio = math.exp(float(log_ratio))
        clipped = min(max(ratio, 1.0 - clip_ratio_low), 1.0 + clip_ratio_high)
        policy = max(-float(advantage) * ratio, -float(advantage) * clipped)
    return ratio, policy


def successor_duration_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    durations: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    gamma: float = 1.0,
    gae_lambda: float = 1.0,
    loss_mask: torch.Tensor | None = None,
    normalize_advantages: bool = False,
    **_kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decision-boundary GAE with physical-duration discount powers."""

    if torch is None:
        raise RuntimeError("Torch is required for successor advantage tensors")
    tensors = (rewards, values, next_values, durations, terminated, truncated)
    if any(tensor.shape != rewards.shape for tensor in tensors):
        raise ValueError("successor GAE tensors must have identical shapes")
    if rewards.ndim != 2 or rewards.shape[1] != 1:
        raise ValueError("successor GAE expects [decision, 1] tensors")
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("successor gamma and lambda must be in [0, 1]")
    if bool((durations < 1).any()) or bool((terminated & truncated).any()):
        raise ValueError("successor GAE duration or terminal flags are invalid")
    if loss_mask is None:
        loss_mask = torch.ones_like(terminated)
    loss_mask = loss_mask.bool()
    advantages_raw, returns_raw = successor_duration_gae_core(
        rewards.detach().cpu().reshape(-1).tolist(),
        values.detach().cpu().reshape(-1).tolist(),
        next_values.detach().cpu().reshape(-1).tolist(),
        durations.detach().cpu().reshape(-1).tolist(),
        terminated.detach().cpu().reshape(-1).tolist(),
        truncated.detach().cpu().reshape(-1).tolist(),
        gamma,
        gae_lambda,
    )
    advantages = torch.tensor(
        advantages_raw, dtype=torch.float32, device=rewards.device
    ).reshape_as(rewards)
    returns = torch.tensor(
        returns_raw, dtype=torch.float32, device=rewards.device
    ).reshape_as(rewards)
    advantages = advantages * loss_mask
    if normalize_advantages and loss_mask.any():
        valid = advantages[loss_mask]
        advantages = torch.where(
            loss_mask,
            (advantages - valid.mean()) / valid.std(unbiased=False).clamp(min=1e-6),
            torch.zeros_like(advantages),
        )
    return advantages, returns


def successor_joint_ppo_loss(
    *,
    route_logprobs: torch.Tensor,
    waypoint_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    waypoint_predictions: torch.Tensor,
    waypoint_targets: torch.Tensor,
    waypoint_mask: torch.Tensor,
    decision_mask: torch.Tensor,
    valid_padding: torch.Tensor,
    clip_ratio_low: float,
    clip_ratio_high: float,
    value_clip: float,
    waypoint_coef: float,
    value_coef: float,
    **_kwargs,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """PPO over the sampled route+waypoint joint action, once per macro."""

    if torch is None:
        raise RuntimeError("Torch is required for successor PPO tensors")
    mask = decision_mask.bool() & valid_padding.bool()
    expected = old_logprobs.shape
    if any(
        tensor.shape != expected
        for tensor in (
            route_logprobs,
            waypoint_logprobs,
            advantages,
            values,
            old_values,
            returns,
            mask,
        )
    ):
        raise ValueError("successor joint PPO scalar tensors must share shape")
    if waypoint_predictions.shape != waypoint_targets.shape or waypoint_predictions.shape[-1] != 2:
        raise ValueError("successor waypoint prediction must match [..., 2] targets")
    if waypoint_mask.shape != expected:
        raise ValueError("successor waypoint mask shape mismatch")
    ratio, policy = successor_joint_policy_core(
        route_logprobs,
        waypoint_logprobs,
        old_logprobs,
        advantages,
        clip_ratio_low,
        clip_ratio_high,
        waypoint_mask=waypoint_mask,
    )
    policy_loss = _masked_mean(policy, mask)
    clipped_values = old_values + (values - old_values).clamp(-value_clip, value_clip)
    value_error = torch.maximum((values - returns).square(), (clipped_values - returns).square())
    value_loss = 0.5 * _masked_mean(value_error, mask)
    waypoint_rows = waypoint_mask.bool() & mask
    waypoint_error = (waypoint_predictions - waypoint_targets).square().mean(dim=-1)
    waypoint_loss = _masked_mean(waypoint_error, waypoint_rows)
    total = policy_loss + float(value_coef) * value_loss + float(waypoint_coef) * waypoint_loss
    metrics = {
        "actor/policy_loss": policy_loss.detach(),
        "actor/value_loss": value_loss.detach(),
        "actor/waypoint_loss": waypoint_loss.detach(),
        "actor/joint_ratio": _masked_mean(ratio.detach(), mask),
    }
    return total, metrics


def install_successor_algorithm_registry() -> None:
    registrations = (
        (
            ADV_REGISTRY,
            SUCCESSOR_ADVANTAGE_NAME,
            successor_duration_gae,
            register_advantage,
        ),
        (
            LOSS_REGISTRY,
            SUCCESSOR_POLICY_LOSS_NAME,
            successor_joint_ppo_loss,
            register_policy_loss,
        ),
    )
    for registry, name, function, decorator in registrations:
        existing = registry.get(name)
        if (
            existing is not None
            and existing is not function
            and getattr(existing, "__wrapped__", None) is not function
        ):
            raise RuntimeError(f"refusing to overwrite existing algorithm registry entry: {name}")
        if existing is None:
            decorator(name)(function)


install_successor_algorithm_registry()


def require_successor_sync_state_names(param_names_need_sync) -> None:
    """Fail closed unless RLinf will sync readiness and its matching version."""

    names = set(param_names_need_sync or ())
    required = {"bc_ready", "bc_readiness_digest", "policy_version"}
    missing = required - names
    if missing:
        raise RuntimeError(
            "successor persistent sync state is missing: "
            + ", ".join(sorted(missing))
        )


class P8SuccessorActorWorker(EmbodiedFSDPActor):
    """FSDP actor consuming only typed successor decision-row batches."""

    successor_replay_schema = SUCCESSOR_REPLAY_SCHEMA

    def __init__(self, cfg):
        if torch is None:
            raise RuntimeError("Torch is required to launch the successor actor worker")
        if cfg.algorithm.adv_type != SUCCESSOR_ADVANTAGE_NAME:
            raise ValueError(f"successor actor requires adv_type={SUCCESSOR_ADVANTAGE_NAME}")
        if cfg.algorithm.loss_type != SUCCESSOR_POLICY_LOSS_NAME:
            raise ValueError(f"successor actor requires loss_type={SUCCESSOR_POLICY_LOSS_NAME}")
        bc_cfg = cfg.algorithm.get("successor_bc", {})
        self.successor_bc_enabled = bool(_cfg_get(bc_cfg, "enabled", False))
        self.successor_bc_coef = float(_cfg_get(bc_cfg, "coef", 1.0))
        if self.successor_bc_coef < 0.0 or not math.isfinite(self.successor_bc_coef):
            raise ValueError("successor BC coefficient must be finite and non-negative")
        install_successor_algorithm_registry()
        super().__init__(cfg)

    def init_worker(self) -> None:
        super().init_worker()
        # RLinf v0.3 records trainable parameter and persistent-buffer names
        # before FSDP wrapping, then supplies this exact list to weight_syncer.
        require_successor_sync_state_names(self.param_names_need_sync)

    async def recv_rollout_trajectories(self, input_channel) -> None:
        send_num = (
            self._component_placement.get_world_size("env")
            * self.stage_num
            * int(self.cfg.env.train.rollout_epoch)
        )
        recv_num = self._component_placement.get_world_size("actor")
        recv_list = []
        for _ in range(compute_split_num(send_num, recv_num)):
            payload = await input_channel.get(async_op=True).async_wait()
            if not isinstance(payload, SuccessorReplayBatch):
                raise TypeError("successor actor requires SuccessorReplayBatch")
            recv_list.append(payload)
        self.rollout_batch = SuccessorReplayBatch.merge(recv_list).to_actor_batch()

    def compute_advantages_and_returns(self) -> dict[str, torch.Tensor]:
        fn = get_adv_and_returns(self.cfg.algorithm.adv_type)
        advantages, returns = fn(
            rewards=self.rollout_batch["rewards"],
            values=self.rollout_batch["old_values"],
            next_values=self.rollout_batch["next_values"],
            durations=self.rollout_batch["durations"],
            terminated=self.rollout_batch["terminated"],
            truncated=self.rollout_batch["truncated"],
            gamma=self.cfg.algorithm.get("gamma", 1.0),
            gae_lambda=self.cfg.algorithm.get("gae_lambda", 1.0),
            loss_mask=self.rollout_batch["loss_mask"],
            normalize_advantages=self.cfg.algorithm.get("normalize_advantages", False),
        )
        self.rollout_batch["advantages"] = advantages
        self.rollout_batch["returns"] = returns
        mask = self.rollout_batch["loss_mask"]
        return {
            "rollout/reward_mean": _masked_mean(self.rollout_batch["rewards"], mask),
            "rollout/duration_mean": _masked_mean(
                self.rollout_batch["durations"].float(), mask
            ),
        }

    def train_micro_batch(self, micro_batch, metrics, *, is_last: bool) -> None:
        micro_batch = put_tensor_device(micro_batch, self.device)
        backward_ctx = self.before_micro_batch(self.model, is_last_micro_batch=is_last)
        with self.amp_context:
            output = self.model(
                forward_inputs=micro_batch["forward_context"],
                route_actions=micro_batch["route_actions"],
                waypoint_targets=micro_batch["waypoint_targets"],
                expert_waypoint_targets=(
                    micro_batch["expert_waypoint_targets"]
                    if self.successor_bc_enabled
                    else None
                ),
                compute_logprobs=True,
                compute_values=True,
                use_cache=False,
            )
        required = {
            "route_logprobs",
            "waypoint_logprobs",
            "values",
            "waypoint_predictions",
        }
        if self.successor_bc_enabled:
            required.update(
                {
                    "raw_route_logits",
                    "expert_route_candidate_mask",
                    "expert_waypoint_logprobs",
                }
            )
        missing = required - set(output)
        if missing:
            raise RuntimeError(
                "successor model output is missing: " + ", ".join(sorted(missing))
            )
        loss_fn = get_policy_loss(self.cfg.algorithm.loss_type)
        loss, metrics_data = loss_fn(
            route_logprobs=output["route_logprobs"],
            waypoint_logprobs=output["waypoint_logprobs"],
            old_logprobs=micro_batch["prev_logprobs"],
            advantages=micro_batch["advantages"],
            values=output["values"],
            old_values=micro_batch["old_values"],
            returns=micro_batch["returns"],
            waypoint_predictions=output["waypoint_predictions"],
            waypoint_targets=micro_batch["waypoint_targets"],
            waypoint_mask=micro_batch["waypoint_mask"],
            decision_mask=micro_batch["decision_mask"],
            valid_padding=micro_batch["valid_padding"],
            clip_ratio_low=self.cfg.algorithm.clip_ratio_low,
            clip_ratio_high=self.cfg.algorithm.clip_ratio_high,
            value_clip=self.cfg.algorithm.get("value_clip", 0.2),
            waypoint_coef=self.cfg.algorithm.get("waypoint_coef", 1.0),
            value_coef=self.cfg.algorithm.get("value_coef", 0.5),
        )
        if self.successor_bc_enabled:
            target_mask = micro_batch["expert_target_mask"].bool()
            if not torch.equal(target_mask, micro_batch["loss_mask"].bool()):
                raise RuntimeError(
                    "successor BC requires an explicit expert target for every decision row"
                )
            if "expert_waypoint_logprobs" not in output:
                raise RuntimeError("successor model omitted expert waypoint likelihoods")
            expert_routes = micro_batch["expert_route_targets"].long()
            expert_waypoint_mask = micro_batch["expert_waypoint_mask"].bool()
            expert_allowed = output["expert_route_candidate_mask"].gather(
                -1, expert_routes.unsqueeze(-1)
            ).squeeze(-1)
            if not bool(expert_allowed.all()):
                raise RuntimeError("expert route target is excluded by its candidate mask")
            expected_waypoint_mask = expert_routes == 1
            if not torch.equal(expert_waypoint_mask, expected_waypoint_mask):
                raise RuntimeError("expert waypoint mask disagrees with expert route")
            bc_route_logits = output["raw_route_logits"].masked_fill(
                ~output["expert_route_candidate_mask"], -torch.inf
            )
            route_loss = F.cross_entropy(
                bc_route_logits.reshape(-1, 3), expert_routes.reshape(-1)
            )
            waypoint_nll, waypoint_regression = successor_bc_waypoint_terms(
                expert_waypoint_logprobs=output["expert_waypoint_logprobs"],
                waypoint_predictions=output["waypoint_predictions"],
                expert_waypoint_targets=micro_batch["expert_waypoint_targets"],
                expert_waypoint_mask=expert_waypoint_mask,
            )
            bc_loss = route_loss + waypoint_nll + waypoint_regression
            loss = loss + self.successor_bc_coef * bc_loss
            metrics_data.update(
                {
                    "actor/bc_route_loss": route_loss.detach(),
                    "actor/bc_waypoint_nll": waypoint_nll.detach(),
                    "actor/bc_waypoint_regression": waypoint_regression.detach(),
                }
            )
        loss = loss / self.gradient_accumulation
        with backward_ctx:
            self.grad_scaler.scale(loss).backward()
        metrics_data["actor/total_loss"] = loss.detach()
        append_to_dict(metrics, metrics_data)
