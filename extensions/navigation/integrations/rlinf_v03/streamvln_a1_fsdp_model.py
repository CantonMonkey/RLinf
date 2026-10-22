"""RLinf/FSDP root module for the A1 contextual topology-waypoint head.

The original StreamVLN backbone deliberately stays outside this module.  The
FSDP root contains only the trainable 3604-wide projection, topology encoder,
hybrid actor, and value head.  A policy adapter remains responsible for
episode/topology state and points ``root_forward`` at ``manager.model``.

This file follows the established G5/G8 integration boundary: an
``FSDPModelManager`` subclass implements only ``model_provider_func``.  It does
not initialize distributed state, load Habitat, or construct the frozen 8B
feature provider.
"""

from __future__ import annotations

from collections import Counter
import hashlib
from typing import Mapping

import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal, kl_divergence

try:
    from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager

    RLINF_FSDP_AVAILABLE = True
    _RLINF_IMPORT_ERROR: Exception | None = None
except (ImportError, ModuleNotFoundError) as error:  # Local pure-Torch tests.
    FSDPModelManager = object  # type: ignore[assignment,misc]
    RLINF_FSDP_AVAILABLE = False
    _RLINF_IMPORT_ERROR = error

try:
    from .streamvln_contextual_topology_waypoint import (
        CONTEXTUAL_FEATURE_DIM,
        CONTEXTUAL_FEATURE_SCHEMA_SHA256,
        StreamVLNContextualTopologyWaypointPolicy,
        StreamVLNContextualTopologyWaypointPolicyAdapter,
    )
    from .topology_waypoint_stack import EpisodeTopologyState
except ImportError:  # Direct staged-file mode on the experiment host.
    from streamvln_contextual_topology_waypoint import (
        CONTEXTUAL_FEATURE_DIM,
        CONTEXTUAL_FEATURE_SCHEMA_SHA256,
        StreamVLNContextualTopologyWaypointPolicy,
        StreamVLNContextualTopologyWaypointPolicyAdapter,
    )
    from topology_waypoint_stack import EpisodeTopologyState


A1_MODEL_KIND = "streamvln_a1_contextual_topology_waypoint_head"
A1_OPERATION_NAMES = ("sample", "replay", "bootstrap", "bc_loss", "distribution")
A1_TRAINABLE_GROUPS = (
    "observation_encoder.projection",
    "topology_encoder",
    "context_fusion",
    "candidate_scorer",
    "candidate_slot_bias",
    "residual_mean_head",
    "residual_log_std",
    "value_head",
)

_MISSING = object()


def _cfg(cfg: object, name: str, default: object = _MISSING) -> object:
    if isinstance(cfg, Mapping) and name in cfg:
        return cfg[name]
    if hasattr(cfg, name):
        return getattr(cfg, name)
    if default is not _MISSING:
        return default
    raise KeyError(f"A1 model config is missing {name}")


def _valid_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _state_schema(state: Mapping[str, object]) -> dict[str, dict[str, object]]:
    schema = {}
    for name, value in sorted(state.items()):
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise TypeError("A1 head state must map string names to tensors")
        schema[name] = {
            "shape": tuple(int(item) for item in value.shape),
            "dtype": str(value.dtype),
        }
    return schema


def _group_for_parameter(name: str) -> str:
    matches = tuple(
        group
        for group in A1_TRAINABLE_GROUPS
        if name == group or name.startswith(f"{group}.")
    )
    if len(matches) != 1:
        raise RuntimeError(f"A1 trainable parameter has no unique group: {name}")
    return matches[0]


class StreamVLNA1FSDPModel(nn.Module):
    """Small FSDP root exposing exactly the four A1 training operations."""

    def __init__(self, policy: StreamVLNContextualTopologyWaypointPolicy) -> None:
        super().__init__()
        if not isinstance(policy, StreamVLNContextualTopologyWaypointPolicy):
            raise TypeError("A1 FSDP root requires the contextual hybrid policy")
        self.policy = policy
        self._operation_counts: Counter[str] = Counter()
        self._validate_trainable_scope()

    def _validate_trainable_scope(self) -> None:
        names = tuple(
            name
            for name, parameter in self.policy.named_parameters()
            if parameter.requires_grad
        )
        if not names:
            raise RuntimeError("A1 FSDP root found no trainable head parameters")
        observed_groups = {_group_for_parameter(name) for name in names}
        if observed_groups != set(A1_TRAINABLE_GROUPS):
            raise RuntimeError("A1 trainable parameter groups drifted")
        if any(not parameter.requires_grad for parameter in self.policy.parameters()):
            raise RuntimeError("A1 FSDP root unexpectedly contains frozen parameters")

    @staticmethod
    def _batch_one(operation: str, kwargs: Mapping[str, object]) -> None:
        forward_inputs = kwargs.get("forward_inputs")
        if not isinstance(forward_inputs, Mapping):
            raise ValueError(f"A1 {operation} requires forward_inputs")
        features = forward_inputs.get("streamvln_features")
        if (
            not isinstance(features, torch.Tensor)
            or features.shape != (1, CONTEXTUAL_FEATURE_DIM)
        ):
            raise ValueError(f"A1 {operation} requires Feature-v2 batch size one")
        if operation == "replay":
            candidate = kwargs.get("candidate_action")
            raw = kwargs.get("raw_action")
            if not isinstance(candidate, torch.Tensor) or candidate.shape != (1,):
                raise ValueError("A1 replay candidate_action must have shape [1]")
            if not isinstance(raw, torch.Tensor) or raw.shape != (1, 2):
                raise ValueError("A1 replay raw_action must have shape [1,2]")
        elif operation == "bc_loss":
            candidate = kwargs.get("target_candidate")
            waypoint = kwargs.get("target_waypoint")
            if not isinstance(candidate, torch.Tensor) or candidate.shape != (1,):
                raise ValueError("A1 BC target_candidate must have shape [1]")
            if not isinstance(waypoint, torch.Tensor) or waypoint.shape != (1, 2):
                raise ValueError("A1 BC target_waypoint must have shape [1,2]")

    def forward(self, *, operation: str, **kwargs):
        if operation not in A1_OPERATION_NAMES:
            raise ValueError(f"unsupported A1 FSDP operation {operation!r}")
        self._batch_one(operation, kwargs)
        if operation == "distribution":
            logits, means, log_std, _, continuous_mask, _ = self.policy._distribution(
                kwargs["forward_inputs"]
            )
            result = {
                "route_logits": logits,
                "residual_mean": means,
                "residual_log_std": log_std,
                "continuous_mask": continuous_mask,
            }
        else:
            result = self.policy(operation=operation, **kwargs)
        self._operation_counts[operation] += 1
        return result

    def operation_audit(self) -> dict[str, object]:
        counts = {
            operation: int(self._operation_counts.get(operation, 0))
            for operation in A1_OPERATION_NAMES
        }
        return {
            "path": "manager.model",
            "operations": counts,
            "manager_model_forward_count": sum(counts.values()),
            "local_policy_callback_forward_count": 0,
            "batch_size": 1,
        }

    def reset_operation_audit(self) -> None:
        self._operation_counts.clear()

    def trainable_manifest(self) -> dict[str, object]:
        names = tuple(
            sorted(
                name
                for name, parameter in self.policy.named_parameters()
                if parameter.requires_grad
            )
        )
        return {
            "model_kind": A1_MODEL_KIND,
            "external_backbone": True,
            "backbone_in_fsdp_root": False,
            "feature_dim": CONTEXTUAL_FEATURE_DIM,
            "feature_schema_sha256": CONTEXTUAL_FEATURE_SCHEMA_SHA256,
            "release_manifest_sha256": self.policy.release_manifest_sha256,
            "action_distribution": (
                "masked_categorical_plus_conditional_gaussian_residual"
            ),
            "batch_size": 1,
            "trainable_groups": list(A1_TRAINABLE_GROUPS),
            "trainable_names": list(names),
            "trainable_parameter_count": int(
                sum(
                    parameter.numel()
                    for parameter in self.policy.parameters()
                    if parameter.requires_grad
                )
            ),
        }

    def head_state_dict(self) -> dict[str, torch.Tensor]:
        """Return an isolated CPU state containing no external backbone data."""

        return {
            name: value.detach().cpu().clone()
            for name, value in self.policy.state_dict().items()
        }

    def head_state_schema(self) -> dict[str, dict[str, object]]:
        return _state_schema(self.policy.state_dict())

    def load_head_state_dict(self, state: Mapping[str, object]) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("A1 head checkpoint state must be a mapping")
        expected = self.head_state_schema()
        if _state_schema(state) != expected:
            raise RuntimeError("A1 head checkpoint state schema differs")
        self.policy.load_state_dict(dict(state), strict=True)

    def changed_trainable_groups(
        self,
        before: Mapping[str, object],
    ) -> tuple[str, ...]:
        """Audit exact head changes using the manifest's public group names."""

        after = self.head_state_dict()
        if _state_schema(before) != _state_schema(after):
            raise RuntimeError("A1 before/after head state schemas differ")
        changed = set()
        for name, current in after.items():
            previous = before[name]
            if not isinstance(previous, torch.Tensor):
                raise TypeError("A1 before-state entry is not a tensor")
            if not torch.equal(previous.detach().cpu(), current):
                changed.add(_group_for_parameter(name))
        return tuple(group for group in A1_TRAINABLE_GROUPS if group in changed)


def get_model(cfg: object, torch_dtype=None) -> StreamVLNA1FSDPModel:
    """RLinf model-provider factory; never loads StreamVLN or Habitat."""

    if torch_dtype not in (None, torch.float32):
        raise ValueError("A1 contextual head currently requires FP32 parameters")
    feature_dim = int(_cfg(cfg, "feature_dim", CONTEXTUAL_FEATURE_DIM))
    schema = str(
        _cfg(cfg, "feature_schema_sha256", CONTEXTUAL_FEATURE_SCHEMA_SHA256)
    )
    if feature_dim != CONTEXTUAL_FEATURE_DIM or schema != CONTEXTUAL_FEATURE_SCHEMA_SHA256:
        raise ValueError("A1 model config differs from Feature-v2")
    policy = StreamVLNContextualTopologyWaypointPolicy(
        release_manifest_sha256=str(_cfg(cfg, "release_manifest_sha256")),
        hidden_dim=int(_cfg(cfg, "hidden_dim", 128)),
        topology_max_nodes=int(_cfg(cfg, "topology_max_nodes", 161)),
        topology_model_dim=int(_cfg(cfg, "topology_model_dim", 32)),
        max_candidates=int(_cfg(cfg, "max_candidates", 5)),
        residual_limit_m=float(_cfg(cfg, "residual_limit_m", 0.2)),
        initial_log_std=float(_cfg(cfg, "initial_log_std", -0.5)),
        feature_dim=feature_dim,
    )
    return StreamVLNA1FSDPModel(policy)


def build_frozen_reference_model(
    cfg: object,
    head_state: Mapping[str, object],
) -> StreamVLNA1FSDPModel:
    """Create the small frozen BC/KL anchor without an 8B backbone copy."""

    reference = get_model(cfg)
    reference.load_head_state_dict(head_state)
    reference.requires_grad_(False)
    reference.eval()
    return reference


def hybrid_reference_kl(
    reference: Mapping[str, object],
    current: Mapping[str, object],
) -> torch.Tensor:
    """Exact KL for categorical route plus conditional diagonal residual.

    The reference tensors must be detached outputs from a frozen BC head.  The
    current tensors stay differentiable and normally come from
    ``manager.model(operation='distribution', ...)``.
    """

    keys = {
        "route_logits",
        "residual_mean",
        "residual_log_std",
        "continuous_mask",
    }
    if not isinstance(reference, Mapping) or set(reference) != keys:
        raise ValueError("A1 reference distribution schema drifted")
    if not isinstance(current, Mapping) or set(current) != keys:
        raise ValueError("A1 current distribution schema drifted")
    ref = {name: reference[name] for name in keys}
    cur = {name: current[name] for name in keys}
    if any(not isinstance(value, torch.Tensor) for value in (*ref.values(), *cur.values())):
        raise TypeError("A1 hybrid KL fields must be tensors")
    if any(value.requires_grad for value in ref.values()):
        raise ValueError("A1 BC reference distribution must be detached")
    expected = {
        "route_logits": (1, 5),
        "residual_mean": (1, 5, 2),
        "residual_log_std": (1, 5, 2),
        "continuous_mask": (1, 5),
    }
    for name, shape in expected.items():
        if ref[name].shape != shape or cur[name].shape != shape:
            raise ValueError(f"A1 hybrid KL {name} shape drifted")
        if ref[name].device != cur[name].device:
            raise ValueError(f"A1 hybrid KL {name} devices differ")
    if ref["continuous_mask"].dtype != torch.bool or cur[
        "continuous_mask"
    ].dtype != torch.bool:
        raise TypeError("A1 hybrid KL continuous masks must be boolean")
    for side in (ref, cur):
        logits = side["route_logits"]
        if not logits.dtype.is_floating_point:
            raise TypeError("A1 hybrid KL route logits must be floating point")
        if bool(torch.isnan(logits).any()) or bool(torch.isposinf(logits).any()):
            raise ValueError("A1 hybrid KL route logits contain invalid values")
    ref_mask = torch.isfinite(ref["route_logits"])
    cur_mask = torch.isfinite(cur["route_logits"])
    if not torch.equal(ref_mask.detach().cpu(), cur_mask.detach().cpu()):
        raise ValueError("A1 hybrid KL candidate masks differ")
    if not bool(ref_mask.any()) or not bool(cur_mask.any()):
        raise ValueError("A1 hybrid KL has no valid candidate")
    if not torch.equal(
        ref["continuous_mask"].detach().cpu(),
        cur["continuous_mask"].detach().cpu(),
    ):
        raise ValueError("A1 hybrid KL continuous masks differ")
    for name in ("residual_mean", "residual_log_std"):
        if not bool(torch.isfinite(ref[name]).all()) or not bool(
            torch.isfinite(cur[name]).all()
        ):
            raise ValueError(f"A1 hybrid KL {name} must be finite")

    reference_route = Categorical(logits=ref["route_logits"])
    current_route = Categorical(logits=cur["route_logits"])
    route_kl = kl_divergence(reference_route, current_route)
    residual_kl = kl_divergence(
        Normal(ref["residual_mean"], ref["residual_log_std"].exp()),
        Normal(cur["residual_mean"], cur["residual_log_std"].exp()),
    ).sum(dim=-1)
    conditional = (
        reference_route.probs
        * ref["continuous_mask"].to(dtype=reference_route.probs.dtype)
        * residual_kl
    ).sum(dim=-1)
    result = route_kl + conditional
    if result.shape != (1,) or not bool(torch.isfinite(result).all()):
        raise RuntimeError("A1 hybrid reference KL is non-finite")
    return result


def model_config_from_manifest(
    manifest: Mapping[str, object],
    *,
    release_manifest_sha256: str,
) -> dict[str, object]:
    if not isinstance(manifest, Mapping):
        raise TypeError("A1 manifest must be a mapping")
    if not _valid_sha256(release_manifest_sha256):
        raise ValueError("released StreamVLN manifest SHA-256 is invalid")
    architecture = manifest.get("architecture")
    frozen_prior = manifest.get("frozen_prior")
    if not isinstance(architecture, Mapping) or not isinstance(frozen_prior, Mapping):
        raise ValueError("A1 manifest lacks architecture or frozen prior")
    config = {
        "model_type": A1_MODEL_KIND,
        "release_manifest_sha256": release_manifest_sha256,
        "feature_dim": frozen_prior.get("feature_dim"),
        "feature_schema_sha256": frozen_prior.get("schema_sha256"),
        "hidden_dim": architecture.get("hidden_dim"),
        "topology_max_nodes": architecture.get("topology_max_nodes"),
        "topology_model_dim": architecture.get("topology_model_dim"),
        "max_candidates": architecture.get("max_candidates"),
        "residual_limit_m": architecture.get("residual_limit_m"),
        "initial_log_std": architecture.get("initial_log_std"),
    }
    # Construction is the final strict type/range validation.
    get_model(config)
    return config


def manager_config_from_manifest(
    manifest: Mapping[str, object],
    *,
    release_manifest_sha256: str,
    total_training_steps: int,
) -> object:
    """Build the same one-rank/no-shard RLinf config shape used by G5/G8."""

    if type(total_training_steps) is not int or total_training_steps < 1:
        raise ValueError("total_training_steps must be a positive integer")
    budget = manifest.get("budget") if isinstance(manifest, Mapping) else None
    if not isinstance(budget, Mapping):
        raise ValueError("A1 manifest lacks its training budget")
    learning_rate = float(budget.get("bc_learning_rate"))
    weight_decay = float(budget.get("bc_weight_decay"))
    raw = {
        "model": model_config_from_manifest(
            manifest, release_manifest_sha256=release_manifest_sha256
        ),
        "optim": {
            "lr": learning_rate,
            "value_lr": learning_rate,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_eps": 1e-8,
            "weight_decay": weight_decay,
            "clip_grad": 1.0,
            "total_training_steps": total_training_steps,
            "lr_warmup_steps": 0,
            "lr_scheduler": "constant",
        },
        "fsdp_config": {
            "strategy": "fsdp",
            "sharding_strategy": "no_shard",
            "gradient_checkpointing": False,
            "cpu_offload": False,
            "offload_pin_memory": False,
            "reshard_after_forward": False,
            "enable_gradient_accumulation": False,
            "forward_prefetch": False,
            "limit_all_gathers": False,
            "backward_prefetch": None,
            "use_orig_params": True,
            "use_liger_kernel": False,
            "wrap_policy": {"disable": True},
            "mixed_precision": {
                "param_dtype": None,
                "reduce_dtype": None,
                "buffer_dtype": None,
            },
            "amp_autocast": {"enabled": False, "precision": "bf16"},
            "grad_scaler": {"enabled": False},
        },
    }
    try:
        from omegaconf import OmegaConf
    except (ImportError, ModuleNotFoundError):
        return raw
    return OmegaConf.create(raw)


if RLINF_FSDP_AVAILABLE:

    class StreamVLNA1FSDPModelManager(FSDPModelManager):
        """RLinf manager whose FSDP root is the small A1 policy only."""

        def model_provider_func(self):
            return get_model(self._cfg.model)

else:

    class StreamVLNA1FSDPModelManager:  # pragma: no cover - exercised remotely.
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            raise RuntimeError(
                "RLinf FSDPModelManager is unavailable in this Python environment"
            ) from _RLINF_IMPORT_ERROR


def create_fsdp_manager(
    cfg: object,
    *,
    world_size: int = 1,
    rank: int = 0,
):
    """Instantiate, but do not initialize, the RLinf manager.

    The live provider owns distributed initialization and calls
    ``setup_model_and_optimizer()`` exactly as G5/G8 do.
    """

    if not RLINF_FSDP_AVAILABLE:
        raise RuntimeError(
            "RLinf FSDPModelManager is unavailable in this Python environment"
        ) from _RLINF_IMPORT_ERROR
    if type(world_size) is not int or world_size != 1 or type(rank) is not int or rank != 0:
        raise ValueError("A1 pilot currently requires one rank: world_size=1, rank=0")
    return StreamVLNA1FSDPModelManager(cfg, world_size=world_size, rank=rank)


def unwrap_a1_model(root_model: object) -> StreamVLNA1FSDPModel:
    """Return the A1 core from either a direct root or an FSDP wrapper."""

    core = root_model
    if not isinstance(core, StreamVLNA1FSDPModel):
        core = getattr(root_model, "module", None)
    if not isinstance(core, StreamVLNA1FSDPModel):
        raise TypeError("manager.model does not wrap StreamVLNA1FSDPModel")
    return core


def root_forward_audit(root_model: object) -> dict[str, object]:
    """Return the exact evidence shape consumed by the A1 runner."""

    core = unwrap_a1_model(root_model)
    audit = core.operation_audit()
    audit["fsdp_wrapped"] = (
        root_model is not core
        and type(root_model).__name__ == "FullyShardedDataParallel"
    )
    return audit


def build_policy_adapter(
    root_model: object,
    topology_state: EpisodeTopologyState,
    *,
    mode: str = "train",
    deterministic: bool | None = None,
    policy_version: int = 0,
) -> StreamVLNContextualTopologyWaypointPolicyAdapter:
    """Bind episode state to the real root-forward callable."""

    if not callable(root_model):
        raise TypeError("A1 root model must be callable")
    core = unwrap_a1_model(root_model)
    return StreamVLNContextualTopologyWaypointPolicyAdapter(
        core.policy,
        topology_state,
        mode=mode,
        deterministic=deterministic,
        policy_version=policy_version,
        root_forward=root_model,
    )


__all__ = [
    "A1_MODEL_KIND",
    "A1_OPERATION_NAMES",
    "A1_TRAINABLE_GROUPS",
    "RLINF_FSDP_AVAILABLE",
    "StreamVLNA1FSDPModel",
    "StreamVLNA1FSDPModelManager",
    "build_frozen_reference_model",
    "build_policy_adapter",
    "create_fsdp_manager",
    "get_model",
    "hybrid_reference_kl",
    "manager_config_from_manifest",
    "model_config_from_manifest",
    "root_forward_audit",
    "unwrap_a1_model",
]
