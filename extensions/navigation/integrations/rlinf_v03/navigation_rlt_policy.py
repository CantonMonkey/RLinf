"""Faithful two-stage Navigation-RLT policy over a frozen LightNav prefix.

Stage 1 learns one 256-D ``z_rl`` token from LightNav's final-layer contextual
prefix hidden states and causal topology tokens.  Randomly masked valid prefix
and topology positions are reconstructed with masked MSE.  LightNav remains an
external frozen producer: this module rejects gradient-bearing context inputs.

Stage 2 requires Stage 1 to be frozen.  Its actor and twin Q critics consume
``z_rl`` plus a frozen reference embedding.  A tensor-only compiler maps the
requested KEEP, ADJUST, STOP, or RECOVER action to the exact canonical action
seen by the critics.  A persistent behavior-support mask intersects the
compiler-valid gates for acting and target bootstrapping.  Training uses a
minimum target Q, finite four-candidate CQL, and a Native-reference actor anchor
with reference dropout.  Stage 2 replaces the frozen Stage-1 actor bias with a
support-aware behavior prior before optimization.  The regularized objective
adds fixed-prior KL(behavior || actor) on legal supported gates.  There is no
SAC entropy target.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


SCHEMA_NAME = "navigation_rlt_lightnav"
SCHEMA_VERSION = 3
FEATURE_SCHEMA = "lightnav_contextual_prefix_causal_topology_v2"
TOPOLOGY_PACKING_SCHEMA = "causal_nodes_then_global_zero_pad_v1"
PREFIX_SELECTION_SCHEMA = "keep_endpoints_uniform_middle_v1"
CANONICAL_ACTION_SCHEMA = "mode_one_hot4_plus_resolved_se2_v1"
Z_RL_DIM = 256
GATE_NAMES = ("KEEP", "ADJUST", "STOP", "RECOVER")
KEEP_GATE = 0
ADJUST_GATE = 1
STOP_GATE = 2
RECOVER_GATE = 3
GATE_COUNT = len(GATE_NAMES)
RESIDUAL_AXES = ("forward_m", "yaw_rad")
CANONICAL_ACTION_DIM = len(GATE_NAMES) + 3
STAGE2_BEHAVIOR_PRIOR_UNIFORM_MIX = 0.05
EXECUTOR_FORWARD_RANGE_M = (0.0, 0.25)
EXECUTOR_YAW_RANGE_RAD = (-math.radians(30.0), math.radians(30.0))


class NavigationRLTContractError(ValueError):
    """Raised when an input, action, training phase, or checkpoint is invalid."""


def _positive_int(value: int, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise NavigationRLTContractError(f"{name} must be positive")
    return value


def _finite_scalar(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise NavigationRLTContractError(f"{name} must be finite")
    return value


def _require_finite(value: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise NavigationRLTContractError(f"{name} contains non-finite values")


def _require_integer_tensor(value: torch.Tensor, name: str) -> None:
    integer_dtypes = {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
    if value.dtype not in integer_dtypes:
        raise NavigationRLTContractError(f"{name} must have an integer dtype")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class NavigationRLTConfig:
    """Static feature dimensions and executable action bounds."""

    prefix_hidden_dim: int
    topology_dim: int
    reference_dim: int
    canonical_action_dim: int = CANONICAL_ACTION_DIM
    topology_node_dim: int = 8
    topology_global_dim: int = 14
    topology_packing_schema: str = TOPOLOGY_PACKING_SCHEMA
    attention_heads: int = 8
    transformer_layers: int = 2
    feedforward_dim: int = 768
    critic_hidden_dim: int = 256
    dropout: float = 0.0
    max_prefix_tokens: int = 192
    max_artifact_prefix_tokens: int = 4096
    prefix_selection_schema: str = PREFIX_SELECTION_SCHEMA
    max_topology_tokens: int = 64
    residual_forward_limit_m: float = 0.05
    residual_yaw_limit_rad: float = math.radians(10.0)
    initial_log_std: float = -1.5
    min_log_std: float = -5.0
    max_log_std: float = 1.0
    strict_frozen_inputs: bool = True

    def __post_init__(self) -> None:
        for name in (
            "prefix_hidden_dim",
            "topology_dim",
            "reference_dim",
            "canonical_action_dim",
            "topology_node_dim",
            "topology_global_dim",
            "attention_heads",
            "transformer_layers",
            "feedforward_dim",
            "critic_hidden_dim",
            "max_prefix_tokens",
            "max_artifact_prefix_tokens",
            "max_topology_tokens",
        ):
            _positive_int(getattr(self, name), name)
        if self.canonical_action_dim != CANONICAL_ACTION_DIM:
            raise NavigationRLTContractError(
                f"canonical_action_dim must be {CANONICAL_ACTION_DIM}: "
                "four mode one-hot values plus resolved SE(2)"
            )
        if self.prefix_selection_schema != PREFIX_SELECTION_SCHEMA:
            raise NavigationRLTContractError("prefix selection schema differs")
        if self.max_prefix_tokens < 2:
            raise NavigationRLTContractError(
                "max_prefix_tokens must preserve both sequence endpoints"
            )
        if self.max_prefix_tokens > self.max_artifact_prefix_tokens:
            raise NavigationRLTContractError(
                "policy prefix limit cannot exceed the LightNav artifact limit"
            )
        if self.topology_packing_schema != TOPOLOGY_PACKING_SCHEMA:
            raise NavigationRLTContractError("topology packing schema differs")
        if self.topology_node_dim > self.topology_dim:
            raise NavigationRLTContractError(
                "topology_node_dim cannot exceed topology_dim"
            )
        if self.topology_global_dim > self.topology_dim:
            raise NavigationRLTContractError(
                "topology_global_dim cannot exceed topology_dim"
            )
        if self.max_topology_tokens < 2:
            raise NavigationRLTContractError(
                "max_topology_tokens must reserve a node and one global token"
            )
        if Z_RL_DIM % self.attention_heads != 0:
            raise NavigationRLTContractError(
                "256-D z_rl must be divisible by attention_heads"
            )
        dropout = _finite_scalar(self.dropout, "dropout")
        if not 0.0 <= dropout < 1.0:
            raise NavigationRLTContractError("dropout must be in [0, 1)")
        for name in ("residual_forward_limit_m", "residual_yaw_limit_rad"):
            if _finite_scalar(getattr(self, name), name) <= 0.0:
                raise NavigationRLTContractError(f"{name} must be positive")
        minimum = _finite_scalar(self.min_log_std, "min_log_std")
        maximum = _finite_scalar(self.max_log_std, "max_log_std")
        initial = _finite_scalar(self.initial_log_std, "initial_log_std")
        if not minimum <= initial <= maximum:
            raise NavigationRLTContractError(
                "initial_log_std must lie inside [min_log_std, max_log_std]"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NavigationRLTPrefixSelection:
    """Explicit deterministic reduction of a LightNav prefix artifact.

    ``selected_source_indices`` refers to positions in the source artifact and
    is ``-1`` in right-padded output slots.  The two length tensors are kept so
    rollout/replay manifests can record whether selection occurred.
    """

    contextual_prefix_hidden: torch.Tensor
    prefix_mask: torch.Tensor
    selected_source_indices: torch.Tensor
    original_valid_lengths: torch.Tensor
    selected_valid_lengths: torch.Tensor
    source_sequence_width: int
    selection_schema: str = PREFIX_SELECTION_SCHEMA


def select_prefix_tokens(
    contextual_prefix_hidden: torch.Tensor,
    prefix_mask: torch.Tensor,
    *,
    config: NavigationRLTConfig,
) -> NavigationRLTPrefixSelection:
    """Select at most ``max_prefix_tokens`` with reproducible uniform coverage.

    Valid tokens remain in source order.  If selection is necessary, output
    slot ``j`` uses valid-token ordinal
    ``floor(j * (L - 1) / (K - 1))`` for original valid length ``L`` and policy
    limit ``K``.  This preserves the first and last valid tokens and uniformly
    covers the valid middle.  The result is always right-padded to ``K``.
    """

    if not isinstance(config, NavigationRLTConfig):
        raise NavigationRLTContractError("config must be NavigationRLTConfig")
    if (
        not isinstance(contextual_prefix_hidden, torch.Tensor)
        or contextual_prefix_hidden.ndim != 3
    ):
        raise NavigationRLTContractError(
            "contextual_prefix_hidden must have shape [B, S, Dp]"
        )
    if (
        not contextual_prefix_hidden.is_floating_point()
        or contextual_prefix_hidden.requires_grad
    ):
        raise NavigationRLTContractError(
            "contextual_prefix_hidden must be a detached floating tensor"
        )
    batch, source_width, hidden_dim = contextual_prefix_hidden.shape
    if batch < 1 or source_width < 1 or hidden_dim != config.prefix_hidden_dim:
        raise NavigationRLTContractError(
            "contextual_prefix_hidden dimensions differ from config"
        )
    if source_width > config.max_artifact_prefix_tokens:
        raise NavigationRLTContractError(
            "LightNav prefix artifact exceeds max_artifact_prefix_tokens"
        )
    if not isinstance(prefix_mask, torch.Tensor) or prefix_mask.shape != (
        batch,
        source_width,
    ):
        raise NavigationRLTContractError(
            "prefix_mask shape differs from contextual_prefix_hidden"
        )
    if prefix_mask.dtype != torch.bool or prefix_mask.requires_grad:
        raise NavigationRLTContractError("prefix_mask must be detached torch.bool")
    if prefix_mask.device != contextual_prefix_hidden.device:
        raise NavigationRLTContractError(
            "contextual_prefix_hidden and prefix_mask must share one device"
        )
    _require_finite(contextual_prefix_hidden, "contextual_prefix_hidden")
    prefix_mask = prefix_mask.detach()
    original_lengths = prefix_mask.sum(dim=1, dtype=torch.int64)
    if bool((original_lengths < 1).any()):
        raise NavigationRLTContractError("prefix_mask contains an empty row")
    if bool(
        (
            contextual_prefix_hidden.masked_select(~prefix_mask.unsqueeze(-1))
            != 0
        ).any()
    ):
        raise NavigationRLTContractError(
            "masked LightNav prefix positions must be exactly zero"
        )

    policy_limit = config.max_prefix_tokens
    selected_hidden = contextual_prefix_hidden.new_zeros(
        (batch, policy_limit, config.prefix_hidden_dim)
    )
    selected_mask = torch.zeros(
        (batch, policy_limit),
        dtype=torch.bool,
        device=contextual_prefix_hidden.device,
    )
    selected_indices = torch.full(
        (batch, policy_limit),
        -1,
        dtype=torch.int64,
        device=contextual_prefix_hidden.device,
    )
    selected_lengths = torch.minimum(
        original_lengths,
        torch.full_like(original_lengths, policy_limit),
    )
    for row in range(batch):
        valid_source_indices = torch.nonzero(
            prefix_mask[row], as_tuple=False
        ).flatten()
        original_length = int(original_lengths[row].item())
        selected_length = int(selected_lengths[row].item())
        if original_length <= policy_limit:
            source_indices = valid_source_indices
        else:
            ordinals = (
                torch.arange(
                    policy_limit,
                    device=contextual_prefix_hidden.device,
                    dtype=torch.int64,
                )
                * (original_length - 1)
            ) // (policy_limit - 1)
            source_indices = valid_source_indices.index_select(0, ordinals)
        if source_indices.numel() != selected_length:
            raise NavigationRLTContractError("prefix selection length drifted")
        selected_hidden[row, :selected_length] = contextual_prefix_hidden[
            row
        ].index_select(0, source_indices)
        selected_mask[row, :selected_length] = True
        selected_indices[row, :selected_length] = source_indices

    return NavigationRLTPrefixSelection(
        contextual_prefix_hidden=selected_hidden,
        prefix_mask=selected_mask,
        selected_source_indices=selected_indices,
        original_valid_lengths=original_lengths,
        selected_valid_lengths=selected_lengths,
        source_sequence_width=int(source_width),
    )


@dataclass(frozen=True)
class NavigationRLTTopologyPack:
    """Deterministically packed topology tokens and their independent masks."""

    tokens: torch.Tensor
    token_mask: torch.Tensor
    node_mask: torch.Tensor
    global_feature_mask: torch.Tensor


def pack_topology_tokens(
    node_targets: torch.Tensor,
    node_mask: torch.Tensor,
    global_targets: torch.Tensor,
    global_feature_mask: torch.Tensor,
    *,
    config: NavigationRLTConfig,
) -> NavigationRLTTopologyPack:
    """Pack causal node rows followed by one global row without learned state.

    Node and global features occupy the leading coordinates of
    ``topology_dim`` and are zero-padded on the right.  Token slots are padded
    to ``max_topology_tokens``.  The node mask and per-feature global mask are
    preserved separately from the packed token mask.
    """

    if not isinstance(config, NavigationRLTConfig):
        raise NavigationRLTContractError("config must be NavigationRLTConfig")
    if not isinstance(node_targets, torch.Tensor) or node_targets.ndim != 3:
        raise NavigationRLTContractError("node_targets must have shape [B, N, Dn]")
    if not node_targets.is_floating_point() or node_targets.requires_grad:
        raise NavigationRLTContractError(
            "node_targets must be a detached floating tensor"
        )
    batch, nodes, node_dim = node_targets.shape
    if batch < 1 or nodes < 1 or node_dim != config.topology_node_dim:
        raise NavigationRLTContractError("node_targets dimensions differ from config")
    if nodes + 1 > config.max_topology_tokens:
        raise NavigationRLTContractError(
            "node targets plus the global token exceed max_topology_tokens"
        )
    if not isinstance(node_mask, torch.Tensor) or node_mask.shape != (batch, nodes):
        raise NavigationRLTContractError("node_mask shape differs from node_targets")
    if node_mask.dtype != torch.bool or node_mask.requires_grad:
        raise NavigationRLTContractError("node_mask must be detached torch.bool")
    if not isinstance(global_targets, torch.Tensor) or global_targets.shape != (
        batch,
        config.topology_global_dim,
    ):
        raise NavigationRLTContractError("global_targets dimensions differ from config")
    if not global_targets.is_floating_point() or global_targets.requires_grad:
        raise NavigationRLTContractError(
            "global_targets must be a detached floating tensor"
        )
    if (
        not isinstance(global_feature_mask, torch.Tensor)
        or global_feature_mask.shape != global_targets.shape
    ):
        raise NavigationRLTContractError(
            "global_feature_mask shape differs from global_targets"
        )
    if global_feature_mask.dtype != torch.bool or global_feature_mask.requires_grad:
        raise NavigationRLTContractError(
            "global_feature_mask must be detached torch.bool"
        )
    devices = {
        node_targets.device,
        node_mask.device,
        global_targets.device,
        global_feature_mask.device,
    }
    if len(devices) != 1:
        raise NavigationRLTContractError("topology packing inputs must share one device")
    _require_finite(node_targets, "node_targets")
    _require_finite(global_targets, "global_targets")
    node_mask = node_mask.detach()
    global_feature_mask = global_feature_mask.detach()
    node_lengths = node_mask.sum(dim=1)
    if bool((node_lengths < 1).any()):
        raise NavigationRLTContractError("node_mask contains an empty graph")
    expected_node_mask = (
        torch.arange(nodes, device=node_targets.device).unsqueeze(0)
        < node_lengths.unsqueeze(1)
    )
    if not torch.equal(node_mask, expected_node_mask):
        raise NavigationRLTContractError("node_mask must be a right-padded prefix")
    if bool((~global_feature_mask.any(dim=1)).any()):
        raise NavigationRLTContractError(
            "global_feature_mask contains an empty global token"
        )
    if bool((node_targets.masked_select(~node_mask.unsqueeze(-1)) != 0).any()):
        raise NavigationRLTContractError("padded node targets must be exactly zero")
    if bool((global_targets.masked_select(~global_feature_mask) != 0).any()):
        raise NavigationRLTContractError("masked global targets must be exactly zero")

    tokens = node_targets.new_zeros(
        (batch, config.max_topology_tokens, config.topology_dim)
    )
    tokens[:, :nodes, : config.topology_node_dim] = node_targets.detach()
    tokens[:, nodes, : config.topology_global_dim] = global_targets.detach().to(
        dtype=node_targets.dtype
    )
    token_mask = torch.zeros(
        (batch, config.max_topology_tokens),
        dtype=torch.bool,
        device=node_targets.device,
    )
    token_mask[:, :nodes] = node_mask
    token_mask[:, nodes] = True
    return NavigationRLTTopologyPack(
        tokens=tokens,
        token_mask=token_mask,
        node_mask=node_mask.clone(),
        global_feature_mask=global_feature_mask.clone(),
    )


def canonical_action_from_mode_and_se2(
    mode: torch.Tensor, resolved_se2: torch.Tensor
) -> torch.Tensor:
    """Return the strict 7-D critic action: one-hot mode plus resolved SE(2)."""

    if not isinstance(mode, torch.Tensor) or mode.ndim != 1:
        raise NavigationRLTContractError("mode must have shape [B]")
    _require_integer_tensor(mode, "mode")
    if not isinstance(resolved_se2, torch.Tensor) or resolved_se2.shape != (
        mode.shape[0],
        3,
    ):
        raise NavigationRLTContractError("resolved_se2 must have shape [B, 3]")
    if not resolved_se2.is_floating_point():
        raise NavigationRLTContractError("resolved_se2 must be floating point")
    if mode.device != resolved_se2.device:
        raise NavigationRLTContractError("mode and resolved_se2 must share one device")
    mode = mode.to(dtype=torch.int64)
    if bool(((mode < 0) | (mode >= GATE_COUNT)).any()):
        raise NavigationRLTContractError("mode is outside KEEP/ADJUST/STOP/RECOVER")
    _require_finite(resolved_se2, "resolved_se2")
    one_hot = F.one_hot(mode, num_classes=GATE_COUNT).to(dtype=resolved_se2.dtype)
    result = torch.cat((one_hot, resolved_se2), dim=-1)
    if result.shape[-1] != CANONICAL_ACTION_DIM:
        raise NavigationRLTContractError("canonical action width drifted")
    return result


@dataclass(frozen=True)
class Stage1LossConfig:
    prefix_mask_ratio: float = 0.25
    topology_mask_ratio: float = 0.25
    prefix_weight: float = 1.0
    topology_weight: float = 1.0

    def __post_init__(self) -> None:
        for name in ("prefix_mask_ratio", "topology_mask_ratio"):
            value = _finite_scalar(getattr(self, name), name)
            if not 0.0 < value <= 1.0:
                raise NavigationRLTContractError(f"{name} must be in (0, 1]")
        for name in ("prefix_weight", "topology_weight"):
            if _finite_scalar(getattr(self, name), name) <= 0.0:
                raise NavigationRLTContractError(f"{name} must be positive")


@dataclass(frozen=True)
class Stage2LossConfig:
    reference_gate_weight: float = 0.25
    reference_residual_weight: float = 0.10
    reference_dropout_probability: float = 0.20
    target_tau: float = 0.005
    cql_alpha: float = 1.0
    cql_temperature: float = 1.0
    behavior_kl_weight: float = 0.0
    behavior_gate_prior: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        for name in ("reference_gate_weight", "reference_residual_weight"):
            if _finite_scalar(getattr(self, name), name) < 0.0:
                raise NavigationRLTContractError(f"{name} must be non-negative")
        probability = _finite_scalar(
            self.reference_dropout_probability, "reference_dropout_probability"
        )
        if not 0.0 <= probability <= 1.0:
            raise NavigationRLTContractError(
                "reference_dropout_probability must be in [0, 1]"
            )
        tau = _finite_scalar(self.target_tau, "target_tau")
        if not 0.0 < tau <= 1.0:
            raise NavigationRLTContractError("target_tau must be in (0, 1]")
        if _finite_scalar(self.cql_alpha, "cql_alpha") < 0.0:
            raise NavigationRLTContractError("cql_alpha must be non-negative")
        if _finite_scalar(self.cql_temperature, "cql_temperature") <= 0.0:
            raise NavigationRLTContractError("cql_temperature must be positive")
        if _finite_scalar(self.behavior_kl_weight, "behavior_kl_weight") < 0.0:
            raise NavigationRLTContractError("behavior_kl_weight must be non-negative")
        if self.behavior_gate_prior is not None:
            prior = self.behavior_gate_prior
            if (not isinstance(prior, tuple) or len(prior) != GATE_COUNT
                    or any(not math.isfinite(float(value)) or value < 0.0 for value in prior)
                    or not math.isclose(sum(prior), 1.0, rel_tol=0.0, abs_tol=1.0e-5)):
                raise NavigationRLTContractError("behavior_gate_prior must be a probability tuple[4]")
        if self.behavior_kl_weight > 0.0 and self.behavior_gate_prior is None:
            raise NavigationRLTContractError("positive behavior KL requires a fixed behavior prior")


def supported_behavior_gate_kl(
    gate_logits: torch.Tensor,
    valid_gate_mask: torch.Tensor,
    behavior_prior: torch.Tensor,
) -> torch.Tensor:
    """KL(behavior || actor), renormalized on each row's legal support.

    Compute in FP32 logits so tiny BF16 probabilities retain restoring gradients.
    The prior is training-only; inference and checkpoint tensor schemas are unchanged.
    """
    if gate_logits.ndim != 2 or gate_logits.shape[-1] != GATE_COUNT:
        raise NavigationRLTContractError("behavior KL logits must have shape [batch,4]")
    if valid_gate_mask.shape != gate_logits.shape or valid_gate_mask.dtype != torch.bool:
        raise NavigationRLTContractError("behavior KL valid mask differs")
    if behavior_prior.shape != (GATE_COUNT,) or behavior_prior.requires_grad:
        raise NavigationRLTContractError("behavior KL prior must be detached [4]")
    prior = behavior_prior.detach().to(device=gate_logits.device, dtype=torch.float32)
    valid = valid_gate_mask.detach().to(device=gate_logits.device)
    if not bool(torch.isfinite(prior).all()) or bool((prior < 0.0).any()):
        raise NavigationRLTContractError("behavior KL prior is invalid")
    if bool((valid & (prior.unsqueeze(0) <= 0.0)).any()) or bool((~valid.any(dim=1)).any()):
        raise NavigationRLTContractError("behavior KL legal support has no positive prior mass")
    rows = torch.where(valid, prior.unsqueeze(0), torch.zeros_like(gate_logits, dtype=torch.float32))
    rows = rows / rows.sum(dim=-1, keepdim=True)
    logits = gate_logits.float().masked_fill(~valid, torch.finfo(torch.float32).min)
    log_pi = F.log_softmax(logits, dim=-1)
    safe_log_prior = torch.where(valid, rows, torch.ones_like(rows)).log()
    terms = torch.where(valid, rows * (safe_log_prior - log_pi), torch.zeros_like(rows))
    result = terms.sum(dim=-1)
    _require_finite(result, "behavior_gate_kl")
    return result


@dataclass(frozen=True)
class NavigationRLTContext:
    """Frozen final-layer LightNav prefix and causal topology tokens."""

    contextual_prefix_hidden: torch.Tensor
    prefix_mask: torch.Tensor
    topology_tokens: torch.Tensor
    topology_mask: torch.Tensor


@dataclass(frozen=True)
class NavigationRLTCompilerFields:
    """Tensor-only parent/executor contract used to compile policy actions.

    Every base action is canonical 7-D ``[mode one-hot(4), resolved SE(2)]``.
    KEEP and ADJUST therefore have different one-hot prefixes even when their
    zero-residual SE(2) is equal. ``adjust_residual_basis`` maps physical
    forward/yaw residuals into the same canonical vector. RECOVER is legal only
    where ``recover_parent_valid`` is true and ``recover_parent_index >= 0``.
    """

    valid_gate_mask: torch.Tensor
    reference_action: torch.Tensor
    adjust_base_action: torch.Tensor
    adjust_residual_basis: torch.Tensor
    stop_action: torch.Tensor
    recover_parent_action: torch.Tensor
    recover_parent_valid: torch.Tensor
    recover_parent_index: torch.Tensor


@dataclass(frozen=True)
class NavigationRLTRepresentation:
    z_rl: torch.Tensor
    prefix_reconstruction: torch.Tensor | None = None
    topology_reconstruction: torch.Tensor | None = None


@dataclass(frozen=True)
class NavigationRLTAction:
    gate: torch.Tensor
    raw_residual: torch.Tensor
    bounded_residual: torch.Tensor
    executable_residual: torch.Tensor
    canonical_action: torch.Tensor
    gate_logits: torch.Tensor
    gate_probabilities: torch.Tensor
    log_prob: torch.Tensor
    recover_parent_index: torch.Tensor


@dataclass(frozen=True)
class NavigationRLTTransitionBatch:
    """Replay rows; behavior versions may be mixed inside one batch."""

    context: NavigationRLTContext
    next_context: NavigationRLTContext
    reference_embedding: torch.Tensor
    next_reference_embedding: torch.Tensor
    compiler: NavigationRLTCompilerFields
    next_compiler: NavigationRLTCompilerFields
    canonical_executed_action: torch.Tensor
    reward: torch.Tensor
    discount: torch.Tensor
    reference_gate: torch.Tensor
    behavior_gate: torch.Tensor
    behavior_raw_residual: torch.Tensor
    behavior_policy_version: torch.Tensor
    sample_weight: torch.Tensor | None = None


@dataclass(frozen=True)
class NavigationRLTZTransitionBatch:
    """Stage-2 replay rows with frozen current and next RLT tokens."""

    z_rl: torch.Tensor
    next_z_rl: torch.Tensor
    reference_embedding: torch.Tensor
    next_reference_embedding: torch.Tensor
    compiler: NavigationRLTCompilerFields
    next_compiler: NavigationRLTCompilerFields
    canonical_executed_action: torch.Tensor
    reward: torch.Tensor
    discount: torch.Tensor
    reference_gate: torch.Tensor
    behavior_gate: torch.Tensor
    behavior_raw_residual: torch.Tensor
    behavior_policy_version: torch.Tensor
    sample_weight: torch.Tensor | None = None


class _Projection(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, Z_RL_DIM),
            nn.GELU(),
            nn.LayerNorm(Z_RL_DIM),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class NavigationRLTStage1(nn.Module):
    """Masked reconstruction encoder producing exactly one 256-D z_rl."""

    def __init__(self, config: NavigationRLTConfig) -> None:
        super().__init__()
        self.config = config
        self.prefix_projection = _Projection(config.prefix_hidden_dim)
        self.topology_projection = _Projection(config.topology_dim)
        self.query = nn.Parameter(torch.empty(1, 1, Z_RL_DIM))
        self.prefix_mask_token = nn.Parameter(torch.empty(1, 1, Z_RL_DIM))
        self.topology_mask_token = nn.Parameter(torch.empty(1, 1, Z_RL_DIM))
        self.modality_embedding = nn.Parameter(torch.empty(3, Z_RL_DIM))
        self.prefix_position = nn.Parameter(
            torch.empty(config.max_prefix_tokens, Z_RL_DIM)
        )
        self.topology_position = nn.Parameter(
            torch.empty(config.max_topology_tokens, Z_RL_DIM)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=Z_RL_DIM,
            nhead=config.attention_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=config.transformer_layers
        )
        self.z_norm = nn.LayerNorm(Z_RL_DIM)
        self.prefix_decoder = nn.Sequential(
            nn.Linear(Z_RL_DIM, Z_RL_DIM),
            nn.GELU(),
            nn.LayerNorm(Z_RL_DIM),
            nn.Linear(Z_RL_DIM, config.prefix_hidden_dim),
        )
        self.topology_decoder = nn.Sequential(
            nn.Linear(Z_RL_DIM, Z_RL_DIM),
            nn.GELU(),
            nn.LayerNorm(Z_RL_DIM),
            nn.Linear(Z_RL_DIM, config.topology_dim),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for parameter in (
            self.query,
            self.prefix_mask_token,
            self.topology_mask_token,
            self.modality_embedding,
            self.prefix_position,
            self.topology_position,
        ):
            nn.init.normal_(parameter, mean=0.0, std=0.02)

    def _external(
        self,
        value: torch.Tensor,
        *,
        name: str,
        ndim: int,
        final_dim: int,
    ) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise NavigationRLTContractError(f"{name} must be a torch.Tensor")
        if value.ndim != ndim or value.shape[-1] != final_dim:
            raise NavigationRLTContractError(
                f"{name} must have rank {ndim} and final width {final_dim}; "
                f"received {tuple(value.shape)}"
            )
        if not value.is_floating_point():
            raise NavigationRLTContractError(f"{name} must be floating point")
        if self.config.strict_frozen_inputs and value.requires_grad:
            raise NavigationRLTContractError(
                f"{name} requires gradients; LightNav/context inputs must be frozen"
            )
        _require_finite(value, name)
        return value.detach().to(device=self.query.device, dtype=self.query.dtype)

    @staticmethod
    def _valid_mask(
        value: torch.Tensor,
        *,
        name: str,
        batch: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != (batch, width):
            shape = tuple(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__
            raise NavigationRLTContractError(
                f"{name} must have shape [{batch}, {width}]; received {shape}"
            )
        if value.dtype != torch.bool:
            raise NavigationRLTContractError(f"{name} must have dtype torch.bool")
        result = value.detach().to(device=device)
        if bool((~result.any(dim=1)).any()):
            raise NavigationRLTContractError(f"{name} contains an empty row")
        return result

    def prepare_context(
        self, context: NavigationRLTContext
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(context, NavigationRLTContext):
            raise NavigationRLTContractError("context must be NavigationRLTContext")
        prefix = self._external(
            context.contextual_prefix_hidden,
            name="contextual_prefix_hidden",
            ndim=3,
            final_dim=self.config.prefix_hidden_dim,
        )
        topology = self._external(
            context.topology_tokens,
            name="topology_tokens",
            ndim=3,
            final_dim=self.config.topology_dim,
        )
        batch = int(prefix.shape[0])
        if topology.shape[0] != batch:
            raise NavigationRLTContractError("prefix/topology batch sizes differ")
        if not 1 <= prefix.shape[1] <= self.config.max_prefix_tokens:
            raise NavigationRLTContractError("prefix token count is outside its contract")
        if not 1 <= topology.shape[1] <= self.config.max_topology_tokens:
            raise NavigationRLTContractError("topology token count is outside its contract")
        prefix_valid = self._valid_mask(
            context.prefix_mask,
            name="prefix_mask",
            batch=batch,
            width=int(prefix.shape[1]),
            device=prefix.device,
        )
        topology_valid = self._valid_mask(
            context.topology_mask,
            name="topology_mask",
            batch=batch,
            width=int(topology.shape[1]),
            device=prefix.device,
        )
        return prefix, prefix_valid, topology, topology_valid

    @staticmethod
    def sample_reconstruction_mask(
        valid: torch.Tensor,
        ratio: float,
        *,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        batch, _ = valid.shape
        result = torch.zeros_like(valid)
        scores = torch.rand(
            valid.shape,
            device=valid.device,
            dtype=torch.float32,
            generator=generator,
        )
        scores = scores.masked_fill(~valid, 2.0)
        for row in range(batch):
            count = int(valid[row].sum().item())
            masked = max(1, int(math.ceil(count * float(ratio))))
            # Leave one visible token when possible; the other modality and query
            # still make a one-token sequence valid.
            if count > 1:
                masked = min(masked, count - 1)
            indices = torch.argsort(scores[row])[:masked]
            result[row, indices] = True
        return result

    @staticmethod
    def validate_reconstruction_mask(
        reconstruction: torch.Tensor,
        valid: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        if not isinstance(reconstruction, torch.Tensor) or reconstruction.shape != valid.shape:
            raise NavigationRLTContractError(
                f"{name} must have shape {tuple(valid.shape)}"
            )
        if reconstruction.dtype != torch.bool:
            raise NavigationRLTContractError(f"{name} must have dtype torch.bool")
        reconstruction = reconstruction.detach().to(device=valid.device)
        if bool((reconstruction & ~valid).any()):
            raise NavigationRLTContractError(f"{name} selects padded tokens")
        if bool((~reconstruction.any(dim=1)).any()):
            raise NavigationRLTContractError(f"{name} must mask a token in every row")
        return reconstruction

    def forward(
        self,
        context: NavigationRLTContext,
        *,
        prefix_reconstruction_mask: torch.Tensor | None = None,
        topology_reconstruction_mask: torch.Tensor | None = None,
    ) -> NavigationRLTRepresentation:
        prefix, prefix_valid, topology, topology_valid = self.prepare_context(context)
        batch, prefix_width = prefix.shape[:2]
        topology_width = topology.shape[1]
        prefix_encoded = self.prefix_projection(prefix)
        topology_encoded = self.topology_projection(topology)
        if prefix_reconstruction_mask is not None:
            prefix_reconstruction_mask = self.validate_reconstruction_mask(
                prefix_reconstruction_mask, prefix_valid, "prefix_reconstruction_mask"
            )
            prefix_encoded = torch.where(
                prefix_reconstruction_mask.unsqueeze(-1),
                self.prefix_mask_token.expand(batch, prefix_width, -1),
                prefix_encoded,
            )
        if topology_reconstruction_mask is not None:
            topology_reconstruction_mask = self.validate_reconstruction_mask(
                topology_reconstruction_mask,
                topology_valid,
                "topology_reconstruction_mask",
            )
            topology_encoded = torch.where(
                topology_reconstruction_mask.unsqueeze(-1),
                self.topology_mask_token.expand(batch, topology_width, -1),
                topology_encoded,
            )
        prefix_encoded = (
            prefix_encoded
            + self.prefix_position[:prefix_width].unsqueeze(0)
            + self.modality_embedding[1].reshape(1, 1, -1)
        )
        topology_encoded = (
            topology_encoded
            + self.topology_position[:topology_width].unsqueeze(0)
            + self.modality_embedding[2].reshape(1, 1, -1)
        )
        query = self.query.expand(batch, -1, -1)
        query = query + self.modality_embedding[0].reshape(1, 1, -1)
        tokens = torch.cat((query, prefix_encoded, topology_encoded), dim=1)
        valid = torch.cat(
            (
                torch.ones((batch, 1), dtype=torch.bool, device=tokens.device),
                prefix_valid,
                topology_valid,
            ),
            dim=1,
        )
        transformed = self.transformer(tokens, src_key_padding_mask=~valid)
        z_rl = self.z_norm(transformed[:, 0])
        _require_finite(z_rl, "z_rl")
        prefix_reconstruction = None
        topology_reconstruction = None
        if prefix_reconstruction_mask is not None:
            prefix_latent = (
                z_rl[:, None, :]
                + self.prefix_position[:prefix_width].unsqueeze(0)
                + self.modality_embedding[1].reshape(1, 1, -1)
            )
            prefix_reconstruction = self.prefix_decoder(prefix_latent)
            _require_finite(prefix_reconstruction, "prefix_reconstruction")
        if topology_reconstruction_mask is not None:
            topology_latent = (
                z_rl[:, None, :]
                + self.topology_position[:topology_width].unsqueeze(0)
                + self.modality_embedding[2].reshape(1, 1, -1)
            )
            topology_reconstruction = self.topology_decoder(topology_latent)
            _require_finite(topology_reconstruction, "topology_reconstruction")
        return NavigationRLTRepresentation(
            z_rl=z_rl,
            prefix_reconstruction=prefix_reconstruction,
            topology_reconstruction=topology_reconstruction,
        )


class _HybridActor(nn.Module):
    def __init__(self, config: NavigationRLTConfig) -> None:
        super().__init__()
        self.config = config
        self.trunk = nn.Sequential(
            nn.Linear(Z_RL_DIM + config.reference_dim, Z_RL_DIM),
            nn.GELU(),
            nn.LayerNorm(Z_RL_DIM),
            nn.Linear(Z_RL_DIM, Z_RL_DIM),
            nn.GELU(),
            nn.LayerNorm(Z_RL_DIM),
        )
        self.gate_head = nn.Linear(Z_RL_DIM, GATE_COUNT)
        self.residual_head = nn.Linear(Z_RL_DIM, len(RESIDUAL_AXES))
        self.log_std = nn.Parameter(
            torch.full((len(RESIDUAL_AXES),), float(config.initial_log_std))
        )
        self.register_buffer(
            "residual_limits",
            torch.tensor(
                (
                    config.residual_forward_limit_m,
                    config.residual_yaw_limit_rad,
                ),
                dtype=torch.float32,
            ),
        )
        with torch.no_grad():
            nn.init.zeros_(self.gate_head.weight)
            self.gate_head.bias.copy_(torch.tensor((0.0, -4.0, -8.0, -6.0)))
            nn.init.zeros_(self.residual_head.weight)
            nn.init.zeros_(self.residual_head.bias)

    def distribution(
        self,
        z_rl: torch.Tensor,
        reference_embedding: torch.Tensor,
        valid_gate_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if z_rl.ndim != 2 or z_rl.shape[-1] != Z_RL_DIM:
            raise NavigationRLTContractError("z_rl must have shape [B, 256]")
        if reference_embedding.shape != (z_rl.shape[0], self.config.reference_dim):
            raise NavigationRLTContractError("reference_embedding shape differs")
        if valid_gate_mask.shape != (z_rl.shape[0], GATE_COUNT):
            raise NavigationRLTContractError("valid_gate_mask shape differs")
        if valid_gate_mask.dtype != torch.bool or bool((~valid_gate_mask.any(dim=1)).any()):
            raise NavigationRLTContractError(
                "valid_gate_mask must be bool and contain a valid gate per row"
            )
        hidden = self.trunk(torch.cat((z_rl, reference_embedding), dim=-1))
        raw_logits = self.gate_head(hidden)
        masked_logits = raw_logits.masked_fill(
            ~valid_gate_mask, torch.finfo(raw_logits.dtype).min
        )
        log_probabilities = masked_logits.log_softmax(dim=-1)
        probabilities = log_probabilities.exp()
        residual_mean = self.residual_head(hidden)
        residual_log_std = self.log_std.clamp(
            min=self.config.min_log_std, max=self.config.max_log_std
        ).expand_as(residual_mean)
        for name, value in (
            ("gate_logits", masked_logits),
            ("gate_probabilities", probabilities),
            ("residual_mean", residual_mean),
            ("residual_log_std", residual_log_std),
        ):
            _require_finite(value, name)
        return {
            "gate_logits": masked_logits,
            "gate_log_probabilities": log_probabilities,
            "gate_probabilities": probabilities,
            "residual_mean": residual_mean,
            "residual_log_std": residual_log_std,
        }

    @staticmethod
    def continuous_log_prob(
        raw: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor
    ) -> torch.Tensor:
        normalized = (raw - mean) * torch.exp(-log_std)
        gaussian = -0.5 * (
            normalized.square() + 2.0 * log_std + math.log(2.0 * math.pi)
        ).sum(dim=-1)
        squash = torch.log(1.0 - torch.tanh(raw).square() + 1.0e-6).sum(dim=-1)
        return gaussian - squash


class _QNetwork(nn.Module):
    def __init__(
        self, canonical_action_dim: int, reference_dim: int, hidden_dim: int
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(Z_RL_DIM + canonical_action_dim, hidden_dim)
        self.reference_linear = nn.Linear(reference_dim, hidden_dim, bias=False)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.reference_linear.weight)

    def forward(
        self,
        value: torch.Tensor,
        reference_embedding: torch.Tensor,
        *,
        detached_parameters: bool = False,
    ) -> torch.Tensor:
        if detached_parameters:
            value = F.silu(
                F.linear(value, self.linear1.weight.detach(), self.linear1.bias.detach())
                + F.linear(
                    reference_embedding,
                    self.reference_linear.weight.detach(),
                    bias=None,
                )
            )
            value = F.silu(
                F.linear(value, self.linear2.weight.detach(), self.linear2.bias.detach())
            )
            return F.linear(
                value, self.output.weight.detach(), self.output.bias.detach()
            ).squeeze(-1)
        value = F.silu(
            self.linear1(value) + self.reference_linear(reference_embedding)
        )
        value = F.silu(self.linear2(value))
        return self.output(value).squeeze(-1)


class NavigationRLTPolicy(nn.Module):
    """Two-stage Navigation-RLT model with an explicit freeze boundary."""

    def __init__(self, config: NavigationRLTConfig) -> None:
        super().__init__()
        if not isinstance(config, NavigationRLTConfig):
            raise NavigationRLTContractError("config must be NavigationRLTConfig")
        self.config = config
        self.stage1 = NavigationRLTStage1(config)
        self.actor = _HybridActor(config)
        self.q1 = _QNetwork(
            config.canonical_action_dim, config.reference_dim, config.critic_hidden_dim
        )
        self.q2 = _QNetwork(
            config.canonical_action_dim, config.reference_dim, config.critic_hidden_dim
        )
        self.target_q1 = copy.deepcopy(self.q1)
        self.target_q2 = copy.deepcopy(self.q2)
        self.register_buffer("stage1_frozen", torch.tensor(False, dtype=torch.bool))
        self.register_buffer(
            "gate_support_mask",
            torch.ones(GATE_COUNT, dtype=torch.bool),
            persistent=False,
        )
        self._freeze_target_critics()

    def _freeze_target_critics(self) -> None:
        self.target_q1.requires_grad_(False)
        self.target_q2.requires_grad_(False)
        self.target_q1.eval()
        self.target_q2.eval()

    def freeze_stage1(self) -> None:
        self.stage1.requires_grad_(False)
        self.stage1.eval()
        self.stage1_frozen.fill_(True)

    @torch.no_grad()
    def set_gate_support_mask(self, mask: torch.Tensor) -> None:
        """Set the replay-supported gates used by actor and target policy."""

        if not isinstance(mask, torch.Tensor) or mask.shape != (GATE_COUNT,):
            raise NavigationRLTContractError(
                f"gate_support_mask must have shape [{GATE_COUNT}]"
            )
        if mask.dtype != torch.bool or mask.requires_grad:
            raise NavigationRLTContractError(
                "gate_support_mask must be a detached bool tensor"
            )
        if not bool(mask.any()):
            raise NavigationRLTContractError(
                "gate_support_mask must contain at least one supported gate"
            )
        self.gate_support_mask.copy_(
            mask.detach().to(device=self.gate_support_mask.device)
        )

    @torch.no_grad()
    def initialize_stage2_gate_prior_from_behavior_counts(
        self, gate_counts: Sequence[int]
    ) -> torch.Tensor:
        """Reset the loaded Stage-1 actor to a support-aware replay prior.

        Stage 1 deliberately leaves the actor frozen at its conservative
        KEEP-heavy construction bias.  Stage 2 must replace that stale bias
        *after* loading the frozen Stage-1 checkpoint.  The empirical eligible
        behavior marginal is mixed with five percent uniform mass over the
        supported gates.  For any two or more supported gates this gives a
        strict margin below the trainer's 98 percent collapse boundary while
        retaining the replay distribution as the dominant prior.
        """

        if not bool(self.stage1_frozen.item()):
            raise NavigationRLTContractError(
                "Stage 2 gate prior requires a frozen Stage 1"
            )
        if (
            not isinstance(gate_counts, (list, tuple))
            or len(gate_counts) != GATE_COUNT
        ):
            raise NavigationRLTContractError(
                f"gate_counts must contain exactly {GATE_COUNT} integers"
            )
        if any(type(value) is not int or value < 0 for value in gate_counts):
            raise NavigationRLTContractError(
                "gate_counts must contain non-negative integers"
            )
        device = self.actor.gate_head.bias.device
        dtype = self.actor.gate_head.bias.dtype
        support = self.gate_support_mask.to(device=device)
        counts = torch.tensor(gate_counts, device=device, dtype=torch.float64)
        if bool((counts[support] <= 0).any()):
            raise NavigationRLTContractError(
                "every replay-supported gate must have positive behavior count"
            )
        supported_gate_count = int(support.sum().item())
        if supported_gate_count < 2:
            raise NavigationRLTContractError(
                "Stage 2 gating requires at least two replay-supported gates"
            )
        empirical = counts[support] / counts[support].sum()
        uniform = torch.full_like(empirical, 1.0 / supported_gate_count)
        mixed = (
            (1.0 - STAGE2_BEHAVIOR_PRIOR_UNIFORM_MIX) * empirical
            + STAGE2_BEHAVIOR_PRIOR_UNIFORM_MIX * uniform
        )
        prior = torch.zeros(GATE_COUNT, device=device, dtype=dtype)
        prior[support] = mixed.to(dtype=dtype)
        if float(prior.max().item()) >= 0.98:
            raise NavigationRLTContractError(
                "support-aware Stage 2 gate prior lacks collapse-boundary margin"
            )
        self.actor.gate_head.weight.zero_()
        self.actor.gate_head.bias.zero_()
        self.actor.gate_head.bias[support] = prior[support].log()
        return prior.detach().clone()

    def _supported_gate_mask(self, compiler_valid: torch.Tensor) -> torch.Tensor:
        if compiler_valid.ndim != 2 or compiler_valid.shape[-1] != GATE_COUNT:
            raise NavigationRLTContractError("compiler valid gate mask shape differs")
        support = self.gate_support_mask.to(device=compiler_valid.device)
        valid = compiler_valid & support.unsqueeze(0)
        if bool((~valid.any(dim=1)).any()):
            raise NavigationRLTContractError(
                "compiler-valid gates have no replay-supported action"
            )
        return valid

    def stage1_parameters(self):
        if bool(self.stage1_frozen.item()):
            raise NavigationRLTContractError("Stage 1 is already frozen")
        return self.stage1.parameters()

    def stage2_parameters(self):
        if not bool(self.stage1_frozen.item()):
            raise NavigationRLTContractError("freeze_stage1() is required before Stage 2")
        for module in (self.actor, self.q1, self.q2):
            yield from module.parameters()

    def train(self, mode: bool = True) -> "NavigationRLTPolicy":
        super().train(mode)
        self.target_q1.eval()
        self.target_q2.eval()
        if bool(self.stage1_frozen.item()):
            self.stage1.eval()
        return self

    @property
    def residual_limits(self) -> torch.Tensor:
        return self.actor.residual_limits

    def encode(self, context: NavigationRLTContext) -> torch.Tensor:
        return self.stage1(context).z_rl

    def stage1_loss(
        self,
        context: NavigationRLTContext,
        loss_config: Stage1LossConfig,
        *,
        prefix_reconstruction_mask: torch.Tensor | None = None,
        topology_reconstruction_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        if bool(self.stage1_frozen.item()):
            raise NavigationRLTContractError("Stage 1 loss is disabled after freezing")
        if not isinstance(loss_config, Stage1LossConfig):
            raise NavigationRLTContractError("loss_config must be Stage1LossConfig")
        prefix, prefix_valid, topology, topology_valid = self.stage1.prepare_context(context)
        if prefix_reconstruction_mask is None:
            prefix_reconstruction_mask = self.stage1.sample_reconstruction_mask(
                prefix_valid, loss_config.prefix_mask_ratio, generator=generator
            )
        else:
            prefix_reconstruction_mask = self.stage1.validate_reconstruction_mask(
                prefix_reconstruction_mask,
                prefix_valid,
                "prefix_reconstruction_mask",
            )
        if topology_reconstruction_mask is None:
            topology_reconstruction_mask = self.stage1.sample_reconstruction_mask(
                topology_valid, loss_config.topology_mask_ratio, generator=generator
            )
        else:
            topology_reconstruction_mask = self.stage1.validate_reconstruction_mask(
                topology_reconstruction_mask,
                topology_valid,
                "topology_reconstruction_mask",
            )
        representation = self.stage1(
            context,
            prefix_reconstruction_mask=prefix_reconstruction_mask,
            topology_reconstruction_mask=topology_reconstruction_mask,
        )
        if representation.prefix_reconstruction is None or representation.topology_reconstruction is None:
            raise NavigationRLTContractError("Stage 1 reconstructions are missing")
        prefix_rows = (
            representation.prefix_reconstruction - prefix
        ).square().mean(dim=-1)
        topology_rows = (
            representation.topology_reconstruction - topology
        ).square().mean(dim=-1)
        prefix_loss = prefix_rows[prefix_reconstruction_mask].mean()
        topology_loss = topology_rows[topology_reconstruction_mask].mean()
        total = (
            float(loss_config.prefix_weight) * prefix_loss
            + float(loss_config.topology_weight) * topology_loss
        )
        result = {
            "loss": total,
            "prefix_reconstruction_loss": prefix_loss,
            "topology_reconstruction_loss": topology_loss,
            "z_rl_variance": representation.z_rl.float().var(dim=0, unbiased=False).mean(),
            "prefix_masked_tokens": prefix_reconstruction_mask.sum().to(dtype=torch.float32),
            "topology_masked_tokens": topology_reconstruction_mask.sum().to(dtype=torch.float32),
        }
        for name, value in result.items():
            _require_finite(value, name)
        return result

    def _external_matrix(
        self,
        value: torch.Tensor,
        *,
        name: str,
        batch: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != (batch, width):
            raise NavigationRLTContractError(
                f"{name} must have shape [{batch}, {width}]"
            )
        if not value.is_floating_point() or (
            self.config.strict_frozen_inputs and value.requires_grad
        ):
            raise NavigationRLTContractError(
                f"{name} must be a detached floating tensor"
            )
        _require_finite(value, name)
        return value.detach().to(device=device, dtype=dtype)

    def _prepare_frozen_z(self, value: torch.Tensor, *, name: str) -> torch.Tensor:
        if not isinstance(value, torch.Tensor) or value.ndim != 2:
            raise NavigationRLTContractError(f"{name} must have shape [B, {Z_RL_DIM}]")
        if value.shape[1] != Z_RL_DIM or not value.is_floating_point():
            raise NavigationRLTContractError(f"{name} must have shape [B, {Z_RL_DIM}]")
        if value.requires_grad:
            raise NavigationRLTContractError(f"{name} must be a frozen replay tensor")
        _require_finite(value, name)
        parameter = self.actor.gate_head.weight
        return value.detach().to(device=parameter.device, dtype=parameter.dtype)

    def _prepare_canonical_executed_action(
        self,
        value: torch.Tensor,
        *,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        action = self._external_matrix(
            value,
            name="canonical_executed_action",
            batch=batch,
            width=CANONICAL_ACTION_DIM,
            device=device,
            dtype=dtype,
        )
        modes = action[:, :GATE_COUNT].argmax(dim=-1)
        rebuilt = canonical_action_from_mode_and_se2(
            modes, action[:, GATE_COUNT:]
        )
        if not torch.allclose(action, rebuilt, rtol=0.0, atol=0.0):
            raise NavigationRLTContractError(
                "canonical_executed_action must contain an exact 4-D mode one-hot"
            )
        stop_rows = modes.eq(STOP_GATE)
        if bool((action[stop_rows, GATE_COUNT:].abs() > 1.0e-7).any()):
            raise NavigationRLTContractError(
                "canonical executed STOP must have zero resolved SE(2)"
            )
        return action

    def _prepare_reference(
        self, value: torch.Tensor, *, batch: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        return self._external_matrix(
            value,
            name="reference_embedding",
            batch=batch,
            width=self.config.reference_dim,
            device=device,
            dtype=dtype,
        )

    def _prepare_compiler(
        self,
        compiler: NavigationRLTCompilerFields,
        *,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> NavigationRLTCompilerFields:
        if not isinstance(compiler, NavigationRLTCompilerFields):
            raise NavigationRLTContractError(
                "compiler must be NavigationRLTCompilerFields"
            )
        valid = compiler.valid_gate_mask
        if not isinstance(valid, torch.Tensor) or tuple(valid.shape) != (batch, GATE_COUNT):
            raise NavigationRLTContractError(
                f"valid_gate_mask must have shape [{batch}, {GATE_COUNT}]"
            )
        if valid.dtype != torch.bool:
            raise NavigationRLTContractError("valid_gate_mask must have dtype torch.bool")
        valid = valid.detach().to(device=device)
        if bool((~valid.any(dim=1)).any()):
            raise NavigationRLTContractError("valid_gate_mask contains an empty row")
        reference_action = self._external_matrix(
            compiler.reference_action,
            name="reference_action",
            batch=batch,
            width=self.config.canonical_action_dim,
            device=device,
            dtype=dtype,
        )
        adjust_base_action = self._external_matrix(
            compiler.adjust_base_action,
            name="adjust_base_action",
            batch=batch,
            width=self.config.canonical_action_dim,
            device=device,
            dtype=dtype,
        )
        stop_action = self._external_matrix(
            compiler.stop_action,
            name="stop_action",
            batch=batch,
            width=self.config.canonical_action_dim,
            device=device,
            dtype=dtype,
        )
        recover_action = self._external_matrix(
            compiler.recover_parent_action,
            name="recover_parent_action",
            batch=batch,
            width=self.config.canonical_action_dim,
            device=device,
            dtype=dtype,
        )
        for name, action, gate_index in (
            ("reference_action", reference_action, KEEP_GATE),
            ("adjust_base_action", adjust_base_action, ADJUST_GATE),
            ("stop_action", stop_action, STOP_GATE),
            ("recover_parent_action", recover_action, RECOVER_GATE),
        ):
            expected_mode = F.one_hot(
                torch.full((batch,), gate_index, device=device, dtype=torch.int64),
                num_classes=GATE_COUNT,
            ).to(dtype=dtype)
            if not torch.allclose(
                action[:, :GATE_COUNT], expected_mode, rtol=0.0, atol=0.0
            ):
                raise NavigationRLTContractError(
                    f"{name} has the wrong canonical mode one-hot"
                )
        if not torch.allclose(
            reference_action[:, GATE_COUNT:],
            adjust_base_action[:, GATE_COUNT:],
            rtol=1.0e-6,
            atol=1.0e-7,
        ):
            raise NavigationRLTContractError(
                "ADJUST base SE(2) must equal the KEEP reference SE(2)"
            )
        if bool((stop_action[:, GATE_COUNT:].abs() > 1.0e-7).any()):
            raise NavigationRLTContractError("STOP canonical SE(2) must be zero")
        basis = compiler.adjust_residual_basis
        if not isinstance(basis, torch.Tensor) or tuple(basis.shape) != (
            batch,
            len(RESIDUAL_AXES),
            self.config.canonical_action_dim,
        ):
            raise NavigationRLTContractError(
                "adjust_residual_basis has the wrong shape"
            )
        if not basis.is_floating_point() or (
            self.config.strict_frozen_inputs and basis.requires_grad
        ):
            raise NavigationRLTContractError(
                "adjust_residual_basis must be a detached floating tensor"
            )
        _require_finite(basis, "adjust_residual_basis")
        basis = basis.detach().to(device=device, dtype=dtype)
        expected_basis = torch.zeros_like(basis)
        expected_basis[:, 0, GATE_COUNT] = 1.0
        expected_basis[:, 1, GATE_COUNT + 2] = 1.0
        if not torch.allclose(basis, expected_basis, rtol=0.0, atol=0.0):
            raise NavigationRLTContractError(
                "adjust_residual_basis must map forward/yaw only and preserve mode one-hot"
            )
        parent_valid = compiler.recover_parent_valid
        if not isinstance(parent_valid, torch.Tensor) or parent_valid.shape not in (
            (batch,),
            (batch, 1),
        ):
            raise NavigationRLTContractError("recover_parent_valid shape differs")
        if parent_valid.dtype != torch.bool:
            raise NavigationRLTContractError("recover_parent_valid must be bool")
        parent_valid = parent_valid.reshape(batch).detach().to(device=device)
        parent_index = compiler.recover_parent_index
        if not isinstance(parent_index, torch.Tensor) or parent_index.shape not in (
            (batch,),
            (batch, 1),
        ):
            raise NavigationRLTContractError("recover_parent_index shape differs")
        _require_integer_tensor(parent_index, "recover_parent_index")
        if parent_index.requires_grad:
            raise NavigationRLTContractError("recover_parent_index must be detached")
        parent_index = parent_index.reshape(batch).detach().to(
            device=device, dtype=torch.int64
        )
        if bool((parent_valid & (parent_index < 0)).any()):
            raise NavigationRLTContractError(
                "valid RECOVER parent must have a non-negative index"
            )
        if bool((~parent_valid & (parent_index != -1)).any()):
            raise NavigationRLTContractError(
                "invalid RECOVER parent must use index -1"
            )
        if not bool(valid[:, STOP_GATE].all()):
            raise NavigationRLTContractError("STOP must always be compiler-valid")
        if not torch.equal(valid[:, RECOVER_GATE], parent_valid):
            raise NavigationRLTContractError(
                "RECOVER gate validity must equal causal parent availability"
            )
        return NavigationRLTCompilerFields(
            valid_gate_mask=valid,
            reference_action=reference_action,
            adjust_base_action=adjust_base_action,
            adjust_residual_basis=basis,
            stop_action=stop_action,
            recover_parent_action=recover_action,
            recover_parent_valid=parent_valid,
            recover_parent_index=parent_index,
        )

    def _candidate_action_table_from_bounded(
        self,
        bounded_residual: torch.Tensor,
        compiler: NavigationRLTCompilerFields,
    ) -> torch.Tensor:
        batch = compiler.reference_action.shape[0]
        if bounded_residual.shape != (batch, len(RESIDUAL_AXES)):
            raise NavigationRLTContractError("bounded residual/compiler batch shapes differ")
        if not bounded_residual.is_floating_point():
            raise NavigationRLTContractError("bounded_residual must be floating point")
        if bounded_residual.device != compiler.reference_action.device:
            raise NavigationRLTContractError(
                "bounded_residual and compiler actions must share one device"
            )
        _require_finite(bounded_residual, "bounded_residual")
        adjusted_unclipped = compiler.adjust_base_action + torch.einsum(
            "br,bra->ba", bounded_residual, compiler.adjust_residual_basis
        )
        adjusted_se2 = adjusted_unclipped[:, GATE_COUNT:]
        adjusted_se2 = torch.stack(
            (
                adjusted_se2[:, 0].clamp(*EXECUTOR_FORWARD_RANGE_M),
                adjusted_se2[:, 1],
                adjusted_se2[:, 2].clamp(*EXECUTOR_YAW_RANGE_RAD),
            ),
            dim=-1,
        )
        adjust_gate = torch.full(
            (batch,), ADJUST_GATE, device=adjusted_se2.device, dtype=torch.int64
        )
        adjusted = canonical_action_from_mode_and_se2(adjust_gate, adjusted_se2)
        result = torch.stack(
            (
                compiler.reference_action,
                adjusted,
                compiler.stop_action,
                compiler.recover_parent_action,
            ),
            dim=1,
        )
        _require_finite(result, "candidate_canonical_actions")
        return result

    def compile_action(
        self,
        gate: torch.Tensor,
        bounded_residual: torch.Tensor,
        compiler: NavigationRLTCompilerFields,
    ) -> torch.Tensor:
        batch = compiler.reference_action.shape[0]
        if not isinstance(gate, torch.Tensor) or gate.shape != (batch,):
            raise NavigationRLTContractError("gate/compiler batch shapes differ")
        _require_integer_tensor(gate, "gate")
        gate = gate.to(device=compiler.reference_action.device, dtype=torch.int64)
        if bool(((gate < 0) | (gate >= GATE_COUNT)).any()):
            raise NavigationRLTContractError("gate is outside KEEP/ADJUST/STOP/RECOVER")
        if bounded_residual.shape != (batch, len(RESIDUAL_AXES)):
            raise NavigationRLTContractError("action/compiler batch shapes differ")
        selected_valid = compiler.valid_gate_mask.gather(1, gate[:, None]).squeeze(1)
        if not bool(selected_valid.all()):
            raise NavigationRLTContractError("actor selected a compiler-invalid gate")
        candidates = self._candidate_action_table_from_bounded(
            bounded_residual, compiler
        )
        result = candidates.gather(
            1,
            gate[:, None, None].expand(-1, 1, self.config.canonical_action_dim),
        ).squeeze(1)
        _require_finite(result, "canonical_action")
        return result

    def actor_distribution(
        self,
        z_rl: torch.Tensor,
        reference_embedding: torch.Tensor,
        compiler: NavigationRLTCompilerFields,
    ) -> dict[str, torch.Tensor]:
        return self.actor.distribution(
            z_rl,
            reference_embedding,
            self._supported_gate_mask(compiler.valid_gate_mask),
        )

    def _sample_from_distribution(
        self,
        distribution: Mapping[str, torch.Tensor],
        compiler: NavigationRLTCompilerFields,
        *,
        deterministic: bool,
        generator: torch.Generator | None,
    ) -> NavigationRLTAction:
        probabilities = distribution["gate_probabilities"]
        if deterministic:
            gate = probabilities.argmax(dim=-1)
            raw = distribution["residual_mean"]
        else:
            gate = torch.multinomial(
                probabilities, 1, replacement=True, generator=generator
            ).squeeze(-1)
            noise = torch.randn(
                distribution["residual_mean"].shape,
                device=probabilities.device,
                dtype=probabilities.dtype,
                generator=generator,
            )
            raw = distribution["residual_mean"] + noise * torch.exp(
                distribution["residual_log_std"]
            )
        bounded = torch.tanh(raw) * self.residual_limits.to(
            device=raw.device, dtype=raw.dtype
        )
        executable = torch.where(
            gate.eq(ADJUST_GATE).unsqueeze(-1), bounded, torch.zeros_like(bounded)
        )
        canonical = self.compile_action(gate, executable, compiler)
        gate_log_prob = distribution["gate_log_probabilities"].gather(
            1, gate[:, None]
        ).squeeze(1)
        continuous = self.actor.continuous_log_prob(
            raw,
            distribution["residual_mean"],
            distribution["residual_log_std"],
        )
        log_prob = gate_log_prob + gate.eq(ADJUST_GATE).to(raw.dtype) * continuous
        _require_finite(log_prob, "action_log_prob")
        return NavigationRLTAction(
            gate=gate,
            raw_residual=raw,
            bounded_residual=bounded,
            executable_residual=executable,
            canonical_action=canonical,
            gate_logits=distribution["gate_logits"],
            gate_probabilities=probabilities,
            log_prob=log_prob,
            recover_parent_index=torch.where(
                gate.eq(RECOVER_GATE),
                compiler.recover_parent_index,
                torch.full_like(compiler.recover_parent_index, -1),
            ),
        )

    @torch.no_grad()
    def act_deterministic(
        self,
        context: NavigationRLTContext,
        reference_embedding: torch.Tensor,
        compiler: NavigationRLTCompilerFields,
    ) -> NavigationRLTAction:
        z_rl = self.encode(context)
        reference = self._prepare_reference(
            reference_embedding,
            batch=z_rl.shape[0],
            device=z_rl.device,
            dtype=z_rl.dtype,
        )
        prepared = self._prepare_compiler(
            compiler, batch=z_rl.shape[0], device=z_rl.device, dtype=z_rl.dtype
        )
        distribution = self.actor_distribution(z_rl, reference, prepared)
        return self._sample_from_distribution(
            distribution, prepared, deterministic=True, generator=None
        )

    @torch.no_grad()
    def act_stochastic(
        self,
        context: NavigationRLTContext,
        reference_embedding: torch.Tensor,
        compiler: NavigationRLTCompilerFields,
        *,
        generator: torch.Generator | None = None,
    ) -> NavigationRLTAction:
        z_rl = self.encode(context)
        reference = self._prepare_reference(
            reference_embedding,
            batch=z_rl.shape[0],
            device=z_rl.device,
            dtype=z_rl.dtype,
        )
        prepared = self._prepare_compiler(
            compiler, batch=z_rl.shape[0], device=z_rl.device, dtype=z_rl.dtype
        )
        distribution = self.actor_distribution(z_rl, reference, prepared)
        return self._sample_from_distribution(
            distribution, prepared, deterministic=False, generator=generator
        )

    def q_values(
        self,
        z_rl: torch.Tensor,
        reference_embedding: torch.Tensor,
        canonical_action: torch.Tensor,
        *,
        target: bool = False,
        actor_grad: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if target and actor_grad:
            raise NavigationRLTContractError("target and actor_grad are mutually exclusive")
        if z_rl.ndim != 2 or z_rl.shape[-1] != Z_RL_DIM:
            raise NavigationRLTContractError("critic z_rl must have shape [B, 256]")
        if canonical_action.shape != (
            z_rl.shape[0],
            self.config.canonical_action_dim,
        ):
            raise NavigationRLTContractError("canonical action shape differs")
        if reference_embedding.shape != (
            z_rl.shape[0],
            self.config.reference_dim,
        ):
            raise NavigationRLTContractError("critic reference_embedding shape differs")
        _require_finite(z_rl, "critic_z_rl")
        _require_finite(reference_embedding, "critic_reference_embedding")
        _require_finite(canonical_action, "canonical_executed_action")
        value = torch.cat((z_rl, canonical_action), dim=-1)
        first = self.target_q1 if target else self.q1
        second = self.target_q2 if target else self.q2
        q1 = first(
            value, reference_embedding, detached_parameters=actor_grad
        )
        q2 = second(
            value, reference_embedding, detached_parameters=actor_grad
        )
        _require_finite(q1, "q1")
        _require_finite(q2, "q2")
        return q1, q2

    @staticmethod
    def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if value.shape != weight.shape:
            raise NavigationRLTContractError("loss value/weight shapes differ")
        return (value * weight).sum() / weight.sum().clamp_min(1.0e-8)

    @torch.no_grad()
    def soft_update_targets(self, tau: float) -> None:
        tau = _finite_scalar(tau, "tau")
        if not 0.0 < tau <= 1.0:
            raise NavigationRLTContractError("tau must be in (0, 1]")
        for source, target in ((self.q1, self.target_q1), (self.q2, self.target_q2)):
            for source_parameter, target_parameter in zip(
                source.parameters(), target.parameters(), strict=True
            ):
                target_parameter.lerp_(source_parameter, tau)
        self._freeze_target_critics()

    def _batch_vector(
        self,
        value: torch.Tensor,
        *,
        name: str,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not isinstance(value, torch.Tensor) or value.shape not in (
            (batch,),
            (batch, 1),
        ):
            raise NavigationRLTContractError(
                f"{name} must have shape [{batch}] or [{batch}, 1]"
            )
        if value.requires_grad or not value.is_floating_point():
            raise NavigationRLTContractError(
                f"{name} must be a detached floating tensor"
            )
        result = value.reshape(batch).detach().to(device=device, dtype=dtype)
        _require_finite(result, name)
        return result

    def _behavior_gate(
        self,
        value: torch.Tensor,
        *,
        batch: int,
        device: torch.device,
        compiler: NavigationRLTCompilerFields,
    ) -> torch.Tensor:
        if not isinstance(value, torch.Tensor) or value.shape not in (
            (batch,),
            (batch, 1),
        ):
            raise NavigationRLTContractError("behavior_gate shape differs")
        _require_integer_tensor(value, "behavior_gate")
        if value.requires_grad:
            raise NavigationRLTContractError("behavior_gate must be detached")
        gate = value.reshape(batch).detach().to(device=device, dtype=torch.int64)
        if bool(((gate < 0) | (gate >= GATE_COUNT)).any()):
            raise NavigationRLTContractError(
                "behavior_gate is outside KEEP/ADJUST/STOP/RECOVER"
            )
        if not bool(compiler.valid_gate_mask.gather(1, gate[:, None]).all()):
            raise NavigationRLTContractError(
                "behavior_gate is invalid under its recorded compiler fields"
            )
        support = self.gate_support_mask.to(device=device)
        if not bool(support.gather(0, gate).all()):
            raise NavigationRLTContractError(
                "behavior_gate is absent from the persisted replay support"
            )
        return gate

    def _reference_gate(
        self,
        value: torch.Tensor,
        *,
        batch: int,
        device: torch.device,
        compiler: NavigationRLTCompilerFields,
    ) -> torch.Tensor:
        if not isinstance(value, torch.Tensor) or value.shape not in (
            (batch,),
            (batch, 1),
        ):
            raise NavigationRLTContractError("reference_gate shape differs")
        _require_integer_tensor(value, "reference_gate")
        if value.requires_grad:
            raise NavigationRLTContractError("reference_gate must be detached")
        gate = value.reshape(batch).detach().to(device=device, dtype=torch.int64)
        if bool(((gate != KEEP_GATE) & (gate != STOP_GATE)).any()):
            raise NavigationRLTContractError(
                "reference_gate must encode Native KEEP or Native STOP"
            )
        if not bool(compiler.valid_gate_mask.gather(1, gate[:, None]).all()):
            raise NavigationRLTContractError(
                "reference_gate is invalid under its recorded compiler fields"
            )
        return gate

    def _behavior_versions(
        self, value: torch.Tensor, *, batch: int, device: torch.device
    ) -> torch.Tensor:
        if not isinstance(value, torch.Tensor) or value.shape not in (
            (batch,),
            (batch, 1),
        ):
            raise NavigationRLTContractError("behavior_policy_version shape differs")
        _require_integer_tensor(value, "behavior_policy_version")
        if value.requires_grad:
            raise NavigationRLTContractError("behavior_policy_version must be detached")
        result = value.reshape(batch).detach().to(device=device, dtype=torch.int64)
        if bool((result < 0).any()):
            raise NavigationRLTContractError(
                "behavior_policy_version must be non-negative"
            )
        # Deliberately no uniqueness check: an off-policy batch may mix versions.
        return result

    def _candidate_actions(
        self,
        distribution: Mapping[str, torch.Tensor],
        compiler: NavigationRLTCompilerFields,
    ) -> torch.Tensor:
        batch = distribution["residual_mean"].shape[0]
        bounded = torch.tanh(distribution["residual_mean"]) * self.residual_limits.to(
            device=distribution["residual_mean"].device,
            dtype=distribution["residual_mean"].dtype,
        )
        if batch != compiler.reference_action.shape[0]:
            raise NavigationRLTContractError("actor/compiler batch sizes differ")
        # The same helper serves sampled compilation and candidate-Q evaluation,
        # so ADJUST clipping and its one-hot identity cannot drift.
        return self._candidate_action_table_from_bounded(bounded, compiler)

    def stage2_loss(
        self,
        replay: NavigationRLTTransitionBatch,
        loss_config: Stage2LossConfig,
        *,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode frozen contexts, then reuse the frozen-z Stage-2 formula."""

        if not isinstance(replay, NavigationRLTTransitionBatch):
            raise NavigationRLTContractError(
                "replay must be NavigationRLTTransitionBatch"
            )
        if not bool(self.stage1_frozen.item()):
            raise NavigationRLTContractError(
                "Stage 2 requires freeze_stage1() before optimizer construction"
            )
        with torch.no_grad():
            z_rl = self.encode(replay.context)
            next_z_rl = self.encode(replay.next_context)
        frozen_replay = NavigationRLTZTransitionBatch(
            z_rl=z_rl.detach(),
            next_z_rl=next_z_rl.detach(),
            reference_embedding=replay.reference_embedding,
            next_reference_embedding=replay.next_reference_embedding,
            compiler=replay.compiler,
            next_compiler=replay.next_compiler,
            canonical_executed_action=replay.canonical_executed_action,
            reward=replay.reward,
            discount=replay.discount,
            reference_gate=replay.reference_gate,
            behavior_gate=replay.behavior_gate,
            behavior_raw_residual=replay.behavior_raw_residual,
            behavior_policy_version=replay.behavior_policy_version,
            sample_weight=replay.sample_weight,
        )
        return self.stage2_loss_from_z(
            frozen_replay, loss_config, generator=generator
        )

    def stage2_loss_from_z(
        self,
        replay: NavigationRLTZTransitionBatch,
        loss_config: Stage2LossConfig,
        *,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute Stage-2 losses directly from frozen replay ``z_rl`` tensors."""

        if not bool(self.stage1_frozen.item()):
            raise NavigationRLTContractError(
                "Stage 2 requires freeze_stage1() before optimizer construction"
            )
        if not isinstance(replay, NavigationRLTZTransitionBatch):
            raise NavigationRLTContractError(
                "replay must be NavigationRLTZTransitionBatch"
            )
        if not isinstance(loss_config, Stage2LossConfig):
            raise NavigationRLTContractError("loss_config must be Stage2LossConfig")
        z_rl = self._prepare_frozen_z(replay.z_rl, name="z_rl")
        next_z_rl = self._prepare_frozen_z(replay.next_z_rl, name="next_z_rl")
        if z_rl.shape != next_z_rl.shape or z_rl.shape[0] < 1:
            raise NavigationRLTContractError(
                "current/next frozen z_rl shapes must match and be non-empty"
            )
        batch = int(z_rl.shape[0])
        device = z_rl.device
        dtype = z_rl.dtype
        reference = self._prepare_reference(
            replay.reference_embedding,
            batch=batch,
            device=device,
            dtype=dtype,
        )
        next_reference = self._prepare_reference(
            replay.next_reference_embedding,
            batch=batch,
            device=device,
            dtype=dtype,
        )
        compiler = self._prepare_compiler(
            replay.compiler, batch=batch, device=device, dtype=dtype
        )
        next_compiler = self._prepare_compiler(
            replay.next_compiler, batch=batch, device=device, dtype=dtype
        )
        executed = self._prepare_canonical_executed_action(
            replay.canonical_executed_action,
            batch=batch,
            device=device,
            dtype=dtype,
        )
        reward = self._batch_vector(
            replay.reward,
            name="reward",
            batch=batch,
            device=device,
            dtype=dtype,
        )
        discount = self._batch_vector(
            replay.discount,
            name="discount",
            batch=batch,
            device=device,
            dtype=dtype,
        )
        if bool(((discount < 0.0) | (discount > 1.0)).any()):
            raise NavigationRLTContractError("discount must lie in [0, 1]")
        if replay.sample_weight is None:
            weight = torch.ones_like(reward)
        else:
            weight = self._batch_vector(
                replay.sample_weight,
                name="sample_weight",
                batch=batch,
                device=device,
                dtype=dtype,
            )
            if bool((weight <= 0.0).any()):
                raise NavigationRLTContractError("sample_weight must be positive")
        behavior_gate = self._behavior_gate(
            replay.behavior_gate,
            batch=batch,
            device=device,
            compiler=compiler,
        )
        reference_gate = self._reference_gate(
            replay.reference_gate,
            batch=batch,
            device=device,
            compiler=compiler,
        )
        behavior_versions = self._behavior_versions(
            replay.behavior_policy_version, batch=batch, device=device
        )
        behavior_raw = self._external_matrix(
            replay.behavior_raw_residual,
            name="behavior_raw_residual",
            batch=batch,
            width=len(RESIDUAL_AXES),
            device=device,
            dtype=dtype,
        )

        with torch.no_grad():
            next_distribution = self.actor_distribution(
                next_z_rl, next_reference, next_compiler
            )
            next_action = self._sample_from_distribution(
                next_distribution,
                next_compiler,
                deterministic=True,
                generator=None,
            )
            target_q1, target_q2 = self.q_values(
                next_z_rl,
                next_reference,
                next_action.canonical_action,
                target=True,
            )
            td_target = reward + discount * torch.minimum(target_q1, target_q2)

        data_q1, data_q2 = self.q_values(z_rl, reference, executed)
        critic1_loss = self._weighted_mean(
            F.smooth_l1_loss(data_q1, td_target, reduction="none"), weight
        )
        critic2_loss = self._weighted_mean(
            F.smooth_l1_loss(data_q2, td_target, reduction="none"), weight
        )
        with torch.no_grad():
            cql_distribution = self.actor_distribution(z_rl, reference, compiler)
            cql_candidates = self._candidate_actions(
                cql_distribution, compiler
            ).detach()
        cql_q1_candidates = []
        cql_q2_candidates = []
        for gate_index in range(GATE_COUNT):
            first, second = self.q_values(
                z_rl, reference, cql_candidates[:, gate_index]
            )
            cql_q1_candidates.append(first)
            cql_q2_candidates.append(second)
        cql_q1_tensor = torch.stack(cql_q1_candidates, dim=-1)
        cql_q2_tensor = torch.stack(cql_q2_candidates, dim=-1)
        # CQL must score every structurally legal action, including a gate that
        # the behavior policy never executed.  The replay-support intersection
        # is reserved for actor sampling and target-policy bootstrapping.
        cql_valid = compiler.valid_gate_mask
        cql_temperature = float(loss_config.cql_temperature)
        cql_floor = torch.finfo(cql_q1_tensor.dtype).min
        cql_q1_masked = cql_q1_tensor.masked_fill(~cql_valid, cql_floor)
        cql_q2_masked = cql_q2_tensor.masked_fill(~cql_valid, cql_floor)
        cql1_rows = (
            cql_temperature
            * torch.logsumexp(cql_q1_masked / cql_temperature, dim=-1)
            - data_q1
        )
        cql2_rows = (
            cql_temperature
            * torch.logsumexp(cql_q2_masked / cql_temperature, dim=-1)
            - data_q2
        )
        cql1_loss = self._weighted_mean(cql1_rows, weight)
        cql2_loss = self._weighted_mean(cql2_rows, weight)
        cql_loss = cql1_loss + cql2_loss
        critic_bellman_loss = critic1_loss + critic2_loss
        critic_loss = (
            critic_bellman_loss + float(loss_config.cql_alpha) * cql_loss
        )

        dropout_probability = float(loss_config.reference_dropout_probability)
        if dropout_probability == 0.0:
            keep_reference = torch.ones(
                (batch, 1), device=device, dtype=torch.bool
            )
        elif dropout_probability == 1.0:
            keep_reference = torch.zeros(
                (batch, 1), device=device, dtype=torch.bool
            )
        else:
            keep_reference = torch.rand(
                (batch, 1), device=device, dtype=torch.float32, generator=generator
            ) >= dropout_probability
        actor_reference = torch.where(
            keep_reference, reference, torch.zeros_like(reference)
        )
        actor_distribution = self.actor_distribution(
            z_rl, actor_reference, compiler
        )
        candidate_actions = self._candidate_actions(actor_distribution, compiler)
        q_candidates = []
        for gate_index in range(GATE_COUNT):
            first, second = self.q_values(
                z_rl,
                reference,
                candidate_actions[:, gate_index],
                actor_grad=True,
            )
            q_candidates.append(torch.minimum(first, second))
        q_candidates_tensor = torch.stack(q_candidates, dim=-1)
        expected_q = (
            actor_distribution["gate_probabilities"] * q_candidates_tensor
        ).sum(dim=-1)
        actor_rl_loss = -self._weighted_mean(expected_q, weight)
        reference_gate_supported = self.gate_support_mask.to(
            device=device
        ).gather(0, reference_gate)
        # A Native reference gate can be structurally valid without appearing
        # in the behavior data.  It remains useful provenance, but cannot be a
        # BC target while the actor is intentionally support-masked.  Use a
        # finite supported target for those rows and give them zero anchor
        # weight, avoiding both an out-of-support update and -inf cross entropy.
        supported_fallback = actor_distribution[
            "gate_probabilities"
        ].detach().argmax(dim=-1)
        gate_anchor_target = torch.where(
            reference_gate_supported, reference_gate, supported_fallback
        )
        gate_anchor_rows = F.cross_entropy(
            actor_distribution["gate_logits"], gate_anchor_target, reduction="none"
        )
        gate_anchor_weight = weight * reference_gate_supported.to(dtype=weight.dtype)
        gate_anchor = self._weighted_mean(gate_anchor_rows, gate_anchor_weight)
        # The reference policy never proposes an ADJUST residual.  Exploratory
        # behavior residuals remain replay diagnostics and must not become BC
        # targets, so the conservative first-stage anchor is exactly zero.
        residual_rows = torch.tanh(
            actor_distribution["residual_mean"]
        ).square().mean(dim=-1)
        residual_anchor = self._weighted_mean(residual_rows, weight)
        reference_anchor = (
            float(loss_config.reference_gate_weight) * gate_anchor
            + float(loss_config.reference_residual_weight) * residual_anchor
        )
        if loss_config.behavior_kl_weight > 0.0:
            fixed_prior = torch.tensor(
                loss_config.behavior_gate_prior, device=device, dtype=torch.float32
            )
            if bool((fixed_prior[~self.gate_support_mask.to(device=device)] != 0.0).any()):
                raise NavigationRLTContractError("behavior prior assigns mass outside replay support")
            behavior_kl_rows = supported_behavior_gate_kl(
                actor_distribution["gate_logits"],
                self._supported_gate_mask(compiler.valid_gate_mask),
                fixed_prior,
            )
            behavior_gate_kl = self._weighted_mean(behavior_kl_rows, weight)
        else:
            behavior_gate_kl = actor_rl_loss * 0.0
        behavior_gate_kl_weighted = float(loss_config.behavior_kl_weight) * behavior_gate_kl
        actor_loss = actor_rl_loss + reference_anchor + behavior_gate_kl_weighted
        total = critic_loss + actor_loss
        unique_versions = torch.unique(behavior_versions)
        result = {
            "loss": total,
            "critic_loss": critic_loss,
            "critic_bellman_loss": critic_bellman_loss,
            "critic1_loss": critic1_loss,
            "critic2_loss": critic2_loss,
            "cql_loss": cql_loss,
            "cql1_loss": cql1_loss,
            "cql2_loss": cql2_loss,
            "cql_structural_candidate_count_mean": cql_valid.float().sum(dim=-1).mean(),
            "actor_loss": actor_loss,
            "actor_rl_loss": actor_rl_loss,
            "behavior_gate_kl_loss": behavior_gate_kl,
            "behavior_gate_kl_weighted_loss": behavior_gate_kl_weighted,
            "actor_gate_entropy": -(
                actor_distribution["gate_probabilities"].float()
                * F.log_softmax(actor_distribution["gate_logits"].float(), dim=-1)
            ).sum(dim=-1).mean(),
            "keep_q_mean": q_candidates_tensor[:, KEEP_GATE].mean(),
            "adjust_q_mean": q_candidates_tensor[:, ADJUST_GATE].mean(),
            "stop_q_mean": q_candidates_tensor[:, STOP_GATE].mean(),
            "reference_anchor_loss": reference_anchor,
            "reference_gate_loss": gate_anchor,
            "reference_gate_anchor_supported_fraction": (
                reference_gate_supported.float().mean()
            ),
            "reference_residual_loss": residual_anchor,
            "q1_data_mean": data_q1.mean(),
            "q2_data_mean": data_q2.mean(),
            "minimum_target_q_mean": torch.minimum(target_q1, target_q2).mean(),
            "td_target_mean": td_target.mean(),
            "actor_expected_q_mean": expected_q.mean(),
            "reference_dropout_fraction": (~keep_reference).float().mean(),
            "behavior_version_count": unique_versions.numel() * torch.ones(
                (), device=device, dtype=torch.float32
            ),
            "behavior_reference_gate_agreement": behavior_gate.eq(
                reference_gate
            ).float().mean(),
            "behavior_adjust_fraction": behavior_gate.eq(ADJUST_GATE).float().mean(),
            "behavior_raw_residual_norm_mean": behavior_raw.norm(dim=-1).mean(),
            "reference_stop_fraction": reference_gate.eq(STOP_GATE).float().mean(),
            "keep_probability_mean": actor_distribution["gate_probabilities"][:, KEEP_GATE].mean(),
            "adjust_probability_mean": actor_distribution["gate_probabilities"][:, ADJUST_GATE].mean(),
            "stop_probability_mean": actor_distribution["gate_probabilities"][:, STOP_GATE].mean(),
            "recover_probability_mean": actor_distribution["gate_probabilities"][:, RECOVER_GATE].mean(),
        }
        for name, value in result.items():
            _require_finite(value, name)
        return result

    def state_metadata(self) -> dict[str, Any]:
        trainable = {
            name: {
                "shape": list(parameter.shape),
                "numel": int(parameter.numel()),
                "dtype": str(parameter.dtype),
            }
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }
        return {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "feature_schema": FEATURE_SCHEMA,
            "config": self.config.to_dict(),
            "z_rl_dim": Z_RL_DIM,
            "prefix_selection_schema": PREFIX_SELECTION_SCHEMA,
            "max_artifact_prefix_tokens": self.config.max_artifact_prefix_tokens,
            "max_prefix_tokens": self.config.max_prefix_tokens,
            "topology_packing_schema": TOPOLOGY_PACKING_SCHEMA,
            "topology_node_dim": self.config.topology_node_dim,
            "topology_global_dim": self.config.topology_global_dim,
            "max_topology_tokens": self.config.max_topology_tokens,
            "canonical_action_schema": CANONICAL_ACTION_SCHEMA,
            "canonical_action_dim": CANONICAL_ACTION_DIM,
            "stage1_frozen": bool(self.stage1_frozen.item()),
            "stage1_objective": "masked_mse_prefix_and_topology_reconstruction",
            "stage2_objective": (
                "reference_conditioned_min_target_twin_q_with_finite_gate_cql_"
                "and_native_reference_actor_anchor_dropout"
            ),
            "stage2_input_modes": ["frozen_z_rl", "frozen_context"],
            "entropy_target_enabled": False,
            "cql_enabled": True,
            "cql_candidate_count": GATE_COUNT,
            "stage2_behavior_prior_uniform_mix": STAGE2_BEHAVIOR_PRIOR_UNIFORM_MIX,
            "gate_support_mask": [
                bool(value) for value in self.gate_support_mask.detach().cpu().tolist()
            ],
            "supported_gate_names": [
                name
                for name, supported in zip(
                    GATE_NAMES,
                    self.gate_support_mask.detach().cpu().tolist(),
                    strict=True,
                )
                if bool(supported)
            ],
            "gate_names": list(GATE_NAMES),
            "residual_axes": list(RESIDUAL_AXES),
            "lightnav_context": "final_layer_contextual_prefix_hidden",
            "lightnav_backbone_external": True,
            "lightnav_backbone_trainable": False,
            "ttt_enabled": False,
            "target_critics_trainable": False,
            "mixed_behavior_versions_supported": True,
            "trainable_parameter_count": sum(
                item["numel"] for item in trainable.values()
            ),
            "trainable_parameters": trainable,
        }

    def _validate_state(self) -> None:
        if (
            self.gate_support_mask.shape != (GATE_COUNT,)
            or self.gate_support_mask.dtype != torch.bool
            or not bool(self.gate_support_mask.any())
        ):
            raise NavigationRLTContractError(
                "gate_support_mask must be bool[4] with at least one supported gate"
            )
        for name, value in self.state_dict().items():
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise NavigationRLTContractError(
                    f"state tensor {name} contains non-finite values"
                )
        for prefix, module in (("target_q1", self.target_q1), ("target_q2", self.target_q2)):
            for name, parameter in module.named_parameters():
                if parameter.requires_grad:
                    raise NavigationRLTContractError(
                        f"{prefix}.{name} unexpectedly requires gradients"
                    )
        if bool(self.stage1_frozen.item()):
            for name, parameter in self.stage1.named_parameters():
                if parameter.requires_grad:
                    raise NavigationRLTContractError(
                        f"frozen Stage 1 parameter {name} requires gradients"
                    )

    def checkpoint_state(
        self,
        *,
        training_step: int,
        policy_version: int,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if int(training_step) < 0 or int(policy_version) < 0:
            raise NavigationRLTContractError(
                "training_step and policy_version must be non-negative"
            )
        self._validate_state()
        metadata = self.state_metadata()
        metadata.update(
            {
                "training_step": int(training_step),
                "policy_version": int(policy_version),
                "extra": dict(extra_metadata or {}),
            }
        )
        try:
            json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError) as error:
            raise NavigationRLTContractError(
                "checkpoint metadata must be JSON serializable"
            ) from error
        return {
            "checkpoint_schema": f"{SCHEMA_NAME}/v{SCHEMA_VERSION}",
            "metadata": metadata,
            "model_state": {
                name: value.detach().cpu().clone()
                for name, value in self.state_dict().items()
            },
        }

    def save_checkpoint(
        self,
        path: Path,
        *,
        training_step: int,
        policy_version: int,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        path = Path(path)
        if path.exists() and path.is_symlink():
            raise NavigationRLTContractError(
                "checkpoint destination must not be a symbolic link"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.checkpoint_state(
            training_step=training_step,
            policy_version=policy_version,
            extra_metadata=extra_metadata,
        )
        staging = path.with_name(path.name + ".staging")
        if staging.exists() and staging.is_symlink():
            raise NavigationRLTContractError(
                "checkpoint staging path must not be a symbolic link"
            )
        torch.save(payload, staging)
        staging.replace(path)
        return {
            "path": str(path.resolve()),
            "sha256": _sha256_file(path),
            "size_bytes": int(path.stat().st_size),
            "metadata": payload["metadata"],
        }

    @classmethod
    def load_checkpoint(
        cls,
        path: Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> tuple["NavigationRLTPolicy", dict[str, Any]]:
        raw_path = Path(path)
        if raw_path.is_symlink():
            raise NavigationRLTContractError("checkpoint must not be a symbolic link")
        path = raw_path.resolve(strict=True)
        if not path.is_file():
            raise NavigationRLTContractError("checkpoint must be a regular file")
        try:
            payload = torch.load(path, map_location=map_location, weights_only=True)
        except TypeError as error:
            raise NavigationRLTContractError(
                "this PyTorch version lacks safe weights_only checkpoint loading"
            ) from error
        if not isinstance(payload, dict) or set(payload) != {
            "checkpoint_schema",
            "metadata",
            "model_state",
        }:
            raise NavigationRLTContractError(
                "checkpoint members differ from the contract"
            )
        if payload["checkpoint_schema"] != f"{SCHEMA_NAME}/v{SCHEMA_VERSION}":
            raise NavigationRLTContractError("checkpoint schema is unsupported")
        metadata = payload["metadata"]
        if not isinstance(metadata, dict):
            raise NavigationRLTContractError("checkpoint metadata must be an object")
        if (
            metadata.get("schema_name") != SCHEMA_NAME
            or metadata.get("schema_version") != SCHEMA_VERSION
            or metadata.get("feature_schema") != FEATURE_SCHEMA
        ):
            raise NavigationRLTContractError("checkpoint metadata schema differs")
        try:
            config = NavigationRLTConfig(**metadata["config"])
        except (KeyError, TypeError, ValueError) as error:
            raise NavigationRLTContractError("checkpoint config is invalid") from error
        expected_contract_metadata = {
            "z_rl_dim": Z_RL_DIM,
            "prefix_selection_schema": PREFIX_SELECTION_SCHEMA,
            "max_artifact_prefix_tokens": config.max_artifact_prefix_tokens,
            "max_prefix_tokens": config.max_prefix_tokens,
            "topology_packing_schema": TOPOLOGY_PACKING_SCHEMA,
            "topology_node_dim": config.topology_node_dim,
            "topology_global_dim": config.topology_global_dim,
            "max_topology_tokens": config.max_topology_tokens,
            "canonical_action_schema": CANONICAL_ACTION_SCHEMA,
            "canonical_action_dim": CANONICAL_ACTION_DIM,
        }
        for name, expected in expected_contract_metadata.items():
            if metadata.get(name) != expected:
                raise NavigationRLTContractError(
                    f"checkpoint metadata {name} differs from its config/schema"
                )
        model = cls(config)
        state = payload["model_state"]
        if not isinstance(state, dict):
            raise NavigationRLTContractError(
                "checkpoint tensors differ from the model contract"
            )
        expected_state = model.state_dict()
        expected_keys = set(expected_state)
        state_keys = set(state)
        legacy_missing = {
            "q1.reference_linear.weight",
            "q2.reference_linear.weight",
            "target_q1.reference_linear.weight",
            "target_q2.reference_linear.weight",
        }
        migrated_legacy_stage1 = state_keys == expected_keys - legacy_missing
        if migrated_legacy_stage1:
            legacy_extra = metadata.get("extra")
            if not (
                metadata.get("stage1_frozen") is True
                and metadata.get("cql_enabled") is False
                and "gate_support_mask" not in metadata
                and "supported_gate_names" not in metadata
                and type(metadata.get("policy_version")) is int
                and metadata.get("policy_version") == 0
                and isinstance(legacy_extra, dict)
                and legacy_extra.get("training_phase") == "stage1"
            ):
                raise NavigationRLTContractError(
                    "legacy tensor migration is restricted to frozen Stage-1 checkpoints"
                )
            state = dict(state)
            for name in legacy_missing:
                state[name] = torch.zeros_like(expected_state[name])
        elif state_keys != expected_keys:
            raise NavigationRLTContractError(
                "checkpoint tensors differ from the model contract"
            )
        for name, value in state.items():
            if not isinstance(value, torch.Tensor):
                raise NavigationRLTContractError(
                    f"checkpoint state member {name} is not a tensor"
                )
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise NavigationRLTContractError(
                    f"checkpoint state member {name} contains non-finite values"
                )
        model.load_state_dict(state, strict=True)
        if migrated_legacy_stage1:
            metadata = dict(metadata)
            metadata["gate_support_mask"] = [True] * GATE_COUNT
            metadata["supported_gate_names"] = list(GATE_NAMES)
            metadata["load_migration"] = (
                "legacy_stage1_reference_linear_zero_and_gate_support_v1"
            )
        else:
            metadata_support = metadata.get("gate_support_mask")
            if not (
                isinstance(metadata_support, list)
                and len(metadata_support) == GATE_COUNT
                and all(type(value) is bool for value in metadata_support)
                and any(metadata_support)
            ):
                raise NavigationRLTContractError(
                    "checkpoint metadata gate_support_mask is invalid"
                )
            model.set_gate_support_mask(torch.tensor(metadata_support, dtype=torch.bool))
            expected_supported_names = [
                name
                for name, supported in zip(
                    GATE_NAMES, metadata_support, strict=True
                )
                if supported
            ]
            if metadata.get("supported_gate_names") != expected_supported_names:
                raise NavigationRLTContractError(
                    "checkpoint metadata supported_gate_names differs from mask"
                )
        model._freeze_target_critics()
        if bool(model.stage1_frozen.item()):
            model.freeze_stage1()
        model._validate_state()
        return model, metadata


__all__ = [
    "ADJUST_GATE",
    "CANONICAL_ACTION_DIM",
    "CANONICAL_ACTION_SCHEMA",
    "FEATURE_SCHEMA",
    "GATE_COUNT",
    "GATE_NAMES",
    "KEEP_GATE",
    "NavigationRLTAction",
    "NavigationRLTCompilerFields",
    "NavigationRLTConfig",
    "NavigationRLTContext",
    "NavigationRLTContractError",
    "NavigationRLTPolicy",
    "NavigationRLTPrefixSelection",
    "NavigationRLTRepresentation",
    "NavigationRLTTopologyPack",
    "NavigationRLTTransitionBatch",
    "NavigationRLTZTransitionBatch",
    "PREFIX_SELECTION_SCHEMA",
    "RECOVER_GATE",
    "RESIDUAL_AXES",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "STAGE2_BEHAVIOR_PRIOR_UNIFORM_MIX",
    "STOP_GATE",
    "Stage1LossConfig",
    "Stage2LossConfig",
    "TOPOLOGY_PACKING_SCHEMA",
    "Z_RL_DIM",
    "canonical_action_from_mode_and_se2",
    "pack_topology_tokens",
    "select_prefix_tokens",
]
