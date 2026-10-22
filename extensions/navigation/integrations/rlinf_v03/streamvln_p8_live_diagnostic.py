"""Live, train-only providers for the bounded P8 policy-preserving diagnostic.

The three bound provider methods are suitable for ``run_diagnostic``.  One
provider object owns one lazily loaded StreamVLN base policy, so Gate A, B and C
cannot accidentally load independent backbones.  Gate A performs matched
zero-residual equivalence, Gate B runs expert plus bounded DAgger BC, and Gate C
runs matched KEEP_ORIGINAL and frozen learned-residual development episodes.
Every reported metric comes from a live train-split rollout or cached live row.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from rlinf.envs import get_env_cls

try:
    from .run_streamvln_p8_train_diagnostic import ProtocolError
    from .streamvln_p8_residual_policy import (
        ACTION_NAMES,
        ACTION_COUNT,
        INTENT_NAMES,
        NAVIGATE_INTENT,
        RECOVER_INTENT,
        STOP_INTENT,
        StreamVLNPolicyPreservingResidual,
    )
    from .streamvln_p8_topology import P8TopologyTensors
    from .streamvln_ppo_policy import (
        ACTION_CHUNK,
        OriginalStreamSession,
        sample_pre_refactor_release_reference,
    )
    from .streamvln_executor_contract import feedback_from_dict
    from .streamvln_topology import (
        NAVIGATE_MODE,
        RECOVER_MODE,
        OnlineTopology,
        compile_waypoint,
    )
except ImportError:  # Direct staged-file execution.
    from run_streamvln_p8_train_diagnostic import ProtocolError
    from streamvln_p8_residual_policy import (
        ACTION_NAMES,
        ACTION_COUNT,
        INTENT_NAMES,
        NAVIGATE_INTENT,
        RECOVER_INTENT,
        STOP_INTENT,
        StreamVLNPolicyPreservingResidual,
    )
    from streamvln_p8_topology import P8TopologyTensors
    from streamvln_ppo_policy import (
        ACTION_CHUNK,
        OriginalStreamSession,
        sample_pre_refactor_release_reference,
    )
    from streamvln_executor_contract import feedback_from_dict
    from streamvln_topology import (
        NAVIGATE_MODE,
        RECOVER_MODE,
        OnlineTopology,
        compile_waypoint,
    )


ALLOWED_SPLIT = "train"
PRELOAD_OBSERVATION_KEYS = frozenset({
    "rgb",
    "instruction_utf8",
    "instruction_utf8_mask",
    "scene_utf8",
    "scene_utf8_mask",
    "episode_utf8",
    "episode_utf8_mask",
    "env_id",
    "episode_serial",
    "episode_step",
    "episode_start",
    "previous_actions",
    "previous_action_mask",
    "terminal_frozen",
})
NODE_FEATURE_DIM = 8
EDGE_FEATURE_DIM = 4
CLASS_WEIGHT_NORMALIZATION = (
    "total_rows_divided_by_present_class_count_times_class_count"
)
TRAINABLE_GROUP_PREFIXES = {
    "residual": (
        "token_projection.",
        "residual_hidden.",
        "residual_output.",
    ),
    "topology": ("topology_encoder.",),
    "decision": ("decision_projection.", "condition_fusion."),
    "waypoint": ("waypoint_head.", "waypoint_projection."),
    "intent": ("intent_head.",),
}
FROZEN_PARAMETER_PREFIXES = ("value_head.",)


def make_env(run_cfg: Mapping, *, split: str, episode_index: int, max_steps: int):
    """Construct only the frozen single StreamVLN environment needed by P8."""

    env_cfg = OmegaConf.create(
        {
            "env_type": "streamvln",
            "seed": run_cfg["seed"],
            "auto_reset": False,
            "max_episode_steps": int(max_steps),
            "python_bin": run_cfg["habitat_python"],
            "server_script": run_cfg["habitat_server"],
            "asset_root": run_cfg["asset_root"],
            "dataset_split": str(split),
            "episode_index": int(episode_index),
            "video_cfg": {"save_video": False},
            "success_distance_m": 3.0,
            "habitat_sim_python": run_cfg["habitat_sim_python"],
            "habitat_env_lib": run_cfg["habitat_env_lib"],
            "egl_vendor_json": run_cfg["egl_vendor_json"],
            "rgb_size": 64,
            "max_instruction_tokens": 200,
            "instruction_bytes": 512,
            "scene_bytes": 384,
            "episode_bytes": 64,
            "env_id_base": 0,
            "startup_timeout_s": run_cfg["habitat_startup_timeout_s"],
        }
    )
    env_cls = get_env_cls("streamvln", OmegaConf.create({}))
    return env_cls(env_cfg, 1, 0, 1, None)


def manifest_seed(manifest: Mapping, name: str) -> int:
    try:
        value = manifest["seeds"][name]
    except (KeyError, TypeError) as error:
        raise ProtocolError(f"manifest.seeds.{name} is required") from error
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError(f"manifest.seeds.{name} must be an integer")
    return int(value)


def training_contract_from_manifest(manifest: Mapping) -> dict:
    """Read every optimization choice from the frozen v2 manifest."""

    try:
        training = manifest["training"]
        architecture = manifest["architecture"]
        executor = manifest["executor"]
        loss = training["loss"]
        optimizer = training["optimizer"]
        sampling = training["sampling"]
        class_weights = training["class_weights"]
        token_loss = loss["token"]
        intent_loss = loss["intent"]
        waypoint_loss = loss["waypoint"]
    except (KeyError, TypeError) as error:
        raise ProtocolError("P8 v2 manifest is missing a frozen training field") from error

    learning_rate = float(training["learning_rate"])
    weight_decay = float(training["weight_decay"])
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ProtocolError("P8 learning rate must be finite and positive")
    if not math.isfinite(weight_decay) or weight_decay < 0.0:
        raise ProtocolError("P8 weight decay must be finite and nonnegative")
    if training.get("trainable_precision") != "fp32":
        raise ProtocolError("P8 trainable precision must be fp32")
    if training.get("frozen_backbone_precision") != "bf16":
        raise ProtocolError("P8 frozen backbone precision must be bf16")
    coefficients = {
        "token": float(token_loss["coefficient"]),
        "intent": float(intent_loss["coefficient"]),
        "waypoint": float(waypoint_loss["coefficient"]),
    }
    if any(not math.isfinite(value) or value <= 0.0 for value in coefficients.values()):
        raise ProtocolError("all three P8 loss coefficients must be finite and positive")
    smooth_l1_beta = float(waypoint_loss["beta"])
    if (
        waypoint_loss.get("type") != "SmoothL1Loss"
        or waypoint_loss.get("reduction") != "mean"
        or not math.isfinite(smooth_l1_beta)
        or smooth_l1_beta <= 0.0
    ):
        raise ProtocolError("P8 waypoint loss must be mean SmoothL1Loss with positive beta")

    betas = tuple(float(value) for value in optimizer["betas"])
    if (
        optimizer.get("type") != "AdamW"
        or len(betas) != 2
        or not all(math.isfinite(value) and 0.0 <= value < 1.0 for value in betas)
        or not 0.0 <= betas[0] < betas[1] < 1.0
    ):
        raise ProtocolError("P8 optimizer must be AdamW with valid frozen betas")
    eps = float(optimizer["eps"])
    if not math.isfinite(eps) or eps <= 0.0:
        raise ProtocolError("P8 AdamW epsilon must be finite and positive")
    if not isinstance(optimizer.get("amsgrad"), bool):
        raise ProtocolError("P8 AdamW amsgrad must be explicitly boolean")
    if optimizer.get("gradient_clip") is not None:
        raise ProtocolError("P8 v2 freezes gradient_clip=None")
    if optimizer.get("scheduler") is not None:
        raise ProtocolError("P8 v2 freezes scheduler=None")
    if sampling.get("method") != "torch.randint" or sampling.get("replacement") is not True:
        raise ProtocolError("P8 minibatches must use torch.randint sampling with replacement")

    expected_phase_sources = {
        "warmup": "deduplicated_expert_rows",
        "final": "deduplicated_expert_plus_dagger_rows",
    }
    if (
        class_weights.get("method") != "inverse_frequency"
        or class_weights.get("normalization") != CLASS_WEIGHT_NORMALIZATION
        or class_weights.get("phase_sources") != expected_phase_sources
        or class_weights.get("computed_once_per_phase") is not True
        or class_weights.get("row_duplication") is not False
    ):
        raise ProtocolError("P8 class-weight formula or phase scope differs from the manifest")

    if int(architecture["node_feature_dim"]) != NODE_FEATURE_DIM:
        raise ProtocolError("P8 node feature dimension differs from the live topology")
    if int(architecture["edge_feature_dim"]) != EDGE_FEATURE_DIM:
        raise ProtocolError("P8 edge feature dimension differs from the live topology")
    if int(architecture["topology_model_dim"]) <= 0:
        raise ProtocolError("P8 topology model dimension must be positive")
    max_waypoint_m = float(architecture["max_waypoint_m"])
    if not math.isfinite(max_waypoint_m) or max_waypoint_m <= 0.0:
        raise ProtocolError("P8 max waypoint must be finite and positive")
    if architecture.get("value_head_trainable") is not False:
        raise ProtocolError("P8 architecture requires a frozen value head")
    if training.get("value_head_trainable") is not False:
        raise ProtocolError("P8 training requires a frozen value head")
    expert_lookahead_m = float(executor["expert_lookahead_m"])
    if not math.isfinite(expert_lookahead_m) or expert_lookahead_m <= 0.0:
        raise ProtocolError("P8 expert lookahead must be finite and positive")
    if int(executor["action_chunk_size"]) != ACTION_CHUNK:
        raise ProtocolError("P8 executor action chunk differs from the released policy")
    if int(executor["blocked_forward_threshold"]) < 1:
        raise ProtocolError("P8 blocked-forward threshold must be positive")

    return {
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "loss_coefficients": coefficients,
        "smooth_l1_beta": smooth_l1_beta,
        "smooth_l1_reduction": str(waypoint_loss["reduction"]),
        "adamw_betas": betas,
        "adamw_eps": eps,
        "adamw_amsgrad": bool(optimizer["amsgrad"]),
        "gradient_clip": optimizer["gradient_clip"],
        "scheduler": optimizer["scheduler"],
        "sampling_method": str(sampling["method"]),
        "sampling_replacement": bool(sampling["replacement"]),
        "class_weight_method": str(class_weights["method"]),
        "class_weight_normalization": str(class_weights["normalization"]),
        "class_weight_phase_sources": dict(class_weights["phase_sources"]),
        "expert_lookahead_m": expert_lookahead_m,
        "blocked_forward_threshold": int(executor["blocked_forward_threshold"]),
        "topology_model_dim": int(architecture["topology_model_dim"]),
        "node_feature_dim": int(architecture["node_feature_dim"]),
        "edge_feature_dim": int(architecture["edge_feature_dim"]),
        "max_waypoint_m": max_waypoint_m,
    }


def validate_train_manifest(manifest: Mapping) -> None:
    source = manifest.get("source", {})
    if source.get("allowed_splits") != [ALLOWED_SPLIT]:
        raise ProtocolError("live P8 provider requires the train-only manifest")
    if int(manifest.get("training", {}).get("ppo_updates", -1)) != 0:
        raise ProtocolError("live P8 provider forbids PPO updates")
    if int(manifest.get("runtime", {}).get("model_load_count", -1)) != 1:
        raise ProtocolError("live P8 provider requires exactly one base model load")
    for name in ("selection", "base_wrapper_init", "residual_init"):
        manifest_seed(manifest, name)
    training_contract_from_manifest(manifest)


def progress_event_json(index: int, stage: str, event: str, payload: Mapping) -> str:
    if int(index) < 1 or not stage or not event or not isinstance(payload, Mapping):
        raise ProtocolError("P8 progress event is malformed")
    return json.dumps(
        {
            "sequence": int(index),
            "stage": str(stage),
            "event": str(event),
            "payload": dict(payload),
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def stable_seed(seed: int, *parts: object) -> int:
    payload = "|".join([str(int(seed)), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "little") % (
        2**63 - 1
    )


def fixed_class_weights_from_counts(counts: Sequence[int]) -> tuple[float, ...]:
    """Freeze inverse-frequency weights once; absent classes receive zero."""

    normalized = tuple(int(value) for value in counts)
    if not normalized or any(value < 0 for value in normalized) or not any(normalized):
        raise ProtocolError("class-weight counts must be nonnegative and nonempty")
    present = sum(value > 0 for value in normalized)
    total = sum(normalized)
    return tuple(total / (present * value) if value else 0.0 for value in normalized)


def deduplicate_state_records(records: Iterable) -> list:
    """Keep one row per state hash and reject conflicting repeated labels."""

    unique = {}
    for record in records:
        if isinstance(record, Mapping):
            state_hash = str(record["state_hash"])
            signature = record.get("target_signature")
        else:
            state_hash = str(record.state_hash)
            signature = record.target_signature
        if not state_hash:
            raise ProtocolError("training row has an empty state hash")
        previous = unique.get(state_hash)
        if previous is not None:
            old_signature = (
                previous.get("target_signature")
                if isinstance(previous, Mapping)
                else previous.target_signature
            )
            if old_signature != signature:
                raise ProtocolError("one P8 state hash has conflicting supervision")
            continue
        unique[state_hash] = record
    return list(unique.values())


def _tensor_digest(hasher, tensor: torch.Tensor) -> None:
    value = tensor.detach().contiguous().cpu()
    hasher.update(str(value.dtype).encode("ascii"))
    hasher.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    # Hash storage bytes directly: NumPy cannot represent every Torch dtype
    # used by the frozen backbone (notably bfloat16).
    storage = value.reshape(1) if value.ndim == 0 else value
    hasher.update(storage.view(torch.uint8).numpy().tobytes())


def state_sha256(
    *,
    episode_id: str,
    scene: str,
    decision_index: int,
    feature: torch.Tensor,
    topology: P8TopologyTensors,
) -> str:
    if int(decision_index) < 0:
        raise ProtocolError("state decision index must be nonnegative")
    hasher = hashlib.sha256()
    hasher.update(
        json.dumps(
            # The step is deliberately excluded: revisiting the same observed
            # causal state must deduplicate rather than manufacture a new row.
            [str(episode_id), str(scene)],
            separators=(",", ":"),
        ).encode("utf-8")
    )
    _tensor_digest(hasher, feature.float())
    for name, tensor in sorted(topology.as_dict().items()):
        hasher.update(name.encode("ascii"))
        _tensor_digest(hasher, tensor)
    return hasher.hexdigest()


def residual_sha256(module: torch.nn.Module) -> str:
    """Hash the complete small FP32 residual module."""

    hasher = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        hasher.update(name.encode("utf-8"))
        _tensor_digest(hasher, tensor)
    return hasher.hexdigest()


def frozen_backbone_full_sha256(module: torch.nn.Module) -> str:
    """Hash every frozen backbone parameter and buffer, including metadata."""

    backbone = getattr(module, "backbone", getattr(module, "model", module))
    named = [
        *(('parameter:' + name, tensor) for name, tensor in backbone.named_parameters()),
        *(('buffer:' + name, tensor) for name, tensor in backbone.named_buffers()),
    ]
    if not named:
        raise ProtocolError("frozen backbone has no tensors to hash")
    hasher = hashlib.sha256()
    for name, tensor in sorted(named):
        hasher.update(name.encode("utf-8"))
        _tensor_digest(hasher, tensor)
    return hasher.hexdigest()


def parameter_fingerprint(base_policy, residual_policy) -> dict[str, str]:
    return {
        "backbone_full_sha256": frozen_backbone_full_sha256(base_policy),
        "residual_full_sha256": residual_sha256(residual_policy),
    }


def expected_trainable_parameter_names(module: torch.nn.Module) -> tuple[str, ...]:
    """Return the complete manifest-locked P8 trainable parameter inventory."""

    parameters = dict(module.named_parameters())
    expected = {
        name
        for name in parameters
        if any(
            name.startswith(prefix)
            for prefixes in TRAINABLE_GROUP_PREFIXES.values()
            for prefix in prefixes
        )
    }
    frozen = {
        name
        for name in parameters
        if any(name.startswith(prefix) for prefix in FROZEN_PARAMETER_PREFIXES)
    }
    if not expected or expected | frozen != set(parameters) or expected & frozen:
        unexpected = sorted(set(parameters) - expected - frozen)
        raise ProtocolError(
            "P8 residual parameter inventory differs from the frozen architecture: "
            + repr(unexpected)
        )
    actual_trainable = {name for name, value in parameters.items() if value.requires_grad}
    if actual_trainable != expected:
        raise ProtocolError("P8 residual requires_grad scope differs from the frozen inventory")
    if any(parameters[name].dtype != torch.float32 for name in expected):
        raise ProtocolError("P8 trainable inventory contains a non-FP32 parameter")
    if any(parameters[name].requires_grad for name in frozen):
        raise ProtocolError("P8 frozen parameter inventory contains a trainable tensor")
    return tuple(sorted(expected))


def _named_tensor_hashes(module, *, trainable: bool) -> dict[str, str]:
    result = {}
    for name, parameter in module.named_parameters():
        if bool(parameter.requires_grad) == bool(trainable):
            hasher = hashlib.sha256()
            _tensor_digest(hasher, parameter)
            result["parameter:" + name] = hasher.hexdigest()
    if not trainable:
        for name, buffer in module.named_buffers():
            hasher = hashlib.sha256()
            _tensor_digest(hasher, buffer)
            result["buffer:" + name] = hasher.hexdigest()
    return result


def trainable_group_change_audit(
    before: Mapping[str, str], after: Mapping[str, str]
) -> dict[str, dict[str, object]]:
    if set(before) != set(after):
        raise ProtocolError("P8 trainable parameter inventory changed during Gate B")
    audit = {}
    for group, prefixes in TRAINABLE_GROUP_PREFIXES.items():
        names = sorted(
            name
            for name in before
            if any(name.startswith("parameter:" + prefix) for prefix in prefixes)
        )
        if not names:
            raise ProtocolError(f"P8 trainable group {group} has no declared tensors")
        changed = [name for name in names if before[name] != after[name]]
        audit[group] = {
            "changed": bool(changed),
            "all_changed": len(changed) == len(names),
            "changed_tensors": changed,
            "tensor_count": len(names),
        }
    return audit

def _fit_features(values: Sequence[float], width: int) -> list[float]:
    if width < 1:
        raise ProtocolError("topology feature width must be positive")
    row = [float(value) for value in values[:width]]
    return row + [0.0] * (width - len(row))


class CausalTopologyMemory:
    """OnlineTopology plus causal visual, visit, traversal and blockage state."""

    def __init__(
        self,
        *,
        max_decisions: int,
        node_hidden_dim: int,
        node_feature_dim: int = NODE_FEATURE_DIM,
        edge_feature_dim: int = EDGE_FEATURE_DIM,
    ) -> None:
        self.online = OnlineTopology(max_decisions=max_decisions)
        self.node_hidden_dim = int(node_hidden_dim)
        self.node_feature_dim = int(node_feature_dim)
        self.edge_feature_dim = int(edge_feature_dim)
        if min(self.node_hidden_dim, self.node_feature_dim, self.edge_feature_dim) < 1:
            raise ProtocolError("P8 topology dimensions must be positive")
        self.node_hidden: list[torch.Tensor] = []
        self.edge_traversals: dict[tuple[int, int], int] = {}
        self.edge_blockages: dict[tuple[int, int], int] = {}
        self._sync_nodes()

    def _sync_nodes(self) -> None:
        while len(self.node_hidden) < len(self.online.nodes):
            self.node_hidden.append(torch.zeros(self.node_hidden_dim, dtype=torch.float32))

    def observe(self, feature: torch.Tensor) -> None:
        value = feature.detach().float().reshape(-1, feature.shape[-1]).mean(dim=0).cpu()
        if value.numel() != self.node_hidden_dim:
            raise ProtocolError("StreamVLN feature width differs from P8 node hidden width")
        self._sync_nodes()
        self.node_hidden[self.online.current_node] = value.clone()

    def update_from_feedback(self, feedback: Mapping) -> None:
        validated = feedback_from_dict(feedback)
        before = int(self.online.current_node)
        blocked_before = int(self.online.blocked_forwards)
        self.online.update_from_feedback(dict(feedback))
        self._sync_nodes()
        after = int(self.online.current_node)
        edge = tuple(sorted((before, after)))
        traversed_forward = any(
            row.action == 1 and row.realized_path_m > 0.0
            for row in validated.primitives
        )
        if traversed_forward:
            self.edge_traversals[edge] = self.edge_traversals.get(edge, 0) + 1
        blocked_delta = int(self.online.blocked_forwards) - blocked_before
        if blocked_delta:
            self.edge_blockages[edge] = self.edge_blockages.get(edge, 0) + blocked_delta

    def tensors(self, *, device=None) -> P8TopologyTensors:
        self._sync_nodes()
        count = len(self.online.nodes)
        node_hidden = torch.stack(self.node_hidden, dim=0)
        node_features = []
        adjacency = torch.eye(count, dtype=torch.bool)
        edge_features = torch.zeros(
            count, count, self.edge_feature_dim, dtype=torch.float32
        )
        parent_index = torch.full((count,), -1, dtype=torch.int64)
        for index, node in enumerate(self.online.nodes):
            degree = sum(index in edge for edge in self.online.edges)
            parent_index[index] = -1 if node.parent is None else int(node.parent)
            node_features.append(
                _fit_features(
                    (
                        node.x,
                        node.z,
                        node.x - self.online.x,
                        node.z - self.online.z,
                        node.visits,
                        float(index == self.online.current_node),
                        degree,
                        float(node.parent is not None),
                    ),
                    self.node_feature_dim,
                )
            )
        for left, right in self.online.edges:
            adjacency[left, right] = adjacency[right, left] = True
        for left in range(count):
            for right in range(count):
                edge = tuple(sorted((left, right)))
                distance = math.hypot(
                    self.online.nodes[left].x - self.online.nodes[right].x,
                    self.online.nodes[left].z - self.online.nodes[right].z,
                )
                edge_features[left, right] = torch.tensor(
                    _fit_features(
                        (
                            self.edge_traversals.get(edge, 0),
                            self.edge_blockages.get(edge, 0),
                            distance,
                            float(adjacency[left, right]),
                        ),
                        self.edge_feature_dim,
                    ),
                    dtype=torch.float32,
                )
        return P8TopologyTensors(
            node_hidden=node_hidden.unsqueeze(0),
            node_features=torch.tensor(node_features, dtype=torch.float32).unsqueeze(0),
            node_mask=torch.ones((1, count), dtype=torch.bool),
            adjacency=adjacency.unsqueeze(0),
            edge_features=edge_features.unsqueeze(0),
            current_index=torch.tensor([self.online.current_node], dtype=torch.int64),
            parent_index=parent_index.unsqueeze(0),
        ).canonical(device=device)


@dataclass(frozen=True)
class DecisionTrace:
    base_logits: tuple[tuple[float, ...], ...]
    sampled_tokens: tuple[int, ...]
    proposed_actions: tuple[int, ...]
    executed_actions: tuple[int, ...]
    stop_position: int


@dataclass(frozen=True)
class EpisodeTrace:
    decisions: tuple[DecisionTrace, ...]
    terminated: bool
    truncated: bool
    terminal_metrics: Mapping[str, float]


@dataclass
class P8TrainingRow:
    state_hash: str
    target_signature: tuple
    token_hidden: torch.Tensor
    base_logits: torch.Tensor
    actions: torch.Tensor
    executed_mask: torch.Tensor
    topology: P8TopologyTensors
    target_intent: int
    target_waypoint: torch.Tensor
    waypoint_mask: bool


@dataclass(frozen=True)
class DAggerDecisionTrace:
    episode_id: str
    decision_index: int
    state_hash: str
    sampled_actions: tuple[int, ...]
    executed_actions: tuple[int, ...]
    expert_actions: tuple[int, ...]
    target_intent: int
    behavior_log_probs: tuple[float, ...]
    replay_log_probs: tuple[float, ...]
    replay_ratio_deviation: float
    blocked_before: int
    parent_available: bool
    teacher_session_isolated: bool


@dataclass(frozen=True)
class DAggerCollection:
    rows: tuple[P8TrainingRow, ...]
    traces: tuple[DAggerDecisionTrace, ...]
    sample_replay_ratio_deviation: float


@dataclass(frozen=True)
class DevelopmentEpisode:
    success: bool
    stopped: bool
    primitive_counts: Mapping[str, int]
    mode_counts: Mapping[str, int]
    intent_mode_counts: Mapping[str, int]
    blocked_state_count: int
    recover_trigger_count: int
    recover_alignment_miss_count: int
    recover_outcomes: tuple[Mapping[str, object], ...]
    terminal_metrics: Mapping[str, float]


@dataclass(frozen=True)
class FrozenClassWeights:
    action: tuple[float, ...]
    intent: tuple[float, ...]
    dataset_hash: str


def clone_teacher_session(session: OriginalStreamSession) -> OriginalStreamSession:
    """Clone only mutable causal state; the released model/tokenizer stay shared."""

    teacher = copy.copy(session)
    teacher.features = dict(session.features)
    teacher.actions = list(session.actions)
    teacher.parts = list(session.parts)
    teacher.predictor_positions = list(session.predictor_positions)
    teacher.value_positions = list(session.value_positions)
    teacher.sampled_ids = list(session.sampled_ids)
    return teacher


def _session_state_signature(session: OriginalStreamSession) -> tuple:
    return (
        session.scene,
        session.instruction,
        tuple(session.features),
        tuple(session.actions),
        tuple(id(value) for value in session.parts),
        tuple(session.predictor_positions),
        tuple(session.value_positions),
        tuple(session.sampled_ids),
        session.block_start,
        id(session.memory_prefix),
    )


def _normal_cpu_float(tensor: torch.Tensor) -> torch.Tensor:
    # Cached inference tensors must become ordinary tensors before an FP32
    # trainable layer saves them for backward.
    with torch.inference_mode(False):
        return tensor.detach().to(device="cpu", dtype=torch.float32).clone()


def training_row_dataset_sha256(rows: Sequence[P8TrainingRow]) -> str:
    """Hash state identity plus every supervised label used by Gate B."""

    if not rows:
        raise ProtocolError("cannot hash an empty P8 training dataset")
    digest = hashlib.sha256()
    for row in rows:
        actions = [int(value) for value in row.actions.tolist()]
        executed_mask = [bool(value) for value in row.executed_mask.tolist()]
        waypoint = [float(value) for value in row.target_waypoint.tolist()]
        record = {
            "state_hash": str(row.state_hash),
            "target_signature": row.target_signature,
            "actions": actions,
            "executed_mask": executed_mask,
            "target_intent": int(row.target_intent),
            "target_waypoint": waypoint,
            "waypoint_mask": bool(row.waypoint_mask),
        }
        digest.update(
            json.dumps(
                record,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
    return digest.hexdigest()


def freeze_class_weights(rows: Sequence[P8TrainingRow]) -> FrozenClassWeights:
    if not rows:
        raise ProtocolError("cannot freeze class weights for an empty P8 dataset")
    action_counts = [0] * ACTION_COUNT
    intent_counts = [0] * 3
    for row in rows:
        for action, valid in zip(row.actions.tolist(), row.executed_mask.tolist()):
            if valid:
                action_counts[int(action)] += 1
        intent_counts[int(row.target_intent)] += 1
    return FrozenClassWeights(
        action=fixed_class_weights_from_counts(action_counts),
        intent=fixed_class_weights_from_counts(intent_counts),
        dataset_hash=training_row_dataset_sha256(rows),
    )


def _pad_token_rows(values: Sequence[torch.Tensor], width: int, fill: float = 0.0):
    tail = values[0].shape[1:]
    output = torch.full((len(values), width, *tail), fill, dtype=values[0].dtype)
    mask = torch.zeros((len(values), width), dtype=torch.bool)
    for index, value in enumerate(values):
        length = int(value.shape[0])
        if not 1 <= length <= width:
            raise ProtocolError("P8 token row length is outside 1..4")
        output[index, :length] = value
        mask[index, :length] = True
    return output, mask


def _stack_topologies(rows: Sequence[P8TrainingRow], device) -> P8TopologyTensors:
    canonical = [row.topology.canonical(device="cpu") for row in rows]
    max_nodes = max(item.max_nodes for item in canonical)
    node_hidden_dim = canonical[0].node_hidden.shape[-1]
    node_feature_dim = canonical[0].node_features.shape[-1]
    edge_feature_dim = canonical[0].edge_features.shape[-1]
    batch = len(rows)
    node_hidden = torch.zeros(batch, max_nodes, node_hidden_dim)
    node_features = torch.zeros(batch, max_nodes, node_feature_dim)
    node_mask = torch.zeros(batch, max_nodes, dtype=torch.bool)
    adjacency = torch.zeros(batch, max_nodes, max_nodes, dtype=torch.bool)
    edge_features = torch.zeros(batch, max_nodes, max_nodes, edge_feature_dim)
    current_index = torch.zeros(batch, dtype=torch.int64)
    parent_index = torch.full((batch, max_nodes), -1, dtype=torch.int64)
    for index, item in enumerate(canonical):
        nodes = item.max_nodes
        node_hidden[index, :nodes] = item.node_hidden[0]
        node_features[index, :nodes] = item.node_features[0]
        node_mask[index, :nodes] = item.node_mask[0]
        adjacency[index, :nodes, :nodes] = item.adjacency[0]
        edge_features[index, :nodes, :nodes] = item.edge_features[0]
        current_index[index] = item.current_index[0]
        parent_index[index, :nodes] = item.parent_index[0]
    return P8TopologyTensors(
        node_hidden=node_hidden,
        node_features=node_features,
        node_mask=node_mask,
        adjacency=adjacency,
        edge_features=edge_features,
        current_index=current_index,
        parent_index=parent_index,
    ).canonical(device=device)


def train_residual_fp32(
    policy: StreamVLNPolicyPreservingResidual,
    rows: Sequence[P8TrainingRow],
    *,
    updates: int,
    batch_size: int,
    seed: int,
    class_weights: FrozenClassWeights,
    training_contract: Mapping,
    optimizer: torch.optim.Optimizer | None = None,
    generator: torch.Generator | None = None,
) -> dict:
    """Bounded FP32 BC using only optimization values frozen in the manifest."""

    rows = deduplicate_state_records(rows)
    if updates < 1 or batch_size < 1 or not rows:
        raise ProtocolError("bounded P8 training requires rows, updates and batch size")
    if class_weights.dataset_hash != freeze_class_weights(rows).dataset_hash:
        raise ProtocolError("P8 class weights were not frozen for this exact labeled dataset")
    if training_contract["sampling_method"] != "torch.randint":
        raise ProtocolError("P8 training requires the manifest torch.randint sampler")
    if training_contract["sampling_replacement"] is not True:
        raise ProtocolError("P8 training requires sampling with replacement")
    if training_contract["class_weight_method"] != "inverse_frequency":
        raise ProtocolError("P8 training requires frozen inverse-frequency weights")
    if training_contract["class_weight_normalization"] != CLASS_WEIGHT_NORMALIZATION:
        raise ProtocolError("P8 class-weight normalization differs from the manifest")
    if training_contract["gradient_clip"] is not None:
        raise ProtocolError("P8 training contract requires gradient_clip=None")
    if training_contract["scheduler"] is not None:
        raise ProtocolError("P8 training contract requires scheduler=None")
    if any(parameter.dtype != torch.float32 for parameter in policy.parameters()):
        raise ProtocolError("P8 residual optimizer received a non-FP32 parameter")
    device = next(policy.parameters()).device
    expected_names = expected_trainable_parameter_names(policy)
    named_parameters = dict(policy.named_parameters())
    trainable = [named_parameters[name] for name in expected_names]

    learning_rate = float(training_contract["learning_rate"])
    weight_decay = float(training_contract["weight_decay"])
    betas = tuple(float(value) for value in training_contract["adamw_betas"])
    eps = float(training_contract["adamw_eps"])
    amsgrad = bool(training_contract["adamw_amsgrad"])
    if optimizer is None:
        optimizer = torch.optim.AdamW(
            trainable,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            amsgrad=amsgrad,
        )
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if optimizer_ids != {id(parameter) for parameter in trainable}:
        raise ProtocolError("P8 AdamW parameter scope differs from declared trainables")
    defaults = optimizer.defaults
    optimizer_matches_manifest = (
        isinstance(optimizer, torch.optim.AdamW)
        and math.isclose(float(defaults["lr"]), learning_rate, rel_tol=0.0, abs_tol=0.0)
        and math.isclose(
            float(defaults["weight_decay"]), weight_decay, rel_tol=0.0, abs_tol=0.0
        )
        and tuple(float(value) for value in defaults["betas"]) == betas
        and math.isclose(float(defaults["eps"]), eps, rel_tol=0.0, abs_tol=0.0)
        and bool(defaults["amsgrad"]) is amsgrad
    )
    if not optimizer_matches_manifest:
        raise ProtocolError("P8 AdamW settings differ from the manifest")
    if generator is None:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))

    action_weight = torch.tensor(class_weights.action, dtype=torch.float32, device=device)
    intent_weight = torch.tensor(class_weights.intent, dtype=torch.float32, device=device)
    coefficients = training_contract["loss_coefficients"]
    beta = float(training_contract["smooth_l1_beta"])
    reduction = str(training_contract["smooth_l1_reduction"])
    losses = []
    component_losses = []
    finite_gradient_names = set()
    nonzero_gradient_names = set()
    for _ in range(int(updates)):
        # torch.randint is the frozen with-replacement sampler.
        indices = torch.randint(len(rows), (int(batch_size),), generator=generator).tolist()
        batch = [rows[index] for index in indices]
        hidden, mask = _pad_token_rows([row.token_hidden for row in batch], ACTION_CHUNK)
        logits, logits_mask = _pad_token_rows([row.base_logits for row in batch], ACTION_CHUNK)
        if not torch.equal(mask, logits_mask):
            raise ProtocolError("P8 hidden/logit token masks disagree")
        actions = torch.stack([row.actions for row in batch])
        executed_mask = torch.stack([row.executed_mask for row in batch]).bool()
        if not torch.equal(mask, executed_mask.cpu()):
            raise ProtocolError("P8 supervision mask differs from cached token prefix")
        topology = _stack_topologies(batch, device)
        outputs = policy(
            hidden.to(device),
            logits.to(device),
            topology,
            actions=actions.to(device),
            executed_mask=executed_mask.to(device),
        )
        token_loss = F.cross_entropy(
            outputs["policy_logits"][executed_mask.to(device)],
            actions.to(device)[executed_mask.to(device)],
            weight=action_weight,
        )
        target_intent = torch.tensor(
            [row.target_intent for row in batch], dtype=torch.int64, device=device
        )
        intent_loss = F.cross_entropy(
            outputs["intent_logits"], target_intent, weight=intent_weight
        )
        target_waypoint = torch.stack([row.target_waypoint for row in batch]).to(device)
        waypoint_mask = torch.tensor(
            [row.waypoint_mask for row in batch], dtype=torch.bool, device=device
        )
        waypoint_loss = (
            F.smooth_l1_loss(
                outputs["waypoint_mean"][waypoint_mask],
                target_waypoint[waypoint_mask],
                beta=beta,
                reduction=reduction,
            )
            if bool(waypoint_mask.any())
            else outputs["waypoint_mean"].sum() * 0.0
        )
        loss = (
            float(coefficients["token"]) * token_loss
            + float(coefficients["intent"]) * intent_loss
            + float(coefficients["waypoint"]) * waypoint_loss
        )
        if not bool(torch.isfinite(loss)):
            raise ProtocolError("P8 BC loss is nonfinite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(
            parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
            for parameter in policy.parameters()
        ):
            raise ProtocolError("P8 BC gradient is nonfinite")
        for name in expected_names:
            gradient = named_parameters[name].grad
            if gradient is None:
                continue
            finite_gradient_names.add(name)
            if bool(gradient.ne(0).any()):
                nonzero_gradient_names.add(name)
        # gradient_clip=None and scheduler=None are validated above; neither is run.
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        component_losses.append(
            {
                "token": float(token_loss.detach().cpu()),
                "intent": float(intent_loss.detach().cpu()),
                "waypoint": float(waypoint_loss.detach().cpu()),
            }
        )
    return {
        "optimizer_steps": int(updates),
        "losses": losses,
        "component_losses": component_losses,
        "optimizer_matches_manifest": True,
        "sampling_with_replacement": True,
        "gradient_clip": None,
        "scheduler": None,
        "expected_trainable_parameter_names": list(expected_names),
        "finite_gradient_names": sorted(finite_gradient_names),
        "nonzero_gradient_names": sorted(nonzero_gradient_names),
    }

def _row_trace_complete(row: P8TrainingRow) -> bool:
    token_count = int(row.token_hidden.shape[0])
    return (
        len(row.state_hash) == 64
        and bool(row.target_signature)
        and row.token_hidden.ndim == 2
        and row.base_logits.shape == (token_count, ACTION_COUNT)
        and row.actions.shape == (ACTION_CHUNK,)
        and row.executed_mask.shape == (ACTION_CHUNK,)
        and row.executed_mask.dtype == torch.bool
        and int(row.executed_mask.sum()) == token_count
        and row.target_waypoint.shape == (2,)
        and 0 <= int(row.target_intent) < len(INTENT_NAMES)
    )


@torch.inference_mode()
def evaluate_training_rows(
    policy: StreamVLNPolicyPreservingResidual,
    rows: Sequence[P8TrainingRow],
    *,
    batch_size: int,
) -> dict:
    """Evaluate every unique BC row and expose separate action/intent recalls."""

    rows = deduplicate_state_records(rows)
    if not rows or batch_size < 1:
        raise ProtocolError("Gate B full-set evaluation requires rows and batch size")
    device = next(policy.parameters()).device
    token_correct = token_total = 0
    stop_token_positive = stop_token_correct = 0
    intent_positive = [0] * len(INTENT_NAMES)
    intent_correct = [0] * len(INTENT_NAMES)
    waypoint_errors = []
    original_terms = []
    residual_terms = []
    stop_ranks = []
    target_ranks = []
    margins = []
    all_finite = True
    for start in range(0, len(rows), int(batch_size)):
        batch = rows[start : start + int(batch_size)]
        hidden, mask = _pad_token_rows([row.token_hidden for row in batch], ACTION_CHUNK)
        base_logits, logits_mask = _pad_token_rows(
            [row.base_logits for row in batch], ACTION_CHUNK
        )
        if not torch.equal(mask, logits_mask):
            raise ProtocolError("Gate B evaluation cache masks disagree")
        actions = torch.stack([row.actions for row in batch]).to(device)
        executed_mask = torch.stack([row.executed_mask for row in batch]).bool().to(device)
        if not torch.equal(mask, executed_mask.cpu()):
            raise ProtocolError("Gate B evaluation ignores an executed-prefix mask")
        outputs = policy(
            hidden.to(device),
            base_logits.to(device),
            _stack_topologies(batch, device),
            actions=actions,
            executed_mask=executed_mask,
        )
        tensor_outputs = [value for value in outputs.values() if isinstance(value, torch.Tensor)]
        all_finite = all_finite and all(
            bool(torch.isfinite(value).all()) for value in tensor_outputs
        )
        policy_logits = outputs["policy_logits"]
        prediction = policy_logits.argmax(dim=-1)
        token_correct += int(((prediction == actions) & executed_mask).sum())
        token_total += int(executed_mask.sum())

        # The STOP hard gate is the released action decision itself: a STOP
        # target must have STOP as first-token top-1.  Auxiliary intent cannot
        # satisfy or replace this action-level criterion.
        stop_token_targets = executed_mask[:, 0] & actions[:, 0].eq(0)
        stop_token_positive += int(stop_token_targets.sum())
        stop_token_correct += int(
            (stop_token_targets & prediction[:, 0].eq(0)).sum()
        )

        target_intent = torch.tensor(
            [row.target_intent for row in batch], dtype=torch.int64, device=device
        )
        predicted_intent = outputs["intent_logits"].argmax(dim=-1)
        for intent in range(len(INTENT_NAMES)):
            positive = target_intent == intent
            intent_positive[intent] += int(positive.sum())
            intent_correct[intent] += int(((predicted_intent == intent) & positive).sum())
        target_waypoint = torch.stack([row.target_waypoint for row in batch]).to(device)
        waypoint_mask = torch.tensor(
            [row.waypoint_mask for row in batch], dtype=torch.bool, device=device
        )
        if bool(waypoint_mask.any()):
            waypoint_errors.extend(
                torch.linalg.vector_norm(
                    outputs["waypoint_mean"][waypoint_mask]
                    - target_waypoint[waypoint_mask],
                    dim=-1,
                ).cpu().tolist()
            )
        base_log_probs = outputs["base_logits"].log_softmax(dim=-1)
        original_terms.extend(
            base_log_probs.gather(-1, actions[..., None]).squeeze(-1)[executed_mask]
            .cpu()
            .tolist()
        )
        residual_terms.extend(
            outputs["residual_logits"]
            .gather(-1, actions[..., None])
            .squeeze(-1)[executed_mask]
            .cpu()
            .tolist()
        )
        sorted_logits, _ = policy_logits.sort(dim=-1, descending=True)
        margins.extend((sorted_logits[..., 0] - sorted_logits[..., 1])[executed_mask].cpu().tolist())
        target_values = policy_logits.gather(-1, actions[..., None]).squeeze(-1)
        target_rank = 1 + (policy_logits > target_values[..., None]).sum(dim=-1)
        target_ranks.extend(target_rank[executed_mask].cpu().tolist())
        stop_values = policy_logits[:, 0, 0]
        stop_ranks.extend(
            (1 + (policy_logits[:, 0, :] > stop_values[:, None]).sum(dim=-1))
            .cpu()
            .tolist()
        )
    recalls = [
        intent_correct[index] / intent_positive[index]
        if intent_positive[index]
        else 0.0
        for index in range(len(INTENT_NAMES))
    ]
    finite_scalars = original_terms + residual_terms + waypoint_errors + margins
    all_finite = all_finite and all(math.isfinite(float(value)) for value in finite_scalars)
    return {
        "token_top1_accuracy": token_correct / token_total if token_total else 0.0,
        "navigate_intent_positive_count": intent_positive[NAVIGATE_INTENT],
        "navigate_intent_recall": recalls[NAVIGATE_INTENT],
        "stop_intent_positive_count": intent_positive[STOP_INTENT],
        "stop_intent_recall": recalls[STOP_INTENT],
        "stop_token_positive_count": stop_token_positive,
        "stop_token_recall": (
            stop_token_correct / stop_token_positive if stop_token_positive else 0.0
        ),
        "recover_positive_count": intent_positive[RECOVER_INTENT],
        "recover_recall": (
            recalls[RECOVER_INTENT]
            if intent_positive[RECOVER_INTENT]
            else "not_applicable"
        ),
        "waypoint_mean_l2_m": (
            sum(waypoint_errors) / len(waypoint_errors) if waypoint_errors else 0.0
        ),
        "all_finite": all_finite,
        "trace_fields_complete": all(_row_trace_complete(row) for row in rows),
        "trace_diagnostics": {
            "original_policy_term_mean": sum(original_terms) / len(original_terms),
            "learned_residual_mean": sum(residual_terms) / len(residual_terms),
            "waypoint_alignment_l2_mean": (
                sum(waypoint_errors) / len(waypoint_errors) if waypoint_errors else None
            ),
            "stop_rank_mean": sum(stop_ranks) / len(stop_ranks),
            "target_rank_mean": sum(target_ranks) / len(target_ranks),
            "top1_margin_min": min(margins),
        },
    }


def gate_b_hard_checks(metrics: Mapping, thresholds: Mapping) -> dict[str, bool]:
    recover_valid = (
        int(metrics["recover_positive_count"]) == 0
        and metrics["recover_recall"] == "not_applicable"
    ) or (
        int(metrics["recover_positive_count"]) > 0
        and isinstance(metrics["recover_recall"], (int, float))
        and math.isfinite(float(metrics["recover_recall"]))
    )
    return {
        "token_top1_accuracy": float(metrics["token_top1_accuracy"])
        >= float(thresholds["token_top1_accuracy_min"]),
        "navigate_intent_recall": float(metrics["navigate_intent_recall"])
        >= float(thresholds["navigate_intent_recall_min"]),
        "stop_intent_recall": float(metrics["stop_intent_recall"])
        >= float(thresholds["stop_intent_recall_min"]),
        "stop_token_recall": float(metrics["stop_token_recall"])
        >= float(thresholds["stop_token_recall_min"]),
        "waypoint_mean_l2_m": float(metrics["waypoint_mean_l2_m"])
        <= float(thresholds["waypoint_mean_l2_m_max"]),
        "all_finite": bool(metrics["all_finite"]),
        "trace_fields_complete": bool(metrics["trace_fields_complete"]),
        "recover_semantics": bool(recover_valid),
    }


def gate_b_metrics_pass(metrics: Mapping, thresholds: Mapping) -> bool:
    return all(gate_b_hard_checks(metrics, thresholds).values())

def classify_executed_mode(
    executed_actions: Sequence[int],
    *,
    predicted_intent: int,
    actionable_blocked: bool,
    recovery_actions: Sequence[int] | None,
) -> int:
    """Classify a decision by verified behavior rather than auxiliary intent alone."""

    executed = tuple(int(value) for value in executed_actions)
    if not executed or any(not 0 <= value < ACTION_COUNT for value in executed):
        raise ProtocolError("executed decision prefix is invalid")
    if 0 in executed:
        return STOP_INTENT
    if (
        actionable_blocked
        and int(predicted_intent) == RECOVER_INTENT
        and recovery_actions is not None
        and executed == tuple(int(value) for value in recovery_actions)
    ):
        return RECOVER_INTENT
    return NAVIGATE_INTENT


def _teacher_force_open_decision(
    session: OriginalStreamSession,
    *,
    target_actions: Sequence[int],
    generator: torch.Generator,
    state_hash: str,
    topology: P8TopologyTensors,
    target_intent: int,
    target_waypoint: Sequence[float],
) -> P8TrainingRow:
    actions = [int(value) for value in target_actions]
    if not 1 <= len(actions) <= ACTION_CHUNK or any(not 0 <= value < ACTION_COUNT for value in actions):
        raise ProtocolError("teacher-forced P8 action prefix is invalid")
    if 0 in actions and actions.index(0) != len(actions) - 1:
        raise ProtocolError("teacher-forced STOP must end the executed prefix")
    hidden_rows, logit_rows = [], []
    for action in actions:
        sampled = session.next_token(generator, greedy=True)
        hidden_rows.append(_normal_cpu_float(sampled["token_hidden"]))
        logit_rows.append(_normal_cpu_float(sampled["base_logits"]))
        session.commit_token(int(session.action_ids[action]))
    session.finish_decision()
    action_tensor = torch.zeros(ACTION_CHUNK, dtype=torch.int64)
    mask = torch.zeros(ACTION_CHUNK, dtype=torch.bool)
    action_tensor[: len(actions)] = torch.tensor(actions, dtype=torch.int64)
    mask[: len(actions)] = True
    waypoint = torch.tensor(target_waypoint, dtype=torch.float32).reshape(2)
    return P8TrainingRow(
        state_hash=state_hash,
        target_signature=(tuple(actions), int(target_intent), tuple(waypoint.tolist())),
        token_hidden=torch.stack(hidden_rows),
        base_logits=torch.stack(logit_rows),
        actions=action_tensor,
        executed_mask=mask,
        topology=topology.canonical(device="cpu"),
        target_intent=int(target_intent),
        target_waypoint=waypoint,
        waypoint_mask=int(target_intent) == NAVIGATE_INTENT,
    )


def _canonical_metrics(raw: Mapping) -> dict[str, float]:
    metrics = {}
    for key, value in raw.items():
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ProtocolError("terminal metric is not numeric")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ProtocolError("terminal metric is nonfinite")
        metrics[str(key)] = numeric
    return metrics


def _validate_identity(identity: Mapping, spec: Mapping) -> None:
    if (
        str(identity.get("episode_id")) != str(spec["episode_id"])
        or Path(str(identity.get("scene_id", ""))).stem != str(spec["scene"])
    ):
        raise ProtocolError("live P8 environment identity differs from the manifest")


class LiveP8DiagnosticProviders:
    """Stateful Gate providers sharing one released StreamVLN model load."""

    def __init__(
        self,
        *,
        manifest: Mapping,
        run_cfg: Mapping,
        base_policy_loader: Callable[[], object],
        residual_policy_loader: Callable[[object], StreamVLNPolicyPreservingResidual],
        env_factory: Callable = make_env,
    ) -> None:
        validate_train_manifest(manifest)
        self.manifest = dict(manifest)
        self.training_contract = training_contract_from_manifest(manifest)
        self.run_cfg = dict(run_cfg)
        self.base_policy_loader = base_policy_loader
        self.residual_policy_loader = residual_policy_loader
        self.env_factory = env_factory
        self.base_policy = None
        self.residual_policy = None
        self.model_load_count = 0
        self.split_read_counts = {ALLOWED_SPLIT: 0}
        self.probe_split_read_counts = {ALLOWED_SPLIT: 0}
        self._gate_a_passed = False
        self._gate_b_passed = False
        self._gate_b_checkpoint_hash = None
        self._progress_index = 0

    def _emit_progress(self, stage: str, event: str, **payload) -> None:
        self._progress_index += 1
        line = progress_event_json(self._progress_index, stage, event, payload)
        output_dir = Path(self.run_cfg["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "progress.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
        print(line, flush=True)

    def _verify_residual_architecture(
        self, policy: StreamVLNPolicyPreservingResidual
    ) -> None:
        contract = self.training_contract
        encoder = policy.topology_encoder
        actual = {
            "topology_model_dim": int(policy.model_dim),
            "node_feature_dim": int(encoder.node_feature_dim),
            "edge_feature_dim": int(encoder.edge_feature_dim),
            "max_waypoint_m": float(policy.max_waypoint_m),
        }
        expected = {
            key: contract[key]
            for key in (
                "topology_model_dim",
                "node_feature_dim",
                "edge_feature_dim",
                "max_waypoint_m",
            )
        }
        if actual != expected or int(encoder.model_dim) != int(contract["topology_model_dim"]):
            raise ProtocolError("loaded P8 residual architecture differs from the manifest")
        if any(parameter.requires_grad for parameter in policy.value_head.parameters()):
            raise ProtocolError("P8 value head is not frozen")
        expected_trainable_parameter_names(policy)

    def _load_once(self):
        if self.base_policy is None:
            self.base_policy = self.base_policy_loader()
            self.model_load_count += 1
            if self.model_load_count != 1:
                raise ProtocolError("P8 base model was loaded more than once")
            self.base_policy.eval()
            self.base_policy.requires_grad_(False)
            self.residual_policy = self.residual_policy_loader(self.base_policy)
            if not isinstance(self.residual_policy, StreamVLNPolicyPreservingResidual):
                raise ProtocolError("P8 residual loader returned the wrong policy type")
            # Gate B has no value target.  Keep the value head outside AdamW so
            # an unsupervised head cannot be reported as trained.
            for parameter in self.residual_policy.value_head.parameters():
                parameter.requires_grad_(False)
            self._verify_residual_architecture(self.residual_policy)
        return self.base_policy, self.residual_policy

    def _train_env(self, spec: Mapping, *, max_steps: int):
        self.split_read_counts[ALLOWED_SPLIT] += 1
        return self.env_factory(
            self.run_cfg,
            split=ALLOWED_SPLIT,
            episode_index=int(spec["index"]),
            max_steps=int(max_steps),
        )

    def preload_env_probe(self) -> dict:
        """Load and reset every selected Habitat scene before any 7B load."""

        if (
            self.base_policy is not None
            or self.residual_policy is not None
            or self.model_load_count != 0
        ):
            raise ProtocolError("P8 environment probe must precede the base-model load")
        if self.split_read_counts != {ALLOWED_SPLIT: 0}:
            raise ProtocolError("P8 scientific split reads occurred before the environment probe")
        if self.probe_split_read_counts != {ALLOWED_SPLIT: 0}:
            raise ProtocolError("P8 environment probe may run only once")

        specs = list(self.manifest["episodes"]["tiny_train"]) + list(
            self.manifest["episodes"]["train_development"]
        )
        if (
            len(specs) != 16
            or len({int(spec["index"]) for spec in specs}) != 16
            or len({str(spec["scene"]) for spec in specs}) != 16
        ):
            raise ProtocolError("P8 environment probe requires 16 distinct frozen identities")

        identities = []
        for probe_index, spec in enumerate(specs):
            env = None
            failure = None
            identity = None
            try:
                self.probe_split_read_counts[ALLOWED_SPLIT] += 1
                env = self.env_factory(
                    self.run_cfg,
                    split=ALLOWED_SPLIT,
                    episode_index=int(spec["index"]),
                    max_steps=ACTION_CHUNK,
                )
                obs, identity = env.reset()
                _validate_identity(identity, spec)
                if set(obs) != PRELOAD_OBSERVATION_KEYS:
                    raise ProtocolError("pre-load StreamVLN observation keys changed")
                rgb = obs["rgb"]
                if (
                    not isinstance(rgb, torch.Tensor)
                    or tuple(rgb.shape) != (1, 64, 64, 3)
                    or rgb.dtype != torch.uint8
                ):
                    raise ProtocolError("pre-load StreamVLN RGB tensor contract changed")
                for mask_name in (
                    "instruction_utf8_mask",
                    "scene_utf8_mask",
                    "episode_utf8_mask",
                ):
                    mask = obs[mask_name]
                    if (
                        not isinstance(mask, torch.Tensor)
                        or mask.dtype != torch.bool
                        or not bool(mask.any().item())
                    ):
                        raise ProtocolError(
                            f"pre-load StreamVLN {mask_name} tensor contract changed"
                        )
                if (
                    obs["episode_start"].dtype != torch.bool
                    or not bool(obs["episode_start"].item())
                    or bool(obs["previous_action_mask"].any().item())
                    or bool(obs["terminal_frozen"].item())
                ):
                    raise ProtocolError("pre-load StreamVLN reset-state contract changed")
                expert = env.expert_waypoint(
                    lookahead_m=self.training_contract["expert_lookahead_m"]
                )
                if (
                    expert.get("source") != "train-shortest-path"
                    or not math.isclose(
                        float(expert.get("lookahead_m", float("nan"))),
                        self.training_contract["expert_lookahead_m"],
                    )
                ):
                    raise ProtocolError("pre-load expert-waypoint contract changed")
            except Exception as error:  # Converted below with frozen identity context.
                failure = error
            finally:
                if env is not None:
                    try:
                        env.close()
                    except Exception as error:
                        if failure is None:
                            failure = error
            if failure is not None:
                raise ProtocolError(
                    "P8 pre-load environment probe failed for "
                    f"episode {spec['episode_id']} at probe index {probe_index}"
                ) from failure
            identities.append({
                "index": int(spec["index"]),
                "episode_id": str(identity["episode_id"]),
                "scene": Path(str(identity["scene_id"])).stem,
            })

        if (
            self.base_policy is not None
            or self.residual_policy is not None
            or self.model_load_count != 0
            or self.split_read_counts != {ALLOWED_SPLIT: 0}
        ):
            raise ProtocolError("P8 environment probe changed model or scientific-read state")
        return {
            "schema_version": 1,
            "kind": "streamvln-p8-preload-env-probe",
            "episode_count": len(identities),
            "scene_count": len({row["scene"] for row in identities}),
            "identities": identities,
            "observation_keys": sorted(PRELOAD_OBSERVATION_KEYS),
            "rgb_shape": [1, 64, 64, 3],
            "rgb_dtype": "torch.uint8",
            "observation_valid": True,
            "expert_waypoint_valid": True,
            "model_load_count": self.model_load_count,
            "probe_split_read_counts": {
                "train": self.probe_split_read_counts[ALLOWED_SPLIT],
                "val_seen": 0,
                "val_unseen": 0,
            },
            "scientific_split_read_counts": {
                "train": self.split_read_counts[ALLOWED_SPLIT],
                "val_seen": 0,
                "val_unseen": 0,
            },
        }

    @staticmethod
    def _session(base_policy) -> OriginalStreamSession:
        session = base_policy._ensure_session()
        if not isinstance(session, OriginalStreamSession):
            raise ProtocolError("base policy did not expose OriginalStreamSession")
        return session

    @staticmethod
    def _feature(base_policy, session, obs) -> torch.Tensor:
        rgb = base_policy._row(obs, "rgb").detach().cpu().numpy().astype("uint8")
        return session.encode_observation(rgb)

    @staticmethod
    @torch.inference_mode()
    def _sample_open_decision(
        session: OriginalStreamSession,
        generator: torch.Generator,
        residual_callback=None,
    ):
        tokens, actions, logits, residuals, hidden_rows, log_probs = [], [], [], [], [], []

        def recording_callback(hidden, base):
            value = residual_callback(hidden, base)
            residuals.append(value.detach().float().cpu())
            return value

        callback = recording_callback if residual_callback is not None else None
        for _ in range(ACTION_CHUNK):
            sampled = session.next_token(generator, residual=callback)
            tokens.append(int(sampled["token"]))
            actions.append(int(sampled["action"]))
            logits.append(_normal_cpu_float(sampled["base_logits"]))
            hidden_rows.append(_normal_cpu_float(sampled["token_hidden"]))
            log_probs.append(float(sampled["log_prob"]))
            session.commit_token(tokens[-1])
            if actions[-1] == 0:
                break
        session.finish_decision()
        return tokens, actions, logits, residuals, hidden_rows, log_probs

    @torch.inference_mode()
    def _rollout_equivalence_episode(
        self, spec: Mapping, *, use_release_reference: bool
    ) -> EpisodeTrace:
        base_policy, residual_policy = self._load_once()
        session = self._session(base_policy)
        max_steps = int(self.manifest["runtime"]["max_primitive_steps"])
        max_decisions = max_steps // ACTION_CHUNK
        node_hidden_dim = int(residual_policy.topology_encoder.node_hidden_dim)
        memory = CausalTopologyMemory(
            max_decisions=max_decisions,
            node_hidden_dim=node_hidden_dim,
            node_feature_dim=int(residual_policy.topology_encoder.node_feature_dim),
            edge_feature_dim=int(residual_policy.topology_encoder.edge_feature_dim),
        )
        env = self._train_env(spec, max_steps=max_steps)
        decisions = []
        last_info = None
        terminated = truncated = False
        try:
            obs, identity = env.reset()
            _validate_identity(identity, spec)
            scene, episode, instruction = base_policy._identity(obs)
            session.begin_task(scene + "::" + episode, instruction)
            for decision_index in range(max_decisions):
                step = len(session.actions)
                feature = self._feature(base_policy, session, obs)
                memory.observe(feature)
                topology = memory.tensors(device=next(residual_policy.parameters()).device)
                generator = torch.Generator(device=session.device).manual_seed(
                    stable_seed(manifest_seed(self.manifest, "base_wrapper_init"), spec["episode_id"], step)
                )
                if use_release_reference:
                    sampled = sample_pre_refactor_release_reference(
                        session,
                        step,
                        feature,
                        generator,
                        greedy=False,
                    )
                    tokens = list(sampled["sampled_tokens"])
                    actions = list(sampled["proposed_actions"])
                    base_logits = list(sampled["base_logits"])
                    residuals = []
                else:
                    session.open_decision(step, feature)
                    callback = residual_policy.decision_residual_callback(topology)
                    tokens, actions, base_logits, residuals, _, _ = self._sample_open_decision(
                        session, generator, callback
                    )
                if not use_release_reference and any(
                    bool(row.ne(0).any()) for row in residuals
                ):
                    raise ProtocolError("Gate A zero residual is not exactly zero")
                padded = torch.zeros((1, ACTION_CHUNK, 1), dtype=torch.int64)
                padded[0, : len(actions), 0] = torch.tensor(actions, dtype=torch.int64)
                obs_list, _, terminations, truncations, infos = env.chunk_step(
                    padded, execute_count=len(actions)
                )
                last_info = infos[0]
                raw_feedback = last_info.get("executor_feedback")
                if not isinstance(raw_feedback, Mapping):
                    raise ProtocolError("Gate A lacks executor feedback")
                feedback = feedback_from_dict(raw_feedback)
                executed = tuple(int(value) for value in feedback.executed_actions)
                memory.update_from_feedback(raw_feedback)
                session.acknowledge(actions, executed)
                decisions.append(
                    DecisionTrace(
                        base_logits=tuple(tuple(float(value) for value in row) for row in base_logits),
                        sampled_tokens=tuple(tokens),
                        proposed_actions=tuple(actions),
                        executed_actions=executed,
                        stop_position=actions.index(0) + 1 if 0 in actions else 0,
                    )
                )
                obs = obs_list[0]
                terminated = bool(terminations.any())
                truncated = bool(truncations.any())
                if terminated or truncated:
                    break
            if last_info is None or "episode" not in last_info or not (terminated or truncated):
                raise ProtocolError("Gate A episode lacks terminal evidence")
            return EpisodeTrace(
                decisions=tuple(decisions),
                terminated=terminated,
                truncated=truncated,
                terminal_metrics=_canonical_metrics(last_info["episode"]),
            )
        finally:
            env.close()

    def gate_a_provider(self) -> dict:
        base_policy, residual_policy = self._load_once()
        residual_policy.eval()
        before = parameter_fingerprint(base_policy, residual_policy)
        fields = (
            "base_logits",
            "sampled_tokens",
            "primitive_actions",
            "proposed_prefix",
            "executed_prefix",
            "stop_position",
            "termination",
            "terminal_metrics",
        )
        mismatches = {key: 0 for key in fields}
        equivalence_specs = self.manifest["episodes"]["equivalence"]
        for episode_index, spec in enumerate(equivalence_specs, start=1):
            original = self._rollout_equivalence_episode(
                spec, use_release_reference=True
            )
            residual = self._rollout_equivalence_episode(
                spec, use_release_reference=False
            )
            if [row.base_logits for row in original.decisions] != [
                row.base_logits for row in residual.decisions
            ]:
                mismatches["base_logits"] += 1
            if [row.sampled_tokens for row in original.decisions] != [
                row.sampled_tokens for row in residual.decisions
            ]:
                mismatches["sampled_tokens"] += 1
            if [row.proposed_actions for row in original.decisions] != [
                row.proposed_actions for row in residual.decisions
            ]:
                mismatches["primitive_actions"] += 1
                mismatches["proposed_prefix"] += 1
            if [row.executed_actions for row in original.decisions] != [
                row.executed_actions for row in residual.decisions
            ]:
                mismatches["executed_prefix"] += 1
            if [row.stop_position for row in original.decisions] != [
                row.stop_position for row in residual.decisions
            ]:
                mismatches["stop_position"] += 1
            if (original.terminated, original.truncated) != (
                residual.terminated,
                residual.truncated,
            ):
                mismatches["termination"] += 1
            if original.terminal_metrics != residual.terminal_metrics:
                mismatches["terminal_metrics"] += 1
            self._emit_progress(
                "gate_a",
                "matched_episode_pair_complete",
                episode_index=episode_index,
                episode_count=len(equivalence_specs),
                episode_id=str(spec["episode_id"]),
                mismatch_counts=dict(mismatches),
            )
        after = parameter_fingerprint(base_policy, residual_policy)
        self._gate_a_passed = all(value == 0 for value in mismatches.values()) and before == after
        return {
            "episode_count": len(self.manifest["episodes"]["equivalence"]),
            "mismatch_counts": mismatches,
            "optimizer_steps": 0,
            "parameter_hash_unchanged": before == after,
            "parameter_hash_scope": {
                "backbone": "complete_named_parameters_and_buffers",
                "residual": "complete_state_dict",
            },
            "backbone_full_hash_before": before["backbone_full_sha256"],
            "backbone_full_hash_after": after["backbone_full_sha256"],
            "residual_hash_before": before["residual_full_sha256"],
            "residual_hash_after": after["residual_full_sha256"],
            "model_load_count": self.model_load_count,
            "release_reference": "pre_refactor_sampler_at_5434ab0",
            "split_read_counts": {
                "train": self.split_read_counts[ALLOWED_SPLIT],
                "val_seen": 0,
                "val_unseen": 0,
            },
        }

    def gate_b_provider(self) -> dict:
        if not self._gate_a_passed:
            raise ProtocolError("live Gate B cannot run before exact Gate A equivalence")
        base_policy, residual_policy = self._load_once()
        training = self.manifest["training"]
        contract = self.training_contract
        specs = self.manifest["episodes"]["tiny_train"]
        if len(specs) != 8 or len({item["episode_id"] for item in specs}) != 8:
            raise ProtocolError("Gate B requires the frozen eight expert identities")
        start_reads = self.split_read_counts[ALLOWED_SPLIT]
        backbone_before = frozen_backbone_full_sha256(base_policy)
        residual_before = residual_sha256(residual_policy)
        expected_trainable_names = expected_trainable_parameter_names(residual_policy)
        expected_trainable_keys = {
            "parameter:" + name for name in expected_trainable_names
        }
        trainable_before = _named_tensor_hashes(residual_policy, trainable=True)
        if set(trainable_before) != expected_trainable_keys:
            raise ProtocolError("P8 pre-training parameter scope differs from the frozen inventory")
        undeclared_before = _named_tensor_hashes(residual_policy, trainable=False)
        value_head_before = _named_tensor_hashes(
            residual_policy.value_head, trainable=False
        )
        self._emit_progress("gate_b", "started", episode_count=len(specs))

        expert_rows = []
        for episode_index, spec in enumerate(specs, start=1):
            expert_rows.extend(
                collect_expert_rows(
                    providers=self,
                    spec=spec,
                    max_decisions=int(training["expert_max_decisions_per_episode"]),
                )
            )
            self._emit_progress(
                "gate_b",
                "expert_episode_complete",
                episode_index=episode_index,
                episode_count=len(specs),
                episode_id=str(spec["episode_id"]),
                collected_row_count=len(expert_rows),
            )
        expert_rows = deduplicate_state_records(expert_rows)
        if not expert_rows:
            raise ProtocolError("Gate B expert collection produced no unique states")
        warmup_weights = freeze_class_weights(expert_rows)
        named_parameters = dict(residual_policy.named_parameters())
        trainable = [named_parameters[name] for name in expected_trainable_names]
        optimizer = torch.optim.AdamW(
            trainable,
            lr=float(contract["learning_rate"]),
            weight_decay=float(contract["weight_decay"]),
            betas=tuple(contract["adamw_betas"]),
            eps=float(contract["adamw_eps"]),
            amsgrad=bool(contract["adamw_amsgrad"]),
        )
        batch_generator = torch.Generator(device="cpu").manual_seed(
            stable_seed(manifest_seed(self.manifest, "residual_init"), "bc-minibatches")
        )
        residual_policy.train()
        warmup_result = train_residual_fp32(
            residual_policy,
            expert_rows,
            updates=int(training["warmup_updates"]),
            batch_size=int(training["batch_size"]),
            training_contract=contract,
            seed=stable_seed(manifest_seed(self.manifest, "residual_init"), "warmup"),
            class_weights=warmup_weights,
            optimizer=optimizer,
            generator=batch_generator,
        )
        finite_gradient_names = set(warmup_result["finite_gradient_names"])
        nonzero_gradient_names = set(warmup_result["nonzero_gradient_names"])
        self._emit_progress(
            "gate_b",
            "warmup_complete",
            optimizer_steps=int(warmup_result["optimizer_steps"]),
            final_loss=float(warmup_result["losses"][-1]),
            unique_state_count=len(expert_rows),
        )

        residual_policy.eval()
        dagger = collect_bounded_dagger_rows(
            providers=self,
            specs=specs,
            max_decisions_per_episode=int(
                training["dagger_max_decisions_per_episode"]
            ),
        )
        self._emit_progress(
            "gate_b",
            "dagger_complete",
            unique_state_count=len(dagger.rows),
            trace_count=len(dagger.traces),
            sample_replay_ratio_deviation=float(
                dagger.sample_replay_ratio_deviation
            ),
        )
        combined_rows = deduplicate_state_records([*expert_rows, *dagger.rows])
        final_weights = freeze_class_weights(combined_rows)
        interval = int(training["metric_interval_updates"])
        final_ceiling = int(training["final_updates"])
        required_consecutive = int(training["consecutive_gate_b_passes"])
        if interval < 1 or final_ceiling < 1 or required_consecutive < 1:
            raise ProtocolError("Gate B update/evaluation schedule is invalid")
        consecutive = 0
        final_steps = 0
        evaluation_history = []
        final_losses = []
        residual_policy.train()
        while final_steps < final_ceiling:
            block = min(interval, final_ceiling - final_steps)
            block_result = train_residual_fp32(
                residual_policy,
                combined_rows,
                updates=block,
                batch_size=int(training["batch_size"]),
                training_contract=contract,
                seed=stable_seed(manifest_seed(self.manifest, "residual_init"), "final", final_steps),
                class_weights=final_weights,
                optimizer=optimizer,
                generator=batch_generator,
            )
            final_losses.extend(block_result["losses"])
            finite_gradient_names.update(block_result["finite_gradient_names"])
            nonzero_gradient_names.update(block_result["nonzero_gradient_names"])
            final_steps += block
            residual_policy.eval()
            measured = evaluate_training_rows(
                residual_policy,
                combined_rows,
                batch_size=int(training["batch_size"]),
            )
            measured["sample_replay_ratio_deviation"] = float(
                dagger.sample_replay_ratio_deviation
            )
            hard_gate_checks = gate_b_hard_checks(
                measured, self.manifest["gates"]["b"]
            )
            hard_gate_checks["sample_replay_ratio_deviation"] = (
                measured["sample_replay_ratio_deviation"]
                <= float(
                    self.manifest["gates"]["b"][
                        "sample_replay_ratio_deviation_max"
                    ]
                )
            )
            current_trainable = _named_tensor_hashes(
                residual_policy, trainable=True
            )
            hard_gate_checks["trainable_scope_ready"] = (
                set(current_trainable) == expected_trainable_keys
                and finite_gradient_names == set(expected_trainable_names)
                and nonzero_gradient_names == set(expected_trainable_names)
                and all(
                    trainable_before[key] != current_trainable[key]
                    for key in expected_trainable_keys
                )
            )
            passed = all(hard_gate_checks.values())
            consecutive = consecutive + 1 if passed else 0
            evaluation_history.append(
                {
                    "optimizer_step": final_steps,
                    **measured,
                    "passed_thresholds": bool(passed),
                    "hard_gate_checks": dict(hard_gate_checks),
                    "consecutive_passes": consecutive,
                }
            )
            self._emit_progress(
                "gate_b",
                "metric_check_complete",
                optimizer_step=final_steps,
                passed_thresholds=bool(passed),
                consecutive_passes=consecutive,
                token_top1_accuracy=float(measured["token_top1_accuracy"]),
                stop_token_recall=float(measured["stop_token_recall"]),
                stop_intent_recall=float(measured["stop_intent_recall"]),
                navigate_intent_recall=float(measured["navigate_intent_recall"]),
                waypoint_mean_l2_m=float(measured["waypoint_mean_l2_m"]),
                sample_replay_ratio_deviation=float(
                    measured["sample_replay_ratio_deviation"]
                ),
                trainable_scope_ready=bool(
                    hard_gate_checks["trainable_scope_ready"]
                ),
            )
            if consecutive >= required_consecutive:
                break
            residual_policy.train()
        residual_policy.eval()
        if not evaluation_history:
            raise ProtocolError("Gate B performed no full unique-set evaluation")
        measured = dict(evaluation_history[-1])

        backbone_after = frozen_backbone_full_sha256(base_policy)
        residual_after = residual_sha256(residual_policy)
        trainable_after = _named_tensor_hashes(residual_policy, trainable=True)
        if set(trainable_after) != expected_trainable_keys:
            raise ProtocolError("P8 post-training parameter scope differs from the frozen inventory")
        undeclared_after = _named_tensor_hashes(residual_policy, trainable=False)
        value_head_after = _named_tensor_hashes(
            residual_policy.value_head, trainable=False
        )
        trainable_group_changes = trainable_group_change_audit(
            trainable_before, trainable_after
        )
        required_trainable_groups_changed = all(
            bool(item["all_changed"]) for item in trainable_group_changes.values()
        )
        parameter_change_checks = {
            key.removeprefix("parameter:"): trainable_before[key] != trainable_after[key]
            for key in sorted(expected_trainable_keys)
        }
        all_expected_trainables_changed = all(parameter_change_checks.values())
        trainable_changed = all_expected_trainables_changed
        value_head_frozen_unchanged = (
            value_head_before == value_head_after
            and not any(
                parameter.requires_grad
                for parameter in residual_policy.value_head.parameters()
            )
        )
        undeclared_unchanged = undeclared_before == undeclared_after
        name_by_id = {id(parameter): name for name, parameter in named_parameters.items()}
        optimizer_parameter_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        optimizer_scope = optimizer_parameter_ids == {
            id(named_parameters[name]) for name in expected_trainable_names
        }
        optimizer_parameter_names = sorted(
            name_by_id.get(parameter_id, "<undeclared>")
            for parameter_id in optimizer_parameter_ids
        )
        gradient_scope_exact = (
            finite_gradient_names == set(expected_trainable_names)
            and nonzero_gradient_names == set(expected_trainable_names)
        )
        dagger_trace_complete = all(
            trace.teacher_session_isolated
            and 1 <= len(trace.executed_actions) <= len(trace.sampled_actions)
            and trace.executed_actions
            == trace.sampled_actions[: len(trace.executed_actions)]
            and len(trace.behavior_log_probs)
            == len(trace.replay_log_probs)
            == len(trace.executed_actions)
            and math.isfinite(trace.replay_ratio_deviation)
            for trace in dagger.traces
        )
        all_losses = warmup_result["losses"] + final_losses
        all_finite = bool(measured["all_finite"]) and all(
            math.isfinite(float(value)) for value in all_losses
        )
        final_hard_gate_checks = dict(measured["hard_gate_checks"])
        final_hard_gate_checks.update(
            {
                "consecutive_passes": consecutive >= required_consecutive,
                "required_trainable_groups_changed": required_trainable_groups_changed,
                "all_expected_trainables_changed": all_expected_trainables_changed,
                "gradient_scope_exact": gradient_scope_exact,
                "value_head_frozen_unchanged": value_head_frozen_unchanged,
                "undeclared_tensors_unchanged": undeclared_unchanged,
                "frozen_backbone_hash_unchanged": backbone_before == backbone_after,
                "optimizer_scope_exact": optimizer_scope,
                "dagger_trace_fields_complete": dagger_trace_complete,
                "all_training_values_finite": all_finite,
            }
        )
        self._gate_b_passed = all(final_hard_gate_checks.values())
        self._gate_b_checkpoint_hash = residual_after
        return {
            "token_top1_accuracy": float(measured["token_top1_accuracy"]),
            "action_top1_accuracy": float(measured["token_top1_accuracy"]),
            "navigate_intent_positive_count": int(
                measured["navigate_intent_positive_count"]
            ),
            "navigate_intent_recall": float(measured["navigate_intent_recall"]),
            "stop_intent_positive_count": int(measured["stop_intent_positive_count"]),
            "stop_intent_recall": float(measured["stop_intent_recall"]),
            "stop_token_positive_count": int(measured["stop_token_positive_count"]),
            "stop_token_recall": float(measured["stop_token_recall"]),
            "waypoint_mean_l2_m": float(measured["waypoint_mean_l2_m"]),
            "sample_replay_ratio_deviation": float(
                measured["sample_replay_ratio_deviation"]
            ),
            "sample_replay_scope": "live_dagger_sample_vs_batch_size_one_replay_same_checkpoint",
            "consecutive_passes": consecutive,
            "recover_positive_count": int(measured["recover_positive_count"]),
            "recover_recall": measured["recover_recall"],
            "trainable_tensors_changed": trainable_changed,
            "required_trainable_groups_changed": (
                required_trainable_groups_changed
            ),
            "trainable_group_changes": trainable_group_changes,
            "expected_trainable_parameter_names": list(expected_trainable_names),
            "optimizer_parameter_names": optimizer_parameter_names,
            "finite_gradient_names": sorted(finite_gradient_names),
            "nonzero_gradient_names": sorted(nonzero_gradient_names),
            "parameter_change_checks": parameter_change_checks,
            "all_expected_trainables_changed": all_expected_trainables_changed,
            "gradient_scope_exact": gradient_scope_exact,
            "value_head_frozen_unchanged": value_head_frozen_unchanged,
            "hard_gate_checks": final_hard_gate_checks,
            "hard_gate_passed": bool(self._gate_b_passed),
            "undeclared_tensors_unchanged": undeclared_unchanged and optimizer_scope,
            "optimizer_scope_exact": optimizer_scope,
            "frozen_backbone_hash_unchanged": backbone_before == backbone_after,
            "backbone_hash_scope": "complete_named_parameters_and_buffers",
            "backbone_full_hash_before": backbone_before,
            "backbone_full_hash_after": backbone_after,
            "residual_full_hash_before": residual_before,
            "residual_full_hash_after": residual_after,
            "all_finite": all_finite,
            "trace_fields_complete": bool(measured["trace_fields_complete"])
            and dagger_trace_complete,
            "trace_diagnostics": measured["trace_diagnostics"],
            "expert_unique_state_count": len(expert_rows),
            "dagger_unique_state_count": len(dagger.rows),
            "combined_unique_state_count": len(combined_rows),
            "warmup_optimizer_steps": int(warmup_result["optimizer_steps"]),
            "final_optimizer_steps": final_steps,
            "class_weight_hashes": {
                "warmup": warmup_weights.dataset_hash,
                "final": final_weights.dataset_hash,
            },
            "training_contract": {
                "loss_coefficients": dict(contract["loss_coefficients"]),
                "smooth_l1_beta": float(contract["smooth_l1_beta"]),
                "smooth_l1_reduction": contract["smooth_l1_reduction"],
                "adamw_betas": list(contract["adamw_betas"]),
                "adamw_eps": float(contract["adamw_eps"]),
                "adamw_amsgrad": bool(contract["adamw_amsgrad"]),
                "gradient_clip": contract["gradient_clip"],
                "scheduler": contract["scheduler"],
                "sampling_method": contract["sampling_method"],
                "sampling_replacement": contract["sampling_replacement"],
                "class_weight_method": contract["class_weight_method"],
                "class_weight_normalization": contract[
                    "class_weight_normalization"
                ],
            },
            "evaluation_history": evaluation_history,
            "dagger_trace_count": len(dagger.traces),
            "split_read_counts": {
                "train": self.split_read_counts[ALLOWED_SPLIT] - start_reads,
                "val_seen": 0,
                "val_unseen": 0,
            },
        }

    @torch.inference_mode()
    def _rollout_development_episode(
        self,
        spec: Mapping,
        *,
        learned: bool,
    ) -> DevelopmentEpisode:
        base_policy, residual_policy = self._load_once()
        max_steps = int(self.manifest["runtime"]["max_primitive_steps"])
        max_decisions = max_steps // ACTION_CHUNK
        session = self._session(base_policy)
        memory = CausalTopologyMemory(
            max_decisions=max_decisions,
            node_hidden_dim=int(residual_policy.topology_encoder.node_hidden_dim),
            node_feature_dim=int(residual_policy.topology_encoder.node_feature_dim),
            edge_feature_dim=int(residual_policy.topology_encoder.edge_feature_dim),
        )
        env = self._train_env(spec, max_steps=max_steps)
        primitive_counts = {name: 0 for name in ACTION_NAMES}
        mode_counts = {name: 0 for name in INTENT_NAMES}
        intent_mode_counts = {name: 0 for name in INTENT_NAMES}
        blocked_state_count = 0
        recover_trigger_count = 0
        recover_alignment_miss_count = 0
        recover_outcomes = []
        stopped = False
        last_info = None
        terminated = truncated = False
        try:
            obs, identity = env.reset()
            _validate_identity(identity, spec)
            scene, episode, instruction = base_policy._identity(obs)
            session.begin_task(scene + "::" + episode, instruction)
            for decision_index in range(max_decisions):
                step = len(session.actions)
                feature = self._feature(base_policy, session, obs)
                memory.observe(feature)
                topology = memory.tensors(
                    device=next(residual_policy.parameters()).device
                )
                actionable_blocked = (
                    memory.online.consecutive_blocked_forwards
                    >= self.training_contract["blocked_forward_threshold"]
                    and memory.online.nodes[memory.online.current_node].parent is not None
                )
                if actionable_blocked:
                    blocked_state_count += 1
                    recovery_actions = compile_waypoint(
                        (0.0, 0.0), RECOVER_MODE, memory.online
                    )
                else:
                    recovery_actions = None
                session.open_decision(step, feature)
                generator = torch.Generator(device=session.device).manual_seed(
                    stable_seed(
                        manifest_seed(self.manifest, "residual_init"), "gate-c", spec["episode_id"], step
                    )
                )
                callback = (
                    residual_policy.decision_residual_callback(topology)
                    if learned
                    else None
                )
                (
                    _,
                    actions,
                    _,
                    _,
                    hidden_rows,
                    _,
                ) = self._sample_open_decision(session, generator, callback)
                if learned:
                    condition = residual_policy.encode_condition(hidden_rows[0], topology)
                    predicted_intent = int(
                        condition["intent_logits"].argmax(dim=-1).item()
                    )
                    intent_mode_counts[INTENT_NAMES[predicted_intent]] += 1
                else:
                    predicted_intent = NAVIGATE_INTENT
                padded = torch.zeros((1, ACTION_CHUNK, 1), dtype=torch.int64)
                padded[0, : len(actions), 0] = torch.tensor(actions)
                obs_list, _, terminations, truncations, infos = env.chunk_step(
                    padded, execute_count=len(actions)
                )
                last_info = infos[0]
                raw_feedback = last_info.get("executor_feedback")
                if not isinstance(raw_feedback, Mapping):
                    raise ProtocolError("Gate C lacks executor feedback")
                feedback = feedback_from_dict(raw_feedback)
                executed = tuple(int(value) for value in feedback.executed_actions)
                for action in executed:
                    primitive_counts[ACTION_NAMES[action]] += 1
                executed_mode = classify_executed_mode(
                    executed,
                    predicted_intent=predicted_intent,
                    actionable_blocked=actionable_blocked,
                    recovery_actions=recovery_actions,
                )
                if learned:
                    mode_counts[INTENT_NAMES[executed_mode]] += 1
                    if executed_mode == RECOVER_INTENT:
                        recover_trigger_count += 1
                    elif actionable_blocked and predicted_intent == RECOVER_INTENT:
                        recover_alignment_miss_count += 1
                stopped = stopped or 0 in executed
                blocked_before_update = int(memory.online.consecutive_blocked_forwards)
                node_before_update = int(memory.online.current_node)
                memory.update_from_feedback(raw_feedback)
                if learned and executed_mode == RECOVER_INTENT:
                    recover_outcomes.append(
                        {
                            "episode_id": str(spec["episode_id"]),
                            "decision_index": decision_index,
                            "actionable_blocked_before": bool(actionable_blocked),
                            "consecutive_blocked_before": blocked_before_update,
                            "consecutive_blocked_after": int(
                                memory.online.consecutive_blocked_forwards
                            ),
                            "node_before": node_before_update,
                            "node_after": int(memory.online.current_node),
                            "realized_path_m": float(feedback.realized_path_m),
                        }
                    )
                session.acknowledge(actions, executed)
                obs = obs_list[0]
                terminated = bool(terminations.any())
                truncated = bool(truncations.any())
                if terminated or truncated:
                    break
            if last_info is None or "episode" not in last_info or not (terminated or truncated):
                raise ProtocolError("Gate C episode lacks terminal evidence")
            terminal_metrics = _canonical_metrics(last_info["episode"])
            return DevelopmentEpisode(
                success=bool(terminal_metrics.get("success", 0.0) > 0.0),
                stopped=bool(stopped),
                primitive_counts=primitive_counts,
                mode_counts=mode_counts,
                intent_mode_counts=intent_mode_counts,
                blocked_state_count=blocked_state_count,
                recover_trigger_count=recover_trigger_count,
                recover_alignment_miss_count=recover_alignment_miss_count,
                recover_outcomes=tuple(recover_outcomes),
                terminal_metrics=terminal_metrics,
            )
        finally:
            env.close()

    def gate_c_provider(self) -> dict:
        if not self._gate_b_passed or not self._gate_b_checkpoint_hash:
            raise ProtocolError("live Gate C cannot run before a passing Gate B checkpoint")
        base_policy, residual_policy = self._load_once()
        if residual_sha256(residual_policy) != self._gate_b_checkpoint_hash:
            raise ProtocolError("Gate B residual changed before Gate C")
        specs = self.manifest["episodes"]["train_development"]
        if len(specs) != 8 or len({item["scene"] for item in specs}) != 8:
            raise ProtocolError("Gate C requires eight scene-disjoint train identities")
        start_reads = self.split_read_counts[ALLOWED_SPLIT]
        before = parameter_fingerprint(base_policy, residual_policy)
        checkpoint_before = hashlib.sha256(
            json.dumps(before, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest()
        keep_episodes = []
        learned_episodes = []
        residual_policy.eval()
        for episode_index, spec in enumerate(specs, start=1):
            keep_episodes.append(
                self._rollout_development_episode(spec, learned=False)
            )
            learned_episodes.append(
                self._rollout_development_episode(spec, learned=True)
            )
            self._emit_progress(
                "gate_c",
                "matched_episode_pair_complete",
                episode_index=episode_index,
                episode_count=len(specs),
                episode_id=str(spec["episode_id"]),
                keep_original_success=bool(keep_episodes[-1].success),
                learned_success=bool(learned_episodes[-1].success),
                learned_stopped=bool(learned_episodes[-1].stopped),
            )
        after = parameter_fingerprint(base_policy, residual_policy)
        checkpoint_after = hashlib.sha256(
            json.dumps(after, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest()
        primitive_counts = {name: 0 for name in ACTION_NAMES}
        mode_counts = {name: 0 for name in INTENT_NAMES}
        intent_mode_counts = {name: 0 for name in INTENT_NAMES}
        recover_outcomes = []
        for episode in learned_episodes:
            for name, value in episode.primitive_counts.items():
                primitive_counts[name] += int(value)
            for name, value in episode.mode_counts.items():
                mode_counts[name] += int(value)
            for name, value in episode.intent_mode_counts.items():
                intent_mode_counts[name] += int(value)
            recover_outcomes.extend(episode.recover_outcomes)
        return {
            "episode_count": len(learned_episodes),
            "learned_success_count": sum(episode.success for episode in learned_episodes),
            "keep_original_success_count": sum(episode.success for episode in keep_episodes),
            "successful_stop_count": sum(
                episode.success and episode.stopped for episode in learned_episodes
            ),
            "stop_count": primitive_counts["stop"],
            "primitive_counts": primitive_counts,
            "mode_counts": mode_counts,
            "intent_mode_counts": intent_mode_counts,
            "left_turn_count": primitive_counts["left"],
            "right_turn_count": primitive_counts["right"],
            "blocked_state_count": sum(
                episode.blocked_state_count for episode in learned_episodes
            ),
            "recover_trigger_count": sum(
                episode.recover_trigger_count for episode in learned_episodes
            ),
            "recover_alignment_miss_count": sum(
                episode.recover_alignment_miss_count
                for episode in learned_episodes
            ),
            "recover_outcomes": recover_outcomes,
            "checkpoint_hash_before": checkpoint_before,
            "checkpoint_hash_after": checkpoint_after,
            "checkpoint_hash_scope": {
                "backbone": "complete_named_parameters_and_buffers",
                "residual": "complete_state_dict",
            },
            "backbone_full_hash_before": before["backbone_full_sha256"],
            "backbone_full_hash_after": after["backbone_full_sha256"],
            "residual_full_hash_before": before["residual_full_sha256"],
            "residual_full_hash_after": after["residual_full_sha256"],
            "episode_metrics": {
                "keep_original": [episode.terminal_metrics for episode in keep_episodes],
                "learned_residual": [episode.terminal_metrics for episode in learned_episodes],
            },
            "split_read_counts": {
                "train": self.split_read_counts[ALLOWED_SPLIT] - start_reads,
                "val_seen": 0,
                "val_unseen": 0,
            },
        }


@torch.inference_mode()
def collect_expert_rows(
    *,
    providers: LiveP8DiagnosticProviders,
    spec: Mapping,
    max_decisions: int,
) -> list[P8TrainingRow]:
    """Collect one real train-only expert trajectory with unique state rows."""

    base_policy, residual_policy = providers._load_once()
    session = providers._session(base_policy)
    memory = CausalTopologyMemory(
        max_decisions=max_decisions,
        node_hidden_dim=int(residual_policy.topology_encoder.node_hidden_dim),
        node_feature_dim=int(residual_policy.topology_encoder.node_feature_dim),
        edge_feature_dim=int(residual_policy.topology_encoder.edge_feature_dim),
    )
    env = providers._train_env(spec, max_steps=max_decisions * ACTION_CHUNK)
    rows = []
    try:
        obs, identity = env.reset()
        _validate_identity(identity, spec)
        scene, episode, instruction = base_policy._identity(obs)
        session.begin_task(scene + "::" + episode, instruction)
        for decision_index in range(max_decisions):
            feature = providers._feature(base_policy, session, obs)
            memory.observe(feature)
            topology = memory.tensors(device="cpu")
            label = env.expert_waypoint(providers.training_contract["expert_lookahead_m"])
            target_intent = STOP_INTENT if bool(label["stop"]) else NAVIGATE_INTENT
            target_actions = (
                [0]
                if target_intent == STOP_INTENT
                else compile_waypoint(label["waypoint_local_m"], NAVIGATE_MODE, memory.online)
            )
            state_hash = state_sha256(
                episode_id=str(spec["episode_id"]),
                scene=str(spec["scene"]),
                decision_index=decision_index,
                feature=feature,
                topology=topology,
            )
            session.open_decision(len(session.actions), feature)
            row = _teacher_force_open_decision(
                session,
                target_actions=target_actions,
                generator=torch.Generator(device=session.device).manual_seed(
                    stable_seed(manifest_seed(providers.manifest, "residual_init"), "expert", state_hash)
                ),
                state_hash=state_hash,
                topology=topology,
                target_intent=target_intent,
                target_waypoint=label["waypoint_local_m"],
            )
            padded = torch.zeros((1, ACTION_CHUNK, 1), dtype=torch.int64)
            padded[0, : len(target_actions), 0] = torch.tensor(target_actions)
            obs_list, _, terminations, truncations, infos = env.chunk_step(
                padded, execute_count=len(target_actions)
            )
            raw_feedback = infos[0].get("executor_feedback")
            if not isinstance(raw_feedback, Mapping):
                raise ProtocolError("expert collection lacks executor feedback")
            feedback = feedback_from_dict(raw_feedback)
            executed = [int(value) for value in feedback.executed_actions]
            if not executed or executed != target_actions[: len(executed)]:
                raise ProtocolError("expert executor did not preserve the teacher prefix")
            row.executed_mask = torch.tensor(feedback.executed_mask, dtype=torch.bool)
            duration = int(row.executed_mask.sum())
            if duration != len(executed) or duration > int(row.token_hidden.shape[0]):
                raise ProtocolError("expert cache differs from the real executed prefix")
            row.token_hidden = row.token_hidden[:duration]
            row.base_logits = row.base_logits[:duration]
            row.actions = torch.zeros(ACTION_CHUNK, dtype=torch.int64)
            row.actions[:duration] = torch.tensor(executed, dtype=torch.int64)
            row.target_signature = (
                tuple(executed),
                int(row.target_intent),
                tuple(row.target_waypoint.tolist()),
            )
            memory.update_from_feedback(raw_feedback)
            session.acknowledge(target_actions, executed)
            rows.append(row)
            obs = obs_list[0]
            if bool(terminations.any()) or bool(truncations.any()):
                break
        return deduplicate_state_records(rows)
    finally:
        env.close()


@torch.inference_mode()
def collect_bounded_dagger_rows(
    *,
    providers: LiveP8DiagnosticProviders,
    specs: Sequence[Mapping],
    max_decisions_per_episode: int,
) -> DAggerCollection:
    """Execute learned histories while caching expert prefixes in isolated sessions."""

    if max_decisions_per_episode < 1:
        raise ProtocolError("bounded DAgger requires a positive decision ceiling")
    base_policy, residual_policy = providers._load_once()
    rows: list[P8TrainingRow] = []
    traces: list[DAggerDecisionTrace] = []
    for spec in specs:
        session = providers._session(base_policy)
        memory = CausalTopologyMemory(
            max_decisions=max_decisions_per_episode,
            node_hidden_dim=int(residual_policy.topology_encoder.node_hidden_dim),
            node_feature_dim=int(residual_policy.topology_encoder.node_feature_dim),
            edge_feature_dim=int(residual_policy.topology_encoder.edge_feature_dim),
        )
        env = providers._train_env(
            spec, max_steps=max_decisions_per_episode * ACTION_CHUNK
        )
        try:
            obs, identity = env.reset()
            _validate_identity(identity, spec)
            scene, episode, instruction = base_policy._identity(obs)
            session.begin_task(scene + "::" + episode, instruction)
            for decision_index in range(max_decisions_per_episode):
                step = len(session.actions)
                feature = providers._feature(base_policy, session, obs)
                memory.observe(feature)
                topology_cpu = memory.tensors(device="cpu")
                label = env.expert_waypoint(providers.training_contract["expert_lookahead_m"])
                blocked_before = int(memory.online.consecutive_blocked_forwards)
                parent_available = (
                    memory.online.nodes[memory.online.current_node].parent is not None
                )
                if (
                    blocked_before
                    >= providers.training_contract["blocked_forward_threshold"]
                    and parent_available
                ):
                    target_intent = RECOVER_INTENT
                    target_waypoint = memory.online.backtrack_waypoint()
                    if target_waypoint is None:
                        raise ProtocolError("RECOVER label lost its causal parent waypoint")
                    target_actions = compile_waypoint(
                        target_waypoint, RECOVER_MODE, memory.online
                    )
                elif bool(label["stop"]):
                    target_intent = STOP_INTENT
                    target_waypoint = (0.0, 0.0)
                    target_actions = [0]
                else:
                    target_intent = NAVIGATE_INTENT
                    target_waypoint = label["waypoint_local_m"]
                    target_actions = compile_waypoint(
                        target_waypoint, NAVIGATE_MODE, memory.online
                    )
                state_hash = state_sha256(
                    episode_id=str(spec["episode_id"]),
                    scene=str(spec["scene"]),
                    decision_index=decision_index,
                    feature=feature,
                    topology=topology_cpu,
                )

                live_before = _session_state_signature(session)
                teacher = clone_teacher_session(session)
                teacher.open_decision(step, feature)
                teacher_row = _teacher_force_open_decision(
                    teacher,
                    target_actions=target_actions,
                    generator=torch.Generator(device=teacher.device).manual_seed(
                        stable_seed(
                            manifest_seed(providers.manifest, "residual_init"), "dagger-teacher", state_hash
                        )
                    ),
                    state_hash=state_hash,
                    topology=topology_cpu,
                    target_intent=target_intent,
                    target_waypoint=target_waypoint,
                )
                if _session_state_signature(session) != live_before:
                    raise ProtocolError("teacher-forced cache contaminated the live session")

                topology_device = topology_cpu.canonical(
                    device=next(residual_policy.parameters()).device
                )
                session.open_decision(step, feature)
                generator = torch.Generator(device=session.device).manual_seed(
                    stable_seed(
                        manifest_seed(providers.manifest, "residual_init"), "dagger-sample", spec["episode_id"], step
                    )
                )
                callback = residual_policy.decision_residual_callback(topology_device)
                (
                    tokens,
                    sampled_actions,
                    base_logits,
                    _,
                    hidden_rows,
                    behavior_log_probs,
                ) = providers._sample_open_decision(session, generator, callback)
                padded = torch.zeros((1, ACTION_CHUNK, 1), dtype=torch.int64)
                padded[0, : len(sampled_actions), 0] = torch.tensor(sampled_actions)
                obs_list, _, terminations, truncations, infos = env.chunk_step(
                    padded, execute_count=len(sampled_actions)
                )
                raw_feedback = infos[0].get("executor_feedback")
                if not isinstance(raw_feedback, Mapping):
                    raise ProtocolError("DAgger collection lacks executor feedback")
                feedback = feedback_from_dict(raw_feedback)
                executed = tuple(int(value) for value in feedback.executed_actions)
                duration = len(executed)
                if not 1 <= duration <= len(sampled_actions):
                    raise ProtocolError("DAgger executor returned an invalid prefix length")
                if executed != tuple(sampled_actions[:duration]):
                    raise ProtocolError("DAgger executor did not preserve the sampled prefix")
                expected_mask = tuple(
                    index < duration for index in range(ACTION_CHUNK)
                )
                if tuple(bool(value) for value in feedback.executed_mask) != expected_mask:
                    raise ProtocolError("DAgger executor mask is not the actual contiguous prefix")
                replay_outputs = residual_policy(
                    torch.stack(hidden_rows[:duration]).unsqueeze(0),
                    torch.stack(base_logits[:duration]).unsqueeze(0),
                    topology_device,
                    actions=torch.tensor(
                        executed, dtype=torch.int64, device=session.device
                    ).unsqueeze(0),
                    executed_mask=torch.ones(
                        (1, duration), dtype=torch.bool, device=session.device
                    ),
                )
                replay_log_probs = replay_outputs["token_log_probs"][0].cpu().tolist()
                behavior_prefix = behavior_log_probs[:duration]
                replay_ratio = abs(
                    math.exp(sum(replay_log_probs) - sum(behavior_prefix)) - 1.0
                )
                memory.update_from_feedback(raw_feedback)
                session.acknowledge(sampled_actions, executed)
                rows.append(teacher_row)
                traces.append(
                    DAggerDecisionTrace(
                        episode_id=str(spec["episode_id"]),
                        decision_index=decision_index,
                        state_hash=state_hash,
                        sampled_actions=tuple(sampled_actions),
                        executed_actions=executed,
                        expert_actions=tuple(target_actions),
                        target_intent=int(target_intent),
                        behavior_log_probs=tuple(float(value) for value in behavior_prefix),
                        replay_log_probs=tuple(float(value) for value in replay_log_probs),
                        replay_ratio_deviation=float(replay_ratio),
                        blocked_before=blocked_before,
                        parent_available=bool(parent_available),
                        teacher_session_isolated=True,
                    )
                )
                obs = obs_list[0]
                if bool(terminations.any()) or bool(truncations.any()):
                    break
        finally:
            env.close()
    unique_rows = tuple(deduplicate_state_records(rows))
    if not unique_rows or not traces:
        raise ProtocolError("bounded DAgger produced no real on-policy states")
    return DAggerCollection(
        rows=unique_rows,
        traces=tuple(traces),
        sample_replay_ratio_deviation=max(
            trace.replay_ratio_deviation for trace in traces
        ),
    )


def build_live_gate_providers(**kwargs):
    """Return one shared provider object and its runner-compatible callables."""

    providers = LiveP8DiagnosticProviders(**kwargs)
    return providers, providers.gate_a_provider, providers.gate_b_provider, providers.gate_c_provider
