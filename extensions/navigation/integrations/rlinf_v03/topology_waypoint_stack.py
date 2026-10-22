"""Reusable topology-choice plus metric-waypoint plugin for ``navrl.stack``.

The policy sees only RGB, instruction tokens, and a causal graph assembled from
executed primitive feedback.  A categorical decision chooses a frontier,
known parent node, or STOP.  Non-STOP choices additionally sample a bounded
two-dimensional residual around the candidate anchor.  PPO evaluates the raw
Gaussian residual, while the deterministic anchor/residual transform produces
the metric waypoint sent to the executor.

This module deliberately lives in the integration layer.  It does not modify
the model- and simulator-independent stack core.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from navrl.stack import (
    ActionExecutor,
    BootstrapValue,
    CheckpointAdapter,
    DecisionBatch,
    EpisodeIdentity,
    EpisodeMetrics,
    ExecutionReceipt,
    PolicyAdapter,
    PolicyEvaluation,
    PolicySample,
    PrefixActionExecutor,
    PrimitiveStep,
    TaskAdapter,
    TaskReset,
)

try:
    from .streamvln_p8_successor_action import (
        STOP_ACTION,
        WaypointCompilerConfig,
        compile_metric_waypoint,
    )
    from .streamvln_p8_topology import (
        P8CausalTopologySnapshot,
        P8TopologyEncoder,
        P8TopologyTensors,
        snapshot_online_topology,
    )
    from .streamvln_topology import (
        CAUSAL_EDGE_FEATURE_DIM,
        CAUSAL_GLOBAL_FEATURE_NAMES,
        CAUSAL_NODE_FEATURE_DIM,
        FORWARD_ACTION,
        OnlineTopology,
    )
except ImportError:  # Direct staged-file mode used by bounded remote smokes.
    from streamvln_p8_successor_action import (
        STOP_ACTION,
        WaypointCompilerConfig,
        compile_metric_waypoint,
    )
    from streamvln_p8_topology import (
        P8CausalTopologySnapshot,
        P8TopologyEncoder,
        P8TopologyTensors,
        snapshot_online_topology,
    )
    from streamvln_topology import (
        CAUSAL_EDGE_FEATURE_DIM,
        CAUSAL_GLOBAL_FEATURE_NAMES,
        CAUSAL_NODE_FEATURE_DIM,
        FORWARD_ACTION,
        OnlineTopology,
    )


ACTION_SCHEMA = "topology_waypoint_v1"
CANDIDATE_KIND_FRONTIER = "frontier"
CANDIDATE_KIND_TOPOLOGY_NODE = "topology_node"
CANDIDATE_KIND_STOP = "stop"
STOP_CANDIDATE_ID = "stop"
CANDIDATE_KINDS = frozenset(
    (CANDIDATE_KIND_FRONTIER, CANDIDATE_KIND_TOPOLOGY_NODE, CANDIDATE_KIND_STOP)
)
OBSERVATION_KEYS = frozenset(("rgb", "instruction_ids", "instruction_mask"))
MAX_CANDIDATES = 5
CANDIDATE_FEATURE_DIM = 8


def _finite(value: object, name: str) -> float:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "reshape") and hasattr(value, "numel"):
        if int(value.numel()) != 1:
            raise ValueError(f"{name} must be scalar")
        value = value.reshape(-1)[0].item()
    elif hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _boolean(value: object, name: str) -> bool:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "reshape") and hasattr(value, "numel"):
        if int(value.numel()) != 1:
            raise ValueError(f"{name} must be scalar")
        value = value.reshape(-1)[0].item()
    elif hasattr(value, "item"):
        value = value.item()
    if type(value) is not bool:
        raise TypeError(f"{name} must be boolean")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class TopologyCandidate:
    """One stable high-level option in the current causal map snapshot."""

    candidate_id: str
    kind: str
    forward_m: float
    lateral_m: float
    valid: bool = True
    continuous: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise ValueError("candidate_id must be a non-empty string")
        if self.kind not in CANDIDATE_KINDS:
            raise ValueError("unknown topology candidate kind")
        if type(self.valid) is not bool or type(self.continuous) is not bool:
            raise TypeError("candidate valid/continuous flags must be bools")
        forward = _finite(self.forward_m, "candidate forward_m")
        lateral = _finite(self.lateral_m, "candidate lateral_m")
        if self.kind == CANDIDATE_KIND_STOP:
            if self.candidate_id != STOP_CANDIDATE_ID:
                raise ValueError("STOP candidate id is fixed")
            if self.continuous or not self.valid or (forward, lateral) != (0.0, 0.0):
                raise ValueError("STOP must be valid, discrete, and geometry-free")
        elif not self.continuous:
            raise ValueError("non-STOP candidates require a continuous residual")
        object.__setattr__(self, "forward_m", forward)
        object.__setattr__(self, "lateral_m", lateral)

    def canonical(self) -> tuple[object, ...]:
        return (
            self.candidate_id,
            self.kind,
            self.forward_m,
            self.lateral_m,
            self.valid,
            self.continuous,
        )

    @classmethod
    def from_canonical(cls, row: Sequence[object]) -> "TopologyCandidate":
        if not isinstance(row, (tuple, list)) or len(row) != 6:
            raise ValueError("candidate canonical row has the wrong schema")
        return cls(*row)


def candidate_digest(candidates: Sequence[TopologyCandidate]) -> str:
    if len(candidates) != MAX_CANDIDATES:
        raise ValueError("candidate snapshot must have fixed width five")
    document = [candidate.canonical() for candidate in candidates]
    payload = json.dumps(
        document, ensure_ascii=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tensor_context_digest(forward_inputs: Mapping[str, object]) -> str:
    """Bind replay to the exact detached observation/topology/candidate tensors."""

    digest = hashlib.sha256()
    for name in sorted(forward_inputs):
        value = forward_inputs[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError("forward context digest accepts tensors only")
        tensor = value.detach().cpu().contiguous()
        header = json.dumps(
            {
                "name": name,
                "dtype": str(tensor.dtype),
                "shape": [int(item) for item in tensor.shape],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        raw = tensor.numpy().tobytes(order="C")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


@dataclass(frozen=True)
class TopologyDecisionSnapshot:
    identity: EpisodeIdentity
    decision_index: int
    physical_step: int
    candidates: tuple[TopologyCandidate, ...]
    digest: str
    topology: P8CausalTopologySnapshot

    def __post_init__(self) -> None:
        if not isinstance(self.identity, EpisodeIdentity):
            raise TypeError("snapshot identity must be EpisodeIdentity")
        if len(self.candidates) != MAX_CANDIDATES:
            raise ValueError("snapshot candidate width drifted")
        if self.digest != candidate_digest(self.candidates):
            raise ValueError("snapshot candidate digest differs from its payload")
        if self.topology.episode_token != self.identity.reset_token:
            raise ValueError("topology snapshot belongs to another reset")
        if (
            self.topology.decision_index != self.decision_index
            or self.topology.physical_step != self.physical_step
        ):
            raise ValueError("topology snapshot provenance differs")


class EpisodeTopologyState:
    """Episode-bound causal topology shared by the policy and executor."""

    def __init__(
        self,
        *,
        max_decisions: int = 8,
        topology_max_nodes: int = 16,
        compiler_config: WaypointCompilerConfig = WaypointCompilerConfig(),
        frontier_distance_m: float = 0.75,
        frontier_lateral_m: float = 0.5,
        residual_limit_m: float = 0.2,
    ) -> None:
        if type(max_decisions) is not int or max_decisions < 1:
            raise ValueError("max_decisions must be a positive integer")
        if type(topology_max_nodes) is not int or topology_max_nodes < 2:
            raise ValueError("topology_max_nodes must be at least two")
        for value, name in (
            (frontier_distance_m, "frontier_distance_m"),
            (frontier_lateral_m, "frontier_lateral_m"),
            (residual_limit_m, "residual_limit_m"),
        ):
            if _finite(value, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        safety_floor = compiler_config.arrival_tolerance_m + math.sqrt(2.0) * float(
            residual_limit_m
        )
        if float(frontier_distance_m) <= safety_floor:
            raise ValueError("frontier anchor is too close for the residual bound")
        self.max_decisions = max_decisions
        self.topology_max_nodes = topology_max_nodes
        self.compiler_config = compiler_config
        self.frontier_distance_m = float(frontier_distance_m)
        self.frontier_lateral_m = float(frontier_lateral_m)
        self.residual_limit_m = float(residual_limit_m)
        self.identity: EpisodeIdentity | None = None
        self.topology: OnlineTopology | None = None
        self._cumulative_path_m = 0.0
        self._cumulative_collisions = 0.0

    def bind(self, identity: EpisodeIdentity) -> None:
        if not isinstance(identity, EpisodeIdentity):
            raise TypeError("topology state identity must be EpisodeIdentity")
        self.identity = identity
        self.topology = OnlineTopology(
            max_decisions=self.max_decisions,
            forward_step_m=self.compiler_config.forward_step_m,
            turn_degrees=math.degrees(self.compiler_config.turn_radians),
        )
        self._cumulative_path_m = 0.0
        self._cumulative_collisions = 0.0

    def _require_identity(self, identity: EpisodeIdentity) -> OnlineTopology:
        if identity != self.identity or self.topology is None:
            raise RuntimeError("stale or unbound topology reset identity")
        return self.topology

    def _candidates(self, topology: OnlineTopology) -> tuple[TopologyCandidate, ...]:
        distance = self.frontier_distance_m
        lateral = self.frontier_lateral_m
        candidates = [
            TopologyCandidate("frontier:forward", CANDIDATE_KIND_FRONTIER, distance, 0.0),
            TopologyCandidate(
                "frontier:left", CANDIDATE_KIND_FRONTIER, distance, lateral
            ),
            TopologyCandidate(
                "frontier:right", CANDIDATE_KIND_FRONTIER, distance, -lateral
            ),
        ]
        parent = topology.nodes[topology.current_node].parent
        parent_waypoint = topology.backtrack_waypoint()
        parent_valid = False
        parent_id = "node:unavailable"
        parent_forward = parent_lateral = 0.0
        if parent is not None and parent_waypoint is not None:
            parent_forward, parent_lateral = parent_waypoint
            parent_distance = math.hypot(parent_forward, parent_lateral)
            parent_valid = parent_distance > (
                self.compiler_config.arrival_tolerance_m
                + math.sqrt(2.0) * self.residual_limit_m
            )
            parent_id = f"node:{int(parent)}"
        candidates.append(
            TopologyCandidate(
                parent_id,
                CANDIDATE_KIND_TOPOLOGY_NODE,
                parent_forward,
                parent_lateral,
                valid=parent_valid,
            )
        )
        candidates.append(
            TopologyCandidate(
                STOP_CANDIDATE_ID,
                CANDIDATE_KIND_STOP,
                0.0,
                0.0,
                valid=True,
                continuous=False,
            )
        )
        return tuple(candidates)

    def snapshot(
        self,
        identity: EpisodeIdentity,
        decision_index: int,
        physical_step: int,
    ) -> TopologyDecisionSnapshot:
        topology = self._require_identity(identity)
        if topology.decision_count != decision_index:
            raise RuntimeError("topology decision count differs from policy decision")
        if topology.primitive_count != physical_step:
            raise RuntimeError("topology primitive count differs from physical step")
        candidates = self._candidates(topology)
        graph = snapshot_online_topology(
            topology,
            episode_token=identity.reset_token,
            physical_step=physical_step,
            decision_index=decision_index,
            max_nodes=self.topology_max_nodes,
        )
        return TopologyDecisionSnapshot(
            identity,
            decision_index,
            physical_step,
            candidates,
            candidate_digest(candidates),
            graph,
        )

    def apply_receipt(self, receipt: ExecutionReceipt) -> None:
        topology = self._require_identity(receipt.identity)
        if receipt.decision_index != topology.decision_count:
            raise RuntimeError("execution receipt decision is stale for topology")
        if receipt.physical_step != topology.primitive_count:
            raise RuntimeError("execution receipt physical step is stale for topology")
        actions = tuple(int(action) for action in receipt.active_primitives)
        collisions: list[bool] = []
        realized: list[float] = []
        blocked: list[bool] = []
        for action, feedback in zip(actions, receipt.primitive_feedback):
            path_value = feedback.get("path_length")
            if path_value is not None:
                cumulative_path = _finite(path_value, "cumulative path_length")
                if cumulative_path + 1e-7 < self._cumulative_path_m:
                    raise RuntimeError("causal path_length moved backwards")
                increment = max(0.0, cumulative_path - self._cumulative_path_m)
                self._cumulative_path_m = cumulative_path
            elif "realized_path_m" in feedback:
                increment = _finite(feedback["realized_path_m"], "realized_path_m")
                if increment < 0.0:
                    raise ValueError("realized_path_m must be nonnegative")
                self._cumulative_path_m += increment
            else:
                increment = self.compiler_config.forward_step_m if action == FORWARD_ACTION else 0.0

            collision_value = feedback.get("collision")
            count_value = feedback.get("collision_count")
            if collision_value is not None:
                collision = _boolean(collision_value, "collision")
            elif count_value is not None:
                cumulative_collisions = _finite(count_value, "collision_count")
                if cumulative_collisions + 1e-7 < self._cumulative_collisions:
                    raise RuntimeError("causal collision_count moved backwards")
                collision = cumulative_collisions > self._cumulative_collisions + 0.5
            else:
                collision = False
            if count_value is not None:
                self._cumulative_collisions = _finite(count_value, "collision_count")
            elif collision:
                self._cumulative_collisions += 1.0

            is_blocked = bool(
                action == FORWARD_ACTION
                and (collision or increment < 0.5 * self.compiler_config.forward_step_m)
            )
            collisions.append(collision)
            realized.append(increment)
            blocked.append(is_blocked)
        topology.update(
            actions,
            collisions,
            realized_path_m=realized,
            blocked_forwards=blocked,
        )

    def compatibility(self) -> Mapping[str, object]:
        return {
            "max_decisions": self.max_decisions,
            "topology_max_nodes": self.topology_max_nodes,
            "frontier_distance_m": self.frontier_distance_m,
            "frontier_lateral_m": self.frontier_lateral_m,
            "residual_limit_m": self.residual_limit_m,
            "compiler": asdict(self.compiler_config),
            "candidate_schema": ACTION_SCHEMA,
            "candidate_order": [
                "frontier:forward",
                "frontier:left",
                "frontier:right",
                "node:<parent-or-unavailable>",
                STOP_CANDIDATE_ID,
            ],
        }


class _FrozenVLNEncoder(nn.Module):
    """Small RGB/instruction encoder kept frozen during the S2 PPO smoke."""

    output_dim = 128

    def __init__(self, *, vocab_size: int, max_instruction_tokens: int, image_size: int):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.max_instruction_tokens = int(max_instruction_tokens)
        self.image_size = int(image_size)
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.Tanh(),
        )
        self.token_embedding = nn.Embedding(self.vocab_size, 64)
        self.instruction_encoder = nn.GRU(64, 128, batch_first=True)
        self.fusion = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if set(obs) != OBSERVATION_KEYS:
            raise ValueError("topology policy observation keys drifted")
        rgb = obs["rgb"]
        ids = obs["instruction_ids"]
        mask = obs["instruction_mask"]
        if rgb.ndim != 4 or rgb.shape[-1] != 3 or rgb.dtype != torch.uint8:
            raise ValueError("rgb must be batched uint8 HWC")
        if (
            ids.ndim != 2
            or ids.shape[1] != self.max_instruction_tokens
            or ids.dtype != torch.int64
            or mask.dtype != torch.bool
            or mask.shape != ids.shape
            or rgb.shape[0] != ids.shape[0]
        ):
            raise ValueError("instruction tensor contract drifted")
        if bool((ids < 0).any()) or bool((ids >= self.vocab_size).any()):
            raise ValueError("instruction token is outside the vocabulary")
        lengths = mask.sum(dim=1)
        if not bool((lengths > 0).all()):
            raise ValueError("instruction must contain at least one real token")
        image = rgb.permute(0, 3, 1, 2).float().div(255.0)
        if image.shape[-2:] != (self.image_size, self.image_size):
            image = F.interpolate(
                image,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        visual = self.visual_encoder(image)
        encoded, _ = self.instruction_encoder(self.token_embedding(ids))
        rows = torch.arange(encoded.shape[0], device=encoded.device)
        language = encoded[rows, lengths - 1]
        return self.fusion(torch.cat((visual, language), dim=-1))


class TopologyWaypointPolicy(nn.Module):
    """Lightweight visual-language graph policy with a hybrid action density."""

    def __init__(
        self,
        *,
        vocab_size: int,
        max_instruction_tokens: int = 200,
        image_size: int = 64,
        hidden_dim: int = 128,
        topology_max_nodes: int = 16,
        topology_model_dim: int = 32,
        max_candidates: int = MAX_CANDIDATES,
        residual_limit_m: float = 0.2,
        initial_log_std: float = -0.5,
    ) -> None:
        super().__init__()
        if vocab_size < 2 or max_instruction_tokens < 1 or image_size < 16:
            raise ValueError("invalid visual-language encoder dimensions")
        if hidden_dim < 1 or topology_max_nodes < 2 or topology_model_dim < 1:
            raise ValueError("invalid topology policy dimensions")
        if max_candidates != MAX_CANDIDATES:
            raise ValueError("topology policy currently requires five candidate slots")
        if _finite(residual_limit_m, "residual_limit_m") <= 0.0:
            raise ValueError("residual_limit_m must be positive")
        self.vocab_size = int(vocab_size)
        self.max_instruction_tokens = int(max_instruction_tokens)
        self.image_size = int(image_size)
        self.hidden_dim = int(hidden_dim)
        self.topology_max_nodes = int(topology_max_nodes)
        self.topology_model_dim = int(topology_model_dim)
        self.max_candidates = int(max_candidates)
        self.residual_limit_m = float(residual_limit_m)

        self.observation_encoder = _FrozenVLNEncoder(
            vocab_size=self.vocab_size,
            max_instruction_tokens=self.max_instruction_tokens,
            image_size=self.image_size,
        )
        self.topology_encoder = P8TopologyEncoder(
            node_hidden_dim=1,
            node_feature_dim=CAUSAL_NODE_FEATURE_DIM,
            edge_feature_dim=CAUSAL_EDGE_FEATURE_DIM,
            model_dim=self.topology_model_dim,
        )
        fused_dim = (
            self.observation_encoder.output_dim
            + self.topology_model_dim
            + len(CAUSAL_GLOBAL_FEATURE_NAMES)
        )
        self.context_fusion = nn.Sequential(
            nn.Linear(fused_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
        )
        candidate_input_dim = self.hidden_dim + CANDIDATE_FEATURE_DIM
        self.candidate_scorer = nn.Sequential(
            nn.Linear(candidate_input_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1),
        )
        # A slot bias is useful in its own right (the stable slots have distinct
        # frontier/node/STOP semantics) and lets bounded diagnostics select a
        # known branch without changing the candidate features.
        self.candidate_slot_bias = nn.Parameter(
            torch.zeros(self.max_candidates, dtype=torch.float32)
        )
        self.residual_mean_head = nn.Sequential(
            nn.Linear(candidate_input_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 2),
        )
        self.residual_log_std = nn.Parameter(
            torch.full((2,), float(initial_log_std), dtype=torch.float32)
        )
        self.value_head = nn.Linear(self.hidden_dim, 1)
        self.float()

    @property
    def frozen_parameter_names(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                name
                for name, parameter in self.named_parameters()
                if not parameter.requires_grad
            )
        )

    def _device(self):
        return self.value_head.weight.device

    @staticmethod
    def _candidate_features(candidates: Sequence[TopologyCandidate]) -> torch.Tensor:
        rows = []
        for candidate in candidates:
            distance = math.hypot(candidate.forward_m, candidate.lateral_m)
            bearing = math.atan2(candidate.lateral_m, candidate.forward_m) if distance else 0.0
            rows.append(
                (
                    candidate.forward_m / 1.5,
                    candidate.lateral_m / 1.5,
                    distance / 1.5,
                    math.sin(bearing),
                    math.cos(bearing),
                    float(candidate.kind == CANDIDATE_KIND_FRONTIER),
                    float(candidate.kind == CANDIDATE_KIND_TOPOLOGY_NODE),
                    float(candidate.kind == CANDIDATE_KIND_STOP),
                )
            )
        return torch.tensor([rows], dtype=torch.float32)

    def prepare_forward_inputs(
        self,
        observation: Mapping[str, object],
        snapshot: TopologyDecisionSnapshot,
        *,
        policy_version: int,
    ) -> dict[str, torch.Tensor]:
        if not isinstance(observation, Mapping) or set(observation) != OBSERVATION_KEYS:
            raise ValueError("topology policy requires RGB and instruction tensors only")
        if type(policy_version) is not int or policy_version < 0:
            raise ValueError("policy_version must be a non-negative integer")
        result: dict[str, torch.Tensor] = {}
        for key in OBSERVATION_KEYS:
            value = observation[key]
            if not isinstance(value, torch.Tensor):
                value = torch.as_tensor(value)
            result[key] = value.detach().cpu().clone()
        tensors = snapshot.topology.tensors.canonical()
        for name, value in tensors.as_dict().items():
            result[f"topology_{name}"] = value.detach().cpu().clone()
        result["topology_global_features"] = torch.tensor(
            [snapshot.topology.global_features], dtype=torch.float32
        )
        result["candidate_features"] = self._candidate_features(snapshot.candidates)
        result["candidate_mask"] = torch.tensor(
            [[candidate.valid for candidate in snapshot.candidates]], dtype=torch.bool
        )
        result["continuous_mask"] = torch.tensor(
            [[candidate.continuous for candidate in snapshot.candidates]], dtype=torch.bool
        )
        result["candidate_anchors"] = torch.tensor(
            [
                [
                    (candidate.forward_m, candidate.lateral_m)
                    for candidate in snapshot.candidates
                ]
            ],
            dtype=torch.float32,
        )
        result["policy_version"] = torch.tensor([policy_version], dtype=torch.int64)
        return result

    def _distribution(self, forward_inputs: Mapping[str, object]):
        required = set(OBSERVATION_KEYS) | {
            "topology_node_hidden",
            "topology_node_features",
            "topology_node_mask",
            "topology_adjacency",
            "topology_edge_features",
            "topology_current_index",
            "topology_parent_index",
            "topology_global_features",
            "candidate_features",
            "candidate_mask",
            "continuous_mask",
            "candidate_anchors",
            "policy_version",
        }
        if set(forward_inputs) != required:
            missing = sorted(required - set(forward_inputs))
            extra = sorted(set(forward_inputs) - required)
            raise ValueError(f"topology forward context drifted: missing={missing}, extra={extra}")
        device = self._device()
        obs = {
            key: forward_inputs[key].to(device=device)
            for key in OBSERVATION_KEYS
        }
        topology = P8TopologyTensors(
            node_hidden=forward_inputs["topology_node_hidden"].to(device),
            node_features=forward_inputs["topology_node_features"].to(device),
            node_mask=forward_inputs["topology_node_mask"].to(device),
            adjacency=forward_inputs["topology_adjacency"].to(device),
            edge_features=forward_inputs["topology_edge_features"].to(device),
            current_index=forward_inputs["topology_current_index"].to(device),
            parent_index=forward_inputs["topology_parent_index"].to(device),
        ).canonical()
        global_features = forward_inputs["topology_global_features"].to(
            device=device, dtype=torch.float32
        )
        candidate_features = forward_inputs["candidate_features"].to(
            device=device, dtype=torch.float32
        )
        candidate_mask = forward_inputs["candidate_mask"].to(device=device, dtype=torch.bool)
        continuous_mask = forward_inputs["continuous_mask"].to(
            device=device, dtype=torch.bool
        )
        anchors = forward_inputs["candidate_anchors"].to(
            device=device, dtype=torch.float32
        )
        if candidate_features.shape != (1, self.max_candidates, CANDIDATE_FEATURE_DIM):
            raise ValueError("candidate feature tensor shape drifted")
        if candidate_mask.shape != (1, self.max_candidates) or not bool(candidate_mask.any()):
            raise ValueError("candidate mask tensor shape drifted")
        if continuous_mask.shape != candidate_mask.shape or anchors.shape != (
            1,
            self.max_candidates,
            2,
        ):
            raise ValueError("candidate geometry tensor shape drifted")
        visual_language = self.observation_encoder(obs)
        graph = self.topology_encoder(topology)["context"]
        hidden = self.context_fusion(
            torch.cat((visual_language, graph, global_features), dim=-1)
        )
        expanded = hidden[:, None, :].expand(-1, self.max_candidates, -1)
        candidate_context = torch.cat((expanded, candidate_features), dim=-1)
        raw_logits = self.candidate_scorer(candidate_context).squeeze(-1)
        raw_logits = raw_logits + self.candidate_slot_bias.reshape(1, -1)
        logits = raw_logits.masked_fill(~candidate_mask, -torch.inf)
        means = self.residual_mean_head(candidate_context)
        log_std = self.residual_log_std.clamp(-5.0, 2.0).reshape(1, 1, 2).expand_as(means)
        values = self.value_head(hidden).squeeze(-1)
        return logits, means, log_std, values, continuous_mask, anchors

    def _sample(self, forward_inputs: Mapping[str, object], *, deterministic: bool):
        with torch.no_grad():
            logits, means, log_std, values, continuous_mask, anchors = self._distribution(
                forward_inputs
            )
            distribution = Categorical(logits=logits)
            candidate = logits.argmax(dim=-1) if deterministic else distribution.sample()
            batch = torch.arange(logits.shape[0], device=logits.device)
            selected_mean = means[batch, candidate]
            selected_log_std = log_std[batch, candidate]
            active = continuous_mask[batch, candidate]
            raw = (
                selected_mean
                if deterministic
                else Normal(selected_mean, selected_log_std.exp()).sample()
            )
            raw = torch.where(active[:, None], raw, torch.zeros_like(raw))
            residual = self.residual_limit_m * torch.tanh(raw)
            waypoint = anchors[batch, candidate] + residual
            waypoint = torch.where(active[:, None], waypoint, torch.zeros_like(waypoint))
            route_logprob = distribution.log_prob(candidate)
            raw_logprob = Normal(selected_mean, selected_log_std.exp()).log_prob(raw).sum(-1)
            raw_logprob = torch.where(active, raw_logprob, torch.zeros_like(raw_logprob))
            normal_entropy = Normal(means, log_std.exp()).entropy().sum(-1)
            entropy = distribution.entropy() + (
                distribution.probs * continuous_mask.float() * normal_entropy
            ).sum(-1)
            return {
                "candidate": candidate,
                "raw": raw,
                "waypoint": waypoint,
                "route_logprob": route_logprob,
                "raw_logprob": raw_logprob,
                "value": values,
                "entropy": entropy,
                "continuous": active,
            }

    def _replay(
        self,
        forward_inputs: Mapping[str, object],
        candidate_action: torch.Tensor,
        raw_action: torch.Tensor,
    ):
        logits, means, log_std, values, continuous_mask, anchors = self._distribution(
            forward_inputs
        )
        candidate = candidate_action.to(device=logits.device, dtype=torch.int64).reshape(-1)
        raw = raw_action.to(device=logits.device, dtype=torch.float32).reshape(-1, 2)
        if candidate.shape != (logits.shape[0],) or raw.shape != (logits.shape[0], 2):
            raise ValueError("replay action shape differs from policy batch")
        if bool(((candidate < 0) | (candidate >= self.max_candidates)).any()):
            raise ValueError("replay candidate index is outside the fixed extent")
        batch = torch.arange(logits.shape[0], device=logits.device)
        valid = torch.isfinite(logits[batch, candidate])
        if not bool(valid.all()):
            raise ValueError("replay selected a masked topology candidate")
        active = continuous_mask[batch, candidate]
        if bool((~active[:, None] & (raw != 0.0)).any()):
            raise ValueError("discrete candidate replay cannot carry a raw residual")
        selected_mean = means[batch, candidate]
        selected_log_std = log_std[batch, candidate]
        distribution = Categorical(logits=logits)
        route_logprob = distribution.log_prob(candidate)
        raw_logprob = Normal(selected_mean, selected_log_std.exp()).log_prob(raw).sum(-1)
        raw_logprob = torch.where(active, raw_logprob, torch.zeros_like(raw_logprob))
        waypoint = anchors[batch, candidate] + self.residual_limit_m * torch.tanh(raw)
        waypoint = torch.where(active[:, None], waypoint, torch.zeros_like(waypoint))
        normal_entropy = Normal(means, log_std.exp()).entropy().sum(-1)
        entropy = distribution.entropy() + (
            distribution.probs * continuous_mask.float() * normal_entropy
        ).sum(-1)
        return {
            "route_logprob": route_logprob,
            "raw_logprob": raw_logprob,
            "value": values,
            "entropy": entropy,
            "waypoint": waypoint,
            "continuous": active,
        }

    def forward(self, *, operation: str, **kwargs):
        if operation == "sample":
            return self._sample(
                kwargs["forward_inputs"], deterministic=bool(kwargs.get("deterministic", False))
            )
        if operation == "replay":
            return self._replay(
                kwargs["forward_inputs"],
                kwargs["candidate_action"],
                kwargs["raw_action"],
            )
        if operation == "bootstrap":
            *_, values, _, _ = self._distribution(kwargs["forward_inputs"])
            return values
        raise ValueError(f"unknown topology waypoint operation {operation!r}")


def _context_candidates(context: Mapping[str, object]) -> tuple[TopologyCandidate, ...]:
    rows = context.get("candidates")
    if not isinstance(rows, tuple) or len(rows) != MAX_CANDIDATES:
        raise ValueError("replay context candidate rows are invalid")
    candidates = tuple(TopologyCandidate.from_canonical(row) for row in rows)
    digest = context.get("candidate_digest")
    if digest != candidate_digest(candidates):
        raise ValueError("replay context candidate digest was tampered")
    expected_fields = {
        "candidate_ids": tuple(row.candidate_id for row in candidates),
        "candidate_kinds": tuple(row.kind for row in candidates),
        "candidate_mask": tuple(row.valid for row in candidates),
        "candidate_continuous_mask": tuple(row.continuous for row in candidates),
        "candidate_anchors_m": tuple(
            (row.forward_m, row.lateral_m) for row in candidates
        ),
    }
    for name, expected in expected_fields.items():
        if context.get(name) != expected:
            raise ValueError(f"replay context {name} differs from candidate snapshot")
    return candidates


def _validate_payload(
    sample: PolicySample,
    *,
    residual_limit_m: float,
) -> tuple[TopologyCandidate, int, tuple[float, float], tuple[float, float]]:
    payload = sample.action_payload
    if len(payload) != 8 or payload[0] != ACTION_SCHEMA:
        raise ValueError("topology waypoint action payload schema drifted")
    candidates = _context_candidates(sample.replay_context)
    digest = payload[1]
    if digest != sample.replay_context.get("candidate_digest"):
        raise ValueError("action payload candidate digest differs from replay context")
    slot = payload[2]
    if type(slot) is not int or not 0 <= slot < len(candidates):
        raise ValueError("action payload candidate slot is invalid")
    candidate = candidates[slot]
    if not candidate.valid or payload[3] != candidate.candidate_id:
        raise ValueError("action payload selected the wrong or masked candidate")
    raw = (_finite(payload[4], "raw residual 0"), _finite(payload[5], "raw residual 1"))
    waypoint = (
        _finite(payload[6], "metric waypoint forward"),
        _finite(payload[7], "metric waypoint lateral"),
    )
    if candidate.continuous:
        expected = (
            candidate.forward_m + residual_limit_m * math.tanh(raw[0]),
            candidate.lateral_m + residual_limit_m * math.tanh(raw[1]),
        )
        if any(
            not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-6)
            for left, right in zip(waypoint, expected)
        ):
            raise ValueError("resolved metric waypoint differs from anchor plus residual")
        if len(sample.behavior_log_prob_terms) != 2:
            raise ValueError("continuous candidate requires route and residual log-prob terms")
    else:
        if raw != (0.0, 0.0) or waypoint != (0.0, 0.0):
            raise ValueError("STOP payload must not carry continuous geometry")
        if len(sample.behavior_log_prob_terms) != 1:
            raise ValueError("STOP requires only the categorical log-prob term")
    return candidate, slot, raw, waypoint


class TopologyWaypointPolicyAdapter(PolicyAdapter):
    action_distribution = "masked_categorical_plus_conditional_gaussian_residual"

    def __init__(
        self,
        policy: TopologyWaypointPolicy,
        topology_state: EpisodeTopologyState,
        *,
        mode: str = "train",
        deterministic: bool | None = None,
        policy_version: int = 0,
        root_forward: Callable[..., object] | None = None,
    ) -> None:
        if mode not in ("train", "eval"):
            raise ValueError("topology policy mode must be train or eval")
        if deterministic is not None and type(deterministic) is not bool:
            raise TypeError("deterministic must be bool or None")
        if type(policy_version) is not int or policy_version < 0:
            raise ValueError("policy_version must be a non-negative integer")
        if policy.topology_max_nodes != topology_state.topology_max_nodes:
            raise ValueError("policy and state topology extents differ")
        if not math.isclose(
            policy.residual_limit_m,
            topology_state.residual_limit_m,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("policy and state residual transforms differ")
        self.policy = policy
        self.topology_state = topology_state
        self.mode = mode
        self.deterministic = (mode == "eval") if deterministic is None else deterministic
        self.root_forward = root_forward
        self._policy_version = policy_version
        self._identity: EpisodeIdentity | None = None
        self._generation = 0

    def _forward(self, operation: str, **kwargs):
        root = self.root_forward if self.root_forward is not None else self.policy
        return root(operation=operation, **kwargs)

    def _require_identity(self, identity: EpisodeIdentity) -> None:
        if identity != self._identity:
            raise RuntimeError("stale or unbound topology policy reset identity")

    def reset_state(self, identity: EpisodeIdentity) -> None:
        if not isinstance(identity, EpisodeIdentity):
            raise TypeError("policy identity must be EpisodeIdentity")
        self._identity = identity
        self._generation += 1
        self.topology_state.bind(identity)

    def sample(
        self,
        observation: object,
        identity: EpisodeIdentity,
        decision_index: int,
        physical_step: int,
    ) -> PolicySample:
        self._require_identity(identity)
        snapshot = self.topology_state.snapshot(identity, decision_index, physical_step)
        forward_inputs = self.policy.prepare_forward_inputs(
            observation, snapshot, policy_version=self.policy_version
        )
        output = self._forward(
            "sample",
            forward_inputs=forward_inputs,
            deterministic=self.deterministic,
        )
        slot = int(output["candidate"].reshape(-1)[0].item())
        candidate = snapshot.candidates[slot]
        raw = tuple(float(value) for value in output["raw"].reshape(-1).tolist())
        waypoint = tuple(float(value) for value in output["waypoint"].reshape(-1).tolist())
        terms = (output["route_logprob"].reshape(-1)[0],)
        if candidate.continuous:
            terms += (output["raw_logprob"].reshape(-1)[0],)
        return PolicySample(
            identity=identity,
            decision_index=decision_index,
            physical_step=physical_step,
            action_payload=(
                ACTION_SCHEMA,
                snapshot.digest,
                slot,
                candidate.candidate_id,
                raw[0],
                raw[1],
                waypoint[0],
                waypoint[1],
            ),
            replay_context={
                "adapter_kind": ACTION_SCHEMA,
                "generation": self._generation,
                "candidate_digest": snapshot.digest,
                "candidates": tuple(row.canonical() for row in snapshot.candidates),
                "candidate_ids": tuple(row.candidate_id for row in snapshot.candidates),
                "candidate_kinds": tuple(row.kind for row in snapshot.candidates),
                "candidate_mask": tuple(row.valid for row in snapshot.candidates),
                "candidate_continuous_mask": tuple(
                    row.continuous for row in snapshot.candidates
                ),
                "candidate_anchors_m": tuple(
                    (row.forward_m, row.lateral_m) for row in snapshot.candidates
                ),
                "forward_inputs": forward_inputs,
                "forward_inputs_digest": _tensor_context_digest(forward_inputs),
            },
            behavior_log_prob_terms=terms,
            value=output["value"].reshape(-1)[0],
            policy_version=self.policy_version,
            action_distribution=self.action_distribution,
        )

    def _validate_bound_sample(self, sample: PolicySample):
        self._require_identity(sample.identity)
        context = sample.replay_context
        if (
            context.get("adapter_kind") != ACTION_SCHEMA
            or context.get("generation") != self._generation
        ):
            raise RuntimeError("topology replay context belongs to another adapter generation")
        if sample.policy_version != self.policy_version:
            raise RuntimeError("topology replay policy version differs from behavior version")
        forward_inputs = context.get("forward_inputs")
        if not isinstance(forward_inputs, Mapping) or not forward_inputs:
            raise ValueError("topology replay context has no forward inputs")
        if context.get("forward_inputs_digest") != _tensor_context_digest(forward_inputs):
            raise ValueError("topology replay forward inputs were tampered")
        candidates = _context_candidates(context)
        expected_mask = torch.tensor(
            [[candidate.valid for candidate in candidates]], dtype=torch.bool
        )
        expected_continuous = torch.tensor(
            [[candidate.continuous for candidate in candidates]], dtype=torch.bool
        )
        expected_anchors = torch.tensor(
            [[(candidate.forward_m, candidate.lateral_m) for candidate in candidates]],
            dtype=torch.float32,
        )
        expected_features = self.policy._candidate_features(candidates)
        for name, expected in (
            ("candidate_mask", expected_mask),
            ("continuous_mask", expected_continuous),
            ("candidate_anchors", expected_anchors),
            ("candidate_features", expected_features),
        ):
            actual = forward_inputs.get(name)
            if not isinstance(actual, torch.Tensor) or not torch.equal(
                actual.detach().cpu(), expected
            ):
                raise ValueError(f"topology replay {name} differs from candidate snapshot")
        context_version = forward_inputs.get("policy_version")
        if context_version is None or int(context_version.reshape(-1)[0].item()) != self.policy_version:
            raise RuntimeError("topology root-forward context carries a stale policy version")
        return _validate_payload(sample, residual_limit_m=self.policy.residual_limit_m)

    def behavior_joint_log_prob(
        self, sample: PolicySample, execution: ExecutionReceipt
    ) -> object:
        candidate, _, _, waypoint = self._validate_bound_sample(sample)
        if (
            execution.identity != sample.identity
            or execution.decision_index != sample.decision_index
            or execution.physical_step != sample.physical_step
            or execution.source_action_payload != sample.action_payload
        ):
            raise RuntimeError("execution receipt belongs to another topology sample")
        expected_credit = (True,) * len(sample.behavior_log_prob_terms)
        if execution.credited_log_prob_mask != expected_credit:
            raise RuntimeError("hybrid action requires whole-decision log-prob credit")
        proposed = (
            (STOP_ACTION,)
            if not candidate.continuous
            else compile_metric_waypoint(
                waypoint, config=self.topology_state.compiler_config
            )
        )
        if execution.active_primitives != proposed[: execution.duration]:
            raise RuntimeError("executed primitive prefix differs from metric waypoint")
        return sum(sample.behavior_log_prob_terms)

    def evaluate(self, batch: DecisionBatch) -> tuple[PolicyEvaluation, ...]:
        self._require_identity(batch.identity)
        rows = []
        for record in batch.records:
            sample = record.sample
            candidate, slot, raw, _ = self._validate_bound_sample(sample)
            self.behavior_joint_log_prob(sample, record.execution)
            output = self._forward(
                "replay",
                forward_inputs=sample.replay_context["forward_inputs"],
                candidate_action=torch.tensor([slot], dtype=torch.int64),
                raw_action=torch.tensor([raw], dtype=torch.float32),
            )
            joint = output["route_logprob"].reshape(-1)[0]
            if candidate.continuous:
                joint = joint + output["raw_logprob"].reshape(-1)[0]
            rows.append(
                PolicyEvaluation(
                    identity=sample.identity,
                    decision_index=sample.decision_index,
                    joint_log_prob=joint,
                    value=output["value"].reshape(-1)[0],
                    policy_version=self.policy_version,
                    entropy=output["entropy"].reshape(-1)[0],
                )
            )
        return tuple(rows)

    def bootstrap(
        self,
        observation: object,
        identity: EpisodeIdentity,
        physical_step: int,
    ) -> BootstrapValue:
        self._require_identity(identity)
        topology = self.topology_state._require_identity(identity)
        snapshot = self.topology_state.snapshot(
            identity, topology.decision_count, physical_step
        )
        forward_inputs = self.policy.prepare_forward_inputs(
            observation, snapshot, policy_version=self.policy_version
        )
        with torch.no_grad():
            value = self._forward("bootstrap", forward_inputs=forward_inputs)
        return BootstrapValue(
            identity=identity,
            physical_step=physical_step,
            policy_version=self.policy_version,
            value=value.reshape(-1)[0],
        )

    def trainable_parameter_names(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                name
                for name, parameter in self.policy.named_parameters()
                if parameter.requires_grad
            )
        )

    def trainable_manifest(self) -> Mapping[str, object]:
        return {
            "source_kind": "lightweight_topology_metric_waypoint",
            "action_distribution": self.action_distribution,
            # The callable may be RLinf ``manager.model`` in production, the
            # policy module in a single-process smoke, or a test spy.  Presence
            # alone is not evidence that an RLinf/FSDP manager was used.
            "forward_path": (
                "injected_root_callable"
                if self.root_forward is not None
                else "direct_policy_call"
            ),
            "trainable_names": list(self.trainable_parameter_names()),
            "frozen_names": list(self.policy.frozen_parameter_names),
            "policy_version": self.policy_version,
        }

    @property
    def policy_version(self) -> int:
        return self._policy_version

    def mark_updated(self) -> None:
        self._policy_version += 1


class TopologyWaypointActionExecutor(ActionExecutor):
    """Compile one hybrid action, execute its prefix, then update topology once."""

    def __init__(
        self,
        topology_state: EpisodeTopologyState,
        *,
        compiler_config: WaypointCompilerConfig = WaypointCompilerConfig(),
        max_width: int = 4,
    ) -> None:
        if compiler_config != topology_state.compiler_config:
            raise ValueError("executor and topology compiler contracts differ")
        if max_width != compiler_config.max_primitives:
            raise ValueError("executor width must equal compiler max_primitives")
        self.topology_state = topology_state
        self.compiler_config = compiler_config
        self._executor = PrefixActionExecutor(
            max_width=max_width,
            expand_action=self._expand,
            pad_action=None,
            credit_mode="whole_decision",
        )

    def _expand(self, payload: object) -> Sequence[int]:
        if not isinstance(payload, tuple):
            raise TypeError("topology action payload must be a tuple")
        # PrefixActionExecutor receives only the payload, so use a structural
        # decode here.  Full candidate/digest/residual validation happens below
        # against the immutable PolicySample replay context.
        if len(payload) != 8 or payload[0] != ACTION_SCHEMA:
            raise ValueError("topology action payload schema drifted")
        candidate_id = payload[3]
        if candidate_id == STOP_CANDIDATE_ID:
            return (STOP_ACTION,)
        waypoint = (
            _finite(payload[6], "metric waypoint forward"),
            _finite(payload[7], "metric waypoint lateral"),
        )
        return compile_metric_waypoint(waypoint, config=self.compiler_config)

    def execute(self, task: TaskAdapter, sample: PolicySample) -> ExecutionReceipt:
        candidate, _, _, _ = _validate_payload(
            sample, residual_limit_m=self.topology_state.residual_limit_m
        )
        self.topology_state._require_identity(sample.identity)
        if (candidate.kind == CANDIDATE_KIND_STOP) != (
            sample.action_payload[3] == STOP_CANDIDATE_ID
        ):
            raise ValueError("STOP candidate semantics drifted")
        receipt = self._executor.execute(task, sample)
        self.topology_state.apply_receipt(receipt)
        return receipt


class TopologyWaypointCheckpointAdapter(CheckpointAdapter):
    """Strict full-state codec bound to graph, candidate and compiler semantics."""

    def __init__(self, adapter: TopologyWaypointPolicyAdapter):
        self.adapter = adapter

    @staticmethod
    def _state_schema(state: Mapping[str, object]) -> Mapping[str, object]:
        return {
            str(name): {
                "shape": [int(item) for item in value.shape],
                "dtype": str(value.dtype),
            }
            for name, value in sorted(state.items())
        }

    def checkpoint_compatibility(self) -> Mapping[str, object]:
        policy = self.adapter.policy
        return {
            "checkpoint_kind": "topology_waypoint_full_state_v1",
            "policy_class": type(policy).__qualname__,
            "policy_semantics": {
                "vocab_size": policy.vocab_size,
                "max_instruction_tokens": policy.max_instruction_tokens,
                "image_size": policy.image_size,
                "hidden_dim": policy.hidden_dim,
                "topology_max_nodes": policy.topology_max_nodes,
                "topology_model_dim": policy.topology_model_dim,
                "max_candidates": policy.max_candidates,
                "residual_limit_m": policy.residual_limit_m,
                "action_distribution": self.adapter.action_distribution,
            },
            "topology_action_contract": self.adapter.topology_state.compatibility(),
            "state_schema": self._state_schema(policy.state_dict()),
            "trainable_names": list(self.adapter.trainable_parameter_names()),
            "frozen_names": list(policy.frozen_parameter_names),
        }

    def checkpoint_metadata(self) -> Mapping[str, object]:
        return {
            "checkpoint_kind": "topology_waypoint_full_state_v1",
            "policy_version": self.adapter.policy_version,
        }

    def save_checkpoint(self, staging_directory: Path) -> Mapping[str, str]:
        path = Path(staging_directory) / "topology_waypoint_policy.pt"
        torch.save(
            {
                "state_dict": self.adapter.policy.state_dict(),
                "policy_version": self.adapter.policy_version,
            },
            path,
        )
        return {path.name: _sha256(path)}

    @staticmethod
    def _read_payload(directory: Path) -> Mapping[str, object]:
        path = Path(directory) / "topology_waypoint_policy.pt"
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, Mapping) or set(payload) != {
            "state_dict",
            "policy_version",
        }:
            raise ValueError("topology waypoint checkpoint payload is incomplete")
        if not isinstance(payload["state_dict"], Mapping):
            raise ValueError("topology waypoint checkpoint state_dict is invalid")
        version = payload["policy_version"]
        if type(version) is not int or version < 0:
            raise ValueError("topology waypoint checkpoint policy_version is invalid")
        return payload

    def inspect_checkpoint_metadata(self, directory: Path) -> Mapping[str, object]:
        payload = self._read_payload(directory)
        return {
            "checkpoint_kind": "topology_waypoint_full_state_v1",
            "policy_version": payload["policy_version"],
        }

    def load_checkpoint(self, directory: Path) -> None:
        payload = self._read_payload(directory)
        saved_state = payload["state_dict"]
        if self._state_schema(saved_state) != self._state_schema(
            self.adapter.policy.state_dict()
        ):
            raise RuntimeError("topology waypoint checkpoint state schema differs")
        self.adapter.policy.load_state_dict(saved_state, strict=True)
        self.adapter._policy_version = int(payload["policy_version"])
        if self.adapter._identity is not None:
            self.adapter.reset_state(self.adapter._identity)


class RLinfVLNTaskAdapter(TaskAdapter):
    """Wrap the fixed-episode RLinf ``VLNEnv`` vector-style API for the stack."""

    def __init__(self, environment: object, *, env_slot: int = 0) -> None:
        if type(env_slot) is not int or env_slot < 0:
            raise ValueError("env_slot must be a non-negative integer")
        self.environment = environment
        self.env_slot = env_slot
        self._reset_count = 0
        self._identity: EpisodeIdentity | None = None
        self._physical_step = 0
        self._path_length_m = 0.0
        self._collision_count = 0
        self._stop_called = False
        self._timed_out = False
        self._terminated = False
        self._success: bool | None = None
        self._spl: float | None = None
        self._navigation_error_m: float | None = None
        self._action_counts: dict[str, int] = {}

    def _selected_episode_id(self) -> str:
        identity = getattr(self.environment, "identity", None)
        if not isinstance(identity, Mapping):
            raise RuntimeError("RLinf VLN environment has no fixed episode identity")
        episode_id = identity.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            raise RuntimeError("RLinf VLN environment episode id is invalid")
        return episode_id

    def reset(self, episode: object | None = None, *, seed: int | None = None) -> TaskReset:
        del seed  # The fixed train episode is selected in the environment config.
        result = self.environment.reset()
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError("RLinf VLN reset must return (observation, info)")
        observation, _info = result
        if not isinstance(observation, Mapping) or set(observation) != OBSERVATION_KEYS:
            raise ValueError("RLinf VLN policy observation contract drifted")
        episode_id = self._selected_episode_id()
        if episode is not None and str(episode) not in {
            episode_id,
            str(getattr(self.environment, "identity", {}).get("episode_index")),
        }:
            raise ValueError("requested episode differs from the environment's fixed train episode")
        self._reset_count += 1
        self._identity = EpisodeIdentity(
            episode_id=episode_id,
            reset_token=f"slot={self.env_slot}:reset={self._reset_count}",
            env_slot=self.env_slot,
        )
        self._physical_step = 0
        self._path_length_m = 0.0
        self._collision_count = 0
        self._stop_called = False
        self._timed_out = False
        self._terminated = False
        self._success = self._spl = self._navigation_error_m = None
        self._action_counts = {}
        return TaskReset(self._identity, dict(observation))

    def step(self, primitive_action: object) -> PrimitiveStep:
        if self._identity is None:
            raise RuntimeError("call reset before stepping RLinf VLN")
        action = int(primitive_action)
        observation, reward, terminated, truncated, info = self.environment.step(action)
        if not isinstance(observation, Mapping) or not isinstance(info, Mapping):
            raise TypeError("RLinf VLN step must return mapping observation and info")
        self._physical_step += 1
        terminated_flag = _boolean(terminated, "terminated")
        truncated_flag = _boolean(truncated, "truncated")
        if terminated_flag and truncated_flag:
            raise ValueError("RLinf VLN step cannot both terminate and truncate")
        self._path_length_m = _finite(info["path_length"], "path_length")
        self._collision_count = int(round(_finite(info["collision_count"], "collision_count")))
        self._stop_called = self._stop_called or action == STOP_ACTION
        self._timed_out = self._timed_out or truncated_flag
        self._terminated = self._terminated or terminated_flag
        self._action_counts[str(action)] = self._action_counts.get(str(action), 0) + 1
        if "success" in info:
            self._success = bool(round(_finite(info["success"], "success")))
        episode_info = info.get("episode")
        if isinstance(episode_info, Mapping):
            if "spl" in episode_info:
                self._spl = _finite(episode_info["spl"], "spl")
            if "distance_to_goal" in episode_info:
                self._navigation_error_m = _finite(
                    episode_info["distance_to_goal"], "distance_to_goal"
                )
        return PrimitiveStep(
            identity=self._identity,
            physical_step=self._physical_step,
            observation=dict(observation),
            reward=_finite(reward, "reward"),
            terminated=terminated_flag,
            truncated=truncated_flag,
            info=dict(info),
        )

    def get_metrics(self) -> EpisodeMetrics:
        if self._identity is None:
            raise RuntimeError("call reset before reading RLinf VLN metrics")
        outcome_available = self._terminated or self._timed_out
        return EpisodeMetrics(
            identity=self._identity,
            path_length_m=self._path_length_m,
            collisions=self._collision_count,
            stop_called=self._stop_called,
            timed_out=self._timed_out,
            action_counts=dict(self._action_counts),
            success=self._success if outcome_available else None,
            spl=self._spl if outcome_available else None,
            navigation_error_m=(
                self._navigation_error_m if outcome_available else None
            ),
            extras={
                "bounded_segment": not outcome_available,
                "latest_success_signal": (
                    self._success if not outcome_available else None
                ),
            },
        )

    def close(self) -> None:
        self.environment.close()


__all__ = [
    "ACTION_SCHEMA",
    "CANDIDATE_KIND_FRONTIER",
    "CANDIDATE_KIND_STOP",
    "CANDIDATE_KIND_TOPOLOGY_NODE",
    "EpisodeTopologyState",
    "RLinfVLNTaskAdapter",
    "STOP_CANDIDATE_ID",
    "TopologyCandidate",
    "TopologyDecisionSnapshot",
    "TopologyWaypointActionExecutor",
    "TopologyWaypointCheckpointAdapter",
    "TopologyWaypointPolicy",
    "TopologyWaypointPolicyAdapter",
    "candidate_digest",
]
