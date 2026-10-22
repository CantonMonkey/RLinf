"""Masked tensor contract and graph encoder for StreamVLN P8 topology memory."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
import torch.nn as nn

try:
    from .streamvln_topology import (
        CAUSAL_EDGE_FEATURE_DIM,
        CAUSAL_GLOBAL_FEATURE_NAMES,
        CAUSAL_NODE_FEATURE_DIM,
        OnlineTopology,
        causal_topology_payload,
    )
except ImportError:  # Direct staged-file tests.
    from streamvln_topology import (
        CAUSAL_EDGE_FEATURE_DIM,
        CAUSAL_GLOBAL_FEATURE_NAMES,
        CAUSAL_NODE_FEATURE_DIM,
        OnlineTopology,
        causal_topology_payload,
    )


TOPOLOGY_KEYS = (
    "node_hidden",
    "node_features",
    "node_mask",
    "adjacency",
    "edge_features",
    "current_index",
    "parent_index",
)

_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}

CAUSAL_NODE_HIDDEN_DIM = 1
CAUSAL_ROUTE_COUNT = 3
CAUSAL_GLOBAL_FEATURE_DIM = len(CAUSAL_GLOBAL_FEATURE_NAMES)


@dataclass(frozen=True)
class P8CausalTopologySnapshot:
    """Identity-bound, history-only topology state sent to one policy decision."""

    episode_token: str
    physical_step: int
    decision_index: int
    route_candidate_mask: tuple[bool, bool, bool]
    global_features: tuple[float, ...]
    tensors: "P8TopologyTensors"

    def __post_init__(self) -> None:
        if not isinstance(self.episode_token, str) or not self.episode_token:
            raise ValueError("causal topology snapshot requires an episode token")
        if type(self.decision_index) is not int or self.decision_index < 0:
            raise ValueError("causal topology snapshot decision_index is invalid")
        if type(self.physical_step) is not int or self.physical_step < 0:
            raise ValueError("causal topology snapshot physical_step is invalid")
        if (
            not isinstance(self.route_candidate_mask, tuple)
            or len(self.route_candidate_mask) != CAUSAL_ROUTE_COUNT
            or any(type(value) is not bool for value in self.route_candidate_mask)
            or not any(self.route_candidate_mask)
        ):
            raise ValueError("causal route candidate mask is invalid")
        if len(self.global_features) != CAUSAL_GLOBAL_FEATURE_DIM or any(
            not math.isfinite(float(value)) for value in self.global_features
        ):
            raise ValueError("causal topology global feature contract drifted")
        self.tensors.validate()


def snapshot_online_topology(
    topology: OnlineTopology,
    *,
    episode_token: str,
    physical_step: int,
    decision_index: int,
    max_nodes: int,
) -> P8CausalTopologySnapshot:
    """Encode only current/past dead-reckoned nodes and executed-feedback edges.

    This adapter deliberately has no access to goals, demonstrations, geodesics,
    reference paths, or future simulator state.
    """

    if topology.decision_count != decision_index:
        raise ValueError("topology decision count disagrees with frame identity")
    payload = causal_topology_payload(topology, max_nodes=max_nodes)

    node_hidden = torch.zeros(
        1, max_nodes, CAUSAL_NODE_HIDDEN_DIM, dtype=torch.float32
    )
    node_features = torch.zeros(
        1, max_nodes, CAUSAL_NODE_FEATURE_DIM, dtype=torch.float32
    )
    node_mask = torch.zeros(1, max_nodes, dtype=torch.bool)
    adjacency = torch.zeros(1, max_nodes, max_nodes, dtype=torch.bool)
    edge_features = torch.zeros(
        1, max_nodes, max_nodes, CAUSAL_EDGE_FEATURE_DIM, dtype=torch.float32
    )
    parent_index = torch.full((1, max_nodes), -1, dtype=torch.int64)

    for index, features in enumerate(payload.node_features):
        node_mask[0, index] = True
        parent_index[0, index] = payload.parent_index[index]
        node_features[0, index] = torch.tensor(features, dtype=torch.float32)

    for source, row in enumerate(payload.adjacency):
        for target, is_edge in enumerate(row):
            if is_edge:
                adjacency[0, source, target] = True
                edge_features[0, source, target] = torch.tensor(
                    payload.edge_features[source][target], dtype=torch.float32
                )

    tensors = P8TopologyTensors(
        node_hidden=node_hidden,
        node_features=node_features,
        node_mask=node_mask,
        adjacency=adjacency,
        edge_features=edge_features,
        current_index=torch.tensor([payload.current_index], dtype=torch.int64),
        parent_index=parent_index,
    ).canonical()
    return P8CausalTopologySnapshot(
        episode_token=episode_token,
        physical_step=physical_step,
        decision_index=decision_index,
        route_candidate_mask=(True, True, True),
        global_features=payload.global_features,
        tensors=tensors,
    )


@dataclass(frozen=True)
class P8TopologyTensors:
    """Padded causal topology tensors for one policy batch.

    ``parent_index[b, n]`` is the parent of node ``n`` or ``-1`` for a root.
    Values stored in masked node or edge slots are deliberately ignored.
    """

    node_hidden: torch.Tensor
    node_features: torch.Tensor
    node_mask: torch.Tensor
    adjacency: torch.Tensor
    edge_features: torch.Tensor
    current_index: torch.Tensor
    parent_index: torch.Tensor

    @classmethod
    def from_mapping(cls, values: Mapping[str, torch.Tensor]) -> "P8TopologyTensors":
        missing = [name for name in TOPOLOGY_KEYS if name not in values]
        if missing:
            raise KeyError(f"missing P8 topology tensors: {', '.join(missing)}")
        return cls(**{name: values[name] for name in TOPOLOGY_KEYS})

    @property
    def batch_size(self) -> int:
        return int(self.node_mask.shape[0])

    @property
    def max_nodes(self) -> int:
        return int(self.node_mask.shape[1])

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {name: getattr(self, name) for name in TOPOLOGY_KEYS}

    def validate(self) -> "P8TopologyTensors":
        tensors = self.as_dict()
        if any(not isinstance(value, torch.Tensor) for value in tensors.values()):
            raise TypeError("all P8 topology fields must be torch tensors")
        devices = {value.device for value in tensors.values()}
        if len(devices) != 1:
            raise ValueError("P8 topology tensors must share one device")
        if self.node_hidden.ndim != 3 or self.node_features.ndim != 3:
            raise ValueError("node_hidden and node_features must have shape [B, N, D]")
        batch, nodes = self.node_hidden.shape[:2]
        if batch < 1 or nodes < 1 or tuple(self.node_features.shape[:2]) != (batch, nodes):
            raise ValueError("P8 topology node batch shape mismatch")
        if tuple(self.node_mask.shape) != (batch, nodes) or self.node_mask.dtype != torch.bool:
            raise ValueError("node_mask must be bool with shape [B, N]")
        if tuple(self.adjacency.shape) != (batch, nodes, nodes):
            raise ValueError("adjacency must have shape [B, N, N]")
        if self.adjacency.dtype != torch.bool:
            raise ValueError("adjacency must be a bool tensor")
        if self.edge_features.ndim != 4 or tuple(self.edge_features.shape[:3]) != (
            batch,
            nodes,
            nodes,
        ):
            raise ValueError("edge_features must have shape [B, N, N, D]")
        if tuple(self.current_index.shape) != (batch,):
            raise ValueError("current_index must have shape [B]")
        if tuple(self.parent_index.shape) != (batch, nodes):
            raise ValueError("parent_index must have shape [B, N]")
        if self.current_index.dtype not in _INTEGER_DTYPES:
            raise ValueError("current_index must contain integer indices")
        if self.parent_index.dtype not in _INTEGER_DTYPES:
            raise ValueError("parent_index must contain integer indices")
        if not self.node_hidden.is_floating_point() or not self.node_features.is_floating_point():
            raise ValueError("P8 node tensors must be floating point")
        if not self.edge_features.is_floating_point():
            raise ValueError("P8 edge_features must be floating point")
        if bool((self.node_mask.sum(dim=1) == 0).any()):
            raise ValueError("every P8 topology row must contain a valid node")

        current = self.current_index.to(dtype=torch.int64)
        if bool(((current < 0) | (current >= nodes)).any()):
            raise ValueError("current_index is outside the padded node extent")
        if not bool(self.node_mask.gather(1, current[:, None]).all()):
            raise ValueError("current_index must select a valid node")

        parents = self.parent_index.to(dtype=torch.int64)
        in_range = (parents == -1) | ((parents >= 0) & (parents < nodes))
        if not bool(in_range[self.node_mask].all()):
            raise ValueError("a valid node has an invalid parent_index")
        safe_parents = parents.clamp(min=0, max=nodes - 1)
        parent_is_valid = self.node_mask.gather(1, safe_parents)
        has_parent = self.node_mask & (parents >= 0)
        if not bool(parent_is_valid[has_parent].all()):
            raise ValueError("a valid node points to a masked parent")

        if not bool(torch.isfinite(self.node_hidden[self.node_mask]).all()):
            raise ValueError("valid node_hidden entries must be finite")
        if not bool(torch.isfinite(self.node_features[self.node_mask]).all()):
            raise ValueError("valid node_features entries must be finite")
        pair_mask = (
            self.node_mask[:, :, None]
            & self.node_mask[:, None, :]
            & self.adjacency
        )
        if bool(pair_mask.any()) and not bool(
            torch.isfinite(self.edge_features[pair_mask]).all()
        ):
            raise ValueError("active edge_features entries must be finite")
        return self

    def canonical(self, *, device: torch.device | str | None = None) -> "P8TopologyTensors":
        """Detach, cast to the P8 contract, and erase all padded content."""

        self.validate()
        target = self.node_hidden.device if device is None else torch.device(device)
        node_mask = self.node_mask.detach().to(device=target, dtype=torch.bool)
        adjacency = self.adjacency.detach().to(device=target, dtype=torch.bool)
        pair_mask = node_mask[:, :, None] & node_mask[:, None, :] & adjacency
        node_hidden = self.node_hidden.detach().to(device=target, dtype=torch.float32)
        node_features = self.node_features.detach().to(device=target, dtype=torch.float32)
        edge_features = self.edge_features.detach().to(device=target, dtype=torch.float32)
        node_hidden = torch.where(node_mask[..., None], node_hidden, torch.zeros_like(node_hidden))
        node_features = torch.where(
            node_mask[..., None], node_features, torch.zeros_like(node_features)
        )
        edge_features = torch.where(
            pair_mask[..., None], edge_features, torch.zeros_like(edge_features)
        )
        parents = self.parent_index.detach().to(device=target, dtype=torch.int64)
        parents = torch.where(node_mask, parents, torch.full_like(parents, -1))
        return P8TopologyTensors(
            node_hidden=node_hidden,
            node_features=node_features,
            node_mask=node_mask,
            adjacency=pair_mask,
            edge_features=edge_features,
            current_index=self.current_index.detach().to(device=target, dtype=torch.int64),
            parent_index=parents,
        )

    def index_select(self, index: torch.Tensor) -> "P8TopologyTensors":
        index = index.to(device=self.node_mask.device, dtype=torch.int64).reshape(-1)
        return P8TopologyTensors(
            **{name: value.index_select(0, index) for name, value in self.as_dict().items()}
        )


def as_p8_topology(
    topology: P8TopologyTensors | Mapping[str, torch.Tensor],
) -> P8TopologyTensors:
    if isinstance(topology, P8TopologyTensors):
        return topology
    if isinstance(topology, Mapping):
        return P8TopologyTensors.from_mapping(topology)
    raise TypeError("topology must be P8TopologyTensors or a tensor mapping")


class P8TopologyEncoder(nn.Module):
    """Encode real node and edge tensors without allowing padding to contribute."""

    def __init__(
        self,
        *,
        node_hidden_dim: int,
        node_feature_dim: int,
        edge_feature_dim: int,
        model_dim: int = 128,
    ) -> None:
        super().__init__()
        dimensions = (node_hidden_dim, node_feature_dim, edge_feature_dim, model_dim)
        if any(int(value) <= 0 for value in dimensions):
            raise ValueError("P8 topology dimensions must be positive")
        self.node_hidden_dim = int(node_hidden_dim)
        self.node_feature_dim = int(node_feature_dim)
        self.edge_feature_dim = int(edge_feature_dim)
        self.model_dim = int(model_dim)

        self.node_hidden_projection = nn.Linear(self.node_hidden_dim, self.model_dim)
        self.node_feature_projection = nn.Linear(self.node_feature_dim, self.model_dim)
        self.edge_projection = nn.Linear(self.edge_feature_dim, self.model_dim)
        self.node_norm = nn.LayerNorm(self.model_dim)
        self.message_projection = nn.Linear(self.model_dim, self.model_dim)
        self.message_norm = nn.LayerNorm(self.model_dim)
        self.query_projection = nn.Linear(self.model_dim, self.model_dim, bias=False)
        self.key_projection = nn.Linear(self.model_dim, self.model_dim, bias=False)
        self.context_projection = nn.Linear(3 * self.model_dim, self.model_dim)
        self.context_norm = nn.LayerNorm(self.model_dim)
        self.float()

    def forward(
        self,
        topology: P8TopologyTensors | Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        device = self.node_hidden_projection.weight.device
        topology = as_p8_topology(topology).canonical(device=device)
        if topology.node_hidden.shape[-1] != self.node_hidden_dim:
            raise ValueError("P8 node_hidden width mismatch")
        if topology.node_features.shape[-1] != self.node_feature_dim:
            raise ValueError("P8 node_features width mismatch")
        if topology.edge_features.shape[-1] != self.edge_feature_dim:
            raise ValueError("P8 edge_features width mismatch")

        node_state = self.node_norm(
            self.node_hidden_projection(topology.node_hidden)
            + self.node_feature_projection(topology.node_features)
        )
        node_state = torch.where(
            topology.node_mask[..., None], node_state, torch.zeros_like(node_state)
        )
        edge_state = self.edge_projection(topology.edge_features)
        pair_weight = topology.adjacency[..., None].to(dtype=torch.float32)
        messages = node_state[:, None, :, :] + edge_state
        degree = pair_weight.sum(dim=2).clamp_min(1.0)
        aggregate = (messages * pair_weight).sum(dim=2) / degree
        node_state = self.message_norm(node_state + self.message_projection(aggregate))
        node_state = torch.where(
            topology.node_mask[..., None], node_state, torch.zeros_like(node_state)
        )

        batch_index = torch.arange(topology.batch_size, device=device)
        current_state = node_state[batch_index, topology.current_index]
        current_parent = topology.parent_index.gather(
            1, topology.current_index[:, None]
        ).squeeze(1)
        has_parent = current_parent >= 0
        parent_state = node_state[batch_index, current_parent.clamp_min(0)]
        parent_state = torch.where(
            has_parent[:, None], parent_state, torch.zeros_like(parent_state)
        )

        query = self.query_projection(current_state)
        keys = self.key_projection(node_state)
        attention_logits = (keys * query[:, None, :]).sum(-1) / math.sqrt(self.model_dim)
        attention_logits = attention_logits.masked_fill(
            ~topology.node_mask, torch.finfo(attention_logits.dtype).min
        )
        attention = attention_logits.softmax(dim=-1)
        attention = attention * topology.node_mask.to(dtype=attention.dtype)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        graph_state = (node_state * attention[..., None]).sum(dim=1)
        context = self.context_norm(
            self.context_projection(
                torch.cat((current_state, parent_state, graph_state), dim=-1)
            )
        )
        return {
            "context": context.float(),
            "node_states": node_state.float(),
            "attention_weights": attention.float(),
            "current_state": current_state.float(),
            "parent_state": parent_state.float(),
        }
