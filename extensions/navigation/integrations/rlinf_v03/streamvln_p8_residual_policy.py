"""FP32 policy-preserving token residual for the released StreamVLN policy."""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .streamvln_p8_topology import P8TopologyEncoder, P8TopologyTensors
except ImportError:
    from streamvln_p8_topology import P8TopologyEncoder, P8TopologyTensors


ACTION_NAMES = ("stop", "forward", "left", "right")
ACTION_COUNT = len(ACTION_NAMES)
ACTION_CHUNK = 4
STOP_ACTION = 0
FORWARD_ACTION = 1
LEFT_ACTION = 2
RIGHT_ACTION = 3

INTENT_NAMES = ("navigate", "stop", "recover")
INTENT_COUNT = len(INTENT_NAMES)
NAVIGATE_INTENT = 0
STOP_INTENT = 1
RECOVER_INTENT = 2


class StreamVLNPolicyPreservingResidual(nn.Module):
    """Add a zero-initialized residual to the released four-token logits.

    The frozen StreamVLN token hidden states and base logits are always detached
    and converted to FP32. Continuous waypoints are supervised auxiliaries and
    residual conditions; they never add a separate policy likelihood term.
    """

    def __init__(
        self,
        *,
        token_hidden_dim: int,
        node_hidden_dim: int,
        node_feature_dim: int,
        edge_feature_dim: int,
        model_dim: int = 128,
        max_waypoint_m: float = 1.25,
    ) -> None:
        super().__init__()
        dimensions = (
            token_hidden_dim,
            node_hidden_dim,
            node_feature_dim,
            edge_feature_dim,
            model_dim,
        )
        if any(int(value) <= 0 for value in dimensions):
            raise ValueError("P8 policy dimensions must be positive")
        if not float(max_waypoint_m) > 0.0:
            raise ValueError("max_waypoint_m must be positive")
        self.token_hidden_dim = int(token_hidden_dim)
        self.model_dim = int(model_dim)
        self.max_waypoint_m = float(max_waypoint_m)

        self.topology_encoder = P8TopologyEncoder(
            node_hidden_dim=int(node_hidden_dim),
            node_feature_dim=int(node_feature_dim),
            edge_feature_dim=int(edge_feature_dim),
            model_dim=self.model_dim,
        )
        self.decision_projection = nn.Sequential(
            nn.Linear(self.token_hidden_dim, self.model_dim),
            nn.GELU(),
            nn.LayerNorm(self.model_dim),
        )
        self.condition_fusion = nn.Sequential(
            nn.Linear(2 * self.model_dim, self.model_dim),
            nn.GELU(),
            nn.LayerNorm(self.model_dim),
        )
        self.waypoint_head = nn.Linear(self.model_dim, 2)
        self.intent_head = nn.Linear(self.model_dim, INTENT_COUNT)
        self.value_head = nn.Linear(self.model_dim, 1)

        self.token_projection = nn.Sequential(
            nn.Linear(self.token_hidden_dim, self.model_dim),
            nn.GELU(),
            nn.LayerNorm(self.model_dim),
        )
        self.waypoint_projection = nn.Sequential(
            nn.Linear(2, self.model_dim),
            nn.GELU(),
            nn.LayerNorm(self.model_dim),
        )
        self.residual_hidden = nn.Sequential(
            nn.Linear(3 * self.model_dim, self.model_dim),
            nn.GELU(),
            nn.LayerNorm(self.model_dim),
        )
        self.residual_output = nn.Linear(self.model_dim, ACTION_COUNT)
        with torch.no_grad():
            nn.init.zeros_(self.residual_output.weight)
            nn.init.zeros_(self.residual_output.bias)
            nn.init.zeros_(self.intent_head.weight)
            nn.init.zeros_(self.intent_head.bias)
        self.float()
        self._assert_fp32_parameters()

    def _assert_fp32_parameters(self) -> None:
        non_fp32 = [
            name for name, parameter in self.named_parameters()
            if parameter.dtype != torch.float32
        ]
        if non_fp32:
            raise RuntimeError("P8 trainable parameters must remain FP32: " + ", ".join(non_fp32))

    def trainable_manifest(self) -> dict:
        parameters = {
            name: {
                "numel": int(parameter.numel()),
                "dtype": str(parameter.dtype),
                "requires_grad": bool(parameter.requires_grad),
            }
            for name, parameter in self.named_parameters()
        }
        return {
            "action_names": ACTION_NAMES,
            "intent_names": INTENT_NAMES,
            "waypoint_is_policy_action": False,
            "parameter_count": sum(item["numel"] for item in parameters.values()),
            "parameters": parameters,
        }

    def _normalize_token_inputs(
        self,
        token_hidden: torch.Tensor,
        base_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if token_hidden.ndim == 1:
            token_hidden = token_hidden[None, None, :]
        elif token_hidden.ndim == 2:
            token_hidden = token_hidden[:, None, :]
        elif token_hidden.ndim != 3:
            raise ValueError("token_hidden must have shape [H], [B, H], or [B, T, H]")
        if base_logits.ndim == 1:
            base_logits = base_logits[None, None, :]
        elif base_logits.ndim == 2:
            base_logits = base_logits[:, None, :]
        elif base_logits.ndim != 3:
            raise ValueError("base_logits must have shape [4], [B, 4], or [B, T, 4]")
        if token_hidden.shape[-1] != self.token_hidden_dim:
            raise ValueError("P8 token_hidden width mismatch")
        if base_logits.shape[-1] != ACTION_COUNT:
            raise ValueError("P8 preserves exactly four released action logits")
        if tuple(token_hidden.shape[:2]) != tuple(base_logits.shape[:2]):
            raise ValueError("token_hidden and base_logits batch/token shapes differ")
        device = self.residual_output.weight.device
        return (
            token_hidden.detach().to(device=device, dtype=torch.float32),
            base_logits.detach().to(device=device, dtype=torch.float32),
        )

    def encode_condition(
        self,
        decision_hidden: torch.Tensor,
        topology: P8TopologyTensors | Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        self._assert_fp32_parameters()
        if decision_hidden.ndim == 1:
            decision_hidden = decision_hidden[None, :]
        if decision_hidden.ndim != 2 or decision_hidden.shape[-1] != self.token_hidden_dim:
            raise ValueError("decision_hidden must have shape [B, H]")
        device = self.residual_output.weight.device
        decision_hidden = decision_hidden.detach().to(device=device, dtype=torch.float32)
        topology_outputs = self.topology_encoder(topology)
        if topology_outputs["context"].shape[0] != decision_hidden.shape[0]:
            raise ValueError("decision_hidden and topology batch sizes differ")
        decision_state = self.decision_projection(decision_hidden)
        fused = self.condition_fusion(
            torch.cat((decision_state, topology_outputs["context"]), dim=-1)
        )
        waypoint_mean = self.max_waypoint_m * torch.tanh(self.waypoint_head(fused))
        return {
            **topology_outputs,
            "condition": fused.float(),
            "waypoint_mean": waypoint_mean.float(),
            "intent_logits": self.intent_head(fused).float(),
            "values": self.value_head(fused).float().squeeze(-1),
        }

    def token_outputs(
        self,
        token_hidden: torch.Tensor,
        base_logits: torch.Tensor,
        condition: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        token_hidden, base_logits = self._normalize_token_inputs(token_hidden, base_logits)
        required = ("condition", "waypoint_mean")
        if any(name not in condition for name in required):
            raise KeyError("P8 condition is missing condition or waypoint_mean")
        batch, tokens = token_hidden.shape[:2]
        condition_state = condition["condition"]
        waypoint_mean = condition["waypoint_mean"]
        if tuple(condition_state.shape) != (batch, self.model_dim):
            raise ValueError("P8 condition batch shape mismatch")
        if tuple(waypoint_mean.shape) != (batch, 2):
            raise ValueError("P8 waypoint condition shape mismatch")
        token_state = self.token_projection(token_hidden)
        topology_state = condition_state[:, None, :].expand(-1, tokens, -1)
        waypoint_state = self.waypoint_projection(waypoint_mean)[:, None, :].expand(
            -1, tokens, -1
        )
        residual_state = self.residual_hidden(
            torch.cat((token_state, topology_state, waypoint_state), dim=-1)
        )
        residual_logits = self.residual_output(residual_state).float()
        policy_logits = base_logits + residual_logits
        return {
            "base_logits": base_logits,
            "residual_logits": residual_logits,
            "policy_logits": policy_logits,
            "policy_log_probs": policy_logits.log_softmax(dim=-1),
        }

    @staticmethod
    def _normalize_actions(
        actions: torch.Tensor,
        executed_mask: torch.Tensor,
        *,
        batch: int,
        tokens: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        def expand_rows(value: torch.Tensor, name: str) -> torch.Tensor:
            if value.ndim == 0:
                value = value.reshape(1, 1)
            elif value.ndim == 1:
                if tokens == 1 and value.shape[0] == batch:
                    value = value[:, None]
                elif batch == 1 and value.shape[0] == tokens:
                    value = value[None, :]
                else:
                    raise ValueError(f"ambiguous {name} shape for P8 token batch")
            if value.ndim != 2 or tuple(value.shape) != (batch, tokens):
                raise ValueError(f"{name} must have shape [B, T]")
            return value

        actions = expand_rows(actions, "actions").detach().to(device=device, dtype=torch.int64)
        executed_mask = expand_rows(executed_mask, "executed_mask").detach().to(
            device=device
        )
        if executed_mask.dtype != torch.bool:
            raise ValueError("executed_mask must be a bool tensor")
        if bool((executed_mask.sum(dim=1) == 0).any()):
            raise ValueError("each decision must execute at least one token")
        invalid = ((actions < 0) | (actions >= ACTION_COUNT)) & executed_mask
        if bool(invalid.any()):
            raise ValueError("executed action is outside the released four-action vocabulary")
        return actions, executed_mask

    @classmethod
    def joint_action_terms(
        cls,
        policy_logits: torch.Tensor,
        actions: torch.Tensor,
        executed_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if policy_logits.ndim != 3 or policy_logits.shape[-1] != ACTION_COUNT:
            raise ValueError("policy_logits must have shape [B, T, 4]")
        batch, tokens = policy_logits.shape[:2]
        actions, executed_mask = cls._normalize_actions(
            actions,
            executed_mask,
            batch=batch,
            tokens=tokens,
            device=policy_logits.device,
        )
        safe_actions = torch.where(executed_mask, actions, torch.zeros_like(actions))
        log_distribution = policy_logits.log_softmax(dim=-1)
        selected = log_distribution.gather(-1, safe_actions[..., None]).squeeze(-1)
        token_log_probs = torch.where(
            executed_mask, selected, torch.zeros_like(selected)
        )
        probabilities = log_distribution.exp()
        token_entropy = -(probabilities * log_distribution).sum(dim=-1)
        mask = executed_mask.to(dtype=token_entropy.dtype)
        counts = mask.sum(dim=-1)
        return {
            "token_log_probs": token_log_probs,
            "decision_log_probs": token_log_probs.sum(dim=-1),
            "token_entropy": torch.where(
                executed_mask, token_entropy, torch.zeros_like(token_entropy)
            ),
            "masked_mean_entropy": (token_entropy * mask).sum(dim=-1) / counts,
            "executed_token_count": counts.to(dtype=torch.int64),
        }

    def forward(
        self,
        token_hidden: torch.Tensor,
        base_logits: torch.Tensor,
        topology: P8TopologyTensors | Mapping[str, torch.Tensor],
        *,
        actions: torch.Tensor | None = None,
        executed_mask: torch.Tensor | None = None,
        decision_hidden: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        token_hidden, base_logits = self._normalize_token_inputs(token_hidden, base_logits)
        if decision_hidden is None:
            decision_hidden = token_hidden[:, 0, :]
        condition = self.encode_condition(decision_hidden, topology)
        token = self.token_outputs(token_hidden, base_logits, condition)
        outputs = {**condition, **token}
        if (actions is None) != (executed_mask is None):
            raise ValueError("actions and executed_mask must be supplied together")
        if actions is not None:
            terms = self.joint_action_terms(token["policy_logits"], actions, executed_mask)
            outputs.update(terms)
            outputs["log_probs"] = terms["decision_log_probs"]
            outputs["entropy"] = terms["masked_mean_entropy"]
        return outputs

    def residual_for_token(
        self,
        token_hidden: torch.Tensor,
        base_logits: torch.Tensor,
        topology: P8TopologyTensors | Mapping[str, torch.Tensor],
        *,
        condition: Mapping[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Return a shape-[4] residual suitable for OriginalStreamSession.next_token."""

        if token_hidden.ndim != 1 or base_logits.ndim != 1:
            raise ValueError("residual_for_token expects one token hidden and four base logits")
        if condition is None:
            condition = self.encode_condition(token_hidden, topology)
        outputs = self.token_outputs(token_hidden, base_logits, condition)
        return outputs["residual_logits"][0, 0]

    def decision_residual_callback(
        self,
        topology: P8TopologyTensors | Mapping[str, torch.Tensor],
    ):
        """Create a session callback with one condition fixed by the first token.

        A fresh callback must be created for every four-token decision.  This
        matches batched replay, whose default decision hidden is token zero.
        """

        cached_condition = None

        def residual(token_hidden: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
            nonlocal cached_condition
            if cached_condition is None:
                cached_condition = self.encode_condition(token_hidden, topology)
            return self.residual_for_token(
                token_hidden,
                base_logits,
                topology,
                condition=cached_condition,
            )

        return residual

    @staticmethod
    def auxiliary_loss(
        outputs: Mapping[str, torch.Tensor],
        *,
        target_intents: torch.Tensor,
        target_waypoints: torch.Tensor,
        waypoint_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute intent and waypoint supervision without waypoint likelihood."""

        logits = outputs["intent_logits"]
        waypoints = outputs["waypoint_mean"]
        target_intents = target_intents.to(
            device=logits.device, dtype=torch.int64
        ).reshape(-1)
        target_waypoints = target_waypoints.to(
            device=waypoints.device, dtype=torch.float32
        ).reshape(-1, 2)
        waypoint_mask = waypoint_mask.to(device=waypoints.device, dtype=torch.bool).reshape(-1)
        if logits.shape[0] != target_intents.shape[0] or waypoints.shape != target_waypoints.shape:
            raise ValueError("P8 auxiliary target batch shape mismatch")
        if bool(((target_intents < 0) | (target_intents >= INTENT_COUNT)).any()):
            raise ValueError("P8 intent target is outside navigate/stop/recover")
        intent_loss = F.cross_entropy(logits, target_intents)
        if bool(waypoint_mask.any()):
            waypoint_loss = F.smooth_l1_loss(
                waypoints[waypoint_mask], target_waypoints[waypoint_mask]
            )
        else:
            waypoint_loss = waypoints.sum() * 0.0
        return {
            "intent_loss": intent_loss,
            "waypoint_loss": waypoint_loss,
            "auxiliary_loss": intent_loss + waypoint_loss,
        }
