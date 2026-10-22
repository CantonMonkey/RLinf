"""G6 high-level head on the frozen original StreamVLN representation."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from streamvln_topological_policy import StreamVLNTopologicalPolicy, _cfg
from streamvln_topology import FEATURE_NAMES, STOP_MODE


class StreamVLNRecoveryPolicy(StreamVLNTopologicalPolicy):
    """Checkpoint-compatible G5 head with balanced three-mode supervision."""

    def bc_loss(
        self,
        state_features: torch.Tensor,
        topology_features: torch.Tensor,
        target_modes: torch.Tensor,
        target_waypoints: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        outputs = self._head_outputs(state_features, topology_features)
        target_modes = target_modes.to(dtype=torch.int64).reshape(-1)
        target_waypoints = target_waypoints.float().reshape(-1, 2)
        mode_loss = F.cross_entropy(outputs["mode_logits"], target_modes)
        active = target_modes != STOP_MODE
        waypoint_loss = (
            F.smooth_l1_loss(outputs["waypoint_mean"][active], target_waypoints[active])
            if bool(active.any())
            else outputs["waypoint_mean"].sum() * 0.0
        )
        predicted = outputs["mode_logits"].argmax(-1)
        result = {
            "loss": mode_loss + 2.0 * waypoint_loss,
            "mode_loss": mode_loss,
            "waypoint_loss": waypoint_loss,
            "mode_accuracy": (predicted == target_modes).float().mean(),
        }
        for mode, name in enumerate(("navigate", "stop", "recover")):
            mask = target_modes == mode
            result[f"mode_accuracy_{name}"] = (
                (predicted[mask] == target_modes[mask]).float().mean()
                if bool(mask.any())
                else predicted.float().sum() * 0.0
            )
        return result

    def head_distribution(
        self,
        state_features: torch.Tensor,
        topology_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self._head_outputs(state_features, topology_features)

    def forward(self, operation: str, **kwargs):
        if operation == "bc_loss":
            return self.bc_loss(
                kwargs["state_features"],
                kwargs["topology_features"],
                kwargs["target_modes"],
                kwargs["target_waypoints"],
            )
        if operation == "head_distribution":
            return self.head_distribution(
                kwargs["state_features"], kwargs["topology_features"]
            )
        return super().forward(operation, **kwargs)


def get_model(cfg, torch_dtype=None) -> StreamVLNRecoveryPolicy:
    del torch_dtype
    return StreamVLNRecoveryPolicy(
        stream_source=str(_cfg(cfg, "stream_source")),
        base_model=str(_cfg(cfg, "base_model")),
        tokenizer=str(_cfg(cfg, "tokenizer")),
        vision_tower=str(_cfg(cfg, "vision_tower")),
        expected_source_commit=str(_cfg(cfg, "expected_source_commit", "")),
        head_checkpoint=_cfg(cfg, "head_checkpoint"),
        trainable=bool(_cfg(cfg, "trainable", True)),
        context_tokens=int(_cfg(cfg, "context_tokens", 4096)),
        seed=int(_cfg(cfg, "seed", 20260927)),
        topology_dim=int(_cfg(cfg, "topology_dim", len(FEATURE_NAMES))),
        head_hidden=int(_cfg(cfg, "head_hidden", 256)),
    )
