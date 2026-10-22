"""Executor-aligned topology and continuous-waypoint policy for StreamVLN G8."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from rlinf.models.embodiment.streamvln_topological_policy import (
        StreamVLNTopologicalPolicy,
        _cfg,
        canonical_parameter_name,
    )
except ImportError:  # Direct staged-file checks.
    from streamvln_topological_policy import (
        StreamVLNTopologicalPolicy,
        _cfg,
        canonical_parameter_name,
    )

try:
    from .streamvln_g8_action import (
        ACTION_COUNT,
        NAVIGATE_ACTION_COUNT,
        candidate_endpoint_matrix,
    )
    from .streamvln_topology import FEATURE_NAMES
except ImportError:
    from streamvln_g8_action import (
        ACTION_COUNT,
        NAVIGATE_ACTION_COUNT,
        candidate_endpoint_matrix,
    )
    from streamvln_topology import FEATURE_NAMES


G8_TRAINABLE_PREFIXES = (
    "state_projection.",
    "topology_encoder.",
    "fusion.",
    "waypoint_head.",
    "mode_head.",
    "candidate_head.",
    "value_head.",
)


class StreamVLNG8Policy(StreamVLNTopologicalPolicy):
    """Frozen original StreamVLN plus a replayable 28-way macro policy."""

    def __init__(
        self,
        *,
        alignment_coefficient: float = 4.0,
        **kwargs,
    ) -> None:
        if alignment_coefficient <= 0.0:
            raise ValueError("G8 alignment coefficient must be positive")
        head_checkpoint = kwargs.pop("head_checkpoint", None)
        super().__init__(**kwargs)
        self.alignment_coefficient = float(alignment_coefficient)
        self.candidate_head = nn.Linear(self.head_hidden, NAVIGATE_ACTION_COUNT)
        self.candidate_head.to(dtype=torch.bfloat16)
        with torch.no_grad():
            nn.init.zeros_(self.candidate_head.weight)
            nn.init.zeros_(self.candidate_head.bias)
        endpoints = torch.tensor(candidate_endpoint_matrix(), dtype=torch.float32)
        if tuple(endpoints.shape) != (NAVIGATE_ACTION_COUNT, 2):
            raise RuntimeError("G8 executable endpoint tensor shape changed")
        self.register_buffer("candidate_endpoints", endpoints, persistent=True)

        if head_checkpoint:
            from pathlib import Path
            from safetensors.torch import load_file as load_safetensors

            state = load_safetensors(
                Path(head_checkpoint).expanduser().resolve(strict=True),
                device="cpu",
            )
            self.load_high_level_state(state)

        # G8 has no Gaussian action likelihood.  The legacy scalar remains in
        # the checkpoint for compatibility but is not trainable or replayed.
        self.waypoint_log_std.requires_grad_(False)
        if not self.trainable_mode:
            self.candidate_head.requires_grad_(False)
            self.eval()
        self._assert_g8_contract()

    def _assert_g8_contract(self) -> None:
        trainable = {
            canonical_parameter_name(name)
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }
        unexpected = {
            name for name in trainable
            if not any(name.startswith(prefix) for prefix in G8_TRAINABLE_PREFIXES)
        }
        if unexpected:
            raise RuntimeError(
                "G8 has unexpected trainable parameters: "
                + ", ".join(sorted(unexpected))
            )
        if self.waypoint_log_std.requires_grad:
            raise RuntimeError("G8 Gaussian waypoint log standard deviation must be frozen")
        if self.trainable_mode:
            required = (
                "waypoint_head.",
                "mode_head.",
                "candidate_head.",
                "value_head.",
            )
            if any(not any(name.startswith(prefix) for name in trainable) for prefix in required):
                raise RuntimeError("G8 declared heads must all be trainable")

    def trainable_manifest(self) -> dict:
        manifest = super().trainable_manifest()
        trainable = {
            canonical_parameter_name(name): int(parameter.numel())
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }
        manifest.update({
            "action_schema": "streamvln_g8_executor_aligned_action_v1",
            "action_count": ACTION_COUNT,
            "navigate_candidate_count": NAVIGATE_ACTION_COUNT,
            "gaussian_waypoint_likelihood": False,
            "alignment_coefficient_per_m2": self.alignment_coefficient,
            "trainable_parameter_count": sum(trainable.values()),
            "trainable_names": sorted(trainable),
            "waypoint_log_std_trainable": False,
        })
        manifest["trainable_groups"] = {
            "high_level_policy": sum(
                count for name, count in trainable.items()
                if not name.startswith("value_head.")
            ),
            "value_head": sum(
                count for name, count in trainable.items()
                if name.startswith("value_head.")
            ),
        }
        return manifest

    def _g8_outputs(
        self,
        state_features: torch.Tensor,
        topology_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        outputs = super()._head_outputs(state_features, topology_features)
        fused = outputs["fused"]
        dtype = self.candidate_head.weight.dtype
        learned = self.candidate_head(fused.to(dtype=dtype)).float()
        means = outputs["waypoint_mean"].float()
        endpoints = self.candidate_endpoints.to(device=means.device, dtype=means.dtype)
        squared_distance = (means[:, None, :] - endpoints[None, :, :]).square().sum(-1)
        navigate_logits = (
            learned
            + outputs["mode_logits"][:, 0:1]
            - self.alignment_coefficient * squared_distance
        )
        action_logits = torch.cat(
            (
                navigate_logits,
                outputs["mode_logits"][:, 1:2],
                outputs["mode_logits"][:, 2:3],
            ),
            dim=-1,
        )
        if action_logits.shape[-1] != ACTION_COUNT:
            raise RuntimeError("G8 action logit width changed")
        return {
            **outputs,
            "candidate_prior_logits": learned,
            "candidate_squared_distance_m2": squared_distance,
            "action_logits": action_logits,
        }

    @staticmethod
    def _categorical_terms(
        action_logits: torch.Tensor,
        action_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action_ids = action_ids.to(dtype=torch.int64).reshape(-1)
        if (
            action_logits.ndim != 2
            or action_logits.shape[-1] != ACTION_COUNT
            or action_logits.shape[0] != action_ids.shape[0]
        ):
            raise ValueError("G8 action batch shape mismatch")
        if bool(((action_ids < 0) | (action_ids >= ACTION_COUNT)).any()):
            raise ValueError("G8 action id outside the frozen vocabulary")
        log_distribution = action_logits.log_softmax(-1)
        log_probs = log_distribution.gather(1, action_ids[:, None]).squeeze(1)
        probabilities = log_distribution.exp()
        entropy = -(probabilities * log_distribution).sum(-1)
        return log_probs, entropy

    @torch.no_grad()
    def sample_g8(
        self,
        state_features: torch.Tensor,
        topology_features: torch.Tensor,
        *,
        seed: int,
        deterministic: bool = False,
    ) -> dict[str, torch.Tensor]:
        outputs = self._g8_outputs(state_features, topology_features)
        if deterministic:
            action_ids = outputs["action_logits"].argmax(-1)
        else:
            generator = torch.Generator(device=outputs["action_logits"].device)
            generator.manual_seed(int(seed) % (2**63 - 1))
            action_ids = torch.multinomial(
                outputs["action_logits"].softmax(-1),
                1,
                generator=generator,
            ).squeeze(-1)
        log_probs, entropy = self._categorical_terms(outputs["action_logits"], action_ids)
        return {
            "action_ids": action_ids,
            "behavior_log_probs": log_probs,
            "entropy": entropy,
            "values": outputs["values"],
            "waypoint_mean": outputs["waypoint_mean"],
            "mode_logits": outputs["mode_logits"],
            "action_logits": outputs["action_logits"],
            "actor_version": torch.tensor(
                self.actor_version, device=outputs["values"].device
            ),
        }

    def replay_g8(
        self,
        state_features: torch.Tensor,
        topology_features: torch.Tensor,
        action_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        outputs = self._g8_outputs(state_features, topology_features)
        log_probs, entropy = self._categorical_terms(outputs["action_logits"], action_ids)
        return {
            "log_probs": log_probs,
            "entropy": entropy,
            "values": outputs["values"],
            "waypoint_mean": outputs["waypoint_mean"],
            "mode_logits": outputs["mode_logits"],
            "action_logits": outputs["action_logits"],
        }

    def bc_loss_g8(
        self,
        state_features: torch.Tensor,
        topology_features: torch.Tensor,
        target_action_ids: torch.Tensor,
        target_waypoints: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        outputs = self._g8_outputs(state_features, topology_features)
        targets = target_action_ids.to(dtype=torch.int64).reshape(-1)
        waypoints = target_waypoints.float().reshape(-1, 2)
        if targets.shape[0] != outputs["action_logits"].shape[0] or waypoints.shape[0] != targets.shape[0]:
            raise ValueError("G8 BC target batch shape mismatch")
        action_loss = F.cross_entropy(outputs["action_logits"], targets)
        active = targets < NAVIGATE_ACTION_COUNT
        waypoint_loss = (
            F.smooth_l1_loss(outputs["waypoint_mean"][active], waypoints[active])
            if bool(active.any())
            else outputs["waypoint_mean"].sum() * 0.0
        )
        loss = action_loss + 2.0 * waypoint_loss
        return {
            "loss": loss,
            "action_loss": action_loss,
            "waypoint_loss": waypoint_loss,
            "action_accuracy": (outputs["action_logits"].argmax(-1) == targets).float().mean(),
        }

    def forward(self, operation: str, **kwargs):
        if operation == "sample_g8":
            return self.sample_g8(
                kwargs["state_features"],
                kwargs["topology_features"],
                seed=int(kwargs["seed"]),
                deterministic=bool(kwargs.get("deterministic", False)),
            )
        if operation == "replay_g8":
            return self.replay_g8(
                kwargs["state_features"],
                kwargs["topology_features"],
                kwargs["action_ids"],
            )
        if operation == "bc_loss_g8":
            return self.bc_loss_g8(
                kwargs["state_features"],
                kwargs["topology_features"],
                kwargs["target_action_ids"],
                kwargs["target_waypoints"],
            )
        return super().forward(operation, **kwargs)


def get_model(cfg, torch_dtype=None) -> StreamVLNG8Policy:
    del torch_dtype
    return StreamVLNG8Policy(
        stream_source=str(_cfg(cfg, "stream_source")),
        base_model=str(_cfg(cfg, "base_model")),
        tokenizer=str(_cfg(cfg, "tokenizer")),
        vision_tower=str(_cfg(cfg, "vision_tower")),
        expected_source_commit=str(_cfg(cfg, "expected_source_commit", "")),
        head_checkpoint=_cfg(cfg, "head_checkpoint"),
        trainable=bool(_cfg(cfg, "trainable", True)),
        context_tokens=int(_cfg(cfg, "context_tokens", 4096)),
        seed=int(_cfg(cfg, "seed", 2026092808)),
        topology_dim=int(_cfg(cfg, "topology_dim", len(FEATURE_NAMES))),
        head_hidden=int(_cfg(cfg, "head_hidden", 256)),
        alignment_coefficient=float(_cfg(cfg, "alignment_coefficient", 4.0)),
    )
