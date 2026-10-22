"""Frozen original StreamVLN with a trainable topological waypoint head for G5."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors

try:
    from rlinf.models.embodiment.streamvln_ppo_policy import (
        ACTION_CHUNK,
        StreamVLNPPOPolicy,
        _cfg,
    )
except ImportError:  # Direct staged-file checks.
    from streamvln_ppo_policy import (
        ACTION_CHUNK,
        StreamVLNPPOPolicy,
        _cfg,
    )
from streamvln_topology import FEATURE_NAMES, NAVIGATE_MODE


HEAD_PREFIXES = (
    "state_projection.",
    "topology_encoder.",
    "fusion.",
    "waypoint_head.",
    "mode_head.",
    "candidate_head.",
    "value_head.",
)
HEAD_SCALARS = ("waypoint_log_std",)


def canonical_parameter_name(name: str) -> str:
    return name.replace("_fsdp_wrapped_module.", "")


class StreamVLNTopologicalPolicy(StreamVLNPPOPolicy):
    """Frozen 8B backbone plus a small waypoint/mode/value policy."""

    def __init__(
        self,
        *,
        stream_source: str,
        base_model: str,
        tokenizer: str,
        vision_tower: str,
        expected_source_commit: str = "",
        head_checkpoint: str | None = None,
        trainable: bool = True,
        context_tokens: int = 4096,
        seed: int = 20260926,
        topology_dim: int = len(FEATURE_NAMES),
        head_hidden: int = 256,
    ) -> None:
        super().__init__(
            stream_source=stream_source,
            base_model=base_model,
            tokenizer=tokenizer,
            vision_tower=vision_tower,
            expected_source_commit=expected_source_commit,
            adapter_checkpoint=None,
            trainable=False,
            temperature=1.0,
            context_tokens=context_tokens,
            seed=seed,
            action_dim=1,
            num_action_chunks=4,
            discrete_n_actions=4,
            deterministic_eval=False,
        )
        if topology_dim != len(FEATURE_NAMES) or head_hidden < 32:
            raise ValueError("invalid topology feature or head width contract")
        self.topology_dim = int(topology_dim)
        self.head_hidden = int(head_hidden)
        self.trainable_mode = bool(trainable)
        hidden_size = int(self.value_head.in_features)

        self.state_projection = nn.Sequential(
            nn.Linear(hidden_size, head_hidden),
            nn.GELU(),
            nn.LayerNorm(head_hidden),
        )
        self.topology_encoder = nn.Sequential(
            nn.Linear(topology_dim, 64),
            nn.GELU(),
            nn.LayerNorm(64),
        )
        self.fusion = nn.Sequential(
            nn.Linear(head_hidden + 64, head_hidden),
            nn.GELU(),
            nn.LayerNorm(head_hidden),
        )
        self.waypoint_head = nn.Linear(head_hidden, 2)
        self.mode_head = nn.Linear(head_hidden, 3)
        self.value_head = nn.Linear(head_hidden, 1)
        self.waypoint_log_std = nn.Parameter(torch.tensor([-1.5, -1.5]))
        self._initialize_head_biases()
        for module in (
            self.state_projection,
            self.topology_encoder,
            self.fusion,
            self.waypoint_head,
            self.mode_head,
            self.value_head,
        ):
            module.to(dtype=torch.bfloat16)
        self.waypoint_log_std.data = self.waypoint_log_std.data.to(torch.bfloat16)

        self.actor_version = 0
        self._open_state_step = None
        if head_checkpoint:
            state = load_safetensors(
                Path(head_checkpoint).expanduser().resolve(strict=True),
                device="cpu",
            )
            self.load_high_level_state(state)
        if not trainable:
            self.requires_grad_(False)
            self.eval()
        self._assert_high_level_contract()

    def _initialize_head_biases(self) -> None:
        with torch.no_grad():
            self.waypoint_head.bias.copy_(torch.tensor([0.55, 0.0]))
            self.mode_head.bias.copy_(torch.tensor([1.5, -2.0, -2.0]))

    def _assert_high_level_contract(self) -> None:
        trainable = {
            canonical_parameter_name(name)
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }
        allowed = {
            name
            for name in (canonical_parameter_name(item) for item, _ in self.named_parameters())
            if name in HEAD_SCALARS or any(name.startswith(prefix) for prefix in HEAD_PREFIXES)
        }
        if trainable - allowed:
            raise RuntimeError(
                "G5 has unexpected trainable parameters: "
                + ", ".join(sorted(trainable - allowed))
            )
        if self.trainable_mode and (
            not any(name.startswith("waypoint_head.") for name in trainable)
            or not any(name.startswith("mode_head.") for name in trainable)
            or not any(name.startswith("value_head.") for name in trainable)
        ):
            raise RuntimeError("G5 waypoint, mode, and value heads must all be trainable")
        forbidden = [
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad and any(
                marker in name
                for marker in ("backbone.", "vision_tower", "mm_projector", "lm_head")
            )
        ]
        if forbidden:
            raise RuntimeError("G5 original StreamVLN backbone is not frozen")

    def high_level_state(self) -> dict[str, torch.Tensor]:
        state = {}
        for raw_name, parameter in self.named_parameters():
            name = canonical_parameter_name(raw_name)
            if name in HEAD_SCALARS or any(name.startswith(prefix) for prefix in HEAD_PREFIXES):
                if name in state:
                    raise RuntimeError(f"duplicate G5 checkpoint key {name}")
                state[name] = parameter.detach().cpu().contiguous()
        if not state:
            raise RuntimeError("G5 high-level checkpoint is empty")
        return state

    def load_high_level_state(self, state: dict[str, torch.Tensor]) -> None:
        parameters = {
            canonical_parameter_name(name): parameter
            for name, parameter in self.named_parameters()
        }
        expected = {
            name
            for name in parameters
            if name in HEAD_SCALARS or any(name.startswith(prefix) for prefix in HEAD_PREFIXES)
        }
        if set(state) != expected:
            missing = sorted(expected - set(state))
            extra = sorted(set(state) - expected)
            raise RuntimeError(f"G5 checkpoint key mismatch; missing={missing}, extra={extra}")
        with torch.no_grad():
            for name, tensor in state.items():
                target = parameters[name]
                if tuple(target.shape) != tuple(tensor.shape):
                    raise RuntimeError(f"G5 checkpoint shape mismatch for {name}")
                target.copy_(tensor.to(device=target.device, dtype=target.dtype))

    def trainable_manifest(self) -> dict:
        trainable = {
            canonical_parameter_name(name): int(parameter.numel())
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }
        actor_count = sum(
            count for name, count in trainable.items()
            if not name.startswith("value_head.")
        )
        value_count = sum(
            count for name, count in trainable.items()
            if name.startswith("value_head.")
        )
        return {
            "source_kind": "upstream-original",
            "memory_adapter": False,
            "secondary_post_training_used": False,
            "backbone_trainable": False,
            "topology_feature_names": list(FEATURE_NAMES),
            "topology_dim": self.topology_dim,
            "trainable_parameter_count": sum(trainable.values()),
            "trainable_groups": {
                "high_level_policy": actor_count,
                "value_head": value_count,
            },
            "trainable_names": sorted(trainable),
        }

    @torch.inference_mode()
    def encode_state(
        self,
        obs: dict[str, torch.Tensor],
        *,
        terminal_bootstrap: bool = False,
    ) -> torch.Tensor:
        session = self._ensure_session()
        scene, episode, instruction = self._identity(obs)
        task_key = scene + "::" + episode
        step = int(self._row(obs, "episode_step").item())
        episode_start = bool(self._row(obs, "episode_start").item())
        terminal_frozen = bool(self._row(obs, "terminal_frozen").item())
        if step < 0 or step % ACTION_CHUNK:
            raise ValueError("G5 state encoding requires a four-step boundary")
        if terminal_bootstrap and not terminal_frozen:
            raise ValueError("terminal bootstrap requires a frozen terminal observation")
        if terminal_frozen and not terminal_bootstrap:
            raise ValueError("live state encoding rejects frozen terminal observations")
        if step == 0:
            if not episode_start:
                raise ValueError("G5 step zero must be an episode start")
            session.begin_task(task_key, instruction)
            self._open_state_step = None
        else:
            if episode_start or session.scene != task_key or session.instruction != instruction:
                raise ValueError("G5 episode identity changed within a rollout")
            if self._open_state_step is None:
                raise RuntimeError("G5 continued state has no open previous decision")
            previous = self._row(obs, "previous_actions")
            previous_mask = self._row(obs, "previous_action_mask").bool()
            executed = [int(value) for value in previous[previous_mask].detach().cpu().tolist()]
            if len(executed) != ACTION_CHUNK:
                raise ValueError("G5 nonterminal decisions must execute four primitive actions")
            if step != len(session.actions) + len(executed):
                raise ValueError("G5 physical step disagrees with action history")
            for action in executed:
                session.append_action_token(int(self.action_token_ids[action]))
            session.actions.extend(executed)
            session.close_decision()
        rgb = self._row(obs, "rgb").detach().cpu().numpy().astype(np.uint8)
        feature = session.encode_observation(rgb)
        session.prepare_observation(step, feature)
        hidden = session._hidden(torch.cat(session.parts))[-1]
        # A time-limit bootstrap consumes the final observation but does not
        # open another policy decision.  The following episode must reset the
        # StreamVLN session before any further live encoding.
        self._open_state_step = None if terminal_bootstrap else step
        return hidden.detach().float()

    def _head_outputs(
        self,
        state_features: torch.Tensor,
        topology_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if state_features.ndim == 1:
            state_features = state_features.unsqueeze(0)
        if topology_features.ndim == 1:
            topology_features = topology_features.unsqueeze(0)
        if (
            state_features.ndim != 2
            or topology_features.ndim != 2
            or state_features.shape[0] != topology_features.shape[0]
            or topology_features.shape[1] != self.topology_dim
        ):
            raise ValueError("G5 head input shape mismatch")
        dtype = self.state_projection[0].weight.dtype
        state = self.state_projection(state_features.to(dtype=dtype))
        topology = self.topology_encoder(topology_features.to(dtype=dtype))
        fused = self.fusion(torch.cat((state, topology), dim=-1))
        waypoint_mean = 1.25 * torch.tanh(self.waypoint_head(fused).float())
        mode_logits = self.mode_head(fused).float()
        values = self.value_head(fused).float().squeeze(-1)
        log_std = self.waypoint_log_std.float().clamp(-3.0, -0.25)
        return {
            "waypoint_mean": waypoint_mean,
            "mode_logits": mode_logits,
            "values": values,
            "waypoint_log_std": log_std.expand_as(waypoint_mean),
            "fused": fused.float(),
        }

    @staticmethod
    def _distribution_terms(
        outputs: dict[str, torch.Tensor],
        modes: torch.Tensor,
        waypoints: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        modes = modes.to(dtype=torch.int64).reshape(-1)
        waypoints = waypoints.float().reshape(-1, 2)
        logits = outputs["mode_logits"]
        means = outputs["waypoint_mean"]
        log_std = outputs["waypoint_log_std"]
        if logits.shape[0] != modes.shape[0] or means.shape != waypoints.shape:
            raise ValueError("G5 sampled action batch shape mismatch")
        mode_log_probs = logits.log_softmax(-1).gather(1, modes[:, None]).squeeze(1)
        mode_probs = logits.softmax(-1)
        mode_entropy = -(mode_probs * logits.log_softmax(-1)).sum(-1)
        std = log_std.exp()
        waypoint_log_probs = (
            -0.5 * ((waypoints - means) / std).square()
            - log_std
            - 0.5 * torch.log(torch.tensor(2.0 * torch.pi, device=means.device))
        ).sum(-1)
        waypoint_entropy = (
            log_std + 0.5 * (1.0 + torch.log(torch.tensor(2.0 * torch.pi, device=means.device)))
        ).sum(-1)
        active = (modes == NAVIGATE_MODE).float()
        return (
            mode_log_probs + active * waypoint_log_probs,
            mode_entropy + active * waypoint_entropy,
        )

    @torch.no_grad()
    def sample_head(
        self,
        state_features: torch.Tensor,
        topology_features: torch.Tensor,
        *,
        seed: int,
        deterministic: bool = False,
    ) -> dict[str, torch.Tensor]:
        outputs = self._head_outputs(state_features, topology_features)
        generator = torch.Generator(device=outputs["mode_logits"].device)
        generator.manual_seed(int(seed) % (2**63 - 1))
        if deterministic:
            modes = outputs["mode_logits"].argmax(-1)
            waypoints = outputs["waypoint_mean"]
        else:
            modes = torch.multinomial(
                outputs["mode_logits"].softmax(-1), 1, generator=generator
            ).squeeze(-1)
            noise = torch.randn(
                outputs["waypoint_mean"].shape,
                device=outputs["waypoint_mean"].device,
                generator=generator,
            )
            waypoints = (
                outputs["waypoint_mean"] + noise * outputs["waypoint_log_std"].exp()
            )
        log_probs, entropy = self._distribution_terms(outputs, modes, waypoints)
        return {
            "modes": modes,
            "waypoints": waypoints,
            "behavior_log_probs": log_probs,
            "entropy": entropy,
            "values": outputs["values"],
            "waypoint_mean": outputs["waypoint_mean"],
            "mode_logits": outputs["mode_logits"],
            "actor_version": torch.tensor(
                self.actor_version, device=outputs["values"].device
            ),
        }

    def replay_head(
        self,
        state_features: torch.Tensor,
        topology_features: torch.Tensor,
        modes: torch.Tensor,
        waypoints: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        outputs = self._head_outputs(state_features, topology_features)
        log_probs, entropy = self._distribution_terms(outputs, modes, waypoints)
        return {
            "log_probs": log_probs,
            "entropy": entropy,
            "values": outputs["values"],
            "waypoint_mean": outputs["waypoint_mean"],
            "mode_logits": outputs["mode_logits"],
        }

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
        class_weights = torch.tensor(
            [1.0, 4.0, 0.25],
            dtype=torch.float32,
            device=outputs["mode_logits"].device,
        )
        mode_loss = F.cross_entropy(
            outputs["mode_logits"], target_modes, weight=class_weights
        )
        active = target_modes == NAVIGATE_MODE
        waypoint_loss = (
            F.smooth_l1_loss(
                outputs["waypoint_mean"][active],
                target_waypoints[active],
            )
            if bool(active.any())
            else outputs["waypoint_mean"].sum() * 0.0
        )
        loss = mode_loss + 2.0 * waypoint_loss
        accuracy = (outputs["mode_logits"].argmax(-1) == target_modes).float().mean()
        return {
            "loss": loss,
            "mode_loss": mode_loss,
            "waypoint_loss": waypoint_loss,
            "mode_accuracy": accuracy,
        }

    def mark_updated(self) -> None:
        self.actor_version += 1

    def forward(self, operation: str, **kwargs):
        if operation == "encode_state":
            return self.encode_state(kwargs["obs"])
        if operation == "encode_bootstrap_state":
            return self.encode_state(kwargs["obs"], terminal_bootstrap=True)
        if operation == "sample_head":
            return self.sample_head(
                kwargs["state_features"],
                kwargs["topology_features"],
                seed=int(kwargs["seed"]),
                deterministic=bool(kwargs.get("deterministic", False)),
            )
        if operation == "replay_head":
            return self.replay_head(
                kwargs["state_features"],
                kwargs["topology_features"],
                kwargs["modes"],
                kwargs["waypoints"],
            )
        if operation == "bc_loss":
            return self.bc_loss(
                kwargs["state_features"],
                kwargs["topology_features"],
                kwargs["target_modes"],
                kwargs["target_waypoints"],
            )
        raise ValueError(f"unknown StreamVLN G5 operation {operation!r}")


def get_model(cfg, torch_dtype=None) -> StreamVLNTopologicalPolicy:
    del torch_dtype
    return StreamVLNTopologicalPolicy(
        stream_source=str(_cfg(cfg, "stream_source")),
        base_model=str(_cfg(cfg, "base_model")),
        tokenizer=str(_cfg(cfg, "tokenizer")),
        vision_tower=str(_cfg(cfg, "vision_tower")),
        expected_source_commit=str(_cfg(cfg, "expected_source_commit", "")),
        head_checkpoint=_cfg(cfg, "head_checkpoint"),
        trainable=bool(_cfg(cfg, "trainable", True)),
        context_tokens=int(_cfg(cfg, "context_tokens", 4096)),
        seed=int(_cfg(cfg, "seed", 20260926)),
        topology_dim=int(_cfg(cfg, "topology_dim", len(FEATURE_NAMES))),
        head_hidden=int(_cfg(cfg, "head_hidden", 256)),
    )
