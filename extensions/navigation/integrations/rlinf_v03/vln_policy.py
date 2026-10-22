"""Small RGB-and-instruction categorical policy for the R2R PPO smoke.

The only policy observations are a Habitat RGB frame and R2R dataset token
IDs/mask. Goal, pose, path and evaluation metrics are deliberately excluded.
This file is staged into RLinf v0.3 as the ``vln_policy`` model implementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType


OBS_KEYS = frozenset({"rgb", "instruction_ids", "instruction_mask"})


class VLNPolicy(nn.Module, BasePolicy):
    """One-frame visual encoder and order-sensitive instruction GRU."""

    def __init__(
        self,
        vocab_size: int,
        max_instruction_tokens: int = 200,
        image_size: int = 64,
        action_dim: int = 1,
        num_action_chunks: int = 1,
        discrete_n_actions: int = 4,
        add_value_head: bool = True,
    ) -> None:
        super().__init__()
        if vocab_size < 2:
            raise ValueError("vocab_size must include PAD and real R2R tokens")
        if max_instruction_tokens < 1 or image_size < 16:
            raise ValueError("Invalid instruction length or image encoder size")
        if (action_dim, num_action_chunks, discrete_n_actions) != (1, 1, 4):
            raise ValueError("R2R discrete policy requires one of four actions per step")
        if not add_value_head:
            raise ValueError("R2R PPO/GAE requires a value head")

        self.vocab_size = int(vocab_size)
        self.max_instruction_tokens = int(max_instruction_tokens)
        self.image_size = int(image_size)
        self.action_dim = 1
        self.num_action_chunks = 1
        self.discrete_n_actions = 4
        self.torch_compile_enabled = False
        self.cuda_graph_manager = None

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
        # PAD_INDEX=0 belongs to the official R2R vocabulary. The mask, rather
        # than the token value alone, identifies padding for the sequence.
        self.token_embedding = nn.Embedding(self.vocab_size, 64)
        self.instruction_encoder = nn.GRU(64, 128, batch_first=True)
        self.fusion = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )
        self.actor_logits = nn.Linear(128, 4)
        self.value_head = nn.Linear(128, 1)

    def _check_observation(self, obs: dict[str, torch.Tensor]) -> None:
        unexpected = set(obs) - OBS_KEYS
        missing = OBS_KEYS - set(obs)
        if missing or unexpected:
            raise ValueError(
                f"R2R policy observation keys must be {sorted(OBS_KEYS)}; "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        rgb = obs["rgb"]
        ids = obs["instruction_ids"]
        mask = obs["instruction_mask"]
        if not all(isinstance(x, torch.Tensor) for x in (rgb, ids, mask)):
            raise TypeError("R2R policy observations must be tensors")
        if rgb.ndim != 4 or rgb.shape[-1] != 3 or rgb.shape[1] < 16 or rgb.shape[2] < 16:
            raise ValueError("rgb must be batched HWC with three channels")
        if rgb.dtype != torch.uint8:
            raise TypeError("rgb must remain uint8 in PPO trajectories")
        if ids.ndim != 2 or ids.shape[1] != self.max_instruction_tokens:
            raise ValueError("instruction_ids must have the configured fixed length")
        if ids.dtype != torch.int64 or mask.dtype != torch.bool or mask.shape != ids.shape:
            raise TypeError("instruction_ids/mask require int64/bool matching tensors")
        if rgb.shape[0] != ids.shape[0]:
            raise ValueError("RGB and instruction batch sizes differ")

    def preprocess_env_obs(self, env_obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self._check_observation(env_obs)
        device = next(self.parameters()).device
        return {key: env_obs[key].to(device) for key in OBS_KEYS}

    def _features(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        rgb = obs["rgb"].permute(0, 3, 1, 2).float().div_(255.0)
        if rgb.shape[-2:] != (self.image_size, self.image_size):
            rgb = F.interpolate(
                rgb,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        visual = self.visual_encoder(rgb)

        mask = obs["instruction_mask"]
        lengths = mask.sum(dim=1)
        if not bool(torch.all(lengths > 0)):
            raise ValueError("R2R instruction must contain at least one real token")
        tokens = self.token_embedding(obs["instruction_ids"])
        # Padding follows every real token. Taking the last real GRU output
        # preserves word order and ignores subsequent padding updates.
        encoded, _ = self.instruction_encoder(tokens)
        row = torch.arange(encoded.shape[0], device=encoded.device)
        language = encoded[row, lengths - 1]
        return self.fusion(torch.cat((visual, language), dim=-1))

    def logits_and_values(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Expose both heads for deterministic diagnostics and PPO replay."""
        features = self._features(obs)
        return self.actor_logits(features).float(), self.value_head(features).float()

    def forward(
        self,
        forward_type=ForwardType.DEFAULT,
        operation: str | None = None,
        **kwargs,
    ):
        if forward_type != ForwardType.DEFAULT:
            raise NotImplementedError("The R2R policy currently supports PPO only")
        # ``operation`` keeps every parameter-dependent stack call behind the
        # registered module root.  RLinf/FSDP callers pass ``manager.model`` as
        # that root; the legacy worker path omits it and remains unchanged.
        if operation is None or operation == "replay":
            return self.default_forward(**kwargs)
        if operation == "sample":
            return self.predict_action_batch(
                env_obs=kwargs["env_obs"],
                mode=kwargs.get("mode", "train"),
            )
        if operation == "bootstrap":
            obs = self.preprocess_env_obs(kwargs["env_obs"])
            _, values = self.logits_and_values(obs)
            return values
        raise ValueError(f"unknown VLN policy operation {operation!r}")

    def default_forward(
        self,
        forward_inputs: dict[str, torch.Tensor],
        compute_logprobs: bool = True,
        compute_entropy: bool = True,
        compute_values: bool = True,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        obs = {key: forward_inputs[key] for key in OBS_KEYS}
        self._check_observation(obs)
        if "action" not in forward_inputs:
            raise ValueError("PPO replay requires the sampled action")
        logits, values = self.logits_and_values(obs)
        probs = Categorical(logits=logits)
        action = forward_inputs["action"].reshape(-1).long().clone()
        if action.numel() != logits.shape[0]:
            raise ValueError("Action batch size differs from observation batch")
        output: dict[str, torch.Tensor] = {}
        if compute_logprobs:
            output["logprobs"] = probs.log_prob(action).unsqueeze(-1)
        if compute_entropy:
            output["entropy"] = probs.entropy().unsqueeze(-1)
        if compute_values:
            output["values"] = values
        return output

    @torch.inference_mode()
    def predict_action_batch(
        self,
        env_obs: dict[str, torch.Tensor],
        calculate_logprobs: bool = True,
        calculate_values: bool = True,
        return_obs: bool = True,
        mode: str = "train",
        **kwargs,
    ) -> tuple[torch.Tensor, dict]:
        if not calculate_logprobs or not return_obs:
            raise ValueError("R2R PPO needs behavior logprobs and replay observations")
        obs = self.preprocess_env_obs(env_obs)
        logits, values = self.logits_and_values(obs)
        probs = Categorical(logits=logits)
        if mode == "train":
            index = probs.sample()
        elif mode == "eval":
            index = logits.argmax(dim=-1)
        else:
            raise ValueError(f"Unknown R2R policy mode: {mode}")
        action = index.unsqueeze(-1)
        logprobs = probs.log_prob(index.clone()).unsqueeze(-1)
        if not calculate_values:
            values = torch.zeros_like(logprobs)
        forward_inputs = {key: obs[key] for key in OBS_KEYS}
        forward_inputs.update(action=action, model_action=action)
        return action.reshape(-1, 1, 1), {
            "prev_logprobs": logprobs,
            "prev_values": values,
            "forward_inputs": forward_inputs,
        }


def get_model(cfg, torch_dtype=None) -> VLNPolicy:
    """RLinf model-registry builder; parameter dtype follows its actor setup."""
    del torch_dtype
    return VLNPolicy(
        vocab_size=int(cfg.vocab_size),
        max_instruction_tokens=int(cfg.get("max_instruction_tokens", 200)),
        image_size=int(cfg.get("image_size", 64)),
        action_dim=int(cfg.get("action_dim", 1)),
        num_action_chunks=int(cfg.get("num_action_chunks", 1)),
        discrete_n_actions=int(cfg.get("discrete_n_actions", 4)),
        add_value_head=bool(cfg.get("add_value_head", True)),
    )
