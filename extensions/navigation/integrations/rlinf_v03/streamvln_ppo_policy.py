"""In-process differentiable StreamVLN categorical policy for RLinf FSDP G3."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import subprocess
import sys

import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoConfig, AutoTokenizer

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType


ACTION_NAMES = ("STOP", "↑", "←", "→")
ACTION_CHUNK = 4


def _cfg(cfg, key, default=None):
    return cfg.get(key, default) if hasattr(cfg, "get") else getattr(cfg, key, default)


def _decode_fixed(values: torch.Tensor, mask: torch.Tensor) -> str:
    payload = bytes(
        int(value)
        for value, valid in zip(
            values.detach().cpu().tolist(), mask.detach().cpu().tolist()
        )
        if bool(valid)
    )
    text = payload.decode("utf-8")
    if not text:
        raise ValueError("StreamVLN identity text must be non-empty")
    return text


def _git_head(path: Path) -> str:
    result = subprocess.run(
        [
            "git", "-c", f"safe.directory={path}",
            "-C", str(path), "rev-parse", "HEAD",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=15,
    )
    if result.returncode:
        raise RuntimeError(f"cannot resolve StreamVLN source commit at {path}")
    return result.stdout.strip()


class OriginalStreamSession:
    """Causal four-action session over the unmodified upstream StreamVLN model."""

    image_marker = "<image>"
    memory_marker = "<memory>"

    def __init__(self, model, tokenizer, temperature: float, context_tokens: int):
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = float(temperature)
        self.context_tokens = int(context_tokens)
        ids = [tokenizer.encode(name, add_special_tokens=False) for name in ACTION_NAMES]
        if any(len(value) != 1 for value in ids):
            raise ValueError("the four original actions must be single tokens")
        self.action_ids = torch.tensor(
            [value[0] for value in ids], dtype=torch.int64, device=self.device
        )
        self.id_to_action = {int(token): i for i, token in enumerate(self.action_ids)}
        self.scene = None
        self.instruction = None
        self.features = {}
        self.actions = []
        self.parts = []
        self.predictor_positions = []
        self.value_positions = []
        self.sampled_ids = []
        self.block_start = None
        self.memory_prefix = None

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    def amp(self):
        return torch.autocast(
            self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"
        )

    def begin_task(self, scene: str, instruction: str):
        self.scene, self.instruction = scene, instruction
        self.features, self.actions, self.parts = {}, [], []
        self.predictor_positions, self.value_positions, self.sampled_ids = [], [], []
        self.block_start = None
        self.memory_prefix = None

    @torch.no_grad()
    def encode_observation(self, rgb):
        from PIL import Image

        if not isinstance(rgb, Image.Image):
            rgb = Image.fromarray(np.asarray(rgb)).convert("RGB")
        processor = self.model.get_vision_tower().image_processor
        pixels = processor.preprocess(images=rgb, return_tensors="pt")["pixel_values"]
        with self.amp():
            features = self.model.encode_images(pixels.to(self.device, self.dtype))
            return self.model.get_2dPool(features, 2)[0].detach()

    def _text(self, text: str):
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        token_ids = torch.tensor(ids, dtype=torch.int64, device=self.device)
        return self.model.get_model().embed_tokens(token_ids)

    def _append_prompt(self, content: str, image: torch.Tensor):
        content = content.replace(self.memory_marker, "\0M\0").replace(
            self.image_marker, "\0I\0"
        )
        self.parts.append(self._text("<|im_start|>user\n"))
        for piece in content.split("\0"):
            if piece == "M":
                if self.memory_prefix is None:
                    raise ValueError("missing original StreamVLN history prefix")
                self.parts.append(self.memory_prefix)
            elif piece == "I":
                self.parts.append(image)
            elif piece:
                self.parts.append(self._text(piece))
        self.parts.append(self._text("<|im_end|>\n<|im_start|>assistant\n"))

    def prepare_observation(self, step: int, feature: torch.Tensor):
        if step != len(self.actions) or step % ACTION_CHUNK:
            raise ValueError("observations must align to four-action chunks")
        self.features[step] = feature.to(self.device, self.dtype)
        if step % 32 == 0:
            self.block_start = step
            self.parts, self.predictor_positions = [], []
            self.value_positions, self.sampled_ids = [], []
            self.memory_prefix = None
            if step:
                history_steps = list(range(0, step, step // 8))
                self.memory_prefix = torch.cat([self.features[i] for i in history_steps])
            self.parts.append(
                self._text(
                    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                )
            )
            content = (
                "You are an autonomous navigation assistant. Your task is to "
                + self.instruction
                + " Devise an action sequence to follow the instruction using the four actions: "
                "TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, MOVE FORWARD (↑) "
                "by 25 centimeters, or STOP."
            )
            if step:
                content += " These are your historical observations: <memory>."
            content += " you can see <image>."
        else:
            content = "you can see <image>."
        self._append_prompt(content, self.features[step])
        self.value_positions.append(sum(part.shape[0] for part in self.parts) - 1)

    def _hidden(self, embeddings: torch.Tensor):
        length = int(embeddings.shape[0])
        if length > self.context_tokens:
            raise ValueError(
                f"original StreamVLN block length {length} exceeds {self.context_tokens}"
            )
        embeddings = torch.nn.functional.pad(
            embeddings, (0, 0, 0, self.context_tokens - length)
        )
        with self.amp():
            output = self.model.get_model()(
                inputs_embeds=embeddings.unsqueeze(0),
                use_cache=False,
                return_dict=True,
            )
        return output.last_hidden_state[0, :length]

    def _action_logits(self, hidden: torch.Tensor):
        with torch.autocast(self.device.type, enabled=False):
            rows = self.model.lm_head.weight[self.action_ids].float()
            return torch.nn.functional.linear(hidden.float(), rows) / self.temperature

    def append_action_token(self, token: int):
        if token not in self.id_to_action:
            raise ValueError("token is outside the declared original action set")
        self.predictor_positions.append(sum(part.shape[0] for part in self.parts) - 1)
        self.sampled_ids.append(int(token))
        token_ids = torch.tensor([token], dtype=torch.int64, device=self.device)
        self.parts.append(self.model.get_model().embed_tokens(token_ids))

    def close_decision(self):
        self.parts.append(self._text("<|im_end|>\n"))

    def open_decision(self, step: int, feature: torch.Tensor):
        self.prepare_observation(step, feature)

    def next_token(
        self,
        generator,
        *,
        greedy: bool = False,
        residual=None,
    ):
        hidden = self._hidden(torch.cat(self.parts))[-1]
        base_logits = self._action_logits(hidden)
        if callable(residual):
            residual = residual(hidden.detach().float(), base_logits.detach().float())
        if residual is None:
            categorical = base_logits.log_softmax(-1)
        else:
            if not isinstance(residual, torch.Tensor):
                raise TypeError("action-logit residual must be a tensor or callable")
            if residual.shape != base_logits.shape:
                raise ValueError("action-logit residual must match the four base logits")
            policy_logits = base_logits.float() + residual.to(
                device=base_logits.device, dtype=torch.float32
            )
            categorical = policy_logits.log_softmax(-1)
        action = (
            int(categorical.argmax())
            if greedy
            else int(torch.multinomial(categorical.exp(), 1, generator=generator))
        )
        token = int(self.action_ids[action])
        return {
            "token": token,
            "action": action,
            "log_prob": float(categorical[action]),
            "token_hidden": hidden.detach().float(),
            "base_logits": base_logits.detach().float(),
        }

    def commit_token(self, token: int):
        self.append_action_token(token)

    def finish_decision(self):
        self.close_decision()

    @torch.no_grad()
    def sample(
        self, step: int, feature: torch.Tensor, generator, *, greedy: bool = False
    ):
        self.open_decision(step, feature)
        tokens, actions, log_probs = [], [], []
        for _ in range(ACTION_CHUNK):
            sampled = self.next_token(generator, greedy=greedy)
            token = sampled["token"]
            action = sampled["action"]
            tokens.append(token)
            actions.append(action)
            log_probs.append(sampled["log_prob"])
            self.commit_token(token)
            if action == 0:
                break
        self.finish_decision()
        return {
            "step": step,
            "sampled_tokens": tokens,
            "proposed_actions": actions,
            "behavior_log_probs": log_probs,
            "temperature": self.temperature,
            "greedy": bool(greedy),
        }

    def acknowledge(self, proposed_actions, executed_actions):
        if list(proposed_actions[: len(executed_actions)]) != list(executed_actions):
            raise ValueError("executed actions disagree with original policy proposal")
        self.actions.extend(int(item) for item in executed_actions)


@torch.no_grad()
def sample_pre_refactor_release_reference(
    session: OriginalStreamSession,
    step: int,
    feature: torch.Tensor,
    generator,
    *,
    greedy: bool = False,
):
    """Frozen copy of the committed pre-P8 sampler used as Gate A's oracle.

    This deliberately does not call ``open_decision``, ``next_token``,
    ``commit_token`` or ``finish_decision``.  It preserves the sampler body
    from commit 5434ab0 so the refactored path cannot validate itself.
    """

    session.prepare_observation(step, feature)
    tokens, actions, log_probs, base_logits_rows = [], [], [], []
    for _ in range(ACTION_CHUNK):
        hidden = session._hidden(torch.cat(session.parts))[-1]
        base_logits = session._action_logits(hidden)
        categorical = base_logits.log_softmax(-1)
        action = (
            int(categorical.argmax())
            if greedy
            else int(torch.multinomial(categorical.exp(), 1, generator=generator))
        )
        token = int(session.action_ids[action])
        tokens.append(token)
        actions.append(action)
        log_probs.append(float(categorical[action]))
        base_logits_rows.append(base_logits.detach().float().cpu().clone())
        session.append_action_token(token)
        if action == 0:
            break
    session.close_decision()
    return {
        "step": int(step),
        "sampled_tokens": tokens,
        "proposed_actions": actions,
        "behavior_log_probs": log_probs,
        "base_logits": base_logits_rows,
        "temperature": session.temperature,
        "greedy": bool(greedy),
        "reference_commit": "5434ab0",
    }


class StreamVLNPPOPolicy(nn.Module, BasePolicy):
    """Frozen StreamVLN base plus a small language LoRA and state-value head.

    The policy is loaded in the RLinf actor process. No RPC boundary appears in
    the replay path, so PPO gradients reach the listed Qwen LoRA parameters.
    """

    _no_split_modules = ["Qwen2DecoderLayer"]

    def __init__(
        self,
        *,
        stream_source: str,
        base_model: str,
        tokenizer: str,
        vision_tower: str,
        expected_source_commit: str = "",
        adapter_checkpoint: str | None = None,
        trainable: bool = True,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_layers=(24, 25, 26, 27),
        lora_targets=("q_proj", "v_proj"),
        temperature: float = 1.0,
        context_tokens: int = 512,
        seed: int = 20260926,
        action_dim: int = 1,
        num_action_chunks: int = 4,
        discrete_n_actions: int = 4,
        deterministic_eval: bool = False,
    ) -> None:
        super().__init__()
        if (action_dim, num_action_chunks, discrete_n_actions) != (1, 4, 4):
            raise ValueError("StreamVLN PPO requires four categorical actions")
        if temperature <= 0 or context_tokens < 256:
            raise ValueError("invalid StreamVLN temperature or context")
        self.action_dim = 1
        self.num_action_chunks = 4
        self.discrete_n_actions = 4
        self.torch_compile_enabled = False
        self.cuda_graph_manager = None
        self.temperature = float(temperature)
        self.context_tokens = int(context_tokens)
        self.seed = int(seed)
        self.behavior_version = 0
        self.actor_version = 0
        self.lora_layers = tuple(int(item) for item in lora_layers)
        self.lora_targets = tuple(str(item) for item in lora_targets)
        self.trainable_mode = bool(trainable)
        self.deterministic_eval = bool(deterministic_eval)

        source = Path(stream_source).expanduser().resolve(strict=True)
        base_path = Path(base_model).expanduser().resolve(strict=True)
        tokenizer_path = Path(tokenizer).expanduser().resolve(strict=True)
        vision_path = Path(vision_tower).expanduser().resolve(strict=True)
        if expected_source_commit:
            actual = _git_head(source)
            if actual != expected_source_commit:
                raise RuntimeError(
                    f"StreamVLN source commit {actual} differs from {expected_source_commit}"
                )
        sys.path[:0] = [str(source / "streamvln"), str(source)]
        # Transformers 4.57 moved these legacy BERT helpers. Exporting them
        # before importing upstream LLaVA keeps the pristine source worktree
        # unchanged and does not alter model computation.
        import transformers.modeling_utils as modeling_utils
        from transformers import pytorch_utils

        for helper_name in (
            "apply_chunking_to_forward",
            "find_pruneable_heads_and_indices",
            "prune_linear_layer",
        ):
            if not hasattr(modeling_utils, helper_name):
                setattr(modeling_utils, helper_name, getattr(pytorch_utils, helper_name))
        from model.stream_video_vln import StreamVLNForCausalLM

        config = AutoConfig.from_pretrained(base_path, local_files_only=True)
        config.mm_vision_tower = config.vision_tower = str(vision_path)
        config.num_history = 8
        config.use_cache = False
        if not hasattr(config, "layer_types"):
            config.layer_types = ["full_attention"] * int(config.num_hidden_layers)
        base = StreamVLNForCausalLM.from_pretrained(
            base_path,
            config=config,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        tower = base.get_vision_tower()
        if not tower.is_loaded:
            tower.load_model()
        base.requires_grad_(False)
        base.eval()

        if adapter_checkpoint:
            checkpoint = Path(adapter_checkpoint).expanduser().resolve(strict=True)
            self.backbone = PeftModel.from_pretrained(
                base,
                checkpoint,
                is_trainable=bool(trainable),
                local_files_only=True,
            )
            self.source_kind = "upstream-plus-explicit-adapter"
        elif not trainable:
            # P8's frozen prior is the released model itself.  A zero LoRA is
            # mathematically inert, but constructing one would still add a
            # randomly initialized secondary module to the object whose
            # identity Gate A is meant to preserve.
            self.backbone = base
            self.source_kind = "upstream-original"
        else:
            layer_pattern = "|".join(str(layer) for layer in self.lora_layers)
            target_pattern = "|".join(self.lora_targets)
            language_only_targets = (
                rf"^model\.layers\.({layer_pattern})\.self_attn\."
                rf"({target_pattern})$"
            )
            lora = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=not bool(trainable),
                r=int(lora_rank),
                lora_alpha=int(lora_alpha),
                lora_dropout=0.0,
                bias="none",
                target_modules=language_only_targets,
            )
            self.backbone = get_peft_model(base, lora)
            self.source_kind = "upstream-plus-trainable-lora"
        # A single RLinf FSDP root requires one flattened parameter dtype.
        # Keep the frozen original backbone and the small trainable additions
        # in BF16; logits and PPO scalars are promoted to FP32 explicitly.
        self.backbone.to(dtype=torch.bfloat16)
        hidden_size = int(config.hidden_size)
        self.value_head = nn.Linear(hidden_size, 1, bias=True, dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, local_files_only=True
        )
        self._navigation_cls = OriginalStreamSession
        self._session = None
        self._sample = None

        action_ids = [
            self.tokenizer.encode(name, add_special_tokens=False)
            for name in ACTION_NAMES
        ]
        if any(len(ids) != 1 for ids in action_ids):
            raise ValueError("all four declared actions must be single tokens")
        self.action_token_ids = tuple(ids[0] for ids in action_ids)

        if not trainable:
            self.requires_grad_(False)
            self.eval()
        self._assert_trainable_contract()

    def _assert_trainable_contract(self) -> None:
        unexpected = [
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
            and "lora_" not in name
            and not name.startswith("value_head.")
        ]
        if unexpected:
            raise RuntimeError("unexpected trainable parameters: " + ", ".join(unexpected))
        if self.trainable_mode:
            trainable = [name for name, p in self.named_parameters() if p.requires_grad]
            if not any("lora_" in name for name in trainable):
                raise RuntimeError("StreamVLN PPO has no trainable LoRA parameters")
            if not any(name.startswith("value_head.") for name in trainable):
                raise RuntimeError("StreamVLN PPO has no trainable value head")
        forbidden = [
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
            and any(
                marker in name
                for marker in ("vision_tower", "mm_projector", "embed_tokens", "lm_head")
            )
        ]
        if forbidden:
            raise RuntimeError("frozen base contract violated: " + ", ".join(forbidden))

    def trainable_manifest(self) -> dict:
        trainable = {
            name: int(parameter.numel())
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }
        groups = {
            "lora": sum(count for name, count in trainable.items() if "lora_" in name),
            "value_head": sum(
                count for name, count in trainable.items()
                if name.startswith("value_head.")
            ),
        }
        return {
            "lora_layers": list(self.lora_layers),
            "lora_targets": list(self.lora_targets),
            "trainable_parameter_count": sum(trainable.values()),
            "trainable_groups": groups,
            "trainable_names": sorted(trainable),
            "frozen_base_parameter_count": sum(
                parameter.numel()
                for parameter in self.parameters()
                if not parameter.requires_grad
            ),
            "source_kind": self.source_kind,
            "memory_adapter": False,
            "attention_implementation": "sdpa",
        }

    @property
    def device(self):
        return next(self.backbone.parameters()).device

    def _ensure_session(self):
        if self._session is None or self._session.device != self.device:
            self._session = self._navigation_cls(
                self.backbone,
                self.tokenizer,
                temperature=self.temperature,
                context_tokens=self.context_tokens,
            )
        return self._session

    @staticmethod
    def _row(obs: dict[str, torch.Tensor], key: str) -> torch.Tensor:
        value = obs[key]
        if value.shape[0] != 1:
            raise ValueError("G3 bounded policy currently requires one environment")
        return value[0]

    def _identity(self, obs):
        instruction = _decode_fixed(
            self._row(obs, "instruction_utf8"),
            self._row(obs, "instruction_utf8_mask"),
        )
        scene = _decode_fixed(
            self._row(obs, "scene_utf8"),
            self._row(obs, "scene_utf8_mask"),
        )
        episode = _decode_fixed(
            self._row(obs, "episode_utf8"),
            self._row(obs, "episode_utf8_mask"),
        )
        return scene, episode, instruction

    @torch.no_grad()
    def sample_observation(self, obs: dict[str, torch.Tensor]) -> dict:
        session = self._ensure_session()
        scene, episode, instruction = self._identity(obs)
        task_key = scene + "::" + episode
        step = int(self._row(obs, "episode_step").item())
        episode_start = bool(self._row(obs, "episode_start").item())
        terminal_frozen = bool(self._row(obs, "terminal_frozen").item())
        if terminal_frozen or step < 0 or step % ACTION_CHUNK:
            raise ValueError("StreamVLN sampling requires a live four-step boundary")
        if step == 0:
            if not episode_start:
                raise ValueError("step zero must be marked as an episode start")
            session.begin_task(task_key, instruction)
            self._sample = None
        else:
            if episode_start or session.scene != task_key or session.instruction != instruction:
                raise ValueError("StreamVLN episode identity changed within a rollout")
            if self._sample is None:
                raise RuntimeError("continued sampling has no previous proposal")
            previous = self._row(obs, "previous_actions")
            previous_mask = self._row(obs, "previous_action_mask").bool()
            executed = previous[previous_mask].detach().cpu().tolist()
            if step != len(session.actions) + len(executed):
                raise ValueError("StreamVLN physical step disagrees with action history")
            session.acknowledge(self._sample["proposed_actions"], executed)
        rgb = self._row(obs, "rgb").detach().cpu().numpy().astype(np.uint8)
        feature = session.encode_observation(rgb)
        generator = torch.Generator(device=self.device)
        task_seed = int.from_bytes(
            hashlib.sha256(task_key.encode("utf-8")).digest()[:8], "little"
        )
        generator.manual_seed((self.seed ^ task_seed ^ step) % (2**63 - 1))
        sample = session.sample(
            step,
            feature,
            generator=generator,
            greedy=self.deterministic_eval,
        )
        count = len(sample["proposed_actions"])
        actions = torch.zeros(ACTION_CHUNK, dtype=torch.int64, device=self.device)
        mask = torch.zeros(ACTION_CHUNK, dtype=torch.bool, device=self.device)
        old = torch.zeros(ACTION_CHUNK, dtype=torch.float32, device=self.device)
        actions[:count] = torch.tensor(
            sample["proposed_actions"], dtype=torch.int64, device=self.device
        )
        mask[:count] = True
        old[:count] = torch.tensor(
            sample["behavior_log_probs"], dtype=torch.float32, device=self.device
        )
        self._sample = dict(sample)
        self.behavior_version = self.actor_version
        return {
            "actions": actions,
            "executed_mask": mask,
            "behavior_log_probs": old,
            "sampled_tokens": torch.tensor(
                sample["sampled_tokens"], dtype=torch.int64, device=self.device
            ),
            "behavior_version": torch.tensor(
                self.behavior_version, dtype=torch.int64, device=self.device
            ),
        }

    @torch.no_grad()
    def restore_observation(
        self,
        obs: dict[str, torch.Tensor],
        sampled_tokens: torch.Tensor,
        proposed_actions: torch.Tensor,
    ) -> None:
        """Rebuild a frozen teacher-forced decision in a fresh process."""
        session = self._ensure_session()
        scene, episode, instruction = self._identity(obs)
        step = int(self._row(obs, "episode_step").item())
        if step != 0:
            raise ValueError("independent G3 restore expects the initial observation")
        rgb = self._row(obs, "rgb").detach().cpu().numpy().astype(np.uint8)
        session.begin_task(scene + "::" + episode, instruction)
        feature = session.encode_observation(rgb)
        session.prepare_observation(step, feature)
        tokens = [int(item) for item in sampled_tokens.detach().cpu().tolist()]
        actions = [int(item) for item in proposed_actions.detach().cpu().tolist()]
        if not 1 <= len(tokens) <= ACTION_CHUNK or len(tokens) != len(actions):
            raise ValueError("restored action/token sequences are inconsistent")
        for token in tokens:
            session.append_action_token(token)
        session.close_decision()
        self._sample = {
            "step": step,
            "sampled_tokens": tokens,
            "proposed_actions": actions,
            "behavior_log_probs": [],
            "temperature": self.temperature,
        }

    def replay_current(self) -> dict[str, torch.Tensor]:
        if self._sample is None:
            raise RuntimeError("sample_observation must run before replay")
        session = self._ensure_session()
        hidden = session._hidden(torch.cat(session.parts))
        positions = torch.tensor(
            session.predictor_positions, dtype=torch.int64, device=self.device
        )
        selected_hidden = hidden.index_select(0, positions)
        logits = session._action_logits(selected_hidden)
        all_log_probs = logits.log_softmax(-1)
        labels = torch.tensor(
            [session.id_to_action[token] for token in session.sampled_ids],
            dtype=torch.int64,
            device=self.device,
        )
        selected = all_log_probs.gather(1, labels[:, None]).squeeze(1)
        entropy = -(all_log_probs.exp() * all_log_probs).sum(-1)
        value_positions = torch.tensor(
            session.value_positions, dtype=torch.int64, device=self.device
        )
        value_hidden = hidden.index_select(0, value_positions)
        state_values = self.value_head(
            value_hidden.to(self.value_head.weight.dtype)
        ).float().squeeze(-1)
        return {
            "log_probs": selected,
            "entropy": entropy,
            "state_value": state_values[-1],
            "state_values": state_values,
            "actor_version": torch.tensor(
                self.actor_version, dtype=torch.int64, device=self.device
            ),
        }

    @torch.no_grad()
    def bootstrap_observation(
        self,
        obs: dict[str, torch.Tensor],
        executed_actions: torch.Tensor,
        terminal: bool,
    ) -> torch.Tensor:
        if terminal:
            return torch.zeros((), dtype=torch.float32, device=self.device)
        session = self._ensure_session()
        executed = [int(item) for item in executed_actions.detach().cpu().tolist()]
        if len(executed) != ACTION_CHUNK:
            raise ValueError("nonterminal chunks must execute all four primitive actions")
        step = int(self._row(obs, "episode_step").item())
        if step != len(session.actions) + len(executed):
            raise ValueError("StreamVLN bootstrap step disagrees with action history")
        session.acknowledge(self._sample["proposed_actions"], executed)
        if step != len(session.actions):
            raise RuntimeError("StreamVLN bootstrap acknowledgement is incomplete")
        rgb = self._row(obs, "rgb").detach().cpu().numpy().astype(np.uint8)
        feature = session.encode_observation(rgb)
        session.prepare_observation(step, feature)
        hidden = session._hidden(torch.cat(session.parts))[-1]
        return self.value_head(hidden.to(self.value_head.weight.dtype)).float().squeeze(-1)

    def mark_updated(self) -> None:
        self.actor_version += 1

    def predict_action_batch(self, *args, **kwargs):
        raise NotImplementedError(
            "G3 uses the bounded decision-boundary runner; distributed rollout "
            "state transfer is a later milestone"
        )

    def default_forward(self, *args, **kwargs):
        raise NotImplementedError(
            "G3 replay must remain inside the registered RLinf FSDP actor"
        )

    def forward(self, operation: str = "replay", forward_type=ForwardType.DEFAULT, **kwargs):
        del forward_type
        if operation == "sample":
            return self.sample_observation(kwargs["obs"])
        if operation == "restore":
            self.restore_observation(
                kwargs["obs"], kwargs["sampled_tokens"], kwargs["proposed_actions"]
            )
            return {}
        if operation == "replay":
            return self.replay_current()
        if operation == "bootstrap":
            return self.bootstrap_observation(
                kwargs["obs"], kwargs["executed_actions"], bool(kwargs["terminal"])
            )
        raise ValueError(f"unknown StreamVLN PPO operation {operation!r}")


def get_model(cfg, torch_dtype=None) -> StreamVLNPPOPolicy:
    del torch_dtype
    layers = _cfg(cfg, "lora_layers", [24, 25, 26, 27])
    targets = _cfg(cfg, "lora_targets", ["q_proj", "v_proj"])
    return StreamVLNPPOPolicy(
        stream_source=str(_cfg(cfg, "stream_source")),
        base_model=str(_cfg(cfg, "base_model")),
        tokenizer=str(_cfg(cfg, "tokenizer")),
        vision_tower=str(_cfg(cfg, "vision_tower")),
        expected_source_commit=str(_cfg(cfg, "expected_source_commit", "")),
        adapter_checkpoint=_cfg(cfg, "adapter_checkpoint"),
        trainable=bool(_cfg(cfg, "trainable", True)),
        lora_rank=int(_cfg(cfg, "lora_rank", 8)),
        lora_alpha=int(_cfg(cfg, "lora_alpha", 16)),
        lora_layers=tuple(int(item) for item in layers),
        lora_targets=tuple(str(item) for item in targets),
        temperature=float(_cfg(cfg, "temperature", 1.0)),
        context_tokens=int(_cfg(cfg, "context_tokens", 512)),
        seed=int(_cfg(cfg, "seed", 20260926)),
        action_dim=int(_cfg(cfg, "action_dim", 1)),
        num_action_chunks=int(_cfg(cfg, "num_action_chunks", 4)),
        discrete_n_actions=int(_cfg(cfg, "discrete_n_actions", 4)),
        deterministic_eval=bool(_cfg(cfg, "deterministic_eval", False)),
    )
