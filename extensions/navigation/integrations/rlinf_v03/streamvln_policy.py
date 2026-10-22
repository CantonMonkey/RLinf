"""RLinf-side client for the isolated frozen StreamVLN categorical actor.

The validated StreamVLN stack and RLinf v0.3 require incompatible Python and
Transformers versions.  This module therefore contains no StreamVLN imports:
it sends exact RGB/instruction episode observations to a long-lived actor
process and stores fixed-shape replay rows in RLinf trajectories.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
import select
import subprocess
import threading
import time

import torch
import torch.nn as nn

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType

try:
    from .streamvln_context import (
        ACTION_CHUNK,
        decode_fixed_utf8,
        snapshot_from_wire,
        snapshot_to_wire,
    )
    from .streamvln_replay import ReplaySpec, pack_snapshot, unpack_row
except ImportError:  # Direct staged-file contract tests.
    from streamvln_context import (
        ACTION_CHUNK,
        decode_fixed_utf8,
        snapshot_from_wire,
        snapshot_to_wire,
    )
    from streamvln_replay import ReplaySpec, pack_snapshot, unpack_row


WIRE_PREFIX = "STREAMRPC "
OBS_KEYS = frozenset({
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

REPLAY_TRAILING_RANK = {
    "env_id": 0,
    "episode_serial": 0,
    "step": 0,
    "block_start": 0,
    "instruction_utf8": 1,
    "instruction_utf8_mask": 1,
    "scene_utf8": 1,
    "scene_utf8_mask": 1,
    "episode_utf8": 1,
    "episode_utf8_mask": 1,
    "history_rgb": 2,
    "history_mask": 1,
    "block_rgb": 2,
    "block_mask": 1,
    "block_actions": 1,
    "block_action_mask": 1,
    "current_actions": 1,
    "current_action_mask": 1,
}


def _cfg(cfg, key, default=None):
    return cfg.get(key, default) if hasattr(cfg, "get") else getattr(cfg, key, default)


def _prefix_actions(values, mask) -> list[int]:
    values = list(values)
    mask = list(mask)
    if len(values) != ACTION_CHUNK or len(mask) != ACTION_CHUNK:
        raise ValueError("previous action tensors must have fixed width four")
    actions = []
    padded = False
    for value, valid in zip(values, mask):
        valid = bool(valid)
        value = int(value)
        if not 0 <= value < 4:
            raise ValueError("previous action outside 0..3")
        if valid:
            if padded:
                raise ValueError("previous action mask must be prefix-contiguous")
            actions.append(value)
        else:
            padded = True
            if value != 0:
                raise ValueError("previous action padding must be STOP/zero")
    return actions


class StreamVLNCategoricalPolicy(nn.Module, BasePolicy):
    """Frozen four-action StreamVLN policy connected through STREAMRPC."""

    def __init__(
        self,
        *,
        stream_python: str,
        stream_source: str,
        base_model: str,
        tokenizer: str,
        vision_tower: str,
        server_script: str | None = None,
        expected_source_commit: str = "",
        actor_device: str = "cuda:0",
        image_size: int = 64,
        instruction_bytes: int = 512,
        scene_bytes: int = 384,
        episode_bytes: int = 64,
        temperature: float = 1.0,
        context_tokens: int = 4096,
        seed: int = 20260905,
        startup_timeout_s: float = 300.0,
        request_timeout_s: float = 180.0,
        action_dim: int = 1,
        num_action_chunks: int = 4,
        discrete_n_actions: int = 4,
    ) -> None:
        super().__init__()
        if (action_dim, num_action_chunks, discrete_n_actions) != (1, 4, 4):
            raise ValueError("StreamVLN requires four categorical actions per decision")
        if image_size < 16 or temperature <= 0 or context_tokens < 1:
            raise ValueError("invalid StreamVLN image/sampling configuration")
        self.action_dim = 1
        self.num_action_chunks = 4
        self.discrete_n_actions = 4
        self.torch_compile_enabled = False
        self.cuda_graph_manager = None
        self.spec = ReplaySpec(
            height=int(image_size),
            width=int(image_size),
            instruction_bytes=int(instruction_bytes),
            scene_bytes=int(scene_bytes),
            episode_bytes=int(episode_bytes),
        )
        self.request_timeout_s = float(request_timeout_s)
        self._lock = threading.Lock()
        self._request_serial = 0
        self._closed = False
        self.register_buffer("_device_anchor", torch.zeros((), dtype=torch.float32), persistent=False)

        paths = {
            "stream_python": Path(stream_python).expanduser().resolve(strict=True),
            "stream_source": Path(stream_source).expanduser().resolve(strict=True),
            "base_model": Path(base_model).expanduser().resolve(strict=True),
            "tokenizer": Path(tokenizer).expanduser().resolve(strict=True),
            "vision_tower": Path(vision_tower).expanduser().resolve(strict=True),
            "server_script": Path(server_script).expanduser().resolve(strict=True)
            if server_script else Path(__file__).with_name("streamvln_policy_server.py").resolve(strict=True),
        }
        command = [
            str(paths["stream_python"]), "-B", str(paths["server_script"]),
            "--stream-source", str(paths["stream_source"]),
            "--base", str(paths["base_model"]),
            "--tokenizer", str(paths["tokenizer"]),
            "--vision-tower", str(paths["vision_tower"]),
            "--device", str(actor_device),
            "--temperature", str(float(temperature)),
            "--context-tokens", str(int(context_tokens)),
            "--seed", str(int(seed)),
        ]
        if expected_source_commit:
            command.extend(("--expected-source-commit", str(expected_source_commit)))
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )
        try:
            self.handshake = self._read_rpc(float(startup_timeout_s))
            if self.handshake.get("ready") is not True:
                raise RuntimeError("StreamVLN actor did not become ready: " + repr(self.handshake))
            if self.handshake.get("replay_differentiable") is not False:
                raise RuntimeError("G2 frozen actor declared an unexpected replay mode")
            if len(self.handshake.get("action_token_ids", ())) != 4:
                raise RuntimeError("StreamVLN actor did not expose four action tokens")
        except BaseException:
            self.close()
            raise

    def _read_rpc(self, timeout_s: float) -> dict:
        if self.process.stdout is None:
            raise RuntimeError("StreamVLN actor stdout is closed")
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("StreamVLN actor response timed out")
            available, _, _ = select.select([self.process.stdout], [], [], remaining)
            if not available:
                raise TimeoutError("StreamVLN actor response timed out")
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError(
                    "StreamVLN actor exited before responding, exit code "
                    + str(self.process.poll())
                )
            if not line.startswith(WIRE_PREFIX):
                continue
            response = json.loads(line[len(WIRE_PREFIX):])
            if not isinstance(response, dict):
                raise ValueError("StreamVLN actor response must be an object")
            return response

    def _rpc(self, operation: str, **payload) -> dict:
        with self._lock:
            if self._closed or self.process.stdin is None or self.process.poll() is not None:
                raise RuntimeError("StreamVLN actor is not running")
            self._request_serial += 1
            request_id = self._request_serial
            request = {"request_id": request_id, "operation": operation, **payload}
            self.process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
            self.process.stdin.flush()
            response = self._read_rpc(self.request_timeout_s)
            if response.get("request_id") != request_id:
                raise RuntimeError("StreamVLN actor response ID mismatch")
            if "error" in response:
                raise RuntimeError(
                    "StreamVLN actor request failed: " + str(response["error"])
                    + "\n" + str(response.get("traceback", ""))
                )
            return response

    def _check_observation(self, obs: dict[str, torch.Tensor]) -> int:
        if set(obs) != OBS_KEYS:
            raise ValueError(
                f"StreamVLN observation keys must be {sorted(OBS_KEYS)}; got {sorted(obs)}"
            )
        if any(not isinstance(value, torch.Tensor) for value in obs.values()):
            raise TypeError("StreamVLN observations must be tensors")
        rgb = obs["rgb"]
        if rgb.ndim != 4 or tuple(rgb.shape[1:]) != (self.spec.height, self.spec.width, 3):
            raise ValueError("StreamVLN RGB must be batched fixed-size HWC")
        if rgb.dtype != torch.uint8:
            raise TypeError("StreamVLN RGB must remain uint8")
        batch = int(rgb.shape[0])
        expected = {
            "instruction_utf8": (batch, self.spec.instruction_bytes),
            "instruction_utf8_mask": (batch, self.spec.instruction_bytes),
            "scene_utf8": (batch, self.spec.scene_bytes),
            "scene_utf8_mask": (batch, self.spec.scene_bytes),
            "episode_utf8": (batch, self.spec.episode_bytes),
            "episode_utf8_mask": (batch, self.spec.episode_bytes),
            "previous_actions": (batch, 4),
            "previous_action_mask": (batch, 4),
        }
        for key, shape in expected.items():
            if tuple(obs[key].shape) != shape:
                raise ValueError(f"{key} has shape {tuple(obs[key].shape)}, expected {shape}")
        for key in ("env_id", "episode_serial", "episode_step", "episode_start", "terminal_frozen"):
            if tuple(obs[key].shape) != (batch,):
                raise ValueError(f"{key} must be a batch vector")
        for key in ("instruction_utf8_mask", "scene_utf8_mask", "episode_utf8_mask", "previous_action_mask", "episode_start", "terminal_frozen"):
            if obs[key].dtype != torch.bool:
                raise TypeError(f"{key} must be bool")
        for key in ("env_id", "episode_serial", "episode_step", "previous_actions"):
            if obs[key].dtype != torch.int64:
                raise TypeError(f"{key} must be int64")
        for key in ("instruction_utf8", "scene_utf8", "episode_utf8"):
            if obs[key].dtype != torch.uint8:
                raise TypeError(f"{key} must be uint8")
        if bool(obs["terminal_frozen"].any()):
            raise ValueError("terminal-frozen frames cannot request another policy decision")
        return batch

    @staticmethod
    def _cpu_list(tensor: torch.Tensor, index: int):
        return tensor[index].detach().cpu().tolist()

    def _wire_observation(self, obs, index: int) -> dict:
        rgb = obs["rgb"][index].detach().cpu().contiguous()
        instruction = decode_fixed_utf8(
            self._cpu_list(obs["instruction_utf8"], index),
            [bool(value) for value in self._cpu_list(obs["instruction_utf8_mask"], index)],
        )
        scene = decode_fixed_utf8(
            self._cpu_list(obs["scene_utf8"], index),
            [bool(value) for value in self._cpu_list(obs["scene_utf8_mask"], index)],
        )
        episode = decode_fixed_utf8(
            self._cpu_list(obs["episode_utf8"], index),
            [bool(value) for value in self._cpu_list(obs["episode_utf8_mask"], index)],
        )
        previous = _prefix_actions(
            self._cpu_list(obs["previous_actions"], index),
            self._cpu_list(obs["previous_action_mask"], index),
        )
        return {
            "env_id": int(obs["env_id"][index]),
            "episode_serial": int(obs["episode_serial"][index]),
            "scene": scene,
            "episode_id": episode,
            "instruction": instruction,
            "image_shape": list(rgb.shape),
            "rgb_b64": base64.b64encode(rgb.numpy().tobytes()).decode("ascii"),
            "step": int(obs["episode_step"][index]),
            "episode_start": bool(obs["episode_start"][index]),
            "previous_actions": previous,
        }

    @staticmethod
    def _tensor_row(row: dict) -> dict[str, torch.Tensor]:
        result = {}
        for key, value in row.items():
            if key.endswith("_mask"):
                dtype = torch.bool
            elif key in ("history_rgb", "block_rgb") or key.endswith("_utf8"):
                dtype = torch.uint8
            else:
                dtype = torch.int64
            result[key] = torch.tensor(value, dtype=dtype)
        return result

    @staticmethod
    def _stack_tensor_rows(rows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        keys = set(rows[0])
        if any(set(row) != keys for row in rows):
            raise ValueError("StreamVLN replay tensor schemas differ")
        return {key: torch.stack([row[key] for row in rows], dim=0) for key in rows[0]}

    @torch.inference_mode()
    def predict_action_batch(
        self,
        env_obs: dict[str, torch.Tensor],
        calculate_logprobs: bool = True,
        calculate_values: bool = True,
        return_obs: bool = True,
        mode: str = "train",
        **_kwargs,
    ):
        if not calculate_logprobs or not return_obs:
            raise ValueError("StreamVLN rollout requires behavior logprobs and replay rows")
        if mode not in ("train", "eval"):
            raise ValueError("StreamVLN mode must be train or eval")
        batch = self._check_observation(env_obs)
        response = self._rpc(
            "predict",
            mode=mode,
            observations=[self._wire_observation(env_obs, index) for index in range(batch)],
        )
        results = response.get("results")
        if not isinstance(results, list) or len(results) != batch:
            raise RuntimeError("StreamVLN actor returned the wrong batch size")

        actions = torch.zeros(batch, 4, 1, dtype=torch.int64)
        logprobs = torch.zeros(batch, 4, dtype=torch.float32)
        action_mask = torch.zeros(batch, 4, dtype=torch.bool)
        replay_rows = []
        for index, result in enumerate(results):
            proposed = list(result["actions"])
            behavior = list(result["behavior_log_probs"])
            if not 1 <= len(proposed) <= 4 or len(behavior) != len(proposed):
                raise RuntimeError("StreamVLN actor returned an invalid action chunk")
            actions[index, : len(proposed), 0] = torch.tensor(proposed, dtype=torch.int64)
            logprobs[index, : len(proposed)] = torch.tensor(behavior, dtype=torch.float32)
            action_mask[index, : len(proposed)] = True
            snapshot = snapshot_from_wire(result["replay"])
            replay_rows.append(self._tensor_row(pack_snapshot(snapshot, self.spec)))
        forward_inputs = self._stack_tensor_rows(replay_rows)
        forward_inputs["action"] = actions[..., 0].clone()
        forward_inputs["model_action"] = actions[..., 0].clone()
        forward_inputs["action_mask"] = action_mask
        values = torch.zeros_like(logprobs) if calculate_values else torch.zeros_like(logprobs)
        return actions, {
            "prev_logprobs": logprobs,
            "prev_values": values,
            "forward_inputs": forward_inputs,
        }

    def _restore_rows(self, forward_inputs: dict[str, torch.Tensor]) -> list[dict]:
        missing = set(REPLAY_TRAILING_RANK) - set(forward_inputs)
        if missing:
            raise ValueError("StreamVLN replay fields are missing: " + ", ".join(sorted(missing)))
        scalar = forward_inputs["env_id"]
        count = scalar.numel()
        columns = {}
        for key, trailing_rank in REPLAY_TRAILING_RANK.items():
            tensor = forward_inputs[key].detach().cpu()
            tail = tuple(tensor.shape[-trailing_rank:]) if trailing_rank else ()
            columns[key] = tensor.reshape((count,) + tail).tolist()
        rows = []
        for index in range(count):
            packed = {key: columns[key][index] for key in REPLAY_TRAILING_RANK}
            rows.append(snapshot_to_wire(unpack_row(packed, self.spec)))
        return rows

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type != ForwardType.DEFAULT:
            raise NotImplementedError("G2 StreamVLN supports frozen categorical replay only")
        return self.default_forward(**kwargs)

    def default_forward(
        self,
        forward_inputs: dict[str, torch.Tensor],
        compute_logprobs: bool = True,
        compute_entropy: bool = True,
        compute_values: bool = True,
        **_kwargs,
    ) -> dict[str, torch.Tensor]:
        if not compute_logprobs:
            raise ValueError("StreamVLN replay must compute current logprobs")
        rows = self._restore_rows(forward_inputs)
        response = self._rpc("replay", rows=rows)
        results = response.get("results")
        if not isinstance(results, list) or len(results) != len(rows):
            raise RuntimeError("StreamVLN replay returned the wrong batch size")
        logprobs = torch.zeros(len(rows), 4, dtype=torch.float32)
        entropy = torch.zeros_like(logprobs)
        for index, result in enumerate(results):
            current = list(result["log_probs"])
            current_entropy = list(result["entropy"])
            if not 1 <= len(current) <= 4 or len(current_entropy) != len(current):
                raise RuntimeError("StreamVLN replay returned an invalid action likelihood")
            logprobs[index, : len(current)] = torch.tensor(current, dtype=torch.float32)
            entropy[index, : len(current)] = torch.tensor(current_entropy, dtype=torch.float32)
        output = {"logprobs": logprobs}
        if compute_entropy:
            output["entropy"] = entropy
        if compute_values:
            output["values"] = torch.zeros_like(logprobs)
        return output

    def close(self) -> None:
        if getattr(self, "_closed", True):
            return
        self._closed = True
        process = getattr(self, "process", None)
        if process is None:
            return
        try:
            if process.poll() is None and process.stdin is not None:
                self._request_serial += 1
                process.stdin.write(json.dumps({
                    "request_id": self._request_serial,
                    "operation": "close",
                }) + "\n")
                process.stdin.flush()
                process.wait(timeout=10)
        except (BrokenPipeError, TimeoutError, subprocess.TimeoutExpired):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        finally:
            if process.stdin is not None:
                process.stdin.close()
            if process.stdout is not None:
                process.stdout.close()
            self.process = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def get_model(cfg, torch_dtype=None) -> StreamVLNCategoricalPolicy:
    del torch_dtype
    return StreamVLNCategoricalPolicy(
        stream_python=str(_cfg(cfg, "stream_python")),
        stream_source=str(_cfg(cfg, "stream_source")),
        base_model=str(_cfg(cfg, "base_model")),
        tokenizer=str(_cfg(cfg, "tokenizer")),
        vision_tower=str(_cfg(cfg, "vision_tower")),
        server_script=_cfg(cfg, "server_script"),
        expected_source_commit=str(_cfg(cfg, "expected_source_commit", "")),
        actor_device=str(_cfg(cfg, "actor_device", "cuda:0")),
        image_size=int(_cfg(cfg, "image_size", 64)),
        instruction_bytes=int(_cfg(cfg, "instruction_bytes", 512)),
        scene_bytes=int(_cfg(cfg, "scene_bytes", 384)),
        episode_bytes=int(_cfg(cfg, "episode_bytes", 64)),
        temperature=float(_cfg(cfg, "temperature", 1.0)),
        context_tokens=int(_cfg(cfg, "context_tokens", 4096)),
        seed=int(_cfg(cfg, "seed", 20260905)),
        startup_timeout_s=float(_cfg(cfg, "startup_timeout_s", 300)),
        request_timeout_s=float(_cfg(cfg, "request_timeout_s", 180)),
        action_dim=int(_cfg(cfg, "action_dim", 1)),
        num_action_chunks=int(_cfg(cfg, "num_action_chunks", 4)),
        discrete_n_actions=int(_cfg(cfg, "discrete_n_actions", 4)),
    )
