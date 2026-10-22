"""Four-action RLinf environment contract for StreamVLN over Habitat R2R."""

from __future__ import annotations

import numpy as np
import torch
try:
    import gymnasium as gym
except ModuleNotFoundError as exc:  # Pinned StreamVLN Python 3.9 ships Gym 0.23.
    if exc.name != "gymnasium":
        raise
    import gym

if __package__:
    from rlinf.envs.vln.vln_env import VLNEnv
    from .streamvln_executor_contract import chunk_feedback, primitive_execution
    from .streamvln_p8_official_frame import (
        OFFICIAL_FRAME_SENSOR_CONTRACT,
        OfficialFrame,
        official_frame_contract_sha256,
    )
else:  # Explicit direct-file mode for local contract tests only.
    from vln_env import VLNEnv
    from streamvln_executor_contract import chunk_feedback, primitive_execution
    from streamvln_p8_official_frame import (
        OFFICIAL_FRAME_SENSOR_CONTRACT,
        OfficialFrame,
        official_frame_contract_sha256,
    )


ACTION_CHUNK = 4
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


def encode_fixed_utf8(text: str, width: int) -> tuple[list[int], list[bool]]:
    if not isinstance(text, str) or not text:
        raise ValueError("StreamVLN text must be non-empty")
    payload = text.encode("utf-8")
    if width < 1 or len(payload) > width:
        raise ValueError(f"StreamVLN text requires {len(payload)} UTF-8 bytes, limit is {width}")
    return (
        list(payload) + [0] * (width - len(payload)),
        [index < len(payload) for index in range(width)],
    )


def _fixed_actions(actions) -> tuple[list[int], list[bool]]:
    actions = [int(value) for value in actions]
    if len(actions) > ACTION_CHUNK or any(not 0 <= value < 4 for value in actions):
        raise ValueError("executed actions must contain at most four categories")
    return (
        actions + [0] * (ACTION_CHUNK - len(actions)),
        [index < len(actions) for index in range(ACTION_CHUNK)],
    )


class StreamVLNEnv(VLNEnv):
    """Expose raw language/RGB identity and execute one four-action decision."""

    def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info):
        self.instruction_bytes = int(cfg.get("instruction_bytes", 512))
        self.scene_bytes = int(cfg.get("scene_bytes", 384))
        self.episode_bytes = int(cfg.get("episode_bytes", 64))
        self.env_id = int(cfg.get("env_id_base", 0)) + int(seed_offset)
        self._episode_serial = -1
        self._previous_actions = []
        self._previous_path_length = 0.0
        self._terminal_frozen = False
        super().__init__(cfg, num_envs, seed_offset, total_num_processes, worker_info)
        if self.official_streamvln_frame:
            ready = getattr(self, "service_ready", None)
            if (
                not isinstance(ready, dict)
                or ready.get("official_streamvln_frame") is not True
                or ready.get("official_frame_contract_sha256")
                != official_frame_contract_sha256()
                or ready.get("official_frame_sensor_contract")
                != OFFICIAL_FRAME_SENSOR_CONTRACT
            ):
                self.close()
                raise RuntimeError(
                    "Habitat service did not expose the pinned official frame channel"
                )
        if self.max_episode_steps % ACTION_CHUNK:
            self.close()
            raise ValueError("StreamVLN max_episode_steps must align to four-action chunks")
        instruction, instruction_mask = encode_fixed_utf8(
            self.identity["instruction_text"], self.instruction_bytes
        )
        scene, scene_mask = encode_fixed_utf8(self.identity["scene_id"], self.scene_bytes)
        episode, episode_mask = encode_fixed_utf8(
            self.identity["episode_id"], self.episode_bytes
        )
        self._fixed_text = {
            "instruction_utf8": instruction,
            "instruction_utf8_mask": instruction_mask,
            "scene_utf8": scene,
            "scene_utf8_mask": scene_mask,
            "episode_utf8": episode,
            "episode_utf8_mask": episode_mask,
        }
        self.observation_space = gym.spaces.Dict({
            "rgb": gym.spaces.Box(
                0, 255, (self.rgb_size, self.rgb_size, 3), dtype=np.uint8
            ),
            "instruction_utf8": gym.spaces.Box(
                0, 255, (self.instruction_bytes,), dtype=np.uint8
            ),
            "instruction_utf8_mask": gym.spaces.MultiBinary(self.instruction_bytes),
            "scene_utf8": gym.spaces.Box(0, 255, (self.scene_bytes,), dtype=np.uint8),
            "scene_utf8_mask": gym.spaces.MultiBinary(self.scene_bytes),
            "episode_utf8": gym.spaces.Box(0, 255, (self.episode_bytes,), dtype=np.uint8),
            "episode_utf8_mask": gym.spaces.MultiBinary(self.episode_bytes),
            "env_id": gym.spaces.Box(0, np.iinfo(np.int64).max, (), dtype=np.int64),
            "episode_serial": gym.spaces.Box(0, np.iinfo(np.int64).max, (), dtype=np.int64),
            "episode_step": gym.spaces.Box(0, np.iinfo(np.int64).max, (), dtype=np.int64),
            "episode_start": gym.spaces.MultiBinary(1),
            "previous_actions": gym.spaces.Box(0, 3, (4,), dtype=np.int64),
            "previous_action_mask": gym.spaces.MultiBinary(4),
            "terminal_frozen": gym.spaces.MultiBinary(1),
        })

    def _stream_obs(self, base_obs, *, episode_start: bool):
        if set(base_obs) != {"rgb", "instruction_ids", "instruction_mask"}:
            raise RuntimeError("base R2R observation schema changed")
        actions, action_mask = _fixed_actions(self._previous_actions)
        return {
            "rgb": base_obs["rgb"],
            **{
                key: torch.tensor([value], dtype=(torch.bool if key.endswith("_mask") else torch.uint8))
                for key, value in self._fixed_text.items()
            },
            "env_id": torch.tensor([self.env_id], dtype=torch.int64),
            "episode_serial": torch.tensor([self._episode_serial], dtype=torch.int64),
            "episode_step": self.elapsed_steps.to(dtype=torch.int64).clone(),
            "episode_start": torch.tensor([episode_start], dtype=torch.bool),
            "previous_actions": torch.tensor([actions], dtype=torch.int64),
            "previous_action_mask": torch.tensor([action_mask], dtype=torch.bool),
            "terminal_frozen": torch.tensor([self._terminal_frozen], dtype=torch.bool),
        }

    def reset(self, **kwargs):
        base_obs, info = super().reset(**kwargs)
        self._episode_serial += 1
        self._previous_actions = []
        self._previous_path_length = 0.0
        self._terminal_frozen = False
        return self._stream_obs(base_obs, episode_start=True), info

    def observe_official_frame(self) -> OfficialFrame:
        """Read the current private RGB-D/pose/intrinsics without policy leakage."""

        if not self.official_streamvln_frame:
            raise RuntimeError("official StreamVLN frame channel is disabled")
        step = int(self.elapsed_steps.item())
        if not isinstance(self._official_episode_generation, str):
            raise RuntimeError("official StreamVLN episode generation is unavailable")
        response = self._request({
            "cmd": "observe_official_frame",
            "expected_service_step": step,
            "episode_generation": self._official_episode_generation,
        })
        return OfficialFrame.from_service_payload(
            response,
            env_id=self.env_id,
            episode_serial=self._episode_serial,
            episode_generation=self._official_episode_generation,
            physical_step=step,
            identity=self.identity,
        )

    def chunk_step(self, chunk_actions, execute_count=None):
        array = np.asarray(chunk_actions)
        if array.shape != (1, ACTION_CHUNK, 1):
            raise ValueError("StreamVLN chunk actions must have shape [1, 4, 1]")
        proposed = [int(value) for value in array.reshape(-1)]
        if any(not 0 <= value < 4 for value in proposed):
            raise ValueError("StreamVLN action outside 0..3")
        if execute_count is not None:
            if type(execute_count) is not int or not 1 <= execute_count <= ACTION_CHUNK:
                raise ValueError("StreamVLN execute_count must be an integer in 1..4")
            proposed = proposed[:execute_count]
        first_stop = proposed.index(0) if 0 in proposed else None
        executed_target = proposed if first_stop is None else proposed[: first_stop + 1]

        rewards = []
        terminations = []
        truncations = []
        collision_trace = []
        execution_trace = []
        final_obs = None
        final_infos = {}
        for index, action in enumerate(executed_target):
            final_obs, reward, terminated, truncated, final_infos = super().step([[action]])
            rewards.append(reward)
            terminations.append(terminated)
            truncations.append(truncated)
            collision = final_infos.get("collision")
            if collision is None or not hasattr(collision, "item"):
                raise RuntimeError("StreamVLN executor did not report collision feedback")
            path_length = final_infos.get("path_length")
            if path_length is None or not hasattr(path_length, "item"):
                raise RuntimeError("StreamVLN executor did not report realized path length")
            collision_value = bool(collision.item())
            current_path_length = float(path_length.item())
            collision_trace.append(collision_value)
            execution_trace.append(primitive_execution(
                primitive_index=index,
                action=action,
                previous_path_length_m=self._previous_path_length,
                current_path_length_m=current_path_length,
                collision=collision_value,
                environment_reward=float(reward.item()),
                terminated=bool(terminated.item()),
                truncated=bool(truncated.item()),
            ))
            self._previous_path_length = current_path_length
            ended = bool(terminated.item()) or bool(truncated.item())
            if ended:
                if index + 1 != len(executed_target):
                    raise RuntimeError("Habitat ended before the declared executed-action prefix")
                break
        if final_obs is None:
            raise RuntimeError("StreamVLN decision executed no physical action")
        self._previous_actions = executed_target[: len(rewards)]
        self._terminal_frozen = bool(terminations[-1].item()) or bool(truncations[-1].item())
        final_infos = dict(final_infos)
        final_infos["chunk_collisions"] = list(collision_trace)
        final_infos["executor_feedback"] = chunk_feedback(
            proposed, execution_trace
        ).to_dict()

        while len(rewards) < ACTION_CHUNK:
            rewards.append(torch.zeros_like(rewards[-1]))
            terminations.append(torch.zeros_like(terminations[-1]))
            truncations.append(torch.zeros_like(truncations[-1]))
        return (
            [self._stream_obs(final_obs, episode_start=False)],
            torch.stack(rewards, dim=1),
            torch.stack(terminations, dim=1),
            torch.stack(truncations, dim=1),
            [final_infos],
        )
