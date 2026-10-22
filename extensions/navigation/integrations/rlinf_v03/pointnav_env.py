"""RLinf v0.3 single-environment bridge to the Python 3.9 Habitat service.

The simulator lives in its own process and exposes only the two-dimensional
PointGoal vector.  This class intentionally supports one environment and one
action per chunk for the first PPO smoke.
"""

from __future__ import annotations

import json
import select
import subprocess
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch


def validate_service_ready(ready, *, split, dataset_path, selected_indices):
    """Reject a Habitat process that loaded a different split or episode set."""
    if not isinstance(ready, dict) or not ready.get("ready"):
        raise RuntimeError("Habitat PointNav service did not become ready")
    if tuple(ready.get("actions", ())) != (
        "stop", "move_forward", "turn_left", "turn_right"
    ):
        raise RuntimeError("Habitat PointNav service action protocol mismatch")
    if ready.get("split") != split:
        raise RuntimeError("Habitat PointNav service dataset split mismatch")
    reported_path = ready.get("dataset_path")
    if not isinstance(reported_path, str) or Path(reported_path).resolve() != dataset_path:
        raise RuntimeError("Habitat PointNav service dataset path mismatch")
    reported_indices = ready.get("selected_indices")
    if (
        not isinstance(reported_indices, list)
        or any(type(index) is not int for index in reported_indices)
        or reported_indices != selected_indices
    ):
        raise RuntimeError("Habitat PointNav service episode indices mismatch")
    episode_ids = ready.get("episode_ids")
    if (
        not isinstance(episode_ids, list)
        or len(episode_ids) != len(selected_indices)
        or any(not isinstance(item, str) or not item for item in episode_ids)
        or len(set(episode_ids)) != len(episode_ids)
    ):
        raise RuntimeError("Habitat PointNav service episode IDs invalid")
    scene_id = ready.get("scene_id")
    if not isinstance(scene_id, str) or not scene_id:
        raise RuntimeError("Habitat PointNav service scene ID invalid")
    return frozenset(episode_ids), scene_id


class PointNavEnv(gym.Env):
    def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info):
        super().__init__()
        if num_envs != 1:
            raise ValueError("PointNav PPO smoke currently requires one environment")
        if cfg.auto_reset:
            raise ValueError("PointNav PPO smoke requires auto_reset=false")
        self.cfg = cfg
        self.num_envs = 1
        self.group_size = 1
        self.num_group = 1
        self.seed = int(cfg.seed) + int(seed_offset)
        self.worker_info = worker_info
        self.total_num_processes = int(total_num_processes)
        self.max_episode_steps = int(cfg.max_episode_steps)
        self._elapsed_steps = torch.zeros(1, dtype=torch.int32)
        self._is_start = True
        self.video_cfg = cfg.video_cfg
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Dict(
            {"states": gym.spaces.Box(-np.inf, np.inf, (2,), dtype=np.float32)}
        )

        python_bin = Path(str(cfg.python_bin)).expanduser()
        server_script = Path(str(cfg.server_script)).expanduser()
        asset_root = Path(str(cfg.asset_root)).expanduser()
        for path in (python_bin, server_script, asset_root):
            if not path.exists():
                raise FileNotFoundError(path)
        self.dataset_split = str(cfg.get("dataset_split", ""))
        if self.dataset_split not in ("train", "val"):
            raise ValueError("PointNav dataset_split must be explicitly train or val")
        dataset_path = (
            asset_root.resolve() / "datasets/pointnav/habitat-test-scenes/v1"
            / self.dataset_split / f"{self.dataset_split}.json.gz"
        ).resolve(strict=True)
        command = [
            str(python_bin),
            "-B",
            str(server_script),
            "--asset-root",
            str(asset_root),
            "--max-steps",
            str(self.max_episode_steps),
            "--split",
            self.dataset_split,
        ]
        episode_indices = cfg.get("episode_indices", None)
        if episode_indices is None:
            selected_indices = [int(cfg.get("episode_index", 0))]
            command.extend(("--episode-index", str(selected_indices[0])))
        else:
            selected_indices = [int(index) for index in episode_indices]
            if not selected_indices or len(set(selected_indices)) != len(selected_indices):
                raise ValueError("episode_indices must be nonempty and unique")
            command.extend(("--episode-indices", ",".join(map(str, selected_indices))))
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )
        try:
            ready = self._read_response(timeout=60)
            self._selected_episode_ids, self._selected_scene_id = validate_service_ready(
                ready,
                split=self.dataset_split,
                dataset_path=dataset_path,
                selected_indices=selected_indices,
            )
            self._episode_index_by_id = dict(zip(ready["episode_ids"], selected_indices))
            self._current_episode_id = None
        except BaseException:
            self.close()
            raise

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = bool(value)

    @property
    def info_logging_keys(self):
        return []

    def _read_response(self, timeout=30):
        if self.process.stdout is None:
            raise RuntimeError("PointNav service stdout is closed")
        available, _, _ = select.select([self.process.stdout], [], [], timeout)
        if not available:
            raise TimeoutError("PointNav service response timed out")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(
                "PointNav service exited before responding, exit code "
                + str(self.process.poll())
            )
        response = json.loads(line)
        if "error" in response:
            raise RuntimeError("PointNav service: " + response["error"])
        return response

    def _request(self, payload):
        if self.process.stdin is None or self.process.poll() is not None:
            raise RuntimeError("PointNav service is not running")
        self.process.stdin.write(json.dumps(payload) + "\n")
        self.process.stdin.flush()
        return self._read_response()

    @staticmethod
    def _obs(response):
        states = torch.as_tensor(response["obs"], dtype=torch.float32).reshape(1, 2)
        if not bool(torch.isfinite(states).all()):
            raise ValueError("PointNav service returned a nonfinite observation")
        return {"states": states}

    def _episode_info(self, raw):
        episode_id = raw.get("episode_id")
        if (
            episode_id != self._current_episode_id
            or raw.get("scene_id") != self._selected_scene_id
        ):
            raise RuntimeError("Habitat PointNav terminal episode mismatch")
        return {
            "success": torch.tensor([float(raw["success"])], dtype=torch.float32),
            "spl": torch.tensor([float(raw["spl"])], dtype=torch.float32),
            "episode_index": torch.tensor(
                [self._episode_index_by_id[episode_id]], dtype=torch.int64
            ),
            "geodesic_distance": torch.tensor(
                [float(raw["geodesic_distance"])], dtype=torch.float32
            ),
            "path_length": torch.tensor([float(raw["path_length"])], dtype=torch.float32),
            "collision_count": torch.tensor(
                [float(raw["collision_count"])], dtype=torch.float32
            ),
        }

    def reset(self, **_kwargs):
        response = self._request({"cmd": "reset"})
        info = response.get("info", {})
        if (
            info.get("episode_id") not in self._selected_episode_ids
            or info.get("scene_id") != self._selected_scene_id
        ):
            self.close()
            raise RuntimeError("Habitat PointNav service reset episode mismatch")
        self._current_episode_id = info["episode_id"]
        self._elapsed_steps.zero_()
        self._is_start = True
        return self._obs(response), info

    def step(self, actions):
        array = np.asarray(actions)
        if array.size != 1:
            raise ValueError("PointNav step expects one scalar action")
        item = array.reshape(-1)[0]
        if not np.issubdtype(type(item), np.integer):
            raise ValueError("PointNav action must be an integer category")
        action = int(item)
        if action < 0 or action > 3:
            raise ValueError("PointNav action outside 0..3")
        response = self._request({"cmd": "step", "action": action})
        info = response["info"]
        if (
            info.get("episode_id") != self._current_episode_id
            or info.get("scene_id") != self._selected_scene_id
        ):
            self.close()
            raise RuntimeError("Habitat PointNav service step episode mismatch")
        if not info.get("terminal_frozen", False):
            self._elapsed_steps += 1
        # Keep the final episode summary visible to RLinf's fixed-length
        # rollout logger; the one-shot done flag and reward are not repeated.
        infos = {"success": torch.tensor([float(info["success"])])}
        if response["terminated"] or response["truncated"] or info.get(
            "terminal_frozen", False
        ):
            infos["episode"] = self._episode_info(info)
        self._is_start = False
        return (
            self._obs(response),
            torch.tensor([response["reward"]], dtype=torch.float32),
            torch.tensor([response["terminated"]], dtype=torch.bool),
            torch.tensor([response["truncated"]], dtype=torch.bool),
            infos,
        )

    def chunk_step(self, chunk_actions):
        array = np.asarray(chunk_actions)
        if array.shape != (1, 1, 1):
            raise ValueError("PointNav chunk actions must have shape [1, 1, 1]")
        obs, reward, terminated, truncated, infos = self.step(array[:, 0, :])
        return (
            [obs],
            reward.unsqueeze(1),
            terminated.unsqueeze(1),
            truncated.unsqueeze(1),
            [infos],
        )

    def update_reset_state_ids(self):
        # EnvWorker.bootstrap_step performs the reset at the next rollout.
        return None

    def close(self):
        process = getattr(self, "process", None)
        if process is None:
            return
        try:
            if process.poll() is None and process.stdin is not None:
                process.stdin.write('{"cmd":"close"}\n')
                process.stdin.flush()
                process.wait(timeout=5)
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
