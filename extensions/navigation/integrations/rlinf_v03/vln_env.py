"""Single-environment RLinf v0.3 bridge to the isolated Habitat R2R service.

Only RGB and the selected episode's official instruction tokens enter the
policy observation. Simulator goals, poses, reference paths and metrics stay
outside that observation. This is the staged 64-pixel, 200-token R2R route;
the existing PointNav bridge is independent.
"""

from __future__ import annotations

import base64
import binascii
import gzip
import hashlib
import json
import math
import os
import select
import subprocess
from pathlib import Path

try:
    import gymnasium as gym
except ModuleNotFoundError as exc:  # Pinned StreamVLN Python 3.9 ships Gym 0.23.
    if exc.name != "gymnasium":
        raise
    import gym
import numpy as np
import torch


ACTIONS = ("stop", "move_forward", "turn_left", "turn_right")
OBS_KEYS = frozenset(("rgb", "instruction_ids", "instruction_mask"))
RGB_KEYS = frozenset(("encoding", "shape_hwc", "data_b64"))
RGB_ENCODING = "base64/raw-u8-hwc"
DEPTH_KEYS = frozenset(("encoding", "shape_hw", "min_m", "max_m", "data_b64"))
DEPTH_ENCODING = "base64/raw-f32le-hw"
DEPTH_MAX_METERS = 10.0
SCENE_PREFIX = "data/scene_datasets/"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_r2r_identity(dataset_path, scenes_dir, episode_index, token_width):
    """Derive the expected service identity from the read-only official file."""
    with gzip.open(dataset_path, "rt", encoding="utf-8") as stream:
        document = json.load(stream)
    if not isinstance(document, dict):
        raise ValueError("R2R dataset must be an object")
    episodes = document.get("episodes")
    vocab = document.get("instruction_vocab")
    words = vocab.get("word_list") if isinstance(vocab, dict) else None
    if not isinstance(episodes, list) or not 0 <= episode_index < len(episodes):
        raise ValueError("R2R episode_index outside dataset")
    if not isinstance(words, list) or not words or vocab.get("PAD_INDEX") != 0:
        raise ValueError("R2R vocabulary or PAD_INDEX is invalid")
    episode = episodes[episode_index]
    if not isinstance(episode, dict):
        raise ValueError("R2R selected episode is invalid")
    episode_id = episode.get("episode_id")
    scene = episode.get("scene_id")
    instruction = episode.get("instruction")
    if not isinstance(episode_id, (str, int)) or isinstance(episode_id, bool):
        raise ValueError("R2R selected episode ID is invalid")
    if not isinstance(scene, str) or not scene or "\\" in scene:
        raise ValueError("R2R selected scene is invalid")
    if not isinstance(instruction, dict):
        raise ValueError("R2R selected instruction is invalid")
    text = instruction.get("instruction_text")
    tokens = instruction.get("instruction_tokens")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("R2R selected instruction text is invalid")
    if (
        not isinstance(tokens, list)
        or not 0 < len(tokens) <= token_width
        or any(type(item) is not int or not 0 <= item < len(words) for item in tokens)
    ):
        raise ValueError("R2R selected instruction tokens are invalid")
    seen_pad = False
    for token in tokens:
        if token == 0:
            seen_pad = True
        elif seen_pad:
            raise ValueError("R2R instruction has a non-PAD token after PAD")
    ids = tokens + [0] * (token_width - len(tokens))
    if not any(ids):
        raise ValueError("R2R selected instruction is empty")

    relative_scene = scene.removeprefix(SCENE_PREFIX)
    if Path(relative_scene).is_absolute() or ".." in Path(relative_scene).parts:
        raise ValueError("R2R selected scene escapes scenes_dir")
    scene_root = scenes_dir.resolve(strict=True)
    scene_path = (scene_root / relative_scene).resolve(strict=True)
    if not scene_path.is_relative_to(scene_root) or scene_path.suffix != ".glb":
        raise ValueError("R2R selected scene escapes scenes_dir")
    if not scene_path.with_suffix(".navmesh").is_file():
        raise FileNotFoundError(scene_path.with_suffix(".navmesh"))

    return {
        "dataset_sha256": sha256_file(dataset_path),
        "episode_index": episode_index,
        "episode_id": str(episode_id),
        "scene_id": str(scene_path),
        "vocab_size": len(words),
        "vocab_sha256": hashlib.sha256(
            json.dumps(words, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "instruction_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "instruction_text": text,
        "instruction_ids": ids,
        "instruction_mask": [token != 0 for token in ids],
    }


def validate_service_ready(
    ready, *, split, dataset_path, identity, rgb_size, token_width,
    success_distance_m=3.0,
):
    """Reject a process on the wrong split, file, scene, episode or vocabulary."""
    if not isinstance(ready, dict) or ready.get("ready") is not True:
        raise RuntimeError("Habitat R2R service did not become ready")
    if tuple(ready.get("actions", ())) != ACTIONS:
        raise RuntimeError("Habitat R2R service action protocol mismatch")
    if ready.get("split") != split:
        raise RuntimeError("Habitat R2R service split mismatch")
    path = ready.get("dataset_path")
    if not isinstance(path, str) or Path(path).resolve() != dataset_path:
        raise RuntimeError("Habitat R2R service dataset path mismatch")
    for key in (
        "dataset_sha256", "episode_index", "episode_id", "vocab_sha256",
        "vocab_size", "instruction_sha256",
    ):
        if type(ready.get(key)) is not type(identity[key]) or ready[key] != identity[key]:
            raise RuntimeError(f"Habitat R2R service {key} mismatch")
    scene = ready.get("scene_id")
    if not isinstance(scene, str) or Path(scene).resolve() != Path(identity["scene_id"]):
        raise RuntimeError("Habitat R2R service scene_id mismatch")
    if type(ready.get("rgb_size")) is not int or ready["rgb_size"] != rgb_size:
        raise RuntimeError("Habitat R2R service RGB size mismatch")
    if type(ready.get("max_instruction_tokens")) is not int or ready["max_instruction_tokens"] != token_width:
        raise RuntimeError("Habitat R2R service instruction width mismatch")
    value = ready.get("success_distance_m")
    if (
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(value) or not math.isclose(value, success_distance_m)
    ):
        raise RuntimeError("Habitat R2R service Success/SPL distance mismatch")
    if ready.get("train_expert_available") is not (split == "train"):
        raise RuntimeError("Habitat R2R service train expert capability mismatch")


def validate_policy_payload(payload, *, rgb_size, token_width, identity):
    """Return only validated policy data; reject any privileged extra key."""
    if not isinstance(payload, dict) or set(payload) != OBS_KEYS:
        raise ValueError("Habitat R2R observation has unexpected policy keys")
    rgb = payload["rgb"]
    if not isinstance(rgb, dict) or set(rgb) != RGB_KEYS:
        raise ValueError("Habitat R2R RGB payload has unexpected keys")
    if rgb["encoding"] != RGB_ENCODING or rgb["shape_hwc"] != [rgb_size, rgb_size, 3]:
        raise ValueError("Habitat R2R RGB encoding or HWC shape mismatch")
    data = rgb["data_b64"]
    if not isinstance(data, str):
        raise ValueError("Habitat R2R RGB data must be base64 text")
    try:
        pixels = base64.b64decode(data, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("Habitat R2R RGB data is invalid base64") from exc
    if len(pixels) != rgb_size * rgb_size * 3:
        raise ValueError("Habitat R2R RGB byte count mismatch")
    ids = payload["instruction_ids"]
    mask = payload["instruction_mask"]
    if (
        not isinstance(ids, list)
        or not isinstance(mask, list)
        or len(ids) != token_width
        or len(mask) != token_width
        or any(type(item) is not int or not 0 <= item < identity["vocab_size"] for item in ids)
        or any(type(item) is not bool for item in mask)
        or not any(mask)
        or mask != [item != 0 for item in ids]
    ):
        raise ValueError("Habitat R2R instruction tensor contract mismatch")
    if ids != identity["instruction_ids"] or mask != identity["instruction_mask"]:
        raise RuntimeError("Habitat R2R instruction changed from selected episode")
    return pixels, ids, mask


def decode_metric_depth_payload(payload, *, depth_size):
    if not isinstance(payload, dict) or set(payload) != DEPTH_KEYS:
        raise ValueError("metric-depth payload schema mismatch")
    if (
        payload["encoding"] != DEPTH_ENCODING
        or payload["shape_hw"] != [depth_size, depth_size]
        or payload["min_m"] != 0.0
        or payload["max_m"] != DEPTH_MAX_METERS
        or not isinstance(payload["data_b64"], str)
    ):
        raise ValueError("metric-depth metadata mismatch")
    try:
        raw = base64.b64decode(payload["data_b64"], validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("metric-depth base64 is invalid") from exc
    if len(raw) != depth_size * depth_size * 4:
        raise ValueError("metric-depth byte count mismatch")
    depth = np.frombuffer(raw, dtype="<f4").copy().reshape(depth_size, depth_size)
    if (
        not np.isfinite(depth).all()
        or float(depth.min()) < 0.0
        or float(depth.max()) > DEPTH_MAX_METERS + 1e-4
    ):
        raise ValueError("metric depth is outside configured meters")
    return depth


def _finite_number(raw, key, *, lower=None, upper=None):
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Habitat R2R {key} is not finite numeric data")
    value = float(value)
    if lower is not None and value < lower or upper is not None and value > upper:
        raise ValueError(f"Habitat R2R {key} outside expected range")
    return value


class VLNEnv(gym.Env):
    """One R2R episode per Habitat process, one discrete action per chunk."""

    def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info):
        super().__init__()
        if num_envs != 1:
            raise ValueError("R2R VLN smoke requires one environment")
        if cfg.auto_reset:
            raise ValueError("R2R VLN smoke requires auto_reset=false")
        self.cfg = cfg
        self.num_envs = self.group_size = self.num_group = 1
        self.seed = int(cfg.seed) + int(seed_offset)
        self.worker_info = worker_info
        self.total_num_processes = int(total_num_processes)
        self.max_episode_steps = int(cfg.max_episode_steps)
        self.rgb_size = int(cfg.get("rgb_size", 64))
        self.token_width = int(cfg.get("max_instruction_tokens", 200))
        self.official_streamvln_frame = cfg.get(
            "official_streamvln_frame", False
        )
        if type(self.official_streamvln_frame) is not bool:
            raise TypeError("official_streamvln_frame must be boolean")
        self._official_episode_generation = None
        self.success_distance_m = float(cfg.get("success_distance_m", 3.0))
        if self.max_episode_steps < 1 or self.rgb_size < 16 or self.token_width < 1:
            raise ValueError("R2R policy requires positive max steps, RGB size and token width")
        if not math.isfinite(self.success_distance_m) or not math.isclose(self.success_distance_m, 3.0):
            raise ValueError("Staged R2R Success/SPL distance must be 3.0 m")
        self._elapsed_steps = torch.zeros(1, dtype=torch.int32)
        self._is_start = True
        self._terminal_metrics = None
        self.video_cfg = cfg.video_cfg
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Dict({
            "rgb": gym.spaces.Box(
                0, 255, (self.rgb_size, self.rgb_size, 3), dtype=np.uint8
            ),
            "instruction_ids": gym.spaces.Box(
                0, np.iinfo(np.int64).max, (self.token_width,), dtype=np.int64
            ),
            "instruction_mask": gym.spaces.MultiBinary(self.token_width),
        })

        # Keep the declared virtual-environment entrypoint intact.  Resolving a
        # venv's ``bin/python`` symlink launches the base interpreter directly
        # and loses the venv's package context on some Python installations.
        python_bin = Path(str(cfg.python_bin)).expanduser()
        if not python_bin.is_absolute():
            python_bin = python_bin.absolute()
        if not python_bin.is_file():
            raise ValueError("R2R python_bin must be an existing file")
        server_script = Path(str(cfg.server_script)).expanduser().resolve(strict=True)
        asset_root = Path(str(cfg.asset_root)).expanduser().resolve(strict=True)
        self.dataset_split = str(cfg.get("dataset_split", ""))
        if self.dataset_split not in ("train", "val_seen", "val_unseen"):
            raise ValueError("R2R dataset_split must be train, val_seen or val_unseen")
        episode_index = cfg.get("episode_index", None)
        if type(episode_index) is not int or episode_index < 0:
            raise ValueError("R2R episode_index must be an explicit nonnegative integer")
        self.dataset_path = (
            asset_root / "datasets" / "r2r" / self.dataset_split
            / f"{self.dataset_split}.json.gz"
        ).resolve(strict=True)
        scenes_dir = (asset_root / "scene_datasets").resolve(strict=True)
        self.identity = selected_r2r_identity(
            self.dataset_path, scenes_dir, episode_index, self.token_width
        )
        project_root = server_script.parents[3]
        habitat_sim_python = Path(str(cfg.get(
            "habitat_sim_python", project_root / "repos/habitat-sim-cpu-egl/src_python"
        ))).expanduser().resolve(strict=True)
        habitat_env_lib = Path(str(cfg.get(
            "habitat_env_lib", project_root / "envs/habitat-cpu-egl/lib"
        ))).expanduser().resolve(strict=True)
        egl_vendor_json = Path(str(cfg.get(
            "egl_vendor_json", "/usr/share/glvnd/egl_vendor.d/50_mesa.json"
        ))).expanduser().resolve(strict=True)
        if not habitat_sim_python.is_dir() or not habitat_env_lib.is_dir() or not egl_vendor_json.is_file():
            raise ValueError("Private Habitat-Sim source, library or Mesa EGL vendor path is invalid")
        # A Ray worker may have a different library path or visible GPU. Set
        # the known-good Mesa runtime only on the Habitat subprocess; the
        # parent RLinf policy process keeps its own CUDA allocation.
        child_env = os.environ.copy()
        child_env.update({
            "PYTHONPATH": str(habitat_sim_python),
            "LD_LIBRARY_PATH": str(habitat_env_lib),
            "CUDA_VISIBLE_DEVICES": "",
            "LIBGL_ALWAYS_SOFTWARE": "1",
            "EGL_PLATFORM": "surfaceless",
            "HABITAT_SIM_SOFTWARE_EGL": "1",
            "__EGL_VENDOR_LIBRARY_FILENAMES": str(egl_vendor_json),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        })
        command = [
            str(python_bin), "-B", str(server_script),
            "--dataset-path", str(self.dataset_path),
            "--scenes-dir", str(scenes_dir),
            "--split", self.dataset_split,
            "--episode-index", str(episode_index),
            "--rgb-size", str(self.rgb_size),
            "--max-steps", str(self.max_episode_steps),
            "--max-instruction-tokens", str(self.token_width),
        ]
        if self.official_streamvln_frame:
            command.append("--official-streamvln-frame")
        self.startup_timeout_s = float(cfg.get("startup_timeout_s", 60.0))
        if not math.isfinite(self.startup_timeout_s) or not 0 < self.startup_timeout_s <= 3600:
            raise ValueError("R2R startup_timeout_s must be finite and in (0, 3600]")
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=None, text=True, bufsize=1, env=child_env,
        )
        try:
            ready = self._read_response(timeout=self.startup_timeout_s)
            self.service_ready = dict(ready)
            validate_service_ready(
                ready, split=self.dataset_split, dataset_path=self.dataset_path,
                identity=self.identity, rgb_size=self.rgb_size,
                token_width=self.token_width,
                success_distance_m=self.success_distance_m,
            )
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
            raise RuntimeError("Habitat R2R service stdout is closed")
        available, _, _ = select.select([self.process.stdout], [], [], timeout)
        if not available:
            raise TimeoutError("Habitat R2R service response timed out")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(
                "Habitat R2R service exited before responding, exit code "
                + str(self.process.poll())
            )
        response = json.loads(line)
        if not isinstance(response, dict):
            raise ValueError("Habitat R2R service response must be an object")
        if "error" in response:
            raise RuntimeError("Habitat R2R service: " + str(response["error"]))
        return response

    def _request(self, payload, timeout=30):
        if self.process.stdin is None or self.process.poll() is not None:
            raise RuntimeError("Habitat R2R service is not running")
        self.process.stdin.write(json.dumps(payload) + "\n")
        self.process.stdin.flush()
        return self._read_response(timeout=timeout)

    def _obs(self, response):
        pixels, ids, mask = validate_policy_payload(
            response.get("obs"), rgb_size=self.rgb_size,
            token_width=self.token_width, identity=self.identity,
        )
        return {
            "rgb": torch.from_numpy(
                np.frombuffer(pixels, dtype=np.uint8).copy().reshape(
                    1, self.rgb_size, self.rgb_size, 3
                )
            ),
            "instruction_ids": torch.tensor([ids], dtype=torch.int64),
            "instruction_mask": torch.tensor([mask], dtype=torch.bool),
        }

    def _validate_identity(self, info):
        if not isinstance(info, dict):
            raise RuntimeError("Habitat R2R response has no episode identity")
        scene = info.get("scene_id")
        if (
            info.get("episode_id") != self.identity["episode_id"]
            or not isinstance(scene, str)
            or Path(scene).resolve() != Path(self.identity["scene_id"])
            or info.get("instruction_sha256") != self.identity["instruction_sha256"]
        ):
            raise RuntimeError("Habitat R2R episode, scene or instruction identity mismatch")

    def _episode_info(self, raw):
        self._validate_identity(raw)
        metrics = {
            "success": _finite_number(raw, "success", lower=0, upper=1),
            "spl": _finite_number(raw, "spl", lower=0, upper=1),
            "path_length": _finite_number(raw, "path_length", lower=0),
            "collision_count": _finite_number(raw, "collision_count", lower=0),
        }
        if "distance_to_goal" in raw:
            metrics["distance_to_goal"] = _finite_number(raw, "distance_to_goal", lower=0)
        if self._terminal_metrics is not None and metrics != self._terminal_metrics:
            raise RuntimeError("Habitat R2R terminal metrics changed after episode end")
        return {
            **{key: torch.tensor([value], dtype=torch.float32)
               for key, value in metrics.items()},
            "episode_index": torch.tensor([self.identity["episode_index"]], dtype=torch.int64),
        }, metrics

    def reset(self, **_kwargs):
        response = self._request({"cmd": "reset"})
        try:
            reset_info = response.get("info")
            self._validate_identity(reset_info)
            if self.official_streamvln_frame:
                episode_generation = reset_info.get("episode_generation")
                if (
                    not isinstance(episode_generation, str)
                    or len(episode_generation) != 32
                    or any(
                        character not in "0123456789abcdef"
                        for character in episode_generation
                    )
                ):
                    raise RuntimeError(
                        "Habitat R2R service returned an invalid episode generation"
                    )
                self._official_episode_generation = episode_generation
            else:
                self._official_episode_generation = None
            obs = self._obs(response)
        except BaseException:
            self.close()
            raise
        self._terminal_metrics = None
        self._elapsed_steps.zero_()
        self._is_start = True
        return obs, {
            "episode_id": self.identity["episode_id"],
            "scene_id": self.identity["scene_id"],
        }

    def expert_waypoint(self, lookahead_m=1.0):
        """Query train-only labels through a channel separate from policy observations."""
        if self.dataset_split != "train":
            raise RuntimeError("expert waypoint queries are restricted to the train split")
        response = self._request({
            "cmd": "expert_waypoint",
            "lookahead_m": float(lookahead_m),
        })
        expected = {
            "schema_version", "source", "lookahead_m", "waypoint_local_m",
            "bearing_rad", "target_distance_m", "distance_to_goal_m", "stop",
        }
        if not isinstance(response, dict) or set(response) != expected:
            raise ValueError("Habitat expert waypoint schema mismatch")
        if response["schema_version"] != 1 or response["source"] != "train-shortest-path":
            raise ValueError("Habitat expert waypoint provenance mismatch")
        waypoint = response["waypoint_local_m"]
        if (
            not isinstance(waypoint, list) or len(waypoint) != 2
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in waypoint)
            or type(response["stop"]) is not bool
        ):
            raise ValueError("Habitat expert waypoint values are invalid")
        numeric = [
            *waypoint, response["lookahead_m"], response["bearing_rad"],
            response["target_distance_m"], response["distance_to_goal_m"],
        ]
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(float(value)) for value in numeric):
            raise ValueError("Habitat expert waypoint contains nonfinite values")
        if not 0.25 <= float(response["lookahead_m"]) <= 2.0:
            raise ValueError("Habitat expert waypoint lookahead is outside the contract")
        if response["stop"] and any(abs(float(value)) > 1e-8 for value in waypoint):
            raise ValueError("Habitat STOP label must have a zero local waypoint")
        return {
            **response,
            "waypoint_local_m": [float(value) for value in waypoint],
            "lookahead_m": float(response["lookahead_m"]),
            "bearing_rad": float(response["bearing_rad"]),
            "target_distance_m": float(response["target_distance_m"]),
            "distance_to_goal_m": float(response["distance_to_goal_m"]),
        }

    def observe_aligned_depth(self):
        """Read current metric depth for the explicit G7.8 local controller."""
        if self.dataset_split != "train":
            raise RuntimeError("aligned depth diagnostic is restricted to train")
        response = self._request({"cmd": "observe_aligned_depth"})
        expected = {"schema_version", "source", "state_unchanged", "metric_depth"}
        if not isinstance(response, dict) or set(response) != expected:
            raise ValueError("runtime aligned-depth response schema mismatch")
        if (
            response["schema_version"] != 1
            or response["source"] != "runtime-aligned-metric-depth"
            or response["state_unchanged"] is not True
        ):
            raise ValueError("runtime aligned-depth provenance mismatch")
        return decode_metric_depth_payload(
            response["metric_depth"], depth_size=self.rgb_size
        )

    def probe_depth_candidates(self, candidates):
        """Return train-only post-turn metric depth and forward-collision labels."""
        if self.dataset_split != "train":
            raise RuntimeError("depth candidate probes are restricted to train")
        if not isinstance(candidates, (list, tuple)) or not 1 <= len(candidates) <= 64:
            raise ValueError("depth probes require 1..64 candidates")
        payload = []
        names = set()
        for candidate in candidates:
            name = getattr(candidate, "name", None)
            actions = getattr(candidate, "actions", None)
            if (
                not isinstance(name, str)
                or not name
                or name in names
                or not isinstance(actions, tuple)
                or not 2 <= len(actions) <= 8
                or any(type(action) is not int or action not in (1, 2, 3) for action in actions)
                or 1 not in actions
            ):
                raise ValueError("depth probe input violates macro contract")
            first_forward = actions.index(1)
            if any(action == 1 for action in actions[:first_forward]) or any(
                action != 1 for action in actions[first_forward:]
            ):
                raise ValueError("depth macro must be turn prefix then forward suffix")
            names.add(name)
            payload.append({"name": name, "actions": list(actions)})
        response = self._request(
            {"cmd": "probe_depth_candidates", "candidates": payload}
        )
        expected = {
            "schema_version", "source", "policy_observation_unchanged",
            "state_restored", "max_position_restore_error_m",
            "max_rotation_restore_error", "candidates",
        }
        if not isinstance(response, dict) or set(response) != expected:
            raise ValueError("depth probe response schema mismatch")
        if (
            response["schema_version"] != 1
            or response["source"] != "train-simulator-aligned-depth-branch-rollout"
            or response["policy_observation_unchanged"] is not True
            or response["state_restored"] is not True
        ):
            raise ValueError("depth probe provenance or restoration failed")
        position_error = _finite_number(
            response, "max_position_restore_error_m", lower=0
        )
        rotation_error = _finite_number(
            response, "max_rotation_restore_error", lower=0
        )
        if position_error > 1e-6 or rotation_error > 1e-6:
            raise RuntimeError("depth probe restoration exceeds tolerance")
        rows = response["candidates"]
        if not isinstance(rows, list) or len(rows) != len(payload):
            raise ValueError("depth probe result count mismatch")
        normalized = []
        for request, row in zip(payload, rows):
            required = {
                "name", "actions", "turn_prefix", "forward_suffix",
                "metric_depth", "forward_collisions", "unsafe", "end_position",
            }
            if not isinstance(row, dict) or set(row) != required:
                raise ValueError("depth probe row schema mismatch")
            if row["name"] != request["name"] or row["actions"] != request["actions"]:
                raise RuntimeError("depth probe identity changed")
            first_forward = request["actions"].index(1)
            if (
                row["turn_prefix"] != request["actions"][:first_forward]
                or row["forward_suffix"] != request["actions"][first_forward:]
            ):
                raise RuntimeError("depth probe phase split changed")
            collisions = row["forward_collisions"]
            if (
                not isinstance(collisions, list)
                or len(collisions) != len(row["forward_suffix"])
                or any(type(value) is not bool for value in collisions)
                or row["unsafe"] is not any(collisions)
            ):
                raise ValueError("depth probe collision labels are invalid")
            end_position = row["end_position"]
            if (
                not isinstance(end_position, list)
                or len(end_position) != 3
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    for value in end_position
                )
            ):
                raise ValueError("depth probe end position is invalid")
            normalized.append(
                {
                    "name": row["name"],
                    "actions": list(row["actions"]),
                    "turn_prefix": list(row["turn_prefix"]),
                    "forward_suffix": list(row["forward_suffix"]),
                    "metric_depth": decode_metric_depth_payload(
                        row["metric_depth"], depth_size=self.rgb_size
                    ),
                    "forward_collisions": list(collisions),
                    "unsafe": bool(row["unsafe"]),
                    "end_position": [float(value) for value in end_position],
                }
            )
        return {
            "source": response["source"],
            "max_position_restore_error_m": position_error,
            "max_rotation_restore_error": rotation_error,
            "candidates": normalized,
        }

    def probe_macro_audit(self, candidates):
        """Return train-only repeated primitive traces for the fixed G7.9 audit."""
        if self.dataset_split != "train":
            raise RuntimeError("macro audit is restricted to train")
        if not isinstance(candidates, (list, tuple)) or len(candidates) != 32:
            raise ValueError("G7.9 macro audit requires exactly 32 candidates")
        payload = []
        names = set()
        for candidate in candidates:
            name = getattr(candidate, "name", None)
            actions = getattr(candidate, "actions", None)
            if (
                not isinstance(name, str)
                or not name
                or name in names
                or not isinstance(actions, tuple)
                or not 2 <= len(actions) <= 8
                or any(type(action) is not int or action not in (1, 2, 3) for action in actions)
                or 1 not in actions
            ):
                raise ValueError("macro audit input violates exact-macro contract")
            first_forward = actions.index(1)
            if (
                len(actions[first_forward:]) < 2
                or any(action == 1 for action in actions[:first_forward])
                or any(action != 1 for action in actions[first_forward:])
            ):
                raise ValueError("macro audit requires turn prefix then >=2 forwards")
            names.add(name)
            payload.append({"name": name, "actions": list(actions)})
        response = self._request(
            {"cmd": "probe_macro_audit", "candidates": payload}
        )
        expected = {
            "schema_version", "source", "policy_observation_unchanged",
            "state_restored", "forward_step_m", "horizon_forward_steps",
            "physical_horizon_m", "repeat_count",
            "max_position_restore_error_m", "max_rotation_restore_error",
            "candidates",
        }
        if not isinstance(response, dict) or set(response) != expected:
            raise ValueError("macro audit response schema mismatch")
        if (
            response["schema_version"] != 1
            or response["source"] != "train-simulator-exact-macro-controllability-audit"
            or response["policy_observation_unchanged"] is not True
            or response["state_restored"] is not True
            or response["forward_step_m"] != 0.25
            or response["horizon_forward_steps"] != 2
            or response["physical_horizon_m"] != 0.5
            or response["repeat_count"] != 2
        ):
            raise ValueError("macro audit provenance or preregistration changed")
        position_error = _finite_number(
            response, "max_position_restore_error_m", lower=0
        )
        rotation_error = _finite_number(
            response, "max_rotation_restore_error", lower=0
        )
        if position_error > 1e-6 or rotation_error > 1e-6:
            raise RuntimeError("macro audit restoration exceeds tolerance")
        rows = response["candidates"]
        if not isinstance(rows, list) or len(rows) != len(payload):
            raise ValueError("macro audit result count mismatch")

        def finite_vector(value, length, name):
            if (
                not isinstance(value, list)
                or len(value) != length
                or any(
                    isinstance(item, bool)
                    or not isinstance(item, (int, float))
                    or not math.isfinite(float(item))
                    for item in value
                )
            ):
                raise ValueError("macro audit " + name + " is invalid")
            return [float(item) for item in value]

        normalized = []
        repeat_expected = {
            "turn_collisions", "post_turn_pose", "metric_depth", "depth_sha256",
            "forward_direction", "commanded_distance_m", "intended_endpoint",
            "privileged_endpoint", "privileged_endpoint_error_m",
            "privileged_swept_path_free", "primitive_trace",
            "full_collision_trace", "full_unsafe", "horizon_collision_trace",
            "horizon_unsafe", "final_pose", "realized_forward_m",
            "realized_lateral_error_m", "endpoint_error_m",
            "heading_error_degrees",
        }
        pose_expected = {"position", "rotation_coefficients"}
        primitive_expected = {
            "primitive_index", "action", "collision", "before_position",
            "after_position", "displacement_m", "forward_progress_m",
        }
        scalar_keys = (
            "commanded_distance_m", "privileged_endpoint_error_m",
            "realized_forward_m", "realized_lateral_error_m",
            "endpoint_error_m", "heading_error_degrees",
        )
        for request, row in zip(payload, rows):
            if (
                not isinstance(row, dict)
                or set(row) != {
                    "name", "actions", "turn_prefix", "forward_suffix", "repeats"
                }
                or row["name"] != request["name"]
                or row["actions"] != request["actions"]
            ):
                raise ValueError("macro audit candidate identity changed")
            first_forward = request["actions"].index(1)
            forward_suffix = request["actions"][first_forward:]
            if (
                row["turn_prefix"] != request["actions"][:first_forward]
                or row["forward_suffix"] != forward_suffix
                or not isinstance(row["repeats"], list)
                or len(row["repeats"]) != 2
            ):
                raise ValueError("macro audit phase or repeat count changed")
            repeats = []
            for repeat in row["repeats"]:
                if not isinstance(repeat, dict) or set(repeat) != repeat_expected:
                    raise ValueError("macro audit repeat schema mismatch")
                for key in ("post_turn_pose", "final_pose"):
                    pose = repeat[key]
                    if not isinstance(pose, dict) or set(pose) != pose_expected:
                        raise ValueError("macro audit pose schema mismatch")
                turn_collisions = repeat["turn_collisions"]
                full_trace = repeat["full_collision_trace"]
                horizon_trace = repeat["horizon_collision_trace"]
                if (
                    not isinstance(turn_collisions, list)
                    or len(turn_collisions) != first_forward
                    or any(type(value) is not bool for value in turn_collisions)
                    or not isinstance(full_trace, list)
                    or len(full_trace) != len(forward_suffix)
                    or any(type(value) is not bool for value in full_trace)
                    or horizon_trace != full_trace[:2]
                    or repeat["full_unsafe"] is not any(full_trace)
                    or repeat["horizon_unsafe"] is not any(horizon_trace)
                    or type(repeat["privileged_swept_path_free"]) is not bool
                ):
                    raise ValueError("macro audit collision trace is invalid")
                primitives = repeat["primitive_trace"]
                if not isinstance(primitives, list) or len(primitives) != len(full_trace):
                    raise ValueError("macro audit primitive trace count changed")
                normalized_primitives = []
                for index, primitive in enumerate(primitives):
                    if (
                        not isinstance(primitive, dict)
                        or set(primitive) != primitive_expected
                        or primitive["primitive_index"] != index
                        or primitive["action"] != 1
                        or primitive["collision"] is not full_trace[index]
                    ):
                        raise ValueError("macro audit primitive identity changed")
                    normalized_primitives.append({
                        **primitive,
                        "before_position": finite_vector(
                            primitive["before_position"], 3, "before position"
                        ),
                        "after_position": finite_vector(
                            primitive["after_position"], 3, "after position"
                        ),
                        "displacement_m": _finite_number(
                            primitive, "displacement_m", lower=0
                        ),
                        "forward_progress_m": _finite_number(
                            primitive, "forward_progress_m"
                        ),
                    })
                normalized_repeat = dict(repeat)
                normalized_repeat["metric_depth"] = decode_metric_depth_payload(
                    repeat["metric_depth"], depth_size=self.rgb_size
                )
                if (
                    not isinstance(repeat["depth_sha256"], str)
                    or len(repeat["depth_sha256"]) != 64
                    or any(character not in "0123456789abcdef" for character in repeat["depth_sha256"])
                ):
                    raise ValueError("macro audit depth hash is invalid")
                for key in scalar_keys:
                    normalized_repeat[key] = _finite_number(
                        repeat, key, lower=0 if key not in ("realized_forward_m",) else None
                    )
                for key in ("forward_direction", "intended_endpoint", "privileged_endpoint"):
                    normalized_repeat[key] = finite_vector(repeat[key], 3, key)
                for key in ("post_turn_pose", "final_pose"):
                    normalized_repeat[key] = {
                        "position": finite_vector(
                            repeat[key]["position"], 3, key + " position"
                        ),
                        "rotation_coefficients": finite_vector(
                            repeat[key]["rotation_coefficients"], 4, key + " rotation"
                        ),
                    }
                normalized_repeat["primitive_trace"] = normalized_primitives
                repeats.append(normalized_repeat)
            normalized.append({
                "name": row["name"],
                "actions": list(row["actions"]),
                "turn_prefix": list(row["turn_prefix"]),
                "forward_suffix": list(row["forward_suffix"]),
                "repeats": repeats,
            })
        return {
            "source": response["source"],
            "forward_step_m": 0.25,
            "horizon_forward_steps": 2,
            "physical_horizon_m": 0.5,
            "repeat_count": 2,
            "max_position_restore_error_m": position_error,
            "max_rotation_restore_error": rotation_error,
            "candidates": normalized,
        }


    def probe_geometry_controller(self, candidates):
        """Return G7.11 matched explicit-geometry controller branches."""
        if self.dataset_split != "train":
            raise RuntimeError("G7.11 controller audit is restricted to train")
        if not isinstance(candidates, (list, tuple)) or len(candidates) != 32:
            raise ValueError("G7.11 controller audit requires exactly 32 candidates")
        payload = []
        names = set()
        for candidate in candidates:
            name = getattr(candidate, "name", None)
            actions = getattr(candidate, "actions", None)
            if (
                not isinstance(name, str)
                or not name
                or name in names
                or not isinstance(actions, tuple)
                or not 2 <= len(actions) <= 8
                or any(type(action) is not int or action not in (1, 2, 3) for action in actions)
                or 1 not in actions
            ):
                raise ValueError("G7.11 candidate violates exact-macro contract")
            first_forward = actions.index(1)
            if (
                len(actions[first_forward:]) < 2
                or any(action == 1 for action in actions[:first_forward])
                or any(action != 1 for action in actions[first_forward:])
            ):
                raise ValueError("G7.11 requires turn prefix then >=2 forwards")
            names.add(name)
            payload.append({"name": name, "actions": list(actions)})
        response = self._request(
            {"cmd": "probe_geometry_controller", "candidates": payload},
            timeout=900,
        )
        expected = {
            "schema_version", "source", "policy_observation_unchanged",
            "state_restored", "repeat_count", "variants", "depth_geometry",
            "endpoint_tolerance_m", "navmesh_sample_spacing_m",
            "navmesh_minimum_clearance_m", "max_position_restore_error_m",
            "max_rotation_restore_error", "candidates",
        }
        variants = [
            "open_loop", "endpoint_geometry", "depth_receding",
            "navmesh_receding_oracle",
        ]
        geometry = {
            "height": 64, "width": 64, "hfov_degrees": 90.0,
            "fx": 32.0, "fy": 32.0, "cx": 31.5, "cy": 31.5,
            "sensor_height_m": 1.25, "agent_height_m": 1.5,
            "agent_radius_m": 0.1, "safety_margin_m": 0.02,
            "corridor_half_width_m": 0.12,
            "obstacle_min_height_m": 0.05,
            "obstacle_max_height_m": 1.45,
            "primitive_brake_distance_m": 0.37,
        }
        if (
            not isinstance(response, dict)
            or set(response) != expected
            or response["schema_version"] != 1
            or response["source"]
            != "train-simulator-explicit-depth-footprint-controller-audit"
            or response["policy_observation_unchanged"] is not True
            or response["state_restored"] is not True
            or response["repeat_count"] != 2
            or response["variants"] != variants
            or response["depth_geometry"] != geometry
            or response["endpoint_tolerance_m"] != 0.001
            or response["navmesh_sample_spacing_m"] != 0.125
            or response["navmesh_minimum_clearance_m"] != 0.02
        ):
            raise ValueError("G7.11 response provenance or preregistration changed")
        position_error = _finite_number(
            response, "max_position_restore_error_m", lower=0
        )
        rotation_error = _finite_number(
            response, "max_rotation_restore_error", lower=0
        )
        if position_error > 1e-6 or rotation_error > 1e-6:
            raise RuntimeError("G7.11 state restoration exceeds tolerance")
        rows = response["candidates"]
        if not isinstance(rows, list) or len(rows) != len(payload):
            raise ValueError("G7.11 candidate count changed")
        repeat_keys = {
            "variant", "turn_collisions", "post_turn_pose", "forward_direction",
            "initial_depth_sha256", "initial_corridor_measurement",
            "full_commanded_distance_m", "full_intended_endpoint",
            "endpoint_geometry_filtered_endpoint", "endpoint_geometry_error_m",
            "endpoint_geometry_allow", "decisions", "brake_index",
            "primitive_trace", "executed_forward_primitives",
            "commanded_forward_primitives", "attempted_distance_m", "completed",
            "raw_collision_trace", "final_pose", "realized_forward_m",
            "realized_lateral_error_m", "attempted_endpoint_error_m",
            "full_endpoint_error_m", "intended_clearance_trace",
            "actual_clearance_trace", "minimum_intended_clearance_m",
            "minimum_actual_clearance_m", "controlled_motion",
            "severe_motion_exposure",
        }
        normalized = []
        for request, row in zip(payload, rows):
            if (
                not isinstance(row, dict)
                or set(row) != {
                    "name", "actions", "turn_prefix", "forward_suffix",
                    "initial_metric_depth", "initial_depth_sha256", "variants",
                }
                or row["name"] != request["name"]
                or row["actions"] != request["actions"]
                or not isinstance(row["initial_depth_sha256"], str)
                or len(row["initial_depth_sha256"]) != 64
                or not isinstance(row["variants"], dict)
                or list(row["variants"]) != variants
            ):
                raise ValueError("G7.11 candidate identity or variants changed")
            first_forward = request["actions"].index(1)
            if (
                row["turn_prefix"] != request["actions"][:first_forward]
                or row["forward_suffix"] != request["actions"][first_forward:]
            ):
                raise ValueError("G7.11 macro phase split changed")
            initial_depth = decode_metric_depth_payload(
                row["initial_metric_depth"], depth_size=self.rgb_size
            )
            normalized_variants = {}
            for variant in variants:
                repeats = row["variants"][variant]
                if not isinstance(repeats, list) or len(repeats) != 2:
                    raise ValueError("G7.11 repeat count changed")
                normalized_repeats = []
                for repeat in repeats:
                    if (
                        not isinstance(repeat, dict)
                        or set(repeat) != repeat_keys
                        or repeat["variant"] != variant
                        or repeat["initial_depth_sha256"]
                        != row["initial_depth_sha256"]
                    ):
                        raise ValueError("G7.11 repeat identity changed")
                    executed = repeat["executed_forward_primitives"]
                    commanded = repeat["commanded_forward_primitives"]
                    primitives = repeat["primitive_trace"]
                    collisions = repeat["raw_collision_trace"]
                    if (
                        type(executed) is not int
                        or type(commanded) is not int
                        or commanded != len(row["forward_suffix"])
                        or not 0 <= executed <= commanded
                        or not isinstance(primitives, list)
                        or len(primitives) != executed
                        or not isinstance(collisions, list)
                        or len(collisions) != executed
                        or any(type(value) is not bool for value in collisions)
                        or repeat["completed"] is not (executed == commanded)
                        or abs(
                            _finite_number(repeat, "attempted_distance_m", lower=0)
                            - executed * 0.25
                        ) > 1e-8
                    ):
                        raise ValueError("G7.11 executed motion trace changed")
                    controlled = repeat["controlled_motion"]
                    if executed == 0:
                        if controlled is not None or repeat["severe_motion_exposure"] is not False:
                            raise ValueError("G7.11 zero-motion severity changed")
                    elif (
                        not isinstance(controlled, dict)
                        or set(controlled) != {
                            "severe_exposure", "continuous_severity",
                            "realized_ratio", "minimum_primitive_progress_m",
                            "maximum_primitive_progress_deficit_m",
                        }
                        or type(controlled["severe_exposure"]) is not bool
                        or repeat["severe_motion_exposure"]
                        is not controlled["severe_exposure"]
                    ):
                        raise ValueError("G7.11 controlled severity changed")
                    normalized_repeats.append(repeat)
                normalized_variants[variant] = normalized_repeats
            normalized.append(
                {
                    "name": row["name"],
                    "actions": list(row["actions"]),
                    "turn_prefix": list(row["turn_prefix"]),
                    "forward_suffix": list(row["forward_suffix"]),
                    "initial_metric_depth": initial_depth,
                    "initial_depth_sha256": row["initial_depth_sha256"],
                    "variants": normalized_variants,
                }
            )
        return {
            "source": response["source"],
            "repeat_count": 2,
            "variants": variants,
            "depth_geometry": geometry,
            "max_position_restore_error_m": position_error,
            "max_rotation_restore_error": rotation_error,
            "candidates": normalized,
        }


    def probe_ground_support(self, candidates):
        """Return G7.13 repeated causal pre-action depth observations."""
        if self.dataset_split != "train":
            raise RuntimeError("G7.13 ground-support audit is restricted to train")
        if not isinstance(candidates, (list, tuple)) or not 1 <= len(candidates) <= 32:
            raise ValueError("G7.13 requires between one and 32 observation cases")
        payload = []
        names = set()
        for candidate in candidates:
            if not isinstance(candidate, dict) or set(candidate) != {
                "name", "actions", "observe_before_forward_index",
            }:
                raise ValueError("G7.13 observation-case schema changed")
            name = candidate["name"]
            actions = candidate["actions"]
            critical = candidate["observe_before_forward_index"]
            if (
                not isinstance(name, str)
                or not name
                or name in names
                or not isinstance(actions, list)
                or not 2 <= len(actions) <= 8
                or any(type(action) is not int or action not in (1, 2, 3) for action in actions)
                or 1 not in actions
                or type(critical) is not int
            ):
                raise ValueError("G7.13 observation-case identity is invalid")
            first_forward = actions.index(1)
            if (
                len(actions[first_forward:]) < 2
                or any(action == 1 for action in actions[:first_forward])
                or any(action != 1 for action in actions[first_forward:])
                or not 1 <= critical <= len(actions[first_forward:])
            ):
                raise ValueError("G7.13 observation timing is invalid")
            names.add(name)
            payload.append(
                {
                    "name": name,
                    "actions": list(actions),
                    "observe_before_forward_index": critical,
                }
            )
        response = self._request(
            {"cmd": "probe_ground_support", "candidates": payload},
            timeout=900,
        )
        expected = {
            "schema_version", "source", "policy_observation_unchanged",
            "state_restored", "repeat_count", "observation_contract",
            "max_position_restore_error_m", "max_rotation_restore_error",
            "candidates",
        }
        contract = {
            "timing": "after-turn-and-critical-prefix-minus-one-forwards; before-critical-forward",
            "critical_primitive_executed": False,
            "depth_height": 64,
            "depth_width": 64,
            "hfov_degrees": 90.0,
            "fx": 32.0,
            "fy": 32.0,
            "cx": 31.5,
            "cy": 31.5,
            "sensor_height_m": 1.25,
            "agent_radius_m": 0.1,
            "corridor_half_width_m": 0.12,
            "ground_forward_max_m": 2.0,
            "ground_height_tolerance_m": 0.1,
            "depth_missing_epsilon_m": 0.0001,
        }
        if (
            not isinstance(response, dict)
            or set(response) != expected
            or response["schema_version"] != 1
            or response["source"]
            != "train-simulator-causal-ground-support-observability-audit"
            or response["policy_observation_unchanged"] is not True
            or response["state_restored"] is not True
            or response["repeat_count"] != 2
            or response["observation_contract"] != contract
        ):
            raise ValueError("G7.13 response provenance or preregistration changed")
        position_error = _finite_number(
            response, "max_position_restore_error_m", lower=0
        )
        rotation_error = _finite_number(
            response, "max_rotation_restore_error", lower=0
        )
        if position_error > 1e-6 or rotation_error > 1e-6:
            raise RuntimeError("G7.13 state restoration exceeds tolerance")
        rows = response["candidates"]
        if not isinstance(rows, list) or len(rows) != len(payload):
            raise ValueError("G7.13 observation-case count changed")
        repeat_keys = {
            "turn_collisions", "executed_prefix_count", "prefix_trace",
            "pre_action_pose", "forward_direction", "metric_depth",
            "depth_sha256",
        }
        pose_keys = {"position", "rotation_coefficients"}
        primitive_keys = {
            "primitive_index", "action", "collision", "before_position",
            "after_position", "displacement_m", "forward_progress_m",
            "lateral_displacement_m",
        }

        def finite_vector(value, length, name):
            if (
                not isinstance(value, list)
                or len(value) != length
                or any(
                    isinstance(item, bool)
                    or not isinstance(item, (int, float))
                    or not math.isfinite(float(item))
                    for item in value
                )
            ):
                raise ValueError("G7.13 " + name + " is invalid")
            return [float(item) for item in value]

        normalized = []
        for request, row in zip(payload, rows):
            first_forward = request["actions"].index(1)
            expected_turn = request["actions"][:first_forward]
            expected_forward = request["actions"][first_forward:]
            if (
                not isinstance(row, dict)
                or set(row) != {
                    "name", "actions", "turn_prefix", "forward_suffix",
                    "observe_before_forward_index", "repeats",
                }
                or row["name"] != request["name"]
                or row["actions"] != request["actions"]
                or row["turn_prefix"] != expected_turn
                or row["forward_suffix"] != expected_forward
                or row["observe_before_forward_index"]
                != request["observe_before_forward_index"]
                or not isinstance(row["repeats"], list)
                or len(row["repeats"]) != 2
            ):
                raise ValueError("G7.13 observation-case response changed")
            repeats = []
            prefix_count = request["observe_before_forward_index"] - 1
            for repeat in row["repeats"]:
                if not isinstance(repeat, dict) or set(repeat) != repeat_keys:
                    raise ValueError("G7.13 repeat schema changed")
                if (
                    repeat["executed_prefix_count"] != prefix_count
                    or not isinstance(repeat["turn_collisions"], list)
                    or len(repeat["turn_collisions"]) != len(expected_turn)
                    or any(type(value) is not bool for value in repeat["turn_collisions"])
                    or not isinstance(repeat["prefix_trace"], list)
                    or len(repeat["prefix_trace"]) != prefix_count
                    or not isinstance(repeat["depth_sha256"], str)
                    or len(repeat["depth_sha256"]) != 64
                    or any(character not in "0123456789abcdef" for character in repeat["depth_sha256"])
                    or not isinstance(repeat["pre_action_pose"], dict)
                    or set(repeat["pre_action_pose"]) != pose_keys
                ):
                    raise ValueError("G7.13 repeat identity changed")
                primitives = []
                for index, primitive in enumerate(repeat["prefix_trace"]):
                    if (
                        not isinstance(primitive, dict)
                        or set(primitive) != primitive_keys
                        or primitive["primitive_index"] != index
                        or primitive["action"] != 1
                        or type(primitive["collision"]) is not bool
                    ):
                        raise ValueError("G7.13 prefix trace changed")
                    normalized_primitive = dict(primitive)
                    normalized_primitive["before_position"] = finite_vector(
                        primitive["before_position"], 3, "before position"
                    )
                    normalized_primitive["after_position"] = finite_vector(
                        primitive["after_position"], 3, "after position"
                    )
                    for key in (
                        "displacement_m", "forward_progress_m",
                        "lateral_displacement_m",
                    ):
                        normalized_primitive[key] = _finite_number(
                            primitive, key,
                            lower=0 if key != "forward_progress_m" else None,
                        )
                    primitives.append(normalized_primitive)
                normalized_repeat = {
                    **repeat,
                    "prefix_trace": primitives,
                    "pre_action_pose": {
                        "position": finite_vector(
                            repeat["pre_action_pose"]["position"], 3,
                            "pre-action position",
                        ),
                        "rotation_coefficients": finite_vector(
                            repeat["pre_action_pose"]["rotation_coefficients"], 4,
                            "pre-action rotation",
                        ),
                    },
                    "forward_direction": finite_vector(
                        repeat["forward_direction"], 3, "forward direction"
                    ),
                    "metric_depth": decode_metric_depth_payload(
                        repeat["metric_depth"], depth_size=self.rgb_size
                    ),
                }
                repeats.append(normalized_repeat)
            normalized.append(
                {
                    "name": row["name"],
                    "actions": list(row["actions"]),
                    "turn_prefix": list(row["turn_prefix"]),
                    "forward_suffix": list(row["forward_suffix"]),
                    "observe_before_forward_index": row[
                        "observe_before_forward_index"
                    ],
                    "repeats": repeats,
                }
            )
        return {
            "source": response["source"],
            "repeat_count": 2,
            "observation_contract": contract,
            "max_position_restore_error_m": position_error,
            "max_rotation_restore_error": rotation_error,
            "candidates": normalized,
        }

    def probe_immediate_footprint_visibility(self, candidates):
        """Return G7.14 repeated pitched-depth visibility observations."""
        if self.dataset_split != "train":
            raise RuntimeError("G7.14 visibility ceiling is restricted to train")
        if not isinstance(candidates, (list, tuple)) or not 1 <= len(candidates) <= 32:
            raise ValueError("G7.14 requires between one and 32 observation cases")
        payload = []
        names = set()
        for candidate in candidates:
            if not isinstance(candidate, dict) or set(candidate) != {
                "name", "actions", "observe_before_forward_index",
            }:
                raise ValueError("G7.14 observation-case schema changed")
            name = candidate["name"]
            actions = candidate["actions"]
            critical = candidate["observe_before_forward_index"]
            if (
                not isinstance(name, str)
                or not name
                or name in names
                or not isinstance(actions, list)
                or not 2 <= len(actions) <= 8
                or any(type(action) is not int or action not in (1, 2, 3) for action in actions)
                or 1 not in actions
                or type(critical) is not int
            ):
                raise ValueError("G7.14 observation-case identity is invalid")
            first_forward = actions.index(1)
            if (
                len(actions[first_forward:]) < 2
                or any(action == 1 for action in actions[:first_forward])
                or any(action != 1 for action in actions[first_forward:])
                or not 1 <= critical <= len(actions[first_forward:])
            ):
                raise ValueError("G7.14 observation timing is invalid")
            names.add(name)
            payload.append(
                {
                    "name": name,
                    "actions": list(actions),
                    "observe_before_forward_index": critical,
                }
            )
        response = self._request(
            {"cmd": "probe_immediate_footprint_visibility", "candidates": payload},
            timeout=900,
        )
        expected = {
            "schema_version", "source", "policy_observation_unchanged",
            "state_restored", "repeat_count", "observation_contract",
            "max_position_restore_error_m", "max_rotation_restore_error",
            "candidates",
        }
        target_geometry = {
            "target_ray_count": 48,
            "forward_bin_counts": [18, 18, 12],
            "lateral_bin_counts": [16, 16, 16],
            "grid_counts": [[6, 6, 6], [6, 6, 6], [4, 4, 4]],
            "minimum_expected_forward_m": 0.033987915407854924,
            "maximum_expected_forward_m": 0.3115141955835961,
            "minimum_expected_lateral_m": -0.10053271911657698,
            "maximum_expected_lateral_m": 0.10053271911657698,
            "minimum_expected_optical_depth_m": 1.2323914232248423,
            "maximum_expected_optical_depth_m": 1.2868188046921853,
        }
        contract = {
            "timing": "after-turn-and-critical-prefix-minus-one-forwards; before-critical-forward",
            "critical_primitive_executed": False,
            "depth_height": 64,
            "depth_width": 64,
            "hfov_degrees": 90.0,
            "fx": 32.0,
            "fy": 32.0,
            "cx": 31.5,
            "cy": 31.5,
            "sensor_height_m": 1.25,
            "forward_step_m": 0.25,
            "agent_radius_m": 0.1,
            "corridor_half_width_m": 0.12,
            "depth_pitch_radians": -1.373400766945016,
            "depth_pitch_degrees": -78.69006752597979,
            "depth_orientation_xyz_radians": [-1.373400766945016, 0.0, 0.0],
            "policy_rgb_orientation_unchanged": True,
            "target_geometry": target_geometry,
        }
        if (
            not isinstance(response, dict)
            or set(response) != expected
            or response["schema_version"] != 1
            or response["source"]
            != "train-simulator-immediate-footprint-visual-visibility-ceiling"
            or response["policy_observation_unchanged"] is not True
            or response["state_restored"] is not True
            or response["repeat_count"] != 2
            or response["observation_contract"] != contract
        ):
            raise ValueError("G7.14 response provenance or preregistration changed")
        position_error = _finite_number(
            response, "max_position_restore_error_m", lower=0
        )
        rotation_error = _finite_number(
            response, "max_rotation_restore_error", lower=0
        )
        if position_error > 1e-6 or rotation_error > 1e-6:
            raise RuntimeError("G7.14 state restoration exceeds tolerance")
        rows = response["candidates"]
        if not isinstance(rows, list) or len(rows) != len(payload):
            raise ValueError("G7.14 observation-case count changed")
        repeat_keys = {
            "turn_collisions", "executed_prefix_count", "prefix_trace",
            "pre_action_pose", "forward_direction", "metric_depth",
            "depth_sha256", "sensor_extrinsics",
        }
        pose_keys = {"position", "rotation_coefficients"}
        primitive_keys = {
            "primitive_index", "action", "collision", "before_position",
            "after_position", "displacement_m", "forward_progress_m",
            "lateral_displacement_m",
        }
        extrinsic_keys = {
            "relative_position_agent_frame", "optical_axis_agent_frame",
            "expected_relative_position_agent_frame",
            "expected_optical_axis_agent_frame",
        }

        def finite_vector(value, length, name):
            if (
                not isinstance(value, list)
                or len(value) != length
                or any(
                    isinstance(item, bool)
                    or not isinstance(item, (int, float))
                    or not math.isfinite(float(item))
                    for item in value
                )
            ):
                raise ValueError("G7.14 " + name + " is invalid")
            return [float(item) for item in value]

        normalized = []
        for request, row in zip(payload, rows):
            first_forward = request["actions"].index(1)
            expected_turn = request["actions"][:first_forward]
            expected_forward = request["actions"][first_forward:]
            if (
                not isinstance(row, dict)
                or set(row) != {
                    "name", "actions", "turn_prefix", "forward_suffix",
                    "observe_before_forward_index", "repeats",
                }
                or row["name"] != request["name"]
                or row["actions"] != request["actions"]
                or row["turn_prefix"] != expected_turn
                or row["forward_suffix"] != expected_forward
                or row["observe_before_forward_index"]
                != request["observe_before_forward_index"]
                or not isinstance(row["repeats"], list)
                or len(row["repeats"]) != 2
            ):
                raise ValueError("G7.14 observation-case response changed")
            repeats = []
            prefix_count = request["observe_before_forward_index"] - 1
            for repeat in row["repeats"]:
                if not isinstance(repeat, dict) or set(repeat) != repeat_keys:
                    raise ValueError("G7.14 repeat schema changed")
                if (
                    repeat["executed_prefix_count"] != prefix_count
                    or not isinstance(repeat["turn_collisions"], list)
                    or len(repeat["turn_collisions"]) != len(expected_turn)
                    or any(type(value) is not bool for value in repeat["turn_collisions"])
                    or not isinstance(repeat["prefix_trace"], list)
                    or len(repeat["prefix_trace"]) != prefix_count
                    or not isinstance(repeat["depth_sha256"], str)
                    or len(repeat["depth_sha256"]) != 64
                    or any(character not in "0123456789abcdef" for character in repeat["depth_sha256"])
                    or not isinstance(repeat["pre_action_pose"], dict)
                    or set(repeat["pre_action_pose"]) != pose_keys
                    or not isinstance(repeat["sensor_extrinsics"], dict)
                    or set(repeat["sensor_extrinsics"]) != extrinsic_keys
                ):
                    raise ValueError("G7.14 repeat identity changed")
                primitives = []
                for index, primitive in enumerate(repeat["prefix_trace"]):
                    if (
                        not isinstance(primitive, dict)
                        or set(primitive) != primitive_keys
                        or primitive["primitive_index"] != index
                        or primitive["action"] != 1
                        or type(primitive["collision"]) is not bool
                    ):
                        raise ValueError("G7.14 prefix trace changed")
                    normalized_primitive = dict(primitive)
                    normalized_primitive["before_position"] = finite_vector(
                        primitive["before_position"], 3, "before position"
                    )
                    normalized_primitive["after_position"] = finite_vector(
                        primitive["after_position"], 3, "after position"
                    )
                    for key in (
                        "displacement_m", "forward_progress_m",
                        "lateral_displacement_m",
                    ):
                        normalized_primitive[key] = _finite_number(
                            primitive, key,
                            lower=0 if key != "forward_progress_m" else None,
                        )
                    primitives.append(normalized_primitive)
                extrinsics = {
                    key: finite_vector(repeat["sensor_extrinsics"][key], 3, key)
                    for key in extrinsic_keys
                }
                if (
                    extrinsics["expected_relative_position_agent_frame"]
                    != [0.0, 1.25, 0.0]
                    or any(
                        abs(left - right) > 1e-12
                        for left, right in zip(
                            extrinsics["expected_optical_axis_agent_frame"],
                            [0.0, -0.9805806756909201, -0.19611613513818404],
                        )
                    )
                ):
                    raise ValueError("G7.14 expected sensor extrinsics changed")
                repeats.append(
                    {
                        **repeat,
                        "prefix_trace": primitives,
                        "pre_action_pose": {
                            "position": finite_vector(
                                repeat["pre_action_pose"]["position"], 3,
                                "pre-action position",
                            ),
                            "rotation_coefficients": finite_vector(
                                repeat["pre_action_pose"]["rotation_coefficients"], 4,
                                "pre-action rotation",
                            ),
                        },
                        "forward_direction": finite_vector(
                            repeat["forward_direction"], 3, "forward direction"
                        ),
                        "sensor_extrinsics": extrinsics,
                        "metric_depth": decode_metric_depth_payload(
                            repeat["metric_depth"], depth_size=self.rgb_size
                        ),
                    }
                )
            normalized.append(
                {
                    "name": row["name"],
                    "actions": list(row["actions"]),
                    "turn_prefix": list(row["turn_prefix"]),
                    "forward_suffix": list(row["forward_suffix"]),
                    "observe_before_forward_index": row[
                        "observe_before_forward_index"
                    ],
                    "repeats": repeats,
                }
            )
        return {
            "source": response["source"],
            "repeat_count": 2,
            "observation_contract": contract,
            "max_position_restore_error_m": position_error,
            "max_rotation_restore_error": rotation_error,
            "candidates": normalized,
        }

    def probe_immediate_footprint_outcomes(self, candidates):
        """Return G7.15 action-preceding depth plus one-forward outcomes."""
        if self.dataset_split != "train":
            raise RuntimeError("G7.15 executed-outcome audit is restricted to train")
        if not isinstance(candidates, (list, tuple)) or len(candidates) != 32:
            raise ValueError("G7.15 requires the exact 32-case bank")
        payload = []
        names = set()
        for candidate in candidates:
            if not isinstance(candidate, dict) or set(candidate) != {
                "name", "actions", "observe_before_forward_index",
            }:
                raise ValueError("G7.15 observation-case schema changed")
            name = candidate["name"]
            actions = candidate["actions"]
            critical = candidate["observe_before_forward_index"]
            if (
                not isinstance(name, str)
                or not name
                or name in names
                or not isinstance(actions, list)
                or not 2 <= len(actions) <= 8
                or any(type(action) is not int or action not in (1, 2, 3) for action in actions)
                or 1 not in actions
                or type(critical) is not int
            ):
                raise ValueError("G7.15 observation-case identity is invalid")
            first_forward = actions.index(1)
            forward_suffix = actions[first_forward:]
            if (
                len(forward_suffix) < 2
                or any(action == 1 for action in actions[:first_forward])
                or any(action != 1 for action in forward_suffix)
                or critical != len(forward_suffix)
            ):
                raise ValueError("G7.15 must observe before the final forward primitive")
            names.add(name)
            payload.append(
                {
                    "name": name,
                    "actions": list(actions),
                    "observe_before_forward_index": critical,
                }
            )
        response = self._request(
            {"cmd": "probe_immediate_footprint_outcomes", "candidates": payload},
            timeout=900,
        )
        expected = {
            "schema_version", "source", "policy_observation_unchanged",
            "state_restored", "repeat_count", "observation_contract",
            "max_position_restore_error_m", "max_rotation_restore_error",
            "candidates",
        }
        target_geometry = {
            "target_ray_count": 48,
            "forward_bin_counts": [18, 18, 12],
            "lateral_bin_counts": [16, 16, 16],
            "grid_counts": [[6, 6, 6], [6, 6, 6], [4, 4, 4]],
            "minimum_expected_forward_m": 0.033987915407854924,
            "maximum_expected_forward_m": 0.3115141955835961,
            "minimum_expected_lateral_m": -0.10053271911657698,
            "maximum_expected_lateral_m": 0.10053271911657698,
            "minimum_expected_optical_depth_m": 1.2323914232248423,
            "maximum_expected_optical_depth_m": 1.2868188046921853,
        }
        contract = {
            "timing": "after-turn-and-all-but-final-forward; before-final-forward",
            "critical_primitive_executed": True,
            "critical_action": 1,
            "critical_action_count_per_repeat": 1,
            "commanded_distance_m": 0.25,
            "depth_height": 64,
            "depth_width": 64,
            "hfov_degrees": 90.0,
            "fx": 32.0,
            "fy": 32.0,
            "cx": 31.5,
            "cy": 31.5,
            "sensor_height_m": 1.25,
            "forward_step_m": 0.25,
            "agent_radius_m": 0.1,
            "corridor_half_width_m": 0.12,
            "depth_pitch_radians": -1.373400766945016,
            "depth_pitch_degrees": -78.69006752597979,
            "depth_orientation_xyz_radians": [-1.373400766945016, 0.0, 0.0],
            "policy_rgb_orientation_unchanged": True,
            "target_geometry": target_geometry,
        }
        if (
            not isinstance(response, dict)
            or set(response) != expected
            or response["schema_version"] != 1
            or response["source"]
            != "train-simulator-frozen-immediate-footprint-executed-outcome-audit"
            or response["policy_observation_unchanged"] is not True
            or response["state_restored"] is not True
            or response["repeat_count"] != 2
            or response["observation_contract"] != contract
        ):
            raise ValueError("G7.15 response provenance or preregistration changed")
        position_error = _finite_number(
            response, "max_position_restore_error_m", lower=0
        )
        rotation_error = _finite_number(
            response, "max_rotation_restore_error", lower=0
        )
        if position_error > 1e-6 or rotation_error > 1e-6:
            raise RuntimeError("G7.15 state restoration exceeds tolerance")
        rows = response["candidates"]
        if not isinstance(rows, list) or len(rows) != len(payload):
            raise ValueError("G7.15 observation-case count changed")
        repeat_keys = {
            "turn_collisions", "executed_prefix_count", "prefix_trace",
            "pre_action_pose", "forward_direction", "metric_depth",
            "depth_sha256", "sensor_extrinsics", "critical_trace",
            "post_action_pose",
        }
        pose_keys = {"position", "rotation_coefficients"}
        prefix_keys = {
            "primitive_index", "action", "collision", "before_position",
            "after_position", "displacement_m", "forward_progress_m",
            "lateral_displacement_m",
        }
        critical_keys = prefix_keys | {"intended_endpoint", "endpoint_error_m"}
        extrinsic_keys = {
            "relative_position_agent_frame", "optical_axis_agent_frame",
            "expected_relative_position_agent_frame",
            "expected_optical_axis_agent_frame",
        }

        def finite_vector(value, length, name):
            if (
                not isinstance(value, list)
                or len(value) != length
                or any(
                    isinstance(item, bool)
                    or not isinstance(item, (int, float))
                    or not math.isfinite(float(item))
                    for item in value
                )
            ):
                raise ValueError("G7.15 " + name + " is invalid")
            return [float(item) for item in value]

        normalized = []
        for request, row in zip(payload, rows):
            first_forward = request["actions"].index(1)
            expected_turn = request["actions"][:first_forward]
            expected_forward = request["actions"][first_forward:]
            if (
                not isinstance(row, dict)
                or set(row) != {
                    "name", "actions", "turn_prefix", "forward_suffix",
                    "observe_before_forward_index", "repeats",
                }
                or row["name"] != request["name"]
                or row["actions"] != request["actions"]
                or row["turn_prefix"] != expected_turn
                or row["forward_suffix"] != expected_forward
                or row["observe_before_forward_index"] != len(expected_forward)
                or not isinstance(row["repeats"], list)
                or len(row["repeats"]) != 2
            ):
                raise ValueError("G7.15 observation-case response changed")
            repeats = []
            prefix_count = len(expected_forward) - 1
            for repeat in row["repeats"]:
                if not isinstance(repeat, dict) or set(repeat) != repeat_keys:
                    raise ValueError("G7.15 repeat schema changed")
                if (
                    repeat["executed_prefix_count"] != prefix_count
                    or not isinstance(repeat["turn_collisions"], list)
                    or len(repeat["turn_collisions"]) != len(expected_turn)
                    or any(type(value) is not bool for value in repeat["turn_collisions"])
                    or not isinstance(repeat["prefix_trace"], list)
                    or len(repeat["prefix_trace"]) != prefix_count
                    or not isinstance(repeat["depth_sha256"], str)
                    or len(repeat["depth_sha256"]) != 64
                    or any(character not in "0123456789abcdef" for character in repeat["depth_sha256"])
                    or not isinstance(repeat["pre_action_pose"], dict)
                    or set(repeat["pre_action_pose"]) != pose_keys
                    or not isinstance(repeat["post_action_pose"], dict)
                    or set(repeat["post_action_pose"]) != pose_keys
                    or not isinstance(repeat["sensor_extrinsics"], dict)
                    or set(repeat["sensor_extrinsics"]) != extrinsic_keys
                ):
                    raise ValueError("G7.15 repeat identity changed")
                primitives = []
                for index, primitive in enumerate(repeat["prefix_trace"]):
                    if (
                        not isinstance(primitive, dict)
                        or set(primitive) != prefix_keys
                        or primitive["primitive_index"] != index
                        or primitive["action"] != 1
                        or type(primitive["collision"]) is not bool
                    ):
                        raise ValueError("G7.15 prefix trace changed")
                    item = dict(primitive)
                    item["before_position"] = finite_vector(
                        primitive["before_position"], 3, "prefix before position"
                    )
                    item["after_position"] = finite_vector(
                        primitive["after_position"], 3, "prefix after position"
                    )
                    for key in (
                        "displacement_m", "forward_progress_m",
                        "lateral_displacement_m",
                    ):
                        item[key] = _finite_number(
                            primitive,
                            key,
                            lower=0 if key != "forward_progress_m" else None,
                        )
                    primitives.append(item)
                critical = repeat["critical_trace"]
                if (
                    not isinstance(critical, dict)
                    or set(critical) != critical_keys
                    or critical["primitive_index"] != 0
                    or critical["action"] != 1
                    or type(critical["collision"]) is not bool
                ):
                    raise ValueError("G7.15 critical trace changed")
                critical_normalized = dict(critical)
                for key in (
                    "before_position", "after_position", "intended_endpoint",
                ):
                    critical_normalized[key] = finite_vector(
                        critical[key], 3, "critical " + key
                    )
                for key in (
                    "displacement_m", "forward_progress_m",
                    "lateral_displacement_m", "endpoint_error_m",
                ):
                    critical_normalized[key] = _finite_number(
                        critical,
                        key,
                        lower=0 if key != "forward_progress_m" else None,
                    )
                pre_pose = {
                    "position": finite_vector(
                        repeat["pre_action_pose"]["position"], 3,
                        "pre-action position",
                    ),
                    "rotation_coefficients": finite_vector(
                        repeat["pre_action_pose"]["rotation_coefficients"], 4,
                        "pre-action rotation",
                    ),
                }
                post_pose = {
                    "position": finite_vector(
                        repeat["post_action_pose"]["position"], 3,
                        "post-action position",
                    ),
                    "rotation_coefficients": finite_vector(
                        repeat["post_action_pose"]["rotation_coefficients"], 4,
                        "post-action rotation",
                    ),
                }
                direction = finite_vector(
                    repeat["forward_direction"], 3, "forward direction"
                )
                intended = [
                    pre_pose["position"][index] + 0.25 * direction[index]
                    for index in range(3)
                ]
                if (
                    math.dist(critical_normalized["before_position"], pre_pose["position"]) > 1e-6
                    or math.dist(critical_normalized["after_position"], post_pose["position"]) > 1e-6
                    or math.dist(critical_normalized["intended_endpoint"], intended) > 1e-6
                ):
                    raise ValueError("G7.15 critical pose geometry changed")
                extrinsics = {
                    key: finite_vector(repeat["sensor_extrinsics"][key], 3, key)
                    for key in extrinsic_keys
                }
                if (
                    extrinsics["expected_relative_position_agent_frame"]
                    != [0.0, 1.25, 0.0]
                    or any(
                        abs(left - right) > 1e-12
                        for left, right in zip(
                            extrinsics["expected_optical_axis_agent_frame"],
                            [0.0, -0.9805806756909201, -0.19611613513818404],
                        )
                    )
                ):
                    raise ValueError("G7.15 expected sensor extrinsics changed")
                repeats.append(
                    {
                        **repeat,
                        "prefix_trace": primitives,
                        "pre_action_pose": pre_pose,
                        "post_action_pose": post_pose,
                        "forward_direction": direction,
                        "sensor_extrinsics": extrinsics,
                        "critical_trace": critical_normalized,
                        "metric_depth": decode_metric_depth_payload(
                            repeat["metric_depth"], depth_size=self.rgb_size
                        ),
                    }
                )
            normalized.append(
                {
                    "name": row["name"],
                    "actions": list(row["actions"]),
                    "turn_prefix": list(row["turn_prefix"]),
                    "forward_suffix": list(row["forward_suffix"]),
                    "observe_before_forward_index": row[
                        "observe_before_forward_index"
                    ],
                    "repeats": repeats,
                }
            )
        return {
            "source": response["source"],
            "repeat_count": 2,
            "observation_contract": contract,
            "max_position_restore_error_m": position_error,
            "max_rotation_restore_error": rotation_error,
            "candidates": normalized,
        }
    def probe_severity_audit(self, candidates):
        """Return G7.10 repeated motion-severity and clearance traces."""
        if self.dataset_split != "train":
            raise RuntimeError("severity audit is restricted to train")
        if not isinstance(candidates, (list, tuple)) or len(candidates) != 32:
            raise ValueError("G7.10 severity audit requires exactly 32 candidates")
        payload = []
        names = set()
        for candidate in candidates:
            name = getattr(candidate, "name", None)
            actions = getattr(candidate, "actions", None)
            if (
                not isinstance(name, str)
                or not name
                or name in names
                or not isinstance(actions, tuple)
                or not 2 <= len(actions) <= 8
                or any(type(action) is not int or action not in (1, 2, 3) for action in actions)
                or 1 not in actions
            ):
                raise ValueError("severity audit input violates exact-macro contract")
            first_forward = actions.index(1)
            if (
                len(actions[first_forward:]) < 2
                or any(action == 1 for action in actions[:first_forward])
                or any(action != 1 for action in actions[first_forward:])
            ):
                raise ValueError("severity audit requires turn prefix then >=2 forwards")
            names.add(name)
            payload.append({"name": name, "actions": list(actions)})
        response = self._request(
            {"cmd": "probe_severity_audit", "candidates": payload}
        )
        expected = {
            "schema_version", "source", "policy_observation_unchanged",
            "state_restored", "forward_step_m", "horizon_forward_steps",
            "physical_horizon_m", "repeat_count",
            "max_position_restore_error_m", "max_rotation_restore_error",
            "clearance_spacing_m", "clearance_search_radius_m", "candidates",
        }
        if not isinstance(response, dict) or set(response) != expected:
            raise ValueError("severity audit response schema mismatch")
        if (
            response["schema_version"] != 1
            or response["source"] != "train-simulator-frozen-severity-clearance-audit"
            or response["policy_observation_unchanged"] is not True
            or response["state_restored"] is not True
            or response["forward_step_m"] != 0.25
            or response["horizon_forward_steps"] != 2
            or response["physical_horizon_m"] != 0.5
            or response["repeat_count"] != 2
            or response["clearance_spacing_m"] != 0.125
            or response["clearance_search_radius_m"] != 2.0
        ):
            raise ValueError("severity audit provenance or preregistration changed")
        position_error = _finite_number(
            response, "max_position_restore_error_m", lower=0
        )
        rotation_error = _finite_number(
            response, "max_rotation_restore_error", lower=0
        )
        if position_error > 1e-6 or rotation_error > 1e-6:
            raise RuntimeError("severity audit restoration exceeds tolerance")
        rows = response["candidates"]
        if not isinstance(rows, list) or len(rows) != len(payload):
            raise ValueError("severity audit result count mismatch")

        def finite_vector(value, length, name):
            if (
                not isinstance(value, list)
                or len(value) != length
                or any(
                    isinstance(item, bool)
                    or not isinstance(item, (int, float))
                    or not math.isfinite(float(item))
                    for item in value
                )
            ):
                raise ValueError("severity audit " + name + " is invalid")
            return [float(item) for item in value]

        def clearance_trace(value, name):
            required = {
                "path_distance_m", "position", "clearance_m",
                "nearest_obstacle_within_radius", "navigable",
            }
            if not isinstance(value, list) or len(value) < 2:
                raise ValueError("severity audit " + name + " is empty")
            normalized = []
            previous_distance = -1.0
            for item in value:
                if not isinstance(item, dict) or set(item) != required:
                    raise ValueError("severity audit clearance schema mismatch")
                distance = _finite_number(item, "path_distance_m", lower=0)
                clearance = _finite_number(item, "clearance_m", lower=0)
                if (
                    distance + 1e-9 < previous_distance
                    or clearance > 2.0 + 1e-6
                    or type(item["nearest_obstacle_within_radius"]) is not bool
                    or type(item["navigable"]) is not bool
                ):
                    raise ValueError("severity audit clearance values are invalid")
                normalized.append({
                    "path_distance_m": distance,
                    "position": finite_vector(item["position"], 3, "clearance position"),
                    "clearance_m": clearance,
                    "nearest_obstacle_within_radius": item[
                        "nearest_obstacle_within_radius"
                    ],
                    "navigable": item["navigable"],
                })
                previous_distance = distance
            return normalized

        repeat_extra = {
            "clearance_spacing_m", "clearance_search_radius_m",
            "intended_clearance_trace", "actual_clearance_trace",
            "minimum_intended_clearance_m", "minimum_actual_clearance_m",
            "no_sliding_endpoint", "no_sliding_endpoint_error_m",
        }
        base_repeat = {
            "turn_collisions", "post_turn_pose", "metric_depth", "depth_sha256",
            "forward_direction", "commanded_distance_m", "intended_endpoint",
            "privileged_endpoint", "privileged_endpoint_error_m",
            "privileged_swept_path_free", "primitive_trace",
            "full_collision_trace", "full_unsafe", "horizon_collision_trace",
            "horizon_unsafe", "final_pose", "realized_forward_m",
            "realized_lateral_error_m", "endpoint_error_m",
            "heading_error_degrees",
        }
        normalized_rows = []
        for request, row in zip(payload, rows):
            if (
                not isinstance(row, dict)
                or set(row) != {
                    "name", "actions", "turn_prefix", "forward_suffix", "repeats"
                }
                or row["name"] != request["name"]
                or row["actions"] != request["actions"]
                or not isinstance(row["repeats"], list)
                or len(row["repeats"]) != 2
            ):
                raise ValueError("severity audit candidate identity changed")
            first_forward = request["actions"].index(1)
            if (
                row["turn_prefix"] != request["actions"][:first_forward]
                or row["forward_suffix"] != request["actions"][first_forward:]
            ):
                raise ValueError("severity audit phase split changed")
            repeats = []
            for repeat in row["repeats"]:
                if not isinstance(repeat, dict) or set(repeat) != base_repeat | repeat_extra:
                    raise ValueError("severity audit repeat schema mismatch")
                if (
                    repeat["clearance_spacing_m"] != 0.125
                    or repeat["clearance_search_radius_m"] != 2.0
                    or not isinstance(repeat["depth_sha256"], str)
                    or len(repeat["depth_sha256"]) != 64
                ):
                    raise ValueError("severity audit repeat provenance changed")
                full_trace = repeat["full_collision_trace"]
                primitives = repeat["primitive_trace"]
                if (
                    not isinstance(full_trace, list)
                    or len(full_trace) != len(row["forward_suffix"])
                    or any(type(value) is not bool for value in full_trace)
                    or not isinstance(primitives, list)
                    or len(primitives) != len(full_trace)
                    or repeat["full_unsafe"] is not any(full_trace)
                ):
                    raise ValueError("severity audit collision trace is invalid")
                intended = clearance_trace(
                    repeat["intended_clearance_trace"], "intended trace"
                )
                actual = clearance_trace(
                    repeat["actual_clearance_trace"], "actual trace"
                )
                minimum_intended = _finite_number(
                    repeat, "minimum_intended_clearance_m", lower=0
                )
                minimum_actual = _finite_number(
                    repeat, "minimum_actual_clearance_m", lower=0
                )
                if (
                    abs(minimum_intended - min(item["clearance_m"] for item in intended)) > 1e-6
                    or abs(minimum_actual - min(item["clearance_m"] for item in actual)) > 1e-6
                ):
                    raise ValueError("severity audit minimum clearance changed")
                normalized = dict(repeat)
                normalized["metric_depth"] = decode_metric_depth_payload(
                    repeat["metric_depth"], depth_size=self.rgb_size
                )
                normalized["intended_clearance_trace"] = intended
                normalized["actual_clearance_trace"] = actual
                normalized["minimum_intended_clearance_m"] = minimum_intended
                normalized["minimum_actual_clearance_m"] = minimum_actual
                normalized["no_sliding_endpoint"] = finite_vector(
                    repeat["no_sliding_endpoint"], 3, "no-sliding endpoint"
                )
                normalized["no_sliding_endpoint_error_m"] = _finite_number(
                    repeat, "no_sliding_endpoint_error_m", lower=0
                )
                repeats.append(normalized)
            normalized_rows.append({
                "name": row["name"],
                "actions": list(row["actions"]),
                "turn_prefix": list(row["turn_prefix"]),
                "forward_suffix": list(row["forward_suffix"]),
                "repeats": repeats,
            })
        return {
            "source": response["source"],
            "forward_step_m": 0.25,
            "repeat_count": 2,
            "clearance_spacing_m": 0.125,
            "clearance_search_radius_m": 2.0,
            "max_position_restore_error_m": position_error,
            "max_rotation_restore_error": rotation_error,
            "candidates": normalized_rows,
        }


    def probe_aligned_candidates(self, candidates):
        """Return train-only candidate-aligned RGB and forward-collision labels."""
        if self.dataset_split != "train":
            raise RuntimeError("aligned candidate probes are restricted to train")
        if not isinstance(candidates, (list, tuple)) or not 1 <= len(candidates) <= 64:
            raise ValueError("aligned probes require 1..64 candidates")
        payload = []
        names = set()
        for candidate in candidates:
            name = getattr(candidate, "name", None)
            actions = getattr(candidate, "actions", None)
            if (
                not isinstance(name, str)
                or not name
                or name in names
                or not isinstance(actions, tuple)
                or not 1 <= len(actions) <= 8
                or any(type(action) is not int or action not in (1, 2, 3) for action in actions)
                or 1 not in actions
            ):
                raise ValueError("aligned probe input violates macro contract")
            names.add(name)
            payload.append({"name": name, "actions": list(actions)})
        response = self._request(
            {"cmd": "probe_aligned_candidates", "candidates": payload}
        )
        expected = {
            "schema_version", "source", "policy_observation_unchanged",
            "state_restored", "max_position_restore_error_m",
            "max_rotation_restore_error", "candidates",
        }
        if not isinstance(response, dict) or set(response) != expected:
            raise ValueError("aligned probe response schema mismatch")
        if (
            response["schema_version"] != 1
            or response["source"] != "train-simulator-aligned-branch-rollout"
            or response["policy_observation_unchanged"] is not True
            or response["state_restored"] is not True
        ):
            raise ValueError("aligned probe provenance or restoration failed")
        position_error = _finite_number(
            response, "max_position_restore_error_m", lower=0
        )
        rotation_error = _finite_number(
            response, "max_rotation_restore_error", lower=0
        )
        if position_error > 1e-6 or rotation_error > 1e-6:
            raise RuntimeError("aligned probe restoration exceeds tolerance")
        rows = response["candidates"]
        if not isinstance(rows, list) or len(rows) != len(payload):
            raise ValueError("aligned probe result count mismatch")
        normalized = []
        for request, row in zip(payload, rows):
            required = {
                "name", "actions", "turn_prefix", "forward_suffix",
                "aligned_rgb", "forward_collisions", "unsafe", "end_position",
            }
            if not isinstance(row, dict) or set(row) != required:
                raise ValueError("aligned probe row schema mismatch")
            if row["name"] != request["name"] or row["actions"] != request["actions"]:
                raise RuntimeError("aligned probe identity changed")
            first_forward = request["actions"].index(1)
            if (
                row["turn_prefix"] != request["actions"][:first_forward]
                or row["forward_suffix"] != request["actions"][first_forward:]
            ):
                raise RuntimeError("aligned probe phase split changed")
            rgb = row["aligned_rgb"]
            if not isinstance(rgb, dict) or set(rgb) != RGB_KEYS:
                raise ValueError("aligned probe RGB schema mismatch")
            if rgb["encoding"] != RGB_ENCODING or rgb["shape_hwc"] != [
                self.rgb_size, self.rgb_size, 3
            ]:
                raise ValueError("aligned probe RGB metadata mismatch")
            try:
                pixels = base64.b64decode(rgb["data_b64"], validate=True)
            except (binascii.Error, ValueError, TypeError) as exc:
                raise ValueError("aligned probe RGB base64 is invalid") from exc
            if len(pixels) != self.rgb_size * self.rgb_size * 3:
                raise ValueError("aligned probe RGB byte count mismatch")
            collisions = row["forward_collisions"]
            if (
                not isinstance(collisions, list)
                or len(collisions) != len(row["forward_suffix"])
                or any(type(value) is not bool for value in collisions)
                or type(row["unsafe"]) is not bool
                or row["unsafe"] != any(collisions)
            ):
                raise ValueError("aligned probe collision labels are invalid")
            end_position = row["end_position"]
            if (
                not isinstance(end_position, list)
                or len(end_position) != 3
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in end_position
                )
            ):
                raise ValueError("aligned probe end position is invalid")
            normalized.append({
                "name": row["name"],
                "actions": list(row["actions"]),
                "turn_prefix": list(row["turn_prefix"]),
                "forward_suffix": list(row["forward_suffix"]),
                "aligned_rgb": np.frombuffer(
                    pixels, dtype=np.uint8
                ).copy().reshape(self.rgb_size, self.rgb_size, 3),
                "forward_collisions": list(collisions),
                "unsafe": bool(row["unsafe"]),
                "end_position": [float(value) for value in end_position],
            })
        return {
            "source": response["source"],
            "state_restored": True,
            "policy_observation_unchanged": True,
            "max_position_restore_error_m": position_error,
            "max_rotation_restore_error": rotation_error,
            "candidates": normalized,
        }

    def probe_candidates(self, candidates):
        """Query train-only branch-rollout collision labels outside policy observations."""
        if self.dataset_split != "train":
            raise RuntimeError("candidate probes are restricted to the train split")
        if not isinstance(candidates, (list, tuple)) or not 1 <= len(candidates) <= 64:
            raise ValueError("candidate probes require 1..64 candidates")
        payload = []
        names = set()
        for candidate in candidates:
            name = getattr(candidate, "name", None)
            actions = getattr(candidate, "actions", None)
            if (
                not isinstance(name, str)
                or not name
                or name in names
                or not isinstance(actions, tuple)
                or not 1 <= len(actions) <= 8
                or any(type(action) is not int or action not in (1, 2, 3) for action in actions)
                or 1 not in actions
            ):
                raise ValueError("candidate probe input violates the macro contract")
            names.add(name)
            payload.append({"name": name, "actions": list(actions)})
        response = self._request({"cmd": "probe_candidates", "candidates": payload})
        expected = {
            "schema_version", "source", "policy_observation_unchanged",
            "state_restored", "max_position_restore_error_m",
            "max_rotation_restore_error", "candidates",
        }
        if not isinstance(response, dict) or set(response) != expected:
            raise ValueError("candidate probe response schema mismatch")
        if (
            response["schema_version"] != 1
            or response["source"] != "train-simulator-branch-rollout"
            or response["policy_observation_unchanged"] is not True
            or response["state_restored"] is not True
        ):
            raise ValueError("candidate probe provenance or restoration guard failed")
        position_error = _finite_number(
            response, "max_position_restore_error_m", lower=0
        )
        rotation_error = _finite_number(
            response, "max_rotation_restore_error", lower=0
        )
        if position_error > 1e-6 or rotation_error > 1e-6:
            raise RuntimeError("candidate probe restoration exceeds tolerance")
        rows = response["candidates"]
        if not isinstance(rows, list) or len(rows) != len(payload):
            raise ValueError("candidate probe result count mismatch")
        normalized = []
        for request, row in zip(payload, rows):
            required = {
                "name", "actions", "collisions", "unsafe", "forward_attempts",
                "forward_collisions", "end_position",
            }
            if not isinstance(row, dict) or set(row) != required:
                raise ValueError("candidate probe row schema mismatch")
            if row["name"] != request["name"] or row["actions"] != request["actions"]:
                raise RuntimeError("candidate probe identity changed")
            collisions = row["collisions"]
            if (
                not isinstance(collisions, list)
                or len(collisions) != len(request["actions"])
                or any(type(value) is not bool for value in collisions)
                or type(row["unsafe"]) is not bool
            ):
                raise ValueError("candidate probe collision labels are invalid")
            forward_labels = [
                collision for action, collision in zip(request["actions"], collisions)
                if action == 1
            ]
            if (
                row["unsafe"] != any(forward_labels)
                or row["forward_attempts"] != len(forward_labels)
                or row["forward_collisions"] != sum(forward_labels)
            ):
                raise RuntimeError("candidate probe collision aggregate mismatch")
            end_position = row["end_position"]
            if (
                not isinstance(end_position, list)
                or len(end_position) != 3
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in end_position
                )
            ):
                raise ValueError("candidate probe end position is invalid")
            normalized.append({
                "name": row["name"],
                "actions": list(row["actions"]),
                "collisions": list(collisions),
                "unsafe": bool(row["unsafe"]),
                "forward_attempts": int(row["forward_attempts"]),
                "forward_collisions": int(row["forward_collisions"]),
                "end_position": [float(value) for value in end_position],
            })
        return {
            "source": response["source"],
            "state_restored": True,
            "policy_observation_unchanged": True,
            "max_position_restore_error_m": position_error,
            "max_rotation_restore_error": rotation_error,
            "candidates": normalized,
        }

    def step(self, actions):
        array = np.asarray(actions)
        if array.size != 1:
            raise ValueError("R2R step expects one scalar action")
        item = array.reshape(-1)[0]
        if not np.issubdtype(type(item), np.integer):
            raise ValueError("R2R action must be an integer category")
        action = int(item)
        if not 0 <= action < len(ACTIONS):
            raise ValueError("R2R action outside 0..3")
        response = self._request({"cmd": "step", "action": action})
        try:
            info = response.get("info")
            self._validate_identity(info)
            obs = self._obs(response)
            reward = _finite_number(response, "reward")
            terminated = response.get("terminated")
            truncated = response.get("truncated")
            frozen = info.get("terminal_frozen")
            collision = info.get("collision")
            if any(type(value) is not bool for value in (
                terminated, truncated, frozen, collision
            )):
                raise ValueError("Habitat R2R done/frozen/collision flags must be booleans")
            if terminated and truncated:
                raise ValueError("Habitat R2R cannot terminate and truncate together")
            if frozen:
                if self._terminal_metrics is None or terminated or truncated or reward != 0:
                    raise RuntimeError("Habitat R2R terminal freeze protocol mismatch")
            elif self._terminal_metrics is not None:
                raise RuntimeError("Habitat R2R resumed an ended episode")
            else:
                self._elapsed_steps += 1
            infos = {
                "success": torch.tensor([
                    _finite_number(info, "success", lower=0, upper=1)
                ], dtype=torch.float32),
                "collision": torch.tensor([collision], dtype=torch.bool),
                # Causal executor feedback.  These values are measured after
                # the requested primitive and are intentionally distinct from
                # simulator-oracle distance-to-goal or navmesh state.
                "path_length": torch.tensor([
                    _finite_number(info, "path_length", lower=0)
                ], dtype=torch.float32),
                "collision_count": torch.tensor([
                    _finite_number(info, "collision_count", lower=0)
                ], dtype=torch.float32),
                "terminal_frozen": torch.tensor([frozen], dtype=torch.bool),
            }
            if terminated or truncated or frozen:
                infos["episode"], metrics = self._episode_info(info)
                if not frozen:
                    self._terminal_metrics = metrics
        except BaseException:
            self.close()
            raise
        self._is_start = False
        return (
            obs,
            torch.tensor([reward], dtype=torch.float32),
            torch.tensor([terminated], dtype=torch.bool),
            torch.tensor([truncated], dtype=torch.bool),
            infos,
        )

    def chunk_step(self, chunk_actions):
        array = np.asarray(chunk_actions)
        if array.shape != (1, 1, 1):
            raise ValueError("R2R chunk actions must have shape [1, 1, 1]")
        obs, reward, terminated, truncated, infos = self.step(array[:, 0, :])
        return (
            [obs], reward.unsqueeze(1), terminated.unsqueeze(1),
            truncated.unsqueeze(1), [infos],
        )

    def update_reset_state_ids(self):
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
