"""Single-scene Habitat R2R JSONL service for the isolated RLinf VLN bridge.

Policy observations contain RGB and official R2R instruction IDs/mask only.
The Habitat goal, pose and reference path remain inside this process. Scalar
reward and episode metrics travel in separate response fields for PPO/logging.
This service targets the private Habitat-Lab 0.2.4 Python 3.9/Mesa environment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import json
import math
import os
import secrets
import sys
import traceback
import types
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


class _OfficialFrameNamespaceError(RuntimeError):
    """The Habitat child cannot safely install its model-free protocol package."""


_CHILD_NAMESPACE_ROOT_ATTRIBUTE = "_streamvln_habitat_child_namespace_root"


def _install_streamvln_protocol_namespace(repo_root):
    """Install canonical lightweight packages without executing RLinf initializers."""

    root = Path(repo_root).resolve()
    package_paths = (
        ("rlinf", root / "rlinf"),
        ("rlinf.envs", root / "rlinf" / "envs"),
        (
            "rlinf.envs.streamvln",
            root / "rlinf" / "envs" / "streamvln",
        ),
    )
    for name, path in package_paths:
        if not path.is_dir():
            raise _OfficialFrameNamespaceError(
                f"Habitat child protocol package directory is missing: {name}"
            )
    protocol_root = package_paths[-1][1]
    for filename in (
        "streamvln_p8_official_keep.py",
        "streamvln_p8_official_frame.py",
    ):
        if not (protocol_root / filename).is_file():
            raise _OfficialFrameNamespaceError(
                f"Habitat child protocol module is missing: {filename}"
            )

    root_marker = str(root)
    for name, path in package_paths:
        existing = sys.modules.get(name)
        if existing is not None:
            if getattr(existing, _CHILD_NAMESPACE_ROOT_ATTRIBUTE, None) != root_marker:
                raise _OfficialFrameNamespaceError(
                    f"Habitat child refuses preloaded parent package: {name}"
                )
            if list(getattr(existing, "__path__", ())) != [str(path)]:
                raise _OfficialFrameNamespaceError(
                    f"Habitat child namespace path drifted: {name}"
                )
            continue

        package = types.ModuleType(name)
        package.__package__ = name
        package.__path__ = [str(path)]
        spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        spec.submodule_search_locations = [str(path)]
        package.__spec__ = spec
        setattr(package, _CHILD_NAMESPACE_ROOT_ATTRIBUTE, root_marker)
        sys.modules[name] = package
        parent_name, _, child_name = name.rpartition(".")
        if parent_name:
            setattr(sys.modules[parent_name], child_name, package)

_OFFICIAL_FRAME_PROTOCOL_ERROR = None
try:
    _install_streamvln_protocol_namespace(_REPO_ROOT)
    from rlinf.envs.streamvln.streamvln_p8_official_frame import (
        DEPTH_ENCODING,
        MATRIX_ENCODING,
        OFFICIAL_CAMERA_ORIENTATION_XYZ,
        OFFICIAL_CAMERA_POSITION_XYZ,
        OFFICIAL_DEPTH_MAX_M,
        OFFICIAL_DEPTH_MIN_M,
        OFFICIAL_FORWARD_STEP_M,
        OFFICIAL_FRAME_SCHEMA_VERSION,
        OFFICIAL_FRAME_SENSOR_CONTRACT,
        OFFICIAL_FRAME_SOURCE,
        OFFICIAL_RGB_HEIGHT,
        OFFICIAL_RGB_HFOV_DEG,
        OFFICIAL_RGB_WIDTH,
        OFFICIAL_TURN_ANGLE_DEG,
        RGB_ENCODING,
        encode_official_blob,
        official_frame_contract_sha256,
    )
except _OfficialFrameNamespaceError as exc:
    _OFFICIAL_FRAME_PROTOCOL_ERROR = exc
except ModuleNotFoundError as exc:
    if exc.name not in {
        "rlinf",
        "rlinf.envs",
        "rlinf.envs.streamvln",
        "rlinf.envs.streamvln.streamvln_p8_official_frame",
        "rlinf.envs.streamvln.streamvln_p8_official_keep",
    }:
        raise
    _OFFICIAL_FRAME_PROTOCOL_ERROR = exc

if __package__:
    from .habitat_r2r_rgb_instruction_smoke import (
        ACTIONS,
        file_sha256,
        make_policy_obs,
        make_task_config,
        read_r2r_metadata,
        validate_instruction,
    )
else:
    from habitat_r2r_rgb_instruction_smoke import (
    ACTIONS,
    file_sha256,
    make_policy_obs,
    make_task_config,
    read_r2r_metadata,
    validate_instruction,
    )


SUCCESS_DISTANCE_METERS = 3.0


def interpolate_path_point(points, lookahead_m):
    """Return the point at a bounded arc length along a polyline."""
    if not math.isfinite(lookahead_m) or lookahead_m <= 0:
        raise ValueError("expert lookahead must be positive and finite")
    if len(points) < 2:
        raise ValueError("shortest path must contain at least two points")
    remaining = float(lookahead_m)
    previous = points[0]
    for current in points[1:]:
        delta = current - previous
        length = float((delta * delta).sum() ** 0.5)
        if not math.isfinite(length):
            raise ValueError("shortest path contains a nonfinite segment")
        if length > 1e-8 and remaining <= length:
            return previous + delta * (remaining / length)
        remaining -= length
        previous = current
    return points[-1]


def local_waypoint_components(local_delta):
    """Convert Habitat camera coordinates to forward/left-positive meters."""
    if len(local_delta) < 3:
        raise ValueError("local waypoint delta must have three coordinates")
    forward_m = float(-local_delta[2])
    lateral_m = float(-local_delta[0])
    if not math.isfinite(forward_m) or not math.isfinite(lateral_m):
        raise ValueError("local waypoint delta contains nonfinite values")
    return forward_m, lateral_m


def vocabulary_sha256(words):
    payload = json.dumps(words, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def identity(frame):
    return {
        "episode_id": frame["episode_id"],
        "scene_id": frame["scene_id"],
        "instruction_sha256": frame["instruction_sha256"],
    }


class R2RSession:
    """STOP/timeout/reward protocol, kept independent of Habitat for testing."""

    def __init__(self, backend, *, max_steps, progress_weight=1.0,
                 success_bonus=10.0, step_penalty=0.01, collision_penalty=0.2):
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.backend = backend
        self.max_steps = max_steps
        self.progress_weight = progress_weight
        self.success_bonus = success_bonus
        self.step_penalty = step_penalty
        self.collision_penalty = collision_penalty
        self.previous_geodesic = None
        self.last_frame = None
        self.final_info = None
        self.steps = 0
        self.episode_generation = None

    def reset(self):
        frame = self.backend.reset()
        self.previous_geodesic = self._distance(frame)
        self.last_frame = frame
        self.final_info = None
        self.steps = 0
        info = identity(frame)
        if bool(getattr(self.backend, "official_streamvln_frame", False)):
            previous_generation = self.episode_generation
            self.episode_generation = secrets.token_hex(16)
            while self.episode_generation == previous_generation:
                self.episode_generation = secrets.token_hex(16)
            info["episode_generation"] = self.episode_generation
        else:
            self.episode_generation = None
        return {"obs": frame["policy_obs"], "info": info}

    @staticmethod
    def _distance(frame):
        value = float(frame["geodesic_distance"])
        if not math.isfinite(value) or value < 0:
            raise ValueError("R2R geodesic distance must be finite and nonnegative")
        return value

    def step(self, action):
        if self.last_frame is None:
            raise RuntimeError("reset must precede step")
        if type(action) is not int or not 0 <= action < len(ACTIONS):
            raise ValueError("R2R action must be integer 0..3")
        if self.final_info is not None:
            return {
                "obs": self.last_frame["policy_obs"],
                "reward": 0.0,
                "terminated": False,
                "truncated": False,
                "info": dict(self.final_info, terminal_frozen=True),
            }

        frame = self.backend.step(ACTIONS[action])
        if identity(frame) != identity(self.last_frame):
            raise RuntimeError("R2R episode/scene/instruction changed within rollout")
        self.steps += 1
        distance = self._distance(frame)
        success = bool(frame["success"])
        stop_called = action == 0
        if success and not stop_called:
            raise ValueError("Habitat reported R2R success without STOP")
        collision = bool(frame["collision"])
        reward = (
            self.progress_weight * (self.previous_geodesic - distance)
            + (self.success_bonus if success else 0.0)
            - self.step_penalty
            - (self.collision_penalty if collision else 0.0)
        )
        self.previous_geodesic = distance
        truncated = not stop_called and (
            self.steps >= self.max_steps or bool(frame["episode_over"])
        )
        terminated = stop_called
        info = {
            **identity(frame),
            "success": float(success),
            "spl": float(frame["spl"]),
            "distance_to_goal": distance,
            "path_length": float(frame["path_length"]),
            "collision_count": int(frame["collision_count"]),
            "collision": collision,
            "stop_called": stop_called,
            "timeout": truncated,
            "terminal_frozen": False,
        }
        if not math.isfinite(reward) or not math.isfinite(info["spl"]):
            raise ValueError("nonfinite R2R reward or SPL")
        self.last_frame = frame
        if terminated or truncated:
            self.final_info = dict(info)
        return {
            "obs": frame["policy_obs"],
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
        }

    def expert_waypoint(self, lookahead_m):
        """Return train-only supervision without adding it to policy observations."""
        if self.last_frame is None:
            raise RuntimeError("reset must precede expert supervision")
        if self.final_info is not None:
            raise RuntimeError("expert supervision is unavailable after episode end")
        return self.backend.expert_waypoint(lookahead_m)

    def observe_official_frame(self, *, expected_service_step, episode_generation):
        """Return the current private frame without advancing Habitat state."""

        if self.last_frame is None:
            raise RuntimeError("reset must precede official frame observation")
        if self.final_info is not None:
            raise RuntimeError("official frame is unavailable after episode end")
        if type(expected_service_step) is not int or expected_service_step != self.steps:
            raise RuntimeError("official frame request has a stale service step")
        if (
            not isinstance(episode_generation, str)
            or len(episode_generation) != 32
            or any(character not in "0123456789abcdef" for character in episode_generation)
            or episode_generation != self.episode_generation
        ):
            raise RuntimeError("official frame request has a stale episode generation")
        return self.backend.official_frame_payload(
            self.last_frame,
            self.steps,
            episode_generation,
        )

    def close(self):
        self.backend.close()


class HabitatR2RBackend:
    def __init__(self, args):
        self.official_streamvln_frame = bool(
            getattr(args, "official_streamvln_frame", False)
        )
        if self.official_streamvln_frame and _OFFICIAL_FRAME_PROTOCOL_ERROR is not None:
            raise RuntimeError(
                "official StreamVLN frame protocol modules are not deployed"
            ) from _OFFICIAL_FRAME_PROTOCOL_ERROR

        import habitat
        import numpy as np
        from habitat.config import read_write

        self.np = np
        self.include_depth = bool(getattr(args, "include_depth", False)) or (
            self.official_streamvln_frame
        )
        self.depth_max_m = float(getattr(args, "depth_max_m", 10.0))
        self.depth_pitch_radians = float(
            getattr(args, "depth_pitch_radians", 0.0)
        )
        self.dataset_path = args.dataset_path.resolve(strict=True)
        self.scenes_dir = args.scenes_dir.resolve(strict=True)
        raw, words, scene_file, _ = read_r2r_metadata(
            self.dataset_path, args.episode_index, self.scenes_dir
        )
        self.words = words
        self.vocab_sha256 = vocabulary_sha256(words)
        self.config = make_task_config(
            habitat, read_write, self.dataset_path, self.scenes_dir,
            args.split, args.max_steps, args.rgb_size,
            include_depth=self.include_depth,
            depth_max_m=self.depth_max_m,
            depth_pitch_radians=self.depth_pitch_radians,
        )
        with read_write(self.config):
            if self.official_streamvln_frame:
                from habitat.config.default_structured_configs import (
                    CompassSensorConfig,
                    GPSSensorConfig,
                )

                sensors = self.config.habitat.simulator.agents.main_agent.sim_sensors
                rgb_sensor = sensors.rgb_sensor
                depth_sensor = sensors.depth_sensor
                for sensor in (rgb_sensor, depth_sensor):
                    sensor.width = OFFICIAL_RGB_WIDTH
                    sensor.height = OFFICIAL_RGB_HEIGHT
                    sensor.hfov = OFFICIAL_RGB_HFOV_DEG
                    sensor.position = list(OFFICIAL_CAMERA_POSITION_XYZ)
                    sensor.orientation = list(OFFICIAL_CAMERA_ORIENTATION_XYZ)
                depth_sensor.normalize_depth = True
                depth_sensor.min_depth = OFFICIAL_DEPTH_MIN_M
                depth_sensor.max_depth = OFFICIAL_DEPTH_MAX_M
                self.config.habitat.task.lab_sensors.update(
                    {
                        "gps_sensor": GPSSensorConfig(),
                        "compass_sensor": CompassSensorConfig(),
                    }
                )
                self.config.habitat.simulator.forward_step_size = OFFICIAL_FORWARD_STEP_M
                self.config.habitat.simulator.turn_angle = OFFICIAL_TURN_ANGLE_DEG
            if "success_distance" in self.config.habitat.task:
                self.config.habitat.task.success_distance = SUCCESS_DISTANCE_METERS
            self.config.habitat.task.measurements.success.success_distance = (
                SUCCESS_DISTANCE_METERS
            )
            if "success_distance" in self.config.habitat.task.measurements.spl:
                self.config.habitat.task.measurements.spl.success_distance = (
                    SUCCESS_DISTANCE_METERS
                )
        dataset = habitat.make_dataset(
            id_dataset=self.config.habitat.dataset.type,
            config=self.config.habitat.dataset,
        )
        self.episode = dataset.episodes[args.episode_index]
        if (
            str(self.episode.episode_id) != str(raw["episode_id"])
            or Path(self.episode.scene_id).resolve() != scene_file
            or self.episode.instruction.instruction_text
            != raw["instruction"]["instruction_text"]
            or list(self.episode.instruction.instruction_tokens)
            != raw["instruction"]["instruction_tokens"]
        ):
            raise RuntimeError("Habitat R2R episode differs from audited source JSON")
        if len(self.episode.instruction.instruction_tokens) > args.max_instruction_tokens:
            raise ValueError("R2R instruction exceeds configured token width")
        dataset.episodes = [self.episode]  # one scene per process
        self.env = habitat.Env(config=self.config, dataset=dataset)
        success_measure = self.env.task.measurements.measures["success"]
        self.success_distance_m = float(success_measure._success_distance)
        if not math.isclose(self.success_distance_m, SUCCESS_DISTANCE_METERS):
            raise RuntimeError("Habitat Success measure did not use 3.0 m threshold")
        self.rgb_size = args.rgb_size
        self.max_instruction_tokens = args.max_instruction_tokens
        self.split = args.split
        self.episode_index = args.episode_index
        self.scene_id = str(self.episode.scene_id)
        self.episode_id = str(self.episode.episode_id)
        self.instruction_sha256 = hashlib.sha256(
            self.episode.instruction.instruction_text.encode("utf-8")
        ).hexdigest()
        self.previous_position = None
        self.initial_height = None
        self.path_length = 0.0
        self.collision_count = 0

    def ready(self):
        ready = {
            "ready": True,
            "actions": list(ACTIONS),
            "split": self.split,
            "dataset_path": str(self.dataset_path),
            "dataset_sha256": file_sha256(self.dataset_path),
            "episode_index": self.episode_index,
            "episode_id": self.episode_id,
            "scene_id": self.scene_id,
            "vocab_sha256": self.vocab_sha256,
            "vocab_size": len(self.words),
            "rgb_size": self.rgb_size,
            "max_instruction_tokens": self.max_instruction_tokens,
            "instruction_sha256": self.instruction_sha256,
            "success_distance_m": self.success_distance_m,
            "train_expert_available": self.split == "train",
        }
        if self.official_streamvln_frame:
            ready.update(
                {
                    "official_streamvln_frame": True,
                    "official_frame_contract_sha256": official_frame_contract_sha256(),
                    "official_frame_sensor_contract": OFFICIAL_FRAME_SENSOR_CONTRACT,
                }
            )
        return ready

    def expert_waypoint(self, lookahead_m):
        """Build a local shortest-path target for supervised train examples only."""
        if self.split != "train":
            raise RuntimeError("expert waypoint queries are restricted to the train split")
        if isinstance(lookahead_m, bool) or not isinstance(lookahead_m, (int, float)):
            raise ValueError("expert lookahead must be numeric")
        lookahead_m = float(lookahead_m)
        if not math.isfinite(lookahead_m) or not 0.25 <= lookahead_m <= 2.0:
            raise ValueError("expert lookahead must be within [0.25, 2.0] meters")

        import habitat_sim
        from habitat_sim.utils.common import quat_rotate_vector

        state = self.env.sim.get_agent_state()
        start = self.np.asarray(state.position, dtype=self.np.float64)
        goal = self.np.asarray(self.episode.goals[0].position, dtype=self.np.float64)
        distance = float(self.env.get_metrics()["distance_to_goal"])
        stop = distance <= self.success_distance_m
        if stop:
            forward_m = lateral_m = bearing_rad = 0.0
            target_distance_m = 0.0
        else:
            query = habitat_sim.ShortestPath()
            query.requested_start = start
            query.requested_end = goal
            if not self.env.sim.pathfinder.find_path(query) or len(query.points) < 2:
                raise RuntimeError("Habitat could not produce an expert shortest path")
            points = [self.np.asarray(point, dtype=self.np.float64) for point in query.points]
            target = interpolate_path_point(points, lookahead_m)
            world_delta = target - start
            local_delta = quat_rotate_vector(state.rotation.inverse(), world_delta)
            forward_m, lateral_m = local_waypoint_components(local_delta)
            target_distance_m = float(self.np.linalg.norm(world_delta[[0, 2]]))
            bearing_rad = math.atan2(lateral_m, forward_m)
        values = (forward_m, lateral_m, bearing_rad, target_distance_m, distance)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("expert waypoint contains nonfinite values")
        return {
            "schema_version": 1,
            "source": "train-shortest-path",
            "lookahead_m": lookahead_m,
            "waypoint_local_m": [forward_m, lateral_m],
            "bearing_rad": bearing_rad,
            "target_distance_m": target_distance_m,
            "distance_to_goal_m": distance,
            "stop": bool(stop),
        }

    def _frame(self, observations, collision):
        expected = {"rgb", "instruction"}
        if self.include_depth:
            expected.add("depth")
        if self.official_streamvln_frame:
            expected.update(("gps", "compass"))
        if set(observations) != expected:
            raise ValueError("R2R observations differ from configured sensors")
        if str(self.env.current_episode.episode_id) != self.episode_id:
            raise RuntimeError("R2R Habitat episode changed")
        if str(self.env.current_episode.scene_id) != self.scene_id:
            raise RuntimeError("R2R Habitat scene changed")
        ids, mask = validate_instruction(
            observations, self.episode, len(self.words), self.max_instruction_tokens
        )
        rgb = self.np.ascontiguousarray(observations["rgb"])
        expected_rgb_shape = (
            (OFFICIAL_RGB_HEIGHT, OFFICIAL_RGB_WIDTH, 3)
            if self.official_streamvln_frame
            else (self.rgb_size, self.rgb_size, 3)
        )
        if rgb.dtype != self.np.uint8 or rgb.shape != expected_rgb_shape:
            raise ValueError("R2R RGB dtype/shape differs from configured uint8 HWC")
        policy_rgb = rgb
        if self.official_streamvln_frame:
            rows = self.np.rint(
                self.np.linspace(0, OFFICIAL_RGB_HEIGHT - 1, self.rgb_size)
            ).astype(self.np.int64)
            columns = self.np.rint(
                self.np.linspace(0, OFFICIAL_RGB_WIDTH - 1, self.rgb_size)
            ).astype(self.np.int64)
            policy_rgb = self.np.ascontiguousarray(rgb[rows][:, columns])
        if self.include_depth:
            depth = self.np.asarray(observations["depth"])
            if depth.ndim == 3 and depth.shape[-1] == 1:
                depth = depth[..., 0]
            expected_depth_shape = (
                (OFFICIAL_RGB_HEIGHT, OFFICIAL_RGB_WIDTH)
                if self.official_streamvln_frame
                else (self.rgb_size, self.rgb_size)
            )
            if (
                depth.shape != expected_depth_shape
                or depth.dtype.kind != "f"
                or depth.dtype.itemsize not in (4, 8)
                or not bool(self.np.isfinite(depth).all())
                or float(depth.min()) < 0.0
                or float(depth.max())
                > (1.0 + 1e-4 if self.official_streamvln_frame else self.depth_max_m + 1e-4)
            ):
                raise ValueError("R2R internal depth violates its configured contract")
        policy_obs = make_policy_obs(
            policy_rgb.tobytes(), policy_rgb.shape, ids, mask
        )
        metrics = self.env.get_metrics()
        position = self.np.asarray(self.env.sim.get_agent_state().position)
        if self.previous_position is not None:
            self.path_length += float(self.np.linalg.norm(position - self.previous_position))
        self.previous_position = position.copy()
        self.collision_count += int(collision)
        frame = {
            "policy_obs": policy_obs,
            "episode_id": self.episode_id,
            "scene_id": self.scene_id,
            "instruction_sha256": self.instruction_sha256,
            "geodesic_distance": float(metrics["distance_to_goal"]),
            "success": bool(metrics["success"]),
            "spl": float(metrics["spl"]),
            "path_length": self.path_length,
            "collision": bool(collision),
            "collision_count": self.collision_count,
            "episode_over": bool(self.env.episode_over),
        }
        if self.official_streamvln_frame:
            if self.initial_height is None:
                raise RuntimeError("official StreamVLN initial height is unavailable")
            gps = self.np.asarray(observations["gps"], dtype=self.np.float64).reshape(-1)
            compass = self.np.asarray(
                observations["compass"], dtype=self.np.float64
            ).reshape(-1)
            if (
                gps.shape != (2,)
                or compass.shape != (1,)
                or not bool(self.np.isfinite(gps).all())
                or not bool(self.np.isfinite(compass).all())
            ):
                raise ValueError("official StreamVLN GPS/compass payload is invalid")
            height = float(position[1]) - self.initial_height
            camera_position = self.np.asarray(
                [gps[0], -gps[1], OFFICIAL_CAMERA_POSITION_XYZ[1] + height],
                dtype=self.np.float64,
            )
            yaw = float(compass[0])
            transform = self.np.asarray(
                [
                    [math.cos(yaw), -math.sin(yaw), 0.0, camera_position[0]],
                    [math.sin(yaw), math.cos(yaw), 0.0, camera_position[1]],
                    [0.0, 0.0, 1.0, camera_position[2]],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=self.np.float64,
            )
            axis_align = self.np.asarray(
                [
                    [0.0, 0.0, 1.0, 0.0],
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=self.np.float64,
            )
            focal = OFFICIAL_RGB_WIDTH / (
                2.0 * math.tan(math.radians(OFFICIAL_RGB_HFOV_DEG) / 2.0)
            )
            intrinsics = self.np.asarray(
                [
                    [focal, 0.0, (OFFICIAL_RGB_WIDTH - 1.0) / 2.0, 0.0],
                    [0.0, focal, (OFFICIAL_RGB_HEIGHT - 1.0) / 2.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype="<f8",
            )
            frame["official_frame_material"] = {
                "rgb": rgb.tobytes(order="C"),
                "depth": self.np.ascontiguousarray(depth, dtype="<f4").tobytes(order="C"),
                "pose": self.np.ascontiguousarray(
                    transform @ axis_align, dtype="<f8"
                ).tobytes(order="C"),
                "intrinsics": intrinsics.tobytes(order="C"),
            }
        return frame

    def official_frame_payload(self, frame, service_step, episode_generation):
        if not self.official_streamvln_frame:
            raise RuntimeError("official StreamVLN frame channel is disabled")
        material = frame.get("official_frame_material")
        if not isinstance(material, dict) or set(material) != {
            "rgb", "depth", "pose", "intrinsics"
        }:
            raise RuntimeError("official StreamVLN frame material is unavailable")
        return {
            "schema_version": OFFICIAL_FRAME_SCHEMA_VERSION,
            "source": OFFICIAL_FRAME_SOURCE,
            "state_unchanged": True,
            "service_step": service_step,
            "episode_generation": episode_generation,
            "episode_id": frame["episode_id"],
            "scene_id": frame["scene_id"],
            "instruction_sha256": frame["instruction_sha256"],
            "sensor_contract": OFFICIAL_FRAME_SENSOR_CONTRACT,
            "contract_sha256": official_frame_contract_sha256(),
            "rgb": encode_official_blob(material["rgb"], encoding=RGB_ENCODING),
            "depth": encode_official_blob(material["depth"], encoding=DEPTH_ENCODING),
            "pose": encode_official_blob(material["pose"], encoding=MATRIX_ENCODING),
            "intrinsics": encode_official_blob(
                material["intrinsics"], encoding=MATRIX_ENCODING
            ),
        }

    def reset(self):
        observations = self.env.reset()
        self.initial_height = float(self.env.sim.get_agent_state().position[1])
        self.previous_position = None
        self.path_length = 0.0
        self.collision_count = 0
        return self._frame(observations, False)

    def step(self, name):
        observations = self.env.step(name)
        return self._frame(observations, self.env.sim.previous_step_collided)

    def close(self):
        self.env.close()


def serve_requests(session, request_lines, protocol):
    for line in request_lines:
        try:
            request = json.loads(line)
            if request.get("cmd") == "reset":
                response = session.reset()
            elif request.get("cmd") == "step":
                response = session.step(request.get("action"))
            elif request.get("cmd") == "expert_waypoint":
                response = session.expert_waypoint(request.get("lookahead_m"))
            elif request.get("cmd") == "observe_official_frame":
                if set(request) != {
                    "cmd", "expected_service_step", "episode_generation"
                }:
                    raise ValueError("official frame request schema mismatch")
                response = session.observe_official_frame(
                    expected_service_step=request["expected_service_step"],
                    episode_generation=request["episode_generation"],
                )
            elif request.get("cmd") == "close":
                protocol.write('{"closed":true}\n')
                break
            else:
                raise ValueError("unknown R2R command")
            protocol.write(json.dumps(response, allow_nan=False, separators=(",", ":")) + "\n")
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            protocol.write(json.dumps({"error": str(exc)}) + "\n")
            break


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--scenes-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val_seen", "val_unseen"), required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--rgb-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--max-instruction-tokens", type=int, default=200)
    parser.add_argument("--official-streamvln-frame", action="store_true")
    args = parser.parse_args(argv)
    if args.rgb_size < 1 or args.max_steps < 1 or args.max_instruction_tokens < 1:
        parser.error("rgb-size, max-steps and max-instruction-tokens must be positive")
    if args.dataset_path.name != f"{args.split}.json.gz":
        parser.error("dataset filename must match split")
    return args


def main(argv=None):
    args = parse_args(argv)
    # Habitat/C++ may log to fd 1. Reserve a duplicated original fd for JSONL.
    sys.stdout.flush()
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    backend = HabitatR2RBackend(args)
    session = R2RSession(backend, max_steps=args.max_steps)
    protocol.write(json.dumps(backend.ready(), allow_nan=False) + "\n")
    try:
        serve_requests(session, sys.stdin, protocol)
    finally:
        session.close()
        protocol.close()


if __name__ == "__main__":
    main()
