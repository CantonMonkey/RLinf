"""Private CPU-EGL RGB-D PointNav train service with bounded video capture.

The PPO policy receives only the two-float PointGoal, reward, termination, and
scalar metrics through the same stdio protocol as pointnav_service.py. RGB,
metric depth, and privileged pose are written only to offline artifacts. A
single Habitat process cycles a selected same-scene subset of official train
episodes; each reset is captured, including resets after STOP or time limits.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import sys
import tempfile
import traceback
from pathlib import Path

from habitat_rgb_offscreen_acceptance import encode_png
from habitat_rgbd_pointnav_acceptance import (
    DEPTH_MAX_METERS,
    HEIGHT,
    WIDTH,
    capture_frame,
    finish_video,
    make_task_config,
    scalar_metrics,
    sha256_file,
    start_video,
    validate_frame,
    write_video_frame,
)
from pointnav_service import (
    ACTION_NAMES,
    PointNavSession,
    pointnav_dataset_path,
    select_episodes,
)


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class VisualTrainBackend:
    """One rendered Habitat train environment with a bounded recorded prefix."""

    def __init__(self, args, stage_dir):
        import habitat
        import numpy as np
        from habitat.config import read_write
        from habitat.config.default_structured_configs import CollisionsMeasurementConfig

        self.np = np
        self.asset_root = args.asset_root.resolve(strict=True)
        self.data_path = pointnav_dataset_path(self.asset_root, "train").resolve(strict=True)
        self.scenes_dir = self.asset_root / "scene_datasets"
        config_args = argparse.Namespace(max_steps=args.max_steps, gpu_device_id=0)
        self.config = make_task_config(
            habitat, read_write, CollisionsMeasurementConfig, config_args,
            self.data_path, self.scenes_dir,
        )
        with read_write(self.config):
            self.config.habitat.dataset.split = "train"
            self.config.habitat.environment.iterator_options.cycle = True
        dataset = habitat.make_dataset(
            id_dataset=self.config.habitat.dataset.type,
            config=self.config.habitat.dataset,
        )
        selected = select_episodes(
            dataset.episodes,
            episode_indices=args.episode_indices,
            split="train",
        )
        dataset.episodes = selected
        self.episode_ids = [str(episode.episode_id) for episode in selected]
        self.scene_id = str(selected[0].scene_id)
        self.scene = (
            self.scenes_dir / "habitat-test-scenes" / Path(self.scene_id).name
        ).resolve(strict=True)
        self.env = habitat.Env(config=self.config, dataset=dataset)
        specs = self.env.sim.sim_config.agents[0].sensor_specifications
        if not self.env.sim.sim_config.sim_cfg.create_renderer or len(specs) != 2:
            self.env.close()
            raise RuntimeError("RGB+depth renderer/sensor specifications absent")
        self.stage_dir = stage_dir
        self.video_path = stage_dir / "rgb_depth.mp4"
        self.video = start_video(self.video_path, args.fps)
        self.max_recorded_frames = args.max_recorded_frames
        self.depth_frames = []
        self.recorded_frames = 0
        self.last_frame_stats = None
        self.last_recorded_frame_index = None
        self.last_metrics = None
        self.last_pose = None
        self.previous_position = None
        self.path_length = 0.0
        self.collision_count = 0
        self.episode_ordinal = -1

    def _frame(self, observations, collision):
        episode = self.env.current_episode
        pointgoal = self.np.asarray(
            observations["pointgoal_with_gps_compass"], dtype=self.np.float32
        )
        if pointgoal.shape != (2,) or not self.np.isfinite(pointgoal).all():
            raise RuntimeError("PointGoal must be two finite values")
        metrics = scalar_metrics(self.env.get_metrics())
        state = self.env.sim.get_agent_state()
        position = self.np.asarray(state.position, dtype=self.np.float64)
        goal = self.np.asarray(episode.goals[0].position, dtype=self.np.float64)
        if self.previous_position is not None:
            self.path_length += float(self.np.linalg.norm(position - self.previous_position))
        self.previous_position = position.copy()
        self.collision_count += int(collision)
        expected_episode_id = self.episode_ids[self.episode_ordinal % len(self.episode_ids)]
        if str(episode.episode_id) != expected_episode_id:
            raise RuntimeError(
                f"Habitat train episode order changed: got {episode.episode_id}, "
                f"expected {expected_episode_id}"
            )
        if str(episode.scene_id) != self.scene_id:
            raise RuntimeError("Habitat switched scenes inside one visual train process")
        self.last_recorded_frame_index = None
        self.last_frame_stats = None
        if self.recorded_frames < self.max_recorded_frames:
            rgb, depth, preview, combined, stats = capture_frame(observations, self.np)
            if self.recorded_frames == 0:
                (self.stage_dir / "rgb_reset.png").write_bytes(encode_png(rgb))
                (self.stage_dir / "depth_reset.png").write_bytes(encode_png(preview))
            write_video_frame(self.video, combined)
            self.depth_frames.append(depth)
            self.last_recorded_frame_index = self.recorded_frames
            self.last_frame_stats = stats
            self.recorded_frames += 1
        else:
            validate_frame(observations, self.np)
        self.last_metrics = metrics
        self.last_pose = {
            "position_xyz": [float(x) for x in position],
            "rotation_xyzw": [
                float(state.rotation.imag[0]), float(state.rotation.imag[1]),
                float(state.rotation.imag[2]), float(state.rotation.real),
            ],
            "goal_position_xyz": [float(x) for x in goal],
        }
        return {
            "states": pointgoal.tolist(),
            "geodesic_distance": metrics["distance_to_goal"],
            "euclidean_distance": float(self.np.linalg.norm(position - goal)),
            "success": bool(metrics["success"]),
            "spl": metrics["spl"],
            "collision": bool(collision),
            "collision_count": self.collision_count,
            "path_length": self.path_length,
            "episode_over": bool(self.env.episode_over),
            "episode_id": str(episode.episode_id),
            "scene_id": str(episode.scene_id),
        }

    def reset(self):
        self.episode_ordinal += 1
        self.previous_position = None
        self.path_length = 0.0
        self.collision_count = 0
        return self._frame(self.env.reset(), False)

    def step(self, action_name):
        return self._frame(
            self.env.step(action_name), self.env.sim.previous_step_collided
        )

    def close(self):
        self.env.close()
        if self.video is not None and self.video.poll() is None:
            self.video.kill()
            self.video.wait()


class VisualTrainService:
    def __init__(self, args):
        if args.output_dir.exists():
            raise FileExistsError("visual output directory already exists")
        args.output_dir.parent.mkdir(parents=True, exist_ok=True)
        self.stage = tempfile.TemporaryDirectory(
            prefix=".train-rgbd-", dir=str(args.output_dir.parent)
        )
        self.stage_dir = Path(self.stage.name)
        self.args = args
        try:
            self.backend = VisualTrainBackend(args, self.stage_dir)
        except BaseException:
            self.stage.cleanup()
            raise
        self.session = PointNavSession(
            self.backend, max_steps=args.max_steps,
            premature_stop_penalty=args.premature_stop_penalty,
        )
        self.trace = (self.stage_dir / "trace.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        )
        self.events = 0
        self.steps = 0
        self.completed = 0
        self.successes = 0
        self.reward_sum = 0.0
        self.start_distance = None
        self.manifest = None

    def _record(self, event, action, response):
        row = {
            "event": event,
            "event_index": self.events,
            "episode_ordinal": self.backend.episode_ordinal,
            "action": action,
            "pointgoal": response["obs"],
            "response": response,
            "metrics": self.backend.last_metrics,
            "recorded_frame_index": self.backend.last_recorded_frame_index,
            "frame": self.backend.last_frame_stats,
            "offline_diagnostic": self.backend.last_pose,
            "episode_over": bool(self.backend.env.episode_over),
        }
        self.trace.write(canonical_json(row) + "\n")
        self.events += 1

    def reset(self):
        if self.manifest is not None:
            raise RuntimeError("service already finalized")
        if self.session.last_frame is not None and self.session.final_info is None:
            raise RuntimeError("reset before episode completion")
        response = self.session.reset()
        self.start_distance = float(response["info"]["geodesic_distance"])
        self._record("reset", None, response)
        return response

    def step(self, action):
        if self.manifest is not None:
            raise RuntimeError("service already finalized")
        if self.session.last_frame is None or self.session.final_info is not None:
            raise RuntimeError("no active train episode")
        response = self.session.step(action)
        info = response["info"]
        threshold = float(
            self.backend.config.habitat.task.measurements.success.success_distance
        )
        expected_success = action == 0 and float(info["geodesic_distance"]) < threshold
        if bool(info["success"]) != expected_success:
            raise RuntimeError("Habitat STOP/success threshold changed")
        if (response["terminated"] or response["truncated"]) and self.start_distance > 0:
            expected_spl = (
                self.start_distance
                / max(self.start_distance, float(info["path_length"]))
                if expected_success else 0.0
            )
            if not math.isclose(float(info["spl"]), expected_spl, abs_tol=1e-4):
                raise RuntimeError("Habitat SPL differs from path-length computation")
        self._record("step", ACTION_NAMES[action], response)
        self.steps += 1
        self.reward_sum += float(response["reward"])
        if response["terminated"] or response["truncated"]:
            self.completed += 1
            self.successes += int(response["info"]["success"])
        return response

    def finalize(self):
        if self.manifest is not None:
            return self.manifest
        if self.steps == 0:
            raise RuntimeError("cannot finalize before at least one train step")
        self.trace.close()
        video, self.backend.video = self.backend.video, None
        decoded = finish_video(video, self.backend.video_path, self.backend.recorded_frames)
        self.backend.np.save(
            self.stage_dir / "depth_frames_meters.npy",
            self.backend.np.stack(self.backend.depth_frames),
        )
        files = {}
        for name in (
            "rgb_reset.png", "depth_reset.png", "depth_frames_meters.npy",
            "trace.jsonl", "rgb_depth.mp4",
        ):
            path = self.stage_dir / name
            files[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        manifest = {
            "schema_version": 1,
            "mode": "pointnav_train_split_rgbd_ppo_rollout",
            "split": "train",
            "episode_indices": list(self.args.episode_indices),
            "episode_ids": self.backend.episode_ids,
            "scene_id": self.backend.scene_id,
            "scene_sha256": sha256_file(self.backend.scene),
            "dataset_sha256": sha256_file(self.backend.data_path),
            "service_script_sha256": sha256_file(Path(__file__)),
            "pointnav_session_sha256": sha256_file(
                Path(__file__).with_name("pointnav_service.py")
            ),
            "renderer": "private Habitat-Sim CPU EGL / Mesa llvmpipe",
            "renderer_created": True,
            "pointgoal_only_policy_interface": True,
            "privileged_geometry_sent_to_policy": False,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "versions": {
                "python": platform.python_version(),
                "habitat_lab": importlib.metadata.version("habitat-lab"),
                "habitat_sim": __import__("habitat_sim").__version__,
                "numpy": self.backend.np.__version__,
            },
            "sensor": {
                "rgb_shape_hwc": [HEIGHT, WIDTH, 3],
                "depth_shape_hwc": [HEIGHT, WIDTH, 1],
                "depth_units": "meters",
                "depth_preview_scale_meters": DEPTH_MAX_METERS,
            },
            "video": {
                "layout": "RGB left, metric-depth grayscale right",
                "codec": "H.264/yuv420p",
                "resolution_hw": [HEIGHT, 2 * WIDTH],
                "fps": self.args.fps,
                "frames": self.backend.recorded_frames,
                "decoded_frames": decoded,
                "full_decode_validated": True,
                "max_recorded_frames": self.args.max_recorded_frames,
                "sampled_prefix_only": self.events > self.backend.recorded_frames,
            },
            "outcome": {
                "environment_steps": self.steps,
                "resets": self.backend.episode_ordinal + 1,
                "completed_episodes": self.completed,
                "train_successes": self.successes,
                "reward_sum": self.reward_sum,
            },
            "files": files,
        }
        (self.stage_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if self.args.output_dir.exists():
            raise FileExistsError("visual output directory appeared during run")
        os.replace(str(self.stage_dir), str(self.args.output_dir))
        self.manifest = manifest
        return self.manifest

    def close(self):
        self.trace.close()
        self.backend.close()
        self.stage.cleanup()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train",), required=True)
    parser.add_argument("--episode-indices", required=True)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--premature-stop-penalty", type=float, default=0.0)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--max-recorded-frames", type=int, default=256)
    args = parser.parse_args(argv)
    try:
        args.episode_indices = [int(value) for value in args.episode_indices.split(",")]
    except ValueError:
        parser.error("episode indices must be comma-separated integers")
    if (
        not args.episode_indices
        or len(args.episode_indices) != len(set(args.episode_indices))
        or any(index < 0 for index in args.episode_indices)
        or not 2 <= args.max_steps <= 512
        or not math.isfinite(args.premature_stop_penalty)
        or args.premature_stop_penalty < 0
        or not 1 <= args.fps <= 60
        or not 2 <= args.max_recorded_frames <= 512
    ):
        parser.error("invalid episode indices, step limit, penalty, FPS, or video frame limit")
    return args


def main():
    args = parse_args()
    # Habitat C++ messages may reach fd 1; keep an independent JSON-only pipe.
    sys.stdout.flush()
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    import habitat_sim
    if habitat_sim.__version__ != "0.2.4" or importlib.metadata.version("habitat-lab") != "0.2.4":
        raise RuntimeError("requires isolated Habitat-Sim/Lab 0.2.4")
    service = VisualTrainService(args)
    protocol.write(canonical_json({
        "ready": True,
        "actions": ACTION_NAMES,
        "split": "train",
        "dataset_path": str(service.backend.data_path),
        "renderer_created": True,
    }) + "\n")
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                command = request.get("cmd")
                if command == "reset":
                    response = service.reset()
                elif command == "step":
                    action = request.get("action")
                    if isinstance(action, bool) or not isinstance(action, int) or not 0 <= action < 4:
                        raise ValueError("action must be integer 0..3")
                    response = service.step(action)
                elif command == "finalize":
                    response = {"finalized": True, "manifest": service.finalize()}
                elif command == "close":
                    response = {"closed": True}
                else:
                    raise ValueError("unknown command")
                protocol.write(canonical_json(response) + "\n")
                if command == "close":
                    break
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                protocol.write(canonical_json({"error": str(exc)}) + "\n")
                break
    finally:
        service.close()
        protocol.close()


if __name__ == "__main__":
    main()
