"""Private CPU-EGL RGB-D PointNav service for an external PointGoal policy.

The service returns only a two-float PointGoal observation, scalar reward,
termination flags, and scalar metrics. RGB/depth and privileged geometry stay
in its offline trace/video. It is meant to run in the isolated Habitat 3.9
environment, driven over stdio by a separate RLinf policy process.
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

from habitat_rgbd_pointnav_acceptance import (
    DATASET_RELATIVE_PATH,
    DEPTH_MAX_METERS,
    HEIGHT,
    WIDTH,
    capture_frame,
    finish_video,
    make_task_config,
    scalar_metrics,
    sha256_file,
    start_video,
    write_video_frame,
)
from habitat_rgb_offscreen_acceptance import encode_png
from pointnav_service import ACTION_NAMES, PointNavSession


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class VisualBackend:
    """One official episode with visual capture hidden from the policy socket."""

    def __init__(self, asset_root, episode_index, max_steps, stage_dir, fps):
        import habitat
        import numpy as np
        from habitat.config import read_write
        from habitat.config.default_structured_configs import CollisionsMeasurementConfig

        self.np = np
        self.asset_root = Path(asset_root).resolve(strict=True)
        self.data_path = (self.asset_root / DATASET_RELATIVE_PATH).resolve(strict=True)
        self.scenes_dir = self.asset_root / "scene_datasets"
        args = argparse.Namespace(max_steps=max_steps, gpu_device_id=0)
        self.config = make_task_config(
            habitat, read_write, CollisionsMeasurementConfig, args,
            self.data_path, self.scenes_dir,
        )
        dataset = habitat.make_dataset(
            id_dataset=self.config.habitat.dataset.type,
            config=self.config.habitat.dataset,
        )
        if not 0 <= episode_index < len(dataset.episodes):
            raise IndexError("episode index outside official val dataset")
        dataset.episodes = [dataset.episodes[episode_index]]
        self.episode = dataset.episodes[0]
        self.scene = (
            self.scenes_dir / "habitat-test-scenes" / Path(self.episode.scene_id).name
        ).resolve(strict=True)
        self.env = habitat.Env(config=self.config, dataset=dataset)
        specs = self.env.sim.sim_config.agents[0].sensor_specifications
        if not self.env.sim.sim_config.sim_cfg.create_renderer or len(specs) != 2:
            raise RuntimeError("RGB+depth renderer/sensor specifications absent")
        self.stage_dir = stage_dir
        self.video_path = stage_dir / "rgb_depth.mp4"
        self.video = start_video(self.video_path, fps)
        self.depth_frames = []
        self.frame_stats = None
        self.metrics = None
        self.pose = None
        self.last_frame = None
        self.previous_position = None
        self.path_length = 0.0
        self.collision_count = 0

    def _frame(self, observations, collision):
        pointgoal = self.np.asarray(
            observations["pointgoal_with_gps_compass"], dtype=self.np.float32
        )
        if pointgoal.shape != (2,) or not self.np.isfinite(pointgoal).all():
            raise RuntimeError("PointGoal must be two finite values")
        metrics = scalar_metrics(self.env.get_metrics())
        state = self.env.sim.get_agent_state()
        position = self.np.asarray(state.position, dtype=self.np.float64)
        goal = self.np.asarray(self.episode.goals[0].position, dtype=self.np.float64)
        if self.previous_position is not None:
            self.path_length += float(self.np.linalg.norm(position - self.previous_position))
        self.previous_position = position.copy()
        self.collision_count += int(collision)
        rgb, depth, preview, combined, stats = capture_frame(observations, self.np)
        if not self.depth_frames:
            (self.stage_dir / "rgb_reset.png").write_bytes(encode_png(rgb))
            (self.stage_dir / "depth_reset.png").write_bytes(encode_png(preview))
        write_video_frame(self.video, combined)
        self.depth_frames.append(depth)
        self.frame_stats = stats
        self.metrics = metrics
        self.pose = {
            "position_xyz": [float(x) for x in position],
            "rotation_xyzw": [
                float(state.rotation.imag[0]), float(state.rotation.imag[1]),
                float(state.rotation.imag[2]), float(state.rotation.real),
            ],
            "goal_position_xyz": [float(x) for x in goal],
        }
        self.last_frame = {
            "states": pointgoal.tolist(),
            "geodesic_distance": metrics["distance_to_goal"],
            "euclidean_distance": float(self.np.linalg.norm(position - goal)),
            "success": bool(metrics["success"]),
            "spl": metrics["spl"],
            "collision": bool(collision),
            "collision_count": self.collision_count,
            "path_length": self.path_length,
            "episode_over": bool(self.env.episode_over),
            "episode_id": str(self.episode.episode_id),
            "scene_id": str(self.episode.scene_id),
        }
        return self.last_frame

    def reset(self):
        if self.depth_frames:
            raise RuntimeError("one reset only for this fixed official episode")
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


class VisualService:
    def __init__(self, args):
        if args.output_dir.exists():
            raise FileExistsError("output directory already exists")
        args.output_dir.parent.mkdir(parents=True, exist_ok=True)
        self.stage = tempfile.TemporaryDirectory(
            prefix=".checkpoint-rgbd-", dir=str(args.output_dir.parent)
        )
        self.stage_dir = Path(self.stage.name)
        self.args = args
        self.backend = VisualBackend(
            args.asset_root, args.episode_index, args.max_steps,
            self.stage_dir, args.fps,
        )
        self.session = PointNavSession(self.backend, max_steps=args.max_steps)
        self.trace = []
        self.finished = False

    def _trace_row(self, event, action, response):
        row = {
            "event": event,
            "step": len(self.trace),
            "action": action,
            "pointgoal": response["obs"],
            "response": response,
            "metrics": self.backend.metrics,
            "frame": self.backend.frame_stats,
            "offline_diagnostic": self.backend.pose,
            "episode_over": bool(self.backend.env.episode_over),
        }
        self.trace.append(row)

    def reset(self):
        if self.trace:
            raise RuntimeError("single reset already consumed")
        response = self.session.reset()
        self._trace_row("reset", None, response)
        return response

    def step(self, action):
        if not self.trace or self.session.final_info is not None:
            raise RuntimeError("episode not active")
        response = self.session.step(action)
        self._trace_row("step", ACTION_NAMES[action], response)
        return response

    def finalize(self):
        if not self.trace or self.session.final_info is None:
            raise RuntimeError("episode must reach STOP or timeout before finalize")
        if self.finished:
            raise RuntimeError("already finalized")
        final = self.trace[-1]
        final_response = final["response"]
        final_info = final_response["info"]
        success = bool(final_info["success"])
        stop_called = bool(final_info["stop_called"])
        threshold = float(
            self.backend.config.habitat.task.measurements.success.success_distance
        )
        if success != (stop_called and final_info["geodesic_distance"] < threshold):
            raise RuntimeError("Habitat STOP/success threshold changed")
        start_distance = self.trace[0]["metrics"]["distance_to_goal"]
        path_length = float(final_info["path_length"])
        expected_spl = start_distance / max(start_distance, path_length) if success else 0.0
        if not math.isclose(float(final_info["spl"]), expected_spl, abs_tol=1e-4):
            raise RuntimeError("Habitat SPL differs from path-length computation")
        if len(self.trace) < 3:
            raise RuntimeError("episode has fewer than two actions")
        if len({r["frame"]["rgb_raw_sha256"] for r in self.trace}) < 2:
            raise RuntimeError("RGB frames did not change")
        if len({r["frame"]["depth_raw_sha256"] for r in self.trace}) < 2:
            raise RuntimeError("depth frames did not change")
        self.backend.video, video = None, self.backend.video
        decoded = finish_video(video, self.backend.video_path, len(self.trace))
        self.backend.np.save(
            self.stage_dir / "depth_frames_meters.npy",
            self.backend.np.stack(self.backend.depth_frames),
        )
        with (self.stage_dir / "trace.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for row in self.trace:
                stream.write(canonical_json(row) + "\n")
        files = {}
        for name in (
            "rgb_reset.png", "depth_reset.png", "depth_frames_meters.npy",
            "trace.jsonl", "rgb_depth.mp4",
        ):
            path = self.stage_dir / name
            files[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        result = {
            "schema_version": 1,
            "mode": "checkpoint_driven_rgbd_pointnav",
            "episode_index": self.args.episode_index,
            "episode_id": str(self.backend.episode.episode_id),
            "scene_id": str(self.backend.episode.scene_id),
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
                "frames": len(self.trace),
                "decoded_frames": decoded,
                "full_decode_validated": True,
            },
            "outcome": {
                "steps": len(self.trace) - 1,
                "success": success,
                "spl": float(final_info["spl"]),
                "stop_called": stop_called,
                "termination_reason": final_info["termination_reason"],
                "final_geodesic_distance_meters": final_info["geodesic_distance"],
                "final_euclidean_distance_meters": final_info["distance_to_goal"],
                "path_length_meters": path_length,
                "collisions": final_info["collision_count"],
                "reward_sum": sum(
                    float(row["response"]["reward"])
                    for row in self.trace if row["event"] == "step"
                ),
            },
            "files": files,
        }
        (self.stage_dir / "manifest.json").write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if self.args.output_dir.exists():
            raise FileExistsError("output directory appeared during run")
        os.replace(str(self.stage_dir), str(self.args.output_dir))
        self.finished = True
        return result

    def close(self):
        self.backend.close()
        self.stage.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--fps", type=int, default=6)
    args = parser.parse_args()
    if not 0 <= args.episode_index or not 2 <= args.max_steps <= 64 or args.fps < 1:
        parser.error("episode >=0, max steps 2..64, and FPS >=1 required")
    # Habitat C++ logging may reach fd 1. Preserve a separate JSON-only pipe.
    sys.stdout.flush()
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    import habitat_sim
    from importlib.metadata import version
    if habitat_sim.__version__ != "0.2.4" or version("habitat-lab") != "0.2.4":
        raise RuntimeError("requires isolated Habitat-Sim/Lab 0.2.4")
    service = VisualService(args)
    protocol.write(canonical_json({"ready": True, "actions": ACTION_NAMES}) + "\n")
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
                if command in ("finalize", "close"):
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
