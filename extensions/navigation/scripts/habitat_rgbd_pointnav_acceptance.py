"""Accept one rendered Habitat-Lab 0.2.4 PointNav episode with RGB and depth.

The default policy is a fixed six-action diagnostic sequence, not a learned
navigation policy. --check-config uses only the Python standard library and
never imports Habitat, opens a renderer, or touches a GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from habitat_rgb_offscreen_acceptance import encode_png, sha256_file


DATASET_RELATIVE_PATH = Path(
    "datasets/pointnav/habitat-test-scenes/v1/val/val.json.gz"
)
TEST_SCENE_RELATIVE_PATH = Path(
    "scene_datasets/habitat-test-scenes/skokloster-castle.glb"
)
SCRIPTED_ACTIONS = (
    "move_forward",
    "turn_left",
    "move_forward",
    "turn_right",
    "move_forward",
    "stop",
)
HEIGHT = WIDTH = 256
DEPTH_MAX_METERS = 10.0


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def static_check(args: argparse.Namespace) -> dict:
    asset_root = args.asset_root.resolve() if args.asset_root else None
    return {
        "schema_version": 1,
        "mode": "static_check",
        "renderer_created": False,
        "habitat_imported": False,
        "visual_render_validated": False,
        "pointnav_validated": False,
        "rl_training_validated": False,
        "asset_root": str(asset_root) if asset_root else None,
        "dataset_relative_path": DATASET_RELATIVE_PATH.as_posix(),
        "dataset_exists": (
            (asset_root / DATASET_RELATIVE_PATH).is_file() if asset_root else None
        ),
        "episode_zero_scene_relative_path": TEST_SCENE_RELATIVE_PATH.as_posix(),
        "episode_zero_scene_exists": (
            (asset_root / TEST_SCENE_RELATIVE_PATH).is_file() if asset_root else None
        ),
        "episode_index": args.episode_index,
        "policy": args.policy,
        "scripted_actions": list(SCRIPTED_ACTIONS) if args.policy == "scripted" else None,
        "max_steps": args.max_steps,
        "rgb_resolution_hw": [HEIGHT, WIDTH],
        "depth_resolution_hw": [HEIGHT, WIDTH],
        "depth_units": "meters",
        "depth_max_meters": DEPTH_MAX_METERS,
        "gpu_device_id": args.gpu_device_id,
        "fps": args.fps,
        "output_dir": str(args.output_dir.resolve()) if args.output_dir else None,
    }


def pose(sim: object) -> dict:
    state = sim.get_agent_state()
    quat = state.rotation
    return {
        "position_xyz": [float(x) for x in state.position],
        "rotation_xyzw": [
            float(quat.imag[0]),
            float(quat.imag[1]),
            float(quat.imag[2]),
            float(quat.real),
        ],
    }


def scalar_metrics(metrics: dict) -> dict:
    collision = metrics["collisions"]
    return {
        "distance_to_goal": float(metrics["distance_to_goal"]),
        "success": float(metrics["success"]),
        "spl": float(metrics["spl"]),
        "collisions": {
            "count": int(collision["count"]),
            "is_collision": bool(collision["is_collision"]),
        },
    }


def pointgoal(observations: dict) -> list[float]:
    values = [float(x) for x in observations["pointgoal_with_gps_compass"]]
    if len(values) != 2 or not all(math.isfinite(x) for x in values):
        raise RuntimeError("Unexpected PointGoal observation")
    return values


def validate_frame(observations: dict, np):
    if "rgb" not in observations or "depth" not in observations:
        raise RuntimeError("Habitat-Lab did not return both RGB and depth")
    rgb = np.ascontiguousarray(observations["rgb"])
    depth_hwc = np.asarray(observations["depth"])
    if rgb.shape != (HEIGHT, WIDTH, 3) or rgb.dtype != np.uint8:
        raise RuntimeError(f"Unexpected RGB: shape={rgb.shape}, dtype={rgb.dtype}")
    if depth_hwc.shape != (HEIGHT, WIDTH, 1) or depth_hwc.dtype != np.float32:
        raise RuntimeError(
            f"Unexpected depth: shape={depth_hwc.shape}, dtype={depth_hwc.dtype}"
        )
    depth = np.ascontiguousarray(depth_hwc[:, :, 0])
    if not np.isfinite(depth).all():
        raise RuntimeError("Depth contains NaN or infinity")
    if float(depth.min()) < 0 or float(depth.max()) > DEPTH_MAX_METERS + 1e-4:
        raise RuntimeError("Depth falls outside configured metric range")
    return rgb, depth


def depth_preview(depth, np):
    gray = np.rint(np.clip(depth / DEPTH_MAX_METERS, 0.0, 1.0) * 255.0).astype(
        np.uint8
    )
    return np.ascontiguousarray(np.repeat(gray[:, :, None], 3, axis=2))


def capture_frame(observations: dict, np):
    rgb, depth = validate_frame(observations, np)
    preview = depth_preview(depth, np)
    combined = np.ascontiguousarray(np.concatenate((rgb, preview), axis=1))
    stats = {
        "rgb_shape_hwc": list(rgb.shape),
        "rgb_dtype": str(rgb.dtype),
        "rgb_min": int(rgb.min()),
        "rgb_max": int(rgb.max()),
        "rgb_raw_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
        "depth_shape_hw": list(depth.shape),
        "depth_dtype": str(depth.dtype),
        "depth_min_meters": float(depth.min()),
        "depth_max_meters": float(depth.max()),
        "depth_raw_sha256": hashlib.sha256(depth.tobytes()).hexdigest(),
    }
    return rgb, depth, preview, combined, stats


def start_video(path: Path, fps: int) -> subprocess.Popen:
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{2 * WIDTH}x{HEIGHT}", "-framerate", str(fps),
        "-i", "pipe:0", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", "-f", "mp4", str(path),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def write_video_frame(process: subprocess.Popen, frame) -> None:
    if process.stdin is None:
        raise RuntimeError("ffmpeg stdin is unavailable")
    process.stdin.write(frame.tobytes())


def finish_video(process: subprocess.Popen, path: Path, expected_frames: int) -> int:
    if process.stdin is None or process.stderr is None:
        raise RuntimeError("ffmpeg pipes are unavailable")
    process.stdin.close()
    error = process.stderr.read().decode("utf-8", errors="replace")
    if process.wait() != 0 or not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg encode failed: {error}")
    decoded = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
         "-f", "null", "-"],
        capture_output=True, text=True, check=False, timeout=120,
    )
    if decoded.returncode != 0:
        raise RuntimeError(f"ffmpeg full decode failed: {decoded.stderr}")
    probed = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=False, timeout=120,
    )
    if probed.returncode != 0 or probed.stdout.strip() != str(expected_frames):
        raise RuntimeError(
            "Decoded video frame count mismatch: "
            f"expected {expected_frames}, observed {probed.stdout.strip()}; "
            f"ffprobe error: {probed.stderr}"
        )
    return expected_frames


def make_task_config(habitat, read_write, CollisionsMeasurementConfig, args, data_path, scenes_dir):
    """Construct the official RGB-D PointNav task without creating a simulator."""
    config = habitat.get_config("benchmark/nav/pointnav/pointnav_habitat_test.yaml")
    with read_write(config):
        config.habitat.dataset.split = "val"
        config.habitat.dataset.data_path = str(data_path)
        config.habitat.dataset.scenes_dir = str(scenes_dir)
        config.habitat.environment.max_episode_steps = args.max_steps
        config.habitat.environment.iterator_options.shuffle = False
        config.habitat.environment.iterator_options.cycle = False
        config.habitat.simulator.create_renderer = True
        config.habitat.simulator.habitat_sim_v0.gpu_device_id = args.gpu_device_id
        # Habitat-Lab's gpu_gpu controls GPU->GPU sensor transfer. Keep CPU NumPy
        # frames available for hashes, PNG, metric-depth .npy, and FFmpeg.
        config.habitat.simulator.habitat_sim_v0.gpu_gpu = False
        config.habitat.task.measurements.collisions = CollisionsMeasurementConfig()
        sensors = config.habitat.simulator.agents.main_agent.sim_sensors
        if set(sensors) != {"rgb_sensor", "depth_sensor"}:
            raise RuntimeError(f"Unexpected PointNav visual sensors: {list(sensors)}")
        for sensor in sensors.values():
            sensor.height = HEIGHT
            sensor.width = WIDTH
        sensors.depth_sensor.normalize_depth = False
        sensors.depth_sensor.min_depth = 0.0
        sensors.depth_sensor.max_depth = DEPTH_MAX_METERS
    return config


def check_api(args: argparse.Namespace) -> dict:
    """Exercise Habitat-Lab's config schema without building a Simulator."""
    import habitat
    import habitat_sim
    from habitat.config import read_write
    from habitat.config.default_structured_configs import CollisionsMeasurementConfig

    asset_root = args.asset_root.resolve() if args.asset_root else Path("/unprovided-assets")
    config = make_task_config(
        habitat, read_write, CollisionsMeasurementConfig, args,
        asset_root / DATASET_RELATIVE_PATH, asset_root / "scene_datasets",
    )
    sensors = config.habitat.simulator.agents.main_agent.sim_sensors
    return {
        "schema_version": 1,
        "mode": "api_config_only",
        "renderer_created": False,
        "habitat_imported": True,
        "visual_render_validated": False,
        "pointnav_validated": False,
        "rl_training_validated": False,
        "habitat_sim_version": habitat_sim.__version__,
        "habitat_lab_version": importlib.metadata.version("habitat-lab"),
        "sensor_keys": sorted(sensors),
        "rgb_resolution_hw": [sensors.rgb_sensor.height, sensors.rgb_sensor.width],
        "depth_resolution_hw": [sensors.depth_sensor.height, sensors.depth_sensor.width],
        "depth_normalized": bool(sensors.depth_sensor.normalize_depth),
        "depth_max_meters": float(sensors.depth_sensor.max_depth),
        "gpu_device_id": int(config.habitat.simulator.habitat_sim_v0.gpu_device_id),
        "gpu_gpu": bool(config.habitat.simulator.habitat_sim_v0.gpu_gpu),
        "collisions_measurement_type": str(config.habitat.task.measurements.collisions.type),
        "dataset_path": str(config.habitat.dataset.data_path),
    }


def run(args: argparse.Namespace) -> dict:
    import habitat
    import habitat_sim
    import numpy as np
    from habitat.config import read_write
    from habitat.config.default_structured_configs import CollisionsMeasurementConfig
    from habitat.sims.habitat_simulator.actions import HabitatSimActions
    from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

    if habitat_sim.__version__ != "0.2.4":
        raise RuntimeError(f"Expected Habitat-Sim 0.2.4, found {habitat_sim.__version__}")
    lab_version = importlib.metadata.version("habitat-lab")
    if lab_version != "0.2.4":
        raise RuntimeError(f"Expected Habitat-Lab 0.2.4, found {lab_version}")
    asset_root = args.asset_root.resolve(strict=True)
    data_path = (asset_root / DATASET_RELATIVE_PATH).resolve(strict=True)
    scenes_dir = asset_root / "scene_datasets"
    if not data_path.is_file() or not scenes_dir.is_dir():
        raise FileNotFoundError("Official PointNav dataset or scenes directory missing")

    config = make_task_config(
        habitat, read_write, CollisionsMeasurementConfig, args, data_path, scenes_dir
    )

    dataset = habitat.make_dataset(
        id_dataset=config.habitat.dataset.type, config=config.habitat.dataset
    )
    if args.episode_index >= len(dataset.episodes):
        raise IndexError(
            f"Episode index {args.episode_index} outside {len(dataset.episodes)} episodes"
        )
    dataset.episodes = [dataset.episodes[args.episode_index]]
    selected = dataset.episodes[0]
    scene = scenes_dir / "habitat-test-scenes" / Path(selected.scene_id).name
    if not scene.is_file():
        raise FileNotFoundError(scene)

    stage_parent = args.output_dir.resolve().parent
    stage_parent.mkdir(parents=True, exist_ok=True)
    init_start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=".habitat-rgbd-", dir=stage_parent) as stage:
        stage_dir = Path(stage)
        video_path = stage_dir / "rgb_depth.mp4"
        process = None
        try:
            with habitat.Env(config=config, dataset=dataset) as env:
                env_init_seconds = time.perf_counter() - init_start
                specs = env.sim.sim_config.agents[0].sensor_specifications
                if not env.sim.sim_config.sim_cfg.create_renderer or len(specs) != 2:
                    raise RuntimeError("RGB+depth renderer or sensor specs were not created")
                if env.sim.sim_config.sim_cfg.gpu_device_id != args.gpu_device_id:
                    raise RuntimeError("Habitat-Sim GPU ID differs from requested logical ID")
                process = start_video(video_path, args.fps)
                rollout_start = time.perf_counter()
                reset_start = time.perf_counter()
                observations = env.reset()
                reset_seconds = time.perf_counter() - reset_start
                episode = env.current_episode
                if str(episode.episode_id) != str(selected.episode_id):
                    raise RuntimeError("Wrong PointNav episode selected")
                initial_metrics = scalar_metrics(env.get_metrics())
                initial_pose = pose(env.sim)
                rgb, depth, preview, combined, stats = capture_frame(observations, np)
                (stage_dir / "rgb_reset.png").write_bytes(encode_png(rgb))
                (stage_dir / "depth_reset.png").write_bytes(encode_png(preview))
                write_video_frame(process, combined)
                depth_frames = [depth]
                trace = [{
                    "step": 0, "action": "reset", "pose": initial_pose,
                    "pointgoal": pointgoal(observations), "metrics": initial_metrics,
                    "frame": stats, "episode_over": False,
                }]
                previous_position = initial_pose["position_xyz"]
                path_length = 0.0
                step_seconds = 0.0
                follower = (
                    ShortestPathFollower(
                        env.sim,
                        goal_radius=0.9 * float(
                            config.habitat.task.measurements.success.success_distance
                        ),
                        return_one_hot=False,
                        stop_on_error=False,
                    ) if args.policy == "oracle" else None
                )
                for index in range(args.max_steps):
                    if env.episode_over:
                        break
                    if follower is not None:
                        action_id = follower.get_next_action(episode.goals[0].position)
                        action = {
                            HabitatSimActions.stop: "stop",
                            HabitatSimActions.move_forward: "move_forward",
                            HabitatSimActions.turn_left: "turn_left",
                            HabitatSimActions.turn_right: "turn_right",
                        }.get(action_id)
                        if action is None:
                            raise RuntimeError(f"Oracle returned unsupported action {action_id!r}")
                    else:
                        action = SCRIPTED_ACTIONS[index]
                    step_start = time.perf_counter()
                    observations = env.step(action)
                    step_seconds += time.perf_counter() - step_start
                    metrics = scalar_metrics(env.get_metrics())
                    agent_pose = pose(env.sim)
                    path_length += math.dist(previous_position, agent_pose["position_xyz"])
                    previous_position = agent_pose["position_xyz"]
                    rgb, depth, preview, combined, stats = capture_frame(observations, np)
                    write_video_frame(process, combined)
                    depth_frames.append(depth)
                    trace.append({
                        "step": index + 1, "action": action, "pose": agent_pose,
                        "pointgoal": pointgoal(observations), "metrics": metrics,
                        "frame": stats, "episode_over": bool(env.episode_over),
                    })
                    if action == "stop":
                        break
                rollout_wall_seconds = time.perf_counter() - rollout_start
                if not env.episode_over:
                    raise RuntimeError("PointNav episode did not terminate")
                if len(trace) < 3:
                    raise RuntimeError("No stepped PointNav sequence was observed")
                if not any(
                    row["frame"]["rgb_max"] > row["frame"]["rgb_min"]
                    for row in trace
                ) or not any(
                    row["frame"]["depth_max_meters"]
                    > row["frame"]["depth_min_meters"]
                    for row in trace
                ):
                    raise RuntimeError("RGB or depth had no spatial variation")
                if len({row["frame"]["rgb_raw_sha256"] for row in trace}) < 2:
                    raise RuntimeError("RGB did not change across PointNav steps")
                if len({row["frame"]["depth_raw_sha256"] for row in trace}) < 2:
                    raise RuntimeError("Depth did not change across PointNav steps")
                final_metrics = trace[-1]["metrics"]
                stop_called = trace[-1]["action"] == "stop"
                success = bool(final_metrics["success"])
                threshold = float(config.habitat.task.measurements.success.success_distance)
                if success != (stop_called and final_metrics["distance_to_goal"] < threshold):
                    raise RuntimeError("Habitat PointNav STOP/success semantics changed")
                start_distance = initial_metrics["distance_to_goal"]
                expected_spl = (
                    start_distance / max(start_distance, path_length) if success else 0.0
                )
                if not math.isclose(final_metrics["spl"], expected_spl, abs_tol=1e-4):
                    raise RuntimeError("Habitat PointNav SPL is inconsistent with path length")
                np.save(stage_dir / "depth_frames_meters.npy", np.stack(depth_frames))
                with (stage_dir / "trace.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
                    for row in trace:
                        stream.write(canonical_json(row) + "\n")
            decoded_frames = finish_video(process, video_path, len(trace))
            process = None
            end_to_end_rollout_seconds = time.perf_counter() - rollout_start
        finally:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()

        files = {}
        for name in (
            "rgb_reset.png", "depth_reset.png", "depth_frames_meters.npy",
            "trace.jsonl", "rgb_depth.mp4",
        ):
            path = stage_dir / name
            files[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        result = {
            "schema_version": 1,
            "mode": "rgbd_pointnav_episode",
            "renderer_created": True,
            "visual_render_validated": True,
            "pointnav_validated": True,
            "video_full_decode_validated": True,
            "rl_training_validated": False,
            "policy": args.policy,
            "policy_uses_privileged_target_and_pose": args.policy == "oracle",
            "episode_index": args.episode_index,
            "episode_id": str(episode.episode_id),
            "scene_id": str(episode.scene_id),
            "scene_sha256": sha256_file(scene),
            "dataset_sha256": sha256_file(data_path),
            "script_sha256": sha256_file(Path(__file__)),
            "asset_root": str(asset_root),
            "gpu_device_id": args.gpu_device_id,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "versions": {
                "python": platform.python_version(),
                "habitat_lab": lab_version,
                "habitat_sim": habitat_sim.__version__,
                "numpy": np.__version__,
            },
            "sensor": {
                "rgb_shape_hwc": [HEIGHT, WIDTH, 3],
                "depth_shape_hwc": [HEIGHT, WIDTH, 1],
                "depth_units": "meters",
                "depth_clip_range_meters": [0.0, DEPTH_MAX_METERS],
                "depth_preview_scale_meters": DEPTH_MAX_METERS,
            },
            "video": {
                "layout": "RGB left, metric-depth grayscale right",
                "codec": "H.264/yuv420p",
                "resolution_hw": [HEIGHT, WIDTH * 2],
                "fps": args.fps,
                "frames": len(trace),
                "decoded_frames": decoded_frames,
            },
            "outcome": {
                "steps": len(trace) - 1,
                "stop_called": stop_called,
                "success": success,
                "spl": final_metrics["spl"],
                "final_geodesic_distance_meters": final_metrics["distance_to_goal"],
                "path_length_meters": path_length,
                "collisions": final_metrics["collisions"]["count"],
                "termination_reason": (
                    "stop_success" if success else "stop_failure" if stop_called else "timeout"
                ),
            },
            "timing": {
                "environment_init_seconds": env_init_seconds,
                "reset_seconds": reset_seconds,
                "step_calls_seconds": step_seconds,
                "rollout_and_frame_write_seconds": rollout_wall_seconds,
                "rollout_through_video_finalize_seconds": end_to_end_rollout_seconds,
                "step_calls_per_second": (len(trace) - 1) / step_seconds,
                "rollout_steps_per_second": (len(trace) - 1) / rollout_wall_seconds,
                "rollout_through_video_finalize_steps_per_second": (
                    (len(trace) - 1) / end_to_end_rollout_seconds
                ),
            },
            "files": files,
        }
        (stage_dir / "manifest.json").write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if args.output_dir.exists():
            raise FileExistsError(args.output_dir)
        os.replace(stage_dir, args.output_dir.resolve())
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--gpu-device-id", type=int, default=0)
    parser.add_argument("--policy", choices=("scripted", "oracle"), default="scripted")
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument(
        "--check-config", action="store_true",
        help="Print a static plan without importing Habitat, rendering, or requiring assets",
    )
    parser.add_argument(
        "--check-api", action="store_true",
        help="Validate Habitat-Lab 0.2.4 task configuration without creating a simulator",
    )
    args = parser.parse_args()
    if args.episode_index < 0 or args.gpu_device_id < 0 or args.fps < 1:
        parser.error("episode index and GPU ID must be nonnegative; FPS must be positive")
    if args.max_steps < (len(SCRIPTED_ACTIONS) if args.policy == "scripted" else 2):
        parser.error("--max-steps is too small for the selected diagnostic policy")
    if args.check_config and args.check_api:
        parser.error("choose only one of --check-config and --check-api")
    if args.check_config:
        print(canonical_json(static_check(args)))
        return 0
    if args.check_api:
        print(canonical_json(check_api(args)))
        return 0
    if args.asset_root is None or args.output_dir is None:
        parser.error("--asset-root and --output-dir are required for rendering")
    if args.output_dir.exists():
        parser.error("--output-dir must not already exist; use a new run directory")
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        parser.error("ffmpeg and ffprobe are required to verify the acceptance video")
    result = run(args)
    print(canonical_json({
        "status": "ok",
        "mode": result["mode"],
        "output_dir": str(args.output_dir.resolve()),
        "manifest": str(args.output_dir.resolve() / "manifest.json"),
        "steps": result["outcome"]["steps"],
        "success": result["outcome"]["success"],
        "video_full_decode_validated": result["video_full_decode_validated"],
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
