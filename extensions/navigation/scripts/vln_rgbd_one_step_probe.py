"""One-episode R2R RGB/depth/instruction Habitat probe on private CPU Mesa.

Read the source dataset and scene assets without modifying them. Output only a
new directory below the private A800 project's runs directory. The goal and
reference path are used by Habitat's task, never exposed as policy observations.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path

from habitat_rgb_offscreen_acceptance import encode_png, sha256_file


PRIVATE_RUNS = Path("/srv/navigation/runs")
RESOLUTION = 64
DEPTH_MAX_METERS = 10.0


def inspect_observation(observation, metrics, instruction, np):
    keys = sorted(observation)
    if keys != ["depth", "instruction", "rgb"]:
        raise RuntimeError(f"Unexpected VLN observation keys: {keys}")
    if observation["instruction"]["text"] != instruction:
        raise RuntimeError("Instruction sensor differs from the selected episode")
    if not all(key in metrics for key in ("distance_to_goal", "success", "spl")):
        raise RuntimeError("VLN navigation metrics are missing")

    rgb = np.asarray(observation["rgb"])
    depth = np.asarray(observation["depth"])
    if rgb.shape != (RESOLUTION, RESOLUTION, 3) or rgb.dtype != np.uint8:
        raise RuntimeError(f"Unexpected RGB frame: {rgb.shape} {rgb.dtype}")
    if depth.shape != (RESOLUTION, RESOLUTION, 1) or not np.isfinite(depth).all():
        raise RuntimeError(f"Unexpected depth frame: {depth.shape} {depth.dtype}")
    if float(depth.min()) < 0 or float(depth.max()) > DEPTH_MAX_METERS + 1e-4:
        raise RuntimeError("Depth values fall outside the configured metric range")

    stats = {
        "observation_keys": keys,
        "rgb_shape": list(rgb.shape),
        "rgb_dtype": str(rgb.dtype),
        "rgb_min": int(rgb.min()),
        "rgb_max": int(rgb.max()),
        "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
        "depth_shape": list(depth.shape),
        "depth_dtype": str(depth.dtype),
        "depth_min_m": float(depth.min()),
        "depth_max_m": float(depth.max()),
        "depth_sha256": hashlib.sha256(depth.tobytes()).hexdigest(),
        "distance_to_goal": float(metrics["distance_to_goal"]),
        "success": float(metrics["success"]),
        "spl": float(metrics["spl"]),
    }
    return stats, rgb, depth


def write_depth_preview(path, depth, np):
    gray = np.rint(np.clip(depth[:, :, 0] / DEPTH_MAX_METERS, 0, 1) * 255).astype(
        np.uint8
    )
    preview = np.ascontiguousarray(np.repeat(gray[:, :, None], 3, axis=2))
    path.write_bytes(encode_png(preview))


def run(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get(
        "HABITAT_SIM_SOFTWARE_EGL"
    ) != "1":
        raise RuntimeError("This probe requires hidden CUDA and private software EGL")
    if os.environ.get("LIBGL_ALWAYS_SOFTWARE") != "1":
        raise RuntimeError("This probe requires Mesa software rendering")

    output = args.output_dir.resolve()
    if not output.is_relative_to(PRIVATE_RUNS.resolve(strict=True)) or output.exists():
        raise ValueError("Output must be a new directory below the private runs root")

    import habitat
    import habitat_sim
    import numpy as np
    from habitat.config import read_write
    from habitat.datasets.vln.r2r_vln_dataset import VLNDatasetV1

    if habitat_sim.__version__ != "0.2.4":
        raise RuntimeError(f"Expected Habitat-Sim 0.2.4, got {habitat_sim.__version__}")

    source = args.dataset_file.resolve(strict=True)
    scene_root = args.scene_root.resolve(strict=True)
    with gzip.open(source, "rt", encoding="utf-8") as handle:
        document = json.load(handle)
    selected = document["episodes"][args.episode_index]
    instruction = selected["instruction"]["instruction_text"]
    subset = {
        "instruction_vocab": document["instruction_vocab"],
        "episodes": [selected],
    }
    dataset = VLNDatasetV1()
    dataset.from_json(json.dumps(subset), scenes_dir=str(scene_root))

    config = habitat.get_config("benchmark/nav/vln_r2r.yaml")
    with read_write(config):
        config.habitat.dataset.split = args.split
        config.habitat.dataset.data_path = str(source)
        config.habitat.dataset.scenes_dir = str(scene_root)
        config.habitat.environment.max_episode_steps = 2
        config.habitat.environment.iterator_options.shuffle = False
        config.habitat.environment.iterator_options.cycle = False
        config.habitat.simulator.create_renderer = True
        config.habitat.simulator.habitat_sim_v0.gpu_device_id = 0
        config.habitat.simulator.habitat_sim_v0.gpu_gpu = False
        sensors = config.habitat.simulator.agents.main_agent.sim_sensors
        if set(sensors) != {"rgb_sensor", "depth_sensor"}:
            raise RuntimeError(f"Unexpected visual sensors: {sorted(sensors)}")
        for sensor in sensors.values():
            sensor.height = RESOLUTION
            sensor.width = RESOLUTION
        sensors.depth_sensor.normalize_depth = False
        sensors.depth_sensor.min_depth = 0.0
        sensors.depth_sensor.max_depth = DEPTH_MAX_METERS

    with habitat.Env(config=config, dataset=dataset) as env:
        if not env.sim.sim_config.sim_cfg.create_renderer:
            raise RuntimeError("Habitat renderer was not created")
        reset_observation = env.reset()
        if str(env.current_episode.episode_id) != str(selected["episode_id"]):
            raise RuntimeError("Habitat reset selected a different episode")
        reset, rgb_reset, depth_reset = inspect_observation(
            reset_observation, env.get_metrics(), instruction, np
        )
        step_observation = env.step("move_forward")
        step, rgb_step, depth_step = inspect_observation(
            step_observation, env.get_metrics(), instruction, np
        )
        episode_over = bool(env.episode_over)

    result = {
        "status": "ok",
        "split": args.split,
        "dataset_file": str(source),
        "dataset_sha256": sha256_file(source),
        "episode_index": args.episode_index,
        "episode_id": str(selected["episode_id"]),
        "episode_sha256": hashlib.sha256(
            json.dumps(selected, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "scene_id": selected["scene_id"],
        "instruction_sha256": hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
        "instruction_chars": len(instruction),
        "habitat_sim_version": habitat_sim.__version__,
        "renderer_created": True,
        "resolution_hw": [RESOLUTION, RESOLUTION],
        "depth_max_meters": DEPTH_MAX_METERS,
        "action": "move_forward",
        "reset": reset,
        "step": step,
        "episode_over_after_step": episode_over,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "rgb_reset.png").write_bytes(encode_png(rgb_reset))
    (output / "rgb_step.png").write_bytes(encode_png(rgb_step))
    write_depth_preview(output / "depth_reset.png", depth_reset, np)
    write_depth_preview(output / "depth_step.png", depth_step, np)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val_unseen"), required=True)
    parser.add_argument("--dataset-file", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    result = run(parser.parse_args())
    print(json.dumps({
        "status": result["status"],
        "split": result["split"],
        "episode_id": result["episode_id"],
        "observation_keys": result["reset"]["observation_keys"],
        "reset_distance": result["reset"]["distance_to_goal"],
        "step_distance": result["step"]["distance_to_goal"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
