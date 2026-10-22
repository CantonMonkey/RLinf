"""Bounded R2R Habitat RGB/instruction probe; never feed simulator goals to policy.

Run this in the private Habitat-Lab 0.2.4 Python 3.9 environment. The resulting
``policy_obs.jsonl`` contains only RGB bytes and official instruction token IDs.
Simulator geometry and metrics are written to a separate evaluator-only file.
This is a reset/step interface probe, not a trained VLN policy or benchmark.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import math
from pathlib import Path


ACTIONS = ("stop", "move_forward", "turn_left", "turn_right")
SCENE_PREFIX = "data/scene_datasets/"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_r2r_metadata(dataset_path: Path, episode_index: int, scenes_dir: Path):
    """Check the real R2RVLN-v1 JSON schema and the referenced MP3D scene."""
    with gzip.open(dataset_path, "rt", encoding="utf-8") as stream:
        document = json.load(stream)
    if not isinstance(document, dict) or not isinstance(document.get("episodes"), list):
        raise ValueError("R2R dataset must contain an episodes list")
    if not 0 <= episode_index < len(document["episodes"]):
        raise IndexError("episode index outside R2R split")
    words = document.get("instruction_vocab", {}).get("word_list")
    if not isinstance(words, list) or not words:
        raise ValueError("R2R instruction_vocab.word_list is missing")
    episode = document["episodes"][episode_index]
    if not isinstance(episode, dict):
        raise ValueError("R2R episode must be an object")
    instruction = episode.get("instruction")
    if not isinstance(instruction, dict):
        raise ValueError("R2R episode has no instruction object")
    text = instruction.get("instruction_text")
    tokens = instruction.get("instruction_tokens")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("R2R episode has no instruction text")
    if (
        not isinstance(tokens, list)
        or not tokens
        or any(type(token) is not int or token < 0 or token >= len(words) for token in tokens)
    ):
        raise ValueError("R2R episode has invalid official instruction tokens")
    if not isinstance(episode.get("goals"), list) or not episode["goals"]:
        raise ValueError("R2R episode has no navigation goal")
    if not isinstance(episode.get("reference_path"), list) or not episode["reference_path"]:
        raise ValueError("R2R episode has no reference path")
    scene_id = episode.get("scene_id")
    if not isinstance(scene_id, str) or not scene_id.endswith(".glb"):
        raise ValueError("R2R episode has no MP3D GLB scene")
    relative_scene = scene_id.removeprefix(SCENE_PREFIX)
    scene_file = (scenes_dir / relative_scene).resolve(strict=True)
    if not scene_file.is_relative_to(scenes_dir.resolve(strict=True)):
        raise ValueError("R2R scene escapes scenes_dir")
    navmesh = scene_file.with_suffix(".navmesh")
    if not navmesh.is_file():
        raise FileNotFoundError(navmesh)
    return episode, words, scene_file, navmesh


def validate_instruction(observation: dict, episode, vocab_size: int, max_tokens: int):
    """Use Habitat's official InstructionSensor and reject identity drift."""
    item = observation.get("instruction")
    if not isinstance(item, dict):
        raise ValueError("Habitat did not return its instruction sensor")
    text = item.get("text")
    tokens = item.get("tokens")
    if text != episode.instruction.instruction_text:
        raise ValueError("Habitat instruction text differs from selected R2R episode")
    if not isinstance(tokens, (list, tuple)) or not tokens:
        raise ValueError("Habitat instruction tokens are missing")
    if list(tokens) != list(episode.instruction.instruction_tokens):
        raise ValueError("Habitat instruction tokens differ from selected R2R episode")
    if len(tokens) > max_tokens:
        raise ValueError("R2R instruction exceeds configured token width")
    if any(type(token) is not int or token < 0 or token >= vocab_size for token in tokens):
        raise ValueError("R2R instruction token ID outside official vocabulary")
    # The official R2R files already use PAD_INDEX=0 and 200-wide sequences.
    # Preserve those IDs exactly, including pre-existing padding.
    seen_pad = False
    for token in tokens:
        if token == 0:
            seen_pad = True
        elif seen_pad:
            raise ValueError("R2R instruction has a non-PAD token after PAD")
    ids = list(tokens) + [0] * (max_tokens - len(tokens))
    mask = [token != 0 for token in ids]
    return ids, mask


def make_policy_obs(rgb_bytes: bytes, shape_hwc, ids, mask):
    """The only observation fields permitted across the future policy bridge."""
    shape = tuple(shape_hwc)
    if len(shape) != 3 or shape[2] != 3 or shape[0] <= 0 or shape[1] <= 0:
        raise ValueError("RGB must have positive HWC shape with three channels")
    if len(rgb_bytes) != shape[0] * shape[1] * 3:
        raise ValueError("RGB byte count does not match HWC shape")
    if len(ids) != len(mask) or not ids:
        raise ValueError("instruction IDs/mask shape mismatch")
    if any(type(token) is not int or token < 0 for token in ids):
        raise ValueError("instruction IDs must be nonnegative integers")
    if any(type(value) is not bool for value in mask):
        raise ValueError("instruction mask must contain booleans")
    if any(ids[index] != 0 for index, present in enumerate(mask) if not present):
        raise ValueError("padded instruction IDs must be zero")
    return {
        "rgb": {
            "encoding": "base64/raw-u8-hwc",
            "shape_hwc": list(shape),
            "data_b64": base64.b64encode(rgb_bytes).decode("ascii"),
        },
        "instruction_ids": ids,
        "instruction_mask": mask,
    }


def make_task_config(
    habitat, read_write, dataset_path, scenes_dir, split, max_steps, rgb_size,
    *, include_depth=False, depth_max_m=10.0, depth_pitch_radians=0.0,
):
    if (
        type(include_depth) is not bool
        or not math.isfinite(depth_max_m)
        or depth_max_m <= 0
        or not math.isfinite(depth_pitch_radians)
        or not -math.pi / 2 < depth_pitch_radians < math.pi / 2
    ):
        raise ValueError("invalid optional R2R depth configuration")
    config = habitat.get_config("benchmark/nav/vln_r2r.yaml")
    with read_write(config):
        config.habitat.dataset.split = split
        config.habitat.dataset.data_path = str(dataset_path)
        config.habitat.dataset.scenes_dir = str(scenes_dir)
        config.habitat.environment.max_episode_steps = max_steps
        config.habitat.environment.iterator_options.shuffle = False
        config.habitat.environment.iterator_options.cycle = False
        config.habitat.simulator.create_renderer = True
        config.habitat.simulator.habitat_sim_v0.gpu_device_id = 0
        config.habitat.simulator.habitat_sim_v0.gpu_gpu = False
        sensors = config.habitat.simulator.agents.main_agent.sim_sensors
        if include_depth:
            depth_sensor = sensors.depth_sensor
            depth_sensor.height = rgb_size
            depth_sensor.width = rgb_size
            depth_sensor.normalize_depth = False
            depth_sensor.min_depth = 0.0
            depth_sensor.max_depth = float(depth_max_m)
            depth_sensor.orientation = [float(depth_pitch_radians), 0.0, 0.0]
        else:
            # RGB enters the policy. Depth is opt-in for explicit lower-controller
            # diagnostics and never appears in the default policy observation.
            sensors.pop("depth_sensor", None)
        rgb_sensor = sensors.rgb_sensor
        rgb_sensor.height = rgb_size
        rgb_sensor.width = rgb_size
    if config.habitat.dataset.type != "R2RVLN-v1" or config.habitat.task.type != "VLN-v0":
        raise RuntimeError("Expected official Habitat R2R VLN dataset/task")
    if set(config.habitat.task.lab_sensors) != {"instruction_sensor"}:
        raise RuntimeError("Unexpected R2R task sensor set")
    expected = {"rgb_sensor", "depth_sensor"} if include_depth else {"rgb_sensor"}
    if set(config.habitat.simulator.agents.main_agent.sim_sensors) != expected:
        raise RuntimeError("Unexpected R2R visual sensor set")
    return config


def make_record(observations, env, selected_episode, vocab_size, max_tokens, rgb_size, np):
    if str(env.current_episode.episode_id) != str(selected_episode.episode_id):
        raise RuntimeError("R2R episode identity changed during smoke")
    ids, mask = validate_instruction(observations, selected_episode, vocab_size, max_tokens)
    if "rgb" not in observations:
        raise RuntimeError("Habitat R2R RGB sensor is absent")
    rgb = np.ascontiguousarray(observations["rgb"])
    if rgb.dtype != np.uint8:
        raise ValueError("Habitat RGB must be uint8")
    if rgb.shape != (rgb_size, rgb_size, 3):
        raise ValueError("Habitat RGB shape differs from requested sensor size")
    policy_obs = make_policy_obs(rgb.tobytes(), rgb.shape, ids, mask)
    metrics = env.get_metrics()
    state = env.sim.get_agent_state()
    evaluator_only = {
        "episode_id": str(selected_episode.episode_id),
        "scene_id": str(selected_episode.scene_id),
        "instruction_text": selected_episode.instruction.instruction_text,
        "trajectory_id": selected_episode.trajectory_id,
        "goal_positions_xyz": [list(goal.position) for goal in selected_episode.goals],
        "reference_path": selected_episode.reference_path,
        "agent_position_xyz": [float(value) for value in state.position],
        "distance_to_goal": float(metrics["distance_to_goal"]),
        "success": bool(metrics["success"]),
        "spl": float(metrics["spl"]),
        "episode_over": bool(env.episode_over),
        "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
    }
    return policy_obs, evaluator_only


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--scenes-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val_seen", "val_unseen"), required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--actions", default="move_forward")
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--rgb-size", type=int, default=64)
    parser.add_argument("--max-instruction-tokens", type=int, default=200)
    args = parser.parse_args(argv)
    if args.max_steps < 1 or args.max_instruction_tokens < 1 or args.rgb_size < 1:
        parser.error("max-steps, rgb-size and max-instruction-tokens must be positive")
    args.actions = [name.strip() for name in args.actions.split(",")]
    if not args.actions or any(name not in ACTIONS for name in args.actions):
        parser.error("actions must be comma-separated R2R action names")
    if len(args.actions) > args.max_steps:
        parser.error("actions exceed max-steps")
    if args.dataset_path.name != f"{args.split}.json.gz":
        parser.error("dataset filename must match split")
    return args


def main(argv=None):
    args = parse_args(argv)
    dataset_path = args.dataset_path.resolve(strict=True)
    scenes_dir = args.scenes_dir.resolve(strict=True)
    raw_episode, words, scene_file, navmesh = read_r2r_metadata(
        dataset_path, args.episode_index, scenes_dir
    )
    if len(raw_episode["instruction"]["instruction_tokens"]) > args.max_instruction_tokens:
        raise ValueError("selected R2R instruction exceeds configured token width")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    import habitat
    import numpy as np
    from habitat.config import read_write

    config = make_task_config(
        habitat, read_write, dataset_path, scenes_dir, args.split, args.max_steps,
        args.rgb_size,
    )
    dataset = habitat.make_dataset(id_dataset=config.habitat.dataset.type, config=config.habitat.dataset)
    selected = dataset.episodes[args.episode_index]
    if (
        str(selected.episode_id) != str(raw_episode["episode_id"])
        or Path(selected.scene_id).resolve() != scene_file
        or selected.instruction.instruction_text != raw_episode["instruction"]["instruction_text"]
    ):
        raise RuntimeError("Habitat R2R episode differs from audited JSON index")
    dataset.episodes = [selected]  # never switch MP3D scenes in this process
    args.output_dir.mkdir(parents=True)
    try:
        env = habitat.Env(config=config, dataset=dataset)
        try:
            policy_path = args.output_dir / "policy_obs.jsonl"
            eval_path = args.output_dir / "evaluator_only.jsonl"
            with policy_path.open("x", encoding="utf-8") as policy_stream, eval_path.open(
                "x", encoding="utf-8"
            ) as eval_stream:
                def write_record(event, action, step_id, frame):
                    policy_obs, evaluator = make_record(
                        frame, env, selected, len(words), args.max_instruction_tokens,
                        args.rgb_size, np,
                    )
                    # Geometry never shares a file or JSON object with policy data.
                    policy_stream.write(json.dumps(policy_obs, separators=(",", ":")) + "\n")
                    evaluator.update(event=event, action=action, step_id=step_id)
                    eval_stream.write(json.dumps(evaluator, separators=(",", ":")) + "\n")

                write_record("reset", None, 0, env.reset())
                frames = 1
                for name in args.actions:
                    if env.episode_over:
                        break
                    write_record("step", name, frames, env.step(name))
                    frames += 1
        finally:
            env.close()
        manifest = {
            "mode": "r2r_rgb_instruction_reset_step_smoke",
            "split": args.split,
            "dataset_path": str(dataset_path),
            "dataset_sha256": file_sha256(dataset_path),
            "scene_path": str(scene_file),
            "navmesh_path": str(navmesh),
            "episode_index": args.episode_index,
            "episode_id": str(selected.episode_id),
            "vocab_size": len(words),
            "vocab_sha256": hashlib.sha256(
                json.dumps(words, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "max_instruction_tokens": args.max_instruction_tokens,
            "rgb_size": args.rgb_size,
            "actions_requested": args.actions,
            "policy_input_keys": ["rgb", "instruction_ids", "instruction_mask"],
            "policy_obs_sha256": file_sha256(policy_path),
            "evaluator_only_sha256": file_sha256(eval_path),
            "frames": frames,
        }
        (args.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps({"output_dir": str(args.output_dir), "frames": frames, "episode_id": manifest["episode_id"]}))
    except BaseException:
        # Keep any partial output for diagnosis; it cannot be mistaken for a completed run.
        raise


if __name__ == "__main__":
    main()
