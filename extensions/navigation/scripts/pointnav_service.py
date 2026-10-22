"""Isolated Habitat-Lab PointNav process using newline-delimited JSON.

The RLinf process can run a different Python and PyTorch version. Its policy
receives only the official PointGoal vector. An explicit diagnostic request
exposes pose and goal for offline trace rendering, never through observations
or normal reset/step responses. Habitat is imported lazily for unit tests.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from pathlib import Path


ACTION_NAMES = ("stop", "move_forward", "turn_left", "turn_right")
DATASET_SPLITS = ("train", "val")


def pointnav_dataset_path(asset_root, split="val"):
    """Return the official PointNav file for one explicit dataset split."""
    if split not in DATASET_SPLITS:
        raise ValueError(f"unsupported PointNav split: {split}")
    return (
        Path(asset_root).resolve() / "datasets" / "pointnav"
        / "habitat-test-scenes" / "v1" / split / f"{split}.json.gz"
    )


def select_episodes(episodes, *, episode_index=0, episode_indices=None, split="val"):
    """Select a bounded, same-scene subset without changing dataset order."""
    selected_indices = [episode_index] if episode_indices is None else list(episode_indices)
    if not selected_indices or len(set(selected_indices)) != len(selected_indices):
        raise ValueError("episode indices must be nonempty and unique")
    if any(not 0 <= index < len(episodes) for index in selected_indices):
        raise IndexError(f"episode index outside PointNav {split} dataset")
    selected = [episodes[index] for index in selected_indices]
    if len({episode.scene_id for episode in selected}) != 1:
        raise ValueError("one PointNav process cannot switch scenes")
    return selected


class PointNavSession:
    def __init__(
        self,
        backend,
        *,
        max_steps=32,
        progress_weight=1.0,
        success_bonus=10.0,
        step_penalty=0.01,
        collision_penalty=0.2,
        premature_stop_penalty=0.0,
    ):
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.backend = backend
        self.max_steps = int(max_steps)
        self.progress_weight = float(progress_weight)
        self.success_bonus = float(success_bonus)
        self.step_penalty = float(step_penalty)
        self.collision_penalty = float(collision_penalty)
        self.premature_stop_penalty = float(premature_stop_penalty)
        if not math.isfinite(self.premature_stop_penalty) or self.premature_stop_penalty < 0:
            raise ValueError("premature STOP penalty must be finite and nonnegative")
        self.steps = 0
        self.previous_geodesic = None
        self.last_frame = None
        self.final_info = None

    def reset(self):
        frame = self.backend.reset()
        self.steps = 0
        self.previous_geodesic = self._distance(frame)
        self.last_frame = frame
        self.final_info = None
        return {
            "obs": self._states(frame),
            "info": {
                "episode_id": str(frame["episode_id"]),
                "scene_id": str(frame["scene_id"]),
                "geodesic_distance": self.previous_geodesic,
            },
        }

    @staticmethod
    def _distance(frame):
        distance = float(frame["geodesic_distance"])
        if not math.isfinite(distance) or distance < 0:
            raise ValueError("PointNav geodesic distance must be finite and nonnegative")
        return distance

    @staticmethod
    def _states(frame):
        states = [float(value) for value in frame["states"]]
        if len(states) != 2 or not all(math.isfinite(value) for value in states):
            raise ValueError("PointGoal observation must contain two finite floats")
        return states

    def diagnostic(self):
        """Return privileged geometry only on a separate explicit request."""
        if self.last_frame is None:
            raise RuntimeError("reset must precede diagnostic")
        frame = self.last_frame
        fields = (
            ("position_xyz", 3),
            ("rotation_xyzw", 4),
            ("goal_position_xyz", 3),
        )
        result = {}
        for name, expected_size in fields:
            values = [float(value) for value in frame[name]]
            if len(values) != expected_size or not all(math.isfinite(value) for value in values):
                raise ValueError(f"invalid PointNav diagnostic {name}")
            result[name] = values
        result["episode_id"] = str(frame["episode_id"])
        result["scene_id"] = str(frame["scene_id"])
        return result

    def step(self, action):
        if self.last_frame is None:
            raise RuntimeError("reset must precede step")
        if isinstance(action, bool) or not isinstance(action, int) or not 0 <= action < 4:
            raise ValueError("PointNav action must be an integer from 0 to 3")
        if self.final_info is not None:
            # RLinf keeps asking for actions until its fixed rollout length.
            # Preserve the terminal observation and emit no extra transition.
            return {
                "obs": self._states(self.last_frame),
                "reward": 0.0,
                "terminated": False,
                "truncated": False,
                "info": dict(self.final_info, terminal_frozen=True),
            }

        frame = self.backend.step(ACTION_NAMES[action])
        self.steps += 1
        current_geodesic = self._distance(frame)
        success = bool(frame["success"])
        stop_called = action == 0
        if success and not stop_called:
            raise ValueError("Habitat reported success without STOP")
        collision = bool(frame["collision"])
        reward = (
            self.progress_weight * (self.previous_geodesic - current_geodesic)
            + (self.success_bonus if success else 0.0)
            - self.step_penalty
            - (self.collision_penalty if collision else 0.0)
            - (self.premature_stop_penalty if stop_called and not success else 0.0)
        )
        self.previous_geodesic = current_geodesic
        timeout = not stop_called and (
            self.steps >= self.max_steps or bool(frame["episode_over"])
        )
        terminated = stop_called
        truncated = timeout
        info = {
            "success": success,
            "distance_to_goal": float(frame["euclidean_distance"]),
            "geodesic_distance": current_geodesic,
            "path_length": float(frame["path_length"]),
            "spl": float(frame["spl"]),
            "collision": collision,
            "collision_count": int(frame["collision_count"]),
            "stop_called": stop_called,
            "timeout": timeout,
            "invalid_action": False,
            "termination_reason": (
                "success" if success else "premature_stop"
            ) if stop_called else ("timeout" if timeout else None),
            "episode_id": str(frame["episode_id"]),
            "scene_id": str(frame["scene_id"]),
        }
        self.last_frame = frame
        if terminated or truncated:
            self.final_info = dict(info)
        return {
            "obs": self._states(frame),
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
        }

    def close(self):
        self.backend.close()


class HabitatPointNavBackend:
    """One official Habitat-Lab PointNav task without visual sensors."""

    def __init__(
        self, asset_root, *, max_steps=32, episode_index=0, episode_indices=None,
        split="val",
    ):
        import habitat
        import numpy as np
        from habitat.config import read_write
        from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim

        self.np = np
        self.HabitatSim = HabitatSim
        self.original_create_sim_config = HabitatSim.create_sim_config
        asset_root = Path(asset_root).resolve()
        scenes_dir = asset_root / "scene_datasets"
        data_path = pointnav_dataset_path(asset_root, split)
        if not data_path.is_file():
            raise FileNotFoundError(data_path)
        config = habitat.get_config("benchmark/nav/pointnav/pointnav_habitat_test.yaml")
        with read_write(config):
            config.habitat.dataset.split = split
            config.habitat.dataset.data_path = str(data_path)
            config.habitat.dataset.scenes_dir = str(scenes_dir)
            config.habitat.simulator.agents.main_agent.sim_sensors.clear()
            config.habitat.simulator.create_renderer = False
            config.habitat.environment.max_episode_steps = int(max_steps)
            config.habitat.environment.iterator_options.shuffle = False
            config.habitat.environment.iterator_options.cycle = True

        def create_no_renderer_config(sim, sensor_suite):
            sim_config = self.original_create_sim_config(sim, sensor_suite)
            if sim_config.agents[0].sensor_specifications:
                raise AssertionError("visual sensor unexpectedly enabled")
            sim_config.sim_cfg.create_renderer = False
            return sim_config

        HabitatSim.create_sim_config = create_no_renderer_config
        selected_indices = (
            [episode_index] if episode_indices is None else list(episode_indices)
        )
        try:
            dataset = habitat.make_dataset(
                id_dataset=config.habitat.dataset.type,
                config=config.habitat.dataset,
            )
            selected_episodes = select_episodes(
                dataset.episodes, episode_index=episode_index,
                episode_indices=episode_indices, split=split,
            )
            dataset.episodes = selected_episodes
            self.env = habitat.Env(config=config, dataset=dataset)
        except BaseException:
            HabitatSim.create_sim_config = self.original_create_sim_config
            raise
        self.previous_position = None
        self.path_length = 0.0
        self.collision_count = 0
        self.split = split
        self.data_path = data_path
        self.selected_indices = selected_indices
        self.episode_ids = [str(episode.episode_id) for episode in selected_episodes]
        self.scene_id = str(selected_episodes[0].scene_id)

    def _frame(self, observations, collision):
        states = self.np.asarray(
            observations["pointgoal_with_gps_compass"], dtype=self.np.float32
        )
        if states.shape != (2,):
            raise ValueError("PointGoal sensor shape changed")
        metrics = self.env.get_metrics()
        episode = self.env.current_episode
        position = self.np.asarray(self.env.sim.get_agent_state().position)
        rotation = self.env.sim.get_agent_state().rotation
        goal = self.np.asarray(episode.goals[0].position)
        if self.previous_position is not None:
            self.path_length += float(self.np.linalg.norm(position - self.previous_position))
        self.previous_position = position.copy()
        self.collision_count += int(collision)
        return {
            "states": states.tolist(),
            # Habitat's distance_to_goal with distance_to=POINT is geodesic.
            "geodesic_distance": float(metrics["distance_to_goal"]),
            "euclidean_distance": float(self.np.linalg.norm(position - goal)),
            "success": bool(metrics["success"]),
            "spl": float(metrics["spl"]),
            "collision": bool(collision),
            "collision_count": self.collision_count,
            "path_length": self.path_length,
            "episode_over": bool(self.env.episode_over),
            "episode_id": str(episode.episode_id),
            "scene_id": str(episode.scene_id),
            # This remains private to the service except for the explicit
            # diagnostic command used by offline evaluators.
            "position_xyz": [float(value) for value in position],
            "rotation_xyzw": [
                float(rotation.imag[0]), float(rotation.imag[1]),
                float(rotation.imag[2]), float(rotation.real),
            ],
            "goal_position_xyz": [float(value) for value in goal],
        }

    def reset(self):
        observations = self.env.reset()
        self.previous_position = None
        self.path_length = 0.0
        self.collision_count = 0
        return self._frame(observations, False)

    def step(self, action_name):
        observations = self.env.step(action_name)
        return self._frame(observations, self.env.sim.previous_step_collided)

    def close(self):
        try:
            self.env.close()
        finally:
            self.HabitatSim.create_sim_config = self.original_create_sim_config


def serve_requests(session, request_lines, protocol):
    for line in request_lines:
        try:
            request = json.loads(line)
            if request.get("cmd") == "reset":
                result = session.reset()
            elif request.get("cmd") == "step":
                result = session.step(request.get("action"))
            elif request.get("cmd") == "diagnostic":
                result = session.diagnostic()
            elif request.get("cmd") == "close":
                protocol.write(json.dumps({"closed": True}) + "\n")
                break
            else:
                raise ValueError("unknown PointNav command")
            protocol.write(json.dumps(result, allow_nan=False) + "\n")
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            protocol.write(json.dumps({"error": str(exc)}) + "\n")


def service_ready_payload(backend):
    """Report the dataset and episodes actually loaded by Habitat."""
    return {
        "ready": True,
        "actions": ACTION_NAMES,
        "split": backend.split,
        "dataset_path": str(backend.data_path),
        "selected_indices": backend.selected_indices,
        "episode_ids": backend.episode_ids,
        "scene_id": backend.scene_id,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--split", choices=DATASET_SPLITS, default="val")
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument(
        "--premature-stop-penalty", type=float, default=0.0,
        help="optional extra cost for unsuccessful STOP; default preserves the reference reward",
    )
    episode_arg = parser.add_mutually_exclusive_group()
    episode_arg.add_argument("--episode-index", type=int, default=0)
    episode_arg.add_argument("--episode-indices", type=str)
    args = parser.parse_args(argv)
    if not math.isfinite(args.premature_stop_penalty) or args.premature_stop_penalty < 0:
        parser.error("--premature-stop-penalty must be finite and nonnegative")
    return args


def main():
    args = parse_args()

    # C++/Habitat logs may write to fd 1. Keep the original fd for JSON only.
    sys.stdout.flush()
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    backend = HabitatPointNavBackend(
        args.asset_root,
        max_steps=args.max_steps,
        split=args.split,
        episode_index=args.episode_index,
        episode_indices=(
            [int(item) for item in args.episode_indices.split(",")]
            if args.episode_indices is not None
            else None
        ),
    )
    session = PointNavSession(
        backend, max_steps=args.max_steps,
        premature_stop_penalty=args.premature_stop_penalty,
    )
    protocol.write(json.dumps(service_ready_payload(backend)) + "\n")
    try:
        serve_requests(session, sys.stdin, protocol)
    finally:
        session.close()
        protocol.close()


if __name__ == "__main__":
    main()
