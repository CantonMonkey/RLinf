"""Record deterministic, nonvisual Habitat-Lab PointNav random rollouts.

Uses the official Habitat test scenes and val episodes in this project's data/
directory. It borrows only the preinstalled Habitat-Lab 0.2.4 interpreter on
A800; the shared installation, dataset, scenes, and GPU remain untouched.
Agent poses are recorded after each action for evaluation/visualization only.
The random action policy never receives a pose or the PointGoal observation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True

MOVEMENT_ACTIONS = ("move_forward", "turn_left", "turn_right")


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def pose(sim: object) -> dict[str, list[float]]:
    state = sim.get_agent_state()
    quat = state.rotation
    # Habitat uses numpy-quaternion; store explicit x,y,z,w order.
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
        "distance_to_goal_reward": float(metrics["distance_to_goal_reward"]),
        "collisions": {
            "count": int(collision["count"]),
            "is_collision": bool(collision["is_collision"]),
        },
    }


def pointgoal(observations: dict) -> list[float]:
    return [float(x) for x in observations["pointgoal_with_gps_compass"]]


def run(args: argparse.Namespace) -> dict:
    import habitat
    from habitat.config import read_write
    from habitat.config.default_structured_configs import CollisionsMeasurementConfig
    from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
    from habitat.sims.habitat_simulator.actions import HabitatSimActions
    from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

    project_root = args.project_root.resolve()
    scenes_dir = project_root / "data" / "scene_datasets"
    data_path = (
        project_root
        / "data"
        / "datasets"
        / "pointnav"
        / "habitat-test-scenes"
        / "v1"
        / "val"
        / "val.json.gz"
    )
    if not data_path.is_file():
        raise FileNotFoundError(data_path)

    config = habitat.get_config("benchmark/nav/pointnav/pointnav_habitat_test.yaml")
    with read_write(config):
        config.habitat.seed = args.seed
        config.habitat.dataset.split = "val"
        config.habitat.dataset.data_path = str(data_path)
        config.habitat.dataset.scenes_dir = str(scenes_dir)
        config.habitat.simulator.agents.main_agent.sim_sensors.clear()
        config.habitat.simulator.create_renderer = False
        config.habitat.environment.max_episode_steps = args.max_steps
        config.habitat.environment.iterator_options.shuffle = False
        config.habitat.environment.iterator_options.group_by_scene = False
        config.habitat.environment.iterator_options.cycle = False
        config.habitat.task.measurements.collisions = CollisionsMeasurementConfig()

    # Habitat-Lab 0.2.4 does not forward create_renderer to Habitat-Sim.
    # Patch the configuration builder in this process and restore it afterward.
    original_create_sim_config = HabitatSim.create_sim_config

    def create_no_renderer_config(self, sensor_suite):
        sim_config = original_create_sim_config(self, sensor_suite)
        assert not sim_config.agents[0].sensor_specifications
        sim_config.sim_cfg.create_renderer = False
        return sim_config

    HabitatSim.create_sim_config = create_no_renderer_config
    try:
        dataset = habitat.make_dataset(
            id_dataset=config.habitat.dataset.type, config=config.habitat.dataset
        )
        if len(dataset.episodes) < args.episode_start + args.episodes:
            raise ValueError(
                f"Requested through episode {args.episode_start + args.episodes - 1}, "
                f"dataset has {len(dataset.episodes)}"
            )
        dataset.episodes = dataset.episodes[
            args.episode_start : args.episode_start + args.episodes
        ]
        scenes = sorted({str(episode.scene_id) for episode in dataset.episodes})
        if len(scenes) != 1:
            raise ValueError(
                "The old no-renderer Habitat-Sim crashes on scene changes; "
                "run one scene per process, then merge the traces"
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temp_output = args.output.with_name(args.output.name + ".tmp")
        results = []
        with habitat.Env(config=config, dataset=dataset) as env, temp_output.open(
            "w", encoding="utf-8", newline="\n"
        ) as stream:
            env.seed(args.seed)
            reset_seconds = 0.0
            step_seconds = 0.0
            rollout_start = time.perf_counter()
            for local_episode_index in range(args.episodes):
                episode_index = args.episode_start + local_episode_index
                reset_start = time.perf_counter()
                observations = env.reset()
                reset_seconds += time.perf_counter() - reset_start
                assert not env.sim.sim_config.sim_cfg.create_renderer
                assert not env.sim.sim_config.agents[0].sensor_specifications
                episode = env.current_episode
                initial_pointgoal = pointgoal(observations)
                initial_metrics = scalar_metrics(env.get_metrics())
                initial_pose = pose(env.sim)
                trace = []
                rng = random.Random(args.seed + 1000003 * episode_index)
                previous_position = initial_pose["position_xyz"]
                path_length = 0.0
                movement_collision_count = 0
                stop_called = False
                follower = (
                    ShortestPathFollower(
                        env.sim,
                        goal_radius=0.9
                        * float(config.habitat.task.measurements.success.success_distance),
                        return_one_hot=False,
                        stop_on_error=False,
                    )
                    if args.policy == "oracle"
                    else None
                )
                while not env.episode_over:
                    if follower is not None:
                        # Evaluation-only oracle: Habitat's follower reads the
                        # privileged goal position and simulator agent state.
                        action_id = follower.get_next_action(episode.goals[0].position)
                        action = {
                            HabitatSimActions.stop: "stop",
                            HabitatSimActions.move_forward: "move_forward",
                            HabitatSimActions.turn_left: "turn_left",
                            HabitatSimActions.turn_right: "turn_right",
                        }.get(action_id)
                        if action is None:
                            raise RuntimeError(f"Follower returned {action_id!r}")
                    else:
                        # Delayed stochastic STOP yields both stop and timeout
                        # cases. The random policy sees only RNG and step index.
                        action = (
                            "stop"
                            if len(trace) >= args.min_moves
                            and rng.random() < args.stop_probability
                            else rng.choice(MOVEMENT_ACTIONS)
                        )
                    step_start = time.perf_counter()
                    observations = env.step(action)
                    step_seconds += time.perf_counter() - step_start
                    metrics = scalar_metrics(env.get_metrics())
                    agent_pose = pose(env.sim)
                    position = agent_pose["position_xyz"]
                    path_length += math.dist(previous_position, position)
                    previous_position = position
                    if action == "move_forward" and metrics["collisions"]["is_collision"]:
                        movement_collision_count += 1
                    stop_called = stop_called or action == "stop"
                    trace.append(
                        {
                            "step": len(trace) + 1,
                            "action": action,
                            "pointgoal": pointgoal(observations),
                            "pose": agent_pose,
                            "metrics": metrics,
                            "episode_over": bool(env.episode_over),
                        }
                    )
                final_metrics = trace[-1]["metrics"]
                success = bool(final_metrics["success"])
                if success != (
                    stop_called
                    and final_metrics["distance_to_goal"]
                    < float(config.habitat.task.measurements.success.success_distance)
                ):
                    raise AssertionError("Habitat STOP/Success semantics changed")
                ref_distance = initial_metrics["distance_to_goal"]
                expected_spl = (
                    ref_distance / max(ref_distance, path_length) if success else 0.0
                )
                if not math.isclose(final_metrics["spl"], expected_spl, abs_tol=1e-5):
                    raise AssertionError(
                        f"Habitat SPL mismatch: {final_metrics['spl']} vs {expected_spl}"
                    )
                outcome = {
                    "success": success,
                    "spl": final_metrics["spl"],
                    "distance_to_goal": final_metrics["distance_to_goal"],
                    "path_length": path_length,
                    "collision_count_official": final_metrics["collisions"]["count"],
                    "collision_count_move_forward": movement_collision_count,
                    "stop_called": stop_called,
                    "timeout": not stop_called and len(trace) == args.max_steps,
                    "termination_reason": (
                        "stop_success" if success else "stop_failure" if stop_called else "timeout"
                    ),
                    "steps": len(trace),
                }
                record = {
                    "episode_index": episode_index,
                    "episode_id": str(episode.episode_id),
                    "scene_id": str(episode.scene_id),
                    "goal_position_xyz": [float(x) for x in episode.goals[0].position],
                    "initial_pointgoal": initial_pointgoal,
                    "initial_pose": initial_pose,
                    "initial_metrics": initial_metrics,
                    "actions": [step["action"] for step in trace],
                    "steps": trace,
                    "outcome": outcome,
                }
                if follower is not None:
                    record["policy"] = "geodesic_oracle"
                    record["policy_uses_privileged_target_and_pose"] = True
                stream.write(canonical_json(record) + "\n")
                stream.flush()  # Preserve the last completed episode after a native crash.
                results.append(outcome)
            rollout_wall_seconds = time.perf_counter() - rollout_start
        temp_output.replace(args.output)
    finally:
        HabitatSim.create_sim_config = original_create_sim_config

    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    summary = {
        "status": "ok",
        "simulator": "Habitat-Lab 0.2.4 / Habitat-Sim 0.2.4",
        "dataset": "official habitat-test-scenes PointNav v1 val",
        "seed": args.seed,
        "episode_start": args.episode_start,
        "scene_id": scenes[0],
        "episodes": len(results),
        "max_steps": args.max_steps,
        "min_moves": args.min_moves,
        "stop_probability": args.stop_probability,
        "success_count": sum(int(r["success"]) for r in results),
        "mean_spl": sum(r["spl"] for r in results) / len(results),
        "mean_final_distance": sum(r["distance_to_goal"] for r in results) / len(results),
        "total_steps": sum(r["steps"] for r in results),
        "total_official_collisions": sum(r["collision_count_official"] for r in results),
        "total_move_forward_collisions": sum(
            r["collision_count_move_forward"] for r in results
        ),
        "stop_count": sum(int(r["stop_called"]) for r in results),
        "timeout_count": sum(int(r["timeout"]) for r in results),
        "visual_habitat_validated": False,
        "renderer_created": False,
        "trace_sha256": digest,
        "trace_path": str(args.output),
    }
    if args.policy == "oracle":
        summary.pop("min_moves")
        summary.pop("stop_probability")
        summary["policy"] = "geodesic_oracle"
        summary["policy_uses_privileged_target_and_pose"] = True
        summary["reset_seconds"] = reset_seconds
        summary["step_seconds"] = step_seconds
        summary["rollout_wall_seconds"] = rollout_wall_seconds
        summary["step_calls_per_second"] = sum(r["steps"] for r in results) / step_seconds
        summary["rollout_steps_per_second"] = (
            sum(r["steps"] for r in results) / rollout_wall_seconds
        )
        summary["timing_scope"] = "No-renderer single environment; excludes Python startup and RGB rendering"
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(canonical_json(summary) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--min-moves", type=int, default=4)
    parser.add_argument("--stop-probability", type=float, default=0.04)
    parser.add_argument("--policy", choices=("random", "oracle"), default="random")
    args = parser.parse_args()
    if args.episodes <= 0 or args.max_steps <= 0 or args.episode_start < 0:
        parser.error("episodes and max-steps must be positive; episode-start >= 0")
    if not (0 <= args.min_moves < args.max_steps):
        parser.error("min-moves must be in [0, max-steps)")
    if not (0 <= args.stop_probability <= 1):
        parser.error("stop-probability must be in [0, 1]")
    print(canonical_json(run(args)))


if __name__ == "__main__":
    main()
