"""Run one official Habitat-Lab PointNav episode without visual rendering.

This validates an existing Habitat-Lab 0.2.4 task setup, not visual Habitat,
policy inference, or RL training. No installed package or shared repo is edited.
"""

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True


def main() -> None:
    import habitat
    import habitat_sim
    import numpy as np
    from habitat.config import read_write
    from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim

    project_root = Path(__file__).resolve().parents[3]
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
    assert data_path.is_file(), data_path
    assert (scenes_dir / "habitat-test-scenes" / "skokloster-castle.glb").is_file()
    assert (scenes_dir / "habitat-test-scenes" / "skokloster-castle.navmesh").is_file()

    config = habitat.get_config(
        "benchmark/nav/pointnav/pointnav_habitat_test.yaml"
    )
    with read_write(config):
        config.habitat.dataset.split = "val"
        config.habitat.dataset.data_path = str(data_path)
        config.habitat.dataset.scenes_dir = str(scenes_dir)
        config.habitat.simulator.agents.main_agent.sim_sensors.clear()
        config.habitat.simulator.create_renderer = False
        config.habitat.environment.iterator_options.shuffle = False
    assert not config.habitat.simulator.agents.main_agent.sim_sensors

    # Habitat-Lab 0.2.4 does not copy simulator.create_renderer to the
    # Habitat-Sim configuration. Set it in memory before Simulator.__init__.
    original_create_sim_config = HabitatSim.create_sim_config

    def create_no_renderer_config(self, sensor_suite):
        sim_config = original_create_sim_config(self, sensor_suite)
        assert not sim_config.agents[0].sensor_specifications
        sim_config.sim_cfg.create_renderer = False
        assert not sim_config.sim_cfg.create_renderer
        return sim_config

    HabitatSim.create_sim_config = create_no_renderer_config
    try:
        dataset = habitat.make_dataset(
            id_dataset=config.habitat.dataset.type,
            config=config.habitat.dataset,
        )
        assert dataset.episodes, "PointNav val dataset has no episodes"
        dataset.episodes = dataset.episodes[:1]
        with habitat.Env(config=config, dataset=dataset) as env:
            observations = env.reset()
            episode = env.current_episode
            assert episode.episode_id == "0", episode.episode_id
            assert not env.sim.sim_config.sim_cfg.create_renderer
            assert not env.sim.sim_config.agents[0].sensor_specifications
            reset_metrics = env.get_metrics()

            env.step("move_forward")
            move_metrics = env.get_metrics()
            assert not env.episode_over, "MOVE unexpectedly ended the episode"

            env.step("stop")
            stop_metrics = env.get_metrics()
            assert env.episode_over, "STOP did not end the episode"
            assert "success" in stop_metrics and "spl" in stop_metrics
            assert float(stop_metrics["success"]) == 0.0
            assert float(stop_metrics["spl"]) == 0.0

            # In memory only: choose a navigable point 0.1 m along the
            # official shortest path from the goal toward the original start.
            # This keeps the goal/scene/episode data intact on disk and gives
            # SPL a strictly positive reference distance.
            path = habitat_sim.ShortestPath()
            path.requested_start = np.asarray(
                episode.goals[0].position, dtype=np.float32
            )
            path.requested_end = np.asarray(
                episode.start_position, dtype=np.float32
            )
            assert env.sim.pathfinder.find_path(path)
            remaining = 0.1
            near_goal_start = None
            for before, after in zip(path.points, path.points[1:]):
                before = np.asarray(before, dtype=np.float32)
                after = np.asarray(after, dtype=np.float32)
                segment_length = float(np.linalg.norm(after - before))
                if remaining <= segment_length:
                    near_goal_start = before + (
                        remaining / segment_length
                    ) * (after - before)
                    break
                remaining -= segment_length
            assert near_goal_start is not None
            near_goal_start = env.sim.pathfinder.snap_point(near_goal_start)
            assert env.sim.pathfinder.is_navigable(near_goal_start)
            near_goal_distance = env.sim.geodesic_distance(
                near_goal_start, episode.goals[0].position
            )
            success_distance = config.habitat.task.measurements.success.success_distance
            assert 0.01 < near_goal_distance < success_distance, (
                near_goal_distance,
                success_distance,
            )

            episode.start_position = [float(v) for v in near_goal_start]
            env.current_episode = episode
            success_observations = env.reset()
            positive_reset_metrics = env.get_metrics()
            assert not env.episode_over
            env.step("stop")
            positive_stop_metrics = env.get_metrics()
            assert env.episode_over
            assert float(positive_stop_metrics["success"]) == 1.0
            assert abs(float(positive_stop_metrics["spl"]) - 1.0) < 1e-6

            print(
                json.dumps(
                    {
                        "status": "ok",
                        "episode_id": episode.episode_id,
                        "scene_id": episode.scene_id,
                        "initial_observation_keys": sorted(observations),
                        "premature_stop_control": {
                            "actions": ["move_forward", "stop"],
                            "distance_to_goal": {
                                "reset": float(reset_metrics["distance_to_goal"]),
                                "after_move": float(move_metrics["distance_to_goal"]),
                                "after_stop": float(stop_metrics["distance_to_goal"]),
                            },
                            "success": float(stop_metrics["success"]),
                            "spl": float(stop_metrics["spl"]),
                        },
                        "near_goal_positive": {
                            "actions": ["stop"],
                            "modified_start_position": episode.start_position,
                            "initial_observation_keys": sorted(success_observations),
                            "distance_to_goal_reset": float(
                                positive_reset_metrics["distance_to_goal"]
                            ),
                            "distance_to_goal_stop": float(
                                positive_stop_metrics["distance_to_goal"]
                            ),
                            "success": float(positive_stop_metrics["success"]),
                            "spl": float(positive_stop_metrics["spl"]),
                        },
                        "episode_over": env.episode_over,
                        "renderer_created": env.sim.sim_config.sim_cfg.create_renderer,
                        "visual_habitat_validated": False,
                    },
                    sort_keys=True,
                )
            )
    finally:
        HabitatSim.create_sim_config = original_create_sim_config


if __name__ == "__main__":
    main()
