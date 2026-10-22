"""Read-only Habitat-Sim navmesh smoke with rendering explicitly disabled.

This is standalone Python 3.9 code for an already installed Habitat-Sim 0.2.4
environment. It does not import the navrl core or request RGB/depth observations.
"""

import argparse
import json
import math
from pathlib import Path


def xyz(point):
    return [float(point[0]), float(point[1]), float(point[2])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--navmesh", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    scene = args.scene.resolve(strict=True)
    navmesh = args.navmesh.resolve(strict=True)
    if scene.suffix != ".glb" or navmesh.suffix != ".navmesh":
        parser.error("expected a .glb scene and a .navmesh file")
    if navmesh != scene.with_suffix(".navmesh"):
        parser.error("navmesh must be the adjacent file auto-loaded for this scene")

    import habitat_sim

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(scene)
    sim_cfg.create_renderer = False
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = []
    config = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(config)
    try:
        pathfinder = sim.pathfinder
        if not pathfinder.is_loaded:
            if not pathfinder.load_nav_mesh(str(navmesh)):
                raise RuntimeError("navmesh did not load")
        pathfinder.seed(args.seed)
        start = pathfinder.get_random_navigable_point()
        if not pathfinder.is_navigable(start):
            raise RuntimeError("first sampled point is not navigable")

        goal = None
        path = habitat_sim.ShortestPath()
        for _ in range(64):
            candidate = pathfinder.get_random_navigable_point()
            if not pathfinder.is_navigable(candidate):
                continue
            if math.dist(xyz(start), xyz(candidate)) < 0.5:
                continue
            path.requested_start = start
            path.requested_end = candidate
            if pathfinder.find_path(path) and math.isfinite(path.geodesic_distance):
                goal = candidate
                break
        if goal is None:
            raise RuntimeError("no second reachable navigable point found in 64 samples")

        # One non-visual control step checks that the loaded scene can drive an
        # agent. No camera sensor exists, so this cannot exercise RGB rendering.
        sim.reset()
        agent = sim.initialize_agent(0)
        state = agent.get_state()
        state.position = start
        agent.set_state(state)
        before = xyz(agent.get_state().position)
        observations = sim.step("move_forward")
        after = xyz(agent.get_state().position)

        print(
            json.dumps(
                {
                    "status": "ok",
                    "scene": str(scene),
                    "navmesh": str(navmesh),
                    "renderer_created": False,
                    "sensors": 0,
                    "navmesh_loaded": bool(pathfinder.is_loaded),
                    "start_navigable": bool(pathfinder.is_navigable(start)),
                    "goal_navigable": bool(pathfinder.is_navigable(goal)),
                    "start": xyz(start),
                    "goal": xyz(goal),
                    "geodesic_distance_m": float(path.geodesic_distance),
                    "path_points": len(path.points),
                    "agent_step_called": True,
                    "agent_step_displacement_m": math.dist(before, after),
                    "observation_keys": sorted(observations.keys()),
                    "visual_habitat_validated": False,
                    "limitation": "This does not validate visual Habitat or RGB/depth rendering.",
                },
                sort_keys=True,
            )
        )
    finally:
        sim.close()


if __name__ == "__main__":
    main()
