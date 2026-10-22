"""Run deterministic mock episodes and optionally save portable JSONL rollouts.

From the rlinf-nav directory:
    python -m scripts.smoke_env --episodes 100 --seed 7
    python -m scripts.smoke_env --policy random --episodes 100 --seed 7
    python -m scripts.smoke_env --policy random --episodes 3 --trace-jsonl output/mock_rollouts.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

from navrl.contracts import NavAction
from navrl.envs.grid_mock import DIRECTIONS, GridNavEnv, shortest_path
from navrl.episode import GridEpisode


def route_actions(path: tuple[tuple[int, int], ...], heading: int) -> list[NavAction]:
    """Convert a BFS cell route into discrete turns, moves, then STOP."""
    actions: list[NavAction] = []
    for current, next_cell in zip(path, path[1:]):
        delta = (next_cell[0] - current[0], next_cell[1] - current[1])
        target_heading = DIRECTIONS.index(delta)
        turn = (target_heading - heading) % 4
        if turn == 1:
            actions.append(NavAction.TURN_LEFT)
        elif turn == 2:
            actions.extend((NavAction.TURN_LEFT, NavAction.TURN_LEFT))
        elif turn == 3:
            actions.append(NavAction.TURN_RIGHT)
        heading = target_heading
        actions.append(NavAction.MOVE_FORWARD)
    actions.append(NavAction.STOP)
    return actions


def replay_record(record: dict) -> None:
    """Verify that the saved episode and action sequence reproduce every scalar."""
    spec = record["episode"]
    episode = GridEpisode(
        episode_id=spec["episode_id"],
        start=tuple(spec["start"]),
        goal=tuple(spec["goal"]),
        heading=spec["heading"],
        instruction=spec["instruction"],
    )
    env = GridNavEnv(grid=record["grid"], max_steps=128)
    try:
        env.reset(episode)
        for action, expected in zip(record["actions"], record["steps"], strict=True):
            _, reward, terminated, truncated, info = env.step(action)
            actual = {"reward": reward, "terminated": terminated, "truncated": truncated, "info": info}
            if actual != expected:
                raise AssertionError(f"replay mismatch in {episode.episode_id}")
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--policy", choices=("oracle", "random"), default="oracle")
    parser.add_argument("--trace-jsonl", type=Path, default=None)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")

    env = GridNavEnv(seed=args.seed, max_steps=128)
    action_rng = random.Random(args.seed)
    traces: list[dict] = []
    success_count = 0
    spl_sum = 0.0
    collision_count = 0
    termination_counts: dict[str, int] = {}
    try:
        for _ in range(args.episodes):
            env.reset()
            episode = env.current_episode
            if args.policy == "oracle":
                path = shortest_path(env.grid, episode.start, episode.goal)
                assert path is not None
                planned_actions = iter(route_actions(path, episode.heading))
            step_records: list[dict] = []
            executed_actions: list[NavAction] = []
            while True:
                if args.policy == "oracle":
                    action = next(planned_actions, None)
                    if action is None:
                        raise AssertionError(f"oracle exhausted actions in {episode.episode_id}")
                else:
                    action = action_rng.choice(tuple(NavAction))
                executed_actions.append(action)
                _, reward, terminated, truncated, info = env.step(action)
                step_records.append(
                    {"reward": reward, "terminated": terminated, "truncated": truncated, "info": info}
                )
                if terminated or truncated:
                    break
            if args.policy == "oracle" and not info["success"]:
                raise AssertionError(f"oracle failed in {episode.episode_id}: {info}")
            success_count += int(info["success"])
            spl_sum += info["spl"]
            collision_count += info["collision_count"]
            reason = info["termination_reason"] or "unknown"
            termination_counts[reason] = termination_counts.get(reason, 0) + 1
            if args.trace_jsonl is not None:
                record = {
                    "schema": 1,
                    "policy": args.policy,
                    "grid": env.grid,
                    "episode": asdict(episode),
                    "actions": [action.value for action in executed_actions],
                    "steps": step_records,
                }
                replay_record(record)
                traces.append(record)
    finally:
        env.close()

    if args.trace_jsonl is not None:
        args.trace_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.trace_jsonl.open("w", encoding="utf-8", newline="\n") as stream:
            for record in traces:
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "episodes": args.episodes,
                "policy": args.policy,
                "seed": args.seed,
                "successes": success_count,
                "mean_spl": spl_sum / args.episodes,
                "collisions": collision_count,
                "termination_counts": termination_counts,
                "trace_jsonl": str(args.trace_jsonl) if args.trace_jsonl is not None else None,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
