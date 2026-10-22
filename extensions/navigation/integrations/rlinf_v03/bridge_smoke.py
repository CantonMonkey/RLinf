"""Read-only real Habitat transition and RLinf bridge shape smoke on A800.

This can run in the existing Habitat Python 3.9 environment.  It aliases the
old installed Gym package only for this isolated test; the RLinf v0.3 runtime
uses its own Gymnasium installation.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

try:
    import gymnasium  # noqa: F401
except ImportError:
    import gym

    sys.modules["gymnasium"] = gym


class Config(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


def main():
    root = Path("/srv/navigation")
    env_file = root / "repos" / "RLinf-v0.3" / "rlinf" / "envs" / "pointnav" / "pointnav_env.py"
    spec = importlib.util.spec_from_file_location("pointnav_env_smoke", env_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = Config(
        seed=7,
        auto_reset=False,
        max_episode_steps=4,
        python_bin=str(root.parent / "miniforge3" / "envs" / "streamvln" / "bin" / "python"),
        server_script=str(root / "repos" / "rlinf-nav" / "scripts" / "pointnav_service.py"),
        asset_root=str(root / "data"),
        dataset_split="train",
        episode_indices=[50, 51],
        video_cfg=Config(save_video=False),
    )
    env = module.PointNavEnv(cfg, num_envs=1, seed_offset=0, total_num_processes=1, worker_info=None)
    try:
        obs, _ = env.reset()
        assert tuple(obs["states"].shape) == (1, 2)
        move = env.chunk_step(np.array([[[1]]], dtype=np.int64))
        assert tuple(move[1].shape) == (1, 1)
        assert tuple(move[2].shape) == (1, 1)
        assert tuple(move[3].shape) == (1, 1)
        assert not bool(move[2][0, 0])
        stopped = env.chunk_step(np.array([[[0]]], dtype=np.int64))
        assert bool(stopped[2][0, 0])
        assert not bool(stopped[3][0, 0])
        assert float(stopped[4][0]["episode"]["success"][0]) == 0.0
        assert int(stopped[4][0]["episode"]["episode_index"][0]) == 50
        frozen = env.chunk_step(np.array([[[2]]], dtype=np.int64))
        assert float(frozen[1][0, 0]) == 0.0
        assert not bool(frozen[2][0, 0] or frozen[3][0, 0])
        assert env.elapsed_steps.item() == 2
        print(
            "train_bridge_ok states=(1,2) reward/term/trunc=(1,1) "
            "stop_failure=true frozen_reward=0"
        )
    finally:
        env.close()

    multi_cfg = Config(**vars(cfg))
    multi_cfg.dataset_split = "val"
    multi_cfg.max_episode_steps = 64
    multi_cfg.episode_indices = [50, 51, 52, 53, 54]
    multi_env = module.PointNavEnv(
        multi_cfg,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )
    try:
        episode_ids = []
        scenes = []
        for expected_index in range(50, 55):
            _, reset_info = multi_env.reset()
            episode_ids.append(reset_info["episode_id"])
            scenes.append(reset_info["scene_id"])
            stopped = multi_env.chunk_step(np.array([[[0]]], dtype=np.int64))
            assert bool(stopped[2][0, 0])
            assert int(stopped[4][0]["episode"]["episode_index"][0]) == expected_index
        assert len(set(episode_ids)) == 5, episode_ids
        assert len(set(scenes)) == 1, scenes
        print("same_scene_eval_ok episode_ids=" + ",".join(episode_ids))
    finally:
        multi_env.close()


if __name__ == "__main__":
    main()
