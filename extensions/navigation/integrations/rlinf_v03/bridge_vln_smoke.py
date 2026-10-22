"""Bounded real Habitat R2R ↔ RLinf VLNEnv bridge smoke on A800.

Run with the private RLinf Python, CUDA hidden on the parent for this smoke.
VLNEnv starts its own private Mesa Habitat child and closes it on exit.
This verifies interface and metrics, not policy inference or PPO training.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


class Config(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val_unseen"), default="train")
    parser.add_argument("--episode-index", type=int, default=4852)
    args = parser.parse_args()
    project = Path("/srv/navigation")
    source = Path(__file__).with_name("vln_env.py")
    spec = importlib.util.spec_from_file_location("a800_vln_env_bridge", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = Config(
        seed=7, auto_reset=False, max_episode_steps=4,
        python_bin=str(project / "envs/habitat-cpu-egl/bin/python"),
        server_script=str(project / "repos/rlinf-nav/scripts/habitat_r2r_service.py"),
        asset_root="/cache/navigation-example", dataset_split=args.split,
        episode_index=args.episode_index, rgb_size=64,
        max_instruction_tokens=200, success_distance_m=3.0,
        habitat_sim_python=str(project / "repos/habitat-sim-cpu-egl/src_python"),
        habitat_env_lib=str(project / "envs/habitat-cpu-egl/lib"),
        egl_vendor_json="/usr/share/glvnd/egl_vendor.d/50_mesa.json",
        video_cfg=Config(save_video=False),
    )
    env = module.VLNEnv(cfg, 1, 0, 1, None)
    process = env.process
    try:
        obs, info = env.reset()
        assert set(obs) == module.OBS_KEYS
        assert tuple(obs["rgb"].shape) == (1, 64, 64, 3)
        assert obs["rgb"].dtype == torch.uint8
        assert tuple(obs["instruction_ids"].shape) == (1, 200)
        assert obs["instruction_ids"].dtype == torch.int64
        assert tuple(obs["instruction_mask"].shape) == (1, 200)
        assert obs["instruction_mask"].dtype == torch.bool
        assert info["episode_id"] == env.identity["episode_id"]
        assert info["scene_id"] == env.identity["scene_id"]
        initial_rgb_sha = hashlib.sha256(obs["rgb"].numpy().tobytes()).hexdigest()
        move = env.chunk_step(np.array([[[1]]], dtype=np.int64))
        assert tuple(move[1].shape) == (1, 1)
        assert tuple(move[2].shape) == (1, 1)
        assert tuple(move[3].shape) == (1, 1)
        assert not bool(move[2][0, 0] or move[3][0, 0])
        moved_rgb_sha = hashlib.sha256(move[0][0]["rgb"].numpy().tobytes()).hexdigest()
        stop = env.chunk_step(np.array([[[0]]], dtype=np.int64))
        assert bool(stop[2][0, 0]) and not bool(stop[3][0, 0])
        metrics = stop[4][0]["episode"]
        assert int(metrics["episode_index"][0]) == args.episode_index
        for key in ("success", "spl", "path_length", "collision_count", "distance_to_goal"):
            assert key in metrics and math.isfinite(float(metrics[key][0]))
        frozen = env.chunk_step(np.array([[[2]]], dtype=np.int64))
        assert float(frozen[1][0, 0]) == 0.0
        assert not bool(frozen[2][0, 0] or frozen[3][0, 0])
        assert int(env.elapsed_steps[0]) == 2
        for key in ("success", "spl", "path_length", "collision_count", "distance_to_goal"):
            assert torch.equal(frozen[4][0]["episode"][key], metrics[key])
        result = {
            "status": "real_r2r_vln_env_bridge_ok",
            "split": args.split,
            "episode_index": args.episode_index,
            "episode_id": info["episode_id"],
            "observation_keys": sorted(obs),
            "rgb_shape": list(obs["rgb"].shape),
            "rgb_dtype": str(obs["rgb"].dtype),
            "instruction_ids_shape": list(obs["instruction_ids"].shape),
            "instruction_ids_dtype": str(obs["instruction_ids"].dtype),
            "instruction_mask_shape": list(obs["instruction_mask"].shape),
            "instruction_mask_dtype": str(obs["instruction_mask"].dtype),
            "rgb_reset_sha256": initial_rgb_sha,
            "rgb_move_sha256": moved_rgb_sha,
            "success": float(metrics["success"][0]),
            "spl": float(metrics["spl"][0]),
            "distance_to_goal": float(metrics["distance_to_goal"][0]),
            "frozen_reward": float(frozen[1][0, 0]),
        }
    finally:
        env.close()
    assert process.poll() is not None
    result["service_exit_code"] = process.returncode
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
