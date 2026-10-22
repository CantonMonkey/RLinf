"""Validate PointNav configurations with the actual installed RLinf v0.3."""

import os
from pathlib import Path

from hydra import compose, initialize_config_dir
import ray

from rlinf.config import validate_cfg


config_dir = (
    Path(os.environ.get("RLINF_CHECKOUT", Path(__file__).resolve().parents[4]))
    / "examples"
    / "embodiment"
    / "config"
)
try:
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        for config_name in ("pointnav_ppo_smoke", "pointnav_ppo_eval_64"):
            cfg = validate_cfg(compose(config_name=config_name))
            assert cfg.env.eval.env_type == "pointnav"
            assert cfg.env.train.dataset_split == "train"
            assert cfg.env.eval.dataset_split == "val"
            assert cfg.actor.model.discrete_n_actions == 4
            print(config_name, "validate_cfg_ok")
finally:
    if ray.is_initialized():
        ray.shutdown()
