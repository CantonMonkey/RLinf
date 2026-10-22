"""Compose the isolated RLinf v0.3 PointNav PPO configuration."""

import os
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


config_dir = (
    Path(os.environ.get("RLINF_CHECKOUT", Path(__file__).resolve().parents[4]))
    / "examples"
    / "embodiment"
    / "config"
)
with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
    cfg = compose(config_name="pointnav_ppo_smoke")
    OmegaConf.resolve(cfg)
    eval_cfg = compose(config_name="pointnav_ppo_eval_64")
    OmegaConf.resolve(eval_cfg)

assert cfg.env.train.env_type == cfg.env.eval.env_type == "pointnav"
assert cfg.env.train.dataset_split == "train"
assert cfg.env.train.episode_indices == [50, 51]
assert cfg.env.eval.dataset_split == "val"
assert cfg.env.eval.episode_index == 50
assert cfg.actor.model.discrete_n_actions == 4
assert cfg.actor.model.action_dim == cfg.actor.model.num_action_chunks == 1
assert cfg.env.train.max_steps_per_rollout_epoch == cfg.env.train.max_episode_steps
assert eval_cfg.runner.only_eval
assert eval_cfg.env.eval.episode_indices == [50, 51, 52, 53, 54]
assert eval_cfg.env.eval.max_episode_steps == 64
assert eval_cfg.env.eval.rollout_epoch == 5
assert eval_cfg.rollout.model.discrete_n_actions == 4
print("pointnav_config_ok", cfg.cluster.component_placement, cfg.actor.model.model_type)
