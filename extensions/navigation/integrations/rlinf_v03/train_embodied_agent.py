# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gzip
import importlib
import json
import math
import uuid
from pathlib import Path

import hydra
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.scheduler import Cluster
from rlinf.utils.metric_utils import compute_evaluate_metrics
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.reward.reward_worker import EmbodiedRewardWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

mp.set_start_method("spawn", force=True)


def _worker_cfg_get(value, key, default=None):
    getter = getattr(value, "get", None)
    return getter(key, default) if callable(getter) else getattr(value, key, default)


def preflight_successor_release_anchor(cfg, *, importer=importlib.import_module):
    """Gate successor startup before validation can perform placement work."""

    runner = _worker_cfg_get(cfg, "runner", {})
    if _worker_cfg_get(runner, "worker_pair", "generic") != "p8_successor":
        return None
    adapter_name = _worker_cfg_get(
        runner,
        "p8_successor_adapter_module",
        "rlinf.workers.streamvln_p8_official_adapter",
    )
    adapter = importer(adapter_name)
    check = getattr(adapter, "_require_pinned_release_manifest_sha256", None)
    if not callable(check):
        raise RuntimeError("P8 successor adapter lacks the release-anchor gate")
    return check()


def select_embodied_worker_pair(cfg):
    """Select the successor pair atomically or return the unchanged generic pair."""

    runner = _worker_cfg_get(cfg, "runner", {})
    env_cfg = _worker_cfg_get(cfg, "env", {})
    worker_pair = _worker_cfg_get(runner, "worker_pair", "generic")
    configured_envs = [
        candidate
        for name in ("train", "eval")
        if (candidate := _worker_cfg_get(env_cfg, name, None)) is not None
    ]
    successor_envs = [
        candidate
        for candidate in configured_envs
        if _worker_cfg_get(candidate, "executor_mode", "baseline") == "p8_successor"
    ]
    if worker_pair == "generic":
        if successor_envs:
            raise ValueError(
                "executor_mode=p8_successor requires runner.worker_pair=p8_successor"
            )
        return MultiStepRolloutWorker, EnvWorker
    if worker_pair != "p8_successor":
        raise ValueError("runner.worker_pair must be 'generic' or 'p8_successor'")
    if not configured_envs or len(successor_envs) != len(configured_envs):
        raise ValueError(
            "runner.worker_pair=p8_successor requires every configured env to use "
            "executor_mode=p8_successor"
        )
    if any(_worker_cfg_get(env, "env_type", None) != "streamvln" for env in configured_envs):
        raise ValueError("the P8 successor worker pair requires env_type=streamvln")

    from rlinf.workers.streamvln_p8_successor_workers import (
        P8SuccessorEnvWorker,
        P8SuccessorRolloutWorker,
        require_successor_production_ready,
    )

    require_successor_production_ready(cfg)
    return P8SuccessorRolloutWorker, P8SuccessorEnvWorker


def select_embodied_worker_trio(cfg, generic_actor_worker_cls):
    """Select rollout, environment, and actor workers as one atomic mode."""

    rollout_worker_cls, env_worker_cls = select_embodied_worker_pair(cfg)
    runner = _worker_cfg_get(cfg, "runner", {})
    if _worker_cfg_get(runner, "worker_pair", "generic") == "generic":
        return rollout_worker_cls, env_worker_cls, generic_actor_worker_cls
    if _worker_cfg_get(runner, "use_training_pipeline", False):
        raise ValueError("P8 successor does not support the generic training pipeline")
    from rlinf.workers.streamvln_p8_successor_actor import (
        P8SuccessorActorWorker,
        SUCCESSOR_ADVANTAGE_NAME,
        SUCCESSOR_POLICY_LOSS_NAME,
        SUCCESSOR_REPLAY_SCHEMA,
    )

    if P8SuccessorActorWorker.successor_replay_schema != SUCCESSOR_REPLAY_SCHEMA:
        raise RuntimeError("successor actor worker schema mismatch")
    algorithm = _worker_cfg_get(cfg, "algorithm", {})
    if _worker_cfg_get(algorithm, "adv_type", None) != SUCCESSOR_ADVANTAGE_NAME:
        raise ValueError(
            f"P8 successor requires algorithm.adv_type={SUCCESSOR_ADVANTAGE_NAME}"
        )
    if _worker_cfg_get(algorithm, "loss_type", None) != SUCCESSOR_POLICY_LOSS_NAME:
        raise ValueError(
            f"P8 successor requires algorithm.loss_type={SUCCESSOR_POLICY_LOSS_NAME}"
        )
    bc_cfg = _worker_cfg_get(algorithm, "successor_bc", {})
    if bool(_worker_cfg_get(bc_cfg, "enabled", False)):
        if bool(_worker_cfg_get(runner, "only_eval", False)):
            raise ValueError("P8 successor BC is train-only")
        if _worker_cfg_get(bc_cfg, "expert_source", None) != "train_shortest_path":
            raise ValueError(
                "P8 successor BC requires expert_source=train_shortest_path"
            )
    actor_model = _worker_cfg_get(_worker_cfg_get(cfg, "actor", {}), "model", {})
    if _worker_cfg_get(actor_model, "model_type", None) != "streamvln_p8_successor":
        raise ValueError("P8 successor requires actor.model.model_type=streamvln_p8_successor")
    return rollout_worker_cls, env_worker_cls, P8SuccessorActorWorker


def select_generic_actor_worker(cfg):
    """Preserve the pinned generic actor selection without importing successor code."""

    use_training_pipeline = bool(cfg.runner.get("use_training_pipeline", False))
    if cfg.algorithm.loss_type == "embodied_sac":
        if use_training_pipeline:
            raise ValueError(
                "runner.use_training_pipeline=True is not supported for embodied_sac."
            )
        from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy

        return EmbodiedSACFSDPPolicy
    if cfg.algorithm.loss_type == "embodied_dagger":
        if use_training_pipeline:
            raise ValueError(
                "runner.use_training_pipeline=True is not supported for embodied_dagger."
            )
        from rlinf.workers.actor.fsdp_dagger_policy_worker import (
            EmbodiedDAGGERFSDPPolicy,
        )

        return EmbodiedDAGGERFSDPPolicy
    if cfg.algorithm.loss_type == "embodied_nft":
        if use_training_pipeline:
            raise ValueError(
                "runner.use_training_pipeline=True is not supported for embodied_nft."
            )
        from rlinf.workers.actor.fsdp_nft_policy_worker import EmbodiedNFTFSDPPolicy

        return EmbodiedNFTFSDPPolicy
    if use_training_pipeline:
        from rlinf.workers.actor.fsdp_actor_worker_pipeline import (
            PipelineEmbodiedFSDPActor,
        )

        return PipelineEmbodiedFSDPActor
    from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor

    return EmbodiedFSDPActor


def collect_eval_episode_metrics(cfg, eval_metrics_list):
    """Keep verified per-episode metrics before RLinf reduces them to means."""
    eval_cfg = cfg.env.eval
    env_type = str(eval_cfg.get("env_type", "pointnav"))
    if int(eval_cfg.total_num_envs) != 1:
        raise ValueError("Traceable evaluation requires one environment")
    if env_type == "pointnav":
        if eval_cfg.dataset_split != "val":
            raise ValueError("Traceable PointNav evaluation requires official val")
        expected = [int(index) for index in eval_cfg.episode_indices]
        if not expected or len(expected) != len(set(expected)):
            raise ValueError("Evaluation episode_indices must be nonempty and unique")
        dataset_path = (
            Path(str(eval_cfg.asset_root)).expanduser()
            / "datasets/pointnav/habitat-test-scenes/v1/val/val.json.gz"
        ).resolve(strict=True)
        required_metrics = (
            "success", "spl", "geodesic_distance", "path_length", "collision_count"
        )
    elif env_type == "vln":
        if eval_cfg.dataset_split != "val_unseen":
            raise ValueError("Traceable R2R evaluation requires val_unseen")
        expected = [int(eval_cfg.episode_index)]
        dataset_path = (
            Path(str(eval_cfg.asset_root)).expanduser()
            / "datasets/r2r/val_unseen/val_unseen.json.gz"
        ).resolve(strict=True)
        required_metrics = ("success", "spl", "path_length", "collision_count")
    else:
        raise ValueError(f"Unsupported traceable evaluation environment: {env_type}")
    if int(eval_cfg.rollout_epoch) != len(expected):
        raise ValueError("Evaluation rollout_epoch must equal the number of episode_indices")
    with gzip.open(dataset_path, "rt", encoding="utf-8") as stream:
        official_episodes = json.load(stream)["episodes"]
    if any(index < 0 or index >= len(official_episodes) for index in expected):
        raise ValueError("Evaluation episode_index is outside the official val file")

    records = []
    metric_shards = []
    for shard_number, shard in enumerate(eval_metrics_list):
        if "episode_index" not in shard:
            raise ValueError("Evaluation worker omitted episode_index")
        columns = {
            key: value.detach().cpu().reshape(-1).tolist()
            for key, value in shard.items()
        }
        indices = columns["episode_index"]
        if any(len(values) != len(indices) for values in columns.values()):
            raise ValueError("Evaluation worker metric vectors have different lengths")
        for required in required_metrics:
            if required not in columns:
                raise ValueError(f"Evaluation worker omitted {required}")
        metric_shards.append({key: value for key, value in shard.items() if key != "episode_index"})
        for position, index in enumerate(indices):
            if type(index) is not int or index not in expected:
                raise ValueError(f"Unexpected {eval_cfg.dataset_split} episode_index: {index!r}")
            episode = official_episodes[index]
            episode_id = episode.get("episode_id")
            if episode_id is None:
                raise ValueError(f"{eval_cfg.dataset_split} episode {index} has no episode_id")
            row = {
                "episode_index": index,
                "episode_id": str(episode_id),
                "scene_id": str(episode["scene_id"]),
                "worker_shard": shard_number,
            }
            for key, values in columns.items():
                if key == "episode_index":
                    continue
                value = float(values[position])
                if not math.isfinite(value):
                    raise ValueError(f"Nonfinite evaluation metric {key} for val index {index}")
                row[key] = value
            records.append(row)

    observed = [row["episode_index"] for row in records]
    if sorted(observed) != sorted(expected):
        raise ValueError(
            f"Evaluation did not cover each configured {eval_cfg.dataset_split} index once: {observed}"
        )
    return {
        "status": "ok",
        "env_type": env_type,
        "dataset_split": str(eval_cfg.dataset_split),
        "dataset_path": str(dataset_path),
        "expected_episode_indices": expected,
        "observed_episode_indices": observed,
        "episodes": records,
    }, metric_shards


def write_eval_episode_metrics(cfg, record, global_step):
    """Create one complete, separate JSON artifact per evaluation attempt."""
    output_dir = (
        Path(str(cfg.runner.logger.log_path)).expanduser()
        / str(cfg.runner.logger.experiment_name)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"eval_episodes_step_{global_step}_{uuid.uuid4().hex}.json"
    temporary_path = output_path.with_suffix(".json.tmp")
    with temporary_path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    temporary_path.replace(output_path)
    return output_path


def evaluate_with_episode_metrics(runner, cfg):
    """Use v0.3's evaluation channel sequence and retain raw worker vectors."""
    env_handle = runner.env.evaluate(
        input_channel=runner.env_channel,
        rollout_channel=runner.rollout_channel,
    )
    rollout_handle = runner.rollout.evaluate(
        input_channel=runner.rollout_channel,
        output_channel=runner.env_channel,
    )
    env_results = env_handle.wait()
    rollout_handle.wait()
    eval_metrics_list = [result for result in env_results if result is not None]
    record, metric_shards = collect_eval_episode_metrics(cfg, eval_metrics_list)
    metrics = compute_evaluate_metrics(metric_shards)
    if int(metrics.get("num_trajectories", -1)) != len(record["episodes"]):
        raise ValueError("RLinf evaluation trajectory count disagrees with episode metrics")
    record["global_step"] = int(runner.global_step)
    record["checkpoint_path"] = str(Path(str(cfg.runner.ckpt_path)).expanduser().resolve())
    record["summary"] = {
        key: int(value) if key == "num_trajectories" else float(value)
        for key, value in metrics.items()
    }
    output_path = write_eval_episode_metrics(cfg, record, runner.global_step)
    return metrics, output_path


@hydra.main(
    version_base="1.1", config_path="config", config_name="maniskill_ppo_openvlaoft"
)
def main(cfg) -> None:
    preflight_successor_release_anchor(cfg)
    if cfg.runner.only_eval:
        # In v0.3 the rollout worker has no weight syncer in only_eval mode.
        # Its ckpt_path loader and the actor's ckpt_path loader must both see
        # the same complete model state before evaluate() is called.
        resume_dir = cfg.runner.get("resume_dir", None)
        checkpoint = cfg.runner.get("ckpt_path", None)
        if not resume_dir or not checkpoint:
            raise ValueError(
                "Embodied evaluation requires both runner.resume_dir "
                "and runner.ckpt_path; resume_dir alone restores only the actor"
            )
        checkpoint = Path(str(checkpoint)).expanduser().resolve(strict=True)
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise ValueError(f"Evaluation model checkpoint is missing or empty: {checkpoint}")
        expected = (
            Path(str(resume_dir)).expanduser().resolve(strict=True)
            / "actor/model_state_dict/full_weights.pt"
        )
        if checkpoint != expected:
            raise ValueError(
                "Evaluation ckpt_path must be the full_weights.pt "
                "belonging to runner.resume_dir"
            )
    cfg = validate_cfg(cfg)
    generic_actor_worker_cls = None
    if _worker_cfg_get(cfg.runner, "worker_pair", "generic") == "generic":
        generic_actor_worker_cls = select_generic_actor_worker(cfg)
    rollout_worker_cls, env_worker_cls, actor_worker_cls = select_embodied_worker_trio(
        cfg, generic_actor_worker_cls
    )
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(
        cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.per_worker_log_path
    )
    component_placement = HybridComponentPlacement(cfg, cluster)

    # Create actor worker group
    actor_placement = component_placement.get_strategy("actor")
    actor_group = actor_worker_cls.create_group(cfg).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=actor_placement
    )

    # Create rollout worker group
    rollout_placement = component_placement.get_strategy("rollout")
    rollout_group = rollout_worker_cls.create_group(cfg).launch(
        cluster, name=cfg.rollout.group_name, placement_strategy=rollout_placement
    )

    # Create env worker group
    env_placement = component_placement.get_strategy("env")
    env_group = env_worker_cls.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=env_placement
    )

    reward_group = None
    if cfg.get("reward", {}).get("use_reward_model", False) and not cfg.get(
        "reward", {}
    ).get("standalone_realworld", False):
        # Create reward worker group
        reward_placement = component_placement.get_strategy("reward")
        reward_group = EmbodiedRewardWorker.create_group(cfg).launch(
            cluster, name=cfg.reward.group_name, placement_strategy=reward_placement
        )

    runner = EmbodiedRunner(
        cfg=cfg,
        actor=actor_group,
        rollout=rollout_group,
        env=env_group,
        reward=reward_group,
    )

    runner.init_workers()
    if cfg.runner.only_eval:
        # The v0.3 EmbodiedRunner.run() always enters training, while EnvWorker
        # intentionally has no train env in only_eval mode. Both workers loaded
        # ckpt_path during init; syncing here would call a missing weight syncer.
        try:
            metrics, episode_metrics_path = evaluate_with_episode_metrics(runner, cfg)
            runner.metric_logger.log(
                data={f"eval/{key}": value for key, value in metrics.items()},
                step=runner.global_step,
            )
            print(metrics)
            print(f"Per-episode evaluation: {episode_metrics_path}")
        finally:
            runner._finish_run()
    else:
        runner.run()


if __name__ == "__main__":
    main()
