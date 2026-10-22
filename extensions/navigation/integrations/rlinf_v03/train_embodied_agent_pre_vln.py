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


def collect_eval_episode_metrics(cfg, eval_metrics_list):
    """Keep the per-episode PointNav metrics before RLinf reduces them to means."""
    eval_cfg = cfg.env.eval
    if eval_cfg.dataset_split != "val" or int(eval_cfg.total_num_envs) != 1:
        raise ValueError("Traceable PointNav evaluation requires one official val environment")
    expected = [int(index) for index in eval_cfg.episode_indices]
    if not expected or len(expected) != len(set(expected)):
        raise ValueError("Evaluation episode_indices must be nonempty and unique")
    if int(eval_cfg.rollout_epoch) != len(expected):
        raise ValueError("Evaluation rollout_epoch must equal the number of episode_indices")

    dataset_path = (
        Path(str(eval_cfg.asset_root)).expanduser()
        / "datasets/pointnav/habitat-test-scenes/v1/val/val.json.gz"
    ).resolve(strict=True)
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
        for required in ("success", "spl", "geodesic_distance", "path_length", "collision_count"):
            if required not in columns:
                raise ValueError(f"Evaluation worker omitted {required}")
        metric_shards.append({key: value for key, value in shard.items() if key != "episode_index"})
        for position, index in enumerate(indices):
            if type(index) is not int or index not in expected:
                raise ValueError(f"Unexpected official val episode_index: {index!r}")
            episode = official_episodes[index]
            episode_id = episode.get("episode_id")
            if episode_id is None:
                raise ValueError(f"Official val episode {index} has no episode_id")
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
        raise ValueError(f"Evaluation did not cover each configured val index once: {observed}")
    return {
        "status": "ok",
        "dataset_split": "val",
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
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(
        cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.per_worker_log_path
    )
    component_placement = HybridComponentPlacement(cfg, cluster)

    # Create actor worker group
    actor_placement = component_placement.get_strategy("actor")
    use_training_pipeline = bool(cfg.runner.get("use_training_pipeline", False))

    if cfg.algorithm.loss_type == "embodied_sac":
        if use_training_pipeline:
            raise ValueError(
                "runner.use_training_pipeline=True is not supported for embodied_sac."
            )
        from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy

        actor_worker_cls = EmbodiedSACFSDPPolicy
    elif cfg.algorithm.loss_type == "embodied_dagger":
        if use_training_pipeline:
            raise ValueError(
                "runner.use_training_pipeline=True is not supported for embodied_dagger."
            )
        from rlinf.workers.actor.fsdp_dagger_policy_worker import (
            EmbodiedDAGGERFSDPPolicy,
        )

        actor_worker_cls = EmbodiedDAGGERFSDPPolicy
    elif cfg.algorithm.loss_type == "embodied_nft":
        if use_training_pipeline:
            raise ValueError(
                "runner.use_training_pipeline=True is not supported for embodied_nft."
            )
        from rlinf.workers.actor.fsdp_nft_policy_worker import EmbodiedNFTFSDPPolicy

        actor_worker_cls = EmbodiedNFTFSDPPolicy
    else:
        if use_training_pipeline:
            from rlinf.workers.actor.fsdp_actor_worker_pipeline import (
                PipelineEmbodiedFSDPActor,
            )

            actor_worker_cls = PipelineEmbodiedFSDPActor
        else:
            from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor

            actor_worker_cls = EmbodiedFSDPActor

    actor_group = actor_worker_cls.create_group(cfg).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=actor_placement
    )

    # Create rollout worker group
    rollout_placement = component_placement.get_strategy("rollout")
    rollout_group = MultiStepRolloutWorker.create_group(cfg).launch(
        cluster, name=cfg.rollout.group_name, placement_strategy=rollout_placement
    )

    # Create env worker group
    env_placement = component_placement.get_strategy("env")
    env_group = EnvWorker.create_group(cfg).launch(
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
            print(f"PointNav per-episode evaluation: {episode_metrics_path}")
        finally:
            runner._finish_run()
    else:
        runner.run()


if __name__ == "__main__":
    main()
