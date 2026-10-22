"""Bounded G2 StreamVLN evaluation through RLinf's MultiStepRolloutWorker path.

This entrypoint deliberately creates no actor, optimizer, FSDP group or PPO
runner.  It exercises RLinf's model/env registries, exact rollout predict
method, four-action environment chunks, fixed replay rows and independent
restore while the frozen StreamVLN model lives in its validated environment.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from rlinf.config import SupportedModel
from rlinf.envs import get_env_cls
from rlinf.models import get_model
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


EXPECTED_OBS_KEYS = {
    "rgb",
    "instruction_utf8",
    "instruction_utf8_mask",
    "scene_utf8",
    "scene_utf8_mask",
    "episode_utf8",
    "episode_utf8_mask",
    "env_id",
    "episode_serial",
    "episode_step",
    "episode_start",
    "previous_actions",
    "previous_action_mask",
    "terminal_frozen",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=15,
    )
    if result.returncode:
        raise RuntimeError(f"cannot resolve Git HEAD for {path}")
    return result.stdout.strip()


def tensor_json(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: tensor_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensor_json(item) for item in value]
    return value


def reorder_batch(batch: dict[str, torch.Tensor], order: torch.Tensor):
    return {key: value.index_select(0, order) for key, value in batch.items()}


def compare_replay(sampled, replayed, mask) -> float:
    difference = (sampled - replayed).abs()
    valid = difference[mask]
    return float(valid.max()) if valid.numel() else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rlinf-root", required=True)
    parser.add_argument("--stream-python", required=True)
    parser.add_argument("--stream-source", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--vision-tower", required=True)
    parser.add_argument("--policy-server", required=True)
    parser.add_argument("--habitat-python", required=True)
    parser.add_argument("--habitat-server", required=True)
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--habitat-sim-python", required=True)
    parser.add_argument("--habitat-env-lib", required=True)
    parser.add_argument(
        "--egl-vendor-json",
        default="/usr/share/glvnd/egl_vendor.d/50_mesa.json",
    )
    parser.add_argument("--episode-indices", default="1189,1190")
    parser.add_argument("--split", default="train", choices=("train", "val_unseen"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--max-episode-steps", type=int, default=40)
    parser.add_argument("--max-decisions", type=int, default=2)
    parser.add_argument("--mode", choices=("train", "eval"), default="train")
    parser.add_argument("--replay-guard", type=float, default=0.03)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.max_decisions < 1:
        raise ValueError("max-decisions must be positive")
    episodes = [int(value) for value in args.episode_indices.split(",")]
    if len(episodes) < 2 or len(set(episodes)) != len(episodes):
        raise ValueError("G2 requires at least two distinct episode indices")
    if args.max_episode_steps % 4:
        raise ValueError("max-episode-steps must align to four-action chunks")

    roots = {
        key: Path(value).expanduser().resolve(strict=True)
        for key, value in {
            "rlinf_root": args.rlinf_root,
            "stream_python": args.stream_python,
            "stream_source": args.stream_source,
            "base_model": args.base_model,
            "tokenizer": args.tokenizer,
            "vision_tower": args.vision_tower,
            "policy_server": args.policy_server,
            "habitat_python": args.habitat_python,
            "habitat_server": args.habitat_server,
            "asset_root": args.asset_root,
            "habitat_sim_python": args.habitat_sim_python,
            "habitat_env_lib": args.habitat_env_lib,
            "egl_vendor_json": args.egl_vendor_json,
        }.items()
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    model_cfg = OmegaConf.create({
        "model_type": "streamvln_categorical",
        "model_path": "",
        "stream_python": str(roots["stream_python"]),
        "stream_source": str(roots["stream_source"]),
        "server_script": str(roots["policy_server"]),
        "base_model": str(roots["base_model"]),
        "tokenizer": str(roots["tokenizer"]),
        "vision_tower": str(roots["vision_tower"]),
        "expected_source_commit": git_head(roots["stream_source"]),
        "actor_device": args.device,
        "image_size": args.image_size,
        "instruction_bytes": 512,
        "scene_bytes": 384,
        "episode_bytes": 64,
        "action_dim": 1,
        "num_action_chunks": 4,
        "discrete_n_actions": 4,
        "temperature": 1.0,
        "context_tokens": 4096,
        "seed": 20260905,
        "startup_timeout_s": 600,
        "request_timeout_s": 240,
        "precision": "32",
        "add_value_head": False,
        "is_lora": False,
        "lora_rank": 32,
        "load_to_device": False,
    })
    if SupportedModel(model_cfg.model_type).value != "streamvln_categorical":
        raise RuntimeError("StreamVLN model registration is not active")

    env_cls = get_env_cls("streamvln", OmegaConf.create({}))
    policy = None
    envs = []
    record = {
        "schema_version": 1,
        "kind": "rlinf-streamvln-g2-frozen-eval",
        "rlinf_head": git_head(roots["rlinf_root"]),
        "streamvln_head": git_head(roots["stream_source"]),
        "episodes": episodes,
        "mode": args.mode,
        "max_decisions": args.max_decisions,
        "observation_keys": sorted(EXPECTED_OBS_KEYS),
        "decisions": [],
        "replay_guard": args.replay_guard,
        "ppo_started": False,
    }
    try:
        policy = get_model(model_cfg)
        record["actor_handshake"] = dict(policy.handshake)
        worker = object.__new__(MultiStepRolloutWorker)
        worker.model_cfg = model_cfg
        worker.algorithm_cfg = OmegaConf.create({"loss_type": "actor_critic"})
        worker._train_sampling_params = {}
        worker._eval_sampling_params = {}
        worker.hf_model = policy
        worker.expert_model = None

        predict_impl = inspect.unwrap(MultiStepRolloutWorker.predict)
        if predict_impl is MultiStepRolloutWorker.predict:
            raise RuntimeError("RLinf Worker.timer did not preserve the wrapped predict method")
        predict_path = Path(inspect.getsourcefile(predict_impl)).resolve(strict=True)
        record["rlinf_predict_source"] = {
            "path": str(predict_path),
            "sha256": sha256_file(predict_path),
            "qualname": predict_impl.__qualname__,
        }

        for env_id, episode_index in enumerate(episodes):
            env_cfg = OmegaConf.create({
                "env_type": "streamvln",
                "seed": 7,
                "auto_reset": False,
                "max_episode_steps": args.max_episode_steps,
                "python_bin": str(roots["habitat_python"]),
                "server_script": str(roots["habitat_server"]),
                "asset_root": str(roots["asset_root"]),
                "dataset_split": args.split,
                "episode_index": episode_index,
                "video_cfg": {"save_video": False},
                "success_distance_m": 3.0,
                "habitat_sim_python": str(roots["habitat_sim_python"]),
                "habitat_env_lib": str(roots["habitat_env_lib"]),
                "egl_vendor_json": str(roots["egl_vendor_json"]),
                "rgb_size": args.image_size,
                "max_instruction_tokens": 200,
                "instruction_bytes": 512,
                "scene_bytes": 384,
                "episode_bytes": 64,
                "env_id_base": 0,
            })
            envs.append(env_cls(env_cfg, 1, env_id, len(episodes), None))

        observations = []
        active = []
        for env_id, env in enumerate(envs):
            obs, _ = env.reset()
            if set(obs) != EXPECTED_OBS_KEYS:
                raise RuntimeError("StreamVLN environment exposed an unexpected observation")
            observations.append(obs)
            active.append(env_id)

        max_error = 0.0
        for decision_index in range(args.max_decisions):
            merged = MultiStepRolloutWorker._merge_obs_batches(
                [{"obs": obs} for obs in observations]
            )["obs"]
            actions, sampled = predict_impl(worker, merged, mode=args.mode)
            replayed = policy.default_forward(
                forward_inputs=sampled["forward_inputs"],
                compute_logprobs=True,
                compute_entropy=True,
                compute_values=True,
            )
            mask = sampled["forward_inputs"]["current_action_mask"]
            error = compare_replay(sampled["prev_logprobs"], replayed["logprobs"], mask)
            max_error = max(max_error, error)

            order = torch.arange(len(active) - 1, -1, -1, dtype=torch.int64)
            shuffled_inputs = reorder_batch(sampled["forward_inputs"], order)
            shuffled = policy.default_forward(
                forward_inputs=shuffled_inputs,
                compute_logprobs=True,
                compute_entropy=True,
                compute_values=False,
            )
            restore_error = compare_replay(
                replayed["logprobs"].index_select(0, order),
                shuffled["logprobs"],
                mask.index_select(0, order),
            )
            max_error = max(max_error, restore_error)

            repeat_actions, repeat_sampled = predict_impl(worker, merged, mode=args.mode)
            if not torch.equal(actions, repeat_actions):
                raise RuntimeError("replay changed the live StreamVLN session actions")
            idempotent_error = compare_replay(
                sampled["prev_logprobs"],
                repeat_sampled["prev_logprobs"],
                mask,
            )
            max_error = max(max_error, idempotent_error)

            decision_record = {
                "decision_index": decision_index,
                "active_env_ids": list(active),
                "episode_steps": tensor_json(merged["episode_step"]),
                "actions": tensor_json(actions[..., 0]),
                "action_mask": tensor_json(mask),
                "sample_replay_max_error": error,
                "shuffled_restore_max_error": restore_error,
                "idempotent_repeat_max_error": idempotent_error,
                "env_results": [],
            }
            next_observations = []
            next_active = []
            for row, env_id in enumerate(active):
                obs_list, rewards, terminations, truncations, infos = envs[env_id].chunk_step(
                    actions[row : row + 1]
                )
                ended = bool(torch.logical_or(terminations, truncations).any())
                decision_record["env_results"].append({
                    "env_id": env_id,
                    "episode_index": episodes[env_id],
                    "rewards": tensor_json(rewards),
                    "terminations": tensor_json(terminations),
                    "truncations": tensor_json(truncations),
                    "info": tensor_json(infos[0]),
                })
                if not ended:
                    next_active.append(env_id)
                    next_observations.append(obs_list[0])
            record["decisions"].append(decision_record)
            active, observations = next_active, next_observations
            if not active:
                break

        record["max_replay_error"] = max_error
        record["replay_guard_passed"] = max_error <= args.replay_guard
        record["distinct_env_ids_observed"] = sorted({
            env_id
            for decision in record["decisions"]
            for env_id in decision["active_env_ids"]
        })
        if len(record["distinct_env_ids_observed"]) < 2:
            raise RuntimeError("real batched rollout did not observe two independent environments")
        if not record["replay_guard_passed"]:
            raise RuntimeError(
                f"sample/replay error {max_error} exceeds guard {args.replay_guard}"
            )
        output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        print(json.dumps({
            "output": str(output),
            "decisions": len(record["decisions"]),
            "max_replay_error": max_error,
            "replay_guard_passed": record["replay_guard_passed"],
            "distinct_env_ids_observed": record["distinct_env_ids_observed"],
        }, sort_keys=True))
    finally:
        for env in envs:
            env.close()
        if policy is not None:
            policy.close()


if __name__ == "__main__":
    main()
