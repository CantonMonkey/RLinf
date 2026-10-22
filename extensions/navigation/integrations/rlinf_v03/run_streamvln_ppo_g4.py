"""Bounded multi-scene PPO and matched val_unseen comparison for StreamVLN G4."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from safetensors.torch import load_file as load_safetensors

from rlinf.envs import get_env_cls
from rlinf.models import get_model

from run_streamvln_ppo_g3 import (
    G3FSDPManager,
    ORIGINAL_STREAM_COMMIT,
    canonical_lora_state,
    compare_frozen,
    git_head,
    gradient_audit,
    init_distributed,
    jsonable,
    make_manager_cfg as make_g3_manager_cfg,
    make_policy_cfg,
    sampled_frozen_tensors,
    save_checkpoint,
    sha256_file,
    trainable_changes,
    trainable_snapshot,
)
from streamvln_chunk_ppo import collapse_chunk_rewards, variable_duration_gae


class G4FSDPManager(G3FSDPManager):
    """The validated G3 RLinf FSDP manager with a multi-update schedule."""


def scalar(value):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("expected a scalar tensor")
        return value.detach().cpu().item()
    return value


def mean(rows, key):
    return sum(float(row[key]) for row in rows) / len(rows)


def validate_manifest(manifest: dict) -> None:
    if manifest.get("schema_version") != 1:
        raise ValueError("G4 manifest schema is not version 1")
    for split in ("train", "val_unseen"):
        rows = manifest.get(split)
        if not isinstance(rows, list) or len(rows) != 8:
            raise ValueError(f"G4 {split} must contain exactly eight episodes")
        indices = [row.get("index") for row in rows]
        scenes = [row.get("scene") for row in rows]
        if (
            any(type(index) is not int or index < 0 for index in indices)
            or len(set(indices)) != len(indices)
            or any(not isinstance(scene, str) or not scene for scene in scenes)
            or len(set(scenes)) != len(scenes)
        ):
            raise ValueError(f"G4 {split} episode or scene selection is invalid")
    if set(row["scene"] for row in manifest["train"]) & set(
        row["scene"] for row in manifest["val_unseen"]
    ):
        raise ValueError("G4 train and val_unseen scene sets overlap")


def validate_dataset_hashes(run_cfg: dict) -> dict:
    asset_root = Path(run_cfg["asset_root"]).resolve(strict=True)
    actual = {}
    for split in ("train", "val_unseen"):
        path = asset_root / "datasets" / "r2r" / split / f"{split}.json.gz"
        digest = sha256_file(path.resolve(strict=True))
        expected = run_cfg["manifest"]["source"][f"{split}_sha256"]
        if digest != expected:
            raise RuntimeError(f"G4 {split} dataset hash differs from manifest")
        actual[split] = {"path": str(path), "sha256": digest}
    return actual


def make_g4_manager_cfg(run_cfg: dict):
    cfg = make_g3_manager_cfg(run_cfg)
    cfg.optim.total_training_steps = len(run_cfg["manifest"]["train"])
    cfg.optim.lr = float(run_cfg["learning_rate"])
    cfg.optim.value_lr = float(run_cfg["value_learning_rate"])
    return cfg


def make_env(run_cfg: dict, *, split: str, episode_index: int, max_steps: int):
    env_cfg = OmegaConf.create(
        {
            "env_type": "streamvln",
            "seed": run_cfg["seed"],
            "auto_reset": False,
            "max_episode_steps": max_steps,
            "python_bin": run_cfg["habitat_python"],
            "server_script": run_cfg["habitat_server"],
            "asset_root": run_cfg["asset_root"],
            "dataset_split": split,
            "episode_index": int(episode_index),
            "video_cfg": {"save_video": False},
            "success_distance_m": 3.0,
            "habitat_sim_python": run_cfg["habitat_sim_python"],
            "habitat_env_lib": run_cfg["habitat_env_lib"],
            "egl_vendor_json": run_cfg["egl_vendor_json"],
            "rgb_size": 64,
            "max_instruction_tokens": 200,
            "instruction_bytes": 512,
            "scene_bytes": 384,
            "episode_bytes": 64,
            "env_id_base": 0,
            "startup_timeout_s": run_cfg.get("habitat_startup_timeout_s", 60.0),
        }
    )
    env_cls = get_env_cls("streamvln", OmegaConf.create({}))
    return env_cls(env_cfg, 1, 0, 1, None)


def train_episode(manager, core, run_cfg: dict, spec: dict, update_index: int) -> dict:
    env = make_env(
        run_cfg,
        split="train",
        episode_index=spec["index"],
        max_steps=run_cfg["max_eval_steps"],
    )
    decisions = []
    try:
        obs, reset_info = env.reset()
        if str(reset_info["episode_id"]) != str(spec["episode_id"]):
            raise RuntimeError("G4 train episode ID differs from manifest")
        if Path(reset_info["scene_id"]).stem != spec["scene"]:
            raise RuntimeError("G4 train scene differs from manifest")
        for _ in range(run_cfg["max_train_decisions"]):
            with torch.no_grad():
                sampled = manager.model(operation="sample", obs=obs)
            action_mask = sampled["executed_mask"].bool()
            duration = int(action_mask.sum())
            if not 1 <= duration <= 4:
                raise RuntimeError("G4 sampled an invalid action duration")
            actions = sampled["actions"]
            obs_list, rewards, terminations, truncations, infos = env.chunk_step(
                actions.detach().cpu().reshape(1, 4, 1)
            )
            terminal = bool(torch.logical_or(terminations, truncations).any())
            decisions.append(
                {
                    "actions": actions[:duration].detach().cpu().tolist(),
                    "mask": action_mask.detach().cpu().tolist(),
                    "behavior_log_probs": sampled["behavior_log_probs"][action_mask]
                    .detach()
                    .cpu()
                    .tolist(),
                    "primitive_rewards": rewards[0].detach().cpu().tolist(),
                    "duration": duration,
                    "terminal": terminal,
                    "info": jsonable(infos[0]),
                    "behavior_version": int(sampled["behavior_version"]),
                }
            )
            obs = obs_list[0]
            if terminal:
                break

        replay = manager.model(operation="replay")
        state_values = replay["state_values"]
        if int(state_values.numel()) != len(decisions):
            raise RuntimeError("G4 replay value count differs from decisions")
        total_actions = sum(item["duration"] for item in decisions)
        if int(replay["log_probs"].numel()) != total_actions:
            raise RuntimeError("G4 replay action count differs from executed prefix")

        last = decisions[-1]
        with torch.no_grad():
            bootstrap = manager.model(
                operation="bootstrap",
                obs=obs,
                executed_actions=torch.tensor(
                    last["actions"], dtype=torch.int64, device=core.device
                ),
                terminal=last["terminal"],
            )

        current_sums = []
        behavior_sums = []
        entropy_means = []
        cursor = 0
        for item in decisions:
            end = cursor + item["duration"]
            current_sums.append(replay["log_probs"][cursor:end].float().sum())
            behavior_sums.append(
                torch.tensor(
                    item["behavior_log_probs"],
                    dtype=torch.float32,
                    device=core.device,
                ).sum()
            )
            entropy_means.append(replay["entropy"][cursor:end].float().mean())
            cursor = end
        ratios = torch.stack(
            [torch.exp(current - behavior) for current, behavior in zip(current_sums, behavior_sums)]
        )
        ratio_deviation = float((ratios - 1.0).abs().max())
        if ratio_deviation > 0.03:
            raise RuntimeError(f"G4 on-policy ratio deviates by {ratio_deviation}")

        chunk_rewards, durations = collapse_chunk_rewards(
            [item["primitive_rewards"] for item in decisions],
            [item["mask"] for item in decisions],
            run_cfg["gamma"],
        )
        values = [float(value.detach()) for value in state_values]
        next_values = values[1:] + [float(bootstrap.detach())]
        dones = [bool(item["terminal"]) for item in decisions]
        advantages, returns, deltas = variable_duration_gae(
            chunk_rewards,
            values,
            next_values,
            durations,
            dones,
            run_cfg["gamma"],
            run_cfg["gae_lambda"],
        )
        advantage_tensor = torch.tensor(
            advantages, dtype=torch.float32, device=core.device
        )
        return_tensor = torch.tensor(returns, dtype=torch.float32, device=core.device)
        clipped = ratios.clamp(
            1.0 - run_cfg["clip_ratio"], 1.0 + run_cfg["clip_ratio"]
        )
        actor_loss = -torch.minimum(
            ratios * advantage_tensor, clipped * advantage_tensor
        ).mean()
        value_loss = 0.5 * (state_values.float() - return_tensor).square().mean()
        entropy = torch.stack(entropy_means).mean()
        loss = (
            actor_loss
            + run_cfg["value_coef"] * value_loss
            - run_cfg["entropy_coef"] * entropy
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("G4 PPO loss is non-finite")
        manager.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        audit = gradient_audit(core)
        if (
            not audit["finite"]
            or audit["group_norms"]["lora"] <= 0
            or audit["group_norms"]["value_head"] <= 0
            or audit["frozen_with_grad"]
        ):
            raise RuntimeError("G4 gradient isolation audit failed")
        grad_norm, learning_rates = manager.optimizer_step()
        core.mark_updated()

        return {
            "update": update_index,
            "episode_index": spec["index"],
            "episode_id": str(spec["episode_id"]),
            "scene": spec["scene"],
            "source_geodesic_m": spec["source_geodesic_m"],
            "decisions": decisions,
            "decision_count": len(decisions),
            "primitive_step_count": sum(durations),
            "chunk_rewards": chunk_rewards,
            "durations": durations,
            "state_values": values,
            "bootstrap_value": float(bootstrap),
            "deltas": deltas,
            "advantages": advantages,
            "returns": returns,
            "ratios": [float(value.detach()) for value in ratios],
            "max_ratio_deviation": ratio_deviation,
            "loss": {
                "total": float(loss.detach()),
                "actor": float(actor_loss.detach()),
                "value": float(value_loss.detach()),
                "entropy": float(entropy.detach()),
                "grad_norm": float(torch.as_tensor(grad_norm)),
                "learning_rates": [float(value) for value in learning_rates],
            },
            "gradient_audit": audit,
            "actor_version_after": core.actor_version,
        }
    finally:
        env.close()


def run_train(run_cfg: dict) -> dict:
    output = Path(run_cfg["output_dir"])
    torch.manual_seed(run_cfg["seed"])
    torch.cuda.manual_seed_all(run_cfg["seed"])
    init_distributed(run_cfg["master_port"])
    manager = None
    started = time.time()
    record = {}
    try:
        manager = G4FSDPManager(make_g4_manager_cfg(run_cfg), world_size=1, rank=0)
        manager.setup_model_and_optimizer()
        if type(manager.model).__name__ != "FullyShardedDataParallel":
            raise RuntimeError("G4 actor is not wrapped by RLinf FSDP")
        core = manager.model.module
        manifest = core.trainable_manifest()
        if (
            manifest["source_kind"] != "upstream-original"
            or manifest["memory_adapter"]
            or manifest["trainable_parameter_count"] != 364033
        ):
            raise RuntimeError("G4 model source or trainable contract is invalid")
        trainable_before = trainable_snapshot(core)
        frozen_before = sampled_frozen_tensors(core)
        updates = []
        for update_index, spec in enumerate(run_cfg["manifest"]["train"], start=1):
            updates.append(train_episode(manager, core, run_cfg, spec, update_index))
        changes = trainable_changes(core, trainable_before)
        frozen_checks = compare_frozen(core, frozen_before)
        if changes["lora"]["changed_tensors"] < 1 or changes["value_head"]["changed_tensors"] < 1:
            raise RuntimeError("G4 trainable parameters did not change")
        if not all(frozen_checks.values()):
            raise RuntimeError("G4 changed a frozen representative")
        if core.actor_version != len(updates):
            raise RuntimeError("G4 actor version does not equal update count")

        checkpoint = output / f"checkpoint_step_{core.actor_version}"
        metadata = {
            "schema": "streamvln_original_rlinf_ppo_g4_v1",
            "actor_version": core.actor_version,
            "source_kind": "upstream-original",
            "stream_source_commit": run_cfg["stream_commit"],
            "rlinf_source_commit": run_cfg["rlinf_commit"],
            "lora_layers": [24, 25, 26, 27],
            "lora_targets": ["q_proj", "v_proj"],
            "memory_adapter": False,
            "secondary_post_training_used": False,
            "train_episode_indices": [row["index"] for row in run_cfg["manifest"]["train"]],
        }
        hashes = save_checkpoint(core, checkpoint, metadata)
        record = {
            "schema_version": 1,
            "kind": "rlinf-streamvln-original-g4-small-scale-train",
            "source_contract": {
                "stream_commit": run_cfg["stream_commit"],
                "rlinf_commit": run_cfg["rlinf_commit"],
                "base_model": run_cfg["base_model"],
                "source_kind": "upstream-original",
                "memory_adapter": False,
                "secondary_post_training_used": False,
            },
            "fsdp": {
                "manager": type(manager).__name__,
                "wrapper": type(manager.model).__name__,
                "world_size": 1,
                "sharding_strategy": "no_shard",
                "use_orig_params": True,
            },
            "manifest": manifest,
            "dataset_sources": run_cfg["dataset_sources"],
            "updates": updates,
            "update_count": len(updates),
            "train_episode_count": len(updates),
            "train_scene_count": len(set(row["scene"] for row in updates)),
            "train_primitive_steps": sum(row["primitive_step_count"] for row in updates),
            "max_preupdate_ratio_deviation": max(row["max_ratio_deviation"] for row in updates),
            "parameter_changes": changes,
            "frozen_representatives_unchanged": frozen_checks,
            "checkpoint": {"path": str(checkpoint), "files_sha256": hashes},
            "elapsed_s": time.time() - started,
            "all_guards_passed": (
                len(updates) == 8
                and len(set(row["scene"] for row in updates)) == 8
                and max(row["max_ratio_deviation"] for row in updates) <= 0.03
                and all(not row["gradient_audit"]["frozen_with_grad"] for row in updates)
                and all(frozen_checks.values())
            ),
        }
        (output / "train_result.json").write_text(
            json.dumps(jsonable(record), indent=2, sort_keys=True) + "\n"
        )
        if not record["all_guards_passed"]:
            raise RuntimeError("G4 training guards did not all pass")
    finally:
        if manager is not None:
            try:
                manager.optimizer.zero_grad(set_to_none=True)
            except Exception:
                pass
        del manager
        gc.collect()
        torch.cuda.empty_cache()
        if dist.is_initialized():
            dist.destroy_process_group()
        gc.collect()
        torch.cuda.empty_cache()
    return record


def adapter_load_audit(model, checkpoint: Path) -> dict:
    saved = load_safetensors(checkpoint / "adapter" / "adapter_model.safetensors")
    loaded = canonical_lora_state(model.backbone)
    keys_match = set(saved) == set(loaded)
    max_error = float("inf")
    if keys_match:
        max_error = max(
            float((loaded[key].float() - saved[key].float()).abs().max())
            for key in saved
        )
    return {
        "keys_match": keys_match,
        "tensor_count": len(saved),
        "parameter_count": sum(value.numel() for value in saved.values()),
        "max_weight_error": max_error,
        "passed": (
            keys_match
            and len(saved) == 16
            and sum(value.numel() for value in saved.values()) == 360448
            and max_error == 0.0
        ),
    }


def evaluate_episode(model, run_cfg: dict, spec: dict) -> dict:
    env = make_env(
        run_cfg,
        split="val_unseen",
        episode_index=spec["index"],
        max_steps=run_cfg["max_eval_steps"],
    )
    actions = []
    decisions = []
    last_infos = None
    last_terminated = False
    last_truncated = False
    try:
        obs, reset_info = env.reset()
        if str(reset_info["episode_id"]) != str(spec["episode_id"]):
            raise RuntimeError("G4 eval episode ID differs from manifest")
        if Path(reset_info["scene_id"]).stem != spec["scene"]:
            raise RuntimeError("G4 eval scene differs from manifest")
        for decision_index in range(run_cfg["max_eval_steps"] // 4):
            with torch.inference_mode():
                sampled = model(operation="sample", obs=obs)
            mask = sampled["executed_mask"].bool()
            duration = int(mask.sum())
            proposed = sampled["actions"]
            obs_list, rewards, terminations, truncations, infos = env.chunk_step(
                proposed.detach().cpu().reshape(1, 4, 1)
            )
            executed = proposed[:duration].detach().cpu().tolist()
            actions.extend(executed)
            last_terminated = bool(terminations.any())
            last_truncated = bool(truncations.any())
            last_infos = infos[0]
            decisions.append(
                {
                    "decision": decision_index,
                    "actions": executed,
                    "primitive_rewards": rewards[0].detach().cpu().tolist(),
                    "terminated": last_terminated,
                    "truncated": last_truncated,
                }
            )
            obs = obs_list[0]
            if last_terminated or last_truncated:
                break
        if last_infos is None or "episode" not in last_infos:
            raise RuntimeError("G4 evaluation did not produce terminal episode metrics")
        metrics = {key: float(scalar(value)) for key, value in last_infos["episode"].items()}
        success = metrics["success"]
        if success == 1.0:
            outcome = "success"
        elif actions and actions[-1] == 0:
            outcome = "premature_stop"
        elif last_truncated:
            outcome = "timeout"
        else:
            outcome = "other_failure"
        return {
            "episode_index": spec["index"],
            "episode_id": str(spec["episode_id"]),
            "scene": spec["scene"],
            "source_geodesic_m": spec["source_geodesic_m"],
            "success": success,
            "spl": metrics["spl"],
            "navigation_error_m": metrics["distance_to_goal"],
            "path_length_m": metrics["path_length"],
            "collision_count": int(metrics["collision_count"]),
            "primitive_steps": len(actions),
            "decision_count": len(decisions),
            "stop_called": bool(actions and actions[-1] == 0),
            "timeout": last_truncated,
            "outcome": outcome,
            "actions": actions,
            "decisions": decisions,
        }
    finally:
        env.close()


def run_evaluation(run_cfg: dict, variant: str) -> dict:
    output = Path(run_cfg["output_dir"])
    train_result = json.loads((output / "train_result.json").read_text())
    checkpoint = Path(train_result["checkpoint"]["path"])
    eval_cfg = dict(run_cfg)
    eval_cfg["context_tokens"] = run_cfg["eval_context_tokens"]
    adapter = str(checkpoint / "adapter") if variant == "rl" else None
    cfg = make_policy_cfg(eval_cfg, trainable=False, adapter_checkpoint=adapter)
    cfg.deterministic_eval = True
    model = get_model(cfg)
    if variant == "rl":
        value_state = torch.load(
            checkpoint / "value_head.pt", map_location="cpu", weights_only=True
        )
        model.value_head.load_state_dict(value_state, strict=True)
    model.to("cuda:0")
    model.eval()
    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable_count != 0:
        raise RuntimeError("G4 evaluation policy is not frozen")
    adapter_audit = adapter_load_audit(model, checkpoint) if variant == "rl" else None
    if variant == "rl" and not adapter_audit["passed"]:
        raise RuntimeError("G4 RL checkpoint adapter failed exact load audit")
    started = time.time()
    episodes = [
        evaluate_episode(model, run_cfg, spec)
        for spec in run_cfg["manifest"]["val_unseen"]
    ]
    outcomes = {
        key: sum(row["outcome"] == key for row in episodes)
        for key in ("success", "premature_stop", "timeout", "other_failure")
    }
    aggregate = {
        "episode_count": len(episodes),
        "scene_count": len(set(row["scene"] for row in episodes)),
        "success_rate": mean(episodes, "success"),
        "spl": mean(episodes, "spl"),
        "navigation_error_m": mean(episodes, "navigation_error_m"),
        "path_length_m": mean(episodes, "path_length_m"),
        "collision_count": mean(episodes, "collision_count"),
        "primitive_steps": mean(episodes, "primitive_steps"),
        "outcomes": outcomes,
    }
    result = {
        "schema_version": 1,
        "kind": "streamvln-original-g4-matched-val-unseen",
        "variant": variant,
        "deterministic_greedy": True,
        "source_kind": "upstream-original",
        "memory_adapter": False,
        "secondary_post_training_used": False,
        "dataset": run_cfg["dataset_sources"]["val_unseen"],
        "checkpoint": str(checkpoint) if variant == "rl" else None,
        "checkpoint_actor_version": train_result["update_count"] if variant == "rl" else 0,
        "adapter_load_audit": adapter_audit,
        "trainable_parameter_count": trainable_count,
        "episodes": episodes,
        "aggregate": aggregate,
        "elapsed_s": time.time() - started,
        "all_guards_passed": (
            len(episodes) == 8
            and len(set(row["scene"] for row in episodes)) == 8
            and (variant == "frozen" or adapter_audit["passed"])
        ),
    }
    path = output / f"eval_{variant}.json"
    path.write_text(json.dumps(jsonable(result), indent=2, sort_keys=True) + "\n")
    if not result["all_guards_passed"]:
        raise RuntimeError(f"G4 {variant} evaluation guards did not all pass")
    return result


def compare_results(run_cfg: dict, train_record: dict, started: float) -> dict:
    output = Path(run_cfg["output_dir"])
    frozen = json.loads((output / "eval_frozen.json").read_text())
    rl = json.loads((output / "eval_rl.json").read_text())
    frozen_ids = [(row["episode_index"], row["scene"]) for row in frozen["episodes"]]
    rl_ids = [(row["episode_index"], row["scene"]) for row in rl["episodes"]]
    if frozen_ids != rl_ids:
        raise RuntimeError("G4 frozen and RL evaluation episode order differs")
    metrics = ("success_rate", "spl", "navigation_error_m", "path_length_m", "collision_count")
    delta = {
        metric: rl["aggregate"][metric] - frozen["aggregate"][metric]
        for metric in metrics
    }
    result = {
        "schema_version": 1,
        "kind": "streamvln-original-g4-small-scale-comparison",
        "claim_scope": "bounded matched comparison; not convergence or SOTA evidence",
        "source_contract": train_record["source_contract"],
        "dataset_sources": run_cfg["dataset_sources"],
        "train": train_record,
        "frozen": frozen,
        "rl": rl,
        "delta_rl_minus_frozen": delta,
        "matched_episode_order": frozen_ids,
        "elapsed_s_total": time.time() - started,
        "all_guards_passed": (
            train_record["all_guards_passed"]
            and frozen["all_guards_passed"]
            and rl["all_guards_passed"]
            and frozen_ids == rl_ids
        ),
    }
    result_path = output / "result.json"
    result_path.write_text(json.dumps(jsonable(result), indent=2, sort_keys=True) + "\n")
    result["result_sha256"] = sha256_file(result_path)
    if not result["all_guards_passed"]:
        raise RuntimeError("G4 comparison guards did not all pass")
    print(
        json.dumps(
            {
                "all_guards_passed": True,
                "result": str(result_path),
                "result_sha256": result["result_sha256"],
                "frozen": frozen["aggregate"],
                "rl": rl["aggregate"],
                "delta_rl_minus_frozen": delta,
            },
            sort_keys=True,
        )
    )
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("train", "eval"), default="train")
    parser.add_argument("--variant", choices=("frozen", "rl"))
    parser.add_argument("--run-config")
    parser.add_argument("--manifest")
    parser.add_argument("--output-dir")
    parser.add_argument("--rlinf-root")
    parser.add_argument("--stream-source")
    parser.add_argument("--base-model")
    parser.add_argument("--tokenizer")
    parser.add_argument("--vision-tower")
    parser.add_argument("--habitat-python")
    parser.add_argument("--habitat-server")
    parser.add_argument("--asset-root")
    parser.add_argument("--habitat-sim-python")
    parser.add_argument("--habitat-env-lib")
    parser.add_argument(
        "--egl-vendor-json",
        default="/usr/share/glvnd/egl_vendor.d/50_mesa.json",
    )
    parser.add_argument("--train-context-tokens", type=int, default=1024)
    parser.add_argument("--eval-context-tokens", type=int, default=4096)
    parser.add_argument("--max-train-decisions", type=int, default=2)
    parser.add_argument("--max-eval-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--master-port", type=int, default=29585)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.phase == "eval":
        if not args.run_config or not args.variant:
            raise ValueError("eval phase requires --run-config and --variant")
        run_cfg = json.loads(Path(args.run_config).read_text())
        validate_manifest(run_cfg["manifest"])
        run_evaluation(run_cfg, args.variant)
        return

    required = (
        "manifest",
        "output_dir",
        "rlinf_root",
        "stream_source",
        "base_model",
        "tokenizer",
        "vision_tower",
        "habitat_python",
        "habitat_server",
        "asset_root",
        "habitat_sim_python",
        "habitat_env_lib",
    )
    if any(getattr(args, key) is None for key in required):
        raise ValueError("train phase is missing required path arguments")
    if args.max_train_decisions not in (1, 2) or args.max_eval_steps % 4:
        raise ValueError("G4 train decisions or eval step alignment is invalid")
    if args.train_context_tokens < 512 or args.eval_context_tokens < 2048:
        raise ValueError("G4 context budgets are too small")
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("G4 output directory must be fresh")
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        key: str(Path(getattr(args, key)).expanduser().resolve(strict=True))
        for key in (
            "manifest",
            "rlinf_root",
            "stream_source",
            "base_model",
            "tokenizer",
            "vision_tower",
            "habitat_python",
            "habitat_server",
            "asset_root",
            "habitat_sim_python",
            "habitat_env_lib",
            "egl_vendor_json",
        )
    }
    manifest = json.loads(Path(paths["manifest"]).read_text())
    validate_manifest(manifest)
    stream_commit = git_head(Path(paths["stream_source"]))
    if stream_commit != ORIGINAL_STREAM_COMMIT:
        raise RuntimeError("G4 requires the pinned original StreamVLN commit")
    rlinf_commit = git_head(Path(paths["rlinf_root"]))
    run_cfg = {
        **paths,
        "output_dir": str(output),
        "manifest": manifest,
        "stream_commit": stream_commit,
        "rlinf_commit": rlinf_commit,
        "context_tokens": args.train_context_tokens,
        "train_context_tokens": args.train_context_tokens,
        "eval_context_tokens": args.eval_context_tokens,
        "max_train_decisions": args.max_train_decisions,
        "max_eval_steps": args.max_eval_steps,
        "seed": args.seed,
        "master_port": args.master_port,
        "learning_rate": args.learning_rate,
        "value_learning_rate": args.learning_rate,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_ratio": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.001,
    }
    run_cfg["dataset_sources"] = validate_dataset_hashes(run_cfg)
    run_config_path = output / "run_config.json"
    run_config_path.write_text(json.dumps(run_cfg, indent=2, sort_keys=True) + "\n")
    started = time.time()
    train_record = run_train(run_cfg)
    for variant in ("frozen", "rl"):
        subprocess.run(
            [
                sys.executable,
                "-B",
                str(Path(__file__).resolve()),
                "--phase",
                "eval",
                "--variant",
                variant,
                "--run-config",
                str(run_config_path),
            ],
            check=True,
            env=os.environ.copy(),
        )
    compare_results(run_cfg, train_record, started)


if __name__ == "__main__":
    main()
