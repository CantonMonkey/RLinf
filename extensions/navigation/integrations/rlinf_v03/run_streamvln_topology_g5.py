"""G5 topological waypoint warm-start and high-level PPO comparison."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors

from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager

from run_streamvln_ppo_g3 import init_distributed
from run_streamvln_ppo_g4 import (
    make_env,
    mean,
    validate_dataset_hashes,
    validate_manifest,
)
from streamvln_chunk_ppo import collapse_chunk_rewards, variable_duration_gae
from streamvln_topological_policy import get_model as get_topological_model
from streamvln_topology import (
    FEATURE_NAMES,
    MODE_NAMES,
    NAVIGATE_MODE,
    OnlineTopology,
    STOP_MODE,
    compile_waypoint,
    executed_prefix,
)


ORIGINAL_STREAM_COMMIT = "e48f6ff7e9201d93aae003e8f64b04c00cec13bc"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head(path: Path) -> str:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={path}", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=20,
    )
    if result.returncode:
        raise RuntimeError(f"cannot resolve git head for {path}: {result.stderr}")
    return result.stdout.strip()


def jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def stable_seed(*parts) -> int:
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def make_policy_cfg(run_cfg: dict, *, trainable: bool, head_checkpoint=None):
    return OmegaConf.create(
        {
            "model_type": "streamvln_ppo",
            "model_path": "",
            "stream_source": run_cfg["stream_source"],
            "base_model": run_cfg["base_model"],
            "tokenizer": run_cfg["tokenizer"],
            "vision_tower": run_cfg["vision_tower"],
            "expected_source_commit": run_cfg["stream_commit"],
            "head_checkpoint": head_checkpoint,
            "trainable": trainable,
            "context_tokens": run_cfg["context_tokens"],
            "seed": run_cfg["seed"],
            "topology_dim": len(FEATURE_NAMES),
            "head_hidden": run_cfg["head_hidden"],
            "precision": "bf16",
            "add_value_head": True,
            "is_lora": False,
            "load_to_device": False,
        }
    )


class G5FSDPManager(FSDPModelManager):
    def model_provider_func(self):
        return get_topological_model(self._cfg.model)


def make_manager_cfg(run_cfg: dict, *, total_training_steps: int):
    return OmegaConf.create(
        {
            "model": make_policy_cfg(run_cfg, trainable=True),
            "optim": {
                "lr": run_cfg["learning_rate"],
                "value_lr": run_cfg["value_learning_rate"],
                "adam_beta1": 0.9,
                "adam_beta2": 0.999,
                "adam_eps": 1e-8,
                "weight_decay": 0.0,
                "clip_grad": 1.0,
                "total_training_steps": int(total_training_steps),
                "lr_warmup_steps": 0,
                "lr_scheduler": "constant",
            },
            "fsdp_config": {
                "strategy": "fsdp",
                "sharding_strategy": "no_shard",
                "gradient_checkpointing": False,
                "cpu_offload": False,
                "offload_pin_memory": False,
                "reshard_after_forward": False,
                "enable_gradient_accumulation": False,
                "forward_prefetch": False,
                "limit_all_gathers": False,
                "backward_prefetch": None,
                "use_orig_params": True,
                "use_liger_kernel": False,
                "wrap_policy": {"disable": True},
                "mixed_precision": {
                    "param_dtype": None,
                    "reduce_dtype": None,
                    "buffer_dtype": None,
                },
                "amp_autocast": {"enabled": False, "precision": "bf16"},
                "grad_scaler": {"enabled": False},
            },
        }
    )


def model_device(core) -> torch.device:
    return next(core.parameters()).device


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(list(tensor.shape)).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def parameter_snapshot(core):
    return {name: tensor.detach().cpu().clone() for name, tensor in core.high_level_state().items()}


def parameter_changes(core, before):
    after = core.high_level_state()
    rows = {}
    for name in sorted(before):
        delta = (after[name].float() - before[name].float()).abs()
        rows[name] = {"changed": bool(torch.any(delta > 0)), "max_abs": float(delta.max())}
    return {
        "changed_tensor_count": sum(row["changed"] for row in rows.values()),
        "tensor_count": len(rows),
        "max_abs": max(row["max_abs"] for row in rows.values()),
        "tensors": rows,
    }


def gradient_audit(core):
    sums = {"high_level_policy": 0.0, "value_head": 0.0}
    nonzero = {"high_level_policy": 0, "value_head": 0}
    finite = True
    frozen_with_grad = []
    for name, parameter in core.named_parameters():
        grad = parameter.grad
        if parameter.requires_grad:
            group = "value_head" if "value_head" in name else "high_level_policy"
            if grad is not None:
                finite = finite and bool(torch.isfinite(grad).all())
                norm = float(grad.detach().float().norm())
                sums[group] += norm
                nonzero[group] += int(norm > 0)
        elif grad is not None and bool(torch.any(grad != 0)):
            frozen_with_grad.append(name)
    return {
        "finite": finite,
        "group_norm_sums": sums,
        "nonzero_tensors": nonzero,
        "frozen_with_grad": frozen_with_grad,
    }


def topology_tensor(topology, device):
    return torch.tensor(topology.feature_vector(), dtype=torch.float32, device=device)


def collision_trace(info, duration):
    trace = info.get("chunk_collisions")
    if not isinstance(trace, list) or len(trace) != duration or any(type(x) is not bool for x in trace):
        raise RuntimeError("G5 executor collision trace differs from executed prefix")
    return trace


def execute_high_level(env, topology, mode, waypoint):
    proposed = compile_waypoint(waypoint, mode, topology)
    obs_list, rewards, terminations, truncations, infos = env.chunk_step(
        torch.tensor(proposed, dtype=torch.int64).reshape(1, 4, 1)
    )
    executed = executed_prefix(proposed)
    trace = collision_trace(infos[0], len(executed))
    feedback = infos[0].get("executor_feedback")
    if feedback is None:
        raise RuntimeError("G7.16 executor feedback is missing")
    if feedback.get("executed_actions") != executed:
        raise RuntimeError("G7.16 executor feedback differs from executed prefix")
    topology.update_from_feedback(feedback)
    return {
        "obs": obs_list[0],
        "rewards": rewards[0].detach().cpu().tolist(),
        "info": infos[0],
        "proposed_actions": proposed,
        "executed_actions": executed,
        "collisions": trace,
        "duration": len(executed),
        "terminal": bool(torch.logical_or(terminations, truncations).any()),
        "terminated": bool(terminations.any()),
        "truncated": bool(truncations.any()),
    }


def collect_supervision(manager, core, run_cfg):
    device = model_device(core)
    rows, episodes = [], []
    for spec in run_cfg["manifest"]["train"]:
        env = make_env(
            run_cfg,
            split="train",
            episode_index=spec["index"],
            max_steps=run_cfg["max_bc_decisions"] * 4,
        )
        topology = OnlineTopology(max_decisions=run_cfg["max_bc_decisions"])
        record = {"episode_index": spec["index"], "episode_id": str(spec["episode_id"]), "scene": spec["scene"], "labels": []}
        try:
            obs, identity = env.reset()
            if str(identity["episode_id"]) != str(spec["episode_id"]) or Path(identity["scene_id"]).stem != spec["scene"]:
                raise RuntimeError("G5 supervised episode identity differs from manifest")
            for decision in range(run_cfg["max_bc_decisions"]):
                label = env.expert_waypoint(run_cfg["expert_lookahead_m"])
                with torch.no_grad():
                    state = manager.model(operation="encode_state", obs=obs)
                top = topology_tensor(topology, device)
                mode = STOP_MODE if label["stop"] else NAVIGATE_MODE
                target = [float(value) for value in label["waypoint_local_m"]]
                rows.append({"state": state.cpu().float(), "topology": top.cpu().float(), "mode": mode, "waypoint": target})
                step = execute_high_level(env, topology, mode, target)
                record["labels"].append(
                    {
                        "decision": decision,
                        "mode": MODE_NAMES[mode],
                        "waypoint_local_m": target,
                        "distance_to_goal_m": label["distance_to_goal_m"],
                        "actions": step["executed_actions"],
                        "collisions": step["collisions"],
                    }
                )
                obs = step["obs"]
                if step["terminal"]:
                    break
            record["topology"] = topology.summary()
            episodes.append(record)
        finally:
            env.close()
    stop_count = sum(row["mode"] == STOP_MODE for row in rows)
    if len(rows) < len(run_cfg["manifest"]["train"]) or stop_count < 1:
        raise RuntimeError("G5 supervised dataset has insufficient examples or STOP labels")
    tensors = {
        "state_features": torch.stack([row["state"] for row in rows]),
        "topology_features": torch.stack([row["topology"] for row in rows]),
        "target_modes": torch.tensor([row["mode"] for row in rows], dtype=torch.int64),
        "target_waypoints": torch.tensor([row["waypoint"] for row in rows], dtype=torch.float32),
    }
    summary = {
        "sample_count": len(rows),
        "episode_count": len(episodes),
        "scene_count": len({row["scene"] for row in episodes}),
        "navigate_count": len(rows) - stop_count,
        "stop_count": stop_count,
        "recover_count": 0,
        "expert_source": "train-shortest-path",
        "expert_lookahead_m": run_cfg["expert_lookahead_m"],
        "policy_observation_excludes_expert": True,
        "episodes": episodes,
    }
    return tensors, summary


def train_behavior_cloning(manager, core, run_cfg, dataset):
    device = model_device(core)
    count = int(dataset["target_modes"].shape[0])
    batch_size = min(run_cfg["bc_batch_size"], count)
    history, before = [], parameter_snapshot(core)
    for epoch in range(run_cfg["bc_epochs"]):
        generator = torch.Generator(device="cpu").manual_seed(run_cfg["seed"] + epoch)
        order, batches = torch.randperm(count, generator=generator), []
        for start in range(0, count, batch_size):
            indices = order[start:start + batch_size]
            batch = {key: value[indices].to(device) for key, value in dataset.items()}
            result = manager.model(operation="bc_loss", **batch)
            if not bool(torch.isfinite(result["loss"])):
                raise RuntimeError("G5 behavior-cloning loss is nonfinite")
            manager.optimizer.zero_grad(set_to_none=True)
            result["loss"].backward()
            audit = gradient_audit(core)
            if not audit["finite"] or audit["group_norm_sums"]["high_level_policy"] <= 0 or audit["frozen_with_grad"]:
                raise RuntimeError("G5 behavior-cloning gradient isolation failed")
            grad_norm, learning_rates = manager.optimizer_step()
            core.mark_updated()
            batches.append(
                {
                    **{key: float(value.detach()) for key, value in result.items()},
                    "grad_norm": float(torch.as_tensor(grad_norm)),
                    "learning_rates": [float(x) for x in learning_rates],
                    "gradient_audit": audit,
                }
            )
        history.append(
            {
                "epoch": epoch + 1,
                "loss": sum(x["loss"] for x in batches) / len(batches),
                "mode_loss": sum(x["mode_loss"] for x in batches) / len(batches),
                "waypoint_loss": sum(x["waypoint_loss"] for x in batches) / len(batches),
                "mode_accuracy": sum(x["mode_accuracy"] for x in batches) / len(batches),
                "batches": batches,
            }
        )
    changes = parameter_changes(core, before)
    if changes["changed_tensor_count"] < 1:
        raise RuntimeError("G5 behavior cloning changed no high-level parameters")
    return {
        "epochs": run_cfg["bc_epochs"],
        "batch_size": batch_size,
        "history": history,
        "initial_loss": history[0]["loss"],
        "final_loss": history[-1]["loss"],
        "final_mode_accuracy": history[-1]["mode_accuracy"],
        "parameter_changes": changes,
    }


def ppo_episode(manager, core, run_cfg, spec, variant):
    device = model_device(core)
    env = make_env(
        run_cfg,
        split="train",
        episode_index=spec["index"],
        max_steps=run_cfg["max_rl_decisions"] * 4,
    )
    topology = OnlineTopology(max_decisions=run_cfg["max_rl_decisions"])
    decisions = []
    try:
        obs, identity = env.reset()
        if str(identity["episode_id"]) != str(spec["episode_id"]) or Path(identity["scene_id"]).stem != spec["scene"]:
            raise RuntimeError("G5 PPO episode identity differs from manifest")
        for decision_index in range(run_cfg["max_rl_decisions"]):
            with torch.no_grad():
                state = manager.model(operation="encode_state", obs=obs)
                top = topology_tensor(topology, device)
                sampled = manager.model(
                    operation="sample_head",
                    state_features=state,
                    topology_features=top,
                    seed=stable_seed(run_cfg["seed"], variant, spec["episode_id"], decision_index),
                    deterministic=False,
                )
            mode = int(sampled["modes"].reshape(-1)[0])
            waypoint = sampled["waypoints"].reshape(-1, 2)[0].cpu().tolist()
            step = execute_high_level(env, topology, mode, waypoint)
            decisions.append(
                {
                    "state": state.detach(),
                    "topology": top.detach(),
                    "mode": mode,
                    "waypoint": waypoint,
                    "behavior_log_prob": float(sampled["behavior_log_probs"].reshape(-1)[0]),
                    "primitive_rewards": step["rewards"],
                    "duration": step["duration"],
                    "actions": step["executed_actions"],
                    "collisions": step["collisions"],
                    "terminal": step["terminal"],
                    "info": jsonable(step["info"]),
                }
            )
            obs = step["obs"]
            if step["terminal"]:
                break
        if not decisions:
            raise RuntimeError("G5 PPO episode produced no decisions")
        # Match the batch-one sampling numerics for the BF16 head. Each replay
        # remains differentiable; concatenation forms the episode PPO loss.
        replay_rows = [
            manager.model(
                operation="replay_head",
                state_features=row["state"],
                topology_features=row["topology"],
                modes=torch.tensor([row["mode"]], dtype=torch.int64, device=device),
                waypoints=torch.tensor([row["waypoint"]], dtype=torch.float32, device=device),
            )
            for row in decisions
        ]
        replay = {
            key: torch.cat([item[key] for item in replay_rows], dim=0)
            for key in ("log_probs", "entropy", "values", "waypoint_mean", "mode_logits")
        }
        behavior = torch.tensor([row["behavior_log_prob"] for row in decisions], dtype=torch.float32, device=device)
        ratios = torch.exp(replay["log_probs"].float() - behavior)
        ratio_deviation = float((ratios - 1.0).abs().max())
        if ratio_deviation > 0.03:
            raise RuntimeError(f"G5 on-policy ratio deviates by {ratio_deviation}")
        if decisions[-1]["terminal"]:
            bootstrap = 0.0
        else:
            with torch.no_grad():
                next_state = manager.model(operation="encode_state", obs=obs)
                next_topology = topology_tensor(topology, device)
                row = manager.model(
                    operation="replay_head",
                    state_features=next_state,
                    topology_features=next_topology,
                    modes=torch.tensor([NAVIGATE_MODE], device=device),
                    waypoints=torch.zeros((1, 2), device=device),
                )
                bootstrap = float(row["values"].reshape(-1)[0])
        masks = [[index < row["duration"] for index in range(4)] for row in decisions]
        chunk_rewards, durations = collapse_chunk_rewards(
            [row["primitive_rewards"] for row in decisions], masks, run_cfg["gamma"]
        )
        values = [float(value) for value in replay["values"].detach()]
        advantages, returns, deltas = variable_duration_gae(
            chunk_rewards,
            values,
            values[1:] + [bootstrap],
            durations,
            [row["terminal"] for row in decisions],
            run_cfg["gamma"],
            run_cfg["gae_lambda"],
        )
        advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)
        clipped = ratios.clamp(1.0 - run_cfg["clip_ratio"], 1.0 + run_cfg["clip_ratio"])
        actor_loss = -torch.minimum(ratios * advantages, clipped * advantages).mean()
        value_loss = 0.5 * (replay["values"].float() - returns_tensor).square().mean()
        entropy = replay["entropy"].float().mean()
        loss = actor_loss + run_cfg["value_coef"] * value_loss - run_cfg["entropy_coef"] * entropy
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("G5 PPO loss is nonfinite")
        manager.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        audit = gradient_audit(core)
        if (
            not audit["finite"]
            or audit["group_norm_sums"]["high_level_policy"] <= 0
            or audit["group_norm_sums"]["value_head"] <= 0
            or audit["frozen_with_grad"]
        ):
            raise RuntimeError("G5 PPO gradient isolation failed")
        grad_norm, learning_rates = manager.optimizer_step()
        core.mark_updated()
        return {
            "episode_index": spec["index"],
            "episode_id": str(spec["episode_id"]),
            "scene": spec["scene"],
            "decision_count": len(decisions),
            "primitive_step_count": sum(durations),
            "decisions": [
                {key: jsonable(value) for key, value in row.items() if key not in ("state", "topology")}
                for row in decisions
            ],
            "chunk_rewards": chunk_rewards,
            "durations": durations,
            "values": values,
            "bootstrap": bootstrap,
            "deltas": deltas,
            "advantages": advantages.detach().cpu().tolist(),
            "returns": returns,
            "ratios": ratios.detach().cpu().tolist(),
            "max_ratio_deviation": ratio_deviation,
            "loss": {
                "total": float(loss.detach()),
                "actor": float(actor_loss.detach()),
                "value": float(value_loss.detach()),
                "entropy": float(entropy.detach()),
                "grad_norm": float(torch.as_tensor(grad_norm)),
                "learning_rates": [float(x) for x in learning_rates],
            },
            "gradient_audit": audit,
            "topology": topology.summary(),
            "actor_version_after": core.actor_version,
        }
    finally:
        env.close()


def train_ppo(manager, core, run_cfg, variant):
    before = parameter_snapshot(core)
    updates = [ppo_episode(manager, core, run_cfg, spec, variant) for spec in run_cfg["manifest"]["train"]]
    changes = parameter_changes(core, before)
    if changes["changed_tensor_count"] < 1:
        raise RuntimeError(f"G5 {variant} PPO changed no trainable parameters")
    return {
        "variant": variant,
        "update_count": len(updates),
        "episode_count": len(updates),
        "scene_count": len({row["scene"] for row in updates}),
        "primitive_steps": sum(row["primitive_step_count"] for row in updates),
        "max_preupdate_ratio_deviation": max(row["max_ratio_deviation"] for row in updates),
        "updates": updates,
        "parameter_changes": changes,
    }


def save_head_checkpoint(core, directory, metadata):
    directory.mkdir(parents=True, exist_ok=False)
    state = core.high_level_state()
    policy_path = directory / "policy.safetensors"
    save_safetensors(state, policy_path)
    persisted = load_safetensors(policy_path)
    if set(persisted) != set(state):
        raise RuntimeError("G5 persisted checkpoint keys changed")
    max_error = max(float((persisted[name].float() - state[name].float()).abs().max()) for name in state)
    if max_error != 0.0:
        raise RuntimeError("G5 persisted checkpoint differs from live weights")
    metadata_path = directory / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return {
        "directory": str(directory),
        "policy": str(policy_path),
        "tensor_count": len(state),
        "parameter_count": sum(tensor.numel() for tensor in state.values()),
        "state_sha256": state_sha256(state),
        "policy_sha256": sha256_file(policy_path),
        "metadata_sha256": sha256_file(metadata_path),
        "max_persist_error": max_error,
    }


def cleanup_manager(manager):
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


def setup_manager(run_cfg, total_steps):
    torch.manual_seed(run_cfg["seed"])
    torch.cuda.manual_seed_all(run_cfg["seed"])
    init_distributed(run_cfg["master_port"])
    manager = G5FSDPManager(make_manager_cfg(run_cfg, total_training_steps=total_steps), world_size=1, rank=0)
    manager.setup_model_and_optimizer()
    if type(manager.model).__name__ != "FullyShardedDataParallel":
        raise RuntimeError("G5 actor is not wrapped by RLinf FSDP")
    core = manager.model.module
    manifest = core.trainable_manifest()
    if (
        manifest["source_kind"] != "upstream-original"
        or manifest["memory_adapter"]
        or manifest["secondary_post_training_used"]
        or manifest["backbone_trainable"]
        or not 500_000 <= manifest["trainable_parameter_count"] <= 2_000_000
    ):
        raise RuntimeError("G5 model source or trainable scope is invalid")
    return manager, core, manifest


def run_bc_rl_train(run_cfg):
    output = Path(run_cfg["output_dir"])
    upper = run_cfg["bc_epochs"] * math.ceil(
        len(run_cfg["manifest"]["train"]) * run_cfg["max_bc_decisions"] / run_cfg["bc_batch_size"]
    )
    manager, started = None, time.time()
    try:
        manager, core, manifest = setup_manager(run_cfg, upper + 8)
        initial_sha = state_sha256(parameter_snapshot(core))
        dataset, supervision = collect_supervision(manager, core, run_cfg)
        bc = train_behavior_cloning(manager, core, run_cfg, dataset)
        bc_checkpoint = save_head_checkpoint(
            core,
            output / "checkpoint_bc",
            {
                "schema": "streamvln_topological_g5_v1",
                "variant": "bc_only",
                "stream_commit": run_cfg["stream_commit"],
                "rlinf_commit": run_cfg["rlinf_commit"],
                "val_unseen_used": False,
                "memory_adapter": False,
                "secondary_post_training_used": False,
            },
        )
        ppo = train_ppo(manager, core, run_cfg, "bc_rl")
        bc_rl_checkpoint = save_head_checkpoint(
            core,
            output / "checkpoint_bc_rl",
            {
                "schema": "streamvln_topological_g5_v1",
                "variant": "bc_rl",
                "stream_commit": run_cfg["stream_commit"],
                "rlinf_commit": run_cfg["rlinf_commit"],
                "val_unseen_used": False,
                "memory_adapter": False,
                "secondary_post_training_used": False,
            },
        )
        record = {
            "schema_version": 1,
            "kind": "streamvln-g5-supervised-warmstart-plus-ppo",
            "source_contract": {
                "stream_commit": run_cfg["stream_commit"],
                "rlinf_commit": run_cfg["rlinf_commit"],
                "source_kind": "upstream-original",
                "memory_adapter": False,
                "secondary_post_training_used": False,
            },
            "manifest": manifest,
            "initial_head_sha256": initial_sha,
            "supervision": supervision,
            "behavior_cloning": bc,
            "ppo": ppo,
            "checkpoints": {"bc_only": bc_checkpoint, "bc_rl": bc_rl_checkpoint},
            "elapsed_s": time.time() - started,
        }
        record["all_guards_passed"] = (
            supervision["episode_count"] == 8
            and supervision["scene_count"] == 8
            and supervision["stop_count"] >= 1
            and ppo["update_count"] == 8
            and ppo["max_preupdate_ratio_deviation"] <= 0.03
        )
        if not record["all_guards_passed"]:
            raise RuntimeError("G5 BC+RL training guards did not all pass")
        (output / "train_bc_rl.json").write_text(json.dumps(jsonable(record), indent=2, sort_keys=True) + "\n")
        return record
    finally:
        cleanup_manager(manager)


def run_rl_only_train(run_cfg):
    output = Path(run_cfg["output_dir"])
    reference = json.loads((output / "train_bc_rl.json").read_text())
    manager, started = None, time.time()
    try:
        manager, core, manifest = setup_manager(run_cfg, 8)
        initial_sha = state_sha256(parameter_snapshot(core))
        if initial_sha != reference["initial_head_sha256"]:
            raise RuntimeError("G5 RL-only and BC+RL did not start from identical heads")
        ppo = train_ppo(manager, core, run_cfg, "rl_only")
        checkpoint = save_head_checkpoint(
            core,
            output / "checkpoint_rl_only",
            {
                "schema": "streamvln_topological_g5_v1",
                "variant": "rl_only",
                "stream_commit": run_cfg["stream_commit"],
                "rlinf_commit": run_cfg["rlinf_commit"],
                "val_unseen_used": False,
                "memory_adapter": False,
                "secondary_post_training_used": False,
            },
        )
        record = {
            "schema_version": 1,
            "kind": "streamvln-g5-rl-without-warmstart",
            "manifest": manifest,
            "initial_head_sha256": initial_sha,
            "ppo": ppo,
            "checkpoint": checkpoint,
            "elapsed_s": time.time() - started,
            "all_guards_passed": ppo["update_count"] == 8 and ppo["max_preupdate_ratio_deviation"] <= 0.03,
        }
        if not record["all_guards_passed"]:
            raise RuntimeError("G5 RL-only training guards did not all pass")
        (output / "train_rl_only.json").write_text(json.dumps(jsonable(record), indent=2, sort_keys=True) + "\n")
        return record
    finally:
        cleanup_manager(manager)


def checkpoint_load_audit(model, checkpoint_path):
    saved, loaded = load_safetensors(checkpoint_path), model.high_level_state()
    keys_match = set(saved) == set(loaded)
    max_error = float("inf")
    if keys_match:
        max_error = max(float((saved[name].float() - loaded[name].float()).abs().max()) for name in saved)
    return {
        "keys_match": keys_match,
        "tensor_count": len(saved),
        "parameter_count": sum(tensor.numel() for tensor in saved.values()),
        "max_weight_error": max_error,
        "passed": keys_match and max_error == 0.0,
    }


def evaluate_episode(model, run_cfg, spec, variant):
    env = make_env(
        run_cfg,
        split="val_unseen",
        episode_index=spec["index"],
        max_steps=run_cfg["max_eval_steps"],
    )
    topology = OnlineTopology(max_decisions=run_cfg["max_eval_steps"] // 4)
    actions, decisions, last_info = [], [], None
    last_truncated = False
    try:
        obs, identity = env.reset()
        if str(identity["episode_id"]) != str(spec["episode_id"]) or Path(identity["scene_id"]).stem != spec["scene"]:
            raise RuntimeError("G5 eval episode identity differs from manifest")
        for decision_index in range(run_cfg["max_eval_steps"] // 4):
            with torch.inference_mode():
                state = model(operation="encode_state", obs=obs)
                top = topology_tensor(topology, model_device(model))
                sampled = model(
                    operation="sample_head",
                    state_features=state,
                    topology_features=top,
                    seed=stable_seed(run_cfg["seed"], variant, spec["episode_id"], decision_index),
                    deterministic=True,
                )
            mode = int(sampled["modes"].reshape(-1)[0])
            waypoint = sampled["waypoints"].reshape(-1, 2)[0].cpu().tolist()
            step = execute_high_level(env, topology, mode, waypoint)
            actions.extend(step["executed_actions"])
            last_info, last_truncated = step["info"], step["truncated"]
            decisions.append(
                {
                    "decision": decision_index,
                    "mode": MODE_NAMES[mode],
                    "waypoint_local_m": waypoint,
                    "mode_logits": sampled["mode_logits"].reshape(-1, 3)[0].cpu().tolist(),
                    "waypoint_mean": sampled["waypoint_mean"].reshape(-1, 2)[0].cpu().tolist(),
                    "actions": step["executed_actions"],
                    "collisions": step["collisions"],
                    "primitive_rewards": step["rewards"],
                    "terminated": step["terminated"],
                    "truncated": step["truncated"],
                    "topology": topology.summary(),
                }
            )
            obs = step["obs"]
            if step["terminal"]:
                break
        if last_info is None or "episode" not in last_info:
            raise RuntimeError("G5 evaluation did not produce terminal episode metrics")
        metrics = {key: float(value.item()) for key, value in last_info["episode"].items()}
        if metrics["success"] == 1.0:
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
            "success": metrics["success"],
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
            "topology": topology.summary(),
        }
    finally:
        env.close()


def run_evaluation(run_cfg, variant):
    output = Path(run_cfg["output_dir"])
    checkpoint = {
        "bc_only": output / "checkpoint_bc/policy.safetensors",
        "bc_rl": output / "checkpoint_bc_rl/policy.safetensors",
        "rl_only": output / "checkpoint_rl_only/policy.safetensors",
    }[variant].resolve(strict=True)
    model = get_topological_model(
        make_policy_cfg(run_cfg, trainable=False, head_checkpoint=str(checkpoint))
    )
    model.to("cuda:0")
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("G5 evaluation model is not frozen")
    audit = checkpoint_load_audit(model, checkpoint)
    if not audit["passed"]:
        raise RuntimeError("G5 checkpoint exact-load audit failed")
    started = time.time()
    episodes = [evaluate_episode(model, run_cfg, spec, variant) for spec in run_cfg["manifest"]["val_unseen"]]
    outcomes = {
        key: sum(row["outcome"] == key for row in episodes)
        for key in ("success", "premature_stop", "timeout", "other_failure")
    }
    aggregate = {
        "episode_count": len(episodes),
        "scene_count": len({row["scene"] for row in episodes}),
        "success_rate": mean(episodes, "success"),
        "spl": mean(episodes, "spl"),
        "navigation_error_m": mean(episodes, "navigation_error_m"),
        "path_length_m": mean(episodes, "path_length_m"),
        "collision_count": mean(episodes, "collision_count"),
        "primitive_steps": mean(episodes, "primitive_steps"),
        "topology_nodes": sum(row["topology"]["node_count"] for row in episodes) / len(episodes),
        "outcomes": outcomes,
    }
    result = {
        "schema_version": 1,
        "kind": "streamvln-g5-fixed-val-unseen",
        "variant": variant,
        "deterministic_greedy": True,
        "source_kind": "upstream-original",
        "memory_adapter": False,
        "secondary_post_training_used": False,
        "expert_interface_used": False,
        "checkpoint": str(checkpoint),
        "checkpoint_load_audit": audit,
        "episodes": episodes,
        "aggregate": aggregate,
        "elapsed_s": time.time() - started,
    }
    result["all_guards_passed"] = len(episodes) == 8 and len({row["scene"] for row in episodes}) == 8 and audit["passed"]
    if not result["all_guards_passed"]:
        raise RuntimeError(f"G5 {variant} evaluation guards did not all pass")
    (output / f"eval_{variant}.json").write_text(json.dumps(jsonable(result), indent=2, sort_keys=True) + "\n")
    return result


def compare_results(run_cfg, started):
    output = Path(run_cfg["output_dir"])
    variants = {name: json.loads((output / f"eval_{name}.json").read_text()) for name in ("rl_only", "bc_only", "bc_rl")}
    orders = {
        name: [(row["episode_index"], row["scene"]) for row in result["episodes"]]
        for name, result in variants.items()
    }
    if len({tuple(value) for value in orders.values()}) != 1:
        raise RuntimeError("G5 evaluation episode order differs across variants")
    expected = [(row["index"], row["scene"]) for row in run_cfg["manifest"]["val_unseen"]]
    if next(iter(orders.values())) != expected:
        raise RuntimeError("G5 evaluation order differs from the fixed manifest")
    g4_path = Path(run_cfg["g4_frozen_result"]).resolve(strict=True)
    if sha256_file(g4_path) != run_cfg["g4_frozen_sha256"]:
        raise RuntimeError("G5 external G4 frozen anchor hash changed")
    g4 = json.loads(g4_path.read_text())
    if [(row["episode_index"], row["scene"]) for row in g4["episodes"]] != expected or g4["variant"] != "frozen":
        raise RuntimeError("G5 external G4 anchor does not match the fixed episodes")
    metrics = ("success_rate", "spl", "navigation_error_m", "path_length_m", "collision_count", "primitive_steps")
    delta = {
        "bc_rl_minus_rl_only": {
            metric: variants["bc_rl"]["aggregate"][metric] - variants["rl_only"]["aggregate"][metric]
            for metric in metrics
        },
        "bc_rl_minus_bc_only": {
            metric: variants["bc_rl"]["aggregate"][metric] - variants["bc_only"]["aggregate"][metric]
            for metric in metrics
        },
    }
    action_changes = {
        pair: sum(left["actions"] != right["actions"] for left, right in zip(variants[a]["episodes"], variants[b]["episodes"]))
        for pair, a, b in (
            ("bc_rl_vs_rl_only", "bc_rl", "rl_only"),
            ("bc_rl_vs_bc_only", "bc_rl", "bc_only"),
            ("bc_only_vs_rl_only", "bc_only", "rl_only"),
        )
    }
    result = {
        "schema_version": 1,
        "kind": "streamvln-g5-topological-waypoint-comparison",
        "claim_scope": "bounded matched G5 comparison; G4 frozen is an external action-space anchor, not a causal executor-matched baseline",
        "source_contract": {
            "stream_commit": run_cfg["stream_commit"],
            "rlinf_commit": run_cfg["rlinf_commit"],
            "source_kind": "upstream-original",
            "memory_adapter": False,
            "secondary_post_training_used": False,
        },
        "matched_episode_order": expected,
        "g5_variants": variants,
        "g4_frozen_anchor": {
            "path": str(g4_path),
            "sha256": run_cfg["g4_frozen_sha256"],
            "aggregate": g4["aggregate"],
            "executor_matched_to_g5": False,
        },
        "matched_delta": delta,
        "paired_action_sequence_changes": action_changes,
        "elapsed_s_total": time.time() - started,
        "all_guards_passed": all(value["all_guards_passed"] for value in variants.values()),
    }
    path = output / "result.json"
    path.write_text(json.dumps(jsonable(result), indent=2, sort_keys=True) + "\n")
    print(json.dumps(
        {
            "all_guards_passed": result["all_guards_passed"],
            "result": str(path),
            "result_sha256": sha256_file(path),
            "aggregates": {name: value["aggregate"] for name, value in variants.items()},
            "g4_frozen_anchor": g4["aggregate"],
            "matched_delta": delta,
            "paired_action_sequence_changes": action_changes,
        },
        sort_keys=True,
    ))
    return result


def run_subprocess(config_path, phase, variant=None):
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--phase", phase, "--run-config", str(config_path)]
    if variant:
        command.extend(["--variant", variant])
    subprocess.run(command, check=True, env=os.environ.copy())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("orchestrate", "bc_rl", "rl_only", "eval"), default="orchestrate")
    parser.add_argument("--variant", choices=("rl_only", "bc_only", "bc_rl"))
    parser.add_argument("--run-config")
    for name in (
        "manifest",
        "g4-frozen-result",
        "g4-frozen-sha256",
        "output-dir",
        "rlinf-root",
        "stream-source",
        "base-model",
        "tokenizer",
        "vision-tower",
        "habitat-python",
        "habitat-server",
        "asset-root",
        "habitat-sim-python",
        "habitat-env-lib",
    ):
        parser.add_argument("--" + name)
    parser.add_argument("--egl-vendor-json", default="/usr/share/glvnd/egl_vendor.d/50_mesa.json")
    parser.add_argument("--context-tokens", type=int, default=4096)
    parser.add_argument("--head-hidden", type=int, default=256)
    parser.add_argument("--bc-epochs", type=int, default=8)
    parser.add_argument("--bc-batch-size", type=int, default=16)
    parser.add_argument("--max-bc-decisions", type=int, default=12)
    parser.add_argument("--max-rl-decisions", type=int, default=10)
    parser.add_argument("--max-eval-steps", type=int, default=40)
    parser.add_argument("--expert-lookahead-m", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--master-port", type=int, default=29586)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--value-learning-rate", type=float, default=3e-4)
    return parser.parse_args()


def load_run_config(path):
    run_cfg = json.loads(Path(path).read_text())
    validate_manifest(run_cfg["manifest"])
    return run_cfg


def main():
    args = parse_args()
    if args.phase != "orchestrate":
        if not args.run_config:
            raise ValueError("G5 subprocess phase requires --run-config")
        run_cfg = load_run_config(args.run_config)
        if args.phase == "bc_rl":
            run_bc_rl_train(run_cfg)
        elif args.phase == "rl_only":
            run_rl_only_train(run_cfg)
        elif args.phase == "eval":
            if not args.variant:
                raise ValueError("G5 eval phase requires --variant")
            run_evaluation(run_cfg, args.variant)
        return
    required = (
        "manifest", "g4_frozen_result", "g4_frozen_sha256", "output_dir",
        "rlinf_root", "stream_source", "base_model", "tokenizer",
        "vision_tower", "habitat_python", "habitat_server", "asset_root",
        "habitat_sim_python", "habitat_env_lib",
    )
    if any(getattr(args, key) is None for key in required):
        raise ValueError("G5 orchestration is missing required path arguments")
    if (
        args.max_eval_steps % 4
        or args.max_bc_decisions < 2
        or args.max_rl_decisions < 2
        or args.bc_epochs < 1
        or args.bc_batch_size < 1
        or args.context_tokens < 2048
    ):
        raise ValueError("G5 bounded training/evaluation budget is invalid")
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("G5 output directory must be fresh")
    output.mkdir(parents=True, exist_ok=True)
    path_keys = (
        "manifest", "g4_frozen_result", "rlinf_root", "stream_source",
        "base_model", "tokenizer", "vision_tower", "habitat_python",
        "habitat_server", "asset_root", "habitat_sim_python",
        "habitat_env_lib", "egl_vendor_json",
    )
    paths = {key: str(Path(getattr(args, key)).expanduser().resolve(strict=True)) for key in path_keys}
    manifest = json.loads(Path(paths["manifest"]).read_text())
    validate_manifest(manifest)
    stream_commit = git_head(Path(paths["stream_source"]))
    if stream_commit != ORIGINAL_STREAM_COMMIT:
        raise RuntimeError("G5 requires the pinned original StreamVLN commit")
    run_cfg = {
        **paths,
        "output_dir": str(output),
        "manifest": manifest,
        "stream_commit": stream_commit,
        "rlinf_commit": git_head(Path(paths["rlinf_root"])),
        "g4_frozen_sha256": args.g4_frozen_sha256,
        "context_tokens": args.context_tokens,
        "head_hidden": args.head_hidden,
        "bc_epochs": args.bc_epochs,
        "bc_batch_size": args.bc_batch_size,
        "max_bc_decisions": args.max_bc_decisions,
        "max_rl_decisions": args.max_rl_decisions,
        "max_eval_steps": args.max_eval_steps,
        "expert_lookahead_m": args.expert_lookahead_m,
        "seed": args.seed,
        "master_port": args.master_port,
        "learning_rate": args.learning_rate,
        "value_learning_rate": args.value_learning_rate,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_ratio": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.001,
    }
    run_cfg["dataset_sources"] = validate_dataset_hashes(run_cfg)
    if sha256_file(Path(paths["g4_frozen_result"])) != args.g4_frozen_sha256:
        raise RuntimeError("G5 provided G4 frozen anchor hash differs")
    config_path = output / "run_config.json"
    config_path.write_text(json.dumps(run_cfg, indent=2, sort_keys=True) + "\n")
    started = time.time()
    # Every FSDP phase owns its CUDA context and distributed rendezvous.
    # Process exit guarantees the 8B frozen backbone is released before the
    # next phase loads it.
    run_subprocess(config_path, "bc_rl")
    run_subprocess(config_path, "rl_only")
    for variant in ("rl_only", "bc_only", "bc_rl"):
        run_subprocess(config_path, "eval", variant)
    compare_results(run_cfg, started)


if __name__ == "__main__":
    main()
