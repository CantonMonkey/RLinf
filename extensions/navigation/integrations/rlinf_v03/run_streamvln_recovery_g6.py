"""G6 collision recovery, balanced warm-start, and batched PPO experiment."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file as load_safetensors

from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager

from run_streamvln_ppo_g4 import make_env, mean, validate_dataset_hashes, validate_manifest
from run_streamvln_topology_g5 import (
    ORIGINAL_STREAM_COMMIT,
    checkpoint_load_audit,
    cleanup_manager,
    collision_trace,
    git_head,
    gradient_audit,
    jsonable,
    make_manager_cfg,
    model_device,
    parameter_changes,
    parameter_snapshot,
    save_head_checkpoint,
    sha256_file,
    stable_seed,
    state_sha256,
    topology_tensor,
)
from streamvln_chunk_ppo import collapse_chunk_rewards, variable_duration_gae
from streamvln_recovery import (
    compile_recovery_aware,
    recovery_required,
    recovery_target,
    resolved_choice,
    waypoint_execution_error,
)
from streamvln_recovery_policy import get_model as get_recovery_model
from streamvln_topology import (
    FEATURE_NAMES,
    MODE_NAMES,
    NAVIGATE_MODE,
    OnlineTopology,
    RECOVER_MODE,
    STOP_MODE,
    executed_prefix,
)


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


class G6FSDPManager(FSDPModelManager):
    def model_provider_func(self):
        return get_recovery_model(self._cfg.model)


def setup_manager(run_cfg: dict, total_steps: int):
    from run_streamvln_ppo_g3 import init_distributed

    torch.manual_seed(run_cfg["seed"])
    torch.cuda.manual_seed_all(run_cfg["seed"])
    init_distributed(run_cfg["master_port"])
    cfg = make_manager_cfg(run_cfg, total_training_steps=total_steps)
    cfg.model = make_policy_cfg(run_cfg, trainable=True)
    manager = G6FSDPManager(cfg, world_size=1, rank=0)
    manager.setup_model_and_optimizer()
    if type(manager.model).__name__ != "FullyShardedDataParallel":
        raise RuntimeError("G6 actor is not wrapped by RLinf FSDP")
    core = manager.model.module
    manifest = core.trainable_manifest()
    if (
        manifest["source_kind"] != "upstream-original"
        or manifest["memory_adapter"]
        or manifest["secondary_post_training_used"]
        or manifest["backbone_trainable"]
        or not 500_000 <= manifest["trainable_parameter_count"] <= 2_000_000
    ):
        raise RuntimeError("G6 model source or trainable scope is invalid")
    return manager, core, manifest


def execute_choice(env, topology, mode, waypoint):
    target = [float(value) for value in waypoint]
    start_pose = (topology.x, topology.z, topology.heading)
    proposed = compile_recovery_aware(target, mode, topology)
    obs_list, rewards, terminations, truncations, infos = env.chunk_step(
        torch.tensor(proposed, dtype=torch.int64).reshape(1, 4, 1)
    )
    executed = executed_prefix(proposed)
    trace = collision_trace(infos[0], len(executed))
    topology.update(executed, trace)
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
        "waypoint_execution": waypoint_execution_error(target, start_pose, topology),
    }


def validate_identity(identity, spec, phase):
    if (
        str(identity["episode_id"]) != str(spec["episode_id"])
        or Path(identity["scene_id"]).stem != spec["scene"]
    ):
        raise RuntimeError(f"G6 {phase} episode identity differs from manifest")


def collect_supervision(manager, core, run_cfg):
    device = model_device(core)
    rows, episodes = [], []
    for pass_index in range(run_cfg["collection_passes"]):
        recovery_enabled = pass_index > 0
        for spec in run_cfg["manifest"]["train"]:
            env = make_env(
                run_cfg,
                split="train",
                episode_index=spec["index"],
                max_steps=run_cfg["max_bc_decisions"] * 4,
            )
            topology = OnlineTopology(max_decisions=run_cfg["max_bc_decisions"])
            record = {
                "pass": pass_index,
                "recovery_enabled": recovery_enabled,
                "episode_index": spec["index"],
                "episode_id": str(spec["episode_id"]),
                "scene": spec["scene"],
                "labels": [],
            }
            try:
                obs, identity = env.reset()
                validate_identity(identity, spec, "supervision")
                for decision in range(run_cfg["max_bc_decisions"]):
                    label = env.expert_waypoint(run_cfg["expert_lookahead_m"])
                    with torch.no_grad():
                        state = manager.model(operation="encode_state", obs=obs)
                    top = topology_tensor(topology, device)
                    if label["stop"]:
                        mode = STOP_MODE
                        target = [float(value) for value in label["waypoint_local_m"]]
                    elif recovery_enabled and recovery_required(
                        topology, blocked_threshold=run_cfg["blocked_threshold"]
                    ):
                        mode = RECOVER_MODE
                        target = list(recovery_target(topology))
                    else:
                        mode = NAVIGATE_MODE
                        target = [float(value) for value in label["waypoint_local_m"]]
                    rows.append(
                        {
                            "state": state.cpu().float(),
                            "topology": top.cpu().float(),
                            "mode": mode,
                            "waypoint": target,
                        }
                    )
                    step = execute_choice(env, topology, mode, target)
                    record["labels"].append(
                        {
                            "decision": decision,
                            "mode": MODE_NAMES[mode],
                            "waypoint_local_m": target,
                            "distance_to_goal_m": label["distance_to_goal_m"],
                            "actions": step["executed_actions"],
                            "collisions": step["collisions"],
                            "waypoint_execution": step["waypoint_execution"],
                        }
                    )
                    obs = step["obs"]
                    if step["terminal"]:
                        break
                record["topology"] = topology.summary()
                episodes.append(record)
            finally:
                env.close()

    raw_counts = Counter(row["mode"] for row in rows)
    if any(raw_counts[mode] < 1 for mode in (NAVIGATE_MODE, STOP_MODE, RECOVER_MODE)):
        raise RuntimeError(f"G6 train-only collection lacks a navigation mode: {dict(raw_counts)}")
    grouped = {mode: [row for row in rows if row["mode"] == mode] for mode in range(3)}
    balanced_count = max(raw_counts.values())
    balanced = []
    for mode in range(3):
        source = grouped[mode]
        balanced.extend(source[index % len(source)] for index in range(balanced_count))
    generator = torch.Generator(device="cpu").manual_seed(run_cfg["seed"] + 611)
    order = torch.randperm(len(balanced), generator=generator).tolist()
    balanced = [balanced[index] for index in order]
    tensors = {
        "state_features": torch.stack([row["state"] for row in balanced]),
        "topology_features": torch.stack([row["topology"] for row in balanced]),
        "target_modes": torch.tensor([row["mode"] for row in balanced], dtype=torch.int64),
        "target_waypoints": torch.tensor([row["waypoint"] for row in balanced], dtype=torch.float32),
    }
    balanced_counts = Counter(row["mode"] for row in balanced)
    summary = {
        "raw_sample_count": len(rows),
        "balanced_sample_count": len(balanced),
        "episode_rollout_count": len(episodes),
        "source_episode_count": len(run_cfg["manifest"]["train"]),
        "scene_count": len({row["scene"] for row in episodes}),
        "collection_passes": run_cfg["collection_passes"],
        "raw_mode_counts": {MODE_NAMES[key]: raw_counts[key] for key in range(3)},
        "balanced_mode_counts": {MODE_NAMES[key]: balanced_counts[key] for key in range(3)},
        "expert_source": "train-shortest-path-only",
        "expert_lookahead_m": run_cfg["expert_lookahead_m"],
        "recover_source": "online-collision-and-parent-topology",
        "policy_observation_excludes_expert": True,
        "val_unseen_used": False,
        "episodes": episodes,
    }
    return tensors, summary


def train_behavior_cloning(manager, core, run_cfg, dataset):
    device = model_device(core)
    count = int(dataset["target_modes"].shape[0])
    batch_size = min(run_cfg["bc_batch_size"], count)
    history, before = [], parameter_snapshot(core)
    metric_names = (
        "loss",
        "mode_loss",
        "waypoint_loss",
        "mode_accuracy",
        "mode_accuracy_navigate",
        "mode_accuracy_stop",
        "mode_accuracy_recover",
    )
    for epoch in range(run_cfg["bc_epochs"]):
        generator = torch.Generator(device="cpu").manual_seed(run_cfg["seed"] + epoch)
        order, batches = torch.randperm(count, generator=generator), []
        for start in range(0, count, batch_size):
            indices = order[start : start + batch_size]
            batch = {key: value[indices].to(device) for key, value in dataset.items()}
            result = manager.model(operation="bc_loss", **batch)
            if not bool(torch.isfinite(result["loss"])):
                raise RuntimeError("G6 behavior-cloning loss is nonfinite")
            manager.optimizer.zero_grad(set_to_none=True)
            result["loss"].backward()
            audit = gradient_audit(core)
            if (
                not audit["finite"]
                or audit["group_norm_sums"]["high_level_policy"] <= 0
                or audit["frozen_with_grad"]
            ):
                raise RuntimeError("G6 behavior-cloning gradient isolation failed")
            grad_norm, learning_rates = manager.optimizer_step()
            core.mark_updated()
            batches.append(
                {
                    **{key: float(result[key].detach()) for key in metric_names},
                    "grad_norm": float(torch.as_tensor(grad_norm)),
                    "learning_rates": [float(value) for value in learning_rates],
                    "gradient_audit": audit,
                }
            )
        history.append(
            {
                "epoch": epoch + 1,
                **{
                    name: sum(row[name] for row in batches) / len(batches)
                    for name in metric_names
                },
                "batch_count": len(batches),
            }
        )
    changes = parameter_changes(core, before)
    if changes["changed_tensor_count"] < 1:
        raise RuntimeError("G6 behavior cloning changed no high-level parameters")
    return {
        "epochs": run_cfg["bc_epochs"],
        "batch_size": batch_size,
        "history": history,
        "initial_loss": history[0]["loss"],
        "final_loss": history[-1]["loss"],
        "final_mode_accuracy": history[-1]["mode_accuracy"],
        "final_per_mode_accuracy": {
            name: history[-1][f"mode_accuracy_{name}"]
            for name in ("navigate", "stop", "recover")
        },
        "parameter_changes": changes,
    }


def collect_ppo_episode(manager, core, run_cfg, spec, episode_slot):
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
        validate_identity(identity, spec, "PPO")
        for decision_index in range(run_cfg["max_rl_decisions"]):
            with torch.no_grad():
                state = manager.model(operation="encode_state", obs=obs)
                top = topology_tensor(topology, device)
                sampled = manager.model(
                    operation="sample_head",
                    state_features=state,
                    topology_features=top,
                    seed=stable_seed(
                        run_cfg["seed"], "g6_batch_rl", spec["episode_id"], decision_index
                    ),
                    deterministic=False,
                )
                reference = manager.model(
                    operation="head_distribution",
                    state_features=state,
                    topology_features=top,
                )
            policy_mode = int(sampled["modes"].reshape(-1)[0])
            policy_waypoint = sampled["waypoints"].reshape(-1, 2)[0].cpu().tolist()
            choice = resolved_choice(
                policy_mode,
                policy_waypoint,
                topology,
                recovery_enabled=True,
                blocked_threshold=run_cfg["blocked_threshold"],
            )
            step = execute_choice(
                env, topology, choice["executed_mode"], choice["waypoint_local_m"]
            )
            decisions.append(
                {
                    "episode_slot": episode_slot,
                    "decision_index": decision_index,
                    "state": state.detach().cpu().float(),
                    "topology_features": top.detach().cpu().float(),
                    "policy_mode": policy_mode,
                    "executed_mode": choice["executed_mode"],
                    "waypoint": policy_waypoint,
                    "executed_waypoint": choice["waypoint_local_m"],
                    "recovery_override": choice["recovery_override"],
                    "choice_source": choice["choice_source"],
                    "behavior_log_prob": float(sampled["behavior_log_probs"].reshape(-1)[0]),
                    "value": float(sampled["values"].reshape(-1)[0]),
                    "reference_mode_logits": reference["mode_logits"].reshape(-1, 3)[0].cpu().tolist(),
                    "reference_waypoint_mean": reference["waypoint_mean"].reshape(-1, 2)[0].cpu().tolist(),
                    "reference_waypoint_log_std": reference["waypoint_log_std"].reshape(-1, 2)[0].cpu().tolist(),
                    "primitive_rewards": step["rewards"],
                    "duration": step["duration"],
                    "actions": step["executed_actions"],
                    "collisions": step["collisions"],
                    "waypoint_execution": step["waypoint_execution"],
                    "terminal": step["terminal"],
                    "info": jsonable(step["info"]),
                }
            )
            obs = step["obs"]
            if step["terminal"]:
                break
        if not decisions:
            raise RuntimeError("G6 PPO episode produced no decisions")
        if decisions[-1]["terminal"]:
            bootstrap = 0.0
        else:
            with torch.no_grad():
                next_state = manager.model(operation="encode_state", obs=obs)
                next_topology = topology_tensor(topology, device)
                next_dist = manager.model(
                    operation="head_distribution",
                    state_features=next_state,
                    topology_features=next_topology,
                )
                bootstrap = float(next_dist["values"].reshape(-1)[0])
        masks = [[index < row["duration"] for index in range(4)] for row in decisions]
        chunk_rewards, durations = collapse_chunk_rewards(
            [row["primitive_rewards"] for row in decisions], masks, run_cfg["gamma"]
        )
        values = [row["value"] for row in decisions]
        advantages, returns, deltas = variable_duration_gae(
            chunk_rewards,
            values,
            values[1:] + [bootstrap],
            durations,
            [row["terminal"] for row in decisions],
            run_cfg["gamma"],
            run_cfg["gae_lambda"],
        )
        for row, advantage, target_return in zip(decisions, advantages, returns):
            row["advantage"] = float(advantage)
            row["return"] = float(target_return)
        return {
            "episode_index": spec["index"],
            "episode_id": str(spec["episode_id"]),
            "scene": spec["scene"],
            "decision_count": len(decisions),
            "primitive_step_count": sum(durations),
            "bootstrap": bootstrap,
            "chunk_rewards": chunk_rewards,
            "durations": durations,
            "deltas": deltas,
            "decisions": decisions,
            "topology": topology.summary(),
        }
    finally:
        env.close()


def replay_individual(manager, rows, indices, device):
    outputs = []
    for index in indices:
        row = rows[int(index)]
        outputs.append(
            manager.model(
                operation="replay_head",
                state_features=row["state"].to(device),
                topology_features=row["topology_features"].to(device),
                modes=torch.tensor([row["policy_mode"]], dtype=torch.int64, device=device),
                waypoints=torch.tensor([row["waypoint"]], dtype=torch.float32, device=device),
            )
        )
    return {
        key: torch.cat([item[key] for item in outputs], dim=0)
        for key in ("log_probs", "entropy", "values", "waypoint_mean", "mode_logits")
    }


def reference_kl(replay, rows, indices, device):
    ref_logits = torch.tensor(
        [rows[int(index)]["reference_mode_logits"] for index in indices],
        dtype=torch.float32,
        device=device,
    )
    ref_mean = torch.tensor(
        [rows[int(index)]["reference_waypoint_mean"] for index in indices],
        dtype=torch.float32,
        device=device,
    )
    ref_log_std = torch.tensor(
        [rows[int(index)]["reference_waypoint_log_std"] for index in indices],
        dtype=torch.float32,
        device=device,
    )
    current_logits = replay["mode_logits"].float()
    ref_log_probs = ref_logits.log_softmax(-1)
    ref_probs = ref_log_probs.exp()
    categorical = (ref_probs * (ref_log_probs - current_logits.log_softmax(-1))).sum(-1)
    current_log_std = torch.clamp(
        manager_log_std(replay, rows, indices, device), -3.0, -0.25
    )
    current_mean = replay["waypoint_mean"].float()
    gaussian = (
        current_log_std
        - ref_log_std
        + (ref_log_std.mul(2).exp() + (ref_mean - current_mean).square())
        / (2.0 * current_log_std.mul(2).exp())
        - 0.5
    ).sum(-1)
    active = torch.tensor(
        [rows[int(index)]["policy_mode"] != STOP_MODE for index in indices],
        dtype=torch.float32,
        device=device,
    )
    combined = categorical + active * gaussian
    return combined, categorical, gaussian


def manager_log_std(replay, rows, indices, device):
    # waypoint_log_std is a trainable scalar vector not returned by replay_head.
    # The live value is reached through the owning FSDP module in train_batched_ppo;
    # this function is rebound there through the module-level cache.
    del replay, rows, device
    if _LIVE_LOG_STD is None:
        raise RuntimeError("G6 live log-std cache is unset")
    return _LIVE_LOG_STD.expand(len(indices), 2)


_LIVE_LOG_STD = None


def train_batched_ppo(manager, core, run_cfg):
    global _LIVE_LOG_STD
    before = parameter_snapshot(core)
    actor_version_at_collection = core.actor_version
    episodes = [
        collect_ppo_episode(manager, core, run_cfg, spec, slot)
        for slot, spec in enumerate(run_cfg["manifest"]["train"])
    ]
    if core.actor_version != actor_version_at_collection:
        raise RuntimeError("G6 policy changed during multi-episode rollout collection")
    rows = [row for episode in episodes for row in episode["decisions"]]
    device = model_device(core)
    all_indices = list(range(len(rows)))
    with torch.no_grad():
        initial = replay_individual(manager, rows, all_indices, device)
        behavior = torch.tensor(
            [row["behavior_log_prob"] for row in rows], dtype=torch.float32, device=device
        )
        initial_ratios = torch.exp(initial["log_probs"].float() - behavior)
        max_preupdate_ratio_deviation = float((initial_ratios - 1.0).abs().max())
    if max_preupdate_ratio_deviation > 0.03:
        raise RuntimeError(
            f"G6 pre-update on-policy ratio deviates by {max_preupdate_ratio_deviation}"
        )
    advantages = torch.tensor([row["advantage"] for row in rows], dtype=torch.float32)
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
    returns = torch.tensor([row["return"] for row in rows], dtype=torch.float32)
    updates = []
    _LIVE_LOG_STD = core.waypoint_log_std.float().clamp(-3.0, -0.25)
    try:
        for epoch in range(run_cfg["ppo_epochs"]):
            generator = torch.Generator(device="cpu").manual_seed(run_cfg["seed"] + 700 + epoch)
            order = torch.randperm(len(rows), generator=generator)
            for start in range(0, len(rows), run_cfg["ppo_batch_size"]):
                indices = order[start : start + run_cfg["ppo_batch_size"]].tolist()
                replay = replay_individual(manager, rows, indices, device)
                old_log_probs = torch.tensor(
                    [rows[index]["behavior_log_prob"] for index in indices],
                    dtype=torch.float32,
                    device=device,
                )
                ratios = torch.exp(replay["log_probs"].float() - old_log_probs)
                batch_advantages = advantages[indices].to(device)
                clipped = ratios.clamp(
                    1.0 - run_cfg["clip_ratio"], 1.0 + run_cfg["clip_ratio"]
                )
                actor_loss = -torch.minimum(
                    ratios * batch_advantages, clipped * batch_advantages
                ).mean()
                batch_returns = returns[indices].to(device)
                value_loss = 0.5 * (
                    replay["values"].float() - batch_returns
                ).square().mean()
                entropy = replay["entropy"].float().mean()
                _LIVE_LOG_STD = core.waypoint_log_std.float().clamp(-3.0, -0.25)
                combined_kl, categorical_kl, gaussian_kl = reference_kl(
                    replay, rows, indices, device
                )
                kl = combined_kl.mean()
                loss = (
                    actor_loss
                    + run_cfg["value_coef"] * value_loss
                    - run_cfg["entropy_coef"] * entropy
                    + run_cfg["bc_kl_coef"] * kl
                )
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("G6 PPO loss is nonfinite")
                manager.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                audit = gradient_audit(core)
                if (
                    not audit["finite"]
                    or audit["group_norm_sums"]["high_level_policy"] <= 0
                    or audit["group_norm_sums"]["value_head"] <= 0
                    or audit["frozen_with_grad"]
                ):
                    raise RuntimeError("G6 PPO gradient isolation failed")
                grad_norm, learning_rates = manager.optimizer_step()
                core.mark_updated()
                updates.append(
                    {
                        "epoch": epoch + 1,
                        "batch_index": len(updates),
                        "sample_count": len(indices),
                        "episode_slots": sorted(
                            {rows[index]["episode_slot"] for index in indices}
                        ),
                        "ratio_mean": float(ratios.detach().mean()),
                        "ratio_max_deviation": float((ratios.detach() - 1.0).abs().max()),
                        "loss": float(loss.detach()),
                        "actor_loss": float(actor_loss.detach()),
                        "value_loss": float(value_loss.detach()),
                        "entropy": float(entropy.detach()),
                        "bc_reference_kl": float(kl.detach()),
                        "categorical_kl": float(categorical_kl.detach().mean()),
                        "gaussian_kl": float(gaussian_kl.detach().mean()),
                        "grad_norm": float(torch.as_tensor(grad_norm)),
                        "learning_rates": [float(value) for value in learning_rates],
                        "gradient_audit": audit,
                        "actor_version_after": core.actor_version,
                    }
                )
        _LIVE_LOG_STD = core.waypoint_log_std.float().clamp(-3.0, -0.25)
        with torch.no_grad():
            final = replay_individual(manager, rows, all_indices, device)
            final_kl, final_cat, final_gaussian = reference_kl(
                final, rows, all_indices, device
            )
        changes = parameter_changes(core, before)
        if changes["changed_tensor_count"] < 1:
            raise RuntimeError("G6 batched PPO changed no trainable parameters")
        serialized_episodes = []
        for episode in episodes:
            item = {key: value for key, value in episode.items() if key != "decisions"}
            item["decisions"] = [
                {
                    key: jsonable(value)
                    for key, value in row.items()
                    if key not in ("state", "topology_features")
                }
                for row in episode["decisions"]
            ]
            serialized_episodes.append(item)
        return {
            "collection_policy_version": actor_version_at_collection,
            "collection_episode_count": len(episodes),
            "collection_scene_count": len({row["scene"] for row in episodes}),
            "decision_count": len(rows),
            "primitive_steps": sum(row["primitive_step_count"] for row in episodes),
            "all_episodes_collected_before_update": True,
            "max_preupdate_ratio_deviation": max_preupdate_ratio_deviation,
            "ppo_epochs": run_cfg["ppo_epochs"],
            "batch_size": run_cfg["ppo_batch_size"],
            "update_count": len(updates),
            "minimum_episode_count_per_update": min(
                len(row["episode_slots"]) for row in updates
            ),
            "recovery_override_count": sum(
                row["recovery_override"] for episode in episodes for row in episode["decisions"]
            ),
            "final_bc_reference_kl": {
                "mean": float(final_kl.mean()),
                "max": float(final_kl.max()),
                "categorical_mean": float(final_cat.mean()),
                "gaussian_mean": float(final_gaussian.mean()),
            },
            "updates": updates,
            "episodes": serialized_episodes,
            "parameter_changes": changes,
        }
    finally:
        _LIVE_LOG_STD = None


def run_train(run_cfg):
    output = Path(run_cfg["output_dir"])
    manager, started = None, time.time()
    try:
        manager, core, manifest = setup_manager(run_cfg, total_steps=256)
        initial_sha = state_sha256(parameter_snapshot(core))
        dataset, supervision = collect_supervision(manager, core, run_cfg)
        bc = train_behavior_cloning(manager, core, run_cfg, dataset)
        bc_checkpoint = save_head_checkpoint(
            core,
            output / "checkpoint_bc_balanced",
            {
                "schema": "streamvln_recovery_g6_v1",
                "variant": "bc_balanced",
                "stream_commit": run_cfg["stream_commit"],
                "rlinf_commit": run_cfg["rlinf_commit"],
                "val_unseen_used": False,
                "memory_adapter": False,
                "secondary_post_training_used": False,
            },
        )
        ppo = train_batched_ppo(manager, core, run_cfg)
        rl_checkpoint = save_head_checkpoint(
            core,
            output / "checkpoint_bc_batch_rl",
            {
                "schema": "streamvln_recovery_g6_v1",
                "variant": "bc_batch_rl",
                "stream_commit": run_cfg["stream_commit"],
                "rlinf_commit": run_cfg["rlinf_commit"],
                "val_unseen_used": False,
                "memory_adapter": False,
                "secondary_post_training_used": False,
            },
        )
        record = {
            "schema_version": 1,
            "kind": "streamvln-g6-balanced-warmstart-batched-ppo",
            "source_contract": {
                "stream_commit": run_cfg["stream_commit"],
                "rlinf_commit": run_cfg["rlinf_commit"],
                "source_kind": "upstream-original",
                "memory_adapter": False,
                "secondary_post_training_used": False,
                "val_unseen_used_for_training": False,
            },
            "manifest": manifest,
            "initial_head_sha256": initial_sha,
            "supervision": supervision,
            "behavior_cloning": bc,
            "ppo": ppo,
            "checkpoints": {"bc_balanced": bc_checkpoint, "bc_batch_rl": rl_checkpoint},
            "elapsed_s": time.time() - started,
        }
        counts = supervision["balanced_mode_counts"]
        record["guard_checks"] = {
            "source_episode_count": supervision["source_episode_count"] == 8,
            "scene_count": supervision["scene_count"] == 8,
            "balanced_modes": counts["navigate"] == counts["stop"] == counts["recover"],
            "ppo_episode_count": ppo["collection_episode_count"] == 8,
            "single_collection_policy": ppo["all_episodes_collected_before_update"],
            "multi_episode_minibatches": ppo["minimum_episode_count_per_update"] >= 2,
            "preupdate_ratio": ppo["max_preupdate_ratio_deviation"] <= 0.03,
            "bc_kl_mean": ppo["final_bc_reference_kl"]["mean"] <= run_cfg["max_bc_kl"],
            "bc_kl_hard_max": ppo["final_bc_reference_kl"]["max"] <= run_cfg["max_bc_kl_hard"],
        }
        record["all_guards_passed"] = all(record["guard_checks"].values())
        (output / "train.json").write_text(
            json.dumps(jsonable(record), indent=2, sort_keys=True) + "\n"
        )
        if not record["all_guards_passed"]:
            failed = sorted(name for name, passed in record["guard_checks"].items() if not passed)
            raise RuntimeError(f"G6 training guards did not all pass: {failed}")
        return record
    finally:
        cleanup_manager(manager)


def oracle_episode(run_cfg, spec, recovery_enabled):
    env = make_env(
        run_cfg,
        split="train",
        episode_index=spec["index"],
        max_steps=run_cfg["max_oracle_steps"],
    )
    topology = OnlineTopology(max_decisions=run_cfg["max_oracle_steps"] // 4)
    actions, decisions, last_info = [], [], None
    last_truncated = False
    try:
        obs, identity = env.reset()
        validate_identity(identity, spec, "oracle")
        for decision_index in range(run_cfg["max_oracle_steps"] // 4):
            label = env.expert_waypoint(run_cfg["expert_lookahead_m"])
            policy_mode = STOP_MODE if label["stop"] else NAVIGATE_MODE
            choice = resolved_choice(
                policy_mode,
                label["waypoint_local_m"],
                topology,
                recovery_enabled=recovery_enabled,
                blocked_threshold=run_cfg["blocked_threshold"],
            )
            step = execute_choice(
                env, topology, choice["executed_mode"], choice["waypoint_local_m"]
            )
            actions.extend(step["executed_actions"])
            last_info, last_truncated = step["info"], step["truncated"]
            decisions.append(
                {
                    "decision": decision_index,
                    "oracle_mode": MODE_NAMES[policy_mode],
                    "executed_mode": MODE_NAMES[choice["executed_mode"]],
                    "recovery_override": choice["recovery_override"],
                    "actions": step["executed_actions"],
                    "collisions": step["collisions"],
                    "waypoint_execution": step["waypoint_execution"],
                }
            )
            obs = step["obs"]
            if step["terminal"]:
                break
        if last_info is None or "episode" not in last_info:
            raise RuntimeError("G6 oracle diagnostic lacks terminal metrics")
        metrics = {key: float(value.item()) for key, value in last_info["episode"].items()}
        return {
            "episode_index": spec["index"],
            "episode_id": str(spec["episode_id"]),
            "scene": spec["scene"],
            "success": metrics["success"],
            "spl": metrics["spl"],
            "navigation_error_m": metrics["distance_to_goal"],
            "path_length_m": metrics["path_length"],
            "collision_count": int(metrics["collision_count"]),
            "primitive_steps": len(actions),
            "timeout": last_truncated,
            "recovery_override_count": sum(row["recovery_override"] for row in decisions),
            "actions": actions,
            "decisions": decisions,
            "topology": topology.summary(),
        }
    finally:
        env.close()


def aggregate_episodes(episodes):
    decisions = [decision for episode in episodes for decision in episode["decisions"]]
    actions = [action for episode in episodes for action in episode["actions"]]
    executed_modes = Counter(decision.get("executed_mode", "unknown") for decision in decisions)
    policy_modes = Counter(decision.get("policy_mode", decision.get("oracle_mode", "unknown")) for decision in decisions)
    errors = [decision["waypoint_execution"]["l2_m"] for decision in decisions]
    recovery_errors = [
        decision["waypoint_execution"]["l2_m"]
        for decision in decisions
        if decision.get("executed_mode") == "recover"
    ]
    return {
        "episode_count": len(episodes),
        "scene_count": len({row["scene"] for row in episodes}),
        "success_rate": mean(episodes, "success"),
        "spl": mean(episodes, "spl"),
        "navigation_error_m": mean(episodes, "navigation_error_m"),
        "path_length_m": mean(episodes, "path_length_m"),
        "collision_count": mean(episodes, "collision_count"),
        "primitive_steps": mean(episodes, "primitive_steps"),
        "timeout_count": sum(row["timeout"] for row in episodes),
        "recovery_override_count": sum(row.get("recovery_override_count", 0) for row in episodes),
        "policy_mode_counts": dict(policy_modes),
        "executed_mode_counts": dict(executed_modes),
        "action_counts": {str(action): actions.count(action) for action in range(4)},
        "waypoint_execution_error_m": sum(errors) / max(1, len(errors)),
        "recovery_waypoint_execution_error_m": (
            sum(recovery_errors) / len(recovery_errors) if recovery_errors else None
        ),
    }


def run_oracle(run_cfg):
    output = Path(run_cfg["output_dir"])
    variants = {}
    for name, enabled in (("plain_compiler", False), ("recovery_compiler", True)):
        episodes = [
            oracle_episode(run_cfg, spec, enabled) for spec in run_cfg["manifest"]["train"]
        ]
        variants[name] = {
            "recovery_enabled": enabled,
            "episodes": episodes,
            "aggregate": aggregate_episodes(episodes),
        }
    result = {
        "schema_version": 1,
        "kind": "streamvln-g6-train-only-oracle-waypoint-compiler-diagnostic",
        "privileged_shortest_path_used": True,
        "split": "train",
        "policy_claim": False,
        "max_primitive_steps": run_cfg["max_oracle_steps"],
        "variants": variants,
        "all_guards_passed": all(
            value["aggregate"]["episode_count"] == 8
            and value["aggregate"]["scene_count"] == 8
            for value in variants.values()
        ),
    }
    if not result["all_guards_passed"]:
        raise RuntimeError("G6 oracle diagnostic guards did not pass")
    (output / "oracle_diagnostic.json").write_text(
        json.dumps(jsonable(result), indent=2, sort_keys=True) + "\n"
    )
    return result


def evaluate_episode(model, run_cfg, spec, variant, recovery_enabled):
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
        validate_identity(identity, spec, "evaluation")
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
            policy_mode = int(sampled["modes"].reshape(-1)[0])
            policy_waypoint = sampled["waypoints"].reshape(-1, 2)[0].cpu().tolist()
            choice = resolved_choice(
                policy_mode,
                policy_waypoint,
                topology,
                recovery_enabled=recovery_enabled,
                blocked_threshold=run_cfg["blocked_threshold"],
            )
            step = execute_choice(
                env, topology, choice["executed_mode"], choice["waypoint_local_m"]
            )
            actions.extend(step["executed_actions"])
            last_info, last_truncated = step["info"], step["truncated"]
            decisions.append(
                {
                    "decision": decision_index,
                    "policy_mode": MODE_NAMES[policy_mode],
                    "executed_mode": MODE_NAMES[choice["executed_mode"]],
                    "choice_source": choice["choice_source"],
                    "recovery_override": choice["recovery_override"],
                    "policy_waypoint_local_m": policy_waypoint,
                    "executed_waypoint_local_m": choice["waypoint_local_m"],
                    "mode_logits": sampled["mode_logits"].reshape(-1, 3)[0].cpu().tolist(),
                    "actions": step["executed_actions"],
                    "collisions": step["collisions"],
                    "waypoint_execution": step["waypoint_execution"],
                    "terminated": step["terminated"],
                    "truncated": step["truncated"],
                    "topology": topology.summary(),
                }
            )
            obs = step["obs"]
            if step["terminal"]:
                break
        if last_info is None or "episode" not in last_info:
            raise RuntimeError("G6 evaluation did not produce terminal episode metrics")
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
            "recovery_override_count": sum(row["recovery_override"] for row in decisions),
            "actions": actions,
            "decisions": decisions,
            "topology": topology.summary(),
        }
    finally:
        env.close()


def run_evaluation(run_cfg, variant):
    output = Path(run_cfg["output_dir"])
    variants = {
        "bc_plain": (output / "checkpoint_bc_balanced/policy.safetensors", False),
        "bc_recovery": (output / "checkpoint_bc_balanced/policy.safetensors", True),
        "bc_batch_rl_recovery": (output / "checkpoint_bc_batch_rl/policy.safetensors", True),
    }
    checkpoint, recovery_enabled = variants[variant]
    checkpoint = checkpoint.resolve(strict=True)
    model = get_recovery_model(
        make_policy_cfg(run_cfg, trainable=False, head_checkpoint=str(checkpoint))
    )
    model.to("cuda:0")
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("G6 evaluation model is not frozen")
    audit = checkpoint_load_audit(model, checkpoint)
    if not audit["passed"]:
        raise RuntimeError("G6 checkpoint exact-load audit failed")
    started = time.time()
    episodes = [
        evaluate_episode(model, run_cfg, spec, variant, recovery_enabled)
        for spec in run_cfg["manifest"]["val_unseen"]
    ]
    aggregate = aggregate_episodes(episodes)
    aggregate["outcomes"] = {
        key: sum(row["outcome"] == key for row in episodes)
        for key in ("success", "premature_stop", "timeout", "other_failure")
    }
    result = {
        "schema_version": 1,
        "kind": "streamvln-g6-fixed-val-unseen",
        "variant": variant,
        "deterministic_greedy": True,
        "recovery_enabled": recovery_enabled,
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
    result["all_guards_passed"] = (
        len(episodes) == 8
        and len({row["scene"] for row in episodes}) == 8
        and audit["passed"]
    )
    if not result["all_guards_passed"]:
        raise RuntimeError(f"G6 {variant} evaluation guards did not all pass")
    (output / f"eval_{variant}.json").write_text(
        json.dumps(jsonable(result), indent=2, sort_keys=True) + "\n"
    )
    return result


def compare_results(run_cfg, started):
    output = Path(run_cfg["output_dir"])
    names = ("bc_plain", "bc_recovery", "bc_batch_rl_recovery")
    variants = {
        name: json.loads((output / f"eval_{name}.json").read_text()) for name in names
    }
    expected = [(row["index"], row["scene"]) for row in run_cfg["manifest"]["val_unseen"]]
    orders = {
        name: [(row["episode_index"], row["scene"]) for row in result["episodes"]]
        for name, result in variants.items()
    }
    if any(order != expected for order in orders.values()):
        raise RuntimeError("G6 evaluation order differs from the fixed manifest")
    g5_path = Path(run_cfg["g5_result"]).resolve(strict=True)
    if sha256_file(g5_path) != run_cfg["g5_result_sha256"]:
        raise RuntimeError("G6 external G5 result hash changed")
    g5 = json.loads(g5_path.read_text())
    for name in ("bc_only", "bc_rl"):
        order = [
            (row["episode_index"], row["scene"])
            for row in g5["g5_variants"][name]["episodes"]
        ]
        if order != expected:
            raise RuntimeError("G6 G5 anchor episode order differs")
    oracle = json.loads((output / "oracle_diagnostic.json").read_text())
    train = json.loads((output / "train.json").read_text())
    metrics = (
        "success_rate",
        "spl",
        "navigation_error_m",
        "path_length_m",
        "collision_count",
        "primitive_steps",
        "waypoint_execution_error_m",
    )
    deltas = {
        "recovery_minus_plain_bc": {
            metric: variants["bc_recovery"]["aggregate"][metric]
            - variants["bc_plain"]["aggregate"][metric]
            for metric in metrics
        },
        "batch_rl_minus_bc_with_recovery": {
            metric: variants["bc_batch_rl_recovery"]["aggregate"][metric]
            - variants["bc_recovery"]["aggregate"][metric]
            for metric in metrics
        },
    }
    action_changes = {
        pair: sum(
            left["actions"] != right["actions"]
            for left, right in zip(variants[a]["episodes"], variants[b]["episodes"])
        )
        for pair, a, b in (
            ("bc_recovery_vs_plain", "bc_recovery", "bc_plain"),
            (
                "bc_batch_rl_recovery_vs_bc_recovery",
                "bc_batch_rl_recovery",
                "bc_recovery",
            ),
        )
    }
    result = {
        "schema_version": 1,
        "kind": "streamvln-g6-recovery-and-stable-rl-comparison",
        "claim_scope": "bounded matched G6 causal controls; train oracle is an execution upper-bound diagnostic only; G5 is a fixed prior-run anchor",
        "source_contract": {
            "stream_commit": run_cfg["stream_commit"],
            "rlinf_commit": run_cfg["rlinf_commit"],
            "source_kind": "upstream-original",
            "memory_adapter": False,
            "secondary_post_training_used": False,
        },
        "matched_episode_order": expected,
        "g6_variants": variants,
        "train_only_oracle_diagnostic": oracle,
        "training": train,
        "g5_prior_anchor": {
            "path": str(g5_path),
            "sha256": run_cfg["g5_result_sha256"],
            "matched_episodes_and_budget": True,
            "bc_only": g5["g5_variants"]["bc_only"]["aggregate"],
            "bc_rl": g5["g5_variants"]["bc_rl"]["aggregate"],
        },
        "matched_delta": deltas,
        "paired_action_sequence_changes": action_changes,
        "elapsed_s_total": time.time() - started,
        "all_guards_passed": (
            all(value["all_guards_passed"] for value in variants.values())
            and oracle["all_guards_passed"]
            and train["all_guards_passed"]
        ),
    }
    path = output / "result.json"
    path.write_text(json.dumps(jsonable(result), indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "all_guards_passed": result["all_guards_passed"],
                "result": str(path),
                "result_sha256": sha256_file(path),
                "aggregates": {
                    name: value["aggregate"] for name, value in variants.items()
                },
                "oracle": {
                    name: value["aggregate"]
                    for name, value in oracle["variants"].items()
                },
                "matched_delta": deltas,
                "paired_action_sequence_changes": action_changes,
            },
            sort_keys=True,
        )
    )
    return result


def run_subprocess(config_path, phase, variant=None):
    command = [
        sys.executable,
        "-B",
        str(Path(__file__).resolve()),
        "--phase",
        phase,
        "--run-config",
        str(config_path),
    ]
    if variant:
        command.extend(["--variant", variant])
    subprocess.run(command, check=True, env=os.environ.copy())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=("orchestrate", "oracle", "train", "eval"), default="orchestrate"
    )
    parser.add_argument(
        "--variant", choices=("bc_plain", "bc_recovery", "bc_batch_rl_recovery")
    )
    for name in (
        "manifest",
        "g5-result",
        "g5-result-sha256",
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
    parser.add_argument("--run-config")
    parser.add_argument("--egl-vendor-json", default="/usr/share/glvnd/egl_vendor.d/50_mesa.json")
    parser.add_argument("--context-tokens", type=int, default=4096)
    parser.add_argument("--head-hidden", type=int, default=256)
    parser.add_argument("--bc-epochs", type=int, default=8)
    parser.add_argument("--bc-batch-size", type=int, default=24)
    parser.add_argument("--collection-passes", type=int, default=2)
    parser.add_argument("--max-bc-decisions", type=int, default=12)
    parser.add_argument("--max-rl-decisions", type=int, default=10)
    parser.add_argument("--max-eval-steps", type=int, default=40)
    parser.add_argument("--max-oracle-steps", type=int, default=40)
    parser.add_argument("--expert-lookahead-m", type=float, default=1.0)
    parser.add_argument("--blocked-threshold", type=int, default=2)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--ppo-batch-size", type=int, default=32)
    parser.add_argument("--bc-kl-coef", type=float, default=0.05)
    parser.add_argument("--max-bc-kl", type=float, default=0.5)
    parser.add_argument("--max-bc-kl-hard", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--master-port", type=int, default=29587)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--value-learning-rate", type=float, default=5e-5)
    return parser.parse_args()


def configure_process_git_safe_directory(path):
    count_text = os.environ.get("GIT_CONFIG_COUNT", "0")
    if not count_text.isdigit():
        raise RuntimeError("G6 inherited malformed GIT_CONFIG_COUNT")
    count = int(count_text)
    os.environ["GIT_CONFIG_COUNT"] = str(count + 1)
    os.environ[f"GIT_CONFIG_KEY_{count}"] = "safe.directory"
    os.environ[f"GIT_CONFIG_VALUE_{count}"] = str(Path(path).resolve())


def load_run_config(path):
    run_cfg = json.loads(Path(path).read_text())
    validate_manifest(run_cfg["manifest"])
    return run_cfg


def main():
    args = parse_args()
    if args.phase != "orchestrate":
        if not args.run_config:
            raise ValueError("G6 subprocess phase requires --run-config")
        run_cfg = load_run_config(args.run_config)
        configure_process_git_safe_directory(run_cfg["stream_source"])
        if args.phase == "oracle":
            run_oracle(run_cfg)
        elif args.phase == "train":
            run_train(run_cfg)
        elif args.phase == "eval":
            if not args.variant:
                raise ValueError("G6 eval phase requires --variant")
            run_evaluation(run_cfg, args.variant)
        return

    required = (
        "manifest",
        "g5_result",
        "g5_result_sha256",
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
        raise ValueError("G6 orchestration is missing required path arguments")
    if (
        args.max_eval_steps % 4
        or args.max_oracle_steps % 4
        or args.max_bc_decisions < 2
        or args.max_rl_decisions < 2
        or args.collection_passes < 2
        or args.blocked_threshold < 1
        or args.ppo_epochs < 1
        or args.ppo_batch_size < 2
        or args.learning_rate > 1e-4
        or args.context_tokens < 2048
    ):
        raise ValueError("G6 bounded training/evaluation budget is invalid")
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("G6 output directory must be fresh")
    output.mkdir(parents=True, exist_ok=True)
    path_keys = (
        "manifest",
        "g5_result",
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
    paths = {
        key: str(Path(getattr(args, key)).expanduser().resolve(strict=True))
        for key in path_keys
    }
    manifest = json.loads(Path(paths["manifest"]).read_text())
    validate_manifest(manifest)
    stream_commit = git_head(Path(paths["stream_source"]))
    if stream_commit != ORIGINAL_STREAM_COMMIT:
        raise RuntimeError("G6 requires the pinned original StreamVLN commit")
    if sha256_file(Path(paths["g5_result"])) != args.g5_result_sha256:
        raise RuntimeError("G6 G5 anchor hash differs before execution")
    run_cfg = {
        **paths,
        "output_dir": str(output),
        "manifest": manifest,
        "stream_commit": stream_commit,
        "rlinf_commit": git_head(Path(paths["rlinf_root"])),
        "g5_result_sha256": args.g5_result_sha256,
        "context_tokens": args.context_tokens,
        "head_hidden": args.head_hidden,
        "bc_epochs": args.bc_epochs,
        "bc_batch_size": args.bc_batch_size,
        "collection_passes": args.collection_passes,
        "max_bc_decisions": args.max_bc_decisions,
        "max_rl_decisions": args.max_rl_decisions,
        "max_eval_steps": args.max_eval_steps,
        "max_oracle_steps": args.max_oracle_steps,
        "expert_lookahead_m": args.expert_lookahead_m,
        "blocked_threshold": args.blocked_threshold,
        "ppo_epochs": args.ppo_epochs,
        "ppo_batch_size": args.ppo_batch_size,
        "bc_kl_coef": args.bc_kl_coef,
        "max_bc_kl": args.max_bc_kl,
        "max_bc_kl_hard": args.max_bc_kl_hard,
        "seed": args.seed,
        "master_port": args.master_port,
        "learning_rate": args.learning_rate,
        "value_learning_rate": args.value_learning_rate,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_ratio": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.01,
    }
    validate_dataset_hashes(run_cfg)
    config_path = output / "run_config.json"
    config_path.write_text(json.dumps(run_cfg, indent=2, sort_keys=True) + "\n")
    started = time.time()
    run_subprocess(config_path, "oracle")
    run_subprocess(config_path, "train")
    for variant in ("bc_plain", "bc_recovery", "bc_batch_rl_recovery"):
        run_subprocess(config_path, "eval", variant)
    compare_results(run_cfg, started)


if __name__ == "__main__":
    main()
