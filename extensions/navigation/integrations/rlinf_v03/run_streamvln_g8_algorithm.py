"""Matched BC versus executor-aligned BC+RL algorithm gate for StreamVLN G8."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import subprocess
import time

import torch
from omegaconf import OmegaConf

from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager

from run_streamvln_ppo_g4 import make_env
from run_streamvln_topology_g5 import (
    ORIGINAL_STREAM_COMMIT,
    checkpoint_load_audit,
    cleanup_manager,
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
from run_streamvln_plumbing_g717 import (
    frozen_parameter_sentinels,
    optimizer_parameter_scope,
    small_file_inventory,
)
from streamvln_chunk_ppo import collapse_chunk_rewards, variable_duration_gae
from streamvln_executor_contract import PRIVILEGED_POLICY_KEYS, assert_history_only_keys, feedback_from_dict
from streamvln_g8_action import (
    ACTION_COUNT,
    NAVIGATE_ACTION_COUNT,
    action_mode,
    action_waypoint_local_m,
    assert_g8_action_contract,
    compile_g8_action,
    target_action_id,
)
from streamvln_g8_policy import get_model as get_g8_model
from streamvln_recovery import waypoint_execution_error
from streamvln_topology import MODE_NAMES, NAVIGATE_MODE, OnlineTopology, RECOVER_MODE, STOP_MODE


RLINF_COMMIT = "0505431899574619da86f551bad70b71e0ea2177"
MANIFEST_SHA256 = "59b414d83ab30e38cb4fb2c5bce14b8408bed5f2a23cfeb41aeb84570cb2331a"


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n")


def git_head(path: Path) -> str:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={path}", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"cannot resolve git head for {path}: {result.stderr}")
    return result.stdout.strip()


def make_policy_cfg(run_cfg: dict, *, trainable: bool, head_checkpoint=None):
    return OmegaConf.create({
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
        "topology_dim": 14,
        "head_hidden": run_cfg["head_hidden"],
        "alignment_coefficient": 4.0,
        "precision": "bf16",
        "add_value_head": True,
        "is_lora": False,
        "load_to_device": False,
    })


class G8AlgorithmFSDPManager(FSDPModelManager):
    def model_provider_func(self):
        return get_g8_model(self._cfg.model)


def setup_manager(run_cfg: dict, total_steps: int):
    from run_streamvln_ppo_g3 import init_distributed

    torch.manual_seed(run_cfg["seed"])
    torch.cuda.manual_seed_all(run_cfg["seed"])
    init_distributed(run_cfg["master_port"])
    cfg = make_manager_cfg(run_cfg, total_training_steps=total_steps)
    cfg.model = make_policy_cfg(run_cfg, trainable=True)
    manager = G8AlgorithmFSDPManager(cfg, world_size=1, rank=0)
    manager.setup_model_and_optimizer()
    if type(manager.model).__name__ != "FullyShardedDataParallel":
        raise RuntimeError("G8 algorithm actor is not wrapped by RLinf FSDP")
    core = manager.model.module
    manifest = core.trainable_manifest()
    if (
        manifest["source_kind"] != "upstream-original"
        or manifest["memory_adapter"]
        or manifest["secondary_post_training_used"]
        or manifest["backbone_trainable"]
        or manifest["gaussian_waypoint_likelihood"]
        or manifest["waypoint_log_std_trainable"]
        or manifest["action_count"] != ACTION_COUNT
    ):
        raise RuntimeError("G8 algorithm model boundary is invalid")
    return manager, core, manifest


def validate_identity(identity: dict, spec: dict, phase: str) -> None:
    if (
        str(identity.get("episode_id")) != str(spec["episode_id"])
        or Path(identity.get("scene_id", "")).stem != spec["scene"]
    ):
        raise RuntimeError(f"G8 {phase} episode identity changed")


def execute_action(env, topology: OnlineTopology, action_id: int, decision: int) -> dict:
    del decision
    target = action_waypoint_local_m(action_id, topology)
    start_pose = (topology.x, topology.z, topology.heading)
    proposed = compile_g8_action(action_id, topology)
    obs_list, rewards, terminations, truncations, infos = env.chunk_step(
        torch.tensor(proposed, dtype=torch.int64).reshape(1, 4, 1)
    )
    feedback = feedback_from_dict(infos[0].get("executor_feedback"))
    if list(feedback.proposed_actions) != proposed:
        raise RuntimeError("G8 proposed macro differs from executor feedback")
    topology.update_from_feedback(feedback.to_dict())
    return {
        "obs": obs_list[0],
        "info": infos[0],
        "feedback": feedback,
        "target_waypoint": [float(target[0]), float(target[1])],
        "waypoint_execution_error_m": waypoint_execution_error(target, start_pose, topology)["l2_m"],
        "primitive_rewards": [float(value) for value in rewards[0].detach().cpu().tolist()],
        "terminated": bool(terminations.any()),
        "truncated": bool(truncations.any()),
        "terminal": bool(torch.logical_or(terminations, truncations).any()),
    }


def collect_bc_dataset(manager, core, run_cfg: dict):
    device = model_device(core)
    training = run_cfg["manifest"]["training"]
    rows, episode_records = [], []
    for spec in run_cfg["arm"]["train"]:
        env = make_env(
            run_cfg,
            split="train",
            episode_index=spec["index"],
            max_steps=training["max_bc_decisions"] * 4,
        )
        topology = OnlineTopology(max_decisions=training["max_bc_decisions"])
        decisions = []
        try:
            obs, identity = env.reset()
            validate_identity(identity, spec, "BC")
            for decision in range(training["max_bc_decisions"]):
                label = env.expert_waypoint(training["expert_lookahead_m"])
                mode = STOP_MODE if label["stop"] else NAVIGATE_MODE
                waypoint = [float(value) for value in label["waypoint_local_m"]]
                with torch.no_grad():
                    state = manager.model(operation="encode_state", obs=obs)
                    top = topology_tensor(topology, device)
                action_id = target_action_id(mode, waypoint, topology)
                rows.append({
                    "state": state.detach().cpu().float(),
                    "topology": top.detach().cpu().float(),
                    "action_id": action_id,
                    "mode": mode,
                    "waypoint": waypoint,
                })
                step = execute_action(env, topology, action_id, decision)
                decisions.append({
                    "decision": decision,
                    "action_id": action_id,
                    "mode": MODE_NAMES[mode],
                    "target_waypoint": waypoint,
                    "executed_actions": list(step["feedback"].executed_actions),
                    "blocked_forward_count": step["feedback"].blocked_forward_count,
                })
                obs = step["obs"]
                if step["terminal"]:
                    break
            episode_records.append({**spec, "decision_count": len(decisions), "decisions": decisions})
        finally:
            env.close()
    modes = Counter(row["mode"] for row in rows)
    if not rows or modes[STOP_MODE] < 1 or modes[NAVIGATE_MODE] < 1:
        raise RuntimeError("G8 BC supervision lacks NAVIGATE or STOP")
    navigate = [index for index, row in enumerate(rows) if row["mode"] == NAVIGATE_MODE]
    stop = [index for index, row in enumerate(rows) if row["mode"] == STOP_MODE]
    recover = [index for index, row in enumerate(rows) if row["mode"] == RECOVER_MODE]
    balanced = navigate + [stop[index % len(stop)] for index in range(len(navigate))] + recover
    tensors = {
        "state_features": torch.stack([row["state"] for row in rows]),
        "topology_features": torch.stack([row["topology"] for row in rows]),
        "target_action_ids": torch.tensor([row["action_id"] for row in rows], dtype=torch.int64),
        "target_waypoints": torch.tensor([row["waypoint"] for row in rows], dtype=torch.float32),
    }
    return tensors, balanced, {
        "raw_row_count": len(rows),
        "balanced_row_count": len(balanced),
        "mode_counts": {MODE_NAMES[key]: modes[key] for key in range(3)},
        "balanced_stop_count": len(navigate),
        "episode_count": len(episode_records),
        "scene_count": len({row["scene"] for row in episode_records}),
        "policy_observation_excludes_expert": True,
        "episodes": episode_records,
    }


def train_bc(manager, core, run_cfg: dict, tensors: dict, balanced: list[int]):
    training = run_cfg["manifest"]["training"]
    device = model_device(core)
    before = parameter_snapshot(core)
    history = []
    for epoch in range(training["bc_epochs"]):
        generator = torch.Generator(device="cpu").manual_seed(run_cfg["seed"] + epoch)
        permutation = torch.randperm(len(balanced), generator=generator).tolist()
        epoch_rows = []
        for start in range(0, len(permutation), training["bc_batch_size"]):
            indices = [balanced[index] for index in permutation[start:start + training["bc_batch_size"]]]
            batch = {key: value[indices].to(device) for key, value in tensors.items()}
            result = manager.model(operation="bc_loss_g8", **batch)
            if not bool(torch.isfinite(result["loss"])):
                raise RuntimeError("G8 BC loss is nonfinite")
            manager.optimizer.zero_grad(set_to_none=True)
            result["loss"].backward()
            audit = gradient_audit(core)
            if not audit["finite"] or audit["group_norm_sums"]["high_level_policy"] <= 0 or audit["frozen_with_grad"]:
                raise RuntimeError("G8 BC gradient scope failed")
            grad_norm, learning_rates = manager.optimizer_step()
            core.mark_updated()
            epoch_rows.append({
                "loss": float(result["loss"].detach()),
                "action_loss": float(result["action_loss"].detach()),
                "waypoint_loss": float(result["waypoint_loss"].detach()),
                "action_accuracy": float(result["action_accuracy"].detach()),
                "grad_norm": float(torch.as_tensor(grad_norm)),
                "learning_rates": [float(value) for value in learning_rates],
            })
        history.append({
            "epoch": epoch + 1,
            "loss": sum(row["loss"] for row in epoch_rows) / len(epoch_rows),
            "action_accuracy": sum(row["action_accuracy"] for row in epoch_rows) / len(epoch_rows),
            "update_count": len(epoch_rows),
        })
    return {
        "history": history,
        "update_count": sum(row["update_count"] for row in history),
        "parameter_changes": parameter_changes(core, before),
    }


def metric_value(value) -> float:
    return float(value.item()) if hasattr(value, "item") else float(value)


def evaluate_episode(manager, core, run_cfg: dict, spec: dict, variant: str) -> dict:
    evaluation = run_cfg["manifest"]["evaluation"]
    max_steps = evaluation["max_primitive_steps"]
    env = make_env(run_cfg, split="val_seen", episode_index=spec["index"], max_steps=max_steps)
    topology = OnlineTopology(max_decisions=max_steps // 4)
    actions, action_ids, decisions = [], [], []
    blocked_count = 0
    waypoint_errors = []
    last_info = None
    last_truncated = False
    try:
        obs, identity = env.reset()
        validate_identity(identity, spec, f"{variant} evaluation")
        for decision in range(max_steps // 4):
            with torch.no_grad():
                state = manager.model(operation="encode_state", obs=obs)
                top = topology_tensor(topology, model_device(core))
                sampled = manager.model(
                    operation="sample_g8",
                    state_features=state,
                    topology_features=top,
                    seed=stable_seed(run_cfg["seed"], variant, spec["episode_id"], decision),
                    deterministic=True,
                )
            action_id = int(sampled["action_ids"].reshape(-1)[0])
            step = execute_action(env, topology, action_id, decision)
            executed = list(step["feedback"].executed_actions)
            actions.extend(executed)
            action_ids.append(action_id)
            blocked_count += step["feedback"].blocked_forward_count
            if action_mode(action_id) != STOP_MODE:
                waypoint_errors.append(step["waypoint_execution_error_m"])
            decisions.append({
                "decision": decision,
                "action_id": action_id,
                "mode": MODE_NAMES[action_mode(action_id)],
                "target_waypoint": step["target_waypoint"],
                "executed_actions": executed,
                "collision_count": step["feedback"].collision_count,
                "blocked_forward_count": step["feedback"].blocked_forward_count,
                "waypoint_execution_error_m": step["waypoint_execution_error_m"],
            })
            obs = step["obs"]
            last_info = step["info"]
            last_truncated = step["truncated"]
            if step["terminal"]:
                break
        if last_info is None or "episode" not in last_info:
            raise RuntimeError("G8 development evaluation lacks terminal metrics")
        metrics = {key: metric_value(value) for key, value in last_info["episode"].items()}
        return {
            **spec,
            "success": metrics["success"],
            "spl": metrics["spl"],
            "navigation_error_m": metrics["distance_to_goal"],
            "path_length_m": metrics["path_length"],
            "collision_count": int(metrics["collision_count"]),
            "blocked_forward_count": blocked_count,
            "primitive_steps": len(actions),
            "decision_count": len(action_ids),
            "stop_called": bool(actions and actions[-1] == 0),
            "timeout": last_truncated,
            "action_ids": action_ids,
            "actions": actions,
            "waypoint_execution_error_m": sum(waypoint_errors) / len(waypoint_errors) if waypoint_errors else None,
            "decisions": decisions,
            "topology": topology.summary(),
        }
    finally:
        env.close()


def aggregate_evaluation(episodes: list[dict]) -> dict:
    action_ids = [value for row in episodes for value in row["action_ids"]]
    actions = [value for row in episodes for value in row["actions"]]
    waypoint_errors = [row["waypoint_execution_error_m"] for row in episodes if row["waypoint_execution_error_m"] is not None]
    mode_counts = Counter(MODE_NAMES[action_mode(value)] for value in action_ids)
    return {
        "episode_count": len(episodes),
        "scene_count": len({row["scene"] for row in episodes}),
        "success_rate": sum(row["success"] for row in episodes) / len(episodes),
        "spl": sum(row["spl"] for row in episodes) / len(episodes),
        "navigation_error_m": sum(row["navigation_error_m"] for row in episodes) / len(episodes),
        "path_length_m": sum(row["path_length_m"] for row in episodes) / len(episodes),
        "collision_count": sum(row["collision_count"] for row in episodes) / len(episodes),
        "blocked_forward_count": sum(row["blocked_forward_count"] for row in episodes) / len(episodes),
        "stop_count": sum(row["stop_called"] for row in episodes),
        "timeout_count": sum(row["timeout"] for row in episodes),
        "action_id_counts": {str(value): action_ids.count(value) for value in sorted(set(action_ids))},
        "mode_counts": dict(mode_counts),
        "primitive_counts": {str(value): actions.count(value) for value in range(4)},
        "waypoint_execution_error_m": sum(waypoint_errors) / len(waypoint_errors) if waypoint_errors else None,
    }


def evaluate(manager, core, run_cfg: dict, variant: str) -> dict:
    before = state_sha256(parameter_snapshot(core))
    episodes = [
        evaluate_episode(manager, core, run_cfg, spec, variant)
        for spec in run_cfg["arm"]["development"]
    ]
    after = state_sha256(parameter_snapshot(core))
    if before != after:
        raise RuntimeError("G8 evaluation changed policy parameters")
    return {
        "variant": variant,
        "split": "val_seen",
        "deterministic": True,
        "episodes": episodes,
        "aggregate": aggregate_evaluation(episodes),
        "state_unchanged": True,
    }


def collect_rl_episode(manager, core, run_cfg: dict, spec: dict, slot: int) -> dict:
    training = run_cfg["manifest"]["training"]
    max_decisions = training["max_rl_decisions"]
    env = make_env(run_cfg, split="train", episode_index=spec["index"], max_steps=max_decisions * 4)
    topology = OnlineTopology(max_decisions=max_decisions)
    decisions = []
    try:
        obs, identity = env.reset()
        validate_identity(identity, spec, "RL collection")
        final_obs = None
        final_terminated = False
        for decision in range(max_decisions):
            with torch.no_grad():
                state = manager.model(operation="encode_state", obs=obs)
                top = topology_tensor(topology, model_device(core))
                sampled = manager.model(
                    operation="sample_g8",
                    state_features=state,
                    topology_features=top,
                    seed=stable_seed(run_cfg["seed"], "g8_rl", spec["episode_id"], decision),
                    deterministic=False,
                )
            action_id = int(sampled["action_ids"].reshape(-1)[0])
            step = execute_action(env, topology, action_id, decision)
            feedback = step["feedback"]
            reward, duration = collapse_chunk_rewards(
                [step["primitive_rewards"]],
                [list(feedback.executed_mask)],
                training["gamma_per_primitive"],
            )
            decisions.append({
                "state": state.detach().cpu().float(),
                "topology": top.detach().cpu().float(),
                "action_id": action_id,
                "behavior_log_prob": float(sampled["behavior_log_probs"].reshape(-1)[0]),
                "reference_action_logits": sampled["action_logits"].reshape(-1, ACTION_COUNT)[0].detach().cpu().float(),
                "value": float(sampled["values"].reshape(-1)[0]),
                "reward": float(reward[0]),
                "duration": int(duration[0]),
                "terminated": step["terminated"],
                "truncated": step["truncated"],
                "executor_feedback": feedback.to_dict(),
            })
            obs = step["obs"]
            final_obs = obs
            final_terminated = step["terminated"]
            if step["terminal"]:
                break
        if not decisions:
            raise RuntimeError("G8 RL collection produced no decisions")
        if final_terminated:
            bootstrap = 0.0
        else:
            if final_obs is None or not bool(final_obs["terminal_frozen"].item()):
                raise RuntimeError("G8 RL timeout lacks frozen final observation")
            with torch.no_grad():
                state = manager.model(operation="encode_bootstrap_state", obs=final_obs)
                top = topology_tensor(topology, model_device(core))
                sampled = manager.model(
                    operation="sample_g8",
                    state_features=state,
                    topology_features=top,
                    seed=stable_seed(run_cfg["seed"], "g8_rl_bootstrap", spec["episode_id"]),
                    deterministic=True,
                )
                bootstrap = float(sampled["values"].reshape(-1)[0])
        values = [row["value"] for row in decisions]
        next_values = values[1:] + [bootstrap]
        advantages, returns, deltas = variable_duration_gae(
            [row["reward"] for row in decisions],
            values,
            next_values,
            [row["duration"] for row in decisions],
            [row["terminated"] for row in decisions],
            training["gamma_per_primitive"],
            training["gae_lambda_per_macro"],
        )
        for row, advantage, return_value, delta in zip(decisions, advantages, returns, deltas):
            row.update({"advantage": advantage, "return": return_value, "delta": delta})
        return {**spec, "slot": slot, "bootstrap": bootstrap, "decisions": decisions}
    finally:
        env.close()


def categorical_kl(reference_logits: torch.Tensor, current_logits: torch.Tensor) -> torch.Tensor:
    reference_log_probs = reference_logits.float().log_softmax(-1)
    return (reference_log_probs.exp() * (reference_log_probs - current_logits.float().log_softmax(-1))).sum(-1)


def train_rl(manager, core, run_cfg: dict, bc_tensors: dict, balanced: list[int]):
    training = run_cfg["manifest"]["training"]
    collection_version = core.actor_version
    episodes = [
        collect_rl_episode(manager, core, run_cfg, spec, slot)
        for slot, spec in enumerate(run_cfg["arm"]["train"])
    ]
    if core.actor_version != collection_version:
        raise RuntimeError("G8 policy changed during complete rollout collection")
    rows = [row for episode in episodes for row in episode["decisions"]]
    device = model_device(core)
    states = torch.stack([row["state"] for row in rows]).to(device)
    tops = torch.stack([row["topology"] for row in rows]).to(device)
    actions = torch.tensor([row["action_id"] for row in rows], dtype=torch.int64, device=device)
    behavior = torch.tensor([row["behavior_log_prob"] for row in rows], dtype=torch.float32, device=device)
    advantages = torch.tensor([row["advantage"] for row in rows], dtype=torch.float32)
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
    returns = torch.tensor([row["return"] for row in rows], dtype=torch.float32)
    reference_logits = torch.stack([row["reference_action_logits"] for row in rows]).to(device)
    with torch.no_grad():
        initial = manager.model(operation="replay_g8", state_features=states, topology_features=tops, action_ids=actions)
        preupdate_ratio = float((torch.exp(initial["log_probs"].float() - behavior) - 1.0).abs().max())
    if preupdate_ratio > 0.03:
        raise RuntimeError("G8 algorithm pre-update ratio exceeds engineering guard")

    before = parameter_snapshot(core)
    frozen_before, _ = frozen_parameter_sentinels(core)
    scope = optimizer_parameter_scope(manager, core)
    updates = []
    update_index = 0
    for epoch in range(training["ppo_epochs"]):
        generator = torch.Generator(device="cpu").manual_seed(run_cfg["seed"] + 500 + epoch)
        order = torch.randperm(len(rows), generator=generator).tolist()
        for start in range(0, len(order), training["ppo_batch_size"]):
            indices = order[start:start + training["ppo_batch_size"]]
            replay = manager.model(
                operation="replay_g8",
                state_features=states[indices],
                topology_features=tops[indices],
                action_ids=actions[indices],
            )
            ratios = torch.exp(replay["log_probs"].float() - behavior[indices])
            batch_advantages = advantages[indices].to(device)
            clipped = ratios.clamp(1.0 - training["clip_ratio"], 1.0 + training["clip_ratio"])
            actor_loss = -torch.minimum(ratios * batch_advantages, clipped * batch_advantages).mean()
            value_loss = 0.5 * (replay["values"].float() - returns[indices].to(device)).square().mean()
            entropy = replay["entropy"].float().mean()
            bc_indices = [balanced[(update_index * training["ppo_batch_size"] + offset) % len(balanced)] for offset in range(len(indices))]
            bc = manager.model(
                operation="bc_loss_g8",
                state_features=bc_tensors["state_features"][bc_indices].to(device),
                topology_features=bc_tensors["topology_features"][bc_indices].to(device),
                target_action_ids=bc_tensors["target_action_ids"][bc_indices].to(device),
                target_waypoints=bc_tensors["target_waypoints"][bc_indices].to(device),
            )
            loss = (
                actor_loss
                + training["value_coefficient"] * value_loss
                - training["entropy_coefficient"] * entropy
                + training["bc_auxiliary_coefficient"] * bc["loss"]
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("G8 PPO loss is nonfinite")
            manager.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            audit = gradient_audit(core)
            if not audit["finite"] or audit["group_norm_sums"]["high_level_policy"] <= 0 or audit["group_norm_sums"]["value_head"] <= 0 or audit["frozen_with_grad"]:
                raise RuntimeError("G8 PPO gradient scope failed")
            grad_norm, learning_rates = manager.optimizer_step()
            core.mark_updated()
            updates.append({
                "update": update_index + 1,
                "sample_count": len(indices),
                "loss": float(loss.detach()),
                "actor_loss": float(actor_loss.detach()),
                "value_loss": float(value_loss.detach()),
                "entropy": float(entropy.detach()),
                "bc_auxiliary_loss": float(bc["loss"].detach()),
                "ratio_mean": float(ratios.detach().mean()),
                "ratio_max_deviation": float((ratios.detach() - 1.0).abs().max()),
                "grad_norm": float(torch.as_tensor(grad_norm)),
                "learning_rates": [float(value) for value in learning_rates],
            })
            update_index += 1
    with torch.no_grad():
        final = manager.model(operation="replay_g8", state_features=states, topology_features=tops, action_ids=actions)
        final_ratios = torch.exp(final["log_probs"].float() - behavior)
        final_kl = categorical_kl(reference_logits, final["action_logits"])
    frozen_after = {
        name.replace("_fsdp_wrapped_module.", ""): parameter.detach().cpu().clone()
        for name, parameter in core.named_parameters()
        if name.replace("_fsdp_wrapped_module.", "") in frozen_before
    }
    serialized_episodes = []
    for episode in episodes:
        item = {key: value for key, value in episode.items() if key != "decisions"}
        item["decisions"] = [
            {key: value for key, value in row.items() if key not in ("state", "topology", "reference_action_logits")}
            for row in episode["decisions"]
        ]
        serialized_episodes.append(item)
    return {
        "collection_policy_version": collection_version,
        "all_episodes_collected_before_update": True,
        "episode_count": len(episodes),
        "scene_count": len({row["scene"] for row in episodes}),
        "decision_count": len(rows),
        "preupdate_ratio_max_deviation": preupdate_ratio,
        "update_count": len(updates),
        "updates": updates,
        "final_collection_ratio_max_deviation": float((final_ratios - 1.0).abs().max()),
        "final_collection_categorical_kl_mean": float(final_kl.mean()),
        "final_collection_categorical_kl_max": float(final_kl.max()),
        "optimizer_scope": scope,
        "parameter_changes": parameter_changes(core, before),
        "frozen_sentinels_unchanged": state_sha256(frozen_before) == state_sha256(frozen_after),
        "episodes": serialized_episodes,
    }


def validate_manifest(manifest: dict, seed: int) -> dict:
    if manifest["schema_version"] != 1 or manifest["stage"] != "G8 matched BC versus BC+RL algorithm gate":
        raise ValueError("G8 algorithm manifest identity changed")
    if manifest["source"]["val_unseen_read"] or manifest["claim_boundary"]["val_unseen_access"]:
        raise ValueError("G8 algorithm manifest opened val_unseen")
    if seed not in manifest["seeds"] or str(seed) not in manifest["arms"]:
        raise ValueError("G8 seed is not preregistered")
    arm = manifest["arms"][str(seed)]
    if (
        len(arm["train"]) != 32
        or len({row["scene"] for row in arm["train"]}) < 16
        or len(arm["development"]) != 22
        or len({row["scene"] for row in arm["development"]}) != 22
        or not manifest["training"]["bc_stop_balance_required"]
        or manifest["evaluation"]["split"] != "val_seen"
        or manifest["evaluation"]["val_unseen_episode_count"] != 0
    ):
        raise ValueError("G8 algorithm arm or budget changed")
    return arm


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rlinf-root", type=Path, required=True)
    parser.add_argument("--stream-source", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--vision-tower", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--habitat-python", type=Path, required=True)
    parser.add_argument("--habitat-server", type=Path, required=True)
    parser.add_argument("--habitat-sim-python", type=Path, required=True)
    parser.add_argument("--habitat-env-lib", type=Path, required=True)
    parser.add_argument("--egl-vendor-json", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--master-port", type=int, required=True)
    parser.add_argument("--habitat-startup-timeout-s", type=float, default=180.0)
    parser.add_argument("--context-tokens", type=int, default=4096)
    parser.add_argument("--head-hidden", type=int, default=256)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not math.isfinite(args.habitat_startup_timeout_s) or not 0 < args.habitat_startup_timeout_s <= 3600:
        raise ValueError("--habitat-startup-timeout-s must be finite and in (0, 3600]")
    started = time.time()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = args.manifest.expanduser().resolve(strict=True)
    if sha256_file(manifest_path) != MANIFEST_SHA256:
        raise RuntimeError("G8 algorithm manifest hash changed")
    manifest = json.loads(manifest_path.read_text())
    arm = validate_manifest(manifest, args.seed)
    assert_g8_action_contract()
    path_names = (
        "rlinf_root", "stream_source", "base_model", "tokenizer", "vision_tower", "asset_root",
        "habitat_python", "habitat_server", "habitat_sim_python", "habitat_env_lib", "egl_vendor_json",
    )
    paths = {name: str(Path(getattr(args, name)).expanduser().resolve(strict=True)) for name in path_names}
    source_heads = {"stream": git_head(Path(paths["stream_source"])), "rlinf": git_head(Path(paths["rlinf_root"]))}
    if source_heads != {"stream": ORIGINAL_STREAM_COMMIT, "rlinf": RLINF_COMMIT}:
        raise RuntimeError("G8 algorithm source commit changed")
    train_path = Path(manifest["source"]["train_path"])
    val_seen_path = Path(manifest["source"]["val_seen_path"])
    if sha256_file(train_path) != manifest["source"]["train_sha256"] or sha256_file(val_seen_path) != manifest["source"]["val_seen_sha256"]:
        raise RuntimeError("G8 algorithm dataset hash changed")
    training = manifest["training"]
    run_cfg = {
        **paths,
        "stream_commit": ORIGINAL_STREAM_COMMIT,
        "rlinf_commit": RLINF_COMMIT,
        "manifest": manifest,
        "arm": arm,
        "seed": args.seed,
        "master_port": args.master_port,
        "context_tokens": args.context_tokens,
        "head_hidden": args.head_hidden,
        "habitat_startup_timeout_s": args.habitat_startup_timeout_s,
        "learning_rate": training["learning_rate"],
        "value_learning_rate": training["value_learning_rate"],
    }
    config = {
        "kind": "streamvln-g8-matched-bc-vs-bc-rl",
        "manifest": str(manifest_path),
        "manifest_sha256": MANIFEST_SHA256,
        "seed": args.seed,
        "source_heads": source_heads,
        "run": run_cfg,
        "model_load_count": 1,
        "val_unseen_episode_count": 0,
        "model_input_inventory": {
            "base_model": small_file_inventory(Path(paths["base_model"])),
            "tokenizer": small_file_inventory(Path(paths["tokenizer"])),
            "vision_tower": small_file_inventory(Path(paths["vision_tower"])),
        },
    }
    config_path = output / "run_config.json"
    write_json(config_path, config)

    manager = None
    try:
        estimated_bc_updates = training["bc_epochs"] * math.ceil(2048 / training["bc_batch_size"])
        estimated_ppo_updates = math.ceil(len(arm["train"]) * training["max_rl_decisions"] / training["ppo_batch_size"])
        manager, core, trainable_manifest = setup_manager(run_cfg, estimated_bc_updates + estimated_ppo_updates)
        initial_state_sha = state_sha256(parameter_snapshot(core))
        bc_tensors, balanced, supervision = collect_bc_dataset(manager, core, run_cfg)
        bc_training = train_bc(manager, core, run_cfg, bc_tensors, balanced)
        bc_checkpoint = save_head_checkpoint(
            core,
            output / "checkpoint_bc",
            {"schema": "streamvln_g8_v1", "variant": "bc", "seed": args.seed, "val_unseen_used": False},
        )
        bc_load_audit = checkpoint_load_audit(core, Path(bc_checkpoint["policy"]))
        bc_evaluation = evaluate(manager, core, run_cfg, "topology_waypoint_bc")
        write_json(output / "eval_bc.json", bc_evaluation)

        rl_training = train_rl(manager, core, run_cfg, bc_tensors, balanced)
        rl_checkpoint = save_head_checkpoint(
            core,
            output / "checkpoint_bc_rl",
            {"schema": "streamvln_g8_v1", "variant": "bc_rl", "seed": args.seed, "val_unseen_used": False},
        )
        rl_load_audit = checkpoint_load_audit(core, Path(rl_checkpoint["policy"]))
        rl_evaluation = evaluate(manager, core, run_cfg, "topology_waypoint_bc_rl")
        write_json(output / "eval_bc_rl.json", rl_evaluation)
        bc_agg, rl_agg = bc_evaluation["aggregate"], rl_evaluation["aggregate"]
        paired = {
            "success_rate_delta": rl_agg["success_rate"] - bc_agg["success_rate"],
            "spl_delta": rl_agg["spl"] - bc_agg["spl"],
            "navigation_error_m_delta": rl_agg["navigation_error_m"] - bc_agg["navigation_error_m"],
            "path_length_m_delta": rl_agg["path_length_m"] - bc_agg["path_length_m"],
            "collision_count_delta": rl_agg["collision_count"] - bc_agg["collision_count"],
            "blocked_forward_count_delta": rl_agg["blocked_forward_count"] - bc_agg["blocked_forward_count"],
            "stop_count_delta": rl_agg["stop_count"] - bc_agg["stop_count"],
            "timeout_count_delta": rl_agg["timeout_count"] - bc_agg["timeout_count"],
        }
        guards = {
            "manifest_hash": sha256_file(manifest_path) == MANIFEST_SHA256,
            "source_pins": source_heads == {"stream": ORIGINAL_STREAM_COMMIT, "rlinf": RLINF_COMMIT},
            "source_scope": (
                trainable_manifest["source_kind"] == "upstream-original"
                and not trainable_manifest["memory_adapter"]
                and not trainable_manifest["secondary_post_training_used"]
                and not trainable_manifest["backbone_trainable"]
                and not trainable_manifest["gaussian_waypoint_likelihood"]
            ),
            "identity_counts": (
                supervision["episode_count"] == 32
                and supervision["scene_count"] >= 16
                and bc_agg["episode_count"] == rl_agg["episode_count"] == 22
                and bc_agg["scene_count"] == rl_agg["scene_count"] == 22
            ),
            "bc_stop_balance": supervision["balanced_stop_count"] == supervision["mode_counts"]["navigate"],
            "bc_parameters_changed": bc_training["parameter_changes"]["changed_tensor_count"] > 0,
            "checkpoint_roundtrip": bc_load_audit["passed"] and rl_load_audit["passed"],
            "complete_collection_before_rl": rl_training["all_episodes_collected_before_update"],
            "preupdate_ratio": rl_training["preupdate_ratio_max_deviation"] <= 0.03,
            "final_ratio": rl_training["final_collection_ratio_max_deviation"] <= training["final_collection_ratio_max_deviation"],
            "final_categorical_kl": rl_training["final_collection_categorical_kl_max"] <= training["final_collection_categorical_kl_max"],
            "optimizer_scope": (
                rl_training["optimizer_scope"]["optimizer_equals_trainable"]
                and rl_training["optimizer_scope"]["optimizer_frozen_overlap_count"] == 0
            ),
            "rl_parameters_changed": rl_training["parameter_changes"]["changed_tensor_count"] > 0,
            "frozen_sentinels": rl_training["frozen_sentinels_unchanged"],
            "matched_development_schedule": [row["episode_id"] for row in bc_evaluation["episodes"]] == [row["episode_id"] for row in rl_evaluation["episodes"]],
            "zero_val_unseen": manifest["evaluation"]["val_unseen_episode_count"] == 0,
        }
        result = {
            "schema_version": 1,
            "kind": config["kind"],
            "seed": args.seed,
            "all_training_and_evaluation_guards_passed": all(guards.values()),
            "algorithm_gate_member": bool(
                all(guards.values())
                and paired["success_rate_delta"] > 0
                and paired["spl_delta"] >= 0
                and paired["timeout_count_delta"] <= 0
                and len([value for value in rl_agg["mode_counts"].values() if value > 0]) >= 2
                and len([value for value in rl_agg["primitive_counts"].values() if value > 0]) >= 2
            ),
            "source_contract": {
                "stream_commit": ORIGINAL_STREAM_COMMIT,
                "rlinf_commit": RLINF_COMMIT,
                "ttt": False,
                "secondary_post_training": False,
                "val_unseen_used": False,
            },
            "trainable_manifest": trainable_manifest,
            "initial_state_sha256": initial_state_sha,
            "supervision": supervision,
            "bc_training": bc_training,
            "rl_training": rl_training,
            "checkpoints": {"bc": bc_checkpoint, "bc_rl": rl_checkpoint},
            "checkpoint_load_audits": {"bc": bc_load_audit, "bc_rl": rl_load_audit},
            "development": {"bc": bc_agg, "bc_rl": rl_agg, "paired_delta": paired},
            "guards": guards,
            "counts": {"model_loads": 1, "train_episodes": 32, "development_episodes_per_arm": 22, "val_unseen_episodes": 0},
            "elapsed_seconds": time.time() - started,
        }
        result_path = output / "result.json"
        write_json(result_path, result)
        print(json.dumps({
            "seed": args.seed,
            "all_guards": result["all_training_and_evaluation_guards_passed"],
            "algorithm_gate_member": result["algorithm_gate_member"],
            "development": result["development"],
            "result": str(result_path),
        }, indent=2, sort_keys=True))
        if not result["all_training_and_evaluation_guards_passed"]:
            raise SystemExit(2)
    finally:
        if manager is not None:
            cleanup_manager(manager)


if __name__ == "__main__":
    main()
