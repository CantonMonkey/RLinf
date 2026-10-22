"""Bounded, single-update PPO correctness run for original StreamVLN in RLinf FSDP."""

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

from rlinf.envs import get_env_cls
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager
from rlinf.models import get_model

from streamvln_chunk_ppo import collapse_chunk_rewards, variable_duration_gae


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
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def make_policy_cfg(cfg: dict, *, trainable: bool, adapter_checkpoint=None):
    return OmegaConf.create(
        {
            "model_type": "streamvln_ppo",
            "model_path": "",
            "stream_source": cfg["stream_source"],
            "base_model": cfg["base_model"],
            "tokenizer": cfg["tokenizer"],
            "vision_tower": cfg["vision_tower"],
            "expected_source_commit": cfg["stream_commit"],
            "adapter_checkpoint": adapter_checkpoint,
            "trainable": trainable,
            "action_dim": 1,
            "discrete_n_actions": 4,
            "num_action_chunks": 4,
            "temperature": 1.0,
            "context_tokens": cfg["context_tokens"],
            "seed": cfg["seed"],
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_layers": [24, 25, 26, 27],
            "lora_targets": ["q_proj", "v_proj"],
            "precision": "bf16",
            "add_value_head": True,
            # The policy creates exact-path LoRA internally. True here would
            # trigger RLinf's broad generic PEFT wrapper a second time.
            "is_lora": False,
            "lora_rank_legacy": 8,
            "load_to_device": False,
        }
    )


class G3FSDPManager(FSDPModelManager):
    def model_provider_func(self):
        model = get_model(self._cfg.model)
        if model is None:
            raise RuntimeError("streamvln_ppo is not registered in RLinf")
        return model


def make_manager_cfg(run_cfg: dict):
    return OmegaConf.create(
        {
            "model": make_policy_cfg(run_cfg, trainable=True),
            "optim": {
                "lr": run_cfg["learning_rate"],
                "value_lr": run_cfg["value_learning_rate"],
                "adam_beta1": 0.9,
                "adam_beta2": 0.999,
                "adam_eps": 1.0e-8,
                "weight_decay": 0.0,
                "clip_grad": 1.0,
                "total_training_steps": 1,
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


def make_env(run_cfg: dict):
    env_cfg = OmegaConf.create(
        {
            "env_type": "streamvln",
            "seed": 7,
            "auto_reset": False,
            "max_episode_steps": 40,
            "python_bin": run_cfg["habitat_python"],
            "server_script": run_cfg["habitat_server"],
            "asset_root": run_cfg["asset_root"],
            "dataset_split": "train",
            "episode_index": run_cfg["episode_index"],
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
        }
    )
    env_cls = get_env_cls("streamvln", OmegaConf.create({}))
    return env_cls(env_cfg, 1, 0, 1, None)


def sampled_frozen_tensors(core) -> dict:
    candidates = {
        "vision": lambda name: "vision_tower" in name and name.endswith("weight"),
        "projector": lambda name: "mm_projector" in name and name.endswith("weight"),
        "embedding": lambda name: "embed_tokens.weight" in name,
        "lm_head": lambda name: name.endswith("lm_head.weight"),
        "qwen_base": lambda name: (
            "model.norm.weight" in name
            and "vision_tower" not in name
            and "lora_" not in name
        ),
    }
    named = list(core.named_parameters())
    result = {}
    for label, predicate in candidates.items():
        matches = [(name, parameter) for name, parameter in named if predicate(name)]
        if not matches:
            raise RuntimeError(f"cannot find frozen representative {label}")
        name, parameter = matches[0]
        if parameter.requires_grad:
            raise RuntimeError(f"frozen representative {name} is trainable")
        flat = parameter.detach().reshape(-1)
        take = min(2048, flat.numel())
        sample = torch.cat((flat[:take], flat[-take:])).cpu().clone()
        result[label] = {"name": name, "sample": sample}
    return result


def compare_frozen(core, before: dict) -> dict:
    named = dict(core.named_parameters())
    checks = {}
    for label, entry in before.items():
        after = named[entry["name"]].detach().reshape(-1)
        take = entry["sample"].numel() // 2
        sample = torch.cat((after[:take], after[-take:])).cpu()
        checks[label] = bool(torch.equal(entry["sample"], sample))
    return checks


def gradient_audit(core) -> dict:
    sums = {"lora": 0.0, "value_head": 0.0}
    finite = True
    nonzero_tensors = {"lora": 0, "value_head": 0}
    frozen_with_grad = []
    for name, parameter in core.named_parameters():
        grad = parameter.grad
        if parameter.requires_grad:
            group = "value_head" if name.startswith("value_head.") else "lora"
            if grad is None:
                continue
            finite = finite and bool(torch.isfinite(grad).all())
            norm = float(grad.detach().float().norm())
            sums[group] += norm * norm
            if norm > 0:
                nonzero_tensors[group] += 1
        elif grad is not None:
            frozen_with_grad.append(name)
    return {
        "finite": finite,
        "group_norms": {key: math.sqrt(value) for key, value in sums.items()},
        "nonzero_tensors": nonzero_tensors,
        "frozen_with_grad": frozen_with_grad,
    }


def trainable_snapshot(core):
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in core.named_parameters()
        if parameter.requires_grad
    }


def trainable_changes(core, before: dict) -> dict:
    named = dict(core.named_parameters())
    groups = {"lora": [], "value_head": []}
    for name, previous in before.items():
        delta = (named[name].detach().cpu().float() - previous.float()).abs().max().item()
        group = "value_head" if name.startswith("value_head.") else "lora"
        groups[group].append((name, delta))
    return {
        group: {
            "changed_tensors": sum(delta > 0 for _, delta in rows),
            "max_abs_change": max((delta for _, delta in rows), default=0.0),
        }
        for group, rows in groups.items()
    }


def init_distributed(port: int):
    os.environ.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "RANK": "0",
            "WORLD_SIZE": "1",
            "LOCAL_RANK": "0",
        }
    )
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", rank=0, world_size=1)


def canonical_lora_state(backbone) -> dict[str, torch.Tensor]:
    """Return PEFT adapter keys without FSDP's internal wrapper segment."""
    state = {}
    for name, parameter in backbone.named_parameters():
        if "lora_" not in name:
            continue
        key = name.replace("._fsdp_wrapped_module.", ".")
        key = key.replace(".lora_A.default.", ".lora_A.")
        key = key.replace(".lora_B.default.", ".lora_B.")
        if key in state:
            raise RuntimeError(f"duplicate canonical LoRA key: {key}")
        state[key] = parameter.detach().cpu().contiguous()
    tensor_count = len(state)
    parameter_count = sum(value.numel() for value in state.values())
    if tensor_count != 16 or parameter_count != 360448:
        raise RuntimeError(
            f"unexpected LoRA checkpoint payload: {tensor_count} tensors, "
            f"{parameter_count} parameters"
        )
    return state


def save_checkpoint(core, checkpoint: Path, metadata: dict):
    adapter_dir = checkpoint / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    # PEFT cannot filter decoder-layer keys after FSDP adds its wrapper segment.
    # Let it write the config, then persist the canonical adapter tensor names.
    core.backbone.save_pretrained(adapter_dir, safe_serialization=True)
    lora_state = canonical_lora_state(core.backbone)
    adapter_file = adapter_dir / "adapter_model.safetensors"
    save_safetensors(lora_state, adapter_file)
    persisted = load_safetensors(adapter_file)
    if set(persisted) != set(lora_state):
        raise RuntimeError("persisted LoRA checkpoint keys differ from source")
    metadata["adapter_tensor_count"] = len(persisted)
    metadata["adapter_parameter_count"] = sum(
        value.numel() for value in persisted.values()
    )
    torch.save(
        {key: value.detach().cpu() for key, value in core.value_head.state_dict().items()},
        checkpoint / "value_head.pt",
    )
    (checkpoint / "metadata.json").write_text(
        json.dumps(jsonable(metadata), indent=2, sort_keys=True) + "\n"
    )
    files = sorted(path for path in checkpoint.rglob("*") if path.is_file())
    return {str(path.relative_to(checkpoint)): sha256_file(path) for path in files}


def run_train(run_cfg: dict, run_config_path: Path):
    output = Path(run_cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(run_cfg["seed"])
    torch.cuda.manual_seed_all(run_cfg["seed"])
    init_distributed(run_cfg["master_port"])
    manager = None
    env = None
    train_record = {}
    started = time.time()
    try:
        manager = G3FSDPManager(make_manager_cfg(run_cfg), world_size=1, rank=0)
        manager.setup_model_and_optimizer()
        if type(manager.model).__name__ != "FullyShardedDataParallel":
            raise RuntimeError("G3 actor is not wrapped by RLinf FSDP")
        core = manager.model.module
        manifest = core.trainable_manifest()
        if manifest["source_kind"] != "upstream-original" or manifest["memory_adapter"]:
            raise RuntimeError("G3 source contract is not original StreamVLN")
        if manifest["trainable_parameter_count"] != 364033:
            raise RuntimeError("unexpected G3 trainable parameter count")
        if any("vision_tower" in name for name in manifest["trainable_names"]):
            raise RuntimeError("SigLIP received trainable parameters")

        trainable_before = trainable_snapshot(core)
        frozen_before = sampled_frozen_tensors(core)
        env = make_env(run_cfg)
        obs, _ = env.reset()
        torch.save({key: value.cpu() for key, value in obs.items()}, output / "observation.pt")

        with torch.no_grad():
            sampled = manager.model(operation="sample", obs=obs)
        action_mask = sampled["executed_mask"].bool()
        duration = int(action_mask.sum())
        if not 1 <= duration <= 4:
            raise RuntimeError("invalid sampled action duration")
        actions = sampled["actions"]
        obs_list, rewards, terminations, truncations, infos = env.chunk_step(
            actions.detach().cpu().reshape(1, 4, 1)
        )
        next_obs = obs_list[0]
        terminal = bool(torch.logical_or(terminations, truncations).any())

        replay = manager.model(operation="replay")
        with torch.no_grad():
            bootstrap = manager.model(
                operation="bootstrap",
                obs=next_obs,
                executed_actions=actions[:duration],
                terminal=terminal,
            )
        current_chunk_logprob = replay["log_probs"][:duration].float().sum()
        behavior_chunk_logprob = sampled["behavior_log_probs"][action_mask].float().sum()
        ratio = torch.exp(current_chunk_logprob - behavior_chunk_logprob)
        ratio_deviation = float((ratio - 1.0).abs())
        if ratio_deviation > 0.03:
            raise RuntimeError(f"on-policy pre-update ratio deviates by {ratio_deviation}")

        chunk_rewards, durations = collapse_chunk_rewards(
            rewards.detach().cpu().tolist(),
            [action_mask.detach().cpu().tolist()],
            run_cfg["gamma"],
        )
        advantages, returns, deltas = variable_duration_gae(
            chunk_rewards,
            [float(replay["state_value"].detach())],
            [float(bootstrap.detach())],
            durations,
            [terminal],
            run_cfg["gamma"],
            run_cfg["gae_lambda"],
        )
        advantage = torch.tensor(
            advantages[0], dtype=torch.float32, device=current_chunk_logprob.device
        )
        target_return = torch.tensor(
            returns[0], dtype=torch.float32, device=current_chunk_logprob.device
        )
        clipped_ratio = ratio.clamp(1.0 - run_cfg["clip_ratio"], 1.0 + run_cfg["clip_ratio"])
        actor_loss = -torch.minimum(ratio * advantage, clipped_ratio * advantage)
        value_loss = 0.5 * (replay["state_value"].float() - target_return).square()
        entropy = replay["entropy"][:duration].float().mean()
        loss = actor_loss + run_cfg["value_coef"] * value_loss - run_cfg["entropy_coef"] * entropy
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("G3 PPO loss is non-finite")
        loss.backward()

        grad_audit = gradient_audit(core)
        if (
            not grad_audit["finite"]
            or grad_audit["group_norms"]["lora"] <= 0
            or grad_audit["group_norms"]["value_head"] <= 0
            or grad_audit["frozen_with_grad"]
        ):
            raise RuntimeError("G3 gradient isolation audit failed")

        grad_norm, lr_list = manager.optimizer_step()
        core.mark_updated()
        changes = trainable_changes(core, trainable_before)
        frozen_checks = compare_frozen(core, frozen_before)
        if changes["lora"]["changed_tensors"] < 1:
            raise RuntimeError("no LoRA tensor changed")
        if changes["value_head"]["changed_tensors"] < 1:
            raise RuntimeError("value head did not change")
        if not all(frozen_checks.values()):
            raise RuntimeError("a frozen representative changed")

        with torch.no_grad():
            post = manager.model(operation="replay")
        post_log_probs = post["log_probs"][:duration].detach().cpu()
        post_value = post["state_value"].detach().cpu()
        checkpoint = output / "checkpoint_step_1"
        metadata = {
            "schema": "streamvln_original_rlinf_ppo_g3_v1",
            "actor_version": core.actor_version,
            "behavior_version": int(sampled["behavior_version"]),
            "source_kind": "upstream-original",
            "stream_source_commit": run_cfg["stream_commit"],
            "rlinf_source_commit": run_cfg["rlinf_commit"],
            "lora_layers": [24, 25, 26, 27],
            "lora_targets": ["q_proj", "v_proj"],
            "memory_adapter": False,
        }
        checkpoint_hashes = save_checkpoint(core, checkpoint, metadata)
        torch.save(
            {
                "sampled_tokens": sampled["sampled_tokens"].detach().cpu(),
                "proposed_actions": actions[:duration].detach().cpu(),
                "post_log_probs": post_log_probs,
                "post_value": post_value,
            },
            output / "replay_payload.pt",
        )

        train_record = {
            "schema_version": 1,
            "kind": "rlinf-streamvln-original-g3-single-update",
            "source_contract": {
                "stream_source": run_cfg["stream_source"],
                "stream_commit": run_cfg["stream_commit"],
                "rlinf_root": run_cfg["rlinf_root"],
                "rlinf_commit": run_cfg["rlinf_commit"],
                "base_model": run_cfg["base_model"],
                "secondary_post_training_used": False,
                "memory_adapter": False,
            },
            "fsdp": {
                "manager": type(manager).__name__,
                "wrapper": type(manager.model).__name__,
                "strategy": "fsdp",
                "world_size": 1,
                "sharding_strategy": "no_shard",
                "use_orig_params": True,
            },
            "manifest": manifest,
            "rollout": {
                "episode_index": run_cfg["episode_index"],
                "actions": actions.detach().cpu().tolist(),
                "executed_mask": action_mask.detach().cpu().tolist(),
                "duration": duration,
                "primitive_rewards": rewards.detach().cpu().tolist(),
                "chunk_reward": chunk_rewards[0],
                "terminal": terminal,
                "info": jsonable(infos[0]),
                "behavior_version": int(sampled["behavior_version"]),
                "actor_version_before": int(replay["actor_version"]),
                "pre_update_ratio": float(ratio.detach()),
                "pre_update_ratio_deviation": ratio_deviation,
            },
            "advantage": {
                "gamma": run_cfg["gamma"],
                "gae_lambda": run_cfg["gae_lambda"],
                "formula": {
                    "chunk_reward": "sum_j gamma^j r_j over executed prefix",
                    "bootstrap": "gamma^k V_next",
                    "trace": "(gamma*lambda)^k",
                },
                "duration": durations[0],
                "bootstrap_value": float(bootstrap),
                "state_value": float(replay["state_value"].detach()),
                "delta": deltas[0],
                "advantage": advantages[0],
                "return": returns[0],
            },
            "loss": {
                "total": float(loss.detach()),
                "actor": float(actor_loss.detach()),
                "value": float(value_loss.detach()),
                "entropy": float(entropy.detach()),
                "grad_norm": float(torch.as_tensor(grad_norm)),
                "learning_rates": [float(value) for value in lr_list],
            },
            "gradient_audit": grad_audit,
            "parameter_changes": changes,
            "frozen_representatives_unchanged": frozen_checks,
            "version": {
                "actor_after": core.actor_version,
                "checkpoint_version": metadata["actor_version"],
            },
            "checkpoint": {
                "path": str(checkpoint),
                "files_sha256": checkpoint_hashes,
            },
            "elapsed_s_before_independent_eval": time.time() - started,
        }
        (output / "train_result.json").write_text(
            json.dumps(jsonable(train_record), indent=2, sort_keys=True) + "\n"
        )
    finally:
        if env is not None:
            env.close()
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

    subprocess.run(
        [
            sys.executable,
            "-B",
            str(Path(__file__).resolve()),
            "--phase",
            "eval",
            "--run-config",
            str(run_config_path),
        ],
        check=True,
        env=os.environ.copy(),
    )
    eval_record = json.loads((output / "eval_result.json").read_text())
    combined = dict(train_record)
    combined["independent_frozen_eval"] = eval_record
    combined["all_guards_passed"] = bool(
        eval_record["checkpoint_load_passed"]
        and all(combined["frozen_representatives_unchanged"].values())
        and combined["parameter_changes"]["lora"]["changed_tensors"] > 0
        and combined["parameter_changes"]["value_head"]["changed_tensors"] > 0
    )
    combined["elapsed_s_total"] = time.time() - started
    (output / "result.json").write_text(
        json.dumps(jsonable(combined), indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "result": str(output / "result.json"),
        "all_guards_passed": combined["all_guards_passed"],
        "pre_update_ratio": combined["rollout"]["pre_update_ratio"],
        "lora_max_change": combined["parameter_changes"]["lora"]["max_abs_change"],
        "value_max_change": combined["parameter_changes"]["value_head"]["max_abs_change"],
        "checkpoint_eval_error": eval_record["max_logprob_error"],
    }, sort_keys=True))


def run_eval(run_cfg: dict):
    output = Path(run_cfg["output_dir"])
    checkpoint = output / "checkpoint_step_1"
    metadata = json.loads((checkpoint / "metadata.json").read_text())
    payload = torch.load(output / "replay_payload.pt", map_location="cpu", weights_only=True)
    obs = torch.load(output / "observation.pt", map_location="cpu", weights_only=True)
    cfg = make_policy_cfg(
        run_cfg,
        trainable=False,
        adapter_checkpoint=str(checkpoint / "adapter"),
    )
    model = get_model(cfg)
    saved_lora = load_safetensors(checkpoint / "adapter" / "adapter_model.safetensors")
    loaded_lora = canonical_lora_state(model.backbone)
    adapter_keys_match = set(saved_lora) == set(loaded_lora)
    adapter_max_error = float("inf")
    if adapter_keys_match:
        adapter_max_error = max(
            float((loaded_lora[key].float() - saved_lora[key].float()).abs().max())
            for key in saved_lora
        )
    adapter_load_verified = (
        adapter_keys_match
        and len(saved_lora) == 16
        and sum(value.numel() for value in saved_lora.values()) == 360448
        and adapter_max_error == 0.0
        and metadata.get("adapter_tensor_count") == 16
        and metadata.get("adapter_parameter_count") == 360448
    )
    value_state = torch.load(
        checkpoint / "value_head.pt", map_location="cpu", weights_only=True
    )
    model.value_head.load_state_dict(value_state, strict=True)
    model.to("cuda:0")
    model.eval()
    model.actor_version = int(metadata["actor_version"])
    with torch.inference_mode():
        model(
            operation="restore",
            obs=obs,
            sampled_tokens=payload["sampled_tokens"].to("cuda:0"),
            proposed_actions=payload["proposed_actions"].to("cuda:0"),
        )
        replay = model(operation="replay")
    expected = payload["post_log_probs"].float()
    actual = replay["log_probs"].detach().cpu().float()
    max_error = float((actual - expected).abs().max())
    value_error = float(
        (replay["state_value"].detach().cpu().float() - payload["post_value"].float()).abs()
    )
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    result = {
        "separate_process": True,
        "frozen": trainable_count == 0,
        "trainable_parameter_count": trainable_count,
        "loaded_version": int(metadata["actor_version"]),
        "source_kind": metadata["source_kind"],
        "memory_adapter": metadata["memory_adapter"],
        "max_logprob_error": max_error,
        "value_error": value_error,
        "adapter_load_verified": adapter_load_verified,
        "adapter_tensor_count": len(saved_lora),
        "adapter_parameter_count": sum(
            value.numel() for value in saved_lora.values()
        ),
        "adapter_max_weight_error": adapter_max_error,
        "tolerance": 0.03,
        "checkpoint_load_passed": (
            trainable_count == 0
            and metadata["source_kind"] == "upstream-original"
            and metadata["memory_adapter"] is False
            and adapter_load_verified
            and max_error <= 0.03
            and value_error <= 0.03
        ),
        "gpu_allocated_gib": torch.cuda.memory_allocated() / 2**30,
    }
    (output / "eval_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    if not result["checkpoint_load_passed"]:
        raise RuntimeError("independent frozen checkpoint evaluation failed")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("train", "eval"), default="train")
    parser.add_argument("--run-config")
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
    parser.add_argument("--episode-index", type=int, default=1189)
    parser.add_argument("--context-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--master-port", type=int, default=29583)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--value-learning-rate", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.001)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.phase == "eval":
        if not args.run_config:
            raise ValueError("--run-config is required for eval")
        run_cfg = json.loads(Path(args.run_config).read_text())
        run_eval(run_cfg)
        return

    required = (
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
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise ValueError("missing arguments: " + ", ".join(missing))
    roots = {
        name: str(Path(getattr(args, name)).expanduser().resolve(strict=True))
        for name in required
        if name != "output_dir"
    }
    stream_commit = git_head(Path(roots["stream_source"]))
    rlinf_commit = git_head(Path(roots["rlinf_root"]))
    if stream_commit != ORIGINAL_STREAM_COMMIT:
        raise RuntimeError(
            f"G3 requires original StreamVLN {ORIGINAL_STREAM_COMMIT}, got {stream_commit}"
        )
    run_cfg = {
        **roots,
        "output_dir": str(Path(args.output_dir).expanduser().resolve()),
        "stream_commit": stream_commit,
        "rlinf_commit": rlinf_commit,
        "episode_index": args.episode_index,
        "context_tokens": args.context_tokens,
        "seed": args.seed,
        "master_port": args.master_port,
        "learning_rate": args.learning_rate,
        "value_learning_rate": args.value_learning_rate,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_ratio": args.clip_ratio,
        "value_coef": args.value_coef,
        "entropy_coef": args.entropy_coef,
        "egl_vendor_json": str(Path(args.egl_vendor_json).resolve(strict=True)),
    }
    output = Path(run_cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    run_config_path = output / "run_config.json"
    run_config_path.write_text(json.dumps(run_cfg, indent=2, sort_keys=True) + "\n")
    run_train(run_cfg, run_config_path)


if __name__ == "__main__":
    main()
