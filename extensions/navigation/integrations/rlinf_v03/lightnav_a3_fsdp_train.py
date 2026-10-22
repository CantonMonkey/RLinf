#!/usr/bin/env python3
"""Train the LightNav A3 correction head through RLinf FSDP on eight GPUs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel,
    StateDictType,
)
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from omegaconf import OmegaConf

from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager

from lightnav_a3_policy import (
    ADJUST_GATE,
    FEATURE_DIM,
    GATE_NAMES,
    HIDDEN_DIM,
    load_checkpoint,
    save_checkpoint,
    sha256_file,
)


SCHEMA_NAME = "lightnav_a3_rlinf_fsdp_training"
SCHEMA_VERSION = 1
MAX_CHECKPOINT_BYTES = 2 * 1024**3
MAX_RUN_BYTES = 16 * 1024**3


class TrainingContractError(RuntimeError):
    """Raised when training evidence differs from the A3 contract."""


def _atomic_json(path: Path, value: object) -> None:
    staging = path.with_name(path.name + ".staging")
    staging.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    staging.replace(path)


def _tree_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _tensor_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _gpu_uuid(local_rank: int) -> str:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(local_rank),
            "--query-gpu=uuid",
            "--format=csv,noheader",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=20,
    )
    value = completed.stdout.strip()
    if not value.startswith("GPU-"):
        raise TrainingContractError("nvidia-smi did not return a GPU UUID")
    return value


class LightNavA3TorchPolicy(nn.Module):
    """Exact Torch form of the portable NumPy correction head."""

    def __init__(self, checkpoint: str | Path):
        super().__init__()
        parameters, metadata = load_checkpoint(Path(checkpoint))
        self.checkpoint_metadata = metadata
        for name, value in parameters.items():
            self.register_parameter(name, nn.Parameter(torch.from_numpy(value.copy())))

    def forward(
        self,
        *,
        operation: str,
        features: torch.Tensor,
        gates: torch.Tensor | None = None,
        raw_residuals: torch.Tensor | None = None,
    ) -> Mapping[str, torch.Tensor]:
        if operation != "evaluate":
            raise ValueError(f"unsupported A3 operation {operation!r}")
        x = features.float()
        if x.ndim != 2 or x.shape[1] != FEATURE_DIM:
            raise ValueError("A3 features must be [batch,64]")
        hidden = torch.relu(x @ self.w1 + self.b1)
        hidden = torch.relu(hidden @ self.w2 + self.b2)
        logits = hidden @ self.gate_w + self.gate_b
        residual_mean = hidden @ self.residual_w + self.residual_b
        values = (hidden @ self.value_w + self.value_b).squeeze(-1)
        output: dict[str, torch.Tensor] = {
            "logits": logits,
            "residual_mean": residual_mean,
            "values": values,
        }
        if gates is None or raw_residuals is None:
            return output
        if gates.shape != (x.shape[0],) or raw_residuals.shape != (x.shape[0], 2):
            raise ValueError("A3 action batch shape differs")
        log_gate = torch.log_softmax(logits, dim=-1).gather(1, gates.long().unsqueeze(1)).squeeze(1)
        std = torch.exp(self.log_std)
        normal_log_prob = -0.5 * torch.sum(
            torch.square((raw_residuals.float() - residual_mean) / std)
            + 2.0 * self.log_std
            + math.log(2.0 * math.pi),
            dim=-1,
        )
        adjust = (gates == ADJUST_GATE).float()
        joint_log_prob = log_gate + adjust * normal_log_prob
        probabilities = torch.softmax(logits, dim=-1)
        categorical_entropy = -torch.sum(
            probabilities * torch.log(torch.clamp(probabilities, min=1.0e-12)), dim=-1
        )
        gaussian_entropy = torch.sum(
            self.log_std + 0.5 * math.log(2.0 * math.pi * math.e)
        )
        output["joint_log_prob"] = joint_log_prob
        output["entropy"] = categorical_entropy + adjust * gaussian_entropy
        return output


class LightNavA3FSDPManager(FSDPModelManager):
    def model_provider_func(self):
        return LightNavA3TorchPolicy(self._cfg.model.initial_checkpoint)


def _manager_config(args: argparse.Namespace, total_steps: int):
    return OmegaConf.create(
        {
            "model": {
                "model_type": "lightnav_a3_head",
                "model_path": "",
                "initial_checkpoint": str(args.initial_checkpoint.resolve()),
                "precision": "fp32",
                "is_lora": False,
            },
            "optim": {
                "lr": args.learning_rate,
                "value_lr": args.learning_rate,
                "adam_beta1": 0.9,
                "adam_beta2": 0.999,
                "adam_eps": 1.0e-8,
                "weight_decay": 0.0,
                "clip_grad": 1.0,
                "total_training_steps": total_steps,
                "lr_warmup_steps": 0,
                "lr_scheduler": "constant",
            },
            "fsdp_config": {
                "disable": True,
                "strategy": "fsdp",
                "sharding_strategy": "full_shard",
                "gradient_checkpointing": False,
                "cpu_offload": False,
                "offload_pin_memory": False,
                "reshard_after_forward": True,
                "enable_gradient_accumulation": False,
                "forward_prefetch": False,
                "limit_all_gathers": True,
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


class RolloutDataset(Dataset):
    def __init__(self, path: Path, *, world_size: int):
        self.source_path = Path(path).resolve(strict=True)
        rows = []
        with self.source_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise TrainingContractError(f"invalid rollout JSON at line {line_number}") from error
                if row.get("actor_eligible") is not True:
                    raise TrainingContractError("actor dataset contains an ineligible row")
                if int(row.get("policy_version", -1)) != 0:
                    raise TrainingContractError("A3 first experiment requires behavior version 0")
                rows.append(row)
        if not rows:
            raise TrainingContractError("rollout dataset has no actor decisions")
        self.real_row_count = len(rows)
        self.features = torch.tensor([row["features"] for row in rows], dtype=torch.float32)
        self.gates = torch.tensor([row["gate_index"] for row in rows], dtype=torch.long)
        self.raw_residuals = torch.tensor([row["raw_residual"] for row in rows], dtype=torch.float32)
        self.old_log_probs = torch.tensor([row["old_joint_log_prob"] for row in rows], dtype=torch.float32)
        self.old_values = torch.tensor([row["value"] for row in rows], dtype=torch.float32)
        self.returns = torch.tensor([row["return"] for row in rows], dtype=torch.float32)
        advantages = torch.tensor([row["advantage"] for row in rows], dtype=torch.float32)
        self.advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1.0e-6)
        padding = (-self.real_row_count) % int(world_size)
        self.padding_row_count = padding
        self.valid = torch.ones((self.real_row_count,), dtype=torch.bool)
        if padding:
            self.features = torch.cat(
                (self.features, torch.zeros((padding, FEATURE_DIM), dtype=torch.float32))
            )
            self.gates = torch.cat((self.gates, torch.zeros((padding,), dtype=torch.long)))
            self.raw_residuals = torch.cat(
                (self.raw_residuals, torch.zeros((padding, 2), dtype=torch.float32))
            )
            for name in ("old_log_probs", "old_values", "returns", "advantages"):
                value = getattr(self, name)
                setattr(self, name, torch.cat((value, torch.zeros((padding,), dtype=torch.float32))))
            self.valid = torch.cat((self.valid, torch.zeros((padding,), dtype=torch.bool)))
        tensors = (
            self.features,
            self.raw_residuals,
            self.old_log_probs,
            self.old_values,
            self.returns,
            self.advantages,
        )
        if self.features.shape[1:] != (FEATURE_DIM,) or self.raw_residuals.shape[1:] != (2,):
            raise TrainingContractError("rollout tensor shape differs from A3")
        if not all(bool(torch.isfinite(value).all()) for value in tensors):
            raise TrainingContractError("rollout tensors contain non-finite values")
        if bool(((self.gates < 0) | (self.gates >= len(GATE_NAMES))).any()):
            raise TrainingContractError("rollout gate is outside A3")

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, index: int):
        return (
            self.features[index],
            self.gates[index],
            self.raw_residuals[index],
            self.old_log_probs[index],
            self.old_values[index],
            self.returns[index],
            self.advantages[index],
            self.valid[index],
        )


def _full_state(model: FullyShardedDataParallel, *, rank0_only: bool) -> dict[str, torch.Tensor]:
    config = FullStateDictConfig(offload_to_cpu=True, rank0_only=rank0_only)
    with FullyShardedDataParallel.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, config
    ):
        state = model.state_dict()
    normalized = {}
    for name, value in state.items():
        key = name
        for prefix in ("module.", "_fsdp_wrapped_module."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        normalized[key] = value.detach().cpu()
    return normalized


def _max_ratio_deviation(
    model: FullyShardedDataParallel,
    loader: DataLoader,
    device: torch.device,
) -> float:
    local_max = torch.zeros((), device=device)
    with torch.no_grad():
        for batch in loader:
            features, gates, raw, old_log_prob, *_, valid = [item.to(device) for item in batch]
            output = model(
                operation="evaluate", features=features, gates=gates, raw_residuals=raw
            )
            if bool(valid.any().item()):
                deviation = torch.max(
                    torch.abs(torch.exp(output["joint_log_prob"][valid] - old_log_prob[valid]) - 1.0)
                )
                local_max = torch.maximum(local_max, deviation)
    dist.all_reduce(local_max, op=dist.ReduceOp.MAX)
    return float(local_max.item())


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--rollout", required=True, type=Path)
    result.add_argument("--expected-rollout-sha256", required=True)
    result.add_argument("--initial-checkpoint", required=True, type=Path)
    result.add_argument("--expected-initial-sha256", required=True)
    result.add_argument("--run-manifest", required=True, type=Path)
    result.add_argument("--expected-run-manifest-sha256", required=True)
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument("--epochs", type=int, default=4)
    result.add_argument("--batch-size-per-rank", type=int, default=128)
    result.add_argument("--learning-rate", type=float, default=3.0e-4)
    result.add_argument("--clip-ratio", type=float, default=0.2)
    result.add_argument("--value-clip", type=float, default=0.2)
    result.add_argument("--value-coef", type=float, default=0.5)
    result.add_argument("--entropy-coef", type=float, default=0.01)
    result.add_argument("--seed", type=int, default=2026093003)
    result.add_argument("--required-world-size", type=int, default=8)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "-1"))
    if world_size != args.required_world_size or rank < 0 or local_rank < 0:
        raise TrainingContractError(
            f"A3 requires torchrun world_size={args.required_world_size}, got {world_size}"
        )
    if args.epochs < 1 or args.batch_size_per_rank < 1:
        raise TrainingContractError("epochs and batch size must be positive")
    if sha256_file(args.rollout) != args.expected_rollout_sha256:
        raise TrainingContractError("rollout SHA-256 differs")
    if sha256_file(args.initial_checkpoint) != args.expected_initial_sha256:
        raise TrainingContractError("initial checkpoint SHA-256 differs")
    if sha256_file(args.run_manifest) != args.expected_run_manifest_sha256:
        raise TrainingContractError("outer run manifest SHA-256 differs")
    run_manifest = json.loads(args.run_manifest.read_text(encoding="utf-8"))
    if (
        not isinstance(run_manifest, dict)
        or run_manifest.get("schema_name") != "lightnav_a3_run_manifest"
        or run_manifest.get("schema_version") != 1
        or run_manifest.get("policy", {}).get("initial_checkpoint_sha256")
        != args.expected_initial_sha256
        or run_manifest.get("ppo", {}).get("required_world_size")
        != args.required_world_size
    ):
        raise TrainingContractError("outer run manifest contract differs")
    expected_ppo = run_manifest["ppo"]
    actual_ppo = {
        "epochs": args.epochs,
        "batch_size_per_rank": args.batch_size_per_rank,
        "learning_rate": args.learning_rate,
        "clip_ratio": args.clip_ratio,
        "value_clip": args.value_clip,
        "value_coefficient": args.value_coef,
        "entropy_coefficient": args.entropy_coef,
        "required_world_size": args.required_world_size,
    }
    if expected_ppo != actual_ppo:
        raise TrainingContractError("PPO settings differ from the outer run manifest")
    initial_parameters, initial_metadata = load_checkpoint(args.initial_checkpoint)
    if int(initial_metadata["policy_version"]) != 0:
        raise TrainingContractError("initial checkpoint policy version is not zero")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rank_dir = args.output_dir / "ranks"
    rank_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    started = time.time()

    # A collective before model setup proves all requested ranks joined the same job.
    collective = torch.tensor(float(rank), device=device)
    dist.all_reduce(collective, op=dist.ReduceOp.SUM)
    expected_collective = world_size * (world_size - 1) / 2
    if float(collective.item()) != expected_collective:
        raise TrainingContractError("eight-rank NCCL collective differs")

    dataset = RolloutDataset(args.rollout, world_size=world_size)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
        drop_last=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size_per_rank,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    if sampler.total_size != len(dataset) or len(dataset) % world_size:
        raise TrainingContractError("A3 padded sampler would duplicate on-policy rows")
    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * args.epochs
    manager = LightNavA3FSDPManager(
        _manager_config(args, total_steps), world_size=world_size, rank=rank
    )
    manager.setup_model_and_optimizer()
    if not isinstance(manager.model, FullyShardedDataParallel):
        raise TrainingContractError("RLinf did not wrap A3 in FSDP")

    pre_update_deviation = _max_ratio_deviation(manager.model, loader, device)
    if pre_update_deviation > 5.0e-5:
        raise TrainingContractError(
            f"pre-update PPO ratio deviates by {pre_update_deviation}"
        )

    losses: list[dict[str, float]] = []
    gradient_steps = 0
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        for batch in loader:
            features, gates, raw, old_log_prob, old_value, target_return, advantage, valid = [
                item.to(device, non_blocking=True) for item in batch
            ]
            manager.optimizer.zero_grad(set_to_none=True)
            output = manager.model(
                operation="evaluate", features=features, gates=gates, raw_residuals=raw
            )
            ratio = torch.exp(output["joint_log_prob"] - old_log_prob)
            clipped_ratio = torch.clamp(
                ratio, 1.0 - args.clip_ratio, 1.0 + args.clip_ratio
            )
            valid_float = valid.float()
            global_valid = valid_float.sum()
            dist.all_reduce(global_valid, op=dist.ReduceOp.SUM)
            if float(global_valid.item()) <= 0.0:
                raise TrainingContractError("PPO minibatch contains no real on-policy row")
            gradient_scale = world_size / global_valid
            policy_terms = -torch.minimum(ratio * advantage, clipped_ratio * advantage)
            policy_loss = torch.sum(policy_terms * valid_float) * gradient_scale
            clipped_value = old_value + torch.clamp(
                output["values"] - old_value, -args.value_clip, args.value_clip
            )
            value_terms = 0.5 * torch.maximum(
                torch.square(output["values"] - target_return),
                torch.square(clipped_value - target_return),
            )
            value_loss = torch.sum(value_terms * valid_float) * gradient_scale
            entropy = torch.sum(output["entropy"] * valid_float) * gradient_scale
            loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy
            if not bool(torch.isfinite(loss).item()):
                raise TrainingContractError("PPO objective is non-finite")
            loss.backward()
            grad_norm, learning_rates = manager.optimizer_step()
            gradient_steps += 1
            local_policy_sum = torch.sum(policy_terms.detach() * valid_float)
            local_value_sum = torch.sum(value_terms.detach() * valid_float)
            local_entropy_sum = torch.sum(output["entropy"].detach() * valid_float)
            diagnostic = torch.stack(
                (
                    local_policy_sum
                    + args.value_coef * local_value_sum
                    - args.entropy_coef * local_entropy_sum,
                    local_policy_sum,
                    local_value_sum,
                    local_entropy_sum,
                    torch.sum(ratio.detach() * valid_float),
                    torch.sum(
                        (
                            (ratio.detach() < 1.0 - args.clip_ratio)
                            | (ratio.detach() > 1.0 + args.clip_ratio)
                        ).float()
                        * valid_float
                    ),
                )
            )
            dist.all_reduce(diagnostic, op=dist.ReduceOp.SUM)
            diagnostic = diagnostic / global_valid
            losses.append(
                {
                    "loss": float(diagnostic[0].item()),
                    "policy_loss": float(diagnostic[1].item()),
                    "value_loss": float(diagnostic[2].item()),
                    "entropy": float(diagnostic[3].item()),
                    "ratio_mean": float(diagnostic[4].item()),
                    "ratio_clip_fraction": float(diagnostic[5].item()),
                    "grad_norm": float(grad_norm),
                    "learning_rate": float(learning_rates[0]),
                }
            )

    dist.barrier()
    state = _full_state(manager.model, rank0_only=True)
    final_receipt = None
    initial_hash = hashlib.sha256(
        b"".join(
            name.encode("utf-8") + initial_parameters[name].tobytes()
            for name in sorted(initial_parameters)
        )
    ).hexdigest()
    if rank == 0:
        expected_names = set(initial_parameters)
        if set(state) != expected_names:
            raise TrainingContractError(
                f"FSDP state names differ: {sorted(set(state) ^ expected_names)}"
            )
        final_numpy = {
            name: state[name].float().numpy().astype(np.float32, copy=True)
            for name in expected_names
        }
        final_hash = hashlib.sha256(
            b"".join(
                name.encode("utf-8") + final_numpy[name].tobytes()
                for name in sorted(final_numpy)
            )
        ).hexdigest()
        if final_hash == initial_hash:
            raise TrainingContractError("PPO optimizer did not change the A3 policy")
        checkpoint_path = args.output_dir / "checkpoints" / "latest" / "lightnav_a3_policy.npz"
        final_receipt = save_checkpoint(
            checkpoint_path,
            final_numpy,
            seed=int(initial_metadata["seed"]),
            policy_version=1,
        )
        if int(final_receipt["size_bytes"]) > MAX_CHECKPOINT_BYTES:
            raise TrainingContractError("A3 checkpoint exceeds the 2 GiB hard cap")

    rank_receipt = {
        "schema_name": "lightnav_a3_rank_receipt",
        "schema_version": 1,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "pid": os.getpid(),
        "gpu_uuid": _gpu_uuid(local_rank),
        "gpu_name": torch.cuda.get_device_name(local_rank),
        "collective_sum": float(collective.item()),
        "expected_collective_sum": expected_collective,
        "gradient_steps": gradient_steps,
        "dataset_rows": dataset.real_row_count,
        "padding_rows": dataset.padding_row_count,
        "sampler_rows": len(sampler),
        "completed_unix_s": time.time(),
    }
    _atomic_json(rank_dir / f"rank_{rank:03d}.json", rank_receipt)
    dist.barrier()

    if rank == 0:
        rank_receipts = [
            json.loads((rank_dir / f"rank_{index:03d}.json").read_text(encoding="utf-8"))
            for index in range(world_size)
        ]
        uuids = {row["gpu_uuid"] for row in rank_receipts}
        if len(uuids) != world_size or any(row["gradient_steps"] != total_steps for row in rank_receipts):
            raise TrainingContractError("rank receipts do not prove eight-card optimizer participation")
        summary = {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "rlinf_backend": "FSDPModelManager",
            "fsdp_wrapper": type(manager.model).__name__,
            "sharding_strategy": "full_shard",
            "world_size": world_size,
            "unique_gpu_uuid_count": len(uuids),
            "rank_receipts": rank_receipts,
            "rollout": {
                "path": str(args.rollout.resolve()),
                "sha256": args.expected_rollout_sha256,
            "actor_decision_count": dataset.real_row_count,
            "padding_row_count": dataset.padding_row_count,
            },
            "initial_checkpoint": {
                "path": str(args.initial_checkpoint.resolve()),
                "sha256": args.expected_initial_sha256,
                "parameter_content_sha256": initial_hash,
            },
            "run_manifest": {
                "path": str(args.run_manifest.resolve()),
                "sha256": args.expected_run_manifest_sha256,
                "run_id": run_manifest["run_id"],
            },
            "final_checkpoint": final_receipt,
            "epochs": args.epochs,
            "batch_size_per_rank": args.batch_size_per_rank,
            "optimizer_steps_per_rank": total_steps,
            "pre_update_ratio_max_abs_deviation": pre_update_deviation,
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "trainable_parameter_count": sum(value.numel() for value in state.values()),
            "checkpoint_hard_cap_bytes": MAX_CHECKPOINT_BYTES,
            "run_hard_cap_bytes": MAX_RUN_BYTES,
            "started_unix_s": started,
            "completed_unix_s": time.time(),
        }
        _atomic_json(args.output_dir / "training_result.json", summary)
        inventory = {
            str(path.relative_to(args.output_dir)): {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(args.output_dir.rglob("*"))
            if path.is_file() and path.name != "COMPLETED.json"
        }
        _atomic_json(args.output_dir / "artifact_inventory.json", inventory)
        if _tree_size(args.output_dir) > MAX_RUN_BYTES:
            raise TrainingContractError("A3 run exceeds the 16 GiB hard cap")
        _atomic_json(
            args.output_dir / "COMPLETED.json",
            {
                "schema_name": "lightnav_a3_training_completion",
                "schema_version": 1,
                "status": "completed",
                "training_result_sha256": sha256_file(args.output_dir / "training_result.json"),
                "artifact_inventory_sha256": sha256_file(args.output_dir / "artifact_inventory.json"),
            },
        )
        print(json.dumps(summary, sort_keys=True))

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
