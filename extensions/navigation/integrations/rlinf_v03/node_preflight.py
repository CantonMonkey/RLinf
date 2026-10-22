"""Check a PointNav node and print portable RLinf Hydra launch arguments.

This is read-only. It never starts Habitat, Ray, CUDA, or training. The
existing A800 YAML defaults remain intact; pass the generated overrides on
another Linux/WSL node after installing the isolated environments and assets.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path


PINNED_RLINF_COMMIT = "0505431899574619da86f551bad70b71e0ea2177"
SCENE_NAMES = ("skokloster-castle", "van-gogh-room")
DATASETS = {
    split: Path(f"datasets/pointnav/habitat-test-scenes/v1/{split}/{split}.json.gz")
    for split in ("train", "val")
}
PATCH_DESTINATIONS = {
    "envs_init.py": "rlinf/envs/__init__.py",
    "pointnav_init.py": "rlinf/envs/pointnav/__init__.py",
    "pointnav_env.py": "rlinf/envs/pointnav/pointnav_env.py",
    "mlp_init.py": "rlinf/models/embodiment/mlp_policy/__init__.py",
    "mlp_policy.py": "rlinf/models/embodiment/mlp_policy/mlp_policy.py",
    "train_embodied_agent.py": "examples/embodiment/train_embodied_agent.py",
    "pointnav_env.yaml": "examples/embodiment/config/env/pointnav_env.yaml",
    "pointnav_ppo_smoke.yaml": "examples/embodiment/config/pointnav_ppo_smoke.yaml",
    "pointnav_ppo_eval_64.yaml": "examples/embodiment/config/pointnav_ppo_eval_64.yaml",
    "nvidia_gpu.py": "rlinf/scheduler/hardware/accelerators/nvidia_gpu.py",
    "cluster.py": "rlinf/scheduler/cluster/cluster.py",
    "test_gpu_allowlist.py": "tests/unit_tests/test_gpu_allowlist.py",
}
# The A800 private checkout also carries the separately staged R2R registry
# and evaluation entrypoint. Both exact variants preserve the PointNav route.
PATCH_ALTERNATES = {
    "envs_init.py": ("envs_init_vln.py",),
    "train_embodied_agent.py": ("train_embodied_agent_pre_vln.py",),
}


def checked_file(path: Path, label: str) -> str:
    # Preserve venv interpreter symlinks. Resolving python -> its base binary
    # would bypass the virtual environment when this path becomes launch argv.
    absolute = Path(os.path.abspath(path.expanduser()))
    if not absolute.is_file():
        raise ValueError(f"{label} is not a file: {absolute}")
    return str(absolute)


def checked_dir(path: Path, label: str) -> str:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"{label} is not a directory: {resolved}")
    return str(resolved)


def compare_checkpoint_model_tensors(full, dcp_model, torch) -> int:
    """Require identical finite CPU tensor states for rollout and actor."""
    if not isinstance(full, dict) or not isinstance(dcp_model, dict) or not full:
        raise ValueError("RLinf checkpoint model states must be nonempty dictionaries")
    if full.keys() != dcp_model.keys():
        raise ValueError("RLinf actor DCP and full_weights.pt have different model keys")
    for name, tensor in full.items():
        restored = dcp_model[name]
        if not isinstance(tensor, torch.Tensor) or not isinstance(restored, torch.Tensor):
            raise ValueError(f"RLinf checkpoint model entry is not a tensor: {name}")
        if not bool(torch.isfinite(tensor).all()) or not bool(torch.isfinite(restored).all()):
            raise ValueError(f"RLinf checkpoint model tensor is nonfinite: {name}")
        if (
            tensor.shape != restored.shape
            or tensor.dtype != restored.dtype
            or not torch.equal(tensor, restored)
        ):
            raise ValueError(f"RLinf actor DCP and full_weights.pt differ at model tensor: {name}")
    return len(full)


def verify_eval_checkpoint_content(actor_dir: Path) -> int:
    """Strict-load the trusted PointNav model and compare DCP to rollout weights on CPU."""
    try:
        import torch
        from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
        from rlinf.models.embodiment.mlp_policy.mlp_policy import MLPPolicy
    except ImportError as exc:
        raise RuntimeError(
            "CPU checkpoint content verification requires installed Torch and RLinf"
        ) from exc

    full_path = actor_dir / "model_state_dict/full_weights.pt"
    full = torch.load(full_path, map_location="cpu", weights_only=True)
    model = MLPPolicy(
        obs_dim=2,
        action_dim=1,
        num_action_chunks=1,
        add_value_head=True,
        add_q_head=False,
        discrete_n_actions=4,
    )
    try:
        model.load_state_dict(full, strict=True)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("RLinf full_weights.pt does not strictly load into PointNav MLPPolicy") from exc

    # dcp_to_torch_save writes a temporary CPU file; never modify the checkpoint.
    with tempfile.TemporaryDirectory(prefix="pointnav-dcp-") as temporary:
        converted = Path(temporary) / "actor.pt"
        dcp_to_torch_save(actor_dir / "dcp_checkpoint", converted)
        # RLinf's trusted local DCP includes optimizer/RNG metadata which needs
        # the regular PyTorch loader. Do not use this on an untrusted checkpoint.
        training = torch.load(converted, map_location="cpu", weights_only=False)
    if not isinstance(training, dict):
        raise ValueError("RLinf actor DCP is not a checkpoint dictionary")
    fsdp = training.get("fsdp_checkpoint")
    if not isinstance(fsdp, dict):
        raise ValueError("RLinf actor DCP is missing fsdp_checkpoint")
    return compare_checkpoint_model_tensors(full, fsdp.get("model"), torch)


def eval_checkpoint_overrides(node_root: Path, requested: Path | None) -> list[str]:
    """Bind actor resume and rollout load to one local RLinf checkpoint."""
    if requested is None:
        return []
    resume_dir = Path(checked_dir(requested, "evaluation resume directory"))
    if not resume_dir.is_relative_to(node_root):
        raise ValueError("evaluation resume directory must stay under node root")
    if not re.fullmatch(r"global_step_[0-9]+", resume_dir.name):
        raise ValueError("RLinf evaluation resume directory must be named global_step_N")
    actor_dir = resume_dir / "actor"
    if not actor_dir.is_dir():
        raise ValueError("RLinf evaluation resume directory needs an actor/ directory")
    weights = Path(checked_file(
        actor_dir / "model_state_dict/full_weights.pt", "evaluation full model weights"
    ))
    if weights.stat().st_size == 0:
        raise ValueError("RLinf evaluation full_weights.pt is empty")
    dcp = actor_dir / "dcp_checkpoint"
    metadata = dcp / ".metadata"
    if not metadata.is_file() or metadata.stat().st_size == 0 or not any(
        path.is_file() and path.stat().st_size > 0 for path in dcp.glob("*.distcp")
    ):
        raise ValueError("RLinf evaluation actor/ needs a DCP checkpoint")
    verify_eval_checkpoint_content(actor_dir)
    return [f"runner.resume_dir={resume_dir}", f"runner.ckpt_path={weights}"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def episode_key(episode: dict) -> str:
    """Catch a duplicated start/goal even when an episode ID is renamed."""
    scene = str(episode["scene_id"]).replace("\\", "/").split("/")[-1]
    return json.dumps({
        "scene": scene,
        "start_position": episode["start_position"],
        "start_rotation": episode["start_rotation"],
        "goals": [goal["position"] for goal in episode["goals"]],
    }, sort_keys=True, separators=(",", ":"), allow_nan=False)


def read_episodes(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)["episodes"]


def validate_split_config(cfg, *, train_episodes: list[dict], val_episodes: list[dict]) -> dict:
    """Require explicit role splits, valid same-scene selections and strict holdout."""
    by_split = {"train": train_episodes, "val": val_episodes}
    other_keys = {
        "train": {episode_key(item) for item in val_episodes},
        "val": {episode_key(item) for item in train_episodes},
    }
    selected = {}
    for role in ("train", "eval"):
        env = cfg.env[role]
        split = str(env.get("dataset_split", ""))
        required = "train" if role == "train" else "val"
        if split != required:
            raise ValueError(f"{role} must explicitly use the {required} PointNav split")
        raw_indices = env.get("episode_indices", None)
        indices = ([int(env.get("episode_index", -1))] if raw_indices is None
                   else [int(item) for item in raw_indices])
        episodes = by_split[split]
        if (not indices or len(indices) != len(set(indices))
                or any(index < 0 or index >= len(episodes) for index in indices)):
            raise ValueError(f"{role} episode indices must be distinct and within {split}")
        chosen = [episodes[index] for index in indices]
        scenes = {str(item["scene_id"]).replace("\\", "/").split("/")[-1]
                  for item in chosen}
        if len(scenes) != 1:
            raise ValueError(f"{role} PointNav service would cross scenes")
        overlaps = [index for index, item in zip(indices, chosen)
                    if episode_key(item) in other_keys[split]]
        if overlaps:
            raise ValueError(f"{role} selected episodes overlap the other split: {overlaps}")
        selected[role] = {
            "split": split,
            "indices": indices,
            "episode_ids": [str(item["episode_id"]) for item in chosen],
            "scene": next(iter(scenes)),
            "disjoint_from_entire_other_split": True,
        }
    return selected


def inspect(args: argparse.Namespace) -> dict:
    root = Path(checked_dir(args.node_root, "node root"))
    checkout = Path(checked_dir(args.rlinf_checkout or root / "repos/RLinf-v0.3", "RLinf checkout"))
    habitat_python = checked_file(args.habitat_python, "Habitat Python")
    rlinf_python = checked_file(args.rlinf_python, "RLinf Python")
    service = checked_file(
        args.service_script or root / "repos/rlinf-nav/scripts/pointnav_service.py",
        "PointNav service",
    )
    asset_root = Path(checked_dir(args.asset_root or root / "data", "asset root"))
    scenes = [checked_file(asset_root / "scene_datasets/habitat-test-scenes" / f"{name}.{suffix}",
                           f"PointNav {name}.{suffix}")
              for name in SCENE_NAMES for suffix in ("glb", "navmesh")]
    datasets = {split: checked_file(asset_root / path, f"official PointNav {split} dataset")
                for split, path in DATASETS.items()}
    trainer = checked_file(
        checkout / "examples/embodiment/train_embodied_agent.py", "RLinf trainer"
    )
    staging = Path(checked_dir(
        args.staging_dir or Path(service).parent.parent / "integrations/rlinf_v03",
        "bridge staging directory",
    ))
    for staged_name, relative_dest in PATCH_DESTINATIONS.items():
        staged = Path(checked_file(staging / staged_name, f"staged {staged_name}"))
        installed = Path(checked_file(checkout / relative_dest, f"installed {staged_name}"))
        installed_hash = sha256_file(installed)
        alternatives = (
            staging / name for name in PATCH_ALTERNATES.get(staged_name, ())
        )
        if sha256_file(staged) != installed_hash and not any(
            candidate.is_file() and sha256_file(candidate) == installed_hash
            for candidate in alternatives
        ):
            raise ValueError(f"Staged patch differs from installed RLinf file: {staged_name}")
    revision = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    if revision != PINNED_RLINF_COMMIT:
        raise ValueError(f"RLinf revision {revision} differs from {PINNED_RLINF_COMMIT}")

    run_root = (args.run_root or root / "runs/pointnav-ppo-v03").expanduser().resolve()
    video_root = (args.video_root or root / "videos/pointnav-v03").expanduser().resolve()
    if not run_root.is_relative_to(root) or not video_root.is_relative_to(root):
        raise ValueError("run and video output directories must stay under node root")
    overrides = [
        f"env.train.python_bin={habitat_python}",
        f"env.eval.python_bin={habitat_python}",
        f"env.train.server_script={service}",
        f"env.eval.server_script={service}",
        f"env.train.asset_root={asset_root}",
        f"env.eval.asset_root={asset_root}",
        f"env.train.video_cfg.video_base_dir={video_root}",
        f"env.eval.video_cfg.video_base_dir={video_root}",
        f"runner.logger.log_path={run_root}",
    ]
    eval_overrides = eval_checkpoint_overrides(root, args.eval_resume_dir)
    split_configs = None
    if args.compose_check:
        # Hydra composition does not launch Ray or import Habitat. Validate
        # both inherited configs, including the eval-specific episode list.
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf

        config_dir = checkout / "examples/embodiment/config"
        episodes = {split: read_episodes(Path(path)) for split, path in datasets.items()}
        split_configs = {}
        with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
            for config_name in ("pointnav_ppo_smoke", "pointnav_ppo_eval_64"):
                config_overrides = overrides + (
                    eval_overrides if config_name == "pointnav_ppo_eval_64" else []
                )
                cfg = compose(config_name=config_name, overrides=config_overrides)
                OmegaConf.resolve(cfg)
                split_configs[config_name] = validate_split_config(
                    cfg, train_episodes=episodes["train"], val_episodes=episodes["val"]
                )
                for role in ("train", "eval"):
                    env = cfg.env[role]
                    if (
                        env.python_bin != habitat_python
                        or env.server_script != service
                        or env.asset_root != str(asset_root)
                        or env.video_cfg.video_base_dir != str(video_root)
                    ):
                        raise ValueError(f"Hydra override mismatch: {config_name} {role}")
                if cfg.runner.logger.log_path != str(run_root):
                    raise ValueError(f"Hydra output override mismatch: {config_name}")
                if config_name == "pointnav_ppo_eval_64" and eval_overrides:
                    if (cfg.runner.resume_dir != eval_overrides[0].split("=", 1)[1]
                            or cfg.runner.ckpt_path != eval_overrides[1].split("=", 1)[1]):
                        raise ValueError("Hydra evaluation checkpoint overrides mismatch")
    return {
        "status": "static_preflight_ok",
        "hydra_compose_validated": bool(args.compose_check),
        "visual_render_validated": False,
        "cuda_validated": False,
        "training_started": False,
        "node_root": str(root),
        "rlinf_checkout": str(checkout),
        "rlinf_revision": revision,
        "bridge_patch_files_verified": len(PATCH_DESTINATIONS),
        "habitat_python": habitat_python,
        "rlinf_python": rlinf_python,
        "fixed_scene": scenes[0],
        "pointnav_scene_assets": scenes,
        "pointnav_train_dataset": datasets["train"],
        "pointnav_val_dataset": datasets["val"],
        "split_configs_validated": split_configs,
        "hydra_overrides": overrides,
        "eval_resume_dir": eval_overrides[0].split("=", 1)[1] if eval_overrides else None,
        "eval_ckpt_path": eval_overrides[1].split("=", 1)[1] if eval_overrides else None,
        "eval_checkpoint_content_verified": bool(eval_overrides),
        "smoke_argv": ([rlinf_python, trainer, "--config-name", "pointnav_ppo_smoke", *overrides]
                       if args.compose_check else None),
        "eval_argv": ([rlinf_python, trainer, "--config-name", "pointnav_ppo_eval_64",
                       *overrides, *eval_overrides]
                      if args.compose_check and eval_overrides else None),
        "eval_next_gate": ("run with the verified GPU"
                           if args.compose_check and eval_overrides else
                           "rerun --compose-check with --eval-resume-dir after training"),
        "next_gate": ("verify GPU ownership/UUID and graphics/EGL on this node before launch"
                      if args.compose_check else "rerun with --compose-check before using launch arguments"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-root", required=True, type=Path)
    parser.add_argument("--habitat-python", required=True, type=Path)
    parser.add_argument("--rlinf-python", required=True, type=Path)
    parser.add_argument("--rlinf-checkout", type=Path)
    parser.add_argument("--service-script", type=Path)
    parser.add_argument("--staging-dir", type=Path)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--video-root", type=Path)
    parser.add_argument("--eval-resume-dir", type=Path)
    parser.add_argument("--compose-check", action="store_true")
    args = parser.parse_args()
    print(json.dumps(inspect(args), ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
