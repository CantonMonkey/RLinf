#!/usr/bin/env python3
"""Write the immutable outer manifest for one LightNav A3 experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Sequence


class ManifestError(RuntimeError):
    """Raised before a run when an input identity differs from A3."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular(path: Path) -> Path:
    value = path.resolve(strict=True)
    if not value.is_file() or value.is_symlink():
        raise ManifestError(f"input is not a regular file: {path}")
    return value


def _git(source: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(source), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run-id", required=True)
    result.add_argument("--split-manifest", required=True, type=Path)
    result.add_argument("--lightnav-source", required=True, type=Path)
    result.add_argument("--expected-lightnav-commit", required=True)
    result.add_argument("--expected-collision-patch-sha256", required=True)
    result.add_argument("--expected-collision-diff-sha256", required=True)
    result.add_argument("--lightnav-weight", required=True, type=Path)
    result.add_argument("--expected-lightnav-weight-sha256", required=True)
    result.add_argument("--habitat-config", required=True, type=Path)
    result.add_argument("--rlinf-source", required=True, type=Path)
    result.add_argument("--expected-rlinf-commit", required=True)
    result.add_argument("--code-root", required=True, type=Path)
    result.add_argument("--initial-checkpoint", required=True, type=Path)
    result.add_argument("--output", required=True, type=Path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not args.run_id or "/" in args.run_id or "\\" in args.run_id:
        raise ManifestError("run-id must be one path-safe component")
    if args.output.exists():
        raise ManifestError("run manifest already exists")

    split_path = _regular(args.split_manifest)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if (
        split.get("schema_name") != "lightnav_a3_train_scene_split"
        or split.get("schema_version") != 1
        or split.get("upstream_split_read") != "train"
        or split.get("validation_split_reads") != 0
        or set(split.get("splits", {})) != {"train_update", "train_development"}
    ):
        raise ManifestError("split manifest differs from the train-only A3 contract")

    lightnav_source = args.lightnav_source.resolve(strict=True)
    lightnav_commit = _git(lightnav_source, "rev-parse", "HEAD").decode().strip()
    if lightnav_commit != args.expected_lightnav_commit:
        raise ManifestError("LightNav source commit differs")
    collision_diff = _git(
        lightnav_source,
        "diff",
        "--binary",
        "--",
        "habitat_server/lightnav_habitat/vlnce.py",
    )
    if not collision_diff:
        raise ManifestError("isolated LightNav source lacks the A3 collision patch")
    collision_diff_sha = hashlib.sha256(collision_diff).hexdigest()
    if collision_diff_sha != args.expected_collision_diff_sha256:
        raise ManifestError("applied LightNav collision diff is not the reviewed diff")

    rlinf_source = args.rlinf_source.resolve(strict=True)
    rlinf_commit = _git(rlinf_source, "rev-parse", "HEAD").decode().strip()
    if rlinf_commit != args.expected_rlinf_commit:
        raise ManifestError("RLinf source commit differs")

    weight = _regular(args.lightnav_weight)
    weight_sha = sha256_file(weight)
    if weight_sha != args.expected_lightnav_weight_sha256:
        raise ManifestError("LightNav original weight SHA-256 differs")

    habitat_config = _regular(args.habitat_config)
    code_root = args.code_root.resolve(strict=True)
    code_names = (
        "build_lightnav_a3_train_split.py",
        "lightnav_a3_policy.py",
        "lightnav_a3_rollout.py",
        "lightnav_a3_fsdp_train.py",
        "lightnav_a3_gpu_monitor.py",
        "lightnav_a3_obs_archive.py",
        "lightnav_a3_compare.py",
        "lightnav_a3_run_manifest.py",
        "lightnav_a3_verify_run_manifest.py",
        "lightnav_a3_collisions.patch",
        "lightnav_l0_native_probe.py",
        "run_lightnav_a3_habitat_ebk180.sh",
        "run_lightnav_a3_ebk_phase.sh",
        "run_lightnav_a3_a800.sh",
    )
    code = {
        name: {"sha256": sha256_file(_regular(code_root / name))}
        for name in code_names
    }
    collision_patch_sha = code["lightnav_a3_collisions.patch"]["sha256"]
    if collision_patch_sha != args.expected_collision_patch_sha256:
        raise ManifestError("collision patch file is not the reviewed patch")
    subprocess.run(
        [
            "git",
            "-C",
            str(lightnav_source),
            "apply",
            "--unidiff-zero",
            "--reverse",
            "--check",
            str(code_root / "lightnav_a3_collisions.patch"),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if _git(lightnav_source, "status", "--short").decode() != " M habitat_server/lightnav_habitat/vlnce.py\n":
        raise ManifestError("isolated LightNav worktree contains changes outside the collision patch")
    navrl_root = code_root / "navrl"
    if not navrl_root.resolve(strict=True).is_dir():
        raise ManifestError("deployed navrl package is absent")
    for path in sorted(navrl_root.rglob("*.py")):
        relative = str(path.relative_to(code_root)).replace("\\", "/")
        code[relative] = {"sha256": sha256_file(_regular(path))}
    initial_checkpoint = _regular(args.initial_checkpoint)
    manifest = {
        "schema_name": "lightnav_a3_run_manifest",
        "schema_version": 1,
        "run_id": args.run_id,
        "created_unix_s": time.time(),
        "upstream_split": "train",
        "validation_split_reads": 0,
        "lightnav": {
            "source_commit": lightnav_commit,
            "collision_patch_file_sha256": collision_patch_sha,
            "collision_diff_sha256": collision_diff_sha,
            "weight_file": weight.name,
            "weight_sha256": weight_sha,
            "weight_origin": "original_release",
            "ttt_or_secondary_post_training": False,
        },
        "habitat": {
            "config_file": habitat_config.name,
            "config_sha256": sha256_file(habitat_config),
            "image_height": 270,
            "image_width": 480,
            "max_steps": 160,
            "executor_step_contract": {
                "forward_m": [0.0, 0.25],
                "yaw_degrees": [-30.0, 30.0],
                "lateral_actor_credit": False,
            },
        },
        "rlinf": {"source_commit": rlinf_commit, "backend": "FSDPModelManager"},
        "split_manifest": {
            "sha256": sha256_file(split_path),
            "selection_manifest_sha256": split["selection_manifest_sha256"],
            "splits": {
                label: {
                    key: split["splits"][label][key]
                    for key in ("episode_count", "scene_count", "dataset_sha256", "gt_sha256")
                }
                for label in ("train_update", "train_development")
            },
        },
        "policy": {
            "action_gate": ["KEEP", "ADJUST", "STOP"],
            "residual_axes": ["forward_m", "yaw_rad"],
            "native_stop_actor_credit": False,
            "seed": 2026093003,
            "initial_checkpoint_sha256": sha256_file(initial_checkpoint),
        },
        "reward": {
            "formula": "delta_geodesic + 10*success - 0.01 - 0.2*collision - failed_stop - timeout",
            "simulator_truth_is_actor_observation": False,
        },
        "code": code,
        "ppo": {
            "epochs": 4,
            "batch_size_per_rank": 128,
            "learning_rate": 0.0003,
            "clip_ratio": 0.2,
            "value_clip": 0.2,
            "value_coefficient": 0.5,
            "entropy_coefficient": 0.01,
            "required_world_size": 8,
        },
        "expected_outputs": {
            "ebk180": ["baseline_dev", "train_rollout", "post_ppo_dev", "paired_comparison"],
            "a800": ["training", "gpu_telemetry.jsonl", "gpu_telemetry_summary.json"],
            "obs_checkpoint": f"checkpoints/lightnav-a3/{args.run_id}",
            "obs_results": f"results/lightnav-a3/{args.run_id}",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    staging = args.output.with_name(args.output.name + ".staging")
    staging.write_bytes(payload)
    staging.replace(args.output)
    print(json.dumps({"path": str(args.output), "sha256": sha256_file(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
