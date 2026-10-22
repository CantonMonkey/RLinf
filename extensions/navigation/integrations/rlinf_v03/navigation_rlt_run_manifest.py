#!/usr/bin/env python3
"""Build the immutable outer manifest for one train-only Navigation-RLT run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Sequence


SCHEMA_NAME = "navigation_rlt_experiment_manifest"
SCHEMA_VERSION = 1
ORIGINAL_LIGHTNAV_REVISION = "a645828d81a8439651172197ca80a75dc1377977"
ORIGINAL_WEIGHT_SHA256 = "ffc4a925378a881afa761865048eb8d07c55cacf5eaf66548b6641c39f67af18"

CODE_FILES = (
    "build_navigation_rlt_train_split.py",
    "lightnav_a3_policy.py",
    "lightnav_a3_rollout.py",
    "lightnav_l0_native_probe.py",
    "navigation_rlt_action.py",
    "navigation_rlt_compare.py",
    "navigation_rlt_eval.py",
    "navigation_rlt_parallel.py",
    "navigation_rlt_policy.py",
    "navigation_rlt_replay.py",
    "navigation_rlt_relocate_replay.py",
    "navigation_rlt_rollout.py",
    "navigation_rlt_fsdp_train.py",
    "navigation_rlt_promote_replay.py",
    "navigation_rlt_stage1_gate.py",
    "navigation_rlt_prepare_training.py",
    "navigation_rlt_run_manifest.py",
    "navigation_rlt_obs_archive.py",
    "lightnav_a3_gpu_monitor.py",
    "run_navigation_rlt_lightnav_ebk.sh",
    "run_navigation_rlt_ebk_worker.sh",
    "run_navigation_rlt_a800.sh",
    "streamvln_topology.py",
)


class RunManifestError(RuntimeError):
    """Raised when a production run identity is incomplete or ambiguous."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise RunManifestError(f"{name} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise RunManifestError(f"{name} must be hexadecimal") from exc
    return value.lower()


def _regular(path: Path, name: str) -> Path:
    value = path.resolve(strict=True)
    if not value.is_file() or value.is_symlink():
        raise RunManifestError(f"{name} is not a regular file: {path}")
    return value


def _json(path: Path, name: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RunManifestError(f"{name} is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise RunManifestError(f"{name} must contain an object")
    return value


def _git(source: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(source), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def _git_bytes(source: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(source), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def _verify_git(
    source: Path, expected_head: str, expected_context_diff_sha256: str
) -> dict[str, object]:
    root = source.resolve(strict=True)
    if not root.is_dir():
        raise RunManifestError("LightNav source is not a directory")
    head = _git(root, "rev-parse", "HEAD")
    if head != expected_head:
        raise RunManifestError("LightNav patched HEAD differs")
    if _git(root, "status", "--porcelain"):
        raise RunManifestError("LightNav patched worktree is dirty")
    if _git(root, "merge-base", "--is-ancestor", ORIGINAL_LIGHTNAV_REVISION, head) != "":
        # `merge-base --is-ancestor` succeeds silently. This branch is retained for clarity.
        raise AssertionError("unreachable")
    commits = _git(root, "rev-list", "--reverse", f"{ORIGINAL_LIGHTNAV_REVISION}..{head}").splitlines()
    if not commits:
        raise RunManifestError("LightNav patched HEAD has no commits after the original release")
    context_diff = _git_bytes(
        root,
        "diff",
        "--binary",
        f"{ORIGINAL_LIGHTNAV_REVISION}..{commits[0]}",
    )
    observed_context_diff_sha = hashlib.sha256(context_diff).hexdigest()
    if observed_context_diff_sha != expected_context_diff_sha256:
        raise RunManifestError("committed LightNav context diff SHA-256 differs")
    return {
        "original_release_revision": ORIGINAL_LIGHTNAV_REVISION,
        "patched_head": head,
        "patch_commits": commits,
        "committed_context_diff_sha256": observed_context_diff_sha,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run-id", required=True)
    result.add_argument("--split-manifest", required=True, type=Path)
    result.add_argument("--expected-split-manifest-sha256", required=True)
    result.add_argument("--parallel-plan", required=True, type=Path)
    result.add_argument("--expected-parallel-plan-sha256", required=True)
    result.add_argument("--worker-materialization-manifest", required=True, type=Path)
    result.add_argument("--expected-worker-materialization-sha256", required=True)
    result.add_argument("--lightnav-source", required=True, type=Path)
    result.add_argument("--expected-lightnav-head", required=True)
    result.add_argument("--lightnav-context-patch", required=True, type=Path)
    result.add_argument("--expected-context-patch-sha256", required=True)
    result.add_argument("--expected-context-diff-sha256", required=True)
    result.add_argument("--lightnav-weight", required=True, type=Path)
    result.add_argument("--tokenizer", required=True, type=Path)
    result.add_argument("--processor-config", required=True, type=Path)
    result.add_argument("--history-config", required=True, type=Path)
    result.add_argument("--a3-checkpoint", required=True, type=Path)
    result.add_argument("--expected-a3-checkpoint-sha256", required=True)
    result.add_argument("--code-root", required=True, type=Path)
    result.add_argument("--navigation-code-commit", required=True)
    result.add_argument("--rlinf-source", required=True, type=Path)
    result.add_argument("--expected-rlinf-commit", required=True)
    result.add_argument("--output", required=True, type=Path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not args.run_id or Path(args.run_id).name != args.run_id:
        raise RunManifestError("run-id must be one path-safe component")
    if args.output.exists():
        raise RunManifestError("output manifest already exists")

    split_path = _regular(args.split_manifest, "split manifest")
    plan_path = _regular(args.parallel_plan, "parallel plan")
    workers_path = _regular(
        args.worker_materialization_manifest, "worker materialization manifest"
    )
    expected_split_sha = _sha(
        args.expected_split_manifest_sha256, "expected split manifest hash"
    )
    expected_plan_sha = _sha(
        args.expected_parallel_plan_sha256, "expected parallel plan hash"
    )
    expected_workers_sha = _sha(
        args.expected_worker_materialization_sha256,
        "expected worker materialization hash",
    )
    for path, expected, name in (
        (split_path, expected_split_sha, "split manifest"),
        (plan_path, expected_plan_sha, "parallel plan"),
        (workers_path, expected_workers_sha, "worker materialization manifest"),
    ):
        if sha256_file(path) != expected:
            raise RunManifestError(f"{name} SHA-256 differs")

    split = _json(split_path, "split manifest")
    plan = _json(plan_path, "parallel plan")
    workers = _json(workers_path, "worker materialization manifest")
    update = split.get("splits", {}).get("train_update") if isinstance(split.get("splits"), dict) else None
    development = split.get("splits", {}).get("train_development") if isinstance(split.get("splits"), dict) else None
    if (
        split.get("upstream_split_read") != "train"
        or split.get("validation_split_reads") != 0
        or not isinstance(update, dict)
        or update.get("episode_count") != 256
        or not isinstance(development, dict)
        or development.get("episode_count") != 32
    ):
        raise RunManifestError("split manifest is not the fixed 256/32 train-only split")
    if (
        plan.get("run_id") != args.run_id
        or plan.get("split_label") != "train_update"
        or plan.get("worker_count") != 8
        or plan.get("episode_count") != 256
    ):
        raise RunManifestError("parallel plan differs from the fixed 8x32 run")
    if (
        workers.get("run_id") != args.run_id
        or workers.get("status") != "materialized"
        or workers.get("validation_split_reads") != 0
        or workers.get("worker_count") != 8
        or workers.get("episode_count") != 256
        or workers.get("worker_episode_counts") != [32] * 8
    ):
        raise RunManifestError("worker materialization manifest differs from the run")

    expected_context_diff_sha = _sha(
        args.expected_context_diff_sha256, "expected committed context diff hash"
    )
    lightnav = _verify_git(
        args.lightnav_source, args.expected_lightnav_head, expected_context_diff_sha
    )
    context_patch = _regular(args.lightnav_context_patch, "context patch")
    expected_context_patch_sha = _sha(
        args.expected_context_patch_sha256, "expected context patch hash"
    )
    if sha256_file(context_patch) != expected_context_patch_sha:
        raise RunManifestError("context patch SHA-256 differs")

    weight = _regular(args.lightnav_weight, "LightNav weight")
    if sha256_file(weight) != ORIGINAL_WEIGHT_SHA256:
        raise RunManifestError("LightNav weight is not the original release")
    tokenizer = _regular(args.tokenizer, "tokenizer")
    processor = _regular(args.processor_config, "processor config")
    history = _regular(args.history_config, "history config")
    a3_checkpoint = _regular(args.a3_checkpoint, "A3 checkpoint")
    expected_a3_sha = _sha(
        args.expected_a3_checkpoint_sha256, "expected A3 checkpoint hash"
    )
    if sha256_file(a3_checkpoint) != expected_a3_sha:
        raise RunManifestError("A3 checkpoint SHA-256 differs")

    code_root = args.code_root.resolve(strict=True)
    navigation_code_commit = args.navigation_code_commit.lower()
    if len(navigation_code_commit) != 40:
        raise RunManifestError("navigation code commit must be a full Git SHA-1")
    try:
        int(navigation_code_commit, 16)
    except ValueError as exc:
        raise RunManifestError("navigation code commit must be hexadecimal") from exc
    code = {}
    for filename in CODE_FILES:
        path = _regular(code_root / filename, f"code file {filename}")
        code[filename] = sha256_file(path)
    navrl_root = code_root / "navrl"
    if not navrl_root.resolve(strict=True).is_dir() or navrl_root.is_symlink():
        raise RunManifestError("code root lacks a regular navrl package")
    navrl_files = sorted(navrl_root.rglob("*.py"))
    if not navrl_files:
        raise RunManifestError("navrl package contains no Python source")
    for path in navrl_files:
        relative = path.relative_to(code_root).as_posix()
        code[relative] = sha256_file(_regular(path, f"code file {relative}"))

    rlinf_root = args.rlinf_source.resolve(strict=True)
    rlinf_head = _git(rlinf_root, "rev-parse", "HEAD")
    if rlinf_head != args.expected_rlinf_commit:
        raise RunManifestError("RLinf source commit differs")

    manifest = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "run_id": args.run_id,
        "created_unix_s": time.time(),
        "boundaries": {
            "upstream_split_read": "train",
            "validation_split_reads": 0,
            "lightnav_frozen": True,
            "weight_origin": "original_release",
            "ttt_or_secondary_post_training": False,
            "remote_push": False,
        },
        "data": {
            "split_manifest_sha256": expected_split_sha,
            "parallel_plan_sha256": expected_plan_sha,
            "worker_materialization_manifest_sha256": expected_workers_sha,
            "update_episode_count": 256,
            "development_episode_count": 32,
            "worker_count": 8,
            "episodes_per_worker": 32,
        },
        "lightnav": {
            **lightnav,
            "context_patch_sha256": expected_context_patch_sha,
            "weight_sha256": ORIGINAL_WEIGHT_SHA256,
            "tokenizer_sha256": sha256_file(tokenizer),
            "processor_config_sha256": sha256_file(processor),
            "history_config_sha256": sha256_file(history),
            "backend": "hf",
            "context_request": "contextual_prefix_hidden_v1",
            "artifact_prefix_max_tokens": 4096,
            "policy_prefix_max_tokens": 192,
            "prefix_hidden_dim": 2560,
        },
        "behavior": {
            "name": "a3_stochastic",
            "checkpoint_sha256": expected_a3_sha,
            "seed": 2026100201,
        },
        "rlinf": {"source_commit": rlinf_head, "required_world_size": 8},
        "code": {"git_commit": navigation_code_commit, "files": code},
        "protocol": {
            "max_physical_steps": 160,
            "discount_gamma": 0.99,
            "consume_full_feature_artifacts_after_authenticated_replay": True,
            "replay_representation": "selected_prefix_hidden_float16_192",
            "stage1": "masked_prefix_and_topology_reconstruction",
            "stage2": "proposal_conditioned_conservative_twin_q_actor_critic",
            "matched_eval_arms": ["native_lightnav", "a3_ppo", "navigation_rlt"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = args.output.with_name(args.output.name + ".staging")
    staging.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    staging.replace(args.output)
    print(
        json.dumps(
            {"path": str(args.output.resolve()), "sha256": sha256_file(args.output)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
