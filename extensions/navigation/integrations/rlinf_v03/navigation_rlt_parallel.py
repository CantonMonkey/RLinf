"""Deterministic plans and completion checks for Navigation-RLT execution.

The planner is intentionally transport-agnostic: each worker may run locally,
over SSH, or under RLinf/Ray, but it must own its Habitat environment, LightNav
session, reset nonce, and topology state.  Completion is accepted only when all
preassigned episode identities appear exactly once.  ``train_update`` remains
the default rollout split; matched evaluation must explicitly select the fixed
32-episode ``train_development`` split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence


PLAN_SCHEMA = "navigation_rlt_parallel_rollout_plan"
PLAN_VERSION = 1
WORKER_SCHEMA = "navigation_rlt_worker_completion"
WORKER_VERSION = 1
COMPLETION_SCHEMA = "navigation_rlt_parallel_completion"
COMPLETION_VERSION = 1
TRAIN_UPDATE = "train_update"
TRAIN_DEVELOPMENT = "train_development"
PLAN_SPLITS = (TRAIN_UPDATE, TRAIN_DEVELOPMENT)
MATCHED_EVAL_EPISODES = 32


class ParallelContractError(ValueError):
    """Raised when a rollout plan or worker receipt is incomplete or ambiguous."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _fsync_parent(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path.parent, flags)
    except OSError:
        if os.name == "nt":
            return
        raise
    try:
        os.fsync(descriptor)
    except OSError:
        if os.name != "nt":
            raise
    finally:
        os.close(descriptor)


def _file_identity(status: os.stat_result) -> tuple[int, int]:
    return int(status.st_dev), int(status.st_ino)


def _unlink_if_same(path: Path, identity: tuple[int, int]) -> None:
    try:
        status = path.lstat()
    except FileNotFoundError:
        return
    if _file_identity(status) != identity:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if _lexists(path):
        raise FileExistsError(path)
    encoded = json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8") + b"\n"
    fd, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".staging-", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    temporary_identity = _file_identity(os.fstat(fd))
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
        _fsync_parent(path)
        _unlink_if_same(temporary, temporary_identity)
        _fsync_parent(path)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        _unlink_if_same(temporary, temporary_identity)
        raise


def _load_object(path: Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ParallelContractError(f"{path} must contain a JSON object")
    return value


def _episode_identity(value: Mapping[str, object]) -> tuple[str, str, str]:
    episode_id = value.get("episode_id")
    scene = value.get("scene")
    instruction_sha = value.get("instruction_text_sha256")
    if not all(isinstance(item, str) and item for item in (episode_id, scene, instruction_sha)):
        raise ParallelContractError("episode identity is incomplete")
    return str(episode_id), str(scene), str(instruction_sha)


def build_plan(
    split_manifest_path: Path,
    *,
    run_id: str,
    expected_split_sha256: str,
    worker_count: int,
    expected_episode_count: int,
    split_label: str = TRAIN_UPDATE,
) -> dict[str, object]:
    split_manifest_path = Path(split_manifest_path).resolve(strict=True)
    if sha256_file(split_manifest_path) != expected_split_sha256:
        raise ParallelContractError("split manifest SHA-256 differs")
    if not isinstance(run_id, str) or not run_id:
        raise ParallelContractError("run_id must be non-empty")
    if type(worker_count) is not int or worker_count < 1:
        raise ParallelContractError("worker_count must be positive")
    if type(expected_episode_count) is not int or expected_episode_count < worker_count:
        raise ParallelContractError("episode count must be at least worker count")
    if split_label not in PLAN_SPLITS:
        raise ParallelContractError(f"unsupported split label: {split_label}")
    if split_label == TRAIN_DEVELOPMENT and expected_episode_count != MATCHED_EVAL_EPISODES:
        raise ParallelContractError(
            "train_development evaluation requires exactly 32 episodes"
        )
    manifest = _load_object(split_manifest_path)
    if manifest.get("upstream_split_read") != "train" or manifest.get("validation_split_reads") != 0:
        raise ParallelContractError("split manifest is not train-only")
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or not isinstance(splits.get(split_label), dict):
        raise ParallelContractError(f"split manifest has no {split_label} object")
    episodes = splits[split_label].get("episodes")
    if not isinstance(episodes, list) or len(episodes) != expected_episode_count:
        raise ParallelContractError(
            f"{split_label} episode count differs from fixed budget"
        )

    seen: set[tuple[str, str, str]] = set()
    normalized: list[dict[str, object]] = []
    for order, raw in enumerate(episodes):
        if not isinstance(raw, dict):
            raise ParallelContractError("episode row must be an object")
        identity = _episode_identity(raw)
        if identity in seen:
            raise ParallelContractError("split manifest contains a duplicate episode identity")
        seen.add(identity)
        normalized.append(
            {
                "episode_order": order,
                "episode_id": identity[0],
                "scene": identity[1],
                "instruction_text_sha256": identity[2],
            }
        )

    # Round-robin within each scene spreads long scenes across workers while the
    # immutable global order remains available for matched aggregation.
    by_scene: dict[str, list[dict[str, object]]] = {}
    for episode in normalized:
        by_scene.setdefault(str(episode["scene"]), []).append(episode)
    assignments: list[list[dict[str, object]]] = [[] for _ in range(worker_count)]
    cursor = 0
    for scene in sorted(by_scene):
        for episode in by_scene[scene]:
            assignments[cursor % worker_count].append(episode)
            cursor += 1
    workers = [
        {
            "worker_id": index,
            "episode_count": len(rows),
            "episodes": sorted(rows, key=lambda row: int(row["episode_order"])),
        }
        for index, rows in enumerate(assignments)
    ]
    if any(worker["episode_count"] < 1 for worker in workers):
        raise ParallelContractError("planner produced an empty worker")
    return {
        "schema_name": PLAN_SCHEMA,
        "schema_version": PLAN_VERSION,
        "run_id": run_id,
        "split_label": split_label,
        "validation_split_reads": 0,
        "split_manifest": {
            "path": str(split_manifest_path),
            "sha256": expected_split_sha256,
        },
        "worker_count": worker_count,
        "episode_count": expected_episode_count,
        "worker_isolation": [
            "habitat_environment",
            "lightnav_session",
            "reset_nonce",
            "topology_state",
        ],
        "workers": workers,
    }


def verify_completions(plan_path: Path, completion_paths: Sequence[Path]) -> dict[str, object]:
    plan_path = Path(plan_path).resolve(strict=True)
    plan_sha = sha256_file(plan_path)
    plan = _load_object(plan_path)
    if plan.get("schema_name") != PLAN_SCHEMA or plan.get("schema_version") != PLAN_VERSION:
        raise ParallelContractError("parallel plan schema differs")
    workers = plan.get("workers")
    if not isinstance(workers, list) or len(workers) != plan.get("worker_count"):
        raise ParallelContractError("parallel plan worker inventory differs")
    expected: dict[int, list[tuple[str, str, str]]] = {}
    for worker in workers:
        if not isinstance(worker, dict) or type(worker.get("worker_id")) is not int:
            raise ParallelContractError("parallel plan worker row is invalid")
        rows = worker.get("episodes")
        if not isinstance(rows, list):
            raise ParallelContractError("parallel plan worker lacks episodes")
        expected[int(worker["worker_id"])] = [_episode_identity(row) for row in rows]

    receipts: dict[int, dict[str, object]] = {}
    replay_shards: list[dict[str, object]] = []
    observed_global: set[tuple[str, str, str]] = set()
    for raw_path in completion_paths:
        path = Path(raw_path).resolve(strict=True)
        receipt = _load_object(path)
        if (
            receipt.get("schema_name") != WORKER_SCHEMA
            or receipt.get("schema_version") != WORKER_VERSION
            or receipt.get("status") != "completed"
            or receipt.get("plan_sha256") != plan_sha
            or receipt.get("run_id") != plan.get("run_id")
        ):
            raise ParallelContractError(f"worker receipt {path} differs from the plan")
        worker_id = receipt.get("worker_id")
        if type(worker_id) is not int or worker_id not in expected or worker_id in receipts:
            raise ParallelContractError("worker receipt id is missing, unknown, or duplicate")
        episode_rows = receipt.get("episodes")
        shard_rows = receipt.get("replay_shards")
        if not isinstance(episode_rows, list) or not isinstance(shard_rows, list):
            raise ParallelContractError("worker receipt lacks episode/shard rows")
        identities = [_episode_identity(row) for row in episode_rows]
        if identities != expected[worker_id]:
            raise ParallelContractError(f"worker {worker_id} episode sequence differs")
        if len(shard_rows) != len(identities):
            raise ParallelContractError(f"worker {worker_id} shard count differs")
        for identity, shard in zip(identities, shard_rows):
            if identity in observed_global:
                raise ParallelContractError("episode appears in more than one worker receipt")
            observed_global.add(identity)
            if not isinstance(shard, dict) or shard.get("episode_id") != identity[0]:
                raise ParallelContractError("replay shard identity differs")
            sha = shard.get("sha256")
            if not isinstance(sha, str) or len(sha) != 64:
                raise ParallelContractError("replay shard lacks SHA-256")
            replay_shards.append(dict(shard))
        receipt["receipt_path"] = str(path)
        receipt["receipt_sha256"] = sha256_file(path)
        receipts[worker_id] = receipt
    if set(receipts) != set(expected):
        raise ParallelContractError("one or more worker completions are missing")
    expected_global = {identity for rows in expected.values() for identity in rows}
    if observed_global != expected_global:
        raise ParallelContractError("global episode inventory is incomplete")
    return {
        "schema_name": COMPLETION_SCHEMA,
        "schema_version": COMPLETION_VERSION,
        "status": "completed",
        "run_id": plan["run_id"],
        "plan": {"path": str(plan_path), "sha256": plan_sha},
        "worker_count": len(receipts),
        "episode_count": len(observed_global),
        "worker_receipts": [receipts[index] for index in sorted(receipts)],
        "replay_shards": sorted(replay_shards, key=lambda row: str(row["episode_id"])),
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    create.add_argument("--split-manifest", required=True, type=Path)
    create.add_argument("--expected-split-sha256", required=True)
    create.add_argument("--run-id", required=True)
    create.add_argument(
        "--split-label", choices=PLAN_SPLITS, default=TRAIN_UPDATE
    )
    create.add_argument("--worker-count", required=True, type=int)
    create.add_argument("--episode-count", required=True, type=int)
    create.add_argument("--output", required=True, type=Path)
    verify = sub.add_parser("verify")
    verify.add_argument("--plan", required=True, type=Path)
    verify.add_argument("--completion", required=True, type=Path, action="append")
    verify.add_argument("--output", required=True, type=Path)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "create":
        payload = build_plan(
            args.split_manifest,
            run_id=args.run_id,
            expected_split_sha256=args.expected_split_sha256,
            worker_count=args.worker_count,
            expected_episode_count=args.episode_count,
            split_label=args.split_label,
        )
    else:
        payload = verify_completions(args.plan, args.completion)
    atomic_json(args.output, payload)
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
