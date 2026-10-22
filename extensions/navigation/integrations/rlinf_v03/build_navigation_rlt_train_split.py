#!/usr/bin/env python3
"""Freeze Navigation-RLT's train-only update and development datasets.

Only a literal ``train/train.json.gz`` whose compressed SHA-256 matches the
known R2R train source is accepted.  The fixed A3 train-development identities
are preserved.  Every remaining source scene contributes globally unique
trajectories to a deterministic 256-episode update set.

The source has 61 scenes, so after reserving the eight fixed development scenes
only 53 update scenes exist.  Two of those scenes contain fewer than four
unique trajectories.  The frozen quota therefore starts at ``min(4, capacity)``
per scene and gives one additional trajectory, in seeded scene order, to scenes
with at least five trajectories until the exact 256-episode budget is reached.

Worker materialization consumes an already-built train_update dataset plus a
``navigation_rlt_parallel`` plan.  It creates one literal Habitat ``train``
dataset per worker and verifies hashes, episode order, ground truth, and the
exact global partition.  No validation split path is searched or read.
"""

from __future__ import annotations

import argparse
from io import BytesIO
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Mapping, Sequence


SOURCE_TRAIN_SHA256 = "340a80133b2157520354ab055a91d98feb2f42e4bbda17b200c911f8788492ea"
UPDATE_SELECTION_SEED = "navigation-rlt-update-v1-20261002"
DEVELOPMENT_SELECTION_SEED = "lightnav-a3-dev-prior-scene-excluded-v1-20260930"
SOURCE_SCENES = 61
UPDATE_SCENES = 53
DEVELOPMENT_SCENES = 8
UPDATE_EPISODES = 256
DEVELOPMENT_EPISODES = 32
EPISODES_PER_DEVELOPMENT_SCENE = 4
FORBIDDEN_PARTS = frozenset({"val", "val_seen", "val_unseen", "val_seen_synonyms"})
SPLIT_SCHEMA = "navigation_rlt_train_scene_split"
SPLIT_VERSION = 1
PLAN_SCHEMA = "navigation_rlt_parallel_rollout_plan"
PLAN_VERSION = 1
WORKER_SCHEMA = "navigation_rlt_worker_dataset_materialization"
WORKER_VERSION = 1
MATERIALIZATION_SCHEMA = "navigation_rlt_worker_dataset_partition"
MATERIALIZATION_VERSION = 1

FROZEN_DEVELOPMENT_SCENE_EPISODES = (
    ("XcA2TqTSSAj", ("9449", "2214", "10529", "3864")),
    ("17DRP5sb8fy", ("2890", "2944", "3372", "9072")),
    ("JF19kD82Mey", ("7282", "2450", "132", "1069")),
    ("B6ByNegPMKs", ("6417", "1548", "3768", "7332")),
    ("uNb9QFRL6hY", ("10801", "9861", "2880", "3984")),
    ("1LXtFkjw3qL", ("2093", "4909", "6914", "10765")),
    ("ULsKaCPVFJR", ("2413", "9162", "9022", "9965")),
    ("JeFG25nYj2p", ("5842", "5954", "4992", "695")),
)
PRIOR_PROBE_SCENES = frozenset(
    {
        "Uxmj2M2itWa",
        "cV4RVeZvu5T",
        "5LpN3gDmAk7",
        "gTV8FGcVJC9",
        "29hnd4uzFmX",
        "8WUmhLawc2A",
        "vyrNrziPKCB",
        "ac26ZMwG7aT",
        "mJXqzFtmKg4",
        "1pXnuDYAj8r",
        "qoiz87JEwZ2",
        "p5wJjkQkbXX",
        "E9uDoFAP3SH",
        "S9hNv5qa7GM",
        "Vvot9Ly1tCj",
        "VLzqgDo317F",
    }
)


class SplitContractError(RuntimeError):
    """Raised before publication when a train-only contract differs."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _gzip_bytes(payload: bytes) -> bytes:
    stream = BytesIO()
    with gzip.GzipFile(filename="", fileobj=stream, mode="wb", compresslevel=9, mtime=0) as archive:
        archive.write(payload)
    return stream.getvalue()


def _forbidden(path: Path) -> bool:
    return any(part.casefold() in FORBIDDEN_PARTS for part in path.parts)


def _source_path(value: Path) -> Path:
    requested = value.expanduser()
    if requested.name != "train.json.gz" or requested.parent.name != "train" or _forbidden(requested):
        raise SplitContractError("source must be a validation-free literal train/train.json.gz path")
    if requested.is_symlink():
        raise SplitContractError("source dataset may not be a symbolic link")
    path = requested.resolve(strict=True)
    if not path.is_file() or path.name != "train.json.gz" or path.parent.name != "train":
        raise SplitContractError("source must resolve to the literal train/train.json.gz file")
    if _forbidden(path):
        raise SplitContractError("source path contains a forbidden split")
    return path


def _new_root(value: Path, *, source: Path | None = None) -> Path:
    requested = value.expanduser().absolute()
    if _forbidden(requested):
        raise SplitContractError("output path contains a forbidden split")
    if requested.exists() or requested.is_symlink():
        raise SplitContractError("output root must not already exist")
    parent = requested.parent.resolve(strict=True)
    root = parent / requested.name
    if source is not None and (root == source.parent or source in root.parents):
        raise SplitContractError("output may not replace or contain the source")
    return root


def _existing_build_root(value: Path, source: Path) -> Path:
    requested = value.expanduser()
    if requested.is_symlink() or _forbidden(requested):
        raise SplitContractError("built dataset root is a symbolic link or forbidden split")
    root = requested.resolve(strict=True)
    if not root.is_dir() or root == source.parent or source in root.parents:
        raise SplitContractError("built dataset root is invalid")
    return root


def _finite_xyz_path(value: object) -> bool:
    return bool(
        isinstance(value, list)
        and value
        and all(
            isinstance(point, list)
            and len(point) == 3
            and all(
                isinstance(item, (int, float))
                and not isinstance(item, bool)
                and math.isfinite(float(item))
                for item in point
            )
            for point in value
        )
    )


def _scene_name(episode: Mapping[str, object]) -> str:
    scene_id = episode.get("scene_id")
    if not isinstance(scene_id, str):
        raise SplitContractError("episode scene_id is missing")
    parts = scene_id.replace("\\", "/").split("/")
    if len(parts) != 3 or parts[0] != "mp3d" or parts[1] + ".glb" != parts[2]:
        raise SplitContractError("episode scene_id differs from mp3d/<scene>/<scene>.glb")
    return parts[1]


def _load_gzip_json_bytes(compressed: bytes, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(gzip.decompress(compressed).decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SplitContractError(f"{label} is not valid UTF-8 gzip JSON") from error
    if not isinstance(payload, dict):
        raise SplitContractError(f"{label} top-level value is not an object")
    return payload


def _load_source(source: Path) -> tuple[dict[str, object], list[dict[str, object]], str]:
    compressed = source.read_bytes()
    digest = _sha256(compressed)
    if digest != SOURCE_TRAIN_SHA256:
        raise SplitContractError("source train dataset SHA-256 differs from the fixed A3 source")
    payload = _load_gzip_json_bytes(compressed, label="source train dataset")
    if set(payload) != {"episodes", "instruction_vocab"}:
        raise SplitContractError("source top-level schema differs")
    raw = payload["episodes"]
    if not isinstance(raw, list):
        raise SplitContractError("source episodes is not a list")
    episodes: list[dict[str, object]] = []
    episode_ids: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise SplitContractError(f"episode {index} is not an object")
        episode = dict(item)
        episode_id = str(episode.get("episode_id", ""))
        trajectory_id = str(episode.get("trajectory_id", ""))
        if not episode_id or episode_id in episode_ids:
            raise SplitContractError("source episode IDs are missing or repeated")
        if not trajectory_id:
            raise SplitContractError(f"episode {episode_id} trajectory_id is missing")
        episode_ids.add(episode_id)
        reference = episode.get("reference_path")
        goals = episode.get("goals")
        if not _finite_xyz_path(reference):
            raise SplitContractError(f"episode {episode_id} has no finite reference path")
        if reference[0] != episode.get("start_position"):
            raise SplitContractError(f"episode {episode_id} reference start differs")
        if (
            not isinstance(goals, list)
            or len(goals) != 1
            or not isinstance(goals[0], Mapping)
            or reference[-1] != goals[0].get("position")
        ):
            raise SplitContractError(f"episode {episode_id} reference goal differs")
        _scene_name(episode)
        episodes.append(episode)
    scenes = {_scene_name(episode) for episode in episodes}
    if len(scenes) != SOURCE_SCENES:
        raise SplitContractError(f"source scene count differs from fixed {SOURCE_SCENES}")
    return payload, episodes, digest


def _seed_key(*parts: str) -> tuple[str, ...]:
    text = "|".join(parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest(), *parts[1:]


def _select_development(
    episodes: Sequence[dict[str, object]],
) -> list[tuple[int, dict[str, object]]]:
    by_id = {
        str(episode["episode_id"]): (index, episode)
        for index, episode in enumerate(episodes)
    }
    selected: list[tuple[int, dict[str, object]]] = []
    for scene, episode_ids in FROZEN_DEVELOPMENT_SCENE_EPISODES:
        if len(episode_ids) != EPISODES_PER_DEVELOPMENT_SCENE:
            raise SplitContractError("frozen development scene does not contain four episodes")
        for episode_id in episode_ids:
            if episode_id not in by_id:
                raise SplitContractError(f"frozen development episode {episode_id} is absent")
            row = by_id[episode_id]
            if _scene_name(row[1]) != scene:
                raise SplitContractError(f"frozen development episode {episode_id} scene differs")
            selected.append(row)
    scenes = {_scene_name(episode) for _, episode in selected}
    if len(selected) != DEVELOPMENT_EPISODES or len(scenes) != DEVELOPMENT_SCENES:
        raise SplitContractError("fixed development inventory differs from 8 scenes x 4")
    if scenes & PRIOR_PROBE_SCENES:
        raise SplitContractError("development scenes overlap a prior A1/A2 probe")
    trajectory_ids = [str(episode["trajectory_id"]) for _, episode in selected]
    if len(set(trajectory_ids)) != len(trajectory_ids):
        raise SplitContractError("development trajectory IDs are repeated")
    return selected


def _representative_trajectories(
    scene: str,
    rows: Sequence[tuple[int, dict[str, object]]],
) -> list[tuple[int, dict[str, object]]]:
    by_trajectory: dict[str, list[tuple[int, dict[str, object]]]] = {}
    for row in rows:
        trajectory_id = str(row[1]["trajectory_id"])
        by_trajectory.setdefault(trajectory_id, []).append(row)
    representatives: list[tuple[int, dict[str, object]]] = []
    for trajectory_id, candidates in by_trajectory.items():
        representatives.append(
            min(
                candidates,
                key=lambda row: _seed_key(
                    UPDATE_SELECTION_SEED,
                    "episode",
                    scene,
                    trajectory_id,
                    str(row[1]["episode_id"]),
                ),
            )
        )
    representatives.sort(
        key=lambda row: _seed_key(
            UPDATE_SELECTION_SEED,
            "trajectory",
            scene,
            str(row[1]["trajectory_id"]),
            str(row[1]["episode_id"]),
        )
    )
    return representatives


def _select_update(
    episodes: Sequence[dict[str, object]],
    development: Sequence[tuple[int, dict[str, object]]],
) -> tuple[list[tuple[int, dict[str, object]]], list[dict[str, object]]]:
    development_scenes = {_scene_name(episode) for _, episode in development}
    by_scene: dict[str, list[tuple[int, dict[str, object]]]] = {}
    for index, episode in enumerate(episodes):
        scene = _scene_name(episode)
        if scene not in development_scenes:
            by_scene.setdefault(scene, []).append((index, episode))
    if len(by_scene) != UPDATE_SCENES:
        raise SplitContractError(
            f"source has {len(by_scene)} scenes after fixed development exclusion; expected {UPDATE_SCENES}"
        )
    scene_order = sorted(
        by_scene,
        key=lambda scene: _seed_key(UPDATE_SELECTION_SEED, "scene", scene),
    )
    representatives = {
        scene: _representative_trajectories(scene, by_scene[scene])
        for scene in scene_order
    }
    quotas = {scene: min(4, len(representatives[scene])) for scene in scene_order}
    remaining = UPDATE_EPISODES - sum(quotas.values())
    for scene in scene_order:
        if remaining == 0:
            break
        if len(representatives[scene]) >= 5:
            quotas[scene] += 1
            remaining -= 1
    if remaining != 0:
        raise SplitContractError("source capacity cannot satisfy the fixed 256-episode budget")

    selected: list[tuple[int, dict[str, object]]] = []
    scene_receipts: list[dict[str, object]] = []
    for scene_index, scene in enumerate(scene_order):
        rows = representatives[scene][: quotas[scene]]
        selected.extend(rows)
        scene_receipts.append(
            {
                "scene_order": scene_index,
                "scene": scene,
                "unique_trajectory_capacity": len(representatives[scene]),
                "quota": quotas[scene],
                "episode_ids": [str(episode["episode_id"]) for _, episode in rows],
                "trajectory_ids": [str(episode["trajectory_id"]) for _, episode in rows],
            }
        )
    if len(selected) != UPDATE_EPISODES:
        raise SplitContractError("update episode count differs from fixed 256")
    trajectory_ids = [str(episode["trajectory_id"]) for _, episode in selected]
    if len(set(trajectory_ids)) != len(trajectory_ids):
        raise SplitContractError("update trajectory IDs are not globally unique")
    update_scenes = {_scene_name(episode) for _, episode in selected}
    if update_scenes != set(scene_order) or update_scenes & development_scenes:
        raise SplitContractError("update scenes are incomplete or overlap development")
    quota_histogram: dict[int, int] = {}
    for quota in quotas.values():
        quota_histogram[quota] = quota_histogram.get(quota, 0) + 1
    if quota_histogram != {1: 1, 3: 1, 4: 3, 5: 48}:
        raise SplitContractError(f"fixed source quota histogram differs: {quota_histogram}")
    return selected, scene_receipts


def _dataset_bytes(
    source_payload: Mapping[str, object],
    rows: Sequence[tuple[int, dict[str, object]]],
) -> tuple[bytes, bytes]:
    selected = [episode for _, episode in rows]
    dataset = dict(source_payload)
    dataset["episodes"] = selected
    gt = {
        str(episode["episode_id"]): {"locations": episode["reference_path"]}
        for episode in selected
    }
    if len(gt) != len(selected):
        raise SplitContractError("selected episode IDs are not unique")
    return _gzip_bytes(_json_bytes(dataset)), _gzip_bytes(_json_bytes(gt))


def _identity(
    selection_index: int,
    source_index: int,
    episode: Mapping[str, object],
) -> dict[str, object]:
    instruction = episode.get("instruction")
    if not isinstance(instruction, Mapping):
        raise SplitContractError("selected episode instruction is not an object")
    instruction_bytes = _json_bytes(instruction)
    instruction_text = instruction.get("instruction_text")
    if not isinstance(instruction_text, str) or not instruction_text:
        raise SplitContractError("selected episode instruction text is missing")
    return {
        "index": selection_index,
        "source_index": source_index,
        "episode_id": str(episode["episode_id"]),
        "episode_object_sha256": _sha256(_json_bytes(episode)),
        "scene": _scene_name(episode),
        "scene_id": str(episode["scene_id"]),
        "trajectory_id": str(episode["trajectory_id"]),
        "instruction_id": (
            str(instruction["instruction_id"])
            if "instruction_id" in instruction
            else None
        ),
        "instruction_object_sha256": _sha256(instruction_bytes),
        "instruction_text_sha256": _sha256(instruction_text.encode("utf-8")),
        "reference_path_sha256": _sha256(_json_bytes(episode["reference_path"])),
        "reference_path_points": len(episode["reference_path"]),
    }


def build(source: Path) -> tuple[dict[str, tuple[bytes, bytes]], dict[str, object]]:
    source_payload, episodes, source_digest = _load_source(source)
    development = _select_development(episodes)
    update, update_scenes = _select_update(episodes, development)
    all_trajectory_ids = [
        str(episode["trajectory_id"])
        for _, episode in (*update, *development)
    ]
    if len(set(all_trajectory_ids)) != len(all_trajectory_ids):
        raise SplitContractError("update and development trajectory identities overlap")
    payloads = {
        "train_update": _dataset_bytes(source_payload, update),
        "train_development": _dataset_bytes(source_payload, development),
    }
    receipt: dict[str, object] = {
        "schema_name": SPLIT_SCHEMA,
        "schema_version": SPLIT_VERSION,
        "selection": {
            "update_seed": UPDATE_SELECTION_SEED,
            "development_seed": DEVELOPMENT_SELECTION_SEED,
            "update_rule": (
                "sort all 53 non-development scenes by sha256(seed|scene|scene); "
                "choose one deterministic episode per trajectory; assign base quota "
                "min(4, unique_trajectory_capacity), then add one in seeded scene "
                "order to capacity>=5 scenes until exactly 256"
            ),
            "development_rule": (
                "reuse the original A3 frozen 8 scenes x 4 identities after prior-probe exclusion"
            ),
            "source_scene_fact": {
                "source_scenes": SOURCE_SCENES,
                "fixed_development_scenes": DEVELOPMENT_SCENES,
                "available_update_scenes": UPDATE_SCENES,
                "requested_64_disjoint_update_scenes_possible": False,
                "reason": "61 source scenes minus 8 fixed development scenes leaves 53",
            },
            "quota_rule_expected_histogram": {"1": 1, "3": 1, "4": 3, "5": 48},
            "update_scenes": update_scenes,
            "prior_probe_scenes": sorted(PRIOR_PROBE_SCENES),
        },
        "source_path": str(source),
        "source_sha256": source_digest,
        "source_episode_count": len(episodes),
        "source_scene_count": len({_scene_name(episode) for episode in episodes}),
        "upstream_split_read": "train",
        "validation_split_reads": 0,
        "splits": {},
    }
    for label, rows in (("train_update", update), ("train_development", development)):
        dataset_bytes, gt_bytes = payloads[label]
        receipt["splits"][label] = {
            "habitat_split": "train",
            "episode_count": len(rows),
            "scene_count": len({_scene_name(episode) for _, episode in rows}),
            "dataset_sha256": _sha256(dataset_bytes),
            "gt_sha256": _sha256(gt_bytes),
            "episodes": [
                _identity(selection_index, source_index, episode)
                for selection_index, (source_index, episode) in enumerate(rows)
            ],
        }
    stable_selection = {
        key: value for key, value in receipt.items() if key != "source_path"
    }
    receipt["selection_manifest_sha256"] = _sha256(_json_bytes(stable_selection))
    return payloads, receipt


def _write_build(
    root: Path,
    payloads: Mapping[str, tuple[bytes, bytes]],
    receipt: Mapping[str, object],
) -> None:
    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.staging-", dir=str(root.parent)))
    try:
        for label, (dataset, gt) in payloads.items():
            train = staging / label / "train"
            train.mkdir(parents=True)
            (train / "train.json.gz").write_bytes(dataset)
            (train / "train_gt.json.gz").write_bytes(gt)
        (staging / "split_manifest.json").write_bytes(_json_bytes(receipt))
        os.replace(staging, root)
    except BaseException:
        raise


def _load_json_object(path: Path, *, label: str) -> dict[str, object]:
    if path.is_symlink() or _forbidden(path):
        raise SplitContractError(f"{label} path is a symbolic link or forbidden split")
    try:
        payload = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SplitContractError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise SplitContractError(f"{label} must contain a JSON object")
    return payload


def _episode_key(value: Mapping[str, object]) -> tuple[str, str, str]:
    episode_id = value.get("episode_id")
    scene = value.get("scene")
    instruction_sha = value.get("instruction_text_sha256")
    if not all(isinstance(item, str) and item for item in (episode_id, scene, instruction_sha)):
        raise SplitContractError("episode identity is incomplete")
    return str(episode_id), str(scene), str(instruction_sha)


def _read_and_verify_build(
    root: Path,
    source: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, object], str]:
    expected_payloads, expected_manifest = build(source)
    manifest_path = root / "split_manifest.json"
    observed_manifest = _load_json_object(manifest_path, label="split manifest")
    if observed_manifest != expected_manifest:
        raise SplitContractError("built split manifest differs from deterministic source reconstruction")
    manifest_sha = _sha256_file(manifest_path)
    loaded: dict[str, dict[str, object]] = {}
    for label in ("train_update", "train_development"):
        train = root / label / "train"
        dataset_path = train / "train.json.gz"
        gt_path = train / "train_gt.json.gz"
        if dataset_path.is_symlink() or gt_path.is_symlink():
            raise SplitContractError("built dataset files may not be symbolic links")
        dataset_bytes = dataset_path.resolve(strict=True).read_bytes()
        gt_bytes = gt_path.resolve(strict=True).read_bytes()
        expected_dataset, expected_gt = expected_payloads[label]
        if dataset_bytes != expected_dataset or gt_bytes != expected_gt:
            raise SplitContractError(f"built {label} bytes differ from deterministic source reconstruction")
        loaded[label] = _load_gzip_json_bytes(dataset_bytes, label=f"built {label} dataset")
        loaded[label + "_gt"] = _load_gzip_json_bytes(gt_bytes, label=f"built {label} ground truth")
    return observed_manifest, loaded["train_update"], loaded["train_update_gt"], manifest_sha


def _validate_plan(
    plan_path: Path,
    plan: Mapping[str, object],
    manifest: Mapping[str, object],
    manifest_sha: str,
) -> list[tuple[int, list[dict[str, object]]]]:
    if plan.get("schema_name") != PLAN_SCHEMA or plan.get("schema_version") != PLAN_VERSION:
        raise SplitContractError("parallel plan schema differs")
    if (
        plan.get("split_label") != "train_update"
        or plan.get("validation_split_reads") != 0
        or plan.get("episode_count") != UPDATE_EPISODES
    ):
        raise SplitContractError("parallel plan is not the fixed train-only update budget")
    plan_manifest = plan.get("split_manifest")
    if not isinstance(plan_manifest, Mapping) or plan_manifest.get("sha256") != manifest_sha:
        raise SplitContractError("parallel plan split-manifest SHA-256 differs")
    manifest_splits = manifest.get("splits")
    if not isinstance(manifest_splits, Mapping):
        raise SplitContractError("split manifest lacks splits")
    update = manifest_splits.get("train_update")
    if not isinstance(update, Mapping) or not isinstance(update.get("episodes"), list):
        raise SplitContractError("split manifest lacks train_update episodes")
    expected_rows = update["episodes"]
    expected_by_key: dict[tuple[str, str, str], dict[str, object]] = {}
    for order, raw in enumerate(expected_rows):
        if not isinstance(raw, dict) or raw.get("index") != order:
            raise SplitContractError("split manifest episode index differs")
        key = _episode_key(raw)
        if key in expected_by_key:
            raise SplitContractError("split manifest episode identity repeats")
        expected_by_key[key] = raw

    worker_count = plan.get("worker_count")
    workers = plan.get("workers")
    if type(worker_count) is not int or worker_count < 1:
        raise SplitContractError("parallel plan worker count is invalid")
    if not isinstance(workers, list) or len(workers) != worker_count:
        raise SplitContractError("parallel plan worker inventory differs")
    normalized: list[tuple[int, list[dict[str, object]]]] = []
    observed: list[tuple[str, str, str]] = []
    worker_ids: set[int] = set()
    for worker in workers:
        if not isinstance(worker, Mapping) or type(worker.get("worker_id")) is not int:
            raise SplitContractError("parallel plan worker row is invalid")
        worker_id = int(worker["worker_id"])
        if worker_id in worker_ids:
            raise SplitContractError("parallel plan worker id repeats")
        worker_ids.add(worker_id)
        rows = worker.get("episodes")
        if not isinstance(rows, list) or worker.get("episode_count") != len(rows):
            raise SplitContractError("parallel plan worker episode count differs")
        normalized_rows: list[dict[str, object]] = []
        prior_order = -1
        for raw in rows:
            if not isinstance(raw, Mapping) or type(raw.get("episode_order")) is not int:
                raise SplitContractError("parallel plan episode row is invalid")
            episode_order = int(raw["episode_order"])
            key = _episode_key(raw)
            if key not in expected_by_key:
                raise SplitContractError("parallel plan contains an unknown episode identity")
            if episode_order != expected_by_key[key]["index"] or episode_order <= prior_order:
                raise SplitContractError("parallel plan worker episode order differs")
            prior_order = episode_order
            normalized_rows.append(dict(raw))
            observed.append(key)
        normalized.append((worker_id, normalized_rows))
    if worker_ids != set(range(worker_count)):
        raise SplitContractError("parallel plan worker ids are not contiguous from zero")
    if len(observed) != UPDATE_EPISODES or len(set(observed)) != UPDATE_EPISODES:
        raise SplitContractError("parallel plan does not assign each episode exactly once")
    if set(observed) != set(expected_by_key):
        raise SplitContractError("parallel plan is not an exact train_update partition")
    return sorted(normalized, key=lambda item: item[0])


def _verify_worker_dataset(
    dataset_bytes: bytes,
    gt_bytes: bytes,
    expected_episodes: Sequence[Mapping[str, object]],
    parent_gt: Mapping[str, object],
) -> None:
    dataset = _load_gzip_json_bytes(dataset_bytes, label="worker train dataset")
    gt = _load_gzip_json_bytes(gt_bytes, label="worker train ground truth")
    episodes = dataset.get("episodes")
    if not isinstance(episodes, list) or episodes != list(expected_episodes):
        raise SplitContractError("worker dataset episode sequence differs from its plan")
    expected_gt = {
        str(episode["episode_id"]): parent_gt[str(episode["episode_id"])]
        for episode in expected_episodes
    }
    if gt != expected_gt:
        raise SplitContractError("worker ground truth differs from the parent train_update ground truth")


def materialize_workers(
    *,
    source: Path,
    build_root: Path,
    plan_path: Path,
    worker_root: Path,
) -> dict[str, object]:
    manifest, update_dataset, parent_gt, manifest_sha = _read_and_verify_build(build_root, source)
    plan = _load_json_object(plan_path, label="parallel plan")
    plan_path = plan_path.resolve(strict=True)
    plan_sha = _sha256_file(plan_path)
    workers = _validate_plan(plan_path, plan, manifest, manifest_sha)
    update_episodes = update_dataset.get("episodes")
    if not isinstance(update_episodes, list) or len(update_episodes) != UPDATE_EPISODES:
        raise SplitContractError("built train_update dataset episode count differs")
    episode_by_id = {str(episode["episode_id"]): episode for episode in update_episodes}
    if len(episode_by_id) != UPDATE_EPISODES:
        raise SplitContractError("built train_update episode ids repeat")
    if not isinstance(parent_gt, Mapping) or set(parent_gt) != set(episode_by_id):
        raise SplitContractError("built train_update ground truth inventory differs")

    staging = Path(
        tempfile.mkdtemp(prefix=f".{worker_root.name}.staging-", dir=str(worker_root.parent))
    )
    root_receipts: list[dict[str, object]] = []
    observed_ids: list[str] = []
    try:
        for worker_id, plan_rows in workers:
            expected_episodes = [episode_by_id[str(row["episode_id"])] for row in plan_rows]
            worker_dataset = dict(update_dataset)
            worker_dataset["episodes"] = expected_episodes
            worker_gt = {
                str(episode["episode_id"]): parent_gt[str(episode["episode_id"])]
                for episode in expected_episodes
            }
            dataset_bytes = _gzip_bytes(_json_bytes(worker_dataset))
            gt_bytes = _gzip_bytes(_json_bytes(worker_gt))
            _verify_worker_dataset(dataset_bytes, gt_bytes, expected_episodes, parent_gt)
            worker_dir = staging / f"worker_{worker_id:03d}"
            train_dir = worker_dir / "train"
            train_dir.mkdir(parents=True)
            dataset_path = train_dir / "train.json.gz"
            gt_path = train_dir / "train_gt.json.gz"
            dataset_path.write_bytes(dataset_bytes)
            gt_path.write_bytes(gt_bytes)
            if _sha256_file(dataset_path) != _sha256(dataset_bytes) or _sha256_file(gt_path) != _sha256(gt_bytes):
                raise SplitContractError("worker dataset readback SHA-256 differs")
            identity_rows = [
                {
                    "episode_order": int(plan_row["episode_order"]),
                    "episode_id": str(plan_row["episode_id"]),
                    "scene": str(plan_row["scene"]),
                    "instruction_text_sha256": str(plan_row["instruction_text_sha256"]),
                    "trajectory_id": str(episode["trajectory_id"]),
                    "episode_object_sha256": _sha256(_json_bytes(episode)),
                }
                for plan_row, episode in zip(plan_rows, expected_episodes)
            ]
            observed_ids.extend(str(episode["episode_id"]) for episode in expected_episodes)
            receipt: dict[str, object] = {
                "schema_name": WORKER_SCHEMA,
                "schema_version": WORKER_VERSION,
                "status": "materialized",
                "worker_id": worker_id,
                "run_id": plan.get("run_id"),
                "habitat_split": "train",
                "validation_split_reads": 0,
                "plan_sha256": plan_sha,
                "split_manifest_sha256": manifest_sha,
                "episode_count": len(expected_episodes),
                "dataset_relative_path": f"worker_{worker_id:03d}/train/train.json.gz",
                "dataset_sha256": _sha256(dataset_bytes),
                "gt_relative_path": f"worker_{worker_id:03d}/train/train_gt.json.gz",
                "gt_sha256": _sha256(gt_bytes),
                "episodes": identity_rows,
            }
            (worker_dir / "materialization_receipt.json").write_bytes(_json_bytes(receipt))
            root_receipts.append(receipt)
        if len(observed_ids) != UPDATE_EPISODES or len(set(observed_ids)) != UPDATE_EPISODES:
            raise SplitContractError("materialized workers are not an exact unique 256-episode partition")
        if set(observed_ids) != set(episode_by_id):
            raise SplitContractError("materialized worker union differs from train_update")
        root_receipt: dict[str, object] = {
            "schema_name": MATERIALIZATION_SCHEMA,
            "schema_version": MATERIALIZATION_VERSION,
            "status": "materialized",
            "run_id": plan.get("run_id"),
            "habitat_split": "train",
            "validation_split_reads": 0,
            "source_sha256": SOURCE_TRAIN_SHA256,
            "split_manifest": {
                "path": str(build_root / "split_manifest.json"),
                "sha256": manifest_sha,
            },
            "parallel_plan": {"path": str(plan_path), "sha256": plan_sha},
            "worker_count": len(root_receipts),
            "episode_count": len(observed_ids),
            "worker_episode_counts": [receipt["episode_count"] for receipt in root_receipts],
            "workers": root_receipts,
        }
        root_receipt["partition_sha256"] = _sha256(_json_bytes(root_receipt))
        (staging / "worker_materialization_manifest.json").write_bytes(_json_bytes(root_receipt))
        os.replace(staging, worker_root)
        return root_receipt
    except BaseException:
        raise


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source", required=True, type=Path)
    result.add_argument(
        "--output-root",
        type=Path,
        help="new build root, or existing build root with --materialize-workers",
    )
    result.add_argument("--check-only", action="store_true")
    result.add_argument(
        "--materialize-workers",
        metavar="PLAN_JSON",
        type=Path,
        help="materialize an existing built train_update using this parallel plan",
    )
    result.add_argument("--worker-root", type=Path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    source = _source_path(args.source)
    if args.materialize_workers is not None:
        if args.check_only:
            raise SplitContractError("--check-only cannot be combined with --materialize-workers")
        if args.output_root is None or args.worker_root is None:
            raise SplitContractError(
                "worker materialization requires existing --output-root and new --worker-root"
            )
        build_root = _existing_build_root(args.output_root, source)
        worker_root = _new_root(args.worker_root, source=source)
        receipt = materialize_workers(
            source=source,
            build_root=build_root,
            plan_path=args.materialize_workers,
            worker_root=worker_root,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
        return 0
    if args.worker_root is not None:
        raise SplitContractError("--worker-root requires --materialize-workers")
    payloads, receipt = build(source)
    if not args.check_only:
        if args.output_root is None:
            raise SplitContractError("build mode requires --output-root")
        root = _new_root(args.output_root, source=source)
        _write_build(root, payloads, receipt)
    elif args.output_root is not None:
        raise SplitContractError("--check-only does not accept --output-root")
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SplitContractError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
