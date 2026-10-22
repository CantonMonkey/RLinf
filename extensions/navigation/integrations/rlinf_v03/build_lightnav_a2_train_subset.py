#!/usr/bin/env python3
"""Build LightNav A2's fixed train-only R2R dataset and adjacent NDTW GT.

Only a literal ``train/train.json.gz`` source is opened.  The output directory
must also be named ``train`` and receives ``train.json.gz``,
``train_gt.json.gz``, and a build receipt.  No validation split is discovered
or accepted.
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


SOURCE_TRAIN_SHA256 = (
    "340a80133b2157520354ab055a91d98feb2f42e4bbda17b200c911f8788492ea"
)
FORBIDDEN_SPLIT_PARTS = frozenset(
    {"val", "val_seen", "val_unseen", "val_seen_synonyms"}
)
FIXED_EPISODES = (
    {"episode_id": "4836", "index": 4826, "scene": "mJXqzFtmKg4"},
    {"episode_id": "10622", "index": 10603, "scene": "1pXnuDYAj8r"},
    {"episode_id": "6520", "index": 6510, "scene": "qoiz87JEwZ2"},
    {"episode_id": "7298", "index": 7288, "scene": "p5wJjkQkbXX"},
    {"episode_id": "2856", "index": 2849, "scene": "E9uDoFAP3SH"},
    {"episode_id": "6921", "index": 6911, "scene": "S9hNv5qa7GM"},
    {"episode_id": "5572", "index": 5562, "scene": "Vvot9Ly1tCj"},
    {"episode_id": "1937", "index": 1933, "scene": "VLzqgDo317F"},
)


class ContractError(RuntimeError):
    """Raised before output when the fixed train-only contract differs."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _contains_forbidden_split(path: Path) -> bool:
    return any(part.casefold() in FORBIDDEN_SPLIT_PARTS for part in path.parts)


def _source_path(value: Path) -> Path:
    requested = value.expanduser()
    if (
        requested.name != "train.json.gz"
        or requested.parent.name != "train"
        or _contains_forbidden_split(requested)
    ):
        # Reject lexically before any filesystem probe, so a validation path is
        # never resolved, opened, or even checked for existence.
        raise ContractError("source must be a validation-free train/train.json.gz path")
    if requested.is_symlink():
        raise ContractError("source train dataset may not be a symbolic link")
    path = requested.resolve(strict=True)
    if not path.is_file():
        raise ContractError("source train dataset is not a regular file")
    if path.name != "train.json.gz" or path.parent.name != "train":
        raise ContractError("source must be the literal train/train.json.gz path")
    if _contains_forbidden_split(path):
        raise ContractError("source path contains a forbidden validation split")
    return path


def _output_path(value: Path, source: Path) -> Path:
    requested = value.expanduser().absolute()
    if requested.name != "train" or _contains_forbidden_split(requested):
        raise ContractError("output must be a validation-free directory named train")
    if requested.exists() or requested.is_symlink():
        raise ContractError("output train directory must not already exist")
    parent = requested.parent.resolve(strict=True)
    path = parent / requested.name
    if path.name != "train":
        raise ContractError("output directory must be named train")
    if _contains_forbidden_split(path):
        raise ContractError("output path contains a forbidden validation split")
    if path == source.parent:
        raise ContractError("output may not replace the source train directory")
    return path


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _finite_xyz_path(value: object) -> bool:
    return bool(
        isinstance(value, list)
        and value
        and all(
            isinstance(point, list)
            and len(point) == 3
            and all(_finite_number(item) for item in point)
            for point in value
        )
    )


def _load_fixed_episodes(source: Path) -> tuple[dict, list[dict], dict]:
    compressed = source.read_bytes()
    source_sha256 = _sha256(compressed)
    if source_sha256 != SOURCE_TRAIN_SHA256:
        raise ContractError("source train dataset SHA-256 differs from A2")
    try:
        payload = json.loads(gzip.decompress(compressed).decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError("source train dataset is not valid UTF-8 gzip JSON") from error
    if not isinstance(payload, dict) or set(payload) != {
        "episodes",
        "instruction_vocab",
    }:
        raise ContractError("source train dataset top-level schema differs")
    episodes = payload["episodes"]
    if not isinstance(episodes, list):
        raise ContractError("source episodes field is not a list")

    selected: list[dict] = []
    identities: list[dict] = []
    for expected in FIXED_EPISODES:
        matches = [
            (index, episode)
            for index, episode in enumerate(episodes)
            if isinstance(episode, Mapping)
            and str(episode.get("episode_id")) == expected["episode_id"]
        ]
        if len(matches) != 1:
            raise ContractError(
                f"episode {expected['episode_id']} occurs {len(matches)} times"
            )
        index, episode = matches[0]
        scene_id = f"mp3d/{expected['scene']}/{expected['scene']}.glb"
        if index != expected["index"] or episode.get("scene_id") != scene_id:
            raise ContractError(
                f"episode {expected['episode_id']} index/scene differs from A2"
            )
        reference_path = episode.get("reference_path")
        goals = episode.get("goals")
        if not _finite_xyz_path(reference_path):
            raise ContractError(
                f"episode {expected['episode_id']} reference_path is not finite XYZ"
            )
        if reference_path[0] != episode.get("start_position"):
            raise ContractError(
                f"episode {expected['episode_id']} reference path start differs"
            )
        if (
            not isinstance(goals, list)
            or len(goals) != 1
            or not isinstance(goals[0], Mapping)
            or reference_path[-1] != goals[0].get("position")
        ):
            raise ContractError(
                f"episode {expected['episode_id']} reference path goal differs"
            )
        info = episode.get("info")
        distance = info.get("geodesic_distance") if isinstance(info, Mapping) else None
        if not _finite_number(distance) or float(distance) <= 0.0:
            raise ContractError(
                f"episode {expected['episode_id']} lacks finite geodesic distance"
            )
        identities.append(
            {
                **expected,
                "geodesic_distance_m": float(distance),
                "reference_path_point_count": len(reference_path),
            }
        )
        selected.append(dict(episode))

    receipt = {
        "schema_name": "lightnav_a2_train_subset_build",
        "schema_version": 1,
        "split": "train",
        "source_path": str(source),
        "source_sha256": source_sha256,
        "source_episode_count": len(episodes),
        "selected_episode_count": len(selected),
        "selected_episodes": identities,
        "gt_schema": "mapping[string_episode_id].locations=list[xyz]",
    }
    return payload, selected, receipt


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _gzip_bytes(payload: bytes) -> bytes:
    stream = BytesIO()
    with gzip.GzipFile(
        filename="", fileobj=stream, mode="wb", compresslevel=9, mtime=0
    ) as archive:
        archive.write(payload)
    return stream.getvalue()


def _build_bytes(source_payload: dict, selected: Sequence[dict]) -> tuple[bytes, bytes]:
    filtered = dict(source_payload)
    filtered["episodes"] = list(selected)
    gt = {
        str(episode["episode_id"]): {"locations": episode["reference_path"]}
        for episode in selected
    }
    if len(gt) != len(selected):
        raise ContractError("selected episode IDs are not unique for NDTW GT")
    return _gzip_bytes(_json_bytes(filtered)), _gzip_bytes(_json_bytes(gt))


def _write_atomically(output: Path, dataset: bytes, gt: bytes, receipt: dict) -> None:
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=str(output.parent))
    )
    try:
        (staging / "train.json.gz").write_bytes(dataset)
        (staging / "train_gt.json.gz").write_bytes(gt)
        (staging / "build_receipt.json").write_bytes(_json_bytes(receipt))
        os.replace(staging, output)
    except BaseException:
        for name in ("train.json.gz", "train_gt.json.gz", "build_receipt.json"):
            path = staging / name
            if path.is_file() and not path.is_symlink():
                path.unlink()
        if staging.exists():
            staging.rmdir()
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = _source_path(args.source)
    source_payload, selected, receipt = _load_fixed_episodes(source)
    dataset, gt = _build_bytes(source_payload, selected)
    receipt.update(
        {
            "dataset_sha256": _sha256(dataset),
            "gt_sha256": _sha256(gt),
        }
    )
    if not args.check_only:
        if args.output_dir is None:
            raise ContractError("build mode requires --output-dir")
        output = _output_path(args.output_dir, source)
        receipt.update(
            {
                "dataset_path": str(output / "train.json.gz"),
                "gt_path": str(output / "train_gt.json.gz"),
            }
        )
        _write_atomically(output, dataset, gt, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
