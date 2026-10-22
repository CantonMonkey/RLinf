#!/usr/bin/env python3
"""Freeze LightNav A3's scene-disjoint train-only update/evaluation data.

The command lexically accepts only ``train/train.json.gz`` and never searches
for another split.  It deterministically selects four trajectories from each
of 16 update scenes and eight disjoint train-development scenes, yielding
64+32 episodes.  Both outputs remain literal Habitat ``train`` datasets; the
development label describes experiment use, not an upstream validation split.
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
UPDATE_SELECTION_SEED = "lightnav-a3-scene-disjoint-v2-20260930"
DEVELOPMENT_SELECTION_SEED = "lightnav-a3-dev-prior-scene-excluded-v1-20260930"
UPDATE_SCENES = 16
DEVELOPMENT_SCENES = 8
EPISODES_PER_SCENE = 4
FORBIDDEN_PARTS = frozenset({"val", "val_seen", "val_unseen", "val_seen_synonyms"})
FROZEN_SCENE_EPISODES = {
    "train_update": (
        ("E9uDoFAP3SH", ("5849", "3882", "1198", "7807")),
        ("r47D5H71a5s", ("727", "3134", "2147", "867")),
        ("HxpKQynjfin", ("8557", "4863", "7043", "2956")),
        ("mJXqzFtmKg4", ("2714", "2980", "3657", "1347")),
        ("D7N2EKCX4Sj", ("9137", "8353", "9825", "2796")),
        ("s8pcmisQ38h", ("9264", "7565", "8988", "5540")),
        ("p5wJjkQkbXX", ("3398", "2118", "2556", "8909")),
        ("2n8kARJN3HM", ("3796", "2587", "6719", "9480")),
        ("5LpN3gDmAk7", ("4284", "5926", "7203", "3644")),
        ("aayBHfsNo7d", ("7217", "6705", "1818", "921")),
        ("Pm6F8kyY3z2", ("6901", "8915", "10749", "10565")),
        ("PX4nDJXEHrG", ("3374", "3961", "7124", "9876")),
        ("dhjEzFoUFzH", ("5706", "7236", "9915", "10477")),
        ("jh4fc5c5qoQ", ("10037", "732", "1203", "4611")),
        ("82sE5b5pLXE", ("7708", "3754", "3786", "2340")),
        ("b8cTxDM8gDG", ("3830", "7448", "7024", "6232")),
    ),
    "train_development": (
        ("XcA2TqTSSAj", ("9449", "2214", "10529", "3864")),
        ("17DRP5sb8fy", ("2890", "2944", "3372", "9072")),
        ("JF19kD82Mey", ("7282", "2450", "132", "1069")),
        ("B6ByNegPMKs", ("6417", "1548", "3768", "7332")),
        ("uNb9QFRL6hY", ("10801", "9861", "2880", "3984")),
        ("1LXtFkjw3qL", ("2093", "4909", "6914", "10765")),
        ("ULsKaCPVFJR", ("2413", "9162", "9022", "9965")),
        ("JeFG25nYj2p", ("5842", "5954", "4992", "695")),
    ),
}
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
    """Raised before output when the train-only split contract differs."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _forbidden(path: Path) -> bool:
    return any(part.casefold() in FORBIDDEN_PARTS for part in path.parts)


def _source_path(value: Path) -> Path:
    requested = value.expanduser()
    if requested.name != "train.json.gz" or requested.parent.name != "train" or _forbidden(requested):
        raise SplitContractError("source must be a validation-free train/train.json.gz path")
    if requested.is_symlink():
        raise SplitContractError("source dataset may not be a symbolic link")
    path = requested.resolve(strict=True)
    if not path.is_file() or path.name != "train.json.gz" or path.parent.name != "train":
        raise SplitContractError("source must be the literal train/train.json.gz file")
    if _forbidden(path):
        raise SplitContractError("source path contains a forbidden split")
    return path


def _output_root(value: Path, source: Path) -> Path:
    requested = value.expanduser().absolute()
    if _forbidden(requested):
        raise SplitContractError("output path contains a forbidden split")
    if requested.exists() or requested.is_symlink():
        raise SplitContractError("output root must not already exist")
    root = requested.parent.resolve(strict=True) / requested.name
    if root == source.parent or source in root.parents:
        raise SplitContractError("output may not replace or contain the source")
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


def _load(source: Path) -> tuple[dict, list[dict], str]:
    compressed = source.read_bytes()
    digest = _sha256(compressed)
    if digest != SOURCE_TRAIN_SHA256:
        raise SplitContractError("source train dataset SHA-256 differs from A3")
    try:
        payload = json.loads(gzip.decompress(compressed).decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SplitContractError("source is not valid UTF-8 gzip JSON") from error
    if not isinstance(payload, dict) or set(payload) != {"episodes", "instruction_vocab"}:
        raise SplitContractError("source top-level schema differs")
    raw = payload["episodes"]
    if not isinstance(raw, list):
        raise SplitContractError("source episodes is not a list")
    episodes: list[dict] = []
    identities: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise SplitContractError(f"episode {index} is not an object")
        episode = dict(item)
        episode_id = str(episode.get("episode_id", ""))
        if not episode_id or episode_id in identities:
            raise SplitContractError("source episode IDs are missing or repeated")
        identities.add(episode_id)
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
    return payload, episodes, digest


def _select(episodes: Sequence[dict]) -> tuple[list[tuple[int, dict]], list[tuple[int, dict]]]:
    by_id: dict[str, tuple[int, dict]] = {
        str(episode["episode_id"]): (index, episode)
        for index, episode in enumerate(episodes)
    }

    def choose(specification: Sequence[tuple[str, Sequence[str]]]) -> list[tuple[int, dict]]:
        selected: list[tuple[int, dict]] = []
        for scene, episode_ids in specification:
            if len(episode_ids) != EPISODES_PER_SCENE:
                raise SplitContractError("frozen scene does not contain four episodes")
            for episode_id in episode_ids:
                if episode_id not in by_id:
                    raise SplitContractError(f"frozen episode {episode_id} is absent")
                row = by_id[episode_id]
                if _scene_name(row[1]) != scene:
                    raise SplitContractError(f"frozen episode {episode_id} scene differs")
                selected.append(row)
        # Preserve the frozen scene/episode order.  Habitat's scene grouping is
        # then an identity transform even though each selected scene has four
        # episodes.
        return selected

    update = choose(FROZEN_SCENE_EPISODES["train_update"])
    development = choose(FROZEN_SCENE_EPISODES["train_development"])
    if len(update) != 64 or len(development) != 32:
        raise SplitContractError("selected episode count differs from 64+32")
    if {_scene_name(row) for _, row in update} & {_scene_name(row) for _, row in development}:
        raise SplitContractError("update and development scenes overlap")
    if {_scene_name(row) for _, row in development} & PRIOR_PROBE_SCENES:
        raise SplitContractError("development scenes overlap a prior A1/A2 probe")
    trajectory_ids = [str(row.get("trajectory_id", "")) for _, row in (*update, *development)]
    if any(not value for value in trajectory_ids) or len(set(trajectory_ids)) != len(trajectory_ids):
        raise SplitContractError("selected trajectory IDs are missing or repeated")
    return update, development


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _gzip_bytes(payload: bytes) -> bytes:
    stream = BytesIO()
    with gzip.GzipFile(filename="", fileobj=stream, mode="wb", compresslevel=9, mtime=0) as archive:
        archive.write(payload)
    return stream.getvalue()


def _dataset_bytes(source_payload: dict, rows: Sequence[tuple[int, dict]]) -> tuple[bytes, bytes]:
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


def _identity(index: int, episode: Mapping[str, object]) -> dict[str, object]:
    instruction = episode.get("instruction")
    if not isinstance(instruction, Mapping):
        raise SplitContractError("selected episode instruction is not an object")
    instruction_bytes = _json_bytes(instruction)
    instruction_text = instruction.get("instruction_text")
    if not isinstance(instruction_text, str) or not instruction_text:
        raise SplitContractError("selected episode instruction text is missing")
    return {
        "index": index,
        "episode_id": str(episode["episode_id"]),
        "scene": _scene_name(episode),
        "trajectory_id": str(episode.get("trajectory_id")),
        "instruction_id": (
            str(instruction["instruction_id"])
            if "instruction_id" in instruction
            else None
        ),
        "instruction_object_sha256": _sha256(instruction_bytes),
        "instruction_text_sha256": _sha256(instruction_text.encode("utf-8")),
        "reference_path_points": len(episode["reference_path"]),
    }


def build(source: Path) -> tuple[dict[str, tuple[bytes, bytes]], dict[str, object]]:
    source_payload, episodes, source_digest = _load(source)
    update, development = _select(episodes)
    payloads = {
        "train_update": _dataset_bytes(source_payload, update),
        "train_development": _dataset_bytes(source_payload, development),
    }
    receipt: dict[str, object] = {
        "schema_name": "lightnav_a3_train_scene_split",
        "schema_version": 1,
        "selection": {
            "update_seed": UPDATE_SELECTION_SEED,
            "development_seed": DEVELOPMENT_SELECTION_SEED,
            "update_rule": (
                "precomputed by ascending sha256(update_seed|scene|scene), then "
                "sha256(update_seed|episode|episode_id) with first unique trajectory; "
                "frozen as 16 scenes x 4 in manifest order"
            ),
            "development_rule": (
                "precomputed after excluding prior A1/A2 and frozen update scenes, by "
                "ascending sha256(development_seed|scene|scene), then sha256("
                "development_seed|episode|scene|trajectory_id|episode_id) with first "
                "unique trajectory; frozen as 8 scenes x 4 in manifest order"
            ),
            "prior_probe_scenes": sorted(PRIOR_PROBE_SCENES),
        },
        "source_path": str(source),
        "source_sha256": source_digest,
        "source_episode_count": len(episodes),
        "source_scene_count": len({_scene_name(row) for row in episodes}),
        "upstream_split_read": "train",
        "validation_split_reads": 0,
        "splits": {},
    }
    for label, rows in (("train_update", update), ("train_development", development)):
        dataset_bytes, gt_bytes = payloads[label]
        receipt["splits"][label] = {
            "habitat_split": "train",
            "episode_count": len(rows),
            "scene_count": len({_scene_name(row) for _, row in rows}),
            "dataset_sha256": _sha256(dataset_bytes),
            "gt_sha256": _sha256(gt_bytes),
            "episodes": [_identity(index, episode) for index, episode in rows],
        }
    stable_selection = {
        key: value for key, value in receipt.items() if key != "source_path"
    }
    receipt["selection_manifest_sha256"] = _sha256(_json_bytes(stable_selection))
    return payloads, receipt


def _write(root: Path, payloads: Mapping[str, tuple[bytes, bytes]], receipt: Mapping[str, object]) -> None:
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
        # Preserve a failed staging tree for diagnosis.  It is bounded to two
        # tiny filtered datasets and must never replace the requested target.
        raise


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source", required=True, type=Path)
    result.add_argument("--output-root", type=Path)
    result.add_argument("--check-only", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    source = _source_path(args.source)
    payloads, receipt = build(source)
    if not args.check_only:
        if args.output_root is None:
            raise SplitContractError("build mode requires --output-root")
        root = _output_root(args.output_root, source)
        _write(root, payloads, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SplitContractError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
