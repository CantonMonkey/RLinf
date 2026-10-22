"""Read-only provenance and split check for R2R VLN-CE episode/scene assets.

This tool does not load Habitat, initialize CUDA, or alter the source files.
It intentionally prints episode identity and hashes, never instruction text.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path, PurePosixPath


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_episodes(path: Path) -> tuple[list[dict], dict]:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        document = json.load(source)
    if not isinstance(document, dict) or not isinstance(document.get("episodes"), list):
        raise ValueError(f"Invalid R2R episode document: {path}")
    vocab = document.get("instruction_vocab")
    if not isinstance(vocab, dict) or not isinstance(vocab.get("word_list"), list):
        raise ValueError(f"Missing instruction_vocab required by Habitat VLNDatasetV1: {path}")
    episodes = document["episodes"]
    if not episodes:
        raise ValueError(f"Empty R2R episode document: {path}")
    return episodes, vocab


def token_range(episodes: list[dict], vocab_size: int) -> dict:
    minimum = vocab_size
    maximum = -1
    count = 0
    for index, episode in enumerate(episodes):
        instruction = episode.get("instruction") if isinstance(episode, dict) else None
        tokens = instruction.get("instruction_tokens") if isinstance(instruction, dict) else None
        if not isinstance(tokens, list) or not tokens:
            raise ValueError(f"Episode {index} has no instruction tokens")
        for token in tokens:
            if isinstance(token, bool) or not isinstance(token, int) or not 0 <= token < vocab_size:
                raise ValueError(f"Episode {index} has a token outside the vocabulary")
            minimum = min(minimum, token)
            maximum = max(maximum, token)
            count += 1
    return {"min": minimum, "max": maximum, "count": count}


def scene_name(episode: dict) -> str:
    scene = episode.get("scene_id")
    if not isinstance(scene, str) or not scene or "\\" in scene:
        raise ValueError("Invalid scene_id")
    path = PurePosixPath(scene)
    if path.is_absolute() or ".." in path.parts or path.suffix != ".glb":
        raise ValueError(f"Unsafe scene_id: {scene}")
    return scene


def finite_vector(value: object, length: int, label: str) -> None:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"Invalid {label}")
    if any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in value):
        raise ValueError(f"Invalid {label}")


def selected_episode(episodes: list[dict], index: int, scene_root: Path, pad_index: int) -> dict:
    if not 0 <= index < len(episodes):
        raise ValueError(f"Episode index {index} is outside 0..{len(episodes) - 1}")
    episode = episodes[index]
    if not isinstance(episode, dict):
        raise ValueError(f"Episode {index} is not an object")
    scene = scene_name(episode)
    instruction = episode.get("instruction")
    if not isinstance(instruction, dict) or not isinstance(
        instruction.get("instruction_text"), str
    ) or not instruction["instruction_text"].strip():
        raise ValueError(f"Episode {index} has no instruction text")
    finite_vector(episode.get("start_position"), 3, "start_position")
    finite_vector(episode.get("start_rotation"), 4, "start_rotation")
    goals = episode.get("goals")
    if not isinstance(goals, list) or not goals or not isinstance(goals[0], dict):
        raise ValueError(f"Episode {index} has no goal")
    finite_vector(goals[0].get("position"), 3, "goal position")

    root = scene_root.resolve(strict=True)
    glb = (root / scene).resolve(strict=True)
    if not glb.is_relative_to(root) or not glb.is_file():
        raise ValueError(f"Scene is outside root or is not a file: {scene}")
    navmesh = glb.with_suffix(".navmesh")
    if not navmesh.is_file():
        raise FileNotFoundError(f"Missing navmesh for {scene}: {navmesh}")

    identity = {
        "index": index,
        "episode_id": str(episode.get("episode_id")),
        "trajectory_id": str(episode.get("trajectory_id")),
        "scene_id": scene,
        "episode_sha256": hashlib.sha256(
            json.dumps(episode, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "instruction_chars": len(instruction["instruction_text"]),
        "instruction_token_array_len": len(instruction["instruction_tokens"]),
        "instruction_non_pad_tokens": sum(
            token != pad_index for token in instruction["instruction_tokens"]
        ),
        "scene_glb": str(glb),
        "scene_glb_bytes": glb.stat().st_size,
        "scene_glb_sha256": sha256_file(glb),
        "navmesh": str(navmesh),
        "navmesh_sha256": sha256_file(navmesh),
    }
    return identity


def preflight(
    train_path: Path,
    val_unseen_path: Path,
    scene_root: Path,
    train_index: int,
    val_unseen_index: int,
) -> dict:
    train, train_vocab = load_episodes(train_path)
    val_unseen, val_vocab = load_episodes(val_unseen_path)
    if train_vocab != val_vocab:
        raise ValueError("Train and val_unseen instruction vocabularies differ")
    words = train_vocab["word_list"]
    if not words:
        raise ValueError("Empty instruction vocabulary")
    pad_index = train_vocab.get("PAD_INDEX")
    if isinstance(pad_index, bool) or not isinstance(pad_index, int) or not 0 <= pad_index < len(words):
        raise ValueError("Invalid instruction vocabulary PAD_INDEX")
    vocabulary_sha256 = hashlib.sha256(
        json.dumps(train_vocab, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    train_token_range = token_range(train, len(words))
    val_token_range = token_range(val_unseen, len(words))
    train_scenes = {scene_name(episode) for episode in train}
    val_scenes = {scene_name(episode) for episode in val_unseen}
    overlap = sorted(train_scenes & val_scenes)
    if overlap:
        raise ValueError(f"Train and val_unseen share {len(overlap)} scenes: {overlap[:3]}")

    return {
        "status": "ok",
        "scene_root": str(scene_root.resolve(strict=True)),
        "instruction_vocab_sha256": vocabulary_sha256,
        "instruction_vocab_size": len(words),
        "instruction_pad_index": pad_index,
        "train": {
            "source": str(train_path.resolve(strict=True)),
            "source_sha256": sha256_file(train_path),
            "episode_count": len(train),
            "scene_count": len(train_scenes),
            "token_range": train_token_range,
            "selected": selected_episode(train, train_index, scene_root, pad_index),
        },
        "val_unseen": {
            "source": str(val_unseen_path.resolve(strict=True)),
            "source_sha256": sha256_file(val_unseen_path),
            "episode_count": len(val_unseen),
            "scene_count": len(val_scenes),
            "token_range": val_token_range,
            "selected": selected_episode(val_unseen, val_unseen_index, scene_root, pad_index),
        },
        "train_val_unseen_scene_overlap": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val-unseen", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--train-index", type=int, required=True)
    parser.add_argument("--val-unseen-index", type=int, required=True)
    args = parser.parse_args()
    result = preflight(
        args.train, args.val_unseen, args.scene_root,
        args.train_index, args.val_unseen_index,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
