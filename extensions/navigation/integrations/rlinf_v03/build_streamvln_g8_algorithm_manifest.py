"""Build the outcome-free three-seed identity manifest for StreamVLN G8."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path


SEEDS = (2026092808, 2026092809, 2026092810)
TRAIN_COUNT = 32
TRAIN_MIN_SCENES = 16
DEVELOPMENT_COUNT = 22
DEVELOPMENT_MIN_SCENES = 22
TRAIN_SHA256 = "340a80133b2157520354ab055a91d98feb2f42e4bbda17b200c911f8788492ea"
PREREGISTRATION_SHA256 = "dbc4b3927ccd308263a6c2ad34b7fbdbf49ffb84d6eae3b5c704fbe6e74609da"
ENGINEERING_RESULT_SHA256 = "cbcc6c2be7b94f8decaf64b6ec1f98952220e9b0db80a6f854d5fd4a47e67d94"
ENGINEERING_EXCLUSIONS = frozenset({
    (70, "71", "i5noydFURQK"),
    (3944, "3954", "e9zR4mvMWw7"),
    (5052, "5062", "vyrNrziPKCB"),
})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_identities(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        raw = json.load(source)
    episodes = raw.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f"R2R split has no episodes: {path}")
    rows = []
    seen = set()
    for index, episode in enumerate(episodes):
        row = {
            "index": index,
            "episode_id": str(episode["episode_id"]),
            "scene": Path(episode["scene_id"]).stem,
        }
        identity = (row["index"], row["episode_id"], row["scene"])
        if identity in seen:
            raise ValueError("duplicate R2R identity")
        seen.add(identity)
        rows.append(row)
    return rows


def rank(seed: int, role: str, row: dict) -> str:
    text = (
        f"g8|{seed}|{role}|{row['index']}|{row['episode_id']}|{row['scene']}"
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def select_scene_spread(
    rows: list[dict],
    *,
    seed: int,
    role: str,
    count: int,
    minimum_scenes: int,
) -> list[dict]:
    by_scene = defaultdict(list)
    for row in rows:
        by_scene[row["scene"]].append(row)
    if len(by_scene) < minimum_scenes or len(rows) < count:
        raise ValueError(f"insufficient {role} identities or scenes")
    scene_order = sorted(
        by_scene,
        key=lambda scene: hashlib.sha256(
            f"g8|{seed}|{role}|scene|{scene}".encode("utf-8")
        ).hexdigest(),
    )
    selected = []
    selected_keys = set()
    for scene in scene_order[:minimum_scenes]:
        row = min(by_scene[scene], key=lambda item: rank(seed, role, item))
        selected.append(row)
        selected_keys.add((row["index"], row["episode_id"], row["scene"]))
    remainder = sorted(rows, key=lambda item: rank(seed, role, item))
    for row in remainder:
        if len(selected) == count:
            break
        key = (row["index"], row["episode_id"], row["scene"])
        if key not in selected_keys:
            selected.append(row)
            selected_keys.add(key)
    if len(selected) != count or len({row["scene"] for row in selected}) < minimum_scenes:
        raise RuntimeError(f"{role} selection contract failed")
    return selected


def build_manifest(asset_root: Path) -> dict:
    train_path = asset_root / "datasets" / "r2r" / "train" / "train.json.gz"
    development_path = (
        asset_root / "datasets" / "r2r" / "val_seen" / "val_seen.json.gz"
    )
    train_hash = sha256_file(train_path.resolve(strict=True))
    development_hash = sha256_file(development_path.resolve(strict=True))
    if train_hash != TRAIN_SHA256:
        raise RuntimeError("G8 train dataset hash changed")
    train_rows = [
        row for row in load_identities(train_path)
        if (row["index"], row["episode_id"], row["scene"])
        not in ENGINEERING_EXCLUSIONS
    ]
    development_rows = load_identities(development_path)
    arms = {}
    for seed in SEEDS:
        train = select_scene_spread(
            train_rows,
            seed=seed,
            role="train",
            count=TRAIN_COUNT,
            minimum_scenes=TRAIN_MIN_SCENES,
        )
        development = select_scene_spread(
            development_rows,
            seed=seed,
            role="development",
            count=DEVELOPMENT_COUNT,
            minimum_scenes=DEVELOPMENT_MIN_SCENES,
        )
        arms[str(seed)] = {
            "train": train,
            "development": development,
            "counts": {
                "train_episodes": len(train),
                "train_scenes": len({row["scene"] for row in train}),
                "development_episodes": len(development),
                "development_scenes": len({row["scene"] for row in development}),
            },
        }
    return {
        "schema_version": 1,
        "stage": "G8 matched BC versus BC+RL algorithm gate",
        "created": "2026-09-28",
        "selection_timing": "before_any_g8_algorithm_model_load",
        "selection_uses_outcome_fields": False,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "engineering_result_sha256": ENGINEERING_RESULT_SHA256,
        "source": {
            "train_path": str(train_path),
            "train_sha256": train_hash,
            "val_seen_path": str(development_path),
            "val_seen_sha256": development_hash,
            "val_unseen_read": False,
        },
        "selection_rule": (
            "sha256('g8|seed|role|index|episode_id|scene') with deterministic "
            "scene-spread seeding; identity fields only"
        ),
        "engineering_exclusions": [
            {"index": index, "episode_id": episode_id, "scene": scene}
            for index, episode_id, scene in sorted(ENGINEERING_EXCLUSIONS)
        ],
        "seeds": list(SEEDS),
        "arms": arms,
        "training": {
            "max_bc_decisions": 32,
            "bc_epochs": 2,
            "bc_batch_size": 32,
            "bc_sampling": "all NAVIGATE rows plus deterministic repetition of STOP rows to match the NAVIGATE count; retain any causal RECOVER rows",
            "bc_stop_balance_required": True,
            "max_rl_decisions": 20,
            "ppo_epochs": 1,
            "ppo_batch_size": 32,
            "learning_rate": 0.000003,
            "value_learning_rate": 0.000003,
            "clip_ratio": 0.2,
            "value_coefficient": 0.5,
            "entropy_coefficient": 0.01,
            "bc_auxiliary_coefficient": 0.1,
            "gamma_per_primitive": 0.99,
            "gae_lambda_per_macro": 0.95,
            "expert_lookahead_m": 0.75,
            "final_collection_ratio_max_deviation": 0.5,
            "final_collection_categorical_kl_max": 0.1,
            "all_episodes_collected_before_update": True
        },
        "evaluation": {
            "split": "val_seen",
            "max_primitive_steps": 160,
            "deterministic": True,
            "matched_bc_and_bc_rl_schedule": True,
            "val_unseen_episode_count": 0
        },
        "decision_rule": {
            "primary_endpoint": "development_success_rate",
            "seed_majority": "BC+RL SR greater than BC SR in at least two of three seeds",
            "pooled": "pooled BC+RL SR greater than pooled BC SR",
            "noninferiority": "pooled SPL does not decrease and timeout count does not increase",
            "no_collapse": "BC+RL executes at least two action modes and at least two primitive categories",
            "all_training_and_evaluation_guards_must_pass": True
        },
        "claim_boundary": {
            "development_only": True,
            "val_unseen_access": False,
            "robot_execution": False,
            "lower_executor_fixed": True,
            "original_streamvln_only": True,
            "ttt_or_secondary_post_training": False
        }
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    manifest = build_manifest(args.asset_root.expanduser().resolve(strict=True))
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(output),
        "sha256": sha256_file(output),
        "counts": {seed: manifest["arms"][seed]["counts"] for seed in manifest["arms"]},
        "val_unseen_read": manifest["source"]["val_unseen_read"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
