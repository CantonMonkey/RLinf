"""Small fixtures for the R2R source and split preflight."""

import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from integrations.rlinf_v03.vln_asset_preflight import preflight


def episode(scene: str, episode_id: int, instruction: str) -> dict:
    return {
        "episode_id": episode_id,
        "trajectory_id": episode_id,
        "scene_id": scene,
        "start_position": [0.0, 0.0, 0.0],
        "start_rotation": [0.0, 0.0, 0.0, 1.0],
        "goals": [{"position": [1.0, 0.0, 1.0]}],
        "instruction": {"instruction_text": instruction, "instruction_tokens": [1, 2]},
        "reference_path": [[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]],
    }


def dataset(path: Path, item: dict, words=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as target:
        json.dump({
            "instruction_vocab": {
                "word_list": words or ["<pad>", "go", "left"], "PAD_INDEX": 0,
            },
            "episodes": [item],
        }, target)


class VlnAssetPreflightTests(unittest.TestCase):
    def test_distinct_scenes_report_hashes_without_instruction_text(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scenes = root / "scenes"
            for name in ("a", "b"):
                scene = scenes / "mp3d" / name / f"{name}.glb"
                scene.parent.mkdir(parents=True)
                scene.write_bytes(name.encode("ascii") * 8)
                scene.with_suffix(".navmesh").write_bytes(b"nav")
            train = root / "train.json.gz"
            val = root / "val_unseen.json.gz"
            secret_phrase = "Turn toward the study"
            dataset(train, episode("mp3d/a/a.glb", 1, secret_phrase))
            dataset(val, episode("mp3d/b/b.glb", 2, "Go to the hallway"))

            result = preflight(train, val, scenes, 0, 0)
            self.assertEqual(result["train_val_unseen_scene_overlap"], 0)
            self.assertEqual(result["train"]["episode_count"], 1)
            self.assertEqual(result["train"]["source_sha256"], hashlib.sha256(train.read_bytes()).hexdigest())
            self.assertEqual(result["train"]["selected"]["episode_id"], "1")
            self.assertEqual(result["instruction_vocab_size"], 3)
            self.assertEqual(result["train"]["token_range"], {"min": 1, "max": 2, "count": 2})
            self.assertNotIn(secret_phrase, json.dumps(result))

    def test_rejects_mismatched_instruction_vocab(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.json.gz"
            val = root / "val_unseen.json.gz"
            dataset(train, episode("mp3d/a/a.glb", 1, "Go left"))
            dataset(val, episode("mp3d/b/b.glb", 2, "Go right"), ["<pad>", "go", "right"])
            with self.assertRaisesRegex(ValueError, "vocabularies differ"):
                preflight(train, val, root, 0, 0)

    def test_rejects_seen_validation_scene(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.json.gz"
            val = root / "val_unseen.json.gz"
            dataset(train, episode("mp3d/a/a.glb", 1, "Go left"))
            dataset(val, episode("mp3d/a/a.glb", 2, "Go right"))
            with self.assertRaisesRegex(ValueError, "share 1 scenes"):
                preflight(train, val, root, 0, 0)

    def test_rejects_scene_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.json.gz"
            val = root / "val_unseen.json.gz"
            dataset(train, episode("../outside.glb", 1, "Go left"))
            dataset(val, episode("mp3d/b/b.glb", 2, "Go right"))
            with self.assertRaisesRegex(ValueError, "Unsafe scene_id"):
                preflight(train, val, root, 0, 0)


if __name__ == "__main__":
    unittest.main()
