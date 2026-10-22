"""Schema and policy/evaluator boundary checks for the R2R interface probe."""

import base64
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.habitat_r2r_rgb_instruction_smoke import (
    make_policy_obs,
    read_r2r_metadata,
    validate_instruction,
)


class R2RSmokeContractTest(unittest.TestCase):
    def test_official_episode_schema_and_mp3d_scene(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scenes = root / "scene_datasets"
            scene = scenes / "mp3d" / "one" / "one.glb"
            scene.parent.mkdir(parents=True)
            scene.write_bytes(b"glb")
            scene.with_suffix(".navmesh").write_bytes(b"navmesh")
            dataset_path = root / "train.json.gz"
            document = {
                "instruction_vocab": {"word_list": ["<pad>", "walk", "left"]},
                "episodes": [{
                    "episode_id": "42",
                    "scene_id": "data/scene_datasets/mp3d/one/one.glb",
                    "goals": [{"position": [0, 0, 0]}],
                    "reference_path": [[0, 0, 1], [0, 0, 0]],
                    "instruction": {
                        "instruction_text": "walk left",
                        "instruction_tokens": [1, 2],
                    },
                }],
            }
            with gzip.open(dataset_path, "wt", encoding="utf-8") as stream:
                json.dump(document, stream)
            episode, words, actual_scene, navmesh = read_r2r_metadata(
                dataset_path, 0, scenes
            )
            self.assertEqual(episode["episode_id"], "42")
            self.assertEqual(len(words), 3)
            self.assertEqual(actual_scene, scene)
            self.assertEqual(navmesh, scene.with_suffix(".navmesh"))

    def test_official_instruction_ids_are_preserved_and_padded(self):
        episode = SimpleNamespace(
            instruction=SimpleNamespace(
                instruction_text="walk left", instruction_tokens=[1, 2, 0, 0]
            )
        )
        ids, mask = validate_instruction(
            {"instruction": {"text": "walk left", "tokens": [1, 2, 0, 0]}},
            episode, vocab_size=3, max_tokens=4,
        )
        self.assertEqual(ids, [1, 2, 0, 0])
        self.assertEqual(mask, [True, True, False, False])
        with self.assertRaises(ValueError):
            validate_instruction(
                {"instruction": {"text": "other", "tokens": [1, 2, 0, 0]}},
                episode, vocab_size=3, max_tokens=4,
            )

    def test_policy_payload_contains_only_rgb_and_instruction(self):
        payload = make_policy_obs(
            bytes(range(12)), (2, 2, 3), [1, 2, 0], [True, True, False]
        )
        self.assertEqual(
            set(payload), {"rgb", "instruction_ids", "instruction_mask"}
        )
        self.assertEqual(payload["rgb"]["shape_hwc"], [2, 2, 3])
        self.assertEqual(base64.b64decode(payload["rgb"]["data_b64"]), bytes(range(12)))
        self.assertTrue(
            {"pointgoal", "goal_positions_xyz", "reference_path", "agent_position_xyz"}
            .isdisjoint(payload)
        )
        with self.assertRaises(ValueError):
            make_policy_obs(b"short", (2, 2, 3), [1], [True])


if __name__ == "__main__":
    unittest.main()
