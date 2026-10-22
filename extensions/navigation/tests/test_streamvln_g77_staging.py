import json
from pathlib import Path
import unittest

from scripts.habitat_r2r_aligned_passability_service import AlignedPassabilitySession


ROOT = Path(__file__).resolve().parents[1]
INTEGRATION = ROOT / "integrations" / "rlinf_v03"


class FakeAlignedBackend:
    def reset(self):
        return {
            "policy_obs": {
                "rgb": {
                    "encoding": "base64/raw-u8-hwc",
                    "shape_hwc": [1, 1, 3],
                    "data_b64": "AAAA",
                },
                "instruction_ids": [1, 0],
                "instruction_mask": [True, False],
            },
            "episode_id": "1",
            "scene_id": "/scene.glb",
            "instruction_sha256": "hash",
            "geodesic_distance": 4.0,
            "success": False,
            "spl": 0.0,
            "path_length": 0.0,
            "collision": False,
            "collision_count": 0,
            "episode_over": False,
        }

    def probe_aligned_candidates(self, candidates):
        return {
            "schema_version": 1,
            "source": "train-simulator-aligned-branch-rollout",
            "policy_observation_unchanged": True,
            "state_restored": True,
            "max_position_restore_error_m": 0.0,
            "max_rotation_restore_error": 0.0,
            "candidates": [],
        }

    def close(self):
        pass


class G77PassabilityTest(unittest.TestCase):
    def test_aligned_probe_is_outside_policy_observation(self):
        session = AlignedPassabilitySession(FakeAlignedBackend(), max_steps=4)
        with self.assertRaisesRegex(RuntimeError, "reset must precede"):
            session.probe_aligned_candidates([])
        reset = session.reset()
        self.assertEqual(
            set(reset["obs"]), {"rgb", "instruction_ids", "instruction_mask"}
        )
        response = session.probe_aligned_candidates([])
        self.assertTrue(response["state_restored"])
        self.assertNotIn("obs", response)

    def test_navigation_is_causal_and_has_no_branch_probe(self):
        runner = (INTEGRATION / "run_streamvln_passability_g77.py").read_text()
        collection = runner.split("def collect_episode", 1)[1].split(
            "def save_dataset", 1
        )[0]
        navigation = runner.split("def navigation_episode", 1)[1].split(
            "def parse_args", 1
        )[0]
        self.assertIn("env.probe_aligned_candidates(candidates)", collection)
        self.assertNotIn("probe_aligned_candidates", navigation)
        self.assertLess(
            navigation.index("turn_step = execute_sequence"),
            navigation.index("predict_scores(model, score_rows"),
        )
        self.assertIn('"heading_degrees": 0.0', navigation)
        self.assertIn('"navigation_protocol": "turn-observe-translate"', runner)
        self.assertIn('"branch_rollout_used_during_navigation_gate": False', runner)
        self.assertIn('"val_unseen_episode_count": 0', runner)

    def test_manifest_preserves_fixed_scene_disjoint_split(self):
        manifest = json.loads(
            (INTEGRATION / "streamvln_g77_experiment.json").read_text()
        )
        split = manifest["g77_passability_split"]
        self.assertEqual(len(split["train_scenes"]), 6)
        self.assertEqual(len(split["train_dev_scenes"]), 2)
        self.assertFalse(
            set(split["train_scenes"]) & set(split["train_dev_scenes"])
        )
        self.assertEqual(
            split["train_scenes"],
            [row["scene"] for row in manifest["train"][:6]],
        )
        self.assertEqual(
            split["train_dev_scenes"],
            [row["scene"] for row in manifest["train"][6:]],
        )


if __name__ == "__main__":
    unittest.main()