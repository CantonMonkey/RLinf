import json
from pathlib import Path
import unittest

from scripts.habitat_r2r_passability_service import (
    PassabilitySession,
    validate_probe_candidates,
)


ROOT = Path(__file__).resolve().parents[1]
INTEGRATION = ROOT / "integrations" / "rlinf_v03"


class FakeProbeBackend:
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

    def probe_candidates(self, candidates):
        rows = validate_probe_candidates(candidates)
        return {
            "schema_version": 1,
            "source": "train-simulator-branch-rollout",
            "policy_observation_unchanged": True,
            "state_restored": True,
            "max_position_restore_error_m": 0.0,
            "max_rotation_restore_error": 0.0,
            "candidates": [
                {
                    "name": row["name"],
                    "actions": row["actions"],
                    "collisions": [False] * len(row["actions"]),
                    "unsafe": False,
                    "forward_attempts": row["actions"].count(1),
                    "forward_collisions": 0,
                    "end_position": [0.0, 0.0, 0.0],
                }
                for row in rows
            ],
        }

    def close(self):
        pass


class G76PassabilityTest(unittest.TestCase):
    def test_probe_session_keeps_labels_outside_policy_observation(self):
        session = PassabilitySession(FakeProbeBackend(), max_steps=4)
        candidates = [{"name": "macro_p0_d0p5", "actions": [1, 1]}]
        with self.assertRaisesRegex(RuntimeError, "reset must precede"):
            session.probe_candidates(candidates)
        reset = session.reset()
        self.assertEqual(
            set(reset["obs"]), {"rgb", "instruction_ids", "instruction_mask"}
        )
        response = session.probe_candidates(candidates)
        self.assertTrue(response["state_restored"])
        self.assertNotIn("obs", response)
        with self.assertRaises(ValueError):
            validate_probe_candidates([{"name": "bad", "actions": [0, 1]}])

    def test_model_module_declares_rgb_geometry_control_and_fixed_grid(self):
        source = (INTEGRATION / "streamvln_passability_g76.py").read_text()
        self.assertIn("class GeometryPassabilityNet", source)
        self.assertIn("class RGBPassabilityNet", source)
        self.assertIn("require_two_classes=False", source)
        self.assertIn("THRESHOLDS = tuple", source)
        self.assertIn("len(train) != 6", source)
        self.assertIn("len(dev) != 2", source)

    def test_runner_and_manifest_preserve_leakage_boundary(self):
        runner = (INTEGRATION / "run_streamvln_passability_g76.py").read_text()
        self.assertIn('"val_unseen_episode_count": 0', runner)
        self.assertIn('"branch_rollout_used_during_navigation_gate": False', runner)
        self.assertIn('"depth_used": False', runner)
        self.assertIn("first-six/final-two manifest order", runner)
        manifest = json.loads(
            (INTEGRATION / "streamvln_g76_experiment.json").read_text()
        )
        split = manifest["g76_passability_split"]
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
