import ast
import hashlib
import json
from pathlib import Path
import unittest


class StreamVLNG722ProtocolTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]
    INTEGRATION = ROOT / "integrations" / "rlinf_v03"

    def test_manifest_freezes_disjoint_behavior_arms_and_viability_rule(self):
        path = self.INTEGRATION / "streamvln_g722_experiment.json"
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(),
            "3fe63b7afc3f40ed8086c6da6c38496df4a3780360387f324491f8f04751efdd",
        )
        manifest = json.loads(path.read_text())
        protocol = manifest["protocol"]
        self.assertEqual(manifest["selection"]["seeds"], [2026092811, 2026092812, 2026092813])
        scenes = [
            row["scene"] for arm in protocol["behavior_arms"].values() for row in arm
        ]
        self.assertEqual(len(scenes), 12)
        self.assertEqual(len(set(scenes)), 12)
        self.assertFalse(set(scenes) & set(manifest["selection"]["excluded_scenes"]))
        self.assertTrue(protocol["state_zero_and_final_must_exactly_match_g721"])
        self.assertTrue(protocol["behavior_is_nonselective"])
        self.assertFalse(protocol["checkpoint_selection"])
        rule = manifest["decision_rule"]
        self.assertEqual(rule["pooled_environment_reward_delta_min"], 0.0)
        self.assertEqual(rule["minimum_nonnegative_seed_reward_deltas"], 2)
        self.assertEqual(rule["pooled_collision_delta_max"], 0)
        self.assertEqual(rule["pooled_blocked_forward_delta_max"], 0)
        self.assertEqual(rule["pooled_realized_path_retention_min"], 0.8)
        self.assertFalse(rule["posthoc_selection_or_rerun"])
        self.assertEqual(protocol["val_unseen_episode_count"], 0)
        self.assertFalse(manifest["g8_ready"])

    def test_runner_reconstructs_then_restores_matched_behavior_states(self):
        path = self.INTEGRATION / "run_streamvln_matched_behavior_g722.py"
        source = path.read_text()
        ast.parse(source)
        self.assertIn("baseline_hashes != anchor", source)
        self.assertIn("candidate {candidate_index} differs from G7.21", source)
        self.assertIn("final_hashes != anchor", source)
        self.assertIn('seed_phase="g722_behavior"', source)
        self.assertIn("initial_behavior_restore = restore_training_state", source)
        self.assertIn("final_behavior_restore = restore_training_state", source)
        self.assertIn("paired_behavior(behavior_initial, behavior_final)", source)
        self.assertNotIn("sample_target_mode", source)
        self.assertNotIn("save_head_checkpoint", source)
        self.assertNotIn('split="val_unseen"', source)

    def test_recompute_uses_runtime_mode_index_contract(self):
        path = self.INTEGRATION / "recompute_streamvln_g722.py"
        source = path.read_text()
        ast.parse(source)
        self.assertIn('"stop_decision_count": modes[1]', source)
        self.assertIn('"navigate_decision_count": modes[0]', source)
        self.assertIn('"recover_decision_count": modes[2]', source)


if __name__ == "__main__":
    unittest.main()
