import ast
import hashlib
import json
from pathlib import Path
import unittest


class StreamVLNG719StabilityTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]
    INTEGRATION = ROOT / "integrations" / "rlinf_v03"

    def test_manifest_freezes_zero_one_two_causal_audit(self):
        path = self.INTEGRATION / "streamvln_g719_experiment.json"
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(),
            "1453d62dc5f4ceef4a83e1ecdc6d17d3efad552dce4619c2cabf5613c521243f",
        )
        manifest = json.loads(path.read_text())
        protocol = manifest["protocol"]
        self.assertEqual(protocol["update_states"], [0, 1, 2])
        self.assertEqual(protocol["ppo_full_batch_updates"], 2)
        self.assertEqual(protocol["collection_contract"]["expected_rows"], 12)
        self.assertEqual(
            protocol["collection_contract"]["expected_records_sha256"],
            "de94c1876da52efad1773363d3a6062449ab8db45e1295544645957187a8a8ca",
        )
        self.assertEqual(protocol["postupdate_ratio_max_deviation"], 0.5)
        self.assertEqual(protocol["postupdate_distribution_kl_max"], 0.1)
        self.assertTrue(protocol["behavior_metrics_are_descriptive_only"])
        self.assertFalse(protocol["behavior_based_arm_selection"])
        self.assertFalse(protocol["early_stop"])
        self.assertFalse(protocol["checkpoint_selection"])
        self.assertEqual(protocol["val_unseen_episode_count"], 0)
        self.assertFalse(manifest["g8_ready"])

    def test_runner_uses_one_fixed_optimizer_trajectory(self):
        path = self.INTEGRATION / "run_streamvln_stability_g719.py"
        source = path.read_text()
        ast.parse(source)
        self.assertIn("behavior_zero = collect_phase", source)
        self.assertIn("update_one = one_update", source)
        self.assertIn("behavior_one = collect_phase", source)
        self.assertIn("update_two = one_update", source)
        self.assertIn("behavior_two = collect_phase", source)
        self.assertIn("preserve_random_state", source)
        self.assertIn("state_two_reproduces_g718", source)
        self.assertNotIn("sample_target_mode", source)
        self.assertNotIn("save_head_checkpoint", source)
        self.assertEqual(source.count("manager.optimizer_step()"), 1)

    def test_plan_prohibits_behavior_selection_and_g8(self):
        plan_path = self.ROOT.parent / "STREAMVLN_G719_EXPERIMENT_PLAN.md"
        if not plan_path.is_file():
            self.skipTest("Historical experiment plan is not included in the public source export")
        plan = plan_path.read_text()
        self.assertIn("Preregistered on 2026-09-27 before the first G7.19", plan)
        self.assertIn("Behavior reports are nonselective", plan)
        self.assertIn("cannot pass a stability arm", plan)
        self.assertIn("does not authorize early stopping", plan)
        self.assertIn("G8", plan)


if __name__ == "__main__":
    unittest.main()
