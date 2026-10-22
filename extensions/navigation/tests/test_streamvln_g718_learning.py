import ast
import hashlib
import json
from pathlib import Path
import unittest


class StreamVLNG718LearningTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]
    INTEGRATION = ROOT / "integrations" / "rlinf_v03"

    def test_manifest_freezes_unbiased_train_only_matched_control(self):
        path = self.INTEGRATION / "streamvln_g718_experiment.json"
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(),
            "a985243b5aadf19ee8ab37ac534aa1292d095588f8a49427c984e81656864aab",
        )
        manifest = json.loads(path.read_text())
        protocol = manifest["protocol"]
        collection = {row["scene"] for row in protocol["collection_scene_specs"]}
        behavior = {row["scene"] for row in protocol["behavior_scene_specs"]}
        self.assertEqual(protocol["split"], "train")
        self.assertEqual(len(collection), 4)
        self.assertEqual(len(behavior), 4)
        self.assertTrue(collection.isdisjoint(behavior))
        self.assertEqual(protocol["sampling"]["sample_calls_per_decision"], 1)
        self.assertFalse(protocol["sampling"]["target_mode_scan"])
        self.assertFalse(protocol["sampling"]["mode_filter_or_resample"])
        self.assertFalse(protocol["sampling"]["recovery_override"])
        self.assertEqual(protocol["ppo_full_batch_updates"], 2)
        self.assertFalse(protocol["behavior_improvement_required"])
        self.assertFalse(protocol["checkpoint_selection"])
        self.assertEqual(protocol["val_unseen_episode_count"], 0)
        self.assertFalse(manifest["navigation_evaluation"])
        self.assertFalse(manifest["g8_ready"])

    def test_runner_uses_single_samples_and_fixed_two_update_loop(self):
        path = self.INTEGRATION / "run_streamvln_learning_g718.py"
        source = path.read_text()
        ast.parse(source)
        self.assertIn('operation="sample_head"', source)
        self.assertNotIn("sample_target_mode", source)
        self.assertNotIn("resolved_choice", source)
        self.assertIn('"sample_call_count": 1', source)
        self.assertIn('seed_phase="behavior"', source)
        self.assertIn("no_update_control_unchanged", source)
        self.assertIn("structured_distribution_kl", source)
        self.assertIn('operation="encode_bootstrap_state"', source)
        self.assertEqual(source.count("manager.optimizer_step()"), 1)
        self.assertIn('"val_unseen_episode_count": 0', source)
        self.assertNotIn('split="val_unseen"', source)

    def test_required_behavior_reports_are_fixed(self):
        manifest = json.loads(
            (self.INTEGRATION / "streamvln_g718_experiment.json").read_text()
        )
        self.assertEqual(
            set(manifest["required_reports"]),
            {
                "reward_advantage_return_distribution",
                "realized_path_m",
                "collisions_and_blocked_forwards",
                "stop_termination_and_timeout",
                "recover_frequency",
                "primitive_action_distribution",
                "high_level_mode_distribution",
                "paired_before_after_behavior",
            },
        )

    def test_plan_keeps_behavior_result_nonselective(self):
        plan_path = self.ROOT.parent / "STREAMVLN_G718_EXPERIMENT_PLAN.md"
        if not plan_path.is_file():
            self.skipTest("Historical experiment plan is not included in the public source export")
        plan = plan_path.read_text()
        self.assertIn("Preregistered on 2026-09-27 before the first G7.18", plan)
        self.assertIn("No improvement is required to pass", plan)
        self.assertIn("There is no target-mode scan", plan)
        self.assertIn("No checkpoint may be selected", plan)
        self.assertIn("G8 remain closed", plan)


if __name__ == "__main__":
    unittest.main()
