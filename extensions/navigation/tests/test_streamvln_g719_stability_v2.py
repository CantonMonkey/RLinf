import ast
import hashlib
import json
from pathlib import Path
import unittest


class StreamVLNG719StabilityV2Tests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]
    INTEGRATION = ROOT / "integrations" / "rlinf_v03"

    def test_manifest_freezes_optimizer_contract_only_repair(self):
        path = self.INTEGRATION / "streamvln_g719_experiment_v2.json"
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(),
            "bd86c0c6ebd3597f1d4a2bae7b0a546f3e5dc4ff45676e4cc002f44c8448d423",
        )
        manifest = json.loads(path.read_text(encoding="utf-8"))
        protocol = manifest["protocol"]
        self.assertEqual(manifest["repair"]["kind"], "optimizer-contract-only")
        self.assertEqual(
            manifest["repair"]["failed_v1_guard"], "optimizer_initial_state_fixed"
        )
        self.assertEqual(
            protocol["optimizer_initial_contract"],
            {
                "optimizer_class": "AdamW",
                "state_entry_count": 19,
                "parameter_group_count": 2,
                "learning_rates": [1e-5, 1e-5],
                "state_keys": ["exp_avg", "exp_avg_sq", "step"],
                "moment_parameter_count": 1_003_592,
                "max_abs_exp_avg": 0.0,
                "max_abs_exp_avg_sq": 0.0,
                "step_values": [1.0],
            },
        )
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
        path = self.INTEGRATION / "run_streamvln_stability_g719_v2.py"
        source = path.read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn("behavior_zero = collect_phase", source)
        self.assertIn("update_one = one_update", source)
        self.assertIn("behavior_one = collect_phase", source)
        self.assertIn("update_two = one_update", source)
        self.assertIn("behavior_two = collect_phase", source)
        self.assertIn("preserve_random_state", source)
        self.assertIn("optimizer_state_contract", source)
        self.assertIn("v1_protocol_failure_anchored", source)
        self.assertIn("exact_initial_optimizer_eager_zero_contract", source)
        self.assertIn("state_two_reproduces_g718", source)
        self.assertNotIn("sample_target_mode", source)
        self.assertNotIn("save_head_checkpoint", source)
        self.assertEqual(source.count("manager.optimizer_step()"), 1)

    def test_plan_prohibits_behavior_selection_and_g8(self):
        plan_path = self.ROOT.parent / "STREAMVLN_G719_REPAIR_PLAN.md"
        if not plan_path.is_file():
            self.skipTest("Historical experiment plan is not included in the public source export")
        plan = plan_path.read_text(encoding="utf-8")
        self.assertIn("Preregistered on 2026-09-28 before the G7.19 v2", plan)
        self.assertIn("only the optimizer-state guard", plan)
        self.assertIn("Behavior reports remain nonselective", plan)
        self.assertIn("does not authorize early stopping", plan)
        self.assertIn("G8", plan)


if __name__ == "__main__":
    unittest.main()
