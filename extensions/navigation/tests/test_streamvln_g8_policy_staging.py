import ast
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).parents[1]
STAGE = ROOT / "integrations" / "rlinf_v03"


class StreamVLNG8PolicyStagingTest(unittest.TestCase):
    def test_policy_uses_one_categorical_executable_action_distribution(self):
        path = STAGE / "streamvln_g8_policy.py"
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        self.assertIn("candidate_head", text)
        self.assertIn('"action_logits"', text)
        self.assertIn('operation == "sample_g8"', text)
        self.assertIn('operation == "replay_g8"', text)
        self.assertIn('operation == "bc_loss_g8"', text)
        self.assertIn("waypoint_log_std.requires_grad_(False)", text)
        self.assertNotIn("Normal(", text)
        self.assertNotIn("waypoint_log_probs", text)
        self.assertGreater(len(tree.body), 8)

    def test_fused_feature_is_exposed_without_unfreezing_backbone(self):
        base = (STAGE / "streamvln_topological_policy.py").read_text(encoding="utf-8")
        policy = (STAGE / "streamvln_g8_policy.py").read_text(encoding="utf-8")
        self.assertIn('"fused": fused.float()', base)
        self.assertIn('"candidate_head."', base)
        self.assertIn("frozen_backbone", json.dumps(
            json.loads((STAGE / "streamvln_g8_experiment.json").read_text()),
            sort_keys=True,
        ))
        for forbidden in ("distance_to_goal", "shortest_path", "navmesh"):
            self.assertNotIn(forbidden, policy)

    def test_plan_and_manifest_freeze_one_final_unseen_evaluation(self):
        manifest = json.loads(
            (STAGE / "streamvln_g8_experiment.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["final_claim_gate"]["val_unseen_evaluations"], 1)
        self.assertEqual(len(manifest["algorithm_gate"]["seeds"]), 3)
        self.assertTrue(manifest["governance"]["no_g7_successor"])
        self.assertFalse(manifest["governance"]["push"])

    def test_algorithm_runner_is_matched_and_development_only(self):
        text = (STAGE / "run_streamvln_g8_algorithm.py").read_text(encoding="utf-8")
        env_builder = (STAGE / "run_streamvln_ppo_g4.py").read_text(encoding="utf-8")
        tree = ast.parse(text)
        manifest = json.loads(
            (STAGE / "streamvln_g8_algorithm_manifest.json").read_text(encoding="utf-8")
        )
        self.assertIn('operation="sample_g8"', text)
        self.assertIn('operation="replay_g8"', text)
        self.assertIn('split="val_seen"', text)
        self.assertIn("collect_bc_dataset", text)
        self.assertIn("all_episodes_collected_before_update", text)
        self.assertIn('"--habitat-startup-timeout-s"', text)
        self.assertIn('"habitat_startup_timeout_s": args.habitat_startup_timeout_s', text)
        self.assertIn(
            '"startup_timeout_s": run_cfg.get("habitat_startup_timeout_s", 60.0)',
            env_builder,
        )
        self.assertFalse(manifest["source"]["val_unseen_read"])
        self.assertEqual(len(manifest["seeds"]), 3)
        self.assertGreater(len(tree.body), 20)


if __name__ == "__main__":
    unittest.main()
