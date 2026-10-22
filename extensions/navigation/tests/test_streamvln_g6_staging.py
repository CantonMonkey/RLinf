import ast
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).parents[1]
STAGE = ROOT / "integrations" / "rlinf_v03"


class StreamVLNG6StagingTest(unittest.TestCase):
    def test_manifest_reuses_fixed_g5_train_and_val_sets(self):
        g5 = json.loads((STAGE / "streamvln_g5_experiment.json").read_text())
        g6 = json.loads((STAGE / "streamvln_g6_experiment.json").read_text())
        self.assertEqual(g6["train"], g5["train"])
        self.assertEqual(g6["val_unseen"], g5["val_unseen"])
        self.assertEqual(g6["source"], g5["source"])

    def test_runner_preserves_train_val_boundary_and_batch_update(self):
        text = (STAGE / "run_streamvln_recovery_g6.py").read_text(encoding="utf-8")
        ast.parse(text)
        self.assertIn('split="train"', text)
        self.assertIn("env.expert_waypoint", text)
        self.assertIn('split="val_unseen"', text)
        self.assertIn("all_episodes_collected_before_update", text)
        self.assertIn("bc_reference_kl", text)
        self.assertIn('run_subprocess(config_path, "oracle")', text)
        self.assertIn('run_subprocess(config_path, "train")', text)
        self.assertIn('"secondary_post_training_used": False', text)
        self.assertIn("configure_process_git_safe_directory", text)
        self.assertNotIn("del replay, rows, indices, device", text)

    def test_policy_inputs_and_recovery_have_no_goal_privilege(self):
        policy = (STAGE / "streamvln_recovery_policy.py").read_text(encoding="utf-8")
        recovery = (STAGE / "streamvln_recovery.py").read_text(encoding="utf-8")
        ast.parse(policy)
        ast.parse(recovery)
        for forbidden in ("distance_to_goal", "geodesic_distance", "reference_path"):
            self.assertNotIn(forbidden, policy)
            self.assertNotIn(forbidden, recovery)
        self.assertIn("consecutive_blocked_forwards", recovery)
        self.assertIn("backtrack_waypoint", recovery)


if __name__ == "__main__":
    unittest.main()
