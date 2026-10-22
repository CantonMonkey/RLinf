import ast
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).parents[1]
STAGE = ROOT / "integrations" / "rlinf_v03"


class StreamVLNG5RunnerStagingTest(unittest.TestCase):
    def test_manifest_reuses_fixed_g4_train_and_val_sets(self):
        g4 = json.loads((STAGE / "streamvln_g4_experiment.json").read_text())
        g5 = json.loads((STAGE / "streamvln_g5_experiment.json").read_text())
        self.assertEqual(g5["train"], g4["train"])
        self.assertEqual(g5["val_unseen"], g4["val_unseen"])
        self.assertEqual(g5["source"], g4["source"])
        self.assertEqual(g5["g4_frozen_anchor"]["sha256"], "06af75cdf0870d1f9af1ecd16a98ec3bbc7aefe22bbc9b899159c47b4b28f31f")

    def test_runner_has_train_only_supervision_and_matched_g5_controls(self):
        text = (STAGE / "run_streamvln_topology_g5.py").read_text(encoding="utf-8")
        ast.parse(text)
        self.assertIn('split="train"', text)
        self.assertIn("env.expert_waypoint", text)
        self.assertIn('split="val_unseen"', text)
        self.assertIn('for variant in ("rl_only", "bc_only", "bc_rl")', text)
        self.assertIn("initial_head_sha256", text)
        self.assertIn("G4 frozen is an external action-space anchor", text)
        self.assertIn('"secondary_post_training_used": False', text)
        self.assertIn("max_preupdate_ratio_deviation", text)
        self.assertIn("checkpoint_load_audit", text)
        self.assertIn('run_subprocess(config_path, "bc_rl")', text)


if __name__ == "__main__":
    unittest.main()
