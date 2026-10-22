import json
from pathlib import Path
import unittest


ROOT = Path(__file__).parents[1]
STAGE = ROOT / "integrations" / "rlinf_v03"


class StreamVLNG4StagingTest(unittest.TestCase):
    def test_manifest_is_scene_disjoint_and_fixed(self):
        manifest = json.loads((STAGE / "streamvln_g4_experiment.json").read_text())
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(len(manifest["train"]), 8)
        self.assertEqual(len(manifest["val_unseen"]), 8)
        train_scenes = {row["scene"] for row in manifest["train"]}
        val_scenes = {row["scene"] for row in manifest["val_unseen"]}
        self.assertEqual(len(train_scenes), 8)
        self.assertEqual(len(val_scenes), 8)
        self.assertTrue(train_scenes.isdisjoint(val_scenes))
        self.assertEqual(len(manifest["source"]["train_sha256"]), 64)
        self.assertEqual(len(manifest["source"]["val_unseen_sha256"]), 64)

    def test_runner_enforces_matched_unseen_comparison(self):
        text = (STAGE / "run_streamvln_ppo_g4.py").read_text(encoding="utf-8")
        self.assertIn('split="val_unseen"', text)
        self.assertIn('for variant in ("frozen", "rl")', text)
        self.assertIn("frozen_ids != rl_ids", text)
        self.assertIn('"secondary_post_training_used": False', text)
        self.assertIn("max_ratio_deviation", text)
        self.assertIn("adapter_load_audit", text)

    def test_policy_supports_continuation_and_greedy_eval(self):
        text = (STAGE / "streamvln_ppo_policy.py").read_text(encoding="utf-8")
        self.assertIn("continued sampling has no previous proposal", text)
        self.assertIn("session.acknowledge", text)
        self.assertIn("state_values", text)
        self.assertIn("deterministic_eval", text)
        self.assertIn("int(categorical.argmax())", text)
        self.assertIn("bootstrap step disagrees with action history", text)
        self.assertNotIn("expected physical step four", text)


if __name__ == "__main__":
    unittest.main()
