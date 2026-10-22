import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "integrations" / "rlinf_v03" / "run_streamvln_g8_release_reference.py"


class TestStreamVLNG8ReleaseReference(unittest.TestCase):
    def test_runner_keeps_reference_frozen_and_val_seen_only(self):
        source = RUNNER.read_text()
        tree = ast.parse(source)
        self.assertIn('cfg = make_policy_cfg(run_cfg, trainable=False, adapter_checkpoint=None)', source)
        self.assertIn('make_env(run_cfg, split="val_seen"', source)
        self.assertNotIn('split="val_unseen"', source)
        self.assertNotIn("optimizer", {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)})

    def test_runner_uses_every_preregistered_seed(self):
        source = RUNNER.read_text()
        self.assertIn("EXPECTED_SEEDS = (2026092808, 2026092809, 2026092810)", source)
        self.assertIn('for seed in EXPECTED_SEEDS', source)
        self.assertIn('"model_load_count": 1', source)


if __name__ == "__main__":
    unittest.main()
