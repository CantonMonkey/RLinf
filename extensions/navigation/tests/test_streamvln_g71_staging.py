import ast
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
INTEGRATION = ROOT / "integrations" / "rlinf_v03"


class StreamVLNG71StagingTest(unittest.TestCase):
    def test_runner_is_train_only_and_factorial(self):
        text = (INTEGRATION / "run_streamvln_candidates_g71.py").read_text()
        ast.parse(text)
        self.assertIn('split="train"', text)
        self.assertNotIn('split="val_unseen"', text)
        self.assertIn('"val_unseen_episode_count": 0', text)
        for name in (
            "dense_fixed4",
            "dense_adaptive",
            "dense_adaptive_blacklist",
        ):
            self.assertIn(name, text)
        for gate in (
            "adaptive_preserves_plain_sr",
            "adaptive_successes_stop",
            "blacklist_preserves_progress",
            "blacklist_collision_not_worse",
            "blacklist_exercised",
        ):
            self.assertIn(gate, text)

    def test_environment_keeps_fixed_tensor_and_optional_prefix(self):
        text = (INTEGRATION / "streamvln_env.py").read_text()
        tree = ast.parse(text)
        method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "chunk_step"
        )
        arguments = [argument.arg for argument in method.args.args]
        self.assertEqual(arguments, ["self", "chunk_actions", "execute_count"])
        self.assertIn("array.shape != (1, ACTION_CHUNK, 1)", text)
        self.assertIn("proposed = proposed[:execute_count]", text)

    def test_manifest_reuses_g7_split(self):
        g7 = json.loads((INTEGRATION / "streamvln_g7_experiment.json").read_text())
        g71 = json.loads((INTEGRATION / "streamvln_g71_experiment.json").read_text())
        self.assertEqual(g71["source"], g7["source"])
        self.assertEqual(g71["train"], g7["train"])
        self.assertEqual(g71["val_unseen"], g7["val_unseen"])
        self.assertIn("G7.1 train-only", g71["stage"])


if __name__ == "__main__":
    unittest.main()
