import ast
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
INTEGRATION = ROOT / "integrations" / "rlinf_v03"


class StreamVLNG7StagingTest(unittest.TestCase):
    def test_runner_is_train_only_and_has_predeclared_gates(self):
        path = INTEGRATION / "run_streamvln_candidates_g7.py"
        text = path.read_text()
        ast.parse(text)
        self.assertIn('split="train"', text)
        self.assertNotIn('split="val_unseen"', text)
        self.assertIn('"val_unseen_episode_count": 0', text)
        self.assertIn("candidate_preserves_plain_sr", text)
        self.assertIn("candidate_recovery_preserves_progress", text)
        self.assertIn("candidate_recovery_not_immobile", text)
        self.assertIn("configure_process_git_safe_directory", text)
        self.assertIn("import os", text)

    def test_candidate_generator_does_not_accept_goal_or_expert(self):
        module_path = INTEGRATION / "streamvln_candidates.py"
        tree = ast.parse(module_path.read_text())
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "generate_waypoint_candidates"
        )
        arguments = [
            argument.arg
            for argument in function.args.args + function.args.kwonlyargs
        ]
        self.assertNotIn("goal", arguments)
        self.assertNotIn("expert", arguments)

    def test_manifest_reuses_fixed_g6_split(self):
        g6 = json.loads((INTEGRATION / "streamvln_g6_experiment.json").read_text())
        g7 = json.loads((INTEGRATION / "streamvln_g7_experiment.json").read_text())
        self.assertEqual(g7["source"], g6["source"])
        self.assertEqual(g7["train"], g6["train"])
        self.assertEqual(g7["val_unseen"], g6["val_unseen"])
        self.assertIn("train-only", g7["stage"])


if __name__ == "__main__":
    unittest.main()
