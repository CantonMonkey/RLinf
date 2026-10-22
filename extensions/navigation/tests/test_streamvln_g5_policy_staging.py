import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).parents[1]
STAGE = ROOT / "integrations" / "rlinf_v03"


class StreamVLNG5PolicyStagingTest(unittest.TestCase):
    def test_policy_freezes_backbone_and_exposes_high_level_actions(self):
        text = (STAGE / "streamvln_topological_policy.py").read_text(encoding="utf-8")
        tree = ast.parse(text)
        self.assertIn("super().__init__", text)
        self.assertIn("trainable=False", text)
        self.assertIn("waypoint_head", text)
        self.assertIn("mode_head", text)
        self.assertIn("RECOVER", (STAGE / "streamvln_topology.py").read_text())
        self.assertIn('operation == "bc_loss"', text)
        self.assertIn('operation == "replay_head"', text)
        self.assertNotIn("distance_to_goal", text)
        self.assertNotIn("reference_path", text)
        self.assertGreater(len(tree.body), 5)

    def test_topology_features_have_no_privileged_goal_fields(self):
        text = (STAGE / "streamvln_topology.py").read_text(encoding="utf-8")
        for forbidden in ("geodesic_distance", "distance_to_goal", "reference_path"):
            self.assertNotIn(forbidden, text)
        self.assertIn("chunk_collisions", (STAGE / "streamvln_env.py").read_text())


if __name__ == "__main__":
    unittest.main()
