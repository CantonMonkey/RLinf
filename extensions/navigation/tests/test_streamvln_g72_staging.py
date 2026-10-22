import ast
import json
from pathlib import Path
import unittest

from integrations.rlinf_v03.streamvln_candidates_g71 import (
    generate_dense_candidates,
)
from integrations.rlinf_v03.streamvln_topology import OnlineTopology


ROOT = Path(__file__).resolve().parents[1]
INTEGRATION = ROOT / "integrations" / "rlinf_v03"


class StreamVLNG72StagingTest(unittest.TestCase):
    def test_expanded_bank_has_predeclared_geometry(self):
        candidates = generate_dense_candidates(
            OnlineTopology(max_decisions=10),
            distances_m=(0.5, 0.75, 1.0),
            headings_degrees=(0.0, 15.0, -15.0, 30.0, -30.0, 45.0, -45.0, 60.0, -60.0),
        )
        self.assertEqual(len(candidates), 27)
        self.assertEqual({item.distance_m for item in candidates}, {0.5, 0.75, 1.0})
        self.assertEqual(
            {item.heading_degrees for item in candidates},
            {-60.0, -45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0, 60.0},
        )

    def test_runner_is_train_only_and_has_final_gates(self):
        text = (INTEGRATION / "run_streamvln_candidates_g72.py").read_text()
        ast.parse(text)
        self.assertIn('split="train"', text)
        self.assertNotIn('split="val_unseen"', text)
        self.assertIn('"val_unseen_episode_count": 0', text)
        for gate in (
            "coverage",
            "p90_error",
            "success_rate",
            "collision",
            "progress_noninferior",
            "successes_stop",
        ):
            self.assertIn(f'"{gate}"', text)
        self.assertIn('default=0.10', text)

    def test_manifest_reuses_g71_split(self):
        g71 = json.loads((INTEGRATION / "streamvln_g71_experiment.json").read_text())
        g72 = json.loads((INTEGRATION / "streamvln_g72_experiment.json").read_text())
        self.assertEqual(g72["source"], g71["source"])
        self.assertEqual(g72["train"], g71["train"])
        self.assertEqual(g72["val_unseen"], g71["val_unseen"])
        self.assertIn("G7.2 train-only", g72["stage"])


if __name__ == "__main__":
    unittest.main()
