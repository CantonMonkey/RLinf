import json
from pathlib import Path
import unittest

from integrations.rlinf_v03.streamvln_candidates_g74 import (
    first_forward_heading_degrees,
    generate_macro_candidates,
)
from integrations.rlinf_v03.streamvln_candidates_g75 import AngularSectorShield
from integrations.rlinf_v03.streamvln_topology import OnlineTopology


ROOT = Path(__file__).resolve().parents[1]
INTEGRATION = ROOT / "integrations" / "rlinf_v03"


class StreamVLNG75SectorTest(unittest.TestCase):
    def test_sector15_blocks_center_and_adjacent_bins(self):
        topology = OnlineTopology(max_decisions=40)
        candidates = generate_macro_candidates(topology)
        shield = AngularSectorShield(
            topology.turn_radians, sector_radius_bins=1, cooldown_decisions=2
        )
        shield.block_world_heading(0.0)
        shield.begin_decision()
        available = shield.available(candidates, current_world_heading=0.0)
        headings = {first_forward_heading_degrees(row.actions) for row in available}
        self.assertTrue({-15.0, 0.0, 15.0}.isdisjoint(headings))
        self.assertIn(30.0, headings)

    def test_sector30_blocks_five_bins_for_two_decisions(self):
        topology = OnlineTopology(max_decisions=40)
        candidates = generate_macro_candidates(topology)
        shield = AngularSectorShield(
            topology.turn_radians, sector_radius_bins=2, cooldown_decisions=2
        )
        shield.block_world_heading(0.0)
        for _ in range(2):
            shield.begin_decision()
            available = shield.available(candidates, current_world_heading=0.0)
            headings = {
                first_forward_heading_degrees(row.actions) for row in available
            }
            self.assertTrue({-30.0, -15.0, 0.0, 15.0, 30.0}.isdisjoint(headings))
        shield.begin_decision()
        available = shield.available(candidates, current_world_heading=0.0)
        headings = {first_forward_heading_degrees(row.actions) for row in available}
        self.assertIn(0.0, headings)
        self.assertEqual(shield.summary()["sector_radius_degrees"], 30.0)

    def test_runner_and_manifest_preserve_protocol(self):
        runner = (INTEGRATION / "run_streamvln_candidates_g75.py").read_text()
        self.assertIn('VARIANTS = ("sector15", "sector30")', runner)
        self.assertIn("first all-gates-pass in sector15 then sector30 order", runner)
        self.assertIn('"val_unseen_episode_count": 0', runner)
        self.assertIn('"model_loaded": False', runner)
        self.assertNotIn("generate_executable_candidates", runner)

        g74 = json.loads((INTEGRATION / "streamvln_g74_experiment.json").read_text())
        g75 = json.loads((INTEGRATION / "streamvln_g75_experiment.json").read_text())
        self.assertEqual(g75["train"], g74["train"])
        self.assertEqual(g75["val_unseen"], g74["val_unseen"])
        self.assertEqual(g75["source"], g74["source"])
        self.assertEqual(
            g75["stage"], "G7.5 train-only angular collision-sector gate"
        )


if __name__ == "__main__":
    unittest.main()
