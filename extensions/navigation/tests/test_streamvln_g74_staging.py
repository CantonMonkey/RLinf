import json
import math
from pathlib import Path
import unittest

from integrations.rlinf_v03.streamvln_candidates_g74 import (
    WorldHeadingShield,
    first_forward_heading_degrees,
    generate_macro_candidates,
    select_oracle_candidate,
    simulate_action_endpoint,
    trace_collision_world_headings,
)
from integrations.rlinf_v03.streamvln_topology import (
    FORWARD_ACTION, LEFT_ACTION, OnlineTopology, STOP_ACTION
)


ROOT = Path(__file__).resolve().parents[1]
INTEGRATION = ROOT / "integrations" / "rlinf_v03"


class StreamVLNG74CandidateTest(unittest.TestCase):
    def test_macro_bank_has_unique_exact_endpoints(self):
        topology = OnlineTopology(max_decisions=40)
        candidates = generate_macro_candidates(topology)
        self.assertEqual(len(candidates), 33)
        endpoints = set()
        for candidate in candidates:
            self.assertLessEqual(len(candidate.actions), 8)
            self.assertNotIn(STOP_ACTION, candidate.actions)
            actual = simulate_action_endpoint(
                candidate.actions,
                forward_step_m=topology.forward_step_m,
                turn_radians=topology.turn_radians,
            )
            self.assertAlmostEqual(candidate.forward_m, actual[0], places=12)
            self.assertAlmostEqual(candidate.lateral_m, actual[1], places=12)
            self.assertAlmostEqual(
                candidate.first_forward_heading_degrees,
                first_forward_heading_degrees(candidate.actions),
                places=12,
            )
            endpoints.add((round(actual[0], 9), round(actual[1], 9)))
        self.assertEqual(len(endpoints), 33)

    def test_macro_selection_and_horizon(self):
        topology = OnlineTopology(max_decisions=40)
        candidates = generate_macro_candidates(topology)
        selected, error = select_oracle_candidate(
            candidates,
            (0.5, math.sqrt(3.0) / 2.0),
        )
        self.assertEqual(selected.name, "macro_p60_d1p0")
        self.assertEqual(len(selected.actions), 8)
        self.assertAlmostEqual(error, 0.0, places=12)

    def test_world_heading_shield_blocks_two_future_decisions(self):
        topology = OnlineTopology(max_decisions=40)
        candidates = generate_macro_candidates(topology)
        shield = WorldHeadingShield(topology.turn_radians, cooldown_decisions=2)
        shield.block_world_heading(0.0)
        for _ in range(2):
            shield.begin_decision()
            available = shield.available(candidates, current_world_heading=0.0)
            self.assertTrue(available)
            self.assertTrue(
                all(first_forward_heading_degrees(row.actions) != 0.0 for row in available)
            )
        shield.begin_decision()
        available = shield.available(candidates, current_world_heading=0.0)
        self.assertTrue(
            any(first_forward_heading_degrees(row.actions) == 0.0 for row in available)
        )


    def test_collision_heading_trace_is_exact(self):
        topology = OnlineTopology(max_decisions=40)
        headings = trace_collision_world_headings(
            [LEFT_ACTION, FORWARD_ACTION, FORWARD_ACTION],
            [False, True, True],
            start_world_heading=0.0,
            turn_radians=topology.turn_radians,
        )
        self.assertEqual(len(headings), 2)
        self.assertAlmostEqual(headings[0], topology.turn_radians, places=12)
        self.assertAlmostEqual(headings[1], topology.turn_radians, places=12)
    def test_runner_and_manifest_preserve_protocol(self):
        runner = (INTEGRATION / "run_streamvln_candidates_g74.py").read_text()
        for name in ("short_interrupt", "macro_plain", "macro_interrupt"):
            self.assertIn(name, runner)
        self.assertIn('FINAL_VARIANT = "macro_interrupt"', runner)
        self.assertIn('"val_unseen_episode_count": 0', runner)
        self.assertIn('"model_loaded": False', runner)
        self.assertIn('"max_oracle_steps"', runner)

        g73 = json.loads((INTEGRATION / "streamvln_g73_experiment.json").read_text())
        g74 = json.loads((INTEGRATION / "streamvln_g74_experiment.json").read_text())
        self.assertEqual(g74["train"], g73["train"])
        self.assertEqual(g74["val_unseen"], g73["val_unseen"])
        self.assertEqual(g74["source"], g73["source"])
        self.assertEqual(
            g74["stage"], "G7.4 train-only macro and collision-interrupt factorial"
        )


if __name__ == "__main__":
    unittest.main()
