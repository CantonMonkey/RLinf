import json
from pathlib import Path
import unittest

from integrations.rlinf_v03.streamvln_candidates_g73 import (
    compile_executable_candidate,
    generate_executable_candidates,
    select_oracle_executable_candidate,
    simulate_fixed4_endpoint,
)
from integrations.rlinf_v03.streamvln_topology import OnlineTopology, STOP_ACTION


ROOT = Path(__file__).resolve().parents[1]
INTEGRATION = ROOT / "integrations" / "rlinf_v03"


class StreamVLNG73CandidateTest(unittest.TestCase):
    def test_lattice_has_unique_exact_endpoints(self):
        topology = OnlineTopology(max_decisions=10)
        candidates = generate_executable_candidates(topology)
        self.assertEqual(len(candidates), 26)
        endpoints = set()
        for candidate in candidates:
            self.assertEqual(len(candidate.actions), 4)
            self.assertNotIn(STOP_ACTION, candidate.actions)
            actual = simulate_fixed4_endpoint(
                candidate.actions,
                forward_step_m=topology.forward_step_m,
                turn_radians=topology.turn_radians,
            )
            self.assertAlmostEqual(candidate.forward_m, actual[0], places=12)
            self.assertAlmostEqual(candidate.lateral_m, actual[1], places=12)
            endpoints.add((round(actual[0], 9), round(actual[1], 9)))
        self.assertEqual(len(endpoints), len(candidates))

    def test_selection_and_compiler_use_same_candidate(self):
        topology = OnlineTopology(max_decisions=10)
        candidates = generate_executable_candidates(topology)
        selected, error = select_oracle_executable_candidate(
            candidates, (0.7244443697168013, 0.19411428382689055)
        )
        self.assertEqual(selected.name, "fixed4_2111")
        self.assertAlmostEqual(error, 0.0, places=12)
        actions, execute_count = compile_executable_candidate(selected, topology)
        self.assertEqual(tuple(actions), selected.actions)
        self.assertEqual(execute_count, 4)

    def test_runner_and_manifest_preserve_gates(self):
        runner = (INTEGRATION / "run_streamvln_candidates_g73.py").read_text()
        self.assertIn('"g72-result"', runner)
        self.assertIn('"candidate_kind": "deduplicated_fixed4_action_endpoints"', runner)
        self.assertIn('"val_unseen_episode_count": 0', runner)
        self.assertIn('"model_loaded": False', runner)
        self.assertNotIn("generate_dense_candidates", runner)

        g72 = json.loads((INTEGRATION / "streamvln_g72_experiment.json").read_text())
        g73 = json.loads((INTEGRATION / "streamvln_g73_experiment.json").read_text())
        self.assertEqual(g73["train"], g72["train"])
        self.assertEqual(g73["val_unseen"], g72["val_unseen"])
        self.assertEqual(g73["source"], g72["source"])
        self.assertEqual(
            g73["stage"], "G7.3 train-only executor-aligned candidate final gate"
        )


if __name__ == "__main__":
    unittest.main()
