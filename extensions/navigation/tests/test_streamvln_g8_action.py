import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).parents[1]
STAGE = ROOT / "integrations" / "rlinf_v03"
sys.path.insert(0, str(STAGE))

from streamvln_g8_action import (  # noqa: E402
    ACTION_COUNT,
    CANDIDATE_ORDER_SHA256,
    NAVIGATE_ACTION_COUNT,
    RECOVER_ACTION_ID,
    STOP_ACTION_ID,
    action_mode,
    action_waypoint_local_m,
    assert_g8_action_contract,
    compile_g8_action,
    executable_candidates,
    target_action_id,
)
from streamvln_topology import (  # noqa: E402
    FORWARD_ACTION,
    NAVIGATE_MODE,
    RECOVER_MODE,
    STOP_ACTION,
    STOP_MODE,
    OnlineTopology,
)


class StreamVLNG8ActionTest(unittest.TestCase):
    def test_frozen_action_vocabulary_maps_to_exact_macros(self):
        topology = OnlineTopology(max_decisions=4)
        contract = assert_g8_action_contract()
        self.assertEqual(contract["action_count"], 28)
        self.assertEqual(contract["navigate_candidate_count"], 26)
        self.assertEqual(contract["unique_navigate_macros"], 26)
        self.assertEqual(contract["candidate_order_sha256"], CANDIDATE_ORDER_SHA256)

        candidates = executable_candidates(topology)
        for action_id, candidate in enumerate(candidates):
            self.assertEqual(action_mode(action_id), NAVIGATE_MODE)
            self.assertEqual(compile_g8_action(action_id, topology), list(candidate.actions))
            self.assertEqual(action_waypoint_local_m(action_id, topology), candidate.vector())
        self.assertEqual(action_mode(STOP_ACTION_ID), STOP_MODE)
        self.assertEqual(compile_g8_action(STOP_ACTION_ID, topology), [STOP_ACTION] * 4)
        self.assertEqual(action_mode(RECOVER_ACTION_ID), RECOVER_MODE)
        self.assertEqual(len(compile_g8_action(RECOVER_ACTION_ID, topology)), 4)
        self.assertEqual(ACTION_COUNT, NAVIGATE_ACTION_COUNT + 2)

    def test_bc_target_uses_nearest_executable_endpoint(self):
        topology = OnlineTopology(max_decisions=4)
        candidates = executable_candidates(topology)
        for expected, candidate in enumerate(candidates):
            self.assertEqual(
                target_action_id(NAVIGATE_MODE, candidate.vector(), topology),
                expected,
            )
        self.assertEqual(target_action_id(STOP_MODE, (0.0, 0.0), topology), STOP_ACTION_ID)
        self.assertEqual(target_action_id(RECOVER_MODE, (0.0, 0.0), topology), RECOVER_ACTION_ID)

    def test_recover_target_comes_from_causal_topology(self):
        topology = OnlineTopology(max_decisions=4)
        self.assertEqual(action_waypoint_local_m(RECOVER_ACTION_ID, topology), (-0.5, 0.0))
        topology.update([FORWARD_ACTION] * 4, [False] * 4)
        recovery = action_waypoint_local_m(RECOVER_ACTION_ID, topology)
        self.assertIsNotNone(topology.nodes[topology.current_node].parent)
        self.assertLess(recovery[0], -0.9)
        self.assertAlmostEqual(recovery[1], 0.0, places=7)

    def test_machine_contract_matches_implementation(self):
        manifest = json.loads(
            (STAGE / "streamvln_g8_experiment.json").read_text(encoding="utf-8")
        )
        action = manifest["action_contract"]
        self.assertEqual(action["action_count"], ACTION_COUNT)
        self.assertEqual(action["navigate_candidate_count"], NAVIGATE_ACTION_COUNT)
        self.assertEqual(action["stop_action_id"], STOP_ACTION_ID)
        self.assertEqual(action["recover_action_id"], RECOVER_ACTION_ID)
        self.assertEqual(action["candidate_order_sha256"], CANDIDATE_ORDER_SHA256)
        self.assertFalse(action["gaussian_waypoint_likelihood"])


if __name__ == "__main__":
    unittest.main()
