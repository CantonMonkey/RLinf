import inspect
import math
import unittest

from integrations.rlinf_v03.streamvln_candidates import (
    compile_candidate,
    compile_phased_recovery,
    generate_waypoint_candidates,
    select_oracle_candidate,
)
from integrations.rlinf_v03.streamvln_topology import (
    FORWARD_ACTION,
    OnlineTopology,
    STOP_ACTION,
)


class StreamVLNCandidateTests(unittest.TestCase):
    def test_candidate_generation_is_goal_free_and_adds_online_parent(self):
        parameters = inspect.signature(generate_waypoint_candidates).parameters
        self.assertNotIn("goal", parameters)
        self.assertNotIn("expert", parameters)
        topology = OnlineTopology(max_decisions=10)
        names = [item.name for item in generate_waypoint_candidates(topology)]
        self.assertEqual(names, ["forward", "left_front", "right_front"])

        topology.update([FORWARD_ACTION, FORWARD_ACTION], [False, False])
        candidates = generate_waypoint_candidates(topology)
        self.assertEqual([item.name for item in candidates][-1], "parent")
        self.assertEqual(candidates[-1].source, "online_topology")

    def test_oracle_selection_reports_quantization_error(self):
        topology = OnlineTopology(max_decisions=10)
        candidates = generate_waypoint_candidates(topology)
        selected, error = select_oracle_candidate(candidates, [0.85, 0.45])
        self.assertEqual(selected.name, "left_front")
        self.assertLess(error, 0.1)
        actions = compile_candidate(selected, topology)
        self.assertEqual(len(actions), 4)
        self.assertNotIn(STOP_ACTION, actions)

    def test_phased_recovery_aligns_then_translates(self):
        topology = OnlineTopology(max_decisions=10)
        topology.update([FORWARD_ACTION, FORWARD_ACTION], [False, False])
        actions, record = compile_phased_recovery(topology)
        self.assertEqual(record["phase"], "align")
        self.assertEqual(len(actions), 4)
        self.assertNotIn(FORWARD_ACTION, actions)

        topology.heading = math.pi
        actions, record = compile_phased_recovery(topology)
        self.assertEqual(record["phase"], "translate")
        self.assertEqual(actions.count(FORWARD_ACTION), 2)
        self.assertNotIn(STOP_ACTION, actions)

    def test_root_recovery_has_bounded_probe(self):
        topology = OnlineTopology(max_decisions=10)
        actions, record = compile_phased_recovery(topology)
        self.assertEqual(record["phase"], "root_probe")
        self.assertEqual(len(actions), 4)
        self.assertEqual(actions.count(FORWARD_ACTION), 2)


if __name__ == "__main__":
    unittest.main()
