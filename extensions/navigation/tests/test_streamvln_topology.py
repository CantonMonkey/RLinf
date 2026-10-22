import math
import unittest

from integrations.rlinf_v03.streamvln_topology import (
    FEATURE_NAMES,
    FORWARD_ACTION,
    LEFT_ACTION,
    NAVIGATE_MODE,
    OnlineTopology,
    RECOVER_MODE,
    STOP_ACTION,
    STOP_MODE,
    compile_waypoint,
    executed_prefix,
)


class StreamVLNTopologyTests(unittest.TestCase):
    def test_collision_aware_dead_reckoning_and_graph(self):
        topology = OnlineTopology(max_decisions=10)
        self.assertEqual(len(topology.feature_vector()), len(FEATURE_NAMES))
        topology.update([FORWARD_ACTION], [True])
        self.assertEqual((topology.x, topology.z), (0.0, 0.0))
        self.assertEqual(topology.blocked_forwards, 1)
        topology.update([FORWARD_ACTION, FORWARD_ACTION], [False, False])
        self.assertAlmostEqual(topology.z, 0.5)
        self.assertEqual(len(topology.nodes), 2)
        self.assertEqual(len(topology.edges), 1)
        backtrack = topology.backtrack_waypoint()
        self.assertIsNotNone(backtrack)
        self.assertAlmostEqual(backtrack[0], -0.5)
        self.assertAlmostEqual(backtrack[1], 0.0)

    def test_waypoint_compiler_separates_stop_navigation_and_recovery(self):
        topology = OnlineTopology(max_decisions=10)
        self.assertEqual(compile_waypoint([0.0, 0.0], STOP_MODE, topology), [0] * 4)
        self.assertEqual(executed_prefix([0, 0, 0, 0]), [STOP_ACTION])
        left = compile_waypoint([0.5, 0.5], NAVIGATE_MODE, topology)
        self.assertEqual(len(left), 4)
        self.assertEqual(left[0], LEFT_ACTION)
        self.assertNotIn(STOP_ACTION, left)
        recover = compile_waypoint([1.0, 0.0], RECOVER_MODE, topology)
        self.assertEqual(len(recover), 4)
        self.assertNotIn(STOP_ACTION, recover)

    def test_features_use_only_online_execution_state(self):
        topology = OnlineTopology(max_decisions=4)
        topology.update([2, 2, 1, 1], [False] * 4)
        features = topology.feature_vector()
        self.assertTrue(all(math.isfinite(value) for value in features))
        forbidden = {"goal", "geodesic", "reference_path", "success"}
        self.assertTrue(forbidden.isdisjoint(FEATURE_NAMES))


if __name__ == "__main__":
    unittest.main()
