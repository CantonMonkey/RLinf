import math
import unittest

from integrations.rlinf_v03.streamvln_recovery import (
    compile_recovery_aware,
    recovery_required,
    resolved_choice,
    waypoint_execution_error,
)
from integrations.rlinf_v03.streamvln_topology import (
    FORWARD_ACTION,
    NAVIGATE_MODE,
    OnlineTopology,
    RECOVER_MODE,
)


class StreamVLNRecoveryTests(unittest.TestCase):
    def test_collision_trigger_and_goal_free_override(self):
        topology = OnlineTopology(max_decisions=10)
        topology.update([FORWARD_ACTION, FORWARD_ACTION], [True, True])
        self.assertTrue(recovery_required(topology, blocked_threshold=2))
        choice = resolved_choice(
            NAVIGATE_MODE,
            [1.0, 0.0],
            topology,
            recovery_enabled=True,
            blocked_threshold=2,
        )
        self.assertEqual(choice["executed_mode"], RECOVER_MODE)
        self.assertTrue(choice["recovery_override"])
        self.assertEqual(choice["choice_source"], "collision_override")

    def test_recovery_compiler_reserves_a_forward_probe(self):
        topology = OnlineTopology(max_decisions=10)
        actions = compile_recovery_aware([-0.5, 0.0], RECOVER_MODE, topology)
        self.assertEqual(len(actions), 4)
        self.assertIn(FORWARD_ACTION, actions)
        self.assertLessEqual(sum(action != FORWARD_ACTION for action in actions), 3)

    def test_waypoint_error_uses_dead_reckoned_execution_only(self):
        topology = OnlineTopology(max_decisions=10)
        start = (topology.x, topology.z, topology.heading)
        topology.update([FORWARD_ACTION], [False])
        result = waypoint_execution_error([0.25, 0.0], start, topology)
        self.assertAlmostEqual(result["l2_m"], 0.0)
        self.assertTrue(all(math.isfinite(value) for value in result["realized_local_m"]))


if __name__ == "__main__":
    unittest.main()
