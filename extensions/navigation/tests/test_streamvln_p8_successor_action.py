from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).parents[1]
STAGE = ROOT / "integrations" / "rlinf_v03"
sys.path.insert(0, str(STAGE))

from streamvln_p8_successor_action import (  # noqa: E402
    FORWARD_ACTION,
    LEFT_ACTION,
    RIGHT_ACTION,
    STOP_ACTION,
    ParentRecoveryUnavailable,
    RouteDecision,
    RouteName,
    SuccessorPlanner,
)
from streamvln_p8_successor_executor import (  # noqa: E402
    PrimitiveOutcome,
    execute_feedback_aware,
)
from streamvln_topology import OnlineTopology  # noqa: E402


class StreamVLNP8SuccessorActionTests(unittest.TestCase):
    @staticmethod
    def _blocked_root() -> OnlineTopology:
        topology = OnlineTopology(max_decisions=8)
        topology.update([FORWARD_ACTION, FORWARD_ACTION], [True, True])
        return topology

    def test_default_blocked_root_keep_still_raises(self):
        decision = RouteDecision(
            1,
            RouteName.KEEP_ORIGINAL,
            official_action=RIGHT_ACTION,
        )

        with self.assertRaises(ParentRecoveryUnavailable):
            SuccessorPlanner(blocked_threshold=2).plan(
                decision,
                self._blocked_root(),
            )

    def test_opt_in_blocked_root_keep_executes_official_primitive_with_receipt(self):
        topology = self._blocked_root()

        for action in (STOP_ACTION, FORWARD_ACTION, LEFT_ACTION, RIGHT_ACTION):
            with self.subTest(action=action):
                decision = RouteDecision(
                    2,
                    RouteName.KEEP_ORIGINAL,
                    official_action=action,
                )
                planner = SuccessorPlanner(
                    blocked_threshold=2,
                    root_recovery_policy="keep_original",
                )
                plan = planner.plan(decision, topology)

                self.assertEqual(plan.requested_route, RouteName.KEEP_ORIGINAL)
                self.assertEqual(plan.executed_route, RouteName.KEEP_ORIGINAL)
                self.assertEqual(plan.proposed_actions, (action,))
                self.assertEqual(plan.source, "root_keep_fallback")
                self.assertFalse(plan.recovery_override)
                self.assertIsNone(plan.recovery_parent_index)
                self.assertEqual(plan.to_dict()["source"], "root_keep_fallback")
                self.assertIsNone(planner.recovery_parent_index)

                execution = execute_feedback_aware(
                    plan,
                    lambda primitive, _index: PrimitiveOutcome(action=primitive),
                )
                audit = execution.to_trace()
                self.assertEqual(audit["source"], "root_keep_fallback")
                self.assertEqual(audit["compiled_actions"], [action])
                self.assertEqual(audit["executed_actions"], [action])

    def test_opt_in_does_not_change_waypoint_or_parent_recovery_routes(self):
        root = self._blocked_root()
        planner = SuccessorPlanner(
            blocked_threshold=2,
            root_recovery_policy="keep_original",
        )
        with self.assertRaises(ParentRecoveryUnavailable):
            planner.plan(
                RouteDecision(3, RouteName.WAYPOINT, (1.0, 0.0)),
                root,
            )

        topology = OnlineTopology(max_decisions=8)
        topology.update(
            [FORWARD_ACTION, FORWARD_ACTION, FORWARD_ACTION],
            [False, False, False],
        )
        topology.update([FORWARD_ACTION, FORWARD_ACTION], [True, True])
        recovered = SuccessorPlanner(
            blocked_threshold=2,
            root_recovery_policy="keep_original",
        ).plan(
            RouteDecision(
                4,
                RouteName.KEEP_ORIGINAL,
                official_action=RIGHT_ACTION,
            ),
            topology,
        )
        self.assertEqual(recovered.executed_route, RouteName.RECOVER_PARENT)
        self.assertEqual(recovered.source, "deterministic_topology_parent")
        self.assertTrue(recovered.recovery_override)
        self.assertIsNotNone(recovered.recovery_parent_index)

    def test_opt_in_preserves_unblocked_and_exact_keep_parity(self):
        decision = RouteDecision(
            5,
            RouteName.KEEP_ORIGINAL,
            official_action=LEFT_ACTION,
        )
        topology = OnlineTopology(max_decisions=8)
        baseline = SuccessorPlanner().plan(decision, topology)
        opt_in = SuccessorPlanner(
            root_recovery_policy="keep_original",
        ).plan(decision, topology)
        self.assertEqual(opt_in, baseline)
        self.assertEqual(opt_in.source, "official_upstream_queue")

        exact_keep = SuccessorPlanner(
            recovery_enabled=False,
            root_recovery_policy="keep_original",
        ).plan(decision, self._blocked_root())
        self.assertEqual(exact_keep.executed_route, RouteName.KEEP_ORIGINAL)
        self.assertEqual(exact_keep.proposed_actions, (LEFT_ACTION,))
        self.assertEqual(exact_keep.source, "official_upstream_queue")


if __name__ == "__main__":
    unittest.main()
