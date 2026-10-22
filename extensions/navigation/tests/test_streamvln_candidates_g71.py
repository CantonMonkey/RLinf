import inspect
import unittest

from integrations.rlinf_v03.streamvln_candidates_g71 import (
    DirectionFeasibility,
    compile_dense_adaptive,
    generate_dense_candidates,
    select_oracle_dense_candidate,
)
from integrations.rlinf_v03.streamvln_topology import (
    FORWARD_ACTION,
    OnlineTopology,
    STOP_ACTION,
)


class StreamVLNG71CandidateTests(unittest.TestCase):
    def test_dense_candidates_are_goal_free_and_multiscale(self):
        parameters = inspect.signature(generate_dense_candidates).parameters
        self.assertNotIn("goal", parameters)
        self.assertNotIn("expert", parameters)
        topology = OnlineTopology(max_decisions=10)
        candidates = generate_dense_candidates(topology)
        self.assertEqual(len(candidates), 10)
        self.assertEqual(
            {candidate.heading_degrees for candidate in candidates},
            {-30.0, -15.0, 0.0, 15.0, 30.0},
        )
        self.assertEqual(
            {candidate.distance_m for candidate in candidates}, {0.5, 1.0}
        )

    def test_adaptive_compiler_exposes_executed_prefix(self):
        topology = OnlineTopology(max_decisions=10)
        candidates = generate_dense_candidates(topology)
        short_forward = next(
            item
            for item in candidates
            if item.heading_degrees == 0.0 and item.distance_m == 0.5
        )
        actions, count = compile_dense_adaptive(short_forward, topology)
        self.assertEqual(count, 2)
        self.assertEqual(actions[:count], [FORWARD_ACTION, FORWARD_ACTION])
        self.assertEqual(actions[count:], [STOP_ACTION, STOP_ACTION])

        left_short = next(
            item
            for item in candidates
            if item.heading_degrees == 15.0 and item.distance_m == 0.5
        )
        actions, count = compile_dense_adaptive(left_short, topology)
        self.assertEqual(count, 3)
        self.assertEqual(actions[count], STOP_ACTION)

    def test_direction_blacklist_shares_across_distance(self):
        topology = OnlineTopology(max_decisions=10)
        candidates = generate_dense_candidates(topology)
        short = next(
            item
            for item in candidates
            if item.heading_degrees == 0.0 and item.distance_m == 0.5
        )
        long = next(
            item
            for item in candidates
            if item.heading_degrees == 0.0 and item.distance_m == 1.0
        )
        tracker = DirectionFeasibility(cooldown_decisions=2)
        self.assertTrue(
            tracker.update(short, [1, 1], [False, True])
        )
        self.assertNotIn(short, tracker.available(candidates))
        self.assertNotIn(long, tracker.available(candidates))
        tracker.begin_decision()
        self.assertNotIn(short, tracker.available(candidates))
        tracker.begin_decision()
        self.assertIn(short, tracker.available(candidates))

    def test_oracle_dense_selection_prefers_short_candidate(self):
        topology = OnlineTopology(max_decisions=10)
        selected, error = select_oracle_dense_candidate(
            generate_dense_candidates(topology), [0.48, 0.12]
        )
        self.assertEqual(selected.distance_m, 0.5)
        self.assertEqual(selected.heading_degrees, 15.0)
        self.assertLess(error, 0.02)


if __name__ == "__main__":
    unittest.main()
