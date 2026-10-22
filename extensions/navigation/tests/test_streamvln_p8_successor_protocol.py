import math
from pathlib import Path
import pickle
import sys
import unittest


MODULE_DIR = Path(__file__).parents[1] / "integrations" / "rlinf_v03"
sys.path.insert(0, str(MODULE_DIR))

from streamvln_p8_successor_protocol import (  # noqa: E402
    SUCCESSOR_REPLAY_SCHEMA,
    SuccessorDecisionSample,
    SuccessorMacroAccumulator,
    SuccessorRequestBatch,
    SuccessorResultBatch,
    SuccessorStepRequest,
    SuccessorStepResult,
    compute_successor_gae,
)
from streamvln_p8_successor_action import (  # noqa: E402
    CompiledRoute,
    RouteDecision,
    RouteName,
)


class SuccessorProtocolSchemaTest(unittest.TestCase):
    def test_breaking_topology_feedback_schema_has_unique_version(self):
        self.assertEqual(
            SUCCESSOR_REPLAY_SCHEMA,
            "streamvln-p8-successor-decision-v2-topology",
        )

    def test_two_shard_request_and_result_roundtrip(self):
        requests = SuccessorRequestBatch(
            tuple(
                SuccessorStepRequest(
                    episode_token=f"episode-{index}",
                    physical_step=index,
                    decision_index=index // 2,
                    needs_decision=True,
                    decision_mask=True,
                    official_shadow_action=1,
                    decision_sample=SuccessorDecisionSample(
                        decision=RouteDecision(
                            decision_index=index // 2,
                            route=RouteName.WAYPOINT,
                            waypoint_local_m=(0.25, -0.5),
                            episode_token=f"episode-{index}",
                            physical_step=index,
                        ),
                        proposed_actions=(1, 2),
                        logprob=-0.25,
                        value=0.5,
                        official_shadow_action=1,
                    ),
                )
                for index in range(4)
            )
        )
        request_shards = tuple(
            pickle.loads(pickle.dumps(shard))
            for shard in requests.split_by_sizes([1, 3])
        )
        self.assertEqual(SuccessorRequestBatch.merge_shards(request_shards), requests)

        results = SuccessorResultBatch(
            tuple(
                SuccessorStepResult(
                    episode_token=f"episode-{index}",
                    physical_step=index,
                    decision_index=index // 2,
                    needs_decision=False,
                    decision_mask=True,
                    executed_action=1,
                    reward=float(index),
                    realized_path_m=0.25,
                )
                for index in range(4)
            )
        )
        result_shards = tuple(
            pickle.loads(pickle.dumps(shard))
            for shard in results.split_by_sizes([2, 2])
        )
        self.assertEqual(SuccessorResultBatch.merge_shards(result_shards), results)


class SuccessorMacroAccumulatorTest(unittest.TestCase):
    def _request(self, accumulator, *, actions=None, shadow=1):
        sample = None
        if actions is not None:
            decision = RouteDecision(
                decision_index=accumulator.decision_index,
                route=RouteName.WAYPOINT,
                waypoint_local_m=(0.5, 0.0),
                episode_token=accumulator.episode_token,
                physical_step=accumulator.physical_step,
            )
            plan = CompiledRoute(
                decision_index=decision.decision_index,
                requested_route=RouteName.WAYPOINT,
                executed_route=RouteName.WAYPOINT,
                waypoint_local_m=(0.5, 0.0),
                proposed_actions=actions,
                source="test_metric_waypoint",
                recovery_override=False,
                recovery_parent_index=None,
                target_distance_m=0.5,
                target_bearing_rad=0.0,
            )
            sample = SuccessorDecisionSample(
                decision=decision,
                proposed_actions=actions,
                logprob=-0.5,
                value=0.25,
                official_shadow_action=shadow,
                compiled_route=plan,
            )
        return SuccessorStepRequest(
            episode_token=accumulator.episode_token,
            physical_step=accumulator.physical_step,
            decision_index=accumulator.decision_index,
            needs_decision=accumulator.needs_decision,
            decision_mask=accumulator.decision_mask,
            official_shadow_action=shadow,
            decision_sample=sample,
        )

    def _feedback(self, command, *, reward, next_needs_decision, **flags):
        return SuccessorStepResult(
            episode_token=command.episode_token,
            physical_step=command.physical_step,
            decision_index=command.decision_index,
            needs_decision=next_needs_decision,
            decision_mask=command.decision_mask,
            executed_action=command.executed_action,
            reward=reward,
            realized_path_m=0.25,
            **flags,
        )

    def test_four_primitives_close_one_macro_row(self):
        accumulator = SuccessorMacroAccumulator("episode-a")
        transition = None
        actions = (1, 2, 3, 1)
        for index, action in enumerate(actions):
            request = self._request(
                accumulator,
                actions=actions if index == 0 else None,
                shadow=action,
            )
            self.assertEqual(request.decision_mask, index == 0)
            command = accumulator.prepare_step(request)
            feedback = self._feedback(
                command,
                reward=float(index + 1),
                next_needs_decision=index == 3,
            )
            if index == 0:
                self.assertTrue(command.decision_mask)
                self.assertFalse(feedback.needs_decision)
            if index == 3:
                self.assertFalse(command.decision_mask)
                self.assertTrue(feedback.needs_decision)
            transition = accumulator.accept_feedback(feedback)
            if index < 3:
                self.assertIsNone(transition)

        self.assertIsNotNone(transition)
        self.assertEqual(transition.executed_actions, actions)
        self.assertEqual(transition.executed_mask, (True, True, True, True))
        self.assertEqual(transition.primitive_rewards, (1.0, 2.0, 3.0, 4.0))
        self.assertEqual(transition.duration, 4)
        self.assertEqual(transition.end_reason, "plan_complete")
        self.assertEqual(transition.decision_sample.logprob, -0.5)
        self.assertEqual(transition.decision_sample.value, 0.25)
        self.assertEqual(accumulator.physical_step, 4)
        self.assertEqual(accumulator.decision_index, 1)
        self.assertTrue(accumulator.needs_decision)

    def test_second_step_collision_closes_prefix_and_reset_rejects_stale(self):
        accumulator = SuccessorMacroAccumulator("episode-old")
        first = accumulator.prepare_step(self._request(accumulator, actions=(1, 1, 2, 3)))
        self.assertIsNone(
            accumulator.accept_feedback(
                self._feedback(first, reward=0.1, next_needs_decision=False)
            )
        )

        continuation = self._request(accumulator, shadow=1)
        self.assertFalse(continuation.decision_mask)
        second = accumulator.prepare_step(continuation)
        transition = accumulator.accept_feedback(
            self._feedback(
                second,
                reward=-1.0,
                next_needs_decision=True,
                collision=True,
            )
        )
        self.assertEqual(transition.duration, 2)
        self.assertEqual(transition.executed_mask, (True, True, False, False))
        self.assertEqual(transition.primitive_rewards, (0.1, -1.0, 0.0, 0.0))
        self.assertEqual(transition.end_reason, "collision")

        stale_request = SuccessorStepRequest(
            episode_token="episode-old",
            physical_step=2,
            decision_index=1,
            needs_decision=True,
            decision_mask=True,
            official_shadow_action=1,
            decision_sample=SuccessorDecisionSample(
                decision=RouteDecision(
                    decision_index=1,
                    route=RouteName.WAYPOINT,
                    waypoint_local_m=(0.5, 0.0),
                    episode_token="episode-old",
                    physical_step=2,
                ),
                proposed_actions=(1,),
                logprob=-0.5,
                value=0.25,
                official_shadow_action=1,
            ),
        )
        accumulator.reset("episode-new")
        with self.assertRaisesRegex(ValueError, "stale episode_token"):
            accumulator.prepare_step(stale_request)

    def test_official_stop_interrupts_pending_prefix(self):
        accumulator = SuccessorMacroAccumulator("episode-stop")
        first = accumulator.prepare_step(
            self._request(accumulator, actions=(1, 1, 1, 1), shadow=1)
        )
        self.assertIsNone(
            accumulator.accept_feedback(
                self._feedback(first, reward=0.1, next_needs_decision=False)
            )
        )

        continuation = self._request(accumulator, shadow=0)
        self.assertFalse(continuation.decision_mask)
        command = accumulator.prepare_step(continuation)
        self.assertEqual(command.executed_action, 0)
        self.assertTrue(command.official_stop_override)
        transition = accumulator.accept_feedback(
            self._feedback(
                command,
                reward=0.0,
                next_needs_decision=True,
                terminated=True,
            )
        )
        self.assertEqual(transition.duration, 2)
        self.assertEqual(transition.executed_mask, (True, True, False, False))
        self.assertEqual(transition.end_reason, "official_stop_override")
        self.assertEqual(transition.decision_sample.official_shadow_action, 1)
        self.assertFalse(transition.replan_required)
        self.assertTrue(accumulator.needs_decision)

    def test_rejects_duplicate_feedback(self):
        accumulator = SuccessorMacroAccumulator("episode-duplicate")
        command = accumulator.prepare_step(self._request(accumulator, actions=(1,)))
        feedback = self._feedback(command, reward=0.0, next_needs_decision=True)
        self.assertIsNotNone(accumulator.accept_feedback(feedback))
        with self.assertRaisesRegex(RuntimeError, "no matching"):
            accumulator.accept_feedback(feedback)


class SuccessorDurationGAETest(unittest.TestCase):
    def test_opt_in_uses_physical_step_powers_for_k_greater_than_one(self):
        args = dict(
            chunk_rewards=[2.0, 7.0],
            values=[0.25, 0.5],
            next_values=[0.5, 0.75],
            durations=[2, 4],
            terminations=[False, False],
            truncations=[False, True],
            gamma=0.5,
            gae_lambda=0.8,
        )
        advantages, _, deltas = compute_successor_gae(**args, duration_aware=True)
        expected_delta_1 = 2.0 + 0.5**2 * 0.5 - 0.25
        expected_delta_2 = 7.0 + 0.5**4 * 0.75 - 0.5
        expected_advantage_1 = expected_delta_1 + (0.5 * 0.8) ** 2 * expected_delta_2
        self.assertTrue(math.isclose(deltas[0], expected_delta_1))
        self.assertTrue(math.isclose(advantages[0], expected_advantage_1))

    def test_k_one_is_identical_with_opt_in_disabled_or_enabled(self):
        args = dict(
            chunk_rewards=[1.0, -0.25],
            values=[0.2, 0.3],
            next_values=[0.3, 0.0],
            durations=[1, 1],
            terminations=[False, True],
            truncations=[False, False],
            gamma=0.9,
            gae_lambda=0.95,
        )
        baseline = compute_successor_gae(**args, duration_aware=False)
        duration_aware = compute_successor_gae(**args, duration_aware=True)
        self.assertEqual(baseline, duration_aware)


if __name__ == "__main__":
    unittest.main()
