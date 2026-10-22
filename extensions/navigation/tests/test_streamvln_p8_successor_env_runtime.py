"""Torch-backed, model-free tests for the production successor env boundary."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch


ROOT = Path(__file__).parents[1]
STAGE = ROOT / "integrations" / "rlinf_v03"
sys.path.insert(0, str(STAGE))

from streamvln_env import StreamVLNEnv  # noqa: E402
from streamvln_p8_successor_action import (  # noqa: E402
    FORWARD_ACTION,
    STOP_ACTION,
    RouteDecision,
    RouteName,
    SuccessorPlanner,
    compile_route,
)
from streamvln_p8_successor_env import StreamVLNP8SuccessorEnv  # noqa: E402
from streamvln_p8_successor_executor import (  # noqa: E402
    PRODUCTION_SUCCESSOR_EXECUTION_SCHEMA_NAME,
)
from streamvln_topology import OnlineTopology  # noqa: E402


class FakeSuccessorEnv(StreamVLNP8SuccessorEnv):
    def __init__(self, outcomes):
        self.successor_planner = SuccessorPlanner()
        self._outcomes = list(outcomes)
        self.calls = []
        self._previous_path_length = 0.0
        self._previous_actions = []
        self._terminal_frozen = False

    def _base_primitive_step(self, action: int):
        self.calls.append(action)
        if not self._outcomes:
            raise AssertionError("executor consumed an undeclared primitive")
        outcome = self._outcomes.pop(0)
        return (
            {"rgb": torch.zeros((1, 1, 1, 3), dtype=torch.uint8)},
            torch.tensor([outcome.get("reward", 0.0)], dtype=torch.float32),
            torch.tensor([outcome.get("terminated", False)], dtype=torch.bool),
            torch.tensor([outcome.get("truncated", False)], dtype=torch.bool),
            {
                "path_length": torch.tensor(
                    [outcome["path_length"]], dtype=torch.float32
                ),
                "collision": torch.tensor(
                    [outcome.get("collision", False)], dtype=torch.bool
                ),
            },
        )

    def _stream_obs(self, base_obs, *, episode_start: bool):
        return {"episode_start": episode_start, "call_count": len(self.calls)}


class StreamVLNP8SuccessorEnvRuntimeTests(unittest.TestCase):
    def test_collision_discards_tail_and_preserves_fixed_return_shape(self):
        topology = OnlineTopology(max_decisions=8)
        plan = compile_route(
            RouteDecision(0, RouteName.WAYPOINT, (1.0, 0.0)), topology
        )
        env = FakeSuccessorEnv(
            [
                {"path_length": 0.25},
                {"path_length": 0.25, "collision": True},
            ]
        )
        obs, rewards, terminated, truncated, infos = env.successor_step(plan)
        self.assertEqual(env.calls, [FORWARD_ACTION, FORWARD_ACTION])
        self.assertEqual(tuple(rewards.shape), (1, 4))
        self.assertEqual(tuple(terminated.shape), (1, 4))
        self.assertEqual(tuple(truncated.shape), (1, 4))
        self.assertEqual(len(obs), 1)
        feedback = infos[0]["executor_feedback"]
        self.assertEqual(feedback["proposed_actions"], [FORWARD_ACTION] * 4)
        self.assertEqual(feedback["executed_actions"], [FORWARD_ACTION] * 2)
        self.assertEqual(infos[0]["successor_execution"]["replan_reason"], "collision")
        self.assertEqual(
            infos[0]["successor_execution"]["schema_name"],
            PRODUCTION_SUCCESSOR_EXECUTION_SCHEMA_NAME,
        )
        self.assertTrue(infos[0]["successor_execution"]["collision_interrupted"])

    def test_transport_padding_is_not_executed(self):
        topology = OnlineTopology(max_decisions=8)
        plan = compile_route(
            RouteDecision(0, RouteName.WAYPOINT, (0.24, 0.0)), topology
        )
        env = FakeSuccessorEnv([{"path_length": 0.24}])
        env.successor_step(plan)
        self.assertEqual(env.calls, [FORWARD_ACTION])

    def test_transport_rejects_missing_count_float_actions_and_stop_tail(self):
        valid = np.asarray([[[1], [0], [0], [0]]], dtype=np.int64)
        env = FakeSuccessorEnv([])
        with self.assertRaises(ValueError):
            env.chunk_step(valid)
        with self.assertRaises(TypeError):
            env.chunk_step(valid.astype(np.float32), execute_count=1)
        with self.assertRaises(ValueError):
            env.chunk_step(valid, execute_count=2)

    def test_truncation_precedes_stop_in_both_feedback_views(self):
        env = FakeSuccessorEnv([{"path_length": 0.0, "truncated": True}])
        _, _, _, truncated, infos = env.chunk_step(
            np.asarray([[[0], [0], [0], [0]]], dtype=np.int64),
            execute_count=1,
        )
        self.assertTrue(bool(truncated.any()))
        self.assertEqual(infos[0]["executor_feedback"]["end_reason"], "truncated")
        self.assertEqual(
            infos[0]["successor_execution"]["replan_reason"], "truncated"
        )

    def test_reset_clears_lower_layer_recovery_latch(self):
        env = object.__new__(StreamVLNP8SuccessorEnv)
        env.successor_planner = SuccessorPlanner(recovery_parent_index=3)
        sentinel = ({"obs": True}, {"identity": True})
        with patch.object(StreamVLNEnv, "reset", return_value=sentinel):
            self.assertEqual(StreamVLNP8SuccessorEnv.reset(env), sentinel)
        self.assertIsNone(env.successor_planner.recovery_parent_index)

    def test_plan_and_step_uses_env_owned_planner_for_waypoint(self):
        topology = OnlineTopology(max_decisions=8)
        env = FakeSuccessorEnv([{"path_length": 0.25}])
        result = env.plan_and_step(
            RouteDecision(0, RouteName.WAYPOINT, (0.24, 0.0)),
            topology,
        )
        self.assertEqual(env.calls, [FORWARD_ACTION])
        self.assertEqual(
            result[-1][0]["successor_plan"]["executed_route"],
            RouteName.WAYPOINT.value,
        )

    def test_env_plan_entry_rejects_direct_official_keep_binding(self):
        topology = OnlineTopology(max_decisions=8)
        env = FakeSuccessorEnv([])
        decision = RouteDecision(
            0,
            RouteName.KEEP_ORIGINAL,
            official_action=FORWARD_ACTION,
        )
        with self.assertRaisesRegex(ValueError, "cannot bind"):
            env.plan_and_step(decision, topology)

        plan = SuccessorPlanner(recovery_enabled=False).plan(decision, topology)
        with self.assertRaisesRegex(ValueError, "OfficialKeepDriver"):
            env.successor_step(plan)


if __name__ == "__main__":
    unittest.main()
