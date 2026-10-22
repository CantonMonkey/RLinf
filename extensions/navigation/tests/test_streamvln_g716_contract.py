import math
import hashlib
import json
from pathlib import Path
import unittest

from integrations.rlinf_v03.streamvln_executor_contract import (
    FORWARD_ACTION,
    HighLevelCommand,
    assert_history_only_keys,
    chunk_feedback,
    feedback_from_dict,
    primitive_execution,
)
from integrations.rlinf_v03.streamvln_topology import OnlineTopology


def _primitive(index, action, before, after, *, collision=False, reward=0.0,
               terminated=False, truncated=False):
    return primitive_execution(
        primitive_index=index,
        action=action,
        previous_path_length_m=before,
        current_path_length_m=after,
        collision=collision,
        environment_reward=reward,
        terminated=terminated,
        truncated=truncated,
    )


class StreamVLNG716ContractTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]

    def test_high_level_command_and_privileged_blacklist_are_explicit(self):
        command = HighLevelCommand(3, "navigate", 0.8, -0.2)
        self.assertEqual(command.to_dict()["decision_index"], 3)
        HighLevelCommand(4, "stop", 0.0, 0.0)
        with self.assertRaisesRegex(ValueError, "zero local waypoint"):
            HighLevelCommand(4, "stop", 0.1, 0.0)
        assert_history_only_keys(["rgb", "instruction_utf8", "previous_actions"])
        for key in ("navmesh", "geodesic_distance", "future_collision", "severity_label"):
            with self.assertRaisesRegex(ValueError, "privileged"):
                assert_history_only_keys(["rgb", key])

    def test_feedback_uses_post_action_path_increment_and_exact_reward_once(self):
        rows = (
            _primitive(0, 2, 0.0, 0.0, reward=-0.01),
            _primitive(1, FORWARD_ACTION, 0.0, 0.20, collision=True, reward=0.19),
            _primitive(2, FORWARD_ACTION, 0.20, 0.30, reward=-0.11),
        )
        feedback = chunk_feedback([2, 1, 1, 1], rows)
        self.assertEqual(feedback.executed_actions, (2, 1, 1))
        self.assertEqual(feedback.executed_mask, (True, True, True, False))
        self.assertEqual(feedback.collision_count, 1)
        self.assertEqual(feedback.blocked_forward_count, 1)
        self.assertTrue(math.isclose(feedback.commanded_forward_m, 0.5))
        self.assertTrue(math.isclose(feedback.realized_path_m, 0.3))
        self.assertTrue(math.isclose(feedback.environment_reward_sum, 0.07))
        self.assertTrue(math.isclose(
            feedback.discounted_environment_reward(0.9),
            -0.01 + 0.9 * 0.19 + 0.9**2 * -0.11,
        ))
        self.assertEqual(feedback_from_dict(feedback.to_dict()), feedback)

    def test_collision_and_blockage_remain_distinct_for_partial_motion(self):
        moved = _primitive(0, FORWARD_ACTION, 0.0, 0.20, collision=True)
        blocked = _primitive(1, FORWARD_ACTION, 0.20, 0.30, collision=False)
        self.assertTrue(moved.collision)
        self.assertFalse(moved.blocked)
        self.assertFalse(blocked.collision)
        self.assertTrue(blocked.blocked)

        feedback = chunk_feedback([1, 1, 2, 2], [moved, blocked])
        topology = OnlineTopology(max_decisions=4)
        topology.update_from_feedback(feedback.to_dict())
        self.assertTrue(math.isclose(topology.z, 0.30))
        self.assertEqual(topology.forward_attempts, 2)
        self.assertEqual(topology.blocked_forwards, 1)

    def test_feedback_rejects_decreasing_path_and_schema_tampering(self):
        with self.assertRaisesRegex(ValueError, "decreased"):
            _primitive(0, 1, 0.2, 0.1)
        with self.assertRaisesRegex(ValueError, "at least one"):
            chunk_feedback([1], [])
        feedback = chunk_feedback([1], [_primitive(0, 1, 0.0, 0.25)])
        raw = feedback.to_dict()
        raw["future_outcome"] = "oracle"
        with self.assertRaisesRegex(ValueError, "schema"):
            feedback_from_dict(raw)

    def test_frozen_manifest_keeps_g8_closed(self):
        integration = self.ROOT / "integrations" / "rlinf_v03"
        path = integration / "streamvln_g716_contract.json"
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(),
            "3a60baace9865f6df88c2b8369bd988ddced5b64624306b307d5b175676ba15c",
        )
        manifest = json.loads(path.read_text())
        smoke = manifest["deterministic_smoke"]
        self.assertEqual(smoke["split"], "train")
        self.assertEqual(smoke["repeat_count"], 2)
        self.assertEqual(smoke["expected_executed_primitives"], 36)
        self.assertEqual(smoke["val_unseen_episode_count"], 0)
        self.assertFalse(manifest["model_load"])
        self.assertFalse(manifest["rl_update"])
        self.assertFalse(manifest["g8_ready"])
        self.assertTrue(
            set(manifest["high_level_policy_observation_keys"]).isdisjoint(
                manifest["privileged_policy_blacklist"]
            )
        )
        runner = (integration / "run_streamvln_contract_g716.py").read_text()
        self.assertIn("feedback_from_dict", runner)
        self.assertIn("collapse_chunk_rewards", runner)
        self.assertIn('"val_unseen_episode_count": 0', runner)

    def test_historical_plan_keeps_g8_closed(self):
        plan_path = self.ROOT.parent / "STREAMVLN_G716_EXPERIMENT_PLAN.md"
        if not plan_path.is_file():
            self.skipTest("Historical experiment plan is not included in the public source export")
        plan = plan_path.read_text()
        self.assertIn("Preregistered on 2026-09-27 before the first G7.16", plan)
        self.assertIn("G8 and `val_unseen` remain closed", plan)


if __name__ == "__main__":
    unittest.main()
