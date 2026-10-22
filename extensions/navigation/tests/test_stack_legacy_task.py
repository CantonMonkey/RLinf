from __future__ import annotations

import unittest

from navrl.contracts import NavAction
from navrl.envs.grid_mock import GridNavEnv
from navrl.episode import GridEpisode
from navrl.stack import LegacyNavTaskAdapter


class LegacyTaskBridgeTest(unittest.TestCase):
    def test_grid_task_has_fresh_reset_identity_and_typed_metrics(self):
        environment = GridNavEnv(expose_pose=True, max_consecutive_collisions=None)
        task = LegacyNavTaskAdapter(environment)
        episode = GridEpisode("bridge-train", (1, 1), (2, 1), heading=0)
        try:
            first = task.reset(episode, seed=4)
            self.assertIsNone(first.observation["pose"])
            moved = task.step(NavAction.MOVE_FORWARD)
            self.assertEqual(1, moved.physical_step)
            stopped = task.step(NavAction.STOP)
            self.assertTrue(stopped.terminated)
            metrics = task.get_metrics()
            self.assertEqual(first.identity, metrics.identity)
            self.assertTrue(metrics.success)
            self.assertEqual(1.0, metrics.spl)
            self.assertEqual(1.0, metrics.path_length_m)
            self.assertEqual(
                {"MOVE_FORWARD": 1, "STOP": 1}, dict(metrics.action_counts)
            )

            second = task.reset(episode, seed=4)
            self.assertEqual(first.identity.episode_id, second.identity.episode_id)
            self.assertNotEqual(first.identity.reset_token, second.identity.reset_token)
        finally:
            task.close()

    def test_observation_transform_runs_after_pose_is_hidden(self):
        environment = GridNavEnv(expose_pose=True)
        task = LegacyNavTaskAdapter(
            environment,
            observation_transform=lambda obs: {
                "episode": obs["episode_id"],
                "pose_hidden": obs["pose"] is None,
            },
        )
        try:
            reset = task.reset(GridEpisode("transform-train", (1, 1), (2, 1)))
            self.assertEqual(
                {"episode": "transform-train", "pose_hidden": True},
                reset.observation,
            )
        finally:
            task.close()


if __name__ == "__main__":
    unittest.main()
