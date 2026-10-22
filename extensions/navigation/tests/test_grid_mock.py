from __future__ import annotations

import unittest

from navrl.contracts import NavAction
from navrl.envs.grid_mock import GridNavEnv
from navrl.episode import GridEpisode


class GridNavSemanticsTest(unittest.TestCase):
    def setUp(self) -> None:
        # Wall at (4, 1), traversable detour through row 3.
        self.episode = GridEpisode("wall-route", (3, 1), (5, 1), heading=0)

    def test_wall_has_no_progress_or_success(self) -> None:
        env = GridNavEnv(max_consecutive_collisions=None)
        try:
            first = env.reset(self.episode)
            self.assertIsNone(first["pose"])
            self.assertEqual(first["step_id"], 0)
            self.assertEqual(env.get_metrics()["geodesic_distance"], 6.0)
            _, reward, terminated, truncated, info = env.step(NavAction.MOVE_FORWARD)
            self.assertEqual(info["geodesic_distance"], 6.0)
            self.assertEqual(info["path_length"], 0.0)
            self.assertTrue(info["collision"])
            self.assertEqual(info["collision_count"], 1)
            self.assertLess(reward, 0.0)
            self.assertFalse(terminated or truncated)
            _, _, terminated, _, info = env.step(NavAction.STOP)
            self.assertTrue(terminated)
            self.assertFalse(info["success"])
            self.assertEqual(info["spl"], 0.0)
            self.assertEqual(info["termination_reason"], "premature_stop")
        finally:
            env.close()

    def test_detour_progress_and_stop_at_goal(self) -> None:
        env = GridNavEnv()
        try:
            env.reset(self.episode)
            env.step(NavAction.TURN_RIGHT)  # face south
            _, reward, _, _, info = env.step(NavAction.MOVE_FORWARD)
            self.assertEqual(info["geodesic_distance"], 5.0)
            self.assertGreater(reward, 0.0)
            # Reset to a one-step episode to isolate STOP semantics and SPL.
            adjacent = GridEpisode("adjacent", (1, 1), (2, 1), heading=0)
            env.reset(adjacent)
            _, _, _, _, info = env.step(NavAction.MOVE_FORWARD)
            self.assertEqual(info["geodesic_distance"], 0.0)
            self.assertFalse(info["success"])
            _, reward, terminated, truncated, info = env.step(NavAction.STOP)
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertTrue(info["success"])
            self.assertEqual(info["spl"], 1.0)
            self.assertGreater(reward, 0.0)
            with self.assertRaises(RuntimeError):
                env.step(NavAction.STOP)
        finally:
            env.close()

    def test_timeout_collision_limit_and_invalid_action(self) -> None:
        env = GridNavEnv(max_steps=2, max_consecutive_collisions=None)
        try:
            env.reset(self.episode)
            env.step("UNKNOWN")
            _, reward, terminated, truncated, info = env.step(NavAction.TURN_LEFT)
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            self.assertTrue(info["timeout"])
            self.assertEqual(info["termination_reason"], "timeout")
            self.assertFalse(info["invalid_action"])  # per-step, not sticky
            self.assertLess(reward, 0.0)
        finally:
            env.close()
        env = GridNavEnv(max_consecutive_collisions=2)
        try:
            env.reset(self.episode)
            env.step(NavAction.MOVE_FORWARD)
            _, _, terminated, truncated, info = env.step(NavAction.MOVE_FORWARD)
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertEqual(info["termination_reason"], "collision_limit")
            self.assertEqual(info["collision_count"], 2)
        finally:
            env.close()

    def test_success_distance_is_strict(self) -> None:
        adjacent = GridEpisode("threshold", (1, 1), (2, 1), heading=0)
        for radius, expected in ((1.0, False), (1.01, True)):
            env = GridNavEnv(success_radius=radius)
            try:
                env.reset(adjacent)
                _, _, terminated, truncated, info = env.step(NavAction.STOP)
                self.assertTrue(terminated)
                self.assertFalse(truncated)
                self.assertEqual(info["geodesic_distance"], 1.0)
                self.assertEqual(info["success"], expected)
            finally:
                env.close()

    def test_seeded_replay_and_reset_no_leakage(self) -> None:
        left = GridNavEnv(seed=19, max_steps=1)
        right = GridNavEnv(seed=19, max_steps=1)
        try:
            for _ in range(100):
                self.assertEqual(left.reset(), right.reset())
                self.assertEqual(left.current_episode, right.current_episode)
                self.assertEqual(left.get_metrics(), right.get_metrics())
                self.assertEqual(left.step(NavAction.TURN_LEFT), right.step(NavAction.TURN_LEFT))
                self.assertTrue(left.get_metrics()["timeout"])
                self.assertEqual(left.get_metrics()["collision_count"], 0)
            first = left.reset(seed=19)
            initial = GridNavEnv(seed=19)
            try:
                self.assertEqual(first, initial.reset())
            finally:
                initial.close()
        finally:
            left.close()
            right.close()


if __name__ == "__main__":
    unittest.main()
