"""R2R JSONL reward and terminal protocol checks without Habitat installed."""

import unittest

from scripts.habitat_r2r_service import R2RSession, local_waypoint_components


def frame(distance, *, success=False, episode_over=False):
    return {
        "policy_obs": {
            "rgb": {"encoding": "base64/raw-u8-hwc", "shape_hwc": [1, 1, 3], "data_b64": "AAAA"},
            "instruction_ids": [1, 0],
            "instruction_mask": [True, False],
        },
        "episode_id": "4862",
        "scene_id": "/scene/one.glb",
        "instruction_sha256": "instruction-hash",
        "geodesic_distance": distance,
        "success": success,
        "spl": 1.0 if success else 0.0,
        "path_length": 0.25,
        "collision": False,
        "collision_count": 0,
        "episode_over": episode_over,
    }


class FakeBackend:
    official_streamvln_frame = True

    def __init__(self, frames):
        self.frames = iter(frames)
        self.actions = []
        self.official_payload_calls = 0

    def reset(self):
        return next(self.frames)

    def step(self, name):
        self.actions.append(name)
        return next(self.frames)

    def expert_waypoint(self, lookahead_m):
        return {
            "schema_version": 1,
            "source": "train-shortest-path",
            "lookahead_m": float(lookahead_m),
            "waypoint_local_m": [0.8, -0.1],
            "bearing_rad": -0.124,
            "target_distance_m": 0.806,
            "distance_to_goal_m": 4.6,
            "stop": False,
        }

    def official_frame_payload(self, current_frame, service_step, episode_generation):
        self.official_payload_calls += 1
        return {
            "private": True,
            "service_step": service_step,
            "episode_generation": episode_generation,
            "episode_id": current_frame["episode_id"],
        }

    def close(self):
        pass


class R2RServiceTest(unittest.TestCase):
    def test_habitat_camera_x_is_converted_to_left_positive(self):
        forward, lateral = local_waypoint_components([-2.0, 0.0, -3.0])
        self.assertEqual((forward, lateral), (3.0, 2.0))
        forward, lateral = local_waypoint_components([2.0, 0.0, -3.0])
        self.assertEqual((forward, lateral), (3.0, -2.0))

    def test_progress_stop_and_frozen_terminal(self):
        backend = FakeBackend([frame(4.6), frame(4.3), frame(4.3, episode_over=True)])
        session = R2RSession(backend, max_steps=8)
        reset = session.reset()
        self.assertEqual(set(reset["obs"]), {"rgb", "instruction_ids", "instruction_mask"})
        move = session.step(1)
        self.assertAlmostEqual(move["reward"], 0.29)
        self.assertFalse(move["terminated"])
        stop = session.step(0)
        self.assertTrue(stop["terminated"])
        self.assertEqual(stop["info"]["success"], 0.0)
        frozen = session.step(1)
        self.assertEqual(frozen["reward"], 0.0)
        self.assertFalse(frozen["terminated"])
        self.assertTrue(frozen["info"]["terminal_frozen"])
        self.assertEqual(backend.actions, ["move_forward", "stop"])

    def test_expert_waypoint_is_separate_from_policy_observation(self):
        session = R2RSession(FakeBackend([frame(4.6)]), max_steps=8)
        with self.assertRaisesRegex(RuntimeError, "reset must precede"):
            session.expert_waypoint(1.0)
        reset = session.reset()
        self.assertEqual(set(reset["obs"]), {"rgb", "instruction_ids", "instruction_mask"})
        label = session.expert_waypoint(1.0)
        self.assertEqual(label["source"], "train-shortest-path")
        self.assertEqual(label["waypoint_local_m"], [0.8, -0.1])
        self.assertNotIn("obs", label)

    def test_success_bonus_and_timeout(self):
        success = R2RSession(FakeBackend([frame(2.9), frame(2.8, success=True)]), max_steps=2)
        success.reset()
        terminal = success.step(0)
        self.assertAlmostEqual(terminal["reward"], 10.09)
        self.assertEqual(terminal["info"]["success"], 1.0)
        self.assertTrue(terminal["terminated"])
        timeout = R2RSession(FakeBackend([frame(4.6), frame(4.4)]), max_steps=1)
        timeout.reset()
        truncated = timeout.step(1)
        self.assertTrue(truncated["truncated"])
        self.assertFalse(truncated["terminated"])

    def test_official_frame_is_private_and_state_preserving(self):
        backend = FakeBackend([frame(4.6), frame(4.3), frame(4.6)])
        session = R2RSession(backend, max_steps=8)
        reset = session.reset()
        self.assertNotIn("official_frame", reset["obs"])
        generation = reset["info"]["episode_generation"]
        private = session.observe_official_frame(
            expected_service_step=0,
            episode_generation=generation,
        )
        self.assertEqual(private["service_step"], 0)
        self.assertEqual(private["episode_generation"], generation)
        self.assertTrue(private["private"])
        self.assertEqual(backend.actions, [])
        self.assertEqual(backend.official_payload_calls, 1)
        with self.assertRaisesRegex(RuntimeError, "stale service step"):
            session.observe_official_frame(
                expected_service_step=True,
                episode_generation=generation,
            )
        with self.assertRaisesRegex(RuntimeError, "stale episode generation"):
            session.observe_official_frame(
                expected_service_step=0,
                episode_generation="f" * 32,
            )
        self.assertEqual(backend.official_payload_calls, 1)
        session.step(1)
        self.assertEqual(
            session.observe_official_frame(
                expected_service_step=1,
                episode_generation=generation,
            )["service_step"],
            1,
        )
        reset_again = session.reset()
        next_generation = reset_again["info"]["episode_generation"]
        self.assertNotEqual(next_generation, generation)
        with self.assertRaisesRegex(RuntimeError, "stale episode generation"):
            session.observe_official_frame(
                expected_service_step=0,
                episode_generation=generation,
            )
        self.assertEqual(backend.official_payload_calls, 2)
        self.assertEqual(
            session.observe_official_frame(
                expected_service_step=0,
                episode_generation=next_generation,
            )["episode_generation"],
            next_generation,
        )


if __name__ == "__main__":
    unittest.main()
