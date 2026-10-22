"""PointNav transition tests without importing Habitat or RLinf."""

import contextlib
import io
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.pointnav_service import (
    PointNavSession,
    parse_args,
    pointnav_dataset_path,
    select_episodes,
    service_ready_payload,
    serve_requests,
)


class PointNavDatasetSelectionTests(unittest.TestCase):
    def test_ready_reports_loaded_split_and_episode_selection(self):
        backend = SimpleNamespace(
            split="train",
            data_path=Path("/isolated/assets/datasets/train/train.json.gz"),
            selected_indices=[50, 51],
            episode_ids=["50", "51"],
            scene_id="skokloster-castle.glb",
        )
        ready = service_ready_payload(backend)
        self.assertEqual(ready["split"], "train")
        self.assertEqual(ready["selected_indices"], [50, 51])
        self.assertEqual(ready["episode_ids"], ["50", "51"])
        self.assertEqual(ready["scene_id"], "skokloster-castle.glb")

    def test_default_cli_keeps_val_path_and_episode_zero(self):
        args = parse_args(["--asset-root", "/isolated/assets"])
        self.assertEqual(args.split, "val")
        self.assertEqual(args.premature_stop_penalty, 0.0)
        self.assertEqual(args.episode_index, 0)
        self.assertIsNone(args.episode_indices)
        self.assertEqual(
            pointnav_dataset_path(args.asset_root),
            args.asset_root.resolve()
            / "datasets/pointnav/habitat-test-scenes/v1/val/val.json.gz",
        )

    def test_train_cli_resolves_train_file_and_selected_same_scene_indices(self):
        args = parse_args([
            "--asset-root", "/isolated/assets", "--split", "train",
            "--episode-indices", "0,1",
        ])
        self.assertEqual(
            pointnav_dataset_path(args.asset_root, args.split),
            args.asset_root.resolve()
            / "datasets/pointnav/habitat-test-scenes/v1/train/train.json.gz",
        )
        episodes = [
            SimpleNamespace(episode_id="0", scene_id="scene-a"),
            SimpleNamespace(episode_id="1", scene_id="scene-a"),
            SimpleNamespace(episode_id="2", scene_id="scene-b"),
        ]
        selected = select_episodes(
            episodes, episode_indices=[int(x) for x in args.episode_indices.split(",")],
            split=args.split,
        )
        self.assertEqual([episode.episode_id for episode in selected], ["0", "1"])
        with self.assertRaisesRegex(ValueError, "cannot switch scenes"):
            select_episodes(episodes, episode_indices=[0, 2], split="train")
        with self.assertRaisesRegex(IndexError, "PointNav train dataset"):
            select_episodes(episodes, episode_indices=[0, 3], split="train")

    def test_duplicate_or_unknown_selection_rejected(self):
        episodes = [SimpleNamespace(scene_id="scene-a")]
        with self.assertRaisesRegex(ValueError, "nonempty and unique"):
            select_episodes(episodes, episode_indices=[0, 0], split="train")
        with self.assertRaisesRegex(ValueError, "nonempty and unique"):
            select_episodes(episodes, episode_indices=[], split="train")
        with self.assertRaisesRegex(ValueError, "unsupported PointNav split"):
            pointnav_dataset_path(Path("/isolated/assets"), "test")


class FakePointNavBackend:
    def __init__(self, start=1.0, after_move=0.75, collision=False):
        self.start = start
        self.after_move = after_move
        self.collides = collision
        self.distance = start
        self.steps = 0
        self.close_called = False

    def _frame(self, action=None):
        collision = action == "move_forward" and self.collides
        success = action == "stop" and self.distance < 0.2
        return {
            "states": [self.distance, 0.0],
            "geodesic_distance": self.distance,
            "euclidean_distance": self.distance,
            "success": success,
            "spl": float(success),
            "collision": collision,
            "collision_count": int(collision),
            "path_length": 0.0 if collision else self.steps * 0.25,
            "episode_over": action == "stop",
            "episode_id": "0",
            "scene_id": "test-scene",
            "position_xyz": [1.0 - self.distance, 0.0, 0.0],
            "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            "goal_position_xyz": [1.0, 0.0, 0.0],
        }

    def reset(self):
        self.distance = self.start
        self.steps = 0
        return self._frame()

    def step(self, action):
        self.steps += 1
        if action == "move_forward" and not self.collides:
            self.distance = self.after_move
        return self._frame(action)

    def close(self):
        self.close_called = True


class PointNavSessionTests(unittest.TestCase):
    def test_optional_premature_stop_penalty_keeps_success_bonus_and_freeze(self):
        failed = PointNavSession(
            FakePointNavBackend(start=1.0), premature_stop_penalty=1.0,
        )
        failed.reset()
        step = failed.step(0)
        self.assertEqual(step["info"]["termination_reason"], "premature_stop")
        self.assertAlmostEqual(step["reward"], -1.01)
        self.assertEqual(failed.step(1)["reward"], 0.0)

        successful = PointNavSession(
            FakePointNavBackend(start=0.199), premature_stop_penalty=1.0,
        )
        successful.reset()
        self.assertAlmostEqual(successful.step(0)["reward"], 9.99)

        for invalid in (-1.0, float("inf"), float("nan")):
            with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                PointNavSession(FakePointNavBackend(), premature_stop_penalty=invalid)
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args([
                        "--asset-root", "/isolated/assets",
                        "--premature-stop-penalty", str(invalid),
                    ])

    def test_premature_stop_terminates_without_success_bonus_and_freezes(self):
        backend = FakePointNavBackend(start=1.0)
        session = PointNavSession(backend)
        self.assertEqual(len(session.reset()["obs"]), 2)
        stopped = session.step(0)
        self.assertTrue(stopped["terminated"])
        self.assertFalse(stopped["truncated"])
        self.assertFalse(stopped["info"]["success"])
        self.assertEqual(stopped["info"]["termination_reason"], "premature_stop")
        self.assertAlmostEqual(stopped["reward"], -0.01)
        frozen = session.step(1)
        self.assertEqual(backend.steps, 1)
        self.assertEqual(frozen["obs"], stopped["obs"])
        self.assertEqual(frozen["reward"], 0.0)
        self.assertFalse(frozen["terminated"] or frozen["truncated"])
        self.assertTrue(frozen["info"]["terminal_frozen"])
        session.close()
        self.assertTrue(backend.close_called)

    def test_near_goal_stop_uses_strict_habitat_success(self):
        session = PointNavSession(FakePointNavBackend(start=0.199))
        session.reset()
        stopped = session.step(0)
        self.assertTrue(stopped["info"]["success"])
        self.assertEqual(stopped["info"]["spl"], 1.0)
        self.assertAlmostEqual(stopped["reward"], 9.99)

        boundary = PointNavSession(FakePointNavBackend(start=0.2))
        boundary.reset()
        stopped = boundary.step(0)
        self.assertFalse(stopped["info"]["success"])

    def test_geodesic_progress_collision_and_timeout(self):
        session = PointNavSession(FakePointNavBackend(start=1.0, after_move=0.75), max_steps=2)
        session.reset()
        step = session.step(1)
        self.assertAlmostEqual(step["reward"], 0.24)
        self.assertFalse(step["terminated"] or step["truncated"])
        timeout = session.step(2)
        self.assertTrue(timeout["truncated"])
        self.assertFalse(timeout["terminated"])
        self.assertTrue(timeout["info"]["timeout"])
        self.assertEqual(session.step(3)["reward"], 0.0)

        collision = PointNavSession(FakePointNavBackend(collision=True))
        collision.reset()
        hit = collision.step(1)
        self.assertAlmostEqual(hit["reward"], -0.21)
        self.assertTrue(hit["info"]["collision"])
        self.assertFalse(hit["terminated"] or hit["truncated"])

    def test_invalid_action_fails_before_backend_step(self):
        backend = FakePointNavBackend()
        session = PointNavSession(backend)
        session.reset()
        for invalid in (True, -1, 4, 0.5, "stop"):
            with self.assertRaises(ValueError):
                session.step(invalid)
        self.assertEqual(backend.steps, 0)

    def test_bad_json_request_does_not_kill_protocol(self):
        backend = FakePointNavBackend()
        session = PointNavSession(backend)
        requests = io.StringIO('{bad json}\n{"cmd":"reset"}\n{"cmd":"close"}\n')
        response_stream = io.StringIO()
        with contextlib.redirect_stderr(io.StringIO()):
            serve_requests(session, requests, response_stream)
        responses = [json.loads(line) for line in response_stream.getvalue().splitlines()]
        self.assertIn("error", responses[0])
        self.assertEqual(len(responses[1]["obs"]), 2)
        self.assertTrue(responses[2]["closed"])

    def test_diagnostic_is_separate_from_policy_transition(self):
        session = PointNavSession(FakePointNavBackend(start=1.0, after_move=0.75))
        with self.assertRaises(RuntimeError):
            session.diagnostic()
        requests = io.StringIO(
            '{"cmd":"reset"}\n{"cmd":"diagnostic"}\n'
            '{"cmd":"step","action":1}\n{"cmd":"diagnostic"}\n'
            '{"cmd":"close"}\n'
        )
        responses = io.StringIO()
        serve_requests(session, requests, responses)
        reset, initial, step, moved, closed = [
            json.loads(line) for line in responses.getvalue().splitlines()
        ]
        self.assertEqual(reset["obs"], [1.0, 0.0])
        self.assertEqual(step["obs"], [0.75, 0.0])
        self.assertNotIn("position_xyz", reset)
        self.assertNotIn("position_xyz", reset["info"])
        self.assertNotIn("position_xyz", step)
        self.assertNotIn("position_xyz", step["info"])
        self.assertEqual(initial["position_xyz"], [0.0, 0.0, 0.0])
        self.assertEqual(moved["position_xyz"], [0.25, 0.0, 0.0])
        self.assertEqual(initial["goal_position_xyz"], moved["goal_position_xyz"])
        self.assertTrue(closed["closed"])


if __name__ == "__main__":
    unittest.main()
