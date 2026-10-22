from __future__ import annotations

import ast
import base64
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest


RLINF_NAV_ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = RLINF_NAV_ROOT / "integrations" / "rlinf_v03"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import lightnav_l0_native_probe as probe


def prediction(
    seq: int,
    waypoint: tuple[float, float, float],
    *,
    stop: bool = False,
    latency_ms: float = 12.0,
) -> probe.LightNavPrediction:
    rows = ((0.0, 0.0, 0.0),) if stop else (waypoint,)
    return probe.LightNavPrediction(
        seq=seq,
        stop=stop,
        waypoints=rows,
        selected_index=0,
        selected_waypoint=(0.0, 0.0, 0.0) if stop else waypoint,
        latency_ms=latency_ms,
        timings_ms={"llm_ms": latency_ms / 2.0},
        raw_text_sha256="a" * 64,
    )


class FakeModel:
    def __init__(self, predictions):
        self.predictions = list(predictions)
        self.login_ids = []
        self.reset_count = 0
        self.requests = []

    def login(self, client_id):
        self.login_ids.append(client_id)

    def reset(self):
        self.reset_count += 1

    def predict(self, **kwargs):
        self.requests.append(kwargs)
        item = self.predictions.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class FakeEnv:
    def __init__(
        self,
        transitions,
        *,
        collision_metric=True,
        per_step_collision_metric=False,
    ):
        self.rgb = b"fake-rgb"
        self.transitions = list(transitions)
        self.actions = []
        self.collision_metric = collision_metric
        self.per_step_collision_metric = per_step_collision_metric

    def _info(self, **updates):
        value = {
            "scene_id": "scene-a",
            "episode_id": "episode-1",
            "distance_to_goal": 5.0,
            "path_length": 0.0,
            "success": 0.0,
            "spl": 0.0,
            "habitat_time_step": 0.1,
            "lin_vel_range": (0.0, 2.5),
            "ang_vel_range": (-300.0, 300.0),
        }
        if self.collision_metric:
            value["collision_count"] = 0
        elif self.per_step_collision_metric:
            value["collisions"] = {"is_collision": False}
        value.update(updates)
        return value

    def reset(self):
        return {
            "rgb": self.rgb,
            "instruction": {"text": "walk to the chair"},
        }, self._info()

    def step(self, action):
        self.actions.append(action)
        update = self.transitions.pop(0)
        info = self._info(**update["info"])
        return (
            {"rgb": self.rgb, "instruction": {"text": "walk to the chair"}},
            0.0,
            update["terminated"],
            update["truncated"],
            info,
        )


class LightNavProtocolTest(unittest.TestCase):
    def test_wire_requests_match_official_protocol(self):
        self.assertEqual(
            probe.login_request("client-1"),
            {"action": "login", "data": {"clientId": "client-1"}},
        )
        self.assertEqual(probe.reset_request(), {"action": "reset", "data": {}})
        self.assertEqual(
            probe.next_request(seq=3, image_b64="YWJj", instruction="go"),
            {
                "action": "next",
                "data": {"seq": 3, "image": "YWJj", "instruction": "go"},
            },
        )

    def test_jpeg_encoder_round_trip_has_rgb_jpeg(self):
        try:
            import numpy as np
            from PIL import Image
        except ImportError as exc:
            self.skipTest(str(exc))
        rgb = np.zeros((6, 9, 3), dtype=np.uint8)
        rgb[:, :, 1] = 200
        encoded = probe.encode_rgb_jpeg_base64(rgb, quality=95)
        with Image.open(__import__("io").BytesIO(base64.b64decode(encoded))) as image:
            self.assertEqual(image.mode, "RGB")
            self.assertEqual(image.size, (9, 6))

    def test_robust_action_parsing_and_yaw_preservation(self):
        payload = {
            "action": "next",
            "data": {
                "rc": 0,
                "seq": 7,
                "actions": {
                    "step": 2,
                    "actions": [
                        [0.0, 0.25, 0.0],
                        [0.10, -0.03, math.radians(15.0)],
                        [0.20, -0.06, math.radians(30.0)],
                    ],
                },
                "stop": False,
                "latency_ms": 14.5,
                "timings_ms": {"vit_ms": 4.0, "llm_ms": 8.0},
                "raw_text": "<apos_1><act_l0_2><act_l1_3><act_l2_4>",
            },
        }
        parsed = probe.parse_lightnav_response(payload, expected_seq=7)
        # Lateral-only row cannot move the official unicycle controller; the next
        # forward/yaw row is selected, including its yaw rather than reducing to XY.
        self.assertEqual(parsed.selected_index, 1)
        self.assertEqual(parsed.selected_waypoint[:2], (0.10, -0.03))
        self.assertAlmostEqual(parsed.selected_waypoint[2], math.radians(15.0))
        self.assertEqual(len(parsed.raw_text_sha256), 64)

        wrapped = probe.parse_actions_payload([[[0.1, 0.0, 0.2], [0.2, 0.0, 0.3]]])
        flat = probe.parse_actions_payload([0.1, 0.0, 0.2, 0.2, 0.0, 0.3])
        self.assertEqual(wrapped, flat)

    def test_decode_error_and_stop_inconsistency_are_invalid(self):
        error_payload = {
            "action": "next",
            "data": {"rc": 500, "seq": 0, "msg": "missing RVQ level"},
        }
        with self.assertRaises(probe.LightNavDecodeError):
            probe.parse_lightnav_response(error_payload, expected_seq=0)

        inconsistent = {
            "action": "next",
            "data": {
                "rc": 0,
                "seq": 0,
                "actions": {"step": 1, "actions": [[0.2, 0.0, 0.0]]},
                "stop": True,
                "latency_ms": 1.0,
            },
        }
        with self.assertRaisesRegex(probe.LightNavProtocolError, "stop flag"):
            probe.parse_lightnav_response(inconsistent, expected_seq=0)

    def test_non_stop_all_lateral_chunk_is_invalid_not_stop(self):
        response = {
            "action": "next",
            "data": {
                "rc": 0,
                "seq": 0,
                "actions": {"actions": [[0.0, 0.2, 0.0], [0.0, 0.4, 0.0]]},
                "stop": False,
                "latency_ms": 2.0,
            },
        }
        with self.assertRaisesRegex(probe.LightNavProtocolError, "velocity-relevant"):
            probe.parse_lightnav_response(response, expected_seq=0)

    def test_websocket_client_runs_login_reset_next_and_closes_locally(self):
        class FakeSocket:
            def __init__(self):
                self.sent = []
                self.closed = False
                self.responses = [
                    {"action": "login", "data": {"rc": 0}},
                    {"action": "reset", "data": {"rc": 0}},
                    {
                        "action": "next",
                        "data": {
                            "rc": 0,
                            "seq": 0,
                            "actions": {"actions": [[0.0, 0.0, 0.0]]},
                            "stop": True,
                            "latency_ms": 3.0,
                            "raw_text": "stop",
                        },
                    },
                ]

            def send(self, raw):
                self.sent.append(json.loads(raw))

            def recv(self, timeout=None):
                self.timeout = timeout
                return json.dumps(self.responses.pop(0))

            def close(self):
                self.closed = True

        socket = FakeSocket()
        factory_calls = []

        def connect(uri, **kwargs):
            factory_calls.append((uri, kwargs))
            return socket

        client = probe.LightNavWebSocketClient(
            "ws://127.0.0.1:18050", timeout_s=9.0, connect_factory=connect
        )
        client.open()
        client.login("client-1")
        client.reset()
        prediction = client.predict(seq=0, image_b64="YWJj", instruction="go")
        client.close()

        self.assertEqual(factory_calls[0][0], "ws://127.0.0.1:18050")
        self.assertEqual(socket.sent[0], probe.login_request("client-1"))
        self.assertEqual(socket.sent[1], probe.reset_request())
        self.assertEqual(
            socket.sent[2],
            probe.next_request(seq=0, image_b64="YWJj", instruction="go"),
        )
        self.assertTrue(prediction.stop)
        self.assertTrue(socket.closed)


class VelocityConversionTest(unittest.TestCase):
    def setUp(self):
        self.spec = probe.VelocitySpec(0.1, (0.0, 2.5), (-300.0, 300.0))

    def test_official_velocity_mapping_preserves_yaw(self):
        action = probe.waypoint_to_velocity_action(
            (0.25, 99.0, math.radians(30.0)), self.spec
        )
        self.assertAlmostEqual(action["action_args"]["linear_velocity"], 1.0)
        self.assertAlmostEqual(action["action_args"]["angular_velocity"], 1.0)
        self.assertEqual(probe.classify_velocity_action(action, self.spec), "FORWARD_LEFT")

    def test_physical_zero_is_not_raw_zero(self):
        action = probe.zero_waypoint_action(self.spec)
        self.assertEqual(action["action"], "velocity_control")
        self.assertAlmostEqual(action["action_args"]["linear_velocity"], -1.0)
        self.assertAlmostEqual(action["action_args"]["angular_velocity"], 0.0)
        self.assertEqual(probe.classify_velocity_action(action, self.spec), "STOP")


class NativeEpisodeTest(unittest.TestCase):
    @staticmethod
    def _run(env, model, config):
        return probe.run_episode(
            env,
            model,
            config,
            image_encoder=lambda _rgb, quality: f"jpeg-quality-{quality}",
        )

    def test_model_stop_runs_and_reports_required_metrics(self):
        env = FakeEnv(
            [
                {
                    "terminated": False,
                    "truncated": False,
                    "info": {"distance_to_goal": 4.5, "path_length": 0.25},
                },
                {
                    "terminated": True,
                    "truncated": False,
                    "info": {
                        "distance_to_goal": 2.8,
                        "path_length": 0.25,
                        "success": 1.0,
                        "spl": 0.6,
                        "termination_reason": "agent_stop",
                    },
                },
            ]
        )
        model = FakeModel(
            [
                prediction(0, (0.25, 0.0, math.radians(30.0))),
                prediction(1, (0.0, 0.0, 0.0), stop=True),
            ]
        )
        result = self._run(env, model, probe.ProbeConfig(episodes=1, max_steps=4))
        row = result.receipt
        self.assertTrue(row["success"])
        self.assertTrue(row["model_stop"])
        self.assertFalse(row["forced_stop"])
        self.assertFalse(row["timeout"])
        self.assertEqual(row["action_distribution"], {"FORWARD_LEFT": 1, "STOP": 1})
        self.assertAlmostEqual(row["final_navigation_error_m"], 2.8)
        self.assertAlmostEqual(row["path_length_m"], 0.25)
        self.assertEqual(row["collision_count"], 0.0)
        self.assertEqual(row["server_latency"]["count"], 2)
        self.assertAlmostEqual(env.actions[0]["action_args"]["angular_velocity"], 1.0)
        self.assertAlmostEqual(env.actions[1]["action_args"]["linear_velocity"], -1.0)

        causal = row["causal_steps"]
        self.assertEqual(len(causal), 2)
        first = causal[0]
        self.assertEqual(first["seq"], 0)
        self.assertEqual(first["raw_text_sha256"], "a" * 64)
        self.assertEqual(
            first["trajectory_sha256"],
            probe.canonical_sha256(((0.25, 0.0, math.radians(30.0)),)),
        )
        self.assertEqual(first["selected_index"], 0)
        self.assertEqual(first["selected_se2"], [0.25, 0.0, math.radians(30.0)])
        self.assertAlmostEqual(first["normalized_command"]["linear"], 1.0)
        self.assertAlmostEqual(first["normalized_command"]["angular"], 1.0)
        self.assertEqual(first["server_latency_ms"], 12.0)
        self.assertFalse(first["stop_flag"])
        self.assertNotIn("raw_text", first)
        self.assertNotIn("image", first)

        stop = causal[1]
        self.assertEqual(stop["seq"], 1)
        self.assertEqual(stop["raw_text_sha256"], "a" * 64)
        self.assertTrue(stop["stop_flag"])
        self.assertEqual(stop["stop_source"], "model_stop")
        self.assertAlmostEqual(stop["normalized_command"]["linear"], -1.0)
        self.assertAlmostEqual(stop["normalized_command"]["angular"], 0.0)

    def test_decode_error_never_executes_zero_fallback(self):
        env = FakeEnv([])
        model = FakeModel([probe.LightNavDecodeError("bad RVQ")])
        with self.assertRaises(probe.LightNavDecodeError):
            self._run(env, model, probe.ProbeConfig(episodes=1, max_steps=4))
        self.assertEqual(env.actions, [])

    def test_max_step_uses_terminal_physical_zero_and_marks_timeout(self):
        env = FakeEnv(
            [
                {
                    "terminated": False,
                    "truncated": False,
                    "info": {"distance_to_goal": 4.75, "path_length": 0.25},
                },
                {
                    "terminated": True,
                    "truncated": True,
                    "info": {
                        "distance_to_goal": 4.75,
                        "path_length": 0.25,
                        "success": 0.0,
                        "spl": 0.0,
                        "termination_reason": "agent_stop",
                    },
                },
            ]
        )
        model = FakeModel([prediction(0, (0.25, 0.0, 0.0))])
        row = self._run(
            env, model, probe.ProbeConfig(episodes=1, max_steps=2)
        ).receipt
        self.assertTrue(row["forced_stop"])
        self.assertTrue(row["timeout"])
        self.assertEqual(row["stop_source"], "max_step_forced_stop")
        self.assertEqual(len(model.requests), 1)
        self.assertAlmostEqual(env.actions[-1]["action_args"]["linear_velocity"], -1.0)
        forced = row["causal_steps"][-1]
        self.assertIsNone(forced["seq"])
        self.assertIsNone(forced["raw_text_sha256"])
        self.assertEqual(len(forced["trajectory_sha256"]), 64)
        self.assertIsNone(forced["selected_index"])
        self.assertEqual(forced["selected_se2"], [0.0, 0.0, 0.0])
        self.assertTrue(forced["stop_flag"])
        self.assertEqual(forced["stop_source"], "max_step_forced_stop")
        self.assertAlmostEqual(forced["normalized_command"]["linear"], -1.0)
        self.assertAlmostEqual(forced["normalized_command"]["angular"], 0.0)

    def test_missing_collision_is_explicit_or_fail_closed(self):
        transition = {
            "terminated": True,
            "truncated": True,
            "info": {
                "distance_to_goal": 5.0,
                "path_length": 0.0,
                "success": 0.0,
                "spl": 0.0,
            },
        }
        env = FakeEnv([transition], collision_metric=False)
        row = self._run(
            env, FakeModel([]), probe.ProbeConfig(episodes=1, max_steps=1)
        ).receipt
        self.assertIsNone(row["collision_count"])
        self.assertEqual(row["collision_source"], "unavailable")

        env = FakeEnv([transition], collision_metric=False)
        with self.assertRaisesRegex(probe.HabitatProtocolError, "collision"):
            self._run(
                env,
                FakeModel([]),
                probe.ProbeConfig(
                    episodes=1, max_steps=1, require_collision_metric=True
                ),
            )

    def test_false_per_step_collision_metric_is_available_zero(self):
        transition = {
            "terminated": True,
            "truncated": True,
            "info": {
                "distance_to_goal": 5.0,
                "path_length": 0.0,
                "success": 0.0,
                "spl": 0.0,
            },
        }
        env = FakeEnv(
            [transition],
            collision_metric=False,
            per_step_collision_metric=True,
        )
        row = self._run(
            env, FakeModel([]), probe.ProbeConfig(episodes=1, max_steps=1)
        ).receipt
        self.assertEqual(row["collision_count"], 0.0)
        self.assertEqual(row["collision_source"], "per_step_metric")

        with self.assertRaisesRegex(probe.HabitatProtocolError, "must be boolean"):
            probe._collision_observation(
                {"collisions": {"is_collision": "false"}}
            )


class ReceiptAndDependencyTest(unittest.TestCase):
    def test_zmq_close_is_local_only(self):
        calls = []

        class FakeSocket:
            def close(self):
                calls.append("socket.close")

        class FakeContext:
            def term(self):
                calls.append("context.term")

        client = object.__new__(probe.ZmqHabitatClient)
        client.connected = True
        client.socket = FakeSocket()
        client.context = FakeContext()

        def remote_command(*_args, **_kwargs):
            calls.append("remote.command")
            raise AssertionError("close must not contact the Habitat service")

        client._command = remote_command
        client.close()

        self.assertFalse(client.connected)
        self.assertIsNone(client.socket)
        self.assertIsNone(client.context)
        self.assertEqual(calls, ["socket.close", "context.term"])

    def test_atomic_json_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nested" / "receipt.json"
            probe.atomic_write_json(path, {"status": "completed", "value": 2})
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"status": "completed", "value": 2},
            )
            self.assertEqual(list(path.parent.glob("*.tmp-*")), [])

    def test_optional_runtime_dependencies_are_not_top_level_imports(self):
        source_path = MODULE_DIR / "lightnav_l0_native_probe.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        imported = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        self.assertTrue({"numpy", "PIL", "zmq", "websockets"}.isdisjoint(imported))

    def test_aggregate_contains_navigation_stop_collision_and_latency(self):
        rows = [
            {
                "success": True,
                "spl": 0.5,
                "final_navigation_error_m": 2.0,
                "geodesic_progress_m": 3.0,
                "path_length_m": 4.0,
                "collision_count": 1.0,
                "model_stop": True,
                "forced_stop": False,
                "successful_stop": True,
                "failed_stop": False,
                "timeout": False,
                "action_distribution": {"FORWARD": 2, "STOP": 1},
            },
            {
                "success": False,
                "spl": 0.0,
                "final_navigation_error_m": 6.0,
                "geodesic_progress_m": -1.0,
                "path_length_m": 5.0,
                "collision_count": None,
                "model_stop": False,
                "forced_stop": True,
                "successful_stop": False,
                "failed_stop": True,
                "timeout": True,
                "action_distribution": {"LEFT": 3, "STOP": 1},
            },
        ]
        summary = probe.aggregate_episodes(
            rows,
            server_latencies_ms=[10.0, 20.0],
            roundtrip_latencies_ms=[15.0, 25.0],
        )
        self.assertEqual(summary["success_count"], 1)
        self.assertEqual(summary["sr"], 0.5)
        self.assertEqual(summary["mean_navigation_error_m"], 4.0)
        self.assertEqual(summary["collision"]["available_episode_count"], 1)
        self.assertEqual(summary["successful_stop_count"], 1)
        self.assertEqual(summary["timeout_count"], 1)
        self.assertEqual(summary["action_distribution"]["STOP"], 2)
        self.assertEqual(summary["server_latency"]["p50_ms"], 15.0)


if __name__ == "__main__":
    unittest.main()
