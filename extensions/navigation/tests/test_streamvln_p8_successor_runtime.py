"""Model-free tests for the P8 successor production runner bridge."""

from pathlib import Path
import hashlib
import random
import struct
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).parents[1]
STAGE = ROOT / "integrations" / "rlinf_v03"
sys.path.insert(0, str(STAGE))

import streamvln_p8_successor_runtime as runtime_module  # noqa: E402

from streamvln_executor_contract import chunk_feedback, primitive_execution  # noqa: E402
from streamvln_p8_official_keep import (  # noqa: E402
    APPLY_CHAT_TEMPLATE_ADD_GENERATION_PROMPT,
    ATTENTION_IMPLEMENTATION,
    CHAT_TEMPLATE_APPLICATION,
    DECODE_KWARGS,
    EMPTY_ASSISTANT_CONTENT,
    GENERATION_KWARGS,
    MODEL_DTYPE,
    MODEL_INPUT_CONTRACT,
    PINNED_STREAMVLN_COMMIT,
    RENDERED_MESSAGE_ROLES,
    SOURCE_MESSAGE_ROLES,
    SYSTEM_MESSAGE,
    TEST_ONLY_CONTRACT_VALIDATION_LEVEL,
    OfficialEvalSession,
    official_keep_contract_sha256,
    official_runtime_receipt_template,
    validate_official_runtime_receipt,
)
from streamvln_p8_official_frame import (  # noqa: E402
    OFFICIAL_RGB_HEIGHT,
    OFFICIAL_RGB_WIDTH,
    OfficialFrame,
)
from streamvln_p8_successor_action import (  # noqa: E402
    FORWARD_ACTION,
    LEFT_ACTION,
    RIGHT_ACTION,
    STOP_ACTION,
    RouteDecision,
    RouteName,
    SuccessorPlanner,
)
from streamvln_p8_successor_runtime import (  # noqa: E402
    BRIDGE_SCHEMA_NAME,
    OfficialGenerationResult,
    OfficialKeepDriver,
    SuccessorRunnerBridge,
)
from streamvln_topology import OnlineTopology  # noqa: E402


class FlagBatch:
    def __init__(self, value: bool):
        self.value = bool(value)

    def any(self):
        return self.value


class FakeSuccessorEnv:
    def __init__(self, outcomes):
        self.successor_planner = SuccessorPlanner()
        self.outcomes = list(outcomes)
        self.calls = []
        self.path_length = 0.0
        self.env_id = 7
        self.identity = {
            "instruction_text": "walk to the chair",
            "instruction_sha256": hashlib.sha256(
                b"walk to the chair"
            ).hexdigest(),
            "episode_id": "fake-episode",
            "scene_id": "fake-scene",
        }
        self.reset_count = 0
        self._episode_serial = -1
        self._physical_step = 0
        self.frame_step_offset = 0

    def reset(self):
        self.successor_planner.reset_episode()
        self.path_length = 0.0
        self.reset_count += 1
        self._episode_serial += 1
        self._physical_step = 0
        return f"obs-reset-{self.reset_count}", {
            "episode_id": self.identity["episode_id"],
            "scene_id": self.identity["scene_id"],
        }

    def chunk_step(self, payload, execute_count=None):
        if (
            not isinstance(payload, list)
            or len(payload) != 1
            or len(payload[0]) != 4
            or any(not isinstance(item, list) or len(item) != 1 for item in payload[0])
            or execute_count != 1
        ):
            raise AssertionError("bridge did not use one-primitive successor transport")
        action = int(payload[0][0][0])
        outcome = self.outcomes.pop(0)
        previous = self.path_length
        self.path_length = float(outcome.get("path_length", previous))
        row = primitive_execution(
            primitive_index=0,
            action=action,
            previous_path_length_m=previous,
            current_path_length_m=self.path_length,
            collision=bool(outcome.get("collision", False)),
            environment_reward=float(outcome.get("reward", 0.0)),
            terminated=bool(outcome.get("terminated", False)),
            truncated=bool(outcome.get("truncated", False)),
        )
        self.calls.append(action)
        self._physical_step += 1
        feedback = chunk_feedback((action,), (row,)).to_dict()
        index = len(self.calls)
        return (
            [f"obs-{index}"],
            None,
            FlagBatch(row.terminated),
            FlagBatch(row.truncated),
            [{"executor_feedback": feedback}],
        )

    def observe_official_frame(self):
        return fake_official_frame(
            env_id=self.env_id,
            episode_serial=self._episode_serial,
            physical_step=self._physical_step + self.frame_step_offset,
            identity=self.identity,
        )


RGB_BYTES = bytes(OFFICIAL_RGB_HEIGHT * OFFICIAL_RGB_WIDTH * 3)
DEPTH_BYTES = struct.pack("<f", 0.5) * (OFFICIAL_RGB_HEIGHT * OFFICIAL_RGB_WIDTH)
POSE_BYTES = struct.pack(
    "<16d",
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
)
INTRINSICS_BYTES = struct.pack(
    "<16d",
    386, 0, 319.5, 0,
    0, 386, 239.5, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
)


def fake_official_frame(*, env_id, episode_serial, physical_step, identity):
    return OfficialFrame(
        env_id=env_id,
        episode_serial=episode_serial,
        episode_generation=f"{episode_serial:032x}",
        physical_step=physical_step,
        episode_id=identity["episode_id"],
        scene_id=identity["scene_id"],
        instruction_sha256=identity["instruction_sha256"],
        rgb_u8=RGB_BYTES,
        depth_normalized_f32le=DEPTH_BYTES,
        pose_f64le=POSE_BYTES,
        intrinsics_f64le=INTRINSICS_BYTES,
        rgb_sha256=hashlib.sha256(RGB_BYTES).hexdigest(),
        depth_sha256=hashlib.sha256(DEPTH_BYTES).hexdigest(),
        pose_sha256=hashlib.sha256(POSE_BYTES).hexdigest(),
        intrinsics_sha256=hashlib.sha256(INTRINSICS_BYTES).hexdigest(),
    )


def official_receipt() -> dict:
    return official_runtime_receipt_template()


def make_driver(
    decoded_sequences,
    *,
    env_id=7,
    instruction="walk to the chair",
    reset_events=None,
    episode_serial=0,
    episode_identity=None,
    generation_calls=None,
):
    sequences = list(decoded_sequences)
    reset_events = [] if reset_events is None else reset_events

    if episode_identity is None:
        episode_identity = {
            "instruction_text": instruction,
            "episode_id": "fake-episode",
            "scene_id": "fake-scene",
            "instruction_sha256": hashlib.sha256(instruction.encode()).hexdigest(),
        }

    def generate(request, frames):
        if not isinstance(frames, tuple) or any(
            not isinstance(frame, OfficialFrame) for frame in frames
        ):
            raise AssertionError("generation did not receive private OfficialFrame objects")
        if generation_calls is not None:
            generation_calls.append((request, frames))
        decoded = sequences.pop(0)
        return OfficialGenerationResult(
            decoded_sequence=decoded,
            output_ref=f"output-{request.physical_step}",
            past_key_values_ref=f"past-{request.physical_step}",
        )

    def validate_test_contract(receipt):
        validate_official_runtime_receipt(
            receipt,
            validation_level=TEST_ONLY_CONTRACT_VALIDATION_LEVEL,
        )

    with mock.patch.object(
        runtime_module,
        "validate_official_runtime_receipt",
        side_effect=validate_test_contract,
    ):
        return OfficialKeepDriver(
            session=OfficialEvalSession(
                env_id=env_id,
                instruction=instruction,
                conjunction_rng=random.Random(3),
            ),
            runtime_receipt=official_receipt(),
            generation_adapter=generate,
            cache_reset_adapter=lambda event: reset_events.append(dict(event)),
            episode_serial=episode_serial,
            episode_identity=episode_identity,
        )


def make_bridge(env, decoded_sequences, *, exact_control=False, reset_events=None):
    bridge = SuccessorRunnerBridge(
        env=env,
        topology_factory=lambda: OnlineTopology(max_decisions=64),
        official_driver_factory=lambda env_id, identity, episode_serial: make_driver(
            decoded_sequences,
            env_id=env_id,
            instruction=identity["instruction_text"],
            reset_events=reset_events,
            episode_serial=episode_serial,
            episode_identity=identity,
        ),
        exact_control=exact_control,
    )
    bridge.reset_episode()
    return bridge


def live_decision(
    bridge,
    decision_index,
    route,
    waypoint_local_m=(0.0, 0.0),
    *,
    official_action=None,
    episode_token=None,
    physical_step=None,
):
    return RouteDecision(
        decision_index=decision_index,
        route=route,
        waypoint_local_m=waypoint_local_m,
        official_action=official_action,
        episode_token=bridge.episode_token if episode_token is None else episode_token,
        physical_step=(
            bridge.official_driver.session.physical_step
            if physical_step is None
            else physical_step
        ),
    )


class StreamVLNP8SuccessorRuntimeTests(unittest.TestCase):
    def test_prepared_decision_context_is_single_consumption_and_fail_closed(self):
        env = FakeSuccessorEnv(
            [{"path_length": 0.25}, {"path_length": 0.50}]
        )
        original_observe = env.observe_official_frame
        env.observe_official_frame = mock.Mock(wraps=original_observe)
        bridge = make_bridge(env, ["↑ ↑ ↑ ↑"])
        bridge.require_prepared_decision_context = True
        seen = []
        prepared = bridge.prepare_decision_context(
            lambda frame, action: seen.append((frame, action)) or {"feature": "f"}
        )
        self.assertEqual(env.observe_official_frame.call_count, 1)
        self.assertIs(seen[0][0], bridge.official_driver.frames[0])
        self.assertEqual(prepared.policy_context, {"feature": "f"})
        with self.assertRaisesRegex(RuntimeError, "already prepared"):
            bridge.prepare_decision_context()
        result = bridge.execute_decision(
            live_decision(bridge, 0, RouteName.WAYPOINT, (0.50, 0.0))
        )
        self.assertEqual(result.executed_actions, (FORWARD_ACTION, FORWARD_ACTION))
        self.assertEqual(env.observe_official_frame.call_count, 2)

        next_env = FakeSuccessorEnv([])
        next_bridge = make_bridge(next_env, ["↑"])
        next_bridge.require_prepared_decision_context = True
        with self.assertRaisesRegex(RuntimeError, "must be prepared"):
            next_bridge.execute_decision(
                live_decision(next_bridge, 0, RouteName.WAYPOINT, (0.25, 0.0))
            )

    def test_preparation_builder_failure_poisoned_until_reset(self):
        bridge = make_bridge(FakeSuccessorEnv([]), ["↑"])
        with self.assertRaisesRegex(ValueError, "feature failed"):
            bridge.prepare_decision_context(
                lambda _frame, _action: (_ for _ in ()).throw(
                    ValueError("feature failed")
                )
            )
        with self.assertRaisesRegex(RuntimeError, "poisoned"):
            bridge.prepare_decision_context()
        bridge.reset_episode()
        prepared = bridge.prepare_decision_context()
        self.assertEqual(prepared.physical_step, 0)

    def test_exact_control_executes_one_official_action(self):
        env = FakeSuccessorEnv([{"path_length": 0.25}])
        generation_calls = []
        bridge = SuccessorRunnerBridge(
            env=env,
            topology_factory=lambda: OnlineTopology(max_decisions=64),
            official_driver_factory=lambda env_id, identity, episode_serial: make_driver(
                ["↑ →"],
                env_id=env_id,
                instruction=identity["instruction_text"],
                episode_serial=episode_serial,
                episode_identity=identity,
                generation_calls=generation_calls,
            ),
            exact_control=True,
        )
        bridge.reset_episode()
        result = bridge.execute_decision(
            live_decision(bridge, 0, RouteName.KEEP_ORIGINAL)
        )
        self.assertEqual(result.official_shadow_actions, (FORWARD_ACTION,))
        self.assertEqual(result.executed_actions, (FORWARD_ACTION,))
        self.assertEqual(env.calls, [FORWARD_ACTION])
        self.assertEqual(result.to_trace()["schema_name"], BRIDGE_SCHEMA_NAME)
        self.assertEqual(len(generation_calls), 1)
        self.assertEqual(
            tuple(frame.physical_step for frame in generation_calls[0][1]), (0,)
        )
        self.assertEqual(result.official_frame_metadata[0]["physical_step"], 0)
        self.assertNotIn("rgb_u8", result.to_trace()["official_frame_metadata"][0])

    def test_waypoint_prefix_consumes_one_shadow_action_per_physical_step(self):
        topology = OnlineTopology(max_decisions=8)
        env = FakeSuccessorEnv(
            [{"path_length": 0.25}, {"path_length": 0.50}]
        )
        bridge = SuccessorRunnerBridge(
            env=env,
            topology_factory=lambda: topology,
            official_driver_factory=lambda env_id, identity, episode_serial: make_driver(
                ["← → ↑"],
                env_id=env_id,
                instruction=identity["instruction_text"],
                episode_serial=episode_serial,
                episode_identity=identity,
            ),
        )
        bridge.reset_episode()
        result = bridge.execute_decision(
            live_decision(bridge, 0, RouteName.WAYPOINT, (0.50, 0.0))
        )
        self.assertEqual(result.official_shadow_actions, (LEFT_ACTION, RIGHT_ACTION))
        self.assertEqual(result.executed_actions, (FORWARD_ACTION, FORWARD_ACTION))
        self.assertEqual(env.calls, [FORWARD_ACTION, FORWARD_ACTION])
        self.assertEqual(topology.decision_count, 1)
        self.assertEqual(topology.primitive_count, 2)
        self.assertEqual(result.end_reason, "plan_complete")

    def test_collision_discards_waypoint_tail(self):
        env = FakeSuccessorEnv([{"path_length": 0.0, "collision": True}])
        bridge = make_bridge(env, ["→ → → →"])
        result = bridge.execute_decision(
            live_decision(bridge, 0, RouteName.WAYPOINT, (1.0, 0.0))
        )
        self.assertEqual(result.executed_actions, (FORWARD_ACTION,))
        self.assertEqual(result.end_reason, "collision")
        self.assertTrue(result.replan_required)

    def test_official_shadow_stop_interrupts_learned_waypoint(self):
        env = FakeSuccessorEnv([{"path_length": 0.0, "terminated": True}])
        bridge = make_bridge(env, ["STOP"])
        result = bridge.execute_decision(
            live_decision(bridge, 0, RouteName.WAYPOINT, (1.0, 0.0))
        )
        self.assertEqual(result.plan["proposed_actions"], [FORWARD_ACTION] * 4)
        self.assertEqual(result.executed_actions, (STOP_ACTION,))
        self.assertTrue(result.official_stop_override)
        self.assertEqual(result.end_reason, "official_stop_override")

    def test_exact_control_rejects_non_keep_route(self):
        bridge = make_bridge(FakeSuccessorEnv([]), ["↑"], exact_control=True)
        with self.assertRaises(ValueError):
            bridge.execute_decision(
                live_decision(bridge, 0, RouteName.WAYPOINT, (0.25, 0.0))
            )

    def test_decision_indices_must_be_contiguous(self):
        bridge = make_bridge(FakeSuccessorEnv([]), ["↑"])
        with self.assertRaises(ValueError):
            bridge.execute_decision(
                live_decision(bridge, 2, RouteName.WAYPOINT, (0.25, 0.0))
            )

    def test_decision_physical_step_must_match_live_session(self):
        bridge = make_bridge(FakeSuccessorEnv([]), ["↑"])
        with self.assertRaisesRegex(ValueError, "physical step"):
            bridge.execute_decision(
                live_decision(
                    bridge,
                    0,
                    RouteName.WAYPOINT,
                    (0.25, 0.0),
                    physical_step=1,
                )
            )

    def test_private_official_frame_step_must_match_live_session(self):
        env = FakeSuccessorEnv([{"path_length": 0.25}])
        env.frame_step_offset = 1
        bridge = make_bridge(env, ["↑"])
        with self.assertRaisesRegex(RuntimeError, "live episode and physical step"):
            bridge.execute_decision(
                live_decision(bridge, 0, RouteName.KEEP_ORIGINAL)
            )

    def test_policy_cannot_inject_official_keep_action(self):
        bridge = make_bridge(FakeSuccessorEnv([]), ["↑"])
        with self.assertRaisesRegex(ValueError, "must not inject"):
            bridge.execute_decision(
                live_decision(
                    bridge,
                    0,
                    RouteName.KEEP_ORIGINAL,
                    official_action=RIGHT_ACTION,
                )
            )

    def test_stop_then_reset_creates_fresh_episode_state(self):
        env = FakeSuccessorEnv(
            [
                {"path_length": 0.0, "terminated": True},
                {"path_length": 0.25},
            ]
        )
        episode_sequences = iter((["STOP"], ["↑"]))
        drivers = []
        topologies = []
        reset_events = []

        def driver_factory(env_id, identity, episode_serial):
            driver = make_driver(
                next(episode_sequences),
                env_id=env_id,
                instruction=identity["instruction_text"],
                reset_events=reset_events,
                episode_serial=episode_serial,
                episode_identity=identity,
            )
            drivers.append(driver)
            return driver

        def topology_factory():
            topology = OnlineTopology(max_decisions=8)
            topologies.append(topology)
            return topology

        bridge = SuccessorRunnerBridge(
            env=env,
            topology_factory=topology_factory,
            official_driver_factory=driver_factory,
        )
        first_observation, _ = bridge.reset_episode()
        first_token = bridge.episode_token
        first_exact_planner = bridge.exact_planner
        first = bridge.execute_decision(
            live_decision(bridge, 0, RouteName.KEEP_ORIGINAL)
        )
        self.assertEqual(first_observation, "obs-reset-1")
        self.assertEqual(first.executed_actions, (STOP_ACTION,))
        self.assertTrue(drivers[0].session.terminal)
        with self.assertRaises(RuntimeError):
            bridge.execute_decision(
                live_decision(bridge, 1, RouteName.KEEP_ORIGINAL)
            )

        second_observation, _ = bridge.reset_episode()
        self.assertEqual(second_observation, "obs-reset-2")
        self.assertEqual(env.reset_count, 2)
        self.assertIsNot(drivers[0], drivers[1])
        self.assertIsNot(drivers[0].session, drivers[1].session)
        self.assertIsNot(topologies[0], topologies[1])
        self.assertIsNot(first_exact_planner, bridge.exact_planner)
        self.assertNotEqual(first_token, bridge.episode_token)
        self.assertEqual(drivers[1].session.snapshot()["physical_step"], 0)
        self.assertFalse(drivers[1].session.snapshot()["terminal"])
        self.assertEqual(topologies[1].decision_count, 0)
        self.assertEqual(topologies[1].primitive_count, 0)
        self.assertEqual(
            [event["event"] for event in reset_events],
            ["episode_reset", "episode_reset"],
        )

        with self.assertRaisesRegex(ValueError, "episode token"):
            bridge.execute_decision(
                RouteDecision(
                    0,
                    RouteName.KEEP_ORIGINAL,
                    episode_token=first_token,
                    physical_step=0,
                )
            )

        second = bridge.execute_decision(
            live_decision(bridge, 0, RouteName.KEEP_ORIGINAL)
        )
        self.assertEqual(second.executed_actions, (FORWARD_ACTION,))
        self.assertEqual(topologies[1].decision_count, 1)
        self.assertEqual(topologies[1].primitive_count, 1)

    def test_live_cache_reset_hook_fires_at_nonterminal_step_32(self):
        reset_events = []
        generation_calls = []
        outcomes = [
            {"path_length": 0.25 * (index + 1)}
            for index in range(33)
        ]
        env = FakeSuccessorEnv(outcomes)
        bridge = SuccessorRunnerBridge(
            env=env,
            topology_factory=lambda: OnlineTopology(max_decisions=64),
            official_driver_factory=lambda env_id, identity, episode_serial: make_driver(
                [" ".join(["↑"] * 32), "↑"],
                env_id=env_id,
                instruction=identity["instruction_text"],
                reset_events=reset_events,
                episode_serial=episode_serial,
                episode_identity=identity,
                generation_calls=generation_calls,
            ),
            exact_control=True,
        )
        bridge.reset_episode()
        final_result = None
        boundary_result = None
        for index in range(33):
            final_result = bridge.execute_decision(
                live_decision(bridge, index, RouteName.KEEP_ORIGINAL)
            )
            if index == 31:
                boundary_result = final_result
        self.assertIsNotNone(final_result)
        self.assertIsNotNone(boundary_result)
        self.assertEqual(
            [event["event"] for event in reset_events],
            ["episode_reset", "cache_reset"],
        )
        self.assertEqual(reset_events[-1]["next_step"], 32)
        self.assertFalse(reset_events[-1]["terminal"])
        self.assertTrue(boundary_result.official_step_events[-1]["cache_reset"])
        self.assertTrue(
            boundary_result.to_trace()["official_step_events"][-1]["cache_reset"]
        )
        self.assertEqual(len(generation_calls), 2)
        expected_history = (0, 4, 8, 12, 16, 20, 24, 28, 32)
        self.assertEqual(generation_calls[1][0].frame_steps, expected_history)
        self.assertEqual(
            tuple(frame.physical_step for frame in generation_calls[1][1]),
            expected_history,
        )

    def test_live_cache_reset_hook_fires_on_terminal_step_32(self):
        reset_events = []
        outcomes = [
            {"path_length": 0.25 * (index + 1)}
            for index in range(31)
        ] + [{"path_length": 7.75, "terminated": True}]
        bridge = make_bridge(
            FakeSuccessorEnv(outcomes),
            [" ".join(["↑"] * 31 + ["STOP"])],
            exact_control=True,
            reset_events=reset_events,
        )
        final_result = None
        for index in range(32):
            final_result = bridge.execute_decision(
                live_decision(bridge, index, RouteName.KEEP_ORIGINAL)
            )
        self.assertIsNotNone(final_result)
        self.assertTrue(final_result.terminated)
        self.assertEqual(final_result.end_reason, "stop")
        self.assertEqual(
            [event["event"] for event in reset_events],
            ["episode_reset", "cache_reset"],
        )
        self.assertEqual(reset_events[-1]["next_step"], 32)
        self.assertTrue(reset_events[-1]["terminal"])

    def test_queue_crossing_cache_reset_fails_before_undefined_generation(self):
        reset_events = []
        driver = make_driver(
            [" ".join(["↑"] * 33)],
            reset_events=reset_events,
        )
        for step in range(33):
            action = driver.prepare_shadow_action(
                fake_official_frame(
                    env_id=7,
                    episode_serial=0,
                    physical_step=step,
                    identity=driver.episode_identity,
                )
            )
            self.assertEqual(action, FORWARD_ACTION)
            driver.acknowledge_physical_step(
                terminated=False,
                truncated=False,
            )
        self.assertTrue(driver.session.queue_crossed_reset)
        with self.assertRaisesRegex(RuntimeError, "crossed a 32-step cache reset"):
            driver.prepare_shadow_action(
                fake_official_frame(
                    env_id=7,
                    episode_serial=0,
                    physical_step=33,
                    identity=driver.episode_identity,
                )
            )

    def test_successor_factory_rejects_nontrain_before_optional_imports(self):
        from streamvln_p8_successor_runtime import make_successor_env

        for split in ("val_seen", "val_unseen", "", None):
            with self.subTest(split=split):
                with self.assertRaisesRegex(ValueError, "train-only"):
                    make_successor_env({}, split=split, episode_index=0, max_steps=64)


if __name__ == "__main__":
    unittest.main()
