from pathlib import Path
import random
import sys
import unittest


ROOT = Path(__file__).parents[1]
STAGE = ROOT / "integrations" / "rlinf_v03"
sys.path.insert(0, str(STAGE))

from streamvln_p8_official_keep import (  # noqa: E402
    ATTENTION_IMPLEMENTATION,
    APPLY_CHAT_TEMPLATE_ADD_GENERATION_PROMPT,
    CHAT_TEMPLATE_APPLICATION,
    CONJUNCTIONS,
    EMPTY_ASSISTANT_CONTENT,
    FRAME_PREPROCESS_CONTRACT,
    GENERATION_KWARGS,
    IMAGE_TOKEN_INDEX,
    MEMORY_TOKEN_INDEX,
    MODEL_INTERNAL_STATE_CONTRACT,
    PINNED_EVALUATOR_SHA256,
    PINNED_EVALUATOR_SOURCE_RELATIVE_PATH,
    PINNED_MODEL_CODE_SHA256,
    PINNED_MODEL_SOURCE_RELATIVE_PATH,
    PINNED_RELEASE_MANIFEST_SHA256,
    PINNED_SOURCE_IMPORT_ROOT_RELATIVE_PATH,
    PINNED_STREAMVLN_COMMIT,
    MODEL_INPUT_CONTRACT,
    RENDERED_MESSAGE_ROLES,
    SOURCE_MESSAGE_ROLES,
    SYSTEM_MESSAGE,
    TEST_ONLY_CONTRACT_VALIDATION_LEVEL,
    TOKENIZER_PREPROCESS_CONTRACT,
    OfficialKeepQueue,
    OfficialEvalSession,
    official_keep_contract_sha256,
    official_runtime_receipt_template,
    official_history_frame_indices,
    official_prompt,
    official_user_content,
    parse_official_actions,
    parse_with_official_empty_fallback,
    validate_official_runtime_receipt,
)
from streamvln_p8_successor_action import (  # noqa: E402
    FORWARD_ACTION,
    LEFT_ACTION,
    RIGHT_ACTION,
    STOP_ACTION,
    CompiledRoute,
    ParentRecoveryUnavailable,
    RouteDecision,
    RouteName,
    StopBranchRequired,
    SuccessorPlanner,
    WaypointCompilerConfig,
    compile_metric_waypoint,
    compile_route,
    pack_fixed_action_chunk,
)
from streamvln_p8_successor_executor import (  # noqa: E402
    PrimitiveOutcome,
    PURE_SUCCESSOR_EXECUTION_SCHEMA_NAME,
    execute_feedback_aware,
    successor_end_reason,
)
from streamvln_topology import OnlineTopology  # noqa: E402


class StreamVLNP8SuccessorInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.config = WaypointCompilerConfig()
        self.topology = OnlineTopology(max_decisions=32)

    def test_distance_changes_variable_length_navigation(self):
        short = compile_metric_waypoint((0.24, 0.0), config=self.config)
        long = compile_metric_waypoint((1.0, 0.0), config=self.config)
        self.assertEqual(short, (FORWARD_ACTION,))
        self.assertEqual(long, (FORWARD_ACTION,) * 4)
        self.assertLess(len(short), len(long))

    def test_motion_scale_is_fixed_to_habitat_contract(self):
        with self.assertRaises(ValueError):
            WaypointCompilerConfig(forward_step_m=0.5)
        with self.assertRaises(ValueError):
            WaypointCompilerConfig(turn_degrees=30.0)
        with self.assertRaises(ValueError):
            WaypointCompilerConfig(arrival_tolerance_m=0.20)
        incompatible = OnlineTopology(max_decisions=8, forward_step_m=0.5)
        with self.assertRaises(ValueError):
            compile_route(
                RouteDecision(0, RouteName.WAYPOINT, (0.5, 0.0)),
                incompatible,
            )

    def test_bearing_changes_left_and_right_execution(self):
        left = compile_metric_waypoint((0.5, 0.25), config=self.config)
        right = compile_metric_waypoint((0.5, -0.25), config=self.config)
        self.assertEqual(left[:2], (LEFT_ACTION, LEFT_ACTION))
        self.assertEqual(right[:2], (RIGHT_ACTION, RIGHT_ACTION))
        self.assertNotIn(STOP_ACTION, left)
        self.assertNotIn(STOP_ACTION, right)
        self.assertEqual(
            compile_metric_waypoint((-1.0, -0.0), config=self.config),
            (LEFT_ACTION,) * 4,
        )

    def test_near_waypoint_cannot_synthesize_stop(self):
        with self.assertRaises(StopBranchRequired):
            compile_metric_waypoint((0.05, 0.0), config=self.config)
        stop = compile_route(
            RouteDecision(0, RouteName.STOP),
            self.topology,
            config=self.config,
        )
        self.assertEqual(stop.proposed_actions, (STOP_ACTION,))
        self.assertEqual(stop.source, "learned_stop_branch")

    def test_keep_delegates_one_official_action_exactly(self):
        for action in (STOP_ACTION, FORWARD_ACTION, LEFT_ACTION, RIGHT_ACTION):
            plan = compile_route(
                RouteDecision(4, RouteName.KEEP_ORIGINAL, official_action=action),
                self.topology,
            )
            self.assertEqual(plan.proposed_actions, (action,))
            self.assertEqual(plan.executed_route, RouteName.KEEP_ORIGINAL)

    def test_unbound_keep_is_a_valid_policy_decision_but_not_executable(self):
        decision = RouteDecision(4, RouteName.KEEP_ORIGINAL)
        self.assertIsNone(decision.official_action)
        with self.assertRaisesRegex(ValueError, "bound to the official primitive"):
            compile_route(decision, self.topology)

    def test_blocked_waypoint_uses_causal_topology_parent(self):
        self.topology.update(
            [FORWARD_ACTION, FORWARD_ACTION, FORWARD_ACTION],
            [False, False, False],
        )
        parent_target = self.topology.backtrack_waypoint()
        self.assertIsNotNone(parent_target)
        self.topology.update([FORWARD_ACTION, FORWARD_ACTION], [True, True])
        plan = compile_route(
            RouteDecision(7, RouteName.WAYPOINT, (1.0, 0.0)),
            self.topology,
            blocked_threshold=2,
        )
        self.assertEqual(plan.executed_route, RouteName.RECOVER_PARENT)
        self.assertEqual(plan.waypoint_local_m, parent_target)
        self.assertEqual(plan.source, "deterministic_topology_parent")
        self.assertTrue(plan.recovery_override)
        self.assertNotIn(STOP_ACTION, plan.proposed_actions)

    def test_blocked_root_fails_closed_without_inventing_recovery(self):
        self.topology.update([FORWARD_ACTION, FORWARD_ACTION], [True, True])
        with self.assertRaises(ParentRecoveryUnavailable):
            compile_route(
                RouteDecision(1, RouteName.WAYPOINT, (1.0, 0.0)),
                self.topology,
                blocked_threshold=2,
            )
        stop = compile_route(
            RouteDecision(2, RouteName.STOP),
            self.topology,
            blocked_threshold=2,
        )
        self.assertEqual(stop.proposed_actions, (STOP_ACTION,))

    def test_fixed_transport_padding_is_never_executed(self):
        plan = compile_route(
            RouteDecision(0, RouteName.WAYPOINT, (0.24, 0.0)),
            self.topology,
        )
        padded, execute_count = pack_fixed_action_chunk(plan)
        self.assertEqual(padded, (FORWARD_ACTION, STOP_ACTION, STOP_ACTION, STOP_ACTION))
        self.assertEqual(execute_count, 1)

    def test_executor_executes_distance_dependent_prefix_then_replans(self):
        plan = compile_route(
            RouteDecision(2, RouteName.WAYPOINT, (1.0, 0.0)),
            self.topology,
        )
        calls = []

        def step(action, primitive_index):
            calls.append((action, primitive_index))
            return PrimitiveOutcome(action=action, realized_path_m=0.25)

        result = execute_feedback_aware(plan, step)
        self.assertEqual(calls, [(FORWARD_ACTION, index) for index in range(4)])
        self.assertEqual(result.executed_actions, (FORWARD_ACTION,) * 4)
        self.assertEqual(result.end_reason, "plan_complete")
        self.assertTrue(result.replan_required)
        trace = result.to_trace()
        self.assertEqual(trace["schema_name"], PURE_SUCCESSOR_EXECUTION_SCHEMA_NAME)
        self.assertEqual(trace["compiled_actions"], [FORWARD_ACTION] * 4)
        self.assertEqual(trace["executed_actions"], [FORWARD_ACTION] * 4)
        self.assertEqual(trace["waypoint_local_m"], [1.0, 0.0])

    def test_collision_and_blockage_discard_remaining_plan(self):
        plan = compile_route(
            RouteDecision(3, RouteName.WAYPOINT, (1.0, 0.0)),
            self.topology,
        )
        for outcome, expected in (
            (PrimitiveOutcome(FORWARD_ACTION, collision=True), "collision"),
            (PrimitiveOutcome(FORWARD_ACTION, blocked=True), "blocked"),
        ):
            calls = []

            def step(action, primitive_index, value=outcome):
                calls.append((action, primitive_index))
                return value

            result = execute_feedback_aware(plan, step)
            self.assertEqual(len(calls), 1)
            self.assertEqual(result.end_reason, expected)
            self.assertTrue(result.replan_required)

        calls = []

        def collide_second(action, primitive_index):
            calls.append((action, primitive_index))
            return PrimitiveOutcome(action, collision=(primitive_index == 1))

        result = execute_feedback_aware(plan, collide_second)
        self.assertEqual(calls, [(FORWARD_ACTION, 0), (FORWARD_ACTION, 1)])
        self.assertEqual(result.executed_actions, (FORWARD_ACTION, FORWARD_ACTION))
        self.assertEqual(result.end_reason, "collision")

    def test_terminal_precedence_is_shared_and_truncation_wins_over_stop(self):
        reason, replan = successor_end_reason(
            action=STOP_ACTION,
            collision=True,
            blocked=False,
            terminated=False,
            truncated=True,
            plan_complete=True,
        )
        self.assertEqual(reason, "truncated")
        self.assertFalse(replan)

    def test_compiled_route_rejects_transport_coercion(self):
        with self.assertRaises(ValueError):
            CompiledRoute(
                0,
                RouteName.WAYPOINT,
                RouteName.WAYPOINT,
                (0.25, 0.0),
                (1.0,),
                "test",
                False,
                None,
                0.25,
                0.0,
            )

    def test_recovery_parent_stays_latched_after_block_counter_resets(self):
        self.topology.update(
            [FORWARD_ACTION, FORWARD_ACTION, FORWARD_ACTION],
            [False, False, False],
        )
        self.topology.update([FORWARD_ACTION, FORWARD_ACTION], [True, True])
        planner = SuccessorPlanner(blocked_threshold=2)
        decision = RouteDecision(8, RouteName.WAYPOINT, (1.0, 0.0))
        first = planner.plan(decision, self.topology)
        self.assertEqual(first.recovery_parent_index, 0)
        self.topology.update([LEFT_ACTION, FORWARD_ACTION], [False, False])
        self.assertEqual(self.topology.consecutive_blocked_forwards, 0)
        self.assertNotEqual(self.topology.current_node, 0)
        second = planner.plan(decision, self.topology)
        self.assertEqual(second.executed_route, RouteName.RECOVER_PARENT)
        self.assertEqual(second.recovery_parent_index, 0)

    def test_recovery_uses_metric_arrival_not_graph_node_radius(self):
        topology = OnlineTopology(max_decisions=8)
        topology.x = 0.0
        topology.z = 0.30
        topology.current_node = 0
        planner = SuccessorPlanner(recovery_parent_index=0)
        decision = RouteDecision(9, RouteName.WAYPOINT, (0.5, 0.0))
        still_recovering = planner.plan(decision, topology)
        self.assertEqual(still_recovering.executed_route, RouteName.RECOVER_PARENT)
        self.assertEqual(still_recovering.recovery_parent_index, 0)
        topology.z = 0.05
        released = planner.plan(decision, topology)
        self.assertEqual(released.executed_route, RouteName.WAYPOINT)
        self.assertIsNone(released.recovery_parent_index)

    def test_planner_reset_clears_episode_recovery_state(self):
        planner = SuccessorPlanner(recovery_parent_index=3)
        planner.reset_episode()
        self.assertIsNone(planner.recovery_parent_index)

    def test_official_keep_contract_matches_pinned_upstream_semantics(self):
        self.assertEqual(
            PINNED_STREAMVLN_COMMIT,
            "e48f6ff7e9201d93aae003e8f64b04c00cec13bc",
        )
        self.assertEqual(
            PINNED_EVALUATOR_SOURCE_RELATIVE_PATH,
            "streamvln/streamvln_eval.py",
        )
        self.assertEqual(
            PINNED_MODEL_SOURCE_RELATIVE_PATH,
            "streamvln/model/stream_video_vln.py",
        )
        self.assertEqual(PINNED_SOURCE_IMPORT_ROOT_RELATIVE_PATH, "streamvln")
        self.assertEqual(
            PINNED_EVALUATOR_SHA256,
            "4c556b77100ed60cc40aee620947c89428494562402f7d3c5b5120ae12320abb",
        )
        self.assertEqual(
            PINNED_MODEL_CODE_SHA256,
            "e106e16a7de4a8086ed526e55d877506e480d05980ddf1c6909286439acca8dc",
        )
        self.assertEqual(ATTENTION_IMPLEMENTATION, "flash_attention_2")
        self.assertEqual(
            GENERATION_KWARGS,
            {
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": 10000,
                "use_cache": True,
                "return_dict_in_generate": True,
            },
        )
        self.assertEqual(
            official_keep_contract_sha256(),
            "2448c533fe47cc067af999d1fb7cd6ec9efb868af4bb2ebc31a2d2d18b30de87",
        )
        self.assertEqual(TOKENIZER_PREPROCESS_CONTRACT["image_token_index"], -200)
        self.assertEqual(TOKENIZER_PREPROCESS_CONTRACT["memory_token_index"], -300)
        self.assertEqual(IMAGE_TOKEN_INDEX, -200)
        self.assertEqual(MEMORY_TOKEN_INDEX, -300)
        self.assertIn("inputs_embeds", MODEL_INTERNAL_STATE_CONTRACT["model_cache"])
        self.assertIn("filter_depth", FRAME_PREPROCESS_CONTRACT["depth"])
        prompt = official_prompt("walk to the chair", step_id=0)
        self.assertEqual(
            prompt,
            "You are an autonomous navigation assistant. Your task is to walk to the chair "
            "Devise an action sequence to follow the instruction using the four actions: "
            "TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, MOVE FORWARD (↑) by 25 "
            "centimeters, or STOP.",
        )
        self.assertNotIn("<video>", prompt)
        self.assertIn("walk to the chair", prompt)
        self.assertNotIn("<memory>", prompt)
        self.assertIn("<memory>", official_prompt("walk", step_id=1))
        content = official_user_content(
            "walk to the chair",
            step_id=0,
            conjunction=CONJUNCTIONS[0],
        )
        self.assertTrue(content.endswith(" you can see <image>."))
        self.assertEqual(
            official_history_frame_indices(32),
            (0, 4, 8, 12, 16, 20, 24, 28),
        )
        self.assertEqual(official_history_frame_indices(31), ())

    def test_official_regex_order_queue_and_empty_fallback(self):
        decoded = "plan ↑ then ← and STOP; ignore → after stop"
        self.assertEqual(parse_official_actions(decoded), [1, 2, 0, 3])
        self.assertEqual(parse_with_official_empty_fallback("no symbol"), [0])
        queue = OfficialKeepQueue()
        queue.load_decoded_sequence(decoded)
        self.assertEqual([queue.pop_action() for _ in range(4)], [1, 2, 0, 3])
        self.assertTrue(queue.needs_generation)
        with self.assertRaises(RuntimeError):
            queue.pop_action()
        self.assertEqual(parse_official_actions("NONSTOP"), [STOP_ACTION])

    def test_official_state_machine_preserves_variable_queue_and_continuation(self):
        session = OfficialEvalSession(
            env_id=3,
            instruction="walk to the chair",
            conjunction_rng=random.Random(17),
        )
        session.observe("frame-0")
        first = session.build_generation_request()
        self.assertTrue(first.add_system)
        self.assertEqual(first.system_message, SYSTEM_MESSAGE)
        self.assertEqual(first.source_messages[0][0], SOURCE_MESSAGE_ROLES[0])
        self.assertEqual(first.source_messages[1], (SOURCE_MESSAGE_ROLES[1], ""))
        self.assertEqual(first.rendered_messages[0][0], RENDERED_MESSAGE_ROLES[0])
        self.assertEqual(
            first.rendered_messages[1],
            (RENDERED_MESSAGE_ROLES[1], EMPTY_ASSISTANT_CONTENT),
        )
        self.assertEqual(
            first.add_generation_prompt,
            APPLY_CHAT_TEMPLATE_ADD_GENERATION_PROMPT,
        )
        self.assertEqual(first.chat_template_application, CHAT_TEMPLATE_APPLICATION)
        self.assertEqual(first.frame_steps, (0,))
        self.assertIsNone(first.previous_output_ref)
        session.accept_generation(
            decoded_sequence="↑ ←",
            output_ref="output-0",
            past_key_values_ref="past-0",
        )
        self.assertEqual(session.pop_action(), FORWARD_ACTION)
        with self.assertRaises(RuntimeError):
            session.pop_action()
        session.after_env_step()
        session.observe("frame-1")
        self.assertFalse(session.needs_generation)
        self.assertEqual(session.pop_action(), LEFT_ACTION)
        session.after_env_step()
        session.observe("frame-2")
        continuation = session.build_generation_request()
        self.assertFalse(continuation.add_system)
        self.assertEqual(continuation.previous_output_ref, "output-0")
        self.assertEqual(continuation.past_key_values_ref, "past-0")
        self.assertEqual(continuation.frame_steps, (2,))
        self.assertEqual(continuation.time_ids, (0, 1, 2))
        self.assertEqual(continuation.task_type, (0,))
        self.assertEqual(continuation.to_dict()["model_input_contract"], MODEL_INPUT_CONTRACT)
        self.assertNotIn("walk to the chair", continuation.user_content)

    def test_official_stop_latches_terminal_and_discards_queue_tail(self):
        session = OfficialEvalSession(
            env_id=1,
            instruction="walk",
            conjunction_rng=random.Random(1),
        )
        session.observe("frame-0")
        session.build_generation_request()
        session.accept_generation(
            decoded_sequence="STOP →",
            output_ref="output",
            past_key_values_ref="past",
        )
        self.assertEqual(session.pop_action(), STOP_ACTION)
        event = session.after_env_step(terminated=True)
        self.assertTrue(event["terminal"])
        self.assertEqual(event["terminal_reason"], "stop")
        with self.assertRaises(RuntimeError):
            session.observe("frame-1")
        with self.assertRaises(RuntimeError):
            session.pop_action()

    def test_official_session_cannot_advance_without_one_popped_action(self):
        session = OfficialEvalSession(
            env_id=1,
            instruction="walk",
            conjunction_rng=random.Random(1),
        )
        session.observe("frame-0")
        session.build_generation_request()
        session.accept_generation(
            decoded_sequence="↑",
            output_ref="output",
            past_key_values_ref="past",
        )
        with self.assertRaises(RuntimeError):
            session.after_env_step()

    def test_official_state_machine_replays_history_and_reset_at_step_32(self):
        session = OfficialEvalSession(
            env_id=0,
            instruction="go ahead",
            conjunction_rng=random.Random(5),
        )
        for step in range(32):
            session.observe(f"frame-{step}")
            if step == 0:
                session.build_generation_request()
                session.accept_generation(
                    decoded_sequence="↑" * 32,
                    output_ref="output-0",
                    past_key_values_ref="past-0",
                )
            session.pop_action()
            event = session.after_env_step()
        self.assertTrue(event["cache_reset"])
        self.assertEqual(session.snapshot()["reset_steps"], [32])
        self.assertFalse(session.snapshot()["queue_crossed_reset"])
        session.observe("frame-32")
        request = session.build_generation_request()
        self.assertEqual(request.frame_steps, (0, 4, 8, 12, 16, 20, 24, 28, 32))
        self.assertEqual(request.time_ids, (32,))
        self.assertTrue(request.add_system)
        self.assertIn("<memory>", request.user_content)

    def test_official_terminal_step_32_still_applies_upstream_cache_reset(self):
        session = OfficialEvalSession(
            env_id=0,
            instruction="go ahead",
            conjunction_rng=random.Random(9),
        )
        for step in range(32):
            session.observe(f"frame-{step}")
            if step == 0:
                session.build_generation_request()
                session.accept_generation(
                    decoded_sequence="↑" * 31 + "STOP",
                    output_ref="output-0",
                    past_key_values_ref="past-0",
                )
            action = session.pop_action()
            event = session.after_env_step(terminated=action == STOP_ACTION)
        self.assertTrue(event["terminal"])
        self.assertEqual(event["terminal_reason"], "stop")
        self.assertTrue(event["cache_reset"])
        self.assertEqual(session.snapshot()["reset_steps"], [32])

    def test_official_runtime_receipt_rejects_local_categorical_mode(self):
        receipt = official_runtime_receipt_template()
        validate_official_runtime_receipt(
            receipt,
            validation_level=TEST_ONLY_CONTRACT_VALIDATION_LEVEL,
        )
        receipt["attention_implementation"] = "sdpa"
        with self.assertRaises(ValueError):
            validate_official_runtime_receipt(
                receipt,
                validation_level=TEST_ONLY_CONTRACT_VALIDATION_LEVEL,
            )
        receipt["attention_implementation"] = "flash_attention_2"
        receipt["allowed_action_token_mask"] = True
        with self.assertRaises(ValueError):
            validate_official_runtime_receipt(
                receipt,
                validation_level=TEST_ONLY_CONTRACT_VALIDATION_LEVEL,
            )

    def test_runtime_receipt_template_is_deeply_isolated_and_unverified(self):
        first = official_runtime_receipt_template()
        first["generation_kwargs"]["do_sample"] = True
        first["artifact_ids"]["checkpoint"] = "mutated"
        first["source_provenance"]["commit"] = "0" * 40

        second = official_runtime_receipt_template()
        self.assertFalse(second["generation_kwargs"]["do_sample"])
        self.assertNotEqual(second["artifact_ids"]["checkpoint"], "mutated")
        self.assertNotEqual(second["source_provenance"]["commit"], "0" * 40)
        self.assertEqual(second["verification_scope"], "expected_contract_only")
        self.assertIsNone(second["observed_runtime"])
        self.assertEqual(
            PINNED_RELEASE_MANIFEST_SHA256,
            "09a344903944942210a342f5defe8708523ecd74be98ec6be9b896334da471ac",
        )
        self.assertEqual(
            second["pinned_release_manifest_sha256"],
            PINNED_RELEASE_MANIFEST_SHA256,
        )
        validate_official_runtime_receipt(
            second,
            validation_level=TEST_ONLY_CONTRACT_VALIDATION_LEVEL,
        )
        with self.assertRaisesRegex(
            ValueError, "production runtime requires full-content release verification"
        ):
            validate_official_runtime_receipt(second)

        candidate = official_runtime_receipt_template()
        candidate["verification_scope"] = "candidate_full_content"
        candidate["observed_runtime"] = {
            "release_manifest_sha256": "a" * 64
        }
        with self.assertRaisesRegex(
            ValueError, "production runtime requires full-content release verification"
        ):
            validate_official_runtime_receipt(candidate)
        with self.assertRaisesRegex(
            ValueError, "requires expected-only scope"
        ):
            validate_official_runtime_receipt(
                candidate,
                validation_level=TEST_ONLY_CONTRACT_VALIDATION_LEVEL,
            )

    def test_keep_policy_branch_is_recovered_but_matched_control_can_stay_exact(self):
        topology = OnlineTopology(max_decisions=8)
        topology.update(
            [FORWARD_ACTION, FORWARD_ACTION, FORWARD_ACTION],
            [False, False, False],
        )
        topology.update([FORWARD_ACTION, FORWARD_ACTION], [True, True])
        decision = RouteDecision(
            10,
            RouteName.KEEP_ORIGINAL,
            official_action=FORWARD_ACTION,
        )
        shielded = SuccessorPlanner(blocked_threshold=2).plan(decision, topology)
        self.assertEqual(shielded.executed_route, RouteName.RECOVER_PARENT)
        exact_control = SuccessorPlanner(
            blocked_threshold=2,
            recovery_enabled=False,
        ).plan(decision, topology)
        self.assertEqual(exact_control.executed_route, RouteName.KEEP_ORIGINAL)
        self.assertEqual(exact_control.proposed_actions, (FORWARD_ACTION,))

    def test_official_keep_stop_outranks_recovery_shield(self):
        topology = OnlineTopology(max_decisions=8)
        topology.update(
            [FORWARD_ACTION, FORWARD_ACTION, FORWARD_ACTION],
            [False, False, False],
        )
        topology.update([FORWARD_ACTION, FORWARD_ACTION], [True, True])
        plan = SuccessorPlanner(blocked_threshold=2).plan(
            RouteDecision(
                11,
                RouteName.KEEP_ORIGINAL,
                official_action=STOP_ACTION,
            ),
            topology,
        )
        self.assertEqual(plan.executed_route, RouteName.KEEP_ORIGINAL)
        self.assertEqual(plan.proposed_actions, (STOP_ACTION,))
        self.assertFalse(plan.recovery_override)

if __name__ == "__main__":
    unittest.main()
