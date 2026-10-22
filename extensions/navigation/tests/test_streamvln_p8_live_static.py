from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import unittest
from collections.abc import Mapping
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "integrations"
    / "rlinf_v03"
    / "streamvln_p8_live_diagnostic.py"
)
SOURCE = MODULE_PATH.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE, filename=str(MODULE_PATH))


class _ProtocolError(RuntimeError):
    pass


def _load_pure_function(name: str):
    """Load one helper without importing the torch/live-runtime module."""

    function = next(
        node
        for node in TREE.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            function,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {
        "copy": copy,
        "hashlib": hashlib,
        "json": json,
        "math": math,
        "Mapping": Mapping,
        "ProtocolError": _ProtocolError,
        "ACTION_COUNT": 4,
        "NAVIGATE_INTENT": 0,
        "STOP_INTENT": 1,
        "RECOVER_INTENT": 2,
    }
    exec(compile(module, str(MODULE_PATH), "exec"), namespace)
    return namespace[name]


def _class(name: str) -> ast.ClassDef:
    return next(
        node for node in TREE.body if isinstance(node, ast.ClassDef) and node.name == name
    )


def _method(class_name: str, method_name: str) -> ast.FunctionDef:
    owner = _class(class_name)
    return next(
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )


def _returned_dict_keys(function: ast.FunctionDef) -> set[str]:
    keys = set()
    for node in ast.walk(function):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            keys.update(
                key.value
                for key in node.value.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            )
    return keys


class LiveDiagnosticStaticContractTests(unittest.TestCase):
    def test_module_parses_and_exposes_runner_contract(self) -> None:
        classes = {
            node.name for node in TREE.body if isinstance(node, ast.ClassDef)
        }
        functions = {
            node.name for node in TREE.body if isinstance(node, ast.FunctionDef)
        }
        self.assertTrue(
            {
                "CausalTopologyMemory",
                "DecisionTrace",
                "EpisodeTrace",
                "P8TrainingRow",
                "DAggerDecisionTrace",
                "DAggerCollection",
                "DevelopmentEpisode",
                "FrozenClassWeights",
                "LiveP8DiagnosticProviders",
            }.issubset(classes)
        )
        self.assertTrue(
            {
                "validate_train_manifest",
                "stable_seed",
                "fixed_class_weights_from_counts",
                "training_row_dataset_sha256",
                "progress_event_json",
                "deduplicate_state_records",
                "train_residual_fp32",
                "evaluate_training_rows",
                "clone_teacher_session",
                "collect_expert_rows",
                "collect_bounded_dagger_rows",
                "build_live_gate_providers",
            }.issubset(functions)
        )

        for name in ("gate_a_provider", "gate_b_provider", "gate_c_provider"):
            method = _method("LiveP8DiagnosticProviders", name)
            self.assertEqual([arg.arg for arg in method.args.args], ["self"])
            self.assertFalse(method.args.vararg)
            self.assertFalse(method.args.kwarg)

        preload = _method("LiveP8DiagnosticProviders", "preload_env_probe")
        self.assertEqual([arg.arg for arg in preload.args.args], ["self"])
        preload_body = ast.unparse(preload)
        self.assertNotIn("_load_once", preload_body)
        self.assertIn("self.env_factory", preload_body)
        self.assertIn("env.reset()", preload_body)
        self.assertIn("env.close()", preload_body)
        self.assertIn("model_load_count", preload_body)

    def test_train_only_env_construction(self) -> None:
        self.assertNotIn("run_streamvln_ppo_g4", SOURCE)
        self.assertIn("from rlinf.envs import get_env_cls", SOURCE)
        self.assertIn("from omegaconf import OmegaConf", SOURCE)
        assignment = next(
            node
            for node in TREE.body
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            and (
                any(
                    isinstance(target, ast.Name) and target.id == "ALLOWED_SPLIT"
                    for target in getattr(node, "targets", [])
                )
                or (
                    isinstance(getattr(node, "target", None), ast.Name)
                    and node.target.id == "ALLOWED_SPLIT"
                )
            )
        )
        self.assertIsInstance(assignment.value, ast.Constant)
        self.assertEqual(assignment.value.value, "train")

        split_keywords = [
            keyword.value
            for call in ast.walk(TREE)
            if isinstance(call, ast.Call)
            for keyword in call.keywords
            if keyword.arg == "split"
        ]
        self.assertTrue(split_keywords)
        for value in split_keywords:
            allowed = (
                isinstance(value, ast.Name) and value.id == "ALLOWED_SPLIT"
            ) or (isinstance(value, ast.Constant) and value.value == "train")
            self.assertTrue(allowed, ast.unparse(value))

    def test_gate_a_is_live_and_zero_residual_matched(self) -> None:
        for symbol in (
            "OriginalStreamSession",
            "OnlineTopology",
            "decision_residual_callback",
            "frozen_backbone_full_sha256",
            "sample_pre_refactor_release_reference",
            "residual_sha256",
            "base_logits",
            "sampled_tokens",
            "primitive_actions",
            "proposed_prefix",
            "executed_prefix",
            "stop_position",
            "termination",
            "terminal_metrics",
        ):
            self.assertIn(symbol, SOURCE)
        self.assertIn("decision_residual_callback", SOURCE)
        self.assertIn("row.ne(0).any()", SOURCE)
        self.assertIn("pre_refactor_sampler_at_5434ab0", SOURCE)

    def test_training_contract_is_explicit(self) -> None:
        self.assertIn("torch.optim.AdamW", SOURCE)
        self.assertIn("torch.float32", SOURCE)
        self.assertIn("fixed_class_weights_from_counts", SOURCE)
        self.assertIn("deduplicate_state_records", SOURCE)
        self.assertIn("expert_waypoint", SOURCE)
        self.assertIn("commit_token", SOURCE)
        self.assertIn("parameter.requires_grad", SOURCE)
        self.assertIn("value_head.parameters", SOURCE)

    def test_gate_b_payload_and_frozen_schedule(self) -> None:
        method = _method("LiveP8DiagnosticProviders", "gate_b_provider")
        self.assertTrue(
            {
                "token_top1_accuracy",
                "action_top1_accuracy",
                "navigate_intent_recall",
                "stop_intent_recall",
                "stop_token_recall",
                "waypoint_mean_l2_m",
                "sample_replay_ratio_deviation",
                "consecutive_passes",
                "recover_positive_count",
                "recover_recall",
                "trainable_tensors_changed",
                "required_trainable_groups_changed",
                "trainable_group_changes",
                "expected_trainable_parameter_names",
                "optimizer_parameter_names",
                "finite_gradient_names",
                "nonzero_gradient_names",
                "parameter_change_checks",
                "all_expected_trainables_changed",
                "gradient_scope_exact",
                "value_head_frozen_unchanged",
                "hard_gate_checks",
                "hard_gate_passed",
                "undeclared_tensors_unchanged",
                "frozen_backbone_hash_unchanged",
                "all_finite",
                "trace_fields_complete",
                "split_read_counts",
            }.issubset(_returned_dict_keys(method))
        )
        body = ast.unparse(method)
        for key in (
            "warmup_updates",
            "final_updates",
            "metric_interval_updates",
            "dagger_max_decisions_per_episode",
        ):
            self.assertIn(key, body)
        self.assertIn("collect_bounded_dagger_rows", body)

    def test_gate_c_payload_and_matched_arms(self) -> None:
        method = _method("LiveP8DiagnosticProviders", "gate_c_provider")
        self.assertTrue(
            {
                "episode_count",
                "learned_success_count",
                "keep_original_success_count",
                "successful_stop_count",
                "stop_count",
                "primitive_counts",
                "mode_counts",
                "intent_mode_counts",
                "left_turn_count",
                "right_turn_count",
                "blocked_state_count",
                "recover_trigger_count",
                "recover_alignment_miss_count",
                "checkpoint_hash_before",
                "checkpoint_hash_after",
                "split_read_counts",
            }.issubset(_returned_dict_keys(method))
        )
        body = ast.unparse(method)
        self.assertIn("learned=False", body)
        self.assertIn("learned=True", body)
        self.assertIn("train_development", body)
        rollout = ast.unparse(
            _method("LiveP8DiagnosticProviders", "_rollout_development_episode")
        )
        self.assertIn("classify_executed_mode", rollout)
        self.assertIn("feedback.executed_actions", rollout)

    def test_live_rollouts_validate_feedback_and_short_prefix(self) -> None:
        self.assertIn("feedback_from_dict", SOURCE)
        self.assertIn("execute_count=len(actions)", SOURCE)
        self.assertIn("execute_count=len(target_actions)", SOURCE)
        self.assertIn("execute_count=len(sampled_actions)", SOURCE)
        self.assertIn("sampled_actions[:duration]", SOURCE)
        self.assertIn("behavior_prefix = behavior_log_probs[:duration]", SOURCE)
        self.assertIn("hidden_rows[:duration]", SOURCE)
        self.assertIn("@torch.inference_mode()", SOURCE)

    def test_dagger_isolated_teacher_and_recover_contract(self) -> None:
        function = next(
            node
            for node in TREE.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "collect_bounded_dagger_rows"
        )
        body = ast.unparse(function)
        self.assertIn("clone_teacher_session(session)", body)
        self.assertLess(
            body.index("teacher.open_decision"),
            body.index("session.open_decision"),
        )
        self.assertIn("blocked_forward_threshold", body)
        self.assertIn("RECOVER_INTENT", body)
        self.assertIn("backtrack_waypoint", body)
        self.assertIn("replay_ratio", body)

    def test_no_ppo_update_path(self) -> None:
        forbidden_calls = {"ppo_step", "ppo_update", "run_ppo", "PPOTrainer"}
        called = set()
        for call in (node for node in ast.walk(TREE) if isinstance(node, ast.Call)):
            if isinstance(call.func, ast.Name):
                called.add(call.func.id)
            elif isinstance(call.func, ast.Attribute):
                called.add(call.func.attr)
        self.assertTrue(forbidden_calls.isdisjoint(called))


class LiveDiagnosticPureFunctionTests(unittest.TestCase):
    def test_stable_seed_is_deterministic_and_domain_separated(self) -> None:
        stable_seed = _load_pure_function("stable_seed")
        first = stable_seed(2026092811, "gate-a", 2)
        self.assertEqual(first, stable_seed(2026092811, "gate-a", 2))
        self.assertNotEqual(first, stable_seed(2026092811, "gate-a", 3))
        self.assertNotEqual(first, stable_seed(2026092811, "gate-c", 2))
        self.assertGreaterEqual(first, 0)
        self.assertLess(first, 2**63)

    def test_fixed_class_weights_are_deterministic_and_favor_rare_classes(self) -> None:
        weights = _load_pure_function("fixed_class_weights_from_counts")
        counts = [100, 20, 5]
        first = weights(counts)
        self.assertEqual(first, weights(tuple(counts)))
        self.assertEqual(len(first), 3)
        self.assertTrue(all(value > 0.0 for value in first))
        self.assertLess(first[0], first[1])
        self.assertLess(first[1], first[2])
        counts[0] = 1
        self.assertEqual(first, weights([100, 20, 5]))
        with self.assertRaises(_ProtocolError):
            weights([])
        with self.assertRaises(_ProtocolError):
            weights([0, 0, 0])

    def test_teacher_session_clone_isolates_mutable_causal_state(self) -> None:
        clone_teacher_session = _load_pure_function("clone_teacher_session")

        class FakeSession:
            pass

        live = FakeSession()
        live.model = object()
        live.tokenizer = object()
        live.scene = "scene"
        live.instruction = "instruction"
        live.features = {0: object()}
        live.actions = [1, 2]
        live.parts = [object()]
        live.predictor_positions = [3]
        live.value_positions = [4]
        live.sampled_ids = [5]
        live.block_start = 0
        live.memory_prefix = object()

        teacher = clone_teacher_session(live)
        self.assertIsNot(teacher, live)
        self.assertIs(teacher.model, live.model)
        self.assertIs(teacher.tokenizer, live.tokenizer)
        teacher.features[4] = object()
        teacher.actions.append(3)
        teacher.parts.append(object())
        teacher.predictor_positions.append(6)
        teacher.value_positions.append(7)
        teacher.sampled_ids.append(8)
        self.assertEqual(set(live.features), {0})
        self.assertEqual(live.actions, [1, 2])
        self.assertEqual(len(live.parts), 1)
        self.assertEqual(live.predictor_positions, [3])
        self.assertEqual(live.value_positions, [4])
        self.assertEqual(live.sampled_ids, [5])

    def test_gate_b_stop_token_hard_gate_cannot_be_replaced_by_intent(self) -> None:
        helpers = {}
        for name in ("gate_b_hard_checks", "gate_b_metrics_pass"):
            function = next(
                node
                for node in TREE.body
                if isinstance(node, ast.FunctionDef) and node.name == name
            )
            module = ast.Module(
                body=[
                    ast.ImportFrom(
                        module="__future__",
                        names=[ast.alias(name="annotations")],
                        level=0,
                    ),
                    function,
                ],
                type_ignores=[],
            )
            ast.fix_missing_locations(module)
            namespace = {
                "math": math,
                "Mapping": dict,
                **helpers,
            }
            exec(compile(module, str(MODULE_PATH), "exec"), namespace)
            helpers[name] = namespace[name]
        gate_b_metrics_pass = helpers["gate_b_metrics_pass"]
        thresholds = {
            "token_top1_accuracy_min": 0.9,
            "navigate_intent_recall_min": 0.9,
            "stop_intent_recall_min": 0.9,
            "stop_token_recall_min": 0.9,
            "waypoint_mean_l2_m_max": 0.15,
        }
        metrics = {
            "token_top1_accuracy": 0.95,
            "navigate_intent_recall": 0.91,
            "stop_intent_recall": 1.0,
            "stop_token_recall": 0.5,
            "waypoint_mean_l2_m": 0.1,
            "recover_positive_count": 0,
            "recover_recall": "not_applicable",
            "all_finite": True,
            "trace_fields_complete": True,
        }
        self.assertFalse(gate_b_metrics_pass(metrics, thresholds))
        metrics["stop_token_recall"] = 0.9
        self.assertTrue(gate_b_metrics_pass(metrics, thresholds))
        metrics["recover_recall"] = 0.0
        self.assertFalse(gate_b_metrics_pass(metrics, thresholds))
        metrics["recover_positive_count"] = 2
        metrics["recover_recall"] = 0.5
        self.assertTrue(gate_b_metrics_pass(metrics, thresholds))

    def test_dataset_hash_changes_when_any_supervision_label_changes(self) -> None:
        dataset_hash = _load_pure_function("training_row_dataset_sha256")

        class Values:
            def __init__(self, values):
                self.values = values

            def tolist(self):
                return copy.deepcopy(self.values)

        class Row:
            state_hash = "a" * 64
            target_signature = ((1, 1), 0, (0.5, 0.0))
            actions = Values([1, 1, 0, 0])
            executed_mask = Values([True, True, False, False])
            target_intent = 0
            target_waypoint = Values([0.5, 0.0])
            waypoint_mask = True

        baseline = dataset_hash([Row()])
        for field, replacement in (
            ("actions", Values([2, 1, 0, 0])),
            ("executed_mask", Values([True, False, False, False])),
            ("target_intent", 1),
            ("target_waypoint", Values([0.25, 0.0])),
            ("waypoint_mask", False),
        ):
            changed = Row()
            setattr(changed, field, replacement)
            self.assertNotEqual(baseline, dataset_hash([changed]), field)

    def test_progress_event_is_strict_json(self) -> None:
        serialize = _load_pure_function("progress_event_json")
        line = serialize(1, "gate_b", "metric_check_complete", {"score": 0.9})
        self.assertEqual(
            json.loads(line),
            {
                "sequence": 1,
                "stage": "gate_b",
                "event": "metric_check_complete",
                "payload": {"score": 0.9},
            },
        )
        with self.assertRaises(ValueError):
            serialize(2, "gate_b", "bad", {"score": float("nan")})

    def test_gate_c_modes_require_executed_stop_and_aligned_recovery(self) -> None:
        classify = _load_pure_function("classify_executed_mode")
        self.assertEqual(
            classify(
                [0],
                predicted_intent=0,
                actionable_blocked=False,
                recovery_actions=None,
            ),
            1,
        )
        self.assertEqual(
            classify(
                [1, 1, 1, 1],
                predicted_intent=1,
                actionable_blocked=False,
                recovery_actions=None,
            ),
            0,
        )
        self.assertEqual(
            classify(
                [2, 2, 1, 1],
                predicted_intent=2,
                actionable_blocked=True,
                recovery_actions=[2, 2, 1, 1],
            ),
            2,
        )
        self.assertEqual(
            classify(
                [3, 3, 1, 1],
                predicted_intent=2,
                actionable_blocked=True,
                recovery_actions=[2, 2, 1, 1],
            ),
            0,
        )
        with self.assertRaises(_ProtocolError):
            classify(
                [],
                predicted_intent=0,
                actionable_blocked=False,
                recovery_actions=None,
            )


if __name__ == "__main__":
    unittest.main()
