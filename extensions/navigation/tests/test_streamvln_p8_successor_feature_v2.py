import importlib.util
import inspect
import math
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).parents[1]
MODULE_DIR = ROOT / "integrations" / "rlinf_v03"
sys.path.insert(0, str(MODULE_DIR))


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "Torch unavailable locally")
class SuccessorFeatureV2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        import streamvln_p8_successor_policy as policy

        cls.torch = torch
        cls.policy = policy

    def _hidden(self):
        torch = self.torch
        return torch.linspace(-1.0, 1.0, self.policy.SUCCESSOR_FEATURE_V2_CONTEXT_DIM)

    def _depth(self):
        return self.torch.tensor(
            [
                0.20, 0.40, 1.00,
                0.00, 0.00, 0.00,
                0.30, 0.60, 0.75,
                0.10, 0.25, 0.50,
                1.00, 2.00, 0.25,
            ],
            dtype=self.torch.float32,
        )

    def _compose_kwargs(self):
        return {
            "contextual_hidden": self._hidden(),
            "depth_summary": self._depth(),
            "official_shadow_action": 2,
            "physical_step": 11,
            "context_physical_step": 11,
        }

    def test_v1_composer_and_context_version_remain_unchanged(self):
        torch = self.torch
        module = self.policy
        self.assertEqual(module.SUCCESSOR_POLICY_CONTEXT_VERSION, 4)
        composed = module.compose_successor_features(
            frozen_feature=torch.tensor([0.25, -0.5, 0.75, 1.0]),
            official_shadow_action=2,
            physical_step=7,
            feature_physical_step=5,
        )
        self.assertTrue(
            torch.equal(
                composed,
                torch.tensor([0.25, -0.5, 0.75, 1.0, 0.0, 0.0, 1.0, 0.0, 2.0]),
            )
        )
        sampler_source = inspect.getsource(module.P8SuccessorPolicy.sample_successor_decision)
        self.assertNotIn("apply_successor_feature_v2_keep_gate", sampler_source)

    def test_frozen_schema_layout_digest_and_checkpoint_binding(self):
        module = self.policy
        schema = module.successor_feature_v2_schema()
        self.assertEqual(schema["version"], 2)
        self.assertEqual(schema["total_dim"], 3_604)
        self.assertEqual(
            [(item["name"], item["start"], item["stop"], item["dim"]) for item in schema["components"]],
            [
                ("contextual_hidden", 0, 3_584, 3_584),
                ("depth_summary", 3_584, 3_599, 15),
                ("official_action_onehot", 3_599, 3_603, 4),
                ("context_age", 3_603, 3_604, 1),
            ],
        )
        self.assertEqual(schema["context_age"]["gate_requires"], 0)
        self.assertEqual(
            schema["causal_binding"]["component_sources"],
            {
                "contextual_hidden": "generation_latched_contextual_hidden",
                "depth_summary": "current_decision_official_frame_metric_depth",
            },
        )
        self.assertEqual(
            schema["causal_binding"]["same_frame_when"], "context_age == 0"
        )
        self.assertNotIn("same_generate_components", schema["causal_binding"])
        self.assertIn(
            "reset_ordinal_matches_active_session",
            schema["causal_binding"]["runner_audit_requirements"],
        )
        self.assertIn(
            "depth_frame_ref_matches_current_official_frame",
            schema["causal_binding"]["runner_audit_requirements"],
        )
        self.assertIn(
            "depth_physical_step_matches_current_frame",
            schema["causal_binding"]["runner_audit_requirements"],
        )
        schema["components"][0]["stop"] = -1
        self.assertEqual(
            module.successor_feature_v2_schema()["components"][0]["stop"], 3_584
        )
        v1_digest = module.successor_feature_schema_digest(1)
        v2_digest = module.successor_feature_v2_schema_digest()
        self.assertRegex(v1_digest, r"^[0-9a-f]{64}$")
        self.assertRegex(v2_digest, r"^[0-9a-f]{64}$")
        self.assertNotEqual(v1_digest, v2_digest)

        bindings = {
            1: module.successor_feature_checkpoint_binding(1),
            2: module.successor_feature_checkpoint_binding(2),
        }
        for version, binding in bindings.items():
            with self.subTest(valid_version=version):
                self.assertEqual(
                    module.require_successor_feature_checkpoint_binding(
                        binding, expected_schema_version=version
                    ),
                    binding,
                )
        for actual, expected in ((1, 2), (2, 1)):
            with self.subTest(actual=actual, expected=expected):
                with self.assertRaisesRegex(RuntimeError, "binding mismatch"):
                    module.require_successor_feature_checkpoint_binding(
                        bindings[actual], expected_schema_version=expected
                    )
        tampered = dict(bindings[2])
        tampered["successor_feature_schema_sha256"] = "a" * 64
        with self.assertRaisesRegex(RuntimeError, "binding mismatch"):
            module.require_successor_feature_checkpoint_binding(
                tampered, expected_schema_version=2
            )
        reordered = {
            "successor_feature_schema_sha256": v2_digest,
            "successor_feature_schema_version": 2,
        }
        self.assertEqual(
            module.require_successor_feature_checkpoint_binding(
                reordered, expected_schema_version=2
            ),
            reordered,
        )
        extra_key = dict(bindings[2], unrelated=True)
        with self.assertRaisesRegex(ValueError, "schema drifted"):
            module.require_successor_feature_checkpoint_binding(
                extra_key, expected_schema_version=2
            )
        invalid_bindings = [
            ("not_mapping", None, TypeError, "must be a mapping"),
            (
                "missing_digest",
                {"successor_feature_schema_version": 2},
                ValueError,
                "schema drifted",
            ),
            (
                "boolean_version",
                {
                    "successor_feature_schema_version": True,
                    "successor_feature_schema_sha256": v2_digest,
                },
                TypeError,
                "must be an integer",
            ),
            (
                "malformed_digest",
                {
                    "successor_feature_schema_version": 2,
                    "successor_feature_schema_sha256": "A" * 64,
                },
                ValueError,
                "lowercase SHA-256",
            ),
        ]
        for name, binding, error_type, pattern in invalid_bindings:
            with self.subTest(invalid_binding=name):
                with self.assertRaisesRegex(error_type, pattern):
                    module.require_successor_feature_checkpoint_binding(
                        binding, expected_schema_version=2
                    )

    def test_composer_exact_tensor_and_audit_receipt(self):
        torch = self.torch
        module = self.policy
        kwargs = self._compose_kwargs()
        result = module.compose_successor_features_v2(**kwargs)
        self.assertEqual(tuple(result.shape), (3_604,))
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(result.device.type, "cpu")
        self.assertTrue(result.is_contiguous())
        self.assertFalse(result.requires_grad)
        self.assertTrue(torch.equal(result[:3_584], kwargs["contextual_hidden"]))
        self.assertTrue(torch.equal(result[3_584:3_599], kwargs["depth_summary"]))
        self.assertTrue(torch.equal(result[3_599:3_603], torch.tensor([0.0, 0.0, 1.0, 0.0])))
        self.assertEqual(float(result[-1]), 0.0)

        audit = module.audit_successor_features_v2(result)
        self.assertEqual(audit["schema_version"], 2)
        self.assertEqual(audit["schema_sha256"], module.successor_feature_v2_schema_digest())
        self.assertRegex(audit["feature_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(
            set(audit["component_sha256"]),
            {"contextual_hidden", "depth_summary", "official_action_onehot", "context_age"},
        )
        self.assertEqual(audit["official_shadow_action"], 2)
        self.assertEqual(audit["context_age"], 0)
        self.assertTrue(audit["causal_current"])

        stale_kwargs = dict(kwargs)
        stale_kwargs["physical_step"] = 13
        stale = module.compose_successor_features_v2(**stale_kwargs)
        stale_audit = module.audit_successor_features_v2(stale)
        self.assertEqual(stale_audit["context_age"], 2)
        self.assertFalse(stale_audit["causal_current"])

    def test_composer_rejects_dtype_shape_finite_depth_and_age_drift(self):
        torch = self.torch
        module = self.policy
        base = self._compose_kwargs()
        noncontiguous_hidden = torch.zeros(7_168, dtype=torch.float32)[::2]
        frozen_hidden = self._hidden().requires_grad_(True)
        cases = [
            ("hidden_shape", {"contextual_hidden": torch.zeros(3_583)}, ValueError, "shape"),
            ("hidden_dtype", {"contextual_hidden": self._hidden().double()}, TypeError, "float32"),
            ("hidden_noncontiguous", {"contextual_hidden": noncontiguous_hidden}, ValueError, "contiguous"),
            ("hidden_not_frozen", {"contextual_hidden": frozen_hidden}, ValueError, "frozen"),
            ("hidden_nonfinite", {"contextual_hidden": self._hidden().index_fill(0, torch.tensor([4]), math.inf)}, ValueError, "finite"),
            ("depth_shape", {"depth_summary": torch.zeros(14)}, ValueError, "shape"),
            ("depth_dtype", {"depth_summary": self._depth().to(torch.float64)}, TypeError, "float32"),
            ("depth_nonfinite", {"depth_summary": self._depth().index_fill(0, torch.tensor([0]), math.nan)}, ValueError, "finite"),
            ("depth_order", {"depth_summary": self._depth().index_copy(0, torch.tensor([0, 1]), torch.tensor([0.9, 0.2]))}, ValueError, "quantiles"),
            ("depth_fraction", {"depth_summary": self._depth().index_fill(0, torch.tensor([2]), 1.1)}, ValueError, "fractions"),
            ("empty_sector", {"depth_summary": self._depth().index_fill(0, torch.tensor([5]), 0.2)}, ValueError, "empty depth"),
            ("nonempty_zero", {"depth_summary": self._depth().index_fill(0, torch.tensor([0]), 0.0)}, ValueError, "positive quantiles"),
            ("action_bool", {"official_shadow_action": True}, ValueError, "0..3"),
            ("action_range", {"official_shadow_action": 4}, ValueError, "0..3"),
            ("future_context", {"context_physical_step": 12}, ValueError, "non-causal"),
            ("negative_context", {"context_physical_step": -1}, ValueError, "non-causal"),
            ("age_bound", {"physical_step": module.SUCCESSOR_FEATURE_MAX_AGE + 12}, ValueError, "bounded"),
        ]
        for name, replacement, error_type, pattern in cases:
            with self.subTest(name=name):
                kwargs = dict(base)
                kwargs.update(replacement)
                with self.assertRaisesRegex(error_type, pattern):
                    module.compose_successor_features_v2(**kwargs)

    def test_audit_rejects_corrupted_composed_receipts(self):
        torch = self.torch
        module = self.policy
        valid = module.compose_successor_features_v2(**self._compose_kwargs())
        invalid_onehot = valid.clone()
        invalid_onehot[3_599:3_603] = 0.0
        fractional_age = valid.clone()
        fractional_age[-1] = 0.5
        invalid_empty_depth = valid.clone()
        invalid_empty_depth[3_584:3_587] = torch.tensor([0.2, 0.3, 0.0])
        noncontiguous = torch.stack((valid, valid), dim=1)[:, 0]
        requires_grad = valid.clone().requires_grad_(True)
        cases = [
            ("shape", valid[:-1].clone(), ValueError, "shape"),
            ("dtype", valid.double(), TypeError, "float32"),
            ("noncontiguous", noncontiguous, ValueError, "contiguous"),
            ("not_frozen", requires_grad, ValueError, "frozen"),
            ("nonfinite", valid.clone().index_fill(0, torch.tensor([10]), math.nan), ValueError, "finite"),
            ("onehot", invalid_onehot, ValueError, "one-hot"),
            ("age", fractional_age, ValueError, "context age"),
            ("depth", invalid_empty_depth, ValueError, "empty depth"),
        ]
        for name, receipt, error_type, pattern in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(error_type, pattern):
                    module.audit_successor_features_v2(receipt)

    def test_conservative_gate_precedence_and_rejection_reasons(self):
        module = self.policy
        base = {
            "official_shadow_action": 1,
            "route_probabilities": (0.10, 0.80, 0.10),
            "waypoint_local_m": (0.0, 0.5),
            "context_age": 0,
            "recovery_active": False,
            "blocked_parent": False,
            "blocked_root": False,
        }
        cases = [
            (
                "official_stop_first",
                {
                    "official_shadow_action": 0,
                    "route_probabilities": None,
                    "waypoint_local_m": None,
                    "context_age": None,
                    "recovery_active": True,
                    "blocked_parent": True,
                    "blocked_root": True,
                },
                "official_stop_keep",
                None,
                False,
            ),
            ("recovery_before_root", {"recovery_active": True, "blocked_root": True, "context_age": None}, "recovery_or_blocked_parent_keep", None, False),
            ("blocked_parent", {"blocked_parent": True, "context_age": None}, "recovery_or_blocked_parent_keep", None, False),
            ("blocked_root", {"blocked_root": True, "context_age": None}, "blocked_root_keep_fallback", None, True),
            ("missing_context", {"context_age": None, "route_probabilities": None}, "missing_context_keep", None, False),
            ("stale_context", {"context_age": 1, "route_probabilities": None}, "stale_context_keep", None, False),
            ("invalid_context", {"context_age": -1}, "invalid_context_age_keep", None, False),
            ("missing_probabilities", {"route_probabilities": None}, "invalid_route_probabilities_keep", None, False),
            ("probability_shape", {"route_probabilities": (0.1, 0.9)}, "invalid_route_probabilities_keep", None, False),
            ("probability_sum", {"route_probabilities": (0.1, 0.9, 0.1)}, "invalid_route_probabilities_keep", None, False),
            ("probability_nonfinite", {"route_probabilities": (0.1, math.nan, 0.9)}, "invalid_route_probabilities_keep", None, False),
            ("argmax_keep", {"route_probabilities": (0.90, 0.05, 0.05)}, "learned_argmax_not_waypoint_keep", 0.05, False),
            ("learned_stop_has_no_control", {"route_probabilities": (0.05, 0.05, 0.90)}, "learned_argmax_not_waypoint_keep", 0.05, False),
            ("argmax_tie", {"route_probabilities": (0.50, 0.50, 0.00)}, "learned_argmax_not_waypoint_keep", 0.50, False),
            ("low_probability", {"route_probabilities": (0.11, 0.79, 0.10)}, "waypoint_probability_below_threshold_keep", 0.79, False),
            ("missing_waypoint", {"waypoint_local_m": None}, "invalid_waypoint_keep", 0.80, False),
            ("nan_waypoint", {"waypoint_local_m": (math.nan, 0.2)}, "invalid_waypoint_keep", 0.80, False),
            ("short_waypoint", {"waypoint_local_m": (0.2,)}, "invalid_waypoint_keep", 0.80, False),
            ("minimum_is_exclusive", {"waypoint_local_m": (module.SUCCESSOR_FEATURE_V2_WAYPOINT_MIN_M, 0.0)}, "waypoint_distance_out_of_range_keep", 0.80, False),
            ("maximum_exceeded", {"waypoint_local_m": (module.SUCCESSOR_FEATURE_V2_WAYPOINT_MAX_M + 1e-6, 0.0)}, "waypoint_distance_out_of_range_keep", 0.80, False),
            ("same_first_action", {"waypoint_local_m": (0.5, 0.0)}, "compiled_first_action_matches_official_keep", 0.80, False),
        ]
        for name, replacement, reason, p_waypoint, root_fallback in cases:
            with self.subTest(name=name):
                kwargs = dict(base)
                kwargs.update(replacement)
                result = module.apply_successor_feature_v2_keep_gate(**kwargs)
                self.assertFalse(result.accepted)
                self.assertIs(result.selected_route, module.RouteName.KEEP_ORIGINAL)
                self.assertEqual(result.selected_waypoint_local_m, (0.0, 0.0))
                self.assertEqual(result.reason, reason)
                self.assertEqual(result.p_waypoint, p_waypoint)
                self.assertEqual(result.root_fallback, root_fallback)
                self.assertNotEqual(result.selected_route, module.RouteName.STOP)

        invalid_inputs = [
            ("boolean_action", {"official_shadow_action": True}, ValueError, "0..3"),
            ("action_range", {"official_shadow_action": 4}, ValueError, "0..3"),
            ("non_boolean_recovery", {"recovery_active": 1}, TypeError, "must be boolean"),
            ("non_boolean_parent", {"blocked_parent": 1}, TypeError, "must be boolean"),
            ("non_boolean_root", {"blocked_root": 1}, TypeError, "must be boolean"),
        ]
        for name, replacement, error_type, pattern in invalid_inputs:
            with self.subTest(invalid_input=name):
                kwargs = dict(base)
                kwargs.update(replacement)
                with self.assertRaisesRegex(error_type, pattern):
                    module.apply_successor_feature_v2_keep_gate(**kwargs)

    def test_conservative_gate_accepts_only_current_confident_distinct_waypoint(self):
        module = self.policy
        cases = [
            ("threshold_inclusive", 0.80, (0.0, 0.5), 2),
            ("maximum_inclusive", 0.90, (0.0, module.SUCCESSOR_FEATURE_V2_WAYPOINT_MAX_M), 2),
            ("just_above_minimum", 0.95, (module.SUCCESSOR_FEATURE_V2_WAYPOINT_MIN_M + 1e-6, 0.0), 1),
        ]
        for name, p_waypoint, waypoint, first_action in cases:
            with self.subTest(name=name):
                remainder = 1.0 - p_waypoint
                official_action = 2 if first_action == 1 else 1
                result = module.apply_successor_feature_v2_keep_gate(
                    official_shadow_action=official_action,
                    route_probabilities=(remainder / 2.0, p_waypoint, remainder / 2.0),
                    waypoint_local_m=waypoint,
                    context_age=0,
                )
                self.assertTrue(result.accepted)
                self.assertIs(result.selected_route, module.RouteName.WAYPOINT)
                self.assertEqual(result.selected_waypoint_local_m, waypoint)
                self.assertEqual(result.reason, "accepted_waypoint")
                self.assertAlmostEqual(result.p_waypoint, p_waypoint)
                self.assertEqual(result.compiled_first_action, first_action)
                self.assertIsNotNone(result.compiled_actions)
                self.assertFalse(result.root_fallback)
                audit = result.to_dict()
                self.assertEqual(audit["selected_route"], "waypoint")
                self.assertEqual(audit["selected_waypoint_local_m"], list(waypoint))


if __name__ == "__main__":
    unittest.main()
