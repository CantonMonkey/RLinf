from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
INTEGRATION = ROOT / "integrations" / "rlinf_v03"
sys.path.insert(0, str(INTEGRATION))

from streamvln_candidates_g74 import generate_macro_candidates
from streamvln_macro_audit_g79 import (
    EXCLUDED_CANDIDATE,
    classify_route,
    distance_prefix_audit,
    select_registered_candidates,
)
from streamvln_topology import OnlineTopology


class G79MacroAuditTest(unittest.TestCase):
    def test_preregistered_bank_is_exactly_32_per_scene(self):
        bank = generate_macro_candidates(OnlineTopology(max_decisions=40))
        selected = select_registered_candidates(bank)
        self.assertEqual(len(bank), 33)
        self.assertEqual(len(selected), 32)
        self.assertNotIn(EXCLUDED_CANDIDATE, {row.name for row in selected})
        self.assertEqual(
            {row.name for row in bank} - {row.name for row in selected},
            {"macro_p90_d0p5"},
        )

    def test_prefix_audit_and_multi_cause_routing_are_explicit(self):
        rows = [
            {
                "scene": "scene",
                "heading_degrees": 0.0,
                "distance_m": 0.5,
                "candidate": "short",
                "unsafe": True,
            },
            {
                "scene": "scene",
                "heading_degrees": 0.0,
                "distance_m": 1.0,
                "candidate": "long",
                "unsafe": False,
            },
        ]
        audit = distance_prefix_audit(rows, "unsafe")
        self.assertEqual(audit["violation_count"], 1)
        route = classify_route(
            pairing_exact=False,
            collision_repeat_exact=False,
            prefix_monotonic=False,
            privileged_contradiction_rate=0.2,
            endpoint_repeat_max_m=0.02,
            horizon_label_change_rate=0.2,
            horizon_balanced_accuracy_gain=0.2,
        )
        self.assertTrue(route["pairing_issue"])
        self.assertTrue(route["label_horizon_issue"])
        self.assertTrue(route["executor_issue"])
        self.assertEqual(len(route["routes"]), 3)

    def test_runner_is_diagnostic_only_and_freezes_g78(self):
        runner = (
            INTEGRATION / "run_streamvln_macro_audit_g79.py"
        ).read_text()
        self.assertIn("load_frozen_g78", runner)
        self.assertIn('manifest["train"][6:]', runner)
        self.assertIn('"val_unseen_episode_count": 0', runner)
        self.assertIn('"model_fit": False', runner)
        self.assertIn('"navigation_evaluation": False', runner)
        self.assertIn('"g8_ready": False', runner)
        self.assertNotIn("train_depth_model", runner)
        self.assertNotIn("navigation_episode", runner)
        self.assertIn("A_current_pairing_original_full_macro_label", runner)
        self.assertIn("D_strict_pairing_fixed_0p5m_horizon_label", runner)

    def test_service_records_primitive_causality_and_privileged_check(self):
        service = (
            ROOT / "scripts" / "habitat_r2r_macro_audit_service.py"
        ).read_text()
        client = (INTEGRATION / "vln_env.py").read_text()
        self.assertIn("pathfinder.try_step", service)
        self.assertIn('"primitive_trace"', service)
        self.assertIn('"full_collision_trace"', service)
        self.assertIn('"horizon_collision_trace"', service)
        self.assertIn("REPEAT_COUNT = 2", service)
        self.assertIn("HORIZON_METERS = FORWARD_STEP_METERS * HORIZON_FORWARD_STEPS", service)
        self.assertIn("def probe_macro_audit(self, candidates):", client)
        self.assertIn("G7.9 macro audit requires exactly 32 candidates", client)

    def test_manifest_keeps_two_train_dev_scenes_and_zero_eval_use(self):
        manifest = json.loads(
            (INTEGRATION / "streamvln_g79_experiment.json").read_text()
        )
        audit = manifest["g79_macro_audit"]
        self.assertEqual(audit["split"], "train")
        self.assertEqual(audit["decision_index"], 0)
        self.assertEqual(audit["candidate_count_per_scene"], 32)
        self.assertEqual(audit["excluded_candidate"], "macro_p90_d0p5")
        self.assertEqual(audit["repeat_count"], 2)
        self.assertEqual(audit["physical_horizon_m"], 0.5)
        self.assertEqual(
            audit["train_dev_scenes"],
            [row["scene"] for row in manifest["train"][6:]],
        )
        self.assertEqual(
            audit["episode_indices"],
            [row["index"] for row in manifest["train"][6:]],
        )


if __name__ == "__main__":
    unittest.main()
