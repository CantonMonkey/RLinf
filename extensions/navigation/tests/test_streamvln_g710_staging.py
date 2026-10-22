from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
INTEGRATION = ROOT / "integrations" / "rlinf_v03"
sys.path.insert(0, str(INTEGRATION))

from streamvln_severity_g710 import (
    predictor_gates,
    severity_outcome,
    spearman_correlation,
)


def repeat(*, realized, endpoint, lateral, progresses, collisions):
    return {
        "commanded_distance_m": 0.5,
        "realized_forward_m": realized,
        "endpoint_error_m": endpoint,
        "realized_lateral_error_m": lateral,
        "primitive_trace": [
            {"forward_progress_m": value} for value in progresses
        ],
        "full_collision_trace": collisions,
    }


class G710SeverityAuditTest(unittest.TestCase):
    def test_severity_bands_match_preregistered_thresholds(self):
        free = severity_outcome(
            repeat(
                realized=0.49,
                endpoint=0.01,
                lateral=0.0,
                progresses=[0.245, 0.245],
                collisions=[False, False],
            )
        )
        grazing = severity_outcome(
            repeat(
                realized=0.49,
                endpoint=0.01,
                lateral=0.0,
                progresses=[0.245, 0.245],
                collisions=[True, False],
            )
        )
        sliding = severity_outcome(
            repeat(
                realized=0.42,
                endpoint=0.08,
                lateral=0.02,
                progresses=[0.21, 0.21],
                collisions=[True, False],
            )
        )
        severe = severity_outcome(
            repeat(
                realized=0.10,
                endpoint=0.40,
                lateral=0.01,
                progresses=[0.04, 0.06],
                collisions=[True, True],
            )
        )
        self.assertEqual(
            [free["severity_class"], grazing["severity_class"],
             sliding["severity_class"], severe["severity_class"]],
            ["free", "grazing", "sliding", "severe_blockage"],
        )
        self.assertFalse(free["raw_collision"])
        self.assertTrue(grazing["raw_collision"])
        self.assertTrue(severe["severe_blockage"])

    def test_spearman_and_unavailable_grazing_gate_are_explicit(self):
        self.assertAlmostEqual(
            spearman_correlation([0.1, 0.2, 0.3], [1.0, 2.0, 3.0]), 1.0
        )
        gates = predictor_gates(
            spearman=0.9,
            severe_metrics={
                "status": "ok",
                "auroc": 0.9,
                "unsafe_recall": 0.9,
                "balanced_accuracy": 0.9,
            },
            grazing_false_positive_rate=None,
            target_repeat_exact=True,
            pairing_exact=True,
        )
        self.assertTrue(gates["integrity_ready"])
        self.assertFalse(gates["score"]["grazing_false_positive_rate"])
        self.assertFalse(gates["score_ready"])

    def test_manifest_fixes_states_counts_thresholds_and_zero_eval_use(self):
        manifest = json.loads(
            (INTEGRATION / "streamvln_g710_experiment.json").read_text()
        )
        audit = manifest["g710_severity_audit"]
        self.assertEqual(audit["split"], "train")
        self.assertEqual(audit["candidate_count_per_state"], 32)
        self.assertEqual(audit["repeat_count"], 2)
        self.assertEqual(audit["total_rows"], 128)
        self.assertEqual(
            audit["state_definitions"],
            [
                {"name": "reset", "transition_actions": []},
                {"name": "after_fixed_p0_d0p5", "transition_actions": [1, 1]},
            ],
        )
        self.assertEqual(audit["clearance_spacing_m"], 0.125)
        self.assertEqual(
            audit["severity_thresholds"]["severe_realized_ratio"], 0.5
        )
        self.assertEqual(
            audit["train_dev_scenes"],
            [row["scene"] for row in manifest["train"][6:]],
        )

    def test_runner_is_frozen_diagnostic_only(self):
        runner = (INTEGRATION / "run_streamvln_severity_g710.py").read_text()
        self.assertIn("load_frozen_g78", runner)
        self.assertIn('manifest["train"][6:]', runner)
        self.assertIn('"val_unseen_episode_count": 0', runner)
        self.assertIn('"model_fit": False', runner)
        self.assertIn('"navigation_evaluation": False', runner)
        self.assertIn('"g8_ready": False', runner)
        self.assertIn("stop-current-learned-depth-predictor", runner)
        self.assertNotIn("train_depth_model", runner)
        self.assertNotIn("navigation_episode", runner)

    def test_service_records_path_clearance_without_swept_path_claim(self):
        service = (
            ROOT / "scripts" / "habitat_r2r_severity_audit_service.py"
        ).read_text()
        client = (INTEGRATION / "vln_env.py").read_text()
        self.assertIn("distance_to_closest_obstacle", service)
        self.assertIn("try_step_no_sliding", service)
        self.assertIn('"intended_clearance_trace"', service)
        self.assertIn('"actual_clearance_trace"', service)
        self.assertNotIn("continuous_collision_free", service)
        self.assertIn("def probe_severity_audit(self, candidates):", client)
        self.assertIn("G7.10 severity audit requires exactly 32 candidates", client)


if __name__ == "__main__":
    unittest.main()
