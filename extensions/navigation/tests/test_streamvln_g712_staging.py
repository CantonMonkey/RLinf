from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
INTEGRATION = ROOT / "integrations" / "rlinf_v03"
sys.path.insert(0, str(INTEGRATION))

from streamvln_oracle_g712 import (
    joint_gate,
    prefix_outcome,
    select_longest_admissible_prefix,
)


def recorded_repeat(progresses):
    start = [0.0, 0.0, 0.0]
    position = start[:]
    trace = []
    clearances = [{"clearance_m": 0.2}]
    for index, progress in enumerate(progresses):
        before = position[:]
        position = [position[0], position[1], position[2] - progress]
        trace.append(
            {
                "primitive_index": index,
                "before_position": before,
                "after_position": position[:],
                "forward_progress_m": progress,
                "collision": progress < 0.25,
            }
        )
        clearances.append({"clearance_m": 0.2})
    commanded = len(progresses)
    return {
        "commanded_forward_primitives": commanded,
        "post_turn_pose": {"position": start},
        "forward_direction": [0.0, 0.0, -1.0],
        "full_intended_endpoint": [0.0, 0.0, -0.25 * commanded],
        "primitive_trace": trace,
        "actual_clearance_trace": clearances,
    }


class G712TargetAlignedOracleTest(unittest.TestCase):
    def test_safe_recorded_macro_selects_full_prefix(self):
        repeat = recorded_repeat([0.25, 0.25, 0.25])
        selection = select_longest_admissible_prefix(repeat)
        self.assertEqual(selection["selected"]["selected_prefix"], 3)
        self.assertTrue(selection["selected"]["completed"])
        self.assertFalse(selection["selected"]["severe_exposure"])

    def test_first_severe_prefix_stops_and_does_not_consider_recovery(self):
        repeat = recorded_repeat([0.25, 0.0, 0.25])
        selection = select_longest_admissible_prefix(repeat)
        self.assertEqual(selection["selected"]["selected_prefix"], 1)
        self.assertEqual(len(selection["evaluated_prefixes"]), 2)
        self.assertTrue(selection["evaluated_prefixes"][-1]["severe_exposure"])

    def test_severe_first_primitive_selects_explicit_zero_motion(self):
        repeat = recorded_repeat([0.0, 0.25])
        selection = select_longest_admissible_prefix(repeat)
        self.assertEqual(selection["selected"]["selected_prefix"], 0)
        self.assertIsNone(selection["selected"]["continuous_severity"])
        self.assertFalse(selection["selected"]["severe_exposure"])
        self.assertAlmostEqual(prefix_outcome(repeat, 0)["full_endpoint_error_m"], 0.5)

    def test_joint_gate_preserves_g711_safety_and_activity_thresholds(self):
        passing = {
            "severe_exposure_reduction": 0.70,
            "free_grazing_completion_rate": 0.80,
            "free_grazing_progress_retention": 0.90,
            "overall_forward_primitive_retention": 0.50,
            "stratum_severe_nonincrease": True,
        }
        self.assertTrue(joint_gate(passing, integrity_ready=True)["ready"])
        passing["overall_forward_primitive_retention"] = 0.49
        self.assertFalse(joint_gate(passing, integrity_ready=True)["ready"])

    def test_manifest_and_runner_freeze_privileged_non_navigation_scope(self):
        manifest = json.loads(
            (INTEGRATION / "streamvln_g712_experiment.json").read_text()
        )
        audit = manifest["g712_target_aligned_oracle"]
        self.assertEqual(audit["row_count"], 128)
        self.assertEqual(audit["expected_missing_counterfactual_prefixes"], 0)
        self.assertEqual(
            audit["severity_thresholds"],
            {
                "severe_realized_ratio": 0.5,
                "severe_endpoint_error_m": 0.25,
                "severe_lateral_error_m": 0.25,
                "severe_min_primitive_progress_m": 0.05,
            },
        )
        runner = (INTEGRATION / "run_streamvln_oracle_g712.py").read_text()
        self.assertIn('"no_threshold_sweep": True', runner)
        self.assertIn('"no_model_fit": True', runner)
        self.assertIn('"no_navigation_evaluation": True', runner)
        self.assertIn('"zero_val_unseen": True', runner)
        self.assertIn('"g8_ready": False', runner)
        self.assertNotIn("make_env", runner)
        self.assertNotIn("torch", runner)


if __name__ == "__main__":
    unittest.main()
