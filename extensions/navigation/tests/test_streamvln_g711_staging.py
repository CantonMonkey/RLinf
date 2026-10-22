from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
INTEGRATION = ROOT / "integrations" / "rlinf_v03"
sys.path.insert(0, str(INTEGRATION))

from streamvln_geometry_g711 import (
    controlled_motion_severity,
    controller_gate,
    depth_corridor_measurement,
)


class G711GeometryControllerTest(unittest.TestCase):
    def test_depth_corridor_uses_fixed_footprint_and_brake_distance(self):
        clear = [[10.0 for _ in range(64)] for _ in range(64)]
        clear_measurement = depth_corridor_measurement(clear)
        self.assertTrue(clear_measurement["allow_forward"])
        self.assertEqual(clear_measurement["brake_distance_m"], 0.37)
        blocked = [row[:] for row in clear]
        blocked[31][31] = 0.30
        blocked_measurement = depth_corridor_measurement(blocked)
        self.assertFalse(blocked_measurement["allow_forward"])
        self.assertAlmostEqual(blocked_measurement["minimum_forward_m"], 0.30)

    def test_controlled_motion_severity_keeps_zero_motion_explicit(self):
        zero = {
            "attempted_distance_m": 0.0,
            "realized_forward_m": 0.0,
            "attempted_endpoint_error_m": 0.0,
            "realized_lateral_error_m": 0.0,
            "primitive_trace": [],
        }
        self.assertIsNone(controlled_motion_severity(zero))
        severe = {
            "attempted_distance_m": 0.5,
            "realized_forward_m": 0.10,
            "attempted_endpoint_error_m": 0.40,
            "realized_lateral_error_m": 0.01,
            "primitive_trace": [
                {"forward_progress_m": 0.04},
                {"forward_progress_m": 0.06},
            ],
        }
        result = controlled_motion_severity(severe)
        self.assertTrue(result["severe_exposure"])
        self.assertEqual(result["continuous_severity"], 1.0)

    def test_joint_gate_cannot_pass_by_rejecting_all_motion(self):
        safe = {
            "severe_exposure_reduction": 0.8,
            "free_grazing_completion_rate": 0.9,
            "free_grazing_progress_retention": 0.95,
            "overall_forward_primitive_retention": 0.6,
            "stratum_severe_nonincrease": True,
        }
        self.assertTrue(controller_gate(safe, integrity_ready=True)["ready"])
        reject_all = dict(safe)
        reject_all.update(
            {
                "free_grazing_completion_rate": 0.0,
                "free_grazing_progress_retention": 0.0,
                "overall_forward_primitive_retention": 0.0,
            }
        )
        self.assertFalse(
            controller_gate(reject_all, integrity_ready=True)["ready"]
        )

    def test_manifest_freezes_physics_variants_and_gates(self):
        manifest = json.loads(
            (INTEGRATION / "streamvln_g711_experiment.json").read_text()
        )
        audit = manifest["g711_geometry_controller"]
        self.assertEqual(audit["split"], "train")
        self.assertEqual(audit["row_count"], 128)
        self.assertEqual(audit["physical_repeat_count"], 1024)
        self.assertEqual(
            audit["physical_variants"],
            [
                "open_loop",
                "endpoint_geometry",
                "depth_receding",
                "navmesh_receding_oracle",
            ],
        )
        self.assertEqual(
            audit["depth_projection"]["primitive_brake_distance_m"], 0.37
        )
        self.assertEqual(
            audit["gates"]["free_grazing_progress_retention_min"], 0.9
        )
        self.assertEqual(audit["state_definitions"][1]["transition_actions"], [1, 1])

    def test_service_and_runner_preserve_privileged_boundary(self):
        service = (
            ROOT / "scripts" / "habitat_r2r_geometry_controller_service.py"
        ).read_text()
        runner = (INTEGRATION / "run_streamvln_geometry_g711.py").read_text()
        client = (INTEGRATION / "vln_env.py").read_text()
        self.assertIn("depth_corridor_measurement", service)
        self.assertIn("try_step_no_sliding", service)
        self.assertIn("navmesh_receding_oracle", service)
        self.assertIn("causal-metric-depth-footprint-corridor", service)
        self.assertIn('"navigation_evaluation": False', runner)
        self.assertIn('"val_unseen_episode_count": 0', runner)
        self.assertIn('"causal_depth_excludes_privileged_inputs": True', runner)
        self.assertIn('"g8_ready": False', runner)
        self.assertNotIn("train_depth_model", runner)
        self.assertNotIn("navigation_episode", runner)
        self.assertIn("def probe_geometry_controller(self, candidates):", client)


if __name__ == "__main__":
    unittest.main()
