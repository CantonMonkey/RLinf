from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
INTEGRATION = ROOT / "integrations" / "rlinf_v03"
sys.path.insert(0, str(INTEGRATION))

from streamvln_ground_support_g713 import (
    CORRIDOR_HALF_WIDTH_M,
    CX,
    CY,
    FX,
    FY,
    GROUND_FORWARD_MAX_M,
    SENSOR_HEIGHT_M,
    binary_auroc,
    ground_support_observables,
    observability_gate,
)


def expected_ground_depth(kind):
    depth = [[10.0 for _ in range(64)] for _ in range(64)]
    for row_index in range(64):
        vertical = row_index - CY
        if vertical <= 0.0:
            continue
        expected = SENSOR_HEIGHT_M * FY / vertical
        if expected > GROUND_FORWARD_MAX_M:
            continue
        for column_index in range(64):
            lateral = (column_index - CX) * expected / FX
            if abs(lateral) <= CORRIDOR_HALF_WIDTH_M:
                if kind == "support":
                    depth[row_index][column_index] = expected
                elif kind == "occlusion":
                    depth[row_index][column_index] = max(0.01, expected - 0.2)
                elif kind == "missing":
                    depth[row_index][column_index] = 0.0
    return depth


class G713GroundSupportTest(unittest.TestCase):
    def test_expected_ground_rays_partition_fixed_support_types(self):
        support = ground_support_observables(expected_ground_depth("support"))
        missing = ground_support_observables(expected_ground_depth("missing"))
        occlusion = ground_support_observables(expected_ground_depth("occlusion"))
        self.assertGreater(support["ray_count"], 0)
        self.assertEqual(support["ray_count"], missing["ray_count"])
        self.assertEqual(support["ray_count"], occlusion["ray_count"])
        self.assertEqual(support["footprint_support_fraction"], 1.0)
        self.assertEqual(missing["footprint_dropoff_fraction"], 1.0)
        self.assertEqual(occlusion["footprint_occlusion_fraction"], 1.0)

    def test_primary_risk_direction_and_gate_are_fixed(self):
        self.assertEqual(binary_auroc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]), 1.0)
        metrics = {
            "overall_auroc": 0.75,
            "minimum_scene_auroc": 0.65,
            "minimum_state_auroc": 0.65,
            "paired_strict_win_rate": 0.75,
            "positive_median_risk": 0.5,
            "control_median_risk": 0.4,
        }
        self.assertTrue(observability_gate(metrics, integrity_ready=True)["ready"])
        metrics["paired_strict_win_rate"] = 0.74
        self.assertFalse(observability_gate(metrics, integrity_ready=True)["ready"])

    def test_manifest_freezes_pairs_observation_and_no_fit_boundary(self):
        manifest = json.loads(
            (INTEGRATION / "streamvln_g713_experiment.json").read_text()
        )
        audit = manifest["g713_ground_support_observability"]
        pairs = audit["matching"]["pairs"]
        self.assertEqual(len(pairs), 26)
        self.assertEqual(len({(p["scene"], p["state_name"], p["residual_candidate"]) for p in pairs}), 26)
        self.assertEqual(len({(p["scene"], p["state_name"], p["control_candidate"]) for p in pairs}), 26)
        self.assertEqual(audit["matching"]["control_reuse_max"], 1)
        self.assertEqual(audit["matching"]["maximum_heading_gap_degrees"], 120.0)
        self.assertEqual(audit["depth_geometry"]["primary_risk_score"], "footprint_dropoff_fraction")
        self.assertFalse(audit["model_fit"])
        self.assertFalse(audit["threshold_sweep"])
        self.assertFalse(audit["controller_implementation"])
        self.assertFalse(audit["navigation_evaluation"])
        self.assertEqual(audit["val_unseen_episode_count"], 0)

    def test_service_collects_pre_action_depth_without_privileged_geometry(self):
        service = (ROOT / "scripts" / "habitat_r2r_ground_support_service.py").read_text()
        helper = (INTEGRATION / "streamvln_ground_support_g713.py").read_text()
        self.assertIn("critical_prefix_minus_one", service.replace("-", "_"))
        self.assertIn('"critical_primitive_executed": False', service)
        self.assertIn("metric_depth", service)
        self.assertNotIn("try_step_no_sliding", service)
        self.assertNotIn("is_navigable", service)
        self.assertNotIn("navmesh", helper.lower())
        self.assertNotIn("collision", helper.lower())
        self.assertNotIn("severity", helper.lower())

    def test_runner_and_client_keep_controller_and_navigation_closed(self):
        runner = (INTEGRATION / "run_streamvln_ground_support_g713.py").read_text()
        client = (INTEGRATION / "vln_env.py").read_text()
        self.assertIn("def probe_ground_support(self, candidates):", client)
        ground_method = client.split("def probe_ground_support(self, candidates):", 1)[1].split(
            "def probe_severity_audit(self, candidates):", 1
        )[0]
        self.assertIn("def finite_vector(value, length, name):", ground_method)
        self.assertIn('"model_fit": False', runner)
        self.assertIn('"threshold_sweep": False', runner)
        self.assertIn('"controller_implementation": False', runner)
        self.assertIn('"navigation_evaluation": False', runner)
        self.assertIn('"g8_ready": False', runner)
        self.assertNotIn("train_depth_model", runner)
        self.assertNotIn("navigation_episode", runner)


if __name__ == "__main__":
    unittest.main()
