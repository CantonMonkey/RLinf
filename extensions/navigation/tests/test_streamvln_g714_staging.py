import json
import math
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
INTEGRATION = ROOT / "integrations" / "rlinf_v03"
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(INTEGRATION))

from streamvln_visibility_g714 import (  # noqa: E402
    DOWNWARD_PITCH_RADIANS,
    EXPECTED_GRID_COUNTS,
    EXPECTED_OPTICAL_AXIS_AGENT_FRAME,
    EXPECTED_TARGET_RAY_COUNT,
    FROZEN_GEOMETRY_SUMMARY,
    TARGET_RAYS,
    immediate_footprint_observables,
    visibility_gate,
)


def _depth_with(offset):
    depth = [[0.0 for _ in range(64)] for _ in range(64)]
    for ray in TARGET_RAYS:
        depth[ray["row"]][ray["column"]] = max(
            0.0, ray["expected_optical_depth_m"] + offset
        )
    return depth


def test_geometry_is_uniquely_derived_and_matches_manifest():
    assert DOWNWARD_PITCH_RADIANS == -math.atan2(1.25, 0.25)
    assert len(TARGET_RAYS) == EXPECTED_TARGET_RAY_COUNT == 48
    assert FROZEN_GEOMETRY_SUMMARY["forward_bin_counts"] == [18, 18, 12]
    assert FROZEN_GEOMETRY_SUMMARY["lateral_bin_counts"] == [16, 16, 16]
    assert FROZEN_GEOMETRY_SUMMARY["grid_counts"] == [
        list(row) for row in EXPECTED_GRID_COUNTS
    ]
    assert min(value for row in EXPECTED_GRID_COUNTS for value in row) == 4
    assert math.isclose(EXPECTED_OPTICAL_AXIS_AGENT_FRAME[1], -1.25 / math.hypot(1.25, 0.25))
    assert math.isclose(EXPECTED_OPTICAL_AXIS_AGENT_FRAME[2], -0.25 / math.hypot(1.25, 0.25))

    manifest = json.loads(
        (INTEGRATION / "streamvln_g714_experiment.json").read_text()
    )["g714_immediate_footprint_visibility"]
    assert manifest["case_count"] == len(manifest["cases"]) == 26
    assert manifest["camera"]["pitch_radians"] == DOWNWARD_PITCH_RADIANS
    assert manifest["target_region"]["expected_target_ray_count"] == 48
    assert manifest["target_region"]["expected_grid_counts"] == [
        list(row) for row in EXPECTED_GRID_COUNTS
    ]
    assert manifest["pitch_sweep"] is False
    assert manifest["threshold_sweep"] is False
    assert manifest["controller_implementation"] is False
    assert manifest["navigation_evaluation"] is False
    assert manifest["val_unseen_episode_count"] == 0


def test_observables_partition_support_occlusion_and_missing():
    support = immediate_footprint_observables(_depth_with(0.0))
    assert support["support_count"] == 48
    assert support["occlusion_count"] == 0
    assert support["missing_count"] == 0
    assert support["classified_count"] == 48

    occlusion = immediate_footprint_observables(_depth_with(-0.2))
    assert occlusion["support_count"] == 0
    assert occlusion["occlusion_count"] == 48
    assert occlusion["missing_count"] == 0

    missing = immediate_footprint_observables(_depth_with(0.2))
    assert missing["support_count"] == 0
    assert missing["occlusion_count"] == 0
    assert missing["missing_count"] == 48


def test_visibility_gate_requires_every_integrity_dimension():
    metrics = {
        "target_ray_count": 48,
        "minimum_spatial_bin_count": 4,
        "all_target_depths_classified": True,
        "all_repeats_exact": True,
        "maximum_optical_axis_error": 0.0,
        "maximum_sensor_position_error_m": 0.0,
    }
    assert visibility_gate(metrics, integrity_ready=True)["ready"] is True
    for key in list(metrics):
        changed = dict(metrics)
        if key == "target_ray_count":
            changed[key] = 47
        elif key == "minimum_spatial_bin_count":
            changed[key] = 3
        elif isinstance(changed[key], bool):
            changed[key] = False
        else:
            changed[key] = 2e-6
        assert visibility_gate(changed, integrity_ready=True)["ready"] is False
    assert visibility_gate(metrics, integrity_ready=False)["ready"] is False


def test_staging_contains_fixed_pitch_and_strict_protocol():
    smoke = (SCRIPTS / "habitat_r2r_rgb_instruction_smoke.py").read_text()
    base = (SCRIPTS / "habitat_r2r_service.py").read_text()
    service = (SCRIPTS / "habitat_r2r_visibility_service.py").read_text()
    env = (INTEGRATION / "vln_env.py").read_text()
    runner = (INTEGRATION / "run_streamvln_visibility_g714.py").read_text()
    assert "depth_pitch_radians=0.0" in smoke
    assert "depth_sensor.orientation = [float(depth_pitch_radians), 0.0, 0.0]" in smoke
    assert "self.depth_pitch_radians" in base
    assert "args.depth_pitch_radians = DOWNWARD_PITCH_RADIANS" in service
    assert "probe_immediate_footprint_visibility" in service
    assert "critical_primitive_executed\": False" in service
    assert "def probe_immediate_footprint_visibility" in env
    assert "timeout=900" in env
    assert "no_risk_controller_or_navigation" in runner
    assert '"g8_ready": False' in runner


def test_plan_was_preregistered_as_visibility_only():
    plan_path = ROOT.parent / "STREAMVLN_G714_EXPERIMENT_PLAN.md"
    if not plan_path.is_file():
        raise unittest.SkipTest("Historical experiment plan is not included in the public source export")
    plan = plan_path.read_text()
    assert "Preregistered on 2026-09-27 before collecting" in plan
    assert "-78.69006752597979 deg" in plan
    assert "No alternative pitch is rendered or evaluated" in plan
    assert "They are not a usefulness or safety gate" in plan
