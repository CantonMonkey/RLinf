import hashlib
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
INTEGRATION = ROOT / "integrations" / "rlinf_v03"
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(INTEGRATION))

from streamvln_outcome_g715 import (  # noqa: E402
    binary_auroc,
    deterministic_matches,
    visual_risk,
)


def _observable(support, occlusion, missing, *, unclipped=48):
    return {
        "target_ray_count": 48,
        "classified_count": 48,
        "unclipped_count": unclipped,
        "support_fraction": support,
        "occlusion_fraction": occlusion,
        "missing_fraction": missing,
    }


def _row(candidate, heading, severe, risk):
    return {
        "scene": "scene",
        "state_name": "reset",
        "distance_m": 0.5,
        "candidate": candidate,
        "heading_degrees": heading,
        "severe_blockage": severe,
        "visual_risk": risk,
    }


def test_visual_risk_is_exact_equal_weight_non_support_fraction():
    assert visual_risk(_observable(0.25, 0.5, 0.25)) == 0.75
    with pytest.raises(ValueError):
        visual_risk(_observable(0.25, 0.5, 0.25, unclipped=47))
    with pytest.raises(ValueError):
        visual_risk(_observable(0.25, 0.5, 0.20))


def test_binary_auroc_uses_strict_wins_and_half_ties():
    result = binary_auroc([True, True, False, False], [1.0, 0.5, 0.5, 0.0])
    assert result["status"] == "ok"
    assert result["strict_wins"] == 3
    assert result["ties"] == 1
    assert result["auroc"] == 0.875
    assert binary_auroc([True, True], [0.2, 0.4])["status"] == "single_class"


def test_matching_is_max_cardinality_min_gap_and_lexicographic():
    rows = [
        _row("severe_a", 0.0, True, 0.9),
        _row("severe_b", 30.0, True, 0.8),
        _row("control_c", 15.0, False, 0.2),
        _row("control_d", 15.0, False, 0.3),
    ]
    pairs = deterministic_matches(rows, maximum_heading_gap_degrees=30.0)
    assert len(pairs) == 2
    assert [(row["severe_candidate"], row["control_candidate"]) for row in pairs] == [
        ("severe_a", "control_c"),
        ("severe_b", "control_d"),
    ]
    assert all(row["strict_win"] for row in pairs)


def test_manifest_freezes_population_feature_label_matching_and_gates():
    path = INTEGRATION / "streamvln_g715_experiment.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "d7014c667025f40c90681cc8851697e6a2d06e4616bd28b4e50d7bf6c120a522"
    )
    manifest = json.loads(path.read_text())
    prereg = manifest["g715_executed_outcome_discrimination"]
    assert len(prereg["scene_specs"]) == 6
    assert not {
        row["scene"] for row in prereg["scene_specs"]
    }.intersection({"GdvgFV5R1Z5", "JF19kD82Mey"})
    assert len(prereg["cases"]) == prereg["candidate_count_per_state"] == 32
    assert all(
        case["critical_action"] == 1
        and case["observe_before_forward_index"] == len(case["forward_suffix"])
        for case in prereg["cases"]
    )
    assert prereg["row_count"] == 384
    assert prereg["snapshot_outcome_count"] == 768
    assert prereg["risk_score"]["name"] == "equal_weight_non_support_fraction"
    assert prereg["severity_contract"]["source"].startswith("unchanged-g710")
    assert prereg["matching"]["maximum_heading_gap_degrees"] == 30.0
    assert prereg["separation_gates"]["overall_auroc_min"] == 0.75
    for key in (
        "model_load",
        "model_fit",
        "rl_update",
        "pitch_sweep",
        "feature_sweep",
        "threshold_sweep",
        "controller_implementation",
        "navigation_evaluation",
    ):
        assert prereg[key] is False
    assert prereg["val_unseen_episode_count"] == 0


def test_staging_enforces_one_critical_action_and_keeps_g8_closed():
    service = (SCRIPTS / "habitat_r2r_outcome_service.py").read_text()
    env = (INTEGRATION / "vln_env.py").read_text()
    runner = (INTEGRATION / "run_streamvln_outcome_g715.py").read_text()
    assert "self.env.sim.step(1)" in service
    assert '"critical_action_count_per_repeat": 1' in service
    assert "probe_immediate_footprint_outcomes" in service
    assert "def probe_immediate_footprint_outcomes" in env
    assert "critical != len(forward_suffix)" in env
    assert "timeout=900" in env
    assert '"candidate": case["name"]' in runner
    assert "no_model_fit_rl_sweep_controller_or_navigation" in runner
    assert '"g8_ready": False' in runner


def test_plan_was_preregistered_before_new_observations_or_outcomes():
    plan_path = ROOT.parent / "STREAMVLN_G715_EXPERIMENT_PLAN.md"
    if not plan_path.is_file():
        pytest.skip("Historical experiment plan is not included in the public source export")
    plan = plan_path.read_text()
    assert "Preregistered on 2026-09-27 before collecting or inspecting any G7.15" in plan
    assert "visual_risk = occlusion_fraction + missing_fraction" in plan
    assert "maximum-cardinality one-to-one bipartite matching" in plan
    assert "G8 remains closed" in plan
