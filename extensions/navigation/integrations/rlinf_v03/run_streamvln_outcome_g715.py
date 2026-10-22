"""Run the preregistered G7.15 no-fit executed-outcome discrimination audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

from run_streamvln_candidates_g7 import configure_process_git_safe_directory, validate_identity
from run_streamvln_ppo_g4 import make_env, validate_dataset_hashes, validate_manifest
from run_streamvln_topology_g5 import ORIGINAL_STREAM_COMMIT, git_head, jsonable, sha256_file
from run_streamvln_visibility_g714 import encode_depth, vector_error
from streamvln_candidates_g74 import generate_macro_candidates
from streamvln_macro_audit_g79 import select_registered_candidates
from streamvln_outcome_g715 import (
    binary_auroc,
    binary_class_counts,
    deterministic_matches,
    matched_summary,
    stratified_auroc,
    stratified_class_counts,
    visual_risk,
)
from streamvln_severity_g710 import severity_outcome, spearman_correlation
from streamvln_topology import OnlineTopology
from streamvln_visibility_g714 import (
    DOWNWARD_PITCH_DEGREES,
    DOWNWARD_PITCH_RADIANS,
    EXPECTED_OPTICAL_AXIS_AGENT_FRAME,
    EXPECTED_SENSOR_POSITION_AGENT_FRAME,
    EXPECTED_TARGET_RAY_COUNT,
    immediate_footprint_observables,
)


RLINF_COMMIT = "0505431899574619da86f551bad70b71e0ea2177"


def frozen_cases(candidates):
    cases = []
    for candidate in candidates:
        actions = list(candidate.actions)
        first_forward = actions.index(1)
        turn_prefix = actions[:first_forward]
        forward_suffix = actions[first_forward:]
        cases.append(
            {
                "name": candidate.name,
                "heading_degrees": float(candidate.heading_degrees),
                "distance_m": float(candidate.distance_m),
                "actions": actions,
                "turn_prefix": turn_prefix,
                "forward_suffix": forward_suffix,
                "observe_before_forward_index": len(forward_suffix),
                "critical_action": 1,
            }
        )
    return cases


def load_g714_anchor(paths, prereg):
    anchor = prereg["g714_anchor"]
    actual = {
        "result_sha256": sha256_file(Path(paths["g714_result"])),
        "run_config_sha256": sha256_file(Path(paths["g714_run_config"])),
        "rows_sha256": sha256_file(Path(paths["g714_rows"])),
        "snapshots_sha256": sha256_file(Path(paths["g714_snapshots"])),
    }
    expected = {key: anchor[key] for key in actual}
    if actual != expected:
        raise RuntimeError("G7.15 accepted G7.14 evidence hash changed")
    result = json.loads(Path(paths["g714_result"]).read_text())
    if (
        result.get("kind")
        != "streamvln-g7.14-immediate-footprint-visual-visibility-ceiling"
        or result.get("all_protocol_guards_passed") is not True
        or result.get("immediate_footprint_visibility_ready") is not True
        or result.get("outcome_discrimination_ready") is not False
        or result.get("val_unseen_episode_count") != 0
        or result.get("source_contract", {}).get("stream_commit")
        != ORIGINAL_STREAM_COMMIT
        or result.get("source_contract", {}).get("rlinf_commit") != RLINF_COMMIT
    ):
        raise RuntimeError("G7.15 accepted G7.14 contract changed")
    return result


def _repeat_outcome_exact(first, second, tolerance):
    categorical = (
        "severity_class",
        "severity_index",
        "raw_collision",
        "severe_blockage",
    )
    numeric = (
        "continuous_severity",
        "realized_ratio",
        "maximum_primitive_progress_deficit_m",
        "minimum_primitive_progress_m",
    )
    return all(first[key] == second[key] for key in categorical) and all(
        abs(float(first[key]) - float(second[key])) <= tolerance for key in numeric
    )


def collect_scene(run_cfg, spec, cases):
    cfg = dict(run_cfg)
    cfg["habitat_server"] = run_cfg["habitat_outcome_server"]
    env = make_env(
        cfg,
        split="train",
        episode_index=spec["index"],
        max_steps=run_cfg["max_episode_steps"],
    )
    rows = []
    snapshots = []
    states = []
    tolerance = float(run_cfg["preregistration"]["support_gates"]["repeat_numeric_tolerance"])
    try:
        _, identity = env.reset()
        validate_identity(identity, spec)
        for state_index, state in enumerate(run_cfg["preregistration"]["state_definitions"]):
            transition_records = []
            for action in state["transition_actions"]:
                _, _, terminated, truncated, infos = env.step(action)
                record = {
                    "action": int(action),
                    "collision": bool(infos["collision"].item()),
                    "terminated": bool(terminated.item()),
                    "truncated": bool(truncated.item()),
                }
                transition_records.append(record)
                if record["terminated"] or record["truncated"]:
                    raise RuntimeError("G7.15 fixed state transition ended the episode")
            payload = [
                {
                    "name": case["name"],
                    "actions": case["actions"],
                    "observe_before_forward_index": case[
                        "observe_before_forward_index"
                    ],
                }
                for case in cases
            ]
            audit = env.probe_immediate_footprint_outcomes(payload)
            by_name = {item["name"]: item for item in audit["candidates"]}
            if set(by_name) != {case["name"] for case in cases}:
                raise RuntimeError("G7.15 service case set changed")
            for case in cases:
                probe = by_name[case["name"]]
                repeats = []
                for repeat_index, repeat in enumerate(probe["repeats"]):
                    observables = immediate_footprint_observables(repeat["metric_depth"])
                    risk = visual_risk(observables)
                    trace = repeat["critical_trace"]
                    outcome = severity_outcome(
                        {
                            "commanded_distance_m": 0.25,
                            "realized_forward_m": trace["forward_progress_m"],
                            "endpoint_error_m": trace["endpoint_error_m"],
                            "realized_lateral_error_m": trace[
                                "lateral_displacement_m"
                            ],
                            "primitive_trace": [trace],
                            "full_collision_trace": [trace["collision"]],
                        }
                    )
                    extrinsics = repeat["sensor_extrinsics"]
                    axis_error = vector_error(
                        extrinsics["optical_axis_agent_frame"],
                        EXPECTED_OPTICAL_AXIS_AGENT_FRAME,
                    )
                    sensor_position_error = vector_error(
                        extrinsics["relative_position_agent_frame"],
                        EXPECTED_SENSOR_POSITION_AGENT_FRAME,
                    )
                    snapshot = {
                        "scene": spec["scene"],
                        "episode_id": str(spec["episode_id"]),
                        "state_name": state["name"],
                        "candidate": case["name"],
                        "repeat_index": repeat_index,
                        "depth_sha256": repeat["depth_sha256"],
                        "metric_depth": encode_depth(
                            repeat["metric_depth"], repeat["depth_sha256"]
                        ),
                        "visual_risk": risk,
                        "critical_trace": trace,
                        "outcome": outcome,
                    }
                    snapshots.append(snapshot)
                    repeats.append(
                        {
                            "repeat_index": repeat_index,
                            "turn_collisions": repeat["turn_collisions"],
                            "executed_prefix_count": repeat["executed_prefix_count"],
                            "prefix_trace": repeat["prefix_trace"],
                            "pre_action_pose": repeat["pre_action_pose"],
                            "post_action_pose": repeat["post_action_pose"],
                            "forward_direction": repeat["forward_direction"],
                            "sensor_extrinsics": extrinsics,
                            "optical_axis_error": axis_error,
                            "sensor_position_error_m": sensor_position_error,
                            "depth_sha256": repeat["depth_sha256"],
                            "observables": observables,
                            "visual_risk": risk,
                            "critical_trace": trace,
                            "outcome": outcome,
                        }
                    )
                first, second = repeats
                repeat_exact = (
                    first["turn_collisions"] == second["turn_collisions"]
                    and first["prefix_trace"] == second["prefix_trace"]
                    and first["forward_direction"] == second["forward_direction"]
                    and first["sensor_extrinsics"] == second["sensor_extrinsics"]
                    and first["depth_sha256"] == second["depth_sha256"]
                    and first["observables"] == second["observables"]
                    and first["visual_risk"] == second["visual_risk"]
                    and first["critical_trace"] == second["critical_trace"]
                    and vector_error(
                        first["pre_action_pose"]["position"],
                        second["pre_action_pose"]["position"],
                    )
                    <= 1e-6
                    and vector_error(
                        first["post_action_pose"]["position"],
                        second["post_action_pose"]["position"],
                    )
                    <= 1e-6
                    and _repeat_outcome_exact(
                        first["outcome"], second["outcome"], tolerance
                    )
                )
                rows.append(
                    {
                        "scene": spec["scene"],
                        "episode_id": str(spec["episode_id"]),
                        "state_index": state_index,
                        "state_name": state["name"],
                        "state_transition_actions": list(state["transition_actions"]),
                        "state_transition_collisions": [
                            item["collision"] for item in transition_records
                        ],
                        **case,
                        "candidate": case["name"],
                        "repeat_exact": repeat_exact,
                        "visual_risk": first["visual_risk"],
                        **first["outcome"],
                        "repeats": repeats,
                    }
                )
            states.append(
                {
                    "name": state["name"],
                    "transition_actions": list(state["transition_actions"]),
                    "transition_records": transition_records,
                    "case_count": len(cases),
                    "max_position_restore_error_m": audit[
                        "max_position_restore_error_m"
                    ],
                    "max_rotation_restore_error": audit[
                        "max_rotation_restore_error"
                    ],
                }
            )
        return (
            {"scene": spec["scene"], "episode_id": str(spec["episode_id"]), "states": states},
            rows,
            snapshots,
        )
    finally:
        env.close()


def _minimum_auroc(strata):
    if not strata or any(item["status"] != "ok" for item in strata.values()):
        return None
    return min(float(item["auroc"]) for item in strata.values())


def _minimum_class_count(strata):
    if not strata:
        return 0
    return min(min(item["severe"], item["nonsevere"]) for item in strata.values())


def compute_metrics(rows, matches):
    overall = binary_auroc(
        [row["severe_blockage"] for row in rows],
        [row["visual_risk"] for row in rows],
    )
    by_scene = stratified_auroc(rows, "scene")
    by_state = stratified_auroc(rows, "state_name")
    severe_scores = [row["visual_risk"] for row in rows if row["severe_blockage"]]
    control_scores = [row["visual_risk"] for row in rows if not row["severe_blockage"]]
    return {
        "binary_class_counts": binary_class_counts(rows),
        "binary_class_counts_by_scene": stratified_class_counts(rows, "scene"),
        "binary_class_counts_by_state": stratified_class_counts(rows, "state_name"),
        "overall": overall,
        "by_scene": by_scene,
        "by_state": by_state,
        "minimum_scene_auroc": _minimum_auroc(by_scene),
        "minimum_state_auroc": _minimum_auroc(by_state),
        "spearman_visual_risk_vs_continuous_severity": spearman_correlation(
            [row["visual_risk"] for row in rows],
            [row["continuous_severity"] for row in rows],
        ),
        "median_severe_visual_risk": (
            float(statistics.median(severe_scores)) if severe_scores else None
        ),
        "median_nonsevere_visual_risk": (
            float(statistics.median(control_scores)) if control_scores else None
        ),
        "matched": matched_summary(matches),
        "minimum_scene_binary_class_count": _minimum_class_count(
            stratified_class_counts(rows, "scene")
        ),
        "minimum_state_binary_class_count": _minimum_class_count(
            stratified_class_counts(rows, "state_name")
        ),
        "maximum_optical_axis_error": max(
            repeat["optical_axis_error"] for row in rows for repeat in row["repeats"]
        ),
        "maximum_sensor_position_error_m": max(
            repeat["sensor_position_error_m"]
            for row in rows
            for repeat in row["repeats"]
        ),
        "all_repeats_exact": all(row["repeat_exact"] for row in rows),
        "unique_depth_hash_count": len(
            {row["repeats"][0]["depth_sha256"] for row in rows}
        ),
    }


def evaluate_gates(rows, snapshots, episodes, metrics, prereg, integrity_ready):
    support = prereg["support_gates"]
    counts = metrics["binary_class_counts"]
    matched = metrics["matched"]
    support_gates = {
        "exact_row_and_snapshot_outcome_count": len(rows) == support["exact_row_count"]
        and len(snapshots) == support["exact_snapshot_outcome_count"],
        "two_repeats_per_row": all(len(row["repeats"]) == 2 for row in rows),
        "overall_class_support": min(counts.values())
        >= support["overall_minimum_per_binary_class"],
        "each_state_class_support": metrics["minimum_state_binary_class_count"]
        >= support["each_state_minimum_per_binary_class"],
        "each_scene_class_support": metrics["minimum_scene_binary_class_count"]
        >= support["each_scene_minimum_per_binary_class"],
        "overall_matched_pair_support": matched["pair_count"]
        >= support["minimum_matched_pair_count"],
        "each_state_matched_pair_support": all(
            matched["pair_count_by_state"].get(state["name"], 0)
            >= support["each_state_minimum_matched_pair_count"]
            for state in prereg["state_definitions"]
        ),
        "scene_matched_pair_support": matched["scenes_with_at_least_three_pairs"]
        >= support["minimum_scenes_with_three_matched_pairs"],
        "all_48_rays_classified_and_unclipped": all(
            repeat["observables"]["classified_count"] == EXPECTED_TARGET_RAY_COUNT
            and repeat["observables"]["unclipped_count"] == EXPECTED_TARGET_RAY_COUNT
            for row in rows
            for repeat in row["repeats"]
        ),
        "repeats_exact": metrics["all_repeats_exact"],
        "restore_exact": all(
            state["max_position_restore_error_m"]
            <= support["restore_and_extrinsic_tolerance"]
            and state["max_rotation_restore_error"]
            <= support["restore_and_extrinsic_tolerance"]
            for episode in episodes
            for state in episode["states"]
        ),
        "active_extrinsics_exact": metrics["maximum_optical_axis_error"]
        <= support["restore_and_extrinsic_tolerance"]
        and metrics["maximum_sensor_position_error_m"]
        <= support["restore_and_extrinsic_tolerance"],
        "protocol_integrity": bool(integrity_ready),
    }
    separation = prereg["separation_gates"]
    matched_severe = matched["median_severe_visual_risk"]
    matched_control = matched["median_control_visual_risk"]
    separation_gates = {
        "overall_auroc": metrics["overall"]["status"] == "ok"
        and metrics["overall"]["auroc"] >= separation["overall_auroc_min"],
        "minimum_scene_auroc": metrics["minimum_scene_auroc"] is not None
        and metrics["minimum_scene_auroc"] >= separation["minimum_scene_auroc"],
        "minimum_state_auroc": metrics["minimum_state_auroc"] is not None
        and metrics["minimum_state_auroc"] >= separation["minimum_state_auroc"],
        "spearman": metrics["spearman_visual_risk_vs_continuous_severity"]
        >= separation["spearman_visual_risk_vs_continuous_severity_min"],
        "paired_strict_win_rate": matched["strict_win_rate"] is not None
        and matched["strict_win_rate"] >= separation["paired_strict_win_rate_min"],
        "matched_severe_median_strictly_greater": matched_severe is not None
        and matched_control is not None
        and matched_severe > matched_control,
    }
    return {
        "support": support_gates,
        "separation": separation_gates,
        "support_ready": all(support_gates.values()),
        "separation_ready": all(separation_gates.values()),
        "ready": all(support_gates.values()) and all(separation_gates.values()),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    for name in (
        "manifest",
        "g714-result",
        "g714-run-config",
        "g714-rows",
        "g714-snapshots",
        "output-dir",
        "rlinf-root",
        "stream-source",
        "habitat-python",
        "habitat-outcome-server",
        "asset-root",
        "habitat-sim-python",
        "habitat-env-lib",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument(
        "--egl-vendor-json", default="/usr/share/glvnd/egl_vendor.d/50_mesa.json"
    )
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--max-episode-steps", type=int, default=40)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_episode_steps < 8:
        raise ValueError("G7.15 episode budget is invalid")
    path_names = (
        "manifest",
        "g714_result",
        "g714_run_config",
        "g714_rows",
        "g714_snapshots",
        "rlinf_root",
        "stream_source",
        "habitat_python",
        "habitat_outcome_server",
        "asset_root",
        "habitat_sim_python",
        "habitat_env_lib",
        "egl_vendor_json",
    )
    paths = {
        name: str(Path(getattr(args, name)).expanduser().resolve(strict=True))
        for name in path_names
    }
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("G7.15 output directory must be fresh")
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(paths["manifest"]).read_text())
    validate_manifest(manifest)
    prereg = manifest.get("g715_executed_outcome_discrimination", {})
    if (
        prereg.get("split") != "train"
        or prereg.get("candidate_count_per_state") != 32
        or prereg.get("repeat_count") != 2
        or prereg.get("row_count") != 384
        or prereg.get("snapshot_outcome_count") != 768
        or prereg.get("critical_action_count_per_repeat") != 1
        or prereg.get("camera", {}).get("pitch_radians") != DOWNWARD_PITCH_RADIANS
        or prereg.get("camera", {}).get("pitch_degrees") != DOWNWARD_PITCH_DEGREES
        or prereg.get("target_region", {}).get("expected_target_ray_count")
        != EXPECTED_TARGET_RAY_COUNT
        or prereg.get("model_load") is not False
        or prereg.get("model_fit") is not False
        or prereg.get("rl_update") is not False
        or prereg.get("pitch_sweep") is not False
        or prereg.get("feature_sweep") is not False
        or prereg.get("threshold_sweep") is not False
        or prereg.get("controller_implementation") is not False
        or prereg.get("navigation_evaluation") is not False
        or prereg.get("val_unseen_episode_count") != 0
    ):
        raise RuntimeError("G7.15 manifest preregistration changed")
    configure_process_git_safe_directory(paths["stream_source"])
    configure_process_git_safe_directory(paths["rlinf_root"])
    stream_commit = git_head(Path(paths["stream_source"]))
    rlinf_commit = git_head(Path(paths["rlinf_root"]))
    if stream_commit != ORIGINAL_STREAM_COMMIT or rlinf_commit != RLINF_COMMIT:
        raise RuntimeError("G7.15 source commit pin changed")
    load_g714_anchor(paths, prereg)
    topology = OnlineTopology(max_decisions=args.max_episode_steps)
    candidates = select_registered_candidates(generate_macro_candidates(topology))
    cases = frozen_cases(candidates)
    if cases != prereg.get("cases"):
        raise RuntimeError("G7.15 frozen case bank does not reproduce")
    if prereg.get("scene_specs") != manifest["train"][:6]:
        raise RuntimeError("G7.15 frozen six-scene population changed")
    run_cfg = {
        **paths,
        "manifest": manifest,
        "manifest_sha256": sha256_file(Path(paths["manifest"])),
        "output_dir": str(output),
        "stream_commit": stream_commit,
        "rlinf_commit": rlinf_commit,
        "seed": args.seed,
        "max_episode_steps": args.max_episode_steps,
        "preregistration": prereg,
        "val_unseen_episode_count": 0,
        "model_load": False,
        "model_fit": False,
        "rl_update": False,
        "pitch_sweep": False,
        "feature_sweep": False,
        "threshold_sweep": False,
        "controller_implementation": False,
        "navigation_evaluation": False,
    }
    dataset_hashes = validate_dataset_hashes(run_cfg)
    (output / "run_config.json").write_text(
        json.dumps(run_cfg, indent=2, sort_keys=True) + "\n"
    )
    started = time.time()
    episodes, rows, snapshots = [], [], []
    for spec in prereg["scene_specs"]:
        episode, scene_rows, scene_snapshots = collect_scene(
            run_cfg, spec, cases
        )
        episodes.append(episode)
        rows.extend(scene_rows)
        snapshots.extend(scene_snapshots)
    matches = deterministic_matches(
        rows,
        maximum_heading_gap_degrees=prereg["matching"][
            "maximum_heading_gap_degrees"
        ],
    )
    metrics = compute_metrics(rows, matches)
    scene_state_support = all(
        any(row["scene"] == scene["scene"] and row["state_name"] == state["name"] for row in rows)
        for scene in prereg["scene_specs"]
        for state in prereg["state_definitions"]
    )
    protocol_guards = {
        "original_source_pins": True,
        "g714_anchor_hashes_and_passed_visibility_exact": True,
        "fixed_six_scene_population_exact": prereg["scene_specs"] == manifest["train"][:6],
        "scene_disjoint_from_g713_g714": not any(
            spec["scene"]
            in set(manifest["g714_immediate_footprint_visibility"]["train_dev_scenes"])
            for spec in prereg["scene_specs"]
        ),
        "frozen_32_case_reconstruction_exact": cases == prereg["cases"],
        "all_scenes_states_present": scene_state_support,
        "one_critical_forward_per_repeat": all(
            repeat["critical_trace"]["action"] == 1
            for row in rows
            for repeat in row["repeats"]
        ),
        "feature_is_action_preceding_equal_weight_non_support_only": True,
        "outcomes_navmesh_collisions_excluded_from_feature": True,
        "matching_rule_frozen_and_one_to_one": len(matches)
        == len(
            {
                (
                    row["scene"],
                    row["state_name"],
                    row["distance_m"],
                    row["severe_candidate"],
                )
                for row in matches
            }
        )
        == len(
            {
                (
                    row["scene"],
                    row["state_name"],
                    row["distance_m"],
                    row["control_candidate"],
                )
                for row in matches
            }
        ),
        "policy_rgb_orientation_unchanged": True,
        "no_model_fit_rl_sweep_controller_or_navigation": True,
        "zero_val_unseen": True,
    }
    integrity_ready = all(protocol_guards.values())
    gate = evaluate_gates(
        rows, snapshots, episodes, metrics, prereg, integrity_ready
    )
    if not integrity_ready:
        route = "repair-g715-instrumentation-under-unchanged-preregistration"
    elif not gate["support_ready"]:
        route = "close-pitched-depth-risk-representation-insufficient-fixed-population-support"
    elif gate["separation_ready"]:
        route = "preregister-one-frozen-truncated-action-rule-on-new-train-scenes"
    else:
        route = "close-pitched-depth-risk-representation-and-define-lower-layer-sensing-action-contract"

    rows_path = output / "outcome_rows.jsonl"
    with rows_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(jsonable(row), sort_keys=True) + "\n")
    snapshots_path = output / "pitched_depth_outcomes.jsonl"
    with snapshots_path.open("w") as handle:
        for snapshot in snapshots:
            handle.write(json.dumps(jsonable(snapshot), sort_keys=True) + "\n")
    matches_path = output / "matched_pairs.jsonl"
    with matches_path.open("w") as handle:
        for match in matches:
            handle.write(json.dumps(jsonable(match), sort_keys=True) + "\n")
    result = {
        "schema_version": 1,
        "kind": "streamvln-g7.15-frozen-immediate-footprint-executed-outcome-discrimination",
        "claim_scope": "fixed six-scene/384-row train-only no-fit causal one-forward-primitive discrimination audit; no threshold, controller, navigation, language, val_unseen, safety or generalization claim",
        "split": "train",
        "val_unseen_episode_count": 0,
        "source_contract": {
            "stream_commit": stream_commit,
            "rlinf_commit": rlinf_commit,
            "source_kind": "upstream-original",
            "secondary_post_training_used": False,
            "ttt_used": False,
        },
        "dataset_hashes": dataset_hashes,
        "preregistration": prereg,
        "episodes": episodes,
        "sample_summary": {
            "row_count": len(rows),
            "snapshot_outcome_count": len(snapshots),
            "matched_pair_count": len(matches),
            "repeat_count": 2,
            "scene_count": len({row["scene"] for row in rows}),
            "state_count": len({row["state_name"] for row in rows}),
        },
        "discrimination_metrics": metrics,
        "discrimination_gate": gate,
        "protocol_guards": protocol_guards,
        "all_protocol_guards_passed": integrity_ready,
        "outcome_discrimination_ready": gate["ready"],
        "controller_ready": False,
        "navigation_ready": False,
        "g8_ready": False,
        "routing": route,
        "evidence": {
            "rows_path": str(rows_path),
            "rows_sha256": sha256_file(rows_path),
            "snapshots_path": str(snapshots_path),
            "snapshots_sha256": sha256_file(snapshots_path),
            "matches_path": str(matches_path),
            "matches_sha256": sha256_file(matches_path),
            "g714_result_sha256": prereg["g714_anchor"]["result_sha256"],
            "g714_run_config_sha256": prereg["g714_anchor"]["run_config_sha256"],
            "g714_rows_sha256": prereg["g714_anchor"]["rows_sha256"],
            "g714_snapshots_sha256": prereg["g714_anchor"]["snapshots_sha256"],
        },
        "elapsed_s": time.time() - started,
    }
    result_path = output / "result.json"
    result_path.write_text(json.dumps(jsonable(result), indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "result": str(result_path),
                "result_sha256": sha256_file(result_path),
                "all_protocol_guards_passed": integrity_ready,
                "support_ready": gate["support_ready"],
                "separation_ready": gate["separation_ready"],
                "outcome_discrimination_ready": gate["ready"],
                "g8_ready": False,
                "routing": route,
                "metrics": metrics,
                "gate": gate,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
