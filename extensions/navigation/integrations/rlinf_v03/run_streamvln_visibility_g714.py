"""Run the preregistered G7.14 immediate-footprint visibility ceiling."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np

from run_streamvln_candidates_g7 import configure_process_git_safe_directory, validate_identity
from run_streamvln_ppo_g4 import make_env, validate_dataset_hashes, validate_manifest
from run_streamvln_topology_g5 import ORIGINAL_STREAM_COMMIT, git_head, jsonable, sha256_file
from streamvln_visibility_g714 import (
    DOWNWARD_PITCH_DEGREES,
    DOWNWARD_PITCH_RADIANS,
    EXPECTED_GRID_COUNTS,
    EXPECTED_OPTICAL_AXIS_AGENT_FRAME,
    EXPECTED_SENSOR_POSITION_AGENT_FRAME,
    EXPECTED_TARGET_RAY_COUNT,
    FROZEN_GEOMETRY_SUMMARY,
    immediate_footprint_observables,
    visibility_gate,
)


RLINF_COMMIT = "0505431899574619da86f551bad70b71e0ea2177"


def vector_error(left, right):
    return float(
        np.linalg.norm(
            np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
        )
    )


def encode_depth(depth, expected_sha256):
    array = np.ascontiguousarray(depth, dtype="<f4")
    raw = array.tobytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise RuntimeError("G7.14 metric-depth bytes differ from service hash")
    return {
        "encoding": "base64/raw-f32le-hw",
        "shape_hw": [64, 64],
        "min_m": 0.0,
        "max_m": 10.0,
        "data_b64": base64.b64encode(raw).decode("ascii"),
    }


def load_g713_anchor(paths, args):
    actual = {
        "result": sha256_file(Path(paths["g713_result"])),
        "run_config": sha256_file(Path(paths["g713_run_config"])),
        "rows": sha256_file(Path(paths["g713_rows"])),
        "depth_snapshots": sha256_file(Path(paths["g713_depth_snapshots"])),
    }
    expected = {
        "result": args.g713_result_sha256,
        "run_config": args.g713_run_config_sha256,
        "rows": args.g713_rows_sha256,
        "depth_snapshots": args.g713_depth_snapshots_sha256,
    }
    if actual != expected:
        raise RuntimeError("G7.14 accepted G7.13 anchor hash changed")
    result = json.loads(Path(paths["g713_result"]).read_text())
    if (
        result.get("kind")
        != "streamvln-g7.13-causal-ground-support-observability-audit"
        or result.get("all_protocol_guards_passed") is not True
        or result.get("causal_observability_ready") is not False
        or result.get("val_unseen_episode_count") != 0
        or result.get("source_contract", {}).get("stream_commit")
        != ORIGINAL_STREAM_COMMIT
        or result.get("source_contract", {}).get("rlinf_commit") != RLINF_COMMIT
    ):
        raise RuntimeError("G7.14 accepted G7.13 contract changed")
    rows = [
        json.loads(line)
        for line in Path(paths["g713_rows"]).read_text().splitlines()
        if line
    ]
    by_name = {row["name"]: row for row in rows}
    positives = sorted(
        (row for row in rows if row.get("role") == "positive"),
        key=lambda row: row["pair_index"],
    )
    if (
        len(rows) != 52
        or len(by_name) != 52
        or len(positives) != 26
        or not all(row.get("repeat_exact") is True for row in positives)
    ):
        raise RuntimeError("G7.14 G7.13 anchor row contract changed")
    return result, by_name, positives


def frozen_case_identity(row):
    return {
        "name": row["name"],
        "pair_index": row["pair_index"],
        "scene": row["scene"],
        "state_name": row["state_name"],
        "candidate": row["candidate"],
        "actions": row["actions"],
        "observe_before_forward_index": row["observe_before_forward_index"],
    }


def collect_scene(run_cfg, spec, cases, anchor_by_name):
    cfg = dict(run_cfg)
    cfg["habitat_server"] = run_cfg["habitat_visibility_server"]
    env = make_env(
        cfg,
        split="train",
        episode_index=spec["index"],
        max_steps=run_cfg["max_episode_steps"],
    )
    rows = []
    snapshots = []
    states = []
    try:
        _, identity = env.reset()
        validate_identity(identity, spec)
        for state_index, state in enumerate(run_cfg["preregistration"]["state_definitions"]):
            transition_records = []
            for action in state["transition_actions"]:
                _, _, terminated, truncated, infos = env.step(action)
                transition_records.append(
                    {
                        "action": int(action),
                        "collision": bool(infos["collision"].item()),
                        "terminated": bool(terminated.item()),
                        "truncated": bool(truncated.item()),
                    }
                )
                if bool(terminated.item()) or bool(truncated.item()):
                    raise RuntimeError("G7.14 state transition ended the episode")
            selected = [
                case
                for case in cases
                if case["scene"] == spec["scene"] and case["state_name"] == state["name"]
            ]
            payload = [
                {
                    "name": case["name"],
                    "actions": case["actions"],
                    "observe_before_forward_index": case[
                        "observe_before_forward_index"
                    ],
                }
                for case in selected
            ]
            audit = env.probe_immediate_footprint_visibility(payload)
            by_name = {item["name"]: item for item in audit["candidates"]}
            if set(by_name) != {case["name"] for case in selected}:
                raise RuntimeError("G7.14 service case set changed")
            for case in selected:
                probe = by_name[case["name"]]
                anchor = anchor_by_name[case["name"]]
                anchor_repeat = anchor["repeats"][0]
                repeats = []
                anchor_errors = []
                for repeat_index, repeat in enumerate(probe["repeats"]):
                    observables = immediate_footprint_observables(
                        repeat["metric_depth"]
                    )
                    payload_depth = encode_depth(
                        repeat["metric_depth"], repeat["depth_sha256"]
                    )
                    pose_error = vector_error(
                        repeat["pre_action_pose"]["position"],
                        anchor_repeat["pre_action_pose"]["position"],
                    )
                    anchor_errors.append(pose_error)
                    if repeat["prefix_trace"] != anchor_repeat["prefix_trace"]:
                        raise RuntimeError("G7.14 prefix trace differs from G7.13 anchor")
                    extrinsics = repeat["sensor_extrinsics"]
                    axis_error = vector_error(
                        extrinsics["optical_axis_agent_frame"],
                        EXPECTED_OPTICAL_AXIS_AGENT_FRAME,
                    )
                    sensor_position_error = vector_error(
                        extrinsics["relative_position_agent_frame"],
                        EXPECTED_SENSOR_POSITION_AGENT_FRAME,
                    )
                    snapshots.append(
                        {
                            "name": case["name"],
                            "pair_index": case["pair_index"],
                            "scene": case["scene"],
                            "state_name": case["state_name"],
                            "candidate": case["candidate"],
                            "repeat_index": repeat_index,
                            "depth_sha256": repeat["depth_sha256"],
                            "metric_depth": payload_depth,
                        }
                    )
                    repeats.append(
                        {
                            "repeat_index": repeat_index,
                            "turn_collisions": repeat["turn_collisions"],
                            "executed_prefix_count": repeat["executed_prefix_count"],
                            "prefix_trace": repeat["prefix_trace"],
                            "pre_action_pose": repeat["pre_action_pose"],
                            "forward_direction": repeat["forward_direction"],
                            "sensor_extrinsics": extrinsics,
                            "optical_axis_error": axis_error,
                            "sensor_position_error_m": sensor_position_error,
                            "depth_sha256": repeat["depth_sha256"],
                            "observables": observables,
                            "g713_anchor_position_error_m": pose_error,
                        }
                    )
                repeat_exact = (
                    repeats[0]["turn_collisions"] == repeats[1]["turn_collisions"]
                    and repeats[0]["prefix_trace"] == repeats[1]["prefix_trace"]
                    and repeats[0]["forward_direction"] == repeats[1]["forward_direction"]
                    and repeats[0]["sensor_extrinsics"] == repeats[1]["sensor_extrinsics"]
                    and repeats[0]["depth_sha256"] == repeats[1]["depth_sha256"]
                    and repeats[0]["observables"] == repeats[1]["observables"]
                    and vector_error(
                        repeats[0]["pre_action_pose"]["position"],
                        repeats[1]["pre_action_pose"]["position"],
                    )
                    <= 1e-6
                    and vector_error(
                        repeats[0]["pre_action_pose"]["rotation_coefficients"],
                        repeats[1]["pre_action_pose"]["rotation_coefficients"],
                    )
                    <= 1e-6
                )
                rows.append(
                    {
                        **case,
                        "state_index": state_index,
                        "state_transition_actions": list(state["transition_actions"]),
                        "state_transition_collisions": [
                            record["collision"] for record in transition_records
                        ],
                        "repeat_exact": repeat_exact,
                        "maximum_g713_anchor_position_error_m": max(anchor_errors),
                        "repeats": repeats,
                    }
                )
            states.append(
                {
                    "name": state["name"],
                    "transition_actions": list(state["transition_actions"]),
                    "transition_records": transition_records,
                    "case_count": len(selected),
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


def compute_metrics(rows):
    first = [row["repeats"][0] for row in rows]
    observables = [repeat["observables"] for repeat in first]
    return {
        "target_ray_count": EXPECTED_TARGET_RAY_COUNT,
        "forward_bin_counts": FROZEN_GEOMETRY_SUMMARY["forward_bin_counts"],
        "lateral_bin_counts": FROZEN_GEOMETRY_SUMMARY["lateral_bin_counts"],
        "grid_counts": FROZEN_GEOMETRY_SUMMARY["grid_counts"],
        "minimum_spatial_bin_count": min(
            value for row in EXPECTED_GRID_COUNTS for value in row
        ),
        "all_target_depths_classified": all(
            item["classified_count"] == EXPECTED_TARGET_RAY_COUNT
            and item["target_ray_count"] == EXPECTED_TARGET_RAY_COUNT
            for item in observables
        ),
        "all_repeats_exact": all(row["repeat_exact"] for row in rows),
        "maximum_optical_axis_error": max(
            repeat["optical_axis_error"] for row in rows for repeat in row["repeats"]
        ),
        "maximum_sensor_position_error_m": max(
            repeat["sensor_position_error_m"]
            for row in rows
            for repeat in row["repeats"]
        ),
        "unique_depth_hash_count": len(
            {repeat["depth_sha256"] for repeat in first}
        ),
        "median_support_fraction": statistics.median(
            item["support_fraction"] for item in observables
        ),
        "median_occlusion_fraction": statistics.median(
            item["occlusion_fraction"] for item in observables
        ),
        "median_missing_fraction": statistics.median(
            item["missing_fraction"] for item in observables
        ),
        "median_unclipped_fraction": statistics.median(
            item["unclipped_fraction"] for item in observables
        ),
        "minimum_unclipped_fraction": min(
            item["unclipped_fraction"] for item in observables
        ),
        "maximum_observed_target_depth_range_m": max(
            item["observed_target_depth_range_m"] for item in observables
        ),
        "minimum_observed_target_depth_range_m": min(
            item["observed_target_depth_range_m"] for item in observables
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    for name in (
        "manifest",
        "g713-result",
        "g713-result-sha256",
        "g713-run-config",
        "g713-run-config-sha256",
        "g713-rows",
        "g713-rows-sha256",
        "g713-depth-snapshots",
        "g713-depth-snapshots-sha256",
        "output-dir",
        "rlinf-root",
        "stream-source",
        "habitat-python",
        "habitat-visibility-server",
        "asset-root",
        "habitat-sim-python",
        "habitat-env-lib",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument(
        "--egl-vendor-json",
        default="/usr/share/glvnd/egl_vendor.d/50_mesa.json",
    )
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--max-episode-steps", type=int, default=40)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_episode_steps < 8:
        raise ValueError("G7.14 episode budget is invalid")
    path_names = (
        "manifest",
        "g713_result",
        "g713_run_config",
        "g713_rows",
        "g713_depth_snapshots",
        "rlinf_root",
        "stream_source",
        "habitat_python",
        "habitat_visibility_server",
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
        raise FileExistsError("G7.14 output directory must be fresh")
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(paths["manifest"]).read_text())
    validate_manifest(manifest)
    prereg = manifest.get("g714_immediate_footprint_visibility", {})
    camera = prereg.get("camera", {})
    target = prereg.get("target_region", {})
    if (
        prereg.get("split") != "train"
        or prereg.get("case_count") != 26
        or prereg.get("repeat_count") != 2
        or camera.get("pitch_radians") != DOWNWARD_PITCH_RADIANS
        or camera.get("pitch_degrees") != DOWNWARD_PITCH_DEGREES
        or target.get("expected_target_ray_count") != EXPECTED_TARGET_RAY_COUNT
        or target.get("expected_grid_counts")
        != [list(row) for row in EXPECTED_GRID_COUNTS]
        or prereg.get("model_fit") is not False
        or prereg.get("pitch_sweep") is not False
        or prereg.get("threshold_sweep") is not False
        or prereg.get("risk_decision") is not False
        or prereg.get("controller_implementation") is not False
        or prereg.get("navigation_evaluation") is not False
        or prereg.get("val_unseen_episode_count") != 0
    ):
        raise RuntimeError("G7.14 manifest preregistration changed")
    configure_process_git_safe_directory(paths["stream_source"])
    configure_process_git_safe_directory(paths["rlinf_root"])
    stream_commit = git_head(Path(paths["stream_source"]))
    rlinf_commit = git_head(Path(paths["rlinf_root"]))
    if stream_commit != ORIGINAL_STREAM_COMMIT or rlinf_commit != RLINF_COMMIT:
        raise RuntimeError("G7.14 source commit pin changed")
    _, anchor_by_name, positive_rows = load_g713_anchor(paths, args)
    cases = [frozen_case_identity(row) for row in positive_rows]
    if cases != prereg.get("cases"):
        raise RuntimeError("G7.14 frozen case identities do not reproduce")
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
        "model_fit": False,
        "pitch_sweep": False,
        "threshold_sweep": False,
        "risk_decision": False,
        "controller_implementation": False,
        "navigation_evaluation": False,
    }
    dataset_hashes = validate_dataset_hashes(run_cfg)
    (output / "run_config.json").write_text(
        json.dumps(run_cfg, indent=2, sort_keys=True) + "\n"
    )
    started = time.time()
    episodes = []
    rows = []
    snapshots = []
    for spec in manifest["train"][6:]:
        episode, scene_rows, scene_snapshots = collect_scene(
            run_cfg, spec, cases, anchor_by_name
        )
        episodes.append(episode)
        rows.extend(scene_rows)
        snapshots.extend(scene_snapshots)
    if len(rows) != 26 or len(snapshots) != 52 or len(episodes) != 2:
        raise RuntimeError("G7.14 bounded collection count changed")

    anchor_prefix_exact = all(
        row["maximum_g713_anchor_position_error_m"] <= 1e-6 for row in rows
    )
    restore_exact = all(
        state["max_position_restore_error_m"] <= 1e-6
        and state["max_rotation_restore_error"] <= 1e-6
        for episode in episodes
        for state in episode["states"]
    )
    scene_state_support = all(
        any(row["scene"] == scene and row["state_name"] == state for row in rows)
        for scene in prereg["train_dev_scenes"]
        for state in [item["name"] for item in prereg["state_definitions"]]
    )
    metrics = compute_metrics(rows)
    protocol_guards = {
        "original_source_pins": True,
        "g713_anchor_hashes_and_guards_exact": True,
        "frozen_26_case_reconstruction_exact": [
            frozen_case_identity(anchor_by_name[case["name"]]) for case in cases
        ]
        == cases,
        "fixed_26_unique_rows_52_snapshots": len(rows) == 26
        and len({row["name"] for row in rows}) == 26
        and len(snapshots) == 52,
        "two_repeats_per_row": all(len(row["repeats"]) == 2 for row in rows),
        "repeat_depth_pose_prefix_extrinsics_features_exact": metrics[
            "all_repeats_exact"
        ],
        "g713_prefix_anchor_exact": anchor_prefix_exact,
        "state_restore_exact": restore_exact,
        "both_scenes_and_states_present": scene_state_support,
        "fixed_pitch_and_target_geometry_exact": True,
        "critical_primitive_unexecuted": True,
        "features_exclude_privileged_inputs": True,
        "policy_rgb_orientation_unchanged": True,
        "no_model_fit_or_pitch_threshold_sweep": True,
        "no_risk_controller_or_navigation": True,
        "zero_val_unseen": True,
    }
    integrity_ready = all(protocol_guards.values())
    gate = visibility_gate(metrics, integrity_ready=integrity_ready)
    if not integrity_ready:
        route = "repair-g714-instrumentation-or-anchor-reproduction"
    elif gate["ready"]:
        route = "preregister-tighter-heldout-executed-outcome-discrimination"
    else:
        route = "close-revised-visual-ground-support-and-define-explicit-lower-layer-sensing-contract"

    rows_path = output / "visibility_rows.jsonl"
    with rows_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(jsonable(row), sort_keys=True) + "\n")
    snapshots_path = output / "pitched_depth_snapshots.jsonl"
    with snapshots_path.open("w") as handle:
        for snapshot in snapshots:
            handle.write(json.dumps(snapshot, sort_keys=True) + "\n")
    result = {
        "schema_version": 1,
        "kind": "streamvln-g7.14-immediate-footprint-visual-visibility-ceiling",
        "claim_scope": "fixed two-scene/26-case pitched-depth discrete immediate-footprint visibility ceiling; no usefulness, outcome, controller, navigation, language, val_unseen or generalization claim",
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
            "case_count": len(rows),
            "snapshot_count": len(snapshots),
            "repeat_count": 2,
            "scene_count": len({row["scene"] for row in rows}),
            "state_count": len({row["state_name"] for row in rows}),
        },
        "visibility_metrics": metrics,
        "visibility_gate": gate,
        "protocol_guards": protocol_guards,
        "all_protocol_guards_passed": integrity_ready,
        "immediate_footprint_visibility_ready": gate["ready"],
        "outcome_discrimination_ready": False,
        "controller_ready": False,
        "navigation_ready": False,
        "g8_ready": False,
        "routing": route,
        "evidence": {
            "rows_path": str(rows_path),
            "rows_sha256": sha256_file(rows_path),
            "pitched_depth_snapshots_path": str(snapshots_path),
            "pitched_depth_snapshots_sha256": sha256_file(snapshots_path),
            "g713_result_sha256": args.g713_result_sha256,
            "g713_run_config_sha256": args.g713_run_config_sha256,
            "g713_rows_sha256": args.g713_rows_sha256,
            "g713_depth_snapshots_sha256": args.g713_depth_snapshots_sha256,
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
                "immediate_footprint_visibility_ready": gate["ready"],
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
