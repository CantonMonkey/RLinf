"""Run the bounded G7.10 collision-severity / contact-geometry audit."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from run_streamvln_candidates_g7 import configure_process_git_safe_directory, validate_identity
from run_streamvln_macro_audit_g79 import load_frozen_g78, pose_errors
from run_streamvln_ppo_g4 import make_env, validate_dataset_hashes, validate_manifest
from run_streamvln_topology_g5 import ORIGINAL_STREAM_COMMIT, git_head, jsonable, sha256_file
from streamvln_candidates_g74 import generate_macro_candidates
from streamvln_depth_g78 import predict_depth_scores
from streamvln_macro_audit_g79 import (
    EXCLUDED_CANDIDATE,
    REGISTERED_CANDIDATES_PER_SCENE,
    distribution,
    safe_binary_metrics,
    select_registered_candidates,
)
from streamvln_severity_g710 import (
    SEVERITY_ORDER,
    class_summary,
    predictor_gates,
    severity_outcome,
    spearman_correlation,
)
from streamvln_topology import OnlineTopology


RLINF_COMMIT = "0505431899574619da86f551bad70b71e0ea2177"
STATE_DEFINITIONS = [
    {"name": "reset", "transition_actions": []},
    {"name": "after_fixed_p0_d0p5", "transition_actions": [1, 1]},
]


def _expected_preregistration(manifest):
    return {
        "split": "train",
        "train_dev_scenes": [row["scene"] for row in manifest["train"][6:]],
        "episode_indices": [row["index"] for row in manifest["train"][6:]],
        "state_definitions": STATE_DEFINITIONS,
        "candidate_count_per_state": 32,
        "excluded_candidate": EXCLUDED_CANDIDATE,
        "repeat_count": 2,
        "total_rows": 128,
        "forward_step_m": 0.25,
        "clearance_spacing_m": 0.125,
        "clearance_search_radius_m": 2.0,
        "severity_thresholds": {
            "free_endpoint_tolerance_m": 0.05,
            "free_lateral_tolerance_m": 0.05,
            "free_realized_ratio": 0.90,
            "sliding_primitive_deficit_m": 0.05,
            "severe_endpoint_error_m": 0.25,
            "severe_lateral_error_m": 0.25,
            "severe_realized_ratio": 0.50,
            "severe_min_primitive_progress_m": 0.05,
        },
        "predictor_gates": {
            "spearman_min": 0.30,
            "severe_auroc_min": 0.75,
            "severe_recall_min": 0.80,
            "severe_balanced_accuracy_min": 0.70,
            "grazing_false_positive_rate_max": 0.30,
        },
    }


def _repeat_outcomes_exact(first, second, tolerance=1e-8):
    if (
        first["severity_class"] != second["severity_class"]
        or first["severe_blockage"] != second["severe_blockage"]
        or first["raw_collision"] != second["raw_collision"]
    ):
        return False
    numeric = (
        "continuous_severity",
        "realized_ratio",
        "maximum_primitive_progress_deficit_m",
        "minimum_primitive_progress_m",
    )
    return all(abs(float(first[key]) - float(second[key])) <= tolerance for key in numeric)


def _grazing_false_positive_rate(rows, score_key, threshold):
    grazing = [row for row in rows if row["severity_class"] == "grazing"]
    if not grazing:
        return None
    return sum(float(row[score_key]) >= threshold for row in grazing) / len(grazing)


def _metric_bundle(rows, score_key, threshold):
    labels = [row["severe_blockage"] for row in rows]
    scores = [row[score_key] for row in rows]
    severities = [row["continuous_severity"] for row in rows]
    return {
        "count": len(rows),
        "class_counts": class_summary(rows),
        "continuous_severity": distribution(severities),
        "spearman_score_vs_continuous_severity": spearman_correlation(scores, severities),
        "severe_blockage": safe_binary_metrics(labels, scores, threshold),
        "grazing_false_positive_rate": _grazing_false_positive_rate(
            rows, score_key, threshold
        ),
    }


def _stratified_metrics(rows, key, score_key, threshold):
    return {
        value: _metric_bundle(
            [row for row in rows if row[key] == value], score_key, threshold
        )
        for value in sorted({row[key] for row in rows})
    }


def _state_stability(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["scene"], row["candidate"]), {})[
            row["state_name"]
        ] = row
    pairs = []
    for (scene, candidate), states in sorted(grouped.items()):
        if set(states) != {item["name"] for item in STATE_DEFINITIONS}:
            raise RuntimeError("G7.10 state pair is incomplete")
        reset = states["reset"]
        shifted = states["after_fixed_p0_d0p5"]
        pairs.append(
            {
                "scene": scene,
                "candidate": candidate,
                "class_changed": reset["severity_class"] != shifted["severity_class"],
                "severe_changed": reset["severe_blockage"] != shifted["severe_blockage"],
                "continuous_severity_delta": shifted["continuous_severity"]
                - reset["continuous_severity"],
                "score_delta": shifted["score"] - reset["score"],
                "minimum_intended_clearance_delta_m": shifted[
                    "minimum_intended_clearance_m"
                ]
                - reset["minimum_intended_clearance_m"],
            }
        )
    return {
        "pair_count": len(pairs),
        "class_change_rate": sum(row["class_changed"] for row in pairs) / len(pairs),
        "severe_label_change_rate": sum(row["severe_changed"] for row in pairs)
        / len(pairs),
        "absolute_continuous_severity_change": distribution(
            abs(row["continuous_severity_delta"]) for row in pairs
        ),
        "absolute_score_change": distribution(abs(row["score_delta"]) for row in pairs),
        "absolute_minimum_intended_clearance_change_m": distribution(
            abs(row["minimum_intended_clearance_delta_m"]) for row in pairs
        ),
    }


def collect_scene(run_cfg, spec, candidates, prereg):
    cfg = dict(run_cfg)
    cfg["habitat_server"] = run_cfg["habitat_severity_audit_server"]
    env = make_env(
        cfg,
        split="train",
        episode_index=spec["index"],
        max_steps=run_cfg["max_episode_steps"],
    )
    rows = []
    state_summaries = []
    try:
        _, identity = env.reset()
        validate_identity(identity, spec)
        for state_index, state in enumerate(prereg["state_definitions"]):
            transitions = []
            for action in state["transition_actions"]:
                _, _, terminated, truncated, infos = env.step(action)
                transition = {
                    "action": int(action),
                    "collision": bool(infos["collision"].item()),
                    "terminated": bool(terminated.item()),
                    "truncated": bool(truncated.item()),
                }
                transitions.append(transition)
                if transition["terminated"] or transition["truncated"]:
                    raise RuntimeError("G7.10 fixed state transition ended the episode")
            audit = env.probe_severity_audit(candidates)
            if (
                audit["repeat_count"] != 2
                or audit["clearance_spacing_m"] != 0.125
                or audit["clearance_search_radius_m"] != 2.0
            ):
                raise RuntimeError("G7.10 service preregistration changed")
            by_name = {row["name"]: row for row in audit["candidates"]}
            if set(by_name) != {candidate.name for candidate in candidates}:
                raise RuntimeError("G7.10 service candidate bank changed")
            for candidate in candidates:
                probe = by_name[candidate.name]
                first, second = probe["repeats"]
                first_outcome = severity_outcome(first)
                second_outcome = severity_outcome(second)
                position_error, rotation_error = pose_errors(
                    first["post_turn_pose"], second["post_turn_pose"]
                )
                final_repeat_error = float(
                    np.linalg.norm(
                        np.asarray(first["final_pose"]["position"], dtype=np.float64)
                        - np.asarray(second["final_pose"]["position"], dtype=np.float64)
                    )
                )
                evidence = []
                for repeat in (first, second):
                    item = dict(repeat)
                    item.pop("metric_depth")
                    evidence.append(item)
                rows.append(
                    {
                        "scene": spec["scene"],
                        "episode_id": str(spec["episode_id"]),
                        "state_index": state_index,
                        "state_name": state["name"],
                        "state_transition_actions": list(state["transition_actions"]),
                        "state_transition_collisions": [
                            item["collision"] for item in transitions
                        ],
                        "candidate": candidate.name,
                        "heading_degrees": float(candidate.heading_degrees),
                        "model_heading_degrees": 0.0,
                        "distance_m": float(candidate.distance_m),
                        "actions": list(candidate.actions),
                        "turn_prefix": list(probe["turn_prefix"]),
                        "forward_suffix": list(probe["forward_suffix"]),
                        "depth_sha256_first": first["depth_sha256"],
                        "depth_sha256_second": second["depth_sha256"],
                        "post_turn_position_repeat_error_m": position_error,
                        "post_turn_rotation_repeat_error": rotation_error,
                        "final_position_repeat_error_m": final_repeat_error,
                        "collision_repeat_exact": first["full_collision_trace"]
                        == second["full_collision_trace"],
                        "clearance_trace_repeat_exact": (
                            first["intended_clearance_trace"]
                            == second["intended_clearance_trace"]
                            and first["actual_clearance_trace"]
                            == second["actual_clearance_trace"]
                        ),
                        "target_repeat_exact": _repeat_outcomes_exact(
                            first_outcome, second_outcome
                        ),
                        **first_outcome,
                        "minimum_intended_clearance_m": float(
                            first["minimum_intended_clearance_m"]
                        ),
                        "minimum_actual_clearance_m": float(
                            first["minimum_actual_clearance_m"]
                        ),
                        "no_sliding_endpoint_error_m": float(
                            first["no_sliding_endpoint_error_m"]
                        ),
                        "repeats": evidence,
                        "_depth_first": first["metric_depth"],
                        "_depth_second": second["metric_depth"],
                    }
                )
            state_summaries.append(
                {
                    "name": state["name"],
                    "transition_actions": list(state["transition_actions"]),
                    "transition_records": transitions,
                    "candidate_count": len(by_name),
                    "max_position_restore_error_m": audit[
                        "max_position_restore_error_m"
                    ],
                    "max_rotation_restore_error": audit[
                        "max_rotation_restore_error"
                    ],
                }
            )
        return {
            "scene": spec["scene"],
            "episode_id": str(spec["episode_id"]),
            "states": state_summaries,
        }, rows
    finally:
        env.close()


def parse_args():
    parser = argparse.ArgumentParser()
    for name in (
        "manifest",
        "g78-result",
        "g78-result-sha256",
        "g78-depth-checkpoint",
        "g78-depth-checkpoint-sha256",
        "output-dir",
        "rlinf-root",
        "stream-source",
        "habitat-python",
        "habitat-severity-audit-server",
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
        raise ValueError("G7.10 episode budget is invalid")
    path_names = (
        "manifest",
        "g78_result",
        "g78_depth_checkpoint",
        "rlinf_root",
        "stream_source",
        "habitat_python",
        "habitat_severity_audit_server",
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
        raise FileExistsError("G7.10 output directory must be fresh")
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(paths["manifest"]).read_text())
    validate_manifest(manifest)
    prereg = manifest.get("g710_severity_audit", {})
    expected_prereg = _expected_preregistration(manifest)
    if prereg != expected_prereg:
        raise RuntimeError("G7.10 manifest preregistration changed")
    configure_process_git_safe_directory(paths["stream_source"])
    configure_process_git_safe_directory(paths["rlinf_root"])
    stream_commit = git_head(Path(paths["stream_source"]))
    rlinf_commit = git_head(Path(paths["rlinf_root"]))
    if stream_commit != ORIGINAL_STREAM_COMMIT or rlinf_commit != RLINF_COMMIT:
        raise RuntimeError("G7.10 source commit pin changed")
    topology = OnlineTopology(max_decisions=args.max_episode_steps)
    candidates = select_registered_candidates(generate_macro_candidates(topology))
    if len(candidates) != REGISTERED_CANDIDATES_PER_SCENE:
        raise RuntimeError("G7.10 fixed candidate count changed")
    run_cfg = {
        **paths,
        "manifest": manifest,
        "output_dir": str(output),
        "g78_result_sha256": args.g78_result_sha256,
        "g78_depth_checkpoint_sha256": args.g78_depth_checkpoint_sha256,
        "stream_commit": stream_commit,
        "rlinf_commit": rlinf_commit,
        "seed": args.seed,
        "max_episode_steps": args.max_episode_steps,
        "preregistration": expected_prereg,
        "candidate_names": [candidate.name for candidate in candidates],
        "val_unseen_episode_count": 0,
        "model_fit": False,
        "navigation_evaluation": False,
    }
    dataset_hashes = validate_dataset_hashes(run_cfg)
    (output / "run_config.json").write_text(
        json.dumps(run_cfg, indent=2, sort_keys=True) + "\n"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, model, threshold = load_frozen_g78(run_cfg, device)
    started = time.time()
    episodes, rows = [], []
    for spec in manifest["train"][6:]:
        episode, scene_rows = collect_scene(run_cfg, spec, candidates, prereg)
        episodes.append(episode)
        rows.extend(scene_rows)
    if len(rows) != prereg["total_rows"] or len(episodes) != 2:
        raise RuntimeError("G7.10 fixed sample count changed")

    first_model_rows = [
        {
            "metric_depth": row["_depth_first"],
            "heading_degrees": row["model_heading_degrees"],
            "distance_m": row["distance_m"],
            "unsafe": row["severe_blockage"],
        }
        for row in rows
    ]
    second_model_rows = [
        {
            "metric_depth": row["_depth_second"],
            "heading_degrees": row["model_heading_degrees"],
            "distance_m": row["distance_m"],
            "unsafe": row["severe_blockage"],
        }
        for row in rows
    ]
    first_scores = predict_depth_scores(model, first_model_rows, device=device)
    second_scores = predict_depth_scores(model, second_model_rows, device=device)
    for row, first_score, second_score in zip(rows, first_scores, second_scores):
        row["score"] = float(first_score)
        row["repeat_score"] = float(second_score)
        row["score_repeat_difference"] = abs(float(first_score) - float(second_score))
        row.pop("_depth_first")
        row.pop("_depth_second")

    overall = _metric_bundle(rows, "score", threshold)
    by_scene = _stratified_metrics(rows, "scene", "score", threshold)
    by_state = _stratified_metrics(rows, "state_name", "score", threshold)
    by_scene_state = {
        scene: _stratified_metrics(
            [row for row in rows if row["scene"] == scene],
            "state_name",
            "score",
            threshold,
        )
        for scene in sorted({row["scene"] for row in rows})
    }
    depth_hash_exact_rate = sum(
        row["depth_sha256_first"] == row["depth_sha256_second"] for row in rows
    ) / len(rows)
    target_repeat_exact_rate = sum(row["target_repeat_exact"] for row in rows) / len(rows)
    collision_repeat_exact_rate = sum(row["collision_repeat_exact"] for row in rows) / len(rows)
    clearance_repeat_exact_rate = sum(
        row["clearance_trace_repeat_exact"] for row in rows
    ) / len(rows)
    max_score_repeat_difference = max(row["score_repeat_difference"] for row in rows)
    max_post_turn_position_error = max(
        row["post_turn_position_repeat_error_m"] for row in rows
    )
    max_post_turn_rotation_error = max(
        row["post_turn_rotation_repeat_error"] for row in rows
    )
    max_final_position_error = max(row["final_position_repeat_error_m"] for row in rows)
    pairing_exact = (
        depth_hash_exact_rate == 1.0
        and max_score_repeat_difference <= 1e-8
        and max_post_turn_position_error <= 1e-6
        and max_post_turn_rotation_error <= 1e-6
    )
    target_repeat_exact = (
        target_repeat_exact_rate == 1.0
        and collision_repeat_exact_rate == 1.0
        and clearance_repeat_exact_rate == 1.0
        and max_final_position_error <= 1e-6
    )
    gates = predictor_gates(
        spearman=overall["spearman_score_vs_continuous_severity"],
        severe_metrics=overall["severe_blockage"],
        grazing_false_positive_rate=overall["grazing_false_positive_rate"],
        target_repeat_exact=target_repeat_exact,
        pairing_exact=pairing_exact,
    )
    class_counts = overall["class_counts"]
    target_coherent = (
        gates["integrity_ready"]
        and overall["severe_blockage"]["status"] == "ok"
        and sum(value > 0 for value in class_counts.values()) >= 2
        and overall["continuous_severity"]["max"]
        > overall["continuous_severity"]["p50"]
    )
    severe_metrics = overall["severe_blockage"]
    score_near_random = (
        severe_metrics.get("status") == "ok"
        and float(severe_metrics["auroc"]) < 0.60
        and abs(float(overall["spearman_score_vs_continuous_severity"])) < 0.20
    )
    if not gates["integrity_ready"]:
        route = "repair-instrumentation-or-state-pairing-before-any-controller-work"
    elif target_coherent and score_near_random:
        route = "stop-current-learned-depth-predictor-and-design-explicit-geometric-local-planner-controller"
    elif not target_coherent:
        route = "supplement-severity-target-support-before-controller-selection"
    elif not gates["score_ready"]:
        route = "freeze-diagnostic-and-test-explicit-geometric-local-planner-controller"
    else:
        route = "run-separate-controller-only-experiment-before-navigation"

    state_names = {item["name"] for item in prereg["state_definitions"]}
    protocol_guards = {
        "fixed_two_train_dev_scenes": {row["scene"] for row in rows}
        == set(prereg["train_dev_scenes"]),
        "fixed_128_rows": len(rows) == 128,
        "fixed_32_candidates_per_scene_state": all(
            sum(
                row["scene"] == scene and row["state_name"] == state
                for row in rows
            )
            == 32
            for scene in prereg["train_dev_scenes"]
            for state in state_names
        ),
        "fixed_reset_and_nonreset_states": {row["state_name"] for row in rows}
        == state_names,
        "fixed_nonreset_prefix": prereg["state_definitions"] == STATE_DEFINITIONS,
        "two_repeats": all(len(row["repeats"]) == 2 for row in rows),
        "state_restored": all(
            state["max_position_restore_error_m"] <= 1e-6
            and state["max_rotation_restore_error"] <= 1e-6
            for episode in episodes
            for state in episode["states"]
        ),
        "frozen_predictor": True,
        "no_model_fit": True,
        "no_navigation_evaluation": True,
        "zero_val_unseen": True,
        "policy_observation_excludes_depth_clearance_and_labels": True,
        "try_step_not_used_as_continuous_collision_proof": True,
    }
    evidence_path = output / "severity_rows.jsonl"
    with evidence_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(jsonable(row), sort_keys=True) + "\n")
    result = {
        "schema_version": 1,
        "kind": "streamvln-g7.10-frozen-collision-severity-contact-geometry-audit",
        "claim_scope": "train-only fixed-state diagnostic of frozen depth score versus continuous motion severity and navmesh clearance; no fitting, navigation, RGB/language, or val_unseen claim",
        "split": "train",
        "val_unseen_episode_count": 0,
        "source_contract": {
            "stream_commit": stream_commit,
            "rlinf_commit": rlinf_commit,
            "source_kind": "upstream-original",
            "secondary_post_training_used": False,
            "ttt_used": False,
            "frozen_g78_depth_predictor": True,
            "frozen_g78_threshold": threshold,
        },
        "dataset_hashes": dataset_hashes,
        "preregistration": prereg,
        "episodes": episodes,
        "sample_summary": {
            "row_count": len(rows),
            "scene_count": len({row["scene"] for row in rows}),
            "state_count": len({row["state_name"] for row in rows}),
            "class_counts": class_counts,
            "raw_collision_count": sum(row["raw_collision"] for row in rows),
            "severe_blockage_count": sum(row["severe_blockage"] for row in rows),
        },
        "predictor_audit": {
            "overall": overall,
            "by_scene": by_scene,
            "by_state": by_state,
            "by_scene_state": by_scene_state,
            "gates": gates,
        },
        "repeat_and_pairing_audit": {
            "target_repeat_exact": target_repeat_exact,
            "target_repeat_exact_rate": target_repeat_exact_rate,
            "collision_repeat_exact_rate": collision_repeat_exact_rate,
            "clearance_trace_repeat_exact_rate": clearance_repeat_exact_rate,
            "depth_hash_exact_rate": depth_hash_exact_rate,
            "max_score_repeat_difference": max_score_repeat_difference,
            "max_post_turn_position_repeat_error_m": max_post_turn_position_error,
            "max_post_turn_rotation_repeat_error": max_post_turn_rotation_error,
            "max_final_position_repeat_error_m": max_final_position_error,
            "pairing_exact": pairing_exact,
        },
        "continuous_motion_audit": {
            "realized_ratio": distribution(row["realized_ratio"] for row in rows),
            "endpoint_error_m": distribution(
                row["repeats"][0]["endpoint_error_m"] for row in rows
            ),
            "lateral_error_m": distribution(
                row["repeats"][0]["realized_lateral_error_m"] for row in rows
            ),
            "minimum_primitive_progress_m": distribution(
                row["minimum_primitive_progress_m"] for row in rows
            ),
            "minimum_intended_clearance_m": distribution(
                row["minimum_intended_clearance_m"] for row in rows
            ),
            "minimum_actual_clearance_m": distribution(
                row["minimum_actual_clearance_m"] for row in rows
            ),
            "no_sliding_endpoint_error_m": distribution(
                row["no_sliding_endpoint_error_m"] for row in rows
            ),
        },
        "state_stability": _state_stability(rows),
        "target_coherent": target_coherent,
        "score_near_random": score_near_random,
        "routing": route,
        "protocol_guards": protocol_guards,
        "all_protocol_guards_passed": all(protocol_guards.values()),
        "g8_ready": False,
        "evidence": {
            "rows_path": str(evidence_path),
            "rows_sha256": sha256_file(evidence_path),
            "g78_result": paths["g78_result"],
            "g78_result_sha256": args.g78_result_sha256,
            "g78_depth_checkpoint": paths["g78_depth_checkpoint"],
            "g78_depth_checkpoint_sha256": args.g78_depth_checkpoint_sha256,
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
                "all_protocol_guards_passed": result["all_protocol_guards_passed"],
                "g8_ready": False,
                "routing": route,
                "target_coherent": target_coherent,
                "score_near_random": score_near_random,
                "sample_summary": result["sample_summary"],
                "overall": overall,
                "repeat_and_pairing_audit": result["repeat_and_pairing_audit"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
