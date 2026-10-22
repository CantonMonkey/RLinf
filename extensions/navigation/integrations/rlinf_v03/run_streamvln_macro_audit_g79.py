"""Run the bounded G7.9 collision-horizon / exact-macro oracle ladder."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from run_streamvln_candidates_g7 import configure_process_git_safe_directory, validate_identity
from run_streamvln_ppo_g4 import make_env, validate_dataset_hashes, validate_manifest
from run_streamvln_topology_g5 import ORIGINAL_STREAM_COMMIT, git_head, jsonable, sha256_file
from streamvln_candidates_g74 import generate_macro_candidates
from streamvln_depth_g78 import DepthPassabilityNet, predict_depth_scores
from streamvln_macro_audit_g79 import (
    EXCLUDED_CANDIDATE,
    REGISTERED_CANDIDATES_PER_SCENE,
    classify_route,
    distance_prefix_audit,
    distribution,
    safe_binary_metrics,
    select_registered_candidates,
)
from streamvln_topology import OnlineTopology


RLINF_COMMIT = "0505431899574619da86f551bad70b71e0ea2177"


def pose_errors(first, second):
    position = float(
        np.linalg.norm(
            np.asarray(first["position"], dtype=np.float64)
            - np.asarray(second["position"], dtype=np.float64)
        )
    )
    a = np.asarray(first["rotation_coefficients"], dtype=np.float64)
    b = np.asarray(second["rotation_coefficients"], dtype=np.float64)
    rotation = float(min(np.max(np.abs(a - b)), np.max(np.abs(a + b))))
    return position, rotation


def per_scene_metrics(rows, score_key, label_key, threshold):
    return {
        scene: safe_binary_metrics(
            [row[label_key] for row in rows if row["scene"] == scene],
            [row[score_key] for row in rows if row["scene"] == scene],
            threshold,
        )
        for scene in sorted({row["scene"] for row in rows})
    }


def horizon_consistency(rows, label_key):
    groups = {}
    for row in rows:
        key = (row["scene"], float(row["heading_degrees"]))
        groups.setdefault(key, []).append(bool(row[label_key]))
    violations = [
        {"scene": scene, "heading_degrees": heading}
        for (scene, heading), labels in sorted(groups.items())
        if len(set(labels)) != 1
    ]
    return {
        "group_count": len(groups),
        "violation_count": len(violations),
        "violations": violations,
    }


def load_frozen_g78(run_cfg, device):
    result_path = Path(run_cfg["g78_result"])
    checkpoint_path = Path(run_cfg["g78_depth_checkpoint"])
    if sha256_file(result_path) != run_cfg["g78_result_sha256"]:
        raise RuntimeError("G7.8 result hash changed")
    if sha256_file(checkpoint_path) != run_cfg["g78_depth_checkpoint_sha256"]:
        raise RuntimeError("G7.8 depth checkpoint hash changed")
    result = json.loads(result_path.read_text())
    if (
        result.get("kind") != "streamvln-g7.8-aligned-depth-local-controller-upper-bound"
        or result.get("source_contract", {}).get("stream_commit") != ORIGINAL_STREAM_COMMIT
        or result.get("source_contract", {}).get("rlinf_commit") != RLINF_COMMIT
        or result.get("val_unseen_episode_count") != 0
    ):
        raise RuntimeError("G7.8 frozen source contract changed")
    threshold = float(result["models"]["depth_geometry"]["dev"]["threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("G7.8 frozen threshold is invalid")
    model = DepthPassabilityNet().to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return result, model, threshold


def collect_scene(run_cfg, spec, candidates):
    cfg = dict(run_cfg)
    cfg["habitat_server"] = run_cfg["habitat_macro_audit_server"]
    env = make_env(
        cfg,
        split="train",
        episode_index=spec["index"],
        max_steps=run_cfg["max_episode_steps"],
    )
    try:
        _, identity = env.reset()
        validate_identity(identity, spec)
        audit = env.probe_macro_audit(candidates)
        if audit["physical_horizon_m"] != 0.5 or audit["repeat_count"] != 2:
            raise RuntimeError("G7.9 service preregistration changed")
        by_name = {row["name"]: row for row in audit["candidates"]}
        if set(by_name) != {candidate.name for candidate in candidates}:
            raise RuntimeError("G7.9 service candidate bank changed")
        rows = []
        for candidate in candidates:
            probe = by_name[candidate.name]
            first, second = probe["repeats"]
            position_error, rotation_error = pose_errors(
                first["post_turn_pose"], second["post_turn_pose"]
            )
            endpoint_repeat_error = float(
                np.linalg.norm(
                    np.asarray(first["final_pose"]["position"], dtype=np.float64)
                    - np.asarray(second["final_pose"]["position"], dtype=np.float64)
                )
            )
            row = {
                "scene": spec["scene"],
                "episode_id": str(spec["episode_id"]),
                "decision": 0,
                "candidate": candidate.name,
                "heading_degrees": float(candidate.heading_degrees),
                "model_heading_degrees": 0.0,
                "distance_m": float(candidate.distance_m),
                "actions": list(candidate.actions),
                "turn_prefix": list(probe["turn_prefix"]),
                "forward_suffix": list(probe["forward_suffix"]),
                "current_depth_sha256": first["depth_sha256"],
                "strict_depth_sha256": second["depth_sha256"],
                "post_turn_position_repeat_error_m": position_error,
                "post_turn_rotation_repeat_error": rotation_error,
                "endpoint_repeat_error_m": endpoint_repeat_error,
                "collision_repeat_exact": (
                    first["full_collision_trace"] == second["full_collision_trace"]
                ),
                "current_original_unsafe": bool(first["full_unsafe"]),
                "strict_original_unsafe": bool(second["full_unsafe"]),
                "current_horizon_unsafe": bool(first["horizon_unsafe"]),
                "strict_horizon_unsafe": bool(second["horizon_unsafe"]),
                "repeats": [],
                "_current_depth": first["metric_depth"],
                "_strict_depth": second["metric_depth"],
            }
            for repeat in (first, second):
                evidence = dict(repeat)
                evidence.pop("metric_depth")
                row["repeats"].append(evidence)
            rows.append(row)
        return {
            "scene": spec["scene"],
            "episode_id": str(spec["episode_id"]),
            "candidate_count": len(rows),
            "max_position_restore_error_m": audit["max_position_restore_error_m"],
            "max_rotation_restore_error": audit["max_rotation_restore_error"],
        }, rows
    finally:
        env.close()


def parse_args():
    parser = argparse.ArgumentParser()
    for name in (
        "manifest", "g78-result", "g78-result-sha256",
        "g78-depth-checkpoint", "g78-depth-checkpoint-sha256",
        "output-dir", "rlinf-root", "stream-source", "habitat-python",
        "habitat-macro-audit-server", "asset-root", "habitat-sim-python",
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
        raise ValueError("G7.9 episode budget is invalid")
    path_names = (
        "manifest", "g78_result", "g78_depth_checkpoint", "rlinf_root",
        "stream_source", "habitat_python", "habitat_macro_audit_server",
        "asset_root", "habitat_sim_python", "habitat_env_lib",
        "egl_vendor_json",
    )
    paths = {
        name: str(Path(getattr(args, name)).expanduser().resolve(strict=True))
        for name in path_names
    }
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("G7.9 output directory must be fresh")
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(paths["manifest"]).read_text())
    validate_manifest(manifest)
    prereg = manifest.get("g79_macro_audit", {})
    expected_prereg = {
        "split": "train",
        "decision_index": 0,
        "train_dev_scenes": [row["scene"] for row in manifest["train"][6:]],
        "episode_indices": [row["index"] for row in manifest["train"][6:]],
        "candidate_count_per_scene": 32,
        "excluded_candidate": EXCLUDED_CANDIDATE,
        "repeat_count": 2,
        "forward_step_m": 0.25,
        "horizon_forward_steps": 2,
        "physical_horizon_m": 0.5,
    }
    if prereg != expected_prereg:
        raise RuntimeError("G7.9 manifest preregistration changed")
    configure_process_git_safe_directory(paths["stream_source"])
    configure_process_git_safe_directory(paths["rlinf_root"])
    stream_commit = git_head(Path(paths["stream_source"]))
    rlinf_commit = git_head(Path(paths["rlinf_root"]))
    if stream_commit != ORIGINAL_STREAM_COMMIT or rlinf_commit != RLINF_COMMIT:
        raise RuntimeError("G7.9 source commit pin changed")
    topology = OnlineTopology(max_decisions=args.max_episode_steps)
    candidates = select_registered_candidates(generate_macro_candidates(topology))
    if len(candidates) != REGISTERED_CANDIDATES_PER_SCENE:
        raise RuntimeError("G7.9 fixed candidate count changed")
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
    g78_result, model, threshold = load_frozen_g78(run_cfg, device)
    started = time.time()
    episodes, rows = [], []
    for spec in manifest["train"][6:]:
        episode, scene_rows = collect_scene(run_cfg, spec, candidates)
        episodes.append(episode)
        rows.extend(scene_rows)
    if len(rows) != 64 or len(episodes) != 2:
        raise RuntimeError("G7.9 fixed sample count changed")

    current_model_rows = [
        {
            "metric_depth": row["_current_depth"],
            "heading_degrees": row["model_heading_degrees"],
            "distance_m": row["distance_m"],
            "unsafe": row["current_original_unsafe"],
        }
        for row in rows
    ]
    strict_model_rows = [
        {
            "metric_depth": row["_strict_depth"],
            "heading_degrees": row["model_heading_degrees"],
            "distance_m": row["distance_m"],
            "unsafe": row["strict_horizon_unsafe"],
        }
        for row in rows
    ]
    current_scores = predict_depth_scores(model, current_model_rows, device=device)
    strict_scores = predict_depth_scores(model, strict_model_rows, device=device)
    for row, current_score, strict_score in zip(rows, current_scores, strict_scores):
        row["current_score"] = float(current_score)
        row["strict_score"] = float(strict_score)
        row.pop("_current_depth")
        row.pop("_strict_depth")

    ladder = {
        "A_current_pairing_original_full_macro_label": safe_binary_metrics(
            [row["current_original_unsafe"] for row in rows],
            [row["current_score"] for row in rows],
            threshold,
        ),
        "B_strict_post_turn_pairing_original_full_macro_label": safe_binary_metrics(
            [row["current_original_unsafe"] for row in rows],
            [row["strict_score"] for row in rows],
            threshold,
        ),
        "C_current_pairing_fixed_0p5m_horizon_label": safe_binary_metrics(
            [row["current_horizon_unsafe"] for row in rows],
            [row["current_score"] for row in rows],
            threshold,
        ),
        "D_strict_pairing_fixed_0p5m_horizon_label": safe_binary_metrics(
            [row["strict_horizon_unsafe"] for row in rows],
            [row["strict_score"] for row in rows],
            threshold,
        ),
    }
    ladder_by_scene = {
        "A": per_scene_metrics(
            rows, "current_score", "current_original_unsafe", threshold
        ),
        "B": per_scene_metrics(
            rows, "strict_score", "current_original_unsafe", threshold
        ),
        "C": per_scene_metrics(
            rows, "current_score", "current_horizon_unsafe", threshold
        ),
        "D": per_scene_metrics(
            rows, "strict_score", "strict_horizon_unsafe", threshold
        ),
    }
    depth_hash_exact_rate = sum(
        row["current_depth_sha256"] == row["strict_depth_sha256"] for row in rows
    ) / len(rows)
    collision_repeat_exact_rate = sum(
        row["collision_repeat_exact"] for row in rows
    ) / len(rows)
    max_score_difference = max(
        abs(row["current_score"] - row["strict_score"]) for row in rows
    )
    max_post_turn_position_error = max(
        row["post_turn_position_repeat_error_m"] for row in rows
    )
    max_post_turn_rotation_error = max(
        row["post_turn_rotation_repeat_error"] for row in rows
    )
    endpoint_repeat = distribution(
        row["endpoint_repeat_error_m"] for row in rows
    )
    endpoint_error = distribution(
        repeat["endpoint_error_m"]
        for row in rows
        for repeat in row["repeats"]
    )
    heading_error = distribution(
        repeat["heading_error_degrees"]
        for row in rows
        for repeat in row["repeats"]
    )
    commanded_realized_error = distribution(
        abs(repeat["commanded_distance_m"] - repeat["realized_forward_m"])
        for row in rows
        for repeat in row["repeats"]
    )
    privileged_rows = [
        repeat
        for row in rows
        for repeat in row["repeats"]
        if repeat["privileged_swept_path_free"]
    ]
    privileged_contradictions = [
        repeat for repeat in privileged_rows if repeat["full_unsafe"]
    ]
    privileged_contradiction_rate = (
        len(privileged_contradictions) / len(privileged_rows)
        if privileged_rows
        else 1.0
    )
    prefix_first = distance_prefix_audit(rows, "current_original_unsafe")
    prefix_second = distance_prefix_audit(rows, "strict_original_unsafe")
    horizon_first = horizon_consistency(rows, "current_horizon_unsafe")
    horizon_second = horizon_consistency(rows, "strict_horizon_unsafe")
    horizon_label_change_rate = sum(
        row["current_original_unsafe"] != row["current_horizon_unsafe"]
        for row in rows
    ) / len(rows)
    if ladder["A_current_pairing_original_full_macro_label"]["status"] == "ok" and ladder["D_strict_pairing_fixed_0p5m_horizon_label"]["status"] == "ok":
        horizon_gain = (
            ladder["D_strict_pairing_fixed_0p5m_horizon_label"]["balanced_accuracy"]
            - ladder["A_current_pairing_original_full_macro_label"]["balanced_accuracy"]
        )
    else:
        horizon_gain = float("nan")
    pairing_exact = (
        depth_hash_exact_rate == 1.0
        and max_score_difference <= 1e-8
        and max_post_turn_position_error <= 1e-6
        and max_post_turn_rotation_error <= 1e-6
    )
    prefix_monotonic = (
        prefix_first["violation_count"] == 0
        and prefix_second["violation_count"] == 0
        and horizon_first["violation_count"] == 0
        and horizon_second["violation_count"] == 0
    )
    route = classify_route(
        pairing_exact=pairing_exact,
        collision_repeat_exact=collision_repeat_exact_rate == 1.0,
        prefix_monotonic=prefix_monotonic,
        privileged_contradiction_rate=privileged_contradiction_rate,
        endpoint_repeat_max_m=endpoint_repeat["max"],
        horizon_label_change_rate=horizon_label_change_rate,
        horizon_balanced_accuracy_gain=horizon_gain,
    )
    protocol_guards = {
        "fixed_two_train_dev_scenes": {row["scene"] for row in rows} == set(prereg["train_dev_scenes"]),
        "fixed_64_candidates": len(rows) == 64,
        "fixed_32_candidates_per_scene": all(
            sum(row["scene"] == scene for row in rows) == 32
            for scene in prereg["train_dev_scenes"]
        ),
        "two_repeats": all(len(row["repeats"]) == 2 for row in rows),
        "state_restored": all(
            episode["max_position_restore_error_m"] <= 1e-6
            and episode["max_rotation_restore_error"] <= 1e-6
            for episode in episodes
        ),
        "frozen_predictor": True,
        "no_model_fit": True,
        "no_navigation_evaluation": True,
        "zero_val_unseen": True,
        "policy_observation_excludes_depth_and_labels": True,
    }
    evidence_path = output / "macro_audit_rows.jsonl"
    with evidence_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(jsonable(row), sort_keys=True) + "\n")
    result = {
        "schema_version": 1,
        "kind": "streamvln-g7.9-collision-horizon-exact-macro-oracle-ladder",
        "claim_scope": "train-only diagnostic of post-turn pairing, collision-label horizon, and exact-macro controllability; no model fit, navigation, RGB/language, or val_unseen claim",
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
            "candidate_count": len(rows),
            "unsafe_original": sum(row["current_original_unsafe"] for row in rows),
            "unsafe_fixed_horizon": sum(row["current_horizon_unsafe"] for row in rows),
            "depth_hash_exact_rate": depth_hash_exact_rate,
            "collision_repeat_exact_rate": collision_repeat_exact_rate,
        },
        "oracle_ladder": ladder,
        "oracle_ladder_by_scene": ladder_by_scene,
        "pairing_audit": {
            "exact": pairing_exact,
            "depth_hash_exact_rate": depth_hash_exact_rate,
            "max_score_difference": max_score_difference,
            "max_post_turn_position_error_m": max_post_turn_position_error,
            "max_post_turn_rotation_error": max_post_turn_rotation_error,
        },
        "label_horizon_audit": {
            "horizon_label_change_rate": horizon_label_change_rate,
            "balanced_accuracy_gain_D_over_A": (
                None if not math.isfinite(horizon_gain) else horizon_gain
            ),
            "physical_horizon_m": 0.5,
        },
        "executor_audit": {
            "collision_repeat_exact_rate": collision_repeat_exact_rate,
            "endpoint_repeat_error_m": endpoint_repeat,
            "endpoint_error_m": endpoint_error,
            "heading_error_degrees": heading_error,
            "commanded_vs_realized_forward_error_m": commanded_realized_error,
            "privileged_free_count": len(privileged_rows),
            "privileged_free_but_collision_count": len(privileged_contradictions),
            "privileged_contradiction_rate": privileged_contradiction_rate,
            "distance_prefix_repeat_0": prefix_first,
            "distance_prefix_repeat_1": prefix_second,
            "fixed_horizon_consistency_repeat_0": horizon_first,
            "fixed_horizon_consistency_repeat_1": horizon_second,
        },
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
    result_path.write_text(
        json.dumps(jsonable(result), indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "result": str(result_path),
                "result_sha256": sha256_file(result_path),
                "all_protocol_guards_passed": result["all_protocol_guards_passed"],
                "g8_ready": False,
                "routing": route,
                "pairing_audit": result["pairing_audit"],
                "label_horizon_audit": result["label_horizon_audit"],
                "executor_audit": {
                    "collision_repeat_exact_rate": collision_repeat_exact_rate,
                    "endpoint_repeat_error_m": endpoint_repeat,
                    "privileged_contradiction_rate": privileged_contradiction_rate,
                    "distance_prefix_violations": (
                        prefix_first["violation_count"] + prefix_second["violation_count"]
                    ),
                },
                "oracle_ladder": ladder,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
