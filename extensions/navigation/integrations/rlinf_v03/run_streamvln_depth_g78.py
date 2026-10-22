"""G7.8 train-only aligned-depth local-controller upper-bound gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from run_streamvln_candidates_g7 import configure_process_git_safe_directory, validate_identity
from run_streamvln_candidates_g71 import aggregate
from run_streamvln_candidates_g75 import build_gates, execute_sequence, load_g74_anchor
from run_streamvln_ppo_g4 import make_env, validate_dataset_hashes, validate_manifest
from run_streamvln_topology_g5 import ORIGINAL_STREAM_COMMIT, git_head, jsonable, sha256_file
from streamvln_candidates_g74 import WorldHeadingShield, generate_macro_candidates, select_oracle_candidate
from streamvln_depth_g78 import (
    DepthPassabilityNet,
    GeometryPassabilityNet,
    per_scene_metrics,
    predict_depth_scores,
    select_threshold,
    threshold_sensitivity,
    train_depth_model,
    validate_scene_split,
)
from streamvln_topology import LEFT_ACTION, OnlineTopology, STOP_ACTION


def summarize_rows(rows):
    scenes = {}
    for row in rows:
        item = scenes.setdefault(row["scene"], {"count": 0, "unsafe": 0, "safe": 0})
        item["count"] += 1
        item["unsafe"] += int(row["unsafe"])
        item["safe"] += int(not row["unsafe"])
    return {
        "count": len(rows),
        "unsafe": sum(int(row["unsafe"]) for row in rows),
        "safe": sum(int(not row["unsafe"]) for row in rows),
        "scene_count": len(scenes),
        "by_scene": scenes,
    }


def collect_episode(run_cfg, spec):
    cfg = dict(run_cfg)
    cfg["habitat_server"] = run_cfg["habitat_depth_passability_server"]
    env = make_env(cfg, split="train", episode_index=spec["index"], max_steps=run_cfg["max_oracle_steps"])
    topology = OnlineTopology(max_decisions=run_cfg["max_oracle_steps"])
    shield = WorldHeadingShield(topology.turn_radians, cooldown_decisions=run_cfg["shield_cooldown_decisions"])
    rows, actions = [], []
    decisions = 0
    max_position_error = max_rotation_error = 0.0
    try:
        obs, identity = env.reset()
        validate_identity(identity, spec)
        for decision_index in range(run_cfg["max_oracle_steps"]):
            if len(actions) >= run_cfg["max_oracle_steps"]:
                break
            shield.begin_decision()
            label = env.expert_waypoint(run_cfg["expert_lookahead_m"])
            target = [float(value) for value in label["waypoint_local_m"]]
            if label["stop"]:
                step = execute_sequence(
                    env, topology, [STOP_ACTION], [0.0, 0.0],
                    interrupt_on_collision=True,
                    remaining_steps=run_cfg["max_oracle_steps"] - len(actions),
                )
            else:
                candidates = generate_macro_candidates(topology)
                probe = env.probe_depth_candidates(candidates)
                max_position_error = max(max_position_error, probe["max_position_restore_error_m"])
                max_rotation_error = max(max_rotation_error, probe["max_rotation_restore_error"])
                by_name = {row["name"]: row for row in probe["candidates"]}
                if set(by_name) != {candidate.name for candidate in candidates}:
                    raise RuntimeError("G7.8 probe candidate set changed")
                for candidate in candidates:
                    label_row = by_name[candidate.name]
                    depth = np.ascontiguousarray(
                        label_row["metric_depth"], dtype=np.float32
                    )
                    if depth.shape != (64, 64) or not np.isfinite(depth).all():
                        raise ValueError("G7.8 aligned metric-depth shape changed")
                    depth_sha = hashlib.sha256(depth.astype("<f4").tobytes()).hexdigest()
                    rows.append({
                        "scene": spec["scene"],
                        "episode_id": str(spec["episode_id"]),
                        "decision": decision_index,
                        "candidate": candidate.name,
                        "heading_degrees": 0.0,
                        "original_heading_degrees": candidate.heading_degrees,
                        "distance_m": candidate.distance_m,
                        "unsafe": bool(label_row["unsafe"]),
                        "forward_collisions": int(
                            sum(label_row["forward_collisions"])
                        ),
                        "depth_sha256": depth_sha,
                        "metric_depth": depth,
                    })
                available = shield.available(candidates, current_world_heading=topology.heading)
                selected, _ = select_oracle_candidate(available, target)
                step = execute_sequence(
                    env, topology, selected.actions, selected.vector(),
                    interrupt_on_collision=True,
                    remaining_steps=run_cfg["max_oracle_steps"] - len(actions),
                )
                for heading in step["collision_world_headings"]:
                    shield.block_world_heading(heading)
            actions.extend(step["executed_actions"])
            decisions += 1
            obs = step["obs"]
            if step["terminal"]:
                break
        return {
            "scene": spec["scene"],
            "episode_id": str(spec["episode_id"]),
            "row_count": len(rows),
            "decision_count": decisions,
            "primitive_steps": len(actions),
            "max_position_restore_error_m": max_position_error,
            "max_rotation_restore_error": max_rotation_error,
        }, rows
    finally:
        env.close()


def save_dataset(output, rows, split):
    ordered_scenes = split["train_scenes"] + split["train_dev_scenes"]
    scene_index = {scene: index for index, scene in enumerate(ordered_scenes)}
    path = output / "passability_dataset.npz"
    np.savez_compressed(
        path,
        metric_depth=np.stack([row["metric_depth"] for row in rows]).astype("<f4"),
        heading_degrees=np.asarray([row["heading_degrees"] for row in rows], dtype=np.float32),
        distance_m=np.asarray([row["distance_m"] for row in rows], dtype=np.float32),
        unsafe=np.asarray([row["unsafe"] for row in rows], dtype=np.bool_),
        scene_index=np.asarray([scene_index[row["scene"]] for row in rows], dtype=np.int16),
        decision=np.asarray([row["decision"] for row in rows], dtype=np.int16),
    )
    index_path = output / "passability_dataset_index.json"
    index = {
        "schema_version": 1,
        "source": "train-simulator-aligned-depth-branch-rollout",
        "controller_inputs": ["aligned_metric_depth_64x64", "candidate_distance"],
        "forbidden_policy_inputs": ["shortest_path", "goal_coordinates", "navmesh_oracle", "future_collision"],
        "split": split,
        "summary": summarize_rows(rows),
        "npz_sha256": sha256_file(path),
    }
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    return {
        "path": str(path),
        "sha256": index["npz_sha256"],
        "index_path": str(index_path),
        "index_sha256": sha256_file(index_path),
        "summary": index["summary"],
    }


def navigation_episode(run_cfg, spec, model, threshold, device):
    cfg = dict(run_cfg)
    cfg["habitat_server"] = run_cfg["habitat_depth_passability_server"]
    env = make_env(
        cfg,
        split="train",
        episode_index=spec["index"],
        max_steps=run_cfg["max_oracle_steps"],
    )
    topology = OnlineTopology(max_decisions=run_cfg["max_oracle_steps"])
    shield = WorldHeadingShield(
        topology.turn_radians,
        cooldown_decisions=run_cfg["shield_cooldown_decisions"],
    )
    actions, decisions, last_info = [], [], None
    last_truncated = False
    blocked_translation_count = 0
    fallback_turn_count = 0
    try:
        obs, identity = env.reset()
        validate_identity(identity, spec)
        for decision_index in range(run_cfg["max_oracle_steps"]):
            if len(actions) >= run_cfg["max_oracle_steps"]:
                break
            shield.begin_decision()
            label = env.expert_waypoint(run_cfg["expert_lookahead_m"])
            oracle_target = [float(value) for value in label["waypoint_local_m"]]
            candidates = ()
            available_names = []
            unrestricted = selected = None
            unrestricted_error = error = None
            redirected = False
            passability_score = None
            forward_executed = False
            turn_prefix = []
            forward_suffix = []
            collision_headings = []
            interrupted = False
            if label["stop"]:
                target = [0.0, 0.0]
                step = execute_sequence(
                    env,
                    topology,
                    [STOP_ACTION],
                    target,
                    interrupt_on_collision=True,
                    remaining_steps=run_cfg["max_oracle_steps"] - len(actions),
                )
                executed_actions = step["executed_actions"]
                collisions = step["collisions"]
                waypoint_execution = step["waypoint_execution"]
            else:
                candidates = generate_macro_candidates(topology)
                unrestricted, unrestricted_error = select_oracle_candidate(
                    candidates, oracle_target
                )
                available = shield.available(
                    candidates, current_world_heading=topology.heading
                )
                available_names = [candidate.name for candidate in available]
                selected, error = select_oracle_candidate(available, oracle_target)
                redirected = selected.name != unrestricted.name
                proposed = list(selected.actions)
                first_forward = proposed.index(1)
                turn_prefix = proposed[:first_forward]
                forward_suffix = proposed[first_forward:]
                target = list(selected.vector())
                executed_actions = []
                collisions = []
                remaining = run_cfg["max_oracle_steps"] - len(actions)
                turn_step = None
                if turn_prefix and remaining > 0:
                    turn_step = execute_sequence(
                        env,
                        topology,
                        turn_prefix,
                        [0.0, 0.0],
                        interrupt_on_collision=False,
                        remaining_steps=remaining,
                    )
                    executed_actions.extend(turn_step["executed_actions"])
                    collisions.extend(turn_step["collisions"])
                    remaining -= turn_step["execute_count"]
                    obs = turn_step["obs"]
                metric_depth = env.observe_aligned_depth()
                score_rows = [{
                    "metric_depth": metric_depth,
                    "heading_degrees": 0.0,
                    "distance_m": selected.distance_m,
                    "unsafe": False,
                }]
                passability_score = float(
                    predict_depth_scores(model, score_rows, device=device)[0]
                )
                safe = passability_score < threshold
                if safe and remaining > 0 and (
                    turn_step is None or not turn_step["terminal"]
                ):
                    forward_step = execute_sequence(
                        env,
                        topology,
                        forward_suffix,
                        [selected.distance_m, 0.0],
                        interrupt_on_collision=True,
                        remaining_steps=remaining,
                    )
                    executed_actions.extend(forward_step["executed_actions"])
                    collisions.extend(forward_step["collisions"])
                    collision_headings = forward_step["collision_world_headings"]
                    interrupted = forward_step["interrupted_on_collision"]
                    waypoint_execution = forward_step["waypoint_execution"]
                    step = forward_step
                    forward_executed = True
                    for heading in collision_headings:
                        shield.block_world_heading(heading)
                else:
                    blocked_translation_count += 1
                    if turn_step is None:
                        fallback = execute_sequence(
                            env,
                            topology,
                            [LEFT_ACTION],
                            [0.0, 0.0],
                            interrupt_on_collision=False,
                            remaining_steps=remaining,
                        )
                        executed_actions.extend(fallback["executed_actions"])
                        collisions.extend(fallback["collisions"])
                        step = fallback
                        fallback_turn_count += 1
                    else:
                        step = turn_step
                    waypoint_execution = {
                        "target_local_m": target,
                        "realized_local_m": [0.0, 0.0],
                        "l2_m": float(selected.distance_m),
                    }
            actions.extend(executed_actions)
            last_info, last_truncated = step["info"], step["truncated"]
            decisions.append({
                "decision": decision_index,
                "oracle_stop": bool(label["stop"]),
                "oracle_waypoint_local_m": oracle_target,
                "candidate_set": [candidate.as_dict() for candidate in candidates],
                "available_candidate_names": available_names,
                "unrestricted_candidate": None if unrestricted is None else unrestricted.name,
                "selected_candidate": None if selected is None else selected.name,
                "blacklist_redirected": redirected,
                "candidate_quantization_error_m": error,
                "unrestricted_quantization_error_m": unrestricted_error,
                "candidate_covered": None if error is None else error <= run_cfg["coverage_radius_m"],
                "selected_blocked_forward": False,
                "passability_threshold": threshold,
                "passability_score": passability_score,
                "turn_prefix": turn_prefix,
                "forward_suffix": forward_suffix,
                "forward_executed": forward_executed,
                "executed_waypoint_local_m": target,
                "proposed_actions": [STOP_ACTION] if selected is None else list(selected.actions),
                "execute_count": len(executed_actions),
                "actions": executed_actions,
                "collisions": collisions,
                "collision_world_headings": collision_headings,
                "interrupted_on_collision": interrupted,
                "terminated": step["terminated"],
                "truncated": step["truncated"],
                "waypoint_execution": waypoint_execution,
                "feasibility": shield.summary(),
                "topology": topology.summary(),
            })
            obs = step["obs"]
            if step["terminal"]:
                break
        if last_info is None or "episode" not in last_info:
            raise RuntimeError("G7.8 navigation gate lacks terminal metrics")
        metrics = {
            key: float(value.item()) for key, value in last_info["episode"].items()
        }
        stop_called = bool(actions and actions[-1] == STOP_ACTION)
        success = metrics["success"]
        outcome = (
            "success"
            if success
            else "premature_stop"
            if stop_called
            else "timeout"
            if last_truncated
            else "other_failure"
        )
        return {
            "episode_index": spec["index"],
            "episode_id": str(spec["episode_id"]),
            "scene": spec["scene"],
            "source_geodesic_m": float(spec["source_geodesic_m"]),
            "success": success,
            "spl": metrics["spl"],
            "navigation_error_m": metrics["distance_to_goal"],
            "progress_m": float(spec["source_geodesic_m"])
            - metrics["distance_to_goal"],
            "path_length_m": metrics["path_length"],
            "collision_count": int(metrics["collision_count"]),
            "primitive_steps": len(actions),
            "decision_count": len(decisions),
            "timeout": last_truncated,
            "stop_called": stop_called,
            "outcome": outcome,
            "blocked_translation_count": blocked_translation_count,
            "fallback_turn_count": fallback_turn_count,
            "actions": actions,
            "decisions": decisions,
            "topology": topology.summary(),
            "feasibility": shield.summary(),
        }
    finally:
        env.close()


def parse_args():
    parser = argparse.ArgumentParser()
    for name in (
        "manifest", "g74-result", "g74-result-sha256", "output-dir", "rlinf-root",
        "stream-source", "habitat-python", "habitat-server",
        "habitat-depth-passability-server", "asset-root", "habitat-sim-python",
        "habitat-env-lib",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--egl-vendor-json", default="/usr/share/glvnd/egl_vendor.d/50_mesa.json")
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--max-oracle-steps", type=int, default=40)
    parser.add_argument("--expert-lookahead-m", type=float, default=1.0)
    parser.add_argument("--coverage-radius-m", type=float, default=0.5)
    parser.add_argument("--shield-cooldown-decisions", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--geometry-epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--min-unsafe-recall", type=float, default=0.80)
    parser.add_argument("--min-safe-recall", type=float, default=0.70)
    parser.add_argument("--min-balanced-accuracy", type=float, default=0.75)
    parser.add_argument("--min-depth-gain", type=float, default=0.03)
    parser.add_argument("--min-candidate-coverage", type=float, default=0.90)
    parser.add_argument("--max-candidate-p90-error-m", type=float, default=0.50)
    parser.add_argument("--min-success-rate", type=float, default=0.75)
    parser.add_argument("--min-spl", type=float, default=0.75)
    parser.add_argument("--max-collision-count", type=float, default=1.375)
    parser.add_argument("--progress-noninferiority-m", type=float, default=0.10)
    return parser.parse_args()


def main():
    args = parse_args()
    probabilities = (
        args.min_unsafe_recall, args.min_safe_recall, args.min_balanced_accuracy,
        args.min_depth_gain, args.min_candidate_coverage, args.min_success_rate,
        args.min_spl,
    )
    if (
        args.max_oracle_steps < 8 or args.epochs < 1 or args.geometry_epochs < 1
        or args.learning_rate <= 0 or args.shield_cooldown_decisions < 1
        or any(not 0.0 <= value <= 1.0 for value in probabilities)
        or args.max_candidate_p90_error_m <= 0 or args.max_collision_count < 0
        or args.progress_noninferiority_m < 0
    ):
        raise ValueError("G7.8 bounded configuration is invalid")
    path_names = (
        "manifest", "g74_result", "rlinf_root", "stream_source", "habitat_python",
        "habitat_server", "habitat_depth_passability_server", "asset_root",
        "habitat_sim_python", "habitat_env_lib", "egl_vendor_json",
    )
    paths = {name: str(Path(getattr(args, name)).expanduser().resolve(strict=True)) for name in path_names}
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("G7.8 output directory must be fresh")
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(paths["manifest"]).read_text())
    validate_manifest(manifest)
    split = manifest.get("g78_passability_split", {})
    train_scenes = split.get("train_scenes", [])
    dev_scenes = split.get("train_dev_scenes", [])
    validate_scene_split(train_scenes, dev_scenes)
    if train_scenes != [row["scene"] for row in manifest["train"][:6]] or dev_scenes != [row["scene"] for row in manifest["train"][6:]]:
        raise RuntimeError("G7.8 split must be first-six/final-two manifest order")
    configure_process_git_safe_directory(paths["stream_source"])
    configure_process_git_safe_directory(paths["rlinf_root"])
    if git_head(Path(paths["stream_source"])) != ORIGINAL_STREAM_COMMIT:
        raise RuntimeError("G7.8 requires pinned original StreamVLN")
    probe_topology = OnlineTopology(max_decisions=args.max_oracle_steps)
    run_cfg = {
        **paths,
        "output_dir": str(output),
        "g74_result_sha256": args.g74_result_sha256,
        "manifest": manifest,
        "stream_commit": git_head(Path(paths["stream_source"])),
        "rlinf_commit": git_head(Path(paths["rlinf_root"])),
        "seed": args.seed,
        "max_oracle_steps": args.max_oracle_steps,
        "expert_lookahead_m": args.expert_lookahead_m,
        "coverage_radius_m": args.coverage_radius_m,
        "shield_cooldown_decisions": args.shield_cooldown_decisions,
        "epochs": args.epochs,
        "geometry_epochs": args.geometry_epochs,
        "learning_rate": args.learning_rate,
        "min_unsafe_recall": args.min_unsafe_recall,
        "min_safe_recall": args.min_safe_recall,
        "min_balanced_accuracy": args.min_balanced_accuracy,
        "min_depth_gain": args.min_depth_gain,
        "min_candidate_coverage": args.min_candidate_coverage,
        "max_candidate_p90_error_m": args.max_candidate_p90_error_m,
        "min_success_rate": args.min_success_rate,
        "min_spl": args.min_spl,
        "max_collision_count": args.max_collision_count,
        "progress_noninferiority_m": args.progress_noninferiority_m,
        "macro_candidate_count": len(generate_macro_candidates(probe_topology)),
        "train_scenes": train_scenes,
        "train_dev_scenes": dev_scenes,
    }
    dataset_hashes = validate_dataset_hashes(run_cfg)
    anchor = load_g74_anchor(run_cfg)
    (output / "run_config.json").write_text(json.dumps(run_cfg, indent=2, sort_keys=True) + "\n")
    started = time.time()
    collection_episodes, rows = [], []
    for spec in manifest["train"]:
        episode, episode_rows = collect_episode(run_cfg, spec)
        collection_episodes.append(episode)
        rows.extend(episode_rows)
    if len(collection_episodes) != 8 or max(row["decision"] for row in rows) >= args.max_oracle_steps:
        raise RuntimeError("G7.8 collection guard failed")
    dataset_artifact = save_dataset(output, rows, split)
    train_rows = [row for row in rows if row["scene"] in set(train_scenes)]
    dev_rows = [row for row in rows if row["scene"] in set(dev_scenes)]
    if set(row["scene"] for row in train_rows) & set(row["scene"] for row in dev_rows):
        raise RuntimeError("G7.8 train/dev leakage detected")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    geometry_model = GeometryPassabilityNet()
    geometry_training = train_depth_model(
        geometry_model, train_rows, seed=args.seed, device=device,
        epochs=args.geometry_epochs, learning_rate=args.learning_rate,
    )
    geometry_scores = predict_depth_scores(geometry_model, dev_rows, device=device)
    geometry_dev = select_threshold(
        [row["unsafe"] for row in dev_rows], geometry_scores,
        min_unsafe_recall=args.min_unsafe_recall, min_safe_recall=args.min_safe_recall,
    )
    torch.manual_seed(args.seed)
    depth_model = DepthPassabilityNet()
    depth_training = train_depth_model(
        depth_model, train_rows, seed=args.seed, device=device,
        epochs=args.epochs, learning_rate=args.learning_rate,
    )
    depth_scores = predict_depth_scores(depth_model, dev_rows, device=device)
    depth_dev = select_threshold(
        [row["unsafe"] for row in dev_rows], depth_scores,
        min_unsafe_recall=args.min_unsafe_recall, min_safe_recall=args.min_safe_recall,
    )
    depth_per_scene = per_scene_metrics(
        dev_rows, depth_scores, depth_dev["threshold"]
    )
    geometry_per_scene = per_scene_metrics(
        dev_rows, geometry_scores, geometry_dev["threshold"]
    )
    depth_threshold_sensitivity = threshold_sensitivity(dev_rows, depth_scores)
    geometry_threshold_sensitivity = threshold_sensitivity(dev_rows, geometry_scores)
    geometry_path = output / "geometry_passability.pt"
    depth_path = output / "depth_passability.pt"
    torch.save(geometry_model.state_dict(), geometry_path)
    torch.save(depth_model.state_dict(), depth_path)
    representation_gates = {
        "unsafe_recall": depth_dev["unsafe_recall"] >= args.min_unsafe_recall,
        "safe_recall": depth_dev["safe_recall"] >= args.min_safe_recall,
        "balanced_accuracy": depth_dev["balanced_accuracy"] >= args.min_balanced_accuracy,
        "depth_gain_over_geometry": depth_dev["balanced_accuracy"] >= geometry_dev["balanced_accuracy"] + args.min_depth_gain,
        "scene_disjoint": not bool(set(train_scenes) & set(dev_scenes)),
        "labels_policy_separated": all(
            episode["max_position_restore_error_m"] <= 1e-6
            and episode["max_rotation_restore_error"] <= 1e-6
            for episode in collection_episodes
        ),
    }
    representation_ready = all(representation_gates.values())
    if representation_ready:
        navigation_episodes = [
            navigation_episode(
                run_cfg, spec, depth_model, depth_dev["threshold"], device
            )
            for spec in manifest["train"]
        ]
        navigation_summary = aggregate(
            navigation_episodes, run_cfg["coverage_radius_m"]
        )
        navigation_gates = build_gates(anchor, navigation_summary, run_cfg)
        navigation_status = "executed_after_representation_pass"
        navigation_guard = (
            navigation_summary["episode_count"] == 8
            and navigation_summary["scene_count"] == 8
            and all(
                episode["primitive_steps"] <= args.max_oracle_steps
                for episode in navigation_episodes
            )
        )
    else:
        navigation_episodes = []
        navigation_summary = None
        navigation_gates = None
        navigation_status = "skipped_representation_gate_failed"
        navigation_guard = True
    local_controller_ready = bool(
        representation_ready
        and navigation_gates is not None
        and all(navigation_gates.values())
    )
    all_guards_passed = (
        len(train_rows) + len(dev_rows) == len(rows)
        and summarize_rows(train_rows)["scene_count"] == 6
        and summarize_rows(dev_rows)["scene_count"] == 2
        and set(depth_per_scene) == set(dev_scenes)
        and set(geometry_per_scene) == set(dev_scenes)
        and navigation_guard
    )
    result = {
        "schema_version": 1,
        "kind": "streamvln-g7.8-aligned-depth-local-controller-upper-bound",
        "claim_scope": "train-only aligned-depth local-controller upper bound; depth is an explicit diagnostic/lower-controller input, not an RGB, language, or val_unseen claim",
        "split": "train",
        "val_unseen_episode_count": 0,
        "privileged_shortest_path_used_for_collection_path_and_navigation_selection": True,
        "branch_rollout_labels_used_for_training_only": True,
        "branch_rollout_used_during_navigation_gate": False,
        "navigation_protocol": "turn-observe-depth-translate",
        "aligned_observation_after_actual_turn_prefix": True,
        "navigation_condition": "all-representation-gates-pass",
        "source_contract": {
            "stream_commit": run_cfg["stream_commit"],
            "rlinf_commit": run_cfg["rlinf_commit"],
            "source_kind": "upstream-original",
            "original_stream_model_loaded": False,
            "local_controller_observation": ["aligned_metric_depth_after_actual_turn_prefix", "candidate_distance"],
            "forbidden_policy_inputs": ["shortest_path", "goal_coordinates", "navmesh_oracle", "future_collision"],
            "depth_used": True,
            "depth_role": "explicit-local-controller-diagnostic",
            "memory_adapter": False,
            "secondary_post_training_used": False,
        },
        "dataset_hashes": dataset_hashes,
        "scene_split": split,
        "dataset_artifact": dataset_artifact,
        "collection_episodes": collection_episodes,
        "models": {
            "geometry_only": {
                "training": geometry_training, "dev": geometry_dev,
                "dev_by_scene": geometry_per_scene,
                "threshold_sensitivity": geometry_threshold_sensitivity,
                "checkpoint": str(geometry_path), "checkpoint_sha256": sha256_file(geometry_path),
            },
            "depth_geometry": {
                "training": depth_training, "dev": depth_dev,
                "dev_by_scene": depth_per_scene,
                "threshold_sensitivity": depth_threshold_sensitivity,
                "checkpoint": str(depth_path), "checkpoint_sha256": sha256_file(depth_path),
            },
        },
        "representation_gates": representation_gates,
        "navigation": {
            "status": navigation_status,
            "episodes": navigation_episodes,
            "aggregate": navigation_summary,
            "gates": navigation_gates,
        },
        "local_controller_ready": local_controller_ready,
        "g8_ready": False,
        "routing": (
            "freeze-modular-local-controller-or-test-original-frozen-visual-tower"
            if local_controller_ready
            else "diagnose-gate-controller-behavior"
            if representation_ready
            else "stop-representation-swaps-and-audit-label-horizon-macro-controllability"
        ),
        "all_guards_passed": all_guards_passed,
        "g74_anchor": {
            "path": run_cfg["g74_result"], "sha256": run_cfg["g74_result_sha256"],
            "macro_interrupt": anchor["variants"]["macro_interrupt"]["aggregate"],
        },
        "elapsed_s": time.time() - started,
    }
    result_path = output / "result.json"
    result_path.write_text(json.dumps(jsonable(result), indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "result": str(result_path),
        "result_sha256": sha256_file(result_path),
        "g8_ready": result["g8_ready"],
        "local_controller_ready": local_controller_ready,
        "routing": result["routing"],
        "representation_gates": representation_gates,
        "depth_dev": depth_dev,
        "geometry_dev": geometry_dev,
        "navigation_status": navigation_status,
        "navigation_gates": navigation_gates,
        "navigation_aggregate": navigation_summary,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
