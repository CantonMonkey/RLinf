"""G7.6 train-only observation-conditioned local passability gate."""

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
from streamvln_passability_g76 import (
    GeometryPassabilityNet,
    RGBPassabilityNet,
    predict_scores,
    select_threshold,
    train_model,
    validate_scene_split,
)
from streamvln_topology import OnlineTopology, STOP_ACTION


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
    cfg["habitat_server"] = run_cfg["habitat_passability_server"]
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
                probe = env.probe_candidates(candidates)
                max_position_error = max(max_position_error, probe["max_position_restore_error_m"])
                max_rotation_error = max(max_rotation_error, probe["max_rotation_restore_error"])
                image = obs["rgb"][0].detach().cpu().numpy().astype(np.uint8, copy=True)
                if image.shape != (64, 64, 3):
                    raise ValueError("G7.6 RGB shape changed")
                image_sha = hashlib.sha256(image.tobytes()).hexdigest()
                by_name = {row["name"]: row for row in probe["candidates"]}
                if set(by_name) != {candidate.name for candidate in candidates}:
                    raise RuntimeError("G7.6 probe candidate set changed")
                for candidate in candidates:
                    label_row = by_name[candidate.name]
                    rows.append({
                        "scene": spec["scene"],
                        "episode_id": str(spec["episode_id"]),
                        "decision": decision_index,
                        "candidate": candidate.name,
                        "heading_degrees": candidate.heading_degrees,
                        "distance_m": candidate.distance_m,
                        "unsafe": bool(label_row["unsafe"]),
                        "forward_collisions": int(label_row["forward_collisions"]),
                        "rgb_sha256": image_sha,
                        "rgb": image,
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
        rgb=np.stack([row["rgb"] for row in rows]).astype(np.uint8),
        heading_degrees=np.asarray([row["heading_degrees"] for row in rows], dtype=np.float32),
        distance_m=np.asarray([row["distance_m"] for row in rows], dtype=np.float32),
        unsafe=np.asarray([row["unsafe"] for row in rows], dtype=np.bool_),
        scene_index=np.asarray([scene_index[row["scene"]] for row in rows], dtype=np.int16),
        decision=np.asarray([row["decision"] for row in rows], dtype=np.int16),
    )
    index_path = output / "passability_dataset_index.json"
    index = {
        "schema_version": 1,
        "source": "train-simulator-branch-rollout",
        "policy_inputs": ["rgb_64x64", "candidate_heading", "candidate_distance"],
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


def score_candidates(model, image, candidates, device):
    rows = [{
        "rgb": image,
        "heading_degrees": candidate.heading_degrees,
        "distance_m": candidate.distance_m,
        "unsafe": False,
    } for candidate in candidates]
    scores = predict_scores(model, rows, device=device)
    return {candidate.name: float(score) for candidate, score in zip(candidates, scores)}


def navigation_episode(run_cfg, spec, model, threshold, device):
    env = make_env(run_cfg, split="train", episode_index=spec["index"], max_steps=run_cfg["max_oracle_steps"])
    topology = OnlineTopology(max_decisions=run_cfg["max_oracle_steps"])
    shield = WorldHeadingShield(topology.turn_radians, cooldown_decisions=run_cfg["shield_cooldown_decisions"])
    actions, decisions, last_info = [], [], None
    last_truncated = False
    forced_count = 0
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
            selected = None
            error = None
            scores = {}
            available_names = []
            forced = False
            redirected = False
            if label["stop"]:
                proposed, target = [STOP_ACTION], [0.0, 0.0]
            else:
                candidates = generate_macro_candidates(topology)
                shielded = shield.available(candidates, current_world_heading=topology.heading)
                unrestricted, _ = select_oracle_candidate(candidates, oracle_target)
                image = obs["rgb"][0].detach().cpu().numpy().astype(np.uint8, copy=False)
                scores = score_candidates(model, image, shielded, device)
                available = tuple(candidate for candidate in shielded if scores[candidate.name] < threshold)
                if not available:
                    minimum = min(scores.values())
                    available = tuple(candidate for candidate in shielded if math.isclose(scores[candidate.name], minimum, abs_tol=1e-12))
                    forced = True
                    forced_count += 1
                available_names = [candidate.name for candidate in available]
                selected, error = select_oracle_candidate(available, oracle_target)
                redirected = selected.name != unrestricted.name
                proposed, target = list(selected.actions), list(selected.vector())
            step = execute_sequence(
                env, topology, proposed, target,
                interrupt_on_collision=True,
                remaining_steps=run_cfg["max_oracle_steps"] - len(actions),
            )
            for heading in step["collision_world_headings"]:
                shield.block_world_heading(heading)
            actions.extend(step["executed_actions"])
            last_info, last_truncated = step["info"], step["truncated"]
            decisions.append({
                "decision": decision_index,
                "oracle_stop": bool(label["stop"]),
                "oracle_waypoint_local_m": oracle_target,
                "candidate_set": [candidate.as_dict() for candidate in candidates],
                "available_candidate_names": available_names,
                "unrestricted_candidate": None if selected is None else unrestricted.name,
                "selected_candidate": None if selected is None else selected.name,
                "blacklist_redirected": redirected,
                "candidate_quantization_error_m": error,
                "unrestricted_quantization_error_m": None,
                "candidate_covered": None if error is None else error <= run_cfg["coverage_radius_m"],
                "selected_blocked_forward": False,
                "passability_threshold": threshold,
                "passability_scores": scores,
                "forced_min_risk": forced,
                "executed_waypoint_local_m": target,
                "proposed_actions": list(proposed),
                "execute_count": step["execute_count"],
                "actions": step["executed_actions"],
                "collisions": step["collisions"],
                "collision_world_headings": step["collision_world_headings"],
                "interrupted_on_collision": step["interrupted_on_collision"],
                "terminated": step["terminated"],
                "truncated": step["truncated"],
                "waypoint_execution": step["waypoint_execution"],
                "feasibility": shield.summary(),
                "topology": topology.summary(),
            })
            obs = step["obs"]
            if step["terminal"]:
                break
        if last_info is None or "episode" not in last_info:
            raise RuntimeError("G7.6 navigation gate lacks terminal metrics")
        metrics = {key: float(value.item()) for key, value in last_info["episode"].items()}
        stop_called = bool(actions and actions[-1] == STOP_ACTION)
        success = metrics["success"]
        outcome = "success" if success else "premature_stop" if stop_called else "timeout" if last_truncated else "other_failure"
        return {
            "episode_index": spec["index"],
            "episode_id": str(spec["episode_id"]),
            "scene": spec["scene"],
            "source_geodesic_m": float(spec["source_geodesic_m"]),
            "success": success,
            "spl": metrics["spl"],
            "navigation_error_m": metrics["distance_to_goal"],
            "progress_m": float(spec["source_geodesic_m"]) - metrics["distance_to_goal"],
            "path_length_m": metrics["path_length"],
            "collision_count": int(metrics["collision_count"]),
            "primitive_steps": len(actions),
            "decision_count": len(decisions),
            "timeout": last_truncated,
            "stop_called": stop_called,
            "outcome": outcome,
            "forced_min_risk_count": forced_count,
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
        "habitat-passability-server", "asset-root", "habitat-sim-python",
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
    parser.add_argument("--min-rgb-gain", type=float, default=0.03)
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
        args.min_rgb_gain, args.min_candidate_coverage, args.min_success_rate,
        args.min_spl,
    )
    if (
        args.max_oracle_steps < 8 or args.epochs < 1 or args.geometry_epochs < 1
        or args.learning_rate <= 0 or args.shield_cooldown_decisions < 1
        or any(not 0.0 <= value <= 1.0 for value in probabilities)
        or args.max_candidate_p90_error_m <= 0 or args.max_collision_count < 0
        or args.progress_noninferiority_m < 0
    ):
        raise ValueError("G7.6 bounded configuration is invalid")
    path_names = (
        "manifest", "g74_result", "rlinf_root", "stream_source", "habitat_python",
        "habitat_server", "habitat_passability_server", "asset_root",
        "habitat_sim_python", "habitat_env_lib", "egl_vendor_json",
    )
    paths = {name: str(Path(getattr(args, name)).expanduser().resolve(strict=True)) for name in path_names}
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("G7.6 output directory must be fresh")
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(paths["manifest"]).read_text())
    validate_manifest(manifest)
    split = manifest.get("g76_passability_split", {})
    train_scenes = split.get("train_scenes", [])
    dev_scenes = split.get("train_dev_scenes", [])
    validate_scene_split(train_scenes, dev_scenes)
    if train_scenes != [row["scene"] for row in manifest["train"][:6]] or dev_scenes != [row["scene"] for row in manifest["train"][6:]]:
        raise RuntimeError("G7.6 split must be first-six/final-two manifest order")
    configure_process_git_safe_directory(paths["stream_source"])
    configure_process_git_safe_directory(paths["rlinf_root"])
    if git_head(Path(paths["stream_source"])) != ORIGINAL_STREAM_COMMIT:
        raise RuntimeError("G7.6 requires pinned original StreamVLN")
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
        "min_rgb_gain": args.min_rgb_gain,
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
        raise RuntimeError("G7.6 collection guard failed")
    dataset_artifact = save_dataset(output, rows, split)
    train_rows = [row for row in rows if row["scene"] in set(train_scenes)]
    dev_rows = [row for row in rows if row["scene"] in set(dev_scenes)]
    if set(row["scene"] for row in train_rows) & set(row["scene"] for row in dev_rows):
        raise RuntimeError("G7.6 train/dev leakage detected")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    geometry_model = GeometryPassabilityNet()
    geometry_training = train_model(
        geometry_model, train_rows, seed=args.seed, device=device,
        epochs=args.geometry_epochs, learning_rate=args.learning_rate,
    )
    geometry_scores = predict_scores(geometry_model, dev_rows, device=device)
    geometry_dev = select_threshold(
        [row["unsafe"] for row in dev_rows], geometry_scores,
        min_unsafe_recall=args.min_unsafe_recall, min_safe_recall=args.min_safe_recall,
    )
    torch.manual_seed(args.seed)
    rgb_model = RGBPassabilityNet()
    rgb_training = train_model(
        rgb_model, train_rows, seed=args.seed, device=device,
        epochs=args.epochs, learning_rate=args.learning_rate,
    )
    rgb_scores = predict_scores(rgb_model, dev_rows, device=device)
    rgb_dev = select_threshold(
        [row["unsafe"] for row in dev_rows], rgb_scores,
        min_unsafe_recall=args.min_unsafe_recall, min_safe_recall=args.min_safe_recall,
    )
    geometry_path = output / "geometry_passability.pt"
    rgb_path = output / "rgb_passability.pt"
    torch.save(geometry_model.state_dict(), geometry_path)
    torch.save(rgb_model.state_dict(), rgb_path)
    representation_gates = {
        "unsafe_recall": rgb_dev["unsafe_recall"] >= args.min_unsafe_recall,
        "safe_recall": rgb_dev["safe_recall"] >= args.min_safe_recall,
        "balanced_accuracy": rgb_dev["balanced_accuracy"] >= args.min_balanced_accuracy,
        "rgb_gain_over_geometry": rgb_dev["balanced_accuracy"] >= geometry_dev["balanced_accuracy"] + args.min_rgb_gain,
        "scene_disjoint": not bool(set(train_scenes) & set(dev_scenes)),
        "labels_policy_separated": all(
            episode["max_position_restore_error_m"] <= 1e-6
            and episode["max_rotation_restore_error"] <= 1e-6
            for episode in collection_episodes
        ),
    }
    navigation_episodes = [
        navigation_episode(run_cfg, spec, rgb_model, rgb_dev["threshold"], device)
        for spec in manifest["train"]
    ]
    navigation_summary = aggregate(navigation_episodes, run_cfg["coverage_radius_m"])
    navigation_gates = build_gates(anchor, navigation_summary, run_cfg)
    all_guards_passed = (
        len(train_rows) + len(dev_rows) == len(rows)
        and summarize_rows(train_rows)["scene_count"] == 6
        and summarize_rows(dev_rows)["scene_count"] == 2
        and navigation_summary["episode_count"] == 8
        and navigation_summary["scene_count"] == 8
        and all(episode["primitive_steps"] <= args.max_oracle_steps for episode in navigation_episodes)
    )
    result = {
        "schema_version": 1,
        "kind": "streamvln-g7.6-observation-conditioned-passability-gate",
        "claim_scope": "train-only RGB observability and privileged-goal navigation upper bound; no val_unseen or learned-language-policy claim",
        "split": "train",
        "val_unseen_episode_count": 0,
        "privileged_shortest_path_used_for_collection_path_and_navigation_selection": True,
        "branch_rollout_labels_used_for_training_only": True,
        "branch_rollout_used_during_navigation_gate": False,
        "source_contract": {
            "stream_commit": run_cfg["stream_commit"],
            "rlinf_commit": run_cfg["rlinf_commit"],
            "source_kind": "upstream-original",
            "original_stream_model_loaded": False,
            "policy_observation": ["rgb", "candidate_heading", "candidate_distance"],
            "forbidden_policy_inputs": ["shortest_path", "goal_coordinates", "navmesh_oracle", "future_collision"],
            "depth_used": False,
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
                "checkpoint": str(geometry_path), "checkpoint_sha256": sha256_file(geometry_path),
            },
            "rgb_geometry": {
                "training": rgb_training, "dev": rgb_dev,
                "checkpoint": str(rgb_path), "checkpoint_sha256": sha256_file(rgb_path),
            },
        },
        "representation_gates": representation_gates,
        "navigation": {"episodes": navigation_episodes, "aggregate": navigation_summary, "gates": navigation_gates},
        "g8_ready": all(representation_gates.values()) and all(navigation_gates.values()),
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
        "representation_gates": representation_gates,
        "rgb_dev": rgb_dev,
        "geometry_dev": geometry_dev,
        "navigation_gates": navigation_gates,
        "navigation_aggregate": navigation_summary,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
