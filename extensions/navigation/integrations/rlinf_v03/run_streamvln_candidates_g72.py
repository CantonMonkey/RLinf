"""G7.2 expanded candidate geometry train-only final entry gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from run_streamvln_candidates_g7 import (
    configure_process_git_safe_directory,
    validate_identity,
)
from run_streamvln_candidates_g71 import aggregate, execute_actions
from run_streamvln_ppo_g4 import make_env, validate_dataset_hashes, validate_manifest
from run_streamvln_topology_g5 import (
    ORIGINAL_STREAM_COMMIT,
    git_head,
    jsonable,
    sha256_file,
)
from streamvln_candidates_g71 import (
    compile_dense_fixed,
    generate_dense_candidates,
    select_oracle_dense_candidate,
)
from streamvln_topology import OnlineTopology, STOP_ACTION


DISTANCES_M = (0.5, 0.75, 1.0)
HEADINGS_DEGREES = (0.0, 15.0, -15.0, 30.0, -30.0, 45.0, -45.0, 60.0, -60.0)


def oracle_episode(run_cfg, spec):
    env = make_env(
        run_cfg,
        split="train",
        episode_index=spec["index"],
        max_steps=run_cfg["max_oracle_steps"],
    )
    topology = OnlineTopology(
        max_decisions=run_cfg["max_oracle_steps"] // 4
    )
    actions, decisions, last_info = [], [], None
    last_truncated = False
    try:
        obs, identity = env.reset()
        validate_identity(identity, spec)
        for decision_index in range(run_cfg["max_oracle_steps"] // 4):
            label = env.expert_waypoint(run_cfg["expert_lookahead_m"])
            oracle_waypoint = [float(value) for value in label["waypoint_local_m"]]
            selected = None
            quantization_error = None
            if label["stop"]:
                candidates = ()
                proposed = [STOP_ACTION] * 4
                execute_count = 1
                target = [0.0, 0.0]
            else:
                candidates = generate_dense_candidates(
                    topology,
                    distances_m=DISTANCES_M,
                    headings_degrees=HEADINGS_DEGREES,
                )
                selected, quantization_error = select_oracle_dense_candidate(
                    candidates, oracle_waypoint
                )
                proposed, execute_count = compile_dense_fixed(
                    selected, topology
                )
                target = list(selected.vector())
            step = execute_actions(
                env, topology, proposed, execute_count, target
            )
            actions.extend(step["executed_actions"])
            last_info, last_truncated = step["info"], step["truncated"]
            decisions.append(
                {
                    "decision": decision_index,
                    "oracle_stop": bool(label["stop"]),
                    "oracle_waypoint_local_m": oracle_waypoint,
                    "candidate_set": [
                        candidate.as_dict() for candidate in candidates
                    ],
                    "available_candidate_names": [
                        candidate.name for candidate in candidates
                    ],
                    "unrestricted_candidate": (
                        None if selected is None else selected.name
                    ),
                    "selected_candidate": (
                        None if selected is None else selected.name
                    ),
                    "blacklist_redirected": False,
                    "candidate_quantization_error_m": quantization_error,
                    "unrestricted_quantization_error_m": quantization_error,
                    "candidate_covered": (
                        None
                        if quantization_error is None
                        else quantization_error <= run_cfg["coverage_radius_m"]
                    ),
                    "selected_blocked_forward": False,
                    "executed_waypoint_local_m": target,
                    "proposed_actions": step["proposed_actions"],
                    "execute_count": step["execute_count"],
                    "actions": step["executed_actions"],
                    "collisions": step["collisions"],
                    "terminated": step["terminated"],
                    "truncated": step["truncated"],
                    "waypoint_execution": step["waypoint_execution"],
                    "feasibility": {
                        "cooldown_decisions": 0,
                        "blocked_direction_bins": {},
                    },
                    "topology": topology.summary(),
                }
            )
            obs = step["obs"]
            if step["terminal"]:
                break
        if last_info is None or "episode" not in last_info:
            raise RuntimeError("G7.2 oracle diagnostic lacks terminal metrics")
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
            "actions": actions,
            "decisions": decisions,
            "topology": topology.summary(),
        }
    finally:
        env.close()


def load_g71_anchor(run_cfg):
    path = Path(run_cfg["g71_result"]).resolve(strict=True)
    if sha256_file(path) != run_cfg["g71_result_sha256"]:
        raise RuntimeError("G7.2 G7.1 result anchor hash changed")
    result = json.loads(path.read_text())
    if (
        result.get("kind")
        != "streamvln-g7.1-adaptive-duration-and-collision-feasibility"
        or result.get("val_unseen_episode_count") != 0
        or not result.get("all_guards_passed")
    ):
        raise RuntimeError("G7.2 anchor contract is invalid")
    return result


def build_gates(anchor, summary, run_cfg):
    baseline = anchor["variants"]["dense_fixed4"]["aggregate"]
    successes = int(round(summary["success_rate"] * summary["episode_count"]))
    return {
        "coverage": summary["candidate_coverage_rate"]
        >= run_cfg["min_candidate_coverage"],
        "p90_error": summary["candidate_quantization_error_p90_m"]
        <= run_cfg["max_candidate_p90_error_m"],
        "success_rate": summary["success_rate"] >= run_cfg["min_success_rate"],
        "spl": summary["spl"] >= run_cfg["min_spl"],
        "collision": summary["collision_count"] <= run_cfg["max_collision_count"],
        "progress_noninferior": summary["progress_m"]
        >= baseline["progress_m"] - run_cfg["progress_noninferiority_m"],
        "successes_stop": summary["successful_stop_count"] == successes,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    for name in (
        "manifest",
        "g71-result",
        "g71-result-sha256",
        "output-dir",
        "rlinf-root",
        "stream-source",
        "habitat-python",
        "habitat-server",
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
    parser.add_argument("--max-oracle-steps", type=int, default=40)
    parser.add_argument("--expert-lookahead-m", type=float, default=1.0)
    parser.add_argument("--coverage-radius-m", type=float, default=0.5)
    parser.add_argument("--min-candidate-coverage", type=float, default=0.90)
    parser.add_argument("--max-candidate-p90-error-m", type=float, default=0.50)
    parser.add_argument("--min-success-rate", type=float, default=0.75)
    parser.add_argument("--min-spl", type=float, default=0.75)
    parser.add_argument("--max-collision-count", type=float, default=1.375)
    parser.add_argument("--progress-noninferiority-m", type=float, default=0.10)
    return parser.parse_args()


def main():
    args = parse_args()
    if (
        args.max_oracle_steps < 4
        or args.max_oracle_steps % 4
        or not 0.0 < args.coverage_radius_m
        or not 0.0 <= args.min_candidate_coverage <= 1.0
        or args.max_candidate_p90_error_m <= 0.0
        or not 0.0 <= args.min_success_rate <= 1.0
        or not 0.0 <= args.min_spl <= 1.0
        or args.max_collision_count < 0.0
        or args.progress_noninferiority_m < 0.0
    ):
        raise ValueError("G7.2 bounded configuration is invalid")
    path_names = (
        "manifest",
        "g71_result",
        "rlinf_root",
        "stream_source",
        "habitat_python",
        "habitat_server",
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
        raise FileExistsError("G7.2 output directory must be fresh")
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(paths["manifest"]).read_text())
    validate_manifest(manifest)
    configure_process_git_safe_directory(paths["stream_source"])
    configure_process_git_safe_directory(paths["rlinf_root"])
    if git_head(Path(paths["stream_source"])) != ORIGINAL_STREAM_COMMIT:
        raise RuntimeError("G7.2 requires the pinned original StreamVLN source")
    run_cfg = {
        **paths,
        "output_dir": str(output),
        "g71_result_sha256": args.g71_result_sha256,
        "manifest": manifest,
        "stream_commit": git_head(Path(paths["stream_source"])),
        "rlinf_commit": git_head(Path(paths["rlinf_root"])),
        "seed": args.seed,
        "max_oracle_steps": args.max_oracle_steps,
        "expert_lookahead_m": args.expert_lookahead_m,
        "coverage_radius_m": args.coverage_radius_m,
        "min_candidate_coverage": args.min_candidate_coverage,
        "max_candidate_p90_error_m": args.max_candidate_p90_error_m,
        "min_success_rate": args.min_success_rate,
        "min_spl": args.min_spl,
        "max_collision_count": args.max_collision_count,
        "progress_noninferiority_m": args.progress_noninferiority_m,
        "candidate_distances_m": list(DISTANCES_M),
        "candidate_headings_degrees": list(HEADINGS_DEGREES),
    }
    dataset_hashes = validate_dataset_hashes(run_cfg)
    anchor = load_g71_anchor(run_cfg)
    (output / "run_config.json").write_text(
        json.dumps(run_cfg, indent=2, sort_keys=True) + "\n"
    )
    started = time.time()
    episodes = [
        oracle_episode(run_cfg, spec) for spec in run_cfg["manifest"]["train"]
    ]
    summary = aggregate(episodes, run_cfg["coverage_radius_m"])
    variant = {
        "episodes": episodes,
        "aggregate": summary,
        "all_guards_passed": (
            summary["episode_count"] == 8
            and summary["scene_count"] == 8
            and all(
                episode["primitive_steps"] <= run_cfg["max_oracle_steps"]
                for episode in episodes
            )
        ),
    }
    gates = build_gates(anchor, summary, run_cfg)
    result = {
        "schema_version": 1,
        "kind": "streamvln-g7.2-expanded-candidate-final-gate",
        "claim_scope": "privileged train-only candidate geometry upper bound; no learned-policy or val_unseen claim",
        "split": "train",
        "val_unseen_episode_count": 0,
        "privileged_shortest_path_used": True,
        "source_contract": {
            "stream_commit": run_cfg["stream_commit"],
            "rlinf_commit": run_cfg["rlinf_commit"],
            "source_kind": "upstream-original",
            "model_loaded": False,
            "memory_adapter": False,
            "secondary_post_training_used": False,
        },
        "dataset_hashes": dataset_hashes,
        "g71_anchor": {
            "path": run_cfg["g71_result"],
            "sha256": run_cfg["g71_result_sha256"],
            "dense_fixed4": anchor["variants"]["dense_fixed4"]["aggregate"],
        },
        "expanded_fixed4": variant,
        "g8_entry_gates": gates,
        "g8_ready": all(gates.values()),
        "all_guards_passed": variant["all_guards_passed"],
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
                "g8_ready": result["g8_ready"],
                "gates": gates,
                "aggregate": summary,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
