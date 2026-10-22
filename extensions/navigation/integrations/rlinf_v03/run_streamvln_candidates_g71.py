"""G7.1 adaptive-duration and collision-feasibility train-only diagnostic."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import time

import torch

from run_streamvln_candidates_g7 import (
    configure_process_git_safe_directory,
    percentile,
    validate_identity,
)
from run_streamvln_ppo_g4 import make_env, validate_dataset_hashes, validate_manifest
from run_streamvln_topology_g5 import (
    ORIGINAL_STREAM_COMMIT,
    collision_trace,
    git_head,
    jsonable,
    sha256_file,
)
from streamvln_candidates_g71 import (
    DirectionFeasibility,
    compile_dense_adaptive,
    compile_dense_fixed,
    generate_dense_candidates,
    select_oracle_dense_candidate,
)
from streamvln_recovery import waypoint_execution_error
from streamvln_topology import OnlineTopology, STOP_ACTION, executed_prefix


VARIANTS = ("dense_fixed4", "dense_adaptive", "dense_adaptive_blacklist")


def execute_actions(env, topology, proposed, execute_count, target_local_m):
    if len(proposed) != 4 or not 1 <= execute_count <= 4:
        raise RuntimeError("G7.1 action contract requires four proposed and 1..4 executed")
    start_pose = (topology.x, topology.z, topology.heading)
    obs_list, rewards, terminations, truncations, infos = env.chunk_step(
        torch.tensor(proposed, dtype=torch.int64).reshape(1, 4, 1),
        execute_count=execute_count,
    )
    executed = executed_prefix(proposed[:execute_count])
    trace = collision_trace(infos[0], len(executed))
    topology.update(executed, trace)
    return {
        "obs": obs_list[0],
        "rewards": rewards[0].detach().cpu().tolist(),
        "info": infos[0],
        "proposed_actions": [int(action) for action in proposed],
        "executed_actions": executed,
        "execute_count": len(executed),
        "collisions": trace,
        "terminal": bool(torch.logical_or(terminations, truncations).any()),
        "terminated": bool(terminations.any()),
        "truncated": bool(truncations.any()),
        "waypoint_execution": waypoint_execution_error(
            target_local_m, start_pose, topology
        ),
    }


def oracle_episode(run_cfg, spec, variant):
    if variant not in VARIANTS:
        raise ValueError(f"unknown G7.1 variant: {variant}")
    env = make_env(
        run_cfg,
        split="train",
        episode_index=spec["index"],
        max_steps=run_cfg["max_oracle_steps"],
    )
    topology = OnlineTopology(max_decisions=run_cfg["max_oracle_decisions"])
    tracker = DirectionFeasibility(
        cooldown_decisions=run_cfg["blacklist_cooldown_decisions"]
    )
    adaptive = variant != "dense_fixed4"
    blacklist = variant == "dense_adaptive_blacklist"
    actions, decisions, last_info = [], [], None
    last_truncated = False
    try:
        obs, identity = env.reset()
        validate_identity(identity, spec)
        for decision_index in range(run_cfg["max_oracle_decisions"]):
            tracker.begin_decision()
            label = env.expert_waypoint(run_cfg["expert_lookahead_m"])
            oracle_waypoint = [float(value) for value in label["waypoint_local_m"]]
            all_candidates = ()
            available = ()
            unrestricted = None
            selected = None
            quantization_error = None
            unrestricted_error = None
            redirected = False
            if label["stop"]:
                proposed = [STOP_ACTION] * 4
                execute_count = 1
                target = [0.0, 0.0]
            else:
                all_candidates = generate_dense_candidates(topology)
                unrestricted, unrestricted_error = select_oracle_dense_candidate(
                    all_candidates, oracle_waypoint
                )
                available = (
                    tracker.available(all_candidates) if blacklist else all_candidates
                )
                selected, quantization_error = select_oracle_dense_candidate(
                    available, oracle_waypoint
                )
                redirected = selected.name != unrestricted.name
                target = list(selected.vector())
                if adaptive:
                    proposed, execute_count = compile_dense_adaptive(
                        selected, topology
                    )
                else:
                    proposed, execute_count = compile_dense_fixed(
                        selected, topology
                    )
            step = execute_actions(
                env, topology, proposed, execute_count, target
            )
            blocked_forward = False
            if blacklist and selected is not None:
                blocked_forward = tracker.update(
                    selected, step["executed_actions"], step["collisions"]
                )
            actions.extend(step["executed_actions"])
            last_info, last_truncated = step["info"], step["truncated"]
            decisions.append(
                {
                    "decision": decision_index,
                    "oracle_stop": bool(label["stop"]),
                    "oracle_waypoint_local_m": oracle_waypoint,
                    "candidate_set": [
                        candidate.as_dict() for candidate in all_candidates
                    ],
                    "available_candidate_names": [
                        candidate.name for candidate in available
                    ],
                    "unrestricted_candidate": (
                        None if unrestricted is None else unrestricted.name
                    ),
                    "selected_candidate": None if selected is None else selected.name,
                    "blacklist_redirected": redirected,
                    "candidate_quantization_error_m": quantization_error,
                    "unrestricted_quantization_error_m": unrestricted_error,
                    "candidate_covered": (
                        None
                        if quantization_error is None
                        else quantization_error <= run_cfg["coverage_radius_m"]
                    ),
                    "selected_blocked_forward": blocked_forward,
                    "executed_waypoint_local_m": target,
                    "proposed_actions": step["proposed_actions"],
                    "execute_count": step["execute_count"],
                    "actions": step["executed_actions"],
                    "collisions": step["collisions"],
                    "terminated": step["terminated"],
                    "truncated": step["truncated"],
                    "waypoint_execution": step["waypoint_execution"],
                    "feasibility": tracker.summary(),
                    "topology": topology.summary(),
                }
            )
            obs = step["obs"]
            if step["terminal"]:
                break
        if last_info is None or "episode" not in last_info:
            raise RuntimeError("G7.1 oracle diagnostic lacks terminal metrics")
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
            "feasibility": tracker.summary(),
        }
    finally:
        env.close()


def aggregate(episodes, coverage_radius_m):
    decisions = [
        decision for episode in episodes for decision in episode["decisions"]
    ]
    candidate_rows = [
        decision
        for decision in decisions
        if decision["candidate_quantization_error_m"] is not None
    ]
    errors = [
        float(decision["candidate_quantization_error_m"])
        for decision in candidate_rows
    ]
    realized_errors = [
        float(decision["waypoint_execution"]["l2_m"])
        for decision in candidate_rows
    ]
    actions = [action for episode in episodes for action in episode["actions"]]

    def avg(key):
        return sum(float(episode[key]) for episode in episodes) / len(episodes)

    successes = [episode for episode in episodes if episode["success"]]
    selected = Counter(
        decision["selected_candidate"] for decision in candidate_rows
    )
    duration_counts = Counter(
        decision["execute_count"] for decision in decisions
    )
    return {
        "episode_count": len(episodes),
        "scene_count": len({episode["scene"] for episode in episodes}),
        "success_rate": avg("success"),
        "spl": avg("spl"),
        "navigation_error_m": avg("navigation_error_m"),
        "progress_m": avg("progress_m"),
        "positive_progress_count": sum(
            episode["progress_m"] > 0 for episode in episodes
        ),
        "path_length_m": avg("path_length_m"),
        "collision_count": avg("collision_count"),
        "primitive_steps": avg("primitive_steps"),
        "decision_count": avg("decision_count"),
        "timeout_count": sum(episode["timeout"] for episode in episodes),
        "stop_count": sum(episode["stop_called"] for episode in episodes),
        "successful_stop_count": sum(
            episode["stop_called"] for episode in successes
        ),
        "outcome_counts": dict(
            Counter(episode["outcome"] for episode in episodes)
        ),
        "action_counts": {
            str(action): actions.count(action) for action in range(4)
        },
        "execute_count_distribution": {
            str(key): value for key, value in sorted(duration_counts.items())
        },
        "candidate_decision_count": len(candidate_rows),
        "candidate_selection_counts": dict(selected),
        "candidate_coverage_radius_m": coverage_radius_m,
        "candidate_coverage_rate": sum(
            decision["candidate_covered"] for decision in candidate_rows
        )
        / len(candidate_rows),
        "candidate_quantization_error_mean_m": sum(errors) / len(errors),
        "candidate_quantization_error_p90_m": percentile(errors, 0.90),
        "candidate_realized_error_mean_m": sum(realized_errors)
        / len(realized_errors),
        "candidate_realized_error_p90_m": percentile(realized_errors, 0.90),
        "blacklist_redirect_count": sum(
            decision["blacklist_redirected"] for decision in candidate_rows
        ),
        "selected_blocked_forward_count": sum(
            decision["selected_blocked_forward"] for decision in candidate_rows
        ),
    }


def load_g7_anchor(run_cfg):
    path = Path(run_cfg["g7_result"]).resolve(strict=True)
    if sha256_file(path) != run_cfg["g7_result_sha256"]:
        raise RuntimeError("G7.1 G7 result anchor hash changed")
    result = json.loads(path.read_text())
    if (
        result.get("kind")
        != "streamvln-g7-train-only-candidate-and-recovery-upper-bound"
        or result.get("val_unseen_episode_count") != 0
        or not result.get("all_guards_passed")
    ):
        raise RuntimeError("G7.1 anchor contract is invalid")
    expected = [
        (row["index"], str(row["episode_id"]), row["scene"])
        for row in run_cfg["manifest"]["train"]
    ]
    for variant in result["g7_variants"].values():
        actual = [
            (row["episode_index"], str(row["episode_id"]), row["scene"])
            for row in variant["episodes"]
        ]
        if actual != expected:
            raise RuntimeError("G7.1 anchor episode order differs")
    return result


def build_gates(anchor, variants, run_cfg):
    plain = anchor["g6_train_only_oracle_anchor"]["variants"][
        "plain_compiler"
    ]["aggregate"]
    fixed = variants["dense_fixed4"]["aggregate"]
    adaptive = variants["dense_adaptive"]["aggregate"]
    blacklist = variants["dense_adaptive_blacklist"]["aggregate"]
    tolerance = 1e-9
    return {
        "dense_coverage": adaptive["candidate_coverage_rate"]
        >= run_cfg["min_candidate_coverage"],
        "dense_p90_error": adaptive["candidate_quantization_error_p90_m"]
        <= run_cfg["max_candidate_p90_error_m"],
        "adaptive_preserves_plain_sr": adaptive["success_rate"] + tolerance
        >= plain["success_rate"],
        "adaptive_not_worse_than_dense_fixed_sr": adaptive["success_rate"]
        + tolerance
        >= fixed["success_rate"],
        "adaptive_not_worse_than_dense_fixed_progress": adaptive["progress_m"]
        + tolerance
        >= fixed["progress_m"],
        "adaptive_successes_stop": adaptive["successful_stop_count"]
        == int(round(adaptive["success_rate"] * adaptive["episode_count"])),
        "blacklist_preserves_sr": blacklist["success_rate"] + tolerance
        >= adaptive["success_rate"],
        "blacklist_preserves_progress": blacklist["progress_m"] + tolerance
        >= adaptive["progress_m"],
        "blacklist_collision_not_worse": blacklist["collision_count"]
        <= adaptive["collision_count"] + tolerance,
        "blacklist_exercised": blacklist["blacklist_redirect_count"] > 0,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    for name in (
        "manifest",
        "g7-result",
        "g7-result-sha256",
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
    parser.add_argument("--max-oracle-decisions", type=int, default=20)
    parser.add_argument("--expert-lookahead-m", type=float, default=1.0)
    parser.add_argument("--coverage-radius-m", type=float, default=0.5)
    parser.add_argument("--min-candidate-coverage", type=float, default=0.90)
    parser.add_argument("--max-candidate-p90-error-m", type=float, default=0.50)
    parser.add_argument("--blacklist-cooldown-decisions", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    if (
        args.max_oracle_steps < 4
        or args.max_oracle_decisions < 1
        or not 0.0 < args.coverage_radius_m
        or not 0.0 <= args.min_candidate_coverage <= 1.0
        or args.max_candidate_p90_error_m <= 0.0
        or args.blacklist_cooldown_decisions < 1
    ):
        raise ValueError("G7.1 bounded configuration is invalid")
    path_names = (
        "manifest",
        "g7_result",
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
        raise FileExistsError("G7.1 output directory must be fresh")
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(paths["manifest"]).read_text())
    validate_manifest(manifest)
    configure_process_git_safe_directory(paths["stream_source"])
    configure_process_git_safe_directory(paths["rlinf_root"])
    if git_head(Path(paths["stream_source"])) != ORIGINAL_STREAM_COMMIT:
        raise RuntimeError("G7.1 requires the pinned original StreamVLN source")
    run_cfg = {
        **paths,
        "output_dir": str(output),
        "g7_result_sha256": args.g7_result_sha256,
        "manifest": manifest,
        "stream_commit": git_head(Path(paths["stream_source"])),
        "rlinf_commit": git_head(Path(paths["rlinf_root"])),
        "seed": args.seed,
        "max_oracle_steps": args.max_oracle_steps,
        "max_oracle_decisions": args.max_oracle_decisions,
        "expert_lookahead_m": args.expert_lookahead_m,
        "coverage_radius_m": args.coverage_radius_m,
        "min_candidate_coverage": args.min_candidate_coverage,
        "max_candidate_p90_error_m": args.max_candidate_p90_error_m,
        "blacklist_cooldown_decisions": args.blacklist_cooldown_decisions,
    }
    dataset_hashes = validate_dataset_hashes(run_cfg)
    anchor = load_g7_anchor(run_cfg)
    (output / "run_config.json").write_text(
        json.dumps(run_cfg, indent=2, sort_keys=True) + "\n"
    )
    started = time.time()
    variants = {}
    for name in VARIANTS:
        episodes = [
            oracle_episode(run_cfg, spec, name)
            for spec in run_cfg["manifest"]["train"]
        ]
        summary = aggregate(episodes, run_cfg["coverage_radius_m"])
        variants[name] = {
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
        (output / f"{name}.json").write_text(
            json.dumps(jsonable(variants[name]), indent=2, sort_keys=True) + "\n"
        )

    gates = build_gates(anchor, variants, run_cfg)
    result = {
        "schema_version": 1,
        "kind": "streamvln-g7.1-adaptive-duration-and-collision-feasibility",
        "claim_scope": "privileged train-only executor upper bound; no learned-policy or val_unseen claim",
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
        "g7_anchor": {
            "path": run_cfg["g7_result"],
            "sha256": run_cfg["g7_result_sha256"],
        },
        "variants": variants,
        "g8_entry_gates": gates,
        "g8_ready": all(gates.values()),
        "all_guards_passed": all(
            variant["all_guards_passed"] for variant in variants.values()
        ),
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
                "aggregates": {
                    name: variant["aggregate"]
                    for name, variant in variants.items()
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
