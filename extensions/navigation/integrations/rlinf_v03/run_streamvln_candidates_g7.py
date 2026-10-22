"""Train-only candidate-waypoint and phased-recovery upper-bound diagnostic."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import time

import torch

from run_streamvln_ppo_g4 import make_env, validate_dataset_hashes, validate_manifest
from run_streamvln_topology_g5 import (
    ORIGINAL_STREAM_COMMIT,
    collision_trace,
    git_head,
    jsonable,
    sha256_file,
)
from streamvln_candidates import (
    compile_candidate,
    compile_mode,
    generate_waypoint_candidates,
    select_oracle_candidate,
)
from streamvln_recovery import recovery_required, recovery_target, waypoint_execution_error
from streamvln_topology import (
    NAVIGATE_MODE,
    OnlineTopology,
    RECOVER_MODE,
    STOP_ACTION,
    STOP_MODE,
    executed_prefix,
)


VARIANTS = (
    "continuous_phased_recovery",
    "candidates_plain",
    "candidates_phased_recovery",
)


def validate_identity(identity: dict, spec: dict) -> None:
    if (
        str(identity["episode_id"]) != str(spec["episode_id"])
        or Path(identity["scene_id"]).stem != spec["scene"]
    ):
        raise RuntimeError("G7 train oracle identity differs from fixed manifest")


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("percentile fraction must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def execute_actions(env, topology, proposed, target_local_m):
    if len(proposed) != 4:
        raise RuntimeError("G7 compiler must propose exactly four primitive actions")
    start_pose = (topology.x, topology.z, topology.heading)
    obs_list, rewards, terminations, truncations, infos = env.chunk_step(
        torch.tensor(proposed, dtype=torch.int64).reshape(1, 4, 1)
    )
    executed = executed_prefix(proposed)
    trace = collision_trace(infos[0], len(executed))
    topology.update(executed, trace)
    return {
        "obs": obs_list[0],
        "rewards": rewards[0].detach().cpu().tolist(),
        "info": infos[0],
        "proposed_actions": [int(action) for action in proposed],
        "executed_actions": executed,
        "collisions": trace,
        "terminal": bool(torch.logical_or(terminations, truncations).any()),
        "terminated": bool(terminations.any()),
        "truncated": bool(truncations.any()),
        "waypoint_execution": waypoint_execution_error(
            target_local_m, start_pose, topology
        ),
    }


def oracle_episode(run_cfg: dict, spec: dict, variant: str) -> dict:
    if variant not in VARIANTS:
        raise ValueError(f"unknown G7 variant: {variant}")
    env = make_env(
        run_cfg,
        split="train",
        episode_index=spec["index"],
        max_steps=run_cfg["max_oracle_steps"],
    )
    topology = OnlineTopology(max_decisions=run_cfg["max_oracle_steps"] // 4)
    candidate_variant = variant.startswith("candidates_")
    recovery_enabled = variant.endswith("phased_recovery")
    actions, decisions, last_info = [], [], None
    last_truncated = False
    try:
        obs, identity = env.reset()
        validate_identity(identity, spec)
        for decision_index in range(run_cfg["max_oracle_steps"] // 4):
            label = env.expert_waypoint(run_cfg["expert_lookahead_m"])
            oracle_waypoint = [float(value) for value in label["waypoint_local_m"]]
            candidates = ()
            selected = None
            quantization_error = None
            covered = None
            recovery_override = False
            phase = "navigate"

            if label["stop"]:
                mode = STOP_MODE
                target = [0.0, 0.0]
                proposed, compiler = compile_mode(mode, target, topology)
                phase = compiler["phase"]
            elif recovery_enabled and recovery_required(
                topology, blocked_threshold=run_cfg["blocked_threshold"]
            ):
                mode = RECOVER_MODE
                target = list(recovery_target(topology))
                proposed, compiler = compile_mode(mode, target, topology)
                target = list(compiler["target_local_m"])
                phase = compiler["phase"]
                recovery_override = True
            else:
                mode = NAVIGATE_MODE
                if candidate_variant:
                    candidates = generate_waypoint_candidates(
                        topology,
                        distance_m=run_cfg["candidate_distance_m"],
                        branch_degrees=run_cfg["candidate_branch_degrees"],
                    )
                    selected, quantization_error = select_oracle_candidate(
                        candidates, oracle_waypoint
                    )
                    target = list(selected.vector())
                    covered = quantization_error <= run_cfg["coverage_radius_m"]
                    proposed = compile_candidate(selected, topology)
                    compiler = {"phase": "navigate_candidate"}
                else:
                    target = oracle_waypoint
                    proposed, compiler = compile_mode(mode, target, topology)
                phase = compiler["phase"]

            step = execute_actions(env, topology, proposed, target)
            actions.extend(step["executed_actions"])
            last_info, last_truncated = step["info"], step["truncated"]
            successful_forward = any(
                action == 1 and not collision
                for action, collision in zip(
                    step["executed_actions"], step["collisions"]
                )
            )
            decisions.append(
                {
                    "decision": decision_index,
                    "oracle_stop": bool(label["stop"]),
                    "oracle_waypoint_local_m": oracle_waypoint,
                    "mode": ("stop" if mode == STOP_MODE else "recover" if mode == RECOVER_MODE else "navigate"),
                    "recovery_override": recovery_override,
                    "recovery_phase": phase if recovery_override else None,
                    "recovery_successful_forward": bool(
                        recovery_override and successful_forward
                    ),
                    "candidate_set": [candidate.as_dict() for candidate in candidates],
                    "selected_candidate": None if selected is None else selected.name,
                    "candidate_quantization_error_m": quantization_error,
                    "candidate_covered": covered,
                    "executed_waypoint_local_m": target,
                    "actions": step["executed_actions"],
                    "collisions": step["collisions"],
                    "terminated": step["terminated"],
                    "truncated": step["truncated"],
                    "waypoint_execution": step["waypoint_execution"],
                    "topology": topology.summary(),
                }
            )
            obs = step["obs"]
            if step["terminal"]:
                break

        if last_info is None or "episode" not in last_info:
            raise RuntimeError("G7 oracle diagnostic lacks terminal metrics")
        metrics = {
            key: float(value.item()) for key, value in last_info["episode"].items()
        }
        success = metrics["success"]
        stop_called = bool(actions and actions[-1] == STOP_ACTION)
        if success:
            outcome = "success"
        elif stop_called:
            outcome = "premature_stop"
        elif last_truncated:
            outcome = "timeout"
        else:
            outcome = "other_failure"
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
            "timeout": last_truncated,
            "stop_called": stop_called,
            "outcome": outcome,
            "recovery_override_count": sum(
                decision["recovery_override"] for decision in decisions
            ),
            "actions": actions,
            "decisions": decisions,
            "topology": topology.summary(),
        }
    finally:
        env.close()


def aggregate(episodes: list[dict], coverage_radius_m: float) -> dict:
    decisions = [decision for episode in episodes for decision in episode["decisions"]]
    actions = [action for episode in episodes for action in episode["actions"]]
    candidate_rows = [
        decision
        for decision in decisions
        if decision["candidate_quantization_error_m"] is not None
    ]
    errors = [
        float(decision["candidate_quantization_error_m"])
        for decision in candidate_rows
    ]
    recovery_rows = [decision for decision in decisions if decision["recovery_override"]]
    outcomes = Counter(episode["outcome"] for episode in episodes)
    selected = Counter(
        decision["selected_candidate"]
        for decision in candidate_rows
        if decision["selected_candidate"] is not None
    )

    def avg(key: str) -> float:
        return sum(float(episode[key]) for episode in episodes) / len(episodes)

    return {
        "episode_count": len(episodes),
        "scene_count": len({episode["scene"] for episode in episodes}),
        "success_rate": avg("success"),
        "spl": avg("spl"),
        "navigation_error_m": avg("navigation_error_m"),
        "progress_m": avg("progress_m"),
        "positive_progress_count": sum(episode["progress_m"] > 0 for episode in episodes),
        "path_length_m": avg("path_length_m"),
        "collision_count": avg("collision_count"),
        "primitive_steps": avg("primitive_steps"),
        "timeout_count": sum(episode["timeout"] for episode in episodes),
        "stop_count": sum(episode["stop_called"] for episode in episodes),
        "outcome_counts": dict(outcomes),
        "recovery_override_count": len(recovery_rows),
        "recovery_successful_forward_count": sum(
            decision["recovery_successful_forward"] for decision in recovery_rows
        ),
        "action_counts": {str(action): actions.count(action) for action in range(4)},
        "candidate_decision_count": len(candidate_rows),
        "candidate_selection_counts": dict(selected),
        "candidate_coverage_radius_m": coverage_radius_m,
        "candidate_coverage_rate": (
            sum(decision["candidate_covered"] for decision in candidate_rows)
            / len(candidate_rows)
            if candidate_rows
            else None
        ),
        "candidate_quantization_error_mean_m": (
            sum(errors) / len(errors) if errors else None
        ),
        "candidate_quantization_error_p90_m": percentile(errors, 0.90),
    }


def load_g6_oracle(run_cfg: dict) -> dict:
    path = Path(run_cfg["g6_oracle"]).resolve(strict=True)
    if sha256_file(path) != run_cfg["g6_oracle_sha256"]:
        raise RuntimeError("G7 G6 train-only oracle anchor hash changed")
    oracle = json.loads(path.read_text())
    if (
        oracle.get("kind")
        != "streamvln-g6-train-only-oracle-waypoint-compiler-diagnostic"
        or oracle.get("split") != "train"
        or not oracle.get("privileged_shortest_path_used")
        or not oracle.get("all_guards_passed")
    ):
        raise RuntimeError("G7 G6 oracle anchor contract is invalid")
    expected = [
        (row["index"], str(row["episode_id"]), row["scene"])
        for row in run_cfg["manifest"]["train"]
    ]
    for variant in ("plain_compiler", "recovery_compiler"):
        actual = [
            (row["episode_index"], str(row["episode_id"]), row["scene"])
            for row in oracle["variants"][variant]["episodes"]
        ]
        if actual != expected:
            raise RuntimeError("G7 G6 oracle episode order differs from manifest")
    return oracle


def build_gates(anchors: dict, variants: dict, run_cfg: dict) -> dict:
    plain = anchors["plain_compiler"]["aggregate"]
    g6_recovery = anchors["recovery_compiler"]["aggregate"]
    continuous = variants["continuous_phased_recovery"]["aggregate"]
    candidate = variants["candidates_plain"]["aggregate"]
    recovered = variants["candidates_phased_recovery"]["aggregate"]
    tolerance = 1e-9
    return {
        "continuous_phased_restores_plain_sr": continuous["success_rate"] + tolerance
        >= plain["success_rate"],
        "continuous_phased_improves_g6_recovery_sr": continuous["success_rate"]
        > g6_recovery["success_rate"],
        "continuous_phased_collision_not_worse_than_plain": continuous[
            "collision_count"
        ]
        <= plain["collision_count"] + tolerance,
        "candidate_preserves_plain_sr": candidate["success_rate"] + tolerance
        >= plain["success_rate"],
        "candidate_coverage": candidate["candidate_coverage_rate"]
        >= run_cfg["min_candidate_coverage"],
        "candidate_p90_error": candidate["candidate_quantization_error_p90_m"]
        <= run_cfg["max_candidate_p90_error_m"],
        "candidate_recovery_preserves_sr": recovered["success_rate"] + tolerance
        >= candidate["success_rate"],
        "candidate_recovery_preserves_progress": recovered["progress_m"] + tolerance
        >= candidate["progress_m"],
        "candidate_recovery_collision_not_worse": recovered["collision_count"]
        <= candidate["collision_count"] + tolerance,
        "candidate_recovery_not_immobile": recovered["path_length_m"] + tolerance
        >= run_cfg["min_recovery_path_fraction"] * candidate["path_length_m"],
    }


def parse_args():
    parser = argparse.ArgumentParser()
    for name in (
        "manifest",
        "g6-oracle",
        "g6-oracle-sha256",
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
        "--egl-vendor-json", default="/usr/share/glvnd/egl_vendor.d/50_mesa.json"
    )
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--max-oracle-steps", type=int, default=40)
    parser.add_argument("--expert-lookahead-m", type=float, default=1.0)
    parser.add_argument("--blocked-threshold", type=int, default=2)
    parser.add_argument("--candidate-distance-m", type=float, default=1.0)
    parser.add_argument("--candidate-branch-degrees", type=float, default=30.0)
    parser.add_argument("--coverage-radius-m", type=float, default=0.5)
    parser.add_argument("--min-candidate-coverage", type=float, default=0.75)
    parser.add_argument("--max-candidate-p90-error-m", type=float, default=0.75)
    parser.add_argument("--min-recovery-path-fraction", type=float, default=0.8)
    return parser.parse_args()


def configure_process_git_safe_directory(path):
    count_text = os.environ.get("GIT_CONFIG_COUNT", "0")
    if not count_text.isdigit():
        raise RuntimeError("G7 inherited malformed GIT_CONFIG_COUNT")
    count = int(count_text)
    os.environ["GIT_CONFIG_COUNT"] = str(count + 1)
    os.environ[f"GIT_CONFIG_KEY_{count}"] = "safe.directory"
    os.environ[f"GIT_CONFIG_VALUE_{count}"] = str(Path(path).resolve())


def main():
    args = parse_args()
    if (
        args.max_oracle_steps < 4
        or args.max_oracle_steps % 4
        or args.blocked_threshold < 1
        or args.candidate_distance_m <= 0.0
        or not 0.0 < args.candidate_branch_degrees < 90.0
        or not 0.0 < args.coverage_radius_m
        or not 0.0 <= args.min_candidate_coverage <= 1.0
        or args.max_candidate_p90_error_m <= 0.0
        or not 0.0 < args.min_recovery_path_fraction <= 1.0
    ):
        raise ValueError("G7 bounded oracle configuration is invalid")
    path_names = (
        "manifest",
        "g6_oracle",
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
        raise FileExistsError("G7 output directory must be fresh")
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(paths["manifest"]).read_text())
    validate_manifest(manifest)
    configure_process_git_safe_directory(paths["stream_source"])
    configure_process_git_safe_directory(paths["rlinf_root"])
    if git_head(Path(paths["stream_source"])) != ORIGINAL_STREAM_COMMIT:
        raise RuntimeError("G7 requires the pinned original StreamVLN source")
    run_cfg = {
        **paths,
        "output_dir": str(output),
        "g6_oracle_sha256": args.g6_oracle_sha256,
        "manifest": manifest,
        "stream_commit": git_head(Path(paths["stream_source"])),
        "rlinf_commit": git_head(Path(paths["rlinf_root"])),
        "seed": args.seed,
        "max_oracle_steps": args.max_oracle_steps,
        "expert_lookahead_m": args.expert_lookahead_m,
        "blocked_threshold": args.blocked_threshold,
        "candidate_distance_m": args.candidate_distance_m,
        "candidate_branch_degrees": args.candidate_branch_degrees,
        "coverage_radius_m": args.coverage_radius_m,
        "min_candidate_coverage": args.min_candidate_coverage,
        "max_candidate_p90_error_m": args.max_candidate_p90_error_m,
        "min_recovery_path_fraction": args.min_recovery_path_fraction,
    }
    dataset_hashes = validate_dataset_hashes(run_cfg)
    anchor = load_g6_oracle(run_cfg)
    (output / "run_config.json").write_text(
        json.dumps(run_cfg, indent=2, sort_keys=True) + "\n"
    )
    started = time.time()
    variants = {}
    for variant in VARIANTS:
        episodes = [
            oracle_episode(run_cfg, spec, variant)
            for spec in run_cfg["manifest"]["train"]
        ]
        summary = aggregate(episodes, run_cfg["coverage_radius_m"])
        variants[variant] = {
            "episodes": episodes,
            "aggregate": summary,
            "all_guards_passed": (
                summary["episode_count"] == 8
                and summary["scene_count"] == 8
                and all(len(episode["actions"]) <= run_cfg["max_oracle_steps"] for episode in episodes)
            ),
        }
        (output / f"{variant}.json").write_text(
            json.dumps(jsonable(variants[variant]), indent=2, sort_keys=True) + "\n"
        )

    gates = build_gates(anchor["variants"], variants, run_cfg)
    result = {
        "schema_version": 1,
        "kind": "streamvln-g7-train-only-candidate-and-recovery-upper-bound",
        "claim_scope": "privileged train-only execution upper bound; no learned-policy or val_unseen performance claim",
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
        "g6_train_only_oracle_anchor": {
            "path": run_cfg["g6_oracle"],
            "sha256": run_cfg["g6_oracle_sha256"],
            "variants": anchor["variants"],
        },
        "g7_variants": variants,
        "g8_entry_gates": gates,
        "g8_ready": all(gates.values()),
        "all_guards_passed": all(
            variant["all_guards_passed"] for variant in variants.values()
        ),
        "elapsed_s": time.time() - started,
    }
    result_path = output / "result.json"
    result_path.write_text(json.dumps(jsonable(result), indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "result": str(result_path),
                "result_sha256": sha256_file(result_path),
                "g8_ready": result["g8_ready"],
                "g8_entry_gates": gates,
                "aggregates": {
                    name: value["aggregate"] for name, value in variants.items()
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
