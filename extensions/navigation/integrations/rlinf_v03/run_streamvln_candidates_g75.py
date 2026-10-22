"""G7.5 angular collision-sector train-only final safety gate."""

from __future__ import annotations

import argparse
import json
import math
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
from streamvln_candidates_g74 import (
    generate_macro_candidates,
    select_oracle_candidate,
    trace_collision_world_headings,
)
from streamvln_candidates_g75 import AngularSectorShield
from streamvln_recovery import waypoint_execution_error
from streamvln_topology import (
    FORWARD_ACTION,
    LEFT_ACTION,
    OnlineTopology,
    RIGHT_ACTION,
    STOP_ACTION,
)


VARIANTS = ("sector15", "sector30")


def execute_sequence(
    env,
    topology,
    sequence,
    target_local_m,
    *,
    interrupt_on_collision,
    remaining_steps,
):
    if not sequence or remaining_steps < 1:
        raise ValueError("G7.5 execution requires a nonempty sequence and budget")
    planned = [int(action) for action in sequence[:remaining_steps]]
    start_pose = (topology.x, topology.z, topology.heading)
    executed_actions = []
    collisions = []
    collision_world_headings = []
    last_step = None
    offset = 0
    interrupted = False
    while offset < len(planned):
        width = 1 if interrupt_on_collision else min(4, len(planned) - offset)
        chunk = planned[offset : offset + width]
        proposed = chunk + [STOP_ACTION] * (4 - len(chunk))
        segment_start_heading = topology.heading
        step = execute_actions(
            env,
            topology,
            proposed,
            len(chunk),
            [0.0, 0.0],
        )
        last_step = step
        collision_world_headings.extend(
            trace_collision_world_headings(
                step["executed_actions"],
                step["collisions"],
                start_world_heading=segment_start_heading,
                turn_radians=topology.turn_radians,
            )
        )
        executed_actions.extend(step["executed_actions"])
        collisions.extend(step["collisions"])
        offset += len(step["executed_actions"])
        if step["terminal"]:
            break
        if interrupt_on_collision and collision_world_headings:
            interrupted = True
            break
    if last_step is None:
        raise RuntimeError("G7.5 executor produced no environment step")
    return {
        "obs": last_step["obs"],
        "info": last_step["info"],
        "terminal": last_step["terminal"],
        "terminated": last_step["terminated"],
        "truncated": last_step["truncated"],
        "planned_actions": planned,
        "executed_actions": executed_actions,
        "collisions": collisions,
        "execute_count": len(executed_actions),
        "collision_world_headings": collision_world_headings,
        "interrupted_on_collision": interrupted,
        "waypoint_execution": waypoint_execution_error(
            target_local_m, start_pose, topology
        ),
    }


def oracle_episode(run_cfg, spec, variant):
    if variant not in VARIANTS:
        raise ValueError(f"unknown G7.5 variant: {variant}")
    macro = True
    interrupt = True
    env = make_env(
        run_cfg,
        split="train",
        episode_index=spec["index"],
        max_steps=run_cfg["max_oracle_steps"],
    )
    topology = OnlineTopology(max_decisions=run_cfg["max_oracle_steps"])
    sector_radius_bins = 1 if variant == "sector15" else 2
    shield = AngularSectorShield(
        topology.turn_radians,
        sector_radius_bins=sector_radius_bins,
        cooldown_decisions=run_cfg["shield_cooldown_decisions"],
    )
    actions, decisions, last_info = [], [], None
    last_truncated = False
    try:
        obs, identity = env.reset()
        validate_identity(identity, spec)
        for decision_index in range(run_cfg["max_oracle_steps"]):
            if len(actions) >= run_cfg["max_oracle_steps"]:
                break
            if interrupt:
                shield.begin_decision()
            label = env.expert_waypoint(run_cfg["expert_lookahead_m"])
            oracle_waypoint = [float(value) for value in label["waypoint_local_m"]]
            all_candidates = ()
            available = ()
            unrestricted = None
            selected = None
            unrestricted_error = None
            quantization_error = None
            redirected = False
            if label["stop"]:
                proposed = [STOP_ACTION]
                target = [0.0, 0.0]
            else:
                all_candidates = generate_macro_candidates(topology)

                unrestricted, unrestricted_error = select_oracle_candidate(
                    all_candidates, oracle_waypoint
                )
                available = (
                    shield.available(
                        all_candidates,
                        current_world_heading=topology.heading,
                    )
                    if interrupt
                    else all_candidates
                )
                selected, quantization_error = select_oracle_candidate(
                    available, oracle_waypoint
                )
                redirected = selected.name != unrestricted.name
                proposed = list(selected.actions)
                target = list(selected.vector())
            step = execute_sequence(
                env,
                topology,
                proposed,
                target,
                interrupt_on_collision=interrupt,
                remaining_steps=run_cfg["max_oracle_steps"] - len(actions),
            )
            if interrupt:
                for world_heading in step["collision_world_headings"]:
                    shield.block_world_heading(world_heading)
            actions.extend(step["executed_actions"])
            last_info, last_truncated = step["info"], step["truncated"]
            decisions.append(
                {
                    "decision": decision_index,
                    "variant": variant,
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
                    "selected_candidate": (
                        None if selected is None else selected.name
                    ),
                    "blacklist_redirected": redirected,
                    "candidate_quantization_error_m": quantization_error,
                    "unrestricted_quantization_error_m": unrestricted_error,
                    "candidate_covered": (
                        None
                        if quantization_error is None
                        else quantization_error <= run_cfg["coverage_radius_m"]
                    ),
                    "selected_blocked_forward": False,
                    "executed_waypoint_local_m": target,
                    "proposed_actions": list(proposed),
                    "execute_count": step["execute_count"],
                    "actions": step["executed_actions"],
                    "collisions": step["collisions"],
                    "collision_world_headings": step[
                        "collision_world_headings"
                    ],
                    "interrupted_on_collision": step[
                        "interrupted_on_collision"
                    ],
                    "terminated": step["terminated"],
                    "truncated": step["truncated"],
                    "waypoint_execution": step["waypoint_execution"],
                    "feasibility": shield.summary(),
                    "topology": topology.summary(),
                }
            )
            obs = step["obs"]
            if step["terminal"]:
                break
        if last_info is None or "episode" not in last_info:
            raise RuntimeError("G7.5 oracle diagnostic lacks terminal metrics")
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
            "feasibility": shield.summary(),
        }
    finally:
        env.close()


def load_g74_anchor(run_cfg):
    path = Path(run_cfg["g74_result"]).resolve(strict=True)
    if sha256_file(path) != run_cfg["g74_result_sha256"]:
        raise RuntimeError("G7.5 G7.4 result anchor hash changed")
    result = json.loads(path.read_text())
    if (
        result.get("kind")
        != "streamvln-g7.4-macro-interrupt-factorial-final-gate"
        or result.get("val_unseen_episode_count") != 0
        or not result.get("all_guards_passed")
    ):
        raise RuntimeError("G7.5 G7.4 anchor contract is invalid")
    return result


def build_gates(anchor, summary, run_cfg):
    baseline = anchor["g73_anchor"]["g71_dense_fixed4"]
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
        "g74-result",
        "g74-result-sha256",
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
    parser.add_argument("--shield-cooldown-decisions", type=int, default=2)
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
        args.max_oracle_steps < 8
        or not 0.0 < args.coverage_radius_m
        or args.shield_cooldown_decisions < 1
        or not 0.0 <= args.min_candidate_coverage <= 1.0
        or args.max_candidate_p90_error_m <= 0.0
        or not 0.0 <= args.min_success_rate <= 1.0
        or not 0.0 <= args.min_spl <= 1.0
        or args.max_collision_count < 0.0
        or args.progress_noninferiority_m < 0.0
    ):
        raise ValueError("G7.5 bounded configuration is invalid")
    path_names = (
        "manifest",
        "g74_result",
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
        raise FileExistsError("G7.5 output directory must be fresh")
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(paths["manifest"]).read_text())
    validate_manifest(manifest)
    configure_process_git_safe_directory(paths["stream_source"])
    configure_process_git_safe_directory(paths["rlinf_root"])
    if git_head(Path(paths["stream_source"])) != ORIGINAL_STREAM_COMMIT:
        raise RuntimeError("G7.5 requires the pinned original StreamVLN source")
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
        "min_candidate_coverage": args.min_candidate_coverage,
        "max_candidate_p90_error_m": args.max_candidate_p90_error_m,
        "min_success_rate": args.min_success_rate,
        "min_spl": args.min_spl,
        "max_collision_count": args.max_collision_count,
        "progress_noninferiority_m": args.progress_noninferiority_m,
        "macro_candidate_count": len(generate_macro_candidates(probe_topology)),
        "variants": list(VARIANTS),
    }
    dataset_hashes = validate_dataset_hashes(run_cfg)
    anchor = load_g74_anchor(run_cfg)
    (output / "run_config.json").write_text(
        json.dumps(run_cfg, indent=2, sort_keys=True) + "\n"
    )
    started = time.time()
    variants = {}
    for variant_name in VARIANTS:
        episodes = [
            oracle_episode(run_cfg, spec, variant_name)
            for spec in run_cfg["manifest"]["train"]
        ]
        summary = aggregate(episodes, run_cfg["coverage_radius_m"])
        variants[variant_name] = {
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
    gates_by_variant = {
        name: build_gates(anchor, value["aggregate"], run_cfg)
        for name, value in variants.items()
    }
    selected_variant = next(
        (name for name in VARIANTS if all(gates_by_variant[name].values())),
        None,
    )
    entry_variant = selected_variant or VARIANTS[-1]
    gates = gates_by_variant[entry_variant]
    result = {
        "schema_version": 1,
        "kind": "streamvln-g7.5-angular-sector-final-safety-gate",
        "claim_scope": "privileged train-only angular collision-sector upper bound; no learned-policy or val_unseen claim",
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
        "g74_anchor": {
            "path": run_cfg["g74_result"],
            "sha256": run_cfg["g74_result_sha256"],
            "exact_bin_macro_interrupt": anchor["variants"]["macro_interrupt"]["aggregate"],
            "g71_dense_fixed4": anchor["g73_anchor"]["g71_dense_fixed4"],
        },
        "variants": variants,
        "selection_rule": "first all-gates-pass in sector15 then sector30 order",
        "selected_variant": selected_variant,
        "g8_entry_variant": entry_variant,
        "g8_entry_gates_by_variant": gates_by_variant,
        "g8_entry_gates": gates,
        "g8_ready": selected_variant is not None,
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
                "selected_variant": selected_variant,
                "gates": gates,
                "gates_by_variant": gates_by_variant,
                "aggregates": {
                    name: value["aggregate"] for name, value in variants.items()
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
