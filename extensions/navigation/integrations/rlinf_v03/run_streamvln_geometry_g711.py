"""Run the bounded G7.11 explicit geometry controller gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from run_streamvln_candidates_g7 import configure_process_git_safe_directory, validate_identity
from run_streamvln_macro_audit_g79 import load_frozen_g78
from run_streamvln_ppo_g4 import make_env, validate_dataset_hashes, validate_manifest
from run_streamvln_topology_g5 import ORIGINAL_STREAM_COMMIT, git_head, jsonable, sha256_file
from streamvln_candidates_g74 import generate_macro_candidates
from streamvln_depth_g78 import predict_depth_scores
from streamvln_geometry_g711 import controller_gate
from streamvln_macro_audit_g79 import (
    EXCLUDED_CANDIDATE,
    REGISTERED_CANDIDATES_PER_SCENE,
    distribution,
    select_registered_candidates,
)
from streamvln_severity_g710 import severity_outcome
from streamvln_topology import OnlineTopology


RLINF_COMMIT = "0505431899574619da86f551bad70b71e0ea2177"
STATE_DEFINITIONS = [
    {"name": "reset", "transition_actions": []},
    {"name": "after_fixed_p0_d0p5", "transition_actions": [1, 1]},
]
PHYSICAL_VARIANTS = [
    "open_loop",
    "endpoint_geometry",
    "depth_receding",
    "navmesh_receding_oracle",
]
ALL_VARIANTS = [
    "open_loop",
    "frozen_g78_reference",
    "endpoint_geometry",
    "depth_receding",
    "navmesh_receding_oracle",
]


def expected_preregistration(manifest):
    return {
        "split": "train",
        "train_dev_scenes": [row["scene"] for row in manifest["train"][6:]],
        "episode_indices": [row["index"] for row in manifest["train"][6:]],
        "state_definitions": STATE_DEFINITIONS,
        "candidate_count_per_state": 32,
        "excluded_candidate": EXCLUDED_CANDIDATE,
        "repeat_count": 2,
        "row_count": 128,
        "physical_variants": PHYSICAL_VARIANTS,
        "physical_repeat_count": 1024,
        "frozen_g78_reference": True,
        "forward_step_m": 0.25,
        "turn_angle_degrees": 15.0,
        "depth_projection": {
            "height": 64,
            "width": 64,
            "hfov_degrees": 90.0,
            "fx": 32.0,
            "fy": 32.0,
            "cx": 31.5,
            "cy": 31.5,
            "sensor_height_m": 1.25,
            "agent_height_m": 1.5,
            "agent_radius_m": 0.1,
            "safety_margin_m": 0.02,
            "corridor_half_width_m": 0.12,
            "obstacle_min_height_m": 0.05,
            "obstacle_max_height_m": 1.45,
            "primitive_brake_distance_m": 0.37,
            "depth_min_m": 0.0,
            "depth_max_m": 10.0,
            "normalize_depth": False,
        },
        "endpoint_geometry": {"no_sliding_endpoint_tolerance_m": 0.001},
        "navmesh_receding_oracle": {
            "no_sliding_endpoint_tolerance_m": 0.001,
            "sample_spacing_m": 0.125,
            "minimum_boundary_clearance_m": 0.02,
        },
        "gates": {
            "severe_exposure_reduction_min": 0.70,
            "free_grazing_completion_min": 0.80,
            "free_grazing_progress_retention_min": 0.90,
            "overall_forward_primitive_retention_min": 0.50,
            "maximum_restore_error_m": 0.000001,
        },
        "g710_anchor": {
            "result_sha256": "3bdac8e1878a5c19d36de1c0c3924c67f4ba7fc07aef3ab0463d74de37d609f6",
            "rows_sha256": "c6d24549bb3c113d8887b1565088089fc398431e9e44efcee413e17e7bf62c8a",
        },
    }


def _open_severity(repeat):
    return severity_outcome(
        {
            "commanded_distance_m": repeat["full_commanded_distance_m"],
            "realized_forward_m": repeat["realized_forward_m"],
            "endpoint_error_m": repeat["full_endpoint_error_m"],
            "realized_lateral_error_m": repeat["realized_lateral_error_m"],
            "primitive_trace": repeat["primitive_trace"],
            "full_collision_trace": repeat["raw_collision_trace"],
        }
    )


def _load_g710_anchor(result_path, result_hash, rows_path, rows_hash):
    if sha256_file(result_path) != result_hash or sha256_file(rows_path) != rows_hash:
        raise RuntimeError("G7.10 anchor hash changed")
    result = json.loads(result_path.read_text())
    if (
        result.get("kind")
        != "streamvln-g7.10-frozen-collision-severity-contact-geometry-audit"
        or result.get("all_protocol_guards_passed") is not True
        or result.get("val_unseen_episode_count") != 0
        or result.get("source_contract", {}).get("stream_commit")
        != ORIGINAL_STREAM_COMMIT
        or result.get("source_contract", {}).get("rlinf_commit") != RLINF_COMMIT
    ):
        raise RuntimeError("G7.10 accepted source contract changed")
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line]
    by_key = {
        (row["scene"], row["state_name"], row["candidate"]): row for row in rows
    }
    if len(rows) != 128 or len(by_key) != 128:
        raise RuntimeError("G7.10 anchor row count changed")
    return result, by_key


def collect_scene(run_cfg, spec, candidates, prereg):
    cfg = dict(run_cfg)
    cfg["habitat_server"] = run_cfg["habitat_geometry_controller_server"]
    env = make_env(
        cfg,
        split="train",
        episode_index=spec["index"],
        max_steps=run_cfg["max_episode_steps"],
    )
    rows = []
    states = []
    try:
        _, identity = env.reset()
        validate_identity(identity, spec)
        for state_index, state in enumerate(prereg["state_definitions"]):
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
                    raise RuntimeError("G7.11 state transition ended the episode")
            audit = env.probe_geometry_controller(candidates)
            if audit["variants"] != PHYSICAL_VARIANTS or audit["repeat_count"] != 2:
                raise RuntimeError("G7.11 physical variants changed")
            by_name = {row["name"]: row for row in audit["candidates"]}
            if set(by_name) != {candidate.name for candidate in candidates}:
                raise RuntimeError("G7.11 candidate bank changed")
            for candidate in candidates:
                probe = by_name[candidate.name]
                physical = {}
                for variant in PHYSICAL_VARIANTS:
                    repeats = probe["variants"][variant]
                    physical[variant] = {
                        "repeat_exact": repeats[0] == repeats[1],
                        "repeats": repeats,
                    }
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
                        "candidate": candidate.name,
                        "heading_degrees": float(candidate.heading_degrees),
                        "model_heading_degrees": 0.0,
                        "distance_m": float(candidate.distance_m),
                        "actions": list(candidate.actions),
                        "turn_prefix": list(probe["turn_prefix"]),
                        "forward_suffix": list(probe["forward_suffix"]),
                        "initial_depth_sha256": probe["initial_depth_sha256"],
                        "physical": physical,
                        "_initial_depth": probe["initial_metric_depth"],
                    }
                )
            states.append(
                {
                    "name": state["name"],
                    "transition_actions": list(state["transition_actions"]),
                    "transition_records": transition_records,
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
            "states": states,
        }, rows
    finally:
        env.close()


def _record_from_repeat(repeat):
    controlled = repeat["controlled_motion"]
    return {
        "completed": bool(repeat["completed"]),
        "executed_forward_primitives": int(repeat["executed_forward_primitives"]),
        "commanded_forward_primitives": int(repeat["commanded_forward_primitives"]),
        "realized_forward_m": float(repeat["realized_forward_m"]),
        "severe_exposure": bool(repeat["severe_motion_exposure"]),
        "collision_steps": int(sum(repeat["raw_collision_trace"])),
        "raw_collision": bool(any(repeat["raw_collision_trace"])),
        "continuous_severity": (
            None if controlled is None else float(controlled["continuous_severity"])
        ),
        "full_endpoint_error_m": float(repeat["full_endpoint_error_m"]),
        "lateral_error_m": float(repeat["realized_lateral_error_m"]),
        "minimum_actual_clearance_m": float(
            repeat["minimum_actual_clearance_m"]
        ),
    }


def _variant_metrics(rows, variant, open_strata=None):
    records = [row["records"][variant] for row in rows]
    baseline_severe = [
        (row, record)
        for row, record in zip(rows, records)
        if row["open_severity_class"] == "severe_blockage"
    ]
    safe = [
        (row, record)
        for row, record in zip(rows, records)
        if row["open_severity_class"] in ("free", "grazing")
    ]
    open_safe_progress = sum(
        max(0.0, row["records"]["open_loop"]["realized_forward_m"])
        for row, _ in safe
    )
    severe_exposure = sum(record["severe_exposure"] for _, record in baseline_severe)
    severe_count = len(baseline_severe)
    completion = sum(record["completed"] for _, record in safe) / len(safe)
    progress = (
        sum(max(0.0, record["realized_forward_m"]) for _, record in safe)
        / open_safe_progress
    )
    open_primitives = sum(
        row["records"]["open_loop"]["executed_forward_primitives"] for row in rows
    )
    primitive_retention = sum(
        record["executed_forward_primitives"] for record in records
    ) / open_primitives
    by_stratum = {}
    for scene in sorted({row["scene"] for row in rows}):
        for state in sorted({row["state_name"] for row in rows}):
            items = [
                (row, row["records"][variant])
                for row in rows
                if row["scene"] == scene and row["state_name"] == state
            ]
            key = scene + "/" + state
            by_stratum[key] = {
                "count": len(items),
                "severe_exposure_count": sum(
                    record["severe_exposure"] for _, record in items
                ),
                "completed_count": sum(record["completed"] for _, record in items),
                "executed_forward_primitives": sum(
                    record["executed_forward_primitives"] for _, record in items
                ),
            }
    stratum_nonincrease = True
    if open_strata is not None:
        stratum_nonincrease = all(
            value["severe_exposure_count"]
            <= open_strata[key]["severe_exposure_count"]
            for key, value in by_stratum.items()
        )
    severities = [
        record["continuous_severity"]
        for record in records
        if record["continuous_severity"] is not None
    ]
    return {
        "row_count": len(rows),
        "baseline_severe_count": severe_count,
        "severe_exposure_count_on_baseline_severe": severe_exposure,
        "severe_exposure_reduction": (
            1.0 - severe_exposure / severe_count if severe_count else 0.0
        ),
        "free_grazing_count": len(safe),
        "free_grazing_completion_rate": completion,
        "free_grazing_progress_retention": progress,
        "overall_forward_primitive_retention": primitive_retention,
        "stratum_severe_nonincrease": stratum_nonincrease,
        "completed_count": sum(record["completed"] for record in records),
        "zero_motion_count": sum(
            record["executed_forward_primitives"] == 0 for record in records
        ),
        "total_severe_exposure_count": sum(
            record["severe_exposure"] for record in records
        ),
        "raw_collision_row_count": sum(record["raw_collision"] for record in records),
        "collision_step_count": sum(record["collision_steps"] for record in records),
        "executed_motion_continuous_severity": (
            distribution(severities) if severities else None
        ),
        "full_endpoint_error_m": distribution(
            record["full_endpoint_error_m"] for record in records
        ),
        "lateral_error_m": distribution(record["lateral_error_m"] for record in records),
        "minimum_actual_clearance_m": distribution(
            record["minimum_actual_clearance_m"] for record in records
        ),
        "by_scene_state": by_stratum,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    for name in (
        "manifest",
        "g78-result",
        "g78-result-sha256",
        "g78-depth-checkpoint",
        "g78-depth-checkpoint-sha256",
        "g710-result",
        "g710-result-sha256",
        "g710-rows",
        "g710-rows-sha256",
        "output-dir",
        "rlinf-root",
        "stream-source",
        "habitat-python",
        "habitat-geometry-controller-server",
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
        raise ValueError("G7.11 episode budget is invalid")
    path_names = (
        "manifest",
        "g78_result",
        "g78_depth_checkpoint",
        "g710_result",
        "g710_rows",
        "rlinf_root",
        "stream_source",
        "habitat_python",
        "habitat_geometry_controller_server",
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
        raise FileExistsError("G7.11 output directory must be fresh")
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(paths["manifest"]).read_text())
    validate_manifest(manifest)
    prereg = manifest.get("g711_geometry_controller", {})
    expected_prereg = expected_preregistration(manifest)
    if prereg != expected_prereg:
        raise RuntimeError("G7.11 manifest preregistration changed")
    configure_process_git_safe_directory(paths["stream_source"])
    configure_process_git_safe_directory(paths["rlinf_root"])
    stream_commit = git_head(Path(paths["stream_source"]))
    rlinf_commit = git_head(Path(paths["rlinf_root"]))
    if stream_commit != ORIGINAL_STREAM_COMMIT or rlinf_commit != RLINF_COMMIT:
        raise RuntimeError("G7.11 source commit pin changed")
    _, anchor = _load_g710_anchor(
        Path(paths["g710_result"]),
        args.g710_result_sha256,
        Path(paths["g710_rows"]),
        args.g710_rows_sha256,
    )
    topology = OnlineTopology(max_decisions=args.max_episode_steps)
    candidates = select_registered_candidates(generate_macro_candidates(topology))
    if len(candidates) != REGISTERED_CANDIDATES_PER_SCENE:
        raise RuntimeError("G7.11 fixed candidate count changed")
    run_cfg = {
        **paths,
        "manifest": manifest,
        "output_dir": str(output),
        "g78_result_sha256": args.g78_result_sha256,
        "g78_depth_checkpoint_sha256": args.g78_depth_checkpoint_sha256,
        "g710_result_sha256": args.g710_result_sha256,
        "g710_rows_sha256": args.g710_rows_sha256,
        "stream_commit": stream_commit,
        "rlinf_commit": rlinf_commit,
        "seed": args.seed,
        "max_episode_steps": args.max_episode_steps,
        "preregistration": prereg,
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
    if len(rows) != 128 or len(episodes) != 2:
        raise RuntimeError("G7.11 fixed sample count changed")
    model_rows = [
        {
            "metric_depth": row["_initial_depth"],
            "heading_degrees": row["model_heading_degrees"],
            "distance_m": row["distance_m"],
            "unsafe": False,
        }
        for row in rows
    ]
    scores = predict_depth_scores(model, model_rows, device=device)
    anchor_exact = True
    anchor_differences = []
    for row, score in zip(rows, scores):
        key = (row["scene"], row["state_name"], row["candidate"])
        anchored = anchor.get(key)
        if anchored is None:
            raise RuntimeError("G7.11 row is absent from G7.10 anchor")
        open_repeat = row["physical"]["open_loop"]["repeats"][0]
        open_severity = _open_severity(open_repeat)
        differences = {
            "depth_hash": row["initial_depth_sha256"]
            != anchored["depth_sha256_first"],
            "severity_class": open_severity["severity_class"]
            != anchored["severity_class"],
            "continuous_severity": abs(
                open_severity["continuous_severity"]
                - float(anchored["continuous_severity"])
            )
            > 1e-8,
            "score": abs(float(score) - float(anchored["score"])) > 1e-8,
        }
        if any(differences.values()):
            anchor_exact = False
            anchor_differences.append({"key": list(key), "differences": differences})
        row["score"] = float(score)
        row["open_severity_class"] = open_severity["severity_class"]
        row["open_continuous_severity"] = open_severity["continuous_severity"]
        row["records"] = {}
        for variant in PHYSICAL_VARIANTS:
            row["records"][variant] = _record_from_repeat(
                row["physical"][variant]["repeats"][0]
            )
        accepted = float(score) < threshold
        open_record = row["records"]["open_loop"]
        row["records"]["frozen_g78_reference"] = (
            dict(open_record)
            if accepted
            else {
                "completed": False,
                "executed_forward_primitives": 0,
                "commanded_forward_primitives": open_record[
                    "commanded_forward_primitives"
                ],
                "realized_forward_m": 0.0,
                "severe_exposure": False,
                "collision_steps": 0,
                "raw_collision": False,
                "continuous_severity": None,
                "full_endpoint_error_m": float(row["distance_m"]),
                "lateral_error_m": 0.0,
                "minimum_actual_clearance_m": open_repeat[
                    "actual_clearance_trace"
                ][0]["clearance_m"],
            }
        )
        row["frozen_g78_allow"] = accepted
        row.pop("_initial_depth")

    physical_repeat_exact = all(
        row["physical"][variant]["repeat_exact"]
        for row in rows
        for variant in PHYSICAL_VARIANTS
    )
    state_restore_exact = all(
        state["max_position_restore_error_m"] <= 1e-6
        and state["max_rotation_restore_error"] <= 1e-6
        for episode in episodes
        for state in episode["states"]
    )
    integrity_ready = anchor_exact and physical_repeat_exact and state_restore_exact
    metrics = {}
    metrics["open_loop"] = _variant_metrics(rows, "open_loop")
    open_strata = metrics["open_loop"]["by_scene_state"]
    for variant in ALL_VARIANTS[1:]:
        metrics[variant] = _variant_metrics(rows, variant, open_strata=open_strata)
    gates = {
        "depth_receding": controller_gate(
            metrics["depth_receding"], integrity_ready=integrity_ready
        ),
        "navmesh_receding_oracle": controller_gate(
            metrics["navmesh_receding_oracle"], integrity_ready=integrity_ready
        ),
    }
    if not integrity_ready:
        route = "repair-g711-instrumentation-or-g710-anchor-reproduction"
    elif gates["depth_receding"]["ready"] and gates[
        "navmesh_receding_oracle"
    ]["ready"]:
        route = "freeze-explicit-depth-controller-and-design-separate-causal-navigation-gate"
    elif gates["navmesh_receding_oracle"]["ready"]:
        route = "repair-depth-footprint-geometry-or-test-explicit-occupancy-lidar-style-lower-layer"
    else:
        route = "repair-macro-controller-or-navmesh-oracle-contract-before-navigation"

    protocol_guards = {
        "fixed_two_train_dev_scenes": {row["scene"] for row in rows}
        == set(prereg["train_dev_scenes"]),
        "fixed_128_rows": len(rows) == 128,
        "fixed_32_candidates_per_scene_state": all(
            sum(
                row["scene"] == scene and row["state_name"] == state["name"]
                for row in rows
            )
            == 32
            for scene in prereg["train_dev_scenes"]
            for state in prereg["state_definitions"]
        ),
        "fixed_physical_variants": all(
            list(row["physical"]) == PHYSICAL_VARIANTS for row in rows
        ),
        "two_repeats": all(
            len(row["physical"][variant]["repeats"]) == 2
            for row in rows
            for variant in PHYSICAL_VARIANTS
        ),
        "repeat_exact": physical_repeat_exact,
        "state_restore_exact": state_restore_exact,
        "g710_anchor_exact": anchor_exact,
        "frozen_g78_reference": True,
        "causal_depth_excludes_privileged_inputs": True,
        "no_model_fit": True,
        "no_navigation_evaluation": True,
        "zero_val_unseen": True,
        "policy_observation_excludes_depth_and_privileged_labels": True,
    }
    evidence_path = output / "controller_rows.jsonl"
    with evidence_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(jsonable(row), sort_keys=True) + "\n")
    result = {
        "schema_version": 1,
        "kind": "streamvln-g7.11-explicit-depth-footprint-controller-gate",
        "claim_scope": "train-only unit-level controller diagnostic on fixed restored states and exact macros; no fitting, navigation, language, RGB, val_unseen or generalization claim",
        "split": "train",
        "val_unseen_episode_count": 0,
        "source_contract": {
            "stream_commit": stream_commit,
            "rlinf_commit": rlinf_commit,
            "source_kind": "upstream-original",
            "secondary_post_training_used": False,
            "ttt_used": False,
            "frozen_g78_reference": True,
            "frozen_g78_threshold": threshold,
        },
        "dataset_hashes": dataset_hashes,
        "preregistration": prereg,
        "episodes": episodes,
        "sample_summary": {
            "row_count": len(rows),
            "scene_count": len({row["scene"] for row in rows}),
            "state_count": len({row["state_name"] for row in rows}),
            "open_severity_class_counts": {
                name: sum(row["open_severity_class"] == name for row in rows)
                for name in ("free", "grazing", "sliding", "severe_blockage")
            },
        },
        "anchor_audit": {
            "exact": anchor_exact,
            "difference_count": len(anchor_differences),
            "differences": anchor_differences,
        },
        "repeat_and_restore_audit": {
            "physical_repeat_exact": physical_repeat_exact,
            "state_restore_exact": state_restore_exact,
        },
        "variant_metrics": metrics,
        "controller_gates": gates,
        "routing": route,
        "protocol_guards": protocol_guards,
        "all_protocol_guards_passed": all(protocol_guards.values()),
        "g8_ready": False,
        "evidence": {
            "rows_path": str(evidence_path),
            "rows_sha256": sha256_file(evidence_path),
            "g710_result": paths["g710_result"],
            "g710_result_sha256": args.g710_result_sha256,
            "g710_rows": paths["g710_rows"],
            "g710_rows_sha256": args.g710_rows_sha256,
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
                "anchor_audit": result["anchor_audit"],
                "controller_gates": gates,
                "variant_metrics": {
                    name: {
                        key: value
                        for key, value in metrics[name].items()
                        if key not in ("by_scene_state",)
                    }
                    for name in ALL_VARIANTS
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
