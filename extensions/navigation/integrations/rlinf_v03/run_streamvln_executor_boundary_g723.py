"""Run one preregistered G7.23 executor-boundary sensitivity arm."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import time

import numpy as np

from run_streamvln_atomic_guard_g720 import candidate_gates, evaluate_state, one_update, prepare_batch
from run_streamvln_learning_g718 import (
    ORIGINAL_STREAM_COMMIT,
    RLINF_COMMIT,
    aggregate_behavior,
    collect_phase,
    flat_rows,
    paired_behavior,
    write_jsonl,
)
from run_streamvln_plumbing_g717 import (
    dataset_hashes,
    frozen_parameter_sentinels,
    git_head,
    optimizer_parameter_scope,
    sha256_file,
    small_file_inventory,
    write_json,
)
from run_streamvln_recovery_g6 import setup_manager
from run_streamvln_safety_transfer_g721 import (
    collection_semantics,
    compare_probes,
    fixed_continuation_probe,
    prepare_evaluation_batch,
)
from run_streamvln_stability_g719_v2 import optimizer_state_contract
from run_streamvln_topology_g5 import cleanup_manager, parameter_snapshot, state_sha256
from streamvln_atomic_state import capture_training_state, restore_training_state, training_state_hashes
from streamvln_executor_contract import PRIVILEGED_POLICY_KEYS, assert_history_only_keys


MANIFEST_SHA256 = "59af15397be0d6642474459ad74e21da4fb7896a28c76ea4e9f6da7326cc8297"
G722_LOCAL_COMMIT = "56b8521e2c9755e9d819bd5bc39a79e169c08528"
G722_MANIFEST_SHA256 = "3fe63b7afc3f40ed8086c6da6c38496df4a3780360387f324491f8f04751efdd"
G722_RECOMPUTE_SHA256 = "52cb5b26e8749c2289b3c89d4a2ab4edf03f1759cbf9ec74bc5603f7e656b61c"
G722_RESULT_SHA256 = {
    2026092811: "4bc921f86658bc5b446223c131d2c57f85a611c62424c9968976b4045d70a3bf",
    2026092812: "44e52aa34a49c5dde4696ea7a0434002bd6d2585471336e35bf08c30e131cb30",
    2026092813: "40c625d8714119a2ca9af954a5345dd899c9f9b667be99c8af38860092d149d4",
}
FROZEN_BOUNDARY_ARMS = {
    "2026092811": [
        {
            "episode_id": "4538",
            "index": 4528,
            "scene": "17DRP5sb8fy"
        },
        {
            "episode_id": "577",
            "index": 576,
            "scene": "17DRP5sb8fy"
        },
        {
            "episode_id": "6004",
            "index": 5994,
            "scene": "D7G3Y4RVNrH"
        },
        {
            "episode_id": "10045",
            "index": 10026,
            "scene": "D7G3Y4RVNrH"
        },
        {
            "episode_id": "6462",
            "index": 6452,
            "scene": "YmJkqBEsHnH"
        },
        {
            "episode_id": "2023",
            "index": 2019,
            "scene": "gZ6f7yhEvPG"
        },
        {
            "episode_id": "5944",
            "index": 5934,
            "scene": "HxpKQynjfin"
        },
        {
            "episode_id": "7044",
            "index": 7034,
            "scene": "HxpKQynjfin"
        }
    ],
    "2026092812": [
        {
            "episode_id": "6092",
            "index": 6082,
            "scene": "17DRP5sb8fy"
        },
        {
            "episode_id": "8504",
            "index": 8491,
            "scene": "17DRP5sb8fy"
        },
        {
            "episode_id": "8432",
            "index": 8419,
            "scene": "D7G3Y4RVNrH"
        },
        {
            "episode_id": "2955",
            "index": 2948,
            "scene": "HxpKQynjfin"
        },
        {
            "episode_id": "8194",
            "index": 8181,
            "scene": "D7G3Y4RVNrH"
        },
        {
            "episode_id": "1416",
            "index": 1412,
            "scene": "YmJkqBEsHnH"
        },
        {
            "episode_id": "2022",
            "index": 2018,
            "scene": "gZ6f7yhEvPG"
        },
        {
            "episode_id": "6461",
            "index": 6451,
            "scene": "YmJkqBEsHnH"
        }
    ],
    "2026092813": [
        {
            "episode_id": "3717",
            "index": 3707,
            "scene": "17DRP5sb8fy"
        },
        {
            "episode_id": "9936",
            "index": 9917,
            "scene": "D7G3Y4RVNrH"
        },
        {
            "episode_id": "576",
            "index": 575,
            "scene": "17DRP5sb8fy"
        },
        {
            "episode_id": "8555",
            "index": 8542,
            "scene": "HxpKQynjfin"
        },
        {
            "episode_id": "10044",
            "index": 10025,
            "scene": "D7G3Y4RVNrH"
        },
        {
            "episode_id": "1787",
            "index": 1783,
            "scene": "YmJkqBEsHnH"
        },
        {
            "episode_id": "8557",
            "index": 8544,
            "scene": "HxpKQynjfin"
        },
        {
            "episode_id": "1415",
            "index": 1411,
            "scene": "YmJkqBEsHnH"
        }
    ]
}


def validate_manifest(manifest: dict) -> dict:
    expected = {
        "schema_version", "stage", "source_contract", "source", "predecessor",
        "selection", "policy_observation_keys", "privileged_policy_blacklist",
        "protocol", "decision_rule", "required_reports", "model_load", "rl_update",
        "navigation_evaluation", "controller_comparison", "robot_execution", "g8_ready",
        "new_policy_training_or_selection",
    }
    if set(manifest) != expected or manifest["schema_version"] != 1:
        raise ValueError("G7.23 manifest schema changed")
    if manifest["source_contract"] != {
        "stream_commit": ORIGINAL_STREAM_COMMIT,
        "rlinf_commit": RLINF_COMMIT,
        "original_stream_only": True,
        "ttt_or_secondary_post_training": False,
    }:
        raise ValueError("G7.23 source contract changed")
    predecessor = manifest["predecessor"]
    if (
        predecessor["accepted_local_commit"] != G722_LOCAL_COMMIT
        or predecessor["manifest_sha256"] != G722_MANIFEST_SHA256
        or {int(k): v for k, v in predecessor["result_sha256"].items()} != G722_RESULT_SHA256
        or predecessor["evidence_recompute_sha256"] != G722_RECOMPUTE_SHA256
    ):
        raise ValueError("G7.23 predecessor anchor changed")
    assert_history_only_keys(manifest["policy_observation_keys"])
    if set(manifest["privileged_policy_blacklist"]) != PRIVILEGED_POLICY_KEYS:
        raise ValueError("G7.23 privileged-data blacklist changed")
    protocol = manifest["protocol"]
    selection = manifest["selection"]
    all_boundary = [row for rows in FROZEN_BOUNDARY_ARMS.values() for row in rows]
    per_seed_scene_counts = {
        seed: {scene: sum(row["scene"] == scene for row in rows) for scene in {row["scene"] for row in rows}}
        for seed, rows in FROZEN_BOUNDARY_ARMS.items()
    }
    if (
        protocol["boundary_arms"] != FROZEN_BOUNDARY_ARMS
        or selection["seeds"] != [int(seed) for seed in FROZEN_BOUNDARY_ARMS]
        or selection["episodes_per_seed"] != 8
        or selection["eligible_new_scene_count"] != 5
        or selection["outcome_or_policy_fields_used"] is not False
        or selection["global_episode_identity_disjoint"] is not True
        or selection["all_prior_scenes_excluded"] is not True
        or selection["replacement_after_collection"] is not False
        or len(all_boundary) != 24
        or len({row["episode_id"] for row in all_boundary}) != 24
        or {row["scene"] for row in all_boundary} != set(selection["eligible_new_scenes"])
        or {row["scene"] for row in all_boundary} & set(selection["excluded_scenes"])
        or any(max(counts.values()) > 2 for counts in per_seed_scene_counts.values())
        or protocol["split"] != "train"
        or protocol["sampling"]["sample_calls_per_decision"] != 1
        or protocol["sampling"]["target_mode_scan"] is not False
        or protocol["sampling"]["mode_filter_or_resample"] is not False
        or protocol["sampling"]["recovery_override"] is not False
        or protocol["max_decisions_per_episode"] != 3
        or protocol["max_primitives_per_episode"] != 12
        or protocol["candidate_update_limit"] != 3
        or protocol["stop_candidate_loop_after_first_rejection"] is not True
        or protocol["state_zero_and_final_must_exactly_match_g721"] is not True
        or protocol["checkpoint_selection"] is not False
        or protocol["val_unseen_episode_count"] != 0
        or protocol["no_new_policy_state_or_behavior_selection"] is not True
    ):
        raise ValueError("G7.23 protocol changed")
    boundary = protocol["boundary_contract"]
    if (
        boundary["episodes_per_state"] != 8
        or boundary["same_initial_observation"] is not True
        or boundary["same_sampling_seed_schedule"] is not True
        or boundary["same_executor_and_budget"] is not True
        or boundary["causal_attribution_requires_equal_current_state_and_topology_hash"] is not True
        or boundary["causal_attribution_requires_all_prior_executed_sequences_equal"] is not True
        or boundary["after_first_sequence_divergence_is_descriptive_only"] is not True
        or boundary["may_affect_update_commit_stop_checkpoint_or_rerun"] is not False
    ):
        raise ValueError("G7.23 causal boundary contract changed")
    compiler = protocol["compiler_contract"]
    if compiler != {
        "feedback_equality_tolerance": 1e-9,
        "navigate_small_waypoint_shift_l2_max_m": 0.05,
        "rounding": "python_round_abs_bearing_over_turn_radians",
        "turn_degrees": 15.0,
        "turn_step_boundaries_degrees": [7.5, 22.5, 37.5, 52.5],
    }:
        raise ValueError("G7.23 compiler contract changed")
    rule = manifest["decision_rule"]
    if (
        rule["minimum_flip_count"] != 2
        or rule["minimum_flip_scene_count"] != 2
        or rule["minimum_harmful_flip_count"] != 2
        or rule["minimum_harmful_flip_scene_count"] != 2
        or rule["same_sequence_causal_control_feedback_must_match"] is not True
        or rule["posthoc_selection_or_rerun"] is not False
    ):
        raise ValueError("G7.23 decision rule changed")
    if any(manifest[key] is not False for key in (
        "navigation_evaluation", "controller_comparison", "robot_execution", "g8_ready",
        "new_policy_training_or_selection",
    )) or manifest["model_load"] is not True or manifest["rl_update"] is not True:
        raise ValueError("G7.23 claim boundary changed")
    return protocol


def exact_restore(evidence: dict) -> bool:
    return bool(
        evidence
        and evidence["exact"] is True
        and evidence["before"] == evidence["after"]
        and all(evidence["component_equal"].values())
    )


def behavior_delta(initial: dict, final: dict) -> dict:
    keys = (
        "decision_count", "primitive_count", "environment_reward_sum",
        "discounted_chunk_reward_sum", "realized_path_m", "collision_count",
        "blocked_forward_count", "stop_decision_count", "recover_decision_count",
        "navigate_decision_count", "terminated_episode_count", "timeout_episode_count",
    )
    delta = {key: final[key] - initial[key] for key in keys}
    delta["realized_path_retention"] = (
        final["realized_path_m"] / initial["realized_path_m"]
        if initial["realized_path_m"] > 0 else 1.0
    )
    return delta


def annotate_policy_input_hashes(episodes) -> None:
    for episode in episodes:
        for row in episode["rows"]:
            row["state_features_sha256"] = state_sha256({"state": row["state"]})
            row["topology_features_sha256"] = state_sha256({"topology": row["topology_features"]})
            row["policy_input_sha256"] = state_sha256({
                "state": row["state"], "topology": row["topology_features"],
            })


def approximately_equal(left, right, tolerance: float) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isfinite(float(left)) and math.isfinite(float(right)) and abs(float(left) - float(right)) <= tolerance
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            approximately_equal(left[key], right[key], tolerance) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            approximately_equal(a, b, tolerance) for a, b in zip(left, right)
        )
    return left == right


def compiler_metadata(waypoint, compiler: dict) -> dict:
    forward, lateral = (float(value) for value in waypoint)
    bearing = math.degrees(math.atan2(lateral, forward))
    absolute = abs(bearing)
    turn_degrees = float(compiler["turn_degrees"])
    turn_steps_uncapped = int(round(absolute / turn_degrees))
    turn_steps = min(4, turn_steps_uncapped)
    nearest = min(compiler["turn_step_boundaries_degrees"], key=lambda value: abs(absolute - value))
    turn_action = 2 if bearing > 0 else 3
    compiled = [turn_action] * turn_steps + [1] * (4 - turn_steps)
    return {
        "bearing_degrees": bearing,
        "absolute_bearing_degrees": absolute,
        "nearest_half_step_boundary_degrees": nearest,
        "signed_margin_to_nearest_boundary_degrees": absolute - nearest,
        "absolute_margin_to_nearest_boundary_degrees": abs(absolute - nearest),
        "turn_steps_uncapped": turn_steps_uncapped,
        "turn_steps": turn_steps,
        "compiled_actions": compiled,
    }


def feedback_delta(before: dict, after: dict) -> dict:
    keys = (
        "environment_reward_sum", "collision_count", "blocked_forward_count",
        "realized_path_m", "commanded_forward_m",
    )
    return {key: after[key] - before[key] for key in keys}


def analyze_executor_boundary(initial, final, compiler: dict) -> dict:
    if len(initial) != len(final):
        raise RuntimeError("G7.23 boundary episode count changed")
    rows = []
    episodes = []
    controls_match = True
    compiler_holds = True
    for before_episode, after_episode in zip(initial, final):
        if before_episode["episode_id"] != after_episode["episode_id"] or before_episode["scene"] != after_episode["scene"]:
            raise RuntimeError("G7.23 boundary identity changed")
        identity_match = before_episode["initial_observation_sha256"] == after_episode["initial_observation_sha256"]
        schedule_match = before_episode["sampling_seed_schedule"] == after_episode["sampling_seed_schedule"]
        prior_executed_sequences_equal = True
        episode_rows = []
        max_rows = max(len(before_episode["rows"]), len(after_episode["rows"]))
        for decision_index in range(max_rows):
            if decision_index >= len(before_episode["rows"]) or decision_index >= len(after_episode["rows"]):
                rows.append({
                    "episode_id": before_episode["episode_id"], "scene": before_episode["scene"],
                    "decision_index": decision_index, "paired_row_present": False,
                    "causal_prefix": False, "descriptive_only": True,
                })
                episode_rows.append(rows[-1])
                prior_executed_sequences_equal = False
                continue
            before = before_episode["rows"][decision_index]
            after = after_episode["rows"][decision_index]
            input_hashes_equal = (
                before["state_features_sha256"] == after["state_features_sha256"]
                and before["topology_features_sha256"] == after["topology_features_sha256"]
                and before["policy_input_sha256"] == after["policy_input_sha256"]
            )
            causal = bool(identity_match and schedule_match and prior_executed_sequences_equal and input_hashes_equal)
            same_mode = before["policy_mode"] == after["policy_mode"]
            navigate = same_mode and before["policy_mode"] == 0
            waypoint_l2 = math.dist(before["waypoint"], after["waypoint"])
            small_shift = navigate and waypoint_l2 <= compiler["navigate_small_waypoint_shift_l2_max_m"]
            before_proposed = list(before["executor_feedback"]["proposed_actions"])
            after_proposed = list(after["executor_feedback"]["proposed_actions"])
            before_executed = list(before["executor_feedback"]["executed_actions"])
            after_executed = list(after["executor_feedback"]["executed_actions"])
            proposed_flip = before_proposed != after_proposed
            executed_flip = before_executed != after_executed
            before_compiler = compiler_metadata(before["waypoint"], compiler) if before["policy_mode"] == 0 else None
            after_compiler = compiler_metadata(after["waypoint"], compiler) if after["policy_mode"] == 0 else None
            row_compiler_holds = (
                (before_compiler is None or before_compiler["compiled_actions"] == before_proposed)
                and (after_compiler is None or after_compiler["compiled_actions"] == after_proposed)
            )
            compiler_holds = compiler_holds and row_compiler_holds
            delta = feedback_delta(before["executor_feedback"], after["executor_feedback"])
            boundary_flip = bool(causal and navigate and small_shift and proposed_flip)
            harmful = bool(boundary_flip and (
                delta["environment_reward_sum"] < -1e-6
                or delta["collision_count"] > 0
                or delta["blocked_forward_count"] > 0
            ))
            same_sequence_control = bool(causal and before_executed == after_executed)
            feedback_match = (
                approximately_equal(before["executor_feedback"], after["executor_feedback"], compiler["feedback_equality_tolerance"])
                if same_sequence_control else None
            )
            if same_sequence_control:
                controls_match = controls_match and feedback_match
            record = {
                "episode_id": before_episode["episode_id"], "scene": before_episode["scene"],
                "decision_index": decision_index, "paired_row_present": True,
                "initial_observation_match": identity_match, "seed_schedule_match": schedule_match,
                "all_prior_executed_sequences_equal": prior_executed_sequences_equal,
                "state_features_match": before["state_features_sha256"] == after["state_features_sha256"],
                "topology_features_match": before["topology_features_sha256"] == after["topology_features_sha256"],
                "policy_input_match": input_hashes_equal, "causal_prefix": causal,
                "descriptive_only": not causal, "state_zero_mode": before["policy_mode"],
                "final_mode": after["policy_mode"], "same_mode": same_mode,
                "same_mode_navigate": navigate, "state_zero_waypoint": before["waypoint"],
                "final_waypoint": after["waypoint"], "waypoint_l2_m": waypoint_l2,
                "small_waypoint_shift": small_shift, "state_zero_compiler": before_compiler,
                "final_compiler": after_compiler, "compiler_contract_holds": row_compiler_holds,
                "state_zero_proposed_actions": before_proposed, "final_proposed_actions": after_proposed,
                "state_zero_executed_actions": before_executed, "final_executed_actions": after_executed,
                "proposed_sequence_flip": proposed_flip, "executed_sequence_flip": executed_flip,
                "causal_boundary_flip": boundary_flip, "feedback_delta": delta,
                "harmful_boundary_flip": harmful, "same_sequence_causal_control": same_sequence_control,
                "same_sequence_feedback_match": feedback_match,
            }
            rows.append(record)
            episode_rows.append(record)
            if executed_flip:
                prior_executed_sequences_equal = False
        episodes.append({
            "episode_id": before_episode["episode_id"], "scene": before_episode["scene"],
            "initial_observation_match": identity_match, "seed_schedule_match": schedule_match,
            "causal_row_count": sum(row.get("causal_prefix", False) for row in episode_rows),
            "boundary_flip_count": sum(row.get("causal_boundary_flip", False) for row in episode_rows),
            "harmful_boundary_flip_count": sum(row.get("harmful_boundary_flip", False) for row in episode_rows),
        })
    flips = [row for row in rows if row.get("causal_boundary_flip")]
    harmful = [row for row in flips if row["harmful_boundary_flip"]]
    causal_controls = [row for row in rows if row.get("same_sequence_causal_control")]
    return {
        "episodes": episodes,
        "decision_rows": rows,
        "summary": {
            "paired_episode_count": len(episodes), "decision_row_count": len(rows),
            "causal_prefix_row_count": sum(row.get("causal_prefix", False) for row in rows),
            "descriptive_only_row_count": sum(row.get("descriptive_only", False) for row in rows),
            "same_mode_navigate_causal_row_count": sum(row.get("causal_prefix", False) and row.get("same_mode_navigate", False) for row in rows),
            "small_shift_causal_navigate_row_count": sum(row.get("causal_prefix", False) and row.get("small_waypoint_shift", False) for row in rows),
            "causal_boundary_flip_count": len(flips),
            "causal_boundary_flip_scenes": sorted({row["scene"] for row in flips}),
            "harmful_boundary_flip_count": len(harmful),
            "harmful_boundary_flip_scenes": sorted({row["scene"] for row in harmful}),
            "pooled_harmful_flip_reward_delta": sum(row["feedback_delta"]["environment_reward_sum"] for row in harmful),
            "pooled_harmful_flip_collision_delta": sum(row["feedback_delta"]["collision_count"] for row in harmful),
            "pooled_harmful_flip_blocked_forward_delta": sum(row["feedback_delta"]["blocked_forward_count"] for row in harmful),
            "mode_flip_count_all_rows": sum(row.get("paired_row_present", False) and not row.get("same_mode", True) for row in rows),
            "executed_sequence_flip_count_all_rows": sum(row.get("executed_sequence_flip", False) for row in rows),
            "same_sequence_causal_control_count": len(causal_controls),
            "same_sequence_causal_control_feedback_match": controls_match,
            "compiler_contract_holds": compiler_holds,
        },
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--g722-results", type=Path, nargs=3, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rlinf-root", type=Path, required=True)
    parser.add_argument("--stream-source", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--vision-tower", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--habitat-python", type=Path, required=True)
    parser.add_argument("--habitat-server", type=Path, required=True)
    parser.add_argument("--habitat-sim-python", type=Path, required=True)
    parser.add_argument("--habitat-env-lib", type=Path, required=True)
    parser.add_argument("--egl-vendor-json", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--master-port", type=int, required=True)
    parser.add_argument("--context-tokens", type=int, default=4096)
    parser.add_argument("--head-hidden", type=int, default=256)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    started = time.time()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = args.manifest.resolve(strict=True)
    if sha256_file(manifest_path) != MANIFEST_SHA256:
        raise RuntimeError("G7.23 frozen manifest hash changed")
    manifest = json.loads(manifest_path.read_text())
    protocol = validate_manifest(manifest)
    seed_key = str(args.seed)
    if seed_key not in FROZEN_BOUNDARY_ARMS:
        raise RuntimeError("G7.23 seed is not preregistered")
    g721_specs = protocol["g721_arms"][seed_key]
    boundary_specs = protocol["boundary_arms"][seed_key]
    anchor = protocol["g721_reconstruction_anchors"][seed_key]

    predecessor_results = {}
    for path in [item.resolve(strict=True) for item in args.g722_results]:
        result = json.loads(path.read_text())
        seed = int(result["seed"])
        if seed not in G722_RESULT_SHA256 or sha256_file(path) != G722_RESULT_SHA256[seed]:
            raise RuntimeError("G7.23 G7.22 result anchor changed")
        if result["all_protocol_guards_passed"] is not True:
            raise RuntimeError("G7.23 predecessor protocol is invalid")
        predecessor_results[seed] = str(path)
    if set(predecessor_results) != set(G722_RESULT_SHA256):
        raise RuntimeError("G7.23 predecessor result set changed")

    path_names = (
        "rlinf_root", "stream_source", "base_model", "tokenizer", "vision_tower",
        "asset_root", "habitat_python", "habitat_server", "habitat_sim_python",
        "habitat_env_lib", "egl_vendor_json",
    )
    paths = {
        name: str(Path(getattr(args, name)).expanduser().resolve(strict=True))
        for name in path_names
    }
    source_heads = {
        "stream": git_head(Path(paths["stream_source"])),
        "rlinf": git_head(Path(paths["rlinf_root"])),
    }
    if source_heads != {"stream": ORIGINAL_STREAM_COMMIT, "rlinf": RLINF_COMMIT}:
        raise RuntimeError("G7.23 source commit changed")
    datasets = dataset_hashes(Path(paths["asset_root"]), manifest)
    script_root = Path(__file__).resolve().parent
    implementation_names = (
        "run_streamvln_executor_boundary_g723.py", "run_streamvln_safety_transfer_g721.py",
        "run_streamvln_atomic_guard_g720.py", "streamvln_atomic_state.py",
        "run_streamvln_learning_g718.py", "run_streamvln_plumbing_g717.py",
        "streamvln_topological_policy.py", "streamvln_recovery_policy.py",
        "streamvln_executor_contract.py", "streamvln_topology.py", "streamvln_chunk_ppo.py",
    )
    implementation_paths = {name: script_root / name for name in implementation_names}
    implementation_paths.update({
        "rlinf/envs/streamvln/streamvln_env.py": Path(paths["rlinf_root"]) / "rlinf/envs/streamvln/streamvln_env.py",
        "rlinf/envs/streamvln/streamvln_executor_contract.py": Path(paths["rlinf_root"]) / "rlinf/envs/streamvln/streamvln_executor_contract.py",
        "rlinf/models/embodiment/streamvln_ppo_policy.py": Path(paths["rlinf_root"]) / "rlinf/models/embodiment/streamvln_ppo_policy.py",
    })
    implementation = {
        name: sha256_file(path.resolve(strict=True)) for name, path in implementation_paths.items()
    }
    run_cfg = {
        **paths,
        "output_dir": str(output),
        "stream_commit": ORIGINAL_STREAM_COMMIT,
        "rlinf_commit": RLINF_COMMIT,
        "manifest": {
            "source": manifest["source"],
            "train": g721_specs["decision"] + g721_specs["audit"] + boundary_specs,
            "val_unseen": [],
        },
        "protocol": protocol,
        "policy_observation_keys": manifest["policy_observation_keys"],
        "context_tokens": args.context_tokens,
        "head_hidden": args.head_hidden,
        "seed": args.seed,
        "master_port": args.master_port,
        "learning_rate": protocol["learning_rate"],
        "value_learning_rate": protocol["value_learning_rate"],
        "gamma": protocol["gamma"],
        "gae_lambda": protocol["gae_lambda"],
        "clip_ratio": protocol["clip_ratio"],
        "value_coef": protocol["value_coef"],
        "entropy_coef": protocol["entropy_coef"],
    }
    config = {
        "kind": "streamvln-g7.23-executor-boundary-sensitivity",
        "manifest": str(manifest_path),
        "manifest_sha256": MANIFEST_SHA256,
        "predecessor_results": predecessor_results,
        "predecessor_sha256": G722_RESULT_SHA256,
        "source_heads": source_heads,
        "datasets": datasets,
        "implementation_sha256": implementation,
        "model_input_inventory": {
            "base_model": small_file_inventory(Path(paths["base_model"])),
            "tokenizer": small_file_inventory(Path(paths["tokenizer"])),
            "vision_tower": small_file_inventory(Path(paths["vision_tower"])),
        },
        "run": run_cfg,
        "seed": args.seed,
        "g721_scene_specs": g721_specs,
        "boundary_scene_specs": boundary_specs,
        "model_load_count": 1,
        "checkpoint_selection": False,
        "outcome_based_selection": False,
        "val_unseen_episode_count": 0,
    }
    config_path = output / "run_config.json"
    write_json(config_path, config)

    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    manager = core = None
    try:
        manager, core, trainable_manifest = setup_manager(
            run_cfg, total_steps=protocol["candidate_update_limit"]
        )
        optimizer_scope = optimizer_parameter_scope(manager, core)
        optimizer_initial = optimizer_state_contract(manager.optimizer)
        if optimizer_initial != protocol["optimizer_initial_contract"]:
            raise RuntimeError("G7.23 initial optimizer differs from eager-zero contract")
        initial_parameters = parameter_snapshot(core)
        initial_parameter_hash = state_sha256(initial_parameters)
        frozen_initial, frozen_names = frozen_parameter_sentinels(core)
        frozen_initial_hash = state_sha256(frozen_initial)

        decision_collection = collect_phase(
            manager, core, run_cfg, g721_specs["decision"],
            phase=f"g721_seed_{args.seed}_decision", seed_phase="g721_decision",
        )
        audit_collection = collect_phase(
            manager, core, run_cfg, g721_specs["audit"],
            phase=f"g721_seed_{args.seed}_audit", seed_phase="g721_audit",
        )
        decision_path = output / "reconstructed_decision_collection_records.jsonl"
        audit_path = output / "reconstructed_audit_collection_records.jsonl"
        write_jsonl(decision_path, decision_collection)
        write_jsonl(audit_path, audit_collection)
        if state_sha256(parameter_snapshot(core)) != initial_parameter_hash:
            raise RuntimeError("G7.23 G7.21 collection changed parameters")
        decision_batch = prepare_batch(manager, core, run_cfg, decision_collection)
        audit_batch = prepare_evaluation_batch(
            manager, core, audit_collection, protocol["audit_contract"], "audit"
        )
        if (
            decision_batch["preupdate_ratio_max_deviation"] > protocol["preupdate_ratio_max_deviation"]
            or audit_batch["preupdate_ratio_max_deviation"] > protocol["preupdate_ratio_max_deviation"]
        ):
            raise RuntimeError("G7.23 reconstructed state-zero ratio differs from one")
        state_zero = evaluate_state(
            manager, core, decision_batch, initial_parameters, "decision_state_zero"
        )
        baseline_snapshot = capture_training_state(manager, core)
        baseline_hashes = training_state_hashes(baseline_snapshot)
        if baseline_hashes != anchor["baseline_training_state_hashes"]:
            raise RuntimeError("G7.23 baseline state does not exactly reconstruct G7.21")

        boundary_initial = collect_phase(
            manager, core, run_cfg, boundary_specs,
            phase=f"g723_seed_{args.seed}_state_zero_boundary", seed_phase="g723_boundary",
        )
        annotate_policy_input_hashes(boundary_initial)
        boundary_initial_path = output / "state_zero_boundary_records.jsonl"
        write_jsonl(boundary_initial_path, boundary_initial)
        initial_boundary_restore = restore_training_state(manager, core, baseline_snapshot)
        if not initial_boundary_restore["exact"]:
            raise RuntimeError("G7.23 state-zero boundary collection did not restore exactly")

        previous_report = state_zero
        accepted_snapshot = baseline_snapshot
        candidates = []
        for candidate_index in range(1, protocol["candidate_update_limit"] + 1):
            before_snapshot = capture_training_state(manager, core)
            before_hashes = training_state_hashes(before_snapshot)
            reference_probe = fixed_continuation_probe(manager, decision_batch)
            reference_probe_restore = restore_training_state(manager, core, before_snapshot)
            if not reference_probe_restore["exact"]:
                raise RuntimeError("G7.23 reference probe did not restore exactly")
            before_parameters = parameter_snapshot(core)
            frozen_before, names_before = frozen_parameter_sentinels(core)
            if names_before != frozen_names:
                raise RuntimeError("G7.23 frozen sentinel schema changed")
            update = one_update(manager, core, run_cfg, decision_batch, candidate_index)
            report = evaluate_state(
                manager, core, decision_batch, before_parameters, f"candidate_{candidate_index}"
            )
            frozen_after, names_after = frozen_parameter_sentinels(core)
            if names_after != frozen_names:
                raise RuntimeError("G7.23 frozen sentinel schema changed after candidate")
            after_snapshot = capture_training_state(manager, core)
            after_hashes = training_state_hashes(after_snapshot)
            gates = candidate_gates(
                report, previous_report, update, protocol,
                actor_version_before=before_snapshot["actor_version"],
                optimizer_steps_before=before_snapshot["optimizer_steps"],
                frozen_before=state_sha256(frozen_before),
                frozen_after=state_sha256(frozen_after),
                input_sha256=decision_batch["input_sha256"],
            )
            accepted = all(gates.values())
            restoration = continuation = probe_cleanup_restore = None
            if accepted:
                accepted_snapshot = after_snapshot
                previous_report = report
                disposition = "committed_all_decision_gates_passed"
            else:
                restoration = restore_training_state(manager, core, before_snapshot)
                if not restoration["exact"]:
                    raise RuntimeError("G7.23 rejected candidate did not restore exactly")
                restored_probe = fixed_continuation_probe(manager, decision_batch)
                continuation = compare_probes(reference_probe, restored_probe)
                if not continuation["exact"]:
                    raise RuntimeError("G7.23 rejected candidate changed continuation")
                probe_cleanup_restore = restore_training_state(manager, core, before_snapshot)
                if not probe_cleanup_restore["exact"]:
                    raise RuntimeError("G7.23 continuation cleanup was not exact")
                accepted_snapshot = before_snapshot
                disposition = "rejected_restored_continuation_equal"
            row = {
                "candidate_index": candidate_index,
                "accepted": accepted,
                "disposition": disposition,
                "failed_gates": [name for name, passed in gates.items() if not passed],
                "gates": gates,
                "update": update,
                "state": report,
                "training_state_hashes_before": before_hashes,
                "training_state_hashes_candidate": after_hashes,
                "reference_probe": reference_probe,
                "reference_probe_restore": reference_probe_restore,
                "restoration": restoration,
                "continuation_equivalence": continuation,
                "probe_cleanup_restore": probe_cleanup_restore,
            }
            expected = anchor["candidates"][candidate_index - 1]
            reconstruction = {
                "candidate_index": candidate_index == expected["candidate_index"],
                "accepted": accepted == expected["accepted"],
                "failed_gates": row["failed_gates"] == expected["failed_gates"],
                "before_state": before_hashes == expected["training_state_hashes_before"],
                "candidate_state": after_hashes == expected["training_state_hashes_candidate"],
                "reference_probe": reference_probe == expected["reference_probe"],
                "rejected_continuation_probe": (
                    expected["rejected_continuation_probe"] is None
                    if accepted else continuation["restored"] == expected["rejected_continuation_probe"]
                ),
            }
            row["g721_reconstruction"] = reconstruction
            if not all(reconstruction.values()):
                raise RuntimeError(f"G7.23 candidate {candidate_index} differs from G7.21")
            candidates.append(row)
            if not accepted:
                break

        final_snapshot = capture_training_state(manager, core)
        final_hashes = training_state_hashes(final_snapshot)
        if final_hashes != anchor["final_training_state_hashes"]:
            raise RuntimeError("G7.23 final state does not exactly reconstruct G7.21")
        if final_hashes != training_state_hashes(accepted_snapshot):
            raise RuntimeError("G7.23 final state differs from last accepted state")

        boundary_final = collect_phase(
            manager, core, run_cfg, boundary_specs,
            phase=f"g723_seed_{args.seed}_final_boundary", seed_phase="g723_boundary",
        )
        annotate_policy_input_hashes(boundary_final)
        boundary_final_path = output / "final_boundary_records.jsonl"
        write_jsonl(boundary_final_path, boundary_final)
        final_boundary_restore = restore_training_state(manager, core, final_snapshot)
        if not final_boundary_restore["exact"]:
            raise RuntimeError("G7.23 final boundary collection did not restore exactly")
        final_after_boundary_hashes = training_state_hashes(capture_training_state(manager, core))
        frozen_final, final_names = frozen_parameter_sentinels(core)

        initial_metrics = aggregate_behavior(boundary_initial)
        final_metrics = aggregate_behavior(boundary_final)
        paired = paired_behavior(boundary_initial, boundary_final)
        delta = behavior_delta(initial_metrics, final_metrics)
        boundary_analysis = analyze_executor_boundary(
            boundary_initial, boundary_final, protocol["compiler_contract"]
        )
        guards = {
            "source_and_predecessors_anchored": source_heads == {
                "stream": ORIGINAL_STREAM_COMMIT, "rlinf": RLINF_COMMIT
            } and set(predecessor_results) == set(G722_RESULT_SHA256),
            "dataset_hashes_fixed": all(
                datasets[split]["sha256"] == manifest["source"][f"{split}_sha256"]
                for split in datasets
            ),
            "preregistered_seed_and_scene_specs": boundary_specs == FROZEN_BOUNDARY_ARMS[seed_key],
            "boundary_episodes_and_scenes_frozen": len(boundary_specs) == 8
            and set(row["scene"] for row in boundary_specs).issubset(set(manifest["selection"]["eligible_new_scenes"])),
            "ordinary_reconstruction_collection_semantics": collection_semantics(decision_collection)
            and collection_semantics(audit_collection),
            "ordinary_boundary_collection_semantics": collection_semantics(boundary_initial)
            and collection_semantics(boundary_final),
            "initial_optimizer_eager_zero_contract": optimizer_initial == protocol["optimizer_initial_contract"],
            "optimizer_scope_is_high_level_only": optimizer_scope["optimizer_equals_trainable"]
            and optimizer_scope["optimizer_frozen_overlap_count"] == 0,
            "baseline_exactly_reconstructed": baseline_hashes == anchor["baseline_training_state_hashes"],
            "every_candidate_exactly_reconstructed": all(
                all(row["g721_reconstruction"].values()) for row in candidates
            ) and len(candidates) == len(anchor["candidates"]),
            "final_exactly_reconstructed": final_hashes == anchor["final_training_state_hashes"],
            "boundary_snapshots_restored_exactly": exact_restore(initial_boundary_restore)
            and exact_restore(final_boundary_restore)
            and final_after_boundary_hashes == final_hashes,
            "paired_identity_observation_and_seed_schedule": paired["all_initial_observations_match"]
            and paired["all_seed_schedules_match"],
            "frozen_parameters_unchanged": final_names == frozen_names
            and state_sha256(frozen_final) == frozen_initial_hash,
            "no_new_policy_state_or_selection": protocol["no_new_policy_state_or_behavior_selection"]
            and config["checkpoint_selection"] is False and config["outcome_based_selection"] is False,
            "compiler_contract_holds": boundary_analysis["summary"]["compiler_contract_holds"],
            "same_sequence_causal_controls_match": boundary_analysis["summary"]["same_sequence_causal_control_feedback_match"],
            "zero_val_unseen": protocol["val_unseen_episode_count"] == 0,
        }
        protocol_valid = all(guards.values())
        result = {
            "kind": config["kind"],
            "seed": args.seed,
            "all_protocol_guards_passed": protocol_valid,
            "outcome": "executor_boundary_sensitivity_measured" if protocol_valid else "protocol_invalid",
            "navigation_or_generalization_claim": False,
            "source_contract": manifest["source_contract"],
            "trainable_manifest": trainable_manifest,
            "optimizer_scope": optimizer_scope,
            "optimizer_initial_contract": optimizer_initial,
            "counts": {
                "model_loads": 1,
                "reconstruction_decision_episodes": len(decision_collection),
                "reconstruction_audit_episodes": len(audit_collection),
                "candidate_updates_attempted": len(candidates),
                "candidate_updates_committed": sum(row["accepted"] for row in candidates),
                "state_zero_boundary_episodes": len(boundary_initial),
                "final_boundary_episodes": len(boundary_final),
                "val_unseen_episodes": 0,
            },
            "initial_high_level_sha256": initial_parameter_hash,
            "decision_state_zero": state_zero,
            "baseline_training_state_hashes": baseline_hashes,
            "candidates": candidates,
            "final_training_state_hashes": final_hashes,
            "boundary": {
                "state_zero": {
                    "records_sha256": sha256_file(boundary_initial_path),
                    "metrics": initial_metrics,
                },
                "final": {
                    "records_sha256": sha256_file(boundary_final_path),
                    "metrics": final_metrics,
                },
                "delta_final_minus_state_zero": delta,
                "paired_episodes": paired,
                "analysis": boundary_analysis,
            },
            "snapshot_restores": {
                "after_state_zero_boundary": initial_boundary_restore,
                "after_final_boundary": final_boundary_restore,
            },
            "protocol_guards": guards,
            "g8_ready": False,
            "elapsed_seconds": time.time() - started,
            "evidence": {
                "manifest_sha256": MANIFEST_SHA256,
                "run_config_sha256": sha256_file(config_path),
                "reconstructed_decision_records_sha256": sha256_file(decision_path),
                "reconstructed_audit_records_sha256": sha256_file(audit_path),
                "state_zero_boundary_records_sha256": sha256_file(boundary_initial_path),
                "final_boundary_records_sha256": sha256_file(boundary_final_path),
            },
        }
        result_path = output / "result.json"
        write_json(result_path, result)
        print(json.dumps({
            "result": str(result_path),
            "result_sha256": sha256_file(result_path),
            "seed": args.seed,
            "protocol_valid": protocol_valid,
            "causal_boundary_flip_count": boundary_analysis["summary"]["causal_boundary_flip_count"],
            "harmful_boundary_flip_count": boundary_analysis["summary"]["harmful_boundary_flip_count"],
            "flip_scenes": boundary_analysis["summary"]["causal_boundary_flip_scenes"],
        }, sort_keys=True))
    finally:
        if manager is not None:
            cleanup_manager(manager)


if __name__ == "__main__":
    main()
