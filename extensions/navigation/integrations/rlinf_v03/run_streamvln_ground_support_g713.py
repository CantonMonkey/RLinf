"""Run the preregistered G7.13 causal ground-support observability audit."""

from __future__ import annotations

import argparse
import base64
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np

from run_streamvln_candidates_g7 import configure_process_git_safe_directory, validate_identity
from run_streamvln_ppo_g4 import make_env, validate_dataset_hashes, validate_manifest
from run_streamvln_topology_g5 import ORIGINAL_STREAM_COMMIT, git_head, jsonable, sha256_file
from streamvln_candidates_g74 import generate_macro_candidates
from streamvln_ground_support_g713 import (
    binary_auroc,
    ground_support_observables,
    observability_gate,
)
from streamvln_macro_audit_g79 import (
    REGISTERED_CANDIDATES_PER_SCENE,
    select_registered_candidates,
)
from streamvln_topology import OnlineTopology


RLINF_COMMIT = "0505431899574619da86f551bad70b71e0ea2177"
PRIMARY_RISK = "footprint_dropoff_fraction"
DESCRIPTIVE_SCORES = [
    "footprint_support_fraction",
    "footprint_occlusion_fraction",
    "center_dropoff_fraction",
    "left_dropoff_fraction",
    "right_dropoff_fraction",
    "maximum_side_dropoff_fraction",
]


def circular_gap(left, right):
    return abs((float(left) - float(right) + 180.0) % 360.0 - 180.0)


def load_anchor(result_path, result_hash, config_path, config_hash, rows_path, rows_hash, kind):
    actual = {
        "result": sha256_file(result_path),
        "run_config": sha256_file(config_path),
        "rows": sha256_file(rows_path),
    }
    expected = {"result": result_hash, "run_config": config_hash, "rows": rows_hash}
    if actual != expected:
        raise RuntimeError("G7.13 accepted anchor hash changed")
    result = json.loads(result_path.read_text())
    if (
        result.get("kind") != kind
        or result.get("all_protocol_guards_passed") is not True
        or result.get("val_unseen_episode_count") != 0
        or result.get("source_contract", {}).get("stream_commit")
        != ORIGINAL_STREAM_COMMIT
        or result.get("source_contract", {}).get("rlinf_commit") != RLINF_COMMIT
    ):
        raise RuntimeError("G7.13 accepted anchor contract changed")
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line]
    by_key = {(row["scene"], row["state_name"], row["candidate"]): row for row in rows}
    if len(rows) != 128 or len(by_key) != 128:
        raise RuntimeError("G7.13 accepted anchor row count changed")
    return result, by_key


def reconstruct_pairs(g711_rows, g712_rows):
    residuals = []
    controls = []
    for key in sorted(g711_rows):
        left = g711_rows[key]
        right = g712_rows.get(key)
        if right is None:
            raise RuntimeError("G7.13 G7.11/G7.12 row join changed")
        commanded = int(right["records"]["open_loop"]["commanded_forward_primitives"])
        selected = int(right["selected_prefix"])
        item = {
            "scene": left["scene"],
            "state_name": left["state_name"],
            "candidate": left["candidate"],
            "heading_degrees": float(left["heading_degrees"]),
            "commanded_forward_primitives": commanded,
            "selected_prefix": selected,
        }
        if (
            left["records"]["open_loop"]["severe_exposure"]
            and left["records"]["depth_receding"]["severe_exposure"]
        ):
            item["critical_prefix"] = selected + 1
            if not 1 <= item["critical_prefix"] <= commanded:
                raise RuntimeError("G7.13 residual critical prefix changed")
            residuals.append(item)
        if (
            not left["records"]["open_loop"]["severe_exposure"]
            and selected == commanded
        ):
            controls.append(item)
    if len(residuals) != 26:
        raise RuntimeError("G7.13 residual-positive count changed")

    grouped = {}
    for item in residuals:
        group = (
            item["scene"],
            item["state_name"],
            item["commanded_forward_primitives"],
        )
        grouped.setdefault(group, []).append(item)
    assigned = []
    for group, targets in sorted(grouped.items()):
        targets = sorted(targets, key=lambda row: (row["critical_prefix"], row["candidate"]))
        eligible = sorted(
            [
                row
                for row in controls
                if (
                    row["scene"],
                    row["state_name"],
                    row["commanded_forward_primitives"],
                )
                == group
            ],
            key=lambda row: row["candidate"],
        )
        best = None
        for proposal in itertools.permutations(eligible, len(targets)):
            gaps = [
                circular_gap(target["heading_degrees"], control["heading_degrees"])
                for target, control in zip(targets, proposal)
            ]
            objective = (
                max(gaps),
                sum(gaps),
                tuple(control["candidate"] for control in proposal),
            )
            if best is None or objective < best[0]:
                best = (objective, proposal)
        if best is None:
            raise RuntimeError("G7.13 one-to-one safe matching is infeasible")
        assigned.extend(zip(targets, best[1]))

    pairs = []
    ordered = sorted(
        assigned,
        key=lambda pair: (
            pair[0]["scene"],
            pair[0]["state_name"],
            pair[0]["critical_prefix"],
            pair[0]["candidate"],
        ),
    )
    for pair_index, (target, control) in enumerate(ordered, 1):
        pairs.append(
            {
                "pair_index": pair_index,
                "scene": target["scene"],
                "state_name": target["state_name"],
                "critical_prefix": target["critical_prefix"],
                "commanded_forward_primitives": target[
                    "commanded_forward_primitives"
                ],
                "residual_candidate": target["candidate"],
                "residual_heading_degrees": target["heading_degrees"],
                "control_candidate": control["candidate"],
                "control_heading_degrees": control["heading_degrees"],
                "heading_gap_degrees": circular_gap(
                    target["heading_degrees"], control["heading_degrees"]
                ),
            }
        )
    return pairs


def encode_depth(depth, expected_sha256):
    array = np.asarray(depth, dtype="<f4")
    if array.shape != (64, 64):
        raise RuntimeError("G7.13 decoded depth shape changed")
    raw = array.tobytes(order="C")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise RuntimeError("G7.13 decoded depth bytes changed")
    return {
        "dtype": "float32-le",
        "shape": [64, 64],
        "data_b64": base64.b64encode(raw).decode("ascii"),
    }


def vector_error(left, right):
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))


def candidate_cases(pairs, candidate_by_name):
    cases = []
    for pair in pairs:
        for role, field in (("positive", "residual_candidate"), ("control", "control_candidate")):
            candidate = candidate_by_name[pair[field]]
            if len([action for action in candidate.actions if action == 1]) != pair[
                "commanded_forward_primitives"
            ]:
                raise RuntimeError("G7.13 candidate command length changed")
            cases.append(
                {
                    "name": f"pair_{pair['pair_index']:02d}_{role}",
                    "pair_index": pair["pair_index"],
                    "role": role,
                    "scene": pair["scene"],
                    "state_name": pair["state_name"],
                    "candidate": pair[field],
                    "actions": list(candidate.actions),
                    "observe_before_forward_index": pair["critical_prefix"],
                }
            )
    if len(cases) != 52 or len({case["name"] for case in cases}) != 52:
        raise RuntimeError("G7.13 case expansion changed")
    return cases


def collect_scene(run_cfg, spec, cases, g711_rows):
    cfg = dict(run_cfg)
    cfg["habitat_server"] = run_cfg["habitat_ground_support_server"]
    env = make_env(
        cfg,
        split="train",
        episode_index=spec["index"],
        max_steps=run_cfg["max_episode_steps"],
    )
    rows = []
    snapshots = []
    states = []
    try:
        _, identity = env.reset()
        validate_identity(identity, spec)
        for state_index, state in enumerate(run_cfg["preregistration"]["state_definitions"]):
            transition_records = []
            for action in state["transition_actions"]:
                _, _, terminated, truncated, infos = env.step(action)
                transition_records.append(
                    {
                        "action": int(action),
                        "collision": bool(infos["collision"].item()),
                        "terminated": bool(terminated.item()),
                        "truncated": bool(truncated.item()),
                    }
                )
                if bool(terminated.item()) or bool(truncated.item()):
                    raise RuntimeError("G7.13 state transition ended the episode")
            selected = [
                case
                for case in cases
                if case["scene"] == spec["scene"] and case["state_name"] == state["name"]
            ]
            payload = [
                {
                    "name": case["name"],
                    "actions": case["actions"],
                    "observe_before_forward_index": case[
                        "observe_before_forward_index"
                    ],
                }
                for case in selected
            ]
            audit = env.probe_ground_support(payload)
            by_name = {row["name"]: row for row in audit["candidates"]}
            if set(by_name) != {case["name"] for case in selected}:
                raise RuntimeError("G7.13 service case set changed")
            for case in selected:
                probe = by_name[case["name"]]
                anchor = g711_rows[(case["scene"], case["state_name"], case["candidate"])]
                anchor_repeat = anchor["physical"]["open_loop"]["repeats"][0]
                prefix_count = case["observe_before_forward_index"] - 1
                expected_trace = anchor_repeat["primitive_trace"][:prefix_count]
                expected_position = (
                    anchor_repeat["post_turn_pose"]["position"]
                    if prefix_count == 0
                    else expected_trace[-1]["after_position"]
                )
                repeats = []
                anchor_errors = []
                for repeat_index, repeat in enumerate(probe["repeats"]):
                    observables = ground_support_observables(repeat["metric_depth"])
                    payload_depth = encode_depth(
                        repeat["metric_depth"], repeat["depth_sha256"]
                    )
                    anchor_error = vector_error(
                        repeat["pre_action_pose"]["position"], expected_position
                    )
                    anchor_errors.append(anchor_error)
                    if repeat["prefix_trace"] != expected_trace:
                        raise RuntimeError("G7.13 executed prefix differs from G7.11 anchor")
                    snapshots.append(
                        {
                            "name": case["name"],
                            "pair_index": case["pair_index"],
                            "role": case["role"],
                            "scene": case["scene"],
                            "state_name": case["state_name"],
                            "candidate": case["candidate"],
                            "repeat_index": repeat_index,
                            "depth_sha256": repeat["depth_sha256"],
                            "metric_depth": payload_depth,
                        }
                    )
                    repeats.append(
                        {
                            "repeat_index": repeat_index,
                            "turn_collisions": repeat["turn_collisions"],
                            "executed_prefix_count": repeat["executed_prefix_count"],
                            "prefix_trace": repeat["prefix_trace"],
                            "pre_action_pose": repeat["pre_action_pose"],
                            "forward_direction": repeat["forward_direction"],
                            "depth_sha256": repeat["depth_sha256"],
                            "observables": observables,
                            "g711_anchor_position_error_m": anchor_error,
                        }
                    )
                repeat_exact = (
                    repeats[0]["turn_collisions"] == repeats[1]["turn_collisions"]
                    and repeats[0]["prefix_trace"] == repeats[1]["prefix_trace"]
                    and repeats[0]["forward_direction"] == repeats[1]["forward_direction"]
                    and repeats[0]["depth_sha256"] == repeats[1]["depth_sha256"]
                    and repeats[0]["observables"] == repeats[1]["observables"]
                    and vector_error(
                        repeats[0]["pre_action_pose"]["position"],
                        repeats[1]["pre_action_pose"]["position"],
                    )
                    <= 1e-6
                    and vector_error(
                        repeats[0]["pre_action_pose"]["rotation_coefficients"],
                        repeats[1]["pre_action_pose"]["rotation_coefficients"],
                    )
                    <= 1e-6
                )
                rows.append(
                    {
                        **{key: value for key, value in case.items() if key != "actions"},
                        "actions": case["actions"],
                        "state_index": state_index,
                        "state_transition_actions": list(state["transition_actions"]),
                        "state_transition_collisions": [
                            record["collision"] for record in transition_records
                        ],
                        "label_residual_severe": case["role"] == "positive",
                        "repeat_exact": repeat_exact,
                        "maximum_g711_anchor_position_error_m": max(anchor_errors),
                        "repeats": repeats,
                        "risk_score": repeats[0]["observables"][PRIMARY_RISK],
                    }
                )
            states.append(
                {
                    "name": state["name"],
                    "transition_actions": list(state["transition_actions"]),
                    "transition_records": transition_records,
                    "case_count": len(selected),
                    "max_position_restore_error_m": audit[
                        "max_position_restore_error_m"
                    ],
                    "max_rotation_restore_error": audit[
                        "max_rotation_restore_error"
                    ],
                }
            )
        return (
            {"scene": spec["scene"], "episode_id": str(spec["episode_id"]), "states": states},
            rows,
            snapshots,
        )
    finally:
        env.close()


def compute_metrics(rows):
    labels = [bool(row["label_residual_severe"]) for row in rows]
    feature_aurocs = {}
    for feature in [PRIMARY_RISK, *DESCRIPTIVE_SCORES]:
        scores = [float(row["repeats"][0]["observables"][feature]) for row in rows]
        direction = -1.0 if feature == "footprint_support_fraction" else 1.0
        feature_aurocs[feature] = binary_auroc(
            [direction * value for value in scores], labels
        )
    overall = feature_aurocs[PRIMARY_RISK]
    by_scene = {}
    for scene in sorted({row["scene"] for row in rows}):
        subset = [row for row in rows if row["scene"] == scene]
        by_scene[scene] = binary_auroc(
            [float(row["risk_score"]) for row in subset],
            [bool(row["label_residual_severe"]) for row in subset],
        )
    by_state = {}
    for state in sorted({row["state_name"] for row in rows}):
        subset = [row for row in rows if row["state_name"] == state]
        by_state[state] = binary_auroc(
            [float(row["risk_score"]) for row in subset],
            [bool(row["label_residual_severe"]) for row in subset],
        )
    paired = {}
    for row in rows:
        paired.setdefault(row["pair_index"], {})[row["role"]] = row
    if len(paired) != 26 or any(set(value) != {"positive", "control"} for value in paired.values()):
        raise RuntimeError("G7.13 matched pairs are incomplete")
    strict_wins = sum(
        value["positive"]["risk_score"] > value["control"]["risk_score"]
        for value in paired.values()
    )
    positive = [float(row["risk_score"]) for row in rows if row["label_residual_severe"]]
    control = [float(row["risk_score"]) for row in rows if not row["label_residual_severe"]]
    return {
        "primary_risk_score": PRIMARY_RISK,
        "overall_auroc": overall,
        "by_scene_auroc": by_scene,
        "minimum_scene_auroc": min(by_scene.values()),
        "by_state_auroc": by_state,
        "minimum_state_auroc": min(by_state.values()),
        "paired_strict_win_count": strict_wins,
        "paired_strict_win_rate": strict_wins / 26.0,
        "positive_median_risk": statistics.median(positive),
        "control_median_risk": statistics.median(control),
        "feature_aurocs": feature_aurocs,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    for name in (
        "manifest",
        "g711-result",
        "g711-result-sha256",
        "g711-run-config",
        "g711-run-config-sha256",
        "g711-rows",
        "g711-rows-sha256",
        "g712-result",
        "g712-result-sha256",
        "g712-run-config",
        "g712-run-config-sha256",
        "g712-rows",
        "g712-rows-sha256",
        "output-dir",
        "rlinf-root",
        "stream-source",
        "habitat-python",
        "habitat-ground-support-server",
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
        raise ValueError("G7.13 episode budget is invalid")
    path_names = (
        "manifest",
        "g711_result",
        "g711_run_config",
        "g711_rows",
        "g712_result",
        "g712_run_config",
        "g712_rows",
        "rlinf_root",
        "stream_source",
        "habitat_python",
        "habitat_ground_support_server",
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
        raise FileExistsError("G7.13 output directory must be fresh")
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(paths["manifest"]).read_text())
    validate_manifest(manifest)
    prereg = manifest.get("g713_ground_support_observability", {})
    if (
        prereg.get("split") != "train"
        or prereg.get("pair_count") != 26
        or prereg.get("row_count") != 52
        or prereg.get("repeat_count") != 2
        or prereg.get("depth_geometry", {}).get("primary_risk_score") != PRIMARY_RISK
        or prereg.get("depth_geometry", {}).get("descriptive_scores")
        != DESCRIPTIVE_SCORES
        or prereg.get("model_fit") is not False
        or prereg.get("threshold_sweep") is not False
        or prereg.get("controller_implementation") is not False
        or prereg.get("navigation_evaluation") is not False
        or prereg.get("val_unseen_episode_count") != 0
    ):
        raise RuntimeError("G7.13 manifest preregistration changed")
    configure_process_git_safe_directory(paths["stream_source"])
    configure_process_git_safe_directory(paths["rlinf_root"])
    stream_commit = git_head(Path(paths["stream_source"]))
    rlinf_commit = git_head(Path(paths["rlinf_root"]))
    if stream_commit != ORIGINAL_STREAM_COMMIT or rlinf_commit != RLINF_COMMIT:
        raise RuntimeError("G7.13 source commit pin changed")
    _, g711_rows = load_anchor(
        Path(paths["g711_result"]),
        args.g711_result_sha256,
        Path(paths["g711_run_config"]),
        args.g711_run_config_sha256,
        Path(paths["g711_rows"]),
        args.g711_rows_sha256,
        "streamvln-g7.11-explicit-depth-footprint-controller-gate",
    )
    _, g712_rows = load_anchor(
        Path(paths["g712_result"]),
        args.g712_result_sha256,
        Path(paths["g712_run_config"]),
        args.g712_run_config_sha256,
        Path(paths["g712_rows"]),
        args.g712_rows_sha256,
        "streamvln-g712-target-aligned-privileged-prefix-oracle-audit",
    )
    reconstructed_pairs = reconstruct_pairs(g711_rows, g712_rows)
    frozen_pairs = prereg.get("matching", {}).get("pairs")
    if reconstructed_pairs != frozen_pairs:
        raise RuntimeError("G7.13 frozen matched pairs do not reproduce from anchors")
    if (
        max(pair["heading_gap_degrees"] for pair in reconstructed_pairs) != 120.0
        or sum(pair["heading_gap_degrees"] for pair in reconstructed_pairs) != 1095.0
    ):
        raise RuntimeError("G7.13 frozen heading-match summary changed")
    topology = OnlineTopology(max_decisions=args.max_episode_steps)
    candidates = select_registered_candidates(generate_macro_candidates(topology))
    if len(candidates) != REGISTERED_CANDIDATES_PER_SCENE:
        raise RuntimeError("G7.13 exact macro bank changed")
    candidate_by_name = {candidate.name: candidate for candidate in candidates}
    cases = candidate_cases(reconstructed_pairs, candidate_by_name)
    run_cfg = {
        **paths,
        "manifest": manifest,
        "output_dir": str(output),
        "stream_commit": stream_commit,
        "rlinf_commit": rlinf_commit,
        "seed": args.seed,
        "max_episode_steps": args.max_episode_steps,
        "preregistration": prereg,
        "reconstructed_pairs": reconstructed_pairs,
        "val_unseen_episode_count": 0,
        "model_fit": False,
        "threshold_sweep": False,
        "controller_implementation": False,
        "navigation_evaluation": False,
    }
    dataset_hashes = validate_dataset_hashes(run_cfg)
    (output / "run_config.json").write_text(
        json.dumps(run_cfg, indent=2, sort_keys=True) + "\n"
    )
    started = time.time()
    episodes = []
    rows = []
    snapshots = []
    for spec in manifest["train"][6:]:
        episode, scene_rows, scene_snapshots = collect_scene(
            run_cfg, spec, cases, g711_rows
        )
        episodes.append(episode)
        rows.extend(scene_rows)
        snapshots.extend(scene_snapshots)
    if len(rows) != 52 or len(snapshots) != 104 or len(episodes) != 2:
        raise RuntimeError("G7.13 bounded collection count changed")

    repeat_exact = all(row["repeat_exact"] for row in rows)
    anchor_prefix_exact = all(
        row["maximum_g711_anchor_position_error_m"] <= 1e-6 for row in rows
    )
    restore_exact = all(
        state["max_position_restore_error_m"] <= 1e-6
        and state["max_rotation_restore_error"] <= 1e-6
        for episode in episodes
        for state in episode["states"]
    )
    class_scene_state_support = all(
        any(
            row["scene"] == scene
            and row["state_name"] == state
            and row["label_residual_severe"] is label
            for row in rows
        )
        for scene in prereg["train_dev_scenes"]
        for state in [item["name"] for item in prereg["state_definitions"]]
        for label in (False, True)
    )
    protocol_guards = {
        "original_source_pins": True,
        "g711_anchor_hashes_and_guards_exact": True,
        "g712_anchor_hashes_and_guards_exact": True,
        "frozen_pair_reconstruction_exact": reconstructed_pairs == frozen_pairs,
        "fixed_26_pairs_52_unique_rows": len(rows) == 52
        and len({(row["scene"], row["state_name"], row["candidate"]) for row in rows})
        == 52,
        "two_repeats_per_row": all(len(row["repeats"]) == 2 for row in rows),
        "repeat_depth_pose_prefix_features_exact": repeat_exact,
        "g711_prefix_anchor_exact": anchor_prefix_exact,
        "state_restore_exact": restore_exact,
        "both_classes_in_both_scenes_and_states": class_scene_state_support,
        "causal_features_exclude_privileged_inputs": True,
        "no_model_fit": True,
        "no_threshold_sweep": True,
        "no_controller_implementation": True,
        "no_navigation_evaluation": True,
        "zero_val_unseen": True,
    }
    integrity_ready = all(protocol_guards.values())
    metrics = compute_metrics(rows)
    gate = observability_gate(metrics, integrity_ready=integrity_ready)
    if not integrity_ready:
        route = "repair-g713-instrumentation-or-anchor-reproduction"
    elif gate["ready"]:
        route = "freeze-ground-support-observables-and-run-fresh-truncated-prefix-controller-gate"
    else:
        route = "close-forward-depth-ground-support-representation-and-reassess-observation-action-or-sensor-contract"

    rows_path = output / "observation_rows.jsonl"
    with rows_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(jsonable(row), sort_keys=True) + "\n")
    snapshots_path = output / "depth_snapshots.jsonl"
    with snapshots_path.open("w") as handle:
        for row in snapshots:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    result = {
        "schema_version": 1,
        "kind": "streamvln-g7.13-causal-ground-support-observability-audit",
        "claim_scope": "fixed two-scene/two-state no-fit pre-action metric-depth observability audit; no controller, navigation, language, RGB, val_unseen or generalization claim",
        "split": "train",
        "val_unseen_episode_count": 0,
        "source_contract": {
            "stream_commit": stream_commit,
            "rlinf_commit": rlinf_commit,
            "source_kind": "upstream-original",
            "secondary_post_training_used": False,
            "ttt_used": False,
        },
        "dataset_hashes": dataset_hashes,
        "preregistration": prereg,
        "episodes": episodes,
        "sample_summary": {
            "pair_count": 26,
            "row_count": len(rows),
            "positive_count": sum(row["label_residual_severe"] for row in rows),
            "control_count": sum(not row["label_residual_severe"] for row in rows),
            "snapshot_count": len(snapshots),
        },
        "observability_metrics": metrics,
        "observability_gate": gate,
        "protocol_guards": protocol_guards,
        "all_protocol_guards_passed": integrity_ready,
        "causal_observability_ready": gate["ready"],
        "controller_ready": False,
        "navigation_ready": False,
        "g8_ready": False,
        "routing": route,
        "evidence": {
            "rows_path": str(rows_path),
            "rows_sha256": sha256_file(rows_path),
            "depth_snapshots_path": str(snapshots_path),
            "depth_snapshots_sha256": sha256_file(snapshots_path),
            "g711_result_sha256": args.g711_result_sha256,
            "g711_run_config_sha256": args.g711_run_config_sha256,
            "g711_rows_sha256": args.g711_rows_sha256,
            "g712_result_sha256": args.g712_result_sha256,
            "g712_run_config_sha256": args.g712_run_config_sha256,
            "g712_rows_sha256": args.g712_rows_sha256,
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
                "all_protocol_guards_passed": integrity_ready,
                "causal_observability_ready": gate["ready"],
                "g8_ready": False,
                "routing": route,
                "metrics": metrics,
                "gate": gate,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
