"""Run the bounded G7.12 target-aligned privileged prefix-oracle audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

from streamvln_oracle_g712 import joint_gate, select_longest_admissible_prefix


ORIGINAL_STREAM_COMMIT = "e48f6ff7e9201d93aae003e8f64b04c00cec13bc"
RLINF_COMMIT = "0505431899574619da86f551bad70b71e0ea2177"
G711_RESULT_SHA256 = "edc26aa72f0296e5968a92d68dd132ca713f19c35dc5ea22b5821a500fa82ead"
G711_CONFIG_SHA256 = "8b23dfaa6a21bf9e1a19f572b649620ecd7c27adfcff6b41d8f23e0ce965c614"
G711_ROWS_SHA256 = "9dca2012f3cb08aadac6f429e2e0b8a572705ecc681f3264b9e88261a66c4fac"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(path):
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={path}",
            "-C",
            str(path),
            "rev-parse",
            "HEAD",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def distribution(values):
    values = sorted(float(value) for value in values)
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("G7.12 distribution is empty or non-finite")

    def percentile(fraction):
        index = (len(values) - 1) * fraction
        lower = int(index)
        upper = min(lower + 1, len(values) - 1)
        weight = index - lower
        return values[lower] * (1.0 - weight) + values[upper] * weight

    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "max": values[-1],
    }


def expected_contract():
    return {
        "split": "train",
        "train_dev_scenes": ["GdvgFV5R1Z5", "JF19kD82Mey"],
        "episode_indices": [1743, 7270],
        "state_definitions": [
            {"name": "reset", "transition_actions": []},
            {"name": "after_fixed_p0_d0p5", "transition_actions": [1, 1]},
        ],
        "candidate_count_per_state": 32,
        "row_count": 128,
        "recorded_repeat_count": 2,
        "selection_rule": "longest-prefix-with-every-prefix-nonsevere",
        "prefix_source": "accepted-g711-open-loop-exact-recorded-primitive-traces",
        "missing_counterfactual_policy": "stop-and-replay-only-missing-prefixes",
        "expected_missing_counterfactual_prefixes": 0,
        "severity_thresholds": {
            "severe_realized_ratio": 0.50,
            "severe_endpoint_error_m": 0.25,
            "severe_lateral_error_m": 0.25,
            "severe_min_primitive_progress_m": 0.05,
        },
        "variants": [
            "open_loop",
            "strict_navmesh_g711",
            "severity_aligned_prefix_oracle",
        ],
        "gates": {
            "severe_exposure_reduction_min": 0.70,
            "free_grazing_completion_min": 0.80,
            "free_grazing_progress_retention_min": 0.90,
            "overall_forward_primitive_retention_min": 0.50,
        },
        "g711_anchor": {
            "result_sha256": G711_RESULT_SHA256,
            "run_config_sha256": G711_CONFIG_SHA256,
            "rows_sha256": G711_ROWS_SHA256,
        },
        "model_fit": False,
        "navigation_evaluation": False,
        "val_unseen_episode_count": 0,
    }


def validate_anchor(result, rows):
    expected_classes = {
        "free": 54,
        "grazing": 7,
        "sliding": 28,
        "severe_blockage": 39,
    }
    if (
        result.get("all_protocol_guards_passed") is not True
        or result.get("anchor_audit", {}).get("exact") is not True
        or result.get("repeat_and_restore_audit", {}).get("physical_repeat_exact")
        is not True
        or result.get("repeat_and_restore_audit", {}).get("state_restore_exact")
        is not True
        or result.get("sample_summary", {}).get("row_count") != 128
        or result.get("sample_summary", {}).get("open_severity_class_counts")
        != expected_classes
        or result.get("val_unseen_episode_count") != 0
        or result.get("g8_ready") is not False
        or result.get("source_contract", {}).get("stream_commit")
        != ORIGINAL_STREAM_COMMIT
        or result.get("source_contract", {}).get("rlinf_commit") != RLINF_COMMIT
        or result.get("source_contract", {}).get("ttt_used") is not False
        or result.get("source_contract", {}).get("secondary_post_training_used")
        is not False
        or len(rows) != 128
    ):
        raise RuntimeError("G7.12 accepted G7.11 anchor changed")
    identities = {
        (row.get("scene"), row.get("state_name"), row.get("candidate"))
        for row in rows
    }
    if len(identities) != 128:
        raise RuntimeError("G7.12 row identity changed")


def variant_metrics(rows, variant, open_strata=None):
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
    open_primitives = sum(
        row["records"]["open_loop"]["executed_forward_primitives"] for row in rows
    )
    by_stratum = {}
    for scene in sorted({row["scene"] for row in rows}):
        for state in sorted({row["state_name"] for row in rows}):
            items = [
                row["records"][variant]
                for row in rows
                if row["scene"] == scene and row["state_name"] == state
            ]
            by_stratum[f"{scene}/{state}"] = {
                "count": len(items),
                "severe_exposure_count": sum(item["severe_exposure"] for item in items),
                "completed_count": sum(item["completed"] for item in items),
                "executed_forward_primitives": sum(
                    item["executed_forward_primitives"] for item in items
                ),
            }
    nonincrease = True
    if open_strata is not None:
        nonincrease = all(
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
        "baseline_severe_count": len(baseline_severe),
        "severe_exposure_count_on_baseline_severe": severe_exposure,
        "severe_exposure_reduction": 1.0 - severe_exposure / len(baseline_severe),
        "free_grazing_count": len(safe),
        "free_grazing_completion_rate": sum(
            record["completed"] for _, record in safe
        )
        / len(safe),
        "free_grazing_progress_retention": sum(
            max(0.0, record["realized_forward_m"]) for _, record in safe
        )
        / open_safe_progress,
        "overall_forward_primitive_retention": sum(
            record["executed_forward_primitives"] for record in records
        )
        / open_primitives,
        "stratum_severe_nonincrease": nonincrease,
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
        "selected_prefix_count": distribution(
            record["executed_forward_primitives"] for record in records
        ),
        "by_scene_state": by_stratum,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--g711-result", required=True)
    parser.add_argument("--g711-result-sha256", required=True)
    parser.add_argument("--g711-run-config", required=True)
    parser.add_argument("--g711-run-config-sha256", required=True)
    parser.add_argument("--g711-rows", required=True)
    parser.add_argument("--g711-rows-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rlinf-root", required=True)
    parser.add_argument("--stream-source", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    paths = {
        name: Path(getattr(args, name)).expanduser().resolve(strict=True)
        for name in (
            "manifest",
            "g711_result",
            "g711_run_config",
            "g711_rows",
            "rlinf_root",
            "stream_source",
        )
    }
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("G7.12 output directory must be fresh")
    output.mkdir(parents=True, exist_ok=True)
    supplied_hashes = {
        "result": args.g711_result_sha256,
        "run_config": args.g711_run_config_sha256,
        "rows": args.g711_rows_sha256,
    }
    expected_hashes = {
        "result": G711_RESULT_SHA256,
        "run_config": G711_CONFIG_SHA256,
        "rows": G711_ROWS_SHA256,
    }
    actual_hashes = {
        "result": sha256_file(paths["g711_result"]),
        "run_config": sha256_file(paths["g711_run_config"]),
        "rows": sha256_file(paths["g711_rows"]),
    }
    if supplied_hashes != expected_hashes or actual_hashes != expected_hashes:
        raise RuntimeError("G7.12 G7.11 anchor hash changed")
    manifest = json.loads(paths["manifest"].read_text())
    if manifest.get("g712_target_aligned_oracle") != expected_contract():
        raise RuntimeError("G7.12 preregistration changed")
    stream_commit = git_head(paths["stream_source"])
    rlinf_commit = git_head(paths["rlinf_root"])
    if stream_commit != ORIGINAL_STREAM_COMMIT or rlinf_commit != RLINF_COMMIT:
        raise RuntimeError("G7.12 source commit pin changed")
    g711_result = json.loads(paths["g711_result"].read_text())
    source_rows = [
        json.loads(line)
        for line in paths["g711_rows"].read_text().splitlines()
        if line.strip()
    ]
    validate_anchor(g711_result, source_rows)
    run_config = {
        "manifest": str(paths["manifest"]),
        "g711_result": str(paths["g711_result"]),
        "g711_run_config": str(paths["g711_run_config"]),
        "g711_rows": str(paths["g711_rows"]),
        "g711_hashes": actual_hashes,
        "output_dir": str(output),
        "rlinf_root": str(paths["rlinf_root"]),
        "stream_source": str(paths["stream_source"]),
        "rlinf_commit": rlinf_commit,
        "stream_commit": stream_commit,
        "preregistration": expected_contract(),
        "model_fit": False,
        "navigation_evaluation": False,
        "val_unseen_episode_count": 0,
    }
    (output / "run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n"
    )
    started = time.time()
    rows = []
    repeat_exact = True
    missing_prefixes = 0
    for source in source_rows:
        physical = source.get("physical", {}).get("open_loop", {})
        repeats = physical.get("repeats")
        if (
            physical.get("repeat_exact") is not True
            or not isinstance(repeats, list)
            or len(repeats) != 2
        ):
            raise RuntimeError("G7.12 exact open-loop repeats are missing")
        selected = [select_longest_admissible_prefix(repeat) for repeat in repeats]
        if selected[0] != selected[1]:
            repeat_exact = False
        for repeat, selection in zip(repeats, selected):
            commanded = repeat["commanded_forward_primitives"]
            if len(repeat["primitive_trace"]) < commanded:
                missing_prefixes += 1
            if selection["selected"]["selected_prefix"] > commanded:
                raise RuntimeError("G7.12 selected prefix exceeds recorded trace")
        record = selected[0]["selected"]
        rows.append(
            {
                "scene": source["scene"],
                "episode_id": source["episode_id"],
                "state_name": source["state_name"],
                "state_index": source["state_index"],
                "candidate": source["candidate"],
                "actions": source["actions"],
                "open_severity_class": source["open_severity_class"],
                "selection_rule": "longest-prefix-with-every-prefix-nonsevere",
                "evaluated_prefixes": selected[0]["evaluated_prefixes"],
                "selected_prefix": record["selected_prefix"],
                "repeat_exact": selected[0] == selected[1],
                "records": {
                    "open_loop": source["records"]["open_loop"],
                    "strict_navmesh_g711": source["records"][
                        "navmesh_receding_oracle"
                    ],
                    "severity_aligned_prefix_oracle": record,
                },
            }
        )
    if missing_prefixes != 0:
        raise RuntimeError(
            "G7.12 missing counterfactual prefixes require a separate exact replay"
        )
    metrics = {"open_loop": variant_metrics(rows, "open_loop")}
    open_strata = metrics["open_loop"]["by_scene_state"]
    for variant in ("strict_navmesh_g711", "severity_aligned_prefix_oracle"):
        metrics[variant] = variant_metrics(rows, variant, open_strata=open_strata)
    integrity = (
        repeat_exact
        and missing_prefixes == 0
        and len(rows) == 128
        and all(row["repeat_exact"] for row in rows)
        and metrics["open_loop"]["baseline_severe_count"] == 39
        and metrics["open_loop"]["free_grazing_count"] == 61
    )
    gate = joint_gate(
        metrics["severity_aligned_prefix_oracle"], integrity_ready=integrity
    )
    rows_path = output / "oracle_rows.jsonl"
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    selected_counts = {}
    for row in rows:
        key = str(row["selected_prefix"])
        selected_counts[key] = selected_counts.get(key, 0) + 1
    result = {
        "schema_version": 1,
        "kind": "streamvln-g712-target-aligned-privileged-prefix-oracle-audit",
        "split": "train",
        "source_contract": {
            "source_kind": "upstream-original",
            "stream_commit": stream_commit,
            "rlinf_commit": rlinf_commit,
            "ttt_used": False,
            "secondary_post_training_used": False,
        },
        "g711_anchor": {
            "hashes": actual_hashes,
            "all_protocol_guards_passed": True,
            "repeat_and_restore_exact": True,
        },
        "preregistration": expected_contract(),
        "sample_summary": {
            "row_count": len(rows),
            "scene_count": len({row["scene"] for row in rows}),
            "state_count": len({row["state_name"] for row in rows}),
            "selected_prefix_counts": selected_counts,
            "reused_recorded_repeat_count": len(rows) * 2,
            "missing_counterfactual_prefix_count": missing_prefixes,
            "new_physical_replay_count": 0,
        },
        "protocol_guards": {
            "fixed_128_rows": len(rows) == 128,
            "fixed_two_train_dev_scenes": len({row["scene"] for row in rows}) == 2,
            "fixed_two_states": len({row["state_name"] for row in rows}) == 2,
            "g711_hash_anchor_exact": actual_hashes == expected_hashes,
            "g711_protocol_anchor_passed": True,
            "recorded_repeat_exact": repeat_exact,
            "all_selected_prefixes_recorded": missing_prefixes == 0,
            "severity_thresholds_unchanged": True,
            "no_threshold_sweep": True,
            "no_model_fit": True,
            "no_navigation_evaluation": True,
            "zero_val_unseen": True,
        },
        "all_protocol_guards_passed": integrity,
        "variant_metrics": metrics,
        "oracle_gate": gate,
        "causal_observability_audit_ready": bool(gate["ready"]),
        "g8_ready": False,
        "routing": (
            "run-causal-ground-support-observability-audit"
            if gate["ready"]
            else "repair-macro-controller-contract-before-perception"
        ),
        "claim_scope": "train-only privileged target-aligned oracle contract audit on accepted G7.11 recorded prefixes; no causal controller, navigation, language, RGB, val_unseen or generalization claim",
        "evidence": {
            "rows_path": str(rows_path),
            "rows_sha256": sha256_file(rows_path),
        },
        "elapsed_s": time.time() - started,
        "val_unseen_episode_count": 0,
    }
    if not integrity:
        raise RuntimeError("G7.12 integrity guard failed")
    result_path = output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "all_protocol_guards_passed": integrity,
                "oracle_gate": gate,
                "causal_observability_audit_ready": result[
                    "causal_observability_audit_ready"
                ],
                "g8_ready": False,
                "routing": result["routing"],
                "variant_metrics": metrics,
                "result": str(result_path),
                "result_sha256": sha256_file(result_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
