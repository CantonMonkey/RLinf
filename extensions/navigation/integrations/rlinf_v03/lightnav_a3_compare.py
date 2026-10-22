#!/usr/bin/env python3
"""Create the matched before/after result for the LightNav A3 pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence


class ComparisonError(RuntimeError):
    """Raised when two evaluation receipts are not a matched A3 pair."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ComparisonError(f"result is not an object: {path}")
    return value


def _mean(rows: Sequence[Mapping[str, object]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def _summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    collision_available = all(row["collision_count"] is not None for row in rows)
    return {
        "episode_count": len(rows),
        "success_rate": sum(bool(row["success"]) for row in rows) / len(rows),
        "mean_spl": _mean(rows, "spl"),
        "mean_final_navigation_error_m": _mean(rows, "final_navigation_error_m"),
        "mean_path_length_m": _mean(rows, "path_length_m"),
        "collision_count": (
            sum(float(row["collision_count"]) for row in rows)
            if collision_available
            else None
        ),
        "blocked_step_count": sum(int(row["blocked_step_count"]) for row in rows),
        "native_stop_count": sum(row["stop_source"] == "native_lightnav_stop" for row in rows),
        "learned_stop_count": sum(row["stop_source"] == "learned_stop" for row in rows),
        "timeout_count": sum(bool(row["timeout"]) for row in rows),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--baseline", required=True, type=Path)
    result.add_argument("--post", required=True, type=Path)
    result.add_argument("--training-result", required=True, type=Path)
    result.add_argument("--expected-run-manifest-sha256", required=True)
    result.add_argument("--output-json", required=True, type=Path)
    result.add_argument("--output-markdown", required=True, type=Path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    baseline = _load(args.baseline)
    post = _load(args.post)
    training = _load(args.training_result)
    for name, result, version in (("baseline", baseline, 0), ("post", post, 1)):
        if (
            result.get("schema_name") != "lightnav_a3_rollout"
            or result.get("status") != "completed"
            or result.get("mode") != "eval"
            or result.get("deterministic") is not True
            or result.get("experiment_split_label") != "train_development"
            or result.get("run_manifest", {}).get("sha256")
            != args.expected_run_manifest_sha256
            or result.get("checkpoint", {}).get("policy_version") != version
        ):
            raise ComparisonError(f"{name} result differs from the matched A3 contract")
    if (
        training.get("schema_name") != "lightnav_a3_rlinf_fsdp_training"
        or training.get("status") != "completed"
        or training.get("run_manifest", {}).get("sha256")
        != args.expected_run_manifest_sha256
        or training.get("final_checkpoint", {}).get("sha256")
        != post.get("checkpoint", {}).get("sha256")
    ):
        raise ComparisonError("post checkpoint is not the completed A3 training output")

    before_rows = baseline.get("episodes")
    after_rows = post.get("episodes")
    if not isinstance(before_rows, list) or not isinstance(after_rows, list) or not before_rows:
        raise ComparisonError("evaluation episode rows are absent")
    before_ids = [(row["scene_id"], row["episode_id"], row["episode_order"]) for row in before_rows]
    after_ids = [(row["scene_id"], row["episode_id"], row["episode_order"]) for row in after_rows]
    if before_ids != after_ids:
        raise ComparisonError("baseline and post-PPO episodes are not the same ordered identities")

    before = _summary(before_rows)
    after = _summary(after_rows)
    deltas = {
        key: (None if before[key] is None or after[key] is None else float(after[key]) - float(before[key]))
        for key in before
        if key != "episode_count"
    }
    paired = []
    for identity, first, second in zip(before_ids, before_rows, after_rows):
        paired.append(
            {
                "scene_id": identity[0],
                "episode_id": identity[1],
                "episode_order": identity[2],
                "success_before": bool(first["success"]),
                "success_after": bool(second["success"]),
                "spl_delta": float(second["spl"]) - float(first["spl"]),
                "final_navigation_error_delta_m": float(second["final_navigation_error_m"])
                - float(first["final_navigation_error_m"]),
                "path_length_delta_m": float(second["path_length_m"])
                - float(first["path_length_m"]),
                "collision_delta": (
                    None
                    if first["collision_count"] is None or second["collision_count"] is None
                    else float(second["collision_count"]) - float(first["collision_count"])
                ),
                "stop_source_before": first["stop_source"],
                "stop_source_after": second["stop_source"],
            }
        )
    output = {
        "schema_name": "lightnav_a3_matched_comparison",
        "schema_version": 1,
        "run_manifest_sha256": args.expected_run_manifest_sha256,
        "baseline_result_sha256": sha256_file(args.baseline),
        "post_result_sha256": sha256_file(args.post),
        "training_result_sha256": sha256_file(args.training_result),
        "before": before,
        "after": after,
        "after_minus_before": deltas,
        "paired_episodes": paired,
        "claim_boundary": (
            "single-seed train-only pilot; native LightNav STOP remains outside actor control"
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    metrics = (
        ("SR", "success_rate"),
        ("SPL", "mean_spl"),
        ("final NE (m)", "mean_final_navigation_error_m"),
        ("path (m)", "mean_path_length_m"),
        ("collisions", "collision_count"),
        ("blocked steps", "blocked_step_count"),
        ("native STOP", "native_stop_count"),
        ("learned STOP", "learned_stop_count"),
        ("timeout", "timeout_count"),
    )
    lines = [
        "# LightNav A3 matched result",
        "",
        "| metric | no-update | PPO | delta |",
        "|---|---:|---:|---:|",
    ]
    for label, key in metrics:
        values = (before[key], after[key], deltas[key])
        cells = ["n/a" if value is None else f"{float(value):.6g}" for value in values]
        lines.append(f"| {label} | {cells[0]} | {cells[1]} | {cells[2]} |")
    lines.extend(
        (
            "",
            "Single-seed R2R train-only pilot. Native LightNav STOP remains outside actor control.",
            "",
        )
    )
    args.output_markdown.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(json.dumps({"output": str(args.output_json), "sha256": sha256_file(args.output_json)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
