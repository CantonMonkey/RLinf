"""Merge scene-isolated Habitat-Lab PointNav JSONL segments and audit metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output in args.parts:
        parser.error("output must differ from every part")

    expected_index = 0
    run_options = None
    outcomes = []
    part_evidence = []
    first_three = bytearray()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp_output = args.output.with_name(args.output.name + ".tmp")
    with temp_output.open("wb") as merged:
        for part in args.parts:
            summary_path = part.with_suffix(".summary.json")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            digest = sha256(part)
            if summary["trace_sha256"] != digest:
                raise AssertionError(f"Segment checksum mismatch: {part}")
            options = (
                summary["seed"],
                summary["max_steps"],
                summary["min_moves"],
                summary["stop_probability"],
            )
            if run_options is None:
                run_options = options
            elif run_options != options:
                raise AssertionError("Inconsistent rollout parameters")
            if summary["episode_start"] != expected_index:
                raise AssertionError("Segments are not contiguous or in index order")
            part_count = 0
            with part.open("rb") as source:
                for line in source:
                    record = json.loads(line)
                    if record["episode_index"] != expected_index:
                        raise AssertionError("Episode index mismatch")
                    if record["scene_id"] != summary["scene_id"]:
                        raise AssertionError("Scene mismatch")
                    if [step["action"] for step in record["steps"]] != record["actions"]:
                        raise AssertionError("Action trace mismatch")
                    if len(record["steps"]) != record["outcome"]["steps"]:
                        raise AssertionError("Step count mismatch")
                    if expected_index < 3:
                        first_three.extend(line)
                    outcomes.append(record["outcome"])
                    merged.write(line)
                    expected_index += 1
                    part_count += 1
            if part_count != summary["episodes"]:
                raise AssertionError("Segment episode count mismatch")
            part_evidence.append(
                {
                    "path": str(part),
                    "sha256": digest,
                    "episodes": part_count,
                    "scene_id": summary["scene_id"],
                }
            )
    temp_output.replace(args.output)
    if not outcomes:
        raise AssertionError("No episodes")
    summary = {
        "status": "ok",
        "dataset": "official habitat-test-scenes PointNav v1 val",
        "simulator": "Habitat-Lab 0.2.4 / Habitat-Sim 0.2.4",
        "seed": run_options[0],
        "max_steps": run_options[1],
        "min_moves": run_options[2],
        "stop_probability": run_options[3],
        "episodes": len(outcomes),
        "success_count": sum(int(r["success"]) for r in outcomes),
        "mean_spl": sum(r["spl"] for r in outcomes) / len(outcomes),
        "mean_final_distance": sum(r["distance_to_goal"] for r in outcomes)
        / len(outcomes),
        "total_steps": sum(r["steps"] for r in outcomes),
        "total_official_collisions": sum(r["collision_count_official"] for r in outcomes),
        "total_move_forward_collisions": sum(
            r["collision_count_move_forward"] for r in outcomes
        ),
        "stop_count": sum(int(r["stop_called"]) for r in outcomes),
        "timeout_count": sum(int(r["timeout"]) for r in outcomes),
        "parts": part_evidence,
        "first3_sha256": hashlib.sha256(first_three).hexdigest(),
        "trace_sha256": sha256(args.output),
        "trace_path": str(args.output),
        "renderer_created": False,
        "visual_habitat_validated": False,
        "scene_transition_workaround": (
            "One no-renderer Habitat Env per scene/process; this old native "
            "version segfaulted at the val scene transition after episode 49."
        ),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
