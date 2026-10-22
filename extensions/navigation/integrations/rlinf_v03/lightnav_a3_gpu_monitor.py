#!/usr/bin/env python3
"""Sample and summarize eight-GPU utilization for the bounded A3 train window."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Sequence


FIELDS = (
    "index",
    "uuid",
    "memory.used",
    "memory.total",
    "utilization.gpu",
    "power.draw",
)


def _query() -> list[dict[str, object]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=" + ",".join(FIELDS),
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=20,
    )
    rows = []
    for values in csv.reader(completed.stdout.splitlines(), skipinitialspace=True):
        if len(values) != len(FIELDS):
            raise RuntimeError("nvidia-smi GPU row differs from A3 telemetry schema")
        rows.append(
            {
                "index": int(values[0]),
                "uuid": values[1],
                "memory_used_mib": float(values[2]),
                "memory_total_mib": float(values[3]),
                "gpu_util_percent": float(values[4]),
                "power_w": float(values[5]),
            }
        )
    if len(rows) != 8 or len({row["uuid"] for row in rows}) != 8:
        raise RuntimeError("A3 telemetry requires eight unique GPUs")
    return rows


def sample(output: Path, stop_file: Path, interval: float) -> int:
    if interval < 0.5:
        raise ValueError("telemetry interval must be at least 0.5 seconds")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n", buffering=1) as stream:
        while not stop_file.exists():
            timestamp = time.time()
            for row in _query():
                row["unix_s"] = timestamp
                stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            time.sleep(interval)
    return 0


def summarize(source: Path, output: Path) -> int:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    with source.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            grouped[str(row["uuid"])].append(row)
    if len(grouped) != 8:
        raise RuntimeError("telemetry summary lacks eight GPU UUIDs")
    gpus = []
    for uuid, rows in sorted(grouped.items(), key=lambda item: int(item[1][0]["index"])):
        utilization = [float(row["gpu_util_percent"]) for row in rows]
        memory = [float(row["memory_used_mib"]) for row in rows]
        power = [float(row["power_w"]) for row in rows]
        if not all(math.isfinite(value) for value in (*utilization, *memory, *power)):
            raise RuntimeError("telemetry contains non-finite values")
        gpus.append(
            {
                "index": int(rows[0]["index"]),
                "uuid": uuid,
                "sample_count": len(rows),
                "memory_total_mib": float(rows[0]["memory_total_mib"]),
                "memory_used_mean_mib": sum(memory) / len(memory),
                "memory_used_max_mib": max(memory),
                "gpu_util_mean_percent": sum(utilization) / len(utilization),
                "gpu_util_max_percent": max(utilization),
                "power_mean_w": sum(power) / len(power),
                "power_max_w": max(power),
            }
        )
    summary = {
        "schema_name": "lightnav_a3_gpu_telemetry",
        "schema_version": 1,
        "gpu_count": len(gpus),
        "sample_interval_s": (
            None
            if len(next(iter(grouped.values()))) < 2
            else float(next(iter(grouped.values()))[1]["unix_s"])
            - float(next(iter(grouped.values()))[0]["unix_s"])
        ),
        "gpus": gpus,
    }
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    sample_parser = sub.add_parser("sample")
    sample_parser.add_argument("--output", required=True, type=Path)
    sample_parser.add_argument("--stop-file", required=True, type=Path)
    sample_parser.add_argument("--interval", type=float, default=1.0)
    summary_parser = sub.add_parser("summarize")
    summary_parser.add_argument("--source", required=True, type=Path)
    summary_parser.add_argument("--output", required=True, type=Path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "sample":
        return sample(args.output, args.stop_file, args.interval)
    return summarize(args.source, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
