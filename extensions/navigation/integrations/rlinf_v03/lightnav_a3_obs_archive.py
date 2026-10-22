#!/usr/bin/env python3
"""Copy bounded LightNav A3 artifacts to the mounted OBS bucket and read them back."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Sequence


MAX_CHECKPOINT_BYTES = 2 * 1024**3
MAX_RUN_BYTES = 16 * 1024**3


class ArchiveError(RuntimeError):
    """Raised when the Turbo-to-OBS archive boundary is violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _contained(path: Path, root: Path) -> Path:
    resolved = path.resolve(strict=True)
    root_resolved = root.resolve(strict=True)
    try:
        resolved.relative_to(root_resolved)
    except ValueError as error:
        raise ArchiveError(f"{resolved} is outside {root_resolved}") from error
    return resolved


def _atomic_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".staging")
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    staging.write_bytes(payload)
    staging.replace(path)
    return hashlib.sha256(payload).hexdigest()


def _inventory(root: Path) -> dict[str, dict[str, object]]:
    return {
        str(path.relative_to(root)): {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _copy_verified(source: Path, destination: Path) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_hash = sha256_file(source)
    if destination.exists():
        if not destination.is_file() or sha256_file(destination) != source_hash:
            raise ArchiveError(f"OBS destination conflicts with {source.name}")
    else:
        temporary = destination.with_name(destination.name + ".partial")
        if temporary.exists():
            raise ArchiveError(f"stale partial OBS object exists: {temporary}")
        shutil.copy2(source, temporary)
        if sha256_file(temporary) != source_hash:
            raise ArchiveError(f"OBS partial readback differs for {source.name}")
        temporary.replace(destination)
    readback_hash = sha256_file(destination)
    if readback_hash != source_hash:
        raise ArchiveError(f"OBS readback differs for {source.name}")
    return {
        "source": str(source),
        "destination": str(destination),
        "size_bytes": source.stat().st_size,
        "sha256": source_hash,
        "readback_sha256": readback_hash,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run-root", required=True, type=Path)
    result.add_argument("--run-id", required=True)
    result.add_argument("--phase", required=True, choices=("checkpoint", "final"))
    result.add_argument(
        "--obs-root", type=Path, default=Path("/obs/navigation-example")
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not args.run_id or "/" in args.run_id or "\\" in args.run_id:
        raise ArchiveError("run-id must be one path-safe component")
    run_root = _contained(args.run_root, Path("/home/navcloud/work"))
    obs_root = _contained(args.obs_root, Path("/obs/navigation-example"))
    local_inventory = _inventory(run_root)
    total_bytes = sum(int(row["size_bytes"]) for row in local_inventory.values())
    if total_bytes > MAX_RUN_BYTES:
        raise ArchiveError("A3 Turbo run exceeds the 16 GiB hard cap")

    if args.phase == "checkpoint":
        relative_files = [
            Path("training/checkpoints/latest/lightnav_a3_policy.npz"),
            Path("training/training_result.json"),
            Path("training/artifact_inventory.json"),
            Path("training/COMPLETED.json"),
        ]
    else:
        relative_files = [
            Path(relative)
            for relative in sorted(local_inventory)
            if not relative.endswith(".partial") and not relative.endswith(".staging")
        ]

    records = []
    for relative in relative_files:
        source = (run_root / relative).resolve(strict=True)
        if not source.is_file() or source.is_symlink():
            raise ArchiveError(f"archive source is not a regular file: {relative}")
        if "checkpoints" in relative.parts and source.stat().st_size > MAX_CHECKPOINT_BYTES:
            raise ArchiveError("A3 checkpoint exceeds the 2 GiB hard cap")
        if relative.parts[:2] == ("training", "checkpoints"):
            destination = (
                obs_root
                / "checkpoints"
                / "lightnav-a3"
                / args.run_id
                / Path(*relative.parts[2:])
            )
        else:
            destination = obs_root / "results" / "lightnav-a3" / args.run_id / relative
        records.append(_copy_verified(source, destination))

    manifest = {
        "schema_name": "lightnav_a3_obs_archive",
        "schema_version": 1,
        "status": "completed",
        "phase": args.phase,
        "run_id": args.run_id,
        "source_run_root": str(run_root),
        "source_total_bytes": total_bytes,
        "checkpoint_hard_cap_bytes": MAX_CHECKPOINT_BYTES,
        "run_hard_cap_bytes": MAX_RUN_BYTES,
        "files": records,
        "completed_unix_s": time.time(),
    }
    manifest_path = obs_root / "manifests" / "lightnav-a3" / f"{args.run_id}.{args.phase}.json"
    _atomic_json(manifest_path, manifest)
    manifest_hash = sha256_file(manifest_path)
    completion_path = (
        obs_root
        / "results"
        / "lightnav-a3"
        / args.run_id
        / ("CHECKPOINT_COMPLETE" if args.phase == "checkpoint" else "COMPLETE")
    )
    expected_completion_hash = _atomic_json(
        completion_path,
        {
            "schema_name": "lightnav_a3_obs_completion",
            "schema_version": 1,
            "phase": args.phase,
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_hash,
            "completed_unix_s": time.time(),
        },
    )
    if sha256_file(completion_path) != expected_completion_hash:
        raise ArchiveError("OBS completion readback differs")
    print(json.dumps({"manifest": str(manifest_path), "sha256": manifest_hash}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
