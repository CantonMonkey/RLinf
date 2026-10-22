#!/usr/bin/env python3
"""Verify and rebind transferred Navigation-RLT prefix replay paths.

The replay bytes are copied by an external transfer step.  This command does
not copy, modify, or delete replay shards.  It verifies that the destination
tree is an exact, schema-valid image of the source manifest inventory, then
publishes a new manifest whose shard paths are absolute destination paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tempfile
import time
from typing import Mapping, Sequence

import navigation_rlt_replay as replay_module
from navigation_rlt_replay import (
    REPRESENTATION_PREFIX_HIDDEN,
    build_replay_manifest,
    load_episode_shard,
    load_replay_manifest,
    save_replay_manifest,
    sha256_file,
)


SCHEMA_NAME = "navigation_rlt_replay_relocation"
SCHEMA_VERSION = 1
OUTPUT_MANIFEST_NAME = "replay_manifest.json"
RECEIPT_NAME = "relocation_receipt.json"
COMPLETED_NAME = "COMPLETED.json"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
VALIDATION_COMPONENT = re.compile(r"^(?:val|validation)(?:[_-].*)?$", re.IGNORECASE)


class ReplayRelocationError(RuntimeError):
    """Raised when transferred replay cannot be proven content-preserving."""


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ReplayRelocationError("relocation evidence is not canonical JSON") from error


def _payload_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _validate_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ReplayRelocationError(f"{name} must be a lowercase SHA-256")
    return value


def _positive_integer(value: object, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ReplayRelocationError(f"{name} must be a positive integer")
    return value


def _regular_input(path: Path, name: str) -> Path:
    raw = Path(path)
    if raw.is_symlink():
        raise ReplayRelocationError(f"{name} must not be a symbolic link")
    resolved = raw.resolve(strict=True)
    mode = resolved.stat().st_mode
    if not stat.S_ISREG(mode):
        raise ReplayRelocationError(f"{name} must be a regular file")
    return resolved


def _regular_directory(path: Path, name: str) -> Path:
    raw = Path(path)
    if raw.is_symlink():
        raise ReplayRelocationError(f"{name} must not be a symbolic link")
    resolved = raw.resolve(strict=True)
    if not resolved.is_dir():
        raise ReplayRelocationError(f"{name} must be a directory")
    return resolved


def _output_directory(path: Path) -> Path:
    raw = Path(path)
    absolute = raw if raw.is_absolute() else Path.cwd() / raw
    absolute = Path(os.path.abspath(absolute))
    if os.path.lexists(absolute):
        raise FileExistsError(absolute)
    parent = absolute.parent
    if parent.is_symlink():
        raise ReplayRelocationError("output-dir parent must not be a symbolic link")
    parent = parent.resolve(strict=True)
    if not parent.is_dir():
        raise ReplayRelocationError("output-dir parent must be a directory")
    final = parent / absolute.name
    if os.path.lexists(final):
        raise FileExistsError(final)
    return final


def _canonical_source_path(value: object, name: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ReplayRelocationError(f"{name} must be a non-empty POSIX path")
    if "\\" in value or "\x00" in value or not value.startswith("/"):
        raise ReplayRelocationError(f"{name} must be an absolute POSIX path")
    if value.startswith("//"):
        raise ReplayRelocationError(f"{name} must have one POSIX root slash")
    raw_parts = value.split("/")[1:]
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise ReplayRelocationError(f"{name} is not a canonical traversal-free path")
    path = PurePosixPath(value)
    if str(path) != value:
        raise ReplayRelocationError(f"{name} is not canonical")
    return path


def _relative_source_path(
    shard_path: object, source_root: PurePosixPath
) -> PurePosixPath:
    path = _canonical_source_path(shard_path, "source shard path")
    try:
        relative = path.relative_to(source_root)
    except ValueError as error:
        raise ReplayRelocationError(
            "source shard path is outside the declared source-root"
        ) from error
    if relative == PurePosixPath(".") or relative.suffix.lower() != ".npz":
        raise ReplayRelocationError("source shard path must name a nested .npz file")
    for component in relative.parts:
        if VALIDATION_COMPONENT.fullmatch(component):
            raise ReplayRelocationError(
                "source shard path contains a validation split component"
            )
    return relative


def _destination_npz_inventory(root: Path) -> set[Path]:
    inventory: set[Path] = set()
    for current_value, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(current_value)
        for name in directory_names:
            candidate = current / name
            if candidate.is_symlink():
                raise ReplayRelocationError(
                    "destination replay tree contains a symbolic-link directory"
                )
        for name in file_names:
            candidate = current / name
            if candidate.is_symlink():
                raise ReplayRelocationError(
                    "destination replay tree contains a symbolic-link file"
                )
            if candidate.suffix.lower() == ".npz":
                mode = candidate.stat().st_mode
                if not stat.S_ISREG(mode):
                    raise ReplayRelocationError(
                        "destination replay inventory contains a non-regular .npz"
                    )
                inventory.add(candidate.resolve(strict=True))
    return inventory


def _verify_receipt_metadata(
    source_receipt: Mapping[str, object], metadata: Mapping[str, object]
) -> None:
    for name in (
        "run_id",
        "episode_id",
        "reset_token",
        "row_count",
        "source_fingerprint",
        "representation_kind",
        "feature_stage",
        "behavior_policy_id",
        "behavior_policy_version",
    ):
        if metadata.get(name) != source_receipt.get(name):
            raise ReplayRelocationError(
                f"destination shard metadata {name} differs from the source receipt"
            )


def _assert_manifest_rebound_only(
    original: Mapping[str, object], relocated: Mapping[str, object]
) -> None:
    old_common = {key: value for key, value in original.items() if key != "shards"}
    new_common = {key: value for key, value in relocated.items() if key != "shards"}
    if old_common != new_common:
        raise ReplayRelocationError("relocated manifest changed replay-level content")
    old_shards = original["shards"]
    new_shards = relocated["shards"]
    if not isinstance(old_shards, list) or not isinstance(new_shards, list):
        raise ReplayRelocationError("manifest shard inventories are invalid")
    if len(old_shards) != len(new_shards):
        raise ReplayRelocationError("relocated manifest changed shard count")
    for old, new in zip(old_shards, new_shards, strict=True):
        old_content = {key: value for key, value in old.items() if key != "path"}
        new_content = {key: value for key, value in new.items() if key != "path"}
        if old_content != new_content:
            raise ReplayRelocationError("relocation changed a shard receipt")


def _atomic_json_new(path: Path, value: object) -> None:
    if os.path.lexists(path):
        raise FileExistsError(path)
    payload = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".staging-", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if os.path.lexists(path):
            raise FileExistsError(path)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def relocate(args: argparse.Namespace) -> dict[str, object]:
    started = time.time()
    expected_manifest_sha = _validate_sha256(
        args.expected_original_manifest_sha256,
        "expected original manifest SHA-256",
    )
    expected_source = _validate_sha256(
        args.expected_source_fingerprint,
        "expected source fingerprint",
    )
    expected_episodes = _positive_integer(
        args.expected_episode_count,
        "expected episode count",
    )
    if not isinstance(args.expected_run_id, str) or not args.expected_run_id:
        raise ReplayRelocationError("expected run-id must be non-empty")

    source_root = _canonical_source_path(args.source_root, "source-root")
    original_path = _regular_input(args.original_manifest, "original manifest")
    destination_root = _regular_directory(
        args.destination_replay_root, "destination replay root"
    )
    output_dir = _output_directory(args.output_dir)

    original = load_replay_manifest(
        original_path,
        expected_sha256=expected_manifest_sha,
        expected_source_fingerprint=expected_source,
        verify_shards=False,
    )
    if original["run_id"] != args.expected_run_id:
        raise ReplayRelocationError("original manifest run_id differs")
    if original["expected_episode_count"] != expected_episodes:
        raise ReplayRelocationError("original manifest episode budget differs")
    if original["completed_episode_count"] != expected_episodes:
        raise ReplayRelocationError("original replay is incomplete")
    if original["representation_kind"] != REPRESENTATION_PREFIX_HIDDEN:
        raise ReplayRelocationError("original replay is not prefix_hidden")
    if original["split_label"] != "train_update" or original["validation_split_reads"] != 0:
        raise ReplayRelocationError("original replay records validation split access")

    planned: list[tuple[Mapping[str, object], PurePosixPath, Path]] = []
    source_paths: set[str] = set()
    destination_paths: set[Path] = set()
    content_hashes: set[str] = set()
    for receipt_value in original["shards"]:
        if not isinstance(receipt_value, Mapping):
            raise ReplayRelocationError("original shard receipt is not an object")
        relative = _relative_source_path(receipt_value.get("path"), source_root)
        source_path = str(receipt_value["path"])
        if source_path in source_paths:
            raise ReplayRelocationError("original manifest repeats a shard path")
        source_paths.add(source_path)
        digest = _validate_sha256(receipt_value.get("sha256"), "shard SHA-256")
        if digest in content_hashes:
            raise ReplayRelocationError("original manifest repeats a shard content hash")
        content_hashes.add(digest)
        destination = destination_root.joinpath(*relative.parts)
        if destination in destination_paths:
            raise ReplayRelocationError("destination path mapping is not one-to-one")
        destination_paths.add(destination)
        planned.append((receipt_value, relative, destination))

    observed_inventory = _destination_npz_inventory(destination_root)
    expected_inventory = {path for _, _, path in planned}
    missing = sorted(str(path) for path in expected_inventory - observed_inventory)
    extra = sorted(str(path) for path in observed_inventory - expected_inventory)
    if missing or extra:
        raise ReplayRelocationError(
            f"destination .npz inventory differs; missing={missing}, extra={extra}"
        )

    relocated_receipts: list[dict[str, object]] = []
    inventory_rows: list[dict[str, object]] = []
    for source_receipt, relative, destination in planned:
        if destination.is_symlink():
            raise ReplayRelocationError("destination shard must not be a symbolic link")
        resolved = destination.resolve(strict=True)
        try:
            resolved.relative_to(destination_root)
        except ValueError as error:
            raise ReplayRelocationError("destination shard escapes replay root") from error
        if not stat.S_ISREG(resolved.stat().st_mode):
            raise ReplayRelocationError("destination shard must be a regular file")
        expected_size = _positive_integer(
            source_receipt.get("size_bytes"), "source shard size_bytes"
        )
        if resolved.stat().st_size != expected_size:
            raise ReplayRelocationError("destination shard byte size differs")
        expected_sha = str(source_receipt["sha256"])
        if sha256_file(resolved) != expected_sha:
            raise ReplayRelocationError("destination shard SHA-256 differs")
        _, metadata = load_episode_shard(
            resolved,
            expected_sha256=expected_sha,
            expected_source_fingerprint=expected_source,
            expected_representation_kind=REPRESENTATION_PREFIX_HIDDEN,
        )
        _verify_receipt_metadata(source_receipt, metadata)
        relocated = dict(source_receipt)
        relocated["path"] = str(resolved)
        relocated_receipts.append(relocated)
        inventory_rows.append(
            {
                "relative_path": relative.as_posix(),
                "sha256": expected_sha,
                "size_bytes": expected_size,
                "episode_id": source_receipt["episode_id"],
                "reset_token": source_receipt["reset_token"],
                "row_count": source_receipt["row_count"],
            }
        )

    relocated_manifest = build_replay_manifest(
        relocated_receipts,
        run_id=args.expected_run_id,
        expected_episode_count=expected_episodes,
        expected_source_fingerprint=expected_source,
    )
    _assert_manifest_rebound_only(original, relocated_manifest)
    inventory_rows.sort(key=lambda row: str(row["relative_path"]))
    inventory_fingerprint = _payload_sha256(inventory_rows)

    output_dir.mkdir()
    manifest_path = output_dir / OUTPUT_MANIFEST_NAME
    manifest_receipt = save_replay_manifest(manifest_path, relocated_manifest)
    new_manifest_sha = str(manifest_receipt["sha256"])
    verified = load_replay_manifest(
        manifest_path,
        expected_sha256=new_manifest_sha,
        expected_source_fingerprint=expected_source,
        verify_shards=True,
    )
    _assert_manifest_rebound_only(original, verified)

    receipt = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "run_id": args.expected_run_id,
        "source_fingerprint": expected_source,
        "representation_kind": REPRESENTATION_PREFIX_HIDDEN,
        "episode_count": expected_episodes,
        "transition_count": verified["transition_count"],
        "original_manifest": {
            "path": str(original_path),
            "sha256": expected_manifest_sha,
            "inaccessible_source_root": str(source_root),
        },
        "destination": {
            "replay_root": str(destination_root),
            "manifest_path": str(manifest_path),
            "manifest_sha256": new_manifest_sha,
        },
        "inventory": {
            "fingerprint_sha256": inventory_fingerprint,
            "shard_count": len(inventory_rows),
            "total_size_bytes": sum(int(row["size_bytes"]) for row in inventory_rows),
            "rows": inventory_rows,
        },
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
        "started_unix_s": started,
        "completed_unix_s": time.time(),
    }
    receipt_path = output_dir / RECEIPT_NAME
    _atomic_json_new(receipt_path, receipt)
    receipt_sha = sha256_file(receipt_path)
    completion_path = output_dir / COMPLETED_NAME
    _atomic_json_new(
        completion_path,
        {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "run_id": args.expected_run_id,
            "original_manifest_sha256": expected_manifest_sha,
            "relocated_manifest_sha256": new_manifest_sha,
            "inventory_fingerprint_sha256": inventory_fingerprint,
            "relocation_receipt_sha256": receipt_sha,
        },
    )
    return {
        "status": "completed",
        "output_dir": str(output_dir),
        "relocated_manifest": str(manifest_path),
        "relocated_manifest_sha256": new_manifest_sha,
        "relocation_receipt": str(receipt_path),
        "relocation_receipt_sha256": receipt_sha,
        "inventory_fingerprint_sha256": inventory_fingerprint,
        "episode_count": expected_episodes,
        "transition_count": verified["transition_count"],
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--original-manifest", required=True, type=Path)
    result.add_argument("--expected-original-manifest-sha256", required=True)
    result.add_argument(
        "--source-root",
        required=True,
        help="Inaccessible source-host POSIX root used only for lexical path mapping",
    )
    result.add_argument("--destination-replay-root", required=True, type=Path)
    result.add_argument("--expected-run-id", required=True)
    result.add_argument("--expected-source-fingerprint", required=True)
    result.add_argument("--expected-episode-count", required=True, type=int)
    result.add_argument("--output-dir", required=True, type=Path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    result = relocate(args)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        ReplayRelocationError,
        replay_module.ReplayContractError,
        FileExistsError,
        OSError,
    ) as error:
        print(
            json.dumps(
                {
                    "schema_name": SCHEMA_NAME,
                    "schema_version": SCHEMA_VERSION,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                sort_keys=True,
                allow_nan=False,
            ),
            file=sys.stderr,
        )
        raise SystemExit(2)
