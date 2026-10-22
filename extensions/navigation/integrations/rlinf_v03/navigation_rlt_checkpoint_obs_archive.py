#!/usr/bin/env python3
"""Append or verify one immutable Navigation-RLT checkpoint OBS attempt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Mapping, Sequence
import uuid


OBS_MOUNT = Path("/obs")
DEFAULT_OBS_ROOT = OBS_MOUNT / "navigation-example"
EXPECTED_OBS_MOUNT_SOURCE = "s3fs[/navigation]"
EXPECTED_OBS_FILESYSTEM = "fuse.s3fs"
# Early per-stage attempts stay separate from the final canonical archive root.
CHECKPOINT_NAMESPACE = Path("checkpoints/navigation-rlt-stage-archives")
ARCHIVE_SCHEMA = "navigation_rlt_checkpoint_obs_archive"
COMPLETION_SCHEMA = "navigation_rlt_checkpoint_obs_completion"
SCHEMA_VERSION = 1
MAX_CHECKPOINT_BYTES = 2 * 1024**3
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
ATTEMPT_ID_PATTERN = re.compile(r"attempt-[0-9a-f]{32}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
DIGEST_NAMESPACE = "sha256"
PHASES = ("stage1", "stage2")
CHECKPOINT_NAME = "checkpoint.pt"
MANIFEST_NAME = "manifest.json"
COMPLETION_NAME = "COMPLETED.json"


class CheckpointArchiveError(RuntimeError):
    """Raised when a per-phase checkpoint archive is unsafe or inconsistent."""


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_json(path: Path, name: str) -> dict[str, object]:
    def reject_constant(value: str) -> None:
        raise CheckpointArchiveError(
            f"{name} contains forbidden non-finite constant {value}"
        )

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=reject_constant
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CheckpointArchiveError(f"{name} is not strict JSON") from error
    if not isinstance(value, dict):
        raise CheckpointArchiveError(f"{name} must contain an object")
    return value


def _json_bytes(value: Mapping[str, object]) -> bytes:
    try:
        text = json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise CheckpointArchiveError("archive output is not strict JSON") from error
    return (text + "\n").encode("utf-8")


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise CheckpointArchiveError(f"{name} must be a lowercase SHA-256")
    return value


def _positive_size(value: object, name: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 1
        or value > MAX_CHECKPOINT_BYTES
    ):
        raise CheckpointArchiveError(f"{name} is outside the checkpoint size limit")
    return value


def _regular_file(path: Path, name: str) -> Path:
    raw = Path(path).expanduser().absolute()
    if raw.is_symlink():
        raise CheckpointArchiveError(f"{name} must not be a symlink")
    try:
        resolved = raw.resolve(strict=True)
        status = resolved.lstat()
    except OSError as error:
        raise CheckpointArchiveError(f"{name} is missing") from error
    if resolved != raw:
        raise CheckpointArchiveError(f"{name} must not traverse a symlink")
    if not stat.S_ISREG(status.st_mode):
        raise CheckpointArchiveError(f"{name} must be a regular file")
    return resolved


def _safe_existing_directory(path: Path, name: str) -> Path:
    raw = Path(path).expanduser().absolute()
    if raw.is_symlink():
        raise CheckpointArchiveError(f"{name} must not be a symlink")
    try:
        resolved = raw.resolve(strict=True)
        status = resolved.lstat()
    except OSError as error:
        raise CheckpointArchiveError(f"{name} is missing") from error
    if resolved != raw:
        raise CheckpointArchiveError(f"{name} must not traverse a symlink")
    if not stat.S_ISDIR(status.st_mode):
        raise CheckpointArchiveError(f"{name} must be a directory")
    return resolved


def _verify_obs_mount(destination: Path) -> None:
    environment = dict(os.environ)
    environment["LC_ALL"] = "C"
    destination_text = destination.as_posix()
    try:
        completed = subprocess.run(
            [
                "findmnt",
                "-T",
                destination_text,
                "-n",
                "-o",
                "TARGET,SOURCE,FSTYPE,OPTIONS",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            env=environment,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise CheckpointArchiveError("cannot establish the live OBS mount") from error
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if completed.returncode != 0 or len(lines) != 1:
        raise CheckpointArchiveError("findmnt did not identify exactly one OBS mount")
    fields = lines[0].split()
    if len(fields) != 4:
        raise CheckpointArchiveError("findmnt returned an unexpected OBS mount record")
    target, source, filesystem, options_text = fields
    options = frozenset(options_text.split(","))
    if target != OBS_MOUNT.as_posix():
        raise CheckpointArchiveError(
            "the destination longest-match mountpoint is not exactly /obs"
        )
    if source != EXPECTED_OBS_MOUNT_SOURCE:
        raise CheckpointArchiveError("the OBS mount source/bucket differs")
    if filesystem != EXPECTED_OBS_FILESYSTEM:
        raise CheckpointArchiveError("the OBS filesystem is not fuse.s3fs")
    if "rw" not in options or "ro" in options:
        raise CheckpointArchiveError("the OBS mount is not read-write")


def _canonical_obs_root(value: Path) -> Path:
    raw = Path(value).expanduser().absolute()
    expected_raw = DEFAULT_OBS_ROOT.expanduser().absolute()
    if raw != expected_raw:
        raise CheckpointArchiveError(
            f"OBS root must be the canonical path {expected_raw}"
        )
    _verify_obs_mount(raw)
    mount = _safe_existing_directory(OBS_MOUNT, "OBS mount")
    root = _safe_existing_directory(raw, "OBS root")
    expected = _safe_existing_directory(expected_raw, "canonical OBS root")
    if root != expected:
        raise CheckpointArchiveError("OBS root resolves away from its canonical path")
    try:
        root.relative_to(mount)
    except ValueError as error:
        raise CheckpointArchiveError("OBS root escapes the OBS mount") from error
    return root


def _ensure_directory(parent: Path, child: Path, name: str) -> Path:
    try:
        child.relative_to(parent)
    except ValueError as error:
        raise CheckpointArchiveError(f"{name} escapes its fixed parent") from error
    if _lexists(child):
        if child.is_symlink() or not child.is_dir():
            raise CheckpointArchiveError(f"{name} is not a safe directory")
        resolved = child.resolve(strict=True)
        if resolved != child.absolute():
            raise CheckpointArchiveError(f"{name} must not traverse a symlink")
        try:
            resolved.relative_to(parent)
        except ValueError as error:
            raise CheckpointArchiveError(f"{name} resolves outside its parent") from error
        return resolved
    try:
        child.mkdir(mode=0o700)
    except FileExistsError:
        return _ensure_directory(parent, child, name)
    return child.resolve(strict=True)


def _existing_phase_root(root: Path, run_id: str, phase: str) -> Path:
    checkpoint_root = _safe_existing_directory(root / "checkpoints", "checkpoint root")
    namespace = _safe_existing_directory(
        checkpoint_root / "navigation-rlt-stage-archives",
        "Navigation-RLT stage-checkpoint namespace",
    )
    run_root = _safe_existing_directory(namespace / run_id, "run archive root")
    return _safe_existing_directory(run_root / phase, "phase archive root")


def _create_attempt(
    phase_root: Path, checkpoint_sha256: str
) -> tuple[str, Path]:
    digest_namespace = _ensure_directory(
        phase_root,
        phase_root / DIGEST_NAMESPACE,
        "SHA-256 attempt namespace",
    )
    digest_root = _ensure_directory(
        digest_namespace,
        digest_namespace / checkpoint_sha256,
        "checkpoint digest root",
    )
    for _ in range(64):
        attempt_id = f"attempt-{uuid.uuid4().hex}"
        attempt = digest_root / attempt_id
        try:
            attempt.mkdir(mode=0o700)
        except FileExistsError:
            continue
        return attempt_id, attempt.resolve(strict=True)
    raise CheckpointArchiveError("could not allocate a unique archive attempt")


def _write_new_payload(destination: Path, payload: bytes, name: str) -> None:
    if _lexists(destination):
        raise CheckpointArchiveError(f"unique {name} already exists")
    try:
        descriptor = os.open(
            destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
    except FileExistsError as error:
        raise CheckpointArchiveError(f"unique {name} already exists") from error
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    published = _regular_file(destination, name)
    if published.read_bytes() != payload:
        raise CheckpointArchiveError(f"{name} readback differs")


def _copy_new_checkpoint(
    source: Path, destination: Path, expected_sha256: str, expected_size: int
) -> None:
    if _lexists(destination):
        raise CheckpointArchiveError("unique checkpoint object already exists")
    source_before = source.stat()
    digest = hashlib.sha256()
    copied = 0
    try:
        descriptor = os.open(
            destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
    except FileExistsError as error:
        raise CheckpointArchiveError("unique checkpoint object already exists") from error
    try:
        with source.open("rb") as input_stream, os.fdopen(
            descriptor, "wb", closefd=True
        ) as output_stream:
            while True:
                block = input_stream.read(1024 * 1024)
                if not block:
                    break
                copied += len(block)
                if copied > MAX_CHECKPOINT_BYTES:
                    raise CheckpointArchiveError("checkpoint exceeds the 2 GiB hard cap")
                digest.update(block)
                output_stream.write(block)
            output_stream.flush()
            os.fsync(output_stream.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    source_after = source.stat()
    before = (
        source_before.st_dev,
        source_before.st_ino,
        source_before.st_size,
        source_before.st_mtime_ns,
    )
    after = (
        source_after.st_dev,
        source_after.st_ino,
        source_after.st_size,
        source_after.st_mtime_ns,
    )
    if before != after:
        raise CheckpointArchiveError("checkpoint changed while it was copied")
    if copied != expected_size or digest.hexdigest() != expected_sha256:
        raise CheckpointArchiveError("copied checkpoint SHA-256 or size differs")
    archived = _regular_file(destination, "archived checkpoint")
    if archived.stat().st_size != expected_size or sha256_file(archived) != expected_sha256:
        raise CheckpointArchiveError("archived checkpoint readback differs")


def _expected_manifest(
    *,
    run_id: str,
    phase: str,
    attempt_id: str,
    attempt: Path,
    source: Path,
    source_size: int,
    expected_sha256: str,
    archived: Path,
) -> dict[str, object]:
    return {
        "schema_name": ARCHIVE_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "readback_verified",
        "run_id": run_id,
        "phase": phase,
        "attempt_id": attempt_id,
        "attempt_root": str(attempt),
        "source_checkpoint": {
            "path": str(source),
            "sha256": expected_sha256,
            "size_bytes": source_size,
        },
        "archived_checkpoint": {
            "path": str(archived),
            "sha256": expected_sha256,
            "size_bytes": source_size,
            "readback_sha256": expected_sha256,
            "content_address": {
                "algorithm": "sha256",
                "digest": expected_sha256,
            },
        },
        "checkpoint_hard_cap_bytes": MAX_CHECKPOINT_BYTES,
    }


def _expected_completion(
    *,
    run_id: str,
    phase: str,
    attempt_id: str,
    attempt: Path,
    phase_root: Path,
    archived: Path,
    checkpoint_sha256: str,
    checkpoint_size: int,
    manifest_payload: bytes,
) -> dict[str, object]:
    manifest = attempt / MANIFEST_NAME
    return {
        "schema_name": COMPLETION_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "run_id": run_id,
        "phase": phase,
        "attempt_id": attempt_id,
        "attempt_root": str(attempt),
        "phase_root": str(phase_root),
        "archive_manifest": {
            "path": str(manifest),
            "sha256": hashlib.sha256(manifest_payload).hexdigest(),
            "size_bytes": len(manifest_payload),
        },
        "checkpoint": {
            "path": str(archived),
            "sha256": checkpoint_sha256,
            "size_bytes": checkpoint_size,
        },
    }


def _bound_regular_file(
    value: object, expected: Path, attempt: Path, name: str
) -> Path:
    if not isinstance(value, str):
        raise CheckpointArchiveError(f"{name} path is not a string")
    path = _regular_file(Path(value), name)
    try:
        path.relative_to(attempt)
    except ValueError as error:
        raise CheckpointArchiveError(f"{name} escapes its attempt") from error
    if path != expected:
        raise CheckpointArchiveError(f"{name} path differs")
    return path


def _completion_identity_exact(
    completion: Mapping[str, object],
    *,
    run_id: str,
    phase: str,
    attempt_id: str,
    attempt: Path,
    phase_root: Path,
) -> bool:
    return (
        completion.get("schema_name") == COMPLETION_SCHEMA
        and completion.get("schema_version") == SCHEMA_VERSION
        and completion.get("status") == "completed"
        and completion.get("run_id") == run_id
        and completion.get("phase") == phase
        and completion.get("attempt_id") == attempt_id
        and completion.get("attempt_root") == str(attempt)
        and completion.get("phase_root") == str(phase_root)
    )


def _identified_completion(
    attempt: Path, *, run_id: str, phase: str, phase_root: Path
) -> tuple[Path, dict[str, object]] | None:
    attempt_id = attempt.name
    if ATTEMPT_ID_PATTERN.fullmatch(attempt_id) is None:
        return None
    if attempt.is_symlink() or not attempt.is_dir():
        return None
    try:
        resolved = attempt.resolve(strict=True)
    except OSError:
        return None
    if resolved != attempt.absolute():
        return None
    completion_path = attempt / COMPLETION_NAME
    if (
        not _lexists(completion_path)
        or completion_path.is_symlink()
        or not completion_path.is_file()
    ):
        return None
    try:
        completion = _strict_json(completion_path, "completion marker")
    except (CheckpointArchiveError, OSError):
        return None
    if not _completion_identity_exact(
        completion,
        run_id=run_id,
        phase=phase,
        attempt_id=attempt_id,
        attempt=attempt,
        phase_root=phase_root,
    ):
        return None
    return completion_path, completion


def _validate_identified_completion_strict(
    attempt: Path,
    *,
    digest_sha256: str,
    run_id: str,
    phase: str,
    phase_root: Path,
    completion_path: Path,
    completion: Mapping[str, object],
) -> dict[str, object]:
    attempt_id = attempt.name
    if {entry.name for entry in attempt.iterdir()} != {
        CHECKPOINT_NAME,
        MANIFEST_NAME,
        COMPLETION_NAME,
    }:
        raise CheckpointArchiveError("complete attempt inventory differs")
    completion_path = _regular_file(completion_path, "completion marker")

    checkpoint_binding = completion.get("checkpoint")
    manifest_binding = completion.get("archive_manifest")
    if not isinstance(checkpoint_binding, dict) or not isinstance(manifest_binding, dict):
        raise CheckpointArchiveError("completion marker bindings differ")
    checkpoint_sha256 = _sha256(
        checkpoint_binding.get("sha256"), "completion checkpoint SHA-256"
    )
    checkpoint_size = _positive_size(
        checkpoint_binding.get("size_bytes"), "completion checkpoint size"
    )
    if checkpoint_sha256 != digest_sha256:
        raise CheckpointArchiveError(
            "completed checkpoint does not match its SHA-256 branch"
        )
    checkpoint_path = _bound_regular_file(
        checkpoint_binding.get("path"),
        attempt / CHECKPOINT_NAME,
        attempt,
        "archived checkpoint",
    )
    if (
        checkpoint_path.stat().st_size != checkpoint_size
        or sha256_file(checkpoint_path) != checkpoint_sha256
    ):
        raise CheckpointArchiveError("archived checkpoint readback differs")

    manifest_path = _bound_regular_file(
        manifest_binding.get("path"),
        attempt / MANIFEST_NAME,
        attempt,
        "archive manifest",
    )
    manifest_payload = manifest_path.read_bytes()
    if (
        manifest_binding.get("size_bytes") != len(manifest_payload)
        or manifest_binding.get("sha256")
        != hashlib.sha256(manifest_payload).hexdigest()
    ):
        raise CheckpointArchiveError("archive manifest hash binding differs")
    manifest = _strict_json(manifest_path, "archive manifest")
    source_binding = manifest.get("source_checkpoint")
    archived_binding = manifest.get("archived_checkpoint")
    if not isinstance(source_binding, dict) or not isinstance(archived_binding, dict):
        raise CheckpointArchiveError("archive manifest bindings differ")
    if (
        manifest.get("schema_name") != ARCHIVE_SCHEMA
        or manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "readback_verified"
        or manifest.get("run_id") != run_id
        or manifest.get("phase") != phase
        or manifest.get("attempt_id") != attempt_id
        or manifest.get("attempt_root") != str(attempt)
        or manifest.get("checkpoint_hard_cap_bytes") != MAX_CHECKPOINT_BYTES
        or not isinstance(source_binding.get("path"), str)
        or not source_binding.get("path")
        or source_binding.get("sha256") != checkpoint_sha256
        or source_binding.get("size_bytes") != checkpoint_size
        or archived_binding.get("path") != str(checkpoint_path)
        or archived_binding.get("sha256") != checkpoint_sha256
        or archived_binding.get("readback_sha256") != checkpoint_sha256
        or archived_binding.get("size_bytes") != checkpoint_size
        or archived_binding.get("content_address")
        != {"algorithm": "sha256", "digest": checkpoint_sha256}
    ):
        raise CheckpointArchiveError("archive manifest identity or binding differs")

    return {
        "attempt_id": attempt_id,
        "attempt_root": str(attempt),
        "phase_root": str(phase_root),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
            "size_bytes": checkpoint_size,
        },
        "manifest": {
            "path": str(manifest_path),
            "sha256": hashlib.sha256(manifest_payload).hexdigest(),
            "size_bytes": len(manifest_payload),
        },
        "completion_marker": {
            "path": str(completion_path),
            "sha256": sha256_file(completion_path),
            "size_bytes": completion_path.stat().st_size,
        },
    }


def _scan_complete_attempts(
    phase_root: Path, *, run_id: str, phase: str
) -> list[dict[str, object]]:
    completed: list[dict[str, object]] = []
    digest_namespace = phase_root / DIGEST_NAMESPACE
    if not _lexists(digest_namespace):
        return completed
    digest_namespace = _safe_existing_directory(
        digest_namespace, "SHA-256 attempt namespace"
    )
    for digest_root in sorted(digest_namespace.iterdir(), key=lambda item: item.name):
        if SHA256_PATTERN.fullmatch(digest_root.name) is None:
            continue
        if digest_root.is_symlink() or not digest_root.is_dir():
            continue
        resolved_digest = digest_root.resolve(strict=True)
        if resolved_digest != digest_root.absolute():
            continue
        for attempt in sorted(digest_root.iterdir(), key=lambda item: item.name):
            identified = _identified_completion(
                attempt, run_id=run_id, phase=phase, phase_root=phase_root
            )
            if identified is None:
                continue
            completion_path, completion = identified
            completed.append(
                _validate_identified_completion_strict(
                    attempt,
                    digest_sha256=digest_root.name,
                    run_id=run_id,
                    phase=phase,
                    phase_root=phase_root,
                    completion_path=completion_path,
                    completion=completion,
                )
            )
    return completed


def _select_complete_attempt(
    attempts: Sequence[dict[str, object]],
    *,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
) -> dict[str, object]:
    if not attempts:
        raise CheckpointArchiveError("no valid completed archive attempt exists")
    checkpoint_ids = {
        (attempt["checkpoint"]["sha256"], attempt["checkpoint"]["size_bytes"])
        for attempt in attempts
    }
    if len(checkpoint_ids) != 1:
        raise CheckpointArchiveError(
            "valid completed attempts bind different checkpoints"
        )
    selected = min(attempts, key=lambda attempt: str(attempt["attempt_id"]))
    checkpoint = selected["checkpoint"]
    if expected_sha256 is not None and checkpoint["sha256"] != expected_sha256:
        raise CheckpointArchiveError(
            "an existing valid completion binds a different checkpoint"
        )
    if expected_size is not None and checkpoint["size_bytes"] != expected_size:
        raise CheckpointArchiveError(
            "an existing valid completion binds a different checkpoint"
        )
    return selected


def _receipt(
    attempt: dict[str, object],
    *,
    run_id: str,
    phase: str,
    publication: str,
    valid_attempt_count: int,
) -> dict[str, object]:
    return {
        "schema_name": COMPLETION_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "run_id": run_id,
        "phase": phase,
        "attempt_id": attempt["attempt_id"],
        "archive_root": attempt["attempt_root"],
        "phase_root": attempt["phase_root"],
        "checkpoint": attempt["checkpoint"],
        "manifest": attempt["manifest"],
        "completion_marker": attempt["completion_marker"],
        "valid_attempt_count": valid_attempt_count,
        "publication": {
            "attempt": publication,
            "checkpoint": publication,
            "manifest": publication,
            "completion": publication,
        },
    }


def _validate_run_phase(run_id: str, phase: str) -> None:
    if RUN_ID_PATTERN.fullmatch(run_id) is None or run_id in (".", ".."):
        raise CheckpointArchiveError("run-id is not one safe path component")
    if phase not in PHASES:
        raise CheckpointArchiveError(f"phase must be one of {PHASES}")


def verify_completed_phase(
    *,
    expected_checkpoint_sha256: str,
    run_id: str,
    phase: str,
    obs_root: Path = DEFAULT_OBS_ROOT,
) -> dict[str, object]:
    _validate_run_phase(run_id, phase)
    expected_sha256 = _sha256(
        expected_checkpoint_sha256, "expected checkpoint SHA-256"
    )
    root = _canonical_obs_root(obs_root)
    phase_root = _existing_phase_root(root, run_id, phase)
    attempts = _scan_complete_attempts(phase_root, run_id=run_id, phase=phase)
    selected = _select_complete_attempt(
        attempts, expected_sha256=expected_sha256
    )
    return _receipt(
        selected,
        run_id=run_id,
        phase=phase,
        publication="reused",
        valid_attempt_count=len(attempts),
    )


def archive_checkpoint(
    *,
    checkpoint: Path,
    expected_checkpoint_sha256: str,
    run_id: str,
    phase: str,
    obs_root: Path = DEFAULT_OBS_ROOT,
) -> dict[str, object]:
    _validate_run_phase(run_id, phase)
    expected_sha256 = _sha256(
        expected_checkpoint_sha256, "expected checkpoint SHA-256"
    )
    source = _regular_file(checkpoint, "checkpoint")
    source_size = _positive_size(source.stat().st_size, "checkpoint size")
    if sha256_file(source) != expected_sha256:
        raise CheckpointArchiveError("checkpoint SHA-256 differs")

    root = _canonical_obs_root(obs_root)
    checkpoint_root = _ensure_directory(root, root / "checkpoints", "checkpoint root")
    namespace = _ensure_directory(
        checkpoint_root,
        checkpoint_root / "navigation-rlt-stage-archives",
        "Navigation-RLT stage-checkpoint namespace",
    )
    run_root = _ensure_directory(namespace, namespace / run_id, "run archive root")
    phase_root = _ensure_directory(run_root, run_root / phase, "phase archive root")

    attempts = _scan_complete_attempts(phase_root, run_id=run_id, phase=phase)
    if attempts:
        selected = _select_complete_attempt(
            attempts,
            expected_sha256=expected_sha256,
            expected_size=source_size,
        )
        return _receipt(
            selected,
            run_id=run_id,
            phase=phase,
            publication="reused",
            valid_attempt_count=len(attempts),
        )

    attempt_id, attempt = _create_attempt(phase_root, expected_sha256)
    archived = attempt / CHECKPOINT_NAME
    _copy_new_checkpoint(source, archived, expected_sha256, source_size)
    manifest = _expected_manifest(
        run_id=run_id,
        phase=phase,
        attempt_id=attempt_id,
        attempt=attempt,
        source=source,
        source_size=source_size,
        expected_sha256=expected_sha256,
        archived=archived,
    )
    manifest_payload = _json_bytes(manifest)
    _write_new_payload(attempt / MANIFEST_NAME, manifest_payload, "archive manifest")
    completion = _expected_completion(
        run_id=run_id,
        phase=phase,
        attempt_id=attempt_id,
        attempt=attempt,
        phase_root=phase_root,
        archived=archived,
        checkpoint_sha256=expected_sha256,
        checkpoint_size=source_size,
        manifest_payload=manifest_payload,
    )
    _write_new_payload(
        attempt / COMPLETION_NAME,
        _json_bytes(completion),
        "completion marker",
    )

    completed = _scan_complete_attempts(phase_root, run_id=run_id, phase=phase)
    _select_complete_attempt(
        completed,
        expected_sha256=expected_sha256,
        expected_size=source_size,
    )
    try:
        created = next(
            item for item in completed if item["attempt_id"] == attempt_id
        )
    except StopIteration as error:
        raise CheckpointArchiveError(
            "new completion was not accepted by strict readback"
        ) from error
    return _receipt(
        created,
        run_id=run_id,
        phase=phase,
        publication="created",
        valid_attempt_count=len(completed),
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--checkpoint", type=Path)
    result.add_argument("--expected-checkpoint-sha256")
    result.add_argument("--run-id", required=True)
    result.add_argument("--phase", required=True, choices=PHASES)
    result.add_argument("--obs-root", type=Path, default=DEFAULT_OBS_ROOT)
    result.add_argument("--verify-completed-phase", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    if arguments.verify_completed_phase:
        if arguments.checkpoint is not None:
            raise CheckpointArchiveError(
                "phase verification does not accept a checkpoint path"
            )
        if arguments.expected_checkpoint_sha256 is None:
            raise CheckpointArchiveError(
                "expected checkpoint SHA-256 is required for phase verification"
            )
        result = verify_completed_phase(
            expected_checkpoint_sha256=arguments.expected_checkpoint_sha256,
            run_id=arguments.run_id,
            phase=arguments.phase,
            obs_root=arguments.obs_root,
        )
    else:
        if arguments.checkpoint is None or arguments.expected_checkpoint_sha256 is None:
            raise CheckpointArchiveError(
                "checkpoint and expected checkpoint SHA-256 are required"
            )
        result = archive_checkpoint(
            checkpoint=arguments.checkpoint,
            expected_checkpoint_sha256=arguments.expected_checkpoint_sha256,
            run_id=arguments.run_id,
            phase=arguments.phase,
            obs_root=arguments.obs_root,
        )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CheckpointArchiveError, FileExistsError, OSError) as error:
        print(
            json.dumps(
                {
                    "schema_name": ARCHIVE_SCHEMA,
                    "schema_version": SCHEMA_VERSION,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(2)
