#!/usr/bin/env python3
"""Create-only finalizer for a Navigation-RLT evaluation runtime on ebk.

This program is intentionally outside the immutable 35-source evaluation
closure.  It validates the 37-member deployment ZIP, materializes it below a
unique hidden staging directory, and publishes that directory with Linux
``renameat2(RENAME_NOREPLACE)``.  It never deletes or overwrites the input
archive, a failed staging directory, or a competing final directory.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import time
from typing import BinaryIO, Mapping, Sequence
import uuid
import zipfile


EVALUATION_INVENTORY_SCHEMA = "navigation_rlt_evaluation_code_inventory"
DEPLOYMENT_SCHEMA = "navigation_rlt_evaluation_deployment"
RESULT_SCHEMA = "navigation_rlt_remote_evaluation_runtime_finalize_result"
SCHEMA_VERSION = 1
RENAME_NOREPLACE = 1
AT_FDCWD = -100
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
GIT_SHA1_PATTERN = re.compile(r"[0-9a-f]{40}")
MAX_ARCHIVE_BYTES = 1024**3
MAX_UNCOMPRESSED_BYTES = 256 * 1024**2
EVALUATION_WRAPPER = "run_navigation_rlt_ebk_eval.sh"
EVALUATION_CODE_FILES = (
    "build_navigation_rlt_train_split.py",
    "lightnav_a3_policy.py",
    "lightnav_a3_rollout.py",
    "lightnav_l0_native_probe.py",
    "navigation_rlt_action.py",
    "navigation_rlt_compare.py",
    "navigation_rlt_eval.py",
    "navigation_rlt_obs_archive.py",
    "navigation_rlt_parallel.py",
    "navigation_rlt_policy.py",
    "navigation_rlt_post_training.py",
    "navigation_rlt_replay.py",
    "navigation_rlt_rollout.py",
    "streamvln_executor_contract.py",
    "streamvln_topology.py",
    "navrl/__init__.py",
    "navrl/contracts.py",
    "navrl/envs/__init__.py",
    "navrl/envs/base.py",
    "navrl/envs/grid_mock.py",
    "navrl/episode.py",
    "navrl/metrics.py",
    "navrl/policies/__init__.py",
    "navrl/policies/base.py",
    "navrl/rewards.py",
    "navrl/stack/__init__.py",
    "navrl/stack/checkpoint.py",
    "navrl/stack/collector.py",
    "navrl/stack/contracts.py",
    "navrl/stack/execution.py",
    "navrl/stack/interfaces.py",
    "navrl/stack/legacy.py",
    "navrl/stack/ppo.py",
    "navrl/stack/returns.py",
    EVALUATION_WRAPPER,
)
EVALUATION_CODE_FILE_SET = frozenset(EVALUATION_CODE_FILES)
PLAN_MEMBER = "eval-plan.json"
INVENTORY_MEMBER = "eval-code/evaluation_code_inventory.json"
ARCHIVE_MEMBERS = frozenset(
    {PLAN_MEMBER, INVENTORY_MEMBER}
    | {f"eval-code/{name}" for name in EVALUATION_CODE_FILES}
)
DEPLOYMENT_MARKER = "DEPLOYMENT.json"


class RemoteRuntimeFinalizeError(RuntimeError):
    """Raised when the requested runtime cannot be safely finalized."""


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_stream(stream: BinaryIO) -> str:
    position = stream.tell()
    stream.seek(0)
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    stream.seek(position)
    return digest.hexdigest()


def _strict_json_bytes(payload: bytes, name: str) -> dict[str, object]:
    def reject_constant(value: str) -> object:
        raise RemoteRuntimeFinalizeError(
            f"{name} contains forbidden non-finite constant {value}"
        )

    def reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise RemoteRuntimeFinalizeError(
                    f"{name} contains duplicate object key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_pairs,
        )
    except UnicodeDecodeError as error:
        raise RemoteRuntimeFinalizeError(f"{name} is not UTF-8 JSON") from error
    except json.JSONDecodeError as error:
        raise RemoteRuntimeFinalizeError(f"{name} is not strict JSON") from error
    if not isinstance(value, dict):
        raise RemoteRuntimeFinalizeError(f"{name} must contain an object")
    return value


def _json_bytes(value: Mapping[str, object]) -> bytes:
    try:
        return (
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RemoteRuntimeFinalizeError(
            "deployment marker cannot be encoded as strict JSON"
        ) from error


def _expected_sha256(value: str, name: str) -> str:
    if SHA256_PATTERN.fullmatch(value) is None:
        raise RemoteRuntimeFinalizeError(f"{name} must be a lowercase SHA-256")
    return value


def _expected_commit(value: str) -> str:
    if GIT_SHA1_PATTERN.fullmatch(value) is None:
        raise RemoteRuntimeFinalizeError(
            "expected navigation code commit must be a lowercase full Git SHA-1"
        )
    return value


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _safe_existing_directory(path: Path, name: str) -> Path:
    raw = _absolute(path)
    if raw.is_symlink():
        raise RemoteRuntimeFinalizeError(f"{name} must not be a symbolic link")
    try:
        status = raw.lstat()
        resolved = raw.resolve(strict=True)
    except OSError as error:
        raise RemoteRuntimeFinalizeError(f"{name} is absent") from error
    if resolved != raw:
        raise RemoteRuntimeFinalizeError(f"{name} must not traverse a symbolic link")
    if not stat.S_ISDIR(status.st_mode):
        raise RemoteRuntimeFinalizeError(f"{name} must be a directory")
    return resolved


def _strict_descendant(path: Path, root: Path, name: str) -> Path:
    raw = _absolute(path)
    try:
        relative = raw.relative_to(root)
    except ValueError as error:
        raise RemoteRuntimeFinalizeError(f"{name} escapes the run root") from error
    if not relative.parts:
        raise RemoteRuntimeFinalizeError(f"{name} must be below the run root")
    return raw


def _safe_archive_path(path: Path, root: Path) -> Path:
    raw = _strict_descendant(path, root, "deployment archive")
    _safe_existing_directory(raw.parent, "deployment archive parent")
    if raw.is_symlink():
        raise RemoteRuntimeFinalizeError(
            "deployment archive must not be a symbolic link"
        )
    try:
        status = raw.lstat()
        resolved = raw.resolve(strict=True)
    except OSError as error:
        raise RemoteRuntimeFinalizeError("deployment archive is absent") from error
    if resolved != raw:
        raise RemoteRuntimeFinalizeError(
            "deployment archive must not traverse a symbolic link"
        )
    if not stat.S_ISREG(status.st_mode):
        raise RemoteRuntimeFinalizeError("deployment archive must be a regular file")
    return resolved


def _safe_absent_final(path: Path, root: Path) -> Path:
    raw = _strict_descendant(path, root, "final runtime root")
    parent = _safe_existing_directory(raw.parent, "final runtime parent")
    if parent != raw.parent:
        raise RemoteRuntimeFinalizeError(
            "final runtime root must not traverse a symbolic link"
        )
    if _lexists(raw):
        raise FileExistsError(f"final runtime root already exists: {raw}")
    return raw


def _stat_identity(status: os.stat_result) -> dict[str, int]:
    return {
        "device": int(status.st_dev),
        "inode": int(status.st_ino),
        "size_bytes": int(status.st_size),
        "mtime_ns": int(status.st_mtime_ns),
        "ctime_ns": int(status.st_ctime_ns),
    }


def _open_archive(path: Path) -> tuple[BinaryIO, dict[str, int]]:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if nofollow:
        flags |= nofollow
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RemoteRuntimeFinalizeError(
            "deployment archive could not be opened without following links"
        ) from error
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode):
            raise RemoteRuntimeFinalizeError(
                "deployment archive descriptor is not a regular file"
            )
        stream = os.fdopen(descriptor, "rb", closefd=True)
    except BaseException:
        os.close(descriptor)
        raise
    return stream, _stat_identity(status)


def _assert_archive_unchanged(
    stream: BinaryIO,
    path: Path,
    expected_identity: Mapping[str, int],
    expected_sha256: str,
) -> None:
    descriptor_identity = _stat_identity(os.fstat(stream.fileno()))
    try:
        path_status = path.lstat()
    except OSError as error:
        raise RemoteRuntimeFinalizeError(
            "deployment archive path disappeared during finalization"
        ) from error
    if path.is_symlink() or not stat.S_ISREG(path_status.st_mode):
        raise RemoteRuntimeFinalizeError(
            "deployment archive path changed into a link or non-regular file"
        )
    path_identity = _stat_identity(path_status)
    if descriptor_identity != dict(expected_identity) or path_identity != dict(
        expected_identity
    ):
        raise RemoteRuntimeFinalizeError(
            "deployment archive identity changed during finalization"
        )
    if _sha256_stream(stream) != expected_sha256:
        raise RemoteRuntimeFinalizeError(
            "deployment archive content changed during finalization"
        )


def _canonical_member_name(name: str) -> bool:
    if not name or "\\" in name or "\x00" in name:
        return False
    pure = PurePosixPath(name)
    return (
        not pure.is_absolute()
        and pure.as_posix() == name
        and all(part not in ("", ".", "..") for part in pure.parts)
    )


def _regular_zip_member(info: zipfile.ZipInfo) -> bool:
    if info.is_dir() or info.flag_bits & 0x1:
        return False
    mode = (info.external_attr >> 16) & 0xFFFF
    file_type = stat.S_IFMT(mode)
    if info.create_system == 3 and file_type not in (0, stat.S_IFREG):
        return False
    # DOS directory bit.  Python's is_dir() only checks the trailing slash.
    if info.external_attr & 0x10:
        return False
    return True


def _validate_inventory(
    payload: bytes,
    expected_inventory_sha256: str,
    expected_commit: str,
) -> tuple[dict[str, object], dict[str, str]]:
    if _sha256_bytes(payload) != expected_inventory_sha256:
        raise RemoteRuntimeFinalizeError(
            "evaluation code inventory SHA-256 differs"
        )
    inventory = _strict_json_bytes(payload, "evaluation code inventory")
    required_fields = {
        "schema_name",
        "schema_version",
        "navigation_code_commit",
        "files",
    }
    if set(inventory) != required_fields:
        raise RemoteRuntimeFinalizeError(
            "evaluation code inventory fields differ"
        )
    if inventory.get("schema_name") != EVALUATION_INVENTORY_SCHEMA:
        raise RemoteRuntimeFinalizeError(
            "evaluation code inventory schema name differs"
        )
    if type(inventory.get("schema_version")) is not int or inventory.get(
        "schema_version"
    ) != SCHEMA_VERSION:
        raise RemoteRuntimeFinalizeError(
            "evaluation code inventory schema version differs"
        )
    if inventory.get("navigation_code_commit") != expected_commit:
        raise RemoteRuntimeFinalizeError(
            "evaluation code inventory navigation commit differs"
        )
    files = inventory.get("files")
    if not isinstance(files, dict) or set(files) != EVALUATION_CODE_FILE_SET:
        raise RemoteRuntimeFinalizeError(
            "evaluation code inventory source set differs"
        )
    validated: dict[str, str] = {}
    for name in EVALUATION_CODE_FILES:
        digest = files.get(name)
        if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
            raise RemoteRuntimeFinalizeError(
                f"evaluation source {name} inventory SHA-256 is invalid"
            )
        validated[name] = digest
    return inventory, validated


def _read_validated_archive(
    stream: BinaryIO,
    expected_plan_sha256: str,
    expected_inventory_sha256: str,
    expected_commit: str,
) -> tuple[dict[str, bytes], dict[str, str]]:
    try:
        with zipfile.ZipFile(stream, "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(infos) != len(ARCHIVE_MEMBERS):
                raise RemoteRuntimeFinalizeError(
                    "deployment archive must contain exactly 37 members"
                )
            if len(names) != len(set(names)):
                raise RemoteRuntimeFinalizeError(
                    "deployment archive contains duplicate members"
                )
            if set(names) != ARCHIVE_MEMBERS:
                raise RemoteRuntimeFinalizeError(
                    "deployment archive member set differs"
                )
            if any(not _canonical_member_name(name) for name in names):
                raise RemoteRuntimeFinalizeError(
                    "deployment archive contains a non-canonical path"
                )
            if any(not _regular_zip_member(info) for info in infos):
                raise RemoteRuntimeFinalizeError(
                    "deployment archive contains a link or non-regular member"
                )
            total_size = sum(info.file_size for info in infos)
            if total_size > MAX_UNCOMPRESSED_BYTES:
                raise RemoteRuntimeFinalizeError(
                    "deployment archive uncompressed payload is too large"
                )
            payloads = {info.filename: archive.read(info) for info in infos}
    except (
        OSError,
        RuntimeError,
        NotImplementedError,
        zipfile.BadZipFile,
        zipfile.LargeZipFile,
    ) as error:
        if isinstance(error, RemoteRuntimeFinalizeError):
            raise
        raise RemoteRuntimeFinalizeError(
            "deployment archive is not a valid bounded ZIP"
        ) from error

    if _sha256_bytes(payloads[PLAN_MEMBER]) != expected_plan_sha256:
        raise RemoteRuntimeFinalizeError("evaluation plan SHA-256 differs")
    _, source_hashes = _validate_inventory(
        payloads[INVENTORY_MEMBER],
        expected_inventory_sha256,
        expected_commit,
    )
    for name, expected_hash in source_hashes.items():
        if _sha256_bytes(payloads[f"eval-code/{name}"]) != expected_hash:
            raise RemoteRuntimeFinalizeError(
                f"evaluation source {name} SHA-256 differs"
            )
    return payloads, source_hashes


def _allocate_staging(final_root: Path) -> Path:
    for _ in range(64):
        staging = final_root.with_name(
            f".{final_root.name}.staging-{uuid.uuid4().hex}"
        )
        try:
            staging.mkdir(mode=0o700)
        except FileExistsError:
            continue
        _fsync_directory(staging.parent)
        return staging
    raise RemoteRuntimeFinalizeError(
        "could not allocate a unique hidden runtime staging directory"
    )


def _ensure_staging_directory(staging: Path, relative: PurePosixPath) -> Path:
    current = staging
    for part in relative.parts:
        candidate = current / part
        if _lexists(candidate):
            if candidate.is_symlink():
                raise RemoteRuntimeFinalizeError(
                    f"staging directory became a symbolic link: {candidate}"
                )
            status = candidate.lstat()
            if not stat.S_ISDIR(status.st_mode):
                raise RemoteRuntimeFinalizeError(
                    f"staging directory path is not a directory: {candidate}"
                )
        else:
            candidate.mkdir(mode=0o700)
        current = candidate
    return current


def _write_new_file(path: Path, payload: bytes) -> None:
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
            0o600,
        )
    except FileExistsError as error:
        raise RemoteRuntimeFinalizeError(
            f"staging file already exists: {path}"
        ) from error
    with os.fdopen(descriptor, "wb", closefd=True) as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    except OSError:
        if os.name != "nt":
            raise
    finally:
        os.close(descriptor)


def _materialize_payloads(
    staging: Path,
    payloads: Mapping[str, bytes],
    deployment_payload: bytes,
) -> None:
    directories: set[Path] = {staging}
    for name in sorted(payloads):
        relative = PurePosixPath(name)
        parent = _ensure_staging_directory(staging, relative.parent)
        directories.add(parent)
        current = parent
        while current != staging:
            directories.add(current)
            current = current.parent
        destination = staging.joinpath(*relative.parts)
        _write_new_file(destination, payloads[name])

    for name in sorted(payloads):
        relative = PurePosixPath(name)
        path = staging.joinpath(*relative.parts)
        expected_mode = (
            0o555
            if name == f"eval-code/{EVALUATION_WRAPPER}"
            else 0o444
        )
        os.chmod(path, expected_mode, follow_symlinks=False)
    for directory in sorted(directories - {staging}, key=lambda item: len(item.parts), reverse=True):
        os.chmod(directory, 0o555, follow_symlinks=False)

    # The completion marker is deliberately the final file created in staging.
    marker = staging / DEPLOYMENT_MARKER
    _write_new_file(marker, deployment_payload)
    os.chmod(marker, 0o444, follow_symlinks=False)
    os.chmod(staging, 0o555, follow_symlinks=False)
    for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        _fsync_directory(directory)


def _validate_materialized_runtime(
    root: Path,
    payloads: Mapping[str, bytes],
    source_hashes: Mapping[str, str],
    deployment_payload: bytes,
) -> None:
    expected_files = set(payloads) | {DEPLOYMENT_MARKER}
    observed_files: set[str] = set()
    observed_directories: list[Path] = [root]
    for current, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        if current_path.is_symlink():
            raise RemoteRuntimeFinalizeError(
                "materialized runtime contains a symbolic-link directory"
            )
        for directory_name in directory_names:
            child = current_path / directory_name
            if child.is_symlink() or not stat.S_ISDIR(child.lstat().st_mode):
                raise RemoteRuntimeFinalizeError(
                    "materialized runtime contains a non-directory entry"
                )
            observed_directories.append(child)
        for file_name in file_names:
            child = current_path / file_name
            if child.is_symlink() or not stat.S_ISREG(child.lstat().st_mode):
                raise RemoteRuntimeFinalizeError(
                    "materialized runtime contains a link or non-regular file"
                )
            observed_files.add(child.relative_to(root).as_posix())
    if observed_files != expected_files:
        raise RemoteRuntimeFinalizeError(
            "materialized runtime file set differs"
        )
    for name, payload in payloads.items():
        path = root.joinpath(*PurePosixPath(name).parts)
        if path.read_bytes() != payload:
            raise RemoteRuntimeFinalizeError(
                f"materialized runtime readback differs: {name}"
            )
        expected_mode = (
            0o555
            if name == f"eval-code/{EVALUATION_WRAPPER}"
            else 0o444
        )
        if stat.S_IMODE(path.lstat().st_mode) != expected_mode:
            raise RemoteRuntimeFinalizeError(
                f"materialized runtime mode differs: {name}"
            )
    for name, digest in source_hashes.items():
        path = root.joinpath("eval-code", *PurePosixPath(name).parts)
        if _sha256_bytes(path.read_bytes()) != digest:
            raise RemoteRuntimeFinalizeError(
                f"materialized evaluation source SHA-256 differs: {name}"
            )
    marker = root / DEPLOYMENT_MARKER
    if marker.read_bytes() != deployment_payload:
        raise RemoteRuntimeFinalizeError("deployment marker readback differs")
    if stat.S_IMODE(marker.lstat().st_mode) != 0o444:
        raise RemoteRuntimeFinalizeError("deployment marker mode differs")
    for directory in observed_directories:
        if stat.S_IMODE(directory.lstat().st_mode) != 0o555:
            raise RemoteRuntimeFinalizeError(
                f"materialized runtime directory mode differs: {directory}"
            )


def _renameat2_noreplace(source: Path, destination: Path) -> None:
    if not sys.platform.startswith("linux"):
        raise RemoteRuntimeFinalizeError(
            "runtime publication requires Linux renameat2(RENAME_NOREPLACE)"
        )
    try:
        function = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as error:
        raise RemoteRuntimeFinalizeError(
            "renameat2(RENAME_NOREPLACE) is unavailable"
        ) from error
    function.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    function.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = function(
        AT_FDCWD,
        os.fsencode(source),
        AT_FDCWD,
        os.fsencode(destination),
        RENAME_NOREPLACE,
    )
    if result != 0:
        error_number = ctypes.get_errno() or errno.EIO
        if error_number == errno.EEXIST:
            raise FileExistsError(destination)
        raise OSError(error_number, os.strerror(error_number), str(destination))


def _publish_runtime(staging: Path, final_root: Path) -> None:
    staging_parent = staging.parent.resolve(strict=True)
    final_parent = final_root.parent.resolve(strict=True)
    if staging_parent != final_parent:
        raise RemoteRuntimeFinalizeError(
            "runtime publication source and destination must share one parent"
        )
    if staging.is_symlink() or not stat.S_ISDIR(staging.lstat().st_mode):
        raise RemoteRuntimeFinalizeError(
            "runtime staging root is not a safe directory"
        )
    _renameat2_noreplace(staging, final_root)
    _fsync_directory(final_root.parent)


def finalize_runtime(
    *,
    run_root: Path,
    run_id: str,
    archive: Path,
    expected_archive_sha256: str,
    expected_archive_size_bytes: int,
    final_root: Path,
    expected_eval_plan_sha256: str,
    expected_evaluation_code_inventory_sha256: str,
    expected_navigation_code_commit: str,
) -> dict[str, object]:
    """Validate, stage, and atomically publish one immutable runtime."""

    if RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise RemoteRuntimeFinalizeError("run ID is invalid")
    archive_sha256 = _expected_sha256(
        expected_archive_sha256, "expected archive SHA-256"
    )
    plan_sha256 = _expected_sha256(
        expected_eval_plan_sha256, "expected evaluation plan SHA-256"
    )
    inventory_sha256 = _expected_sha256(
        expected_evaluation_code_inventory_sha256,
        "expected evaluation code inventory SHA-256",
    )
    commit = _expected_commit(expected_navigation_code_commit)
    if (
        type(expected_archive_size_bytes) is not int
        or expected_archive_size_bytes < 1
        or expected_archive_size_bytes > MAX_ARCHIVE_BYTES
    ):
        raise RemoteRuntimeFinalizeError(
            "expected archive size is outside the allowed range"
        )

    safe_run_root = _safe_existing_directory(run_root, "run root")
    if safe_run_root.name != run_id:
        raise RemoteRuntimeFinalizeError("run root basename differs from run ID")
    archive_path = _safe_archive_path(archive, safe_run_root)
    destination = _safe_absent_final(final_root, safe_run_root)

    stream, archive_identity = _open_archive(archive_path)
    with stream:
        if archive_identity["size_bytes"] != expected_archive_size_bytes:
            raise RemoteRuntimeFinalizeError("deployment archive size differs")
        if _sha256_stream(stream) != archive_sha256:
            raise RemoteRuntimeFinalizeError("deployment archive SHA-256 differs")
        payloads, source_hashes = _read_validated_archive(
            stream,
            plan_sha256,
            inventory_sha256,
            commit,
        )
        _assert_archive_unchanged(
            stream, archive_path, archive_identity, archive_sha256
        )
        if _lexists(destination):
            raise FileExistsError(
                f"final runtime root already exists: {destination}"
            )

        deployment = {
            "schema_name": DEPLOYMENT_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "status": "verified",
            "run_id": run_id,
            "run_root": str(safe_run_root),
            "deployment_root": str(destination),
            "archive": {
                "path": str(archive_path),
                "sha256": archive_sha256,
                "size_bytes": expected_archive_size_bytes,
                "identity": dict(archive_identity),
                "member_count": len(ARCHIVE_MEMBERS),
            },
            "archive_sha256": archive_sha256,
            "archive_size_bytes": expected_archive_size_bytes,
            "evaluation_plan_sha256": plan_sha256,
            "evaluation_code_inventory_sha256": inventory_sha256,
            "navigation_code_commit": commit,
            "file_count": len(source_hashes),
            "source_file_count": len(source_hashes),
            "source_files": dict(sorted(source_hashes.items())),
            "permission_contract": {
                "directories": "0555",
                "wrapper": "0555",
                "other_files": "0444",
            },
            "validation_split_reads": 0,
            "completed_unix_ns": time.time_ns(),
        }
        deployment_payload = _json_bytes(deployment)
        staging = _allocate_staging(destination)
        _materialize_payloads(staging, payloads, deployment_payload)
        _validate_materialized_runtime(
            staging, payloads, source_hashes, deployment_payload
        )
        _assert_archive_unchanged(
            stream, archive_path, archive_identity, archive_sha256
        )
        if _lexists(destination):
            raise FileExistsError(
                f"final runtime root already exists: {destination}"
            )
        _publish_runtime(staging, destination)
        _assert_archive_unchanged(
            stream, archive_path, archive_identity, archive_sha256
        )

    published_root = _safe_existing_directory(
        destination, "published runtime root"
    )
    if published_root != destination:
        raise RemoteRuntimeFinalizeError(
            "published runtime root resolves away from its fixed path"
        )
    _validate_materialized_runtime(
        published_root, payloads, source_hashes, deployment_payload
    )
    marker_path = published_root / DEPLOYMENT_MARKER
    marker_payload = marker_path.read_bytes()
    return {
        "schema_name": RESULT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "verified",
        "run_id": run_id,
        "run_root": str(safe_run_root),
        "deployment_root": str(destination),
        "archive": {
            "path": str(archive_path),
            "sha256": archive_sha256,
            "size_bytes": expected_archive_size_bytes,
            "identity": dict(archive_identity),
            "member_count": len(ARCHIVE_MEMBERS),
        },
        "archive_sha256": archive_sha256,
        "archive_size_bytes": expected_archive_size_bytes,
        "evaluation_plan_sha256": plan_sha256,
        "evaluation_code_inventory_sha256": inventory_sha256,
        "navigation_code_commit": commit,
        "file_count": len(source_hashes),
        "source_file_count": len(source_hashes),
        "deployed_file_count": len(payloads) + 1,
        "deployment_marker": {
            "path": str(marker_path),
            "sha256": _sha256_bytes(marker_payload),
            "size_bytes": len(marker_payload),
        },
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Create-only finalizer for the immutable Navigation-RLT ebk "
            "evaluation runtime"
        )
    )
    result.add_argument("--run-root", type=Path, required=True)
    result.add_argument("--run-id", required=True)
    result.add_argument("--archive", type=Path, required=True)
    result.add_argument("--expected-archive-sha256", required=True)
    result.add_argument("--expected-archive-size-bytes", type=int, required=True)
    result.add_argument("--final-root", type=Path, required=True)
    result.add_argument("--expected-eval-plan-sha256", required=True)
    result.add_argument(
        "--expected-evaluation-code-inventory-sha256", required=True
    )
    result.add_argument("--expected-navigation-code-commit", required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        result = finalize_runtime(
            run_root=arguments.run_root,
            run_id=arguments.run_id,
            archive=arguments.archive,
            expected_archive_sha256=arguments.expected_archive_sha256,
            expected_archive_size_bytes=arguments.expected_archive_size_bytes,
            final_root=arguments.final_root,
            expected_eval_plan_sha256=arguments.expected_eval_plan_sha256,
            expected_evaluation_code_inventory_sha256=(
                arguments.expected_evaluation_code_inventory_sha256
            ),
            expected_navigation_code_commit=(
                arguments.expected_navigation_code_commit
            ),
        )
    except (RemoteRuntimeFinalizeError, FileExistsError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            result,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
