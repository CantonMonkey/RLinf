"""Fail-closed storage preflight for Navigation-RLT writes."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence, TextIO


GIB = 1024**3
TURBO_SOURCE = (
    "navigation."
    "storage.example.invalid:/"
)

# A write is allowed only when the projected ordinary-user space remains at
# least 100 GiB. Profile warning thresholds are intentionally higher.
MIN_PROJECTED_AVAILABLE_BYTES = 100 * GIB
TURBO_AVAILABLE_WARNING_BYTES = 200 * GIB
EBK180_AVAILABLE_WARNING_BYTES = 150 * GIB

BTRFS_MINIMUM_FREE_WARNING_BYTES = 80 * GIB
BTRFS_MINIMUM_FREE_STOP_BYTES = 50 * GIB
BTRFS_METADATA_WARNING_PERCENT = 90.0
BTRFS_METADATA_STOP_PERCENT = 95.0

SCHEMA_NAME = "navigation_rlt_storage_guard"
SCHEMA_VERSION = 1

PROFILE_POLICIES: Mapping[str, Mapping[str, Any]] = {
    "turbo": {
        "default_path": "/home/navcloud/work",
        "available_warning_bytes": TURBO_AVAILABLE_WARNING_BYTES,
        "filesystems": ("nfs", "nfs4"),
        "source_equals": TURBO_SOURCE,
        "storage_class": "sfs_turbo",
    },
    "ebk180": {
        "default_path": "/home/navgpu/work",
        "available_warning_bytes": EBK180_AVAILABLE_WARNING_BYTES,
        "filesystems": ("btrfs",),
        "source_equals": "/dev/nvme0n1p2",
        "storage_class": "btrfs",
    },
}


class StorageGuardError(RuntimeError):
    """Raised when storage safety cannot be established."""


def _default_statvfs(path: str) -> Any:
    statvfs = getattr(os, "statvfs", None)
    if statvfs is None:
        raise StorageGuardError("os.statvfs is unavailable on this platform")
    return statvfs(path)


def _default_read_mountinfo() -> str:
    try:
        return Path("/proc/self/mountinfo").read_text(encoding="utf-8")
    except OSError as exc:
        raise StorageGuardError("cannot read /proc/self/mountinfo") from exc


def _default_read_btrfs_usage(path: str) -> str:
    environment = dict(os.environ)
    environment["LC_ALL"] = "C"
    try:
        completed = subprocess.run(
            ["btrfs", "filesystem", "usage", "--raw", path],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            env=environment,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise StorageGuardError("cannot query btrfs filesystem usage") from exc
    if completed.returncode != 0:
        raise StorageGuardError(
            f"btrfs filesystem usage failed with exit code {completed.returncode}"
        )
    return completed.stdout


@dataclass(frozen=True)
class GuardDependencies:
    """Operating-system probes injected by tests and callers."""

    statvfs: Callable[[str], Any] = _default_statvfs
    read_mountinfo: Callable[[], str] = _default_read_mountinfo
    read_btrfs_usage: Callable[[str], str] = _default_read_btrfs_usage
    now: Callable[[], float] = time.time


@dataclass(frozen=True)
class MountRecord:
    mount_point: Path
    filesystem: str
    source: str
    mount_options: frozenset[str]
    super_options: frozenset[str]

    @property
    def writable(self) -> bool:
        return (
            "rw" in self.mount_options
            and "rw" in self.super_options
            and "ro" not in self.mount_options
            and "ro" not in self.super_options
        )


_MOUNTINFO_ESCAPE = re.compile(r"\\([0-7]{3})")
_BTRFS_FREE = re.compile(
    r"^\s*Free \(estimated\):\s*(\d+)\s*\(min:\s*(\d+)\)\s*$",
    re.MULTILINE,
)
_BTRFS_METADATA = re.compile(
    r"^\s*Metadata(?:,[^:]+)?:\s*Size:\s*(\d+)\s*,\s*Used:\s*(\d+)"
    r"(?:\s+\([^\r\n]*\))?\s*$",
    re.MULTILINE,
)


def _decode_mountinfo(value: str) -> str:
    return _MOUNTINFO_ESCAPE.sub(lambda match: chr(int(match.group(1), 8)), value)


def _parse_mountinfo(text: str) -> list[MountRecord]:
    records: list[MountRecord] = []
    for line in text.splitlines():
        if " - " not in line:
            continue
        left_text, right_text = line.split(" - ", 1)
        left = left_text.split()
        right = right_text.split()
        if len(left) < 6 or len(right) < 3:
            continue
        records.append(
            MountRecord(
                mount_point=Path(_decode_mountinfo(left[4])),
                filesystem=right[0].lower(),
                source=_decode_mountinfo(right[1]),
                mount_options=frozenset(left[5].split(",")),
                super_options=frozenset(right[2].split(",")),
            )
        )
    return records


def _record_contains(record: MountRecord, target: Path) -> bool:
    try:
        target.relative_to(record.mount_point)
    except (ValueError, OSError):
        return False
    return True


def _find_mount(target: Path, mountinfo: str) -> MountRecord:
    candidates = [
        record for record in _parse_mountinfo(mountinfo) if _record_contains(record, target)
    ]
    if not candidates:
        raise StorageGuardError(f"no mountinfo entry contains {target}")
    return max(candidates, key=lambda record: len(record.mount_point.parts))


def _ordinary_available_bytes(stats: Any) -> int:
    try:
        blocks = int(stats.f_bavail)
        fragment_size = int(stats.f_frsize)
    except (AttributeError, TypeError, ValueError) as exc:
        raise StorageGuardError("statvfs result lacks valid f_bavail/f_frsize") from exc
    if blocks < 0 or fragment_size <= 0:
        raise StorageGuardError("statvfs returned invalid available-space fields")
    return blocks * fragment_size


def parse_btrfs_usage(text: str) -> tuple[int, float]:
    """Return minimum estimated free bytes and aggregate metadata use percent."""

    free_match = _BTRFS_FREE.search(text)
    metadata_matches = _BTRFS_METADATA.findall(text)
    if free_match is None:
        raise StorageGuardError("btrfs usage lacks minimum estimated free bytes")
    if not metadata_matches:
        raise StorageGuardError("btrfs usage lacks metadata size and use")

    minimum_free_bytes = int(free_match.group(2))
    metadata_size_bytes = sum(int(size) for size, _ in metadata_matches)
    metadata_used_bytes = sum(int(used) for _, used in metadata_matches)
    if metadata_size_bytes <= 0 or metadata_used_bytes > metadata_size_bytes:
        raise StorageGuardError("btrfs usage contains invalid metadata totals")
    metadata_percent = metadata_used_bytes * 100.0 / metadata_size_bytes
    return minimum_free_bytes, metadata_percent


def _validate_pending_write(pending_write_bytes: int) -> int:
    if not isinstance(pending_write_bytes, int) or isinstance(pending_write_bytes, bool):
        raise StorageGuardError("pending_write_bytes must be a non-negative integer")
    value = int(pending_write_bytes)
    if value < 0:
        raise StorageGuardError("pending_write_bytes must be a non-negative integer")
    return value


def evaluate_storage(
    profile: str,
    path: str | os.PathLike[str] | None,
    pending_write_bytes: int,
    *,
    dependencies: GuardDependencies | None = None,
) -> dict[str, Any]:
    """Evaluate one projected write and return a machine-readable decision."""

    if profile not in PROFILE_POLICIES:
        raise StorageGuardError(f"unknown storage profile: {profile}")
    pending = _validate_pending_write(pending_write_bytes)
    policy = PROFILE_POLICIES[profile]
    target_input = Path(path if path is not None else policy["default_path"])
    try:
        target = target_input.resolve(strict=True)
    except OSError as exc:
        raise StorageGuardError(f"storage path does not exist: {target_input}") from exc
    if not target.is_dir():
        raise StorageGuardError(f"storage path is not a directory: {target}")

    probes = dependencies or GuardDependencies()
    try:
        mountinfo = probes.read_mountinfo()
    except StorageGuardError:
        raise
    except Exception as exc:
        raise StorageGuardError("mountinfo probe failed") from exc
    mount = _find_mount(target, mountinfo)
    allowed_filesystems = tuple(policy["filesystems"])
    if mount.filesystem not in allowed_filesystems:
        expected = "/".join(allowed_filesystems)
        raise StorageGuardError(
            f"{profile} storage must use {expected}, found {mount.filesystem}"
        )
    if not mount.writable:
        raise StorageGuardError(f"{profile} storage mount is not read-write")
    source_equals = policy.get("source_equals")
    if source_equals is not None and mount.source != source_equals:
        raise StorageGuardError(f"{profile} storage mount source is not the expected source")

    try:
        available_bytes = _ordinary_available_bytes(probes.statvfs(str(target)))
    except StorageGuardError:
        raise
    except Exception as exc:
        raise StorageGuardError("statvfs probe failed") from exc
    projected_available_bytes = available_bytes - pending

    warnings: list[str] = []
    blocking_reasons: list[str] = []
    warning_bytes = int(policy["available_warning_bytes"])
    if projected_available_bytes < warning_bytes:
        warnings.append("ordinary_available_below_profile_warning")
    if projected_available_bytes < MIN_PROJECTED_AVAILABLE_BYTES:
        blocking_reasons.append("ordinary_available_below_stop")

    btrfs: dict[str, Any] | None = None
    if profile == "ebk180":
        try:
            usage_text = probes.read_btrfs_usage(str(target))
        except StorageGuardError:
            raise
        except Exception as exc:
            raise StorageGuardError("btrfs usage probe failed") from exc
        minimum_free_bytes, metadata_percent = parse_btrfs_usage(usage_text)
        projected_minimum_free_bytes = minimum_free_bytes - pending
        if projected_minimum_free_bytes < BTRFS_MINIMUM_FREE_WARNING_BYTES:
            warnings.append("btrfs_minimum_free_below_warning")
        if projected_minimum_free_bytes < BTRFS_MINIMUM_FREE_STOP_BYTES:
            blocking_reasons.append("btrfs_minimum_free_below_stop")
        if metadata_percent >= BTRFS_METADATA_WARNING_PERCENT:
            warnings.append("btrfs_metadata_at_or_above_warning")
        if metadata_percent >= BTRFS_METADATA_STOP_PERCENT:
            blocking_reasons.append("btrfs_metadata_at_or_above_stop")
        btrfs = {
            "metadata_percent": metadata_percent,
            "metadata_stop_percent": BTRFS_METADATA_STOP_PERCENT,
            "metadata_warning_percent": BTRFS_METADATA_WARNING_PERCENT,
            "minimum_free_bytes": minimum_free_bytes,
            "minimum_free_stop_bytes": BTRFS_MINIMUM_FREE_STOP_BYTES,
            "minimum_free_warning_bytes": BTRFS_MINIMUM_FREE_WARNING_BYTES,
            "projected_minimum_free_bytes": projected_minimum_free_bytes,
        }

    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "checked_at_unix": int(probes.now()),
        "status": "blocked" if blocking_reasons else "passed",
        "profile": profile,
        "path": str(target),
        "pending_write_bytes": pending,
        "ordinary_space": {
            "available_bytes": available_bytes,
            "projected_available_bytes": projected_available_bytes,
            "stop_bytes": MIN_PROJECTED_AVAILABLE_BYTES,
            "warning_bytes": warning_bytes,
        },
        "mount": {
            "filesystem": mount.filesystem,
            "mount_point": str(mount.mount_point),
            "source": mount.source,
            "source_verified": True,
            "storage_class": policy["storage_class"],
            "writable": mount.writable,
        },
        "btrfs": btrfs,
        "warnings": warnings,
        "blocking_reasons": blocking_reasons,
    }


def atomic_write_status(path: str | os.PathLike[str], value: Mapping[str, Any]) -> None:
    """Atomically replace a JSON status file after flushing its contents."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.staging-",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        if os.name != "nt":
            directory_descriptor = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def _error_receipt(
    profile: str | None,
    path: str | None,
    pending_write_bytes: object,
    message: str,
    now: Callable[[], float],
) -> dict[str, Any]:
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "checked_at_unix": int(now()),
        "status": "error",
        "profile": profile,
        "path": path,
        "pending_write_bytes": pending_write_bytes,
        "error": message,
        "warnings": [],
        "blocking_reasons": ["storage_safety_not_established"],
    }


def _emit_json(stream: TextIO, value: Mapping[str, Any]) -> None:
    stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    stream.flush()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, choices=sorted(PROFILE_POLICIES))
    parser.add_argument("--path", help="Existing directory on the guarded filesystem")
    parser.add_argument("--pending-write-bytes", required=True, type=int)
    parser.add_argument("--status-output", help="Optional atomically replaced JSON status")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    dependencies: GuardDependencies | None = None,
    stdout: TextIO | None = None,
) -> int:
    arguments = _parser().parse_args(argv)
    probes = dependencies or GuardDependencies()
    output = stdout or sys.stdout
    selected_path = arguments.path or str(
        PROFILE_POLICIES[arguments.profile]["default_path"]
    )
    try:
        receipt = evaluate_storage(
            arguments.profile,
            selected_path,
            arguments.pending_write_bytes,
            dependencies=probes,
        )
        if arguments.status_output:
            atomic_write_status(arguments.status_output, receipt)
    except Exception as exc:
        message = str(exc) if isinstance(exc, StorageGuardError) else "storage guard failed"
        receipt = _error_receipt(
            arguments.profile,
            selected_path,
            arguments.pending_write_bytes,
            message,
            probes.now,
        )
        if arguments.status_output:
            try:
                atomic_write_status(arguments.status_output, receipt)
            except Exception:
                receipt["error"] = f"{message}; status output failed"
        _emit_json(output, receipt)
        return 2

    _emit_json(output, receipt)
    return 3 if receipt["status"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
