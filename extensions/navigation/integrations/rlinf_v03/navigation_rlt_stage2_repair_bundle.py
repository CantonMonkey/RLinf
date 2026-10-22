#!/usr/bin/env python3
"""Create and verify the immutable Navigation-RLT Stage-2 repair bundle.

This helper deliberately keeps the repair code in a sibling of the original
``RUN_ROOT/code`` tree.  It creates an exact nine-file code inventory and a
deterministic archive, emits the Stage-2 repair protocol only after the
prepared replay support is known, and provides bounded pre-launch and
post-training lineage receipts.  Every published artifact uses create-only
semantics.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tarfile
from typing import Mapping, Sequence


SHA256_RE = re.compile(r"[0-9a-f]{64}")
SHA1_RE = re.compile(r"[0-9a-f]{40}")
RUN_ID_DEFAULT = "navigation-rlt-20261002-v2"
REPAIR_DIR_NAME = "stage2-repair-v2"
CODE_INVENTORY_SCHEMA = "navigation_rlt_code_archive_inventory"
CODE_VERIFICATION_SCHEMA = "navigation_rlt_a800_code_verification"
REPAIR_PROTOCOL_SCHEMA = "navigation_rlt_stage2_repair_protocol"
REPAIR_OBJECTIVE_SCHEMA = "proposal_conditioned_finite_candidate_cql_v1"
ELIGIBILITY_SCHEMA = "credit_mask_true_and_requested_gate_equals_executed_gate_v1"
GATE_NAMES = ["KEEP", "ADJUST", "STOP", "RECOVER"]
TRAINING_ADDITIONS = {
    "objective_schema": REPAIR_OBJECTIVE_SCHEMA,
    "cql_alpha": 1.0,
    "cql_temperature": 1.0,
    "minimum_behavior_gate_count": 1,
}

# These names are consumed by navigation_rlt_obs_archive.py and
# navigation_rlt_post_training.py.  The repair run wrapper has a distinct
# source name locally, then is installed under the canonical inventory name in
# the independent repair code root.
CODE_SOURCES = {
    "navigation_rlt_fsdp_train.py": "navigation_rlt_fsdp_train.py",
    "navigation_rlt_policy.py": "navigation_rlt_policy.py",
    "navigation_rlt_replay.py": "navigation_rlt_replay.py",
    "navigation_rlt_relocate_replay.py": "navigation_rlt_relocate_replay.py",
    "navigation_rlt_stage1_gate.py": "navigation_rlt_stage1_gate.py",
    "navigation_rlt_prepare_training.py": "navigation_rlt_prepare_training.py",
    "navigation_rlt_promote_replay.py": "navigation_rlt_promote_replay.py",
    "run_navigation_rlt_a800.sh": "run_navigation_rlt_a800_stage2_repair.sh",
    "lightnav_a3_gpu_monitor.py": "lightnav_a3_gpu_monitor.py",
}
EXECUTABLE_FILES = frozenset({"run_navigation_rlt_a800.sh"})
PROVENANCE_SOURCES = (
    "navigation_rlt_stage2_repair_bundle.py",
    "continue_navigation_rlt_a800_stage2_repair.sh",
    "continue_navigation_rlt_a800_stage2_repair_checkpoint_obs.sh",
    "launch_navigation_rlt_stage2_repair_v2.sh",
)


class RepairBundleError(RuntimeError):
    """Raised when repair provenance cannot be established exactly."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise RepairBundleError(f"{name} must be a lowercase SHA-256")
    return value


def _sha1(value: object, name: str) -> str:
    if not isinstance(value, str) or SHA1_RE.fullmatch(value) is None:
        raise RepairBundleError(f"{name} must be a lowercase 40-character commit")
    return value


def _regular_file(path: Path, name: str, *, root: Path | None = None) -> Path:
    raw = Path(path)
    if raw.is_symlink():
        raise RepairBundleError(f"{name} must not be a symbolic link")
    try:
        result = raw.resolve(strict=True)
    except OSError as error:
        raise RepairBundleError(f"{name} is missing or inaccessible") from error
    if not result.is_file():
        raise RepairBundleError(f"{name} must be a regular file")
    if root is not None:
        try:
            result.relative_to(root)
        except ValueError as error:
            raise RepairBundleError(f"{name} escapes {root}") from error
    return result


def _regular_directory(path: Path, name: str, *, root: Path | None = None) -> Path:
    raw = Path(path)
    if raw.is_symlink():
        raise RepairBundleError(f"{name} must not be a symbolic link")
    try:
        result = raw.resolve(strict=True)
    except OSError as error:
        raise RepairBundleError(f"{name} is missing or inaccessible") from error
    if not result.is_dir():
        raise RepairBundleError(f"{name} must be a directory")
    if root is not None:
        try:
            result.relative_to(root)
        except ValueError as error:
            raise RepairBundleError(f"{name} escapes {root}") from error
    return result


def _load_json(path: Path, name: str, *, root: Path | None = None) -> tuple[Path, dict[str, object]]:
    path = _regular_file(path, name, root=root)
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda item: (_ for _ in ()).throw(
                RepairBundleError(f"{name} contains non-finite {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RepairBundleError(f"{name} is not strict JSON") from error
    if not isinstance(value, dict):
        raise RepairBundleError(f"{name} must contain an object")
    return path, value


def _write_new(path: Path, payload: bytes, mode: int = 0o400) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        raise
    if path.read_bytes() != payload:
        raise RepairBundleError(f"create-only readback differs: {path}")


def _write_json_new(path: Path, value: object, mode: int = 0o400) -> str:
    try:
        payload = (
            json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RepairBundleError("output is not strict JSON") from error
    _write_new(path, payload, mode)
    return hashlib.sha256(payload).hexdigest()


def _verify_expected_file(path: Path, expected: object, name: str, *, root: Path) -> Path:
    path = _regular_file(path, name, root=root)
    if sha256_file(path) != _sha256(expected, f"expected {name} SHA-256"):
        raise RepairBundleError(f"{name} SHA-256 differs")
    return path


def _roots_overlap(first: Path, second: Path) -> bool:
    for candidate, parent in ((first, second), (second, first)):
        try:
            candidate.relative_to(parent)
        except ValueError:
            continue
        return True
    return False


def _preload_repair_sources(
    source_root: Path,
) -> tuple[dict[str, bytes], dict[str, bytes]]:
    """Resolve and read the complete repair closure before publishing paths."""

    code_targets = tuple(CODE_SOURCES)
    provenance_targets = tuple(PROVENANCE_SOURCES)
    if len(code_targets) != 9 or len(set(code_targets)) != len(code_targets):
        raise RepairBundleError("repair code source map must have nine unique targets")
    if len(provenance_targets) != 4 or len(set(provenance_targets)) != len(
        provenance_targets
    ):
        raise RepairBundleError("repair provenance source list must have four unique targets")
    for name in (*code_targets, *provenance_targets):
        relative = PurePosixPath(name)
        if relative.is_absolute() or len(relative.parts) != 1 or relative.name != name:
            raise RepairBundleError("repair source target name is unsafe")

    code_payloads: dict[str, bytes] = {}
    provenance_payloads: dict[str, bytes] = {}
    for destination_name, source_name in CODE_SOURCES.items():
        source = _regular_file(
            source_root / source_name, f"source {source_name}", root=source_root
        )
        try:
            code_payloads[destination_name] = source.read_bytes()
        except OSError as error:
            raise RepairBundleError(f"source {source_name} is unreadable") from error
    for destination_name in PROVENANCE_SOURCES:
        source = _regular_file(
            source_root / destination_name,
            f"repair provenance source {destination_name}",
            root=source_root,
        )
        try:
            provenance_payloads[destination_name] = source.read_bytes()
        except OSError as error:
            raise RepairBundleError(
                f"repair provenance source {destination_name} is unreadable"
            ) from error
    return code_payloads, provenance_payloads


def _inventory_rows(code_root: Path) -> list[dict[str, object]]:
    code_root = _regular_directory(code_root, "repair code root")
    if os.name == "posix" and stat.S_IMODE(code_root.stat().st_mode) != 0o500:
        raise RepairBundleError("repair code root mode must be exactly 0500")

    expected_names = set(CODE_SOURCES)
    try:
        entries = list(os.scandir(code_root))
    except OSError as error:
        raise RepairBundleError("repair code root is unreadable") from error
    if len(entries) != len(expected_names) or {entry.name for entry in entries} != expected_names:
        raise RepairBundleError("repair code root is not the exact nine-file closure")

    live: dict[str, Path] = {}
    for entry in entries:
        if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
            raise RepairBundleError(
                "repair code root contains a symlink or non-regular entry"
            )
        path = _regular_file(
            Path(entry.path), f"repair source {entry.name}", root=code_root
        )
        expected_mode = 0o500 if entry.name in EXECUTABLE_FILES else 0o400
        if (
            os.name == "posix"
            and stat.S_IMODE(entry.stat(follow_symlinks=False).st_mode)
            != expected_mode
        ):
            raise RepairBundleError(
                f"repair source {entry.name} mode must be exactly {expected_mode:04o}"
            )
        live[entry.name] = path

    rows: list[dict[str, object]] = []
    for name in sorted(CODE_SOURCES):
        path = live[name]
        rows.append(
            {"path": name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        )
    return rows


def _create_deterministic_tar(path: Path, code_root: Path) -> str:
    if os.path.lexists(path):
        raise FileExistsError(path)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for name in sorted(CODE_SOURCES):
            source = _regular_file(code_root / name, f"repair source {name}", root=code_root)
            data = source.read_bytes()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o500 if name in EXECUTABLE_FILES else 0o400
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            archive.addfile(info, io.BytesIO(data))
    _write_new(path, buffer.getvalue(), 0o400)
    return sha256_file(path)


def _tree_identity(rows: Sequence[Mapping[str, object]]) -> str:
    payload = json.dumps(list(rows), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _verify_tar(archive_path: Path, code_root: Path) -> list[dict[str, object]]:
    archive_path = _regular_file(archive_path, "repair code archive")
    expected = sorted(CODE_SOURCES)
    readback: list[dict[str, object]] = []
    try:
        with tarfile.open(archive_path, mode="r:") as archive:
            members = archive.getmembers()
            if [member.name for member in members] != expected:
                raise RepairBundleError("repair code archive member inventory differs")
            for member in members:
                if not member.isfile() or member.issym() or member.islnk():
                    raise RepairBundleError("repair code archive contains a non-regular member")
                expected_mode = (
                    0o500 if member.name in EXECUTABLE_FILES else 0o400
                )
                if stat.S_IMODE(member.mode) != expected_mode:
                    raise RepairBundleError(
                        f"repair code archive member {member.name} mode differs"
                    )
                stream = archive.extractfile(member)
                if stream is None:
                    raise RepairBundleError("repair code archive member is unreadable")
                live = _regular_file(code_root / member.name, "live repair source", root=code_root)
                payload = stream.read()
                if payload != live.read_bytes():
                    raise RepairBundleError("repair code archive readback differs from live source")
                readback.append(
                    {
                        "path": member.name,
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "size_bytes": len(payload),
                    }
                )
    except (tarfile.TarError, OSError) as error:
        raise RepairBundleError("repair code archive is unreadable") from error
    return readback


def materialize_code(args: argparse.Namespace) -> dict[str, object]:
    run_root = _regular_directory(args.run_root, "run root")
    source_root = _regular_directory(args.source_root, "repair source root")
    repair_commit = _sha1(args.navigation_code_commit, "repair code commit")
    superseded = _sha1(args.supersedes_navigation_code_commit, "superseded code commit")
    if repair_commit == superseded:
        raise RepairBundleError("repair code commit must differ from Stage1")
    original_root = _regular_directory(run_root / "code", "original code root", root=run_root)
    code_root = run_root / f"code-stage2-repair-{repair_commit}"
    repair_root = run_root / "provenance" / REPAIR_DIR_NAME
    if os.path.lexists(code_root) or os.path.lexists(repair_root):
        raise FileExistsError(code_root if os.path.lexists(code_root) else repair_root)
    if _roots_overlap(code_root.resolve(strict=False), original_root):
        raise RepairBundleError("repair and original code roots overlap")

    protocol_path = _verify_expected_file(
        args.base_training_protocol,
        args.expected_base_training_protocol_sha256,
        "base training protocol",
        root=run_root,
    )
    protocol_sha = sha256_file(protocol_path)
    _, protocol = _load_json(protocol_path, "base training protocol", root=run_root)
    if (
        protocol.get("schema_name") != "navigation_rlt_training_protocol"
        or protocol.get("schema_version") != 1
        or protocol.get("run_id") != args.run_id
    ):
        raise RepairBundleError("base training protocol identity differs")
    stage1_verification = _verify_expected_file(
        args.stage1_code_verification,
        args.expected_stage1_code_verification_sha256,
        "Stage1 code verification",
        root=run_root,
    )
    stage1_verification_sha = sha256_file(stage1_verification)
    _, stage1_value = _load_json(stage1_verification, "Stage1 code verification", root=run_root)
    if (
        stage1_value.get("schema_name") != CODE_VERIFICATION_SCHEMA
        or stage1_value.get("schema_version") != 1
        or stage1_value.get("status") != "passed"
        or stage1_value.get("navigation_code_commit") != superseded
    ):
        raise RepairBundleError("Stage1 code verification identity differs")
    if protocol.get("navigation_training_code_commit") != superseded:
        raise RepairBundleError("base protocol and Stage1 code commit differ")

    # Resolve and read the entire 9+4 source closure before creating either
    # destination.  A missing, unsafe, or unreadable late source must not leave
    # a partial immutable repair bundle behind.
    code_payloads, provenance_payloads = _preload_repair_sources(source_root)

    code_root.mkdir(mode=0o700)
    for destination_name, payload in code_payloads.items():
        mode = 0o500 if destination_name in EXECUTABLE_FILES else 0o400
        destination = code_root / destination_name
        _write_new(destination, payload, mode)
        destination.chmod(mode)
    code_root.chmod(0o500)
    rows = _inventory_rows(code_root)
    if len(rows) != 9:
        raise RepairBundleError("repair code bundle must contain exactly nine files")

    # The directory remains owner-writable because the repair protocol and the
    # downstream four-argument handoff are published later.  Every individual
    # provenance file is create-only and read-only after publication.
    repair_root.mkdir(mode=0o700, parents=True)
    provenance_rows: list[dict[str, object]] = []
    for name, payload in provenance_payloads.items():
        destination = repair_root / name
        _write_new(destination, payload, 0o500)
        destination.chmod(0o500)
        provenance_rows.append(
            {
                "path": name,
                "sha256": sha256_file(destination),
                "size_bytes": destination.stat().st_size,
                "mode": 0o500,
            }
        )
    launcher_inventory_path = repair_root / "launcher_inventory.json"
    launcher_inventory_sha = _write_json_new(
        launcher_inventory_path,
        {
            "schema_name": "navigation_rlt_stage2_repair_launcher_inventory",
            "schema_version": 1,
            "navigation_code_commit": repair_commit,
            "file_count": len(provenance_rows),
            "files": provenance_rows,
        },
    )
    archive_path = repair_root / f"code-stage2-repair-{repair_commit}.tar"
    archive_sha = _create_deterministic_tar(archive_path, code_root)
    archive_readback = _verify_tar(archive_path, code_root)
    if archive_readback != rows:
        raise RepairBundleError("repair code archive members differ from inventory")
    inventory_path = repair_root / "code_inventory.json"
    inventory = {
        "schema_name": CODE_INVENTORY_SCHEMA,
        "schema_version": 1,
        "navigation_code_commit": repair_commit,
        "archive_name": archive_path.name,
        "archive_sha256": archive_sha,
        "archive_size_bytes": archive_path.stat().st_size,
        "file_count": len(rows),
        "files": rows,
    }
    inventory_sha = _write_json_new(inventory_path, inventory)
    verification_path = repair_root / "code_verification.json"
    verification = {
        "schema_name": CODE_VERIFICATION_SCHEMA,
        "schema_version": 1,
        "status": "passed",
        "navigation_code_commit": repair_commit,
        "supersedes_navigation_code_commit": superseded,
        "code_root": str(code_root.resolve(strict=True)),
        "file_count": len(rows),
        "inventory_path": str(inventory_path.resolve(strict=True)),
        "inventory_sha256": inventory_sha,
        "archive_path": str(archive_path.resolve(strict=True)),
        "archive_sha256": archive_sha,
        "archive_readback": {
            "status": "passed",
            "member_count": len(archive_readback),
            "members": archive_readback,
            "tree_sha256": _tree_identity(archive_readback),
        },
    }
    verification_sha = _write_json_new(verification_path, verification)
    context_path = repair_root / "bundle_context.json"
    context = {
        "schema_name": "navigation_rlt_stage2_repair_bundle_context",
        "schema_version": 1,
        "status": "verified",
        "run_id": args.run_id,
        "base_training_protocol": {"path": str(protocol_path), "sha256": protocol_sha},
        "stage1_code_verification": {
            "path": str(stage1_verification),
            "sha256": stage1_verification_sha,
        },
        "stage2_repair_code_verification": {
            "path": str(verification_path.resolve(strict=True)),
            "sha256": verification_sha,
        },
        "navigation_training_code_commit": repair_commit,
        "launcher_inventory": {
            "path": str(launcher_inventory_path.resolve(strict=True)),
            "sha256": launcher_inventory_sha,
        },
        "validation_split_reads": 0,
    }
    context_sha = _write_json_new(context_path, context)
    verified = verify_bundle_paths(run_root, verification_path, verification_sha)
    return {
        "status": "materialized",
        "run_id": args.run_id,
        **verified,
        "bundle_context": {"path": str(context_path), "sha256": context_sha},
    }


def verify_bundle_paths(
    run_root: Path, verification_path: Path, expected_verification_sha: str
) -> dict[str, object]:
    run_root = _regular_directory(run_root, "run root")
    verification_path = _verify_expected_file(
        verification_path,
        expected_verification_sha,
        "repair code verification",
        root=run_root,
    )
    _, verification = _load_json(verification_path, "repair code verification", root=run_root)
    expected_fields = {
        "schema_name",
        "schema_version",
        "status",
        "navigation_code_commit",
        "supersedes_navigation_code_commit",
        "code_root",
        "file_count",
        "inventory_path",
        "inventory_sha256",
        "archive_path",
        "archive_sha256",
        "archive_readback",
    }
    if (
        set(verification) != expected_fields
        or verification.get("schema_name") != CODE_VERIFICATION_SCHEMA
        or verification.get("schema_version") != 1
        or verification.get("status") != "passed"
        or verification.get("file_count") != 9
    ):
        raise RepairBundleError("repair code verification contract differs")
    commit = _sha1(verification.get("navigation_code_commit"), "repair code commit")
    superseded = _sha1(
        verification.get("supersedes_navigation_code_commit"), "superseded code commit"
    )
    if commit == superseded:
        raise RepairBundleError("repair code commit must differ from Stage1")
    code_root = _regular_directory(Path(str(verification["code_root"])), "repair code root", root=run_root)
    original_root = _regular_directory(run_root / "code", "original code root", root=run_root)
    if _roots_overlap(code_root, original_root):
        raise RepairBundleError("repair code root is not independent")
    if code_root.name != f"code-stage2-repair-{commit}":
        raise RepairBundleError("repair code root name does not bind the repair commit")
    live_rows = _inventory_rows(code_root)
    inventory_path = _verify_expected_file(
        Path(str(verification["inventory_path"])),
        verification.get("inventory_sha256"),
        "repair code inventory",
        root=run_root,
    )
    _, inventory = _load_json(inventory_path, "repair code inventory", root=run_root)
    rows = inventory.get("files")
    inventory_fields = {
        "schema_name",
        "schema_version",
        "navigation_code_commit",
        "archive_name",
        "archive_sha256",
        "archive_size_bytes",
        "file_count",
        "files",
    }
    if (
        set(inventory) != inventory_fields
        or inventory.get("schema_name") != CODE_INVENTORY_SCHEMA
        or inventory.get("schema_version") != 1
        or inventory.get("navigation_code_commit") != commit
        or inventory.get("file_count") != 9
        or type(inventory.get("archive_size_bytes")) is not int
        or inventory.get("archive_size_bytes", -1) <= 0
        or not isinstance(rows, list)
        or len(rows) != 9
    ):
        raise RepairBundleError("repair code inventory identity/count differs")
    _sha256(inventory.get("archive_sha256"), "repair code archive SHA-256")
    indexed: dict[str, Mapping[str, object]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"path", "sha256", "size_bytes"}:
            raise RepairBundleError("repair code inventory row is invalid")
        raw_path = row.get("path")
        size_bytes = row.get("size_bytes")
        if (
            not isinstance(raw_path, str)
            or type(size_bytes) is not int
            or size_bytes < 0
        ):
            raise RepairBundleError("repair code inventory path is invalid")
        _sha256(row.get("sha256"), f"source {raw_path} SHA-256")
        relative = PurePosixPath(raw_path)
        if (
            relative.is_absolute()
            or relative.as_posix() != raw_path
            or any(part in {"", ".", ".."} for part in relative.parts)
            or raw_path in indexed
        ):
            raise RepairBundleError("repair code inventory path is invalid")
        indexed[raw_path] = row
    if set(indexed) != set(CODE_SOURCES):
        raise RepairBundleError("repair code inventory is not the exact nine-file closure")
    expected_rows = [dict(indexed[name]) | {"path": name} for name in sorted(indexed)]
    if live_rows != expected_rows:
        raise RepairBundleError("live repair source differs from code inventory")
    for name, row in indexed.items():
        live = _regular_file(code_root / name, f"live repair source {name}", root=code_root)
        if (
            sha256_file(live) != _sha256(row.get("sha256"), f"source {name} SHA-256")
            or live.stat().st_size != row.get("size_bytes")
        ):
            raise RepairBundleError("live repair source differs from code inventory")
    archive_path = _verify_expected_file(
        Path(str(verification["archive_path"])),
        verification.get("archive_sha256"),
        "repair code archive",
        root=run_root,
    )
    if (
        inventory.get("archive_name") != archive_path.name
        or inventory.get("archive_sha256") != sha256_file(archive_path)
        or inventory.get("archive_size_bytes") != archive_path.stat().st_size
    ):
        raise RepairBundleError("repair code archive identity differs")
    archive_readback = _verify_tar(archive_path, code_root)
    # Reorder each row to the same logical mapping; mapping equality is order-independent.
    if archive_readback != expected_rows:
        raise RepairBundleError("repair code archive members differ from inventory")
    recorded_readback = verification.get("archive_readback")
    expected_readback = {
        "status": "passed",
        "member_count": 9,
        "members": archive_readback,
        "tree_sha256": _tree_identity(archive_readback),
    }
    if recorded_readback != expected_readback:
        raise RepairBundleError("repair code archive independent readback differs")
    return {
        "code_root": str(code_root),
        "navigation_training_code_commit": commit,
        "code_verification": {
            "path": str(verification_path), "sha256": sha256_file(verification_path)
        },
        "code_inventory": {"path": str(inventory_path), "sha256": sha256_file(inventory_path)},
        "code_archive": {"path": str(archive_path), "sha256": sha256_file(archive_path)},
        "file_count": 9,
    }


def _support_contract(value: object) -> dict[str, object]:
    fields = {
        "eligibility_schema",
        "gate_names",
        "gate_counts",
        "gate_support_mask",
        "minimum_behavior_gate_count",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise RepairBundleError("Stage2 support fields differ")
    counts = value.get("gate_counts")
    mask = value.get("gate_support_mask")
    minimum = value.get("minimum_behavior_gate_count")
    if (
        value.get("eligibility_schema") != ELIGIBILITY_SCHEMA
        or value.get("gate_names") != GATE_NAMES
        or type(minimum) is not int
        or minimum != 1
        or not isinstance(counts, list)
        or len(counts) != 4
        or any(type(item) is not int or item < 0 for item in counts)
        or not isinstance(mask, list)
        or len(mask) != 4
        or any(type(item) is not bool for item in mask)
        or mask != [count >= minimum for count in counts]
        or not any(mask)
    ):
        raise RepairBundleError("Stage2 support/count contract differs")
    if counts[3] != 0 or mask[3] is not False:
        raise RepairBundleError("RECOVER must have zero behavior support before Stage2")
    return dict(value)


def _prepared_contract(
    run_root: Path,
    prepare_dir: Path,
    *,
    run_id: str,
    checkpoint_sha: str,
    replay_sha: str,
) -> tuple[Path, str, dict[str, object], dict[str, object]]:
    prepare_dir = _regular_directory(prepare_dir, "Stage2 prepare directory", root=run_root)
    marker_path, marker = _load_json(prepare_dir / "COMPLETED.json", "Stage2 prepare completion", root=run_root)
    manifest_path, manifest = _load_json(prepare_dir / "run_manifest.json", "Stage2 run manifest", root=run_root)
    receipt_path, receipt = _load_json(prepare_dir / "prepare_receipt.json", "Stage2 prepare receipt", root=run_root)
    manifest_sha = sha256_file(manifest_path)
    receipt_sha = sha256_file(receipt_path)
    if marker != {
        "schema_name": "navigation_rlt_prepare_training",
        "schema_version": 1,
        "status": "completed",
        "operation": "stage2",
        "artifact_kind": "run_manifest",
        "artifact_sha256": manifest_sha,
        "receipt_sha256": receipt_sha,
    }:
        raise RepairBundleError("Stage2 prepare completion differs")
    training = manifest.get("training")
    support = _support_contract(manifest.get("stage2_support"))
    if not isinstance(training, Mapping):
        raise RepairBundleError("Stage2 training contract is absent")
    additions = {key: training.get(key) for key in TRAINING_ADDITIONS}
    if additions != TRAINING_ADDITIONS:
        raise RepairBundleError("Stage2 repair objective/CQL contract differs")
    boundaries = manifest.get("boundaries")
    if (
        manifest.get("schema_name") != "navigation_rlt_training_run_manifest"
        or manifest.get("schema_version") != 1
        or manifest.get("run_id") != run_id
        or manifest.get("phase") != "stage2"
        or manifest.get("required_world_size") != 8
        or manifest.get("stage1_checkpoint") != {"sha256": checkpoint_sha}
        or not isinstance(manifest.get("replay"), Mapping)
        or manifest["replay"].get("manifest_sha256") != replay_sha
        or not isinstance(boundaries, Mapping)
        or boundaries.get("upstream_split_read") != "train"
        or boundaries.get("split_label") != "train_update"
        or boundaries.get("validation_split_reads") != 0
        or receipt.get("schema_name") != "navigation_rlt_prepare_training"
        or receipt.get("status") != "completed"
        or receipt.get("operation") != "stage2"
        or receipt.get("run_id") != run_id
        or receipt.get("stage1_checkpoint", {}).get("sha256") != checkpoint_sha
        or receipt.get("replay", {}).get("manifest_sha256") != replay_sha
        or receipt.get("training") != dict(training)
        or receipt.get("data", {}).get("gate_support_mask") != support["gate_support_mask"]
    ):
        raise RepairBundleError("Stage2 prepared input lineage differs")
    return manifest_path, manifest_sha, dict(manifest), support


def _context(run_root: Path, run_id: str) -> tuple[Path, dict[str, object]]:
    path, value = _load_json(
        run_root / "provenance" / REPAIR_DIR_NAME / "bundle_context.json",
        "repair bundle context",
        root=run_root,
    )
    expected_fields = {
        "schema_name",
        "schema_version",
        "status",
        "run_id",
        "base_training_protocol",
        "stage1_code_verification",
        "stage2_repair_code_verification",
        "navigation_training_code_commit",
        "launcher_inventory",
        "validation_split_reads",
    }
    if (
        set(value) != expected_fields
        or value.get("schema_name") != "navigation_rlt_stage2_repair_bundle_context"
        or value.get("schema_version") != 1
        or value.get("status") != "verified"
        or value.get("run_id") != run_id
        or value.get("validation_split_reads") != 0
    ):
        raise RepairBundleError("repair bundle context identity differs")
    _sha1(value.get("navigation_training_code_commit"), "repair bundle context commit")
    for name in (
        "base_training_protocol",
        "stage1_code_verification",
        "stage2_repair_code_verification",
        "launcher_inventory",
    ):
        reference = value.get(name)
        if (
            not isinstance(reference, Mapping)
            or set(reference) != {"path", "sha256"}
            or not isinstance(reference.get("path"), str)
        ):
            raise RepairBundleError(f"repair bundle context {name} reference differs")
        _sha256(reference.get("sha256"), f"repair bundle context {name} SHA-256")
    return path, value


def _verify_launcher_inventory(
    run_root: Path, context: Mapping[str, object]
) -> dict[str, object]:
    repair_root = _regular_directory(
        run_root / "provenance" / REPAIR_DIR_NAME,
        "repair provenance root",
        root=run_root,
    )
    reference = context.get("launcher_inventory")
    if not isinstance(reference, Mapping):
        raise RepairBundleError("repair launcher inventory reference is absent")
    inventory_path = _verify_expected_file(
        Path(str(reference.get("path"))),
        reference.get("sha256"),
        "repair launcher inventory",
        root=run_root,
    )
    if inventory_path != (repair_root / "launcher_inventory.json").resolve(strict=True):
        raise RepairBundleError("repair launcher inventory path differs")
    _, inventory = _load_json(
        inventory_path, "repair launcher inventory", root=run_root
    )
    expected_fields = {
        "schema_name",
        "schema_version",
        "navigation_code_commit",
        "file_count",
        "files",
    }
    rows = inventory.get("files")
    if (
        set(inventory) != expected_fields
        or inventory.get("schema_name")
        != "navigation_rlt_stage2_repair_launcher_inventory"
        or inventory.get("schema_version") != 1
        or inventory.get("navigation_code_commit")
        != context.get("navigation_training_code_commit")
        or inventory.get("file_count") != 4
        or not isinstance(rows, list)
        or len(rows) != 4
    ):
        raise RepairBundleError("repair launcher inventory identity/count differs")

    indexed: dict[str, Mapping[str, object]] = {}
    for row in rows:
        if (
            not isinstance(row, Mapping)
            or set(row) != {"path", "sha256", "size_bytes", "mode"}
            or not isinstance(row.get("path"), str)
            or type(row.get("size_bytes")) is not int
            or row.get("size_bytes", -1) < 0
            or type(row.get("mode")) is not int
            or row.get("mode") != 0o500
            or row.get("path") in indexed
        ):
            raise RepairBundleError("repair launcher inventory row differs")
        name = str(row["path"])
        relative = PurePosixPath(name)
        if relative.is_absolute() or len(relative.parts) != 1 or relative.name != name:
            raise RepairBundleError("repair launcher inventory path is unsafe")
        _sha256(row.get("sha256"), f"repair launcher {name} SHA-256")
        indexed[name] = row
    if set(indexed) != set(PROVENANCE_SOURCES):
        raise RepairBundleError("repair launcher inventory file closure differs")

    verified_rows: list[dict[str, object]] = []
    for name in sorted(indexed):
        row = indexed[name]
        source = _regular_file(
            repair_root / name, f"repair launcher {name}", root=repair_root
        )
        actual_mode = stat.S_IMODE(source.stat().st_mode)
        if (
            source.stat().st_size != row["size_bytes"]
            or sha256_file(source) != row["sha256"]
            or (os.name == "posix" and actual_mode != row["mode"])
        ):
            raise RepairBundleError(f"repair launcher {name} differs from inventory")
        verified_rows.append(dict(row))
    return {
        "path": str(inventory_path),
        "sha256": sha256_file(inventory_path),
        "file_count": 4,
        "files": verified_rows,
    }


def inspect_stage1_command(args: argparse.Namespace) -> dict[str, object]:
    """Validate the completed Stage1 gate and its 256-episode parent replay."""
    run_root = _regular_directory(args.run_root, "run root")
    stage_path, stage = _load_json(
        run_root / "navigation_rlt_stage1" / "STAGE.json",
        "Stage1 marker",
        root=run_root,
    )
    checkpoint_ref = stage.get("output", {}).get("checkpoint")
    verification_ref = stage.get("verification")
    if (
        stage.get("schema_name") != "navigation_rlt_a800_stage"
        or stage.get("schema_version") != 1
        or stage.get("status") != "completed"
        or stage.get("exit_code") != 0
        or stage.get("phase") != "stage1"
        or Path(str(stage.get("run_root"))).resolve(strict=True) != run_root
        or not isinstance(checkpoint_ref, Mapping)
        or not isinstance(verification_ref, Mapping)
    ):
        raise RepairBundleError("Stage1 is absent, partial, failed, or mismatched")
    checkpoint = _verify_expected_file(
        Path(str(checkpoint_ref.get("path"))),
        checkpoint_ref.get("sha256"),
        "Stage1 checkpoint",
        root=run_root,
    )
    verification = _verify_expected_file(
        Path(str(verification_ref.get("path"))),
        verification_ref.get("sha256"),
        "Stage1 verification",
        root=run_root,
    )
    _, verification_value = _load_json(verification, "Stage1 verification", root=run_root)
    if (
        verification_value.get("schema_name") != "navigation_rlt_a800_verification"
        or verification_value.get("schema_version") != 1
        or verification_value.get("status") != "passed"
        or verification_value.get("phase") != "stage1"
        or verification_value.get("world_size") != 8
        or verification_value.get("unique_gpu_uuid_count") != 8
        or verification_value.get("checkpoint")
        != {"path": str(checkpoint), "sha256": sha256_file(checkpoint)}
    ):
        raise RepairBundleError("Stage1 eight-rank verification differs")
    run_ref = stage.get("run_manifest")
    replay_ref = stage.get("replay_manifest")
    if not isinstance(run_ref, Mapping) or not isinstance(replay_ref, Mapping):
        raise RepairBundleError("Stage1 marker lacks run/replay provenance")
    _verify_expected_file(
        Path(str(run_ref.get("path"))), run_ref.get("sha256"), "Stage1 run manifest", root=run_root
    )
    _verify_expected_file(
        Path(str(replay_ref.get("path"))), replay_ref.get("sha256"), "Stage1 replay manifest", root=run_root
    )

    gate_marker_path, gate_marker = _load_json(
        run_root / "navigation_rlt_stage1_gate" / "COMPLETED.json",
        "Stage1 gate completion",
        root=run_root,
    )
    gate_receipt_path, gate_receipt = _load_json(
        run_root / "navigation_rlt_stage1_gate" / "stage1_gate_receipt.json",
        "Stage1 gate receipt",
        root=run_root,
    )
    gate_receipt_sha = sha256_file(gate_receipt_path)
    if gate_marker != {
        "schema_name": "navigation_rlt_stage1_gate",
        "schema_version": 1,
        "status": "completed",
        "operation": "evaluate",
        "gate_passed": True,
        "receipt_sha256": gate_receipt_sha,
        "checkpoint_sha256": sha256_file(checkpoint),
    }:
        raise RepairBundleError("Stage1 gate is not an exact PASS")
    gate_checkpoint = gate_receipt.get("checkpoint")
    gate_split = gate_receipt.get("split")
    if (
        gate_receipt.get("schema_name") != "navigation_rlt_stage1_gate"
        or gate_receipt.get("schema_version") != 1
        or gate_receipt.get("status") != "passed"
        or gate_receipt.get("operation") != "evaluate"
        or not isinstance(gate_checkpoint, Mapping)
        or gate_checkpoint.get("sha256") != sha256_file(checkpoint)
        or gate_checkpoint.get("run_manifest_sha256") != run_ref.get("sha256")
        or gate_checkpoint.get("replay_manifest_sha256") != replay_ref.get("sha256")
        or not isinstance(gate_split, Mapping)
        or gate_split.get("train_manifest_sha256") != replay_ref.get("sha256")
        or gate_split.get("train_episode_count") != 224
        or gate_split.get("dev_episode_count") != 32
        or gate_split.get("episode_overlap_count") != 0
    ):
        raise RepairBundleError("Stage1 gate receipt lineage differs")

    full_path, full = _load_json(
        run_root / "navigation_rlt_prefix_relocated" / "replay_manifest.json",
        "full prefix replay manifest",
        root=run_root,
    )
    relocation_path, relocation = _load_json(
        run_root / "navigation_rlt_prefix_relocated" / "COMPLETED.json",
        "full prefix relocation completion",
        root=run_root,
    )
    full_sha = sha256_file(full_path)
    if (
        relocation.get("schema_name") != "navigation_rlt_replay_relocation"
        or relocation.get("status") != "completed"
        or relocation.get("run_id") != args.run_id
        or relocation.get("relocated_manifest_sha256") != full_sha
        or full.get("schema_name") != "navigation_rlt_replay_manifest"
        or full.get("schema_version") != 3
        or full.get("run_id") != args.run_id
        or full.get("representation_kind") != "prefix_hidden"
        or full.get("expected_episode_count") != 256
        or full.get("completed_episode_count") != 256
        or full.get("split_label") != "train_update"
        or full.get("validation_split_reads") != 0
        or gate_split.get("parent_replay_manifest_sha256") != full_sha
        or gate_split.get("source_fingerprint") != full.get("source_fingerprint")
    ):
        raise RepairBundleError("full 256-episode parent replay lineage differs")
    return {
        "status": "passed",
        "stage_marker": {"path": str(stage_path), "sha256": sha256_file(stage_path)},
        "stage1_checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
        "stage1_run_manifest_sha256": run_ref.get("sha256"),
        "stage1_replay_manifest_sha256": replay_ref.get("sha256"),
        "stage1_gate_completion": {"path": str(gate_marker_path), "sha256": sha256_file(gate_marker_path)},
        "stage1_gate_receipt": {"path": str(gate_receipt_path), "sha256": gate_receipt_sha},
        "full_prefix_replay": {
            "path": str(full_path),
            "sha256": full_sha,
            "source_fingerprint": full.get("source_fingerprint"),
        },
        "relocation_completion": {"path": str(relocation_path), "sha256": sha256_file(relocation_path)},
        "validation_split_reads": 0,
    }


def verify_promotion_command(args: argparse.Namespace) -> dict[str, object]:
    run_root = _regular_directory(args.run_root, "run root")
    stage1 = inspect_stage1_command(argparse.Namespace(run_root=run_root, run_id=args.run_id))
    output = _regular_directory(args.promotion_dir, "Stage2 promotion directory", root=run_root)
    marker_path, marker = _load_json(output / "COMPLETED.json", "promotion completion", root=run_root)
    manifest_path, manifest = _load_json(output / "replay_manifest.json", "promoted replay manifest", root=run_root)
    receipt_path, receipt = _load_json(output / "promotion_receipt.json", "promotion receipt", root=run_root)
    manifest_sha = sha256_file(manifest_path)
    receipt_sha = sha256_file(receipt_path)
    checkpoint_sha = stage1["stage1_checkpoint"]["sha256"]
    gate_marker_sha = stage1["stage1_gate_completion"]["sha256"]
    gate_receipt_sha = stage1["stage1_gate_receipt"]["sha256"]
    if marker != {
        "schema_name": "navigation_rlt_replay_promotion",
        "schema_version": 1,
        "status": "completed",
        "promotion_receipt_sha256": receipt_sha,
        "output_manifest_sha256": manifest_sha,
        "stage1_checkpoint_sha256": checkpoint_sha,
        "stage1_gate_marker_sha256": gate_marker_sha,
        "stage1_gate_receipt_sha256": gate_receipt_sha,
    }:
        raise RepairBundleError("Stage2 promotion completion differs")
    if (
        manifest.get("schema_name") != "navigation_rlt_replay_manifest"
        or manifest.get("schema_version") != 3
        or manifest.get("run_id") != args.run_id
        or manifest.get("representation_kind") != "frozen_z_rl"
        or manifest.get("expected_episode_count") != 256
        or manifest.get("completed_episode_count") != 256
        or manifest.get("split_label") != "train_update"
        or manifest.get("validation_split_reads") != 0
        or manifest.get("behavior_policy_id") != "navigation_rlt_warmstart_lineage_v1"
        or manifest.get("behavior_policy_versions") != [100001]
        or receipt.get("schema_name") != "navigation_rlt_replay_promotion"
        or receipt.get("schema_version") != 1
        or receipt.get("status") != "completed"
        or receipt.get("stage1_checkpoint", {}).get("sha256") != checkpoint_sha
        or receipt.get("stage1_gate", {}).get("marker_sha256") != gate_marker_sha
        or receipt.get("stage1_gate", {}).get("receipt_sha256") != gate_receipt_sha
        or receipt.get("input", {}).get("manifest_sha256")
        != stage1["full_prefix_replay"]["sha256"]
        or receipt.get("output", {}).get("manifest_sha256") != manifest_sha
    ):
        raise RepairBundleError("Stage2 promotion lineage differs")
    return {
        "status": "passed",
        "manifest": {"path": str(manifest_path), "sha256": manifest_sha},
        "completion": {"path": str(marker_path), "sha256": sha256_file(marker_path)},
        "receipt": {"path": str(receipt_path), "sha256": receipt_sha},
        "source_fingerprint": manifest.get("source_fingerprint"),
        "stage1": stage1,
        "validation_split_reads": 0,
    }


def verify_admission_command(args: argparse.Namespace) -> dict[str, object]:
    """Verify every pre-existing repair input without creating any artifact."""

    run_root = _regular_directory(args.run_root, "run root")
    context_path, context = _context(run_root, args.run_id)
    launcher_inventory = _verify_launcher_inventory(run_root, context)

    base_ref = context["base_training_protocol"]
    stage1_code_ref = context["stage1_code_verification"]
    repair_code_ref = context["stage2_repair_code_verification"]
    assert isinstance(base_ref, Mapping)
    assert isinstance(stage1_code_ref, Mapping)
    assert isinstance(repair_code_ref, Mapping)
    base_path = _verify_expected_file(
        Path(str(base_ref["path"])),
        base_ref["sha256"],
        "base training protocol",
        root=run_root,
    )
    stage1_code_path = _verify_expected_file(
        Path(str(stage1_code_ref["path"])),
        stage1_code_ref["sha256"],
        "Stage1 code verification",
        root=run_root,
    )
    repair_code_path = _verify_expected_file(
        Path(str(repair_code_ref["path"])),
        repair_code_ref["sha256"],
        "Stage2 repair code verification",
        root=run_root,
    )
    _, base = _load_json(base_path, "base training protocol", root=run_root)
    _, stage1_code = _load_json(
        stage1_code_path, "Stage1 code verification", root=run_root
    )
    repair_bundle = verify_bundle_paths(
        run_root, repair_code_path, str(repair_code_ref["sha256"])
    )
    if (
        base.get("schema_name") != "navigation_rlt_training_protocol"
        or base.get("schema_version") != 1
        or base.get("run_id") != args.run_id
        or stage1_code.get("schema_name") != CODE_VERIFICATION_SCHEMA
        or stage1_code.get("schema_version") != 1
        or stage1_code.get("status") != "passed"
        or stage1_code.get("navigation_code_commit")
        != base.get("navigation_training_code_commit")
        or repair_bundle["navigation_training_code_commit"]
        != context.get("navigation_training_code_commit")
    ):
        raise RepairBundleError("repair admission code lineage differs")

    stage1 = inspect_stage1_command(
        argparse.Namespace(run_root=run_root, run_id=args.run_id)
    )
    checkpoint_path = Path(str(stage1["stage1_checkpoint"]["path"]))
    checkpoint_sha = str(stage1["stage1_checkpoint"]["sha256"])
    promotion_dir = run_root / "navigation_rlt_stage2_replay"
    prepare_dir = run_root / "navigation_rlt_stage2_prepare"
    protocol_path = run_root / "provenance" / REPAIR_DIR_NAME / "stage2_repair_protocol.json"

    promotion: dict[str, object] | None = None
    if os.path.lexists(promotion_dir):
        _regular_directory(
            promotion_dir, "existing Stage2 promotion directory", root=run_root
        )
        promotion = verify_promotion_command(
            argparse.Namespace(
                run_root=run_root,
                run_id=args.run_id,
                promotion_dir=promotion_dir,
            )
        )

    prepared: dict[str, object] | None = None
    if os.path.lexists(prepare_dir):
        _regular_directory(prepare_dir, "existing Stage2 prepare directory", root=run_root)
        if promotion is None:
            raise RepairBundleError(
                "Stage2 prepare exists without an exact completed promotion"
            )
        promoted_manifest = Path(str(promotion["manifest"]["path"]))
        promoted_sha = str(promotion["manifest"]["sha256"])
        manifest_path, manifest_sha, _, support = _prepared_contract(
            run_root,
            prepare_dir,
            run_id=args.run_id,
            checkpoint_sha=checkpoint_sha,
            replay_sha=promoted_sha,
        )
        prepared = {
            "path": str(prepare_dir.resolve(strict=True)),
            "run_manifest": {"path": str(manifest_path), "sha256": manifest_sha},
            "stage2_support": support,
            "stage2_replay_manifest": {
                "path": str(promoted_manifest),
                "sha256": promoted_sha,
            },
        }

    protocol: dict[str, object] | None = None
    if os.path.lexists(protocol_path):
        if promotion is None or prepared is None:
            raise RepairBundleError(
                "Stage2 repair protocol exists without exact promotion and prepare"
            )
        protocol_file = _regular_file(
            protocol_path, "existing Stage2 repair protocol", root=run_root
        )
        protocol = verify_protocol(
            run_root=run_root,
            run_id=args.run_id,
            protocol_path=protocol_file,
            expected_protocol_sha=sha256_file(protocol_file),
            stage1_checkpoint=checkpoint_path,
            expected_checkpoint_sha=checkpoint_sha,
            replay_manifest=Path(
                str(prepared["stage2_replay_manifest"]["path"])
            ),
            expected_replay_sha=str(
                prepared["stage2_replay_manifest"]["sha256"]
            ),
            prepare_dir=prepare_dir,
        )

    return {
        "status": "passed",
        "run_id": args.run_id,
        "bundle_context": {
            "path": str(context_path),
            "sha256": sha256_file(context_path),
        },
        "launcher_inventory": launcher_inventory,
        "repair_code": repair_bundle,
        "stage1": stage1,
        "existing": {
            "promotion": "completed" if promotion is not None else "absent",
            "prepare": "completed" if prepared is not None else "absent",
            "protocol": "completed" if protocol is not None else "absent",
        },
        "validation_split_reads": 0,
    }


def verify_stage2_command(args: argparse.Namespace) -> dict[str, object]:
    run_root = _regular_directory(args.run_root, "run root")
    protocol_result = verify_protocol(
        run_root=run_root,
        run_id=args.run_id,
        protocol_path=args.stage2_repair_protocol,
        expected_protocol_sha=args.expected_stage2_repair_protocol_sha256,
        stage1_checkpoint=args.stage1_checkpoint,
        expected_checkpoint_sha=args.expected_stage1_checkpoint_sha256,
        replay_manifest=args.stage2_replay_manifest,
        expected_replay_sha=args.expected_stage2_replay_manifest_sha256,
        prepare_dir=args.stage2_prepare_dir,
    )
    marker_path, marker = _load_json(
        run_root / "navigation_rlt_stage2" / "STAGE.json", "Stage2 marker", root=run_root
    )
    checkpoint_ref = marker.get("output", {}).get("checkpoint")
    verification_ref = marker.get("verification")
    if (
        marker.get("schema_name") != "navigation_rlt_a800_stage"
        or marker.get("schema_version") != 1
        or marker.get("status") != "completed"
        or marker.get("exit_code") != 0
        or marker.get("phase") != "stage2"
        or marker.get("run_manifest", {}).get("sha256")
        != protocol_result["run_manifest"]["sha256"]
        or marker.get("replay_manifest", {}).get("sha256")
        != protocol_result["stage2_replay_manifest"]["sha256"]
        or marker.get("stage1_checkpoint", {}).get("sha256")
        != protocol_result["stage1_checkpoint"]["sha256"]
        or not isinstance(checkpoint_ref, Mapping)
        or not isinstance(verification_ref, Mapping)
    ):
        raise RepairBundleError("Stage2 is absent, partial, failed, or mismatched")
    checkpoint = _verify_expected_file(
        Path(str(checkpoint_ref.get("path"))), checkpoint_ref.get("sha256"), "Stage2 checkpoint", root=run_root
    )
    verification = _verify_expected_file(
        Path(str(verification_ref.get("path"))), verification_ref.get("sha256"), "Stage2 verification", root=run_root
    )
    _, verification_value = _load_json(verification, "Stage2 verification", root=run_root)
    if (
        verification_value.get("schema_name") != "navigation_rlt_a800_verification"
        or verification_value.get("schema_version") != 1
        or verification_value.get("status") != "passed"
        or verification_value.get("phase") != "stage2"
        or verification_value.get("world_size") != 8
        or verification_value.get("unique_gpu_uuid_count") != 8
        or verification_value.get("checkpoint")
        != {"path": str(checkpoint), "sha256": sha256_file(checkpoint)}
    ):
        raise RepairBundleError("Stage2 eight-rank verification differs")
    return {
        "status": "completed",
        "stage_marker": {"path": str(marker_path), "sha256": sha256_file(marker_path)},
        "checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
        "verification": {"path": str(verification), "sha256": sha256_file(verification)},
        "repair_protocol": protocol_result["protocol"],
        "validation_split_reads": 0,
    }


def write_protocol(args: argparse.Namespace) -> dict[str, object]:
    run_root = _regular_directory(args.run_root, "run root")
    _, context = _context(run_root, args.run_id)
    base_ref = context.get("base_training_protocol")
    stage1_ref = context.get("stage1_code_verification")
    repair_ref = context.get("stage2_repair_code_verification")
    if not all(isinstance(value, Mapping) for value in (base_ref, stage1_ref, repair_ref)):
        raise RepairBundleError("repair bundle context references are incomplete")
    base_path = _verify_expected_file(Path(str(base_ref["path"])), base_ref.get("sha256"), "base training protocol", root=run_root)
    stage1_code_path = _verify_expected_file(Path(str(stage1_ref["path"])), stage1_ref.get("sha256"), "Stage1 code verification", root=run_root)
    repair_code_path = _verify_expected_file(Path(str(repair_ref["path"])), repair_ref.get("sha256"), "repair code verification", root=run_root)
    repair_bundle = verify_bundle_paths(run_root, repair_code_path, str(repair_ref["sha256"]))
    _, base = _load_json(base_path, "base training protocol", root=run_root)
    _, stage1_code = _load_json(stage1_code_path, "Stage1 code verification", root=run_root)
    if (
        base.get("schema_name") != "navigation_rlt_training_protocol"
        or base.get("schema_version") != 1
        or base.get("run_id") != args.run_id
        or stage1_code.get("schema_name") != CODE_VERIFICATION_SCHEMA
        or stage1_code.get("status") != "passed"
        or stage1_code.get("navigation_code_commit") != base.get("navigation_training_code_commit")
        or repair_bundle["navigation_training_code_commit"]
        == stage1_code.get("navigation_code_commit")
    ):
        raise RepairBundleError("base/Stage1/repair code lineage differs")
    checkpoint = _verify_expected_file(args.stage1_checkpoint, args.expected_stage1_checkpoint_sha256, "Stage1 checkpoint", root=run_root)
    replay = _verify_expected_file(args.stage2_replay_manifest, args.expected_stage2_replay_manifest_sha256, "Stage2 replay manifest", root=run_root)
    _, replay_value = _load_json(replay, "Stage2 replay manifest", root=run_root)
    if (
        replay_value.get("schema_name") != "navigation_rlt_replay_manifest"
        or replay_value.get("schema_version") != 3
        or replay_value.get("run_id") != args.run_id
        or replay_value.get("representation_kind") != "frozen_z_rl"
        or replay_value.get("expected_episode_count") != 256
        or replay_value.get("completed_episode_count") != 256
        or replay_value.get("split_label") != "train_update"
        or replay_value.get("validation_split_reads") != 0
    ):
        raise RepairBundleError("Stage2 replay is not the completed 256-episode train package")
    manifest_path, manifest_sha, _, support = _prepared_contract(
        run_root,
        args.stage2_prepare_dir,
        run_id=args.run_id,
        checkpoint_sha=sha256_file(checkpoint),
        replay_sha=sha256_file(replay),
    )
    protocol_path = run_root / "provenance" / REPAIR_DIR_NAME / "stage2_repair_protocol.json"
    protocol = {
        "schema_name": REPAIR_PROTOCOL_SCHEMA,
        "schema_version": 1,
        "run_id": args.run_id,
        "base_training_protocol_sha256": sha256_file(base_path),
        "stage1_code_verification_sha256": sha256_file(stage1_code_path),
        "stage2_repair_code_verification_sha256": sha256_file(repair_code_path),
        "navigation_training_code_commit": repair_bundle["navigation_training_code_commit"],
        "stage1_checkpoint_sha256": sha256_file(checkpoint),
        "stage2_replay_manifest_sha256": sha256_file(replay),
        "stage2_run_manifest_sha256": manifest_sha,
        "validation_split_reads": 0,
        "stage2_training_additions": dict(TRAINING_ADDITIONS),
        "stage2_support": support,
    }
    protocol_sha = _write_json_new(protocol_path, protocol)
    return {
        "status": "completed",
        "protocol": {"path": str(protocol_path), "sha256": protocol_sha},
        "run_manifest": {"path": str(manifest_path), "sha256": manifest_sha},
        "stage1_checkpoint_sha256": sha256_file(checkpoint),
        "stage2_replay_manifest_sha256": sha256_file(replay),
        "stage2_run_manifest_sha256": manifest_sha,
        "stage2_support": support,
    }


def verify_code_command(args: argparse.Namespace) -> dict[str, object]:
    run_root = _regular_directory(args.run_root, "run root")
    result = verify_bundle_paths(
        run_root,
        args.stage2_repair_code_verification,
        args.expected_stage2_repair_code_verification_sha256,
    )
    return {"status": "passed", **result}


def verify_protocol(
    *,
    run_root: Path,
    run_id: str,
    protocol_path: Path,
    expected_protocol_sha: str,
    stage1_checkpoint: Path,
    expected_checkpoint_sha: str,
    replay_manifest: Path,
    expected_replay_sha: str,
    prepare_dir: Path,
) -> dict[str, object]:
    protocol_path = _verify_expected_file(protocol_path, expected_protocol_sha, "Stage2 repair protocol", root=run_root)
    _, protocol = _load_json(protocol_path, "Stage2 repair protocol", root=run_root)
    expected_fields = {
        "schema_name",
        "schema_version",
        "run_id",
        "base_training_protocol_sha256",
        "stage1_code_verification_sha256",
        "stage2_repair_code_verification_sha256",
        "navigation_training_code_commit",
        "stage1_checkpoint_sha256",
        "stage2_replay_manifest_sha256",
        "stage2_run_manifest_sha256",
        "validation_split_reads",
        "stage2_training_additions",
        "stage2_support",
    }
    if (
        set(protocol) != expected_fields
        or protocol.get("schema_name") != REPAIR_PROTOCOL_SCHEMA
        or protocol.get("schema_version") != 1
        or protocol.get("run_id") != run_id
        or protocol.get("validation_split_reads") != 0
        or protocol.get("stage2_training_additions") != TRAINING_ADDITIONS
    ):
        raise RepairBundleError("Stage2 repair protocol fields differ")
    support = _support_contract(protocol.get("stage2_support"))
    checkpoint = _verify_expected_file(stage1_checkpoint, expected_checkpoint_sha, "Stage1 checkpoint", root=run_root)
    replay = _verify_expected_file(replay_manifest, expected_replay_sha, "Stage2 replay manifest", root=run_root)
    if (
        protocol.get("stage1_checkpoint_sha256") != sha256_file(checkpoint)
        or protocol.get("stage2_replay_manifest_sha256") != sha256_file(replay)
    ):
        raise RepairBundleError("Stage2 repair protocol checkpoint/replay binding differs")
    manifest_path, manifest_sha, _, prepared_support = _prepared_contract(
        run_root,
        prepare_dir,
        run_id=run_id,
        checkpoint_sha=sha256_file(checkpoint),
        replay_sha=sha256_file(replay),
    )
    if prepared_support != support:
        raise RepairBundleError("Stage2 repair protocol support differs from prepare output")
    if protocol.get("stage2_run_manifest_sha256") != manifest_sha:
        raise RepairBundleError("Stage2 repair protocol run-manifest binding differs")
    _, context = _context(run_root, run_id)
    references = (
        ("base_training_protocol", "base_training_protocol_sha256"),
        ("stage1_code_verification", "stage1_code_verification_sha256"),
        ("stage2_repair_code_verification", "stage2_repair_code_verification_sha256"),
    )
    for context_name, protocol_name in references:
        ref = context.get(context_name)
        if not isinstance(ref, Mapping):
            raise RepairBundleError(f"repair context lacks {context_name}")
        checked = _verify_expected_file(Path(str(ref["path"])), ref.get("sha256"), context_name, root=run_root)
        if protocol.get(protocol_name) != sha256_file(checked):
            raise RepairBundleError(f"repair protocol {context_name} binding differs")
    repair_ref = context["stage2_repair_code_verification"]
    repair_bundle = verify_bundle_paths(run_root, Path(str(repair_ref["path"])), str(repair_ref["sha256"]))
    if protocol.get("navigation_training_code_commit") != repair_bundle["navigation_training_code_commit"]:
        raise RepairBundleError("repair protocol code commit differs")
    return {
        "status": "passed",
        "protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)},
        "run_manifest": {"path": str(manifest_path), "sha256": manifest_sha},
        "stage1_checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
        "stage2_replay_manifest": {"path": str(replay), "sha256": sha256_file(replay)},
        "stage2_support": support,
        "repair_code": repair_bundle,
        "validation_split_reads": 0,
    }


def verify_prelaunch_command(args: argparse.Namespace) -> dict[str, object]:
    run_root = _regular_directory(args.run_root, "run root")
    result = verify_protocol(
        run_root=run_root,
        run_id=args.run_id,
        protocol_path=args.stage2_repair_protocol,
        expected_protocol_sha=args.expected_stage2_repair_protocol_sha256,
        stage1_checkpoint=args.stage1_checkpoint,
        expected_checkpoint_sha=args.expected_stage1_checkpoint_sha256,
        replay_manifest=args.stage2_replay_manifest,
        expected_replay_sha=args.expected_stage2_replay_manifest_sha256,
        prepare_dir=args.stage2_prepare_dir,
    )
    new_lock = run_root / ".launch-navigation-rlt-stage2-repair-v2"
    old_lock = run_root / ".launch-navigation-rlt-stage2"
    if old_lock.is_symlink() or os.path.lexists(old_lock):
        raise RepairBundleError("legacy Stage2 launch lock exists; repair launch is forbidden")
    if new_lock.is_symlink() or not new_lock.is_dir():
        raise RepairBundleError("fixed Stage2 repair launch lock is absent or unsafe")
    result["launch_lock"] = str(new_lock.resolve(strict=True))
    return result


def write_post_args(args: argparse.Namespace) -> dict[str, object]:
    run_root = _regular_directory(args.run_root, "run root")
    _, context = _context(run_root, args.run_id)
    repair_ref = context.get("stage2_repair_code_verification")
    if not isinstance(repair_ref, Mapping):
        raise RepairBundleError("repair code verification reference is absent")
    verification = _verify_expected_file(Path(str(repair_ref["path"])), repair_ref.get("sha256"), "repair code verification", root=run_root)
    protocol = _verify_expected_file(args.stage2_repair_protocol, args.expected_stage2_repair_protocol_sha256, "Stage2 repair protocol", root=run_root)
    flags = [
        "--stage2-repair-code-verification",
        str(verification),
        "--expected-stage2-repair-code-verification-sha256",
        sha256_file(verification),
        "--stage2-repair-protocol",
        str(protocol),
        "--expected-stage2-repair-protocol-sha256",
        sha256_file(protocol),
    ]
    output = run_root / "provenance" / REPAIR_DIR_NAME / "POST_TRAINING_REPAIR_ARGS.json"
    value = {
        "schema_name": "navigation_rlt_stage2_repair_post_training_args",
        "schema_version": 1,
        "status": "completed",
        "run_id": args.run_id,
        "validation_split_reads": 0,
        "argument_names": flags[0::2],
        "argv": flags,
    }
    digest = _write_json_new(output, value)
    return {"status": "completed", "path": str(output), "sha256": digest, "argv": flags}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)

    materialize = commands.add_parser("materialize-code")
    materialize.add_argument("--run-root", required=True, type=Path)
    materialize.add_argument("--source-root", required=True, type=Path)
    materialize.add_argument("--run-id", default=RUN_ID_DEFAULT)
    materialize.add_argument("--navigation-code-commit", required=True)
    materialize.add_argument("--supersedes-navigation-code-commit", required=True)
    materialize.add_argument("--base-training-protocol", required=True, type=Path)
    materialize.add_argument("--expected-base-training-protocol-sha256", required=True)
    materialize.add_argument("--stage1-code-verification", required=True, type=Path)
    materialize.add_argument("--expected-stage1-code-verification-sha256", required=True)

    verify_code = commands.add_parser("verify-code")
    verify_code.add_argument("--run-root", required=True, type=Path)
    verify_code.add_argument("--stage2-repair-code-verification", required=True, type=Path)
    verify_code.add_argument("--expected-stage2-repair-code-verification-sha256", required=True)

    protocol = commands.add_parser("write-protocol")
    protocol.add_argument("--run-root", required=True, type=Path)
    protocol.add_argument("--run-id", default=RUN_ID_DEFAULT)
    protocol.add_argument("--stage1-checkpoint", required=True, type=Path)
    protocol.add_argument("--expected-stage1-checkpoint-sha256", required=True)
    protocol.add_argument("--stage2-replay-manifest", required=True, type=Path)
    protocol.add_argument("--expected-stage2-replay-manifest-sha256", required=True)
    protocol.add_argument("--stage2-prepare-dir", required=True, type=Path)

    inspect_stage1 = commands.add_parser("inspect-stage1")
    inspect_stage1.add_argument("--run-root", required=True, type=Path)
    inspect_stage1.add_argument("--run-id", default=RUN_ID_DEFAULT)

    promotion = commands.add_parser("verify-promotion")
    promotion.add_argument("--run-root", required=True, type=Path)
    promotion.add_argument("--run-id", default=RUN_ID_DEFAULT)
    promotion.add_argument("--promotion-dir", required=True, type=Path)

    admission = commands.add_parser("verify-admission")
    admission.add_argument("--run-root", required=True, type=Path)
    admission.add_argument("--run-id", default=RUN_ID_DEFAULT)

    preflight = commands.add_parser("verify-prelaunch")
    preflight.add_argument("--run-root", required=True, type=Path)
    preflight.add_argument("--run-id", default=RUN_ID_DEFAULT)
    preflight.add_argument("--stage2-repair-protocol", required=True, type=Path)
    preflight.add_argument("--expected-stage2-repair-protocol-sha256", required=True)
    preflight.add_argument("--stage1-checkpoint", required=True, type=Path)
    preflight.add_argument("--expected-stage1-checkpoint-sha256", required=True)
    preflight.add_argument("--stage2-replay-manifest", required=True, type=Path)
    preflight.add_argument("--expected-stage2-replay-manifest-sha256", required=True)
    preflight.add_argument("--stage2-prepare-dir", required=True, type=Path)

    stage2 = commands.add_parser("verify-stage2")
    stage2.add_argument("--run-root", required=True, type=Path)
    stage2.add_argument("--run-id", default=RUN_ID_DEFAULT)
    stage2.add_argument("--stage2-repair-protocol", required=True, type=Path)
    stage2.add_argument("--expected-stage2-repair-protocol-sha256", required=True)
    stage2.add_argument("--stage1-checkpoint", required=True, type=Path)
    stage2.add_argument("--expected-stage1-checkpoint-sha256", required=True)
    stage2.add_argument("--stage2-replay-manifest", required=True, type=Path)
    stage2.add_argument("--expected-stage2-replay-manifest-sha256", required=True)
    stage2.add_argument("--stage2-prepare-dir", required=True, type=Path)

    post = commands.add_parser("write-post-args")
    post.add_argument("--run-root", required=True, type=Path)
    post.add_argument("--run-id", default=RUN_ID_DEFAULT)
    post.add_argument("--stage2-repair-protocol", required=True, type=Path)
    post.add_argument("--expected-stage2-repair-protocol-sha256", required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    operations = {
        "materialize-code": materialize_code,
        "verify-code": verify_code_command,
        "write-protocol": write_protocol,
        "inspect-stage1": inspect_stage1_command,
        "verify-promotion": verify_promotion_command,
        "verify-admission": verify_admission_command,
        "verify-prelaunch": verify_prelaunch_command,
        "verify-stage2": verify_stage2_command,
        "write-post-args": write_post_args,
    }
    output = operations[args.command](args)
    print(json.dumps(output, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RepairBundleError, FileExistsError) as error:
        print(
            json.dumps(
                {
                    "schema_name": "navigation_rlt_stage2_repair_bundle",
                    "schema_version": 1,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(2)
