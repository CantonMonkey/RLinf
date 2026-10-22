# Copyright 2026 CantonMonkey.
# SPDX-License-Identifier: Apache-2.0

"""Stage Navigation-RLT source templates without installing or running them.

The receipt checks copied source bytes only. It is not a run manifest, a training
authorization, or evidence that paths, assets, dependencies and storage are ready.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import stat
from pathlib import Path, PurePosixPath
from typing import Sequence

RECEIPT_NAME = "SOURCE_STAGE.json"
SCHEMA_NAME = "navigation_rlt_source_stage"
INTEGRATION = PurePosixPath("integrations/rlinf_v03")
DIRECT_HELPERS = (
    "navigation_rlt_post_training.py",
    "navigation_rlt_storage_guard.py",
    "streamvln_executor_contract.py",
)


class StageError(ValueError):
    """Raised when the source or staged tree cannot be verified."""


def _relative(value: object) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise StageError("Source paths must be nonempty relative POSIX paths")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in value.split("/")):
        raise StageError("Absolute paths and path traversal are forbidden")
    return path


def _reject_links(path: Path) -> None:
    for component in (path, *path.parents):
        info = component.lstat()
        reparse = getattr(info, "st_file_attributes", 0) & getattr(
            stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0
        )
        if stat.S_ISLNK(info.st_mode) or reparse:
            raise StageError(f"Symlinks and reparse points are forbidden: {component}")


def _root(path: Path) -> Path:
    absolute = Path(os.path.abspath(path))
    _reject_links(absolute)
    if not absolute.is_dir():
        raise StageError(f"Expected a directory: {absolute}")
    return absolute


def _read(root: Path, relative: str) -> bytes:
    path = root.joinpath(*_relative(relative).parts)
    _reject_links(path)
    if not path.is_file():
        raise StageError(f"Expected a regular source file: {relative}")
    return path.read_bytes()


def _tree(data: bytes, filename: str) -> ast.Module:
    try:
        return ast.parse(data.decode("utf-8-sig"), filename=filename)
    except (UnicodeError, SyntaxError) as error:
        raise StageError(f"Cannot parse Python source: {filename}") from error


def collect_sources(source_root: Path) -> dict[str, tuple[str, bytes]]:
    """Collect the declared code inventory, local imports and navrl Python files.

    Args:
        source_root: Navigation extension root, containing integrations and navrl.

    Returns:
        Destination-relative paths mapped to source-relative paths and bytes.

    Raises:
        StageError: A source inventory or path is invalid.
    """
    root = _root(source_root)
    declaration = str(INTEGRATION / "navigation_rlt_run_manifest.py")
    tree = _tree(_read(root, declaration), declaration)
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "CODE_FILES"
            for target in node.targets
        )
    ]
    if len(assignments) != 1:
        raise StageError("Expected one literal Navigation-RLT CODE_FILES inventory")
    try:
        names = ast.literal_eval(assignments[0].value)
    except (ValueError, TypeError, SyntaxError) as error:
        raise StageError("CODE_FILES must be a literal list or tuple") from error
    if not isinstance(names, (list, tuple)) or not names:
        raise StageError("CODE_FILES must be a nonempty list or tuple")
    for name in names:
        relative = _relative(name)
        if len(relative.parts) != 1 or relative.suffix not in (".py", ".sh"):
            raise StageError("CODE_FILES requires flat Python or shell source names")
    if len(set(names)) != len(names):
        raise StageError("CODE_FILES contains duplicate paths")

    pending = list(names) + list(DIRECT_HELPERS)
    result: dict[str, tuple[str, bytes]] = {}
    while pending:
        name = pending.pop()
        if name in result:
            continue
        source = str(INTEGRATION / name)
        content = _read(root, source)
        result[name] = (source, content)
        if name.endswith(".py"):
            for node in ast.walk(_tree(content, source)):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [alias.name.split(".")[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules = [node.module.split(".")[0]]
                for module in modules:
                    candidate = root.joinpath(*INTEGRATION.parts, module + ".py")
                    if os.path.lexists(candidate):
                        pending.append(candidate.name)

    package = _root(root / "navrl")
    for path in sorted(package.rglob("*")):
        _reject_links(path)
        if path.is_file() and path.suffix == ".py":
            relative = path.relative_to(root).as_posix()
            result[relative] = (relative, _read(root, relative))
    if "navrl/__init__.py" not in result:
        raise StageError("The navrl source package is incomplete")
    return dict(sorted(result.items()))


def stage(source_root: Path, destination: Path) -> str:
    """Copy sources into a new flat code directory and return its receipt SHA.

    Args:
        source_root: Navigation extension source directory.
        destination: Nonexistent directory outside source_root; its parent exists.

    Returns:
        SHA-256 of SOURCE_STAGE.json, suitable for pinning subsequent verification.
    """
    root = _root(source_root)
    target = Path(os.path.abspath(destination))
    if os.path.lexists(target):
        raise StageError("Destination already exists; staging never overwrites it")
    _root(target.parent)
    if target.is_relative_to(root):
        raise StageError("Destination must be outside the source extension")
    sources = collect_sources(root)
    records = [
        {
            "path": name,
            "source_relative_path": source,
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        for name, (source, content) in sources.items()
    ]
    receipt = {
        "schema_name": SCHEMA_NAME,
        "schema_version": 1,
        "scope": "source_templates_only",
        "training_authorized": False,
        "deployment_validated": False,
        "files": records,
    }
    target.mkdir()
    for name, (_source, content) in sources.items():
        path = target.joinpath(*PurePosixPath(name).parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(content)
    raw = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with (target / RECEIPT_NAME).open("xb") as stream:
        stream.write(raw)
    digest = hashlib.sha256(raw).hexdigest()
    verify(target, expected_receipt_sha256=digest)
    return digest


def verify(destination: Path, *, expected_receipt_sha256: str | None = None) -> int:
    """Verify the staged file set and bytes without importing or running code.

    Args:
        destination: Previously staged code directory.
        expected_receipt_sha256: Optional independently retained receipt digest.

    Returns:
        Number of verified source files. Without a pinned receipt digest, this
        verifies internal consistency only, not origin or experiment authority.
    """
    root = _root(destination)
    raw = _read(root, RECEIPT_NAME)
    digest = hashlib.sha256(raw).hexdigest()
    if expected_receipt_sha256 is not None and digest != expected_receipt_sha256:
        raise StageError("Source-stage receipt SHA-256 differs from the pinned value")
    receipt = json.loads(raw)
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema_name") != SCHEMA_NAME
        or receipt.get("schema_version") != 1
        or receipt.get("scope") != "source_templates_only"
        or receipt.get("training_authorized") is not False
        or receipt.get("deployment_validated") is not False
    ):
        raise StageError("Invalid source-only receipt; it cannot authorize training")
    records = receipt.get("files")
    if not isinstance(records, list) or not records:
        raise StageError("Source-stage receipt has no file inventory")
    expected = {RECEIPT_NAME}
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "path",
            "source_relative_path",
            "bytes",
            "sha256",
        }:
            raise StageError("Invalid source-stage file record")
        name = str(_relative(record["path"]))
        _relative(record["source_relative_path"])
        if name in expected:
            raise StageError("Duplicate source-stage inventory path")
        expected.add(name)
        content = _read(root, name)
        if (
            len(content) != record["bytes"]
            or hashlib.sha256(content).hexdigest() != record["sha256"]
        ):
            raise StageError(f"Staged source bytes changed: {name}")
    observed = set()
    for path in root.rglob("*"):
        _reject_links(path)
        if path.is_file():
            observed.add(path.relative_to(root).as_posix())
    if observed != expected:
        raise StageError("Staged file set differs from the source-stage receipt")
    return len(records)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the source staging or verification CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    stage_parser = commands.add_parser(
        "stage", help="Copy source templates to a NEW code directory"
    )
    stage_parser.add_argument(
        "--source-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    stage_parser.add_argument("--destination", type=Path, required=True)
    verify_parser = commands.add_parser(
        "verify", help="Check source bytes; does not validate a deployment"
    )
    verify_parser.add_argument("--destination", type=Path, required=True)
    verify_parser.add_argument("--expected-receipt-sha256")
    args = parser.parse_args(argv)
    try:
        if args.command == "stage":
            digest = stage(args.source_root, args.destination)
            result = {"source_receipt_sha256": digest, "training_authorized": False}
        else:
            count = verify(
                args.destination,
                expected_receipt_sha256=args.expected_receipt_sha256,
            )
            result = {"verified_source_files": count, "training_authorized": False}
    except (StageError, OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        parser.exit(2, f"source staging failed: {error}\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
