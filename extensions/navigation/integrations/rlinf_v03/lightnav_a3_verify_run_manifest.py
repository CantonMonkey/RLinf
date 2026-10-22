#!/usr/bin/env python3
"""Verify that deployed LightNav A3 inputs still match the frozen run manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Sequence


class VerificationError(RuntimeError):
    """Raised when deployed run inputs drift from the frozen manifest."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular(path: Path) -> Path:
    value = path.resolve(strict=True)
    if not value.is_file() or value.is_symlink():
        raise VerificationError(f"input is not a regular file: {path}")
    return value


def _git(source: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(source), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", required=True, type=Path)
    result.add_argument("--expected-manifest-sha256", required=True)
    result.add_argument("--run-id", required=True)
    result.add_argument("--role", required=True, choices=("ebk", "a800"))
    result.add_argument("--code-root", required=True, type=Path)
    result.add_argument("--rlinf-source", required=True, type=Path)
    result.add_argument("--initial-checkpoint", required=True, type=Path)
    result.add_argument("--lightnav-source", type=Path)
    result.add_argument("--habitat-config", type=Path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    manifest_path = _regular(args.manifest)
    if sha256_file(manifest_path) != args.expected_manifest_sha256:
        raise VerificationError("outer run manifest SHA-256 differs")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_name") != "lightnav_a3_run_manifest"
        or manifest.get("schema_version") != 1
        or manifest.get("run_id") != args.run_id
        or not args.run_id
        or args.run_id in {".", ".."}
        or "/" in args.run_id
        or "\\" in args.run_id
    ):
        raise VerificationError("run identity differs from the outer manifest")

    code_root = args.code_root.resolve(strict=True)
    code = manifest.get("code")
    if not isinstance(code, dict) or not code:
        raise VerificationError("outer manifest has no code inventory")
    for relative, receipt in code.items():
        candidate = _regular(code_root / relative)
        try:
            candidate.relative_to(code_root)
        except ValueError as error:
            raise VerificationError("code inventory escapes code root") from error
        if sha256_file(candidate) != receipt.get("sha256"):
            raise VerificationError(f"deployed code differs: {relative}")

    rlinf_source = args.rlinf_source.resolve(strict=True)
    rlinf_commit = _git(rlinf_source, "rev-parse", "HEAD").decode().strip()
    if rlinf_commit != manifest.get("rlinf", {}).get("source_commit"):
        raise VerificationError("RLinf source commit differs")
    if sha256_file(_regular(args.initial_checkpoint)) != manifest.get("policy", {}).get(
        "initial_checkpoint_sha256"
    ):
        raise VerificationError("initial head differs")

    if args.role == "ebk":
        if args.lightnav_source is None or args.habitat_config is None:
            raise VerificationError("ebk verification requires LightNav source and config")
        lightnav_source = args.lightnav_source.resolve(strict=True)
        lightnav_commit = _git(lightnav_source, "rev-parse", "HEAD").decode().strip()
        if lightnav_commit != manifest.get("lightnav", {}).get("source_commit"):
            raise VerificationError("LightNav source commit differs")
        collision_diff = _git(
            lightnav_source,
            "diff",
            "--binary",
            "--",
            "habitat_server/lightnav_habitat/vlnce.py",
        )
        if hashlib.sha256(collision_diff).hexdigest() != manifest.get("lightnav", {}).get(
            "collision_diff_sha256"
        ):
            raise VerificationError("LightNav collision diff differs")
        if sha256_file(_regular(args.habitat_config)) != manifest.get("habitat", {}).get(
            "config_sha256"
        ):
            raise VerificationError("Habitat config differs")

    print(json.dumps({"run_id": args.run_id, "role": args.role, "status": "verified"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
