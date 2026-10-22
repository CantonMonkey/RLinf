# Copyright 2026 CantonMonkey.
# SPDX-License-Identifier: Apache-2.0

"""Test-only fixed anchors; never imported by a production navigation runner."""

from __future__ import annotations

import hashlib
import json
import shutil
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Iterator
from unittest.mock import patch

NAVIGATION = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "p8_public_contract"
PROFILE_SHA256 = "558c4fb7f5bd184f51e7e3deda738525b7888705c754cda33705f1b7b9de6df1"
ANCHORS = {
    "causal": frozenset({
        "EXPECTED_READINESS_MANIFEST_RAW_SHA256",
        "EXPECTED_READINESS_MANIFEST_CANONICAL_SHA256",
        "EXPECTED_READINESS_RECEIPT_FILE_SHA256",
        "EXPECTED_READINESS_RECEIPT_SHA256",
        "EXPECTED_CANDIDATE_SHA256",
        "EXPECTED_POLICY_STATE_SHA256",
    }),
    "feature": frozenset({
        "EXPECTED_READINESS_MANIFEST_RAW_SHA256",
        "EXPECTED_READINESS_MANIFEST_CANONICAL_SHA256",
        "EXPECTED_CAUSAL_MANIFEST_RAW_SHA256",
        "EXPECTED_CAUSAL_MANIFEST_CANONICAL_SHA256",
        "EXPECTED_CAUSAL_RECEIPT_FILE_SHA256",
        "EXPECTED_CAUSAL_RECEIPT_SHA256",
        "EXPECTED_CAUSAL_RUNNER_SHA256",
    }),
}


def verify_fixture_profile(directory: Path = FIXTURES) -> dict:
    """Verify independently pinned fixture bytes before injecting test anchors."""
    raw = (directory / "profile.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != PROFILE_SHA256:
        raise ValueError("public P8 test profile bytes drifted")
    profile = json.loads(raw)
    if profile["purpose"] != "synthetic-unit-tests-only-not-training-authorization":
        raise ValueError("expected an explicitly synthetic unit-test profile")
    if set(profile["anchors"]) != set(ANCHORS):
        raise ValueError("unexpected fixture anchor group")
    for group, allowed in ANCHORS.items():
        if set(profile["anchors"][group]) != allowed:
            raise ValueError("only explicit identity anchors may be injected")
    if set(profile["fixtures"]) != {
        "readiness.json",
        "causal.json",
        "feature_v2_ab.json",
    }:
        raise ValueError("unexpected public P8 fixture inventory")
    for name, expected in profile["fixtures"].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"public P8 test fixture bytes drifted: {name}")
    return profile


@contextmanager
def public_causal_contract(module, directory: Path = FIXTURES) -> Iterator[dict]:
    """Inject only the causal test identities, with automatic restoration."""
    profile = verify_fixture_profile(directory)
    with patch.multiple(module, **profile["anchors"]["causal"]):
        yield profile


@contextmanager
def public_p8_contract(module, directory: Path = FIXTURES) -> Iterator[dict]:
    """Scope synthetic identities to one unit test and restore on every exit.

    Protocol constants, hash functions and validators are never patched.
    Production entrypoints have no option for enabling this test context.
    """
    profile = verify_fixture_profile(directory)
    with ExitStack() as stack:
        stack.enter_context(
            patch.multiple(module.causal, **profile["anchors"]["causal"])
        )
        stack.enter_context(patch.multiple(module, **profile["anchors"]["feature"]))
        yield profile


def stage_fixture_implementation(manifest: dict, destination: Path) -> None:
    """Copy source into a temporary test deployment without changing any digest."""
    for row in manifest["implementation"]["files"]:
        source_relative = Path(row["source_relative_path"]).relative_to("rlinf-nav")
        source = (NAVIGATION / source_relative).resolve(strict=True)
        target = (destination / row["deployment_relative_path"]).resolve()
        if not source.is_relative_to(NAVIGATION) or not target.is_relative_to(
            destination.resolve()
        ):
            raise ValueError("implementation path escaped its test directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
