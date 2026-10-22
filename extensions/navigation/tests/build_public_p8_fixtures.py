# Copyright 2026 CantonMonkey.
# SPDX-License-Identifier: Apache-2.0

"""Explicitly build a new unit-test fixture snapshot for developer review.

This command never runs as part of tests. It writes only to a new directory.
Review the generated diff and update the pinned profile digest in
p8_public_fixture.py separately. It creates no experiment authorization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

NAVIGATION = Path(__file__).resolve().parents[1]
INTEGRATION = NAVIGATION / "integrations" / "rlinf_v03"
PURPOSE = "synthetic-unit-tests-only-not-training-authorization"


def digest(data: bytes) -> str:
    """Return the SHA-256 of exact bytes."""
    return hashlib.sha256(data).hexdigest()


def canonical(value: object) -> str:
    """Match the historical runner's canonical JSON hashing convention."""
    return digest(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def encode(value: object) -> bytes:
    """Encode the reviewed fixture consistently on Windows and Linux."""
    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def load(name: str) -> dict:
    """Read a public deployment template."""
    return json.loads((INTEGRATION / name).read_text(encoding="utf-8"))


def refresh_source_records(manifest: dict) -> None:
    """Bind a new test snapshot to reviewed public source, never old receipts."""
    for row in manifest["implementation"]["files"]:
        relative = Path(row["source_relative_path"]).relative_to("rlinf-nav")
        source = (NAVIGATION / relative).resolve(strict=True)
        if not source.is_relative_to(NAVIGATION):
            raise ValueError("fixture source escaped the navigation extension")
        data = source.read_bytes()
        row["bytes"], row["sha256"] = len(data), digest(data)


def build(destination: Path) -> str:
    """Write three synthetic manifests and their independent fixed anchors."""
    if destination.exists():
        raise FileExistsError("fixture generation requires a new output directory")
    readiness = load("streamvln_p8_successor_bc_readiness_manifest.json")
    causal = load("streamvln_p8_successor_causal_probe_manifest.json")
    feature = load("streamvln_p8_successor_feature_v2_ab_manifest.json")
    refresh_source_records(readiness)
    refresh_source_records(feature)
    readiness_bytes = encode(readiness)
    readiness_raw, readiness_canonical = digest(readiness_bytes), canonical(readiness)

    def synthetic(label: str) -> str:
        return digest((PURPOSE + "/" + label).encode("utf-8"))

    causal_anchors = {
        "EXPECTED_READINESS_MANIFEST_RAW_SHA256": readiness_raw,
        "EXPECTED_READINESS_MANIFEST_CANONICAL_SHA256": readiness_canonical,
        "EXPECTED_READINESS_RECEIPT_FILE_SHA256": synthetic("readiness-receipt-file"),
        "EXPECTED_READINESS_RECEIPT_SHA256": synthetic("readiness-receipt"),
        "EXPECTED_CANDIDATE_SHA256": synthetic("candidate-checkpoint"),
        "EXPECTED_POLICY_STATE_SHA256": synthetic("candidate-policy-state"),
    }
    binding = causal["readiness_binding"]
    for field, anchor in {
        "readiness_manifest_raw_sha256": "EXPECTED_READINESS_MANIFEST_RAW_SHA256",
        "readiness_manifest_canonical_sha256": "EXPECTED_READINESS_MANIFEST_CANONICAL_SHA256",
        "readiness_receipt_file_sha256": "EXPECTED_READINESS_RECEIPT_FILE_SHA256",
        "readiness_receipt_sha256": "EXPECTED_READINESS_RECEIPT_SHA256",
        "candidate_checkpoint_sha256": "EXPECTED_CANDIDATE_SHA256",
        "candidate_policy_state_sha256": "EXPECTED_POLICY_STATE_SHA256",
    }.items():
        binding[field] = causal_anchors[anchor]
    causal_source = (
        INTEGRATION / "run_streamvln_p8_successor_causal_probe.py"
    ).read_bytes()
    causal["implementation"]["runner_bytes"] = len(causal_source)
    causal["implementation"]["runner_sha256"] = digest(causal_source)
    causal_bytes = encode(causal)
    feature_anchors = {
        "EXPECTED_READINESS_MANIFEST_RAW_SHA256": readiness_raw,
        "EXPECTED_READINESS_MANIFEST_CANONICAL_SHA256": readiness_canonical,
        "EXPECTED_CAUSAL_MANIFEST_RAW_SHA256": digest(causal_bytes),
        "EXPECTED_CAUSAL_MANIFEST_CANONICAL_SHA256": canonical(causal),
        "EXPECTED_CAUSAL_RECEIPT_FILE_SHA256": synthetic("causal-receipt-file"),
        "EXPECTED_CAUSAL_RECEIPT_SHA256": synthetic("causal-receipt"),
        "EXPECTED_CAUSAL_RUNNER_SHA256": digest(causal_source),
    }
    for field in feature["lineage"]:
        if field != "causal_authorize_feature_v2":
            feature["lineage"][field] = feature_anchors["EXPECTED_" + field.upper()]
    files = {
        "readiness.json": readiness_bytes,
        "causal.json": causal_bytes,
        "feature_v2_ab.json": encode(feature),
    }
    profile = {
        "schema_version": 1,
        "purpose": PURPOSE,
        "fixtures": {name: digest(data) for name, data in files.items()},
        "anchors": {"causal": causal_anchors, "feature": feature_anchors},
    }
    destination.mkdir(parents=True)
    for name, data in files.items():
        (destination / name).write_bytes(data)
    profile_bytes = encode(profile)
    (destination / "profile.json").write_bytes(profile_bytes)
    return digest(profile_bytes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(build(parser.parse_args().output))
