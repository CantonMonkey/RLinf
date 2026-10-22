# Copyright 2026 CantonMonkey.
# SPDX-License-Identifier: Apache-2.0

"""Verify test packaging without weakening historical production identity checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from build_public_p8_fixtures import build
from p8_public_fixture import (
    ANCHORS,
    FIXTURES,
    NAVIGATION,
    PROFILE_SHA256,
    public_p8_contract,
    verify_fixture_profile,
)

INTEGRATION = NAVIGATION / "integrations" / "rlinf_v03"
sys.path.insert(0, str(INTEGRATION))
import run_streamvln_p8_successor_feature_v2_ab as runner  # noqa: E402


def historical_anchors() -> dict:
    """Snapshot the values that must survive both normal and exceptional exits."""
    return {
        group: {name: getattr(module, name) for name in names}
        for group, names in ANCHORS.items()
        for module in [runner.causal if group == "causal" else runner]
    }


class PublicFixtureTests(unittest.TestCase):
    """Keep synthetic test identities separate from production lineage."""

    def test_explicit_generator_reproduces_the_reviewed_fixture_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            generated = Path(directory) / "new-fixtures"
            self.assertEqual(build(generated), PROFILE_SHA256)
            for path in generated.glob("*.json"):
                self.assertEqual(path.read_bytes(), (FIXTURES / path.name).read_bytes())

    def assert_rejected_before_provider(
        self, readiness: Path, causal: Path, manifest: Path, output: Path
    ) -> None:
        """Assert early rejection without needing assets or heavy dependencies."""
        preflight, provider = Mock(), Mock()
        args = argparse.Namespace(
            stage="feature_v2_ab",
            split="train",
            output=output,
            readiness_manifest=readiness,
            causal_manifest=causal,
            manifest=manifest,
        )
        with self.assertRaisesRegex(
            runner.ProtocolError, "manifest file bytes drifted"
        ):
            runner.run(args, preflight_provider=preflight, ab_provider=provider)
        preflight.assert_not_called()
        self.assertEqual(provider.mock_calls, [])
        self.assertFalse(output.exists())

    def test_production_rejects_templates_and_test_fixtures(self):
        with tempfile.TemporaryDirectory() as directory:
            for paths in (
                (
                    FIXTURES / "readiness.json",
                    FIXTURES / "causal.json",
                    FIXTURES / "feature_v2_ab.json",
                ),
                tuple(
                    INTEGRATION / name
                    for name in (
                        "streamvln_p8_successor_bc_readiness_manifest.json",
                        "streamvln_p8_successor_causal_probe_manifest.json",
                        "streamvln_p8_successor_feature_v2_ab_manifest.json",
                    )
                ),
            ):
                with self.subTest(readiness=paths[0].name):
                    self.assert_rejected_before_provider(
                        *paths, Path(directory) / "must-not-exist.json"
                    )

    def test_context_restores_anchors_on_normal_and_exceptional_exit(self):
        original = historical_anchors()
        with public_p8_contract(runner):
            self.assertNotEqual(historical_anchors(), original)
        self.assertEqual(historical_anchors(), original)
        with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
            with public_p8_contract(runner):
                raise RuntimeError("synthetic failure")
        self.assertEqual(historical_anchors(), original)

    def test_fixture_tampering_is_rejected_before_anchor_injection(self):
        original = historical_anchors()
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "fixtures"
            shutil.copytree(FIXTURES, copied)
            path = copied / "readiness.json"
            path.write_bytes(path.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "fixture bytes drifted"):
                with public_p8_contract(runner, copied):
                    self.fail("invalid fixture must not enter the context")
        self.assertEqual(historical_anchors(), original)

    def test_json_whitespace_change_still_fails_raw_byte_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            modified = Path(directory) / "readiness.json"
            modified.write_bytes((FIXTURES / "readiness.json").read_bytes() + b" ")
            self.assertEqual(
                runner.readiness._json_hash(json.loads(modified.read_bytes())),
                runner.readiness._json_hash(
                    json.loads((FIXTURES / "readiness.json").read_bytes())
                ),
            )
            with public_p8_contract(runner):
                self.assert_rejected_before_provider(
                    modified,
                    FIXTURES / "causal.json",
                    FIXTURES / "feature_v2_ab.json",
                    Path(directory) / "must-not-exist.json",
                )

    def test_fixed_fixture_inventory_and_readiness_source_closure(self):
        verify_fixture_profile()
        manifest = json.loads((FIXTURES / "readiness.json").read_bytes())
        for row in manifest["implementation"]["files"]:
            source = NAVIGATION / Path(row["source_relative_path"]).relative_to(
                "rlinf-nav"
            )
            with self.subTest(source=row["source_relative_path"]):
                data = source.read_bytes()
                self.assertEqual(len(data), row["bytes"])
                self.assertEqual(hashlib.sha256(data).hexdigest(), row["sha256"])


if __name__ == "__main__":
    unittest.main()
