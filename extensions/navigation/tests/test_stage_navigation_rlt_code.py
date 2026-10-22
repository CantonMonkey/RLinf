# Copyright 2026 CantonMonkey.
# SPDX-License-Identifier: Apache-2.0

"""Checks for source-only Navigation-RLT staging and byte verification."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import stage_navigation_rlt_code as staging


class NavigationSourceStageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.integration = self.source / "integrations/rlinf_v03"
        self.integration.mkdir(parents=True)
        (self.integration / "navigation_rlt_run_manifest.py").write_text(
            "CODE_FILES = ('navigation_rlt_run_manifest.py', 'trainer.py', 'run.sh')\n",
            encoding="utf-8",
        )
        (self.integration / "trainer.py").write_text(
            "import helper\n", encoding="utf-8"
        )
        (self.integration / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
        (self.integration / "run.sh").write_text(
            "#!/bin/sh\nexit 0\n", encoding="utf-8"
        )
        for name in staging.DIRECT_HELPERS:
            (self.integration / name).write_text("VALUE = 100\n", encoding="utf-8")
        (self.source / "navrl").mkdir()
        (self.source / "navrl/__init__.py").write_text("", encoding="utf-8")
        (self.integration / "historical_manifest.json").write_text(
            "{}", encoding="utf-8"
        )
        (self.integration / "old_receipt.json").write_text("{}", encoding="utf-8")
        self.destination = self.root / "code"

    def test_stages_import_closure_and_package_without_old_evidence(self):
        digest = staging.stage(self.source, self.destination)
        self.assertEqual(
            staging.verify(self.destination, expected_receipt_sha256=digest), 8
        )
        self.assertTrue((self.destination / "helper.py").is_file())
        self.assertTrue((self.destination / "navrl/__init__.py").is_file())
        self.assertFalse((self.destination / "historical_manifest.json").exists())
        self.assertFalse((self.destination / "old_receipt.json").exists())
        receipt = json.loads((self.destination / staging.RECEIPT_NAME).read_text())
        self.assertFalse(receipt["training_authorized"])
        self.assertFalse(receipt["deployment_validated"])

    def test_changed_and_missing_files_are_rejected(self):
        staging.stage(self.source, self.destination)
        helper = self.destination / "helper.py"
        original = helper.read_bytes()
        helper.write_bytes(original + b"# drift\n")
        with self.assertRaisesRegex(staging.StageError, "bytes changed"):
            staging.verify(self.destination)
        helper.unlink()
        with self.assertRaises(FileNotFoundError):
            staging.verify(self.destination)

    def test_existing_destination_is_preserved(self):
        self.destination.mkdir()
        marker = self.destination / "keep.txt"
        marker.write_text("user data", encoding="utf-8")
        with self.assertRaisesRegex(staging.StageError, "already exists"):
            staging.stage(self.source, self.destination)
        self.assertEqual(marker.read_text(), "user data")
        self.assertEqual(list(self.destination.iterdir()), [marker])

    def test_missing_source_fails_before_creating_destination(self):
        (self.integration / "run.sh").unlink()
        with self.assertRaises(FileNotFoundError):
            staging.stage(self.source, self.destination)
        self.assertFalse(self.destination.exists())

    def test_source_inventory_traversal_is_rejected(self):
        (self.integration / "navigation_rlt_run_manifest.py").write_text(
            "CODE_FILES = ('../outside.py',)\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(staging.StageError, "traversal"):
            staging.stage(self.source, self.destination)
        self.assertFalse(self.destination.exists())

    def test_missing_required_helper_fails_before_creating_destination(self):
        (self.integration / "navigation_rlt_post_training.py").unlink()
        with self.assertRaises(FileNotFoundError):
            staging.stage(self.source, self.destination)
        self.assertFalse(self.destination.exists())

    def test_receipt_pin_and_traversal_are_checked(self):
        digest = staging.stage(self.source, self.destination)
        receipt_path = self.destination / staging.RECEIPT_NAME
        receipt = json.loads(receipt_path.read_text())
        receipt["files"][0]["path"] = "../outside.py"
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        with self.assertRaisesRegex(staging.StageError, "pinned"):
            staging.verify(self.destination, expected_receipt_sha256=digest)
        with self.assertRaisesRegex(staging.StageError, "traversal"):
            staging.verify(self.destination)

    def test_symlink_source_is_rejected(self):
        target = self.root / "external.py"
        target.write_text("VALUE = 2\n", encoding="utf-8")
        link = self.integration / "helper.py"
        link.unlink()
        try:
            link.symlink_to(target)
        except OSError as error:
            self.skipTest(f"Creating symlinks is unavailable: {error}")
        with self.assertRaisesRegex(staging.StageError, "Symlinks"):
            staging.stage(self.source, self.destination)
        self.assertFalse(self.destination.exists())

    def test_current_inventory_includes_direct_helpers_without_p8(self):
        sources = staging.collect_sources(Path(__file__).resolve().parents[1])
        self.assertIn("navigation_rlt_post_training.py", sources)
        self.assertIn("navigation_rlt_storage_guard.py", sources)
        self.assertIn("streamvln_executor_contract.py", sources)
        self.assertIn("navrl/stack/ppo.py", sources)
        self.assertFalse(any("p8" in path.lower() for path in sources))
        self.assertTrue(all(Path(path).suffix in (".py", ".sh") for path in sources))


if __name__ == "__main__":
    unittest.main()
