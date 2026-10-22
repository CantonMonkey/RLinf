from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT / "integrations" / "rlinf_v03"
LAUNCHER = MODULE_DIR / "launch_lightnav_a3_training_budget_receipt.sh"
GENERATOR = MODULE_DIR / "lightnav_a3_training_budget_receipt.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LightNavA3TrainingBudgetReceiptLauncherTests(unittest.TestCase):
    def test_launcher_binds_exact_production_generator(self) -> None:
        script = LAUNCHER.read_text(encoding="utf-8")
        expected_generator_sha = (
            "a4b596ed05d6610e7ba859e53f2427108fbf16f9ca746166c66f28d745b929fe"
        )
        self.assertEqual(_sha256(GENERATOR), expected_generator_sha)
        self.assertIn(f"GENERATOR_SHA256={expected_generator_sha}", script)
        self.assertIn("A3_RUN_ID=lightnav-a3-20260930-v1", script)
        self.assertIn(
            'FINAL_MANIFEST="$OBS_ROOT/manifests/lightnav-a3/'
            '$A3_RUN_ID.final.json"',
            script,
        )
        self.assertIn(
            'ARCHIVE_COMPLETION="$OBS_ROOT/results/lightnav-a3/'
            '$A3_RUN_ID/COMPLETE"',
            script,
        )
        self.assertIn("# 130 archived files", script)
        self.assertIn('"$PYTHON" -B "$GENERATOR" \\', script)
        self.assertIn('--obs-root "$OBS_ROOT"', script)
        self.assertIn('--final-manifest "$FINAL_MANIFEST"', script)
        self.assertIn('--completion "$ARCHIVE_COMPLETION"', script)
        self.assertIn('--output "$RECEIPT"', script)

    def test_launcher_checks_live_obs_longest_match_and_fresh_turbo_guard(self) -> None:
        script = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("RUN=/srv/navigation/runs/", script)
        self.assertIn("RUN_ID=navigation-rlt-20261002-v2", script)
        self.assertIn("OBS_MOUNT=/obs", script)
        self.assertIn("OBS_ROOT=/obs/navigation-example", script)
        self.assertIn("EXPECTED_OBS_SOURCE='s3fs[/navigation]'", script)
        self.assertIn(
            'mount_record=$(LC_ALL=C findmnt -T "$OBS_ROOT" -n '
            "-o TARGET,SOURCE,FSTYPE,OPTIONS)",
            script,
        )
        self.assertIn('[[ "$target" == "$OBS_MOUNT" ]]', script)
        self.assertIn('[[ "$source" == "$EXPECTED_OBS_SOURCE" ]]', script)
        self.assertIn('[[ "$filesystem" == "fuse.s3fs" ]]', script)
        self.assertIn('[[ ",$options," == *,rw,*', script)
        self.assertIn("PENDING_WRITE_BYTES=17179869184", script)
        self.assertIn(
            "STORAGE_GUARD_SHA256="
            "2cc42194509297c11d31d475244147372ce6c5ae474a02c36f3932cebfd3df67",
            script,
        )
        self.assertIn('guard_json=$("$PYTHON" -B "$STORAGE_GUARD"', script)
        self.assertNotIn("--status-output", script)
        self.assertLess(
            script.index('guard_json=$("$PYTHON" -B "$STORAGE_GUARD"'),
            script.index('mkdir -- "$OUTPUT_ROOT"'),
        )

    def test_launcher_is_fixed_create_only_and_binds_completion(self) -> None:
        script = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn(
            'OUTPUT_ROOT="$PROVENANCE_PARENT/a3-training-budget-receipt-v3"',
            script,
        )
        self.assertIn(
            'RECEIPT="$OUTPUT_ROOT/A3_TRAINING_BUDGET_RECEIPT.json"', script
        )
        self.assertIn('COMPLETION="$OUTPUT_ROOT/COMPLETED.json"', script)
        self.assertIn(
            '[[ ! -e "$OUTPUT_ROOT" && ! -L "$OUTPUT_ROOT" ]]', script
        )
        self.assertIn('mkdir -- "$OUTPUT_ROOT"', script)
        self.assertGreaterEqual(script.count("os.O_EXCL"), 2)
        self.assertIn(
            '"schema_name": "navigation_rlt_a3_training_budget_receipt_completion"',
            script,
        )
        self.assertIn('"run_id": run_id', script)
        self.assertIn('"source_run_id": a3_run_id', script)
        self.assertIn('"path": str(receipt_path)', script)
        self.assertIn('"sha256": expected_receipt_sha', script)
        self.assertIn('"schema_name": receipt["schema_name"]', script)
        self.assertIn('"run_id": receipt["run_id"]', script)
        self.assertIn('"validation_split_reads": 0', script)
        self.assertIn('fail("a receipt artifact resolves inside OBS")', script)
        self.assertNotIn("rm -", script)
        self.assertNotIn(".unlink(", script)
        self.assertNotIn("os.replace", script)
        self.assertNotIn("val_seen", script)
        self.assertNotIn("val_unseen", script)
        self.assertNotIn("P8", script)
        self.assertIn('generator_json=$("$PYTHON" -B "$GENERATOR"', script)
        self.assertIn('"rollout_inventory_coverage": coverage', script)
        self.assertIn('coverage.get("full_rollout_inventory_readback") is not False', script)
        self.assertIn('entry.get("payload_readback") is not False', script)

    def test_embedded_python_is_syntactically_valid(self) -> None:
        script = LAUNCHER.read_text(encoding="utf-8")
        blocks = []
        remainder = script
        while "<<'PY'\n" in remainder:
            _, remainder = remainder.split("<<'PY'\n", maxsplit=1)
            block, remainder = remainder.split("\nPY\n", maxsplit=1)
            blocks.append(block)
        self.assertEqual(len(blocks), 3)
        for index, block in enumerate(blocks):
            with self.subTest(index=index):
                compile(block, f"<launcher-heredoc-{index}>", "exec")

    @unittest.skipUnless(shutil.which("bash"), "bash is required for syntax check")
    def test_launcher_has_valid_bash_syntax(self) -> None:
        launcher_path = str(LAUNCHER)
        if os.name == "nt":
            resolved = LAUNCHER.resolve()
            launcher_path = (
                f"/mnt/{resolved.drive[0].lower()}"
                + resolved.as_posix().split(":", maxsplit=1)[1]
            )
        completed = subprocess.run(
            [shutil.which("bash") or "bash", "-n", launcher_path],
            check=False,
            capture_output=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            completed.stderr.decode("utf-8", errors="replace"),
        )


if __name__ == "__main__":
    unittest.main()
