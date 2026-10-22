from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch


MODULE_DIR = Path(__file__).resolve().parents[1] / "integrations" / "rlinf_v03"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import navigation_rlt_storage_guard as guard  # noqa: E402


GIB = 1024**3


def _mountinfo(
    path: Path,
    filesystem: str,
    options: str = "rw",
    super_options: str | None = None,
    source: str | None = None,
) -> str:
    escaped = str(path).replace("\\", "\\134").replace(" ", "\\040")
    effective_super_options = super_options or options
    effective_source = source
    if effective_source is None:
        effective_source = (
            guard.TURBO_SOURCE
            if filesystem in {"nfs", "nfs4"}
            else "/dev/nvme0n1p2"
        )
    return (
        f"36 25 0:32 / {escaped} {options},relatime - "
        f"{filesystem} {effective_source} {effective_super_options},vers=4.1\n"
    )


def _btrfs_usage(minimum_free: int, metadata_used: int, metadata_size: int) -> str:
    return (
        "Overall:\n"
        f"    Free (estimated): {minimum_free} (min: {minimum_free})\n"
        f"Metadata,DUP: Size: {metadata_size}, Used: {metadata_used} (91.00%)\n"
    )


def _dependencies(
    root: Path,
    *,
    filesystem: str,
    available: int,
    options: str = "rw",
    btrfs_usage: str = "",
) -> guard.GuardDependencies:
    return guard.GuardDependencies(
        statvfs=lambda _path: SimpleNamespace(f_bavail=available, f_frsize=1),
        read_mountinfo=lambda: _mountinfo(root, filesystem, options),
        read_btrfs_usage=lambda _path: btrfs_usage,
        now=lambda: 1_800_000_000.0,
    )


class NavigationRLTStorageGuardTests(unittest.TestCase):
    def test_policy_constants_are_exact_binary_gibibytes(self) -> None:
        self.assertEqual(guard.MIN_PROJECTED_AVAILABLE_BYTES, 100 * GIB)
        self.assertEqual(guard.TURBO_AVAILABLE_WARNING_BYTES, 200 * GIB)
        self.assertEqual(guard.EBK180_AVAILABLE_WARNING_BYTES, 150 * GIB)
        self.assertEqual(guard.BTRFS_MINIMUM_FREE_WARNING_BYTES, 80 * GIB)
        self.assertEqual(guard.BTRFS_MINIMUM_FREE_STOP_BYTES, 50 * GIB)
        self.assertEqual(guard.BTRFS_METADATA_WARNING_PERCENT, 90.0)
        self.assertEqual(guard.BTRFS_METADATA_STOP_PERCENT, 95.0)
        self.assertEqual(
            guard.PROFILE_POLICIES["turbo"]["source_equals"],
            guard.TURBO_SOURCE,
        )
        self.assertEqual(
            guard.PROFILE_POLICIES["ebk180"]["source_equals"],
            "/dev/nvme0n1p2",
        )

    def test_turbo_uses_bavail_and_warns_below_200_gib_after_pending(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            dependencies = _dependencies(
                root, filesystem="nfs4", available=250 * GIB
            )
            receipt = guard.evaluate_storage(
                "turbo", root, 60 * GIB, dependencies=dependencies
            )

        self.assertEqual(receipt["status"], "passed")
        self.assertEqual(
            receipt["ordinary_space"]["projected_available_bytes"], 190 * GIB
        )
        self.assertEqual(
            receipt["warnings"], ["ordinary_available_below_profile_warning"]
        )
        self.assertEqual(receipt["mount"]["filesystem"], "nfs4")
        self.assertTrue(receipt["mount"]["writable"])

    def test_exact_100_gib_floor_passes_and_one_byte_below_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            dependencies = _dependencies(
                root, filesystem="nfs", available=120 * GIB
            )
            passing = guard.evaluate_storage(
                "turbo", root, 20 * GIB, dependencies=dependencies
            )
            blocked = guard.evaluate_storage(
                "turbo", root, 20 * GIB + 1, dependencies=dependencies
            )

        self.assertEqual(passing["status"], "passed")
        self.assertEqual(blocked["status"], "blocked")
        self.assertIn("ordinary_available_below_stop", blocked["blocking_reasons"])

    def test_turbo_rejects_non_nfs_and_read_only_mounts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            ext4 = _dependencies(root, filesystem="ext4", available=300 * GIB)
            read_only = _dependencies(
                root, filesystem="nfs4", available=300 * GIB, options="ro"
            )
            with self.assertRaisesRegex(guard.StorageGuardError, "must use nfs/nfs4"):
                guard.evaluate_storage("turbo", root, 0, dependencies=ext4)
            with self.assertRaisesRegex(guard.StorageGuardError, "not read-write"):
                guard.evaluate_storage("turbo", root, 0, dependencies=read_only)

    def test_turbo_rejects_read_only_superblock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            dependencies = guard.GuardDependencies(
                statvfs=lambda _path: SimpleNamespace(
                    f_bavail=300 * GIB, f_frsize=1
                ),
                read_mountinfo=lambda: _mountinfo(
                    root, "nfs4", options="rw", super_options="ro"
                ),
                read_btrfs_usage=lambda _path: "",
                now=lambda: 1_800_000_000.0,
            )
            with self.assertRaisesRegex(guard.StorageGuardError, "not read-write"):
                guard.evaluate_storage("turbo", root, 0, dependencies=dependencies)

    def test_turbo_rejects_nfs_from_wrong_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            dependencies = guard.GuardDependencies(
                statvfs=lambda _path: SimpleNamespace(
                    f_bavail=300 * GIB, f_frsize=1
                ),
                read_mountinfo=lambda: _mountinfo(
                    root, "nfs4", source="generic-nfs.internal:/share"
                ),
                read_btrfs_usage=lambda _path: "",
                now=lambda: 1_800_000_000.0,
            )
            with self.assertRaisesRegex(guard.StorageGuardError, "expected source"):
                guard.evaluate_storage("turbo", root, 0, dependencies=dependencies)

    def test_turbo_rejects_lookalike_sfs_sources(self) -> None:
        lookalikes = (
            "unrelated.sfsturbo.internal:/another-share",
            "prefix.example/.sfsturbo.internal:/",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            for source in lookalikes:
                with self.subTest(source=source):
                    dependencies = guard.GuardDependencies(
                        statvfs=lambda _path: SimpleNamespace(
                            f_bavail=300 * GIB, f_frsize=1
                        ),
                        read_mountinfo=lambda source=source: _mountinfo(
                            root, "nfs", source=source
                        ),
                        read_btrfs_usage=lambda _path: "",
                        now=lambda: 1_800_000_000.0,
                    )
                    with self.assertRaisesRegex(
                        guard.StorageGuardError, "expected source"
                    ):
                        guard.evaluate_storage(
                            "turbo", root, 0, dependencies=dependencies
                        )

    def test_ebk180_warns_for_projected_btrfs_minimum_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            dependencies = _dependencies(
                root,
                filesystem="btrfs",
                available=170 * GIB,
                btrfs_usage=_btrfs_usage(90 * GIB, 91, 100),
            )
            receipt = guard.evaluate_storage(
                "ebk180", root, 20 * GIB, dependencies=dependencies
            )

        self.assertEqual(receipt["status"], "passed")
        self.assertEqual(receipt["btrfs"]["projected_minimum_free_bytes"], 70 * GIB)
        self.assertEqual(receipt["btrfs"]["metadata_percent"], 91.0)
        self.assertIn("btrfs_minimum_free_below_warning", receipt["warnings"])
        self.assertIn("btrfs_metadata_at_or_above_warning", receipt["warnings"])

    def test_ebk180_blocks_below_50_gib_or_at_95_percent_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            dependencies = _dependencies(
                root,
                filesystem="btrfs",
                available=200 * GIB,
                btrfs_usage=_btrfs_usage(69 * GIB, 95, 100),
            )
            receipt = guard.evaluate_storage(
                "ebk180", root, 20 * GIB, dependencies=dependencies
            )

        self.assertEqual(receipt["status"], "blocked")
        self.assertIn(
            "btrfs_minimum_free_below_stop", receipt["blocking_reasons"]
        )
        self.assertIn(
            "btrfs_metadata_at_or_above_stop", receipt["blocking_reasons"]
        )

    def test_ebk180_fails_closed_when_btrfs_usage_is_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            dependencies = _dependencies(
                root,
                filesystem="btrfs",
                available=200 * GIB,
                btrfs_usage="Overall:\n    Free (estimated): 1 (min: 1)\n",
            )
            with self.assertRaisesRegex(guard.StorageGuardError, "lacks metadata"):
                guard.evaluate_storage("ebk180", root, 0, dependencies=dependencies)

    def test_ebk180_rejects_btrfs_from_wrong_device(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            dependencies = guard.GuardDependencies(
                statvfs=lambda _path: SimpleNamespace(
                    f_bavail=200 * GIB, f_frsize=1
                ),
                read_mountinfo=lambda: _mountinfo(
                    root, "btrfs", source="/dev/nvme1n1p2"
                ),
                read_btrfs_usage=lambda _path: _btrfs_usage(100 * GIB, 10, 100),
                now=lambda: 1_800_000_000.0,
            )
            with self.assertRaisesRegex(guard.StorageGuardError, "expected source"):
                guard.evaluate_storage("ebk180", root, 0, dependencies=dependencies)

    def test_longest_mount_match_is_used(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            nested = root / "nested"
            nested.mkdir()
            text = _mountinfo(root, "ext4") + _mountinfo(nested, "nfs4")
            dependencies = guard.GuardDependencies(
                statvfs=lambda _path: SimpleNamespace(
                    f_bavail=300 * GIB, f_frsize=1
                ),
                read_mountinfo=lambda: text,
                read_btrfs_usage=lambda _path: "",
                now=lambda: 1_800_000_000.0,
            )
            receipt = guard.evaluate_storage(
                "turbo", nested, 0, dependencies=dependencies
            )

        self.assertEqual(receipt["mount"]["mount_point"], str(nested))

    def test_atomic_status_replaces_complete_json_and_cleans_staging(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "status.json"
            destination.write_text('{"old":true}\n', encoding="utf-8")
            guard.atomic_write_status(destination, {"status": "passed"})

            self.assertEqual(
                json.loads(destination.read_text(encoding="utf-8")),
                {"status": "passed"},
            )
            self.assertEqual(list(root.glob(".status.json.staging-*")), [])

    def test_cli_emits_json_and_returns_three_for_blocked_write(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            status = root / "guard.json"
            dependencies = _dependencies(
                root, filesystem="nfs4", available=100 * GIB
            )
            stdout = io.StringIO()
            exit_code = guard.main(
                [
                    "--profile",
                    "turbo",
                    "--path",
                    str(root),
                    "--pending-write-bytes",
                    "1",
                    "--status-output",
                    str(status),
                ],
                dependencies=dependencies,
                stdout=stdout,
            )
            receipt = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 3)
            self.assertEqual(receipt["status"], "blocked")
            self.assertEqual(json.loads(status.read_text(encoding="utf-8")), receipt)

    def test_cli_fails_closed_as_json_when_mount_cannot_be_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            dependencies = _dependencies(
                root, filesystem="ext4", available=300 * GIB
            )
            stdout = io.StringIO()
            exit_code = guard.main(
                [
                    "--profile",
                    "turbo",
                    "--path",
                    str(root),
                    "--pending-write-bytes",
                    "0",
                ],
                dependencies=dependencies,
                stdout=stdout,
            )

        receipt = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 2)
        self.assertEqual(receipt["status"], "error")
        self.assertEqual(
            receipt["blocking_reasons"], ["storage_safety_not_established"]
        )

    def test_atomic_status_failure_never_leaves_staging_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "status.json"
            with patch.object(guard.os, "replace", side_effect=OSError("failure")):
                with self.assertRaises(OSError):
                    guard.atomic_write_status(destination, {"status": "passed"})
            self.assertFalse(destination.exists())
            self.assertEqual(list(root.glob(".status.json.staging-*")), [])


if __name__ == "__main__":
    unittest.main()
