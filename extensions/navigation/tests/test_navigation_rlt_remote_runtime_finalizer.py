from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile


MODULE_DIR = Path(__file__).resolve().parents[1] / "integrations" / "rlinf_v03"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import navigation_rlt_remote_runtime_finalizer as finalizer  # noqa: E402


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _regular_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name)
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    return info


class RuntimeFixture:
    def __init__(self, parent: Path) -> None:
        self.run_id = "navigation-rlt-test"
        self.commit = "4" * 40
        self.root = parent / self.run_id
        self.root.mkdir()
        self.archive = self.root / "runtime.zip"
        self.final = self.root / "eval-runtime"
        self.plan = b'{"schema_name":"test-eval-plan"}\n'
        self.sources = {
            name: f"payload:{name}\n".encode("utf-8")
            for name in finalizer.EVALUATION_CODE_FILES
        }
        self.inventory = {
            "schema_name": finalizer.EVALUATION_INVENTORY_SCHEMA,
            "schema_version": 1,
            "navigation_code_commit": self.commit,
            "files": {
                name: _sha_bytes(payload)
                for name, payload in sorted(self.sources.items())
            },
        }
        self.inventory_payload = (
            json.dumps(self.inventory, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        self.write_archive()

    def write_archive(
        self,
        *,
        member_name_override: tuple[str, str] | None = None,
        member_mode_override: tuple[str, int] | None = None,
        source_override: tuple[str, bytes] | None = None,
    ) -> None:
        sources = dict(self.sources)
        if source_override is not None:
            sources[source_override[0]] = source_override[1]
        with zipfile.ZipFile(
            self.archive, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            members: list[tuple[str, bytes]] = [
                (finalizer.PLAN_MEMBER, self.plan),
                (finalizer.INVENTORY_MEMBER, self.inventory_payload),
            ]
            members.extend(
                (f"eval-code/{name}", sources[name])
                for name in finalizer.EVALUATION_CODE_FILES
            )
            for name, payload in members:
                stored_name = (
                    member_name_override[1]
                    if member_name_override is not None
                    and name == member_name_override[0]
                    else name
                )
                info = _regular_info(stored_name)
                if (
                    member_mode_override is not None
                    and name == member_mode_override[0]
                ):
                    info.external_attr = member_mode_override[1] << 16
                archive.writestr(info, payload)

    def arguments(self, **overrides: object) -> dict[str, object]:
        values: dict[str, object] = {
            "run_root": self.root,
            "run_id": self.run_id,
            "archive": self.archive,
            "expected_archive_sha256": _sha_file(self.archive),
            "expected_archive_size_bytes": self.archive.stat().st_size,
            "final_root": self.final,
            "expected_eval_plan_sha256": _sha_bytes(self.plan),
            "expected_evaluation_code_inventory_sha256": _sha_bytes(
                self.inventory_payload
            ),
            "expected_navigation_code_commit": self.commit,
        }
        values.update(overrides)
        return values


def _rename_new(source: Path, destination: Path) -> None:
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    source.rename(destination)


class NavigationRLTRemoteRuntimeFinalizerTests(unittest.TestCase):
    def _finalize_for_unit_test(
        self, fixture: RuntimeFixture
    ) -> tuple[dict[str, object], list[tuple[Path, int]]]:
        chmod_calls: list[tuple[Path, int]] = []
        real_chmod = finalizer.os.chmod

        def chmod_spy(path, mode, **kwargs):
            chmod_calls.append((Path(path), mode))
            try:
                return real_chmod(path, mode, **kwargs)
            except NotImplementedError:
                return real_chmod(path, mode)

        # Windows cannot represent an executable read-only file mode.  The
        # production path is Linux-only; the WSL test run exercises real mode
        # readback and renameat2.  These patches keep the portable unit suite
        # focused on ordering and create-only behavior.
        with patch.object(finalizer, "_fsync_directory"), patch.object(
            finalizer.os, "chmod", side_effect=chmod_spy
        ), patch.object(
            finalizer, "_validate_materialized_runtime"
        ), patch.object(
            finalizer, "_renameat2_noreplace", side_effect=_rename_new
        ):
            result = finalizer.finalize_runtime(**fixture.arguments())
        return result, chmod_calls

    def test_valid_publish_binds_every_identity_and_keeps_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RuntimeFixture(Path(temporary))
            write_order: list[str] = []
            real_write = finalizer._write_new_file

            def record_write(path: Path, payload: bytes) -> None:
                write_order.append(path.name)
                real_write(path, payload)

            with patch.object(
                finalizer, "_write_new_file", side_effect=record_write
            ):
                result, chmod_calls = self._finalize_for_unit_test(fixture)

            self.assertEqual(result["status"], "verified")
            self.assertEqual(result["run_id"], fixture.run_id)
            self.assertEqual(result["archive"]["sha256"], _sha_file(fixture.archive))
            self.assertEqual(
                result["evaluation_plan_sha256"], _sha_bytes(fixture.plan)
            )
            self.assertEqual(
                result["evaluation_code_inventory_sha256"],
                _sha_bytes(fixture.inventory_payload),
            )
            self.assertEqual(result["navigation_code_commit"], fixture.commit)
            self.assertEqual(result["source_file_count"], 35)
            self.assertEqual(result["file_count"], 35)
            self.assertEqual(result["archive"]["member_count"], 37)
            self.assertEqual(result["deployed_file_count"], 38)
            self.assertEqual(write_order[-1], finalizer.DEPLOYMENT_MARKER)
            self.assertTrue(fixture.archive.is_file())
            self.assertTrue(fixture.final.is_dir())
            marker = fixture.final / finalizer.DEPLOYMENT_MARKER
            self.assertEqual(result["deployment_marker"]["sha256"], _sha_file(marker))
            requested_modes = {mode for _path, mode in chmod_calls}
            self.assertEqual(requested_modes, {0o444, 0o555})

    def test_repeat_and_preexisting_conflict_preserve_existing_final(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RuntimeFixture(Path(temporary))
            self._finalize_for_unit_test(fixture)
            marker = fixture.final / finalizer.DEPLOYMENT_MARKER
            before = marker.read_bytes()
            with self.assertRaises(FileExistsError):
                finalizer.finalize_runtime(**fixture.arguments())
            self.assertEqual(marker.read_bytes(), before)
            self.assertTrue(fixture.archive.is_file())

    def test_publish_race_preserves_competing_final_and_complete_staging(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RuntimeFixture(Path(temporary))

            def competing_publish(_source: Path, destination: Path) -> None:
                destination.mkdir()
                (destination / "sentinel").write_bytes(b"sentinel")
                raise FileExistsError(destination)

            real_chmod = finalizer.os.chmod

            def portable_chmod(path, mode, **kwargs):
                try:
                    return real_chmod(path, mode, **kwargs)
                except NotImplementedError:
                    return real_chmod(path, mode)

            with patch.object(finalizer, "_fsync_directory"), patch.object(
                finalizer.os, "chmod", side_effect=portable_chmod
            ), patch.object(
                finalizer, "_validate_materialized_runtime"
            ), patch.object(
                finalizer,
                "_renameat2_noreplace",
                side_effect=competing_publish,
            ):
                with self.assertRaises(FileExistsError):
                    finalizer.finalize_runtime(**fixture.arguments())
            self.assertEqual(
                (fixture.final / "sentinel").read_bytes(), b"sentinel"
            )
            staging = list(
                fixture.root.glob(f".{fixture.final.name}.staging-*")
            )
            self.assertEqual(len(staging), 1)
            self.assertTrue((staging[0] / finalizer.DEPLOYMENT_MARKER).is_file())
            self.assertTrue(fixture.archive.is_file())

    def test_expected_hash_mismatches_fail_before_staging(self) -> None:
        cases = {
            "archive": {"expected_archive_sha256": "0" * 64},
            "plan": {"expected_eval_plan_sha256": "0" * 64},
            "inventory": {
                "expected_evaluation_code_inventory_sha256": "0" * 64
            },
        }
        for label, override in cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                fixture = RuntimeFixture(Path(temporary))
                with self.assertRaises(finalizer.RemoteRuntimeFinalizeError):
                    finalizer.finalize_runtime(**fixture.arguments(**override))
                self.assertTrue(fixture.archive.is_file())
                self.assertFalse(fixture.final.exists())
                self.assertEqual(
                    list(fixture.root.glob(f".{fixture.final.name}.staging-*")), []
                )

    def test_inventory_bound_source_hash_rejects_tampered_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RuntimeFixture(Path(temporary))
            fixture.write_archive(
                source_override=("navigation_rlt_eval.py", b"tampered\n")
            )
            with self.assertRaisesRegex(
                finalizer.RemoteRuntimeFinalizeError,
                "navigation_rlt_eval.py SHA-256 differs",
            ):
                finalizer.finalize_runtime(**fixture.arguments())
            self.assertTrue(fixture.archive.is_file())
            self.assertFalse(fixture.final.exists())

    def test_zip_slip_member_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RuntimeFixture(Path(temporary))
            fixture.write_archive(
                member_name_override=(
                    "eval-code/navigation_rlt_eval.py",
                    "eval-code/../navigation_rlt_eval.py",
                )
            )
            with self.assertRaises(finalizer.RemoteRuntimeFinalizeError):
                finalizer.finalize_runtime(**fixture.arguments())
            self.assertFalse((fixture.root / "navigation_rlt_eval.py").exists())
            self.assertTrue(fixture.archive.is_file())

    def test_zip_link_and_nonregular_members_are_rejected(self) -> None:
        cases = {
            "symlink": stat.S_IFLNK | 0o777,
            "fifo": stat.S_IFIFO | 0o644,
        }
        for label, member_mode in cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                fixture = RuntimeFixture(Path(temporary))
                fixture.write_archive(
                    member_mode_override=(
                        "eval-code/navigation_rlt_eval.py",
                        member_mode,
                    )
                )
                with self.assertRaisesRegex(
                    finalizer.RemoteRuntimeFinalizeError,
                    "link or non-regular member",
                ):
                    finalizer.finalize_runtime(**fixture.arguments())
                self.assertTrue(fixture.archive.is_file())
                self.assertFalse(fixture.final.exists())

    def test_archive_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RuntimeFixture(Path(temporary))
            target = fixture.root / "real-runtime.zip"
            fixture.archive.rename(target)
            try:
                fixture.archive.symlink_to(target)
            except OSError as error:
                self.skipTest(f"symbolic links unavailable: {error}")
            arguments = fixture.arguments(
                expected_archive_sha256=_sha_file(target),
                expected_archive_size_bytes=target.stat().st_size,
            )
            with self.assertRaisesRegex(
                finalizer.RemoteRuntimeFinalizeError, "must not be a symbolic link"
            ):
                finalizer.finalize_runtime(**arguments)

    def test_byte_identical_archive_repack_changes_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RuntimeFixture(Path(temporary))
            stream, identity = finalizer._open_archive(fixture.archive)
            expected_hash = _sha_file(fixture.archive)
            stream.close()
            saved = fixture.root / "original-runtime.zip"
            fixture.archive.rename(saved)
            shutil.copyfile(saved, fixture.archive)
            with saved.open("rb") as original_stream:
                with self.assertRaisesRegex(
                    finalizer.RemoteRuntimeFinalizeError, "identity changed"
                ):
                    finalizer._assert_archive_unchanged(
                        original_stream, fixture.archive, identity, expected_hash
                    )

    def test_rename_failure_retains_complete_staging_and_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RuntimeFixture(Path(temporary))
            real_chmod = finalizer.os.chmod

            def portable_chmod(path, mode, **kwargs):
                try:
                    return real_chmod(path, mode, **kwargs)
                except NotImplementedError:
                    return real_chmod(path, mode)

            with ExitStack() as stack:
                stack.enter_context(patch.object(finalizer, "_fsync_directory"))
                stack.enter_context(
                    patch.object(
                        finalizer.os, "chmod", side_effect=portable_chmod
                    )
                )
                stack.enter_context(
                    patch.object(finalizer, "_validate_materialized_runtime")
                )
                stack.enter_context(
                    patch.object(
                        finalizer,
                        "_renameat2_noreplace",
                        side_effect=OSError(errno := 22, os.strerror(errno)),
                    )
                )
                with self.assertRaises(OSError):
                    finalizer.finalize_runtime(**fixture.arguments())
            staging = list(
                fixture.root.glob(f".{fixture.final.name}.staging-*")
            )
            self.assertEqual(len(staging), 1)
            self.assertTrue((staging[0] / finalizer.DEPLOYMENT_MARKER).is_file())
            self.assertTrue(fixture.archive.is_file())
            self.assertFalse(fixture.final.exists())

    @unittest.skipUnless(sys.platform.startswith("linux"), "requires Linux")
    def test_linux_end_to_end_permissions_and_noreplace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RuntimeFixture(Path(temporary))
            result = finalizer.finalize_runtime(**fixture.arguments())
            self.assertEqual(result["status"], "verified")
            self.assertTrue(fixture.archive.is_file())
            self.assertEqual(stat.S_IMODE(fixture.final.lstat().st_mode), 0o555)
            for current, directory_names, file_names in os.walk(fixture.final):
                current_path = Path(current)
                self.assertEqual(stat.S_IMODE(current_path.lstat().st_mode), 0o555)
                for directory_name in directory_names:
                    self.assertEqual(
                        stat.S_IMODE((current_path / directory_name).lstat().st_mode),
                        0o555,
                    )
                for file_name in file_names:
                    path = current_path / file_name
                    relative = path.relative_to(fixture.final).as_posix()
                    expected_mode = (
                        0o555
                        if relative
                        == f"eval-code/{finalizer.EVALUATION_WRAPPER}"
                        else 0o444
                    )
                    self.assertEqual(
                        stat.S_IMODE(path.lstat().st_mode), expected_mode
                    )
            with self.assertRaises(FileExistsError):
                finalizer.finalize_runtime(**fixture.arguments())

    def test_production_publisher_has_no_ordinary_rename_fallback_or_delete(self) -> None:
        source = (MODULE_DIR / "navigation_rlt_remote_runtime_finalizer.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("renameat2", source)
        self.assertIn("RENAME_NOREPLACE", source)
        self.assertNotIn("os.replace", source)
        self.assertNotIn("shutil.rmtree", source)
        self.assertNotIn(".unlink(", source)


if __name__ == "__main__":
    unittest.main()
