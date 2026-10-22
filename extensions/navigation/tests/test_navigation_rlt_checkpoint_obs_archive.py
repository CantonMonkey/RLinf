from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


MODULE_DIR = Path(__file__).resolve().parents[1] / "integrations" / "rlinf_v03"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import navigation_rlt_checkpoint_obs_archive as archive  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class NavigationRLTCheckpointOBSArchiveTests(unittest.TestCase):
    def _roots(self, root: Path) -> tuple[Path, Path]:
        obs_mount = root / "obs"
        obs_root = obs_mount / "navigation-example"
        obs_root.mkdir(parents=True)
        return obs_mount, obs_root

    def _archive(
        self,
        checkpoint: Path,
        obs_mount: Path,
        obs_root: Path,
        *,
        phase: str = "stage1",
    ) -> dict[str, object]:
        with mock.patch.object(archive, "OBS_MOUNT", obs_mount), mock.patch.object(
            archive, "DEFAULT_OBS_ROOT", obs_root
        ), mock.patch.object(archive, "_verify_obs_mount"):
            return archive.archive_checkpoint(
                checkpoint=checkpoint,
                expected_checkpoint_sha256=_sha(checkpoint),
                run_id="navigation-rlt-test",
                phase=phase,
                obs_root=obs_root,
            )

    def _phase_root(self, obs_root: Path, phase: str = "stage1") -> Path:
        return (
            obs_root
            / "checkpoints"
            / "navigation-rlt-stage-archives"
            / "navigation-rlt-test"
            / phase
        )

    def _attempts(self, phase_root: Path) -> list[Path]:
        digest_namespace = phase_root / archive.DIGEST_NAMESPACE
        if not digest_namespace.exists():
            return []
        return sorted(digest_namespace.glob("*/attempt-*"))

    def test_creates_unique_attempt_and_reuses_across_source_path_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            obs_mount, obs_root = self._roots(root)
            first_source = root / "first.pt"
            second_source = root / "moved.pt"
            first_source.write_bytes(b"checkpoint-payload")
            second_source.write_bytes(first_source.read_bytes())

            first = self._archive(first_source, obs_mount, obs_root)
            attempt = Path(first["archive_root"])
            self.assertRegex(attempt.name, archive.ATTEMPT_ID_PATTERN)
            self.assertEqual(attempt.parent.name, _sha(first_source))
            self.assertEqual(attempt.parent.parent.name, archive.DIGEST_NAMESPACE)
            self.assertEqual(
                {entry.name for entry in attempt.iterdir()},
                {
                    archive.CHECKPOINT_NAME,
                    archive.MANIFEST_NAME,
                    archive.COMPLETION_NAME,
                },
            )
            self.assertEqual(
                first["publication"],
                {
                    "attempt": "created",
                    "checkpoint": "created",
                    "manifest": "created",
                    "completion": "created",
                },
            )
            manifest = json.loads(
                (attempt / archive.MANIFEST_NAME).read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["source_checkpoint"]["path"], str(first_source))
            self.assertEqual(
                manifest["archived_checkpoint"]["content_address"],
                {"algorithm": "sha256", "digest": _sha(first_source)},
            )
            before = {path.name: path.read_bytes() for path in attempt.iterdir()}

            second = self._archive(second_source, obs_mount, obs_root)
            self.assertEqual(Path(second["archive_root"]), attempt)
            self.assertEqual(second["valid_attempt_count"], 1)
            self.assertEqual(
                second["publication"],
                {
                    "attempt": "reused",
                    "checkpoint": "reused",
                    "manifest": "reused",
                    "completion": "reused",
                },
            )
            self.assertEqual(before, {path.name: path.read_bytes() for path in attempt.iterdir()})
            with mock.patch.object(archive, "OBS_MOUNT", obs_mount), mock.patch.object(
                archive, "DEFAULT_OBS_ROOT", obs_root
            ), mock.patch.object(archive, "_verify_obs_mount"):
                verified = archive.verify_completed_phase(
                    expected_checkpoint_sha256=_sha(first_source),
                    run_id="navigation-rlt-test",
                    phase="stage1",
                    obs_root=obs_root,
                )
            self.assertEqual(verified["attempt_id"], first["attempt_id"])

    def test_verify_completed_phase_requires_and_binds_expected_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            obs_mount, obs_root = self._roots(root)
            archived_checkpoint = root / "archived-stage1.pt"
            local_stage1_checkpoint = root / "local-stage1.pt"
            archived_checkpoint.write_bytes(b"archived-stage1-checkpoint")
            local_stage1_checkpoint.write_bytes(b"different-local-stage1-checkpoint")
            self._archive(archived_checkpoint, obs_mount, obs_root)

            with mock.patch.object(archive, "OBS_MOUNT", obs_mount), mock.patch.object(
                archive, "DEFAULT_OBS_ROOT", obs_root
            ), mock.patch.object(archive, "_verify_obs_mount"):
                with self.assertRaisesRegex(
                    archive.CheckpointArchiveError, "different checkpoint"
                ):
                    archive.verify_completed_phase(
                        expected_checkpoint_sha256=_sha(local_stage1_checkpoint),
                        run_id="navigation-rlt-test",
                        phase="stage1",
                        obs_root=obs_root,
                    )

            with self.assertRaisesRegex(
                archive.CheckpointArchiveError,
                "expected checkpoint SHA-256 is required",
            ):
                archive.main(
                    [
                        "--verify-completed-phase",
                        "--run-id",
                        "navigation-rlt-test",
                        "--phase",
                        "stage1",
                    ]
                )

    def test_valid_completion_for_different_checkpoint_is_a_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            obs_mount, obs_root = self._roots(root)
            first = root / "first.pt"
            second = root / "second.pt"
            first.write_bytes(b"checkpoint-one")
            second.write_bytes(b"checkpoint-two")
            self._archive(first, obs_mount, obs_root)
            phase_root = self._phase_root(obs_root)

            with self.assertRaisesRegex(
                archive.CheckpointArchiveError, "different checkpoint"
            ):
                self._archive(second, obs_mount, obs_root)
            self.assertEqual(len(self._attempts(phase_root)), 1)

    def test_scans_all_digest_branches_and_rejects_two_valid_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            obs_mount, obs_root = self._roots(root)
            first = root / "first.pt"
            second = root / "second.pt"
            first.write_bytes(b"checkpoint-one")
            second.write_bytes(b"checkpoint-two")
            self._archive(first, obs_mount, obs_root)
            phase_root = self._phase_root(obs_root)
            second_sha = _sha(second)
            attempt_id, attempt = archive._create_attempt(phase_root, second_sha)
            archived = attempt / archive.CHECKPOINT_NAME
            archive._copy_new_checkpoint(
                second, archived, second_sha, second.stat().st_size
            )
            manifest = archive._expected_manifest(
                run_id="navigation-rlt-test",
                phase="stage1",
                attempt_id=attempt_id,
                attempt=attempt,
                source=second,
                source_size=second.stat().st_size,
                expected_sha256=second_sha,
                archived=archived,
            )
            manifest_payload = archive._json_bytes(manifest)
            archive._write_new_payload(
                attempt / archive.MANIFEST_NAME,
                manifest_payload,
                "archive manifest",
            )
            completion = archive._expected_completion(
                run_id="navigation-rlt-test",
                phase="stage1",
                attempt_id=attempt_id,
                attempt=attempt,
                phase_root=phase_root,
                archived=archived,
                checkpoint_sha256=second_sha,
                checkpoint_size=second.stat().st_size,
                manifest_payload=manifest_payload,
            )
            archive._write_new_payload(
                attempt / archive.COMPLETION_NAME,
                archive._json_bytes(completion),
                "completion marker",
            )

            with mock.patch.object(archive, "OBS_MOUNT", obs_mount), mock.patch.object(
                archive, "DEFAULT_OBS_ROOT", obs_root
            ), mock.patch.object(archive, "_verify_obs_mount"):
                with self.assertRaisesRegex(
                    archive.CheckpointArchiveError, "different checkpoints"
                ):
                    archive.verify_completed_phase(
                        expected_checkpoint_sha256=_sha(first),
                        run_id="navigation-rlt-test",
                        phase="stage1",
                        obs_root=obs_root,
                    )

    def test_identified_completed_attempt_tampering_is_a_hard_failure(self) -> None:
        for target in ("checkpoint", "manifest", "completion"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                obs_mount, obs_root = self._roots(root)
                checkpoint = root / "source.pt"
                checkpoint.write_bytes(b"checkpoint-payload")
                result = self._archive(checkpoint, obs_mount, obs_root)
                attempt = Path(result["archive_root"])
                if target == "checkpoint":
                    (attempt / archive.CHECKPOINT_NAME).write_bytes(
                        b"tampered-checkpoint"
                    )
                elif target == "manifest":
                    with (attempt / archive.MANIFEST_NAME).open("ab") as stream:
                        stream.write(b" ")
                else:
                    completion_path = attempt / archive.COMPLETION_NAME
                    completion = json.loads(
                        completion_path.read_text(encoding="utf-8")
                    )
                    completion["checkpoint"]["size_bytes"] += 1
                    completion_path.write_text(
                        json.dumps(completion, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )

                with mock.patch.object(
                    archive, "OBS_MOUNT", obs_mount
                ), mock.patch.object(
                    archive, "DEFAULT_OBS_ROOT", obs_root
                ), mock.patch.object(archive, "_verify_obs_mount"):
                    with self.assertRaises(archive.CheckpointArchiveError):
                        archive.verify_completed_phase(
                            expected_checkpoint_sha256=_sha(checkpoint),
                            run_id="navigation-rlt-test",
                            phase="stage1",
                            obs_root=obs_root,
                        )
                with self.assertRaises(archive.CheckpointArchiveError):
                    self._archive(checkpoint, obs_mount, obs_root)
                self.assertEqual(len(self._attempts(self._phase_root(obs_root))), 1)

    def test_non_json_and_incomplete_completions_are_preserved_and_bypassed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            obs_mount, obs_root = self._roots(root)
            checkpoint = root / "source.pt"
            checkpoint.write_bytes(b"checkpoint-payload")
            phase_root = self._phase_root(obs_root)
            phase_root.mkdir(parents=True)
            digest_root = (
                phase_root / archive.DIGEST_NAMESPACE / _sha(checkpoint)
            )
            digest_root.mkdir(parents=True)
            partial_payloads = (b'{"schema_name":', b'{"status":"completed"}\n')
            partials: list[Path] = []
            for index, payload in enumerate(partial_payloads):
                attempt = digest_root / f"attempt-{index:032x}"
                attempt.mkdir()
                (attempt / archive.CHECKPOINT_NAME).write_bytes(b"partial")
                (attempt / archive.MANIFEST_NAME).write_bytes(b"{}\n")
                (attempt / archive.COMPLETION_NAME).write_bytes(payload)
                partials.append(attempt)

            result = self._archive(checkpoint, obs_mount, obs_root)
            self.assertEqual(result["publication"]["attempt"], "created")
            self.assertEqual(len(self._attempts(phase_root)), 3)
            for attempt, payload in zip(partials, partial_payloads):
                self.assertEqual((attempt / archive.COMPLETION_NAME).read_bytes(), payload)

    def _assert_crash_recovery(self, failure_point: str) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            obs_mount, obs_root = self._roots(root)
            checkpoint = root / "source.pt"
            checkpoint.write_bytes(b"checkpoint-payload")

            if failure_point == "checkpoint":
                def fail_checkpoint(_source, destination, _sha256, _size):
                    destination.write_bytes(b"partial-checkpoint")
                    raise RuntimeError("crash mid checkpoint")

                patcher = mock.patch.object(
                    archive, "_copy_new_checkpoint", side_effect=fail_checkpoint
                )
            else:
                real_write = archive._write_new_payload

                def fail_payload(destination, payload, name):
                    if name == failure_point:
                        destination.write_bytes(b'{"partial":')
                        raise RuntimeError(f"crash mid {failure_point}")
                    return real_write(destination, payload, name)

                patcher = mock.patch.object(
                    archive, "_write_new_payload", side_effect=fail_payload
                )

            with mock.patch.object(archive, "OBS_MOUNT", obs_mount), mock.patch.object(
                archive, "DEFAULT_OBS_ROOT", obs_root
            ), mock.patch.object(archive, "_verify_obs_mount"), patcher:
                with self.assertRaisesRegex(RuntimeError, f"mid {failure_point}"):
                    archive.archive_checkpoint(
                        checkpoint=checkpoint,
                        expected_checkpoint_sha256=_sha(checkpoint),
                        run_id="navigation-rlt-test",
                        phase="stage1",
                        obs_root=obs_root,
                    )

            phase_root = self._phase_root(obs_root)
            first_attempts = self._attempts(phase_root)
            self.assertEqual(len(first_attempts), 1)
            partial_snapshot = {
                path.name: path.read_bytes() for path in first_attempts[0].iterdir()
            }

            result = self._archive(checkpoint, obs_mount, obs_root)
            self.assertEqual(result["publication"]["attempt"], "created")
            self.assertEqual(len(self._attempts(phase_root)), 2)
            self.assertEqual(
                partial_snapshot,
                {path.name: path.read_bytes() for path in first_attempts[0].iterdir()},
            )

    def test_recovers_after_crash_mid_checkpoint(self) -> None:
        self._assert_crash_recovery("checkpoint")

    def test_recovers_after_crash_mid_manifest(self) -> None:
        self._assert_crash_recovery("archive manifest")

    def test_recovers_after_crash_mid_completion(self) -> None:
        self._assert_crash_recovery("completion marker")

    def test_completion_is_written_last(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            obs_mount, obs_root = self._roots(root)
            checkpoint = root / "source.pt"
            checkpoint.write_bytes(b"checkpoint-payload")
            real_write = archive._write_new_payload
            order: list[str] = []

            def record(destination, payload, name):
                order.append(name)
                return real_write(destination, payload, name)

            with mock.patch.object(archive, "OBS_MOUNT", obs_mount), mock.patch.object(
                archive, "DEFAULT_OBS_ROOT", obs_root
            ), mock.patch.object(archive, "_verify_obs_mount"), mock.patch.object(
                archive, "_write_new_payload", side_effect=record
            ):
                archive.archive_checkpoint(
                    checkpoint=checkpoint,
                    expected_checkpoint_sha256=_sha(checkpoint),
                    run_id="navigation-rlt-test",
                    phase="stage1",
                    obs_root=obs_root,
                )
            self.assertEqual(order, ["archive manifest", "completion marker"])

    def test_rejects_mismatch_symlink_oversize_and_noncanonical_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            obs_mount, obs_root = self._roots(root)
            checkpoint = root / "source.pt"
            checkpoint.write_bytes(b"checkpoint-payload")
            with mock.patch.object(archive, "OBS_MOUNT", obs_mount), mock.patch.object(
                archive, "DEFAULT_OBS_ROOT", obs_root
            ), mock.patch.object(archive, "_verify_obs_mount"):
                with self.assertRaisesRegex(archive.CheckpointArchiveError, "SHA-256 differs"):
                    archive.archive_checkpoint(
                        checkpoint=checkpoint,
                        expected_checkpoint_sha256="0" * 64,
                        run_id="navigation-rlt-test",
                        phase="stage1",
                        obs_root=obs_root,
                    )
                with self.assertRaisesRegex(archive.CheckpointArchiveError, "canonical path"):
                    archive.archive_checkpoint(
                        checkpoint=checkpoint,
                        expected_checkpoint_sha256=_sha(checkpoint),
                        run_id="navigation-rlt-test",
                        phase="stage1",
                        obs_root=root / "other",
                    )
                link = root / "link.pt"
                try:
                    link.symlink_to(checkpoint)
                except OSError:
                    with mock.patch.object(Path, "is_symlink", return_value=True):
                        with self.assertRaisesRegex(
                            archive.CheckpointArchiveError, "symlink"
                        ):
                            archive._regular_file(checkpoint, "checkpoint")
                else:
                    with self.assertRaisesRegex(
                        archive.CheckpointArchiveError, "symlink"
                    ):
                        archive.archive_checkpoint(
                            checkpoint=link,
                            expected_checkpoint_sha256=_sha(checkpoint),
                            run_id="navigation-rlt-test",
                            phase="stage1",
                            obs_root=obs_root,
                        )
                with mock.patch.object(archive, "MAX_CHECKPOINT_BYTES", 4):
                    with self.assertRaisesRegex(archive.CheckpointArchiveError, "size limit"):
                        archive.archive_checkpoint(
                            checkpoint=checkpoint,
                            expected_checkpoint_sha256=_sha(checkpoint),
                            run_id="navigation-rlt-test",
                            phase="stage1",
                            obs_root=obs_root,
                        )

    def test_mount_probe_queries_destination_and_rejects_nested_mount(self) -> None:
        valid = SimpleNamespace(
            returncode=0,
            stdout="/obs s3fs[/navigation] fuse.s3fs rw,nosuid,nodev\n",
        )
        destination = Path("/obs/navigation-example")
        with mock.patch.object(archive.subprocess, "run", return_value=valid) as run:
            archive._verify_obs_mount(destination)
        self.assertEqual(
            run.call_args.args[0],
            [
                "findmnt",
                "-T",
                "/obs/navigation-example",
                "-n",
                "-o",
                "TARGET,SOURCE,FSTYPE,OPTIONS",
            ],
        )
        nested = SimpleNamespace(
            returncode=0,
            stdout=(
                "/obs/navigation-example nested-source fuse.s3fs "
                "rw,nosuid,nodev\n"
            ),
        )
        with mock.patch.object(archive.subprocess, "run", return_value=nested):
            with self.assertRaisesRegex(
                archive.CheckpointArchiveError, "not exactly /obs"
            ):
                archive._verify_obs_mount(destination)

    def test_archive_avoids_rename_directory_fsync_overwrite_and_deletion(self) -> None:
        source = Path(archive.__file__).read_text(encoding="utf-8")
        self.assertNotIn("renameat2", source)
        self.assertNotIn("_fsync_directory", source)
        self.assertNotIn("os.replace(", source)
        self.assertNotIn(".unlink(", source)


class NavigationRLTCheckpointOBSWrapperTests(unittest.TestCase):
    def _wrapper(self, name: str) -> str:
        return (MODULE_DIR / name).read_text(encoding="utf-8")

    def test_wrappers_guard_destination_mount_before_successor(self) -> None:
        names = (
            "continue_navigation_rlt_a800_stage1_retry004_checkpoint_obs.sh",
            "continue_navigation_rlt_a800_stage2_checkpoint_obs.sh",
        )
        for name in names:
            with self.subTest(name=name):
                source = self._wrapper(name)
                self.assertIn(
                    "PENDING_WRITE_BYTES=$((16 * 1024 * 1024 * 1024))", source
                )
                self.assertIn(
                    'findmnt -T "$OBS_ROOT" -n -o TARGET,SOURCE,FSTYPE,OPTIONS',
                    source,
                )
                self.assertIn("EXPECTED_OBS_SOURCE='s3fs[/navigation]'", source)
                self.assertIn('[[ "$target" == "$OBS_MOUNT" ]]', source)
                self.assertIn('[[ "$filesystem" == "fuse.s3fs" ]]', source)
                self.assertNotIn("df --output=avail", source)
                self.assertIn("storage_guard || guard_status=$?", source)
                self.assertIn("if (( guard_status != 0 )); then", source)
                self.assertIn(
                    'archive_verified_checkpoint "${checkpoint[0]}" "${checkpoint[1]}"',
                    source,
                )
                call_block = source.rsplit(
                    'require_immutable "$ARCHIVER" "$ARCHIVER_SHA256"', 1
                )[1]
                successor_call = call_block.rindex('\n"$SUCCESSOR"')
                self.assertLess(call_block.index("storage_guard"), successor_call)
                self.assertLess(call_block.index("verify_obs_mount"), successor_call)

    def test_stage2_scans_complete_stage1_attempt_before_successor(self) -> None:
        source = self._wrapper(
            "continue_navigation_rlt_a800_stage2_checkpoint_obs.sh"
        )
        self.assertIn(
            'STAGE1_MARKER="$RUN/navigation_rlt_stage1/STAGE.json"', source
        )
        self.assertIn("--verify-completed-phase", source)
        self.assertIn('--expected-checkpoint-sha256 "$checkpoint_sha"', source)
        self.assertIn("--phase stage1", source)
        self.assertNotIn("STAGE1_ARCHIVE=", source)
        call_block = source.rsplit(
            'require_immutable "$ARCHIVER" "$ARCHIVER_SHA256"', 1
        )[1]
        local_stage1_check = call_block.index(
            'stage1_checkpoint_output=$(verified_checkpoint stage1 "$STAGE1_MARKER")'
        )
        bound_archive_check = call_block.index(
            'verify_stage1_archive "${stage1_checkpoint[1]}"'
        )
        self.assertLess(local_stage1_check, bound_archive_check)
        self.assertLess(
            bound_archive_check,
            call_block.rindex('\n"$SUCCESSOR"'),
        )


if __name__ == "__main__":
    unittest.main()
