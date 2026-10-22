from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest import mock


MODULE_DIR = Path(__file__).resolve().parents[1] / "integrations" / "rlinf_v03"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import navigation_rlt_stage2_repair_bundle as repair  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


class Stage2RepairLauncherTests(unittest.TestCase):
    def _materialize_args(
        self, root: Path, *, omit_source: str | None = None
    ) -> tuple[Path, argparse.Namespace]:
        run = root / "run"
        (run / "code").mkdir(parents=True)
        source = root / "source"
        source.mkdir()
        for source_name in set(repair.CODE_SOURCES.values()) | set(
            repair.PROVENANCE_SOURCES
        ):
            if source_name == omit_source:
                continue
            (source / source_name).write_bytes(
                f"source:{source_name}\n".encode("utf-8")
            )
        stage1_commit = "1" * 40
        repair_commit = "2" * 40
        protocol = _json(
            run / "provenance" / "training_protocol.json",
            {
                "schema_name": "navigation_rlt_training_protocol",
                "schema_version": 1,
                "run_id": repair.RUN_ID_DEFAULT,
                "navigation_training_code_commit": stage1_commit,
            },
        )
        stage1_verification = _json(
            run / "provenance" / "stage1_code_verification.json",
            {
                "schema_name": repair.CODE_VERIFICATION_SCHEMA,
                "schema_version": 1,
                "status": "passed",
                "navigation_code_commit": stage1_commit,
            },
        )
        return run, argparse.Namespace(
            run_root=run,
            source_root=source,
            run_id=repair.RUN_ID_DEFAULT,
            navigation_code_commit=repair_commit,
            supersedes_navigation_code_commit=stage1_commit,
            base_training_protocol=protocol,
            expected_base_training_protocol_sha256=_sha(protocol),
            stage1_code_verification=stage1_verification,
            expected_stage1_code_verification_sha256=_sha(stage1_verification),
        )

    def _materialized(self, root: Path) -> tuple[Path, dict[str, object]]:
        run, args = self._materialize_args(root)
        result = repair.materialize_code(args)
        return run, result

    def _prepared_inputs(
        self,
        run: Path,
        *,
        recover_count: int = 0,
        recover_support: bool = False,
    ) -> tuple[Path, Path, Path]:
        checkpoint = run / "navigation_rlt_stage1" / "checkpoint.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"stage1 checkpoint")
        replay = _json(
            run / "navigation_rlt_stage2_replay" / "replay_manifest.json",
            {
                "schema_name": "navigation_rlt_replay_manifest",
                "schema_version": 3,
                "run_id": repair.RUN_ID_DEFAULT,
                "representation_kind": "frozen_z_rl",
                "expected_episode_count": 256,
                "completed_episode_count": 256,
                "split_label": "train_update",
                "validation_split_reads": 0,
            },
        )
        prepare = run / "navigation_rlt_stage2_prepare"
        prepare.mkdir()
        support = {
            "eligibility_schema": repair.ELIGIBILITY_SCHEMA,
            "gate_names": repair.GATE_NAMES,
            "gate_counts": [9, 7, 3, recover_count],
            "gate_support_mask": [True, True, True, recover_support],
            "minimum_behavior_gate_count": 1,
        }
        training = {
            "epochs": 10,
            "batch_size_per_rank": 64,
            **repair.TRAINING_ADDITIONS,
        }
        manifest = _json(
            prepare / "run_manifest.json",
            {
                "schema_name": "navigation_rlt_training_run_manifest",
                "schema_version": 1,
                "run_id": repair.RUN_ID_DEFAULT,
                "phase": "stage2",
                "required_world_size": 8,
                "boundaries": {
                    "upstream_split_read": "train",
                    "split_label": "train_update",
                    "validation_split_reads": 0,
                },
                "stage1_checkpoint": {"sha256": _sha(checkpoint)},
                "replay": {"manifest_sha256": _sha(replay)},
                "training": training,
                "stage2_support": support,
            },
        )
        receipt = _json(
            prepare / "prepare_receipt.json",
            {
                "schema_name": "navigation_rlt_prepare_training",
                "schema_version": 1,
                "status": "completed",
                "operation": "stage2",
                "run_id": repair.RUN_ID_DEFAULT,
                "stage1_checkpoint": {"sha256": _sha(checkpoint)},
                "replay": {"manifest_sha256": _sha(replay)},
                "training": training,
                "data": {"gate_support_mask": support["gate_support_mask"]},
            },
        )
        _json(
            prepare / "COMPLETED.json",
            {
                "schema_name": "navigation_rlt_prepare_training",
                "schema_version": 1,
                "status": "completed",
                "operation": "stage2",
                "artifact_kind": "run_manifest",
                "artifact_sha256": _sha(manifest),
                "receipt_sha256": _sha(receipt),
            },
        )
        return checkpoint, replay, prepare

    def test_materialize_creates_exact_nine_file_archive_with_readback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run, result = self._materialized(Path(temporary))
            self.assertEqual(result["file_count"], 9)
            verification = json.loads(
                Path(result["code_verification"]["path"]).read_text(encoding="utf-8")
            )
            readback = verification["archive_readback"]
            self.assertEqual(readback["status"], "passed")
            self.assertEqual(readback["member_count"], 9)
            self.assertEqual(
                [row["path"] for row in readback["members"]],
                sorted(repair.CODE_SOURCES),
            )
            checked = repair.verify_bundle_paths(
                run,
                Path(result["code_verification"]["path"]),
                result["code_verification"]["sha256"],
            )
            self.assertEqual(checked["file_count"], 9)
            self.assertNotEqual(checked["code_root"], str((run / "code").resolve()))

    def test_verify_rejects_extra_file_or_directory_in_live_code_root(self) -> None:
        for extra_kind in ("file", "directory"):
            with self.subTest(extra_kind=extra_kind), tempfile.TemporaryDirectory() as temporary:
                run, result = self._materialized(Path(temporary))
                code_root = Path(result["code_root"])
                code_root.chmod(0o700)
                if extra_kind == "file":
                    (code_root / "unexpected.py").write_bytes(b"unexpected\n")
                else:
                    (code_root / "unexpected").mkdir()
                code_root.chmod(0o500)
                with self.assertRaisesRegex(
                    repair.RepairBundleError, "exact nine-file closure"
                ):
                    repair.verify_bundle_paths(
                        run,
                        Path(result["code_verification"]["path"]),
                        result["code_verification"]["sha256"],
                    )

    def test_no_bytecode_import_preserves_exact_code_closure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, args = self._materialize_args(root)
            import_source = args.source_root / "navigation_rlt_policy.py"
            import_source.write_text("VALUE = 1\n", encoding="utf-8")
            result = repair.materialize_code(args)
            code_root = Path(result["code_root"])
            subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-c",
                    (
                        "import importlib.util, pathlib, sys; "
                        "path=pathlib.Path(sys.argv[1]); "
                        "spec=importlib.util.spec_from_file_location('repair_import', path); "
                        "module=importlib.util.module_from_spec(spec); "
                        "spec.loader.exec_module(module)"
                    ),
                    str(code_root / "navigation_rlt_policy.py"),
                ],
                check=True,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            self.assertFalse((code_root / "__pycache__").exists())
            checked = repair.verify_bundle_paths(
                run,
                Path(result["code_verification"]["path"]),
                result["code_verification"]["sha256"],
            )
            self.assertEqual(checked["file_count"], 9)

    @unittest.skipUnless(os.name == "posix", "POSIX mode semantics required")
    def test_verify_rejects_writable_code_root_or_run_wrapper(self) -> None:
        for writable_target in ("root", "wrapper"):
            with self.subTest(
                writable_target=writable_target
            ), tempfile.TemporaryDirectory() as temporary:
                run, result = self._materialized(Path(temporary))
                code_root = Path(result["code_root"])
                target = (
                    code_root
                    if writable_target == "root"
                    else code_root / "run_navigation_rlt_a800.sh"
                )
                target.chmod(0o700)
                with self.assertRaisesRegex(repair.RepairBundleError, "mode"):
                    repair.verify_bundle_paths(
                        run,
                        Path(result["code_verification"]["path"]),
                        result["code_verification"]["sha256"],
                    )

    def test_materialize_preloads_late_source_before_creating_destinations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = repair.PROVENANCE_SOURCES[-1]
            run, args = self._materialize_args(root, omit_source=missing)
            with self.assertRaisesRegex(repair.RepairBundleError, "missing"):
                repair.materialize_code(args)
            self.assertEqual(list(run.glob("code-stage2-repair-*")), [])
            self.assertFalse(
                (run / "provenance" / repair.REPAIR_DIR_NAME).exists()
            )

    def test_launcher_inventory_verifies_exact_files_and_rejects_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run, _ = self._materialized(Path(temporary))
            _, context = repair._context(run, repair.RUN_ID_DEFAULT)
            result = repair._verify_launcher_inventory(run, context)
            self.assertEqual(result["file_count"], 4)
            self.assertEqual(
                {row["path"] for row in result["files"]},
                set(repair.PROVENANCE_SOURCES),
            )
            self.assertIn(
                "launch_navigation_rlt_stage2_repair_v2.sh",
                repair.PROVENANCE_SOURCES,
            )
            self.assertNotIn(
                "launch_navigation_rlt_stage2_repair_v1.sh",
                repair.PROVENANCE_SOURCES,
            )
            target = (
                run
                / "provenance"
                / repair.REPAIR_DIR_NAME
                / repair.PROVENANCE_SOURCES[-1]
            )
            target.chmod(0o700)
            target.write_bytes(target.read_bytes() + b"tamper\n")
            with self.assertRaisesRegex(repair.RepairBundleError, "inventory"):
                repair._verify_launcher_inventory(run, context)

    def test_admission_accepts_absence_and_rejects_partial_intermediate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run, _ = self._materialized(Path(temporary))
            checkpoint = run / "navigation_rlt_stage1" / "checkpoint.pt"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"checkpoint")
            full_replay = _json(
                run / "navigation_rlt_prefix_relocated" / "replay_manifest.json",
                {"source_fingerprint": "fixture"},
            )
            stage1 = {
                "stage1_checkpoint": {
                    "path": str(checkpoint.resolve()),
                    "sha256": _sha(checkpoint),
                },
                "full_prefix_replay": {
                    "path": str(full_replay.resolve()),
                    "sha256": _sha(full_replay),
                    "source_fingerprint": "fixture",
                },
            }
            args = argparse.Namespace(run_root=run, run_id=repair.RUN_ID_DEFAULT)
            with mock.patch.object(
                repair, "inspect_stage1_command", return_value=stage1
            ):
                result = repair.verify_admission_command(args)
                self.assertEqual(
                    result["existing"],
                    {"promotion": "absent", "prepare": "absent", "protocol": "absent"},
                )
                (run / "navigation_rlt_stage2_replay").mkdir()
                with self.assertRaises(repair.RepairBundleError):
                    repair.verify_admission_command(args)

    def test_archive_readback_rejects_arbitrary_empty_and_incomplete_tar(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run, result = self._materialized(Path(temporary))
            code_root = Path(result["code_root"])
            arbitrary = Path(temporary) / "arbitrary.tar"
            arbitrary.write_bytes(b"not a tar archive")
            with self.assertRaisesRegex(repair.RepairBundleError, "unreadable"):
                repair._verify_tar(arbitrary, code_root)

            empty = Path(temporary) / "empty.tar"
            with tarfile.open(empty, "w"):
                pass
            with self.assertRaisesRegex(
                repair.RepairBundleError, "member inventory differs"
            ):
                repair._verify_tar(empty, code_root)

            incomplete = Path(temporary) / "incomplete.tar"
            with tarfile.open(incomplete, "w") as archive:
                payload = b"one file"
                info = tarfile.TarInfo(sorted(repair.CODE_SOURCES)[0])
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
            with self.assertRaisesRegex(
                repair.RepairBundleError, "member inventory differs"
            ):
                repair._verify_tar(incomplete, code_root)

            wrong_mode = Path(temporary) / "wrong-mode.tar"
            with tarfile.open(wrong_mode, "w") as archive:
                for index, name in enumerate(sorted(repair.CODE_SOURCES)):
                    payload = (code_root / name).read_bytes()
                    info = tarfile.TarInfo(name)
                    info.size = len(payload)
                    info.mode = (
                        0o600
                        if index == 0
                        else (0o500 if name in repair.EXECUTABLE_FILES else 0o400)
                    )
                    archive.addfile(info, io.BytesIO(payload))
            with self.assertRaisesRegex(repair.RepairBundleError, "mode differs"):
                repair._verify_tar(wrong_mode, code_root)

    def test_protocol_binds_checkpoint_replay_run_manifest_and_support(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run, _ = self._materialized(Path(temporary))
            checkpoint, replay, prepare = self._prepared_inputs(run)
            result = repair.write_protocol(
                argparse.Namespace(
                    run_root=run,
                    run_id=repair.RUN_ID_DEFAULT,
                    stage1_checkpoint=checkpoint,
                    expected_stage1_checkpoint_sha256=_sha(checkpoint),
                    stage2_replay_manifest=replay,
                    expected_stage2_replay_manifest_sha256=_sha(replay),
                    stage2_prepare_dir=prepare,
                )
            )
            protocol = json.loads(
                Path(result["protocol"]["path"]).read_text(encoding="utf-8")
            )
            self.assertEqual(protocol["stage1_checkpoint_sha256"], _sha(checkpoint))
            self.assertEqual(protocol["stage2_replay_manifest_sha256"], _sha(replay))
            self.assertEqual(
                protocol["stage2_run_manifest_sha256"],
                _sha(prepare / "run_manifest.json"),
            )
            self.assertEqual(
                protocol["stage2_support"]["gate_support_mask"],
                [True, True, True, False],
            )

    def test_protocol_refuses_recover_behavior_support(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run, _ = self._materialized(Path(temporary))
            checkpoint, replay, prepare = self._prepared_inputs(
                run, recover_count=1, recover_support=True
            )
            with self.assertRaisesRegex(repair.RepairBundleError, "RECOVER"):
                repair.write_protocol(
                    argparse.Namespace(
                        run_root=run,
                        run_id=repair.RUN_ID_DEFAULT,
                        stage1_checkpoint=checkpoint,
                        expected_stage1_checkpoint_sha256=_sha(checkpoint),
                        stage2_replay_manifest=replay,
                        expected_stage2_replay_manifest_sha256=_sha(replay),
                        stage2_prepare_dir=prepare,
                    )
                )

    def test_actual_wrapper_objective_block_accepts_historical_and_kl_v2(self) -> None:
        script = (MODULE_DIR / "run_navigation_rlt_a800_stage2_repair.sh").read_text(
            encoding="utf-8"
        )
        start = script.index("training = run.get('training')\n")
        end = script.index("support = run.get('stage2_support')", start)
        actual_block = compile(script[start:end], str(MODULE_DIR / "run_navigation_rlt_a800_stage2_repair.sh"), "exec")

        def fail(message):
            raise SystemExit(message)

        for schema, additions in (
            ("proposal_conditioned_finite_candidate_cql_v1", {}),
            ("proposal_conditioned_finite_candidate_cql_behavior_kl_v2", {"behavior_kl_weight": 1.0}),
        ):
            with self.subTest(schema=schema):
                training = {
                    "objective_schema": schema,
                    "cql_alpha": 1.0,
                    "cql_temperature": 1.0,
                    "minimum_behavior_gate_count": 1,
                    **additions,
                }
                exec(actual_block, {"run": {"training": training}, "fail": fail})

    def test_actual_wrapper_objective_block_rejects_unknown_or_invalid_kl(self) -> None:
        script = (MODULE_DIR / "run_navigation_rlt_a800_stage2_repair.sh").read_text(
            encoding="utf-8"
        )
        start = script.index("training = run.get('training')\n")
        end = script.index("support = run.get('stage2_support')", start)
        actual_block = compile(script[start:end], str(MODULE_DIR / "run_navigation_rlt_a800_stage2_repair.sh"), "exec")

        def fail(message):
            raise SystemExit(message)

        base = {
            "objective_schema": "proposal_conditioned_finite_candidate_cql_behavior_kl_v2",
            "cql_alpha": 1.0,
            "cql_temperature": 1.0,
            "minimum_behavior_gate_count": 1,
            "behavior_kl_weight": 1.0,
        }
        cases = (
            ("unknown schema", {**base, "objective_schema": "unknown"}),
            ("missing KL weight", {name: value for name, value in base.items() if name != "behavior_kl_weight"}),
            ("wrong KL weight", {**base, "behavior_kl_weight": 0.25}),
            ("boolean KL weight", {**base, "behavior_kl_weight": True}),
            ("legacy schema with KL", {**base, "objective_schema": "proposal_conditioned_finite_candidate_cql_v1"}),
            ("changed CQL", {**base, "cql_alpha": 0.25}),
        )
        for label, training in cases:
            with self.subTest(label=label):
                with self.assertRaises(SystemExit):
                    exec(actual_block, {"run": {"training": training}, "fail": fail})

    def test_new_entrypoints_use_only_repair_lock_and_emit_four_lineage_flags(self) -> None:
        run_wrapper = (
            MODULE_DIR / "run_navigation_rlt_a800_stage2_repair.sh"
        ).read_text(encoding="utf-8")
        successor = (
            MODULE_DIR / "continue_navigation_rlt_a800_stage2_repair.sh"
        ).read_text(encoding="utf-8")
        safe_wrapper = (
            MODULE_DIR
            / "continue_navigation_rlt_a800_stage2_repair_checkpoint_obs.sh"
        ).read_text(encoding="utf-8")
        v1_launcher_path = (
            MODULE_DIR / "launch_navigation_rlt_stage2_repair_v1.sh"
        )
        self.assertTrue(v1_launcher_path.is_file())
        v1_launcher = v1_launcher_path.read_text(encoding="utf-8")
        launcher = (
            MODULE_DIR / "launch_navigation_rlt_stage2_repair_v2.sh"
        ).read_text(encoding="utf-8")
        combined = "\n".join((run_wrapper, successor, safe_wrapper, launcher))
        self.assertEqual(repair.REPAIR_DIR_NAME, "stage2-repair-v2")
        self.assertIn("stage2-repair-v1", v1_launcher)
        self.assertNotIn("stage2-repair-v2", v1_launcher)
        for script in (run_wrapper, successor, safe_wrapper, launcher):
            self.assertIn(".launch-navigation-rlt-stage2-repair-v2", script)
        self.assertIn("legacy Stage2", combined)
        self.assertNotIn("continue_navigation_rlt_a800_stage2.sh\"", combined)
        self.assertNotIn("SUCCESSOR_SHA256=121d43", combined)
        self.assertIn("counts[3] != 0", run_wrapper)
        self.assertIn("mask[3] is not False", run_wrapper)
        for script in (launcher, successor, safe_wrapper, run_wrapper):
            self.assertIn("export PYTHONDONTWRITEBYTECODE=1", script)
            self.assertEqual(
                script.splitlines()[2], "export PYTHONDONTWRITEBYTECODE=1"
            )
            self.assertNotIn("unset PYTHONDONTWRITEBYTECODE", script)
        self.assertLess(
            launcher.index("export PYTHONDONTWRITEBYTECODE=1"),
            launcher.index("prelock_json="),
        )
        self.assertLess(
            successor.index("export PYTHONDONTWRITEBYTECODE=1"),
            successor.index('"$PYTHON" -B "$HELPER" verify-promotion'),
        )
        self.assertLess(
            run_wrapper.index("export PYTHONDONTWRITEBYTECODE=1"),
            run_wrapper.index('"$TORCHRUN" --standalone'),
        )
        self.assertNotIn('"$PYTHON" -B "$PROMOTER"', successor)
        self.assertNotIn('"$PYTHON" -B "$PREPARER"', successor)
        self.assertNotIn("navigation_rlt_promote_replay.py", successor)
        self.assertNotIn("navigation_rlt_prepare_training.py", successor)
        self.assertIn("verify-promotion", successor)
        self.assertIn("verify-prelaunch", successor)
        self.assertIn("v2 recovery refuses to recompute", successor)
        self.assertIn("verify-admission", launcher)
        self.assertIn(
            'CANONICAL_LAUNCHER="$REPAIR_PROVENANCE/launch_navigation_rlt_stage2_repair_v2.sh"',
            launcher,
        )
        self.assertIn('SELF_REALPATH=$(readlink -f -- "$0")', launcher)
        canonical_guard = '[[ "$SELF_REALPATH" == "$CANONICAL_LAUNCHER" ]]'
        self.assertLess(
            launcher.index(canonical_guard), launcher.index('exec 9<"$0"')
        )
        self.assertLess(
            launcher.index(
                'prelock_json=$("$PYTHON" -B "$HELPER" verify-admission'
            ),
            launcher.index('mkdir -- "$NEW_LOCK"'),
        )
        self.assertIn(
            "V1_ADMISSION_SHA256=d658ae6208f8a1846b77ae2e356a6bd128d943a724d71f15f21b49b04aeee776",
            launcher,
        )
        self.assertIn(
            "V1_STARTED_SHA256=8467ada7d0fffbf843c86334906db0c8128a1a393958f9ab7a7d65fffc447ee7",
            launcher,
        )
        self.assertIn(
            'require_evidence_400 "$V1_ADMISSION" "$V1_ADMISSION_SHA256"',
            launcher,
        )
        self.assertIn(
            'require_evidence_400 "$V1_STARTED" "$V1_STARTED_SHA256"',
            launcher,
        )
        self.assertIn('require_evidence_400 "$V1_PID_FILE"', launcher)
        self.assertIn('if kill -0 "$v1_pid" 2>/dev/null; then', launcher)
        self.assertIn('"previous_attempt": {', launcher)
        self.assertNotIn('rm -rf -- "$V1_LOCK"', launcher)
        self.assertNotIn('chmod ', launcher)
        for field, expected in (
            ("existing.promotion", "completed"),
            ("existing.prepare", "completed"),
            ("existing.protocol", "absent"),
        ):
            self.assertIn(
                f'[[ "$(json_get_value "$prelock_json" {field})" == "{expected}" ]]',
                launcher,
            )
        self.assertLess(
            launcher.index("existing.promotion"),
            launcher.index('mkdir -- "$NEW_LOCK"'),
        )
        self.assertIn(
            "STORAGE_GUARD_SHA256=2cc42194509297c11d31d475244147372ce6c5ae474a02c36f3932cebfd3df67",
            launcher,
        )
        self.assertIn(
            "ARCHIVER_SHA256=c49ddbb03f83d0ff515a9171ac745b82d3f0d12acf1ad247294213b4baf7119f",
            launcher,
        )
        for script in (launcher, safe_wrapper):
            self.assertIn("stat -c '%a' -- \"$path\"", script)
            self.assertIn('"$numeric_mode" == "500"', script)
            self.assertIn('require_immutable "$STORAGE_GUARD"', script)
            self.assertIn('require_immutable "$ARCHIVER"', script)
        self.assertIn("uuid.uuid4()", launcher)
        self.assertIn("turbo-stage2-repair-v2-attempt-$attempt_id.json", launcher)
        self.assertNotIn(
            'TURBO_RECEIPT="$RUN/storage-safety/turbo-stage2-repair-v2.json"',
            launcher,
        )
        self.assertNotIn("Stage2 repair Turbo receipt already exists", launcher)
        for flag in (
            "--stage2-repair-code-verification",
            "--expected-stage2-repair-code-verification-sha256",
            "--stage2-repair-protocol",
            "--expected-stage2-repair-protocol-sha256",
        ):
            self.assertIn(flag, repair.write_post_args.__code__.co_consts)

    @unittest.skipUnless(os.name == "posix", "POSIX launcher execution required")
    def test_launcher_copy_is_rejected_before_admission(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "copied-launcher.sh"
            shutil.copyfile(
                MODULE_DIR / "launch_navigation_rlt_stage2_repair_v2.sh", copied
            )
            copied.chmod(0o500)
            result = subprocess.run(
                ["bash", str(copied)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("canonical inventory path", result.stderr)


if __name__ == "__main__":
    unittest.main()
