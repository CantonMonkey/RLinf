from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT / "integrations" / "rlinf_v03"
LAUNCHER = MODULE_DIR / "launch_navigation_rlt_stage2_anticollapse_formal_successor.sh"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class NavigationRLTAntiCollapseFormalLauncherTest(unittest.TestCase):
    def test_launcher_binds_exact_anticollapse_closure_and_storage_identity(self) -> None:
        script = LAUNCHER.read_text(encoding="utf-8")
        expected = {
            "navigation_rlt_fsdp_train.py": (
                "1660cc508f43911a08203714ce9741a819509dede431fd651b290f50ca5edd29"
            ),
            "navigation_rlt_policy.py": (
                "4dc7b09c169f430147646b73ddb3c9c273f872f2c541c7cff6bec58c25ec26c9"
            ),
            "navigation_rlt_replay.py": (
                "f8fdfd33568bbdb83047c6372181c6c71822381ee049b30d32aea267e5743c94"
            ),
            "lightnav_a3_gpu_monitor.py": (
                "9e890e8550a3c881a2ad5b87c112f8123b5046231f39b3fd3804653ec553903f"
            ),
            "run_navigation_rlt_a800_stage2_repair.sh": (
                "5a05fc885d207778d6e77e9113a34c5921c040d73e9f20ba7e8949ba870f8f09"
            ),
        }
        for name, digest in expected.items():
            with self.subTest(name=name):
                self.assertEqual(_sha256(MODULE_DIR / name), digest)
                self.assertIn(digest, script)
        self.assertIn(
            "ANTI_COLLAPSE_COMMIT=fe6b16d6a8dd5421d9667a9dd55c65fcf234cc44",
            script,
        )
        self.assertIn(
            "BASE_PROTOCOL_SHA256=8c9ae4586e365a94d2aeacf3467374bb3f01b795f81a288474d81d17e8c1d43e",
            script,
        )
        self.assertIn(
            "STORAGE_GUARD_SHA256=2cc42194509297c11d31d475244147372ce6c5ae474a02c36f3932cebfd3df67",
            script,
        )
        self.assertIn(
            "TURBO_SOURCE='navigation."
            "storage.example.invalid:/'",
            script,
        )
        self.assertIn('mount.get("mount_point") != "/home/navcloud/work"', script)
        self.assertIn('mount.get("source") != source', script)
        self.assertIn('mount.get("source") != turbo_source', script)
        self.assertIn('mount.get("filesystem") not in {"nfs", "nfs4"}', script)
        self.assertIn("space.get(\"projected_available_bytes\", -1) < 107_374_182_400", script)

    def test_smoke_gate_recomputes_every_required_proof(self) -> None:
        script = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn(
            'SMOKE_ID=stage2-anticollapse-fe6b16d-2epoch-v1', script
        )
        self.assertIn('SMOKE_ROOT="$RUN/stage2-smokes/$SMOKE_ID"', script)
        self.assertIn('SMOKE_COMPLETED="$SMOKE_ROOT/SMOKE_COMPLETED.json"', script)
        self.assertIn('if set(smoke) != expected_smoke_fields:', script)
        for reference in (
            '"smoke_run_manifest"',
            '"stage_marker"',
            '"training_result"',
            '"checkpoint"',
            '"a800_verification"',
            '"telemetry_summary"',
            '"turbo_guard"',
        ):
            self.assertIn(reference, script)
        self.assertIn('if reference.get("sha256") != digest:', script)
        self.assertIn('smoke.get("epochs") != 2', script)
        self.assertIn('not isinstance(smoke_epochs, list) or len(smoke_epochs) != 2', script)
        self.assertIn('if max(values) >= 0.98:', script)
        self.assertIn('smoke_result.get("collapsed_epoch_streak") != 0', script)
        self.assertIn('smoke_result.get("fsdp_wrapper") != "FullyShardedDataParallel"', script)
        self.assertIn('if len(uuids) != 8:', script)
        self.assertIn('finite_tree(losses, "smoke.losses")', script)
        self.assertIn('restore.get("status") != "passed"', script)
        self.assertIn('components["stage1"] != initial["stage1"]', script)
        self.assertIn(
            'actual_smoke_implementation_changes != {"trainer_sha256", "policy_sha256"}',
            script,
        )
        self.assertNotIn(
            '"implementation.replay_sha256",\n        "implementation.trainer_sha256",',
            script,
        )
        self.assertLess(script.index("input_json="), script.index('mkdir -- "$FORMAL_ROOT"'))
        self.assertLess(
            script.index('[[ "$(json_get "$input_json" status)" == passed ]]'),
            script.index('"$PYTHON" -B "$STORAGE_GUARD"'),
        )

    def test_formal_successor_is_isolated_create_only_and_preserves_budget(self) -> None:
        script = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn('FORMAL_ID=stage2-behavior-kl-10epoch-v2', script)
        self.assertIn('FORMAL_PARENT="$RUN/stage2-successors"', script)
        self.assertIn('FORMAL_ROOT="$FORMAL_PARENT/$FORMAL_ID"', script)
        self.assertIn(
            'CODE_ROOT="$FORMAL_ROOT/code-stage2-repair-fe6b16d6a8dd"', script
        )
        self.assertIn('[[ ! -e "$FORMAL_ROOT" && ! -L "$FORMAL_ROOT" ]]', script)
        self.assertIn('mkdir -- "$FORMAL_ROOT"', script)
        self.assertIn('mkdir -- "$COMPAT_LOCK"', script)
        self.assertIn("os.O_EXCL", script)
        self.assertNotIn("rm -", script)
        self.assertNotIn("shutil.rmtree", script)
        self.assertNotIn(".unlink(", script)
        self.assertNotIn("os.replace", script)
        self.assertNotIn("val_seen", script)
        self.assertNotIn("val_unseen", script)
        self.assertNotIn("P8", script)
        self.assertLess(
            script.index('"$PYTHON" -B "$STORAGE_GUARD"'),
            script.index('mkdir -- "$FORMAL_ROOT"'),
        )
        self.assertIn(
            'if derived["training"] != expected_training or base_training.get("epochs") != 10:',
            script,
        )
        self.assertIn(
            'raise SystemExit("formal manifest changed a field outside implementation")',
            script,
        )
        self.assertIn(
            'if changed != ["policy_sha256", "trainer_sha256"]:', script
        )
        self.assertIn('"training_changed_paths": ["training.objective_schema", "training.behavior_kl_weight"]', script)
        self.assertIn('"formal_epochs": derived["training"]["epochs"]', script)
        self.assertIn(
            '"$FORMAL_ROOT" stage2 \\\n  "$DERIVED_MANIFEST" "$DERIVED_SHA256"',
            script,
        )
        self.assertIn('"$CHECKPOINT" "$CHECKPOINT_SHA256"', script)

    def test_formal_completion_requires_ten_stable_epochs_and_is_training_scoped(self) -> None:
        script = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn('not isinstance(epochs, list) or len(epochs) != 10', script)
        self.assertIn('if max(values) >= 0.98:', script)
        self.assertIn('result.get("collapsed_epoch_streak") != 0', script)
        self.assertIn('result.get("fsdp_wrapper") != "FullyShardedDataParallel"', script)
        self.assertIn('result.get("unique_gpu_uuid_count") != 8', script)
        self.assertIn('finite_tree(losses, "losses")', script)
        self.assertIn('"grad_norm" not in losses.get(summary_name, {})', script)
        self.assertIn('restore.get("status") != "passed"', script)
        self.assertIn('completion_path = formal_root / "FORMAL_COMPLETED.json"', script)
        self.assertIn('"schema_name": "navigation_rlt_stage2_behavior_kl_formal"', script)
        self.assertIn('"training_scope_only": True', script)
        self.assertIn('"legacy_canonical_archive_compatible": False', script)
        self.assertIn(
            '"downstream_archive_contract": "successor_aware_archive_required_v1"',
            script,
        )

    def test_checkpoint_receipt_accepts_metadata_and_rejects_identity_tampering(self) -> None:
        script = LAUNCHER.read_text(encoding="utf-8")
        start = script.index('restore = smoke_result.get("independent_restore")')
        end = script.index('\nsys.path.insert(0, str(source_root))', start)
        gate = compile(script[start:end], "smoke_checkpoint_gate.py", "exec")
        components = {
            name: hashlib.sha256(name.encode()).hexdigest()
            for name in ("stage1", "actor", "q1", "q2", "target_q1", "target_q2")
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "policy.pt"
            checkpoint.write_bytes(b"fixture checkpoint bytes")
            digest = _sha256(checkpoint)
            restore = {
                "schema_name": "navigation_rlt_independent_restore",
                "schema_version": 1,
                "status": "passed",
                "checkpoint_sha256": digest,
                "stage1_frozen": True,
                "training_step": 2,
            }
            result = {
                "final_checkpoint": {
                    "path": str(checkpoint), "sha256": digest,
                    "size_bytes": checkpoint.stat().st_size,
                    "metadata": {"training_step": 2, "extra": {"training_phase": "stage2"}},
                },
                "independent_restore": restore,
                "component_parameter_sha256": components,
            }
            smoke = {
                "component_parameter_sha256": components,
                "independent_restore": restore,
                "optimizer_steps_per_rank": 2,
            }

            def fail(message: str) -> None:
                raise ValueError(message)

            def verify(candidate: dict) -> None:
                exec(gate, {
                    "smoke_result": candidate, "smoke": smoke, "steps": 2,
                    "smoke_checkpoint_path": checkpoint, "smoke_checkpoint_sha": digest,
                    "re": re, "fail": fail,
                })

            verify(result)
            for field, value in (("path", "wrong.pt"), ("sha256", "0" * 64),
                                 ("size_bytes", 0), ("metadata", None)):
                with self.subTest(field=field):
                    tampered = copy.deepcopy(result)
                    tampered["final_checkpoint"][field] = value
                    with self.assertRaisesRegex(ValueError, "checkpoint/restore/component"):
                        verify(tampered)
        self.assertIn('smoke_checkpoint_ref.get("metadata") != smoke_metadata', script)

    def test_embedded_python_blocks_compile(self) -> None:
        script = LAUNCHER.read_text(encoding="utf-8")
        blocks = re.findall(r"<<'PY'\n(.*?)\nPY\n", script, flags=re.DOTALL)
        self.assertGreaterEqual(len(blocks), 5)
        for index, block in enumerate(blocks):
            with self.subTest(index=index):
                compile(block, f"formal_launcher_heredoc_{index}.py", "exec")

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
