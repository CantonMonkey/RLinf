from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT / "integrations" / "rlinf_v03"
LAUNCHER = MODULE_DIR / "launch_navigation_rlt_stage2_anticollapse_smoke.sh"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class NavigationRLTAntiCollapseSmokeLauncherTest(unittest.TestCase):
    def test_launcher_binds_exact_anticollapse_closure(self) -> None:
        script = LAUNCHER.read_text(encoding="utf-8")
        expected = {
            "navigation_rlt_fsdp_train.py": (
                "da4cf14e636842b161a0705df9945267eac75bee5f41ccf6f61a7410d0047e62"
            ),
            "navigation_rlt_policy.py": (
                "edf4ce64433542df4f798589ac72b76acaa874cc74178b2916c8f7422262afd5"
            ),
            "navigation_rlt_replay.py": (
                "f8fdfd33568bbdb83047c6372181c6c71822381ee049b30d32aea267e5743c94"
            ),
            "lightnav_a3_gpu_monitor.py": (
                "9e890e8550a3c881a2ad5b87c112f8123b5046231f39b3fd3804653ec553903f"
            ),
            "run_navigation_rlt_a800_stage2_repair.sh": (
                "f931936696d7971355880b0d292963f4329bda984a1a2149aba9d7087723d7b6"
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
            "TURBO_SOURCE='navigation."
            "storage.example.invalid:/'",
            script,
        )

    def test_launcher_is_create_only_isolated_and_checks_two_epoch_outcome(self) -> None:
        script = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn(
            "SMOKE_ID=stage2-anticollapse-fe6b16d-2epoch-v1", script
        )
        self.assertIn('SMOKE_PARENT="$RUN/stage2-smokes"', script)
        self.assertIn(
            'CODE_ROOT="$SMOKE_ROOT/code-stage2-repair-fe6b16d6a8dd"', script
        )
        self.assertIn('[[ ! -e "$SMOKE_ROOT" && ! -L "$SMOKE_ROOT" ]]', script)
        self.assertIn('mkdir -- "$SMOKE_ROOT"', script)
        self.assertIn('mkdir -- "$COMPAT_LOCK"', script)
        self.assertIn("os.O_EXCL", script)
        self.assertNotIn("rm -", script)
        self.assertNotIn("shutil.rmtree", script)
        self.assertNotIn("val_seen", script)
        self.assertNotIn("val_unseen", script)
        self.assertNotIn("P8", script)
        self.assertLess(script.index("input_json="), script.index('mkdir -- "$SMOKE_ROOT"'))
        self.assertLess(
            script.index('"$PYTHON" -B "$STORAGE_GUARD"'),
            script.index('mkdir -- "$SMOKE_ROOT"'),
        )
        self.assertIn('derived["training"]["epochs"] = 2', script)
        self.assertIn(
            'base_protocol.get("stage2_run_manifest_sha256") != base_sha', script
        )
        self.assertIn(
            'base_protocol.get("stage2_replay_manifest_sha256") != replay_sha',
            script,
        )
        self.assertIn(
            '"training_changed_paths": ["training.epochs"]', script
        )
        self.assertIn(
            'fail("smoke changed a Stage2 training parameter other than epochs")',
            script,
        )
        self.assertIn(
            'fail("smoke manifest changed a field outside epochs/implementation")',
            script,
        )
        self.assertIn(
            'or float(initialization.get("maximum_gate_probability", 1.0)) >= 0.98',
            script,
        )
        self.assertIn("not isinstance(epochs, list) or len(epochs) != 2", script)
        self.assertIn("if max(values) >= 0.98:", script)
        self.assertIn('result.get("collapsed_epoch_streak") != 0', script)
        self.assertIn('result.get("fsdp_wrapper") != "FullyShardedDataParallel"', script)
        self.assertIn("if len(uuids) != 8:", script)
        self.assertIn('finite_tree(result.get("losses"), "losses")', script)
        self.assertIn('components["stage1"] != initial["stage1"]', script)
        self.assertIn('restore.get("status") != "passed"', script)
        self.assertIn('telemetry.get("gpu_count") != 8', script)
        self.assertIn('completion_path = smoke_root / "SMOKE_COMPLETED.json"', script)

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
