"""CPU-only checks for the opt-in actor/rollout physical GPU UUID gate."""

from __future__ import annotations

import ast
import io
import json
import os
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import gpu_uuid_gate


EXPECTED = "GPU-11111111-1111-1111-1111-111111111111"
OTHER = "GPU-22222222-2222-2222-2222-222222222222"
HERE = Path(__file__).resolve().parent


def worker_env() -> dict[str, str]:
    return {
        gpu_uuid_gate.ENABLE_ENV: "1",
        gpu_uuid_gate.EXPECTED_ENV: EXPECTED,
        gpu_uuid_gate.ALLOWED_ENV: "7",
        "CUDA_VISIBLE_DEVICES": "7",
        "VISIBLE_DEVICES": "0",
        "LOCAL_RANK": "0",
        "GROUP_NAME": "ActorGroup",
        "RANK": "0",
    }


def device(uuid_value: str) -> dict[str, object]:
    return {
        "cuda_device_count": 1,
        "cuda_zero_uuid": uuid_value,
        "context_before": None,
        "context_after": None,
    }


class GPUUUIDGateTests(unittest.TestCase):
    def test_disabled_never_loads_cuda_or_prints(self):
        output = io.StringIO()
        with patch.dict(os.environ, {}, clear=True), patch.object(
            gpu_uuid_gate, "_visible_gpu_uuid_without_context",
            side_effect=AssertionError("CUDA must remain untouched"),
        ), redirect_stdout(output):
            self.assertIsNone(gpu_uuid_gate.verify_gpu_worker_identity("actor"))
        self.assertEqual(output.getvalue(), "")

    def test_actor_match_records_worker_identity(self):
        output = io.StringIO()
        with patch.dict(os.environ, worker_env(), clear=True), patch.object(
            gpu_uuid_gate, "_visible_gpu_uuid_without_context", return_value=device(EXPECTED)
        ), redirect_stdout(output):
            result = gpu_uuid_gate.verify_gpu_worker_identity("actor")
        emitted = json.loads(output.getvalue())
        self.assertEqual(result, emitted)
        self.assertTrue(result["ok"])
        self.assertEqual(result["pid"], os.getpid())
        self.assertEqual(result["cuda_visible_devices"], "7")
        self.assertEqual(result["cuda_zero_uuid"], EXPECTED)
        self.assertEqual(result["context_before"], None)
        self.assertEqual(result["context_after"], None)

    def test_rollout_mismatch_fails_and_emits_failure_record(self):
        output = io.StringIO()
        with patch.dict(os.environ, worker_env(), clear=True), patch.object(
            gpu_uuid_gate, "_visible_gpu_uuid_without_context", return_value=device(OTHER)
        ), redirect_stdout(output):
            with self.assertRaisesRegex(RuntimeError, "physical GPU UUID"):
                gpu_uuid_gate.verify_gpu_worker_identity("rollout")
        emitted = json.loads(output.getvalue())
        self.assertFalse(emitted["ok"])
        self.assertEqual(emitted["cuda_zero_uuid"], OTHER)
        self.assertEqual(emitted["role"], "rollout")

    def test_enabled_without_expected_uuid_fails_before_cuda(self):
        env = worker_env()
        del env[gpu_uuid_gate.EXPECTED_ENV]
        with patch.dict(os.environ, env, clear=True), patch.object(
            gpu_uuid_gate, "_visible_gpu_uuid_without_context",
            side_effect=AssertionError("CUDA must remain untouched"),
        ), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, gpu_uuid_gate.EXPECTED_ENV):
                gpu_uuid_gate.verify_gpu_worker_identity("actor")

    def test_wrong_allowlist_fails_before_cuda(self):
        env = worker_env()
        env[gpu_uuid_gate.ALLOWED_ENV] = "6"
        with patch.dict(os.environ, env, clear=True), patch.object(
            gpu_uuid_gate, "_visible_gpu_uuid_without_context",
            side_effect=AssertionError("CUDA must remain untouched"),
        ), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "mapping disagrees"):
                gpu_uuid_gate.verify_gpu_worker_identity("actor")

    def test_gate_runs_before_actor_and_rollout_device_setup(self):
        def init_statements(filename: str, class_name: str) -> list[str]:
            tree = ast.parse((HERE / filename).read_text(encoding="utf-8"))
            klass = next(
                node for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == class_name
            )
            init = next(
                node for node in klass.body
                if isinstance(node, ast.FunctionDef) and node.name == "__init__"
            )
            return [ast.unparse(node) for node in init.body]

        actor = init_statements("fsdp_actor_worker_gpu_uuid.py", "EmbodiedFSDPActor")
        rollout = init_statements("huggingface_worker_vln.py", "MultiStepRolloutWorker")
        self.assertIn("verify_gpu_worker_identity('actor')", actor[1])
        self.assertIn("super().__init__(cfg.actor", actor[2])
        self.assertIn("verify_gpu_worker_identity('rollout')", rollout[1])
        self.assertLess(
            next(i for i, statement in enumerate(rollout) if "verify_gpu_worker_identity" in statement),
            next(i for i, statement in enumerate(rollout) if "current_device()" in statement),
        )


if __name__ == "__main__":
    unittest.main()
