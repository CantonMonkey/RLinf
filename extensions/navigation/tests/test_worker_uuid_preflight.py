"""Pure logic checks for the A800 UUID gate; never starts Ray or CUDA."""

import argparse
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def load_preflight():
    path = Path(__file__).resolve().parents[1] / "integrations/rlinf_v03/worker_uuid_preflight.py"
    spec = importlib.util.spec_from_file_location("worker_uuid_preflight_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {spec.name: module}):
        spec.loader.exec_module(module)
    return module


class WorkerUuidPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.probe = load_preflight()

    def test_target_identity_must_match_host_ordinal(self):
        expected = "GPU-22222222-2222-2222-2222-222222222222"
        smi = f"7, {expected}, 0, 0\n6, GPU-00000000-0000-0000-0000-000000000000, 100, 5\n"
        self.assertEqual(self.probe.select_target_gpu(smi, 7, expected)["uuid"], expected)
        with self.assertRaisesRegex(ValueError, "UUID changed"):
            self.probe.select_target_gpu(smi, 6, expected)
        with self.assertRaisesRegex(ValueError, "absent"):
            self.probe.select_target_gpu(smi, 5, expected)

    def test_invalid_target_arguments_fail_before_ray_import(self):
        for bad in ("-1", "01", "1;rm"):
            with self.subTest(bad=bad), self.assertRaises(argparse.ArgumentTypeError):
                self.probe.gpu_ordinal(bad)
        for bad in ("7", "GPU-abc", "GPU-22222222-2222-2222-2222-222222222222;rm"):
            with self.subTest(bad=bad), self.assertRaises(argparse.ArgumentTypeError):
                self.probe.gpu_uuid(bad)
        with self.assertRaises(argparse.ArgumentTypeError):
            self.probe.node_ip("127.0.0.1")
        with patch.dict(sys.modules, {"ray": None}):
            load_preflight()  # Importing the entry point must not import Ray.

    def test_ray_head_uses_only_private_ports_and_one_logical_gpu(self):
        argv = self.probe.ray_start_argv(Path("/private/ray"), "192.0.2.10", Path("/safe/p/ab"))
        self.assertIn("--num-gpus=1", argv)
        self.assertIn("--port=23979", argv)
        self.assertIn(f"--temp-dir={Path('/safe/p/ab')}", argv)
        self.assertNotIn("--num-gpus=8", argv)

    def test_driver_drops_inherited_ray_address_and_pythonpath(self):
        with patch.dict(os.environ, {"PYTHONPATH": "wrong-code", "RAY_ADDRESS": "other-cluster"}):
            self.probe.set_driver_environment({"CUDA_VISIBLE_DEVICES": "7"}, "192.0.2.10")
            self.assertNotIn("PYTHONPATH", os.environ)
            self.assertEqual(os.environ["RAY_ADDRESS"], "192.0.2.10:23979")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "7")

    def test_cleanup_rejects_outside_or_replaced_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary) / "p"
            parent.mkdir()
            own = parent / "ab"
            own.mkdir()
            identity = (own.stat().st_dev, own.stat().st_ino)
            with patch.object(self.probe, "owned_private_directory"):
                with patch.object(self.probe.shutil, "rmtree") as remove:
                    with self.assertRaisesRegex(RuntimeError, "outside"):
                        self.probe.remove_own_ray_temp(Path(temporary) / "other", parent, identity, 1)
                    with self.assertRaisesRegex(RuntimeError, "changed identity"):
                        self.probe.remove_own_ray_temp(own, parent, (identity[0], identity[1] + 1), 1)
                    remove.assert_not_called()

    def test_ray_temp_reservation_skips_existing_short_name(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary) / "p"
            parent.mkdir()
            old = parent / "ab"
            old.mkdir()
            with patch.object(self.probe, "owned_private_directory"):
                with patch.object(self.probe.os, "fsencode", return_value=b"/short/p/cd"):
                    with patch.object(self.probe.secrets, "choice", side_effect=list("abcd")):
                        fresh, identity = self.probe.reserve_ray_temp(parent, 1)
            self.assertEqual(fresh.name, "cd")
            self.assertTrue(old.is_dir())
            self.assertEqual(identity, (fresh.stat().st_dev, fresh.stat().st_ino))


if __name__ == "__main__":
    unittest.main()
