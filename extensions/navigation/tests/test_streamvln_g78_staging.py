from __future__ import annotations

import base64
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
import types
import unittest
from unittest.mock import patch

from scripts.habitat_r2r_depth_passability_service import DepthPassabilitySession


ROOT = Path(__file__).resolve().parents[1]
INTEGRATION = ROOT / "integrations" / "rlinf_v03"


class FakeMetricDepth:
    def __init__(self, values):
        self.values = list(values)
        self.shape = (len(self.values),)

    def copy(self):
        return self

    def reshape(self, *shape):
        if math.prod(shape) != len(self.values):
            raise ValueError("reshape mismatch")
        self.shape = tuple(shape)
        return self

    def min(self):
        return min(self.values)

    def max(self):
        return max(self.values)


class FakeFinite:
    def __init__(self, values):
        self.values = values

    def all(self):
        return all(math.isfinite(value) for value in self.values)


def load_depth_decoder():
    """Load the decoder without requiring the local ML environment."""
    gym = types.ModuleType("gymnasium")
    gym.Env = type("Env", (), {})
    gym.spaces = types.SimpleNamespace()
    torch = types.ModuleType("torch")
    numpy = types.ModuleType("numpy")
    numpy.uint8 = "uint8"
    numpy.int64 = "int64"
    numpy.integer = int
    numpy.iinfo = lambda _dtype: types.SimpleNamespace(max=2**63 - 1)
    numpy.asarray = lambda values: values
    numpy.issubdtype = lambda kind, parent: issubclass(kind, parent)
    numpy.frombuffer = lambda raw, dtype: FakeMetricDepth(
        struct.unpack("<" + "f" * (len(raw) // 4), raw)
    )
    numpy.isfinite = lambda depth: FakeFinite(depth.values)
    path = INTEGRATION / "vln_env.py"
    spec = importlib.util.spec_from_file_location("g78_depth_decoder", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules, {"gymnasium": gym, "numpy": numpy, "torch": torch}
    ):
        spec.loader.exec_module(module)
    return module


def depth_payload(values, size=64):
    values = list(values)
    return {
        "encoding": "base64/raw-f32le-hw",
        "shape_hw": [size, size],
        "min_m": 0.0,
        "max_m": 10.0,
        "data_b64": base64.b64encode(
            struct.pack("<" + "f" * len(values), *values)
        ).decode("ascii"),
    }


class FakeDepthBackend:
    def reset(self):
        return {
            "policy_obs": {
                "rgb": {
                    "encoding": "base64/raw-u8-hwc",
                    "shape_hwc": [1, 1, 3],
                    "data_b64": "AAAA",
                },
                "instruction_ids": [1, 0],
                "instruction_mask": [True, False],
            },
            "episode_id": "1",
            "scene_id": "/scene.glb",
            "instruction_sha256": "hash",
            "geodesic_distance": 4.0,
            "success": False,
            "spl": 0.0,
            "path_length": 0.0,
            "collision": False,
            "collision_count": 0,
            "episode_over": False,
        }

    def observe_aligned_depth(self):
        return {
            "schema_version": 1,
            "source": "runtime-aligned-metric-depth",
            "state_unchanged": True,
            "metric_depth": depth_payload([1.0] * (64 * 64)),
        }

    def probe_depth_candidates(self, candidates):
        return {
            "schema_version": 1,
            "source": "train-simulator-aligned-depth-branch-rollout",
            "policy_observation_unchanged": True,
            "state_restored": True,
            "max_position_restore_error_m": 0.0,
            "max_rotation_restore_error": 0.0,
            "candidates": [],
        }

    def close(self):
        pass


class G78DepthControllerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bridge = load_depth_decoder()

    def test_depth_stays_outside_policy_observation(self):
        session = DepthPassabilitySession(FakeDepthBackend(), max_steps=4)
        with self.assertRaisesRegex(RuntimeError, "reset must precede"):
            session.observe_aligned_depth()
        with self.assertRaisesRegex(RuntimeError, "reset must precede"):
            session.probe_depth_candidates([])
        reset = session.reset()
        self.assertEqual(
            set(reset["obs"]), {"rgb", "instruction_ids", "instruction_mask"}
        )
        observation = session.observe_aligned_depth()
        self.assertTrue(observation["state_unchanged"])
        self.assertNotIn("obs", observation)
        probe = session.probe_depth_candidates([])
        self.assertTrue(probe["state_restored"])
        self.assertNotIn("obs", probe)

    def test_metric_depth_decoder_accepts_f32le_and_rejects_bad_values(self):
        depth = [index * (10.0 / 15.0) for index in range(16)]
        decoded = self.bridge.decode_metric_depth_payload(
            depth_payload(depth, size=4), depth_size=4
        )
        self.assertEqual(decoded.shape, (4, 4))
        for actual, expected in zip(decoded.values, depth):
            self.assertAlmostEqual(actual, expected, places=5)
        short = dict(
            depth_payload(depth, size=4),
            data_b64=base64.b64encode(b"short").decode(),
        )
        with self.assertRaisesRegex(ValueError, "byte count"):
            self.bridge.decode_metric_depth_payload(short, depth_size=4)
        bad = list(depth)
        bad[0] = float("nan")
        with self.assertRaisesRegex(ValueError, "outside configured meters"):
            self.bridge.decode_metric_depth_payload(
                depth_payload(bad, size=4), depth_size=4
            )

    def test_runner_enforces_causal_depth_and_conditional_navigation(self):
        runner = (INTEGRATION / "run_streamvln_depth_g78.py").read_text()
        collection = runner.split("def collect_episode", 1)[1].split(
            "def save_dataset", 1
        )[0]
        navigation = runner.split("def navigation_episode", 1)[1].split(
            "def parse_args", 1
        )[0]
        self.assertIn("env.probe_depth_candidates(candidates)", collection)
        self.assertNotIn("probe_depth_candidates", navigation)
        self.assertLess(
            navigation.index("turn_step = execute_sequence"),
            navigation.index("metric_depth = env.observe_aligned_depth()"),
        )
        self.assertLess(
            navigation.index("metric_depth = env.observe_aligned_depth()"),
            navigation.index("predict_depth_scores(model, score_rows"),
        )
        self.assertIn("if representation_ready:", runner)
        self.assertIn('"navigation_condition": "all-representation-gates-pass"', runner)
        self.assertIn('"branch_rollout_used_during_navigation_gate": False', runner)
        self.assertIn('"val_unseen_episode_count": 0', runner)
        self.assertIn('"depth_role": "explicit-local-controller-diagnostic"', runner)
        self.assertIn('"g8_ready": False', runner)

    def test_model_and_sensor_sources_make_depth_explicit_opt_in(self):
        model = (INTEGRATION / "streamvln_depth_g78.py").read_text()
        sensor = (ROOT / "scripts" / "habitat_r2r_rgb_instruction_smoke.py").read_text()
        service = (ROOT / "scripts" / "habitat_r2r_service.py").read_text()
        depth_service = (
            ROOT / "scripts" / "habitat_r2r_depth_passability_service.py"
        ).read_text()
        smoke = (INTEGRATION / "smoke_streamvln_depth_g78.py").read_text()
        self.assertIn("class DepthPassabilityNet", model)
        self.assertIn("def per_scene_metrics", model)
        self.assertIn("def threshold_sensitivity", model)
        self.assertIn("include_depth=False", sensor)
        self.assertIn('expected = {"rgb_sensor", "depth_sensor"}', sensor)
        self.assertIn('expected.add("depth")', service)
        self.assertIn("policy_obs = make_policy_obs", service)
        self.assertIn("self.env.sim.get_observations_at()", depth_service)
        self.assertNotIn("self.env.sim.get_sensor_observations()", depth_service)
        self.assertIn('"depth_outside_policy_observation"', smoke)
        self.assertIn('"straight_matches_initial"', smoke)
        self.assertIn('"branch_turned_matches_actual"', smoke)

    def test_manifest_preserves_fixed_scene_disjoint_split(self):
        manifest = json.loads(
            (INTEGRATION / "streamvln_g78_experiment.json").read_text()
        )
        split = manifest["g78_passability_split"]
        self.assertEqual(len(split["train_scenes"]), 6)
        self.assertEqual(len(split["train_dev_scenes"]), 2)
        self.assertFalse(set(split["train_scenes"]) & set(split["train_dev_scenes"]))
        self.assertEqual(
            split["train_scenes"], [row["scene"] for row in manifest["train"][:6]]
        )
        self.assertEqual(
            split["train_dev_scenes"], [row["scene"] for row in manifest["train"][6:]]
        )


if __name__ == "__main__":
    unittest.main()
