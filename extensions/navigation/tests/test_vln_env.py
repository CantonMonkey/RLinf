"""Local, dependency-free contract checks for the staged RLinf R2R bridge."""

from __future__ import annotations

import base64
import gzip
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class FakeTensor:
    def __init__(self, shape, dtype, values=None):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.values = values

    def zero_(self):
        self.values = [0] * self.shape[0]
        return self

    def __iadd__(self, value):
        self.values = [item + value for item in self.values]
        return self

    def unsqueeze(self, axis):
        shape = list(self.shape)
        shape.insert(axis, 1)
        return FakeTensor(shape, self.dtype, self.values)


def nested_shape(values):
    if isinstance(values, list):
        return (len(values),) + (nested_shape(values[0]) if values else ())
    return ()


class FakePixels:
    def __init__(self, pixels):
        self.pixels = pixels
        self.shape = (len(pixels),)

    def copy(self):
        return self

    def reshape(self, *shape):
        self.shape = tuple(shape)
        return self


class FakeArray:
    def __init__(self, values):
        self.values = values
        self.shape = nested_shape(values)
        self.size = math.prod(self.shape)

    def reshape(self, *_shape):
        def flatten(item):
            if isinstance(item, list):
                return [x for child in item for x in flatten(child)]
            return [item]
        return flatten(self.values)

    def __getitem__(self, key):
        if key == (slice(None), 0, slice(None)):
            return FakeArray([item[0] for item in self.values])
        raise AssertionError(key)


class Config(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


def load_bridge():
    gym = types.ModuleType("gymnasium")
    gym.Env = type("Env", (), {})
    gym.spaces = SimpleNamespace(
        Discrete=lambda *_args, **_kwargs: None,
        Box=lambda *_args, **_kwargs: None,
        MultiBinary=lambda *_args, **_kwargs: None,
        Dict=lambda value: value,
    )
    numpy = types.ModuleType("numpy")
    numpy.uint8 = "uint8"
    numpy.int64 = "int64"
    numpy.integer = int
    numpy.iinfo = lambda _dtype: SimpleNamespace(max=2**63 - 1)
    numpy.frombuffer = lambda pixels, dtype: FakePixels(pixels)
    numpy.asarray = lambda values: values if isinstance(values, FakeArray) else FakeArray(values)
    numpy.issubdtype = lambda kind, parent: issubclass(kind, parent)
    torch = types.ModuleType("torch")
    torch.uint8 = "uint8"
    torch.int32 = "int32"
    torch.int64 = "int64"
    torch.float32 = "float32"
    torch.bool = "bool"
    torch.zeros = lambda count, dtype: FakeTensor((count,), dtype, [0] * count)
    torch.from_numpy = lambda array: FakeTensor(array.shape, "uint8", array.pixels)
    torch.tensor = lambda values, dtype: FakeTensor(nested_shape(values), dtype, values)
    local_path = Path(__file__).resolve().parents[1] / "integrations/rlinf_v03/vln_env.py"
    remote_path = Path(__file__).resolve().parents[2] / "rlinf/envs/vln/vln_env.py"
    path = local_path if local_path.is_file() else remote_path
    spec = importlib.util.spec_from_file_location("staged_vln_env_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"gymnasium": gym, "numpy": numpy, "torch": torch}):
        spec.loader.exec_module(module)
    return module


def fixture(root, split="train"):
    dataset = root / f"datasets/r2r/{split}/{split}.json.gz"
    dataset.parent.mkdir(parents=True)
    scene = root / "scene_datasets/mp3d/house/house.glb"
    scene.parent.mkdir(parents=True)
    scene.write_bytes(b"glb")
    scene.with_suffix(".navmesh").write_bytes(b"navmesh")
    words = ["<pad>", "walk", "left"]
    text = "walk left"
    document = {
        "instruction_vocab": {"word_list": words, "PAD_INDEX": 0},
        "episodes": [{
            "episode_id": "4862",
            "scene_id": "data/scene_datasets/mp3d/house/house.glb",
            "instruction": {"instruction_text": text, "instruction_tokens": [1, 2, 0]},
        }],
    }
    with gzip.open(dataset, "wt", encoding="utf-8") as stream:
        json.dump(document, stream)
    return dataset, scene, words, text


def policy_obs(module, identity):
    pixels = bytes(range(256)) * (64 * 64 * 3 // 256)
    return {
        "rgb": {
            "encoding": module.RGB_ENCODING,
            "shape_hwc": [64, 64, 3],
            "data_b64": base64.b64encode(pixels).decode("ascii"),
        },
        "instruction_ids": list(identity["instruction_ids"]),
        "instruction_mask": list(identity["instruction_mask"]),
    }


def ready(module, dataset, identity, split="train"):
    return {
        "ready": True,
        "actions": list(module.ACTIONS),
        "split": split,
        "dataset_path": str(dataset),
        "dataset_sha256": identity["dataset_sha256"],
        "episode_index": 0,
        "episode_id": identity["episode_id"],
        "scene_id": identity["scene_id"],
        "vocab_sha256": identity["vocab_sha256"],
        "vocab_size": identity["vocab_size"],
        "instruction_sha256": identity["instruction_sha256"],
        "rgb_size": 64,
        "max_instruction_tokens": 200,
        "success_distance_m": 3.0,
        "train_expert_available": split == "train",
    }


class VlnEnvContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bridge = load_bridge()

    def test_identity_and_ready_are_bound_to_selected_r2r_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, scene, words, text = fixture(root)
            identity = self.bridge.selected_r2r_identity(dataset, root / "scene_datasets", 0, 200)
            self.assertEqual(identity["dataset_sha256"], self.bridge.sha256_file(dataset))
            self.assertEqual(identity["scene_id"], str(scene))
            self.assertEqual(identity["episode_id"], "4862")
            self.assertEqual(identity["instruction_sha256"], hashlib.sha256(text.encode()).hexdigest())
            self.assertEqual(identity["vocab_size"], len(words))
            self.assertEqual(identity["instruction_ids"][:4], [1, 2, 0, 0])
            handshake = ready(self.bridge, dataset, identity)
            validate = self.bridge.validate_service_ready
            validate(handshake, split="train", dataset_path=dataset, identity=identity,
                     rgb_size=64, token_width=200)
            for key, wrong in (
                ("split", "val_unseen"), ("dataset_path", str(root / "other.json.gz")),
                ("dataset_sha256", "0" * 64), ("episode_index", 1),
                ("episode_id", "631"), ("scene_id", str(root / "other.glb")),
                ("vocab_sha256", "1" * 64), ("vocab_size", 99),
                ("instruction_sha256", "2" * 64),
                ("rgb_size", 128), ("max_instruction_tokens", 128),
                ("success_distance_m", 0.2),
            ):
                with self.subTest(key=key), self.assertRaises(RuntimeError):
                    validate(dict(handshake, **{key: wrong}), split="train",
                             dataset_path=dataset, identity=identity,
                             rgb_size=64, token_width=200)

    def test_base64_rgb_and_exact_policy_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, *_ = fixture(root)
            identity = self.bridge.selected_r2r_identity(dataset, root / "scene_datasets", 0, 200)
            payload = policy_obs(self.bridge, identity)
            pixels, ids, mask = self.bridge.validate_policy_payload(
                payload, rgb_size=64, token_width=200, identity=identity
            )
            self.assertEqual(len(pixels), 64 * 64 * 3)
            self.assertEqual(ids[:3], [1, 2, 0])
            self.assertEqual(mask[:3], [True, True, False])
            for key in ("pointgoal", "goal", "pose", "reference_path", "depth"):
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "unexpected"):
                    self.bridge.validate_policy_payload(
                        dict(payload, **{key: 42}), rgb_size=64, token_width=200,
                        identity=identity,
                    )
            bad_rgb = dict(payload["rgb"], data_b64="!")
            with self.assertRaisesRegex(ValueError, "base64"):
                self.bridge.validate_policy_payload(
                    dict(payload, rgb=bad_rgb), rgb_size=64, token_width=200,
                    identity=identity,
                )
            bad_rgb = dict(payload["rgb"], data_b64=base64.b64encode(b"short").decode())
            with self.assertRaisesRegex(ValueError, "byte count"):
                self.bridge.validate_policy_payload(
                    dict(payload, rgb=bad_rgb), rgb_size=64, token_width=200,
                    identity=identity,
                )
            bad_ids = list(identity["instruction_ids"])
            bad_ids[0] = 2
            with self.assertRaisesRegex(RuntimeError, "changed"):
                self.bridge.validate_policy_payload(
                    dict(payload, instruction_ids=bad_ids), rgb_size=64, token_width=200,
                    identity=identity,
                )
            bad_mask = list(identity["instruction_mask"])
            bad_mask[2] = True
            with self.assertRaisesRegex(ValueError, "contract"):
                self.bridge.validate_policy_payload(
                    dict(payload, instruction_mask=bad_mask), rgb_size=64, token_width=200,
                    identity=identity,
                )

    def test_rlinf_reset_chunk_shapes_and_frozen_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, scene, *_ = fixture(root)
            python_bin = root / "python"
            script = root / "project/repos/rlinf/scripts/service.py"
            python_bin.touch()
            script.parent.mkdir(parents=True)
            script.touch()
            sim_python = root / "habitat-sim/src_python"
            env_lib = root / "habitat-env/lib"
            vendor = root / "mesa/50_mesa.json"
            sim_python.mkdir(parents=True)
            env_lib.mkdir(parents=True)
            vendor.parent.mkdir(parents=True)
            vendor.touch()
            identity = self.bridge.selected_r2r_identity(dataset, root / "scene_datasets", 0, 200)
            handshake = ready(self.bridge, dataset, identity)
            cfg = Config(
                seed=7, auto_reset=False, max_episode_steps=4,
                python_bin=str(python_bin), server_script=str(script),
                asset_root=str(root), dataset_split="train", episode_index=0,
                video_cfg=Config(save_video=False), success_distance_m=3.0,
                habitat_sim_python=str(sim_python), habitat_env_lib=str(env_lib),
                egl_vendor_json=str(vendor),
                startup_timeout_s=180.0,
            )
            original_resolve = Path.resolve

            def reject_python_bin_resolve(path, *args, **kwargs):
                if path == python_bin:
                    raise AssertionError("declared venv entrypoint must not be resolved")
                return original_resolve(path, *args, **kwargs)

            with patch.object(self.bridge.Path, "resolve", new=reject_python_bin_resolve), \
                    patch.object(self.bridge.subprocess, "Popen", return_value=object()) as popen, \
                    patch.object(self.bridge.VLNEnv, "_read_response", return_value=handshake) as read_response:
                env = self.bridge.VLNEnv(cfg, 1, 0, 1, None)
            read_response.assert_called_once_with(timeout=180.0)
            command = popen.call_args.args[0]
            self.assertEqual(command[0], str(python_bin.absolute()))
            self.assertEqual(command[command.index("--split") + 1], "train")
            self.assertEqual(command[command.index("--episode-index") + 1], "0")
            self.assertEqual(command[command.index("--rgb-size") + 1], "64")
            child_env = popen.call_args.kwargs["env"]
            self.assertEqual(child_env["PYTHONPATH"], str(sim_python))
            self.assertEqual(child_env["LD_LIBRARY_PATH"], str(env_lib))
            self.assertEqual(child_env["__EGL_VENDOR_LIBRARY_FILENAMES"], str(vendor))
            for key, expected in (
                ("CUDA_VISIBLE_DEVICES", ""), ("LIBGL_ALWAYS_SOFTWARE", "1"),
                ("EGL_PLATFORM", "surfaceless"), ("HABITAT_SIM_SOFTWARE_EGL", "1"),
                ("OMP_NUM_THREADS", "1"), ("OPENBLAS_NUM_THREADS", "1"),
            ):
                self.assertEqual(child_env[key], expected)
            self.assertIsNot(child_env, os.environ)
            self.assertEqual(set(env.observation_space), self.bridge.OBS_KEYS)
            obs_payload = policy_obs(self.bridge, identity)
            info = {
                "episode_id": "4862", "scene_id": str(scene),
                "instruction_sha256": identity["instruction_sha256"],
                "terminal_frozen": False, "success": 0.0, "spl": 0.0,
                "path_length": 0.0, "collision_count": 0.0,
                "collision": False,
            }
            reset_response = {"obs": obs_payload, "info": info}
            terminal = {
                "obs": obs_payload, "info": dict(info, path_length=0.25),
                "reward": 0.0, "terminated": True, "truncated": False,
            }
            frozen = {
                "obs": obs_payload,
                "info": dict(terminal["info"], terminal_frozen=True),
                "reward": 0.0, "terminated": False, "truncated": False,
            }
            responses = iter((reset_response, terminal, frozen))
            env._request = lambda _payload: next(responses)
            obs, reset_info = env.reset()
            self.assertEqual(set(obs), self.bridge.OBS_KEYS)
            self.assertEqual(obs["rgb"].shape, (1, 64, 64, 3))
            self.assertEqual(obs["rgb"].dtype, "uint8")
            self.assertEqual(obs["instruction_ids"].shape, (1, 200))
            self.assertEqual(obs["instruction_ids"].dtype, "int64")
            self.assertEqual(obs["instruction_mask"].shape, (1, 200))
            self.assertEqual(obs["instruction_mask"].dtype, "bool")
            self.assertEqual(reset_info, {"episode_id": "4862", "scene_id": str(scene)})
            result = env.chunk_step([[[0]]])
            self.assertEqual([tensor.shape for tensor in result[1:4]], [(1, 1)] * 3)
            self.assertEqual(result[4][0]["episode"]["episode_index"].values, [0])
            self.assertEqual(env.elapsed_steps.values, [1])
            frozen_result = env.chunk_step([[[2]]])
            self.assertEqual(frozen_result[1].values, [0.0])
            self.assertEqual(env.elapsed_steps.values, [1])
            self.assertEqual(frozen_result[4][0]["episode"]["path_length"].values, [0.25])

    def test_val_seen_is_a_read_only_supported_r2r_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, *_ = fixture(root, split="val_seen")
            python_bin = root / "python"
            script = root / "project/repos/rlinf/scripts/service.py"
            python_bin.touch()
            script.parent.mkdir(parents=True)
            script.touch()
            sim_python = root / "habitat-sim/src_python"
            env_lib = root / "habitat-env/lib"
            vendor = root / "mesa/50_mesa.json"
            sim_python.mkdir(parents=True)
            env_lib.mkdir(parents=True)
            vendor.parent.mkdir(parents=True)
            vendor.touch()
            identity = self.bridge.selected_r2r_identity(dataset, root / "scene_datasets", 0, 200)
            handshake = ready(self.bridge, dataset, identity, split="val_seen")
            cfg = Config(
                seed=7, auto_reset=False, max_episode_steps=4,
                python_bin=str(python_bin), server_script=str(script),
                asset_root=str(root), dataset_split="val_seen", episode_index=0,
                video_cfg=Config(save_video=False), success_distance_m=3.0,
                habitat_sim_python=str(sim_python), habitat_env_lib=str(env_lib),
                egl_vendor_json=str(vendor),
            )
            with patch.object(self.bridge.subprocess, "Popen", return_value=object()) as popen, \
                    patch.object(self.bridge.VLNEnv, "_read_response", return_value=handshake):
                env = self.bridge.VLNEnv(cfg, 1, 0, 1, None)
            command = popen.call_args.args[0]
            self.assertEqual(env.dataset_split, "val_seen")
            self.assertEqual(env.dataset_path, dataset.resolve())
            self.assertEqual(command[command.index("--split") + 1], "val_seen")
            with self.assertRaisesRegex(RuntimeError, "train split"):
                env.expert_waypoint(1.0)

    def test_train_only_expert_waypoint_schema(self):
        env = object.__new__(self.bridge.VLNEnv)
        env.dataset_split = "train"
        response = {
            "schema_version": 1,
            "source": "train-shortest-path",
            "lookahead_m": 1.0,
            "waypoint_local_m": [0.8, -0.1],
            "bearing_rad": -0.124,
            "target_distance_m": 0.806,
            "distance_to_goal_m": 4.6,
            "stop": False,
        }
        requests = []
        env._request = lambda payload: requests.append(payload) or response
        label = env.expert_waypoint(1.0)
        self.assertEqual(requests, [{"cmd": "expert_waypoint", "lookahead_m": 1.0}])
        self.assertEqual(label["waypoint_local_m"], [0.8, -0.1])
        env.dataset_split = "val_unseen"
        with self.assertRaisesRegex(RuntimeError, "train split"):
            env.expert_waypoint(1.0)

    def test_step_rejects_episode_instruction_or_terminal_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, scene, *_ = fixture(root)
            identity = self.bridge.selected_r2r_identity(dataset, root / "scene_datasets", 0, 200)
            env = object.__new__(self.bridge.VLNEnv)
            env.identity = identity
            env.rgb_size = 64
            env.token_width = 200
            env._elapsed_steps = FakeTensor((1,), "int32", [0])
            env._terminal_metrics = None
            env._is_start = True
            obs = policy_obs(self.bridge, identity)
            info = {
                "episode_id": "4862", "scene_id": str(scene),
                "instruction_sha256": identity["instruction_sha256"],
                "terminal_frozen": False, "success": 0.0, "spl": 0.0,
                "path_length": 0.0, "collision_count": 0.0,
                "collision": False,
            }
            response = {
                "obs": obs, "info": info, "reward": 0.0,
                "terminated": False, "truncated": False,
            }
            for changed_info in (
                dict(info, episode_id="631"),
                dict(info, scene_id=str(root / "wrong.glb")),
                dict(info, instruction_sha256="0" * 64),
            ):
                env._request = lambda _payload, data=changed_info: dict(response, info=data)
                with self.subTest(changed_info=changed_info), \
                        patch.object(env, "close") as close, \
                        self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                    env.step(1)
                close.assert_called_once()
            env._terminal_metrics = {
                "success": 0.0, "spl": 0.0, "path_length": 0.25,
                "collision_count": 0.0,
            }
            drift = dict(info, terminal_frozen=True, path_length=0.5)
            env._request = lambda _payload: dict(response, info=drift)
            with patch.object(env, "close") as close, \
                    self.assertRaisesRegex(RuntimeError, "terminal metrics changed"):
                env.step(2)
            close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
