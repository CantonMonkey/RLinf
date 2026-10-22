"""Check the RLinf bridge handshake without importing Torch or Habitat."""

import ast
import builtins
import gzip
import importlib.util
import json
import math
import sys
import tempfile
import types
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class Config(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


def load_bridge():
    gym = types.ModuleType("gymnasium")
    gym.Env = type("Env", (), {})
    gym.spaces = SimpleNamespace(
        Discrete=lambda *_args, **_kwargs: None,
        Box=lambda *_args, **_kwargs: None,
        Dict=lambda *_args, **_kwargs: None,
    )
    numpy = types.ModuleType("numpy")
    numpy.inf = math.inf
    numpy.float32 = float
    torch = types.ModuleType("torch")
    torch.int32 = 0
    torch.zeros = lambda *_args, **_kwargs: SimpleNamespace(zero_=lambda: None)
    path = Path(__file__).resolve().parents[1] / "integrations/rlinf_v03/pointnav_env.py"
    spec = importlib.util.spec_from_file_location("pointnav_bridge_split_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"gymnasium": gym, "numpy": numpy, "torch": torch}):
        spec.loader.exec_module(module)
    return module


def load_node_preflight():
    path = Path(__file__).resolve().parents[1] / "integrations/rlinf_v03/node_preflight.py"
    spec = importlib.util.spec_from_file_location("pointnav_node_preflight_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_eval_helpers():
    """Execute pure evaluation helpers without importing Hydra, Torch, or Ray."""
    path = Path(__file__).resolve().parents[1] / "integrations/rlinf_v03/train_embodied_agent.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {
        "collect_eval_episode_metrics",
        "write_eval_episode_metrics",
        "evaluate_with_episode_metrics",
    }
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {"Path": Path, "gzip": gzip, "json": json, "math": math, "uuid": uuid}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class FakeTensor:
    def __init__(self, values):
        self.values = list(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def reshape(self, *_args):
        return self

    def tolist(self):
        return list(self.values)


class BridgeSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bridge = load_bridge()

    def test_ready_rejects_wrong_split_path_or_episodes(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "train.json.gz"
            dataset.touch()
            ready = {
                "ready": True,
                "actions": ["stop", "move_forward", "turn_left", "turn_right"],
                "split": "train",
                "dataset_path": str(dataset),
                "selected_indices": [50, 51],
                "episode_ids": ["50", "51"],
                "scene_id": "skokloster-castle.glb",
            }
            check = self.bridge.validate_service_ready
            self.assertEqual(
                check(ready, split="train", dataset_path=dataset, selected_indices=[50, 51]),
                (frozenset({"50", "51"}), "skokloster-castle.glb"),
            )
            for changed, value in (
                ("split", "val"),
                ("dataset_path", str(Path(directory) / "val.json.gz")),
                ("selected_indices", [0, 1]),
                ("episode_ids", ["50"]),
                ("scene_id", ""),
            ):
                with self.subTest(changed=changed), self.assertRaises(RuntimeError):
                    check(
                        dict(ready, **{changed: value}),
                        split="train", dataset_path=dataset,
                        selected_indices=[50, 51],
                    )

    def test_launch_explicit_split_and_reject_wrong_reset_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python_bin = root / "python"
            service = root / "pointnav_service.py"
            dataset = root / "datasets/pointnav/habitat-test-scenes/v1/train/train.json.gz"
            dataset.parent.mkdir(parents=True)
            for path in (python_bin, service, dataset):
                path.touch()
            ready = {
                "ready": True,
                "actions": ["stop", "move_forward", "turn_left", "turn_right"],
                "split": "train",
                "dataset_path": str(dataset),
                "selected_indices": [50, 51],
                "episode_ids": ["50", "51"],
                "scene_id": "skokloster-castle.glb",
            }
            cfg = Config(
                seed=7, auto_reset=False, max_episode_steps=8,
                python_bin=str(python_bin), server_script=str(service),
                asset_root=str(root), dataset_split="train",
                episode_indices=[50, 51], video_cfg=Config(save_video=False),
            )
            bridge = self.bridge.PointNavEnv
            with patch.object(self.bridge.subprocess, "Popen", return_value=object()) as popen, \
                    patch.object(bridge, "_read_response", return_value=ready), \
                    patch.object(bridge, "close") as close:
                env = bridge(cfg, 1, 0, 1, None)
                command = popen.call_args.args[0]
                self.assertEqual(command[command.index("--split") + 1], "train")
                self.assertEqual(command[command.index("--episode-indices") + 1], "50,51")
                env._request = lambda _: {
                    "info": {"episode_id": "0", "scene_id": "skokloster-castle.glb"}
                }
                with self.assertRaisesRegex(RuntimeError, "reset episode mismatch"):
                    env.reset()
                close.assert_called_once()

    def test_terminal_metric_carries_validated_official_file_index(self):
        env = object.__new__(self.bridge.PointNavEnv)
        env._current_episode_id = "room-episode-0"
        env._selected_scene_id = "room.glb"
        env._episode_index_by_id = {"room-episode-0": 50}
        raw = {
            "episode_id": "room-episode-0", "scene_id": "room.glb",
            "success": False, "spl": 0.0, "geodesic_distance": 1.5,
            "path_length": 0.25, "collision_count": 1,
        }
        fake_torch = SimpleNamespace(
            float32="float32", int64="int64",
            tensor=lambda values, dtype: FakeTensor(
                [float(value) if dtype == "float32" else value for value in values]
            ),
        )
        with patch.object(self.bridge, "torch", fake_torch):
            info = env._episode_info(raw)
            self.assertEqual(info["episode_index"].tolist(), [50])
            self.assertEqual(info["success"].tolist(), [0.0])
            with self.assertRaisesRegex(RuntimeError, "terminal episode mismatch"):
                env._episode_info(dict(raw, episode_id="room-episode-1"))
            with self.assertRaisesRegex(RuntimeError, "terminal episode mismatch"):
                env._episode_info(dict(raw, scene_id="other.glb"))

    def test_step_rejects_mid_episode_identity_change(self):
        env = object.__new__(self.bridge.PointNavEnv)
        env._current_episode_id = "room-episode-0"
        env._selected_scene_id = "room.glb"
        env._request = lambda _payload: {
            "info": {"episode_id": "room-episode-1", "scene_id": "room.glb"}
        }
        scalar = SimpleNamespace(size=1, reshape=lambda *_args: [1])
        fake_numpy = SimpleNamespace(
            asarray=lambda _actions: scalar,
            integer=int,
            issubdtype=lambda *_args: True,
        )
        with patch.object(self.bridge, "np", fake_numpy), \
                patch.object(env, "close") as close:
            with self.assertRaisesRegex(RuntimeError, "step episode mismatch"):
                env.step(1)
            close.assert_called_once()


class EvalEpisodeArtifactTests(unittest.TestCase):
    @staticmethod
    def fixture(root):
        dataset = root / "datasets/pointnav/habitat-test-scenes/v1/val/val.json.gz"
        dataset.parent.mkdir(parents=True)
        episodes = [
            {"episode_id": f"id-{index}", "scene_id": "room.glb"}
            for index in range(55)
        ]
        with gzip.open(dataset, "wt", encoding="utf-8") as stream:
            json.dump({"episodes": episodes}, stream)
        cfg = Config(
            env=Config(eval=Config(
                dataset_split="val", total_num_envs=1,
                episode_indices=[50, 51, 52, 53, 54], rollout_epoch=5,
                asset_root=str(root),
            )),
            runner=Config(
                ckpt_path=str(root / "checkpoint.pt"),
                logger=Config(log_path=str(root / "runs"), experiment_name="eval-fixture"),
            ),
        )
        shard = {
            "episode_index": FakeTensor([50, 51, 52, 53, 54]),
            "success": FakeTensor([0.0, 1.0, 0.0, 0.0, 1.0]),
            "spl": FakeTensor([0.0, 0.5, 0.0, 0.0, 0.25]),
            "geodesic_distance": FakeTensor([2.0, 0.1, 3.0, 4.0, 0.2]),
            "path_length": FakeTensor([1.0, 1.5, 2.0, 2.5, 3.0]),
            "collision_count": FakeTensor([0.0, 1.0, 0.0, 1.0, 0.0]),
        }
        return cfg, shard

    def test_saves_distinct_val_episode_ids_and_preserves_aggregate_metrics(self):
        helpers = load_eval_helpers()
        with tempfile.TemporaryDirectory() as directory:
            cfg, shard = self.fixture(Path(directory))
            record, metric_shards = helpers["collect_eval_episode_metrics"](cfg, [shard])
            self.assertEqual(record["observed_episode_indices"], [50, 51, 52, 53, 54])
            self.assertEqual([row["episode_id"] for row in record["episodes"]],
                             [f"id-{index}" for index in range(50, 55)])
            self.assertEqual([row["success"] for row in record["episodes"]],
                             [0.0, 1.0, 0.0, 0.0, 1.0])
            self.assertNotIn("episode_index", metric_shards[0])
            self.assertIs(metric_shards[0]["success"], shard["success"])
            output = helpers["write_eval_episode_metrics"](cfg, record, 1)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), record)
            self.assertEqual(len(list(output.parent.glob("*.json"))), 1)
            self.assertEqual(list(output.parent.glob("*.tmp")), [])

    def test_rejects_missing_duplicate_misaligned_or_nonfinite_episodes(self):
        collect = load_eval_helpers()["collect_eval_episode_metrics"]
        with tempfile.TemporaryDirectory() as directory:
            cfg, shard = self.fixture(Path(directory))
            with self.assertRaisesRegex(ValueError, "cover each configured val index"):
                collect(cfg, [dict(shard, episode_index=FakeTensor([50, 51, 52, 53, 53]))])
            with self.assertRaisesRegex(ValueError, "different lengths"):
                collect(cfg, [dict(shard, spl=FakeTensor([0.0]))])
            with self.assertRaisesRegex(ValueError, "Nonfinite"):
                collect(cfg, [dict(shard, success=FakeTensor([0.0, float("nan"), 0.0, 0.0, 1.0]))])
            cfg.env.eval.rollout_epoch = 4
            with self.assertRaisesRegex(ValueError, "rollout_epoch"):
                collect(cfg, [shard])

    def test_eval_entry_keeps_original_means_and_writes_before_reporting(self):
        helpers = load_eval_helpers()
        with tempfile.TemporaryDirectory() as directory:
            cfg, shard = self.fixture(Path(directory))
            events = []

            class Handle:
                def __init__(self, name, result):
                    self.name = name
                    self.result = result

                def wait(self):
                    events.append(f"wait:{self.name}")
                    return self.result

            def aggregate(shards):
                events.append("aggregate")
                self.assertEqual(len(shards), 1)
                self.assertNotIn("episode_index", shards[0])
                values = shards[0]["success"].tolist()
                return {"success": sum(values) / len(values), "num_trajectories": len(values)}

            helpers["compute_evaluate_metrics"] = aggregate
            runner = Config(
                env=Config(evaluate=lambda **_kwargs: Handle("env", [shard])),
                rollout=Config(evaluate=lambda **_kwargs: Handle("rollout", None)),
                env_channel=object(), rollout_channel=object(), global_step=3,
            )
            metrics, output = helpers["evaluate_with_episode_metrics"](runner, cfg)
            self.assertEqual(events, ["wait:env", "wait:rollout", "aggregate"])
            self.assertEqual(metrics, {"success": 0.4, "num_trajectories": 5})
            record = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(record["summary"], metrics)
            self.assertEqual(record["global_step"], 3)
            self.assertEqual(len(record["episodes"]), 5)


class NodePreflightEvalArgTests(unittest.TestCase):
    def test_eval_checkpoint_must_exist_under_isolated_node_root(self):
        preflight = load_node_preflight()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            checkpoint = root / "runs" / "global_step_1"
            actor = checkpoint / "actor"
            actor.mkdir(parents=True)
            weights = actor / "model_state_dict/full_weights.pt"
            weights.parent.mkdir()
            weights.write_bytes(b"trusted model fixture")
            dcp = actor / "dcp_checkpoint"
            dcp.mkdir()
            (dcp / ".metadata").write_bytes(b"metadata")
            (dcp / "__0_0.distcp").write_bytes(b"checkpoint shard")
            self.assertEqual(preflight.eval_checkpoint_overrides(root, None), [])
            with patch.object(preflight, "verify_eval_checkpoint_content", return_value=15) as verify:
                self.assertEqual(
                    preflight.eval_checkpoint_overrides(root, checkpoint),
                    [f"runner.resume_dir={checkpoint}", f"runner.ckpt_path={weights}"],
                )
                verify.assert_called_once_with(actor)
            weights.unlink()
            with self.assertRaisesRegex(ValueError, "full model weights"):
                preflight.eval_checkpoint_overrides(root, checkpoint)
            weights.touch()
            with self.assertRaisesRegex(ValueError, "empty"):
                preflight.eval_checkpoint_overrides(root, checkpoint)
            weights.write_bytes(b"trusted model fixture")
            (dcp / ".metadata").unlink()
            with self.assertRaisesRegex(ValueError, "DCP checkpoint"):
                preflight.eval_checkpoint_overrides(root, checkpoint)
            (dcp / ".metadata").write_bytes(b"metadata")
            with self.assertRaises(FileNotFoundError):
                preflight.eval_checkpoint_overrides(root, root / "missing")
            invalid_name = root / "runs" / "checkpoint"
            invalid_name.mkdir()
            with self.assertRaisesRegex(ValueError, "global_step_N"):
                preflight.eval_checkpoint_overrides(root, invalid_name)
            missing_actor = root / "runs" / "global_step_2"
            missing_actor.mkdir()
            with self.assertRaisesRegex(ValueError, "actor/"):
                preflight.eval_checkpoint_overrides(root, missing_actor)
            with tempfile.TemporaryDirectory() as outside:
                with self.assertRaisesRegex(ValueError, "under node root"):
                    preflight.eval_checkpoint_overrides(root, Path(outside))


    def test_cpu_verifier_rejects_mismatched_or_nonfinite_model_tensors(self):
        preflight = load_node_preflight()

        class FakeTensor:
            def __init__(self, values):
                self.values = tuple(values)
                self.shape = (len(self.values),)
                self.dtype = "float32"

        fake_torch = SimpleNamespace(
            Tensor=FakeTensor,
            isfinite=lambda tensor: SimpleNamespace(
                all=lambda: all(math.isfinite(value) for value in tensor.values)
            ),
            equal=lambda left, right: left.values == right.values,
        )
        full = {"layer": FakeTensor([1.0, 2.0])}
        dcp = {"layer": FakeTensor([1.0, 2.0])}
        self.assertEqual(preflight.compare_checkpoint_model_tensors(full, dcp, fake_torch), 1)
        dcp["layer"] = FakeTensor([1.0, 3.0])
        with self.assertRaisesRegex(ValueError, "differ at model tensor"):
            preflight.compare_checkpoint_model_tensors(full, dcp, fake_torch)
        dcp["layer"] = FakeTensor([float("nan"), 2.0])
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            preflight.compare_checkpoint_model_tensors(full, dcp, fake_torch)
        with self.assertRaisesRegex(ValueError, "different model keys"):
            preflight.compare_checkpoint_model_tensors(full, {"other": FakeTensor([1.0])}, fake_torch)

    def test_cpu_verifier_requires_torch_and_rlinf_before_eval(self):
        preflight = load_node_preflight()
        original_import = builtins.__import__

        def no_torch(name, *args, **kwargs):
            if name == "torch":
                raise ModuleNotFoundError("No module named 'torch'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=no_torch):
            with self.assertRaisesRegex(RuntimeError, "requires installed Torch and RLinf"):
                preflight.verify_eval_checkpoint_content(Path("unused"))


class EvalEntryCheckpointTests(unittest.TestCase):
    def test_only_eval_requires_matched_actor_and_rollout_checkpoint_before_ray(self):
        # Execute the actual entry guard without importing RLinf or starting Ray.
        source = Path(__file__).resolve().parents[1] / "integrations/rlinf_v03/train_embodied_agent.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        entry = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        entry.decorator_list = []
        class ReachedValidation(Exception):
            pass
        namespace = {
            "Path": Path,
            "validate_cfg": lambda _cfg: (_ for _ in ()).throw(ReachedValidation()),
        }
        exec(compile(ast.Module(body=[entry], type_ignores=[]), str(source), "exec"), namespace)
        main = namespace["main"]
        self.assertFalse(any(
            isinstance(node, ast.Attribute) and node.attr == "update_rollout_weights"
            for node in ast.walk(entry)
        ))
        with tempfile.TemporaryDirectory() as directory:
            resume = Path(directory) / "global_step_1"
            weights = resume / "actor/model_state_dict/full_weights.pt"
            weights.parent.mkdir(parents=True)
            weights.write_bytes(b"trusted model fixture")
            with self.assertRaisesRegex(ValueError, "both runner.resume_dir and runner.ckpt_path"):
                main(Config(runner=Config(only_eval=True, resume_dir=str(resume), ckpt_path=None)))
            with self.assertRaisesRegex(ValueError, "both runner.resume_dir and runner.ckpt_path"):
                main(Config(runner=Config(only_eval=True, resume_dir=None, ckpt_path=str(weights))))
            other = Path(directory) / "other.pt"
            other.write_bytes(b"another model")
            with self.assertRaisesRegex(ValueError, "belonging to runner.resume_dir"):
                main(Config(runner=Config(only_eval=True, resume_dir=str(resume), ckpt_path=str(other))))
            with self.assertRaises(ReachedValidation):
                main(Config(runner=Config(only_eval=True, resume_dir=str(resume), ckpt_path=str(weights))))


if __name__ == "__main__":
    unittest.main()
