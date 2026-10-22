"""Check R2R val_unseen episode accounting without importing Ray or Habitat."""

import ast
import gzip
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


class Config(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


class FakeTensor:
    def __init__(self, values):
        self.values = list(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def reshape(self, *_shape):
        return self

    def tolist(self):
        return list(self.values)


def load_collector():
    path = Path(__file__).resolve().parents[1] / "integrations/rlinf_v03/train_embodied_agent.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    method = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "collect_eval_episode_metrics"
    )
    scope = {"Path": Path, "gzip": gzip, "json": json, "math": math}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    return scope["collect_eval_episode_metrics"]


class VLNEvalMetricsTests(unittest.TestCase):
    def setUp(self):
        self.collect = load_collector()

    @staticmethod
    def fixture(root):
        path = root / "datasets/r2r/val_unseen/val_unseen.json.gz"
        path.parent.mkdir(parents=True)
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            json.dump({"episodes": [
                {"episode_id": "631", "scene_id": "mp3d/unseen/room.glb"}
            ]}, stream)
        cfg = Config(env=Config(eval=Config(
            env_type="vln", dataset_split="val_unseen", total_num_envs=1,
            episode_index=0, rollout_epoch=1, asset_root=str(root),
        )))
        shard = {
            "episode_index": FakeTensor([0]),
            "success": FakeTensor([0.0]),
            "spl": FakeTensor([0.0]),
            "path_length": FakeTensor([0.5]),
            "collision_count": FakeTensor([0.0]),
            "distance_to_goal": FakeTensor([4.2]),
        }
        return cfg, shard

    def test_val_unseen_identity_and_metrics_without_pointnav_geodesic(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, shard = self.fixture(Path(directory))
            record, aggregated = self.collect(cfg, [shard])
            self.assertEqual(record["env_type"], "vln")
            self.assertEqual(record["dataset_split"], "val_unseen")
            self.assertEqual(record["observed_episode_indices"], [0])
            self.assertEqual(record["episodes"][0]["episode_id"], "631")
            self.assertEqual(record["episodes"][0]["scene_id"], "mp3d/unseen/room.glb")
            self.assertEqual(record["episodes"][0]["distance_to_goal"], 4.2)
            self.assertNotIn("episode_index", aggregated[0])
            self.assertIs(aggregated[0]["success"], shard["success"])

    def test_rejects_wrong_split_episode_or_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, shard = self.fixture(Path(directory))
            cfg.env.eval.dataset_split = "train"
            with self.assertRaisesRegex(ValueError, "val_unseen"):
                self.collect(cfg, [shard])
            cfg.env.eval.dataset_split = "val_unseen"
            with self.assertRaisesRegex(ValueError, "Unexpected"):
                self.collect(cfg, [dict(shard, episode_index=FakeTensor([1]))])
            with self.assertRaisesRegex(ValueError, "collision_count"):
                self.collect(cfg, [{key: value for key, value in shard.items() if key != "collision_count"}])
            with self.assertRaisesRegex(ValueError, "Nonfinite"):
                self.collect(cfg, [dict(shard, spl=FakeTensor([float("nan")]))])
            cfg.env.eval.rollout_epoch = 2
            with self.assertRaisesRegex(ValueError, "rollout_epoch"):
                self.collect(cfg, [shard])


if __name__ == "__main__":
    unittest.main()
