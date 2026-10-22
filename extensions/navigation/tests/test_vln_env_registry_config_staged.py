"""Static checks for the isolated RLinf v0.3 R2R environment/config staging."""

from __future__ import annotations

import runpy
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml


STAGE = Path(__file__).resolve().parents[1] / "integrations" / "rlinf_v03"


def read_yaml(name: str) -> dict:
    with (STAGE / name).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


class StagedVLNEnvConfigTests(unittest.TestCase):
    def test_registry_keeps_pointnav_and_adds_vln_lazy_branch(self) -> None:
        self.assertTrue((STAGE / "vln_init.py").is_file())
        registry = runpy.run_path(str(STAGE / "envs_init_vln.py"))
        fake_vln = type("VLNEnv", (), {})
        fake_pointnav = type("PointNavEnv", (), {})
        modules = {
            "rlinf": types.ModuleType("rlinf"),
            "rlinf.envs": types.ModuleType("rlinf.envs"),
            "rlinf.envs.vln": types.ModuleType("rlinf.envs.vln"),
            "rlinf.envs.vln.vln_env": types.ModuleType("rlinf.envs.vln.vln_env"),
            "rlinf.envs.pointnav": types.ModuleType("rlinf.envs.pointnav"),
            "rlinf.envs.pointnav.pointnav_env": types.ModuleType(
                "rlinf.envs.pointnav.pointnav_env"
            ),
        }
        modules["rlinf.envs.vln.vln_env"].VLNEnv = fake_vln
        modules["rlinf.envs.pointnav.pointnav_env"].PointNavEnv = fake_pointnav
        with patch.dict(sys.modules, modules):
            self.assertIs(registry["get_env_cls"]("vln"), fake_vln)
            self.assertIs(registry["get_env_cls"]("pointnav"), fake_pointnav)

    def test_streamvln_executor_mode_selects_successor_explicitly(self) -> None:
        registry = runpy.run_path(str(STAGE / "envs_init_vln.py"))
        fake_baseline = type("StreamVLNEnv", (), {})
        fake_successor = type("StreamVLNP8SuccessorEnv", (), {})
        modules = {
            "rlinf": types.ModuleType("rlinf"),
            "rlinf.envs": types.ModuleType("rlinf.envs"),
            "rlinf.envs.streamvln": types.ModuleType("rlinf.envs.streamvln"),
            "rlinf.envs.streamvln.streamvln_env": types.ModuleType(
                "rlinf.envs.streamvln.streamvln_env"
            ),
            "rlinf.envs.streamvln.streamvln_p8_successor_env": types.ModuleType(
                "rlinf.envs.streamvln.streamvln_p8_successor_env"
            ),
        }
        modules["rlinf.envs.streamvln.streamvln_env"].StreamVLNEnv = fake_baseline
        modules[
            "rlinf.envs.streamvln.streamvln_p8_successor_env"
        ].StreamVLNP8SuccessorEnv = fake_successor
        with patch.dict(sys.modules, modules):
            self.assertIs(
                registry["get_env_cls"]("streamvln", {"executor_mode": "baseline"}),
                fake_baseline,
            )
            self.assertIs(
                registry["get_env_cls"](
                    "streamvln", {"executor_mode": "p8_successor"}
                ),
                fake_successor,
            )
            with self.assertRaises(ValueError):
                registry["get_env_cls"](
                    "streamvln", {"executor_mode": "unregistered"}
                )

    def test_env_and_policy_observation_contract(self) -> None:
        env = read_yaml("vln_r2r_env.yaml")
        model = read_yaml("vln_policy.yaml")
        self.assertEqual(env["env_type"], "vln")
        self.assertEqual(env["asset_root"], "/cache/navigation-example")
        self.assertIsNone(env["dataset_split"])
        self.assertIsNone(env["episode_index"])
        self.assertEqual(env["rgb_size"], model["image_size"])
        self.assertEqual(env["max_instruction_tokens"], model["max_instruction_tokens"])
        self.assertEqual(model["vocab_size"], 2504)
        self.assertEqual(model["model_type"], "vln_policy")
        self.assertEqual(model["discrete_n_actions"], 4)
        self.assertEqual(env["success_distance_m"], 3.0)
        self.assertFalse(env["auto_reset"])
        self.assertFalse(env["video_cfg"]["save_video"])

    def test_train_and_eval_split_and_step_budget(self) -> None:
        smoke = read_yaml("vln_r2r_ppo_smoke.yaml")
        heldout = read_yaml("vln_r2r_ppo_eval.yaml")
        self.assertIn("env/vln_r2r_env@env.train", smoke["defaults"])
        self.assertIn("env/vln_r2r_env@env.eval", smoke["defaults"])
        self.assertIn("model/vln_policy@actor.model", smoke["defaults"])
        self.assertEqual(smoke["env"]["train"]["dataset_split"], "train")
        self.assertEqual(smoke["env"]["train"]["episode_index"], 4852)
        self.assertEqual(smoke["env"]["eval"]["dataset_split"], "val_unseen")
        self.assertEqual(smoke["env"]["eval"]["episode_index"], 630)
        self.assertEqual(smoke["actor"]["model"]["model_type"], "vln_policy")
        self.assertEqual(smoke["actor"]["model"]["vocab_size"], 2504)
        self.assertEqual(smoke["actor"]["model"]["max_instruction_tokens"], 200)
        self.assertEqual(smoke["actor"]["micro_batch_size"], 8)
        self.assertEqual(smoke["actor"]["global_batch_size"], 8)
        self.assertEqual(heldout["defaults"], ["vln_r2r_ppo_smoke", "_self_"])
        self.assertTrue(heldout["runner"]["only_eval"])
        self.assertEqual(heldout["env"]["eval"]["rollout_epoch"], 1)
        self.assertEqual(heldout["env"]["eval"]["max_episode_steps"], 500)
        self.assertEqual(heldout["env"]["eval"]["max_steps_per_rollout_epoch"], 500)
        self.assertTrue(heldout["env"]["eval"]["video_cfg"]["save_video"])


if __name__ == "__main__":
    unittest.main()
