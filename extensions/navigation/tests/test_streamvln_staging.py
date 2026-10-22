"""Static wiring checks for the pinned RLinf v0.3 StreamVLN staging patch."""

from __future__ import annotations

from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "integrations/rlinf_v03"


class StreamVLNStagingTests(unittest.TestCase):
    def text(self, name):
        return (STAGE / name).read_text(encoding="utf-8")

    def test_model_environment_and_worker_are_registered_once(self):
        config = self.text("config_vln.py")
        models = self.text("models_init_vln.py")
        envs = self.text("envs_init_vln.py")
        worker = self.text("huggingface_worker_vln.py")
        self.assertIn('SupportedModel.register(\n    "streamvln_categorical"', config)
        self.assertIn("SupportedModel.STREAMVLN_CATEGORICAL,", config)
        self.assertIn("_build_streamvln_categorical", models)
        self.assertIn('STREAMVLN = "streamvln"', envs)
        self.assertIn("return StreamVLNEnv", envs)
        self.assertEqual(worker.count("SupportedModel.STREAMVLN_CATEGORICAL,"), 2)

    def test_configs_are_fixed_chunk_and_do_not_bind_windows_paths(self):
        policy = yaml.safe_load(self.text("streamvln_policy.yaml"))
        env = yaml.safe_load(self.text("streamvln_env.yaml"))
        frozen = yaml.safe_load(self.text("streamvln_frozen_eval.yaml"))
        self.assertEqual(policy["model_type"], "streamvln_categorical")
        self.assertEqual(policy["num_action_chunks"], 4)
        self.assertEqual(env["env_type"], "streamvln")
        self.assertEqual(env["max_episode_steps"] % 4, 0)
        self.assertTrue(frozen["runner"]["only_eval"])
        combined = "\n".join(
            self.text(name)
            for name in (
                "streamvln_policy.yaml",
                "streamvln_env.yaml",
                "streamvln_frozen_eval.yaml",
                "streamvln_policy.py",
                "streamvln_env.py",
            )
        )
        for prefix in ("C:\\\\", "D:\\\\", "C:/", "D:/"):
            self.assertNotIn(prefix, combined)

    def test_bounded_entry_uses_real_rlinf_predict_without_ppo(self):
        source = self.text("eval_streamvln.py")
        self.assertIn("inspect.unwrap(MultiStepRolloutWorker.predict)", source)
        self.assertIn('"ppo_started": False', source)
        self.assertIn("shuffled_restore_max_error", source)
        self.assertNotIn("optimizer.step", source)


if __name__ == "__main__":
    unittest.main()
