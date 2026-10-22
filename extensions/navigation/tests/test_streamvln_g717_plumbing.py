import ast
import hashlib
import json
import math
from pathlib import Path
import unittest

from integrations.rlinf_v03.streamvln_chunk_ppo import variable_duration_gae


class StreamVLNG717PlumbingTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]
    INTEGRATION = ROOT / "integrations" / "rlinf_v03"

    def test_manifest_is_frozen_train_only_and_claim_bounded(self):
        path = self.INTEGRATION / "streamvln_g717_experiment.json"
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(),
            "f8937a8845dad1488b714c4862210b1e4b0eb460dbc87726f5c6bf9f2f1f0838",
        )
        manifest = json.loads(path.read_text())
        smoke = manifest["plumbing_smoke"]
        self.assertEqual(smoke["split"], "train")
        self.assertEqual(
            [(row["target_mode"], row["target_mode_id"]) for row in smoke["scene_specs"]],
            [("navigate", 0), ("stop", 1), ("recover", 2)],
        )
        self.assertEqual(smoke["expected_executed_lengths"], [4, 1, 4])
        self.assertEqual(smoke["expected_end_boundaries"], ["truncated", "terminated", "truncated"])
        self.assertEqual(smoke["optimizer_update_limit"], 1)
        self.assertEqual(smoke["val_unseen_episode_count"], 0)
        self.assertTrue(
            smoke["mode_coverage_sampling"]["coverage_conditioned_not_unbiased_learning_data"]
        )
        self.assertFalse(manifest["navigation_evaluation"])
        self.assertFalse(manifest["checkpoint_selection"])
        self.assertFalse(manifest["g8_ready"])
        self.assertTrue(
            set(manifest["policy_observation_keys"]).isdisjoint(
                manifest["privileged_policy_blacklist"]
            )
        )

    def test_terminal_and_truncation_bootstrap_math_are_distinct(self):
        gamma = 0.99
        reward = 0.5
        value = 0.2
        bootstrap = 0.7
        trunc_advantage, trunc_return, _ = variable_duration_gae(
            [reward], [value], [bootstrap], [4], [False], gamma, 0.95
        )
        terminal_advantage, terminal_return, _ = variable_duration_gae(
            [reward], [value], [bootstrap], [1], [True], gamma, 0.95
        )
        self.assertTrue(math.isclose(
            trunc_advantage[0], reward + gamma**4 * bootstrap - value
        ))
        self.assertTrue(math.isclose(trunc_return[0], reward + gamma**4 * bootstrap))
        self.assertTrue(math.isclose(terminal_advantage[0], reward - value))
        self.assertTrue(math.isclose(terminal_return[0], reward))

    def test_runner_has_one_update_and_explicit_bootstrap_operation(self):
        runner_path = self.INTEGRATION / "run_streamvln_plumbing_g717.py"
        runner = runner_path.read_text()
        ast.parse(runner)
        self.assertIn('operation="encode_bootstrap_state"', runner)
        self.assertIn("feedback_from_dict", runner)
        self.assertIn("update_from_feedback", runner)
        self.assertIn("variable_duration_gae", runner)
        self.assertEqual(runner.count("manager.optimizer_step()"), 1)
        self.assertIn('"val_unseen_episode_count": 0', runner)
        self.assertNotIn('split="val_unseen"', runner)
        self.assertIn("coverage_conditioned_batch", runner)

        policy_path = self.INTEGRATION / "streamvln_topological_policy.py"
        policy = policy_path.read_text()
        ast.parse(policy)
        self.assertIn('if operation == "encode_bootstrap_state"', policy)
        self.assertIn("terminal bootstrap requires a frozen terminal observation", policy)
        self.assertIn("live state encoding rejects frozen terminal observations", policy)

        backbone_policy = (self.INTEGRATION / "streamvln_ppo_policy.py").read_text(encoding="utf-8")
        ast.parse(backbone_policy)
        self.assertIn('f"safe.directory={path}"', backbone_policy)

    def test_plan_preregisters_narrow_routing(self):
        plan_path = self.ROOT.parent / "STREAMVLN_G717_EXPERIMENT_PLAN.md"
        if not plan_path.is_file():
            self.skipTest("Historical experiment plan is not included in the public source export")
        plan = plan_path.read_text()
        self.assertIn("Preregistered on 2026-09-27 before the first G7.17", plan)
        self.assertIn("not an unbiased policy rollout", plan)
        self.assertIn("exactly one RLinf optimizer step", plan)
        self.assertIn("G8 and `val_unseen` remain closed", plan)


if __name__ == "__main__":
    unittest.main()
