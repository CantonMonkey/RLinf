import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import unittest


class StreamVLNG723ProtocolTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]
    INTEGRATION = ROOT / "integrations" / "rlinf_v03"

    def test_manifest_freezes_exhausted_split_sampling_and_clustered_rule(self):
        path = self.INTEGRATION / "streamvln_g723_experiment.json"
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(),
            "59af15397be0d6642474459ad74e21da4fb7896a28c76ea4e9f6da7326cc8297",
        )
        manifest = json.loads(path.read_text())
        selection = manifest["selection"]
        protocol = manifest["protocol"]
        specs = [row for arm in protocol["boundary_arms"].values() for row in arm]
        self.assertEqual(selection["seeds"], [2026092811, 2026092812, 2026092813])
        self.assertEqual(len(specs), 24)
        self.assertEqual(len({row["episode_id"] for row in specs}), 24)
        self.assertEqual({row["scene"] for row in specs}, set(selection["eligible_new_scenes"]))
        self.assertEqual(len(selection["eligible_new_scenes"]), 5)
        self.assertFalse({row["scene"] for row in specs} & set(selection["excluded_scenes"]))
        for arm in protocol["boundary_arms"].values():
            for scene in {row["scene"] for row in arm}:
                self.assertLessEqual(sum(row["scene"] == scene for row in arm), 2)
        self.assertTrue(protocol["state_zero_and_final_must_exactly_match_g721"])
        self.assertTrue(protocol["no_new_policy_state_or_behavior_selection"])
        self.assertEqual(protocol["compiler_contract"]["turn_step_boundaries_degrees"], [7.5, 22.5, 37.5, 52.5])
        rule = manifest["decision_rule"]
        self.assertEqual(rule["minimum_flip_count"], 2)
        self.assertEqual(rule["minimum_flip_scene_count"], 2)
        self.assertEqual(rule["minimum_harmful_flip_count"], 2)
        self.assertEqual(rule["minimum_harmful_flip_scene_count"], 2)
        self.assertFalse(rule["posthoc_selection_or_rerun"])
        self.assertFalse(manifest["new_policy_training_or_selection"])
        self.assertEqual(protocol["val_unseen_episode_count"], 0)
        self.assertFalse(manifest["g8_ready"])

    def test_runner_records_input_hashes_and_stops_causal_attribution_after_divergence(self):
        path = self.INTEGRATION / "run_streamvln_executor_boundary_g723.py"
        source = path.read_text()
        ast.parse(source)
        self.assertIn('row["state_features_sha256"]', source)
        self.assertIn('row["topology_features_sha256"]', source)
        self.assertIn('row["policy_input_sha256"]', source)
        self.assertIn("prior_executed_sequences_equal", source)
        self.assertIn("if executed_flip", source)
        self.assertIn('seed_phase="g723_boundary"', source)
        self.assertIn("baseline_hashes != anchor", source)
        self.assertIn("final_hashes != anchor", source)
        self.assertNotIn("sample_target_mode", source)
        self.assertNotIn("save_head_checkpoint", source)
        self.assertNotIn('split="val_unseen"', source)

    def test_independent_recompute_detects_one_boundary_flip_then_marks_later_row_descriptive(self):
        path = self.INTEGRATION / "recompute_streamvln_g723.py"
        spec = importlib.util.spec_from_file_location("g723_recompute", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        compiler = {
            "feedback_equality_tolerance": 1e-9,
            "navigate_small_waypoint_shift_l2_max_m": 0.05,
            "rounding": "python_round_abs_bearing_over_turn_radians",
            "turn_degrees": 15.0,
            "turn_step_boundaries_degrees": [7.5, 22.5, 37.5, 52.5],
        }

        def feedback(actions, reward, collision):
            return {
                "proposed_actions": actions,
                "executed_actions": actions,
                "environment_reward_sum": reward,
                "collision_count": collision,
                "blocked_forward_count": collision,
                "realized_path_m": 1.0 - collision,
                "commanded_forward_m": 1.0,
            }

        def row(angle, actions, reward, collision, decision):
            return {
                "decision_index": decision,
                "policy_mode": 0,
                "waypoint": [1.0, math.tan(math.radians(angle))],
                "state_features_sha256": "a" * 64,
                "topology_features_sha256": "b" * 64,
                "policy_input_sha256": "c" * 64,
                "executor_feedback": feedback(actions, reward, collision),
            }

        before = {
            "episode_id": "1", "scene": "scene", "initial_observation_sha256": "obs",
            "sampling_seed_schedule": [1, 2],
            "rows": [row(7.4, [1, 1, 1, 1], 0.0, 0, 0), row(7.4, [1, 1, 1, 1], 0.0, 0, 1)],
        }
        after = {
            "episode_id": "1", "scene": "scene", "initial_observation_sha256": "obs",
            "sampling_seed_schedule": [1, 2],
            "rows": [row(7.6, [2, 1, 1, 1], -0.2, 1, 0), row(7.4, [1, 1, 1, 1], 0.0, 0, 1)],
        }
        result = module.analyze_boundary([before], [after], compiler)
        self.assertEqual(result["summary"]["causal_boundary_flip_count"], 1)
        self.assertEqual(result["summary"]["harmful_boundary_flip_count"], 1)
        self.assertTrue(result["decision_rows"][0]["causal_prefix"])
        self.assertTrue(result["decision_rows"][1]["descriptive_only"])

    def test_independent_metric_comparison_accepts_runtime_distribution_fields(self):
        path = self.INTEGRATION / "recompute_streamvln_g723.py"
        spec = importlib.util.spec_from_file_location("g723_recompute_metrics", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        runtime = {"episode_count": 8, "decision_count": 24, "reward_distribution": {"count": 24}}
        independent = {"episode_count": 8, "decision_count": 24}
        self.assertTrue(module.numeric_mapping_close(runtime, independent))
        self.assertFalse(module.numeric_mapping_close(runtime, {"episode_count": 7}))


if __name__ == "__main__":
    unittest.main()
