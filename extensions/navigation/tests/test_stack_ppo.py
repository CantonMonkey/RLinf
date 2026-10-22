from __future__ import annotations

import math
import unittest

from navrl.stack import (
    assert_pre_update_ratio,
    compute_ppo_objective,
    compute_ratio_metrics,
)


class StackPPOTest(unittest.TestCase):
    def test_clipped_policy_value_and_entropy_objective(self):
        old_logprobs = [0.0, 0.0]
        new_logprobs = [math.log(1.5), math.log(0.5)]
        result = compute_ppo_objective(
            old_joint_log_probs=old_logprobs,
            new_joint_log_probs=new_logprobs,
            old_values=[0.0, 1.0],
            new_values=[1.0, 0.0],
            returns=[1.0, 0.0],
            advantages=[2.0, -2.0],
            entropies=[0.4, 0.6],
            clip_ratio_low=0.2,
            clip_ratio_high=0.2,
            value_clip=0.2,
            value_coef=0.5,
            entropy_coef=0.1,
        )

        # Positive advantage clips 1.5 down to 1.2; negative advantage uses
        # the worse clipped ratio 0.8 instead of the raw 0.5.
        self.assertAlmostEqual(-0.4, result.policy_loss)
        # The clipped value predictions are farther from the targets: 0.8^2.
        self.assertAlmostEqual(0.32, result.value_loss)
        self.assertAlmostEqual(0.5, result.mean_entropy)
        self.assertAlmostEqual(-0.5, result.entropy_loss)
        self.assertAlmostEqual(-0.29, result.total_loss)
        self.assertEqual((1.5, 0.5), result.ratio_metrics.ratios)
        self.assertAlmostEqual(1.0, result.ratio_metrics.mean)
        self.assertAlmostEqual(0.5, result.ratio_metrics.max_abs_deviation)
        self.assertAlmostEqual(1.0, result.ratio_metrics.clip_fraction)

    def test_mask_excludes_padding_from_losses_and_ratio_metrics(self):
        result = compute_ppo_objective(
            old_joint_log_probs=[0.0, 0.0],
            new_joint_log_probs=[0.0, 100.0],
            old_values=[0.0, 0.0],
            new_values=[0.0, 100.0],
            returns=[0.0, -100.0],
            advantages=[1.0, -100.0],
            entropies=[0.25, -100.0],
            mask=[True, False],
        )
        self.assertEqual(1, result.active_count)
        self.assertEqual((1.0,), result.ratio_metrics.ratios)
        self.assertAlmostEqual(-1.0, result.policy_loss)
        self.assertAlmostEqual(0.0, result.value_loss)
        self.assertAlmostEqual(0.25, result.mean_entropy)

    def test_pre_update_ratio_check_passes_exact_replay_and_rejects_drift(self):
        exact = assert_pre_update_ratio([-2.0, -0.25], [-2.0, -0.25])
        self.assertEqual((1.0, 1.0), exact.ratios)
        self.assertEqual(0.0, exact.max_abs_deviation)

        with self.assertRaisesRegex(ValueError, "pre-update PPO ratio deviates"):
            assert_pre_update_ratio(
                [0.0], [math.log(1.031)], tolerance=0.03
            )
        accepted = assert_pre_update_ratio(
            [0.0], [math.log(1.029)], tolerance=0.03
        )
        self.assertAlmostEqual(0.029, accepted.max_abs_deviation)

    def test_ratio_metrics_and_objective_reject_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "same non-zero length"):
            compute_ratio_metrics([0.0], [0.0, 0.0])
        with self.assertRaisesRegex(ValueError, "at least one"):
            compute_ratio_metrics([0.0], [0.0], mask=[False])
        with self.assertRaisesRegex(ValueError, "clip_ratio_low"):
            compute_ppo_objective(
                old_joint_log_probs=[0.0],
                new_joint_log_probs=[0.0],
                old_values=[0.0],
                new_values=[0.0],
                returns=[0.0],
                advantages=[0.0],
                clip_ratio_low=1.1,
            )
        with self.assertRaisesRegex(ValueError, "overflows"):
            compute_ratio_metrics([0.0], [1000.0])


if __name__ == "__main__":
    unittest.main()
