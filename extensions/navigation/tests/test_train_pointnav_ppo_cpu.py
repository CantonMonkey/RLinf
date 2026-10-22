"""Essential CPU PPO math and RLinf checkpoint contracts (no simulator)."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations/rlinf_v03"))

from train_pointnav_ppo_cpu import (
    ROOT, check_rlinf_source, compute_gae, episode_geometry_signature,
    make_policy, save_checkpoint, train_episode_indices,
)


class GaeBoundaryTests(unittest.TestCase):
    def test_stop_never_bootstraps(self):
        advantages, returns = compute_gae(
            rewards=[1.0], values=[0.4], next_values=[99.0],
            terminated=[True], truncated=[False], gamma=0.9, gae_lambda=0.8,
        )
        self.assertAlmostEqual(advantages[0], 0.6)
        self.assertAlmostEqual(returns[0], 1.0)

    def test_time_limit_bootstraps_without_crossing_reset(self):
        advantages, returns = compute_gae(
            rewards=[0.0, 10.0], values=[0.5, 0.0], next_values=[0.8, 99.0],
            terminated=[False, True], truncated=[True, False],
            gamma=0.9, gae_lambda=1.0,
        )
        self.assertAlmostEqual(advantages[0], 0.22)
        self.assertAlmostEqual(returns[0], 0.72)
        self.assertAlmostEqual(advantages[1], 10.0)

    def test_nonterminal_rollout_edge_uses_next_value(self):
        advantages, returns = compute_gae(
            rewards=[0.0, 0.0], values=[1.0, 2.0], next_values=[2.0, 3.0],
            terminated=[False, False], truncated=[False, False],
            gamma=1.0, gae_lambda=1.0,
        )
        self.assertEqual(advantages, [2.0, 1.0])
        self.assertEqual(returns, [3.0, 3.0])

    def test_impossible_double_boundary_rejected(self):
        with self.assertRaises(ValueError):
            compute_gae([1.0], [0.0], [0.0], [True], [True], 0.99, 0.95)

    def test_geometry_signature_catches_renamed_episode(self):
        episode = {"episode_id": "train-1", "scene_id": "scene.glb",
                   "start_position": [1, 2, 3], "start_rotation": [0, 0, 0, 1],
                   "goals": [{"position": [4, 5, 6]}]}
        renamed = dict(episode, episode_id="val-9")
        self.assertEqual(episode_geometry_signature(episode),
                         episode_geometry_signature(renamed))


class RlinfCheckpointTests(unittest.TestCase):
    def test_official_train_selection_rejects_val_overlap(self):
        asset_root = ROOT / "data"
        if not asset_root.is_dir():
            if os.environ.get("RLINF_TEST_REQUIRED") == "1":
                self.fail("official Habitat test assets missing")
            self.skipTest("official Habitat test assets unavailable locally")
        indices, ids, train_sha, val_sha = train_episode_indices(asset_root, "50,51")
        self.assertEqual(indices, [50, 51])
        self.assertEqual(len(ids), 2)
        self.assertEqual(len(train_sha), 64)
        self.assertEqual(len(val_sha), 64)
        with self.assertRaisesRegex(ValueError, "overlap official val"):
            train_episode_indices(asset_root, "0,1")

    def test_matches_rlinf_gae_without_truncation(self):
        try:
            import torch
            from rlinf.algorithms.advantages import compute_gae_advantages_and_returns
        except ImportError:
            if os.environ.get("RLINF_TEST_REQUIRED") == "1":
                raise
            self.skipTest("RLinf/Torch private CPU environment unavailable")
        rewards = [0.1, -0.2, 0.3]
        values = [0.5, 0.6, 0.4]
        next_values = [0.6, 0.4, 0.1]
        for final_stop in (False, True):
            ours_adv, ours_ret = compute_gae(
                rewards, values, next_values,
                [False, False, final_stop], [False] * 3, 0.99, 0.95,
            )
            reference_adv, reference_ret = compute_gae_advantages_and_returns(
                rewards=torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1),
                values=torch.tensor(values + [next_values[-1]], dtype=torch.float32).unsqueeze(-1),
                dones=torch.tensor([False, False, False, final_stop]).unsqueeze(-1),
                gamma=0.99, gae_lambda=0.95, normalize_advantages=False,
            )
            torch.testing.assert_close(torch.tensor(ours_adv), reference_adv[:, 0], atol=1e-6, rtol=0)
            torch.testing.assert_close(torch.tensor(ours_ret), reference_ret[:, 0], atol=1e-6, rtol=0)

    def test_rlinf_loss_update_and_strict_checkpoint(self):
        try:
            import torch
            from rlinf.algorithms.losses import compute_ppo_actor_critic_loss
        except ImportError:
            if os.environ.get("RLINF_TEST_REQUIRED") == "1":
                raise
            self.skipTest("RLinf/Torch private CPU environment unavailable")
        torch.set_num_threads(1)
        torch.manual_seed(43)
        self.assertEqual(len(check_rlinf_source(ROOT / "repos/RLinf-v0.3")), 64)
        policy = make_policy()
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
        states = torch.tensor([[1.0, 0.2], [2.0, -0.3], [3.0, 1.2]], dtype=torch.float32)
        actions, sampled = policy.predict_action_batch(env_obs={"states": states}, mode="train")
        self.assertEqual(tuple(actions.shape), (3, 1, 1))
        output = policy(forward_inputs={"states": states, "action": actions.reshape(3, 1)},
                        compute_logprobs=True, compute_entropy=True, compute_values=True)
        advantages = torch.tensor([[1.0], [-0.5], [0.3]], dtype=torch.float32)
        old_values = sampled["prev_values"].detach().clone()
        old_logprobs = sampled["prev_logprobs"].detach().clone()
        returns = old_values + advantages
        loss, _ = compute_ppo_actor_critic_loss(
            logprobs=output["logprobs"], old_logprobs=old_logprobs,
            values=output["values"], prev_values=old_values,
            advantages=advantages, returns=returns,
            clip_ratio_low=0.2, clip_ratio_high=0.2,
            value_clip=1.0, huber_delta=10.0,
            loss_mask=torch.ones_like(advantages, dtype=torch.bool),
        )
        self.assertTrue(bool(torch.isfinite(loss)))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with tempfile.TemporaryDirectory() as tmp:
            record = save_checkpoint(torch, policy, optimizer, Path(tmp), 1, 3, 43)
            self.assertTrue(record["strict_load"])
            weights = torch.load(record["full_weights"], map_location="cpu", weights_only=True)
            restored = make_policy()
            restored.load_state_dict(weights, strict=True)
            incomplete = dict(weights)
            incomplete.pop("actor_mean.weight")
            with self.assertRaises(RuntimeError):
                restored.load_state_dict(incomplete, strict=True)


if __name__ == "__main__":
    unittest.main()
