"""Real Torch S1 integration test for the lightweight VLN plugin."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

from integrations.rlinf_v03.vln_stack_adapters import (
    VLNPolicyCheckpointAdapter,
    VLNPolicyStackAdapter,
)
from integrations.rlinf_v03.vln_stack_ppo import compute_torch_ppo_objective
from navrl.stack import (
    DecisionCollector,
    EpisodeIdentity,
    EpisodeMetrics,
    PrefixActionExecutor,
    PrimitiveStep,
    TaskAdapter,
    TaskReset,
    compute_duration_aware_gae,
    load_checkpoint_verified,
    save_checkpoint_atomic,
)


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch

    from tests.test_vln_policy import load_policy_module


class TwoDecisionTorchTask(TaskAdapter):
    """Small train-only task with real policy-shaped tensor observations."""

    def __init__(self):
        self.identity = None
        self.physical_step = 0
        self.reset_count = 0
        self.actions = []
        self.closed = False

    @staticmethod
    def observation(step):
        rgb = torch.zeros(1, 64, 64, 3, dtype=torch.uint8)
        rgb[:, 8 + step : 32 + step, :, step % 3] = 80 + 30 * step
        instruction_ids = torch.zeros(1, 200, dtype=torch.int64)
        instruction_mask = torch.zeros(1, 200, dtype=torch.bool)
        instruction_ids[0, :4] = torch.tensor((4, 8 + step, 12, 16))
        instruction_mask[0, :4] = True
        return {
            "rgb": rgb,
            "instruction_ids": instruction_ids,
            "instruction_mask": instruction_mask,
        }

    def reset(self, episode=None, *, seed=None):
        self.reset_count += 1
        episode_id = str(episode or "train-episode")
        self.identity = EpisodeIdentity(
            episode_id,
            f"{episode_id}:seed-{seed}:reset-{self.reset_count}",
        )
        self.physical_step = 0
        self.actions = []
        return TaskReset(self.identity, self.observation(0))

    def step(self, primitive_action):
        self.physical_step += 1
        self.actions.append(int(primitive_action))
        return PrimitiveStep(
            identity=self.identity,
            physical_step=self.physical_step,
            observation=self.observation(self.physical_step),
            reward=1.0 - 0.2 * self.physical_step,
            info={"action": int(primitive_action)},
        )

    def get_metrics(self):
        return EpisodeMetrics(
            identity=self.identity,
            path_length_m=0.25 * self.physical_step,
            collisions=0,
            stop_called=False,
            timed_out=False,
            action_counts={str(action): self.actions.count(action) for action in self.actions},
        )

    def close(self):
        self.closed = True


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is unavailable")
class LightweightTorchIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy_module = load_policy_module()

    def make_policy(self):
        return self.policy_module.VLNPolicy(vocab_size=2504)

    def test_episode_update_checkpoint_and_fresh_reset(self):
        torch.manual_seed(101)
        policy = self.make_policy()
        adapter = VLNPolicyStackAdapter(policy, root_forward=policy)
        task = TwoDecisionTorchTask()
        executor = PrefixActionExecutor(
            max_width=1,
            expand_action=lambda payload: payload,
            credit_mode="whole_decision",
        )
        batch = DecisionCollector().collect_episode(
            task,
            adapter,
            executor,
            max_decisions=2,
            episode="train-episode",
            seed=17,
        )
        self.assertEqual(2, len(batch.records))
        self.assertEqual((0, 1), tuple(row.sample.decision_index for row in batch.records))
        self.assertEqual((0, 1), tuple(row.sample.physical_step for row in batch.records))

        evaluations = adapter.evaluate(batch)
        gae = compute_duration_aware_gae(batch, gamma=0.97, gae_lambda=0.91)
        objective = compute_torch_ppo_objective(
            batch,
            evaluations,
            gae,
            entropy_coef=0.01,
            pre_update_tolerance=1e-6,
        )
        self.assertLessEqual(
            objective.diagnostics["ratio_max_abs_deviation"], 1e-6
        )
        self.assertTrue(objective.diagnostics["pre_update_guard_applied"])
        self.assertTrue(bool(torch.isfinite(objective.total_loss.detach())))

        declared = adapter.trainable_parameter_names()
        named = dict(policy.named_parameters())
        self.assertEqual(set(declared), {name for name, row in named.items() if row.requires_grad})
        before = {name: named[name].detach().clone() for name in declared}
        optimizer = torch.optim.Adam((named[name] for name in declared), lr=1e-3)
        optimizer.zero_grad(set_to_none=True)
        objective.total_loss.backward()

        gradient_sum = 0.0
        for name in declared:
            gradient = named[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertTrue(bool(torch.isfinite(gradient).all()), name)
            gradient_sum += float(gradient.detach().abs().sum())
        self.assertGreater(gradient_sum, 0.0)
        self.assertGreater(float(named["actor_logits.weight"].grad.abs().sum()), 0.0)
        self.assertGreater(float(named["value_head.weight"].grad.abs().sum()), 0.0)

        optimizer.step()
        adapter.mark_updated()
        changed = {
            name for name in declared if not torch.equal(before[name], named[name].detach())
        }
        self.assertTrue(changed)
        self.assertTrue(any(name.startswith("actor_logits.") for name in changed))
        self.assertTrue(any(name.startswith("value_head.") for name in changed))
        self.assertEqual(1, adapter.policy_version)

        comparison_observation = task.observation(1)
        with torch.no_grad():
            expected_logits, expected_value = policy.logits_and_values(
                policy.preprocess_env_obs(comparison_observation)
            )
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "checkpoint"
            saved = save_checkpoint_atomic(
                VLNPolicyCheckpointAdapter(adapter), checkpoint
            )
            self.assertEqual(1, saved.metadata["policy_version"])

            restored_policy = self.make_policy()
            restored_adapter = VLNPolicyStackAdapter(
                restored_policy, root_forward=restored_policy
            )
            loaded = load_checkpoint_verified(
                VLNPolicyCheckpointAdapter(restored_adapter), checkpoint
            )
            self.assertEqual(saved.files_sha256, loaded.files_sha256)
            self.assertEqual(1, restored_adapter.policy_version)
            with torch.no_grad():
                actual_logits, actual_value = restored_policy.logits_and_values(
                    restored_policy.preprocess_env_obs(comparison_observation)
                )
            self.assertTrue(torch.equal(expected_logits, actual_logits))
            self.assertTrue(torch.equal(expected_value, actual_value))

            active_policy = self.make_policy()
            active_adapter = VLNPolicyStackAdapter(
                active_policy, policy_version=1, root_forward=active_policy
            )
            stale_batch = DecisionCollector().collect_episode(
                TwoDecisionTorchTask(),
                active_adapter,
                executor,
                max_decisions=1,
                episode="checkpoint-generation",
                seed=23,
            )
            load_checkpoint_verified(
                VLNPolicyCheckpointAdapter(active_adapter), checkpoint
            )
            with self.assertRaisesRegex((RuntimeError, ValueError), "generation|adapter"):
                active_adapter.evaluate(stale_batch)

            same_shape_policy = self.policy_module.VLNPolicy(
                vocab_size=2504, image_size=80
            )
            same_shape_adapter = VLNPolicyStackAdapter(
                same_shape_policy, root_forward=same_shape_policy
            )
            same_shape_before = {
                name: value.detach().clone()
                for name, value in same_shape_policy.state_dict().items()
            }
            with self.assertRaisesRegex(RuntimeError, "compatibility"):
                load_checkpoint_verified(
                    VLNPolicyCheckpointAdapter(same_shape_adapter), checkpoint
                )
            for name, value in same_shape_policy.state_dict().items():
                self.assertTrue(torch.equal(same_shape_before[name], value), name)

            first_reset = task.reset("train-episode", seed=29)
            torch.manual_seed(303)
            restored_adapter.reset_state(first_reset.identity)
            first_sample = restored_adapter.sample(
                first_reset.observation, first_reset.identity, 0, 0
            )
            second_reset = task.reset("train-episode", seed=29)
            self.assertNotEqual(
                first_reset.identity.reset_token, second_reset.identity.reset_token
            )
            torch.manual_seed(303)
            restored_adapter.reset_state(second_reset.identity)
            second_sample = restored_adapter.sample(
                second_reset.observation, second_reset.identity, 0, 0
            )
            self.assertEqual(first_sample.action_payload, second_sample.action_payload)
            self.assertTrue(
                torch.equal(
                    first_sample.behavior_log_prob_terms[0],
                    second_sample.behavior_log_prob_terms[0],
                )
            )
            self.assertTrue(torch.equal(first_sample.value, second_sample.value))
            self.assertNotEqual(
                first_sample.replay_context["generation"],
                second_sample.replay_context["generation"],
            )


if __name__ == "__main__":
    unittest.main()
