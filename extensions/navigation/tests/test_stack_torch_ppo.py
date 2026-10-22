from __future__ import annotations

import ast
from dataclasses import replace
import importlib.util
import math
from pathlib import Path
import unittest

from integrations.rlinf_v03 import vln_stack_ppo
from integrations.rlinf_v03.vln_stack_ppo import compute_torch_ppo_objective
from navrl.stack import (
    DecisionBatch,
    DecisionRecord,
    EpisodeIdentity,
    ExecutionReceipt,
    GAEOutput,
    PolicyEvaluation,
    PolicySample,
)


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch


def _batch() -> DecisionBatch:
    identity = EpisodeIdentity("train-episode", "reset-1")
    records = []
    for index in range(2):
        terminated = index == 1
        action = (1 - index,)
        sample = PolicySample(
            identity=identity,
            decision_index=index,
            physical_step=index,
            action_payload=action,
            replay_context={"observation": index},
            behavior_log_prob_terms=(0.0,),
            value=0.0,
            policy_version=7,
            action_distribution="categorical_chunk",
        )
        execution = ExecutionReceipt(
            identity=identity,
            decision_index=index,
            physical_step=index,
            source_action_payload=action,
            executed_primitives=action,
            executed_mask=(True,),
            credited_log_prob_mask=(True,),
            primitive_rewards=(1.0,),
            duration=1,
            next_observation={"observation": index + 1},
            terminated=terminated,
            truncated=False,
            end_reason="terminated" if terminated else "complete",
            primitive_feedback=({},),
        )
        records.append(
            DecisionRecord(
                sample=sample,
                execution=execution,
                old_joint_log_prob=0.0,
                next_value=0.0,
                bootstrap_identity=None if terminated else identity,
            )
        )
    return DecisionBatch(tuple(records))


def _gae() -> GAEOutput:
    return GAEOutput(
        discounted_rewards=(1.0, 1.0),
        deltas=(1.0, -0.5),
        advantages=(1.0, -0.5),
        returns=(0.5, -0.25),
    )


class TorchBridgeStaticTest(unittest.TestCase):
    def test_module_keeps_torch_import_lazy(self):
        source_path = Path(vln_stack_ppo.__file__)
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        imports = [
            alias.name
            for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        ]
        self.assertNotIn("torch", imports)
        self.assertFalse(hasattr(vln_stack_ppo, "torch"))

    def test_public_bridge_does_not_own_optimizer_or_backward(self):
        source = Path(vln_stack_ppo.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        called_attributes = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertNotIn("backward", called_attributes)
        self.assertNotIn("step", called_attributes)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is unavailable")
class TorchBridgeTest(unittest.TestCase):
    def _evaluations(self, log_probs=None):
        batch = _batch()
        log_probs = log_probs or (0.0, 0.0)
        return tuple(
            PolicyEvaluation(
                identity=record.sample.identity,
                decision_index=record.sample.decision_index,
                joint_log_prob=torch.tensor(log_probs[index], requires_grad=True),
                value=torch.tensor(0.1 if index == 0 else -0.1, requires_grad=True),
                policy_version=record.sample.policy_version,
                entropy=torch.tensor(0.2 + 0.2 * index, requires_grad=True),
            )
            for index, record in enumerate(batch.records)
        )

    def test_objective_is_differentiable_and_ratio_guard_passes(self):
        evaluations = self._evaluations()
        result = compute_torch_ppo_objective(
            _batch(), evaluations, _gae(), entropy_coef=0.01
        )
        self.assertTrue(result.total_loss.requires_grad)
        self.assertEqual(2, result.diagnostics["active_count"])
        self.assertEqual(7, result.diagnostics["policy_version"])
        self.assertEqual(0.0, result.diagnostics["ratio_max_abs_deviation"])
        self.assertTrue(result.diagnostics["pre_update_guard_applied"])

        result.total_loss.backward()
        for evaluation in evaluations:
            self.assertIsNotNone(evaluation.joint_log_prob.grad)
            self.assertTrue(torch.isfinite(evaluation.joint_log_prob.grad).item())
            self.assertIsNotNone(evaluation.value.grad)
            self.assertTrue(torch.isfinite(evaluation.value.grad).item())
            self.assertIsNotNone(evaluation.entropy.grad)
            self.assertTrue(torch.isfinite(evaluation.entropy.grad).item())

    def test_ratio_guard_fails_before_update_and_can_be_skipped_later(self):
        evaluations = self._evaluations((math.log(1.2), 0.0))
        with self.assertRaisesRegex(ValueError, "pre-update PPO ratio"):
            compute_torch_ppo_objective(_batch(), evaluations, _gae())

        result = compute_torch_ppo_objective(
            _batch(), evaluations, _gae(), pre_update_tolerance=None
        )
        self.assertFalse(result.diagnostics["pre_update_guard_applied"])
        self.assertAlmostEqual(
            0.2, result.diagnostics["ratio_max_abs_deviation"], places=5
        )

    def test_rejects_identity_index_and_version_mismatches(self):
        baseline = self._evaluations()
        cases = (
            (
                "identity mismatch",
                replace(
                    baseline[0],
                    identity=EpisodeIdentity("train-episode", "another-reset"),
                ),
            ),
            ("index mismatch", replace(baseline[0], decision_index=1)),
            ("policy version mismatch", replace(baseline[0], policy_version=8)),
        )
        for message, replacement in cases:
            with self.subTest(message=message):
                rows = (replacement, baseline[1])
                with self.assertRaisesRegex(ValueError, message):
                    compute_torch_ppo_objective(_batch(), rows, _gae())

    def test_rejects_misaligned_gae_and_missing_weighted_entropy(self):
        evaluations = self._evaluations()
        short_gae = GAEOutput((1.0,), (1.0,), (1.0,), (1.0,))
        with self.assertRaisesRegex(ValueError, "GAE field"):
            compute_torch_ppo_objective(_batch(), evaluations, short_gae)

        no_entropy = tuple(replace(row, entropy=None) for row in evaluations)
        with self.assertRaisesRegex(ValueError, "entropy is required"):
            compute_torch_ppo_objective(
                _batch(), no_entropy, _gae(), entropy_coef=0.01
            )


if __name__ == "__main__":
    unittest.main()
