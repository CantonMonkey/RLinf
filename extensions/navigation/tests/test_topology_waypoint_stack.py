"""S2 contract tests for the replaceable topology + metric-waypoint plugin.

The test task is deliberately train-only and synthetic.  These checks establish
the stack/action/replay contract; they do not make a navigation-quality claim.
"""

from __future__ import annotations

from dataclasses import replace
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest

from navrl.stack import (
    DecisionBatch,
    DecisionCollector,
    DecisionRecord,
    EpisodeIdentity,
    EpisodeMetrics,
    PolicySample,
    PrimitiveStep,
    TaskAdapter,
    TaskReset,
    compute_duration_aware_gae,
    load_checkpoint_verified,
    save_checkpoint_atomic,
)
from integrations.rlinf_v03.streamvln_p8_successor_action import (
    FORWARD_ACTION,
    LEFT_ACTION,
    RIGHT_ACTION,
    STOP_ACTION,
    compile_metric_waypoint,
)


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch

    from integrations.rlinf_v03.topology_waypoint_stack import (
        EpisodeTopologyState,
        TopologyCandidate,
        TopologyWaypointActionExecutor,
        TopologyWaypointCheckpointAdapter,
        TopologyWaypointPolicy,
        TopologyWaypointPolicyAdapter,
    )
    from integrations.rlinf_v03.vln_stack_ppo import compute_torch_ppo_objective


class TwoDecisionMetricTask(TaskAdapter):
    """Primitive task whose second high-level choice can terminate with STOP."""

    def __init__(self):
        self.identity = None
        self.physical_step = 0
        self.reset_count = 0
        self.actions = []

    @staticmethod
    def observation(step):
        rgb = torch.zeros(1, 64, 64, 3, dtype=torch.uint8)
        rgb[:, 6 + min(step, 20) : 26 + min(step, 20), :, step % 3] = 80 + step
        instruction_ids = torch.zeros(1, 200, dtype=torch.int64)
        instruction_mask = torch.zeros(1, 200, dtype=torch.bool)
        instruction_ids[0, :5] = torch.tensor((4, 12, 8 + step, 19, 2))
        instruction_mask[0, :5] = True
        return {
            "rgb": rgb,
            "instruction_ids": instruction_ids,
            "instruction_mask": instruction_mask,
        }

    def reset(self, episode=None, *, seed=None):
        self.reset_count += 1
        episode_id = str(episode or "train-topology-waypoint")
        self.identity = EpisodeIdentity(
            episode_id,
            f"{episode_id}:seed-{seed}:reset-{self.reset_count}",
        )
        self.physical_step = 0
        self.actions = []
        return TaskReset(self.identity, self.observation(0))

    def step(self, primitive_action):
        primitive_action = int(primitive_action)
        self.physical_step += 1
        self.actions.append(primitive_action)
        return PrimitiveStep(
            identity=self.identity,
            physical_step=self.physical_step,
            observation=self.observation(self.physical_step),
            reward=0.4 + 0.1 * self.physical_step,
            terminated=primitive_action == 0,
            info={
                "action": primitive_action,
                "collision": False,
                "realized_path_m": 0.25 if primitive_action == 1 else 0.0,
                "blocked_forward": False,
            },
        )

    def get_metrics(self):
        return EpisodeMetrics(
            identity=self.identity,
            path_length_m=0.25 * self.actions.count(1),
            collisions=0,
            stop_called=0 in self.actions,
            timed_out=False,
            action_counts={str(action): self.actions.count(action) for action in set(self.actions)},
        )

    def close(self):
        return None


class EarlyStopMetricTask(TwoDecisionMetricTask):
    """End during a macro to verify whole-decision hybrid credit."""

    def step(self, primitive_action):
        row = super().step(primitive_action)
        return replace(row, terminated=True)


class RootForwardSpy:
    def __init__(self, policy):
        self.policy = policy
        self.operations = []

    def __call__(self, *, operation, **kwargs):
        self.operations.append(operation)
        return self.policy(operation=operation, **kwargs)


class MetricCompilerContractTest(unittest.TestCase):
    def test_distance_and_turn_direction_change_macro(self):
        self.assertEqual((FORWARD_ACTION,), compile_metric_waypoint((0.25, 0.0)))
        self.assertEqual(
            (FORWARD_ACTION, FORWARD_ACTION, FORWARD_ACTION),
            compile_metric_waypoint((0.75, 0.0)),
        )
        self.assertEqual(LEFT_ACTION, compile_metric_waypoint((0.75, 0.25))[0])
        self.assertEqual(RIGHT_ACTION, compile_metric_waypoint((0.75, -0.25))[0])


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is unavailable")
class TopologyWaypointStackTest(unittest.TestCase):
    """The smallest checks needed to admit a second policy family to the stack."""

    def make_policy(self, **overrides):
        options = {
            "vocab_size": 2504,
            "hidden_dim": 32,
            "topology_max_nodes": 8,
            "topology_model_dim": 16,
            "max_candidates": 5,
            "residual_limit_m": 0.2,
        }
        options.update(overrides)
        return TopologyWaypointPolicy(**options)

    def make_stack(self, *, mode="train", **state_overrides):
        state_options = {
            "max_decisions": 8,
            "topology_max_nodes": 8,
            "frontier_distance_m": 0.75,
            "frontier_lateral_m": 0.5,
            "residual_limit_m": 0.2,
        }
        state_options.update(state_overrides)
        state = EpisodeTopologyState(**state_options)
        policy = self.make_policy()
        adapter = TopologyWaypointPolicyAdapter(
            policy,
            state,
            mode=mode,
            root_forward=policy,
        )
        executor = TopologyWaypointActionExecutor(state)
        return policy, state, adapter, executor

    @staticmethod
    def _configure_heads(policy, *, candidate_slot):
        """Prefer STOP or a frontier while leaving a real categorical density."""

        with torch.no_grad():
            for parameter in policy.candidate_scorer.parameters():
                parameter.zero_()
            policy.candidate_slot_bias.fill_(-4.0)
            policy.candidate_slot_bias[candidate_slot] = 4.0
            for parameter in policy.residual_mean_head.parameters():
                parameter.zero_()

    def _candidate_slots(self, sample):
        context = sample.replay_context
        required = {"candidates", "candidate_digest", "generation", "forward_inputs"}
        missing = required - set(context)
        if missing:
            raise AssertionError(f"replay context misses frozen candidate fields: {sorted(missing)}")
        candidates = tuple(
            TopologyCandidate.from_canonical(row) for row in context["candidates"]
        )
        forward = context["forward_inputs"]
        valid = tuple(bool(value) for value in forward["candidate_mask"].reshape(-1))
        continuous = tuple(
            bool(value) for value in forward["continuous_mask"].reshape(-1)
        )
        anchors = forward["candidate_anchors"].reshape(-1, 2)
        context_digest = context["candidate_digest"]
        self.assertEqual(64, len(context_digest))
        for candidate, anchor in zip(candidates, anchors):
            self.assertTrue(
                torch.allclose(
                    anchor,
                    torch.tensor((candidate.forward_m, candidate.lateral_m)),
                )
            )
        kinds = tuple(candidate.kind for candidate in candidates)
        nav = next(
            index
            for index, (is_valid, is_continuous) in enumerate(zip(valid, continuous))
            if is_valid and is_continuous
        )
        stop = next(
            index
            for index, (kind, is_valid) in enumerate(zip(kinds, valid))
            if is_valid and kind == "stop"
        )
        return nav, stop

    def _force_kind(self, policy, state, adapter, task, kind):
        first = task.reset("candidate-probe", seed=3)
        adapter.reset_state(first.identity)
        probe = adapter.sample(first.observation, first.identity, 0, 0)
        nav_slot, stop_slot = self._candidate_slots(probe)
        self._configure_heads(
            policy,
            candidate_slot=nav_slot if kind == "navigate" else stop_slot,
        )
        reset = task.reset("forced-candidate", seed=5)
        adapter.reset_state(reset.identity)
        sample = adapter.sample(reset.observation, reset.identity, 0, 0)
        self.assertEqual(
            nav_slot if kind == "navigate" else stop_slot,
            int(sample.action_payload[2]),
        )
        return reset, sample

    def test_candidate_value_object_rejects_nonfinite_metric_geometry(self):
        candidate = TopologyCandidate("frontier-a", "frontier", 0.75, -0.5)
        self.assertEqual("frontier-a", candidate.candidate_id)
        self.assertTrue(candidate.valid)
        self.assertTrue(candidate.continuous)
        with self.assertRaises((TypeError, ValueError)):
            TopologyCandidate("bad", "frontier", math.inf, 0.0)

    def test_executor_compiles_stop_as_one_primitive(self):
        policy, state, adapter, executor = self.make_stack(mode="eval")
        task = TwoDecisionMetricTask()
        _, sample = self._force_kind(policy, state, adapter, task, "stop")
        receipt = executor.execute(task, sample)
        self.assertEqual((STOP_ACTION,), receipt.active_primitives)
        self.assertEqual(1, receipt.duration)
        self.assertTrue(receipt.terminated)
        self.assertEqual((True,), receipt.credited_log_prob_mask)
        self.assertEqual(1, len(sample.behavior_log_prob_terms))
        old_log_prob = adapter.behavior_joint_log_prob(sample, receipt)
        batch = DecisionBatch(
            (
                DecisionRecord(
                    sample=sample,
                    execution=receipt,
                    old_joint_log_prob=old_log_prob,
                    next_value=0.0,
                    bootstrap_identity=None,
                ),
            )
        )
        evaluated = adapter.evaluate(batch)[0]
        ratio = torch.exp(evaluated.joint_log_prob - old_log_prob)
        self.assertTrue(torch.allclose(ratio, torch.ones_like(ratio), atol=1e-6))

    def test_partial_metric_macro_credits_both_hybrid_terms(self):
        policy, state, adapter, executor = self.make_stack(mode="eval")
        task = EarlyStopMetricTask()
        _, sample = self._force_kind(policy, state, adapter, task, "navigate")
        self.assertEqual(2, len(sample.behavior_log_prob_terms))
        self.assertGreater(len(executor._expand(sample.action_payload)), 1)
        receipt = executor.execute(task, sample)
        self.assertEqual(1, receipt.duration)
        self.assertEqual((True, True), receipt.credited_log_prob_mask)
        adapter.behavior_joint_log_prob(sample, receipt)

    def test_sample_execute_replay_ratio_and_duration_aware_gae(self):
        torch.manual_seed(101)
        policy, state, adapter, executor = self.make_stack()
        task = TwoDecisionMetricTask()

        # Read the frozen candidate schema once, then make a continuous choice
        # overwhelmingly likely without changing its finite log probability.
        reset = task.reset("head-probe", seed=7)
        adapter.reset_state(reset.identity)
        probe = adapter.sample(reset.observation, reset.identity, 0, 0)
        nav_slot, _ = self._candidate_slots(probe)
        self._configure_heads(policy, candidate_slot=nav_slot)
        root_spy = RootForwardSpy(policy)
        adapter.root_forward = root_spy

        batch = DecisionCollector().collect_episode(
            task,
            adapter,
            executor,
            max_decisions=2,
            episode="train-two-decision",
            seed=11,
        )
        self.assertEqual(2, len(batch.records))
        durations = tuple(row.execution.duration for row in batch.records)
        self.assertTrue(all(1 <= value <= 4 for value in durations))
        self.assertEqual(
            (0, durations[0]),
            tuple(row.sample.physical_step for row in batch.records),
        )
        for row in batch.records:
            self.assertEqual(
                (True,) * len(row.sample.behavior_log_prob_terms),
                row.execution.credited_log_prob_mask,
            )
            self.assertEqual(row.sample.action_payload, row.execution.source_action_payload)

        evaluations = adapter.evaluate(batch)
        self.assertIn("sample", root_spy.operations)
        self.assertIn("replay", root_spy.operations)
        self.assertIn("bootstrap", root_spy.operations)
        for row, evaluated in zip(batch.records, evaluations):
            ratio = torch.exp(evaluated.joint_log_prob - row.old_joint_log_prob)
            self.assertTrue(torch.allclose(ratio, torch.ones_like(ratio), atol=1e-6))

        gamma, gae_lambda = 0.9, 0.8
        gae = compute_duration_aware_gae(batch, gamma, gae_lambda)
        last = batch.records[1]
        expected_delta_1 = (
            gae.discounted_rewards[1]
            + (0.0 if last.execution.terminated else gamma ** durations[1] * float(last.next_value))
            - float(last.sample.value)
        )
        first = batch.records[0]
        expected_delta_0 = (
            gae.discounted_rewards[0]
            + gamma ** durations[0] * float(first.next_value)
            - float(first.sample.value)
        )
        expected_advantage_0 = expected_delta_0 + (
            (gamma * gae_lambda) ** durations[0]
        ) * gae.advantages[1]
        self.assertAlmostEqual(expected_delta_1, gae.deltas[1], places=6)
        self.assertAlmostEqual(expected_advantage_0, gae.advantages[0], places=6)

        objective = compute_torch_ppo_objective(
            batch,
            evaluations,
            gae,
            entropy_coef=0.01,
            pre_update_tolerance=1e-6,
        )
        self.assertLessEqual(objective.diagnostics["ratio_max_abs_deviation"], 1e-6)
        self.assertTrue(bool(torch.isfinite(objective.total_loss.detach())))

        declared = adapter.trainable_parameter_names()
        named = dict(policy.named_parameters())
        self.assertEqual(
            set(declared),
            {name for name, value in named.items() if value.requires_grad},
        )
        frozen_before = {
            name: value.detach().clone()
            for name, value in named.items()
            if not value.requires_grad
        }
        trainable_before = {
            name: named[name].detach().clone()
            for name in declared
        }
        optimizer = torch.optim.Adam((named[name] for name in declared), lr=1e-3)
        optimizer.zero_grad(set_to_none=True)
        objective.total_loss.backward()
        finite_nonzero = {
            name
            for name in declared
            if named[name].grad is not None
            and bool(torch.isfinite(named[name].grad).all())
            and float(named[name].grad.detach().abs().sum()) > 0.0
        }
        self.assertTrue(finite_nonzero)
        self.assertTrue(
            any(
                "candidate_scorer" in name or name == "candidate_slot_bias"
                for name in finite_nonzero
            )
        )
        self.assertTrue(any("residual_mean_head" in name for name in finite_nonzero))
        self.assertIn("residual_log_std", finite_nonzero)
        self.assertTrue(any("value_head" in name for name in finite_nonzero))
        optimizer.step()
        adapter.mark_updated()
        changed = {
            name
            for name in declared
            if not torch.equal(trainable_before[name], named[name].detach())
        }
        self.assertTrue(finite_nonzero <= changed)
        for name, expected in frozen_before.items():
            self.assertTrue(torch.equal(expected, named[name].detach()), name)

    def test_checkpoint_restore_rejects_incompatible_target_before_mutation(self):
        torch.manual_seed(211)
        policy, state, adapter, _ = self.make_stack()
        adapter.mark_updated()
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "checkpoint"
            saved = save_checkpoint_atomic(
                TopologyWaypointCheckpointAdapter(adapter), checkpoint
            )
            self.assertEqual(1, saved.metadata["policy_version"])

            restored_policy, _, restored_adapter, _ = self.make_stack()
            loaded = load_checkpoint_verified(
                TopologyWaypointCheckpointAdapter(restored_adapter), checkpoint
            )
            self.assertEqual(saved.files_sha256, loaded.files_sha256)
            self.assertEqual(1, restored_adapter.policy_version)
            for name, value in policy.state_dict().items():
                self.assertTrue(torch.equal(value, restored_policy.state_dict()[name]), name)

            incompatible_policy = self.make_policy(hidden_dim=48)
            incompatible_state = EpisodeTopologyState(
                max_decisions=8,
                topology_max_nodes=8,
                frontier_distance_m=0.75,
                frontier_lateral_m=0.5,
                residual_limit_m=0.2,
            )
            incompatible_adapter = TopologyWaypointPolicyAdapter(
                incompatible_policy,
                incompatible_state,
                root_forward=incompatible_policy,
            )
            before = {
                name: value.detach().clone()
                for name, value in incompatible_policy.state_dict().items()
            }
            version_before = incompatible_adapter.policy_version
            generation_before = incompatible_adapter._generation
            with self.assertRaisesRegex(RuntimeError, "compatibility"):
                load_checkpoint_verified(
                    TopologyWaypointCheckpointAdapter(incompatible_adapter), checkpoint
                )
            for name, value in incompatible_policy.state_dict().items():
                self.assertTrue(torch.equal(before[name], value), name)
            self.assertEqual(version_before, incompatible_adapter.policy_version)
            self.assertEqual(generation_before, incompatible_adapter._generation)

            active_policy, _, active_adapter, active_executor = self.make_stack()
            stale_batch = DecisionCollector().collect_episode(
                TwoDecisionMetricTask(),
                active_adapter,
                active_executor,
                max_decisions=1,
                episode="checkpoint-generation",
                seed=19,
            )
            load_checkpoint_verified(
                TopologyWaypointCheckpointAdapter(active_adapter), checkpoint
            )
            with self.assertRaisesRegex((RuntimeError, ValueError), "generation|stale"):
                active_adapter.evaluate(stale_batch)

    def test_reset_isolates_history_and_replay_rejects_payload_tampering(self):
        torch.manual_seed(313)
        policy, state, adapter, executor = self.make_stack(mode="eval")
        task = TwoDecisionMetricTask()
        reset, sample = self._force_kind(policy, state, adapter, task, "navigate")
        execution = executor.execute(task, sample)
        next_value = adapter.bootstrap(
            execution.next_observation,
            reset.identity,
            execution.duration,
        )
        record = DecisionRecord(
            sample=sample,
            execution=execution,
            old_joint_log_prob=adapter.behavior_joint_log_prob(sample, execution),
            next_value=next_value.value,
            bootstrap_identity=reset.identity,
        )
        batch = DecisionBatch((record,))
        adapter.evaluate(batch)

        tampered_payload = list(sample.action_payload)
        tampered_payload[-2] = float(tampered_payload[-2]) + 0.125
        tampered_payload = tuple(tampered_payload)
        tampered_sample = replace(sample, action_payload=tampered_payload)
        tampered_execution = replace(execution, source_action_payload=tampered_payload)
        tampered_record = replace(
            record,
            sample=tampered_sample,
            execution=tampered_execution,
        )
        with self.assertRaisesRegex(
            (RuntimeError, ValueError),
            "payload|tamper|candidate|digest|waypoint|residual",
        ):
            adapter.evaluate(DecisionBatch((tampered_record,)))

        tampered_context = dict(sample.replay_context)
        tampered_context["candidate_digest"] = "0" * 64
        digest_sample = replace(sample, replay_context=tampered_context)
        digest_execution = replace(execution, source_action_payload=digest_sample.action_payload)
        digest_record = replace(record, sample=digest_sample, execution=digest_execution)
        with self.assertRaisesRegex((RuntimeError, ValueError), "tamper|digest"):
            adapter.evaluate(DecisionBatch((digest_record,)))

        candidate_context = dict(sample.replay_context)
        candidate_rows = list(candidate_context["candidates"])
        changed_row = list(candidate_rows[0])
        changed_row[2] = float(changed_row[2]) + 0.01
        candidate_rows[0] = tuple(changed_row)
        candidate_context["candidates"] = tuple(candidate_rows)
        candidate_sample = replace(sample, replay_context=candidate_context)
        candidate_execution = replace(
            execution, source_action_payload=candidate_sample.action_payload
        )
        candidate_record = replace(
            record, sample=candidate_sample, execution=candidate_execution
        )
        with self.assertRaisesRegex((RuntimeError, ValueError), "tamper|digest"):
            adapter.evaluate(DecisionBatch((candidate_record,)))

        tensor_context = dict(sample.replay_context)
        forward_inputs = dict(tensor_context["forward_inputs"])
        forward_inputs["candidate_anchors"] = forward_inputs[
            "candidate_anchors"
        ].clone()
        forward_inputs["candidate_anchors"][0, 0, 0] += 0.01
        tensor_context["forward_inputs"] = forward_inputs
        tensor_sample = replace(sample, replay_context=tensor_context)
        tensor_execution = replace(
            execution, source_action_payload=tensor_sample.action_payload
        )
        tensor_record = replace(
            record, sample=tensor_sample, execution=tensor_execution
        )
        with self.assertRaisesRegex((RuntimeError, ValueError), "tamper|snapshot"):
            adapter.evaluate(DecisionBatch((tensor_record,)))

        generation = sample.replay_context["generation"]
        self.assertGreater(state.topology.decision_count, 0)
        new_reset = task.reset("forced-candidate", seed=5)
        adapter.reset_state(new_reset.identity)
        self.assertEqual(0, state.topology.decision_count)
        self.assertEqual(0, state.topology.primitive_count)
        self.assertNotEqual(generation, adapter.sample(
            new_reset.observation, new_reset.identity, 0, 0
        ).replay_context["generation"])
        with self.assertRaisesRegex((RuntimeError, ValueError), "stale|reset|generation"):
            adapter.evaluate(batch)


if __name__ == "__main__":
    unittest.main()
