from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from navrl.stack import (
    BootstrapValue,
    CheckpointAdapter,
    DecisionBatch,
    DecisionCollector,
    EpisodeMetrics,
    EpisodeIdentity,
    PolicyAdapter,
    PolicyEvaluation,
    PolicySample,
    PrefixActionExecutor,
    PrimitiveStep,
    TaskAdapter,
    TaskReset,
    compute_duration_aware_gae,
    duration_aware_gae,
    load_checkpoint_verified,
    save_checkpoint_atomic,
)


class TinyTask(TaskAdapter):
    def __init__(self):
        self.identity = None
        self.step_index = 0
        self.reset_index = 0
        self.closed = False
        self.actions = []

    def reset(self, episode=None, *, seed=None):
        episode_id = str(episode or "train-episode")
        self.reset_index += 1
        self.identity = EpisodeIdentity(
            episode_id, f"{episode_id}:{seed}:reset:{self.reset_index}"
        )
        self.step_index = 0
        self.actions = []
        return TaskReset(self.identity, {"episode": episode_id, "step": 0})

    def step(self, primitive_action):
        self.step_index += 1
        self.actions.append(primitive_action)
        terminated = primitive_action == 0
        return PrimitiveStep(
            identity=self.identity,
            physical_step=self.step_index,
            observation={"episode": self.identity.episode_id, "step": self.step_index},
            reward=float(primitive_action),
            terminated=terminated,
            info={"action": primitive_action},
        )

    def get_metrics(self):
        counts = {str(action): self.actions.count(action) for action in set(self.actions)}
        return EpisodeMetrics(
            identity=self.identity,
            path_length_m=float(self.step_index),
            collisions=0,
            stop_called=0 in self.actions,
            timed_out=False,
            action_counts=counts,
            success=None,
            spl=None,
            navigation_error_m=None,
            extras={"steps": self.step_index},
        )

    def close(self):
        self.closed = True


class TinyPolicy(PolicyAdapter):
    def __init__(self):
        self.identity = None
        self._version = 0
        self.bootstrap_calls = 0

    def reset_state(self, identity):
        self.identity = identity

    def sample(self, observation, identity, decision_index, physical_step):
        if identity != self.identity:
            raise RuntimeError("stale policy identity")
        payload = (1, 2) if decision_index == 0 else (0,)
        return PolicySample(
            identity=identity,
            decision_index=decision_index,
            physical_step=physical_step,
            action_payload=payload,
            replay_context={"decision": decision_index, "observation": observation},
            behavior_log_prob_terms=tuple(
                -0.1 * (decision_index + offset + 1)
                for offset in range(len(payload))
            ),
            value=0.25 * (decision_index + 1),
            policy_version=self._version,
            action_distribution="categorical_chunk",
        )

    def behavior_joint_log_prob(self, sample, execution):
        return sum(
            term
            for term, credited in zip(
                sample.behavior_log_prob_terms,
                execution.credited_log_prob_mask,
            )
            if credited
        )

    def evaluate(self, batch):
        return tuple(
            PolicyEvaluation(
                identity=row.sample.identity,
                decision_index=row.sample.decision_index,
                joint_log_prob=row.old_joint_log_prob,
                value=row.sample.value,
                entropy=0.5,
                policy_version=self._version,
            )
            for row in batch.records
        )

    def bootstrap(self, observation, identity, physical_step):
        self.bootstrap_calls += 1
        return BootstrapValue(identity, physical_step, self.policy_version, 0.75)

    def trainable_parameter_names(self):
        return ("actor", "value")

    @property
    def policy_version(self):
        return self._version


class TinyCheckpoint(CheckpointAdapter):
    def __init__(self, value="weights-v1", version=3, architecture="tiny-v1"):
        self.value = value
        self.version = version
        self.architecture = architecture
        self.load_calls = 0

    def checkpoint_compatibility(self):
        return {
            "architecture": self.architecture,
            "codec": "tiny-json-v1",
            "trainable_names": ("actor", "value"),
        }

    def checkpoint_metadata(self):
        return {"policy_version": self.version, "trainable_names": ["actor", "value"]}

    @staticmethod
    def _read_payload(directory: Path):
        payload = json.loads(
            (directory / "nested" / "weights.txt").read_text(encoding="utf-8")
        )
        if set(payload) != {"value", "version"}:
            raise ValueError("invalid tiny checkpoint fields")
        value = payload["value"]
        version = payload["version"]
        if not isinstance(value, str) or not isinstance(version, int):
            raise ValueError("invalid tiny checkpoint state")
        return value, version

    def inspect_checkpoint_metadata(self, directory: Path):
        _, version = self._read_payload(directory)
        return {"policy_version": version, "trainable_names": ["actor", "value"]}

    def save_checkpoint(self, staging_directory: Path):
        path = staging_directory / "nested" / "weights.txt"
        path.parent.mkdir()
        payload = json.dumps(
            {"value": self.value, "version": self.version},
            sort_keys=True,
            separators=(",", ":"),
        )
        path.write_text(payload, encoding="utf-8")
        return {"nested/weights.txt": hashlib.sha256(payload.encode()).hexdigest()}

    def load_checkpoint(self, directory: Path):
        self.load_calls += 1
        value, version = self._read_payload(directory)
        self.value = value
        self.version = version


class StackCoreTest(unittest.TestCase):
    def test_collects_two_decisions_with_exact_execution_identity(self):
        task = TinyTask()
        policy = TinyPolicy()
        executor = PrefixActionExecutor(
            max_width=2,
            expand_action=lambda payload: payload,
            pad_action=None,
            credit_mode="executed_prefix",
        )
        batch = DecisionCollector().collect_episode(
            task, policy, executor, max_decisions=2, episode="train-1", seed=7
        )
        self.assertEqual("train-1", batch.identity.episode_id)
        self.assertEqual(2, len(batch.records))
        first, second = batch.records
        self.assertEqual((1, 2), first.execution.active_primitives)
        self.assertEqual((True, True), first.execution.executed_mask)
        self.assertEqual(2, first.execution.duration)
        self.assertAlmostEqual(-0.3, first.old_joint_log_prob)
        self.assertEqual(second.sample.value, first.next_value)
        self.assertEqual((0,), second.execution.active_primitives)
        self.assertEqual((True, False), second.execution.executed_mask)
        self.assertTrue(second.execution.terminated)
        self.assertEqual(0.0, second.next_value)
        self.assertIsNone(second.bootstrap_identity)
        self.assertEqual(0, policy.bootstrap_calls)
        metrics = task.get_metrics()
        self.assertEqual(first.sample.identity, metrics.identity)
        self.assertTrue(metrics.stop_called)
        self.assertEqual({"0": 1, "1": 1, "2": 1}, dict(metrics.action_counts))
        evaluations = policy.evaluate(batch)
        self.assertEqual(
            [row.old_joint_log_prob for row in batch.records],
            [row.joint_log_prob for row in evaluations],
        )
        gae = compute_duration_aware_gae(batch, gamma=0.9, gae_lambda=0.8)
        self.assertEqual((2.8, 0.0), gae.discounted_rewards)
        self.assertEqual(2, len(gae.advantages))

    def test_behavior_joint_log_prob_credits_only_executed_prefix(self):
        task = TinyTask()
        policy = TinyPolicy()
        executor = PrefixActionExecutor(
            max_width=2,
            expand_action=lambda payload: payload,
            pad_action=None,
            stop_after=lambda row: "safety_interrupt" if row.physical_step == 1 else None,
            credit_mode="executed_prefix",
        )
        batch = DecisionCollector().collect_episode(
            task, policy, executor, max_decisions=1, episode="train-prefix", seed=3
        )
        row = batch.records[0]
        self.assertEqual(1, row.execution.duration)
        self.assertEqual("safety_interrupt", row.execution.end_reason)
        self.assertAlmostEqual(-0.1, row.old_joint_log_prob)

        gae = compute_duration_aware_gae(batch, gamma=0.9, gae_lambda=0.8)
        self.assertEqual((1.0,), gae.discounted_rewards)

    def test_macro_action_can_expand_to_multiple_primitives(self):
        class MacroPolicy(TinyPolicy):
            def sample(self, observation, identity, decision_index, physical_step):
                self._require_bound(identity)
                return PolicySample(
                    identity=identity,
                    decision_index=decision_index,
                    physical_step=physical_step,
                    action_payload=(0.5, -0.25),
                    replay_context={"observation": observation},
                    behavior_log_prob_terms=(-0.4,),
                    value=0.1,
                    policy_version=0,
                    action_distribution="gaussian_waypoint",
                )

            def _require_bound(self, identity):
                if identity != self.identity:
                    raise RuntimeError("stale policy identity")

        task = TinyTask()
        policy = MacroPolicy()
        executor = PrefixActionExecutor(
            max_width=2,
            expand_action=lambda waypoint: (1, 2),
        )
        batch = DecisionCollector().collect_episode(
            task, policy, executor, max_decisions=1, episode="train-waypoint"
        )
        row = batch.records[0]
        self.assertEqual((0.5, -0.25), row.execution.source_action_payload)
        self.assertEqual((1, 2), row.execution.active_primitives)
        self.assertEqual((True,), row.execution.credited_log_prob_mask)
        self.assertAlmostEqual(-0.4, row.old_joint_log_prob)

    def test_bad_prefix_credit_fails_before_any_task_side_effect(self):
        task = TinyTask()
        reset = task.reset("train-no-side-effect")
        sample = PolicySample(
            identity=reset.identity,
            decision_index=0,
            physical_step=0,
            action_payload=(1, 2),
            replay_context={"observation": reset.observation},
            behavior_log_prob_terms=(-0.1,),
            value=0.0,
            policy_version=0,
            action_distribution="invalid_prefix_fixture",
        )
        executor = PrefixActionExecutor(
            max_width=2,
            expand_action=lambda payload: payload,
            credit_mode="executed_prefix",
        )
        with self.assertRaisesRegex(ValueError, "one log-prob term"):
            executor.execute(task, sample)
        self.assertEqual(0, task.step_index)
        self.assertEqual([], task.actions)

    def test_batch_rejects_mixed_behavior_policy_versions_at_construction(self):
        batch = DecisionCollector().collect_episode(
            TinyTask(),
            TinyPolicy(),
            PrefixActionExecutor(
                max_width=2,
                expand_action=lambda payload: payload,
                credit_mode="executed_prefix",
            ),
            max_decisions=2,
        )
        changed_sample = replace(batch.records[1].sample, policy_version=1)
        changed_record = replace(batch.records[1], sample=changed_sample)
        with self.assertRaisesRegex(ValueError, "policy versions"):
            DecisionBatch((batch.records[0], changed_record))

    def test_truncation_bootstraps_but_cuts_trace(self):
        result = duration_aware_gae(
            chunk_rewards=[1.0, 2.0],
            values=[0.5, 0.25],
            next_values=[0.25, 4.0],
            durations=[1, 2],
            terminated=[False, False],
            truncated=[False, True],
            gamma=0.9,
            gae_lambda=0.8,
        )
        expected_last = 2.0 + 0.9**2 * 4.0 - 0.25
        self.assertAlmostEqual(expected_last, result.deltas[1])
        self.assertAlmostEqual(expected_last, result.advantages[1])
        expected_first = 1.0 + 0.9 * 0.25 - 0.5 + 0.9 * 0.8 * expected_last
        self.assertAlmostEqual(expected_first, result.advantages[0])

    def test_reset_identity_rejects_stale_policy_state(self):
        task = TinyTask()
        policy = TinyPolicy()
        first = task.reset("train-a", seed=1)
        policy.reset_state(first.identity)
        stale = first.identity
        second = task.reset("train-a", seed=1)
        policy.reset_state(second.identity)
        with self.assertRaisesRegex(RuntimeError, "stale"):
            policy.sample(second.observation, stale, 0, 0)

    def test_collector_rejects_reused_reset_token(self):
        class BrokenResetTask(TinyTask):
            def reset(self, episode=None, *, seed=None):
                reset = super().reset(episode, seed=seed)
                self.identity = EpisodeIdentity(reset.identity.episode_id, "reused-token")
                return TaskReset(self.identity, reset.observation)

        collector = DecisionCollector()
        task = BrokenResetTask()
        policy = TinyPolicy()
        executor = PrefixActionExecutor(
            max_width=2,
            expand_action=lambda payload: payload,
            credit_mode="executed_prefix",
        )
        collector.collect_episode(task, policy, executor, max_decisions=1)
        with self.assertRaisesRegex(RuntimeError, "reused"):
            collector.collect_episode(task, policy, executor, max_decisions=1)

    def test_collector_rejects_nonconsecutive_reset_token_reuse(self):
        class ScriptedResetTask(TinyTask):
            def __init__(self):
                super().__init__()
                self.tokens = iter(("token-a", "token-b", "token-a"))

            def reset(self, episode=None, *, seed=None):
                reset = super().reset(episode, seed=seed)
                self.identity = EpisodeIdentity(reset.identity.episode_id, next(self.tokens))
                return TaskReset(self.identity, reset.observation)

        collector = DecisionCollector()
        task = ScriptedResetTask()
        executor = PrefixActionExecutor(
            max_width=2,
            expand_action=lambda payload: payload,
            credit_mode="executed_prefix",
        )
        collector.collect_episode(task, TinyPolicy(), executor, max_decisions=1)
        collector.collect_episode(task, TinyPolicy(), executor, max_decisions=1)
        with self.assertRaisesRegex(RuntimeError, "reused"):
            collector.collect_episode(task, TinyPolicy(), executor, max_decisions=1)

    def test_collector_rejects_sample_and_bootstrap_provenance_drift(self):
        class WrongVersionPolicy(TinyPolicy):
            def sample(self, observation, identity, decision_index, physical_step):
                return replace(
                    super().sample(observation, identity, decision_index, physical_step),
                    policy_version=self.policy_version + 1,
                )

        executor = PrefixActionExecutor(
            max_width=2,
            expand_action=lambda payload: payload,
            credit_mode="executed_prefix",
        )
        with self.assertRaisesRegex(RuntimeError, "behavior version"):
            DecisionCollector().collect_episode(
                TinyTask(), WrongVersionPolicy(), executor, max_decisions=1
            )

        class WrongBootstrapPolicy(TinyPolicy):
            def bootstrap(self, observation, identity, physical_step):
                return BootstrapValue(identity, physical_step + 1, self.policy_version, 0.75)

        with self.assertRaisesRegex(RuntimeError, "bootstrap provenance"):
            DecisionCollector().collect_episode(
                TinyTask(), WrongBootstrapPolicy(), executor, max_decisions=1
            )

    def test_atomic_checkpoint_roundtrip_and_hash_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = TinyCheckpoint()
            receipt = save_checkpoint_atomic(source, root / "checkpoint")
            self.assertEqual(
                "weights-v1",
                json.loads(
                    (receipt.directory / "nested" / "weights.txt").read_text()
                )["value"],
            )
            self.assertEqual(["nested/weights.txt"], list(receipt.files_sha256))
            self.assertEqual("tiny-v1", receipt.compatibility["architecture"])
            restored = TinyCheckpoint(value="empty", version=-1)
            loaded = load_checkpoint_verified(restored, receipt.directory)
            self.assertEqual("weights-v1", restored.value)
            self.assertEqual(3, restored.version)
            self.assertEqual(1, restored.load_calls)
            self.assertEqual(receipt.files_sha256, loaded.files_sha256)

            incompatible = TinyCheckpoint(
                value="compatibility-sentinel",
                version=41,
                architecture="tiny-v2",
            )
            with self.assertRaisesRegex(RuntimeError, "compatibility"):
                load_checkpoint_verified(incompatible, receipt.directory)
            self.assertEqual("compatibility-sentinel", incompatible.value)
            self.assertEqual(41, incompatible.version)
            self.assertEqual(0, incompatible.load_calls)

            manifest_path = receipt.directory / "stack_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["metadata"]["policy_version"] = 99
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True), encoding="utf-8"
            )
            metadata_guarded = TinyCheckpoint(
                value="metadata-sentinel", version=43
            )
            with self.assertRaisesRegex(RuntimeError, "payload metadata"):
                load_checkpoint_verified(metadata_guarded, receipt.directory)
            self.assertEqual("metadata-sentinel", metadata_guarded.value)
            self.assertEqual(43, metadata_guarded.version)
            self.assertEqual(0, metadata_guarded.load_calls)

            (receipt.directory / "nested" / "weights.txt").write_text(
                "tampered", encoding="utf-8"
            )
            hash_guarded = TinyCheckpoint(value="hash-sentinel", version=42)
            with self.assertRaisesRegex(RuntimeError, "hash"):
                load_checkpoint_verified(hash_guarded, receipt.directory)
            self.assertEqual("hash-sentinel", hash_guarded.value)
            self.assertEqual(42, hash_guarded.version)
            self.assertEqual(0, hash_guarded.load_calls)


if __name__ == "__main__":
    unittest.main()
