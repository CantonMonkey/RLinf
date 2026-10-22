"""Focused, dependency-free checks for the two S1 policy adapters."""

from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from navrl.stack import (
    CheckpointAdapter,
    DecisionBatch,
    DecisionRecord,
    EpisodeIdentity,
    ExecutionReceipt,
    PolicyAdapter,
    load_checkpoint_verified,
    save_checkpoint_atomic,
)


ROOT = Path(__file__).resolve().parents[1]


def load_adapters():
    path = ROOT / "integrations/rlinf_v03/vln_stack_adapters.py"
    spec = importlib.util.spec_from_file_location("s1_vln_stack_adapters", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ADAPTERS = load_adapters()


class FakeParameter:
    def __init__(
        self,
        count: int,
        *,
        trainable: bool,
        shape=None,
        dtype="torch.bfloat16",
    ):
        self._count = count
        self.requires_grad = trainable
        self.shape = tuple(shape or (count,))
        self.dtype = dtype

    def numel(self):
        return self._count


class FakePeftConfig:
    def __init__(self, alpha, *, bias="none", init_lora_weights=True):
        self.alpha = alpha
        self.bias = bias
        self.init_lora_weights = init_lora_weights

    def to_dict(self):
        return {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "r": 8,
            "target_modules": r"^model\.layers\.(24|25|26|27)\.self_attn\.(q_proj|v_proj)$",
            "lora_alpha": self.alpha,
            "lora_dropout": 0.0,
            "fan_in_fan_out": False,
            "bias": self.bias,
            "use_rslora": False,
            "use_dora": False,
            "rank_pattern": {},
            "alpha_pattern": {},
            "init_lora_weights": self.init_lora_weights,
        }


class FakePeftBackbone:
    def __init__(self, *, alpha, trainable, bias="none", init_lora_weights=True):
        self.peft_config = {
            "default": FakePeftConfig(
                alpha, bias=bias, init_lora_weights=init_lora_weights
            )
        }
        self._parameters = tuple(
            (
                f"model.layers.{24 + index // 4}.self_attn.q_proj.lora_{'A' if index % 2 == 0 else 'B'}.default.weight_{index}",
                FakeParameter(22528, trainable=trainable),
            )
            for index in range(16)
        )

    def named_parameters(self):
        return iter(self._parameters)


class FakeValueHead:
    def state_dict(self):
        return {
            "weight": FakeParameter(8, trainable=True, shape=(1, 8)),
            "bias": FakeParameter(1, trainable=True, shape=(1,)),
        }


class FakeCheckpointStreamPolicy:
    def __init__(
        self,
        *,
        alpha=16,
        temperature=1.0,
        actor_version=3,
        trainable=True,
        seed=20260926,
        deterministic_eval=False,
        bias="none",
        init_lora_weights=True,
    ):
        self.backbone = FakePeftBackbone(
            alpha=alpha,
            trainable=trainable,
            bias=bias,
            init_lora_weights=init_lora_weights,
        )
        self.value_head = FakeValueHead()
        self.temperature = temperature
        self.context_tokens = 512
        self.action_token_ids = (101, 102, 103, 104)
        self.action_dim = 1
        self.num_action_chunks = 4
        self.discrete_n_actions = 4
        self.seed = seed
        self.deterministic_eval = deterministic_eval
        self.actor_version = actor_version
        self._session = None
        self._sample = None
        self.source_kind = (
            "upstream-plus-trainable-lora"
            if trainable
            else "upstream-plus-explicit-adapter"
        )

    def trainable_manifest(self):
        names = [name for name, row in self.backbone.named_parameters() if row.requires_grad]
        return {
            "source_kind": self.source_kind,
            "trainable_names": names,
            "trainable_parameter_count": sum(
                row.numel() for _, row in self.backbone.named_parameters() if row.requires_grad
            ),
        }


class FakeStreamCheckpointCodec(ADAPTERS.StreamVLNCheckpointAdapter):
    """Dependency-free payload used to exercise the real compatibility contract."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_calls = 0

    def save_checkpoint(self, staging_directory):
        value = str(self.adapter.policy_version)
        path = staging_directory / "fake_actor_version.txt"
        path.write_text(value, encoding="utf-8")
        return {path.name: hashlib.sha256(value.encode()).hexdigest()}

    def load_checkpoint(self, directory):
        self.load_calls += 1
        self.adapter.policy.actor_version = int(
            (directory / "fake_actor_version.txt").read_text(encoding="utf-8")
        )

    def inspect_checkpoint_metadata(self, directory):
        version = int(
            (directory / "fake_actor_version.txt").read_text(encoding="utf-8")
        )
        return {
            "checkpoint_kind": "streamvln_lora_value_v1",
            "actor_version": version,
        }


FROZEN_DEPENDENCIES = {
    "stream_source_commit": "8" * 40,
    "base_model_manifest_sha256": "a" * 64,
    "tokenizer_manifest_sha256": "b" * 64,
    "vision_tower_manifest_sha256": "c" * 64,
}


class FakeVLNPolicy:
    def __init__(self):
        self.last_action = 2
        self.root_calls = []

    def __call__(self, *, operation, **kwargs):
        self.root_calls.append(operation)
        if operation == "sample":
            return self.predict_action_batch(kwargs["env_obs"], mode=kwargs["mode"])
        if operation == "replay":
            return self.default_forward(**kwargs)
        if operation == "bootstrap":
            _, values = self.logits_and_values(kwargs["env_obs"])
            return values
        raise ValueError(operation)

    def named_parameters(self):
        return iter(
            (
                ("actor.weight", FakeParameter(4, trainable=True)),
                ("value.weight", FakeParameter(1, trainable=True)),
                ("frozen.weight", FakeParameter(8, trainable=False)),
            )
        )

    def predict_action_batch(self, observation, mode="train"):
        if mode != "train":
            raise AssertionError("unexpected fake mode")
        forward = {"observation": observation, "action": (self.last_action,)}
        return (self.last_action,), {
            "prev_logprobs": (-0.25,),
            "prev_values": (0.5,),
            "forward_inputs": forward,
        }

    def default_forward(self, *, forward_inputs, **kwargs):
        del kwargs
        if forward_inputs["action"] != (self.last_action,):
            raise AssertionError("replay action changed")
        return {"logprobs": (-0.25,), "entropy": (0.75,), "values": (0.5,)}

    def preprocess_env_obs(self, observation):
        return observation

    def logits_and_values(self, observation):
        del observation
        return ((0.0, 1.0, 2.0, 3.0),), (0.625,)


class FakeSession:
    def __init__(self):
        self.sampled_ids = []
        self.log_probs = []
        self.entropies = []
        self.state_values = []
        self.actions = []
        self.parts = ["prompt"]
        self.features = {}


class FakeStreamPolicy:
    def __init__(self):
        self.actor_version = 3
        self.behavior_version = 3
        self._session = None
        self._sample = None
        self.root_calls = []

    def __call__(self, *, operation, **kwargs):
        self.root_calls.append(operation)
        if operation == "sample":
            return self.sample_observation(kwargs["obs"])
        if operation == "replay":
            return self.replay_current()
        if operation == "bootstrap":
            return self.bootstrap_observation(
                kwargs["obs"], kwargs["executed_actions"], kwargs["terminal"]
            )
        raise ValueError(operation)

    def _ensure(self):
        if self._session is None:
            self._session = FakeSession()
        return self._session

    def sample_observation(self, observation):
        session = self._ensure()
        decision = int(observation["decision"])
        if decision == 0:
            actions = (1, 2, 3, 1)
            mask = (True, True, True, True)
            terms = (-0.1, -0.2, -0.3, -0.4)
            value = 0.4
        else:
            actions = (1, 2, 3, 1)
            mask = (True, True, True, True)
            terms = (-0.6, -0.7, -0.8, -0.9)
            value = 0.2
        count = sum(mask)
        session.sampled_ids.extend(range(count))
        session.log_probs.extend(terms[:count])
        session.entropies.extend((0.5,) * count)
        session.state_values.append(value)
        session.parts.append(f"decision-{decision}")
        self._sample = {"proposed_actions": list(actions[:count])}
        self.behavior_version = self.actor_version
        return {
            "actions": actions,
            "executed_mask": mask,
            "behavior_log_probs": terms,
            "sampled_tokens": tuple(range(count)),
            "behavior_version": self.behavior_version,
        }

    def replay_current(self):
        session = self._ensure()
        return {
            "log_probs": tuple(session.log_probs),
            "entropy": tuple(session.entropies),
            "state_value": session.state_values[-1],
            "state_values": tuple(session.state_values),
            "actor_version": self.actor_version,
        }

    def bootstrap_observation(self, observation, executed_actions, terminal):
        if terminal:
            raise AssertionError("live S1 bootstrap was marked terminal")
        session = self._ensure()
        actions = list(executed_actions)
        session.actions.extend(actions)
        session.parts.append("bootstrap")
        session.features[observation["physical_step"]] = "feature"
        return 0.875

    def trainable_manifest(self):
        return {
            "source_kind": "upstream-plus-trainable-lora",
            "trainable_names": ["backbone.lora_A", "value_head.weight"],
            "trainable_parameter_count": 9,
        }

    def mark_updated(self):
        self.actor_version += 1


def receipt(
    sample,
    *,
    duration: int,
    terminated: bool = False,
    next_observation=None,
):
    width = len(sample.action_payload)
    primitives = sample.action_payload[:duration] + (None,) * (width - duration)
    return ExecutionReceipt(
        identity=sample.identity,
        decision_index=sample.decision_index,
        physical_step=sample.physical_step,
        source_action_payload=sample.action_payload,
        executed_primitives=primitives,
        executed_mask=(True,) * duration + (False,) * (width - duration),
        credited_log_prob_mask=(True,) * duration
        + (False,) * (len(sample.behavior_log_prob_terms) - duration),
        primitive_rewards=(1.0,) * duration + (0.0,) * (width - duration),
        duration=duration,
        next_observation=next_observation or {"physical_step": sample.physical_step + duration},
        terminated=terminated,
        truncated=False,
        end_reason="terminated" if terminated else "proposal_prefix",
        primitive_feedback=tuple({"index": index} for index in range(duration)),
    )


def record(adapter, sample, execution, *, next_value, bootstrap_identity):
    return DecisionRecord(
        sample=sample,
        execution=execution,
        old_joint_log_prob=adapter.behavior_joint_log_prob(sample, execution),
        next_value=next_value,
        bootstrap_identity=bootstrap_identity,
    )


class S1PluginAdapterTest(unittest.TestCase):
    def test_lightweight_adapter_replays_and_rejects_stale_reset(self):
        policy = FakeVLNPolicy()
        adapter = ADAPTERS.VLNPolicyStackAdapter(policy, root_forward=policy)
        self.assertIsInstance(adapter, PolicyAdapter)
        identity = EpisodeIdentity("train-a", "reset-a")
        adapter.reset_state(identity)
        sample = adapter.sample({"rgb": "frame"}, identity, 0, 0)
        execution = receipt(sample, duration=1, terminated=True)
        row = record(adapter, sample, execution, next_value=0.0, bootstrap_identity=None)
        batch = DecisionBatch((row,))

        self.assertEqual((2,), sample.action_payload)
        self.assertEqual((-0.25,), sample.behavior_log_prob_terms)
        self.assertEqual(-0.25, row.old_joint_log_prob)
        evaluated = adapter.evaluate(batch)
        self.assertEqual(-0.25, evaluated[0].joint_log_prob)
        self.assertEqual(0.5, evaluated[0].value)
        bootstrap = adapter.bootstrap({}, identity, 1)
        self.assertEqual(identity, bootstrap.identity)
        self.assertEqual(1, bootstrap.physical_step)
        self.assertEqual(0, bootstrap.policy_version)
        self.assertEqual(0.625, bootstrap.value)
        self.assertEqual(("actor.weight", "value.weight"), adapter.trainable_parameter_names())
        self.assertFalse(adapter.trainable_manifest()["stateful_replay"])
        self.assertEqual("rlinf_root", adapter.trainable_manifest()["forward_path"])
        self.assertEqual(["sample", "replay", "bootstrap"], policy.root_calls)
        self.assertTrue(issubclass(ADAPTERS.VLNPolicyCheckpointAdapter, CheckpointAdapter))
        adapter.mark_updated()
        with self.assertRaisesRegex(RuntimeError, "version"):
            adapter.evaluate(batch)

        replacement = EpisodeIdentity("train-b", "reset-b")
        adapter.reset_state(replacement)
        with self.assertRaisesRegex(RuntimeError, "stale"):
            adapter.evaluate(batch)
        with self.assertRaisesRegex(RuntimeError, "stale"):
            adapter.sample({}, identity, 0, 0)

    def test_stream_adapter_uses_executed_prefix_and_ordered_multirow_replay(self):
        policy = FakeStreamPolicy()
        adapter = ADAPTERS.StreamVLNPolicyStackAdapter(policy, root_forward=policy)
        self.assertIsInstance(adapter, PolicyAdapter)
        identity = EpisodeIdentity("train-stream", "reset-stream")
        adapter.reset_state(identity)

        first = adapter.sample({"batch_size": 1, "decision": 0}, identity, 0, 0)
        first_execution = receipt(
            first,
            duration=4,
            next_observation={"batch_size": 1, "decision": 1, "physical_step": 4},
        )
        self.assertAlmostEqual(-1.0, adapter.behavior_joint_log_prob(first, first_execution))

        session_before = copy.deepcopy(vars(policy._session))
        bootstrap = adapter.bootstrap(first_execution.next_observation, identity, 4)
        self.assertEqual(identity, bootstrap.identity)
        self.assertEqual(4, bootstrap.physical_step)
        self.assertEqual(3, bootstrap.policy_version)
        self.assertEqual(0.875, bootstrap.value)
        self.assertEqual(session_before, vars(policy._session))
        with self.assertRaisesRegex(RuntimeError, "full four-primitive"):
            adapter.bootstrap(first_execution.next_observation, identity, 2)

        second = adapter.sample(first_execution.next_observation, identity, 1, 4)
        second_execution = receipt(second, duration=2, terminated=True)
        wrong_credit = replace(
            second_execution,
            credited_log_prob_mask=(True,) * len(second.behavior_log_prob_terms),
        )
        with self.assertRaisesRegex(RuntimeError, "executed_prefix"):
            adapter.behavior_joint_log_prob(second, wrong_credit)
        first_row = record(
            adapter,
            first,
            first_execution,
            next_value=second.value,
            bootstrap_identity=identity,
        )
        second_row = record(
            adapter,
            second,
            second_execution,
            next_value=0.0,
            bootstrap_identity=None,
        )
        batch = DecisionBatch((first_row, second_row))
        evaluated = adapter.evaluate(batch)

        self.assertEqual(2, len(evaluated))
        self.assertAlmostEqual(-1.0, evaluated[0].joint_log_prob)
        self.assertAlmostEqual(-1.3, evaluated[1].joint_log_prob)
        self.assertEqual((0.4, 0.2), tuple(row.value for row in evaluated))
        self.assertEqual((0.5, 0.5), tuple(row.entropy for row in evaluated))
        self.assertEqual(3, evaluated[0].policy_version)
        self.assertEqual(
            "ordered_b1_current_block", adapter.trainable_manifest()["replay_batching"]
        )
        self.assertEqual(
            4, adapter.trainable_manifest()["nonterminal_boundary_primitives"]
        )
        self.assertEqual("rlinf_root", adapter.trainable_manifest()["forward_path"])
        self.assertEqual(
            ["sample", "replay", "bootstrap", "sample", "replay", "replay"],
            policy.root_calls,
        )

        with self.assertRaisesRegex((RuntimeError, ValueError), "stale|shuffled|complete"):
            adapter.evaluate(DecisionBatch((first_row,)))
        with self.assertRaisesRegex(ValueError, "B=1"):
            adapter.sample({"batch_size": 2, "decision": 2}, identity, 2, 3)
        adapter.mark_updated()
        with self.assertRaisesRegex(RuntimeError, "version"):
            adapter.evaluate(batch)

        replacement = EpisodeIdentity("train-stream-b", "reset-stream-b")
        adapter.reset_state(replacement)
        self.assertIsNone(policy._session)
        self.assertIsNone(policy._sample)
        with self.assertRaisesRegex(RuntimeError, "stale"):
            adapter.evaluate(batch)

    def test_stream_checkpoint_compatibility_precedes_all_mutation(self):
        source_policy = FakeCheckpointStreamPolicy(trainable=True, actor_version=3)
        source = FakeStreamCheckpointCodec(
            ADAPTERS.StreamVLNPolicyStackAdapter(source_policy),
            frozen_dependency_ids=FROZEN_DEPENDENCIES,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "checkpoint"
            saved = save_checkpoint_atomic(source, checkpoint)

            # A frozen independent restore has different runtime trainability and
            # source_kind, but the same persisted model identity.
            restored_policy = FakeCheckpointStreamPolicy(
                trainable=False, actor_version=0
            )
            restored = FakeStreamCheckpointCodec(
                ADAPTERS.StreamVLNPolicyStackAdapter(restored_policy),
                frozen_dependency_ids=FROZEN_DEPENDENCIES,
            )
            loaded = load_checkpoint_verified(restored, checkpoint)
            self.assertEqual(saved.compatibility, loaded.compatibility)
            self.assertEqual(1, restored.load_calls)
            self.assertEqual(3, restored_policy.actor_version)

            mismatches = (
                (
                    "lora alpha",
                    FakeCheckpointStreamPolicy(alpha=32, actor_version=11),
                    FROZEN_DEPENDENCIES,
                ),
                (
                    "temperature",
                    FakeCheckpointStreamPolicy(temperature=0.5, actor_version=12),
                    FROZEN_DEPENDENCIES,
                ),
                (
                    "sampling seed",
                    FakeCheckpointStreamPolicy(seed=7, actor_version=14),
                    FROZEN_DEPENDENCIES,
                ),
                (
                    "deterministic evaluation",
                    FakeCheckpointStreamPolicy(
                        deterministic_eval=True, actor_version=15
                    ),
                    FROZEN_DEPENDENCIES,
                ),
                (
                    "frozen dependency",
                    FakeCheckpointStreamPolicy(actor_version=13),
                    {
                        **FROZEN_DEPENDENCIES,
                        "base_model_manifest_sha256": "d" * 64,
                    },
                ),
            )
            for label, policy, dependencies in mismatches:
                with self.subTest(label=label):
                    target = FakeStreamCheckpointCodec(
                        ADAPTERS.StreamVLNPolicyStackAdapter(policy),
                        frozen_dependency_ids=dependencies,
                    )
                    before = policy.actor_version
                    with self.assertRaisesRegex(RuntimeError, "compatibility"):
                        load_checkpoint_verified(target, checkpoint)
                    self.assertEqual(0, target.load_calls)
                    self.assertEqual(before, policy.actor_version)

    def test_stream_codec_direct_load_checks_contract_before_tensor_io(self):
        source_policy = FakeCheckpointStreamPolicy(alpha=16, actor_version=3)
        source = ADAPTERS.StreamVLNCheckpointAdapter(
            ADAPTERS.StreamVLNPolicyStackAdapter(source_policy),
            frozen_dependency_ids=FROZEN_DEPENDENCIES,
        )
        target_policy = FakeCheckpointStreamPolicy(alpha=32, actor_version=17)
        target = ADAPTERS.StreamVLNCheckpointAdapter(
            ADAPTERS.StreamVLNPolicyStackAdapter(target_policy),
            frozen_dependency_ids=FROZEN_DEPENDENCIES,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "stack_state.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "actor_version": 3,
                        "compatibility": source.checkpoint_compatibility(),
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "compatibility"):
                target.load_checkpoint(root)
        self.assertEqual(17, target_policy.actor_version)

        for label, policy in (
            ("trained bias", FakeCheckpointStreamPolicy(bias="all")),
            (
                "base-mutating initializer",
                FakeCheckpointStreamPolicy(init_lora_weights="pissa"),
            ),
        ):
            with self.subTest(label=label):
                codec = ADAPTERS.StreamVLNCheckpointAdapter(
                    ADAPTERS.StreamVLNPolicyStackAdapter(policy),
                    frozen_dependency_ids=FROZEN_DEPENDENCIES,
                )
                with self.assertRaisesRegex(RuntimeError, "extra PEFT state"):
                    codec.checkpoint_compatibility()

    def test_s1_source_closure_is_training_only(self):
        sources = list((ROOT / "navrl/stack").glob("*.py"))
        sources.extend(
            (
                ROOT / "integrations/rlinf_v03/vln_stack_adapters.py",
                Path(__file__),
            )
        )
        blocked = ("val" + "_" + "seen", "val" + "_" + "unseen")
        for path in sources:
            text = path.read_text(encoding="utf-8")
            for marker in blocked:
                self.assertNotIn(marker, text, str(path))


if __name__ == "__main__":
    unittest.main()
