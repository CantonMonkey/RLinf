"""Thin ``navrl.stack`` adapters for the two staged VLN policies.

The lightweight policy is stateless and can replay ordinary decision batches.
The original StreamVLN policy keeps its causal token history inside the model,
so its S1 adapter accepts only the currently bound, complete ordered rows from
one B=1 context block.  It fails closed on stale or shuffled rows instead of
claiming that the mutable upstream session is a pure batched policy.
"""

from __future__ import annotations

from contextlib import nullcontext
import copy
import hashlib
import json
from pathlib import Path
import re
from typing import Callable, Mapping

from navrl.stack import (
    BootstrapValue,
    CheckpointAdapter,
    DecisionBatch,
    EpisodeIdentity,
    ExecutionReceipt,
    PolicyAdapter,
    PolicyEvaluation,
    PolicySample,
)


def _flat_python(value: object) -> list[object]:
    """Detach tensor-like values only when converting actions or masks."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "reshape") and hasattr(value, "tolist"):
        return list(value.reshape(-1).tolist())
    if isinstance(value, (tuple, list)):
        result: list[object] = []
        for item in value:
            result.extend(_flat_python(item))
        return result
    return [value]


def _flat_length(value: object) -> int:
    if hasattr(value, "numel"):
        return int(value.numel())
    return len(_flat_python(value))


def _slice_sum(value: object, start: int, stop: int) -> object:
    """Sum a flat slice while retaining gradients for tensor-like inputs."""
    if hasattr(value, "reshape"):
        return value.reshape(-1)[start:stop].sum()
    flat = _flat_python(value)[start:stop]
    if not flat:
        raise ValueError("cannot sum an empty policy slice")
    return sum(flat)


def _slice_mean(value: object, start: int, stop: int) -> object:
    if stop <= start:
        raise ValueError("cannot average an empty policy slice")
    if hasattr(value, "reshape"):
        return value.reshape(-1)[start:stop].mean()
    flat = _flat_python(value)[start:stop]
    if not flat:
        raise ValueError("cannot average an empty policy slice")
    return sum(flat) / len(flat)


def _scalar(value: object, index: int = 0) -> object:
    if hasattr(value, "reshape"):
        flat = value.reshape(-1)
        if int(flat.numel()) != 1 and not -int(flat.numel()) <= index < int(flat.numel()):
            raise ValueError("policy scalar index is outside the returned value")
        return flat[index]
    flat = _flat_python(value)
    if len(flat) != 1 and not -len(flat) <= index < len(flat):
        raise ValueError("policy scalar index is outside the returned value")
    return flat[index]


def _integer_scalar(value: object) -> int:
    item = _scalar(value)
    if hasattr(item, "item"):
        item = item.item()
    return int(item)


def _prefix_length(mask: object) -> int:
    values = [bool(item) for item in _flat_python(mask)]
    duration = 0
    padding = False
    for valid in values:
        if valid:
            if padding:
                raise ValueError("policy action mask must be prefix-contiguous")
            duration += 1
        else:
            padding = True
    if duration < 1:
        raise ValueError("policy must propose at least one primitive")
    return duration


def _raw_prefix(value: object, duration: int) -> object:
    if hasattr(value, "reshape"):
        return value.reshape(-1)[:duration]
    return tuple(_flat_python(value)[:duration])


def _terms(value: object, count: int) -> tuple[object, ...]:
    if hasattr(value, "reshape"):
        flat = value.reshape(-1)
        if int(flat.numel()) < count:
            raise ValueError("policy returned fewer log-prob terms than actions")
        return tuple(flat[index] for index in range(count))
    flat = _flat_python(value)
    if len(flat) < count:
        raise ValueError("policy returned fewer log-prob terms than actions")
    return tuple(flat[:count])


def _behavior_prefix(sample: PolicySample, execution: ExecutionReceipt) -> object:
    if sample.identity != execution.identity:
        raise RuntimeError("sample and execution reset identities differ")
    if sample.decision_index != execution.decision_index:
        raise RuntimeError("sample and execution decision indices differ")
    if sample.physical_step != execution.physical_step:
        raise RuntimeError("sample and execution physical steps differ")
    if sample.action_payload != execution.source_action_payload:
        raise RuntimeError("execution receipt belongs to another sampled payload")
    mask = execution.credited_log_prob_mask
    if len(mask) != len(sample.behavior_log_prob_terms):
        raise RuntimeError("execution credit mask and behavior terms differ")
    return sum(
        term for term, credited in zip(sample.behavior_log_prob_terms, mask) if credited
    )


def _require_direct_prefix(
    sample: PolicySample,
    execution: ExecutionReceipt,
    *,
    credit_mode: str,
) -> None:
    if execution.active_primitives != sample.action_payload[: execution.duration]:
        raise RuntimeError("executed primitives differ from the sampled action prefix")
    term_count = len(sample.behavior_log_prob_terms)
    if credit_mode == "whole_decision":
        expected = (True,) * term_count
    elif credit_mode == "executed_prefix":
        if execution.duration > term_count:
            raise RuntimeError("execution duration exceeds policy log-prob terms")
        expected = (True,) * execution.duration + (False,) * (
            term_count - execution.duration
        )
    else:
        raise ValueError("unknown policy credit mode")
    if execution.credited_log_prob_mask != expected:
        raise RuntimeError(f"{credit_mode} log-prob credit mask is required")


def _trainable_names(module: object) -> tuple[str, ...]:
    if not hasattr(module, "named_parameters"):
        return ()
    return tuple(
        sorted(
            name
            for name, parameter in module.named_parameters()
            if bool(getattr(parameter, "requires_grad", False))
        )
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _torch_no_grad():
    try:
        import torch
    except ImportError:
        return nullcontext()
    return torch.no_grad()


class VLNPolicyStackAdapter(PolicyAdapter):
    """Adapt the existing one-step categorical ``VLNPolicy`` without copying it."""

    action_distribution = "categorical"

    def __init__(
        self,
        policy: object,
        *,
        mode: str = "train",
        policy_version: int = 0,
        root_forward: Callable[..., object] | None = None,
    ):
        if mode not in ("train", "eval"):
            raise ValueError("VLN policy mode must be train or eval")
        if type(policy_version) is not int or policy_version < 0:
            raise ValueError("policy_version must be a non-negative integer")
        self.policy = policy
        self.root_forward = root_forward
        self.mode = mode
        self._identity: EpisodeIdentity | None = None
        self._policy_version = policy_version
        self._generation = 0

    def _forward(self, operation: str, **kwargs) -> object:
        if self.root_forward is not None:
            return self.root_forward(operation=operation, **kwargs)
        if operation == "sample":
            return self.policy.predict_action_batch(
                kwargs["env_obs"], mode=kwargs.get("mode", "train")
            )
        if operation == "replay":
            return self.policy.default_forward(**kwargs)
        if operation == "bootstrap":
            prepared = (
                self.policy.preprocess_env_obs(kwargs["env_obs"])
                if hasattr(self.policy, "preprocess_env_obs")
                else kwargs["env_obs"]
            )
            _, values = self.policy.logits_and_values(prepared)
            return values
        raise ValueError(f"unknown lightweight VLN operation {operation!r}")

    def _require_identity(self, identity: EpisodeIdentity) -> None:
        if not isinstance(identity, EpisodeIdentity):
            raise TypeError("policy identity must be EpisodeIdentity")
        if identity != self._identity:
            raise RuntimeError("stale or unbound policy reset identity")

    def reset_state(self, identity: EpisodeIdentity) -> None:
        if not isinstance(identity, EpisodeIdentity):
            raise TypeError("policy identity must be EpisodeIdentity")
        self._identity = identity
        self._generation += 1

    def sample(
        self,
        observation: object,
        identity: EpisodeIdentity,
        decision_index: int,
        physical_step: int,
    ) -> PolicySample:
        self._require_identity(identity)
        actions, sampled = self._forward(
            "sample", env_obs=observation, mode=self.mode
        )
        action_values = _flat_python(actions)
        if len(action_values) != 1:
            raise ValueError("one stack identity requires a single VLN policy row")
        forward_inputs = sampled.get("forward_inputs")
        if not isinstance(forward_inputs, Mapping) or not forward_inputs:
            raise ValueError("VLN policy did not return replay forward inputs")
        return PolicySample(
            identity=identity,
            decision_index=decision_index,
            physical_step=physical_step,
            action_payload=(int(action_values[0]),),
            replay_context={
                "adapter_kind": "vln_policy",
                "generation": self._generation,
                "forward_inputs": forward_inputs,
            },
            behavior_log_prob_terms=(_scalar(sampled["prev_logprobs"]),),
            value=_scalar(sampled["prev_values"]),
            policy_version=self.policy_version,
            action_distribution=self.action_distribution,
        )

    def behavior_joint_log_prob(
        self, sample: PolicySample, execution: ExecutionReceipt
    ) -> object:
        self._require_identity(sample.identity)
        if sample.replay_context.get("generation") != self._generation:
            raise RuntimeError("stale VLN replay generation")
        _require_direct_prefix(sample, execution, credit_mode="whole_decision")
        return _behavior_prefix(sample, execution)

    def evaluate(self, batch: DecisionBatch) -> tuple[PolicyEvaluation, ...]:
        self._require_identity(batch.identity)
        rows = []
        for record in batch.records:
            sample = record.sample
            self._require_identity(sample.identity)
            if sample.policy_version != self.policy_version:
                raise RuntimeError("VLN replay policy version differs from its behavior version")
            _require_direct_prefix(
                sample, record.execution, credit_mode="whole_decision"
            )
            context = sample.replay_context
            if (
                context.get("adapter_kind") != "vln_policy"
                or context.get("generation") != self._generation
            ):
                raise ValueError("VLN replay context belongs to another adapter")
            forward_inputs = context.get("forward_inputs")
            if not isinstance(forward_inputs, Mapping) or not forward_inputs:
                raise ValueError("VLN replay context has no forward inputs")
            output = self._forward(
                "replay",
                forward_inputs=forward_inputs,
                compute_logprobs=True,
                compute_entropy=True,
                compute_values=True,
            )
            rows.append(
                PolicyEvaluation(
                    identity=sample.identity,
                    decision_index=sample.decision_index,
                    joint_log_prob=_slice_sum(
                        output["logprobs"],
                        0,
                        sum(record.execution.credited_log_prob_mask),
                    ),
                    value=_scalar(output["values"]),
                    policy_version=self.policy_version,
                    entropy=_slice_sum(output["entropy"], 0, 1),
                )
            )
        return tuple(rows)

    def bootstrap(
        self,
        observation: object,
        identity: EpisodeIdentity,
        physical_step: int,
    ) -> BootstrapValue:
        self._require_identity(identity)
        with _torch_no_grad():
            values = self._forward("bootstrap", env_obs=observation)
        if _flat_length(values) != 1:
            raise ValueError("one stack identity requires one bootstrap value")
        return BootstrapValue(
            identity=identity,
            physical_step=physical_step,
            policy_version=self.policy_version,
            value=_scalar(values),
        )

    def trainable_parameter_names(self) -> tuple[str, ...]:
        return _trainable_names(self.policy)

    def trainable_manifest(self) -> Mapping[str, object]:
        names = self.trainable_parameter_names()
        return {
            "source_kind": "lightweight_vln_policy",
            "stateful_replay": False,
            "forward_path": "rlinf_root" if self.root_forward is not None else "direct_local",
            "trainable_names": list(names),
            "policy_version": self.policy_version,
        }

    @property
    def policy_version(self) -> int:
        return self._policy_version

    def mark_updated(self) -> None:
        self._policy_version += 1


class StreamVLNPolicyStackAdapter(PolicyAdapter):
    """Ordered B=1 bridge around ``StreamVLNPPOPolicy``'s mutable session.

    Its executor must use executed-prefix log-probability credit.
    """

    action_distribution = "autoregressive_categorical_chunk"

    def __init__(
        self,
        policy: object,
        *,
        root_forward: Callable[..., object] | None = None,
    ):
        self.policy = policy
        self.root_forward = root_forward
        self._identity: EpisodeIdentity | None = None
        self._generation = 0
        self._serial = 0
        self._rows: list[dict[str, object]] = []

    def _forward(self, operation: str, **kwargs) -> object:
        if self.root_forward is not None:
            return self.root_forward(operation=operation, **kwargs)
        if operation == "sample":
            return self.policy.sample_observation(kwargs["obs"])
        if operation == "replay":
            return self.policy.replay_current()
        if operation == "bootstrap":
            return self.policy.bootstrap_observation(
                kwargs["obs"], kwargs["executed_actions"], bool(kwargs["terminal"])
            )
        raise ValueError(f"unknown StreamVLN operation {operation!r}")

    def _require_identity(self, identity: EpisodeIdentity) -> None:
        if not isinstance(identity, EpisodeIdentity):
            raise TypeError("policy identity must be EpisodeIdentity")
        if identity != self._identity:
            raise RuntimeError("stale or unbound StreamVLN reset identity")

    @staticmethod
    def _require_single_observation(observation: object) -> None:
        if not isinstance(observation, Mapping):
            return
        if "episode_step" in observation:
            value = observation["episode_step"]
            shape = getattr(value, "shape", None)
            if shape is not None and len(shape) and int(shape[0]) != 1:
                raise ValueError("StreamVLN S1 replay supports only ordered B=1")
        if observation.get("batch_size", 1) != 1:
            raise ValueError("StreamVLN S1 replay supports only ordered B=1")

    def reset_state(self, identity: EpisodeIdentity) -> None:
        if not isinstance(identity, EpisodeIdentity):
            raise TypeError("policy identity must be EpisodeIdentity")
        self._identity = identity
        self._generation += 1
        self._serial = 0
        self._rows = []
        # These are the complete mutable rollout roots in StreamVLNPPOPolicy.
        # Actor/version weights deliberately survive an episode reset.
        if hasattr(self.policy, "_session"):
            self.policy._session = None
        if hasattr(self.policy, "_sample"):
            self.policy._sample = None

    def sample(
        self,
        observation: object,
        identity: EpisodeIdentity,
        decision_index: int,
        physical_step: int,
    ) -> PolicySample:
        self._require_identity(identity)
        self._require_single_observation(observation)
        if not self._rows:
            if decision_index != 0 or physical_step != 0:
                raise RuntimeError("first StreamVLN decision must start at index and step zero")
        else:
            previous = self._rows[-1]
            prior_sample = previous["sample"]
            if decision_index != prior_sample.decision_index + 1:
                raise RuntimeError("StreamVLN decisions must remain ordered")
            elapsed = physical_step - prior_sample.physical_step
            if elapsed != 4:
                raise RuntimeError(
                    "continued StreamVLN sampling requires a full four-primitive boundary"
                )
            if physical_step // 32 != self._rows[0]["block_index"]:
                raise RuntimeError(
                    "StreamVLN S1 replay cannot cross an upstream 32-step context block"
                )

        sampled = self._forward("sample", obs=observation)
        duration = _prefix_length(sampled["executed_mask"])
        raw_actions = sampled["actions"]
        proposed = tuple(int(item) for item in _flat_python(raw_actions)[:duration])
        if len(proposed) != duration:
            raise ValueError("StreamVLN action payload is shorter than its mask")
        with _torch_no_grad():
            preview = self._forward("replay")
        token_end = _flat_length(preview["log_probs"])
        token_start = token_end - duration
        if token_start < 0:
            raise RuntimeError("StreamVLN replay returned fewer tokens than it sampled")
        value_count = _flat_length(preview["state_values"])
        value_index = value_count - 1
        if value_index < 0:
            raise RuntimeError("StreamVLN replay returned no decision value")
        behavior_version = _integer_scalar(sampled["behavior_version"])
        if behavior_version != self.policy_version:
            raise RuntimeError("StreamVLN behavior and actor versions disagree at sampling")
        self._serial += 1
        context = {
            "adapter_kind": "streamvln_stateful_ordered_b1",
            "generation": self._generation,
            "serial": self._serial,
            "token_start": token_start,
            "token_end": token_end,
            "value_index": value_index,
            "behavior_version": behavior_version,
        }
        result = PolicySample(
            identity=identity,
            decision_index=decision_index,
            physical_step=physical_step,
            action_payload=proposed,
            replay_context=context,
            behavior_log_prob_terms=_terms(sampled["behavior_log_probs"], duration),
            value=_scalar(preview["state_values"], value_index),
            policy_version=behavior_version,
            action_distribution=self.action_distribution,
        )
        self._rows.append({
            "sample": result,
            "raw_actions": raw_actions,
            "block_index": physical_step // 32,
        })
        return result

    def behavior_joint_log_prob(
        self, sample: PolicySample, execution: ExecutionReceipt
    ) -> object:
        self._require_identity(sample.identity)
        self._require_saved(sample)
        _require_direct_prefix(sample, execution, credit_mode="executed_prefix")
        return _behavior_prefix(sample, execution)

    def _require_saved(self, sample: PolicySample) -> dict[str, object]:
        context = sample.replay_context
        serial = context.get("serial")
        if type(serial) is not int or not 1 <= serial <= len(self._rows):
            raise RuntimeError("stale or shuffled StreamVLN replay row")
        saved = self._rows[serial - 1]
        current = saved["sample"]
        if (
            sample is not current
            or context.get("adapter_kind") != "streamvln_stateful_ordered_b1"
            or context.get("generation") != self._generation
            or context.get("behavior_version") != sample.policy_version
        ):
            raise RuntimeError("stale or shuffled StreamVLN replay row")
        return saved

    def evaluate(self, batch: DecisionBatch) -> tuple[PolicyEvaluation, ...]:
        self._require_identity(batch.identity)
        if len(batch.records) != len(self._rows):
            raise ValueError("StreamVLN replay batch must match the complete ordered block")
        for index, record in enumerate(batch.records):
            saved = self._require_saved(record.sample)
            if saved is not self._rows[index]:
                raise RuntimeError("stale or shuffled StreamVLN replay row")
            if record.sample.policy_version != self.policy_version:
                raise RuntimeError(
                    "StreamVLN replay policy version differs from its behavior version"
                )
            _require_direct_prefix(
                record.sample, record.execution, credit_mode="executed_prefix"
            )
        output = self._forward("replay")
        evaluations = []
        for record in batch.records:
            sample = record.sample
            start = int(sample.replay_context["token_start"])
            proposal_stop = int(sample.replay_context["token_end"])
            stop = start + sum(record.execution.credited_log_prob_mask)
            if stop > proposal_stop:
                raise RuntimeError("StreamVLN execution exceeds its sampled token prefix")
            if _flat_length(output["log_probs"]) < stop:
                raise RuntimeError(
                    "StreamVLN mutable replay history no longer contains this row"
                )
            value_index = int(sample.replay_context["value_index"])
            if _flat_length(output["state_values"]) <= value_index:
                raise RuntimeError(
                    "StreamVLN mutable value history no longer contains this row"
                )
            evaluations.append(
                PolicyEvaluation(
                    identity=sample.identity,
                    decision_index=sample.decision_index,
                    joint_log_prob=_slice_sum(output["log_probs"], start, stop),
                    value=_scalar(output["state_values"], value_index),
                    policy_version=self.policy_version,
                    entropy=_slice_mean(output["entropy"], start, stop),
                )
            )
        return tuple(evaluations)

    @staticmethod
    def _snapshot_session(session: object) -> dict[str, object]:
        state = {}
        for name, value in vars(session).items():
            if isinstance(value, (list, dict, set)):
                state[name] = copy.copy(value)
            else:
                state[name] = value
        return state

    def bootstrap(
        self,
        observation: object,
        identity: EpisodeIdentity,
        physical_step: int,
    ) -> BootstrapValue:
        self._require_identity(identity)
        self._require_single_observation(observation)
        if not self._rows:
            raise RuntimeError("StreamVLN bootstrap requires a current proposal")
        latest = self._rows[-1]
        sample = latest["sample"]
        duration = physical_step - sample.physical_step
        if not 1 <= duration <= len(sample.action_payload):
            raise RuntimeError("StreamVLN bootstrap step is outside the proposal prefix")
        if duration != 4:
            raise RuntimeError(
                "nonterminal StreamVLN bootstrap requires a full four-primitive boundary"
            )
        session = getattr(self.policy, "_session", None)
        if session is None:
            raise RuntimeError("StreamVLN bootstrap has no bound causal session")
        session_state = self._snapshot_session(session)
        sample_state = getattr(self.policy, "_sample", None)
        try:
            with _torch_no_grad():
                value = self._forward(
                    "bootstrap",
                    obs=observation,
                    executed_actions=_raw_prefix(latest["raw_actions"], duration),
                    terminal=False,
                )
            return BootstrapValue(
                identity=identity,
                physical_step=physical_step,
                policy_version=self.policy_version,
                value=value,
            )
        finally:
            self.policy._session = session
            vars(session).clear()
            vars(session).update(session_state)
            if hasattr(self.policy, "_sample"):
                self.policy._sample = sample_state

    def trainable_parameter_names(self) -> tuple[str, ...]:
        manifest = self.policy.trainable_manifest()
        names = manifest.get("trainable_names")
        if isinstance(names, (tuple, list)):
            return tuple(str(name) for name in names)
        return _trainable_names(self.policy)

    def trainable_manifest(self) -> Mapping[str, object]:
        manifest = dict(self.policy.trainable_manifest())
        manifest.update(
            {
                "stack_adapter": "streamvln_stateful_ordered_b1",
                "stateful_replay": True,
                "replay_batching": "ordered_b1_current_block",
                "nonterminal_boundary_primitives": 4,
                "forward_path": "rlinf_root" if self.root_forward is not None else "direct_local",
                "policy_version": self.policy_version,
            }
        )
        return manifest

    @property
    def policy_version(self) -> int:
        return int(self.policy.actor_version)

    def mark_updated(self) -> None:
        self.policy.mark_updated()


class VLNPolicyCheckpointAdapter(CheckpointAdapter):
    """Strict full-state checkpoint codec for the lightweight policy."""

    def __init__(self, adapter: VLNPolicyStackAdapter):
        self.adapter = adapter

    @staticmethod
    def _state_schema(state: Mapping[str, object]) -> Mapping[str, object]:
        return {
            str(name): {
                "shape": [int(item) for item in value.shape],
                "dtype": str(value.dtype),
            }
            for name, value in sorted(state.items())
        }

    def checkpoint_compatibility(self) -> Mapping[str, object]:
        policy = self.adapter.policy
        return {
            "checkpoint_kind": "lightweight_vln_full_state_v1",
            "policy_class": type(policy).__qualname__,
            "policy_semantics": {
                "vocab_size": int(policy.vocab_size),
                "max_instruction_tokens": int(policy.max_instruction_tokens),
                "image_size": int(policy.image_size),
                "action_dim": int(policy.action_dim),
                "num_action_chunks": int(policy.num_action_chunks),
                "discrete_n_actions": int(policy.discrete_n_actions),
            },
            "state_schema": self._state_schema(policy.state_dict()),
            "trainable_names": list(self.adapter.trainable_parameter_names()),
        }

    def checkpoint_metadata(self) -> Mapping[str, object]:
        return {
            "checkpoint_kind": "lightweight_vln_full_state_v1",
            "policy_version": self.adapter.policy_version,
        }

    def save_checkpoint(self, staging_directory: Path) -> Mapping[str, str]:
        import torch

        path = Path(staging_directory) / "policy_state.pt"
        torch.save(
            {
                "state_dict": self.adapter.policy.state_dict(),
                "policy_version": self.adapter.policy_version,
            },
            path,
        )
        return {path.name: _sha256(path)}

    @staticmethod
    def _read_payload(directory: Path) -> Mapping[str, object]:
        import torch

        path = Path(directory) / "policy_state.pt"
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        if set(payload) != {"state_dict", "policy_version"}:
            raise ValueError("lightweight VLN checkpoint payload is incomplete")
        version = payload["policy_version"]
        if type(version) is not int or version < 0:
            raise ValueError("checkpoint policy_version is invalid")
        saved_state = payload["state_dict"]
        if not isinstance(saved_state, Mapping):
            raise ValueError("lightweight VLN state_dict is invalid")
        return payload

    def inspect_checkpoint_metadata(self, directory: Path) -> Mapping[str, object]:
        payload = self._read_payload(directory)
        return {
            "checkpoint_kind": "lightweight_vln_full_state_v1",
            "policy_version": payload["policy_version"],
        }

    def load_checkpoint(self, directory: Path) -> None:
        payload = self._read_payload(directory)
        version = payload["policy_version"]
        saved_state = payload["state_dict"]
        current_state = self.adapter.policy.state_dict()
        if self._state_schema(saved_state) != self._state_schema(current_state):
            raise RuntimeError("lightweight VLN checkpoint state schema differs")
        self.adapter.policy.load_state_dict(saved_state, strict=True)
        self.adapter._policy_version = version
        if self.adapter._identity is not None:
            self.adapter.reset_state(self.adapter._identity)


class StreamVLNCheckpointAdapter(CheckpointAdapter):
    """LoRA/value codec bound to exact frozen dependencies and policy semantics."""

    expected_lora_tensors = 16
    expected_lora_parameters = 360448
    required_dependency_ids = {
        "stream_source_commit": 40,
        "base_model_manifest_sha256": 64,
        "tokenizer_manifest_sha256": 64,
        "vision_tower_manifest_sha256": 64,
    }
    peft_contract_fields = (
        "peft_type",
        "task_type",
        "r",
        "target_modules",
        "exclude_modules",
        "lora_alpha",
        "lora_dropout",
        "fan_in_fan_out",
        "bias",
        "use_rslora",
        "use_dora",
        "use_qalora",
        "qalora_group_size",
        "lora_bias",
        "alora_invocation_tokens",
        "trainable_token_indices",
        "init_lora_weights",
        "loftq_config",
        "eva_config",
        "corda_config",
        "megatron_config",
        "megatron_core",
        "rank_pattern",
        "alpha_pattern",
        "layers_to_transform",
        "layers_pattern",
        "layer_replication",
        "modules_to_save",
        "target_parameters",
    )
    peft_contract_defaults = {
        "lora_dropout": 0.0,
        "fan_in_fan_out": False,
        "bias": "none",
        "use_rslora": False,
        "use_dora": False,
        "use_qalora": False,
        "qalora_group_size": 16,
        "lora_bias": False,
        "rank_pattern": {},
        "alpha_pattern": {},
        "init_lora_weights": True,
        "loftq_config": {},
        "eva_config": None,
        "corda_config": None,
        "megatron_config": None,
        "megatron_core": "megatron.core",
    }

    def __init__(
        self,
        adapter: StreamVLNPolicyStackAdapter,
        *,
        frozen_dependency_ids: Mapping[str, str],
    ):
        self.adapter = adapter
        if set(frozen_dependency_ids) != set(self.required_dependency_ids):
            raise ValueError(
                "StreamVLN checkpoint requires exact source/base/tokenizer/vision identities"
            )
        normalized = {}
        for name, width in self.required_dependency_ids.items():
            value = frozen_dependency_ids[name]
            if not isinstance(value, str) or re.fullmatch(
                rf"[0-9a-fA-F]{{{width}}}", value
            ) is None:
                raise ValueError(f"invalid frozen dependency identity: {name}")
            normalized[name] = value.lower()
        self.frozen_dependency_ids = normalized

    @classmethod
    def _normalize_config_value(cls, value: object) -> object:
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        if hasattr(value, "value") and isinstance(value.value, (str, bool, int, float)):
            return value.value
        if isinstance(value, Mapping):
            return {
                str(key): cls._normalize_config_value(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        if isinstance(value, (set, frozenset)):
            return sorted(
                (cls._normalize_config_value(item) for item in value),
                key=lambda item: json.dumps(item, sort_keys=True),
            )
        if isinstance(value, (tuple, list)):
            return [cls._normalize_config_value(item) for item in value]
        return str(value)

    @classmethod
    def _peft_contract_from_mapping(cls, config: Mapping[str, object]) -> Mapping[str, object]:
        contract = {
            name: cls._normalize_config_value(
                config.get(name, cls.peft_contract_defaults.get(name))
            )
            for name in cls.peft_contract_fields
        }
        for name in (
            "target_modules",
            "exclude_modules",
            "modules_to_save",
            "target_parameters",
            "layers_to_transform",
        ):
            value = contract[name]
            if isinstance(value, list):
                contract[name] = sorted(
                    value, key=lambda item: json.dumps(item, sort_keys=True)
                )
        unsupported = {
            name: contract[name]
            for name in (
                "use_dora",
                "use_qalora",
                "lora_bias",
                "modules_to_save",
                "trainable_token_indices",
                "target_parameters",
                "layer_replication",
                "loftq_config",
                "eva_config",
                "corda_config",
                "megatron_config",
            )
            if contract[name] not in (None, False, [], {})
        }
        if contract["bias"] != "none":
            unsupported["bias"] = contract["bias"]
        if contract["init_lora_weights"] is not True:
            unsupported["init_lora_weights"] = contract["init_lora_weights"]
        if unsupported:
            raise RuntimeError(
                "StreamVLN checkpoint codec does not cover extra PEFT state: "
                + ", ".join(sorted(unsupported))
            )
        return contract

    @classmethod
    def _peft_contract(cls, backbone: object) -> Mapping[str, object]:
        configs = getattr(backbone, "peft_config", None)
        if not isinstance(configs, Mapping) or len(configs) != 1:
            raise RuntimeError("StreamVLN checkpoint requires exactly one PEFT adapter")
        config = next(iter(configs.values()))
        if hasattr(config, "to_dict"):
            raw = config.to_dict()
        elif isinstance(config, Mapping):
            raw = config
        else:
            raise TypeError("StreamVLN PEFT config is not serializable")
        if not isinstance(raw, Mapping):
            raise TypeError("StreamVLN PEFT config must serialize to a mapping")
        return cls._peft_contract_from_mapping(raw)

    @staticmethod
    def _tensor_schema(state: Mapping[str, object]) -> Mapping[str, object]:
        return {
            str(name): {
                "shape": [int(item) for item in value.shape],
                "dtype": str(value.dtype),
            }
            for name, value in sorted(state.items())
        }

    @staticmethod
    def _canonical_lora_parameters(backbone: object) -> dict[str, object]:
        state = {}
        for name, parameter in backbone.named_parameters():
            if "lora_" not in name:
                continue
            key = name.replace("._fsdp_wrapped_module.", ".")
            key = key.replace(".lora_A.default.", ".lora_A.")
            key = key.replace(".lora_B.default.", ".lora_B.")
            if key in state:
                raise RuntimeError(f"duplicate canonical LoRA key: {key}")
            state[key] = parameter
        return state

    def _validated_lora(self) -> dict[str, object]:
        state = self._canonical_lora_parameters(self.adapter.policy.backbone)
        count = sum(int(parameter.numel()) for parameter in state.values())
        if len(state) != self.expected_lora_tensors or count != self.expected_lora_parameters:
            raise RuntimeError(
                f"unexpected LoRA checkpoint payload: {len(state)} tensors, {count} parameters"
            )
        return state

    def checkpoint_metadata(self) -> Mapping[str, object]:
        return {
            "checkpoint_kind": "streamvln_lora_value_v1",
            "actor_version": self.adapter.policy_version,
        }

    def checkpoint_compatibility(self) -> Mapping[str, object]:
        policy = self.adapter.policy
        lora = self._validated_lora()
        return {
            "checkpoint_kind": "streamvln_lora_value_v1",
            "frozen_dependencies": dict(self.frozen_dependency_ids),
            "peft_contract": self._peft_contract(policy.backbone),
            "policy_semantics": {
                "temperature": float(policy.temperature),
                "context_tokens": int(policy.context_tokens),
                "action_token_ids": [int(item) for item in policy.action_token_ids],
                "action_dim": int(policy.action_dim),
                "num_action_chunks": int(policy.num_action_chunks),
                "discrete_n_actions": int(policy.discrete_n_actions),
                "seed": int(policy.seed),
                "deterministic_eval": bool(policy.deterministic_eval),
            },
            "lora_tensor_count": self.expected_lora_tensors,
            "lora_parameter_count": self.expected_lora_parameters,
            "lora_schema": self._tensor_schema(lora),
            "value_head_schema": self._tensor_schema(policy.value_head.state_dict()),
        }

    @staticmethod
    def _read_stack_state(directory: Path) -> Mapping[str, object]:
        root = Path(directory)
        stack_state = json.loads(
            (root / "stack_state.json").read_text(encoding="utf-8")
        )
        if set(stack_state) != {"schema_version", "actor_version", "compatibility"}:
            raise ValueError("StreamVLN stack checkpoint state is incomplete")
        if stack_state["schema_version"] != 1:
            raise ValueError("unsupported StreamVLN stack checkpoint schema")
        version = stack_state["actor_version"]
        if type(version) is not int or version < 0:
            raise ValueError("checkpoint actor_version is invalid")
        if not isinstance(stack_state["compatibility"], Mapping):
            raise ValueError("StreamVLN stack compatibility is invalid")
        return stack_state

    def inspect_checkpoint_metadata(self, directory: Path) -> Mapping[str, object]:
        stack_state = self._read_stack_state(directory)
        return {
            "checkpoint_kind": "streamvln_lora_value_v1",
            "actor_version": stack_state["actor_version"],
        }

    def save_checkpoint(self, staging_directory: Path) -> Mapping[str, str]:
        import torch
        from safetensors.torch import save_file

        root = Path(staging_directory)
        adapter_dir = root / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        backbone = self.adapter.policy.backbone
        if not hasattr(backbone, "save_pretrained"):
            raise TypeError("StreamVLN backbone must expose the original PEFT saver")
        compatibility = dict(self.checkpoint_compatibility())
        backbone.save_pretrained(adapter_dir, safe_serialization=True)
        config_path = adapter_dir / "adapter_config.json"
        if not config_path.is_file():
            raise RuntimeError("PEFT saver did not write adapter_config.json")
        saved_config = json.loads(config_path.read_text(encoding="utf-8"))
        if self._peft_contract_from_mapping(saved_config) != compatibility["peft_contract"]:
            raise RuntimeError("saved PEFT config differs from the active LoRA contract")
        lora = {
            key: value.detach().cpu().contiguous()
            for key, value in self._validated_lora().items()
        }
        save_file(lora, str(adapter_dir / "adapter_model.safetensors"))
        torch.save(
            {
                key: value.detach().cpu()
                for key, value in self.adapter.policy.value_head.state_dict().items()
            },
            root / "value_head.pt",
        )
        (root / "stack_state.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "actor_version": self.adapter.policy_version,
                    "compatibility": compatibility,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return {
            path.relative_to(root).as_posix(): _sha256(path)
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }

    def load_checkpoint(self, directory: Path) -> None:
        root = Path(directory)
        stack_state = self._read_stack_state(root)
        version = stack_state["actor_version"]
        current_compatibility = dict(self.checkpoint_compatibility())
        if stack_state["compatibility"] != current_compatibility:
            raise RuntimeError("StreamVLN checkpoint compatibility differs before restore")
        saved_config = json.loads(
            (root / "adapter" / "adapter_config.json").read_text(encoding="utf-8")
        )
        if self._peft_contract_from_mapping(saved_config) != current_compatibility["peft_contract"]:
            raise RuntimeError("persisted PEFT config differs before restore")

        import torch
        from safetensors.torch import load_file

        saved = load_file(str(root / "adapter" / "adapter_model.safetensors"))
        current = self._validated_lora()
        if set(saved) != set(current):
            raise RuntimeError("persisted LoRA checkpoint keys differ from the policy")
        try:
            value_state = torch.load(root / "value_head.pt", map_location="cpu", weights_only=True)
        except TypeError:
            value_state = torch.load(root / "value_head.pt", map_location="cpu")
        current_value = self.adapter.policy.value_head.state_dict()
        if not isinstance(value_state, Mapping) or set(value_state) != set(current_value):
            raise RuntimeError("persisted value-head keys differ from the policy")
        if self._tensor_schema(saved) != current_compatibility["lora_schema"]:
            raise RuntimeError("persisted LoRA tensor schema differs from the policy")
        if self._tensor_schema(value_state) != current_compatibility["value_head_schema"]:
            raise RuntimeError("persisted value-head tensor schema differs from the policy")

        prepared_lora = {
            key: saved[key].to(device=parameter.device, dtype=parameter.dtype)
            for key, parameter in current.items()
        }
        prepared_value = {
            key: value_state[key].to(device=value.device, dtype=value.dtype)
            for key, value in current_value.items()
        }
        with torch.no_grad():
            for key, parameter in current.items():
                parameter.copy_(prepared_lora[key])
            for key, value in current_value.items():
                value.copy_(prepared_value[key])
        self.adapter.policy.actor_version = version
        if self.adapter._identity is not None:
            self.adapter.reset_state(self.adapter._identity)


# Short aliases for callers that name the underlying policies directly.
VLNPolicyAdapter = VLNPolicyStackAdapter
StreamVLNPolicyAdapter = StreamVLNPolicyStackAdapter
StreamVLNPPOPolicyAdapter = StreamVLNPolicyStackAdapter


__all__ = [
    "StreamVLNCheckpointAdapter",
    "StreamVLNPPOPolicyAdapter",
    "StreamVLNPolicyAdapter",
    "StreamVLNPolicyStackAdapter",
    "VLNPolicyAdapter",
    "VLNPolicyCheckpointAdapter",
    "VLNPolicyStackAdapter",
]
