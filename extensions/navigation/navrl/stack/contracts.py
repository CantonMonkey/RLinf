"""Typed identities and replay rows shared by every VLN policy plugin.

Tensor-like scalar values stay opaque so an integration can preserve gradients.
The dependency-free core validates ordinary Python numbers and all causal
identities, masks, durations, rewards, and termination boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Real
from typing import Mapping


def _finite_if_real(value: object, name: str) -> None:
    if isinstance(value, bool):
        raise TypeError(f"{name} must not be boolean")
    if isinstance(value, Real) and not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")


def _validate_index(value: int, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _prefix_length(mask: tuple[bool, ...]) -> int:
    if not mask or any(type(item) is not bool for item in mask):
        raise ValueError("executed_mask must be a non-empty tuple of bools")
    duration = 0
    padding = False
    for valid in mask:
        if valid:
            if padding:
                raise ValueError("executed_mask must be prefix-contiguous")
            duration += 1
        else:
            padding = True
    if duration < 1:
        raise ValueError("a decision must execute at least one primitive")
    return duration


@dataclass(frozen=True)
class EpisodeIdentity:
    episode_id: str
    reset_token: str
    env_slot: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.episode_id, str) or not self.episode_id:
            raise ValueError("episode_id must be a non-empty string")
        if not isinstance(self.reset_token, str) or not self.reset_token:
            raise ValueError("reset_token must be a non-empty string")
        _validate_index(self.env_slot, "env_slot")


@dataclass(frozen=True)
class TaskReset:
    identity: EpisodeIdentity
    observation: object

    def __post_init__(self) -> None:
        if not isinstance(self.identity, EpisodeIdentity):
            raise TypeError("TaskReset.identity must be EpisodeIdentity")


@dataclass(frozen=True)
class PrimitiveStep:
    identity: EpisodeIdentity
    physical_step: int
    observation: object
    reward: float
    terminated: bool = False
    truncated: bool = False
    info: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.identity, EpisodeIdentity):
            raise TypeError("PrimitiveStep.identity must be EpisodeIdentity")
        if type(self.physical_step) is not int or self.physical_step < 1:
            raise ValueError("physical_step must be a positive integer")
        _finite_if_real(self.reward, "reward")
        if type(self.terminated) is not bool or type(self.truncated) is not bool:
            raise TypeError("termination flags must be bools")
        if self.terminated and self.truncated:
            raise ValueError("a primitive cannot terminate and truncate")
        if not isinstance(self.info, Mapping):
            raise TypeError("PrimitiveStep.info must be a mapping")


@dataclass(frozen=True)
class EpisodeMetrics:
    """Task-owned navigation outcome with a stable cross-plugin core."""

    identity: EpisodeIdentity
    path_length_m: float
    collisions: int
    stop_called: bool
    timed_out: bool
    action_counts: Mapping[str, int]
    success: bool | None = None
    spl: float | None = None
    navigation_error_m: float | None = None
    extras: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.identity, EpisodeIdentity):
            raise TypeError("EpisodeMetrics.identity must be EpisodeIdentity")
        _finite_if_real(self.path_length_m, "path_length_m")
        if float(self.path_length_m) < 0.0:
            raise ValueError("path_length_m must be non-negative")
        _validate_index(self.collisions, "collisions")
        if type(self.stop_called) is not bool or type(self.timed_out) is not bool:
            raise TypeError("stop_called and timed_out must be bools")
        if self.success is not None and type(self.success) is not bool:
            raise TypeError("success must be bool or None")
        if self.spl is not None:
            _finite_if_real(self.spl, "spl")
            if not 0.0 <= float(self.spl) <= 1.0:
                raise ValueError("spl must be in [0, 1]")
        if self.navigation_error_m is not None:
            _finite_if_real(self.navigation_error_m, "navigation_error_m")
            if float(self.navigation_error_m) < 0.0:
                raise ValueError("navigation_error_m must be non-negative")
        if not isinstance(self.action_counts, Mapping):
            raise TypeError("action_counts must be a mapping")
        for name, count in self.action_counts.items():
            if not isinstance(name, str) or not name:
                raise ValueError("action count names must be non-empty strings")
            _validate_index(count, f"action_counts[{name!r}]")
        if not isinstance(self.extras, Mapping):
            raise TypeError("EpisodeMetrics.extras must be a mapping")


@dataclass(frozen=True)
class PolicySample:
    identity: EpisodeIdentity
    decision_index: int
    physical_step: int
    action_payload: tuple[object, ...]
    replay_context: Mapping[str, object]
    behavior_log_prob_terms: tuple[object, ...]
    value: object
    policy_version: int
    action_distribution: str

    def __post_init__(self) -> None:
        if not isinstance(self.identity, EpisodeIdentity):
            raise TypeError("PolicySample.identity must be EpisodeIdentity")
        _validate_index(self.decision_index, "decision_index")
        _validate_index(self.physical_step, "physical_step")
        _validate_index(self.policy_version, "policy_version")
        if not isinstance(self.action_payload, tuple) or not self.action_payload:
            raise ValueError("action_payload must be a non-empty canonical tuple")
        if not isinstance(self.replay_context, Mapping) or not self.replay_context:
            raise ValueError("replay_context must be a non-empty mapping")
        if not isinstance(self.behavior_log_prob_terms, tuple) or not self.behavior_log_prob_terms:
            raise ValueError("behavior_log_prob_terms must be a non-empty tuple")
        for term in self.behavior_log_prob_terms:
            _finite_if_real(term, "behavior log-prob term")
        if not isinstance(self.action_distribution, str) or not self.action_distribution:
            raise ValueError("action_distribution must be a non-empty string")
        _finite_if_real(self.value, "value")


@dataclass(frozen=True)
class PolicyEvaluation:
    identity: EpisodeIdentity
    decision_index: int
    joint_log_prob: object
    value: object
    policy_version: int
    entropy: object | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.identity, EpisodeIdentity):
            raise TypeError("PolicyEvaluation.identity must be EpisodeIdentity")
        _validate_index(self.decision_index, "decision_index")
        _validate_index(self.policy_version, "policy_version")
        _finite_if_real(self.joint_log_prob, "joint_log_prob")
        _finite_if_real(self.value, "value")
        if self.entropy is not None:
            _finite_if_real(self.entropy, "entropy")


@dataclass(frozen=True)
class BootstrapValue:
    """A value estimate bound to the exact reset, step, and actor version."""

    identity: EpisodeIdentity
    physical_step: int
    policy_version: int
    value: object

    def __post_init__(self) -> None:
        if not isinstance(self.identity, EpisodeIdentity):
            raise TypeError("BootstrapValue.identity must be EpisodeIdentity")
        _validate_index(self.physical_step, "physical_step")
        _validate_index(self.policy_version, "policy_version")
        _finite_if_real(self.value, "bootstrap value")


@dataclass(frozen=True)
class ExecutionReceipt:
    identity: EpisodeIdentity
    decision_index: int
    physical_step: int
    source_action_payload: tuple[object, ...]
    executed_primitives: tuple[object, ...]
    executed_mask: tuple[bool, ...]
    credited_log_prob_mask: tuple[bool, ...]
    primitive_rewards: tuple[float, ...]
    duration: int
    next_observation: object
    terminated: bool
    truncated: bool
    end_reason: str
    primitive_feedback: tuple[Mapping[str, object], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.identity, EpisodeIdentity):
            raise TypeError("ExecutionReceipt.identity must be EpisodeIdentity")
        _validate_index(self.decision_index, "decision_index")
        _validate_index(self.physical_step, "physical_step")
        if not isinstance(self.source_action_payload, tuple) or not self.source_action_payload:
            raise ValueError("source_action_payload must be a non-empty canonical tuple")
        if not isinstance(self.executed_primitives, tuple):
            raise TypeError("executed_primitives must be a fixed-width tuple")
        if len(self.executed_primitives) != len(self.executed_mask):
            raise ValueError("executed primitives and mask widths differ")
        if len(self.primitive_rewards) != len(self.executed_mask):
            raise ValueError("primitive rewards and mask widths differ")
        actual_duration = _prefix_length(self.executed_mask)
        _prefix_length(self.credited_log_prob_mask)
        if type(self.duration) is not int or self.duration != actual_duration:
            raise ValueError("duration must equal the executed-mask prefix length")
        for reward in self.primitive_rewards:
            _finite_if_real(reward, "primitive reward")
        if any(float(value) != 0.0 for value in self.primitive_rewards[self.duration :]):
            raise ValueError("padded primitive rewards must be zero")
        if type(self.terminated) is not bool or type(self.truncated) is not bool:
            raise TypeError("termination flags must be bools")
        if self.terminated and self.truncated:
            raise ValueError("a decision cannot terminate and truncate")
        if not isinstance(self.end_reason, str) or not self.end_reason:
            raise ValueError("end_reason must be a non-empty string")
        if len(self.primitive_feedback) != self.duration:
            raise ValueError("primitive_feedback must contain one mapping per executed primitive")
        if any(not isinstance(item, Mapping) for item in self.primitive_feedback):
            raise TypeError("primitive_feedback rows must be mappings")

    @property
    def active_primitives(self) -> tuple[object, ...]:
        return self.executed_primitives[: self.duration]


@dataclass(frozen=True)
class DecisionRecord:
    sample: PolicySample
    execution: ExecutionReceipt
    old_joint_log_prob: object
    next_value: object
    bootstrap_identity: EpisodeIdentity | None

    def __post_init__(self) -> None:
        if not isinstance(self.sample, PolicySample) or not isinstance(
            self.execution, ExecutionReceipt
        ):
            raise TypeError("DecisionRecord requires typed sample and execution records")
        if self.sample.identity != self.execution.identity:
            raise ValueError("sample and execution episode identities differ")
        if self.sample.decision_index != self.execution.decision_index:
            raise ValueError("sample and execution decision indices differ")
        if self.sample.physical_step != self.execution.physical_step:
            raise ValueError("sample and execution physical steps differ")
        if self.sample.action_payload != self.execution.source_action_payload:
            raise ValueError("execution receipt belongs to another sampled action payload")
        if len(self.execution.credited_log_prob_mask) != len(
            self.sample.behavior_log_prob_terms
        ):
            raise ValueError("credit mask must align with behavior log-prob terms")
        _finite_if_real(self.old_joint_log_prob, "old_joint_log_prob")
        _finite_if_real(self.next_value, "next_value")
        if self.execution.terminated:
            if isinstance(self.next_value, Real) and float(self.next_value) != 0.0:
                raise ValueError("terminated decisions require zero bootstrap value")
            if self.bootstrap_identity is not None:
                raise ValueError("terminated decisions cannot have a bootstrap identity")
        elif self.bootstrap_identity != self.sample.identity:
            raise ValueError("live/truncated decisions must bootstrap within the same episode reset")


@dataclass(frozen=True)
class DecisionBatch:
    records: tuple[DecisionRecord, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported DecisionBatch schema")
        if not isinstance(self.records, tuple) or not self.records:
            raise ValueError("DecisionBatch must contain at least one record")
        if any(not isinstance(row, DecisionRecord) for row in self.records):
            raise TypeError("DecisionBatch records must be DecisionRecord objects")
        identity = self.records[0].sample.identity
        policy_version = self.records[0].sample.policy_version
        physical_step = self.records[0].sample.physical_step
        for expected_index, row in enumerate(self.records):
            if row.sample.identity != identity:
                raise ValueError("one DecisionBatch cannot cross an episode reset")
            if row.sample.policy_version != policy_version:
                raise ValueError("a DecisionBatch cannot mix behavior policy versions")
            if row.sample.decision_index != expected_index:
                raise ValueError("decision indices must be contiguous from zero")
            if row.sample.physical_step != physical_step:
                raise ValueError("physical-step identities are not contiguous")
            physical_step += row.execution.duration
            if expected_index + 1 < len(self.records) and (
                row.execution.terminated or row.execution.truncated
            ):
                raise ValueError("terminal/truncated decision must end the batch")

    @property
    def identity(self) -> EpisodeIdentity:
        return self.records[0].sample.identity

    @property
    def policy_version(self) -> int:
        versions = {row.sample.policy_version for row in self.records}
        if len(versions) != 1:
            raise ValueError("a DecisionBatch cannot mix behavior policy versions")
        return versions.pop()
