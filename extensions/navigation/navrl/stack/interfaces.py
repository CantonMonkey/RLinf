"""Abstract plugin boundary for tasks, policies, executors, and checkpoints."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Mapping

from .contracts import (
    BootstrapValue,
    DecisionBatch,
    EpisodeMetrics,
    EpisodeIdentity,
    ExecutionReceipt,
    PolicyEvaluation,
    PolicySample,
    PrimitiveStep,
    TaskReset,
)


class TaskAdapter(ABC):
    @abstractmethod
    def reset(self, episode: object | None = None, *, seed: int | None = None) -> TaskReset:
        """Reset one task and issue a fresh reset identity."""

    @abstractmethod
    def step(self, primitive_action: object) -> PrimitiveStep:
        """Execute exactly one physical primitive."""

    @abstractmethod
    def get_metrics(self) -> EpisodeMetrics:
        """Return detached SR/SPL/NE, path, collision, stop, and timeout fields."""

    @abstractmethod
    def close(self) -> None:
        """Release resources; repeated calls must be safe."""


class PolicyAdapter(ABC):
    @abstractmethod
    def reset_state(self, identity: EpisodeIdentity) -> None:
        """Clear all recurrent/history state and bind it to one reset identity."""

    @abstractmethod
    def sample(
        self,
        observation: object,
        identity: EpisodeIdentity,
        decision_index: int,
        physical_step: int,
    ) -> PolicySample:
        """Sample one high-level action under the behavior policy."""

    @abstractmethod
    def behavior_joint_log_prob(
        self,
        sample: PolicySample,
        execution: ExecutionReceipt,
    ) -> object:
        """Sum only behavior log-prob terms credited by the execution receipt."""

    @abstractmethod
    def evaluate(self, batch: DecisionBatch) -> tuple[PolicyEvaluation, ...]:
        """Recompute current joint log-probability/value for replay rows."""

    @abstractmethod
    def bootstrap(
        self,
        observation: object,
        identity: EpisodeIdentity,
        physical_step: int,
    ) -> BootstrapValue:
        """Evaluate V(next observation) without changing rollout history."""

    @abstractmethod
    def trainable_parameter_names(self) -> tuple[str, ...]:
        """Return the exact optimizer-owned parameter names."""

    @property
    @abstractmethod
    def policy_version(self) -> int:
        """Monotonic behavior-policy version."""


class ActionExecutor(ABC):
    @abstractmethod
    def execute(self, task: TaskAdapter, sample: PolicySample) -> ExecutionReceipt:
        """Execute the sampled payload and report the exact credited prefix."""


class CheckpointAdapter(ABC):
    @abstractmethod
    def checkpoint_compatibility(self) -> Mapping[str, object]:
        """Return stable, JSON-serializable pre-load compatibility metadata.

        This contract must describe every static dependency that can change the
        restored policy's meaning (for example architecture, frozen model
        identities, adapter configuration, and action semantics).  It must not
        include mutable runtime state such as the current policy version.
        """

    @abstractmethod
    def checkpoint_metadata(self) -> Mapping[str, object]:
        """Return source, version, and trainable-scope metadata."""

    @abstractmethod
    def inspect_checkpoint_metadata(self, directory: Path) -> Mapping[str, object]:
        """Read prospective restored metadata without mutating policy state."""

    @abstractmethod
    def save_checkpoint(self, staging_directory: Path) -> Mapping[str, str]:
        """Write model-specific payloads and return relative path -> SHA-256."""

    @abstractmethod
    def load_checkpoint(self, directory: Path) -> None:
        """Strictly restore a checkpoint payload."""
