"""Model- and simulator-neutral contracts for the RLinf-VLN training stack."""

from .checkpoint import CheckpointReceipt, load_checkpoint_verified, save_checkpoint_atomic
from .collector import DecisionCollector
from .contracts import (
    BootstrapValue,
    DecisionBatch,
    DecisionRecord,
    EpisodeMetrics,
    EpisodeIdentity,
    ExecutionReceipt,
    PolicyEvaluation,
    PolicySample,
    PrimitiveStep,
    TaskReset,
)
from .execution import PrefixActionExecutor
from .interfaces import ActionExecutor, CheckpointAdapter, PolicyAdapter, TaskAdapter
from .legacy import LegacyNavTaskAdapter
from .ppo import (
    PPOObjective,
    RatioMetrics,
    assert_pre_update_ratio,
    compute_ppo_objective,
    compute_ratio_metrics,
)
from .returns import (
    GAEOutput,
    collapse_primitive_rewards,
    compute_duration_aware_gae,
    discounted_prefix_reward,
    duration_aware_gae,
)

__all__ = [
    "ActionExecutor",
    "BootstrapValue",
    "CheckpointAdapter",
    "CheckpointReceipt",
    "DecisionBatch",
    "DecisionCollector",
    "DecisionRecord",
    "EpisodeMetrics",
    "EpisodeIdentity",
    "ExecutionReceipt",
    "GAEOutput",
    "LegacyNavTaskAdapter",
    "PolicyAdapter",
    "PolicyEvaluation",
    "PolicySample",
    "PPOObjective",
    "PrefixActionExecutor",
    "PrimitiveStep",
    "RatioMetrics",
    "TaskAdapter",
    "TaskReset",
    "collapse_primitive_rewards",
    "assert_pre_update_ratio",
    "compute_duration_aware_gae",
    "compute_ppo_objective",
    "compute_ratio_metrics",
    "discounted_prefix_reward",
    "duration_aware_gae",
    "load_checkpoint_verified",
    "save_checkpoint_atomic",
]
