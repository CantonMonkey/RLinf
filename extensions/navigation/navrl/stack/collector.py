"""Bounded decision collector shared by all policy/action plugins."""

from __future__ import annotations

from .contracts import BootstrapValue, DecisionBatch, DecisionRecord
from .interfaces import ActionExecutor, PolicyAdapter, TaskAdapter


class DecisionCollector:
    def __init__(self) -> None:
        self._seen_reset_tokens_by_slot: dict[int, set[str]] = {}

    def collect_episode(
        self,
        task: TaskAdapter,
        policy: PolicyAdapter,
        executor: ActionExecutor,
        *,
        max_decisions: int,
        episode: object | None = None,
        seed: int | None = None,
    ) -> DecisionBatch:
        if type(max_decisions) is not int or max_decisions < 1:
            raise ValueError("max_decisions must be a positive integer")
        reset = task.reset(episode, seed=seed)
        identity = reset.identity
        seen_tokens = self._seen_reset_tokens_by_slot.setdefault(identity.env_slot, set())
        if identity.reset_token in seen_tokens:
            raise RuntimeError("task reset reused a stale reset token")
        seen_tokens.add(identity.reset_token)
        policy.reset_state(identity)
        behavior_version = policy.policy_version
        observation = reset.observation
        physical_step = 0
        pending = []
        for decision_index in range(max_decisions):
            sample = policy.sample(
                observation,
                identity,
                decision_index,
                physical_step,
            )
            if sample.identity != identity or sample.decision_index != decision_index:
                raise RuntimeError("policy sample identity differs from collector identity")
            if sample.physical_step != physical_step:
                raise RuntimeError("policy sample physical step differs from collector state")
            if sample.policy_version != behavior_version:
                raise RuntimeError("policy sample carries the wrong behavior version")
            if policy.policy_version != behavior_version:
                raise RuntimeError("policy version changed during rollout collection")
            execution = executor.execute(task, sample)
            old_joint_log_prob = policy.behavior_joint_log_prob(sample, execution)
            pending.append((sample, execution, old_joint_log_prob))
            observation = execution.next_observation
            physical_step += execution.duration
            if execution.terminated or execution.truncated:
                break

        records = []
        for index, (sample, execution, old_joint_log_prob) in enumerate(pending):
            if execution.terminated:
                next_value = 0.0
                bootstrap_identity = None
            elif index + 1 < len(pending):
                next_value = pending[index + 1][0].value
                bootstrap_identity = identity
            else:
                bootstrap = policy.bootstrap(observation, identity, physical_step)
                if not isinstance(bootstrap, BootstrapValue):
                    raise TypeError("policy bootstrap must return BootstrapValue")
                if (
                    bootstrap.identity != identity
                    or bootstrap.physical_step != physical_step
                    or bootstrap.policy_version != behavior_version
                ):
                    raise RuntimeError("bootstrap provenance differs from the rollout boundary")
                next_value = bootstrap.value
                bootstrap_identity = identity
            records.append(
                DecisionRecord(
                    sample=sample,
                    execution=execution,
                    old_joint_log_prob=old_joint_log_prob,
                    next_value=next_value,
                    bootstrap_identity=bootstrap_identity,
                )
            )
        return DecisionBatch(tuple(records))
