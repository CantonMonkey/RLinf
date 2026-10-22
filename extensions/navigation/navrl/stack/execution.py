"""Generic primitive-prefix executor used by discrete and chunk policies."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from .contracts import ExecutionReceipt, PolicySample, PrimitiveStep
from .interfaces import ActionExecutor, TaskAdapter


class PrefixActionExecutor(ActionExecutor):
    def __init__(
        self,
        *,
        max_width: int,
        expand_action: Callable[[object], Sequence[object]],
        pad_action: object = None,
        stop_after: Callable[[PrimitiveStep], str | None] | None = None,
        credit_mode: str = "whole_decision",
    ) -> None:
        if type(max_width) is not int or max_width < 1:
            raise ValueError("max_width must be a positive integer")
        if not callable(expand_action):
            raise TypeError("expand_action must be callable")
        self.max_width = max_width
        self.expand_action = expand_action
        self.pad_action = pad_action
        self.stop_after = stop_after
        if credit_mode not in ("whole_decision", "executed_prefix"):
            raise ValueError("credit_mode must be whole_decision or executed_prefix")
        self.credit_mode = credit_mode

    def execute(self, task: TaskAdapter, sample: PolicySample) -> ExecutionReceipt:
        proposed = tuple(self.expand_action(sample.action_payload))
        if not 1 <= len(proposed) <= self.max_width:
            raise ValueError("expanded action payload is outside executor width")
        term_count = len(sample.behavior_log_prob_terms)
        if self.credit_mode == "executed_prefix" and len(proposed) != term_count:
            raise ValueError(
                "executed-prefix credit requires one log-prob term per proposed primitive"
            )
        executed: list[object] = []
        rewards: list[float] = []
        feedback = []
        final: PrimitiveStep | None = None
        end_reason = "proposal_exhausted"
        for primitive in proposed:
            row = task.step(primitive)
            if row.identity != sample.identity:
                raise RuntimeError("task reset identity changed during action execution")
            expected_step = sample.physical_step + len(executed) + 1
            if row.physical_step != expected_step:
                raise RuntimeError("task physical-step identity is not contiguous")
            executed.append(primitive)
            rewards.append(float(row.reward))
            feedback.append(dict(row.info))
            final = row
            if row.terminated:
                end_reason = "terminated"
                break
            if row.truncated:
                end_reason = "truncated"
                break
            if self.stop_after is not None:
                reason = self.stop_after(row)
                if reason is not None:
                    if not isinstance(reason, str) or not reason:
                        raise ValueError("stop_after must return None or a non-empty reason")
                    end_reason = reason
                    break
        if final is None:
            raise RuntimeError("executor produced no primitive feedback")
        duration = len(executed)
        width = self.max_width
        if self.credit_mode == "executed_prefix":
            credit_mask = (True,) * duration + (False,) * (term_count - duration)
        else:
            credit_mask = (True,) * term_count
        return ExecutionReceipt(
            identity=sample.identity,
            decision_index=sample.decision_index,
            physical_step=sample.physical_step,
            source_action_payload=sample.action_payload,
            executed_primitives=tuple(executed) + (self.pad_action,) * (width - duration),
            executed_mask=(True,) * duration + (False,) * (width - duration),
            credited_log_prob_mask=credit_mask,
            primitive_rewards=tuple(rewards) + (0.0,) * (width - duration),
            duration=duration,
            next_observation=final.observation,
            terminated=final.terminated,
            truncated=final.truncated,
            end_reason=end_reason,
            primitive_feedback=tuple(feedback),
        )
