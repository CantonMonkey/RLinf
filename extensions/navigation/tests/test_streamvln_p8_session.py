"""Behavior-lock tests for the policy-preserving P8 session primitives."""

import ast
from pathlib import Path
import unittest
from unittest import mock

import torch


POLICY_PATH = (
    Path(__file__).resolve().parents[1]
    / "integrations"
    / "rlinf_v03"
    / "streamvln_ppo_policy.py"
)


def _load_session_contract():
    """Load the real session and frozen reference without optional RLinf stacks."""

    source = POLICY_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(POLICY_PATH))
    session = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "OriginalStreamSession"
    )
    reference = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "sample_pre_refactor_release_reference"
    )
    module = ast.Module(body=[session, reference], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "torch": torch,
        "np": None,
        "ACTION_NAMES": ("STOP", "↑", "←", "→"),
        "ACTION_CHUNK": 4,
    }
    exec(compile(module, str(POLICY_PATH), "exec"), namespace)
    return (
        namespace["OriginalStreamSession"],
        namespace["sample_pre_refactor_release_reference"],
    )


OriginalStreamSession, sample_pre_refactor_release_reference = _load_session_contract()


def _forced_logits(action: int) -> torch.Tensor:
    logits = torch.full((4,), -1_000.0, dtype=torch.float32)
    logits[action] = 0.0
    return logits


class _SyntheticSession(OriginalStreamSession):
    """Small deterministic shell around the real session lifecycle methods."""

    def __init__(self, logits):
        self.logits = [torch.as_tensor(row, dtype=torch.float32) for row in logits]
        self.action_ids = torch.tensor([101, 102, 103, 104], dtype=torch.int64)
        self.parts = [torch.zeros((1, 1), dtype=torch.float32)]
        self.temperature = 1.0
        self.prepared = []
        self.committed = []
        self.closed = 0
        self.logit_calls = 0
        self.lifecycle = []

    def prepare_observation(self, step, feature):
        self.prepared.append((int(step), feature.detach().clone()))

    def _hidden(self, embeddings):
        return torch.tensor([[float(self.logit_calls)]], dtype=torch.float32)

    def _action_logits(self, hidden):
        logits = self.logits[self.logit_calls].clone()
        self.logit_calls += 1
        return logits

    def append_action_token(self, token):
        self.committed.append(int(token))
        self.parts.append(torch.tensor([[float(token)]], dtype=torch.float32))

    def close_decision(self):
        self.closed += 1

    def open_decision(self, step, feature):
        self.lifecycle.append("open")
        return super().open_decision(step, feature)

    def next_token(self, generator, *, greedy=False, residual=None):
        self.lifecycle.append("next")
        return super().next_token(
            generator, greedy=greedy, residual=residual
        )

    def commit_token(self, token):
        self.lifecycle.append("commit")
        return super().commit_token(token)

    def finish_decision(self):
        self.lifecycle.append("finish")
        return super().finish_decision()


@torch.no_grad()
def _legacy_sample(session, step, feature, generator, *, greedy=False):
    """Exact pre-refactor loop retained only as the behavior oracle."""

    session.prepare_observation(step, feature)
    tokens, actions, log_probs = [], [], []
    for _ in range(4):
        hidden = session._hidden(torch.cat(session.parts))[-1]
        categorical = session._action_logits(hidden).log_softmax(-1)
        action = (
            int(categorical.argmax())
            if greedy
            else int(torch.multinomial(categorical.exp(), 1, generator=generator))
        )
        token = int(session.action_ids[action])
        tokens.append(token)
        actions.append(action)
        log_probs.append(float(categorical[action]))
        session.append_action_token(token)
        if action == 0:
            break
    session.close_decision()
    return {
        "step": step,
        "sampled_tokens": tokens,
        "proposed_actions": actions,
        "behavior_log_probs": log_probs,
        "temperature": session.temperature,
        "greedy": bool(greedy),
    }


def _run_with_multinomial_trace(call):
    original = torch.multinomial
    probabilities = []

    def recorded(input, num_samples, replacement=False, *, generator=None, out=None):
        probabilities.append(input.detach().clone())
        return original(
            input,
            num_samples,
            replacement=replacement,
            generator=generator,
        )

    with mock.patch.object(torch, "multinomial", side_effect=recorded):
        result = call()
    return result, probabilities


class StreamVLNP8SessionTests(unittest.TestCase):
    def test_frozen_no_adapter_path_keeps_the_original_backbone(self):
        tree = ast.parse(POLICY_PATH.read_text(encoding="utf-8"))
        constructor = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "__init__"
            and any(arg.arg == "adapter_checkpoint" for arg in node.args.kwonlyargs)
        )
        rendered = ast.unparse(constructor)
        self.assertIn("elif not trainable", rendered)
        self.assertIn("self.backbone = base", rendered)
        self.assertIn("self.source_kind = 'upstream-original'", rendered)

    def test_refactor_matches_legacy_rng_and_multinomial_order(self):
        logits = [
            torch.tensor([-1_000.0, 0.3, -0.2, 0.1]),
            torch.tensor([-1_000.0, -0.4, 0.6, 0.2]),
            torch.tensor([-1_000.0, 0.0, 0.2, 0.8]),
            torch.tensor([-1_000.0, 0.7, 0.1, -0.3]),
        ]
        feature = torch.tensor([7.0])
        legacy_session = _SyntheticSession(logits)
        refactored_session = _SyntheticSession(logits)
        legacy_generator = torch.Generator().manual_seed(20260928)
        refactored_generator = torch.Generator().manual_seed(20260928)

        legacy, legacy_trace = _run_with_multinomial_trace(
            lambda: sample_pre_refactor_release_reference(
                legacy_session, 0, feature, legacy_generator, greedy=False
            )
        )
        refactored, refactored_trace = _run_with_multinomial_trace(
            lambda: refactored_session.sample(
                0, feature, refactored_generator, greedy=False
            )
        )

        for key in (
            "step",
            "sampled_tokens",
            "proposed_actions",
            "behavior_log_probs",
            "temperature",
            "greedy",
        ):
            self.assertEqual(refactored[key], legacy[key])
        self.assertEqual(legacy["reference_commit"], "5434ab0")
        self.assertEqual(len(legacy["base_logits"]), len(legacy["sampled_tokens"]))
        self.assertEqual(refactored_session.committed, legacy_session.committed)
        self.assertTrue(
            torch.equal(refactored_generator.get_state(), legacy_generator.get_state())
        )
        self.assertEqual(len(refactored_trace), len(legacy_trace))
        for actual, expected in zip(refactored_trace, legacy_trace):
            self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(
            refactored_session.lifecycle,
            ["open", "next", "commit", "next", "commit", "next", "commit", "next", "commit", "finish"],
        )

    def test_stop_is_committed_at_each_token_position(self):
        feature = torch.tensor([3.0])
        for stop_position in range(1, 5):
            with self.subTest(stop_position=stop_position):
                actions = [2] * (stop_position - 1) + [0]
                session = _SyntheticSession([_forced_logits(action) for action in actions])
                sample = session.sample(
                    0,
                    feature,
                    torch.Generator().manual_seed(100 + stop_position),
                )

                self.assertEqual(sample["proposed_actions"], actions)
                self.assertEqual(
                    sample["sampled_tokens"],
                    [int(session.action_ids[action]) for action in actions],
                )
                self.assertEqual(session.committed, sample["sampled_tokens"])
                self.assertEqual(session.logit_calls, stop_position)
                self.assertEqual(session.closed, 1)
                self.assertEqual(session.lifecycle[0], "open")
                self.assertEqual(session.lifecycle[-1], "finish")
                self.assertEqual(session.lifecycle.count("next"), stop_position)
                self.assertEqual(session.lifecycle.count("commit"), stop_position)

    def test_full_chunks_preserve_pure_left_and_right_turns(self):
        for turn_action in (2, 3):
            with self.subTest(turn_action=turn_action):
                session = _SyntheticSession([_forced_logits(turn_action)] * 4)
                sample = session.sample(
                    0,
                    torch.tensor([5.0]),
                    torch.Generator().manual_seed(220 + turn_action),
                )

                self.assertEqual(sample["proposed_actions"], [turn_action] * 4)
                self.assertEqual(
                    sample["sampled_tokens"],
                    [int(session.action_ids[turn_action])] * 4,
                )
                self.assertNotIn(0, sample["proposed_actions"])
                self.assertEqual(session.logit_calls, 4)
                self.assertEqual(session.closed, 1)

    def test_zero_residual_matches_default_and_exposes_fp32_cache(self):
        logits = [torch.tensor([-0.7, 0.2, 0.6, -0.1])]
        default_session = _SyntheticSession(logits)
        residual_session = _SyntheticSession(logits)
        feature = torch.tensor([1.0])
        default_session.open_decision(0, feature)
        residual_session.open_decision(0, feature)
        default_generator = torch.Generator().manual_seed(17)
        residual_generator = torch.Generator().manual_seed(17)

        default = default_session.next_token(default_generator)
        residual = residual_session.next_token(
            residual_generator,
            residual=lambda hidden, base: torch.zeros_like(base),
        )

        for key in ("token", "action", "log_prob"):
            self.assertEqual(residual[key], default[key])
        self.assertTrue(torch.equal(residual["base_logits"], default["base_logits"]))
        self.assertTrue(torch.equal(residual["token_hidden"], default["token_hidden"]))
        self.assertEqual(residual["base_logits"].dtype, torch.float32)
        self.assertEqual(residual["token_hidden"].dtype, torch.float32)
        self.assertFalse(residual["base_logits"].requires_grad)
        self.assertFalse(residual["token_hidden"].requires_grad)
        self.assertTrue(
            torch.equal(residual_generator.get_state(), default_generator.get_state())
        )


if __name__ == "__main__":
    unittest.main()
