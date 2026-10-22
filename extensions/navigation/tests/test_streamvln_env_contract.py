"""Dependency-free checks for StreamVLN four-action environment semantics."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


class Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class Stacked:
    def __init__(self, values):
        self.values = [value.item() for value in values]


class FakeArray:
    def __init__(self, values):
        self.values = values
        self.shape = (1, 4, 1)

    def reshape(self, *_shape):
        return [value for row in self.values for chunk in row for value in chunk]


class FakeVLNEnv:
    def step(self, actions):
        action = int(actions[0][0])
        self.executed.append(action)
        terminated = action == self.terminate_on
        return (
            {"frame": len(self.executed)},
            Scalar(float(action)),
            Scalar(terminated),
            Scalar(False),
            {
                "count": len(self.executed),
                "collision": Scalar(action == 1),
                "path_length": Scalar(self.path_length),
            },
        )


def load_module():
    gym = types.ModuleType("gymnasium")
    gym.spaces = types.SimpleNamespace()
    numpy = types.ModuleType("numpy")
    numpy.asarray = lambda values: FakeArray(values)
    numpy.uint8 = "uint8"
    numpy.int64 = "int64"
    numpy.iinfo = lambda _dtype: types.SimpleNamespace(max=2**63 - 1)
    torch = types.ModuleType("torch")
    torch.bool = "bool"
    torch.uint8 = "uint8"
    torch.int64 = "int64"
    torch.zeros_like = lambda _value: Scalar(0)
    torch.stack = lambda values, dim: Stacked(values)
    torch.tensor = lambda *args, **kwargs: None
    vln = types.ModuleType("vln_env")
    vln.VLNEnv = FakeVLNEnv
    path = Path(__file__).resolve().parents[1] / "integrations/rlinf_v03/streamvln_env.py"
    contract_path = path.with_name("streamvln_executor_contract.py")
    contract_spec = importlib.util.spec_from_file_location(
        "streamvln_executor_contract", contract_path
    )
    contract = importlib.util.module_from_spec(contract_spec)
    with patch.dict(sys.modules, {"streamvln_executor_contract": contract}):
        contract_spec.loader.exec_module(contract)
    spec = importlib.util.spec_from_file_location("staged_streamvln_env_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {
            "gymnasium": gym,
            "numpy": numpy,
            "torch": torch,
            "vln_env": vln,
            "streamvln_executor_contract": contract,
        },
    ):
        spec.loader.exec_module(module)
    return module


class StreamVLNEnvContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def make_env(self, terminate_on=0):
        env = object.__new__(self.module.StreamVLNEnv)
        env.executed = []
        env.terminate_on = terminate_on
        env.path_length = 0.0
        env._previous_actions = []
        env._previous_path_length = 0.0
        env._terminal_frozen = False
        env._stream_obs = lambda base, episode_start: {
            "base": base,
            "episode_start": episode_start,
        }
        return env

    def test_utf8_and_action_padding_are_prefix_contiguous(self):
        values, mask = self.module.encode_fixed_utf8("向左走", 16)
        self.assertEqual(bytes(value for value, valid in zip(values, mask) if valid).decode(), "向左走")
        self.assertFalse(any(mask[9:]))
        actions, action_mask = self.module._fixed_actions([1, 2, 0])
        self.assertEqual(actions, [1, 2, 0, 0])
        self.assertEqual(action_mask, [True, True, True, False])
        with self.assertRaisesRegex(ValueError, "limit"):
            self.module.encode_fixed_utf8("abcd", 3)

    def test_stop_executes_prefix_and_pads_without_extra_physics(self):
        env = self.make_env(terminate_on=0)
        obs, rewards, terminations, truncations, infos = env.chunk_step(
            [[[1], [2], [0], [0]]]
        )
        self.assertEqual(env.executed, [1, 2, 0])
        self.assertEqual(env._previous_actions, [1, 2, 0])
        self.assertTrue(env._terminal_frozen)
        self.assertEqual(rewards.values, [1.0, 2.0, 0.0, 0])
        self.assertEqual(terminations.values, [False, False, True, 0])
        self.assertEqual(truncations.values, [False, False, False, 0])
        self.assertEqual(obs[0]["base"], {"frame": 3})
        self.assertEqual(infos[0]["count"], 3)
        self.assertEqual(infos[0]["chunk_collisions"], [True, False, False])
        feedback = infos[0]["executor_feedback"]
        self.assertEqual(feedback["executed_actions"], [1, 2, 0])
        self.assertEqual(feedback["blocked_forward_count"], 1)
        self.assertEqual(feedback["environment_reward_sum"], 3.0)
        self.assertEqual(feedback["end_reason"], "stop")

    def test_explicit_executed_prefix_preserves_padding_and_history_mask(self):
        env = self.make_env(terminate_on=0)
        obs, rewards, terminations, truncations, infos = env.chunk_step(
            [[[2], [1], [1], [1]]], execute_count=2
        )
        self.assertEqual(env.executed, [2, 1])
        self.assertEqual(env._previous_actions, [2, 1])
        self.assertFalse(env._terminal_frozen)
        self.assertEqual(rewards.values, [2.0, 1.0, 0, 0])
        self.assertEqual(terminations.values, [False, False, 0, 0])
        self.assertEqual(infos[0]["chunk_collisions"], [False, True])
        actions, mask = self.module._fixed_actions(env._previous_actions)
        self.assertEqual(actions, [2, 1, 0, 0])
        self.assertEqual(mask, [True, True, False, False])
        with self.assertRaisesRegex(ValueError, "execute_count"):
            env.chunk_step([[[1], [1], [1], [1]]], execute_count=0)

    def test_unannounced_mid_chunk_terminal_is_rejected(self):
        env = self.make_env(terminate_on=2)
        with self.assertRaisesRegex(RuntimeError, "executed-action prefix"):
            env.chunk_step([[[1], [2], [3], [1]]])


if __name__ == "__main__":
    unittest.main()
