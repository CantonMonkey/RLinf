"""Focused CPU contract tests for the staged R2R policy."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from enum import Enum
from pathlib import Path
from unittest.mock import patch

try:
    import torch
except ImportError:  # The lightweight Windows project env has no PyTorch.
    torch = None


def load_policy_module():
    path = Path(__file__).resolve().parents[1] / "integrations/rlinf_v03/vln_policy.py"
    spec = importlib.util.spec_from_file_location("staged_vln_policy_test", path)
    module = importlib.util.module_from_spec(spec)
    try:
        from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType  # noqa: F401
    except ImportError:
        class ForwardType(Enum):
            DEFAULT = "default"
            SFT = "sft"

        class BasePolicy:
            pass

        packages = {
            name: types.ModuleType(name)
            for name in (
                "rlinf",
                "rlinf.models",
                "rlinf.models.embodiment",
                "rlinf.models.embodiment.base_policy",
            )
        }
        packages["rlinf.models.embodiment.base_policy"].BasePolicy = BasePolicy
        packages["rlinf.models.embodiment.base_policy"].ForwardType = ForwardType
        with patch.dict(sys.modules, packages):
            spec.loader.exec_module(module)
    else:
        spec.loader.exec_module(module)
    return module


@unittest.skipUnless(torch is not None, "PyTorch is required for R2R policy CPU tests")
class VLNPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_policy_module()

    def setUp(self):
        torch.manual_seed(13)
        self.policy = self.module.VLNPolicy(vocab_size=2504)
        self.obs = {
            "rgb": torch.zeros(2, 64, 64, 3, dtype=torch.uint8),
            "instruction_ids": torch.zeros(2, 200, dtype=torch.int64),
            "instruction_mask": torch.zeros(2, 200, dtype=torch.bool),
        }
        self.obs["rgb"][0, 12:40, :, 0] = 200
        self.obs["rgb"][1, :, 12:40, 1] = 170
        self.obs["instruction_ids"][:, :3] = torch.tensor([[4, 5, 6], [4, 5, 7]])
        self.obs["instruction_mask"][:, :3] = True

    def test_categorical_ppo_replay_gradient_and_checkpoint(self):
        actions, sampled = self.policy.predict_action_batch(self.obs, mode="train")
        self.assertEqual((2, 1, 1), tuple(actions.shape))
        self.assertEqual(torch.int64, actions.dtype)
        self.assertEqual((2, 1), tuple(sampled["prev_logprobs"].shape))
        self.assertEqual((2, 1), tuple(sampled["prev_values"].shape))
        self.assertEqual(
            {"rgb", "instruction_ids", "instruction_mask", "action", "model_action"},
            set(sampled["forward_inputs"]),
        )
        self.assertEqual(torch.uint8, sampled["forward_inputs"]["rgb"].dtype)
        out = self.policy(
            forward_inputs=sampled["forward_inputs"],
            compute_logprobs=True,
            compute_entropy=True,
            compute_values=True,
        )
        for name in ("logprobs", "entropy", "values"):
            self.assertEqual((2, 1), tuple(out[name].shape))
            self.assertTrue(bool(torch.isfinite(out[name]).all()))
        self.assertTrue(
            torch.allclose(out["logprobs"], sampled["prev_logprobs"], atol=1e-6)
        )
        loss = -(out["logprobs"] * torch.tensor([[1.0], [-1.0]])).mean()
        loss = loss + (out["values"] - torch.tensor([[0.5], [-0.5]])).square().mean()
        loss.backward()
        for parameter in self.policy.parameters():
            if parameter.grad is not None:
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
        self.assertGreater(float(self.policy.visual_encoder[0].weight.grad.abs().sum()), 0)
        self.assertGreater(float(self.policy.token_embedding.weight.grad.abs().sum()), 0)

        restored = self.module.VLNPolicy(vocab_size=2504)
        restored.load_state_dict(self.policy.state_dict(), strict=True)
        logits_a, value_a = self.policy.logits_and_values(self.obs)
        logits_b, value_b = restored.logits_and_values(self.obs)
        self.assertTrue(torch.equal(logits_a, logits_b))
        self.assertTrue(torch.equal(value_a, value_b))

    def test_stack_operations_match_legacy_policy_paths(self):
        direct_actions, direct_sample = self.policy.predict_action_batch(
            self.obs, mode="eval"
        )
        root_actions, root_sample = self.policy(
            operation="sample", env_obs=self.obs, mode="eval"
        )
        self.assertTrue(torch.equal(direct_actions, root_actions))
        self.assertTrue(
            torch.equal(
                direct_sample["prev_logprobs"], root_sample["prev_logprobs"]
            )
        )
        self.assertTrue(
            torch.equal(direct_sample["prev_values"], root_sample["prev_values"])
        )

        legacy_replay = self.policy(
            forward_inputs=root_sample["forward_inputs"],
            compute_logprobs=True,
            compute_entropy=True,
            compute_values=True,
        )
        root_replay = self.policy(
            operation="replay",
            forward_inputs=root_sample["forward_inputs"],
            compute_logprobs=True,
            compute_entropy=True,
            compute_values=True,
        )
        self.assertEqual(set(legacy_replay), set(root_replay))
        for name in legacy_replay:
            self.assertTrue(torch.equal(legacy_replay[name], root_replay[name]), name)

        prepared = self.policy.preprocess_env_obs(self.obs)
        _, direct_values = self.policy.logits_and_values(prepared)
        root_values = self.policy(operation="bootstrap", env_obs=self.obs)
        self.assertTrue(torch.equal(direct_values, root_values))

        with self.assertRaisesRegex(ValueError, "unknown VLN policy operation"):
            self.policy(operation="not-an-operation")

    def test_both_modalities_change_outputs_and_padding_does_not(self):
        base = self.policy.preprocess_env_obs(self.obs)
        base_logits, base_values = self.policy.logits_and_values(base)

        changed_rgb = {key: value.clone() for key, value in self.obs.items()}
        changed_rgb["rgb"][0].fill_(255)
        rgb_logits, rgb_values = self.policy.logits_and_values(changed_rgb)
        self.assertGreater(float((base_logits[0] - rgb_logits[0]).abs().max()), 1e-6)
        self.assertGreater(float((base_values[0] - rgb_values[0]).abs().max()), 1e-6)

        changed_text = {key: value.clone() for key, value in self.obs.items()}
        changed_text["instruction_ids"][0, 2] = 81
        text_logits, text_values = self.policy.logits_and_values(changed_text)
        self.assertGreater(float((base_logits[0] - text_logits[0]).abs().max()), 1e-6)
        self.assertGreater(float((base_values[0] - text_values[0]).abs().max()), 1e-6)

        changed_padding = {key: value.clone() for key, value in self.obs.items()}
        changed_padding["instruction_ids"][0, 80] = 82
        pad_logits, pad_values = self.policy.logits_and_values(changed_padding)
        self.assertTrue(torch.equal(base_logits, pad_logits))
        self.assertTrue(torch.equal(base_values, pad_values))

    def test_eval_is_argmax_and_privileged_inputs_are_rejected(self):
        first, _ = self.policy.predict_action_batch(self.obs, mode="eval")
        second, _ = self.policy.predict_action_batch(self.obs, mode="eval")
        self.assertTrue(torch.equal(first, second))
        logits, _ = self.policy.logits_and_values(self.obs)
        self.assertTrue(torch.equal(first[:, 0, 0], logits.argmax(dim=-1)))
        for bad_key in ("pointgoal", "goal", "pose", "reference_path", "depth"):
            with self.subTest(bad_key=bad_key):
                bad = dict(self.obs)
                bad[bad_key] = torch.zeros(2, 1)
                with self.assertRaisesRegex(ValueError, "unexpected"):
                    self.policy.predict_action_batch(bad)

    def test_rejects_empty_instruction_and_wrong_observation_contract(self):
        bad = {key: value.clone() for key, value in self.obs.items()}
        bad["instruction_mask"].zero_()
        with self.assertRaisesRegex(ValueError, "at least one"):
            self.policy.predict_action_batch(bad)
        bad = dict(self.obs)
        bad["rgb"] = bad["rgb"].float()
        with self.assertRaisesRegex(TypeError, "uint8"):
            self.policy.predict_action_batch(bad)
        bad = dict(self.obs)
        bad["instruction_ids"] = bad["instruction_ids"][:, :128]
        with self.assertRaisesRegex(ValueError, "fixed length"):
            self.policy.predict_action_batch(bad)


if __name__ == "__main__":
    unittest.main()
