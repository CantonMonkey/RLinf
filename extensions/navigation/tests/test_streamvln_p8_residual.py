from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).parents[1]
STAGE = ROOT / "integrations" / "rlinf_v03"
sys.path.insert(0, str(STAGE))

from streamvln_p8_residual_policy import (  # noqa: E402
    ACTION_COUNT,
    FORWARD_ACTION,
    LEFT_ACTION,
    RIGHT_ACTION,
    STOP_ACTION,
    StreamVLNPolicyPreservingResidual,
)
from streamvln_p8_topology import (  # noqa: E402
    TOPOLOGY_KEYS,
    P8TopologyTensors,
)
from streamvln_p8_live_diagnostic import (  # noqa: E402
    ALLOWED_SPLIT,
    PRELOAD_OBSERVATION_KEYS,
    LiveP8DiagnosticProviders,
    ProtocolError,
    expected_trainable_parameter_names,
    frozen_backbone_full_sha256,
)


class StreamVLNP8ResidualTests(unittest.TestCase):
    TOKEN_HIDDEN = 8
    NODE_HIDDEN = 6
    NODE_FEATURES = 5
    EDGE_FEATURES = 3

    def setUp(self):
        torch.manual_seed(20260928)
        self.policy = StreamVLNPolicyPreservingResidual(
            token_hidden_dim=self.TOKEN_HIDDEN,
            node_hidden_dim=self.NODE_HIDDEN,
            node_feature_dim=self.NODE_FEATURES,
            edge_feature_dim=self.EDGE_FEATURES,
            model_dim=16,
        )
        self.topology = self._topology(batch=2)
        self.token_hidden = torch.randn(2, 4, self.TOKEN_HIDDEN)
        self.base_logits = torch.randn(2, 4, ACTION_COUNT)

    def _topology(self, *, batch: int, requires_grad: bool = False):
        if batch not in (1, 2):
            raise ValueError("test topology supports one or two rows")
        node_hidden = torch.randn(2, 4, self.NODE_HIDDEN)
        node_features = torch.randn(2, 4, self.NODE_FEATURES)
        edge_features = torch.randn(2, 4, 4, self.EDGE_FEATURES)
        node_mask = torch.tensor(
            [[True, True, True, False], [True, True, False, False]]
        )
        adjacency = torch.tensor(
            [
                [
                    [True, True, False, True],
                    [True, True, True, True],
                    [False, True, True, True],
                    [True, True, True, True],
                ],
                [
                    [True, True, True, True],
                    [True, True, True, True],
                    [True, True, True, True],
                    [True, True, True, True],
                ],
            ],
            dtype=torch.bool,
        )
        current_index = torch.tensor([2, 1])
        parent_index = torch.tensor([[-1, 0, 1, 999], [-1, 0, 777, 888]])
        topology = P8TopologyTensors(
            node_hidden=node_hidden,
            node_features=node_features,
            node_mask=node_mask,
            adjacency=adjacency,
            edge_features=edge_features,
            current_index=current_index,
            parent_index=parent_index,
        ).index_select(torch.arange(batch))
        if requires_grad:
            values = topology.as_dict()
            values["node_hidden"] = values["node_hidden"].detach().requires_grad_()
            values["node_features"] = values["node_features"].detach().requires_grad_()
            values["edge_features"] = values["edge_features"].detach().requires_grad_()
            topology = P8TopologyTensors(**values)
        return topology

    def _activate_residual(self):
        with torch.no_grad():
            self.policy.residual_output.weight.normal_(mean=0.0, std=0.1)
            self.policy.residual_output.bias.copy_(
                torch.tensor([0.05, -0.02, 0.03, -0.04])
            )

    def test_zero_residual_exactly_preserves_release_policy(self):
        actions = self.base_logits.argmax(dim=-1)
        executed_mask = torch.ones(2, 4, dtype=torch.bool)
        outputs = self.policy(
            self.token_hidden,
            self.base_logits,
            self.topology,
            actions=actions,
            executed_mask=executed_mask,
        )
        self.assertTrue(torch.equal(outputs["residual_logits"], torch.zeros_like(self.base_logits)))
        self.assertTrue(torch.equal(outputs["policy_logits"], self.base_logits))
        self.assertTrue(torch.equal(outputs["policy_logits"].argmax(-1), actions))

        base_generator = torch.Generator().manual_seed(91)
        residual_generator = torch.Generator().manual_seed(91)
        base_sample = torch.multinomial(
            self.base_logits[0, 0].softmax(-1), 1, generator=base_generator
        )
        residual_sample = torch.multinomial(
            outputs["policy_logits"][0, 0].softmax(-1),
            1,
            generator=residual_generator,
        )
        self.assertTrue(torch.equal(base_sample, residual_sample))
        self.assertEqual(tuple(outputs["intent_logits"].shape), (2, 3))
        self.assertEqual(tuple(outputs["values"].shape), (2,))
        self.assertEqual(tuple(outputs["waypoint_mean"].shape), (2, 2))

    def test_executed_mask_controls_joint_logprob_and_entropy(self):
        base_logits = torch.zeros(2, 4, ACTION_COUNT)
        actions = torch.tensor(
            [
                [STOP_ACTION, LEFT_ACTION, 99, -1],
                [RIGHT_ACTION, 123, -7, 88],
            ]
        )
        executed_mask = torch.tensor(
            [[True, True, False, False], [True, False, False, False]]
        )
        outputs = self.policy(
            self.token_hidden,
            base_logits,
            self.topology,
            actions=actions,
            executed_mask=executed_mask,
        )
        log_four = torch.log(torch.tensor(4.0))
        self.assertTrue(
            torch.allclose(outputs["decision_log_probs"], torch.tensor([-2.0, -1.0]) * log_four)
        )
        self.assertTrue(torch.allclose(outputs["masked_mean_entropy"], log_four.expand(2)))
        self.assertTrue(torch.equal(outputs["executed_token_count"], torch.tensor([2, 1])))
        self.assertTrue(torch.equal(outputs["token_log_probs"][~executed_mask], torch.zeros(5)))

        changed = base_logits.clone()
        changed[~executed_mask] = torch.tensor([100.0, -100.0, 50.0, -50.0])
        changed_outputs = self.policy(
            self.token_hidden,
            changed,
            self.topology,
            actions=actions,
            executed_mask=executed_mask,
        )
        self.assertTrue(
            torch.equal(outputs["decision_log_probs"], changed_outputs["decision_log_probs"])
        )
        self.assertTrue(
            torch.equal(outputs["masked_mean_entropy"], changed_outputs["masked_mean_entropy"])
        )

    def test_single_row_matches_the_same_batched_row(self):
        self._activate_residual()
        batch = self.policy(self.token_hidden, self.base_logits, self.topology)
        row_topology = self.topology.index_select(torch.tensor([0]))
        row = self.policy(
            self.token_hidden[0:1], self.base_logits[0:1], row_topology
        )
        for key in (
            "policy_logits",
            "residual_logits",
            "waypoint_mean",
            "intent_logits",
            "values",
            "context",
            "attention_weights",
        ):
            self.assertTrue(torch.allclose(batch[key][0:1], row[key], atol=1e-6, rtol=1e-6), key)

    def test_session_callback_fixes_first_token_condition_for_replay(self):
        self._activate_residual()
        topology = self.topology.index_select(torch.tensor([0]))
        callback = self.policy.decision_residual_callback(topology)
        sampled_residuals = torch.stack(
            [
                callback(self.token_hidden[0, token], self.base_logits[0, token])
                for token in range(4)
            ]
        )
        replay = self.policy(
            self.token_hidden[0:1], self.base_logits[0:1], topology
        )
        self.assertTrue(
            torch.allclose(
                sampled_residuals,
                replay["residual_logits"][0],
                atol=1e-6,
                rtol=1e-6,
            )
        )

    def test_stop_forward_and_both_turns_keep_four_action_dimensions(self):
        actions = torch.tensor(
            [[STOP_ACTION, FORWARD_ACTION, LEFT_ACTION, RIGHT_ACTION]]
        )
        mask = torch.ones(1, 4, dtype=torch.bool)
        outputs = self.policy(
            self.token_hidden[0:1],
            self.base_logits[0:1],
            self.topology.index_select(torch.tensor([0])),
            actions=actions,
            executed_mask=mask,
        )
        self.assertEqual(ACTION_COUNT, 4)
        self.assertEqual(tuple(outputs["policy_logits"].shape), (1, 4, 4))
        expected = outputs["policy_logits"].log_softmax(-1).gather(
            -1, actions[..., None]
        ).squeeze(-1).sum(-1)
        self.assertTrue(torch.equal(outputs["decision_log_probs"], expected))
        with self.assertRaisesRegex(ValueError, "exactly four"):
            self.policy(
                self.token_hidden[0:1, 0],
                torch.zeros(1, 3),
                self.topology.index_select(torch.tensor([0])),
            )

    def test_topology_padding_and_masked_nodes_have_no_effect(self):
        self._activate_residual()
        topology = self.topology.index_select(torch.tensor([0]))
        changed = P8TopologyTensors(**{key: value.clone() for key, value in topology.as_dict().items()})
        changed.node_hidden[:, 3].fill_(1e20)
        changed.node_features[:, 3].fill_(-1e20)
        changed.edge_features[:, 3, :, :].fill_(1e19)
        changed.edge_features[:, :, 3, :].fill_(-1e19)
        changed.adjacency[:, 3, :] = ~changed.adjacency[:, 3, :]
        changed.adjacency[:, :, 3] = ~changed.adjacency[:, :, 3]
        changed.parent_index[:, 3] = 2**30
        base = self.policy(self.token_hidden[0:1], self.base_logits[0:1], topology)
        padded = self.policy(self.token_hidden[0:1], self.base_logits[0:1], changed)
        self.assertEqual(tuple(topology.as_dict()), TOPOLOGY_KEYS)
        for key in (
            "policy_logits",
            "residual_logits",
            "waypoint_mean",
            "intent_logits",
            "values",
            "context",
            "attention_weights",
        ):
            self.assertTrue(torch.equal(base[key], padded[key]), key)
        self.assertTrue(torch.equal(base["attention_weights"][:, 3], torch.zeros(1)))

    def test_parameters_are_fp32_and_gradients_stop_at_frozen_inputs(self):
        topology = self._topology(batch=1, requires_grad=True)
        token_hidden = torch.randn(1, 4, self.TOKEN_HIDDEN, requires_grad=True)
        base_logits = torch.randn(1, 4, ACTION_COUNT, requires_grad=True)
        outputs = self.policy(
            token_hidden,
            base_logits,
            topology,
            actions=torch.tensor([[STOP_ACTION, FORWARD_ACTION, LEFT_ACTION, RIGHT_ACTION]]),
            executed_mask=torch.ones(1, 4, dtype=torch.bool),
        )
        loss = (
            -outputs["decision_log_probs"].mean()
            + outputs["waypoint_mean"].square().mean()
            + outputs["intent_logits"].square().mean()
            + outputs["values"].square().mean()
        )
        loss.backward()

        for name, parameter in self.policy.named_parameters():
            self.assertEqual(parameter.dtype, torch.float32, name)
            self.assertTrue(parameter.requires_grad, name)
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertIsNone(token_hidden.grad)
        self.assertIsNone(base_logits.grad)
        self.assertIsNone(topology.node_hidden.grad)
        self.assertIsNone(topology.node_features.grad)
        self.assertIsNone(topology.edge_features.grad)

    def test_full_backbone_hash_detects_an_unsampled_tensor_change(self):
        class Wrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = torch.nn.Sequential(
                    torch.nn.Linear(5, 7),
                    torch.nn.Linear(7, 3),
                )

        wrapper = Wrapper()
        before = frozen_backbone_full_sha256(wrapper)
        with torch.no_grad():
            wrapper.backbone[-1].weight[-1, -1].add_(1.0)
        self.assertNotEqual(before, frozen_backbone_full_sha256(wrapper))

    def test_exact_trainable_inventory_rejects_missing_or_new_parameters(self):
        for parameter in self.policy.value_head.parameters():
            parameter.requires_grad_(False)
        names = expected_trainable_parameter_names(self.policy)
        self.assertTrue(any(name.startswith("decision_projection.") for name in names))
        self.assertTrue(any(name.startswith("condition_fusion.") for name in names))

        first = dict(self.policy.named_parameters())[names[0]]
        first.requires_grad_(False)
        with self.assertRaisesRegex(ProtocolError, "requires_grad scope"):
            expected_trainable_parameter_names(self.policy)


class StreamVLNP8PreloadProbeTests(unittest.TestCase):
    def test_all_selected_scenes_reset_and_close_before_model_load(self):
        provider = object.__new__(LiveP8DiagnosticProviders)
        specs = [
            {"index": index, "episode_id": f"episode-{index}", "scene": f"scene-{index}"}
            for index in range(16)
        ]
        provider.manifest = {
            "episodes": {
                "tiny_train": specs[:8],
                "train_development": specs[8:],
            }
        }
        provider.training_contract = {"expert_lookahead_m": 1.0}
        provider.run_cfg = {}
        provider.base_policy = None
        provider.residual_policy = None
        provider.model_load_count = 0
        provider.split_read_counts = {ALLOWED_SPLIT: 0}
        provider.probe_split_read_counts = {ALLOWED_SPLIT: 0}
        provider.base_policy_loader = lambda: self.fail("base model must not load")
        provider.residual_policy_loader = lambda _base: self.fail(
            "residual model must not load"
        )
        resets = []
        closes = []

        class FakeEnv:
            def __init__(self, spec):
                self.spec = spec

            def reset(self):
                resets.append(self.spec["index"])
                obs = {
                    key: torch.zeros(1, dtype=torch.int64)
                    for key in PRELOAD_OBSERVATION_KEYS
                }
                obs["rgb"] = torch.zeros((1, 64, 64, 3), dtype=torch.uint8)
                for key in (
                    "instruction_utf8_mask",
                    "scene_utf8_mask",
                    "episode_utf8_mask",
                ):
                    obs[key] = torch.ones((1, 1), dtype=torch.bool)
                obs["episode_start"] = torch.tensor([True], dtype=torch.bool)
                obs["previous_action_mask"] = torch.zeros((1, 4), dtype=torch.bool)
                obs["terminal_frozen"] = torch.tensor([False], dtype=torch.bool)
                return obs, {
                    "episode_id": self.spec["episode_id"],
                    "scene_id": f"/frozen/{self.spec['scene']}.glb",
                }

            def close(self):
                closes.append(self.spec["index"])

            def expert_waypoint(self, lookahead_m=1.0):
                return {
                    "source": "train-shortest-path",
                    "lookahead_m": float(lookahead_m),
                }

        def env_factory(_run_cfg, *, split, episode_index, max_steps):
            self.assertEqual(split, ALLOWED_SPLIT)
            self.assertEqual(max_steps, 4)
            return FakeEnv(specs[episode_index])

        provider.env_factory = env_factory
        receipt = provider.preload_env_probe()
        self.assertEqual(resets, list(range(16)))
        self.assertEqual(closes, list(range(16)))
        self.assertEqual(receipt["episode_count"], 16)
        self.assertEqual(receipt["scene_count"], 16)
        self.assertTrue(receipt["expert_waypoint_valid"])
        self.assertEqual(receipt["model_load_count"], 0)
        self.assertEqual(
            receipt["probe_split_read_counts"],
            {"train": 16, "val_seen": 0, "val_unseen": 0},
        )
        self.assertEqual(
            receipt["scientific_split_read_counts"],
            {"train": 0, "val_seen": 0, "val_unseen": 0},
        )
        self.assertIsNone(provider.base_policy)
        self.assertIsNone(provider.residual_policy)


if __name__ == "__main__":
    unittest.main()
