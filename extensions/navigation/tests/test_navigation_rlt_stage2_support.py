from __future__ import annotations

from argparse import Namespace
import ast
import inspect
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch


MODULE_DIR = Path(__file__).resolve().parents[1] / "integrations" / "rlinf_v03"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))


# The local contract test does not construct an RLinf manager.  Keep the import
# usable in a lightweight test environment while the A800 runtime supplies the
# real package for distributed training.
try:
    from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager
except ModuleNotFoundError:
    rlinf_module = types.ModuleType("rlinf")
    hybrid_module = types.ModuleType("rlinf.hybrid_engines")
    fsdp_module = types.ModuleType("rlinf.hybrid_engines.fsdp")
    manager_module = types.ModuleType(
        "rlinf.hybrid_engines.fsdp.fsdp_model_manager"
    )

    class FSDPModelManager:  # pragma: no cover - import-only local stub
        pass

    manager_module.FSDPModelManager = FSDPModelManager
    sys.modules["rlinf"] = rlinf_module
    sys.modules["rlinf.hybrid_engines"] = hybrid_module
    sys.modules["rlinf.hybrid_engines.fsdp"] = fsdp_module
    sys.modules[
        "rlinf.hybrid_engines.fsdp.fsdp_model_manager"
    ] = manager_module

try:
    from omegaconf import OmegaConf
except ModuleNotFoundError:
    omega_module = types.ModuleType("omegaconf")

    class OmegaConf:  # pragma: no cover - import-only local stub
        @staticmethod
        def create(value):
            return value

    omega_module.OmegaConf = OmegaConf
    sys.modules["omegaconf"] = omega_module


import navigation_rlt_fsdp_train as trainer  # noqa: E402
import navigation_rlt_prepare_training as preparer  # noqa: E402


class NavigationRLTStage2SupportTest(unittest.TestCase):
    def _epoch_tail(self, *, rank: int, keep: float, streak: int):
        # Execute the production epoch tail without constructing FSDP/CUDA.
        main = ast.parse(inspect.getsource(trainer.main)).body[0]
        epoch_loop = next(
            node
            for node in main.body
            if isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == "_epoch"
        )
        code = compile(
            ast.fix_missing_locations(
                ast.Module(body=[epoch_loop.body[-1]], type_ignores=[])
            ),
            trainer.__file__,
            "exec",
        )
        probabilities = {
            "keep_probability_mean": keep,
            "adjust_probability_mean": 1.0 - keep,
            "stop_probability_mean": 0.0,
            "recover_probability_mean": 0.0,
        }
        epoch_losses = trainer._LossSummary()
        for offset in (0.0, 2.0):
            epoch_losses.add(
                {
                    "actor_rl_loss": -3.0 + offset,
                    "reference_anchor_loss": 1.0 + offset,
                    "critic_bellman_loss": 4.0 + offset,
                    "cql_loss": 2.0 + offset,
                    "behavior_gate_kl_loss": 0.1 + offset,
                    "behavior_gate_kl_weighted_loss": 0.2 + offset,
                    "actor_gate_entropy": 0.3 + offset,
                }
            )
        namespace = {
            "phase": "stage2",
            "rank": rank,
            "_epoch": 4,
            "gradient_steps": 10,
            "epoch_probability_steps": 2,
            "epoch_probability_sums": {
                name: value * 2 for name, value in probabilities.items()
            },
            "epoch_losses": epoch_losses,
            "action_probability_epochs": [],
            "collapsed_epoch_streak": streak,
        }
        return code, {**trainer.__dict__, **namespace}

    def test_epoch_log_is_flushed_before_collapse_failure(self) -> None:
        code, namespace = self._epoch_tail(rank=0, keep=0.99, streak=1)
        with mock.patch("builtins.print") as printed:
            with self.assertRaisesRegex(
                trainer.NavigationRLTTrainingError,
                "two consecutive epochs",
            ):
                exec(code, namespace)
        printed.assert_called_once()
        self.assertTrue(printed.call_args.kwargs["flush"])
        event = json.loads(printed.call_args.args[0])
        self.assertEqual(event["event"], "navigation_rlt_stage2_epoch")
        self.assertEqual(event["epoch_index"], 4)
        self.assertEqual(event["optimizer_steps"], 10)
        self.assertEqual(event["collapsed_epoch_streak"], 2)
        self.assertEqual(event["global_gate_probabilities"]["keep_probability_mean"], 0.99)
        self.assertEqual(event["actor_rl_loss"], -2.0)
        self.assertEqual(event["reference_anchor_loss"], 2.0)
        self.assertEqual(event["critic_bellman_loss"], 5.0)
        self.assertEqual(event["cql_loss"], 3.0)
        self.assertAlmostEqual(event["losses_mean"]["behavior_gate_kl_loss"], 1.1)
        self.assertAlmostEqual(
            event["losses_mean"]["behavior_gate_kl_weighted_loss"], 1.2
        )
        self.assertAlmostEqual(event["losses_mean"]["actor_gate_entropy"], 1.3)
        self.assertTrue(event["finite"])

    def test_epoch_log_is_rank_zero_only_and_streak_resets(self) -> None:
        code, namespace = self._epoch_tail(rank=1, keep=0.6, streak=1)
        with mock.patch("builtins.print") as printed:
            exec(code, namespace)
        printed.assert_not_called()
        self.assertEqual(namespace["collapsed_epoch_streak"], 0)
        self.assertEqual(len(namespace["action_probability_epochs"]), 1)

    def setUp(self) -> None:
        self.checkpoint_sha = "a" * 64
        self.arrays = {
            "credit_mask": np.asarray([True, True, True, False, True]),
            "requested_mode": np.asarray([0, 1, 3, 2, 1], dtype=np.int64),
            "executed_mode": np.asarray([0, 1, 2, 2, 0], dtype=np.int64),
        }
        self.metadata = {
            "rlt_checkpoint_sha256": self.checkpoint_sha,
            "z_rl_dim": 256,
            "reference_dim": 4,
            "topology_node_dim": 2,
            "topology_global_dim": 3,
            "topology_max_nodes": 4,
        }
        self.replay = {
            "source_fingerprint": "b" * 64,
            "transition_count": 5,
            "shards": [
                {"path": "episode.npz", "sha256": "c" * 64, "row_count": 5}
            ],
        }

    def test_prepare_counts_only_credited_matching_gates(self) -> None:
        config = Namespace(
            reference_dim=4,
            topology_node_dim=2,
            topology_global_dim=3,
            max_topology_tokens=5,
        )
        with mock.patch.object(
            preparer,
            "load_episode_shard",
            return_value=(self.arrays, self.metadata),
        ):
            evidence = preparer._stage2_eligible(
                self.replay, self.checkpoint_sha, config
            )

        self.assertEqual(evidence["global_training_items"], 2)
        self.assertEqual(evidence["credit_ineligible_items_excluded"], 1)
        self.assertEqual(evidence["gate_mismatch_items_excluded"], 2)
        self.assertEqual(evidence["gate_counts"], [1, 1, 0, 0])
        self.assertEqual(evidence["gate_support_mask"], [True, True, False, False])
        self.assertEqual(evidence["minimum_behavior_gate_count"], 1)

    def test_trainer_uses_the_same_global_support_contract(self) -> None:
        with mock.patch.object(
            trainer,
            "_load_shard",
            return_value=(self.arrays, self.metadata),
        ):
            rows, evidence = trainer._eligible_stage2_rows(
                self.replay,
                checkpoint_sha256=self.checkpoint_sha,
                minimum_behavior_gate_count=1,
            )

        self.assertEqual(rows, [0, 1])
        self.assertEqual(evidence["gate_counts"], [1, 1, 0, 0])
        self.assertEqual(evidence["gate_support_mask"], [True, True, False, False])
        support = {
            name: evidence[name]
            for name in (
                "eligibility_schema",
                "gate_names",
                "gate_counts",
                "gate_support_mask",
                "minimum_behavior_gate_count",
            )
        }
        self.assertEqual(
            trainer._validate_stage2_support(
                support, minimum_behavior_gate_count=1
            ),
            support,
        )

    def test_support_mask_must_be_derived_from_counts(self) -> None:
        invalid = {
            "eligibility_schema": trainer.STAGE2_ELIGIBILITY_SCHEMA,
            "gate_names": list(trainer.GATE_NAMES),
            "gate_counts": [1, 1, 0, 0],
            "gate_support_mask": [True, True, False, True],
            "minimum_behavior_gate_count": 1,
        }
        with self.assertRaisesRegex(
            trainer.NavigationRLTTrainingError,
            "support mask differs from counts",
        ):
            trainer._validate_stage2_support(
                invalid, minimum_behavior_gate_count=1
            )

    def test_fixed_stage2_objective_fields_enter_training_evidence(self) -> None:
        args = Namespace(
            epochs=2,
            batch_size_per_rank=4,
            learning_rate=1.0e-4,
            weight_decay=0.0,
            clip_grad=1.0,
            reference_gate_weight=0.5,
            reference_residual_weight=0.25,
            reference_dropout_probability=0.1,
            target_tau=0.01,
            policy_version=2,
        )
        training = preparer._stage2_training(args)
        self.assertEqual(training["objective_schema"], trainer.STAGE2_OBJECTIVE_SCHEMA)
        self.assertEqual(training["cql_alpha"], 1.0)
        self.assertEqual(training["cql_temperature"], 1.0)
        self.assertEqual(training["minimum_behavior_gate_count"], 1)

    def test_manager_replaces_gate_bias_after_loading_stage1_checkpoint(self) -> None:
        config = trainer.NavigationRLTConfig(
            prefix_hidden_dim=6,
            topology_dim=6,
            reference_dim=4,
            topology_node_dim=3,
            topology_global_dim=4,
            attention_heads=8,
            transformer_layers=1,
            feedforward_dim=64,
            critic_hidden_dim=32,
            dropout=0.0,
            max_prefix_tokens=4,
            max_artifact_prefix_tokens=8,
            max_topology_tokens=4,
        )
        policy = trainer.NavigationRLTPolicy(config)
        policy.freeze_stage1()
        stage1_hash = trainer._component_hash(policy.state_dict(), ("stage1.",))
        support = (True, True, True, False)
        counts = (9679, 8850, 242, 0)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "stage1.pt"
            policy.save_checkpoint(
                checkpoint,
                training_step=1,
                policy_version=0,
                extra_metadata={
                    "training_phase": "stage1",
                    "stage1_parameter_sha256": stage1_hash,
                },
            )
            manager = object.__new__(trainer.NavigationRLTFSDPManager)
            manager.phase = "stage2"
            manager.model_config = config
            manager.stage1_checkpoint = checkpoint
            manager.expected_stage1_sha256 = trainer.sha256_file(checkpoint)
            manager.stage2_gate_support_mask = support
            manager.stage2_gate_counts = counts
            manager.stage2_initial_gate_prior = None
            manager.initial_component_hashes = None
            manager.initial_checkpoint_metadata = None
            root = manager.model_provider_func()

        self.assertEqual(root.policy.gate_support_mask.tolist(), list(support))
        self.assertIsNotNone(manager.stage2_initial_gate_prior)
        self.assertLess(max(manager.stage2_initial_gate_prior), 0.98)
        self.assertEqual(manager.stage2_initial_gate_prior[-1], 0.0)
        self.assertTrue(
            torch.equal(
                root.policy.actor.gate_head.weight,
                torch.zeros_like(root.policy.actor.gate_head.weight),
            )
        )

    def test_stage2_checkpoint_reconstruction_persists_support_mask(self) -> None:
        config = trainer.NavigationRLTConfig(
            prefix_hidden_dim=6,
            topology_dim=6,
            reference_dim=4,
            topology_node_dim=3,
            topology_global_dim=4,
            attention_heads=8,
            transformer_layers=1,
            feedforward_dim=64,
            critic_hidden_dim=32,
            dropout=0.0,
            max_prefix_tokens=4,
            max_artifact_prefix_tokens=8,
            max_topology_tokens=4,
        )
        policy = trainer.NavigationRLTPolicy(config)
        state = policy.state_dict()
        initial_hashes = {
            "stage1": trainer._component_hash(state, ("stage1.",)),
            "actor": "0" * 64,
            "q1": "1" * 64,
            "q2": "2" * 64,
            "target_q1": "3" * 64,
            "target_q2": "4" * 64,
        }
        support = [True, True, False, False]
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "policy.pt"
            trainer._save_checkpoint_from_state(
                state,
                config,
                checkpoint,
                phase="stage2",
                training_step=1,
                policy_version=1,
                initial_hashes=initial_hashes,
                run_manifest_sha256="d" * 64,
                replay_manifest_sha256="e" * 64,
                stage2_gate_support_mask=support,
            )
            restored, metadata = trainer.NavigationRLTPolicy.load_checkpoint(
                checkpoint, map_location="cpu"
            )

        self.assertEqual(metadata["gate_support_mask"], support)
        self.assertEqual(restored.gate_support_mask.tolist(), support)

    def test_stage2_checkpoint_cannot_be_reused_as_stage1_initialization(self) -> None:
        config = trainer.NavigationRLTConfig(
            prefix_hidden_dim=6,
            topology_dim=6,
            reference_dim=4,
            topology_node_dim=3,
            topology_global_dim=4,
            attention_heads=8,
            transformer_layers=1,
            feedforward_dim=64,
            critic_hidden_dim=32,
            dropout=0.0,
            max_prefix_tokens=4,
            max_artifact_prefix_tokens=8,
            max_topology_tokens=4,
        )
        policy = trainer.NavigationRLTPolicy(config)
        policy.freeze_stage1()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "stage2.pt"
            policy.save_checkpoint(
                checkpoint,
                training_step=3,
                policy_version=7,
                extra_metadata={"training_phase": "stage2"},
            )
            digest = trainer.sha256_file(checkpoint)
            args = Namespace(
                stage1_checkpoint=checkpoint,
                expected_stage1_checkpoint_sha256=digest,
            )
            with self.assertRaisesRegex(
                preparer.PrepareTrainingError,
                "not an eligible frozen Stage 1",
            ):
                preparer._stage2_checkpoint(args)

            manager = object.__new__(trainer.NavigationRLTFSDPManager)
            manager.phase = "stage2"
            manager.model_config = config
            manager.stage1_checkpoint = checkpoint
            manager.expected_stage1_sha256 = digest
            manager.stage2_gate_support_mask = (True, True, True, False)
            manager.stage2_gate_counts = (9, 7, 3, 0)
            manager.stage2_initial_gate_prior = None
            manager.initial_component_hashes = None
            manager.initial_checkpoint_metadata = None
            with self.assertRaisesRegex(
                trainer.NavigationRLTTrainingError,
                "not an eligible frozen Stage 1",
            ):
                manager.model_provider_func()


if __name__ == "__main__":
    unittest.main()
