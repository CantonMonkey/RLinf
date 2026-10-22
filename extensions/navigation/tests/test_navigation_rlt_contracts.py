from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch


MODULE_DIR = Path(__file__).resolve().parents[1] / "integrations" / "rlinf_v03"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from navigation_rlt_action import (  # noqa: E402
    ADJUST,
    KEEP,
    RECOVER,
    STOP,
    ActionContractError,
    compile_action as compile_numpy_action,
    valid_mode_mask,
)
from navigation_rlt_parallel import (  # noqa: E402
    WORKER_SCHEMA,
    WORKER_VERSION,
    atomic_json,
    build_plan,
    sha256_file as parallel_sha256_file,
    verify_completions,
)
from navigation_rlt_policy import (  # noqa: E402
    ADJUST_GATE,
    CANONICAL_ACTION_DIM,
    KEEP_GATE,
    RECOVER_GATE,
    STOP_GATE,
    NavigationRLTCompilerFields,
    NavigationRLTConfig,
    NavigationRLTContext,
    NavigationRLTContractError,
    NavigationRLTPolicy,
    NavigationRLTTransitionBatch,
    NavigationRLTZTransitionBatch,
    Stage1LossConfig,
    Stage2LossConfig,
    canonical_action_from_mode_and_se2,
    pack_topology_tokens,
    select_prefix_tokens,
    supported_behavior_gate_kl,
)
from navigation_rlt_replay import (  # noqa: E402
    END_REASON_NAMES,
    FEATURE_STAGE_PREFIX_HIDDEN,
    LIGHTNAV_SOURCE_REVISION,
    REFERENCE_DIM,
    REFERENCE_SCHEMA,
    REPRESENTATION_PREFIX_HIDDEN,
    ReplayContractError,
    SCHEMA_NAME as REPLAY_SCHEMA_NAME,
    SCHEMA_VERSION as REPLAY_SCHEMA_VERSION,
    build_replay_manifest,
    concatenate_episode_shards,
    load_episode_shard,
    save_episode_shard,
    source_fingerprint,
)


def _policy_config() -> NavigationRLTConfig:
    return NavigationRLTConfig(
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


def _selected_context(config: NavigationRLTConfig) -> tuple[
    NavigationRLTContext, object, object
]:
    hidden = torch.arange(2 * 6 * config.prefix_hidden_dim, dtype=torch.float32)
    hidden = hidden.reshape(2, 6, config.prefix_hidden_dim) / 100.0
    prefix_mask = torch.tensor(
        [[True, True, True, True, True, True], [True, True, True, True, True, False]]
    )
    hidden[1, 5] = 0.0
    selected = select_prefix_tokens(hidden, prefix_mask, config=config)

    node_targets = torch.tensor(
        [
            [[0.1, 0.0, 1.0], [0.2, 0.1, 2.0]],
            [[0.3, -0.1, 3.0], [0.0, 0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    node_mask = torch.tensor([[True, True], [True, False]])
    global_targets = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
        dtype=torch.float32,
    )
    global_mask = torch.ones_like(global_targets, dtype=torch.bool)
    topology = pack_topology_tokens(
        node_targets,
        node_mask,
        global_targets,
        global_mask,
        config=config,
    )
    context = NavigationRLTContext(
        contextual_prefix_hidden=selected.contextual_prefix_hidden,
        prefix_mask=selected.prefix_mask,
        topology_tokens=topology.tokens,
        topology_mask=topology.token_mask,
    )
    return context, selected, topology


def _compiler(batch: int = 2) -> NavigationRLTCompilerFields:
    reference_se2 = torch.tensor(
        [[0.20, 0.00, 0.10], [0.15, 0.02, -0.10]], dtype=torch.float32
    )[:batch]
    parent_se2 = torch.tensor(
        [[0.10, 0.05, 0.00], [0.12, -0.03, 0.05]], dtype=torch.float32
    )[:batch]
    reference = canonical_action_from_mode_and_se2(
        torch.full((batch,), KEEP_GATE, dtype=torch.int64), reference_se2
    )
    adjust = canonical_action_from_mode_and_se2(
        torch.full((batch,), ADJUST_GATE, dtype=torch.int64), reference_se2
    )
    stop = canonical_action_from_mode_and_se2(
        torch.full((batch,), STOP_GATE, dtype=torch.int64),
        torch.zeros((batch, 3), dtype=torch.float32),
    )
    recover = canonical_action_from_mode_and_se2(
        torch.full((batch,), RECOVER_GATE, dtype=torch.int64), parent_se2
    )
    basis = torch.zeros((batch, 2, CANONICAL_ACTION_DIM), dtype=torch.float32)
    basis[:, 0, 4] = 1.0
    basis[:, 1, 6] = 1.0
    return NavigationRLTCompilerFields(
        valid_gate_mask=torch.ones((batch, 4), dtype=torch.bool),
        reference_action=reference,
        adjust_base_action=adjust,
        adjust_residual_basis=basis,
        stop_action=stop,
        recover_parent_action=recover,
        recover_parent_valid=torch.ones((batch,), dtype=torch.bool),
        recover_parent_index=torch.arange(batch, dtype=torch.int64),
    )


def _z_replay(
    policy: NavigationRLTPolicy,
    context: NavigationRLTContext,
    *,
    behavior_gate: torch.Tensor | None = None,
    behavior_raw: torch.Tensor | None = None,
) -> NavigationRLTZTransitionBatch:
    with torch.no_grad():
        z_rl = policy.encode(context).detach()
        next_z_rl = policy.encode(context).detach()
    compiler = _compiler(z_rl.shape[0])
    executed_gate = torch.tensor([KEEP_GATE, ADJUST_GATE], dtype=torch.int64)
    executed = policy.compile_action(
        executed_gate,
        torch.zeros((z_rl.shape[0], 2), dtype=torch.float32),
        policy._prepare_compiler(
            compiler,
            batch=z_rl.shape[0],
            device=z_rl.device,
            dtype=z_rl.dtype,
        ),
    ).detach()
    return NavigationRLTZTransitionBatch(
        z_rl=z_rl,
        next_z_rl=next_z_rl,
        reference_embedding=torch.tensor(
            [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]], dtype=torch.float32
        ),
        next_reference_embedding=torch.tensor(
            [[0.2, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.2]], dtype=torch.float32
        ),
        compiler=compiler,
        next_compiler=compiler,
        canonical_executed_action=executed,
        reward=torch.tensor([0.5, -0.2], dtype=torch.float32),
        discount=torch.tensor([0.99, 0.0], dtype=torch.float32),
        reference_gate=torch.tensor([KEEP_GATE, KEEP_GATE], dtype=torch.int64),
        behavior_gate=(
            behavior_gate
            if behavior_gate is not None
            else torch.tensor([KEEP_GATE, ADJUST_GATE], dtype=torch.int64)
        ),
        behavior_raw_residual=(
            behavior_raw
            if behavior_raw is not None
            else torch.tensor([[0.0, 0.0], [1.0, -1.0]], dtype=torch.float32)
        ),
        behavior_policy_version=torch.tensor([1, 7], dtype=torch.int64),
    )


class NavigationRLTPolicyContractTests(unittest.TestCase):
    def test_prefix_selection_topology_pack_and_strict_canonical_action(self) -> None:
        config = _policy_config()
        context, selected, topology = _selected_context(config)
        _, selected_again, _ = _selected_context(config)

        self.assertTrue(
            torch.equal(selected.selected_source_indices, selected_again.selected_source_indices)
        )
        self.assertEqual(selected.original_valid_lengths.tolist(), [6, 5])
        self.assertEqual(selected.selected_valid_lengths.tolist(), [4, 4])
        self.assertEqual(selected.source_sequence_width, 6)
        self.assertEqual(selected.selected_source_indices[0].tolist(), [0, 1, 3, 5])
        self.assertEqual(selected.selected_source_indices[1].tolist(), [0, 1, 2, 4])
        self.assertEqual(tuple(context.contextual_prefix_hidden.shape), (2, 4, 6))

        too_wide = torch.zeros((1, 9, 6), dtype=torch.float32)
        with self.assertRaises(NavigationRLTContractError):
            select_prefix_tokens(
                too_wide, torch.ones((1, 9), dtype=torch.bool), config=config
            )
        raw_context = NavigationRLTContext(
            contextual_prefix_hidden=torch.zeros((1, 5, 6)),
            prefix_mask=torch.ones((1, 5), dtype=torch.bool),
            topology_tokens=torch.zeros((1, 1, 6)),
            topology_mask=torch.ones((1, 1), dtype=torch.bool),
        )
        with self.assertRaises(NavigationRLTContractError):
            NavigationRLTPolicy(config).encode(raw_context)

        self.assertEqual(tuple(topology.tokens.shape), (2, 4, 6))
        self.assertEqual(topology.token_mask.tolist(), [[True, True, True, False], [True, False, True, False]])
        self.assertTrue(torch.equal(topology.tokens[0, 2, :4], torch.tensor([0.1, 0.2, 0.3, 0.4])))
        self.assertTrue(torch.equal(topology.tokens[1, 1], torch.zeros(6)))

        modes = torch.tensor([KEEP_GATE, ADJUST_GATE, STOP_GATE, RECOVER_GATE])
        se2 = torch.tensor(
            [[0.2, 0.0, 0.1], [0.21, 0.0, 0.11], [0.0, 0.0, 0.0], [0.1, 0.1, 0.0]]
        )
        canonical = canonical_action_from_mode_and_se2(modes, se2)
        self.assertEqual(tuple(canonical.shape), (4, 7))
        self.assertTrue(torch.equal(canonical[:, :4], torch.eye(4)))
        self.assertTrue(torch.equal(canonical[:, 4:], se2))

    def test_compiler_output_equals_selected_candidate(self) -> None:
        policy = NavigationRLTPolicy(_policy_config())
        compiler = policy._prepare_compiler(
            _compiler(2), batch=2, device=torch.device("cpu"), dtype=torch.float32
        )
        gates = torch.tensor([ADJUST_GATE, RECOVER_GATE], dtype=torch.int64)
        bounded = torch.tensor([[0.03, 0.05], [0.0, 0.0]], dtype=torch.float32)
        candidates = policy._candidate_action_table_from_bounded(bounded, compiler)
        compiled = policy.compile_action(gates, bounded, compiler)
        expected = candidates[torch.arange(2), gates]
        self.assertTrue(torch.equal(compiled, expected))
        self.assertTrue(torch.equal(compiled[:, :4], torch.eye(4)[gates]))

    def test_stage1_masked_reconstruction_backward_keeps_inputs_frozen(self) -> None:
        policy = NavigationRLTPolicy(_policy_config())
        context, _, _ = _selected_context(policy.config)
        prefix_mask = torch.tensor(
            [[True, False, False, False], [True, False, False, False]]
        )
        topology_mask = torch.tensor(
            [[False, False, True, False], [False, False, True, False]]
        )
        losses = policy.stage1_loss(
            context,
            Stage1LossConfig(),
            prefix_reconstruction_mask=prefix_mask,
            topology_reconstruction_mask=topology_mask,
        )
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        losses["loss"].backward()
        self.assertTrue(any(parameter.grad is not None for parameter in policy.stage1.parameters()))
        for value in (
            context.contextual_prefix_hidden,
            context.prefix_mask,
            context.topology_tokens,
            context.topology_mask,
        ):
            self.assertIsNone(value.grad)
            self.assertFalse(value.requires_grad)

    def test_stage2_context_and_frozen_z_contracts(self) -> None:
        policy = NavigationRLTPolicy(_policy_config())
        context, _, _ = _selected_context(policy.config)
        policy.freeze_stage1()
        z_replay = _z_replay(policy, context)
        loss_config = Stage2LossConfig(reference_dropout_probability=0.0)

        context_replay = NavigationRLTTransitionBatch(
            context=context,
            next_context=context,
            reference_embedding=z_replay.reference_embedding,
            next_reference_embedding=z_replay.next_reference_embedding,
            compiler=z_replay.compiler,
            next_compiler=z_replay.next_compiler,
            canonical_executed_action=z_replay.canonical_executed_action,
            reward=z_replay.reward,
            discount=z_replay.discount,
            reference_gate=z_replay.reference_gate,
            behavior_gate=z_replay.behavior_gate,
            behavior_raw_residual=z_replay.behavior_raw_residual,
            behavior_policy_version=z_replay.behavior_policy_version,
        )
        context_losses = policy.stage2_loss(context_replay, loss_config)
        z_losses = policy.stage2_loss_from_z(z_replay, loss_config)
        self.assertTrue(torch.isfinite(context_losses["loss"]))
        self.assertTrue(torch.isfinite(z_losses["loss"]))
        self.assertEqual(float(z_losses["behavior_version_count"]), 2.0)

        changed_behavior = _z_replay(
            policy,
            context,
            behavior_gate=torch.tensor([RECOVER_GATE, STOP_GATE], dtype=torch.int64),
            behavior_raw=torch.tensor([[9.0, -4.0], [-3.0, 2.0]], dtype=torch.float32),
        )
        changed_losses = policy.stage2_loss_from_z(changed_behavior, loss_config)
        for name in ("loss", "reference_anchor_loss", "reference_gate_loss", "reference_residual_loss"):
            self.assertTrue(torch.allclose(z_losses[name], changed_losses[name], rtol=0.0, atol=0.0))

        policy.zero_grad(set_to_none=True)
        z_losses["loss"].backward()
        self.assertTrue(any(parameter.grad is not None for parameter in policy.actor.parameters()))
        self.assertTrue(any(parameter.grad is not None for parameter in policy.q1.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in policy.stage1.parameters()))
        for target in (policy.target_q1, policy.target_q2):
            self.assertTrue(all(not parameter.requires_grad for parameter in target.parameters()))
            self.assertTrue(all(parameter.grad is None for parameter in target.parameters()))

    def test_reference_conditioning_changes_twin_q_values(self) -> None:
        policy = NavigationRLTPolicy(_policy_config())
        z_rl = torch.zeros((2, 256), dtype=torch.float32)
        reference_zero = torch.zeros((2, policy.config.reference_dim), dtype=torch.float32)
        reference_one = torch.ones((2, policy.config.reference_dim), dtype=torch.float32)
        compiler = policy._prepare_compiler(
            _compiler(2), batch=2, device=torch.device("cpu"), dtype=torch.float32
        )
        action = compiler.reference_action
        with torch.no_grad():
            for critic in (policy.q1, policy.q2):
                critic.linear1.weight.zero_()
                critic.linear1.bias.zero_()
                critic.reference_linear.weight.fill_(1.0)
                critic.linear2.weight.fill_(1.0 / policy.config.critic_hidden_dim)
                critic.linear2.bias.zero_()
                critic.output.weight.fill_(1.0 / policy.config.critic_hidden_dim)
                critic.output.bias.zero_()
            zero_q = policy.q_values(z_rl, reference_zero, action)
            one_q = policy.q_values(z_rl, reference_one, action)
        for zero_value, one_value in zip(zero_q, one_q, strict=True):
            self.assertFalse(torch.allclose(zero_value, one_value))
            self.assertTrue(bool((one_value > zero_value).all()))

    def test_finite_four_candidate_cql_matches_manual_and_detaches_actor(self) -> None:
        policy = NavigationRLTPolicy(_policy_config())
        context, _, _ = _selected_context(policy.config)
        policy.freeze_stage1()
        policy.set_gate_support_mask(torch.ones(4, dtype=torch.bool))
        replay = _z_replay(policy, context)
        loss_config = Stage2LossConfig(
            reference_dropout_probability=0.0,
            cql_alpha=1.0,
            cql_temperature=0.7,
        )
        losses = policy.stage2_loss_from_z(replay, loss_config)

        with torch.no_grad():
            z_rl = policy._prepare_frozen_z(replay.z_rl, name="z_rl")
            reference = policy._prepare_reference(
                replay.reference_embedding,
                batch=z_rl.shape[0],
                device=z_rl.device,
                dtype=z_rl.dtype,
            )
            compiler = policy._prepare_compiler(
                replay.compiler,
                batch=z_rl.shape[0],
                device=z_rl.device,
                dtype=z_rl.dtype,
            )
            distribution = policy.actor_distribution(z_rl, reference, compiler)
            candidates = policy._candidate_actions(distribution, compiler)
            self.assertEqual(tuple(candidates.shape), (2, 4, CANONICAL_ACTION_DIM))
            self.assertTrue(
                torch.equal(
                    candidates[:, RECOVER_GATE, :4],
                    torch.eye(4, dtype=torch.float32)[RECOVER_GATE].expand(2, -1),
                )
            )
            q1_candidates = []
            q2_candidates = []
            for gate_index in range(4):
                first, second = policy.q_values(
                    z_rl, reference, candidates[:, gate_index]
                )
                q1_candidates.append(first)
                q2_candidates.append(second)
            q1_candidates = torch.stack(q1_candidates, dim=-1)
            q2_candidates = torch.stack(q2_candidates, dim=-1)
            data_q1, data_q2 = policy.q_values(
                z_rl, reference, replay.canonical_executed_action
            )
            temperature = loss_config.cql_temperature
            expected_cql1 = (
                temperature * torch.logsumexp(q1_candidates / temperature, dim=-1)
                - data_q1
            ).mean()
            expected_cql2 = (
                temperature * torch.logsumexp(q2_candidates / temperature, dim=-1)
                - data_q2
            ).mean()
        self.assertTrue(torch.allclose(losses["cql1_loss"], expected_cql1, atol=1.0e-6))
        self.assertTrue(torch.allclose(losses["cql2_loss"], expected_cql2, atol=1.0e-6))
        self.assertEqual(float(losses["cql_structural_candidate_count_mean"]), 4.0)

        policy.zero_grad(set_to_none=True)
        losses["cql_loss"].backward()
        self.assertTrue(all(parameter.grad is None for parameter in policy.actor.parameters()))
        self.assertTrue(any(parameter.grad is not None for parameter in policy.q1.parameters()))

    def test_zero_support_recover_is_actor_masked_but_structurally_cql_penalized(self) -> None:
        policy = NavigationRLTPolicy(_policy_config())
        context, _, _ = _selected_context(policy.config)
        policy.freeze_stage1()
        policy.set_gate_support_mask(
            torch.tensor([True, True, True, False], dtype=torch.bool)
        )
        replay = _z_replay(policy, context)
        loss_config = Stage2LossConfig(
            reference_dropout_probability=0.0,
            cql_alpha=1.0,
            cql_temperature=1.0,
        )
        prepared = policy._prepare_compiler(
            replay.compiler,
            batch=replay.z_rl.shape[0],
            device=replay.z_rl.device,
            dtype=replay.z_rl.dtype,
        )
        distribution = policy.actor_distribution(
            replay.z_rl, replay.reference_embedding, prepared
        )
        self.assertTrue(
            torch.equal(
                distribution["gate_probabilities"][:, RECOVER_GATE],
                torch.zeros(replay.z_rl.shape[0], dtype=torch.float32),
            )
        )

        with torch.no_grad():
            for critic in (policy.q1, policy.q2):
                for parameter in critic.parameters():
                    parameter.zero_()
        baseline = policy.stage2_loss_from_z(replay, loss_config)
        self.assertEqual(
            float(baseline["cql_structural_candidate_count_mean"]), 4.0
        )

        with torch.no_grad():
            recover_input = replay.z_rl.shape[-1] + RECOVER_GATE
            for critic in (policy.q1, policy.q2):
                critic.linear1.weight[:, recover_input] = 1.0
                critic.linear2.weight.fill_(1.0 / policy.config.critic_hidden_dim)
                critic.output.weight.fill_(1.0 / policy.config.critic_hidden_dim)
        raised_recover = policy.stage2_loss_from_z(replay, loss_config)
        self.assertGreater(
            float(raised_recover["cql_loss"].detach()),
            float(baseline["cql_loss"].detach()),
        )

    def test_stage2_behavior_prior_replaces_loaded_keep_collapse_bias(self) -> None:
        policy = NavigationRLTPolicy(_policy_config())
        policy.freeze_stage1()
        support = torch.tensor([True, True, True, False], dtype=torch.bool)
        policy.set_gate_support_mask(support)
        prepared = policy._prepare_compiler(
            _compiler(2), batch=2, device=torch.device("cpu"), dtype=torch.float32
        )
        z_rl = torch.zeros((2, 256), dtype=torch.float32)
        reference = torch.zeros((2, policy.config.reference_dim), dtype=torch.float32)
        before = policy.actor_distribution(z_rl, reference, prepared)[
            "gate_probabilities"
        ][0]
        self.assertGreater(float(before.max()), 0.98)

        counts = [9679, 8850, 242, 0]
        prior = policy.initialize_stage2_gate_prior_from_behavior_counts(counts)
        empirical = torch.tensor(counts[:3], dtype=torch.float32)
        empirical = empirical / empirical.sum()
        expected = 0.95 * empirical + 0.05 * torch.full_like(empirical, 1.0 / 3.0)
        self.assertTrue(torch.allclose(prior[:3], expected, atol=1.0e-7, rtol=0.0))
        self.assertEqual(float(prior[RECOVER_GATE]), 0.0)
        self.assertLess(float(prior.max()), 0.98)
        self.assertTrue(
            torch.equal(
                policy.actor.gate_head.weight,
                torch.zeros_like(policy.actor.gate_head.weight),
            )
        )

        after = policy.actor_distribution(z_rl, reference, prepared)[
            "gate_probabilities"
        ][0]
        self.assertTrue(torch.allclose(after, prior, atol=1.0e-7, rtol=0.0))

    def test_stage2_behavior_prior_has_margin_with_any_supported_alternative(self) -> None:
        for alternative in (ADJUST_GATE, STOP_GATE, RECOVER_GATE):
            with self.subTest(alternative=alternative):
                policy = NavigationRLTPolicy(_policy_config())
                policy.freeze_stage1()
                support = torch.zeros(4, dtype=torch.bool)
                support[KEEP_GATE] = True
                support[alternative] = True
                policy.set_gate_support_mask(support)
                counts = [0, 0, 0, 0]
                counts[KEEP_GATE] = 1_000_000
                counts[alternative] = 1
                prior = policy.initialize_stage2_gate_prior_from_behavior_counts(
                    counts
                )
                self.assertLess(float(prior.max()), 0.98)
                self.assertAlmostEqual(float(prior.sum()), 1.0, places=6)
                self.assertTrue(torch.equal(prior > 0.0, support))

    def test_stage2_behavior_prior_rejects_single_gate_support(self) -> None:
        policy = NavigationRLTPolicy(_policy_config())
        policy.freeze_stage1()
        policy.set_gate_support_mask(
            torch.tensor([True, False, False, False], dtype=torch.bool)
        )
        with self.assertRaisesRegex(
            NavigationRLTContractError,
            "at least two replay-supported gates",
        ):
            policy.initialize_stage2_gate_prior_from_behavior_counts([11, 0, 0, 0])

    def test_out_of_support_native_reference_skips_gate_anchor(self) -> None:
        policy = NavigationRLTPolicy(_policy_config())
        context, _, _ = _selected_context(policy.config)
        policy.freeze_stage1()
        policy.set_gate_support_mask(
            torch.tensor([False, True, False, False], dtype=torch.bool)
        )
        replay = _z_replay(
            policy,
            context,
            behavior_gate=torch.tensor(
                [ADJUST_GATE, ADJUST_GATE], dtype=torch.int64
            ),
        )
        losses = policy.stage2_loss_from_z(
            replay,
            Stage2LossConfig(reference_dropout_probability=0.0),
        )
        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertEqual(float(losses["reference_gate_loss"].detach()), 0.0)
        self.assertEqual(
            float(losses["reference_gate_anchor_supported_fraction"]), 0.0
        )

    def test_supported_behavior_kl_matches_masked_manual_and_zero(self) -> None:
        prior = torch.tensor([0.6, 0.3, 0.1, 0.0], dtype=torch.float32)
        mask = torch.tensor(
            [[True, True, False, False], [True, False, True, False]],
            dtype=torch.bool,
        )
        # The large illegal RECOVER logit must have no effect on either row.
        logits = torch.tensor(
            [[math.log(0.5), math.log(0.25), math.log(0.25), 100.0]] * 2,
            dtype=torch.float32,
        )
        actual = supported_behavior_gate_kl(logits, mask, prior)
        expected = torch.tensor(
            [
                0.0,
                (6.0 / 7.0) * math.log((6.0 / 7.0) / (2.0 / 3.0))
                + (1.0 / 7.0) * math.log((1.0 / 7.0) / (1.0 / 3.0)),
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(actual, expected, rtol=0.0, atol=1.0e-6))
        matched_logits = prior.clamp_min(1.0e-30).log().expand(2, -1)
        matched = supported_behavior_gate_kl(matched_logits, mask, prior)
        self.assertTrue(torch.allclose(matched, torch.zeros(2), rtol=0.0, atol=1.0e-6))
        self.assertFalse(prior.requires_grad)

    def test_supported_behavior_kl_restores_collapsed_bf16_gate_gradient(self) -> None:
        logits = torch.tensor(
            [[120.0, -120.0, -120.0, 500.0]],
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        mask = torch.tensor([[True, True, True, False]], dtype=torch.bool)
        prior = torch.tensor([0.5, 0.4, 0.1, 0.0], dtype=torch.float32)
        loss = supported_behavior_gate_kl(logits, mask, prior).mean()
        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))
        expected_gradient = torch.tensor([[0.5, -0.4, -0.1, 0.0]])
        self.assertTrue(
            torch.allclose(logits.grad.float(), expected_gradient, rtol=0.0, atol=2.0e-3)
        )
        self.assertGreater(float(logits.grad[0, KEEP_GATE]), 0.0)
        self.assertLess(float(logits.grad[0, ADJUST_GATE]), 0.0)
        self.assertLess(float(logits.grad[0, STOP_GATE]), 0.0)
        self.assertEqual(float(logits.grad[0, RECOVER_GATE]), 0.0)
        self.assertIsNone(prior.grad)

    def test_stage2_behavior_kl_backward_updates_only_actor(self) -> None:
        policy = NavigationRLTPolicy(_policy_config())
        context, _, _ = _selected_context(policy.config)
        policy.freeze_stage1()
        policy.set_gate_support_mask(torch.tensor([True, True, True, False]))
        replay = _z_replay(policy, context)
        losses = policy.stage2_loss_from_z(
            replay,
            Stage2LossConfig(
                reference_dropout_probability=0.0,
                behavior_kl_weight=1.0,
                behavior_gate_prior=(0.5, 0.4, 0.1, 0.0),
            ),
        )
        self.assertGreater(float(losses["behavior_gate_kl_loss"].detach()), 0.0)
        self.assertTrue(
            torch.allclose(
                losses["actor_loss"],
                losses["actor_rl_loss"]
                + losses["reference_anchor_loss"]
                + losses["behavior_gate_kl_weighted_loss"],
                rtol=0.0,
                atol=1.0e-6,
            )
        )
        policy.zero_grad(set_to_none=True)
        losses["behavior_gate_kl_weighted_loss"].backward()
        self.assertTrue(
            any(
                parameter.grad is not None and bool((parameter.grad != 0.0).any())
                for parameter in policy.actor.parameters()
            )
        )
        self.assertEqual(float(policy.actor.gate_head.bias.grad[RECOVER_GATE]), 0.0)
        for module in (policy.stage1, policy.q1, policy.q2, policy.target_q1, policy.target_q2):
            self.assertTrue(all(parameter.grad is None for parameter in module.parameters()))
        self.assertIsNone(replay.z_rl.grad)

    def test_behavior_kl_config_validation_and_checkpoint_extra_roundtrip(self) -> None:
        invalid = (
            {"behavior_kl_weight": -1.0},
            {"behavior_kl_weight": float("nan")},
            {"behavior_kl_weight": 1.0},
            {"behavior_kl_weight": 1.0, "behavior_gate_prior": (0.5, 0.5, 0.0)},
            {"behavior_kl_weight": 1.0, "behavior_gate_prior": (0.5, 0.3, 0.1, 0.0)},
            {"behavior_kl_weight": 1.0, "behavior_gate_prior": (0.6, 0.5, -0.1, 0.0)},
            {"behavior_kl_weight": 1.0, "behavior_gate_prior": (float("nan"), 0.4, 0.1, 0.0)},
        )
        for values in invalid:
            with self.subTest(values=values):
                with self.assertRaises(NavigationRLTContractError):
                    Stage2LossConfig(**values)
        self.assertEqual(Stage2LossConfig().behavior_kl_weight, 0.0)
        self.assertIsNone(Stage2LossConfig().behavior_gate_prior)

        policy = NavigationRLTPolicy(_policy_config())
        policy.freeze_stage1()
        policy.set_gate_support_mask(torch.tensor([True, True, True, False]))
        counts = [9679, 8850, 242, 0]
        prior = policy.initialize_stage2_gate_prior_from_behavior_counts(counts)
        regularization = {
            "objective_schema": "proposal_conditioned_finite_candidate_cql_behavior_kl_v2",
            "direction": "behavior_to_actor_forward_kl",
            "behavior_kl_weight": 1.0,
            "behavior_gate_prior": prior.tolist(),
            "gate_counts": counts,
            "supported_uniform_mix": 0.05,
        }
        extra = {
            "training_phase": "stage2",
            "stage2_actor_regularization": regularization,
        }
        before_state_keys = set(policy.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "behavior_kl_policy.pt"
            policy.save_checkpoint(
                checkpoint,
                training_step=3,
                policy_version=2,
                extra_metadata=extra,
            )
            restored, metadata = NavigationRLTPolicy.load_checkpoint(checkpoint)
        self.assertEqual(metadata["extra"], extra)
        self.assertEqual(set(restored.state_dict()), before_state_keys)
        reconstructed_config = Stage2LossConfig(
            behavior_kl_weight=metadata["extra"]["stage2_actor_regularization"]["behavior_kl_weight"],
            behavior_gate_prior=tuple(
                metadata["extra"]["stage2_actor_regularization"]["behavior_gate_prior"]
            ),
        )
        self.assertEqual(reconstructed_config.behavior_kl_weight, 1.0)
        self.assertEqual(reconstructed_config.behavior_gate_prior, tuple(prior.tolist()))
        self.assertEqual(restored.gate_support_mask.tolist(), [True, True, True, False])

    def test_checkpoint_metadata_roundtrip(self) -> None:
        policy = NavigationRLTPolicy(_policy_config())
        policy.freeze_stage1()
        support = torch.tensor([True, True, True, False], dtype=torch.bool)
        policy.set_gate_support_mask(support)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "navigation_rlt.pt"
            receipt = policy.save_checkpoint(
                checkpoint,
                training_step=11,
                policy_version=3,
                extra_metadata={"protocol": "contract-test"},
            )
            loaded, metadata = NavigationRLTPolicy.load_checkpoint(checkpoint)
        self.assertEqual(metadata["training_step"], 11)
        self.assertEqual(metadata["policy_version"], 3)
        self.assertEqual(metadata["extra"], {"protocol": "contract-test"})
        self.assertEqual(receipt["metadata"], metadata)
        self.assertEqual(metadata["gate_support_mask"], support.tolist())
        self.assertEqual(metadata["supported_gate_names"], ["KEEP", "ADJUST", "STOP"])
        self.assertTrue(bool(loaded.stage1_frozen.item()))
        self.assertTrue(torch.equal(loaded.gate_support_mask.cpu(), support))
        prepared = loaded._prepare_compiler(
            _compiler(2), batch=2, device=torch.device("cpu"), dtype=torch.float32
        )
        distribution = loaded.actor_distribution(
            torch.zeros((2, 256), dtype=torch.float32),
            torch.zeros((2, loaded.config.reference_dim), dtype=torch.float32),
            prepared,
        )
        self.assertTrue(
            torch.equal(
                distribution["gate_probabilities"][:, RECOVER_GATE],
                torch.zeros(2, dtype=torch.float32),
            )
        )
        for name, value in policy.state_dict().items():
            self.assertTrue(torch.equal(value.cpu(), loaded.state_dict()[name].cpu()))

    def test_legacy_stage1_migration_is_exact_and_zero_initializes_reference_q(self) -> None:
        policy = NavigationRLTPolicy(_policy_config())
        policy.freeze_stage1()
        legacy = policy.checkpoint_state(
            training_step=9,
            policy_version=0,
            extra_metadata={"training_phase": "stage1"},
        )
        missing_reference = {
            "q1.reference_linear.weight",
            "q2.reference_linear.weight",
            "target_q1.reference_linear.weight",
            "target_q2.reference_linear.weight",
        }
        for name in missing_reference:
            legacy["model_state"].pop(name)
        legacy["metadata"].pop("gate_support_mask")
        legacy["metadata"].pop("supported_gate_names")
        legacy["metadata"]["cql_enabled"] = False
        legacy["metadata"]["stage2_objective"] = (
            "min_target_twin_q_with_native_reference_bc_and_dropout"
        )

        bad_legacy = copy.deepcopy(legacy)
        bad_legacy["model_state"].pop("actor.gate_head.bias")
        with tempfile.TemporaryDirectory() as directory:
            legacy_path = Path(directory) / "legacy-stage1.pt"
            bad_path = Path(directory) / "bad-legacy-stage1.pt"
            torch.save(legacy, legacy_path)
            torch.save(bad_legacy, bad_path)
            loaded, metadata = NavigationRLTPolicy.load_checkpoint(legacy_path)
            with self.assertRaises(NavigationRLTContractError):
                NavigationRLTPolicy.load_checkpoint(bad_path)

        self.assertEqual(
            metadata["load_migration"],
            "legacy_stage1_reference_linear_zero_and_gate_support_v1",
        )

        old_stage2 = dict(legacy)
        old_stage2["metadata"] = dict(legacy["metadata"])
        old_stage2["metadata"]["policy_version"] = 7
        old_stage2["metadata"]["extra"] = {"training_phase": "stage2"}
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "old-stage2.pt"
            torch.save(old_stage2, checkpoint)
            with self.assertRaisesRegex(
                NavigationRLTContractError,
                "restricted to frozen Stage-1",
            ):
                NavigationRLTPolicy.load_checkpoint(checkpoint)
        self.assertEqual(metadata["gate_support_mask"], [True, True, True, True])
        for name in missing_reference:
            self.assertTrue(torch.equal(loaded.state_dict()[name], torch.zeros_like(loaded.state_dict()[name])))


class NavigationRLTActionContractTests(unittest.TestCase):
    def test_keep_and_adjust_share_executor_bounded_native_base(self) -> None:
        modes = valid_mode_mask(native_stop=False, has_recovery_parent=False)
        raw_reference = (0.40, 0.02, 1.20)
        keep = compile_numpy_action(
            mode=KEEP,
            valid_modes=modes,
            reference_waypoint=raw_reference,
        )
        adjust = compile_numpy_action(
            mode=ADJUST,
            valid_modes=modes,
            reference_waypoint=raw_reference,
            raw_residual=(-10.0, -10.0),
        )
        self.assertTrue(
            np.allclose(
                keep.resolved_se2,
                (0.25, 0.02, np.deg2rad(30.0)),
                rtol=0.0,
                atol=1.0e-7,
            )
        )
        self.assertTrue(
            np.allclose(
                adjust.resolved_se2,
                (0.20, 0.02, np.deg2rad(20.0)),
                rtol=0.0,
                atol=1.0e-6,
            )
        )

    def test_stop_alias_is_rejected_and_recover_requires_causal_parent(self) -> None:
        native_stop_modes = valid_mode_mask(
            native_stop=True, has_recovery_parent=False
        )
        self.assertEqual(native_stop_modes, (False, False, True, False))
        with self.assertRaises(ActionContractError):
            compile_numpy_action(
                mode=KEEP,
                valid_modes=native_stop_modes,
                reference_waypoint=(0.0, 0.0, 0.0),
            )
        stop = compile_numpy_action(
            mode=STOP,
            valid_modes=native_stop_modes,
            reference_waypoint=(0.0, 0.0, 0.0),
        )
        self.assertEqual(stop.canonical_action, (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0))

        recover_modes = valid_mode_mask(
            native_stop=False, has_recovery_parent=True
        )
        with self.assertRaises(ActionContractError):
            compile_numpy_action(
                mode=RECOVER,
                valid_modes=recover_modes,
                reference_waypoint=(0.2, 0.0, 0.0),
            )
        recover = compile_numpy_action(
            mode=RECOVER,
            valid_modes=recover_modes,
            reference_waypoint=(0.2, 0.0, 0.0),
            recovery_parent_waypoint=(0.1, 0.1, 0.2),
            recovery_parent_index=2,
        )
        self.assertEqual(recover.recovery_parent_index, 2)
        self.assertEqual(recover.canonical_action[:4], (0.0, 0.0, 0.0, 1.0))
        self.assertGreaterEqual(len(recover.primitives), 1)
        self.assertLessEqual(len(recover.primitives), 4)


def _replay_metadata(*, episode: str, reset: str, version: int) -> dict[str, object]:
    return {
        "schema_name": REPLAY_SCHEMA_NAME,
        "schema_version": REPLAY_SCHEMA_VERSION,
        "run_id": "navigation-rlt-contract-run",
        "episode_id": episode,
        "reset_token": reset,
        "scene_id": "scene.glb",
        "instruction_sha256": "a" * 64,
        "split_label": "train_update",
        "env_slot": version,
        "row_count": 1,
        "lightnav_source_revision": LIGHTNAV_SOURCE_REVISION,
        "lightnav_hook_sha256": "b" * 64,
        "lightnav_weight_sha256": "c" * 64,
        "tokenizer_sha256": "d" * 64,
        "processor_sha256": "e" * 64,
        "history_config_sha256": "f" * 64,
        "prefix_feature_schema": "contextual_prefix_hidden_v1",
        "prefix_feature_source_sha256": "1" * 64,
        "topology_schema": "causal_topology_contract_v1",
        "reference_schema": REFERENCE_SCHEMA,
        "reference_dim": REFERENCE_DIM,
        "action_schema": "canonical7_v1",
        "executor_schema": "bounded_se2_v1",
        "representation_kind": REPRESENTATION_PREFIX_HIDDEN,
        "feature_stage": FEATURE_STAGE_PREFIX_HIDDEN,
        "topology_max_nodes": 2,
        "topology_node_dim": 3,
        "topology_global_dim": 4,
        "max_primitive_steps": 2,
        "discount_gamma": 0.9,
        "behavior_policy_id": "navigation-rlt-warmstart",
        "behavior_policy_version": version,
        "prefix_max_tokens": 3,
        "prefix_hidden_dim": 5,
    }


def _replay_arrays(*, version: int) -> dict[str, np.ndarray]:
    prefix = np.arange(15, dtype=np.float32).reshape(1, 3, 5) / 10.0
    prefix_mask = np.ones((1, 3), dtype=np.bool_)
    nodes = np.asarray([[[0.1, 0.0, 1.0], [0.2, 0.1, 2.0]]], dtype=np.float32)
    node_mask = np.ones((1, 2), dtype=np.bool_)
    global_targets = np.asarray([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    global_mask = np.ones((1, 4), dtype=np.bool_)
    trajectory = np.zeros((1, 10, 3), dtype=np.float32)
    trajectory[0, 0] = np.asarray([0.2, 0.0, 0.1], dtype=np.float32)
    reference = np.linspace(-1.0, 1.0, REFERENCE_DIM, dtype=np.float32)[None, :]
    state = {
        "prefix_hidden": prefix,
        "prefix_mask": prefix_mask,
        "topology_node_targets": nodes,
        "topology_node_mask": node_mask,
        "topology_global_targets": global_targets,
        "topology_global_mask": global_mask,
        "reference_embedding": reference,
        "reference_trajectory": trajectory,
        "reference_selected_index": np.asarray([0], dtype=np.int64),
        "native_stop": np.asarray([False], dtype=np.bool_),
        "valid_mode_mask": np.asarray([[True, True, True, False]], dtype=np.bool_),
        "recovery_parent_index": np.asarray([-1], dtype=np.int64),
    }
    arrays = dict(state)
    arrays.update({"next_" + name: value.copy() for name, value in state.items()})
    arrays.update(
        {
            "requested_mode": np.asarray([KEEP], dtype=np.int64),
            "executed_mode": np.asarray([KEEP], dtype=np.int64),
            "raw_residual": np.zeros((1, 2), dtype=np.float32),
            "bounded_residual": np.zeros((1, 2), dtype=np.float32),
            "resolved_se2": trajectory[:, 0].copy(),
            "safety_override": np.asarray([False], dtype=np.bool_),
            "credit_mask": np.asarray([True], dtype=np.bool_),
            "primitive_rewards": np.asarray([[0.5, 0.0]], dtype=np.float32),
            "executed_mask": np.asarray([[True, False]], dtype=np.bool_),
            "duration": np.asarray([1], dtype=np.int64),
            "discounted_reward": np.asarray([0.5], dtype=np.float32),
            "collision": np.asarray([False], dtype=np.bool_),
            "blocked": np.asarray([False], dtype=np.bool_),
            "terminated": np.asarray([True], dtype=np.bool_),
            "truncated": np.asarray([False], dtype=np.bool_),
            "done": np.asarray([True], dtype=np.bool_),
            "end_reason": np.asarray([END_REASON_NAMES.index("success")], dtype=np.int64),
            "decision_index": np.asarray([0], dtype=np.int64),
            "physical_step": np.asarray([0], dtype=np.int64),
            "behavior_policy_version": np.asarray([version], dtype=np.int64),
        }
    )
    return arrays


class NavigationRLTReplayContractTests(unittest.TestCase):
    def test_executor_bounded_keep_preserves_raw_reference_observation(self) -> None:
        arrays = _replay_arrays(version=1)
        arrays["reference_trajectory"][0, 0] = np.asarray(
            [0.40, 0.02, 1.20], dtype=np.float32
        )
        arrays["resolved_se2"][0] = np.asarray(
            [0.25, 0.02, np.deg2rad(30.0)], dtype=np.float32
        )
        with tempfile.TemporaryDirectory() as directory_name:
            path = Path(directory_name) / "bounded-keep.npz"
            save_episode_shard(
                path,
                arrays,
                _replay_metadata(episode="bounded-keep", reset="reset", version=1),
            )
            loaded, _ = load_episode_shard(path)
        self.assertTrue(
            np.array_equal(
                loaded["reference_trajectory"][0, 0],
                np.asarray([0.40, 0.02, 1.20], dtype=np.float32),
            )
        )

    def test_v3_atomic_roundtrip_source_tamper_and_mixed_versions(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            receipts = []
            paths = []
            for index, version in enumerate((1, 7)):
                path = directory / f"episode-{index}.npz"
                metadata = _replay_metadata(
                    episode=f"episode-{index}", reset=f"reset-{index}", version=version
                )
                receipt = save_episode_shard(path, _replay_arrays(version=version), metadata)
                self.assertFalse(any(directory.glob(path.name + ".staging-*")))
                loaded, loaded_metadata = load_episode_shard(
                    path,
                    expected_sha256=str(receipt["sha256"]),
                    expected_source_fingerprint=str(receipt["source_fingerprint"]),
                    expected_representation_kind=REPRESENTATION_PREFIX_HIDDEN,
                )
                self.assertEqual(loaded["reference_embedding"].shape, (1, 64))
                self.assertEqual(loaded_metadata["reference_dim"], 64)
                self.assertEqual(
                    loaded_metadata["source_fingerprint"], source_fingerprint(metadata)
                )
                changed_source = dict(metadata)
                changed_source["lightnav_weight_sha256"] = "2" * 64
                self.assertNotEqual(
                    source_fingerprint(metadata), source_fingerprint(changed_source)
                )
                receipts.append(receipt)
                paths.append(path)

            manifest = build_replay_manifest(
                receipts,
                run_id="navigation-rlt-contract-run",
                expected_episode_count=2,
                expected_source_fingerprint=str(receipts[0]["source_fingerprint"]),
            )
            self.assertEqual(manifest["behavior_policy_versions"], [1, 7])
            batch = concatenate_episode_shards(
                paths,
                expected_source_fingerprint=str(receipts[0]["source_fingerprint"]),
            )
            self.assertEqual(batch.rows, 2)
            self.assertEqual(batch.behavior_policy_versions, (1, 7))

            tampered = directory / "tampered.npz"
            tampered.write_bytes(paths[0].read_bytes() + b"tamper")
            with self.assertRaises(ReplayContractError):
                load_episode_shard(tampered, expected_sha256=str(receipts[0]["sha256"]))
            with self.assertRaises(FileExistsError):
                save_episode_shard(paths[0], _replay_arrays(version=1), _replay_metadata(episode="episode-0", reset="reset-0", version=1))


class NavigationRLTParallelContractTests(unittest.TestCase):
    def test_exact_worker_sharding_and_completion(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            episodes = [
                {
                    "episode_id": f"ep-{index}",
                    "scene": f"scene-{index % 2}",
                    "instruction_text_sha256": hashlib.sha256(
                        f"instruction-{index}".encode("utf-8")
                    ).hexdigest(),
                }
                for index in range(6)
            ]
            split_path = directory / "split.json"
            split_path.write_text(
                json.dumps(
                    {
                        "upstream_split_read": "train",
                        "validation_split_reads": 0,
                        "splits": {"train_update": {"episodes": episodes}},
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            split_sha = parallel_sha256_file(split_path)
            plan = build_plan(
                split_path,
                run_id="navigation-rlt-parallel-contract",
                expected_split_sha256=split_sha,
                worker_count=3,
                expected_episode_count=6,
            )
            plan_path = directory / "plan.json"
            atomic_json(plan_path, plan)
            plan_sha = parallel_sha256_file(plan_path)

            assigned = [
                (row["episode_id"], row["scene"], row["instruction_text_sha256"])
                for worker in plan["workers"]
                for row in worker["episodes"]
            ]
            expected = [
                (row["episode_id"], row["scene"], row["instruction_text_sha256"])
                for row in episodes
            ]
            self.assertEqual(len(assigned), len(set(assigned)))
            self.assertEqual(set(assigned), set(expected))
            self.assertEqual([worker["episode_count"] for worker in plan["workers"]], [2, 2, 2])

            completion_paths = []
            for worker in plan["workers"]:
                completion = {
                    "schema_name": WORKER_SCHEMA,
                    "schema_version": WORKER_VERSION,
                    "status": "completed",
                    "plan_sha256": plan_sha,
                    "run_id": plan["run_id"],
                    "worker_id": worker["worker_id"],
                    "episodes": worker["episodes"],
                    "replay_shards": [
                        {"episode_id": row["episode_id"], "sha256": "a" * 64}
                        for row in worker["episodes"]
                    ],
                }
                path = directory / f"worker-{worker['worker_id']}.json"
                atomic_json(path, completion)
                completion_paths.append(path)
            result = verify_completions(plan_path, completion_paths)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["worker_count"], 3)
            self.assertEqual(result["episode_count"], 6)
            self.assertEqual(len(result["replay_shards"]), 6)


if __name__ == "__main__":
    unittest.main()
