"""Dependency-injected A1 tests for frozen StreamVLN context plus S2 hybrid RL.

These tests never load the released 8B checkpoint or Habitat.  They validate
the integration contract that a later bounded A800 runner can exercise with the
real provider: temporal latching, synchronized STOP, replay provenance,
gradient isolation, and head-only checkpoint recovery.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
from pathlib import Path
import struct
from types import SimpleNamespace
import tempfile
import unittest

from navrl.stack import EpisodeIdentity, load_checkpoint_verified, save_checkpoint_atomic


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn

    from integrations.rlinf_v03.streamvln_contextual_topology_waypoint import (
        CONTEXTUAL_FEATURE_DIM,
        ExternalShadowStopRequired,
        StreamVLNContextualFeatureProvider,
        StreamVLNContextualHeadCheckpointAdapter,
        StreamVLNContextualTopologyWaypointPolicy,
        StreamVLNContextualTopologyWaypointPolicyAdapter,
    )
    from integrations.rlinf_v03.topology_waypoint_stack import (
        EpisodeTopologyState,
        _tensor_context_digest,
    )


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


if TORCH_AVAILABLE:

    def _tensor_sha(value: torch.Tensor) -> str:
        values = tuple(float(item) for item in value.detach().cpu().reshape(-1).tolist())
        return hashlib.sha256(struct.pack(f"<{len(values)}f", *values)).hexdigest()


    class FrozenFakeBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.tensor([1.0]), requires_grad=False)


    class FakeGenerationAdapter:
        def __init__(self):
            self.model = FrozenFakeBackbone()
            self.reset_ordinal = 0
            self.generation_ordinal = 0
            self._snapshot = None
            self.tensor_auditor = lambda tensor: {"sha256": _tensor_sha(tensor)}

        def latch(self, frame):
            ordinal = self.generation_ordinal
            hidden = torch.full(
                (3584,), float(ordinal + 1), dtype=torch.float32
            ).contiguous()
            self._snapshot = SimpleNamespace(
                contextual_hidden=hidden,
                depth_summary=torch.empty(0),
                physical_step=frame.physical_step,
                episode_generation=frame.episode_generation,
                frame_ref=frame.frame_ref,
                depth_sha256=frame.depth_sha256,
                reset_ordinal=self.reset_ordinal,
                generation_ordinal=ordinal,
                prefill_length=11 + ordinal,
                past_key_values_ref=_sha(f"kv-{ordinal}"),
                inputs_embeds_ref=_sha(f"embeds-{ordinal}"),
            )
            self.generation_ordinal += 1

        def latest_feature_v2_snapshot(self):
            if self._snapshot is None:
                raise RuntimeError("no fake contextual latch")
            snapshot = self._snapshot
            return SimpleNamespace(
                contextual_hidden=snapshot.contextual_hidden.detach().clone(),
                depth_summary=snapshot.depth_summary,
                physical_step=snapshot.physical_step,
                episode_generation=snapshot.episode_generation,
                frame_ref=snapshot.frame_ref,
                depth_sha256=snapshot.depth_sha256,
                reset_ordinal=snapshot.reset_ordinal,
                generation_ordinal=snapshot.generation_ordinal,
                prefill_length=snapshot.prefill_length,
                past_key_values_ref=snapshot.past_key_values_ref,
                inputs_embeds_ref=snapshot.inputs_embeds_ref,
            )


    class FakeOfficialDriver:
        def __init__(self, adapter, schedule):
            self.adapter = adapter
            self.schedule = dict(schedule)
            self.session = SimpleNamespace(physical_step=0)
            self._pending_action = None

        def prepare_shadow_action(self, frame):
            step = self.session.physical_step
            if frame.physical_step != step or self._pending_action is not None:
                raise RuntimeError("fake official cadence drifted")
            action, generate = self.schedule[step]
            if generate:
                self.adapter.latch(frame)
            self._pending_action = int(action)
            return int(action)

        def acknowledge_physical_step(self, *, terminated, truncated):
            if self._pending_action is None:
                raise RuntimeError("fake official action was not prepared")
            completed = self.session.physical_step
            action = self._pending_action
            self._pending_action = None
            self.session.physical_step += 1
            return {
                "completed_step": completed,
                "next_step": self.session.physical_step,
                "completed_action": action,
                "terminal": bool(terminated or truncated or action == 0),
            }


    def _frame(step: int):
        return SimpleNamespace(
            physical_step=step,
            episode_generation="1" * 32,
            frame_ref="sha256:" + _sha(f"frame-{step}"),
            depth_sha256=_sha(f"depth-{step}"),
        )


    def _depth_builder(frame):
        summary = torch.tensor(
            [value for _ in range(5) for value in (1.0, 2.0, 1.0)],
            dtype=torch.float32,
        ).contiguous()
        audit = {"sha256": _tensor_sha(summary)}
        source = {
            "frame_ref": frame.frame_ref,
            "depth_sha256": frame.depth_sha256,
            "physical_step": frame.physical_step,
        }
        return summary, audit, source


    def _provider(schedule):
        adapter = FakeGenerationAdapter()
        driver = FakeOfficialDriver(adapter, schedule)
        provider = StreamVLNContextualFeatureProvider(
            adapter,
            driver,
            depth_summary_builder=_depth_builder,
        )
        return provider, adapter, driver


    def _policy(release="a" * 64):
        return StreamVLNContextualTopologyWaypointPolicy(
            release_manifest_sha256=release,
            hidden_dim=32,
            topology_max_nodes=8,
            topology_model_dim=8,
            residual_limit_m=0.2,
        )


    def _state(max_decisions=4):
        return EpisodeTopologyState(
            max_decisions=max_decisions,
            topology_max_nodes=8,
            residual_limit_m=0.2,
        )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for contextual integration tests")
class StreamVLNContextualFeatureProviderTests(unittest.TestCase):
    def test_fresh_stale_and_mid_macro_generation_boundary(self):
        provider, _adapter, _driver = _provider(
            {
                0: (1, True),
                1: (2, False),
                2: (1, True),
                3: (2, False),
            }
        )

        first = provider.prepare(_frame(0))
        self.assertTrue(first.causal_current)
        self.assertEqual(first.context_age, 0)
        self.assertEqual(first.generation_ordinal, 0)
        provider.acknowledge(terminated=False, truncated=False)

        stale = provider.prepare(_frame(1))
        self.assertFalse(stale.causal_current)
        self.assertEqual(stale.context_age, 1)
        self.assertEqual(stale.context_frame_ref, first.context_frame_ref)
        provider.acknowledge(terminated=False, truncated=False)

        # A new official generation may occur inside a variable-duration macro.
        refreshed = provider.prepare(_frame(2))
        self.assertTrue(refreshed.causal_current)
        self.assertEqual(refreshed.generation_ordinal, 1)
        self.assertEqual(refreshed.context_frame_ref, _frame(2).frame_ref)
        provider.acknowledge(terminated=False, truncated=False)

        stale_again = provider.prepare(_frame(3))
        self.assertEqual(stale_again.context_age, 1)
        self.assertEqual(stale_again.generation_ordinal, 1)
        self.assertEqual(stale_again.features.numel(), CONTEXTUAL_FEATURE_DIM)

    def test_shadow_stop_bypasses_actor_credit_and_requires_synchronized_termination(self):
        provider, _adapter, _driver = _provider({0: (0, True)})
        feature = provider.prepare(_frame(0))
        self.assertTrue(feature.force_stop)

        policy = _policy()
        state = _state(max_decisions=1)
        actor_forward_count = 0

        def actor_forward(*, operation, **kwargs):
            nonlocal actor_forward_count
            actor_forward_count += 1
            return policy(operation=operation, **kwargs)

        adapter = StreamVLNContextualTopologyWaypointPolicyAdapter(
            policy,
            state,
            deterministic=False,
            root_forward=actor_forward,
        )
        identity = EpisodeIdentity("train-stop", "reset-stop")
        adapter.reset_state(identity)
        observation = feature.to_policy_observation()
        with self.assertRaisesRegex(ExternalShadowStopRequired, "outside the actor"):
            adapter.sample(observation, identity, decision_index=0, physical_step=0)
        with self.assertRaisesRegex(ExternalShadowStopRequired, "may not be bootstrapped"):
            adapter.bootstrap(observation, identity, physical_step=0)
        self.assertEqual(actor_forward_count, 0)

        with self.assertRaisesRegex(RuntimeError, "synchronized STOP"):
            provider.acknowledge(terminated=False, truncated=False)
        event = provider.acknowledge(terminated=True, truncated=False)
        self.assertTrue(event["terminal"])

    def test_provider_reset_rejects_pending_or_nonzero_driver(self):
        provider, adapter, _driver = _provider({0: (1, True), 1: (1, False)})
        provider.prepare(_frame(0))
        with self.assertRaisesRegex(RuntimeError, "unacknowledged"):
            provider.reset()
        provider.acknowledge(terminated=False, truncated=False)
        with self.assertRaisesRegex(RuntimeError, "step-zero"):
            provider.reset()

        adapter.reset_ordinal += 1
        replacement = FakeOfficialDriver(adapter, {0: (1, True)})
        provider.reset(replacement)
        self.assertEqual(provider.provider_generation, 2)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for contextual integration tests")
class StreamVLNContextualPolicyTests(unittest.TestCase):
    def _sample(self):
        provider, backbone, _driver = _provider({0: (1, True)})
        feature = provider.prepare(_frame(0))
        policy = _policy()
        state = _state()
        adapter = StreamVLNContextualTopologyWaypointPolicyAdapter(
            policy, state, deterministic=True
        )
        identity = EpisodeIdentity("train-context", "reset-context")
        adapter.reset_state(identity)
        sample = adapter.sample(feature.to_policy_observation(), identity, 0, 0)
        return provider, backbone, policy, state, adapter, identity, sample

    def test_replay_payload_binds_feature_provenance_and_reset(self):
        _provider_value, _backbone, _policy_value, _state_value, adapter, identity, sample = (
            self._sample()
        )
        adapter._validate_bound_sample(sample)

        context = dict(sample.replay_context)
        forward = dict(context["forward_inputs"])
        forward["streamvln_generation_ordinal"] = torch.tensor([9], dtype=torch.int64)
        context["forward_inputs"] = forward
        # Repair the broad context digest to prove the independent provenance
        # binding still rejects the changed generation identity.
        context["forward_inputs_digest"] = _tensor_context_digest(forward)
        tampered = replace(sample, replay_context=context)
        with self.assertRaisesRegex(ValueError, "feature provenance"):
            adapter._validate_bound_sample(tampered)

        stop_context = dict(sample.replay_context)
        stop_forward = dict(stop_context["forward_inputs"])
        stop_forward["streamvln_force_stop"] = torch.tensor([True], dtype=torch.bool)
        stop_context["forward_inputs"] = stop_forward
        stop_context["forward_inputs_digest"] = _tensor_context_digest(stop_forward)
        stop_replay = replace(sample, replay_context=stop_context)
        with self.assertRaisesRegex(ValueError, "external shadow STOP"):
            adapter._validate_bound_sample(stop_replay)

        adapter.reset_state(EpisodeIdentity("train-context", "reset-context-2"))
        with self.assertRaisesRegex(RuntimeError, "stale or unbound"):
            adapter._validate_bound_sample(sample)
        self.assertEqual(identity.episode_id, "train-context")

    def test_frozen_backbone_has_no_grad_while_context_head_trains(self):
        provider, backbone, policy, state, adapter, identity, _sample_value = self._sample()
        observation = provider.pending.to_policy_observation()
        snapshot = state.snapshot(identity, decision_index=0, physical_step=0)
        forward = policy.prepare_forward_inputs(
            observation, snapshot, policy_version=adapter.policy_version
        )
        target_waypoint = torch.tensor([[0.75, 0.0]], dtype=torch.float32)
        losses = policy(
            operation="bc_loss",
            forward_inputs=forward,
            target_candidate=torch.tensor([0], dtype=torch.int64),
            target_waypoint=target_waypoint,
        )
        losses["loss"].backward()

        projection_gradients = [
            parameter.grad
            for name, parameter in policy.named_parameters()
            if name.startswith("observation_encoder.projection")
        ]
        self.assertTrue(any(gradient is not None for gradient in projection_gradients))
        self.assertTrue(
            any(
                gradient is not None and float(gradient.detach().abs().sum()) > 0.0
                for gradient in projection_gradients
            )
        )
        log_std_gradient = policy.residual_log_std.grad
        self.assertIsNotNone(log_std_gradient)
        self.assertTrue(bool(torch.isfinite(log_std_gradient).all()))
        self.assertGreater(float(log_std_gradient.detach().abs().sum()), 0.0)
        self.assertTrue(bool(torch.isfinite(losses["gaussian_nll"])))
        self.assertTrue(all(not parameter.requires_grad for parameter in backbone.model.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in backbone.model.parameters()))
        self.assertFalse(observation["streamvln_features"].requires_grad)
        trainable = adapter.trainable_parameter_names()
        for required in (
            "observation_encoder.projection",
            "topology_encoder",
            "context_fusion",
            "candidate_scorer",
            "candidate_slot_bias",
            "residual_mean_head",
            "residual_log_std",
            "value_head",
        ):
            self.assertTrue(any(required in name for name in trainable), required)

    def test_bc_stop_has_no_gaussian_supervision(self):
        provider, _backbone, _driver = _provider({0: (1, True)})
        feature = provider.prepare(_frame(0))
        policy = _policy()
        state = _state()
        adapter = StreamVLNContextualTopologyWaypointPolicyAdapter(
            policy, state, deterministic=True
        )
        identity = EpisodeIdentity("train-bc-stop", "reset-bc-stop")
        adapter.reset_state(identity)
        snapshot = state.snapshot(identity, decision_index=0, physical_step=0)
        forward = policy.prepare_forward_inputs(
            feature.to_policy_observation(),
            snapshot,
            policy_version=adapter.policy_version,
        )
        losses = policy(
            operation="bc_loss",
            forward_inputs=forward,
            target_candidate=torch.tensor(
                [policy.max_candidates - 1], dtype=torch.int64
            ),
            target_waypoint=torch.zeros(1, 2, dtype=torch.float32),
        )
        self.assertEqual(float(losses["gaussian_nll"].detach()), 0.0)
        self.assertEqual(tuple(losses["raw_waypoint_target"].shape), (0, 2))
        losses["loss"].backward()
        self.assertIsNone(policy.residual_log_std.grad)

        with self.assertRaisesRegex(ValueError, "must be finite"):
            policy(
                operation="bc_loss",
                forward_inputs=forward,
                target_candidate=torch.tensor([0], dtype=torch.int64),
                target_waypoint=torch.tensor([[float("nan"), 0.0]]),
            )

    def test_head_only_checkpoint_round_trip_and_release_binding(self):
        _provider_value, _backbone, policy, _state_value, adapter, identity, _sample_value = (
            self._sample()
        )
        with torch.no_grad():
            next(iter(policy.parameters())).add_(0.125)
        adapter.mark_updated()
        expected = {
            name: value.detach().cpu().clone() for name, value in policy.state_dict().items()
        }

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "contextual-head"
            saved = save_checkpoint_atomic(
                StreamVLNContextualHeadCheckpointAdapter(adapter), checkpoint
            )

            restored_policy = _policy()
            restored_adapter = StreamVLNContextualTopologyWaypointPolicyAdapter(
                restored_policy, _state(), deterministic=True
            )
            restored_adapter.reset_state(identity)
            loaded = load_checkpoint_verified(
                StreamVLNContextualHeadCheckpointAdapter(restored_adapter), checkpoint
            )
            self.assertEqual(saved.files_sha256, loaded.files_sha256)
            self.assertEqual(restored_adapter.policy_version, adapter.policy_version)
            for name, value in restored_policy.state_dict().items():
                self.assertTrue(torch.equal(value.detach().cpu(), expected[name]), name)

            incompatible_policy = _policy(release="b" * 64)
            incompatible_adapter = StreamVLNContextualTopologyWaypointPolicyAdapter(
                incompatible_policy, _state(), deterministic=True
            )
            before = {
                name: value.detach().cpu().clone()
                for name, value in incompatible_policy.state_dict().items()
            }
            with self.assertRaisesRegex(RuntimeError, "compatibility differs"):
                load_checkpoint_verified(
                    StreamVLNContextualHeadCheckpointAdapter(incompatible_adapter),
                    checkpoint,
                )
            for name, value in incompatible_policy.state_dict().items():
                self.assertTrue(torch.equal(value.detach().cpu(), before[name]), name)


if __name__ == "__main__":
    unittest.main()
