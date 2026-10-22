"""Pure-Torch contract tests for the A1 RLinf/FSDP model root."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
RLINF_FSDP_AVAILABLE = False

if TORCH_AVAILABLE:
    import torch

    from navrl.stack import EpisodeIdentity
    from integrations.rlinf_v03.streamvln_a1_fsdp_model import (
        A1_OPERATION_NAMES,
        A1_TRAINABLE_GROUPS,
        RLINF_FSDP_AVAILABLE,
        build_frozen_reference_model,
        build_policy_adapter,
        create_fsdp_manager,
        get_model,
        hybrid_reference_kl,
        manager_config_from_manifest,
        model_config_from_manifest,
        root_forward_audit,
        unwrap_a1_model,
    )
    from integrations.rlinf_v03.streamvln_contextual_topology_waypoint import (
        CONTEXTUAL_FEATURE_SCHEMA_SHA256,
        ContextualFeatureObservation,
        ExternalShadowStopRequired,
    )
    from integrations.rlinf_v03.streamvln_p8_successor_policy import (
        audit_successor_features_v2,
        compose_successor_features_v2,
    )
    from integrations.rlinf_v03.topology_waypoint_stack import EpisodeTopologyState


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


if TORCH_AVAILABLE:

    def _receipt(*, action: int = 1) -> ContextualFeatureObservation:
        hidden = torch.linspace(-1.0, 1.0, 3584, dtype=torch.float32)
        depth = torch.tensor(
            [value for _ in range(5) for value in (1.0, 2.0, 1.0)],
            dtype=torch.float32,
        )
        features = compose_successor_features_v2(
            contextual_hidden=hidden,
            depth_summary=depth,
            official_shadow_action=action,
            physical_step=0,
            context_physical_step=0,
        )
        audit = audit_successor_features_v2(features)
        return ContextualFeatureObservation(
            features=features,
            schema_sha256=CONTEXTUAL_FEATURE_SCHEMA_SHA256,
            feature_sha256=audit["feature_sha256"],
            context_frame_ref="sha256:" + _sha("context-frame"),
            current_frame_ref="sha256:" + _sha("current-frame"),
            current_depth_sha256=_sha("depth"),
            episode_generation="1" * 32,
            past_key_values_ref=_sha("kv"),
            inputs_embeds_ref=_sha("embeds"),
            reset_ordinal=1,
            generation_ordinal=0,
            prefill_length=12,
            context_physical_step=0,
            current_physical_step=0,
            context_age=0,
            official_shadow_action=action,
            force_stop=action == 0,
        )


    def _manifest() -> dict:
        path = (
            Path(__file__).resolve().parents[1]
            / "integrations"
            / "rlinf_v03"
            / "streamvln_a1_contextual_hybrid_manifest.json"
        )
        return json.loads(path.read_text(encoding="utf-8"))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for A1 FSDP glue tests")
class StreamVLNA1FSDPModelTest(unittest.TestCase):
    def setUp(self):
        self.manifest = _manifest()
        self.config = model_config_from_manifest(
            self.manifest, release_manifest_sha256="a" * 64
        )
        self.model = get_model(self.config)
        architecture = self.manifest["architecture"]
        self.state = EpisodeTopologyState(
            max_decisions=4,
            topology_max_nodes=architecture["topology_max_nodes"],
            residual_limit_m=architecture["residual_limit_m"],
        )
        self.adapter = build_policy_adapter(
            self.model,
            self.state,
            deterministic=True,
        )
        self.identity = EpisodeIdentity("a1-train", "a1-reset")
        self.adapter.reset_state(self.identity)

    def test_all_four_operations_use_batch_one_root_and_are_counted(self):
        observation = _receipt().to_policy_observation()
        sample = self.adapter.sample(observation, self.identity, 0, 0)
        forward = sample.replay_context["forward_inputs"]
        slot = sample.action_payload[2]
        raw = sample.action_payload[4:6]

        replay = self.model(
            operation="replay",
            forward_inputs=forward,
            candidate_action=torch.tensor([slot], dtype=torch.int64),
            raw_action=torch.tensor([raw], dtype=torch.float32),
        )
        self.assertEqual(tuple(replay["route_logprob"].shape), (1,))
        bootstrap = self.adapter.bootstrap(observation, self.identity, 0)
        self.assertTrue(bool(torch.isfinite(bootstrap.value)))

        anchor = forward["candidate_anchors"][:, 0].clone()
        bc = self.model(
            operation="bc_loss",
            forward_inputs=forward,
            target_candidate=torch.tensor([0], dtype=torch.int64),
            target_waypoint=anchor,
        )
        bc["loss"].backward()
        self.assertIsNotNone(self.model.policy.residual_log_std.grad)

        audit = self.model.operation_audit()
        self.assertEqual(audit["batch_size"], 1)
        self.assertEqual(
            audit["operations"],
            {
                "sample": 1,
                "replay": 1,
                "bootstrap": 1,
                "bc_loss": 1,
                "distribution": 0,
            },
        )
        self.assertEqual(audit["manager_model_forward_count"], 4)

        batched = dict(forward)
        batched["streamvln_features"] = forward["streamvln_features"].repeat(2, 1)
        with self.assertRaisesRegex(ValueError, "batch size one"):
            self.model(
                operation="bootstrap",
                forward_inputs=batched,
            )
        self.assertEqual(self.model.operation_audit()["operations"]["bootstrap"], 1)

    def test_external_shadow_stop_never_reaches_root_forward(self):
        stop_observation = _receipt(action=0).to_policy_observation()
        with self.assertRaisesRegex(ExternalShadowStopRequired, "outside the actor"):
            self.adapter.sample(stop_observation, self.identity, 0, 0)
        self.assertEqual(
            self.model.operation_audit()["manager_model_forward_count"], 0
        )

    def test_head_state_and_trainable_groups_are_exact(self):
        manifest = self.model.trainable_manifest()
        self.assertFalse(manifest["backbone_in_fsdp_root"])
        self.assertEqual(tuple(manifest["trainable_groups"]), A1_TRAINABLE_GROUPS)
        self.assertEqual(len(manifest["trainable_names"]), 40)
        self.assertEqual(set(self.model.state_dict()), {
            f"policy.{name}" for name in self.model.head_state_dict()
        })

        before = self.model.head_state_dict()
        with torch.no_grad():
            self.model.policy.candidate_slot_bias.add_(0.25)
        self.assertEqual(
            self.model.changed_trainable_groups(before), ("candidate_slot_bias",)
        )
        self.model.load_head_state_dict(before)
        for name, value in self.model.head_state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)

        broken = dict(before)
        broken.pop(next(iter(broken)))
        with self.assertRaisesRegex(RuntimeError, "schema differs"):
            self.model.load_head_state_dict(broken)

    def test_frozen_bc_reference_exposes_exact_differentiable_hybrid_kl(self):
        reference = build_frozen_reference_model(
            self.config, self.model.head_state_dict()
        )
        self.assertTrue(all(not parameter.requires_grad for parameter in reference.parameters()))
        observation = _receipt().to_policy_observation()
        snapshot = self.state.snapshot(self.identity, 0, 0)
        forward = self.model.policy.prepare_forward_inputs(
            observation,
            snapshot,
            policy_version=self.adapter.policy_version,
        )
        with torch.no_grad():
            reference_distribution = reference(
                operation="distribution", forward_inputs=forward
            )
            unchanged = self.model(
                operation="distribution", forward_inputs=forward
            )
        self.assertEqual(float(hybrid_reference_kl(reference_distribution, unchanged)), 0.0)

        with torch.no_grad():
            self.model.policy.candidate_slot_bias[0].add_(0.2)
            self.model.policy.residual_mean_head[-1].bias.add_(0.1)
        current = self.model(operation="distribution", forward_inputs=forward)
        kl = hybrid_reference_kl(reference_distribution, current)
        self.assertGreater(float(kl.detach()), 0.0)
        kl.mean().backward()
        self.assertIsNotNone(self.model.policy.candidate_slot_bias.grad)
        self.assertTrue(bool(torch.isfinite(self.model.policy.candidate_slot_bias.grad).all()))

    def test_manifest_builds_g5_g8_compatible_manager_config(self):
        cfg = manager_config_from_manifest(
            self.manifest,
            release_manifest_sha256="b" * 64,
            total_training_steps=8,
        )

        def read(value, key):
            return value[key] if isinstance(value, dict) else getattr(value, key)

        fsdp = read(cfg, "fsdp_config")
        self.assertEqual(read(fsdp, "sharding_strategy"), "no_shard")
        self.assertTrue(read(fsdp, "use_orig_params"))
        self.assertTrue(read(read(fsdp, "wrap_policy"), "disable"))
        self.assertEqual(
            read(read(cfg, "model"), "feature_schema_sha256"),
            CONTEXTUAL_FEATURE_SCHEMA_SHA256,
        )
        self.assertIs(unwrap_a1_model(self.model), self.model)
        self.assertFalse(root_forward_audit(self.model)["fsdp_wrapped"])
        self.assertEqual(
            set(A1_OPERATION_NAMES),
            {"sample", "replay", "bootstrap", "bc_loss", "distribution"},
        )

    @unittest.skipIf(RLINF_FSDP_AVAILABLE, "missing-RLinf guard only")
    def test_manager_factory_fails_explicitly_without_rlinf(self):
        cfg = manager_config_from_manifest(
            self.manifest,
            release_manifest_sha256="c" * 64,
            total_training_steps=1,
        )
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            create_fsdp_manager(cfg)


if __name__ == "__main__":
    unittest.main()
