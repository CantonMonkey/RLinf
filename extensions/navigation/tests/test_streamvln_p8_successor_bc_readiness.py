import argparse
from collections import Counter
from enum import Enum
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


MODULE_DIR = Path(__file__).resolve().parents[1] / "integrations" / "rlinf_v03"
SPEC = importlib.util.spec_from_file_location(
    "run_streamvln_p8_successor_bc_readiness",
    MODULE_DIR / "run_streamvln_p8_successor_bc_readiness.py",
)
BC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BC)
PRODUCTION_MANIFEST = MODULE_DIR / "streamvln_p8_successor_bc_readiness_manifest.json"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def signed_receipt(stage, *, expert_file_sha=None, ceiling="a" * 64, release="b" * 64):
    value = {
        "stage": stage,
        "gate_pass": True,
        "ceiling_manifest_raw_sha256": ceiling,
        "release_manifest_raw_sha256": release,
        "result": {"stage": stage, "split": "train"},
        "preflight": {},
    }
    if stage == "keep":
        value["preflight"]["expert_receipt"] = {"file_sha256": expert_file_sha}
    value["receipt_sha256"] = BC._json_hash(value)
    return value


def metrics(*, success, successful_stop, routes=None, root_recovery_unavailable=0):
    return {
        "episode_count": 8,
        "success_count": success,
        "successful_stop_count": successful_stop,
        "timeout_count": 8 - success,
        "root_recovery_unavailable_count": root_recovery_unavailable,
        "stop_count": max(1, successful_stop),
        "sr": success / 8,
        "spl": success / 8,
        "mean_navigation_error_m": 2.0,
        "total_path_length_m": 18.0,
        "collision_count": 2,
        "action_counts": {"STOP": 2, "FORWARD": 20, "LEFT": 3, "RIGHT": 0},
        "route_counts": routes or {"stop": 2, "keep_original": 8, "waypoint": 7},
    }


def passing_payload(candidate):
    return {
        "candidate_checkpoint": str(candidate),
        "candidate_bc_ready": 0,
        "optimizer_steps": 2000,
        "value_head_frozen": True,
        "development_evaluation_count": 1,
        "train_label_counts": {"stop": 3, "keep_original": 50, "waypoint": 60},
        "split_read_counts": {"train": 32, "val_seen": 0, "val_unseen": 0},
        "matched_keep": metrics(success=2, successful_stop=2),
        "learned": metrics(success=1, successful_stop=1),
        "learned_episode_route_counts": [
            {"stop": 1, "keep_original": 2, "waypoint": 2} for _ in range(8)
        ],
        "recover": {
            "eligible_blocked_count": 2,
            "covered_next_decision_count": 2,
            "recover_parent_count": 1,
            "alignment_miss_count": 0,
            "root_recovery_unavailable_count": 0,
        },
    }


class FakeRouteName(Enum):
    STOP = "stop"
    KEEP_ORIGINAL = "keep_original"
    WAYPOINT = "waypoint"
    RECOVER_PARENT = "recover_parent"


class FakeParentRecoveryUnavailable(RuntimeError):
    pass


class FakeRouteDecision:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeActionModule:
    STOP_ACTION = 0
    RouteName = FakeRouteName
    RouteDecision = FakeRouteDecision
    ParentRecoveryUnavailable = FakeParentRecoveryUnavailable

    @staticmethod
    def compile_metric_waypoint(_waypoint):
        return (1,)


class FakeEpisodeEnv:
    def __init__(self):
        self.identity = {"instruction_text": "go forward"}
        self.closed = False

    def expert_waypoint(self, _lookahead):
        return {
            "stop": False,
            "waypoint_local_m": [1.0, 0.0],
            "distance_to_goal_m": 4.5,
        }

    def close(self):
        self.closed = True


class FakeEpisodeBridge:
    def __init__(self, failure_type):
        self.failure_type = failure_type
        self.episode_token = "episode-token"
        self._next_decision_index = 0
        self.topology = SimpleNamespace(
            consecutive_blocked_forwards=2,
            current_node=0,
            nodes=[SimpleNamespace(parent=None)],
        )

    def reset_episode(self):
        return None

    def prepare_decision_context(self, context_builder):
        context = context_builder(SimpleNamespace(physical_step=0), 1)
        return SimpleNamespace(policy_context=context, physical_step=0)

    def execute_decision(self, _decision):
        raise self.failure_type("root recovery unavailable")


class FakeProvider:
    def __init__(self, root, mutate=None):
        self.root = root
        self.mutate = mutate
        self.promotions = []

    def execute(self, **_kwargs):
        candidate = self.root / "candidate.pt"
        candidate.write_text(json.dumps({"bc_ready": 0}), encoding="utf-8")
        value = passing_payload(candidate)
        if self.mutate:
            self.mutate(value)
        return value

    def promote(self, *, evidence_digest, candidate_path, promoted_path):
        value = {
            "bc_ready": 1,
            "evidence_digest": evidence_digest,
            "candidate": str(candidate_path),
        }
        promoted_path.write_text(json.dumps(value), encoding="utf-8")
        self.promotions.append(value)
        return {"bc_ready": 1, "path": str(promoted_path), "evidence_digest": evidence_digest}


class ReadinessContractTests(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads(PRODUCTION_MANIFEST.read_text(encoding="utf-8"))

    def fake_preflight(self, _args, manifest, _contract):
        return {
            "status": "pass",
            "manifest_sha256": BC._json_hash(manifest),
            "predecessors": {
                "expert": {"receipt_sha256": "e" * 64},
                "keep": {"receipt_sha256": "k" * 64},
            },
            "guards": {"model_load_count": 0},
        }

    def make_args(self, root, *, split="train", stage="readiness"):
        manifest = root / "manifest.json"
        write_json(manifest, self.manifest)
        return argparse.Namespace(
            stage=stage,
            split=split,
            manifest=manifest,
            release_manifest=root / "release.json",
            rlinf_root=root / "rlinf",
            source_root=root / "source",
            checkpoint=root / "checkpoint",
            tokenizer=root / "tokenizer",
            vision=root / "vision",
            asset_root=root / "assets",
            expert_receipt=root / "expert.json",
            keep_receipt=root / "keep.json",
            device="cuda:0",
            output=root / "receipt.json",
        )

    def make_episode_provider(self, failure_type=FakeParentRecoveryUnavailable):
        provider = object.__new__(BC.CanonicalReadinessProvider)
        env = FakeEpisodeEnv()
        bridge = FakeEpisodeBridge(failure_type)
        provider.runtime = SimpleNamespace(
            make_successor_env=lambda *_args, **_kwargs: env,
            SuccessorRunnerBridge=lambda **_kwargs: bridge,
        )
        provider.action = FakeActionModule
        provider.contract = {"seed": 1}
        provider.manifest = {
            "architecture": {"topology_max_nodes": 4},
            "executor": {
                "expert_lookahead_m": 1.0,
                "blocked_forward_threshold": 2,
            },
        }
        provider.topology_module = SimpleNamespace(
            OnlineTopology=lambda **_kwargs: object(),
            snapshot_online_topology=lambda *_args, **_kwargs: object(),
        )
        context = {"features": object()}
        provider.policy_module = SimpleNamespace(
            make_successor_forward_context=lambda **_kwargs: context,
        )
        provider.adapter = object()
        provider.official_module = object()
        provider.ceiling = SimpleNamespace(
            _terminal_metrics=lambda *_args: (_ for _ in ()).throw(
                AssertionError("root recovery failure must use current-state metrics")
            )
        )
        provider._split_reads = Counter({"train": 0, "val_seen": 0, "val_unseen": 0})
        provider._run_cfg = lambda: {}
        provider._identity_matches = lambda _identity, _spec: True
        provider._ensure_policy = lambda _feature, _topology: None

        def learned_sample(frame, _shadow, _context, **_kwargs):
            return SimpleNamespace(
                decision=FakeRouteDecision(
                    decision_index=frame.decision_index,
                    route=FakeRouteName.KEEP_ORIGINAL,
                    waypoint_local_m=(0.0, 0.0),
                    episode_token=frame.episode_token,
                    physical_step=frame.physical_step,
                )
            )

        provider._learned_sample = learned_sample
        return provider, env

    def test_production_manifest_freezes_8_plus_8_and_training_budget(self):
        contract = BC.validate_protocol_manifest(self.manifest)
        self.assertEqual(len(contract["train"]), 8)
        self.assertEqual(len(contract["development"]), 8)
        self.assertTrue(
            {row["scene"] for row in contract["train"]}.isdisjoint(
                {row["scene"] for row in contract["development"]}
            )
        )
        self.assertEqual(contract["training"]["warmup_updates"], 500)
        self.assertEqual(contract["training"]["final_updates"], 1500)
        self.assertEqual(contract["training"]["batch_size"], 32)
        self.assertEqual(
            self.manifest["runtime"]["model_python"], BC.REQUIRED_MODEL_PYTHON
        )

    def test_nontrain_rejected_before_manifest_or_provider_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.make_args(root, split="val_seen")
            args.manifest.unlink()
            calls = []

            def provider(*_args):
                calls.append(True)
                raise AssertionError("must not run")

            with self.assertRaisesRegex(ValueError, "literal train"):
                BC.run(args, preflight_provider=provider, readiness_provider=provider)
            self.assertEqual(calls, [])
            self.assertFalse(args.output.exists())

    def test_label_rule_stop_then_keep_match_else_waypoint(self):
        self.assertEqual(
            BC.label_successor_route(
                expert_stop=True, official_shadow_action=1, expert_first_primitive=2
            ),
            "stop",
        )
        self.assertEqual(
            BC.label_successor_route(
                expert_stop=False, official_shadow_action=2, expert_first_primitive=2
            ),
            "keep_original",
        )
        self.assertEqual(
            BC.label_successor_route(
                expert_stop=False, official_shadow_action=1, expert_first_primitive=2
            ),
            "waypoint",
        )

    def test_duplicate_state_conflict_fails_closed(self):
        digest = "c" * 64
        with self.assertRaisesRegex(BC.ProtocolError, "conflicting labels"):
            BC.deduplicate_rows(
                [
                    {"state_hash": digest, "route_label": "stop", "waypoint_target": [0, 0], "expert_first_primitive": 0},
                    {"state_hash": digest, "route_label": "waypoint", "waypoint_target": [1, 0], "expert_first_primitive": 1},
                ]
            )

    def test_inverse_frequency_requires_all_three_labels(self):
        rows = [
            {"state_hash": character * 64, "route_label": route, "waypoint_target": [0, 0], "expert_first_primitive": index}
            for index, (character, route) in enumerate((("a", "stop"), ("b", "keep_original"), ("c", "waypoint")))
        ]
        weights = BC.inverse_frequency_weights(rows)
        self.assertEqual(weights, {"stop": 1.0, "keep_original": 1.0, "waypoint": 1.0})
        with self.assertRaisesRegex(BC.ProtocolError, "all three"):
            BC.inverse_frequency_weights(rows[:2])

    def test_gate_passes_only_complete_train_only_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            payload = passing_payload(Path(temporary) / "candidate.pt")
            gate = BC.adjudicate_readiness(payload, gate=self.manifest["gate"])
            self.assertTrue(gate["passed"])
            self.assertTrue(all(gate["checks"].values()))

            payload["split_read_counts"]["val_unseen"] = 1
            negative = BC.adjudicate_readiness(payload, gate=self.manifest["gate"])
            self.assertFalse(negative["passed"])
            self.assertFalse(negative["checks"]["train_only"])

    def test_gate_requires_every_blocked_next_decision_and_real_recover(self):
        with tempfile.TemporaryDirectory() as temporary:
            payload = passing_payload(Path(temporary) / "candidate.pt")
            payload["recover"]["covered_next_decision_count"] = 1
            payload["recover"]["recover_parent_count"] = 0
            gate = BC.adjudicate_readiness(payload, gate=self.manifest["gate"])
            self.assertFalse(gate["passed"])
            self.assertFalse(gate["checks"]["recover_coverage"])
            self.assertFalse(gate["checks"]["real_recover_parent"])

    def test_development_root_recovery_failure_is_recorded_without_fake_execution(self):
        spec = {"index": 7, "episode_id": "8", "scene": "scene"}
        patches = (
            mock.patch.object(
                BC,
                "extract_policy_feature",
                return_value=(object(), SimpleNamespace(feature_audit={})),
            ),
            mock.patch.object(BC, "_tensor_mapping_hash", return_value="a" * 64),
            mock.patch.object(BC, "state_sha256", return_value="b" * 64),
        )
        with patches[0], patches[1], patches[2]:
            for mode in ("keep", "learned"):
                with self.subTest(mode=mode):
                    provider, env = self.make_episode_provider()
                    episode = provider._run_episode(
                        spec=spec,
                        mode=mode,
                        decision_budget=2,
                        physical_budget=8,
                        collect_rows=False,
                    )
                    self.assertTrue(env.closed)
                    self.assertFalse(episode["terminated"])
                    self.assertTrue(episode["truncated"])
                    self.assertEqual(
                        episode["failure_reason"], "root_recovery_unavailable"
                    )
                    self.assertEqual(
                        episode["terminal_metrics_source"],
                        "current_state_at_root_recovery_failure",
                    )
                    self.assertEqual(
                        episode["recover"],
                        {
                            "eligible_blocked_count": 0,
                            "covered_next_decision_count": 0,
                            "recover_parent_count": 0,
                            "alignment_miss_count": 0,
                            "root_recovery_unavailable_count": 1,
                        },
                    )
                    self.assertEqual(episode["executed_actions"], [])
                    self.assertTrue(all(value == 0 for value in episode["route_counts"].values()))
                    self.assertTrue(all(value == 0 for value in episode["action_counts"].values()))
                    self.assertFalse(episode["successful_stop"])
                    self.assertEqual(
                        episode["metrics"],
                        {
                            "success": 0.0,
                            "spl": 0.0,
                            "path_length": 0.0,
                            "collision_count": 0.0,
                            "distance_to_goal": 4.5,
                        },
                    )

                    aggregate = BC.CanonicalReadinessProvider._aggregate([episode] * 8)
                    self.assertEqual(aggregate["root_recovery_unavailable_count"], 8)
                    self.assertEqual(aggregate["timeout_count"], 0)

    def test_training_modes_and_unrelated_runtime_errors_still_fail_closed(self):
        spec = {"index": 7, "episode_id": "8", "scene": "scene"}
        patches = (
            mock.patch.object(
                BC,
                "extract_policy_feature",
                return_value=(object(), SimpleNamespace(feature_audit={})),
            ),
            mock.patch.object(BC, "_tensor_mapping_hash", return_value="a" * 64),
            mock.patch.object(BC, "state_sha256", return_value="b" * 64),
        )
        with patches[0], patches[1], patches[2]:
            for mode in ("expert", "dagger"):
                with self.subTest(mode=mode):
                    provider, env = self.make_episode_provider()
                    with self.assertRaises(FakeParentRecoveryUnavailable):
                        provider._run_episode(
                            spec=spec,
                            mode=mode,
                            decision_budget=2,
                            physical_budget=8,
                            collect_rows=False,
                        )
                    self.assertTrue(env.closed)

            provider, env = self.make_episode_provider(RuntimeError)
            with self.assertRaisesRegex(RuntimeError, "root recovery unavailable"):
                provider._run_episode(
                    spec=spec,
                    mode="keep",
                    decision_budget=2,
                    physical_budget=8,
                    collect_rows=False,
                )
            self.assertTrue(env.closed)

    def test_root_recovery_integrity_failure_does_not_change_other_gate_rules(self):
        with tempfile.TemporaryDirectory() as temporary:
            payload = passing_payload(Path(temporary) / "candidate.pt")
            payload["matched_keep"]["root_recovery_unavailable_count"] = 1
            payload["recover"]["root_recovery_unavailable_count"] = 1
            gate = BC.adjudicate_readiness(payload, gate=self.manifest["gate"])
            self.assertFalse(gate["passed"])
            self.assertFalse(gate["checks"]["root_recovery_available"])
            self.assertTrue(
                all(
                    value
                    for name, value in gate["checks"].items()
                    if name != "root_recovery_available"
                )
            )
            self.assertEqual(
                gate["matched_keep"]["root_recovery_unavailable_count"], 1
            )
            self.assertEqual(
                gate["recover"]["root_recovery_unavailable_count"], 1
            )

            del payload["learned"]["root_recovery_unavailable_count"]
            with self.assertRaisesRegex(
                BC.ProtocolError, "learned.root_recovery_unavailable_count"
            ):
                BC.adjudicate_readiness(payload, gate=self.manifest["gate"])

    def test_predecessors_bind_keep_to_exact_expert_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expert_path = root / "expert.json"
            write_json(expert_path, signed_receipt("expert"))
            keep_path = root / "keep.json"
            write_json(
                keep_path,
                signed_receipt("keep", expert_file_sha=BC._file_sha256(expert_path)),
            )
            observed = BC.validate_predecessors(expert_path, keep_path)
            self.assertEqual(observed["expert"]["file_sha256"], BC._file_sha256(expert_path))

            tampered = signed_receipt("keep", expert_file_sha="d" * 64)
            keep_path.unlink()
            write_json(keep_path, tampered)
            with self.assertRaisesRegex(BC.ProtocolError, "another expert"):
                BC.validate_predecessors(expert_path, keep_path)

    def test_pass_promotes_once_with_evidence_digest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.make_args(root)
            provider = FakeProvider(root)
            receipt = BC.run(
                args,
                preflight_provider=self.fake_preflight,
                readiness_provider=provider,
            )
            self.assertTrue(receipt["gate_pass"])
            self.assertEqual(len(provider.promotions), 1)
            self.assertEqual(
                provider.promotions[0]["evidence_digest"], receipt["evidence_digest"]
            )
            self.assertEqual(receipt["promotion"]["bc_ready"], 1)
            evidence = root / "bc_readiness_evidence.json"
            self.assertEqual(BC._file_sha256(evidence), receipt["evidence_digest"])
            self.assertEqual(
                receipt["readiness_evidence"]["file_sha256"],
                receipt["evidence_digest"],
            )
            self.assertTrue(args.output.exists())

    def test_failed_gate_saves_candidate_and_receipt_without_promotion(self):
        def fail_label_coverage(value):
            value["train_label_counts"]["waypoint"] = 0

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.make_args(root)
            provider = FakeProvider(root, fail_label_coverage)
            receipt = BC.run(
                args,
                preflight_provider=self.fake_preflight,
                readiness_provider=provider,
            )
            self.assertFalse(receipt["gate_pass"])
            self.assertEqual(provider.promotions, [])
            self.assertIsNone(receipt["promotion"])
            self.assertTrue(Path(receipt["result"]["candidate_checkpoint"]).is_file())
            self.assertTrue(args.output.is_file())

    def test_passing_gate_without_promoter_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.make_args(root)

            def provider(**_kwargs):
                candidate = root / "candidate.pt"
                candidate.write_text("candidate")
                return passing_payload(candidate)

            with self.assertRaisesRegex(BC.ProtocolError, "promotion"):
                BC.run(
                    args,
                    preflight_provider=self.fake_preflight,
                    readiness_provider=provider,
                )
            self.assertFalse(args.output.exists())

    def test_cli_has_no_seed_budget_or_episode_override(self):
        destinations = {action.dest for action in BC.build_parser()._actions}
        for forbidden in (
            "seed",
            "episode_index",
            "max_steps",
            "warmup_updates",
            "final_updates",
            "batch_size",
        ):
            self.assertNotIn(forbidden, destinations)


if __name__ == "__main__":
    unittest.main()
