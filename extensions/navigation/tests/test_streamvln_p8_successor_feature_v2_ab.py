from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


RLINF_NAV_ROOT = Path(__file__).resolve().parents[1]
ROOT = RLINF_NAV_ROOT.parent
MODULE_DIR = RLINF_NAV_ROOT / "integrations" / "rlinf_v03"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import run_streamvln_p8_successor_feature_v2_ab as module


MANIFEST_PATH = MODULE_DIR / "streamvln_p8_successor_feature_v2_ab_manifest.json"
READINESS_PATH = MODULE_DIR / "streamvln_p8_successor_bc_readiness_manifest.json"
CAUSAL_PATH = MODULE_DIR / "streamvln_p8_successor_causal_probe_manifest.json"


def load_contract():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    readiness = json.loads(READINESS_PATH.read_text(encoding="utf-8"))
    causal = json.loads(CAUSAL_PATH.read_text(encoding="utf-8"))
    contract = module.validate_feature_v2_ab_manifest(
        manifest,
        readiness_manifest=readiness,
        causal_manifest=causal,
    )
    return manifest, contract


def nav(
    *, success: int, successful_stop: int = 1, forward: int = 30,
    left: int = 35, right: int = 35,
):
    return {
        "episode_count": 8,
        "success_count": success,
        "successful_stop_count": successful_stop,
        "timeout_count": 8 - success,
        "root_recovery_unavailable_count": 0,
        "stop_count": max(1, successful_stop),
        "sr": success / 8,
        "spl": success / 8,
        "mean_navigation_error_m": 1.0,
        "total_path_length_m": 10.0,
        "collision_count": 0.0,
        "action_counts": {
            "STOP": max(1, successful_stop),
            "FORWARD": forward,
            "LEFT": left,
            "RIGHT": right,
        },
        "route_counts": {
            "stop": 0,
            "keep_original": 20,
            "waypoint": 5,
        },
    }


def teacher_metrics(*, balanced: float, agreement: float):
    return {
        "decision_count": 20,
        "route_support": {"stop": 4, "keep_original": 8, "waypoint": 8},
        "route_nonstop_bal_acc": balanced,
        "stop": {"recall": 0.5, "false_positive": 1},
        "waypoint": {
            "first_action_agreement": agreement,
            "median_bearing_abs_error_deg": 15.0,
            "p90_bearing_abs_error_deg": 45.0,
            "median_vector_l2_m": 0.25,
        },
    }


def adjudication_payload():
    keep = nav(success=5)
    v1 = nav(success=2)
    v2 = nav(success=4)
    v2.update(
        {
            "accepted_override_count": 4,
            "accepted_override_episode_count": 2,
            "successful_episode_with_accepted_override_count": 1,
            "recover": {
                "eligible_blocked_count": 1,
                "covered_next_decision_count": 1,
                "recover_parent_count": 1,
                "alignment_miss_count": 0,
                "root_recovery_unavailable_count": 0,
                "root_keep_fallback_count": 1,
            },
        }
    )
    return {
        "teacher_forced": {
            module.ARM_V1: {"metrics": teacher_metrics(balanced=0.55, agreement=0.55)},
            module.ARM_V2: {"metrics": teacher_metrics(balanced=0.65, agreement=0.65)},
        },
        "navigation": {
            "matched_keep": keep,
            module.ARM_V1: v1,
            module.ARM_V2: v2,
        },
    }


class FakeABProvider:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def execute(self, **_kwargs):
        self.calls += 1
        return copy.deepcopy(self.payload)


class FeatureV2ABTests(unittest.TestCase):
    def test_manifest_freezes_authorized_matched_protocol_and_full_closure(self):
        manifest, contract = load_contract()
        self.assertEqual(len(contract["train"]), 8)
        self.assertEqual(len(contract["development"]), 8)
        self.assertEqual(
            {row["deployment_relative_path"] for row in contract["files"]},
            set(module.REQUIRED_IMPLEMENTATION_DEPLOYMENTS),
        )
        self.assertEqual(contract["protocol"]["train_reads"], 48)
        self.assertEqual(contract["training"]["updates_per_arm"], 2000)
        self.assertFalse(manifest["claim_boundary"]["bc_promotion"])
        self.assertFalse(manifest["claim_boundary"]["ppo"])
        self.assertTrue(manifest["lineage"]["causal_authorize_feature_v2"])

        changed = copy.deepcopy(manifest)
        changed["thresholds"]["teacher_route"][
            "feature_v2_delta_vs_v1_min"
        ] = 0.09
        with self.assertRaises(module.ProtocolError):
            module.validate_feature_v2_ab_manifest(
                changed,
                readiness_manifest=json.loads(READINESS_PATH.read_text()),
                causal_manifest=json.loads(CAUSAL_PATH.read_text()),
            )

    def test_literal_split_rejection_precedes_every_path_and_provider_access(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            output = base / "must_not_exist.json"
            args = argparse.Namespace(
                stage="feature_v2_ab",
                split="val_unseen",
                manifest=base / "missing-manifest.json",
                readiness_manifest=base / "missing-readiness.json",
                causal_manifest=base / "missing-causal.json",
                causal_receipt=base / "missing-receipt.json",
                release_manifest=base / "missing-release.json",
                rlinf_root=base / "missing-rlinf",
                source_root=base / "missing-source",
                checkpoint=base / "missing-checkpoint",
                tokenizer=base / "missing-tokenizer",
                vision=base / "missing-vision",
                asset_root=base / "missing-assets",
                device="cuda:0",
                output=output,
            )
            provider = FakeABProvider({})
            with self.assertRaises(ValueError):
                module.run(
                    args,
                    preflight_provider=lambda *_args: self.fail(
                        "preflight ran before split rejection"
                    ),
                    ab_provider=provider,
                )
            self.assertEqual(provider.calls, 0)
            self.assertFalse(output.exists())

    def test_frozen_adjudication_passes_exact_boundaries_and_fails_one_miss(self):
        payload = adjudication_payload()
        gate = module.adjudicate_feature_v2_ab(
            payload, thresholds=module.EXPECTED_THRESHOLDS
        )
        self.assertTrue(gate["passed"])
        self.assertTrue(all(gate["checks"].values()))
        self.assertEqual(
            gate["closed_loop_observed"]["feature_v2_root_keep_fallback_count"],
            1,
        )

        failed = copy.deepcopy(payload)
        failed["navigation"][module.ARM_V2]["recover"][
            "alignment_miss_count"
        ] = 1
        gate = module.adjudicate_feature_v2_ab(
            failed, thresholds=module.EXPECTED_THRESHOLDS
        )
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["checks"]["feature_v2_recover_alignment"])

    def test_implementation_hash_drift_is_fail_closed(self):
        manifest, _contract = load_contract()
        with tempfile.TemporaryDirectory() as directory:
            deploy_root = Path(directory)
            for row in manifest["implementation"]["files"]:
                source_relative = Path(row["source_relative_path"])
                source_relative = source_relative.relative_to("rlinf-nav")
                source = (RLINF_NAV_ROOT / source_relative).resolve(strict=True)
                self.assertTrue(source.is_relative_to(RLINF_NAV_ROOT))
                target = deploy_root / row["deployment_relative_path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            self.assertEqual(
                len(module.verify_implementation(manifest, rlinf_root=deploy_root)),
                len(module.REQUIRED_IMPLEMENTATION_DEPLOYMENTS),
            )
            target = deploy_root / manifest["implementation"]["files"][0][
                "deployment_relative_path"
            ]
            target.write_bytes(target.read_bytes() + b"drift")
            with self.assertRaises(module.ProtocolError):
                module.verify_implementation(manifest, rlinf_root=deploy_root)

    def test_current_depth_provenance_rejects_each_mismatch(self):
        class FakeTensor:
            pass

        frame = SimpleNamespace(
            frame_ref="sha256:" + "1" * 64,
            depth_sha256="2" * 64,
            physical_step=0,
            episode_generation="3" * 32,
        )
        v1 = SimpleNamespace(
            feature=FakeTensor(),
            physical_step=0,
            episode_generation=frame.episode_generation,
            frame_ref=frame.frame_ref,
            reset_ordinal=0,
            generation_ordinal=1,
            feature_audit={"sha256": "4" * 64},
        )
        v2 = SimpleNamespace(
            contextual_hidden=FakeTensor(),
            physical_step=0,
            episode_generation=frame.episode_generation,
            frame_ref=frame.frame_ref,
            reset_ordinal=0,
            generation_ordinal=0,
            contextual_hidden_audit={"sha256": "5" * 64},
        )
        provider = object.__new__(module.CanonicalFeatureV2ABProvider)
        provider.adapter = SimpleNamespace(
            latest_feature_v2_snapshot=lambda: v2,
            extract_frozen_feature_snapshot=lambda **_kwargs: v1,
            tensor_auditor=lambda _tensor: {"sha256": "6" * 64},
            reset_ordinal=0,
            generation_ordinal=1,
        )
        provider._v1_latch = None
        provider._v1_latch_key = None
        provider._feature_pair_counts = {
            "v1_latch_updates": 0,
            "decision_states": 0,
            "fresh_context_states": 0,
            "stale_context_states": 0,
        }
        provider._pretraining_feature_guard = None
        provider._optimizer_steps = 0
        good_source = {
            "frame_ref": frame.frame_ref,
            "depth_sha256": frame.depth_sha256,
            "physical_step": frame.physical_step,
        }
        provider.adapter.reset_ordinal = 1
        with self.assertRaises(module.ProtocolError):
            provider._paired_features(
                frame=frame,
                instruction="go",
                official_shadow_action=1,
                physical_step=0,
            )
        provider.adapter.reset_ordinal = 0
        v2.episode_generation = "7" * 32
        with self.assertRaises(module.ProtocolError):
            provider._paired_features(
                frame=frame,
                instruction="go",
                official_shadow_action=1,
                physical_step=0,
            )
        v2.episode_generation = frame.episode_generation
        provider.adapter.generation_ordinal = 2
        with self.assertRaises(module.ProtocolError):
            provider._paired_features(
                frame=frame,
                instruction="go",
                official_shadow_action=1,
                physical_step=0,
            )
        provider.adapter.generation_ordinal = 1
        cases = (
            {**good_source, "frame_ref": "sha256:" + "9" * 64},
            {**good_source, "depth_sha256": "9" * 64},
            {**good_source, "physical_step": 1},
        )
        for source in cases:
            with self.subTest(source=source):
                provider._v1_latch = None
                provider._v1_latch_key = None
                provider._current_depth_summary = lambda _frame, source=source: (
                    FakeTensor(),
                    {"sha256": "6" * 64},
                    source,
                )
                with self.assertRaises(module.ProtocolError):
                    provider._paired_features(
                        frame=frame,
                        instruction="go",
                        official_shadow_action=1,
                        physical_step=0,
                    )
        provider._v1_latch = None
        provider._v1_latch_key = None
        provider._current_depth_summary = lambda _frame: (
            FakeTensor(),
            {"sha256": "8" * 64},
            good_source,
        )
        with self.assertRaises(module.ProtocolError):
            provider._paired_features(
                frame=frame,
                instruction="go",
                official_shadow_action=1,
                physical_step=0,
            )

    def test_fake_provider_writes_self_hashed_two_candidate_receipt(self):
        manifest, contract = load_contract()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            candidates = {}
            for arm, version in ((module.ARM_V1, 1), (module.ARM_V2, 2)):
                path = base / f"{arm}.pt"
                path.write_bytes(f"candidate-{arm}".encode())
                candidates[arm] = {
                    "path": str(path),
                    "bc_ready": 0,
                    "feature_schema_binding": {
                        "successor_feature_schema_version": version,
                        "successor_feature_schema_sha256": (
                            module.EXPECTED_FEATURE_SCHEMA_DIGESTS[version]
                        ),
                    },
                    "policy_state_sha256": str(version) * 64,
                    "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            episodes = [
                {"episode_id": row["episode_id"]}
                for row in contract["development"]
            ]
            payload = adjudication_payload()
            payload.update(
                {
                    "candidates": candidates,
                    "candidate_bc_ready": {
                        module.ARM_V1: 0,
                        module.ARM_V2: 0,
                    },
                    "paired_dataset_sha256": "a" * 64,
                    "train_label_counts": {
                        "stop": 4,
                        "keep_original": 8,
                        "waypoint": 8,
                    },
                    "split_read_counts": {
                        "train": 48,
                        "val_seen": 0,
                        "val_unseen": 0,
                    },
                    "training": {
                        arm: {
                            "warmup": {
                                "updates": 500,
                                "dataset_sha256": "b" * 64,
                            },
                            "final": {
                                "updates": 1500,
                                "dataset_sha256": "a" * 64,
                            },
                        }
                        for arm in module.ARMS
                    },
                    "pretraining_feature_guard": {
                        "passed": True,
                        "optimizer_steps_before_guard": 0,
                        "adapter_feature_v2_opt_in": True,
                        "default_adapter_feature_v2_opt_in": False,
                        "feature_v1_dim": 7173,
                        "feature_v2_dim": 3604,
                        "feature_v2_schema_version": 2,
                        "same_frame_ref": True,
                        "same_physical_step": True,
                        "same_reset_ordinal": True,
                        "same_episode_generation": True,
                        "both_fresh": True,
                        "feature_v2_schema_sha256": (
                            module.EXPECTED_FEATURE_SCHEMA_DIGESTS[2]
                        ),
                        "contextual_hidden_sha256": "c" * 64,
                    },
                    "feature_pair_counts": {
                        "decision_states": 20,
                        "fresh_context_states": 10,
                        "stale_context_states": 10,
                        "v1_latch_updates": 10,
                    },
                    "value_head_frozen": True,
                    "guards": {
                        "model_load_count": 1,
                        "single_process": True,
                        "train_only": True,
                        "optimizer_steps_per_arm": 2000,
                        "optimizer_steps_total": 4000,
                        "ppo_updates": 0,
                        "promotion_attempts": 0,
                        "retune_attempts": 0,
                    },
                    "collection": {},
                    "episodes": {
                        "matched_keep": copy.deepcopy(episodes),
                        module.ARM_V1: copy.deepcopy(episodes),
                        module.ARM_V2: copy.deepcopy(episodes),
                    },
                }
            )
            for arm in module.ARMS:
                payload["teacher_forced"][arm]["episodes"] = copy.deepcopy(
                    episodes
                )
                payload["navigation"][arm].update(
                    {
                        "accepted_override_count": (
                            4 if arm == module.ARM_V2 else 0
                        ),
                        "accepted_override_episode_count": (
                            2 if arm == module.ARM_V2 else 0
                        ),
                        "successful_episode_with_accepted_override_count": (
                            1 if arm == module.ARM_V2 else 0
                        ),
                        "recover": {
                            "eligible_blocked_count": 0,
                            "covered_next_decision_count": 0,
                            "recover_parent_count": 0,
                            "alignment_miss_count": 0,
                            "root_recovery_unavailable_count": 0,
                            "root_keep_fallback_count": 0,
                        },
                    }
                )
            output = base / "receipt.json"
            args = argparse.Namespace(
                stage="feature_v2_ab",
                split="train",
                manifest=MANIFEST_PATH,
                readiness_manifest=READINESS_PATH,
                causal_manifest=CAUSAL_PATH,
                causal_receipt=base / "unused-causal.json",
                release_manifest=base / "unused-release.json",
                rlinf_root=RLINF_NAV_ROOT,
                source_root=base / "unused-source",
                checkpoint=base / "unused-checkpoint",
                tokenizer=base / "unused-tokenizer",
                vision=base / "unused-vision",
                asset_root=base / "unused-assets",
                device="cuda:0",
                output=output,
            )
            provider = FakeABProvider(payload)
            with mock.patch.object(
                module,
                "verify_implementation",
                return_value=manifest["implementation"]["files"],
            ):
                receipt = module.run(
                    args,
                    preflight_provider=lambda *_args: {"status": "pass"},
                    ab_provider=provider,
                )
            self.assertEqual(provider.calls, 1)
            self.assertTrue(receipt["gate_pass"])
            self.assertIsNone(receipt["promotion"])
            loaded = json.loads(output.read_text(encoding="utf-8"))
            digest = loaded.pop("receipt_sha256")
            self.assertEqual(digest, module.readiness._json_hash(loaded))
            self.assertEqual(set(receipt["result"]["candidates"]), set(module.ARMS))
            bad = copy.deepcopy(receipt["result"])
            bad["episodes"]["matched_keep"][1] = copy.deepcopy(
                bad["episodes"]["matched_keep"][0]
            )
            with self.assertRaises(module.ProtocolError):
                module.validate_feature_v2_ab_payload(bad, contract)


if __name__ == "__main__":
    unittest.main()
