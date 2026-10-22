from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest


RLINF_NAV_ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = RLINF_NAV_ROOT / "integrations" / "rlinf_v03"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import run_streamvln_a1_contextual_hybrid as module
import streamvln_a1_live_provider as live_provider


MANIFEST_PATH = MODULE_DIR / "streamvln_a1_contextual_hybrid_manifest.json"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class A1ContextualHybridContractTest(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        self.contract = module.validate_manifest(self.manifest)
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.bc_checkpoint = root / "bc.pt"
        self.ppo_checkpoint = root / "ppo.pt"
        self.bc_checkpoint.write_bytes(b"same-bc-initialization")
        self.ppo_checkpoint.write_bytes(b"ppo-updated-head")

    def tearDown(self):
        self.temporary.cleanup()

    def _episodes(self, *, successes: int, improved: bool = False):
        rows = []
        for index, identity in enumerate(self.contract["development"]):
            success = index < successes
            if success:
                stop_source = "policy" if index % 2 == 0 else "official_shadow"
            else:
                stop_source = "none"
            receipts = []
            physical_step = 0
            normal_durations = (4, 4, 4) if stop_source == "official_shadow" else (4, 4, 4, 1)
            for decision_index, duration in enumerate(normal_durations):
                receipts.append(
                    {
                        "decision_index": decision_index,
                        "physical_step_start": physical_step,
                        "proposed_primitive_count": duration,
                        "executed_primitive_count": duration,
                        "official_shadow_stop_phase": "none",
                        "actor_sampled": True,
                        "actor_credit_eligible": True,
                        "topology_receipt_applied": True,
                    }
                )
                physical_step += duration
            primitive_counts = {"FORWARD": 10, "LEFT": 2, "RIGHT": 1}
            if stop_source == "official_shadow":
                receipts.append(
                    {
                        "decision_index": len(receipts),
                        "physical_step_start": physical_step,
                        "proposed_primitive_count": 1,
                        "executed_primitive_count": 1,
                        "official_shadow_stop_phase": "predecision",
                        "actor_sampled": False,
                        "actor_credit_eligible": False,
                        "topology_receipt_applied": False,
                    }
                )
                primitive_counts = {"FORWARD": 10, "LEFT": 2, "STOP": 1}
            rows.append(
                {
                    **identity,
                    "success": success,
                    "spl": 0.6 if success else 0.0,
                    "initial_distance_m": 6.0,
                    "final_navigation_error_m": (
                        0.5 if success and improved else 1.0 if success else 3.0 if improved else 4.5
                    ),
                    "timeout": not success,
                    "stop_source": stop_source,
                    "successful_stop": success,
                    "executed_trace_sha256": _sha(
                        f"{'ppo' if improved else 'bc'}-{index}".encode()
                    ),
                    "candidate_counts": {"frontier:forward": 2, "stop": 1},
                    "primitive_counts": primitive_counts,
                    "waypoint_macro_count": 1 if index < 5 else 0,
                    "path_length_m": 3.0,
                    "collision_count": 0,
                    "blocked_forward_count": 0,
                    "episode_return": 2.0 if success else -1.0,
                    "geodesic_progress_m": 3.0 if success else 1.5,
                    "post_shadow_stop_primitive_count": 0,
                    "external_stop_actor_credit_count": 0,
                    "decision_receipts": receipts,
                }
            )
        return {"episodes": rows}

    def _source(self):
        return {
            "streamvln_commit": self.manifest["source_contract"]["streamvln_commit"],
            "feature_provider": "StreamVLNContextualFeatureProvider",
            "feature_dim": 3604,
            "feature_schema_sha256": self.manifest["frozen_prior"]["schema_sha256"],
            "feature_v2_enabled": True,
            "frozen_prior": True,
            "ttt_or_secondary_loaded": False,
            "model_load_count": 1,
            "official_shadow_stop_mode": "synchronized_physical_boundary_override",
            "split_reads": {"train": 24, "val_seen": 0, "val_unseen": 0},
        }

    def _one_receipt(self):
        return {
            "decision_index": 0,
            "physical_step_start": 0,
            "proposed_primitive_count": 1,
            "executed_primitive_count": 1,
            "official_shadow_stop_phase": "none",
            "actor_sampled": True,
            "actor_credit_eligible": True,
            "topology_receipt_applied": True,
        }

    def _sync(self, receipt_groups):
        rows = [row for group in receipt_groups for row in group]
        executed = sum(row["executed_primitive_count"] for row in rows)
        return {
            "executed_primitive_count": executed,
            "physical_boundary_ack_count": executed,
            "post_shadow_stop_primitive_count": 0,
            "external_stop_actor_credit_count": 0,
            "sampled_after_predecision_shadow_stop_count": 0,
            "predecision_shadow_stop_count": 0,
            "mid_macro_shadow_stop_count": 0,
        }

    def _bc_training(self):
        required = [
            value
            for value in self.contract["required_trainable_groups"]
            if value != "value_head"
        ]
        traversals = []
        receipt_groups = []
        for traversal_index in range(16):
            identity = self.contract["train"][traversal_index % 8]
            receipts = [self._one_receipt()]
            traversals.append(
                {
                    **identity,
                    "pass": "expert" if traversal_index < 8 else "dagger",
                    "traversal_index": traversal_index,
                    "decision_receipts": receipts,
                }
            )
            receipt_groups.append(receipts)
        return {
            "checkpoint": {
                "path": str(self.bc_checkpoint),
                "sha256": module._artifact_sha256(self.bc_checkpoint),
                "restore_exact": True,
            },
            "supervision_steps": 1000,
            "episode_identities": [dict(row) for row in self.contract["train"]],
            "traversals": traversals,
            "objective": copy.deepcopy(self.manifest["bc_objective"]),
            "label_support": {"stop": 4, "non_stop": 32, "waypoint": 24},
            "finite_losses": True,
            "finite_gradients": True,
            "frozen_prior_unchanged": True,
            "frozen_stack_core_unchanged": True,
            "changed_parameter_groups": required,
            "official_stop_sync": self._sync(receipt_groups),
        }

    def _ppo_training(self):
        episode_rollouts = []
        receipt_groups = []
        for update_index, identity in enumerate(self.contract["train"]):
            receipts = [self._one_receipt()]
            episode_rollouts.append(
                {
                    **identity,
                    "update_index": update_index,
                    "actor_record_count": 1,
                    "dropped_external_decision_count": 0,
                    "decision_receipts": receipts,
                }
            )
            receipt_groups.append(receipts)
        return {
            "checkpoint": {
                "path": str(self.ppo_checkpoint),
                "sha256": module._artifact_sha256(self.ppo_checkpoint),
                "restore_exact": True,
            },
            "initialization_checkpoint_sha256": module._artifact_sha256(
                self.bc_checkpoint
            ),
            "finite_losses": True,
            "finite_gradients": True,
            "frozen_prior_unchanged": True,
            "frozen_stack_core_unchanged": True,
            "changed_parameter_groups": list(
                self.contract["required_trainable_groups"]
            ),
            "official_stop_sync": self._sync(receipt_groups),
            "updates": [
                {
                    "pre_update_ratio_max_abs_deviation": 0.0,
                    "bc_reference_kl": 0.02,
                    "reward": copy.deepcopy(self.manifest["reward"]),
                    "objective": copy.deepcopy(self.manifest["ppo"]),
                }
                for _ in range(8)
            ],
            "root_forward": {
                "path": "manager.model",
                "fsdp_wrapped": True,
                "manager_model_forward_count": 24,
                "local_policy_callback_forward_count": 0,
                "operations": {"sample": 8, "replay": 8, "bootstrap": 8},
            },
            "episode_identities": [dict(row) for row in self.contract["train"]],
            "episode_rollouts": episode_rollouts,
        }

    def _positive_payload(self):
        bc_eval = self._episodes(successes=2)
        return {
            "source_audit": self._source(),
            "bc_training": self._bc_training(),
            "bc_evaluation": bc_eval,
            "ppo_training": self._ppo_training(),
            "matched_evaluation": {
                "bc": copy.deepcopy(bc_eval),
                "bc_ppo": self._episodes(successes=3, improved=True),
            },
        }

    def test_manifest_reuses_exact_p8_train_and_scene_disjoint_development(self):
        lineage = module.validate_lineage(
            self.manifest, self.contract, root=RLINF_NAV_ROOT
        )
        self.assertRegex(lineage["episode_identity_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(len(self.contract["train"]), 8)
        self.assertEqual(len(self.contract["development"]), 8)
        self.assertFalse(
            {row["scene"] for row in self.contract["train"]}
            & {row["scene"] for row in self.contract["development"]}
        )
        self.assertEqual(self.manifest["dataset"]["allowed_splits"], ["train"])

    def test_gate_a_failure_stops_without_ppo(self):
        payload = {
            "source_audit": self._source(),
            "bc_training": self._bc_training(),
            "bc_evaluation": self._episodes(successes=0),
            "ppo_training": None,
            "matched_evaluation": None,
        }
        result = module.validate_payload(payload, self.manifest, self.contract)
        self.assertEqual(result["status"], "bc_competence_failed")
        self.assertFalse(result["ppo_started"])

    def test_gate_a_failure_rejects_any_ppo_evidence(self):
        payload = self._positive_payload()
        payload["bc_evaluation"] = self._episodes(successes=0)
        payload["matched_evaluation"]["bc"] = copy.deepcopy(
            payload["bc_evaluation"]
        )
        with self.assertRaisesRegex(module.ProtocolError, "PPO ran"):
            module.validate_payload(payload, self.manifest, self.contract)

    def test_ppo_fails_closed_without_manager_model_fsdp_root(self):
        payload = self._positive_payload()
        payload["ppo_training"]["root_forward"] = {
            "path": "local_policy_callback",
            "fsdp_wrapped": False,
            "manager_model_forward_count": 0,
            "local_policy_callback_forward_count": 3,
            "operations": {"sample": 1, "replay": 1, "bootstrap": 1},
        }
        with self.assertRaisesRegex(module.ProtocolError, "manager.model/FSDP"):
            module.validate_payload(payload, self.manifest, self.contract)

    def test_ppo_requires_all_eight_episodes_and_updates(self):
        payload = self._positive_payload()
        payload["ppo_training"]["updates"].pop()
        with self.assertRaisesRegex(module.ProtocolError, "exactly eight"):
            module.validate_payload(payload, self.manifest, self.contract)

    def test_bc_requires_expert_then_dagger_sixteen_traversals(self):
        payload = self._positive_payload()
        payload["bc_training"]["traversals"].pop()
        with self.assertRaisesRegex(module.ProtocolError, "sixteen"):
            module.validate_payload(payload, self.manifest, self.contract)

    def test_shadow_stop_may_not_execute_or_credit_a_pending_macro(self):
        payload = self._positive_payload()
        payload["bc_evaluation"]["episodes"][1][
            "post_shadow_stop_primitive_count"
        ] = 1
        payload["matched_evaluation"]["bc"] = copy.deepcopy(
            payload["bc_evaluation"]
        )
        with self.assertRaisesRegex(module.ProtocolError, "shadow STOP isolation"):
            module.validate_payload(payload, self.manifest, self.contract)

    def test_mid_macro_shadow_stop_drops_sampled_actor_credit(self):
        payload = self._positive_payload()
        for field in ("bc_evaluation",):
            receipt = payload[field]["episodes"][1]["decision_receipts"][-1]
            receipt["official_shadow_stop_phase"] = "mid_macro"
            receipt["actor_sampled"] = True
            receipt["proposed_primitive_count"] = 2
        payload["matched_evaluation"]["bc"] = copy.deepcopy(
            payload["bc_evaluation"]
        )
        result = module.validate_payload(payload, self.manifest, self.contract)
        self.assertTrue(result["gate_a"]["pass"])

    def test_positive_signal_requires_matched_navigation_gain(self):
        result = module.validate_payload(
            self._positive_payload(), self.manifest, self.contract
        )
        self.assertEqual(result["status"], "pilot_signal_positive")
        self.assertTrue(result["gate_a"]["pass"])
        self.assertTrue(result["gate_b"]["pass"])
        self.assertEqual(result["gate_b"]["deltas"]["success"], 1)
        self.assertGreaterEqual(
            result["gate_b"]["deltas"]["changed_executed_trace_count"], 2
        )

    def test_preflight_writes_receipt_without_loading_live_provider(self):
        output = Path(self.temporary.name) / "preflight.json"
        args = argparse.Namespace(
            manifest=MANIFEST_PATH,
            split="train",
            stage="preflight",
            provider="",
            output=output,
        )
        result = module.run(args)
        self.assertEqual(result["status"], "preflight_passed_live_provider_not_run")
        self.assertFalse(result["ppo_started"])
        self.assertEqual(
            json.loads(output.read_text(encoding="utf-8"))["receipt_sha256"],
            result["receipt_sha256"],
        )

    def test_live_provider_pure_helpers_preserve_action_and_stop_target(self):
        self.assertEqual(live_provider._single_action_transport(3), [[[3], [0], [0], [0]]])
        with self.assertRaisesRegex(live_provider.LiveProviderError, "0..3"):
            live_provider._single_action_transport(4)
        snapshot = SimpleNamespace(
            candidates=(
                SimpleNamespace(
                    candidate_id="frontier:forward",
                    valid=True,
                    continuous=True,
                    forward_m=0.75,
                    lateral_m=0.0,
                ),
                SimpleNamespace(
                    candidate_id="stop",
                    valid=True,
                    continuous=False,
                    forward_m=0.0,
                    lateral_m=0.0,
                ),
            )
        )
        self.assertEqual(
            live_provider._nearest_candidate(
                snapshot, {"stop": True, "waypoint_local_m": (0.0, 0.0)}
            ),
            (1, (0.0, 0.0)),
        )


if __name__ == "__main__":
    unittest.main()
