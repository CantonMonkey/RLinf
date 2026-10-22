import argparse
import copy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest


MODULE_DIR = Path(__file__).resolve().parents[1] / "integrations" / "rlinf_v03"
SPEC = importlib.util.spec_from_file_location(
    "run_streamvln_p8_successor_causal_probe",
    MODULE_DIR / "run_streamvln_p8_successor_causal_probe.py",
)
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)
PROBE_MANIFEST = MODULE_DIR / "streamvln_p8_successor_causal_probe_manifest.json"
READINESS_MANIFEST = (
    MODULE_DIR / "streamvln_p8_successor_bc_readiness_manifest.json"
)


def fake_navigation():
    return {
        "episode_count": 8,
        "success_count": 4,
        "successful_stop_count": 1,
        "root_recovery_unavailable_count": 0,
        "action_counts": {
            "STOP": 1,
            "FORWARD": 30,
            "LEFT": 35,
            "RIGHT": 35,
        },
    }


def fake_teacher():
    return {
        "decision_count": 20,
        "route_support": {"stop": 4, "keep_original": 8, "waypoint": 8},
        "route_nonstop_bal_acc": 0.65,
        "stop": {"recall": 0.5, "false_positive": 1},
        "waypoint": {
            "first_action_agreement": 0.65,
            "median_bearing_abs_error_deg": 15.0,
            "p90_bearing_abs_error_deg": 45.0,
            "median_vector_l2_m": 0.25,
        },
    }


def fake_payload(development):
    episodes = [
        {"episode_id": row["episode_id"]} for row in development
    ]
    return {
        "candidate_checkpoint_sha256": PROBE.EXPECTED_CANDIDATE_SHA256,
        "candidate_policy_state_sha256": PROBE.EXPECTED_POLICY_STATE_SHA256,
        "teacher_forced": {
            "metrics": fake_teacher(),
            "episodes": [
                {"episode_id": row["episode_id"]} for row in development
            ],
        },
        "arms": {
            arm: {
                "navigation": fake_navigation(),
                "counterfactual_diagnostics": fake_teacher(),
                "episodes": copy.deepcopy(episodes),
            }
            for arm in PROBE.ARMS
        },
        "split_read_counts": {
            "train": 24,
            "val_seen": 0,
            "val_unseen": 0,
        },
        "guards": {
            "model_load_count": 1,
            "candidate_load_count": 1,
            "training_updates": 0,
            "ppo_updates": 0,
            "promotion_attempts": 0,
            "single_process": True,
            "train_only": True,
        },
    }


class CausalProbeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.probe_manifest = json.loads(PROBE_MANIFEST.read_text(encoding="utf-8"))
        cls.readiness_manifest = json.loads(
            READINESS_MANIFEST.read_text(encoding="utf-8")
        )

    def make_args(self, root, *, split="train"):
        return argparse.Namespace(
            split=split,
            probe_manifest=PROBE_MANIFEST,
            readiness_manifest=READINESS_MANIFEST,
            readiness_receipt=root / "readiness.json",
            candidate=root / "candidate.pt",
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
            output=root / "probe_receipt.json",
        )

    def test_manifest_binds_v5_and_exact_scene_separated_development(self):
        contract = PROBE.validate_probe_manifest(
            self.probe_manifest, self.readiness_manifest
        )
        self.assertEqual(len(contract["development"]), 8)
        self.assertEqual(
            len({row["scene"] for row in contract["development"]}), 8
        )
        self.assertEqual(contract["protocol"]["decision_budget_per_episode"], 160)
        self.assertEqual(set(self.probe_manifest["arms"]), set(PROBE.ARMS))
        self.assertFalse(self.probe_manifest["claim_boundary"]["training"])
        self.assertFalse(self.probe_manifest["claim_boundary"]["bc_promotion"])

    def test_manifest_rejects_episode_or_arm_drift(self):
        changed = copy.deepcopy(self.probe_manifest)
        changed["development_episodes"][0]["episode_id"] = "changed"
        with self.assertRaisesRegex(PROBE.ProtocolError, "differ"):
            PROBE.validate_probe_manifest(changed, self.readiness_manifest)
        changed = copy.deepcopy(self.probe_manifest)
        changed["arms"][PROBE.ARM_ROUTE]["waypoint_source_when_waypoint"] = (
            "candidate"
        )
        with self.assertRaisesRegex(PROBE.ProtocolError, "arms drifted"):
            PROBE.validate_probe_manifest(changed, self.readiness_manifest)

    def test_manifest_rejects_validation_or_training_scope(self):
        changed = copy.deepcopy(self.probe_manifest)
        changed["split_contract"]["allowed_splits"] = ["train", "val_seen"]
        with self.assertRaisesRegex(PROBE.ProtocolError, "split contract"):
            PROBE.validate_probe_manifest(changed, self.readiness_manifest)
        changed = copy.deepcopy(self.probe_manifest)
        changed["claim_boundary"]["training"] = True
        with self.assertRaisesRegex(PROBE.ProtocolError, "claim boundary"):
            PROBE.validate_probe_manifest(changed, self.readiness_manifest)

    def test_manifest_rejects_threshold_or_attribution_drift(self):
        changed = copy.deepcopy(self.probe_manifest)
        changed["thresholds"]["arm_recovered"]["success_count_min"] = 3
        with self.assertRaisesRegex(PROBE.ProtocolError, "thresholds drifted"):
            PROBE.validate_probe_manifest(changed, self.readiness_manifest)
        changed = copy.deepcopy(self.probe_manifest)
        changed["attribution_rule"]["otherwise_authorize_feature_v2"] = True
        with self.assertRaisesRegex(PROBE.ProtocolError, "attribution rule"):
            PROBE.validate_probe_manifest(changed, self.readiness_manifest)

    def test_literal_train_rejection_precedes_all_path_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.make_args(root, split="val_unseen")
            args.probe_manifest = root / "missing.json"
            args.readiness_manifest = root / "missing-readiness.json"
            with self.assertRaisesRegex(ValueError, "literal train split"):
                PROBE.run(args)
            self.assertFalse(args.output.exists())

    def test_teacher_summary_records_required_failure_modes(self):
        records = [
            {
                "truth_route": "stop",
                "predicted_route": "waypoint",
                "waypoint_vector_l2_m": None,
                "selected_waypoint_zero_forward": True,
                "true_stop_predicted_waypoint_candidate_fallback": True,
            },
            {
                "truth_route": "waypoint",
                "predicted_route": "stop",
                "waypoint_vector_l2_m": 1.0,
                "waypoint_bearing_abs_error_rad": math.pi / 2,
                "waypoint_distance_abs_error_m": 0.25,
                "oracle_waypoint_first_action": 1,
                "predicted_waypoint_first_action": 2,
                "predicted_waypoint_zero_forward": True,
                "selected_waypoint_zero_forward": None,
            },
            {
                "truth_route": "keep_original",
                "predicted_route": "keep_original",
                "waypoint_vector_l2_m": 0.5,
                "waypoint_bearing_abs_error_rad": 0.0,
                "waypoint_distance_abs_error_m": 0.5,
                "oracle_waypoint_first_action": 1,
                "predicted_waypoint_first_action": 1,
                "predicted_waypoint_zero_forward": False,
                "selected_waypoint_zero_forward": False,
            },
        ]
        summary = PROBE.summarize_teacher_forced(records)
        self.assertEqual(summary["decision_count"], 3)
        self.assertEqual(summary["route_confusion"]["stop"]["waypoint"], 1)
        self.assertEqual(summary["stop"]["false_positive"], 1)
        self.assertEqual(summary["stop"]["false_negative"], 1)
        self.assertEqual(summary["waypoint"]["comparison_count"], 2)
        self.assertAlmostEqual(
            summary["waypoint"]["mean_bearing_abs_error_deg"], 45.0
        )
        self.assertEqual(
            summary["waypoint"]["compiled_first_action_confusion"]["FORWARD"]["LEFT"],
            1,
        )
        self.assertEqual(
            summary["waypoint"]["predicted_zero_forward_macro_count"], 1
        )
        self.assertEqual(summary["route_support"]["stop"], 1)
        self.assertAlmostEqual(
            summary["waypoint"]["first_action_agreement"], 0.5
        )
        self.assertAlmostEqual(
            summary["waypoint"]["median_vector_l2_m"], 0.75
        )
        self.assertEqual(
            summary["route_arm_true_stop_predicted_waypoint_fallback"][
                "true_stop_predicted_waypoint_candidate_fallback_count"
            ],
            1,
        )
        self.assertFalse(
            summary["route_arm_true_stop_predicted_waypoint_fallback"][
                "is_oracle"
            ]
        )

    def test_route_arm_true_stop_waypoint_uses_candidate_not_fake_oracle(self):
        selected = PROBE._select_hybrid_execution(
            arm=PROBE.ARM_ROUTE,
            predicted_route="waypoint",
            truth_route="stop",
            predicted_waypoint=(0.4, -0.2),
            oracle_waypoint=None,
        )
        self.assertEqual(selected["selected_route"], "waypoint")
        self.assertEqual(selected["selected_waypoint"], (0.4, -0.2))
        self.assertEqual(
            selected["selected_waypoint_source"],
            "candidate_deterministic_mean_true_stop_fallback",
        )
        self.assertTrue(
            selected["true_stop_predicted_waypoint_candidate_fallback"]
        )

        selected = PROBE._select_hybrid_execution(
            arm=PROBE.ARM_ROUTE,
            predicted_route="waypoint",
            truth_route="waypoint",
            predicted_waypoint=(0.4, -0.2),
            oracle_waypoint=(1.0, 0.1),
        )
        self.assertEqual(selected["selected_waypoint"], (1.0, 0.1))
        self.assertEqual(
            selected["selected_waypoint_source"],
            "train_shortest_path_expert",
        )
        self.assertFalse(
            selected["true_stop_predicted_waypoint_candidate_fallback"]
        )

    def test_frozen_threshold_boundaries_are_inclusive(self):
        contract = PROBE.validate_probe_manifest(
            self.probe_manifest, self.readiness_manifest
        )
        payload = fake_payload(contract["development"])
        decision = PROBE.adjudicate_frozen_attribution(
            payload,
            thresholds=contract["thresholds"],
            attribution_rule=contract["attribution_rule"],
        )
        self.assertTrue(decision["arms"][PROBE.ARM_ROUTE]["recovered"])
        self.assertTrue(decision["arms"][PROBE.ARM_WAYPOINT]["recovered"])
        self.assertTrue(decision["teacher"]["sample_sufficient"])
        self.assertTrue(decision["teacher"]["route_group_pass"])
        self.assertTrue(decision["teacher"]["waypoint_group_pass"])
        self.assertFalse(decision["authorize_feature_v2"])

    def test_frozen_thresholds_fail_immediately_outside_boundary(self):
        contract = PROBE.validate_probe_manifest(
            self.probe_manifest, self.readiness_manifest
        )
        payload = fake_payload(contract["development"])
        payload["arms"][PROBE.ARM_ROUTE]["navigation"]["action_counts"] = {
            "STOP": 1,
            "FORWARD": 294,
            "LEFT": 353,
            "RIGHT": 353,
        }
        teacher = payload["teacher_forced"]["metrics"]
        teacher["route_nonstop_bal_acc"] = 0.649999
        teacher["stop"]["recall"] = 0.499999
        teacher["stop"]["false_positive"] = 2
        teacher["waypoint"].update({
            "first_action_agreement": 0.649999,
            "median_bearing_abs_error_deg": 15.000001,
            "p90_bearing_abs_error_deg": 45.000001,
            "median_vector_l2_m": 0.250001,
        })
        decision = PROBE.adjudicate_frozen_attribution(
            payload,
            thresholds=contract["thresholds"],
            attribution_rule=contract["attribution_rule"],
        )
        self.assertFalse(decision["arms"][PROBE.ARM_ROUTE]["recovered"])
        self.assertFalse(decision["teacher"]["route_group_pass"])
        self.assertFalse(decision["teacher"]["waypoint_group_pass"])

    def test_forward_nonstop_fraction_boundary_is_independently_inclusive(self):
        contract = PROBE.validate_probe_manifest(
            self.probe_manifest, self.readiness_manifest
        )
        payload = fake_payload(contract["development"])
        navigation = payload["arms"][PROBE.ARM_ROUTE]["navigation"]
        navigation["action_counts"] = {
            "STOP": 1,
            "FORWARD": 295,
            "LEFT": 352,
            "RIGHT": 353,
        }
        at_boundary = PROBE.adjudicate_frozen_attribution(
            payload,
            thresholds=contract["thresholds"],
            attribution_rule=contract["attribution_rule"],
        )["arms"][PROBE.ARM_ROUTE]
        self.assertAlmostEqual(
            at_boundary["observed"]["forward_nonstop_fraction"], 0.295
        )
        self.assertTrue(at_boundary["recovered"])
        self.assertTrue(all(at_boundary["checks"].values()))

        navigation["action_counts"] = {
            "STOP": 1,
            "FORWARD": 294,
            "LEFT": 353,
            "RIGHT": 353,
        }
        below_boundary = PROBE.adjudicate_frozen_attribution(
            payload,
            thresholds=contract["thresholds"],
            attribution_rule=contract["attribution_rule"],
        )["arms"][PROBE.ARM_ROUTE]
        self.assertAlmostEqual(
            below_boundary["observed"]["forward_nonstop_fraction"], 0.294
        )
        self.assertFalse(below_boundary["recovered"])
        self.assertEqual(
            {
                name
                for name, passed in below_boundary["checks"].items()
                if not passed
            },
            {"forward_nonstop_fraction"},
        )

    def test_feature_v2_authorization_requires_the_exact_joint_failure(self):
        contract = PROBE.validate_probe_manifest(
            self.probe_manifest, self.readiness_manifest
        )
        payload = fake_payload(contract["development"])
        for arm in PROBE.ARMS:
            payload["arms"][arm]["navigation"]["success_count"] = 0
        teacher = payload["teacher_forced"]["metrics"]
        teacher["route_nonstop_bal_acc"] = 0.0
        teacher["waypoint"]["first_action_agreement"] = 0.0
        decision = PROBE.adjudicate_frozen_attribution(
            payload,
            thresholds=contract["thresholds"],
            attribution_rule=contract["attribution_rule"],
        )
        self.assertTrue(decision["authorize_feature_v2"])
        self.assertEqual(
            decision["attribution"],
            "joint_teacher_failure_authorizes_feature_v2",
        )

        payload["arms"][PROBE.ARM_ROUTE]["navigation"]["success_count"] = 4
        decision = PROBE.adjudicate_frozen_attribution(
            payload,
            thresholds=contract["thresholds"],
            attribution_rule=contract["attribution_rule"],
        )
        self.assertFalse(decision["authorize_feature_v2"])

        payload["arms"][PROBE.ARM_ROUTE]["navigation"]["success_count"] = 0
        teacher["route_support"]["stop"] = 3
        decision = PROBE.adjudicate_frozen_attribution(
            payload,
            thresholds=contract["thresholds"],
            attribution_rule=contract["attribution_rule"],
        )
        self.assertFalse(decision["authorize_feature_v2"])
        self.assertEqual(decision["attribution"], "insufficient_teacher_samples")

        teacher["route_support"]["stop"] = 4
        teacher["route_nonstop_bal_acc"] = 0.65
        teacher["stop"] = {"recall": 0.5, "false_positive": 1}
        decision = PROBE.adjudicate_frozen_attribution(
            payload,
            thresholds=contract["thresholds"],
            attribution_rule=contract["attribution_rule"],
        )
        self.assertFalse(decision["authorize_feature_v2"])
        self.assertTrue(decision["teacher"]["route_group_pass"])

        teacher["route_nonstop_bal_acc"] = 0.0
        teacher["waypoint"].update({
            "first_action_agreement": 0.65,
            "median_bearing_abs_error_deg": 15.0,
            "p90_bearing_abs_error_deg": 45.0,
            "median_vector_l2_m": 0.25,
        })
        decision = PROBE.adjudicate_frozen_attribution(
            payload,
            thresholds=contract["thresholds"],
            attribution_rule=contract["attribution_rule"],
        )
        self.assertFalse(decision["authorize_feature_v2"])
        self.assertTrue(decision["teacher"]["waypoint_group_pass"])

    def test_waypoint_arm_only_recovery_blocks_authorization_and_attributes_route(self):
        contract = PROBE.validate_probe_manifest(
            self.probe_manifest, self.readiness_manifest
        )
        payload = fake_payload(contract["development"])
        for arm in PROBE.ARMS:
            payload["arms"][arm]["navigation"]["success_count"] = 0
        payload["arms"][PROBE.ARM_WAYPOINT]["navigation"]["success_count"] = 4
        teacher = payload["teacher_forced"]["metrics"]
        teacher["route_nonstop_bal_acc"] = 0.0
        teacher["waypoint"]["first_action_agreement"] = 0.0

        decision = PROBE.adjudicate_frozen_attribution(
            payload,
            thresholds=contract["thresholds"],
            attribution_rule=contract["attribution_rule"],
        )

        self.assertTrue(decision["teacher"]["sample_sufficient"])
        self.assertFalse(decision["teacher"]["route_group_pass"])
        self.assertFalse(decision["teacher"]["waypoint_group_pass"])
        self.assertFalse(decision["arms"][PROBE.ARM_ROUTE]["recovered"])
        self.assertTrue(decision["arms"][PROBE.ARM_WAYPOINT]["recovered"])
        self.assertFalse(decision["authorize_feature_v2"])
        self.assertEqual(
            decision["attribution"],
            "route_stop_branch_failure_supported",
        )

    def test_runner_implementation_self_check_rejects_byte_drift(self):
        observed = PROBE.verify_probe_implementation(self.probe_manifest)
        self.assertEqual(
            observed["runner_sha256"],
            self.probe_manifest["implementation"]["runner_sha256"],
        )
        with tempfile.TemporaryDirectory() as directory:
            copied = (
                Path(directory)
                / "integrations"
                / "rlinf_v03"
                / "run_streamvln_p8_successor_causal_probe.py"
            )
            copied.parent.mkdir(parents=True)
            copied.write_bytes(
                (MODULE_DIR / copied.name).read_bytes() + b"\n# drift\n"
            )
            with self.assertRaisesRegex(PROBE.ProtocolError, "bytes/hash drifted"):
                PROBE.verify_probe_implementation(
                    self.probe_manifest, runner_path=copied
                )

    def test_oracle_waypoint_exposure_is_scoped_and_restored(self):
        original = PROBE.readiness.label_successor_route
        with PROBE._expose_oracle_waypoint_to_probe():
            self.assertEqual(
                PROBE.readiness.label_successor_route(
                    expert_stop=False,
                    official_shadow_action=1,
                    expert_first_primitive=1,
                ),
                "waypoint",
            )
        self.assertIs(PROBE.readiness.label_successor_route, original)

    def test_run_writes_atomic_diagnostic_receipt_without_promotion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.make_args(root)
            contract = PROBE.validate_probe_manifest(
                self.probe_manifest, self.readiness_manifest
            )

            def lineage(_args, _manifest, _contract):
                return {"status": "pass", "production": {"paths": {}}}

            payload = fake_payload(contract["development"])
            result = PROBE.run(
                args,
                lineage_provider=lineage,
                probe_provider=lambda **_kwargs: payload,
            )
            self.assertTrue(args.output.is_file())
            self.assertEqual(result["promotion"], None)
            self.assertFalse(result["claim_boundary"]["training"])
            self.assertFalse(result["claim_boundary"]["validation_split_access"])
            self.assertFalse(result["authorize_feature_v2"])
            self.assertTrue(
                result["result"]["arms"][PROBE.ARM_ROUTE]["recovered"]
            )
            self.assertTrue(
                result["result"]["teacher_forced"]["judgement"][
                    "sample_sufficient"
                ]
            )
            on_disk = json.loads(args.output.read_text(encoding="utf-8"))
            self.assertEqual(on_disk, result)
            unsigned = dict(on_disk)
            embedded = unsigned.pop("receipt_sha256")
            self.assertEqual(PROBE.readiness._json_hash(unsigned), embedded)

    def test_invalid_provider_result_leaves_no_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.make_args(root)
            contract = PROBE.validate_probe_manifest(
                self.probe_manifest, self.readiness_manifest
            )
            payload = fake_payload(contract["development"])
            payload["split_read_counts"]["val_seen"] = 1
            with self.assertRaisesRegex(PROBE.ProtocolError, "split-read"):
                PROBE.run(
                    args,
                    lineage_provider=lambda *_args: {"status": "pass"},
                    probe_provider=lambda **_kwargs: payload,
                )
            self.assertFalse(args.output.exists())


if __name__ == "__main__":
    unittest.main()
