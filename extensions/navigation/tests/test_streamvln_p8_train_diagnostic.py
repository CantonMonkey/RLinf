import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "integrations" / "rlinf_v03"


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, STAGE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILDER = load_module(
    "build_streamvln_p8_train_diagnostic_manifest",
    "build_streamvln_p8_train_diagnostic_manifest.py",
)
RUNNER = load_module(
    "run_streamvln_p8_train_diagnostic",
    "run_streamvln_p8_train_diagnostic.py",
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestP8Manifest(unittest.TestCase):
    def write_train(self, root, *, changed_outcomes=False):
        path = root / "datasets" / "r2r" / "train" / "train.json.gz"
        path.parent.mkdir(parents=True)
        episodes = []
        for scene in range(20):
            scene_dir = root / "scene_datasets" / "mp3d" / f"scene-{scene}"
            scene_dir.mkdir(parents=True, exist_ok=True)
            (scene_dir / f"scene-{scene}.glb").write_bytes(f"glb-{scene}".encode())
            (scene_dir / f"scene-{scene}.navmesh").write_bytes(f"nav-{scene}".encode())
            for member in range(2):
                prefixes = ("", "data/scene_datasets/", "scene_datasets/")
                episodes.append({
                    "episode_id": f"e-{scene}-{member}",
                    "scene_id": f"{prefixes[scene % len(prefixes)]}mp3d/scene-{scene}/scene-{scene}.glb",
                    "instruction": "changed" if changed_outcomes else "original",
                    "path": [999, scene] if changed_outcomes else [scene, member],
                    "success": bool(changed_outcomes),
                })
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            json.dump({"episodes": episodes}, stream, sort_keys=True)
        return path

    def build_fixture(self, root, *, changed_outcomes=False):
        path = self.write_train(root, changed_outcomes=changed_outcomes)
        original = BUILDER.TRAIN_SHA256
        try:
            BUILDER.TRAIN_SHA256 = sha256(path)
            return BUILDER.build_manifest(root, verify_external=False)
        finally:
            BUILDER.TRAIN_SHA256 = original

    def test_selection_is_deterministic_disjoint_and_identity_only(self):
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first = self.build_fixture(Path(first_dir))
            second = self.build_fixture(Path(second_dir), changed_outcomes=True)
        for key in ("tiny_train", "equivalence", "train_development", "counts"):
            self.assertEqual(first["episodes"][key], second["episodes"][key])
        self.assertEqual(first["episodes"]["equivalence"], first["episodes"]["tiny_train"][:4])
        tiny_scenes = {row["scene"] for row in first["episodes"]["tiny_train"]}
        dev_scenes = {row["scene"] for row in first["episodes"]["train_development"]}
        self.assertEqual(len(tiny_scenes), 8)
        self.assertEqual(len(dev_scenes), 8)
        self.assertFalse(tiny_scenes.intersection(dev_scenes))
        self.assertEqual(first["source"]["allowed_splits"], ["train"])
        self.assertFalse(first["source"]["val_seen_read"])
        self.assertFalse(first["source"]["val_unseen_read"])

    def test_atomic_serialization_is_byte_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.build_fixture(root)
            first = root / "first.json"
            second = root / "second.json"
            BUILDER.write_json_atomic(first, manifest)
            BUILDER.write_json_atomic(second, manifest)
            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_scene_path_normalization_rejects_escape_and_malformed_ids(self):
        accepted = (
            "mp3d/scene-a/scene-a.glb",
            "scene_datasets/mp3d/scene-a/scene-a.glb",
            "data/scene_datasets/mp3d/scene-a/scene-a.glb",
        )
        for source in accepted:
            self.assertEqual(
                str(BUILDER.normalized_r2r_scene_path(source)),
                "mp3d/scene-a/scene-a.glb",
            )
        rejected = (
            "/mp3d/scene-a/scene-a.glb",
            "mp3d/../scene-a/scene-a.glb",
            "mp3d/scene-a/other.glb",
            "mp3d/scene-a/scene-a.navmesh",
            "hm3d/scene-a/scene-a.glb",
            "mp3d\\scene-a\\scene-a.glb",
        )
        for source in rejected:
            with self.assertRaises(ValueError, msg=source):
                BUILDER.normalized_r2r_scene_path(source)

    def test_protocol_fields_are_fully_frozen(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.build_fixture(Path(directory))
        self.assertEqual(manifest["schema_version"], 2)
        self.assertEqual(manifest["runtime"]["context_tokens"], 4096)
        self.assertFalse(manifest["runtime"]["deterministic_eval"])
        self.assertEqual(manifest["runtime"]["determinism"]["cublas_workspace_config"], ":4096:8")
        self.assertFalse(manifest["runtime"]["determinism"]["torch_use_deterministic_algorithms"])
        self.assertEqual(manifest["executor"]["action_token_ids"], [50669, 76286, 71858, 51018])
        self.assertEqual(manifest["training"]["optimizer"]["gradient_clip"], None)
        self.assertEqual(manifest["training"]["sampling"], {"method": "torch.randint", "replacement": True})
        self.assertFalse(manifest["training"]["value_head_trainable"])
        self.assertEqual(len(manifest["episodes"]["selected_scene_assets"]), 16)
        self.assertEqual(manifest["runtime_identity"]["rlinf"]["packages"]["omegaconf"], "2.3.1")
        self.assertIn("integration/streamvln_executor_contract.py", BUILDER.OVERLAY_FILES)
        self.assertIn("integration/streamvln_topology.py", BUILDER.OVERLAY_FILES)

    def test_live_loader_freezes_and_checks_original_base(self):
        runner_text = (STAGE / "run_streamvln_p8_train_diagnostic.py").read_text(encoding="utf-8")
        self.assertIn("policy.requires_grad_(False)", runner_text)
        self.assertIn('!= "upstream-original"', runner_text)

    def test_live_required_paths_use_symlink_preserving_validator(self):
        args = mock.Mock()
        args.habitat_python = Path("/declared/venv/bin/python")
        with mock.patch.object(
            RUNNER,
            "_absolute_existing",
            return_value=args.habitat_python,
        ) as validator:
            paths = RUNNER._required_paths(args, ("habitat_python",))
        self.assertEqual(paths["habitat_python"], args.habitat_python)
        validator.assert_called_once_with(args.habitat_python)

    def test_sources_contain_no_validation_dataset_path(self):
        builder_text = (STAGE / "build_streamvln_p8_train_diagnostic_manifest.py").read_text(encoding="utf-8")
        runner_text = (STAGE / "run_streamvln_p8_train_diagnostic.py").read_text(encoding="utf-8")
        self.assertNotIn('datasets" / "r2r" / "val_seen', builder_text)
        self.assertNotIn('datasets" / "r2r" / "val_unseen', builder_text)
        self.assertNotIn('split="val_seen"', runner_text)
        self.assertNotIn('split="val_unseen"', runner_text)
        self.assertNotIn("ppo_step", runner_text.lower())

    def test_formal_manifest_is_pinned_and_valid(self):
        if RUNNER.MANIFEST_SHA256 == "UNPINNED":
            self.skipTest("formal schema-v2 manifest must be regenerated and pinned remotely")
        path = STAGE / "streamvln_p8_train_diagnostic_manifest.json"
        self.assertEqual(sha256(path), RUNNER.MANIFEST_SHA256)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        RUNNER.validate_manifest(manifest)
        self.assertEqual(manifest["source"]["train_sha256"], BUILDER.TRAIN_SHA256)


class TestP8GateAdjudication(unittest.TestCase):
    def thresholds(self):
        return {
            "a": {
                "mismatch_count_max": 0,
                "optimizer_steps": 0,
                "parameter_hash_unchanged": True,
            },
            "b": {
                "token_top1_accuracy_min": 0.90,
                "navigate_intent_recall_min": 0.90,
                "stop_intent_recall_min": 0.90,
                "stop_token_recall_min": 0.90,
                "waypoint_mean_l2_m_max": 0.15,
                "sample_replay_ratio_deviation_max": 0.001,
                "consecutive_passes": 2,
            },
            "c": {
                "learned_success_count_min": 1,
                "successful_stop_count_min": 1,
                "max_success_deficit_vs_keep_original": 1,
                "primitive_category_count_min": 2,
                "decision_mode_count_min": 2,
                "turn_count_for_handedness_check": 20,
                "single_turn_direction_fraction_max": 0.95,
                "recover_required_when_blocked": True,
                "recover_alignment_miss_max": 0,
            },
        }

    def gate_a(self, mismatch=0):
        return {
            "episode_count": 4,
            "model_load_count": 1,
            "optimizer_steps": 0,
            "parameter_hash_unchanged": True,
            "release_reference": "pre_refactor_sampler_at_5434ab0",
            "parameter_hash_scope": {
                "backbone": "complete_named_parameters_and_buffers",
                "residual": "complete_state_dict",
            },
            "backbone_full_hash_before": "a" * 64,
            "backbone_full_hash_after": "a" * 64,
            "residual_hash_before": "b" * 64,
            "residual_hash_after": "b" * 64,
            "mismatch_counts": {
                key: mismatch
                for key in (
                    "base_logits",
                    "sampled_tokens",
                    "primitive_actions",
                    "proposed_prefix",
                    "executed_prefix",
                    "stop_position",
                    "termination",
                    "terminal_metrics",
                )
            },
            "split_read_counts": {"train": 8, "val_seen": 0, "val_unseen": 0},
        }

    def gate_b(self, **updates):
        payload = {
            "token_top1_accuracy": 0.90,
            "navigate_intent_recall": 0.90,
            "stop_intent_recall": 0.90,
            "stop_token_recall": 0.90,
            "waypoint_mean_l2_m": 0.15,
            "sample_replay_ratio_deviation": 0.001,
            "consecutive_passes": 2,
            "trainable_tensors_changed": True,
            "expected_trainable_parameter_names": ["a", "b"],
            "optimizer_parameter_names": ["a", "b"],
            "finite_gradient_names": ["a", "b"],
            "nonzero_gradient_names": ["a", "b"],
            "parameter_change_checks": {"a": True, "b": True},
            "all_expected_trainables_changed": True,
            "gradient_scope_exact": True,
            "optimizer_scope_exact": True,
            "required_trainable_groups_changed": True,
            "value_head_frozen_unchanged": True,
            "undeclared_tensors_unchanged": True,
            "frozen_backbone_hash_unchanged": True,
            "backbone_hash_scope": "complete_named_parameters_and_buffers",
            "all_finite": True,
            "trace_fields_complete": True,
            "recover_positive_count": 0,
            "recover_recall": "not_applicable",
            "split_read_counts": {"train": 16, "val_seen": 0, "val_unseen": 0},
        }
        payload.update(updates)
        return payload

    def gate_c(self, **updates):
        payload = {
            "episode_count": 8,
            "learned_success_count": 1,
            "keep_original_success_count": 2,
            "successful_stop_count": 1,
            "stop_count": 2,
            "primitive_counts": {"forward": 10, "left": 5, "right": 0},
            "mode_counts": {"navigate": 8, "stop": 2, "recover": 0},
            "left_turn_count": 10,
            "right_turn_count": 1,
            "blocked_state_count": 0,
            "recover_trigger_count": 0,
            "recover_outcomes": [],
            "recover_alignment_miss_count": 0,
            "checkpoint_hash_before": "c" * 64,
            "checkpoint_hash_after": "c" * 64,
            "checkpoint_hash_scope": {
                "backbone": "complete_named_parameters_and_buffers",
                "residual": "complete_state_dict",
            },
            "backbone_full_hash_before": "a" * 64,
            "backbone_full_hash_after": "a" * 64,
            "residual_full_hash_before": "b" * 64,
            "residual_full_hash_after": "b" * 64,
            "split_read_counts": {"train": 16, "val_seen": 0, "val_unseen": 0},
        }
        payload.update(updates)
        return payload

    def test_gate_b_thresholds_are_inclusive(self):
        result = RUNNER.adjudicate_gate_b(self.gate_b(), self.thresholds()["b"])
        self.assertTrue(result["passed"])
        result = RUNNER.adjudicate_gate_b(
            self.gate_b(stop_token_recall=0.899999), self.thresholds()["b"]
        )
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["stop_token_recall"])

    def test_gate_b_does_not_invent_recover_recall(self):
        result = RUNNER.adjudicate_gate_b(
            self.gate_b(recover_positive_count=0, recover_recall=1.0),
            self.thresholds()["b"],
        )
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["recover_reporting_valid"])

    def test_gate_b_rejects_incomplete_trainable_evidence(self):
        result = RUNNER.adjudicate_gate_b(
            self.gate_b(nonzero_gradient_names=["a"]), self.thresholds()["b"]
        )
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["exact_trainable_evidence"])

    def test_gate_c_enforces_recovery_and_handedness_conditionally(self):
        passing = RUNNER.adjudicate_gate_c(self.gate_c(), self.thresholds()["c"])
        self.assertTrue(passing["passed"])
        blocked = RUNNER.adjudicate_gate_c(
            self.gate_c(blocked_state_count=1), self.thresholds()["c"]
        )
        self.assertFalse(blocked["checks"]["recover_when_blocked"])
        collapsed = RUNNER.adjudicate_gate_c(
            self.gate_c(left_turn_count=20, right_turn_count=0), self.thresholds()["c"]
        )
        self.assertFalse(collapsed["checks"]["handedness"])
        misaligned = RUNNER.adjudicate_gate_c(
            self.gate_c(
                blocked_state_count=1,
                recover_trigger_count=1,
                recover_outcomes=[],
            ),
            self.thresholds()["c"],
        )
        self.assertFalse(misaligned["checks"]["recover_outcomes_aligned"])

    def test_gate_c_rejects_malformed_numeric_payloads_as_protocol_errors(self):
        malformed = self.gate_c(blocked_state_count="not-an-int")
        with self.assertRaises(RUNNER.ProtocolError):
            RUNNER.adjudicate_gate_c(malformed, self.thresholds()["c"])

        huge_path = self.gate_c(
            blocked_state_count=1,
            recover_trigger_count=1,
            recover_outcomes=[{
                "episode_id": "e-1",
                "decision_index": 0,
                "actionable_blocked_before": True,
                "consecutive_blocked_before": 2,
                "consecutive_blocked_after": 0,
                "realized_path_m": 10**400,
            }],
        )
        with self.assertRaises(RUNNER.ProtocolError):
            RUNNER.adjudicate_gate_c(huge_path, self.thresholds()["c"])

        malformed_after_semantic_failure = self.gate_c(
            blocked_state_count=1,
            recover_trigger_count=2,
            recover_outcomes=[
                {
                    "episode_id": "e-1",
                    "decision_index": 0,
                    "actionable_blocked_before": True,
                    "consecutive_blocked_before": 1,
                    "consecutive_blocked_after": 0,
                    "realized_path_m": 0.0,
                },
                {
                    "episode_id": "e-2",
                    "decision_index": "malformed",
                    "actionable_blocked_before": True,
                    "consecutive_blocked_before": 2,
                    "consecutive_blocked_after": 0,
                    "realized_path_m": 0.0,
                },
            ],
        )
        with self.assertRaises(RUNNER.ProtocolError):
            RUNNER.adjudicate_gate_c(
                malformed_after_semantic_failure, self.thresholds()["c"]
            )
        malformed = self.gate_c(
            blocked_state_count=1,
            recover_trigger_count=1,
            recover_outcomes=[{
                "episode_id": "e-1",
                "decision_index": "not-an-int",
                "actionable_blocked_before": True,
                "consecutive_blocked_before": 2,
                "consecutive_blocked_after": 0,
                "realized_path_m": 0.0,
            }],
        )
        with self.assertRaises(RUNNER.ProtocolError):
            RUNNER.adjudicate_gate_c(malformed, self.thresholds()["c"])

    def test_validation_split_access_is_protocol_error(self):
        with self.assertRaises(RUNNER.ProtocolError):
            RUNNER.adjudicate_gate_b(
                self.gate_b(split_read_counts={"train": 1, "val_seen": 1, "val_unseen": 0}),
                self.thresholds()["b"],
            )

    def test_gate_payload_numeric_types_are_strict(self):
        with self.assertRaises(RUNNER.ProtocolError):
            RUNNER.adjudicate_gate_b(
                self.gate_b(split_read_counts={"train": 1, "val_seen": "0", "val_unseen": 0}),
                self.thresholds()["b"],
            )
        with self.assertRaises(RUNNER.ProtocolError):
            RUNNER.adjudicate_gate_b(
                self.gate_b(token_top1_accuracy="0.8"),
                self.thresholds()["b"],
            )
        malformed = self.gate_c(
            blocked_state_count=1,
            recover_trigger_count=1,
            recover_outcomes=[{
                "episode_id": "e-1",
                "decision_index": 0.0,
                "actionable_blocked_before": True,
                "consecutive_blocked_before": 2,
                "consecutive_blocked_after": 0,
                "realized_path_m": 0.0,
            }],
        )
        with self.assertRaises(RUNNER.ProtocolError):
            RUNNER.adjudicate_gate_c(malformed, self.thresholds()["c"])


class TestP8DiagnosticStateMachine(unittest.TestCase):
    def write_manifest(self, root):
        manifest = TestP8Manifest().build_fixture(root)
        manifest["implementation"]["files"] = [
            {
                "logical_name": f"fixture-{index}",
                "path": str(root / f"fixture-{index}.py"),
                "sha256": "0" * 64,
                "normalization": "none",
            }
            for index in range(15)
        ]
        path = root / "manifest.json"
        path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
        return path

    @staticmethod
    def preload_receipt(manifest_path):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = (
            manifest["episodes"]["tiny_train"]
            + manifest["episodes"]["train_development"]
        )
        return {
            "schema_version": 1,
            "kind": RUNNER.PRELOAD_PROBE_KIND,
            "episode_count": 16,
            "scene_count": 16,
            "identities": [
                {
                    "index": row["index"],
                    "episode_id": row["episode_id"],
                    "scene": row["scene"],
                }
                for row in rows
            ],
            "observation_keys": RUNNER.PRELOAD_OBSERVATION_KEYS,
            "rgb_shape": [1, 64, 64, 3],
            "rgb_dtype": "torch.uint8",
            "observation_valid": True,
            "expert_waypoint_valid": True,
            "model_load_count": 0,
            "probe_split_read_counts": {
                "train": 16,
                "val_seen": 0,
                "val_unseen": 0,
            },
            "scientific_split_read_counts": {
                "train": 0,
                "val_seen": 0,
                "val_unseen": 0,
            },
        }

    def test_gate_a_failure_stops_before_training_and_writes_completion(self):
        gate_test = TestP8GateAdjudication()
        calls = {"b": 0, "c": 0}

        def forbidden(name):
            def call():
                calls[name] += 1
                raise AssertionError(f"Gate {name} must not run")
            return call

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.write_manifest(root)
            output = root / "result"
            result = RUNNER.run_diagnostic(
                manifest_path=manifest,
                output_dir=output,
                expected_manifest_sha256=sha256(manifest),
                gate_a_provider=lambda: gate_test.gate_a(mismatch=1),
                gate_b_provider=forbidden("b"),
                gate_c_provider=forbidden("c"),
            )
            self.assertEqual(result["decision"], "stopped_at_a")
            self.assertFalse(result["eligible_for_full_p8"])
            self.assertEqual(calls, {"b": 0, "c": 0})
            self.assertTrue((output / "result.json").is_file())
            self.assertTrue((output / "artifact_inventory.json").is_file())
            self.assertTrue((output / "COMPLETED.json").is_file())
            self.assertTrue((output / "gate_a_raw.json").is_file())
            self.assertFalse((output / "gate_b_bc_learnability.json").exists())

    def test_all_gates_pass_and_only_then_set_eligibility(self):
        gate_test = TestP8GateAdjudication()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.write_manifest(root)
            output = root / "result"
            result = RUNNER.run_diagnostic(
                manifest_path=manifest,
                output_dir=output,
                expected_manifest_sha256=sha256(manifest),
                gate_a_provider=gate_test.gate_a,
                gate_b_provider=gate_test.gate_b,
                gate_c_provider=gate_test.gate_c,
                source_heads={"stream": "original", "rlinf": "fixed"},
            )
            self.assertEqual(result["decision"], "pass")
            self.assertTrue(result["eligible_for_full_p8"])
            self.assertEqual(result["completed_gates"], ["a", "b", "c"])
            self.assertEqual(result["split_read_counts"]["val_seen"], 0)
            self.assertEqual(result["split_read_counts"]["val_unseen"], 0)

    def test_atomic_start_and_preload_probe_precede_gate_a(self):
        gate_test = TestP8GateAdjudication()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.write_manifest(root)
            output = root / "result"
            events = []

            def preload():
                self.assertTrue((output / "RUN_STARTED.json").is_file())
                self.assertFalse((output / "gate_a_raw.json").exists())
                events.append("preload")
                return self.preload_receipt(manifest)

            def gate_a():
                self.assertTrue((output / "PRELOAD_ENV_PROBE.json").is_file())
                events.append("gate_a")
                return gate_test.gate_a()

            result = RUNNER.run_diagnostic(
                manifest_path=manifest,
                output_dir=output,
                expected_manifest_sha256=sha256(manifest),
                preload_provider=preload,
                gate_a_provider=gate_a,
                gate_b_provider=gate_test.gate_b,
                gate_c_provider=gate_test.gate_c,
            )
            self.assertEqual(events, ["preload", "gate_a"])
            self.assertEqual(result["decision"], "pass")
            self.assertEqual(
                result["preload_probe_split_read_counts"],
                {"train": 16, "val_seen": 0, "val_unseen": 0},
            )

    def test_invalid_preload_probe_stops_before_gate_a(self):
        gate_test = TestP8GateAdjudication()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.write_manifest(root)
            output = root / "result"
            receipt = self.preload_receipt(manifest)
            receipt["model_load_count"] = 1
            with self.assertRaises(RUNNER.ProtocolError):
                RUNNER.run_diagnostic(
                    manifest_path=manifest,
                    output_dir=output,
                    expected_manifest_sha256=sha256(manifest),
                    preload_provider=lambda: receipt,
                    gate_a_provider=lambda: self.fail("Gate A must not run"),
                    gate_b_provider=gate_test.gate_b,
                    gate_c_provider=gate_test.gate_c,
                )
            self.assertTrue((output / "RUN_STARTED.json").is_file())
            self.assertFalse((output / "gate_a_raw.json").exists())
            self.assertFalse((output / "COMPLETED.json").exists())

    def test_protocol_error_has_no_completion_marker(self):
        gate_test = TestP8GateAdjudication()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.write_manifest(root)
            output = root / "result"
            with self.assertRaises(RUNNER.ProtocolError):
                RUNNER.run_diagnostic(
                    manifest_path=manifest,
                    output_dir=output,
                    expected_manifest_sha256="0" * 64,
                    gate_a_provider=gate_test.gate_a,
                    gate_b_provider=gate_test.gate_b,
                    gate_c_provider=gate_test.gate_c,
                )
            self.assertFalse((output / "COMPLETED.json").exists())

    def test_preflight_only_never_constructs_live_provider(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.write_manifest(root)
            output = root / "preflight"
            pinned = RUNNER.MANIFEST_SHA256
            RUNNER.MANIFEST_SHA256 = sha256(manifest)
            try:
                receipt = {
                    "source_heads": {"stream": "s", "rlinf": "r", "habitat_sim": "h"},
                    "identity": {"torch_imported": False, "model_constructed": False},
                }
                with mock.patch.object(RUNNER, "verify_preload_contract", return_value=receipt), mock.patch.object(
                    RUNNER, "_live_providers", side_effect=AssertionError("provider must not be constructed")
                ):
                    code = RUNNER.main([
                        "--manifest", str(manifest),
                        "--output-dir", str(output),
                        "--preflight-only",
                    ])
            finally:
                RUNNER.MANIFEST_SHA256 = pinned
            self.assertEqual(code, 0)
            result = json.loads((output / "PRELOAD_PREFLIGHT.json").read_text(encoding="utf-8"))
            self.assertFalse(result["torch_imported"])
            self.assertFalse(result["provider_constructed"])
            self.assertEqual(result["model_load_count"], 0)


if __name__ == "__main__":
    unittest.main()
