from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


MODULE_DIR = Path(__file__).resolve().parents[1] / "integrations" / "rlinf_v03"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import navigation_rlt_obs_archive as archive  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


class NavigationRLTArchiveProvenanceTests(unittest.TestCase):
    def test_atomic_writer_preserves_competing_sentinel(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "archive.json"
            real_link = archive.os.link

            def competing_link(source, target, **kwargs):
                Path(target).write_bytes(b"sentinel")
                return real_link(source, target, **kwargs)

            with patch.object(archive.os, "link", side_effect=competing_link):
                with self.assertRaises(FileExistsError):
                    archive._atomic_json_new(destination, {"replacement": True})
            self.assertEqual(destination.read_bytes(), b"sentinel")
            self.assertEqual(list(root.glob("archive.json.staging-*")), [])

    def test_directory_publisher_uses_renameat2_noreplace(self) -> None:
        calls: list[tuple[object, ...]] = []

        class RenameAt2:
            argtypes = None
            restype = None

            def __call__(self, *args):
                calls.append(args)
                archive.ctypes.set_errno(archive.errno.EEXIST)
                return -1

        class Libc:
            renameat2 = RenameAt2()

        with patch.object(archive.sys, "platform", "linux"), patch.object(
            archive.ctypes, "CDLL", return_value=Libc()
        ):
            with self.assertRaises(FileExistsError):
                archive._renameat2_noreplace(Path("staging"), Path("final"))

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][-1], 1)

    def test_directory_publisher_preserves_race_created_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staging = root / ".run.partial"
            destination = root / "run"
            staging.mkdir()
            (staging / "payload").write_bytes(b"payload")

            def competing_publish(_source: Path, target: Path) -> None:
                target.mkdir()
                (target / "sentinel").write_bytes(b"sentinel")
                raise FileExistsError(target)

            with patch.object(
                archive, "_renameat2_noreplace", side_effect=competing_publish
            ):
                with self.assertRaises(FileExistsError):
                    archive._publish_directory_new(staging, destination)

            self.assertEqual((destination / "sentinel").read_bytes(), b"sentinel")
            self.assertEqual((staging / "payload").read_bytes(), b"payload")

    def test_second_directory_publish_collision_rolls_first_back(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint_parent = root / "checkpoints"
            result_parent = root / "results"
            checkpoint_parent.mkdir()
            result_parent.mkdir()
            checkpoint_staging = checkpoint_parent / ".run.partial"
            checkpoint_final = checkpoint_parent / "run"
            result_staging = result_parent / ".run.partial"
            result_final = result_parent / "run"
            checkpoint_staging.mkdir()
            result_staging.mkdir()
            result_final.mkdir()
            (checkpoint_staging / "checkpoint").write_bytes(b"checkpoint")
            (result_staging / "result").write_bytes(b"result")
            (result_final / "sentinel").write_bytes(b"sentinel")

            def rename_new(source: Path, destination: Path) -> None:
                if archive._lexists(destination):
                    raise FileExistsError(destination)
                source.rename(destination)

            with patch.object(
                archive, "_renameat2_noreplace", side_effect=rename_new
            ):
                with self.assertRaises(FileExistsError):
                    archive._publish_archive_pair(
                        checkpoint_staging,
                        checkpoint_final,
                        result_staging,
                        result_final,
                    )

            self.assertFalse(checkpoint_final.exists())
            self.assertEqual(
                (checkpoint_staging / "checkpoint").read_bytes(), b"checkpoint"
            )
            self.assertEqual((result_staging / "result").read_bytes(), b"result")
            self.assertEqual((result_final / "sentinel").read_bytes(), b"sentinel")

    def test_evaluation_inventory_and_matched_inputs_bind_directly(self) -> None:
        commit = "a" * 40
        collection_sha = "b" * 64
        checkpoint_sha = "c" * 64
        files = {
            name: _sha(
                archive._evaluation_source(
                    MODULE_DIR, Path(name), f"evaluation source {name}"
                )[0]
            )
            for name in archive.EVALUATION_CODE_FILES
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inventory_path = _write(
                root / "evaluation_code_inventory.json",
                {
                    "schema_name": archive.EVALUATION_INVENTORY_SCHEMA,
                    "schema_version": 1,
                    "navigation_code_commit": commit,
                    "files": files,
                },
            )
            identity, _ = archive._evaluation_code_evidence(
                inventory_path, _sha(inventory_path)
            )
            self.assertEqual(identity["commit"], commit)
            self.assertNotIn("source_root", identity)

            inventory_value = {
                "schema_name": archive.EVALUATION_INVENTORY_SCHEMA,
                "schema_version": 1,
                "navigation_code_commit": commit,
                "files": files,
            }
            inventory_cases = {
                "missing": {
                    **inventory_value,
                    "files": {
                        name: digest
                        for name, digest in files.items()
                        if name != archive.EVALUATION_CODE_FILES[-1]
                    },
                },
                "extra": {
                    **inventory_value,
                    "files": {**files, "unexpected.py": "d" * 64},
                },
                "tampered": {
                    **inventory_value,
                    "files": {
                        **files,
                        archive.EVALUATION_CODE_FILES[0]: "e" * 64,
                    },
                },
            }
            for name, value in inventory_cases.items():
                with self.subTest(inventory=name):
                    candidate = _write(root / f"{name}.json", value)
                    with self.assertRaises(archive.ArchiveError):
                        archive._evaluation_code_evidence(candidate, _sha(candidate))

            run_root = root / "run"
            inputs: dict[str, object] = {}
            for arm in archive.EVALUATION_ARMS:
                receipt = _write(
                    run_root / "evaluations" / arm / "evaluation.json",
                    {
                        "schema_name": "navigation_rlt_matched_train_development_eval",
                        "schema_version": 1,
                        "status": "completed",
                        "run_id": "run",
                        "run_manifest_sha256": collection_sha,
                        "collection_manifest_sha256": collection_sha,
                        "evaluation_code": {
                            "commit": commit,
                            "inventory_sha256": identity["inventory_sha256"],
                        },
                    },
                )
                inputs[arm] = {"path": str(receipt), "sha256": _sha(receipt)}
            matched = {
                "schema_name": archive.MATCHED_SCHEMA,
                "schema_version": archive.MATCHED_VERSION,
                "status": "completed",
                "run_id": "run",
                "collection_manifest_sha256": collection_sha,
                "evaluation_code_commit": commit,
                "evaluation_code_inventory_sha256": identity["inventory_sha256"],
                "navigation_rlt_checkpoint_sha256": checkpoint_sha,
                "checkpoints": {"navigation_rlt": {"sha256": checkpoint_sha}},
                "matched_contract": {"run_manifest_sha256": collection_sha},
                "inputs": inputs,
            }
            self.assertEqual(
                archive._matched_checkpoint_sha256(matched, run_id="run"),
                checkpoint_sha,
            )
            artifacts = archive._matched_input_evidence(
                matched,
                run_root=run_root,
                run_id="run",
                collection_manifest_sha256=collection_sha,
                evaluation_code=identity,
            )
            self.assertEqual(len(artifacts), 3)

            matched["checkpoints"]["navigation_rlt"]["sha256"] = "d" * 64
            with self.assertRaises(archive.ArchiveError):
                archive._matched_checkpoint_sha256(matched, run_id="run")

    def test_explicit_evaluation_source_root_is_exact_and_has_no_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_root = root / "eval-code"
            files: dict[str, str] = {}
            for name in archive.EVALUATION_CODE_FILES:
                source = source_root / Path(name)
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_bytes(f"source:{name}".encode("utf-8"))
                files[name] = _sha(source)
            inventory = _write(
                root / "inventory.json",
                {
                    "schema_name": archive.EVALUATION_INVENTORY_SCHEMA,
                    "schema_version": 1,
                    "navigation_code_commit": "a" * 40,
                    "files": files,
                },
            )
            identity, artifacts = archive._evaluation_code_evidence(
                inventory,
                _sha(inventory),
                source_root=source_root,
            )
            self.assertEqual(identity["source_root"], str(source_root.resolve()))
            self.assertEqual(len(artifacts), len(files) + 1)

            missing = source_root / "navrl" / "__init__.py"
            missing.unlink()
            with self.assertRaisesRegex(archive.ArchiveError, "missing or ambiguous"):
                archive._evaluation_code_evidence(
                    inventory,
                    _sha(inventory),
                    source_root=source_root,
                )

    def test_successor_arguments_bind_formal_marker_and_shared_helper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "navigation-rlt-test"
            formal_root = run_root / "stage2-successors" / "formal-v1"
            marker = formal_root / "navigation_rlt_stage2" / "STAGE.json"
            marker.parent.mkdir(parents=True)
            marker.write_text("{}\n", encoding="utf-8")
            formal = _write(
                formal_root / "FORMAL_COMPLETED.json",
                {
                    "stage_marker": {"path": str(marker), "sha256": _sha(marker)}
                },
            )
            args = SimpleNamespace(
                run_id=run_root.name,
                stage2_formal_completion=formal,
                expected_stage2_formal_completion_sha256=_sha(formal),
                stage2_source_root=formal_root / "sources",
                evaluation_source_root=formal_root / "eval-code",
            )
            expected = {
                "stage_run_root": str(formal_root),
                "formal_completion": {"path": str(formal), "sha256": _sha(formal)},
            }
            with patch.object(
                archive.post_training, "_stage2_evidence", return_value=expected
            ) as shared:
                self.assertEqual(
                    archive._stage2_successor_evidence(args, run_root=run_root),
                    expected,
                )
            shared.assert_called_once_with(
                marker,
                expected_marker_sha256=_sha(marker),
                run_root=run_root,
                run_id=run_root.name,
                formal_completion_path=formal,
                expected_formal_completion_sha256=_sha(formal),
                source_root=args.stage2_source_root,
            )

            args.evaluation_source_root = None
            with self.assertRaisesRegex(archive.ArchiveError, "all present"):
                archive._stage2_successor_evidence(args, run_root=run_root)

    def test_behavior_kl_additions_are_strict(self) -> None:
        value = {
            "objective_schema": archive.STAGE2_BEHAVIOR_KL_OBJECTIVE_SCHEMA,
            "cql_alpha": 1.0,
            "cql_temperature": 1.0,
            "minimum_behavior_gate_count": 1,
            "behavior_kl_weight": 1.0,
        }
        self.assertEqual(
            archive._stage2_training_additions(value, name="behavior KL"), value
        )
        with self.assertRaisesRegex(archive.ArchiveError, "weight"):
            archive._stage2_training_additions(
                {**value, "behavior_kl_weight": 0.5}, name="behavior KL"
            )
        with self.assertRaisesRegex(archive.ArchiveError, "fields"):
            archive._stage2_training_additions(
                {**value, "unexpected": True}, name="behavior KL"
            )

    def test_training_protocol_binds_actual_stage_gate_model_and_collection(self) -> None:
        run_id = "navigation-rlt-test"
        collection_commit = "1" * 40
        training_commit = "2" * 40
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary) / run_id
            code_root = run_root / "code"
            code_root.mkdir(parents=True)
            model_protocol = {
                "attention_heads": 8,
                "critic_hidden_dim": 256,
                "dropout": 0.0,
                "feedforward_dim": 768,
                "initial_log_std": -1.5,
                "max_log_std": 1.0,
                "max_prefix_tokens": 192,
                "min_log_std": -5.0,
                "residual_forward_limit_m": 0.05,
                "residual_yaw_limit_rad": 0.17453292519943295,
                "transformer_layers": 2,
            }
            model_config = {**model_protocol, "prefix_hidden_dim": 2560}
            model_contract = {
                "config": model_config,
                "reference_embedding_schema": "lightnav_reference_embedding_v1",
                "topology_packing_schema": "lightnav_topology_packing_v1",
                "canonical_action_schema": "navigation_rlt_action_v1",
            }
            stage1_protocol = {
                "seed": 2026100202,
                "epochs": 5,
                "batch_size_per_rank": 16,
                "learning_rate": 0.0001,
                "weight_decay": 0.0001,
                "clip_grad": 1.0,
                "prefix_mask_ratio": 0.25,
                "topology_mask_ratio": 0.25,
                "prefix_weight": 1.0,
                "topology_weight": 1.0,
                "required_world_size": 8,
                "expected_episode_count": 224,
            }
            stage2_protocol = {
                "seed": 2026100203,
                "epochs": 10,
                "batch_size_per_rank": 64,
                "learning_rate": 0.0003,
                "weight_decay": 0.0001,
                "clip_grad": 1.0,
                "reference_gate_weight": 0.25,
                "reference_residual_weight": 0.1,
                "reference_dropout_probability": 0.2,
                "target_tau": 0.005,
                "policy_version": 200001,
                "required_world_size": 8,
                "expected_episode_count": 256,
            }
            gate_protocol = {
                "seed": 20261002,
                "batch_size": 64,
                "prefix_mask_ratio": 0.25,
                "topology_mask_ratio": 0.25,
                "minimum_relative_improvement": 0.0,
                "minimum_z_rl_variance": 1e-8,
            }
            collection = _write(
                run_root / "provenance" / "collection.json",
                {
                    "schema_name": archive.COLLECTION_MANIFEST_SCHEMA,
                    "schema_version": 1,
                    "run_id": run_id,
                    "boundaries": {
                        "upstream_split_read": "train",
                        "validation_split_reads": 0,
                        "lightnav_frozen": True,
                        "weight_origin": "original_release",
                        "ttt_or_secondary_post_training": False,
                        "remote_push": False,
                    },
                    "data": {
                        "split_manifest_sha256": "4" * 64,
                        "parallel_plan_sha256": "5" * 64,
                        "worker_materialization_manifest_sha256": "6" * 64,
                        "update_episode_count": 256,
                        "development_episode_count": 32,
                        "worker_count": 8,
                        "episodes_per_worker": 32,
                    },
                    "behavior": {
                        "name": "a3_stochastic",
                        "checkpoint_sha256": "7" * 64,
                        "seed": 2026100201,
                    },
                    "rlinf": {"source_commit": "8" * 40, "required_world_size": 8},
                    "code": {
                        "git_commit": collection_commit,
                        "files": {"navigation_rlt_rollout.py": "3" * 64},
                    },
                },
            )
            collection_sha = _sha(collection)
            code_archive = run_root / "provenance" / "training-code.tar.gz"
            code_archive.write_bytes(b"verified training bundle")
            code_archive_sha = _sha(code_archive)
            inventory = _write(
                run_root / "provenance" / "code_inventory.json",
                {
                    "schema_name": archive.TRAINING_INVENTORY_SCHEMA,
                    "schema_version": 1,
                    "navigation_code_commit": training_commit,
                    "archive_name": code_archive.name,
                    "archive_sha256": code_archive_sha,
                    "archive_size_bytes": code_archive.stat().st_size,
                    "file_count": 0,
                    "files": [],
                },
            )
            verification = _write(
                run_root / "provenance" / "code_verification.json",
                {
                    "schema_name": archive.TRAINING_CODE_VERIFICATION_SCHEMA,
                    "schema_version": 1,
                    "status": "passed",
                    "navigation_code_commit": training_commit,
                    "supersedes_navigation_code_commit": collection_commit,
                    "code_root": str(code_root),
                    "file_count": 0,
                    "inventory_path": str(inventory),
                    "inventory_sha256": _sha(inventory),
                    "archive_path": str(code_archive),
                    "archive_sha256": code_archive_sha,
                },
            )

            stage1_checkpoint = run_root / "stage1.pt"
            stage2_checkpoint = run_root / "stage2.pt"
            stage1_checkpoint.write_bytes(b"stage1")
            stage2_checkpoint.write_bytes(b"stage2")
            stage1_run_sha = "9" * 64
            stage1_replay_sha = "a" * 64
            stage1_training = {
                key: value
                for key, value in stage1_protocol.items()
                if key not in {"seed", "required_world_size", "expected_episode_count"}
            }
            stage2_training = {
                key: value
                for key, value in stage2_protocol.items()
                if key not in {"seed", "required_world_size", "expected_episode_count"}
            }
            gate_receipt = _write(
                run_root / "stage1_gate_receipt.json",
                {
                    "schema_name": "navigation_rlt_stage1_gate",
                    "schema_version": 1,
                    "status": "passed",
                    "operation": "evaluate",
                    "split": {
                        "train_episode_count": 224,
                        "dev_episode_count": 32,
                        "train_manifest_sha256": stage1_replay_sha,
                    },
                    "checkpoint": {
                        "sha256": _sha(stage1_checkpoint),
                        "run_manifest_sha256": stage1_run_sha,
                        "replay_manifest_sha256": stage1_replay_sha,
                        "config": model_config,
                    },
                    "evaluation_contract": {
                        "state_scope": "current_and_next_transition_states",
                        "baseline": "representation_train_featurewise_constant_mean",
                        "mask_algorithm": "seeded-mask-v1",
                        **gate_protocol,
                        "device": "cuda:0",
                    },
                },
            )

            def stage(
                phase: str,
                checkpoint: Path,
                section: dict[str, object],
                training: dict[str, object],
                episode_count: int,
            ) -> archive.StageEvidence:
                return archive.StageEvidence(
                    phase=phase,
                    stage_root=run_root / phase,
                    checkpoint=checkpoint,
                    checkpoint_sha256=_sha(checkpoint),
                    training_result={"world_size": 8, "training": training},
                    run_manifest={
                        "run_id": run_id,
                        "phase": phase,
                        "seed": section["seed"],
                        "required_world_size": 8,
                        "model": model_contract,
                        "training": training,
                    },
                    replay_manifest={
                        "expected_episode_count": episode_count,
                        "completed_episode_count": episode_count,
                    },
                    replay_summary={},
                    run_manifest_sha256=(stage1_run_sha if phase == "stage1" else "b" * 64),
                    replay_manifest_sha256=(
                        stage1_replay_sha if phase == "stage1" else "c" * 64
                    ),
                    run_package_receipt={},
                    replay_package_receipt=(
                        {}
                        if phase == "stage1"
                        else {
                            "stage1_gate": {
                                "receipt_path": str(gate_receipt),
                                "receipt_sha256": _sha(gate_receipt),
                            }
                        }
                    ),
                    source_claims=[],
                    artifacts=[],
                )

            stage1 = stage(
                "stage1", stage1_checkpoint, stage1_protocol, stage1_training, 224
            )
            stage2 = stage(
                "stage2", stage2_checkpoint, stage2_protocol, stage2_training, 256
            )
            protocol_value = {
                "schema_name": archive.TRAINING_PROTOCOL_SCHEMA,
                "schema_version": 1,
                "run_id": run_id,
                "preregistration_commit": "d" * 40,
                "navigation_training_code_commit": training_commit,
                "boundaries": {
                    "lightnav_frozen": True,
                    "remote_push_allowed": False,
                    "ttt_allowed": False,
                    "validation_splits_allowed": False,
                },
                "rollout": {
                    "behavior": "a3",
                    "episode_count": 256,
                    "outer_manifest_sha256": collection_sha,
                    "parallel_plan_sha256": "5" * 64,
                    "split": "train",
                },
                "model": model_protocol,
                "stage1": stage1_protocol,
                "heldout_gate": gate_protocol,
                "stage2": stage2_protocol,
                "stage1_checkpoint_provenance_gate": {
                    "required": True,
                    "run_manifest_sha256_source": (
                        "navigation_rlt_stage1/STAGE.json:run_manifest.sha256"
                    ),
                    "replay_manifest_sha256_source": (
                        "navigation_rlt_stage1/STAGE.json:replay_manifest.sha256"
                    ),
                    "required_cli_arguments": [
                        "--expected-stage1-run-manifest-sha256",
                        "--expected-stage1-replay-manifest-sha256",
                    ],
                },
            }
            training_protocol = _write(
                run_root / "provenance" / "training_protocol.json", protocol_value
            )

            def validate(
                **repair: object,
            ) -> tuple[dict[str, object], list[archive.Artifact]]:
                return archive._training_provenance_evidence(
                    run_root=run_root,
                    run_id=run_id,
                    collection_manifest_path=collection,
                    expected_collection_sha256=collection_sha,
                    code_verification_path=verification,
                    expected_code_verification_sha256=_sha(verification),
                    training_protocol_path=training_protocol,
                    expected_training_protocol_sha256=_sha(training_protocol),
                    stages=[stage1, stage2],
                    training_source_inventory={"files": {}},
                    **repair,
                )

            evidence, artifacts = validate()
            self.assertEqual(evidence["collection_code_commit"], collection_commit)
            self.assertEqual(evidence["training_code_commit"], training_commit)
            self.assertEqual(len(artifacts), 5)

            stage2.run_manifest["training"]["target_tau"] = 0.25
            with self.assertRaises(archive.ArchiveError):
                validate()
            stage2.run_manifest["training"]["target_tau"] = stage2_protocol["target_tau"]

            repair_commit = "e" * 40
            repair_root = run_root / "stage2-repair-code"
            repair_root.mkdir()
            repair_hashes: dict[str, str] = {}
            repair_inventory_rows: list[dict[str, object]] = []
            for claim_name, filename in archive.SOURCE_FILES.items():
                source = repair_root / filename
                source.write_bytes(f"repair source {filename}".encode("utf-8"))
                repair_hashes[claim_name] = _sha(source)
                repair_inventory_rows.append(
                    {
                        "path": filename,
                        "sha256": _sha(source),
                        "size_bytes": source.stat().st_size,
                    }
                )
            stage2.source_claims = [
                {
                    claim_name: repair_hashes[claim_name]
                    for claim_name in archive.STAGE2_REPAIR_REQUIRED_SOURCE_CLAIMS
                }
            ]
            repair_archive = run_root / "provenance" / "stage2-repair-code.tar.gz"
            repair_archive.write_bytes(b"verified Stage2 repair bundle")
            repair_inventory = _write(
                run_root / "provenance" / "stage2-repair-inventory.json",
                {
                    "schema_name": archive.TRAINING_INVENTORY_SCHEMA,
                    "schema_version": 1,
                    "navigation_code_commit": repair_commit,
                    "archive_name": repair_archive.name,
                    "archive_sha256": _sha(repair_archive),
                    "archive_size_bytes": repair_archive.stat().st_size,
                    "file_count": len(repair_inventory_rows),
                    "files": repair_inventory_rows,
                },
            )
            repair_verification = _write(
                run_root / "provenance" / "stage2-repair-verification.json",
                {
                    "schema_name": archive.TRAINING_CODE_VERIFICATION_SCHEMA,
                    "schema_version": 1,
                    "status": "passed",
                    "navigation_code_commit": repair_commit,
                    "supersedes_navigation_code_commit": training_commit,
                    "code_root": str(repair_root),
                    "file_count": len(repair_inventory_rows),
                    "inventory_path": str(repair_inventory),
                    "inventory_sha256": _sha(repair_inventory),
                    "archive_path": str(repair_archive),
                    "archive_sha256": _sha(repair_archive),
                },
            )
            additions = {
                "objective_schema": archive.STAGE2_REPAIR_OBJECTIVE_SCHEMA,
                "cql_alpha": 1.0,
                "cql_temperature": 1.0,
                "minimum_behavior_gate_count": 1,
            }
            support = {
                "eligibility_schema": archive.STAGE2_ELIGIBILITY_SCHEMA,
                "gate_names": list(archive.STAGE2_GATE_NAMES),
                "gate_counts": [7, 4, 2, 0],
                "gate_support_mask": [True, True, True, False],
                "minimum_behavior_gate_count": 1,
            }
            repaired_training = {**stage2_training, **additions}
            stage2.run_manifest["training"] = repaired_training
            stage2.training_result["training"] = dict(repaired_training)
            stage2.run_manifest["stage2_support"] = support
            stage2.training_result["stage2_support"] = dict(support)
            stage2.training_result["rank_receipts"] = [
                {"data": dict(support)} for _ in range(8)
            ]
            stage2.training_result["final_checkpoint"] = {
                "metadata": {
                    "stage2_objective": (
                        "reference_conditioned_min_target_twin_q_with_finite_gate_cql_"
                        "and_native_reference_actor_anchor_dropout"
                    ),
                    "cql_enabled": True,
                    "cql_candidate_count": 4,
                    "entropy_target_enabled": False,
                    "gate_support_mask": [True, True, True, False],
                    "supported_gate_names": ["KEEP", "ADJUST", "STOP"],
                }
            }
            repair_protocol = _write(
                run_root / "provenance" / "stage2-repair-protocol.json",
                {
                    "schema_name": archive.STAGE2_REPAIR_PROTOCOL_SCHEMA,
                    "schema_version": 1,
                    "run_id": run_id,
                    "base_training_protocol_sha256": _sha(training_protocol),
                    "stage1_code_verification_sha256": _sha(verification),
                    "stage1_checkpoint_sha256": stage1.checkpoint_sha256,
                    "stage2_repair_code_verification_sha256": _sha(
                        repair_verification
                    ),
                    "stage2_replay_manifest_sha256": (
                        stage2.replay_manifest_sha256
                    ),
                    "stage2_run_manifest_sha256": stage2.run_manifest_sha256,
                    "navigation_training_code_commit": repair_commit,
                    "validation_split_reads": 0,
                    "stage2_training_additions": additions,
                    "stage2_support": support,
                },
            )
            repair_args = {
                "stage2_repair_code_verification_path": repair_verification,
                "expected_stage2_repair_code_verification_sha256": _sha(
                    repair_verification
                ),
                "stage2_repair_protocol_path": repair_protocol,
                "expected_stage2_repair_protocol_sha256": _sha(repair_protocol),
            }
            repaired_evidence, repaired_artifacts = validate(**repair_args)
            self.assertEqual(
                repaired_evidence["stage2_repair_code_commit"], repair_commit
            )
            self.assertEqual(
                set(repaired_evidence["training_source_identities"]),
                {"stage1", "stage2_repair"},
            )
            self.assertEqual(len(repaired_artifacts), 9)

            repair_protocol_value = json.loads(
                repair_protocol.read_text(encoding="utf-8")
            )
            repair_protocol_value["stage1_checkpoint_sha256"] = "0" * 64
            _write(repair_protocol, repair_protocol_value)
            repair_args["expected_stage2_repair_protocol_sha256"] = _sha(
                repair_protocol
            )
            with self.assertRaisesRegex(
                archive.ArchiveError, "source/boundary identity"
            ):
                validate(**repair_args)
            repair_protocol_value["stage1_checkpoint_sha256"] = (
                stage1.checkpoint_sha256
            )
            _write(repair_protocol, repair_protocol_value)
            repair_args["expected_stage2_repair_protocol_sha256"] = _sha(
                repair_protocol
            )

            tampered_source = repair_root / archive.SOURCE_FILES["policy_sha256"]
            original_source = tampered_source.read_bytes()
            tampered_source.write_bytes(b"tampered repair policy")
            with self.assertRaisesRegex(archive.ArchiveError, "live source differs"):
                validate(**repair_args)
            tampered_source.write_bytes(original_source)

            base_run_manifest_sha = stage2.run_manifest_sha256
            successor_additions = {
                **additions,
                "objective_schema": archive.STAGE2_BEHAVIOR_KL_OBJECTIVE_SCHEMA,
                "behavior_kl_weight": 1.0,
            }
            successor_training = {**stage2_training, **successor_additions}
            stage2.run_manifest["training"] = successor_training
            stage2.training_result["training"] = dict(successor_training)
            stage2.run_manifest_sha256 = "f" * 64
            successor_identity = {
                "base_run_manifest": {
                    "path": str(run_root / "navigation_rlt_stage2_prepare/run_manifest.json"),
                    "sha256": base_run_manifest_sha,
                },
                "base_preparation_claims": {
                    claim: repair_hashes[claim]
                    for claim in (
                        "trainer_sha256",
                        "policy_sha256",
                        "replay_sha256",
                        "preparer_sha256",
                    )
                },
                "replay_producer_claims": {
                    claim: repair_hashes[claim]
                    for claim in (
                        "promoter_sha256",
                        "policy_sha256",
                        "replay_sha256",
                    )
                },
            }
            stage2.successor = {
                "successor_source_identity": successor_identity,
                "repair_contract": {"training_additions": successor_additions},
            }
            successor_evidence, _ = validate(**repair_args)
            self.assertEqual(
                successor_evidence["training_source_identities"]["stage2_repair"][
                    "stage2_run_manifest_sha256"
                ],
                base_run_manifest_sha,
            )
            self.assertEqual(
                successor_evidence["training_source_identities"]["stage2_successor"],
                successor_identity,
            )
            successor_identity["base_preparation_claims"]["policy_sha256"] = "0" * 64
            with self.assertRaisesRegex(archive.ArchiveError, "historical source"):
                validate(**repair_args)
            successor_identity["base_preparation_claims"]["policy_sha256"] = (
                repair_hashes["policy_sha256"]
            )

            bad_support = dict(support)
            bad_support["gate_support_mask"] = [True, True, True, True]
            with self.assertRaisesRegex(archive.ArchiveError, "support mask"):
                archive._stage2_support_contract(
                    bad_support, name="bad Stage2 support"
                )
            bad_additions = dict(additions)
            bad_additions["cql_alpha"] = 0.5
            with self.assertRaisesRegex(archive.ArchiveError, "cql_alpha"):
                archive._stage2_training_additions(
                    bad_additions, name="bad Stage2 additions"
                )

            gate_value = json.loads(gate_receipt.read_text(encoding="utf-8"))
            gate_value["evaluation_contract"]["seed"] = 1
            _write(gate_receipt, gate_value)
            stage2.replay_package_receipt["stage1_gate"]["receipt_sha256"] = _sha(
                gate_receipt
            )
            with self.assertRaises(archive.ArchiveError):
                validate(**repair_args)

    def test_reviewed_post_training_inputs_archive_exact_files_and_hashes(self) -> None:
        run_id = "navigation-rlt-reviewed-inputs"
        checkpoint_sha = "1" * 64
        collection_sha = "2" * 64
        protocol_sha = "3" * 64
        evaluation_code = {"commit": "4" * 40, "inventory_sha256": "5" * 64}
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary) / run_id
            run_root.mkdir()
            update = [
                {
                    "episode_id": str(index),
                    "scene": f"update-{index}",
                    "instruction_text_sha256": "6" * 64,
                }
                for index in range(256)
            ]
            development = [
                {
                    "episode_id": str(index),
                    "scene": f"development-{index}",
                    "instruction_text_sha256": hashlib.sha256(
                        f"instruction-{index}".encode()
                    ).hexdigest(),
                }
                for index in range(32)
            ]
            split = _write(
                run_root / "data" / "split_manifest.json",
                {
                    "schema_name": "navigation_rlt_train_scene_split",
                    "schema_version": 1,
                    "upstream_split_read": "train",
                    "validation_split_reads": 0,
                    "splits": {
                        "train_update": {
                            "episode_count": 256,
                            "episodes": update,
                        },
                        "train_development": {
                            "episode_count": 32,
                            "dataset_sha256": "7" * 64,
                            "gt_sha256": "8" * 64,
                            "episodes": development,
                        },
                    },
                },
            )
            evaluation = _write(
                run_root / "evaluation_plan.json",
                {
                    "schema_name": "navigation_rlt_parallel_rollout_plan",
                    "schema_version": 1,
                    "run_id": run_id,
                    "split_label": "train_development",
                    "validation_split_reads": 0,
                    "episode_count": 32,
                    "split_manifest": {"path": str(split), "sha256": _sha(split)},
                    "worker_count": 1,
                    "workers": [
                        {
                            "worker_id": 0,
                            "episodes": [
                                {
                                    "episode_order": index,
                                    "episode_id": row["episode_id"],
                                    "scene": row["scene"],
                                    "instruction_text_sha256": row[
                                        "instruction_text_sha256"
                                    ],
                                }
                                for index, row in enumerate(development)
                            ],
                        }
                    ],
                },
            )
            matched = _write(run_root / "post-training" / "matched.json", {"ok": True})
            markdown = run_root / "post-training" / "matched.md"
            markdown.write_text("# Matched comparison\n", encoding="utf-8")
            plan_value = {
                "schema_name": "navigation_rlt_post_training_plan",
                "schema_version": 1,
                "status": "planned",
                "run_id": run_id,
                "a800_run_root": str(run_root),
                "stage2": {"checkpoint": {"sha256": checkpoint_sha}},
                "collection_manifest": {"sha256": collection_sha},
                "split_manifest": {"path": str(split), "sha256": _sha(split)},
                "evaluation_plan": {
                    "path": str(evaluation),
                    "sha256": _sha(evaluation),
                    "episode_count": 32,
                },
                "evaluation_code": evaluation_code,
                "training_protocol": {"sha256": protocol_sha},
                "outputs": {
                    "matched_result": str(matched),
                    "matched_markdown": str(markdown),
                    "completion": str(
                        run_root / "post-training" / "POST_TRAINING_COMPLETED.json"
                    ),
                },
            }
            plan = _write(run_root / "post-training" / "plan.json", plan_value)
            completion = _write(
                run_root / "post-training" / "POST_TRAINING_COMPLETED.json",
                {"status": "completed"},
            )

            def reviewed(
                expected_plan_sha: str = _sha(plan),
            ) -> tuple[dict[str, str], list[archive.Artifact]]:
                return archive._reviewed_post_training_inputs(
                    post_training_plan_path=plan,
                    expected_post_training_plan_sha256=expected_plan_sha,
                    evaluation_plan_path=evaluation,
                    expected_evaluation_plan_sha256=_sha(evaluation),
                    split_manifest_path=split,
                    expected_split_manifest_sha256=_sha(split),
                    matched_markdown_path=markdown,
                    expected_matched_markdown_sha256=_sha(markdown),
                    matched_result_path=matched.resolve(strict=True),
                    run_root=run_root.resolve(strict=True),
                    run_id=run_id,
                    stage2_checkpoint_sha256=checkpoint_sha,
                    collection_manifest_sha256=collection_sha,
                    collection_split_manifest_sha256=_sha(split),
                    training_protocol_sha256=protocol_sha,
                    evaluation_code=evaluation_code,
                )

            identity, artifacts = reviewed()
            self.assertEqual(identity["post_training_plan_sha256"], _sha(plan))
            self.assertEqual(len(artifacts), 4)
            self.assertNotIn(completion.resolve(), {row.source for row in artifacts})

            bad_plan = json.loads(json.dumps(plan_value))
            bad_plan["evaluation_plan"]["sha256"] = "9" * 64
            _write(plan, bad_plan)
            with self.assertRaises(archive.ArchiveError):
                reviewed(_sha(plan))


if __name__ == "__main__":
    unittest.main()
