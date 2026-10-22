from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


MODULE_DIR = Path(__file__).resolve().parents[1] / "integrations" / "rlinf_v03"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import navigation_rlt_compare as compare  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


class NavigationRLTCompareHandoffTests(unittest.TestCase):
    def test_atomic_writer_preserves_competing_sentinel(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "matched.json"
            real_link = compare.os.link

            def competing_link(source, target, **kwargs):
                Path(target).write_bytes(b"sentinel")
                return real_link(source, target, **kwargs)

            with patch.object(compare.os, "link", side_effect=competing_link):
                with self.assertRaises(FileExistsError):
                    compare._atomic_write(destination, "replacement\n")
            self.assertEqual(destination.read_bytes(), b"sentinel")
            self.assertEqual(list(root.glob("matched.json.staging-*")), [])

    def _receipt(
        self,
        arm: str,
        *,
        run_id: str,
        collection_sha: str,
        plan_sha: str,
        split_sha: str,
        model: dict[str, object],
        checkpoint_sha: str | None,
        evaluation_commit: str,
        inventory_sha: str,
    ) -> dict[str, object]:
        behavior = compare.BEHAVIOR_BY_ARM[arm]
        if arm == compare.ARM_NATIVE:
            version = 0
            source = {"kind": "native", "lightnav_revision": "1" * 40}
        else:
            checkpoint_version = 1 if arm == compare.ARM_A3 else 2
            version = (
                100_000 + checkpoint_version
                if arm == compare.ARM_A3
                else 1_000_000 + checkpoint_version
            )
            source = {
                "kind": behavior,
                "checkpoint_sha256": checkpoint_sha,
                "checkpoint_policy_version": checkpoint_version,
            }
        contextual = arm == compare.ARM_RLT
        rows = []
        for index in range(32):
            rows.append(
                {
                    "episode_order": index,
                    "episode_id": str(index),
                    "scene": f"scene_{index // 4}",
                    "instruction_text_sha256": f"{index + 1:064x}",
                    "behavior": behavior,
                    "behavior_policy_version": version,
                    "success": False,
                    "spl": 0.0,
                    "final_navigation_error_m": 1.0,
                    "path_length_m": 0.0,
                    "collision_count": 0.0,
                    "blocked_step_count": 0,
                    "stop_source": "none",
                    "timeout": True,
                    "actor_decision_count": 1,
                    "requested_mode_distribution": {
                        "KEEP": 1,
                        "ADJUST": 0,
                        "STOP": 0,
                        "RECOVER": 0,
                    },
                    "executed_mode_distribution": {
                        "KEEP": 1,
                        "ADJUST": 0,
                        "STOP": 0,
                        "RECOVER": 0,
                    },
                    "safety_override_count": 0,
                    "context_request_count": 1 if contextual else 0,
                    "model_latency": {
                        "count": 1,
                        "mean_ms": 1.0,
                        "p50_ms": 1.0,
                        "p95_ms": 1.0,
                        "max_ms": 1.0,
                    },
                    "max_consecutive_recover": 0,
                }
            )
        aggregate = {
            "episode_count": 32,
            "success_count": 0,
            "success_rate": 0.0,
            "SR": 0.0,
            "mean_spl": 0.0,
            "SPL": 0.0,
            "mean_final_navigation_error_m": 1.0,
            "final_NE_m": 1.0,
            "mean_path_length_m": 0.0,
            "path_m": 0.0,
            "collision_count": 0.0,
            "collisions": 0.0,
            "blocked_step_count": 0,
            "blocked": 0,
            "native_stop_count": 0,
            "learned_stop_count": 0,
            "timeout_count": 32,
            "timeouts": 32,
            "actor_decision_count": 32,
            "safety_override_count": 0,
            "requested_mode_distribution": {
                "KEEP": 32,
                "ADJUST": 0,
                "STOP": 0,
                "RECOVER": 0,
            },
            "executed_mode_distribution": {
                "KEEP": 32,
                "ADJUST": 0,
                "STOP": 0,
                "RECOVER": 0,
            },
            "latency": {
                "count": 32,
                "mean_ms": 1.0,
                "p50_ms": 1.0,
                "p95_ms": 1.0,
                "max_ms": 1.0,
            },
            "model_latency": {
                "count": 32,
                "mean_ms": 1.0,
                "p50_ms": 1.0,
                "p95_ms": 1.0,
                "max_ms": 1.0,
            },
        }
        feature = {
            "source_sha256": "9" * 64 if contextual else None,
            "artifact_capture_rule": "capture" if contextual else None,
            "prefix_feature_schema": "prefix" if contextual else None,
            "prefix_selection_schema": "selection" if contextual else None,
            "prefix_selection_rule": "rule" if contextual else None,
        }
        return {
            "schema_name": compare.INPUT_SCHEMA,
            "schema_version": compare.INPUT_VERSION,
            "status": "completed",
            "run_id": run_id,
            "run_manifest_sha256": collection_sha,
            "run_manifest": {"sha256": collection_sha},
            "evaluation_code": {
                "commit": evaluation_commit,
                "inventory_sha256": inventory_sha,
            },
            "behavior": behavior,
            "behavior_policy_id": compare.BEHAVIOR_POLICY_ID,
            "behavior_policy_version": version,
            "behavior_source": source,
            "deterministic_policy": True,
            "split_label": "train_development",
            "upstream_split_read": "train",
            "validation_split_reads": 0,
            "training_replay_written": False,
            "seed": 2026100202,
            "evaluation_plan": {
                "sha256": plan_sha,
                "split_manifest_sha256": split_sha,
            },
            "executor": {
                "schema": "executor",
                "max_physical_steps": 160,
                "jpeg_quality": 90,
                "recover_loop_definition": "executed consecutive RECOVER >= 2",
            },
            "schemas": {
                "reference": "reference",
                "topology": "topology",
                "topology_node_dim": 1,
                "action": "action",
            },
            "model_identity": model,
            "context": {
                "requested": contextual,
                "request_count": 32 if contextual else 0,
                **feature,
                "full_artifacts_cleaned": contextual,
            },
            "episodes": rows,
            "aggregate": aggregate,
        }

    def test_comparator_emits_archive_binding_and_rejects_tampered_eval_identity(self) -> None:
        run_id = "navigation-rlt-test"
        collection_sha = "a" * 64
        plan_sha = "b" * 64
        split_sha = "c" * 64
        a3_sha = "d" * 64
        rlt_sha = "e" * 64
        commit = "f" * 40
        model = {
            "lightnav_source_revision": "1" * 40,
            "lightnav_hook_sha256": "2" * 64,
            "lightnav_weight_sha256": "3" * 64,
            "tokenizer_sha256": "4" * 64,
            "processor_sha256": "5" * 64,
            "history_config_sha256": "6" * 64,
            "original_release_weights": True,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inventory = _write(
                root / "inventory.json",
                {
                    "schema_name": compare.EVALUATION_INVENTORY_SCHEMA,
                    "schema_version": compare.EVALUATION_INVENTORY_VERSION,
                    "navigation_code_commit": commit,
                    "files": {
                        name: _sha(compare._evaluation_source(name))
                        for name in compare.EVALUATION_CODE_FILES
                    },
                },
            )
            inventory_sha = _sha(inventory)
            paths = {}
            for arm, checkpoint in (
                (compare.ARM_NATIVE, None),
                (compare.ARM_A3, a3_sha),
                (compare.ARM_RLT, rlt_sha),
            ):
                paths[arm] = _write(
                    root / f"{arm}.json",
                    self._receipt(
                        arm,
                        run_id=run_id,
                        collection_sha=collection_sha,
                        plan_sha=plan_sha,
                        split_sha=split_sha,
                        model=model,
                        checkpoint_sha=checkpoint,
                        evaluation_commit=commit,
                        inventory_sha=inventory_sha,
                    ),
                )
            native = json.loads(paths[compare.ARM_NATIVE].read_text(encoding="utf-8"))
            protocol_sha = compare._object_sha256(
                compare._protocol_contract(
                    native,
                    arm=compare.ARM_NATIVE,
                    expected_run_id=run_id,
                    expected_eval_plan_sha256=plan_sha,
                    expected_split_manifest_sha256=split_sha,
                    expected_seed=2026100202,
                )
            )
            output_json = root / "matched.json"
            output_markdown = root / "matched.md"
            arguments = [
                "--native", str(paths[compare.ARM_NATIVE]),
                "--a3", str(paths[compare.ARM_A3]),
                "--navigation-rlt", str(paths[compare.ARM_RLT]),
                "--expected-run-id", run_id,
                "--expected-run-manifest-sha256", collection_sha,
                "--expected-eval-plan-sha256", plan_sha,
                "--expected-split-manifest-sha256", split_sha,
                "--expected-protocol-sha256", protocol_sha,
                "--expected-model-sha256", compare._object_sha256(model),
                "--expected-lightnav-weight-sha256", str(model["lightnav_weight_sha256"]),
                "--expected-a3-checkpoint-sha256", a3_sha,
                "--expected-rlt-checkpoint-sha256", rlt_sha,
                "--expected-seed", "2026100202",
                "--evaluation-code-inventory", str(inventory),
                "--expected-evaluation-code-inventory-sha256", inventory_sha,
                "--output-json", str(output_json),
                "--output-markdown", str(output_markdown),
            ]
            self.assertEqual(compare.main(arguments), 0)
            output = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(output["run_id"], run_id)
            self.assertEqual(output["collection_manifest_sha256"], collection_sha)
            self.assertEqual(output["evaluation_code_commit"], commit)
            self.assertEqual(output["evaluation_code_inventory_sha256"], inventory_sha)
            self.assertEqual(output["navigation_rlt_checkpoint_sha256"], rlt_sha)

            tampered = json.loads(paths[compare.ARM_RLT].read_text(encoding="utf-8"))
            tampered["evaluation_code"]["commit"] = "0" * 40
            _write(paths[compare.ARM_RLT], tampered)
            arguments[-4:] = [
                "--output-json", str(root / "tampered.json"),
                "--output-markdown", str(root / "tampered.md"),
            ]
            with self.assertRaises(compare.ComparisonError):
                compare.main(arguments)


if __name__ == "__main__":
    unittest.main()
