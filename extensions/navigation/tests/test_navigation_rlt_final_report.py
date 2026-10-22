from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


MODULE_DIR = Path(__file__).resolve().parents[1] / "integrations" / "rlinf_v03"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import navigation_rlt_final_report as report  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return path


def _mode_summary(counts: dict[str, int]) -> dict[str, object]:
    total = sum(counts.values())
    return {
        "total": total,
        "counts": counts,
        "ratios": {name: count / total for name, count in counts.items()},
    }


def _a3_budget_receipt(root: Path) -> Path:
    return _write(
        root / report.A3_BUDGET_RECEIPT_FILENAME,
        {
            "schema_name": report.A3_BUDGET_RECEIPT_SCHEMA,
            "schema_version": 1,
            "status": "verified",
            "run_id": report.A3_RUN_ID,
            "scope": {
                "upstream_split_read": "train",
                "validation_split_reads": 0,
                "train_only": True,
            },
            "provenance": dict(report.A3_EXPECTED_PROVENANCE),
            "budget": {
                "rollout_episode_count": 64,
                "unique_actor_decisions": 4642,
                "actor_eligibility": dict(report.A3_ACTOR_ELIGIBILITY),
                "optimizer_epochs": 4,
                "optimizer_presentations": 18568,
                "padding_excluded": True,
            },
        },
    )


class NavigationRLTFinalReportTests(unittest.TestCase):
    def _fixture(
        self,
        root: Path,
        *,
        validation_split_reads: int = 0,
        gpu_name_start: str = "NVIDIA A800-SXM4-80GB",
        gpu_name_end: str | None = None,
        telemetry_sample_counts: list[int] | None = None,
        matched_a3_checkpoint_sha256: str | None = None,
        stage2_rank_data_overrides: dict[str, object] | None = None,
        stage2_support_overrides: dict[str, object] | None = None,
    ) -> Path:
        run_id = "navigation-rlt-test"
        result_root = root / "obs" / "results" / "navigation-rlt" / run_id
        result_root.mkdir(parents=True)
        uuids = [f"GPU-{index:08d}" for index in range(8)]
        records: list[dict[str, object]] = []

        def archived(relative: str, value: object) -> Path:
            path = _write(result_root / Path(relative), value)
            digest = _sha(path)
            records.append(
                {
                    "kind": "test artifact",
                    "source": str(path),
                    "group": "results",
                    "relative_path": relative,
                    "size_bytes": path.stat().st_size,
                    "sha256": digest,
                    "destination": str(path.resolve()),
                    "readback_sha256": digest,
                }
            )
            return path

        replay_values: dict[str, dict[str, object]] = {}
        replay_paths: dict[str, Path] = {}
        phase_values = {
            "stage1": {"episodes": 224, "transitions": 800, "items": 1600, "epochs": 2, "steps": 25},
            "stage2": {"episodes": 256, "transitions": 2000, "items": 1200, "epochs": 3, "steps": 30},
        }
        for phase, parameters in phase_values.items():
            source = ("1" if phase == "stage1" else "2") * 64
            representation = "prefix_hidden" if phase == "stage1" else "frozen_z_rl"
            replay_value = {
                "schema_name": report.REPLAY_SCHEMA,
                "schema_version": 3,
                "run_id": run_id,
                "representation_kind": representation,
                "split_label": "train_update",
                "validation_split_reads": validation_split_reads,
                "expected_episode_count": parameters["episodes"],
                "completed_episode_count": parameters["episodes"],
                "transition_count": parameters["transitions"],
                "source_fingerprint": source,
            }
            replay_path = archived(
                f"{phase}/provenance/replay_manifest.json", replay_value
            )
            replay_values[phase] = replay_value
            replay_paths[phase] = replay_path

            local_items = parameters["items"] // 8
            stage2_support: dict[str, object] | None = None
            if phase == "stage2":
                stage2_support = {
                    "eligibility_schema": report.STAGE2_ELIGIBILITY_SCHEMA,
                    "gate_names": list(report.MODES),
                    "gate_counts": [400, 300, 300, 200],
                    "gate_support_mask": [True, True, True, True],
                    "minimum_behavior_gate_count": 1,
                }
                if stage2_support_overrides is not None:
                    stage2_support.update(stage2_support_overrides)
            rank_receipts = []
            for rank, uuid in enumerate(uuids):
                wall = 100.0 + rank
                rank_data: dict[str, object] = {
                    "global_real_items": parameters["items"],
                    "local_real_items": local_items,
                }
                if phase == "stage2":
                    assert stage2_support is not None
                    rank_data.update(
                        {
                            "global_padding_items": 0,
                            "global_credit_ineligible_items_excluded": 500,
                            "global_gate_mismatch_items_excluded": 300,
                            **stage2_support,
                            "local_sampler_items": local_items,
                            "local_padding_items": 0,
                            "sampler_indices_sha256": f"{rank + 10:064x}",
                            "sampler": "torch.utils.data.DistributedSampler",
                            "sampler_real_row_duplication": 0,
                        }
                    )
                    if stage2_rank_data_overrides is not None:
                        rank_data.update(stage2_rank_data_overrides)
                rank_receipts.append(
                    {
                        "schema_name": "navigation_rlt_rank_receipt",
                        "schema_version": 1,
                        "phase": phase,
                        "rank": rank,
                        "local_rank": rank,
                        "world_size": 8,
                        "gradient_steps": parameters["steps"],
                        "data": rank_data,
                        "gpu_start": {
                            "gpu_uuid": uuid,
                            "gpu_name": gpu_name_start,
                        },
                        "gpu_end": {
                            "gpu_uuid": uuid,
                            "gpu_name": (
                                gpu_name_start if gpu_name_end is None else gpu_name_end
                            ),
                        },
                        "cuda_peak_allocated_bytes": 10_000 + rank,
                        "cuda_peak_reserved_bytes": 20_000 + rank,
                        "training_wall_time_s": wall,
                        "optimizer_steps_per_second": parameters["steps"] / wall,
                        "real_item_epochs_per_second": (
                            local_items * parameters["epochs"] / wall
                        ),
                    }
                )
            action_probabilities = []
            if phase == "stage2":
                action_probabilities = [
                    {
                        "keep_probability_mean": 0.4,
                        "adjust_probability_mean": 0.3,
                        "stop_probability_mean": 0.2,
                        "recover_probability_mean": 0.1,
                    }
                    for _ in range(parameters["epochs"])
                ]
            training_value = {
                "schema_name": report.TRAINING_SCHEMA,
                "schema_version": 1,
                "status": "completed",
                "phase": phase,
                "run_id": run_id,
                "world_size": 8,
                "unique_gpu_uuid_count": 8,
                "rank_receipts": rank_receipts,
                "replay_manifest": {
                    "path": str(replay_path),
                    "sha256": _sha(replay_path),
                    "source_fingerprint": source,
                    "representation_kind": representation,
                    "episode_count": parameters["episodes"],
                    "transition_count": parameters["transitions"],
                    "behavior_policy_versions": [100001],
                },
                "training": {
                    "epochs": parameters["epochs"],
                    "batch_size_per_rank": 16 if phase == "stage1" else 64,
                },
                "optimizer_steps_per_rank": parameters["steps"],
                "action_probability_epochs": action_probabilities,
            }
            if phase == "stage2":
                training_value["stage2_support"] = stage2_support
            archived(f"{phase}/training/training_result.json", training_value)
            telemetry_value = {
                "schema_name": report.TELEMETRY_SCHEMA,
                "schema_version": 1,
                "gpu_count": 8,
                "sample_interval_s": 30.0,
                "gpus": [
                    {
                        "index": index,
                        "uuid": uuid,
                        "sample_count": (
                            5
                            if telemetry_sample_counts is None
                            else telemetry_sample_counts[index]
                        ),
                        "memory_total_mib": 81920.0,
                        "memory_used_mean_mib": 1000.0 + index,
                        "memory_used_max_mib": 2000.0 + index,
                        "gpu_util_mean_percent": 60.0 + index,
                        "gpu_util_max_percent": 90.0 + index,
                        "power_mean_w": 200.0 + index,
                        "power_max_w": 300.0 + index,
                    }
                    for index, uuid in enumerate(uuids)
                ],
            }
            archived(f"{phase}/gpu_telemetry_summary.json", telemetry_value)

        def arm_summary(successes: int, spl: float, adjust: int) -> dict[str, object]:
            decisions = 320
            requested = {
                "KEEP": decisions - adjust - 20 - 10,
                "ADJUST": adjust,
                "STOP": 20,
                "RECOVER": 10,
            }
            executed = dict(requested)
            return {
                "episode_count": 32,
                "success_count": successes,
                "success_rate": successes / 32,
                "mean_spl": spl,
                "mean_final_navigation_error_m": 4.0 - spl,
                "mean_path_length_m": 8.0 + spl,
                "collision_count": 2.0,
                "blocked_step_count": 12,
                "native_stop_count": 12,
                "learned_stop_count": 8,
                "budget_stop_count": 4,
                "no_stop_count": 8,
                "timeout_count": 4,
                "actor_decision_count": decisions,
                "safety_override_count": 1,
                "model_latency": {
                    "count": decisions,
                    "mean_ms": 5.0 + adjust / 100,
                    "p50_ms": 4.0 + adjust / 100,
                    "p95_ms": 8.0 + adjust / 100,
                    "max_ms": 10.0 + adjust / 100,
                },
                "requested_modes": _mode_summary(requested),
                "executed_modes": _mode_summary(executed),
            }

        summaries = {
            "native_lightnav": arm_summary(10, 0.30, 40),
            "a3_ppo": arm_summary(12, 0.35, 50),
            "navigation_rlt": arm_summary(14, 0.40, 60),
        }
        matched_value = {
            "schema_name": report.MATCHED_SCHEMA,
            "schema_version": 1,
            "status": "completed",
            "run_id": run_id,
            "navigation_rlt_checkpoint_sha256": "4" * 64,
            "checkpoints": {
                "native_lightnav": {
                    "kind": "original_lightnav",
                    "sha256": "3" * 64,
                    "policy_version": None,
                },
                "a3_ppo": {
                    "kind": "a3_ppo",
                    "sha256": (
                        report.A3_EXPECTED_PROVENANCE["checkpoint_sha256"]
                        if matched_a3_checkpoint_sha256 is None
                        else matched_a3_checkpoint_sha256
                    ),
                    "policy_version": 0,
                },
                "navigation_rlt": {
                    "kind": "navigation_rlt",
                    "sha256": "4" * 64,
                    "policy_version": 1,
                },
            },
            "matched_contract": {
                "protocol": {
                    "split_label": "train_development",
                    "upstream_split_read": "train",
                    "validation_split_reads": 0,
                    "training_replay_written": False,
                }
            },
            "summaries": summaries,
            "deltas": {
                "rlt_minus_a3": {
                    "success_count": 2.0,
                    "success_rate": 2 / 32,
                    "mean_spl": 0.05,
                }
            },
            "gate": {"passed": True},
            "result_to_claim_boundary": {"verdict": "partial"},
        }
        matched_path = archived("matched/matched_result.json", matched_value)
        readback_sha = "a" * 64
        manifest = {
            "schema_name": report.ARCHIVE_SCHEMA,
            "schema_version": 1,
            "status": "readback_verified",
            "run_id": run_id,
            "destinations": {
                "checkpoints": str(root / "obs" / "checkpoints" / run_id),
                "results": str(result_root.resolve()),
            },
            "replay": {
                phase: {
                    "representation_kind": replay_values[phase]["representation_kind"],
                    "source_fingerprint": replay_values[phase]["source_fingerprint"],
                    "episode_count": replay_values[phase]["completed_episode_count"],
                    "transition_count": replay_values[phase]["transition_count"],
                }
                for phase in report.PHASES
            },
            "matched_result": {"sha256": _sha(matched_path)},
            "files": records,
            "independent_readback_sha256": readback_sha,
        }
        manifest_path = _write(result_root / "archive_manifest.json", manifest)
        _write(
            result_root / "COMPLETED.json",
            {
                "schema_name": report.ARCHIVE_COMPLETION_SCHEMA,
                "schema_version": 1,
                "status": "completed",
                "run_id": run_id,
                "archive_manifest": str(manifest_path.resolve()),
                "archive_manifest_sha256": _sha(manifest_path),
                "independent_readback_sha256": readback_sha,
                "checkpoint_root": str(root / "obs" / "checkpoints" / run_id),
                "result_root": str(result_root.resolve()),
            },
        )
        return result_root

    def test_build_and_create_only_write_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result_root = self._fixture(root)
            receipt = _a3_budget_receipt(root)
            value = report.build_report(
                result_root, a3_training_budget_receipt=receipt
            )
            self.assertTrue(value["scope"]["train_only"])
            self.assertEqual(value["scope"]["validation_split_reads"], 0)
            self.assertEqual(
                value["training"]["stage2"]["budget"]["transition_count"], 2000
            )
            self.assertEqual(
                value["fixed_budget_normalized_effect_proxy"][
                    "raw_rlt_replay_transition_count"
                ],
                2000,
            )
            proxy = value["fixed_budget_normalized_effect_proxy"]
            self.assertEqual(
                proxy["denominator_scope"], "method_specific_unique_actor_decisions"
            )
            self.assertTrue(proxy["a3_training_budget_available"])
            self.assertEqual(
                proxy["training_budgets"]["a3_ppo"]["unique_actor_decisions"],
                4642,
            )
            self.assertEqual(
                proxy["training_budgets"]["navigation_rlt"][
                    "unique_actor_decisions"
                ],
                1200,
            )
            rlt_budget = proxy["training_budgets"]["navigation_rlt"]
            self.assertEqual(
                rlt_budget["eligibility_schema"], report.STAGE2_ELIGIBILITY_SCHEMA
            )
            self.assertEqual(rlt_budget["credit_ineligible_items_excluded"], 500)
            self.assertEqual(rlt_budget["gate_mismatch_items_excluded"], 300)
            self.assertEqual(rlt_budget["sampler_real_row_duplication"], 0)
            self.assertTrue(rlt_budget["unique_real_rows_verified"])
            self.assertTrue(rlt_budget["padding_excluded"])
            self.assertAlmostEqual(
                proxy["one_point_vs_native"]["a3_ppo"][
                    "per_1000_unique_actor_decisions"
                ]["success_count"],
                2 * 1000 / 4642,
            )
            self.assertAlmostEqual(
                proxy["one_point_vs_native"]["navigation_rlt"][
                    "per_1000_unique_actor_decisions"
                ]["success_count"],
                4 * 1000 / 1200,
            )
            self.assertEqual(proxy["matched_evaluation_episode_count"], 32)
            self.assertEqual(proxy["matched_evaluation_seed_count"], 1)
            self.assertTrue(proxy["not_a_sample_efficiency_estimate"])
            gpu = value["training"]["stage1"]["gpu"]
            self.assertTrue(gpu["a800_identity_verified"])
            self.assertEqual(
                gpu["gpu_model_name_normalized"], "NVIDIA A800-SXM4-80GB"
            )
            self.assertEqual(gpu["sample_interval_s"], 30.0)
            self.assertEqual(gpu["sample_count_per_gpu"], 5)
            self.assertEqual(gpu["sampled_coverage_lower_bound_s"], 120.0)
            self.assertAlmostEqual(
                value["training"]["stage1"]["gpu"][
                    "gpu_util_mean_percent_across_gpus"
                ],
                63.5,
            )
            output_json = root / "final.json"
            output_markdown = root / "final.md"
            output_completion = root / "final.COMPLETED.json"
            publication = report.write_report_bundle(
                value, output_json, output_markdown, output_completion
            )
            self.assertEqual(
                publication["publication"],
                {"json": "created", "markdown": "created", "completion": "created"},
            )
            self.assertEqual(
                json.loads(output_json.read_text(encoding="utf-8"))["schema_name"],
                report.REPORT_SCHEMA,
            )
            written = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(
                written["matched"]["summaries"]["navigation_rlt"]["model_latency"],
                {
                    "count": 320,
                    "mean_ms": 5.6,
                    "p50_ms": 4.6,
                    "p95_ms": 8.6,
                    "max_ms": 10.6,
                },
            )
            markdown = output_markdown.read_text(encoding="utf-8")
            self.assertIn("not a sample-efficiency curve or estimate", markdown)
            self.assertIn("No direct NCCL", markdown)
            self.assertIn("success count / 32", markdown)
            self.assertIn("| navigation_rlt | 14 / 32 |", markdown)
            self.assertIn("latency mean ms", markdown)
            self.assertIn("latency p95 ms", markdown)
            self.assertIn("| 5.6 | 8.6 |", markdown)
            self.assertIn("## Sampled 8×A800 utilization", markdown)
            self.assertIn("| stage1 | NVIDIA A800-SXM4-80GB | 30 | 5 |", markdown)
            self.assertIn("fixed-budget normalized effect proxy", markdown)
            self.assertIn("unique actor-eligible training decisions", markdown)
            self.assertIn("raw replay contains **2000** transitions", markdown)
            self.assertIn("validated A3 receipt contains **4642**", markdown)
            self.assertIn("32 episodes from one deterministic seed", markdown)
            completion = json.loads(output_completion.read_text(encoding="utf-8"))
            self.assertEqual(
                completion["schema_name"], report.REPORT_COMPLETION_SCHEMA
            )
            self.assertEqual(completion["run_id"], value["run_id"])
            self.assertEqual(completion["validation_split_reads"], 0)
            for key, path in (
                ("json", output_json),
                ("markdown", output_markdown),
            ):
                self.assertEqual(completion["outputs"][key]["path"], str(path.absolute()))
                self.assertEqual(completion["outputs"][key]["sha256"], _sha(path))
                self.assertEqual(
                    completion["outputs"][key]["size_bytes"], path.stat().st_size
                )
            before = (
                output_json.read_bytes(),
                output_markdown.read_bytes(),
                output_completion.read_bytes(),
            )
            repeated = report.write_report_bundle(
                value, output_json, output_markdown, output_completion
            )
            self.assertEqual(
                repeated["publication"],
                {"json": "reused", "markdown": "reused", "completion": "reused"},
            )
            self.assertEqual(
                before,
                (
                    output_json.read_bytes(),
                    output_markdown.read_bytes(),
                    output_completion.read_bytes(),
                ),
            )

    def test_authoritative_write_requires_a3_budget_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result_root = self._fixture(root)
            value = report.build_report(result_root)
            outputs = (
                root / "final.json",
                root / "final.md",
                root / "final.COMPLETED.json",
            )
            with self.assertRaisesRegex(
                report.FinalReportError,
                "A3 training-budget receipt is required",
            ):
                report.write_report_bundle(value, *outputs)
            self.assertTrue(all(not path.exists() for path in outputs))

    def test_valid_a3_budget_receipt_enables_method_specific_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result_root = self._fixture(root)
            receipt = _a3_budget_receipt(root)
            value = report.build_report(
                result_root, a3_training_budget_receipt=receipt
            )
            proxy = value["fixed_budget_normalized_effect_proxy"]
            self.assertTrue(proxy["a3_training_budget_available"])
            a3_budget = proxy["training_budgets"]["a3_ppo"]
            self.assertEqual(a3_budget["unique_actor_decisions"], 4642)
            self.assertEqual(a3_budget["optimizer_presentations"], 18568)
            self.assertTrue(a3_budget["padding_excluded"])
            self.assertEqual(
                value["matched"]["checkpoints"]["a3_ppo"]["sha256"],
                a3_budget["provenance"]["checkpoint_sha256"],
            )
            self.assertEqual(
                value["inputs"]["a3_training_budget_receipt"]["sha256"],
                _sha(receipt),
            )
            self.assertAlmostEqual(
                proxy["one_point_vs_native"]["a3_ppo"][
                    "per_1000_unique_actor_decisions"
                ]["success_count"],
                2 * 1000 / 4642,
            )
            self.assertEqual(proxy["raw_rlt_replay_transition_count"], 2000)
            markdown = report.render_markdown(value)
            self.assertIn("validated A3 receipt contains **4642**", markdown)
            self.assertIn("**18568** non-padding optimizer presentations", markdown)
            self.assertIn("one evaluation seed", markdown)

    def test_rejects_a3_budget_checkpoint_not_used_by_matched_arm(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result_root = self._fixture(
                root, matched_a3_checkpoint_sha256="0" * 64
            )
            receipt = _a3_budget_receipt(root)
            with self.assertRaisesRegex(
                report.FinalReportError,
                "checkpoint does not match matched A3 arm",
            ):
                report.build_report(
                    result_root, a3_training_budget_receipt=receipt
                )

    def test_rejects_tampered_stage2_unique_decision_evidence(self) -> None:
        cases = (
            (
                "eligibility schema",
                {"eligibility_schema": "tampered"},
                None,
            ),
            (
                "real-row duplication",
                {"sampler_real_row_duplication": 1},
                None,
            ),
            (
                "eligibility exclusions",
                {"global_credit_ineligible_items_excluded": 499},
                None,
            ),
            (
                "padding coverage",
                {"local_padding_items": 1},
                None,
            ),
            (
                "support summary",
                None,
                {"gate_counts": [399, 300, 300, 200]},
            ),
        )
        for name, rank_overrides, support_overrides in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                result_root = self._fixture(
                    root,
                    stage2_rank_data_overrides=rank_overrides,
                    stage2_support_overrides=support_overrides,
                )
                with self.assertRaises(report.FinalReportError):
                    report.build_report(result_root)

    def test_rejects_a3_budget_receipt_with_changed_eligibility_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result_root = self._fixture(root)
            receipt = _a3_budget_receipt(root)
            value = json.loads(receipt.read_text(encoding="utf-8"))
            value["budget"]["actor_eligibility"]["native_stop_excluded"] = False
            _write(receipt, value)
            with self.assertRaisesRegex(
                report.FinalReportError, "actor eligibility semantics differ"
            ):
                report.build_report(
                    result_root, a3_training_budget_receipt=receipt
                )

    def test_rejects_changed_a3_budget_hash_presentations_or_padding(self) -> None:
        cases = (
            (
                ("provenance", "replay_sha256"),
                "0" * 64,
                "training-budget provenance differs",
            ),
            (
                ("budget", "optimizer_presentations"),
                18569,
                "training budget differs",
            ),
            (
                ("budget", "padding_excluded"),
                False,
                "training budget differs",
            ),
        )
        for keys, replacement, message in cases:
            with self.subTest(field=".".join(keys)):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    result_root = self._fixture(root)
                    receipt = _a3_budget_receipt(root)
                    value = json.loads(receipt.read_text(encoding="utf-8"))
                    value[keys[0]][keys[1]] = replacement
                    _write(receipt, value)
                    with self.assertRaisesRegex(report.FinalReportError, message):
                        report.build_report(
                            result_root, a3_training_budget_receipt=receipt
                        )

    def test_rejects_tampered_archived_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result_root = self._fixture(Path(temporary))
            training = result_root / "stage2" / "training" / "training_result.json"
            training.write_bytes(training.read_bytes() + b" ")
            with self.assertRaisesRegex(report.FinalReportError, "current readback differs"):
                report.build_report(result_root)

    def test_rejects_validation_split_read_even_when_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result_root = self._fixture(
                Path(temporary), validation_split_reads=1
            )
            with self.assertRaisesRegex(report.FinalReportError, "train-only contract"):
                report.build_report(result_root)

    def test_rejects_non_a800_gpu_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result_root = self._fixture(
                Path(temporary), gpu_name_start="NVIDIA A100-SXM4-80GB"
            )
            with self.assertRaisesRegex(report.FinalReportError, "not an NVIDIA A800"):
                report.build_report(result_root)

    def test_rejects_gpu_model_drift_between_start_and_end(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result_root = self._fixture(
                Path(temporary),
                gpu_name_end="NVIDIA A800 80GB PCIe",
            )
            with self.assertRaisesRegex(report.FinalReportError, "GPU model differs"):
                report.build_report(result_root)

    def test_rejects_unequal_telemetry_sample_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result_root = self._fixture(
                Path(temporary), telemetry_sample_counts=[5, 5, 5, 5, 5, 5, 5, 4]
            )
            with self.assertRaisesRegex(report.FinalReportError, "sample counts differ"):
                report.build_report(result_root)

    def test_preexisting_markdown_prevents_partial_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result_root = self._fixture(root)
            value = report.build_report(
                result_root, a3_training_budget_receipt=_a3_budget_receipt(root)
            )
            output_json = root / "final.json"
            output_markdown = root / "final.md"
            output_completion = root / "final.COMPLETED.json"
            output_markdown.write_text("sentinel", encoding="utf-8")
            with self.assertRaisesRegex(report.FinalReportError, "bytes differ"):
                report.write_report_bundle(
                    value, output_json, output_markdown, output_completion
                )
            self.assertFalse(output_json.exists())
            self.assertEqual(output_markdown.read_text(encoding="utf-8"), "sentinel")

    def test_reentry_after_failure_after_json_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result_root = self._fixture(root)
            value = report.build_report(
                result_root, a3_training_budget_receipt=_a3_budget_receipt(root)
            )
            output_json = root / "final.json"
            output_markdown = root / "final.md"
            output_completion = root / "final.COMPLETED.json"
            original = report._publish_or_reuse
            calls = 0

            def fail_second(path: Path, payload: bytes, name: str) -> str:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("fault after JSON publication")
                return original(path, payload, name)

            with mock.patch.object(report, "_publish_or_reuse", side_effect=fail_second):
                with self.assertRaisesRegex(RuntimeError, "after JSON"):
                    report.write_report_bundle(
                        value, output_json, output_markdown, output_completion
                    )
            self.assertTrue(output_json.is_file())
            self.assertFalse(output_markdown.exists())
            self.assertFalse(output_completion.exists())

            publication = report.write_report_bundle(
                value, output_json, output_markdown, output_completion
            )
            self.assertEqual(
                publication["publication"],
                {"json": "reused", "markdown": "created", "completion": "created"},
            )

    def test_completion_is_published_last_and_reentry_completes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result_root = self._fixture(root)
            value = report.build_report(
                result_root, a3_training_budget_receipt=_a3_budget_receipt(root)
            )
            output_json = root / "final.json"
            output_markdown = root / "final.md"
            output_completion = root / "final.COMPLETED.json"
            original = report._publish_or_reuse
            calls = 0

            def fail_third(path: Path, payload: bytes, name: str) -> str:
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise RuntimeError("fault before completion marker")
                return original(path, payload, name)

            with mock.patch.object(report, "_publish_or_reuse", side_effect=fail_third):
                with self.assertRaisesRegex(RuntimeError, "before completion"):
                    report.write_report_bundle(
                        value, output_json, output_markdown, output_completion
                    )
            self.assertTrue(output_json.is_file())
            self.assertTrue(output_markdown.is_file())
            self.assertFalse(output_completion.exists())

            publication = report.write_report_bundle(
                value, output_json, output_markdown, output_completion
            )
            self.assertEqual(
                publication["publication"],
                {"json": "reused", "markdown": "reused", "completion": "created"},
            )


if __name__ == "__main__":
    unittest.main()
