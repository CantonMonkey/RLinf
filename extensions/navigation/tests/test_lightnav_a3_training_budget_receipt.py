from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


RLINF_NAV_ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = RLINF_NAV_ROOT / "integrations" / "rlinf_v03"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import lightnav_a3_training_budget_receipt as receipt_tool


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class _ArchiveFixture:
    def __init__(
        self,
        root: Path,
        *,
        invalid_actor: bool = False,
        invalid_rank_padding: bool = False,
        rollout_actor_mode: str = "exact",
    ) -> None:
        self.root = root
        self.obs_root = root / "obs"
        self.obs_root.mkdir()
        self.run_id = "lightnav-a3-test"
        self.source_root = root / "source" / self.run_id
        self.records: list[dict[str, object]] = []
        self.by_relative: dict[str, tuple[bytes, str, int]] = {}
        self.episode_count = 2
        self.evaluation_episode_count = 1
        self.decision_count = 8
        self.epochs = 2
        self.world_size = 8
        self.batch_size = 1
        self.steps_per_rank = 2
        if rollout_actor_mode not in {
            "exact",
            "canonical",
            "tampered_sha",
            "tampered_size",
            "ambiguous",
            "unsafe",
            "unexpected",
            "diagnostic_bad_hash",
            "diagnostic_bad_size",
        }:
            raise ValueError("unsupported rollout actor mode")

        trainer = b"# fixed trainer semantics\n"
        trainer_sha = self.add("code/lightnav_a3_fsdp_train.py", trainer)[0]

        split = {
            "schema_name": "lightnav_a3_train_scene_split",
            "schema_version": 1,
            "upstream_split_read": "train",
            "validation_split_reads": 0,
            "splits": {
                "train_update": {
                    "habitat_split": "train",
                    "episode_count": self.episode_count,
                },
                "train_development": {
                    "habitat_split": "train",
                    "episode_count": self.evaluation_episode_count,
                },
            },
        }
        split_bytes = _json_bytes(split)
        split_sha = self.add("ebk_archive/data/split_manifest.json", split_bytes)[0]

        run_manifest = {
            "schema_name": "lightnav_a3_run_manifest",
            "schema_version": 1,
            "run_id": self.run_id,
            "upstream_split": "train",
            "validation_split_reads": 0,
            "split_manifest": {"sha256": split_sha},
            "policy": {"native_stop_actor_credit": False},
            "ppo": {
                "epochs": self.epochs,
                "batch_size_per_rank": self.batch_size,
                "required_world_size": self.world_size,
            },
            "code": {
                "lightnav_a3_fsdp_train.py": {"sha256": trainer_sha},
            },
        }
        run_bytes = _json_bytes(run_manifest)
        run_sha = self.add("input/run_manifest.json", run_bytes)[0]

        actor_rows = []
        for index in range(self.decision_count):
            actor_rows.append(
                {
                    "episode_order": index // 4,
                    "episode_id": f"episode-{index // 4}",
                    "scene_id": f"scene-{index // 4}",
                    "step": index % 4,
                    "actor_eligible": False if invalid_actor and index == 0 else True,
                    "policy_version": 0,
                    "native_stop": False,
                    "external_forced_stop": False,
                }
            )
        replay_bytes = b"".join(
            (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode(
                "utf-8"
            )
            for row in actor_rows
        )
        replay_sha = self.add("input/actor_decisions.jsonl", replay_bytes)[0]
        if rollout_actor_mode == "exact":
            rollout_parent = "ebk_archive/train_rollout"
            self.add(f"{rollout_parent}/actor_decisions.jsonl", replay_bytes)
        else:
            rollout_parent = "input/train_rollout_receipt"
            if rollout_actor_mode == "ambiguous":
                self.add(
                    f"{rollout_parent}/actor_decisions.jsonl",
                    b"conflicting replay artifact\n",
                )

        episodes = [
            {"episode_order": index, "actor_decision_count": 4}
            for index in range(self.episode_count)
        ]
        rollout_result = {
            "schema_name": "lightnav_a3_rollout",
            "schema_version": 1,
            "status": "completed",
            "mode": "train",
            "deterministic": False,
            "upstream_split": "train",
            "experiment_split_label": "train_update",
            "validation_split_reads": 0,
            "run_manifest": {"sha256": run_sha, "run_id": self.run_id},
            "split_manifest": {"sha256": split_sha},
            "episodes_requested": self.episode_count,
            "checkpoint": {"policy_version": 0},
            "episodes": episodes,
            "aggregate": {"actor_decision_count": self.decision_count},
            "actor_decisions": {
                "sha256": replay_sha,
                "row_count": self.decision_count,
            },
        }
        result_bytes = _json_bytes(rollout_result)
        result_sha, result_size = self.add(
            f"{rollout_parent}/result.json", result_bytes
        )
        inventory_actor_relative = (
            "../actor_decisions.jsonl"
            if rollout_actor_mode == "unsafe"
            else "actor_decisions.jsonl"
        )
        inventory_actor_sha = (
            "0" * 64 if rollout_actor_mode == "tampered_sha" else replay_sha
        )
        inventory_actor_size = len(replay_bytes) + (
            1 if rollout_actor_mode == "tampered_size" else 0
        )
        rollout_inventory = {
            inventory_actor_relative: {
                "sha256": inventory_actor_sha,
                "size_bytes": inventory_actor_size,
            },
            "result.json": {"sha256": result_sha, "size_bytes": result_size},
        }
        if rollout_actor_mode != "exact":
            progress_hash = _sha(b"episode progress diagnostic")
            progress_size = len(b"episode progress diagnostic")
            if rollout_actor_mode == "diagnostic_bad_hash":
                progress_hash = "not-a-sha256"
            if rollout_actor_mode == "diagnostic_bad_size":
                progress_size = -1
            rollout_inventory.update(
                {
                    "episode_progress.jsonl": {
                        "sha256": progress_hash,
                        "size_bytes": progress_size,
                    },
                    "execution_trace.jsonl": {
                        "sha256": _sha(b"execution trace diagnostic"),
                        "size_bytes": len(b"execution trace diagnostic"),
                    },
                }
            )
            if rollout_actor_mode == "unexpected":
                rollout_inventory["unexpected.jsonl"] = {
                    "sha256": _sha(b"unexpected"),
                    "size_bytes": len(b"unexpected"),
                }
        rollout_inventory_bytes = _json_bytes(rollout_inventory)
        rollout_inventory_sha = self.add(
            f"{rollout_parent}/artifact_inventory.json", rollout_inventory_bytes
        )[0]
        self.add(
            f"{rollout_parent}/COMPLETED.json",
            _json_bytes(
                {
                    "schema_name": "lightnav_a3_completion",
                    "schema_version": 1,
                    "status": "completed",
                    "result_sha256": result_sha,
                    "inventory_sha256": rollout_inventory_sha,
                }
            ),
        )

        checkpoint_bytes = b"portable-a3-checkpoint"
        checkpoint_sha, checkpoint_size = self.add(
            "training/checkpoints/latest/lightnav_a3_policy.npz", checkpoint_bytes
        )
        rank_receipts = []
        padding_rows = 0
        sampler_rows = 1
        for rank in range(self.world_size):
            rank_receipt = {
                "schema_name": "lightnav_a3_rank_receipt",
                "schema_version": 1,
                "rank": rank,
                "local_rank": rank,
                "world_size": self.world_size,
                "pid": 1000 + rank,
                "gpu_uuid": f"GPU-{rank:08d}",
                "gpu_name": "NVIDIA A800-SXM4-80GB",
                "collective_sum": 28.0,
                "expected_collective_sum": 28.0,
                "gradient_steps": self.steps_per_rank,
                "dataset_rows": self.decision_count,
                "padding_rows": (
                    1 if invalid_rank_padding and rank == 0 else padding_rows
                ),
                "sampler_rows": sampler_rows,
                "completed_unix_s": 1.0,
            }
            rank_receipts.append(rank_receipt)
            self.add(
                f"training/ranks/rank_{rank:03d}.json",
                _json_bytes(rank_receipt),
            )
        training_result = {
            "schema_name": "lightnav_a3_rlinf_fsdp_training",
            "schema_version": 1,
            "status": "completed",
            "rlinf_backend": "FSDPModelManager",
            "fsdp_wrapper": "FullyShardedDataParallel",
            "sharding_strategy": "full_shard",
            "world_size": self.world_size,
            "unique_gpu_uuid_count": self.world_size,
            "rank_receipts": rank_receipts,
            "rollout": {
                "sha256": replay_sha,
                "actor_decision_count": self.decision_count,
                "padding_row_count": padding_rows,
            },
            "run_manifest": {"sha256": run_sha, "run_id": self.run_id},
            "final_checkpoint": {
                "sha256": checkpoint_sha,
                "size_bytes": checkpoint_size,
                "metadata": {"policy_version": 1},
            },
            "epochs": self.epochs,
            "batch_size_per_rank": self.batch_size,
            "optimizer_steps_per_rank": self.steps_per_rank,
        }
        training_result_bytes = _json_bytes(training_result)
        training_result_sha, training_result_size = self.add(
            "training/training_result.json", training_result_bytes
        )
        training_inventory: dict[str, dict[str, object]] = {}
        for relative, (_, item_hash, item_size) in sorted(self.by_relative.items()):
            if not relative.startswith("training/"):
                continue
            name = relative.removeprefix("training/")
            training_inventory[name] = {
                "sha256": item_hash,
                "size_bytes": item_size,
            }
        self.assert_training_result_inventory(
            training_inventory, training_result_sha, training_result_size
        )
        training_inventory_bytes = _json_bytes(training_inventory)
        training_inventory_sha = self.add(
            "training/artifact_inventory.json", training_inventory_bytes
        )[0]
        self.add(
            "training/COMPLETED.json",
            _json_bytes(
                {
                    "schema_name": "lightnav_a3_training_completion",
                    "schema_version": 1,
                    "status": "completed",
                    "training_result_sha256": training_result_sha,
                    "artifact_inventory_sha256": training_inventory_sha,
                }
            ),
        )

        source_total = sum(int(record["size_bytes"]) for record in self.records)
        manifest = {
            "schema_name": "lightnav_a3_obs_archive",
            "schema_version": 1,
            "status": "completed",
            "phase": "final",
            "run_id": self.run_id,
            "source_run_root": str(self.source_root),
            "source_total_bytes": source_total,
            "checkpoint_hard_cap_bytes": 2 * 1024**3,
            "run_hard_cap_bytes": 16 * 1024**3,
            "files": self.records,
            "completed_unix_s": 2.0,
        }
        self.manifest_path = (
            self.obs_root / "manifests" / "lightnav-a3" / f"{self.run_id}.final.json"
        )
        self.manifest_path.parent.mkdir(parents=True)
        self.manifest_path.write_bytes(_json_bytes(manifest))
        final_manifest_sha = receipt_tool.sha256_file(self.manifest_path)
        self.completion_path = (
            self.obs_root
            / "results"
            / "lightnav-a3"
            / self.run_id
            / "COMPLETE"
        )
        self.completion_path.parent.mkdir(parents=True, exist_ok=True)
        self.completion_path.write_bytes(
            _json_bytes(
                {
                    "schema_name": "lightnav_a3_obs_completion",
                    "schema_version": 1,
                    "phase": "final",
                    "manifest": str(self.manifest_path),
                    "manifest_sha256": final_manifest_sha,
                    "completed_unix_s": 3.0,
                }
            )
        )
        self.contract = receipt_tool.A3BudgetContract(
            run_id=self.run_id,
            source_run_root=str(self.source_root),
            final_manifest_sha256=final_manifest_sha,
            completion_sha256=receipt_tool.sha256_file(self.completion_path),
            run_manifest_sha256=run_sha,
            split_manifest_sha256=split_sha,
            replay_sha256=replay_sha,
            checkpoint_sha256=checkpoint_sha,
            manifest_file_count=len(self.records),
            source_total_bytes=source_total,
            rollout_episode_count=self.episode_count,
            evaluation_episode_count=self.evaluation_episode_count,
            unique_actor_decisions=self.decision_count,
            optimizer_epochs=self.epochs,
            world_size=self.world_size,
            batch_size_per_rank=self.batch_size,
            optimizer_steps_per_rank=self.steps_per_rank,
            checkpoint_size_bytes=checkpoint_size,
        )

    @staticmethod
    def assert_training_result_inventory(
        inventory: dict[str, dict[str, object]],
        result_sha: str,
        result_size: int,
    ) -> None:
        assert inventory["training_result.json"] == {
            "sha256": result_sha,
            "size_bytes": result_size,
        }

    def add(self, relative: str, payload: bytes) -> tuple[str, int]:
        destination = (
            self.obs_root / "results" / "lightnav-a3" / self.run_id / relative
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        item_hash = _sha(payload)
        item_size = len(payload)
        source = self.source_root / Path(relative)
        self.records.append(
            {
                "source": str(source),
                "destination": str(destination),
                "size_bytes": item_size,
                "sha256": item_hash,
                "readback_sha256": item_hash,
            }
        )
        self.by_relative[relative] = (payload, item_hash, item_size)
        return item_hash, item_size


class LightNavA3TrainingBudgetReceiptTests(unittest.TestCase):
    def test_verified_receipt_is_final_report_compatible_and_create_only(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _ArchiveFixture(root)
            coverage: dict[str, object] = {}
            receipt = receipt_tool.build_receipt(
                fixture.manifest_path,
                fixture.completion_path,
                fixture.obs_root,
                fixture.contract,
                coverage_out=coverage,
            )
            self.assertEqual("verified", receipt["status"])
            self.assertEqual(
                {
                    "schema_name",
                    "schema_version",
                    "status",
                    "run_id",
                    "scope",
                    "provenance",
                    "budget",
                },
                set(receipt),
            )
            self.assertTrue(coverage["full_rollout_inventory_readback"])
            self.assertEqual(
                self.decision_presentations(fixture),
                receipt["budget"]["optimizer_presentations"],
            )
            self.assertTrue(receipt["budget"]["padding_excluded"])
            output = root / receipt_tool.RECEIPT_FILENAME
            expected_sha = receipt_tool.write_receipt_create_only(
                output, receipt, fixture.obs_root
            )
            self.assertEqual(expected_sha, receipt_tool.sha256_file(output))
            with self.assertRaisesRegex(
                receipt_tool.A3BudgetReceiptError, "already exists"
            ):
                receipt_tool.write_receipt_create_only(
                    output, receipt, fixture.obs_root
                )

    def test_canonical_actor_relocation_is_verified(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            fixture = _ArchiveFixture(
                Path(temporary_directory), rollout_actor_mode="canonical"
            )
            coverage: dict[str, object] = {}
            receipt = receipt_tool.build_receipt(
                fixture.manifest_path,
                fixture.completion_path,
                fixture.obs_root,
                fixture.contract,
                coverage_out=coverage,
            )
            self.assertEqual("verified", receipt["status"])
            self.assertFalse(coverage["full_rollout_inventory_readback"])
            self.assertEqual(
                {"actor_decisions.jsonl", "result.json"},
                set(coverage["budget_required_entries"]),
            )
            diagnostics = coverage["unarchived_diagnostic_entries"]
            self.assertEqual(
                {"episode_progress.jsonl", "execution_trace.jsonl"},
                set(diagnostics),
            )
            self.assertTrue(
                all(not value["payload_readback"] for value in diagnostics.values())
            )

    def test_cli_stdout_discloses_coverage_outside_the_budget_receipt(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _ArchiveFixture(root, rollout_actor_mode="canonical")
            output = root / receipt_tool.RECEIPT_FILENAME
            stdout = io.StringIO()
            with mock.patch.object(
                receipt_tool, "PRODUCTION_CONTRACT", fixture.contract
            ), redirect_stdout(stdout):
                exit_code = receipt_tool.main(
                    [
                        "--obs-root",
                        str(fixture.obs_root),
                        "--final-manifest",
                        str(fixture.manifest_path),
                        "--completion",
                        str(fixture.completion_path),
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(0, exit_code)
            emitted = json.loads(stdout.getvalue())
            self.assertFalse(
                emitted["coverage"]["full_rollout_inventory_readback"]
            )
            stored = json.loads(output.read_text(encoding="utf-8"))
            self.assertNotIn("coverage", stored)

    def test_canonical_actor_relocation_rejects_tampered_hash_or_size(self):
        for mode in ("tampered_sha", "tampered_size"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                fixture = _ArchiveFixture(Path(directory), rollout_actor_mode=mode)
                with self.assertRaisesRegex(
                    receipt_tool.A3BudgetReceiptError,
                    "artifact readback differs",
                ):
                    receipt_tool.build_receipt(
                        fixture.manifest_path,
                        fixture.completion_path,
                        fixture.obs_root,
                        fixture.contract,
                    )

    def test_canonical_actor_relocation_rejects_ambiguous_exact_mapping(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            fixture = _ArchiveFixture(
                Path(temporary_directory), rollout_actor_mode="ambiguous"
            )
            with self.assertRaisesRegex(
                receipt_tool.A3BudgetReceiptError,
                "mapping is ambiguous",
            ):
                receipt_tool.build_receipt(
                    fixture.manifest_path,
                    fixture.completion_path,
                    fixture.obs_root,
                    fixture.contract,
                )

    def test_canonical_actor_relocation_rejects_unsafe_inventory_path(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            fixture = _ArchiveFixture(
                Path(temporary_directory), rollout_actor_mode="unsafe"
            )
            with self.assertRaisesRegex(
                receipt_tool.A3BudgetReceiptError,
                "file set differs",
            ):
                receipt_tool.build_receipt(
                    fixture.manifest_path,
                    fixture.completion_path,
                    fixture.obs_root,
                    fixture.contract,
                )

    def test_production_rollout_inventory_rejects_unexpected_key(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            fixture = _ArchiveFixture(
                Path(temporary_directory), rollout_actor_mode="unexpected"
            )
            with self.assertRaisesRegex(
                receipt_tool.A3BudgetReceiptError,
                "file set differs",
            ):
                receipt_tool.build_receipt(
                    fixture.manifest_path,
                    fixture.completion_path,
                    fixture.obs_root,
                    fixture.contract,
                )

    def test_production_rollout_inventory_rejects_invalid_diagnostic_metadata(self):
        for mode in ("diagnostic_bad_hash", "diagnostic_bad_size"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                fixture = _ArchiveFixture(Path(directory), rollout_actor_mode=mode)
                with self.assertRaises(receipt_tool.A3BudgetReceiptError):
                    receipt_tool.build_receipt(
                        fixture.manifest_path,
                        fixture.completion_path,
                        fixture.obs_root,
                        fixture.contract,
                    )

    def test_actor_eligibility_is_recomputed_from_every_replay_row(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _ArchiveFixture(root, invalid_actor=True)
            with self.assertRaisesRegex(
                receipt_tool.A3BudgetReceiptError, "eligibility semantics"
            ):
                receipt_tool.build_receipt(
                    fixture.manifest_path,
                    fixture.completion_path,
                    fixture.obs_root,
                    fixture.contract,
                )

    def test_rank_padding_tamper_cannot_claim_nonpadding_presentations(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _ArchiveFixture(root, invalid_rank_padding=True)
            with self.assertRaisesRegex(
                receipt_tool.A3BudgetReceiptError, "rank evidence"
            ):
                receipt_tool.build_receipt(
                    fixture.manifest_path,
                    fixture.completion_path,
                    fixture.obs_root,
                    fixture.contract,
                )

    @staticmethod
    def decision_presentations(fixture: _ArchiveFixture) -> int:
        return fixture.decision_count * fixture.epochs


if __name__ == "__main__":
    unittest.main()
