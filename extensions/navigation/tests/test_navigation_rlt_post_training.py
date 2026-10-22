from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile


MODULE_DIR = Path(__file__).resolve().parents[1] / "integrations" / "rlinf_v03"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import navigation_rlt_post_training as post  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _metadata_npz(path: Path, value: object) -> Path:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    header = repr(
        {"descr": "|u1", "fortran_order": False, "shape": (len(payload),)}
    ).encode("latin1")
    padding = (16 - ((10 + len(header) + 1) % 16)) % 16
    header = header + b" " * padding + b"\n"
    npy = b"\x93NUMPY" + bytes((1, 0)) + struct.pack("<H", len(header)) + header + payload
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("metadata_json.npy", npy)
    return path


def _returned_arm_plan(root: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    episodes = [
        {
            "episode_order": index,
            "scene": f"scene-{index // 4}",
            "episode_id": str(index),
            "instruction_text_sha256": f"{index + 1:064x}",
        }
        for index in range(32)
    ]
    return (
        {
            "run_id": "navigation-rlt-test",
            "outputs": {
                key: str(path) for key, path in post._fixed_outputs(root).items()
            },
            "remote": {"host": "ebk-test"},
            "collection_manifest": {"sha256": "a" * 64},
            "split_manifest": {"sha256": "b" * 64},
            "evaluation_plan": {
                "sha256": "c" * 64,
                "ordered_episode_identity_sha256": post._object_sha256(
                    [
                        [
                            row["episode_order"],
                            row["scene"],
                            row["episode_id"],
                            row["instruction_text_sha256"],
                        ]
                        for row in episodes
                    ]
                ),
            },
            "evaluation_code": {"inventory_sha256": "d" * 64},
            "contract": {
                "a3_checkpoint_sha256": "e" * 64,
                "arm_configs": {
                    arm: {"sha256": f"{index + 1:064x}"}
                    for index, arm in enumerate(post.ARMS)
                },
            },
            "stage2": {
                "checkpoint": {"sha256": "f" * 64},
                "context_source_sha256": "1" * 64,
            },
        },
        episodes,
    )


def _relay_plan(root: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    plan, episodes = _returned_arm_plan(root)
    checkpoint = {
        "path": str(root / "stage2" / "checkpoint.pt"),
        "sha256": plan["stage2"]["checkpoint"]["sha256"],
        "size_bytes": 123,
    }
    plan["a800_run_root"] = str(root)
    plan["stage2"]["checkpoint"] = checkpoint
    plan["remote"]["rlt_checkpoint"] = (
        "/srv/navigation-rlt-test/checkpoints/navigation_rlt_stage2.pt"
    )
    plan["evaluation_code"]["commit"] = "2" * 40
    plan["contract"]["expected_protocol_sha256"] = "3" * 64
    plan["contract"]["expected_model_sha256"] = "4" * 64
    plan["obs_root"] = str(root / "obs")
    return plan, episodes


def _verified_copy_receipt(
    path: Path, plan: dict[str, object], **overrides: object
) -> Path:
    checkpoint = plan["stage2"]["checkpoint"]
    value = {
        "schema_name": post.COPY_SCHEMA,
        "schema_version": 1,
        "status": "verified",
        "run_id": plan["run_id"],
        "source": checkpoint,
        "destination": {
            "path": plan["remote"]["rlt_checkpoint"],
            "sha256": checkpoint["sha256"],
            "size_bytes": checkpoint["size_bytes"],
        },
        "temporary_path": "/srv/navigation-rlt-test/checkpoints/.uploading",
        "temporary_absent_after_publish": True,
        "atomic_publish": True,
        "completed_unix_s": 1.0,
    }
    value.update(overrides)
    return _write(path, value)


def _canonical_returned_arm_files(
    root: Path,
    plan: dict[str, object],
    episodes: list[dict[str, object]],
    arms: tuple[str, ...] = post.ARMS,
) -> None:
    for arm in arms:
        _returned_arm_files(
            root
            / "post-training"
            / "evaluations"
            / post.ARM_TO_BEHAVIOR[arm],
            plan,
            episodes,
            arm,
        )


def _matched_result(
    path: Path, plan: dict[str, object], handoff: dict[str, object]
) -> Path:
    del handoff
    return _write(
        path,
        {
            "schema_name": post.MATCHED_SCHEMA,
            "status": "completed",
            "run_id": plan["run_id"],
            "collection_manifest_sha256": plan["collection_manifest"]["sha256"],
            "evaluation_code_inventory_sha256": plan["evaluation_code"][
                "inventory_sha256"
            ],
            "navigation_rlt_checkpoint_sha256": plan["stage2"]["checkpoint"][
                "sha256"
            ],
        },
    )


def _returned_arm_files(
    root: Path,
    plan: dict[str, object],
    episodes: list[dict[str, object]],
    arm: str,
    **overrides: object,
) -> tuple[Path, Path]:
    behavior = post.ARM_TO_EVAL_BEHAVIOR[arm]
    checkpoint = (
        None
        if arm == "native_lightnav"
        else (
            plan["contract"]["a3_checkpoint_sha256"]
            if arm == "a3_ppo"
            else plan["stage2"]["checkpoint"]["sha256"]
        )
    )
    value = {
        "schema_name": "navigation_rlt_matched_train_development_eval",
        "schema_version": 1,
        "status": "completed",
        "run_id": plan["run_id"],
        "behavior": behavior,
        "split_label": "train_development",
        "upstream_split_read": "train",
        "validation_split_reads": 0,
        "training_replay_written": False,
        "collection_manifest_sha256": plan["collection_manifest"]["sha256"],
        "run_manifest_sha256": plan["collection_manifest"]["sha256"],
        "evaluation_plan": {
            "sha256": plan["evaluation_plan"]["sha256"],
            "split_manifest_sha256": plan["split_manifest"]["sha256"],
        },
        "evaluation_code": {
            "inventory_sha256": plan["evaluation_code"]["inventory_sha256"]
        },
        "executor": {"max_physical_steps": post.MAX_STEPS},
        "seed": post.EVAL_SEED,
        "behavior_source": {"checkpoint_sha256": checkpoint},
        "context": {
            "source_sha256": (
                plan["stage2"]["context_source_sha256"]
                if arm == "navigation_rlt"
                else None
            )
        },
        "episodes": episodes,
    }
    value.update(overrides)
    evaluation = _write(root / "evaluation.json", value)
    completion = _write(
        root / "COMPLETED.json",
        {
            "schema_name": value["schema_name"],
            "schema_version": 1,
            "status": "completed",
            "behavior": behavior,
            "collection_manifest_sha256": plan["collection_manifest"]["sha256"],
            "evaluation_code": {
                "inventory_sha256": plan["evaluation_code"]["inventory_sha256"]
            },
            "evaluation_sha256": _sha(evaluation),
            "training_replay_written": False,
        },
    )
    return evaluation, completion


def _directory_link(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=True)
        return
    except OSError:
        if os.name != "nt":
            raise
    environment = dict(os.environ)
    environment["NAVRLT_TEST_LINK"] = str(link)
    environment["NAVRLT_TEST_TARGET"] = str(target)
    command = [
        "pwsh",
        "-NoProfile",
        "-Command",
        (
            "New-Item -ItemType Junction -Path $env:NAVRLT_TEST_LINK "
            "-Target $env:NAVRLT_TEST_TARGET "
            "-ErrorAction Stop | Out-Null"
        ),
    ]
    subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )


def _stage2_successor_fixture(
    run_root: Path, *, formal_overrides: dict[str, object] | None = None,
    checkpoint_regularization_overrides: dict[str, object] | None = None,
    derived_overrides: dict[str, object] | None = None,
) -> tuple[Path, Path, Path]:
    """Small hash-bound fixture for the existing successor handoff contract."""
    run_id = run_root.name
    formal_root = run_root / "stage2-successors" / "kl-successor"
    source_root = formal_root / "code"
    source_root.mkdir(parents=True)
    names = {"trainer_sha256": "navigation_rlt_fsdp_train.py",
             "policy_sha256": "navigation_rlt_policy.py", "replay_sha256": "navigation_rlt_replay.py",
             "wrapper_sha256": "run_navigation_rlt_a800_stage2_repair.sh",
             "gpu_monitor_sha256": "lightnav_a3_gpu_monitor.py"}
    claims = {}
    for claim, name in names.items():
        path = source_root / name
        path.write_bytes(name.encode())
        claims[claim] = _sha(path)
        path.chmod(0o400)
    def ref(path: Path) -> dict[str, object]:
        return {"path": str(path), "sha256": _sha(path)}
    counts = [9, 5, 1, 0]
    support = {"eligibility_schema": post.STAGE2_ELIGIBILITY_SCHEMA,
               "gate_names": list(post.STAGE2_GATE_NAMES), "gate_counts": counts,
               "gate_support_mask": [True, True, True, False], "minimum_behavior_gate_count": 1}
    regularization = {"objective_schema": post.STAGE2_BEHAVIOR_KL_OBJECTIVE_SCHEMA,
                      "behavior_kl_weight": 1.0, "direction": "behavior_to_actor_forward_kl",
                      "gate_counts": counts, "supported_uniform_mix": .05,
                      "behavior_gate_prior": [.95*n/15 + (.05/3 if n else 0) for n in counts]}
    shard = _metadata_npz(run_root / "navigation_rlt_stage2_replay/shard.npz",
        {"prefix_feature_source_sha256": "8"*64, "source_fingerprint": "9"*64,
         "representation_kind": "frozen_z_rl"})
    replay = _write(run_root / "navigation_rlt_stage2_replay/replay_manifest.json",
        {"schema_name": "navigation_rlt_replay_manifest", "schema_version": 3,
         "run_id": run_id, "split_label": "train_update", "validation_split_reads": 0,
         "representation_kind": "frozen_z_rl", "expected_episode_count": 256,
         "completed_episode_count": 256, "source_fingerprint": "9"*64,
         "shards": [ref(shard)]*256})
    stage1 = run_root / "stage1.pt"
    stage1.write_bytes(b"frozen stage1")
    training_config = {"objective_schema": post.STAGE2_REPAIR_OBJECTIVE_SCHEMA,
                       "cql_alpha": 1.0, "cql_temperature": 1.0,
                       "minimum_behavior_gate_count": 1, "epochs": 10}
    base_impl = {"policy_sha256": "a"*64, "trainer_sha256": "b"*64,
                 "replay_sha256": claims["replay_sha256"]}
    base_value = {"schema_name": "navigation_rlt_training_run_manifest", "schema_version": 1,
        "run_id": run_id, "phase": "stage2", "training": training_config,
        "implementation": base_impl, "stage2_support": support,
        "stage1_checkpoint": ref(stage1), "replay": {"manifest_sha256": _sha(replay)},
        "boundaries": {"split_label": "train_update", "upstream_split_read": "train",
                       "validation_split_reads": 0, "lightnav_frozen": True, "ttt_enabled": False}}
    base = _write(run_root / "navigation_rlt_stage2_prepare/run_manifest.json", base_value)
    prepare_receipt = _write(base.parent / "prepare_receipt.json",
        {"schema_name": "navigation_rlt_prepare_training", "schema_version": 1,
         "status": "completed", "run_id": run_id,
         "implementation": {**base_impl, "preparer_sha256": "c"*64},
         "replay": {"completion_marker": {"path": str(replay.parent/"COMPLETED.json")}}})
    _write(base.parent / "COMPLETED.json", {"schema_name": "navigation_rlt_prepare_training",
        "schema_version": 1, "status": "completed", "artifact_sha256": _sha(base),
        "receipt_sha256": _sha(prepare_receipt)})
    promotion = _write(replay.parent / "promotion_receipt.json",
        {"schema_name": "navigation_rlt_replay_promotion", "schema_version": 1,
         "status": "completed", "run_id": run_id,
         "implementation": {"policy_sha256": base_impl["policy_sha256"],
             "replay_sha256": claims["replay_sha256"], "promoter_sha256": "d"*64}})
    _write(replay.parent / "COMPLETED.json", {"schema_name": "navigation_rlt_replay_promotion",
        "schema_version": 1, "status": "completed", "output_manifest_sha256": _sha(replay),
        "promotion_receipt_sha256": _sha(promotion)})
    derived = json.loads(json.dumps(base_value))
    derived["training"].update(objective_schema=post.STAGE2_BEHAVIOR_KL_OBJECTIVE_SCHEMA, behavior_kl_weight=1.0)
    derived["implementation"].update({key: claims[key] for key in ("policy_sha256", "trainer_sha256")})
    if derived_overrides:
        derived["training"].update(derived_overrides)
    manifest = _write(formal_root / "prepare/run_manifest.json", derived)
    smoke = _write(run_root / "smoke.json", {"status": "completed"})
    guard = _write(run_root / "guard.json", {"status": "passed"})
    derivation = _write(formal_root / "prepare/DERIVATION.json",
        {"schema_name": post.STAGE2_DERIVATION_SCHEMA, "schema_version": 1, "status": "completed",
         "validation_split_reads": 0, "anti_collapse_commit": "1"*40,
         "base_epochs": 10, "formal_epochs": 10, "base_run_manifest": ref(base),
         "derived_run_manifest": ref(manifest), "implementation": derived["implementation"],
         "training_changed_paths": ["training.objective_schema", "training.behavior_kl_weight"],
         "implementation_changed_paths": ["implementation.policy_sha256", "implementation.trainer_sha256"],
         "smoke_completion": ref(smoke), "turbo_guard": ref(guard)})
    _write(derivation.parent / "COMPLETED.json", {"schema_name": post.STAGE2_DERIVATION_SCHEMA,
        "schema_version": 1, "status": "completed", "validation_split_reads": 0,
        "run_manifest_sha256": _sha(manifest), "derivation_sha256": _sha(derivation)})
    stage_root = formal_root / "navigation_rlt_stage2"
    checkpoint = stage_root / "training/checkpoints/latest/navigation_rlt_policy.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    metadata_regularization = {**regularization, **(checkpoint_regularization_overrides or {})}
    metadata = {"stage1_frozen": True, "stage2_objective":
        "reference_conditioned_min_target_twin_q_with_finite_gate_cql_and_native_reference_actor_anchor_dropout",
        "cql_enabled": True, "cql_candidate_count": 4, "entropy_target_enabled": False,
        "gate_support_mask": support["gate_support_mask"], "supported_gate_names": ["KEEP", "ADJUST", "STOP"],
        "stage2_behavior_prior_uniform_mix": .05,
        "extra": {"training_phase": "stage2", "run_manifest_sha256": _sha(manifest),
                  "replay_manifest_sha256": _sha(replay), "stage2_actor_regularization": metadata_regularization}}
    restore = {"status": "passed", "checkpoint_sha256": _sha(checkpoint),
               "stage1_frozen": True, "stage2_actor_regularization": regularization}
    epochs = [{"keep_probability_mean": .6, "adjust_probability_mean": .37,
               "stop_probability_mean": .03, "recover_probability_mean": 0.}]*10
    components = {name: f"{index:064x}" for index, name in enumerate(("actor", "q1", "q2", "target_q1", "target_q2", "stage1"), 1)}
    result_value = {"schema_name": post.TRAINING_RESULT_SCHEMA, "schema_version": 1,
        "status": "completed", "phase": "stage2", "run_id": run_id,
        "final_checkpoint": {**ref(checkpoint), "metadata": metadata}, "run_manifest": ref(manifest),
        "replay_manifest": ref(replay), "independent_restore": restore, "training": derived["training"],
        "stage2_support": support, "stage1_checkpoint_input": ref(stage1), "world_size": 8,
        "unique_gpu_uuid_count": 8, "rank_receipts": [{"data": support}]*8,
        "component_parameter_sha256": components, "action_probability_epochs": epochs,
        "optimizer_steps_per_rank": 370, "collapsed_epoch_streak": 0,
        "fsdp_wrapper": "FullyShardedDataParallel"}
    result = _write(stage_root / "training/training_result.json", result_value)
    verification = _write(stage_root / "verification.json", {"schema_name": post.VERIFICATION_SCHEMA,
        "schema_version": 1, "status": "passed", "phase": "stage2", "world_size": 8,
        "unique_gpu_uuid_count": 8, "checkpoint": ref(checkpoint), "training_result": ref(result)})
    telemetry = _write(stage_root / "telemetry.json", {"status": "completed"})
    marker = _write(stage_root / "STAGE.json", {"schema_name": post.STAGE_SCHEMA, "schema_version": 1,
        "status": "completed", "phase": "stage2", "exit_code": 0, "run_root": str(formal_root),
        "output": {"checkpoint": ref(checkpoint), "telemetry_summary": str(telemetry)},
        "verification": ref(verification), "replay_manifest": ref(replay),
        "implementation": {key: claims[key] for key in ("wrapper_sha256", "gpu_monitor_sha256")}})
    formal_value = {"schema_name": post.STAGE2_FORMAL_SCHEMA, "schema_version": 1,
        "status": "completed", "run_id": run_id, "formal_id": formal_root.name,
        "anti_collapse_commit": "1"*40, "actor_regularization": regularization,
        "epochs": 10, "collapse_threshold": .98, "collapsed_epoch_streak": 0,
        "initial_maximum_gate_probability": max(regularization["behavior_gate_prior"]),
        "action_probability_epochs": epochs, "world_size": 8, "unique_gpu_uuid_count": 8,
        "fsdp_wrapper": "FullyShardedDataParallel", "optimizer_steps_per_rank": 370,
        "component_parameter_sha256": components, "base_run_manifest": ref(base), "smoke_completion": ref(smoke),
        "formal_run_manifest": ref(manifest), "stage_marker": ref(marker), "training_result": ref(result),
        "checkpoint": ref(checkpoint), "independent_restore": restore, "a800_verification": ref(verification),
        "telemetry_summary": ref(telemetry), "turbo_guard": ref(guard), "training_scope_only": True,
        "legacy_canonical_archive_compatible": False, "downstream_archive_contract": "successor_aware_archive_required_v1",
        "validation_split_reads": 0}
    formal_value.update(formal_overrides or {})
    formal = _write(formal_root / "FORMAL_COMPLETED.json", formal_value)
    return marker, formal, source_root


class NavigationRLTPostTrainingTests(unittest.TestCase):
    def test_successor_evidence_binds_actual_sources_without_relabelling_producers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve() / "navigation-rlt-successor"
            marker, formal, source_root = _stage2_successor_fixture(root)
            evidence = post._stage2_evidence(marker, expected_marker_sha256=_sha(marker),
                run_root=root, run_id=root.name, formal_completion_path=formal,
                expected_formal_completion_sha256=_sha(formal), source_root=source_root)
            self.assertEqual(evidence["stage_run_root"], str(formal.parent))
            identity = evidence["successor_source_identity"]
            self.assertNotEqual(identity["source_claims"]["policy_sha256"],
                                identity["replay_producer_claims"]["policy_sha256"])
            self.assertNotIn("code_commit", identity)
            self.assertNotIn("preparer_sha256", identity["source_claims"])
            self.assertEqual(identity["base_prepare_receipt_value"]["replay"]["completion_marker"]["path"],
                             identity["promotion_completion"]["path"])
            for source in identity["source_files"].values():
                self.assertEqual(source["sha256"], _sha(Path(source["path"])))
            with self.assertRaisesRegex(post.PostTrainingError, "schema/status/root"):
                post._stage2_evidence(marker, expected_marker_sha256=_sha(marker), run_root=root, run_id=root.name)
            with self.assertRaises(post.PostTrainingError):
                post._stage2_evidence(marker, expected_marker_sha256=_sha(marker), run_root=root, run_id=root.name,
                    formal_completion_path=formal, expected_formal_completion_sha256="0"*64, source_root=source_root)
            outside = Path(temporary).resolve() / "outside"
            outside.mkdir()
            with self.assertRaises(post.PostTrainingError):
                post._stage2_evidence(marker, expected_marker_sha256=_sha(marker), run_root=root, run_id=root.name,
                    formal_completion_path=formal, expected_formal_completion_sha256=_sha(formal), source_root=outside)
            source = source_root / "navigation_rlt_policy.py"
            source.chmod(0o600)
            source.write_bytes(b"tampered new policy")
            with self.assertRaises(post.PostTrainingError):
                post._stage2_evidence(marker, expected_marker_sha256=_sha(marker), run_root=root, run_id=root.name,
                    formal_completion_path=formal, expected_formal_completion_sha256=_sha(formal), source_root=source_root)

    def test_successor_semantic_completion_derived_manifest_and_kl_mismatch_rejected(self) -> None:
        cases = (
            {"formal_overrides": {"status": "failed"}},
            {"formal_overrides": {"schema_name": "navigation_rlt_prepare_training"}},
            {"formal_overrides": {"validation_split_reads": True}},
            {"formal_overrides": {"collapsed_epoch_streak": 1}},
            {"checkpoint_regularization_overrides": {"behavior_kl_weight": .5}},
            {"derived_overrides": {"learning_rate": .123}},
        )
        for index, kwargs in enumerate(cases):
            with self.subTest(case=index), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve() / "navigation-rlt-successor"
                marker, formal, source_root = _stage2_successor_fixture(root, **kwargs)
                with self.assertRaises(post.PostTrainingError):
                    post._stage2_evidence(marker, expected_marker_sha256=_sha(marker), run_root=root, run_id=root.name,
                        formal_completion_path=formal, expected_formal_completion_sha256=_sha(formal), source_root=source_root)

    def test_stage2_training_additions_require_exact_registered_kl_weight(self) -> None:
        legacy = {"objective_schema": post.STAGE2_REPAIR_OBJECTIVE_SCHEMA, "cql_alpha": 1.,
                  "cql_temperature": 1., "minimum_behavior_gate_count": 1}
        self.assertEqual(post._stage2_training_additions(legacy, name="legacy", exact=True), legacy)
        modern = {**legacy, "objective_schema": post.STAGE2_BEHAVIOR_KL_OBJECTIVE_SCHEMA, "behavior_kl_weight": 1.}
        self.assertEqual(post._stage2_training_additions(modern, name="modern", exact=True), modern)
        for weight in (None, True, 0., .5, float("nan")):
            with self.subTest(weight=weight), self.assertRaises(post.PostTrainingError):
                candidate = {**modern, "behavior_kl_weight": weight}
                if weight is None:
                    candidate.pop("behavior_kl_weight")
                post._stage2_training_additions(candidate, name="invalid")
        for candidate in ({**legacy, "behavior_kl_weight": 1.}, {**modern, "objective_schema": "unknown"}):
            with self.assertRaises(post.PostTrainingError):
                post._stage2_training_additions(candidate, name="invalid")

    def test_explicit_evaluation_source_exports_real_inventory_and_keeps_driver_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve() / "navigation-rlt-source"
            source_root = root / "immutable-evaluation-code"
            source_root.mkdir(parents=True)
            files = {}
            for name in post.REQUIRED_EVAL_FILES:
                source = source_root / name
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_bytes(("immutable evaluation source "+name).encode())
                source.chmod(0o400)
                files[name] = _sha(source)
            inventory = _write(root / "inventory.json", {"schema_name": post.EVAL_INVENTORY_SCHEMA,
                "schema_version": 1, "navigation_code_commit": "7"*40, "files": files})
            code, _ = post._evaluation_inventory_evidence(inventory, expected_sha256=_sha(inventory), source_root=source_root)
            self.assertEqual(code["source_root"], str(source_root))
            self.assertNotEqual(files["navigation_rlt_post_training.py"], _sha(Path(post.__file__)))
            with self.assertRaises(post.PostTrainingError):
                post._evaluation_inventory_evidence(inventory, expected_sha256=_sha(inventory))
            eval_plan = _write(root / "eval-plan.json", {"status": "planned"})
            plan = {"evaluation_plan": {"path": str(eval_plan), "sha256": _sha(eval_plan)},
                    "evaluation_code": {**code, "inventory_path": str(inventory)}}
            archive = root / "evaluation-runtime.zip"
            post._build_evaluation_runtime_archive(plan, root, archive)
            with zipfile.ZipFile(archive) as zipped:
                self.assertEqual(zipped.read("eval-code/navigation_rlt_post_training.py"),
                                 (source_root / "navigation_rlt_post_training.py").read_bytes())
                self.assertEqual(len(zipped.namelist()), len(post.REQUIRED_EVAL_FILES)+2)
            lineage = {"training_source_identities": {"stage2_repair": {"code_commit": "8"*40},
                                                      "stage2_successor": {"source_root": "new trainer"}}}
            post._require_repair_evaluation_code_commit(code, lineage)
            with self.assertRaises(post.PostTrainingError):
                post._require_repair_evaluation_code_commit({key:value for key,value in code.items() if key!="source_root"}, lineage)
            with self.assertRaises(post.PostTrainingError):
                post._require_repair_evaluation_code_commit(code, {"training_source_identities": {"stage2_repair": {"code_commit": "8"*40}}})
            alias = root / "evaluation-alias"
            _directory_link(alias, source_root)
            try:
                with self.assertRaises(post.PostTrainingError):
                    post._evaluation_inventory_evidence(inventory, expected_sha256=_sha(inventory), source_root=alias)
                with self.assertRaises(post.PostTrainingError):
                    post._local_evaluation_source("../inventory.json", source_root)
            finally:
                if alias.is_symlink():
                    alias.unlink()
                else:
                    alias.rmdir()


    def test_atomic_file_publication_preserves_competing_sentinels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "result.json"
            real_link = post.os.link

            def competing_link(source, target, **kwargs):
                Path(target).write_bytes(b"sentinel")
                return real_link(source, target, **kwargs)

            with patch.object(post.os, "link", side_effect=competing_link):
                with self.assertRaises(FileExistsError):
                    post._atomic_json_new(destination, {"new": True})
            self.assertEqual(destination.read_bytes(), b"sentinel")
            self.assertEqual(list(root.glob("result.json.staging-*")), [])

            returned = root / "post-training" / "evaluations" / "native" / "evaluation.json"
            payload = b"downloaded"

            def copy_back(command, **_kwargs):
                Path(command[-1]).write_bytes(payload)
                return subprocess.CompletedProcess(command, 0, "", "")

            def competing_download_link(source, target, **kwargs):
                Path(target).write_bytes(b"download sentinel")
                return real_link(source, target, **kwargs)

            with patch.object(post, "_run", side_effect=copy_back), patch.object(
                post.os, "link", side_effect=competing_download_link
            ):
                with self.assertRaises(FileExistsError):
                    post._scp_from(
                        "ebk-test",
                        "/remote/result.json",
                        returned,
                        expected_sha256=hashlib.sha256(payload).hexdigest(),
                        expected_size_bytes=len(payload),
                        timeout=1,
                        name="download",
                        containment_root=root,
                    )
            self.assertEqual(returned.read_bytes(), b"download sentinel")
            self.assertEqual(list(returned.parent.glob("evaluation.json.staging-*")), [])

    def test_download_recheck_rejects_parent_link_drift_without_external_delete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            destination = (
                root
                / "post-training"
                / "evaluations"
                / "native"
                / "evaluation.json"
            )
            outside = Path(temporary) / "outside"
            outside.mkdir()
            payload = b"downloaded"
            observed: dict[str, Path] = {}

            def drift_parent(command, **_kwargs):
                staging = Path(command[-1])
                staging.write_bytes(payload)
                original_parent = staging.parent.with_name("native-original")
                staging.parent.rename(original_parent)
                outside_staging = outside / staging.name
                outside_staging.write_bytes(payload)
                _directory_link(staging.parent, outside)
                observed["outside_staging"] = outside_staging
                observed["original_staging"] = original_parent / staging.name
                return subprocess.CompletedProcess(command, 0, "", "")

            with patch.object(post, "_run", side_effect=drift_parent):
                with self.assertRaisesRegex(
                    post.PostTrainingError, "link/reparse point"
                ):
                    post._scp_from(
                        "ebk-test",
                        "/remote/result.json",
                        destination,
                        expected_sha256=hashlib.sha256(payload).hexdigest(),
                        expected_size_bytes=len(payload),
                        timeout=1,
                        name="download",
                        containment_root=root,
                    )
            self.assertEqual(observed["outside_staging"].read_bytes(), payload)
            self.assertTrue(observed["original_staging"].is_file())
            self.assertFalse((outside / "evaluation.json").exists())

    def test_local_returned_arm_matches_direct_return_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            local_root = parent / "local"
            local_root.mkdir()
            local_plan, episodes = _returned_arm_plan(local_root)
            evaluation, completion = _returned_arm_files(
                local_root / "inbox", local_plan, episodes, "navigation_rlt"
            )
            local_row = post._publish_local_returned_arm(
                local_plan,
                local_root,
                "navigation_rlt",
                evaluation,
                completion,
                expected_evaluation_sha256=_sha(evaluation),
                expected_evaluation_size_bytes=evaluation.stat().st_size,
                expected_completion_sha256=_sha(completion),
                expected_completion_size_bytes=completion.stat().st_size,
            )
            self.assertTrue(evaluation.is_file())
            self.assertTrue(completion.is_file())

            direct_root = parent / "direct"
            direct_root.mkdir()
            direct_plan, direct_episodes = _returned_arm_plan(direct_root)
            remote_eval, remote_done = _returned_arm_files(
                parent / "remote", direct_plan, direct_episodes, "navigation_rlt"
            )
            payloads = {
                "/remote/evaluation.json": remote_eval.read_bytes(),
                "/remote/COMPLETED.json": remote_done.read_bytes(),
            }

            def local_download(_host, source, destination, **kwargs):
                destination = Path(destination)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(payloads[source])
                self.assertEqual(_sha(destination), kwargs["expected_sha256"])
                self.assertEqual(
                    destination.stat().st_size, kwargs["expected_size_bytes"]
                )
                return destination.resolve(strict=True)

            remote_result = {
                "evaluation_path": "/remote/evaluation.json",
                "evaluation_sha256": _sha(remote_eval),
                "evaluation_size_bytes": remote_eval.stat().st_size,
                "completion_path": "/remote/COMPLETED.json",
                "completion_sha256": _sha(remote_done),
                "completion_size_bytes": remote_done.stat().st_size,
            }
            with patch.object(post, "_scp_from", side_effect=local_download), patch.object(
                post,
                "_publish_local_returned_arm",
                wraps=post._publish_local_returned_arm,
            ) as publish:
                direct_row = post._return_arm(
                    direct_plan, direct_root, "navigation_rlt", remote_result
                )
            publish.assert_called_once()
            for key in ("arm", "behavior", "arm_config_sha256", "ordered_episode_identity_sha256"):
                self.assertEqual(local_row[key], direct_row[key])
            self.assertEqual(
                local_row["evaluation"]["sha256"], direct_row["evaluation"]["sha256"]
            )
            self.assertEqual(
                local_row["completion"]["sha256"], direct_row["completion"]["sha256"]
            )

    def test_finalize_remote_copy_cli_publishes_checkpoint_and_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "remote-copy-test"
            checkpoint_root = run_root / "checkpoints"
            checkpoint_root.mkdir(parents=True)
            temporary_checkpoint = checkpoint_root / ".checkpoint.pt.uploading"
            temporary_checkpoint.write_bytes(b"checkpoint payload")
            temporary_identity = post._file_identity(temporary_checkpoint.lstat())
            final = checkpoint_root / "checkpoint.pt"
            receipt = run_root / "post-training" / "VERIFIED_COPY.json"
            source_path = "/home/navcloud/work/run/stage2/checkpoint.pt"
            argv = [
                "finalize-remote-copy",
                "--run-root",
                str(run_root),
                "--run-id",
                run_root.name,
                "--source-path",
                source_path,
                "--expected-checkpoint-sha256",
                _sha(temporary_checkpoint),
                "--expected-checkpoint-size-bytes",
                str(temporary_checkpoint.stat().st_size),
                "--temporary",
                str(temporary_checkpoint),
                "--final",
                str(final),
                "--receipt",
                str(receipt),
            ]
            with patch("builtins.print") as printed:
                self.assertEqual(post.main(argv), 0)
            result = json.loads(printed.call_args.args[0])
            value = _load(receipt)
            self.assertFalse(temporary_checkpoint.exists())
            self.assertEqual(post._file_identity(final.lstat()), temporary_identity)
            self.assertEqual(final.read_bytes(), b"checkpoint payload")
            self.assertEqual(value["schema_name"], post.COPY_SCHEMA)
            self.assertEqual(value["source"]["path"], source_path)
            self.assertEqual(value["destination"]["path"], str(final))
            self.assertTrue(value["atomic_publish"])
            self.assertTrue(value["temporary_absent_after_publish"])
            self.assertEqual(result["receipt_sha256"], _sha(receipt))
            self.assertEqual(result["checkpoint_sha256"], _sha(final))

    def test_finalize_remote_copy_rejects_conflict_and_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            for case in ("conflict", "hash"):
                with self.subTest(case=case):
                    run_root = parent / case
                    run_root.mkdir()
                    uploading = run_root / "checkpoint.pt.uploading"
                    uploading.write_bytes(b"checkpoint payload")
                    final = run_root / "checkpoint.pt"
                    receipt = run_root / "VERIFIED_COPY.json"
                    if case == "conflict":
                        final.write_bytes(b"sentinel")
                    expected_sha = _sha(uploading) if case == "conflict" else "0" * 64
                    args = argparse.Namespace(
                        run_root=run_root,
                        run_id=run_root.name,
                        source_path="/a800/checkpoint.pt",
                        expected_checkpoint_sha256=expected_sha,
                        expected_checkpoint_size_bytes=uploading.stat().st_size,
                        temporary=uploading,
                        final=final,
                        receipt=receipt,
                    )
                    error = FileExistsError if case == "conflict" else post.PostTrainingError
                    with self.assertRaises(error):
                        post.finalize_remote_copy(args)
                    self.assertTrue(uploading.is_file())
                    self.assertFalse(receipt.exists())
                    if case == "conflict":
                        self.assertEqual(final.read_bytes(), b"sentinel")
                    else:
                        self.assertFalse(final.exists())

    def test_finalize_remote_copy_preserves_temporary_on_publish_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "publish-failure"
            run_root.mkdir()
            uploading = run_root / "checkpoint.pt.uploading"
            uploading.write_bytes(b"checkpoint payload")
            final = run_root / "checkpoint.pt"
            receipt = run_root / "VERIFIED_COPY.json"
            args = argparse.Namespace(
                run_root=run_root,
                run_id=run_root.name,
                source_path="/a800/checkpoint.pt",
                expected_checkpoint_sha256=_sha(uploading),
                expected_checkpoint_size_bytes=uploading.stat().st_size,
                temporary=uploading,
                final=final,
                receipt=receipt,
            )
            with patch.object(
                post, "_publish_file_new", side_effect=OSError("injected publish failure")
            ):
                with self.assertRaisesRegex(OSError, "injected publish failure"):
                    post.finalize_remote_copy(args)
            self.assertEqual(uploading.read_bytes(), b"checkpoint payload")
            self.assertFalse(final.exists())
            self.assertFalse(receipt.exists())

    def test_publish_returned_copy_cli_hash_binds_and_revalidates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "navigation-rlt-test"
            run_root.mkdir()
            plan, _ = _relay_plan(run_root)
            plan_path = _write(run_root / "reviewed-plan.json", {"reviewed": True})
            plan_sha = _sha(plan_path)
            returned = _verified_copy_receipt(
                run_root / "relay" / "VERIFIED_COPY.json", plan
            )
            publication = run_root / "post-training" / "copy-return.json"
            argv = [
                "publish-returned-copy",
                "--plan",
                str(plan_path),
                "--expected-plan-sha256",
                plan_sha,
                "--verified-copy",
                str(returned),
                "--expected-verified-copy-sha256",
                _sha(returned),
                "--expected-verified-copy-size-bytes",
                str(returned.stat().st_size),
                "--receipt",
                str(publication),
            ]
            with patch.object(
                post, "_load_plan", return_value=(plan_path, plan, plan_sha)
            ), patch.object(
                post, "_revalidate_plan", return_value=(run_root, plan["run_id"])
            ) as revalidate, patch.object(post, "_ssh_json") as ssh, patch.object(
                post, "_scp_to"
            ) as scp_to, patch.object(post, "_scp_from") as scp_from, patch(
                "builtins.print"
            ):
                self.assertEqual(post.main(argv), 0)
            revalidate.assert_called_once_with(plan)
            ssh.assert_not_called()
            scp_to.assert_not_called()
            scp_from.assert_not_called()
            canonical = run_root / "post-training" / "VERIFIED_COPY.json"
            self.assertEqual(canonical.read_bytes(), returned.read_bytes())
            value = _load(publication)
            self.assertEqual(
                value["schema_name"], "navigation_rlt_returned_copy_publication"
            )
            self.assertEqual(value["result"]["path"], str(canonical))
            self.assertEqual(value["result"]["sha256"], _sha(canonical))
            self.assertEqual(value["validation_split_reads"], 0)

    def test_publish_returned_copy_rejects_key_identity_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "navigation-rlt-test"
            run_root.mkdir()
            plan, _ = _relay_plan(run_root)
            plan_path = _write(run_root / "reviewed-plan.json", {"reviewed": True})
            plan_sha = _sha(plan_path)
            returned = _verified_copy_receipt(
                run_root / "relay" / "VERIFIED_COPY.json",
                plan,
                run_id="wrong-run",
            )
            args = argparse.Namespace(
                plan=plan_path,
                expected_plan_sha256=plan_sha,
                verified_copy=returned,
                expected_verified_copy_sha256=_sha(returned),
                expected_verified_copy_size_bytes=returned.stat().st_size,
                receipt=run_root / "post-training" / "copy-return.json",
            )
            with patch.object(
                post, "_load_plan", return_value=(plan_path, plan, plan_sha)
            ), patch.object(
                post, "_revalidate_plan", return_value=(run_root, plan["run_id"])
            ):
                with self.assertRaisesRegex(
                    post.PostTrainingError, "verified-copy receipt differs"
                ):
                    post.publish_returned_copy(args)
            self.assertFalse(
                (run_root / "post-training" / "VERIFIED_COPY.json").exists()
            )
            self.assertFalse(Path(args.receipt).exists())

    def test_publish_returned_arm_cli_revalidates_and_writes_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "run"
            run_root.mkdir()
            plan, episodes = _returned_arm_plan(run_root)
            plan_path = _write(run_root / "reviewed-plan.json", {"reviewed": True})
            plan_sha = _sha(plan_path)
            evaluation, completion = _returned_arm_files(
                run_root / "inbox", plan, episodes, "a3_ppo"
            )
            receipt = run_root / "post-training" / "a3-return.json"
            argv = [
                "publish-returned-arm",
                "--plan",
                str(plan_path),
                "--expected-plan-sha256",
                plan_sha,
                "--receipt",
                str(receipt),
                "--arm",
                "a3_ppo",
                "--evaluation",
                str(evaluation),
                "--completion",
                str(completion),
                "--expected-evaluation-sha256",
                _sha(evaluation),
                "--expected-evaluation-size-bytes",
                str(evaluation.stat().st_size),
                "--expected-completion-sha256",
                _sha(completion),
                "--expected-completion-size-bytes",
                str(completion.stat().st_size),
            ]
            with patch.object(
                post, "_load_plan", return_value=(plan_path, plan, plan_sha)
            ), patch.object(
                post, "_revalidate_plan", return_value=(run_root, plan["run_id"])
            ) as revalidate, patch("builtins.print"):
                post.main(argv)
            revalidate.assert_called_once_with(plan)
            value = _load(receipt)
            self.assertEqual(value["schema_name"], "navigation_rlt_returned_arm_publication")
            self.assertEqual(value["result"]["arm"], "a3_ppo")
            self.assertEqual(value["post_training_plan"]["sha256"], plan_sha)
            self.assertEqual(value["validation_split_reads"], 0)

    def test_finalize_relay_cli_writes_completion_after_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "navigation-rlt-test"
            run_root.mkdir()
            plan, episodes = _relay_plan(run_root)
            plan["training_source_identities"] = {
                "stage1": {"code_commit": "1" * 40},
                "stage2_repair": {"code_commit": "2" * 40},
            }
            plan_path = _write(run_root / "reviewed-plan.json", {"reviewed": True})
            plan_sha = _sha(plan_path)
            copy_path = _verified_copy_receipt(
                run_root / "post-training" / "VERIFIED_COPY.json", plan
            )
            _canonical_returned_arm_files(run_root, plan, episodes)

            def compare(
                observed_plan: dict[str, object],
                observed_root: Path,
                handoff: dict[str, object],
            ) -> tuple[Path, str, Path, str]:
                self.assertIs(observed_plan, plan)
                self.assertEqual(observed_root, run_root)
                self.assertEqual(
                    [row["arm"] for row in handoff["arms"]], list(post.ARMS)
                )
                outputs = post._fixed_outputs(run_root)
                matched = _write(
                    outputs["matched_result"],
                    {
                        "schema_name": post.MATCHED_SCHEMA,
                        "status": "completed",
                        "run_id": plan["run_id"],
                    },
                )
                markdown = outputs["matched_markdown"]
                markdown.write_text("matched\n", encoding="utf-8")
                return matched, _sha(matched), markdown, _sha(markdown)

            argv = [
                "finalize-relay",
                "--plan",
                str(plan_path),
                "--expected-plan-sha256",
                plan_sha,
            ]
            with patch.object(
                post, "_load_plan", return_value=(plan_path, plan, plan_sha)
            ), patch.object(
                post, "_revalidate_plan", return_value=(run_root, plan["run_id"])
            ) as revalidate, patch.object(
                post, "_invoke_compare", side_effect=compare
            ) as invoke_compare, patch.object(
                post, "_invoke_archive"
            ) as archive, patch.object(post, "_ssh_json") as ssh, patch.object(
                post, "_scp_to"
            ) as scp_to, patch.object(post, "_scp_from") as scp_from, patch.object(
                post, "_run_remote_arm"
            ) as remote_arm, patch(
                "builtins.print"
            ) as printed:
                self.assertEqual(post.main(argv), 0)
            revalidate.assert_called_once_with(plan)
            invoke_compare.assert_called_once()
            archive.assert_called_once()
            ssh.assert_not_called()
            scp_to.assert_not_called()
            scp_from.assert_not_called()
            remote_arm.assert_not_called()
            handoff_path = run_root / "post-training" / "REMOTE_EVAL_COMPLETED.json"
            completion_path = (
                run_root / "post-training" / "POST_TRAINING_COMPLETED.json"
            )
            handoff = _load(handoff_path)
            completion = _load(completion_path)
            self.assertEqual(handoff["arm_order"], list(post.ARMS))
            self.assertEqual(
                [row["arm"] for row in handoff["arms"]], list(post.ARMS)
            )
            self.assertEqual(completion["status"], "completed")
            self.assertEqual(
                completion["training_source_identities"],
                plan["training_source_identities"],
            )
            self.assertEqual(
                completion["verified_copy_receipt"]["sha256"], _sha(copy_path)
            )
            self.assertEqual(
                completion["evaluation_handoff_receipt"]["sha256"],
                _sha(handoff_path),
            )
            self.assertEqual(
                completion["matched_result"]["sha256"],
                _sha(run_root / "post-training" / "matched_comparison.json"),
            )
            self.assertEqual(
                json.loads(printed.call_args.args[0])["completion_sha256"],
                _sha(completion_path),
            )
            archive_args = archive.call_args.args
            self.assertEqual(archive_args[1], plan_path)
            self.assertEqual(archive_args[7], copy_path)
            self.assertEqual(archive_args[9], handoff_path)

    def test_finalize_relay_reuses_identical_intermediate_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "navigation-rlt-test"
            run_root.mkdir()
            plan, episodes = _relay_plan(run_root)
            plan_path = _write(run_root / "reviewed-plan.json", {"reviewed": True})
            plan_sha = _sha(plan_path)
            _verified_copy_receipt(
                run_root / "post-training" / "VERIFIED_COPY.json", plan
            )
            _canonical_returned_arm_files(run_root, plan, episodes)
            returned = post._canonical_returned_arms(plan, run_root)
            handoff_path, _ = post._write_handoff(plan, run_root, returned)
            handoff = _load(handoff_path)
            matched_path = _matched_result(
                run_root / "post-training" / "matched_comparison.json",
                plan,
                handoff,
            )
            markdown_path = run_root / "post-training" / "matched_comparison.md"
            markdown_path.write_text("matched\n", encoding="utf-8")
            handoff_identity = post._file_identity(handoff_path.lstat())
            matched_identity = post._file_identity(matched_path.lstat())
            markdown_identity = post._file_identity(markdown_path.lstat())
            matched_bytes = matched_path.read_bytes()
            markdown_bytes = markdown_path.read_bytes()

            def regenerate(
                _plan: dict[str, object],
                _root: Path,
                _handoff: dict[str, object],
                output_json: Path,
                output_markdown: Path,
            ) -> tuple[Path, str, Path, str]:
                output_json.write_bytes(matched_bytes)
                output_markdown.write_bytes(markdown_bytes)
                return (
                    output_json.resolve(strict=True),
                    _sha(output_json),
                    output_markdown.resolve(strict=True),
                    _sha(output_markdown),
                )

            args = argparse.Namespace(
                plan=plan_path, expected_plan_sha256=plan_sha
            )
            with patch.object(
                post, "_load_plan", return_value=(plan_path, plan, plan_sha)
            ), patch.object(
                post, "_revalidate_plan", return_value=(run_root, plan["run_id"])
            ), patch.object(
                post, "_invoke_compare_outputs", side_effect=regenerate
            ) as regenerated, patch.object(
                post, "_invoke_compare"
            ) as fresh_compare, patch.object(post, "_invoke_archive") as archive:
                result = post.finalize_relay(args)
            regenerated.assert_called_once()
            fresh_compare.assert_not_called()
            archive.assert_called_once()
            self.assertEqual(result["status"], "completed")
            self.assertEqual(post._file_identity(handoff_path.lstat()), handoff_identity)
            self.assertEqual(post._file_identity(matched_path.lstat()), matched_identity)
            self.assertEqual(
                post._file_identity(markdown_path.lstat()), markdown_identity
            )
            self.assertEqual(matched_path.read_bytes(), matched_bytes)
            self.assertEqual(markdown_path.read_bytes(), markdown_bytes)

    def test_finalize_relay_rejects_conflicting_intermediate_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "navigation-rlt-test"
            run_root.mkdir()
            plan, episodes = _relay_plan(run_root)
            plan_path = _write(run_root / "reviewed-plan.json", {"reviewed": True})
            plan_sha = _sha(plan_path)
            _verified_copy_receipt(
                run_root / "post-training" / "VERIFIED_COPY.json", plan
            )
            _canonical_returned_arm_files(run_root, plan, episodes)
            returned = post._canonical_returned_arms(plan, run_root)
            handoff_path, _ = post._write_handoff(plan, run_root, returned)
            handoff = _load(handoff_path)
            matched_path = _matched_result(
                run_root / "post-training" / "matched_comparison.json",
                plan,
                handoff,
            )
            matched = _load(matched_path)
            matched["run_id"] = "wrong-run"
            _write(matched_path, matched)
            markdown_path = run_root / "post-training" / "matched_comparison.md"
            markdown_path.write_text("matched\n", encoding="utf-8")
            args = argparse.Namespace(
                plan=plan_path, expected_plan_sha256=plan_sha
            )
            with patch.object(
                post, "_load_plan", return_value=(plan_path, plan, plan_sha)
            ), patch.object(
                post, "_revalidate_plan", return_value=(run_root, plan["run_id"])
            ), patch.object(
                post, "_invoke_compare_outputs"
            ) as regenerated, patch.object(post, "_invoke_archive") as archive:
                with self.assertRaisesRegex(
                    post.PostTrainingError, "matched comparison identity differs"
                ):
                    post.finalize_relay(args)
            regenerated.assert_not_called()
            archive.assert_not_called()
            self.assertEqual(_load(matched_path)["run_id"], "wrong-run")
            self.assertFalse(
                (run_root / "post-training" / "POST_TRAINING_COMPLETED.json").exists()
            )

    def test_finalize_relay_rejects_missing_arm_before_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "navigation-rlt-test"
            run_root.mkdir()
            plan, episodes = _relay_plan(run_root)
            plan_path = _write(run_root / "reviewed-plan.json", {"reviewed": True})
            plan_sha = _sha(plan_path)
            _verified_copy_receipt(
                run_root / "post-training" / "VERIFIED_COPY.json", plan
            )
            _canonical_returned_arm_files(run_root, plan, episodes, post.ARMS[:2])
            args = argparse.Namespace(
                plan=plan_path, expected_plan_sha256=plan_sha
            )
            with patch.object(
                post, "_load_plan", return_value=(plan_path, plan, plan_sha)
            ), patch.object(
                post, "_revalidate_plan", return_value=(run_root, plan["run_id"])
            ) as revalidate, patch.object(post, "_invoke_compare") as compare, patch.object(
                post, "_invoke_archive"
            ) as archive:
                with self.assertRaises(post.PostTrainingError):
                    post.finalize_relay(args)
            revalidate.assert_called_once_with(plan)
            compare.assert_not_called()
            archive.assert_not_called()
            self.assertFalse(
                (run_root / "post-training" / "REMOTE_EVAL_COMPLETED.json").exists()
            )
            self.assertFalse(
                (run_root / "post-training" / "POST_TRAINING_COMPLETED.json").exists()
            )

    def test_finalize_relay_rejects_existing_terminal_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "navigation-rlt-test"
            run_root.mkdir()
            plan, _ = _relay_plan(run_root)
            plan_path = _write(run_root / "reviewed-plan.json", {"reviewed": True})
            plan_sha = _sha(plan_path)
            completion = (
                run_root / "post-training" / "POST_TRAINING_COMPLETED.json"
            )
            completion.parent.mkdir()
            completion.write_bytes(b"sentinel")
            args = argparse.Namespace(
                plan=plan_path, expected_plan_sha256=plan_sha
            )
            with patch.object(
                post, "_load_plan", return_value=(plan_path, plan, plan_sha)
            ), patch.object(
                post, "_revalidate_plan", return_value=(run_root, plan["run_id"])
            ), patch.object(post, "_invoke_compare") as compare, patch.object(
                post, "_invoke_archive"
            ) as archive:
                with self.assertRaises(FileExistsError):
                    post.finalize_relay(args)
            compare.assert_not_called()
            archive.assert_not_called()
            self.assertEqual(completion.read_bytes(), b"sentinel")
            self.assertFalse(
                (run_root / "post-training" / "REMOTE_EVAL_COMPLETED.json").exists()
            )

    def test_finalize_relay_rejects_canonical_copy_identity_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "navigation-rlt-test"
            run_root.mkdir()
            plan, _ = _relay_plan(run_root)
            plan_path = _write(run_root / "reviewed-plan.json", {"reviewed": True})
            plan_sha = _sha(plan_path)
            _verified_copy_receipt(
                run_root / "post-training" / "VERIFIED_COPY.json",
                plan,
                run_id="wrong-run",
            )
            args = argparse.Namespace(
                plan=plan_path, expected_plan_sha256=plan_sha
            )
            with patch.object(
                post, "_load_plan", return_value=(plan_path, plan, plan_sha)
            ), patch.object(
                post, "_revalidate_plan", return_value=(run_root, plan["run_id"])
            ), patch.object(post, "_invoke_compare") as compare, patch.object(
                post, "_invoke_archive"
            ) as archive:
                with self.assertRaisesRegex(
                    post.PostTrainingError, "verified-copy receipt differs"
                ):
                    post.finalize_relay(args)
            compare.assert_not_called()
            archive.assert_not_called()
            self.assertFalse(
                (run_root / "post-training" / "REMOTE_EVAL_COMPLETED.json").exists()
            )

    def test_local_returned_arm_rejects_train_only_guard_tampering(self) -> None:
        cases = {
            "split": {"split_label": "val_unseen"},
            "upstream": {"upstream_split_read": "val"},
            "validation_read": {"validation_split_reads": 1},
            "training_write": {"training_replay_written": True},
            "episode_count": {"episodes": "truncate"},
            "context": {
                "_arm": "navigation_rlt",
                "context": {"source_sha256": "0" * 64},
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            for name, overrides in cases.items():
                with self.subTest(name=name):
                    run_root = parent / name
                    run_root.mkdir()
                    plan, episodes = _returned_arm_plan(run_root)
                    values = dict(overrides)
                    arm = str(values.pop("_arm", "native_lightnav"))
                    if values.get("episodes") == "truncate":
                        values["episodes"] = episodes[:-1]
                    episode_rows = values.pop("episodes", episodes)
                    evaluation, completion = _returned_arm_files(
                        run_root / "inbox",
                        plan,
                        episode_rows,
                        arm,
                        **values,
                    )
                    with self.assertRaises(post.PostTrainingError):
                        post._publish_local_returned_arm(
                            plan,
                            run_root,
                            arm,
                            evaluation,
                            completion,
                            expected_evaluation_sha256=_sha(evaluation),
                            expected_evaluation_size_bytes=evaluation.stat().st_size,
                            expected_completion_sha256=_sha(completion),
                            expected_completion_size_bytes=completion.stat().st_size,
                        )
                    self.assertTrue(evaluation.is_file())
                    self.assertTrue(completion.is_file())
                    self.assertFalse(
                        (run_root / "post-training" / "evaluations").exists()
                    )

    def test_local_returned_arm_is_hash_bound_and_create_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "run"
            run_root.mkdir()
            plan, episodes = _returned_arm_plan(run_root)
            evaluation, completion = _returned_arm_files(
                run_root / "inbox", plan, episodes, "native_lightnav"
            )
            with self.assertRaisesRegex(post.PostTrainingError, "readback differs"):
                post._publish_local_returned_arm(
                    plan,
                    run_root,
                    "native_lightnav",
                    evaluation,
                    completion,
                    expected_evaluation_sha256="0" * 64,
                    expected_evaluation_size_bytes=evaluation.stat().st_size,
                    expected_completion_sha256=_sha(completion),
                    expected_completion_size_bytes=completion.stat().st_size,
                )
            target = (
                run_root
                / "post-training"
                / "evaluations"
                / "native"
                / "evaluation.json"
            )
            target.parent.mkdir(parents=True)
            target.write_bytes(b"sentinel")
            with self.assertRaisesRegex(
                post.PostTrainingError, "existing canonical evaluation differs"
            ):
                post._publish_local_returned_arm(
                    plan,
                    run_root,
                    "native_lightnav",
                    evaluation,
                    completion,
                    expected_evaluation_sha256=_sha(evaluation),
                    expected_evaluation_size_bytes=evaluation.stat().st_size,
                    expected_completion_sha256=_sha(completion),
                    expected_completion_size_bytes=completion.stat().st_size,
                )
            self.assertEqual(target.read_bytes(), b"sentinel")
            self.assertTrue(evaluation.is_file())
            self.assertTrue(completion.is_file())
            self.assertFalse((target.parent / "COMPLETED.json").exists())

    def test_local_returned_arm_retries_after_completion_publish_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "run"
            run_root.mkdir()
            plan, episodes = _returned_arm_plan(run_root)
            evaluation, completion = _returned_arm_files(
                run_root / "inbox", plan, episodes, "native_lightnav"
            )
            arguments = dict(
                expected_evaluation_sha256=_sha(evaluation),
                expected_evaluation_size_bytes=evaluation.stat().st_size,
                expected_completion_sha256=_sha(completion),
                expected_completion_size_bytes=completion.stat().st_size,
            )
            real_copy = post._copy_file_new
            calls = 0

            def fail_second(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected completion publication failure")
                return real_copy(*args, **kwargs)

            with patch.object(post, "_copy_file_new", side_effect=fail_second):
                with self.assertRaisesRegex(OSError, "injected completion"):
                    post._publish_local_returned_arm(
                        plan,
                        run_root,
                        "native_lightnav",
                        evaluation,
                        completion,
                        **arguments,
                    )
            target_root = run_root / "post-training" / "evaluations" / "native"
            published = target_root / "evaluation.json"
            self.assertEqual(_sha(published), arguments["expected_evaluation_sha256"])
            self.assertFalse((target_root / "COMPLETED.json").exists())
            identity = post._file_identity(published.lstat())
            row = post._publish_local_returned_arm(
                plan,
                run_root,
                "native_lightnav",
                evaluation,
                completion,
                **arguments,
            )
            self.assertEqual(post._file_identity(published.lstat()), identity)
            self.assertEqual(row["completion"]["sha256"], _sha(target_root / "COMPLETED.json"))

    def test_execute_rejects_mutated_outputs_before_remote_work(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            run_root = parent / "run"
            run_root.mkdir()
            plan_path = _write(run_root / "plan.json", {"planned": True})
            plan = {
                "run_id": "run",
                "a800_run_root": str(run_root),
                "outputs": {
                    key: str(path) for key, path in post._fixed_outputs(run_root).items()
                },
            }
            plan["outputs"]["completion"] = str(parent / "outside.json")
            args = argparse.Namespace(
                plan=plan_path, expected_plan_sha256="0" * 64
            )
            with patch.object(
                post, "_load_plan", return_value=(plan_path, plan, "0" * 64)
            ), patch.object(
                post, "_revalidate_plan", return_value=(run_root, "run")
            ), patch.object(post, "_deploy_evaluation_runtime") as remote:
                with self.assertRaisesRegex(post.PostTrainingError, "output path differs"):
                    post.execute_plan(args)
            remote.assert_not_called()
            self.assertFalse((parent / "outside.json").exists())

    def test_execute_rejects_linked_output_parent_before_remote_work(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            run_root = parent / "run"
            outside = parent / "outside"
            run_root.mkdir()
            outside.mkdir()
            plan_path = _write(run_root / "plan.json", {"planned": True})
            linked_parent = run_root / "post-training"
            _directory_link(linked_parent, outside)
            plan = {
                "run_id": "run",
                "a800_run_root": str(run_root),
                "outputs": {
                    key: str(path) for key, path in post._fixed_outputs(run_root).items()
                },
            }
            args = argparse.Namespace(
                plan=plan_path, expected_plan_sha256="0" * 64
            )
            with patch.object(
                post, "_load_plan", return_value=(plan_path, plan, "0" * 64)
            ), patch.object(
                post, "_revalidate_plan", return_value=(run_root, "run")
            ), patch.object(post, "_deploy_evaluation_runtime") as remote:
                with self.assertRaisesRegex(
                    post.PostTrainingError, "link/reparse point"
                ):
                    post.execute_plan(args)
            remote.assert_not_called()
            self.assertEqual(list(outside.iterdir()), [])

    def test_execute_rejects_linked_plan_parent_before_remote_work(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            run_root = parent / "run"
            real_parent = run_root / "real-plans"
            run_root.mkdir()
            real_parent.mkdir()
            linked_parent = run_root / "linked-plans"
            _directory_link(linked_parent, real_parent)
            plan = {
                "schema_name": post.PLAN_SCHEMA,
                "schema_version": post.PLAN_VERSION,
                "status": "planned",
                "run_id": "run",
                "a800_run_root": str(run_root),
                "outputs": {
                    key: str(path) for key, path in post._fixed_outputs(run_root).items()
                },
            }
            resolved_plan = _write(real_parent / "plan.json", plan)
            lexical_plan = linked_parent / "plan.json"
            args = argparse.Namespace(
                plan=lexical_plan, expected_plan_sha256=_sha(resolved_plan)
            )
            with patch.object(
                post, "_revalidate_plan", return_value=(run_root, "run")
            ), patch.object(post, "_deploy_evaluation_runtime") as remote:
                with self.assertRaisesRegex(
                    post.PostTrainingError, "link/reparse point"
                ):
                    post.execute_plan(args)
            remote.assert_not_called()

    def test_embedded_remote_publishers_are_no_replace(self) -> None:
        self.assertIn("renameat2", post.REMOTE_FINALIZE_EVAL_DEPLOYMENT)
        self.assertIn("RENAME_NOREPLACE", post.REMOTE_FINALIZE_EVAL_DEPLOYMENT)
        self.assertNotIn("os.replace", post.REMOTE_FINALIZE_EVAL_DEPLOYMENT)
        self.assertIn("os.link(temporary,destination", post.REMOTE_FINALIZE_COPY)
        self.assertNotIn("os.replace", post.REMOTE_FINALIZE_COPY)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = b"checkpoint"
            expected = hashlib.sha256(payload).hexdigest()
            real_link = post.os.link
            real_open = post.os.open
            sync_anchor = root / "directory-sync"
            sync_anchor.write_bytes(b"")

            def remote_open(path, flags, *mode):
                if os.name == "nt" and not mode and Path(path).is_dir():
                    return real_open(sync_anchor, os.O_RDWR)
                return real_open(path, flags, *mode)

            upload = root / "checkpoint.uploading"
            final = root / "checkpoint.pt"
            receipt = root / "receipt.json"
            upload.write_bytes(payload)

            def checkpoint_race(source, target, **kwargs):
                if Path(target) == final:
                    final.write_bytes(b"checkpoint sentinel")
                return real_link(source, target, **kwargs)

            argv = [
                "remote-copy",
                str(root),
                "run",
                "/source/checkpoint.pt",
                expected,
                str(len(payload)),
                str(upload),
                str(final),
                str(receipt),
            ]
            with patch.object(sys, "argv", argv), patch.object(
                post.os, "link", side_effect=checkpoint_race
            ), patch.object(post.os, "open", side_effect=remote_open):
                with self.assertRaises(FileExistsError):
                    exec(post.REMOTE_FINALIZE_COPY, {})
            self.assertEqual(final.read_bytes(), b"checkpoint sentinel")
            self.assertFalse(receipt.exists())

            upload = root / "checkpoint-2.uploading"
            final = root / "checkpoint-2.pt"
            receipt = root / "receipt-2.json"
            upload.write_bytes(payload)
            link_count = 0

            def receipt_race(source, target, **kwargs):
                nonlocal link_count
                link_count += 1
                if link_count == 2:
                    receipt.write_bytes(b"receipt sentinel")
                return real_link(source, target, **kwargs)

            argv[6:] = [str(upload), str(final), str(receipt)]
            with patch.object(sys, "argv", argv), patch.object(
                post.os, "link", side_effect=receipt_race
            ), patch.object(post.os, "open", side_effect=remote_open):
                with self.assertRaises(FileExistsError):
                    exec(post.REMOTE_FINALIZE_COPY, {})
            self.assertEqual(final.read_bytes(), payload)
            self.assertEqual(receipt.read_bytes(), b"receipt sentinel")

    def test_plan_binds_stage2_replay_context_and_rejects_checkpoint_tamper(self) -> None:
        run_id = "navigation-rlt-test"
        with tempfile.TemporaryDirectory() as temporary:
            runs_root = Path(temporary)
            run_root = runs_root / run_id
            stage_root = run_root / "navigation_rlt_stage2"
            training_root = stage_root / "training"
            checkpoint = training_root / "checkpoints" / "latest" / "navigation_rlt_policy.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"stage2 checkpoint")
            checkpoint_sha = _sha(checkpoint)
            context_sha = "8" * 64
            source_fingerprint = "9" * 64
            shard = _metadata_npz(
                run_root / "replay" / "shard.npz",
                {
                    "prefix_feature_source_sha256": context_sha,
                    "source_fingerprint": source_fingerprint,
                    "representation_kind": "frozen_z_rl",
                },
            )
            shard_row = {
                "path": str(shard),
                "sha256": _sha(shard),
                "episode_id": "episode",
                "reset_token": "reset",
                "size_bytes": shard.stat().st_size,
                "row_count": 1,
            }
            replay = _write(
                run_root / "replay" / "manifest.json",
                {
                    "schema_name": "navigation_rlt_replay_manifest",
                    "schema_version": 3,
                    "run_id": run_id,
                    "split_label": "train_update",
                    "validation_split_reads": 0,
                    "representation_kind": "frozen_z_rl",
                    "expected_episode_count": 256,
                    "completed_episode_count": 256,
                    "source_fingerprint": source_fingerprint,
                    "shards": [dict(shard_row) for _ in range(256)],
                },
            )
            training_run_manifest = _write(
                stage_root / "run_manifest.json",
                {
                    "schema_name": "navigation_rlt_training_run_manifest",
                    "schema_version": 1,
                    "run_id": run_id,
                    "phase": "stage2",
                    "replay": {"manifest_sha256": _sha(replay)},
                },
            )
            training_result = _write(
                training_root / "training_result.json",
                {
                    "schema_name": post.TRAINING_RESULT_SCHEMA,
                    "schema_version": 1,
                    "status": "completed",
                    "phase": "stage2",
                    "run_id": run_id,
                    "final_checkpoint": {
                        "path": str(checkpoint),
                        "sha256": checkpoint_sha,
                        "metadata": {
                            "stage1_frozen": True,
                            "extra": {
                                "training_phase": "stage2",
                                "run_manifest_sha256": _sha(training_run_manifest),
                                "replay_manifest_sha256": _sha(replay),
                            },
                        },
                    },
                    "run_manifest": {
                        "path": str(training_run_manifest),
                        "sha256": _sha(training_run_manifest),
                    },
                    "replay_manifest": {
                        "path": str(replay),
                        "sha256": _sha(replay),
                    },
                    "independent_restore": {
                        "status": "passed",
                        "checkpoint_sha256": checkpoint_sha,
                    },
                },
            )
            verification = _write(
                stage_root / "verification.json",
                {
                    "schema_name": post.VERIFICATION_SCHEMA,
                    "schema_version": 1,
                    "status": "passed",
                    "phase": "stage2",
                    "world_size": 8,
                    "unique_gpu_uuid_count": 8,
                    "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha},
                    "training_result": {
                        "path": str(training_result),
                        "sha256": _sha(training_result),
                    },
                },
            )
            marker = _write(
                stage_root / "STAGE.json",
                {
                    "schema_name": post.STAGE_SCHEMA,
                    "schema_version": 1,
                    "status": "completed",
                    "phase": "stage2",
                    "exit_code": 0,
                    "run_root": str(run_root),
                    "output": {
                        "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha}
                    },
                    "verification": {"path": str(verification), "sha256": _sha(verification)},
                    "replay_manifest": {"path": str(replay), "sha256": _sha(replay)},
                },
            )

            split_rows = [
                {
                    "episode_order": index,
                    "episode_id": str(index),
                    "scene": f"scene-{index // 4}",
                    "instruction_text_sha256": f"{index + 1:064x}",
                }
                for index in range(32)
            ]
            split = _write(
                run_root / "provenance" / "split_manifest.json",
                {
                    "upstream_split_read": "train",
                    "validation_split_reads": 0,
                    "splits": {
                        "train_development": {
                            "episode_count": 32,
                            "dataset_sha256": "1" * 64,
                            "gt_sha256": "2" * 64,
                            "episodes": split_rows,
                        }
                    },
                },
            )
            eval_plan = _write(
                run_root / "post-training" / "eval-plan.json",
                {
                    "schema_name": post.EVAL_PLAN_SCHEMA,
                    "schema_version": 1,
                    "run_id": run_id,
                    "split_label": "train_development",
                    "validation_split_reads": 0,
                    "episode_count": 32,
                    "worker_count": 1,
                    "split_manifest": {"sha256": _sha(split)},
                    "workers": [
                        {"worker_id": 0, "episode_count": 32, "episodes": split_rows}
                    ],
                },
            )
            lightnav = {
                "original_release_revision": "1" * 40,
                "context_patch_sha256": "3" * 64,
                "weight_sha256": "4" * 64,
                "tokenizer_sha256": "5" * 64,
                "processor_config_sha256": "6" * 64,
                "history_config_sha256": "7" * 64,
            }
            collection = _write(
                run_root / "provenance" / "collection.json",
                {
                    "schema_name": post.COLLECTION_SCHEMA,
                    "schema_version": 1,
                    "run_id": run_id,
                    "boundaries": {
                        "upstream_split_read": "train",
                        "validation_split_reads": 0,
                        "lightnav_frozen": True,
                        "weight_origin": "original_release",
                        "ttt_or_secondary_post_training": False,
                    },
                    "protocol": {
                        "max_physical_steps": 160,
                        "matched_eval_arms": list(post.ARMS),
                    },
                    "data": {
                        "split_manifest_sha256": _sha(split),
                        "development_episode_count": 32,
                    },
                    "behavior": {"checkpoint_sha256": "b" * 64},
                    "code": {"git_commit": "2" * 40},
                    "lightnav": lightnav,
                },
            )
            inventory = _write(
                run_root / "post-training" / "evaluation_code_inventory.json",
                {
                    "schema_name": post.EVAL_INVENTORY_SCHEMA,
                    "schema_version": 1,
                    "navigation_code_commit": "a" * 40,
                    "files": {
                        name: _sha(post._local_evaluation_source(name))
                        for name in post.REQUIRED_EVAL_FILES
                    },
                },
            )
            code_verification = _write(
                run_root / "provenance" / "code_verification.json",
                {
                    "schema_name": "navigation_rlt_a800_code_verification",
                    "schema_version": 1,
                },
            )
            training_protocol = _write(
                run_root / "provenance" / "training_protocol.json",
                {
                    "schema_name": "navigation_rlt_training_protocol",
                    "schema_version": 1,
                    "run_id": run_id,
                },
            )
            expected_model_sha = post._object_sha256(post._model_identity(_load(collection)))
            output = run_root / "post-training" / "plan.json"
            args = argparse.Namespace(
                runs_root=runs_root,
                run_root=run_root,
                run_id=run_id,
                stage2_marker=marker,
                expected_stage2_marker_sha256=_sha(marker),
                collection_manifest=collection,
                expected_collection_manifest_sha256=_sha(collection),
                split_manifest=split,
                expected_split_manifest_sha256=_sha(split),
                eval_plan=eval_plan,
                expected_eval_plan_sha256=_sha(eval_plan),
                evaluation_code_inventory=inventory,
                expected_evaluation_code_inventory_sha256=_sha(inventory),
                training_code_verification=code_verification,
                expected_training_code_verification_sha256=_sha(code_verification),
                training_protocol=training_protocol,
                expected_training_protocol_sha256=_sha(training_protocol),
                expected_a3_checkpoint_sha256="b" * 64,
                expected_protocol_sha256="c" * 64,
                expected_model_sha256=expected_model_sha,
                ebk_host="ebk-test",
                ebk_run_root=f"/srv/navigation/runs/{run_id}",
                obs_root=run_root / "obs",
                output=output,
            )

            def repair_lineage(commit: str) -> dict[str, object]:
                return {
                    "stage2_repair_code_verification": {
                        "path": str(run_root / "provenance" / "repair-code.json"),
                        "sha256": "d" * 64,
                    },
                    "stage2_repair_protocol": {
                        "path": str(run_root / "provenance" / "repair-protocol.json"),
                        "sha256": "e" * 64,
                    },
                    "training_source_identities": {
                        "stage1": {"code_commit": "2" * 40},
                        "stage2_repair": {"code_commit": commit},
                    },
                }

            matching_lineage = repair_lineage("a" * 40)
            args.output = run_root / "post-training" / "repair-plan.json"
            with patch.object(
                post, "_stage2_repair_lineage", return_value=matching_lineage
            ):
                repair_plan = post.build_plan(args)
            self.assertEqual(
                repair_plan["evaluation_code"]["commit"],
                repair_plan["training_source_identities"]["stage2_repair"][
                    "code_commit"
                ],
            )

            args.output = run_root / "post-training" / "wrong-runtime-plan.json"
            with patch.object(
                post,
                "_stage2_repair_lineage",
                return_value=repair_lineage("f" * 40),
            ), self.assertRaisesRegex(
                post.PostTrainingError,
                "evaluation code commit differs from Stage2 repair code commit",
            ):
                post.build_plan(args)

            repair_plan["stage2"]["repair_contract"] = {"test_only": True}
            with patch.object(
                post, "_stage2_evidence", return_value=repair_plan["stage2"]
            ), patch.object(
                post, "_stage2_repair_lineage", return_value=matching_lineage
            ):
                self.assertEqual(
                    post._revalidate_plan(repair_plan), (run_root, run_id)
                )

            mismatched_plan = json.loads(json.dumps(repair_plan))
            mismatched_lineage = repair_lineage("f" * 40)
            mismatched_plan.update(mismatched_lineage)
            with patch.object(
                post, "_stage2_evidence", return_value=mismatched_plan["stage2"]
            ), patch.object(
                post, "_stage2_repair_lineage", return_value=mismatched_lineage
            ), self.assertRaisesRegex(
                post.PostTrainingError,
                "evaluation code commit differs from Stage2 repair code commit",
            ):
                post._revalidate_plan(mismatched_plan)

            args.output = output
            plan = post.build_plan(args)
            self.assertEqual(plan["stage2"]["context_source_sha256"], context_sha)
            self.assertEqual(plan["contract"]["arm_order"], list(post.ARMS))
            self.assertEqual(plan["contract"]["max_physical_steps"], 160)
            checkpoint.write_bytes(b"tampered")
            with self.assertRaises(post.PostTrainingError):
                post._revalidate_plan(plan)

            checkpoint.write_bytes(b"stage2 checkpoint")
            tampered_training = _load(training_result)
            tampered_training["final_checkpoint"]["metadata"]["extra"][
                "replay_manifest_sha256"
            ] = "0" * 64
            _write(training_result, tampered_training)
            verification_value = _load(verification)
            verification_value["training_result"]["sha256"] = _sha(training_result)
            _write(verification, verification_value)
            marker_value = _load(marker)
            marker_value["verification"]["sha256"] = _sha(verification)
            _write(marker, marker_value)
            args.expected_stage2_marker_sha256 = _sha(marker)
            args.output = run_root / "post-training" / "tampered-plan.json"
            with self.assertRaisesRegex(
                post.PostTrainingError, "checkpoint replay binding"
            ):
                post.build_plan(args)

    def test_stage2_repair_lineage_binds_two_sources_and_recover_false(self) -> None:
        run_id = "navigation-rlt-repair"
        collection_commit = "1" * 40
        stage1_commit = "2" * 40
        repair_commit = "3" * 40
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary) / run_id
            stage1_root = run_root / "code"
            repair_root = run_root / "stage2-repair-code"
            stage1_root.mkdir(parents=True)
            repair_root.mkdir()

            def verification(
                name: str,
                *,
                commit: str,
                supersedes: str,
                code_root: Path,
            ) -> Path:
                source_rows: list[dict[str, object]] = []
                if name == "stage2-repair":
                    for filename in post.TRAINING_SOURCE_FILES:
                        source = code_root / filename
                        source.write_bytes(f"{name} source {filename}".encode("utf-8"))
                        source_rows.append(
                            {
                                "path": filename,
                                "sha256": _sha(source),
                                "size_bytes": source.stat().st_size,
                            }
                        )
                code_archive = run_root / "provenance" / f"{name}.tar.gz"
                code_archive.parent.mkdir(parents=True, exist_ok=True)
                code_archive.write_bytes(name.encode("utf-8"))
                inventory = _write(
                    run_root / "provenance" / f"{name}-inventory.json",
                    {
                        "schema_name": post.TRAINING_INVENTORY_SCHEMA,
                        "schema_version": 1,
                        "navigation_code_commit": commit,
                        "archive_name": code_archive.name,
                        "archive_sha256": _sha(code_archive),
                        "archive_size_bytes": code_archive.stat().st_size,
                        "file_count": len(source_rows),
                        "files": source_rows,
                    },
                )
                return _write(
                    run_root / "provenance" / f"{name}-verification.json",
                    {
                        "schema_name": post.TRAINING_CODE_VERIFICATION_SCHEMA,
                        "schema_version": 1,
                        "status": "passed",
                        "navigation_code_commit": commit,
                        "supersedes_navigation_code_commit": supersedes,
                        "code_root": str(code_root),
                        "file_count": len(source_rows),
                        "inventory_path": str(inventory),
                        "inventory_sha256": _sha(inventory),
                        "archive_path": str(code_archive),
                        "archive_sha256": _sha(code_archive),
                    },
                )

            stage1_verification = verification(
                "stage1",
                commit=stage1_commit,
                supersedes=collection_commit,
                code_root=stage1_root,
            )
            repair_verification = verification(
                "stage2-repair",
                commit=repair_commit,
                supersedes=stage1_commit,
                code_root=repair_root,
            )
            repair_claims = {
                claim: _sha(repair_root / filename)
                for claim, filename in post.TRAINING_SOURCE_FILES_BY_CLAIM.items()
                if claim in post.STAGE2_REPAIR_REQUIRED_SOURCE_CLAIMS
            }
            base_protocol = _write(
                run_root / "provenance" / "training-protocol.json",
                {
                    "schema_name": post.TRAINING_PROTOCOL_SCHEMA,
                    "schema_version": 1,
                    "run_id": run_id,
                    "navigation_training_code_commit": stage1_commit,
                },
            )
            additions = {
                "objective_schema": post.STAGE2_REPAIR_OBJECTIVE_SCHEMA,
                "cql_alpha": 1.0,
                "cql_temperature": 1.0,
                "minimum_behavior_gate_count": 1,
            }
            support = {
                "eligibility_schema": post.STAGE2_ELIGIBILITY_SCHEMA,
                "gate_names": list(post.STAGE2_GATE_NAMES),
                "gate_counts": [9, 5, 1, 0],
                "gate_support_mask": [True, True, True, False],
                "minimum_behavior_gate_count": 1,
            }
            stage1_checkpoint_sha = "4" * 64
            stage2_replay_sha = "5" * 64
            stage2_run_sha = "6" * 64
            repair_protocol_value = {
                "schema_name": post.STAGE2_REPAIR_PROTOCOL_SCHEMA,
                "schema_version": 1,
                "run_id": run_id,
                "base_training_protocol_sha256": _sha(base_protocol),
                "stage1_code_verification_sha256": _sha(stage1_verification),
                "stage1_checkpoint_sha256": stage1_checkpoint_sha,
                "stage2_repair_code_verification_sha256": _sha(
                    repair_verification
                ),
                "stage2_replay_manifest_sha256": stage2_replay_sha,
                "stage2_run_manifest_sha256": stage2_run_sha,
                "navigation_training_code_commit": repair_commit,
                "validation_split_reads": 0,
                "stage2_training_additions": additions,
                "stage2_support": support,
            }
            repair_protocol = _write(
                run_root / "provenance" / "stage2-repair-protocol.json",
                repair_protocol_value,
            )
            args = argparse.Namespace(
                stage2_repair_code_verification=repair_verification,
                expected_stage2_repair_code_verification_sha256=_sha(
                    repair_verification
                ),
                stage2_repair_protocol=repair_protocol,
                expected_stage2_repair_protocol_sha256=_sha(repair_protocol),
            )
            lineage = post._stage2_repair_lineage(
                args=args,
                stage2={
                    "repair_contract": {
                        "training_additions": additions,
                        "stage2_support": support,
                        "stage1_checkpoint_sha256": stage1_checkpoint_sha,
                        "validation_split_reads": 0,
                        "source_claims": repair_claims,
                    },
                    "replay_manifest": {"sha256": stage2_replay_sha},
                    "training_run_manifest": {"sha256": stage2_run_sha},
                },
                collection={"code": {"git_commit": collection_commit}},
                run_root=run_root,
                run_id=run_id,
                base_verification_ref={
                    "path": str(stage1_verification),
                    "sha256": _sha(stage1_verification),
                },
                base_protocol_ref={
                    "path": str(base_protocol),
                    "sha256": _sha(base_protocol),
                },
            )
            self.assertEqual(
                set(lineage["training_source_identities"]),
                {"stage1", "stage2_repair"},
            )
            self.assertEqual(
                lineage["training_source_identities"]["stage2_repair"][
                    "code_commit"
                ],
                repair_commit,
            )

            repair_protocol_value["stage2_run_manifest_sha256"] = "0" * 64
            _write(repair_protocol, repair_protocol_value)
            args.expected_stage2_repair_protocol_sha256 = _sha(repair_protocol)
            with self.assertRaisesRegex(
                post.PostTrainingError, "protocol/actual identity"
            ):
                post._stage2_repair_lineage(
                    args=args,
                    stage2={
                        "repair_contract": {
                            "training_additions": additions,
                            "stage2_support": support,
                            "stage1_checkpoint_sha256": stage1_checkpoint_sha,
                            "validation_split_reads": 0,
                            "source_claims": repair_claims,
                        },
                        "replay_manifest": {"sha256": stage2_replay_sha},
                        "training_run_manifest": {"sha256": stage2_run_sha},
                    },
                    collection={"code": {"git_commit": collection_commit}},
                    run_root=run_root,
                    run_id=run_id,
                    base_verification_ref={
                        "path": str(stage1_verification),
                        "sha256": _sha(stage1_verification),
                    },
                    base_protocol_ref={
                        "path": str(base_protocol),
                        "sha256": _sha(base_protocol),
                    },
                )
            repair_protocol_value["stage2_run_manifest_sha256"] = stage2_run_sha
            _write(repair_protocol, repair_protocol_value)
            args.expected_stage2_repair_protocol_sha256 = _sha(repair_protocol)

            tampered_source = repair_root / "navigation_rlt_policy.py"
            original_source = tampered_source.read_bytes()
            tampered_source.write_bytes(b"tampered repair policy")
            with self.assertRaisesRegex(post.PostTrainingError, "live source differs"):
                post._stage2_repair_lineage(
                    args=args,
                    stage2={
                        "repair_contract": {
                            "training_additions": additions,
                            "stage2_support": support,
                            "stage1_checkpoint_sha256": stage1_checkpoint_sha,
                            "validation_split_reads": 0,
                            "source_claims": repair_claims,
                        },
                        "replay_manifest": {"sha256": stage2_replay_sha},
                        "training_run_manifest": {"sha256": stage2_run_sha},
                    },
                    collection={"code": {"git_commit": collection_commit}},
                    run_root=run_root,
                    run_id=run_id,
                    base_verification_ref={
                        "path": str(stage1_verification),
                        "sha256": _sha(stage1_verification),
                    },
                    base_protocol_ref={
                        "path": str(base_protocol),
                        "sha256": _sha(base_protocol),
                    },
                )
            tampered_source.write_bytes(original_source)

            repair_protocol_value["stage2_support"] = {
                **support,
                "gate_support_mask": [True, True, True, True],
            }
            _write(repair_protocol, repair_protocol_value)
            args.expected_stage2_repair_protocol_sha256 = _sha(repair_protocol)
            with self.assertRaisesRegex(post.PostTrainingError, "support mask"):
                post._stage2_repair_lineage(
                    args=args,
                    stage2={
                        "repair_contract": {
                            "training_additions": additions,
                            "stage2_support": support,
                            "stage1_checkpoint_sha256": stage1_checkpoint_sha,
                            "validation_split_reads": 0,
                            "source_claims": repair_claims,
                        },
                        "replay_manifest": {"sha256": stage2_replay_sha},
                        "training_run_manifest": {"sha256": stage2_run_sha},
                    },
                    collection={"code": {"git_commit": collection_commit}},
                    run_root=run_root,
                    run_id=run_id,
                    base_verification_ref={
                        "path": str(stage1_verification),
                        "sha256": _sha(stage1_verification),
                    },
                    base_protocol_ref={
                        "path": str(base_protocol),
                        "sha256": _sha(base_protocol),
                    },
                )

    def test_stage2_package_source_claims_bind_completed_receipts(self) -> None:
        run_id = "navigation-rlt-package-claims"
        cases = (
            (
                "run",
                "navigation_rlt_prepare_training",
                "prepare_receipt.json",
                "artifact_sha256",
                "receipt_sha256",
                {"trainer_sha256", "policy_sha256", "replay_sha256", "preparer_sha256"},
            ),
            (
                "replay",
                "navigation_rlt_replay_promotion",
                "promotion_receipt.json",
                "output_manifest_sha256",
                "promotion_receipt_sha256",
                {"promoter_sha256", "policy_sha256", "replay_sha256"},
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary) / run_id
            for label, schema, receipt_name, manifest_field, receipt_field, claims in cases:
                package = run_root / label
                manifest = _write(package / "manifest.json", {"kind": label})
                implementation = {
                    claim: hashlib.sha256(claim.encode("utf-8")).hexdigest()
                    for claim in claims
                }
                receipt = _write(
                    package / receipt_name,
                    {
                        "status": "completed",
                        "run_id": run_id,
                        "implementation": implementation,
                    },
                )
                _write(
                    package / "COMPLETED.json",
                    {
                        "schema_name": schema,
                        "schema_version": 1,
                        "status": "completed",
                        manifest_field: _sha(manifest),
                        receipt_field: _sha(receipt),
                    },
                )
                self.assertEqual(
                    post._package_source_claims(
                        manifest,
                        _sha(manifest),
                        run_root=run_root,
                        run_id=run_id,
                        name=f"Stage2 {label} package",
                        expected_schema=schema,
                        expected_claims=claims,
                    ),
                    implementation,
                )

    def test_inventory_requires_the_complete_runtime_dependency_set(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            inventory = Path(temporary) / "inventory.json"
            files = {
                name: _sha(post._local_evaluation_source(name))
                for name in post.REQUIRED_EVAL_FILES
            }
            files.pop("navrl/stack/returns.py")
            _write(
                inventory,
                {
                    "schema_name": post.EVAL_INVENTORY_SCHEMA,
                    "schema_version": 1,
                    "navigation_code_commit": "a" * 40,
                    "files": files,
                },
            )
            with self.assertRaisesRegex(post.PostTrainingError, "schema/files"):
                post._evaluation_inventory_evidence(
                    inventory, expected_sha256=_sha(inventory)
                )

    def test_scp_and_downstream_commands_use_the_exact_argv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.bin"
            source.write_bytes(b"payload")
            with patch.object(
                post,
                "_run",
                return_value=subprocess.CompletedProcess([], 0, "", ""),
            ) as run:
                post._scp_to(
                    "ebk-test",
                    source,
                    "/remote/upload.bin",
                    timeout=1,
                    name="upload",
                )
                self.assertEqual(
                    run.call_args.args[0],
                    [
                        "scp",
                        "-O",
                        "--",
                        str(source),
                        "ebk-test:/remote/upload.bin",
                    ],
                )

            copied = root / "returned.bin"

            def copy_back(command, **_kwargs):
                Path(command[-1]).write_bytes(b"payload")
                return subprocess.CompletedProcess(command, 0, "", "")

            with patch.object(post, "_run", side_effect=copy_back) as run:
                post._scp_from(
                    "ebk-test",
                    "/remote/result.bin",
                    copied,
                    expected_sha256=_sha(source),
                    expected_size_bytes=7,
                    timeout=1,
                    name="download",
                )
                self.assertEqual(run.call_args.args[0][:3], ["scp", "-O", "--"])

            matched_path = root / "post-training" / "matched_comparison.json"
            markdown_path = root / "post-training" / "matched_comparison.md"
            plan = {
                "run_id": "navigation-rlt-test",
                "a800_run_root": str(root),
                "outputs": {
                    key: str(path) for key, path in post._fixed_outputs(root).items()
                },
                "collection_manifest": {"path": str(root / "collection.json"), "sha256": "a" * 64},
                "split_manifest": {
                    "path": str(root / "split.json"),
                    "sha256": "b" * 64,
                },
                "evaluation_plan": {
                    "path": str(root / "eval-plan.json"),
                    "sha256": "c" * 64,
                },
                "evaluation_code": {
                    "inventory_path": str(root / "inventory.json"),
                    "inventory_sha256": "d" * 64,
                },
                "contract": {
                    "expected_protocol_sha256": "e" * 64,
                    "expected_model_sha256": "f" * 64,
                    "a3_checkpoint_sha256": "1" * 64,
                    "model_identity": {"lightnav_weight_sha256": "2" * 64},
                },
                "stage2": {"checkpoint": {"sha256": "3" * 64}},
                "training_code_verification": {"path": str(root / "code.json"), "sha256": "4" * 64},
                "training_protocol": {"path": str(root / "protocol.json"), "sha256": "5" * 64},
                "stage2_repair_code_verification": {
                    "path": str(root / "repair-code.json"),
                    "sha256": "8" * 64,
                },
                "stage2_repair_protocol": {
                    "path": str(root / "repair-protocol.json"),
                    "sha256": "9" * 64,
                },
                "obs_root": str(root / "obs"),
            }
            handoff = {
                "arms": [
                    {"arm": arm, "evaluation": {"path": str(root / f"{arm}.json")}}
                    for arm in post.ARMS
                ]
            }
            commands: list[list[str]] = []

            def fake_run(command, **_kwargs):
                command = list(command)
                commands.append(command)
                if command[1].endswith("navigation_rlt_compare.py"):
                    _write(
                        matched_path,
                        {
                            "schema_name": post.MATCHED_SCHEMA,
                            "status": "completed",
                            "run_id": plan["run_id"],
                            "collection_manifest_sha256": "a" * 64,
                            "evaluation_code_inventory_sha256": "d" * 64,
                            "navigation_rlt_checkpoint_sha256": "3" * 64,
                        },
                    )
                    markdown_path.write_text("ok\n", encoding="utf-8")
                return subprocess.CompletedProcess(command, 0, "", "")

            with patch.object(post, "_run", side_effect=fake_run):
                (
                    comparison,
                    comparison_sha,
                    matched_markdown,
                    matched_markdown_sha,
                ) = post._invoke_compare(plan, root, handoff)
                post._invoke_archive(
                    plan,
                    root / "post-training-plan.json",
                    "0" * 64,
                    comparison,
                    comparison_sha,
                    matched_markdown,
                    matched_markdown_sha,
                    root / "copy.json",
                    "6" * 64,
                    root / "handoff.json",
                    "7" * 64,
                )
            compare_argv, archive_argv = commands
            self.assertIn("--native", compare_argv)
            self.assertIn("--a3", compare_argv)
            self.assertIn("--navigation-rlt", compare_argv)
            self.assertNotIn("--native-result", compare_argv)
            self.assertEqual(
                archive_argv[archive_argv.index("--run-root") + 1], str(root)
            )
            self.assertEqual(
                archive_argv[
                    archive_argv.index("--stage2-repair-code-verification") + 1
                ],
                str(root / "repair-code.json"),
            )
            self.assertEqual(
                archive_argv[archive_argv.index("--stage2-repair-protocol") + 1],
                str(root / "repair-protocol.json"),
            )

            remote_plan = {
                "run_id": "navigation-rlt-test",
                "remote": {
                    "host": "ebk-test",
                    "wrapper": "/remote/run_navigation_rlt_ebk_eval.sh",
                    "run_root": "/remote/navigation-rlt-test",
                    "a3_checkpoint": "/remote/a3.npz",
                    "rlt_checkpoint": "/remote/rlt.pt",
                    "evaluation_root": "/remote/evaluations",
                },
                "contract": {"a3_checkpoint_sha256": "8" * 64},
                "collection_manifest": {"sha256": "9" * 64},
                "split_manifest": {"sha256": "a" * 64},
                "evaluation_plan": {"sha256": "b" * 64},
                "evaluation_code": {"inventory_sha256": "c" * 64},
                "stage2": {
                    "checkpoint": {"sha256": "d" * 64},
                    "context_source_sha256": "e" * 64,
                },
            }
            with patch.object(
                post,
                "_run",
                return_value=subprocess.CompletedProcess([], 0, "", ""),
            ) as run, patch.object(
                post, "_ssh_json", return_value={"status": "verified"}
            ):
                post._run_remote_arm(remote_plan, "native_lightnav")
            remote_argv = shlex.split(run.call_args.args[0][3])
            self.assertEqual(remote_argv[:2], ["bash", remote_plan["remote"]["wrapper"]])

    def test_evaluation_deployment_contains_every_hash_bound_runtime_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "navigation-rlt-test"
            run_root.mkdir()
            eval_plan = _write(run_root / "eval-plan.json", {"plan": True})
            inventory = _write(
                run_root / "inventory.json",
                {
                    "schema_name": post.EVAL_INVENTORY_SCHEMA,
                    "schema_version": 1,
                    "navigation_code_commit": "a" * 40,
                    "files": {
                        name: _sha(post._local_evaluation_source(name))
                        for name in post.REQUIRED_EVAL_FILES
                    },
                },
            )
            remote_root = "/remote/navigation-rlt-test"
            plan = {
                "run_id": "navigation-rlt-test",
                "outputs": {
                    key: str(path)
                    for key, path in post._fixed_outputs(run_root).items()
                },
                "remote": {
                    "host": "ebk-test",
                    "run_root": remote_root,
                    "evaluation_deployment_upload": remote_root + "/.eval-runtime.zip.uploading",
                    "evaluation_deployment_root": remote_root + "/eval-runtime",
                },
                "evaluation_plan": {"path": str(eval_plan), "sha256": _sha(eval_plan)},
                "evaluation_code": {
                    "inventory_path": str(inventory),
                    "inventory_sha256": _sha(inventory),
                    "files": _load(inventory)["files"],
                },
            }
            observed: dict[str, object] = {}

            def capture_upload(_host, source, _destination, **_kwargs):
                observed["sha256"] = _sha(source)
                observed["size"] = source.stat().st_size
                with zipfile.ZipFile(source, "r") as archive:
                    observed["names"] = set(archive.namelist())

            def ssh_response(_host, _code, _args, *, name, **_kwargs):
                if name.endswith("precheck"):
                    return {"status": "ready"}
                return {
                    "status": "verified",
                    "archive_sha256": observed["sha256"],
                    "evaluation_plan_sha256": _sha(eval_plan),
                    "evaluation_code_inventory_sha256": _sha(inventory),
                    "file_count": len(post.REQUIRED_EVAL_FILES),
                    "deployment_root": remote_root + "/eval-runtime",
                }

            with patch.object(post, "_scp_to", side_effect=capture_upload), patch.object(
                post, "_ssh_json", side_effect=ssh_response
            ):
                result = post._deploy_evaluation_runtime(plan, run_root)
            self.assertEqual(result["status"], "verified")
            self.assertEqual(
                observed["names"],
                {
                    "eval-plan.json",
                    "eval-code/evaluation_code_inventory.json",
                    *(f"eval-code/{name}" for name in post.REQUIRED_EVAL_FILES),
                },
            )
            persisted = run_root / "post-training" / "relay-runtime.zip"
            with patch.object(post, "_ssh_json") as ssh, patch.object(
                post, "_scp_to"
            ) as scp:
                persisted_ref = post._build_evaluation_runtime_archive(
                    plan, run_root, persisted
                )
            ssh.assert_not_called()
            scp.assert_not_called()
            self.assertEqual(persisted_ref["sha256"], _sha(persisted))
            with zipfile.ZipFile(persisted, "r") as archive:
                self.assertEqual(set(archive.namelist()), observed["names"])
            with self.assertRaises(FileExistsError):
                post._build_evaluation_runtime_archive(plan, run_root, persisted)

            reviewed_plan = _write(run_root / "reviewed-plan.json", {"reviewed": True})
            reviewed_sha = _sha(reviewed_plan)
            exported = run_root / "post-training" / "cli-runtime.zip"
            receipt = run_root / "post-training" / "cli-runtime.json"
            cli = [
                "export-runtime",
                "--plan",
                str(reviewed_plan),
                "--expected-plan-sha256",
                reviewed_sha,
                "--output",
                str(exported),
                "--receipt",
                str(receipt),
            ]
            with patch.object(
                post, "_load_plan", return_value=(reviewed_plan, plan, reviewed_sha)
            ), patch.object(
                post, "_revalidate_plan", return_value=(run_root, plan["run_id"])
            ) as revalidate, patch("builtins.print"):
                post.main(cli)
            revalidate.assert_called_once_with(plan)
            receipt_value = _load(receipt)
            self.assertEqual(receipt_value["archive"]["sha256"], _sha(exported))
            self.assertEqual(receipt_value["post_training_plan"]["sha256"], reviewed_sha)

            original_eval_plan = eval_plan.read_bytes()
            eval_plan.write_bytes(b'{"tampered":true}\n')
            with patch.object(
                post, "_load_plan", return_value=(reviewed_plan, plan, reviewed_sha)
            ), patch.object(
                post, "_revalidate_plan", return_value=(run_root, plan["run_id"])
            ) as revalidate, patch("builtins.print"):
                with self.assertRaisesRegex(post.PostTrainingError, "evaluation plan SHA-256"):
                    post.main(
                        cli[:-4]
                        + [
                            "--output",
                            str(run_root / "post-training" / "bad-plan.zip"),
                            "--receipt",
                            str(run_root / "post-training" / "bad-plan.json"),
                        ]
                    )
            revalidate.assert_called_once_with(plan)
            eval_plan.write_bytes(original_eval_plan)

            inventory.write_text('{"tampered":true}\n', encoding="utf-8")
            with patch.object(
                post, "_load_plan", return_value=(reviewed_plan, plan, reviewed_sha)
            ), patch.object(
                post, "_revalidate_plan", return_value=(run_root, plan["run_id"])
            ) as revalidate, patch("builtins.print"):
                with self.assertRaisesRegex(
                    post.PostTrainingError, "evaluation code inventory SHA-256"
                ):
                    post.main(
                        cli[:-4]
                        + [
                            "--output",
                            str(run_root / "post-training" / "bad-inventory.zip"),
                            "--receipt",
                            str(run_root / "post-training" / "bad-inventory.json"),
                        ]
                    )
            revalidate.assert_called_once_with(plan)

    def test_ebk_wrapper_uses_formal_build_paths_and_live_asset_hashes(self) -> None:
        wrapper = (MODULE_DIR / "run_navigation_rlt_ebk_eval.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('SPLIT_MANIFEST="$RUN_ROOT/data/build/split_manifest.json"', wrapper)
        self.assertIn('DATA_ROOT="$RUN_ROOT/data/build/train_development"', wrapper)
        self.assertIn('FEATURE_ROOT="$FEATURE_PARENT/$BEHAVIOR"', wrapper)
        for variable in (
            "MODEL_WEIGHT",
            "MODEL_TOKENIZER",
            "MODEL_PROCESSOR",
            "MODEL_HISTORY_CONFIG",
        ):
            self.assertIn(f'sha256sum "${variable}"', wrapper)

    def test_ebk_wrapper_runtime_outputs_are_create_only(self) -> None:
        wrapper = (MODULE_DIR / "run_navigation_rlt_ebk_eval.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("umask 077", wrapper)
        self.assertIn("set -o noclobber", wrapper)
        self.assertIn(
            '[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]]', wrapper
        )
        self.assertGreaterEqual(
            wrapper.count('[[ ! -e "$path" && ! -L "$path" ]]'), 2
        )
        self.assertIn(
            'for path in "$RUN_ROOT/evaluations" "$STATUS_ROOT" "$LOG_ROOT" "$FEATURE_PARENT"; do',
            wrapper,
        )
        self.assertIn(
            '[[ -d "$path" && ! -L "$path" ]] || die "created evaluation parent',
            wrapper,
        )
        for target in (
            "LIGHTNAV_LOG",
            "HABITAT_LOG",
            "LIGHTNAV_PID_FILE",
            "HABITAT_PID_FILE",
            "EVAL_LOG",
        ):
            self.assertIn(f'>"${target}"', wrapper)
        self.assertNotIn(">|", wrapper)

    def test_bash_noclobber_preserves_existing_sentinel(self) -> None:
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash is unavailable")
        target = f"/tmp/navigation-rlt-noclobber-{os.getpid()}-{id(self)}"
        link = target + "-link"
        referent = target + "-referent"
        quoted_target = shlex.quote(target)
        quoted_link = shlex.quote(link)
        quoted_referent = shlex.quote(referent)
        script = "\n".join(
            (
                "set -euo pipefail",
                "umask 077",
                f"rm -f -- {quoted_target} {quoted_link} {quoted_referent}",
                f"trap 'rm -f -- {quoted_target} {quoted_link} {quoted_referent}' EXIT",
                f"printf sentinel > {quoted_target}",
                "set -o noclobber",
                f"if printf replacement > {quoted_target}; then exit 10; fi",
                f"ln -s {quoted_referent} {quoted_link}",
                f"if printf replacement > {quoted_link}; then exit 11; fi",
                f"grep -qx sentinel {quoted_target}",
                f"test -L {quoted_link}",
                f"test ! -e {quoted_referent}",
            )
        )
        completed = subprocess.run(
            [bash, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            completed.stderr.decode("utf-8", errors="replace"),
        )


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
