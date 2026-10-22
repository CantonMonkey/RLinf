"""Bounded train-only matched Feature-v2 A/B for the P8 StreamVLN successor.

This is a single-run result-to-claim experiment.  It compares the accepted
7173-D mean-pooled feature with the authorized 3604-D contextual/depth feature
on paired states, labels, head capacity, optimizer schedule, evaluation
episodes, and conservative KEEP gating.  It never reads validation splits,
promotes a checkpoint, runs PPO, enters G9, or retunes from the result.
"""

from __future__ import annotations

import argparse
from collections import Counter
import importlib
import inspect
import json
import math
from pathlib import Path
import random
import subprocess
import sys
from typing import Callable, Mapping, Protocol, Sequence


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import run_streamvln_p8_successor_bc_readiness as readiness
import run_streamvln_p8_successor_causal_probe as causal


SCHEMA_NAME = "streamvln_p8_successor_feature_v2_ab"
SCHEMA_VERSION = 1
MANIFEST_SCHEMA_NAME = "streamvln_p8_successor_feature_v2_ab_manifest"
TRAIN_SPLIT = "train"
ARM_V1 = "feature_v1"
ARM_V2 = "feature_v2"
ARMS = (ARM_V1, ARM_V2)
EXPECTED_EPISODES = 8
EXPECTED_TRAIN_READS = 48
EXPECTED_READINESS_MANIFEST_RAW_SHA256 = (
    "61396c099763a3c340ec66056e9fce5244baffc5e6c7ee3fb4f852b984f05ae2"
)
EXPECTED_READINESS_MANIFEST_CANONICAL_SHA256 = (
    "d340a3b9c1db5eb80465dc996f93561ecde3ec3ff2f8648fddd0367966421e80"
)
EXPECTED_CAUSAL_MANIFEST_RAW_SHA256 = (
    "a027e71d27303021813ea94913d18c37d177aef6a17277f46c29c55ebbd7f87e"
)
EXPECTED_CAUSAL_MANIFEST_CANONICAL_SHA256 = (
    "5a49481cb924a59a9ec19d92959deb51dbaf8f660fef897a4f312390c3ecd91c"
)
EXPECTED_CAUSAL_RECEIPT_FILE_SHA256 = (
    "98db17588d7ab7539b7f5834b818526ec45898e848016c878881d560e8ebf531"
)
EXPECTED_CAUSAL_RECEIPT_SHA256 = (
    "d19d8687f1f903b506a654289f49d6044616918144b632008200641480149773"
)
EXPECTED_CAUSAL_RUNNER_SHA256 = (
    "e7385827f10e9430462b0a866ec2f817d4aacab6ef5f3b148b2e79bd5d7375c8"
)
EXPECTED_SOURCE_COMMITS = {
    "habitat_sim": "f179b584bcd713c5a2a998132211e2cae881d6d1",
    "rlinf": "0505431899574619da86f551bad70b71e0ea2177",
    "streamvln": "e48f6ff7e9201d93aae003e8f64b04c00cec13bc",
}
EXPECTED_FEATURE_SCHEMA_DIGESTS = {
    1: "4135d429fe39d82ba6e3cd929dc20653d0cce0ffe95e008392fb4b37a894b16b",
    2: "6846e128e948348050924550dfa32d8050c0745b848edce6cac0d3c5be8cb9ba",
}
EXPECTED_PROTOCOL = {
    "collection": {
        "expert": {
            "behavior": "train_shortest_path_expert",
            "decision_budget_per_episode": 32,
            "episode_count": 8,
            "physical_budget_per_episode": 128,
        },
        "exact_keep": {
            "behavior": "official_exact_keep",
            "decision_budget_per_episode": 20,
            "episode_count": 8,
            "physical_budget_per_episode": 80,
            "reuse_expert_episode_identities": True,
        },
        "paired_state_identity": (
            "episode_scene_collection_role_decision_frame_topology_label"
        ),
    },
    "evaluation": {
        "teacher_forced_expert": {
            "decision_budget_per_episode": 160,
            "episode_count": 8,
            "physical_budget_per_episode": 160,
        },
        "matched_exact_keep": {
            "decision_budget_per_episode": 160,
            "episode_count": 8,
            "physical_budget_per_episode": 160,
            "passes": 1,
        },
        "learned_arm": {
            "decision_budget_per_episode": 160,
            "episode_count_per_arm": 8,
            "physical_budget_per_episode": 160,
            "deterministic": True,
            "passes_per_arm": 1,
        },
    },
    "feature_pairing": {
        "v1": (
            "mean_pooled_visual_instruction_latched_on_v2_generation_ordinal"
        ),
        "v2": (
            "generation_latched_contextual_hidden_plus_current_metric_depth"
        ),
        "temporal_state": "physical_step_minus_generation_latch_step",
        "same_topology_snapshot": True,
        "same_labels": True,
    },
    "learned_execution": {
        "gate": "apply_successor_feature_v2_keep_gate",
        "learned_stop_execution": "keep_original",
        "root_recovery_policy": "keep_original",
        "waypoint_probability_min": 0.8,
        "fresh_context_age_required": 0,
    },
    "single_model_load": True,
    "single_process": True,
    "train_reads": EXPECTED_TRAIN_READS,
}
EXPECTED_TRAINING = {
    "warmup_updates": 500,
    "final_updates": 1500,
    "updates_per_arm": 2000,
    "batch_size": 32,
    "learning_rate": 0.0001,
    "weight_decay": 0.0,
    "optimizer": {
        "type": "AdamW",
        "betas": [0.9, 0.999],
        "eps": 1e-08,
        "amsgrad": False,
    },
    "class_weights": {"method": "inverse_frequency", "computed": "once_per_phase"},
    "loss": {"route": "weighted_cross_entropy", "waypoint": {"type": "SmoothL1Loss", "beta": 1.0}},
    "same_learner_seed": True,
    "value_head_trainable": False,
    "ppo_updates": 0,
}
EXPECTED_THRESHOLDS = {
    "teacher_sample_min": {"keep_original": 8, "waypoint": 8, "stop": 4},
    "teacher_route": {
        "feature_v2_route_nonstop_bal_acc_min": 0.65,
        "feature_v2_delta_vs_v1_min": 0.10,
        "feature_v2_stop_recall_min": 0.5,
        "feature_v2_premature_stop_count_max": 1,
    },
    "teacher_waypoint": {
        "feature_v2_first_action_agreement_min": 0.65,
        "feature_v2_first_action_delta_vs_v1_min": 0.10,
        "feature_v2_bearing_abs_error_median_deg_max": 15.0,
        "feature_v2_bearing_abs_error_p90_deg_max": 45.0,
        "feature_v2_vector_l2_median_m_max": 0.25,
        "percentile_method": "nearest_rank",
    },
    "closed_loop": {
        "feature_v2_max_success_deficit_vs_keep": 1,
        "feature_v2_success_count_min": 1,
        "feature_v2_successful_stop_count_min": 1,
        "feature_v2_success_delta_vs_v1_min": 2,
        "feature_v2_root_recovery_unavailable_count_max": 0,
        "feature_v2_recover_alignment_miss_count_max": 0,
        "feature_v2_forward_nonstop_fraction_min": 0.295,
        "feature_v2_accepted_override_count_min": 4,
        "feature_v2_accepted_override_episode_count_min": 2,
        "feature_v2_successful_episode_with_accepted_override_count_min": 1,
    },
}
EXPECTED_CLAIM_BOUNDARY = {
    "bc_candidate_only": True,
    "bc_promotion": False,
    "cuda_worker_smoke": False,
    "g9": False,
    "one_shot_no_retune": True,
    "ppo": False,
    "remote_push": False,
    "result_to_claim_on_failure": True,
    "validation_split_access": False,
}
REQUIRED_IMPLEMENTATION_DEPLOYMENTS = frozenset(
    {
        "integrations/rlinf_v03/run_streamvln_p8_successor_feature_v2_ab.py",
        "integrations/rlinf_v03/run_streamvln_p8_successor_bc_readiness.py",
        "integrations/rlinf_v03/run_streamvln_p8_successor_causal_probe.py",
        "integrations/rlinf_v03/run_streamvln_p8_successor_ceiling.py",
        "rlinf/envs/__init__.py",
        "rlinf/envs/vln/__init__.py",
        "rlinf/envs/vln/vln_env.py",
        "rlinf/envs/streamvln/__init__.py",
        "rlinf/envs/streamvln/streamvln_env.py",
        "rlinf/envs/streamvln/streamvln_executor_contract.py",
        "rlinf/envs/streamvln/streamvln_chunk_ppo.py",
        "rlinf/envs/streamvln/streamvln_p8_official_frame.py",
        "rlinf/envs/streamvln/streamvln_p8_official_adapter.py",
        "rlinf/envs/streamvln/streamvln_p8_official_keep.py",
        "rlinf/envs/streamvln/streamvln_p8_successor_action.py",
        "rlinf/envs/streamvln/streamvln_p8_successor_env.py",
        "rlinf/envs/streamvln/streamvln_p8_successor_executor.py",
        "rlinf/envs/streamvln/streamvln_p8_successor_policy.py",
        "rlinf/envs/streamvln/streamvln_p8_successor_protocol.py",
        "rlinf/envs/streamvln/streamvln_p8_successor_runtime.py",
        "rlinf/envs/streamvln/streamvln_p8_topology.py",
        "rlinf/envs/streamvln/streamvln_topology.py",
        "scripts/habitat_r2r_service.py",
    }
)


ProtocolError = readiness.ProtocolError


def _strict_keys(value: Mapping, expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ProtocolError(
            f"{name} keys drifted: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def _identity(value: object, name: str) -> dict:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{name} must be an object")
    row = dict(value)
    _strict_keys(row, {"episode_id", "index", "scene"}, name)
    if not isinstance(row["episode_id"], str) or not row["episode_id"]:
        raise ProtocolError(f"{name}.episode_id must be non-empty")
    if type(row["index"]) is not int or row["index"] < 0:
        raise ProtocolError(f"{name}.index must be a nonnegative integer")
    if not isinstance(row["scene"], str) or not row["scene"]:
        raise ProtocolError(f"{name}.scene must be non-empty")
    return row


def _implementation_records(value: object) -> list[dict]:
    if not isinstance(value, list) or not value:
        raise ProtocolError("feature-v2 implementation closure must be a non-empty list")
    records = []
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise ProtocolError(f"implementation.files[{index}] must be an object")
        row = dict(raw)
        _strict_keys(
            row,
            {"bytes", "deployment_relative_path", "sha256", "source_relative_path"},
            f"implementation.files[{index}]",
        )
        for key in ("deployment_relative_path", "source_relative_path"):
            if not isinstance(row[key], str) or not row[key] or "\\" in row[key]:
                raise ProtocolError(f"implementation.files[{index}].{key} is invalid")
        readiness._strict_int(row["bytes"], f"implementation.files[{index}].bytes", minimum=1)
        readiness._sha(row["sha256"], f"implementation.files[{index}].sha256")
        records.append(row)
    deployments = {row["deployment_relative_path"] for row in records}
    if deployments != REQUIRED_IMPLEMENTATION_DEPLOYMENTS or len(deployments) != len(records):
        raise ProtocolError("feature-v2 implementation deployment closure drifted")
    return records


def validate_feature_v2_ab_manifest(
    value: object, *, readiness_manifest: Mapping, causal_manifest: Mapping
) -> dict:
    if not isinstance(value, Mapping):
        raise ProtocolError("feature-v2 A/B manifest root must be an object")
    manifest = dict(value)
    _strict_keys(
        manifest,
        {
            "architecture",
            "claim_boundary",
            "created",
            "dataset",
            "executor",
            "implementation",
            "lineage",
            "protocol",
            "runtime",
            "schema_name",
            "schema_version",
            "seeds",
            "source_contract",
            "split_contract",
            "stage",
            "thresholds",
            "training",
        },
        "feature-v2 A/B manifest",
    )
    if (
        manifest["schema_name"] != MANIFEST_SCHEMA_NAME
        or manifest["schema_version"] != 1
        or manifest["stage"] != "P8 successor train-only matched Feature-v2 A/B"
    ):
        raise ProtocolError("unsupported feature-v2 A/B manifest")
    if manifest["split_contract"] != {
        "allowed_splits": [TRAIN_SPLIT],
        "train_only": True,
        "val_seen_reads": 0,
        "val_unseen_reads": 0,
    }:
        raise ProtocolError("feature-v2 A/B split contract drifted")
    if manifest["claim_boundary"] != EXPECTED_CLAIM_BOUNDARY:
        raise ProtocolError("feature-v2 A/B claim boundary drifted")
    if manifest["protocol"] != EXPECTED_PROTOCOL:
        raise ProtocolError("feature-v2 A/B paired execution protocol drifted")
    if manifest["training"] != EXPECTED_TRAINING:
        raise ProtocolError("feature-v2 A/B training protocol drifted")
    if manifest["thresholds"] != EXPECTED_THRESHOLDS:
        raise ProtocolError("feature-v2 A/B thresholds drifted")
    expected_source = {
        "commits": EXPECTED_SOURCE_COMMITS,
        "data_source_manifest": readiness_manifest["source_contract"][
            "data_source_manifest"
        ],
        "original_streamvln_only": True,
        "ttt_or_secondary_post_training": False,
    }
    if manifest["source_contract"] != expected_source:
        raise ProtocolError("feature-v2 A/B source contract drifted")

    readiness_contract = readiness.validate_protocol_manifest(readiness_manifest)
    causal_contract = causal.validate_probe_manifest(causal_manifest, readiness_manifest)
    dataset = manifest["dataset"]
    if not isinstance(dataset, Mapping):
        raise ProtocolError("feature-v2 A/B dataset contract is missing")
    train = [
        _identity(row, f"dataset.expert_train[{index}]")
        for index, row in enumerate(dataset.get("expert_train", ()))
    ]
    development = [
        _identity(row, f"dataset.train_development[{index}]")
        for index, row in enumerate(dataset.get("train_development", ()))
    ]
    if train != readiness_contract["train"] or development != readiness_contract["development"]:
        raise ProtocolError("feature-v2 A/B episode identities differ from readiness")
    if dataset.get("train_sha256") != readiness_contract["source"]["train_sha256"]:
        raise ProtocolError("feature-v2 A/B train dataset digest drifted")
    if dataset.get("selected_scene_assets") != readiness_manifest["dataset"].get(
        "selected_scene_assets"
    ):
        raise ProtocolError("feature-v2 A/B selected scene assets drifted")
    if dataset.get("train_path") != readiness_manifest["dataset"].get("train_path"):
        raise ProtocolError("feature-v2 A/B train path contract drifted")
    if manifest["executor"] != readiness_manifest["executor"]:
        raise ProtocolError("feature-v2 A/B executor differs from readiness")
    if manifest["runtime"] != readiness_manifest["runtime"]:
        raise ProtocolError("feature-v2 A/B runtime differs from readiness")
    if manifest["seeds"] != {"learner": readiness_contract["seed"]}:
        raise ProtocolError("feature-v2 A/B learner seed drifted")
    if manifest["architecture"] != {
        "feature_v1_dim": 7173,
        "feature_v1_schema_sha256": EXPECTED_FEATURE_SCHEMA_DIGESTS[1],
        "feature_v1_schema_version": 1,
        "feature_v2_dim": 3604,
        "feature_v2_schema_sha256": EXPECTED_FEATURE_SCHEMA_DIGESTS[2],
        "feature_v2_schema_version": 2,
        "hidden_dim": 256,
        "max_waypoint_m": 1.25,
        "topology_max_nodes": 161,
        "topology_model_dim": 128,
    }:
        raise ProtocolError("feature-v2 A/B architecture drifted")

    expected_lineage = {
        "causal_manifest_canonical_sha256": EXPECTED_CAUSAL_MANIFEST_CANONICAL_SHA256,
        "causal_manifest_raw_sha256": EXPECTED_CAUSAL_MANIFEST_RAW_SHA256,
        "causal_receipt_file_sha256": EXPECTED_CAUSAL_RECEIPT_FILE_SHA256,
        "causal_receipt_sha256": EXPECTED_CAUSAL_RECEIPT_SHA256,
        "causal_runner_sha256": EXPECTED_CAUSAL_RUNNER_SHA256,
        "causal_authorize_feature_v2": True,
        "readiness_manifest_canonical_sha256": EXPECTED_READINESS_MANIFEST_CANONICAL_SHA256,
        "readiness_manifest_raw_sha256": EXPECTED_READINESS_MANIFEST_RAW_SHA256,
    }
    if manifest["lineage"] != expected_lineage:
        raise ProtocolError("feature-v2 A/B lineage binding drifted")
    if readiness._json_hash(readiness_manifest) != EXPECTED_READINESS_MANIFEST_CANONICAL_SHA256:
        raise ProtocolError("readiness manifest canonical digest drifted")
    if readiness._json_hash(causal_manifest) != EXPECTED_CAUSAL_MANIFEST_CANONICAL_SHA256:
        raise ProtocolError("causal manifest canonical digest drifted")
    if causal_contract["development"] != development:
        raise ProtocolError("causal probe development identities drifted")
    implementation = manifest["implementation"]
    if not isinstance(implementation, Mapping):
        raise ProtocolError("feature-v2 A/B implementation binding is missing")
    _strict_keys(implementation, {"files"}, "implementation")
    files = _implementation_records(implementation["files"])
    return {
        "architecture": dict(manifest["architecture"]),
        "development": development,
        "executor": dict(manifest["executor"]),
        "files": files,
        "protocol": dict(manifest["protocol"]),
        "readiness": readiness_contract,
        "seed": int(manifest["seeds"]["learner"]),
        "thresholds": dict(manifest["thresholds"]),
        "train": train,
        "training": dict(manifest["training"]),
    }


def verify_implementation(
    manifest: Mapping, *, rlinf_root: Path, runner_path: Path | None = None
) -> list[dict]:
    """Verify deployed bytes, including the loaded runner, before any model load."""

    root = rlinf_root.expanduser().resolve(strict=True)
    records = _implementation_records(manifest.get("implementation", {}).get("files"))
    observed = []
    for row in records:
        path = (root / row["deployment_relative_path"]).resolve(strict=True)
        if row["deployment_relative_path"].endswith(
            "run_streamvln_p8_successor_feature_v2_ab.py"
        ) and runner_path is not None:
            path = runner_path.expanduser().resolve(strict=True)
        if path.stat().st_size != row["bytes"] or readiness._file_sha256(path) != row["sha256"]:
            raise ProtocolError(
                "feature-v2 A/B implementation drifted: "
                f"{row['deployment_relative_path']}"
            )
        observed.append(dict(row))
    return observed


def validate_causal_authorization(
    *, receipt_path: Path, causal_manifest_path: Path, readiness_manifest_path: Path
) -> dict:
    if readiness._file_sha256(readiness_manifest_path) != EXPECTED_READINESS_MANIFEST_RAW_SHA256:
        raise ProtocolError("readiness manifest file bytes drifted")
    if readiness._file_sha256(causal_manifest_path) != EXPECTED_CAUSAL_MANIFEST_RAW_SHA256:
        raise ProtocolError("causal manifest file bytes drifted")
    if readiness._file_sha256(receipt_path) != EXPECTED_CAUSAL_RECEIPT_FILE_SHA256:
        raise ProtocolError("causal authorization receipt file bytes drifted")
    value = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ProtocolError("causal authorization receipt must be an object")
    receipt = dict(value)
    embedded = receipt.pop("receipt_sha256", None)
    if embedded != EXPECTED_CAUSAL_RECEIPT_SHA256 or readiness._json_hash(receipt) != embedded:
        raise ProtocolError("causal authorization receipt self-hash drifted")
    result = receipt.get("result")
    split_reads = result.get("split_read_counts") if isinstance(result, Mapping) else None
    if (
        receipt.get("schema_name") != causal.SCHEMA_NAME
        or receipt.get("stage") != "causal_probe"
        or receipt.get("split") != TRAIN_SPLIT
        or receipt.get("authorize_feature_v2") is not True
        or receipt.get("promotion") is not None
        or split_reads != {"train": 24, "val_seen": 0, "val_unseen": 0}
    ):
        raise ProtocolError("causal receipt does not authorize this train-only Feature-v2 A/B")
    attribution = receipt.get("frozen_attribution")
    if not isinstance(attribution, Mapping) or (
        attribution.get("attribution") != "joint_teacher_failure_authorizes_feature_v2"
    ):
        raise ProtocolError("causal receipt attribution differs from the authorized branch")
    return {
        "file_sha256": EXPECTED_CAUSAL_RECEIPT_FILE_SHA256,
        "receipt_sha256": EXPECTED_CAUSAL_RECEIPT_SHA256,
        "authorize_feature_v2": True,
        "attribution": attribution["attribution"],
    }


def production_preflight(args: argparse.Namespace, manifest: Mapping, contract: Mapping) -> dict:
    """Read-only source, asset, lineage, and implementation verification."""

    if str(args.device) != manifest["runtime"].get("device") or str(args.device) != "cuda:0":
        raise ProtocolError(
            "feature-v2 A/B is frozen to one exposed A800 as process-local cuda:0"
        )
    expected_python = Path(manifest["runtime"]["model_python"]).resolve(strict=True)
    if Path(sys.executable).resolve(strict=True) != expected_python:
        raise ProtocolError("feature-v2 A/B must run in the pinned StreamVLN Python")
    roots = {}
    for name in ("rlinf_root", "source_root", "checkpoint", "tokenizer", "vision", "asset_root"):
        path = Path(str(getattr(args, name))).expanduser().resolve(strict=True)
        if not path.is_dir():
            raise ProtocolError(f"--{name.replace('_', '-')} must be a directory")
        roots[name] = path
    if str(roots["rlinf_root"]) not in sys.path:
        sys.path.insert(0, str(roots["rlinf_root"]))
    release_path = Path(args.release_manifest).expanduser().resolve(strict=True)
    release = json.loads(release_path.read_text(encoding="utf-8"))
    ceiling = importlib.import_module("run_streamvln_p8_successor_ceiling")
    release_verified = ceiling._default_release_verifier(
        manifest=release,
        rlinf_root=roots["rlinf_root"],
        source_root=roots["source_root"],
        checkpoint=roots["checkpoint"],
        tokenizer=roots["tokenizer"],
        vision=roots["vision"],
    )
    commit_roots = {
        "rlinf": roots["rlinf_root"],
        "streamvln": roots["source_root"],
        "habitat_sim": Path(manifest["runtime"]["habitat_sim_python"])
        .expanduser()
        .resolve(strict=True)
        .parent,
    }
    observed_commits = {}
    for name, root in commit_roots.items():
        try:
            completed = subprocess.run(
                [
                    "git",
                    "-c",
                    f"safe.directory={root}",
                    "-C",
                    str(root),
                    "rev-parse",
                    "HEAD",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ProtocolError(f"cannot verify {name} source commit") from exc
        observed = completed.stdout.strip()
        if observed != EXPECTED_SOURCE_COMMITS[name]:
            raise ProtocolError(f"{name} source commit drifted")
        observed_commits[name] = observed
    readiness_manifest_path = Path(args.readiness_manifest).expanduser().resolve(strict=True)
    causal_manifest_path = Path(args.causal_manifest).expanduser().resolve(strict=True)
    causal_receipt_path = Path(args.causal_receipt).expanduser().resolve(strict=True)
    causal_authorization = validate_causal_authorization(
        receipt_path=causal_receipt_path,
        causal_manifest_path=causal_manifest_path,
        readiness_manifest_path=readiness_manifest_path,
    )
    train_path = roots["asset_root"] / "datasets" / "r2r" / "train" / "train.json.gz"
    if readiness._file_sha256(train_path) != manifest["dataset"]["train_sha256"]:
        raise ProtocolError("feature-v2 A/B train dataset hash drifted")
    data_source_record = manifest["source_contract"]["data_source_manifest"]
    data_source_path = roots["rlinf_root"] / data_source_record["relative_path"]
    if readiness._file_sha256(data_source_path) != data_source_record["raw_sha256"]:
        raise ProtocolError("feature-v2 A/B diagnostic source manifest drifted")
    scene_assets = []
    for record in manifest["dataset"]["selected_scene_assets"]:
        observed = {"scene": record["scene"]}
        for kind in ("glb", "navmesh"):
            item = record[kind]
            path = roots["asset_root"] / Path(item["path"]).relative_to(
                Path(manifest["dataset"]["train_path"]).parents[3]
            )
            path = path.resolve(strict=True)
            if path.stat().st_size != int(item["bytes"]) or readiness._file_sha256(path) != item["sha256"]:
                raise ProtocolError(f"scene {record['scene']} {kind} bytes drifted")
            observed[kind] = {"path": str(path), "bytes": path.stat().st_size, "sha256": item["sha256"]}
        scene_assets.append(observed)
    implementation = verify_implementation(manifest, rlinf_root=roots["rlinf_root"])
    return {
        "status": "pass",
        "split": TRAIN_SPLIT,
        "device": str(args.device),
        "release_manifest_raw_sha256": readiness._file_sha256(release_path),
        "release_manifest_canonical_sha256": release_verified["anchor"],
        "source_commits": observed_commits,
        "causal_authorization": causal_authorization,
        "implementation": implementation,
        "scene_assets": scene_assets,
        "paths": {**{name: str(path) for name, path in roots.items()}, "model_python": str(expected_python)},
        "guards": {
            "model_load_count": 0,
            "optimizer_steps": 0,
            "val_seen_reads": 0,
            "val_unseen_reads": 0,
            "ppo_updates": 0,
            "promotion_attempts": 0,
        },
    }


def paired_state_sha256(
    *, episode_id: str, scene: str, collection_role: str, decision_index: int,
    frame_ref: str, topology_digest: str, route_label: str,
) -> str:
    if (
        not episode_id
        or not scene
        or collection_role not in {"expert", "exact_keep", "teacher_forced"}
        or type(decision_index) is not int
        or decision_index < 0
        or not isinstance(frame_ref, str)
        or not frame_ref
        or route_label not in readiness.ROUTES
    ):
        raise ProtocolError("paired feature-independent state identity is invalid")
    readiness._sha(topology_digest, "topology_digest")
    return readiness._json_hash(
        {
            "collection_role": collection_role,
            "decision_index": decision_index,
            "episode_id": episode_id,
            "frame_ref": frame_ref,
            "route_label": route_label,
            "scene": scene,
            "topology_digest": topology_digest,
        }
    )


def deduplicate_paired_rows(rows: Sequence[Mapping]) -> list[dict]:
    unique: dict[str, dict] = {}
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise ProtocolError(f"paired row {index} must be an object")
        row = dict(raw)
        state_hash = readiness._sha(row.get("state_hash"), f"paired_rows[{index}].state_hash")
        if row.get("route_label") not in readiness.ROUTES:
            raise ProtocolError(f"paired row {index} route label is invalid")
        contexts = row.get("forward_contexts")
        if not isinstance(contexts, Mapping) or set(contexts) != set(ARMS):
            raise ProtocolError(f"paired row {index} lacks the exact two feature contexts")
        signature = readiness._json_hash(
            {
                "expert_first_primitive": row.get("expert_first_primitive"),
                "route_label": row.get("route_label"),
                "waypoint_target": row.get("waypoint_target"),
            }
        )
        if state_hash in unique:
            previous = unique[state_hash]
            previous_signature = readiness._json_hash(
                {
                    "expert_first_primitive": previous.get("expert_first_primitive"),
                    "route_label": previous.get("route_label"),
                    "waypoint_target": previous.get("waypoint_target"),
                }
            )
            if signature != previous_signature:
                raise ProtocolError("duplicate paired state has conflicting labels")
            continue
        unique[state_hash] = row
    return list(unique.values())


def project_arm_rows(rows: Sequence[Mapping], arm: str) -> list[dict]:
    if arm not in ARMS:
        raise ValueError("unknown feature arm")
    paired = deduplicate_paired_rows(rows)
    return [
        {
            "state_hash": row["state_hash"],
            "route_label": row["route_label"],
            "waypoint_target": list(row["waypoint_target"]),
            "expert_first_primitive": int(row["expert_first_primitive"]),
            "episode_id": row["episode_id"],
            "scene": row["scene"],
            "decision_index": int(row["decision_index"]),
            "forward_context": {
                key: value.detach().cpu().clone()
                for key, value in row["forward_contexts"][arm].items()
            },
        }
        for row in paired
    ]


def _finite_metric(value: object, name: str) -> float:
    return readiness._finite(value, name, minimum=0.0)


def _teacher_metrics(value: object, name: str) -> dict:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{name} teacher metrics are missing")
    result = dict(value)
    supports = result.get("route_support")
    if not isinstance(supports, Mapping) or set(supports) != set(readiness.ROUTES):
        raise ProtocolError(f"{name} teacher route support drifted")
    result["route_support"] = {
        route: readiness._strict_int(supports[route], f"{name}.route_support.{route}")
        for route in readiness.ROUTES
    }
    return result


def _forward_nonstop_fraction(navigation: Mapping, name: str) -> float | None:
    actions = navigation.get("action_counts")
    if not isinstance(actions, Mapping) or set(actions) != set(readiness.ACTION_NAMES):
        raise ProtocolError(f"{name} action counts drifted")
    normalized = {
        action: readiness._strict_int(actions[action], f"{name}.action_counts.{action}")
        for action in readiness.ACTION_NAMES
    }
    nonstop = sum(normalized[action] for action in ("FORWARD", "LEFT", "RIGHT"))
    return normalized["FORWARD"] / nonstop if nonstop else None


def adjudicate_feature_v2_ab(payload: Mapping, *, thresholds: Mapping) -> dict:
    teacher = payload.get("teacher_forced")
    if not isinstance(teacher, Mapping):
        raise ProtocolError("feature-v2 A/B teacher evidence is missing")
    v1 = _teacher_metrics(teacher.get(ARM_V1, {}).get("metrics"), ARM_V1)
    v2 = _teacher_metrics(teacher.get(ARM_V2, {}).get("metrics"), ARM_V2)
    sample_min = thresholds["teacher_sample_min"]
    sample_checks = {
        route: v2["route_support"][route] >= int(sample_min[route])
        for route in readiness.ROUTES
    }

    route_threshold = thresholds["teacher_route"]
    v1_bal = _finite_metric(v1.get("route_nonstop_bal_acc"), "v1.route_nonstop_bal_acc")
    v2_bal = _finite_metric(v2.get("route_nonstop_bal_acc"), "v2.route_nonstop_bal_acc")
    v2_stop = v2.get("stop")
    if not isinstance(v2_stop, Mapping):
        raise ProtocolError("feature-v2 STOP metrics are missing")
    v2_stop_recall = _finite_metric(v2_stop.get("recall"), "v2.stop.recall")
    premature_stop = readiness._strict_int(v2_stop.get("false_positive"), "v2.stop.false_positive")
    route_checks = {
        "feature_v2_route_nonstop_bal_acc": v2_bal
        >= float(route_threshold["feature_v2_route_nonstop_bal_acc_min"]),
        "feature_v2_route_delta_vs_v1": (v2_bal - v1_bal) + 1e-12
        >= float(route_threshold["feature_v2_delta_vs_v1_min"]),
        "feature_v2_stop_recall": v2_stop_recall
        >= float(route_threshold["feature_v2_stop_recall_min"]),
        "feature_v2_premature_stop_count": premature_stop
        <= int(route_threshold["feature_v2_premature_stop_count_max"]),
    }

    waypoint_threshold = thresholds["teacher_waypoint"]
    v1_wp = v1.get("waypoint")
    v2_wp = v2.get("waypoint")
    if not isinstance(v1_wp, Mapping) or not isinstance(v2_wp, Mapping):
        raise ProtocolError("feature-v2 A/B waypoint metrics are missing")
    v1_agreement = _finite_metric(v1_wp.get("first_action_agreement"), "v1.waypoint.first_action_agreement")
    v2_agreement = _finite_metric(v2_wp.get("first_action_agreement"), "v2.waypoint.first_action_agreement")
    median_bearing = _finite_metric(v2_wp.get("median_bearing_abs_error_deg"), "v2.waypoint.median_bearing")
    p90_bearing = _finite_metric(v2_wp.get("p90_bearing_abs_error_deg"), "v2.waypoint.p90_bearing")
    median_l2 = _finite_metric(v2_wp.get("median_vector_l2_m"), "v2.waypoint.median_l2")
    waypoint_checks = {
        "feature_v2_first_action_agreement": v2_agreement
        >= float(waypoint_threshold["feature_v2_first_action_agreement_min"]),
        "feature_v2_first_action_delta_vs_v1": (
            v2_agreement - v1_agreement
        )
        + 1e-12
        >= float(waypoint_threshold["feature_v2_first_action_delta_vs_v1_min"]),
        "feature_v2_median_bearing": median_bearing
        <= float(waypoint_threshold["feature_v2_bearing_abs_error_median_deg_max"]),
        "feature_v2_p90_bearing": p90_bearing
        <= float(waypoint_threshold["feature_v2_bearing_abs_error_p90_deg_max"]),
        "feature_v2_median_vector_l2": median_l2
        <= float(waypoint_threshold["feature_v2_vector_l2_median_m_max"]),
    }

    navigation = payload.get("navigation")
    if not isinstance(navigation, Mapping):
        raise ProtocolError("feature-v2 A/B navigation evidence is missing")
    keep = readiness._validate_nav_metrics(navigation.get("matched_keep"), role="matched_keep")
    learned_v1 = readiness._validate_nav_metrics(navigation.get(ARM_V1), role=ARM_V1)
    learned_v2 = readiness._validate_nav_metrics(navigation.get(ARM_V2), role=ARM_V2)
    v2_extra = navigation.get(ARM_V2)
    if not isinstance(v2_extra, Mapping):
        raise ProtocolError("feature-v2 closed-loop evidence is invalid")
    forward_fraction = _forward_nonstop_fraction(v2_extra, ARM_V2)
    closed_threshold = thresholds["closed_loop"]
    accepted_count = readiness._strict_int(v2_extra.get("accepted_override_count"), "v2.accepted_override_count")
    accepted_episodes = readiness._strict_int(v2_extra.get("accepted_override_episode_count"), "v2.accepted_override_episode_count")
    successful_accepted = readiness._strict_int(
        v2_extra.get("successful_episode_with_accepted_override_count"),
        "v2.successful_episode_with_accepted_override_count",
    )
    recover = v2_extra.get("recover")
    if not isinstance(recover, Mapping):
        raise ProtocolError("feature-v2 recovery evidence is missing")
    eligible_blocked = readiness._strict_int(
        recover.get("eligible_blocked_count"), "v2.recover.eligible_blocked_count"
    )
    covered_next = readiness._strict_int(
        recover.get("covered_next_decision_count"),
        "v2.recover.covered_next_decision_count",
    )
    alignment_miss = readiness._strict_int(
        recover.get("alignment_miss_count"), "v2.recover.alignment_miss_count"
    )
    closed_checks = {
        "feature_v2_keep_noninferiority": learned_v2["success_count"]
        >= keep["success_count"]
        - int(closed_threshold["feature_v2_max_success_deficit_vs_keep"]),
        "feature_v2_success": learned_v2["success_count"]
        >= int(closed_threshold["feature_v2_success_count_min"]),
        "feature_v2_successful_stop": learned_v2["successful_stop_count"]
        >= int(closed_threshold["feature_v2_successful_stop_count_min"]),
        "feature_v2_success_delta_vs_v1": (
            learned_v2["success_count"] - learned_v1["success_count"]
        )
        >= int(closed_threshold["feature_v2_success_delta_vs_v1_min"]),
        "feature_v2_root_recovery_available": learned_v2["root_recovery_unavailable_count"]
        <= int(closed_threshold["feature_v2_root_recovery_unavailable_count_max"]),
        "feature_v2_recover_alignment": alignment_miss
        <= int(closed_threshold["feature_v2_recover_alignment_miss_count_max"]),
        "feature_v2_recover_coverage": covered_next == eligible_blocked,
        "feature_v2_forward_nonstop_fraction": forward_fraction is not None
        and forward_fraction
        >= float(closed_threshold["feature_v2_forward_nonstop_fraction_min"]),
        "feature_v2_accepted_override_count": accepted_count
        >= int(closed_threshold["feature_v2_accepted_override_count_min"]),
        "feature_v2_accepted_override_episode_count": accepted_episodes
        >= int(closed_threshold["feature_v2_accepted_override_episode_count_min"]),
        "feature_v2_successful_episode_with_accepted_override": successful_accepted
        >= int(
            closed_threshold[
                "feature_v2_successful_episode_with_accepted_override_count_min"
            ]
        ),
    }
    checks = {**{f"sample_{key}": value for key, value in sample_checks.items()}, **route_checks, **waypoint_checks, **closed_checks}
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "sample_sufficient": all(sample_checks.values()),
        "teacher_observed": {
            "feature_v1_route_nonstop_bal_acc": v1_bal,
            "feature_v2_route_nonstop_bal_acc": v2_bal,
            "route_delta_v2_minus_v1": v2_bal - v1_bal,
            "feature_v2_stop_recall": v2_stop_recall,
            "feature_v2_premature_stop_count": premature_stop,
            "feature_v1_first_action_agreement": v1_agreement,
            "feature_v2_first_action_agreement": v2_agreement,
            "first_action_delta_v2_minus_v1": v2_agreement - v1_agreement,
            "feature_v2_median_bearing_abs_error_deg": median_bearing,
            "feature_v2_p90_bearing_abs_error_deg": p90_bearing,
            "feature_v2_median_vector_l2_m": median_l2,
        },
        "closed_loop_observed": {
            "matched_keep_success_count": keep["success_count"],
            "feature_v1_success_count": learned_v1["success_count"],
            "feature_v2_success_count": learned_v2["success_count"],
            "feature_v2_successful_stop_count": learned_v2["successful_stop_count"],
            "feature_v2_root_recovery_unavailable_count": learned_v2["root_recovery_unavailable_count"],
            "feature_v2_eligible_blocked_count": eligible_blocked,
            "feature_v2_covered_next_decision_count": covered_next,
            "feature_v2_recover_alignment_miss_count": alignment_miss,
            "feature_v2_recover_parent_count": readiness._strict_int(
                recover.get("recover_parent_count"), "v2.recover.recover_parent_count"
            ),
            "feature_v2_root_keep_fallback_count": readiness._strict_int(
                recover.get("root_keep_fallback_count"),
                "v2.recover.root_keep_fallback_count",
            ),
            "feature_v2_forward_nonstop_fraction": forward_fraction,
            "feature_v2_accepted_override_count": accepted_count,
            "feature_v2_accepted_override_episode_count": accepted_episodes,
            "feature_v2_successful_episode_with_accepted_override_count": successful_accepted,
        },
    }


class FeatureV2ABProvider(Protocol):
    def execute(
        self, *, manifest: Mapping, contract: Mapping, preflight: Mapping,
        args: argparse.Namespace,
    ) -> Mapping:
        ...


class CanonicalFeatureV2ABProvider(readiness.CanonicalReadinessProvider):
    """One-model-load live provider for the frozen paired A/B protocol."""

    MODES = frozenset(
        {
            "expert_collect",
            "keep_collect",
            "teacher_forced",
            "matched_keep",
            "learned_v1",
            "learned_v2",
        }
    )

    def __init__(
        self, *, args: argparse.Namespace, manifest: Mapping, contract: Mapping,
        preflight: Mapping, readiness_manifest: Mapping,
    ) -> None:
        super().__init__(
            args=args,
            manifest=readiness_manifest,
            contract=contract["readiness"],
            preflight=preflight,
        )
        self.ab_manifest = manifest
        self.ab_contract = contract
        self.manifest = manifest
        self.contract = contract
        self.policy_v1 = None
        self.policy_v2 = None
        self.policy = None
        self._v1_latch = None
        self._v1_latch_key = None
        self._optimizer_steps = 0
        self._pretraining_feature_guard: dict | None = None
        self._feature_pair_counts = Counter(
            {
                "decision_states": 0,
                "fresh_context_states": 0,
                "stale_context_states": 0,
                "v1_latch_updates": 0,
            }
        )

    def _load_once(self) -> None:
        if self.adapter is not None:
            return
        import torch

        rlinf_root = Path(self.preflight["paths"]["rlinf_root"])
        if str(rlinf_root) not in sys.path:
            sys.path.insert(0, str(rlinf_root))
        self.runtime = readiness._canonical_import(
            rlinf_root, "streamvln_p8_successor_runtime"
        )
        self.action = readiness._canonical_import(
            rlinf_root, "streamvln_p8_successor_action"
        )
        self.protocol = readiness._canonical_import(
            rlinf_root, "streamvln_p8_successor_protocol"
        )
        self.policy_module = readiness._canonical_import(
            rlinf_root, "streamvln_p8_successor_policy"
        )
        self.topology_module = readiness._canonical_import(
            rlinf_root, "streamvln_p8_topology"
        )
        self.adapter_module = readiness._canonical_import(
            rlinf_root, "streamvln_p8_official_adapter"
        )
        self.official_module = readiness._canonical_import(
            rlinf_root, "streamvln_p8_official_keep"
        )
        self.ceiling = importlib.import_module("run_streamvln_p8_successor_ceiling")
        factory = self.adapter_module.PinnedStreamVLNGenerationAdapter.from_released_checkpoint
        constructor = self.adapter_module.PinnedStreamVLNGenerationAdapter
        default_factory = inspect.signature(factory).parameters["enable_feature_v2"].default
        default_constructor = inspect.signature(constructor).parameters[
            "enable_feature_v2"
        ].default
        if default_factory is not False or default_constructor is not False:
            raise ProtocolError("default official adapter path no longer keeps Feature-v2 off")
        release_path = Path(self.args.release_manifest).expanduser().resolve(strict=True)
        release = json.loads(release_path.read_text(encoding="utf-8"))
        self.adapter = factory(
            model_path=self.preflight["paths"]["checkpoint"],
            tokenizer_path=self.preflight["paths"]["tokenizer"],
            vision_tower_path=self.preflight["paths"]["vision"],
            source_root=self.preflight["paths"]["source_root"],
            artifact_manifest=release,
            device=str(self.args.device),
            local_rank=0,
            env_count=1,
            model_env_id=0,
            enable_feature_v2=True,
        )
        if self.adapter.enable_feature_v2 is not True:
            raise ProtocolError("Feature-v2 adapter opt-in did not take effect")
        schema = self.policy_module.successor_feature_v2_schema()
        if (
            schema.get("version") != 2
            or schema.get("total_dim") != 3604
            or self.policy_module.SUCCESSOR_FEATURE_V2_DIM != 3604
            or self.policy_module.successor_feature_schema_digest(1)
            != EXPECTED_FEATURE_SCHEMA_DIGESTS[1]
            or self.policy_module.successor_feature_schema_digest(2)
            != EXPECTED_FEATURE_SCHEMA_DIGESTS[2]
        ):
            raise ProtocolError("Feature-v2 schema/dimension preflight drifted")
        self._model_load_count = 1
        seed = int(self.ab_contract["seed"])
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _run_cfg(self) -> dict:
        runtime = self.ab_manifest["runtime"]
        return {
            "seed": int(self.ab_contract["seed"]),
            "habitat_python": runtime["habitat_python"],
            "habitat_server": str(
                Path(self.preflight["paths"]["rlinf_root"])
                / "scripts"
                / "habitat_r2r_service.py"
            ),
            "asset_root": self.preflight["paths"]["asset_root"],
            "habitat_sim_python": runtime["habitat_sim_python"],
            "habitat_env_lib": runtime["habitat_env_lib"],
            "egl_vendor_json": runtime["egl_vendor_json"],
            "habitat_startup_timeout_s": float(runtime["habitat_startup_timeout_s"]),
        }

    def _ensure_policies(self, v1_feature, v2_feature, topology_snapshot) -> None:
        import torch

        architecture = self.ab_contract["architecture"]
        if int(v1_feature.numel()) != int(architecture["feature_v1_dim"]):
            raise ProtocolError("live Feature-v1 dimension drifted")
        if int(v2_feature.numel()) != int(architecture["feature_v2_dim"]):
            raise ProtocolError("live Feature-v2 dimension drifted")
        if self.policy_v1 is not None or self.policy_v2 is not None:
            if self.policy_v1 is None or self.policy_v2 is None:
                raise ProtocolError("paired policies were only partially initialized")
            return
        topology_max_nodes = int(topology_snapshot.tensors.max_nodes)
        if topology_max_nodes != int(architecture["topology_max_nodes"]):
            raise ProtocolError("live topology extent drifted")

        def create(feature_dim: int):
            return self.policy_module.P8SuccessorPolicy(
                feature_dim=feature_dim,
                hidden_dim=int(architecture["hidden_dim"]),
                topology_max_nodes=topology_max_nodes,
                topology_model_dim=int(architecture["topology_model_dim"]),
                max_waypoint_m=float(architecture["max_waypoint_m"]),
            ).float().to(self.args.device)

        seed = int(self.ab_contract["seed"])
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.policy_v1 = create(int(v1_feature.numel()))
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.policy_v2 = create(int(v2_feature.numel()))
        self.policy_v1.bc_ready.zero_()
        self.policy_v2.bc_ready.zero_()

    def _current_depth_summary(self, frame):
        return self.adapter_module.build_feature_v2_depth_summary(
            frame,
            torch_module=self.adapter.torch,
            tensor_auditor=self.adapter.tensor_auditor,
        )

    def _paired_features(
        self, *, frame, instruction: str, official_shadow_action: int,
        physical_step: int,
    ) -> tuple[object, object, dict]:
        snapshot_v2 = self.adapter.latest_feature_v2_snapshot()
        if snapshot_v2.reset_ordinal != self.adapter.reset_ordinal:
            raise ProtocolError("Feature-v2 snapshot is not bound to the active reset")
        if snapshot_v2.episode_generation != frame.episode_generation:
            raise ProtocolError("Feature-v2 snapshot escaped the active episode generation")
        if self.adapter.generation_ordinal != snapshot_v2.generation_ordinal + 1:
            raise ProtocolError("Feature-v2 contextual generation latch ordinal drifted")
        latch_key = (snapshot_v2.reset_ordinal, snapshot_v2.generation_ordinal)
        fresh_latch = latch_key != self._v1_latch_key
        if fresh_latch:
            self._v1_latch = self.adapter.extract_frozen_feature_snapshot(
                frame=frame, instruction=instruction
            )
            self._v1_latch_key = latch_key
            self._feature_pair_counts["v1_latch_updates"] += 1
        if self._v1_latch is None:
            raise ProtocolError("Feature-v1 latch is unavailable")
        snapshot_v1 = self._v1_latch
        depth_summary, depth_audit, depth_source_audit = self._current_depth_summary(frame)
        observed_depth_audit = dict(self.adapter.tensor_auditor(depth_summary))
        if (
            depth_source_audit.get("frame_ref") != frame.frame_ref
            or depth_source_audit.get("depth_sha256") != frame.depth_sha256
            or depth_source_audit.get("physical_step") != frame.physical_step
            or observed_depth_audit.get("sha256") != depth_audit.get("sha256")
        ):
            raise ProtocolError("Feature-v2 current-depth provenance drifted")
        feature_v1 = self.policy_module.compose_successor_features(
            frozen_feature=snapshot_v1.feature,
            official_shadow_action=int(official_shadow_action),
            physical_step=int(physical_step),
            feature_physical_step=int(snapshot_v1.physical_step),
        )
        feature_v2 = self.policy_module.compose_successor_features_v2(
            contextual_hidden=snapshot_v2.contextual_hidden,
            depth_summary=depth_summary,
            official_shadow_action=int(official_shadow_action),
            physical_step=int(physical_step),
            context_physical_step=int(snapshot_v2.physical_step),
        )
        v2_audit = self.policy_module.audit_successor_features_v2(feature_v2)
        age_v1 = int(float(feature_v1[-1].item()))
        age_v2 = int(v2_audit["context_age"])
        if age_v1 != age_v2:
            raise ProtocolError("paired Feature-v1/v2 temporal ages differ")
        if snapshot_v1.reset_ordinal != snapshot_v2.reset_ordinal:
            raise ProtocolError("paired Feature-v1/v2 reset ordinals differ")
        if snapshot_v1.episode_generation != snapshot_v2.episode_generation:
            raise ProtocolError("paired Feature-v1/v2 episode generations differ")
        if fresh_latch and (
            snapshot_v1.physical_step != snapshot_v2.physical_step
            or snapshot_v1.frame_ref != snapshot_v2.frame_ref
            or snapshot_v2.frame_ref != frame.frame_ref
            or age_v1 != 0
            or age_v2 != 0
        ):
            raise ProtocolError("same-generation Feature-v1/v2 freshness alignment drifted")
        self._feature_pair_counts["decision_states"] += 1
        self._feature_pair_counts[
            "fresh_context_states" if age_v2 == 0 else "stale_context_states"
        ] += 1
        audit = {
            "fresh_latch": fresh_latch,
            "context_age": age_v2,
            "reset_ordinal": int(snapshot_v2.reset_ordinal),
            "active_reset_ordinal": int(self.adapter.reset_ordinal),
            "feature_v1_generation_ordinal_post_generate": int(
                snapshot_v1.generation_ordinal
            ),
            "feature_v2_generation_ordinal": int(snapshot_v2.generation_ordinal),
            "adapter_generation_ordinal_post_generate": int(
                self.adapter.generation_ordinal
            ),
            "frame_ref": frame.frame_ref,
            "episode_generation": frame.episode_generation,
            "snapshot_episode_generation": snapshot_v2.episode_generation,
            "feature_v1_sha256": snapshot_v1.feature_audit["sha256"],
            "feature_v2_sha256": v2_audit["feature_sha256"],
            "feature_v2_schema_sha256": v2_audit["schema_sha256"],
            "current_depth_sha256": depth_audit["sha256"],
            "current_depth_source": dict(depth_source_audit),
        }
        if self._pretraining_feature_guard is None:
            if self._optimizer_steps != 0:
                raise ProtocolError("Feature-v2 first-generate guard ran after optimization")
            if not fresh_latch:
                raise ProtocolError("first train state did not execute a fresh generation")
            if snapshot_v1.generation_ordinal != snapshot_v2.generation_ordinal + 1:
                raise ProtocolError("paired generation ordinal convention drifted")
            self._pretraining_feature_guard = {
                "passed": True,
                "optimizer_steps_before_guard": 0,
                "adapter_feature_v2_opt_in": self.adapter.enable_feature_v2 is True,
                "feature_v2_schema_version": 2,
                "feature_v2_dim": int(feature_v2.numel()),
                "feature_v1_dim": int(feature_v1.numel()),
                "same_frame_ref": snapshot_v1.frame_ref == snapshot_v2.frame_ref,
                "same_physical_step": snapshot_v1.physical_step
                == snapshot_v2.physical_step,
                "same_reset_ordinal": snapshot_v1.reset_ordinal
                == snapshot_v2.reset_ordinal,
                "same_episode_generation": snapshot_v1.episode_generation
                == snapshot_v2.episode_generation,
                "both_fresh": age_v1 == age_v2 == 0,
                "contextual_hidden_sha256": snapshot_v2.contextual_hidden_audit[
                    "sha256"
                ],
                "feature_v2_schema_sha256": v2_audit["schema_sha256"],
                "default_adapter_feature_v2_opt_in": False,
            }
            if not all(
                self._pretraining_feature_guard[key] is True
                for key in (
                    "adapter_feature_v2_opt_in",
                    "same_frame_ref",
                    "same_physical_step",
                    "same_reset_ordinal",
                    "same_episode_generation",
                    "both_fresh",
                )
            ):
                raise ProtocolError("Feature-v2 first-generate pretraining guard failed")
        return feature_v1, feature_v2, audit

    def _predict(self, *, policy, context: Mapping) -> tuple[list[float], tuple[float, float]]:
        import torch

        with torch.inference_mode():
            *_, route_logits, raw_mean, _raw_log_std, _values = policy._distribution_terms(
                context
            )
            probabilities = torch.softmax(route_logits, dim=-1).reshape(-1)
            waypoint = policy._waypoint_from_raw(raw_mean).reshape(-1)
        return (
            [float(value) for value in probabilities.detach().cpu().tolist()],
            tuple(float(value) for value in waypoint.detach().cpu().tolist()),
        )

    def _score_teacher_rows(self, rows: Sequence[Mapping], arm: str) -> list[dict]:
        import torch

        if arm not in ARMS:
            raise ValueError("unknown feature arm")
        policy = self.policy_v1 if arm == ARM_V1 else self.policy_v2
        if policy is None:
            raise ProtocolError("teacher scoring requires both trained policies")
        records = []
        for row in deduplicate_paired_rows(rows):
            context = {
                key: value.to(device=self.args.device)
                for key, value in row["forward_contexts"][arm].items()
            }
            with torch.inference_mode():
                *_, route_logits, raw_mean, _raw_log_std, _values = (
                    policy._distribution_terms(context)
                )
                predicted_route = self.policy_module.POLICY_ROUTES[
                    int(route_logits.argmax(dim=-1).item())
                ]
                waypoint_tensor = policy._waypoint_from_raw(raw_mean)
            predicted_waypoint = tuple(
                float(value) for value in waypoint_tensor.reshape(-1).tolist()
            )
            predicted_actions = tuple(
                self.action.compile_metric_waypoint(predicted_waypoint)
            )
            truth_route = self.action.RouteName(str(row["route_label"]))
            record = {
                "truth_route": truth_route.value,
                "predicted_route": predicted_route.value,
                "predicted_waypoint_first_action": int(predicted_actions[0]),
                "predicted_waypoint_zero_forward": (
                    self.action.FORWARD_ACTION not in predicted_actions
                ),
                "oracle_waypoint_first_action": None,
                "waypoint_vector_l2_m": None,
                "waypoint_bearing_abs_error_rad": None,
                "waypoint_distance_abs_error_m": None,
                "selected_waypoint_zero_forward": None,
            }
            if truth_route is self.action.RouteName.WAYPOINT:
                target = tuple(float(value) for value in row["waypoint_target"])
                target_actions = tuple(self.action.compile_metric_waypoint(target))
                if int(target_actions[0]) != int(row["expert_first_primitive"]):
                    raise ProtocolError("teacher waypoint compilation differs from label")
                record.update(
                    {
                        "oracle_waypoint_first_action": int(target_actions[0]),
                        "waypoint_vector_l2_m": math.hypot(
                            predicted_waypoint[0] - target[0],
                            predicted_waypoint[1] - target[1],
                        ),
                        "waypoint_bearing_abs_error_rad": causal._angle_error_radians(
                            math.atan2(predicted_waypoint[1], predicted_waypoint[0]),
                            math.atan2(target[1], target[0]),
                        ),
                        "waypoint_distance_abs_error_m": abs(
                            math.hypot(*predicted_waypoint) - math.hypot(*target)
                        ),
                    }
                )
            records.append(record)
        return records

    def _run_episode(
        self, *, spec: Mapping, mode: str, decision_budget: int,
        physical_budget: int, collect_rows: bool,
    ) -> dict:
        if mode not in self.MODES:
            raise ValueError("unknown Feature-v2 A/B episode mode")
        exact_control = mode in {"keep_collect", "matched_keep"}
        learned = mode in {"learned_v1", "learned_v2"}
        env = self.runtime.make_successor_env(
            self._run_cfg(),
            split=TRAIN_SPLIT,
            episode_index=int(spec["index"]),
            max_steps=int(physical_budget),
        )
        self._split_reads[TRAIN_SPLIT] += 1
        if learned:
            env.successor_planner = self.action.SuccessorPlanner(
                blocked_threshold=int(
                    self.ab_contract["executor"]["blocked_forward_threshold"]
                ),
                root_recovery_policy="keep_original",
            )
        rows = []
        route_counts = Counter({route: 0 for route in readiness.ROUTES})
        action_counts = Counter({name: 0 for name in readiness.ACTION_NAMES})
        gate_reasons = Counter()
        gate_records = []
        recover = {
            "eligible_blocked_count": 0,
            "covered_next_decision_count": 0,
            "recover_parent_count": 0,
            "alignment_miss_count": 0,
            "root_recovery_unavailable_count": 0,
            "root_keep_fallback_count": 0,
        }
        executed_actions = []
        realized_path_m = 0.0
        collision_count = 0
        terminated = truncated = False
        exhausted = False
        failure_reason = None
        prepared_data: dict[str, object] = {}
        self._v1_latch = None
        self._v1_latch_key = None

        try:
            identity = None

            def official_driver_factory(env_id: int, selected: Mapping, episode_serial: int):
                session = self.official_module.OfficialEvalSession(
                    env_id=env_id,
                    instruction=selected["instruction_text"],
                    conjunction_rng=random.Random(
                        int(self.ab_contract["seed"]) + int(spec["index"])
                    ),
                )
                return self.adapter.make_driver(
                    session=session,
                    episode_serial=episode_serial,
                    episode_identity=selected,
                )

            bridge = self.runtime.SuccessorRunnerBridge(
                env=env,
                topology_factory=lambda: self.topology_module.OnlineTopology(
                    max_decisions=int(
                        self.ab_contract["architecture"]["topology_max_nodes"]
                    )
                    - 1,
                    forward_step_m=0.25,
                    turn_degrees=15.0,
                ),
                official_driver_factory=official_driver_factory,
                exact_control=exact_control,
                require_prepared_decision_context=True,
            )
            bridge.reset_episode()
            identity = getattr(env, "identity", {})
            if not self._identity_matches(identity, spec):
                raise ProtocolError("live Habitat identity differs from Feature-v2 manifest")

            collection_role = {
                "expert_collect": "expert",
                "keep_collect": "exact_keep",
                "teacher_forced": "teacher_forced",
            }.get(mode, "teacher_forced")

            def context_builder(frame, shadow_action):
                expert = env.expert_waypoint(
                    float(self.ab_contract["executor"]["expert_lookahead_m"])
                )
                if (
                    not isinstance(expert, Mapping)
                    or type(expert.get("stop")) is not bool
                    or not isinstance(expert.get("waypoint_local_m"), (list, tuple))
                    or len(expert["waypoint_local_m"]) != 2
                ):
                    raise ProtocolError("train shortest-path expert payload drifted")
                waypoint = tuple(float(value) for value in expert["waypoint_local_m"])
                expert_actions = (
                    (self.action.STOP_ACTION,)
                    if expert["stop"]
                    else self.action.compile_metric_waypoint(waypoint)
                )
                feature_v1, feature_v2, feature_audit = self._paired_features(
                    frame=frame,
                    instruction=identity["instruction_text"],
                    official_shadow_action=int(shadow_action),
                    physical_step=int(frame.physical_step),
                )
                topology_snapshot = self.topology_module.snapshot_online_topology(
                    bridge.topology,
                    episode_token=str(bridge.episode_token),
                    physical_step=int(frame.physical_step),
                    decision_index=int(bridge._next_decision_index),
                    max_nodes=int(
                        self.ab_contract["architecture"]["topology_max_nodes"]
                    ),
                )
                self._ensure_policies(feature_v1, feature_v2, topology_snapshot)
                contexts = {
                    ARM_V1: self.policy_module.make_successor_forward_context(
                        features=feature_v1,
                        topology_snapshot=topology_snapshot,
                        policy_version=0,
                    ),
                    ARM_V2: self.policy_module.make_successor_forward_context(
                        features=feature_v2,
                        topology_snapshot=topology_snapshot,
                        policy_version=0,
                    ),
                }
                topology_hashes = {
                    arm: readiness._tensor_mapping_hash(
                        {key: value for key, value in context.items() if key != "features"}
                    )
                    for arm, context in contexts.items()
                }
                if len(set(topology_hashes.values())) != 1:
                    raise ProtocolError("paired arms received different topology snapshots")
                label = readiness.label_successor_route(
                    expert_stop=expert["stop"],
                    official_shadow_action=int(shadow_action),
                    expert_first_primitive=int(expert_actions[0]),
                )
                target_waypoint = waypoint if label == "waypoint" else (0.0, 0.0)
                state_hash = paired_state_sha256(
                    episode_id=str(spec["episode_id"]),
                    scene=str(spec["scene"]),
                    collection_role=collection_role,
                    decision_index=int(bridge._next_decision_index),
                    frame_ref=str(frame.frame_ref),
                    topology_digest=next(iter(topology_hashes.values())),
                    route_label=label,
                )
                if collect_rows:
                    rows.append(
                        {
                            "state_hash": state_hash,
                            "route_label": label,
                            "waypoint_target": list(target_waypoint),
                            "expert_first_primitive": int(expert_actions[0]),
                            "episode_id": str(spec["episode_id"]),
                            "scene": str(spec["scene"]),
                            "decision_index": int(bridge._next_decision_index),
                            "forward_contexts": {
                                arm: {
                                    key: value.detach().cpu().clone()
                                    for key, value in context.items()
                                }
                                for arm, context in contexts.items()
                            },
                            "feature_pair_audit": feature_audit,
                        }
                    )
                blocked = int(bridge.topology.consecutive_blocked_forwards)
                current = int(bridge.topology.current_node)
                parent = bridge.topology.nodes[current].parent
                recovery_active = (
                    getattr(env.successor_planner, "recovery_parent_index", None)
                    is not None
                )
                blocked_now = blocked >= int(
                    self.ab_contract["executor"]["blocked_forward_threshold"]
                )
                prepared_data.clear()
                prepared_data.update(
                    {
                        "contexts": contexts,
                        "expert": expert,
                        "label": label,
                        "waypoint": waypoint,
                        "target_waypoint": target_waypoint,
                        "shadow_action": int(shadow_action),
                        "context_age": int(feature_audit["context_age"]),
                        "recovery_active": bool(recovery_active),
                        "blocked_parent": bool(blocked_now and parent is not None),
                        "blocked_root": bool(blocked_now and parent is None),
                    }
                )
                return contexts

            decisions = 0
            while decisions < decision_budget and not (terminated or truncated):
                prepared = bridge.prepare_decision_context(context_builder)
                if prepared.policy_context is not prepared_data["contexts"]:
                    raise ProtocolError("prepared paired context identity drifted")
                expert = prepared_data["expert"]
                shadow_action = int(prepared_data["shadow_action"])
                if mode in {"expert_collect", "teacher_forced"}:
                    route = (
                        self.action.RouteName.STOP
                        if expert["stop"]
                        else self.action.RouteName.WAYPOINT
                    )
                    waypoint = (
                        (0.0, 0.0)
                        if expert["stop"]
                        else tuple(prepared_data["waypoint"])
                    )
                elif mode in {"keep_collect", "matched_keep"}:
                    route = self.action.RouteName.KEEP_ORIGINAL
                    waypoint = (0.0, 0.0)
                else:
                    arm = ARM_V1 if mode == "learned_v1" else ARM_V2
                    policy = self.policy_v1 if arm == ARM_V1 else self.policy_v2
                    probabilities, predicted_waypoint = self._predict(
                        policy=policy,
                        context=prepared_data["contexts"][arm],
                    )
                    gate = self.policy_module.apply_successor_feature_v2_keep_gate(
                        official_shadow_action=shadow_action,
                        route_probabilities=probabilities,
                        waypoint_local_m=predicted_waypoint,
                        context_age=int(prepared_data["context_age"]),
                        recovery_active=bool(prepared_data["recovery_active"]),
                        blocked_parent=bool(prepared_data["blocked_parent"]),
                        blocked_root=bool(prepared_data["blocked_root"]),
                    )
                    gate_record = gate.to_dict()
                    gate_record["decision_index"] = decisions
                    gate_records.append(gate_record)
                    gate_reasons[gate.reason] += 1
                    route = gate.selected_route
                    waypoint = gate.selected_waypoint_local_m
                decision = self.action.RouteDecision(
                    decision_index=decisions,
                    route=route,
                    waypoint_local_m=tuple(waypoint),
                    episode_token=bridge.episode_token,
                    physical_step=prepared.physical_step,
                )
                try:
                    result = bridge.execute_decision(decision)
                except self.action.ParentRecoveryUnavailable:
                    if not learned:
                        raise
                    recover["root_recovery_unavailable_count"] += 1
                    failure_reason = "root_recovery_unavailable"
                    truncated = True
                    break
                decisions += 1
                requested = str(result.plan["requested_route"])
                route_counts[requested] += 1
                for action in result.executed_actions:
                    action_counts[readiness.ACTION_NAMES[int(action)]] += 1
                    executed_actions.append(int(action))
                for primitive in result.primitive_feedback:
                    if not isinstance(primitive, Mapping):
                        raise ProtocolError("successor primitive feedback must be an object")
                    realized_path_m += readiness._finite(
                        primitive.get("realized_path_m"),
                        "primitive.realized_path_m",
                        minimum=0.0,
                    )
                    collision = primitive.get("collision")
                    if type(collision) is not bool:
                        raise ProtocolError("successor primitive collision must be boolean")
                    collision_count += int(collision)
                actual_recover = (
                    result.plan["executed_route"]
                    == self.action.RouteName.RECOVER_PARENT.value
                )
                recover["recover_parent_count"] += int(actual_recover)
                root_keep = result.plan.get("source") == "root_keep_fallback"
                recover["root_keep_fallback_count"] += int(root_keep)
                if prepared_data["blocked_parent"]:
                    recover["eligible_blocked_count"] += 1
                    recover["covered_next_decision_count"] += 1
                    recover["alignment_miss_count"] += int(not actual_recover)
                terminated = bool(result.terminated)
                truncated = bool(result.truncated)
            if not (terminated or truncated):
                exhausted = True

            if mode in {"expert_collect", "keep_collect", "teacher_forced"}:
                return {
                    "rows": deduplicate_paired_rows(rows),
                    "episode_id": str(spec["episode_id"]),
                    "scene": str(spec["scene"]),
                    "decision_count": decisions,
                    "physical_steps": len(executed_actions),
                    "exhausted": exhausted,
                    "recover": recover,
                }
            if not (terminated or truncated):
                raise ProtocolError("development episode did not end within 160 primitives")
            if failure_reason == "root_recovery_unavailable":
                current_distance = readiness._finite(
                    prepared_data["expert"].get("distance_to_goal_m"),
                    "root recovery current distance",
                    minimum=0.0,
                )
                terminal = {
                    "success": 0.0,
                    "spl": 0.0,
                    "path_length": realized_path_m,
                    "collision_count": float(collision_count),
                    "distance_to_goal": current_distance,
                }
                terminal_metrics_source = "current_state_at_root_recovery_failure"
            else:
                terminal = self.ceiling._terminal_metrics(env, None)
                terminal_metrics_source = "environment_terminal"
            return {
                "episode_id": str(spec["episode_id"]),
                "scene": str(spec["scene"]),
                "terminated": terminated,
                "truncated": truncated,
                "failure_reason": failure_reason,
                "terminal_metrics_source": terminal_metrics_source,
                "metrics": terminal,
                "route_counts": dict(route_counts),
                "action_counts": dict(action_counts),
                "executed_actions": executed_actions,
                "successful_stop": bool(
                    executed_actions
                    and executed_actions[-1] == self.action.STOP_ACTION
                    and terminated
                    and failure_reason is None
                    and float(terminal["success"]) > 0.0
                ),
                "recover": recover,
                "gate": {
                    "decision_count": len(gate_records),
                    "accepted_override_count": sum(
                        bool(record["accepted"]) for record in gate_records
                    ),
                    "reason_counts": dict(gate_reasons),
                    "records": gate_records,
                },
            }
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()

    @staticmethod
    def _aggregate_learned(episodes: Sequence[Mapping]) -> dict:
        aggregate = readiness.CanonicalReadinessProvider._aggregate(episodes)
        recover_keys = (
            "eligible_blocked_count",
            "covered_next_decision_count",
            "recover_parent_count",
            "alignment_miss_count",
            "root_recovery_unavailable_count",
            "root_keep_fallback_count",
        )
        recover = {
            key: sum(int(episode["recover"][key]) for episode in episodes)
            for key in recover_keys
        }
        accepted = [int(episode["gate"]["accepted_override_count"]) for episode in episodes]
        reasons = Counter()
        for episode in episodes:
            reasons.update(episode["gate"]["reason_counts"])
        aggregate.update(
            {
                "accepted_override_count": sum(accepted),
                "accepted_override_episode_count": sum(value > 0 for value in accepted),
                "successful_episode_with_accepted_override_count": sum(
                    accepted[index] > 0
                    and float(episode["metrics"]["success"]) > 0.0
                    for index, episode in enumerate(episodes)
                ),
                "gate_reason_counts": dict(reasons),
                "recover": recover,
                "forward_nonstop_fraction": _forward_nonstop_fraction(
                    aggregate, "learned aggregate"
                ),
            }
        )
        return aggregate

    def _save_candidate(
        self, *, arm: str, policy, manifest: Mapping,
        expert_dataset_sha256: str, combined_dataset_sha256: str,
    ) -> dict:
        import torch

        version = 1 if arm == ARM_V1 else 2
        binding = self.policy_module.successor_feature_checkpoint_binding(version)
        self.policy_module.require_successor_feature_checkpoint_binding(
            binding, expected_schema_version=version
        )
        output_parent = Path(self.args.output).expanduser().absolute().parent
        output_parent.mkdir(parents=True, exist_ok=True)
        path = output_parent / f"bc_candidate_{arm}.pt"
        if path.exists():
            raise FileExistsError(f"refusing to replace candidate checkpoint: {path}")
        policy.bc_ready.zero_()
        torch.save(
            {
                "schema_name": SCHEMA_NAME,
                "schema_version": SCHEMA_VERSION,
                "arm": arm,
                "bc_ready": 0,
                "feature_schema_binding": binding,
                "policy_state_dict": policy.state_dict(),
                "manifest_sha256": readiness._json_hash(manifest),
                "expert_dataset_sha256": expert_dataset_sha256,
                "combined_dataset_sha256": combined_dataset_sha256,
            },
            path,
        )
        return {
            "path": str(path),
            "bc_ready": 0,
            "feature_schema_binding": binding,
            "policy_state_sha256": readiness._module_state_hash(policy),
            "file_sha256": readiness._file_sha256(path),
        }

    def execute(
        self, *, manifest: Mapping, contract: Mapping, preflight: Mapping,
        args: argparse.Namespace,
    ) -> Mapping:
        del preflight, args
        self._load_once()
        training = contract["training"]
        collection_protocol = contract["protocol"]["collection"]
        expert_rows = []
        expert_traces = []
        for spec in contract["train"]:
            result = self._run_episode(
                spec=spec,
                mode="expert_collect",
                decision_budget=int(
                    collection_protocol["expert"]["decision_budget_per_episode"]
                ),
                physical_budget=int(
                    collection_protocol["expert"]["physical_budget_per_episode"]
                ),
                collect_rows=True,
            )
            expert_rows.extend(result["rows"])
            expert_traces.append({key: value for key, value in result.items() if key != "rows"})
        expert_rows = deduplicate_paired_rows(expert_rows)
        if self._pretraining_feature_guard is None or not self._pretraining_feature_guard["passed"]:
            raise ProtocolError("Feature-v2 first-generate pretraining guard did not pass")
        if self._optimizer_steps != 0:
            raise ProtocolError("optimizer advanced before fixed paired collection completed")
        keep_rows = []
        keep_traces = []
        for spec in contract["train"]:
            result = self._run_episode(
                spec=spec,
                mode="keep_collect",
                decision_budget=int(
                    collection_protocol["exact_keep"]["decision_budget_per_episode"]
                ),
                physical_budget=int(
                    collection_protocol["exact_keep"]["physical_budget_per_episode"]
                ),
                collect_rows=True,
            )
            keep_rows.extend(result["rows"])
            keep_traces.append({key: value for key, value in result.items() if key != "rows"})
        keep_rows = deduplicate_paired_rows(keep_rows)
        combined = deduplicate_paired_rows([*expert_rows, *keep_rows])
        if self.policy_v1 is None or self.policy_v2 is None:
            raise ProtocolError("paired collection did not initialize both policies")
        if [row["state_hash"] for row in combined] != [
            row["state_hash"] for row in deduplicate_paired_rows(combined)
        ]:
            raise ProtocolError("paired dataset order drifted")

        training_results = {}
        optimizers = {}
        for arm, policy in ((ARM_V1, self.policy_v1), (ARM_V2, self.policy_v2)):
            warmup, optimizer = readiness.train_successor_bc_phase(
                policy=policy,
                policy_module=self.policy_module,
                rows=project_arm_rows(expert_rows, arm),
                training=training,
                updates=int(training["warmup_updates"]),
                seed=int(contract["seed"]),
            )
            training_results[arm] = {"warmup": warmup}
            optimizers[arm] = optimizer
            self._optimizer_steps += int(warmup["updates"])
        for arm, policy in ((ARM_V1, self.policy_v1), (ARM_V2, self.policy_v2)):
            final, _optimizer = readiness.train_successor_bc_phase(
                policy=policy,
                policy_module=self.policy_module,
                rows=project_arm_rows(combined, arm),
                training=training,
                updates=int(training["final_updates"]),
                seed=int(contract["seed"]) + 1,
                optimizer=optimizers[arm],
            )
            training_results[arm]["final"] = final
            self._optimizer_steps += int(final["updates"])
        if (
            training_results[ARM_V1]["warmup"]["dataset_sha256"]
            != training_results[ARM_V2]["warmup"]["dataset_sha256"]
            or training_results[ARM_V1]["final"]["dataset_sha256"]
            != training_results[ARM_V2]["final"]["dataset_sha256"]
        ):
            raise ProtocolError("paired arms trained on different state identities or labels")
        candidates = {
            arm: self._save_candidate(
                arm=arm,
                policy=policy,
                manifest=manifest,
                expert_dataset_sha256=training_results[arm]["warmup"][
                    "dataset_sha256"
                ],
                combined_dataset_sha256=training_results[arm]["final"][
                    "dataset_sha256"
                ],
            )
            for arm, policy in ((ARM_V1, self.policy_v1), (ARM_V2, self.policy_v2))
        }

        evaluation = contract["protocol"]["evaluation"]
        teacher_records = {arm: [] for arm in ARMS}
        teacher_episodes = {arm: [] for arm in ARMS}
        for spec in contract["development"]:
            result = self._run_episode(
                spec=spec,
                mode="teacher_forced",
                decision_budget=int(
                    evaluation["teacher_forced_expert"][
                        "decision_budget_per_episode"
                    ]
                ),
                physical_budget=int(
                    evaluation["teacher_forced_expert"][
                        "physical_budget_per_episode"
                    ]
                ),
                collect_rows=True,
            )
            for arm in ARMS:
                records = self._score_teacher_rows(result["rows"], arm)
                teacher_records[arm].extend(records)
                teacher_episodes[arm].append(
                    {
                        "episode_id": str(spec["episode_id"]),
                        "scene": str(spec["scene"]),
                        "decision_count": int(result["decision_count"]),
                        "physical_steps": int(result["physical_steps"]),
                        "exhausted": bool(result["exhausted"]),
                        "unique_state_count": len(result["rows"]),
                        "metrics": causal.summarize_teacher_forced(records),
                    }
                )
        teacher = {
            arm: {
                "metrics": causal.summarize_teacher_forced(teacher_records[arm]),
                "episodes": teacher_episodes[arm],
            }
            for arm in ARMS
        }
        keep_episodes = [
            self._run_episode(
                spec=spec,
                mode="matched_keep",
                decision_budget=int(
                    evaluation["matched_exact_keep"]["decision_budget_per_episode"]
                ),
                physical_budget=int(
                    evaluation["matched_exact_keep"]["physical_budget_per_episode"]
                ),
                collect_rows=False,
            )
            for spec in contract["development"]
        ]
        learned_episodes = {
            arm: [
                self._run_episode(
                    spec=spec,
                    mode="learned_v1" if arm == ARM_V1 else "learned_v2",
                    decision_budget=int(
                        evaluation["learned_arm"]["decision_budget_per_episode"]
                    ),
                    physical_budget=int(
                        evaluation["learned_arm"]["physical_budget_per_episode"]
                    ),
                    collect_rows=False,
                )
                for spec in contract["development"]
            ]
            for arm in ARMS
        }
        navigation = {
            "matched_keep": self._aggregate(keep_episodes),
            **{
                arm: self._aggregate_learned(learned_episodes[arm])
                for arm in ARMS
            },
        }
        labels = Counter(row["route_label"] for row in combined)
        value_head_frozen = all(
            training_results[arm][phase]["value_head_hash_before"]
            == training_results[arm][phase]["value_head_hash_after"]
            for arm in ARMS
            for phase in ("warmup", "final")
        )
        return {
            "candidates": candidates,
            "candidate_bc_ready": {arm: 0 for arm in ARMS},
            "paired_dataset_sha256": training_results[ARM_V1]["final"][
                "dataset_sha256"
            ],
            "train_label_counts": {
                route: int(labels[route]) for route in readiness.ROUTES
            },
            "split_read_counts": {
                "train": int(self._split_reads["train"]),
                "val_seen": 0,
                "val_unseen": 0,
            },
            "teacher_forced": teacher,
            "navigation": navigation,
            "training": training_results,
            "collection": {
                "expert": expert_traces,
                "exact_keep": keep_traces,
                "expert_unique_rows": len(expert_rows),
                "exact_keep_unique_rows": len(keep_rows),
                "combined_unique_rows": len(combined),
            },
            "episodes": {
                "matched_keep": keep_episodes,
                **learned_episodes,
            },
            "pretraining_feature_guard": dict(self._pretraining_feature_guard),
            "feature_pair_counts": dict(self._feature_pair_counts),
            "value_head_frozen": value_head_frozen,
            "guards": {
                "model_load_count": int(self._model_load_count),
                "single_process": True,
                "train_only": True,
                "optimizer_steps_per_arm": int(training["updates_per_arm"]),
                "optimizer_steps_total": int(self._optimizer_steps),
                "ppo_updates": 0,
                "promotion_attempts": 0,
                "retune_attempts": 0,
            },
        }


def validate_feature_v2_ab_payload(payload: object, contract: Mapping) -> dict:
    if not isinstance(payload, Mapping):
        raise ProtocolError("feature-v2 A/B provider returned a non-object")
    result = dict(payload)
    if result.get("candidate_bc_ready") != {ARM_V1: 0, ARM_V2: 0}:
        raise ProtocolError("feature-v2 A/B candidates must remain bc_ready=0")
    candidates = result.get("candidates")
    if not isinstance(candidates, Mapping) or set(candidates) != set(ARMS):
        raise ProtocolError("feature-v2 A/B must save exactly two candidate checkpoints")
    normalized_candidates = {}
    for arm, version in ((ARM_V1, 1), (ARM_V2, 2)):
        row = candidates[arm]
        if not isinstance(row, Mapping):
            raise ProtocolError(f"candidate {arm} evidence is invalid")
        path = Path(str(row.get("path", ""))).expanduser().resolve(strict=True)
        if not path.is_file() or row.get("bc_ready") not in (0, 0.0, False):
            raise ProtocolError(f"candidate {arm} is missing or already ready")
        binding = row.get("feature_schema_binding")
        if (
            not isinstance(binding, Mapping)
            or binding.get("successor_feature_schema_version") != version
        ):
            raise ProtocolError(f"candidate {arm} feature schema binding drifted")
        digest = readiness._sha(
            binding.get("successor_feature_schema_sha256"),
            f"candidate.{arm}.feature_schema_sha256",
        )
        if digest != EXPECTED_FEATURE_SCHEMA_DIGESTS[version]:
            raise ProtocolError(f"candidate {arm} feature schema digest drifted")
        file_sha = readiness._file_sha256(path)
        if row.get("file_sha256") != file_sha:
            raise ProtocolError(f"candidate {arm} file SHA-256 drifted")
        normalized_candidates[arm] = {
            "path": str(path),
            "bc_ready": 0,
            "feature_schema_binding": {
                "successor_feature_schema_version": version,
                "successor_feature_schema_sha256": digest,
            },
            "policy_state_sha256": readiness._sha(
                row.get("policy_state_sha256"), f"candidate.{arm}.policy_state_sha256"
            ),
            "file_sha256": file_sha,
        }
    if (
        normalized_candidates[ARM_V1]["feature_schema_binding"][
            "successor_feature_schema_sha256"
        ]
        == normalized_candidates[ARM_V2]["feature_schema_binding"][
            "successor_feature_schema_sha256"
        ]
    ):
        raise ProtocolError("Feature-v1/v2 checkpoint schema digests must differ")
    result["candidates"] = normalized_candidates

    expected_guards = {
        "model_load_count": 1,
        "single_process": True,
        "train_only": True,
        "optimizer_steps_per_arm": 2000,
        "optimizer_steps_total": 4000,
        "ppo_updates": 0,
        "promotion_attempts": 0,
        "retune_attempts": 0,
    }
    if result.get("guards") != expected_guards:
        raise ProtocolError("feature-v2 A/B execution guards drifted")
    if result.get("split_read_counts") != {
        "train": EXPECTED_TRAIN_READS,
        "val_seen": 0,
        "val_unseen": 0,
    }:
        raise ProtocolError("feature-v2 A/B split-read accounting drifted")
    pretraining_guard = result.get("pretraining_feature_guard")
    if (
        not isinstance(pretraining_guard, Mapping)
        or pretraining_guard.get("passed") is not True
        or pretraining_guard.get("optimizer_steps_before_guard") != 0
        or pretraining_guard.get("adapter_feature_v2_opt_in") is not True
        or pretraining_guard.get("default_adapter_feature_v2_opt_in") is not False
        or pretraining_guard.get("feature_v1_dim") != 7173
        or pretraining_guard.get("feature_v2_dim") != 3604
        or pretraining_guard.get("feature_v2_schema_version") != 2
        or any(
            pretraining_guard.get(key) is not True
            for key in (
                "same_frame_ref",
                "same_physical_step",
                "same_reset_ordinal",
                "same_episode_generation",
                "both_fresh",
            )
        )
    ):
        raise ProtocolError("feature-v2 pretraining first-generate guard is incomplete")
    v2_schema_digest = readiness._sha(
        pretraining_guard.get("feature_v2_schema_sha256"),
        "pretraining_feature_guard.feature_v2_schema_sha256",
    )
    if (
        normalized_candidates[ARM_V2]["feature_schema_binding"][
            "successor_feature_schema_sha256"
        ]
        != v2_schema_digest
    ):
        raise ProtocolError("Feature-v2 candidate schema differs from live first generate")
    readiness._sha(
        pretraining_guard.get("contextual_hidden_sha256"),
        "pretraining_feature_guard.contextual_hidden_sha256",
    )
    pair_counts = result.get("feature_pair_counts")
    if (
        not isinstance(pair_counts, Mapping)
        or readiness._strict_int(pair_counts.get("decision_states"), "pair.decision_states", minimum=1)
        < readiness._strict_int(
            pair_counts.get("v1_latch_updates"), "pair.v1_latch_updates", minimum=1
        )
        or readiness._strict_int(
            pair_counts.get("fresh_context_states"), "pair.fresh_context_states", minimum=1
        )
        + readiness._strict_int(
            pair_counts.get("stale_context_states"), "pair.stale_context_states"
        )
        != pair_counts.get("decision_states")
    ):
        raise ProtocolError("feature pair cadence accounting drifted")
    if result.get("value_head_frozen") is not True:
        raise ProtocolError("feature-v2 A/B changed a frozen value head")
    label_counts = result.get("train_label_counts")
    if not isinstance(label_counts, Mapping) or set(label_counts) != set(readiness.ROUTES):
        raise ProtocolError("feature-v2 A/B train labels are incomplete")
    if any(
        readiness._strict_int(label_counts[route], f"train_label_counts.{route}") < 1
        for route in readiness.ROUTES
    ):
        raise ProtocolError("feature-v2 A/B training lacks one route label")

    training = result.get("training")
    if not isinstance(training, Mapping) or set(training) != set(ARMS):
        raise ProtocolError("feature-v2 A/B training evidence lacks one arm")
    dataset_hashes = {}
    for arm in ARMS:
        phases = training[arm]
        if not isinstance(phases, Mapping) or set(phases) != {"warmup", "final"}:
            raise ProtocolError(f"feature-v2 A/B {arm} training phases drifted")
        if phases["warmup"].get("updates") != 500 or phases["final"].get("updates") != 1500:
            raise ProtocolError(f"feature-v2 A/B {arm} optimizer schedule drifted")
        dataset_hashes[arm] = (
            phases["warmup"].get("dataset_sha256"),
            phases["final"].get("dataset_sha256"),
        )
        for digest in dataset_hashes[arm]:
            readiness._sha(digest, f"training.{arm}.dataset_sha256")
    if dataset_hashes[ARM_V1] != dataset_hashes[ARM_V2]:
        raise ProtocolError("feature-v2 A/B arms were not trained on paired labels")
    if result.get("paired_dataset_sha256") != dataset_hashes[ARM_V1][1]:
        raise ProtocolError("feature-v2 A/B paired dataset digest drifted")

    teacher = result.get("teacher_forced")
    expected_ids = [row["episode_id"] for row in contract["development"]]
    if not isinstance(teacher, Mapping) or set(teacher) != set(ARMS):
        raise ProtocolError("feature-v2 A/B teacher evidence lacks one arm")
    for arm in ARMS:
        row = teacher[arm]
        if (
            not isinstance(row, Mapping)
            or not isinstance(row.get("metrics"), Mapping)
            or not isinstance(row.get("episodes"), list)
            or [str(item.get("episode_id")) for item in row["episodes"]]
            != expected_ids
        ):
            raise ProtocolError(f"feature-v2 A/B {arm} teacher evidence is incomplete")
    navigation = result.get("navigation")
    if not isinstance(navigation, Mapping) or set(navigation) != {
        "matched_keep",
        ARM_V1,
        ARM_V2,
    }:
        raise ProtocolError("feature-v2 A/B navigation arms drifted")
    matched_keep = readiness._validate_nav_metrics(
        navigation["matched_keep"], role="matched_keep"
    )
    if matched_keep["episode_count"] != EXPECTED_EPISODES:
        raise ProtocolError("feature-v2 A/B matched KEEP episode count drifted")
    for arm in ARMS:
        nav = readiness._validate_nav_metrics(navigation[arm], role=arm)
        if nav["episode_count"] != EXPECTED_EPISODES:
            raise ProtocolError(f"feature-v2 A/B {arm} episode count drifted")
        extras = navigation[arm]
        if not isinstance(extras.get("recover"), Mapping):
            raise ProtocolError(f"feature-v2 A/B {arm} recovery evidence is missing")
        readiness._strict_int(extras.get("accepted_override_count"), f"{arm}.accepted")
        readiness._strict_int(
            extras.get("accepted_override_episode_count"), f"{arm}.accepted_episodes"
        )
        readiness._strict_int(
            extras.get("successful_episode_with_accepted_override_count"),
            f"{arm}.successful_accepted_episodes",
        )
    episodes = result.get("episodes")
    if not isinstance(episodes, Mapping) or set(episodes) != {
        "matched_keep",
        ARM_V1,
        ARM_V2,
    }:
        raise ProtocolError("feature-v2 A/B episode evidence arms drifted")
    for role in ("matched_keep", ARM_V1, ARM_V2):
        rows = episodes[role]
        if (
            not isinstance(rows, list)
            or [str(row.get("episode_id")) for row in rows] != expected_ids
            or len({str(row.get("episode_id")) for row in rows})
            != EXPECTED_EPISODES
        ):
            raise ProtocolError(f"feature-v2 A/B {role} episode identities drifted")
    return result


def _provider_execute(provider, **kwargs) -> Mapping:
    function = getattr(provider, "execute", provider)
    result = function(**kwargs)
    if not isinstance(result, Mapping):
        raise ProtocolError("feature-v2 A/B provider returned a non-object")
    return result


def run(
    args: argparse.Namespace,
    *,
    preflight_provider: Callable[[argparse.Namespace, Mapping, Mapping], Mapping]
    | None = None,
    ab_provider: FeatureV2ABProvider | Callable | None = None,
) -> dict:
    # Literal split rejection must precede every path read, import, or output access.
    readiness.require_train_split(args.split)
    if args.stage not in {"preflight", "feature_v2_ab"}:
        raise ValueError("stage must be preflight or feature_v2_ab")
    output_path = Path(args.output).expanduser().absolute()
    if output_path.exists():
        raise FileExistsError(f"refusing to replace existing receipt: {output_path}")
    manifest_path = Path(args.manifest).expanduser().resolve(strict=True)
    readiness_manifest_path = Path(args.readiness_manifest).expanduser().resolve(
        strict=True
    )
    causal_manifest_path = Path(args.causal_manifest).expanduser().resolve(strict=True)
    if readiness._file_sha256(readiness_manifest_path) != EXPECTED_READINESS_MANIFEST_RAW_SHA256:
        raise ProtocolError("readiness manifest file bytes drifted")
    if readiness._file_sha256(causal_manifest_path) != EXPECTED_CAUSAL_MANIFEST_RAW_SHA256:
        raise ProtocolError("causal manifest file bytes drifted")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    readiness_manifest = json.loads(readiness_manifest_path.read_text(encoding="utf-8"))
    causal_manifest = json.loads(causal_manifest_path.read_text(encoding="utf-8"))
    contract = validate_feature_v2_ab_manifest(
        manifest,
        readiness_manifest=readiness_manifest,
        causal_manifest=causal_manifest,
    )
    if str(args.device) != "cuda:0":
        raise ProtocolError("frozen Feature-v2 A/B requires process-local cuda:0")
    implementation = verify_implementation(
        manifest, rlinf_root=Path(args.rlinf_root)
    )
    preflight_fn = preflight_provider or production_preflight
    preflight = dict(preflight_fn(args, manifest, contract))
    if preflight.get("status") != "pass":
        raise ProtocolError("feature-v2 A/B preflight did not pass")
    preflight["implementation"] = implementation
    receipt = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "stage": args.stage,
        "split": TRAIN_SPLIT,
        "manifest": {
            "path": str(manifest_path),
            "file_sha256": readiness._file_sha256(manifest_path),
            "canonical_sha256": readiness._json_hash(manifest),
        },
        "lineage": dict(manifest["lineage"]),
        "preflight": preflight,
        "claim_boundary": dict(manifest["claim_boundary"]),
        "guards": {
            "literal_train_only": True,
            "val_seen_reads": 0,
            "val_unseen_reads": 0,
            "ppo_updates": 0,
            "promotion_attempts": 0,
            "ttt_or_secondary_weights": False,
            "single_process": True,
        },
        "promotion": None,
    }
    if args.stage == "preflight":
        receipt.update(
            {
                "gate_pass": None,
                "next_action": "run_one_bounded_feature_v2_ab",
            }
        )
        receipt["receipt_sha256"] = readiness._json_hash(receipt)
        readiness._atomic_json(output_path, receipt)
        return receipt
    if ab_provider is None:
        ab_provider = CanonicalFeatureV2ABProvider(
            args=args,
            manifest=manifest,
            contract=contract,
            preflight=preflight,
            readiness_manifest=readiness_manifest,
        )
    payload = validate_feature_v2_ab_payload(
        _provider_execute(
            ab_provider,
            manifest=manifest,
            contract=contract,
            preflight=preflight,
            args=args,
        ),
        contract,
    )
    gate = adjudicate_feature_v2_ab(payload, thresholds=contract["thresholds"])
    receipt.update(
        {
            "result": payload,
            "gate": gate,
            "gate_pass": bool(gate["passed"]),
            "next_action": (
                "bounded_feature_v2_candidate_supported_await_explicit_next_stage"
                if gate["passed"]
                else "stop_and_run_result_to_claim_without_retune"
            ),
        }
    )
    receipt["receipt_sha256"] = readiness._json_hash(receipt)
    readiness._atomic_json(output_path, receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("preflight", "feature_v2_ab"), required=True
    )
    parser.add_argument("--split", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--readiness-manifest", type=Path, required=True)
    parser.add_argument("--causal-manifest", type=Path, required=True)
    parser.add_argument("--causal-receipt", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--rlinf-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--vision", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
