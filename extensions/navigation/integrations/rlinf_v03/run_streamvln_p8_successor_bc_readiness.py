"""Frozen train-only BC-readiness gate for the P8 StreamVLN successor.

This file owns the narrow gate between the accepted live ceiling and a later
CUDA worker/PPO smoke.  It deliberately has no validation-split or PPO path.
The CLI constructs the canonical live provider directly.  Tests may inject a
fake provider so protocol adjudication stays testable without Habitat/model
imports.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
from typing import Callable, Mapping, Protocol, Sequence


SCHEMA_NAME = "streamvln_p8_successor_bc_readiness"
SCHEMA_VERSION = 1
TRAIN_SPLIT = "train"
ROUTES = ("stop", "keep_original", "waypoint")
ACTION_NAMES = ("STOP", "FORWARD", "LEFT", "RIGHT")
EXPECTED_EPISODES_PER_ROLE = 8
REQUIRED_TRAIN_LABELS = frozenset(ROUTES)
CANONICAL_PACKAGE = "rlinf.envs.streamvln"
REQUIRED_MODEL_PYTHON = "/opt/conda/envs/streamvln/bin/python"
REQUIRED_IMPLEMENTATION_DEPLOYMENTS = frozenset({
    "integrations/rlinf_v03/run_streamvln_p8_successor_bc_readiness.py",
    "integrations/rlinf_v03/run_streamvln_p8_successor_ceiling.py",
    "rlinf/envs/__init__.py",
    "rlinf/envs/vln/__init__.py",
    "rlinf/envs/vln/vln_env.py",
    "rlinf/envs/streamvln/__init__.py",
    "rlinf/envs/streamvln/streamvln_env.py",
    "rlinf/envs/streamvln/streamvln_executor_contract.py",
    "rlinf/envs/streamvln/streamvln_chunk_ppo.py",
    "rlinf/envs/streamvln/streamvln_p8_official_frame.py",
    "rlinf/envs/streamvln/streamvln_p8_official_keep.py",
    "rlinf/envs/streamvln/streamvln_p8_official_adapter.py",
    "rlinf/envs/streamvln/streamvln_p8_successor_action.py",
    "rlinf/envs/streamvln/streamvln_p8_successor_env.py",
    "rlinf/envs/streamvln/streamvln_p8_successor_executor.py",
    "rlinf/envs/streamvln/streamvln_p8_successor_protocol.py",
    "rlinf/envs/streamvln/streamvln_p8_successor_policy.py",
    "rlinf/envs/streamvln/streamvln_p8_successor_runtime.py",
    "rlinf/envs/streamvln/streamvln_p8_topology.py",
    "rlinf/envs/streamvln/streamvln_topology.py",
    "scripts/habitat_r2r_service.py",
})
class ProtocolError(RuntimeError):
    """A fail-closed scientific-protocol violation."""


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _json_hash(value: object) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path_value: str | Path, value: Mapping) -> None:
    path = Path(path_value).expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to replace existing receipt: {path}")
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, newline="\n"
    ) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_canonical_json(path_value: str | Path, value: Mapping) -> Path:
    """Write bytes whose file SHA-256 equals ``_json_hash(value)``."""

    path = Path(path_value).expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to replace existing evidence: {path}")
    payload = _json_bytes(value)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def require_train_split(value: object) -> str:
    """First gate: callers must invoke this before any path or import access."""

    if type(value) is not str or value != TRAIN_SPLIT:
        raise ValueError("BC readiness permits only the literal train split")
    return value


def _strict_int(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ProtocolError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value: object, name: str, *, minimum: float | None = None) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ProtocolError(f"{name} must be finite")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ProtocolError(f"{name} must be >= {minimum}")
    return result


def _sha(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProtocolError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _identity(row: object, name: str) -> dict:
    if not isinstance(row, Mapping):
        raise ProtocolError(f"{name} identity must be an object")
    index = _strict_int(row.get("index"), f"{name}.index")
    episode_id = row.get("episode_id")
    scene = row.get("scene")
    if not isinstance(episode_id, str) or not episode_id:
        raise ProtocolError(f"{name}.episode_id must be nonempty")
    if not isinstance(scene, str) or not scene:
        raise ProtocolError(f"{name}.scene must be nonempty")
    return {"index": index, "episode_id": episode_id, "scene": scene}


def validate_protocol_manifest(manifest: object) -> dict:
    """Validate the authoritative successor manifest, not the old learner."""

    if not isinstance(manifest, Mapping):
        raise ProtocolError("readiness manifest root must be an object")
    if (
        manifest.get("schema_name") != "streamvln_p8_successor_bc_readiness_manifest"
        or manifest.get("schema_version") != 1
        or manifest.get("stage") != "P8 successor train-only BC readiness"
    ):
        raise ProtocolError("unsupported successor readiness manifest")
    source_contract = manifest.get("source_contract")
    source = manifest.get("dataset")
    if not isinstance(source_contract, Mapping):
        raise ProtocolError("manifest source contract is missing")
    data_source = source_contract.get("data_source_manifest")
    if not isinstance(data_source, Mapping):
        raise ProtocolError("manifest does not bind its diagnostic data source")
    _sha(data_source.get("raw_sha256"), "data_source_manifest.raw_sha256")
    if (
        source_contract.get("original_streamvln_only") is not True
        or source_contract.get("ttt_or_secondary_post_training") is not False
    ):
        raise ProtocolError("successor source boundary permits secondary weights")
    if not isinstance(source, Mapping):
        raise ProtocolError("manifest dataset contract is missing")
    if source.get("allowed_splits") != [TRAIN_SPLIT]:
        raise ProtocolError("manifest permits a non-train split")
    if source.get("val_seen_read") is not False or source.get("val_unseen_read") is not False:
        raise ProtocolError("manifest records validation-split access")
    _sha(source.get("train_sha256"), "dataset.train_sha256")

    episodes = source.get("episodes")
    if not isinstance(episodes, Mapping):
        raise ProtocolError("manifest episode contract is missing")
    train = [
        _identity(row, f"expert_train[{index}]")
        for index, row in enumerate(episodes.get("expert_train", ()))
    ]
    dagger = [
        _identity(row, f"dagger_train[{index}]")
        for index, row in enumerate(episodes.get("dagger_train", ()))
    ]
    development = [
        _identity(row, f"train_development[{index}]")
        for index, row in enumerate(episodes.get("train_development", ()))
    ]
    if (
        len(train) != EXPECTED_EPISODES_PER_ROLE
        or len(dagger) != EXPECTED_EPISODES_PER_ROLE
        or len(development) != EXPECTED_EPISODES_PER_ROLE
    ):
        raise ProtocolError("readiness requires exactly 8 train and 8 development episodes")
    if dagger != train:
        raise ProtocolError("bounded DAgger must reuse the exact eight expert identities")
    for role, rows in (("train", train), ("development", development)):
        keys = {(row["index"], row["episode_id"], row["scene"]) for row in rows}
        scenes = {row["scene"] for row in rows}
        if len(keys) != len(rows) or len(scenes) != len(rows):
            raise ProtocolError(f"{role} identities and scenes must be unique")
    if {row["scene"] for row in train} & {row["scene"] for row in development}:
        raise ProtocolError("train and development scenes must be disjoint")

    training = manifest.get("training")
    if not isinstance(training, Mapping):
        raise ProtocolError("manifest training contract is missing")
    exact = {
        "expert_max_decisions_per_episode": 32,
        "dagger_max_decisions_per_episode": 20,
        "warmup_updates": 500,
        "final_updates": 1500,
        "update_schedule": "fixed_500_warmup_plus_1500_final",
        "batch_size": 32,
        "learning_rate": 0.0001,
        "value_head_trainable": False,
        "ppo_updates": 0,
    }
    for key, expected in exact.items():
        if training.get(key) != expected:
            raise ProtocolError(f"manifest training.{key} drifted")
    optimizer = training.get("optimizer")
    if not isinstance(optimizer, Mapping) or optimizer.get("type") != "AdamW":
        raise ProtocolError("readiness requires AdamW")
    if training.get("class_weights", {}).get("method") != "inverse_frequency":
        raise ProtocolError("readiness requires frozen inverse-frequency weights")
    loss = training.get("loss")
    if (
        not isinstance(loss, Mapping)
        or loss.get("waypoint", {}).get("type") != "SmoothL1Loss"
        or float(loss.get("waypoint", {}).get("beta", -1.0)) != 1.0
    ):
        raise ProtocolError("readiness requires waypoint SmoothL1 beta=1")
    seeds = manifest.get("seeds")
    if not isinstance(seeds, Mapping) or type(seeds.get("learner")) is not int:
        raise ProtocolError("manifest lacks the fixed learner seed")
    gate = manifest.get("gate")
    if not isinstance(gate, Mapping):
        raise ProtocolError("manifest lacks the readiness gate")
    if manifest.get("claim_boundary", {}).get("ppo") is not False:
        raise ProtocolError("readiness manifest permits PPO")
    feature = manifest.get("feature_contract")
    if (
        not isinstance(feature, Mapping)
        or feature.get("api")
        != "PinnedStreamVLNGenerationAdapter.extract_frozen_feature_snapshot"
        or feature.get("composition_api") != "compose_successor_features"
        or feature.get("side_effect_free") is not True
    ):
        raise ProtocolError("frozen feature contract drifted")
    implementation = manifest.get("implementation")
    records = implementation.get("files") if isinstance(implementation, Mapping) else None
    if not isinstance(records, list):
        raise ProtocolError("readiness implementation closure is missing")
    deployments = {
        record.get("deployment_relative_path")
        for record in records
        if isinstance(record, Mapping)
    }
    if deployments != REQUIRED_IMPLEMENTATION_DEPLOYMENTS or len(records) != len(deployments):
        raise ProtocolError("readiness implementation deployment closure drifted")
    if manifest.get("architecture", {}).get("feature_dim") != 7173:
        raise ProtocolError("successor readiness requires the pinned 7173-D feature")
    runtime = manifest.get("runtime")
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("model_python") != REQUIRED_MODEL_PYTHON
    ):
        raise ProtocolError("successor model Python runtime drifted")
    return {
        "train": train,
        "development": development,
        "training": dict(training),
        "gate": dict(gate),
        "seed": int(seeds["learner"]),
        "source": dict(source),
    }


def label_successor_route(
    *, expert_stop: object, official_shadow_action: object, expert_first_primitive: object
) -> str:
    """Three-way teacher label; RECOVER_PARENT remains an execution outcome."""

    if type(expert_stop) is not bool:
        raise ProtocolError("expert_stop must be boolean")
    shadow = _strict_int(official_shadow_action, "official_shadow_action")
    expert = _strict_int(expert_first_primitive, "expert_first_primitive")
    if shadow > 3 or expert > 3:
        raise ProtocolError("primitive labels must be in 0..3")
    if expert_stop:
        return "stop"
    if shadow == expert:
        return "keep_original"
    return "waypoint"


def state_sha256(
    *, episode_id: str, scene: str, decision_index: int, feature, topology_digest: str
) -> str:
    """Hash the exact causal state; tensor bytes are intentionally included."""

    if not episode_id or not scene or type(decision_index) is not int or decision_index < 0:
        raise ProtocolError("state hash identity is invalid")
    _sha(topology_digest, "topology_digest")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - production dependency
        raise RuntimeError("state hashing requires torch") from exc
    tensor = torch.as_tensor(feature).detach().cpu().contiguous().to(torch.float32)
    if tensor.ndim != 1 or not bool(torch.isfinite(tensor).all()):
        raise ProtocolError("state feature must be a finite 1-D tensor")
    digest = hashlib.sha256()
    digest.update(_json_bytes({
        "episode_id": episode_id,
        "scene": scene,
        "decision_index": decision_index,
        "topology_digest": topology_digest,
        "shape": list(tensor.shape),
        "dtype": "float32",
    }))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def deduplicate_rows(rows: Sequence[Mapping]) -> list[dict]:
    unique: dict[str, dict] = {}
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise ProtocolError(f"training row {index} must be an object")
        row = dict(raw)
        state_hash = _sha(row.get("state_hash"), f"rows[{index}].state_hash")
        route = row.get("route_label")
        if route not in ROUTES:
            raise ProtocolError(f"rows[{index}] has an invalid route label")
        signature = _json_hash({
            "route_label": route,
            "waypoint_target": row.get("waypoint_target"),
            "expert_first_primitive": row.get("expert_first_primitive"),
        })
        if state_hash in unique:
            previous = unique[state_hash]
            previous_signature = _json_hash({
                "route_label": previous.get("route_label"),
                "waypoint_target": previous.get("waypoint_target"),
                "expert_first_primitive": previous.get("expert_first_primitive"),
            })
            if signature != previous_signature:
                raise ProtocolError("duplicate causal state has conflicting labels")
            continue
        unique[state_hash] = row
    return list(unique.values())


def inverse_frequency_weights(rows: Sequence[Mapping]) -> dict[str, float]:
    rows = deduplicate_rows(rows)
    counts = Counter(str(row["route_label"]) for row in rows)
    if not rows or any(counts[route] == 0 for route in ROUTES):
        raise ProtocolError("inverse-frequency BC requires all three route labels")
    classes = len(ROUTES)
    return {route: len(rows) / (classes * counts[route]) for route in ROUTES}


def extract_policy_feature(
    *, adapter, policy_module, frame, instruction: str, official_shadow_action: int,
    physical_step: int,
):
    """Small adapter boundary for the side-effect-free frozen feature API."""

    snapshot = adapter.extract_frozen_feature_snapshot(
        frame=frame, instruction=instruction
    )
    feature = policy_module.compose_successor_features(
        frozen_feature=snapshot.feature,
        official_shadow_action=official_shadow_action,
        physical_step=physical_step,
        feature_physical_step=snapshot.physical_step,
    )
    return feature, snapshot


def _counts(value: object, *, names: Sequence[str], label: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{label} must be an object")
    return {name: _strict_int(value.get(name, 0), f"{label}.{name}") for name in names}


def _validate_nav_metrics(value: object, *, role: str) -> dict:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{role} metrics must be an object")
    episode_count = _strict_int(value.get("episode_count"), f"{role}.episode_count")
    success_count = _strict_int(value.get("success_count"), f"{role}.success_count")
    successful_stop_count = _strict_int(
        value.get("successful_stop_count"), f"{role}.successful_stop_count"
    )
    timeout_count = _strict_int(value.get("timeout_count"), f"{role}.timeout_count")
    root_recovery_unavailable_count = _strict_int(
        value.get("root_recovery_unavailable_count"),
        f"{role}.root_recovery_unavailable_count",
    )
    stop_count = _strict_int(value.get("stop_count"), f"{role}.stop_count")
    if (
        success_count > episode_count
        or successful_stop_count > stop_count
        or root_recovery_unavailable_count > episode_count
    ):
        raise ProtocolError(f"{role} counts exceed their domains")
    sr = _finite(value.get("sr"), f"{role}.sr", minimum=0.0)
    spl = _finite(value.get("spl"), f"{role}.spl", minimum=0.0)
    if sr > 1.0 or spl > 1.0:
        raise ProtocolError(f"{role} SR/SPL must be in [0,1]")
    action_counts = _counts(value.get("action_counts"), names=ACTION_NAMES, label=f"{role}.actions")
    route_counts = _counts(value.get("route_counts"), names=ROUTES, label=f"{role}.routes")
    return {
        "episode_count": episode_count,
        "success_count": success_count,
        "successful_stop_count": successful_stop_count,
        "timeout_count": timeout_count,
        "root_recovery_unavailable_count": root_recovery_unavailable_count,
        "stop_count": stop_count,
        "sr": sr,
        "spl": spl,
        "mean_navigation_error_m": _finite(
            value.get("mean_navigation_error_m"), f"{role}.mean_navigation_error_m", minimum=0.0
        ),
        "total_path_length_m": _finite(
            value.get("total_path_length_m"), f"{role}.total_path_length_m", minimum=0.0
        ),
        "collision_count": _finite(
            value.get("collision_count"), f"{role}.collision_count", minimum=0.0
        ),
        "action_counts": action_counts,
        "route_counts": route_counts,
    }


def adjudicate_readiness(payload: object, *, gate: Mapping) -> dict:
    """One-shot train-development gate.  No retry or result-aware selection."""

    if not isinstance(payload, Mapping):
        raise ProtocolError("readiness payload must be an object")
    keep = _validate_nav_metrics(payload.get("matched_keep"), role="matched_keep")
    learned = _validate_nav_metrics(payload.get("learned"), role="learned")
    label_counts = _counts(payload.get("train_label_counts"), names=ROUTES, label="train_labels")
    split_reads = _counts(
        payload.get("split_read_counts"),
        names=("train", "val_seen", "val_unseen"),
        label="split_reads",
    )
    recover = payload.get("recover")
    if not isinstance(recover, Mapping):
        raise ProtocolError("readiness recover evidence must be an object")
    eligible = _strict_int(recover.get("eligible_blocked_count"), "recover.eligible_blocked_count")
    covered = _strict_int(recover.get("covered_next_decision_count"), "recover.covered_next_decision_count")
    actual = _strict_int(recover.get("recover_parent_count"), "recover.recover_parent_count")
    alignment_miss = _strict_int(recover.get("alignment_miss_count"), "recover.alignment_miss_count")
    root_recovery_unavailable = _strict_int(
        recover.get("root_recovery_unavailable_count"),
        "recover.root_recovery_unavailable_count",
    )
    if root_recovery_unavailable != (
        keep["root_recovery_unavailable_count"]
        + learned["root_recovery_unavailable_count"]
    ):
        raise ProtocolError("aggregate root recovery evidence disagrees with role counts")
    per_episode = payload.get("learned_episode_route_counts")
    if not isinstance(per_episode, list) or len(per_episode) != EXPECTED_EPISODES_PER_ROLE:
        raise ProtocolError("learned route evidence must contain exactly 8 episodes")
    normalized_episode_routes = []
    for index, row in enumerate(per_episode):
        counts = _counts(row, names=ROUTES, label=f"learned_episode_routes[{index}]")
        normalized_episode_routes.append(counts)
    nonterminal = (
        learned["route_counts"]["keep_original"]
        + learned["route_counts"]["waypoint"]
    )
    dominance_ok = nonterminal > 0 and (
        max(
            learned["route_counts"]["keep_original"],
            learned["route_counts"]["waypoint"],
        )
        / nonterminal
        <= float(gate.get("nonterminal_route_fraction_max", 0.95))
    )
    primitive_classes = sum(count > 0 for count in learned["action_counts"].values())

    checks = {
        "episode_count": keep["episode_count"] == EXPECTED_EPISODES_PER_ROLE
        and learned["episode_count"] == EXPECTED_EPISODES_PER_ROLE,
        "learned_success": learned["success_count"]
        >= int(gate.get("learned_success_count_min", 1)),
        "learned_successful_stop": learned["successful_stop_count"]
        >= int(gate.get("successful_learned_stop_count_min", 1)),
        "keep_noninferiority": learned["success_count"]
        >= keep["success_count"] - int(gate.get("max_success_deficit_vs_keep", 1)),
        "three_train_labels": all(label_counts[route] > 0 for route in ROUTES),
        "two_learned_routes": sum(count > 0 for count in learned["route_counts"].values()) >= 2,
        "nonterminal_route_dominance": dominance_ok,
        "two_primitive_classes": primitive_classes
        >= int(gate.get("primitive_category_count_min", 2)),
        "real_recover_parent": actual
        >= int(gate.get("recover_parent_count_min", 1)),
        "recover_coverage": eligible >= 1 and covered == eligible,
        "recover_alignment": alignment_miss
        <= int(gate.get("recover_alignment_miss_max", 0)),
        "train_only": split_reads["train"] > 0
        and split_reads["val_seen"] == 0
        and split_reads["val_unseen"] == 0,
        "candidate_not_ready": payload.get("candidate_bc_ready") in (0, 0.0, False),
        "value_head_frozen": payload.get("value_head_frozen") is True,
        "single_dev_evaluation": payload.get("development_evaluation_count") == 1,
        "optimizer_steps": payload.get("optimizer_steps") == 2000,
        "root_recovery_available": (
            keep["root_recovery_unavailable_count"] == 0
            and learned["root_recovery_unavailable_count"] == 0
            and root_recovery_unavailable == 0
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "matched_keep": keep,
        "learned": learned,
        "train_label_counts": label_counts,
        "split_read_counts": split_reads,
        "recover": {
            "eligible_blocked_count": eligible,
            "covered_next_decision_count": covered,
            "recover_parent_count": actual,
            "alignment_miss_count": alignment_miss,
            "root_recovery_unavailable_count": root_recovery_unavailable,
        },
        "learned_episode_route_counts": normalized_episode_routes,
    }


def _load_receipt(path_value: object, *, expected_stage: str) -> dict:
    path = Path(str(path_value)).expanduser().resolve(strict=True)
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict):
        raise ProtocolError(f"{expected_stage} receipt root must be an object")
    embedded = _sha(receipt.get("receipt_sha256"), f"{expected_stage}.receipt_sha256")
    observed = _json_hash({key: value for key, value in receipt.items() if key != "receipt_sha256"})
    if embedded != observed:
        raise ProtocolError(f"{expected_stage} receipt internal hash mismatch")
    result = receipt.get("result")
    if (
        receipt.get("gate_pass") is not True
        or receipt.get("stage") != expected_stage
        or not isinstance(result, Mapping)
        or result.get("stage") != expected_stage
        or result.get("split") != TRAIN_SPLIT
    ):
        raise ProtocolError(f"{expected_stage} predecessor did not pass its train gate")
    preflight = receipt.get("preflight")
    if not isinstance(preflight, Mapping):
        raise ProtocolError(f"{expected_stage} predecessor lacks preflight bindings")
    return {
        "path": str(path),
        "file_sha256": _file_sha256(path),
        "receipt_sha256": embedded,
        "ceiling_manifest_raw_sha256": (
            receipt.get("ceiling_manifest_raw_sha256")
            or preflight.get("ceiling_manifest_raw_sha256")
        ),
        "release_manifest_raw_sha256": (
            receipt.get("release_manifest_raw_sha256")
            or preflight.get("release_manifest_raw_sha256")
        ),
        "receipt": receipt,
    }


def validate_predecessors(expert_path: object, keep_path: object) -> dict:
    expert = _load_receipt(expert_path, expected_stage="expert")
    keep = _load_receipt(keep_path, expected_stage="keep")
    for key in ("ceiling_manifest_raw_sha256", "release_manifest_raw_sha256"):
        if not expert[key] or expert[key] != keep[key]:
            raise ProtocolError(f"predecessor {key} binding mismatch")
    keep_bound = keep["receipt"].get("preflight", {}).get("expert_receipt")
    if not isinstance(keep_bound, Mapping):
        raise ProtocolError("KEEP predecessor lacks its expert binding")
    if keep_bound.get("file_sha256") != expert["file_sha256"]:
        raise ProtocolError("KEEP predecessor is bound to another expert receipt")
    expert.pop("receipt")
    keep.pop("receipt")
    return {"expert": expert, "keep": keep}


def _canonical_import(rlinf_root: Path, module_name: str):
    qualified = f"{CANONICAL_PACKAGE}.{module_name}"
    module = importlib.import_module(qualified)
    expected = (rlinf_root / "rlinf" / "envs" / "streamvln" / f"{module_name}.py").resolve(strict=True)
    actual = Path(str(getattr(module, "__file__", ""))).resolve(strict=True)
    if module.__name__ != qualified or actual != expected:
        raise ProtocolError(f"canonical import escaped rlinf_root: {qualified}")
    return module


def production_preflight(args: argparse.Namespace, manifest: Mapping, contract: Mapping) -> dict:
    """Read-only release/asset/receipt verification; never imports torch/Habitat."""

    expected_python = Path(manifest["runtime"]["model_python"]).resolve(strict=True)
    actual_python = Path(sys.executable).resolve(strict=True)
    if actual_python != expected_python:
        raise ProtocolError("readiness must run in the pinned StreamVLN Python")
    roots = {}
    for name in ("rlinf_root", "source_root", "checkpoint", "tokenizer", "vision", "asset_root"):
        path = Path(str(getattr(args, name))).expanduser().resolve(strict=True)
        if not path.is_dir():
            raise ProtocolError(f"--{name.replace('_', '-')} must be a directory")
        roots[name] = path
    root_text = str(roots["rlinf_root"])
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    release_manifest = Path(str(args.release_manifest)).expanduser().resolve(strict=True)
    release_value = json.loads(release_manifest.read_text(encoding="utf-8"))
    # Reuse the accepted ceiling verifier without rerunning Habitat/model code.
    ceiling = importlib.import_module("run_streamvln_p8_successor_ceiling")
    verified = ceiling._default_release_verifier(
        manifest=release_value,
        rlinf_root=roots["rlinf_root"],
        source_root=roots["source_root"],
        checkpoint=roots["checkpoint"],
        tokenizer=roots["tokenizer"],
        vision=roots["vision"],
    )
    predecessors = validate_predecessors(args.expert_receipt, args.keep_receipt)
    predecessor_contract = manifest["predecessors"]
    for role in ("expert", "keep"):
        expected = predecessor_contract[role]
        observed = predecessors[role]
        if (
            observed["file_sha256"] != expected["file_sha256"]
            or observed["receipt_sha256"] != expected["receipt_sha256"]
        ):
            raise ProtocolError(f"{role} receipt differs from authoritative manifest")
    if (
        predecessors["expert"]["ceiling_manifest_raw_sha256"]
        != predecessor_contract["ceiling_manifest_raw_sha256"]
        or predecessors["expert"]["release_manifest_raw_sha256"]
        != predecessor_contract["release_manifest_raw_sha256"]
    ):
        raise ProtocolError("predecessor protocol anchor differs from readiness manifest")
    if predecessors["expert"]["release_manifest_raw_sha256"] != _file_sha256(release_manifest):
        raise ProtocolError("release manifest differs from the accepted predecessors")
    train_path = roots["asset_root"] / "datasets" / "r2r" / "train" / "train.json.gz"
    if _file_sha256(train_path) != contract["source"]["train_sha256"]:
        raise ProtocolError("train dataset hash differs from the frozen manifest")
    source_record = manifest["source_contract"]["data_source_manifest"]
    source_manifest = roots["rlinf_root"] / source_record["relative_path"]
    if _file_sha256(source_manifest) != source_record["raw_sha256"]:
        raise ProtocolError("diagnostic data-source manifest bytes drifted")
    scene_observations = []
    expected_scenes = {
        row["scene"] for row in (*contract["train"], *contract["development"])
    }
    records = manifest["dataset"].get("selected_scene_assets")
    if not isinstance(records, list) or {row.get("scene") for row in records} != expected_scenes:
        raise ProtocolError("readiness manifest does not bind all 16 scene assets")
    for record in records:
        scene = record["scene"]
        observed = {"scene": scene}
        for kind in ("glb", "navmesh"):
            item = record[kind]
            path = roots["asset_root"] / Path(item["path"]).relative_to(
                Path(manifest["dataset"]["train_path"]).parents[3]
            )
            path = path.resolve(strict=True)
            if path.stat().st_size != int(item["bytes"]) or _file_sha256(path) != item["sha256"]:
                raise ProtocolError(f"scene {scene} {kind} bytes drifted")
            observed[kind] = {"path": str(path), "bytes": path.stat().st_size, "sha256": item["sha256"]}
        scene_observations.append(observed)
    implementation_observations = []
    for record in manifest["implementation"]["files"]:
        path = roots["rlinf_root"] / record["deployment_relative_path"]
        if path.stat().st_size != int(record["bytes"]) or _file_sha256(path) != record["sha256"]:
            raise ProtocolError(
                "readiness implementation drifted: "
                f"{record['deployment_relative_path']}"
            )
        implementation_observations.append(dict(record))
    return {
        "status": "pass",
        "split": TRAIN_SPLIT,
        "manifest_sha256": _json_hash(manifest),
        "release_manifest_raw_sha256": _file_sha256(release_manifest),
        "release_manifest_canonical_sha256": verified["anchor"],
        "predecessors": predecessors,
        "scene_assets": scene_observations,
        "implementation": implementation_observations,
        "paths": {
            **{key: str(value) for key, value in roots.items()},
            "model_python": str(expected_python),
        },
        "device": str(args.device),
        "guards": {
            "model_load_count": 0,
            "optimizer_steps": 0,
            "val_seen_reads": 0,
            "val_unseen_reads": 0,
            "ppo_updates": 0,
        },
    }


class ReadinessProvider(Protocol):
    def execute(self, *, manifest: Mapping, contract: Mapping, preflight: Mapping, args: argparse.Namespace) -> Mapping:
        ...

    def promote(self, *, evidence_digest: str, candidate_path: Path, promoted_path: Path) -> Mapping:
        ...


def _tensor_mapping_hash(value: Mapping[str, object]) -> str:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - production dependency
        raise RuntimeError("tensor hashing requires torch") from exc
    digest = hashlib.sha256()
    for key in sorted(value):
        tensor = torch.as_tensor(value[key]).detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(_json_bytes(list(tensor.shape)))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _stack_contexts(rows: Sequence[Mapping], *, device):
    import torch

    keys = tuple(rows[0]["forward_context"])
    if any(tuple(row["forward_context"]) != keys for row in rows):
        raise ProtocolError("BC forward-context schemas differ")
    result = {}
    for key in keys:
        values = [torch.as_tensor(row["forward_context"][key]).detach() for row in rows]
        result[key] = torch.stack(values, dim=0).unsqueeze(1).to(device=device)
    return result


def _module_state_hash(module, *, prefixes: tuple[str, ...] | None = None) -> str:
    state = module.state_dict()
    selected = {
        key: value
        for key, value in state.items()
        if prefixes is None or key.startswith(prefixes)
    }
    return _tensor_mapping_hash(selected)


def train_successor_bc_phase(
    *, policy, policy_module, rows: Sequence[Mapping], training: Mapping,
    updates: int, seed: int, optimizer=None,
) -> tuple[dict, object]:
    """FP32 route CE + waypoint NLL/SmoothL1 with a frozen value head."""

    import torch
    import torch.nn.functional as F

    rows = deduplicate_rows(rows)
    if not rows or type(updates) is not int or updates < 1:
        raise ProtocolError("BC phase requires unique rows and positive updates")
    class_weights = inverse_frequency_weights(rows)
    device = next(policy.parameters()).device
    named_parameters = dict(policy.named_parameters())
    for name, parameter in named_parameters.items():
        parameter.requires_grad_(not name.startswith("value_head."))
    trainable = [
        parameter for name, parameter in named_parameters.items()
        if not name.startswith("value_head.")
    ]
    if any(parameter.dtype != torch.float32 for parameter in trainable):
        raise ProtocolError("successor BC trainables must remain FP32")
    optimizer_contract = training["optimizer"]
    if optimizer is None:
        optimizer = torch.optim.AdamW(
            trainable,
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            betas=tuple(float(value) for value in optimizer_contract["betas"]),
            eps=float(optimizer_contract["eps"]),
            amsgrad=bool(optimizer_contract["amsgrad"]),
        )
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if optimizer_ids != {id(parameter) for parameter in trainable}:
        raise ProtocolError("AdamW parameter scope includes the frozen value head")

    route_index = {
        route.value: index for index, route in enumerate(policy_module.POLICY_ROUTES)
    }
    weight = torch.tensor(
        [class_weights[route.value] for route in policy_module.POLICY_ROUTES],
        dtype=torch.float32,
        device=device,
    )
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    batch_size = int(training["batch_size"])
    beta = float(training["loss"]["waypoint"]["beta"])
    value_before = _module_state_hash(policy, prefixes=("value_head.",))
    losses = []
    for _ in range(updates):
        indices = torch.randint(
            len(rows), (batch_size,), generator=generator
        ).tolist()
        batch = [rows[index] for index in indices]
        context = _stack_contexts(batch, device=device)
        route_targets = torch.tensor(
            [[route_index[row["route_label"]]] for row in batch],
            dtype=torch.int64,
            device=device,
        )
        waypoint_targets = torch.tensor(
            [[list(row["waypoint_target"])] for row in batch],
            dtype=torch.float32,
            device=device,
        )
        outputs = policy(
            forward_inputs=context,
            route_actions=route_targets,
            waypoint_targets=waypoint_targets,
            expert_waypoint_targets=waypoint_targets,
            compute_values=False,
        )
        route_loss = F.cross_entropy(
            outputs["raw_route_logits"].reshape(-1, len(policy_module.POLICY_ROUTES)),
            route_targets.reshape(-1),
            weight=weight,
        )
        waypoint_mask = route_targets.reshape(-1) == route_index["waypoint"]
        if bool(waypoint_mask.any()):
            prediction = outputs["waypoint_predictions"].reshape(-1, 2)[waypoint_mask]
            target = waypoint_targets.reshape(-1, 2)[waypoint_mask]
            waypoint_smooth_l1 = F.smooth_l1_loss(
                prediction, target, beta=beta, reduction="mean"
            )
            waypoint_nll = -outputs["expert_waypoint_logprobs"].reshape(-1)[
                waypoint_mask
            ].mean()
        else:
            waypoint_smooth_l1 = outputs["waypoint_predictions"].sum() * 0.0
            waypoint_nll = outputs["waypoint_predictions"].sum() * 0.0
        loss = route_loss + waypoint_nll + waypoint_smooth_l1
        if not bool(torch.isfinite(loss)):
            raise ProtocolError("BC loss became non-finite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(
            parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
            for parameter in trainable
        ):
            raise ProtocolError("BC gradient became non-finite")
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    value_after = _module_state_hash(policy, prefixes=("value_head.",))
    if value_before != value_after:
        raise ProtocolError("frozen value head changed during BC")
    return ({
        "updates": updates,
        "batch_size": batch_size,
        "row_count": len(rows),
        "dataset_sha256": _json_hash([
            {"state_hash": row["state_hash"], "route_label": row["route_label"]}
            for row in rows
        ]),
        "class_weights": class_weights,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "all_finite": all(math.isfinite(value) for value in losses),
        "value_head_hash_before": value_before,
        "value_head_hash_after": value_after,
    }, optimizer)


class CanonicalReadinessProvider:
    """Single-process, one-model-load live implementation of the frozen gate."""

    def __init__(
        self, *, args: argparse.Namespace, manifest: Mapping,
        contract: Mapping, preflight: Mapping,
    ) -> None:
        self.args = args
        self.manifest = manifest
        self.contract = contract
        self.preflight = preflight
        self.adapter = None
        self.policy = None
        self._candidate_path: Path | None = None
        self._model_load_count = 0
        self._split_reads = Counter({"train": 0, "val_seen": 0, "val_unseen": 0})

    def _load_once(self) -> None:
        if self.adapter is not None:
            return
        import torch

        rlinf_root = Path(self.preflight["paths"]["rlinf_root"])
        root_text = str(rlinf_root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        self.runtime = _canonical_import(rlinf_root, "streamvln_p8_successor_runtime")
        self.action = _canonical_import(rlinf_root, "streamvln_p8_successor_action")
        self.protocol = _canonical_import(rlinf_root, "streamvln_p8_successor_protocol")
        self.policy_module = _canonical_import(rlinf_root, "streamvln_p8_successor_policy")
        self.topology_module = _canonical_import(rlinf_root, "streamvln_p8_topology")
        self.adapter_module = _canonical_import(rlinf_root, "streamvln_p8_official_adapter")
        self.official_module = _canonical_import(rlinf_root, "streamvln_p8_official_keep")
        self.ceiling = importlib.import_module("run_streamvln_p8_successor_ceiling")
        release_path = Path(str(self.args.release_manifest)).expanduser().resolve(strict=True)
        release = json.loads(release_path.read_text(encoding="utf-8"))
        self.adapter = self.adapter_module.PinnedStreamVLNGenerationAdapter.from_released_checkpoint(
            model_path=self.preflight["paths"]["checkpoint"],
            tokenizer_path=self.preflight["paths"]["tokenizer"],
            vision_tower_path=self.preflight["paths"]["vision"],
            source_root=self.preflight["paths"]["source_root"],
            artifact_manifest=release,
            device=str(self.args.device),
            local_rank=0,
            env_count=1,
            model_env_id=0,
        )
        self._model_load_count = 1
        seed = int(self.contract["seed"])
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _run_cfg(self) -> dict:
        runtime = self.manifest["runtime"]
        return {
            "seed": int(self.contract["seed"]),
            "habitat_python": runtime["habitat_python"],
            "habitat_server": str(
                Path(self.preflight["paths"]["rlinf_root"])
                / "scripts" / "habitat_r2r_service.py"
            ),
            "asset_root": self.preflight["paths"]["asset_root"],
            "habitat_sim_python": runtime["habitat_sim_python"],
            "habitat_env_lib": runtime["habitat_env_lib"],
            "egl_vendor_json": runtime["egl_vendor_json"],
            "habitat_startup_timeout_s": float(runtime["habitat_startup_timeout_s"]),
        }

    @staticmethod
    def _identity_matches(observed: Mapping, expected: Mapping) -> bool:
        observed_scene = str(observed.get("scene_id", observed.get("scene", "")))
        return (
            str(observed.get("episode_id")) == expected["episode_id"]
            and expected["scene"] in observed_scene
        )

    def _ensure_policy(self, feature, topology_snapshot) -> None:
        if int(feature.numel()) != int(self.manifest["architecture"]["feature_dim"]):
            raise ProtocolError("live frozen feature differs from the pinned 7173-D contract")
        if self.policy is not None:
            if int(feature.numel()) != self.policy.feature_dim:
                raise ProtocolError("frozen feature dimension changed between frames")
            return
        topology_max_nodes = int(topology_snapshot.tensors.max_nodes)
        architecture = self.manifest["architecture"]
        self.policy = self.policy_module.P8SuccessorPolicy(
            feature_dim=int(feature.numel()),
            hidden_dim=int(architecture["hidden_dim"]),
            topology_max_nodes=topology_max_nodes,
            topology_model_dim=int(architecture["topology_model_dim"]),
            max_waypoint_m=float(architecture["max_waypoint_m"]),
        ).float().to(self.args.device)
        self.policy.bc_ready.zero_()

    def _teacher_sample(self, frame, shadow_action: int, context: Mapping, target) -> object:
        route = self.action.RouteName.STOP if target.stop else self.action.RouteName.WAYPOINT
        waypoint = (0.0, 0.0) if target.stop else tuple(target.waypoint_local_m)
        decision = self.action.RouteDecision(
            decision_index=frame.decision_index,
            route=route,
            waypoint_local_m=waypoint,
            episode_token=frame.episode_token,
            physical_step=frame.physical_step,
        )
        proposed = (
            (self.action.STOP_ACTION,)
            if target.stop
            else self.action.compile_metric_waypoint(waypoint)
        )
        return self.protocol.SuccessorDecisionSample(
            decision=decision,
            proposed_actions=tuple(proposed),
            logprob=0.0,
            value=0.0,
            official_shadow_action=shadow_action,
            forward_context={key: value.detach().cpu() for key, value in context.items()},
        )

    def _keep_sample(self, frame, shadow_action: int, context: Mapping) -> object:
        decision = self.action.RouteDecision(
            decision_index=frame.decision_index,
            route=self.action.RouteName.KEEP_ORIGINAL,
            official_action=shadow_action,
            episode_token=frame.episode_token,
            physical_step=frame.physical_step,
        )
        return self.protocol.SuccessorDecisionSample(
            decision=decision,
            proposed_actions=(shadow_action,),
            logprob=0.0,
            value=0.0,
            official_shadow_action=shadow_action,
            forward_context={key: value.detach().cpu() for key, value in context.items()},
        )

    def _learned_sample(
        self, frame, shadow_action: int, context: Mapping, *, deterministic: bool,
        expert_route, expert_waypoint,
    ) -> object:
        # Readiness evaluation must exercise learned logits while the candidate
        # buffer remains bc_ready=0.  Promotion is allowed only after this gate.
        import torch
        from torch.distributions import Categorical, Normal

        (
            _, _, _, _, route_logits, raw_mean, raw_log_std, values,
        ) = self.policy._distribution_terms(context)
        distribution = Categorical(logits=route_logits)
        route_index = (
            route_logits.argmax(dim=-1) if deterministic else distribution.sample()
        )
        route_logprob = distribution.log_prob(route_index)
        route = self.policy_module.POLICY_ROUTES[int(route_index.item())]
        if route is self.action.RouteName.WAYPOINT:
            raw = raw_mean if deterministic else Normal(raw_mean, raw_log_std.exp()).sample()
            waypoint_tensor = self.policy._waypoint_from_raw(raw)
            waypoint_logprob = self.policy._waypoint_logprob(raw, raw_mean, raw_log_std)
            waypoint = tuple(float(value) for value in waypoint_tensor.reshape(-1).tolist())
            proposed = tuple(self.action.compile_metric_waypoint(waypoint))
            logprob = route_logprob + waypoint_logprob
            official_action = None
        elif route is self.action.RouteName.KEEP_ORIGINAL:
            waypoint = (0.0, 0.0)
            proposed = (int(shadow_action),)
            logprob = route_logprob
            official_action = int(shadow_action)
        else:
            waypoint = (0.0, 0.0)
            proposed = (self.action.STOP_ACTION,)
            logprob = route_logprob
            official_action = None
        decision = self.action.RouteDecision(
            decision_index=frame.decision_index,
            route=route,
            waypoint_local_m=waypoint,
            official_action=official_action,
            episode_token=frame.episode_token,
            physical_step=frame.physical_step,
        )
        return self.protocol.SuccessorDecisionSample(
            decision=decision,
            proposed_actions=proposed,
            logprob=float(logprob.item()),
            value=float(values.item()),
            official_shadow_action=int(shadow_action),
            forward_context={key: value.detach().cpu().clone() for key, value in context.items()},
            expert_route_target=expert_route,
            expert_waypoint_target=expert_waypoint,
        )

    def _run_episode(
        self, *, spec: Mapping, mode: str, decision_budget: int,
        physical_budget: int, collect_rows: bool,
    ) -> dict:
        if mode not in ("expert", "dagger", "keep", "learned"):
            raise ValueError("unknown readiness episode mode")
        env = self.runtime.make_successor_env(
            self._run_cfg(), split=TRAIN_SPLIT,
            episode_index=int(spec["index"]), max_steps=int(physical_budget),
        )
        self._split_reads[TRAIN_SPLIT] += 1
        rows = []
        route_counts = Counter({route: 0 for route in ROUTES})
        action_counts = Counter({name: 0 for name in ACTION_NAMES})
        recover = {
            "eligible_blocked_count": 0,
            "covered_next_decision_count": 0,
            "recover_parent_count": 0,
            "alignment_miss_count": 0,
            "root_recovery_unavailable_count": 0,
        }
        executed_actions = []
        realized_path_m = 0.0
        collision_count = 0
        terminated = truncated = False
        exhausted = False
        failure_reason = None
        prepared_data: dict[str, object] = {}

        try:
            identity = None

            def official_driver_factory(env_id: int, selected: Mapping, episode_serial: int):
                session = self.official_module.OfficialEvalSession(
                    env_id=env_id,
                    instruction=selected["instruction_text"],
                    conjunction_rng=random.Random(
                        int(self.contract["seed"]) + int(spec["index"])
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
                    max_decisions=int(self.manifest["architecture"]["topology_max_nodes"]) - 1,
                    forward_step_m=0.25,
                    turn_degrees=15.0,
                ),
                official_driver_factory=official_driver_factory,
                exact_control=mode == "keep",
                require_prepared_decision_context=True,
            )
            bridge.reset_episode()
            identity = getattr(env, "identity", {})
            if not self._identity_matches(identity, spec):
                raise ProtocolError("live Habitat identity differs from readiness manifest")

            def context_builder(frame, shadow_action):
                expert = env.expert_waypoint(
                    float(self.manifest["executor"]["expert_lookahead_m"])
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
                feature, feature_snapshot = extract_policy_feature(
                    adapter=self.adapter,
                    policy_module=self.policy_module,
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
                    max_nodes=int(self.manifest["architecture"]["topology_max_nodes"]),
                )
                self._ensure_policy(feature, topology_snapshot)
                context = self.policy_module.make_successor_forward_context(
                    features=feature,
                    topology_snapshot=topology_snapshot,
                    policy_version=0,
                )
                label = label_successor_route(
                    expert_stop=expert["stop"],
                    official_shadow_action=int(shadow_action),
                    expert_first_primitive=int(expert_actions[0]),
                )
                target_waypoint = waypoint if label == "waypoint" else (0.0, 0.0)
                topology_values = {
                    key: value for key, value in context.items() if key != "features"
                }
                state_hash = state_sha256(
                    episode_id=str(spec["episode_id"]),
                    scene=str(spec["scene"]),
                    decision_index=int(bridge._next_decision_index),
                    feature=feature,
                    topology_digest=_tensor_mapping_hash(topology_values),
                )
                if collect_rows:
                    rows.append({
                        "state_hash": state_hash,
                        "route_label": label,
                        "waypoint_target": list(target_waypoint),
                        "expert_first_primitive": int(expert_actions[0]),
                        "episode_id": str(spec["episode_id"]),
                        "scene": str(spec["scene"]),
                        "decision_index": int(bridge._next_decision_index),
                        "forward_context": {
                            key: value.detach().cpu().clone()
                            for key, value in context.items()
                        },
                        "feature_audit": dict(feature_snapshot.feature_audit),
                    })
                blocked = int(bridge.topology.consecutive_blocked_forwards)
                current = int(bridge.topology.current_node)
                parent = bridge.topology.nodes[current].parent
                eligible = (
                    blocked
                    >= int(self.manifest["executor"]["blocked_forward_threshold"])
                    and parent is not None
                )
                prepared_data.clear()
                prepared_data.update({
                    "context": context,
                    "expert": expert,
                    "label": label,
                    "waypoint": waypoint,
                    "target_waypoint": target_waypoint,
                    "eligible": eligible,
                    "shadow_action": int(shadow_action),
                })
                return context

            decisions = 0
            while decisions < decision_budget and not (terminated or truncated):
                prepared = bridge.prepare_decision_context(context_builder)
                if prepared.policy_context is not prepared_data["context"]:
                    raise ProtocolError("prepared decision context identity drifted")
                expert = prepared_data["expert"]
                shadow_action = int(prepared_data["shadow_action"])
                if mode == "expert":
                    route = (
                        self.action.RouteName.STOP
                        if expert["stop"]
                        else self.action.RouteName.WAYPOINT
                    )
                    decision = self.action.RouteDecision(
                        decision_index=decisions,
                        route=route,
                        waypoint_local_m=(
                            (0.0, 0.0)
                            if expert["stop"]
                            else tuple(prepared_data["waypoint"])
                        ),
                        episode_token=bridge.episode_token,
                        physical_step=prepared.physical_step,
                    )
                elif mode == "keep":
                    decision = self.action.RouteDecision(
                        decision_index=decisions,
                        route=self.action.RouteName.KEEP_ORIGINAL,
                        episode_token=bridge.episode_token,
                        physical_step=prepared.physical_step,
                    )
                else:
                    label_route = self.action.RouteName(prepared_data["label"])
                    sample = self._learned_sample(
                        argparse.Namespace(
                            episode_token=bridge.episode_token,
                            physical_step=prepared.physical_step,
                            decision_index=decisions,
                        ),
                        shadow_action,
                        prepared_data["context"],
                        deterministic=mode == "learned",
                        expert_route=label_route,
                        expert_waypoint=(
                            tuple(prepared_data["target_waypoint"])
                            if label_route is self.action.RouteName.WAYPOINT
                            else None
                        ),
                    )
                    sampled = sample.decision
                    decision = self.action.RouteDecision(
                        decision_index=sampled.decision_index,
                        route=sampled.route,
                        waypoint_local_m=sampled.waypoint_local_m,
                        episode_token=sampled.episode_token,
                        physical_step=sampled.physical_step,
                    )
                try:
                    result = bridge.execute_decision(decision)
                except self.action.ParentRecoveryUnavailable:
                    if mode not in ("keep", "learned"):
                        raise
                    recover["root_recovery_unavailable_count"] += 1
                    failure_reason = "root_recovery_unavailable"
                    truncated = True
                    break
                decisions += 1
                requested = str(result.plan["requested_route"])
                route_counts[requested] += 1
                for action in result.executed_actions:
                    action_counts[ACTION_NAMES[int(action)]] += 1
                    executed_actions.append(int(action))
                for primitive in result.primitive_feedback:
                    if not isinstance(primitive, Mapping):
                        raise ProtocolError("successor primitive feedback must be an object")
                    realized_path_m += _finite(
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
                if actual_recover:
                    recover["recover_parent_count"] += 1
                if prepared_data["eligible"]:
                    recover["eligible_blocked_count"] += 1
                    recover["covered_next_decision_count"] += 1
                    recover["alignment_miss_count"] += int(not actual_recover)
                terminated = bool(result.terminated)
                truncated = bool(result.truncated)
            if not (terminated or truncated):
                exhausted = True

            if mode in ("expert", "dagger"):
                return {
                    "rows": deduplicate_rows(rows),
                    "episode_id": str(spec["episode_id"]),
                    "decision_count": decisions,
                    "physical_steps": len(executed_actions),
                    "exhausted": exhausted,
                    "recover": recover,
                }
            if not (terminated or truncated):
                raise ProtocolError(
                    "development episode did not end within 160 primitives"
                )
            if failure_reason == "root_recovery_unavailable":
                current_distance = _finite(
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
            }
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()

    @staticmethod
    def _aggregate(episodes: Sequence[Mapping]) -> dict:
        if len(episodes) != EXPECTED_EPISODES_PER_ROLE:
            raise ProtocolError("development aggregate requires exactly 8 episodes")
        actions = Counter({name: 0 for name in ACTION_NAMES})
        routes = Counter({name: 0 for name in ROUTES})
        distances = []
        for episode in episodes:
            actions.update(episode["action_counts"])
            routes.update(episode["route_counts"])
            metrics = episode["metrics"]
            for key in ("distance_to_goal", "navigation_error_m", "navigation_error"):
                if key in metrics:
                    distances.append(float(metrics[key]))
                    break
        successes = sum(int(float(episode["metrics"]["success"])) for episode in episodes)
        root_recovery_unavailable_count = sum(
            int(episode["recover"]["root_recovery_unavailable_count"])
            for episode in episodes
        )
        return {
            "episode_count": len(episodes),
            "success_count": successes,
            "successful_stop_count": sum(bool(episode["successful_stop"]) for episode in episodes),
            "timeout_count": sum(
                bool(episode["truncated"])
                and episode.get("failure_reason") != "root_recovery_unavailable"
                for episode in episodes
            ),
            "root_recovery_unavailable_count": root_recovery_unavailable_count,
            "stop_count": actions["STOP"],
            "sr": successes / len(episodes),
            "spl": sum(float(episode["metrics"]["spl"]) for episode in episodes) / len(episodes),
            "mean_navigation_error_m": sum(distances) / len(distances) if distances else 0.0,
            "total_path_length_m": sum(float(episode["metrics"]["path_length"]) for episode in episodes),
            "collision_count": sum(float(episode["metrics"]["collision_count"]) for episode in episodes),
            "action_counts": dict(actions),
            "route_counts": dict(routes),
        }

    def execute(
        self, *, manifest: Mapping, contract: Mapping,
        preflight: Mapping, args: argparse.Namespace,
    ) -> Mapping:
        import torch

        self._load_once()
        training = contract["training"]
        expert_rows = []
        expert_traces = []
        for spec in contract["train"]:
            result = self._run_episode(
                spec=spec,
                mode="expert",
                decision_budget=int(training["expert_max_decisions_per_episode"]),
                physical_budget=128,
                collect_rows=True,
            )
            expert_rows.extend(result["rows"])
            expert_traces.append({key: value for key, value in result.items() if key != "rows"})
        expert_rows = deduplicate_rows(expert_rows)
        if self.policy is None:
            raise ProtocolError("expert collection produced no frozen features")
        warmup, optimizer = train_successor_bc_phase(
            policy=self.policy,
            policy_module=self.policy_module,
            rows=expert_rows,
            training=training,
            updates=int(training["warmup_updates"]),
            seed=int(contract["seed"]),
        )

        dagger_rows = []
        dagger_traces = []
        for spec in contract["train"]:
            result = self._run_episode(
                spec=spec,
                mode="dagger",
                decision_budget=int(training["dagger_max_decisions_per_episode"]),
                physical_budget=80,
                collect_rows=True,
            )
            dagger_rows.extend(result["rows"])
            dagger_traces.append({key: value for key, value in result.items() if key != "rows"})
        combined = deduplicate_rows([*expert_rows, *dagger_rows])
        final, optimizer = train_successor_bc_phase(
            policy=self.policy,
            policy_module=self.policy_module,
            rows=combined,
            training=training,
            updates=int(training["final_updates"]),
            seed=int(contract["seed"]) + 1,
            optimizer=optimizer,
        )
        self.policy.bc_ready.zero_()
        output_parent = Path(args.output).expanduser().absolute().parent
        output_parent.mkdir(parents=True, exist_ok=True)
        candidate = output_parent / "bc_candidate.pt"
        if candidate.exists():
            raise FileExistsError(f"refusing to replace candidate checkpoint: {candidate}")
        torch.save({
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "bc_ready": 0,
            "policy_state_dict": self.policy.state_dict(),
            "manifest_sha256": _json_hash(manifest),
            "expert_dataset_sha256": warmup["dataset_sha256"],
            "combined_dataset_sha256": final["dataset_sha256"],
        }, candidate)
        self._candidate_path = candidate

        keep_episodes = [
            self._run_episode(
                spec=spec, mode="keep", decision_budget=160,
                physical_budget=160, collect_rows=False,
            )
            for spec in contract["development"]
        ]
        learned_episodes = [
            self._run_episode(
                spec=spec, mode="learned", decision_budget=160,
                physical_budget=160, collect_rows=False,
            )
            for spec in contract["development"]
        ]
        recover = {
            key: sum(int(episode["recover"][key]) for episode in learned_episodes)
            for key in (
                "eligible_blocked_count",
                "covered_next_decision_count",
                "recover_parent_count",
                "alignment_miss_count",
            )
        }
        recover["root_recovery_unavailable_count"] = sum(
            int(episode["recover"]["root_recovery_unavailable_count"])
            for episode in (*keep_episodes, *learned_episodes)
        )
        label_counts = Counter(row["route_label"] for row in combined)
        return {
            "candidate_checkpoint": str(candidate),
            "candidate_bc_ready": 0,
            "candidate_policy_state_sha256": _module_state_hash(self.policy),
            "optimizer_steps": int(warmup["updates"] + final["updates"]),
            "value_head_frozen": (
                warmup["value_head_hash_before"] == warmup["value_head_hash_after"]
                == final["value_head_hash_before"] == final["value_head_hash_after"]
            ),
            "development_evaluation_count": 1,
            "train_label_counts": {route: int(label_counts[route]) for route in ROUTES},
            "split_read_counts": {
                "train": int(self._split_reads["train"]),
                "val_seen": 0,
                "val_unseen": 0,
            },
            "matched_keep": self._aggregate(keep_episodes),
            "learned": self._aggregate(learned_episodes),
            "learned_episode_route_counts": [
                episode["route_counts"] for episode in learned_episodes
            ],
            "recover": recover,
            "training": {"warmup": warmup, "final": final},
            "collection": {
                "expert": expert_traces,
                "dagger": dagger_traces,
                "expert_unique_rows": len(expert_rows),
                "dagger_unique_rows": len(deduplicate_rows(dagger_rows)),
                "combined_unique_rows": len(combined),
            },
            "episodes": {
                "matched_keep": keep_episodes,
                "learned": learned_episodes,
            },
            "hashes": {
                "frozen_adapter_feature_api": self.manifest["feature_contract"]["api"],
                "candidate": _file_sha256(candidate),
            },
            "guards": {
                "model_load_count": self._model_load_count,
                "single_process": True,
                "train_only": True,
                "ppo_updates": 0,
                "value_head_frozen": True,
            },
        }

    def promote(
        self, *, evidence_digest: str, candidate_path: Path,
        promoted_path: Path,
    ) -> Mapping:
        import torch

        if self.policy is None or self._candidate_path != candidate_path:
            raise ProtocolError("promotion does not match the evaluated candidate")
        if self.policy.is_bc_ready():
            raise ProtocolError("candidate was marked ready before adjudication")
        self.policy.promote_bc_readiness(evidence_digest)
        torch.save({
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "bc_ready": 1,
            "readiness_evidence_sha256": evidence_digest,
            "candidate_sha256": _file_sha256(candidate_path),
            "policy_state_dict": self.policy.state_dict(),
            "manifest_sha256": _json_hash(self.manifest),
        }, promoted_path)
        return {
            "path": str(promoted_path),
            "bc_ready": 1,
            "readiness_evidence_sha256": evidence_digest,
            "candidate_sha256": _file_sha256(candidate_path),
        }


def _provider_execute(provider, **kwargs) -> Mapping:
    function = getattr(provider, "execute", provider)
    result = function(**kwargs)
    if not isinstance(result, Mapping):
        raise ProtocolError("readiness provider returned a non-object")
    return result


def _provider_promote(provider, **kwargs) -> Mapping:
    function = getattr(provider, "promote", None)
    if not callable(function):
        raise ProtocolError("passing readiness provider lacks fail-closed promotion")
    result = function(**kwargs)
    if not isinstance(result, Mapping) or result.get("bc_ready") not in (1, 1.0, True):
        raise ProtocolError("promoted checkpoint did not set bc_ready=1")
    return result


def run(
    args: argparse.Namespace,
    *,
    preflight_provider: Callable[[argparse.Namespace, Mapping, Mapping], Mapping] | None = None,
    readiness_provider: ReadinessProvider | Callable | None = None,
) -> dict:
    # No path, import, or output access may move above this line.
    require_train_split(args.split)
    if args.stage not in ("preflight", "readiness"):
        raise ValueError("stage must be preflight or readiness")
    manifest_path = Path(str(args.manifest)).expanduser().resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = validate_protocol_manifest(manifest)
    if str(args.device) != "cuda:0":
        raise ProtocolError("frozen readiness manifest requires process-local cuda:0")
    preflight_fn = preflight_provider or production_preflight
    preflight = dict(preflight_fn(args, manifest, contract))
    if preflight.get("status") != "pass":
        raise ProtocolError("readiness preflight did not pass")

    base = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "stage": args.stage,
        "split": TRAIN_SPLIT,
        "manifest": {
            "path": str(manifest_path),
            "file_sha256": _file_sha256(manifest_path),
            "canonical_sha256": _json_hash(manifest),
        },
        "protocol": {
            "train_episodes": contract["train"],
            "development_episodes": contract["development"],
            "seed": contract["seed"],
            "training": contract["training"],
        },
        "preflight": preflight,
        "guards": {
            "literal_train_only": True,
            "val_seen_reads": 0,
            "val_unseen_reads": 0,
            "ppo_updates": 0,
            "ttt_or_secondary_weights": False,
            "single_process": True,
        },
    }
    if args.stage == "preflight":
        base["gate_pass"] = None
        base["receipt_sha256"] = _json_hash(base)
        _atomic_json(args.output, base)
        return base
    if readiness_provider is None:
        readiness_provider = CanonicalReadinessProvider(
            args=args,
            manifest=manifest,
            contract=contract,
            preflight=preflight,
        )

    payload = dict(_provider_execute(
        readiness_provider,
        manifest=manifest,
        contract=contract,
        preflight=preflight,
        args=args,
    ))
    candidate = Path(str(payload.get("candidate_checkpoint", ""))).expanduser().resolve(strict=True)
    if not candidate.is_file() or payload.get("candidate_bc_ready") not in (0, 0.0, False):
        raise ProtocolError("provider must save a bc_ready=0 candidate before development")
    payload["candidate_checkpoint_sha256"] = _file_sha256(candidate)
    gate = adjudicate_readiness(payload, gate=contract["gate"])
    evidence = {
        "manifest_sha256": base["manifest"]["file_sha256"],
        "predecessors": preflight.get("predecessors"),
        "candidate_checkpoint_sha256": payload["candidate_checkpoint_sha256"],
        "gate": gate,
    }
    evidence_digest = _json_hash(evidence)
    evidence_path = Path(args.output).expanduser().absolute().with_name(
        "bc_readiness_evidence.json"
    )
    _atomic_canonical_json(evidence_path, evidence)
    if _file_sha256(evidence_path) != evidence_digest:
        raise ProtocolError("canonical readiness evidence SHA-256 drifted")
    promotion = None
    if gate["passed"]:
        promoted_path = Path(args.output).expanduser().absolute().with_name("bc_ready_promoted.pt")
        if promoted_path.exists():
            raise FileExistsError(f"refusing to replace promoted checkpoint: {promoted_path}")
        promotion = dict(_provider_promote(
            readiness_provider,
            evidence_digest=evidence_digest,
            candidate_path=candidate,
            promoted_path=promoted_path,
        ))
        promoted = Path(str(promotion.get("path", promoted_path))).expanduser().resolve(strict=True)
        if promoted != promoted_path.resolve(strict=True):
            raise ProtocolError("provider promoted outside the frozen output path")
        promotion.update({"path": str(promoted), "sha256": _file_sha256(promoted)})
    base.update({
        "result": payload,
        "gate": gate,
        "gate_pass": bool(gate["passed"]),
        "evidence_digest": evidence_digest,
        "readiness_evidence": {
            "path": str(evidence_path),
            "file_sha256": evidence_digest,
        },
        "promotion": promotion,
    })
    base["receipt_sha256"] = _json_hash(base)
    _atomic_json(args.output, base)
    return base


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "readiness"), required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--rlinf-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--vision", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--expert-receipt", type=Path, required=True)
    parser.add_argument("--keep-receipt", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
