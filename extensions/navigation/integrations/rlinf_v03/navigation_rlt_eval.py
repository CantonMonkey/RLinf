#!/usr/bin/env python3
"""Matched train-development evaluation for Native, A3, and Navigation-RLT.

One invocation evaluates exactly one behavior against the same immutable
32-episode ``train_development`` plan.  A3 and Navigation-RLT are always
deterministic, and every behavior uses the rollout stack's 160-physical-step
executor.  Native and A3 use the plain LightNav protocol.  Only Navigation-RLT
requests contextual-prefix artifacts; validated full artifacts are deleted by
exact content-addressed identity after durable feature evidence is written.

This evaluator never writes replay shards and accepts no validation split.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence
import time

import numpy as np

from build_navigation_rlt_train_split import FROZEN_DEVELOPMENT_SCENE_EPISODES
from lightnav_a3_policy import CausalTopologyState, sha256_file
from lightnav_a3_rollout import (
    _bool_success,
    _collision_event,
    _episode_metrics,
    _normalized_command,
    _physical_velocity,
    _scene_name,
    _validate_velocity_contract,
    aggregate as aggregate_a3,
)
from lightnav_l0_native_probe import (
    HabitatProtocolError,
    LightNavPrediction,
    LightNavWebSocketClient,
    VelocitySpec,
    ZmqHabitatClient,
    _collision_observation,
    _metric,
    _optional_metric,
    atomic_write_json,
    classify_velocity_action,
    encode_rgb_jpeg_base64,
    extract_instruction,
    latency_summary,
    redact_endpoint,
    waypoint_to_velocity_action,
)
from navigation_rlt_action import (
    MODE_NAMES,
    RECOVER,
    STOP,
    CompiledAction,
    compile_action,
    valid_mode_mask,
)
from navigation_rlt_parallel import PLAN_SCHEMA, PLAN_VERSION, atomic_json
from navigation_rlt_replay import LIGHTNAV_SOURCE_REVISION
from navigation_rlt_rollout import (
    ACTION_SCHEMA,
    ARTIFACT_CAPTURE_RULE,
    BEHAVIOR_A3,
    BEHAVIOR_NAMES,
    BEHAVIOR_NATIVE,
    BEHAVIOR_POLICY_ID,
    BEHAVIOR_RLT,
    EXECUTOR_SCHEMA,
    POLICY_PREFIX_SELECTION_RULE,
    POLICY_PREFIX_SELECTION_SCHEMA,
    PREFIX_FEATURE_SCHEMA,
    REFERENCE_SCHEMA,
    TOPOLOGY_SCHEMA,
    ArtifactFile,
    Behavior,
    ContextArtifact,
    ContextLightNavClient,
    ContinuousOnlineTopology,
    DecisionState,
    _artifact_relative_path,
    _directory_no_reparse,
    _load_behavior,
    _path_entry_exists,
    _query_state as _query_context_state,
    _regular_file_no_reparse,
    _resolve_artifact_root,
    _validate_sha256,
    _verified_output_receipt,
)
from streamvln_topology import CAUSAL_NODE_FEATURE_DIM, causal_topology_payload


SCHEMA_NAME = "navigation_rlt_matched_train_development_eval"
SCHEMA_VERSION = 1
FEATURE_EVIDENCE_SCHEMA = "navigation_rlt_eval_feature_evidence"
FEATURE_EVIDENCE_VERSION = 1
ARTIFACT_CLEANUP_SCHEMA = "navigation_rlt_eval_artifact_cleanup"
ARTIFACT_CLEANUP_VERSION = 1
EVALUATION_CODE_INVENTORY_SCHEMA = "navigation_rlt_evaluation_code_inventory"
EVALUATION_CODE_INVENTORY_VERSION = 1
REQUIRED_EVALUATION_CODE_FILES = frozenset(
    {
        "build_navigation_rlt_train_split.py",
        "lightnav_a3_policy.py",
        "lightnav_a3_rollout.py",
        "lightnav_l0_native_probe.py",
        "navigation_rlt_action.py",
        "navigation_rlt_compare.py",
        "navigation_rlt_eval.py",
        "navigation_rlt_obs_archive.py",
        "navigation_rlt_parallel.py",
        "navigation_rlt_policy.py",
        "navigation_rlt_post_training.py",
        "navigation_rlt_replay.py",
        "navigation_rlt_rollout.py",
        "streamvln_executor_contract.py",
        "streamvln_topology.py",
        "navrl/__init__.py",
        "navrl/contracts.py",
        "navrl/envs/__init__.py",
        "navrl/envs/base.py",
        "navrl/envs/grid_mock.py",
        "navrl/episode.py",
        "navrl/metrics.py",
        "navrl/policies/__init__.py",
        "navrl/policies/base.py",
        "navrl/rewards.py",
        "navrl/stack/__init__.py",
        "navrl/stack/checkpoint.py",
        "navrl/stack/collector.py",
        "navrl/stack/contracts.py",
        "navrl/stack/execution.py",
        "navrl/stack/interfaces.py",
        "navrl/stack/legacy.py",
        "navrl/stack/ppo.py",
        "navrl/stack/returns.py",
        "run_navigation_rlt_ebk_eval.sh",
    }
)
EXPECTED_EPISODES = 32
EXPECTED_SCENES = 8
EXPECTED_EPISODES_PER_SCENE = 4
FIXED_MAX_STEPS = 160
ORIGINAL_LIGHTNAV_WEIGHT_SHA256 = (
    "ffc4a925378a881afa761865048eb8d07c55cacf5eaf66548b6641c39f67af18"
)
RUN_MANIFEST_SCHEMA = "navigation_rlt_experiment_manifest"
RUN_MANIFEST_VERSION = 1
FORBIDDEN_SPLITS = frozenset({"val", "val_seen", "val_unseen", "val_seen_synonyms"})


class NavigationRLTEvalError(RuntimeError):
    """Raised when matched evaluation identity or execution differs."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _forbidden_path(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    return any(part.casefold() in FORBIDDEN_SPLITS for part in Path(value).parts)


def _mode_counts(counter: Mapping[str, int]) -> dict[str, int]:
    return {name: int(counter.get(name, 0)) for name in MODE_NAMES}


def _write_json_receipt(path: Path, value: Mapping[str, object]) -> dict[str, object]:
    atomic_json(path, value)
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": int(resolved.stat().st_size),
    }


def _git_commit(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 40 or value != value.lower():
        raise NavigationRLTEvalError(f"{name} must be a lowercase Git SHA-1")
    try:
        int(value, 16)
    except ValueError as exc:
        raise NavigationRLTEvalError(f"{name} must be a lowercase Git SHA-1") from exc
    return value


def _evaluation_source(name: str) -> Path:
    """Resolve inventory files in both the source checkout and eval bundle."""

    relative = PurePosixPath(name)
    bundle_root = Path(__file__).resolve(strict=True).parent
    candidates = [bundle_root.joinpath(*relative.parts)]
    if relative.parts and relative.parts[0] == "navrl":
        candidates.append(bundle_root.parents[1].joinpath(*relative.parts))
    found: list[Path] = []
    for candidate in candidates:
        if candidate.is_symlink():
            continue
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        if resolved.is_file():
            found.append(resolved)
    if len(found) != 1:
        raise NavigationRLTEvalError(
            f"evaluation code file is absent or ambiguous: {name}"
        )
    return found[0]


def _load_evaluation_code_inventory(
    path: Path, *, expected_sha256: str
) -> tuple[Path, dict[str, object], str]:
    raw_path = Path(path).expanduser()
    if raw_path.is_symlink():
        raise NavigationRLTEvalError("evaluation code inventory must not be a link")
    resolved = raw_path.resolve(strict=True)
    if not resolved.is_file():
        raise NavigationRLTEvalError("evaluation code inventory is not a regular file")
    observed_sha = sha256_file(resolved)
    if observed_sha != _validate_sha256(
        expected_sha256, "evaluation code inventory SHA"
    ):
        raise NavigationRLTEvalError("evaluation code inventory SHA-256 differs")
    try:
        inventory = json.loads(resolved.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NavigationRLTEvalError(
            f"evaluation code inventory is invalid JSON: {exc}"
        ) from exc
    if not isinstance(inventory, dict) or set(inventory) != {
        "schema_name",
        "schema_version",
        "navigation_code_commit",
        "files",
    }:
        raise NavigationRLTEvalError("evaluation code inventory members differ")
    if (
        inventory.get("schema_name") != EVALUATION_CODE_INVENTORY_SCHEMA
        or inventory.get("schema_version") != EVALUATION_CODE_INVENTORY_VERSION
    ):
        raise NavigationRLTEvalError("evaluation code inventory schema differs")
    commit = _git_commit(
        inventory.get("navigation_code_commit"),
        "evaluation code inventory navigation_code_commit",
    )
    files = inventory.get("files")
    if not isinstance(files, Mapping) or set(files) != REQUIRED_EVALUATION_CODE_FILES:
        raise NavigationRLTEvalError(
            "evaluation code inventory lacks one or more required files"
        )

    normalized_files: dict[str, str] = {}
    for name, raw_sha in files.items():
        if not isinstance(name, str):
            raise NavigationRLTEvalError("evaluation code inventory filename is invalid")
        relative = PurePosixPath(name)
        if (
            not relative.parts
            or relative.is_absolute()
            or relative.as_posix() != name
            or any(part in ("", ".", "..") for part in relative.parts)
        ):
            raise NavigationRLTEvalError(
                f"evaluation code inventory filename is not canonical: {name!r}"
            )
        expected_file_sha = _validate_sha256(
            raw_sha, f"evaluation code inventory files[{name}]"
        )
        raw_file = _evaluation_source(name)
        try:
            _regular_file_no_reparse(raw_file, f"evaluation code file {name}")
            actual_file = raw_file.resolve(strict=True)
        except Exception as exc:
            raise NavigationRLTEvalError(
                f"evaluation code file is absent, linked, or escapes the bundle: {name}"
            ) from exc
        if sha256_file(actual_file) != expected_file_sha:
            raise NavigationRLTEvalError(
                f"evaluation code file SHA-256 differs: {name}"
            )
        normalized_files[name] = expected_file_sha
    return (
        resolved,
        {
            "schema_name": EVALUATION_CODE_INVENTORY_SCHEMA,
            "schema_version": EVALUATION_CODE_INVENTORY_VERSION,
            "navigation_code_commit": commit,
            "files": normalized_files,
        },
        observed_sha,
    )


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _load_eval_plan(
    path: Path,
    *,
    expected_sha256: str,
    expected_split_manifest_sha256: str,
) -> tuple[dict[str, object], list[dict[str, object]], str]:
    raw_path = Path(path).expanduser()
    if raw_path.is_symlink() or _forbidden_path(str(raw_path)):
        raise NavigationRLTEvalError("evaluation plan path is a link or validation split")
    path = raw_path.resolve(strict=True)
    observed_sha = sha256_file(path)
    if observed_sha != _validate_sha256(expected_sha256, "evaluation plan SHA"):
        raise NavigationRLTEvalError("evaluation plan SHA-256 differs")
    split_sha = _validate_sha256(
        expected_split_manifest_sha256, "split manifest SHA"
    )
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NavigationRLTEvalError(f"evaluation plan is invalid JSON: {exc}") from exc
    if (
        not isinstance(plan, dict)
        or plan.get("schema_name") != PLAN_SCHEMA
        or plan.get("schema_version") != PLAN_VERSION
        or plan.get("split_label") != "train_development"
        or plan.get("validation_split_reads") != 0
        or plan.get("episode_count") != EXPECTED_EPISODES
        or not isinstance(plan.get("run_id"), str)
        or not plan.get("run_id")
    ):
        raise NavigationRLTEvalError(
            "evaluation plan is not the fixed train-only 32-episode development contract"
        )
    split_ref = plan.get("split_manifest")
    if (
        not isinstance(split_ref, Mapping)
        or split_ref.get("sha256") != split_sha
        or _forbidden_path(split_ref.get("path"))
    ):
        raise NavigationRLTEvalError("evaluation plan split-manifest identity differs")
    workers = plan.get("workers")
    worker_count = plan.get("worker_count")
    if (
        type(worker_count) is not int
        or worker_count < 1
        or not isinstance(workers, list)
        or len(workers) != worker_count
    ):
        raise NavigationRLTEvalError("evaluation plan worker inventory differs")

    episodes: list[dict[str, object]] = []
    worker_ids: set[int] = set()
    for worker in workers:
        if not isinstance(worker, Mapping) or type(worker.get("worker_id")) is not int:
            raise NavigationRLTEvalError("evaluation plan worker row is invalid")
        worker_id = int(worker["worker_id"])
        if worker_id in worker_ids:
            raise NavigationRLTEvalError("evaluation plan worker id repeats")
        worker_ids.add(worker_id)
        rows = worker.get("episodes")
        if not isinstance(rows, list) or worker.get("episode_count") != len(rows):
            raise NavigationRLTEvalError("evaluation plan worker episode count differs")
        for raw in rows:
            if not isinstance(raw, Mapping):
                raise NavigationRLTEvalError("evaluation plan episode row is invalid")
            row = dict(raw)
            if (
                type(row.get("episode_order")) is not int
                or not isinstance(row.get("episode_id"), str)
                or not row.get("episode_id")
                or not isinstance(row.get("scene"), str)
                or not row.get("scene")
            ):
                raise NavigationRLTEvalError("evaluation plan episode identity is incomplete")
            _validate_sha256(
                row.get("instruction_text_sha256"), "planned instruction hash"
            )
            episodes.append(row)
    if worker_ids != set(range(worker_count)):
        raise NavigationRLTEvalError("evaluation plan worker ids are not contiguous")
    episodes.sort(key=lambda row: int(row["episode_order"]))
    if [int(row["episode_order"]) for row in episodes] != list(range(EXPECTED_EPISODES)):
        raise NavigationRLTEvalError("evaluation plan episode order differs from 0..31")
    identities = [
        (str(row["episode_id"]), str(row["scene"]), str(row["instruction_text_sha256"]))
        for row in episodes
    ]
    if len(identities) != EXPECTED_EPISODES or len(set(identities)) != EXPECTED_EPISODES:
        raise NavigationRLTEvalError("evaluation plan is not 32 unique episode identities")
    frozen = [
        (episode_id, scene)
        for scene, episode_ids in FROZEN_DEVELOPMENT_SCENE_EPISODES
        for episode_id in episode_ids
    ]
    observed_frozen = [(str(row["episode_id"]), str(row["scene"])) for row in episodes]
    if observed_frozen != frozen:
        raise NavigationRLTEvalError("evaluation plan differs from fixed A3 development identities")
    scene_counts = Counter(str(row["scene"]) for row in episodes)
    if len(scene_counts) != EXPECTED_SCENES or set(scene_counts.values()) != {
        EXPECTED_EPISODES_PER_SCENE
    }:
        raise NavigationRLTEvalError("evaluation plan is not fixed 8 scenes x 4")
    return plan, episodes, observed_sha


def _load_run_manifest(
    path: Path,
    *,
    expected_sha256: str,
    expected_split_manifest_sha256: str,
    args: argparse.Namespace,
) -> tuple[dict[str, object], str]:
    raw_path = Path(path).expanduser()
    if raw_path.is_symlink() or _forbidden_path(str(raw_path)):
        raise NavigationRLTEvalError("run manifest path is a link or validation split")
    path = raw_path.resolve(strict=True)
    observed_sha = sha256_file(path)
    if observed_sha != _validate_sha256(expected_sha256, "run manifest SHA"):
        raise NavigationRLTEvalError("run manifest SHA-256 differs")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NavigationRLTEvalError(f"run manifest is invalid JSON: {exc}") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_name") != RUN_MANIFEST_SCHEMA
        or manifest.get("schema_version") != RUN_MANIFEST_VERSION
        or not isinstance(manifest.get("run_id"), str)
        or not manifest.get("run_id")
    ):
        raise NavigationRLTEvalError("run manifest schema or run identity differs")
    boundaries = manifest.get("boundaries")
    if (
        not isinstance(boundaries, Mapping)
        or boundaries.get("upstream_split_read") != "train"
        or boundaries.get("validation_split_reads") != 0
        or boundaries.get("lightnav_frozen") is not True
        or boundaries.get("weight_origin") != "original_release"
        or boundaries.get("ttt_or_secondary_post_training") is not False
    ):
        raise NavigationRLTEvalError("run manifest train-only/frozen boundary differs")
    data = manifest.get("data")
    if (
        not isinstance(data, Mapping)
        or data.get("split_manifest_sha256") != expected_split_manifest_sha256
        or data.get("development_episode_count") != EXPECTED_EPISODES
    ):
        raise NavigationRLTEvalError("run manifest development data identity differs")
    lightnav = manifest.get("lightnav")
    if (
        not isinstance(lightnav, Mapping)
        or lightnav.get("original_release_revision") != args.lightnav_source_revision
        or lightnav.get("weight_sha256") != args.lightnav_weight_sha256
        or lightnav.get("context_patch_sha256") != args.lightnav_hook_sha256
        or lightnav.get("tokenizer_sha256") != args.tokenizer_sha256
        or lightnav.get("processor_config_sha256") != args.processor_sha256
        or lightnav.get("history_config_sha256") != args.history_config_sha256
    ):
        raise NavigationRLTEvalError("run manifest LightNav identity differs from CLI")
    protocol = manifest.get("protocol")
    if (
        not isinstance(protocol, Mapping)
        or protocol.get("max_physical_steps") != FIXED_MAX_STEPS
        or protocol.get("matched_eval_arms")
        != ["native_lightnav", "a3_ppo", "navigation_rlt"]
    ):
        raise NavigationRLTEvalError("run manifest matched-evaluation protocol differs")
    return manifest, observed_sha


@dataclass(frozen=True)
class PlainDecisionState:
    prediction: LightNavPrediction
    native_stop: bool
    valid_mode_mask: np.ndarray
    recovery_parent_index: int
    recovery_parent_waypoint: tuple[float, float, float] | None


def _validate_prediction(prediction: LightNavPrediction) -> np.ndarray:
    trajectory = np.asarray(prediction.waypoints, dtype=np.float32)
    if trajectory.shape != (10, 3) or not np.isfinite(trajectory).all():
        raise NavigationRLTEvalError("LightNav trajectory must be finite [10,3]")
    if not 0 <= prediction.selected_index < 10 or not np.allclose(
        trajectory[prediction.selected_index],
        np.asarray(prediction.selected_waypoint, dtype=np.float32),
        rtol=1.0e-6,
        atol=1.0e-6,
    ):
        raise NavigationRLTEvalError("LightNav selected waypoint differs from trajectory")
    return trajectory


def _plain_state(
    prediction: LightNavPrediction,
    topology: ContinuousOnlineTopology,
    *,
    topology_max_nodes: int,
) -> PlainDecisionState:
    _validate_prediction(prediction)
    payload = causal_topology_payload(topology, max_nodes=topology_max_nodes)
    parent_index = int(payload.parent_index[payload.current_index])
    parent_waypoint = None
    if parent_index >= 0:
        node = payload.node_features[parent_index]
        forward = float(node[0])
        lateral = float(node[1])
        parent_waypoint = (
            forward,
            lateral,
            math.atan2(lateral, forward) if math.hypot(forward, lateral) > 1.0e-8 else 0.0,
        )
    return PlainDecisionState(
        prediction=prediction,
        native_stop=bool(prediction.stop),
        valid_mode_mask=np.asarray(
            valid_mode_mask(
                native_stop=bool(prediction.stop), has_recovery_parent=parent_index >= 0
            ),
            dtype=np.bool_,
        ),
        recovery_parent_index=parent_index,
        recovery_parent_waypoint=parent_waypoint,
    )


def _query_eval_state(
    model: LightNavWebSocketClient,
    behavior: Behavior,
    observation: Mapping[str, object],
    instruction: str,
    topology: ContinuousOnlineTopology,
    *,
    seq: int,
    jpeg_quality: int,
    artifact_prefix_max_tokens: int | None,
    policy_prefix_max_tokens: int | None,
    prefix_hidden_dim: int | None,
    topology_max_nodes: int,
    decision_index: int,
    max_steps: int,
    expected_context_source_sha256: str | None,
    feature_artifact_root: Path | None,
) -> PlainDecisionState | DecisionState:
    if behavior.name == BEHAVIOR_RLT:
        if (
            not isinstance(model, ContextLightNavClient)
            or artifact_prefix_max_tokens is None
            or policy_prefix_max_tokens is None
            or prefix_hidden_dim is None
        ):
            raise NavigationRLTEvalError("Navigation-RLT context configuration is incomplete")
        return _query_context_state(
            model,
            observation,
            instruction,
            topology,
            seq=seq,
            jpeg_quality=jpeg_quality,
            artifact_prefix_max_tokens=artifact_prefix_max_tokens,
            policy_prefix_max_tokens=policy_prefix_max_tokens,
            prefix_hidden_dim=prefix_hidden_dim,
            topology_max_nodes=topology_max_nodes,
            decision_index=decision_index,
            max_steps=max_steps,
            expected_source_sha256=expected_context_source_sha256,
            feature_artifact_root=feature_artifact_root,
        )
    if "rgb" not in observation:
        raise HabitatProtocolError("Habitat observation has no RGB frame")
    prediction = model.predict(
        seq=seq,
        image_b64=encode_rgb_jpeg_base64(observation["rgb"], quality=jpeg_quality),
        instruction=instruction,
    )
    return _plain_state(prediction, topology, topology_max_nodes=topology_max_nodes)


def _feature_evidence(
    artifacts: Sequence[ContextArtifact],
    trace: Sequence[Mapping[str, object]],
    *,
    planned_episode: Mapping[str, object],
    context_source_sha256: str,
) -> dict[str, object]:
    unique: dict[tuple[str, str], ContextArtifact] = {}
    for artifact in artifacts:
        unique.setdefault((artifact.identity, str(artifact.manifest_path)), artifact)
    return {
        "schema_name": FEATURE_EVIDENCE_SCHEMA,
        "schema_version": FEATURE_EVIDENCE_VERSION,
        "status": "complete",
        "split_label": "train_development",
        "validation_split_reads": 0,
        "episode_order": int(planned_episode["episode_order"]),
        "episode_id": str(planned_episode["episode_id"]),
        "scene": str(planned_episode["scene"]),
        "instruction_text_sha256": str(planned_episode["instruction_text_sha256"]),
        "context_source_sha256": context_source_sha256,
        "trace": list(trace),
        "artifacts": [
            {
                **artifact.evidence(),
                "manifest_sha256": artifact.manifest_sha256,
                "manifest_size_bytes": artifact.manifest_size_bytes,
                "files": [
                    {
                        "sha256": item.sha256,
                        "size_bytes": item.size_bytes,
                        "name": item.path.name,
                    }
                    for item in artifact.files
                ],
            }
            for artifact in unique.values()
        ],
    }


def _consume_eval_artifacts(
    artifacts: Sequence[ContextArtifact],
    *,
    root: Path,
    receipt_path: Path,
    feature_receipt: Mapping[str, object],
    planned_episode: Mapping[str, object],
) -> dict[str, object]:
    """Delete only fully authenticated files bound to durable eval evidence."""

    root = _resolve_artifact_root(root)
    feature_output = _verified_output_receipt(feature_receipt, "eval feature evidence")
    if not artifacts:
        raise NavigationRLTEvalError("Navigation-RLT episode has no context artifacts")
    deduplicated: dict[tuple[str, str], ContextArtifact] = {}
    identity_paths: dict[str, str] = {}
    manifest_identities: dict[str, str] = {}
    for artifact in artifacts:
        path_key = str(artifact.manifest_path)
        if identity_paths.setdefault(artifact.identity, path_key) != path_key:
            raise NavigationRLTEvalError("one context identity maps to multiple paths")
        if manifest_identities.setdefault(path_key, artifact.identity) != artifact.identity:
            raise NavigationRLTEvalError("one context path maps to multiple identities")
        deduplicated.setdefault((artifact.identity, path_key), artifact)

    receipt: dict[str, object] = {
        "schema_name": ARTIFACT_CLEANUP_SCHEMA,
        "schema_version": ARTIFACT_CLEANUP_VERSION,
        "status": "preflight",
        "episode_order": int(planned_episode["episode_order"]),
        "episode_id": str(planned_episode["episode_id"]),
        "feature_evidence_output": feature_output,
        "root": str(root),
        "artifact_count": len(deduplicated),
        "artifacts": [],
        "started_unix_s": time.time(),
    }
    prepared: list[
        tuple[Path, Path, list[ArtifactFile], dict[str, object]]
    ] = []
    try:
        for artifact in deduplicated.values():
            if artifact.content_root != root:
                raise NavigationRLTEvalError("artifact was not validated against cleanup root")
            prefix_dir = root / artifact.identity[:2]
            identity_dir = prefix_dir / artifact.identity
            if artifact.manifest_path != identity_dir / "manifest.json":
                raise NavigationRLTEvalError("artifact manifest path is not content-addressed")
            expected_files = list(artifact.files) + [
                ArtifactFile(
                    path=artifact.manifest_path,
                    sha256=artifact.manifest_sha256,
                    size_bytes=artifact.manifest_size_bytes,
                )
            ]
            if len({item.path for item in expected_files}) != len(expected_files):
                raise NavigationRLTEvalError("artifact file inventory repeats")
            _directory_no_reparse(prefix_dir, "artifact prefix directory")
            _directory_no_reparse(identity_dir, "artifact identity directory")
            if set(identity_dir.iterdir()) != {item.path for item in expected_files}:
                raise NavigationRLTEvalError("artifact directory differs from manifest inventory")
            file_rows: list[dict[str, object]] = []
            for item in expected_files:
                if item.path.parent != identity_dir:
                    raise NavigationRLTEvalError("artifact file escapes identity directory")
                _regular_file_no_reparse(item.path, "artifact cleanup file")
                if item.path.resolve(strict=True) != item.path:
                    raise NavigationRLTEvalError("artifact cleanup file resolution changed")
                if int(item.path.stat().st_size) != item.size_bytes:
                    raise NavigationRLTEvalError("artifact cleanup file size changed")
                if sha256_file(item.path) != item.sha256:
                    raise NavigationRLTEvalError("artifact cleanup file hash changed")
                file_rows.append(
                    {
                        "path": _artifact_relative_path(item.path, root, "artifact file"),
                        "sha256": item.sha256,
                        "size_bytes": item.size_bytes,
                        "status": "verified",
                    }
                )
            artifact_row: dict[str, object] = {
                "identity": artifact.identity,
                "manifest": _artifact_relative_path(
                    artifact.manifest_path, root, "feature manifest"
                ),
                "status": "ready",
                "files": file_rows,
            }
            cast_rows = receipt["artifacts"]
            if not isinstance(cast_rows, list):
                raise NavigationRLTEvalError("cleanup receipt artifact rows drifted")
            cast_rows.append(artifact_row)
            prepared.append((prefix_dir, identity_dir, expected_files, artifact_row))

        receipt["status"] = "ready"
        atomic_json(receipt_path, receipt)
        for prefix_dir, identity_dir, expected_files, artifact_row in prepared:
            file_rows = artifact_row["files"]
            if not isinstance(file_rows, list):
                raise NavigationRLTEvalError("cleanup receipt file rows drifted")
            for item, file_row in zip(expected_files, file_rows):
                _regular_file_no_reparse(item.path, "artifact cleanup file")
                if sha256_file(item.path) != item.sha256:
                    raise NavigationRLTEvalError("artifact changed after cleanup preflight")
                item.path.unlink()
                if isinstance(file_row, dict):
                    file_row["status"] = "deleted"
            if any(_path_entry_exists(item.path) for item in expected_files):
                raise NavigationRLTEvalError("artifact cleanup left an exact file behind")
            if any(identity_dir.iterdir()):
                raise NavigationRLTEvalError("artifact identity directory is not empty")
            identity_dir.rmdir()
            artifact_row["status"] = "deleted"
            artifact_row["identity_directory"] = "deleted"
            if not any(prefix_dir.iterdir()):
                prefix_dir.rmdir()
                artifact_row["prefix_directory"] = "deleted"
            else:
                artifact_row["prefix_directory"] = "retained_nonempty"
        receipt["status"] = "completed"
        receipt["completed_unix_s"] = time.time()
        atomic_json(receipt_path, receipt)
        return {
            "path": str(receipt_path.resolve(strict=True)),
            "sha256": sha256_file(receipt_path),
            "size_bytes": int(receipt_path.stat().st_size),
            "artifact_count": len(prepared),
            "status": "completed",
        }
    except Exception as exc:
        receipt["status"] = "failed"
        receipt["completed_unix_s"] = time.time()
        receipt["error"] = {"type": type(exc).__name__, "message": str(exc)}
        atomic_json(receipt_path, receipt)
        raise


def run_eval_episode(
    env: Any,
    model: LightNavWebSocketClient,
    behavior: Behavior,
    expected_episode: Mapping[str, object],
    *,
    rng: np.random.Generator,
    max_steps: int,
    jpeg_quality: int,
    artifact_prefix_max_tokens: int | None,
    policy_prefix_max_tokens: int | None,
    prefix_hidden_dim: int | None,
    topology_max_nodes: int,
    expected_context_source_sha256: str | None,
    feature_artifact_root: Path | None,
    require_collisions: bool,
) -> tuple[
    dict[str, object],
    list[float],
    tuple[ContextArtifact, ...],
    list[dict[str, object]],
]:
    observation, reset_info = env.reset()
    model.reset()
    instruction = extract_instruction(observation)
    instruction_sha = _sha256_bytes(instruction.encode("utf-8"))
    actual_episode_id = str(reset_info.get("episode_id", ""))
    actual_scene = _scene_name(reset_info.get("scene_id", ""))
    if (
        actual_episode_id != str(expected_episode.get("episode_id"))
        or actual_scene != str(expected_episode.get("scene"))
        or instruction_sha != str(expected_episode.get("instruction_text_sha256"))
    ):
        raise NavigationRLTEvalError("Habitat reset identity differs from evaluation plan")

    velocity = VelocitySpec.from_info(reset_info)
    _validate_velocity_contract(velocity)
    initial_distance = _metric(
        reset_info,
        ("distance_to_goal", "goal_distance", "navigation_error"),
        "navigation error",
    )
    previous_distance = initial_distance
    minimum_distance = initial_distance
    initial_path = _optional_metric(reset_info, ("path_length",)) or 0.0
    previous_path = initial_path
    collision_cumulative, _, collision_source = _collision_observation(reset_info)
    if require_collisions and collision_source == "unavailable":
        raise HabitatProtocolError("formal matched evaluation requires collision feedback")

    topology = ContinuousOnlineTopology(
        max_decisions=max_steps,
        forward_step_m=0.25,
        turn_degrees=15.0,
        node_radius_m=0.5,
    )
    a3_state = CausalTopologyState(previous_path_length_m=initial_path)
    seq = 0
    current = _query_eval_state(
        model,
        behavior,
        observation,
        instruction,
        topology,
        seq=seq,
        jpeg_quality=jpeg_quality,
        artifact_prefix_max_tokens=artifact_prefix_max_tokens,
        policy_prefix_max_tokens=policy_prefix_max_tokens,
        prefix_hidden_dim=prefix_hidden_dim,
        topology_max_nodes=topology_max_nodes,
        decision_index=0,
        max_steps=max_steps,
        expected_context_source_sha256=expected_context_source_sha256,
        feature_artifact_root=feature_artifact_root,
    )
    seq += 1
    latencies = [float(current.prediction.latency_ms)]
    artifacts: dict[tuple[str, str], ContextArtifact] = {}
    feature_trace: list[dict[str, object]] = []

    def remember(state: PlainDecisionState | DecisionState, decision_index: int) -> None:
        if not isinstance(state, DecisionState):
            return
        artifact = state.artifact
        key = (artifact.identity, str(artifact.manifest_path))
        artifacts.setdefault(key, artifact)
        feature_trace.append(
            {"decision_index": decision_index, "feature": state.feature_evidence()}
        )

    remember(current, 0)
    action_counts: Counter[str] = Counter()
    requested_counts: Counter[str] = Counter()
    executed_counts: Counter[str] = Counter()
    blocked_steps = 0
    collision_events = 0
    override_count = 0
    physical_step = 0
    decision_index = 0
    final_info: Mapping[str, object] = reset_info
    stop_source = "none"
    timed_out = False
    recover_streak = 0
    max_recover_streak = 0

    while True:
        requested: CompiledAction = behavior.request_action(
            current,
            a3_state=a3_state,
            decision_index=decision_index,
            max_steps=max_steps,
            rng=rng,
        )
        requested_counts[MODE_NAMES[requested.requested_mode]] += 1
        remaining = max_steps - physical_step
        if remaining < 1:
            raise NavigationRLTEvalError("physical budget exhausted before terminal action")
        budget_override = requested.requested_mode != STOP and len(requested.primitives) >= remaining
        if budget_override:
            executed = compile_action(
                mode=STOP,
                valid_modes=current.valid_mode_mask,
                reference_waypoint=current.prediction.selected_waypoint,
            )
            override_count += 1
            stop_source = "budget_stop"
        else:
            executed = requested
            if executed.executed_mode == STOP:
                if behavior.name in (BEHAVIOR_NATIVE, BEHAVIOR_A3) and current.native_stop:
                    stop_source = "native_lightnav_stop"
                else:
                    stop_source = "learned_stop"
        executed_counts[MODE_NAMES[executed.executed_mode]] += 1
        if executed.executed_mode == RECOVER:
            recover_streak += 1
            max_recover_streak = max(max_recover_streak, recover_streak)
        else:
            recover_streak = 0

        topology_feedback: list[tuple[float, float, bool, bool]] = []
        done = False
        for primitive in executed.primitives:
            action = waypoint_to_velocity_action(primitive, velocity)
            normalized_linear, normalized_angular = _normalized_command(action)
            physical_linear, physical_angular = _physical_velocity(action, velocity)
            category = classify_velocity_action(action, velocity)
            action_counts[category] += 1
            observation, _, terminated, truncated, final_info = env.step(action)
            if not isinstance(observation, Mapping) or not isinstance(final_info, Mapping):
                raise HabitatProtocolError("Habitat step returned invalid observation/info")
            physical_step += 1
            current_distance = _metric(
                final_info,
                ("distance_to_goal", "goal_distance", "navigation_error"),
                "navigation error",
            )
            minimum_distance = min(minimum_distance, current_distance)
            path_length = _optional_metric(final_info, ("path_length",))
            if path_length is None:
                raise HabitatProtocolError("matched evaluation requires Habitat path_length")
            path_delta = max(0.0, path_length - previous_path)
            collided, collision_cumulative, source = _collision_event(
                final_info, collision_cumulative
            )
            if require_collisions and source == "unavailable":
                raise HabitatProtocolError("matched evaluation lost collision feedback")
            collision_events += int(collided)
            commanded_distance = abs(physical_linear * velocity.time_step_s)
            blocked = commanded_distance >= 0.03 and path_delta < 0.01
            blocked_steps += int(blocked)
            yaw_delta = math.radians(physical_angular) * velocity.time_step_s
            topology_feedback.append((path_delta, yaw_delta, bool(collided), bool(blocked)))
            if behavior.name == BEHAVIOR_A3:
                a3_state.update(
                    normalized_linear=normalized_linear,
                    normalized_angular=normalized_angular,
                    physical_linear_mps=physical_linear,
                    physical_angular_dps=physical_angular,
                    time_step_s=velocity.time_step_s,
                    gate=executed.executed_mode,
                    selected_waypoint=current.prediction.selected_waypoint,
                    path_length_m=path_length,
                )
            previous_distance = current_distance
            previous_path = path_length
            primitive_timeout = bool(budget_override or truncated)
            timed_out = timed_out or primitive_timeout
            stopped = category == "STOP"
            if stopped and not (terminated or truncated):
                raise HabitatProtocolError("physical STOP did not terminate Habitat")
            done = bool(terminated or truncated or stopped or primitive_timeout)
            if done:
                break
        if not topology_feedback:
            raise NavigationRLTEvalError("compiled action executed no physical primitive")
        topology.update_executed(topology_feedback)
        if not done and physical_step >= max_steps:
            raise NavigationRLTEvalError("nonterminal action consumed the physical budget")
        decision_index += 1
        if done:
            break
        current = _query_eval_state(
            model,
            behavior,
            observation,
            instruction,
            topology,
            seq=seq,
            jpeg_quality=jpeg_quality,
            artifact_prefix_max_tokens=artifact_prefix_max_tokens,
            policy_prefix_max_tokens=policy_prefix_max_tokens,
            prefix_hidden_dim=prefix_hidden_dim,
            topology_max_nodes=topology_max_nodes,
            decision_index=decision_index,
            max_steps=max_steps,
            expected_context_source_sha256=expected_context_source_sha256,
            feature_artifact_root=feature_artifact_root,
        )
        seq += 1
        latencies.append(float(current.prediction.latency_ms))
        remember(current, decision_index)

    collision_metric = (
        float(collision_cumulative)
        if collision_cumulative is not None
        else float(collision_events)
    )
    metrics = _episode_metrics(
        reset_info=reset_info,
        final_info=final_info,
        initial_distance=initial_distance,
        minimum_distance=minimum_distance,
        initial_path=initial_path,
        step_count=physical_step,
        collision_count=collision_metric,
        blocked_steps=blocked_steps,
        stop_source=stop_source,
        timeout=timed_out,
        action_counts=action_counts,
        gate_counts=requested_counts,
    )
    metrics.update(
        {
            "episode_order": int(expected_episode["episode_order"]),
            "scene": str(expected_episode["scene"]),
            "instruction_text_sha256": str(expected_episode["instruction_text_sha256"]),
            "behavior": behavior.name,
            "behavior_policy_version": behavior.version,
            "actor_decision_count": decision_index,
            "requested_mode_distribution": _mode_counts(requested_counts),
            "executed_mode_distribution": _mode_counts(executed_counts),
            "safety_override_count": override_count,
            "max_consecutive_recover": max_recover_streak,
            "recover_loop": max_recover_streak >= 2,
            "context_request_count": len(latencies) if behavior.name == BEHAVIOR_RLT else 0,
            "model_latency": latency_summary(latencies),
        }
    )
    return metrics, latencies, tuple(artifacts.values()), feature_trace


def _aggregate(
    episodes: Sequence[Mapping[str, object]],
    latencies: Sequence[float],
) -> dict[str, object]:
    aggregate = aggregate_a3(episodes, latencies)
    requested: Counter[str] = Counter()
    executed: Counter[str] = Counter()
    for episode in episodes:
        requested.update(episode["requested_mode_distribution"])
        executed.update(episode["executed_mode_distribution"])
    aggregate.update(
        {
            "SR": aggregate["success_rate"],
            "SPL": aggregate["mean_spl"],
            "final_NE_m": aggregate["mean_final_navigation_error_m"],
            "path_m": aggregate["mean_path_length_m"],
            "collisions": aggregate["collision_count"],
            "blocked": aggregate["blocked_step_count"],
            "STOP_called": sum(bool(row["stop_called"]) for row in episodes),
            "timeouts": aggregate["timeout_count"],
            "requested_mode_distribution": _mode_counts(requested),
            "executed_mode_distribution": _mode_counts(executed),
            "safety_override_count": sum(
                int(row["safety_override_count"]) for row in episodes
            ),
            "context_request_count": sum(
                int(row["context_request_count"]) for row in episodes
            ),
            "recover_loop_episode_count": sum(
                int(row["max_consecutive_recover"]) >= 2 for row in episodes
            ),
            "max_recover_streak": max(
                int(row["max_consecutive_recover"]) for row in episodes
            ),
            "recover_loop_definition": "executed consecutive RECOVER >= 2",
            "latency": aggregate["model_latency"],
        }
    )
    return aggregate


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_steps != FIXED_MAX_STEPS:
        raise NavigationRLTEvalError("matched evaluation requires max_steps=160")
    if not 1 <= args.jpeg_quality <= 100 or args.topology_max_nodes < 1:
        raise NavigationRLTEvalError("jpeg quality or topology extent is invalid")
    if args.lightnav_source_revision != LIGHTNAV_SOURCE_REVISION:
        raise NavigationRLTEvalError("LightNav source is not the original frozen revision")
    for name in (
        "lightnav_hook_sha256",
        "lightnav_weight_sha256",
        "tokenizer_sha256",
        "processor_sha256",
        "history_config_sha256",
    ):
        _validate_sha256(getattr(args, name), name)
    if args.lightnav_weight_sha256 != ORIGINAL_LIGHTNAV_WEIGHT_SHA256:
        raise NavigationRLTEvalError("LightNav weights are not the frozen original release")
    a3_fields = (args.a3_checkpoint, args.expected_a3_checkpoint_sha256)
    rlt_fields = (args.rlt_checkpoint, args.expected_rlt_checkpoint_sha256)
    if args.behavior == BEHAVIOR_NATIVE and any(
        value is not None for value in (*a3_fields, *rlt_fields)
    ):
        raise NavigationRLTEvalError("Native behavior must not receive a policy checkpoint")
    if args.behavior == BEHAVIOR_A3 and (
        any(value is None for value in a3_fields)
        or any(value is not None for value in rlt_fields)
    ):
        raise NavigationRLTEvalError("A3 requires only its checkpoint and exact SHA-256")
    if args.behavior == BEHAVIOR_RLT and (
        any(value is None for value in rlt_fields)
        or any(value is not None for value in a3_fields)
    ):
        raise NavigationRLTEvalError(
            "Navigation-RLT requires only its checkpoint and exact SHA-256"
        )
    args.deterministic_policy = True
    args.worker_id = 0

    context_values = (
        args.artifact_prefix_max_tokens,
        args.policy_prefix_max_tokens,
        args.prefix_hidden_dim,
        args.expected_context_source_sha256,
        args.feature_artifact_root,
    )
    if args.behavior == BEHAVIOR_RLT:
        if any(value is None for value in context_values):
            raise NavigationRLTEvalError("Navigation-RLT requires all context arguments")
        if (
            args.artifact_prefix_max_tokens < 2
            or args.policy_prefix_max_tokens < 2
            or args.policy_prefix_max_tokens > args.artifact_prefix_max_tokens
            or args.prefix_hidden_dim < 1
        ):
            raise NavigationRLTEvalError("Navigation-RLT context extents are invalid")
        _validate_sha256(
            args.expected_context_source_sha256, "expected context source SHA"
        )
        root = _resolve_artifact_root(args.feature_artifact_root)
        if _is_within(args.output_dir, root) or _is_within(root, args.output_dir):
            raise NavigationRLTEvalError("output and feature artifact roots must be disjoint")
    else:
        if any(value is not None for value in context_values):
            raise NavigationRLTEvalError(
                "Native/A3 must omit context arguments so they cannot request extra features"
            )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--eval-plan", required=True, type=Path)
    result.add_argument("--expected-eval-plan-sha256", required=True)
    result.add_argument("--expected-split-manifest-sha256", required=True)
    result.add_argument("--run-manifest", required=True, type=Path)
    result.add_argument("--expected-run-manifest-sha256", required=True)
    result.add_argument("--evaluation-code-inventory", required=True, type=Path)
    result.add_argument(
        "--expected-evaluation-code-inventory-sha256", required=True
    )
    result.add_argument("--habitat-server", required=True)
    result.add_argument("--lightnav-server", required=True)
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument("--behavior", choices=BEHAVIOR_NAMES, required=True)
    result.add_argument("--a3-checkpoint", type=Path)
    result.add_argument("--expected-a3-checkpoint-sha256")
    result.add_argument("--rlt-checkpoint", type=Path)
    result.add_argument("--expected-rlt-checkpoint-sha256")
    result.add_argument("--policy-device", default="cpu")
    result.add_argument("--max-steps", type=int, default=FIXED_MAX_STEPS)
    result.add_argument("--jpeg-quality", type=int, default=90)
    result.add_argument("--seed", type=int, default=2026100202)
    result.add_argument("--client-id", default="rlinf-navigation-rlt-eval")
    result.add_argument("--topology-max-nodes", type=int, default=128)
    result.add_argument("--artifact-prefix-max-tokens", type=int)
    result.add_argument("--policy-prefix-max-tokens", type=int)
    result.add_argument("--prefix-hidden-dim", type=int)
    result.add_argument("--expected-context-source-sha256")
    result.add_argument("--feature-artifact-root", type=Path)
    result.add_argument("--lightnav-source-revision", required=True)
    result.add_argument("--lightnav-hook-sha256", required=True)
    result.add_argument("--lightnav-weight-sha256", required=True)
    result.add_argument("--tokenizer-sha256", required=True)
    result.add_argument("--processor-sha256", required=True)
    result.add_argument("--history-config-sha256", required=True)
    result.add_argument("--allow-missing-collisions", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    started = time.time()
    output_created = False
    try:
        _validate_args(args)
        (
            evaluation_code_inventory_path,
            evaluation_code_inventory,
            evaluation_code_inventory_sha,
        ) = _load_evaluation_code_inventory(
            args.evaluation_code_inventory,
            expected_sha256=args.expected_evaluation_code_inventory_sha256,
        )
        plan, planned_episodes, plan_sha = _load_eval_plan(
            args.eval_plan,
            expected_sha256=args.expected_eval_plan_sha256,
            expected_split_manifest_sha256=args.expected_split_manifest_sha256,
        )
        run_manifest, run_manifest_sha = _load_run_manifest(
            args.run_manifest,
            expected_sha256=args.expected_run_manifest_sha256,
            expected_split_manifest_sha256=args.expected_split_manifest_sha256,
            args=args,
        )
        if plan["run_id"] != run_manifest["run_id"]:
            raise NavigationRLTEvalError(
                "evaluation plan run_id differs from the outer run manifest"
            )
        behavior = _load_behavior(args)
        if behavior.name in (BEHAVIOR_A3, BEHAVIOR_RLT) and not behavior.deterministic:
            raise NavigationRLTEvalError("learned evaluation behavior must be deterministic")
        if args.output_dir.exists():
            raise FileExistsError(args.output_dir)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        output_created = True
        feature_dir = args.output_dir / "feature_evidence"
        cleanup_dir = args.output_dir / "artifact_cleanup"
        feature_root = None
        if behavior.name == BEHAVIOR_RLT:
            feature_dir.mkdir()
            cleanup_dir.mkdir()
            feature_root = _resolve_artifact_root(args.feature_artifact_root)

        env = ZmqHabitatClient(args.habitat_server)
        model: LightNavWebSocketClient
        if behavior.name == BEHAVIOR_RLT:
            model = ContextLightNavClient(args.lightnav_server, timeout_s=600.0)
        else:
            model = LightNavWebSocketClient(args.lightnav_server, timeout_s=600.0)
        model.open()
        try:
            if isinstance(model, ContextLightNavClient):
                model.login_for_context(f"{args.client_id}-{behavior.name}")
            else:
                model.login(f"{args.client_id}-{behavior.name}")
            rng = np.random.default_rng(args.seed)
            episode_rows: list[dict[str, object]] = []
            all_latencies: list[float] = []
            feature_receipts: list[dict[str, object]] = []
            cleanup_receipts: list[dict[str, object]] = []
            observed: set[tuple[str, str, str]] = set()
            for index, expected in enumerate(planned_episodes):
                metrics, latencies, artifacts, feature_trace = run_eval_episode(
                    env,
                    model,
                    behavior,
                    expected,
                    rng=rng,
                    max_steps=args.max_steps,
                    jpeg_quality=args.jpeg_quality,
                    artifact_prefix_max_tokens=args.artifact_prefix_max_tokens,
                    policy_prefix_max_tokens=args.policy_prefix_max_tokens,
                    prefix_hidden_dim=args.prefix_hidden_dim,
                    topology_max_nodes=args.topology_max_nodes,
                    expected_context_source_sha256=args.expected_context_source_sha256,
                    feature_artifact_root=feature_root,
                    require_collisions=not args.allow_missing_collisions,
                )
                identity = (
                    str(metrics["episode_id"]),
                    str(metrics["scene"]),
                    str(metrics["instruction_text_sha256"]),
                )
                if identity in observed:
                    raise NavigationRLTEvalError("Habitat repeated a development episode")
                observed.add(identity)
                if behavior.name == BEHAVIOR_RLT:
                    if feature_root is None or args.expected_context_source_sha256 is None:
                        raise NavigationRLTEvalError("context cleanup configuration disappeared")
                    stem = f"{int(expected['episode_order']):02d}_{expected['episode_id']}"
                    evidence = _feature_evidence(
                        artifacts,
                        feature_trace,
                        planned_episode=expected,
                        context_source_sha256=args.expected_context_source_sha256,
                    )
                    feature_receipt = _write_json_receipt(
                        feature_dir / f"{stem}.json", evidence
                    )
                    feature_receipt["episode_id"] = str(expected["episode_id"])
                    cleanup_receipt = _consume_eval_artifacts(
                        artifacts,
                        root=feature_root,
                        receipt_path=cleanup_dir / f"{stem}.json",
                        feature_receipt=feature_receipt,
                        planned_episode=expected,
                    )
                    cleanup_receipt["episode_id"] = str(expected["episode_id"])
                    feature_receipts.append(feature_receipt)
                    cleanup_receipts.append(cleanup_receipt)
                episode_rows.append(metrics)
                all_latencies.extend(latencies)
                atomic_json(
                    args.output_dir / "progress.json",
                    {
                        "schema_name": SCHEMA_NAME,
                        "schema_version": SCHEMA_VERSION,
                        "status": "running",
                        "behavior": behavior.name,
                        "completed_episode_count": index + 1,
                        "planned_episode_count": EXPECTED_EPISODES,
                    },
                )
        finally:
            model.close()
            env.close()

        if sha256_file(Path(args.eval_plan).resolve(strict=True)) != plan_sha:
            raise NavigationRLTEvalError("evaluation plan changed during execution")
        if sha256_file(Path(args.run_manifest).resolve(strict=True)) != run_manifest_sha:
            raise NavigationRLTEvalError("run manifest changed during execution")
        (
            final_evaluation_code_inventory_path,
            final_evaluation_code_inventory,
            final_evaluation_code_inventory_sha,
        ) = _load_evaluation_code_inventory(
            args.evaluation_code_inventory,
            expected_sha256=args.expected_evaluation_code_inventory_sha256,
        )
        if (
            final_evaluation_code_inventory_path != evaluation_code_inventory_path
            or final_evaluation_code_inventory_sha != evaluation_code_inventory_sha
            or final_evaluation_code_inventory != evaluation_code_inventory
        ):
            raise NavigationRLTEvalError(
                "evaluation code inventory changed during execution"
            )
        if len(observed) != EXPECTED_EPISODES:
            raise NavigationRLTEvalError("evaluation did not observe all 32 planned identities")
        aggregate = _aggregate(episode_rows, all_latencies)
        completion = {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "run_id": plan["run_id"],
            "behavior": behavior.name,
            "behavior_policy_id": BEHAVIOR_POLICY_ID,
            "behavior_policy_version": behavior.version,
            "behavior_source": behavior.source,
            "deterministic_policy": True,
            "split_label": "train_development",
            "upstream_split_read": "train",
            "validation_split_reads": 0,
            "evaluation_plan": {
                "path": str(Path(args.eval_plan).resolve(strict=True)),
                "sha256": plan_sha,
                "split_manifest_sha256": args.expected_split_manifest_sha256,
            },
            "run_manifest": {
                "path": str(Path(args.run_manifest).resolve(strict=True)),
                "sha256": run_manifest_sha,
            },
            "run_manifest_sha256": run_manifest_sha,
            "collection_manifest_sha256": run_manifest_sha,
            "evaluation_code": {
                "commit": evaluation_code_inventory["navigation_code_commit"],
                "inventory_sha256": evaluation_code_inventory_sha,
            },
            "model_identity": {
                "lightnav_source_revision": args.lightnav_source_revision,
                "lightnav_hook_sha256": args.lightnav_hook_sha256,
                "lightnav_weight_sha256": args.lightnav_weight_sha256,
                "tokenizer_sha256": args.tokenizer_sha256,
                "processor_sha256": args.processor_sha256,
                "history_config_sha256": args.history_config_sha256,
                "original_release_weights": True,
            },
            "executor": {
                "schema": EXECUTOR_SCHEMA,
                "max_physical_steps": args.max_steps,
                "jpeg_quality": args.jpeg_quality,
                "recover_loop_definition": "executed consecutive RECOVER >= 2",
            },
            "context": {
                "requested": behavior.name == BEHAVIOR_RLT,
                "request_count": aggregate["context_request_count"],
                "source_sha256": args.expected_context_source_sha256,
                "artifact_capture_rule": (
                    ARTIFACT_CAPTURE_RULE if behavior.name == BEHAVIOR_RLT else None
                ),
                "prefix_feature_schema": (
                    PREFIX_FEATURE_SCHEMA if behavior.name == BEHAVIOR_RLT else None
                ),
                "prefix_selection_schema": (
                    POLICY_PREFIX_SELECTION_SCHEMA if behavior.name == BEHAVIOR_RLT else None
                ),
                "prefix_selection_rule": (
                    POLICY_PREFIX_SELECTION_RULE if behavior.name == BEHAVIOR_RLT else None
                ),
                "full_artifacts_cleaned": behavior.name == BEHAVIOR_RLT,
            },
            "schemas": {
                "reference": REFERENCE_SCHEMA,
                "topology": TOPOLOGY_SCHEMA,
                "topology_node_dim": CAUSAL_NODE_FEATURE_DIM,
                "action": ACTION_SCHEMA,
            },
            "seed": args.seed,
            "habitat_server": redact_endpoint(args.habitat_server),
            "lightnav_server": redact_endpoint(args.lightnav_server),
            "episodes": episode_rows,
            "feature_evidence": feature_receipts,
            "artifact_cleanup": cleanup_receipts,
            "aggregate": aggregate,
            "training_replay_written": False,
            "started_unix_s": started,
            "completed_unix_s": time.time(),
        }
        completion_path = args.output_dir / "evaluation.json"
        atomic_json(completion_path, completion)
        atomic_json(
            args.output_dir / "COMPLETED.json",
            {
                "schema_name": SCHEMA_NAME,
                "schema_version": SCHEMA_VERSION,
                "status": "completed",
                "behavior": behavior.name,
                "run_manifest_sha256": run_manifest_sha,
                "collection_manifest_sha256": run_manifest_sha,
                "evaluation_code": {
                    "commit": evaluation_code_inventory["navigation_code_commit"],
                    "inventory_sha256": evaluation_code_inventory_sha,
                },
                "evaluation_sha256": sha256_file(completion_path),
                "training_replay_written": False,
            },
        )
        print(json.dumps(aggregate, sort_keys=True))
        return 0
    except Exception as exc:
        if output_created:
            atomic_write_json(
                args.output_dir / "INVALID.json",
                {
                    "schema_name": SCHEMA_NAME,
                    "schema_version": SCHEMA_VERSION,
                    "status": "invalid",
                    "behavior": getattr(args, "behavior", None),
                    "training_replay_written": False,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                    "started_unix_s": started,
                    "completed_unix_s": time.time(),
                },
            )
        print(f"Navigation-RLT eval invalid: {type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
