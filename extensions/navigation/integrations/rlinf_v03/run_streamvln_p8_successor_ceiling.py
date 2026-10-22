"""Bounded train-only live ceiling for the version-isolated P8 successor.

The three explicit stages intentionally have different dependency boundaries:

* ``preflight`` verifies the pinned release bytes and all runtime paths without
  constructing Habitat or loading the released model;
* ``expert`` executes train-shortest-path STOP/WAYPOINT labels through the real
  successor planner and one-primitive feedback loop without loading a model;
* ``keep`` is the only stage allowed to construct the released StreamVLN
  adapter, and executes exact KEEP decisions through ``SuccessorRunnerBridge``.

This runner has no optimizer, BC, PPO, validation-split, TTT, or checkpoint
selection path.  A receipt is written only after the requested stage succeeds.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Callable, Mapping


SCHEMA_NAME = "streamvln_p8_successor_ceiling"
SCHEMA_VERSION = 1
CEILING_MANIFEST_SCHEMA_NAME = "streamvln_p8_successor_live_ceiling"
CEILING_MANIFEST_SCHEMA_VERSION = 1
STAGES = ("preflight", "expert", "keep")
TRAIN_SPLIT = "train"
EXPERT_LOOKAHEAD_M = 1.0
ACTION_CHUNK = 4
ACTION_NAMES = ("STOP", "FORWARD", "LEFT", "RIGHT")
ROLE_MAX_STEPS = {"expert": 160, "keep": 40}
CANONICAL_STREAMVLN_PACKAGE = "rlinf.envs.streamvln"
CEILING_PROTOCOL_CONTRACT_SHA256 = (
    "fe8d7b4fc0c74ea15b76eba92364600493a189747e81ad68ebbc387771adc021"
)
CEILING_MANIFEST_RELATIVE_PATH = Path(
    "integrations/rlinf_v03/streamvln_p8_successor_ceiling_manifest.json"
)
HABITAT_SERVER_RELATIVE_PATH = Path("scripts/habitat_r2r_service.py")


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _json_hash(value: object) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _ceiling_protocol_contract_sha256(manifest: Mapping) -> str:
    """Hash immutable gate fields without the runner/manifest hash cycle."""

    if not isinstance(manifest, Mapping):
        raise ValueError("ceiling manifest root must be an object")
    payload = json.loads(_json_bytes(manifest))
    deployment = payload.get("deployment")
    if not isinstance(deployment, dict):
        raise ValueError("ceiling manifest lacks its deployment contract")
    deployment.pop("implementation_files", None)
    return _json_hash(payload)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_train_split(split: object) -> str:
    """First executable gate: no import or path access may precede this call."""

    if type(split) is not str or split != TRAIN_SPLIT:
        raise ValueError("P8 successor ceiling permits only the literal train split")
    return split


def _import_canonical(name: str, *, rlinf_root: Path):
    qualified_name = f"{CANONICAL_STREAMVLN_PACKAGE}.{name}"
    module = importlib.import_module(qualified_name)
    if module.__name__ != qualified_name:
        raise RuntimeError(f"canonical StreamVLN import identity drifted: {qualified_name}")
    expected_file = (
        rlinf_root / "rlinf" / "envs" / "streamvln" / f"{name}.py"
    ).resolve(strict=True)
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str) or Path(module_file).resolve(strict=True) != expected_file:
        raise RuntimeError(
            f"canonical StreamVLN import escaped verified rlinf_root: {qualified_name}"
        )
    return module


def _regular_file(value: object, *, name: str) -> Path:
    path = Path(str(value)).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"{name} must be a regular file")
    return path


def _preserved_absolute_file(value: object, *, name: str) -> Path:
    """Validate a file while preserving its requested, possibly symlinked path."""

    path = Path(os.path.abspath(Path(str(value)).expanduser()))
    if not path.is_file():
        raise ValueError(f"{name} must be a file")
    return path


def _directory(value: object, *, name: str) -> Path:
    path = Path(str(value)).expanduser().resolve(strict=True)
    if not path.is_dir():
        raise ValueError(f"{name} must be a directory")
    return path


def _read_manifest(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("release manifest root must be an object")
    return value


def _sha256_value(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _manifest_role(manifest: Mapping, args: argparse.Namespace) -> tuple[str, list[dict], dict]:
    dataset = manifest.get("dataset")
    episodes_by_role = dataset.get("episodes") if isinstance(dataset, Mapping) else None
    gates = manifest.get("gates")
    if not isinstance(episodes_by_role, Mapping) or not isinstance(gates, Mapping):
        raise ValueError("ceiling manifest lacks typed episode roles or gates")

    contracts = {}
    for role, fixed_max_steps in ROLE_MAX_STEPS.items():
        rows = episodes_by_role.get(role)
        gate = gates.get(role)
        if not isinstance(rows, list) or not rows or not isinstance(gate, Mapping):
            raise ValueError(f"ceiling manifest lacks the {role} role contract")
        normalized = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"ceiling manifest {role} episode is not an object")
            index = row.get("index")
            episode_id = row.get("episode_id")
            scene = row.get("scene")
            if (
                type(index) is not int
                or index < 0
                or not isinstance(episode_id, str)
                or not episode_id
                or not isinstance(scene, str)
                or not scene
            ):
                raise ValueError(f"ceiling manifest {role} episode identity is invalid")
            normalized.append(
                {"index": index, "episode_id": episode_id, "scene": scene}
            )
        indices = tuple(row["index"] for row in normalized)
        if len(set(indices)) != len(indices):
            raise ValueError(f"ceiling manifest {role} episode indices are not unique")
        episode_ids = tuple(row["episode_id"] for row in normalized)
        scenes = tuple(row["scene"] for row in normalized)
        if len(set(episode_ids)) != len(episode_ids):
            raise ValueError(f"ceiling manifest {role} episode IDs are not unique")
        if len(set(scenes)) != len(scenes):
            raise ValueError(f"ceiling manifest {role} scenes are not unique")
        if gate.get("episode_count") != len(normalized):
            raise ValueError(f"ceiling manifest {role} episode count is inconsistent")
        if gate.get("max_physical_steps_per_episode") != fixed_max_steps:
            raise ValueError(
                f"ceiling manifest {role} max steps must be {fixed_max_steps}"
            )
        if role == "expert":
            threshold_names = ("minimum_successes", "minimum_successful_stops")
        else:
            threshold_names = ("minimum_comparisons_per_episode",)
            if gate.get("required_mismatch_count") != 0:
                raise ValueError("ceiling manifest KEEP mismatch requirement must be zero")
        if any(type(gate.get(name)) is not int or gate[name] < 0 for name in threshold_names):
            raise ValueError(f"ceiling manifest {role} gate threshold is invalid")
        contracts[role] = (normalized, dict(gate))

    expert_rows = contracts["expert"][0]
    keep_rows = contracts["keep"][0]
    if len(keep_rows) != 2 or keep_rows != expert_rows[:2]:
        raise ValueError(
            "ceiling manifest KEEP episodes must equal the first two expert episodes"
        )

    requested_indices = tuple(args.episode_index)
    matches = [
        role
        for role, (rows, gate) in contracts.items()
        if requested_indices == tuple(row["index"] for row in rows)
        and args.max_steps == gate["max_physical_steps_per_episode"]
    ]
    if args.stage == "preflight":
        if len(matches) != 1:
            raise ValueError(
                "preflight must use exactly the frozen expert/max160 or KEEP/max40 role"
            )
        role = matches[0]
    else:
        role = args.stage
        if matches != [role]:
            raise ValueError(
                f"{role} CLI episode order or max_steps overrides the ceiling manifest"
            )
    rows, gate = contracts[role]
    return role, rows, gate


def _load_ceiling_contract(args: argparse.Namespace) -> dict:
    path = _regular_file(args.ceiling_manifest, name="ceiling_manifest")
    raw_sha256 = _file_sha256(path)
    manifest = _read_manifest(path)
    if (
        manifest.get("schema_name") != CEILING_MANIFEST_SCHEMA_NAME
        or manifest.get("schema_version") != CEILING_MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError("ceiling manifest schema identity is invalid")
    if _ceiling_protocol_contract_sha256(manifest) != CEILING_PROTOCOL_CONTRACT_SHA256:
        raise RuntimeError("ceiling manifest protocol contract drifted")
    scope = manifest.get("scope")
    if not isinstance(scope, Mapping) or scope.get("split") != TRAIN_SPLIT:
        raise ValueError("ceiling manifest is not scoped to the literal train split")
    if manifest.get("ceiling_run_seed") != args.seed:
        raise ValueError("CLI seed overrides the frozen ceiling_run_seed")
    role, episodes, gate = _manifest_role(manifest, args)
    return {
        "path": path,
        "raw_sha256": raw_sha256,
        "manifest": manifest,
        "role": role,
        "episodes": episodes,
        "gate": gate,
    }


def _verify_file_record(path: Path, record: Mapping, *, name: str) -> dict:
    expected_bytes = record.get("bytes")
    if type(expected_bytes) is not int or expected_bytes < 0:
        raise ValueError(f"{name} bytes is invalid")
    expected_sha256 = _sha256_value(record.get("sha256"), name=f"{name} sha256")
    if not path.is_file():
        raise ValueError(f"{name} target is not a file")
    observed_bytes = path.stat().st_size
    observed_sha256 = _file_sha256(path)
    if observed_bytes != expected_bytes or observed_sha256 != expected_sha256:
        raise RuntimeError(f"{name} target bytes or SHA-256 drifted")
    return {
        "path": str(path),
        "bytes": observed_bytes,
        "sha256": observed_sha256,
    }


def _mapping(value: object, *, name: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _manifest_path(value: object, *, name: str, preserve_symlink: bool = False) -> Path:
    if (
        not isinstance(value, (str, os.PathLike))
        or not str(value)
        or not Path(value).is_absolute()
    ):
        raise ValueError(f"{name} must be an absolute path")
    expanded = Path(value).expanduser()
    if preserve_symlink:
        return Path(os.path.abspath(expanded))
    return expanded.resolve(strict=False)


def _bind_path(
    observed: Path,
    expected: object,
    *,
    name: str,
    preserve_symlink: bool = False,
) -> str:
    expected_path = _manifest_path(
        expected, name=f"{name} manifest path", preserve_symlink=preserve_symlink
    )
    observed_path = (
        Path(os.path.abspath(observed))
        if preserve_symlink
        else observed.resolve(strict=True)
    )
    if observed_path != expected_path:
        raise RuntimeError(f"{name} path overrides the ceiling manifest")
    return str(observed_path)


def _verify_sha256_record(
    path: Path,
    record: Mapping,
    *,
    name: str,
    sha256_key: str = "sha256",
) -> dict:
    expected_sha256 = _sha256_value(
        record.get(sha256_key), name=f"{name} SHA-256"
    )
    if not path.is_file():
        raise ValueError(f"{name} target is not a file")
    observed_sha256 = _file_sha256(path)
    if observed_sha256 != expected_sha256:
        raise RuntimeError(f"{name} target SHA-256 drifted")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": observed_sha256,
    }


def _verify_runtime_contract(
    manifest: Mapping,
    *,
    habitat_python: Path,
    habitat_server: Path,
    habitat_sim_python: Path,
    habitat_env_lib: Path,
    egl_vendor_json: Path,
    implementation_observations: list[Mapping],
) -> dict:
    runtime = _mapping(manifest.get("runtime"), name="ceiling runtime contract")
    habitat_child = _mapping(
        runtime.get("habitat_child"), name="Habitat child runtime contract"
    )
    _bind_path(
        habitat_python,
        habitat_child.get("python_path"),
        name="Habitat Python",
        preserve_symlink=True,
    )
    python_observation = _verify_sha256_record(
        habitat_python,
        habitat_child,
        name="Habitat Python",
        sha256_key="python_sha256",
    )

    server_rows = [
        row
        for row in implementation_observations
        if row.get("deployment_relative_path")
        == HABITAT_SERVER_RELATIVE_PATH.as_posix()
    ]
    if len(server_rows) != 1:
        raise RuntimeError(
            "ceiling implementation closure must contain one Habitat server"
        )
    _bind_path(
        habitat_server,
        server_rows[0]["path"],
        name="Habitat server",
    )

    sim_patch = _mapping(
        runtime.get("habitat_sim_patch"), name="Habitat-Sim patch contract"
    )
    sim_patch_path = _regular_file(
        _manifest_path(sim_patch.get("path"), name="Habitat-Sim patch path"),
        name="habitat_sim_patch",
    )
    sim_patch_observation = _verify_file_record(
        sim_patch_path, sim_patch, name="Habitat-Sim patch"
    )

    egl_contract = _mapping(
        runtime.get("egl_vendor_json"), name="EGL vendor contract"
    )
    _bind_path(
        egl_vendor_json,
        egl_contract.get("path"),
        name="EGL vendor JSON",
    )
    egl_observation = _verify_file_record(
        egl_vendor_json, egl_contract, name="EGL vendor JSON"
    )

    return {
        "habitat_python": python_observation,
        "habitat_server": dict(server_rows[0]),
        "habitat_sim_python": str(habitat_sim_python),
        "habitat_sim_patch": sim_patch_observation,
        "habitat_env_lib": str(habitat_env_lib),
        "egl_vendor_json": egl_observation,
    }


def _verify_scene_assets(
    manifest: Mapping,
    *,
    asset_root: Path,
    expected_scenes: set[str],
) -> list[dict]:
    records = manifest.get("scene_assets")
    if not isinstance(records, list) or not records:
        raise ValueError("ceiling manifest has no scene asset closure")
    observations = []
    seen = set()
    for ordinal, raw_record in enumerate(records):
        record = _mapping(raw_record, name=f"scene_assets[{ordinal}]")
        scene = record.get("scene")
        if (
            not isinstance(scene, str)
            or not scene
            or Path(scene).name != scene
            or scene in seen
        ):
            raise ValueError("ceiling scene asset identity is invalid or repeated")
        seen.add(scene)
        scene_observation = {"scene": scene}
        for kind, suffix in (("glb", ".glb"), ("navmesh", ".navmesh")):
            file_record = _mapping(
                record.get(kind), name=f"scene_assets[{ordinal}].{kind}"
            )
            expected_path = (
                asset_root / "scene_datasets" / "mp3d" / scene / f"{scene}{suffix}"
            ).resolve(strict=True)
            _bind_path(
                expected_path,
                file_record.get("path"),
                name=f"scene {scene} {kind}",
            )
            scene_observation[kind] = _verify_file_record(
                expected_path,
                file_record,
                name=f"scene {scene} {kind}",
            )
        observations.append(scene_observation)
    if seen != expected_scenes:
        raise RuntimeError("ceiling scene asset closure does not match episode roles")
    return observations


def _verify_cli_path_contract(
    manifest: Mapping,
    *,
    ceiling_manifest: Path,
    release_manifest: Path,
    source_root: Path,
    checkpoint: Path,
    tokenizer: Path,
    vision: Path,
    rlinf_root: Path,
    asset_root: Path,
    habitat_sim_python: Path,
    habitat_env_lib: Path,
    egl_vendor_json: Path,
) -> dict:
    paths = _mapping(manifest.get("paths"), name="ceiling path contract")
    release = _mapping(manifest.get("release"), name="ceiling release contract")
    expected = {
        "ceiling_manifest": rlinf_root / CEILING_MANIFEST_RELATIVE_PATH,
        "release_manifest": rlinf_root
        / str(release.get("release_manifest_deployment_path")),
        "source_root": paths.get("streamvln_source"),
        "checkpoint": paths.get("checkpoint"),
        "tokenizer": paths.get("tokenizer"),
        "vision": paths.get("vision_tower"),
        "rlinf_root": paths.get("deployment_root"),
        "asset_root": paths.get("asset_root"),
        "habitat_sim_python": paths.get("habitat_sim_python"),
        "habitat_env_lib": paths.get("habitat_env_lib"),
        "egl_vendor_json": paths.get("egl_vendor_json"),
    }
    observed = {
        "ceiling_manifest": ceiling_manifest,
        "release_manifest": release_manifest,
        "source_root": source_root,
        "checkpoint": checkpoint,
        "tokenizer": tokenizer,
        "vision": vision,
        "rlinf_root": rlinf_root,
        "asset_root": asset_root,
        "habitat_sim_python": habitat_sim_python,
        "habitat_env_lib": habitat_env_lib,
        "egl_vendor_json": egl_vendor_json,
    }
    return {
        name: _bind_path(path, expected[name], name=name)
        for name, path in observed.items()
    }


def _verify_implementation_targets(manifest: Mapping, rlinf_root: Path) -> list[dict]:
    deployment = manifest.get("deployment")
    records = (
        deployment.get("implementation_files")
        if isinstance(deployment, Mapping)
        else None
    )
    if not isinstance(records, list) or not records:
        raise ValueError("ceiling manifest has no deployment implementation closure")
    root = rlinf_root.resolve(strict=True)
    observed = []
    seen = set()
    for ordinal, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError("ceiling implementation record is not an object")
        relative_text = record.get("deployment_relative_path")
        if not isinstance(relative_text, str) or not relative_text:
            raise ValueError("ceiling implementation deployment path is invalid")
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts or relative_text in seen:
            raise ValueError("ceiling implementation deployment path escapes or repeats")
        seen.add(relative_text)
        target = (root / relative).resolve(strict=True)
        if os.path.commonpath((str(root), str(target))) != str(root):
            raise ValueError("ceiling implementation target escapes the deployment root")
        observation = _verify_file_record(
            target, record, name=f"implementation_files[{ordinal}]"
        )
        observation["deployment_relative_path"] = relative_text
        observed.append(observation)
    return observed


def _episode_indices(values: object) -> tuple[int, ...]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("at least one explicit train episode index is required")
    result = tuple(values)
    if any(type(value) is not int or value < 0 for value in result):
        raise ValueError("episode indices must be nonnegative integers")
    if len(set(result)) != len(result):
        raise ValueError("episode indices must be unique")
    return result


def _validate_common_args(args: argparse.Namespace) -> tuple[int, ...]:
    if args.stage not in STAGES:
        raise ValueError(f"unknown P8 successor ceiling stage: {args.stage!r}")
    indices = _episode_indices(args.episode_index)
    if type(args.seed) is not int or args.seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(args.max_steps) is not int or args.max_steps < ACTION_CHUNK:
        raise ValueError("max_steps must be a positive four-action horizon")
    if args.max_steps % ACTION_CHUNK:
        raise ValueError("max_steps must align to the four-action transport width")
    if (
        isinstance(args.habitat_startup_timeout_s, bool)
        or not isinstance(args.habitat_startup_timeout_s, (int, float))
        or not math.isfinite(float(args.habitat_startup_timeout_s))
        or not 0.0 < float(args.habitat_startup_timeout_s) <= 3600.0
    ):
        raise ValueError("habitat_startup_timeout_s must be finite and in (0, 3600]")
    if args.stage == "keep" and not getattr(args, "expert_receipt", None):
        raise ValueError("KEEP requires --expert-receipt")
    return indices


def _default_release_verifier(
    *,
    manifest: Mapping,
    rlinf_root: Path,
    source_root: Path,
    checkpoint: Path,
    tokenizer: Path,
    vision: Path,
) -> dict:
    """Light provenance verification.  This function never constructs a model."""

    adapter = _import_canonical(
        "streamvln_p8_official_adapter", rlinf_root=rlinf_root
    )
    anchor = adapter._require_pinned_release_manifest_sha256()
    if adapter._manifest_sha256(manifest) != anchor:
        raise RuntimeError("successor ceiling manifest digest does not match pinned release")
    verification = adapter.verify_full_content_artifact_manifest(
        manifest,
        source_root=source_root,
        model_path=checkpoint,
        tokenizer_path=tokenizer,
        vision_tower_path=vision,
    )
    verified_root, source_observation = adapter._verify_source_tree(source_root)
    if verified_root != source_root or verification.get("manifest_sha256") != anchor:
        raise RuntimeError("successor ceiling verified release provenance drifted")
    verification["source"].update(source_observation)
    return {"anchor": anchor, "verification": verification}


def _verify_expert_receipt(
    path: Path,
    *,
    ceiling_manifest_raw_sha256: str,
    expert_rows: list[Mapping],
) -> dict:
    receipt = _read_manifest(path)
    if (
        receipt.get("schema_name") != SCHEMA_NAME
        or receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("status") != "pass"
        or receipt.get("stage") != "expert"
        or receipt.get("gate_pass") is not True
    ):
        raise RuntimeError("KEEP requires a passing expert ceiling receipt")
    receipt_sha256 = _sha256_value(
        receipt.get("receipt_sha256"), name="expert receipt internal SHA-256"
    )
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if _json_hash(unsigned) != receipt_sha256:
        raise RuntimeError("expert ceiling receipt internal SHA-256 drifted")
    preflight_receipt = _mapping(
        receipt.get("preflight"), name="expert receipt preflight"
    )
    gate = _mapping(receipt.get("gate"), name="expert receipt gate")
    result = _mapping(receipt.get("result"), name="expert receipt result")
    expected_indices = [row["index"] for row in expert_rows]
    if (
        receipt.get("ceiling_manifest_raw_sha256")
        != ceiling_manifest_raw_sha256
        or preflight_receipt.get("ceiling_manifest_raw_sha256")
        != ceiling_manifest_raw_sha256
        or preflight_receipt.get("ceiling_role") != "expert"
        or preflight_receipt.get("split") != TRAIN_SPLIT
        or preflight_receipt.get("episode_indices") != expected_indices
        or result.get("stage") != "expert"
        or result.get("split") != TRAIN_SPLIT
        or gate.get("evaluated") is not True
        or gate.get("role") != "expert"
        or gate.get("gate_pass") is not True
    ):
        raise RuntimeError("expert ceiling receipt does not match the frozen expert role")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "file_sha256": _file_sha256(path),
        "receipt_sha256": receipt_sha256,
    }


def preflight(
    args: argparse.Namespace,
    *,
    release_verifier: Callable[..., Mapping] = _default_release_verifier,
) -> dict:
    """Verify formal release and path identities without Habitat/model creation."""

    _require_train_split(args.split)
    indices = _validate_common_args(args)
    ceiling = _load_ceiling_contract(args)

    manifest_path = _regular_file(args.release_manifest, name="release_manifest")
    source_root = _directory(args.source_root, name="source_root")
    checkpoint = _directory(args.checkpoint, name="checkpoint")
    tokenizer = _directory(args.tokenizer, name="tokenizer")
    vision = _directory(args.vision, name="vision")
    rlinf_root = _directory(args.rlinf_root, name="rlinf_root")
    asset_root = _directory(args.asset_root, name="asset_root")
    habitat_python = _preserved_absolute_file(
        args.habitat_python, name="habitat_python"
    )
    habitat_server = _regular_file(args.habitat_server, name="habitat_server")
    habitat_sim_python = _directory(
        args.habitat_sim_python, name="habitat_sim_python"
    )
    habitat_env_lib = _directory(args.habitat_env_lib, name="habitat_env_lib")
    egl_vendor_json = _regular_file(args.egl_vendor_json, name="egl_vendor_json")
    train_dataset = _regular_file(
        asset_root / "datasets" / "r2r" / TRAIN_SPLIT / f"{TRAIN_SPLIT}.json.gz",
        name="train_dataset",
    )
    if not (rlinf_root / "rlinf").is_dir():
        raise ValueError("rlinf_root does not contain the rlinf package")

    ceiling_manifest = ceiling["manifest"]
    release_contract = ceiling_manifest.get("release")
    if not isinstance(release_contract, Mapping):
        raise ValueError("ceiling manifest lacks the release contract")
    path_bindings = _verify_cli_path_contract(
        ceiling_manifest,
        ceiling_manifest=ceiling["path"],
        release_manifest=manifest_path,
        source_root=source_root,
        checkpoint=checkpoint,
        tokenizer=tokenizer,
        vision=vision,
        rlinf_root=rlinf_root,
        asset_root=asset_root,
        habitat_sim_python=habitat_sim_python,
        habitat_env_lib=habitat_env_lib,
        egl_vendor_json=egl_vendor_json,
    )
    release_manifest_raw_sha256 = _file_sha256(manifest_path)
    release_manifest = _read_manifest(manifest_path)
    release_manifest_canonical_sha256 = _json_hash(release_manifest)
    if (
        release_manifest_raw_sha256
        != _sha256_value(
            release_contract.get("release_manifest_raw_sha256"),
            name="release manifest raw SHA-256",
        )
        or release_manifest_canonical_sha256
        != _sha256_value(
            release_contract.get("release_manifest_canonical_sha256"),
            name="release manifest canonical SHA-256",
        )
    ):
        raise RuntimeError("release manifest bytes or canonical identity drifted")

    dataset_contract = ceiling_manifest.get("dataset")
    if not isinstance(dataset_contract, Mapping):
        raise ValueError("ceiling manifest lacks the train dataset contract")
    dataset_observation = _verify_file_record(
        train_dataset,
        {
            "bytes": dataset_contract.get("train_json_bytes"),
            "sha256": dataset_contract.get("train_json_sha256"),
        },
        name="train dataset",
    )
    expected_train_path = dataset_contract.get("train_json_path")
    if not isinstance(expected_train_path, str) or str(train_dataset) != expected_train_path:
        raise RuntimeError("train dataset path overrides the ceiling manifest")

    implementation_observations = _verify_implementation_targets(
        ceiling_manifest, rlinf_root
    )
    runtime_observation = _verify_runtime_contract(
        ceiling_manifest,
        habitat_python=habitat_python,
        habitat_server=habitat_server,
        habitat_sim_python=habitat_sim_python,
        habitat_env_lib=habitat_env_lib,
        egl_vendor_json=egl_vendor_json,
        implementation_observations=implementation_observations,
    )
    episode_roles = dataset_contract.get("episodes")
    if not isinstance(episode_roles, Mapping):
        raise ValueError("ceiling dataset lacks episode roles")
    expected_scenes = {
        row["scene"]
        for role in ("expert", "keep")
        for row in episode_roles[role]
    }
    scene_asset_observations = _verify_scene_assets(
        ceiling_manifest,
        asset_root=asset_root,
        expected_scenes=expected_scenes,
    )
    expert_receipt_observation = None
    if args.stage == "keep":
        expert_receipt_path = _regular_file(
            args.expert_receipt, name="expert_receipt"
        )
        expert_receipt_observation = _verify_expert_receipt(
            expert_receipt_path,
            ceiling_manifest_raw_sha256=ceiling["raw_sha256"],
            expert_rows=list(episode_roles["expert"]),
        )
    verified = dict(
        release_verifier(
            manifest=release_manifest,
            rlinf_root=rlinf_root,
            source_root=source_root,
            checkpoint=checkpoint,
            tokenizer=tokenizer,
            vision=vision,
        )
    )
    anchor = verified.get("anchor")
    verification = verified.get("verification")
    if (
        not isinstance(anchor, str)
        or len(anchor) != 64
        or anchor != release_manifest_canonical_sha256
        or not isinstance(verification, Mapping)
        or verification.get("manifest_sha256") != anchor
        or not isinstance(verification.get("source"), Mapping)
        or not isinstance(verification.get("artifacts"), Mapping)
    ):
        raise RuntimeError("release verifier returned an invalid provenance receipt")

    return {
        "status": "pass",
        "stage": "preflight",
        "split": TRAIN_SPLIT,
        "seed": int(args.seed),
        "max_steps": int(args.max_steps),
        "episode_indices": list(indices),
        "episode_expectations": list(ceiling["episodes"]),
        "ceiling_role": ceiling["role"],
        "gate_requirements": dict(ceiling["gate"]),
        "ceiling_manifest_raw_sha256": ceiling["raw_sha256"],
        "release_manifest_raw_sha256": release_manifest_raw_sha256,
        "release_manifest_canonical_sha256": release_manifest_canonical_sha256,
        "release_manifest_sha256": anchor,
        "source_manifest_sha256": _json_hash(verification["source"]),
        "artifact_manifest_sha256": _json_hash(verification["artifacts"]),
        "train_dataset": dataset_observation,
        "implementation_targets": implementation_observations,
        "path_bindings": path_bindings,
        "runtime_identity": runtime_observation,
        "scene_assets": scene_asset_observations,
        "expert_receipt": expert_receipt_observation,
        "paths": {
            "ceiling_manifest": str(ceiling["path"]),
            "manifest": str(manifest_path),
            "source_root": str(source_root),
            "checkpoint": str(checkpoint),
            "tokenizer": str(tokenizer),
            "vision": str(vision),
            "rlinf_root": str(rlinf_root),
            "asset_root": str(asset_root),
            "train_dataset": str(train_dataset),
            "habitat_python": str(habitat_python),
            "habitat_server": str(habitat_server),
            "habitat_sim_python": str(habitat_sim_python),
            "habitat_env_lib": str(habitat_env_lib),
            "egl_vendor_json": str(egl_vendor_json),
        },
        "guards": {
            "literal_train_only": True,
            "val_seen_reads": 0,
            "val_unseen_reads": 0,
            "model_load_count": 0,
            "optimizer_steps": 0,
            "ttt": False,
            "expert_gate_bound": args.stage != "keep"
            or expert_receipt_observation is not None,
        },
    }


@dataclass(frozen=True)
class ExpertRuntime:
    make_env: Callable
    route_name: object
    route_decision: Callable
    topology_factory: Callable[[int, object], object]
    feedback_from_dict: Callable
    successor_end_reason: Callable


@dataclass(frozen=True)
class KeepRuntime:
    make_env: Callable
    adapter_factory: Callable[..., object]
    bridge_factory: Callable[..., object]
    official_session_factory: Callable[..., object]
    route_name: object
    route_decision: Callable
    topology_factory: Callable[[int], object]


def _load_expert_runtime(rlinf_root: Path) -> ExpertRuntime:
    runtime = _import_canonical("streamvln_p8_successor_runtime", rlinf_root=rlinf_root)
    action = _import_canonical("streamvln_p8_successor_action", rlinf_root=rlinf_root)
    topology = _import_canonical("streamvln_topology", rlinf_root=rlinf_root)
    executor_contract = _import_canonical(
        "streamvln_executor_contract", rlinf_root=rlinf_root
    )
    executor = _import_canonical(
        "streamvln_p8_successor_executor", rlinf_root=rlinf_root
    )

    def topology_factory(max_steps: int, planner):
        return topology.OnlineTopology(
            max_decisions=max_steps,
            forward_step_m=float(planner.config.forward_step_m),
            turn_degrees=float(planner.config.turn_degrees),
        )

    return ExpertRuntime(
        make_env=runtime.make_successor_env,
        route_name=action.RouteName,
        route_decision=action.RouteDecision,
        topology_factory=topology_factory,
        feedback_from_dict=executor_contract.feedback_from_dict,
        successor_end_reason=executor.successor_end_reason,
    )


def _load_keep_runtime(rlinf_root: Path) -> KeepRuntime:
    runtime = _import_canonical("streamvln_p8_successor_runtime", rlinf_root=rlinf_root)
    adapter = _import_canonical("streamvln_p8_official_adapter", rlinf_root=rlinf_root)
    official = _import_canonical("streamvln_p8_official_keep", rlinf_root=rlinf_root)
    action = _import_canonical("streamvln_p8_successor_action", rlinf_root=rlinf_root)
    topology = _import_canonical("streamvln_topology", rlinf_root=rlinf_root)
    return KeepRuntime(
        make_env=runtime.make_successor_env,
        adapter_factory=adapter.PinnedStreamVLNGenerationAdapter.from_released_checkpoint,
        bridge_factory=runtime.SuccessorRunnerBridge,
        official_session_factory=official.OfficialEvalSession,
        route_name=action.RouteName,
        route_decision=action.RouteDecision,
        topology_factory=lambda max_steps: topology.OnlineTopology(
            max_decisions=max_steps,
            forward_step_m=0.25,
            turn_degrees=15.0,
        ),
    )


def _run_cfg(args: argparse.Namespace, preflight_receipt: Mapping) -> dict:
    paths = preflight_receipt["paths"]
    return {
        "seed": int(args.seed),
        "habitat_python": paths["habitat_python"],
        "habitat_server": paths["habitat_server"],
        "asset_root": paths["asset_root"],
        "habitat_sim_python": paths["habitat_sim_python"],
        "habitat_env_lib": paths["habitat_env_lib"],
        "egl_vendor_json": paths["egl_vendor_json"],
        "habitat_startup_timeout_s": float(args.habitat_startup_timeout_s),
    }


def _scalar(value: object, *, name: str):
    while isinstance(value, (list, tuple)):
        if not value:
            raise RuntimeError(f"ceiling environment returned empty {name}")
        value = value[0]
    if hasattr(value, "reshape"):
        value = value.reshape(-1)[0]
    value = value.item() if hasattr(value, "item") else value
    if isinstance(value, bool):
        return value
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise RuntimeError(f"ceiling environment returned invalid {name}")
    return value


def _identity(env, *, expected: Mapping) -> dict:
    raw = getattr(env, "identity", None)
    if not isinstance(raw, Mapping):
        raise RuntimeError("successor environment lacks episode identity")
    expected_index = expected["index"]
    if raw.get("episode_index") != expected_index:
        raise RuntimeError("successor environment selected the wrong train episode")
    result = {
        "episode_index": expected_index,
        "episode_id": raw.get("episode_id"),
        "scene_id": raw.get("scene_id"),
        "instruction_sha256": raw.get("instruction_sha256"),
    }
    if any(not isinstance(result[key], str) or not result[key] for key in result if key != "episode_index"):
        raise RuntimeError("successor environment identity is incomplete")
    if (
        result["episode_id"] != expected["episode_id"]
        or Path(result["scene_id"]).stem != expected["scene"]
    ):
        raise RuntimeError("successor environment identity drifted from the ceiling manifest")
    return result


def _frame_metadata(frame, *, expected_step: int, identity: Mapping) -> dict:
    trace = getattr(frame, "trace_metadata", None)
    if not callable(trace):
        raise TypeError("successor ceiling requires typed official frames")
    result = trace()
    if not isinstance(result, dict):
        raise RuntimeError("official frame metadata is not an object")
    if (
        result.get("physical_step") != expected_step
        or result.get("episode_id") != identity["episode_id"]
        or result.get("scene_id") != identity["scene_id"]
        or result.get("instruction_sha256") != identity["instruction_sha256"]
    ):
        raise RuntimeError("official frame identity drifted from the selected train state")
    return result


def _single_action_payload(action: int) -> list[list[list[int]]]:
    if type(action) is not int or not 0 <= action < ACTION_CHUNK:
        raise ValueError("successor ceiling primitive must be an integer in 0..3")
    return [[[action], *[[0] for _ in range(ACTION_CHUNK - 1)]]]


def _terminal_metrics(env, last_info: Mapping | None) -> dict:
    raw = None
    if isinstance(last_info, Mapping) and isinstance(last_info.get("episode"), Mapping):
        raw = last_info["episode"]
    if raw is None:
        candidate = getattr(env, "_terminal_metrics", None)
        if isinstance(candidate, Mapping):
            raw = candidate
    if not isinstance(raw, Mapping):
        raise RuntimeError("ceiling episode ended without terminal navigation metrics")
    result = {}
    for key, value in raw.items():
        scalar = _scalar(value, name=f"terminal metric {key}")
        result[str(key)] = bool(scalar) if isinstance(scalar, bool) else float(scalar)
    for required in ("success", "spl", "path_length", "collision_count"):
        if required not in result:
            raise RuntimeError(f"ceiling terminal metrics lack {required}")
    if float(result["success"]) not in (0.0, 1.0):
        raise RuntimeError("ceiling success metric must be binary")
    if not 0.0 <= float(result["spl"]) <= 1.0:
        raise RuntimeError("ceiling SPL metric must be in [0, 1]")
    if float(result["path_length"]) < 0.0:
        raise RuntimeError("ceiling path length must be nonnegative")
    collisions = float(result["collision_count"])
    if collisions < 0.0 or not collisions.is_integer():
        raise RuntimeError("ceiling collision count must be a nonnegative integer")
    for key in ("distance_to_goal", "navigation_error_m", "navigation_error"):
        if key in result and float(result[key]) < 0.0:
            raise RuntimeError(f"ceiling {key} must be nonnegative")
    return result


def _close_env(env) -> None:
    close = getattr(env, "close", None)
    if callable(close):
        close()


def _episode_actions(episode: Mapping) -> list[int]:
    actions = []
    for decision in episode.get("decisions", ()):
        values = decision.get("executed_actions", ())
        if not isinstance(values, (list, tuple)):
            raise RuntimeError("ceiling decision action trace is invalid")
        for value in values:
            if type(value) is not int or not 0 <= value < len(ACTION_NAMES):
                raise RuntimeError("ceiling action trace contains an invalid primitive")
            actions.append(value)
    return actions


def _navigation_aggregate(episodes: list[Mapping]) -> dict:
    if not episodes:
        raise RuntimeError("ceiling aggregate requires at least one episode")
    all_actions = [_episode_actions(episode) for episode in episodes]
    success_values = [float(episode["metrics"]["success"]) for episode in episodes]
    final_distances = []
    for episode in episodes:
        metrics = episode["metrics"]
        for key in ("distance_to_goal", "navigation_error_m", "navigation_error"):
            if key in metrics:
                final_distances.append(float(metrics[key]))
                break
    action_counts = {
        name: sum(action == index for actions in all_actions for action in actions)
        for index, name in enumerate(ACTION_NAMES)
    }
    successful_stop_count = sum(
        bool(actions)
        and actions[-1] == 0
        and bool(episode["terminated"])
        and float(episode["metrics"]["success"]) > 0.0
        for episode, actions in zip(episodes, all_actions)
    )
    mean_final_distance = (
        sum(final_distances) / len(final_distances) if final_distances else None
    )
    return {
        "episode_count": len(episodes),
        "scene_count": len({episode["identity"]["scene_id"] for episode in episodes}),
        "success_count": int(sum(success_values)),
        "success_rate": sum(success_values) / len(episodes),
        "successful_stop_count": successful_stop_count,
        "mean_spl": sum(float(episode["metrics"]["spl"]) for episode in episodes)
        / len(episodes),
        "total_path_length_m": sum(
            float(episode["metrics"]["path_length"]) for episode in episodes
        ),
        "collision_count": sum(
            float(episode["metrics"]["collision_count"]) for episode in episodes
        ),
        "mean_final_distance_m": mean_final_distance,
        "mean_navigation_error_m": mean_final_distance,
        "timeout_count": sum(bool(episode["truncated"]) for episode in episodes),
        "stop_count": action_counts["STOP"],
        "action_counts": action_counts,
    }


def run_expert_stage(
    args: argparse.Namespace,
    preflight_receipt: Mapping,
    *,
    runtime: ExpertRuntime | None = None,
) -> dict:
    """Run the privileged executor ceiling without constructing a model."""

    _require_train_split(args.split)
    runtime = runtime or _load_expert_runtime(
        Path(preflight_receipt["paths"]["rlinf_root"])
    )
    episodes = []
    run_cfg = _run_cfg(args, preflight_receipt)
    expectations = preflight_receipt["episode_expectations"]
    for role_position, episode_index in enumerate(_episode_indices(args.episode_index)):
        env = runtime.make_env(
            run_cfg,
            split=TRAIN_SPLIT,
            episode_index=episode_index,
            max_steps=int(args.max_steps),
        )
        try:
            env.reset()
            identity = _identity(env, expected=expectations[role_position])
            planner = getattr(env, "successor_planner", None)
            if planner is None or not callable(getattr(planner, "plan", None)):
                raise TypeError("successor environment lacks its production planner")
            topology = runtime.topology_factory(int(args.max_steps), planner)
            physical_step = 0
            decision_index = 0
            terminated = False
            truncated = False
            last_info = None
            frame_rows = []
            decision_rows = []

            while physical_step < int(args.max_steps) and not (terminated or truncated):
                expert = env.expert_waypoint(EXPERT_LOOKAHEAD_M)
                if not isinstance(expert, Mapping) or type(expert.get("stop")) is not bool:
                    raise RuntimeError("expert waypoint response is invalid")
                waypoint = expert.get("waypoint_local_m")
                if not isinstance(waypoint, (list, tuple)) or len(waypoint) != 2:
                    raise RuntimeError("expert waypoint response lacks a 2D target")
                route = runtime.route_name.STOP if expert["stop"] else runtime.route_name.WAYPOINT
                decision = runtime.route_decision(
                    decision_index=decision_index,
                    route=route,
                    waypoint_local_m=(0.0, 0.0) if expert["stop"] else tuple(waypoint),
                    episode_token=f"expert:{episode_index}",
                    physical_step=physical_step,
                )
                plan = planner.plan(decision, topology)
                executed_actions = []
                collisions = []
                blocked = []
                realized_path = []
                primitive_rows = []
                end_reason = None
                for primitive_index, action in enumerate(plan.proposed_actions):
                    if physical_step >= int(args.max_steps):
                        break
                    frame = env.observe_official_frame()
                    frame_rows.append(
                        _frame_metadata(
                            frame,
                            expected_step=physical_step,
                            identity=identity,
                        )
                    )
                    observations, _, terminations, truncations, infos = env.chunk_step(
                        _single_action_payload(int(action)), execute_count=1
                    )
                    if len(observations) != 1 or len(infos) != 1:
                        raise RuntimeError("successor ceiling requires a singleton environment")
                    feedback = runtime.feedback_from_dict(infos[0].get("executor_feedback"))
                    if len(feedback.primitives) != 1:
                        raise RuntimeError("expert ceiling did not execute exactly one primitive")
                    primitive = feedback.primitives[0]
                    if primitive.action != int(action):
                        raise RuntimeError("expert ceiling primitive identity drifted")
                    terminated = bool(_scalar(terminations, name="termination"))
                    truncated = bool(_scalar(truncations, name="truncation"))
                    if terminated != primitive.terminated or truncated != primitive.truncated:
                        raise RuntimeError("expert ceiling terminal feedback disagrees")
                    executed_actions.append(int(action))
                    collisions.append(bool(primitive.collision))
                    blocked.append(bool(primitive.blocked))
                    realized_path.append(float(primitive.realized_path_m))
                    primitive_rows.append(primitive.to_dict())
                    last_info = infos[0]
                    physical_step += 1
                    if physical_step > int(args.max_steps):
                        raise RuntimeError("expert ceiling exceeded its physical-step cap")
                    end_reason, _ = runtime.successor_end_reason(
                        action=int(action),
                        collision=bool(primitive.collision),
                        blocked=bool(primitive.blocked),
                        terminated=bool(primitive.terminated),
                        truncated=bool(primitive.truncated),
                        plan_complete=primitive_index == len(plan.proposed_actions) - 1,
                    )
                    if end_reason is not None:
                        break
                if not executed_actions:
                    raise RuntimeError("expert ceiling executed no primitive")
                before = int(getattr(topology, "decision_count", -1))
                topology.update(
                    executed_actions,
                    collisions,
                    realized_path_m=realized_path,
                    blocked_forwards=blocked,
                )
                if int(getattr(topology, "decision_count", -1)) != before + 1:
                    raise RuntimeError("expert topology must update exactly once per decision")
                decision_rows.append(
                    {
                        "decision_index": decision_index,
                        "physical_step_start": physical_step - len(executed_actions),
                        "expert": {
                            "source": expert.get("source"),
                            "stop": expert["stop"],
                            "waypoint_local_m": [float(value) for value in waypoint],
                            "distance_to_goal_m": float(expert["distance_to_goal_m"]),
                        },
                        "plan": plan.to_dict(),
                        "executed_actions": executed_actions,
                        "primitive_feedback": primitive_rows,
                        "end_reason": end_reason,
                    }
                )
                decision_index += 1

            if not (terminated or truncated):
                raise RuntimeError("expert ceiling exhausted its fixed budget without termination")
            episodes.append(
                {
                    "identity": identity,
                    "physical_steps": physical_step,
                    "decision_count": decision_index,
                    "terminated": terminated,
                    "truncated": truncated,
                    "official_frames": frame_rows,
                    "decisions": decision_rows,
                    "metrics": _terminal_metrics(env, last_info),
                }
            )
        finally:
            _close_env(env)

    return {
        "status": "pass",
        "stage": "expert",
        "split": TRAIN_SPLIT,
        "seed": int(args.seed),
        "max_steps": int(args.max_steps),
        "expert_lookahead_m": EXPERT_LOOKAHEAD_M,
        "episodes": episodes,
        "aggregate": _navigation_aggregate(episodes),
        "guards": {
            "literal_train_only": True,
            "val_seen_reads": 0,
            "val_unseen_reads": 0,
            "model_load_count": 0,
            "optimizer_steps": 0,
            "ttt": False,
            "one_primitive_round_trip": True,
            "causal_topology_update": True,
        },
    }


def run_keep_stage(
    args: argparse.Namespace,
    preflight_receipt: Mapping,
    *,
    runtime: KeepRuntime | None = None,
) -> dict:
    """Run exact KEEP; this is the sole model-construction path in this module."""

    _require_train_split(args.split)
    runtime = runtime or _load_keep_runtime(
        Path(preflight_receipt["paths"]["rlinf_root"])
    )
    manifest = _read_manifest(Path(preflight_receipt["paths"]["manifest"]))
    adapter = runtime.adapter_factory(
        model_path=preflight_receipt["paths"]["checkpoint"],
        tokenizer_path=preflight_receipt["paths"]["tokenizer"],
        vision_tower_path=preflight_receipt["paths"]["vision"],
        source_root=preflight_receipt["paths"]["source_root"],
        artifact_manifest=manifest,
        device=args.device,
        local_rank=int(args.local_rank),
        env_count=1,
        model_env_id=0,
    )
    episodes = []
    run_cfg = _run_cfg(args, preflight_receipt)
    expectations = preflight_receipt["episode_expectations"]
    for role_position, episode_index in enumerate(_episode_indices(args.episode_index)):
        env = runtime.make_env(
            run_cfg,
            split=TRAIN_SPLIT,
            episode_index=episode_index,
            max_steps=int(args.max_steps),
        )
        try:
            def official_driver_factory(env_id: int, identity: Mapping, episode_serial: int):
                session = runtime.official_session_factory(
                    env_id=env_id,
                    instruction=identity["instruction_text"],
                    conjunction_rng=random.Random(int(args.seed) + episode_index),
                )
                return adapter.make_driver(
                    session=session,
                    episode_serial=episode_serial,
                    episode_identity=identity,
                )

            bridge = runtime.bridge_factory(
                env=env,
                topology_factory=lambda: runtime.topology_factory(int(args.max_steps)),
                official_driver_factory=official_driver_factory,
                exact_control=True,
            )
            bridge.reset_episode()
            identity = _identity(env, expected=expectations[role_position])
            decision_rows = []
            mismatch_count = 0
            terminated = False
            truncated = False
            for decision_index in range(int(args.max_steps)):
                driver = getattr(bridge, "official_driver", None)
                session = getattr(driver, "session", None)
                physical_step = getattr(session, "physical_step", None)
                token = getattr(bridge, "episode_token", None)
                if type(physical_step) is not int or not isinstance(token, str):
                    raise RuntimeError("exact KEEP bridge lacks live episode identity")
                decision = runtime.route_decision(
                    decision_index=decision_index,
                    route=runtime.route_name.KEEP_ORIGINAL,
                    episode_token=token,
                    physical_step=physical_step,
                )
                result = bridge.execute_decision(decision)
                trace = result.to_trace()
                if not isinstance(trace, Mapping):
                    raise RuntimeError("exact KEEP bridge trace is not an object")
                executed_actions = trace.get("executed_actions")
                official_shadow_actions = trace.get("official_shadow_actions")
                terminated = bool(result.terminated)
                truncated = bool(result.truncated)
                if (
                    trace.get("episode_token") != token
                    or trace.get("decision_physical_step") != physical_step
                    or trace.get("plan", {}).get("requested_route")
                    != runtime.route_name.KEEP_ORIGINAL.value
                    or trace.get("plan", {}).get("executed_route")
                    != runtime.route_name.KEEP_ORIGINAL.value
                    or not isinstance(executed_actions, (list, tuple))
                    or len(executed_actions) != 1
                    or not isinstance(official_shadow_actions, (list, tuple))
                    or len(official_shadow_actions) != 1
                    or any(
                        type(action) is not int or not 0 <= action < ACTION_CHUNK
                        for action in (*executed_actions, *official_shadow_actions)
                    )
                    or len(trace.get("official_frame_metadata", ())) != 1
                    or trace.get("terminated") is not terminated
                    or trace.get("truncated") is not truncated
                ):
                    raise RuntimeError("exact KEEP execution identity drifted")
                frame = trace["official_frame_metadata"][0]
                if (
                    frame.get("physical_step") != physical_step
                    or frame.get("episode_id") != identity["episode_id"]
                    or frame.get("scene_id") != identity["scene_id"]
                    or frame.get("instruction_sha256") != identity["instruction_sha256"]
                ):
                    raise RuntimeError("exact KEEP official frame identity drifted")
                action_match = list(executed_actions) == list(official_shadow_actions)
                mismatch_count += int(not action_match)
                trace = dict(trace)
                trace["action_comparison"] = {
                    "executed_actions": list(executed_actions),
                    "official_shadow_actions": list(official_shadow_actions),
                    "match": action_match,
                }
                decision_rows.append(trace)
                if terminated or truncated:
                    break
            if not (terminated or truncated):
                raise RuntimeError("exact KEEP exhausted its fixed budget without termination")
            episodes.append(
                {
                    "identity": identity,
                    "episode_token": bridge.episode_token,
                    "decision_count": len(decision_rows),
                    "physical_steps": sum(len(row["executed_actions"]) for row in decision_rows),
                    "terminated": terminated,
                    "truncated": truncated,
                    "decisions": decision_rows,
                    "comparison_count": len(decision_rows),
                    "mismatch_count": mismatch_count,
                    "metrics": _terminal_metrics(env, None),
                }
            )
        finally:
            _close_env(env)

    aggregate = _navigation_aggregate(episodes)
    aggregate["comparison_count"] = sum(
        int(episode["comparison_count"]) for episode in episodes
    )
    aggregate["mismatch_count"] = sum(
        int(episode["mismatch_count"]) for episode in episodes
    )
    return {
        "status": "pass",
        "stage": "keep",
        "split": TRAIN_SPLIT,
        "seed": int(args.seed),
        "max_steps": int(args.max_steps),
        "episodes": episodes,
        "aggregate": aggregate,
        "guards": {
            "literal_train_only": True,
            "val_seen_reads": 0,
            "val_unseen_reads": 0,
            "model_load_count": 1,
            "optimizer_steps": 0,
            "bc": False,
            "ppo": False,
            "ttt": False,
            "exact_control": True,
            "keep_only": True,
        },
    }


def _evaluate_stage_gate(
    *, stage_result: Mapping, preflight_receipt: Mapping
) -> dict:
    stage = stage_result["stage"]
    if stage == "preflight":
        return {
            "evaluated": False,
            "gate_pass": None,
            "role": preflight_receipt["ceiling_role"],
        }
    role = preflight_receipt["ceiling_role"]
    if stage != role:
        raise RuntimeError("ceiling result stage does not match its frozen role")
    required = dict(preflight_receipt["gate_requirements"])
    aggregate = stage_result.get("aggregate")
    episodes = stage_result.get("episodes")
    if not isinstance(aggregate, Mapping) or not isinstance(episodes, list):
        raise RuntimeError("ceiling stage lacks typed gate observations")
    if role == "expert":
        maximum_physical_steps = max(
            (int(episode["physical_steps"]) for episode in episodes), default=0
        )
        observed = {
            "episode_count": aggregate.get("episode_count"),
            "success_count": aggregate.get("success_count"),
            "successful_stop_count": aggregate.get("successful_stop_count"),
            "maximum_physical_steps_per_episode": maximum_physical_steps,
        }
        conditions = {
            "episode_count": observed["episode_count"] == required["episode_count"],
            "minimum_successes": observed["success_count"]
            >= required["minimum_successes"],
            "minimum_successful_stops": observed["successful_stop_count"]
            >= required["minimum_successful_stops"],
            "max_physical_steps_per_episode": maximum_physical_steps
            <= required["max_physical_steps_per_episode"],
        }
    else:
        comparisons = [int(episode["comparison_count"]) for episode in episodes]
        maximum_physical_steps = max(
            (int(episode["physical_steps"]) for episode in episodes), default=0
        )
        observed = {
            "episode_count": aggregate.get("episode_count"),
            "comparisons_per_episode": comparisons,
            "minimum_comparisons_per_episode": min(comparisons) if comparisons else 0,
            "mismatch_count": aggregate.get("mismatch_count"),
            "maximum_physical_steps_per_episode": maximum_physical_steps,
        }
        conditions = {
            "episode_count": observed["episode_count"] == required["episode_count"],
            "minimum_comparisons_per_episode": observed[
                "minimum_comparisons_per_episode"
            ]
            >= required["minimum_comparisons_per_episode"],
            "mismatch_count": observed["mismatch_count"]
            == required["required_mismatch_count"],
            "max_physical_steps_per_episode": maximum_physical_steps
            <= required["max_physical_steps_per_episode"],
        }
    return {
        "evaluated": True,
        "role": role,
        "required": required,
        "observed": observed,
        "conditions": conditions,
        "gate_pass": all(conditions.values()),
    }


def _finalize_receipt(*, stage_result: Mapping, preflight_receipt: Mapping) -> dict:
    result = dict(stage_result)
    if result.get("status") != "pass" or result.get("stage") not in STAGES:
        raise RuntimeError("ceiling stage did not return a passing typed result")
    gate = _evaluate_stage_gate(
        stage_result=result, preflight_receipt=preflight_receipt
    )
    receipt = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "stage": result["stage"],
        "ceiling_manifest_raw_sha256": preflight_receipt[
            "ceiling_manifest_raw_sha256"
        ],
        "gate_pass": gate["gate_pass"],
        "gate": gate,
        "preflight": dict(preflight_receipt),
        "result": result,
    }
    receipt["receipt_sha256"] = _json_hash(receipt)
    return receipt


def _atomic_write(path_value: object, receipt: Mapping) -> None:
    path = Path(str(path_value)).expanduser().absolute()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite ceiling receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("xb") as stream:
            stream.write(_json_bytes(receipt) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run(
    args: argparse.Namespace,
    *,
    release_verifier: Callable[..., Mapping] = _default_release_verifier,
    expert_runtime: ExpertRuntime | None = None,
    keep_runtime: KeepRuntime | None = None,
) -> dict:
    # Keep this as the first operation.  Tests guard that rejected splits do not
    # reach path verification, dataset access, Habitat imports, or model imports.
    _require_train_split(args.split)
    preflight_receipt = preflight(args, release_verifier=release_verifier)
    if args.stage == "preflight":
        stage_result = preflight_receipt
    elif args.stage == "expert":
        stage_result = run_expert_stage(
            args, preflight_receipt, runtime=expert_runtime
        )
    elif args.stage == "keep":
        stage_result = run_keep_stage(args, preflight_receipt, runtime=keep_runtime)
    else:
        raise ValueError(f"unknown P8 successor ceiling stage: {args.stage!r}")
    receipt = _finalize_receipt(
        stage_result=stage_result,
        preflight_receipt=preflight_receipt,
    )
    _atomic_write(args.output, receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ceiling-manifest", required=True)
    parser.add_argument("--release-manifest", required=True)
    parser.add_argument("--expert-receipt")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--vision", required=True)
    parser.add_argument("--rlinf-root", required=True)
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--habitat-python", required=True)
    parser.add_argument("--habitat-server", required=True)
    parser.add_argument("--habitat-sim-python", required=True)
    parser.add_argument("--habitat-env-lib", required=True)
    parser.add_argument("--egl-vendor-json", required=True)
    parser.add_argument("--episode-index", action="append", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--habitat-startup-timeout-s", type=float, default=180.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--local-rank", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        run(build_parser().parse_args(argv))
    except Exception as exc:
        print(f"P8 successor ceiling failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
