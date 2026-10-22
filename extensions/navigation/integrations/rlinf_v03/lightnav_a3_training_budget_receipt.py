#!/usr/bin/env python3
"""Verify the immutable LightNav A3 archive and publish its training budget.

The OBS tree is read-only to this program.  The only write is the requested
create-only ``A3_TRAINING_BUDGET_RECEIPT.json`` outside the OBS root.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Mapping, Sequence


RECEIPT_SCHEMA = "navigation_rlt_a3_training_budget_receipt"
RECEIPT_FILENAME = "A3_TRAINING_BUDGET_RECEIPT.json"
ARCHIVE_SCHEMA = "lightnav_a3_obs_archive"
ARCHIVE_COMPLETION_SCHEMA = "lightnav_a3_obs_completion"
ROLLOUT_SCHEMA = "lightnav_a3_rollout"
TRAINING_SCHEMA = "lightnav_a3_rlinf_fsdp_training"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
A800_PATTERN = re.compile(r"^NVIDIA A800\b", re.IGNORECASE)
CHECKPOINT_HARD_CAP_BYTES = 2 * 1024**3
RUN_HARD_CAP_BYTES = 16 * 1024**3
PRODUCTION_ROLLOUT_PARENT = "input/train_rollout_receipt"
PRODUCTION_ROLLOUT_BUDGET_ENTRIES = frozenset(
    {"actor_decisions.jsonl", "result.json"}
)
PRODUCTION_ROLLOUT_DIAGNOSTIC_ENTRIES = frozenset(
    {"episode_progress.jsonl", "execution_trace.jsonl"}
)


@dataclass(frozen=True)
class A3BudgetContract:
    run_id: str
    source_run_root: str
    final_manifest_sha256: str
    completion_sha256: str
    run_manifest_sha256: str
    split_manifest_sha256: str
    replay_sha256: str
    checkpoint_sha256: str
    manifest_file_count: int
    source_total_bytes: int
    rollout_episode_count: int
    evaluation_episode_count: int
    unique_actor_decisions: int
    optimizer_epochs: int
    world_size: int
    batch_size_per_rank: int
    optimizer_steps_per_rank: int
    checkpoint_size_bytes: int


PRODUCTION_CONTRACT = A3BudgetContract(
    run_id="lightnav-a3-20260930-v1",
    source_run_root=(
        "/srv/navigation/runs/lightnav-a3-20260930-v1"
    ),
    final_manifest_sha256=(
        "1a485f6ef877d987073bfdebee5766f21747919eac897400c5f0384a237570bb"
    ),
    completion_sha256=(
        "7ca90bfaa745600bc97d550b15bfeac5a74e8f9b4a3675f350d868c4478cfb98"
    ),
    run_manifest_sha256=(
        "85169e0495e7e26044223b2b66d9ac14c654b1da43da389fbfce489bb310df51"
    ),
    split_manifest_sha256=(
        "beaf5ebc44dca6729dd5c7431eba0675dd7c8ddb3c4ebdae3ad410de9fa73a38"
    ),
    replay_sha256=(
        "e1e6b4461c690f7e7c58cda7fa5db65eeb0ff9064e186db006eb76d5e5cd6be4"
    ),
    checkpoint_sha256=(
        "f9a65a6b54efa53422e72a6918a8d6076e5893f821d68b2864b0e420e81c16e1"
    ),
    manifest_file_count=130,
    source_total_bytes=19_479_434,
    rollout_episode_count=64,
    evaluation_episode_count=32,
    unique_actor_decisions=4_642,
    optimizer_epochs=4,
    world_size=8,
    batch_size_per_rank=128,
    optimizer_steps_per_rank=20,
    checkpoint_size_bytes=314_226,
)


class A3BudgetReceiptError(RuntimeError):
    """Raised when the archived A3 evidence cannot support the receipt."""


@dataclass(frozen=True)
class ArchiveRecord:
    relative_source: str
    source: str
    destination: str
    path: Path
    size_bytes: int
    sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    raise A3BudgetReceiptError(f"non-finite JSON constant is forbidden: {value}")


def _load_json(path: Path, name: str) -> dict[str, object]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=_reject_constant
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise A3BudgetReceiptError(f"{name} is not strict JSON") from error
    if not isinstance(value, dict):
        raise A3BudgetReceiptError(f"{name} must contain an object")
    return value


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise A3BudgetReceiptError(f"{name} must be an object")
    return value


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise A3BudgetReceiptError(f"{name} is outside its valid range")
    return value


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise A3BudgetReceiptError(f"{name} is not a lowercase SHA-256")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise A3BudgetReceiptError(f"{name} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise A3BudgetReceiptError(f"{name} is not finite")
    return result


def _regular_file(path: Path, name: str) -> Path:
    raw = Path(path).expanduser().absolute()
    if raw.is_symlink():
        raise A3BudgetReceiptError(f"{name} must not be a symlink")
    try:
        resolved = raw.resolve(strict=True)
    except OSError as error:
        raise A3BudgetReceiptError(f"{name} is missing") from error
    if not resolved.is_file():
        raise A3BudgetReceiptError(f"{name} is not a regular file")
    return resolved


def _under(path: Path, root: Path, name: str) -> Path:
    try:
        path.relative_to(root)
    except ValueError as error:
        raise A3BudgetReceiptError(f"{name} is outside the OBS root") from error
    return path


def _relative_source(source: str, root: str) -> str:
    source_path = Path(source)
    root_path = Path(root)
    try:
        relative = source_path.relative_to(root_path)
    except ValueError as error:
        raise A3BudgetReceiptError("archive source is outside source_run_root") from error
    if not relative.parts or any(part in ("", ".", "..") for part in relative.parts):
        raise A3BudgetReceiptError("archive source relative path is invalid")
    return relative.as_posix()


def _read_archive(
    final_manifest: Path,
    completion: Path,
    obs_root: Path,
    contract: A3BudgetContract,
) -> tuple[
    dict[str, object],
    dict[str, ArchiveRecord],
    dict[str, list[ArchiveRecord]],
]:
    root = Path(obs_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise A3BudgetReceiptError("OBS root is not a directory")
    manifest_path = _under(
        _regular_file(final_manifest, "final archive manifest"), root, "manifest"
    )
    completion_path = _under(
        _regular_file(completion, "final archive completion"), root, "completion"
    )
    if sha256_file(manifest_path) != contract.final_manifest_sha256:
        raise A3BudgetReceiptError("final archive manifest SHA-256 differs")
    if sha256_file(completion_path) != contract.completion_sha256:
        raise A3BudgetReceiptError("final archive completion SHA-256 differs")

    manifest = _load_json(manifest_path, "final archive manifest")
    expected_manifest_keys = {
        "schema_name",
        "schema_version",
        "status",
        "phase",
        "run_id",
        "source_run_root",
        "source_total_bytes",
        "checkpoint_hard_cap_bytes",
        "run_hard_cap_bytes",
        "files",
        "completed_unix_s",
    }
    if set(manifest) != expected_manifest_keys:
        raise A3BudgetReceiptError("final archive manifest field inventory differs")
    if (
        manifest.get("schema_name") != ARCHIVE_SCHEMA
        or manifest.get("schema_version") != 1
        or manifest.get("status") != "completed"
        or manifest.get("phase") != "final"
        or manifest.get("run_id") != contract.run_id
        or manifest.get("source_run_root") != contract.source_run_root
        or manifest.get("source_total_bytes") != contract.source_total_bytes
        or manifest.get("checkpoint_hard_cap_bytes") != CHECKPOINT_HARD_CAP_BYTES
        or manifest.get("run_hard_cap_bytes") != RUN_HARD_CAP_BYTES
    ):
        raise A3BudgetReceiptError("final archive manifest contract differs")
    _finite(manifest.get("completed_unix_s"), "manifest completion time")
    raw_records = manifest.get("files")
    if not isinstance(raw_records, list) or len(raw_records) != contract.manifest_file_count:
        raise A3BudgetReceiptError("final archive file count differs")

    by_relative: dict[str, ArchiveRecord] = {}
    by_hash: dict[str, list[ArchiveRecord]] = {}
    seen_destinations: set[Path] = set()
    total_bytes = 0
    for index, raw_record in enumerate(raw_records):
        record = _mapping(raw_record, f"archive record {index}")
        if set(record) != {
            "source",
            "destination",
            "size_bytes",
            "sha256",
            "readback_sha256",
        }:
            raise A3BudgetReceiptError(f"archive record {index} fields differ")
        source = record.get("source")
        destination = record.get("destination")
        if not isinstance(source, str) or not isinstance(destination, str):
            raise A3BudgetReceiptError(f"archive record {index} paths are invalid")
        relative = _relative_source(source, contract.source_run_root)
        if relative in by_relative:
            raise A3BudgetReceiptError(f"duplicate archive source: {relative}")
        expected_hash = _sha256(record.get("sha256"), f"archive record {index} hash")
        if record.get("readback_sha256") != expected_hash:
            raise A3BudgetReceiptError(f"archive record {index} readback hash differs")
        size_bytes = _integer(
            record.get("size_bytes"), f"archive record {index} size"
        )
        actual = _under(
            _regular_file(Path(destination), f"archive record {index} destination"),
            root,
            f"archive record {index} destination",
        )
        if actual in seen_destinations:
            raise A3BudgetReceiptError("duplicate archive destination")
        seen_destinations.add(actual)
        if actual.stat().st_size != size_bytes or sha256_file(actual) != expected_hash:
            raise A3BudgetReceiptError(f"archive record {index} readback differs")
        normalized = ArchiveRecord(
            relative_source=relative,
            source=source,
            destination=destination,
            path=actual,
            size_bytes=size_bytes,
            sha256=expected_hash,
        )
        by_relative[relative] = normalized
        by_hash.setdefault(expected_hash, []).append(normalized)
        total_bytes += size_bytes
    if total_bytes != contract.source_total_bytes:
        raise A3BudgetReceiptError("archive record sizes do not equal source_total_bytes")

    marker = _load_json(completion_path, "final archive completion")
    if set(marker) != {
        "schema_name",
        "schema_version",
        "phase",
        "manifest",
        "manifest_sha256",
        "completed_unix_s",
    }:
        raise A3BudgetReceiptError("final archive completion fields differ")
    if (
        marker.get("schema_name") != ARCHIVE_COMPLETION_SCHEMA
        or marker.get("schema_version") != 1
        or marker.get("phase") != "final"
        or marker.get("manifest_sha256") != contract.final_manifest_sha256
    ):
        raise A3BudgetReceiptError("final archive completion contract differs")
    marker_manifest = marker.get("manifest")
    if not isinstance(marker_manifest, str):
        raise A3BudgetReceiptError("completion manifest path is invalid")
    if _regular_file(Path(marker_manifest), "completion manifest") != manifest_path:
        raise A3BudgetReceiptError("completion points to a different manifest")
    _finite(marker.get("completed_unix_s"), "completion time")
    return manifest, by_relative, by_hash


def _required_record(
    records: Mapping[str, ArchiveRecord], relative: str, name: str
) -> ArchiveRecord:
    try:
        return records[relative]
    except KeyError as error:
        raise A3BudgetReceiptError(f"{name} is absent from the final archive") from error


def _require_hash_record(
    records: Mapping[str, ArchiveRecord],
    hashes: Mapping[str, list[ArchiveRecord]],
    relative: str,
    expected_hash: str,
    name: str,
) -> ArchiveRecord:
    if expected_hash not in hashes:
        raise A3BudgetReceiptError(f"{name} hash is absent from the final archive")
    record = _required_record(records, relative, name)
    if record.sha256 != expected_hash:
        raise A3BudgetReceiptError(f"{name} SHA-256 differs")
    return record


def _verify_run_and_split(
    records: Mapping[str, ArchiveRecord],
    hashes: Mapping[str, list[ArchiveRecord]],
    contract: A3BudgetContract,
) -> None:
    run_record = _require_hash_record(
        records,
        hashes,
        "input/run_manifest.json",
        contract.run_manifest_sha256,
        "run manifest",
    )
    run_manifest = _load_json(run_record.path, "run manifest")
    if (
        run_manifest.get("schema_name") != "lightnav_a3_run_manifest"
        or run_manifest.get("schema_version") != 1
        or run_manifest.get("run_id") != contract.run_id
        or run_manifest.get("upstream_split") != "train"
        or run_manifest.get("validation_split_reads") != 0
    ):
        raise A3BudgetReceiptError("run manifest scope differs")
    split_reference = _mapping(run_manifest.get("split_manifest"), "run split manifest")
    if split_reference.get("sha256") != contract.split_manifest_sha256:
        raise A3BudgetReceiptError("run manifest split SHA-256 differs")
    ppo = _mapping(run_manifest.get("ppo"), "run PPO settings")
    if (
        ppo.get("epochs") != contract.optimizer_epochs
        or ppo.get("batch_size_per_rank") != contract.batch_size_per_rank
        or ppo.get("required_world_size") != contract.world_size
    ):
        raise A3BudgetReceiptError("run manifest PPO budget differs")
    policy = _mapping(run_manifest.get("policy"), "run policy")
    if policy.get("native_stop_actor_credit") is not False:
        raise A3BudgetReceiptError("run manifest native STOP credit differs")
    code = _mapping(run_manifest.get("code"), "run code inventory")
    trainer_code = _mapping(
        code.get("lightnav_a3_fsdp_train.py"), "trainer code identity"
    )
    trainer_hash = _sha256(trainer_code.get("sha256"), "trainer code SHA-256")
    archived_trainer = _required_record(
        records, "code/lightnav_a3_fsdp_train.py", "archived trainer code"
    )
    if archived_trainer.sha256 != trainer_hash:
        raise A3BudgetReceiptError("archived trainer code differs from run manifest")

    candidates = hashes.get(contract.split_manifest_sha256, [])
    matching_splits: list[dict[str, object]] = []
    for record in candidates:
        try:
            value = _load_json(record.path, "split manifest candidate")
        except A3BudgetReceiptError:
            continue
        if value.get("schema_name") == "lightnav_a3_train_scene_split":
            matching_splits.append(value)
    if not matching_splits:
        raise A3BudgetReceiptError("split manifest hash lacks a valid split contract")
    for split in matching_splits:
        splits = _mapping(split.get("splits"), "split manifest splits")
        update = _mapping(splits.get("train_update"), "train_update split")
        development = _mapping(
            splits.get("train_development"), "train_development split"
        )
        if (
            split.get("schema_version") != 1
            or split.get("upstream_split_read") != "train"
            or split.get("validation_split_reads") != 0
            or update.get("habitat_split") != "train"
            or update.get("episode_count") != contract.rollout_episode_count
            or development.get("habitat_split") != "train"
            or development.get("episode_count") != contract.evaluation_episode_count
        ):
            raise A3BudgetReceiptError("split manifest contract differs")


def _read_actor_rows(path: Path, contract: A3BudgetContract) -> None:
    row_count = 0
    identities: set[tuple[int, str, str, int]] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise A3BudgetReceiptError(
                    f"actor replay contains a blank line at {line_number}"
                )
            try:
                row = json.loads(line, parse_constant=_reject_constant)
            except json.JSONDecodeError as error:
                raise A3BudgetReceiptError(
                    f"actor replay line {line_number} is not strict JSON"
                ) from error
            if not isinstance(row, dict):
                raise A3BudgetReceiptError("actor replay row is not an object")
            if (
                row.get("actor_eligible") is not True
                or type(row.get("policy_version")) is not int
                or row.get("policy_version") != 0
                or row.get("native_stop") is not False
                or row.get("external_forced_stop") is not False
            ):
                raise A3BudgetReceiptError("actor replay eligibility semantics differ")
            episode_order = _integer(
                row.get("episode_order"), "actor replay episode order"
            )
            step = _integer(row.get("step"), "actor replay step")
            episode_id = row.get("episode_id")
            scene_id = row.get("scene_id")
            if not isinstance(episode_id, str) or not episode_id:
                raise A3BudgetReceiptError("actor replay episode identity is invalid")
            if not isinstance(scene_id, str) or not scene_id:
                raise A3BudgetReceiptError("actor replay scene identity is invalid")
            identity = (episode_order, scene_id, episode_id, step)
            if identity in identities:
                raise A3BudgetReceiptError("actor replay contains a duplicate physical step")
            identities.add(identity)
            row_count += 1
    if row_count != contract.unique_actor_decisions:
        raise A3BudgetReceiptError("actor replay row count differs")


def _resolve_rollout_actor_record(
    records: Mapping[str, ArchiveRecord],
    parent: str,
    canonical_actor_replay: ArchiveRecord,
) -> ArchiveRecord:
    archived_relative = f"{parent}/actor_decisions.jsonl"
    exact = records.get(archived_relative)
    if parent != PRODUCTION_ROLLOUT_PARENT:
        return exact or _required_record(
            records, archived_relative, "rollout replay readback"
        )
    if canonical_actor_replay.relative_source != "input/actor_decisions.jsonl":
        raise A3BudgetReceiptError("canonical actor replay archive path differs")
    if exact is not None:
        raise A3BudgetReceiptError("rollout actor replay mapping is ambiguous")
    return canonical_actor_replay


def _verify_inventory(
    inventory: Mapping[str, object],
    records: Mapping[str, ArchiveRecord],
    parent: str,
    name: str,
    *,
    canonical_actor_replay: ArchiveRecord | None = None,
) -> None:
    for relative, raw_receipt in inventory.items():
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise A3BudgetReceiptError(f"{name} contains an invalid relative path")
        if any(part in ("", ".", "..") for part in Path(relative).parts):
            raise A3BudgetReceiptError(f"{name} contains path traversal")
        receipt = _mapping(raw_receipt, f"{name} entry")
        if set(receipt) != {"sha256", "size_bytes"}:
            raise A3BudgetReceiptError(f"{name} entry fields differ")
        archived_relative = f"{parent}/{Path(relative).as_posix()}"
        if relative == "actor_decisions.jsonl" and canonical_actor_replay is not None:
            archived = _resolve_rollout_actor_record(
                records, parent, canonical_actor_replay
            )
        else:
            archived = _required_record(records, archived_relative, f"{name} artifact")
        if (
            receipt.get("sha256") != archived.sha256
            or receipt.get("size_bytes") != archived.size_bytes
        ):
            raise A3BudgetReceiptError(f"{name} artifact readback differs")


def _validated_inventory_entry(
    inventory: Mapping[str, object], relative: str, name: str
) -> dict[str, object]:
    receipt = _mapping(inventory.get(relative), f"{name} entry")
    if set(receipt) != {"sha256", "size_bytes"}:
        raise A3BudgetReceiptError(f"{name} entry fields differ")
    return {
        "sha256": _sha256(receipt.get("sha256"), f"{name} SHA-256"),
        "size_bytes": _integer(receipt.get("size_bytes"), f"{name} size"),
    }


def _coverage_record(record: ArchiveRecord) -> dict[str, object]:
    return {
        "archive_relative": record.relative_source,
        "sha256": record.sha256,
        "size_bytes": record.size_bytes,
        "payload_readback": True,
    }


def _verify_production_rollout_inventory(
    inventory: Mapping[str, object],
    records: Mapping[str, ArchiveRecord],
    parent: str,
    inventory_record: ArchiveRecord,
    replay: ArchiveRecord,
    result_record: ArchiveRecord,
) -> dict[str, object]:
    expected = (
        PRODUCTION_ROLLOUT_BUDGET_ENTRIES
        | PRODUCTION_ROLLOUT_DIAGNOSTIC_ENTRIES
    )
    if parent != PRODUCTION_ROLLOUT_PARENT or set(inventory) != expected:
        raise A3BudgetReceiptError("production rollout inventory file set differs")
    entries = {
        relative: _validated_inventory_entry(
            inventory, relative, f"rollout inventory {relative}"
        )
        for relative in sorted(expected)
    }
    actor_record = _resolve_rollout_actor_record(records, parent, replay)
    actor_receipt = entries["actor_decisions.jsonl"]
    if (
        actor_receipt["sha256"] != actor_record.sha256
        or actor_receipt["size_bytes"] != actor_record.size_bytes
    ):
        raise A3BudgetReceiptError("rollout actor budget artifact readback differs")
    if result_record.relative_source != f"{parent}/result.json":
        raise A3BudgetReceiptError("rollout result archive path differs")
    result_receipt = entries["result.json"]
    if (
        result_receipt["sha256"] != result_record.sha256
        or result_receipt["size_bytes"] != result_record.size_bytes
    ):
        raise A3BudgetReceiptError("rollout result budget artifact readback differs")
    diagnostics = {
        relative: {
            **entries[relative],
            "payload_readback": False,
        }
        for relative in sorted(PRODUCTION_ROLLOUT_DIAGNOSTIC_ENTRIES)
    }
    return {
        "schema_name": "lightnav_a3_rollout_inventory_coverage",
        "schema_version": 1,
        "rollout_parent": parent,
        "sealed_inventory": _coverage_record(inventory_record),
        "budget_required_entries": {
            "actor_decisions.jsonl": _coverage_record(actor_record),
            "result.json": _coverage_record(result_record),
        },
        "unarchived_diagnostic_entries": diagnostics,
        "full_rollout_inventory_readback": False,
    }


def _legacy_rollout_inventory_coverage(
    parent: str,
    inventory_record: ArchiveRecord,
    actor_record: ArchiveRecord,
    result_record: ArchiveRecord,
) -> dict[str, object]:
    return {
        "schema_name": "lightnav_a3_rollout_inventory_coverage",
        "schema_version": 1,
        "rollout_parent": parent,
        "sealed_inventory": _coverage_record(inventory_record),
        "budget_required_entries": {
            "actor_decisions.jsonl": _coverage_record(actor_record),
            "result.json": _coverage_record(result_record),
        },
        "unarchived_diagnostic_entries": {},
        "full_rollout_inventory_readback": True,
    }


def _verify_rollout(
    records: Mapping[str, ArchiveRecord],
    hashes: Mapping[str, list[ArchiveRecord]],
    contract: A3BudgetContract,
) -> dict[str, object]:
    replay = _require_hash_record(
        records,
        hashes,
        "input/actor_decisions.jsonl",
        contract.replay_sha256,
        "actor replay",
    )
    _read_actor_rows(replay.path, contract)

    candidates: list[tuple[ArchiveRecord, dict[str, object]]] = []
    for relative, record in records.items():
        if Path(relative).name != "result.json":
            continue
        try:
            value = _load_json(record.path, "rollout result candidate")
        except A3BudgetReceiptError:
            continue
        actor = value.get("actor_decisions")
        if (
            value.get("schema_name") == ROLLOUT_SCHEMA
            and value.get("mode") == "train"
            and isinstance(actor, dict)
            and actor.get("sha256") == contract.replay_sha256
        ):
            candidates.append((record, value))
    if len(candidates) != 1:
        raise A3BudgetReceiptError("final archive lacks one unambiguous train rollout result")
    result_record, result = candidates[0]
    if (
        result.get("schema_version") != 1
        or result.get("status") != "completed"
        or result.get("deterministic") is not False
        or result.get("upstream_split") != "train"
        or result.get("experiment_split_label") != "train_update"
        or result.get("validation_split_reads") != 0
        or result.get("episodes_requested") != contract.rollout_episode_count
    ):
        raise A3BudgetReceiptError("train rollout scope differs")
    run_manifest = _mapping(result.get("run_manifest"), "rollout run manifest")
    split_manifest = _mapping(result.get("split_manifest"), "rollout split manifest")
    checkpoint = _mapping(result.get("checkpoint"), "rollout checkpoint")
    actor = _mapping(result.get("actor_decisions"), "rollout actor decisions")
    aggregate = _mapping(result.get("aggregate"), "rollout aggregate")
    episodes = result.get("episodes")
    if (
        run_manifest.get("sha256") != contract.run_manifest_sha256
        or run_manifest.get("run_id") != contract.run_id
        or split_manifest.get("sha256") != contract.split_manifest_sha256
        or checkpoint.get("policy_version") != 0
        or actor.get("sha256") != contract.replay_sha256
        or actor.get("row_count") != contract.unique_actor_decisions
        or aggregate.get("actor_decision_count") != contract.unique_actor_decisions
        or not isinstance(episodes, list)
        or len(episodes) != contract.rollout_episode_count
    ):
        raise A3BudgetReceiptError("train rollout budget/provenance differs")
    episode_orders: set[int] = set()
    episode_decisions = 0
    for episode in episodes:
        row = _mapping(episode, "rollout episode")
        episode_orders.add(_integer(row.get("episode_order"), "episode order"))
        episode_decisions += _integer(
            row.get("actor_decision_count"), "episode actor decisions"
        )
    if episode_orders != set(range(contract.rollout_episode_count)):
        raise A3BudgetReceiptError("rollout episode order differs")
    if episode_decisions != contract.unique_actor_decisions:
        raise A3BudgetReceiptError("rollout episode decision sum differs")

    parent = str(Path(result_record.relative_source).parent).replace("\\", "/")
    completion = _required_record(
        records, f"{parent}/COMPLETED.json", "rollout completion"
    )
    inventory_record = _required_record(
        records, f"{parent}/artifact_inventory.json", "rollout inventory"
    )
    completion_value = _load_json(completion.path, "rollout completion")
    if (
        completion_value.get("schema_name") != "lightnav_a3_completion"
        or completion_value.get("schema_version") != 1
        or completion_value.get("status") != "completed"
        or completion_value.get("result_sha256") != result_record.sha256
        or completion_value.get("inventory_sha256") != inventory_record.sha256
    ):
        raise A3BudgetReceiptError("rollout completion differs")
    inventory = _load_json(inventory_record.path, "rollout inventory")
    if parent == PRODUCTION_ROLLOUT_PARENT:
        coverage = _verify_production_rollout_inventory(
            inventory,
            records,
            parent,
            inventory_record,
            replay,
            result_record,
        )
    else:
        _verify_inventory(
            inventory,
            records,
            parent,
            "rollout inventory",
            canonical_actor_replay=replay,
        )
        coverage = _legacy_rollout_inventory_coverage(
            parent,
            inventory_record,
            _resolve_rollout_actor_record(records, parent, replay),
            result_record,
        )
    sibling_replay = _resolve_rollout_actor_record(records, parent, replay)
    if sibling_replay.sha256 != contract.replay_sha256:
        raise A3BudgetReceiptError("rollout replay readback hash differs")
    return coverage


def _verify_training(
    records: Mapping[str, ArchiveRecord],
    hashes: Mapping[str, list[ArchiveRecord]],
    contract: A3BudgetContract,
) -> None:
    result_record = _required_record(
        records, "training/training_result.json", "training result"
    )
    inventory_record = _required_record(
        records, "training/artifact_inventory.json", "training inventory"
    )
    completion_record = _required_record(
        records, "training/COMPLETED.json", "training completion"
    )
    checkpoint_record = _require_hash_record(
        records,
        hashes,
        "training/checkpoints/latest/lightnav_a3_policy.npz",
        contract.checkpoint_sha256,
        "A3 checkpoint",
    )
    if checkpoint_record.size_bytes != contract.checkpoint_size_bytes:
        raise A3BudgetReceiptError("A3 checkpoint size differs")

    completion = _load_json(completion_record.path, "training completion")
    if (
        completion.get("schema_name") != "lightnav_a3_training_completion"
        or completion.get("schema_version") != 1
        or completion.get("status") != "completed"
        or completion.get("training_result_sha256") != result_record.sha256
        or completion.get("artifact_inventory_sha256") != inventory_record.sha256
    ):
        raise A3BudgetReceiptError("training completion differs")
    inventory = _load_json(inventory_record.path, "training inventory")
    _verify_inventory(inventory, records, "training", "training inventory")
    expected_inventory = {
        relative.removeprefix("training/")
        for relative in records
        if relative.startswith("training/")
        and relative
        not in {"training/COMPLETED.json", "training/artifact_inventory.json"}
    }
    if set(inventory) != expected_inventory:
        raise A3BudgetReceiptError("training inventory file set differs")

    result = _load_json(result_record.path, "training result")
    if (
        result.get("schema_name") != TRAINING_SCHEMA
        or result.get("schema_version") != 1
        or result.get("status") != "completed"
        or result.get("rlinf_backend") != "FSDPModelManager"
        or result.get("fsdp_wrapper") != "FullyShardedDataParallel"
        or result.get("sharding_strategy") != "full_shard"
        or result.get("world_size") != contract.world_size
        or result.get("unique_gpu_uuid_count") != contract.world_size
        or result.get("epochs") != contract.optimizer_epochs
        or result.get("batch_size_per_rank") != contract.batch_size_per_rank
        or result.get("optimizer_steps_per_rank")
        != contract.optimizer_steps_per_rank
    ):
        raise A3BudgetReceiptError("training result contract differs")
    rollout = _mapping(result.get("rollout"), "training rollout")
    run_manifest = _mapping(result.get("run_manifest"), "training run manifest")
    checkpoint = _mapping(result.get("final_checkpoint"), "training checkpoint")
    checkpoint_metadata = _mapping(
        checkpoint.get("metadata"), "training checkpoint metadata"
    )
    padding_rows = (-contract.unique_actor_decisions) % contract.world_size
    padded_rows = contract.unique_actor_decisions + padding_rows
    sampler_rows = padded_rows // contract.world_size
    expected_steps = (
        math.ceil(sampler_rows / contract.batch_size_per_rank)
        * contract.optimizer_epochs
    )
    if expected_steps != contract.optimizer_steps_per_rank:
        raise A3BudgetReceiptError("configured optimizer-step budget is inconsistent")
    if (
        rollout.get("sha256") != contract.replay_sha256
        or rollout.get("actor_decision_count") != contract.unique_actor_decisions
        or rollout.get("padding_row_count") != padding_rows
        or run_manifest.get("sha256") != contract.run_manifest_sha256
        or run_manifest.get("run_id") != contract.run_id
        or checkpoint.get("sha256") != contract.checkpoint_sha256
        or checkpoint.get("size_bytes") != contract.checkpoint_size_bytes
        or checkpoint_metadata.get("policy_version") != 1
    ):
        raise A3BudgetReceiptError("training provenance/budget differs")

    raw_rank_receipts = result.get("rank_receipts")
    if not isinstance(raw_rank_receipts, list) or len(raw_rank_receipts) != contract.world_size:
        raise A3BudgetReceiptError("training rank receipt count differs")
    ranks: set[int] = set()
    local_ranks: set[int] = set()
    gpu_uuids: set[str] = set()
    expected_collective = contract.world_size * (contract.world_size - 1) / 2
    for raw_rank in raw_rank_receipts:
        rank_receipt = _mapping(raw_rank, "training rank receipt")
        rank = _integer(rank_receipt.get("rank"), "training rank")
        local_rank = _integer(rank_receipt.get("local_rank"), "training local rank")
        gpu_uuid = rank_receipt.get("gpu_uuid")
        gpu_name = rank_receipt.get("gpu_name")
        if (
            rank_receipt.get("schema_name") != "lightnav_a3_rank_receipt"
            or rank_receipt.get("schema_version") != 1
            or rank_receipt.get("world_size") != contract.world_size
            or _finite(rank_receipt.get("collective_sum"), "collective sum")
            != expected_collective
            or _finite(
                rank_receipt.get("expected_collective_sum"),
                "expected collective sum",
            )
            != expected_collective
            or rank_receipt.get("gradient_steps")
            != contract.optimizer_steps_per_rank
            or rank_receipt.get("dataset_rows")
            != contract.unique_actor_decisions
            or rank_receipt.get("padding_rows") != padding_rows
            or rank_receipt.get("sampler_rows") != sampler_rows
            or not isinstance(gpu_uuid, str)
            or not gpu_uuid.startswith("GPU-")
            or not isinstance(gpu_name, str)
            or A800_PATTERN.match(gpu_name) is None
        ):
            raise A3BudgetReceiptError("training rank evidence differs")
        rank_file = _required_record(
            records, f"training/ranks/rank_{rank:03d}.json", "rank receipt file"
        )
        if _load_json(rank_file.path, "rank receipt file") != rank_receipt:
            raise A3BudgetReceiptError("embedded and archived rank receipts differ")
        ranks.add(rank)
        local_ranks.add(local_rank)
        gpu_uuids.add(gpu_uuid)
    expected_ranks = set(range(contract.world_size))
    if ranks != expected_ranks or local_ranks != expected_ranks:
        raise A3BudgetReceiptError("rank/local-rank coverage differs")
    if len(gpu_uuids) != contract.world_size:
        raise A3BudgetReceiptError("rank receipts do not prove distinct physical GPUs")


def build_receipt(
    final_manifest: Path,
    completion: Path,
    obs_root: Path,
    contract: A3BudgetContract = PRODUCTION_CONTRACT,
    *,
    coverage_out: dict[str, object] | None = None,
) -> dict[str, object]:
    """Read and verify all evidence, returning a final-report-compatible receipt."""

    _, records, hashes = _read_archive(
        final_manifest, completion, obs_root, contract
    )
    _verify_run_and_split(records, hashes, contract)
    rollout_coverage = _verify_rollout(records, hashes, contract)
    _verify_training(records, hashes, contract)
    if coverage_out is not None:
        if coverage_out:
            raise A3BudgetReceiptError("coverage_out must be empty")
        coverage_out.update(rollout_coverage)
    presentations = contract.unique_actor_decisions * contract.optimizer_epochs
    return {
        "schema_name": RECEIPT_SCHEMA,
        "schema_version": 1,
        "status": "verified",
        "run_id": contract.run_id,
        "scope": {
            "upstream_split_read": "train",
            "validation_split_reads": 0,
            "train_only": True,
        },
        "provenance": {
            "run_manifest_sha256": contract.run_manifest_sha256,
            "split_manifest_sha256": contract.split_manifest_sha256,
            "final_manifest_sha256": contract.final_manifest_sha256,
            "replay_sha256": contract.replay_sha256,
            "checkpoint_sha256": contract.checkpoint_sha256,
        },
        "budget": {
            "rollout_episode_count": contract.rollout_episode_count,
            "unique_actor_decisions": contract.unique_actor_decisions,
            "actor_eligibility": {
                "actor_eligible_required": True,
                "policy_version": 0,
                "native_stop_excluded": True,
                "external_forced_stop_excluded": True,
            },
            "optimizer_epochs": contract.optimizer_epochs,
            "optimizer_presentations": presentations,
            "padding_excluded": True,
        },
    }


def write_receipt_create_only(
    output: Path, receipt: Mapping[str, object], obs_root: Path
) -> str:
    raw_output = Path(output).expanduser().absolute()
    if raw_output.name != RECEIPT_FILENAME:
        raise A3BudgetReceiptError(f"output must be named {RECEIPT_FILENAME}")
    parent = raw_output.parent.resolve(strict=True)
    if not parent.is_dir():
        raise A3BudgetReceiptError("output parent is not a directory")
    obs = Path(obs_root).expanduser().resolve(strict=True)
    try:
        parent.relative_to(obs)
    except ValueError:
        pass
    else:
        raise A3BudgetReceiptError("receipt output must be outside the OBS root")
    if os.path.lexists(raw_output):
        raise A3BudgetReceiptError("receipt output already exists")
    payload = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(raw_output, flags, 0o444)
    except FileExistsError as error:
        raise A3BudgetReceiptError("receipt output already exists") from error
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            raw_output.unlink()
        except OSError:
            pass
        raise
    return hashlib.sha256(payload).hexdigest()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--obs-root", type=Path, default=Path("/obs/navigation-example")
    )
    result.add_argument("--final-manifest", type=Path)
    result.add_argument("--completion", type=Path)
    result.add_argument("--output", required=True, type=Path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    final_manifest = args.final_manifest or (
        args.obs_root
        / "manifests"
        / "lightnav-a3"
        / f"{PRODUCTION_CONTRACT.run_id}.final.json"
    )
    completion = args.completion or (
        args.obs_root
        / "results"
        / "lightnav-a3"
        / PRODUCTION_CONTRACT.run_id
        / "COMPLETE"
    )
    coverage: dict[str, object] = {}
    receipt = build_receipt(
        final_manifest,
        completion,
        args.obs_root,
        PRODUCTION_CONTRACT,
        coverage_out=coverage,
    )
    receipt_hash = write_receipt_create_only(args.output, receipt, args.obs_root)
    print(
        json.dumps(
            {
                "status": "verified",
                "path": str(Path(args.output).expanduser().absolute()),
                "sha256": receipt_hash,
                "coverage": coverage,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
