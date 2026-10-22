"""Small atomic envelope around policy-owned checkpoint codecs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Mapping, NoReturn

from .interfaces import CheckpointAdapter


MANIFEST_NAME = "stack_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class CheckpointReceipt:
    directory: Path
    metadata: Mapping[str, object]
    files_sha256: Mapping[str, str]
    compatibility: Mapping[str, object]


def _invalid_json_field(field: str, detail: str) -> NoReturn:
    raise TypeError(f"checkpoint {field} must be JSON-serializable: {detail}")


def _canonical_json(value: object, *, field: str) -> object:
    """Validate and normalize a checkpoint manifest value to JSON types."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            _invalid_json_field(field, "non-finite floats are not supported")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                _invalid_json_field(field, "mapping keys must be strings")
            normalized[key] = _canonical_json(item, field=field)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_canonical_json(item, field=field) for item in value]
    _invalid_json_field(field, f"unsupported value type {type(value).__name__}")


def _canonical_mapping(value: Mapping[str, object], *, field: str) -> dict[str, object]:
    normalized = _canonical_json(value, field=field)
    if not isinstance(normalized, dict):  # Kept explicit for type and contract clarity.
        _invalid_json_field(field, "top-level value must be a mapping")
    return normalized


def save_checkpoint_atomic(adapter: CheckpointAdapter, directory: Path) -> CheckpointReceipt:
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=directory.name + ".staging-", dir=str(directory.parent))
    )
    try:
        declared = dict(adapter.save_checkpoint(staging))
        payload_files = sorted(
            path for path in staging.rglob("*") if path.is_file() and path.name != MANIFEST_NAME
        )
        actual = {path.relative_to(staging).as_posix(): _sha256(path) for path in payload_files}
        if declared != actual:
            raise RuntimeError("checkpoint codec hashes differ from files written")
        metadata = _canonical_mapping(adapter.checkpoint_metadata(), field="metadata")
        compatibility = _canonical_mapping(
            adapter.checkpoint_compatibility(), field="compatibility"
        )
        manifest = {
            "schema_version": 2,
            "metadata": metadata,
            "compatibility": compatibility,
            "files_sha256": actual,
        }
        (staging / MANIFEST_NAME).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(staging, directory)
        return CheckpointReceipt(directory, metadata, actual, compatibility)
    except BaseException:
        # Leave a failed staging directory for diagnosis; never replace or delete a target.
        raise


def load_checkpoint_verified(adapter: CheckpointAdapter, directory: Path) -> CheckpointReceipt:
    directory = Path(directory).resolve(strict=True)
    manifest = json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2:
        raise ValueError("unsupported checkpoint manifest schema")
    expected = manifest.get("files_sha256")
    metadata = manifest.get("metadata")
    compatibility = manifest.get("compatibility")
    if (
        not isinstance(expected, dict)
        or not isinstance(metadata, dict)
        or not isinstance(compatibility, dict)
    ):
        raise ValueError("checkpoint manifest fields are invalid")
    actual = {
        path.relative_to(directory).as_posix(): _sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != MANIFEST_NAME
    }
    if actual != expected:
        raise RuntimeError("checkpoint payload hash verification failed")
    current_compatibility = _canonical_mapping(
        adapter.checkpoint_compatibility(), field="compatibility"
    )
    if current_compatibility != compatibility:
        raise RuntimeError("checkpoint compatibility differs from the target adapter")
    prospective_metadata = _canonical_mapping(
        adapter.inspect_checkpoint_metadata(directory), field="metadata"
    )
    if prospective_metadata != metadata:
        raise RuntimeError("checkpoint payload metadata differs from the manifest")
    adapter.load_checkpoint(directory)
    restored_metadata = _canonical_mapping(adapter.checkpoint_metadata(), field="metadata")
    if restored_metadata != metadata:
        raise RuntimeError("restored checkpoint metadata differs from the manifest")
    return CheckpointReceipt(directory, metadata, actual, compatibility)
