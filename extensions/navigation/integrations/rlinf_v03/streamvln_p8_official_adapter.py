"""Live adapter for the pinned original StreamVLN evaluator.

The adapter is deliberately separate from the model-free KEEP state machine.
It owns the real ``output_ids``, KV cache, and upstream model-internal
``inputs_embeds`` cache while the state machine owns prompt cadence, action
queue semantics, and physical-step timing.  Heavy dependencies are imported
only by :meth:`from_released_checkpoint`; importing this module is safe in the
local model-free test environment.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import sys
from typing import Callable, Mapping

if __package__:
    from .streamvln_p8_official_frame import (
        OFFICIAL_DEPTH_MAX_M,
        OFFICIAL_DEPTH_MIN_M,
        OFFICIAL_PROCESSOR_HEIGHT,
        OFFICIAL_PROCESSOR_WIDTH,
        OFFICIAL_RGB_HEIGHT,
        OFFICIAL_RGB_WIDTH,
        OfficialFrame,
    )
    from .streamvln_p8_official_keep import (
        ATTENTION_IMPLEMENTATION,
        CHAT_TEMPLATE,
        DECODE_KWARGS,
        EMPTY_ASSISTANT_CONTENT,
        FRAME_PREPROCESS_CONTRACT,
        GENERATION_KWARGS,
        IMAGE_TOKEN,
        IMAGE_TOKEN_INDEX,
        MEMORY_TOKEN,
        MEMORY_TOKEN_INDEX,
        MODEL_DTYPE,
        MODEL_INTERNAL_STATE_CONTRACT,
        MODEL_MAX_LENGTH,
        NUM_FRAMES,
        NUM_HISTORY,
        PINNED_TORCH_VERSION,
        PINNED_TRANSFORMERS_VERSION,
        PINNED_ARTIFACT_IDS,
        PINNED_EVALUATOR_SHA256,
        PINNED_EVALUATOR_SOURCE_RELATIVE_PATH,
        PINNED_MODEL_CLASS_MODULE,
        PINNED_MODEL_CODE_SHA256,
        PINNED_MODEL_SOURCE_RELATIVE_PATH,
        PINNED_RELEASE_MANIFEST_SHA256,
        PINNED_SOURCE_IMPORT_ROOT_RELATIVE_PATH,
        PINNED_SOURCE_ROOT_KIND,
        PINNED_STREAMVLN_COMMIT,
        RELEASE_MANIFEST_SCHEMA_NAME,
        RELEASE_MANIFEST_SCHEMA_VERSION,
        RENDERED_MESSAGE_ROLES,
        STATE_AUDIT_SCOPE,
        SYSTEM_MESSAGE,
        TEST_ONLY_CONTRACT_VALIDATION_LEVEL,
        TOKENIZER_PREPROCESS_CONTRACT,
        OfficialGenerationRequest,
        official_runtime_receipt_template,
        validate_official_runtime_receipt,
    )
else:  # Explicit direct-file mode for local contract tests only.
    from streamvln_p8_official_frame import (
        OFFICIAL_DEPTH_MAX_M,
        OFFICIAL_DEPTH_MIN_M,
        OFFICIAL_PROCESSOR_HEIGHT,
        OFFICIAL_PROCESSOR_WIDTH,
        OFFICIAL_RGB_HEIGHT,
        OFFICIAL_RGB_WIDTH,
        OfficialFrame,
    )
    from streamvln_p8_official_keep import (
        ATTENTION_IMPLEMENTATION,
        CHAT_TEMPLATE,
        DECODE_KWARGS,
        EMPTY_ASSISTANT_CONTENT,
        FRAME_PREPROCESS_CONTRACT,
        GENERATION_KWARGS,
        IMAGE_TOKEN,
        IMAGE_TOKEN_INDEX,
        MEMORY_TOKEN,
        MEMORY_TOKEN_INDEX,
        MODEL_DTYPE,
        MODEL_INTERNAL_STATE_CONTRACT,
        MODEL_MAX_LENGTH,
        NUM_FRAMES,
        NUM_HISTORY,
        PINNED_TORCH_VERSION,
        PINNED_TRANSFORMERS_VERSION,
        PINNED_ARTIFACT_IDS,
        PINNED_EVALUATOR_SHA256,
        PINNED_EVALUATOR_SOURCE_RELATIVE_PATH,
        PINNED_MODEL_CLASS_MODULE,
        PINNED_MODEL_CODE_SHA256,
        PINNED_MODEL_SOURCE_RELATIVE_PATH,
        PINNED_RELEASE_MANIFEST_SHA256,
        PINNED_SOURCE_IMPORT_ROOT_RELATIVE_PATH,
        PINNED_SOURCE_ROOT_KIND,
        PINNED_STREAMVLN_COMMIT,
        RELEASE_MANIFEST_SCHEMA_NAME,
        RELEASE_MANIFEST_SCHEMA_VERSION,
        RENDERED_MESSAGE_ROLES,
        STATE_AUDIT_SCOPE,
        SYSTEM_MESSAGE,
        TEST_ONLY_CONTRACT_VALIDATION_LEVEL,
        TOKENIZER_PREPROCESS_CONTRACT,
        OfficialGenerationRequest,
        official_runtime_receipt_template,
        validate_official_runtime_receipt,
    )


ADAPTER_SCHEMA_NAME = "streamvln_p8_pinned_generation_adapter"
ADAPTER_SCHEMA_VERSION = 1
ARTIFACT_MANIFEST_SCHEMA_NAME = RELEASE_MANIFEST_SCHEMA_NAME
ARTIFACT_MANIFEST_SCHEMA_VERSION = RELEASE_MANIFEST_SCHEMA_VERSION
FEATURE_V2_CONTEXT_DIM = 3584
FEATURE_V2_DEPTH_SECTORS = 5
FEATURE_V2_DEPTH_VALUES_PER_SECTOR = 3

def _base_version(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError("official runtime did not expose a package version")
    return value.split("+", 1)[0]


def _json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _stat_signature(value) -> tuple:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


_WINDOWS_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_HF_FILE_SYMLINK_ARTIFACTS = frozenset({"tokenizer", "vision_tower"})


def _is_windows_reparse(value) -> bool:
    return bool(
        int(getattr(value, "st_file_attributes", 0)) & _WINDOWS_REPARSE_POINT
    )


def _ordinary_directory_root(root_value: str | Path, *, name: str) -> Path:
    requested = Path(root_value).expanduser().absolute()
    try:
        before = os.lstat(requested)
    except OSError as exc:
        raise ValueError(f"official {name} root is not readable") from exc
    if stat.S_ISLNK(before.st_mode) or _is_windows_reparse(before):
        raise RuntimeError(f"official {name} root forbids symlink or reparse points")
    if not stat.S_ISDIR(before.st_mode):
        raise ValueError(f"official {name} root must be a directory")
    resolved = requested.resolve(strict=True)
    after = os.lstat(requested)
    if _stat_signature(before) != _stat_signature(after):
        raise RuntimeError(f"official {name} root changed while resolving")
    return resolved


def _ordinary_directory_stat(path: Path, *, name: str):
    value = os.lstat(path)
    if stat.S_ISLNK(value.st_mode) or _is_windows_reparse(value):
        raise RuntimeError(f"official {name} inventory forbids directory symlinks or reparse points")
    if not stat.S_ISDIR(value.st_mode):
        raise RuntimeError(f"official {name} inventory entry is not a directory")
    return value


def _hash_opened_regular_file(path: Path) -> tuple[int, str, tuple]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or _is_windows_reparse(before):
            raise RuntimeError("release manifest entry is not an ordinary regular file")
        for payload in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(payload)
        after = os.fstat(stream.fileno())
    if _stat_signature(before) != _stat_signature(after):
        raise RuntimeError("release manifest file changed while hashing")
    return int(after.st_size), digest.hexdigest(), _stat_signature(after)


def _canonical_link_target(path: Path) -> str:
    target = os.readlink(path)
    target_text = str(target).replace("\\", "/")
    target_path = PurePosixPath(target_text)
    if (
        not target_text
        or target_path.is_absolute()
        or target_path.as_posix() != target_text
    ):
        raise RuntimeError("release manifest symlink target is not canonical-relative")
    return target_text


def _file_manifest_entry(
    path: Path,
    *,
    root: Path,
    allow_file_symlinks: bool,
    trusted_blob_root: Path | None = None,
) -> dict:
    try:
        relative_path = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise RuntimeError("release manifest path escaped its inventory root") from exc
    before = os.lstat(path)
    if stat.S_ISLNK(before.st_mode):
        if not allow_file_symlinks or trusted_blob_root is None:
            raise RuntimeError("release manifest inventory forbids this symlink file")
        link_target = _canonical_link_target(path)
        blob_root_before = _ordinary_directory_stat(
            trusted_blob_root, name="Hugging Face blob"
        )
        resolved = path.resolve(strict=True)
        if resolved.parent != trusted_blob_root:
            raise RuntimeError(
                "release manifest symlink escaped its trusted Hugging Face blob root"
            )
        resolved_before = os.lstat(resolved)
        if (
            stat.S_ISLNK(resolved_before.st_mode)
            or _is_windows_reparse(resolved_before)
            or not stat.S_ISREG(resolved_before.st_mode)
        ):
            raise RuntimeError(
                "release manifest symlink target is not an ordinary blob file"
            )
        size, digest, opened_signature = _hash_opened_regular_file(resolved)
        resolved_after = os.lstat(resolved)
        after = os.lstat(path)
        blob_root_after = _ordinary_directory_stat(
            trusted_blob_root, name="Hugging Face blob"
        )
        if (
            _stat_signature(before) != _stat_signature(after)
            or _canonical_link_target(path) != link_target
            or _stat_signature(resolved_before) != opened_signature
            or _stat_signature(resolved_after) != opened_signature
            or _stat_signature(blob_root_before) != _stat_signature(blob_root_after)
        ):
            raise RuntimeError("release manifest symlink changed while hashing")
        entry_type = "symlink"
    else:
        if _is_windows_reparse(before):
            raise RuntimeError("release manifest inventory forbids reparse-point files")
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError("release manifest entry is not a regular file")
        size, digest, opened_signature = _hash_opened_regular_file(path)
        after = os.lstat(path)
        if (
            _stat_signature(before) != opened_signature
            or _stat_signature(after) != opened_signature
        ):
            raise RuntimeError("release manifest file changed while hashing")
        link_target = None
        entry_type = "regular_file"
    return {
        "relative_path": relative_path,
        "entry_type": entry_type,
        "link_target": link_target,
        "size": size,
        "sha256": digest,
    }


def _inventory_tree(
    root_value: str | Path,
    *,
    name: str,
    python_only: bool,
    allow_file_symlinks: bool,
    trusted_blob_root: Path | None = None,
) -> list[dict]:
    root = _ordinary_directory_root(root_value, name=name)
    if allow_file_symlinks != (trusted_blob_root is not None):
        raise ValueError("release manifest symlink policy is incomplete")
    if trusted_blob_root is not None:
        trusted_blob_root = _ordinary_directory_root(
            trusted_blob_root, name=f"{name} Hugging Face blob"
        )
    root_before = _ordinary_directory_stat(root, name=name)
    rows = []

    def fail_walk(error):
        raise RuntimeError(f"official {name} inventory traversal failed") from error

    for current_value, directory_names, file_names in os.walk(
        root,
        topdown=True,
        onerror=fail_walk,
        followlinks=False,
    ):
        current = Path(current_value)
        _ordinary_directory_stat(current, name=name)
        if python_only:
            directory_names[:] = [
                entry for entry in directory_names if entry != "__pycache__"
            ]
        directory_names.sort()
        file_names.sort()
        for directory_name in directory_names:
            _ordinary_directory_stat(current / directory_name, name=name)
        for file_name in file_names:
            path = current / file_name
            entry_stat = os.lstat(path)
            if python_only and (
                stat.S_ISLNK(entry_stat.st_mode) or _is_windows_reparse(entry_stat)
            ):
                raise RuntimeError(
                    f"official {name} inventory forbids symlink or reparse-point files"
                )
            if python_only and path.suffix != ".py":
                continue
            rows.append(
                _file_manifest_entry(
                    path,
                    root=root,
                    allow_file_symlinks=allow_file_symlinks,
                    trusted_blob_root=trusted_blob_root,
                )
            )
    root_after = _ordinary_directory_stat(root, name=name)
    if _stat_signature(root_before) != _stat_signature(root_after):
        raise RuntimeError(f"official {name} root changed while hashing")
    rows.sort(key=lambda item: item["relative_path"])
    if not rows:
        raise RuntimeError(f"official {name} inventory contains no files")
    return rows


def _artifact_roots(
    *,
    model_path: str | Path,
    tokenizer_path: str | Path,
    vision_tower_path: str | Path,
) -> dict[str, Path]:
    requested = {
        "checkpoint": model_path,
        "tokenizer": tokenizer_path,
        "vision_tower": vision_tower_path,
    }
    roots = {
        name: _ordinary_directory_root(value, name=f"{name} artifact")
        for name, value in requested.items()
    }
    for name, root in roots.items():
        if root.name != PINNED_ARTIFACT_IDS[name]:
            raise RuntimeError(f"official {name} artifact identity drifted")
    return roots


def _trusted_hf_blob_root(snapshot_root: Path, *, name: str) -> Path:
    if name not in _HF_FILE_SYMLINK_ARTIFACTS or snapshot_root.parent.name != "snapshots":
        raise RuntimeError(
            f"official {name} file symlinks require a Hugging Face snapshot root"
        )
    _ordinary_directory_root(snapshot_root.parent, name=f"{name} snapshots")
    repository_root = _ordinary_directory_root(
        snapshot_root.parent.parent, name=f"{name} Hugging Face repository"
    )
    return _ordinary_directory_root(
        repository_root / "blobs", name=f"{name} Hugging Face blob"
    )


def build_full_content_artifact_manifest(
    *,
    source_root: str | Path,
    model_path: str | Path,
    tokenizer_path: str | Path,
    vision_tower_path: str | Path,
) -> dict:
    """Build a candidate diagnostic manifest; this never establishes release identity."""

    source_root_path = _ordinary_directory_root(
        source_root, name="StreamVLN source"
    )
    roots = _artifact_roots(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        vision_tower_path=vision_tower_path,
    )
    trusted_blob_roots = {
        name: _trusted_hf_blob_root(roots[name], name=name)
        for name in _HF_FILE_SYMLINK_ARTIFACTS
    }
    return {
        "schema_name": ARTIFACT_MANIFEST_SCHEMA_NAME,
        "schema_version": ARTIFACT_MANIFEST_SCHEMA_VERSION,
        "verification_scope": "candidate_full_content",
        "source": {
            "source_id": PINNED_STREAMVLN_COMMIT,
            "source_root_kind": PINNED_SOURCE_ROOT_KIND,
            "import_root_relative_path": PINNED_SOURCE_IMPORT_ROOT_RELATIVE_PATH,
            "python_files": _inventory_tree(
                source_root_path,
                name="source Python",
                python_only=True,
                allow_file_symlinks=False,
            ),
        },
        "artifacts": {
            name: {
                "artifact_id": PINNED_ARTIFACT_IDS[name],
                "files": _inventory_tree(
                    root,
                    name=name,
                    python_only=False,
                    allow_file_symlinks=name in _HF_FILE_SYMLINK_ARTIFACTS,
                    trusted_blob_root=trusted_blob_roots.get(name),
                ),
            }
            for name, root in roots.items()
        },
    }

def _normalize_file_manifest(
    files,
    *,
    name: str,
    allow_file_symlinks: bool,
) -> list[dict]:
    if not isinstance(files, list) or not files:
        raise ValueError(f"official {name} file manifest is empty")
    normalized_files = []
    seen = set()
    expected_keys = {
        "relative_path",
        "entry_type",
        "link_target",
        "size",
        "sha256",
    }
    for entry in files:
        if not isinstance(entry, Mapping) or set(entry) != expected_keys:
            raise ValueError(f"official {name} file schema drifted")
        relative_path = entry.get("relative_path")
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError(f"official {name} path is invalid")
        pure_path = PurePosixPath(relative_path)
        if (
            pure_path.is_absolute()
            or any(part in {"", ".", ".."} for part in pure_path.parts)
            or pure_path.as_posix() != relative_path
            or "\\" in relative_path
            or relative_path in seen
        ):
            raise ValueError(f"official {name} path is not canonical")
        seen.add(relative_path)

        entry_type = entry.get("entry_type")
        link_target = entry.get("link_target")
        if entry_type == "regular_file":
            if link_target is not None:
                raise ValueError(f"official {name} regular-file link target is invalid")
        elif entry_type == "symlink" and allow_file_symlinks:
            if not isinstance(link_target, str) or not link_target:
                raise ValueError(f"official {name} symlink target is invalid")
            pure_target = PurePosixPath(link_target)
            if (
                pure_target.is_absolute()
                or pure_target.as_posix() != link_target
                or "\\" in link_target
            ):
                raise ValueError(
                    f"official {name} symlink target is not canonical-relative"
                )
        else:
            raise ValueError(f"official {name} entry type is invalid")

        size = entry.get("size")
        digest = entry.get("sha256")
        if type(size) is not int or size < 0:
            raise ValueError(f"official {name} file size is invalid")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"official {name} file SHA-256 is invalid")
        normalized_files.append(
            {
                "relative_path": relative_path,
                "entry_type": entry_type,
                "link_target": link_target,
                "size": size,
                "sha256": digest,
            }
        )
    if normalized_files != sorted(
        normalized_files, key=lambda item: item["relative_path"]
    ):
        raise ValueError(f"official {name} files are not canonical")
    return normalized_files


def _normalize_artifact_manifest(manifest: Mapping) -> dict:
    if not isinstance(manifest, Mapping):
        raise TypeError("official full-content release manifest must be a mapping")
    expected_top = {
        "schema_name",
        "schema_version",
        "verification_scope",
        "source",
        "artifacts",
    }
    if set(manifest) != expected_top:
        raise ValueError("official full-content release manifest schema drifted")
    if (
        manifest.get("schema_name") != ARTIFACT_MANIFEST_SCHEMA_NAME
        or manifest.get("schema_version") != ARTIFACT_MANIFEST_SCHEMA_VERSION
        or manifest.get("verification_scope") != "candidate_full_content"
    ):
        raise ValueError("official full-content release manifest identity drifted")

    source = manifest.get("source")
    if not isinstance(source, Mapping) or set(source) != {
        "source_id",
        "source_root_kind",
        "import_root_relative_path",
        "python_files",
    }:
        raise ValueError("official source Python manifest schema drifted")
    if (
        source.get("source_id") != PINNED_STREAMVLN_COMMIT
        or source.get("source_root_kind") != PINNED_SOURCE_ROOT_KIND
        or source.get("import_root_relative_path")
        != PINNED_SOURCE_IMPORT_ROOT_RELATIVE_PATH
    ):
        raise ValueError("official source identity drifted")
    source_files = _normalize_file_manifest(
        source.get("python_files"),
        name="source Python",
        allow_file_symlinks=False,
    )
    source_paths = {entry["relative_path"] for entry in source_files}
    required_source_paths = {
        PINNED_MODEL_SOURCE_RELATIVE_PATH,
        PINNED_EVALUATOR_SOURCE_RELATIVE_PATH,
    }
    if not required_source_paths.issubset(source_paths):
        raise ValueError("official source Python manifest lacks required files")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(
        PINNED_ARTIFACT_IDS
    ):
        raise ValueError("official full-content artifact set drifted")
    normalized_artifacts = {}
    for name, artifact_id in PINNED_ARTIFACT_IDS.items():
        artifact = artifacts[name]
        if not isinstance(artifact, Mapping) or set(artifact) != {
            "artifact_id",
            "files",
        }:
            raise ValueError(f"official {name} artifact manifest schema drifted")
        if artifact.get("artifact_id") != artifact_id:
            raise ValueError(f"official {name} artifact identity drifted")
        normalized_artifacts[name] = {
            "artifact_id": artifact_id,
            "files": _normalize_file_manifest(
                artifact.get("files"),
                name=name,
                allow_file_symlinks=name in _HF_FILE_SYMLINK_ARTIFACTS,
            ),
        }
    return {
        "schema_name": ARTIFACT_MANIFEST_SCHEMA_NAME,
        "schema_version": ARTIFACT_MANIFEST_SCHEMA_VERSION,
        "verification_scope": "candidate_full_content",
        "source": {
            "source_id": PINNED_STREAMVLN_COMMIT,
            "source_root_kind": PINNED_SOURCE_ROOT_KIND,
            "import_root_relative_path": PINNED_SOURCE_IMPORT_ROOT_RELATIVE_PATH,
            "python_files": source_files,
        },
        "artifacts": normalized_artifacts,
    }


def _manifest_sha256(manifest: Mapping) -> str:
    return _json_sha256(_normalize_artifact_manifest(manifest))


def verify_full_content_artifact_manifest(
    manifest: Mapping,
    *,
    source_root: str | Path,
    model_path: str | Path,
    tokenizer_path: str | Path,
    vision_tower_path: str | Path,
) -> dict:
    """Verify bytes against a candidate manifest without granting release status."""

    expected = _normalize_artifact_manifest(manifest)
    actual = build_full_content_artifact_manifest(
        source_root=source_root,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        vision_tower_path=vision_tower_path,
    )
    if actual != expected:
        raise RuntimeError("candidate full-content release verification failed")
    return {
        "verification_scope": "candidate_full_content",
        "manifest_sha256": _json_sha256(expected),
        "release_manifest": copy.deepcopy(expected),
        "source": {
            "source_id": expected["source"]["source_id"],
            "source_root_kind": expected["source"]["source_root_kind"],
            "source_import_root_relative_path": expected["source"][
                "import_root_relative_path"
            ],
            "python_file_count": len(expected["source"]["python_files"]),
            "python_content_manifest_sha256": _json_sha256(
                expected["source"]["python_files"]
            ),
        },
        "artifacts": {
            name: {
                "artifact_id": artifact["artifact_id"],
                "file_count": len(artifact["files"]),
                "content_manifest_sha256": _json_sha256(artifact["files"]),
            }
            for name, artifact in expected["artifacts"].items()
        },
    }


def _verify_source_tree(source_root_value: str | Path) -> tuple[Path, dict]:
    source_root = _ordinary_directory_root(
        source_root_value, name="StreamVLN source"
    )
    source_import_root = _ordinary_directory_root(
        source_root / PINNED_SOURCE_IMPORT_ROOT_RELATIVE_PATH,
        name="StreamVLN import",
    )
    try:
        source_import_root.relative_to(source_root)
    except ValueError as exc:
        raise RuntimeError("official StreamVLN import root escaped source root") from exc
    model_entry = _file_manifest_entry(
        source_root / PINNED_MODEL_SOURCE_RELATIVE_PATH,
        root=source_root,
        allow_file_symlinks=False,
    )
    evaluator_entry = _file_manifest_entry(
        source_root / PINNED_EVALUATOR_SOURCE_RELATIVE_PATH,
        root=source_root,
        allow_file_symlinks=False,
    )
    if model_entry["sha256"] != PINNED_MODEL_CODE_SHA256:
        raise RuntimeError("official imported model source SHA-256 drifted")
    if evaluator_entry["sha256"] != PINNED_EVALUATOR_SHA256:
        raise RuntimeError("official evaluator source SHA-256 drifted")
    return source_root, {
        "source_root_kind": PINNED_SOURCE_ROOT_KIND,
        "source_import_root_relative_path": (
            PINNED_SOURCE_IMPORT_ROOT_RELATIVE_PATH
        ),
        "model_source_relative_path": model_entry["relative_path"],
        "model_source_sha256": model_entry["sha256"],
        "evaluator_source_relative_path": evaluator_entry["relative_path"],
        "evaluator_sha256": evaluator_entry["sha256"],
    }


def _install_source_import_root(source_root: Path) -> Path:
    source_root = _ordinary_directory_root(
        source_root, name="StreamVLN source"
    )
    source_import_root = _ordinary_directory_root(
        source_root / PINNED_SOURCE_IMPORT_ROOT_RELATIVE_PATH,
        name="StreamVLN import",
    )
    try:
        source_import_root.relative_to(source_root)
    except ValueError as exc:
        raise RuntimeError("official StreamVLN import root escaped source root") from exc
    import_root_text = str(source_import_root)
    source_root_text = str(source_root)
    pinned_paths = {import_root_text, source_root_text}
    sys.path[:] = [entry for entry in sys.path if str(entry) not in pinned_paths]
    # Upstream imports ``model``/``utils`` from ``source/streamvln`` and
    # ``llava`` from the repository root. Keep that exact order explicit
    # instead of depending on the caller's working directory or PYTHONPATH.
    sys.path[0:0] = [import_root_text, source_root_text]
    importlib.invalidate_caches()
    return source_import_root


def _verify_imported_model_source(model_module, model_class, source_root: Path) -> dict:
    module_file = getattr(model_module, "__file__", None)
    if not isinstance(module_file, str) or not module_file:
        raise RuntimeError("official imported model module has no source file")
    expected = (source_root / PINNED_MODEL_SOURCE_RELATIVE_PATH).resolve(strict=True)
    actual_path = Path(module_file).expanduser().absolute()
    actual_entry = _file_manifest_entry(
        actual_path,
        root=source_root,
        allow_file_symlinks=False,
    )
    actual = actual_path.resolve(strict=True)
    if (
        actual != expected
        or actual_entry["sha256"] != PINNED_MODEL_CODE_SHA256
    ):
        raise RuntimeError("official imported model module source drifted")
    if getattr(model_class, "__module__", None) != PINNED_MODEL_CLASS_MODULE:
        raise RuntimeError("official imported model class module drifted")
    return {
        "imported_model_module_relative_path": actual_entry["relative_path"],
        "imported_model_module_sha256": actual_entry["sha256"],
        "model_class_module": model_class.__module__,
    }

def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_pinned_release_manifest_sha256() -> str:
    anchor = PINNED_RELEASE_MANIFEST_SHA256
    if not _valid_sha256(anchor):
        raise RuntimeError(
            "official release manifest SHA-256 is PENDING; "
            "candidate manifests are diagnostic only"
        )
    return anchor


def _validate_explicit_factory_cuda(device, local_rank: int) -> str:
    if type(local_rank) is not int or local_rank < 0:
        raise ValueError("official factory local_rank must be a nonnegative integer")
    if isinstance(device, str):
        requested = device
        if not requested.startswith("cuda:"):
            raise ValueError("official factory requires explicit cuda:N")
        index_text = requested.removeprefix("cuda:")
        if not index_text.isdigit():
            raise ValueError("official factory requires explicit cuda:N")
        index = int(index_text)
    else:
        device_type = getattr(device, "type", None)
        index = getattr(device, "index", None)
        if device_type != "cuda" or type(index) is not int or index < 0:
            raise ValueError("official factory requires explicit cuda:N")
        requested = str(device)
    if index != local_rank or requested != f"cuda:{local_rank}":
        raise ValueError("official factory device/local_rank mismatch")
    return requested

def _resolve_device(device, torch_module):
    if type(device) is bool:
        raise TypeError("official adapter device cannot be boolean")
    requested = str(device)
    if type(device) is int:
        if device < 0:
            raise ValueError("official adapter CUDA device index must be nonnegative")
        resolved = torch_module.device("cuda", device)
    else:
        resolved = torch_module.device(device)
    if getattr(resolved, "type", None) == "cuda" and getattr(
        resolved, "index", None
    ) is None:
        cuda_module = getattr(torch_module, "cuda", None)
        current_device = getattr(cuda_module, "current_device", None)
        if not callable(current_device):
            raise RuntimeError("official adapter cannot resolve the current CUDA device")
        resolved = torch_module.device("cuda", int(current_device()))
    if getattr(resolved, "type", None) not in {"cpu", "cuda"}:
        raise ValueError("official adapter device type is unsupported")
    return requested, resolved


def _tensor_audit(tensor, torch_module) -> dict:
    """Hash tensor metadata and exact contiguous storage bytes."""

    value = tensor.detach().cpu().contiguous()
    byte_view = value.view(torch_module.uint8)
    payload = byte_view.numpy().tobytes(order="C")
    return {
        "dtype": str(value.dtype),
        "shape": [int(item) for item in value.shape],
        "sha256": hashlib.sha256(payload).hexdigest(),
        "nbytes": len(payload),
    }


def _tensor_shape(tensor, *, name: str) -> tuple[int, ...]:
    shape = getattr(tensor, "shape", None)
    if shape is None:
        raise TypeError(f"official {name} tensor did not expose a shape")
    try:
        result = tuple(int(item) for item in shape)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"official {name} tensor shape is invalid") from exc
    if any(item < 0 for item in result):
        raise ValueError(f"official {name} tensor shape is invalid")
    return result


def _validate_tensor_contract(
    tensor,
    *,
    name: str,
    expected_shape: tuple[int, ...],
    expected_dtype: str,
    expected_device,
) -> None:
    if _tensor_shape(tensor, name=name) != expected_shape:
        raise RuntimeError(f"official {name} tensor shape drifted")
    if str(getattr(tensor, "dtype", None)) != expected_dtype:
        raise RuntimeError(f"official {name} tensor dtype drifted")
    if str(getattr(tensor, "device", None)) != str(expected_device):
        raise RuntimeError(f"official {name} tensor device drifted")


def _validated_tensor_audit(auditor: Callable, tensor, *, name: str) -> dict:
    raw = auditor(tensor)
    if not isinstance(raw, Mapping):
        raise TypeError(f"official {name} tensor auditor returned a non-mapping")
    audit = dict(raw)
    if set(audit) != {"dtype", "shape", "sha256", "nbytes"}:
        raise ValueError(f"official {name} tensor audit schema drifted")
    if not isinstance(audit["dtype"], str) or not audit["dtype"]:
        raise ValueError(f"official {name} tensor audit dtype is invalid")
    if (
        not isinstance(audit["shape"], list)
        or any(type(item) is not int or item < 0 for item in audit["shape"])
    ):
        raise ValueError(f"official {name} tensor audit shape is invalid")
    digest = audit["sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"official {name} tensor audit SHA-256 is invalid")
    if type(audit["nbytes"]) is not int or audit["nbytes"] < 1:
        raise ValueError(f"official {name} tensor audit byte count is invalid")
    return audit


def _cache_structure(value, torch_module):
    if isinstance(value, torch_module.Tensor):
        return {
            "type": "tensor",
            "dtype": str(value.dtype),
            "shape": [int(item) for item in value.shape],
        }
    if isinstance(value, (tuple, list)):
        return [_cache_structure(item, torch_module) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _cache_structure(item, torch_module)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if value is None:
        return None
    if hasattr(value, "key_cache") and hasattr(value, "value_cache"):
        return {
            "type": type(value).__name__,
            "key_cache": _cache_structure(value.key_cache, torch_module),
            "value_cache": _cache_structure(value.value_cache, torch_module),
        }
    return {"type": type(value).__name__}


def _state_identity_tree(value, torch_module):
    """Cheap mutation signature for large KV state without copying its payload."""

    if isinstance(value, torch_module.Tensor):
        try:
            version = int(value._version)
        except RuntimeError:  # Torch inference tensors do not expose a version counter.
            version = None
        try:
            storage_pointer = int(value.untyped_storage().data_ptr())
        except (AttributeError, RuntimeError):
            storage_pointer = None
        return {
            "type": "tensor",
            "identity": id(value),
            "version": version,
            "storage_pointer": storage_pointer,
            "dtype": str(value.dtype),
            "device": str(value.device),
            "shape": [int(item) for item in value.shape],
        }
    if isinstance(value, (tuple, list)):
        return {
            "type": type(value).__name__,
            "identity": id(value),
            "items": [_state_identity_tree(item, torch_module) for item in value],
        }
    if isinstance(value, Mapping):
        return {
            "type": type(value).__name__,
            "identity": id(value),
            "items": {
                str(key): _state_identity_tree(item, torch_module)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            },
        }
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "key_cache") and hasattr(value, "value_cache"):
        return {
            "type": type(value).__name__,
            "identity": id(value),
            "key_cache": _state_identity_tree(value.key_cache, torch_module),
            "value_cache": _state_identity_tree(value.value_cache, torch_module),
        }
    return {"type": type(value).__name__, "identity": id(value)}


def build_official_input_ids(
    *,
    request: OfficialGenerationRequest,
    tokenizer,
    torch_module,
    previous_output_ids=None,
):
    """Mirror ``preprocess_qwen`` from the pinned evaluator exactly."""

    if not isinstance(request, OfficialGenerationRequest):
        raise TypeError("official tokenizer requires OfficialGenerationRequest")
    local_tokenizer = copy.deepcopy(tokenizer)
    local_tokenizer.add_tokens([IMAGE_TOKEN], special_tokens=True)
    local_tokenizer.add_tokens([MEMORY_TOKEN], special_tokens=True)
    image_token_id = local_tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    memory_token_id = local_tokenizer.convert_tokens_to_ids(MEMORY_TOKEN)
    # The pinned code unpacks this pair even though the values are unused.
    additional = tuple(local_tokenizer.additional_special_tokens_ids)
    if len(additional) != 2:
        raise RuntimeError("official tokenizer additional-special-token contract drifted")
    local_tokenizer.chat_template = CHAT_TEMPLATE

    input_ids: list[int] = []
    if request.add_system:
        input_ids.extend(
            local_tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM_MESSAGE}]
            )
        )
    expected_roles = tuple(role for role, _ in request.rendered_messages)
    if expected_roles != RENDERED_MESSAGE_ROLES:
        raise RuntimeError("official rendered message order drifted")
    if request.rendered_messages[-1][1] != EMPTY_ASSISTANT_CONTENT:
        raise RuntimeError("official assistant message must be empty")
    for role, content in request.rendered_messages:
        input_ids.extend(
            local_tokenizer.apply_chat_template(
                [{"role": role, "content": content}]
            )
        )
    for index, token_id in enumerate(input_ids):
        if token_id == image_token_id:
            input_ids[index] = IMAGE_TOKEN_INDEX
        if token_id == memory_token_id:
            input_ids[index] = MEMORY_TOKEN_INDEX
    current = torch_module.tensor([input_ids], dtype=torch_module.long)
    if previous_output_ids is not None:
        current = torch_module.cat(
            [previous_output_ids, current.to(previous_output_ids.device)], dim=1
        )
    return current


@dataclass(frozen=True)
class PreparedOfficialFrames:
    tensors: Mapping[str, object]
    frame_refs: tuple[str, ...]


@dataclass(frozen=True)
class FrozenStreamVLNFeatureSnapshot:
    """Audited frozen clean-e48 feature for one causal physical frame."""

    feature: object
    physical_step: int
    episode_generation: str
    frame_ref: str
    reset_ordinal: int
    generation_ordinal: int
    vision_dim: int
    instruction_dim: int
    vision_audit: Mapping
    instruction_audit: Mapping
    feature_audit: Mapping

    def __post_init__(self) -> None:
        if type(self.physical_step) is not int or self.physical_step < 0:
            raise ValueError("frozen feature physical_step must be nonnegative")
        if not isinstance(self.episode_generation, str) or not self.episode_generation:
            raise ValueError("frozen feature episode generation is invalid")
        if not isinstance(self.frame_ref, str) or not self.frame_ref:
            raise ValueError("frozen feature frame reference is invalid")
        if type(self.reset_ordinal) is not int or self.reset_ordinal < 0:
            raise ValueError("frozen feature requires a reset adapter")
        if type(self.generation_ordinal) is not int or self.generation_ordinal < 0:
            raise ValueError("frozen feature generation ordinal is invalid")
        if (
            type(self.vision_dim) is not int
            or type(self.instruction_dim) is not int
            or self.vision_dim < 1
            or self.vision_dim != self.instruction_dim
        ):
            raise ValueError("frozen feature branch dimensions are invalid")


@dataclass(frozen=True)
class StreamVLNFeatureV2Snapshot:
    """One generation-latched contextual and metric-depth feature pair."""

    contextual_hidden: object
    depth_summary: object
    physical_step: int
    episode_generation: str
    frame_ref: str
    depth_sha256: str
    reset_ordinal: int
    generation_ordinal: int
    prefill_length: int
    past_key_values_ref: str
    inputs_embeds_ref: str
    contextual_hidden_audit: Mapping
    depth_summary_audit: Mapping
    depth_source_audit: Mapping
    state_refs_audit: Mapping

    def __post_init__(self) -> None:
        if _tensor_shape(
            self.contextual_hidden, name="feature-v2 contextual hidden"
        ) != (FEATURE_V2_CONTEXT_DIM,):
            raise ValueError("feature-v2 contextual hidden shape is invalid")
        if _tensor_shape(
            self.depth_summary, name="feature-v2 depth summary"
        ) != (
            FEATURE_V2_DEPTH_SECTORS * FEATURE_V2_DEPTH_VALUES_PER_SECTOR,
        ):
            raise ValueError("feature-v2 depth summary shape is invalid")
        if type(self.physical_step) is not int or self.physical_step < 0:
            raise ValueError("feature-v2 physical_step must be nonnegative")
        if (
            not isinstance(self.episode_generation, str)
            or len(self.episode_generation) != 32
            or any(
                character not in "0123456789abcdef"
                for character in self.episode_generation
            )
        ):
            raise ValueError("feature-v2 episode generation is invalid")
        if (
            not isinstance(self.frame_ref, str)
            or not self.frame_ref.startswith("sha256:")
            or not _valid_sha256(self.frame_ref.removeprefix("sha256:"))
        ):
            raise ValueError("feature-v2 frame reference is invalid")
        if not _valid_sha256(self.depth_sha256):
            raise ValueError("feature-v2 depth SHA-256 is invalid")
        if type(self.reset_ordinal) is not int or self.reset_ordinal < 0:
            raise ValueError("feature-v2 reset ordinal is invalid")
        if type(self.generation_ordinal) is not int or self.generation_ordinal < 0:
            raise ValueError("feature-v2 generation ordinal is invalid")
        if type(self.prefill_length) is not int or self.prefill_length < 1:
            raise ValueError("feature-v2 prefill length is invalid")
        if not _valid_sha256(self.past_key_values_ref):
            raise ValueError("feature-v2 KV reference is invalid")
        if not _valid_sha256(self.inputs_embeds_ref):
            raise ValueError("feature-v2 inputs_embeds reference is invalid")
        for name, value in (
            ("contextual hidden", self.contextual_hidden_audit),
            ("depth summary", self.depth_summary_audit),
            ("depth source", self.depth_source_audit),
            ("state refs", self.state_refs_audit),
        ):
            if not isinstance(value, Mapping):
                raise TypeError(f"feature-v2 {name} audit must be a mapping")


def build_feature_v2_depth_summary(
    frame: OfficialFrame,
    *,
    torch_module,
    tensor_auditor: Callable[[object], Mapping],
) -> tuple[object, dict, dict]:
    """Build five left-to-right metric-depth sector summaries on CPU."""

    if not isinstance(frame, OfficialFrame):
        raise TypeError("feature-v2 depth summary requires an OfficialFrame")
    if not callable(tensor_auditor):
        raise TypeError("feature-v2 depth summary requires a tensor auditor")
    if sys.byteorder != "little":
        raise RuntimeError("feature-v2 f32le depth requires a little-endian host")
    expected_values = OFFICIAL_RGB_HEIGHT * OFFICIAL_RGB_WIDTH
    if len(frame.depth_normalized_f32le) != expected_values * 4:
        raise RuntimeError("feature-v2 depth payload length drifted")
    frombuffer = getattr(torch_module, "frombuffer", None)
    kthvalue = getattr(torch_module, "kthvalue", None)
    if not callable(frombuffer) or not callable(kthvalue):
        raise TypeError("feature-v2 depth summary requires Torch buffer primitives")
    normalized_depth = frombuffer(
        bytearray(frame.depth_normalized_f32le), dtype=torch_module.float32
    )
    if (
        _tensor_shape(normalized_depth, name="feature-v2 normalized depth")
        != (expected_values,)
        or str(getattr(normalized_depth, "dtype", None)) != "torch.float32"
        or str(getattr(normalized_depth, "device", None)) != "cpu"
    ):
        raise RuntimeError("feature-v2 normalized depth buffer contract drifted")
    normalized_depth = normalized_depth.reshape(
        OFFICIAL_RGB_HEIGHT, OFFICIAL_RGB_WIDTH
    )

    summary_values: list[float] = []
    for sector in range(FEATURE_V2_DEPTH_SECTORS):
        start = (sector * OFFICIAL_RGB_WIDTH) // FEATURE_V2_DEPTH_SECTORS
        stop = ((sector + 1) * OFFICIAL_RGB_WIDTH) // FEATURE_V2_DEPTH_SECTORS
        normalized_sector = normalized_depth[:, start:stop]
        metric_sector = (
            normalized_sector * (OFFICIAL_DEPTH_MAX_M - OFFICIAL_DEPTH_MIN_M)
            + OFFICIAL_DEPTH_MIN_M
        )
        valid = (
            torch_module.isfinite(metric_sector)
            & (metric_sector > OFFICIAL_DEPTH_MIN_M)
            & (metric_sector <= OFFICIAL_DEPTH_MAX_M)
        )
        values = metric_sector[valid]
        valid_count = int(values.numel())
        total = int(metric_sector.numel())
        if valid_count == 0:
            summary_values.extend((0.0, 0.0, 0.0))
            continue
        q10_rank = max(1, math.ceil(0.10 * valid_count))
        q50_rank = max(1, math.ceil(0.50 * valid_count))
        summary_values.extend(
            (
                float(kthvalue(values, q10_rank).values.item()),
                float(kthvalue(values, q50_rank).values.item()),
                valid_count / total,
            )
        )
    if len(summary_values) != (
        FEATURE_V2_DEPTH_SECTORS * FEATURE_V2_DEPTH_VALUES_PER_SECTOR
    ) or any(not math.isfinite(value) for value in summary_values):
        raise RuntimeError("feature-v2 depth summary values are invalid")

    summary = torch_module.tensor(summary_values, dtype=torch_module.float32)
    summary = summary.detach().cpu().contiguous()
    if (
        _tensor_shape(summary, name="feature-v2 depth summary")
        != (
            FEATURE_V2_DEPTH_SECTORS * FEATURE_V2_DEPTH_VALUES_PER_SECTOR,
        )
        or str(getattr(summary, "dtype", None)) != "torch.float32"
        or str(getattr(summary, "device", None)) != "cpu"
    ):
        raise RuntimeError("feature-v2 depth summary must be contiguous FP32 CPU data")
    summary_audit = _validated_tensor_audit(
        tensor_auditor, summary, name="feature-v2 depth summary"
    )
    source_audit = {
        "frame_ref": frame.frame_ref,
        "depth_sha256": frame.depth_sha256,
        "physical_step": frame.physical_step,
        "shape_hw": [OFFICIAL_RGB_HEIGHT, OFFICIAL_RGB_WIDTH],
        "source_encoding": "raw-f32le-hw-normalized",
        "host_byteorder": sys.byteorder,
        "metric_range_m": [OFFICIAL_DEPTH_MIN_M, OFFICIAL_DEPTH_MAX_M],
        "valid_rule": "finite_and_min_exclusive_max_inclusive",
        "sector_count": FEATURE_V2_DEPTH_SECTORS,
        "sector_order": "left_to_right",
        "values_per_sector": ["q10_m", "q50_m", "valid_fraction"],
        "percentile_method": "nearest_rank",
    }
    return summary, summary_audit, source_audit


class OfficialFramePreprocessor:
    """Exact RGB/depth/pose/intrinsics preprocessing from the evaluator."""

    def __init__(
        self,
        *,
        image_processor,
        device,
        torch_module,
        numpy_module,
        image_module,
        to_numpy_array: Callable,
        filter_depth: Callable,
    ) -> None:
        self.image_processor = image_processor
        self.device = device
        self.torch = torch_module
        self.np = numpy_module
        self.Image = image_module
        self.to_numpy_array = to_numpy_array
        self.filter_depth = filter_depth
        crop = getattr(image_processor, "crop_size", None)
        if (
            not isinstance(crop, Mapping)
            or crop.get("height") != OFFICIAL_PROCESSOR_HEIGHT
            or crop.get("width") != OFFICIAL_PROCESSOR_WIDTH
        ):
            raise ValueError("official checkpoint image processor crop size drifted")

    def __call__(self, frames: tuple[OfficialFrame, ...]) -> PreparedOfficialFrames:
        if not isinstance(frames, tuple) or not frames:
            raise ValueError("official generation requires a non-empty frame tuple")
        images = []
        depths = []
        poses = []
        intrinsics = []
        for frame in frames:
            if not isinstance(frame, OfficialFrame):
                raise TypeError("official frame preprocessor received an invalid frame")
            rgb = self.np.frombuffer(frame.rgb_u8, dtype=self.np.uint8).reshape(
                OFFICIAL_RGB_HEIGHT, OFFICIAL_RGB_WIDTH, 3
            )
            image = self.Image.fromarray(rgb).convert("RGB")
            if image.size != (OFFICIAL_RGB_WIDTH, OFFICIAL_RGB_HEIGHT):
                raise RuntimeError("official RGB dimensions drifted")
            image_tensor = self.image_processor.preprocess(
                images=image, return_tensors="pt"
            )["pixel_values"][0]

            depth = self.np.frombuffer(
                frame.depth_normalized_f32le, dtype="<f4"
            ).reshape(OFFICIAL_RGB_HEIGHT, OFFICIAL_RGB_WIDTH).copy()
            depth = self.filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
            depth = depth * (OFFICIAL_DEPTH_MAX_M - OFFICIAL_DEPTH_MIN_M)
            depth = depth + OFFICIAL_DEPTH_MIN_M
            depth = depth * 1000
            depth_image = self.Image.fromarray(
                depth.astype(self.np.uint16), mode="I;16"
            ).resize(
                (OFFICIAL_PROCESSOR_WIDTH, OFFICIAL_PROCESSOR_HEIGHT),
                self.Image.NEAREST,
            )
            depth_array = self.to_numpy_array(depth_image) / 1000
            depth_tensor = self.torch.from_numpy(depth_array).float()

            pose_array = self.np.frombuffer(frame.pose_f64le, dtype="<f8").reshape(4, 4)
            pose_tensor = self.torch.from_numpy(pose_array)

            intrinsic = self.np.frombuffer(
                frame.intrinsics_f64le, dtype="<f8"
            ).reshape(4, 4).copy()
            intrinsic[0] /= OFFICIAL_RGB_WIDTH / OFFICIAL_PROCESSOR_WIDTH
            intrinsic[1] /= OFFICIAL_RGB_HEIGHT / OFFICIAL_PROCESSOR_HEIGHT
            intrinsic[0, 2] -= (
                OFFICIAL_PROCESSOR_WIDTH - OFFICIAL_PROCESSOR_HEIGHT
            ) / 2
            intrinsic_tensor = self.torch.from_numpy(intrinsic).float()

            images.append(image_tensor)
            depths.append(depth_tensor)
            poses.append(pose_tensor)
            intrinsics.append(intrinsic_tensor)

        tensors = {
            "images": self.torch.stack(images).unsqueeze(0),
            "depths": self.torch.stack(depths).unsqueeze(0),
            "poses": self.torch.stack(poses).unsqueeze(0),
            "intrinsics": self.torch.stack(intrinsics).unsqueeze(0),
        }
        for key, value in tuple(tensors.items()):
            tensors[key] = value.to(self.device).to(self.torch.bfloat16)
        return PreparedOfficialFrames(
            tensors=tensors,
            frame_refs=tuple(frame.frame_ref for frame in frames),
        )


class PinnedStreamVLNGenerationAdapter:
    """Stateful callable passed directly to :class:`OfficialKeepDriver`."""

    def __init__(
        self,
        *,
        model,
        tokenizer,
        device,
        env_id: int,
        torch_module,
        transformers_version: str,
        frame_preprocessor: Callable[[tuple[OfficialFrame, ...]], PreparedOfficialFrames],
        tensor_auditor: Callable[[object], Mapping] | None = None,
        requested_device=None,
        enable_feature_v2: bool = False,
    ) -> None:
        if type(env_id) is not int or env_id < 0:
            raise ValueError("official adapter env_id must be nonnegative")
        if not callable(frame_preprocessor):
            raise TypeError("official adapter frame preprocessor must be callable")
        if type(enable_feature_v2) is not bool:
            raise TypeError("official adapter feature-v2 opt-in must be boolean")
        self.model = model
        self.tokenizer = tokenizer
        self.requested_device = str(
            device if requested_device is None else requested_device
        )
        _, self.device = _resolve_device(device, torch_module)
        self.env_id = env_id
        self.torch = torch_module
        self.frame_preprocessor = frame_preprocessor
        self.tensor_auditor = tensor_auditor or (
            lambda tensor: _tensor_audit(tensor, self.torch)
        )
        self.output_ids = None
        self.past_key_values = None
        self.output_ref: str | None = None
        self.past_key_values_ref: str | None = None
        self.output_audit: dict | None = None
        self.inputs_embeds_seq_len: int | None = None
        self._last_frozen_feature_snapshot: FrozenStreamVLNFeatureSnapshot | None = None
        self.enable_feature_v2 = enable_feature_v2
        self._feature_v2_norm_module = None
        self._last_feature_v2_snapshot: StreamVLNFeatureV2Snapshot | None = None
        self.reset_ordinal = -1
        self.generation_ordinal = 0
        self.generation_audits: list[dict] = []

        actual_torch = _base_version(getattr(torch_module, "__version__", None))
        actual_transformers = _base_version(transformers_version)
        if actual_torch != PINNED_TORCH_VERSION:
            raise RuntimeError("official adapter Torch version drifted")
        if actual_transformers != PINNED_TRANSFORMERS_VERSION:
            raise RuntimeError("official adapter Transformers version drifted")
        config = getattr(model, "config", None)
        attention = getattr(config, "_attn_implementation", None)
        if attention != ATTENTION_IMPLEMENTATION:
            raise RuntimeError("official adapter attention implementation drifted")
        if not callable(getattr(model, "generate", None)) or not callable(
            getattr(model, "reset_for_env", None)
        ):
            raise TypeError("official adapter requires the released StreamVLN model API")
        self._require_model_internal_state()
        inner = getattr(model, "model", None)
        if getattr(inner, "num_history", None) != NUM_HISTORY:
            raise RuntimeError("official adapter num_history drifted")
        parameters = list(model.parameters())
        if not parameters or any(
            getattr(parameter, "requires_grad", None) is not False
            for parameter in parameters
        ):
            raise RuntimeError("official adapter requires a non-empty frozen model")
        floating_parameters = []
        for parameter in parameters:
            is_floating_point = getattr(parameter, "is_floating_point", None)
            if not callable(is_floating_point):
                raise TypeError("official model parameter floating-point API drifted")
            if bool(is_floating_point()):
                floating_parameters.append(parameter)
        if not floating_parameters or any(
            str(parameter.dtype) != MODEL_DTYPE for parameter in floating_parameters
        ):
            raise RuntimeError("official adapter floating model parameter dtype drifted")
        if any(
            str(getattr(parameter, "device", None)) != str(self.device)
            for parameter in floating_parameters
        ):
            raise RuntimeError("official adapter floating model parameter device drifted")
        if self.enable_feature_v2:
            self._feature_v2_norm_module = self._resolve_feature_v2_norm_module()
        self.runtime_receipt = official_runtime_receipt_template()
        validate_official_runtime_receipt(
            self.runtime_receipt,
            validation_level=TEST_ONLY_CONTRACT_VALIDATION_LEVEL,
        )

    @classmethod
    def from_released_checkpoint(
        cls,
        *,
        model_path: str,
        tokenizer_path: str,
        vision_tower_path: str,
        source_root: str,
        artifact_manifest: Mapping,
        device,
        local_rank: int,
        env_count: int,
        model_env_id: int,
        enable_feature_v2: bool = False,
    ) -> "PinnedStreamVLNGenerationAdapter":
        if not all(
            isinstance(path, str) and path
            for path in (model_path, tokenizer_path, vision_tower_path, source_root)
        ):
            raise ValueError("official source and artifact paths must be explicit")
        if (
            type(env_count) is not int
            or type(model_env_id) is not int
            or env_count != 1
            or model_env_id != 0
        ):
            raise ValueError(
                "official clean e48 factory requires env_count=1 and model_env_id=0"
            )
        requested_device = _validate_explicit_factory_cuda(device, local_rank)
        anchor = _require_pinned_release_manifest_sha256()
        if _manifest_sha256(artifact_manifest) != anchor:
            raise RuntimeError(
                "candidate manifest digest does not match pinned release"
            )

        candidate_observation = verify_full_content_artifact_manifest(
            artifact_manifest,
            source_root=source_root,
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            vision_tower_path=vision_tower_path,
        )
        if candidate_observation["manifest_sha256"] != anchor:
            raise RuntimeError(
                "verified manifest digest does not match pinned release"
            )
        source_root_path, source_observation = _verify_source_tree(source_root)
        candidate_observation["source"].update(source_observation)
        _install_source_import_root(source_root_path)

        torch_module = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
        numpy_module = importlib.import_module("numpy")
        image_module = importlib.import_module("PIL.Image")
        to_numpy_array = importlib.import_module(
            "transformers.image_utils"
        ).to_numpy_array
        filter_depth = importlib.import_module(
            "depth_camera_filtering"
        ).filter_depth
        model_module = importlib.import_module(PINNED_MODEL_CLASS_MODULE)
        model_class = model_module.StreamVLNForCausalLM
        candidate_observation["source"].update(
            _verify_imported_model_source(
                model_module, model_class, source_root_path
            )
        )
        if _base_version(torch_module.__version__) != PINNED_TORCH_VERSION:
            raise RuntimeError("official factory Torch version drifted")
        if _base_version(transformers.__version__) != PINNED_TRANSFORMERS_VERSION:
            raise RuntimeError("official factory Transformers version drifted")
        torch_module.cuda.set_device(local_rank)
        _, resolved_device = _resolve_device(requested_device, torch_module)
        if str(resolved_device) != requested_device:
            raise RuntimeError("official factory resolved CUDA device drifted")

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_path,
            model_max_length=MODEL_MAX_LENGTH,
            padding_side="right",
            local_files_only=True,
        )
        config = transformers.AutoConfig.from_pretrained(
            model_path, local_files_only=True
        )
        config.mm_vision_tower = vision_tower_path
        config.vision_tower = vision_tower_path
        model = model_class.from_pretrained(
            model_path,
            attn_implementation=ATTENTION_IMPLEMENTATION,
            torch_dtype=torch_module.bfloat16,
            config=config,
            low_cpu_mem_usage=False,
            local_files_only=True,
        )
        model.model.num_history = NUM_HISTORY
        model.requires_grad_(False)
        model.to(resolved_device)
        model.eval()
        model.reset(1)
        image_processor = model.get_vision_tower().image_processor
        frame_preprocessor = OfficialFramePreprocessor(
            image_processor=image_processor,
            device=resolved_device,
            torch_module=torch_module,
            numpy_module=numpy_module,
            image_module=image_module,
            to_numpy_array=to_numpy_array,
            filter_depth=filter_depth,
        )
        adapter = cls(
            model=model,
            tokenizer=tokenizer,
            device=resolved_device,
            requested_device=requested_device,
            env_id=model_env_id,
            torch_module=torch_module,
            transformers_version=transformers.__version__,
            frame_preprocessor=frame_preprocessor,
            enable_feature_v2=enable_feature_v2,
        )
        receipt = official_runtime_receipt_template()
        receipt["verification_scope"] = "full_content"
        receipt["observed_runtime"] = {
            "release_manifest_sha256": anchor,
            "release_manifest": copy.deepcopy(
                dict(candidate_observation["release_manifest"])
            ),
            "source": copy.deepcopy(dict(candidate_observation["source"])),
            "artifacts": copy.deepcopy(dict(candidate_observation["artifacts"])),
            "device": {
                "requested": requested_device,
                "resolved": str(adapter.device),
                "local_rank": local_rank,
            },
        }
        validate_official_runtime_receipt(receipt)
        adapter.runtime_receipt = receipt
        return adapter

    def _resolve_feature_v2_norm_module(self):
        hidden_size = getattr(getattr(self.model, "config", None), "hidden_size", None)
        if hidden_size != FEATURE_V2_CONTEXT_DIM:
            raise RuntimeError("feature-v2 requires the pinned 3584-wide language model")
        get_model = getattr(self.model, "get_model", None)
        if not callable(get_model):
            raise TypeError("feature-v2 cannot resolve the released language model")
        inner_model = get_model()
        norm = getattr(inner_model, "norm", None)
        if (
            norm is None
            or not type(norm).__name__.endswith("RMSNorm")
            or not callable(getattr(norm, "register_forward_hook", None))
        ):
            raise TypeError("feature-v2 requires the final language-model RMSNorm")
        return norm

    def _require_model_internal_state(self) -> tuple[list, list]:
        if not hasattr(self.model, "curr_t"):
            raise RuntimeError("official model curr_t state is missing")
        if not hasattr(self.model, "cache"):
            raise RuntimeError("official model cache state is missing")
        curr_t = self.model.curr_t
        cache = self.model.cache
        if not isinstance(curr_t, list):
            raise RuntimeError("official model curr_t state must be a list")
        if not isinstance(cache, list):
            raise RuntimeError("official model cache state must be a list")
        if self.env_id >= len(curr_t) or self.env_id >= len(cache):
            raise RuntimeError("official model state does not cover adapter env_id")
        if type(curr_t[self.env_id]) is not int or curr_t[self.env_id] < 0:
            raise RuntimeError("official model curr_t entry is invalid")
        if not isinstance(cache[self.env_id], Mapping):
            raise RuntimeError("official model cache entry must be a mapping")
        return curr_t, cache

    def _validate_model_input_tensors(
        self,
        model_inputs: Mapping,
        *,
        frame_count: int,
    ) -> None:
        expected = {
            "images": (1, frame_count, 3, OFFICIAL_PROCESSOR_HEIGHT, OFFICIAL_PROCESSOR_WIDTH),
            "depths": (1, frame_count, OFFICIAL_PROCESSOR_HEIGHT, OFFICIAL_PROCESSOR_WIDTH),
            "poses": (1, frame_count, 4, 4),
            "intrinsics": (1, frame_count, 4, 4),
        }
        for name, shape in expected.items():
            _validate_tensor_contract(
                model_inputs[name],
                name=name,
                expected_shape=shape,
                expected_dtype=MODEL_DTYPE,
                expected_device=self.device,
            )
        input_shape = _tensor_shape(model_inputs["inputs"], name="inputs")
        if len(input_shape) != 2 or input_shape[0] != 1 or input_shape[1] < 1:
            raise RuntimeError("official inputs tensor shape drifted")
        if str(getattr(model_inputs["inputs"], "dtype", None)) != "torch.int64":
            raise RuntimeError("official inputs tensor dtype drifted")
        if str(getattr(model_inputs["inputs"], "device", None)) != str(self.device):
            raise RuntimeError("official inputs tensor device drifted")

    def _validate_output_ids(self, output_ids) -> None:
        shape = _tensor_shape(output_ids, name="output_ids")
        if len(shape) != 2 or shape[0] != 1 or shape[1] < 1:
            raise RuntimeError("official output_ids tensor shape drifted")
        if str(getattr(output_ids, "dtype", None)) != "torch.int64":
            raise RuntimeError("official output_ids tensor dtype drifted")
        if str(getattr(output_ids, "device", None)) != str(self.device):
            raise RuntimeError("official output_ids tensor device drifted")

    def _frozen_model_state_signature(self) -> str:
        """Guard official state without copying the potentially huge full KV cache."""

        curr_t, cache = self._require_model_internal_state()
        cache_entry = cache[self.env_id]
        inputs_embeds = cache_entry.get("inputs_embeds")
        cache_other_state = {
            str(key): _state_identity_tree(value, self.torch)
            for key, value in sorted(
                cache_entry.items(), key=lambda pair: str(pair[0])
            )
            if key != "inputs_embeds"
        }
        return _json_sha256(
            {
                "curr_t": curr_t[self.env_id],
                "cache_keys": sorted(str(key) for key in cache_entry),
                "cache_inputs_embeds": (
                    None
                    if inputs_embeds is None
                    else _tensor_audit(inputs_embeds, self.torch)
                ),
                "cache_other_state": cache_other_state,
                "output_ids": (
                    None
                    if self.output_ids is None
                    else _tensor_audit(self.output_ids, self.torch)
                ),
                "past_key_values": _state_identity_tree(
                    self.past_key_values, self.torch
                ),
                "output_ref": self.output_ref,
                "past_key_values_ref": self.past_key_values_ref,
                "inputs_embeds_seq_len": self.inputs_embeds_seq_len,
                "reset_ordinal": self.reset_ordinal,
                "generation_ordinal": self.generation_ordinal,
            }
        )

    def extract_frozen_feature_snapshot(
        self,
        *,
        frame: OfficialFrame,
        instruction: str,
    ) -> FrozenStreamVLNFeatureSnapshot:
        """Extract a side-effect-free visual-language feature from clean e48.

        The current official frame follows the evaluator's exact preprocessor.
        Projected vision tokens and frozen instruction-token embeddings are
        independently mean pooled and L2 normalized before concatenation.
        No generation state may change while this method runs.
        """

        if self.reset_ordinal < 0:
            raise RuntimeError("official adapter must be reset before feature extraction")
        if not isinstance(frame, OfficialFrame) or frame.env_id != self.env_id:
            raise TypeError("frozen feature extraction requires this adapter's OfficialFrame")
        if not isinstance(instruction, str) or not instruction:
            raise ValueError("frozen feature instruction must be non-empty text")
        if hashlib.sha256(instruction.encode("utf-8")).hexdigest() != frame.instruction_sha256:
            raise ValueError("frozen feature instruction identity drifted")

        before = self._frozen_model_state_signature()
        prepared = self.frame_preprocessor((frame,))
        if not isinstance(prepared, PreparedOfficialFrames):
            raise TypeError("official frame preprocessor returned an invalid payload")
        if prepared.frame_refs != (frame.frame_ref,):
            raise RuntimeError("frozen feature frame provenance drifted")
        expected = {
            "images": (1, 1, 3, OFFICIAL_PROCESSOR_HEIGHT, OFFICIAL_PROCESSOR_WIDTH),
            "depths": (1, 1, OFFICIAL_PROCESSOR_HEIGHT, OFFICIAL_PROCESSOR_WIDTH),
            "poses": (1, 1, 4, 4),
            "intrinsics": (1, 1, 4, 4),
        }
        if set(prepared.tensors) != set(expected):
            raise RuntimeError("frozen feature frame tensor keys drifted")
        for name, shape in expected.items():
            _validate_tensor_contract(
                prepared.tensors[name],
                name=f"frozen feature {name}",
                expected_shape=shape,
                expected_dtype=MODEL_DTYPE,
                expected_device=self.device,
            )

        encode_images = getattr(self.model, "encode_images", None)
        pool_images = getattr(self.model, "get_2dPool", None)
        get_model = getattr(self.model, "get_model", None)
        tokenizer_encode = getattr(self.tokenizer, "encode", None)
        if not all(callable(value) for value in (
            encode_images, pool_images, get_model, tokenizer_encode
        )):
            raise TypeError("official frozen feature APIs are unavailable")
        token_ids = tokenizer_encode(instruction, add_special_tokens=False)
        if (
            not isinstance(token_ids, (list, tuple))
            or not token_ids
            or any(type(token_id) is not int or token_id < 0 for token_id in token_ids)
        ):
            raise RuntimeError("official tokenizer returned invalid instruction token ids")
        inner_model = get_model()
        embed_tokens = getattr(inner_model, "embed_tokens", None)
        if not callable(embed_tokens):
            raise TypeError("official frozen instruction embedding API is unavailable")

        with self.torch.inference_mode():
            pixels = prepared.tensors["images"].flatten(0, 1)
            projected = encode_images(pixels)
            pooled = pool_images(projected, 2)
            pooled_shape = _tensor_shape(pooled, name="frozen pooled vision tokens")
            if len(pooled_shape) != 3 or pooled_shape[0] != 1 or min(pooled_shape[1:]) < 1:
                raise RuntimeError("frozen pooled vision token shape drifted")
            vision = pooled[0].to(dtype=self.torch.float32).mean(dim=0)
            instruction_ids = self.torch.tensor(
                token_ids, dtype=self.torch.long, device=self.device
            )
            instruction_tokens = embed_tokens(instruction_ids)
            instruction_shape = _tensor_shape(
                instruction_tokens, name="frozen instruction embeddings"
            )
            if len(instruction_shape) != 2 or min(instruction_shape) < 1:
                raise RuntimeError("frozen instruction embedding shape drifted")
            text = instruction_tokens.to(dtype=self.torch.float32).mean(dim=0)

            hidden_size = getattr(getattr(self.model, "config", None), "hidden_size", None)
            vision_shape = _tensor_shape(vision, name="frozen vision feature")
            text_shape = _tensor_shape(text, name="frozen instruction feature")
            if (
                type(hidden_size) is not int
                or hidden_size < 1
                or vision_shape != (hidden_size,)
                or text_shape != (hidden_size,)
            ):
                raise RuntimeError("frozen feature hidden size drifted")
            if not bool(self.torch.isfinite(vision).all()) or not bool(
                self.torch.isfinite(text).all()
            ):
                raise RuntimeError("frozen feature branch contains non-finite values")
            vision_norm = vision.norm(p=2)
            text_norm = text.norm(p=2)
            if float(vision_norm.item()) <= 0.0 or float(text_norm.item()) <= 0.0:
                raise RuntimeError("frozen feature branch has zero norm")
            vision = (vision / vision_norm).detach().cpu().contiguous()
            text = (text / text_norm).detach().cpu().contiguous()
            feature = self.torch.cat((vision, text), dim=0).to(
                dtype=self.torch.float32
            ).contiguous()

        if _tensor_shape(feature, name="frozen feature") != (2 * hidden_size,):
            raise RuntimeError("frozen feature concatenation shape drifted")
        if str(feature.dtype) != "torch.float32" or str(feature.device) != "cpu":
            raise RuntimeError("frozen feature must be contiguous FP32 CPU data")
        after = self._frozen_model_state_signature()
        if after != before:
            raise RuntimeError("frozen feature extraction mutated official generation state")

        snapshot = FrozenStreamVLNFeatureSnapshot(
            feature=feature,
            physical_step=frame.physical_step,
            episode_generation=frame.episode_generation,
            frame_ref=frame.frame_ref,
            reset_ordinal=self.reset_ordinal,
            generation_ordinal=self.generation_ordinal,
            vision_dim=hidden_size,
            instruction_dim=hidden_size,
            vision_audit=_validated_tensor_audit(
                self.tensor_auditor, vision, name="frozen vision feature"
            ),
            instruction_audit=_validated_tensor_audit(
                self.tensor_auditor, text, name="frozen instruction feature"
            ),
            feature_audit=_validated_tensor_audit(
                self.tensor_auditor, feature, name="frozen concatenated feature"
            ),
        )
        self._last_frozen_feature_snapshot = snapshot
        return snapshot

    def latest_frozen_feature_snapshot(self) -> FrozenStreamVLNFeatureSnapshot:
        snapshot = self._last_frozen_feature_snapshot
        if snapshot is None:
            raise RuntimeError("no frozen feature has been extracted since reset")
        return FrozenStreamVLNFeatureSnapshot(
            feature=snapshot.feature.detach().cpu().clone(),
            physical_step=snapshot.physical_step,
            episode_generation=snapshot.episode_generation,
            frame_ref=snapshot.frame_ref,
            reset_ordinal=snapshot.reset_ordinal,
            generation_ordinal=snapshot.generation_ordinal,
            vision_dim=snapshot.vision_dim,
            instruction_dim=snapshot.instruction_dim,
            vision_audit=copy.deepcopy(dict(snapshot.vision_audit)),
            instruction_audit=copy.deepcopy(dict(snapshot.instruction_audit)),
            feature_audit=copy.deepcopy(dict(snapshot.feature_audit)),
        )

    def latest_feature_v2_snapshot(self) -> StreamVLNFeatureV2Snapshot:
        """Return an isolated copy of the latest successful generation latch."""

        snapshot = self._last_feature_v2_snapshot
        if snapshot is None:
            raise RuntimeError("no feature-v2 snapshot has been latched since reset")
        return StreamVLNFeatureV2Snapshot(
            contextual_hidden=(
                snapshot.contextual_hidden.detach().cpu().clone().contiguous()
            ),
            depth_summary=snapshot.depth_summary.detach().cpu().clone().contiguous(),
            physical_step=snapshot.physical_step,
            episode_generation=snapshot.episode_generation,
            frame_ref=snapshot.frame_ref,
            depth_sha256=snapshot.depth_sha256,
            reset_ordinal=snapshot.reset_ordinal,
            generation_ordinal=snapshot.generation_ordinal,
            prefill_length=snapshot.prefill_length,
            past_key_values_ref=snapshot.past_key_values_ref,
            inputs_embeds_ref=snapshot.inputs_embeds_ref,
            contextual_hidden_audit=copy.deepcopy(
                dict(snapshot.contextual_hidden_audit)
            ),
            depth_summary_audit=copy.deepcopy(dict(snapshot.depth_summary_audit)),
            depth_source_audit=copy.deepcopy(dict(snapshot.depth_source_audit)),
            state_refs_audit=copy.deepcopy(dict(snapshot.state_refs_audit)),
        )

    def reset(self, event: Mapping) -> None:
        if not isinstance(event, Mapping) or event.get("env_id") != self.env_id:
            raise ValueError("official adapter reset identity drifted")
        name = event.get("event")
        if name == "episode_reset":
            if event.get("physical_step") != 0 or event.get("terminal") is not False:
                raise ValueError("official episode reset event drifted")
        elif name == "cache_reset":
            next_step = event.get("next_step")
            if type(next_step) is not int or next_step < 1 or next_step % NUM_FRAMES:
                raise ValueError("official periodic reset event drifted")
        else:
            raise ValueError("unknown official adapter reset event")
        self._require_model_internal_state()
        self.model.reset_for_env(self.env_id)
        curr_t, cache = self._require_model_internal_state()
        self.output_ids = None
        self.past_key_values = None
        self.output_ref = None
        self.past_key_values_ref = None
        self.output_audit = None
        self.inputs_embeds_seq_len = None
        self._last_frozen_feature_snapshot = None
        self._last_feature_v2_snapshot = None
        self.reset_ordinal += 1
        self.generation_ordinal = 0
        if curr_t[self.env_id] != 0 or cache[self.env_id] != {}:
            raise RuntimeError("official model internal cache did not reset")

    def make_driver(self, *, session, episode_serial: int, episode_identity: Mapping):
        """Bind this adapter and its receipt/reset hook as one indivisible driver."""

        if getattr(session, "env_id", None) != self.env_id:
            raise ValueError("official adapter/session env_id mismatch")
        if __package__:
            from .streamvln_p8_successor_runtime import OfficialKeepDriver
        else:  # Explicit direct-file mode for local contract tests only.
            from streamvln_p8_successor_runtime import OfficialKeepDriver
        return OfficialKeepDriver(
            session=session,
            runtime_receipt=self.runtime_receipt,
            generation_adapter=self,
            cache_reset_adapter=self.reset,
            episode_serial=episode_serial,
            episode_identity=episode_identity,
        )

    def _validate_request_and_frames(
        self,
        request: OfficialGenerationRequest,
        frames: tuple[OfficialFrame, ...],
    ) -> None:
        if self.reset_ordinal < 0:
            raise RuntimeError("official adapter must be reset before generation")
        if not isinstance(request, OfficialGenerationRequest):
            raise TypeError("official adapter requires OfficialGenerationRequest")
        if request.env_id != self.env_id:
            raise RuntimeError("official generation env_id drifted")
        if tuple(frame.physical_step for frame in frames) != request.frame_steps:
            raise RuntimeError("official generation frame order drifted")
        if not frames or any(frame.env_id != self.env_id for frame in frames):
            raise RuntimeError("official generation frame env_id drifted")
        generations = {frame.episode_generation for frame in frames}
        if len(generations) != 1:
            raise RuntimeError("official generation mixed reset generations")
        if not request.time_ids or request.time_ids[-1] != request.physical_step:
            raise RuntimeError("official generation time_ids drifted")
        if any(
            current <= previous
            for previous, current in zip(request.time_ids, request.time_ids[1:])
        ):
            raise RuntimeError("official generation time_ids are not strictly increasing")
        if request.previous_output_ref != self.output_ref:
            raise RuntimeError("official generation output reference drifted")
        if request.past_key_values_ref != self.past_key_values_ref:
            raise RuntimeError("official generation KV reference drifted")
        if (self.output_ids is None) != (self.past_key_values is None):
            raise RuntimeError("official adapter output/KV state is inconsistent")
        if (self.output_ids is None) != (request.previous_output_ref is None):
            raise RuntimeError("official generation continuation state drifted")
        if (self.output_ids is None) != (self.output_audit is None):
            raise RuntimeError("official adapter output tensor audit state is inconsistent")
        if (self.output_ids is None) != (self.inputs_embeds_seq_len is None):
            raise RuntimeError(
                "official adapter inputs_embeds temporal state is inconsistent"
            )
        if self.output_ids is not None:
            current_audit = _validated_tensor_audit(
                self.tensor_auditor, self.output_ids, name="output_ids"
            )
            if current_audit != self.output_audit:
                raise RuntimeError("official output_ids tensor changed after generation")

    def __call__(
        self,
        request: OfficialGenerationRequest,
        frames: tuple[OfficialFrame, ...],
    ):
        self._validate_request_and_frames(request, frames)
        prepared = self.frame_preprocessor(frames)
        if not isinstance(prepared, PreparedOfficialFrames):
            raise TypeError("official frame preprocessor returned an invalid payload")
        if prepared.frame_refs != tuple(frame.frame_ref for frame in frames):
            raise RuntimeError("official frame preprocessing provenance drifted")
        expected_tensor_keys = {"images", "depths", "poses", "intrinsics"}
        if set(prepared.tensors) != expected_tensor_keys:
            raise RuntimeError("official frame tensor keys drifted")

        input_ids = build_official_input_ids(
            request=request,
            tokenizer=self.tokenizer,
            torch_module=self.torch,
            previous_output_ids=self.output_ids,
        ).to(self.device)
        model_inputs = dict(prepared.tensors)
        model_inputs.update(
            {
                "inputs": input_ids,
                "env_id": self.env_id,
                "time_ids": [list(request.time_ids)],
                "task_type": [0],
            }
        )
        self._validate_model_input_tensors(model_inputs, frame_count=len(frames))
        tensor_audits = {
            key: _validated_tensor_audit(
                self.tensor_auditor, value, name=key
            )
            for key, value in model_inputs.items()
            if key in expected_tensor_keys or key == "inputs"
        }
        curr_t, _ = self._require_model_internal_state()
        curr_t_before = curr_t[self.env_id]
        feature_v2_capture = None
        feature_v2_depth = None
        feature_v2_frame = None
        if self.enable_feature_v2:
            feature_v2_frame = frames[-1]
            if feature_v2_frame.physical_step != request.physical_step:
                raise RuntimeError("feature-v2 current-frame physical step drifted")
            feature_v2_depth = build_feature_v2_depth_summary(
                feature_v2_frame,
                torch_module=self.torch,
                tensor_auditor=self.tensor_auditor,
            )
            norm = self._feature_v2_norm_module
            if norm is None:
                norm = self._resolve_feature_v2_norm_module()
                self._feature_v2_norm_module = norm

            def capture_first_prefill(_module, _inputs, output):
                nonlocal feature_v2_capture
                if feature_v2_capture is not None:
                    return None
                output_shape = _tensor_shape(
                    output, name="feature-v2 final RMSNorm output"
                )
                if (
                    len(output_shape) != 3
                    or output_shape[0] != 1
                    or output_shape[1] < 1
                    or output_shape[2] != FEATURE_V2_CONTEXT_DIM
                ):
                    raise RuntimeError("feature-v2 prefill hidden shape drifted")
                hidden = (
                    output[0, -1, :]
                    .detach()
                    .to(device="cpu", dtype=self.torch.float32)
                    .contiguous()
                )
                if (
                    _tensor_shape(hidden, name="feature-v2 contextual hidden")
                    != (FEATURE_V2_CONTEXT_DIM,)
                    or str(getattr(hidden, "dtype", None)) != "torch.float32"
                    or str(getattr(hidden, "device", None)) != "cpu"
                    or not bool(self.torch.isfinite(hidden).all())
                ):
                    raise RuntimeError(
                        "feature-v2 contextual hidden must be finite FP32 CPU data"
                    )
                feature_v2_capture = {
                    "contextual_hidden": hidden,
                    "prefill_length": output_shape[1],
                }
                return None

            hook_handle = norm.register_forward_hook(capture_first_prefill)
            try:
                with self.torch.inference_mode():
                    outputs = self.model.generate(
                        **model_inputs,
                        **GENERATION_KWARGS,
                        past_key_values=self.past_key_values,
                    )
            finally:
                hook_handle.remove()
            if feature_v2_capture is None:
                raise RuntimeError("feature-v2 generate did not execute the final RMSNorm")
        else:
            with self.torch.inference_mode():
                outputs = self.model.generate(
                    **model_inputs,
                    **GENERATION_KWARGS,
                    past_key_values=self.past_key_values,
                )
        if not hasattr(outputs, "sequences") or not hasattr(
            outputs, "past_key_values"
        ):
            raise RuntimeError("official model generation output schema drifted")
        curr_t, cache = self._require_model_internal_state()
        if curr_t[self.env_id] != curr_t_before + 1:
            raise RuntimeError("official model curr_t did not advance exactly once")
        cache_entry = cache[self.env_id]
        if "inputs_embeds" not in cache_entry:
            raise RuntimeError("official model inputs_embeds cache is missing")
        inputs_embeds = cache_entry["inputs_embeds"]
        embeds_shape = _tensor_shape(inputs_embeds, name="model inputs_embeds cache")
        if (
            len(embeds_shape) != 3
            or embeds_shape[0] != 1
            or embeds_shape[1] < 1
            or embeds_shape[2] < 1
            or str(getattr(inputs_embeds, "dtype", None)) != MODEL_DTYPE
            or str(getattr(inputs_embeds, "device", None)) != str(self.device)
        ):
            raise RuntimeError("official model inputs_embeds cache contract drifted")

        previous_embeds_seq_len = self.inputs_embeds_seq_len
        current_embeds_seq_len = embeds_shape[1]
        if curr_t_before == 0:
            if previous_embeds_seq_len is not None:
                raise RuntimeError("official first generation retained inputs_embeds state")
        elif (
            previous_embeds_seq_len is None
            or current_embeds_seq_len <= previous_embeds_seq_len
        ):
            raise RuntimeError(
                "official model inputs_embeds cache did not grow on continuation"
            )

        self._validate_output_ids(outputs.sequences)
        if outputs.past_key_values is None:
            raise RuntimeError("official model generation returned no KV cache")
        self.output_ids = outputs.sequences
        self.past_key_values = outputs.past_key_values
        self.inputs_embeds_seq_len = current_embeds_seq_len
        output_audit = _validated_tensor_audit(
            self.tensor_auditor, self.output_ids, name="output_ids"
        )
        self.output_audit = output_audit
        self.output_ref = _json_sha256(
            {
                "schema": ADAPTER_SCHEMA_NAME,
                "version": ADAPTER_SCHEMA_VERSION,
                "env_id": self.env_id,
                "reset_ordinal": self.reset_ordinal,
                "generation_ordinal": self.generation_ordinal,
                "physical_step": request.physical_step,
                "frame_refs": list(prepared.frame_refs),
                "output": output_audit,
            }
        )
        cache_structure = _cache_structure(self.past_key_values, self.torch)
        self.past_key_values_ref = _json_sha256(
            {
                "output_ref": self.output_ref,
                "reset_ordinal": self.reset_ordinal,
                "generation_ordinal": self.generation_ordinal,
                "structure": cache_structure,
            }
        )
        decoded_values = self.tokenizer.batch_decode(
            self.output_ids, **DECODE_KWARGS
        )
        if (
            not isinstance(decoded_values, list)
            or len(decoded_values) != 1
            or not isinstance(decoded_values[0], str)
        ):
            raise RuntimeError("official tokenizer decode output schema drifted")
        decoded = decoded_values[0].strip()
        feature_v2_snapshot = None
        feature_v2_audit = None
        if self.enable_feature_v2:
            if (
                feature_v2_capture is None
                or feature_v2_depth is None
                or feature_v2_frame is None
                or self.past_key_values_ref is None
            ):
                raise RuntimeError("feature-v2 generation candidate is incomplete")
            contextual_hidden = feature_v2_capture["contextual_hidden"]
            prefill_length = feature_v2_capture["prefill_length"]
            depth_summary, depth_summary_audit, depth_source_audit = feature_v2_depth
            contextual_hidden_audit = _validated_tensor_audit(
                self.tensor_auditor,
                contextual_hidden,
                name="feature-v2 contextual hidden",
            )
            inputs_identity_sha256 = _json_sha256(
                _state_identity_tree(inputs_embeds, self.torch)
            )
            kv_identity_sha256 = _json_sha256(
                _state_identity_tree(self.past_key_values, self.torch)
            )
            inputs_embeds_ref = _json_sha256(
                {
                    "env_id": self.env_id,
                    "reset_ordinal": self.reset_ordinal,
                    "generation_ordinal": self.generation_ordinal,
                    "physical_step": request.physical_step,
                    "identity_sha256": inputs_identity_sha256,
                }
            )
            state_refs_audit = {
                "past_key_values_ref": self.past_key_values_ref,
                "past_key_values_identity_sha256": kv_identity_sha256,
                "inputs_embeds_ref": inputs_embeds_ref,
                "inputs_embeds_identity_sha256": inputs_identity_sha256,
                "inputs_embeds_seq_len": current_embeds_seq_len,
                "copies_large_kv_payload": False,
            }
            feature_v2_snapshot = StreamVLNFeatureV2Snapshot(
                contextual_hidden=contextual_hidden,
                depth_summary=depth_summary,
                physical_step=request.physical_step,
                episode_generation=feature_v2_frame.episode_generation,
                frame_ref=feature_v2_frame.frame_ref,
                depth_sha256=feature_v2_frame.depth_sha256,
                reset_ordinal=self.reset_ordinal,
                generation_ordinal=self.generation_ordinal,
                prefill_length=prefill_length,
                past_key_values_ref=self.past_key_values_ref,
                inputs_embeds_ref=inputs_embeds_ref,
                contextual_hidden_audit=contextual_hidden_audit,
                depth_summary_audit=depth_summary_audit,
                depth_source_audit=depth_source_audit,
                state_refs_audit=state_refs_audit,
            )
            feature_v2_audit = {
                "reset_ordinal": self.reset_ordinal,
                "generation_ordinal": self.generation_ordinal,
                "physical_step": request.physical_step,
                "prefill_length": prefill_length,
                "frame_ref": feature_v2_frame.frame_ref,
                "depth_sha256": feature_v2_frame.depth_sha256,
                "contextual_hidden_audit": contextual_hidden_audit,
                "depth_summary_audit": depth_summary_audit,
                "depth_source_audit": depth_source_audit,
                "state_refs_audit": state_refs_audit,
            }
        audit = {
            "schema_name": ADAPTER_SCHEMA_NAME,
            "schema_version": ADAPTER_SCHEMA_VERSION,
            "env_id": self.env_id,
            "reset_ordinal": self.reset_ordinal,
            "generation_ordinal": self.generation_ordinal,
            "physical_step": request.physical_step,
            "frame_refs": list(prepared.frame_refs),
            "request_contract_sha256": request.contract_sha256,
            "tensor_audits": tensor_audits,
            "output_audit": output_audit,
            "output_ref": self.output_ref,
            "past_key_values_ref": self.past_key_values_ref,
            "past_key_values_structure_sha256": _json_sha256(cache_structure),
            "model_curr_t_before": curr_t_before,
            "model_curr_t_after": self.model.curr_t[self.env_id],
            "inputs_embeds_seq_len_before": previous_embeds_seq_len,
            "inputs_embeds_seq_len_after": current_embeds_seq_len,
            "state_audit_scope": copy.deepcopy(STATE_AUDIT_SCOPE),
            "decoded_sha256": hashlib.sha256(decoded.encode("utf-8")).hexdigest(),
            "tokenizer_preprocess_contract": TOKENIZER_PREPROCESS_CONTRACT,
            "frame_preprocess_contract": FRAME_PREPROCESS_CONTRACT,
            "model_internal_state_contract": MODEL_INTERNAL_STATE_CONTRACT,
        }
        if feature_v2_audit is not None:
            audit["feature_v2"] = copy.deepcopy(feature_v2_audit)
        self.generation_audits.append(audit)
        if feature_v2_snapshot is not None:
            self._last_feature_v2_snapshot = feature_v2_snapshot
        self.generation_ordinal += 1

        if __package__:
            from .streamvln_p8_successor_runtime import OfficialGenerationResult
        else:  # Explicit direct-file mode for local contract tests only.
            from streamvln_p8_successor_runtime import OfficialGenerationResult
        return OfficialGenerationResult(
            decoded_sequence=decoded,
            output_ref=self.output_ref,
            past_key_values_ref=self.past_key_values_ref,
        )


__all__ = [
    "ADAPTER_SCHEMA_NAME",
    "ADAPTER_SCHEMA_VERSION",
    "ARTIFACT_MANIFEST_SCHEMA_NAME",
    "ARTIFACT_MANIFEST_SCHEMA_VERSION",
    "FEATURE_V2_CONTEXT_DIM",
    "FEATURE_V2_DEPTH_SECTORS",
    "FEATURE_V2_DEPTH_VALUES_PER_SECTOR",
    "FrozenStreamVLNFeatureSnapshot",
    "OfficialFramePreprocessor",
    "PinnedStreamVLNGenerationAdapter",
    "PreparedOfficialFrames",
    "StreamVLNFeatureV2Snapshot",
    "build_feature_v2_depth_summary",
    "build_full_content_artifact_manifest",
    "build_official_input_ids",
    "verify_full_content_artifact_manifest",
]
