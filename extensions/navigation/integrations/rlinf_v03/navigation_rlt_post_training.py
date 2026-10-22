#!/usr/bin/env python3
"""Fail-closed Stage2-to-evaluation handoff for Navigation-RLT.

The runner is intended to execute on the A800 host after Stage2 has completed.
``plan`` performs local preflight and emits an immutable, reviewable plan.
``execute`` revalidates that plan, transfers the verified Stage2 checkpoint to
ebk through a temporary name, runs Native/A3/Navigation-RLT sequentially, pulls
the six result/completion files back to the A800 run root, then invokes the
matched comparator and OBS archiver.  No remote action occurs in ``plan``.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import stat
import struct
import subprocess
import sys
import tempfile
import time
from typing import Mapping, Sequence
import zipfile


RUNS_ROOT = Path("/srv/navigation/runs")
DEFAULT_OBS_ROOT = Path("/obs/navigation-example")
PLAN_SCHEMA = "navigation_rlt_post_training_plan"
PLAN_VERSION = 1
COPY_SCHEMA = "navigation_rlt_verified_checkpoint_copy"
HANDOFF_SCHEMA = "navigation_rlt_remote_evaluation_handoff"
COMPLETION_SCHEMA = "navigation_rlt_post_training_completion"
STAGE_SCHEMA = "navigation_rlt_a800_stage"
VERIFICATION_SCHEMA = "navigation_rlt_a800_verification"
TRAINING_RESULT_SCHEMA = "navigation_rlt_rlinf_fsdp_training"
COLLECTION_SCHEMA = "navigation_rlt_experiment_manifest"
EVAL_PLAN_SCHEMA = "navigation_rlt_parallel_rollout_plan"
EVAL_INVENTORY_SCHEMA = "navigation_rlt_evaluation_code_inventory"
MATCHED_SCHEMA = "navigation_rlt_matched_comparison"
TRAINING_CODE_VERIFICATION_SCHEMA = "navigation_rlt_a800_code_verification"
TRAINING_PROTOCOL_SCHEMA = "navigation_rlt_training_protocol"
TRAINING_INVENTORY_SCHEMA = "navigation_rlt_code_archive_inventory"
STAGE2_REPAIR_PROTOCOL_SCHEMA = "navigation_rlt_stage2_repair_protocol"
STAGE2_REPAIR_OBJECTIVE_SCHEMA = "proposal_conditioned_finite_candidate_cql_v1"
STAGE2_BEHAVIOR_KL_OBJECTIVE_SCHEMA = (
    "proposal_conditioned_finite_candidate_cql_behavior_kl_v2"
)
STAGE2_FORMAL_SCHEMA = "navigation_rlt_stage2_behavior_kl_formal"
STAGE2_DERIVATION_SCHEMA = "navigation_rlt_anticollapse_formal_prepare"
STAGE2_ELIGIBILITY_SCHEMA = (
    "credit_mask_true_and_requested_gate_equals_executed_gate_v1"
)
STAGE2_GATE_NAMES = ("KEEP", "ADJUST", "STOP", "RECOVER")
STAGE2_CQL_ALPHA = 1.0
STAGE2_CQL_TEMPERATURE = 1.0
STAGE2_MINIMUM_BEHAVIOR_GATE_COUNT = 1
ARMS = ("native_lightnav", "a3_ppo", "navigation_rlt")
BEHAVIORS = ("native", "a3", "navigation_rlt")
ARM_TO_BEHAVIOR = dict(zip(ARMS, BEHAVIORS))
BEHAVIOR_TO_ARM = dict(zip(BEHAVIORS, ARMS))
ARM_TO_EVAL_BEHAVIOR = {
    "native_lightnav": "native",
    "a3_ppo": "a3",
    "navigation_rlt": "navigation-rlt",
}
MAX_STEPS = 160
EVAL_SEED = 2026100202
SHA256_RE = re.compile(r"[0-9a-f]{64}")
SHA1_RE = re.compile(r"[0-9a-f]{40}")
RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
HOST_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.@-]{0,127}")
REMOTE_TOKEN_RE = re.compile(r"[A-Za-z0-9_./:@+=,-]+")
OUTPUT_RELATIVE_PATHS = {
    "root": Path("post-training"),
    "verified_copy_receipt": Path("post-training/VERIFIED_COPY.json"),
    "evaluation_handoff_receipt": Path("post-training/REMOTE_EVAL_COMPLETED.json"),
    "matched_result": Path("post-training/matched_comparison.json"),
    "matched_markdown": Path("post-training/matched_comparison.md"),
    "completion": Path("post-training/POST_TRAINING_COMPLETED.json"),
}
REQUIRED_EVAL_FILES = frozenset(
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
TRAINING_SOURCE_FILES_BY_CLAIM = {
    "trainer_sha256": "navigation_rlt_fsdp_train.py",
    "policy_sha256": "navigation_rlt_policy.py",
    "replay_sha256": "navigation_rlt_replay.py",
    "replay_relocator_sha256": "navigation_rlt_relocate_replay.py",
    "stage1_gate_sha256": "navigation_rlt_stage1_gate.py",
    "preparer_sha256": "navigation_rlt_prepare_training.py",
    "promoter_sha256": "navigation_rlt_promote_replay.py",
    "wrapper_sha256": "run_navigation_rlt_a800.sh",
    "gpu_monitor_sha256": "lightnav_a3_gpu_monitor.py",
}
TRAINING_SOURCE_FILES = frozenset(TRAINING_SOURCE_FILES_BY_CLAIM.values())
STAGE2_REPAIR_REQUIRED_SOURCE_CLAIMS = frozenset(
    {
        "trainer_sha256",
        "policy_sha256",
        "replay_sha256",
        "preparer_sha256",
        "promoter_sha256",
        "wrapper_sha256",
        "gpu_monitor_sha256",
    }
)


class PostTrainingError(RuntimeError):
    """Raised when a post-training identity or transition is unsafe."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise PostTrainingError(f"{name} must be a lowercase SHA-256")
    return value


def _sha1(value: object, name: str) -> str:
    if not isinstance(value, str) or SHA1_RE.fullmatch(value) is None:
        raise PostTrainingError(f"{name} must be a lowercase full Git SHA-1")
    return value


def _stage2_training_additions(
    value: object, *, name: str, exact: bool = False
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise PostTrainingError(f"{name} is absent")
    fields = (
        "objective_schema",
        "cql_alpha",
        "cql_temperature",
        "minimum_behavior_gate_count",
    )
    schema = value.get("objective_schema")
    if schema == STAGE2_BEHAVIOR_KL_OBJECTIVE_SCHEMA:
        fields += ("behavior_kl_weight",)
        weight = value.get("behavior_kl_weight")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight != 1.0:
            raise PostTrainingError("Stage2 behavior KL weight differs")
    elif schema != STAGE2_REPAIR_OBJECTIVE_SCHEMA or "behavior_kl_weight" in value:
        raise PostTrainingError("Stage2 repair objective schema differs")
    if exact and set(value) != set(fields):
        raise PostTrainingError(f"{name} fields differ")
    additions = {field: value.get(field) for field in fields}
    alpha = additions["cql_alpha"]
    temperature = additions["cql_temperature"]
    minimum = additions["minimum_behavior_gate_count"]
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not math.isfinite(float(alpha))
        or float(alpha) != STAGE2_CQL_ALPHA
    ):
        raise PostTrainingError("Stage2 repair cql_alpha differs")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(float(temperature))
        or float(temperature) != STAGE2_CQL_TEMPERATURE
    ):
        raise PostTrainingError("Stage2 repair cql_temperature differs")
    if type(minimum) is not int or minimum != STAGE2_MINIMUM_BEHAVIOR_GATE_COUNT:
        raise PostTrainingError("Stage2 repair minimum behavior gate count differs")
    return additions


def _stage2_support_contract(value: object, *, name: str) -> dict[str, object]:
    fields = {
        "eligibility_schema",
        "gate_names",
        "gate_counts",
        "gate_support_mask",
        "minimum_behavior_gate_count",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise PostTrainingError(f"{name} fields differ")
    if value.get("eligibility_schema") != STAGE2_ELIGIBILITY_SCHEMA:
        raise PostTrainingError("Stage2 repair eligibility schema differs")
    if value.get("gate_names") != list(STAGE2_GATE_NAMES):
        raise PostTrainingError("Stage2 repair gate names differ")
    minimum = value.get("minimum_behavior_gate_count")
    counts = value.get("gate_counts")
    mask = value.get("gate_support_mask")
    if type(minimum) is not int or minimum != STAGE2_MINIMUM_BEHAVIOR_GATE_COUNT:
        raise PostTrainingError("Stage2 repair support minimum gate count differs")
    if (
        not isinstance(counts, list)
        or len(counts) != len(STAGE2_GATE_NAMES)
        or any(type(count) is not int or count < 0 for count in counts)
    ):
        raise PostTrainingError("Stage2 repair gate counts are invalid")
    if (
        not isinstance(mask, list)
        or len(mask) != len(STAGE2_GATE_NAMES)
        or any(type(enabled) is not bool for enabled in mask)
    ):
        raise PostTrainingError("Stage2 repair gate support mask is invalid")
    if mask != [count >= minimum for count in counts] or not any(mask):
        raise PostTrainingError("Stage2 repair gate support mask differs from counts")
    if mask[STAGE2_GATE_NAMES.index("RECOVER")] is not False:
        raise PostTrainingError("Stage2 repair requires RECOVER support to be false")
    return dict(value)


def _load_json(path: Path, name: str) -> dict[str, object]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda item: (_ for _ in ()).throw(
                PostTrainingError(f"{name} contains non-finite {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PostTrainingError(f"{name} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise PostTrainingError(f"{name} must contain an object")
    return value


def _regular_file(path: Path, root: Path, name: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink():
        raise PostTrainingError(f"{name} must not be a symbolic link")
    try:
        resolved = raw.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise PostTrainingError(f"{name} is absent or escapes {root}") from exc
    if not resolved.is_file():
        raise PostTrainingError(f"{name} must be a regular file")
    return resolved


def _regular_directory(path: Path, root: Path, name: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink():
        raise PostTrainingError(f"{name} must not be a symbolic link")
    try:
        resolved = raw.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise PostTrainingError(f"{name} is absent or escapes {root}") from exc
    if not resolved.is_dir():
        raise PostTrainingError(f"{name} must be a directory")
    return resolved


def _fsync_parent(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path.parent, flags)
    except OSError:
        if os.name == "nt":
            return
        raise
    try:
        os.fsync(descriptor)
    except OSError:
        if os.name != "nt":
            raise
    finally:
        os.close(descriptor)


def _file_identity(status: os.stat_result) -> tuple[int, int]:
    return int(status.st_dev), int(status.st_ino)


def _unlink_if_same(path: Path, identity: tuple[int, int]) -> None:
    try:
        status = path.lstat()
    except FileNotFoundError:
        return
    if _file_identity(status) != identity:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _publish_file_new(
    temporary: Path, destination: Path, identity: tuple[int, int]
) -> None:
    """Atomically publish a same-directory file without replacing a winner."""

    temporary_parent = temporary.parent.resolve(strict=True)
    destination_parent = destination.parent.resolve(strict=True)
    if temporary_parent != destination_parent:
        raise PostTrainingError("atomic publication requires one destination directory")
    os.link(temporary, destination, follow_symlinks=False)
    _fsync_parent(destination)
    _unlink_if_same(temporary, identity)
    _fsync_parent(destination)


def _copy_file_new(
    source: Path, destination: Path, *, expected_sha256: str, expected_size_bytes: int
) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".staging-", dir=destination.parent
    )
    temporary = Path(temporary_name)
    temporary_identity = _file_identity(os.fstat(descriptor))
    digest = hashlib.sha256()
    size = 0
    try:
        with source.open("rb") as input_stream, os.fdopen(descriptor, "wb") as output:
            for block in iter(lambda: input_stream.read(1024 * 1024), b""):
                output.write(block)
                digest.update(block)
                size += len(block)
            output.flush()
            os.fsync(output.fileno())
        if size != expected_size_bytes or digest.hexdigest() != expected_sha256:
            raise PostTrainingError("local publication source changed while copying")
        _publish_file_new(temporary, destination, temporary_identity)
    except BaseException:
        _unlink_if_same(temporary, temporary_identity)
        raise


def _atomic_json_new(path: Path, value: object) -> str:
    payload = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".staging-", dir=path.parent
    )
    temporary = Path(temporary_name)
    temporary_identity = _file_identity(os.fstat(descriptor))
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _publish_file_new(temporary, path, temporary_identity)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        _unlink_if_same(temporary, temporary_identity)
        raise
    return hashlib.sha256(payload).hexdigest()


def _path_is_link_or_reparse(path: Path) -> bool:
    try:
        status = path.lstat()
    except OSError as exc:
        raise PostTrainingError(f"cannot inspect path component: {path}") from exc
    attributes = int(getattr(status, "st_file_attributes", 0))
    reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return stat.S_ISLNK(status.st_mode) or bool(attributes & reparse_flag)


def _safe_directory_chain(
    run_root: Path, directory: Path, name: str, *, create: bool
) -> None:
    try:
        relative = directory.relative_to(run_root)
    except ValueError as exc:
        raise PostTrainingError(f"{name} escapes the A800 run root") from exc
    current = run_root
    components = (current, *(run_root.joinpath(*relative.parts[:index]) for index in range(1, len(relative.parts) + 1)))
    for current in components:
        if not os.path.lexists(current):
            if not create:
                raise PostTrainingError(f"{name} parent is absent: {current}")
            try:
                current.mkdir()
            except FileExistsError:
                pass
        if _path_is_link_or_reparse(current):
            raise PostTrainingError(f"{name} parent is a link/reparse point: {current}")
        try:
            status = current.lstat()
        except OSError as exc:
            raise PostTrainingError(f"cannot inspect {name} parent: {current}") from exc
        if not stat.S_ISDIR(status.st_mode):
            raise PostTrainingError(f"{name} parent is not a directory: {current}")


def _fixed_outputs(run_root: Path) -> dict[str, Path]:
    return {key: run_root / relative for key, relative in OUTPUT_RELATIVE_PATHS.items()}


def _validated_outputs(
    plan: Mapping[str, object], run_root: Path, *, create_parent: bool
) -> dict[str, Path]:
    outputs = plan.get("outputs")
    expected = _fixed_outputs(run_root)
    if not isinstance(outputs, Mapping) or set(outputs) != set(expected):
        raise PostTrainingError("post-training output inventory keys differ")
    for key, path in expected.items():
        value = outputs.get(key)
        if not isinstance(value, str) or value != str(path):
            raise PostTrainingError(f"post-training output path differs: {key}")
    _safe_directory_chain(
        run_root,
        expected["root"],
        "post-training output",
        create=create_parent,
    )
    for key, path in expected.items():
        if key != "root":
            _safe_directory_chain(
                run_root,
                path.parent,
                f"post-training output {key}",
                create=create_parent,
            )
    return expected


def _validate_plan_location(
    raw_plan_path: Path, plan_path: Path, run_root: Path
) -> None:
    try:
        raw_plan_path.relative_to(run_root)
        plan_path.relative_to(run_root)
    except ValueError as exc:
        raise PostTrainingError("post-training plan escapes the A800 run root") from exc
    _safe_directory_chain(
        run_root, raw_plan_path.parent, "post-training plan", create=False
    )
    if _path_is_link_or_reparse(raw_plan_path):
        raise PostTrainingError("post-training plan must not be a link/reparse point")
    try:
        status = plan_path.lstat()
    except OSError as exc:
        raise PostTrainingError("post-training plan disappeared") from exc
    if not stat.S_ISREG(status.st_mode):
        raise PostTrainingError("post-training plan is not a regular file")


def _remote_path(value: str, name: str) -> str:
    if not isinstance(value, str) or REMOTE_TOKEN_RE.fullmatch(value) is None:
        raise PostTrainingError(f"{name} contains unsupported characters")
    pure = PurePosixPath(value)
    if not pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise PostTrainingError(f"{name} must be a canonical absolute POSIX path")
    return pure.as_posix()


def _host(value: str) -> str:
    if HOST_RE.fullmatch(value) is None or value.startswith("-"):
        raise PostTrainingError("ebk host is invalid")
    return value


def _verify_file_sha(path: Path, expected: object, name: str) -> str:
    digest = _sha256(expected, f"expected {name} SHA-256")
    if sha256_file(path) != digest:
        raise PostTrainingError(f"{name} SHA-256 differs")
    return digest


def _local_evaluation_source(name: str, source_root: Path | None = None) -> Path:
    """Resolve one inventory entry in either the repo tree or a flat bundle."""

    relative = PurePosixPath(name)
    if source_root is not None:
        root = Path(source_root).expanduser().absolute()
        if _path_is_link_or_reparse(root) or not root.is_dir():
            raise PostTrainingError("explicit evaluation source root is absent or linked")
        root = root.resolve(strict=True)
        candidate = root.joinpath(*relative.parts)
        _safe_directory_chain(root, candidate.parent, "explicit evaluation source", create=False)
        source = _regular_file(candidate, root, f"evaluation source {name}")
        if os.name != "nt" and source.stat().st_mode & 0o222:
            raise PostTrainingError(f"explicit evaluation source {name} must be read-only")
        return source
    code_root = Path(__file__).resolve(strict=True).parent
    candidates = [code_root.joinpath(*relative.parts)]
    if relative.parts and relative.parts[0] == "navrl":
        # Source checkout: integrations/rlinf_v03 sits below rlinf-nav, while
        # deployment bundles place navrl directly beside the evaluator.
        candidates.append(code_root.parents[1].joinpath(*relative.parts))
    regular: list[Path] = []
    for candidate in candidates:
        if candidate.is_symlink():
            continue
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        if resolved.is_file():
            regular.append(resolved)
    if len(regular) != 1:
        raise PostTrainingError(
            f"evaluation source {name} is absent or ambiguous in the local layout"
        )
    return regular[0]


def _npz_metadata(path: Path) -> dict[str, object]:
    """Read the uint8 ``metadata_json`` member without importing NumPy."""

    try:
        with zipfile.ZipFile(path, "r") as archive:
            raw = archive.read("metadata_json.npy")
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise PostTrainingError("Stage2 replay shard lacks metadata_json.npy") from exc
    if len(raw) < 10 or raw[:6] != b"\x93NUMPY":
        raise PostTrainingError("Stage2 replay metadata has an invalid NPY header")
    version = tuple(raw[6:8])
    if version == (1, 0):
        header_size = struct.unpack("<H", raw[8:10])[0]
        header_start = 10
    elif version in ((2, 0), (3, 0)):
        if len(raw) < 12:
            raise PostTrainingError("Stage2 replay metadata NPY header is truncated")
        header_size = struct.unpack("<I", raw[8:12])[0]
        header_start = 12
    else:
        raise PostTrainingError("Stage2 replay metadata NPY version is unsupported")
    header_end = header_start + header_size
    try:
        header = ast.literal_eval(raw[header_start:header_end].decode("latin1").strip())
    except (SyntaxError, ValueError, UnicodeDecodeError) as exc:
        raise PostTrainingError("Stage2 replay metadata NPY header is invalid") from exc
    if not isinstance(header, dict):
        raise PostTrainingError("Stage2 replay metadata NPY header is not an object")
    shape = header.get("shape")
    if (
        header.get("descr") not in ("|u1", "<u1")
        or header.get("fortran_order") is not False
        or not isinstance(shape, tuple)
        or len(shape) != 1
        or type(shape[0]) is not int
        or shape[0] < 2
        or header_end + shape[0] != len(raw)
    ):
        raise PostTrainingError("Stage2 replay metadata is not a canonical uint8 vector")
    try:
        value = json.loads(raw[header_end:].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PostTrainingError("Stage2 replay metadata JSON is invalid") from exc
    if not isinstance(value, dict):
        raise PostTrainingError("Stage2 replay metadata must contain an object")
    return value


def _source_claim_group(
    value: object, *, expected: set[str], name: str
) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise PostTrainingError(f"{name} implementation contract differs")
    return {
        claim: _sha256(value.get(claim), f"{name} {claim}")
        for claim in sorted(expected)
    }


def _package_source_claims(
    manifest_path: Path,
    manifest_sha256: str,
    *,
    run_root: Path,
    run_id: str,
    name: str,
    expected_schema: str,
    expected_claims: set[str],
) -> dict[str, str]:
    package_root: Path | None = None
    marker_path: Path | None = None
    marker: dict[str, object] | None = None
    for candidate in (manifest_path.parent, *manifest_path.parents):
        try:
            candidate.relative_to(run_root)
        except ValueError:
            break
        possible = candidate / "COMPLETED.json"
        if not possible.is_file() or possible.is_symlink():
            continue
        possible_value = _load_json(possible, f"{name} completion")
        if possible_value.get("schema_name") == expected_schema:
            package_root = candidate
            marker_path = _regular_file(possible, run_root, f"{name} completion")
            marker = possible_value
            break
    if package_root is None or marker_path is None or marker is None:
        raise PostTrainingError(f"{name} completed provenance package is absent")
    if marker.get("schema_version") != 1 or marker.get("status") != "completed":
        raise PostTrainingError(f"{name} completion schema/status differs")
    if expected_schema == "navigation_rlt_prepare_training":
        if marker.get("artifact_sha256") != manifest_sha256:
            raise PostTrainingError(f"{name} completion manifest binding differs")
        receipt_name = "prepare_receipt.json"
        receipt_sha_value = marker.get("receipt_sha256")
    elif expected_schema == "navigation_rlt_replay_promotion":
        if marker.get("output_manifest_sha256") != manifest_sha256:
            raise PostTrainingError(f"{name} completion manifest binding differs")
        receipt_name = "promotion_receipt.json"
        receipt_sha_value = marker.get("promotion_receipt_sha256")
    else:
        raise PostTrainingError(f"{name} completion schema is unsupported")
    receipt = _regular_file(package_root / receipt_name, run_root, f"{name} receipt")
    _verify_file_sha(receipt, receipt_sha_value, f"{name} receipt")
    receipt_value = _load_json(receipt, f"{name} receipt")
    if (
        receipt_value.get("status") != "completed"
        or receipt_value.get("run_id") not in (None, run_id)
    ):
        raise PostTrainingError(f"{name} receipt status/run_id differs")
    return _source_claim_group(
        receipt_value.get("implementation"),
        expected=expected_claims,
        name=f"{name} receipt",
    )


def _stage2_successor_identity(
    formal_path: Path, *, expected_sha256: str, source_root: Path,
    run_root: Path, run_id: str, stage_run_root: Path, marker: Mapping[str, object],
    marker_ref: Mapping[str, object], training: Mapping[str, object],
    run_manifest: Mapping[str, object], run_manifest_ref: Mapping[str, object],
    result_ref: Mapping[str, object], verification_ref: Mapping[str, object],
    checkpoint_ref: Mapping[str, object], replay_ref: Mapping[str, object],
    support: Mapping[str, object],
) -> dict[str, object]:
    """Bind a create-only KL successor without relabelling its base producers."""

    formal_sha = _verify_file_sha(formal_path, expected_sha256, "Stage2 formal completion")
    formal = _load_json(formal_path, "Stage2 formal completion")
    if (formal.get("schema_name") != STAGE2_FORMAL_SCHEMA
            or formal.get("schema_version") != 1 or formal.get("status") != "completed"
            or formal.get("run_id") != run_id or formal.get("formal_id") != stage_run_root.name
            or formal.get("epochs") != 10 or formal.get("world_size") != 8
            or formal.get("unique_gpu_uuid_count") != 8
            or formal.get("fsdp_wrapper") != "FullyShardedDataParallel"
            or formal.get("training_scope_only") is not True
            or formal.get("validation_split_reads") != 0
            or type(formal.get("validation_split_reads")) is not int
            or formal.get("collapsed_epoch_streak") != 0
            or formal.get("collapse_threshold") != 0.98
            or formal.get("legacy_canonical_archive_compatible") is not False
            or formal.get("downstream_archive_contract") != "successor_aware_archive_required_v1"):
        raise PostTrainingError("Stage2 formal completion contract differs")
    for field, expected in (("stage_marker", marker_ref), ("training_result", result_ref),
                            ("formal_run_manifest", run_manifest_ref),
                            ("checkpoint", checkpoint_ref), ("a800_verification", verification_ref)):
        if formal.get(field) != expected:
            raise PostTrainingError(f"Stage2 formal {field} binding differs")
    for field in ("independent_restore", "component_parameter_sha256", "action_probability_epochs",
                  "optimizer_steps_per_rank", "collapsed_epoch_streak", "fsdp_wrapper"):
        if formal.get(field) != training.get(field):
            raise PostTrainingError(f"Stage2 formal/training {field} differs")
    if Path(str(checkpoint_ref["path"])) != stage_run_root / "navigation_rlt_stage2/training/checkpoints/latest/navigation_rlt_policy.pt":
        raise PostTrainingError("Stage2 successor checkpoint escapes its fixed output")
    if Path(str(run_manifest_ref["path"])) != stage_run_root / "prepare/run_manifest.json":
        raise PostTrainingError("Stage2 successor derived manifest path differs")
    boundaries = run_manifest.get("boundaries")
    if (not isinstance(boundaries, Mapping) or boundaries.get("split_label") != "train_update"
            or boundaries.get("upstream_split_read") != "train"
            or boundaries.get("validation_split_reads") != 0
            or type(boundaries.get("validation_split_reads")) is not int
            or boundaries.get("lightnav_frozen") is not True or boundaries.get("ttt_enabled") is not False):
        raise PostTrainingError("Stage2 successor train-only boundaries differ")
    epochs = formal.get("action_probability_epochs")
    if not isinstance(epochs, list) or len(epochs) != 10:
        raise PostTrainingError("Stage2 successor epoch evidence differs")
    for epoch in epochs:
        values = [epoch.get(name.lower()+"_probability_mean") for name in STAGE2_GATE_NAMES] if isinstance(epoch, Mapping) else []
        if (len(values) != 4 or any(isinstance(v, bool) or not isinstance(v, (float, int))
                                    or not math.isfinite(v) or v < 0 or v >= .98 for v in values)
                or values[3] != 0 or not math.isclose(sum(values), 1.0, abs_tol=1e-6)):
            raise PostTrainingError("Stage2 successor epoch collapsed or invalid")
    regularization = formal.get("actor_regularization")
    counts = support["gate_counts"]
    total = sum(counts)
    mask = support["gate_support_mask"]
    expected_prior = [.95 * count / total + (.05 / sum(mask) if enabled else 0)
                      for count, enabled in zip(counts, mask, strict=True)]
    prior = regularization.get("behavior_gate_prior") if isinstance(regularization, Mapping) else None
    if (not isinstance(regularization, Mapping)
            or set(regularization) != {"objective_schema", "direction", "behavior_kl_weight",
                                      "behavior_gate_prior", "gate_counts", "supported_uniform_mix"}
            or regularization.get("objective_schema") != STAGE2_BEHAVIOR_KL_OBJECTIVE_SCHEMA
            or type(regularization.get("behavior_kl_weight")) not in (int, float)
            or regularization.get("behavior_kl_weight") != 1.0
            or regularization.get("direction") != "behavior_to_actor_forward_kl"
            or regularization.get("gate_counts") != counts
            or regularization.get("supported_uniform_mix") != .05
            or not isinstance(prior, list) or len(prior) != 4
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                   or not math.isclose(v, expected, rel_tol=0, abs_tol=1e-6)
                   for v, expected in zip(prior, expected_prior, strict=True))):
        raise PostTrainingError("Stage2 successor behavior prior/KL metadata differs")
    metadata = training["final_checkpoint"]["metadata"]
    if (metadata["extra"].get("stage2_actor_regularization") != regularization
            or training["independent_restore"].get("stage2_actor_regularization") != regularization
            or metadata.get("stage2_behavior_prior_uniform_mix") != .05):
        raise PostTrainingError("Stage2 successor checkpoint/restore KL metadata differs")
    def checked_ref(value: object, name: str) -> dict[str, object]:
        if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
            raise PostTrainingError(f"{name} reference differs")
        path = _regular_file(Path(str(value["path"])), run_root, name)
        return {"path": str(path), "sha256": _verify_file_sha(path, value["sha256"], name)}
    base_ref = checked_ref(formal.get("base_run_manifest"), "Stage2 base manifest")
    base = _load_json(Path(str(base_ref["path"])), "Stage2 base manifest")
    if Path(str(base_ref["path"])) != run_root / "navigation_rlt_stage2_prepare/run_manifest.json":
        raise PostTrainingError("Stage2 base manifest path differs")
    _stage2_training_additions(base.get("training"), name="Stage2 base training")
    if base["training"].get("objective_schema") != STAGE2_REPAIR_OBJECTIVE_SCHEMA:
        raise PostTrainingError("Stage2 successor base objective differs")
    expected_derived = json.loads(json.dumps(base))
    expected_derived["training"].update(objective_schema=STAGE2_BEHAVIOR_KL_OBJECTIVE_SCHEMA, behavior_kl_weight=1.0)
    implementation = _source_claim_group(run_manifest.get("implementation"),
        expected={"trainer_sha256", "policy_sha256", "replay_sha256"}, name="successor training")
    for field in ("trainer_sha256", "policy_sha256"):
        expected_derived["implementation"][field] = implementation[field]
    if expected_derived != run_manifest or base["training"].get("epochs") != 10:
        raise PostTrainingError("Stage2 successor changes more than the registered objective/source fields")
    prepare_root = stage_run_root / "prepare"
    derivation_path = _regular_file(prepare_root / "DERIVATION.json", run_root, "Stage2 formal derivation")
    prepare_path = _regular_file(prepare_root / "COMPLETED.json", run_root, "Stage2 formal preparation completion")
    derivation = _load_json(derivation_path, "Stage2 formal derivation")
    prepared = _load_json(prepare_path, "Stage2 formal preparation completion")
    derivation_sha = sha256_file(derivation_path)
    for value in (derivation, prepared):
        if (value.get("schema_name") != STAGE2_DERIVATION_SCHEMA or value.get("schema_version") != 1
                or value.get("status") != "completed" or type(value.get("validation_split_reads")) is not int
                or value.get("validation_split_reads") != 0):
            raise PostTrainingError("Stage2 formal derivation/completion schema differs")
    if (prepared.get("run_manifest_sha256") != run_manifest_ref["sha256"]
            or prepared.get("derivation_sha256") != derivation_sha
            or derivation.get("base_run_manifest") != base_ref
            or derivation.get("derived_run_manifest") != run_manifest_ref
            or derivation.get("implementation") != implementation
            or derivation.get("base_epochs") != 10 or derivation.get("formal_epochs") != 10
            or derivation.get("training_changed_paths") != ["training.objective_schema", "training.behavior_kl_weight"]
            or derivation.get("implementation_changed_paths") != ["implementation.policy_sha256", "implementation.trainer_sha256"]
            or derivation.get("anti_collapse_commit") != formal.get("anti_collapse_commit")):
        raise PostTrainingError("Stage2 formal derivation binding differs")
    _sha1(formal.get("anti_collapse_commit"), "historical anti-collapse initialization commit")
    for field in ("smoke_completion", "turbo_guard"):
        if checked_ref(formal.get(field), f"Stage2 formal {field}") != derivation.get(field):
            raise PostTrainingError(f"Stage2 formal derivation {field} differs")
    telemetry_ref = checked_ref(formal.get("telemetry_summary"), "Stage2 formal telemetry")
    if telemetry_ref["path"] != marker["output"].get("telemetry_summary"):
        raise PostTrainingError("Stage2 formal telemetry path differs")
    source_root = _regular_directory(source_root, stage_run_root, "Stage2 successor source root")
    claims = {**implementation, **_source_claim_group(marker.get("implementation"),
        expected={"wrapper_sha256", "gpu_monitor_sha256"}, name="successor marker")}
    names = {"trainer_sha256":"navigation_rlt_fsdp_train.py", "policy_sha256":"navigation_rlt_policy.py",
             "replay_sha256":"navigation_rlt_replay.py", "wrapper_sha256":"run_navigation_rlt_a800_stage2_repair.sh",
             "gpu_monitor_sha256":"lightnav_a3_gpu_monitor.py"}
    source_files = {}
    for claim, name in names.items():
        source = _regular_file(source_root / name, source_root, f"Stage2 successor {name}")
        if os.name != "nt" and source.stat().st_mode & 0o222:
            raise PostTrainingError("Stage2 successor source must be immutable")
        source_files[claim] = {"path": str(source), "sha256": _verify_file_sha(source, claims[claim], name)}
    base_claims = _package_source_claims(Path(str(base_ref["path"])), str(base_ref["sha256"]),
        run_root=run_root, run_id=run_id, name="Stage2 base preparation",
        expected_schema="navigation_rlt_prepare_training",
        expected_claims={"trainer_sha256", "policy_sha256", "replay_sha256", "preparer_sha256"})
    replay_claims = _package_source_claims(Path(str(replay_ref["path"])), str(replay_ref["sha256"]),
        run_root=run_root, run_id=run_id, name="Stage2 replay producer",
        expected_schema="navigation_rlt_replay_promotion",
        expected_claims={"promoter_sha256", "policy_sha256", "replay_sha256"})
    if any(base_claims[field] != base["implementation"].get(field) for field in implementation):
        raise PostTrainingError("Stage2 base preparation source binding differs")
    result = {
        "source_root": str(source_root), "source_claims": claims, "source_files": source_files,
        "formal_completion": {"path": str(formal_path), "sha256": formal_sha},
        "derivation": {"path": str(derivation_path), "sha256": derivation_sha},
        "prepare_completion": {"path": str(prepare_path), "sha256": sha256_file(prepare_path)},
        "base_run_manifest": base_ref, "base_preparation_claims": base_claims,
        "replay_producer_claims": replay_claims,
    }
    for key, path in (("base_prepare_receipt", Path(str(base_ref["path"])).parent / "prepare_receipt.json"),
                      ("base_prepare_completion", Path(str(base_ref["path"])).parent / "COMPLETED.json"),
                      ("promotion_receipt", Path(str(replay_ref["path"])).parent / "promotion_receipt.json"),
                      ("promotion_completion", Path(str(replay_ref["path"])).parent / "COMPLETED.json")):
        path = _regular_file(path, run_root, key)
        result[key] = {"path": str(path), "sha256": sha256_file(path)}
        result[key+"_value"] = _load_json(path, key)
    return result


def _stage2_evidence(
    marker_path: Path, *, expected_marker_sha256: str, run_root: Path, run_id: str,
    formal_completion_path: Path | None = None,
    expected_formal_completion_sha256: str | None = None,
    source_root: Path | None = None,
) -> dict[str, object]:
    successor_args = (formal_completion_path, expected_formal_completion_sha256, source_root)
    successor = all(value is not None for value in successor_args)
    if any(value is not None for value in successor_args) and not successor:
        raise PostTrainingError("Stage2 successor arguments must be all present")
    stage_run_root = run_root
    if successor:
        formal_path = _regular_file(formal_completion_path, run_root, "Stage2 formal completion")
        stage_run_root = formal_path.parent
        if (stage_run_root.parent != run_root / "stage2-successors"
                or formal_path.name != "FORMAL_COMPLETED.json"
                or Path(marker_path).absolute() != stage_run_root / "navigation_rlt_stage2/STAGE.json"):
            raise PostTrainingError("Stage2 successor root/marker is not canonical")
    marker_path = _regular_file(marker_path, run_root, "Stage2 marker")
    marker_sha = _verify_file_sha(
        marker_path, expected_marker_sha256, "Stage2 marker"
    )
    marker = _load_json(marker_path, "Stage2 marker")
    if (
        marker.get("schema_name") != STAGE_SCHEMA
        or marker.get("schema_version") != 1
        or marker.get("status") != "completed"
        or marker.get("phase") != "stage2"
        or marker.get("exit_code") != 0
        or marker.get("run_root") != str(stage_run_root)
    ):
        raise PostTrainingError("Stage2 marker schema/status/root differs")
    output = marker.get("output")
    verification_ref = marker.get("verification")
    if not isinstance(output, Mapping) or not isinstance(verification_ref, Mapping):
        raise PostTrainingError("Stage2 marker lacks output or verification")
    checkpoint_ref = output.get("checkpoint")
    if not isinstance(checkpoint_ref, Mapping):
        raise PostTrainingError("Stage2 marker lacks its checkpoint")
    checkpoint = _regular_file(
        Path(str(checkpoint_ref.get("path"))), run_root, "Stage2 checkpoint"
    )
    checkpoint_sha = _verify_file_sha(
        checkpoint, checkpoint_ref.get("sha256"), "Stage2 checkpoint"
    )
    verification = _regular_file(
        Path(str(verification_ref.get("path"))), run_root, "Stage2 verification"
    )
    verification_sha = _verify_file_sha(
        verification, verification_ref.get("sha256"), "Stage2 verification"
    )
    verification_value = _load_json(verification, "Stage2 verification")
    if (
        verification_value.get("schema_name") != VERIFICATION_SCHEMA
        or verification_value.get("schema_version") != 1
        or verification_value.get("status") != "passed"
        or verification_value.get("phase") != "stage2"
        or verification_value.get("world_size") != 8
        or verification_value.get("unique_gpu_uuid_count") != 8
    ):
        raise PostTrainingError("Stage2 verification schema/status/GPU evidence differs")
    verified_checkpoint = verification_value.get("checkpoint")
    training_result_ref = verification_value.get("training_result")
    if (
        not isinstance(verified_checkpoint, Mapping)
        or verified_checkpoint.get("path") != str(checkpoint)
        or verified_checkpoint.get("sha256") != checkpoint_sha
        or not isinstance(training_result_ref, Mapping)
    ):
        raise PostTrainingError("Stage2 marker and verification checkpoint differ")
    training_result = _regular_file(
        Path(str(training_result_ref.get("path"))), run_root, "Stage2 training result"
    )
    training_result_sha = _verify_file_sha(
        training_result,
        training_result_ref.get("sha256"),
        "Stage2 training result",
    )
    training = _load_json(training_result, "Stage2 training result")
    final_checkpoint = training.get("final_checkpoint")
    restore = training.get("independent_restore")
    training_run_ref = training.get("run_manifest")
    training_replay_ref = training.get("replay_manifest")
    checkpoint_metadata = (
        final_checkpoint.get("metadata")
        if isinstance(final_checkpoint, Mapping)
        else None
    )
    checkpoint_extra = (
        checkpoint_metadata.get("extra")
        if isinstance(checkpoint_metadata, Mapping)
        else None
    )
    if (
        training.get("schema_name") != TRAINING_RESULT_SCHEMA
        or training.get("schema_version") != 1
        or training.get("status") != "completed"
        or training.get("phase") != "stage2"
        or training.get("run_id") != run_id
        or not isinstance(final_checkpoint, Mapping)
        or final_checkpoint.get("path") != str(checkpoint)
        or final_checkpoint.get("sha256") != checkpoint_sha
        or not isinstance(checkpoint_metadata, Mapping)
        or checkpoint_metadata.get("stage1_frozen") is not True
        or not isinstance(checkpoint_extra, Mapping)
        or checkpoint_extra.get("training_phase") != "stage2"
        or not isinstance(training_run_ref, Mapping)
        or not isinstance(training_replay_ref, Mapping)
        or not isinstance(restore, Mapping)
        or restore.get("status") != "passed"
        or restore.get("checkpoint_sha256") != checkpoint_sha
    ):
        raise PostTrainingError(
            "Stage2 training result/checkpoint/independent restore differs"
        )
    training_run_path = _regular_file(
        Path(str(training_run_ref.get("path"))), run_root, "Stage2 training run manifest"
    )
    training_run_sha = _verify_file_sha(
        training_run_path,
        training_run_ref.get("sha256"),
        "Stage2 training run manifest",
    )
    training_run = _load_json(training_run_path, "Stage2 training run manifest")
    if (
        training_run.get("schema_name") != "navigation_rlt_training_run_manifest"
        or training_run.get("schema_version") != 1
        or training_run.get("run_id") != run_id
        or training_run.get("phase") != "stage2"
        or checkpoint_extra.get("run_manifest_sha256") != training_run_sha
    ):
        raise PostTrainingError("Stage2 checkpoint training-run binding differs")
    replay_ref = marker.get("replay_manifest")
    if not isinstance(replay_ref, Mapping):
        raise PostTrainingError("Stage2 marker lacks its replay manifest")
    replay_path = _regular_file(
        Path(str(replay_ref.get("path"))), run_root, "Stage2 replay manifest"
    )
    replay_sha = _verify_file_sha(
        replay_path, replay_ref.get("sha256"), "Stage2 replay manifest"
    )
    replay = _load_json(replay_path, "Stage2 replay manifest")
    shards = replay.get("shards")
    training_run_replay = training_run.get("replay")
    if (
        replay.get("schema_name") != "navigation_rlt_replay_manifest"
        or replay.get("schema_version") != 3
        or replay.get("run_id") != run_id
        or replay.get("representation_kind") != "frozen_z_rl"
        or replay.get("split_label") != "train_update"
        or type(replay.get("validation_split_reads")) is not int
        or replay.get("validation_split_reads") != 0
        or replay.get("expected_episode_count") != 256
        or replay.get("completed_episode_count") != 256
        or not isinstance(shards, list)
        or len(shards) != 256
    ):
        raise PostTrainingError("Stage2 replay manifest fixed contract differs")
    if (
        training_replay_ref.get("path") != str(replay_path)
        or training_replay_ref.get("sha256") != replay_sha
        or checkpoint_extra.get("replay_manifest_sha256") != replay_sha
        or not isinstance(training_run_replay, Mapping)
        or training_run_replay.get("manifest_sha256") != replay_sha
    ):
        raise PostTrainingError("Stage2 checkpoint replay binding differs")
    first = shards[0]
    if not isinstance(first, Mapping):
        raise PostTrainingError("Stage2 replay manifest first shard is invalid")
    shard = _regular_file(
        Path(str(first.get("path"))), run_root, "Stage2 replay shard"
    )
    _verify_file_sha(shard, first.get("sha256"), "Stage2 replay shard")
    metadata = _npz_metadata(shard)
    context_source_sha = _sha256(
        metadata.get("prefix_feature_source_sha256"),
        "Stage2 replay prefix feature source SHA-256",
    )
    if (
        metadata.get("source_fingerprint") != replay.get("source_fingerprint")
        or metadata.get("representation_kind") != "frozen_z_rl"
    ):
        raise PostTrainingError("Stage2 replay shard source identity differs")
    run_training = training_run.get("training")
    result_training = training.get("training")
    run_support = training_run.get("stage2_support")
    result_support = training.get("stage2_support")
    repair_claimed = any(
        (
            isinstance(candidate, Mapping)
            and candidate.get("objective_schema")
            in (STAGE2_REPAIR_OBJECTIVE_SCHEMA, STAGE2_BEHAVIOR_KL_OBJECTIVE_SCHEMA)
        )
        for candidate in (run_training, result_training)
    ) or run_support is not None or result_support is not None
    repair_contract: dict[str, object] | None = None
    if repair_claimed:
        if (
            not isinstance(run_training, Mapping)
            or not isinstance(result_training, Mapping)
            or dict(run_training) != dict(result_training)
        ):
            raise PostTrainingError("Stage2 repair training parameters differ")
        additions = _stage2_training_additions(
            run_training, name="Stage2 repair training parameters"
        )
        support = _stage2_support_contract(
            run_support, name="Stage2 repair run support"
        )
        if result_support != support:
            raise PostTrainingError("Stage2 repair result support differs")
        if (
            additions["minimum_behavior_gate_count"]
            != support["minimum_behavior_gate_count"]
        ):
            raise PostTrainingError("Stage2 repair train-only/support contract differs")
        stage1_checkpoint_input = training.get("stage1_checkpoint_input")
        stage1_run_checkpoint = training_run.get("stage1_checkpoint")
        if not isinstance(stage1_checkpoint_input, Mapping) or not isinstance(
            stage1_run_checkpoint, Mapping
        ):
            raise PostTrainingError("Stage2 repair lacks its Stage1 checkpoint input")
        stage1_checkpoint_path = _regular_file(
            Path(str(stage1_checkpoint_input.get("path"))),
            run_root,
            "Stage1 checkpoint input",
        )
        stage1_checkpoint_sha = _verify_file_sha(
            stage1_checkpoint_path,
            stage1_checkpoint_input.get("sha256"),
            "Stage1 checkpoint input",
        )
        if stage1_run_checkpoint.get("sha256") != stage1_checkpoint_sha:
            raise PostTrainingError("Stage2 repair Stage1 checkpoint binding differs")
        ranks = training.get("rank_receipts")
        if (
            training.get("world_size") != 8
            or training.get("unique_gpu_uuid_count") != 8
            or not isinstance(ranks, list)
            or len(ranks) != 8
        ):
            raise PostTrainingError("Stage2 repair rank evidence differs")
        for receipt in ranks:
            data = receipt.get("data") if isinstance(receipt, Mapping) else None
            if not isinstance(data, Mapping) or any(
                data.get(field) != support[field]
                for field in (
                    "eligibility_schema",
                    "gate_names",
                    "gate_counts",
                    "gate_support_mask",
                    "minimum_behavior_gate_count",
                )
            ):
                raise PostTrainingError("Stage2 repair rank support evidence differs")
        expected_supported_names = [
            name
            for name, enabled in zip(
                STAGE2_GATE_NAMES, support["gate_support_mask"], strict=True
            )
            if enabled
        ]
        if (
            checkpoint_metadata.get("stage2_objective")
            != (
                "reference_conditioned_min_target_twin_q_with_finite_gate_cql_"
                "and_native_reference_actor_anchor_dropout"
            )
            or checkpoint_metadata.get("cql_enabled") is not True
            or checkpoint_metadata.get("cql_candidate_count")
            != len(STAGE2_GATE_NAMES)
            or checkpoint_metadata.get("entropy_target_enabled") is not False
            or checkpoint_metadata.get("gate_support_mask")
            != support["gate_support_mask"]
            or checkpoint_metadata.get("supported_gate_names")
            != expected_supported_names
        ):
            raise PostTrainingError("Stage2 checkpoint repair metadata differs")
        successor_identity = None
        if successor:
            if additions["objective_schema"] != STAGE2_BEHAVIOR_KL_OBJECTIVE_SCHEMA:
                raise PostTrainingError("Stage2 successor requires the registered KL-v2 objective")
            successor_identity = _stage2_successor_identity(
                formal_path, expected_sha256=expected_formal_completion_sha256,
                source_root=Path(source_root), run_root=run_root, run_id=run_id,
                stage_run_root=stage_run_root, marker=marker,
                marker_ref={"path": str(marker_path), "sha256": marker_sha},
                training=training, run_manifest=training_run,
                run_manifest_ref={"path": str(training_run_path), "sha256": training_run_sha},
                result_ref={"path": str(training_result), "sha256": training_result_sha},
                verification_ref={"path": str(verification), "sha256": verification_sha},
                checkpoint_ref={"path": str(checkpoint), "sha256": checkpoint_sha},
                replay_ref={"path": str(replay_path), "sha256": replay_sha}, support=support,
            )
            source_claims = successor_identity["source_claims"]
        else:
            if additions["objective_schema"] != STAGE2_REPAIR_OBJECTIVE_SCHEMA:
                raise PostTrainingError("KL-v2 requires explicit successor completion provenance")
            claim_groups = (
                _source_claim_group(
                    training_run.get("implementation"),
                    expected={"trainer_sha256", "policy_sha256", "replay_sha256"},
                    name="Stage2 run manifest",
                ),
                _source_claim_group(
                    marker.get("implementation"),
                    expected={"wrapper_sha256", "gpu_monitor_sha256"},
                    name="Stage2 marker",
                ),
                _package_source_claims(
                    training_run_path,
                    training_run_sha,
                    run_root=run_root,
                    run_id=run_id,
                    name="Stage2 run manifest package",
                    expected_schema="navigation_rlt_prepare_training",
                    expected_claims={
                        "trainer_sha256",
                        "policy_sha256",
                        "replay_sha256",
                        "preparer_sha256",
                    },
                ),
                _package_source_claims(
                    replay_path,
                    replay_sha,
                    run_root=run_root,
                    run_id=run_id,
                    name="Stage2 replay promotion package",
                    expected_schema="navigation_rlt_replay_promotion",
                    expected_claims={
                        "promoter_sha256",
                        "policy_sha256",
                        "replay_sha256",
                    },
                ),
            )
            source_claims: dict[str, str] = {}
            for group in claim_groups:
                for claim, digest in group.items():
                    previous = source_claims.setdefault(claim, digest)
                    if previous != digest:
                        raise PostTrainingError(
                            f"Stage2 source claim {claim} differs across provenance receipts"
                        )
            if set(source_claims) != STAGE2_REPAIR_REQUIRED_SOURCE_CLAIMS:
                raise PostTrainingError("Stage2 repair source claim coverage differs")
        repair_contract = {
            "training_additions": additions,
            "stage2_support": support,
            "stage1_checkpoint_sha256": stage1_checkpoint_sha,
            "validation_split_reads": 0,
            "source_claims": source_claims,
        }
    evidence = {
        "marker": {"path": str(marker_path), "sha256": marker_sha},
        "verification": {"path": str(verification), "sha256": verification_sha},
        "training_result": {
            "path": str(training_result),
            "sha256": training_result_sha,
        },
        "training_run_manifest": {
            "path": str(training_run_path),
            "sha256": training_run_sha,
        },
        "replay_manifest": {"path": str(replay_path), "sha256": replay_sha},
        "context_source_sha256": context_source_sha,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_sha,
            "size_bytes": checkpoint.stat().st_size,
        },
    }
    if successor and repair_contract is None:
        raise PostTrainingError("Stage2 successor lacks the repair objective")
    if repair_contract is not None:
        evidence["repair_contract"] = repair_contract
    if successor:
        evidence.update({
            "stage_run_root": str(stage_run_root),
            "formal_completion": {"path": str(formal_path), "sha256": str(expected_formal_completion_sha256)},
            "formal_id": stage_run_root.name,
            "successor_source_identity": successor_identity,
        })
    return evidence


def _collection_evidence(
    path: Path, *, expected_sha256: str, run_root: Path, run_id: str
) -> tuple[dict[str, object], dict[str, object]]:
    path = _regular_file(path, run_root, "collection outer manifest")
    digest = _verify_file_sha(path, expected_sha256, "collection outer manifest")
    value = _load_json(path, "collection outer manifest")
    if (
        value.get("schema_name") != COLLECTION_SCHEMA
        or value.get("schema_version") != 1
        or value.get("run_id") != run_id
    ):
        raise PostTrainingError("collection outer manifest schema/run_id differs")
    boundaries = value.get("boundaries")
    protocol = value.get("protocol")
    code = value.get("code")
    if (
        not isinstance(boundaries, Mapping)
        or boundaries.get("upstream_split_read") != "train"
        or boundaries.get("validation_split_reads") != 0
        or boundaries.get("lightnav_frozen") is not True
        or boundaries.get("weight_origin") != "original_release"
        or boundaries.get("ttt_or_secondary_post_training") is not False
        or not isinstance(protocol, Mapping)
        or protocol.get("max_physical_steps") != MAX_STEPS
        or protocol.get("matched_eval_arms") != list(ARMS)
        or not isinstance(code, Mapping)
    ):
        raise PostTrainingError("collection train-only/evaluation boundary differs")
    _sha1(code.get("git_commit"), "collection code commit")
    lightnav = value.get("lightnav")
    if not isinstance(lightnav, Mapping):
        raise PostTrainingError("collection LightNav identity is absent")
    return value, {"path": str(path), "sha256": digest}


def _evaluation_plan_evidence(
    plan_path: Path,
    *,
    expected_plan_sha256: str,
    split_path: Path,
    expected_split_sha256: str,
    run_root: Path,
    run_id: str,
) -> tuple[dict[str, object], dict[str, object], dict[str, object], str]:
    split_path = _regular_file(split_path, run_root, "train split manifest")
    split_sha = _verify_file_sha(split_path, expected_split_sha256, "train split manifest")
    split = _load_json(split_path, "train split manifest")
    development = (
        split.get("splits", {}).get("train_development")
        if isinstance(split.get("splits"), Mapping)
        else None
    )
    if (
        split.get("upstream_split_read") != "train"
        or split.get("validation_split_reads") != 0
        or not isinstance(development, Mapping)
        or development.get("episode_count") != 32
        or not isinstance(development.get("episodes"), list)
        or len(development["episodes"]) != 32
    ):
        raise PostTrainingError("split manifest is not the fixed 32-episode train development set")
    _sha256(development.get("dataset_sha256"), "development dataset SHA-256")
    _sha256(development.get("gt_sha256"), "development ground-truth SHA-256")

    plan_path = _regular_file(plan_path, run_root, "evaluation plan")
    plan_sha = _verify_file_sha(plan_path, expected_plan_sha256, "evaluation plan")
    plan = _load_json(plan_path, "evaluation plan")
    split_ref = plan.get("split_manifest")
    workers = plan.get("workers")
    if (
        plan.get("schema_name") != EVAL_PLAN_SCHEMA
        or plan.get("schema_version") != 1
        or plan.get("run_id") != run_id
        or plan.get("split_label") != "train_development"
        or plan.get("validation_split_reads") != 0
        or plan.get("episode_count") != 32
        or not isinstance(split_ref, Mapping)
        or split_ref.get("sha256") != split_sha
        or not isinstance(workers, list)
        or plan.get("worker_count") != len(workers)
        or len(workers) != 1
        or not isinstance(workers[0], Mapping)
        or workers[0].get("worker_id") != 0
    ):
        raise PostTrainingError("evaluation plan fixed contract differs")
    rows: list[Mapping[str, object]] = []
    for worker in workers:
        if not isinstance(worker, Mapping) or not isinstance(worker.get("episodes"), list):
            raise PostTrainingError("evaluation plan worker inventory is invalid")
        rows.extend(worker["episodes"])
    rows.sort(key=lambda row: int(row.get("episode_order", -1)))
    identities: list[list[object]] = []
    for order, row in enumerate(rows):
        if row.get("episode_order") != order:
            raise PostTrainingError("evaluation plan episode order differs from 0..31")
        episode_id = row.get("episode_id")
        scene = row.get("scene")
        instruction_sha = _sha256(
            row.get("instruction_text_sha256"), "planned instruction SHA-256"
        )
        if not isinstance(episode_id, str) or not episode_id or not isinstance(scene, str) or not scene:
            raise PostTrainingError("evaluation plan episode identity is invalid")
        identities.append([order, scene, episode_id, instruction_sha])
    if len(identities) != 32 or len({tuple(row) for row in identities}) != 32:
        raise PostTrainingError("evaluation plan does not contain 32 unique identities")
    split_identities = [
        [
            index,
            row.get("scene"),
            str(row.get("episode_id")),
            row.get("instruction_text_sha256"),
        ]
        for index, row in enumerate(development["episodes"])
        if isinstance(row, Mapping)
    ]
    if identities != split_identities:
        raise PostTrainingError("evaluation plan episode order differs from the fixed split")
    return (
        plan,
        {"path": str(plan_path), "sha256": plan_sha},
        {"path": str(split_path), "sha256": split_sha},
        _object_sha256(identities),
    )


def _evaluation_inventory_evidence(
    path: Path, *, expected_sha256: str, source_root: Path | None = None
) -> tuple[dict[str, object], dict[str, object]]:
    raw = Path(path).expanduser()
    if raw.is_symlink():
        raise PostTrainingError("evaluation code inventory must not be a link")
    path = raw.resolve(strict=True)
    if not path.is_file():
        raise PostTrainingError("evaluation code inventory must be a regular file")
    digest = _verify_file_sha(path, expected_sha256, "evaluation code inventory")
    value = _load_json(path, "evaluation code inventory")
    files = value.get("files")
    if (
        set(value) != {"schema_name", "schema_version", "navigation_code_commit", "files"}
        or value.get("schema_name") != EVAL_INVENTORY_SCHEMA
        or value.get("schema_version") != 1
        or not isinstance(files, Mapping)
        or set(files) != REQUIRED_EVAL_FILES
    ):
        raise PostTrainingError("evaluation code inventory schema/files differ")
    commit = _sha1(value.get("navigation_code_commit"), "evaluation code commit")
    normalized: dict[str, str] = {}
    for name, raw_sha in files.items():
        if not isinstance(name, str):
            raise PostTrainingError("evaluation code inventory filename is invalid")
        pure = PurePosixPath(name)
        if (
            not pure.parts
            or pure.is_absolute()
            or pure.as_posix() != name
            or any(part in ("", ".", "..") for part in pure.parts)
        ):
            raise PostTrainingError("evaluation code inventory path is not canonical")
        source = _local_evaluation_source(name, source_root)
        normalized[name] = _verify_file_sha(
            source, raw_sha, f"evaluation source {name}"
        )
    return (
        {
            "commit": commit,
            "inventory_sha256": digest,
            "files": normalized,
            **({} if source_root is None else {"source_root": str(Path(source_root).resolve(strict=True))}),
        },
        {"path": str(path), "sha256": digest},
    )


def _checked_provenance_file(
    path: Path,
    expected_sha256: str,
    *,
    run_root: Path,
    name: str,
    schema: str,
    run_id: str | None,
) -> dict[str, object]:
    path = _regular_file(path, run_root, name)
    digest = _verify_file_sha(path, expected_sha256, name)
    value = _load_json(path, name)
    if value.get("schema_name") != schema or value.get("schema_version") != 1:
        raise PostTrainingError(f"{name} schema differs")
    if run_id is not None and value.get("run_id") != run_id:
        raise PostTrainingError(f"{name} run_id differs")
    return {"path": str(path), "sha256": digest}


def _code_verification_identity(
    path: Path,
    expected_sha256: str,
    *,
    run_root: Path,
    name: str,
    expected_superseded_commit: str,
    expected_code_root: Path | None = None,
    forbidden_code_root: Path | None = None,
    required_source_files: Sequence[str] = (),
    expected_source_claims: Mapping[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    reference = _checked_provenance_file(
        path,
        expected_sha256,
        run_root=run_root,
        name=name,
        schema=TRAINING_CODE_VERIFICATION_SCHEMA,
        run_id=None,
    )
    value = _load_json(Path(str(reference["path"])), name)
    if value.get("status") != "passed":
        raise PostTrainingError(f"{name} status differs")
    commit = _sha1(value.get("navigation_code_commit"), f"{name} code commit")
    superseded = _sha1(
        value.get("supersedes_navigation_code_commit"),
        f"{name} superseded code commit",
    )
    if superseded != expected_superseded_commit:
        raise PostTrainingError(f"{name} superseded commit differs")
    code_root = _regular_directory(
        Path(str(value.get("code_root"))), run_root, f"{name} code root"
    )
    if expected_code_root is not None:
        required_root = _regular_directory(
            expected_code_root, run_root, f"expected {name} code root"
        )
        if code_root != required_root:
            raise PostTrainingError(f"{name} code root differs")
    if forbidden_code_root is not None:
        old_root = _regular_directory(
            forbidden_code_root, run_root, f"forbidden {name} code root"
        )
        roots_overlap = code_root == old_root
        for candidate, parent in ((code_root, old_root), (old_root, code_root)):
            try:
                candidate.relative_to(parent)
            except ValueError:
                continue
            roots_overlap = True
        if roots_overlap:
            raise PostTrainingError(f"{name} code root is not independent")
    inventory = _regular_file(
        Path(str(value.get("inventory_path"))), run_root, f"{name} code inventory"
    )
    inventory_sha = _verify_file_sha(
        inventory, value.get("inventory_sha256"), f"{name} code inventory"
    )
    inventory_value = _load_json(inventory, f"{name} code inventory")
    raw_files = inventory_value.get("files")
    if (
        inventory_value.get("schema_name") != TRAINING_INVENTORY_SCHEMA
        or inventory_value.get("schema_version") != 1
        or inventory_value.get("navigation_code_commit") != commit
        or not isinstance(raw_files, list)
        or inventory_value.get("file_count") != len(raw_files)
        or value.get("file_count") != len(raw_files)
    ):
        raise PostTrainingError(f"{name} code inventory identity/count differs")
    indexed: dict[str, dict[str, object]] = {}
    for raw in raw_files:
        if not isinstance(raw, Mapping):
            raise PostTrainingError(f"{name} code inventory row is invalid")
        raw_path = raw.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise PostTrainingError(f"{name} code inventory path is invalid")
        relative = PurePosixPath(raw_path)
        if (
            relative.is_absolute()
            or relative.as_posix() != raw_path
            or any(part in ("", ".", "..") for part in relative.parts)
            or raw_path in indexed
        ):
            raise PostTrainingError(f"{name} code inventory path is invalid")
        digest = _sha256(raw.get("sha256"), f"{name} source {raw_path} SHA-256")
        size = raw.get("size_bytes")
        if type(size) is not int or size < 0:
            raise PostTrainingError(f"{name} code inventory size is invalid")
        indexed[raw_path] = {"sha256": digest, "size_bytes": size}
    required_files = set(required_source_files)
    missing_files = required_files - set(indexed)
    if missing_files:
        raise PostTrainingError(
            f"{name} code inventory lacks required sources: {sorted(missing_files)}"
        )
    for relative in sorted(required_files):
        live_source = _regular_file(
            code_root / relative, code_root, f"{name} live source {relative}"
        )
        if (
            sha256_file(live_source) != indexed[relative]["sha256"]
            or live_source.stat().st_size != indexed[relative]["size_bytes"]
        ):
            raise PostTrainingError(f"{name} live source differs from code inventory")
    if expected_source_claims is not None:
        if set(expected_source_claims) != STAGE2_REPAIR_REQUIRED_SOURCE_CLAIMS:
            raise PostTrainingError(f"{name} source claim coverage differs")
        for claim, expected_digest in expected_source_claims.items():
            relative = TRAINING_SOURCE_FILES_BY_CLAIM.get(claim)
            if (
                relative is None
                or indexed.get(relative, {}).get("sha256") != expected_digest
            ):
                raise PostTrainingError(
                    f"{name} source claim differs from code inventory"
                )
    code_archive = _regular_file(
        Path(str(value.get("archive_path"))), run_root, f"{name} code archive"
    )
    archive_sha = _verify_file_sha(
        code_archive, value.get("archive_sha256"), f"{name} code archive"
    )
    if (
        inventory_value.get("archive_name") != code_archive.name
        or inventory_value.get("archive_sha256") != archive_sha
        or inventory_value.get("archive_size_bytes") != code_archive.stat().st_size
    ):
        raise PostTrainingError(f"{name} code archive/inventory identity differs")
    return reference, {
        "code_commit": commit,
        "code_root": str(code_root),
        "code_verification_sha256": reference["sha256"],
        "code_inventory_sha256": inventory_sha,
        "code_archive_sha256": archive_sha,
    }


def _stage2_repair_lineage(
    *,
    args: argparse.Namespace,
    stage2: Mapping[str, object],
    collection: Mapping[str, object],
    run_root: Path,
    run_id: str,
    base_verification_ref: Mapping[str, object],
    base_protocol_ref: Mapping[str, object],
) -> dict[str, object] | None:
    names = (
        "stage2_repair_code_verification",
        "expected_stage2_repair_code_verification_sha256",
        "stage2_repair_protocol",
        "expected_stage2_repair_protocol_sha256",
    )
    values = tuple(getattr(args, name, None) for name in names)
    supplied = all(value is not None for value in values)
    if any(value is not None for value in values) and not supplied:
        raise PostTrainingError("Stage2 repair provenance arguments must be all present")
    actual = stage2.get("repair_contract")
    if (actual is not None) != supplied:
        raise PostTrainingError("Stage2 repair provenance does not match the objective")
    if not supplied:
        return None
    if not isinstance(actual, Mapping):
        raise PostTrainingError("Stage2 repair contract is absent")
    successor_identity = stage2.get("successor_source_identity")
    if successor_identity is not None:
        if not isinstance(successor_identity, Mapping):
            raise PostTrainingError("Stage2 successor source identity is absent")
        # The unchanged repair protocol describes the base preparation and replay
        # producer.  Verify it as that historical source, never as the new trainer.
        claims = dict(successor_identity["base_preparation_claims"])
        for key, value in successor_identity["replay_producer_claims"].items():
            if key in claims and claims[key] != value:
                raise PostTrainingError("Stage2 base/replay producer source claims differ")
            claims[key] = value
        repair_verification = _regular_file(Path(str(values[0])), run_root, "Stage2 repair code verification")
        _verify_file_sha(repair_verification, values[1], "Stage2 repair code verification")
        repair_value = _load_json(repair_verification, "Stage2 repair code verification")
        repair_root = _regular_directory(Path(str(repair_value.get("code_root"))), run_root, "Stage2 base code root")
        for key in ("wrapper_sha256", "gpu_monitor_sha256"):
            claims[key] = sha256_file(_regular_file(repair_root / TRAINING_SOURCE_FILES_BY_CLAIM[key], run_root, key))
        base_manifest_ref = successor_identity["base_run_manifest"]
        base_manifest = _load_json(Path(str(base_manifest_ref["path"])), "Stage2 base manifest")
        base_stage2 = {
            **stage2,
            "training_run_manifest": base_manifest_ref,
            "repair_contract": {
                **actual,
                "training_additions": _stage2_training_additions(base_manifest["training"], name="Stage2 base additions"),
                "source_claims": claims,
            },
        }
        base_stage2.pop("successor_source_identity", None)
        lineage = _stage2_repair_lineage(
            args=args, stage2=base_stage2, collection=collection, run_root=run_root,
            run_id=run_id, base_verification_ref=base_verification_ref,
            base_protocol_ref=base_protocol_ref,
        )
        lineage["training_source_identities"]["stage2_successor"] = dict(successor_identity)
        return lineage
    actual_source_claims = actual.get("source_claims")
    if (
        not isinstance(actual_source_claims, Mapping)
        or set(actual_source_claims) != STAGE2_REPAIR_REQUIRED_SOURCE_CLAIMS
    ):
        raise PostTrainingError("Stage2 repair source claims are incomplete")
    collection_code = collection.get("code")
    if not isinstance(collection_code, Mapping):
        raise PostTrainingError("collection code identity is absent")
    collection_commit = _sha1(
        collection_code.get("git_commit"), "collection code commit"
    )
    base_ref, stage1_identity = _code_verification_identity(
        Path(str(base_verification_ref["path"])),
        str(base_verification_ref["sha256"]),
        run_root=run_root,
        name="Stage1 training code verification",
        expected_superseded_commit=collection_commit,
        expected_code_root=run_root / "code",
    )
    base_protocol_path = _regular_file(
        Path(str(base_protocol_ref["path"])), run_root, "training protocol"
    )
    base_protocol = _load_json(base_protocol_path, "training protocol")
    if (
        base_protocol.get("schema_name") != TRAINING_PROTOCOL_SCHEMA
        or base_protocol.get("schema_version") != 1
        or base_protocol.get("run_id") != run_id
        or base_protocol.get("navigation_training_code_commit")
        != stage1_identity["code_commit"]
    ):
        raise PostTrainingError("Stage1 training protocol source identity differs")

    repair_ref, repair_identity = _code_verification_identity(
        Path(str(values[0])),
        str(values[1]),
        run_root=run_root,
        name="Stage2 repair code verification",
        expected_superseded_commit=str(stage1_identity["code_commit"]),
        forbidden_code_root=Path(str(stage1_identity["code_root"])),
        required_source_files=tuple(TRAINING_SOURCE_FILES),
        expected_source_claims=actual_source_claims,
    )
    if repair_identity["code_commit"] == stage1_identity["code_commit"]:
        raise PostTrainingError("Stage2 repair code commit must differ from Stage1")
    repair_protocol_ref = _checked_provenance_file(
        Path(str(values[2])),
        str(values[3]),
        run_root=run_root,
        name="Stage2 repair protocol",
        schema=STAGE2_REPAIR_PROTOCOL_SCHEMA,
        run_id=run_id,
    )
    repair_protocol = _load_json(
        Path(str(repair_protocol_ref["path"])), "Stage2 repair protocol"
    )
    expected_fields = {
        "schema_name",
        "schema_version",
        "run_id",
        "base_training_protocol_sha256",
        "stage1_code_verification_sha256",
        "stage1_checkpoint_sha256",
        "stage2_repair_code_verification_sha256",
        "stage2_replay_manifest_sha256",
        "stage2_run_manifest_sha256",
        "navigation_training_code_commit",
        "validation_split_reads",
        "stage2_training_additions",
        "stage2_support",
    }
    if set(repair_protocol) != expected_fields:
        raise PostTrainingError("Stage2 repair protocol fields differ")
    additions = _stage2_training_additions(
        repair_protocol.get("stage2_training_additions"),
        name="Stage2 repair protocol training additions",
        exact=True,
    )
    support = _stage2_support_contract(
        repair_protocol.get("stage2_support"),
        name="Stage2 repair protocol support",
    )
    stage2_replay = stage2.get("replay_manifest")
    stage2_run = stage2.get("training_run_manifest")
    repair_contract = stage2.get("repair_contract")
    stage1_checkpoint_sha = (
        repair_contract.get("stage1_checkpoint_sha256")
        if isinstance(repair_contract, Mapping)
        else None
    )
    if (
        repair_protocol.get("base_training_protocol_sha256")
        != base_protocol_ref["sha256"]
        or repair_protocol.get("stage1_code_verification_sha256")
        != base_ref["sha256"]
        or not isinstance(stage2_replay, Mapping)
        or not isinstance(stage2_run, Mapping)
        or _sha256(stage1_checkpoint_sha, "Stage1 checkpoint SHA-256")
        != repair_protocol.get("stage1_checkpoint_sha256")
        or repair_protocol.get("stage2_replay_manifest_sha256")
        != stage2_replay.get("sha256")
        or repair_protocol.get("stage2_run_manifest_sha256")
        != stage2_run.get("sha256")
        or repair_protocol.get("stage2_repair_code_verification_sha256")
        != repair_ref["sha256"]
        or repair_protocol.get("navigation_training_code_commit")
        != repair_identity["code_commit"]
        or type(repair_protocol.get("validation_split_reads")) is not int
        or repair_protocol.get("validation_split_reads") != 0
        or actual.get("training_additions") != additions
        or actual.get("stage2_support") != support
        or type(actual.get("validation_split_reads")) is not int
        or actual.get("validation_split_reads") != 0
    ):
        raise PostTrainingError("Stage2 repair protocol/actual identity differs")
    identities = {
        "stage1": {
            **stage1_identity,
            "training_protocol_sha256": base_protocol_ref["sha256"],
        },
        "stage2_repair": {
            **repair_identity,
            "stage2_repair_protocol_sha256": repair_protocol_ref["sha256"],
            "base_training_protocol_sha256": base_protocol_ref["sha256"],
            "stage1_checkpoint_sha256": stage1_checkpoint_sha,
            "stage2_replay_manifest_sha256": stage2_replay["sha256"],
            "stage2_run_manifest_sha256": stage2_run["sha256"],
        },
    }
    return {
        "stage2_repair_code_verification": repair_ref,
        "stage2_repair_protocol": repair_protocol_ref,
        "training_source_identities": identities,
    }


def _require_repair_evaluation_code_commit(
    evaluation_code: Mapping[str, object],
    repair_lineage: Mapping[str, object] | None,
) -> None:
    if repair_lineage is None:
        return
    source_identities = repair_lineage.get("training_source_identities")
    repair_identity = (
        source_identities.get("stage2_repair")
        if isinstance(source_identities, Mapping)
        else None
    )
    if not isinstance(repair_identity, Mapping):
        raise PostTrainingError("Stage2 repair source identity is absent")
    evaluation_commit = _sha1(
        evaluation_code.get("commit"), "evaluation code commit"
    )
    repair_commit = _sha1(
        repair_identity.get("code_commit"), "Stage2 repair code commit"
    )
    if isinstance(source_identities.get("stage2_successor"), Mapping):
        if not isinstance(evaluation_code.get("source_root"), str):
            raise PostTrainingError("successor evaluation requires an explicit immutable source root")
        return
    if evaluation_commit != repair_commit:
        raise PostTrainingError(
            "evaluation code commit differs from Stage2 repair code commit"
        )


def _model_identity(collection: Mapping[str, object]) -> dict[str, object]:
    lightnav = collection.get("lightnav")
    if not isinstance(lightnav, Mapping):
        raise PostTrainingError("collection LightNav identity is absent")
    result = {
        "lightnav_source_revision": _sha1(
            lightnav.get("original_release_revision"), "LightNav source revision"
        ),
        "lightnav_hook_sha256": _sha256(
            lightnav.get("context_patch_sha256"), "LightNav hook SHA-256"
        ),
        "lightnav_weight_sha256": _sha256(
            lightnav.get("weight_sha256"), "LightNav weight SHA-256"
        ),
        "tokenizer_sha256": _sha256(
            lightnav.get("tokenizer_sha256"), "tokenizer SHA-256"
        ),
        "processor_sha256": _sha256(
            lightnav.get("processor_config_sha256"), "processor SHA-256"
        ),
        "history_config_sha256": _sha256(
            lightnav.get("history_config_sha256"), "history config SHA-256"
        ),
        "original_release_weights": True,
    }
    return result


def _arm_config(
    arm: str,
    *,
    evaluation_code: Mapping[str, object],
    collection_sha256: str,
    split_sha256: str,
    eval_plan_sha256: str,
    checkpoint_sha256: str | None,
    context_source_sha256: str,
) -> dict[str, object]:
    return {
        "arm": arm,
        "behavior": ARM_TO_EVAL_BEHAVIOR[arm],
        "collection_manifest_sha256": collection_sha256,
        "split_manifest_sha256": split_sha256,
        "evaluation_plan_sha256": eval_plan_sha256,
        "evaluation_code_commit": evaluation_code["commit"],
        "evaluation_code_inventory_sha256": evaluation_code["inventory_sha256"],
        "checkpoint_sha256": checkpoint_sha256,
        "max_physical_steps": MAX_STEPS,
        "seed": EVAL_SEED,
        "expected_context_source_sha256": (
            context_source_sha256 if arm == "navigation_rlt" else None
        ),
        "deterministic": True,
        "upstream_split_read": "train",
        "validation_split_reads": 0,
    }


def build_plan(args: argparse.Namespace) -> dict[str, object]:
    runs_root = Path(args.runs_root).expanduser().resolve(strict=True)
    run_root = _regular_directory(args.run_root, runs_root, "A800 run root")
    if RUN_ID_RE.fullmatch(args.run_id) is None or run_root.name != args.run_id:
        raise PostTrainingError("run_id is unsafe or differs from the A800 run root")
    output = Path(args.output).expanduser()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    try:
        output.resolve(strict=False).parent.resolve(strict=True).relative_to(run_root)
    except (OSError, ValueError) as exc:
        raise PostTrainingError("post-training plan output must be inside the run root") from exc
    _safe_directory_chain(
        run_root,
        output.resolve(strict=False).parent,
        "post-training plan output",
        create=True,
    )

    stage2 = _stage2_evidence(
        args.stage2_marker,
        expected_marker_sha256=args.expected_stage2_marker_sha256,
        run_root=run_root,
        run_id=args.run_id,
        formal_completion_path=getattr(args, "stage2_formal_completion", None),
        expected_formal_completion_sha256=getattr(args, "expected_stage2_formal_completion_sha256", None),
        source_root=getattr(args, "stage2_source_root", None),
    )
    collection, collection_ref = _collection_evidence(
        args.collection_manifest,
        expected_sha256=args.expected_collection_manifest_sha256,
        run_root=run_root,
        run_id=args.run_id,
    )
    _, eval_plan_ref, split_ref, identity_sha = _evaluation_plan_evidence(
        args.eval_plan,
        expected_plan_sha256=args.expected_eval_plan_sha256,
        split_path=args.split_manifest,
        expected_split_sha256=args.expected_split_manifest_sha256,
        run_root=run_root,
        run_id=args.run_id,
    )
    evaluation_code, evaluation_inventory_ref = _evaluation_inventory_evidence(
        args.evaluation_code_inventory,
        expected_sha256=args.expected_evaluation_code_inventory_sha256,
        source_root=getattr(args, "evaluation_source_root", None),
    )
    training_verification_ref = _checked_provenance_file(
        args.training_code_verification,
        args.expected_training_code_verification_sha256,
        run_root=run_root,
        name="training code verification",
        schema="navigation_rlt_a800_code_verification",
        run_id=None,
    )
    training_protocol_ref = _checked_provenance_file(
        args.training_protocol,
        args.expected_training_protocol_sha256,
        run_root=run_root,
        name="training protocol",
        schema="navigation_rlt_training_protocol",
        run_id=args.run_id,
    )
    repair_lineage = _stage2_repair_lineage(
        args=args,
        stage2=stage2,
        collection=collection,
        run_root=run_root,
        run_id=args.run_id,
        base_verification_ref=training_verification_ref,
        base_protocol_ref=training_protocol_ref,
    )
    _require_repair_evaluation_code_commit(evaluation_code, repair_lineage)
    a3_sha = _sha256(args.expected_a3_checkpoint_sha256, "A3 checkpoint SHA-256")
    protocol_sha = _sha256(args.expected_protocol_sha256, "evaluation protocol SHA-256")
    expected_model_sha = _sha256(args.expected_model_sha256, "model identity SHA-256")
    model = _model_identity(collection)
    if _object_sha256(model) != expected_model_sha:
        raise PostTrainingError("pre-registered model identity SHA-256 differs")
    collection_data = collection.get("data")
    collection_behavior = collection.get("behavior")
    if (
        not isinstance(collection_data, Mapping)
        or collection_data.get("split_manifest_sha256") != split_ref["sha256"]
        or collection_data.get("development_episode_count") != 32
        or not isinstance(collection_behavior, Mapping)
        or collection_behavior.get("checkpoint_sha256") != a3_sha
    ):
        raise PostTrainingError("collection split/A3 evaluation identity differs")

    host = _host(args.ebk_host)
    remote_run_root = _remote_path(args.ebk_run_root, "ebk run root")
    if PurePosixPath(remote_run_root).name != args.run_id:
        raise PostTrainingError("ebk run root basename differs from run_id")
    remote = {
        "host": host,
        "run_root": remote_run_root,
        "collection_manifest": f"{remote_run_root}/run_manifest.json",
        "split_manifest": f"{remote_run_root}/data/build/split_manifest.json",
        "development_data_root": f"{remote_run_root}/data/build/train_development",
        "evaluation_deployment_root": f"{remote_run_root}/eval-runtime",
        "evaluation_deployment_upload": f"{remote_run_root}/.eval-runtime.zip.uploading",
        "evaluation_plan": f"{remote_run_root}/eval-runtime/eval-plan.json",
        "evaluation_code_root": f"{remote_run_root}/eval-runtime/eval-code",
        "evaluation_code_inventory": (
            f"{remote_run_root}/eval-runtime/eval-code/evaluation_code_inventory.json"
        ),
        "wrapper": (
            f"{remote_run_root}/eval-runtime/eval-code/"
            "run_navigation_rlt_ebk_eval.sh"
        ),
        "a3_checkpoint": f"{remote_run_root}/inputs/lightnav_a3_policy_v1.npz",
        "rlt_checkpoint_temporary": (
            f"{remote_run_root}/inputs/.navigation_rlt_stage2.pt.uploading"
        ),
        "rlt_checkpoint": f"{remote_run_root}/inputs/navigation_rlt_stage2.pt",
        "verified_copy_receipt": f"{remote_run_root}/post-training/VERIFIED_COPY.json",
        "evaluation_root": f"{remote_run_root}/evaluations",
    }
    for name, value in remote.items():
        if name != "host":
            _remote_path(str(value), f"remote {name}")

    rlt_sha = str(stage2["checkpoint"]["sha256"])
    checkpoint_by_arm = {
        "native_lightnav": None,
        "a3_ppo": a3_sha,
        "navigation_rlt": rlt_sha,
    }
    arm_configs = {
        arm: _arm_config(
            arm,
            evaluation_code=evaluation_code,
            collection_sha256=str(collection_ref["sha256"]),
            split_sha256=str(split_ref["sha256"]),
            eval_plan_sha256=str(eval_plan_ref["sha256"]),
            checkpoint_sha256=checkpoint_by_arm[arm],
            context_source_sha256=str(stage2["context_source_sha256"]),
        )
        for arm in ARMS
    }
    for config in arm_configs.values():
        config["sha256"] = _object_sha256(config)

    fixed_outputs = _fixed_outputs(run_root)
    plan = {
        "schema_name": PLAN_SCHEMA,
        "schema_version": PLAN_VERSION,
        "status": "planned",
        "run_id": args.run_id,
        "a800_run_root": str(run_root),
        "stage2": stage2,
        "collection_manifest": collection_ref,
        "split_manifest": split_ref,
        "evaluation_plan": {
            **eval_plan_ref,
            "ordered_episode_identity_sha256": identity_sha,
            "episode_count": 32,
        },
        "evaluation_code": {
            **evaluation_code,
            "inventory_path": evaluation_inventory_ref["path"],
        },
        "training_code_verification": training_verification_ref,
        "training_protocol": training_protocol_ref,
        **({} if repair_lineage is None else repair_lineage),
        "remote": remote,
        "contract": {
            "arm_order": list(ARMS),
            "behavior_order": list(BEHAVIORS),
            "max_physical_steps": MAX_STEPS,
            "seed": EVAL_SEED,
            "expected_protocol_sha256": protocol_sha,
            "expected_model_sha256": expected_model_sha,
            "model_identity": model,
            "a3_checkpoint_sha256": a3_sha,
            "arm_configs": arm_configs,
        },
        "outputs": {key: str(path) for key, path in fixed_outputs.items()},
        "obs_root": str(Path(args.obs_root).expanduser()),
        "created_unix_s": time.time(),
    }
    if "source_root" in evaluation_code:
        driver = _regular_file(Path(__file__).resolve(strict=True), Path(__file__).resolve(strict=True).parent, "post-training driver")
        plan["post_training_driver"] = {"path": str(driver), "sha256": sha256_file(driver)}
    _atomic_json_new(output, plan)
    return plan


REMOTE_PREFLIGHT = r'''
import hashlib,json,os,pathlib,sys
(run_root_s,run_id,collection_s,collection_sha,split_s,split_sha,dev_s,
 plan_s,plan_sha,code_s,inventory_s,inventory_sha,wrapper_s,wrapper_sha,
 a3_s,a3_sha,rlt_tmp_s,rlt_final_s,copy_receipt_s)=sys.argv[1:]
root=pathlib.Path(run_root_s)
def fail(m): raise SystemExit(m)
def regular(s,name):
 p=pathlib.Path(s)
 if p.is_symlink() or not p.is_file(): fail(name+' is absent/link')
 rp=p.resolve(strict=True)
 try: rp.relative_to(root.resolve(strict=True))
 except ValueError: fail(name+' escapes run root')
 return rp
def directory(s,name):
 p=pathlib.Path(s)
 if p.is_symlink() or not p.is_dir(): fail(name+' is absent/link')
 rp=p.resolve(strict=True)
 try: rp.relative_to(root.resolve(strict=True))
 except ValueError: fail(name+' escapes run root')
 return rp
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
if root.is_symlink() or not root.is_dir() or root.name!=run_id: fail('remote run root differs')
checks=((collection_s,collection_sha,'collection'),(split_s,split_sha,'split'),
 (plan_s,plan_sha,'eval plan'),(inventory_s,inventory_sha,'eval inventory'),
 (wrapper_s,wrapper_sha,'wrapper'),(a3_s,a3_sha,'A3 checkpoint'))
for p,e,n in checks:
 if sha(regular(p,n))!=e: fail(n+' SHA-256 differs')
dev=directory(dev_s,'development dataset root')
split=json.loads(regular(split_s,'split').read_text())
entry=split['splits']['train_development']
dataset=regular(str(dev/'train'/'train.json.gz'),'development dataset')
gt=regular(str(dev/'train'/'train_gt.json.gz'),'development ground truth')
if sha(dataset)!=entry['dataset_sha256'] or sha(gt)!=entry['gt_sha256']:
 fail('development dataset hashes differ')
inventory=json.loads(regular(inventory_s,'eval inventory').read_text())
code=directory(code_s,'evaluation code root')
for name,expected in inventory['files'].items():
 p=regular(str(code/pathlib.PurePosixPath(name)),'evaluation source '+name)
 if sha(p)!=expected: fail('evaluation source SHA-256 differs: '+name)
for absent in (rlt_tmp_s,rlt_final_s,copy_receipt_s):
 if os.path.lexists(absent): fail('remote output already exists: '+absent)
print(json.dumps({'status':'passed','development_dataset_sha256':entry['dataset_sha256'],
 'development_gt_sha256':entry['gt_sha256'],'evaluation_inventory_sha256':inventory_sha},
 sort_keys=True,separators=(',',':')))
'''


REMOTE_EVAL_DEPLOY_PRECHECK = r'''
import json,os,pathlib,sys
root_s,run_id,upload_s,final_s=sys.argv[1:]
root=pathlib.Path(root_s)
if root.is_symlink() or not root.is_dir() or root.name!=run_id:
 raise SystemExit('remote run root differs')
root=root.resolve(strict=True)
for raw,label in ((upload_s,'deployment upload'),(final_s,'deployment root')):
 p=pathlib.Path(raw)
 try: p.resolve(strict=False).relative_to(root)
 except ValueError: raise SystemExit(label+' escapes run root')
 if os.path.lexists(p): raise SystemExit(label+' already exists')
print(json.dumps({'status':'ready'},sort_keys=True,separators=(',',':')))
'''


REMOTE_FINALIZE_EVAL_DEPLOYMENT = r'''
import ctypes,errno,hashlib,json,os,pathlib,stat,sys,tempfile,time,zipfile
(root_s,run_id,upload_s,archive_sha,size_s,final_s,plan_sha,inventory_sha)=sys.argv[1:]
root=pathlib.Path(root_s)
if root.is_symlink() or not root.is_dir() or root.name!=run_id:
 raise SystemExit('remote run root differs')
root=root.resolve(strict=True); upload=pathlib.Path(upload_s); final=pathlib.Path(final_s)
def inside(p,label):
 try: p.resolve(strict=False).relative_to(root)
 except ValueError: raise SystemExit(label+' escapes run root')
def sha_bytes(value): return hashlib.sha256(value).hexdigest()
def sha_file(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
 return h.hexdigest()
def rename_noreplace(source,destination):
 libc=ctypes.CDLL(None,use_errno=True)
 function=getattr(libc,'renameat2',None)
 if function is None: raise SystemExit('renameat2(RENAME_NOREPLACE) is unavailable')
 function.argtypes=(ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint)
 function.restype=ctypes.c_int
 if function(-100,os.fsencode(source),-100,os.fsencode(destination),1)!=0:
  number=ctypes.get_errno()
  if number==errno.EEXIST: raise FileExistsError(destination)
  raise OSError(number,os.strerror(number),str(destination))
inside(upload,'deployment upload'); inside(final,'deployment root')
if upload.is_symlink() or not upload.is_file(): raise SystemExit('deployment upload is absent/link')
if upload.stat().st_size!=int(size_s) or sha_file(upload)!=archive_sha:
 raise SystemExit('deployment archive readback differs')
if os.path.lexists(final): raise SystemExit('deployment root already exists')
staging=final.with_name('.'+final.name+'.extracting')
inside(staging,'deployment staging root')
if os.path.lexists(staging): raise SystemExit('deployment staging root already exists')
with zipfile.ZipFile(upload,'r') as archive:
 infos=archive.infolist(); names=[info.filename for info in infos]
 if len(names)!=len(set(names)) or any(info.is_dir() for info in infos):
  raise SystemExit('deployment archive contains duplicate/directory entries')
 for info in infos:
  mode=(info.external_attr>>16)&0o170000
  if mode not in (0,stat.S_IFREG): raise SystemExit('deployment archive contains a non-regular entry')
 inventory_name='eval-code/evaluation_code_inventory.json'
 if inventory_name not in names or 'eval-plan.json' not in names:
  raise SystemExit('deployment archive lacks plan or inventory')
 inventory_bytes=archive.read(inventory_name)
 if sha_bytes(inventory_bytes)!=inventory_sha:
  raise SystemExit('deployment inventory SHA-256 differs')
 try: inventory=json.loads(inventory_bytes.decode('utf-8'))
 except (UnicodeDecodeError,json.JSONDecodeError) as exc: raise SystemExit('deployment inventory is invalid JSON') from exc
 files=inventory.get('files') if isinstance(inventory,dict) else None
 if not isinstance(files,dict): raise SystemExit('deployment inventory files are absent')
 expected={'eval-plan.json',inventory_name}|{'eval-code/'+name for name in files}
 if set(names)!=expected: raise SystemExit('deployment archive member set differs from inventory')
 if sha_bytes(archive.read('eval-plan.json'))!=plan_sha:
  raise SystemExit('deployment evaluation plan SHA-256 differs')
 staging.mkdir(mode=0o700)
 try:
  for info in infos:
   relative=pathlib.PurePosixPath(info.filename)
   if relative.is_absolute() or any(part in ('','.','..') for part in relative.parts):
    raise SystemExit('deployment archive path is not canonical')
   destination=staging.joinpath(*relative.parts)
   destination.parent.mkdir(parents=True,exist_ok=True)
   with destination.open('xb') as stream:
    stream.write(archive.read(info)); stream.flush(); os.fsync(stream.fileno())
  for name,expected_sha in files.items():
   path=staging.joinpath('eval-code',*pathlib.PurePosixPath(name).parts)
   if path.is_symlink() or not path.is_file() or sha_file(path)!=expected_sha:
    raise SystemExit('deployed evaluation source SHA-256 differs: '+name)
  if sha_file(staging/'eval-plan.json')!=plan_sha or sha_file(staging/inventory_name)!=inventory_sha:
   raise SystemExit('deployed plan/inventory readback differs')
  receipt={'schema_name':'navigation_rlt_evaluation_deployment','schema_version':1,
   'status':'verified','run_id':run_id,'archive_sha256':archive_sha,
   'evaluation_plan_sha256':plan_sha,'evaluation_code_inventory_sha256':inventory_sha,
   'file_count':len(files),'completed_unix_s':time.time()}
  payload=(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
  with (staging/'DEPLOYMENT.json').open('xb') as stream:
   stream.write(payload); stream.flush(); os.fsync(stream.fileno())
   rename_noreplace(staging,final)
  descriptor=os.open(final.parent,os.O_RDONLY); os.fsync(descriptor); os.close(descriptor)
 except BaseException:
  if staging.exists():
   import shutil; shutil.rmtree(staging)
  raise
upload.unlink()
print(json.dumps({'status':'verified','archive_sha256':archive_sha,
 'evaluation_plan_sha256':plan_sha,'evaluation_code_inventory_sha256':inventory_sha,
 'file_count':len(files),'deployment_root':str(final)},sort_keys=True,separators=(',',':')))
'''


REMOTE_FINALIZE_COPY = r'''
import hashlib,json,os,pathlib,sys,tempfile,time
(root_s,run_id,source_s,expected,size_s,tmp_s,final_s,receipt_s)=sys.argv[1:]
root=pathlib.Path(root_s).resolve(strict=True); tmp=pathlib.Path(tmp_s); final=pathlib.Path(final_s); receipt=pathlib.Path(receipt_s)
def fail(m): raise SystemExit(m)
def inside(p):
 try: p.resolve(strict=False).relative_to(root)
 except ValueError: fail('path escapes remote run root')
for p in (tmp,final,receipt): inside(p)
if tmp.is_symlink() or not tmp.is_file(): fail('temporary checkpoint is absent/link')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def fsync_parent(path):
 fd=os.open(path.parent,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
 try: os.fsync(fd)
 finally: os.close(fd)
def identity(status): return (status.st_dev,status.st_ino)
def safe_unlink(path,expected_identity):
 try: observed=identity(path.lstat())
 except FileNotFoundError: return
 if observed==expected_identity: path.unlink()
def publish_new(temporary,destination,temporary_identity):
 os.link(temporary,destination,follow_symlinks=False)
 fsync_parent(destination)
 safe_unlink(temporary,temporary_identity)
 fsync_parent(destination)
if tmp.stat().st_size!=int(size_s) or sha(tmp)!=expected: fail('temporary checkpoint readback differs')
if os.path.lexists(final) or os.path.lexists(receipt): fail('final checkpoint or receipt exists')
final.parent.mkdir(parents=True,exist_ok=True); receipt.parent.mkdir(parents=True,exist_ok=True)
tmp_identity=identity(tmp.lstat())
try: publish_new(tmp,final,tmp_identity)
except BaseException:
 safe_unlink(tmp,tmp_identity)
 raise
if final.is_symlink() or not final.is_file() or final.stat().st_size!=int(size_s) or sha(final)!=expected:
 fail('atomic final checkpoint readback differs')
value={'schema_name':'navigation_rlt_verified_checkpoint_copy','schema_version':1,
 'status':'verified','run_id':run_id,
 'source':{'path':source_s,'sha256':expected,'size_bytes':int(size_s)},
 'destination':{'path':str(final),'sha256':expected,'size_bytes':int(size_s)},
 'temporary_path':str(tmp),'temporary_absent_after_publish':not os.path.lexists(tmp),
 'atomic_publish':True,'completed_unix_s':time.time()}
payload=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
fd,name=tempfile.mkstemp(prefix=receipt.name+'.staging-',dir=receipt.parent)
temporary=pathlib.Path(name)
temporary_identity=identity(os.fstat(fd))
try:
 with os.fdopen(fd,'wb') as f: f.write(payload); f.flush(); os.fsync(f.fileno())
 publish_new(temporary,receipt,temporary_identity)
except BaseException:
 safe_unlink(temporary,temporary_identity)
 raise
print(json.dumps({'status':'verified','receipt_path':str(receipt),
 'receipt_sha256':hashlib.sha256(payload).hexdigest(),'receipt_size_bytes':len(payload),'checkpoint_sha256':expected,
 'checkpoint_size_bytes':int(size_s)},sort_keys=True,separators=(',',':')))
'''


REMOTE_RESULT = r'''
import hashlib,json,pathlib,sys
(root_s,run_id,behavior,eval_behavior,out_s,collection_sha,split_sha,plan_sha,inventory_sha,checkpoint_sha)=sys.argv[1:]
root=pathlib.Path(root_s).resolve(strict=True); out=pathlib.Path(out_s)
def fail(m): raise SystemExit(m)
def regular(p,name):
 if p.is_symlink() or not p.is_file(): fail(name+' absent/link')
 rp=p.resolve(strict=True)
 try: rp.relative_to(root)
 except ValueError: fail(name+' escapes run root')
 return rp
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
evaluation=regular(out/'evaluation.json','evaluation'); completed=regular(out/'COMPLETED.json','completion')
ev=json.loads(evaluation.read_text()); done=json.loads(completed.read_text())
evsha=sha(evaluation)
if ev.get('schema_name')!='navigation_rlt_matched_train_development_eval' or ev.get('status')!='completed': fail('evaluation schema/status differs')
if ev.get('run_id')!=run_id or ev.get('behavior')!=eval_behavior or ev.get('split_label')!='train_development': fail('evaluation identity differs')
if ev.get('collection_manifest_sha256')!=collection_sha or ev.get('run_manifest_sha256')!=collection_sha: fail('collection identity differs')
if ev.get('evaluation_plan',{}).get('sha256')!=plan_sha or ev.get('evaluation_plan',{}).get('split_manifest_sha256')!=split_sha: fail('plan identity differs')
if ev.get('evaluation_code',{}).get('inventory_sha256')!=inventory_sha: fail('evaluation code identity differs')
if len(ev.get('episodes',[]))!=32 or [x.get('episode_order') for x in ev['episodes']]!=list(range(32)): fail('episode results differ')
if ev.get('executor',{}).get('max_physical_steps')!=160: fail('physical step cap differs')
source=ev.get('behavior_source',{})
if behavior=='a3' and source.get('checkpoint_sha256')!=checkpoint_sha: fail('A3 checkpoint differs')
if behavior=='navigation_rlt' and source.get('checkpoint_sha256')!=checkpoint_sha: fail('RLT checkpoint differs')
if behavior=='native' and checkpoint_sha!='-': fail('native checkpoint contract differs')
if done.get('status')!='completed' or done.get('behavior')!=eval_behavior or done.get('evaluation_sha256')!=evsha: fail('completion marker differs')
print(json.dumps({'status':'verified','evaluation_path':str(evaluation),'evaluation_sha256':evsha,
 'evaluation_size_bytes':evaluation.stat().st_size,'completion_path':str(completed),
 'completion_sha256':sha(completed),'completion_size_bytes':completed.stat().st_size},
 sort_keys=True,separators=(',',':')))
'''


def _run(command: Sequence[str], *, timeout: int, name: str) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        list(command),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise PostTrainingError(f"{name} failed ({completed.returncode}): {detail}")
    return completed


def _ssh_json(host: str, code: str, args: Sequence[str], *, timeout: int, name: str) -> dict[str, object]:
    for arg in args:
        if not isinstance(arg, str) or not arg or REMOTE_TOKEN_RE.fullmatch(arg) is None:
            raise PostTrainingError(f"{name} argument contains unsupported characters")
    remote_command = " ".join(
        shlex.quote(item) for item in ["python3", "-c", code, *args]
    )
    completed = _run(
        ["ssh", "--", host, remote_command],
        timeout=timeout,
        name=name,
    )
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise PostTrainingError(f"{name} returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise PostTrainingError(f"{name} returned a non-object")
    return value


def _scp_to(host: str, source: Path, destination: str, *, timeout: int, name: str) -> None:
    _remote_path(destination, f"{name} destination")
    _run(
        ["scp", "-O", "--", str(source), f"{host}:{destination}"],
        timeout=timeout,
        name=name,
    )


def _scp_from(
    host: str,
    source: str,
    destination: Path,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
    timeout: int,
    name: str,
    containment_root: Path | None = None,
) -> Path:
    _remote_path(source, f"{name} source")
    destination = Path(destination)
    if containment_root is not None:
        _safe_directory_chain(
            containment_root,
            destination.parent,
            name,
            create=True,
        )
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".staging-", dir=destination.parent
    )
    temporary_identity = _file_identity(os.fstat(descriptor))
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        _run(
            ["scp", "-O", "--", f"{host}:{source}", str(temporary)],
            timeout=timeout,
            name=name,
        )
        if (
            temporary.is_symlink()
            or not temporary.is_file()
            or temporary.stat().st_size != expected_size_bytes
            or sha256_file(temporary) != expected_sha256
        ):
            raise PostTrainingError(f"{name} A800 readback differs")
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        if containment_root is not None:
            _safe_directory_chain(
                containment_root,
                destination.parent,
                name,
                create=False,
            )
        _publish_file_new(temporary, destination, temporary_identity)
    except BaseException:
        _unlink_if_same(temporary, temporary_identity)
        raise
    return destination.resolve(strict=True)


def _plan_path(value: object, run_root: Path, name: str) -> Path:
    if not isinstance(value, str):
        raise PostTrainingError(f"{name} path is absent")
    return _regular_file(Path(value), run_root, name)


def _load_plan(path: Path, expected_sha256: str) -> tuple[Path, dict[str, object], str]:
    raw = Path(path).expanduser()
    if not os.path.lexists(raw) or _path_is_link_or_reparse(raw):
        raise PostTrainingError("post-training plan must be an existing regular file")
    try:
        path = raw.resolve(strict=True)
    except OSError as exc:
        raise PostTrainingError("post-training plan is absent") from exc
    if not path.is_file():
        raise PostTrainingError("post-training plan is not a regular file")
    digest = _verify_file_sha(path, expected_sha256, "post-training plan")
    value = _load_json(path, "post-training plan")
    if (
        value.get("schema_name") != PLAN_SCHEMA
        or value.get("schema_version") != PLAN_VERSION
        or value.get("status") != "planned"
    ):
        raise PostTrainingError("post-training plan schema/status differs")
    return path, value, digest


def _revalidate_plan(plan: Mapping[str, object]) -> tuple[Path, str]:
    run_id = plan.get("run_id")
    if not isinstance(run_id, str) or RUN_ID_RE.fullmatch(run_id) is None:
        raise PostTrainingError("post-training plan run_id is invalid")
    raw_run_root = Path(str(plan.get("a800_run_root")))
    if not os.path.lexists(raw_run_root) or _path_is_link_or_reparse(raw_run_root):
        raise PostTrainingError("A800 run root changed")
    try:
        run_root = raw_run_root.resolve(strict=True)
    except OSError as exc:
        raise PostTrainingError("A800 run root changed") from exc
    if not run_root.is_dir() or run_root.name != run_id:
        raise PostTrainingError("A800 run root changed")
    stage2 = plan.get("stage2")
    collection = plan.get("collection_manifest")
    eval_plan = plan.get("evaluation_plan")
    split = plan.get("split_manifest")
    evaluation_code = plan.get("evaluation_code")
    training_verification = plan.get("training_code_verification")
    training_protocol = plan.get("training_protocol")
    if not all(
        isinstance(value, Mapping)
        for value in (
            stage2,
            collection,
            eval_plan,
            split,
            evaluation_code,
            training_verification,
            training_protocol,
        )
    ):
        raise PostTrainingError("post-training plan evidence is incomplete")
    marker = stage2.get("marker")
    if not isinstance(marker, Mapping):
        raise PostTrainingError("post-training plan Stage2 marker is absent")
    refreshed = _stage2_evidence(
        Path(str(marker.get("path"))),
        expected_marker_sha256=str(marker.get("sha256")),
        run_root=run_root,
        run_id=run_id,
        formal_completion_path=(Path(str(stage2["formal_completion"]["path"])) if "formal_completion" in stage2 else None),
        expected_formal_completion_sha256=(stage2["formal_completion"]["sha256"] if "formal_completion" in stage2 else None),
        source_root=(Path(str(stage2["successor_source_identity"]["source_root"])) if "successor_source_identity" in stage2 else None),
    )
    if refreshed != stage2:
        raise PostTrainingError("Stage2 evidence changed after planning")
    for evidence, name in (
        (collection, "collection outer manifest"),
        (eval_plan, "evaluation plan"),
        (split, "train split manifest"),
        (training_verification, "training code verification"),
        (training_protocol, "training protocol"),
    ):
        path = _plan_path(evidence.get("path"), run_root, name)
        if sha256_file(path) != evidence.get("sha256"):
            raise PostTrainingError(f"{name} changed after planning")
    repair_verification = plan.get("stage2_repair_code_verification")
    repair_protocol = plan.get("stage2_repair_protocol")
    source_identities = plan.get("training_source_identities")
    repair_expected = "repair_contract" in refreshed
    refreshed_lineage: dict[str, object] | None = None
    if repair_expected:
        if not all(
            isinstance(value, Mapping)
            for value in (repair_verification, repair_protocol, source_identities)
        ):
            raise PostTrainingError("post-training plan repair lineage is incomplete")
        collection_value = _load_json(
            _plan_path(
                collection.get("path"), run_root, "collection outer manifest"
            ),
            "collection outer manifest",
        )
        refreshed_lineage = _stage2_repair_lineage(
            args=argparse.Namespace(
                stage2_repair_code_verification=repair_verification.get("path"),
                expected_stage2_repair_code_verification_sha256=(
                    repair_verification.get("sha256")
                ),
                stage2_repair_protocol=repair_protocol.get("path"),
                expected_stage2_repair_protocol_sha256=repair_protocol.get("sha256"),
            ),
            stage2=refreshed,
            collection=collection_value,
            run_root=run_root,
            run_id=run_id,
            base_verification_ref=training_verification,
            base_protocol_ref=training_protocol,
        )
        if refreshed_lineage != {
            "stage2_repair_code_verification": dict(repair_verification),
            "stage2_repair_protocol": dict(repair_protocol),
            "training_source_identities": dict(source_identities),
        }:
            raise PostTrainingError("post-training plan repair lineage changed")
    elif any(
        value is not None
        for value in (repair_verification, repair_protocol, source_identities)
    ):
        raise PostTrainingError("legacy post-training plan claims Stage2 repair")
    refreshed_evaluation_code, inventory_ref = _evaluation_inventory_evidence(
        Path(str(evaluation_code.get("inventory_path"))),
        expected_sha256=str(evaluation_code.get("inventory_sha256")),
        source_root=(Path(str(evaluation_code["source_root"])) if "source_root" in evaluation_code else None),
    )
    if "source_root" in evaluation_code:
        if refreshed_evaluation_code != {key: value for key, value in evaluation_code.items() if key != "inventory_path"}:
            raise PostTrainingError("explicit evaluation source identity changed after planning")
        driver = plan.get("post_training_driver")
        if not isinstance(driver, Mapping) or driver.get("path") != str(Path(__file__).resolve(strict=True)):
            raise PostTrainingError("post-training driver identity differs")
        _verify_file_sha(Path(str(driver["path"])), driver.get("sha256"), "post-training driver")
    if inventory_ref["sha256"] != evaluation_code.get("inventory_sha256"):
        raise PostTrainingError("evaluation inventory changed after planning")
    if repair_expected:
        if evaluation_code.get("commit") != refreshed_evaluation_code["commit"]:
            raise PostTrainingError(
                "evaluation code commit changed after repair planning"
            )
        _require_repair_evaluation_code_commit(
            refreshed_evaluation_code, refreshed_lineage
        )
    return run_root, run_id


def _build_evaluation_runtime_archive(
    plan: Mapping[str, object], run_root: Path, archive_path: Path
) -> dict[str, object]:
    """Persist the exact ebk evaluation runtime without contacting a remote host."""

    evaluation_plan = plan["evaluation_plan"]
    evaluation_code = plan["evaluation_code"]
    assert isinstance(evaluation_plan, Mapping)
    assert isinstance(evaluation_code, Mapping)
    plan_path = _plan_path(evaluation_plan.get("path"), run_root, "evaluation plan")
    _verify_file_sha(plan_path, evaluation_plan.get("sha256"), "evaluation plan")
    inventory_path = Path(str(evaluation_code.get("inventory_path"))).resolve(strict=True)
    if inventory_path.is_symlink() or not inventory_path.is_file():
        raise PostTrainingError("evaluation code inventory changed before deployment")
    _verify_file_sha(
        inventory_path, evaluation_code.get("inventory_sha256"), "evaluation code inventory"
    )
    files = evaluation_code.get("files")
    if not isinstance(files, Mapping) or set(files) != REQUIRED_EVAL_FILES:
        raise PostTrainingError("evaluation code dependency set differs before deployment")

    archive_path = Path(archive_path).expanduser().absolute()
    if os.path.lexists(archive_path):
        raise FileExistsError(archive_path)
    _safe_directory_chain(
        run_root, archive_path.parent, "evaluation runtime archive", create=True
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=archive_path.name + ".staging-", dir=archive_path.parent
    )
    temporary = Path(temporary_name)
    temporary_identity = _file_identity(os.fstat(descriptor))
    try:
        with os.fdopen(descriptor, "w+b") as stream:
            with zipfile.ZipFile(
                stream,
                "w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=9,
                allowZip64=False,
            ) as archive:
                archive.write(plan_path, "eval-plan.json")
                archive.write(inventory_path, "eval-code/evaluation_code_inventory.json")
                for name in sorted(files):
                    source = _local_evaluation_source(name, Path(str(evaluation_code["source_root"])) if "source_root" in evaluation_code else None)
                    expected = _sha256(files[name], f"evaluation source {name} SHA-256")
                    if sha256_file(source) != expected:
                        raise PostTrainingError(
                            f"evaluation source {name} changed before deployment"
                        )
                    archive.write(source, f"eval-code/{name}")
            stream.flush()
            os.fsync(stream.fileno())
        result = {
            "path": str(archive_path),
            "sha256": sha256_file(temporary),
            "size_bytes": temporary.stat().st_size,
            "file_count": len(files),
        }
        _publish_file_new(temporary, archive_path, temporary_identity)
        return result
    except BaseException:
        _unlink_if_same(temporary, temporary_identity)
        raise


def _deploy_evaluation_runtime(
    plan: Mapping[str, object], run_root: Path
) -> dict[str, object]:
    """Publish the hash-bound plan and complete eval dependency set atomically."""

    remote = plan["remote"]
    evaluation_plan = plan["evaluation_plan"]
    evaluation_code = plan["evaluation_code"]
    assert isinstance(remote, Mapping)
    assert isinstance(evaluation_plan, Mapping)
    assert isinstance(evaluation_code, Mapping)

    precheck = _ssh_json(
        str(remote["host"]),
        REMOTE_EVAL_DEPLOY_PRECHECK,
        [
            str(remote["run_root"]),
            str(plan["run_id"]),
            str(remote["evaluation_deployment_upload"]),
            str(remote["evaluation_deployment_root"]),
        ],
        timeout=180,
        name="ebk evaluation deployment precheck",
    )
    if precheck.get("status") != "ready":
        raise PostTrainingError("ebk evaluation deployment precheck did not pass")

    output_root = _validated_outputs(plan, run_root, create_parent=True)["root"]
    with tempfile.TemporaryDirectory(
        prefix=".evaluation-deploy-", dir=output_root
    ) as directory:
        archive_path = Path(directory) / "eval-runtime.zip"
        archive_ref = _build_evaluation_runtime_archive(plan, run_root, archive_path)
        archive_sha = str(archive_ref["sha256"])
        archive_size = int(archive_ref["size_bytes"])
        _scp_to(
            str(remote["host"]),
            archive_path,
            str(remote["evaluation_deployment_upload"]),
            timeout=3600,
            name="evaluation runtime upload",
        )

    response = _ssh_json(
        str(remote["host"]),
        REMOTE_FINALIZE_EVAL_DEPLOYMENT,
        [
            str(remote["run_root"]),
            str(plan["run_id"]),
            str(remote["evaluation_deployment_upload"]),
            archive_sha,
            str(archive_size),
            str(remote["evaluation_deployment_root"]),
            str(evaluation_plan["sha256"]),
            str(evaluation_code["inventory_sha256"]),
        ],
        timeout=600,
        name="ebk evaluation deployment finalize",
    )
    if (
        response.get("status") != "verified"
        or response.get("archive_sha256") != archive_sha
        or response.get("evaluation_plan_sha256") != evaluation_plan["sha256"]
        or response.get("evaluation_code_inventory_sha256")
        != evaluation_code["inventory_sha256"]
        or response.get("file_count") != archive_ref["file_count"]
        or response.get("deployment_root")
        != remote["evaluation_deployment_root"]
    ):
        raise PostTrainingError("ebk evaluation deployment evidence differs")
    return response


def _remote_preflight(plan: Mapping[str, object]) -> None:
    remote = plan["remote"]
    contract = plan["contract"]
    evaluation_code = plan["evaluation_code"]
    assert isinstance(remote, Mapping) and isinstance(contract, Mapping)
    assert isinstance(evaluation_code, Mapping)
    wrapper_sha = _sha256(
        evaluation_code["files"]["run_navigation_rlt_ebk_eval.sh"],
        "wrapper SHA-256",
    )
    result = _ssh_json(
        str(remote["host"]),
        REMOTE_PREFLIGHT,
        [
            str(remote["run_root"]),
            str(plan["run_id"]),
            str(remote["collection_manifest"]),
            str(plan["collection_manifest"]["sha256"]),
            str(remote["split_manifest"]),
            str(plan["split_manifest"]["sha256"]),
            str(remote["development_data_root"]),
            str(remote["evaluation_plan"]),
            str(plan["evaluation_plan"]["sha256"]),
            str(remote["evaluation_code_root"]),
            str(remote["evaluation_code_inventory"]),
            str(evaluation_code["inventory_sha256"]),
            str(remote["wrapper"]),
            wrapper_sha,
            str(remote["a3_checkpoint"]),
            str(contract["a3_checkpoint_sha256"]),
            str(remote["rlt_checkpoint_temporary"]),
            str(remote["rlt_checkpoint"]),
            str(remote["verified_copy_receipt"]),
        ],
        timeout=180,
        name="ebk preflight",
    )
    if result.get("status") != "passed":
        raise PostTrainingError("ebk preflight did not pass")


def _transfer_checkpoint(
    plan: Mapping[str, object], run_root: Path
) -> tuple[Path, dict[str, object], str]:
    remote = plan["remote"]
    stage2 = plan["stage2"]
    outputs = plan["outputs"]
    assert isinstance(remote, Mapping) and isinstance(stage2, Mapping)
    assert isinstance(outputs, Mapping)
    checkpoint = stage2["checkpoint"]
    assert isinstance(checkpoint, Mapping)
    local_checkpoint = _regular_file(
        Path(str(checkpoint["path"])), run_root, "Stage2 checkpoint"
    )
    _scp_to(
        str(remote["host"]),
        local_checkpoint,
        str(remote["rlt_checkpoint_temporary"]),
        timeout=7200,
        name="Stage2 checkpoint upload",
    )
    response = _ssh_json(
        str(remote["host"]),
        REMOTE_FINALIZE_COPY,
        [
            str(remote["run_root"]),
            str(plan["run_id"]),
            str(local_checkpoint),
            str(checkpoint["sha256"]),
            str(checkpoint["size_bytes"]),
            str(remote["rlt_checkpoint_temporary"]),
            str(remote["rlt_checkpoint"]),
            str(remote["verified_copy_receipt"]),
        ],
        timeout=7200,
        name="Stage2 checkpoint remote finalize",
    )
    if (
        response.get("status") != "verified"
        or response.get("checkpoint_sha256") != checkpoint["sha256"]
        or response.get("checkpoint_size_bytes") != checkpoint["size_bytes"]
        or type(response.get("receipt_size_bytes")) is not int
        or int(response["receipt_size_bytes"]) < 1
    ):
        raise PostTrainingError("remote checkpoint finalization evidence differs")
    receipt_sha = _sha256(response.get("receipt_sha256"), "verified-copy receipt SHA-256")
    local_receipt = _scp_from(
        str(remote["host"]),
        str(remote["verified_copy_receipt"]),
        _validated_outputs(plan, run_root, create_parent=True)[
            "verified_copy_receipt"
        ],
        expected_sha256=receipt_sha,
        expected_size_bytes=int(response["receipt_size_bytes"]),
        timeout=300,
        name="verified-copy receipt return",
        containment_root=run_root,
    )
    return _validate_verified_copy_receipt(plan, run_root, local_receipt)


def _validate_verified_copy_receipt(
    plan: Mapping[str, object], run_root: Path, receipt_path: Path
) -> tuple[Path, dict[str, object], str]:
    receipt_path = _regular_file(
        receipt_path, run_root, "verified checkpoint copy receipt"
    )
    receipt_sha = sha256_file(receipt_path)
    receipt = _load_json(receipt_path, "verified-copy receipt")
    stage2 = plan.get("stage2")
    remote = plan.get("remote")
    if not isinstance(stage2, Mapping) or not isinstance(remote, Mapping):
        raise PostTrainingError("verified-copy plan evidence is incomplete")
    checkpoint = stage2.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise PostTrainingError("verified-copy Stage2 checkpoint is absent")
    if (
        receipt.get("schema_name") != COPY_SCHEMA
        or receipt.get("schema_version") != 1
        or receipt.get("status") != "verified"
        or receipt.get("run_id") != plan["run_id"]
        or receipt.get("source") != checkpoint
        or receipt.get("destination")
        != {
            "path": remote["rlt_checkpoint"],
            "sha256": checkpoint["sha256"],
            "size_bytes": checkpoint["size_bytes"],
        }
        or receipt.get("atomic_publish") is not True
        or receipt.get("temporary_absent_after_publish") is not True
    ):
        raise PostTrainingError("returned verified-copy receipt differs")
    return receipt_path, receipt, receipt_sha


def _finalize_remote_copy(
    run_root_value: Path,
    run_id: str,
    source_path: str,
    expected_sha256: str,
    expected_size_bytes: int,
    temporary_value: Path,
    final_value: Path,
    receipt_value: Path,
) -> dict[str, object]:
    if RUN_ID_RE.fullmatch(run_id) is None:
        raise PostTrainingError("remote copy run_id is invalid")
    raw_root = Path(run_root_value).expanduser().absolute()
    if not os.path.lexists(raw_root) or _path_is_link_or_reparse(raw_root):
        raise PostTrainingError("remote copy run root is absent or linked")
    try:
        run_root = raw_root.resolve(strict=True)
    except OSError as exc:
        raise PostTrainingError("remote copy run root is absent") from exc
    if not run_root.is_dir() or run_root.name != run_id:
        raise PostTrainingError("remote copy run root differs from run_id")
    if not isinstance(source_path, str) or not source_path:
        raise PostTrainingError("remote copy source path is absent")
    checkpoint_sha = _sha256(expected_sha256, "checkpoint SHA-256")
    if type(expected_size_bytes) is not int or expected_size_bytes < 0:
        raise PostTrainingError("checkpoint size must be a non-negative integer")

    def contained(value: Path, name: str) -> Path:
        path = Path(value).expanduser().absolute()
        try:
            path.relative_to(run_root)
        except ValueError as exc:
            raise PostTrainingError(f"{name} escapes the remote run root") from exc
        return path

    temporary = contained(temporary_value, "temporary checkpoint")
    final = contained(final_value, "final checkpoint")
    receipt_path = contained(receipt_value, "verified-copy receipt")
    if len({temporary, final, receipt_path}) != 3:
        raise PostTrainingError("remote copy paths must be distinct")
    _safe_directory_chain(
        run_root,
        temporary.parent,
        "temporary checkpoint",
        create=False,
    )
    temporary = _regular_file(
        temporary, run_root, "temporary checkpoint"
    )
    if (
        temporary.stat().st_size != expected_size_bytes
        or sha256_file(temporary) != checkpoint_sha
    ):
        raise PostTrainingError("temporary checkpoint readback differs")
    for output in (final, receipt_path):
        if os.path.lexists(output):
            raise FileExistsError(output)
    _safe_directory_chain(
        run_root, final.parent, "final checkpoint", create=True
    )
    _safe_directory_chain(
        run_root, receipt_path.parent, "verified-copy receipt", create=True
    )

    temporary_identity = _file_identity(temporary.lstat())
    _publish_file_new(temporary, final, temporary_identity)
    final_path = _regular_file(final, run_root, "final checkpoint")
    if (
        final_path.stat().st_size != expected_size_bytes
        or sha256_file(final_path) != checkpoint_sha
    ):
        raise PostTrainingError("atomic final checkpoint readback differs")
    receipt = {
        "schema_name": COPY_SCHEMA,
        "schema_version": 1,
        "status": "verified",
        "run_id": run_id,
        "source": {
            "path": source_path,
            "sha256": checkpoint_sha,
            "size_bytes": expected_size_bytes,
        },
        "destination": {
            "path": str(final),
            "sha256": checkpoint_sha,
            "size_bytes": expected_size_bytes,
        },
        "temporary_path": str(temporary),
        "temporary_absent_after_publish": not os.path.lexists(temporary),
        "atomic_publish": True,
        "completed_unix_s": time.time(),
    }
    receipt_sha = _atomic_json_new(receipt_path, receipt)
    return {
        "status": "verified",
        "receipt_path": str(receipt_path),
        "receipt_sha256": receipt_sha,
        "receipt_size_bytes": receipt_path.stat().st_size,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_size_bytes": expected_size_bytes,
    }


def _publish_local_verified_copy_receipt(
    plan: Mapping[str, object],
    run_root: Path,
    source_value: Path,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
) -> tuple[Path, dict[str, object], str]:
    source_input = Path(source_value).expanduser().absolute()
    _safe_directory_chain(
        run_root,
        source_input.parent,
        "returned verified-copy receipt",
        create=False,
    )
    source = _regular_file(
        source_input, run_root, "returned verified-copy receipt"
    )
    source_sha = _sha256(expected_sha256, "verified-copy receipt SHA-256")
    if type(expected_size_bytes) is not int or expected_size_bytes < 1:
        raise PostTrainingError(
            "verified-copy receipt size must be a positive integer"
        )
    if (
        source.stat().st_size != expected_size_bytes
        or sha256_file(source) != source_sha
    ):
        raise PostTrainingError("returned verified-copy receipt readback differs")
    _validate_verified_copy_receipt(plan, run_root, source)
    target = _validated_outputs(plan, run_root, create_parent=True)[
        "verified_copy_receipt"
    ]
    if os.path.lexists(target):
        raise FileExistsError(target)
    _copy_file_new(
        source,
        target,
        expected_sha256=source_sha,
        expected_size_bytes=expected_size_bytes,
    )
    return _validate_verified_copy_receipt(plan, run_root, target)


def _run_remote_arm(plan: Mapping[str, object], arm: str) -> dict[str, object]:
    remote = plan["remote"]
    contract = plan["contract"]
    assert isinstance(remote, Mapping) and isinstance(contract, Mapping)
    behavior = ARM_TO_BEHAVIOR[arm]
    eval_behavior = ARM_TO_EVAL_BEHAVIOR[arm]
    if arm == "native_lightnav":
        checkpoint_path = "-"
        checkpoint_sha = "-"
    elif arm == "a3_ppo":
        checkpoint_path = str(remote["a3_checkpoint"])
        checkpoint_sha = str(contract["a3_checkpoint_sha256"])
    else:
        checkpoint_path = str(remote["rlt_checkpoint"])
        checkpoint_sha = str(plan["stage2"]["checkpoint"]["sha256"])
    wrapper_arguments = [
        "bash",
        str(remote["wrapper"]),
        str(remote["run_root"]),
        behavior,
        "18531",
        "18532",
        str(plan["collection_manifest"]["sha256"]),
        str(plan["split_manifest"]["sha256"]),
        str(plan["evaluation_plan"]["sha256"]),
        str(plan["evaluation_code"]["inventory_sha256"]),
        checkpoint_path,
        checkpoint_sha,
        str(plan["stage2"]["context_source_sha256"]),
    ]
    for item in wrapper_arguments:
        if REMOTE_TOKEN_RE.fullmatch(item) is None:
            raise PostTrainingError("remote evaluation command contains unsafe characters")
    remote_command = " ".join(shlex.quote(item) for item in wrapper_arguments)
    command = ["ssh", "--", str(remote["host"]), remote_command]
    _run(command, timeout=24 * 3600, name=f"{arm} remote evaluation")
    output_dir = f"{remote['evaluation_root']}/{behavior}"
    response = _ssh_json(
        str(remote["host"]),
        REMOTE_RESULT,
        [
            str(remote["run_root"]),
            str(plan["run_id"]),
            behavior,
            eval_behavior,
            output_dir,
            str(plan["collection_manifest"]["sha256"]),
            str(plan["split_manifest"]["sha256"]),
            str(plan["evaluation_plan"]["sha256"]),
            str(plan["evaluation_code"]["inventory_sha256"]),
            checkpoint_sha,
        ],
        timeout=300,
        name=f"{arm} remote result verification",
    )
    if response.get("status") != "verified":
        raise PostTrainingError(f"{arm} remote result did not verify")
    return response


def _validate_returned_arm(
    plan: Mapping[str, object], arm: str, evaluation: Path, completion: Path
) -> dict[str, object]:
    evaluation = Path(evaluation).resolve(strict=True)
    completion = Path(completion).resolve(strict=True)
    if not evaluation.is_file():
        raise PostTrainingError(f"returned {arm} evaluation is absent or linked")
    if not completion.is_file():
        raise PostTrainingError(f"returned {arm} completion is absent or linked")
    evaluation_sha = sha256_file(evaluation)
    completion_sha = sha256_file(completion)
    receipt = _load_json(evaluation, f"{arm} evaluation")
    marker = _load_json(completion, f"{arm} completion")
    episodes = receipt.get("episodes")
    eval_behavior = ARM_TO_EVAL_BEHAVIOR[arm]
    expected_checkpoint = (
        None
        if arm == "native_lightnav"
        else (
            plan["contract"]["a3_checkpoint_sha256"]
            if arm == "a3_ppo"
            else plan["stage2"]["checkpoint"]["sha256"]
        )
    )
    if (
        receipt.get("schema_name") != "navigation_rlt_matched_train_development_eval"
        or receipt.get("schema_version") != 1
        or receipt.get("status") != "completed"
        or receipt.get("run_id") != plan["run_id"]
        or receipt.get("behavior") != eval_behavior
        or receipt.get("split_label") != "train_development"
        or receipt.get("upstream_split_read") != "train"
        or receipt.get("validation_split_reads") != 0
        or receipt.get("training_replay_written") is not False
        or receipt.get("collection_manifest_sha256")
        != plan["collection_manifest"]["sha256"]
        or receipt.get("run_manifest_sha256")
        != plan["collection_manifest"]["sha256"]
        or receipt.get("evaluation_plan", {}).get("sha256")
        != plan["evaluation_plan"]["sha256"]
        or receipt.get("evaluation_code", {}).get("inventory_sha256")
        != plan["evaluation_code"]["inventory_sha256"]
        or receipt.get("evaluation_plan", {}).get("split_manifest_sha256")
        != plan["split_manifest"]["sha256"]
        or receipt.get("executor", {}).get("max_physical_steps") != MAX_STEPS
        or receipt.get("seed") != EVAL_SEED
        or not isinstance(episodes, list)
        or len(episodes) != 32
        or any(not isinstance(row, Mapping) for row in episodes)
        or marker.get("schema_name")
        != "navigation_rlt_matched_train_development_eval"
        or marker.get("schema_version") != 1
        or marker.get("status") != "completed"
        or marker.get("behavior") != eval_behavior
        or marker.get("collection_manifest_sha256")
        != plan["collection_manifest"]["sha256"]
        or marker.get("evaluation_code", {}).get("inventory_sha256")
        != plan["evaluation_code"]["inventory_sha256"]
        or marker.get("evaluation_sha256") != evaluation_sha
        or marker.get("training_replay_written") is not False
    ):
        raise PostTrainingError(f"returned {arm} evaluation contract differs")
    if expected_checkpoint is not None and receipt.get("behavior_source", {}).get(
        "checkpoint_sha256"
    ) != expected_checkpoint:
        raise PostTrainingError(f"returned {arm} checkpoint identity differs")
    if arm == "navigation_rlt" and receipt.get("context", {}).get("source_sha256") != plan["stage2"]["context_source_sha256"]:
        raise PostTrainingError("returned Navigation-RLT context identity differs")
    identities = [
        [
            row.get("episode_order"),
            row.get("scene"),
            row.get("episode_id"),
            row.get("instruction_text_sha256"),
        ]
        for row in episodes
    ]
    identity_sha = _object_sha256(identities)
    if identity_sha != plan["evaluation_plan"]["ordered_episode_identity_sha256"]:
        raise PostTrainingError(f"returned {arm} episode order/identity differs")
    return {
        "arm": arm,
        "behavior": eval_behavior,
        "arm_config_sha256": plan["contract"]["arm_configs"][arm]["sha256"],
        "evaluation": {
            "path": str(evaluation),
            "sha256": evaluation_sha,
            "size_bytes": evaluation.stat().st_size,
        },
        "completion": {
            "path": str(completion),
            "sha256": completion_sha,
            "size_bytes": completion.stat().st_size,
        },
        "ordered_episode_identity_sha256": identity_sha,
    }


def _publish_local_returned_arm(
    plan: Mapping[str, object],
    run_root: Path,
    arm: str,
    evaluation_source: Path,
    completion_source: Path,
    *,
    expected_evaluation_sha256: str,
    expected_evaluation_size_bytes: int,
    expected_completion_sha256: str,
    expected_completion_size_bytes: int,
) -> dict[str, object]:
    evaluation = _regular_file(evaluation_source, run_root, f"{arm} evaluation source")
    completion = _regular_file(completion_source, run_root, f"{arm} completion source")
    for path, expected_sha, expected_size, name in (
        (evaluation, expected_evaluation_sha256, expected_evaluation_size_bytes, "evaluation"),
        (completion, expected_completion_sha256, expected_completion_size_bytes, "completion"),
    ):
        if path.stat().st_size != expected_size or sha256_file(path) != _sha256(
            expected_sha, f"{arm} {name} SHA-256"
        ):
            raise PostTrainingError(f"returned {arm} {name} readback differs")
    _validate_returned_arm(plan, arm, evaluation, completion)
    behavior = ARM_TO_BEHAVIOR[arm]
    local_root = (
        _validated_outputs(plan, run_root, create_parent=True)["root"]
        / "evaluations"
        / behavior
    )
    _safe_directory_chain(
        run_root, local_root, f"{arm} returned evaluation", create=True
    )
    evaluation_target = local_root / "evaluation.json"
    completion_target = local_root / "COMPLETED.json"
    if os.path.lexists(completion_target):
        raise FileExistsError(completion_target)
    if os.path.lexists(evaluation_target):
        if (
            evaluation_target.is_symlink()
            or not evaluation_target.is_file()
            or evaluation_target.stat().st_size != expected_evaluation_size_bytes
            or sha256_file(evaluation_target) != expected_evaluation_sha256
        ):
            raise PostTrainingError("existing canonical evaluation differs")
        _validate_returned_arm(plan, arm, evaluation_target, completion)
    else:
        _copy_file_new(
            evaluation,
            evaluation_target,
            expected_sha256=expected_evaluation_sha256,
            expected_size_bytes=expected_evaluation_size_bytes,
        )
    _copy_file_new(
        completion,
        completion_target,
        expected_sha256=expected_completion_sha256,
        expected_size_bytes=expected_completion_size_bytes,
    )
    return _validate_returned_arm(plan, arm, evaluation_target, completion_target)


def _return_arm(
    plan: Mapping[str, object], run_root: Path, arm: str, remote_result: Mapping[str, object]
) -> dict[str, object]:
    remote = plan["remote"]
    assert isinstance(remote, Mapping)
    evaluation_sha = _sha256(
        remote_result.get("evaluation_sha256"), f"{arm} evaluation SHA-256"
    )
    completion_sha = _sha256(
        remote_result.get("completion_sha256"), f"{arm} completion SHA-256"
    )
    output_root = _validated_outputs(plan, run_root, create_parent=True)["root"]
    completion_target = (
        output_root / "evaluations" / ARM_TO_BEHAVIOR[arm] / "COMPLETED.json"
    )
    if os.path.lexists(completion_target):
        raise FileExistsError(completion_target)
    with tempfile.TemporaryDirectory(prefix=f".{arm}-return-", dir=output_root) as directory:
        staging = Path(directory)
        evaluation = _scp_from(
            str(remote["host"]),
            str(remote_result["evaluation_path"]),
            staging / "evaluation.json",
            expected_sha256=evaluation_sha,
            expected_size_bytes=int(remote_result["evaluation_size_bytes"]),
            timeout=3600,
            name=f"{arm} evaluation return",
            containment_root=run_root,
        )
        completion = _scp_from(
            str(remote["host"]),
            str(remote_result["completion_path"]),
            staging / "COMPLETED.json",
            expected_sha256=completion_sha,
            expected_size_bytes=int(remote_result["completion_size_bytes"]),
            timeout=300,
            name=f"{arm} completion return",
            containment_root=run_root,
        )
        return _publish_local_returned_arm(
            plan,
            run_root,
            arm,
            evaluation,
            completion,
            expected_evaluation_sha256=evaluation_sha,
            expected_evaluation_size_bytes=int(remote_result["evaluation_size_bytes"]),
            expected_completion_sha256=completion_sha,
            expected_completion_size_bytes=int(remote_result["completion_size_bytes"]),
        )


def _handoff_value(
    plan: Mapping[str, object],
    outputs: Sequence[Mapping[str, object]],
    *,
    completed_unix_s: float,
) -> dict[str, object]:
    return {
        "schema_name": HANDOFF_SCHEMA,
        "schema_version": 1,
        "status": "verified",
        "run_id": plan["run_id"],
        "collection_manifest_sha256": plan["collection_manifest"]["sha256"],
        "split_manifest_sha256": plan["split_manifest"]["sha256"],
        "evaluation_plan_sha256": plan["evaluation_plan"]["sha256"],
        "ordered_episode_identity_sha256": plan["evaluation_plan"][
            "ordered_episode_identity_sha256"
        ],
        "evaluation_code_commit": plan["evaluation_code"]["commit"],
        "evaluation_code_inventory_sha256": plan["evaluation_code"][
            "inventory_sha256"
        ],
        "navigation_rlt_checkpoint_sha256": plan["stage2"]["checkpoint"][
            "sha256"
        ],
        "max_physical_steps": MAX_STEPS,
        "seed": EVAL_SEED,
        "expected_protocol_sha256": plan["contract"][
            "expected_protocol_sha256"
        ],
        "expected_model_sha256": plan["contract"]["expected_model_sha256"],
        "arm_configs": plan["contract"]["arm_configs"],
        "arm_order": list(ARMS),
        "arms": list(outputs),
        "completed_unix_s": completed_unix_s,
    }


def _write_handoff(
    plan: Mapping[str, object],
    run_root: Path,
    outputs: Sequence[Mapping[str, object]],
) -> tuple[Path, str]:
    path = _validated_outputs(plan, run_root, create_parent=True)[
        "evaluation_handoff_receipt"
    ]
    value = _handoff_value(plan, outputs, completed_unix_s=time.time())
    digest = _atomic_json_new(path, value)
    return path.resolve(strict=True), digest


def _validate_handoff(
    plan: Mapping[str, object],
    run_root: Path,
    outputs: Sequence[Mapping[str, object]],
    path: Path,
) -> tuple[Path, dict[str, object], str]:
    expected_path = _validated_outputs(plan, run_root, create_parent=False)[
        "evaluation_handoff_receipt"
    ]
    resolved = _regular_file(path, run_root, "evaluation handoff receipt")
    if resolved != expected_path:
        raise PostTrainingError("evaluation handoff path changed")
    value = _load_json(resolved, "evaluation handoff receipt")
    completed = value.get("completed_unix_s")
    if (
        isinstance(completed, bool)
        or not isinstance(completed, (int, float))
        or completed < 0
        or value
        != _handoff_value(plan, outputs, completed_unix_s=float(completed))
    ):
        raise PostTrainingError("evaluation handoff identity/contract differs")
    return resolved, value, sha256_file(resolved)


def _reuse_or_write_handoff(
    plan: Mapping[str, object],
    run_root: Path,
    outputs: Sequence[Mapping[str, object]],
) -> tuple[Path, dict[str, object], str]:
    path = _validated_outputs(plan, run_root, create_parent=True)[
        "evaluation_handoff_receipt"
    ]
    if not os.path.lexists(path):
        _write_handoff(plan, run_root, outputs)
    return _validate_handoff(plan, run_root, outputs, path)


def _canonical_returned_arms(
    plan: Mapping[str, object], run_root: Path
) -> list[dict[str, object]]:
    output_root = _validated_outputs(plan, run_root, create_parent=False)["root"]
    returned: list[dict[str, object]] = []
    for arm in ARMS:
        arm_root = output_root / "evaluations" / ARM_TO_BEHAVIOR[arm]
        _safe_directory_chain(
            run_root, arm_root, f"{arm} canonical evaluation", create=False
        )
        evaluation_path = arm_root / "evaluation.json"
        completion_path = arm_root / "COMPLETED.json"
        evaluation = _regular_file(
            evaluation_path, run_root, f"{arm} canonical evaluation"
        )
        completion = _regular_file(
            completion_path, run_root, f"{arm} canonical completion"
        )
        if evaluation != evaluation_path or completion != completion_path:
            raise PostTrainingError(f"{arm} canonical result path changed")
        returned.append(_validate_returned_arm(plan, arm, evaluation, completion))
    return returned


def _validate_matched_identity(
    plan: Mapping[str, object], matched: Mapping[str, object]
) -> None:
    if (
        matched.get("schema_name") != MATCHED_SCHEMA
        or matched.get("status") != "completed"
        or matched.get("run_id") != plan["run_id"]
        or matched.get("collection_manifest_sha256")
        != plan["collection_manifest"]["sha256"]
        or matched.get("evaluation_code_inventory_sha256")
        != plan["evaluation_code"]["inventory_sha256"]
        or matched.get("navigation_rlt_checkpoint_sha256")
        != plan["stage2"]["checkpoint"]["sha256"]
    ):
        raise PostTrainingError("matched comparison identity differs")


def _invoke_compare_outputs(
    plan: Mapping[str, object],
    run_root: Path,
    handoff: Mapping[str, object],
    output_json: Path,
    output_markdown: Path,
) -> tuple[Path, str, Path, str]:
    evaluation_code = plan["evaluation_code"]
    code_root = Path(str(evaluation_code["source_root"])) if "source_root" in evaluation_code else Path(__file__).resolve(strict=True).parent
    arms = {row["arm"]: row for row in handoff["arms"]}
    command = [
        sys.executable,
        str(code_root / "navigation_rlt_compare.py"),
        "--native",
        str(arms["native_lightnav"]["evaluation"]["path"]),
        "--a3",
        str(arms["a3_ppo"]["evaluation"]["path"]),
        "--navigation-rlt",
        str(arms["navigation_rlt"]["evaluation"]["path"]),
        "--expected-run-id",
        str(plan["run_id"]),
        "--expected-run-manifest-sha256",
        str(plan["collection_manifest"]["sha256"]),
        "--expected-eval-plan-sha256",
        str(plan["evaluation_plan"]["sha256"]),
        "--expected-split-manifest-sha256",
        str(plan["split_manifest"]["sha256"]),
        "--expected-protocol-sha256",
        str(plan["contract"]["expected_protocol_sha256"]),
        "--expected-model-sha256",
        str(plan["contract"]["expected_model_sha256"]),
        "--expected-lightnav-weight-sha256",
        str(plan["contract"]["model_identity"]["lightnav_weight_sha256"]),
        "--expected-a3-checkpoint-sha256",
        str(plan["contract"]["a3_checkpoint_sha256"]),
        "--expected-rlt-checkpoint-sha256",
        str(plan["stage2"]["checkpoint"]["sha256"]),
        "--expected-seed",
        str(EVAL_SEED),
        "--evaluation-code-inventory",
        str(plan["evaluation_code"]["inventory_path"]),
        "--expected-evaluation-code-inventory-sha256",
        str(plan["evaluation_code"]["inventory_sha256"]),
        "--output-json",
        str(output_json),
        "--output-markdown",
        str(output_markdown),
    ]
    _run(command, timeout=3600, name="matched comparator")
    resolved_json = _regular_file(output_json, run_root, "matched comparison JSON")
    resolved_markdown = _regular_file(
        output_markdown, run_root, "matched comparison Markdown"
    )
    _validate_matched_identity(plan, _load_json(resolved_json, "matched comparison"))
    return (
        resolved_json,
        sha256_file(resolved_json),
        resolved_markdown,
        sha256_file(resolved_markdown),
    )


def _invoke_compare(
    plan: Mapping[str, object], run_root: Path, handoff: Mapping[str, object]
) -> tuple[Path, str, Path, str]:
    outputs = _validated_outputs(plan, run_root, create_parent=True)
    return _invoke_compare_outputs(
        plan,
        run_root,
        handoff,
        outputs["matched_result"],
        outputs["matched_markdown"],
    )


def _reuse_or_invoke_compare(
    plan: Mapping[str, object], run_root: Path, handoff: Mapping[str, object]
) -> tuple[Path, str, Path, str]:
    outputs = _validated_outputs(plan, run_root, create_parent=True)
    output_json = outputs["matched_result"]
    output_markdown = outputs["matched_markdown"]
    json_exists = os.path.lexists(output_json)
    markdown_exists = os.path.lexists(output_markdown)
    if json_exists != markdown_exists:
        raise PostTrainingError("matched comparison outputs are incomplete")
    if not json_exists:
        return _invoke_compare(plan, run_root, handoff)

    resolved_json = _regular_file(
        output_json, run_root, "existing matched comparison JSON"
    )
    resolved_markdown = _regular_file(
        output_markdown, run_root, "existing matched comparison Markdown"
    )
    if resolved_json != output_json or resolved_markdown != output_markdown:
        raise PostTrainingError("matched comparison output path changed")
    _validate_matched_identity(
        plan, _load_json(resolved_json, "existing matched comparison")
    )
    json_identity = _file_identity(resolved_json.lstat())
    markdown_identity = _file_identity(resolved_markdown.lstat())
    json_sha = sha256_file(resolved_json)
    markdown_sha = sha256_file(resolved_markdown)
    with tempfile.TemporaryDirectory(
        prefix=".matched-revalidate-", dir=outputs["root"]
    ) as temporary:
        regenerated_json = Path(temporary) / "matched_comparison.json"
        regenerated_markdown = Path(temporary) / "matched_comparison.md"
        _, regenerated_json_sha, _, regenerated_markdown_sha = (
            _invoke_compare_outputs(
                plan,
                run_root,
                handoff,
                regenerated_json,
                regenerated_markdown,
            )
        )
        if (
            regenerated_json_sha != json_sha
            or regenerated_markdown_sha != markdown_sha
        ):
            raise PostTrainingError("existing matched comparison differs")
    if (
        _file_identity(resolved_json.lstat()) != json_identity
        or _file_identity(resolved_markdown.lstat()) != markdown_identity
        or sha256_file(resolved_json) != json_sha
        or sha256_file(resolved_markdown) != markdown_sha
    ):
        raise PostTrainingError("existing matched comparison changed during validation")
    return resolved_json, json_sha, resolved_markdown, markdown_sha


def _invoke_archive(
    plan: Mapping[str, object],
    plan_path: Path,
    plan_sha: str,
    matched_path: Path,
    matched_sha: str,
    matched_markdown_path: Path,
    matched_markdown_sha: str,
    copy_path: Path,
    copy_sha: str,
    handoff_path: Path,
    handoff_sha: str,
) -> None:
    code_root = Path(__file__).resolve(strict=True).parent
    command = [
        sys.executable,
        str(code_root / "navigation_rlt_obs_archive.py"),
        "--run-root",
        str(plan["a800_run_root"]),
        "--run-id",
        str(plan["run_id"]),
        "--post-training-plan",
        str(plan_path),
        "--expected-post-training-plan-sha256",
        plan_sha,
        "--evaluation-plan",
        str(plan["evaluation_plan"]["path"]),
        "--expected-evaluation-plan-sha256",
        str(plan["evaluation_plan"]["sha256"]),
        "--split-manifest",
        str(plan["split_manifest"]["path"]),
        "--expected-split-manifest-sha256",
        str(plan["split_manifest"]["sha256"]),
        "--matched-result",
        str(matched_path),
        "--expected-matched-result-sha256",
        matched_sha,
        "--matched-markdown",
        str(matched_markdown_path),
        "--expected-matched-markdown-sha256",
        matched_markdown_sha,
        "--collection-manifest",
        str(plan["collection_manifest"]["path"]),
        "--expected-collection-manifest-sha256",
        str(plan["collection_manifest"]["sha256"]),
        "--evaluation-code-inventory",
        str(plan["evaluation_code"]["inventory_path"]),
        "--expected-evaluation-code-inventory-sha256",
        str(plan["evaluation_code"]["inventory_sha256"]),
        "--training-code-verification",
        str(plan["training_code_verification"]["path"]),
        "--expected-training-code-verification-sha256",
        str(plan["training_code_verification"]["sha256"]),
        "--training-protocol",
        str(plan["training_protocol"]["path"]),
        "--expected-training-protocol-sha256",
        str(plan["training_protocol"]["sha256"]),
    ]
    repair_verification = plan.get("stage2_repair_code_verification")
    repair_protocol = plan.get("stage2_repair_protocol")
    if repair_verification is not None or repair_protocol is not None:
        if not isinstance(repair_verification, Mapping) or not isinstance(
            repair_protocol, Mapping
        ):
            raise PostTrainingError("post-training plan repair lineage is incomplete")
        command.extend(
            [
                "--stage2-repair-code-verification",
                str(repair_verification["path"]),
                "--expected-stage2-repair-code-verification-sha256",
                str(repair_verification["sha256"]),
                "--stage2-repair-protocol",
                str(repair_protocol["path"]),
                "--expected-stage2-repair-protocol-sha256",
                str(repair_protocol["sha256"]),
            ]
        )
    stage2 = plan["stage2"]
    if "formal_completion" in stage2:
        command.extend([
            "--stage2-formal-completion", str(stage2["formal_completion"]["path"]),
            "--expected-stage2-formal-completion-sha256", str(stage2["formal_completion"]["sha256"]),
            "--stage2-source-root", str(stage2["successor_source_identity"]["source_root"]),
            "--evaluation-source-root", str(plan["evaluation_code"]["source_root"]),
        ])
    command.extend(
        [
            "--verified-copy-receipt",
            str(copy_path),
            "--expected-verified-copy-receipt-sha256",
            copy_sha,
            "--evaluation-handoff-receipt",
            str(handoff_path),
            "--expected-evaluation-handoff-receipt-sha256",
            handoff_sha,
            "--obs-root",
            str(plan["obs_root"]),
        ]
    )
    _run(command, timeout=4 * 3600, name="OBS archive")


def _write_post_training_completion(
    plan: Mapping[str, object],
    run_root: Path,
    plan_path: Path,
    plan_sha: str,
    copy_path: Path,
    copy_sha: str,
    handoff_path: Path,
    handoff_sha: str,
    matched_path: Path,
    matched_sha: str,
) -> dict[str, object]:
    completion = {
        "schema_name": COMPLETION_SCHEMA,
        "schema_version": 1,
        "status": "completed",
        "run_id": plan["run_id"],
        "plan": {"path": str(plan_path), "sha256": plan_sha},
        "verified_copy_receipt": {"path": str(copy_path), "sha256": copy_sha},
        "evaluation_handoff_receipt": {
            "path": str(handoff_path),
            "sha256": handoff_sha,
        },
        "matched_result": {"path": str(matched_path), "sha256": matched_sha},
        "completed_unix_s": time.time(),
    }
    source_identities = plan.get("training_source_identities")
    if source_identities is not None:
        if not isinstance(source_identities, Mapping):
            raise PostTrainingError("post-training plan source identities are invalid")
        completion["training_source_identities"] = dict(source_identities)
    completion_path = _validated_outputs(plan, run_root, create_parent=True)[
        "completion"
    ]
    completion_sha = _atomic_json_new(completion_path, completion)
    return {**completion, "completion_sha256": completion_sha}


def _command_plan(args: argparse.Namespace) -> tuple[Path, dict[str, object], str, Path]:
    raw = Path(args.plan).expanduser().absolute()
    path, plan, digest = _load_plan(args.plan, args.expected_plan_sha256)
    run_root, _ = _revalidate_plan(plan)
    _validate_plan_location(raw, path, run_root)
    return path, plan, digest, run_root


def _new_command_output(path: Path, run_root: Path, name: str) -> Path:
    output = Path(path).expanduser().absolute()
    _safe_directory_chain(run_root, output.parent, name, create=True)
    if os.path.lexists(output):
        raise FileExistsError(output)
    return output


def export_runtime(args: argparse.Namespace) -> dict[str, object]:
    plan_path, plan, plan_sha, run_root = _command_plan(args)
    output = _new_command_output(args.output, run_root, "runtime export")
    receipt_path = _new_command_output(args.receipt, run_root, "runtime export receipt")
    if output == receipt_path:
        raise PostTrainingError("runtime export archive and receipt paths must differ")
    archive = _build_evaluation_runtime_archive(plan, run_root, output)
    receipt = {
        "schema_name": "navigation_rlt_evaluation_runtime_export",
        "schema_version": 1,
        "status": "verified",
        "run_id": plan["run_id"],
        "post_training_plan": {"path": str(plan_path), "sha256": plan_sha},
        "evaluation_plan_sha256": plan["evaluation_plan"]["sha256"],
        "evaluation_code_inventory_sha256": plan["evaluation_code"]["inventory_sha256"],
        "validation_split_reads": 0,
        "archive": archive,
    }
    receipt_sha = _atomic_json_new(receipt_path, receipt)
    return {**receipt, "receipt": {"path": str(receipt_path), "sha256": receipt_sha}}


def publish_returned_arm(args: argparse.Namespace) -> dict[str, object]:
    plan_path, plan, plan_sha, run_root = _command_plan(args)
    receipt_path = _new_command_output(args.receipt, run_root, "returned-arm receipt")
    row = _publish_local_returned_arm(
        plan,
        run_root,
        args.arm,
        args.evaluation,
        args.completion,
        expected_evaluation_sha256=args.expected_evaluation_sha256,
        expected_evaluation_size_bytes=args.expected_evaluation_size_bytes,
        expected_completion_sha256=args.expected_completion_sha256,
        expected_completion_size_bytes=args.expected_completion_size_bytes,
    )
    receipt = {
        "schema_name": "navigation_rlt_returned_arm_publication",
        "schema_version": 1,
        "status": "verified",
        "run_id": plan["run_id"],
        "post_training_plan": {"path": str(plan_path), "sha256": plan_sha},
        "validation_split_reads": 0,
        "result": row,
    }
    receipt_sha = _atomic_json_new(receipt_path, receipt)
    return {**receipt, "receipt": {"path": str(receipt_path), "sha256": receipt_sha}}


def finalize_remote_copy(args: argparse.Namespace) -> dict[str, object]:
    return _finalize_remote_copy(
        args.run_root,
        args.run_id,
        args.source_path,
        args.expected_checkpoint_sha256,
        args.expected_checkpoint_size_bytes,
        args.temporary,
        args.final,
        args.receipt,
    )


def publish_returned_copy(args: argparse.Namespace) -> dict[str, object]:
    plan_path, plan, plan_sha, run_root = _command_plan(args)
    receipt_path = _new_command_output(
        args.receipt, run_root, "returned verified-copy publication receipt"
    )
    canonical = _validated_outputs(plan, run_root, create_parent=True)[
        "verified_copy_receipt"
    ]
    if receipt_path == canonical:
        raise PostTrainingError(
            "publication receipt and canonical verified-copy receipt must differ"
        )
    copy_path, _, copy_sha = _publish_local_verified_copy_receipt(
        plan,
        run_root,
        args.verified_copy,
        expected_sha256=args.expected_verified_copy_sha256,
        expected_size_bytes=args.expected_verified_copy_size_bytes,
    )
    receipt = {
        "schema_name": "navigation_rlt_returned_copy_publication",
        "schema_version": 1,
        "status": "verified",
        "run_id": plan["run_id"],
        "post_training_plan": {"path": str(plan_path), "sha256": plan_sha},
        "validation_split_reads": 0,
        "result": {
            "path": str(copy_path),
            "sha256": copy_sha,
            "size_bytes": copy_path.stat().st_size,
        },
    }
    receipt_sha = _atomic_json_new(receipt_path, receipt)
    return {**receipt, "receipt": {"path": str(receipt_path), "sha256": receipt_sha}}


def finalize_relay(args: argparse.Namespace) -> dict[str, object]:
    plan_path, plan, plan_sha, run_root = _command_plan(args)
    outputs = _validated_outputs(plan, run_root, create_parent=True)
    if os.path.lexists(outputs["completion"]):
        raise FileExistsError(outputs["completion"])
    handoff_exists = os.path.lexists(outputs["evaluation_handoff_receipt"])
    matched_json_exists = os.path.lexists(outputs["matched_result"])
    matched_markdown_exists = os.path.lexists(outputs["matched_markdown"])
    if matched_json_exists != matched_markdown_exists:
        raise PostTrainingError("matched comparison outputs are incomplete")
    if matched_json_exists and not handoff_exists:
        raise PostTrainingError(
            "matched comparison exists without an evaluation handoff"
        )

    copy_path, _, copy_sha = _validate_verified_copy_receipt(
        plan, run_root, outputs["verified_copy_receipt"]
    )
    returned = _canonical_returned_arms(plan, run_root)
    handoff_path, handoff, handoff_sha = _reuse_or_write_handoff(
        plan, run_root, returned
    )
    (
        matched_path,
        matched_sha,
        matched_markdown_path,
        matched_markdown_sha,
    ) = _reuse_or_invoke_compare(plan, run_root, handoff)
    _invoke_archive(
        plan,
        plan_path,
        plan_sha,
        matched_path,
        matched_sha,
        matched_markdown_path,
        matched_markdown_sha,
        copy_path,
        copy_sha,
        handoff_path,
        handoff_sha,
    )
    return _write_post_training_completion(
        plan,
        run_root,
        plan_path,
        plan_sha,
        copy_path,
        copy_sha,
        handoff_path,
        handoff_sha,
        matched_path,
        matched_sha,
    )


def execute_plan(args: argparse.Namespace) -> dict[str, object]:
    raw_plan_path = Path(args.plan).expanduser().absolute()
    plan_path, plan, plan_sha = _load_plan(
        args.plan, args.expected_plan_sha256
    )
    run_root, run_id = _revalidate_plan(plan)
    _validate_plan_location(raw_plan_path, plan_path, run_root)
    outputs = _validated_outputs(plan, run_root, create_parent=True)
    for name, path in outputs.items():
        if name == "root":
            continue
        if os.path.lexists(path):
            raise FileExistsError(path)
    _deploy_evaluation_runtime(plan, run_root)
    _remote_preflight(plan)
    copy_path, _, copy_sha = _transfer_checkpoint(plan, run_root)
    returned: list[dict[str, object]] = []
    for arm in ARMS:
        remote_result = _run_remote_arm(plan, arm)
        returned.append(_return_arm(plan, run_root, arm, remote_result))
    handoff_path, handoff_sha = _write_handoff(plan, run_root, returned)
    handoff = _load_json(handoff_path, "evaluation handoff")
    (
        matched_path,
        matched_sha,
        matched_markdown_path,
        matched_markdown_sha,
    ) = _invoke_compare(plan, run_root, handoff)
    _invoke_archive(
        plan,
        plan_path,
        plan_sha,
        matched_path,
        matched_sha,
        matched_markdown_path,
        matched_markdown_sha,
        copy_path,
        copy_sha,
        handoff_path,
        handoff_sha,
    )
    return _write_post_training_completion(
        plan,
        run_root,
        plan_path,
        plan_sha,
        copy_path,
        copy_sha,
        handoff_path,
        handoff_sha,
        matched_path,
        matched_sha,
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="validate local inputs and write a reviewable plan")
    plan.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    plan.add_argument("--run-root", type=Path, required=True)
    plan.add_argument("--run-id", required=True)
    plan.add_argument("--stage2-marker", type=Path, required=True)
    plan.add_argument("--expected-stage2-marker-sha256", required=True)
    plan.add_argument("--stage2-formal-completion", type=Path)
    plan.add_argument("--expected-stage2-formal-completion-sha256")
    plan.add_argument("--stage2-source-root", type=Path)
    plan.add_argument("--evaluation-source-root", type=Path)
    plan.add_argument("--collection-manifest", type=Path, required=True)
    plan.add_argument("--expected-collection-manifest-sha256", required=True)
    plan.add_argument("--split-manifest", type=Path, required=True)
    plan.add_argument("--expected-split-manifest-sha256", required=True)
    plan.add_argument("--eval-plan", type=Path, required=True)
    plan.add_argument("--expected-eval-plan-sha256", required=True)
    plan.add_argument("--evaluation-code-inventory", type=Path, required=True)
    plan.add_argument("--expected-evaluation-code-inventory-sha256", required=True)
    plan.add_argument("--training-code-verification", type=Path, required=True)
    plan.add_argument("--expected-training-code-verification-sha256", required=True)
    plan.add_argument("--training-protocol", type=Path, required=True)
    plan.add_argument("--expected-training-protocol-sha256", required=True)
    plan.add_argument("--stage2-repair-code-verification", type=Path)
    plan.add_argument("--expected-stage2-repair-code-verification-sha256")
    plan.add_argument("--stage2-repair-protocol", type=Path)
    plan.add_argument("--expected-stage2-repair-protocol-sha256")
    plan.add_argument("--expected-a3-checkpoint-sha256", required=True)
    plan.add_argument("--expected-protocol-sha256", required=True)
    plan.add_argument("--expected-model-sha256", required=True)
    plan.add_argument("--ebk-host", required=True)
    plan.add_argument("--ebk-run-root", required=True)
    plan.add_argument("--obs-root", type=Path, default=DEFAULT_OBS_ROOT)
    plan.add_argument("--output", type=Path, required=True)

    execute = subparsers.add_parser("execute", help="execute an exact reviewed plan")
    execute.add_argument("--plan", type=Path, required=True)
    execute.add_argument("--expected-plan-sha256", required=True)
    export = subparsers.add_parser("export-runtime", help="export reviewed runtime without SSH")
    publish = subparsers.add_parser("publish-returned-arm", help="publish one relayed arm")
    remote_copy = subparsers.add_parser(
        "finalize-remote-copy", help="atomically finalize one relayed checkpoint copy"
    )
    remote_copy.add_argument("--run-root", type=Path, required=True)
    remote_copy.add_argument("--run-id", required=True)
    remote_copy.add_argument("--source-path", required=True)
    remote_copy.add_argument("--expected-checkpoint-sha256", required=True)
    remote_copy.add_argument(
        "--expected-checkpoint-size-bytes", type=int, required=True
    )
    remote_copy.add_argument("--temporary", type=Path, required=True)
    remote_copy.add_argument("--final", type=Path, required=True)
    remote_copy.add_argument("--receipt", type=Path, required=True)
    publish_copy = subparsers.add_parser(
        "publish-returned-copy", help="publish one returned verified-copy receipt"
    )
    finalize = subparsers.add_parser(
        "finalize-relay", help="finalize three canonical relayed arms without SSH"
    )
    for command in (export, publish, publish_copy, finalize):
        command.add_argument("--plan", type=Path, required=True)
        command.add_argument("--expected-plan-sha256", required=True)
    for command in (export, publish, publish_copy):
        command.add_argument("--receipt", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    publish.add_argument("--arm", choices=ARMS, required=True)
    publish.add_argument("--evaluation", type=Path, required=True)
    publish.add_argument("--completion", type=Path, required=True)
    publish.add_argument("--expected-evaluation-sha256", required=True)
    publish.add_argument("--expected-evaluation-size-bytes", type=int, required=True)
    publish.add_argument("--expected-completion-sha256", required=True)
    publish.add_argument("--expected-completion-size-bytes", type=int, required=True)
    publish_copy.add_argument("--verified-copy", type=Path, required=True)
    publish_copy.add_argument("--expected-verified-copy-sha256", required=True)
    publish_copy.add_argument(
        "--expected-verified-copy-size-bytes", type=int, required=True
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "plan":
        value = build_plan(args)
        print(
            json.dumps(
                {
                    "status": value["status"],
                    "run_id": value["run_id"],
                    "output": str(Path(args.output).resolve(strict=True)),
                    "output_sha256": sha256_file(Path(args.output)),
                },
                sort_keys=True,
            )
        )
    elif args.command == "execute":
        value = execute_plan(args)
        print(json.dumps(value, sort_keys=True, allow_nan=False))
    elif args.command == "export-runtime":
        print(json.dumps(export_runtime(args), sort_keys=True, allow_nan=False))
    elif args.command == "publish-returned-arm":
        print(json.dumps(publish_returned_arm(args), sort_keys=True, allow_nan=False))
    elif args.command == "finalize-remote-copy":
        print(json.dumps(finalize_remote_copy(args), sort_keys=True, allow_nan=False))
    elif args.command == "publish-returned-copy":
        print(json.dumps(publish_returned_copy(args), sort_keys=True, allow_nan=False))
    elif args.command == "finalize-relay":
        print(json.dumps(finalize_relay(args), sort_keys=True, allow_nan=False))
    else:
        raise AssertionError(f"unhandled command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
