#!/usr/bin/env python3
"""Bounded HF parity and contextual-prefix artifact smoke for patched LightNav.

The check deliberately loads one model, creates two fresh sessions with identical
zero-RGB histories, and runs greedy inference twice.  The first request is the
unchanged LightNav path; the second opts into ``contextual_prefix_hidden_v1``.
The generated text, decoded trajectory, and decoded signals must be byte-for-byte
identical.  The feature artifact is then independently hashed and structurally
validated.  A passing run writes one atomic JSON receipt and leaves the feature
artifact in place for downstream Navigation-RLT ingestion.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping


RECEIPT_SCHEMA = "navigation_rlt.lightnav_hf_parity.v1"
DEFAULT_INSTRUCTION = "Go to the chair."
MAX_PARITY_HISTORY_FRAMES = 64
MAX_PARITY_NEW_TOKENS = 64
_HEX_RE = re.compile(r"^[0-9a-fA-F]+$")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_digest(value: Any) -> str:
    """Match LightNav feature_artifacts._array_digest exactly."""
    import numpy as np

    array = np.asarray(value)
    header = _canonical_json({"dtype": array.dtype.str, "shape": list(array.shape)})
    return _sha256_bytes(header + b"\0" + array.tobytes(order="C"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _normalise_hex(value: str, *, name: str, lengths: tuple[int, ...]) -> str:
    text = str(value).strip().lower()
    if len(text) not in lengths or _HEX_RE.fullmatch(text) is None:
        choices = "/".join(str(length) for length in lengths)
        raise ValueError(f"{name} must be a {choices}-character hexadecimal hash")
    return text


def _positive_bounded_int(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= MAX_PARITY_HISTORY_FRAMES:
        raise argparse.ArgumentTypeError(
            f"must be in [1, {MAX_PARITY_HISTORY_FRAMES}], got {parsed}"
        )
    return parsed


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _validate_model_identity(value: object) -> dict[str, Any]:
    _require(isinstance(value, dict), "model_identity must be an object")
    identity = dict(value)
    _require(identity.get("algorithm") == "sha256", "model_identity algorithm is not sha256")
    model_hash = str(identity.get("value", "")).lower()
    _require(
        len(model_hash) == 64 and _HEX_RE.fullmatch(model_hash) is not None,
        "model_identity value is not a SHA-256 digest",
    )
    _require(isinstance(identity.get("basis"), dict), "model_identity basis is missing")
    basis = identity["basis"]
    _require(bool(str(basis.get("model_class", ""))), "model_identity model_class is empty")
    return identity


def _validate_mask_ranges(mask: Any, mask_ref: Mapping[str, Any]) -> None:
    import numpy as np

    reconstructed = np.zeros(mask.shape, dtype=np.uint8)
    ranges = mask_ref.get("valid_ranges")
    _require(isinstance(ranges, list), "feature_ref mask valid_ranges is not a list")
    for item in ranges:
        _require(
            isinstance(item, list) and len(item) == 2,
            "feature_ref mask range must be [start, length]",
        )
        start, length = item
        _require(
            isinstance(start, int) and isinstance(length, int),
            "feature_ref mask range values must be integers",
        )
        _require(
            start >= 0 and length > 0 and start + length <= mask.size,
            "feature_ref mask range is out of bounds",
        )
        _require(
            not bool(reconstructed[start : start + length].any()),
            "feature_ref mask ranges overlap",
        )
        reconstructed[start : start + length] = 1
    _require(np.array_equal(reconstructed, mask.astype(np.uint8)), "mask ranges do not match mask")


def _load_and_validate_artifact(
    feature_ref: object,
    *,
    artifact_root: Path,
    engine_model_identity: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np

    _require(isinstance(feature_ref, dict), "feature request returned no feature_ref object")
    ref = dict(feature_ref)
    _require(
        ref.get("schema") == "lightnav.contextual_prefix_hidden.v1",
        "unexpected feature_ref schema",
    )
    _require(
        ref.get("request") == "contextual_prefix_hidden_v1",
        "unexpected feature_ref request",
    )
    local_ref = ref.get("ref")
    _require(isinstance(local_ref, dict), "feature_ref.ref is missing")
    _require(local_ref.get("kind") == "local_manifest", "feature ref is not a local manifest")
    manifest_path = Path(str(local_ref.get("path", ""))).expanduser().resolve()
    artifact_root = artifact_root.expanduser().resolve()
    _require(_is_within(manifest_path, artifact_root), "manifest escaped the artifact root")
    _require(manifest_path.is_file(), f"feature manifest does not exist: {manifest_path}")

    manifest_sha256 = _sha256_file(manifest_path)
    hash_ref = ref.get("hash")
    _require(isinstance(hash_ref, dict), "feature_ref.hash is missing")
    _require(hash_ref.get("algorithm") == "sha256", "feature_ref hash algorithm is not sha256")
    _require(hash_ref.get("manifest") == manifest_sha256, "feature manifest SHA-256 mismatch")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(isinstance(manifest, dict), "feature manifest is not an object")
    _require(manifest.get("schema") == ref["schema"], "manifest schema mismatch")
    _require(manifest.get("request") == ref["request"], "manifest request mismatch")
    _require(manifest.get("identity") == ref.get("identity"), "artifact identity mismatch")

    manifest_model_identity = _validate_model_identity(manifest.get("model_identity"))
    ref_model_identity = _validate_model_identity(ref.get("model_identity"))
    _require(
        manifest_model_identity == ref_model_identity == dict(engine_model_identity),
        "feature model identity does not match the loaded engine",
    )

    files = manifest.get("files")
    array_digests = manifest.get("arrays")
    _require(isinstance(files, dict) and files, "manifest files map is empty")
    _require(isinstance(array_digests, dict), "manifest arrays map is missing")
    required = {"hidden", "attention_mask", "valid_token_indices", "input_ids", "video_grid_thw"}
    _require(required.issubset(files), f"manifest is missing arrays: {sorted(required - set(files))}")
    _require(set(files) == set(array_digests), "manifest files and arrays keys differ")

    arrays: dict[str, Any] = {}
    file_receipts: dict[str, dict[str, Any]] = {}
    seen_paths: set[Path] = set()
    payload_bytes = 0
    for name in sorted(files):
        entry = files[name]
        _require(isinstance(entry, dict), f"file entry {name!r} is not an object")
        relative_name = str(entry.get("path", ""))
        _require(
            bool(relative_name) and Path(relative_name).name == relative_name,
            f"unsafe artifact filename for {name!r}",
        )
        array_path = (manifest_path.parent / relative_name).resolve()
        _require(_is_within(array_path, manifest_path.parent), f"artifact {name!r} escaped its dir")
        _require(array_path not in seen_paths, f"duplicate artifact path for {name!r}")
        seen_paths.add(array_path)
        _require(array_path.is_file(), f"artifact array does not exist: {array_path}")
        file_sha256 = _sha256_file(array_path)
        _require(file_sha256 == entry.get("sha256"), f"file SHA-256 mismatch for {name!r}")
        array = np.load(array_path, allow_pickle=False, mmap_mode="r")
        _require(array.dtype.str == entry.get("dtype"), f"dtype mismatch for {name!r}")
        _require(list(array.shape) == entry.get("shape"), f"shape mismatch for {name!r}")
        semantic_sha256 = _array_digest(array)
        _require(
            semantic_sha256 == array_digests.get(name),
            f"semantic array digest mismatch for {name!r}",
        )
        arrays[name] = array
        payload_bytes += int(array.nbytes)
        file_receipts[name] = {
            "path": str(array_path),
            "sha256": file_sha256,
            "array_sha256": semantic_sha256,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
        }

    hidden = arrays["hidden"]
    mask = arrays["attention_mask"]
    valid_indices = arrays["valid_token_indices"]
    input_ids = arrays["input_ids"]
    video_grid = arrays["video_grid_thw"]
    _require(hidden.ndim == 2 and min(hidden.shape) > 0, "hidden is not non-empty [tokens, dim]")
    _require(np.issubdtype(hidden.dtype, np.floating), "hidden is not floating point")
    _require(bool(np.isfinite(hidden).all()), "hidden contains NaN or infinity")
    _require(mask.ndim == 1 and mask.size > 0, "attention_mask is not a non-empty vector")
    _require(bool(np.isin(mask, [0, 1]).all()), "attention_mask contains values outside {0,1}")
    _require(input_ids.ndim == 1 and input_ids.shape == mask.shape, "input_ids/mask mismatch")
    _require(np.issubdtype(input_ids.dtype, np.integer), "input_ids is not integral")
    _require(valid_indices.ndim == 1, "valid_token_indices is not a vector")
    _require(np.issubdtype(valid_indices.dtype, np.integer), "valid_token_indices is not integral")
    expected_indices = np.flatnonzero(mask.astype(bool))
    _require(
        np.array_equal(valid_indices.astype(np.int64), expected_indices.astype(np.int64)),
        "valid_token_indices does not equal nonzero(attention_mask)",
    )
    _require(hidden.shape[0] == expected_indices.size, "hidden row count does not match mask")
    _require(list(hidden.shape) == ref.get("shape"), "feature_ref hidden shape mismatch")
    _require(video_grid.ndim == 2 and video_grid.shape[1] == 3, "video_grid_thw is not [N,3]")
    _require(np.issubdtype(video_grid.dtype, np.integer), "video_grid_thw is not integral")
    _require(bool((video_grid > 0).all()), "video_grid_thw contains non-positive dimensions")
    _require(payload_bytes == int(manifest.get("payload_bytes", -1)), "payload_bytes mismatch")

    mask_ref = ref.get("mask")
    _require(isinstance(mask_ref, dict), "feature_ref mask metadata is missing")
    _require(mask_ref.get("encoding") == "valid_ranges_start_length", "unknown mask encoding")
    _require(list(mask.shape) == mask_ref.get("shape"), "feature_ref mask shape mismatch")
    _require(int(mask.sum()) == int(mask_ref.get("valid_count", -1)), "mask valid_count mismatch")
    _require(
        mask_ref.get("sha256") == array_digests["attention_mask"],
        "feature_ref mask digest mismatch",
    )
    _validate_mask_ranges(mask, mask_ref)
    _require(hash_ref.get("hidden") == array_digests["hidden"], "feature_ref hidden digest mismatch")

    prompt_identity = _sha256_bytes(
        _canonical_json(
            {
                "input_ids": array_digests["input_ids"],
                "attention_mask": array_digests["attention_mask"],
                "video_grid_thw": array_digests["video_grid_thw"],
            }
        )
    )
    _require(prompt_identity == manifest.get("prompt_identity"), "manifest prompt identity mismatch")
    _require(prompt_identity == ref.get("prompt_identity"), "feature_ref prompt identity mismatch")
    identity_basis = {
        "schema": manifest["schema"],
        "request": manifest["request"],
        "model_identity": manifest_model_identity,
        "prompt_identity": prompt_identity,
        "arrays": dict(array_digests),
    }
    artifact_identity = _sha256_bytes(_canonical_json(identity_basis))
    _require(artifact_identity == manifest.get("identity"), "recomputed artifact identity mismatch")
    _require(artifact_identity == ref.get("identity"), "feature_ref artifact identity mismatch")

    return {
        "identity": artifact_identity,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "prompt_identity": prompt_identity,
        "model_identity": manifest_model_identity,
        "source_dtype": str(manifest.get("source_dtype", "")),
        "storage_dtype": str(manifest.get("storage_dtype", "")),
        "payload_bytes": payload_bytes,
        "hidden_shape": list(hidden.shape),
        "prefix_tokens": int(hidden.shape[0]),
        "full_prefix_tokens": int(mask.shape[0]),
        "files": file_receipts,
    }


def _prediction_basis(prediction: Any) -> dict[str, Any]:
    import numpy as np

    waypoints = np.asarray(prediction.waypoints)
    _require(waypoints.ndim == 2 and waypoints.shape[1] == 3, "waypoints are not [H,3]")
    _require(bool(np.isfinite(waypoints).all()), "waypoints contain NaN or infinity")
    return {
        "raw_text": str(prediction.raw_text),
        "waypoints": {
            "dtype": waypoints.dtype.str,
            "shape": list(waypoints.shape),
            "sha256": _array_digest(waypoints),
        },
        "stop": bool(prediction.stop),
        "visible": None if prediction.visible is None else bool(prediction.visible),
        "traj_id": None if prediction.traj_id is None else int(prediction.traj_id),
        "tpos_id": None if prediction.tpos_id is None else int(prediction.tpos_id),
        "apos_id": None if prediction.apos_id is None else int(prediction.apos_id),
        "opos_id": None if prediction.opos_id is None else int(prediction.opos_id),
    }


def _assert_prediction_parity(plain: Any, feature: Any) -> dict[str, Any]:
    import numpy as np

    plain_basis = _prediction_basis(plain)
    feature_basis = _prediction_basis(feature)
    _require(plain_basis["raw_text"] == feature_basis["raw_text"], "raw_text parity failed")
    plain_waypoints = np.asarray(plain.waypoints)
    feature_waypoints = np.asarray(feature.waypoints)
    _require(plain_waypoints.dtype == feature_waypoints.dtype, "trajectory dtype parity failed")
    _require(plain_waypoints.shape == feature_waypoints.shape, "trajectory shape parity failed")
    _require(np.array_equal(plain_waypoints, feature_waypoints), "trajectory value parity failed")
    for key in ("stop", "visible", "traj_id", "tpos_id", "apos_id", "opos_id"):
        _require(plain_basis[key] == feature_basis[key], f"decoded {key} parity failed")
    _require(plain_basis == feature_basis, "complete decoded output parity failed")
    output_hash = _sha256_bytes(_canonical_json(plain_basis))
    return {"hash": output_hash, **plain_basis}


async def _predict_pair(service: Any, plain_session: Any, feature_session: Any, request: str):
    started = False
    try:
        await service.start()
        started = True
        t0 = time.perf_counter()
        plain = await service.predict(plain_session)
        plain_ms = (time.perf_counter() - t0) * 1000.0
        t1 = time.perf_counter()
        feature = await service.predict(feature_session, feature_request=request)
        feature_ms = (time.perf_counter() - t1) * 1000.0
        return plain, feature, plain_ms, feature_ms
    finally:
        if started:
            await service.stop()


def _build_cli() -> argparse.ArgumentParser:
    from lightnav.serving import ws_server

    parser = ws_server._build_parser()
    parser.prog = "navigation_rlt_lightnav_parity.py"
    parser.description = (
        "One-model, two-session LightNav HF parity and contextual-prefix artifact smoke."
    )
    destinations = {action.dest for action in parser._actions}
    required_patch_args = {
        "feature_request_capability",
        "feature_artifact_dir",
        "feature_max_tokens",
        "feature_max_bytes_mib",
    }
    missing = sorted(required_patch_args - destinations)
    if missing:
        raise RuntimeError(
            "LightNav contextual-prefix patch is not installed; missing parser fields: "
            + ", ".join(missing)
        )
    parity = parser.add_argument_group("bounded Navigation-RLT parity receipt")
    parity.add_argument(
        "--source_hash",
        required=True,
        help="Audited LightNav source hash (40-char Git SHA or 64-char tree SHA-256).",
    )
    parity.add_argument(
        "--weight_sha256",
        required=True,
        help="Precomputed SHA-256 of the immutable original LightNav weight set.",
    )
    parity.add_argument(
        "--receipt_path",
        required=True,
        help="Output path for the single atomic JSON receipt (outside feature_artifact_dir).",
    )
    parity.add_argument(
        "--parity_history_frames",
        type=_positive_bounded_int,
        default=4,
        help=f"Identical zero-RGB frames per fresh session (1..{MAX_PARITY_HISTORY_FRAMES}).",
    )
    parity.add_argument(
        "--parity_instruction",
        default=DEFAULT_INSTRUCTION,
        help=f"Fixed non-empty instruction (default: {DEFAULT_INSTRUCTION!r}).",
    )
    return parser


def _atomic_write_receipt(path: Path, receipt: Mapping[str, Any]) -> bytes:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json(dict(receipt)) + b"\n"
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return payload


def _run(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import torch

    from lightnav.serving import ws_server
    from lightnav.serving.feature_artifacts import CONTEXTUAL_PREFIX_REQUEST
    from lightnav.serving.token_budget import (
        action_token_count,
        decode_token_budget,
        probe_grounding_tokens,
    )
    from lightnav.serving.tracking_service import BatchedTrackingService
    from lightnav.tracking import load_centroids

    total_t0 = time.perf_counter()
    source_hash = _normalise_hex(args.source_hash, name="--source_hash", lengths=(40, 64))
    weight_sha256 = _normalise_hex(
        args.weight_sha256, name="--weight_sha256", lengths=(64,)
    )
    instruction = str(args.parity_instruction).strip()
    _require(bool(instruction), "--parity_instruction must be non-empty")
    _require(args.backend == "hf", "parity smoke requires --backend hf")
    _require(
        args.feature_request_capability == CONTEXTUAL_PREFIX_REQUEST,
        f"pass --feature_request_capability {CONTEXTUAL_PREFIX_REQUEST}",
    )
    _require(1 <= int(args.max_new_tokens) <= MAX_PARITY_NEW_TOKENS, "max_new_tokens is unbounded")

    artifact_root = Path(args.feature_artifact_dir).expanduser().resolve()
    receipt_path = Path(args.receipt_path).expanduser().resolve()
    _require(receipt_path.suffix.lower() == ".json", "--receipt_path must end in .json")
    _require(
        not _is_within(receipt_path, artifact_root),
        "receipt_path must be outside feature_artifact_dir",
    )

    # Generation is explicitly greedy.  Seeds are also fixed so any later sampling
    # regression cannot silently weaken the parity check.
    os.environ["VLN_EVAL_TEMPERATURE"] = "0.0"
    os.environ["VLN_EVAL_TOP_P"] = "1.0"
    os.environ["VLN_EVAL_TOP_K"] = "0"
    os.environ["VLN_EVAL_TRAJ_TOP1"] = "0"
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.set_grad_enabled(False)

    ws_server.resolve_decoder_args(args)
    ws_server._validate_feature_args(args)
    feature_writer = ws_server._build_feature_writer(args)
    _require(feature_writer is not None, "feature artifact writer was not constructed")

    rvq_bundle = None
    centroids = None
    if args.action_tokenizer_bundle:
        from lightnav.traj_vocab import load_rvq_bundle

        rvq_bundle = load_rvq_bundle(
            Path(args.action_tokenizer_bundle),
            int(args.horizon),
            num_frames=0,
            load_cluster_ids=False,
        )

    load_t0 = time.perf_counter()
    engine, bundle = ws_server._build_engine(args)
    model_load_ms = (time.perf_counter() - load_t0) * 1000.0
    _require(getattr(engine, "backend", None) == "hf", "loaded engine is not the HF backend")
    _require(hasattr(engine, "_model_identity"), "patched engine model identity API is missing")
    engine_model_identity = _validate_model_identity(engine._model_identity())

    grounding_tokens = probe_grounding_tokens(bundle.tokenizer)
    decode_budget = decode_token_budget(grounding_tokens, rvq_bundle)
    final_new_tokens = max(int(args.max_new_tokens), int(decode_budget))
    _require(
        final_new_tokens <= MAX_PARITY_NEW_TOKENS,
        f"resolved decode budget {final_new_tokens} exceeds bounded maximum {MAX_PARITY_NEW_TOKENS}",
    )
    if rvq_bundle is None:
        centroids = load_centroids(args.traj_vocab_path, int(args.K), int(args.horizon))

    configured_history = int(args.num_history_frames or bundle.num_history_frames)
    history_frames = int(args.parity_history_frames)
    _require(
        history_frames <= configured_history,
        f"parity history {history_frames} exceeds configured window {configured_history}",
    )
    service = BatchedTrackingService(
        engine=engine,
        bundle=bundle,
        centroids=centroids,
        num_history_frames=configured_history,
        max_new_tokens=final_new_tokens,
        max_batch_size=1,
        max_wait_ms=0.0,
        serve_task=args.task,
        rvq_bundle=rvq_bundle,
        feature_writer=feature_writer,
    )

    height, width = (int(value) for value in bundle.video_size)
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    plain_session = service.make_session(client_id="parity-plain")
    feature_session = service.make_session(client_id="parity-feature")
    plain_session.reset(instruction=instruction)
    feature_session.reset(instruction=instruction)
    for _ in range(history_frames):
        plain_session.observe(rgb)
        feature_session.observe(rgb)
    _require(
        len(plain_session._history_frame_ids) == len(feature_session._history_frame_ids)
        == history_frames,
        "fresh sessions did not retain the requested history length",
    )
    _require(
        plain_session._history_frame_ids == feature_session._history_frame_ids,
        "fresh session frame IDs differ",
    )
    _require(
        all(
            np.array_equal(left, right)
            for left, right in zip(plain_session._history, feature_session._history)
        ),
        "fresh session RGB histories differ",
    )
    plain_video = plain_session._get_video_tensor()
    feature_video = feature_session._get_video_tensor()
    _require(torch.equal(plain_video, feature_video), "preprocessed session videos differ")
    video_array = plain_video.detach().cpu().contiguous().numpy()
    input_basis = {
        "task": str(args.task),
        "instruction": instruction,
        "instruction_sha256": _sha256_bytes(instruction.encode("utf-8")),
        "history_frames": history_frames,
        "frame_ids": list(plain_session._history_frame_ids),
        "rgb": {
            "dtype": rgb.dtype.str,
            "shape": list(rgb.shape),
            "sha256": _array_digest(rgb),
        },
        "model_video": {
            "dtype": video_array.dtype.str,
            "shape": list(video_array.shape),
            "sha256": _array_digest(video_array),
        },
        "generation": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "traj_top1": False,
            "max_new_tokens": final_new_tokens,
        },
    }
    input_identity = _sha256_bytes(_canonical_json(input_basis))

    plain, feature, plain_ms, feature_ms = asyncio.run(
        _predict_pair(
            service,
            plain_session,
            feature_session,
            CONTEXTUAL_PREFIX_REQUEST,
        )
    )
    _require(getattr(plain, "feature_ref", None) is None, "plain path emitted a feature_ref")
    output = _assert_prediction_parity(plain, feature)
    artifact = _load_and_validate_artifact(
        getattr(feature, "feature_ref", None),
        artifact_root=artifact_root,
        engine_model_identity=engine_model_identity,
    )

    feature_timings = dict(getattr(feature, "timings_ms", {}) or {})
    contextual_forward_ms = feature_timings.get("contextual_prefix_forward_ms")
    _require(
        isinstance(contextual_forward_ms, (int, float))
        and math.isfinite(float(contextual_forward_ms))
        and float(contextual_forward_ms) >= 0.0,
        "feature timing lacks finite contextual_prefix_forward_ms",
    )
    total_ms = (time.perf_counter() - total_t0) * 1000.0
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "passed",
        "source_hash": source_hash,
        "weight_sha256": weight_sha256,
        "model": {
            "path": str(Path(args.model_path).expanduser().resolve()),
            "identity": engine_model_identity,
            "load_count": 1,
        },
        "input": {"identity": input_identity, **input_basis},
        "output": output,
        "artifact": artifact,
        "parity": {
            "raw_text_exact": True,
            "trajectory_exact": True,
            "decoded_signals_exact": True,
            "plain_feature_ref_absent": True,
            "requested_feature_ref_valid": True,
        },
        "latency_ms": {
            "model_load": model_load_ms,
            "plain_request": plain_ms,
            "feature_request": feature_ms,
            "contextual_prefix_forward": float(contextual_forward_ms),
            "total": total_ms,
        },
    }
    return receipt


def main(argv: list[str] | None = None) -> int:
    try:
        parser = _build_cli()
        args = parser.parse_args(argv)
        receipt = _run(args)
        payload = _atomic_write_receipt(Path(args.receipt_path), receipt)
        sys.stdout.buffer.write(payload)
        sys.stdout.buffer.flush()
        return 0
    except Exception as exc:
        error = {
            "schema": RECEIPT_SCHEMA,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        sys.stderr.buffer.write(_canonical_json(error) + b"\n")
        sys.stderr.buffer.flush()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
