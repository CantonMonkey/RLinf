"""Opt-in physical GPU identity check for the RLinf actor and rollout workers.

Install as ``rlinf/utils/gpu_uuid_gate.py`` in the isolated RLinf checkout.
The CUDA Driver API calls below enumerate a device without creating a CUDA
context. No Ray, Torch, NVML, or Habitat module is imported by this file.

For a one-GPU smoke, set RLINF_VERIFY_WORKER_GPU_UUID=1 and
RLINF_EXPECTED_GPU_UUID=GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx in the
RLinf driver environment. Leave the enable variable unset for normal runs.
"""

from __future__ import annotations

import ctypes
import json
import os
import re
import uuid


ENABLE_ENV = "RLINF_VERIFY_WORKER_GPU_UUID"
EXPECTED_ENV = "RLINF_EXPECTED_GPU_UUID"
ALLOWED_ENV = "RLINF_ALLOWED_PHYSICAL_GPUS"
_UUID_RE = re.compile(
    r"GPU-([0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12})\Z"
)


def _canonical_gpu_uuid(raw: str | None) -> str:
    match = _UUID_RE.fullmatch(raw or "")
    if match is None:
        raise ValueError(f"{EXPECTED_ENV} must be a canonical GPU UUID")
    return f"GPU-{uuid.UUID(match.group(1))}"


def _visible_gpu_uuid_without_context() -> dict[str, object]:
    """Return CUDA device zero's UUID, proving this thread stayed context-free."""

    class CUuuid(ctypes.Structure):
        _fields_ = [("bytes", ctypes.c_ubyte * 16)]

    lib = ctypes.CDLL("libcuda.so.1")

    def bind(name: str, argtypes: list[object]):
        function = getattr(lib, name)
        function.argtypes = argtypes
        function.restype = ctypes.c_int
        return function

    def checked(function, *args) -> None:
        error = function(*args)
        if error != 0:
            raise RuntimeError(f"{function.__name__} returned CUDA error {error}")

    cu_init = bind("cuInit", [ctypes.c_uint])
    cu_count = bind("cuDeviceGetCount", [ctypes.POINTER(ctypes.c_int)])
    cu_device = bind("cuDeviceGet", [ctypes.POINTER(ctypes.c_int), ctypes.c_int])
    cu_uuid = bind("cuDeviceGetUuid_v2", [ctypes.POINTER(CUuuid), ctypes.c_int])
    cu_context = bind("cuCtxGetCurrent", [ctypes.POINTER(ctypes.c_void_p)])

    checked(cu_init, 0)
    before = ctypes.c_void_p()
    checked(cu_context, ctypes.byref(before))
    if before.value is not None:
        raise RuntimeError("CUDA context existed before worker GPU UUID check")

    count = ctypes.c_int()
    checked(cu_count, ctypes.byref(count))
    if count.value != 1:
        raise RuntimeError(f"GPU UUID gate requires exactly one visible CUDA device; saw {count.value}")
    device = ctypes.c_int()
    checked(cu_device, ctypes.byref(device), 0)
    raw_uuid = CUuuid()
    checked(cu_uuid, ctypes.byref(raw_uuid), device.value)

    after = ctypes.c_void_p()
    checked(cu_context, ctypes.byref(after))
    if after.value is not None:
        raise RuntimeError("Worker GPU UUID check created a CUDA context")
    return {
        "cuda_device_count": count.value,
        "cuda_zero_uuid": f"GPU-{uuid.UUID(bytes=bytes(raw_uuid.bytes))}",
        "context_before": None,
        "context_after": None,
    }


def verify_gpu_worker_identity(role: str) -> dict[str, object] | None:
    """Check this GPU worker before Torch chooses a device; emit one JSON row.

    The gate is deliberately called only by the training actor and rollout
    worker. CPU EnvWorker and ChannelWorker never import libcuda through it.
    """

    enabled = os.environ.get(ENABLE_ENV)
    if enabled in (None, "", "0"):
        return None

    record: dict[str, object] = {
        "event": "rlinf_worker_gpu_uuid_gate",
        "role": role,
        "group": os.environ.get("GROUP_NAME"),
        "rank": os.environ.get("RANK"),
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "visible_devices": os.environ.get("VISIBLE_DEVICES"),
        "local_rank": os.environ.get("LOCAL_RANK"),
        "allowed_physical_gpus": os.environ.get(ALLOWED_ENV),
        "expected_uuid": os.environ.get(EXPECTED_ENV),
    }
    try:
        if enabled != "1":
            raise ValueError(f"{ENABLE_ENV} must be 0 or 1")
        if role not in ("actor", "rollout"):
            raise ValueError(f"GPU UUID gate called for unexpected role {role!r}")
        expected_uuid = _canonical_gpu_uuid(os.environ.get(EXPECTED_ENV))
        record["expected_uuid"] = expected_uuid

        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not re.fullmatch(r"0|[1-9][0-9]*", visible):
            raise RuntimeError("GPU UUID gate requires one numeric CUDA_VISIBLE_DEVICES ordinal")
        if os.environ.get("LOCAL_RANK") != "0":
            raise RuntimeError("GPU UUID gate requires LOCAL_RANK=0")
        allowed = os.environ.get(ALLOWED_ENV)
        if allowed is not None:
            if allowed != visible or os.environ.get("VISIBLE_DEVICES") != "0":
                raise RuntimeError("Worker logical/physical GPU mapping disagrees with allow-list")

        device = _visible_gpu_uuid_without_context()
        record.update(device)
        if device["cuda_zero_uuid"] != expected_uuid:
            raise RuntimeError(
                f"Worker physical GPU UUID {device['cuda_zero_uuid']} != expected {expected_uuid}"
            )
        record["ok"] = True
        return record
    except Exception as exc:
        record["ok"] = False
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        print(json.dumps(record, sort_keys=True), flush=True)
