"""Check one RLinf Ray WorkerGroup's physical GPU UUID without a CUDA context.

Run as the owner of the isolated navigation project. This is a mapping
preflight, not a PPO or memory-capacity test. It starts a private Ray head,
queries one WorkerGroup via the CUDA Driver API, and stops only its own Ray
process group. The result and Ray startup log remain under ``runs/``.

Example on A800, after choosing a usable GPU window::

    python worker_uuid_preflight.py \
      --node-root /srv/navigation \
      --node-ip 192.0.2.10 --gpu-ordinal 7 \
      --expected-uuid GPU-22222222-2222-2222-2222-222222222222

This script does not verify the actor/rollout workers used by a later trainer.
"""

from __future__ import annotations

import argparse
import ctypes
import ipaddress
import json
import os
import re
import secrets
import shutil
import signal
import socket
import stat
import subprocess
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


PORTS = tuple(range(23979, 23988)) + tuple(range(24000, 24201))
HEAD_PORT = 23979
PRIVATE_NAME_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789"
UUID_PATTERN = re.compile(r"GPU-([0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12})\Z")


def gpu_ordinal(value: str) -> int:
    if not re.fullmatch(r"0|[1-9][0-9]*", value):
        raise argparse.ArgumentTypeError("GPU ordinal must be a nonnegative decimal integer")
    return int(value)


def gpu_uuid(value: str) -> str:
    match = UUID_PATTERN.fullmatch(value)
    if match is None:
        raise argparse.ArgumentTypeError("expected UUID must use canonical GPU-xxxxxxxx-... form")
    return f"GPU-{uuid.UUID(match.group(1))}"


def node_ip(value: str) -> str:
    try:
        address = ipaddress.ip_address(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("node IP must be a numeric IPv4 address") from exc
    if address.version != 4 or address.is_loopback or address.is_unspecified or address.is_multicast:
        raise argparse.ArgumentTypeError("node IP must be a non-loopback unicast IPv4 address")
    return str(address)


def select_target_gpu(csv_text: str, ordinal: int, expected_uuid: str) -> dict:
    """Fail before Ray startup if the requested host ordinal changed identity."""
    rows = {}
    for line in csv_text.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            raise ValueError(f"Unexpected nvidia-smi row: {line!r}")
        index = int(fields[0])
        if index in rows:
            raise ValueError(f"Duplicate nvidia-smi GPU index {index}")
        rows[index] = {
            "index": index,
            "uuid": gpu_uuid(fields[1]),
            "memory_used_mib": int(fields[2]),
            "utilization_percent": int(fields[3]),
        }
    target = rows.get(ordinal)
    if target is None:
        raise ValueError(f"GPU ordinal {ordinal} is absent from nvidia-smi")
    if target["uuid"] != expected_uuid:
        raise ValueError(f"GPU ordinal {ordinal} UUID changed: {target['uuid']}")
    return target


def smi_target(ordinal: int, expected_uuid: str) -> dict:
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"],
        text=True,
        timeout=10,
    )
    return select_target_gpu(raw, ordinal, expected_uuid)


def owned_private_directory(path: Path, uid: int) -> None:
    metadata = path.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != uid:
        raise RuntimeError(f"Directory is not a regular directory owned by this account: {path}")
    if metadata.st_mode & 0o077:
        raise RuntimeError(f"Directory is not private (expected mode 0700): {path}")


def prepare_directories(node_root: Path, uid: int) -> tuple[Path, Path]:
    root = node_root.expanduser().resolve(strict=True)
    metadata = root.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != uid or metadata.st_mode & 0o022:
        raise RuntimeError("Project root must be owned by this account and not group/world writable")
    runs = root / "runs"
    runs_metadata = runs.lstat()
    if not stat.S_ISDIR(runs_metadata.st_mode) or runs_metadata.st_uid != uid or runs_metadata.st_mode & 0o022:
        raise RuntimeError("Project runs directory must be owned by this account and not writable by others")
    private_parent = root / "p"  # Short enough for Ray's UNIX socket limit.
    try:
        private_parent.mkdir(mode=0o700)
    except FileExistsError:
        pass
    owned_private_directory(private_parent, uid)
    return runs, private_parent


def reserve_ray_temp(private_parent: Path, uid: int) -> tuple[Path, tuple[int, int]]:
    """Reserve a fresh two-character root; never reuse the historical /r/w."""
    owned_private_directory(private_parent, uid)
    for _attempt in range(80):
        name = "".join(secrets.choice(PRIVATE_NAME_CHARS) for _ in range(2))
        path = private_parent / name
        if len(os.fsencode(path)) > 43:
            raise RuntimeError("Project path is too long for Ray's UNIX sockets")
        try:
            path.mkdir(mode=0o700)
        except FileExistsError:
            continue
        owned_private_directory(path, uid)
        metadata = path.lstat()
        return path, (metadata.st_dev, metadata.st_ino)
    raise RuntimeError("Could not reserve a fresh short private Ray directory")


def remove_own_ray_temp(path: Path, private_parent: Path, identity: tuple[int, int], uid: int) -> None:
    """Remove only the exact private directory created by this invocation."""
    owned_private_directory(private_parent, uid)
    if path.parent != private_parent or not re.fullmatch(r"[a-z0-9]{2}", path.name):
        raise RuntimeError("Refusing cleanup outside the private Ray parent")
    metadata = path.lstat()
    if (metadata.st_dev, metadata.st_ino) != identity:
        raise RuntimeError("Private Ray directory changed identity; refusing cleanup")
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != uid:
        raise RuntimeError("Private Ray directory is not owned by this account")
    # Ray widens its own temp root (observed 0777 on ebk180). Its parent
    # remains 0700; restore the exact, inode-checked child before cleanup.
    path.chmod(0o700)
    owned_private_directory(path, uid)
    if not shutil.rmtree.avoids_symlink_attacks:
        raise RuntimeError("Safe descriptor-based directory removal is unavailable")
    shutil.rmtree(path)


def check_ports_free() -> None:
    busy = []
    for port in PORTS:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            try:
                listener.bind(("0.0.0.0", port))
            except OSError:
                busy.append(port)
    if busy:
        raise RuntimeError(f"Private Ray ports unavailable: {busy}")


def check_ports_no_listeners() -> None:
    raw = subprocess.check_output(["ss", "-H", "-lnt"], text=True, timeout=10)
    active = []
    for line in raw.splitlines():
        fields = line.split()
        if len(fields) < 4:
            continue
        try:
            port = int(fields[3].rsplit(":", 1)[-1])
        except ValueError:
            continue
        if port in PORTS:
            active.append(port)
    if active:
        raise RuntimeError(f"Private Ray ports still listening: {active}")


def ray_group_members(pgid: int) -> list[dict]:
    import pwd  # Linux-only; keep the pure preflight helpers importable on Windows.

    members = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            if os.getpgid(pid) != pgid:
                continue
            metadata = entry.stat()
            name = (entry / "comm").read_text().strip()
        except (OSError, ProcessLookupError):
            continue
        members.append({"pid": pid, "uid": metadata.st_uid,
                        "user": pwd.getpwuid(metadata.st_uid).pw_name, "comm": name})
    return sorted(members, key=lambda item: item["pid"])


def cleanup_ray_group(pgid: int, uid: int) -> dict:
    before = ray_group_members(pgid)
    if any(member["uid"] != uid for member in before):
        raise RuntimeError("Private Ray group contains another account's process")
    if before:
        os.killpg(pgid, signal.SIGTERM)
    deadline = time.monotonic() + 6
    while time.monotonic() < deadline and ray_group_members(pgid):
        time.sleep(0.2)
    remaining = ray_group_members(pgid)
    for member in remaining:
        try:
            if member["uid"] == uid and os.getpgid(member["pid"]) == pgid:
                os.kill(member["pid"], signal.SIGKILL)
        except ProcessLookupError:
            pass
    time.sleep(0.2)
    after = ray_group_members(pgid)
    if after:
        raise RuntimeError(f"Private Ray group still has {len(after)} members")
    return {"before_term": before, "after_kill": after}


def visible_gpu_uuid_without_context() -> dict:
    class CUuuid(ctypes.Structure):
        _fields_ = [("bytes", ctypes.c_ubyte * 16)]

    lib = ctypes.CDLL("libcuda.so.1")

    def bind(name, args):
        function = getattr(lib, name)
        function.argtypes = args
        function.restype = ctypes.c_int
        return function

    def checked(function, *args):
        code = function(*args)
        if code != 0:
            raise RuntimeError(f"{function.__name__} returned CUDA error {code}")

    cu_init = bind("cuInit", [ctypes.c_uint])
    cu_count = bind("cuDeviceGetCount", [ctypes.POINTER(ctypes.c_int)])
    cu_device = bind("cuDeviceGet", [ctypes.POINTER(ctypes.c_int), ctypes.c_int])
    cu_uuid = bind("cuDeviceGetUuid_v2", [ctypes.POINTER(CUuuid), ctypes.c_int])
    cu_context = bind("cuCtxGetCurrent", [ctypes.POINTER(ctypes.c_void_p)])
    checked(cu_init, 0)
    context_before = ctypes.c_void_p()
    checked(cu_context, ctypes.byref(context_before))
    if context_before.value is not None:
        raise RuntimeError("Worker had a CUDA context before UUID enumeration")
    count = ctypes.c_int()
    checked(cu_count, ctypes.byref(count))
    if count.value != 1:
        raise RuntimeError(f"Worker sees {count.value} CUDA devices, expected 1")
    device = ctypes.c_int()
    checked(cu_device, ctypes.byref(device), 0)
    raw_uuid = CUuuid()
    checked(cu_uuid, ctypes.byref(raw_uuid), device.value)
    context_after = ctypes.c_void_p()
    checked(cu_context, ctypes.byref(context_after))
    if context_after.value is not None:
        raise RuntimeError("Worker UUID enumeration created a CUDA context")
    return {"cuda_device_count": count.value,
            "cuda_zero_uuid": f"GPU-{uuid.UUID(bytes=bytes(raw_uuid.bytes))}",
            "context_before": None, "context_after": None}


def run_worker(ordinal: int, expected_uuid: str) -> dict:
    import ray
    from rlinf.scheduler import Cluster, Worker
    from rlinf.scheduler.hardware.accelerators.nvidia_gpu import NvidiaGPUManager

    physical_ordinal = str(ordinal)

    class UUIDProbe(Worker):
        def inspect_mapping(self):
            keys = ("RLINF_ALLOWED_PHYSICAL_GPUS", "CUDA_VISIBLE_DEVICES",
                    "VISIBLE_DEVICES", "LOCAL_ACCELERATOR_RANK",
                    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES")
            actual = {key: os.environ.get(key) for key in keys}
            expected = {
                "RLINF_ALLOWED_PHYSICAL_GPUS": physical_ordinal,
                "CUDA_VISIBLE_DEVICES": physical_ordinal,
                "VISIBLE_DEVICES": "0",
                "LOCAL_ACCELERATOR_RANK": "0",
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
            }
            if actual != expected:
                raise RuntimeError(f"Unexpected RLinf worker GPU mapping: {actual}")
            logical = NvidiaGPUManager.get_visible_devices()
            if logical != [0]:
                raise RuntimeError(f"Unexpected logical GPU ranks: {logical}")
            device = visible_gpu_uuid_without_context()
            if device["cuda_zero_uuid"] != expected_uuid:
                raise RuntimeError(f"Worker CUDA UUID differs from the selected GPU: {device}")
            return {"pid": os.getpid(), "env": actual,
                    "logical_devices": logical, **device}

    cluster = Cluster(num_nodes=1)
    if cluster.num_accelerators != 1 or ray.cluster_resources().get("GPU") != 1.0:
        raise RuntimeError("RLinf and Ray must each expose exactly one logical GPU")
    group = None
    try:
        group = UUIDProbe.create_group().launch(
            cluster=cluster,
            name="rlinf_uuid_probe_" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
        )
        worker = group.inspect_mapping().wait()[0]
        return {"cluster_logical_gpus": cluster.num_accelerators,
                "ray_gpu_resource": ray.cluster_resources().get("GPU"),
                "worker": worker}
    finally:
        if group is not None:
            group._close()
        ray.shutdown()


def ray_start_argv(ray_bin: Path, node_address: str, ray_temp: Path) -> list[str]:
    return [
        str(ray_bin), "start", "--head", f"--node-ip-address={node_address}",
        "--port=23979", "--node-manager-port=23980", "--object-manager-port=23981",
        "--ray-client-server-port=23982", "--dashboard-port=23983",
        "--dashboard-agent-listen-port=23984", "--dashboard-agent-grpc-port=23985",
        "--runtime-env-agent-port=23986", "--metrics-export-port=23987",
        "--min-worker-port=24000", "--max-worker-port=24200",
        "--num-cpus=4", "--num-gpus=1", "--object-store-memory=1073741824",
        "--include-dashboard=false", "--disable-usage-stats", f"--temp-dir={ray_temp}",
    ]


def set_driver_environment(environment: dict[str, str], node_address: str) -> None:
    """Drop inherited cluster/import paths before the driver constructs RLinf."""
    os.environ.pop("PYTHONPATH", None)
    os.environ.pop("RAY_ADDRESS", None)
    os.environ.update(environment)
    os.environ["RAY_ADDRESS"] = f"{node_address}:{HEAD_PORT}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-root", required=True, type=Path)
    parser.add_argument("--node-ip", required=True, type=node_ip)
    parser.add_argument("--gpu-ordinal", required=True, type=gpu_ordinal)
    parser.add_argument("--expected-uuid", required=True, type=gpu_uuid)
    parser.add_argument("--expected-user", choices=("navcloud", "navgpu"), default="navcloud")
    args = parser.parse_args()

    import pwd
    if os.geteuid() == 0 or pwd.getpwuid(os.geteuid()).pw_name != args.expected_user:
        raise RuntimeError(f"Run the isolated UUID probe only as {args.expected_user}")
    uid = os.geteuid()
    runs, private_parent = prepare_directories(args.node_root, uid)
    ray_bin = args.node_root.resolve(strict=True) / "envs/rlinf-v03-cu118/bin/ray"
    if not ray_bin.is_file():
        raise RuntimeError(f"RLinf Ray executable is missing: {ray_bin}")
    run_dir = Path(tempfile.mkdtemp(prefix="gpu-uuid-", dir=runs))
    result = {"started_utc": datetime.now(timezone.utc).isoformat(),
              "result": "error", "gpu_ordinal": args.gpu_ordinal,
              "expected_uuid": args.expected_uuid, "run_dir": str(run_dir)}
    pgid = None
    ray_temp = None
    ray_temp_identity = None

    def on_term(_signum, _frame):
        raise TimeoutError("Probe received SIGTERM; cleaning its private Ray group")

    signal.signal(signal.SIGTERM, on_term)
    error = None
    try:
        result["preflight"] = smi_target(args.gpu_ordinal, args.expected_uuid)
        check_ports_free()
        ray_temp, ray_temp_identity = reserve_ray_temp(private_parent, uid)
        result["ray_temp"] = str(ray_temp)
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment.pop("RAY_ADDRESS", None)
        environment["RLINF_ALLOWED_PHYSICAL_GPUS"] = str(args.gpu_ordinal)
        environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ordinal)
        environment["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
        environment["RAY_TMPDIR"] = str(ray_temp)
        with (run_dir / "ray_start.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                ray_start_argv(ray_bin, args.node_ip, ray_temp),
                env=environment, cwd=run_dir, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            pgid = process.pid
            result["ray_process_group"] = pgid
            if process.wait(timeout=50) != 0:
                raise RuntimeError("Private Ray head failed; see ray_start.log")
        members = ray_group_members(pgid)
        if not members or any(member["uid"] != uid for member in members):
            raise RuntimeError("Private Ray process group ownership could not be verified")
        result["ray_members_after_start"] = members
        set_driver_environment(environment, args.node_ip)
        result["mapping"] = run_worker(args.gpu_ordinal, args.expected_uuid)
        result["postflight_gpu"] = smi_target(args.gpu_ordinal, args.expected_uuid)
        result["result"] = "ok"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        result["error"] = error
    finally:
        group_clean = pgid is None
        if pgid is not None:
            try:
                result["ray_cleanup"] = cleanup_ray_group(pgid, uid)
                group_clean = True
            except Exception as exc:
                result["ray_cleanup_error"] = f"{type(exc).__name__}: {exc}"
        ports_clean = False
        try:
            check_ports_no_listeners()
            ports_clean = True
        except Exception as exc:
            result["postflight_ports_error"] = f"{type(exc).__name__}: {exc}"
        if ray_temp is not None and ray_temp_identity is not None and group_clean and ports_clean:
            try:
                remove_own_ray_temp(ray_temp, private_parent, ray_temp_identity, uid)
                result["ray_temp_removed"] = True
            except Exception as exc:
                result["ray_temp_cleanup_error"] = f"{type(exc).__name__}: {exc}"
        result["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (run_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                                              encoding="utf-8")
        print(json.dumps({key: result.get(key) for key in
                          ("result", "error", "run_dir", "ray_cleanup_error",
                           "postflight_ports_error", "ray_temp_cleanup_error")}, sort_keys=True))
    if error or result.get("ray_cleanup_error") or result.get("postflight_ports_error") or result.get("ray_temp_cleanup_error"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
