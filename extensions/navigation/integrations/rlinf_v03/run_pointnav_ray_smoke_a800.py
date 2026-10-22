"""Bounded, isolated Ray/FSDP PointNav smoke on the A800 Pod.

Run only as navcloud from the private A800 RLinf Python. This wrapper owns a
fresh Ray process group and never stops other Ray jobs.
It records an actual CUDA process on the selected physical GPU when observed.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from worker_uuid_preflight import (
    check_ports_free,
    check_ports_no_listeners,
    cleanup_ray_group,
    prepare_directories,
    ray_group_members,
    ray_start_argv,
    remove_own_ray_temp,
    reserve_ray_temp,
    smi_target,
)


ROOT = Path("/srv/navigation")
PYTHON = ROOT / "envs/rlinf-v03-cu118/bin/python"
HABITAT_PYTHON = Path("/opt/conda/envs/streamvln/bin/python")
CHECKOUT = ROOT / "repos/RLinf-v0.3"
PREFLIGHT = ROOT / "repos/rlinf-nav/integrations/rlinf_v03/node_preflight.py"
NODE_IP = "192.0.2.10"
GPU_ORDINAL = 7
GPU_UUID = "GPU-22222222-2222-2222-2222-222222222222"
MAX_TRAIN_SECONDS = 300


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def own_compute_processes(pgid: int) -> list[dict]:
    members = {member["pid"] for member in ray_group_members(pgid)}
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
         "--format=csv,noheader,nounits"], text=True, timeout=10,
    )
    rows = []
    for line in raw.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 3:
            continue
        try:
            pid = int(fields[0])
        except ValueError:
            continue
        if pid in members:
            if fields[1] != GPU_UUID:
                raise RuntimeError(f"Ray CUDA process {pid} appeared on unexpected GPU {fields[1]}")
            rows.append({"pid": pid, "gpu_uuid": fields[1],
                         "memory_mib": fields[2]})
    return rows


def stop_driver(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def preserve_ray_logs(ray_temp: Path, run_dir: Path) -> None:
    source = ray_temp / "session_latest/logs"
    if not source.is_dir():
        return
    target = run_dir / "ray_logs"
    target.mkdir(mode=0o700)
    names = ["raylet.err", "raylet.out", "gcs_server.err", "gcs_server.out"]
    names += [p.name for p in sorted(source.glob("worker*.err"))[:12]]
    names += [p.name for p in sorted(source.glob("worker*.out"))[:12]]
    for name in names:
        path = source / name
        if path.is_file() and not path.is_symlink():
            (target / name).write_bytes(path.read_bytes()[-500_000:])


def verify_own_ray_group(pgid: int, ray_temp: Path, uid: int) -> list[dict]:
    members = ray_group_members(pgid)
    if any(item["uid"] != uid for item in members):
        raise RuntimeError("Ray process group includes a different account")
    matched = False
    for item in members:
        if item["comm"] not in ("raylet", "gcs_server"):
            continue
        try:
            command = (Path("/proc") / str(item["pid"]) / "cmdline").read_bytes()
        except OSError:
            continue
        if os.fsencode(ray_temp) in command:
            matched = True
            break
    if members and not matched:
        raise RuntimeError("Ray group no longer identifies this private temp directory")
    return members


def main() -> None:
    import pwd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "eval"), default="train")
    parser.add_argument("--eval-resume-dir", type=Path)
    args = parser.parse_args()
    if (args.mode == "eval") != (args.eval_resume_dir is not None):
        parser.error("--eval-resume-dir is required only for --mode eval")

    uid = os.geteuid()
    if uid == 0 or pwd.getpwuid(uid).pw_name != "navcloud":
        raise RuntimeError("Run only as navcloud")
    runs, parent = prepare_directories(ROOT, uid)
    check_ports_free()
    selected = smi_target(GPU_ORDINAL, GPU_UUID)
    free_mib = int(subprocess.check_output(
        ["nvidia-smi", f"--id={GPU_ORDINAL}", "--query-gpu=memory.free",
         "--format=csv,noheader,nounits"], text=True, timeout=10,
    ).strip())
    if free_mib < 40_000:
        raise RuntimeError(f"Insufficient free A800 memory for bounded smoke: {free_mib} MiB")

    run_dir = Path(tempfile.mkdtemp(prefix=f"pointnav-a800-gpu-{args.mode}-", dir=runs))
    result = {"started_utc": now(), "run_dir": str(run_dir), "result": "error",
              "preflight_gpu": selected, "estimated_free_mib": free_mib,
              "mode": args.mode, "eval_resume_dir": str(args.eval_resume_dir) if args.eval_resume_dir else None,
              "max_job_seconds": MAX_TRAIN_SECONDS}
    ray_temp = None
    ray_temp_identity = None
    ray_pgid = None
    driver = None

    def on_term(_signal, _frame):
        raise RuntimeError("Wrapper received termination signal")

    signal.signal(signal.SIGTERM, on_term)
    try:
        preflight_env = os.environ.copy()
        preflight_env.pop("PYTHONPATH", None)
        preflight_env.pop("RAY_ADDRESS", None)
        preflight_env["CUDA_VISIBLE_DEVICES"] = ""
        command = [str(PYTHON), str(PREFLIGHT), "--node-root", str(ROOT),
                   "--habitat-python", str(HABITAT_PYTHON),
                   "--rlinf-python", str(PYTHON), "--run-root", str(run_dir / "rlinf"),
                   "--compose-check"]
        if args.eval_resume_dir is not None:
            command += ["--eval-resume-dir", str(args.eval_resume_dir)]
        preflight = subprocess.run(command, cwd=CHECKOUT, env=preflight_env,
                                   capture_output=True, text=True, timeout=60)
        (run_dir / "node_preflight.json").write_text(preflight.stdout, encoding="utf-8")
        (run_dir / "node_preflight.stderr.log").write_text(preflight.stderr, encoding="utf-8")
        if preflight.returncode != 0:
            raise RuntimeError("PointNav preflight exited nonzero; inspect node_preflight.stderr.log")
        plan = json.loads(preflight.stdout)
        if plan["status"] != "static_preflight_ok" or not plan["hydra_compose_validated"]:
            raise RuntimeError("PointNav installed node preflight failed")
        trainer_argv = plan["eval_argv" if args.mode == "eval" else "smoke_argv"]
        if not trainer_argv or trainer_argv[0] != str(PYTHON):
            raise RuntimeError("Node preflight did not produce the private RLinf trainer")
        if args.mode == "eval" and not plan["eval_checkpoint_content_verified"]:
            raise RuntimeError("Evaluation checkpoint content was not verified")
        result["split_configs_validated"] = plan["split_configs_validated"]
        print(f"preflight_ok run_dir={run_dir}", flush=True)

        check_ports_free()
        ray_temp, ray_temp_identity = reserve_ray_temp(parent, uid)
        result["ray_temp"] = str(ray_temp)
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment.pop("RAY_ADDRESS", None)
        environment.update({"RLINF_ALLOWED_PHYSICAL_GPUS": str(GPU_ORDINAL),
                            "CUDA_VISIBLE_DEVICES": str(GPU_ORDINAL),
                            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                            "RAY_TMPDIR": str(ray_temp)})
        with (run_dir / "ray_start.log").open("w", encoding="utf-8") as log:
            head = subprocess.Popen(
                ray_start_argv(ROOT / "envs/rlinf-v03-cu118/bin/ray", NODE_IP, ray_temp),
                cwd=run_dir, env=environment, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            ray_pgid = head.pid
            result["ray_process_group"] = ray_pgid
            if head.wait(timeout=50) != 0:
                raise RuntimeError("Private Ray head failed; inspect ray_start.log")
        members = verify_own_ray_group(ray_pgid, ray_temp, uid)
        if not members:
            raise RuntimeError("Private Ray process group failed ownership check")
        result["ray_members_after_start"] = members
        print(f"ray_ready pgid={ray_pgid}", flush=True)

        environment["RAY_ADDRESS"] = f"{NODE_IP}:23979"
        with (run_dir / "trainer.log").open("w", encoding="utf-8") as log:
            driver = subprocess.Popen(trainer_argv, cwd=CHECKOUT, env=environment,
                                      stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True)
            result["driver_pid"] = driver.pid
            print(f"trainer_started pid={driver.pid}", flush=True)
            deadline = time.monotonic() + MAX_TRAIN_SECONDS
            observations = {}
            while driver.poll() is None and time.monotonic() < deadline:
                for row in own_compute_processes(ray_pgid):
                    observations[row["pid"]] = row
                time.sleep(2)
            result["ray_cuda_processes"] = sorted(observations.values(), key=lambda x: x["pid"])
            if driver.poll() is None:
                result["timed_out"] = True
                raise TimeoutError(f"Trainer exceeded {MAX_TRAIN_SECONDS} seconds")
            result["trainer_exit_code"] = driver.returncode
            if driver.returncode != 0:
                raise RuntimeError(f"Trainer exited {driver.returncode}; inspect trainer.log")
        result["postflight_gpu"] = smi_target(GPU_ORDINAL, GPU_UUID)
        result["result"] = "trainer_exit_zero"
        print(f"trainer_exit_zero cuda_ray_pids={len(result['ray_cuda_processes'])}", flush=True)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if driver is not None:
            try:
                stop_driver(driver)
            except Exception as exc:
                result["driver_cleanup_error"] = f"{type(exc).__name__}: {exc}"
        if ray_temp is not None:
            try:
                preserve_ray_logs(ray_temp, run_dir)
            except Exception as exc:
                result["ray_log_copy_error"] = f"{type(exc).__name__}: {exc}"
        group_clean = ray_pgid is None
        if ray_pgid is not None:
            try:
                result["ray_members_before_cleanup"] = verify_own_ray_group(
                    ray_pgid, ray_temp, uid,
                )
                result["ray_cleanup"] = cleanup_ray_group(ray_pgid, uid)
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
                remove_own_ray_temp(ray_temp, parent, ray_temp_identity, uid)
                result["ray_temp_removed"] = True
            except Exception as exc:
                result["ray_temp_cleanup_error"] = f"{type(exc).__name__}: {exc}"
        result["finished_utc"] = now()
        (run_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                                              encoding="utf-8")
        print(json.dumps({key: result.get(key) for key in
                          ("result", "error", "run_dir", "trainer_exit_code",
                           "ray_cleanup_error", "postflight_ports_error",
                           "ray_temp_cleanup_error")}, sort_keys=True), flush=True)
    if result["result"] != "trainer_exit_zero" or any(
        key.endswith("_error") for key in result if key not in ("error",)
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
