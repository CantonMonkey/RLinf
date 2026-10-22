"""Bounded, isolated real-R2R VLN Ray/FSDP smoke on the A800 Pod.

Run only as navcloud from the private A800 RLinf Python. This wrapper owns a
fresh Ray process group and never stops other Ray jobs.
It records an actual CUDA process on the selected physical GPU when observed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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
CHECKOUT = ROOT / "repos/RLinf-v0.3"
STAGING = ROOT / "repos/rlinf-nav/integrations/rlinf_v03"
HABITAT_PYTHON = ROOT / "envs/habitat-cpu-egl/bin/python"
TRAINER = CHECKOUT / "examples/embodiment/train_embodied_agent.py"
NODE_IP = "192.0.2.10"
GPU_ORDINAL = 7
GPU_UUID = "GPU-22222222-2222-2222-2222-222222222222"
MAX_TRAIN_SECONDS = 300
MAX_EVAL_SECONDS = 900
VLN_FILES = {
    "config_vln.py": "rlinf/config.py",
    "gpu_uuid_gate.py": "rlinf/utils/gpu_uuid_gate.py",
    "fsdp_actor_worker_gpu_uuid.py": "rlinf/workers/actor/fsdp_actor_worker.py",
    "models_init_vln.py": "rlinf/models/__init__.py",
    "huggingface_worker_vln.py": "rlinf/workers/rollout/hf/huggingface_worker.py",
    "envs_init_vln.py": "rlinf/envs/__init__.py",
    "vln_init.py": "rlinf/envs/vln/__init__.py",
    "vln_env.py": "rlinf/envs/vln/vln_env.py",
    "vln_policy.py": "rlinf/models/embodiment/vln_policy.py",
    "embodied_io_struct_vln.py": "rlinf/data/embodied_io_struct.py",
    "train_embodied_agent.py": "examples/embodiment/train_embodied_agent.py",
    "vln_r2r_env.yaml": "examples/embodiment/config/env/vln_r2r_env.yaml",
    "vln_policy.yaml": "examples/embodiment/config/model/vln_policy.yaml",
    "vln_r2r_ppo_smoke.yaml": "examples/embodiment/config/vln_r2r_ppo_smoke.yaml",
    "vln_r2r_ppo_eval.yaml": "examples/embodiment/config/vln_r2r_ppo_eval.yaml",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checkpoint(resume_dir: Path) -> dict:
    """Bind actor and rollout to the same finite, strictly loadable VLN state."""
    import torch
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
    from rlinf.models.embodiment.vln_policy import VLNPolicy

    if not re.fullmatch(r"global_step_[0-9]+", resume_dir.name):
        raise ValueError("Expected a global_step_N checkpoint directory")
    if not resume_dir.is_relative_to(ROOT / "runs"):
        raise ValueError("Checkpoint must be inside private run artifacts")
    actor = resume_dir / "actor"
    weights = actor / "model_state_dict/full_weights.pt"
    dcp = actor / "dcp_checkpoint"
    if (not weights.is_file() or weights.stat().st_size == 0
            or not (dcp / ".metadata").is_file()
            or not any(p.is_file() and p.stat().st_size for p in dcp.glob("*.distcp"))):
        raise ValueError("Incomplete VLN actor checkpoint")
    full = torch.load(weights, map_location="cpu", weights_only=True)
    model = VLNPolicy(vocab_size=2504, max_instruction_tokens=200, image_size=64,
                      action_dim=1, num_action_chunks=1, discrete_n_actions=4,
                      add_value_head=True)
    model.load_state_dict(full, strict=True)
    if not full or not all(isinstance(x, torch.Tensor) and bool(torch.isfinite(x).all())
                           for x in full.values()):
        raise ValueError("VLN full weights are empty or nonfinite")
    with tempfile.TemporaryDirectory(prefix="vln-dcp-") as temporary:
        converted = Path(temporary) / "actor.pt"
        dcp_to_torch_save(dcp, converted)
        training = torch.load(converted, map_location="cpu", weights_only=False)
    restored = training["fsdp_checkpoint"]["model"]
    if full.keys() != restored.keys() or any(
        value.shape != restored[name].shape or value.dtype != restored[name].dtype
        or not torch.equal(value, restored[name]) for name, value in full.items()
    ):
        raise ValueError("VLN DCP actor state differs from rollout full weights")
    return {"resume_dir": str(resume_dir), "full_weights": str(weights),
            "full_weights_sha256": sha256(weights), "model_tensors": len(full),
            "dcp_matches_full_weights": True}


def inspect_vln_install(run_dir: Path, mode: str, resume_dir: Path | None) -> tuple[list[str], dict]:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    verified = {}
    for staged_name, destination in VLN_FILES.items():
        staged = STAGING / staged_name
        installed = CHECKOUT / destination
        staged_hash = sha256(staged)
        if sha256(installed) != staged_hash:
            raise RuntimeError(f"VLN staged/installed mismatch: {staged_name}")
        verified[staged_name] = staged_hash
    if sha256(STAGING / "cluster.py") != sha256(CHECKOUT / "rlinf/scheduler/cluster/cluster.py"):
        raise RuntimeError("RLinf cluster staged/installed mismatch")
    config_root = CHECKOUT / "examples/embodiment/config"
    overrides = [f"runner.logger.log_path={run_dir / 'rlinf'}",
                 f"env.train.video_cfg.video_base_dir={run_dir / 'videos'}",
                 f"env.eval.video_cfg.video_base_dir={run_dir / 'videos'}"]
    checkpoint = None
    if mode == "eval":
        if resume_dir is None:
            raise ValueError("Eval mode requires --resume-dir")
        checkpoint = verify_checkpoint(resume_dir)
        overrides += [f"runner.resume_dir={checkpoint['resume_dir']}",
                      f"runner.ckpt_path={checkpoint['full_weights']}"]
    elif resume_dir is not None:
        raise ValueError("--resume-dir is only valid in eval mode")
    config_name = "vln_r2r_ppo_eval" if mode == "eval" else "vln_r2r_ppo_smoke"
    with initialize_config_dir(config_dir=str(config_root), version_base="1.1"):
        cfg = compose(config_name=config_name, overrides=overrides)
        OmegaConf.resolve(cfg)
    if (bool(cfg.runner.only_eval) != (mode == "eval")
            or cfg.algorithm.normalize_advantages
            or cfg.actor.model.model_type != "vln_policy"
            or cfg.env.train.dataset_split != "train"
            or cfg.env.train.episode_index != 4852
            or cfg.env.eval.dataset_split != "val_unseen"
            or cfg.env.eval.episode_index != 630
            or cfg.env.train.python_bin != str(HABITAT_PYTHON)
            or cfg.env.eval.python_bin != str(HABITAT_PYTHON)
            or cfg.env.train.asset_root != "/cache/navigation-example"
            or cfg.env.eval.asset_root != "/cache/navigation-example"
            or cfg.actor.global_batch_size != 8
            or (mode == "eval" and
                (not cfg.env.eval.video_cfg.save_video
                 or cfg.env.eval.max_episode_steps != 500
                 or cfg.env.eval.max_steps_per_rollout_epoch != 500))):
        raise RuntimeError("Installed VLN smoke config differs from bounded train/val contract")
    plan = {"status": "vln_static_preflight_ok", "installed_sha256": verified,
            "train_split": "train", "train_episode_index": 4852,
            "eval_split": "val_unseen", "eval_episode_index": 630,
            "cuda_validated": False, "training_started": False,
            "normalize_advantages": False,
            "runner_log_path": str(run_dir / "rlinf"), "mode": mode}
    if checkpoint is not None:
        plan["checkpoint"] = checkpoint
    argv = [str(PYTHON), str(TRAINER), "--config-name", config_name, *overrides]
    return argv, plan


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
    parser.add_argument("--resume-dir", type=Path)
    args = parser.parse_args()

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

    run_dir = Path(tempfile.mkdtemp(prefix=f"vln-a800-gpu-{args.mode}-", dir=runs))
    max_seconds = MAX_EVAL_SECONDS if args.mode == "eval" else MAX_TRAIN_SECONDS
    result = {"started_utc": now(), "run_dir": str(run_dir), "result": "error",
              "preflight_gpu": selected, "estimated_free_mib": free_mib,
              "mode": args.mode, "task": "r2r_vln",
              "max_job_seconds": max_seconds}
    ray_temp = None
    ray_temp_identity = None
    ray_pgid = None
    driver = None

    def on_term(_signal, _frame):
        raise RuntimeError("Wrapper received termination signal")

    signal.signal(signal.SIGTERM, on_term)
    try:
        resume = args.resume_dir.expanduser().resolve(strict=True) if args.resume_dir else None
        trainer_argv, plan = inspect_vln_install(run_dir, args.mode, resume)
        (run_dir / "node_preflight.json").write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        result["config_contract"] = {key: plan[key] for key in
                                     ("train_split", "train_episode_index", "eval_split",
                                      "eval_episode_index", "normalize_advantages")}
        if "checkpoint" in plan:
            result["checkpoint"] = plan["checkpoint"]
        print(f"preflight_ok run_dir={run_dir}", flush=True)

        check_ports_free()
        ray_temp, ray_temp_identity = reserve_ray_temp(parent, uid)
        result["ray_temp"] = str(ray_temp)
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment.pop("RAY_ADDRESS", None)
        environment.update({"RLINF_ALLOWED_PHYSICAL_GPUS": str(GPU_ORDINAL),
                            "CUDA_VISIBLE_DEVICES": str(GPU_ORDINAL),
                            "RLINF_VERIFY_WORKER_GPU_UUID": "1",
                            "RLINF_EXPECTED_GPU_UUID": GPU_UUID,
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
            deadline = time.monotonic() + max_seconds
            observations = {}
            while driver.poll() is None and time.monotonic() < deadline:
                for row in own_compute_processes(ray_pgid):
                    observations[row["pid"]] = row
                time.sleep(2)
            result["ray_cuda_processes"] = sorted(observations.values(), key=lambda x: x["pid"])
            if driver.poll() is None:
                result["timed_out"] = True
                raise TimeoutError(f"Trainer exceeded {max_seconds} seconds")
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
