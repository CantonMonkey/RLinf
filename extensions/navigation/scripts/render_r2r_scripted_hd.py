"""Record a high-resolution, scripted Habitat R2R motion demonstration.

This runner sends fixed navigation actions to the existing Habitat R2R service.
It does not load or evaluate a policy. All outputs go under the private project's
runs directory; the shared R2R and MP3D assets are read only.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import select
import signal
import subprocess
import tempfile
from pathlib import Path


ROOT = Path("/srv/navigation")
ACTIONS = ("stop", "move_forward", "turn_left", "turn_right")
# Two collision-free forward steps from this episode's start prove translation.
# A 220-degree sweep then shows the room and finishes facing its doorway.
# This fixed camera-motion script is independent of policy outputs.
SCRIPT = (1,) * 2 + (3,) * 22


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def request(process: subprocess.Popen, payload: dict, timeout: int = 45) -> dict:
    if process.poll() is not None or process.stdin is None or process.stdout is None:
        raise RuntimeError("Habitat service exited before request")
    process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
    process.stdin.flush()
    ready, _, _ = select.select([process.stdout], [], [], timeout)
    if not ready:
        raise TimeoutError(f"Habitat service timed out on {payload['cmd']}")
    line = process.stdout.readline()
    if not line:
        raise RuntimeError(f"Habitat service closed protocol, exit={process.poll()}")
    response = json.loads(line)
    if not isinstance(response, dict) or "error" in response:
        raise RuntimeError(f"Habitat service error: {response.get('error', response)}")
    return response


def frame_from_response(response: dict, size: int) -> bytes:
    obs = response.get("obs")
    if not isinstance(obs, dict) or set(obs) != {
        "rgb", "instruction_ids", "instruction_mask"
    }:
        raise ValueError("R2R service changed observation protocol")
    rgb = obs["rgb"]
    if not isinstance(rgb, dict) or rgb.get("encoding") != "base64/raw-u8-hwc":
        raise ValueError("R2R service changed RGB encoding")
    if rgb.get("shape_hwc") != [size, size, 3]:
        raise ValueError("R2R service returned the wrong native RGB resolution")
    raw = base64.b64decode(rgb["data_b64"], validate=True)
    if len(raw) != size * size * 3:
        raise ValueError("R2R service returned an incomplete RGB frame")
    return raw


def encode(path: Path, frames: list[bytes], size: int, fps: int) -> dict:
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s:v", f"{size}x{size}",
        "-framerate", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
        "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(path),
    ], input=b"".join(frames), capture_output=True, check=True)
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=codec_name,width,height,nb_read_frames",
        "-of", "json", str(path),
    ], capture_output=True, text=True, check=True)
    streams = json.loads(probe.stdout).get("streams", [])
    if len(streams) != 1 or streams[0] != {
        "codec_name": "h264", "width": size, "height": size,
        "nb_read_frames": str(len(frames)),
    }:
        raise RuntimeError(f"MP4 stream/frame mismatch: {streams}")
    decoded = subprocess.run([
        "ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo",
        "-pix_fmt", "rgb24", "pipe:1",
    ], capture_output=True, check=True)
    if len(decoded.stdout) != len(frames) * size * size * 3:
        raise RuntimeError("MP4 full decode did not return the expected frame count")
    return {
        "path": str(path), "sha256": file_digest(path), "bytes": path.stat().st_size,
        "codec": "h264", "width": size, "height": size,
        "frames": len(frames), "fps": fps,
        "decoded_rgb_sha256": digest(decoded.stdout),
    }


def save_reset_png(path: Path, frame: bytes, size: int) -> dict:
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s:v", f"{size}x{size}",
        "-i", "pipe:0", "-frames:v", "1", str(path),
    ], input=frame, capture_output=True, check=True)
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height", "-of", "json", str(path),
    ], capture_output=True, text=True, check=True)
    if json.loads(probe.stdout).get("streams") != [{
        "codec_name": "png", "width": size, "height": size,
    }]:
        raise RuntimeError("reset PNG resolution/codec mismatch")
    return {"path": str(path), "sha256": file_digest(path), "bytes": path.stat().st_size,
            "width": size, "height": size, "source_rgb_sha256": digest(frame)}


def run(args: argparse.Namespace) -> Path:
    runs = (ROOT / "runs").resolve(strict=True)
    output = Path(tempfile.mkdtemp(prefix="vln-scripted-hd-", dir=runs)).resolve()
    if not output.is_relative_to(runs) or output == runs:
        raise RuntimeError("output escaped the private runs directory")
    dataset = (Path("/cache/navigation-example/datasets/r2r/val_unseen") /
               "val_unseen.json.gz").resolve(strict=True)
    scenes = Path("/cache/navigation-example/scene_datasets").resolve(strict=True)
    python = (ROOT / "envs/habitat-cpu-egl/bin/python").resolve(strict=True)
    service = (ROOT / "repos/rlinf-nav/scripts/habitat_r2r_service.py").resolve(strict=True)
    habitat_python = (ROOT / "repos/habitat-sim-cpu-egl/src_python").resolve(strict=True)
    habitat_lib = (ROOT / "envs/habitat-cpu-egl/lib").resolve(strict=True)
    vendor = Path("/usr/share/glvnd/egl_vendor.d/50_mesa.json").resolve(strict=True)
    child_env = os.environ.copy()
    child_env.update({
        "PYTHONPATH": str(habitat_python), "LD_LIBRARY_PATH": str(habitat_lib),
        "CUDA_VISIBLE_DEVICES": "", "LIBGL_ALWAYS_SOFTWARE": "1",
        "EGL_PLATFORM": "surfaceless", "HABITAT_SIM_SOFTWARE_EGL": "1",
        "__EGL_VENDOR_LIBRARY_FILENAMES": str(vendor), "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1",
    })
    command = [
        str(python), "-B", str(service), "--dataset-path", str(dataset),
        "--scenes-dir", str(scenes), "--split", "val_unseen",
        "--episode-index", "630", "--rgb-size", str(args.rgb_size),
        "--max-steps", str(len(SCRIPT)), "--max-instruction-tokens", "200",
    ]
    log = (output / "habitat_service.log").open("w", encoding="utf-8")
    process = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
        text=True, bufsize=1, env=child_env, start_new_session=True,
    )
    try:
        ready, _, _ = select.select([process.stdout], [], [], 120)
        if not ready:
            raise TimeoutError("Habitat service did not start within 120 seconds")
        line = process.stdout.readline()
        if not line:
            raise RuntimeError(f"Habitat service exited during startup: {process.poll()}")
        identity = json.loads(line)
        if (identity.get("ready") is not True or identity.get("split") != "val_unseen"
                or identity.get("episode_index") != 630
                or identity.get("rgb_size") != args.rgb_size
                or identity.get("actions") != list(ACTIONS)
                or identity.get("dataset_sha256") != file_digest(dataset)):
            raise ValueError("Habitat service identity/resolution mismatch")
        response = request(process, {"cmd": "reset"}, timeout=120)
        frames = [frame_from_response(response, args.rgb_size)]
        identity_info = response.get("info", {})
        for key in ("episode_id", "scene_id", "instruction_sha256"):
            if identity_info.get(key) != identity.get(key):
                raise ValueError(f"reset identity mismatch: {key}")
        trace = [{
            "event": "reset", "step": 0, "rgb_sha256": digest(frames[0]),
            **identity_info,
        }]
        previous_path = 0.0
        terminal = None
        for step, action in enumerate(SCRIPT, start=1):
            response = request(process, {"cmd": "step", "action": action})
            frame = frame_from_response(response, args.rgb_size)
            frames.append(frame)
            info = response.get("info", {})
            if any(info.get(key) != identity_info[key] for key in identity_info):
                raise ValueError("R2R episode identity changed during scripted actions")
            path_length = float(info["path_length"])
            distance = float(info["distance_to_goal"])
            if (not math.isfinite(path_length) or path_length < previous_path
                    or not math.isfinite(distance) or distance < 0):
                raise ValueError("invalid or regressing Habitat motion metric")
            trace.append({
                "event": "step", "step": step, "action_id": action,
                "action": ACTIONS[action], "rgb_sha256": digest(frame),
                "frame_changed": frame != frames[-2],
                "path_length_m": path_length, "step_displacement_m": path_length - previous_path,
                "distance_to_goal_m": distance, "collision": bool(info["collision"]),
                "terminated": response.get("terminated"),
                "truncated": response.get("truncated"),
                "terminal_frozen": info.get("terminal_frozen"),
            })
            previous_path = path_length
            if response.get("terminated") or response.get("truncated"):
                terminal = info
                break
        if terminal is None or len(trace) - 1 != len(SCRIPT):
            raise RuntimeError("scripted demo ended before its bounded action sequence")
        if previous_path <= 0 or not any(row.get("step_displacement_m", 0) > 0 for row in trace):
            raise RuntimeError("scripted demo did not show actual movement")
        if not any(row.get("frame_changed", False) for row in trace):
            raise RuntimeError("scripted demo RGB frames did not change")
        reset_png = save_reset_png(
            output / f"reset_rgb_{args.rgb_size}px.png", frames[0], args.rgb_size
        )
        trace_path = output / "action_trace.jsonl"
        trace_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in trace),
            encoding="utf-8",
        )
        video = encode(
            output / f"scripted_habitat_{args.rgb_size}px.mp4",
            frames, args.rgb_size, args.fps,
        )
        manifest = {
            "status": "ok", "kind": "scripted_simulator_demo",
            "interpretation": "Scripted Habitat motion/render demo; no policy was loaded or evaluated.",
            "renderer": "private Habitat 0.2.4 CPU Mesa llvmpipe",
            "service_script_sha256": file_digest(service),
            "runner_script_sha256": file_digest(Path(__file__)),
            "split": "val_unseen", "episode_index": 630,
            "episode_id": identity["episode_id"], "scene_id": identity["scene_id"],
            "instruction_sha256": identity["instruction_sha256"],
            "dataset_path": str(dataset), "dataset_sha256": identity["dataset_sha256"],
            "source_rgb_size": args.rgb_size, "actions": len(trace) - 1,
            "path_length_m": previous_path,
            "changed_frame_pairs": sum(bool(row.get("frame_changed")) for row in trace),
            "unique_source_frames": len({digest(frame) for frame in frames}),
            "success": float(terminal["success"]), "spl": float(terminal["spl"]),
            "action_trace": {"path": str(trace_path), "sha256": file_digest(trace_path)},
            "reset_png": reset_png,
            "video": video,
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({
            "output_dir": str(output), "video": video["path"],
            "native_rgb": args.rgb_size, "frames": len(frames),
            "path_length_m": previous_path,
            "changed_frame_pairs": manifest["changed_frame_pairs"],
        }, sort_keys=True), flush=True)
        return output
    finally:
        try:
            if process.poll() is None:
                request(process, {"cmd": "close"}, timeout=5)
                process.wait(timeout=5)
        except (RuntimeError, TimeoutError, subprocess.TimeoutExpired, BrokenPipeError):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
        finally:
            if process.stdin is not None:
                process.stdin.close()
            if process.stdout is not None:
                process.stdout.close()
            log.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-size", type=int, choices=(256, 512), default=512)
    parser.add_argument("--fps", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.fps <= 30:
        parser.error("fps must be 1..30")
    return args


if __name__ == "__main__":
    run(parse_args())
