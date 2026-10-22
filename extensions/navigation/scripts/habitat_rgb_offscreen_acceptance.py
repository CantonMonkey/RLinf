"""Render one fixed Habitat-Sim 0.2.4 RGB frame without a display server.

The scene and pose are from the official habitat-test-scenes PointNav validation
episode 0. This checks the visual rendering path only; it does not run a
PointNav episode, an RL rollout, or training.

Use --check-config on a node without NVIDIA EGL. That mode constructs the
SimulatorConfiguration and camera specification but never creates a renderer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import struct
import subprocess
import sys
import zlib
from pathlib import Path


SCENE_RELATIVE_PATH = Path(
    "scene_datasets/habitat-test-scenes/skokloster-castle.glb"
)
# habitat-test-scenes/v1/val episode 0, quaternion coefficients in [x, y, z, w].
AGENT_POSITION = (-1.2676633596420288, 0.2047852873802185, 12.595427513122559)
AGENT_ROTATION_XYZW = (0.0, 0.4536385088584658, 0.0, 0.8911857849408661)
SENSOR_UUID = "color_sensor"
SENSOR_RESOLUTION_HW = (256, 256)
SENSOR_POSITION = (0.0, 1.5, 0.0)
SENSOR_HFOV_DEGREES = 90.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nvidia_driver_version():
    """Return the driver version when nvidia-smi is available on this node."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    versions = sorted(set(result.stdout.splitlines())) if result.returncode == 0 else []
    return versions[0].strip() if len(versions) == 1 else None


def png_chunk(kind: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )


def encode_png(frame) -> bytes:
    """Encode an RGB or RGBA uint8 NumPy frame with the standard library."""
    height, width, channels = frame.shape
    color_type = {3: 2, 4: 6}[channels]
    scanlines = b"".join(b"\x00" + frame[row].tobytes() for row in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", ihdr)
        + png_chunk(b"IDAT", zlib.compress(scanlines, level=9))
        + png_chunk(b"IEND", b"")
    )


def make_config(habitat_sim, scene: Path, gpu_device_id: int):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(scene)
    sim_cfg.create_renderer = True
    sim_cfg.gpu_device_id = gpu_device_id

    sensor = habitat_sim.CameraSensorSpec()
    sensor.uuid = SENSOR_UUID
    sensor.sensor_type = habitat_sim.SensorType.COLOR
    sensor.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor.resolution = list(SENSOR_RESOLUTION_HW)
    sensor.position = list(SENSOR_POSITION)
    sensor.hfov = SENSOR_HFOV_DEGREES
    sensor.gpu2gpu_transfer = False  # Request a CPU NumPy frame for PNG/hash.

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor]
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset-root",
        type=Path,
        required=True,
        help="Directory containing scene_datasets/habitat-test-scenes",
    )
    parser.add_argument(
        "--gpu-device-id",
        type=int,
        default=0,
        help="Logical GPU ID visible to this process (default: 0)",
    )
    parser.add_argument(
        "--output-png", type=Path, help="Required unless --check-config is used"
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Validate API, assets, and sensor configuration without creating a renderer",
    )
    args = parser.parse_args()
    if args.gpu_device_id < 0:
        parser.error("--gpu-device-id must be nonnegative")
    if not args.check_config and args.output_png is None:
        parser.error("--output-png is required for rendering")

    scene = (args.asset_root / SCENE_RELATIVE_PATH).resolve(strict=True)
    if not scene.is_file():
        parser.error("official habitat-test-scenes GLB is missing")

    import habitat_sim
    import numpy as np
    from habitat_sim.utils.common import quat_from_coeffs

    if habitat_sim.__version__ != "0.2.4":
        raise RuntimeError(
            "This acceptance script targets Habitat-Sim 0.2.4; found "
            + str(habitat_sim.__version__)
        )
    config = make_config(habitat_sim, scene, args.gpu_device_id)
    result = {
        "schema_version": 1,
        "mode": "config_only" if args.check_config else "rgb_single_frame",
        "scene": str(scene),
        "scene_sha256": sha256_file(scene),
        "pose_source": "habitat-test-scenes/v1/val episode 0",
        "agent_position_xyz": list(AGENT_POSITION),
        "agent_rotation_xyzw": list(AGENT_ROTATION_XYZW),
        "sensor_uuid": SENSOR_UUID,
        "sensor_resolution_hw": list(SENSOR_RESOLUTION_HW),
        "sensor_position_xyz": list(SENSOR_POSITION),
        "sensor_hfov_degrees": SENSOR_HFOV_DEGREES,
        "gpu_device_id": args.gpu_device_id,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "versions": {
            "python": platform.python_version(),
            "habitat_sim": habitat_sim.__version__,
            "numpy": np.__version__,
            "nvidia_driver": nvidia_driver_version(),
        },
        "visual_render_validated": False,
        "pointnav_validated": False,
        "rl_training_validated": False,
    }

    if args.check_config:
        result["renderer_created"] = False
        result["camera_spec_count"] = len(config.agents[0].sensor_specifications)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0

    sim = habitat_sim.Simulator(config)
    try:
        state = habitat_sim.AgentState()
        state.position = np.asarray(AGENT_POSITION, dtype=np.float32)
        state.rotation = quat_from_coeffs(AGENT_ROTATION_XYZW)
        sim.initialize_agent(0, state)
        observations = sim.get_sensor_observations()
        if SENSOR_UUID not in observations:
            raise RuntimeError("Habitat did not return the configured RGB sensor")
        frame = np.ascontiguousarray(observations[SENSOR_UUID])
        if (
            frame.ndim != 3
            or frame.shape[:2] != SENSOR_RESOLUTION_HW
            or frame.shape[2] not in (3, 4)
            or frame.dtype != np.uint8
        ):
            raise RuntimeError(
                "Unexpected RGB frame: shape="
                + str(frame.shape)
                + ", dtype="
                + str(frame.dtype)
            )
        rgb = frame[:, :, :3]
        if int(rgb.max()) == int(rgb.min()):
            raise RuntimeError("Rendered RGB frame is uniform; check camera/scene")
        raw_sha256 = hashlib.sha256(frame.tobytes()).hexdigest()
        png_bytes = encode_png(frame)
    finally:
        sim.close()

    output_png = args.output_png.resolve()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_png.write_bytes(png_bytes)
    result.update(
        {
            "renderer_created": True,
            "visual_render_validated": True,
            "frame_shape_hwc": list(frame.shape),
            "frame_dtype": str(frame.dtype),
            "frame_raw_sha256": raw_sha256,
            "rgb_min": int(rgb.min()),
            "rgb_max": int(rgb.max()),
            "png": str(output_png),
            "png_bytes": len(png_bytes),
            "png_sha256": hashlib.sha256(png_bytes).hexdigest(),
        }
    )
    manifest = output_png.with_suffix(".json")
    manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
