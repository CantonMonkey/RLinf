"""Private raw-frame contract for pinned StreamVLN inference.

The learned policy must never receive this object.  It binds the exact
Habitat state used by the released evaluator to one environment, episode and
physical step, while keeping the payload model-free for the JSONL bridge.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Mapping

if __package__:
    from .streamvln_p8_official_keep import PINNED_STREAMVLN_COMMIT
else:  # Explicit direct-file mode for local contract tests only.
    from streamvln_p8_official_keep import PINNED_STREAMVLN_COMMIT


OFFICIAL_FRAME_SCHEMA_VERSION = 2
OFFICIAL_FRAME_SOURCE = f"streamvln/{PINNED_STREAMVLN_COMMIT}/raw-frame"
OFFICIAL_RGB_WIDTH = 640
OFFICIAL_RGB_HEIGHT = 480
OFFICIAL_RGB_HFOV_DEG = 79
OFFICIAL_CAMERA_POSITION_XYZ = [0.0, 1.25, 0.0]
OFFICIAL_CAMERA_ORIENTATION_XYZ = [0.0, 0.0, 0.0]
OFFICIAL_DEPTH_MIN_M = 0.0
OFFICIAL_DEPTH_MAX_M = 10.0
OFFICIAL_PROCESSOR_WIDTH = 384
OFFICIAL_PROCESSOR_HEIGHT = 384
OFFICIAL_FORWARD_STEP_M = 0.25
OFFICIAL_TURN_ANGLE_DEG = 15

RGB_ENCODING = "base64/raw-u8-hwc"
DEPTH_ENCODING = "base64/raw-f32le-hw-normalized"
MATRIX_ENCODING = "base64/raw-f64le-row-major"

OFFICIAL_FRAME_SENSOR_CONTRACT = {
    "rgb": {
        "shape_hwc": [OFFICIAL_RGB_HEIGHT, OFFICIAL_RGB_WIDTH, 3],
        "dtype": "uint8",
        "hfov_deg": OFFICIAL_RGB_HFOV_DEG,
        "position_xyz": OFFICIAL_CAMERA_POSITION_XYZ,
        "orientation_xyz": OFFICIAL_CAMERA_ORIENTATION_XYZ,
    },
    "depth": {
        "shape_hw": [OFFICIAL_RGB_HEIGHT, OFFICIAL_RGB_WIDTH],
        "dtype": "float32_le",
        "normalized": True,
        "min_depth_m": OFFICIAL_DEPTH_MIN_M,
        "max_depth_m": OFFICIAL_DEPTH_MAX_M,
        "hfov_deg": OFFICIAL_RGB_HFOV_DEG,
        "position_xyz": OFFICIAL_CAMERA_POSITION_XYZ,
        "orientation_xyz": OFFICIAL_CAMERA_ORIENTATION_XYZ,
    },
    "pose": {
        "shape": [4, 4],
        "dtype": "float64_le",
        "value": "camera_to_episodic_times_upstream_axis_align",
    },
    "intrinsics": {
        "shape": [4, 4],
        "dtype": "float64_le",
        "value": "raw_640x480_rgb_sensor_intrinsics",
    },
    "processor_target_hw": [OFFICIAL_PROCESSOR_HEIGHT, OFFICIAL_PROCESSOR_WIDTH],
    "forward_step_m": OFFICIAL_FORWARD_STEP_M,
    "turn_angle_deg": OFFICIAL_TURN_ANGLE_DEG,
}


def official_frame_contract_sha256() -> str:
    payload = json.dumps(
        {
            "schema_version": OFFICIAL_FRAME_SCHEMA_VERSION,
            "source": OFFICIAL_FRAME_SOURCE,
            "sensor_contract": OFFICIAL_FRAME_SENSOR_CONTRACT,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _decode_blob(raw: object, *, expected_encoding: str, expected_size: int) -> tuple[bytes, str]:
    if not isinstance(raw, Mapping) or set(raw) != {"encoding", "data_b64", "sha256"}:
        raise ValueError("official frame blob schema mismatch")
    if raw["encoding"] != expected_encoding or not isinstance(raw["data_b64"], str):
        raise ValueError("official frame blob encoding mismatch")
    digest = raw["sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("official frame blob SHA-256 is invalid")
    try:
        payload = base64.b64decode(raw["data_b64"], validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("official frame blob is invalid base64") from exc
    if len(payload) != expected_size or _sha256(payload) != digest:
        raise ValueError("official frame blob length or SHA-256 mismatch")
    return payload, digest


def _validate_matrix(payload: bytes, *, name: str) -> None:
    values = tuple(value[0] for value in struct.iter_unpack("<d", payload))
    if len(values) != 16 or any(not math.isfinite(value) for value in values):
        raise ValueError(f"official {name} matrix is invalid")
    if not all(math.isclose(values[index], expected) for index, expected in zip((12, 13, 14, 15), (0, 0, 0, 1))):
        raise ValueError(f"official {name} matrix has an invalid homogeneous row")


@dataclass(frozen=True)
class OfficialFrame:
    """One private StreamVLN frame tied to an exact simulator state."""

    env_id: int
    episode_serial: int
    episode_generation: str
    physical_step: int
    episode_id: str
    scene_id: str
    instruction_sha256: str
    rgb_u8: bytes
    depth_normalized_f32le: bytes
    pose_f64le: bytes
    intrinsics_f64le: bytes
    rgb_sha256: str
    depth_sha256: str
    pose_sha256: str
    intrinsics_sha256: str

    def __post_init__(self) -> None:
        if type(self.env_id) is not int or self.env_id < 0:
            raise ValueError("official frame env_id must be nonnegative")
        if type(self.episode_serial) is not int or self.episode_serial < 0:
            raise ValueError("official frame episode_serial must be nonnegative")
        if (
            not isinstance(self.episode_generation, str)
            or len(self.episode_generation) != 32
            or any(
                character not in "0123456789abcdef"
                for character in self.episode_generation
            )
        ):
            raise ValueError("official frame episode_generation is invalid")
        if type(self.physical_step) is not int or self.physical_step < 0:
            raise ValueError("official frame physical_step must be nonnegative")
        if not isinstance(self.episode_id, str) or not self.episode_id:
            raise ValueError("official frame episode_id must be non-empty")
        if not isinstance(self.scene_id, str) or not self.scene_id:
            raise ValueError("official frame scene_id must be non-empty")
        if (
            not isinstance(self.instruction_sha256, str)
            or len(self.instruction_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.instruction_sha256)
        ):
            raise ValueError("official frame instruction SHA-256 is invalid")
        expected_sizes = {
            "rgb": OFFICIAL_RGB_HEIGHT * OFFICIAL_RGB_WIDTH * 3,
            "depth": OFFICIAL_RGB_HEIGHT * OFFICIAL_RGB_WIDTH * 4,
            "pose": 16 * 8,
            "intrinsics": 16 * 8,
        }
        blobs = {
            "rgb": (self.rgb_u8, self.rgb_sha256),
            "depth": (self.depth_normalized_f32le, self.depth_sha256),
            "pose": (self.pose_f64le, self.pose_sha256),
            "intrinsics": (self.intrinsics_f64le, self.intrinsics_sha256),
        }
        for name, (payload, digest) in blobs.items():
            if not isinstance(payload, bytes) or len(payload) != expected_sizes[name]:
                raise ValueError(f"official {name} payload length mismatch")
            if digest != _sha256(payload):
                raise ValueError(f"official {name} payload SHA-256 mismatch")
        _validate_matrix(self.pose_f64le, name="pose")
        _validate_matrix(self.intrinsics_f64le, name="intrinsics")

    @property
    def frame_ref(self) -> str:
        identity = {
            "contract_sha256": official_frame_contract_sha256(),
            "env_id": self.env_id,
            "episode_serial": self.episode_serial,
            "episode_generation": self.episode_generation,
            "physical_step": self.physical_step,
            "episode_id": self.episode_id,
            "scene_id": self.scene_id,
            "instruction_sha256": self.instruction_sha256,
            "rgb_sha256": self.rgb_sha256,
            "depth_sha256": self.depth_sha256,
            "pose_sha256": self.pose_sha256,
            "intrinsics_sha256": self.intrinsics_sha256,
        }
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()

    def trace_metadata(self) -> dict:
        return {
            "env_id": self.env_id,
            "episode_serial": self.episode_serial,
            "episode_generation": self.episode_generation,
            "physical_step": self.physical_step,
            "episode_id": self.episode_id,
            "scene_id": self.scene_id,
            "instruction_sha256": self.instruction_sha256,
            "frame_ref": self.frame_ref,
            "rgb_sha256": self.rgb_sha256,
            "depth_sha256": self.depth_sha256,
            "pose_sha256": self.pose_sha256,
            "intrinsics_sha256": self.intrinsics_sha256,
            "contract_sha256": official_frame_contract_sha256(),
        }

    def validate_identity(
        self,
        *,
        env_id: int,
        episode_serial: int,
        physical_step: int,
        identity: Mapping,
    ) -> None:
        if (
            self.env_id != env_id
            or self.episode_serial != episode_serial
            or self.physical_step != physical_step
            or self.episode_id != identity.get("episode_id")
            or self.instruction_sha256 != identity.get("instruction_sha256")
        ):
            raise RuntimeError("official frame does not match the live episode and physical step")
        scene = identity.get("scene_id")
        if not isinstance(scene, str) or Path(self.scene_id).resolve() != Path(scene).resolve():
            raise RuntimeError("official frame scene does not match the live episode")

    @classmethod
    def from_service_payload(
        cls,
        payload: object,
        *,
        env_id: int,
        episode_serial: int,
        episode_generation: str,
        physical_step: int,
        identity: Mapping,
    ) -> "OfficialFrame":
        expected = {
            "schema_version",
            "source",
            "state_unchanged",
            "service_step",
            "episode_generation",
            "episode_id",
            "scene_id",
            "instruction_sha256",
            "sensor_contract",
            "contract_sha256",
            "rgb",
            "depth",
            "pose",
            "intrinsics",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("official frame response schema mismatch")
        if (
            payload["schema_version"] != OFFICIAL_FRAME_SCHEMA_VERSION
            or payload["source"] != OFFICIAL_FRAME_SOURCE
            or payload["state_unchanged"] is not True
            or payload["service_step"] != physical_step
            or payload["episode_generation"] != episode_generation
            or payload["sensor_contract"] != OFFICIAL_FRAME_SENSOR_CONTRACT
            or payload["contract_sha256"] != official_frame_contract_sha256()
        ):
            raise ValueError("official frame response provenance mismatch")
        rgb, rgb_hash = _decode_blob(
            payload["rgb"],
            expected_encoding=RGB_ENCODING,
            expected_size=OFFICIAL_RGB_HEIGHT * OFFICIAL_RGB_WIDTH * 3,
        )
        depth, depth_hash = _decode_blob(
            payload["depth"],
            expected_encoding=DEPTH_ENCODING,
            expected_size=OFFICIAL_RGB_HEIGHT * OFFICIAL_RGB_WIDTH * 4,
        )
        pose, pose_hash = _decode_blob(
            payload["pose"], expected_encoding=MATRIX_ENCODING, expected_size=16 * 8
        )
        intrinsics, intrinsics_hash = _decode_blob(
            payload["intrinsics"], expected_encoding=MATRIX_ENCODING, expected_size=16 * 8
        )
        frame = cls(
            env_id=env_id,
            episode_serial=episode_serial,
            episode_generation=payload["episode_generation"],
            physical_step=physical_step,
            episode_id=payload["episode_id"],
            scene_id=payload["scene_id"],
            instruction_sha256=payload["instruction_sha256"],
            rgb_u8=rgb,
            depth_normalized_f32le=depth,
            pose_f64le=pose,
            intrinsics_f64le=intrinsics,
            rgb_sha256=rgb_hash,
            depth_sha256=depth_hash,
            pose_sha256=pose_hash,
            intrinsics_sha256=intrinsics_hash,
        )
        frame.validate_identity(
            env_id=env_id,
            episode_serial=episode_serial,
            physical_step=physical_step,
            identity=identity,
        )
        return frame


def encode_official_blob(payload: bytes, *, encoding: str) -> dict:
    """Encode one service-side blob without importing NumPy in this module."""

    if not isinstance(payload, bytes):
        raise TypeError("official frame service blobs must be bytes")
    return {
        "encoding": encoding,
        "data_b64": base64.b64encode(payload).decode("ascii"),
        "sha256": _sha256(payload),
    }


__all__ = [
    "DEPTH_ENCODING",
    "MATRIX_ENCODING",
    "OFFICIAL_DEPTH_MAX_M",
    "OFFICIAL_DEPTH_MIN_M",
    "OFFICIAL_CAMERA_ORIENTATION_XYZ",
    "OFFICIAL_CAMERA_POSITION_XYZ",
    "OFFICIAL_FORWARD_STEP_M",
    "OFFICIAL_FRAME_SCHEMA_VERSION",
    "OFFICIAL_FRAME_SENSOR_CONTRACT",
    "OFFICIAL_FRAME_SOURCE",
    "OFFICIAL_PROCESSOR_HEIGHT",
    "OFFICIAL_PROCESSOR_WIDTH",
    "OFFICIAL_RGB_HEIGHT",
    "OFFICIAL_RGB_HFOV_DEG",
    "OFFICIAL_RGB_WIDTH",
    "OFFICIAL_TURN_ANGLE_DEG",
    "OfficialFrame",
    "RGB_ENCODING",
    "encode_official_blob",
    "official_frame_contract_sha256",
]
