"""Model-free checks for the private pinned StreamVLN frame channel."""

from pathlib import Path
import copy
import hashlib
import struct
import sys
import unittest


ROOT = Path(__file__).parents[1]
STAGE = ROOT / "integrations" / "rlinf_v03"
sys.path.insert(0, str(STAGE))

from streamvln_p8_official_frame import (  # noqa: E402
    DEPTH_ENCODING,
    MATRIX_ENCODING,
    OFFICIAL_FRAME_SCHEMA_VERSION,
    OFFICIAL_FRAME_SENSOR_CONTRACT,
    OFFICIAL_FRAME_SOURCE,
    OFFICIAL_RGB_HEIGHT,
    OFFICIAL_RGB_WIDTH,
    RGB_ENCODING,
    OfficialFrame,
    encode_official_blob,
    official_frame_contract_sha256,
)


RGB = bytes(OFFICIAL_RGB_HEIGHT * OFFICIAL_RGB_WIDTH * 3)
DEPTH = struct.pack("<f", 0.5) * (OFFICIAL_RGB_HEIGHT * OFFICIAL_RGB_WIDTH)
POSE = struct.pack(
    "<16d",
    0, 0, 1, 0,
    -1, 0, 0, 0,
    0, -1, 0, 1.25,
    0, 0, 0, 1,
)
INTRINSICS = struct.pack(
    "<16d",
    386, 0, 319.5, 0,
    0, 386, 239.5, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
)
INSTRUCTION_HASH = hashlib.sha256(b"walk to the chair").hexdigest()
EPISODE_GENERATION = "0123456789abcdef0123456789abcdef"
IDENTITY = {
    "episode_id": "episode-1",
    "scene_id": "scene.glb",
    "instruction_sha256": INSTRUCTION_HASH,
}


def payload():
    return {
        "schema_version": OFFICIAL_FRAME_SCHEMA_VERSION,
        "source": OFFICIAL_FRAME_SOURCE,
        "state_unchanged": True,
        "service_step": 3,
        "episode_generation": EPISODE_GENERATION,
        "episode_id": IDENTITY["episode_id"],
        "scene_id": IDENTITY["scene_id"],
        "instruction_sha256": INSTRUCTION_HASH,
        "sensor_contract": OFFICIAL_FRAME_SENSOR_CONTRACT,
        "contract_sha256": official_frame_contract_sha256(),
        "rgb": encode_official_blob(RGB, encoding=RGB_ENCODING),
        "depth": encode_official_blob(DEPTH, encoding=DEPTH_ENCODING),
        "pose": encode_official_blob(POSE, encoding=MATRIX_ENCODING),
        "intrinsics": encode_official_blob(INTRINSICS, encoding=MATRIX_ENCODING),
    }


class OfficialFrameTests(unittest.TestCase):
    def test_valid_payload_binds_identity_and_hides_raw_bytes_from_trace(self):
        frame = OfficialFrame.from_service_payload(
            payload(),
            env_id=2,
            episode_serial=7,
            episode_generation=EPISODE_GENERATION,
            physical_step=3,
            identity=IDENTITY,
        )
        self.assertEqual(frame.physical_step, 3)
        self.assertTrue(frame.frame_ref.startswith("sha256:"))
        trace = frame.trace_metadata()
        self.assertEqual(trace["physical_step"], 3)
        self.assertEqual(trace["episode_generation"], EPISODE_GENERATION)
        self.assertNotIn("rgb_u8", trace)
        self.assertNotIn("depth_normalized_f32le", trace)

    def test_payload_hash_tampering_is_rejected(self):
        changed = copy.deepcopy(payload())
        changed["rgb"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "length or SHA-256"):
            OfficialFrame.from_service_payload(
                changed,
                env_id=2,
                episode_serial=7,
                episode_generation=EPISODE_GENERATION,
                physical_step=3,
                identity=IDENTITY,
            )

    def test_service_step_and_episode_identity_are_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "provenance"):
            OfficialFrame.from_service_payload(
                payload(),
                env_id=2,
                episode_serial=7,
                episode_generation=EPISODE_GENERATION,
                physical_step=4,
                identity=IDENTITY,
            )
        stale_generation = copy.deepcopy(payload())
        stale_generation["episode_generation"] = "f" * 32
        with self.assertRaisesRegex(ValueError, "provenance"):
            OfficialFrame.from_service_payload(
                stale_generation,
                env_id=2,
                episode_serial=7,
                episode_generation=EPISODE_GENERATION,
                physical_step=3,
                identity=IDENTITY,
            )
        wrong_identity = dict(IDENTITY, episode_id="other")
        with self.assertRaisesRegex(RuntimeError, "live episode"):
            OfficialFrame.from_service_payload(
                payload(),
                env_id=2,
                episode_serial=7,
                episode_generation=EPISODE_GENERATION,
                physical_step=3,
                identity=wrong_identity,
            )

    def test_nonhomogeneous_pose_is_rejected(self):
        invalid = bytearray(POSE)
        invalid[-8:] = struct.pack("<d", 0.0)
        with self.assertRaisesRegex(ValueError, "homogeneous row"):
            OfficialFrame(
                env_id=2,
                episode_serial=7,
                episode_generation=EPISODE_GENERATION,
                physical_step=3,
                episode_id=IDENTITY["episode_id"],
                scene_id=IDENTITY["scene_id"],
                instruction_sha256=INSTRUCTION_HASH,
                rgb_u8=RGB,
                depth_normalized_f32le=DEPTH,
                pose_f64le=bytes(invalid),
                intrinsics_f64le=INTRINSICS,
                rgb_sha256=hashlib.sha256(RGB).hexdigest(),
                depth_sha256=hashlib.sha256(DEPTH).hexdigest(),
                pose_sha256=hashlib.sha256(invalid).hexdigest(),
                intrinsics_sha256=hashlib.sha256(INTRINSICS).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
