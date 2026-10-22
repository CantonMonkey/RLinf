from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


RLINF_NAV_ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = RLINF_NAV_ROOT / "integrations" / "rlinf_v03"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import lightnav_a3_obs_archive as obs_archive
import lightnav_a3_policy as policy
import lightnav_a3_rollout as rollout
from lightnav_l0_native_probe import HabitatProtocolError, VelocitySpec


def _waypoints() -> list[tuple[float, float, float]]:
    return [
        (0.02 * index, -0.01 * index, math.radians(index))
        for index in range(policy.TRAJECTORY_WIDTH)
    ]


def _policy_action(
    gate: int,
    *,
    bounded_residual: tuple[float, float] = (0.0, 0.0),
) -> policy.PolicyAction:
    return policy.PolicyAction(
        gate=gate,
        raw_residual=bounded_residual,
        bounded_residual=bounded_residual,
        joint_log_prob=0.0,
        value=0.0,
        entropy=0.0,
        gate_probabilities=(1.0, 0.0, 0.0),
    )


class LightNavA3PolicyContractTests(unittest.TestCase):
    def test_build_features_requires_exact_trajectory_and_matching_selection(self):
        waypoints = _waypoints()
        selected_index = 4
        arguments = {
            "selected_index": selected_index,
            "selected_waypoint": waypoints[selected_index],
            "model_stop": False,
            "step_index": 3,
            "max_steps": 20,
            "state": policy.CausalTopologyState(),
        }

        features = policy.build_features(waypoints, **arguments)
        self.assertEqual((policy.FEATURE_DIM,), features.shape)
        self.assertEqual(64, features.size)
        self.assertTrue(np.isfinite(features).all())

        with self.assertRaisesRegex(policy.PolicyContractError, "exactly 10"):
            policy.build_features(waypoints[:-1], **arguments)
        for invalid_index in (-1, policy.TRAJECTORY_WIDTH, True):
            with self.subTest(selected_index=invalid_index):
                with self.assertRaisesRegex(
                    policy.PolicyContractError, "selected index"
                ):
                    policy.build_features(
                        waypoints,
                        **{**arguments, "selected_index": invalid_index},
                    )
        with self.assertRaisesRegex(policy.PolicyContractError, "does not match"):
            policy.build_features(
                waypoints,
                **{
                    **arguments,
                    "selected_waypoint": waypoints[selected_index + 1],
                },
            )

    def test_portable_checkpoint_roundtrip_starts_with_deterministic_keep(self):
        seed = 17
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "lightnav_a3_policy.npz"
            receipt = policy.initialize_checkpoint(checkpoint, seed=seed)
            restored, metadata = policy.load_checkpoint(checkpoint)

            self.assertEqual(0, metadata["policy_version"])
            self.assertEqual(receipt["sha256"], policy.sha256_file(checkpoint))
            self.assertEqual(
                list(policy.initialize_parameters(seed)),
                list(restored),
                "parameter registration order must be identical across torchrun ranks",
            )
            for name, expected in policy.initialize_parameters(seed).items():
                np.testing.assert_array_equal(expected, restored[name])

            portable = policy.NumpyLightNavPolicy.from_checkpoint(checkpoint)
            action = portable.act(
                np.zeros(policy.FEATURE_DIM, dtype=np.float32),
                rng=np.random.default_rng(5),
                deterministic=True,
            )
            self.assertEqual(policy.KEEP_GATE, action.gate)
            self.assertEqual("KEEP", action.gate_name)
            self.assertEqual((0.0, 0.0), action.bounded_residual)

    def test_apply_policy_action_clips_executable_axes_and_honors_stop(self):
        proposal = (0.24, 0.13, math.radians(29.0))
        adjusted = policy.apply_policy_action(
            proposal,
            _policy_action(
                policy.ADJUST_GATE,
                bounded_residual=(0.05, math.radians(10.0)),
            ),
        )
        self.assertAlmostEqual(0.25, adjusted[0])
        self.assertAlmostEqual(proposal[1], adjusted[1])
        self.assertAlmostEqual(math.radians(30.0), adjusted[2])
        self.assertEqual(
            (0.0, 0.0, 0.0),
            policy.apply_policy_action(
                proposal,
                _policy_action(policy.STOP_GATE),
            ),
        )

    def test_causal_topology_integrates_only_realized_path_delta(self):
        state = policy.CausalTopologyState()
        update = {
            "normalized_linear": 1.0,
            "normalized_angular": 0.0,
            "physical_linear_mps": 2.5,
            "physical_angular_dps": 0.0,
            "time_step_s": 0.1,
            "gate": policy.KEEP_GATE,
            "selected_waypoint": (0.25, 0.0, 0.0),
        }

        state.update(path_length_m=0.04, **update)
        self.assertAlmostEqual(0.04, state.x_m)
        self.assertAlmostEqual(0.0, state.y_m)
        self.assertAlmostEqual(0.04, state.last_path_delta_m)

        state.update(path_length_m=0.04, **update)
        self.assertAlmostEqual(0.04, state.x_m)
        self.assertAlmostEqual(0.0, state.last_path_delta_m)
        self.assertTrue(state.blocked_last)


class LightNavA3BoundaryContractTests(unittest.TestCase):
    def test_rollout_velocity_contract_accepts_only_executable_step_bounds(self):
        expected = VelocitySpec(0.1, (0.0, 2.5), (-300.0, 300.0))
        self.assertIsNone(rollout._validate_velocity_contract(expected))

        incompatible = (
            VelocitySpec(0.1, (0.0, 2.0), (-300.0, 300.0)),
            VelocitySpec(0.1, (0.0, 2.5), (-200.0, 200.0)),
        )
        for spec in incompatible:
            with self.subTest(spec=spec):
                with self.assertRaisesRegex(
                    HabitatProtocolError, r"\[0, 0\.25\].*\[-30, 30\]"
                ):
                    rollout._validate_velocity_contract(spec)

    def test_obs_completion_marker_and_copy_are_read_back_by_hash(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            completion = root / "obs" / "COMPLETE"
            payload = {
                "schema_name": "lightnav_a3_obs_completion",
                "schema_version": 1,
                "phase": "final",
            }
            expected_hash = obs_archive._atomic_json(completion, payload)
            self.assertEqual(expected_hash, obs_archive.sha256_file(completion))
            self.assertEqual(payload, json.loads(completion.read_text(encoding="utf-8")))

            copied = root / "readback" / "COMPLETE"
            receipt = obs_archive._copy_verified(completion, copied)
            self.assertEqual(expected_hash, receipt["sha256"])
            self.assertEqual(expected_hash, receipt["readback_sha256"])
            copied.write_bytes(b"tampered")
            with self.assertRaisesRegex(obs_archive.ArchiveError, "conflicts"):
                obs_archive._copy_verified(completion, copied)


if __name__ == "__main__":
    unittest.main()
