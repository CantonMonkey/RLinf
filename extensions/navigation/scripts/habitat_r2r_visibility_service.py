"""Fixed pitched-depth immediate-footprint visibility service for G7.14."""

from __future__ import annotations

import json
import os
import sys
import traceback

if __package__:
    from .habitat_r2r_ground_support_service import (
        GroundSupportBackend,
        GroundSupportSession,
        validate_cases,
    )
    from .habitat_r2r_service import parse_args
else:
    from habitat_r2r_ground_support_service import (
        GroundSupportBackend,
        GroundSupportSession,
        validate_cases,
    )
    from habitat_r2r_service import parse_args

from streamvln_geometry_g711 import (
    AGENT_RADIUS_M,
    CORRIDOR_HALF_WIDTH_M,
    CX,
    CY,
    DEPTH_HEIGHT,
    DEPTH_WIDTH,
    FORWARD_STEP_M,
    FX,
    FY,
    HFOV_DEGREES,
    SENSOR_HEIGHT_M,
)
from streamvln_visibility_g714 import (
    DOWNWARD_PITCH_DEGREES,
    DOWNWARD_PITCH_RADIANS,
    EXPECTED_OPTICAL_AXIS_AGENT_FRAME,
    EXPECTED_SENSOR_POSITION_AGENT_FRAME,
    FROZEN_GEOMETRY_SUMMARY,
)


SOURCE = "train-simulator-immediate-footprint-visual-visibility-ceiling"
REPEAT_COUNT = 2


class ImmediateFootprintVisibilityBackend(GroundSupportBackend):
    def __init__(self, args):
        args.depth_pitch_radians = DOWNWARD_PITCH_RADIANS
        super().__init__(args)

    def ready(self):
        ready = super().ready()
        ready.update(
            {
                "g714_depth_pitch_radians": DOWNWARD_PITCH_RADIANS,
                "g714_depth_pitch_degrees": DOWNWARD_PITCH_DEGREES,
                "g714_depth_orientation_xyz_radians": [
                    DOWNWARD_PITCH_RADIANS,
                    0.0,
                    0.0,
                ],
                "g714_policy_rgb_orientation_unchanged": True,
            }
        )
        return ready

    def _sensor_extrinsics(self):
        from habitat_sim.utils.common import quat_rotate_vector

        state = self.env.sim.get_agent_state()
        sensors = state.sensor_states
        if not isinstance(sensors, dict) or set(sensors) != {"rgb", "depth"}:
            raise RuntimeError("G7.14 active sensor-state set changed")
        sensor = sensors["depth"]
        inverse_agent = state.rotation.inverse()
        relative_rotation = inverse_agent * sensor.rotation
        relative_position = quat_rotate_vector(
            inverse_agent,
            self.np.asarray(sensor.position, dtype=self.np.float64)
            - self.np.asarray(state.position, dtype=self.np.float64),
        )
        optical_axis = quat_rotate_vector(
            relative_rotation,
            self.np.asarray([0.0, 0.0, -1.0], dtype=self.np.float64),
        )
        return {
            "relative_position_agent_frame": [
                float(value) for value in relative_position
            ],
            "optical_axis_agent_frame": [float(value) for value in optical_axis],
            "expected_relative_position_agent_frame": list(
                EXPECTED_SENSOR_POSITION_AGENT_FRAME
            ),
            "expected_optical_axis_agent_frame": list(
                EXPECTED_OPTICAL_AXIS_AGENT_FRAME
            ),
        }

    def _run_observation(self, case):
        result = super()._run_observation(case)
        result["sensor_extrinsics"] = self._sensor_extrinsics()
        return result

    def probe_immediate_footprint_visibility(self, cases):
        if self.split != "train":
            raise RuntimeError("G7.14 visibility ceiling is train-only")
        cases = validate_cases(cases)
        start_position, start_rotation, start_coefficients = self._copy_agent_state()
        results = []
        maximum_position_error = maximum_rotation_error = 0.0
        for case in cases:
            repeats = []
            for _ in range(REPEAT_COUNT):
                try:
                    repeats.append(self._run_observation(case))
                finally:
                    position_error, rotation_error = self._restore_agent_state(
                        start_position, start_rotation, start_coefficients
                    )
                    maximum_position_error = max(maximum_position_error, position_error)
                    maximum_rotation_error = max(maximum_rotation_error, rotation_error)
            results.append({**case, "repeats": repeats})
        return {
            "schema_version": 1,
            "source": SOURCE,
            "policy_observation_unchanged": True,
            "state_restored": True,
            "repeat_count": REPEAT_COUNT,
            "observation_contract": {
                "timing": "after-turn-and-critical-prefix-minus-one-forwards; before-critical-forward",
                "critical_primitive_executed": False,
                "depth_height": DEPTH_HEIGHT,
                "depth_width": DEPTH_WIDTH,
                "hfov_degrees": HFOV_DEGREES,
                "fx": FX,
                "fy": FY,
                "cx": CX,
                "cy": CY,
                "sensor_height_m": SENSOR_HEIGHT_M,
                "forward_step_m": FORWARD_STEP_M,
                "agent_radius_m": AGENT_RADIUS_M,
                "corridor_half_width_m": CORRIDOR_HALF_WIDTH_M,
                "depth_pitch_radians": DOWNWARD_PITCH_RADIANS,
                "depth_pitch_degrees": DOWNWARD_PITCH_DEGREES,
                "depth_orientation_xyz_radians": [
                    DOWNWARD_PITCH_RADIANS,
                    0.0,
                    0.0,
                ],
                "policy_rgb_orientation_unchanged": True,
                "target_geometry": FROZEN_GEOMETRY_SUMMARY,
            },
            "max_position_restore_error_m": maximum_position_error,
            "max_rotation_restore_error": maximum_rotation_error,
            "candidates": results,
        }


class ImmediateFootprintVisibilitySession(GroundSupportSession):
    def probe_immediate_footprint_visibility(self, cases):
        if self.last_frame is None:
            raise RuntimeError("reset must precede G7.14 visibility audit")
        if self.final_info is not None:
            raise RuntimeError("G7.14 visibility audit is unavailable after episode end")
        return self.backend.probe_immediate_footprint_visibility(cases)


def serve_requests(session, request_lines, protocol):
    for line in request_lines:
        try:
            request = json.loads(line)
            command = request.get("cmd")
            if command == "reset":
                response = session.reset()
            elif command == "step":
                response = session.step(request.get("action"))
            elif command == "probe_immediate_footprint_visibility":
                response = session.probe_immediate_footprint_visibility(
                    request.get("candidates")
                )
            elif command == "close":
                protocol.write('{"closed":true}\n')
                break
            else:
                raise ValueError("unknown G7.14 visibility command")
            protocol.write(
                json.dumps(response, allow_nan=False, separators=(",", ":")) + "\n"
            )
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            protocol.write(json.dumps({"error": str(exc)}) + "\n")
            break


def main(argv=None):
    args = parse_args(argv)
    sys.stdout.flush()
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    backend = ImmediateFootprintVisibilityBackend(args)
    session = ImmediateFootprintVisibilitySession(backend, max_steps=args.max_steps)
    protocol.write(json.dumps(backend.ready(), allow_nan=False) + "\n")
    try:
        serve_requests(session, sys.stdin, protocol)
    finally:
        session.close()
        protocol.close()


if __name__ == "__main__":
    main()
