"""Frozen pitched-depth plus one-critical-forward outcome service for G7.15."""

from __future__ import annotations

import json
import os
import sys
import traceback

if __package__:
    from .habitat_r2r_ground_support_service import validate_cases
    from .habitat_r2r_service import parse_args
    from .habitat_r2r_visibility_service import (
        ImmediateFootprintVisibilityBackend,
        ImmediateFootprintVisibilitySession,
    )
else:
    from habitat_r2r_ground_support_service import validate_cases
    from habitat_r2r_service import parse_args
    from habitat_r2r_visibility_service import (
        ImmediateFootprintVisibilityBackend,
        ImmediateFootprintVisibilitySession,
    )

from streamvln_geometry_g711 import FORWARD_STEP_M
from streamvln_visibility_g714 import (
    DOWNWARD_PITCH_DEGREES,
    DOWNWARD_PITCH_RADIANS,
    FROZEN_GEOMETRY_SUMMARY,
)


SOURCE = "train-simulator-frozen-immediate-footprint-executed-outcome-audit"
REPEAT_COUNT = 2


class ImmediateFootprintOutcomeBackend(ImmediateFootprintVisibilityBackend):
    def ready(self):
        ready = super().ready()
        ready.update(
            {
                "g715_exact_critical_action": 1,
                "g715_exact_critical_action_count_per_repeat": 1,
                "g715_commanded_distance_m": FORWARD_STEP_M,
            }
        )
        return ready

    def _run_outcome(self, case):
        observation = self._run_observation(case)
        before = self.np.asarray(
            observation["pre_action_pose"]["position"], dtype=self.np.float64
        )
        direction = self.np.asarray(
            observation["forward_direction"], dtype=self.np.float64
        )
        intended = before + direction * FORWARD_STEP_M
        self.env.sim.step(1)
        after_pose = self._pose()
        after = self.np.asarray(after_pose["position"], dtype=self.np.float64)
        delta = after - before
        progress = float(self.np.dot(delta, direction))
        lateral = delta - progress * direction
        observation.update(
            {
                "critical_trace": {
                    "primitive_index": 0,
                    "action": 1,
                    "collision": bool(self.env.sim.previous_step_collided),
                    "before_position": [float(value) for value in before],
                    "after_position": [float(value) for value in after],
                    "intended_endpoint": [float(value) for value in intended],
                    "displacement_m": float(self.np.linalg.norm(delta)),
                    "forward_progress_m": progress,
                    "lateral_displacement_m": float(self.np.linalg.norm(lateral)),
                    "endpoint_error_m": float(self.np.linalg.norm(after - intended)),
                },
                "post_action_pose": after_pose,
            }
        )
        return observation

    def probe_immediate_footprint_outcomes(self, cases):
        if self.split != "train":
            raise RuntimeError("G7.15 executed-outcome audit is train-only")
        cases = validate_cases(cases)
        if any(
            case["observe_before_forward_index"] != len(case["forward_suffix"])
            for case in cases
        ):
            raise ValueError("G7.15 must observe before the final forward primitive")
        start_position, start_rotation, start_coefficients = self._copy_agent_state()
        results = []
        maximum_position_error = maximum_rotation_error = 0.0
        for case in cases:
            repeats = []
            for _ in range(REPEAT_COUNT):
                try:
                    repeats.append(self._run_outcome(case))
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
                "timing": "after-turn-and-all-but-final-forward; before-final-forward",
                "critical_primitive_executed": True,
                "critical_action": 1,
                "critical_action_count_per_repeat": 1,
                "commanded_distance_m": FORWARD_STEP_M,
                "depth_height": 64,
                "depth_width": 64,
                "hfov_degrees": 90.0,
                "fx": 32.0,
                "fy": 32.0,
                "cx": 31.5,
                "cy": 31.5,
                "sensor_height_m": 1.25,
                "forward_step_m": FORWARD_STEP_M,
                "agent_radius_m": 0.1,
                "corridor_half_width_m": 0.12,
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


class ImmediateFootprintOutcomeSession(ImmediateFootprintVisibilitySession):
    def probe_immediate_footprint_outcomes(self, cases):
        if self.last_frame is None:
            raise RuntimeError("reset must precede G7.15 executed-outcome audit")
        if self.final_info is not None:
            raise RuntimeError("G7.15 audit is unavailable after episode end")
        return self.backend.probe_immediate_footprint_outcomes(cases)


def serve_requests(session, request_lines, protocol):
    for line in request_lines:
        try:
            request = json.loads(line)
            command = request.get("cmd")
            if command == "reset":
                response = session.reset()
            elif command == "step":
                response = session.step(request.get("action"))
            elif command == "probe_immediate_footprint_outcomes":
                response = session.probe_immediate_footprint_outcomes(
                    request.get("candidates")
                )
            elif command == "close":
                protocol.write('{"closed":true}\n')
                break
            else:
                raise ValueError("unknown G7.15 executed-outcome command")
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
    backend = ImmediateFootprintOutcomeBackend(args)
    session = ImmediateFootprintOutcomeSession(backend, max_steps=args.max_steps)
    protocol.write(json.dumps(backend.ready(), allow_nan=False) + "\n")
    try:
        serve_requests(session, sys.stdin, protocol)
    finally:
        session.close()
        protocol.close()


if __name__ == "__main__":
    main()
