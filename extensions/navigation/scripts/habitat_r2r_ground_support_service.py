"""Causal pre-action metric-depth collection service for G7.13."""

from __future__ import annotations

import json
import os
import sys
import traceback

if __package__:
    from .habitat_r2r_geometry_controller_service import (
        GeometryControllerBackend,
        GeometryControllerSession,
    )
    from .habitat_r2r_service import parse_args
else:
    from habitat_r2r_geometry_controller_service import (
        GeometryControllerBackend,
        GeometryControllerSession,
    )
    from habitat_r2r_service import parse_args

from streamvln_geometry_g711 import (
    AGENT_RADIUS_M,
    CORRIDOR_HALF_WIDTH_M,
    CX,
    CY,
    DEPTH_HEIGHT,
    DEPTH_WIDTH,
    FX,
    FY,
    HFOV_DEGREES,
    SENSOR_HEIGHT_M,
)
from streamvln_ground_support_g713 import (
    DEPTH_MISSING_EPSILON_M,
    GROUND_FORWARD_MAX_M,
    GROUND_HEIGHT_TOLERANCE_M,
)


SOURCE = "train-simulator-causal-ground-support-observability-audit"
REPEAT_COUNT = 2


def validate_cases(cases):
    if not isinstance(cases, list) or not 1 <= len(cases) <= 32:
        raise ValueError("G7.13 requires between one and 32 observation cases")
    normalized = []
    names = set()
    for case in cases:
        if not isinstance(case, dict) or set(case) != {
            "name",
            "actions",
            "observe_before_forward_index",
        }:
            raise ValueError("G7.13 observation-case schema changed")
        name = case["name"]
        actions = case["actions"]
        critical = case["observe_before_forward_index"]
        if (
            not isinstance(name, str)
            or not name
            or name in names
            or not isinstance(actions, list)
            or not 2 <= len(actions) <= 8
            or any(type(action) is not int or action not in (1, 2, 3) for action in actions)
            or 1 not in actions
            or type(critical) is not int
        ):
            raise ValueError("G7.13 observation-case identity is invalid")
        first_forward = actions.index(1)
        turn_prefix = actions[:first_forward]
        forward_suffix = actions[first_forward:]
        if (
            len(forward_suffix) < 2
            or any(action == 1 for action in turn_prefix)
            or any(action != 1 for action in forward_suffix)
            or not 1 <= critical <= len(forward_suffix)
        ):
            raise ValueError("G7.13 requires turn then forward and a valid critical index")
        names.add(name)
        normalized.append(
            {
                "name": name,
                "actions": actions,
                "turn_prefix": turn_prefix,
                "forward_suffix": forward_suffix,
                "observe_before_forward_index": critical,
            }
        )
    return normalized


class GroundSupportBackend(GeometryControllerBackend):
    def _run_observation(self, case):
        turn_collisions = []
        for action in case["turn_prefix"]:
            self.env.sim.step(action)
            turn_collisions.append(bool(self.env.sim.previous_step_collided))
        forward_direction = [float(value) for value in self._forward_direction()]
        prefix_trace = []
        prefix_count = case["observe_before_forward_index"] - 1
        for primitive_index, action in enumerate(case["forward_suffix"][:prefix_count]):
            before = self.np.asarray(
                self.env.sim.get_agent_state().position, dtype=self.np.float64
            ).copy()
            self.env.sim.step(action)
            after = self.np.asarray(
                self.env.sim.get_agent_state().position, dtype=self.np.float64
            ).copy()
            direction = self.np.asarray(forward_direction, dtype=self.np.float64)
            delta = after - before
            forward_progress = float(self.np.dot(delta, direction))
            lateral = delta - forward_progress * direction
            prefix_trace.append(
                {
                    "primitive_index": primitive_index,
                    "action": int(action),
                    "collision": bool(self.env.sim.previous_step_collided),
                    "before_position": [float(value) for value in before],
                    "after_position": [float(value) for value in after],
                    "displacement_m": float(self.np.linalg.norm(delta)),
                    "forward_progress_m": forward_progress,
                    "lateral_displacement_m": float(self.np.linalg.norm(lateral)),
                }
            )
        metric_depth, depth_sha256, _ = self._depth_snapshot()
        return {
            "turn_collisions": turn_collisions,
            "executed_prefix_count": prefix_count,
            "prefix_trace": prefix_trace,
            "pre_action_pose": self._pose(),
            "forward_direction": forward_direction,
            "metric_depth": metric_depth,
            "depth_sha256": depth_sha256,
        }

    def probe_ground_support(self, cases):
        if self.split != "train":
            raise RuntimeError("G7.13 ground-support audit is train-only")
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
                "agent_radius_m": AGENT_RADIUS_M,
                "corridor_half_width_m": CORRIDOR_HALF_WIDTH_M,
                "ground_forward_max_m": GROUND_FORWARD_MAX_M,
                "ground_height_tolerance_m": GROUND_HEIGHT_TOLERANCE_M,
                "depth_missing_epsilon_m": DEPTH_MISSING_EPSILON_M,
            },
            "max_position_restore_error_m": maximum_position_error,
            "max_rotation_restore_error": maximum_rotation_error,
            "candidates": results,
        }


class GroundSupportSession(GeometryControllerSession):
    def probe_ground_support(self, cases):
        if self.last_frame is None:
            raise RuntimeError("reset must precede G7.13 ground-support audit")
        if self.final_info is not None:
            raise RuntimeError("G7.13 audit is unavailable after episode end")
        return self.backend.probe_ground_support(cases)


def serve_requests(session, request_lines, protocol):
    for line in request_lines:
        try:
            request = json.loads(line)
            command = request.get("cmd")
            if command == "reset":
                response = session.reset()
            elif command == "step":
                response = session.step(request.get("action"))
            elif command == "probe_ground_support":
                response = session.probe_ground_support(request.get("candidates"))
            elif command == "close":
                protocol.write('{"closed":true}\n')
                break
            else:
                raise ValueError("unknown G7.13 ground-support command")
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
    backend = GroundSupportBackend(args)
    session = GroundSupportSession(backend, max_steps=args.max_steps)
    protocol.write(json.dumps(backend.ready(), allow_nan=False) + "\n")
    try:
        serve_requests(session, sys.stdin, protocol)
    finally:
        session.close()
        protocol.close()


if __name__ == "__main__":
    main()
