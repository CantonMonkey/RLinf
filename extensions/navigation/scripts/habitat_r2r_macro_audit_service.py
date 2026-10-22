"""Train-only exact-macro collision-horizon and controllability audit for G7.9."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import sys
import traceback

if __package__:
    from .habitat_r2r_depth_passability_service import (
        DepthPassabilityBackend,
        DepthPassabilitySession,
    )
    from .habitat_r2r_passability_service import validate_probe_candidates
    from .habitat_r2r_service import parse_args
else:
    from habitat_r2r_depth_passability_service import (
        DepthPassabilityBackend,
        DepthPassabilitySession,
    )
    from habitat_r2r_passability_service import validate_probe_candidates
    from habitat_r2r_service import parse_args


AUDIT_SOURCE = "train-simulator-exact-macro-controllability-audit"
FORWARD_STEP_METERS = 0.25
HORIZON_FORWARD_STEPS = 2
HORIZON_METERS = FORWARD_STEP_METERS * HORIZON_FORWARD_STEPS
REPEAT_COUNT = 2


class MacroAuditBackend(DepthPassabilityBackend):
    def _pose(self):
        from habitat_sim.utils.common import quat_to_coeffs

        state = self.env.sim.get_agent_state()
        position = self.np.asarray(state.position, dtype=self.np.float64)
        rotation = self.np.asarray(
            quat_to_coeffs(state.rotation), dtype=self.np.float64
        )
        return {
            "position": [float(value) for value in position],
            "rotation_coefficients": [float(value) for value in rotation],
        }

    def _forward_direction(self):
        from habitat_sim.utils.common import quat_rotate_vector

        state = self.env.sim.get_agent_state()
        direction = self.np.asarray(
            quat_rotate_vector(
                state.rotation,
                self.np.asarray([0.0, 0.0, -1.0], dtype=self.np.float64),
            ),
            dtype=self.np.float64,
        )
        direction[1] = 0.0
        norm = float(self.np.linalg.norm(direction))
        if not math.isfinite(norm) or norm <= 1e-12:
            raise RuntimeError("macro audit forward direction is invalid")
        return direction / norm

    @staticmethod
    def _depth_sha256(metric_depth):
        raw = base64.b64decode(metric_depth["data_b64"], validate=True)
        return hashlib.sha256(raw).hexdigest()

    def _rotation_error_degrees(self, reference, actual):
        first = self.np.asarray(reference, dtype=self.np.float64)
        second = self.np.asarray(actual, dtype=self.np.float64)
        first /= self.np.linalg.norm(first)
        second /= self.np.linalg.norm(second)
        dot = min(1.0, max(0.0, abs(float(self.np.dot(first, second)))))
        return float(math.degrees(2.0 * math.acos(dot)))

    def _single_repeat(self, turn_prefix, forward_suffix):
        turn_collisions = []
        for action in turn_prefix:
            self.env.sim.step(action)
            turn_collisions.append(bool(self.env.sim.previous_step_collided))
        post_turn_pose = self._pose()
        metric_depth = self._metric_depth()
        depth_sha256 = self._depth_sha256(metric_depth)
        forward_direction = self._forward_direction()
        post_turn_position = self.np.asarray(
            post_turn_pose["position"], dtype=self.np.float64
        )
        commanded_distance_m = len(forward_suffix) * FORWARD_STEP_METERS
        intended_endpoint = (
            post_turn_position + forward_direction * commanded_distance_m
        )
        pathfinder = self.env.sim.pathfinder
        privileged_endpoint = self.np.asarray(
            pathfinder.try_step(post_turn_position, intended_endpoint),
            dtype=self.np.float64,
        )
        privileged_endpoint_error_m = float(
            self.np.linalg.norm(privileged_endpoint - intended_endpoint)
        )
        primitive_trace = []
        for primitive_index, action in enumerate(forward_suffix):
            before = self.np.asarray(
                self.env.sim.get_agent_state().position, dtype=self.np.float64
            ).copy()
            self.env.sim.step(action)
            after = self.np.asarray(
                self.env.sim.get_agent_state().position, dtype=self.np.float64
            ).copy()
            delta = after - before
            primitive_trace.append(
                {
                    "primitive_index": primitive_index,
                    "action": int(action),
                    "collision": bool(self.env.sim.previous_step_collided),
                    "before_position": [float(value) for value in before],
                    "after_position": [float(value) for value in after],
                    "displacement_m": float(self.np.linalg.norm(delta)),
                    "forward_progress_m": float(self.np.dot(delta, forward_direction)),
                }
            )
        final_pose = self._pose()
        final_position = self.np.asarray(
            final_pose["position"], dtype=self.np.float64
        )
        realized = final_position - post_turn_position
        realized_forward_m = float(self.np.dot(realized, forward_direction))
        lateral = realized - realized_forward_m * forward_direction
        collision_trace = [item["collision"] for item in primitive_trace]
        return {
            "turn_collisions": turn_collisions,
            "post_turn_pose": post_turn_pose,
            "metric_depth": metric_depth,
            "depth_sha256": depth_sha256,
            "forward_direction": [float(value) for value in forward_direction],
            "commanded_distance_m": commanded_distance_m,
            "intended_endpoint": [float(value) for value in intended_endpoint],
            "privileged_endpoint": [float(value) for value in privileged_endpoint],
            "privileged_endpoint_error_m": privileged_endpoint_error_m,
            "privileged_swept_path_free": privileged_endpoint_error_m <= 1e-4,
            "primitive_trace": primitive_trace,
            "full_collision_trace": collision_trace,
            "full_unsafe": any(collision_trace),
            "horizon_collision_trace": collision_trace[:HORIZON_FORWARD_STEPS],
            "horizon_unsafe": any(collision_trace[:HORIZON_FORWARD_STEPS]),
            "final_pose": final_pose,
            "realized_forward_m": realized_forward_m,
            "realized_lateral_error_m": float(self.np.linalg.norm(lateral)),
            "endpoint_error_m": float(
                self.np.linalg.norm(final_position - intended_endpoint)
            ),
            "heading_error_degrees": self._rotation_error_degrees(
                post_turn_pose["rotation_coefficients"],
                final_pose["rotation_coefficients"],
            ),
        }

    def probe_macro_audit(self, candidates):
        if self.split != "train":
            raise RuntimeError("macro audit is train-only")
        rows = validate_probe_candidates(candidates)
        start_position, start_rotation, start_coefficients = self._copy_agent_state()
        results = []
        maximum_position_error = maximum_rotation_error = 0.0
        for row in rows:
            actions = row["actions"]
            first_forward = actions.index(1)
            turn_prefix = actions[:first_forward]
            forward_suffix = actions[first_forward:]
            if (
                len(forward_suffix) < HORIZON_FORWARD_STEPS
                or any(action == 1 for action in turn_prefix)
                or any(action != 1 for action in forward_suffix)
            ):
                raise ValueError("macro audit requires turn prefix and >=0.5m forward suffix")
            repeats = []
            for _ in range(REPEAT_COUNT):
                try:
                    repeats.append(
                        self._single_repeat(turn_prefix, forward_suffix)
                    )
                finally:
                    position_error, rotation_error = self._restore_agent_state(
                        start_position, start_rotation, start_coefficients
                    )
                    maximum_position_error = max(
                        maximum_position_error, position_error
                    )
                    maximum_rotation_error = max(
                        maximum_rotation_error, rotation_error
                    )
            results.append(
                {
                    "name": row["name"],
                    "actions": actions,
                    "turn_prefix": turn_prefix,
                    "forward_suffix": forward_suffix,
                    "repeats": repeats,
                }
            )
        if any(
            not math.isfinite(value)
            for value in (maximum_position_error, maximum_rotation_error)
        ):
            raise RuntimeError("macro audit restoration error is nonfinite")
        return {
            "schema_version": 1,
            "source": AUDIT_SOURCE,
            "policy_observation_unchanged": True,
            "state_restored": True,
            "forward_step_m": FORWARD_STEP_METERS,
            "horizon_forward_steps": HORIZON_FORWARD_STEPS,
            "physical_horizon_m": HORIZON_METERS,
            "repeat_count": REPEAT_COUNT,
            "max_position_restore_error_m": maximum_position_error,
            "max_rotation_restore_error": maximum_rotation_error,
            "candidates": results,
        }


class MacroAuditSession(DepthPassabilitySession):
    def probe_macro_audit(self, candidates):
        if self.last_frame is None:
            raise RuntimeError("reset must precede macro audit")
        if self.final_info is not None:
            raise RuntimeError("macro audit is unavailable after episode end")
        return self.backend.probe_macro_audit(candidates)


def serve_requests(session, request_lines, protocol):
    for line in request_lines:
        try:
            request = json.loads(line)
            command = request.get("cmd")
            if command == "reset":
                response = session.reset()
            elif command == "step":
                response = session.step(request.get("action"))
            elif command == "expert_waypoint":
                response = session.expert_waypoint(request.get("lookahead_m"))
            elif command == "observe_aligned_depth":
                response = session.observe_aligned_depth()
            elif command == "probe_macro_audit":
                response = session.probe_macro_audit(request.get("candidates"))
            elif command == "close":
                protocol.write('{"closed":true}\n')
                break
            else:
                raise ValueError("unknown macro-audit command")
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
    backend = MacroAuditBackend(args)
    session = MacroAuditSession(backend, max_steps=args.max_steps)
    protocol.write(json.dumps(backend.ready(), allow_nan=False) + "\n")
    try:
        serve_requests(session, sys.stdin, protocol)
    finally:
        session.close()
        protocol.close()


if __name__ == "__main__":
    main()
