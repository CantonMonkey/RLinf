"""Training-free explicit geometry controller audit for G7.11."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import sys
import traceback

if __package__:
    from .habitat_r2r_severity_audit_service import (
        SeverityAuditBackend,
        SeverityAuditSession,
    )
    from .habitat_r2r_passability_service import validate_probe_candidates
    from .habitat_r2r_service import parse_args
else:
    from habitat_r2r_severity_audit_service import (
        SeverityAuditBackend,
        SeverityAuditSession,
    )
    from habitat_r2r_passability_service import validate_probe_candidates
    from habitat_r2r_service import parse_args

from streamvln_geometry_g711 import (
    AGENT_HEIGHT_M,
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
    OBSTACLE_MAX_HEIGHT_M,
    OBSTACLE_MIN_HEIGHT_M,
    PRIMITIVE_BRAKE_DISTANCE_M,
    SAFETY_MARGIN_M,
    SENSOR_HEIGHT_M,
    controlled_motion_severity,
    depth_corridor_measurement,
)


SOURCE = "train-simulator-explicit-depth-footprint-controller-audit"
VARIANTS = (
    "open_loop",
    "endpoint_geometry",
    "depth_receding",
    "navmesh_receding_oracle",
)
REPEAT_COUNT = 2
ENDPOINT_TOLERANCE_M = 0.001
NAVMESH_SAMPLE_SPACING_M = 0.125
NAVMESH_MINIMUM_CLEARANCE_M = 0.02
NAVMESH_SEARCH_RADIUS_M = 2.0


class GeometryControllerBackend(SeverityAuditBackend):
    @staticmethod
    def _depth_sha256(payload):
        return hashlib.sha256(
            base64.b64decode(payload["data_b64"], validate=True)
        ).hexdigest()

    def _depth_snapshot(self):
        payload = self._metric_depth()
        raw = base64.b64decode(payload["data_b64"], validate=True)
        depth = self.np.frombuffer(raw, dtype="<f4").copy().reshape(
            DEPTH_HEIGHT, DEPTH_WIDTH
        )
        return payload, self._depth_sha256(payload), depth_corridor_measurement(depth)

    def _navmesh_step_decision(self, start, intended):
        pathfinder = self.env.sim.pathfinder
        start = self.np.asarray(start, dtype=self.np.float64)
        intended = self.np.asarray(intended, dtype=self.np.float64)
        filtered = self.np.asarray(
            pathfinder.try_step_no_sliding(
                start.astype(self.np.float32), intended.astype(self.np.float32)
            ),
            dtype=self.np.float64,
        )
        endpoint_error = float(self.np.linalg.norm(filtered - intended))
        distance = float(self.np.linalg.norm(intended - start))
        count = max(1, int(math.ceil(distance / NAVMESH_SAMPLE_SPACING_M)))
        samples = []
        for index in range(count + 1):
            point = start + (intended - start) * (index / count)
            point32 = point.astype(self.np.float32)
            clearance = float(
                pathfinder.distance_to_closest_obstacle(
                    point32, NAVMESH_SEARCH_RADIUS_M
                )
            )
            if math.isnan(clearance) or clearance < 0.0:
                raise RuntimeError("G7.11 navmesh clearance is invalid")
            clearance = (
                min(clearance, NAVMESH_SEARCH_RADIUS_M)
                if math.isfinite(clearance)
                else NAVMESH_SEARCH_RADIUS_M
            )
            samples.append(
                {
                    "position": [float(value) for value in point],
                    "navigable": bool(pathfinder.is_navigable(point32)),
                    "clearance_m": float(clearance),
                }
            )
        allow = (
            endpoint_error <= ENDPOINT_TOLERANCE_M
            and all(item["navigable"] for item in samples)
            and min(item["clearance_m"] for item in samples)
            >= NAVMESH_MINIMUM_CLEARANCE_M
        )
        return {
            "allow_forward": bool(allow),
            "endpoint_error_m": endpoint_error,
            "samples": samples,
        }

    def _run_variant(self, turn_prefix, forward_suffix, variant):
        turn_collisions = []
        for action in turn_prefix:
            self.env.sim.step(action)
            turn_collisions.append(bool(self.env.sim.previous_step_collided))
        post_turn_pose = self._pose()
        forward_direction = self._forward_direction()
        start = self.np.asarray(post_turn_pose["position"], dtype=self.np.float64)
        full_commanded = len(forward_suffix) * FORWARD_STEP_M
        full_intended = start + forward_direction * full_commanded
        initial_payload, initial_depth_sha, initial_measurement = self._depth_snapshot()
        pathfinder = self.env.sim.pathfinder
        endpoint_filtered = self.np.asarray(
            pathfinder.try_step_no_sliding(
                start.astype(self.np.float32), full_intended.astype(self.np.float32)
            ),
            dtype=self.np.float64,
        )
        endpoint_geometry_error = float(
            self.np.linalg.norm(endpoint_filtered - full_intended)
        )
        endpoint_geometry_allow = endpoint_geometry_error <= ENDPOINT_TOLERANCE_M
        decisions = []
        primitive_trace = []
        brake_index = None
        for primitive_index, action in enumerate(forward_suffix):
            before = self.np.asarray(
                self.env.sim.get_agent_state().position, dtype=self.np.float64
            ).copy()
            intended_step = before + forward_direction * FORWARD_STEP_M
            if variant == "open_loop":
                decision = {"allow_forward": True, "source": "open-loop"}
            elif variant == "endpoint_geometry":
                decision = {
                    "allow_forward": bool(endpoint_geometry_allow),
                    "source": "privileged-full-endpoint-only",
                    "full_endpoint_error_m": endpoint_geometry_error,
                }
            elif variant == "depth_receding":
                _, depth_sha, measurement = self._depth_snapshot()
                decision = {
                    **measurement,
                    "depth_sha256": depth_sha,
                    "source": "causal-metric-depth-footprint-corridor",
                }
            elif variant == "navmesh_receding_oracle":
                decision = {
                    **self._navmesh_step_decision(before, intended_step),
                    "source": "privileged-navmesh-receding-oracle",
                }
            else:
                raise ValueError("unknown G7.11 controller variant")
            decision = {"primitive_index": primitive_index, **decision}
            decisions.append(decision)
            if not decision["allow_forward"]:
                brake_index = primitive_index
                break
            self.env.sim.step(action)
            after = self.np.asarray(
                self.env.sim.get_agent_state().position, dtype=self.np.float64
            ).copy()
            delta = after - before
            forward_progress = float(self.np.dot(delta, forward_direction))
            lateral = delta - forward_progress * forward_direction
            primitive_trace.append(
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
        final_pose = self._pose()
        final_position = self.np.asarray(
            final_pose["position"], dtype=self.np.float64
        )
        realized = final_position - start
        realized_forward = float(self.np.dot(realized, forward_direction))
        realized_lateral = realized - realized_forward * forward_direction
        attempted = len(primitive_trace) * FORWARD_STEP_M
        attempted_endpoint = start + forward_direction * attempted
        full_intended_trace = self._segment_clearance_trace(start, full_intended)
        actual_positions = [start] + [
            item["after_position"] for item in primitive_trace
        ]
        actual_trace = []
        cumulative = 0.0
        previous = self.np.asarray(actual_positions[0], dtype=self.np.float64)
        for position in actual_positions:
            current = self.np.asarray(position, dtype=self.np.float64)
            if actual_trace:
                cumulative += float(self.np.linalg.norm(current - previous))
            actual_trace.append(self._clearance_sample(current, cumulative))
            previous = current
        result = {
            "variant": variant,
            "turn_collisions": turn_collisions,
            "post_turn_pose": post_turn_pose,
            "forward_direction": [float(value) for value in forward_direction],
            "initial_depth_sha256": initial_depth_sha,
            "initial_corridor_measurement": initial_measurement,
            "full_commanded_distance_m": full_commanded,
            "full_intended_endpoint": [float(value) for value in full_intended],
            "endpoint_geometry_filtered_endpoint": [
                float(value) for value in endpoint_filtered
            ],
            "endpoint_geometry_error_m": endpoint_geometry_error,
            "endpoint_geometry_allow": bool(endpoint_geometry_allow),
            "decisions": decisions,
            "brake_index": brake_index,
            "primitive_trace": primitive_trace,
            "executed_forward_primitives": len(primitive_trace),
            "commanded_forward_primitives": len(forward_suffix),
            "attempted_distance_m": attempted,
            "completed": len(primitive_trace) == len(forward_suffix),
            "raw_collision_trace": [
                item["collision"] for item in primitive_trace
            ],
            "final_pose": final_pose,
            "realized_forward_m": realized_forward,
            "realized_lateral_error_m": float(
                self.np.linalg.norm(realized_lateral)
            ),
            "attempted_endpoint_error_m": float(
                self.np.linalg.norm(final_position - attempted_endpoint)
            ),
            "full_endpoint_error_m": float(
                self.np.linalg.norm(final_position - full_intended)
            ),
            "intended_clearance_trace": full_intended_trace,
            "actual_clearance_trace": actual_trace,
            "minimum_intended_clearance_m": min(
                item["clearance_m"] for item in full_intended_trace
            ),
            "minimum_actual_clearance_m": min(
                item["clearance_m"] for item in actual_trace
            ),
        }
        result["controlled_motion"] = controlled_motion_severity(result)
        result["severe_motion_exposure"] = bool(
            result["controlled_motion"]
            and result["controlled_motion"]["severe_exposure"]
        )
        return initial_payload, result

    def probe_geometry_controller(self, candidates):
        if self.split != "train":
            raise RuntimeError("G7.11 geometry controller audit is train-only")
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
                len(forward_suffix) < 2
                or any(action == 1 for action in turn_prefix)
                or any(action != 1 for action in forward_suffix)
            ):
                raise ValueError("G7.11 requires turn prefix then >=2 forwards")
            variants = {}
            candidate_depth = None
            candidate_depth_sha = None
            for variant in VARIANTS:
                repeats = []
                for _ in range(REPEAT_COUNT):
                    try:
                        depth_payload, repeat = self._run_variant(
                            turn_prefix, forward_suffix, variant
                        )
                        if candidate_depth is None:
                            candidate_depth = depth_payload
                            candidate_depth_sha = repeat["initial_depth_sha256"]
                        elif repeat["initial_depth_sha256"] != candidate_depth_sha:
                            raise RuntimeError("G7.11 initial aligned depth changed")
                        repeats.append(repeat)
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
                variants[variant] = repeats
            results.append(
                {
                    "name": row["name"],
                    "actions": actions,
                    "turn_prefix": turn_prefix,
                    "forward_suffix": forward_suffix,
                    "initial_metric_depth": candidate_depth,
                    "initial_depth_sha256": candidate_depth_sha,
                    "variants": variants,
                }
            )
        return {
            "schema_version": 1,
            "source": SOURCE,
            "policy_observation_unchanged": True,
            "state_restored": True,
            "repeat_count": REPEAT_COUNT,
            "variants": list(VARIANTS),
            "depth_geometry": {
                "height": DEPTH_HEIGHT,
                "width": DEPTH_WIDTH,
                "hfov_degrees": HFOV_DEGREES,
                "fx": FX,
                "fy": FY,
                "cx": CX,
                "cy": CY,
                "sensor_height_m": SENSOR_HEIGHT_M,
                "agent_height_m": AGENT_HEIGHT_M,
                "agent_radius_m": AGENT_RADIUS_M,
                "safety_margin_m": SAFETY_MARGIN_M,
                "corridor_half_width_m": CORRIDOR_HALF_WIDTH_M,
                "obstacle_min_height_m": OBSTACLE_MIN_HEIGHT_M,
                "obstacle_max_height_m": OBSTACLE_MAX_HEIGHT_M,
                "primitive_brake_distance_m": PRIMITIVE_BRAKE_DISTANCE_M,
            },
            "endpoint_tolerance_m": ENDPOINT_TOLERANCE_M,
            "navmesh_sample_spacing_m": NAVMESH_SAMPLE_SPACING_M,
            "navmesh_minimum_clearance_m": NAVMESH_MINIMUM_CLEARANCE_M,
            "max_position_restore_error_m": maximum_position_error,
            "max_rotation_restore_error": maximum_rotation_error,
            "candidates": results,
        }


class GeometryControllerSession(SeverityAuditSession):
    def probe_geometry_controller(self, candidates):
        if self.last_frame is None:
            raise RuntimeError("reset must precede G7.11 controller audit")
        if self.final_info is not None:
            raise RuntimeError("G7.11 audit is unavailable after episode end")
        return self.backend.probe_geometry_controller(candidates)


def serve_requests(session, request_lines, protocol):
    for line in request_lines:
        try:
            request = json.loads(line)
            command = request.get("cmd")
            if command == "reset":
                response = session.reset()
            elif command == "step":
                response = session.step(request.get("action"))
            elif command == "probe_geometry_controller":
                response = session.probe_geometry_controller(
                    request.get("candidates")
                )
            elif command == "close":
                protocol.write('{"closed":true}\n')
                break
            else:
                raise ValueError("unknown G7.11 geometry-controller command")
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
    backend = GeometryControllerBackend(args)
    session = GeometryControllerSession(backend, max_steps=args.max_steps)
    protocol.write(json.dumps(backend.ready(), allow_nan=False) + "\n")
    try:
        serve_requests(session, sys.stdin, protocol)
    finally:
        session.close()
        protocol.close()


if __name__ == "__main__":
    main()
