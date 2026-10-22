"""Frozen collision-severity and navmesh-clearance audit for G7.10."""

from __future__ import annotations

import json
import math
import os
import sys
import traceback

if __package__:
    from .habitat_r2r_macro_audit_service import (
        MacroAuditBackend,
        MacroAuditSession,
    )
    from .habitat_r2r_service import parse_args
else:
    from habitat_r2r_macro_audit_service import (
        MacroAuditBackend,
        MacroAuditSession,
    )
    from habitat_r2r_service import parse_args


SEVERITY_SOURCE = "train-simulator-frozen-severity-clearance-audit"
CLEARANCE_SPACING_METERS = 0.125
CLEARANCE_SEARCH_RADIUS_METERS = 2.0


class SeverityAuditBackend(MacroAuditBackend):
    def _clearance_sample(self, position, path_distance_m):
        point = self.np.asarray(position, dtype=self.np.float32)
        raw = float(
            self.env.sim.pathfinder.distance_to_closest_obstacle(
                point, CLEARANCE_SEARCH_RADIUS_METERS
            )
        )
        if math.isnan(raw) or raw < 0.0:
            raise RuntimeError("navmesh clearance is invalid")
        within_radius = math.isfinite(raw) and raw <= CLEARANCE_SEARCH_RADIUS_METERS
        clearance = (
            min(raw, CLEARANCE_SEARCH_RADIUS_METERS)
            if math.isfinite(raw)
            else CLEARANCE_SEARCH_RADIUS_METERS
        )
        return {
            "path_distance_m": float(path_distance_m),
            "position": [float(value) for value in point],
            "clearance_m": float(clearance),
            "nearest_obstacle_within_radius": bool(within_radius),
            "navigable": bool(self.env.sim.pathfinder.is_navigable(point)),
        }

    def _segment_clearance_trace(self, start, end):
        first = self.np.asarray(start, dtype=self.np.float64)
        second = self.np.asarray(end, dtype=self.np.float64)
        distance = float(self.np.linalg.norm(second - first))
        count = max(1, int(math.ceil(distance / CLEARANCE_SPACING_METERS)))
        return [
            self._clearance_sample(
                first + (second - first) * (index / count),
                distance * (index / count),
            )
            for index in range(count + 1)
        ]

    def _single_repeat(self, turn_prefix, forward_suffix):
        result = super()._single_repeat(turn_prefix, forward_suffix)
        start = result["post_turn_pose"]["position"]
        intended = result["intended_endpoint"]
        intended_trace = self._segment_clearance_trace(start, intended)
        actual_positions = [start] + [
            item["after_position"] for item in result["primitive_trace"]
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
        no_sliding_endpoint = self.np.asarray(
            self.env.sim.pathfinder.try_step_no_sliding(
                self.np.asarray(start, dtype=self.np.float32),
                self.np.asarray(intended, dtype=self.np.float32),
            ),
            dtype=self.np.float64,
        )
        intended_array = self.np.asarray(intended, dtype=self.np.float64)
        result.update(
            {
                "clearance_spacing_m": CLEARANCE_SPACING_METERS,
                "clearance_search_radius_m": CLEARANCE_SEARCH_RADIUS_METERS,
                "intended_clearance_trace": intended_trace,
                "actual_clearance_trace": actual_trace,
                "minimum_intended_clearance_m": min(
                    item["clearance_m"] for item in intended_trace
                ),
                "minimum_actual_clearance_m": min(
                    item["clearance_m"] for item in actual_trace
                ),
                "no_sliding_endpoint": [
                    float(value) for value in no_sliding_endpoint
                ],
                "no_sliding_endpoint_error_m": float(
                    self.np.linalg.norm(no_sliding_endpoint - intended_array)
                ),
            }
        )
        return result

    def probe_severity_audit(self, candidates):
        response = super().probe_macro_audit(candidates)
        response["source"] = SEVERITY_SOURCE
        response["clearance_spacing_m"] = CLEARANCE_SPACING_METERS
        response["clearance_search_radius_m"] = CLEARANCE_SEARCH_RADIUS_METERS
        return response


class SeverityAuditSession(MacroAuditSession):
    def probe_severity_audit(self, candidates):
        if self.last_frame is None:
            raise RuntimeError("reset must precede severity audit")
        if self.final_info is not None:
            raise RuntimeError("severity audit is unavailable after episode end")
        return self.backend.probe_severity_audit(candidates)


def serve_requests(session, request_lines, protocol):
    for line in request_lines:
        try:
            request = json.loads(line)
            command = request.get("cmd")
            if command == "reset":
                response = session.reset()
            elif command == "step":
                response = session.step(request.get("action"))
            elif command == "probe_severity_audit":
                response = session.probe_severity_audit(
                    request.get("candidates")
                )
            elif command == "close":
                protocol.write('{"closed":true}\n')
                break
            else:
                raise ValueError("unknown severity-audit command")
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
    backend = SeverityAuditBackend(args)
    session = SeverityAuditSession(backend, max_steps=args.max_steps)
    protocol.write(json.dumps(backend.ready(), allow_nan=False) + "\n")
    try:
        serve_requests(session, sys.stdin, protocol)
    finally:
        session.close()
        protocol.close()


if __name__ == "__main__":
    main()
