"""Train-only aligned metric-depth/local-controller diagnostics for G7.8."""

from __future__ import annotations

import base64
import json
import math
import os
import sys
import traceback

if __package__:
    from .habitat_r2r_aligned_passability_service import (
        AlignedPassabilityBackend,
        AlignedPassabilitySession,
    )
    from .habitat_r2r_passability_service import validate_probe_candidates
    from .habitat_r2r_service import parse_args
else:
    from habitat_r2r_aligned_passability_service import (
        AlignedPassabilityBackend,
        AlignedPassabilitySession,
    )
    from habitat_r2r_passability_service import validate_probe_candidates
    from habitat_r2r_service import parse_args


DEPTH_SOURCE = "train-simulator-aligned-depth-branch-rollout"
RUNTIME_DEPTH_SOURCE = "runtime-aligned-metric-depth"
DEPTH_ENCODING = "base64/raw-f32le-hw"
DEPTH_MAX_METERS = 10.0


class DepthPassabilityBackend(AlignedPassabilityBackend):
    def __init__(self, args):
        args.include_depth = True
        args.depth_max_m = DEPTH_MAX_METERS
        super().__init__(args)
        self.depth_size = self.rgb_size
        self.depth_max_m = DEPTH_MAX_METERS

    def ready(self):
        ready = super().ready()
        ready.update(
            {
                "depth_size": self.depth_size,
                "depth_encoding": DEPTH_ENCODING,
                "depth_min_m": 0.0,
                "depth_max_m": self.depth_max_m,
                "depth_policy_role": "explicit-local-controller-diagnostic",
            }
        )
        return ready

    def _metric_depth(self):
        # Habitat-Sim's raw sensor call can contain +inf for rays with no hit.
        # Route the current pose through Habitat-Lab's sensor suite so its
        # declared [min_depth, max_depth] clipping is applied without a step.
        observations = self.env.sim.get_observations_at()
        if observations is None or "depth" not in observations:
            raise RuntimeError("aligned observation has no metric depth")
        depth = self.np.asarray(observations["depth"])
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        depth = self.np.ascontiguousarray(depth, dtype="<f4")
        if depth.shape != (self.depth_size, self.depth_size):
            raise ValueError("aligned metric depth must be float32 HW")
        if (
            not bool(self.np.isfinite(depth).all())
            or float(depth.min()) < 0.0
            or float(depth.max()) > self.depth_max_m + 1e-4
        ):
            raise ValueError("aligned metric depth is outside configured meters")
        return {
            "encoding": DEPTH_ENCODING,
            "shape_hw": list(depth.shape),
            "min_m": 0.0,
            "max_m": self.depth_max_m,
            "data_b64": base64.b64encode(depth.tobytes()).decode("ascii"),
        }

    def observe_aligned_depth(self):
        return {
            "schema_version": 1,
            "source": RUNTIME_DEPTH_SOURCE,
            "state_unchanged": True,
            "metric_depth": self._metric_depth(),
        }

    def probe_depth_candidates(self, candidates):
        if self.split != "train":
            raise RuntimeError("depth candidate probes are train-only")
        rows = validate_probe_candidates(candidates)
        start_position, start_rotation, start_coefficients = self._copy_agent_state()
        results = []
        maximum_position_error = maximum_rotation_error = 0.0
        for row in rows:
            actions = row["actions"]
            first_forward = actions.index(1)
            turn_prefix = actions[:first_forward]
            forward_suffix = actions[first_forward:]
            if any(action == 1 for action in turn_prefix) or any(
                action != 1 for action in forward_suffix
            ):
                raise ValueError("depth macro must be turn prefix then forward suffix")
            metric_depth = None
            collisions = []
            end_position = None
            try:
                for action in turn_prefix:
                    self.env.sim.step(action)
                metric_depth = self._metric_depth()
                for action in forward_suffix:
                    self.env.sim.step(action)
                    collisions.append(bool(self.env.sim.previous_step_collided))
                end_position = [
                    float(value)
                    for value in self.np.asarray(
                        self.env.sim.get_agent_state().position,
                        dtype=self.np.float64,
                    )
                ]
            finally:
                position_error, rotation_error = self._restore_agent_state(
                    start_position, start_rotation, start_coefficients
                )
                maximum_position_error = max(maximum_position_error, position_error)
                maximum_rotation_error = max(maximum_rotation_error, rotation_error)
            results.append(
                {
                    "name": row["name"],
                    "actions": actions,
                    "turn_prefix": turn_prefix,
                    "forward_suffix": forward_suffix,
                    "metric_depth": metric_depth,
                    "forward_collisions": collisions,
                    "unsafe": any(collisions),
                    "end_position": end_position,
                }
            )
        if any(
            not math.isfinite(value)
            for value in (maximum_position_error, maximum_rotation_error)
        ):
            raise RuntimeError("depth branch restoration error is nonfinite")
        return {
            "schema_version": 1,
            "source": DEPTH_SOURCE,
            "policy_observation_unchanged": True,
            "state_restored": True,
            "max_position_restore_error_m": maximum_position_error,
            "max_rotation_restore_error": maximum_rotation_error,
            "candidates": results,
        }


class DepthPassabilitySession(AlignedPassabilitySession):
    def observe_aligned_depth(self):
        if self.last_frame is None:
            raise RuntimeError("reset must precede aligned depth observation")
        if self.final_info is not None:
            raise RuntimeError("aligned depth is unavailable after episode end")
        return self.backend.observe_aligned_depth()

    def probe_depth_candidates(self, candidates):
        if self.last_frame is None:
            raise RuntimeError("reset must precede depth candidate probes")
        if self.final_info is not None:
            raise RuntimeError("depth probes are unavailable after episode end")
        return self.backend.probe_depth_candidates(candidates)


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
            elif command == "probe_depth_candidates":
                response = session.probe_depth_candidates(request.get("candidates"))
            elif command == "close":
                protocol.write('{"closed":true}\n')
                break
            else:
                raise ValueError("unknown depth-passability command")
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
    backend = DepthPassabilityBackend(args)
    session = DepthPassabilitySession(backend, max_steps=args.max_steps)
    protocol.write(json.dumps(backend.ready(), allow_nan=False) + "\n")
    try:
        serve_requests(session, sys.stdin, protocol)
    finally:
        session.close()
        protocol.close()


if __name__ == "__main__":
    main()
