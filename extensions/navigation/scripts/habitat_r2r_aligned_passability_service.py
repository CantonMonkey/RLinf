"""Train-only candidate-aligned RGB branch labels for G7.7."""

from __future__ import annotations

import base64
import json
import math
import os
import sys
import traceback

if __package__:
    from .habitat_r2r_passability_service import (
        PassabilityBackend,
        PassabilitySession,
        validate_probe_candidates,
    )
    from .habitat_r2r_service import parse_args
else:
    from habitat_r2r_passability_service import (
        PassabilityBackend,
        PassabilitySession,
        validate_probe_candidates,
    )
    from habitat_r2r_service import parse_args


ALIGNED_SOURCE = "train-simulator-aligned-branch-rollout"
RGB_ENCODING = "base64/raw-u8-hwc"


class AlignedPassabilityBackend(PassabilityBackend):
    def _aligned_rgb(self):
        observations = self.env.sim.get_sensor_observations()
        if "rgb" not in observations:
            raise RuntimeError("aligned branch observation has no RGB")
        rgb = self.np.ascontiguousarray(observations["rgb"])
        if rgb.ndim == 3 and rgb.shape[-1] == 4:
            rgb = self.np.ascontiguousarray(rgb[..., :3])
        if rgb.dtype != self.np.uint8 or rgb.shape != (
            self.rgb_size,
            self.rgb_size,
            3,
        ):
            raise ValueError("aligned branch RGB must be uint8 HWC")
        return {
            "encoding": RGB_ENCODING,
            "shape_hwc": list(rgb.shape),
            "data_b64": base64.b64encode(rgb.tobytes()).decode("ascii"),
        }

    def probe_aligned_candidates(self, candidates):
        if self.split != "train":
            raise RuntimeError("aligned candidate probes are train-only")
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
                raise ValueError("aligned macro must be turn prefix then forward suffix")
            collisions = []
            aligned_rgb = None
            end_position = None
            try:
                for action in turn_prefix:
                    self.env.sim.step(action)
                aligned_rgb = self._aligned_rgb()
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
                    "aligned_rgb": aligned_rgb,
                    "forward_collisions": collisions,
                    "unsafe": any(collisions),
                    "end_position": end_position,
                }
            )
        if any(
            not math.isfinite(value)
            for value in (maximum_position_error, maximum_rotation_error)
        ):
            raise RuntimeError("aligned branch restoration error is nonfinite")
        return {
            "schema_version": 1,
            "source": ALIGNED_SOURCE,
            "policy_observation_unchanged": True,
            "state_restored": True,
            "max_position_restore_error_m": maximum_position_error,
            "max_rotation_restore_error": maximum_rotation_error,
            "candidates": results,
        }


class AlignedPassabilitySession(PassabilitySession):
    def probe_aligned_candidates(self, candidates):
        if self.last_frame is None:
            raise RuntimeError("reset must precede aligned candidate probes")
        if self.final_info is not None:
            raise RuntimeError("aligned probes are unavailable after episode end")
        return self.backend.probe_aligned_candidates(candidates)


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
            elif command == "probe_aligned_candidates":
                response = session.probe_aligned_candidates(
                    request.get("candidates")
                )
            elif command == "close":
                protocol.write('{"closed":true}\n')
                break
            else:
                raise ValueError("unknown aligned-passability command")
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
    backend = AlignedPassabilityBackend(args)
    session = AlignedPassabilitySession(backend, max_steps=args.max_steps)
    protocol.write(json.dumps(backend.ready(), allow_nan=False) + "\n")
    try:
        serve_requests(session, sys.stdin, protocol)
    finally:
        session.close()
        protocol.close()


if __name__ == "__main__":
    main()
