"""Train-only branch-rollout labels for G7.6 local passability."""

from __future__ import annotations

import json
import math
import os
import sys
import traceback

if __package__:
    from .habitat_r2r_service import (
        ACTIONS,
        HabitatR2RBackend,
        R2RSession,
        parse_args,
    )
else:
    from habitat_r2r_service import (
        ACTIONS,
        HabitatR2RBackend,
        R2RSession,
        parse_args,
    )


PROBE_SOURCE = "train-simulator-branch-rollout"
MAX_CANDIDATES = 64
MAX_ACTIONS = 8


def validate_probe_candidates(candidates):
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= MAX_CANDIDATES:
        raise ValueError("probe candidates must contain 1..64 rows")
    names = set()
    result = []
    for row in candidates:
        if not isinstance(row, dict) or set(row) != {"name", "actions"}:
            raise ValueError("probe candidate schema mismatch")
        name = row["name"]
        actions = row["actions"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("probe candidate names must be unique nonempty strings")
        if (
            not isinstance(actions, list)
            or not 1 <= len(actions) <= MAX_ACTIONS
            or any(type(action) is not int or action not in (1, 2, 3) for action in actions)
            or 1 not in actions
        ):
            raise ValueError("probe actions must be 1..8 non-STOP navigation primitives")
        names.add(name)
        result.append({"name": name, "actions": list(actions)})
    return result


class PassabilityBackend(HabitatR2RBackend):
    def _copy_agent_state(self):
        from habitat_sim.utils.common import quat_from_coeffs, quat_to_coeffs

        state = self.env.sim.get_agent_state()
        position = self.np.asarray(state.position, dtype=self.np.float64).copy()
        coefficients = self.np.asarray(
            quat_to_coeffs(state.rotation), dtype=self.np.float64
        ).copy()
        rotation = quat_from_coeffs(coefficients)
        return position, rotation, coefficients

    def _restore_agent_state(self, position, rotation, coefficients):
        from habitat_sim.utils.common import quat_to_coeffs

        self.env.sim.set_agent_state(position, rotation, reset_sensors=True)
        restored = self.env.sim.get_agent_state()
        position_error = float(
            self.np.linalg.norm(
                self.np.asarray(restored.position, dtype=self.np.float64) - position
            )
        )
        rotation_error = float(
            self.np.max(
                self.np.abs(
                    self.np.asarray(
                        quat_to_coeffs(restored.rotation), dtype=self.np.float64
                    )
                    - coefficients
                )
            )
        )
        if position_error > 1e-6 or rotation_error > 1e-6:
            raise RuntimeError("candidate branch rollout did not restore simulator state")
        return position_error, rotation_error

    def probe_candidates(self, candidates):
        if self.split != "train":
            raise RuntimeError("candidate probes are restricted to the train split")
        rows = validate_probe_candidates(candidates)
        start_position, start_rotation, start_coefficients = self._copy_agent_state()
        results = []
        maximum_position_error = maximum_rotation_error = 0.0
        for row in rows:
            collisions = []
            positions = []
            try:
                for action in row["actions"]:
                    self.env.sim.step(action)
                    collisions.append(bool(self.env.sim.previous_step_collided))
                    position = self.np.asarray(
                        self.env.sim.get_agent_state().position,
                        dtype=self.np.float64,
                    )
                    positions.append([float(value) for value in position])
            finally:
                position_error, rotation_error = self._restore_agent_state(
                    start_position, start_rotation, start_coefficients
                )
                maximum_position_error = max(maximum_position_error, position_error)
                maximum_rotation_error = max(maximum_rotation_error, rotation_error)
            forward_collisions = [
                bool(collision)
                for action, collision in zip(row["actions"], collisions)
                if action == 1
            ]
            results.append(
                {
                    "name": row["name"],
                    "actions": row["actions"],
                    "collisions": collisions,
                    "unsafe": any(forward_collisions),
                    "forward_attempts": len(forward_collisions),
                    "forward_collisions": sum(forward_collisions),
                    "end_position": positions[-1],
                }
            )
        if any(
            not math.isfinite(value)
            for value in (maximum_position_error, maximum_rotation_error)
        ):
            raise RuntimeError("candidate branch restoration error is nonfinite")
        return {
            "schema_version": 1,
            "source": PROBE_SOURCE,
            "policy_observation_unchanged": True,
            "state_restored": True,
            "max_position_restore_error_m": maximum_position_error,
            "max_rotation_restore_error": maximum_rotation_error,
            "candidates": results,
        }


class PassabilitySession(R2RSession):
    def probe_candidates(self, candidates):
        if self.last_frame is None:
            raise RuntimeError("reset must precede candidate probes")
        if self.final_info is not None:
            raise RuntimeError("candidate probes are unavailable after episode end")
        return self.backend.probe_candidates(candidates)


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
            elif command == "probe_candidates":
                response = session.probe_candidates(request.get("candidates"))
            elif command == "close":
                protocol.write('{"closed":true}\n')
                break
            else:
                raise ValueError("unknown R2R passability command")
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
    backend = PassabilityBackend(args)
    session = PassabilitySession(backend, max_steps=args.max_steps)
    protocol.write(json.dumps(backend.ready(), allow_nan=False) + "\n")
    try:
        serve_requests(session, sys.stdin, protocol)
    finally:
        session.close()
        protocol.close()


if __name__ == "__main__":
    main()
