"""Target-aligned privileged prefix oracle for the preregistered G7.12 audit."""

from __future__ import annotations

import math

from streamvln_geometry_g711 import FORWARD_STEP_M, controlled_motion_severity


SEVERE_REALIZED_RATIO = 0.50
SEVERE_ENDPOINT_ERROR_M = 0.25
SEVERE_LATERAL_ERROR_M = 0.25
SEVERE_MIN_PRIMITIVE_PROGRESS_M = 0.05


def _vector(values, name):
    if (
        not isinstance(values, list)
        or len(values) != 3
        or any(isinstance(value, bool) or not math.isfinite(float(value)) for value in values)
    ):
        raise ValueError(f"G7.12 {name} must be a finite three-vector")
    return [float(value) for value in values]


def _distance(left, right):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def prefix_outcome(repeat, prefix_count):
    """Return the G7.10 severity outcome for one recorded forward prefix."""
    if not isinstance(repeat, dict):
        raise ValueError("G7.12 repeat must be an object")
    commanded = repeat.get("commanded_forward_primitives")
    trace = repeat.get("primitive_trace")
    if (
        type(commanded) is not int
        or commanded < 1
        or not isinstance(trace, list)
        or len(trace) != commanded
        or type(prefix_count) is not int
        or not 0 <= prefix_count <= commanded
    ):
        raise ValueError("G7.12 recorded prefix contract changed")
    start = _vector(repeat.get("post_turn_pose", {}).get("position"), "start")
    direction = _vector(repeat.get("forward_direction"), "forward direction")
    direction_norm = math.sqrt(sum(value * value for value in direction))
    if not math.isclose(direction_norm, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("G7.12 forward direction is not normalized")
    full_target = _vector(repeat.get("full_intended_endpoint"), "full target")
    if prefix_count:
        selected_trace = trace[:prefix_count]
        final = _vector(selected_trace[-1].get("after_position"), "prefix final")
    else:
        selected_trace = []
        final = start
    delta = [value - origin for value, origin in zip(final, start)]
    realized = sum(value * axis for value, axis in zip(delta, direction))
    lateral_vector = [
        value - realized * axis for value, axis in zip(delta, direction)
    ]
    lateral = math.sqrt(sum(value * value for value in lateral_vector))
    attempted = prefix_count * FORWARD_STEP_M
    attempted_target = [
        origin + attempted * axis for origin, axis in zip(start, direction)
    ]
    attempted_endpoint_error = _distance(final, attempted_target)
    actual_clearance = repeat.get("actual_clearance_trace")
    if not isinstance(actual_clearance, list) or len(actual_clearance) < prefix_count + 1:
        raise ValueError("G7.12 prefix clearance evidence is missing")
    clearance_values = [
        float(item["clearance_m"]) for item in actual_clearance[: prefix_count + 1]
    ]
    if any(not math.isfinite(value) or value < 0.0 for value in clearance_values):
        raise ValueError("G7.12 prefix clearance is invalid")
    severity_input = {
        "attempted_distance_m": attempted,
        "realized_forward_m": realized,
        "attempted_endpoint_error_m": attempted_endpoint_error,
        "realized_lateral_error_m": lateral,
        "primitive_trace": selected_trace,
    }
    controlled = controlled_motion_severity(severity_input)
    severe = False if controlled is None else bool(controlled["severe_exposure"])
    collisions = [bool(item["collision"]) for item in selected_trace]
    return {
        "selected_prefix": prefix_count,
        "commanded_forward_primitives": commanded,
        "completed": prefix_count == commanded,
        "executed_forward_primitives": prefix_count,
        "attempted_distance_m": attempted,
        "realized_forward_m": realized,
        "attempted_endpoint_error_m": attempted_endpoint_error,
        "full_endpoint_error_m": _distance(final, full_target),
        "lateral_error_m": lateral,
        "minimum_actual_clearance_m": min(clearance_values),
        "collision_steps": sum(collisions),
        "raw_collision": any(collisions),
        "severe_exposure": severe,
        "continuous_severity": (
            None if controlled is None else float(controlled["continuous_severity"])
        ),
        "controlled_motion": controlled,
    }


def select_longest_admissible_prefix(repeat):
    """Select the longest prefix whose every executed prefix is non-severe."""
    commanded = repeat.get("commanded_forward_primitives")
    if type(commanded) is not int or commanded < 1:
        raise ValueError("G7.12 command count changed")
    evaluated = []
    selected = prefix_outcome(repeat, 0)
    for prefix_count in range(1, commanded + 1):
        outcome = prefix_outcome(repeat, prefix_count)
        evaluated.append(
            {
                "prefix_count": prefix_count,
                "severe_exposure": outcome["severe_exposure"],
                "realized_forward_m": outcome["realized_forward_m"],
                "attempted_endpoint_error_m": outcome[
                    "attempted_endpoint_error_m"
                ],
                "lateral_error_m": outcome["lateral_error_m"],
                "minimum_primitive_progress_m": (
                    None
                    if outcome["controlled_motion"] is None
                    else outcome["controlled_motion"][
                        "minimum_primitive_progress_m"
                    ]
                ),
            }
        )
        if outcome["severe_exposure"]:
            break
        selected = outcome
    return {"selected": selected, "evaluated_prefixes": evaluated}


def joint_gate(metrics, *, integrity_ready):
    required = {
        "severe_exposure_reduction": 0.70,
        "free_grazing_completion_rate": 0.80,
        "free_grazing_progress_retention": 0.90,
        "overall_forward_primitive_retention": 0.50,
    }
    gates = {
        name: float(metrics[name]) >= threshold
        for name, threshold in required.items()
    }
    gates["stratum_severe_nonincrease"] = bool(
        metrics["stratum_severe_nonincrease"]
    )
    gates["integrity"] = bool(integrity_ready)
    return {"gates": gates, "ready": all(gates.values())}


__all__ = [
    "SEVERE_ENDPOINT_ERROR_M",
    "SEVERE_LATERAL_ERROR_M",
    "SEVERE_MIN_PRIMITIVE_PROGRESS_M",
    "SEVERE_REALIZED_RATIO",
    "joint_gate",
    "prefix_outcome",
    "select_longest_admissible_prefix",
]
