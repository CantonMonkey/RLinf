"""Pure geometry and gate helpers for the preregistered G7.11 controller."""

from __future__ import annotations

import math

DEPTH_HEIGHT = 64
DEPTH_WIDTH = 64
HFOV_DEGREES = 90.0
FX = 32.0
FY = 32.0
CX = 31.5
CY = 31.5
SENSOR_HEIGHT_M = 1.25
AGENT_HEIGHT_M = 1.50
AGENT_RADIUS_M = 0.10
SAFETY_MARGIN_M = 0.02
CORRIDOR_HALF_WIDTH_M = 0.12
OBSTACLE_MIN_HEIGHT_M = 0.05
OBSTACLE_MAX_HEIGHT_M = AGENT_HEIGHT_M - 0.05
FORWARD_STEP_M = 0.25
PRIMITIVE_BRAKE_DISTANCE_M = (
    FORWARD_STEP_M + AGENT_RADIUS_M + SAFETY_MARGIN_M
)


def depth_corridor_measurement(depth):
    if hasattr(depth, "tolist"):
        depth = depth.tolist()
    if (
        not isinstance(depth, list)
        or len(depth) != DEPTH_HEIGHT
        or any(not isinstance(row, list) or len(row) != DEPTH_WIDTH for row in depth)
    ):
        raise ValueError("G7.11 depth violates the fixed metric contract")
    values = []
    for row_index, row in enumerate(depth):
        for column_index, raw in enumerate(row):
            if isinstance(raw, bool):
                raise ValueError("G7.11 depth contains boolean data")
            z = float(raw)
            if not math.isfinite(z) or z < 0.0 or z > 10.0 + 1e-4:
                raise ValueError("G7.11 depth violates the fixed metric contract")
            x = (column_index - CX) * z / FX
            height = SENSOR_HEIGHT_M + (CY - row_index) * z / FY
            if (
                z > 1e-4
                and abs(x) <= CORRIDOR_HALF_WIDTH_M
                and OBSTACLE_MIN_HEIGHT_M <= height <= OBSTACLE_MAX_HEIGHT_M
            ):
                values.append(z)
    minimum = min(values) if values else 10.0
    return {
        "retained_point_count": len(values),
        "minimum_forward_m": minimum,
        "brake_distance_m": PRIMITIVE_BRAKE_DISTANCE_M,
        "allow_forward": bool(minimum > PRIMITIVE_BRAKE_DISTANCE_M),
    }


def controller_gate(metrics, *, integrity_ready):
    required = (
        "severe_exposure_reduction",
        "free_grazing_completion_rate",
        "free_grazing_progress_retention",
        "overall_forward_primitive_retention",
        "stratum_severe_nonincrease",
    )
    if any(name not in metrics for name in required):
        raise ValueError("G7.11 controller metrics are incomplete")
    gates = {
        "severe_exposure_reduction": float(
            metrics["severe_exposure_reduction"]
        )
        >= 0.70,
        "free_grazing_completion": float(
            metrics["free_grazing_completion_rate"]
        )
        >= 0.80,
        "free_grazing_progress_retention": float(
            metrics["free_grazing_progress_retention"]
        )
        >= 0.90,
        "overall_forward_primitive_retention": float(
            metrics["overall_forward_primitive_retention"]
        )
        >= 0.50,
        "stratum_severe_nonincrease": bool(
            metrics["stratum_severe_nonincrease"]
        ),
        "integrity": bool(integrity_ready),
    }
    return {"gates": gates, "ready": all(gates.values())}


def controlled_motion_severity(repeat):
    attempted = float(repeat["attempted_distance_m"])
    if attempted == 0.0:
        return None
    realized = float(repeat["realized_forward_m"])
    endpoint = float(repeat["attempted_endpoint_error_m"])
    lateral = float(repeat["realized_lateral_error_m"])
    progresses = [
        float(item["forward_progress_m"]) for item in repeat["primitive_trace"]
    ]
    if (
        attempted < 0.0
        or not progresses
        or any(
            not math.isfinite(value)
            for value in (attempted, realized, endpoint, lateral, *progresses)
        )
    ):
        raise ValueError("G7.11 controlled-motion outcome is invalid")
    ratio = realized / attempted
    maximum_deficit = max(max(0.0, FORWARD_STEP_M - value) for value in progresses)
    minimum_progress = min(progresses)
    severe = (
        ratio < 0.50
        or endpoint >= 0.25
        or lateral >= 0.25
        or minimum_progress <= 0.05
    )
    continuous = min(
        1.0,
        max(
            max(0.0, 1.0 - ratio) / 0.50,
            endpoint / 0.25,
            lateral / 0.25,
            maximum_deficit / 0.20,
        ),
    )
    return {
        "severe_exposure": bool(severe),
        "continuous_severity": float(continuous),
        "realized_ratio": float(ratio),
        "minimum_primitive_progress_m": float(minimum_progress),
        "maximum_primitive_progress_deficit_m": float(maximum_deficit),
    }


__all__ = [
    "AGENT_HEIGHT_M",
    "AGENT_RADIUS_M",
    "CORRIDOR_HALF_WIDTH_M",
    "CX",
    "CY",
    "DEPTH_HEIGHT",
    "DEPTH_WIDTH",
    "FORWARD_STEP_M",
    "FX",
    "FY",
    "HFOV_DEGREES",
    "OBSTACLE_MAX_HEIGHT_M",
    "OBSTACLE_MIN_HEIGHT_M",
    "PRIMITIVE_BRAKE_DISTANCE_M",
    "SAFETY_MARGIN_M",
    "SENSOR_HEIGHT_M",
    "controlled_motion_severity",
    "controller_gate",
    "depth_corridor_measurement",
]
