"""Fixed immediate-footprint ray geometry for the G7.14 visibility ceiling."""

from __future__ import annotations

import math

from streamvln_geometry_g711 import (
    CORRIDOR_HALF_WIDTH_M,
    CX,
    CY,
    DEPTH_HEIGHT,
    DEPTH_WIDTH,
    FORWARD_STEP_M,
    FX,
    FY,
    SENSOR_HEIGHT_M,
)


DEPTH_MAX_M = 10.0
DEPTH_MISSING_EPSILON_M = 1e-4
GROUND_HEIGHT_TOLERANCE_M = 0.1
TARGET_FORWARD_MIN_M = 0.0
TARGET_FORWARD_MAX_M = 0.35
TARGET_LATERAL_MIN_M = -CORRIDOR_HALF_WIDTH_M
TARGET_LATERAL_MAX_M = CORRIDOR_HALF_WIDTH_M
FORWARD_BINS_M = ((0.0, 0.125), (0.125, 0.25), (0.25, 0.35))
LATERAL_BINS_M = ((-0.12, -0.04), (-0.04, 0.04), (0.04, 0.12))
DOWNWARD_PITCH_RADIANS = -math.atan2(SENSOR_HEIGHT_M, FORWARD_STEP_M)
DOWNWARD_PITCH_DEGREES = math.degrees(DOWNWARD_PITCH_RADIANS)
PITCH_MAGNITUDE_RADIANS = -DOWNWARD_PITCH_RADIANS
EXPECTED_OPTICAL_AXIS_AGENT_FRAME = (
    0.0,
    -math.sin(PITCH_MAGNITUDE_RADIANS),
    -math.cos(PITCH_MAGNITUDE_RADIANS),
)
EXPECTED_SENSOR_POSITION_AGENT_FRAME = (0.0, SENSOR_HEIGHT_M, 0.0)


def _bin_index(value, bins):
    for index, (lower, upper) in enumerate(bins):
        if lower <= value < upper or (index == len(bins) - 1 and value == upper):
            return index
    return None


def target_ray_geometry():
    """Return the preregistered discrete ground rays in the next swept footprint."""
    alpha = PITCH_MAGNITUDE_RADIANS
    sin_alpha = math.sin(alpha)
    cos_alpha = math.cos(alpha)
    rays = []
    for row in range(DEPTH_HEIGHT):
        image_down = (row - CY) / FY
        down_coefficient = sin_alpha + image_down * cos_alpha
        if down_coefficient <= 0.0:
            continue
        expected_depth = SENSOR_HEIGHT_M / down_coefficient
        expected_forward = expected_depth * (
            cos_alpha - image_down * sin_alpha
        )
        if not TARGET_FORWARD_MIN_M <= expected_forward <= TARGET_FORWARD_MAX_M:
            continue
        for column in range(DEPTH_WIDTH):
            expected_lateral = (column - CX) * expected_depth / FX
            if not TARGET_LATERAL_MIN_M <= expected_lateral <= TARGET_LATERAL_MAX_M:
                continue
            forward_bin = _bin_index(expected_forward, FORWARD_BINS_M)
            lateral_bin = _bin_index(expected_lateral, LATERAL_BINS_M)
            if forward_bin is None or lateral_bin is None:
                raise RuntimeError("G7.14 target ray falls outside frozen bins")
            rays.append(
                {
                    "row": row,
                    "column": column,
                    "expected_optical_depth_m": expected_depth,
                    "expected_forward_m": expected_forward,
                    "expected_lateral_m": expected_lateral,
                    "forward_bin": forward_bin,
                    "lateral_bin": lateral_bin,
                }
            )
    return rays


TARGET_RAYS = target_ray_geometry()
EXPECTED_TARGET_RAY_COUNT = 48
EXPECTED_FORWARD_BIN_COUNTS = (18, 18, 12)
EXPECTED_LATERAL_BIN_COUNTS = (16, 16, 16)
EXPECTED_GRID_COUNTS = ((6, 6, 6), (6, 6, 6), (4, 4, 4))
FROZEN_GEOMETRY_EXTENTS = {
    "minimum_expected_forward_m": 0.033987915407854924,
    "maximum_expected_forward_m": 0.3115141955835961,
    "minimum_expected_lateral_m": -0.10053271911657698,
    "maximum_expected_lateral_m": 0.10053271911657698,
    "minimum_expected_optical_depth_m": 1.2323914232248423,
    "maximum_expected_optical_depth_m": 1.2868188046921853,
}


def geometry_summary():
    forward_counts = [0, 0, 0]
    lateral_counts = [0, 0, 0]
    grid = [[0, 0, 0] for _ in range(3)]
    for ray in TARGET_RAYS:
        forward_counts[ray["forward_bin"]] += 1
        lateral_counts[ray["lateral_bin"]] += 1
        grid[ray["forward_bin"]][ray["lateral_bin"]] += 1
    calculated_extents = {
        "minimum_expected_forward_m": min(
            ray["expected_forward_m"] for ray in TARGET_RAYS
        ),
        "maximum_expected_forward_m": max(
            ray["expected_forward_m"] for ray in TARGET_RAYS
        ),
        "minimum_expected_lateral_m": min(
            ray["expected_lateral_m"] for ray in TARGET_RAYS
        ),
        "maximum_expected_lateral_m": max(
            ray["expected_lateral_m"] for ray in TARGET_RAYS
        ),
        "minimum_expected_optical_depth_m": min(
            ray["expected_optical_depth_m"] for ray in TARGET_RAYS
        ),
        "maximum_expected_optical_depth_m": max(
            ray["expected_optical_depth_m"] for ray in TARGET_RAYS
        ),
    }
    if any(
        abs(calculated_extents[key] - value) > 1e-12
        for key, value in FROZEN_GEOMETRY_EXTENTS.items()
    ):
        raise RuntimeError("G7.14 analytic ray extents changed")
    result = {
        "target_ray_count": len(TARGET_RAYS),
        "forward_bin_counts": forward_counts,
        "lateral_bin_counts": lateral_counts,
        "grid_counts": grid,
        **FROZEN_GEOMETRY_EXTENTS,
    }
    if (
        result["target_ray_count"] != EXPECTED_TARGET_RAY_COUNT
        or tuple(forward_counts) != EXPECTED_FORWARD_BIN_COUNTS
        or tuple(lateral_counts) != EXPECTED_LATERAL_BIN_COUNTS
        or tuple(tuple(row) for row in grid) != EXPECTED_GRID_COUNTS
    ):
        raise RuntimeError("G7.14 frozen analytic ray geometry changed")
    return result


FROZEN_GEOMETRY_SUMMARY = geometry_summary()


def _metric_depth(depth):
    if hasattr(depth, "tolist"):
        depth = depth.tolist()
    if (
        not isinstance(depth, list)
        or len(depth) != DEPTH_HEIGHT
        or any(not isinstance(row, list) or len(row) != DEPTH_WIDTH for row in depth)
    ):
        raise ValueError("G7.14 depth violates the fixed 64x64 metric contract")
    normalized = []
    for row in depth:
        values = []
        for raw in row:
            if isinstance(raw, bool):
                raise ValueError("G7.14 depth contains boolean data")
            value = float(raw)
            if not math.isfinite(value) or value < 0.0 or value > DEPTH_MAX_M + 1e-4:
                raise ValueError("G7.14 depth violates the fixed metric range")
            values.append(value)
        normalized.append(values)
    return normalized


def immediate_footprint_observables(depth):
    """Classify every preregistered target ray without outcome labels."""
    depth = _metric_depth(depth)
    support = occlusion = missing = unclipped = 0
    forward_counts = [0, 0, 0]
    lateral_counts = [0, 0, 0]
    grid = [[0, 0, 0] for _ in range(3)]
    observed = []
    for ray in TARGET_RAYS:
        value = depth[ray["row"]][ray["column"]]
        observed.append(value)
        forward_counts[ray["forward_bin"]] += 1
        lateral_counts[ray["lateral_bin"]] += 1
        grid[ray["forward_bin"]][ray["lateral_bin"]] += 1
        if DEPTH_MISSING_EPSILON_M < value < DEPTH_MAX_M - DEPTH_MISSING_EPSILON_M:
            unclipped += 1
        expected = ray["expected_optical_depth_m"]
        if (
            value <= DEPTH_MISSING_EPSILON_M
            or value > expected + GROUND_HEIGHT_TOLERANCE_M
        ):
            missing += 1
        elif value < expected - GROUND_HEIGHT_TOLERANCE_M:
            occlusion += 1
        else:
            support += 1
    count = len(TARGET_RAYS)
    if support + occlusion + missing != count:
        raise RuntimeError("G7.14 target depth classes do not partition rays")
    result = {
        **FROZEN_GEOMETRY_SUMMARY,
        "support_count": support,
        "occlusion_count": occlusion,
        "missing_count": missing,
        "classified_count": support + occlusion + missing,
        "unclipped_count": unclipped,
        "support_fraction": support / count,
        "occlusion_fraction": occlusion / count,
        "missing_fraction": missing / count,
        "unclipped_fraction": unclipped / count,
        "minimum_observed_target_depth_m": min(observed),
        "maximum_observed_target_depth_m": max(observed),
        "observed_target_depth_range_m": max(observed) - min(observed),
    }
    if (
        forward_counts != result["forward_bin_counts"]
        or lateral_counts != result["lateral_bin_counts"]
        or grid != result["grid_counts"]
        or not math.isclose(
            result["support_fraction"]
            + result["occlusion_fraction"]
            + result["missing_fraction"],
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise RuntimeError("G7.14 visibility observables violate frozen geometry")
    return result


def visibility_gate(metrics, *, integrity_ready):
    gates = {
        "target_ray_count_exact": metrics["target_ray_count"]
        == EXPECTED_TARGET_RAY_COUNT,
        "minimum_spatial_bin_count": metrics["minimum_spatial_bin_count"] >= 4,
        "all_target_depths_classified": metrics["all_target_depths_classified"],
        "all_repeats_exact": metrics["all_repeats_exact"],
        "maximum_optical_axis_error": metrics["maximum_optical_axis_error"] <= 1e-6,
        "maximum_sensor_position_error_m": metrics[
            "maximum_sensor_position_error_m"
        ]
        <= 1e-6,
        "integrity": bool(integrity_ready),
    }
    return {"gates": gates, "ready": all(gates.values())}


__all__ = [
    "DOWNWARD_PITCH_DEGREES",
    "DOWNWARD_PITCH_RADIANS",
    "EXPECTED_GRID_COUNTS",
    "EXPECTED_LATERAL_BIN_COUNTS",
    "EXPECTED_OPTICAL_AXIS_AGENT_FRAME",
    "EXPECTED_SENSOR_POSITION_AGENT_FRAME",
    "EXPECTED_TARGET_RAY_COUNT",
    "FROZEN_GEOMETRY_EXTENTS",
    "EXPECTED_FORWARD_BIN_COUNTS",
    "FROZEN_GEOMETRY_SUMMARY",
    "immediate_footprint_observables",
    "target_ray_geometry",
    "visibility_gate",
]
