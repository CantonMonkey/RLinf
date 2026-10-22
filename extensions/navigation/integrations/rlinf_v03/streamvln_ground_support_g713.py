"""Causal metric-depth ground-support observables for G7.13."""

from __future__ import annotations

import math

from streamvln_geometry_g711 import (
    AGENT_RADIUS_M,
    CORRIDOR_HALF_WIDTH_M,
    CX,
    CY,
    DEPTH_HEIGHT,
    DEPTH_WIDTH,
    FX,
    FY,
    SENSOR_HEIGHT_M,
)


GROUND_FORWARD_MAX_M = 2.0
GROUND_HEIGHT_TOLERANCE_M = AGENT_RADIUS_M
DEPTH_MISSING_EPSILON_M = 1e-4
CENTER_HALF_WIDTH_M = CORRIDOR_HALF_WIDTH_M / 2.0


def _metric_depth(depth):
    if hasattr(depth, "tolist"):
        depth = depth.tolist()
    if (
        not isinstance(depth, list)
        or len(depth) != DEPTH_HEIGHT
        or any(not isinstance(row, list) or len(row) != DEPTH_WIDTH for row in depth)
    ):
        raise ValueError("G7.13 depth violates the fixed 64x64 metric contract")
    normalized = []
    for row in depth:
        values = []
        for raw in row:
            if isinstance(raw, bool):
                raise ValueError("G7.13 depth contains boolean data")
            value = float(raw)
            if not math.isfinite(value) or value < 0.0 or value > 10.0 + 1e-4:
                raise ValueError("G7.13 depth violates the fixed metric range")
            values.append(value)
        normalized.append(values)
    return normalized


def ground_support_observables(depth):
    """Measure fixed expected-ground support without privileged geometry."""
    depth = _metric_depth(depth)
    counts = {
        "ray_count": 0,
        "support_count": 0,
        "dropoff_count": 0,
        "occlusion_count": 0,
        "center_ray_count": 0,
        "center_dropoff_count": 0,
        "left_ray_count": 0,
        "left_dropoff_count": 0,
        "right_ray_count": 0,
        "right_dropoff_count": 0,
    }
    expected_forwards = []
    for row_index, row in enumerate(depth):
        vertical = row_index - CY
        if vertical <= 0.0:
            continue
        expected_forward = SENSOR_HEIGHT_M * FY / vertical
        if expected_forward > GROUND_FORWARD_MAX_M:
            continue
        for column_index, observed in enumerate(row):
            expected_lateral = (column_index - CX) * expected_forward / FX
            if abs(expected_lateral) > CORRIDOR_HALF_WIDTH_M:
                continue
            counts["ray_count"] += 1
            expected_forwards.append(expected_forward)
            if abs(expected_lateral) <= CENTER_HALF_WIDTH_M:
                counts["center_ray_count"] += 1
                region = "center"
            elif expected_lateral < 0.0:
                counts["left_ray_count"] += 1
                region = "left"
            else:
                counts["right_ray_count"] += 1
                region = "right"
            if (
                observed <= DEPTH_MISSING_EPSILON_M
                or observed > expected_forward + GROUND_HEIGHT_TOLERANCE_M
            ):
                counts["dropoff_count"] += 1
                counts[f"{region}_dropoff_count"] += 1
            elif observed < expected_forward - GROUND_HEIGHT_TOLERANCE_M:
                counts["occlusion_count"] += 1
            else:
                counts["support_count"] += 1
    if (
        counts["ray_count"] == 0
        or counts["center_ray_count"] == 0
        or counts["left_ray_count"] == 0
        or counts["right_ray_count"] == 0
        or sum(counts[name] for name in ("support_count", "dropoff_count", "occlusion_count"))
        != counts["ray_count"]
    ):
        raise RuntimeError("G7.13 expected-ground ray contract is empty or inconsistent")

    def fraction(numerator, denominator):
        return float(numerator) / float(denominator)

    left = fraction(counts["left_dropoff_count"], counts["left_ray_count"])
    right = fraction(counts["right_dropoff_count"], counts["right_ray_count"])
    result = {
        **counts,
        "minimum_expected_ground_forward_m": float(min(expected_forwards)),
        "maximum_expected_ground_forward_m": float(max(expected_forwards)),
        "footprint_support_fraction": fraction(counts["support_count"], counts["ray_count"]),
        "footprint_dropoff_fraction": fraction(counts["dropoff_count"], counts["ray_count"]),
        "footprint_occlusion_fraction": fraction(counts["occlusion_count"], counts["ray_count"]),
        "center_dropoff_fraction": fraction(
            counts["center_dropoff_count"], counts["center_ray_count"]
        ),
        "left_dropoff_fraction": left,
        "right_dropoff_fraction": right,
        "maximum_side_dropoff_fraction": max(left, right),
    }
    if not math.isclose(
        result["footprint_support_fraction"]
        + result["footprint_dropoff_fraction"]
        + result["footprint_occlusion_fraction"],
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("G7.13 ground-support fractions do not partition the rays")
    return result


def binary_auroc(scores, labels):
    if (
        not isinstance(scores, list)
        or not isinstance(labels, list)
        or len(scores) != len(labels)
        or any(type(label) is not bool for label in labels)
    ):
        raise ValueError("G7.13 AUROC inputs are invalid")
    positives = [float(score) for score, label in zip(scores, labels) if label]
    negatives = [float(score) for score, label in zip(scores, labels) if not label]
    if not positives or not negatives or any(
        not math.isfinite(value) for value in positives + negatives
    ):
        raise ValueError("G7.13 AUROC requires finite scores and both classes")
    wins = ties = 0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1
            elif positive == negative:
                ties += 1
    return (wins + 0.5 * ties) / (len(positives) * len(negatives))


def observability_gate(metrics, *, integrity_ready):
    gates = {
        "overall_auroc": float(metrics["overall_auroc"]) >= 0.75,
        "minimum_scene_auroc": float(metrics["minimum_scene_auroc"]) >= 0.65,
        "minimum_state_auroc": float(metrics["minimum_state_auroc"]) >= 0.65,
        "paired_strict_win_rate": float(metrics["paired_strict_win_rate"]) >= 0.75,
        "positive_median_risk_gt_control": float(metrics["positive_median_risk"])
        > float(metrics["control_median_risk"]),
        "integrity": bool(integrity_ready),
    }
    return {"gates": gates, "ready": all(gates.values())}


__all__ = [
    "CENTER_HALF_WIDTH_M",
    "DEPTH_MISSING_EPSILON_M",
    "GROUND_FORWARD_MAX_M",
    "GROUND_HEIGHT_TOLERANCE_M",
    "binary_auroc",
    "ground_support_observables",
    "observability_gate",
]
