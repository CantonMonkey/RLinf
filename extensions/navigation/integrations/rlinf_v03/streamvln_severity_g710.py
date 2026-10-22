"""Pure severity-target helpers for the preregistered G7.10 audit."""

from __future__ import annotations

from collections import Counter
import math


FREE_ENDPOINT_TOLERANCE_M = 0.05
FREE_LATERAL_TOLERANCE_M = 0.05
FREE_REALIZED_RATIO = 0.90
SLIDING_PRIMITIVE_DEFICIT_M = 0.05
SEVERE_ENDPOINT_ERROR_M = 0.25
SEVERE_LATERAL_ERROR_M = 0.25
SEVERE_REALIZED_RATIO = 0.50
SEVERE_MIN_PRIMITIVE_PROGRESS_M = 0.05
FORWARD_STEP_M = 0.25
SEVERITY_ORDER = ("free", "grazing", "sliding", "severe_blockage")


def severity_outcome(repeat):
    commanded = float(repeat["commanded_distance_m"])
    realized = float(repeat["realized_forward_m"])
    endpoint = float(repeat["endpoint_error_m"])
    lateral = float(repeat["realized_lateral_error_m"])
    if (
        commanded <= 0.0
        or any(not math.isfinite(value) for value in (realized, endpoint, lateral))
        or not repeat["primitive_trace"]
    ):
        raise ValueError("severity outcome inputs are invalid")
    progresses = [
        float(item["forward_progress_m"]) for item in repeat["primitive_trace"]
    ]
    if any(not math.isfinite(value) for value in progresses):
        raise ValueError("primitive progress is nonfinite")
    realized_ratio = realized / commanded
    deficits = [max(0.0, FORWARD_STEP_M - value) for value in progresses]
    maximum_deficit = max(deficits)
    minimum_progress = min(progresses)
    raw_collision = any(bool(value) for value in repeat["full_collision_trace"])
    severe = (
        realized_ratio < SEVERE_REALIZED_RATIO
        or endpoint >= SEVERE_ENDPOINT_ERROR_M
        or lateral >= SEVERE_LATERAL_ERROR_M
        or minimum_progress <= SEVERE_MIN_PRIMITIVE_PROGRESS_M
    )
    sliding = (
        not severe
        and (
            realized_ratio < FREE_REALIZED_RATIO
            or endpoint > FREE_ENDPOINT_TOLERANCE_M
            or lateral > FREE_LATERAL_TOLERANCE_M
            or maximum_deficit > SLIDING_PRIMITIVE_DEFICIT_M
        )
    )
    if severe:
        label = "severe_blockage"
    elif sliding:
        label = "sliding"
    elif raw_collision:
        label = "grazing"
    else:
        label = "free"
    continuous = min(
        1.0,
        max(
            max(0.0, 1.0 - realized_ratio) / (1.0 - SEVERE_REALIZED_RATIO),
            endpoint / SEVERE_ENDPOINT_ERROR_M,
            lateral / SEVERE_LATERAL_ERROR_M,
            maximum_deficit
            / (FORWARD_STEP_M - SEVERE_MIN_PRIMITIVE_PROGRESS_M),
        ),
    )
    return {
        "severity_class": label,
        "severity_index": SEVERITY_ORDER.index(label),
        "continuous_severity": float(continuous),
        "realized_ratio": float(realized_ratio),
        "maximum_primitive_progress_deficit_m": float(maximum_deficit),
        "minimum_primitive_progress_m": float(minimum_progress),
        "raw_collision": raw_collision,
        "severe_blockage": label == "severe_blockage",
    }


def _ranks(values):
    indexed = sorted(enumerate(float(value) for value in values), key=lambda item: item[1])
    result = [0.0] * len(indexed)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        rank = 0.5 * (start + end - 1) + 1.0
        for offset in range(start, end):
            result[indexed[offset][0]] = rank
        start = end
    return result


def spearman_correlation(first, second):
    if len(first) != len(second) or len(first) < 2:
        raise ValueError("Spearman inputs must align and contain >=2 values")
    x = _ranks(first)
    y = _ranks(second)
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    numerator = sum(
        (left - x_mean) * (right - y_mean) for left, right in zip(x, y)
    )
    x_scale = math.sqrt(sum((value - x_mean) ** 2 for value in x))
    y_scale = math.sqrt(sum((value - y_mean) ** 2 for value in y))
    if x_scale == 0.0 or y_scale == 0.0:
        return 0.0
    return float(numerator / (x_scale * y_scale))


def class_summary(rows, key="severity_class"):
    counts = Counter(row[key] for row in rows)
    return {name: int(counts.get(name, 0)) for name in SEVERITY_ORDER}


def predictor_gates(
    *,
    spearman,
    severe_metrics,
    grazing_false_positive_rate,
    target_repeat_exact,
    pairing_exact,
):
    grazing_available = (
        grazing_false_positive_rate is not None
        and math.isfinite(float(grazing_false_positive_rate))
    )
    if severe_metrics.get("status") != "ok":
        score_gates = {
            "spearman": False,
            "severe_auroc": False,
            "severe_recall": False,
            "severe_balanced_accuracy": False,
            "grazing_false_positive_rate": False,
        }
    else:
        score_gates = {
            "spearman": float(spearman) >= 0.30,
            "severe_auroc": float(severe_metrics["auroc"]) >= 0.75,
            "severe_recall": float(severe_metrics["unsafe_recall"]) >= 0.80,
            "severe_balanced_accuracy": float(
                severe_metrics["balanced_accuracy"]
            ) >= 0.70,
            "grazing_false_positive_rate": grazing_available
            and float(grazing_false_positive_rate) <= 0.30,
        }
    integrity_gates = {
        "target_repeat_exact": bool(target_repeat_exact),
        "pairing_exact": bool(pairing_exact),
    }
    return {
        "score": score_gates,
        "integrity": integrity_gates,
        "score_ready": all(score_gates.values()),
        "integrity_ready": all(integrity_gates.values()),
    }


__all__ = [
    "SEVERITY_ORDER",
    "class_summary",
    "predictor_gates",
    "severity_outcome",
    "spearman_correlation",
]
