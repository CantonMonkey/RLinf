"""Pure analysis helpers for the preregistered G7.9 macro audit."""

from __future__ import annotations

import math
from typing import Sequence



EXCLUDED_CANDIDATE = "macro_p90_d0p5"
REGISTERED_CANDIDATES_PER_SCENE = 32


def select_registered_candidates(candidates: Sequence):
    selected = tuple(
        candidate for candidate in candidates if candidate.name != EXCLUDED_CANDIDATE
    )
    if len(candidates) != 33 or len(selected) != REGISTERED_CANDIDATES_PER_SCENE:
        raise RuntimeError("G7.9 preregistered macro bank changed")
    if EXCLUDED_CANDIDATE not in {candidate.name for candidate in candidates}:
        raise RuntimeError("G7.9 excluded candidate is absent")
    return selected


def distribution(values):
    ordered = sorted(float(value) for value in values)
    if not ordered or any(not math.isfinite(value) for value in ordered):
        raise ValueError("distribution requires finite scalar values")

    def quantile(probability):
        index = (len(ordered) - 1) * probability
        lower = int(math.floor(index))
        upper = int(math.ceil(index))
        weight = index - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "p50": quantile(0.50),
        "p90": quantile(0.90),
        "max": ordered[-1],
    }


def safe_binary_metrics(labels, scores, threshold):
    from streamvln_passability_g76 import binary_metrics
    labels = [bool(value) for value in labels]
    if len(labels) != len(scores) or not labels:
        raise ValueError("metric labels and scores must align")
    if len(set(labels)) != 2:
        return {
            "status": "single_class",
            "count": len(labels),
            "unsafe": sum(labels),
            "safe": len(labels) - sum(labels),
        }
    result = binary_metrics(labels, [float(value) for value in scores], threshold)
    result["status"] = "ok"
    return result


def distance_prefix_audit(rows, label_key):
    grouped = {}
    for row in rows:
        key = (row["scene"], float(row["heading_degrees"]))
        grouped.setdefault(key, []).append(row)
    violations = []
    checked_pairs = 0
    for (scene, heading), items in sorted(grouped.items()):
        items.sort(key=lambda row: (float(row["distance_m"]), row["candidate"]))
        for first, second in zip(items, items[1:]):
            checked_pairs += 1
            if bool(first[label_key]) and not bool(second[label_key]):
                violations.append(
                    {
                        "scene": scene,
                        "heading_degrees": heading,
                        "shorter": first["candidate"],
                        "longer": second["candidate"],
                    }
                )
    return {
        "label": label_key,
        "checked_pairs": checked_pairs,
        "violation_count": len(violations),
        "violations": violations,
    }


def classify_route(
    *,
    pairing_exact,
    collision_repeat_exact,
    prefix_monotonic,
    privileged_contradiction_rate,
    endpoint_repeat_max_m,
    horizon_label_change_rate,
    horizon_balanced_accuracy_gain,
):
    pairing_issue = not bool(pairing_exact)
    executor_issue = (
        not bool(collision_repeat_exact)
        or not bool(prefix_monotonic)
        or float(privileged_contradiction_rate) > 0.05
        or float(endpoint_repeat_max_m) > 0.01
    )
    horizon_issue = (
        float(horizon_label_change_rate) >= 0.10
        or (
            math.isfinite(float(horizon_balanced_accuracy_gain))
            and float(horizon_balanced_accuracy_gain) >= 0.10
        )
    )
    routes = []
    if pairing_issue:
        routes.append("repair-observation-action-pairing")
    if horizon_issue:
        if (
            math.isfinite(float(horizon_balanced_accuracy_gain))
            and float(horizon_balanced_accuracy_gain) >= 0.10
        ):
            routes.append("repair-collision-label-horizon")
        else:
            routes.append("diagnose-collision-target-severity-semantics")
    if executor_issue:
        routes.append("diagnose-contact-geometry-and-macro-deviation")
    if not routes:
        routes.append("freeze-audit-contract-and-revisit-target-or-model")
    return {
        "pairing_issue": pairing_issue,
        "label_horizon_issue": horizon_issue,
        "executor_issue": executor_issue,
        "routes": routes,
    }


__all__ = [
    "EXCLUDED_CANDIDATE",
    "REGISTERED_CANDIDATES_PER_SCENE",
    "classify_route",
    "distance_prefix_audit",
    "distribution",
    "safe_binary_metrics",
    "select_registered_candidates",
]
