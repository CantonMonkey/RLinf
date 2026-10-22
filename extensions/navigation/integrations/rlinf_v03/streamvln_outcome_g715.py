"""Pure no-fit metrics and deterministic matching for G7.15."""

from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
import math
import statistics


EXPECTED_TARGET_RAY_COUNT = 48


def visual_risk(observables):
    """Return the preregistered equal-weight non-support fraction."""
    required = (
        "target_ray_count",
        "classified_count",
        "unclipped_count",
        "support_fraction",
        "occlusion_fraction",
        "missing_fraction",
    )
    if not isinstance(observables, dict) or any(key not in observables for key in required):
        raise ValueError("G7.15 observables are incomplete")
    if (
        observables["target_ray_count"] != EXPECTED_TARGET_RAY_COUNT
        or observables["classified_count"] != EXPECTED_TARGET_RAY_COUNT
        or observables["unclipped_count"] != EXPECTED_TARGET_RAY_COUNT
    ):
        raise ValueError("G7.15 requires 48 classified unclipped target rays")
    support = float(observables["support_fraction"])
    occlusion = float(observables["occlusion_fraction"])
    missing = float(observables["missing_fraction"])
    if (
        any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in (support, occlusion, missing))
        or not math.isclose(support + occlusion + missing, 1.0, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ValueError("G7.15 ray fractions do not form a probability partition")
    risk = occlusion + missing
    if not math.isclose(risk, 1.0 - support, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("G7.15 frozen risk identity changed")
    return float(risk)


def binary_auroc(labels, scores):
    labels = [bool(value) for value in labels]
    scores = [float(value) for value in scores]
    if len(labels) != len(scores) or not labels:
        raise ValueError("G7.15 AUROC inputs must align and be nonempty")
    if any(not math.isfinite(value) for value in scores):
        raise ValueError("G7.15 AUROC scores must be finite")
    positive = [score for label, score in zip(labels, scores) if label]
    negative = [score for label, score in zip(labels, scores) if not label]
    if not positive or not negative:
        return {
            "status": "single_class",
            "count": len(labels),
            "severe": len(positive),
            "nonsevere": len(negative),
            "auroc": None,
        }
    wins = ties = 0
    for severe_score in positive:
        for control_score in negative:
            if severe_score > control_score:
                wins += 1
            elif severe_score == control_score:
                ties += 1
    comparisons = len(positive) * len(negative)
    return {
        "status": "ok",
        "count": len(labels),
        "severe": len(positive),
        "nonsevere": len(negative),
        "pairwise_comparisons": comparisons,
        "strict_wins": wins,
        "ties": ties,
        "auroc": float((wins + 0.5 * ties) / comparisons),
    }


def stratified_auroc(rows, key):
    result = {}
    for value in sorted({row[key] for row in rows}):
        subset = [row for row in rows if row[key] == value]
        result[str(value)] = binary_auroc(
            [row["severe_blockage"] for row in subset],
            [row["visual_risk"] for row in subset],
        )
    return result


def binary_class_counts(rows):
    severe = sum(bool(row["severe_blockage"]) for row in rows)
    return {"severe": int(severe), "nonsevere": int(len(rows) - severe)}


def stratified_class_counts(rows, key):
    return {
        str(value): binary_class_counts([row for row in rows if row[key] == value])
        for value in sorted({row[key] for row in rows})
    }


def _matching_objective(pairs, total_gap, positives, controls):
    names = tuple(
        sorted(
            (positives[left]["candidate"], controls[right]["candidate"])
            for left, right in pairs
        )
    )
    return (-len(pairs), float(total_gap), names)


def _match_stratum(positives, controls, maximum_heading_gap_degrees):
    positives = tuple(sorted(positives, key=lambda row: row["candidate"]))
    controls = tuple(sorted(controls, key=lambda row: row["candidate"]))
    if len({row["candidate"] for row in positives}) != len(positives):
        raise ValueError("G7.15 severe candidates are not unique within a stratum")
    if len({row["candidate"] for row in controls}) != len(controls):
        raise ValueError("G7.15 control candidates are not unique within a stratum")

    @lru_cache(maxsize=None)
    def solve(index, used_mask):
        if index == len(positives):
            return (), 0.0
        best_pairs, best_gap = solve(index + 1, used_mask)
        best_key = _matching_objective(best_pairs, best_gap, positives, controls)
        severe = positives[index]
        for control_index, control in enumerate(controls):
            if used_mask & (1 << control_index):
                continue
            gap = abs(float(severe["heading_degrees"]) - float(control["heading_degrees"]))
            if gap > maximum_heading_gap_degrees + 1e-12:
                continue
            tail_pairs, tail_gap = solve(index + 1, used_mask | (1 << control_index))
            proposal_pairs = ((index, control_index),) + tail_pairs
            proposal_gap = gap + tail_gap
            proposal_key = _matching_objective(
                proposal_pairs, proposal_gap, positives, controls
            )
            if proposal_key < best_key:
                best_pairs, best_gap, best_key = (
                    proposal_pairs,
                    proposal_gap,
                    proposal_key,
                )
        return best_pairs, best_gap

    pairs, _ = solve(0, 0)
    result = []
    for severe_index, control_index in pairs:
        severe = positives[severe_index]
        control = controls[control_index]
        gap = abs(float(severe["heading_degrees"]) - float(control["heading_degrees"]))
        result.append(
            {
                "scene": severe["scene"],
                "state_name": severe["state_name"],
                "distance_m": float(severe["distance_m"]),
                "severe_candidate": severe["candidate"],
                "control_candidate": control["candidate"],
                "severe_heading_degrees": float(severe["heading_degrees"]),
                "control_heading_degrees": float(control["heading_degrees"]),
                "heading_gap_degrees": float(gap),
                "severe_visual_risk": float(severe["visual_risk"]),
                "control_visual_risk": float(control["visual_risk"]),
                "strict_win": bool(severe["visual_risk"] > control["visual_risk"]),
            }
        )
    return sorted(
        result, key=lambda row: (row["severe_candidate"], row["control_candidate"])
    )


def deterministic_matches(rows, maximum_heading_gap_degrees=30.0):
    if not math.isfinite(maximum_heading_gap_degrees) or maximum_heading_gap_degrees < 0.0:
        raise ValueError("G7.15 maximum heading gap is invalid")
    groups = defaultdict(list)
    for row in rows:
        required = (
            "scene",
            "state_name",
            "distance_m",
            "candidate",
            "heading_degrees",
            "severe_blockage",
            "visual_risk",
        )
        if any(key not in row for key in required):
            raise ValueError("G7.15 matching row is incomplete")
        key = (row["scene"], row["state_name"], float(row["distance_m"]))
        groups[key].append(row)
    matched = []
    for key in sorted(groups):
        group = groups[key]
        positives = [row for row in group if row["severe_blockage"]]
        controls = [row for row in group if not row["severe_blockage"]]
        matched.extend(
            _match_stratum(positives, controls, maximum_heading_gap_degrees)
        )
    severe_names = {
        (row["scene"], row["state_name"], row["distance_m"], row["severe_candidate"])
        for row in matched
    }
    control_names = {
        (row["scene"], row["state_name"], row["distance_m"], row["control_candidate"])
        for row in matched
    }
    if len(severe_names) != len(matched) or len(control_names) != len(matched):
        raise RuntimeError("G7.15 deterministic matching is not one-to-one")
    return sorted(
        matched,
        key=lambda row: (
            row["scene"],
            row["state_name"],
            row["distance_m"],
            row["severe_candidate"],
            row["control_candidate"],
        ),
    )


def matched_summary(matches):
    by_state = Counter(row["state_name"] for row in matches)
    by_scene = Counter(row["scene"] for row in matches)
    severe_scores = [float(row["severe_visual_risk"]) for row in matches]
    control_scores = [float(row["control_visual_risk"]) for row in matches]
    return {
        "pair_count": len(matches),
        "strict_win_count": sum(row["strict_win"] for row in matches),
        "strict_win_rate": (
            sum(row["strict_win"] for row in matches) / len(matches)
            if matches
            else None
        ),
        "pair_count_by_state": dict(sorted(by_state.items())),
        "pair_count_by_scene": dict(sorted(by_scene.items())),
        "scenes_with_at_least_three_pairs": sum(value >= 3 for value in by_scene.values()),
        "median_severe_visual_risk": (
            float(statistics.median(severe_scores)) if severe_scores else None
        ),
        "median_control_visual_risk": (
            float(statistics.median(control_scores)) if control_scores else None
        ),
    }


__all__ = [
    "binary_auroc",
    "binary_class_counts",
    "deterministic_matches",
    "matched_summary",
    "stratified_auroc",
    "stratified_class_counts",
    "visual_risk",
]
