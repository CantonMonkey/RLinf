"""Aggregate the preregistered three-seed StreamVLN G8 algorithm gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_MANIFEST_SHA256 = "59b414d83ab30e38cb4fb2c5bce14b8408bed5f2a23cfeb41aeb84570cb2331a"
EXPECTED_SEEDS = (2026092808, 2026092809, 2026092810)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_result(path: Path) -> dict:
    result = json.loads(path.read_text())
    if result.get("schema_version") != 1 or result.get("kind") != "streamvln-g8-matched-bc-vs-bc-rl":
        raise ValueError(f"unexpected G8 result schema: {path}")
    if result.get("seed") not in EXPECTED_SEEDS:
        raise ValueError(f"unregistered G8 seed in {path}")
    if result.get("counts") != {
        "development_episodes_per_arm": 22,
        "model_loads": 1,
        "train_episodes": 32,
        "val_unseen_episodes": 0,
    }:
        raise ValueError(f"G8 result counts changed: {path}")
    contract = result.get("source_contract", {})
    if contract.get("ttt") or contract.get("secondary_post_training") or contract.get("val_unseen_used"):
        raise ValueError(f"G8 source boundary changed: {path}")
    return result


def weighted_mean(results: list[dict], variant: str, metric: str) -> float:
    rows = [item["development"][variant] for item in results]
    total = sum(row["episode_count"] for row in rows)
    return sum(row[metric] * row["episode_count"] for row in rows) / total


def aggregate(manifest_path: Path, result_paths: list[Path]) -> dict:
    if sha256_file(manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("G8 algorithm manifest hash changed")
    manifest = json.loads(manifest_path.read_text())
    if tuple(manifest.get("seeds", ())) != EXPECTED_SEEDS:
        raise ValueError("G8 seed order changed")
    if manifest["evaluation"]["split"] != "val_seen" or manifest["evaluation"]["val_unseen_episode_count"] != 0:
        raise ValueError("G8 development boundary changed")

    results = [load_result(path) for path in result_paths]
    seeds = [item["seed"] for item in results]
    if len(results) != 3 or tuple(sorted(seeds)) != EXPECTED_SEEDS or len(set(seeds)) != 3:
        raise ValueError("G8 aggregation requires exactly one result for every preregistered seed")
    results.sort(key=lambda item: item["seed"])

    seed_rows = []
    for result in results:
        bc = result["development"]["bc"]
        rl = result["development"]["bc_rl"]
        modes = sum(value > 0 for value in rl["mode_counts"].values())
        primitives = sum(value > 0 for value in rl["primitive_counts"].values())
        seed_rows.append({
            "seed": result["seed"],
            "all_guards": bool(result["all_training_and_evaluation_guards_passed"] and all(result["guards"].values())),
            "bc_success_rate": bc["success_rate"],
            "bc_rl_success_rate": rl["success_rate"],
            "success_rate_delta": rl["success_rate"] - bc["success_rate"],
            "spl_delta": rl["spl"] - bc["spl"],
            "timeout_delta": rl["timeout_count"] - bc["timeout_count"],
            "bc_rl_mode_count": modes,
            "bc_rl_primitive_category_count": primitives,
            "no_action_collapse": modes >= 2 and primitives >= 2,
        })

    pooled = {
        variant: {
            "episode_count": sum(item["development"][variant]["episode_count"] for item in results),
            "success_rate": weighted_mean(results, variant, "success_rate"),
            "spl": weighted_mean(results, variant, "spl"),
            "timeout_count": sum(item["development"][variant]["timeout_count"] for item in results),
        }
        for variant in ("bc", "bc_rl")
    }
    pooled["paired_delta"] = {
        "success_rate": pooled["bc_rl"]["success_rate"] - pooled["bc"]["success_rate"],
        "spl": pooled["bc_rl"]["spl"] - pooled["bc"]["spl"],
        "timeout_count": pooled["bc_rl"]["timeout_count"] - pooled["bc"]["timeout_count"],
    }

    gates = {
        "all_training_and_evaluation_guards": all(row["all_guards"] for row in seed_rows),
        "seed_majority_success_rate": sum(row["success_rate_delta"] > 0 for row in seed_rows) >= 2,
        "pooled_success_rate": pooled["paired_delta"]["success_rate"] > 0,
        "pooled_spl_noninferiority": pooled["paired_delta"]["spl"] >= 0,
        "pooled_timeout_noninferiority": pooled["paired_delta"]["timeout_count"] <= 0,
        "no_action_collapse": all(row["no_action_collapse"] for row in seed_rows),
        "zero_val_unseen": all(item["counts"]["val_unseen_episodes"] == 0 for item in results),
    }
    return {
        "schema_version": 1,
        "kind": "streamvln-g8-three-seed-algorithm-gate",
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "seeds": seed_rows,
        "pooled": pooled,
        "gates": gates,
        "algorithm_gate_passed": all(gates.values()),
        "next_stage": "freeze_final_val_unseen_schedule" if all(gates.values()) else "report_algorithm_gate_failure",
        "claim_boundary": {
            "split": "val_seen",
            "val_unseen_episodes": 0,
            "navigation_generalization_claim": False,
        },
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    report = aggregate(args.manifest.resolve(strict=True), [path.resolve(strict=True) for path in args.result])
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
