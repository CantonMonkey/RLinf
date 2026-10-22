"""One-episode hardware check for G7.8 aligned metric-depth causality."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from run_streamvln_candidates_g7 import validate_identity
from run_streamvln_ppo_g4 import make_env
from streamvln_candidates_g74 import generate_macro_candidates
from streamvln_topology import OnlineTopology


def sha256_depth(depth) -> str:
    values = np.ascontiguousarray(depth, dtype="<f4")
    return hashlib.sha256(values.tobytes()).hexdigest()


def parse_args():
    parser = argparse.ArgumentParser()
    for name in (
        "manifest",
        "output",
        "habitat-python",
        "habitat-depth-passability-server",
        "asset-root",
        "habitat-sim-python",
        "habitat-env-lib",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument(
        "--egl-vendor-json",
        default="/usr/share/glvnd/egl_vendor.d/50_mesa.json",
    )
    parser.add_argument("--episode-offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260927)
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = json.loads(Path(args.manifest).resolve(strict=True).read_text())
    spec = manifest["train"][args.episode_offset]
    cfg = {
        "seed": args.seed,
        "habitat_python": str(Path(args.habitat_python).resolve(strict=True)),
        "habitat_server": str(
            Path(args.habitat_depth_passability_server).resolve(strict=True)
        ),
        "asset_root": str(Path(args.asset_root).resolve(strict=True)),
        "habitat_sim_python": str(
            Path(args.habitat_sim_python).resolve(strict=True)
        ),
        "habitat_env_lib": str(Path(args.habitat_env_lib).resolve(strict=True)),
        "egl_vendor_json": str(Path(args.egl_vendor_json).resolve(strict=True)),
    }
    env = make_env(cfg, split="train", episode_index=spec["index"], max_steps=40)
    try:
        obs, identity = env.reset()
        validate_identity(identity, spec)
        policy_keys = sorted(obs)
        initial_depth = env.observe_aligned_depth()
        candidates = generate_macro_candidates(OnlineTopology(max_decisions=40))
        probe = env.probe_depth_candidates(candidates)
        rows = {row["name"]: row for row in probe["candidates"]}
        straight = next(candidate for candidate in candidates if candidate.actions[0] == 1)
        turned = next(
            candidate
            for candidate in candidates
            if candidate.actions[0] in (2, 3)
            and candidate.actions.index(1) == 1
        )
        straight_depth = rows[straight.name]["metric_depth"]
        first_forward = turned.actions.index(1)
        for action in turned.actions[:first_forward]:
            _obs, _reward, terminated, truncated, _info = env.step(action)
            if bool(terminated.item()) or bool(truncated.item()):
                raise RuntimeError("G7.8 smoke turn prefix ended the episode")
        actual_turned_depth = env.observe_aligned_depth()
        branch_turned_depth = rows[turned.name]["metric_depth"]
        result = {
            "schema_version": 1,
            "kind": "streamvln-g7.8-depth-causality-hardware-smoke",
            "episode": spec,
            "candidate_count": len(candidates),
            "unique_depth_count": len(
                {sha256_depth(row["metric_depth"]) for row in rows.values()}
            ),
            "policy_observation_keys": policy_keys,
            "depth_outside_policy_observation": "depth" not in policy_keys,
            "state_restored": True,
            "max_position_restore_error_m": probe["max_position_restore_error_m"],
            "max_rotation_restore_error": probe["max_rotation_restore_error"],
            "straight_candidate": straight.name,
            "straight_matches_initial": bool(
                np.array_equal(straight_depth, initial_depth)
            ),
            "straight_depth_sha256": sha256_depth(straight_depth),
            "initial_depth_sha256": sha256_depth(initial_depth),
            "turned_candidate": turned.name,
            "turned_prefix": list(turned.actions[:first_forward]),
            "branch_turned_matches_actual": bool(
                np.array_equal(branch_turned_depth, actual_turned_depth)
            ),
            "branch_turned_depth_sha256": sha256_depth(branch_turned_depth),
            "actual_turned_depth_sha256": sha256_depth(actual_turned_depth),
            "depth_min_m": float(
                min(row["metric_depth"].min() for row in rows.values())
            ),
            "depth_max_m": float(
                max(row["metric_depth"].max() for row in rows.values())
            ),
        }
        result["all_guards_passed"] = bool(
            result["candidate_count"] == 33
            and result["unique_depth_count"] > 1
            and result["depth_outside_policy_observation"]
            and result["max_position_restore_error_m"] <= 1e-6
            and result["max_rotation_restore_error"] <= 1e-6
            and result["straight_matches_initial"]
            and result["branch_turned_matches_actual"]
            and 0.0 <= result["depth_min_m"] <= result["depth_max_m"] <= 10.0001
        )
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, sort_keys=True))
        if not result["all_guards_passed"]:
            raise RuntimeError("G7.8 depth causality hardware smoke failed")
    finally:
        env.close()


if __name__ == "__main__":
    main()
