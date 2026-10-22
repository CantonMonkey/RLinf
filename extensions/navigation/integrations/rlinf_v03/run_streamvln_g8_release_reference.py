"""Evaluate the original released StreamVLN on the frozen G8 development arms."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import torch
from rlinf.models import get_model

from run_streamvln_ppo_g3 import ORIGINAL_STREAM_COMMIT, make_policy_cfg, sha256_file
from run_streamvln_ppo_g4 import make_env
from run_streamvln_g8_algorithm import MANIFEST_SHA256, RLINF_COMMIT, git_head, metric_value, validate_identity
from streamvln_executor_contract import feedback_from_dict


EXPECTED_SEEDS = (2026092808, 2026092809, 2026092810)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def evaluate_episode(model, run_cfg: dict, spec: dict, seed: int) -> dict:
    max_steps = run_cfg["manifest"]["evaluation"]["max_primitive_steps"]
    env = make_env(run_cfg, split="val_seen", episode_index=spec["index"], max_steps=max_steps)
    actions, decisions = [], []
    blocked_count = 0
    last_info = None
    last_truncated = False
    try:
        obs, identity = env.reset()
        validate_identity(identity, spec, f"release reference seed {seed}")
        for decision in range(max_steps // 4):
            with torch.inference_mode():
                sampled = model(operation="sample", obs=obs)
            proposed = sampled["actions"].detach().cpu().reshape(1, 4, 1)
            obs_list, _, terminations, truncations, infos = env.chunk_step(proposed)
            feedback = feedback_from_dict(infos[0].get("executor_feedback"))
            executed = list(feedback.executed_actions)
            actions.extend(executed)
            blocked_count += feedback.blocked_forward_count
            decisions.append({
                "decision": decision,
                "proposed_actions": list(feedback.proposed_actions),
                "executed_actions": executed,
                "collision_count": feedback.collision_count,
                "blocked_forward_count": feedback.blocked_forward_count,
            })
            obs = obs_list[0]
            last_info = infos[0]
            last_truncated = bool(truncations.any())
            if bool(torch.logical_or(terminations, truncations).any()):
                break
        if last_info is None or "episode" not in last_info:
            raise RuntimeError("G8 release reference lacks terminal metrics")
        metrics = {key: metric_value(value) for key, value in last_info["episode"].items()}
        return {
            **spec,
            "success": metrics["success"],
            "spl": metrics["spl"],
            "navigation_error_m": metrics["distance_to_goal"],
            "path_length_m": metrics["path_length"],
            "collision_count": int(metrics["collision_count"]),
            "blocked_forward_count": blocked_count,
            "primitive_steps": len(actions),
            "decision_count": len(decisions),
            "stop_called": bool(actions and actions[-1] == 0),
            "timeout": last_truncated,
            "actions": actions,
            "decisions": decisions,
        }
    finally:
        env.close()


def aggregate(episodes: list[dict]) -> dict:
    primitives = Counter(value for row in episodes for value in row["actions"])
    return {
        "episode_count": len(episodes),
        "scene_count": len({row["scene"] for row in episodes}),
        "success_rate": sum(row["success"] for row in episodes) / len(episodes),
        "spl": sum(row["spl"] for row in episodes) / len(episodes),
        "navigation_error_m": sum(row["navigation_error_m"] for row in episodes) / len(episodes),
        "path_length_m": sum(row["path_length_m"] for row in episodes) / len(episodes),
        "collision_count": sum(row["collision_count"] for row in episodes) / len(episodes),
        "blocked_forward_count": sum(row["blocked_forward_count"] for row in episodes) / len(episodes),
        "stop_count": sum(row["stop_called"] for row in episodes),
        "timeout_count": sum(row["timeout"] for row in episodes),
        "primitive_counts": {str(value): primitives[value] for value in range(4)},
        "waypoint_execution_error_m": None,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rlinf-root", type=Path, required=True)
    parser.add_argument("--stream-source", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--vision-tower", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--habitat-python", type=Path, required=True)
    parser.add_argument("--habitat-server", type=Path, required=True)
    parser.add_argument("--habitat-sim-python", type=Path, required=True)
    parser.add_argument("--habitat-env-lib", type=Path, required=True)
    parser.add_argument("--egl-vendor-json", type=Path, required=True)
    parser.add_argument("--context-tokens", type=int, default=4096)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    started = time.time()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = args.manifest.expanduser().resolve(strict=True)
    if sha256_file(manifest_path) != MANIFEST_SHA256:
        raise RuntimeError("G8 algorithm manifest hash changed")
    manifest = json.loads(manifest_path.read_text())
    if tuple(manifest["seeds"]) != EXPECTED_SEEDS:
        raise RuntimeError("G8 reference seed identity changed")
    if manifest["evaluation"]["split"] != "val_seen" or manifest["evaluation"]["val_unseen_episode_count"] != 0:
        raise RuntimeError("G8 reference opened val_unseen")

    path_names = (
        "rlinf_root", "stream_source", "base_model", "tokenizer", "vision_tower", "asset_root",
        "habitat_python", "habitat_server", "habitat_sim_python", "habitat_env_lib", "egl_vendor_json",
    )
    paths = {name: str(Path(getattr(args, name)).expanduser().resolve(strict=True)) for name in path_names}
    source_heads = {"stream": git_head(Path(paths["stream_source"])), "rlinf": git_head(Path(paths["rlinf_root"]))}
    if source_heads != {"stream": ORIGINAL_STREAM_COMMIT, "rlinf": RLINF_COMMIT}:
        raise RuntimeError("G8 reference source commit changed")
    if (
        sha256_file(Path(manifest["source"]["val_seen_path"])) != manifest["source"]["val_seen_sha256"]
        or manifest["source"]["val_unseen_read"]
    ):
        raise RuntimeError("G8 reference dataset boundary changed")

    run_cfg = {
        **paths,
        "stream_commit": ORIGINAL_STREAM_COMMIT,
        "rlinf_commit": RLINF_COMMIT,
        "manifest": manifest,
        "seed": EXPECTED_SEEDS[0],
        "context_tokens": args.context_tokens,
    }
    cfg = make_policy_cfg(run_cfg, trainable=False, adapter_checkpoint=None)
    cfg.deterministic_eval = True
    model = get_model(cfg)
    if model is None:
        raise RuntimeError("original StreamVLN policy is not registered")
    model.to("cuda:0")
    model.eval()
    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable_count:
        raise RuntimeError("G8 release reference is not frozen")

    arms = {}
    for seed in EXPECTED_SEEDS:
        episodes = [evaluate_episode(model, run_cfg, spec, seed) for spec in manifest["arms"][str(seed)]["development"]]
        arms[str(seed)] = {"episodes": episodes, "aggregate": aggregate(episodes)}
        write_json(output / f"eval_release_{seed}.json", arms[str(seed)])
    all_episodes = [row for seed in EXPECTED_SEEDS for row in arms[str(seed)]["episodes"]]
    guards = {
        "manifest_hash": sha256_file(manifest_path) == MANIFEST_SHA256,
        "source_pins": source_heads == {"stream": ORIGINAL_STREAM_COMMIT, "rlinf": RLINF_COMMIT},
        "original_release_only": True,
        "frozen": trainable_count == 0,
        "matched_identity_counts": all(arms[str(seed)]["aggregate"]["episode_count"] == 22 for seed in EXPECTED_SEEDS),
        "matched_scene_counts": all(arms[str(seed)]["aggregate"]["scene_count"] == 22 for seed in EXPECTED_SEEDS),
        "zero_val_unseen": manifest["evaluation"]["val_unseen_episode_count"] == 0,
    }
    result = {
        "schema_version": 1,
        "kind": "streamvln-g8-original-release-reference",
        "source_contract": {
            "stream_commit": ORIGINAL_STREAM_COMMIT,
            "rlinf_commit": RLINF_COMMIT,
            "released_checkpoint_only": True,
            "ttt": False,
            "secondary_post_training": False,
            "val_unseen_used": False,
        },
        "model_load_count": 1,
        "trainable_parameter_count": trainable_count,
        "arms": arms,
        "pooled_descriptive": aggregate(all_episodes),
        "guards": guards,
        "all_guards_passed": all(guards.values()),
        "claim_boundary": "descriptive release reference; excluded from the BC versus BC+RL algorithm gate",
        "elapsed_seconds": time.time() - started,
    }
    write_json(output / "result.json", result)
    print(json.dumps({"all_guards": result["all_guards_passed"], "pooled": result["pooled_descriptive"]}, indent=2, sort_keys=True))
    if not result["all_guards_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
