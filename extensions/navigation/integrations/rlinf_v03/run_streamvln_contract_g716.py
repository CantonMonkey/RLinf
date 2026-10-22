"""Run the preregistered G7.16 causal executor-contract smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import torch

from run_streamvln_ppo_g4 import make_env
from streamvln_chunk_ppo import collapse_chunk_rewards
from streamvln_executor_contract import (
    PRIVILEGED_POLICY_KEYS,
    assert_history_only_keys,
    feedback_from_dict,
)
from streamvln_topology import RECOVER_MODE, OnlineTopology, compile_waypoint


ORIGINAL_STREAM_COMMIT = "e48f6ff7e9201d93aae003e8f64b04c00cec13bc"
RLINF_COMMIT = "0505431899574619da86f551bad70b71e0ea2177"
MANIFEST_SHA256 = "3a60baace9865f6df88c2b8369bd988ddced5b64624306b307d5b175676ba15c"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head(path: Path) -> str:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={path}", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=20,
    )
    if result.returncode:
        raise RuntimeError(f"cannot resolve git head for {path}: {result.stderr}")
    return result.stdout.strip()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def observation_sha256(obs: dict, expected_keys: set[str]) -> str:
    if set(obs) != expected_keys:
        raise RuntimeError("G7.16 policy observation schema changed")
    assert_history_only_keys(list(obs))
    digest = hashlib.sha256()
    for key in sorted(obs):
        value = obs[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def dataset_hashes(asset_root: Path, manifest: dict) -> dict:
    result = {}
    for split in ("train", "val_unseen"):
        path = asset_root / "datasets" / "r2r" / split / f"{split}.json.gz"
        actual = sha256_file(path.resolve(strict=True))
        expected = manifest["source"][f"{split}_sha256"]
        if actual != expected:
            raise RuntimeError(f"G7.16 {split} dataset hash changed")
        result[split] = {"path": str(path), "sha256": actual}
    return result


def validate_manifest(manifest: dict) -> dict:
    expected_top = {
        "schema_version", "stage", "source_contract", "source", "high_level_command",
        "lower_layer_feedback", "high_level_policy_observation_keys",
        "privileged_policy_blacklist", "deterministic_smoke", "gates", "model_load",
        "model_fit", "rl_update", "controller_comparison", "navigation_evaluation",
        "g8_ready",
    }
    if set(manifest) != expected_top or manifest["schema_version"] != 1:
        raise ValueError("G7.16 manifest schema changed")
    source = manifest["source_contract"]
    if (
        source["stream_commit"] != ORIGINAL_STREAM_COMMIT
        or source["rlinf_commit"] != RLINF_COMMIT
        or source["original_stream_only"] is not True
        or source["ttt_or_secondary_post_training"] is not False
    ):
        raise ValueError("G7.16 source contract changed")
    smoke = manifest["deterministic_smoke"]
    if (
        smoke["split"] != "train"
        or len(smoke["scene_specs"]) != 2
        or smoke["repeat_count"] != 2
        or smoke["action_chunks"] != [[1, 1, 2, 1], [3, 1, 1, 1], [0, 0, 0, 0]]
        or smoke["expected_executed_lengths"] != [4, 4, 1]
        or smoke["val_unseen_episode_count"] != 0
    ):
        raise ValueError("G7.16 deterministic smoke contract changed")
    if set(manifest["privileged_policy_blacklist"]) != PRIVILEGED_POLICY_KEYS:
        raise ValueError("G7.16 privileged-data blacklist changed")
    assert_history_only_keys(manifest["high_level_policy_observation_keys"])
    for key in (
        "model_load", "model_fit", "rl_update", "controller_comparison",
        "navigation_evaluation", "g8_ready",
    ):
        if manifest[key] is not False:
            raise ValueError(f"G7.16 {key} must remain false")
    return smoke


def make_run_cfg(args, manifest: dict) -> dict:
    return {
        "seed": 20260927,
        "habitat_python": str(Path(args.habitat_python).resolve(strict=True)),
        "habitat_server": str(Path(args.habitat_server).resolve(strict=True)),
        "asset_root": str(Path(args.asset_root).resolve(strict=True)),
        "habitat_sim_python": str(Path(args.habitat_sim_python).resolve(strict=True)),
        "habitat_env_lib": str(Path(args.habitat_env_lib).resolve(strict=True)),
        "egl_vendor_json": str(Path(args.egl_vendor_json).resolve(strict=True)),
        "manifest": {
            "source": manifest["source"],
            "train": manifest["deterministic_smoke"]["scene_specs"],
            "val_unseen": [],
        },
    }


def validate_identity(identity: dict, spec: dict) -> None:
    if (
        str(identity.get("episode_id")) != str(spec["episode_id"])
        or Path(identity.get("scene_id", "")).stem != spec["scene"]
    ):
        raise RuntimeError("G7.16 Habitat episode identity changed")


def collect_repeat(run_cfg: dict, smoke: dict, spec: dict, repeat_index: int) -> dict:
    env = make_env(
        run_cfg,
        split="train",
        episode_index=spec["index"],
        max_steps=smoke["max_episode_steps"],
    )
    topology = OnlineTopology(max_decisions=len(smoke["action_chunks"]))
    expected_obs_keys = set(run_cfg["policy_observation_keys"])
    try:
        obs, identity = env.reset()
        validate_identity(identity, spec)
        observation_hashes = [observation_sha256(obs, expected_obs_keys)]
        chunks = []
        recovery_actions = None
        for decision_index, (proposal, expected_length) in enumerate(zip(
            smoke["action_chunks"], smoke["expected_executed_lengths"]
        )):
            obs_list, rewards, terminations, truncations, infos = env.chunk_step(
                torch.tensor(proposal, dtype=torch.int64).reshape(1, 4, 1)
            )
            feedback = feedback_from_dict(infos[0].get("executor_feedback"))
            if len(feedback.executed_actions) != expected_length:
                raise RuntimeError("G7.16 executed prefix length changed")
            reward_row = [float(value) for value in rewards[0].detach().cpu().tolist()]
            terminated_row = [bool(value) for value in terminations[0].detach().cpu().tolist()]
            truncated_row = [bool(value) for value in truncations[0].detach().cpu().tolist()]
            if any(abs(value) > 1e-8 for value in reward_row[expected_length:]):
                raise RuntimeError("G7.16 padded rewards are not zero")
            reward_sum_error = abs(
                feedback.environment_reward_sum - sum(reward_row[:expected_length])
            )
            collapsed, durations = collapse_chunk_rewards(
                [reward_row], [list(feedback.executed_mask)], smoke["gamma"]
            )
            discounted_error = abs(
                feedback.discounted_environment_reward(smoke["gamma"]) - collapsed[0]
            )
            if reward_sum_error > 1e-6 or discounted_error > 1e-6 or durations != [expected_length]:
                raise RuntimeError("G7.16 reward accounting differs from RLinf chunk math")
            topology.update_from_feedback(feedback.to_dict())
            obs = obs_list[0]
            observation_hashes.append(observation_sha256(obs, expected_obs_keys))
            terminal = any(a or b for a, b in zip(terminated_row, truncated_row))
            if terminal != (decision_index == len(smoke["action_chunks"]) - 1):
                raise RuntimeError("G7.16 episode ended outside the fixed STOP decision")
            chunks.append({
                "decision_index": decision_index,
                "feedback": feedback.to_dict(),
                "primitive_rewards": reward_row,
                "terminated": terminated_row,
                "truncated": truncated_row,
                "reward_sum_error": reward_sum_error,
                "discounted_reward_error": discounted_error,
                "topology": topology.summary(),
            })
            if decision_index == smoke["recovery_probe_after_decision_index"]:
                recovery_actions = compile_waypoint([0.0, 0.0], RECOVER_MODE, topology)
                if len(recovery_actions) != 4 or 0 in recovery_actions:
                    raise RuntimeError("G7.16 recovery did not compile to four non-STOP actions")
        return {
            "scene": spec["scene"],
            "episode_index": spec["index"],
            "episode_id": str(spec["episode_id"]),
            "repeat_index": repeat_index,
            "observation_sha256": observation_hashes,
            "chunks": chunks,
            "recovery_actions": recovery_actions,
            "final_topology": topology.summary(),
        }
    finally:
        env.close()


def repeat_payload(record: dict) -> dict:
    return {
        key: value
        for key, value in record.items()
        if key != "repeat_index"
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stream-source", type=Path, required=True)
    parser.add_argument("--rlinf-source", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--habitat-python", type=Path, required=True)
    parser.add_argument("--habitat-server", type=Path, required=True)
    parser.add_argument("--habitat-sim-python", type=Path, required=True)
    parser.add_argument("--habitat-env-lib", type=Path, required=True)
    parser.add_argument("--egl-vendor-json", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    started = time.time()
    manifest_path = args.manifest.resolve(strict=True)
    if sha256_file(manifest_path) != MANIFEST_SHA256:
        raise RuntimeError("G7.16 frozen manifest hash changed")
    manifest = json.loads(manifest_path.read_text())
    smoke = validate_manifest(manifest)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    stream_source = args.stream_source.resolve(strict=True)
    rlinf_source = args.rlinf_source.resolve(strict=True)
    source_heads = {
        "stream": git_head(stream_source),
        "rlinf": git_head(rlinf_source),
    }
    if source_heads != {"stream": ORIGINAL_STREAM_COMMIT, "rlinf": RLINF_COMMIT}:
        raise RuntimeError("G7.16 source commit changed")
    datasets = dataset_hashes(args.asset_root.resolve(strict=True), manifest)
    run_cfg = make_run_cfg(args, manifest)
    run_cfg["policy_observation_keys"] = manifest["high_level_policy_observation_keys"]
    config = {
        "kind": "streamvln-g7.16-causal-executor-contract",
        "manifest": str(manifest_path),
        "manifest_sha256": MANIFEST_SHA256,
        "source_heads": source_heads,
        "datasets": datasets,
        "run": run_cfg,
        "smoke": smoke,
        "model_load_count": 0,
        "policy_update_count": 0,
        "val_unseen_episode_count": 0,
    }
    config_path = output_dir / "run_config.json"
    write_json(config_path, config)

    records = []
    repeat_exact = {}
    for spec in smoke["scene_specs"]:
        scene_rows = [
            collect_repeat(run_cfg, smoke, spec, repeat_index)
            for repeat_index in range(smoke["repeat_count"])
        ]
        exact = repeat_payload(scene_rows[0]) == repeat_payload(scene_rows[1])
        repeat_exact[spec["scene"]] = exact
        records.extend(scene_rows)

    rows_path = output_dir / "execution_records.jsonl"
    rows_path.write_text("".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    ))
    chunk_count = sum(len(record["chunks"]) for record in records)
    primitive_count = sum(
        len(chunk["feedback"]["executed_actions"])
        for record in records
        for chunk in record["chunks"]
    )
    max_reward_error = max(
        max(chunk["reward_sum_error"], chunk["discounted_reward_error"])
        for record in records
        for chunk in record["chunks"]
    )
    guards = {
        "source_pins": source_heads == {
            "stream": ORIGINAL_STREAM_COMMIT, "rlinf": RLINF_COMMIT
        },
        "manifest_hash": sha256_file(manifest_path) == MANIFEST_SHA256,
        "dataset_hashes": all(
            datasets[split]["sha256"] == manifest["source"][f"{split}_sha256"]
            for split in datasets
        ),
        "record_count": len(records) == smoke["expected_scene_repeat_records"],
        "chunk_count": chunk_count == smoke["expected_chunk_records"],
        "executed_primitive_count": primitive_count == smoke["expected_executed_primitives"],
        "repeat_exact": all(repeat_exact.values()),
        "reward_accounting": max_reward_error <= 1e-6,
        "history_only_observation": not (
            set(manifest["high_level_policy_observation_keys"]) & PRIVILEGED_POLICY_KEYS
        ),
        "recovery_compilation": all(
            len(record["recovery_actions"]) == 4 and 0 not in record["recovery_actions"]
            for record in records
        ),
        "no_model_or_policy_update": config["model_load_count"] == 0
        and config["policy_update_count"] == 0,
        "zero_val_unseen": config["val_unseen_episode_count"] == 0,
    }
    all_passed = all(guards.values())
    result = {
        "kind": "streamvln-g7.16-causal-executor-contract",
        "all_protocol_guards_passed": all_passed,
        "executor_contract_ready": all_passed,
        "source_contract": {
            "stream_commit": ORIGINAL_STREAM_COMMIT,
            "rlinf_commit": RLINF_COMMIT,
            "original_stream_only": True,
            "ttt_or_secondary_post_training": False,
        },
        "counts": {
            "scene_repeat_records": len(records),
            "chunk_records": chunk_count,
            "executed_primitives": primitive_count,
        },
        "repeat_exact_by_scene": repeat_exact,
        "maximum_reward_accounting_error": max_reward_error,
        "protocol_guards": guards,
        "model_load_count": 0,
        "policy_update_count": 0,
        "val_unseen_episode_count": 0,
        "g8_ready": False,
        "permitted_next_stage": "G7.17 bounded train-only high-level RL plumbing smoke"
        if all_passed else "instrumentation repair only",
        "elapsed_seconds": time.time() - started,
        "evidence": {
            "run_config_sha256": sha256_file(config_path),
            "execution_records_sha256": sha256_file(rows_path),
            "manifest_sha256": MANIFEST_SHA256,
        },
    }
    result_path = output_dir / "result.json"
    write_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not all_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
