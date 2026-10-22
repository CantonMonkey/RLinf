"""Run the preregistered G7.19 v2 optimizer-contract repair audit."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from run_streamvln_learning_g718 import (
    FEEDBACK_KEYS,
    ORIGINAL_STREAM_COMMIT,
    RLINF_COMMIT,
    aggregate_behavior,
    collect_phase,
    flat_rows,
    structured_distribution_kl,
    write_jsonl,
)
from run_streamvln_plumbing_g717 import (
    dataset_hashes,
    frozen_parameter_sentinels,
    git_head,
    optimizer_parameter_scope,
    sha256_file,
    small_file_inventory,
    write_json,
)
from run_streamvln_recovery_g6 import replay_individual, setup_manager
from run_streamvln_topology_g5 import (
    cleanup_manager,
    gradient_audit,
    model_device,
    parameter_changes,
    parameter_snapshot,
    state_sha256,
)
from streamvln_executor_contract import PRIVILEGED_POLICY_KEYS, assert_history_only_keys


G718_LOCAL_COMMIT = "42566317b3435d753873be6ef277eb142c8ac929"
G718_MANIFEST_SHA256 = "a985243b5aadf19ee8ab37ac534aa1292d095588f8a49427c984e81656864aab"
G718_RESULT_SHA256 = "0baac522d5c0d8a4b1814ff61e3b75991e4639cf898196c35b7379695a92e95f"
G718_CONFIG_SHA256 = "8ac8243fe40a3ecdbcab8ca600f3d4405a523f80df1f639d35ae3f000530d685"
G718_COLLECTION_SHA256 = "de94c1876da52efad1773363d3a6062449ab8db45e1295544645957187a8a8ca"
G718_BEHAVIOR_ZERO_SHA256 = "8e9ac420eed0566c9e68754cc15df1cd9ffab9035524c17b33036f7a1398f852"
G718_BEHAVIOR_TWO_SHA256 = "08d4123e25fb212447ec9069e50add4307649edab7aa3a2217b5d3567a5fbcbd"
G718_INITIAL_HIGH_LEVEL_SHA256 = "552cbcc40c68a5f57cf6df56b2eeead4154adf7824a16fcf4a16fde39fbf5843"
G718_COLLECTION_INPUT_SHA256 = "05756f740f923bded971b0da8c32fce8606a490bdcf361aed6dd5d7e348ceb02"
G719_V1_MANIFEST_SHA256 = "1453d62dc5f4ceef4a83e1ecdc6d17d3efad552dce4619c2cabf5613c521243f"
G719_V1_RESULT_SHA256 = "f2832097a6c64f9e506a3e7f8954971bef73b73fa0eb3bf01ce9da267b2b9b78"
G719_V1_CONFIG_SHA256 = "9c1ad5041a2e9821251bea8ca9178398aea70dcb76077d6e3ce587325ed10f84"
MANIFEST_SHA256 = "bd86c0c6ebd3597f1d4a2bae7b0a546f3e5dc4ff45676e4cc002f44c8448d423"


def validate_manifest(manifest: dict) -> dict:
    expected = {
        "schema_version", "stage", "source_contract", "source", "predecessor", "repair",
        "policy_observation_keys", "privileged_policy_blacklist", "protocol",
        "decision_rule", "required_reports", "gates", "model_load", "rl_update",
        "navigation_evaluation", "controller_comparison", "robot_execution", "g8_ready",
    }
    if set(manifest) != expected or manifest["schema_version"] != 1:
        raise ValueError("G7.19 v2 manifest schema changed")
    if manifest["stage"] != (
        "G7.19 fixed zero-one-two update causal stability audit, "
        "optimizer-contract repair v2"
    ):
        raise ValueError("G7.19 v2 stage changed")
    if manifest["source_contract"] != {
        "stream_commit": ORIGINAL_STREAM_COMMIT,
        "rlinf_commit": RLINF_COMMIT,
        "original_stream_only": True,
        "ttt_or_secondary_post_training": False,
    }:
        raise ValueError("G7.19 source contract changed")
    predecessor = manifest["predecessor"]
    expected_predecessor = {
        "accepted_local_commit": G718_LOCAL_COMMIT,
        "manifest_sha256": G718_MANIFEST_SHA256,
        "result_sha256": G718_RESULT_SHA256,
        "run_config_sha256": G718_CONFIG_SHA256,
        "collection_records_sha256": G718_COLLECTION_SHA256,
        "behavior_zero_update_sha256": G718_BEHAVIOR_ZERO_SHA256,
        "behavior_two_update_sha256": G718_BEHAVIOR_TWO_SHA256,
        "initial_high_level_sha256": G718_INITIAL_HIGH_LEVEL_SHA256,
        "collection_input_sha256": G718_COLLECTION_INPUT_SHA256,
    }
    if any(predecessor[key] != value for key, value in expected_predecessor.items()):
        raise ValueError("G7.19 predecessor anchor changed")
    if set(manifest["privileged_policy_blacklist"]) != PRIVILEGED_POLICY_KEYS:
        raise ValueError("G7.19 privileged-data blacklist changed")
    if manifest["repair"] != {
        "kind": "optimizer-contract-only",
        "v1_manifest_sha256": G719_V1_MANIFEST_SHA256,
        "v1_result_sha256": G719_V1_RESULT_SHA256,
        "v1_run_config_sha256": G719_V1_CONFIG_SHA256,
        "failed_v1_guard": "optimizer_initial_state_fixed",
        "reason": (
            "RLinf eagerly materializes AdamW slots during setup with zero moments "
            "and step one; v1 incorrectly required an empty state dictionary."
        ),
        "allowed_change": (
            "Replace only the initial optimizer-state guard with the exact live "
            "eager-zero contract."
        ),
        "forbidden_changes": [
            "data", "seed", "thresholds", "learning_rates", "update_rule",
            "update_count", "behavior_contract", "decision_rule",
        ],
    }:
        raise ValueError("G7.19 v2 repair boundary changed")
    assert_history_only_keys(manifest["policy_observation_keys"])
    protocol = manifest["protocol"]
    collection = protocol["collection_scene_specs"]
    behavior = protocol["behavior_scene_specs"]
    if (
        protocol["split"] != "train"
        or len(collection) != 4
        or len(behavior) != 4
        or set(row["scene"] for row in collection) & set(row["scene"] for row in behavior)
        or protocol["sampling"]["sample_calls_per_decision"] != 1
        or protocol["sampling"]["target_mode_scan"] is not False
        or protocol["sampling"]["mode_filter_or_resample"] is not False
        or protocol["sampling"]["recovery_override"] is not False
        or protocol["collection_contract"]["expected_rows"] != 12
        or protocol["collection_contract"]["expected_records_sha256"] != G718_COLLECTION_SHA256
        or protocol["collection_contract"]["expected_input_sha256"] != G718_COLLECTION_INPUT_SHA256
        or protocol["update_states"] != [0, 1, 2]
        or protocol["ppo_full_batch_updates"] != 2
        or protocol["behavior_metrics_are_descriptive_only"] is not True
        or protocol["behavior_based_arm_selection"] is not False
        or protocol["early_stop"] is not False
        or protocol["checkpoint_selection"] is not False
        or protocol["val_unseen_episode_count"] != 0
        or protocol["optimizer_initial_contract"] != {
            "optimizer_class": "AdamW",
            "state_entry_count": 19,
            "parameter_group_count": 2,
            "learning_rates": [1e-5, 1e-5],
            "state_keys": ["exp_avg", "exp_avg_sq", "step"],
            "moment_parameter_count": 1_003_592,
            "max_abs_exp_avg": 0.0,
            "max_abs_exp_avg_sq": 0.0,
            "step_values": [1.0],
        }
    ):
        raise ValueError("G7.19 protocol changed")
    if (
        manifest["decision_rule"]["behavior_outcomes_may_select_arm"] is not False
        or manifest["model_load"] is not True
        or manifest["rl_update"] is not True
        or manifest["navigation_evaluation"] is not False
        or manifest["controller_comparison"] is not False
        or manifest["robot_execution"] is not False
        or manifest["g8_ready"] is not False
    ):
        raise ValueError("G7.19 claim boundary changed")
    return protocol


@contextmanager
def preserve_random_state():
    """Make the inserted one-update behavior arm observational for update two."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        torch.cuda.set_rng_state_all(cuda_state)


def optimizer_state_contract(optimizer) -> dict:
    state_keys = set()
    step_values = set()
    exp_avg_numel = 0
    exp_avg_sq_numel = 0
    max_abs_exp_avg = 0.0
    max_abs_exp_avg_sq = 0.0
    for state in optimizer.state.values():
        state_keys.update(state)
        step = state.get("step")
        if not torch.is_tensor(step) or step.numel() != 1:
            raise RuntimeError("G7.19 v2 optimizer step slot is malformed")
        step_values.add(float(step.detach().float()))
        for name in ("exp_avg", "exp_avg_sq"):
            value = state.get(name)
            if not torch.is_tensor(value):
                raise RuntimeError(f"G7.19 v2 optimizer {name} slot is missing")
            tensor = value.detach().float()
            if name == "exp_avg":
                exp_avg_numel += value.numel()
                max_abs_exp_avg = max(
                    max_abs_exp_avg,
                    float(tensor.abs().max()) if value.numel() else 0.0,
                )
            else:
                exp_avg_sq_numel += value.numel()
                max_abs_exp_avg_sq = max(
                    max_abs_exp_avg_sq,
                    float(tensor.abs().max()) if value.numel() else 0.0,
                )
    moment_parameter_count = (
        exp_avg_numel if exp_avg_numel == exp_avg_sq_numel else -1
    )
    return {
        "optimizer_class": type(optimizer).__name__,
        "state_entry_count": len(optimizer.state),
        "parameter_group_count": len(optimizer.param_groups),
        "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
        "state_keys": sorted(state_keys),
        "moment_parameter_count": moment_parameter_count,
        "max_abs_exp_avg": max_abs_exp_avg,
        "max_abs_exp_avg_sq": max_abs_exp_avg_sq,
        "step_values": sorted(step_values),
    }


def replay_input_bundle(rows) -> dict:
    return {
        **{f"state_{index}": row["state"] for index, row in enumerate(rows)},
        **{f"topology_{index}": row["topology_features"] for index, row in enumerate(rows)},
        "modes": torch.tensor([row["policy_mode"] for row in rows], dtype=torch.int64),
        "waypoints": torch.tensor([row["waypoint"] for row in rows], dtype=torch.float32),
    }


def prepare_batch(manager, core, run_cfg, episodes) -> dict:
    rows = flat_rows(episodes)
    if len(rows) != run_cfg["protocol"]["collection_contract"]["expected_rows"]:
        raise RuntimeError("G7.19 collection row count differs from G7.18")
    device = model_device(core)
    indices = list(range(len(rows)))
    replay_inputs = replay_input_bundle(rows)
    input_sha = state_sha256(replay_inputs)
    if input_sha != G718_COLLECTION_INPUT_SHA256:
        raise RuntimeError("G7.19 collection input bundle differs from G7.18")
    with torch.no_grad():
        reference = replay_individual(manager, rows, indices, device)
        behavior_log_probs = torch.tensor(
            [row["behavior_log_prob"] for row in rows], dtype=torch.float32, device=device
        )
        pre_ratios = torch.exp(reference["log_probs"].float() - behavior_log_probs)
        reference_log_std = core.waypoint_log_std.detach().float().clamp(-3.0, -0.25).clone()
    pre_deviation = float((pre_ratios - 1.0).abs().max())
    if pre_deviation > run_cfg["protocol"]["preupdate_ratio_max_deviation"]:
        raise RuntimeError("G7.19 pre-update ratio differs from one")
    advantages_raw = torch.tensor([row["advantage"] for row in rows], dtype=torch.float32)
    advantage_std = advantages_raw.std(unbiased=False)
    if not bool(torch.isfinite(advantage_std)) or float(advantage_std) <= 1e-8:
        raise RuntimeError("G7.19 collection advantages are degenerate")
    return {
        "rows": rows,
        "indices": indices,
        "device": device,
        "replay_inputs": replay_inputs,
        "input_sha256": input_sha,
        "reference": reference,
        "reference_log_std": reference_log_std,
        "behavior_log_probs": behavior_log_probs,
        "advantages": ((advantages_raw - advantages_raw.mean()) / advantage_std).to(device),
        "returns": torch.tensor(
            [row["return"] for row in rows], dtype=torch.float32, device=device
        ),
        "preupdate_ratio_max_deviation": pre_deviation,
    }


def evaluate_state(manager, core, batch, initial_parameters, state_index: int) -> dict:
    with torch.no_grad():
        replay = replay_individual(
            manager, batch["rows"], batch["indices"], batch["device"]
        )
        ratios = torch.exp(
            replay["log_probs"].float() - batch["behavior_log_probs"]
        )
        current_log_std = core.waypoint_log_std.detach().float().clamp(-3.0, -0.25)
        combined, categorical, gaussian = structured_distribution_kl(
            batch["reference"], replay, batch["reference_log_std"], current_log_std
        )
    changes = parameter_changes(core, initial_parameters)
    return {
        "update_state": state_index,
        "actor_version": core.actor_version,
        "sampled_action_ratio_mean": float(ratios.mean()),
        "sampled_action_ratio_max_deviation": float((ratios - 1.0).abs().max()),
        "structured_distribution_kl": {
            "mean": float(combined.mean()),
            "max": float(combined.max()),
            "categorical_mean": float(categorical.mean()),
            "gaussian_mean": float(gaussian.mean()),
        },
        "parameter_changes_from_zero": changes,
        "collection_input_sha256": state_sha256(batch["replay_inputs"]),
    }


def one_update(manager, core, run_cfg, batch, update_index: int) -> dict:
    replay = replay_individual(
        manager, batch["rows"], batch["indices"], batch["device"]
    )
    ratios = torch.exp(replay["log_probs"].float() - batch["behavior_log_probs"])
    clipped = ratios.clamp(1.0 - run_cfg["clip_ratio"], 1.0 + run_cfg["clip_ratio"])
    actor_loss = -torch.minimum(
        ratios * batch["advantages"], clipped * batch["advantages"]
    ).mean()
    value_loss = 0.5 * (replay["values"].float() - batch["returns"]).square().mean()
    entropy = replay["entropy"].float().mean()
    loss = actor_loss + run_cfg["value_coef"] * value_loss - run_cfg["entropy_coef"] * entropy
    if not all(bool(torch.isfinite(value)) for value in (loss, actor_loss, value_loss, entropy)):
        raise RuntimeError("G7.19 update loss is nonfinite")
    manager.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    audit = gradient_audit(core)
    if (
        not audit["finite"]
        or audit["group_norm_sums"]["high_level_policy"] <= 0
        or audit["group_norm_sums"]["value_head"] <= 0
        or audit["frozen_with_grad"]
    ):
        raise RuntimeError("G7.19 gradient isolation failed")
    grad_norm, learning_rates = manager.optimizer_step()
    core.mark_updated()
    return {
        "update_index": update_index,
        "ratio_before_update_mean": float(ratios.detach().mean()),
        "ratio_before_update_max_deviation": float((ratios.detach() - 1.0).abs().max()),
        "loss": float(loss.detach()),
        "actor_loss": float(actor_loss.detach()),
        "value_loss": float(value_loss.detach()),
        "entropy": float(entropy.detach()),
        "grad_norm": float(torch.as_tensor(grad_norm)),
        "learning_rates": [float(value) for value in learning_rates],
        "gradient_audit": audit,
        "actor_version_after": core.actor_version,
    }


def state_is_bounded(state: dict, protocol: dict) -> bool:
    return (
        state["sampled_action_ratio_max_deviation"]
        <= protocol["postupdate_ratio_max_deviation"]
        and state["structured_distribution_kl"]["max"]
        <= protocol["postupdate_distribution_kl_max"]
        and (
            state["update_state"] == 0
            or max(
                state["sampled_action_ratio_max_deviation"],
                state["structured_distribution_kl"]["mean"],
            )
            >= protocol["postupdate_functional_change_min"]
        )
    )


def matched_three(zero, one, two) -> dict:
    if not len(zero) == len(one) == len(two):
        raise RuntimeError("G7.19 behavior episode count changed")
    pairs = []
    for rows in zip(zero, one, two):
        identities = {(row["scene"], row["episode_id"]) for row in rows}
        initial_hashes = {row["initial_observation_sha256"] for row in rows}
        schedules = {tuple(row["sampling_seed_schedule"]) for row in rows}
        summaries = [aggregate_behavior([row]) for row in rows]
        pairs.append({
            "scene": rows[0]["scene"],
            "episode_id": rows[0]["episode_id"],
            "identity_match": len(identities) == 1,
            "initial_observation_match": len(initial_hashes) == 1,
            "seed_schedule_match": len(schedules) == 1,
            "zero_update": summaries[0],
            "one_update": summaries[1],
            "two_update": summaries[2],
        })
    return {
        "all_identities_match": all(row["identity_match"] for row in pairs),
        "all_initial_observations_match": all(
            row["initial_observation_match"] for row in pairs
        ),
        "all_seed_schedules_match": all(row["seed_schedule_match"] for row in pairs),
        "pairs": pairs,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--g718-result", type=Path, required=True)
    parser.add_argument("--g718-run-config", type=Path, required=True)
    parser.add_argument("--g718-collection", type=Path, required=True)
    parser.add_argument("--g719-v1-result", type=Path, required=True)
    parser.add_argument("--g719-v1-run-config", type=Path, required=True)
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
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--master-port", type=int, default=29609)
    parser.add_argument("--context-tokens", type=int, default=4096)
    parser.add_argument("--head-hidden", type=int, default=256)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    started = time.time()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = args.manifest.resolve(strict=True)
    if sha256_file(manifest_path) != MANIFEST_SHA256:
        raise RuntimeError("G7.19 v2 frozen manifest hash changed")
    manifest = json.loads(manifest_path.read_text())
    protocol = validate_manifest(manifest)
    predecessor_paths = {
        "result": args.g718_result.resolve(strict=True),
        "run_config": args.g718_run_config.resolve(strict=True),
        "collection": args.g718_collection.resolve(strict=True),
    }
    predecessor_hashes = {name: sha256_file(path) for name, path in predecessor_paths.items()}
    if predecessor_hashes != {
        "result": G718_RESULT_SHA256,
        "run_config": G718_CONFIG_SHA256,
        "collection": G718_COLLECTION_SHA256,
    }:
        raise RuntimeError("G7.19 G7.18 artifact anchor changed")
    g718_result = json.loads(predecessor_paths["result"].read_text())
    repair_paths = {
        "v1_result": args.g719_v1_result.resolve(strict=True),
        "v1_run_config": args.g719_v1_run_config.resolve(strict=True),
    }
    repair_hashes = {name: sha256_file(path) for name, path in repair_paths.items()}
    if repair_hashes != {
        "v1_result": G719_V1_RESULT_SHA256,
        "v1_run_config": G719_V1_CONFIG_SHA256,
    }:
        raise RuntimeError("G7.19 v2 v1 failure anchor changed")
    v1_result = json.loads(repair_paths["v1_result"].read_text())
    if (
        v1_result["all_protocol_guards_passed"] is not False
        or v1_result["causal_outcome"] != "protocol_invalid"
        or [
            name for name, passed in v1_result["protocol_guards"].items() if not passed
        ] != ["optimizer_initial_state_fixed"]
    ):
        raise RuntimeError("G7.19 v2 v1 failure is not the preregistered repair case")

    path_names = (
        "rlinf_root", "stream_source", "base_model", "tokenizer", "vision_tower",
        "asset_root", "habitat_python", "habitat_server", "habitat_sim_python",
        "habitat_env_lib", "egl_vendor_json",
    )
    paths = {
        name: str(Path(getattr(args, name)).expanduser().resolve(strict=True))
        for name in path_names
    }
    source_heads = {
        "stream": git_head(Path(paths["stream_source"])),
        "rlinf": git_head(Path(paths["rlinf_root"])),
    }
    if source_heads != {"stream": ORIGINAL_STREAM_COMMIT, "rlinf": RLINF_COMMIT}:
        raise RuntimeError("G7.19 source commit changed")
    datasets = dataset_hashes(Path(paths["asset_root"]), manifest)
    script_root = Path(__file__).resolve().parent
    implementation_names = (
        "run_streamvln_stability_g719_v2.py", "run_streamvln_learning_g718.py",
        "run_streamvln_plumbing_g717.py", "streamvln_topological_policy.py",
        "streamvln_recovery_policy.py", "streamvln_executor_contract.py",
        "streamvln_topology.py", "streamvln_chunk_ppo.py",
    )
    implementation_paths = {name: script_root / name for name in implementation_names}
    implementation_paths.update({
        "rlinf/envs/streamvln/streamvln_env.py": (
            Path(paths["rlinf_root"]) / "rlinf" / "envs" / "streamvln" / "streamvln_env.py"
        ),
        "rlinf/envs/streamvln/streamvln_executor_contract.py": (
            Path(paths["rlinf_root"]) / "rlinf" / "envs" / "streamvln"
            / "streamvln_executor_contract.py"
        ),
        "rlinf/models/embodiment/streamvln_ppo_policy.py": (
            Path(paths["rlinf_root"]) / "rlinf" / "models" / "embodiment"
            / "streamvln_ppo_policy.py"
        ),
    })
    implementation = {
        name: sha256_file(path.resolve(strict=True)) for name, path in implementation_paths.items()
    }
    all_specs = protocol["collection_scene_specs"] + protocol["behavior_scene_specs"]
    run_cfg = {
        **paths,
        "output_dir": str(output),
        "stream_commit": ORIGINAL_STREAM_COMMIT,
        "rlinf_commit": RLINF_COMMIT,
        "manifest": {"source": manifest["source"], "train": all_specs, "val_unseen": []},
        "protocol": protocol,
        "policy_observation_keys": manifest["policy_observation_keys"],
        "context_tokens": args.context_tokens,
        "head_hidden": args.head_hidden,
        "seed": args.seed,
        "master_port": args.master_port,
        "learning_rate": protocol["learning_rate"],
        "value_learning_rate": protocol["value_learning_rate"],
        "gamma": protocol["gamma"],
        "gae_lambda": protocol["gae_lambda"],
        "clip_ratio": protocol["clip_ratio"],
        "value_coef": protocol["value_coef"],
        "entropy_coef": protocol["entropy_coef"],
    }
    config = {
        "kind": "streamvln-g7.19-v2-optimizer-contract-repair-audit",
        "manifest": str(manifest_path),
        "manifest_sha256": MANIFEST_SHA256,
        "predecessor_paths": {name: str(path) for name, path in predecessor_paths.items()},
        "predecessor_hashes": predecessor_hashes,
        "repair_paths": {name: str(path) for name, path in repair_paths.items()},
        "repair_hashes": repair_hashes,
        "source_heads": source_heads,
        "datasets": datasets,
        "implementation_sha256": implementation,
        "model_input_inventory": {
            "base_model": small_file_inventory(Path(paths["base_model"])),
            "tokenizer": small_file_inventory(Path(paths["tokenizer"])),
            "vision_tower": small_file_inventory(Path(paths["vision_tower"])),
        },
        "run": run_cfg,
        "model_load_count": 1,
        "optimizer_update_limit": 2,
        "checkpoint_selection": False,
        "early_stop": False,
        "behavior_based_arm_selection": False,
        "val_unseen_episode_count": 0,
    }
    config_path = output / "run_config.json"
    write_json(config_path, config)

    manager = core = None
    try:
        manager, core, trainable_manifest = setup_manager(run_cfg, total_steps=2)
        optimizer_scope = optimizer_parameter_scope(manager, core)
        optimizer_initial_contract = optimizer_state_contract(manager.optimizer)
        frozen_zero, sentinel_names = frozen_parameter_sentinels(core)
        frozen_hash_zero = state_sha256(frozen_zero)
        initial_parameters = parameter_snapshot(core)
        initial_parameter_hash = state_sha256(initial_parameters)
        if initial_parameter_hash != G718_INITIAL_HIGH_LEVEL_SHA256:
            raise RuntimeError("G7.19 initial high-level parameters differ from G7.18")

        behavior_zero = collect_phase(
            manager, core, run_cfg, protocol["behavior_scene_specs"],
            phase="behavior_control", seed_phase="behavior",
        )
        collection = collect_phase(
            manager, core, run_cfg, protocol["collection_scene_specs"],
            phase="collection", seed_phase="collection",
        )
        collection_path = output / "collection_records.jsonl"
        behavior_zero_path = output / "behavior_zero_update.jsonl"
        write_jsonl(collection_path, collection)
        write_jsonl(behavior_zero_path, behavior_zero)
        if sha256_file(collection_path) != G718_COLLECTION_SHA256:
            raise RuntimeError("G7.19 serialized collection differs from G7.18")
        if sha256_file(behavior_zero_path) != G718_BEHAVIOR_ZERO_SHA256:
            raise RuntimeError("G7.19 zero-update behavior differs from G7.18")
        if state_sha256(parameter_snapshot(core)) != initial_parameter_hash or core.actor_version != 0:
            raise RuntimeError("G7.19 zero-update control changed the policy")

        batch = prepare_batch(manager, core, run_cfg, collection)
        state_zero = evaluate_state(manager, core, batch, initial_parameters, 0)
        update_one = one_update(manager, core, run_cfg, batch, 0)
        state_one = evaluate_state(manager, core, batch, initial_parameters, 1)
        parameters_after_one = state_sha256(parameter_snapshot(core))
        frozen_one, names_one = frozen_parameter_sentinels(core)
        with preserve_random_state():
            behavior_one = collect_phase(
                manager, core, run_cfg, protocol["behavior_scene_specs"],
                phase="behavior_one_update", seed_phase="behavior",
            )
        if state_sha256(parameter_snapshot(core)) != parameters_after_one:
            raise RuntimeError("G7.19 one-update behavior evaluation changed parameters")

        update_two = one_update(manager, core, run_cfg, batch, 1)
        state_two = evaluate_state(manager, core, batch, initial_parameters, 2)
        frozen_two, names_two = frozen_parameter_sentinels(core)
        behavior_two = collect_phase(
            manager, core, run_cfg, protocol["behavior_scene_specs"],
            phase="behavior_updated", seed_phase="behavior",
        )
        behavior_one_path = output / "behavior_one_update.jsonl"
        behavior_two_path = output / "behavior_two_update.jsonl"
        paired_path = output / "paired_behavior.json"
        write_jsonl(behavior_one_path, behavior_one)
        write_jsonl(behavior_two_path, behavior_two)
        paired = matched_three(behavior_zero, behavior_one, behavior_two)
        write_json(paired_path, paired)

        zero_summary = aggregate_behavior(behavior_zero)
        one_summary = aggregate_behavior(behavior_one)
        two_summary = aggregate_behavior(behavior_two)
        updates = [update_one, update_two]
        states = [state_zero, state_one, state_two]
        one_bounded = state_is_bounded(state_one, protocol)
        two_bounded = state_is_bounded(state_two, protocol)
        if one_bounded and not two_bounded:
            causal_outcome = "second_update_trigger_supported"
        elif not one_bounded:
            causal_outcome = "one_update_fails_complete_guard"
        else:
            causal_outcome = "g718_two_update_violation_not_reproduced"
        state_two_reproduces_g718 = (
            abs(
                state_two["sampled_action_ratio_max_deviation"]
                - g718_result["training"]["postupdate_ratio_max_deviation"]
            ) <= 1e-6
            and abs(
                state_two["structured_distribution_kl"]["max"]
                - g718_result["training"]["postupdate_distribution_kl"]["max"]
            ) <= 1e-6
            and sha256_file(behavior_two_path) == G718_BEHAVIOR_TWO_SHA256
        )
        all_rows = flat_rows(collection + behavior_zero + behavior_one + behavior_two)
        updates_valid = all(
            all(math.isfinite(update[key]) for key in (
                "loss", "actor_loss", "value_loss", "entropy", "grad_norm"
            ))
            and update["gradient_audit"]["finite"]
            and update["gradient_audit"]["group_norm_sums"]["high_level_policy"] > 0
            and update["gradient_audit"]["group_norm_sums"]["value_head"] > 0
            and not update["gradient_audit"]["frozen_with_grad"]
            for update in updates
        )
        frozen_unchanged = (
            sentinel_names == names_one == names_two
            and frozen_hash_zero == state_sha256(frozen_one) == state_sha256(frozen_two)
        )
        guards = {
            "source_and_predecessor_hashes": source_heads
            == {"stream": ORIGINAL_STREAM_COMMIT, "rlinf": RLINF_COMMIT}
            and predecessor_hashes["result"] == G718_RESULT_SHA256,
            "v1_protocol_failure_anchored": repair_hashes
            == {
                "v1_result": G719_V1_RESULT_SHA256,
                "v1_run_config": G719_V1_CONFIG_SHA256,
            },
            "dataset_hashes": all(
                datasets[split]["sha256"] == manifest["source"][f"{split}_sha256"]
                for split in datasets
            ),
            "exact_g718_initial_parameter_hash": initial_parameter_hash
            == G718_INITIAL_HIGH_LEVEL_SHA256,
            "exact_g718_collection_records_and_inputs": sha256_file(collection_path)
            == G718_COLLECTION_SHA256
            and batch["input_sha256"] == G718_COLLECTION_INPUT_SHA256,
            "zero_update_control_unchanged": state_zero["actor_version"] == 0
            and state_zero["parameter_changes_from_zero"]["changed_tensor_count"] == 0,
            "fixed_update_states_zero_one_two": [
                state["actor_version"] for state in states
            ] == [0, 1, 2]
            and [update["actor_version_after"] for update in updates] == [1, 2],
            "matched_behavior_contract_all_states": paired["all_identities_match"]
            and paired["all_initial_observations_match"]
            and paired["all_seed_schedules_match"],
            "causal_feedback_reward_bootstrap_and_gae": all(
                set(row["executor_feedback"]) == FEEDBACK_KEYS
                and row["reward_accounting_error"] <= 1e-6
                and math.isfinite(row["advantage"])
                and math.isfinite(row["return"])
                for row in all_rows
            ),
            "preupdate_ratio_near_one": batch["preupdate_ratio_max_deviation"]
            <= protocol["preupdate_ratio_max_deviation"],
            "per_update_finite_nonzero_actor_and_value_gradients": updates_valid,
            "per_state_ratio_and_structured_kl_reported": all(
                math.isfinite(state["sampled_action_ratio_max_deviation"])
                and math.isfinite(state["structured_distribution_kl"]["max"])
                for state in states
            ),
            "only_high_level_parameters_change": optimizer_scope["optimizer_equals_trainable"]
            and optimizer_scope["optimizer_frozen_overlap_count"] == 0
            and state_one["parameter_changes_from_zero"]["changed_tensor_count"] > 0
            and state_two["parameter_changes_from_zero"]["changed_tensor_count"] > 0,
            "frozen_parameters_and_collection_inputs_unchanged": frozen_unchanged
            and all(
                state["collection_input_sha256"] == G718_COLLECTION_INPUT_SHA256
                for state in states
            ),
            "exact_initial_optimizer_eager_zero_contract": optimizer_initial_contract
            == protocol["optimizer_initial_contract"],
            "g718_two_update_reproduced": state_two_reproduces_g718,
            "behavior_metrics_nonselective": protocol["behavior_metrics_are_descriptive_only"]
            and protocol["behavior_based_arm_selection"] is False,
            "no_early_stop_or_checkpoint_selection": protocol["early_stop"] is False
            and protocol["checkpoint_selection"] is False,
            "zero_val_unseen": config["val_unseen_episode_count"] == 0,
        }
        protocol_valid = all(guards.values())
        result = {
            "kind": config["kind"],
            "all_protocol_guards_passed": protocol_valid,
            "causal_outcome": causal_outcome if protocol_valid else "protocol_invalid",
            "second_update_trigger_supported": protocol_valid
            and causal_outcome == "second_update_trigger_supported",
            "one_update_bounded": one_bounded,
            "two_update_bounded": two_bounded,
            "navigation_or_generalization_claim": False,
            "source_contract": manifest["source_contract"],
            "trainable_manifest": trainable_manifest,
            "counts": {
                "model_loads": 1,
                "collection_episodes": len(collection),
                "collection_rows": len(batch["rows"]),
                "behavior_episodes_per_state": len(behavior_zero),
                "optimizer_updates": len(updates),
                "val_unseen_episodes": 0,
            },
            "initial_high_level_sha256": initial_parameter_hash,
            "optimizer_initial_contract": optimizer_initial_contract,
            "repair_anchor": {
                "manifest": manifest["repair"],
                "hashes": repair_hashes,
            },
            "optimizer_scope": optimizer_scope,
            "preupdate_ratio_max_deviation": batch["preupdate_ratio_max_deviation"],
            "states": states,
            "updates": updates,
            "behavior_zero_update": zero_summary,
            "behavior_one_update": one_summary,
            "behavior_two_update": two_summary,
            "paired_behavior": paired,
            "frozen_sentinel_sha256": {
                "zero_update": frozen_hash_zero,
                "one_update": state_sha256(frozen_one),
                "two_update": state_sha256(frozen_two),
            },
            "protocol_guards": guards,
            "g8_ready": False,
            "permitted_next_stage": (
                "independent result-to-claim on fixed causal stability outcome"
                if protocol_valid else "G7.19 protocol repair only"
            ),
            "elapsed_seconds": time.time() - started,
            "evidence": {
                "manifest_sha256": MANIFEST_SHA256,
                "run_config_sha256": sha256_file(config_path),
                "collection_records_sha256": sha256_file(collection_path),
                "behavior_zero_update_sha256": sha256_file(behavior_zero_path),
                "behavior_one_update_sha256": sha256_file(behavior_one_path),
                "behavior_two_update_sha256": sha256_file(behavior_two_path),
                "paired_behavior_sha256": sha256_file(paired_path),
            },
        }
        result_path = output / "result.json"
        write_json(result_path, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        if not protocol_valid:
            raise SystemExit(2)
    finally:
        if manager is not None:
            cleanup_manager(manager)


if __name__ == "__main__":
    main()
