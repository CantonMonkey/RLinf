"""Adjudicate the bounded train-only A1 BC versus BC+PPO pilot.

This file is intentionally a thin orchestrator.  The companion canonical live
provider owns clean-e48 StreamVLN/Habitat collection, while this process freezes
the episode identities, checks the source boundary, applies the BC competence
gate, and refuses PPO evidence that did not traverse ``manager.model`` under
RLinf/FSDP.  The separation keeps the accepted P8 provider and the v0.1 stack
core unchanged.

No result is synthesized here.  ``--stage pilot`` requires an explicit live
provider (``module:attribute``); the canonical factory is
``streamvln_a1_live_provider:create_provider``.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib
import json
import math
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence


SCHEMA_NAME = "streamvln_a1_contextual_hybrid_manifest"
SCHEMA_VERSION = 1
RESULT_SCHEMA_NAME = "streamvln_a1_contextual_hybrid_result"
RESULT_SCHEMA_VERSION = 1
EXPECTED_EPISODES = 8
SHA256_LENGTH = 64
STOP_SOURCES = frozenset({"none", "policy", "official_shadow"})
SHADOW_STOP_PHASES = frozenset({"none", "predecision", "mid_macro"})


class ProtocolError(RuntimeError):
    """Raised when A1 evidence escapes its frozen protocol."""


class A1Provider(Protocol):
    def execute(
        self,
        *,
        manifest: Mapping,
        contract: Mapping,
        args: argparse.Namespace,
    ) -> Mapping:
        ...


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _json_hash(value: object) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_sha256(path: Path) -> str:
    """Hash one checkpoint file or a directory including relative names."""

    path = path.expanduser().resolve(strict=True)
    if path.is_file():
        return _file_sha256(path)
    if not path.is_dir():
        raise ProtocolError(f"checkpoint artifact is not a file or directory: {path}")
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ProtocolError(f"checkpoint directory is empty: {path}")
    for item in files:
        if item.is_symlink():
            raise ProtocolError("checkpoint artifact may not contain symbolic links")
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_file_sha256(item)))
    return digest.hexdigest()


def _sha(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProtocolError(f"{name} must be a lowercase SHA-256")
    return value


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ProtocolError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value: object, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ProtocolError(f"{name} is outside the finite protocol range")
    return result


def _boolean(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ProtocolError(f"{name} must be boolean")
    return value


def _identity(value: object, name: str) -> dict:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{name} must be an object")
    index = _integer(value.get("index"), f"{name}.index")
    episode_id = value.get("episode_id")
    scene = value.get("scene")
    if not isinstance(episode_id, str) or not episode_id:
        raise ProtocolError(f"{name}.episode_id must be nonempty")
    if not isinstance(scene, str) or not scene:
        raise ProtocolError(f"{name}.scene must be nonempty")
    return {"index": index, "episode_id": episode_id, "scene": scene}


def _identities(value: object, name: str) -> tuple[dict, ...]:
    if not isinstance(value, list) or len(value) != EXPECTED_EPISODES:
        raise ProtocolError(f"{name} must contain exactly eight episodes")
    rows = tuple(_identity(row, f"{name}[{index}]") for index, row in enumerate(value))
    keys = {(row["index"], row["episode_id"], row["scene"]) for row in rows}
    if len(keys) != len(rows) or len({row["scene"] for row in rows}) != len(rows):
        raise ProtocolError(f"{name} identities/scenes must be unique")
    return rows


def validate_manifest(manifest: object) -> dict:
    """Validate the frozen A1 contract without importing Torch or Habitat."""

    if not isinstance(manifest, Mapping):
        raise ProtocolError("A1 manifest root must be an object")
    if (
        manifest.get("schema_name") != SCHEMA_NAME
        or manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("stage") != "A1 train-only contextual topology-waypoint pilot"
    ):
        raise ProtocolError("unsupported A1 manifest")

    source = manifest.get("source_contract")
    if not isinstance(source, Mapping):
        raise ProtocolError("A1 source contract is missing")
    if source.get("original_streamvln_only") is not True:
        raise ProtocolError("A1 requires original StreamVLN")
    if source.get("ttt_or_secondary_post_training") is not False:
        raise ProtocolError("A1 forbids TTT and secondary post-training weights")
    if source.get("streamvln_commit") != "e48f6ff7e9201d93aae003e8f64b04c00cec13bc":
        raise ProtocolError("A1 clean-e48 source revision drifted")
    for key in ("readiness_manifest", "feature_v2_manifest"):
        record = source.get(key)
        if not isinstance(record, Mapping) or not isinstance(record.get("relative_path"), str):
            raise ProtocolError(f"A1 {key} lineage is missing")
        _sha(record.get("raw_sha256"), f"source_contract.{key}.raw_sha256")

    dataset = manifest.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ProtocolError("A1 dataset contract is missing")
    if dataset.get("allowed_splits") != ["train"]:
        raise ProtocolError("A1 permits only the literal train split")
    if dataset.get("val_seen_read") is not False or dataset.get("val_unseen_read") is not False:
        raise ProtocolError("A1 records validation-split access")
    _sha(dataset.get("train_sha256"), "dataset.train_sha256")
    episodes = dataset.get("episodes")
    if not isinstance(episodes, Mapping):
        raise ProtocolError("A1 episode contract is missing")
    train = _identities(episodes.get("train"), "dataset.episodes.train")
    development = _identities(
        episodes.get("train_development"), "dataset.episodes.train_development"
    )
    if {row["scene"] for row in train} & {row["scene"] for row in development}:
        raise ProtocolError("A1 train and development scenes must be disjoint")

    prior = manifest.get("frozen_prior")
    if not isinstance(prior, Mapping):
        raise ProtocolError("A1 frozen prior contract is missing")
    if (
        prior.get("provider") != "StreamVLNContextualFeatureProvider"
        or prior.get("feature_dim") != 3604
        or prior.get("schema_version") != 2
        or prior.get("frozen_detached") is not True
    ):
        raise ProtocolError("A1 Feature-v2 prior contract drifted")
    _sha(prior.get("schema_sha256"), "frozen_prior.schema_sha256")

    action = manifest.get("action_contract")
    if not isinstance(action, Mapping):
        raise ProtocolError("A1 action contract is missing")
    required_action = {
        "distribution": "masked_categorical_plus_conditional_gaussian_residual",
        "stop_logprob": "categorical_only",
        "non_stop_logprob": "categorical_plus_raw_residual_gaussian",
        "official_shadow_stop": "force_stop_at_next_physical_boundary",
        "external_stop_actor_credit": False,
    }
    if any(action.get(key) != value for key, value in required_action.items()):
        raise ProtocolError("A1 hybrid action or shadow STOP contract drifted")

    architecture = manifest.get("architecture")
    if not isinstance(architecture, Mapping) or architecture != {
        "hidden_dim": 128,
        "topology_max_nodes": 161,
        "topology_model_dim": 32,
        "max_candidates": 5,
        "residual_limit_m": 0.2,
        "initial_log_std": -0.5,
    }:
        raise ProtocolError("A1 contextual head architecture drifted")

    budget = manifest.get("budget")
    if not isinstance(budget, Mapping):
        raise ProtocolError("A1 budget is missing")
    expected_budget = {
        "seed": 2026093001,
        "bc_supervision_steps": 1000,
        "bc_expert_warmup_steps": 500,
        "bc_dagger_steps": 500,
        "bc_unique_identity_count": 8,
        "bc_traversal_count": 16,
        "bc_collection_passes": ["expert", "dagger"],
        "max_primitives_per_episode": 160,
        "ppo_episode_count": 8,
        "ppo_update_count": 8,
        "causal_replay_block_primitives_max": 32,
        "bc_batch_size": 1,
        "bc_learning_rate": 0.0001,
        "bc_weight_decay": 0.0,
        "bc_waypoint_coef": 2.0,
    }
    if any(budget.get(key) != value for key, value in expected_budget.items()):
        raise ProtocolError("A1 fixed budget drifted")

    bc_objective = manifest.get("bc_objective")
    if bc_objective != {
        "route_term": "candidate_cross_entropy",
        "continuous_terms": [
            "raw_diagonal_gaussian_negative_log_likelihood",
            "metric_waypoint_smooth_l1",
        ],
        "total": "route_term_plus_waypoint_coef_times_sum_of_continuous_terms",
        "waypoint_coef": 2.0,
        "stop_masks_all_continuous_terms": True,
    }:
        raise ProtocolError("A1 BC objective drifted")
    reward = manifest.get("reward")
    if reward != {
        "formula": (
            "delta_geodesic_plus_success_minus_step_minus_collision_minus_"
            "failed_stop_minus_timeout"
        ),
        "delta_geodesic_coef": 1.0,
        "success_coef": 10.0,
        "step_coef": -0.01,
        "collision_coef": -0.2,
        "failed_stop_coef": -1.0,
        "timeout_coef": -1.0,
        "simulator_truth_is_reward_and_metrics_only": True,
    }:
        raise ProtocolError("A1 train reward drifted")
    ppo = manifest.get("ppo")
    if ppo != {
        "clip_ratio": 0.2,
        "gamma_per_primitive": 0.99,
        "gae_lambda": 0.95,
        "value_coef": 0.5,
        "entropy_coef": 0.01,
        "bc_reference_kl_coef": 0.1,
        "bc_reference_kl_max": 0.1,
        "updates_per_episode": 1,
        "final_checkpoint_update": 8,
    }:
        raise ProtocolError("A1 PPO objective or hyperparameters drifted")

    gate_a = manifest.get("gate_a")
    gate_b = manifest.get("gate_b")
    if not isinstance(gate_a, Mapping) or not isinstance(gate_b, Mapping):
        raise ProtocolError("A1 gates are missing")
    expected_a = {
        "success_count_min": 2,
        "successful_stop_count_min": 2,
        "spl_min_exclusive": 0.0,
        "mean_distance_reduction_m_min": 1.0,
        "timeout_count_max": 6,
        "waypoint_episode_count_min": 4,
        "forward_primitive_fraction_min": 0.1,
        "left_primitive_fraction_max": 0.8,
        "right_primitive_fraction_max": 0.8,
        "candidate_category_count_min": 2,
        "primitive_category_count_min": 2,
    }
    expected_b = {
        "success_delta_min": 1,
        "spl_delta_min": 0.0,
        "mean_navigation_error_improvement_m_min": 0.5,
        "timeout_delta_max": 0,
        "changed_executed_trace_count_min": 2,
        "forward_primitive_fraction_min": 0.1,
        "left_primitive_fraction_max": 0.8,
        "right_primitive_fraction_max": 0.8,
    }
    if any(gate_a.get(key) != value for key, value in expected_a.items()):
        raise ProtocolError("A1 BC competence gate drifted")
    if any(gate_b.get(key) != value for key, value in expected_b.items()):
        raise ProtocolError("A1 causal comparison gate drifted")

    training = manifest.get("training_guard")
    if not isinstance(training, Mapping):
        raise ProtocolError("A1 training guard is missing")
    if (
        training.get("ppo_root_forward") != "manager.model"
        or training.get("ppo_fsdp_required") is not True
        or training.get("pre_update_ratio_max_abs_deviation") != 0.001
        or training.get("bc_reference_kl_max") != 0.1
        or training.get("same_bc_initialization") is not True
    ):
        raise ProtocolError("A1 RLinf/PPO guard drifted")
    required_groups = training.get("required_trainable_groups")
    if not isinstance(required_groups, list) or len(required_groups) != len(set(required_groups)):
        raise ProtocolError("A1 trainable group declaration is invalid")

    claim = manifest.get("claim_boundary")
    if not isinstance(claim, Mapping) or any(
        claim.get(key) is not expected
        for key, expected in {
            "train_only": True,
            "single_seed_pilot": True,
            "paper_result": False,
            "cross_scene_generalization": False,
            "g9": False,
            "remote_push": False,
        }.items()
    ):
        raise ProtocolError("A1 claim boundary drifted")

    return {
        "train": train,
        "development": development,
        "required_trainable_groups": tuple(required_groups),
    }


def validate_lineage(manifest: Mapping, contract: Mapping, *, root: Path) -> dict:
    """Bind A1 identities and Feature-v2 schema to accepted P8 manifests."""

    source = manifest["source_contract"]
    loaded = {}
    for key in ("readiness_manifest", "feature_v2_manifest"):
        record = source[key]
        path = (root / record["relative_path"]).resolve(strict=True)
        observed = _file_sha256(path)
        if observed != record["raw_sha256"]:
            raise ProtocolError(f"A1 {key} bytes drifted")
        loaded[key] = json.loads(path.read_text(encoding="utf-8"))

    readiness = loaded["readiness_manifest"]
    readiness_episodes = readiness.get("dataset", {}).get("episodes", {})
    expected_train = tuple(
        _identity(row, f"readiness.expert_train[{index}]")
        for index, row in enumerate(readiness_episodes.get("expert_train", ()))
    )
    expected_dev = tuple(
        _identity(row, f"readiness.train_development[{index}]")
        for index, row in enumerate(readiness_episodes.get("train_development", ()))
    )
    if contract["train"] != expected_train or contract["development"] != expected_dev:
        raise ProtocolError("A1 episode identities differ from P8 readiness")
    if readiness.get("source_contract", {}).get("ttt_or_secondary_post_training") is not False:
        raise ProtocolError("P8 readiness lineage permits secondary weights")

    feature_v2 = loaded["feature_v2_manifest"]
    architecture = feature_v2.get("architecture", {})
    prior = manifest["frozen_prior"]
    if (
        architecture.get("feature_v2_dim") != prior["feature_dim"]
        or architecture.get("feature_v2_schema_version") != prior["schema_version"]
        or architecture.get("feature_v2_schema_sha256") != prior["schema_sha256"]
    ):
        raise ProtocolError("A1 Feature-v2 lineage differs from accepted P8 schema")
    return {
        "readiness_manifest_sha256": source["readiness_manifest"]["raw_sha256"],
        "feature_v2_manifest_sha256": source["feature_v2_manifest"]["raw_sha256"],
        "episode_identity_sha256": _json_hash(
            {"train": contract["train"], "development": contract["development"]}
        ),
    }


def _count_mapping(value: object, name: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{name} must be an object")
    result = {}
    for key, count in value.items():
        if not isinstance(key, str) or not key:
            raise ProtocolError(f"{name} keys must be nonempty strings")
        result[key] = _integer(count, f"{name}.{key}")
    return result


def _decision_receipts(value: object, name: str) -> tuple[dict, ...]:
    """Validate the per-decision external-STOP and actor-credit boundary."""

    if not isinstance(value, list) or not value:
        raise ProtocolError(f"{name} must contain per-decision receipts")
    rows = []
    previous_end = 0
    for position, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise ProtocolError(f"{name}[{position}] must be an object")
        decision_index = _integer(
            raw.get("decision_index"), f"{name}[{position}].decision_index"
        )
        physical_step = _integer(
            raw.get("physical_step_start"),
            f"{name}[{position}].physical_step_start",
        )
        proposed = _integer(
            raw.get("proposed_primitive_count"),
            f"{name}[{position}].proposed_primitive_count",
            minimum=1,
        )
        executed = _integer(
            raw.get("executed_primitive_count"),
            f"{name}[{position}].executed_primitive_count",
            minimum=1,
        )
        if proposed > 4 or executed > 4:
            raise ProtocolError(f"{name}[{position}] exceeds the fixed macro width")
        phase = raw.get("official_shadow_stop_phase")
        if phase not in SHADOW_STOP_PHASES:
            raise ProtocolError(f"{name}[{position}] has an invalid shadow STOP phase")
        sampled = _boolean(raw.get("actor_sampled"), f"{name}[{position}].actor_sampled")
        credit = _boolean(
            raw.get("actor_credit_eligible"),
            f"{name}[{position}].actor_credit_eligible",
        )
        topology_applied = _boolean(
            raw.get("topology_receipt_applied"),
            f"{name}[{position}].topology_receipt_applied",
        )
        if decision_index != position or physical_step != previous_end:
            raise ProtocolError(f"{name} decision/physical-step sequence is not contiguous")
        if credit and not sampled:
            raise ProtocolError(f"{name}[{position}] credits an unsampled actor action")
        if phase == "predecision":
            if sampled or credit or topology_applied or proposed != 1 or executed != 1:
                raise ProtocolError(
                    f"{name}[{position}] predecision STOP reached actor/topology credit"
                )
        elif phase == "mid_macro":
            if credit or topology_applied or not sampled:
                raise ProtocolError(
                    f"{name}[{position}] mid-macro STOP retained actor credit"
                )
        elif sampled != credit or not topology_applied:
            raise ProtocolError(
                f"{name}[{position}] normal decision credit/topology receipt drifted"
            )
        rows.append(
            {
                "decision_index": decision_index,
                "physical_step_start": physical_step,
                "proposed_primitive_count": proposed,
                "executed_primitive_count": executed,
                "official_shadow_stop_phase": phase,
                "actor_sampled": sampled,
                "actor_credit_eligible": credit,
                "topology_receipt_applied": topology_applied,
            }
        )
        previous_end += executed
    return tuple(rows)


def _episode_result(value: object, expected: Mapping, name: str) -> dict:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{name} must be an object")
    identity = _identity(value, name)
    if identity != dict(expected):
        raise ProtocolError(f"{name} identity/order differs from frozen development set")
    success = _boolean(value.get("success"), f"{name}.success")
    spl = _finite(value.get("spl"), f"{name}.spl", minimum=0.0)
    if spl > 1.0:
        raise ProtocolError(f"{name}.spl exceeds one")
    initial = _finite(
        value.get("initial_distance_m"), f"{name}.initial_distance_m", minimum=0.0
    )
    final_ne = _finite(
        value.get("final_navigation_error_m"),
        f"{name}.final_navigation_error_m",
        minimum=0.0,
    )
    timeout = _boolean(value.get("timeout"), f"{name}.timeout")
    source = value.get("stop_source")
    if source not in STOP_SOURCES:
        raise ProtocolError(f"{name}.stop_source is invalid")
    successful_stop = _boolean(
        value.get("successful_stop"), f"{name}.successful_stop"
    )
    if successful_stop != (success and source != "none"):
        raise ProtocolError(f"{name} successful STOP attribution is inconsistent")
    trace = _sha(value.get("executed_trace_sha256"), f"{name}.executed_trace_sha256")
    candidate_counts = _count_mapping(value.get("candidate_counts"), f"{name}.candidate_counts")
    primitive_counts = _count_mapping(value.get("primitive_counts"), f"{name}.primitive_counts")
    waypoint_macros = _integer(
        value.get("waypoint_macro_count"), f"{name}.waypoint_macro_count"
    )
    if sum(primitive_counts.values()) > 160:
        raise ProtocolError(f"{name} exceeded the 160-primitive budget")
    post_stop = _integer(
        value.get("post_shadow_stop_primitive_count"),
        f"{name}.post_shadow_stop_primitive_count",
    )
    external_credit = _integer(
        value.get("external_stop_actor_credit_count"),
        f"{name}.external_stop_actor_credit_count",
    )
    if post_stop != 0 or external_credit != 0:
        raise ProtocolError(f"{name} violated official shadow STOP isolation")
    decision_receipts = _decision_receipts(
        value.get("decision_receipts"), f"{name}.decision_receipts"
    )
    if sum(row["executed_primitive_count"] for row in decision_receipts) != sum(
        primitive_counts.values()
    ):
        raise ProtocolError(f"{name} decision receipts differ from primitive counts")
    external_rows = [
        row
        for row in decision_receipts
        if row["official_shadow_stop_phase"] != "none"
    ]
    if external_rows and decision_receipts[-1] != external_rows[-1]:
        raise ProtocolError(f"{name} executed decisions after official shadow STOP")
    if source == "official_shadow" and len(external_rows) != 1:
        raise ProtocolError(f"{name} shadow STOP attribution lacks one terminal receipt")
    if source != "official_shadow" and external_rows:
        raise ProtocolError(f"{name} has unattributed official shadow STOP receipt")
    return {
        **identity,
        "success": success,
        "spl": spl,
        "initial_distance_m": initial,
        "final_navigation_error_m": final_ne,
        "timeout": timeout,
        "stop_source": source,
        "successful_stop": successful_stop,
        "executed_trace_sha256": trace,
        "candidate_counts": candidate_counts,
        "primitive_counts": primitive_counts,
        "waypoint_macro_count": waypoint_macros,
        "path_length_m": _finite(
            value.get("path_length_m"), f"{name}.path_length_m", minimum=0.0
        ),
        "collision_count": _integer(
            value.get("collision_count"), f"{name}.collision_count"
        ),
        "blocked_forward_count": _integer(
            value.get("blocked_forward_count"), f"{name}.blocked_forward_count"
        ),
        "episode_return": _finite(value.get("episode_return"), f"{name}.episode_return"),
        "geodesic_progress_m": _finite(
            value.get("geodesic_progress_m"), f"{name}.geodesic_progress_m"
        ),
        "decision_receipts": decision_receipts,
    }


def _primitive_count(counts: Mapping[str, int], name: str) -> int:
    aliases = {
        "forward": ("FORWARD", "forward", "1"),
        "left": ("LEFT", "left", "2"),
        "right": ("RIGHT", "right", "3"),
    }
    return sum(counts.get(key, 0) for key in aliases[name])


def normalize_evaluation(value: object, expected: Sequence[Mapping], name: str) -> dict:
    if not isinstance(value, Mapping) or not isinstance(value.get("episodes"), list):
        raise ProtocolError(f"{name} must contain episode evidence")
    episodes = value["episodes"]
    if len(episodes) != len(expected):
        raise ProtocolError(f"{name} must evaluate all eight frozen episodes")
    normalized = tuple(
        _episode_result(row, identity, f"{name}.episodes[{index}]")
        for index, (row, identity) in enumerate(zip(episodes, expected))
    )
    candidates: Counter[str] = Counter()
    primitives: Counter[str] = Counter()
    for row in normalized:
        candidates.update(row["candidate_counts"])
        primitives.update(row["primitive_counts"])
    total_primitives = sum(primitives.values())
    if total_primitives < 1:
        raise ProtocolError(f"{name} executed no primitives")
    forward = _primitive_count(primitives, "forward")
    left = _primitive_count(primitives, "left")
    right = _primitive_count(primitives, "right")
    count = len(normalized)
    aggregate = {
        "episode_count": count,
        "success_count": sum(row["success"] for row in normalized),
        "successful_stop_count": sum(row["successful_stop"] for row in normalized),
        "policy_successful_stop_count": sum(
            row["successful_stop"] and row["stop_source"] == "policy"
            for row in normalized
        ),
        "official_shadow_successful_stop_count": sum(
            row["successful_stop"] and row["stop_source"] == "official_shadow"
            for row in normalized
        ),
        "failed_policy_stop_count": sum(
            (not row["success"]) and row["stop_source"] == "policy"
            for row in normalized
        ),
        "failed_official_shadow_stop_count": sum(
            (not row["success"]) and row["stop_source"] == "official_shadow"
            for row in normalized
        ),
        "timeout_count": sum(row["timeout"] for row in normalized),
        "mean_spl": sum(row["spl"] for row in normalized) / count,
        "mean_initial_distance_m": sum(row["initial_distance_m"] for row in normalized) / count,
        "mean_navigation_error_m": sum(row["final_navigation_error_m"] for row in normalized) / count,
        "mean_path_length_m": sum(row["path_length_m"] for row in normalized) / count,
        "collision_count": sum(row["collision_count"] for row in normalized),
        "blocked_forward_count": sum(row["blocked_forward_count"] for row in normalized),
        "waypoint_episode_count": sum(row["waypoint_macro_count"] > 0 for row in normalized),
        "candidate_counts": dict(sorted(candidates.items())),
        "primitive_counts": dict(sorted(primitives.items())),
        "candidate_category_count": sum(count > 0 for count in candidates.values()),
        "primitive_category_count": sum(count > 0 for count in primitives.values()),
        "forward_primitive_fraction": forward / total_primitives,
        "left_primitive_fraction": left / total_primitives,
        "right_primitive_fraction": right / total_primitives,
        "mean_episode_return": sum(row["episode_return"] for row in normalized) / count,
        "mean_geodesic_progress_m": sum(row["geodesic_progress_m"] for row in normalized) / count,
    }
    aggregate["mean_distance_reduction_m"] = (
        aggregate["mean_initial_distance_m"]
        - aggregate["mean_navigation_error_m"]
    )
    return {"episodes": normalized, "aggregate": aggregate}


def adjudicate_gate_a(evaluation: Mapping, thresholds: Mapping) -> dict:
    aggregate = evaluation["aggregate"]
    checks = {
        "success": aggregate["success_count"] >= thresholds["success_count_min"],
        "successful_stop": aggregate["successful_stop_count"]
        >= thresholds["successful_stop_count_min"],
        "spl": aggregate["mean_spl"] > thresholds["spl_min_exclusive"],
        "distance_reduction": aggregate["mean_distance_reduction_m"]
        >= thresholds["mean_distance_reduction_m_min"],
        "timeout": aggregate["timeout_count"] <= thresholds["timeout_count_max"],
        "waypoint_execution": aggregate["waypoint_episode_count"]
        >= thresholds["waypoint_episode_count_min"],
        "candidate_diversity": aggregate["candidate_category_count"]
        >= thresholds["candidate_category_count_min"],
        "primitive_diversity": aggregate["primitive_category_count"]
        >= thresholds["primitive_category_count_min"],
        "forward_fraction": aggregate["forward_primitive_fraction"]
        >= thresholds["forward_primitive_fraction_min"],
        "left_not_collapsed": aggregate["left_primitive_fraction"]
        <= thresholds["left_primitive_fraction_max"],
        "right_not_collapsed": aggregate["right_primitive_fraction"]
        <= thresholds["right_primitive_fraction_max"],
    }
    return {"pass": all(checks.values()), "checks": checks}


def adjudicate_gate_b(bc: Mapping, bc_ppo: Mapping, thresholds: Mapping) -> dict:
    before = bc["aggregate"]
    after = bc_ppo["aggregate"]
    changed = sum(
        left["executed_trace_sha256"] != right["executed_trace_sha256"]
        for left, right in zip(bc["episodes"], bc_ppo["episodes"])
    )
    deltas = {
        "success": after["success_count"] - before["success_count"],
        "mean_spl": after["mean_spl"] - before["mean_spl"],
        "mean_navigation_error_improvement_m": before["mean_navigation_error_m"]
        - after["mean_navigation_error_m"],
        "timeout": after["timeout_count"] - before["timeout_count"],
        "changed_executed_trace_count": changed,
    }
    checks = {
        "success": deltas["success"] >= thresholds["success_delta_min"],
        "spl": deltas["mean_spl"] >= thresholds["spl_delta_min"],
        "navigation_error": deltas["mean_navigation_error_improvement_m"]
        >= thresholds["mean_navigation_error_improvement_m_min"],
        "timeout": deltas["timeout"] <= thresholds["timeout_delta_max"],
        "executed_trace": changed >= thresholds["changed_executed_trace_count_min"],
        "forward_fraction": after["forward_primitive_fraction"]
        >= thresholds["forward_primitive_fraction_min"],
        "left_not_collapsed": after["left_primitive_fraction"]
        <= thresholds["left_primitive_fraction_max"],
        "right_not_collapsed": after["right_primitive_fraction"]
        <= thresholds["right_primitive_fraction_max"],
    }
    return {"pass": all(checks.values()), "checks": checks, "deltas": deltas}


def _checkpoint(value: object, name: str) -> dict:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{name} checkpoint evidence is missing")
    path_value = value.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise ProtocolError(f"{name}.path must be nonempty")
    path = Path(path_value).expanduser().resolve(strict=True)
    expected = _sha(value.get("sha256"), f"{name}.sha256")
    observed = _artifact_sha256(path)
    if observed != expected:
        raise ProtocolError(f"{name} checkpoint bytes differ from its receipt")
    if value.get("restore_exact") is not True:
        raise ProtocolError(f"{name} checkpoint was not restored exactly")
    return {"path": str(path), "sha256": observed, "restore_exact": True}


def _validate_stop_sync(
    sync: Mapping,
    receipt_groups: Sequence[Sequence[Mapping]],
    *,
    name: str,
) -> dict:
    flattened = [row for group in receipt_groups for row in group]
    expected = {
        "executed_primitive_count": sum(
            row["executed_primitive_count"] for row in flattened
        ),
        "physical_boundary_ack_count": sum(
            row["executed_primitive_count"] for row in flattened
        ),
        "post_shadow_stop_primitive_count": 0,
        "external_stop_actor_credit_count": sum(
            row["actor_credit_eligible"]
            and row["official_shadow_stop_phase"] != "none"
            for row in flattened
        ),
        "sampled_after_predecision_shadow_stop_count": sum(
            row["actor_sampled"]
            and row["official_shadow_stop_phase"] == "predecision"
            for row in flattened
        ),
        "predecision_shadow_stop_count": sum(
            row["official_shadow_stop_phase"] == "predecision"
            for row in flattened
        ),
        "mid_macro_shadow_stop_count": sum(
            row["official_shadow_stop_phase"] == "mid_macro"
            for row in flattened
        ),
    }
    observed = {
        key: _integer(sync.get(key), f"{name}.{key}") for key in expected
    }
    if observed != expected:
        raise ProtocolError(f"{name} differs from per-decision STOP receipts")
    if (
        observed["external_stop_actor_credit_count"] != 0
        or observed["sampled_after_predecision_shadow_stop_count"] != 0
        or observed["post_shadow_stop_primitive_count"] != 0
    ):
        raise ProtocolError(f"{name} official shadow STOP synchronization failed")
    return observed


def _training_guard(
    value: object,
    *,
    name: str,
    manifest: Mapping,
    require_manager: bool,
    required_changed_groups: Sequence[str],
) -> dict:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{name} training evidence is missing")
    checkpoint = _checkpoint(value.get("checkpoint"), f"{name}.checkpoint")
    for key in ("finite_losses", "finite_gradients", "frozen_prior_unchanged", "frozen_stack_core_unchanged"):
        if value.get(key) is not True:
            raise ProtocolError(f"{name}.{key} guard failed")
    changed = value.get("changed_parameter_groups")
    if not isinstance(changed, list) or not set(required_changed_groups).issubset(changed):
        raise ProtocolError(f"{name} lacks required changed parameter groups")
    sync = value.get("official_stop_sync")
    if not isinstance(sync, Mapping):
        raise ProtocolError(f"{name} official STOP evidence is missing")
    normalized = {
        "checkpoint": checkpoint,
        "changed_parameter_groups": tuple(changed),
    }
    if not require_manager:
        if value.get("supervision_steps") != manifest["budget"]["bc_supervision_steps"]:
            raise ProtocolError("BC supervision budget drifted")
        if value.get("objective") != manifest["bc_objective"]:
            raise ProtocolError("BC evidence does not bind the frozen objective")
        rows = value.get("episode_identities")
        if not isinstance(rows, list) or len(rows) != EXPECTED_EPISODES:
            raise ProtocolError("BC must use exactly the frozen eight train episodes")
        normalized_rows = tuple(
            _identity(row, f"bc.episode_identities[{index}]")
            for index, row in enumerate(rows)
        )
        if normalized_rows != tuple(manifest["dataset"]["episodes"]["train"]):
            raise ProtocolError("BC episodes differ from the frozen train set")
        traversals = value.get("traversals")
        traversal_count = manifest["budget"]["bc_traversal_count"]
        if not isinstance(traversals, list) or len(traversals) != traversal_count:
            raise ProtocolError("BC must contain exactly sixteen fixed traversals")
        normalized_traversals = []
        receipt_groups = []
        expected_passes = tuple(manifest["budget"]["bc_collection_passes"])
        for index, traversal in enumerate(traversals):
            if not isinstance(traversal, Mapping):
                raise ProtocolError(f"BC traversal {index} is invalid")
            pass_index, episode_index = divmod(index, EXPECTED_EPISODES)
            expected_identity = manifest["dataset"]["episodes"]["train"][episode_index]
            if traversal.get("pass") != expected_passes[pass_index]:
                raise ProtocolError("BC traversal pass/order drifted")
            if traversal.get("traversal_index") != index:
                raise ProtocolError("BC traversal index drifted")
            if _identity(traversal, f"bc.traversals[{index}]") != expected_identity:
                raise ProtocolError("BC traversal identity/order drifted")
            receipts = _decision_receipts(
                traversal.get("decision_receipts"),
                f"bc.traversals[{index}].decision_receipts",
            )
            normalized_traversals.append(
                {
                    **expected_identity,
                    "pass": expected_passes[pass_index],
                    "traversal_index": index,
                    "decision_receipts": receipts,
                }
            )
            receipt_groups.append(receipts)
        normalized["official_stop_sync"] = _validate_stop_sync(
            sync, receipt_groups, name=f"{name}.official_stop_sync"
        )
        support = value.get("label_support")
        if not isinstance(support, Mapping) or any(
            _integer(support.get(label), f"bc.label_support.{label}", minimum=1) < 1
            for label in ("stop", "non_stop", "waypoint")
        ):
            raise ProtocolError("BC lacks route/STOP and metric-waypoint supervision")
        normalized["traversals"] = tuple(normalized_traversals)
        return normalized

    updates = value.get("updates")
    exact_updates = manifest["budget"]["ppo_update_count"]
    if not isinstance(updates, list) or len(updates) != exact_updates:
        raise ProtocolError("PPO must contain exactly eight fixed updates")
    for index, update in enumerate(updates):
        if not isinstance(update, Mapping):
            raise ProtocolError(f"PPO update {index} evidence is invalid")
        ratio = _finite(
            update.get("pre_update_ratio_max_abs_deviation"),
            f"ppo.updates[{index}].pre_update_ratio_max_abs_deviation",
            minimum=0.0,
        )
        kl = _finite(
            update.get("bc_reference_kl"),
            f"ppo.updates[{index}].bc_reference_kl",
            minimum=0.0,
        )
        if ratio > manifest["training_guard"]["pre_update_ratio_max_abs_deviation"]:
            raise ProtocolError("PPO pre-update ratio guard failed")
        if kl > manifest["training_guard"]["bc_reference_kl_max"]:
            raise ProtocolError("PPO exceeded the frozen BC-reference KL guard")
        if update.get("reward") != manifest["reward"]:
            raise ProtocolError("PPO update reward contract drifted")
        if update.get("objective") != manifest["ppo"]:
            raise ProtocolError("PPO update objective/hyperparameters drifted")
    root = value.get("root_forward")
    if not isinstance(root, Mapping):
        raise ProtocolError("PPO RLinf root-forward evidence is missing")
    operations = root.get("operations")
    if (
        root.get("path") != "manager.model"
        or root.get("fsdp_wrapped") is not True
        or _integer(root.get("manager_model_forward_count"), "ppo.manager_model_forward_count", minimum=1) < 1
        or root.get("local_policy_callback_forward_count") != 0
        or not isinstance(operations, Mapping)
        or any(_integer(operations.get(op), f"ppo.operations.{op}", minimum=1) < 1 for op in ("sample", "replay", "bootstrap"))
    ):
        raise ProtocolError("PPO did not traverse the RLinf manager.model/FSDP root")
    train_rows = value.get("episode_identities")
    if not isinstance(train_rows, list) or len(train_rows) != EXPECTED_EPISODES:
        raise ProtocolError("PPO episode evidence must contain all eight frozen episodes")
    normalized_train = tuple(
        _identity(row, f"ppo.episode_identities[{index}]")
        for index, row in enumerate(train_rows)
    )
    if normalized_train != tuple(manifest["dataset"]["episodes"]["train"]):
        raise ProtocolError("PPO episodes differ from the frozen train set")
    if len(normalized_train) != len(updates):
        raise ProtocolError("A1 requires exactly one PPO update per frozen episode")
    rollouts = value.get("episode_rollouts")
    if not isinstance(rollouts, list) or len(rollouts) != EXPECTED_EPISODES:
        raise ProtocolError("PPO requires one receipt-bearing rollout per update")
    receipt_groups = []
    normalized_rollouts = []
    for index, (rollout, expected_identity) in enumerate(zip(rollouts, normalized_train)):
        if not isinstance(rollout, Mapping):
            raise ProtocolError(f"PPO rollout {index} is invalid")
        if rollout.get("update_index") != index:
            raise ProtocolError("PPO rollout/update order drifted")
        if _identity(rollout, f"ppo.episode_rollouts[{index}]") != expected_identity:
            raise ProtocolError("PPO rollout identity/order drifted")
        receipts = _decision_receipts(
            rollout.get("decision_receipts"),
            f"ppo.episode_rollouts[{index}].decision_receipts",
        )
        eligible = sum(row["actor_credit_eligible"] for row in receipts)
        if _integer(
            rollout.get("actor_record_count"),
            f"ppo.episode_rollouts[{index}].actor_record_count",
            minimum=1,
        ) != eligible:
            raise ProtocolError("PPO actor batch differs from credit-eligible receipts")
        dropped = sum(
            row["actor_sampled"] and not row["actor_credit_eligible"]
            for row in receipts
        )
        if rollout.get("dropped_external_decision_count") != dropped:
            raise ProtocolError("PPO external STOP actor-drop evidence drifted")
        receipt_groups.append(receipts)
        normalized_rollouts.append(
            {
                **expected_identity,
                "update_index": index,
                "actor_record_count": eligible,
                "dropped_external_decision_count": dropped,
                "decision_receipts": receipts,
            }
        )
    normalized["official_stop_sync"] = _validate_stop_sync(
        sync, receipt_groups, name=f"{name}.official_stop_sync"
    )
    normalized.update(
        {
            "updates": tuple(dict(row) for row in updates),
            "root_forward": dict(root),
            "episode_rollouts": tuple(normalized_rollouts),
        }
    )
    return normalized


def _source_audit(value: object, manifest: Mapping) -> dict:
    if not isinstance(value, Mapping):
        raise ProtocolError("live source audit is missing")
    split_reads = value.get("split_reads")
    if (
        value.get("streamvln_commit") != manifest["source_contract"]["streamvln_commit"]
        or value.get("feature_provider") != manifest["frozen_prior"]["provider"]
        or value.get("feature_dim") != 3604
        or value.get("feature_schema_sha256") != manifest["frozen_prior"]["schema_sha256"]
        or value.get("feature_v2_enabled") is not True
        or value.get("frozen_prior") is not True
        or value.get("ttt_or_secondary_loaded") is not False
        or value.get("model_load_count") != 1
        or value.get("official_shadow_stop_mode")
        != "synchronized_physical_boundary_override"
        or not isinstance(split_reads, Mapping)
        or split_reads.get("val_seen") != 0
        or split_reads.get("val_unseen") != 0
        or _integer(split_reads.get("train"), "source_audit.split_reads.train", minimum=1) < 1
    ):
        raise ProtocolError("live clean-e48/train-only source audit failed")
    return dict(value)


def validate_payload(payload: object, manifest: Mapping, contract: Mapping) -> dict:
    if not isinstance(payload, Mapping):
        raise ProtocolError("A1 provider payload must be an object")
    source = _source_audit(payload.get("source_audit"), manifest)
    required = contract["required_trainable_groups"]
    bc_required = tuple(group for group in required if group != "value_head")
    bc_training = _training_guard(
        payload.get("bc_training"),
        name="bc_training",
        manifest=manifest,
        require_manager=False,
        required_changed_groups=bc_required,
    )
    bc_eval = normalize_evaluation(
        payload.get("bc_evaluation"), contract["development"], "bc_evaluation"
    )
    gate_a = adjudicate_gate_a(bc_eval, manifest["gate_a"])
    if not gate_a["pass"]:
        if payload.get("ppo_training") is not None or payload.get("matched_evaluation") is not None:
            raise ProtocolError("PPO ran after BC competence gate failed")
        return {
            "status": "bc_competence_failed",
            "source_audit": source,
            "bc_training": bc_training,
            "bc_evaluation": bc_eval,
            "gate_a": gate_a,
            "ppo_started": False,
            "gate_b": None,
        }

    ppo_value = payload.get("ppo_training")
    if not isinstance(ppo_value, Mapping):
        raise ProtocolError(
            "BC competence passed but no PPO evidence was supplied; refusing a local callback substitute"
        )
    if ppo_value.get("initialization_checkpoint_sha256") != bc_training["checkpoint"]["sha256"]:
        raise ProtocolError("BC and BC+PPO arms do not share the same BC initialization")
    ppo_training = _training_guard(
        ppo_value,
        name="ppo_training",
        manifest=manifest,
        require_manager=True,
        required_changed_groups=required,
    )
    if ppo_training["checkpoint"]["sha256"] == bc_training["checkpoint"]["sha256"]:
        raise ProtocolError("PPO checkpoint did not differ from the shared BC initialization")
    matched = payload.get("matched_evaluation")
    if not isinstance(matched, Mapping):
        raise ProtocolError("matched BC versus BC+PPO evaluation is missing")
    matched_bc = normalize_evaluation(
        matched.get("bc"), contract["development"], "matched_evaluation.bc"
    )
    if _json_hash(matched_bc) != _json_hash(bc_eval):
        raise ProtocolError("matched BC arm differs from the one-shot competence evaluation")
    matched_ppo = normalize_evaluation(
        matched.get("bc_ppo"), contract["development"], "matched_evaluation.bc_ppo"
    )
    gate_b = adjudicate_gate_b(matched_bc, matched_ppo, manifest["gate_b"])
    return {
        "status": "pilot_signal_positive" if gate_b["pass"] else "pilot_signal_negative",
        "source_audit": source,
        "bc_training": bc_training,
        "bc_evaluation": bc_eval,
        "gate_a": gate_a,
        "ppo_started": True,
        "ppo_training": ppo_training,
        "matched_evaluation": {"bc": matched_bc, "bc_ppo": matched_ppo},
        "gate_b": gate_b,
    }


def _load_provider(spec: str, *, manifest: Mapping, contract: Mapping, args: argparse.Namespace):
    if not spec or ":" not in spec:
        raise ProtocolError(
            "pilot requires --provider module:attribute; no local-policy fallback is allowed"
        )
    module_name, attribute = spec.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute, None)
    if not callable(factory):
        raise ProtocolError(f"A1 provider factory is not callable: {spec}")
    provider = factory(manifest=manifest, contract=contract, args=args)
    if not callable(getattr(provider, "execute", None)):
        raise ProtocolError("A1 provider lacks execute()")
    return provider


def _atomic_json(path: Path, value: Mapping) -> None:
    path = path.expanduser().absolute()
    if path.exists():
        raise FileExistsError(f"refusing to replace A1 result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"stale A1 temporary result exists: {temporary}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def run(
    args: argparse.Namespace,
    *,
    provider: A1Provider | Callable | None = None,
) -> dict:
    manifest_path = Path(args.manifest).expanduser().resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = validate_manifest(manifest)
    root = Path(__file__).resolve().parents[2]
    lineage = validate_lineage(manifest, contract, root=root)
    if args.split != "train":
        raise ProtocolError("A1 runner permits only --split train")
    base = {
        "schema_name": RESULT_SCHEMA_NAME,
        "schema_version": RESULT_SCHEMA_VERSION,
        "stage": args.stage,
        "manifest": {
            "path": str(manifest_path),
            "file_sha256": _file_sha256(manifest_path),
            "canonical_sha256": _json_hash(manifest),
        },
        "lineage": lineage,
        "episodes": {
            "train": contract["train"],
            "train_development": contract["development"],
        },
        "claim_boundary": dict(manifest["claim_boundary"]),
    }
    if args.stage == "preflight":
        base.update(
            {
                "status": "preflight_passed_live_provider_not_run",
                "ppo_started": False,
                "next_action": (
                    "run_pilot_with_"
                    "streamvln_a1_live_provider:create_provider"
                ),
                "pilot_blocker": (
                    "live execution requires the pinned clean-e48 checkpoint, "
                    "Habitat assets, CUDA, and RLinf manager.model/FSDP runtime"
                ),
            }
        )
    elif args.stage == "pilot":
        if provider is None:
            provider = _load_provider(
                args.provider, manifest=manifest, contract=contract, args=args
            )
        payload = (
            provider.execute(manifest=manifest, contract=contract, args=args)
            if callable(getattr(provider, "execute", None))
            else provider(manifest=manifest, contract=contract, args=args)
        )
        base.update(validate_payload(payload, manifest, contract))
    else:
        raise ProtocolError("A1 stage must be preflight or pilot")
    base["receipt_sha256"] = _json_hash(base)
    _atomic_json(Path(args.output), base)
    return base


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    default_manifest = Path(__file__).with_name(
        "streamvln_a1_contextual_hybrid_manifest.json"
    )
    parser.add_argument("--stage", choices=("preflight", "pilot"), required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--manifest", type=Path, default=default_manifest)
    parser.add_argument("--provider", default="")
    parser.add_argument("--release-manifest", type=Path)
    parser.add_argument("--rlinf-root", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--vision", type=Path)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
