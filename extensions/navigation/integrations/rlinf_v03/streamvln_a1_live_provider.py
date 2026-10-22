"""Canonical clean-e48/Habitat provider for the A1 BC competence gate.

The provider deliberately implements the first causal boundary completely:
one clean-e48 model load, eight expert plus eight DAgger train traversals,
1,000 batch-one BC updates, an independently restored head checkpoint, and
one deterministic pass over the eight scene-disjoint train-development
episodes.  Official shadow STOP is consumed before the actor and at every
physical primitive boundary.

PPO is fail-closed after a passing Gate A at the observed process boundary:
the official Feature-v2 process is pinned to Torch 2.1.2/Transformers 4.45.1,
while the RLinf FSDP manager requires its Torch 2.6 environment.  The next
stage is a two-process small-head bridge.  A failed Gate A returns a complete
negative payload and never initializes FSDP.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import secrets
import subprocess
import sys
import tempfile
from typing import Mapping, Sequence


class LiveProviderError(RuntimeError):
    """Fail-closed live-provider error."""


class PPOIntegrationBlocked(LiveProviderError):
    """Gate A passed but the version-isolated FSDP process is not connected."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _python_tree_sha256(path: Path) -> str:
    root = Path(path).resolve(strict=True)
    files = sorted(item for item in root.rglob("*.py") if item.is_file())
    if not files:
        raise LiveProviderError(f"Python source tree is empty: {root}")
    digest = hashlib.sha256()
    for item in files:
        relative = item.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_file_sha256(item)))
    return digest.hexdigest()


def _json_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()


def _save_checkpoint_atomic_py39(adapter, directory: Path) -> dict[str, str]:
    """Use the frozen stack envelope without its Python>=3.10 write helper.

    The official StreamVLN runtime is pinned to Python 3.9, where
    ``Path.write_text`` has no ``newline`` keyword.  Reproduce the accepted
    envelope at the integration boundary and leave ``navrl.stack`` unchanged.
    The ordinary stack verifier still performs the independent restore.
    """

    from navrl.stack.checkpoint import MANIFEST_NAME, _canonical_mapping

    target = Path(directory)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=target.name + ".staging-", dir=str(target.parent))
    )
    declared = dict(adapter.save_checkpoint(staging))
    payload_files = sorted(
        path
        for path in staging.rglob("*")
        if path.is_file() and path.name != MANIFEST_NAME
    )
    actual = {
        path.relative_to(staging).as_posix(): _file_sha256(path)
        for path in payload_files
    }
    if declared != actual:
        raise LiveProviderError("checkpoint codec hashes differ from files written")
    manifest = {
        "schema_version": 2,
        "metadata": _canonical_mapping(
            adapter.checkpoint_metadata(), field="metadata"
        ),
        "compatibility": _canonical_mapping(
            adapter.checkpoint_compatibility(), field="compatibility"
        ),
        "files_sha256": actual,
    }
    manifest_path = staging / MANIFEST_NAME
    with manifest_path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(staging), str(target))
    return actual


def _git_head(path: Path) -> str:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={path}", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    if result.returncode:
        raise LiveProviderError(f"cannot resolve source revision: {result.stderr.strip()}")
    return result.stdout.strip()


def _canonical_import(rlinf_root: Path, module_name: str):
    qualified = f"rlinf.envs.streamvln.{module_name}"
    module = importlib.import_module(qualified)
    expected = (
        rlinf_root / "rlinf" / "envs" / "streamvln" / f"{module_name}.py"
    ).resolve(strict=True)
    actual = Path(str(getattr(module, "__file__", ""))).resolve(strict=True)
    if module.__name__ != qualified or actual != expected:
        raise LiveProviderError(f"canonical import escaped rlinf_root: {qualified}")
    return module


def _scalar(value: object, name: str) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numel"):
        if int(value.numel()) != 1:
            raise LiveProviderError(f"{name} must be scalar")
        value = value.reshape(-1)[0].item()
    elif hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LiveProviderError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise LiveProviderError(f"{name} must be finite")
    return result


def _single_action_transport(action: int) -> list[list[list[int]]]:
    if type(action) is not int or not 0 <= action < 4:
        raise LiveProviderError("primitive action must be an integer in 0..3")
    return [[[action], [0], [0], [0]]]


def _identity_matches(observed: Mapping, expected: Mapping) -> bool:
    scene = str(observed.get("scene_id", observed.get("scene", "")))
    return (
        str(observed.get("episode_id")) == expected["episode_id"]
        and expected["scene"] in scene
    )


def _parameter_versions(module) -> tuple[tuple[str, int, int], ...]:
    """Cheap mutation guard for the multi-billion parameter frozen backbone."""

    rows = []
    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            raise LiveProviderError(f"frozen StreamVLN parameter is trainable: {name}")
        rows.append((name, int(getattr(parameter, "_version", -1)), int(parameter.data_ptr())))
    if not rows:
        raise LiveProviderError("frozen StreamVLN model exposes no parameters")
    return tuple(rows)


def _head_snapshot(policy) -> dict[str, object]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in policy.named_parameters()
    }


def _changed_groups(policy, before: Mapping[str, object]) -> list[str]:
    named = dict(policy.named_parameters())
    groups = {
        "observation_encoder.projection": "observation_encoder.projection",
        "topology_encoder": "topology_encoder",
        "context_fusion": "context_fusion",
        "candidate_scorer": "candidate_scorer",
        "candidate_slot_bias": "candidate_slot_bias",
        "residual_mean_head": "residual_mean_head",
        "residual_log_std": "residual_log_std",
        "value_head": "value_head",
    }
    changed = []
    for group, prefix in groups.items():
        if any(
            name.startswith(prefix)
            and not named[name].detach().cpu().equal(previous)
            for name, previous in before.items()
        ):
            changed.append(group)
    return changed


def _gradient_groups(policy) -> tuple[bool, set[str]]:
    finite = True
    nonzero: set[str] = set()
    group_prefixes = (
        "observation_encoder.projection",
        "topology_encoder",
        "context_fusion",
        "candidate_scorer",
        "candidate_slot_bias",
        "residual_mean_head",
        "residual_log_std",
        "value_head",
    )
    for name, parameter in policy.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        finite = finite and bool(gradient.detach().isfinite().all().item())
        if float(gradient.detach().abs().sum().item()) > 0.0:
            for prefix in group_prefixes:
                if name.startswith(prefix):
                    nonzero.add(prefix)
                    break
    return finite, nonzero


def _nearest_candidate(snapshot, expert: Mapping) -> tuple[int, tuple[float, float]]:
    waypoint = tuple(float(value) for value in expert["waypoint_local_m"])
    if bool(expert["stop"]):
        candidate = snapshot.candidates[-1]
        if candidate.candidate_id != "stop":
            raise LiveProviderError("fixed candidate order no longer ends in STOP")
        return len(snapshot.candidates) - 1, (0.0, 0.0)
    eligible = [
        (index, candidate)
        for index, candidate in enumerate(snapshot.candidates)
        if candidate.valid and candidate.continuous
    ]
    if not eligible:
        raise LiveProviderError("expert waypoint has no valid continuous candidate")
    slot, _ = min(
        eligible,
        key=lambda row: (
            (float(row[1].forward_m) - waypoint[0]) ** 2
            + (float(row[1].lateral_m) - waypoint[1]) ** 2,
            row[0],
        ),
    )
    return int(slot), waypoint


class CanonicalA1LiveProvider:
    """Live provider consumed by ``run_streamvln_a1_contextual_hybrid``."""

    def __init__(self, *, manifest: Mapping, contract: Mapping, args) -> None:
        self.manifest = manifest
        self.contract = contract
        self.args = args
        self._loaded = False
        self._split_reads = Counter({"train": 0, "val_seen": 0, "val_unseen": 0})
        self._model_load_count = 0
        self.policy = None
        self.release_sha256: str | None = None

    def _required_path(self, name: str, *, directory: bool = False) -> Path:
        value = getattr(self.args, name, None)
        if value is None:
            raise LiveProviderError(f"canonical A1 provider requires --{name.replace('_', '-')}")
        path = Path(value).expanduser().resolve(strict=True)
        if directory and not path.is_dir():
            raise LiveProviderError(f"A1 path is not a directory: {path}")
        if not directory and not path.exists():
            raise LiveProviderError(f"A1 path does not exist: {path}")
        return path

    def _load_once(self) -> None:
        if self._loaded:
            return
        if str(self.args.device) != "cuda:0":
            raise LiveProviderError("A1 canonical provider is pinned to process-local cuda:0")
        rlinf_root = self._required_path("rlinf_root", directory=True)
        source_root = self._required_path("source_root", directory=True)
        checkpoint = self._required_path("checkpoint")
        tokenizer = self._required_path("tokenizer")
        vision = self._required_path("vision")
        asset_root = self._required_path("asset_root", directory=True)
        release_manifest = self._required_path("release_manifest")
        output_dir_value = getattr(self.args, "output_dir", None)
        if output_dir_value is None:
            raise LiveProviderError("canonical A1 provider requires --output-dir")
        self.output_dir = Path(output_dir_value).expanduser().absolute()
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise FileExistsError(f"A1 output directory is not empty: {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        expected_commit = self.manifest["source_contract"]["streamvln_commit"]
        if _git_head(source_root) != expected_commit:
            raise LiveProviderError("StreamVLN source is not clean e48")
        train_path = Path(self.manifest["dataset"]["train_path"]).resolve(strict=True)
        if _file_sha256(train_path) != self.manifest["dataset"]["train_sha256"]:
            raise LiveProviderError("R2R train dataset bytes drifted")
        if str(rlinf_root) not in sys.path:
            sys.path.insert(0, str(rlinf_root))

        self.runtime = _canonical_import(rlinf_root, "streamvln_p8_successor_runtime")
        self.official_module = _canonical_import(rlinf_root, "streamvln_p8_official_keep")
        self.adapter_module = _canonical_import(rlinf_root, "streamvln_p8_official_adapter")
        self.feedback_module = _canonical_import(rlinf_root, "streamvln_executor_contract")
        self.action_module = _canonical_import(rlinf_root, "streamvln_p8_successor_action")
        self.stack_module = _canonical_import(rlinf_root, "topology_waypoint_stack")
        self.contextual_module = _canonical_import(
            rlinf_root, "streamvln_contextual_topology_waypoint"
        )
        navrl_stack = importlib.import_module("navrl.stack")
        self.navrl_stack_root = Path(navrl_stack.__file__).resolve(strict=True).parent
        self.navrl_stack_sha256_before = _python_tree_sha256(self.navrl_stack_root)
        self.waypoint_compiler_path = Path(self.action_module.__file__).resolve(strict=True)
        self.waypoint_compiler_sha256_before = _file_sha256(
            self.waypoint_compiler_path
        )

        if (
            self.contextual_module.CONTEXTUAL_FEATURE_DIM != 3604
            or self.contextual_module.CONTEXTUAL_FEATURE_SCHEMA_SHA256
            != self.manifest["frozen_prior"]["schema_sha256"]
        ):
            raise LiveProviderError("deployed contextual Feature-v2 schema drifted")
        release = json.loads(release_manifest.read_text(encoding="utf-8"))
        factory = self.adapter_module.PinnedStreamVLNGenerationAdapter.from_released_checkpoint
        self.generation_adapter = factory(
            model_path=str(checkpoint),
            tokenizer_path=str(tokenizer),
            vision_tower_path=str(vision),
            source_root=str(source_root),
            artifact_manifest=release,
            device=str(self.args.device),
            local_rank=0,
            env_count=1,
            model_env_id=0,
            enable_feature_v2=True,
        )
        for parameter in self.generation_adapter.model.parameters():
            parameter.requires_grad_(False)
        self._model_load_count = 1
        self.release_sha256 = _file_sha256(release_manifest)
        self.asset_root = asset_root
        self.rlinf_root = rlinf_root

        import torch

        seed = int(self.manifest["budget"]["seed"])
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        architecture = self.manifest["architecture"]
        self.policy = self.contextual_module.StreamVLNContextualTopologyWaypointPolicy(
            release_manifest_sha256=self.release_sha256,
            hidden_dim=int(architecture["hidden_dim"]),
            topology_max_nodes=int(architecture["topology_max_nodes"]),
            topology_model_dim=int(architecture["topology_model_dim"]),
            max_candidates=int(architecture["max_candidates"]),
            residual_limit_m=float(architecture["residual_limit_m"]),
            initial_log_std=float(architecture["initial_log_std"]),
            feature_dim=3604,
        ).float().to(self.args.device)
        self._loaded = True

    def _run_cfg(self) -> dict:
        runtime = self.manifest["runtime"]
        return {
            "seed": int(self.manifest["budget"]["seed"]),
            "habitat_python": runtime["habitat_python"],
            "habitat_server": str(self.rlinf_root / "scripts" / "habitat_r2r_service.py"),
            "asset_root": str(self.asset_root),
            "habitat_sim_python": runtime["habitat_sim_python"],
            "habitat_env_lib": runtime["habitat_env_lib"],
            "egl_vendor_json": runtime["egl_vendor_json"],
            "habitat_startup_timeout_s": float(runtime["habitat_startup_timeout_s"]),
        }

    def _new_state(self):
        return self.stack_module.EpisodeTopologyState(
            max_decisions=int(self.manifest["budget"]["max_primitives_per_episode"]),
            topology_max_nodes=int(self.manifest["architecture"]["topology_max_nodes"]),
            residual_limit_m=float(self.manifest["architecture"]["residual_limit_m"]),
        )

    def _new_policy(self):
        architecture = self.manifest["architecture"]
        return self.contextual_module.StreamVLNContextualTopologyWaypointPolicy(
            release_manifest_sha256=str(self.release_sha256),
            hidden_dim=int(architecture["hidden_dim"]),
            topology_max_nodes=int(architecture["topology_max_nodes"]),
            topology_model_dim=int(architecture["topology_model_dim"]),
            max_candidates=int(architecture["max_candidates"]),
            residual_limit_m=float(architecture["residual_limit_m"]),
            initial_log_std=float(architecture["initial_log_std"]),
            feature_dim=3604,
        ).float().to(self.args.device)

    def _driver_and_provider(self, env, spec: Mapping):
        identity = getattr(env, "identity", None)
        if not isinstance(identity, Mapping) or not _identity_matches(identity, spec):
            raise LiveProviderError("live Habitat identity differs from A1 manifest")
        session = self.official_module.OfficialEvalSession(
            env_id=int(env.env_id),
            instruction=identity["instruction_text"],
            conjunction_rng=random.Random(
                int(self.manifest["budget"]["seed"]) + int(spec["index"])
            ),
        )
        driver = self.generation_adapter.make_driver(
            session=session,
            episode_serial=int(env._episode_serial),
            episode_identity=identity,
        )
        provider = self.contextual_module.StreamVLNContextualFeatureProvider(
            self.generation_adapter, driver
        )
        return identity, driver, provider

    def _apply_topology_receipt(
        self,
        *,
        state,
        identity,
        decision_index: int,
        physical_step: int,
        actions: Sequence[int],
        rewards: Sequence[float],
        feedback: Sequence[Mapping],
        next_observation: object,
        terminated: bool,
        truncated: bool,
        source: str,
    ) -> None:
        from navrl.stack import ExecutionReceipt

        width = 4
        duration = len(actions)
        receipt = ExecutionReceipt(
            identity=identity,
            decision_index=decision_index,
            physical_step=physical_step,
            source_action_payload=(source, decision_index),
            executed_primitives=tuple(actions) + (None,) * (width - duration),
            executed_mask=tuple(index < duration for index in range(width)),
            credited_log_prob_mask=(True,),
            primitive_rewards=tuple(float(value) for value in rewards)
            + (0.0,) * (width - duration),
            duration=duration,
            next_observation=next_observation,
            terminated=terminated,
            truncated=truncated,
            end_reason=(
                "official_shadow_stop"
                if source == "official_shadow_stop"
                else "terminated"
                if terminated
                else "truncated"
                if truncated
                else "plan_complete"
            ),
            primitive_feedback=tuple(dict(row) for row in feedback),
        )
        state.apply_receipt(receipt)

    def _execute_prefix(
        self,
        *,
        env,
        feature_provider,
        first_observation,
        proposed: Sequence[int],
        state,
        identity,
        decision_index: int,
        physical_step: int,
        actor_sampled: bool,
    ) -> dict:
        if not proposed or len(proposed) > 4:
            raise LiveProviderError("A1 metric waypoint must compile to one to four primitives")
        executed: list[int] = []
        rewards: list[float] = []
        rows: list[dict] = []
        observation = first_observation
        terminated = truncated = False
        external_stop = False
        external_stop_phase = "none"
        next_observation = None
        last_info = None
        distance_before = float(env.expert_waypoint(1.0)["distance_to_goal_m"])
        reward_contract = self.manifest["reward"]
        for index, proposed_action in enumerate(proposed):
            if index:
                frame = env.observe_official_frame()
                observation = feature_provider.prepare(frame)
            action = int(proposed_action)
            if observation.force_stop:
                action = 0
                external_stop = True
                external_stop_phase = "predecision" if index == 0 else "mid_macro"
            observations, reward_tensor, terminations, truncations, infos = env.chunk_step(
                _single_action_transport(action), execute_count=1
            )
            if len(observations) != 1 or len(infos) != 1:
                raise LiveProviderError("A1 successor env returned a non-singleton batch")
            feedback = self.feedback_module.feedback_from_dict(
                infos[0].get("executor_feedback")
            )
            if feedback.executed_actions != (action,) or len(feedback.primitives) != 1:
                raise LiveProviderError("A1 env did not execute exactly one primitive")
            primitive = feedback.primitives[0]
            terminated = bool(terminations.any().item())
            truncated = bool(truncations.any().item())
            if terminated and truncated:
                raise LiveProviderError("A1 primitive cannot terminate and truncate")
            feature_provider.acknowledge(terminated=terminated, truncated=truncated)
            executed.append(action)
            environment_reward = _scalar(
                reward_tensor.reshape(-1)[0], "environment reward"
            )
            episode_metrics = infos[0].get("episode")
            terminal_metrics = episode_metrics if isinstance(episode_metrics, Mapping) else {}
            distance_after = None
            for key in ("distance_to_goal", "navigation_error_m", "navigation_error"):
                if key in terminal_metrics:
                    distance_after = _scalar(
                        terminal_metrics[key], f"primitive terminal metric {key}"
                    )
                    break
            if distance_after is None:
                distance_after = float(
                    env.expert_waypoint(1.0)["distance_to_goal_m"]
                )
            success = bool(
                round(_scalar(terminal_metrics["success"], "primitive success"))
            ) if "success" in terminal_metrics else False
            failed_stop = action == 0 and not success
            training_reward = (
                float(reward_contract["delta_geodesic_coef"])
                * (distance_before - distance_after)
                + float(reward_contract["success_coef"]) * float(success)
                + float(reward_contract["step_coef"])
                + float(reward_contract["collision_coef"])
                * float(bool(primitive.collision))
                + float(reward_contract["failed_stop_coef"]) * float(failed_stop)
                + float(reward_contract["timeout_coef"]) * float(truncated)
            )
            if not math.isfinite(training_reward):
                raise LiveProviderError("A1 fixed training reward became non-finite")
            rewards.append(training_reward)
            primitive_row = primitive.to_dict()
            primitive_row.update(
                {
                    "environment_reward_audit": environment_reward,
                    "distance_before_m": distance_before,
                    "distance_after_m": distance_after,
                    "training_reward": training_reward,
                }
            )
            rows.append(primitive_row)
            distance_before = distance_after
            next_observation = observations[0]
            last_info = infos[0]
            if (
                action == 0
                or terminated
                or truncated
                or bool(primitive.collision)
                or bool(primitive.blocked)
            ):
                break
        if external_stop and (not terminated or executed[-1] != 0):
            raise LiveProviderError("official shadow STOP did not terminate synchronously")
        topology_receipt_applied = not external_stop
        if topology_receipt_applied:
            self._apply_topology_receipt(
                state=state,
                identity=identity,
                decision_index=decision_index,
                physical_step=physical_step,
                actions=executed,
                rewards=rewards,
                feedback=rows,
                next_observation=next_observation,
                terminated=terminated,
                truncated=truncated,
                source="policy_or_teacher",
            )
        decision_receipt = {
            "decision_index": int(decision_index),
            "physical_step_start": int(physical_step),
            "proposed_primitive_count": len(proposed),
            "executed_primitive_count": len(executed),
            "official_shadow_stop_phase": external_stop_phase,
            "actor_sampled": bool(actor_sampled),
            "actor_credit_eligible": bool(actor_sampled and not external_stop),
            "topology_receipt_applied": bool(topology_receipt_applied),
        }
        return {
            "actions": tuple(executed),
            "rewards": tuple(rewards),
            "feedback": tuple(rows),
            "terminated": terminated,
            "truncated": truncated,
            "external_stop": external_stop,
            "next_observation": next_observation,
            "last_info": last_info,
            "decision_receipt": decision_receipt,
        }

    @staticmethod
    def _terminal_metrics(env, last_info: Mapping | None) -> dict:
        raw = None
        if isinstance(last_info, Mapping) and isinstance(last_info.get("episode"), Mapping):
            raw = last_info["episode"]
        if raw is None:
            candidate = getattr(env, "_terminal_metrics", None)
            if isinstance(candidate, Mapping):
                raw = candidate
        if not isinstance(raw, Mapping):
            raise LiveProviderError("A1 episode ended without terminal metrics")
        result = {str(key): _scalar(value, f"terminal metric {key}") for key, value in raw.items()}
        for name in ("success", "spl", "path_length", "collision_count"):
            if name not in result:
                raise LiveProviderError(f"A1 terminal metrics lack {name}")
        return result

    def _run_episode(self, *, spec: Mapping, mode: str, policy_adapter=None) -> dict:
        if mode not in {"expert_collect", "dagger_collect", "bc_eval"}:
            raise ValueError("unknown A1 episode mode")
        env = self.runtime.make_successor_env(
            self._run_cfg(),
            split="train",
            episode_index=int(spec["index"]),
            max_steps=int(self.manifest["budget"]["max_primitives_per_episode"]),
        )
        self._split_reads["train"] += 1
        self._split_reads["val_seen"] += 0
        self._split_reads["val_unseen"] += 0
        rows = []
        candidate_counts: Counter[str] = Counter()
        primitive_counts: Counter[str] = Counter()
        sync = Counter(
            {
                "executed_primitive_count": 0,
                "physical_boundary_ack_count": 0,
                "post_shadow_stop_primitive_count": 0,
                "external_stop_actor_credit_count": 0,
                "sampled_after_predecision_shadow_stop_count": 0,
                "predecision_shadow_stop_count": 0,
                "mid_macro_shadow_stop_count": 0,
            }
        )
        waypoint_macros = 0
        collision_count = blocked_count = 0
        episode_return = 0.0
        trace: list[int] = []
        decision_receipts: list[dict] = []
        last_info = None
        stop_source = "none"
        terminated = truncated = False
        try:
            env.reset()
            observed_identity, _driver, feature_provider = self._driver_and_provider(env, spec)
            from navrl.stack import EpisodeIdentity

            identity = EpisodeIdentity(
                episode_id=str(spec["episode_id"]),
                reset_token=secrets.token_hex(16),
                env_slot=0,
            )
            state = self._new_state()
            if mode in {"dagger_collect", "bc_eval"}:
                if policy_adapter is None:
                    raise LiveProviderError(f"{mode} requires a policy adapter")
                policy_adapter.topology_state = state
                policy_adapter.reset_state(identity)
            else:
                state.bind(identity)

            initial_expert = env.expert_waypoint(1.0)
            initial_distance = float(initial_expert["distance_to_goal_m"])
            physical_step = 0
            decision_index = 0
            while not (terminated or truncated):
                if physical_step >= int(self.manifest["budget"]["max_primitives_per_episode"]):
                    raise LiveProviderError("A1 environment failed to truncate at 160 primitives")
                snapshot = state.snapshot(identity, decision_index, physical_step)
                frame = env.observe_official_frame()
                contextual = feature_provider.prepare(frame)
                policy_observation = contextual.to_policy_observation()
                collect_supervision = mode in {"expert_collect", "dagger_collect"}
                expert = env.expert_waypoint(1.0) if collect_supervision else None
                actor_sampled = False

                if collect_supervision:
                    if not contextual.force_stop:
                        target_slot, target_waypoint = _nearest_candidate(snapshot, expert)
                        forward_inputs = self.policy.prepare_forward_inputs(
                            policy_observation, snapshot, policy_version=0
                        )
                        rows.append(
                            {
                                "forward_inputs": {
                                    key: value.detach().cpu().clone()
                                    for key, value in forward_inputs.items()
                                },
                                "target_candidate": int(target_slot),
                                "target_waypoint": tuple(target_waypoint),
                                "stop": bool(expert["stop"]),
                            }
                        )
                if mode == "expert_collect":
                    if contextual.force_stop:
                        proposed = (0,)
                    elif bool(expert["stop"]):
                        proposed = (0,)
                        stop_source = "policy"
                    else:
                        proposed = tuple(
                            self.action_module.compile_metric_waypoint(
                                tuple(expert["waypoint_local_m"])
                            )
                        )
                        waypoint_macros += 1
                else:
                    if contextual.force_stop:
                        proposed = (0,)
                        stop_source = "official_shadow"
                    else:
                        sample = policy_adapter.sample(
                            policy_observation, identity, decision_index, physical_step
                        )
                        actor_sampled = True
                        candidate_id = str(sample.action_payload[3])
                        candidate_counts[candidate_id] += 1
                        if candidate_id == "stop":
                            proposed = (0,)
                            stop_source = "policy"
                        else:
                            waypoint = (
                                float(sample.action_payload[6]),
                                float(sample.action_payload[7]),
                            )
                            proposed = tuple(
                                self.action_module.compile_metric_waypoint(waypoint)
                            )
                            waypoint_macros += 1

                result = self._execute_prefix(
                    env=env,
                    feature_provider=feature_provider,
                    first_observation=contextual,
                    proposed=proposed,
                    state=state,
                    identity=identity,
                    decision_index=decision_index,
                    physical_step=physical_step,
                    actor_sampled=actor_sampled,
                )
                if result["external_stop"]:
                    stop_source = "official_shadow"
                for action in result["actions"]:
                    primitive_counts[{0: "STOP", 1: "FORWARD", 2: "LEFT", 3: "RIGHT"}[action]] += 1
                    trace.append(int(action))
                for primitive in result["feedback"]:
                    collision_count += int(bool(primitive["collision"]))
                    blocked_count += int(bool(primitive["blocked"]))
                duration = len(result["actions"])
                sync["executed_primitive_count"] += duration
                sync["physical_boundary_ack_count"] += duration
                receipt = dict(result["decision_receipt"])
                decision_receipts.append(receipt)
                phase = receipt["official_shadow_stop_phase"]
                if phase == "predecision":
                    sync["predecision_shadow_stop_count"] += 1
                    sync["sampled_after_predecision_shadow_stop_count"] += int(
                        receipt["actor_sampled"]
                    )
                elif phase == "mid_macro":
                    sync["mid_macro_shadow_stop_count"] += 1
                if phase != "none":
                    sync["external_stop_actor_credit_count"] += int(
                        receipt["actor_credit_eligible"]
                    )
                episode_return += sum(result["rewards"])
                physical_step += duration
                decision_index += 1
                terminated = bool(result["terminated"])
                truncated = bool(result["truncated"])
                last_info = result["last_info"]

            if mode in {"expert_collect", "dagger_collect"}:
                return {
                    "rows": rows,
                    "sync": dict(sync),
                    "physical_steps": physical_step,
                    "episode_id": str(spec["episode_id"]),
                    "decision_receipts": decision_receipts,
                }
            metrics = self._terminal_metrics(env, last_info)
            success = bool(round(float(metrics["success"])))
            final_ne = None
            for key in ("distance_to_goal", "navigation_error_m", "navigation_error"):
                if key in metrics:
                    final_ne = float(metrics[key])
                    break
            if final_ne is None:
                raise LiveProviderError("A1 terminal metrics lack navigation error")
            successful_stop = bool(success and stop_source != "none")
            return {
                "index": int(spec["index"]),
                "episode_id": str(spec["episode_id"]),
                "scene": str(spec["scene"]),
                "success": success,
                "spl": float(metrics["spl"]),
                "initial_distance_m": initial_distance,
                "final_navigation_error_m": final_ne,
                "timeout": bool(truncated),
                "stop_source": stop_source,
                "successful_stop": successful_stop,
                "executed_trace_sha256": _json_hash(trace),
                "candidate_counts": dict(candidate_counts),
                "primitive_counts": dict(primitive_counts),
                "waypoint_macro_count": int(waypoint_macros),
                "path_length_m": float(metrics["path_length"]),
                "collision_count": int(round(float(metrics["collision_count"]))),
                "blocked_forward_count": int(blocked_count),
                "episode_return": float(episode_return),
                "geodesic_progress_m": float(initial_distance - final_ne),
                "post_shadow_stop_primitive_count": 0,
                "external_stop_actor_credit_count": 0,
                "sync": dict(sync),
                "decision_receipts": decision_receipts,
            }
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()

    def _bc_update_phase(
        self,
        rows: Sequence[Mapping],
        *,
        optimizer,
        updates: int,
        generator_seed: int,
    ) -> dict:
        import torch

        if not rows:
            raise LiveProviderError("A1 BC collection produced no supervision rows")
        generator = torch.Generator(device="cpu").manual_seed(
            int(generator_seed)
        )
        losses = []
        finite_gradients = True
        ever_nonzero: set[str] = set()
        if int(self.manifest["budget"]["bc_batch_size"]) != 1:
            raise LiveProviderError("accepted A1 parent policy requires BC batch size one")
        for _ in range(updates):
            row = rows[
                int(torch.randint(len(rows), (1,), generator=generator).item())
            ]
            target_candidate = torch.tensor(
                [int(row["target_candidate"])], dtype=torch.int64
            )
            target_waypoint = torch.tensor(
                [list(row["target_waypoint"])], dtype=torch.float32
            )
            output = self.policy(
                operation="bc_loss",
                forward_inputs=row["forward_inputs"],
                target_candidate=target_candidate,
                target_waypoint=target_waypoint,
                waypoint_coef=float(self.manifest["budget"]["bc_waypoint_coef"]),
            )
            loss = output["loss"]
            if not bool(torch.isfinite(loss).all().item()):
                raise LiveProviderError("A1 BC loss became non-finite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            finite, nonzero = _gradient_groups(self.policy)
            finite_gradients = finite_gradients and finite
            ever_nonzero.update(nonzero)
            if not finite_gradients:
                raise LiveProviderError("A1 BC gradient became non-finite")
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        return {
            "updates": updates,
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "finite_losses": all(math.isfinite(value) for value in losses),
            "finite_gradients": finite_gradients,
            "nonzero_gradient_groups": ever_nonzero,
        }

    def _audit_bc(
        self,
        rows: Sequence[Mapping],
        *,
        phases: Sequence[Mapping],
        head_before: Mapping[str, object],
        backbone_before,
    ) -> dict:
        stop_count = sum(bool(row["stop"]) for row in rows)
        waypoint_count = len(rows) - stop_count
        if stop_count < 1 or waypoint_count < 1:
            raise LiveProviderError("A1 BC rows lack STOP or waypoint support")
        if _parameter_versions(self.generation_adapter.model) != backbone_before:
            raise LiveProviderError("frozen clean-e48 parameter identity/version changed")
        changed = _changed_groups(self.policy, head_before)
        ever_nonzero = set().union(
            *(set(phase["nonzero_gradient_groups"]) for phase in phases)
        )
        required = set(self.contract["required_trainable_groups"]) - {"value_head"}
        if not required.issubset(changed) or not required.issubset(ever_nonzero):
            missing_change = sorted(required - set(changed))
            missing_gradient = sorted(required - ever_nonzero)
            raise LiveProviderError(
                f"A1 BC trainable audit failed: unchanged={missing_change}, "
                f"zero_gradient={missing_gradient}"
            )
        return {
            "updates": sum(int(phase["updates"]) for phase in phases),
            "initial_loss": float(phases[0]["initial_loss"]),
            "final_loss": float(phases[-1]["final_loss"]),
            "finite_losses": all(bool(phase["finite_losses"]) for phase in phases),
            "finite_gradients": all(
                bool(phase["finite_gradients"]) for phase in phases
            ),
            "changed_parameter_groups": changed,
            "label_support": {
                "stop": stop_count,
                "non_stop": waypoint_count,
                "waypoint": waypoint_count,
            },
        }

    def _save_and_restore_bc(self) -> dict:
        from navrl.stack import load_checkpoint_verified

        state = self._new_state()
        adapter = self.contextual_module.StreamVLNContextualTopologyWaypointPolicyAdapter(
            self.policy, state, mode="eval", deterministic=True, root_forward=self.policy
        )
        codec = self.contextual_module.StreamVLNContextualHeadCheckpointAdapter(adapter)
        checkpoint = self.output_dir / "bc_checkpoint"
        saved_files = _save_checkpoint_atomic_py39(codec, checkpoint)
        restored_policy = self._new_policy()
        restored_state = self._new_state()
        restored_adapter = self.contextual_module.StreamVLNContextualTopologyWaypointPolicyAdapter(
            restored_policy,
            restored_state,
            mode="eval",
            deterministic=True,
            root_forward=restored_policy,
        )
        loaded = load_checkpoint_verified(
            self.contextual_module.StreamVLNContextualHeadCheckpointAdapter(restored_adapter),
            checkpoint,
        )
        if saved_files != dict(loaded.files_sha256):
            raise LiveProviderError("A1 BC checkpoint restore receipt drifted")
        for name, value in self.policy.state_dict().items():
            if not value.detach().cpu().equal(
                restored_policy.state_dict()[name].detach().cpu()
            ):
                raise LiveProviderError(f"A1 BC restored tensor differs: {name}")
        protocol = importlib.import_module("run_streamvln_a1_contextual_hybrid")
        return {
            "path": str(checkpoint),
            "sha256": protocol._artifact_sha256(checkpoint),
            "restore_exact": True,
        }

    def _gate_a_receipt(self, payload: Mapping) -> dict:
        protocol = importlib.import_module("run_streamvln_a1_contextual_hybrid")
        normalized = protocol.normalize_evaluation(
            payload["bc_evaluation"], self.contract["development"], "bc_evaluation"
        )
        gate = protocol.adjudicate_gate_a(normalized, self.manifest["gate_a"])
        receipt = {
            "schema_name": "streamvln_a1_gate_a_receipt",
            "schema_version": 1,
            "gate": gate,
            "aggregate": normalized["aggregate"],
            "bc_checkpoint_sha256": payload["bc_training"]["checkpoint"]["sha256"],
            "ppo_initialized": False,
        }
        receipt["receipt_sha256"] = _json_hash(receipt)
        path = self.output_dir / "gate_a_receipt.json"
        if path.exists():
            raise FileExistsError(f"refusing to replace Gate A receipt: {path}")
        path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return gate

    def execute(self, *, manifest: Mapping, contract: Mapping, args) -> Mapping:
        if manifest is not self.manifest or contract is not self.contract or args is not self.args:
            raise LiveProviderError("A1 provider invocation identity drifted")
        self._load_once()
        import torch

        backbone_before = _parameter_versions(self.generation_adapter.model)
        head_before = _head_snapshot(self.policy)
        optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=float(self.manifest["budget"]["bc_learning_rate"]),
            weight_decay=float(self.manifest["budget"]["bc_weight_decay"]),
        )
        expert_collection = [
            self._run_episode(spec=spec, mode="expert_collect")
            for spec in self.contract["train"]
        ]
        expert_rows = [
            row for episode in expert_collection for row in episode["rows"]
        ]
        phase_expert = self._bc_update_phase(
            expert_rows,
            optimizer=optimizer,
            updates=int(self.manifest["budget"]["bc_expert_warmup_steps"]),
            generator_seed=int(self.manifest["budget"]["seed"]),
        )
        dagger_state = self._new_state()
        dagger_adapter = self.contextual_module.StreamVLNContextualTopologyWaypointPolicyAdapter(
            self.policy,
            dagger_state,
            mode="eval",
            deterministic=True,
            root_forward=self.policy,
        )
        dagger_collection = [
            self._run_episode(
                spec=spec,
                mode="dagger_collect",
                policy_adapter=dagger_adapter,
            )
            for spec in self.contract["train"]
        ]
        dagger_rows = [
            row for episode in dagger_collection for row in episode["rows"]
        ]
        rows = expert_rows + dagger_rows
        phase_dagger = self._bc_update_phase(
            rows,
            optimizer=optimizer,
            updates=int(self.manifest["budget"]["bc_dagger_steps"]),
            generator_seed=int(self.manifest["budget"]["seed"]) + 1,
        )
        bc = self._audit_bc(
            rows,
            phases=(phase_expert, phase_dagger),
            head_before=head_before,
            backbone_before=backbone_before,
        )
        if bc["updates"] != int(self.manifest["budget"]["bc_supervision_steps"]):
            raise LiveProviderError("A1 phased BC updates do not sum to 1,000")
        checkpoint = self._save_and_restore_bc()
        eval_state = self._new_state()
        eval_adapter = self.contextual_module.StreamVLNContextualTopologyWaypointPolicyAdapter(
            self.policy,
            eval_state,
            mode="eval",
            deterministic=True,
            root_forward=self.policy,
        )
        development = [
            self._run_episode(
                spec=spec, mode="bc_eval", policy_adapter=eval_adapter
            )
            for spec in self.contract["development"]
        ]
        navrl_stack_sha256_after = _python_tree_sha256(self.navrl_stack_root)
        waypoint_compiler_sha256_after = _file_sha256(self.waypoint_compiler_path)
        if navrl_stack_sha256_after != self.navrl_stack_sha256_before:
            raise LiveProviderError("navrl.stack Python source changed during Gate A")
        if waypoint_compiler_sha256_after != self.waypoint_compiler_sha256_before:
            raise LiveProviderError("waypoint compiler source changed during Gate A")
        collection = expert_collection + dagger_collection
        sync = Counter()
        for episode in collection:
            sync.update(episode["sync"])
        traversals = []
        for traversal_index, episode in enumerate(collection):
            identity = self.contract["train"][traversal_index % 8]
            traversals.append(
                {
                    **dict(identity),
                    "pass": "expert" if traversal_index < 8 else "dagger",
                    "traversal_index": traversal_index,
                    "decision_receipts": list(episode["decision_receipts"]),
                }
            )
        payload = {
            "source_audit": {
                "streamvln_commit": self.manifest["source_contract"]["streamvln_commit"],
                "feature_provider": "StreamVLNContextualFeatureProvider",
                "feature_dim": 3604,
                "feature_schema_sha256": self.manifest["frozen_prior"]["schema_sha256"],
                "feature_v2_enabled": True,
                "frozen_prior": True,
                "ttt_or_secondary_loaded": False,
                "model_load_count": self._model_load_count,
                "official_shadow_stop_mode": "synchronized_physical_boundary_override",
                "split_reads": dict(self._split_reads),
            },
            "bc_training": {
                "checkpoint": checkpoint,
                "supervision_steps": int(self.manifest["budget"]["bc_supervision_steps"]),
                "episode_identities": [dict(row) for row in self.contract["train"]],
                "traversals": traversals,
                "objective": dict(self.manifest["bc_objective"]),
                "finite_losses": bool(bc["finite_losses"]),
                "finite_gradients": bool(bc["finite_gradients"]),
                "frozen_prior_unchanged": True,
                "frozen_stack_core_unchanged": True,
                "frozen_stack_core_sha256": self.navrl_stack_sha256_before,
                "waypoint_compiler_sha256": self.waypoint_compiler_sha256_before,
                "changed_parameter_groups": list(bc["changed_parameter_groups"]),
                "label_support": dict(bc["label_support"]),
                "official_stop_sync": dict(sync),
                "initial_loss": float(bc["initial_loss"]),
                "final_loss": float(bc["final_loss"]),
            },
            "bc_evaluation": {"episodes": development},
            "ppo_training": None,
            "matched_evaluation": None,
        }
        gate_a = self._gate_a_receipt(payload)
        if not gate_a["pass"]:
            return payload
        raise PPOIntegrationBlocked(
            "Gate A passed and gate_a_receipt.json was saved, but PPO was not "
            "initialized in this process: the official Feature-v2 factory is pinned "
            "to Torch 2.1.2/Transformers 4.45.1, while RLinf FSDPModelManager requires "
            "the Torch 2.6 RLinf environment. Continue through a two-process small-head "
            "service: keep Feature-v2/Habitat in this process, send detached 3604-D "
            "features plus causal topology/action records to the RLinf process, and "
            "run sample/replay/bootstrap only through manager.model. External STOP "
            "decisions remain outside the actor batch; mid-macro partial prefixes stay "
            "in executor audit and their sampled decision is dropped."
        )


def create_provider(*, manifest: Mapping, contract: Mapping, args):
    return CanonicalA1LiveProvider(manifest=manifest, contract=contract, args=args)


__all__ = [
    "CanonicalA1LiveProvider",
    "LiveProviderError",
    "PPOIntegrationBlocked",
    "create_provider",
]
