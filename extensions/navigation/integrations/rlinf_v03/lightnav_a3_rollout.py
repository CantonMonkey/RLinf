#!/usr/bin/env python3
"""Collect or evaluate LightNav A3 hybrid actions on train-only R2R.

The script talks to an isolated official Habitat service and the existing
frozen LightNav WebSocket service.  It never stores RGB frames, instructions,
or raw model text.  Simulator goal distance is used only for reward/metrics;
the policy feature vector contains the frozen proposal, action history, and a
causal odometry topology summary.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Mapping, Sequence

import numpy as np

from lightnav_a3_policy import (
    CausalTopologyState,
    GATE_NAMES,
    NumpyLightNavPolicy,
    STOP_GATE,
    apply_policy_action,
    build_features,
    initialize_checkpoint,
    sha256_file,
)
from lightnav_l0_native_probe import (
    HabitatProtocolError,
    LightNavProtocolError,
    LightNavWebSocketClient,
    VelocitySpec,
    ZmqHabitatClient,
    _collision_observation,
    _mapping,
    _metric,
    _optional_metric,
    atomic_write_json,
    classify_velocity_action,
    encode_rgb_jpeg_base64,
    extract_instruction,
    latency_summary,
    redact_endpoint,
    waypoint_to_velocity_action,
    zero_waypoint_action,
)
from navrl.stack.returns import duration_aware_gae


SCHEMA_NAME = "lightnav_a3_rollout"
SCHEMA_VERSION = 1


class A3RolloutError(RuntimeError):
    """Raised when a rollout violates the frozen A3 protocol."""


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise A3RolloutError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise A3RolloutError(f"{name} must be finite")
    return result


def _bool_success(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return bool(round(_finite(value, "success")))


def _physical_velocity(action: Mapping[str, object], spec: VelocitySpec) -> tuple[float, float]:
    args = _mapping(action.get("action_args"), "velocity action args")
    linear_raw = _finite(args.get("linear_velocity"), "linear_velocity")
    angular_raw = _finite(args.get("angular_velocity"), "angular_velocity")
    linear = spec.linear_range_mps[0] + (linear_raw + 1.0) * 0.5 * (
        spec.linear_range_mps[1] - spec.linear_range_mps[0]
    )
    angular = spec.angular_range_dps[0] + (angular_raw + 1.0) * 0.5 * (
        spec.angular_range_dps[1] - spec.angular_range_dps[0]
    )
    return linear, angular


def _validate_velocity_contract(spec: VelocitySpec) -> None:
    """Fail closed if Habitat cannot execute the frozen waypoint bounds exactly."""

    displacement = (
        spec.linear_range_mps[0] * spec.time_step_s,
        spec.linear_range_mps[1] * spec.time_step_s,
    )
    yaw_step = (
        spec.angular_range_dps[0] * spec.time_step_s,
        spec.angular_range_dps[1] * spec.time_step_s,
    )
    expected_displacement = (0.0, 0.25)
    expected_yaw_step = (-30.0, 30.0)
    if not all(
        math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-5)
        for actual, expected in zip(displacement, expected_displacement)
    ) or not all(
        math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-5)
        for actual, expected in zip(yaw_step, expected_yaw_step)
    ):
        raise HabitatProtocolError(
            "A3 requires a [0, 0.25] m and [-30, 30] degree per-step velocity contract"
        )


def _normalized_command(action: Mapping[str, object]) -> tuple[float, float]:
    args = _mapping(action.get("action_args"), "velocity action args")
    return (
        _finite(args.get("linear_velocity"), "linear_velocity"),
        _finite(args.get("angular_velocity"), "angular_velocity"),
    )


def _reward(
    *,
    previous_distance: float,
    current_distance: float,
    success: bool,
    collision_event: bool,
    stopped: bool,
    timeout: bool,
) -> tuple[float, dict[str, float]]:
    components = {
        "geodesic_progress": previous_distance - current_distance,
        "success": 10.0 if success else 0.0,
        "step": -0.01,
        "collision": -0.2 if collision_event else 0.0,
        "failed_stop": -1.0 if stopped and not success else 0.0,
        "timeout": -1.0 if timeout else 0.0,
    }
    return sum(components.values()), components


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> str:
    staging = path.with_name(path.name + ".staging")
    with staging.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    staging.replace(path)
    return sha256_file(path)


def _scene_name(value: object) -> str:
    text = str(value).replace("\\", "/")
    filename = text.rsplit("/", 1)[-1]
    return filename[:-4] if filename.endswith(".glb") else filename


def _load_split_manifest(
    path: Path,
    *,
    expected_sha256: str,
    split_label: str,
    episodes: int,
) -> tuple[dict[str, object], list[dict[str, object]], str]:
    observed_sha256 = sha256_file(path)
    if observed_sha256 != expected_sha256:
        raise A3RolloutError("split manifest SHA-256 differs")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_name") != "lightnav_a3_train_scene_split"
        or manifest.get("schema_version") != 1
        or manifest.get("upstream_split_read") != "train"
        or manifest.get("validation_split_reads") != 0
    ):
        raise A3RolloutError("split manifest contract differs")
    split = manifest.get("splits", {}).get(split_label)
    if not isinstance(split, dict) or split.get("habitat_split") != "train":
        raise A3RolloutError("requested split label is absent from the train manifest")
    expected = split.get("episodes")
    if not isinstance(expected, list) or len(expected) != episodes:
        raise A3RolloutError("requested episode count differs from the split manifest")
    return split, expected, observed_sha256


def _load_run_manifest(
    path: Path,
    *,
    expected_sha256: str,
    split_manifest_sha256: str,
) -> tuple[dict[str, object], str]:
    observed_sha256 = sha256_file(path)
    if observed_sha256 != expected_sha256:
        raise A3RolloutError("run manifest SHA-256 differs")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_name") != "lightnav_a3_run_manifest"
        or manifest.get("schema_version") != 1
        or manifest.get("upstream_split") != "train"
        or manifest.get("validation_split_reads") != 0
        or manifest.get("split_manifest", {}).get("sha256") != split_manifest_sha256
    ):
        raise A3RolloutError("outer run manifest contract differs")
    return manifest, observed_sha256


def _collision_event(
    info: Mapping[str, object],
    previous_cumulative: float | None,
) -> tuple[bool, float | None, str]:
    cumulative, event, source = _collision_observation(info)
    if event is not None:
        return bool(event), cumulative if cumulative is not None else previous_cumulative, source
    if cumulative is not None:
        collided = previous_cumulative is not None and cumulative > previous_cumulative
        return collided, cumulative, source
    return False, previous_cumulative, source


def _episode_metrics(
    *,
    reset_info: Mapping[str, object],
    final_info: Mapping[str, object],
    initial_distance: float,
    minimum_distance: float,
    initial_path: float,
    step_count: int,
    collision_count: float | None,
    blocked_steps: int,
    stop_source: str,
    timeout: bool,
    action_counts: Counter[str],
    gate_counts: Counter[str],
) -> dict[str, object]:
    final_distance = _metric(
        final_info,
        ("distance_to_goal", "goal_distance", "navigation_error"),
        "navigation error",
    )
    path_length = _optional_metric(final_info, ("path_length",))
    if path_length is None:
        raise HabitatProtocolError("A3 requires Habitat path_length")
    success = _bool_success(final_info.get("success", False))
    spl = _finite(final_info.get("spl", 0.0), "spl")
    scene = str(final_info.get("scene_id", reset_info.get("scene_id", "")))
    episode_id = str(final_info.get("episode_id", reset_info.get("episode_id", "")))
    if not scene or not episode_id:
        raise HabitatProtocolError("Habitat info lacks scene_id or episode_id")
    return {
        "scene_id": scene,
        "episode_id": episode_id,
        "success": success,
        "spl": spl,
        "initial_navigation_error_m": initial_distance,
        "final_navigation_error_m": final_distance,
        "minimum_navigation_error_m": minimum_distance,
        "geodesic_progress_m": initial_distance - final_distance,
        "path_length_m": path_length,
        "path_length_delta_m": path_length - initial_path,
        "collision_count": collision_count,
        "collision_source": "habitat_collisions" if collision_count is not None else "unavailable",
        "blocked_step_count": blocked_steps,
        "steps": step_count,
        "stop_source": stop_source,
        "stop_called": stop_source != "none",
        "successful_stop": stop_source != "none" and success,
        "failed_stop": stop_source != "none" and not success,
        "timeout": timeout,
        "action_distribution": dict(sorted(action_counts.items())),
        "gate_distribution": dict(sorted(gate_counts.items())),
    }


def _finish_gae(rows: list[dict[str, object]], *, gamma: float, gae_lambda: float) -> None:
    if not rows:
        return
    values = [float(row["value"]) for row in rows]
    next_values = values[1:] + [0.0]
    output = duration_aware_gae(
        [float(row["reward"]) for row in rows],
        values,
        next_values,
        [1] * len(rows),
        [bool(row["terminated"]) for row in rows],
        [bool(row["truncated"]) for row in rows],
        gamma,
        gae_lambda,
    )
    for row, next_value, advantage, target_return in zip(
        rows, next_values, output.advantages, output.returns
    ):
        row["next_value"] = next_value
        row["advantage"] = advantage
        row["return"] = target_return


def run_episode(
    env: ZmqHabitatClient,
    model: LightNavWebSocketClient,
    policy: NumpyLightNavPolicy,
    *,
    rng: np.random.Generator,
    deterministic: bool,
    max_steps: int,
    jpeg_quality: int,
    require_collisions: bool,
    gamma: float,
    gae_lambda: float,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]], list[float]]:
    observation, reset_info = env.reset()
    model.reset()
    instruction = extract_instruction(observation)
    instruction_sha = hashlib.sha256(instruction.encode("utf-8")).hexdigest()
    velocity = VelocitySpec.from_info(reset_info)
    _validate_velocity_contract(velocity)
    initial_distance = _metric(
        reset_info,
        ("distance_to_goal", "goal_distance", "navigation_error"),
        "navigation error",
    )
    previous_distance = initial_distance
    minimum_distance = initial_distance
    initial_path = _optional_metric(reset_info, ("path_length",)) or 0.0
    state = CausalTopologyState(previous_path_length_m=initial_path)
    previous_collision, _, reset_collision_source = _collision_observation(reset_info)
    collision_cumulative = previous_collision
    if require_collisions and reset_collision_source == "unavailable":
        raise HabitatProtocolError("A3 formal run requires Habitat Collisions metric")

    actor_rows: list[dict[str, object]] = []
    gae_rows: list[dict[str, object]] = []
    trace: list[dict[str, object]] = []
    action_counts: Counter[str] = Counter()
    gate_counts: Counter[str] = Counter()
    latencies: list[float] = []
    blocked_steps = 0
    final_info: Mapping[str, object] = reset_info
    stop_source = "none"
    timeout = False
    seq = 0

    for step_index in range(max_steps):
        external_forced_stop = step_index == max_steps - 1
        prediction = None
        features = None
        actor_action = None
        actor_eligible = False
        native_stop = False
        if external_forced_stop:
            action = zero_waypoint_action(velocity)
            resolved_waypoint = (0.0, 0.0, 0.0)
            stop_source = "budget_stop"
            timeout = True
            value = 0.0
            gate_name = "EXTERNAL_BUDGET_STOP"
        else:
            if "rgb" not in observation:
                raise HabitatProtocolError("Habitat observation has no RGB frame")
            image_b64 = encode_rgb_jpeg_base64(observation["rgb"], quality=jpeg_quality)
            prediction = model.predict(seq=seq, image_b64=image_b64, instruction=instruction)
            seq += 1
            latencies.append(prediction.latency_ms)
            features = build_features(
                prediction.waypoints,
                selected_index=prediction.selected_index,
                selected_waypoint=prediction.selected_waypoint,
                model_stop=prediction.stop,
                step_index=step_index,
                max_steps=max_steps,
                state=state,
            )
            if prediction.stop:
                # Native STOP is frozen-prior behavior.  It is executed before
                # actor sampling so KEEP/STOP aliases never receive PPO credit.
                _, _, value = policy.forward(features)
                action = zero_waypoint_action(velocity)
                resolved_waypoint = (0.0, 0.0, 0.0)
                stop_source = "native_lightnav_stop"
                native_stop = True
                gate_name = "EXTERNAL_NATIVE_STOP"
            else:
                actor_action = policy.act(features, rng=rng, deterministic=deterministic)
                actor_eligible = True
                value = actor_action.value
                resolved_waypoint = apply_policy_action(prediction.selected_waypoint, actor_action)
                action = waypoint_to_velocity_action(resolved_waypoint, velocity)
                gate_name = actor_action.gate_name
                gate_counts[gate_name] += 1
                if actor_action.gate == STOP_GATE:
                    stop_source = "learned_stop"

        normalized_linear, normalized_angular = _normalized_command(action)
        physical_linear, physical_angular = _physical_velocity(action, velocity)
        category = classify_velocity_action(action, velocity)
        action_counts[category] += 1
        observation, _, terminated, truncated, final_info = env.step(action)
        if not isinstance(observation, Mapping) or not isinstance(final_info, Mapping):
            raise HabitatProtocolError("Habitat step returned invalid observation/info")
        current_distance = _metric(
            final_info,
            ("distance_to_goal", "goal_distance", "navigation_error"),
            "navigation error",
        )
        minimum_distance = min(minimum_distance, current_distance)
        path_length = _optional_metric(final_info, ("path_length",))
        if path_length is None:
            raise HabitatProtocolError("A3 requires Habitat path_length")
        collided, collision_cumulative, collision_source = _collision_event(
            final_info, collision_cumulative
        )
        if require_collisions and collision_source == "unavailable":
            raise HabitatProtocolError("A3 lost Habitat Collisions metric during rollout")
        stopped = category == "STOP"
        success = _bool_success(final_info.get("success", False))
        effective_timeout = timeout or (bool(truncated) and not stopped)
        timeout = effective_timeout
        reward, reward_components = _reward(
            previous_distance=previous_distance,
            current_distance=current_distance,
            success=success,
            collision_event=collided,
            stopped=stopped,
            timeout=effective_timeout,
        )

        previous_path = state.previous_path_length_m
        selected_for_state = (
            prediction.selected_waypoint
            if prediction is not None
            else state.previous_selected_waypoint
        )
        state.update(
            normalized_linear=normalized_linear,
            normalized_angular=normalized_angular,
            physical_linear_mps=physical_linear,
            physical_angular_dps=physical_angular,
            time_step_s=velocity.time_step_s,
            gate=actor_action.gate if actor_action is not None else state.previous_gate,
            selected_waypoint=selected_for_state,
            path_length_m=path_length,
        )
        path_delta = path_length - previous_path
        if state.blocked_last:
            blocked_steps += 1

        episode_id = str(final_info.get("episode_id", reset_info.get("episode_id", "")))
        scene_id = str(final_info.get("scene_id", reset_info.get("scene_id", "")))
        row: dict[str, object] = {
            "episode_id": episode_id,
            "scene_id": scene_id,
            "step": step_index,
            "actor_eligible": actor_eligible,
            "native_stop": native_stop,
            "external_forced_stop": external_forced_stop,
            "value": value,
            "reward": reward,
            "reward_components": reward_components,
            # Stack termination identities are mutually exclusive.  A physical
            # zero-velocity budget stop may make Habitat say terminated=True,
            # but the policy boundary remains a truncation.
            "terminated": bool(stopped and not effective_timeout),
            "truncated": bool(effective_timeout),
            "navigation_error_m": current_distance,
            "path_length_m": path_length,
            "path_length_delta_m": path_delta,
            "collision_event": collided,
            "collision_count": collision_cumulative,
            "blocked": state.blocked_last,
            "gate": gate_name,
            "resolved_waypoint": list(resolved_waypoint),
            "normalized_command": [normalized_linear, normalized_angular],
            "action_category": category,
        }
        if prediction is not None:
            row.update(
                {
                    "proposal": list(prediction.selected_waypoint),
                    "proposal_index": prediction.selected_index,
                    "proposal_trajectory_sha256": hashlib.sha256(
                        json.dumps(prediction.waypoints, separators=(",", ":")).encode("utf-8")
                    ).hexdigest(),
                    "model_latency_ms": prediction.latency_ms,
                    "raw_text_sha256": prediction.raw_text_sha256,
                }
            )
        if actor_action is not None and features is not None:
            row.update(
                {
                    "features": features.astype(float).tolist(),
                    "gate_index": actor_action.gate,
                    "gate_probabilities": list(actor_action.gate_probabilities),
                    "raw_residual": list(actor_action.raw_residual),
                    "bounded_residual": list(actor_action.bounded_residual),
                    "old_joint_log_prob": actor_action.joint_log_prob,
                    "old_entropy": actor_action.entropy,
                    "policy_version": int(policy.metadata["policy_version"]),
                }
            )
        gae_rows.append(row)
        if actor_eligible:
            actor_rows.append(row)
        trace.append({key: value for key, value in row.items() if key != "features"})
        previous_distance = current_distance

        if stopped and not (terminated or truncated):
            raise HabitatProtocolError("physical STOP did not terminate Habitat")
        if terminated or truncated or stopped or effective_timeout:
            break

    if not gae_rows:
        raise A3RolloutError("episode produced no physical step")
    _finish_gae(gae_rows, gamma=gamma, gae_lambda=gae_lambda)
    # The same dictionaries are shared by actor_rows, so GAE fields are now visible.
    metrics = _episode_metrics(
        reset_info=reset_info,
        final_info=final_info,
        initial_distance=initial_distance,
        minimum_distance=minimum_distance,
        initial_path=initial_path,
        step_count=len(gae_rows),
        collision_count=collision_cumulative,
        blocked_steps=blocked_steps,
        stop_source=stop_source,
        timeout=timeout,
        action_counts=action_counts,
        gate_counts=gate_counts,
    )
    metrics["instruction_sha256"] = instruction_sha
    metrics["actor_decision_count"] = len(actor_rows)
    return metrics, actor_rows, trace, latencies


def aggregate(episodes: Sequence[Mapping[str, object]], latencies: Sequence[float]) -> dict[str, object]:
    count = len(episodes)
    if count < 1:
        raise A3RolloutError("cannot aggregate zero episodes")
    actions: Counter[str] = Counter()
    gates: Counter[str] = Counter()
    for row in episodes:
        actions.update(row["action_distribution"])
        gates.update(row["gate_distribution"])
    collision_available = all(row["collision_count"] is not None for row in episodes)
    return {
        "episode_count": count,
        "success_count": sum(bool(row["success"]) for row in episodes),
        "success_rate": sum(bool(row["success"]) for row in episodes) / count,
        "mean_spl": sum(float(row["spl"]) for row in episodes) / count,
        "mean_final_navigation_error_m": sum(float(row["final_navigation_error_m"]) for row in episodes) / count,
        "mean_geodesic_progress_m": sum(float(row["geodesic_progress_m"]) for row in episodes) / count,
        "mean_path_length_m": sum(float(row["path_length_m"]) for row in episodes) / count,
        "collision_metric_available": collision_available,
        "collision_count": (
            sum(float(row["collision_count"]) for row in episodes)
            if collision_available
            else None
        ),
        "blocked_step_count": sum(int(row["blocked_step_count"]) for row in episodes),
        "native_stop_count": sum(row["stop_source"] == "native_lightnav_stop" for row in episodes),
        "learned_stop_count": sum(row["stop_source"] == "learned_stop" for row in episodes),
        "timeout_count": sum(bool(row["timeout"]) for row in episodes),
        "actor_decision_count": sum(int(row["actor_decision_count"]) for row in episodes),
        "action_distribution": dict(sorted(actions.items())),
        "gate_distribution": dict(sorted(gates.items())),
        "model_latency": latency_summary(latencies),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--habitat-server", required=True)
    result.add_argument("--lightnav-server", required=True)
    result.add_argument("--checkpoint", required=True, type=Path)
    result.add_argument("--expected-checkpoint-sha256")
    result.add_argument("--initialize-checkpoint", action="store_true")
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument("--mode", choices=("train", "eval"), required=True)
    result.add_argument("--split-manifest", required=True, type=Path)
    result.add_argument("--expected-split-manifest-sha256", required=True)
    result.add_argument("--run-manifest", required=True, type=Path)
    result.add_argument("--expected-run-manifest-sha256", required=True)
    result.add_argument(
        "--split-label",
        required=True,
        choices=("train_update", "train_development"),
    )
    result.add_argument("--episodes", type=int, required=True)
    result.add_argument("--max-steps", type=int, default=160)
    result.add_argument("--jpeg-quality", type=int, default=90)
    result.add_argument("--seed", type=int, default=2026093003)
    result.add_argument("--gamma", type=float, default=0.99)
    result.add_argument("--gae-lambda", type=float, default=0.95)
    result.add_argument("--client-id", default="rlinf-lightnav-a3")
    result.add_argument("--allow-missing-collisions", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.episodes < 1 or args.max_steps < 2:
        raise A3RolloutError("episodes must be positive and max_steps at least two")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.initialize_checkpoint:
        if args.checkpoint.exists():
            raise A3RolloutError("initial checkpoint target already exists")
        initialize_checkpoint(args.checkpoint, seed=args.seed)
    checkpoint_sha = sha256_file(args.checkpoint)
    if args.expected_checkpoint_sha256 and checkpoint_sha != args.expected_checkpoint_sha256:
        raise A3RolloutError("checkpoint SHA-256 differs from the requested identity")
    policy = NumpyLightNavPolicy.from_checkpoint(args.checkpoint)
    split_contract, expected_episodes, split_manifest_sha = _load_split_manifest(
        args.split_manifest,
        expected_sha256=args.expected_split_manifest_sha256,
        split_label=args.split_label,
        episodes=args.episodes,
    )
    run_manifest, run_manifest_sha = _load_run_manifest(
        args.run_manifest,
        expected_sha256=args.expected_run_manifest_sha256,
        split_manifest_sha256=split_manifest_sha,
    )
    if (
        int(policy.metadata["policy_version"]) == 0
        and run_manifest.get("policy", {}).get("initial_checkpoint_sha256")
        != checkpoint_sha
    ):
        raise A3RolloutError("version-0 checkpoint differs from the outer run manifest")
    if (
        run_manifest.get("policy", {}).get("seed") != args.seed
        or run_manifest.get("habitat", {}).get("max_steps") != args.max_steps
    ):
        raise A3RolloutError("rollout settings differ from the outer run manifest")
    rng = np.random.default_rng(args.seed)
    deterministic = args.mode == "eval"
    env = ZmqHabitatClient(args.habitat_server)
    model = LightNavWebSocketClient(args.lightnav_server, timeout_s=600.0)
    model.open()
    started = time.time()
    status = "invalid"
    try:
        model.login(args.client_id)
        episode_rows: list[dict[str, object]] = []
        actor_rows: list[dict[str, object]] = []
        traces: list[dict[str, object]] = []
        latencies: list[float] = []
        seen: set[tuple[str, str]] = set()
        progress_path = args.output_dir / "episode_progress.jsonl"
        for episode_index in range(args.episodes):
            metrics, decisions, trace, episode_latencies = run_episode(
                env,
                model,
                policy,
                rng=rng,
                deterministic=deterministic,
                max_steps=args.max_steps,
                jpeg_quality=args.jpeg_quality,
                require_collisions=not args.allow_missing_collisions,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
            )
            identity = (str(metrics["scene_id"]), str(metrics["episode_id"]))
            if identity in seen:
                raise A3RolloutError("Habitat repeated an episode before the frozen count")
            seen.add(identity)
            expected = expected_episodes[episode_index]
            if (
                str(metrics["episode_id"]) != str(expected.get("episode_id"))
                or _scene_name(metrics["scene_id"]) != str(expected.get("scene"))
                or str(metrics["instruction_sha256"])
                != str(expected.get("instruction_text_sha256"))
            ):
                raise A3RolloutError(
                    f"episode order/scene/instruction differs at position {episode_index}"
                )
            metrics["episode_order"] = episode_index
            for decision in decisions:
                decision["episode_order"] = episode_index
            for step in trace:
                step["episode_order"] = episode_index
            episode_rows.append(metrics)
            actor_rows.extend(decisions)
            traces.extend(trace)
            latencies.extend(episode_latencies)
            with progress_path.open("a", encoding="utf-8", newline="\n") as progress:
                progress.write(
                    json.dumps(metrics, sort_keys=True, separators=(",", ":")) + "\n"
                )
                progress.flush()

        decisions_path = args.output_dir / "actor_decisions.jsonl"
        trace_path = args.output_dir / "execution_trace.jsonl"
        decisions_sha = _write_jsonl(decisions_path, actor_rows)
        trace_sha = _write_jsonl(trace_path, traces)
        receipt = {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "mode": args.mode,
            "deterministic": deterministic,
            "upstream_split": "train",
            "experiment_split_label": args.split_label,
            "validation_split_reads": 0,
            "split_manifest": {
                "path": str(args.split_manifest.resolve()),
                "sha256": split_manifest_sha,
                "selection_manifest_sha256": json.loads(
                    args.split_manifest.read_text(encoding="utf-8")
                )["selection_manifest_sha256"],
                "dataset_sha256": split_contract["dataset_sha256"],
                "gt_sha256": split_contract["gt_sha256"],
            },
            "run_manifest": {
                "path": str(args.run_manifest.resolve()),
                "sha256": run_manifest_sha,
                "run_id": run_manifest["run_id"],
            },
            "episodes_requested": args.episodes,
            "max_steps": args.max_steps,
            "seed": args.seed,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "checkpoint": {
                "path": str(args.checkpoint.resolve()),
                "sha256": checkpoint_sha,
                "policy_version": int(policy.metadata["policy_version"]),
            },
            "habitat_server": redact_endpoint(args.habitat_server),
            "lightnav_server": redact_endpoint(args.lightnav_server),
            "collision_metric_required": not args.allow_missing_collisions,
            "episodes": episode_rows,
            "aggregate": aggregate(episode_rows, latencies),
            "actor_decisions": {
                "path": str(decisions_path),
                "sha256": decisions_sha,
                "row_count": len(actor_rows),
            },
            "execution_trace": {
                "path": str(trace_path),
                "sha256": trace_sha,
                "row_count": len(traces),
            },
            "started_unix_s": started,
            "completed_unix_s": time.time(),
        }
        atomic_write_json(args.output_dir / "result.json", receipt)
        inventory = {
            path.name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in sorted(args.output_dir.iterdir())
            if path.is_file() and path.name != "COMPLETED.json"
        }
        atomic_write_json(args.output_dir / "artifact_inventory.json", inventory)
        atomic_write_json(
            args.output_dir / "COMPLETED.json",
            {
                "schema_name": "lightnav_a3_completion",
                "schema_version": 1,
                "status": "completed",
                "result_sha256": sha256_file(args.output_dir / "result.json"),
                "inventory_sha256": sha256_file(args.output_dir / "artifact_inventory.json"),
            },
        )
        status = "completed"
        print(json.dumps(receipt["aggregate"], sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001 - invalid receipt is the protocol boundary
        atomic_write_json(
            args.output_dir / "INVALID.json",
            {
                "schema_name": SCHEMA_NAME,
                "schema_version": SCHEMA_VERSION,
                "status": "invalid",
                "mode": args.mode,
                "checkpoint_sha256": checkpoint_sha,
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "started_unix_s": started,
                "completed_unix_s": time.time(),
            },
        )
        print(f"A3 rollout invalid: {type(exc).__name__}: {exc}")
        return 2
    finally:
        model.close()
        env.close()
        if status != "completed":
            # Absence of COMPLETED is intentional and is used by the monitor.
            pass


if __name__ == "__main__":
    raise SystemExit(main())
