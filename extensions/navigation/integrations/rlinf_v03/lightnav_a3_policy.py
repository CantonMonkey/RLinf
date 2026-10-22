"""Portable LightNav A3 proposal-gating policy and causal topology features.

The frozen LightNav service remains the visual-language policy.  This module
owns only a small trainable correction head.  The executable action is a
hybrid distribution over ``KEEP``, ``ADJUST``, and ``STOP`` plus a conditional
Gaussian residual in forward/yaw.  LightNav's lateral proposal is retained as
an input feature, but it is never assigned actor credit because the official
unicycle executor cannot execute lateral motion.

The checkpoint format is NumPy ``.npz`` so the Python 3.9 Habitat process can
collect rollouts without importing the RLinf/Torch runtime.  The A800 trainer
loads the same tensors into an RLinf FSDP root and exports the same format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


SCHEMA_NAME = "lightnav_a3_executable_residual_policy"
SCHEMA_VERSION = 1
FEATURE_SCHEMA = "lightnav_trajectory_temporal_topology_v1"
FEATURE_DIM = 64
TRAJECTORY_WIDTH = 10
HIDDEN_DIM = 256
GATE_NAMES = ("KEEP", "ADJUST", "STOP")
KEEP_GATE = 0
ADJUST_GATE = 1
STOP_GATE = 2
RESIDUAL_LIMITS = np.asarray(
    (0.05, math.radians(10.0)), dtype=np.float32
)  # executable forward m, yaw rad
EXECUTOR_FORWARD_RANGE_M = (0.0, 0.25)
EXECUTOR_YAW_RANGE_RAD = (-math.radians(30.0), math.radians(30.0))


class PolicyContractError(RuntimeError):
    """Raised when a portable checkpoint or action violates the A3 contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metadata(*, seed: int, policy_version: int) -> dict[str, object]:
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "feature_schema": FEATURE_SCHEMA,
        "feature_dim": FEATURE_DIM,
        "trajectory_width": TRAJECTORY_WIDTH,
        "hidden_dim": HIDDEN_DIM,
        "gate_names": list(GATE_NAMES),
        "residual_axes": ["forward_m", "yaw_rad"],
        "residual_limits": RESIDUAL_LIMITS.tolist(),
        "lateral_actor_credit": False,
        "seed": int(seed),
        "policy_version": int(policy_version),
    }


def initialize_parameters(seed: int) -> dict[str, np.ndarray]:
    """Return a conservative native-LightNav initialization.

    KEEP and ADJUST begin with equal logits.  Deterministic evaluation resolves
    the tie to KEEP, exactly reproducing the native proposal.  Stochastic
    collection explores small executable residuals.  STOP starts near 0.05%,
    avoiding the compounding premature-stop rate of a seemingly small 1% step
    probability over a long navigation episode.
    """

    rng = np.random.default_rng(int(seed))
    scale_1 = math.sqrt(2.0 / FEATURE_DIM)
    scale_2 = math.sqrt(2.0 / HIDDEN_DIM)
    result = {
        "w1": rng.normal(0.0, scale_1, (FEATURE_DIM, HIDDEN_DIM)).astype(np.float32),
        "b1": np.zeros((HIDDEN_DIM,), dtype=np.float32),
        "w2": rng.normal(0.0, scale_2, (HIDDEN_DIM, HIDDEN_DIM)).astype(np.float32),
        "b2": np.zeros((HIDDEN_DIM,), dtype=np.float32),
        "gate_w": np.zeros((HIDDEN_DIM, len(GATE_NAMES)), dtype=np.float32),
        "gate_b": np.asarray((0.0, 0.0, -7.6), dtype=np.float32),
        "residual_w": np.zeros((HIDDEN_DIM, 2), dtype=np.float32),
        "residual_b": np.zeros((2,), dtype=np.float32),
        "value_w": np.zeros((HIDDEN_DIM, 1), dtype=np.float32),
        "value_b": np.zeros((1,), dtype=np.float32),
        "log_std": np.asarray((-1.5, -1.5), dtype=np.float32),
    }
    return result


def save_checkpoint(
    path: Path,
    parameters: Mapping[str, np.ndarray],
    *,
    seed: int,
    policy_version: int,
) -> dict[str, object]:
    """Atomically write the portable policy checkpoint and return its receipt."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = initialize_parameters(seed)
    if set(parameters) != set(expected):
        raise PolicyContractError("checkpoint parameter names differ from A3")
    payload: dict[str, np.ndarray] = {}
    for name, template in expected.items():
        value = np.asarray(parameters[name], dtype=np.float32)
        if value.shape != template.shape or not np.isfinite(value).all():
            raise PolicyContractError(f"checkpoint tensor {name} is invalid")
        payload[name] = value
    metadata = _metadata(seed=seed, policy_version=policy_version)
    payload["metadata_json"] = np.asarray(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")), dtype=np.str_
    )
    staging = path.with_name(path.name + ".staging")
    with staging.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    staging.replace(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "metadata": metadata,
    }


def initialize_checkpoint(path: Path, *, seed: int) -> dict[str, object]:
    return save_checkpoint(
        path,
        initialize_parameters(seed),
        seed=seed,
        policy_version=0,
    )


def load_checkpoint(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    path = Path(path).resolve(strict=True)
    if not path.is_file() or path.is_symlink():
        raise PolicyContractError("A3 checkpoint must be a regular non-symlink file")
    with np.load(path, allow_pickle=False) as archive:
        names = set(archive.files)
        parameter_names = tuple(initialize_parameters(0))
        expected_names = set(parameter_names) | {"metadata_json"}
        if names != expected_names:
            raise PolicyContractError("A3 checkpoint members differ from the contract")
        metadata = json.loads(str(archive["metadata_json"].reshape(()).item()))
        parameters = {
            name: np.asarray(archive[name], dtype=np.float32).copy()
            for name in parameter_names
        }
    if not isinstance(metadata, dict):
        raise PolicyContractError("A3 checkpoint metadata is not an object")
    static = _metadata(
        seed=int(metadata.get("seed", -1)),
        policy_version=int(metadata.get("policy_version", -1)),
    )
    if metadata != static:
        raise PolicyContractError("A3 checkpoint metadata differs from the contract")
    template = initialize_parameters(int(metadata["seed"]))
    for name, expected in template.items():
        value = parameters[name]
        if value.shape != expected.shape or not np.isfinite(value).all():
            raise PolicyContractError(f"A3 checkpoint tensor {name} is invalid")
    return parameters, metadata


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    weights = np.exp(shifted, dtype=np.float64)
    return (weights / weights.sum()).astype(np.float64)


def _normal_log_prob(value: np.ndarray, mean: np.ndarray, log_std: np.ndarray) -> float:
    variance_term = np.square((value - mean) / np.exp(log_std))
    return float(
        -0.5 * np.sum(variance_term + 2.0 * log_std + math.log(2.0 * math.pi))
    )


def _categorical_entropy(probabilities: np.ndarray) -> float:
    safe = np.clip(probabilities, 1.0e-12, 1.0)
    return float(-np.sum(safe * np.log(safe)))


@dataclass(frozen=True)
class PolicyAction:
    gate: int
    raw_residual: tuple[float, float]
    bounded_residual: tuple[float, float]
    joint_log_prob: float
    value: float
    entropy: float
    gate_probabilities: tuple[float, float, float]

    @property
    def gate_name(self) -> str:
        return GATE_NAMES[self.gate]


class NumpyLightNavPolicy:
    """Dependency-light forward/sampling implementation for rollout hosts."""

    def __init__(self, parameters: Mapping[str, np.ndarray], metadata: Mapping[str, object]):
        self.parameters = {name: np.asarray(value, dtype=np.float32) for name, value in parameters.items()}
        self.metadata = dict(metadata)

    @classmethod
    def from_checkpoint(cls, path: Path) -> "NumpyLightNavPolicy":
        parameters, metadata = load_checkpoint(path)
        return cls(parameters, metadata)

    def forward(self, features: Sequence[float]) -> tuple[np.ndarray, np.ndarray, float]:
        x = np.asarray(features, dtype=np.float32)
        if x.shape != (FEATURE_DIM,) or not np.isfinite(x).all():
            raise PolicyContractError("A3 policy feature vector is invalid")
        p = self.parameters
        hidden = np.maximum(x @ p["w1"] + p["b1"], 0.0)
        hidden = np.maximum(hidden @ p["w2"] + p["b2"], 0.0)
        logits = hidden @ p["gate_w"] + p["gate_b"]
        residual_mean = hidden @ p["residual_w"] + p["residual_b"]
        value = float((hidden @ p["value_w"] + p["value_b"])[0])
        if not np.isfinite(logits).all() or not np.isfinite(residual_mean).all() or not math.isfinite(value):
            raise PolicyContractError("A3 policy forward produced non-finite values")
        return logits.astype(np.float64), residual_mean.astype(np.float64), value

    def act(
        self,
        features: Sequence[float],
        *,
        rng: np.random.Generator,
        deterministic: bool,
    ) -> PolicyAction:
        logits, mean, value = self.forward(features)
        probabilities = _softmax(logits)
        gate = int(np.argmax(probabilities)) if deterministic else int(rng.choice(3, p=probabilities))
        if deterministic:
            raw = mean.copy()
        else:
            raw = rng.normal(mean, np.exp(self.parameters["log_std"]))
        bounded = np.tanh(raw) * RESIDUAL_LIMITS
        log_prob = float(math.log(max(probabilities[gate], 1.0e-12)))
        gaussian_entropy = float(
            np.sum(self.parameters["log_std"] + 0.5 * math.log(2.0 * math.pi * math.e))
        )
        entropy = _categorical_entropy(probabilities)
        if gate == ADJUST_GATE:
            log_prob += _normal_log_prob(raw, mean, self.parameters["log_std"])
            entropy += gaussian_entropy
        return PolicyAction(
            gate=gate,
            raw_residual=(float(raw[0]), float(raw[1])),
            bounded_residual=(float(bounded[0]), float(bounded[1])),
            joint_log_prob=log_prob,
            value=value,
            entropy=entropy,
            gate_probabilities=tuple(float(item) for item in probabilities),
        )


@dataclass
class CausalTopologyState:
    """Small odometry-only topology summary available on a real robot."""

    x_m: float = 0.0
    y_m: float = 0.0
    yaw_rad: float = 0.0
    nodes: list[tuple[float, float]] = field(default_factory=lambda: [(0.0, 0.0)])
    edge_count: int = 0
    local_visit_count: int = 1
    steps_since_new_node: int = 0
    previous_command: tuple[float, float] = (0.0, 0.0)
    previous_gate: int = KEEP_GATE
    blocked_last: bool = False
    previous_selected_waypoint: tuple[float, float, float] = (0.0, 0.0, 0.0)
    previous_path_length_m: float = 0.0
    last_path_delta_m: float = 0.0

    def update(
        self,
        *,
        normalized_linear: float,
        normalized_angular: float,
        physical_linear_mps: float,
        physical_angular_dps: float,
        time_step_s: float,
        gate: int,
        selected_waypoint: Sequence[float],
        path_length_m: float,
    ) -> None:
        delta_yaw = math.radians(float(physical_angular_dps)) * float(time_step_s)
        commanded_distance = float(physical_linear_mps) * float(time_step_s)
        path_delta = max(0.0, float(path_length_m) - self.previous_path_length_m)
        distance = math.copysign(path_delta, commanded_distance) if abs(commanded_distance) > 1.0e-8 else 0.0
        mid_yaw = self.yaw_rad + 0.5 * delta_yaw
        self.x_m += distance * math.cos(mid_yaw)
        self.y_m += distance * math.sin(mid_yaw)
        self.yaw_rad = math.atan2(math.sin(self.yaw_rad + delta_yaw), math.cos(self.yaw_rad + delta_yaw))

        self.blocked_last = abs(commanded_distance) >= 0.03 and path_delta < 0.01
        self.previous_path_length_m = float(path_length_m)
        self.last_path_delta_m = path_delta
        self.previous_command = (float(normalized_linear), float(normalized_angular))
        self.previous_gate = int(gate)
        self.previous_selected_waypoint = tuple(float(item) for item in selected_waypoint)

        distances = [math.hypot(self.x_m - x, self.y_m - y) for x, y in self.nodes]
        nearest = min(distances)
        if nearest >= 0.5:
            self.nodes.append((self.x_m, self.y_m))
            self.edge_count += 1
            self.local_visit_count = 1
            self.steps_since_new_node = 0
        else:
            self.local_visit_count += 1
            self.steps_since_new_node += 1

    def topology_features(self) -> tuple[float, ...]:
        distances = [math.hypot(self.x_m - x, self.y_m - y) for x, y in self.nodes]
        nearest = min(distances)
        revisit = float(nearest < 0.25 and self.steps_since_new_node > 4)
        return (
            min(len(self.nodes) / 64.0, 1.0),
            min(self.edge_count / 64.0, 1.0),
            revisit,
            min(nearest / 2.0, 1.0),
            min(self.local_visit_count / 10.0, 1.0),
            math.sin(self.yaw_rad),
            math.cos(self.yaw_rad),
            min(self.steps_since_new_node / 32.0, 1.0),
        )


def build_features(
    waypoints: Sequence[Sequence[float]],
    *,
    selected_index: int,
    selected_waypoint: Sequence[float],
    model_stop: bool,
    step_index: int,
    max_steps: int,
    state: CausalTopologyState,
) -> np.ndarray:
    """Build the fixed 64-wide policy observation without goal/simulator truth."""

    if max_steps < 1 or not 0 <= step_index < max_steps:
        raise PolicyContractError("step index is outside the A3 feature contract")
    waypoint_rows = tuple(waypoints)
    if len(waypoint_rows) != TRAJECTORY_WIDTH:
        raise PolicyContractError(
            f"LightNav trajectory must contain exactly {TRAJECTORY_WIDTH} waypoints"
        )
    if (
        isinstance(selected_index, (bool, np.bool_))
        or not isinstance(selected_index, (int, np.integer))
        or not 0 <= int(selected_index) < TRAJECTORY_WIDTH
    ):
        raise PolicyContractError("LightNav selected index is outside the trajectory")
    trajectory = np.zeros((TRAJECTORY_WIDTH, 3), dtype=np.float32)
    mask = np.ones((TRAJECTORY_WIDTH,), dtype=np.float32)
    for index, waypoint in enumerate(waypoint_rows):
        if len(waypoint) != 3:
            raise PolicyContractError("LightNav trajectory row is not SE(2)")
        value = np.asarray(waypoint, dtype=np.float32)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise PolicyContractError("LightNav trajectory row is invalid")
        trajectory[index] = value
    selected = np.asarray(selected_waypoint, dtype=np.float32)
    if selected.shape != (3,) or not np.isfinite(selected).all():
        raise PolicyContractError("LightNav selected waypoint is invalid")
    if not np.allclose(
        selected,
        trajectory[int(selected_index)],
        rtol=1.0e-6,
        atol=1.0e-6,
    ):
        raise PolicyContractError("LightNav selected waypoint does not match selected index")
    trajectory[:, 0:2] = np.clip(trajectory[:, 0:2], -1.0, 1.0)
    trajectory[:, 2] = np.clip(trajectory[:, 2] / math.pi, -1.0, 1.0)
    selected_scaled = np.asarray(
        (np.clip(selected[0], -1.0, 1.0), np.clip(selected[1], -1.0, 1.0), np.clip(selected[2] / math.pi, -1.0, 1.0)),
        dtype=np.float32,
    )
    previous = np.asarray(state.previous_selected_waypoint, dtype=np.float32)
    proposal_delta = selected - previous
    proposal_delta_scaled = np.asarray(
        (np.clip(proposal_delta[0], -1.0, 1.0), np.clip(proposal_delta[1], -1.0, 1.0), np.clip(proposal_delta[2] / math.pi, -1.0, 1.0)),
        dtype=np.float32,
    )
    previous_gate = np.zeros((3,), dtype=np.float32)
    previous_gate[state.previous_gate] = 1.0
    parts = (
        trajectory.reshape(-1),
        mask,
        selected_scaled,
        np.asarray((int(selected_index) / 9.0,), dtype=np.float32),
        np.asarray((float(model_stop), step_index / max_steps), dtype=np.float32),
        np.asarray(state.previous_command, dtype=np.float32),
        previous_gate,
        np.asarray((float(state.blocked_last),), dtype=np.float32),
        np.asarray(state.topology_features(), dtype=np.float32),
        proposal_delta_scaled,
        np.asarray((min(state.last_path_delta_m / 0.25, 1.0),), dtype=np.float32),
    )
    result = np.concatenate(parts)
    if result.shape != (FEATURE_DIM,) or not np.isfinite(result).all():
        raise PolicyContractError(f"A3 feature assembly produced shape {result.shape}")
    return result


def apply_policy_action(
    proposal: Sequence[float], action: PolicyAction
) -> tuple[float, float, float]:
    if len(proposal) != 3:
        raise PolicyContractError("LightNav proposal is not SE(2)")
    forward, lateral, yaw = (float(item) for item in proposal)
    if action.gate == STOP_GATE:
        return (0.0, 0.0, 0.0)
    if action.gate == KEEP_GATE:
        return (forward, lateral, yaw)
    if action.gate != ADJUST_GATE:
        raise PolicyContractError("A3 gate is outside the executable action space")
    delta_forward, delta_yaw = action.bounded_residual
    resolved_forward = min(
        EXECUTOR_FORWARD_RANGE_M[1],
        max(EXECUTOR_FORWARD_RANGE_M[0], forward + delta_forward),
    )
    resolved_yaw = min(
        EXECUTOR_YAW_RANGE_RAD[1],
        max(EXECUTOR_YAW_RANGE_RAD[0], yaw + delta_yaw),
    )
    return (resolved_forward, lateral, resolved_yaw)


__all__ = [
    "ADJUST_GATE",
    "CausalTopologyState",
    "FEATURE_DIM",
    "FEATURE_SCHEMA",
    "GATE_NAMES",
    "HIDDEN_DIM",
    "KEEP_GATE",
    "NumpyLightNavPolicy",
    "PolicyAction",
    "PolicyContractError",
    "RESIDUAL_LIMITS",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "STOP_GATE",
    "apply_policy_action",
    "build_features",
    "initialize_checkpoint",
    "initialize_parameters",
    "load_checkpoint",
    "save_checkpoint",
    "sha256_file",
]
