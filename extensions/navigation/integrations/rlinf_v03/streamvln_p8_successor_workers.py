"""Opt-in RLinf worker pair for one-primitive P8 successor round trips."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
import importlib
import math
import secrets
from typing import Callable, Iterable, Mapping

from rlinf.scheduler import Channel
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

try:
    from .streamvln_executor_contract import feedback_from_dict
    from .streamvln_p8_successor_action import ACTION_CHUNK, CompiledRoute, SuccessorPlanner
    from .streamvln_p8_topology import (
        P8CausalTopologySnapshot,
        snapshot_online_topology,
    )
    from .streamvln_topology import OnlineTopology
    from .streamvln_p8_successor_protocol import (
        SUCCESSOR_REPLAY_SCHEMA,
        MacroTransition,
        SuccessorDecisionSample,
        SuccessorMacroAccumulator,
        SuccessorStepRequest,
        SuccessorStepResult,
    )
except ImportError:  # Staged-file tests and same-directory overlay installation.
    from streamvln_executor_contract import feedback_from_dict
    from streamvln_p8_successor_action import ACTION_CHUNK, CompiledRoute, SuccessorPlanner
    from streamvln_p8_topology import (
        P8CausalTopologySnapshot,
        snapshot_online_topology,
    )
    from streamvln_topology import OnlineTopology
    from streamvln_p8_successor_protocol import (
        SUCCESSOR_REPLAY_SCHEMA,
        MacroTransition,
        SuccessorDecisionSample,
        SuccessorMacroAccumulator,
        SuccessorStepRequest,
        SuccessorStepResult,
    )


SUCCESSOR_CHANNEL_TAG = "train_rollout_results"
DEFAULT_ADAPTER_MODULE = "rlinf.workers.streamvln_p8_official_adapter"


def _cfg_get(value: object, key: str, default=None):
    getter = getattr(value, "get", None)
    return getter(key, default) if callable(getter) else getattr(value, key, default)


def _load_symbol(spec: str):
    if not isinstance(spec, str) or ":" not in spec:
        raise RuntimeError("successor factory must use 'module:symbol' syntax")
    module_name, symbol_name = spec.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(f"successor factory module is unavailable: {module_name}") from exc
    factory = getattr(module, symbol_name, None)
    if not callable(factory):
        raise RuntimeError(f"successor factory is not callable: {spec}")
    return factory


def require_successor_production_ready(cfg, *, importer=importlib.import_module) -> str:
    """Reject successor startup until the release anchor and adapters are real."""

    runner = _cfg_get(cfg, "runner", {})
    env_cfg = _cfg_get(cfg, "env", {})
    train_env = _cfg_get(env_cfg, "train", None)
    rollout_cfg = _cfg_get(cfg, "rollout", {})
    if train_env is None or _cfg_get(train_env, "total_num_envs", None) != 1:
        raise RuntimeError("P8 successor currently requires env.train.total_num_envs=1")
    if _cfg_get(train_env, "rollout_epoch", None) != 1:
        raise RuntimeError("P8 successor currently requires env.train.rollout_epoch=1")
    if _cfg_get(rollout_cfg, "pipeline_stage_num", None) != 1:
        raise RuntimeError("P8 successor currently requires rollout.pipeline_stage_num=1")
    if _cfg_get(runner, "p8_successor_model_env_id", 0) != 0:
        raise RuntimeError("P8 successor currently requires model_env_id=0")
    max_episode_steps = _cfg_get(train_env, "max_episode_steps", None)
    model_cfg = _cfg_get(_cfg_get(cfg, "actor", {}), "model", {})
    topology_max_nodes = _cfg_get(model_cfg, "successor_topology_max_nodes", None)
    if type(max_episode_steps) is not int or max_episode_steps < 1:
        raise RuntimeError("P8 successor requires a positive max_episode_steps")
    if type(topology_max_nodes) is not int or topology_max_nodes < max_episode_steps + 1:
        raise RuntimeError(
            "P8 successor topology_max_nodes must cover max_episode_steps + 1"
        )

    adapter_module_name = _cfg_get(
        runner, "p8_successor_adapter_module", DEFAULT_ADAPTER_MODULE
    )
    try:
        adapter_module = importer(adapter_module_name)
    except ImportError as exc:
        raise RuntimeError("P8 successor official adapter is unavailable") from exc
    anchor_check = getattr(adapter_module, "_require_pinned_release_manifest_sha256", None)
    if not callable(anchor_check):
        raise RuntimeError("P8 successor adapter lacks the release-anchor gate")
    anchor = anchor_check()
    loaded_factories = {}
    for key in (
        "p8_successor_env_controller_factory",
        "p8_successor_rollout_controller_factory",
        "p8_successor_actor_replay_adapter",
    ):
        factory_spec = _cfg_get(runner, key, None)
        if not factory_spec:
            raise RuntimeError(f"runner.{key} is required for successor startup")
        loaded_factories[key] = _load_symbol(factory_spec)
    for key, factory in loaded_factories.items():
        if getattr(factory, "successor_replay_schema", None) != SUCCESSOR_REPLAY_SCHEMA:
            raise RuntimeError(f"successor factory schema mismatch: {key}")
    return anchor


@dataclass(frozen=True)
class TrainShortestPathExpertTarget:
    """Private train-only target transported separately from policy observations."""

    episode_token: str
    physical_step: int
    decision_index: int
    waypoint_local_m: tuple[float, float]
    lookahead_m: float
    bearing_rad: float
    target_distance_m: float
    distance_to_goal_m: float
    stop: bool
    source: str = "train-shortest-path"

    def __post_init__(self) -> None:
        if (
            not self.episode_token
            or type(self.physical_step) is not int
            or self.physical_step < 0
            or type(self.decision_index) is not int
            or self.decision_index < 0
        ):
            raise ValueError("expert target identity is invalid")
        if self.source != "train-shortest-path" or type(self.stop) is not bool:
            raise ValueError("expert target provenance is invalid")
        if len(self.waypoint_local_m) != 2:
            raise ValueError("expert target waypoint must be two-dimensional")
        numeric = (
            *self.waypoint_local_m,
            self.lookahead_m,
            self.bearing_rad,
            self.target_distance_m,
            self.distance_to_goal_m,
        )
        if any(not math.isfinite(float(value)) for value in numeric):
            raise ValueError("expert target values must be finite")
        if not 0.25 <= float(self.lookahead_m) <= 2.0:
            raise ValueError("expert target lookahead is outside the train contract")
        if self.stop and any(abs(float(value)) > 1e-8 for value in self.waypoint_local_m):
            raise ValueError("expert STOP target must carry a zero waypoint")

    @classmethod
    def from_payload(
        cls,
        payload: Mapping,
        *,
        episode_token: str,
        physical_step: int,
        decision_index: int,
    ) -> "TrainShortestPathExpertTarget":
        expected = {
            "schema_version", "source", "lookahead_m", "waypoint_local_m",
            "bearing_rad", "target_distance_m", "distance_to_goal_m", "stop",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("train expert payload schema drifted")
        if payload.get("schema_version") != 1:
            raise ValueError("train expert payload version drifted")
        waypoint = payload.get("waypoint_local_m")
        scalar_names = (
            "lookahead_m", "bearing_rad", "target_distance_m", "distance_to_goal_m"
        )
        if (
            not isinstance(waypoint, (list, tuple))
            or len(waypoint) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in waypoint
            )
            or any(
                isinstance(payload.get(name), bool)
                or not isinstance(payload.get(name), (int, float))
                for name in scalar_names
            )
            or type(payload.get("stop")) is not bool
        ):
            raise ValueError("train expert payload waypoint is invalid")
        return cls(
            episode_token=episode_token,
            physical_step=physical_step,
            decision_index=decision_index,
            waypoint_local_m=tuple(float(value) for value in waypoint),
            lookahead_m=float(payload["lookahead_m"]),
            bearing_rad=float(payload["bearing_rad"]),
            target_distance_m=float(payload["target_distance_m"]),
            distance_to_goal_m=float(payload["distance_to_goal_m"]),
            stop=payload["stop"],
            source=payload["source"],
        )


@dataclass(frozen=True)
class SuccessorFrameRequest:
    """EnvWorker-to-rollout payload for one physical frame."""

    env_slot: int
    episode_token: str
    physical_step: int
    decision_index: int
    needs_decision: bool
    decision_mask: bool
    observation: object | None
    official_frame: object | None
    topology_snapshot: P8CausalTopologySnapshot
    expert_target: TrainShortestPathExpertTarget | None = None
    previous_result: SuccessorStepResult | None = None
    terminal_ack: bool = False

    def __post_init__(self) -> None:
        if type(self.env_slot) is not int or self.env_slot < 0:
            raise ValueError("env_slot must be a non-negative integer")
        if not isinstance(self.episode_token, str) or not self.episode_token:
            raise ValueError("episode_token must be a non-empty string")
        if type(self.physical_step) is not int or self.physical_step < 0:
            raise ValueError("physical_step must be a non-negative integer")
        if type(self.decision_index) is not int or self.decision_index < 0:
            raise ValueError("decision_index must be a non-negative integer")
        if type(self.needs_decision) is not bool or type(self.decision_mask) is not bool:
            raise ValueError("decision flags must be bools")
        if self.needs_decision != self.decision_mask:
            raise ValueError("frame decision flags must agree")
        if type(self.terminal_ack) is not bool:
            raise ValueError("terminal_ack must be a bool")
        if not isinstance(self.topology_snapshot, P8CausalTopologySnapshot):
            raise ValueError("frame requires a typed causal topology snapshot")
        if (
            self.topology_snapshot.episode_token != self.episode_token
            or self.topology_snapshot.physical_step != self.physical_step
            or self.topology_snapshot.decision_index != self.decision_index
        ):
            raise ValueError("frame and causal topology snapshot identities differ")
        if not self.terminal_ack and self.official_frame is None:
            raise ValueError("each physical frame requires a private official frame")
        if self.expert_target is not None:
            if not isinstance(self.expert_target, TrainShortestPathExpertTarget):
                raise TypeError("frame expert target must use the typed train schema")
            if (
                not self.needs_decision
                or self.expert_target.episode_token != self.episode_token
                or self.expert_target.physical_step != self.physical_step
                or self.expert_target.decision_index != self.decision_index
            ):
                raise ValueError("frame expert target identity is stale")
        if self.physical_step == 0 and self.previous_result is not None:
            raise ValueError("an episode-start frame cannot carry prior feedback")
        if self.previous_result is not None:
            if (
                self.previous_result.episode_token != self.episode_token
                or self.previous_result.physical_step != self.physical_step - 1
            ):
                raise ValueError("previous_result is stale for this physical frame")
            expected_decision_index = self.previous_result.decision_index + int(
                self.previous_result.needs_decision
            )
            if self.decision_index != expected_decision_index:
                raise ValueError("previous_result decision_index is stale for this frame")
            if self.previous_result.needs_decision != self.needs_decision:
                if not self.terminal_ack:
                    raise ValueError("frame state disagrees with previous feedback")
        if self.terminal_ack:
            if self.previous_result is None or not (
                self.previous_result.terminated or self.previous_result.truncated
            ):
                raise ValueError("terminal_ack requires terminal feedback")
            if self.needs_decision or self.decision_mask:
                raise ValueError("terminal acknowledgements cannot request a decision")
            if self.previous_result.truncated and self.observation is None:
                raise ValueError("truncation acknowledgement requires final_observation")


@dataclass(frozen=True)
class SuccessorFrameBatch:
    items: tuple[SuccessorFrameRequest, ...]

    def __post_init__(self) -> None:
        if not self.items or any(
            not isinstance(item, SuccessorFrameRequest) for item in self.items
        ):
            raise ValueError("frame batch must contain typed frame requests")

    def split_by_sizes(self, sizes: Iterable[int]) -> tuple["SuccessorFrameBatch", ...]:
        sizes = tuple(sizes)
        if not sizes or any(type(size) is not int or size < 1 for size in sizes):
            raise ValueError("split sizes must be positive integers")
        if sum(sizes) != len(self.items):
            raise ValueError("split sizes must sum to the frame batch size")
        offset = 0
        shards = []
        for size in sizes:
            shards.append(SuccessorFrameBatch(self.items[offset : offset + size]))
            offset += size
        return tuple(shards)

    @classmethod
    def merge_shards(cls, shards: Iterable["SuccessorFrameBatch"]):
        shards = tuple(shards)
        if not shards or any(not isinstance(shard, cls) for shard in shards):
            raise ValueError("frame shards must be typed batches")
        return cls(tuple(item for shard in shards for item in shard.items))


@dataclass(frozen=True)
class SuccessorWorkerResponse:
    """Slot-preserving rollout response; request=None acknowledges a closed slot."""

    env_slot: int
    request: SuccessorStepRequest | None
    bootstrap_value: float | None = None

    def __post_init__(self) -> None:
        if type(self.env_slot) is not int or self.env_slot < 0:
            raise ValueError("env_slot must be a non-negative integer")
        if self.request is not None and not isinstance(self.request, SuccessorStepRequest):
            raise ValueError("request must be typed when present")
        if self.request is not None and self.bootstrap_value is not None:
            raise ValueError("physical-step responses cannot also bootstrap")
        if self.bootstrap_value is not None and not math.isfinite(
            float(self.bootstrap_value)
        ):
            raise ValueError("bootstrap_value must be finite")


@dataclass(frozen=True)
class SuccessorWorkerResponseBatch:
    items: tuple[SuccessorWorkerResponse, ...]

    def __post_init__(self) -> None:
        if not self.items or any(
            not isinstance(item, SuccessorWorkerResponse) for item in self.items
        ):
            raise ValueError("response batch must preserve typed environment slots")

    def split_by_sizes(self, sizes: Iterable[int]):
        sizes = tuple(sizes)
        if not sizes or any(type(size) is not int or size < 1 for size in sizes):
            raise ValueError("split sizes must be positive integers")
        if sum(sizes) != len(self.items):
            raise ValueError("split sizes must sum to the response batch size")
        offset = 0
        shards = []
        for size in sizes:
            shards.append(SuccessorWorkerResponseBatch(self.items[offset : offset + size]))
            offset += size
        return tuple(shards)

    @classmethod
    def merge_shards(cls, shards):
        shards = tuple(shards)
        if not shards or any(not isinstance(shard, cls) for shard in shards):
            raise ValueError("response shards must be typed batches")
        return cls(tuple(item for shard in shards for item in shard.items))


class SuccessorRolloutController:
    """Advance official inference every frame and sample only at decision boundaries."""

    def __init__(
        self,
        *,
        official_driver_factory: Callable[[SuccessorFrameRequest], object],
        decision_sampler: Callable[
            [SuccessorFrameRequest, int], SuccessorDecisionSample
        ],
        bootstrap_value_evaluator: Callable[[SuccessorFrameRequest], float],
    ) -> None:
        if not all(
            callable(value)
            for value in (
                official_driver_factory,
                decision_sampler,
                bootstrap_value_evaluator,
            )
        ):
            raise TypeError("successor rollout controller factories must be callable")
        self._official_driver_factory = official_driver_factory
        self._decision_sampler = decision_sampler
        self._bootstrap_value_evaluator = bootstrap_value_evaluator
        self._episode_token: str | None = None
        self._driver = None
        self._last_ack_step = -1

    def _bind(self, frame: SuccessorFrameRequest) -> None:
        if not isinstance(frame, SuccessorFrameRequest):
            raise TypeError("rollout controller requires SuccessorFrameRequest")
        if frame.episode_token != self._episode_token:
            if frame.physical_step != 0:
                raise ValueError("a new rollout episode must start at physical_step zero")
            self._driver = self._official_driver_factory(frame)
            self._episode_token = frame.episode_token
            self._last_ack_step = -1
        if self._driver is None:
            raise RuntimeError("official driver is unavailable")

    def _ack_previous(self, frame: SuccessorFrameRequest) -> None:
        if frame.previous_result is not None:
            if frame.previous_result.physical_step <= self._last_ack_step:
                raise ValueError("official driver received duplicate physical feedback")
            self._driver.acknowledge_physical_step(
                terminated=frame.previous_result.terminated,
                truncated=frame.previous_result.truncated,
            )
            self._last_ack_step = frame.previous_result.physical_step

    def process(self, frame: SuccessorFrameRequest) -> SuccessorWorkerResponse:
        self._bind(frame)
        if frame.terminal_ack:
            assert frame.previous_result is not None
            bootstrap_value = (
                0.0
                if frame.previous_result.terminated
                else float(self._bootstrap_value_evaluator(frame))
            )
            self._ack_previous(frame)
            return SuccessorWorkerResponse(
                env_slot=frame.env_slot,
                request=None,
                bootstrap_value=bootstrap_value,
            )
        self._ack_previous(frame)
        shadow_action = self._driver.prepare_shadow_action(frame.official_frame)
        decision_sample = None
        if frame.needs_decision:
            decision_sample = self._decision_sampler(frame, shadow_action)
            if not isinstance(decision_sample, SuccessorDecisionSample):
                raise TypeError("decision sampler must return SuccessorDecisionSample")
        request = SuccessorStepRequest(
            episode_token=frame.episode_token,
            physical_step=frame.physical_step,
            decision_index=frame.decision_index,
            needs_decision=frame.needs_decision,
            decision_mask=frame.decision_mask,
            official_shadow_action=shadow_action,
            decision_sample=decision_sample,
        )
        return SuccessorWorkerResponse(env_slot=frame.env_slot, request=request)

    def acknowledge_only(self, frame: SuccessorFrameRequest) -> SuccessorWorkerResponse:
        self._bind(frame)
        if frame.terminal_ack:
            assert frame.previous_result is not None
            bootstrap_value = (
                0.0
                if frame.previous_result.terminated
                else float(self._bootstrap_value_evaluator(frame))
            )
        else:
            bootstrap_value = float(self._bootstrap_value_evaluator(frame))
        self._ack_previous(frame)
        return SuccessorWorkerResponse(
            env_slot=frame.env_slot,
            request=None,
            bootstrap_value=bootstrap_value,
        )


def _first_scalar(value: object, *, name: str):
    while isinstance(value, (list, tuple)):
        if not value:
            raise RuntimeError(f"successor environment returned empty {name}")
        value = value[0]
    if hasattr(value, "reshape"):
        value = value.reshape(-1)[0]
    return value.item() if hasattr(value, "item") else value


class SuccessorEnvController:
    """Own one environment pending prefix and emit one macro replay row."""

    def __init__(
        self,
        env,
        *,
        env_slot: int = 0,
        token_factory: Callable[[], str] | None = None,
        max_decisions: int,
        topology_max_nodes: int,
        planner: SuccessorPlanner | None = None,
        bc_expert_enabled: bool = False,
        expert_lookahead_m: float = 1.0,
    ) -> None:
        if not hasattr(env, "reset") or not hasattr(env, "chunk_step"):
            raise TypeError("successor env controller requires reset and chunk_step")
        self.env = env
        if type(env_slot) is not int or env_slot < 0:
            raise ValueError("env_slot must be a non-negative integer")
        self.env_slot = env_slot
        if type(max_decisions) is not int or max_decisions < 1:
            raise ValueError("successor max_decisions must be positive")
        if type(topology_max_nodes) is not int or topology_max_nodes < max_decisions + 1:
            raise ValueError("topology_max_nodes must cover every causal decision node")
        if planner is None:
            planner = getattr(env, "successor_planner", None) or SuccessorPlanner()
        if not isinstance(planner, SuccessorPlanner):
            raise TypeError("successor env controller requires SuccessorPlanner")
        self._token_factory = token_factory or (lambda: secrets.token_hex(16))
        self._last_episode_token: str | None = None
        self._max_decisions = max_decisions
        self._topology_max_nodes = topology_max_nodes
        self._planner = planner
        if type(bc_expert_enabled) is not bool:
            raise TypeError("BC expert enable flag must be boolean")
        if not math.isfinite(float(expert_lookahead_m)) or not 0.25 <= float(
            expert_lookahead_m
        ) <= 2.0:
            raise ValueError("BC expert lookahead is outside the train contract")
        if bc_expert_enabled and not callable(getattr(env, "expert_waypoint", None)):
            raise TypeError("BC expert mode requires the train shortest-path channel")
        self._bc_expert_enabled = bc_expert_enabled
        self._expert_lookahead_m = float(expert_lookahead_m)
        self._topology: OnlineTopology | None = None
        self._accumulator: SuccessorMacroAccumulator | None = None
        self._active_plan: CompiledRoute | None = None
        self._last_closed_plan: CompiledRoute | None = None
        self._terminal_pending = False

    @property
    def topology(self) -> OnlineTopology:
        if self._topology is None:
            raise RuntimeError("successor causal topology is inactive")
        return self._topology

    @property
    def last_closed_plan(self) -> CompiledRoute | None:
        return self._last_closed_plan

    @property
    def active_plan(self) -> CompiledRoute | None:
        return self._active_plan

    def reset_episode(self) -> SuccessorFrameRequest:
        token = self._token_factory()
        if not isinstance(token, str) or not token or token == self._last_episode_token:
            self._accumulator = None
            self._topology = None
            self._active_plan = None
            self._last_closed_plan = None
            self._terminal_pending = False
            self._planner.reset_episode()
            raise RuntimeError("successor reset must rotate the episode token")
        self._accumulator = None
        self._topology = None
        self._active_plan = None
        self._last_closed_plan = None
        self._terminal_pending = False
        self._planner.reset_episode()
        observation, _ = self.env.reset()
        self._topology = OnlineTopology(
            max_decisions=self._max_decisions,
            forward_step_m=self._planner.config.forward_step_m,
            turn_degrees=self._planner.config.turn_degrees,
        )
        self._accumulator = SuccessorMacroAccumulator(token)
        self._last_episode_token = token
        return self._frame(observation, previous_result=None)

    def reset_after_terminal_ack(self) -> SuccessorFrameRequest:
        if not self._terminal_pending:
            raise RuntimeError("environment slot has no terminal acknowledgement pending")
        return self.reset_episode()

    def _frame(self, observation, *, previous_result) -> SuccessorFrameRequest:
        if self._accumulator is None:
            raise RuntimeError("successor environment episode is inactive")
        expert_target = None
        if self._bc_expert_enabled and self._accumulator.needs_decision:
            expert_target = TrainShortestPathExpertTarget.from_payload(
                self.env.expert_waypoint(self._expert_lookahead_m),
                episode_token=self._accumulator.episode_token,
                physical_step=self._accumulator.physical_step,
                decision_index=self._accumulator.decision_index,
            )
        return SuccessorFrameRequest(
            env_slot=self.env_slot,
            episode_token=self._accumulator.episode_token,
            physical_step=self._accumulator.physical_step,
            decision_index=self._accumulator.decision_index,
            needs_decision=self._accumulator.needs_decision,
            decision_mask=self._accumulator.decision_mask,
            observation=observation,
            official_frame=self.env.observe_official_frame(),
            topology_snapshot=snapshot_online_topology(
                self.topology,
                episode_token=self._accumulator.episode_token,
                physical_step=self._accumulator.physical_step,
                decision_index=self._accumulator.decision_index,
                max_nodes=self._topology_max_nodes,
            ),
            expert_target=expert_target,
            previous_result=previous_result,
        )

    def _terminal_frame(
        self, result: SuccessorStepResult, observation
    ) -> SuccessorFrameRequest:
        self._terminal_pending = True
        self._accumulator = None
        return SuccessorFrameRequest(
            env_slot=self.env_slot,
            episode_token=result.episode_token,
            physical_step=result.physical_step + 1,
            decision_index=result.decision_index + 1,
            needs_decision=False,
            decision_mask=False,
            observation=observation,
            official_frame=None,
            topology_snapshot=snapshot_online_topology(
                self.topology,
                episode_token=result.episode_token,
                physical_step=result.physical_step + 1,
                decision_index=result.decision_index + 1,
                max_nodes=self._topology_max_nodes,
            ),
            previous_result=result,
            terminal_ack=True,
        )

    def _compile_boundary_request(
        self, request: SuccessorStepRequest
    ) -> SuccessorStepRequest:
        assert self._accumulator is not None
        expected = (
            self._accumulator.episode_token,
            self._accumulator.physical_step,
            self._accumulator.decision_index,
        )
        actual = (request.episode_token, request.physical_step, request.decision_index)
        if actual != expected:
            raise ValueError(
                f"stale successor request identity: expected {expected!r}, got {actual!r}"
            )
        if not request.needs_decision:
            if self._active_plan is None:
                raise RuntimeError("successor continuation has no active compiled plan")
            return request
        if self._active_plan is not None or request.decision_sample is None:
            raise RuntimeError("successor decision boundary overlaps an active plan")
        plan = self._planner.plan(request.decision_sample.decision, self.topology)
        if plan.requested_route is not request.decision_sample.decision.route:
            raise RuntimeError("executor changed the sampled policy route identity")
        self._active_plan = plan
        return replace(
            request,
            decision_sample=replace(
                request.decision_sample,
                proposed_actions=plan.proposed_actions,
                compiled_route=plan,
            ),
        )

    def _close_topology_once(self, macro: MacroTransition) -> None:
        if self._active_plan is None:
            raise RuntimeError("successor macro closed without an active compiled plan")
        active = slice(0, macro.duration)
        before = self.topology.decision_count
        self.topology.update(
            macro.executed_actions[active],
            macro.primitive_collisions[active],
            realized_path_m=macro.realized_path_m[active],
            blocked_forwards=macro.primitive_blocked[active],
        )
        if self.topology.decision_count != before + 1:
            raise RuntimeError("causal topology must update exactly once per closed macro")
        self._last_closed_plan = self._active_plan
        self._active_plan = None

    def accept(
        self, request: SuccessorStepRequest
    ) -> tuple[SuccessorFrameRequest | None, MacroTransition | None]:
        if self._accumulator is None:
            raise RuntimeError("successor environment episode is inactive")
        request = self._compile_boundary_request(request)
        command = self._accumulator.prepare_step(request)
        payload = [[[command.executed_action], *[[0] for _ in range(ACTION_CHUNK - 1)]]]
        observations, rewards, terminations, truncations, infos = self.env.chunk_step(
            payload, execute_count=1
        )
        if len(observations) != 1 or len(infos) != 1:
            raise RuntimeError("successor environment must return one result")
        feedback = feedback_from_dict(infos[0].get("executor_feedback"))
        if len(feedback.primitives) != 1:
            raise RuntimeError("successor environment must execute one primitive")
        primitive = feedback.primitives[0]
        provisional = SuccessorStepResult(
            episode_token=command.episode_token,
            physical_step=command.physical_step,
            decision_index=command.decision_index,
            needs_decision=False,
            decision_mask=command.decision_mask,
            executed_action=command.executed_action,
            reward=float(_first_scalar(rewards, name="reward")),
            collision=primitive.collision,
            blocked=primitive.blocked,
            terminated=bool(_first_scalar(terminations, name="termination")),
            truncated=bool(_first_scalar(truncations, name="truncation")),
            realized_path_m=primitive.realized_path_m,
        )
        result = replace(
            provisional,
            needs_decision=self._accumulator.feedback_will_close_macro(provisional),
        )
        macro = self._accumulator.accept_feedback(result)
        if macro is not None:
            self._close_topology_once(macro)
        if result.terminated or result.truncated:
            terminal_observation = None
            if result.truncated:
                if "final_observation" not in infos[0] or infos[0]["final_observation"] is None:
                    raise RuntimeError(
                        "truncated successor step requires explicit final_observation"
                    )
                terminal_observation = infos[0]["final_observation"]
            return self._terminal_frame(result, terminal_observation), macro
        return self._frame(observations[0], previous_result=result), macro


class P8SuccessorRolloutWorker(MultiStepRolloutWorker):
    """RLinf rollout worker overriding the runner-called physical-step loop."""

    def __init__(self, cfg):
        require_successor_production_ready(cfg)
        super().__init__(cfg)

    def init_worker(self):
        super().init_worker()
        runner = _cfg_get(self.cfg, "runner", {})
        factory = _load_symbol(_cfg_get(runner, "p8_successor_rollout_controller_factory"))
        controller = factory(worker=self, env_slot=0)
        if not isinstance(controller, SuccessorRolloutController):
            raise TypeError("rollout controller factory returned the wrong type")
        self._successor_controllers = {0: controller}

    async def _recv_frames(self, input_channel: Channel, stage_id: int):
        return await self.recv_from(
            group_name=self.cfg.env.group_name,
            channel=input_channel,
            tag=SUCCESSOR_CHANNEL_TAG,
            route_key=stage_id,
            async_op=True,
            batch_size=self.train_batch_size,
            merge_fn=SuccessorFrameBatch.merge_shards,
            infer_batch_size_fn=lambda batch: len(batch.items),
        ).async_wait()

    def _send_responses(self, output_channel: Channel, stage_id: int, responses) -> None:
        self.send_to(
            group_name=self.cfg.env.group_name,
            channel=output_channel,
            data=SuccessorWorkerResponseBatch(tuple(responses)),
            tag=SUCCESSOR_CHANNEL_TAG,
            route_key=stage_id,
            async_op=True,
            batch_size=self.train_batch_size,
            split_fn=lambda batch, sizes: batch.split_by_sizes(sizes),
        )

    async def generate_one_epoch(self, input_channel: Channel, output_channel: Channel):
        target_steps = int(self.cfg.env.train.max_steps_per_rollout_epoch)
        completed_steps = 0
        while completed_steps < target_steps:
            for stage_id in range(self.num_pipeline_stages):
                frames = await self._recv_frames(input_channel, stage_id)
                responses = []
                for frame in frames.items:
                    controller = self._successor_controllers.get(frame.env_slot)
                    if controller is None:
                        raise RuntimeError("successor frame references an unknown env slot")
                    response = controller.process(frame)
                    responses.append(response)
                    completed_steps += response.request is not None
                self._send_responses(output_channel, stage_id, responses)

        for stage_id in range(self.num_pipeline_stages):
            frames = await self._recv_frames(input_channel, stage_id)
            responses = []
            for frame in frames.items:
                if not frame.terminal_ack and not frame.needs_decision:
                    raise RuntimeError("rollout horizon split a pending successor macro")
                controller = self._successor_controllers.get(frame.env_slot)
                if controller is None:
                    raise RuntimeError("successor final frame references an unknown env slot")
                responses.append(controller.acknowledge_only(frame))
            self._send_responses(output_channel, stage_id, responses)


class P8SuccessorEnvWorker(EnvWorker):
    """RLinf EnvWorker overriding the runner-called interaction loop."""

    def __init__(self, cfg):
        require_successor_production_ready(cfg)
        super().__init__(cfg)

    def init_worker(self):
        super().init_worker()
        runner = _cfg_get(self.cfg, "runner", {})
        controller_factory = _load_symbol(
            _cfg_get(runner, "p8_successor_env_controller_factory")
        )
        self._successor_controllers = {}
        for stage_id, env in enumerate(self.env_list):
            controller = controller_factory(
                worker=self,
                env=env,
                stage_id=stage_id,
                env_slot=0,
            )
            if not isinstance(controller, SuccessorEnvController):
                raise TypeError("env controller factory returned the wrong type")
            self._successor_controllers[stage_id] = {0: controller}
        actor_factory = _load_symbol(_cfg_get(runner, "p8_successor_actor_replay_adapter"))
        self._successor_actor_adapter = actor_factory(worker=self)
        if (
            getattr(self._successor_actor_adapter, "successor_replay_schema", None)
            != SUCCESSOR_REPLAY_SCHEMA
            or not callable(getattr(self._successor_actor_adapter, "convert", None))
            or not callable(getattr(self._successor_actor_adapter, "build_batch", None))
        ):
            raise TypeError("successor actor replay adapter has the wrong schema or API")

    def _append_bootstrapped_macro(
        self,
        rows: list,
        pending_macros: dict,
        pending_final_observations: dict,
        *,
        stage_id: int,
        env_slot: int,
        next_value: float | None,
    ) -> None:
        key = (stage_id, env_slot)
        macro = pending_macros.get(key)
        if macro is None:
            raise RuntimeError("successor bootstrap has no pending macro transition")
        if next_value is None:
            raise RuntimeError("successor macro is missing next_value")
        rows.append(
            self._successor_actor_adapter.convert(
                macro=macro,
                next_value=next_value,
                stage_id=stage_id,
                env_slot=env_slot,
                final_observation=pending_final_observations.get(key),
            )
        )
        pending_macros[key] = None
        pending_final_observations[key] = None

    def _send_frames(self, rollout_channel: Channel, stage_id: int, frames) -> None:
        self.send_to(
            group_name=self.cfg.rollout.group_name,
            channel=rollout_channel,
            data=SuccessorFrameBatch(tuple(frames)),
            tag=SUCCESSOR_CHANNEL_TAG,
            route_key=stage_id,
            async_op=True,
            batch_size=self.train_batch_size,
            split_fn=lambda batch, sizes: batch.split_by_sizes(sizes),
        )

    async def _recv_responses(self, input_channel: Channel, stage_id: int):
        return await self.recv_from(
            group_name=self.cfg.rollout.group_name,
            channel=input_channel,
            tag=SUCCESSOR_CHANNEL_TAG,
            route_key=stage_id,
            async_op=True,
            batch_size=self.train_batch_size,
            merge_fn=SuccessorWorkerResponseBatch.merge_shards,
            infer_batch_size_fn=lambda batch: len(batch.items),
        ).async_wait()

    async def _run_successor_epoch(
        self,
        input_channel: Channel,
        rollout_channel: Channel,
        reward_channel: Channel | None,
        actor_channel: Channel | None,
        *,
        cooperative_yield: bool,
    ):
        if reward_channel is not None:
            raise RuntimeError("successor worker pair does not support a reward worker")
        if actor_channel is None:
            raise RuntimeError("successor macro replay requires an actor channel")
        target_steps = int(self.cfg.env.train.max_steps_per_rollout_epoch)
        completed_steps = {stage_id: 0 for stage_id in range(self.stage_num)}
        pending_macros = {
            (stage_id, env_slot): None
            for stage_id, controllers in self._successor_controllers.items()
            for env_slot in controllers
        }
        pending_final_observations = dict.fromkeys(pending_macros)
        replay_rows = []
        for stage_id, controllers in self._successor_controllers.items():
            self._send_frames(
                rollout_channel,
                stage_id,
                [controller.reset_episode() for controller in controllers.values()],
            )

        while any(count < target_steps for count in completed_steps.values()):
            for stage_id in range(self.stage_num):
                if completed_steps[stage_id] >= target_steps:
                    continue
                if cooperative_yield:
                    await asyncio.sleep(0)
                responses = await self._recv_responses(input_channel, stage_id)
                controllers = self._successor_controllers[stage_id]
                if len(responses.items) != len(controllers):
                    raise RuntimeError("successor response changed env-slot cardinality")
                next_frames = []
                for response in responses.items:
                    controller = controllers.get(response.env_slot)
                    if controller is None:
                        raise RuntimeError("successor response references an unknown env slot")
                    if response.request is None:
                        self._append_bootstrapped_macro(
                            replay_rows,
                            pending_macros,
                            pending_final_observations,
                            stage_id=stage_id,
                            env_slot=response.env_slot,
                            next_value=response.bootstrap_value,
                        )
                        next_frames.append(controller.reset_after_terminal_ack())
                        continue
                    pending = pending_macros[(stage_id, response.env_slot)]
                    if pending is not None:
                        sample = response.request.decision_sample
                        if sample is None or not response.request.decision_mask:
                            raise RuntimeError(
                                "successor macro must bootstrap from the next decision state"
                            )
                        self._append_bootstrapped_macro(
                            replay_rows,
                            pending_macros,
                            pending_final_observations,
                            stage_id=stage_id,
                            env_slot=response.env_slot,
                            next_value=sample.value,
                        )
                    next_frame, macro = controller.accept(response.request)
                    next_frames.append(next_frame)
                    completed_steps[stage_id] += 1
                    if macro is not None:
                        key = (stage_id, response.env_slot)
                        if pending_macros[key] is not None:
                            raise RuntimeError("successor emitted overlapping pending macros")
                        pending_macros[key] = macro
                        if macro.truncated:
                            if next_frame is None or next_frame.observation is None:
                                raise RuntimeError(
                                    "truncated macro lost its explicit final_observation"
                                )
                            pending_final_observations[key] = next_frame.observation
                self._send_frames(rollout_channel, stage_id, next_frames)

        for stage_id in range(self.stage_num):
            responses = await self._recv_responses(input_channel, stage_id)
            if len(responses.items) != len(self._successor_controllers[stage_id]):
                raise RuntimeError("successor final ack changed env-slot cardinality")
            if any(response.request is not None for response in responses.items):
                raise RuntimeError("successor final response must only acknowledge feedback")
            for response in responses.items:
                self._append_bootstrapped_macro(
                    replay_rows,
                    pending_macros,
                    pending_final_observations,
                    stage_id=stage_id,
                    env_slot=response.env_slot,
                    next_value=response.bootstrap_value,
                )
        if any(macro is not None for macro in pending_macros.values()):
            raise RuntimeError("successor epoch ended with an unbootstrapped macro")
        actor_channel.put(
            self._successor_actor_adapter.build_batch(replay_rows), async_op=True
        )
        return {}

    async def _run_interact_once(
        self,
        input_channel: Channel,
        rollout_channel: Channel,
        reward_channel: Channel | None,
        actor_channel: Channel | None,
        *,
        cooperative_yield: bool,
    ):
        for _ in range(self.rollout_epoch):
            await self._run_successor_epoch(
                input_channel,
                rollout_channel,
                reward_channel,
                actor_channel,
                cooperative_yield=cooperative_yield,
            )
        return {}
