"""Production bridge for the P8 successor policy and causal executor.

The frozen failed P8 runner remains unchanged.  This bridge is the only valid
entry point for a successor runner: it consumes one official StreamVLN shadow
action per physical step and executes a learned metric route one primitive at
a time, retaining the original high-level plan until feedback requires replan.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import secrets
from types import MappingProxyType
from typing import Callable, Mapping

if __package__:
    from .streamvln_executor_contract import feedback_from_dict
    from .streamvln_p8_official_keep import (
        OfficialEvalSession,
        OfficialGenerationRequest,
        validate_official_runtime_receipt,
    )
    from .streamvln_p8_official_frame import OfficialFrame
    from .streamvln_p8_successor_action import (
        ACTION_CHUNK,
        STOP_ACTION,
        RouteDecision,
        RouteName,
        SuccessorPlanner,
    )
    from .streamvln_p8_successor_executor import successor_end_reason
else:  # Explicit direct-file mode for local contract tests only.
    from streamvln_executor_contract import feedback_from_dict
    from streamvln_p8_official_keep import (
        OfficialEvalSession,
        OfficialGenerationRequest,
        validate_official_runtime_receipt,
    )
    from streamvln_p8_official_frame import OfficialFrame
    from streamvln_p8_successor_action import (
        ACTION_CHUNK,
        STOP_ACTION,
        RouteDecision,
        RouteName,
        SuccessorPlanner,
    )
    from streamvln_p8_successor_executor import successor_end_reason


BRIDGE_SCHEMA_NAME = "streamvln_p8_successor_bridge_decision"
BRIDGE_SCHEMA_VERSION = 1
CANONICAL_SUCCESSOR_ENV_MODULE = (
    "rlinf.envs.streamvln.streamvln_p8_successor_env"
)


@dataclass(frozen=True)
class OfficialGenerationResult:
    decoded_sequence: str
    output_ref: str
    past_key_values_ref: str


@dataclass(frozen=True)
class PreparedSuccessorDecisionContext:
    """One latched official frame/action plus optional frozen policy context."""

    episode_token: str
    physical_step: int
    decision_index: int
    official_shadow_action: int
    frame_ref: str
    frame_metadata: Mapping
    policy_context: object | None = None

    def __post_init__(self) -> None:
        if (
            not self.episode_token
            or type(self.physical_step) is not int
            or self.physical_step < 0
            or type(self.decision_index) is not int
            or self.decision_index < 0
        ):
            raise ValueError("prepared successor context identity is invalid")
        if (
            type(self.official_shadow_action) is not int
            or not 0 <= self.official_shadow_action <= 3
        ):
            raise TypeError("prepared official shadow action must be in 0..3")
        if not self.frame_ref or not isinstance(self.frame_metadata, Mapping):
            raise ValueError("prepared successor frame metadata is invalid")
        metadata = dict(self.frame_metadata)
        if (
            metadata.get("frame_ref") != self.frame_ref
            or metadata.get("physical_step") != self.physical_step
        ):
            raise ValueError("prepared successor frame metadata identity drifted")
        object.__setattr__(self, "frame_metadata", MappingProxyType(metadata))


@dataclass(frozen=True)
class SuccessorBridgeResult:
    episode_token: str
    decision_physical_step: int
    plan: dict
    official_shadow_actions: tuple[int, ...]
    official_frame_metadata: tuple[dict, ...]
    official_step_events: tuple[dict, ...]
    executed_actions: tuple[int, ...]
    primitive_feedback: tuple[dict, ...]
    end_reason: str
    replan_required: bool
    official_stop_override: bool
    terminated: bool
    truncated: bool
    final_observation: object

    def to_trace(self) -> dict:
        return {
            "schema_name": BRIDGE_SCHEMA_NAME,
            "schema_version": BRIDGE_SCHEMA_VERSION,
            "episode_token": self.episode_token,
            "decision_physical_step": self.decision_physical_step,
            "plan": dict(self.plan),
            "official_shadow_actions": list(self.official_shadow_actions),
            "official_frame_metadata": [
                dict(frame) for frame in self.official_frame_metadata
            ],
            "official_step_events": [dict(event) for event in self.official_step_events],
            "executed_actions": list(self.executed_actions),
            "primitive_feedback": [dict(row) for row in self.primitive_feedback],
            "end_reason": self.end_reason,
            "replan_required": self.replan_required,
            "official_stop_override": self.official_stop_override,
            "terminated": self.terminated,
            "truncated": self.truncated,
        }


class OfficialKeepDriver:
    """Bind the model-free official cadence to one live generation adapter."""

    def __init__(
        self,
        *,
        session: OfficialEvalSession,
        runtime_receipt: Mapping,
        generation_adapter: Callable[
            [OfficialGenerationRequest, tuple[OfficialFrame, ...]],
            OfficialGenerationResult,
        ],
        cache_reset_adapter: Callable[[Mapping], None],
        episode_serial: int,
        episode_identity: Mapping,
    ) -> None:
        if not isinstance(session, OfficialEvalSession):
            raise TypeError("official driver requires an OfficialEvalSession")
        validate_official_runtime_receipt(dict(runtime_receipt))
        if (
            not callable(generation_adapter) or not callable(cache_reset_adapter)
        ):
            raise TypeError("official driver adapters must be callable")
        if type(episode_serial) is not int or episode_serial < 0:
            raise ValueError("official driver episode_serial must be nonnegative")
        if (
            not isinstance(episode_identity, Mapping)
            or episode_identity.get("instruction_text") != session.instruction
            or not isinstance(episode_identity.get("episode_id"), str)
            or not isinstance(episode_identity.get("scene_id"), str)
            or not isinstance(episode_identity.get("instruction_sha256"), str)
        ):
            raise ValueError("official driver episode identity is invalid")
        self.session = session
        self.generation_adapter = generation_adapter
        self.cache_reset_adapter = cache_reset_adapter
        self.episode_serial = episode_serial
        self.episode_identity = dict(episode_identity)
        self.episode_generation: str | None = None
        self.frames: dict[int, OfficialFrame] = {}
        self.cache_reset_adapter(
            {
                "event": "episode_reset",
                "env_id": self.session.env_id,
                "physical_step": 0,
                "terminal": False,
            }
        )

    def prepare_shadow_action(self, frame: OfficialFrame) -> int:
        step = self.session.physical_step
        if not isinstance(frame, OfficialFrame):
            raise TypeError("official driver requires an OfficialFrame")
        frame.validate_identity(
            env_id=self.session.env_id,
            episode_serial=self.episode_serial,
            physical_step=step,
            identity=self.episode_identity,
        )
        if self.episode_generation is None:
            self.episode_generation = frame.episode_generation
        elif frame.episode_generation != self.episode_generation:
            raise RuntimeError("official driver received a frame from another reset generation")
        if step in self.frames:
            raise RuntimeError("official driver already cached this physical frame")
        self.frames[step] = frame
        self.session.observe(frame.frame_ref)
        if self.session.needs_generation:
            if self.session.queue_crossed_reset:
                raise RuntimeError(
                    "official action queue crossed a 32-step cache reset; "
                    "the pinned evaluator's next memory input is undefined for parity"
                )
            request = self.session.build_generation_request()
            generation_frames = tuple(self.frames[index] for index in request.frame_steps)
            if tuple(item.physical_step for item in generation_frames) != request.frame_steps:
                raise RuntimeError("official generation frame order drifted")
            result = self.generation_adapter(request, generation_frames)
            if not isinstance(result, OfficialGenerationResult):
                raise TypeError("official generation adapter returned an invalid result")
            self.session.accept_generation(
                decoded_sequence=result.decoded_sequence,
                output_ref=result.output_ref,
                past_key_values_ref=result.past_key_values_ref,
            )
        return self.session.pop_action()

    def acknowledge_physical_step(self, *, terminated: bool, truncated: bool) -> dict:
        event = self.session.after_env_step(
            terminated=terminated,
            truncated=truncated,
        )
        if event["cache_reset"]:
            self.cache_reset_adapter(
                {
                    "event": "cache_reset",
                    "env_id": self.session.env_id,
                    **event,
                }
            )
        return event


class SuccessorRunnerBridge:
    """Execute one high-level route while preserving physical-step cadence."""

    def __init__(
        self,
        *,
        env,
        topology_factory: Callable[[], object],
        official_driver_factory: Callable[[int, Mapping, int], OfficialKeepDriver],
        exact_control: bool = False,
        require_prepared_decision_context: bool = False,
    ) -> None:
        if (
            not hasattr(env, "successor_planner")
            or not hasattr(env, "chunk_step")
            or not callable(getattr(env, "reset", None))
            or not callable(getattr(env, "observe_official_frame", None))
        ):
            raise TypeError("successor bridge requires StreamVLNP8SuccessorEnv")
        if not callable(topology_factory) or not callable(official_driver_factory):
            raise TypeError("successor bridge episode factories must be callable")
        if (
            type(exact_control) is not bool
            or type(require_prepared_decision_context) is not bool
        ):
            raise TypeError("successor bridge mode flags must be boolean")
        self.env = env
        self.topology_factory = topology_factory
        self.official_driver_factory = official_driver_factory
        self.exact_control = exact_control
        self.require_prepared_decision_context = require_prepared_decision_context
        self.topology = None
        self.official_driver: OfficialKeepDriver | None = None
        self.observation = None
        self.exact_planner: SuccessorPlanner | None = None
        self.episode_token: str | None = None
        self._next_decision_index = 0
        self._episode_active = False
        self._prepared_decision_context: PreparedSuccessorDecisionContext | None = None
        self._preparation_poisoned = False

    @staticmethod
    def _validate_fresh_topology(topology: object) -> None:
        if not hasattr(topology, "update"):
            raise TypeError("topology factory must return an online topology")
        if (
            getattr(topology, "decision_count", None) != 0
            or getattr(topology, "primitive_count", None) != 0
        ):
            raise RuntimeError("topology factory returned non-fresh episode state")

    @staticmethod
    def _validate_fresh_official_driver(
        driver: object,
        *,
        env_id: int,
        instruction: str,
        episode_serial: int,
        identity: Mapping,
    ) -> None:
        if not isinstance(driver, OfficialKeepDriver):
            raise TypeError("official driver factory must return an OfficialKeepDriver")
        session = driver.session
        if session.env_id != env_id or session.instruction != instruction:
            raise RuntimeError("official session does not match the reset episode identity")
        if driver.episode_serial != episode_serial or driver.episode_identity != dict(identity):
            raise RuntimeError("official driver does not match the reset episode identity")
        if driver.frames or driver.episode_generation is not None:
            raise RuntimeError("official driver factory returned cached prior frames")
        snapshot = session.snapshot()
        expected = {
            "physical_step": 0,
            "pending_actions": [],
            "time_ids": [],
            "previous_output_ref": None,
            "past_key_values_ref": None,
            "reset_steps": [],
            "queue_crossed_reset": False,
            "frame_count": 0,
            "terminal": False,
            "terminal_reason": None,
        }
        if any(snapshot.get(key) != value for key, value in expected.items()):
            raise RuntimeError("official driver factory returned non-fresh episode state")
        if (
            session.pending_request is not None
            or session.action_popped_for_step
            or session.popped_action_for_step is not None
        ):
            raise RuntimeError("official driver factory returned an in-flight session")

    def reset_episode(self) -> tuple[object, Mapping]:
        """Reset Habitat and atomically bind fresh state for exactly one episode.

        Factories are invoked only after the environment reset succeeds.  The
        official-driver factory receives the environment id and instruction
        from the selected Habitat episode, so callers cannot silently reuse a
        session created for a different instruction.
        """

        previous_driver = self.official_driver
        previous_session = None if previous_driver is None else previous_driver.session
        previous_topology = self.topology
        previous_token = self.episode_token
        self._episode_active = False
        self.topology = None
        self.official_driver = None
        self.observation = None
        self.exact_planner = None
        self.episode_token = None
        self._next_decision_index = 0
        self._prepared_decision_context = None
        self._preparation_poisoned = False

        observation, reset_info = self.env.reset()
        if not isinstance(reset_info, Mapping):
            raise RuntimeError("successor environment reset info must be a mapping")
        identity = getattr(self.env, "identity", None)
        env_id = getattr(self.env, "env_id", None)
        episode_serial = getattr(self.env, "_episode_serial", None)
        if (
            not isinstance(identity, Mapping)
            or not isinstance(identity.get("instruction_text"), str)
            or not identity["instruction_text"]
            or type(env_id) is not int
            or env_id < 0
            or type(episode_serial) is not int
            or episode_serial < 0
        ):
            raise RuntimeError("successor environment lacks reset episode identity")
        if getattr(self.env.successor_planner, "recovery_parent_index", None) is not None:
            raise RuntimeError("successor environment did not reset recovery state")

        topology = self.topology_factory()
        self._validate_fresh_topology(topology)
        official_driver = self.official_driver_factory(env_id, dict(identity), episode_serial)
        self._validate_fresh_official_driver(
            official_driver,
            env_id=env_id,
            instruction=identity["instruction_text"],
            episode_serial=episode_serial,
            identity=identity,
        )
        if official_driver is previous_driver or official_driver.session is previous_session:
            raise RuntimeError("official driver factory reused prior episode state")
        if topology is previous_topology:
            raise RuntimeError("topology factory reused prior episode state")

        self.topology = topology
        self.official_driver = official_driver
        self.observation = observation
        self.exact_planner = SuccessorPlanner(recovery_enabled=False)
        episode_token = secrets.token_hex(16)
        while episode_token == previous_token:
            episode_token = secrets.token_hex(16)
        self.episode_token = episode_token
        self._next_decision_index = 0
        self._episode_active = True
        return observation, reset_info

    @staticmethod
    def _single_action_transport(action: int) -> list[list[list[int]]]:
        return [[[action], *[[0] for _ in range(ACTION_CHUNK - 1)]]]

    def _plan(
        self,
        *,
        decision: RouteDecision,
        official_action: int | None,
    ):
        if self.topology is None or self.exact_planner is None:
            raise RuntimeError("successor episode must be reset before planning")
        if decision.route is RouteName.KEEP_ORIGINAL:
            if official_action is None:
                raise RuntimeError("official KEEP action was not prepared")
            decision = replace(decision, official_action=official_action)
        elif official_action is not None:
            raise RuntimeError("only KEEP_ORIGINAL may bind an official action")
        planner = self.exact_planner if self.exact_control else self.env.successor_planner
        return planner.plan(decision, self.topology)

    def _prepare_official_shadow_frame(self) -> tuple[int, dict, OfficialFrame]:
        if self.official_driver is None:
            raise RuntimeError("official driver is unavailable")
        frame = self.env.observe_official_frame()
        if not isinstance(frame, OfficialFrame):
            raise TypeError("successor environment returned an invalid official frame")
        action = self.official_driver.prepare_shadow_action(frame)
        return action, frame.trace_metadata(), frame

    def _prepare_official_shadow(self) -> tuple[int, dict]:
        action, metadata, _ = self._prepare_official_shadow_frame()
        return action, metadata

    def prepare_decision_context(
        self,
        context_builder: Callable[[OfficialFrame, int], object] | None = None,
    ) -> PreparedSuccessorDecisionContext:
        """Latch one frame before policy choice so execution cannot consume it twice."""

        if (
            not self._episode_active
            or self.official_driver is None
            or self.episode_token is None
        ):
            raise RuntimeError("successor episode must be active before preparation")
        if self._prepared_decision_context is not None:
            raise RuntimeError("successor decision context is already prepared")
        if self._preparation_poisoned:
            raise RuntimeError("successor decision preparation is poisoned until reset")
        if context_builder is not None and not callable(context_builder):
            raise TypeError("successor context builder must be callable")
        action, metadata, frame = self._prepare_official_shadow_frame()
        physical_step = self.official_driver.session.physical_step
        if frame.physical_step != physical_step:
            raise RuntimeError("prepared official frame physical step drifted")
        try:
            policy_context = (
                None if context_builder is None else context_builder(frame, action)
            )
        except Exception:
            self._preparation_poisoned = True
            raise
        prepared = PreparedSuccessorDecisionContext(
            episode_token=self.episode_token,
            physical_step=physical_step,
            decision_index=self._next_decision_index,
            official_shadow_action=action,
            frame_ref=frame.frame_ref,
            frame_metadata=dict(metadata),
            policy_context=policy_context,
        )
        self._prepared_decision_context = prepared
        return prepared

    def execute_decision(
        self,
        decision: RouteDecision,
    ) -> SuccessorBridgeResult:
        if (
            not self._episode_active
            or self.topology is None
            or self.official_driver is None
            or self.exact_planner is None
            or self.episode_token is None
        ):
            raise RuntimeError("successor episode must be reset before each episode")
        if self._preparation_poisoned:
            raise RuntimeError("successor decision preparation is poisoned until reset")
        if not isinstance(decision, RouteDecision):
            raise TypeError("successor runner requires a RouteDecision")
        if decision.official_action is not None:
            raise ValueError("policy RouteDecision must not inject an official action")
        if decision.episode_token != self.episode_token:
            raise ValueError("RouteDecision episode token does not match the live episode")
        decision_physical_step = self.official_driver.session.physical_step
        if decision.physical_step != decision_physical_step:
            raise ValueError("RouteDecision physical step is stale or out of order")
        if decision.decision_index != self._next_decision_index:
            raise ValueError("successor decision indices must be contiguous from zero")
        route = decision.route
        if self.exact_control and route is not RouteName.KEEP_ORIGINAL:
            raise ValueError("matched exact control may execute only KEEP_ORIGINAL")

        prepared_shadow = None
        prepared_frame_metadata = None
        prepared = self._prepared_decision_context
        if prepared is None and self.require_prepared_decision_context:
            raise RuntimeError("successor decision context must be prepared before execution")
        if prepared is not None:
            expected = (
                self.episode_token,
                decision_physical_step,
                self._next_decision_index,
            )
            actual = (
                prepared.episode_token,
                prepared.physical_step,
                prepared.decision_index,
            )
            if actual != expected:
                raise RuntimeError("prepared successor decision context is stale")
            prepared_shadow = prepared.official_shadow_action
            prepared_frame_metadata = dict(prepared.frame_metadata)
        elif route is RouteName.KEEP_ORIGINAL:
            prepared_shadow, prepared_frame_metadata = self._prepare_official_shadow()
        plan = self._plan(
            decision=decision,
            official_action=(
                prepared_shadow if route is RouteName.KEEP_ORIGINAL else None
            ),
        )
        self._prepared_decision_context = None

        shadow_actions = []
        official_frame_metadata = []
        official_step_events = []
        executed_actions = []
        primitive_rows = []
        collisions = []
        realized_path = []
        blocked = []
        terminated = False
        truncated = False
        end_reason = "plan_complete"
        replan_required = True
        official_stop_override = False

        for index, action in enumerate(plan.proposed_actions):
            if index == 0 and prepared_shadow is not None:
                shadow_action = prepared_shadow
                frame_metadata = prepared_frame_metadata
            else:
                shadow_action, frame_metadata = self._prepare_official_shadow()
            shadow_actions.append(shadow_action)
            official_frame_metadata.append(dict(frame_metadata))
            executed_action = action
            if shadow_action == STOP_ACTION and action != STOP_ACTION:
                executed_action = STOP_ACTION
                official_stop_override = True
            result = self.env.chunk_step(
                self._single_action_transport(executed_action),
                execute_count=1,
            )
            observations, _, terminations, truncations, infos = result
            if len(observations) != 1 or len(infos) != 1:
                raise RuntimeError("successor env returned a non-singleton batch")
            feedback = feedback_from_dict(infos[0].get("executor_feedback"))
            if (
                feedback.proposed_actions != (executed_action,)
                or len(feedback.primitives) != 1
            ):
                raise RuntimeError("successor env did not execute exactly one primitive")
            row = feedback.primitives[0]
            terminated = bool(terminations.any())
            truncated = bool(truncations.any())
            if terminated != row.terminated or truncated != row.truncated:
                raise RuntimeError("successor tensor and feedback terminal flags disagree")
            if executed_action == STOP_ACTION and not (terminated or truncated):
                raise RuntimeError("successor STOP did not end the Habitat episode")
            official_event = self.official_driver.acknowledge_physical_step(
                terminated=terminated,
                truncated=truncated,
            )
            official_step_events.append(official_event)
            self.observation = observations[0]
            executed_actions.append(executed_action)
            primitive_rows.append(row.to_dict())
            collisions.append(row.collision)
            realized_path.append(row.realized_path_m)
            blocked.append(row.blocked)
            reason, should_replan = successor_end_reason(
                action=executed_action,
                collision=row.collision,
                blocked=row.blocked,
                terminated=row.terminated,
                truncated=row.truncated,
                plan_complete=index == len(plan.proposed_actions) - 1,
            )
            if reason is not None:
                if official_stop_override:
                    end_reason, replan_required = "official_stop_override", False
                else:
                    end_reason, replan_required = reason, should_replan
                break

        if not executed_actions:
            raise RuntimeError("successor bridge executed no physical primitive")
        self.topology.update(
            executed_actions,
            collisions,
            realized_path_m=realized_path,
            blocked_forwards=blocked,
        )
        self._next_decision_index += 1
        if terminated or truncated:
            self._episode_active = False
            self._prepared_decision_context = None
        return SuccessorBridgeResult(
            episode_token=self.episode_token,
            decision_physical_step=decision_physical_step,
            plan=plan.to_dict(),
            official_shadow_actions=tuple(shadow_actions),
            official_frame_metadata=tuple(official_frame_metadata),
            official_step_events=tuple(official_step_events),
            executed_actions=tuple(executed_actions),
            primitive_feedback=tuple(primitive_rows),
            end_reason=end_reason,
            replan_required=replan_required,
            official_stop_override=official_stop_override,
            terminated=terminated,
            truncated=truncated,
            final_observation=self.observation,
        )


def make_successor_env(run_cfg: Mapping, *, split: str, episode_index: int, max_steps: int):
    """Construct only the explicitly selected successor environment."""

    if type(split) is not str or split != "train":
        raise ValueError("P8 successor runtime is train-only before gate unlock")

    from omegaconf import OmegaConf
    from rlinf.envs import get_env_cls

    env_cfg = OmegaConf.create(
        {
            "env_type": "streamvln",
            "executor_mode": "p8_successor",
            "seed": run_cfg["seed"],
            "auto_reset": False,
            "max_episode_steps": int(max_steps),
            "python_bin": run_cfg["habitat_python"],
            "server_script": run_cfg["habitat_server"],
            "asset_root": run_cfg["asset_root"],
            "dataset_split": str(split),
            "episode_index": int(episode_index),
            "video_cfg": {"save_video": False},
            "success_distance_m": 3.0,
            "habitat_sim_python": run_cfg["habitat_sim_python"],
            "habitat_env_lib": run_cfg["habitat_env_lib"],
            "egl_vendor_json": run_cfg["egl_vendor_json"],
            "rgb_size": 64,
            "max_instruction_tokens": 200,
            "instruction_bytes": 512,
            "scene_bytes": 384,
            "episode_bytes": 64,
            "env_id_base": 0,
            "official_streamvln_frame": True,
            "startup_timeout_s": run_cfg["habitat_startup_timeout_s"],
        }
    )
    env_cls = get_env_cls(env_cfg.env_type, env_cfg)
    if (
        env_cls.__name__ != "StreamVLNP8SuccessorEnv"
        or env_cls.__module__ != CANONICAL_SUCCESSOR_ENV_MODULE
    ):
        raise RuntimeError(
            "RLinf did not select the canonical P8 successor environment"
        )
    return env_cls(env_cfg, 1, 0, 1, None)


__all__ = [
    "BRIDGE_SCHEMA_NAME",
    "BRIDGE_SCHEMA_VERSION",
    "OfficialGenerationResult",
    "OfficialKeepDriver",
    "PreparedSuccessorDecisionContext",
    "SuccessorBridgeResult",
    "SuccessorRunnerBridge",
    "make_successor_env",
]
