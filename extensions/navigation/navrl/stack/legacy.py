"""Compatibility bridge from the existing ``NavEnvAdapter`` contract."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from .contracts import EpisodeIdentity, EpisodeMetrics, PrimitiveStep, TaskReset
from .interfaces import TaskAdapter


class LegacyNavTaskAdapter(TaskAdapter):
    """Expose an existing navigation environment through the S1 task contract.

    The bridge owns reset provenance and action counts.  Pose is removed from
    policy observations by default even when a diagnostic environment exposes
    it through its legacy observation.
    """

    def __init__(
        self,
        environment: object,
        *,
        env_slot: int = 0,
        observation_transform: Callable[[object], object] | None = None,
        hide_pose: bool = True,
    ) -> None:
        if type(env_slot) is not int or env_slot < 0:
            raise ValueError("env_slot must be a non-negative integer")
        if observation_transform is not None and not callable(observation_transform):
            raise TypeError("observation_transform must be callable")
        self.environment = environment
        self.env_slot = env_slot
        self.observation_transform = observation_transform
        self.hide_pose = bool(hide_pose)
        self._reset_index = 0
        self._identity: EpisodeIdentity | None = None
        self._physical_step = 0
        self._action_counts: dict[str, int] = {}

    def _policy_observation(self, observation: object) -> object:
        if isinstance(observation, Mapping):
            observation = dict(observation)
            if self.hide_pose and "pose" in observation:
                observation["pose"] = None
        if self.observation_transform is not None:
            observation = self.observation_transform(observation)
        return observation

    def reset(self, episode: object | None = None, *, seed: int | None = None) -> TaskReset:
        observation = self.environment.reset(episode, seed=seed)
        if not isinstance(observation, Mapping):
            raise TypeError("legacy reset observation must expose episode_id and step_id")
        episode_id = observation.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError("legacy reset observation has no episode_id")
        if observation.get("step_id") != 0:
            raise ValueError("legacy reset observation must start at physical step zero")
        self._reset_index += 1
        self._identity = EpisodeIdentity(
            episode_id=episode_id,
            reset_token=f"slot={self.env_slot}:reset={self._reset_index}",
            env_slot=self.env_slot,
        )
        self._physical_step = 0
        self._action_counts = {}
        return TaskReset(self._identity, self._policy_observation(observation))

    def step(self, primitive_action: object) -> PrimitiveStep:
        if self._identity is None:
            raise RuntimeError("call reset before stepping the task")
        observation, reward, terminated, truncated, info = self.environment.step(
            primitive_action
        )
        if not isinstance(observation, Mapping) or not isinstance(info, Mapping):
            raise TypeError("legacy step must return mapping observation and info")
        self._physical_step += 1
        if observation.get("episode_id") != self._identity.episode_id:
            raise RuntimeError("legacy environment changed episode identity during a rollout")
        if observation.get("step_id") != self._physical_step:
            raise RuntimeError("legacy environment physical step is not contiguous")
        action_name = getattr(primitive_action, "value", str(primitive_action))
        action_name = str(action_name)
        self._action_counts[action_name] = self._action_counts.get(action_name, 0) + 1
        return PrimitiveStep(
            identity=self._identity,
            physical_step=self._physical_step,
            observation=self._policy_observation(observation),
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=dict(info),
        )

    def get_metrics(self) -> EpisodeMetrics:
        if self._identity is None:
            raise RuntimeError("call reset before reading task metrics")
        info = self.environment.get_metrics()
        if not isinstance(info, Mapping):
            raise TypeError("legacy metrics must be a mapping")
        return EpisodeMetrics(
            identity=self._identity,
            path_length_m=float(info["path_length"]),
            collisions=int(info["collision_count"]),
            stop_called=bool(info["stop_called"]),
            timed_out=bool(info["timeout"]),
            action_counts=dict(self._action_counts),
            success=bool(info["success"]),
            spl=float(info["spl"]),
            navigation_error_m=float(info["geodesic_distance"]),
            extras={
                key: info.get(key)
                for key in (
                    "distance_to_goal",
                    "geodesic_distance",
                    "invalid_action",
                    "termination_reason",
                )
            },
        )

    def close(self) -> None:
        self.environment.close()
