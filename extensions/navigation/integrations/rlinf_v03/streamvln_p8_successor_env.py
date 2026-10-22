"""Version-isolated Habitat executor for P8 successor route plans."""

from __future__ import annotations

import numpy as np
import torch

if __package__:
    from .streamvln_env import ACTION_CHUNK, StreamVLNEnv
    from .streamvln_executor_contract import chunk_feedback, primitive_execution
    from .streamvln_p8_successor_action import (
        CompiledRoute,
        RouteDecision,
        RouteName,
        SuccessorPlanner,
        pack_fixed_action_chunk,
    )
    from .streamvln_p8_successor_executor import (
        PRODUCTION_SUCCESSOR_EXECUTION_SCHEMA_NAME,
        SUCCESSOR_EXECUTION_SCHEMA_VERSION,
        successor_end_reason,
    )
else:  # Explicit direct-file mode for local contract tests only.
    from streamvln_env import ACTION_CHUNK, StreamVLNEnv
    from streamvln_executor_contract import chunk_feedback, primitive_execution
    from streamvln_p8_successor_action import (
        CompiledRoute,
        RouteDecision,
        RouteName,
        SuccessorPlanner,
        pack_fixed_action_chunk,
    )
    from streamvln_p8_successor_executor import (
        PRODUCTION_SUCCESSOR_EXECUTION_SCHEMA_NAME,
        SUCCESSOR_EXECUTION_SCHEMA_VERSION,
        successor_end_reason,
    )


class StreamVLNP8SuccessorEnv(StreamVLNEnv):
    """Execute a semantic 1..4 prefix and discard its tail on blocked feedback."""

    def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info):
        if cfg.get("official_streamvln_frame", False) is not True:
            raise ValueError(
                "P8 successor requires the private pinned StreamVLN frame channel"
            )
        self.successor_planner = SuccessorPlanner()
        super().__init__(cfg, num_envs, seed_offset, total_num_processes, worker_info)

    def reset(self, **kwargs):
        self.successor_planner.reset_episode()
        return super().reset(**kwargs)

    def _base_primitive_step(self, action: int):
        # Skip StreamVLNEnv and invoke its VLNEnv parent exactly once.
        return super(StreamVLNEnv, self).step([[int(action)]])

    def chunk_step(self, chunk_actions, execute_count=None):
        """Use the existing RLinf transport and add a collision/blockage barrier."""

        array = np.asarray(chunk_actions)
        if array.shape != (1, ACTION_CHUNK, 1):
            raise ValueError("successor chunk actions must have shape [1, 4, 1]")
        if not np.issubdtype(array.dtype, np.integer) or np.issubdtype(
            array.dtype, np.bool_
        ):
            raise TypeError("successor chunk actions must be integer categories")
        proposed = [int(value) for value in array.reshape(-1)]
        if any(not 0 <= value < 4 for value in proposed):
            raise ValueError("successor action outside 0..3")
        if type(execute_count) is not int or not 1 <= execute_count <= ACTION_CHUNK:
            raise ValueError("successor execute_count is required and must be an integer in 1..4")
        proposed = proposed[:execute_count]
        if 0 in proposed and proposed != [0]:
            raise ValueError("successor STOP must be the sole semantic primitive")

        rewards = []
        terminations = []
        truncations = []
        execution_trace = []
        final_obs = None
        final_infos = {}
        replan_reason = "plan_complete"

        for index, action in enumerate(proposed):
            final_obs, reward, terminated, truncated, final_infos = self._base_primitive_step(action)
            path_length = final_infos.get("path_length")
            collision = final_infos.get("collision")
            if path_length is None or not hasattr(path_length, "item"):
                raise RuntimeError("successor executor did not report realized path length")
            if collision is None or not hasattr(collision, "item"):
                raise RuntimeError("successor executor did not report collision feedback")
            row = primitive_execution(
                primitive_index=index,
                action=action,
                previous_path_length_m=self._previous_path_length,
                current_path_length_m=float(path_length.item()),
                collision=bool(collision.item()),
                environment_reward=float(reward.item()),
                terminated=bool(terminated.item()),
                truncated=bool(truncated.item()),
            )
            rewards.append(reward)
            terminations.append(terminated)
            truncations.append(truncated)
            execution_trace.append(row)
            self._previous_path_length = float(path_length.item())

            reason, _ = successor_end_reason(
                action=action,
                collision=row.collision,
                blocked=row.blocked,
                terminated=row.terminated,
                truncated=row.truncated,
                plan_complete=index == len(proposed) - 1,
            )
            if reason is not None:
                replan_reason = reason
                break

        if final_obs is None or not execution_trace:
            raise RuntimeError("successor decision executed no physical primitive")
        executed = [row.action for row in execution_trace]
        self._previous_actions = executed
        self._terminal_frozen = bool(terminations[-1].item()) or bool(truncations[-1].item())
        final_infos = dict(final_infos)
        final_infos["chunk_collisions"] = [row.collision for row in execution_trace]
        final_infos["executor_feedback"] = chunk_feedback(
            proposed, execution_trace
        ).to_dict()
        final_infos["successor_replan_reason"] = replan_reason
        final_infos["successor_replan_required"] = bool(
            replan_reason in {"plan_complete", "collision", "blocked"}
        )
        final_infos["successor_collision_interrupted"] = bool(
            replan_reason == "collision" and len(executed) < len(proposed)
        )
        final_infos["successor_execution"] = {
            "schema_name": PRODUCTION_SUCCESSOR_EXECUTION_SCHEMA_NAME,
            "schema_version": SUCCESSOR_EXECUTION_SCHEMA_VERSION,
            "replan_reason": replan_reason,
            "replan_required": final_infos["successor_replan_required"],
            "collision_interrupted": final_infos["successor_collision_interrupted"],
        }

        while len(rewards) < ACTION_CHUNK:
            rewards.append(torch.zeros_like(rewards[-1]))
            terminations.append(torch.zeros_like(terminations[-1]))
            truncations.append(torch.zeros_like(truncations[-1]))
        return (
            [self._stream_obs(final_obs, episode_start=False)],
            torch.stack(rewards, dim=1),
            torch.stack(terminations, dim=1),
            torch.stack(truncations, dim=1),
            [final_infos],
        )

    def successor_step(self, plan: CompiledRoute):
        """Plan-aware entry point used by the successor diagnostic runtime."""

        if not isinstance(plan, CompiledRoute):
            raise TypeError("successor_step requires a CompiledRoute")
        if plan.requested_route is RouteName.KEEP_ORIGINAL:
            raise ValueError(
                "official KEEP must execute through the OfficialKeepDriver bridge"
            )
        padded, execute_count = pack_fixed_action_chunk(plan)
        result = self.chunk_step(
            np.asarray(padded, dtype=np.int64).reshape(1, ACTION_CHUNK, 1),
            execute_count=execute_count,
        )
        infos = result[-1]
        if len(infos) != 1 or not isinstance(infos[0], dict):
            raise RuntimeError("successor executor returned an invalid info batch")
        infos[0]["successor_plan"] = plan.to_dict()
        infos[0]["successor_execution"]["plan"] = plan.to_dict()
        if infos[0]["executor_feedback"]["proposed_actions"] != list(plan.proposed_actions):
            raise RuntimeError("successor transport changed the semantic plan prefix")
        return result

    def plan_and_step(self, decision: RouteDecision, topology):
        """Own recovery state at the lower layer and execute the resulting route."""

        if not isinstance(decision, RouteDecision):
            raise TypeError("plan_and_step requires a RouteDecision")
        if decision.route is RouteName.KEEP_ORIGINAL:
            raise ValueError(
                "plan_and_step cannot bind or execute an official KEEP action"
            )
        plan = self.successor_planner.plan(decision, topology)
        return self.successor_step(plan)
