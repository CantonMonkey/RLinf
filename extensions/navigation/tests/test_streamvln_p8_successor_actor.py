import importlib.util
import inspect
import math
from pathlib import Path
import sys
import types
import unittest

ROOT = Path(__file__).parents[1]
MODULE_DIR = ROOT / "integrations" / "rlinf_v03"
sys.path.insert(0, str(MODULE_DIR))


ADV_REGISTRY = {}
LOSS_REGISTRY = {}


def _register(registry, name):
    def decorator(fn):
        registry[name] = fn
        return fn

    return decorator


def _load_actor_module():
    rlinf = types.ModuleType("rlinf")
    algorithms = types.ModuleType("rlinf.algorithms")
    registry = types.ModuleType("rlinf.algorithms.registry")
    registry.ADV_REGISTRY = ADV_REGISTRY
    registry.LOSS_REGISTRY = LOSS_REGISTRY
    registry.register_advantage = lambda name: _register(ADV_REGISTRY, name)
    registry.register_policy_loss = lambda name: _register(LOSS_REGISTRY, name)
    registry.get_adv_and_returns = ADV_REGISTRY.__getitem__
    registry.get_policy_loss = LOSS_REGISTRY.__getitem__
    utils = types.ModuleType("rlinf.utils")
    metric_utils = types.ModuleType("rlinf.utils.metric_utils")
    metric_utils.append_to_dict = lambda target, values: target.update(values)
    metric_utils.compute_split_num = lambda send, recv: send // recv
    nested = types.ModuleType("rlinf.utils.nested_dict_process")
    nested.put_tensor_device = lambda value, device: value
    workers = types.ModuleType("rlinf.workers")
    actor_pkg = types.ModuleType("rlinf.workers.actor")
    actor_base = types.ModuleType("rlinf.workers.actor.fsdp_actor_worker")
    actor_base.EmbodiedFSDPActor = type("EmbodiedFSDPActor", (), {})
    for name, module in {
        "rlinf": rlinf,
        "rlinf.algorithms": algorithms,
        "rlinf.algorithms.registry": registry,
        "rlinf.utils": utils,
        "rlinf.utils.metric_utils": metric_utils,
        "rlinf.utils.nested_dict_process": nested,
        "rlinf.workers": workers,
        "rlinf.workers.actor": actor_pkg,
        "rlinf.workers.actor.fsdp_actor_worker": actor_base,
    }.items():
        sys.modules[name] = module
    path = MODULE_DIR / "streamvln_p8_successor_actor.py"
    spec = importlib.util.spec_from_file_location("streamvln_p8_successor_actor_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ACTOR = _load_actor_module()

from streamvln_p8_successor_action import (  # noqa: E402
    CompiledRoute,
    RouteDecision,
    RouteName,
)
from streamvln_p8_successor_protocol import (  # noqa: E402
    MacroTransition,
    SuccessorDecisionSample,
    SuccessorMacroAccumulator,
    SuccessorStepRequest,
    SuccessorStepResult,
)


def _sample(*, value=0.5):
    decision = RouteDecision(
        decision_index=0,
        route=RouteName.WAYPOINT,
        waypoint_local_m=(1.25, -0.5),
        episode_token="episode-a",
        physical_step=0,
    )
    plan = CompiledRoute(
        decision_index=0,
        requested_route=RouteName.WAYPOINT,
        executed_route=RouteName.WAYPOINT,
        waypoint_local_m=(1.25, -0.5),
        proposed_actions=(1, 2, 1, 1),
        source="test_metric_waypoint",
        recovery_override=False,
        recovery_parent_index=None,
        target_distance_m=math.hypot(1.25, -0.5),
        target_bearing_rad=math.atan2(-0.5, 1.25),
    )
    return SuccessorDecisionSample(
        decision=decision,
        proposed_actions=(1, 2, 1, 1),
        logprob=-0.75,
        value=value,
        official_shadow_action=1,
        forward_context={"features": (1.0, 2.0)},
        compiled_route=plan,
    )


def _macro(*, duration=4, terminated=False, truncated=False):
    rewards = (1.0, 2.0, 3.0, 4.0)
    actions = (1, 2, 1, 1)
    return MacroTransition(
        episode_token="episode-a",
        physical_step=0,
        decision_index=0,
        decision_mask=True,
        decision_sample=_sample(),
        compiled_route=_sample().compiled_route,
        executed_actions=actions[:duration] + (0,) * (4 - duration),
        executed_mask=(True,) * duration + (False,) * (4 - duration),
        primitive_rewards=rewards[:duration] + (0.0,) * (4 - duration),
        primitive_collisions=(False,) * 4,
        primitive_blocked=(False,) * 4,
        realized_path_m=(0.25,) * duration + (0.0,) * (4 - duration),
        duration=duration,
        terminated=terminated,
        truncated=truncated,
        end_reason="truncated" if truncated else "plan_complete",
        replan_required=not terminated,
    )


class SuccessorActorAdapterTest(unittest.TestCase):
    def test_four_step_macro_becomes_one_complete_decision_row(self):
        adapter = ACTOR.SuccessorActorReplayAdapter(gamma=0.9)
        row = adapter.convert(macro=_macro(), next_value=0.8, stage_id=0, env_slot=0)
        batch = adapter.build_batch([row])

        self.assertEqual(len(batch.rows), 1)
        self.assertEqual(row.route, RouteName.WAYPOINT)
        self.assertEqual(row.waypoint_local_m, (1.25, -0.5))
        self.assertEqual(row.proposed_actions, (1, 2, 1, 1))
        self.assertEqual(row.old_joint_logprob, -0.75)
        self.assertEqual(row.old_value, 0.5)
        self.assertEqual(row.official_shadow_action, 1)
        self.assertEqual(row.duration, 4)
        self.assertEqual(row.executed_mask, (True, True, True, True))
        self.assertAlmostEqual(row.aggregated_reward, 1 + 0.9 * 2 + 0.9**2 * 3 + 0.9**3 * 4)
        self.assertEqual(row.next_value, 0.8)
        self.assertTrue(row.decision_mask and row.valid_padding)
        self.assertEqual(row.forward_context["features"], (1.0, 2.0))

    def test_continuation_feedback_emits_zero_rows(self):
        accumulator = SuccessorMacroAccumulator("episode-a")
        request = SuccessorStepRequest(
            episode_token="episode-a",
            physical_step=0,
            decision_index=0,
            needs_decision=True,
            decision_mask=True,
            official_shadow_action=1,
            decision_sample=_sample(),
        )
        command = accumulator.prepare_step(request)
        macro = accumulator.accept_feedback(
            SuccessorStepResult(
                episode_token="episode-a",
                physical_step=0,
                decision_index=0,
                needs_decision=False,
                decision_mask=True,
                executed_action=command.executed_action,
                reward=1.0,
            )
        )
        rows = []
        if macro is not None:
            rows.append(
                ACTOR.SuccessorActorReplayAdapter(0.9).convert(
                    macro=macro, next_value=0.0, stage_id=0, env_slot=0
                )
            )
        self.assertEqual(rows, [])

    def test_duration_gae_k1_kgt1_and_truncation_bootstrap(self):
        self.assertIs(
            ADV_REGISTRY[ACTOR.SUCCESSOR_ADVANTAGE_NAME], ACTOR.successor_duration_gae
        )
        advantages, returns = ACTOR.successor_duration_gae_core(
            rewards=[1.0, 2.0],
            values=[0.5, 0.25],
            next_values=[0.25, 4.0],
            durations=[1, 2],
            terminated=[False, False],
            truncated=[False, True],
            gamma=0.9,
            gae_lambda=0.8,
        )
        delta_last = 2.0 + 0.9**2 * 4.0 - 0.25
        delta_first = 1.0 + 0.9 * 0.25 - 0.5
        self.assertTrue(math.isclose(advantages[1], delta_last))
        self.assertTrue(
            math.isclose(advantages[0], delta_first + (0.9 * 0.8) * delta_last)
        )
        self.assertTrue(math.isclose(returns[1], delta_last + 0.25))
        final_observation = {"rgb": "terminal-frame"}
        truncated_row = ACTOR.SuccessorActorReplayAdapter(0.9).convert(
            macro=_macro(duration=2, truncated=True),
            next_value=4.0,
            stage_id=0,
            env_slot=0,
            final_observation=final_observation,
        )
        self.assertIs(truncated_row.truncation_final_observation, final_observation)
        with self.assertRaisesRegex(ValueError, "final observation"):
            ACTOR.SuccessorActorReplayAdapter(0.9).convert(
                macro=_macro(duration=2, truncated=True),
                next_value=4.0,
                stage_id=0,
                env_slot=0,
            )
        registered = ADV_REGISTRY[ACTOR.SUCCESSOR_ADVANTAGE_NAME]
        ADV_REGISTRY[ACTOR.SUCCESSOR_ADVANTAGE_NAME] = lambda **kwargs: None
        try:
            with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
                ACTOR.install_successor_algorithm_registry()
        finally:
            ADV_REGISTRY[ACTOR.SUCCESSOR_ADVANTAGE_NAME] = registered

    def test_joint_ratio_and_decision_mask_exclude_primitive_rows(self):
        self.assertIs(
            LOSS_REGISTRY[ACTOR.SUCCESSOR_POLICY_LOSS_NAME],
            ACTOR.successor_joint_ppo_loss,
        )
        waypoint_ratio, waypoint_loss = ACTOR.successor_joint_policy_core(
            route_logprob=math.log(2.0),
            waypoint_logprob=math.log(3.0),
            old_logprob=0.0,
            advantage=1.0,
            clip_ratio_low=0.2,
            clip_ratio_high=0.2,
            waypoint_mask=True,
        )
        keep_ratio, keep_loss = ACTOR.successor_joint_policy_core(
            route_logprob=math.log(2.0),
            waypoint_logprob=20.0,
            old_logprob=0.0,
            advantage=1.0,
            clip_ratio_low=0.2,
            clip_ratio_high=0.2,
            waypoint_mask=False,
        )
        stop_ratio, _ = ACTOR.successor_joint_policy_core(
            route_logprob=math.log(2.0),
            waypoint_logprob=-20.0,
            old_logprob=0.0,
            advantage=1.0,
            clip_ratio_low=0.2,
            clip_ratio_high=0.2,
            waypoint_mask=False,
        )
        decision_mask = [True, False]
        valid_padding = [True, True]
        active = [d and p for d, p in zip(decision_mask, valid_padding)]
        masked_loss = sum(
            loss for loss, enabled in zip([keep_loss, -1e9], active) if enabled
        ) / sum(active)
        self.assertTrue(math.isclose(waypoint_ratio, 6.0))
        self.assertTrue(math.isclose(waypoint_loss, -1.2))
        self.assertTrue(math.isclose(keep_ratio, 2.0))
        self.assertTrue(math.isclose(stop_ratio, 2.0))
        self.assertTrue(math.isclose(masked_loss, -1.2))
        tensor_wrapper = inspect.getsource(ACTOR.successor_joint_ppo_loss)
        self.assertIn("successor_joint_policy_core", tensor_wrapper)
        self.assertIn("mean(dim=-1)", tensor_wrapper)
        self.assertNotIn("mean(dim=-1, keepdim=True)", tensor_wrapper)
        tensor_core = inspect.getsource(ACTOR.successor_joint_policy_core)
        self.assertIn("torch.exp", tensor_core)
        self.assertIn("torch.where", tensor_core)

    def test_persistent_readiness_and_version_are_required_for_sync(self):
        ACTOR.require_successor_sync_state_names(
            [
                "backbone.0.weight",
                "policy_version",
                "bc_ready",
                "bc_readiness_digest",
            ]
        )
        with self.assertRaisesRegex(RuntimeError, "bc_ready"):
            ACTOR.require_successor_sync_state_names(["policy_version"])
        with self.assertRaisesRegex(RuntimeError, "policy_version"):
            ACTOR.require_successor_sync_state_names(
                ["bc_ready", "bc_readiness_digest"]
            )
        with self.assertRaisesRegex(RuntimeError, "bc_readiness_digest"):
            ACTOR.require_successor_sync_state_names(["bc_ready", "policy_version"])

    @unittest.skipUnless(ACTOR.torch is not None, "Torch unavailable locally")
    def test_waypoint_bc_mask_does_not_broadcast_across_batch_rows(self):
        torch = ACTOR.torch
        waypoint_nll, waypoint_regression = ACTOR.successor_bc_waypoint_terms(
            expert_waypoint_logprobs=torch.tensor([[-2.0], [-99.0]]),
            waypoint_predictions=torch.tensor(
                [[[1.0, 3.0]], [[100.0, 100.0]]]
            ),
            expert_waypoint_targets=torch.tensor(
                [[[0.0, 1.0]], [[0.0, 0.0]]]
            ),
            expert_waypoint_mask=torch.tensor([[True], [False]]),
        )
        self.assertEqual(tuple((
            torch.tensor([[[1.0, 3.0]], [[100.0, 100.0]]])
            - torch.tensor([[[0.0, 1.0]], [[0.0, 0.0]]])
        ).square().mean(dim=-1).shape), (2, 1))
        self.assertTrue(torch.allclose(waypoint_nll, torch.tensor(2.0)))
        self.assertTrue(torch.allclose(waypoint_regression, torch.tensor(2.5)))

        zeros = torch.zeros(2, 1)
        ppo_loss, metrics = ACTOR.successor_joint_ppo_loss(
            route_logprobs=zeros,
            waypoint_logprobs=zeros,
            old_logprobs=zeros,
            advantages=zeros,
            values=zeros,
            old_values=zeros,
            returns=zeros,
            waypoint_predictions=torch.tensor(
                [[[1.0, 3.0]], [[100.0, 100.0]]]
            ),
            waypoint_targets=torch.tensor(
                [[[0.0, 1.0]], [[0.0, 0.0]]]
            ),
            waypoint_mask=torch.tensor([[True], [False]]),
            decision_mask=torch.tensor([[True], [True]]),
            valid_padding=torch.tensor([[True], [True]]),
            clip_ratio_low=0.2,
            clip_ratio_high=0.2,
            value_clip=0.2,
            waypoint_coef=1.0,
            value_coef=0.0,
        )
        self.assertTrue(torch.allclose(ppo_loss, torch.tensor(2.5)))
        self.assertTrue(
            torch.allclose(metrics["actor/waypoint_loss"], torch.tensor(2.5))
        )


if __name__ == "__main__":
    unittest.main()
