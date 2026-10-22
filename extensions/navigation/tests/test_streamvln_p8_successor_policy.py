import ast
from dataclasses import replace
import importlib.util
import inspect
import math
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).parents[1]
MODULE_DIR = ROOT / "integrations" / "rlinf_v03"
sys.path.insert(0, str(MODULE_DIR))


POLICY_PATH = MODULE_DIR / "streamvln_p8_successor_policy.py"
TRAINER_PATH = MODULE_DIR / "train_embodied_agent.py"
REGISTRY_PATH = MODULE_DIR / "models_init_vln.py"
CONFIG_PATH = MODULE_DIR / "config_vln.py"
YAML_PATH = MODULE_DIR / "streamvln_p8_successor_bc_smoke.yaml"
ACTOR_PATH = MODULE_DIR / "streamvln_p8_successor_actor.py"


class SuccessorPolicyStaticContractTest(unittest.TestCase):
    def test_exact_policy_routes_mask_before_distribution_and_tanh_jacobian(self):
        source = POLICY_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        self.assertIn("RouteName.KEEP_ORIGINAL", source)
        self.assertIn("RouteName.WAYPOINT", source)
        self.assertIn("RouteName.STOP", source)
        policy_routes = next(
            node for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "POLICY_ROUTES" for target in node.targets)
        )
        self.assertEqual(len(policy_routes.value.elts), 3)
        self.assertNotIn("RouteName.RECOVER_PARENT,", ast.unparse(policy_routes))
        distribution = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_distribution_terms"
        )
        distribution_source = ast.unparse(distribution)
        self.assertLess(distribution_source.index("masked_fill"), distribution_source.index("return"))
        logprob = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_waypoint_logprob"
        )
        logprob_source = ast.unparse(logprob)
        self.assertIn("log_jacobian", logprob_source)
        self.assertIn("base - log_jacobian", logprob_source)
        raw, mean, log_std, scale = 0.2, -0.1, -0.5, 1.25
        base = -0.5 * (((raw - mean) / math.exp(log_std)) ** 2) - log_std - 0.5 * math.log(2 * math.pi)
        transformed = base - math.log(scale * (1.0 - math.tanh(raw) ** 2))
        self.assertTrue(math.isfinite(transformed))
        from streamvln_p8_successor_action import RouteDecision, RouteName
        from streamvln_p8_successor_protocol import SuccessorDecisionSample

        sample = SuccessorDecisionSample(
            decision=RouteDecision(
                decision_index=0,
                route=RouteName.KEEP_ORIGINAL,
                official_action=1,
                episode_token="episode-a",
                physical_step=0,
            ),
            proposed_actions=(1,),
            logprob=-0.2,
            value=0.1,
            official_shadow_action=1,
            expert_route_target=RouteName.STOP,
        )
        self.assertIs(sample.expert_route_target, RouteName.STOP)
        self.assertIsNone(sample.expert_waypoint_target)

    def test_factory_binds_worker_model_and_registry_yaml_are_opt_in(self):
        source = POLICY_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_names = {
            alias.name
            for node in tree.body
            if isinstance(node, ast.Try)
            for statement in node.body + node.handlers[0].body
            if isinstance(statement, ast.ImportFrom)
            and statement.module
            and statement.module.endswith("streamvln_p8_successor_protocol")
            for alias in statement.names
        }
        self.assertIn("SUCCESSOR_REPLAY_SCHEMA", imported_names)
        policy_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "P8SuccessorPolicy"
        )
        policy_methods = {
            node.name for node in policy_class.body if isinstance(node, ast.FunctionDef)
        }
        self.assertIn("default_forward", policy_methods)
        self.assertIn("predict_action_batch", policy_methods)
        factory = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "build_successor_rollout_controller"
        )
        factory_source = ast.unparse(factory)
        self.assertIn("getattr(worker, 'hf_model'", factory_source)
        self.assertNotIn("get_model(", factory_source)
        self.assertIn("controller.synchronized_policy = policy", factory_source)
        self.assertIn("frame.topology_snapshot", factory_source)
        self.assertNotIn("successor_policy_features", factory_source)
        self.assertNotIn("successor_route_candidate_mask", factory_source)
        self.assertIn("P8TopologyEncoder", source)
        self.assertIn(
            "torch.cat((features, global_features, topology_context)", source
        )
        registry = REGISTRY_PATH.read_text(encoding="utf-8")
        config = CONFIG_PATH.read_text(encoding="utf-8")
        yaml_text = YAML_PATH.read_text(encoding="utf-8")
        self.assertIn("STREAMVLN_P8_SUCCESSOR", registry)
        self.assertIn("STREAMVLN_P8_SUCCESSOR", config)
        self.assertIn("worker_pair: p8_successor", yaml_text)
        self.assertIn("only_eval: false", yaml_text)
        self.assertIn("val_check_interval: -1", yaml_text)
        self.assertNotIn("env.eval", yaml_text)
        self.assertIn("total_num_envs: 1", yaml_text)
        self.assertIn("rollout_epoch: 1", yaml_text)
        self.assertIn("pipeline_stage_num: 1", yaml_text)
        model_yaml = (MODULE_DIR / "streamvln_p8_successor_policy.yaml").read_text(
            encoding="utf-8"
        )
        self.assertIn("successor_feature_dim: 7173", model_yaml)
        self.assertIn("successor_topology_max_nodes: 161", model_yaml)
        self.assertIn("successor_topology_model_dim: 128", model_yaml)
        actor_tree = ast.parse(ACTOR_PATH.read_text(encoding="utf-8"))
        actor_class = next(
            node for node in actor_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "P8SuccessorActorWorker"
        )
        constructor = next(
            node for node in actor_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        self.assertEqual([argument.arg for argument in constructor.args.args], ["self", "cfg"])
        constructor_source = ast.unparse(constructor)
        self.assertIn("super().__init__(cfg)", constructor_source)
        self.assertNotIn("mark_bc_ready", ACTOR_PATH.read_text(encoding="utf-8"))
        self.assertIn("bc_readiness_digest", ACTOR_PATH.read_text(encoding="utf-8"))
        self.assertIn("require_successor_sync_state_names", ACTOR_PATH.read_text(encoding="utf-8"))

    def test_release_anchor_preflight_precedes_validate_and_factories_precede_cluster(self):
        tree = ast.parse(TRAINER_PATH.read_text(encoding="utf-8"))
        main = next(
            node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "main"
        )
        calls = [
            ast.unparse(node.func)
            for statement in main.body
            for node in ast.walk(statement)
            if isinstance(node, ast.Call)
        ]
        self.assertLess(
            calls.index("preflight_successor_release_anchor"), calls.index("validate_cfg")
        )
        self.assertLess(calls.index("select_embodied_worker_trio"), calls.index("Cluster"))
        preflight = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "preflight_successor_release_anchor"
        )
        preflight_source = ast.unparse(preflight)
        self.assertIn("_require_pinned_release_manifest_sha256", preflight_source)
        self.assertNotIn("_load_symbol", preflight_source)
        self.assertNotIn("Cluster", preflight_source)


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "Torch unavailable locally")
class SuccessorPolicyTensorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        import streamvln_p8_successor_policy as policy
        from streamvln_p8_topology import snapshot_online_topology
        from streamvln_topology import OnlineTopology

        cls.torch = torch
        cls.policy_module = policy
        cls.snapshot_online_topology = snapshot_online_topology
        cls.OnlineTopology = OnlineTopology

    def _snapshot(self, topology, *, physical_step, decision_index):
        return self.snapshot_online_topology(
            topology,
            episode_token="episode-a",
            physical_step=physical_step,
            decision_index=decision_index,
            max_nodes=5,
        )

    def test_cold_keep_mask_rejection_joint_recompute_and_bc_targets(self):
        torch = self.torch
        module = self.policy_module
        composed = module.compose_successor_features(
            frozen_feature=torch.tensor([0.25, -0.5, 0.75, 1.0]),
            official_shadow_action=2,
            physical_step=7,
            feature_physical_step=5,
        )
        self.assertEqual(composed.numel(), 9)
        self.assertTrue(torch.equal(composed[-5:-1], torch.tensor([0., 0., 1., 0.])))
        self.assertEqual(float(composed[-1]), 2.0)
        policy = module.P8SuccessorPolicy(
            feature_dim=4,
            hidden_dim=8,
            topology_max_nodes=5,
            topology_model_dim=8,
        )
        topology = self.OnlineTopology(max_decisions=4)
        root_snapshot = self._snapshot(topology, physical_step=0, decision_index=0)
        context = module.make_successor_forward_context(
            features=torch.tensor([0.1, -0.2, 0.3, 0.4]),
            topology_snapshot=root_snapshot,
            policy_version=0,
        )
        cold = policy.sample_successor_decision(
            forward_context=context,
            episode_token="episode-a",
            physical_step=0,
            decision_index=0,
            official_shadow_action=2,
            deterministic=False,
            expert_route_target=module.RouteName.STOP,
        )
        self.assertIs(cold.decision.route, module.RouteName.KEEP_ORIGINAL)
        self.assertEqual(cold.proposed_actions, (2,))
        self.assertEqual(cold.logprob, 0.0)
        self.assertIs(cold.expert_route_target, module.RouteName.STOP)
        self.assertTrue(
            torch.equal(
                cold.forward_context["route_candidate_mask"],
                torch.tensor([True, False, False]),
            )
        )
        self.assertTrue(
            torch.equal(
                cold.forward_context["expert_route_candidate_mask"],
                torch.tensor([True, True, True]),
            )
        )
        cold_actor_context = {
            key: value.reshape(1, 1) if value.ndim == 0 else value.unsqueeze(0).unsqueeze(1)
            for key, value in cold.forward_context.items()
        }
        cold_output = policy(
            forward_inputs=cold_actor_context,
            route_actions=torch.tensor([[module.ROUTE_TO_INDEX[module.RouteName.KEEP_ORIGINAL]]]),
            waypoint_targets=torch.zeros(1, 1, 2),
            expert_waypoint_targets=torch.zeros(1, 1, 2),
        )
        self.assertEqual(float(cold_output["route_logprobs"].item()), 0.0)
        cold_route_loss = torch.nn.functional.cross_entropy(
            cold_output["raw_route_logits"].masked_fill(
                ~cold_output["expert_route_candidate_mask"], -torch.inf
            ).reshape(-1, 3),
            torch.tensor([module.ROUTE_TO_INDEX[module.RouteName.STOP]]),
        )
        cold_route_loss.backward()
        self.assertGreater(float(policy.route_head.bias.grad.abs().sum()), 0.0)
        with self.assertRaisesRegex(ValueError, "all false"):
            module.make_successor_forward_context(
                features=torch.ones(4),
                topology_snapshot=replace(
                    root_snapshot,
                    route_candidate_mask=(False, False, False),
                ),
                policy_version=0,
            )

        policy.set_global_step(7)
        readiness_digest = "12" * 32
        policy.promote_bc_readiness(readiness_digest)
        rollout_policy = module.P8SuccessorPolicy(
            feature_dim=4,
            hidden_dim=8,
            topology_max_nodes=5,
            topology_model_dim=8,
            expected_bc_readiness_digest=readiness_digest,
        )
        actor_state = policy.state_dict()
        self.assertIn("bc_ready", actor_state)
        self.assertIn("bc_readiness_digest", actor_state)
        self.assertIn("policy_version", actor_state)
        rollout_policy.load_state_dict(actor_state)
        self.assertTrue(rollout_policy.is_bc_ready())
        self.assertEqual(
            rollout_policy.bc_readiness_receipt_sha256(), readiness_digest
        )
        self.assertEqual(int(rollout_policy.policy_version.item()), 7)
        context = module.make_successor_forward_context(
            features=torch.tensor([0.1, -0.2, 0.3, 0.4]),
            topology_snapshot=root_snapshot,
            policy_version=7,
        )
        with torch.no_grad():
            policy.route_head.weight.zero_()
            policy.route_head.bias.copy_(torch.tensor([-5.0, 5.0, -5.0]))
            policy.waypoint_mean_head.weight.zero_()
            policy.waypoint_mean_head.bias.copy_(torch.tensor([0.3, -0.2]))
        sample = policy.sample_successor_decision(
            forward_context=context,
            episode_token="episode-a",
            physical_step=0,
            decision_index=0,
            official_shadow_action=1,
            deterministic=True,
            expert_route_target=module.RouteName.KEEP_ORIGINAL,
        )
        self.assertIs(sample.decision.route, module.RouteName.WAYPOINT)
        actor_context = {
            key: value.unsqueeze(0).unsqueeze(1) if value.ndim == 0 else value.unsqueeze(0).unsqueeze(1)
            for key, value in context.items()
        }
        # Scalars stack to [B,1], vectors to [B,1,D].
        actor_context["schema_version"] = context["schema_version"].reshape(1, 1)
        actor_context["policy_version"] = context["policy_version"].reshape(1, 1)
        output = policy(
            forward_inputs=actor_context,
            route_actions=torch.tensor([[module.ROUTE_TO_INDEX[module.RouteName.WAYPOINT]]]),
            waypoint_targets=torch.tensor([[sample.decision.waypoint_local_m]]),
        )
        recomputed = output["route_logprobs"] + output["waypoint_logprobs"]
        self.assertTrue(torch.allclose(recomputed, torch.tensor([[sample.logprob]]), atol=1e-5))
        bc_output = policy(
            forward_inputs=actor_context,
            route_actions=torch.tensor([[module.ROUTE_TO_INDEX[module.RouteName.KEEP_ORIGINAL]]]),
            waypoint_targets=torch.zeros(1, 1, 2),
            expert_waypoint_targets=torch.zeros(1, 1, 2),
        )
        bc_route_loss = torch.nn.functional.cross_entropy(
            bc_output["raw_route_logits"].masked_fill(
                ~bc_output["expert_route_candidate_mask"], -torch.inf
            ).reshape(-1, 3),
            torch.tensor([module.ROUTE_TO_INDEX[module.RouteName.KEEP_ORIGINAL]]),
        )
        self.assertTrue(torch.isfinite(bc_route_loss))
        self.assertIn("expert_waypoint_logprobs", bc_output)

        topology.update(
            (1, 1, 1, 1),
            (False, False, False, False),
            realized_path_m=(0.25, 0.25, 0.25, 0.25),
            blocked_forwards=(False, False, False, False),
        )
        moved_context = module.make_successor_forward_context(
            features=torch.tensor([0.1, -0.2, 0.3, 0.4]),
            topology_snapshot=self._snapshot(
                topology, physical_step=4, decision_index=1
            ),
            policy_version=7,
        )
        moved_actor_context = {
            key: value.reshape(1, 1)
            if value.ndim == 0
            else value.unsqueeze(0).unsqueeze(1)
            for key, value in moved_context.items()
        }
        moved_output = policy(
            forward_inputs=moved_actor_context,
            route_actions=torch.tensor([[module.ROUTE_TO_INDEX[module.RouteName.KEEP_ORIGINAL]]]),
            waypoint_targets=torch.zeros(1, 1, 2),
        )
        self.assertTrue(
            not torch.allclose(output["waypoint_predictions"], moved_output["waypoint_predictions"])
            or not torch.allclose(output["values"], moved_output["values"])
        )


if __name__ == "__main__":
    unittest.main()
