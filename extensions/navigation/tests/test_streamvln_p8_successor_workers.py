import ast
import asyncio
import copy
from dataclasses import dataclass, replace
import inspect
import importlib.util
import math
from pathlib import Path
import sys
import types
import unittest

ROOT = Path(__file__).parents[1]
MODULE_DIR = ROOT / "integrations" / "rlinf_v03"
sys.path.insert(0, str(MODULE_DIR))


class _PendingReceive:
    def __init__(self, channel):
        self.channel = channel

    async def async_wait(self):
        return await self.channel.queue.get()


class _GenericWorker:
    def init_worker(self):
        return None

    def recv_from(self, *, channel, **kwargs):
        return _PendingReceive(channel)

    def send_to(self, *, channel, data, **kwargs):
        channel.sent.append(data)
        channel.queue.put_nowait(data)


class _GenericEnvWorker(_GenericWorker):
    pass


class _GenericRolloutWorker(_GenericWorker):
    pass


class _GenericActorWorker(_GenericWorker):
    pass


def _load_workers():
    rlinf = types.ModuleType("rlinf")
    scheduler = types.ModuleType("rlinf.scheduler")
    scheduler.Channel = object
    workers = types.ModuleType("rlinf.workers")
    env_pkg = types.ModuleType("rlinf.workers.env")
    env_module = types.ModuleType("rlinf.workers.env.env_worker")
    env_module.EnvWorker = _GenericEnvWorker
    rollout_pkg = types.ModuleType("rlinf.workers.rollout")
    hf_pkg = types.ModuleType("rlinf.workers.rollout.hf")
    hf_module = types.ModuleType("rlinf.workers.rollout.hf.huggingface_worker")
    hf_module.MultiStepRolloutWorker = _GenericRolloutWorker
    for name, module in {
        "rlinf": rlinf,
        "rlinf.scheduler": scheduler,
        "rlinf.workers": workers,
        "rlinf.workers.env": env_pkg,
        "rlinf.workers.env.env_worker": env_module,
        "rlinf.workers.rollout": rollout_pkg,
        "rlinf.workers.rollout.hf": hf_pkg,
        "rlinf.workers.rollout.hf.huggingface_worker": hf_module,
    }.items():
        sys.modules[name] = module
    topology_stub = None
    if importlib.util.find_spec("torch") is None:
        topology_stub = types.ModuleType("streamvln_p8_topology")

        @dataclass(frozen=True)
        class _Snapshot:
            episode_token: str
            physical_step: int
            decision_index: int
            route_candidate_mask: tuple = (True, True, True)
            global_features: tuple = ()
            tensors: object = None

        def _snapshot(topology, *, episode_token, physical_step, decision_index, max_nodes):
            if topology.decision_count != decision_index:
                raise ValueError("topology decision count disagrees with frame identity")
            if len(topology.nodes) > max_nodes:
                raise RuntimeError("causal topology exceeds the fixed policy node extent")
            return _Snapshot(episode_token, physical_step, decision_index)

        topology_stub.P8CausalTopologySnapshot = _Snapshot
        topology_stub.snapshot_online_topology = _snapshot
        sys.modules["streamvln_p8_topology"] = topology_stub
    path = MODULE_DIR / "streamvln_p8_successor_workers.py"
    spec = importlib.util.spec_from_file_location("streamvln_p8_successor_workers", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if topology_stub is not None:
        sys.modules.pop("streamvln_p8_topology", None)
    sys.modules["rlinf.workers.streamvln_p8_successor_workers"] = module
    return module


WORKERS = _load_workers()

from streamvln_executor_contract import chunk_feedback, primitive_execution  # noqa: E402
from streamvln_p8_successor_action import (  # noqa: E402
    ParentRecoveryUnavailable,
    RouteDecision,
    RouteName,
    SuccessorPlanner,
)
from streamvln_p8_successor_protocol import SuccessorDecisionSample  # noqa: E402
from streamvln_topology import causal_topology_payload  # noqa: E402


def _topology_snapshot(episode_token, physical_step, decision_index, *, max_nodes=5):
    topology = WORKERS.OnlineTopology(max_decisions=max(4, decision_index + 1))
    for _ in range(decision_index):
        topology.update(
            (2,),
            (False,),
            realized_path_m=(0.0,),
            blocked_forwards=(False,),
        )
    return WORKERS.snapshot_online_topology(
        topology,
        episode_token=episode_token,
        physical_step=physical_step,
        decision_index=decision_index,
        max_nodes=max_nodes,
    )


def _load_selector():
    path = MODULE_DIR / "train_embodied_agent.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {
            "_worker_cfg_get",
            "select_embodied_worker_pair",
            "select_embodied_worker_trio",
        }
    ]
    namespace = {
        "EnvWorker": _GenericEnvWorker,
        "MultiStepRolloutWorker": _GenericRolloutWorker,
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return (
        namespace["select_embodied_worker_pair"],
        namespace["select_embodied_worker_trio"],
        source,
    )


SELECT_WORKERS, SELECT_TRIO, TRAINER_SOURCE = _load_selector()


def _successor_cfg():
    return {
        "runner": {
            "worker_pair": "p8_successor",
            "p8_successor_env_controller_factory": "fake:env",
            "p8_successor_rollout_controller_factory": "fake:rollout",
            "p8_successor_actor_replay_adapter": "fake:actor",
        },
        "env": {
            "train": {
                "env_type": "streamvln",
                "executor_mode": "p8_successor",
                "total_num_envs": 1,
                "rollout_epoch": 1,
                "max_episode_steps": 4,
            }
        },
        "rollout": {"pipeline_stage_num": 1},
        "actor": {
            "model": {
                "model_type": "streamvln_p8_successor",
                "successor_topology_max_nodes": 5,
            }
        },
        "algorithm": {
            "adv_type": "p8_successor_duration_gae",
            "loss_type": "p8_successor_joint_ppo",
        },
    }


class SuccessorWorkerSelectorTest(unittest.TestCase):
    def test_baseline_returns_exact_generic_classes_without_adapter_import(self):
        adapter_name = WORKERS.DEFAULT_ADAPTER_MODULE
        sys.modules.pop(adapter_name, None)
        rollout_cls, env_cls = SELECT_WORKERS(
            {"runner": {}, "env": {"train": {"env_type": "streamvln"}}}
        )
        self.assertIs(rollout_cls, _GenericRolloutWorker)
        self.assertIs(env_cls, _GenericEnvWorker)
        trio = SELECT_TRIO(
            {"runner": {}, "env": {"train": {"env_type": "streamvln"}}},
            _GenericActorWorker,
        )
        self.assertEqual(trio, (_GenericRolloutWorker, _GenericEnvWorker, _GenericActorWorker))
        self.assertNotIn(adapter_name, sys.modules)
        readme = (MODULE_DIR / "README.md").read_text(encoding="utf-8")
        self.assertIn(
            "rlinf/workers/streamvln_p8_successor_workers.py", readme
        )
        self.assertIn(
            "from rlinf.workers.streamvln_p8_successor_workers import", TRAINER_SOURCE
        )

    def test_pair_selection_half_selection_and_pending_prelaunch_gate(self):
        adapter = types.ModuleType(WORKERS.DEFAULT_ADAPTER_MODULE)
        adapter._require_pinned_release_manifest_sha256 = lambda: "a" * 64
        sys.modules[WORKERS.DEFAULT_ADAPTER_MODULE] = adapter
        factories = types.ModuleType("fake")
        factories.env = lambda *args, **kwargs: None
        factories.rollout = lambda *args, **kwargs: None
        factories.actor = lambda *args, **kwargs: None
        factories.env.successor_replay_schema = WORKERS.SUCCESSOR_REPLAY_SCHEMA
        factories.rollout.successor_replay_schema = WORKERS.SUCCESSOR_REPLAY_SCHEMA
        factories.actor.successor_replay_schema = WORKERS.SUCCESSOR_REPLAY_SCHEMA
        sys.modules["fake"] = factories
        successor_actor = types.ModuleType("rlinf.workers.streamvln_p8_successor_actor")

        class _SuccessorActorWorker:
            successor_replay_schema = WORKERS.SUCCESSOR_REPLAY_SCHEMA

        successor_actor.P8SuccessorActorWorker = _SuccessorActorWorker
        successor_actor.SUCCESSOR_REPLAY_SCHEMA = WORKERS.SUCCESSOR_REPLAY_SCHEMA
        successor_actor.SUCCESSOR_ADVANTAGE_NAME = "p8_successor_duration_gae"
        successor_actor.SUCCESSOR_POLICY_LOSS_NAME = "p8_successor_joint_ppo"
        sys.modules["rlinf.workers.streamvln_p8_successor_actor"] = successor_actor
        rollout_cls, env_cls = SELECT_WORKERS(_successor_cfg())
        self.assertIs(rollout_cls, WORKERS.P8SuccessorRolloutWorker)
        self.assertIs(env_cls, WORKERS.P8SuccessorEnvWorker)
        self.assertEqual(
            SELECT_TRIO(_successor_cfg(), _GenericActorWorker),
            (WORKERS.P8SuccessorRolloutWorker, WORKERS.P8SuccessorEnvWorker, _SuccessorActorWorker),
        )
        wrong_algorithm = _successor_cfg()
        wrong_algorithm["algorithm"]["adv_type"] = "gae"
        with self.assertRaisesRegex(ValueError, "p8_successor_duration_gae"):
            SELECT_TRIO(wrong_algorithm, _GenericActorWorker)

        readiness_calls = {"importer": 0, "factory": 0}

        def side_effect_importer(_name):
            readiness_calls["importer"] += 1
            return adapter

        original_loader = WORKERS._load_symbol

        def side_effect_loader(spec):
            readiness_calls["factory"] += 1
            return original_loader(spec)

        wrong_count = _successor_cfg()
        wrong_count["env"]["train"]["total_num_envs"] = 2
        WORKERS._load_symbol = side_effect_loader
        try:
            with self.assertRaisesRegex(RuntimeError, "total_num_envs=1"):
                WORKERS.require_successor_production_ready(
                    wrong_count, importer=side_effect_importer
                )
        finally:
            WORKERS._load_symbol = original_loader
        self.assertEqual(readiness_calls, {"importer": 0, "factory": 0})

        factories.rollout.successor_replay_schema = (
            "streamvln-p8-successor-decision-v1"
        )
        with self.assertRaisesRegex(RuntimeError, "factory schema mismatch"):
            WORKERS.require_successor_production_ready(
                _successor_cfg(), importer=side_effect_importer
            )
        factories.rollout.successor_replay_schema = WORKERS.SUCCESSOR_REPLAY_SCHEMA
        readiness_calls.update(importer=0, factory=0)

        wrong_epochs = _successor_cfg()
        wrong_epochs["env"]["train"]["rollout_epoch"] = 2
        WORKERS._load_symbol = side_effect_loader
        try:
            with self.assertRaisesRegex(RuntimeError, "rollout_epoch=1"):
                WORKERS.require_successor_production_ready(
                    wrong_epochs, importer=side_effect_importer
                )
        finally:
            WORKERS._load_symbol = original_loader
        self.assertEqual(readiness_calls, {"importer": 0, "factory": 0})

        wrong_stages = _successor_cfg()
        wrong_stages["rollout"]["pipeline_stage_num"] = 2
        WORKERS._load_symbol = side_effect_loader
        try:
            with self.assertRaisesRegex(RuntimeError, "pipeline_stage_num=1"):
                WORKERS.require_successor_production_ready(
                    wrong_stages, importer=side_effect_importer
                )
        finally:
            WORKERS._load_symbol = original_loader
        self.assertEqual(readiness_calls, {"importer": 0, "factory": 0})

        wrong_model_env = _successor_cfg()
        wrong_model_env["runner"]["p8_successor_model_env_id"] = 1
        WORKERS._load_symbol = side_effect_loader
        try:
            with self.assertRaisesRegex(RuntimeError, "model_env_id=0"):
                WORKERS.require_successor_production_ready(
                    wrong_model_env, importer=side_effect_importer
                )
        finally:
            WORKERS._load_symbol = original_loader
        self.assertEqual(readiness_calls, {"importer": 0, "factory": 0})

        with self.assertRaisesRegex(ValueError, "requires runner.worker_pair"):
            SELECT_WORKERS(
                {
                    "runner": {},
                    "env": {
                        "train": {
                            "env_type": "streamvln",
                            "executor_mode": "p8_successor",
                        }
                    },
                }
            )
        wrong = _successor_cfg()
        wrong["env"]["train"]["env_type"] = "pointnav"
        with self.assertRaisesRegex(ValueError, "env_type=streamvln"):
            SELECT_WORKERS(wrong)

        cluster_factory_calls = 0

        def pending_anchor():
            raise RuntimeError("official release manifest SHA-256 is PENDING")

        adapter._require_pinned_release_manifest_sha256 = pending_anchor
        with self.assertRaisesRegex(RuntimeError, "PENDING"):
            SELECT_WORKERS(_successor_cfg())
            cluster_factory_calls += 1
        self.assertEqual(cluster_factory_calls, 0)
        self.assertLess(
            TRAINER_SOURCE.index("select_embodied_worker_trio("),
            TRAINER_SOURCE.index("cluster = Cluster("),
        )


class _FakeOfficialDriver:
    def __init__(self, events):
        self.events = events
        self.shadow_calls = 0
        self.ack_calls = 0

    def prepare_shadow_action(self, frame):
        self.shadow_calls += 1
        return 1

    def acknowledge_physical_step(self, *, terminated, truncated):
        self.ack_calls += 1
        self.events.append(f"ack-{self.ack_calls}-terminal-{terminated or truncated}")


class _FakeEnv:
    def __init__(self, events):
        self.events = events
        self.execute_counts = []
        self.actions = []
        self.step = 0

    def reset(self):
        self.step = 0
        return {"rgb": "reset"}, {}

    def observe_official_frame(self):
        return ("official-frame", self.step)

    def chunk_step(self, payload, execute_count=None):
        self.assert_shape = (
            len(payload),
            len(payload[0]),
            len(payload[0][0]),
        )
        action = int(payload[0][0][0])
        self.execute_counts.append(execute_count)
        self.actions.append(action)
        self.step += 1
        terminal = self.step == 4
        self.events.append(f"execute-{self.step}-terminal-{terminal}")
        primitive = primitive_execution(
            primitive_index=0,
            action=action,
            previous_path_length_m=0.25 * (self.step - 1),
            current_path_length_m=0.25 * self.step,
            collision=False,
            environment_reward=1.0,
            terminated=terminal,
            truncated=False,
        )
        rewards = [[1.0, 0.0, 0.0, 0.0]]
        terminations = [[terminal, False, False, False]]
        truncations = [[False, False, False, False]]
        info = {"executor_feedback": chunk_feedback([action], [primitive]).to_dict()}
        return [{"rgb": self.step}], rewards, terminations, truncations, [info]


class _ScriptedEnv:
    def __init__(self, script):
        self.script = list(script)
        self.reset_calls = 0
        self.actions = []
        self.step = 0
        self.path_length = 0.0

    def reset(self):
        self.reset_calls += 1
        self.step = 0
        self.path_length = 0.0
        self.actions.clear()
        return {
            "rgb": "present-only",
            "future_path": "must-not-enter-topology",
            "demo_waypoints": "must-not-enter-topology",
            "goal_position": "must-not-enter-topology",
        }, {}

    def observe_official_frame(self):
        return ("official-frame", self.reset_calls, self.step)

    def chunk_step(self, payload, execute_count=None):
        action = int(payload[0][0][0])
        row = self.script[self.step] if self.step < len(self.script) else {}
        realized = float(row.get("realized", 0.25 if action == 1 else 0.0))
        collision = bool(row.get("collision", False))
        terminated = bool(row.get("terminated", False))
        truncated = bool(row.get("truncated", False))
        previous = self.path_length
        self.path_length += realized
        primitive = primitive_execution(
            primitive_index=0,
            action=action,
            previous_path_length_m=previous,
            current_path_length_m=self.path_length,
            collision=collision,
            environment_reward=float(row.get("reward", 0.0)),
            terminated=terminated,
            truncated=truncated,
        )
        self.actions.append(action)
        self.step += 1
        info = {"executor_feedback": chunk_feedback([action], [primitive]).to_dict()}
        if truncated:
            info["final_observation"] = {"rgb": "final"}
        return (
            [{"rgb": f"present-{self.step}"}],
            [[primitive.environment_reward, 0.0, 0.0, 0.0]],
            [[terminated, False, False, False]],
            [[truncated, False, False, False]],
            [info],
        )


def _step_request(frame, *, waypoint=(1.0, 0.0), official_action=1):
    sample = None
    if frame.needs_decision:
        sample = SuccessorDecisionSample(
            decision=RouteDecision(
                decision_index=frame.decision_index,
                route=RouteName.WAYPOINT,
                waypoint_local_m=waypoint,
                episode_token=frame.episode_token,
                physical_step=frame.physical_step,
            ),
            proposed_actions=(1,),
            logprob=-0.25,
            value=0.5,
            official_shadow_action=official_action,
        )
    return WORKERS.SuccessorStepRequest(
        episode_token=frame.episode_token,
        physical_step=frame.physical_step,
        decision_index=frame.decision_index,
        needs_decision=frame.needs_decision,
        decision_mask=frame.decision_mask,
        official_shadow_action=official_action,
        decision_sample=sample,
    )


class _AttrDict(dict):
    __getattr__ = dict.__getitem__


class _FakeChannel:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.sent = []


class _FakeActorChannel:
    def __init__(self):
        self.items = []

    def put(self, value, async_op=False):
        self.items.append(value)


class SuccessorWorkerRoundTripTest(unittest.TestCase):
    def test_train_expert_target_is_typed_and_queried_only_at_boundaries(self):
        class ExpertEnv(_ScriptedEnv):
            def __init__(self):
                super().__init__([{} for _ in range(4)])
                self.expert_calls = 0

            def expert_waypoint(self, lookahead_m):
                self.expert_calls += 1
                return {
                    "schema_version": 1,
                    "source": "train-shortest-path",
                    "lookahead_m": lookahead_m,
                    "waypoint_local_m": [0.5, 0.0],
                    "bearing_rad": 0.0,
                    "target_distance_m": 0.5,
                    "distance_to_goal_m": 4.0,
                    "stop": False,
                }

        env = ExpertEnv()
        controller = WORKERS.SuccessorEnvController(
            env,
            token_factory=lambda: "episode-expert",
            max_decisions=4,
            topology_max_nodes=5,
            bc_expert_enabled=True,
            expert_lookahead_m=1.0,
        )
        frame = controller.reset_episode()
        self.assertEqual(env.expert_calls, 1)
        self.assertIsInstance(
            frame.expert_target, WORKERS.TrainShortestPathExpertTarget
        )
        self.assertEqual(frame.expert_target.decision_index, 0)
        next_frame, macro = controller.accept(_step_request(frame))
        self.assertIsNone(macro)
        self.assertIsNone(next_frame.expert_target)
        self.assertEqual(env.expert_calls, 1)

        passive = WORKERS.SuccessorEnvController(
            ExpertEnv(),
            token_factory=lambda: "episode-passive",
            max_decisions=4,
            topology_max_nodes=5,
        )
        passive_frame = passive.reset_episode()
        self.assertIsNone(passive_frame.expert_target)
        self.assertEqual(passive.env.expert_calls, 0)

    def test_four_primitives_close_one_causal_topology_update(self):
        env = _ScriptedEnv([{} for _ in range(4)])
        controller = WORKERS.SuccessorEnvController(
            env,
            token_factory=lambda: "episode-four",
            max_decisions=8,
            topology_max_nodes=9,
        )
        frame = controller.reset_episode()
        self.assertEqual(controller.topology.decision_count, 0)
        self.assertEqual(len(controller.topology.nodes), 1)
        macro = None
        for index in range(4):
            frame, macro = controller.accept(_step_request(frame))
            if index < 3:
                self.assertIsNone(macro)
                self.assertEqual(controller.topology.decision_count, 0)
        self.assertIsNotNone(macro)
        self.assertEqual(macro.duration, 4)
        self.assertEqual(macro.executed_actions, (1, 1, 1, 1))
        self.assertEqual(macro.realized_path_m, (0.25, 0.25, 0.25, 0.25))
        self.assertEqual(controller.topology.decision_count, 1)
        self.assertEqual(controller.topology.primitive_count, 4)
        self.assertEqual(len(controller.topology.nodes), 2)
        self.assertEqual(frame.topology_snapshot.decision_index, 1)
        self.assertEqual(frame.topology_snapshot.physical_step, 4)

    def test_causal_snapshot_has_no_future_demo_or_goal_input(self):
        env = _ScriptedEnv([])
        controller = WORKERS.SuccessorEnvController(
            env,
            token_factory=lambda: "episode-causal",
            max_decisions=4,
            topology_max_nodes=5,
        )
        frame = controller.reset_episode()
        self.assertEqual(len(controller.topology.nodes), 1)
        self.assertEqual(controller.topology.edges, set())
        self.assertEqual(frame.topology_snapshot.decision_index, 0)
        self.assertEqual(
            set(inspect.signature(WORKERS.snapshot_online_topology).parameters),
            {"topology", "episode_token", "physical_step", "decision_index", "max_nodes"},
        )
        snapshot_fields = set(vars(frame.topology_snapshot))
        for forbidden in ("future", "demo", "goal", "geodesic", "reference"):
            self.assertFalse(any(forbidden in name for name in snapshot_fields))

        topology = WORKERS.OnlineTopology(max_decisions=4)
        topology.update(
            (1, 1, 1, 1),
            (False, False, False, False),
            realized_path_m=(0.25, 0.25, 0.25, 0.25),
            blocked_forwards=(False, False, False, False),
        )
        topology.update(
            (2, 1, 1, 1),
            (False, False, False, False),
            realized_path_m=(0.0, 0.25, 0.25, 0.25),
            blocked_forwards=(False, False, False, False),
        )
        expected = causal_topology_payload(topology, max_nodes=5)
        equivalent = copy.deepcopy(topology)
        angle = 0.61
        translate_x, translate_z = 8.25, -3.75

        def transform(x, z):
            return (
                math.cos(angle) * x + math.sin(angle) * z + translate_x,
                -math.sin(angle) * x + math.cos(angle) * z + translate_z,
            )

        equivalent.x, equivalent.z = transform(topology.x, topology.z)
        equivalent.heading = math.atan2(
            math.sin(topology.heading + angle),
            math.cos(topology.heading + angle),
        )
        for source, target in zip(topology.nodes, equivalent.nodes):
            target.x, target.z = transform(source.x, source.z)
        actual = causal_topology_payload(equivalent, max_nodes=5)
        self.assertEqual(actual.current_index, expected.current_index)
        self.assertEqual(actual.parent_index, expected.parent_index)
        self.assertEqual(actual.adjacency, expected.adjacency)

        def flatten(values):
            for value in values:
                if isinstance(value, tuple):
                    yield from flatten(value)
                else:
                    yield float(value)

        for expected_value, actual_value in zip(
            flatten(
                (
                    expected.node_features,
                    expected.edge_features,
                    expected.global_features,
                )
            ),
            flatten(
                (
                    actual.node_features,
                    actual.edge_features,
                    actual.global_features,
                )
            ),
        ):
            self.assertAlmostEqual(actual_value, expected_value, places=7)

    def test_second_step_collision_truncates_and_next_decision_recovers_parent(self):
        env = _ScriptedEnv(
            [{}, {}, {}, {}, {}, {"collision": True, "realized": 0.0}, {}]
        )
        controller = WORKERS.SuccessorEnvController(
            env,
            token_factory=lambda: "episode-recovery",
            max_decisions=8,
            topology_max_nodes=9,
            planner=SuccessorPlanner(blocked_threshold=1),
        )
        frame = controller.reset_episode()
        for _ in range(4):
            frame, first_macro = controller.accept(_step_request(frame))
        self.assertEqual(first_macro.duration, 4)
        self.assertEqual(controller.topology.current_node, 1)
        self.assertEqual(controller.topology.nodes[1].parent, 0)

        frame, collision_macro = controller.accept(_step_request(frame))
        self.assertIsNone(collision_macro)
        frame, collision_macro = controller.accept(_step_request(frame))
        self.assertEqual(collision_macro.duration, 2)
        self.assertEqual(collision_macro.executed_mask, (True, True, False, False))
        self.assertEqual(
            collision_macro.primitive_collisions,
            (False, True, False, False),
        )
        self.assertEqual(
            collision_macro.primitive_blocked,
            (False, True, False, False),
        )
        self.assertEqual(controller.topology.consecutive_blocked_forwards, 1)

        recovery_request = _step_request(frame, waypoint=(0.75, 0.25))
        sampled_route = recovery_request.decision_sample.decision.route
        frame, recovery_macro = controller.accept(recovery_request)
        self.assertIsNone(recovery_macro)
        plan = controller.active_plan
        self.assertIsNotNone(plan)
        self.assertIs(sampled_route, RouteName.WAYPOINT)
        self.assertIs(plan.requested_route, RouteName.WAYPOINT)
        self.assertIs(plan.executed_route, RouteName.RECOVER_PARENT)
        self.assertTrue(plan.recovery_override)
        self.assertEqual(plan.recovery_parent_index, 0)
        self.assertNotIn(0, plan.proposed_actions)
        self.assertEqual(env.actions[-1], plan.proposed_actions[0])

        root_env = _ScriptedEnv([{"collision": True, "realized": 0.0}])
        root_controller = WORKERS.SuccessorEnvController(
            root_env,
            token_factory=lambda: "episode-root",
            max_decisions=4,
            topology_max_nodes=5,
            planner=SuccessorPlanner(blocked_threshold=1),
        )
        root_frame = root_controller.reset_episode()
        root_frame, root_macro = root_controller.accept(_step_request(root_frame))
        self.assertEqual(root_macro.duration, 1)
        with self.assertRaises(ParentRecoveryUnavailable):
            root_controller.accept(_step_request(root_frame))

    def test_reset_rotates_token_clears_state_and_rejects_stale(self):
        tokens = iter(("episode-old", "episode-new", "episode-new"))
        env = _ScriptedEnv([{}])
        controller = WORKERS.SuccessorEnvController(
            env,
            token_factory=lambda: next(tokens),
            max_decisions=4,
            topology_max_nodes=5,
        )
        old_frame = controller.reset_episode()
        old_continuation, macro = controller.accept(_step_request(old_frame))
        self.assertIsNone(macro)
        self.assertIsNotNone(controller.active_plan)
        with self.assertRaisesRegex(ValueError, "decision_index is stale"):
            replace(
                old_continuation,
                previous_result=replace(
                    old_continuation.previous_result,
                    decision_index=old_continuation.decision_index + 1,
                ),
            )

        new_frame = controller.reset_episode()
        self.assertEqual(new_frame.episode_token, "episode-new")
        self.assertEqual(new_frame.physical_step, 0)
        self.assertEqual(new_frame.decision_index, 0)
        self.assertEqual(controller.topology.decision_count, 0)
        self.assertEqual(len(controller.topology.nodes), 1)
        self.assertIsNone(controller.active_plan)
        with self.assertRaisesRegex(ValueError, "stale successor request identity"):
            controller.accept(_step_request(old_continuation))

        with self.assertRaisesRegex(RuntimeError, "rotate the episode token"):
            controller.reset_episode()
        self.assertEqual(env.reset_calls, 2)
        with self.assertRaisesRegex(RuntimeError, "inactive"):
            _ = controller.topology

    def test_four_roundtrips_sample_once_execute_once_each_and_emit_one_macro(self):
        events = []
        env = _FakeEnv(events)
        driver = _FakeOfficialDriver(events)
        sample_calls = 0

        def sample(frame, shadow_action):
            nonlocal sample_calls
            sample_calls += 1
            return SuccessorDecisionSample(
                decision=RouteDecision(
                    decision_index=frame.decision_index,
                    route=RouteName.WAYPOINT,
                    waypoint_local_m=(1.0, 0.0),
                    episode_token=frame.episode_token,
                    physical_step=frame.physical_step,
                ),
                proposed_actions=(1, 1, 1, 1),
                logprob=-0.25,
                value=0.5,
                official_shadow_action=shadow_action,
            )

        env_controller = WORKERS.SuccessorEnvController(
            env,
            token_factory=lambda: "episode-one",
            max_decisions=4,
            topology_max_nodes=5,
        )
        rollout_controller = WORKERS.SuccessorRolloutController(
            official_driver_factory=lambda frame: driver,
            decision_sampler=sample,
            bootstrap_value_evaluator=lambda frame: 0.75,
        )
        cfg = _AttrDict(
            runner=_AttrDict(
                p8_successor_env_controller_factory="runtime_factories:env",
                p8_successor_rollout_controller_factory="runtime_factories:rollout",
                p8_successor_actor_replay_adapter="runtime_factories:actor",
            ),
            env=_AttrDict(
                group_name="env",
                train=_AttrDict(max_steps_per_rollout_epoch=4),
            ),
            rollout=_AttrDict(group_name="rollout"),
        )
        rollout_worker = object.__new__(WORKERS.P8SuccessorRolloutWorker)
        rollout_worker.cfg = cfg
        rollout_worker.num_pipeline_stages = 1
        rollout_worker.train_batch_size = 1
        env_worker = object.__new__(WORKERS.P8SuccessorEnvWorker)
        env_worker.cfg = cfg
        env_worker.stage_num = 1
        env_worker.rollout_epoch = 1
        env_worker.train_batch_size = 1
        env_worker.env_list = [env]
        replayed_macros = []

        class ReplayAdapter:
            successor_replay_schema = WORKERS.SUCCESSOR_REPLAY_SCHEMA

            def convert(
                self, *, macro, next_value, stage_id, env_slot, final_observation=None
            ):
                replayed_macros.append((stage_id, env_slot, macro, next_value))
                return macro

            def build_batch(self, rows):
                return tuple(rows)

        factories = types.ModuleType("runtime_factories")
        factories.rollout = lambda **kwargs: rollout_controller
        factories.env = lambda **kwargs: env_controller
        factories.actor = lambda **kwargs: ReplayAdapter()
        factories.rollout.successor_replay_schema = WORKERS.SUCCESSOR_REPLAY_SCHEMA
        factories.env.successor_replay_schema = WORKERS.SUCCESSOR_REPLAY_SCHEMA
        factories.actor.successor_replay_schema = WORKERS.SUCCESSOR_REPLAY_SCHEMA
        sys.modules["runtime_factories"] = factories
        rollout_worker.init_worker()
        env_worker.init_worker()
        env_to_rollout = _FakeChannel()
        rollout_to_env = _FakeChannel()
        actor_channel = _FakeActorChannel()

        async def run_pair():
            await asyncio.gather(
                rollout_worker.generate_one_epoch(env_to_rollout, rollout_to_env),
                env_worker._run_interact_once(
                    rollout_to_env,
                    env_to_rollout,
                    None,
                    actor_channel,
                    cooperative_yield=False,
                ),
            )

        asyncio.run(run_pair())

        self.assertEqual(env.assert_shape, (1, 4, 1))
        self.assertEqual(env.execute_counts, [1, 1, 1, 1])
        self.assertEqual(driver.shadow_calls, 4)
        self.assertEqual(driver.ack_calls, 4)
        self.assertLess(
            events.index("execute-4-terminal-True"),
            events.index("ack-4-terminal-True"),
        )
        self.assertEqual(sample_calls, 1)
        self.assertEqual(len(replayed_macros), 1)
        self.assertEqual(len(actor_channel.items), 1)
        macro = replayed_macros[0][2]
        self.assertEqual(replayed_macros[0][3], 0.0)
        self.assertEqual(macro.duration, 4)
        self.assertEqual(macro.executed_mask, (True, True, True, True))
        for batch in env_to_rollout.sent + rollout_to_env.sent:
            self.assertEqual(len(batch.items), 1)

        truncated_result = WORKERS.SuccessorStepResult(
            episode_token="truncated-episode",
            physical_step=0,
            decision_index=0,
            needs_decision=True,
            decision_mask=True,
            executed_action=1,
            reward=1.0,
            truncated=True,
        )
        with self.assertRaisesRegex(ValueError, "final_observation"):
            WORKERS.SuccessorFrameRequest(
                env_slot=0,
                episode_token="truncated-episode",
                physical_step=1,
                decision_index=1,
                needs_decision=False,
                decision_mask=False,
                observation=None,
                official_frame=None,
                topology_snapshot=_topology_snapshot(
                    "truncated-episode", 1, 1
                ),
                previous_result=truncated_result,
                terminal_ack=True,
            )
        terminal_events = []
        terminal_driver = _FakeOfficialDriver(terminal_events)

        def terminal_sample(frame, shadow_action):
            return SuccessorDecisionSample(
                decision=RouteDecision(
                    decision_index=0,
                    route=RouteName.WAYPOINT,
                    waypoint_local_m=(1.0, 0.0),
                    episode_token="truncated-episode",
                    physical_step=0,
                ),
                proposed_actions=(1,),
                logprob=-0.1,
                value=0.2,
                official_shadow_action=shadow_action,
            )

        def terminal_value(frame):
            self.assertEqual(frame.observation, {"final": "observation"})
            terminal_events.append("evaluate-final-observation")
            return 0.9

        terminal_controller = WORKERS.SuccessorRolloutController(
            official_driver_factory=lambda frame: terminal_driver,
            decision_sampler=terminal_sample,
            bootstrap_value_evaluator=terminal_value,
        )
        terminal_controller.process(
            WORKERS.SuccessorFrameRequest(
                env_slot=0,
                episode_token="truncated-episode",
                physical_step=0,
                decision_index=0,
                needs_decision=True,
                decision_mask=True,
                observation={"start": True},
                official_frame=("official", 0),
                topology_snapshot=_topology_snapshot(
                    "truncated-episode", 0, 0
                ),
            )
        )
        terminal_response = terminal_controller.process(
            WORKERS.SuccessorFrameRequest(
                env_slot=0,
                episode_token="truncated-episode",
                physical_step=1,
                decision_index=1,
                needs_decision=False,
                decision_mask=False,
                observation={"final": "observation"},
                official_frame=None,
                topology_snapshot=_topology_snapshot(
                    "truncated-episode", 1, 1
                ),
                previous_result=truncated_result,
                terminal_ack=True,
            )
        )
        self.assertEqual(terminal_response.bootstrap_value, 0.9)
        self.assertEqual(
            terminal_events,
            ["evaluate-final-observation", "ack-1-terminal-True"],
        )


if __name__ == "__main__":
    unittest.main()
