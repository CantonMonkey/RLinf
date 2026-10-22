from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


MODULE_DIR = Path(__file__).resolve().parents[1] / "integrations" / "rlinf_v03"
MODULE_PATH = MODULE_DIR / "run_streamvln_p8_successor_ceiling.py"
SPEC = importlib.util.spec_from_file_location("run_streamvln_p8_successor_ceiling", MODULE_PATH)
CEILING = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CEILING
SPEC.loader.exec_module(CEILING)
PRODUCTION_PROTOCOL_ANCHOR = CEILING.CEILING_PROTOCOL_CONTRACT_SHA256


class RouteName(str, Enum):
    KEEP_ORIGINAL = "keep_original"
    WAYPOINT = "waypoint"
    STOP = "stop"


@dataclass
class RouteDecision:
    decision_index: int
    route: RouteName
    waypoint_local_m: tuple[float, float] = (0.0, 0.0)
    official_action: int | None = None
    episode_token: str | None = None
    physical_step: int | None = None


@dataclass
class Primitive:
    action: int
    collision: bool
    blocked: bool
    terminated: bool
    truncated: bool
    realized_path_m: float

    def to_dict(self):
        return {
            "primitive_index": 0,
            "action": self.action,
            "collision": self.collision,
            "blocked": self.blocked,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "realized_path_m": self.realized_path_m,
        }


@dataclass
class Feedback:
    primitives: tuple[Primitive, ...]


class Frame:
    def __init__(self, env):
        self.env = env

    def trace_metadata(self):
        return {
            "env_id": 0,
            "episode_serial": 0,
            "episode_generation": "0" * 32,
            "physical_step": self.env.physical_step,
            "episode_id": self.env.identity["episode_id"],
            "scene_id": self.env.identity["scene_id"],
            "instruction_sha256": self.env.identity["instruction_sha256"],
            "frame_ref": f"sha256:{self.env.physical_step:064x}",
            "rgb_sha256": "1" * 64,
            "depth_sha256": "2" * 64,
            "pose_sha256": "3" * 64,
            "intrinsics_sha256": "4" * 64,
            "contract_sha256": "5" * 64,
        }


class Plan:
    def __init__(self, decision, actions):
        self.requested_route = decision.route
        self.executed_route = decision.route
        self.proposed_actions = tuple(actions)
        self.decision_index = decision.decision_index
        self.waypoint_local_m = tuple(decision.waypoint_local_m)

    def to_dict(self):
        return {
            "decision_index": self.decision_index,
            "requested_route": self.requested_route.value,
            "executed_route": self.executed_route.value,
            "waypoint_local_m": list(self.waypoint_local_m),
            "proposed_actions": list(self.proposed_actions),
            "source": "test",
            "recovery_override": False,
            "recovery_parent_index": None,
            "target_distance_m": None,
            "target_bearing_rad": None,
        }


class Planner:
    config = types.SimpleNamespace(forward_step_m=0.25, turn_degrees=15.0)

    def plan(self, decision, _topology):
        if decision.route is RouteName.STOP:
            return Plan(decision, (0,))
        if decision.route is RouteName.WAYPOINT:
            return Plan(decision, (3, 1))
        return Plan(decision, (1,))


class Topology:
    def __init__(self):
        self.decision_count = 0
        self.updates = []

    def update(self, actions, collisions, *, realized_path_m, blocked_forwards):
        self.updates.append(
            (tuple(actions), tuple(collisions), tuple(realized_path_m), tuple(blocked_forwards))
        )
        self.decision_count += 1


class ExpertEnv:
    def __init__(self, episode_index):
        self.identity = {
            "episode_index": episode_index,
            "episode_id": f"episode-{episode_index}",
            "scene_id": f"scene-{episode_index}",
            "instruction_sha256": "b" * 64,
            "instruction_text": "go to the target",
        }
        self.successor_planner = Planner()
        self.physical_step = 0
        self.closed = False
        self._terminal_metrics = None

    def reset(self):
        return {"rgb": 0}, {"episode_id": self.identity["episode_id"]}

    def observe_official_frame(self):
        return Frame(self)

    def expert_waypoint(self, lookahead_m):
        assert lookahead_m == 1.0
        if self.physical_step < 2:
            return {
                "schema_version": 1,
                "source": "train-shortest-path",
                "lookahead_m": 1.0,
                "waypoint_local_m": [0.5, 0.0],
                "bearing_rad": 0.0,
                "target_distance_m": 0.5,
                "distance_to_goal_m": 4.0,
                "stop": False,
            }
        return {
            "schema_version": 1,
            "source": "train-shortest-path",
            "lookahead_m": 1.0,
            "waypoint_local_m": [0.0, 0.0],
            "bearing_rad": 0.0,
            "target_distance_m": 0.0,
            "distance_to_goal_m": 1.0,
            "stop": True,
        }

    def chunk_step(self, payload, execute_count):
        assert execute_count == 1
        action = int(payload[0][0][0])
        terminated = action == 0
        primitive = Primitive(
            action=action,
            collision=False,
            blocked=False,
            terminated=terminated,
            truncated=False,
            realized_path_m=0.25 if action == 1 else 0.0,
        )
        self.physical_step += 1
        info = {"executor_feedback": Feedback((primitive,))}
        if terminated:
            self._terminal_metrics = {
                "success": 1.0,
                "spl": 0.75,
                "path_length": 0.25,
                "collision_count": 0.0,
                "distance_to_goal": 1.0,
            }
            info["episode"] = dict(self._terminal_metrics)
        return [{"rgb": self.physical_step}], [0.0], [terminated], [False], [info]

    def close(self):
        self.closed = True


class KeepEnv:
    def __init__(self, episode_index):
        self.identity = {
            "episode_index": episode_index,
            "episode_id": f"episode-{episode_index}",
            "scene_id": f"scene-{episode_index}",
            "instruction_sha256": "c" * 64,
            "instruction_text": "keep the original policy",
        }
        self._terminal_metrics = {
            "success": 1.0,
            "spl": 1.0,
            "path_length": 0.25,
            "collision_count": 0.0,
            "distance_to_goal": 0.5,
        }
        self.closed = False

    def reset(self):
        return {"rgb": 0}, {}

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, **_kwargs):
        self.physical_step = 0


class FakeAdapter:
    def __init__(self, calls):
        self.calls = calls

    def make_driver(self, *, session, episode_serial, episode_identity):
        self.calls.append((episode_serial, dict(episode_identity)))
        return types.SimpleNamespace(session=session)


class BridgeResult:
    def __init__(self, bridge, action, terminated):
        self.bridge = bridge
        self.action = action
        self.terminated = terminated
        self.truncated = False

    def to_trace(self):
        step = self.bridge.last_step
        identity = self.bridge.env.identity
        shadow_action = (
            (self.action + 1) % 4 if getattr(self.bridge, "mismatch", False) else self.action
        )
        return {
            "schema_name": "streamvln_p8_successor_bridge_decision",
            "schema_version": 1,
            "episode_token": self.bridge.episode_token,
            "decision_physical_step": step,
            "plan": {
                "requested_route": RouteName.KEEP_ORIGINAL.value,
                "executed_route": RouteName.KEEP_ORIGINAL.value,
                "proposed_actions": [self.action],
            },
            "official_shadow_actions": [shadow_action],
            "official_frame_metadata": [
                {
                    "physical_step": step,
                    "episode_id": identity["episode_id"],
                    "scene_id": identity["scene_id"],
                    "instruction_sha256": identity["instruction_sha256"],
                }
            ],
            "official_step_events": [{}],
            "executed_actions": [self.action],
            "primitive_feedback": [{}],
            "end_reason": "stop" if self.terminated else "plan_complete",
            "replan_required": not self.terminated,
            "official_stop_override": False,
            "terminated": self.terminated,
            "truncated": False,
        }


class FakeBridge:
    exact_flags = []
    mismatch = False

    def __init__(self, *, env, topology_factory, official_driver_factory, exact_control):
        self.env = env
        self.topology_factory = topology_factory
        self.official_driver_factory = official_driver_factory
        self.exact_control = exact_control
        self.episode_token = "keep-token"
        self.official_driver = None
        self.last_step = 0
        self.calls = 0
        self.exact_flags.append(exact_control)

    def reset_episode(self):
        self.env.reset()
        self.topology_factory()
        self.official_driver = self.official_driver_factory(0, self.env.identity, 0)
        return {"rgb": 0}, {}

    def execute_decision(self, decision):
        assert decision.route is RouteName.KEEP_ORIGINAL
        assert decision.episode_token == self.episode_token
        assert decision.physical_step == self.official_driver.session.physical_step
        self.last_step = self.official_driver.session.physical_step
        action = 1 if self.calls == 0 else 0
        self.calls += 1
        self.official_driver.session.physical_step += 1
        return BridgeResult(self, action, terminated=action == 0)


class MismatchBridge(FakeBridge):
    mismatch = True


class CeilingTest(unittest.TestCase):
    def make_args(self, root: Path, *, stage="preflight", split="train", output=None):
        source = root / "source"
        checkpoint = root / "checkpoint"
        tokenizer = root / "tokenizer"
        vision = root / "vision"
        rlinf = root / "rlinf-root"
        asset = root / "asset"
        env_lib = root / "env-lib"
        habitat_sim_root = root / "habitat-sim"
        habitat_sim_python = habitat_sim_root / "src_python"
        for path in (
            source,
            checkpoint,
            tokenizer,
            vision,
            rlinf / "rlinf",
            env_lib,
            habitat_sim_python,
        ):
            path.mkdir(parents=True, exist_ok=True)
        dataset = asset / "datasets" / "r2r" / "train" / "train.json.gz"
        dataset.parent.mkdir(parents=True, exist_ok=True)
        dataset.write_bytes(b"train")
        scene_records = []
        for episode_index in (7, 8):
            scene = f"scene-{episode_index}"
            scene_dir = asset / "scene_datasets" / "mp3d" / scene
            scene_dir.mkdir(parents=True, exist_ok=True)
            glb = scene_dir / f"{scene}.glb"
            navmesh = scene_dir / f"{scene}.navmesh"
            glb.write_bytes(f"glb-{scene}".encode())
            navmesh.write_bytes(f"navmesh-{scene}".encode())
            scene_records.append(
                {
                    "scene": scene,
                    "glb": {
                        "path": str(glb.resolve()),
                        "bytes": glb.stat().st_size,
                        "sha256": CEILING._file_sha256(glb),
                    },
                    "navmesh": {
                        "path": str(navmesh.resolve()),
                        "bytes": navmesh.stat().st_size,
                        "sha256": CEILING._file_sha256(navmesh),
                    },
                }
            )
        sim_patch = habitat_sim_root / "src" / "esp" / "gfx" / "WindowlessContext.cpp"
        sim_patch.parent.mkdir(parents=True, exist_ok=True)
        sim_patch.write_text("patched habitat sim", encoding="utf-8")
        files = {
            "release-manifest": rlinf / "P8_RELEASE_ARTIFACT_MANIFEST.json",
            "habitat-python": root / "habitat-python",
            "habitat-server": rlinf / "scripts" / "habitat_r2r_service.py",
            "egl-vendor-json": root / "egl.json",
        }
        for path in files.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        files["release-manifest"].write_text("{}", encoding="utf-8")
        for name, path in files.items():
            if name != "release-manifest":
                path.write_text(name, encoding="utf-8")

        target_relative = Path(
            "integrations/rlinf_v03/run_streamvln_p8_successor_ceiling.py"
        )
        target = rlinf / target_relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(MODULE_PATH.read_bytes())
        server_relative = Path("scripts/habitat_r2r_service.py")
        server = rlinf / server_relative
        ceiling_manifest_path = rlinf / CEILING.CEILING_MANIFEST_RELATIVE_PATH
        ceiling_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        episode_rows = [
            {"index": index, "episode_id": f"episode-{index}", "scene": f"scene-{index}"}
            for index in (7, 8)
        ]
        ceiling_manifest = {
            "schema_name": CEILING.CEILING_MANIFEST_SCHEMA_NAME,
            "schema_version": CEILING.CEILING_MANIFEST_SCHEMA_VERSION,
            "ceiling_run_seed": 20260929,
            "scope": {"split": "train"},
            "dataset": {
                "train_json_path": str(dataset.resolve()),
                "train_json_bytes": dataset.stat().st_size,
                "train_json_sha256": CEILING._file_sha256(dataset),
                "episodes": {
                    "expert": episode_rows,
                    "keep": list(episode_rows),
                },
            },
            "deployment": {
                "implementation_files": [
                    {
                        "deployment_relative_path": target_relative.as_posix(),
                        "bytes": target.stat().st_size,
                        "sha256": CEILING._file_sha256(target),
                    },
                    {
                        "deployment_relative_path": server_relative.as_posix(),
                        "bytes": server.stat().st_size,
                        "sha256": CEILING._file_sha256(server),
                    }
                ]
            },
            "release": {
                "release_manifest_raw_sha256": CEILING._file_sha256(
                    files["release-manifest"]
                ),
                "release_manifest_canonical_sha256": CEILING._json_hash({}),
                "release_manifest_deployment_path": "P8_RELEASE_ARTIFACT_MANIFEST.json",
            },
            "gates": {
                "expert": {
                    "episode_count": 2,
                    "max_physical_steps_per_episode": 160,
                    "minimum_successes": 2,
                    "minimum_successful_stops": 2,
                },
                "keep": {
                    "episode_count": 2,
                    "max_physical_steps_per_episode": 40,
                    "minimum_comparisons_per_episode": 1,
                    "required_mismatch_count": 0,
                },
            },
            "paths": {
                "asset_root": str(asset.resolve()),
                "checkpoint": str(checkpoint.resolve()),
                "deployment_root": str(rlinf.resolve()),
                "egl_vendor_json": str(files["egl-vendor-json"].resolve()),
                "habitat_env_lib": str(env_lib.resolve()),
                "habitat_sim_python": str(habitat_sim_python.resolve()),
                "streamvln_source": str(source.resolve()),
                "tokenizer": str(tokenizer.resolve()),
                "vision_tower": str(vision.resolve()),
            },
            "runtime": {
                "habitat_child": {
                    "python_path": str(files["habitat-python"].absolute()),
                    "python_sha256": CEILING._file_sha256(files["habitat-python"]),
                },
                "habitat_sim_patch": {
                    "path": str(sim_patch.resolve()),
                    "bytes": sim_patch.stat().st_size,
                    "sha256": CEILING._file_sha256(sim_patch),
                },
                "egl_vendor_json": {
                    "path": str(files["egl-vendor-json"].resolve()),
                    "bytes": files["egl-vendor-json"].stat().st_size,
                    "sha256": CEILING._file_sha256(files["egl-vendor-json"]),
                },
            },
            "scene_assets": scene_records,
        }
        ceiling_manifest_path.write_text(
            json.dumps(ceiling_manifest, sort_keys=True), encoding="utf-8"
        )
        CEILING.CEILING_PROTOCOL_CONTRACT_SHA256 = (
            CEILING._ceiling_protocol_contract_sha256(ceiling_manifest)
        )
        expert_receipt_path = root / "expert-receipt.json"
        expert_receipt = {
            "schema_name": CEILING.SCHEMA_NAME,
            "schema_version": CEILING.SCHEMA_VERSION,
            "status": "pass",
            "stage": "expert",
            "ceiling_manifest_raw_sha256": CEILING._file_sha256(ceiling_manifest_path),
            "gate_pass": True,
            "gate": {"evaluated": True, "role": "expert", "gate_pass": True},
            "preflight": {
                "ceiling_manifest_raw_sha256": CEILING._file_sha256(ceiling_manifest_path),
                "ceiling_role": "expert",
                "split": "train",
                "episode_indices": [7, 8],
            },
            "result": {"stage": "expert", "split": "train"},
        }
        expert_receipt["receipt_sha256"] = CEILING._json_hash(expert_receipt)
        expert_receipt_path.write_bytes(CEILING._json_bytes(expert_receipt) + b"\n")
        role = "keep" if stage == "keep" else "expert"
        argv = [
            "--stage", stage,
            "--split", split,
            "--output", str(output or root / f"{stage}.json"),
            "--ceiling-manifest", str(ceiling_manifest_path),
            "--release-manifest", str(files["release-manifest"]),
            "--source-root", str(source),
            "--checkpoint", str(checkpoint),
            "--tokenizer", str(tokenizer),
            "--vision", str(vision),
            "--rlinf-root", str(rlinf),
            "--asset-root", str(asset),
            "--habitat-python", str(files["habitat-python"]),
            "--habitat-server", str(files["habitat-server"]),
            "--habitat-sim-python", str(habitat_sim_python),
            "--habitat-env-lib", str(env_lib),
            "--egl-vendor-json", str(files["egl-vendor-json"]),
            "--episode-index", "7",
            "--episode-index", "8",
            "--max-steps", str(CEILING.ROLE_MAX_STEPS[role]),
        ]
        if stage == "keep":
            argv.extend(("--expert-receipt", str(expert_receipt_path)))
        return CEILING.build_parser().parse_args(argv)

    @staticmethod
    def verifier(*, manifest, **_kwargs):
        anchor = CEILING._json_hash(manifest)
        return {
            "anchor": anchor,
            "verification": {
                "manifest_sha256": anchor,
                "source": {"source": "verified"},
                "artifacts": {"artifacts": "verified"},
            },
        }

    def test_authoritative_protocol_anchor_rejects_gate_replacement(self):
        production_manifest = json.loads(
            (MODULE_DIR / "streamvln_p8_successor_ceiling_manifest.json").read_text()
        )
        self.assertEqual(
            CEILING._ceiling_protocol_contract_sha256(production_manifest),
            PRODUCTION_PROTOCOL_ANCHOR,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.make_args(root, stage="expert")
            manifest_path = Path(args.ceiling_manifest)
            manifest = json.loads(manifest_path.read_text())
            manifest["gates"]["expert"]["minimum_successes"] = 0
            manifest_path.write_text(json.dumps(manifest, sort_keys=True))
            with self.assertRaisesRegex(RuntimeError, "protocol contract drifted"):
                CEILING.run(args, release_verifier=self.verifier)

    def test_runtime_scene_hashes_and_cli_paths_fail_closed(self):
        def mutate_python(_args, manifest):
            Path(manifest["runtime"]["habitat_child"]["python_path"]).write_bytes(
                b"changed-python"
            )

        def mutate_sim_patch(_args, manifest):
            Path(manifest["runtime"]["habitat_sim_patch"]["path"]).write_text(
                "changed-patch"
            )

        def mutate_egl(_args, manifest):
            Path(manifest["runtime"]["egl_vendor_json"]["path"]).write_text(
                "changed-egl"
            )

        def mutate_scene(_args, manifest):
            Path(manifest["scene_assets"][0]["glb"]["path"]).write_bytes(
                b"changed-scene"
            )

        def override_server(args, _manifest):
            alternate = Path(args.habitat_server).parent / "alternate-server.py"
            alternate.write_text("alternate")
            args.habitat_server = str(alternate)

        def override_env_lib(args, _manifest):
            alternate = Path(args.habitat_env_lib).parent / "alternate-env-lib"
            alternate.mkdir()
            args.habitat_env_lib = str(alternate)

        cases = (
            ("python", mutate_python, "Habitat Python target SHA-256 drifted"),
            ("sim_patch", mutate_sim_patch, "Habitat-Sim patch target bytes or SHA-256 drifted"),
            ("egl", mutate_egl, "EGL vendor JSON target bytes or SHA-256 drifted"),
            ("scene", mutate_scene, "scene scene-7 glb target bytes or SHA-256 drifted"),
            ("server_path", override_server, "Habitat server path overrides"),
            ("env_lib_path", override_env_lib, "habitat_env_lib path overrides"),
        )
        for name, mutation, message in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                args = self.make_args(root, stage="expert")
                manifest = json.loads(Path(args.ceiling_manifest).read_text())
                mutation(args, manifest)
                with self.assertRaisesRegex(RuntimeError, message):
                    CEILING.run(args, release_verifier=self.verifier)

    def test_canonical_import_must_come_from_verified_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = root / "rlinf" / "envs" / "streamvln" / "module.py"
            expected.parent.mkdir(parents=True)
            expected.write_text("expected")
            wrong = root / "wrong.py"
            wrong.write_text("wrong")
            module = types.SimpleNamespace(
                __name__="rlinf.envs.streamvln.module", __file__=str(wrong)
            )
            with mock.patch.object(CEILING.importlib, "import_module", return_value=module):
                with self.assertRaisesRegex(RuntimeError, "escaped verified rlinf_root"):
                    CEILING._import_canonical("module", rlinf_root=root)

    def test_keep_rejects_rehashed_nonpassing_expert_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.make_args(root, stage="keep")
            receipt_path = Path(args.expert_receipt)
            receipt = json.loads(receipt_path.read_text())
            receipt["gate_pass"] = False
            receipt.pop("receipt_sha256")
            receipt["receipt_sha256"] = CEILING._json_hash(receipt)
            receipt_path.write_bytes(CEILING._json_bytes(receipt) + b"\n")
            with self.assertRaisesRegex(RuntimeError, "passing expert"):
                CEILING.run(args, release_verifier=self.verifier)

    def test_manifest_role_metric_and_step_domains_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.make_args(root, stage="expert")
            manifest_path = Path(args.ceiling_manifest)
            manifest = json.loads(manifest_path.read_text())
            manifest["dataset"]["episodes"]["expert"][1]["scene"] = "scene-7"
            manifest_path.write_text(json.dumps(manifest, sort_keys=True))
            with mock.patch.object(
                CEILING,
                "CEILING_PROTOCOL_CONTRACT_SHA256",
                CEILING._ceiling_protocol_contract_sha256(manifest),
            ):
                with self.assertRaisesRegex(ValueError, "scenes are not unique"):
                    CEILING._load_ceiling_contract(args)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.make_args(root, stage="keep")
            manifest_path = Path(args.ceiling_manifest)
            manifest = json.loads(manifest_path.read_text())
            manifest["dataset"]["episodes"]["keep"][1] = {
                "index": 9,
                "episode_id": "episode-9",
                "scene": "scene-9",
            }
            manifest_path.write_text(json.dumps(manifest, sort_keys=True))
            with mock.patch.object(
                CEILING,
                "CEILING_PROTOCOL_CONTRACT_SHA256",
                CEILING._ceiling_protocol_contract_sha256(manifest),
            ):
                with self.assertRaisesRegex(ValueError, "first two expert"):
                    CEILING._load_ceiling_contract(args)

        for key, value, message in (
            ("success", 1.5, "success metric must be binary"),
            ("spl", 1.5, "SPL metric must be in"),
            ("path_length", -1.0, "path length must be nonnegative"),
            ("collision_count", 0.5, "collision count must be"),
        ):
            with self.subTest(metric=key):
                metrics = {
                    "success": 1.0,
                    "spl": 1.0,
                    "path_length": 0.0,
                    "collision_count": 0.0,
                }
                metrics[key] = value
                with self.assertRaisesRegex(RuntimeError, message):
                    CEILING._terminal_metrics(types.SimpleNamespace(), {"episode": metrics})

        stage_result = {
            "stage": "expert",
            "aggregate": {
                "episode_count": 1,
                "success_count": 1,
                "successful_stop_count": 1,
            },
            "episodes": [{"physical_steps": 161}],
        }
        preflight = {
            "ceiling_role": "expert",
            "gate_requirements": {
                "episode_count": 1,
                "minimum_successes": 1,
                "minimum_successful_stops": 1,
                "max_physical_steps_per_episode": 160,
            },
        }
        gate = CEILING._evaluate_stage_gate(
            stage_result=stage_result, preflight_receipt=preflight
        )
        self.assertFalse(gate["conditions"]["max_physical_steps_per_episode"])

    def test_nontrain_fails_before_paths_or_release_verifier(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.make_args(root, split="val_seen")
            for path in root.iterdir():
                if path.is_file():
                    path.unlink()
            called = []

            def verifier(**_kwargs):
                called.append(True)
                raise AssertionError("release verifier must not run")

            with self.assertRaisesRegex(ValueError, "literal train"):
                CEILING.run(args, release_verifier=verifier)
            self.assertEqual(called, [])
            self.assertFalse(Path(args.output).exists())

    def test_preflight_receipt_is_atomic_and_cannot_load_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.make_args(root)
            model_calls = []
            keep_runtime = CEILING.KeepRuntime(
                make_env=lambda *_args, **_kwargs: None,
                adapter_factory=lambda **_kwargs: model_calls.append(True),
                bridge_factory=lambda **_kwargs: None,
                official_session_factory=lambda **_kwargs: None,
                route_name=RouteName,
                route_decision=RouteDecision,
                topology_factory=lambda _steps: Topology(),
            )
            receipt = CEILING.run(
                args,
                release_verifier=self.verifier,
                keep_runtime=keep_runtime,
            )
            self.assertEqual(model_calls, [])
            self.assertEqual(receipt["stage"], "preflight")
            self.assertEqual(receipt["preflight"]["guards"]["model_load_count"], 0)
            self.assertEqual(receipt["preflight"]["seed"], 20260929)
            self.assertEqual(receipt["preflight"]["max_steps"], 160)
            self.assertEqual(
                receipt["ceiling_manifest_raw_sha256"],
                CEILING._file_sha256(Path(args.ceiling_manifest)),
            )
            self.assertIsNone(receipt["gate_pass"])
            digest = receipt["receipt_sha256"]
            self.assertEqual(
                digest,
                CEILING._json_hash({key: value for key, value in receipt.items() if key != "receipt_sha256"}),
            )
            self.assertEqual(json.loads(Path(args.output).read_text()), receipt)
            with self.assertRaises(FileExistsError):
                CEILING.run(args, release_verifier=self.verifier)

    def test_cli_seed_or_budget_cannot_override_ceiling_manifest(self):
        cases = (
            ("seed", 20260930, "ceiling_run_seed"),
            ("max_steps", 40, "episode order or max_steps"),
        )
        for field, value, message in cases:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                args = self.make_args(root, stage="expert")
                setattr(args, field, value)
                verifier_calls = []

                def verifier(**_kwargs):
                    verifier_calls.append(True)
                    raise AssertionError("release verifier must not run")

                with self.assertRaisesRegex(ValueError, message):
                    CEILING.run(args, release_verifier=verifier)
                self.assertEqual(verifier_calls, [])
                self.assertFalse(Path(args.output).exists())

    def test_habitat_python_path_does_not_resolve_venv_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            requested = root / "venv" / "bin" / "python"
            requested.parent.mkdir(parents=True)
            requested.write_text("venv interpreter", encoding="utf-8")
            resolved_target = root / "base" / "python"
            resolved_target.parent.mkdir(parents=True)
            resolved_target.write_text("base interpreter", encoding="utf-8")
            with mock.patch.object(Path, "resolve", return_value=resolved_target):
                observed = CEILING._preserved_absolute_file(
                    requested, name="habitat_python"
                )
            self.assertEqual(observed, Path(str(requested.absolute())))
            self.assertNotEqual(observed, resolved_target)

    def test_expert_executes_waypoint_then_stop_one_primitive_at_a_time(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.make_args(root, stage="expert")
            topologies = []

            def topology_factory(_max_steps, _planner):
                topology = Topology()
                topologies.append(topology)
                return topology

            runtime = CEILING.ExpertRuntime(
                make_env=lambda _cfg, *, split, episode_index, max_steps: ExpertEnv(episode_index),
                route_name=RouteName,
                route_decision=RouteDecision,
                topology_factory=topology_factory,
                feedback_from_dict=lambda value: value,
                successor_end_reason=lambda **kwargs: (
                    ("stop", False)
                    if kwargs["action"] == 0
                    else (("plan_complete", True) if kwargs["plan_complete"] else (None, False))
                ),
            )
            receipt = CEILING.run(
                args,
                release_verifier=self.verifier,
                expert_runtime=runtime,
            )
            result = receipt["result"]
            self.assertEqual(result["stage"], "expert")
            episode = result["episodes"][0]
            self.assertEqual(episode["physical_steps"], 3)
            self.assertEqual(len(episode["official_frames"]), 3)
            self.assertEqual(
                [row["plan"]["requested_route"] for row in episode["decisions"]],
                ["waypoint", "stop"],
            )
            self.assertEqual(topologies[0].decision_count, 2)
            self.assertEqual(topologies[0].updates[0][0], (3, 1))
            self.assertEqual(topologies[0].updates[1][0], (0,))
            self.assertEqual(result["aggregate"]["success_count"], 2)
            self.assertEqual(result["aggregate"]["successful_stop_count"], 2)
            self.assertEqual(result["aggregate"]["stop_count"], 2)
            self.assertEqual(
                result["aggregate"]["action_counts"],
                {"STOP": 2, "FORWARD": 2, "LEFT": 0, "RIGHT": 2},
            )
            self.assertEqual(result["aggregate"]["total_path_length_m"], 0.5)
            self.assertEqual(result["aggregate"]["collision_count"], 0.0)
            self.assertEqual(result["aggregate"]["mean_final_distance_m"], 1.0)
            self.assertEqual(result["guards"]["model_load_count"], 0)
            self.assertTrue(receipt["gate_pass"])

    def test_keep_is_only_stage_that_constructs_adapter_and_uses_exact_bridge(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.make_args(root, stage="keep")
            adapter_factory_calls = []
            driver_calls = []
            FakeBridge.exact_flags.clear()

            def adapter_factory(**kwargs):
                adapter_factory_calls.append(kwargs)
                return FakeAdapter(driver_calls)

            runtime = CEILING.KeepRuntime(
                make_env=lambda _cfg, *, split, episode_index, max_steps: KeepEnv(episode_index),
                adapter_factory=adapter_factory,
                bridge_factory=FakeBridge,
                official_session_factory=FakeSession,
                route_name=RouteName,
                route_decision=RouteDecision,
                topology_factory=lambda _steps: Topology(),
            )
            receipt = CEILING.run(
                args,
                release_verifier=self.verifier,
                keep_runtime=runtime,
            )
            self.assertEqual(len(adapter_factory_calls), 1)
            self.assertEqual(len(driver_calls), 2)
            self.assertEqual(FakeBridge.exact_flags, [True, True])
            result = receipt["result"]
            self.assertEqual(result["stage"], "keep")
            self.assertTrue(result["guards"]["keep_only"])
            self.assertEqual(result["guards"]["optimizer_steps"], 0)
            self.assertEqual(result["aggregate"]["comparison_count"], 4)
            self.assertEqual(result["aggregate"]["mismatch_count"], 0)
            self.assertEqual(result["aggregate"]["successful_stop_count"], 2)
            self.assertEqual(
                result["aggregate"]["action_counts"],
                {"STOP": 2, "FORWARD": 2, "LEFT": 0, "RIGHT": 0},
            )
            episode = result["episodes"][0]
            self.assertEqual(episode["physical_steps"], 2)
            self.assertEqual(
                [row["executed_actions"] for row in episode["decisions"]],
                [[1], [0]],
            )
            self.assertTrue(episode["terminated"])
            self.assertTrue(receipt["gate_pass"])

    def test_keep_mismatch_is_a_written_negative_scientific_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.make_args(root, stage="keep")
            runtime = CEILING.KeepRuntime(
                make_env=lambda _cfg, *, split, episode_index, max_steps: KeepEnv(
                    episode_index
                ),
                adapter_factory=lambda **_kwargs: FakeAdapter([]),
                bridge_factory=MismatchBridge,
                official_session_factory=FakeSession,
                route_name=RouteName,
                route_decision=RouteDecision,
                topology_factory=lambda _steps: Topology(),
            )
            receipt = CEILING.run(
                args,
                release_verifier=self.verifier,
                keep_runtime=runtime,
            )
            self.assertFalse(receipt["gate_pass"])
            self.assertEqual(receipt["result"]["aggregate"]["comparison_count"], 4)
            self.assertEqual(receipt["result"]["aggregate"]["mismatch_count"], 4)
            self.assertFalse(
                receipt["gate"]["conditions"]["mismatch_count"]
            )
            self.assertEqual(json.loads(Path(args.output).read_text()), receipt)

    def test_failed_stage_never_publishes_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.make_args(root, stage="expert")
            runtime = CEILING.ExpertRuntime(
                make_env=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
                route_name=RouteName,
                route_decision=RouteDecision,
                topology_factory=lambda *_args: Topology(),
                feedback_from_dict=lambda value: value,
                successor_end_reason=lambda **_kwargs: (None, False),
            )
            with self.assertRaisesRegex(RuntimeError, "boom"):
                CEILING.run(
                    args,
                    release_verifier=self.verifier,
                    expert_runtime=runtime,
                )
            self.assertFalse(Path(args.output).exists())


if __name__ == "__main__":
    unittest.main()
