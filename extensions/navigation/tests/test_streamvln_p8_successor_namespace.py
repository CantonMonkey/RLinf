from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest


MODULE_DIR = Path(__file__).resolve().parents[1] / "integrations" / "rlinf_v03"
SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
PACKAGE_MODULES = (
    "streamvln_executor_contract.py",
    "streamvln_p8_official_keep.py",
    "streamvln_p8_official_frame.py",
    "streamvln_p8_successor_action.py",
    "streamvln_p8_successor_executor.py",
    "streamvln_topology.py",
    "streamvln_p8_successor_runtime.py",
    "streamvln_p8_official_adapter.py",
    "streamvln_env.py",
    "streamvln_p8_successor_env.py",
)


IDENTITY_PROBE = r"""
import hashlib
import importlib
import random
import struct
import sys
import types
from pathlib import Path

# The identity probe imports the real production modules but does not require
# the heavyweight numerical stack merely to define the environment classes.
for name in ("numpy", "torch", "gymnasium"):
    sys.modules[name] = types.ModuleType(name)

runner = importlib.import_module("run_streamvln_p8_successor_ceiling")
action = importlib.import_module(
    "rlinf.envs.streamvln.streamvln_p8_successor_action"
)
contract = importlib.import_module(
    "rlinf.envs.streamvln.streamvln_executor_contract"
)
executor = importlib.import_module(
    "rlinf.envs.streamvln.streamvln_p8_successor_executor"
)
frame_module = importlib.import_module(
    "rlinf.envs.streamvln.streamvln_p8_official_frame"
)
official = importlib.import_module(
    "rlinf.envs.streamvln.streamvln_p8_official_keep"
)
adapter = importlib.import_module(
    "rlinf.envs.streamvln.streamvln_p8_official_adapter"
)
runtime = importlib.import_module(
    "rlinf.envs.streamvln.streamvln_p8_successor_runtime"
)
topology_module = importlib.import_module(
    "rlinf.envs.streamvln.streamvln_topology"
)
stream_env = importlib.import_module("rlinf.envs.streamvln.streamvln_env")
successor_env = importlib.import_module(
    "rlinf.envs.streamvln.streamvln_p8_successor_env"
)

assert runtime.OfficialFrame is frame_module.OfficialFrame
assert adapter.OfficialFrame is frame_module.OfficialFrame
assert stream_env.OfficialFrame is frame_module.OfficialFrame
assert executor.CompiledRoute is action.CompiledRoute
assert successor_env.RouteName is action.RouteName
assert successor_env.StreamVLNEnv is stream_env.StreamVLNEnv
assert successor_env.StreamVLNP8SuccessorEnv.__module__ == (
    "rlinf.envs.streamvln.streamvln_p8_successor_env"
)
canonical_modules = (
    action,
    contract,
    executor,
    frame_module,
    official,
    adapter,
    runtime,
    topology_module,
    stream_env,
    successor_env,
)
assert all(
    module.__name__.startswith("rlinf.envs.streamvln.")
    for module in canonical_modules
)
assert action.RouteName.__module__ == (
    "rlinf.envs.streamvln.streamvln_p8_successor_action"
)
assert frame_module.OfficialFrame.__module__ == (
    "rlinf.envs.streamvln.streamvln_p8_official_frame"
)
assert not {
    "streamvln_executor_contract",
    "streamvln_p8_official_keep",
    "streamvln_p8_official_frame",
    "streamvln_p8_successor_action",
    "streamvln_p8_successor_executor",
    "streamvln_topology",
    "streamvln_p8_successor_runtime",
    "streamvln_p8_official_adapter",
    "streamvln_env",
    "streamvln_p8_successor_env",
} & set(sys.modules)

# A registry entry with the right class name but the legacy top-level module
# identity must be rejected before construction.
envs_package = importlib.import_module("rlinf.envs")
decoy_env_cls = type("StreamVLNP8SuccessorEnv", (), {})
decoy_env_cls.__module__ = "streamvln_p8_successor_env"
envs_package.get_env_cls = lambda *_args, **_kwargs: decoy_env_cls


class OmegaConf:
    @staticmethod
    def create(value):
        return types.SimpleNamespace(env_type=value["env_type"])


omegaconf = types.ModuleType("omegaconf")
omegaconf.OmegaConf = OmegaConf
sys.modules["omegaconf"] = omegaconf
try:
    runtime.make_successor_env(
        {
            "seed": 1,
            "habitat_python": "python",
            "habitat_server": "service.py",
            "asset_root": "assets",
            "habitat_sim_python": "src_python",
            "habitat_env_lib": "lib",
            "egl_vendor_json": "egl.json",
            "habitat_startup_timeout_s": 1.0,
        },
        split="train",
        episode_index=0,
        max_steps=40,
    )
except RuntimeError as exc:
    assert "canonical P8 successor environment" in str(exc)
else:
    raise AssertionError("same-name top-level successor environment was accepted")

keep_runtime = runner._load_keep_runtime(Path.cwd())
expert_runtime = runner._load_expert_runtime(Path.cwd())
assert keep_runtime.route_name is action.RouteName
assert keep_runtime.bridge_factory is runtime.SuccessorRunnerBridge
assert expert_runtime.route_name is action.RouteName
assert expert_runtime.route_decision is action.RouteDecision

identity = {
    "instruction_text": "stop at the target",
    "episode_id": "identity-episode",
    "scene_id": "identity-scene",
    "instruction_sha256": hashlib.sha256(b"stop at the target").hexdigest(),
}
rgb = bytes(frame_module.OFFICIAL_RGB_HEIGHT * frame_module.OFFICIAL_RGB_WIDTH * 3)
depth = struct.pack("<f", 0.5) * (
    frame_module.OFFICIAL_RGB_HEIGHT * frame_module.OFFICIAL_RGB_WIDTH
)
matrix = struct.pack(
    "<16d",
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
)
first_frame = frame_module.OfficialFrame(
    env_id=0,
    episode_serial=0,
    episode_generation="0" * 32,
    physical_step=0,
    episode_id=identity["episode_id"],
    scene_id=identity["scene_id"],
    instruction_sha256=identity["instruction_sha256"],
    rgb_u8=rgb,
    depth_normalized_f32le=depth,
    pose_f64le=matrix,
    intrinsics_f64le=matrix,
    rgb_sha256=hashlib.sha256(rgb).hexdigest(),
    depth_sha256=hashlib.sha256(depth).hexdigest(),
    pose_sha256=hashlib.sha256(matrix).hexdigest(),
    intrinsics_sha256=hashlib.sha256(matrix).hexdigest(),
)


class Flag:
    def __init__(self, value):
        self.value = value

    def any(self):
        return self.value


class Env:
    def __init__(self):
        self.identity = dict(identity)
        self.env_id = 0
        self._episode_serial = -1
        self.successor_planner = action.SuccessorPlanner()
        self.observed_types = []

    def reset(self):
        self._episode_serial = 0
        self.successor_planner.reset_episode()
        return {"rgb": 0}, {}

    def observe_official_frame(self):
        self.observed_types.append(type(first_frame))
        return first_frame

    def chunk_step(self, payload, execute_count):
        assert execute_count == 1
        primitive_action = int(payload[0][0][0])
        assert primitive_action == action.STOP_ACTION
        row = contract.primitive_execution(
            primitive_index=0,
            action=primitive_action,
            previous_path_length_m=0.0,
            current_path_length_m=0.0,
            collision=False,
            environment_reward=0.0,
            terminated=True,
            truncated=False,
        )
        feedback = contract.chunk_feedback((primitive_action,), (row,)).to_dict()
        return [{"rgb": 1}], None, Flag(True), Flag(False), [
            {"executor_feedback": feedback}
        ]


env = Env()
generation_frames = []
runtime.validate_official_runtime_receipt = lambda _receipt: None


def driver_factory(env_id, episode_identity, episode_serial):
    session = official.OfficialEvalSession(
        env_id=env_id,
        instruction=episode_identity["instruction_text"],
        conjunction_rng=random.Random(7),
    )

    def generate(_request, frames):
        generation_frames.extend(frames)
        assert type(frames[0]) is frame_module.OfficialFrame
        return runtime.OfficialGenerationResult(
            decoded_sequence="STOP",
            output_ref="output-0",
            past_key_values_ref="past-0",
        )

    return runtime.OfficialKeepDriver(
        session=session,
        runtime_receipt={},
        generation_adapter=generate,
        cache_reset_adapter=lambda _event: None,
        episode_serial=episode_serial,
        episode_identity=episode_identity,
    )


bridge = runtime.SuccessorRunnerBridge(
    env=env,
    topology_factory=lambda: topology_module.OnlineTopology(max_decisions=40),
    official_driver_factory=driver_factory,
    exact_control=True,
)
bridge.reset_episode()
keep_decision = action.RouteDecision(
    decision_index=0,
    route=action.RouteName.KEEP_ORIGINAL,
    episode_token=bridge.episode_token,
    physical_step=0,
)
keep_result = bridge.execute_decision(keep_decision)
assert env.observed_types == [frame_module.OfficialFrame]
assert generation_frames == [first_frame]
assert keep_result.official_shadow_actions == (action.STOP_ACTION,)
assert keep_result.executed_actions == (action.STOP_ACTION,)

expert_decision = expert_runtime.route_decision(
    decision_index=0,
    route=expert_runtime.route_name.STOP,
    episode_token="expert-token",
    physical_step=0,
)
expert_plan = action.SuccessorPlanner().plan(
    expert_decision, topology_module.OnlineTopology(max_decisions=160)
)
assert expert_decision.route is action.RouteName.STOP
assert expert_plan.requested_route is action.RouteName.STOP
assert expert_plan.executed_route is action.RouteName.STOP
print("canonical-namespace-identity-ok")
"""


SERVICE_IMPORT_PROBE = r"""
import importlib.util
from pathlib import Path
import sys

root = Path.cwd()
service_path = root / "scripts" / "habitat_r2r_service.py"
sys.path.insert(0, str(service_path.parent))
spec = importlib.util.spec_from_file_location("habitat_r2r_service", service_path)
service = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = service
spec.loader.exec_module(service)

assert service._OFFICIAL_FRAME_PROTOCOL_ERROR is None
assert service.encode_official_blob.__module__ == (
    "rlinf.envs.streamvln.streamvln_p8_official_frame"
)
assert "rlinf.envs.streamvln.streamvln_p8_official_frame" in sys.modules
assert "rlinf.envs.streamvln.streamvln_p8_official_keep" in sys.modules
assert "rlinf.envs.streamvln.streamvln_env" not in sys.modules
assert "streamvln_p8_official_frame" not in sys.modules
assert "streamvln_p8_official_keep" not in sys.modules
assert "torch" not in sys.modules
assert "gym" not in sys.modules
assert "gymnasium" not in sys.modules
expected_paths = {
    "rlinf": root / "rlinf",
    "rlinf.envs": root / "rlinf" / "envs",
    "rlinf.envs.streamvln": root / "rlinf" / "envs" / "streamvln",
}
for name, expected_path in expected_paths.items():
    package = sys.modules[name]
    assert getattr(
        package, service._CHILD_NAMESPACE_ROOT_ATTRIBUTE
    ) == str(root.resolve())
    assert list(package.__path__) == [str(expected_path.resolve())]
print("canonical-service-import-ok")
"""


SERVICE_PRELOADED_PARENT_PROBE = r"""
import importlib.util
from pathlib import Path
import sys
import types

root = Path.cwd()
service_path = root / "scripts" / "habitat_r2r_service.py"
sys.path.insert(0, str(service_path.parent))
preloaded = types.ModuleType("rlinf")
preloaded.__file__ = str(root / "rlinf" / "__init__.py")
preloaded.__path__ = [str(root / "rlinf")]
sys.modules["rlinf"] = preloaded

spec = importlib.util.spec_from_file_location("habitat_r2r_service", service_path)
service = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = service
spec.loader.exec_module(service)

error = service._OFFICIAL_FRAME_PROTOCOL_ERROR
assert isinstance(error, service._OfficialFrameNamespaceError)
assert "refuses preloaded parent package: rlinf" in str(error)
assert "rlinf.envs" not in sys.modules
assert "rlinf.envs.streamvln.streamvln_p8_official_frame" not in sys.modules
assert "streamvln_p8_official_frame" not in sys.modules
assert "torch" not in sys.modules
assert "gym" not in sys.modules
assert "gymnasium" not in sys.modules

try:
    service.HabitatR2RBackend(
        types.SimpleNamespace(official_streamvln_frame=True)
    )
except RuntimeError as exc:
    assert "official StreamVLN frame protocol modules are not deployed" in str(exc)
    assert exc.__cause__ is error
else:
    raise AssertionError("official backend accepted a preloaded parent package")
print("preloaded-parent-fail-closed-ok")
"""


class CanonicalNamespaceIdentityTest(unittest.TestCase):
    def test_production_sources_have_no_broad_sibling_import_fallback(self):
        package_branch_files = (
            "streamvln_env.py",
            "streamvln_p8_successor_env.py",
            "streamvln_p8_successor_runtime.py",
            "streamvln_p8_official_frame.py",
            "streamvln_p8_official_adapter.py",
            "streamvln_p8_successor_executor.py",
            "streamvln_topology.py",
        )
        for name in package_branch_files:
            with self.subTest(module=name):
                source = (MODULE_DIR / name).read_text(encoding="utf-8")
                self.assertNotIn("except ImportError", source)
                self.assertIn("if __package__:", source)
        runner_source = (
            MODULE_DIR / "run_streamvln_p8_successor_ceiling.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'CANONICAL_STREAMVLN_PACKAGE = "rlinf.envs.streamvln"',
            runner_source,
        )
        self.assertNotIn("_import_sibling", runner_source)
        service_source = (SCRIPT_DIR / "habitat_r2r_service.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "from rlinf.envs.streamvln.streamvln_p8_official_frame import",
            service_source,
        )
        self.assertIn("def _install_streamvln_protocol_namespace", service_source)
        self.assertNotIn('"integrations" / "rlinf_v03"', service_source)
        streamvln_init = (MODULE_DIR / "streamvln_init.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("def __getattr__", streamvln_init)

    def test_habitat_service_import_is_canonical_and_frame_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scripts = root / "scripts"
            streamvln_package = root / "rlinf" / "envs" / "streamvln"
            scripts.mkdir(parents=True)
            streamvln_package.mkdir(parents=True)
            for package in (
                root / "rlinf",
                root / "rlinf" / "envs",
                streamvln_package,
            ):
                (package / "__init__.py").write_text(
                    'raise AssertionError("Habitat child executed a package initializer")\n',
                    encoding="utf-8",
                )
            for name in (
                "streamvln_p8_official_keep.py",
                "streamvln_p8_official_frame.py",
            ):
                shutil.copy2(MODULE_DIR / name, streamvln_package / name)
            for name in (
                "habitat_r2r_service.py",
                "habitat_r2r_rgb_instruction_smoke.py",
            ):
                shutil.copy2(SCRIPT_DIR / name, scripts / name)

            environment = os.environ.copy()
            existing = environment.get("PYTHONPATH")
            search_path = [str(root)]
            if existing:
                search_path.append(existing)
            environment["PYTHONPATH"] = os.pathsep.join(search_path)
            completed = subprocess.run(
                [sys.executable, "-c", textwrap.dedent(SERVICE_IMPORT_PROBE)],
                cwd=root,
                env=environment,
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
            )
            self.assertIn("canonical-service-import-ok", completed.stdout)

    def test_habitat_service_rejects_preloaded_parent_package(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scripts = root / "scripts"
            streamvln_package = root / "rlinf" / "envs" / "streamvln"
            scripts.mkdir(parents=True)
            streamvln_package.mkdir(parents=True)
            for package in (
                root / "rlinf",
                root / "rlinf" / "envs",
                streamvln_package,
            ):
                (package / "__init__.py").write_text(
                    'raise AssertionError("Habitat child executed a package initializer")\n',
                    encoding="utf-8",
                )
            for name in (
                "streamvln_p8_official_keep.py",
                "streamvln_p8_official_frame.py",
            ):
                shutil.copy2(MODULE_DIR / name, streamvln_package / name)
            for name in (
                "habitat_r2r_service.py",
                "habitat_r2r_rgb_instruction_smoke.py",
            ):
                shutil.copy2(SCRIPT_DIR / name, scripts / name)

            environment = os.environ.copy()
            existing = environment.get("PYTHONPATH")
            search_path = [str(root)]
            if existing:
                search_path.append(existing)
            environment["PYTHONPATH"] = os.pathsep.join(search_path)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    textwrap.dedent(SERVICE_PRELOADED_PARENT_PROBE),
                ],
                cwd=root,
                env=environment,
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
            )
            self.assertIn("preloaded-parent-fail-closed-ok", completed.stdout)

    def test_keep_first_frame_and_expert_stop_share_canonical_classes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            streamvln_package = root / "rlinf" / "envs" / "streamvln"
            vln_package = root / "rlinf" / "envs" / "vln"
            streamvln_package.mkdir(parents=True)
            vln_package.mkdir(parents=True)
            for package in (root / "rlinf", root / "rlinf" / "envs"):
                (package / "__init__.py").write_text("", encoding="utf-8")
            shutil.copy2(
                MODULE_DIR / "streamvln_init.py",
                streamvln_package / "__init__.py",
            )
            (vln_package / "__init__.py").write_text("", encoding="utf-8")
            (vln_package / "vln_env.py").write_text(
                "class VLNEnv:\n    pass\n", encoding="utf-8"
            )
            for name in PACKAGE_MODULES:
                shutil.copy2(MODULE_DIR / name, streamvln_package / name)

            environment = os.environ.copy()
            existing = environment.get("PYTHONPATH")
            search_path = [str(root), str(MODULE_DIR)]
            if existing:
                search_path.append(existing)
            environment["PYTHONPATH"] = os.pathsep.join(search_path)
            completed = subprocess.run(
                [sys.executable, "-c", textwrap.dedent(IDENTITY_PROBE)],
                cwd=root,
                env=environment,
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
            )
            self.assertIn("canonical-namespace-identity-ok", completed.stdout)


if __name__ == "__main__":
    unittest.main()
