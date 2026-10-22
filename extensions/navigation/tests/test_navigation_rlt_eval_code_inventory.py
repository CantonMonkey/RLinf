from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


MODULE_DIR = Path(__file__).resolve().parents[1] / "integrations" / "rlinf_v03"
EVAL_PATH = MODULE_DIR / "navigation_rlt_eval.py"
REQUIRED_FILES = (
    "build_navigation_rlt_train_split.py",
    "lightnav_a3_policy.py",
    "lightnav_a3_rollout.py",
    "lightnav_l0_native_probe.py",
    "navigation_rlt_action.py",
    "navigation_rlt_compare.py",
    "navigation_rlt_eval.py",
    "navigation_rlt_obs_archive.py",
    "navigation_rlt_parallel.py",
    "navigation_rlt_policy.py",
    "navigation_rlt_post_training.py",
    "navigation_rlt_replay.py",
    "navigation_rlt_rollout.py",
    "streamvln_executor_contract.py",
    "streamvln_topology.py",
    "navrl/__init__.py",
    "navrl/contracts.py",
    "navrl/envs/__init__.py",
    "navrl/envs/base.py",
    "navrl/envs/grid_mock.py",
    "navrl/episode.py",
    "navrl/metrics.py",
    "navrl/policies/__init__.py",
    "navrl/policies/base.py",
    "navrl/rewards.py",
    "navrl/stack/__init__.py",
    "navrl/stack/checkpoint.py",
    "navrl/stack/collector.py",
    "navrl/stack/contracts.py",
    "navrl/stack/execution.py",
    "navrl/stack/interfaces.py",
    "navrl/stack/legacy.py",
    "navrl/stack/ppo.py",
    "navrl/stack/returns.py",
    "run_navigation_rlt_ebk_eval.sh",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _module(name: str, **attributes: object) -> types.ModuleType:
    result = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(result, key, value)
    return result


def _load_eval_module() -> types.ModuleType:
    class Dummy:
        pass

    def unused(*_args: object, **_kwargs: object) -> object:
        return None

    stubs = {
        "numpy": _module("numpy", ndarray=object),
        "build_navigation_rlt_train_split": _module(
            "build_navigation_rlt_train_split",
            FROZEN_DEVELOPMENT_SCENE_EPISODES=(),
        ),
        "lightnav_a3_policy": _module(
            "lightnav_a3_policy",
            CausalTopologyState=Dummy,
            sha256_file=_sha256_file,
        ),
        "lightnav_a3_rollout": _module(
            "lightnav_a3_rollout",
            _bool_success=unused,
            _collision_event=unused,
            _episode_metrics=unused,
            _normalized_command=unused,
            _physical_velocity=unused,
            _scene_name=unused,
            _validate_velocity_contract=unused,
            aggregate=unused,
        ),
        "lightnav_l0_native_probe": _module(
            "lightnav_l0_native_probe",
            HabitatProtocolError=RuntimeError,
            LightNavPrediction=Dummy,
            LightNavWebSocketClient=Dummy,
            VelocitySpec=Dummy,
            ZmqHabitatClient=Dummy,
            _collision_observation=unused,
            _metric=unused,
            _optional_metric=unused,
            atomic_write_json=unused,
            classify_velocity_action=unused,
            encode_rgb_jpeg_base64=unused,
            extract_instruction=unused,
            latency_summary=unused,
            redact_endpoint=unused,
            waypoint_to_velocity_action=unused,
        ),
        "navigation_rlt_action": _module(
            "navigation_rlt_action",
            MODE_NAMES=("KEEP", "ADJUST", "STOP", "RECOVER"),
            RECOVER=3,
            STOP=2,
            CompiledAction=Dummy,
            compile_action=unused,
            valid_mode_mask=unused,
        ),
        "navigation_rlt_replay": _module(
            "navigation_rlt_replay", LIGHTNAV_SOURCE_REVISION="0" * 40
        ),
        "navigation_rlt_rollout": _module(
            "navigation_rlt_rollout",
            ACTION_SCHEMA="action",
            ARTIFACT_CAPTURE_RULE="capture",
            BEHAVIOR_A3="a3",
            BEHAVIOR_NAMES=("native", "a3", "navigation-rlt"),
            BEHAVIOR_NATIVE="native",
            BEHAVIOR_POLICY_ID="policy",
            BEHAVIOR_RLT="navigation-rlt",
            EXECUTOR_SCHEMA="executor",
            POLICY_PREFIX_SELECTION_RULE="rule",
            POLICY_PREFIX_SELECTION_SCHEMA="selection",
            PREFIX_FEATURE_SCHEMA="feature",
            REFERENCE_SCHEMA="reference",
            TOPOLOGY_SCHEMA="topology",
            ArtifactFile=Dummy,
            Behavior=Dummy,
            ContextArtifact=Dummy,
            ContextLightNavClient=Dummy,
            ContinuousOnlineTopology=Dummy,
            DecisionState=Dummy,
            _artifact_relative_path=unused,
            _directory_no_reparse=unused,
            _load_behavior=unused,
            _path_entry_exists=unused,
            _query_state=unused,
            _regular_file_no_reparse=lambda path, _name: (
                None
                if Path(path).is_file() and not Path(path).is_symlink()
                else (_ for _ in ()).throw(RuntimeError("not a regular file"))
            ),
            _resolve_artifact_root=unused,
            _validate_sha256=lambda value, _name: (
                value
                if isinstance(value, str)
                and len(value) == 64
                and value == value.lower()
                and all(character in "0123456789abcdef" for character in value)
                else (_ for _ in ()).throw(ValueError("invalid SHA-256"))
            ),
            _verified_output_receipt=unused,
        ),
        "streamvln_topology": _module(
            "streamvln_topology",
            CAUSAL_NODE_FEATURE_DIM=1,
            causal_topology_payload=unused,
        ),
    }
    if str(MODULE_DIR) not in sys.path:
        sys.path.insert(0, str(MODULE_DIR))
    spec = importlib.util.spec_from_file_location(
        "_navigation_rlt_eval_inventory_test", EVAL_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load navigation_rlt_eval.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(spec.name, None)
    return module


class NavigationRLTEvalCodeInventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_eval_module()

    def _inventory(self) -> dict[str, object]:
        return {
            "schema_name": "navigation_rlt_evaluation_code_inventory",
            "schema_version": 1,
            "navigation_code_commit": "a" * 40,
            "files": {
                name: _sha256_file(self.module._evaluation_source(name))
                for name in REQUIRED_FILES
            },
        }

    def test_inventory_binds_required_bundle_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            path = Path(directory_name) / "evaluation-code.json"
            path.write_text(json.dumps(self._inventory()), encoding="utf-8")
            resolved, inventory, observed_sha = (
                self.module._load_evaluation_code_inventory(
                    path, expected_sha256=_sha256_file(path)
                )
            )

            self.assertEqual(resolved, path.resolve())
            self.assertEqual(observed_sha, _sha256_file(path))
            self.assertEqual(
                set(inventory["files"]), set(REQUIRED_FILES)
            )
            actions = {
                action.dest: action for action in self.module.parser()._actions
            }
            self.assertTrue(actions["evaluation_code_inventory"].required)
            self.assertTrue(
                actions["expected_evaluation_code_inventory_sha256"].required
            )

    def test_inventory_rejects_tampered_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            path = Path(directory_name) / "evaluation-code.json"
            inventory = self._inventory()
            inventory["files"]["navigation_rlt_compare.py"] = "0" * 64
            path.write_text(json.dumps(inventory), encoding="utf-8")

            with self.assertRaises(self.module.NavigationRLTEvalError):
                self.module._load_evaluation_code_inventory(
                    path, expected_sha256=_sha256_file(path)
                )

    def test_inventory_rejects_one_missing_runtime_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            path = Path(directory_name) / "evaluation-code.json"
            inventory = self._inventory()
            inventory["files"].pop("navrl/stack/returns.py")
            path.write_text(json.dumps(inventory), encoding="utf-8")
            with self.assertRaises(self.module.NavigationRLTEvalError):
                self.module._load_evaluation_code_inventory(
                    path, expected_sha256=_sha256_file(path)
                )


if __name__ == "__main__":
    unittest.main()
