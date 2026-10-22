"""Exercise the exact RLinf observation handoff that would otherwise drop VLN data."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any


STAGING = Path(__file__).resolve().parents[1] / "integrations/rlinf_v03"


def load_env_output_methods():
    path = STAGING / "embodied_io_struct_vln.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    source_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "EnvOutput")
    methods = [node for node in source_class.body if isinstance(node, ast.FunctionDef)
               and node.name in {"prepare_observations", "to_dict"}]
    cls = ast.ClassDef(name="MinimalEnvOutput", bases=[], keywords=[], body=methods, decorator_list=[])
    ast.fix_missing_locations(cls)
    scope = {"Any": Any}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), scope)
    return scope["MinimalEnvOutput"]


def load_batch_inference():
    path = STAGING / "huggingface_worker_vln.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    method = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef) and node.name == "_infer_env_batch_size")
    method.decorator_list = []
    scope = {"Any": Any, "torch": SimpleNamespace(Tensor=FakeTensor)}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    return scope["_infer_env_batch_size"]


class FakeTensor:
    def __init__(self, shape):
        self.shape = shape


class VLNTransportBoundaryTests(unittest.TestCase):
    def test_env_output_retains_exact_vln_keys_and_final_observation(self):
        output = load_env_output_methods()()
        rgb = FakeTensor((1, 64, 64, 3))
        ids = FakeTensor((1, 200))
        mask = FakeTensor((1, 200))
        obs = {"rgb": rgb, "instruction_ids": ids, "instruction_mask": mask}
        output.obs = obs
        output.final_obs = obs
        for name in ("dones", "terminations", "truncations", "rewards", "env_infos",
                     "intervene_actions", "intervene_flags"):
            setattr(output, name, None)
        encoded = output.to_dict()
        self.assertEqual(set(encoded["obs"]), set(obs))
        self.assertEqual(set(encoded["final_obs"]), set(obs))
        self.assertIs(encoded["obs"]["rgb"], rgb)
        self.assertIs(encoded["obs"]["instruction_ids"], ids)
        self.assertIs(encoded["obs"]["instruction_mask"], mask)
        self.assertNotIn("states", encoded["obs"])

    def test_rejects_mixed_privileged_observations_and_keeps_pointnav(self):
        output = load_env_output_methods()()
        obs = {"rgb": object(), "instruction_ids": object(),
               "instruction_mask": object(), "goal": object()}
        with self.assertRaisesRegex(ValueError, "exactly RGB"):
            output.prepare_observations(obs)
        old = output.prepare_observations({"states": FakeTensor((1, 2))})
        self.assertEqual(set(old), {"main_images", "wrist_images",
                                    "extra_view_images", "states", "task_descriptions"})
        self.assertEqual(old["states"].shape, (1, 2))

    def test_rollout_infers_batch_from_vln_rgb(self):
        infer = load_batch_inference()
        obs = {"obs": {"rgb": FakeTensor((2, 64, 64, 3)),
                       "instruction_ids": FakeTensor((2, 200)),
                       "instruction_mask": FakeTensor((2, 200))}}
        self.assertEqual(infer(obs), 2)
        self.assertEqual(infer({"obs": {"states": FakeTensor((1, 2))}}), 1)


if __name__ == "__main__":
    unittest.main()
