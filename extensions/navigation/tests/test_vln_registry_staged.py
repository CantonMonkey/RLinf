"""Check the staged RLinf v0.3 VLN registration without importing Ray or Torch."""

import ast
import dataclasses
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch


STAGING = Path(__file__).resolve().parents[1] / "integrations/rlinf_v03"


def load_supported_model_subset():
    source = STAGING / "config_vln.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    selected = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "SupportedModel":
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "SupportedModel"
            or isinstance(target, ast.Name) and target.id == "EMBODIED_MODEL"
            for target in node.targets
        ):
            selected.append(node)
    namespace = {"dataclasses": dataclasses, "ClassVar": ClassVar,
                 "__name__": "staged_vln_config_subset"}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["SupportedModel"], namespace["EMBODIED_MODEL"]


def model_names_in_compare(node):
    return {
        element.attr
        for descendant in ast.walk(node)
        if isinstance(descendant, ast.List)
        for element in descendant.elts
        if isinstance(element, ast.Attribute)
        and isinstance(element.value, ast.Name)
        and element.value.id == "SupportedModel"
    }


class VLNRegistryStagedTests(unittest.TestCase):
    def test_supported_and_embodied_model_registration(self):
        supported, embodied = load_supported_model_subset()
        self.assertEqual(supported.VLN_POLICY.value, "vln_policy")
        self.assertIn(supported.VLN_POLICY, embodied)
        self.assertIn(supported.MLP_POLICY, embodied)

    def test_registry_routes_only_new_type_to_vln_builder(self):
        supported, embodied = load_supported_model_subset()
        config = types.ModuleType("rlinf.config")
        config.SupportedModel = supported
        config.EMBODIED_MODEL = embodied
        config.torch_dtype_from_precision = lambda precision: f"dtype:{precision}"
        scheduler = types.ModuleType("rlinf.scheduler")
        scheduler.Worker = type("Worker", (), {"torch_platform": None})
        omega = types.ModuleType("omegaconf")
        omega.DictConfig = object
        rlinf = types.ModuleType("rlinf")
        rlinf.__path__ = []
        models = types.ModuleType("rlinf.models")
        models.__path__ = []
        embodiment = types.ModuleType("rlinf.models.embodiment")
        embodiment.__path__ = []
        vln = types.ModuleType("rlinf.models.embodiment.vln_policy")
        seen = []

        def build(cfg, dtype):
            seen.append((cfg, dtype))
            return "vln-model"

        vln.get_model = build
        fake = {"rlinf": rlinf, "rlinf.config": config,
                "rlinf.scheduler": scheduler, "omegaconf": omega,
                "rlinf.models": models, "rlinf.models.embodiment": embodiment,
                "rlinf.models.embodiment.vln_policy": vln}
        source = STAGING / "models_init_vln.py"
        spec = importlib.util.spec_from_file_location("staged_vln_models_test", source)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, fake):
            spec.loader.exec_module(module)
            cfg = types.SimpleNamespace(model_type="vln_policy", precision="32", is_lora=False)
            self.assertEqual(module.get_model(cfg), "vln-model")
        self.assertEqual(seen, [(cfg, "dtype:32")])
        self.assertIn("mlp_policy", module._MODEL_REGISTRY)

    def test_rollout_passes_mode_and_replay_observations_to_vln_policy(self):
        source = STAGING / "huggingface_worker_vln.py"
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        predict = next(node for node in ast.walk(tree)
                       if isinstance(node, ast.FunctionDef) and node.name == "predict")
        model_branches = [node for node in predict.body if isinstance(node, ast.If)
                          and "VLN_POLICY" in model_names_in_compare(node.test)]
        self.assertEqual(len(model_branches), 2)
        branch_text = [ast.unparse(node) for node in model_branches]
        self.assertTrue(any("kwargs = {'mode': mode}" in text for text in branch_text))
        self.assertTrue(any("kwargs['return_obs']" in text for text in branch_text))
        self.assertTrue(all("MLP_POLICY" in model_names_in_compare(node.test)
                            for node in model_branches))


if __name__ == "__main__":
    unittest.main()
