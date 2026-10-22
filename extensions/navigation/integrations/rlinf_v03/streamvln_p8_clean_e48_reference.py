"""Independent clean-e48 reference path used only by the A800 parity run.

This module intentionally has no dependency on the P8 official adapter or its
frozen keep-contract module.  Heavy libraries and the verified upstream
evaluator namespace are supplied by the caller after provenance verification.
"""

from __future__ import annotations

import copy
from collections import OrderedDict, deque
import ast
import hashlib
import itertools
from pathlib import Path
import random
import re
from types import SimpleNamespace


REFERENCE_TENSOR_NAMES = ("images", "depths", "poses", "intrinsics")


def load_verified_evaluator_namespace(evaluator_path: str | Path, *, torch_module):
    """Extract evaluator methods without importing Habitat or running the file."""

    path = Path(evaluator_path).resolve(strict=True)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    source_class = next(
        (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "VLNEvaluator"),
        None,
    )
    if source_class is None:
        raise RuntimeError("clean-e48 evaluator does not expose VLNEvaluator")
    methods = [
        copy.deepcopy(node)
        for node in source_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in {"preprocess_qwen", "parse_actions"}
    ]
    if {method.name for method in methods} != {"preprocess_qwen", "parse_actions"}:
        raise RuntimeError("clean-e48 evaluator parity methods are missing")
    extracted = ast.Module(
        body=[ast.ClassDef(name="VLNEvaluator", bases=[], keywords=[], body=methods, decorator_list=[])],
        type_ignores=[],
    )
    ast.fix_missing_locations(extracted)
    namespace = {
        "copy": copy,
        "itertools": itertools,
        "random": random,
        "re": re,
        "torch": torch_module,
        "transformers": SimpleNamespace(PreTrainedTokenizer=object),
        "DEFAULT_IMAGE_TOKEN": "<image>",
        "DEFAULT_MEMORY_TOKEN": "<memory>",
        "IMAGE_TOKEN_INDEX": -200,
        "MEMORY_TOKEN_INDEX": -300,
    }
    exec(compile(extracted, str(path), "exec"), namespace)
    return SimpleNamespace(VLNEvaluator=namespace["VLNEvaluator"])


def tensor_audit(tensor, torch_module) -> dict:
    """Hash normalized contiguous CPU bytes without adapter code reuse."""

    value = tensor.detach().cpu().contiguous()
    payload = value.view(torch_module.uint8).numpy().tobytes(order="C")
    return {
        "dtype": str(value.dtype),
        "shape": [int(item) for item in value.shape],
        "sha256": hashlib.sha256(payload).hexdigest(),
        "nbytes": len(payload),
    }


def cache_structure(value, torch_module):
    if isinstance(value, torch_module.Tensor):
        return {
            "type": "tensor",
            "dtype": str(value.dtype),
            "shape": [int(item) for item in value.shape],
        }
    if isinstance(value, (tuple, list)):
        return [cache_structure(item, torch_module) for item in value]
    if isinstance(value, dict):
        return {
            str(key): cache_structure(item, torch_module)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if value is None:
        return None
    if hasattr(value, "key_cache") and hasattr(value, "value_cache"):
        return {
            "type": type(value).__name__,
            "key_cache": cache_structure(value.key_cache, torch_module),
            "value_cache": cache_structure(value.value_cache, torch_module),
        }
    return {"type": type(value).__name__}


class ReferenceQueue:
    """Literal e48 action regex and one-pop-per-physical-frame queue."""

    def __init__(self) -> None:
        self.pending = deque()

    def load_actions(self, values) -> list[int]:
        if self.pending:
            raise RuntimeError("reference generation attempted over a non-empty queue")
        actions = list(values) or [0]
        if any(type(value) is not int or value not in (0, 1, 2, 3) for value in actions):
            raise RuntimeError("reference evaluator returned an invalid primitive")
        self.pending.extend(actions)
        return list(self.pending)

    def pop(self) -> int:
        if not self.pending:
            raise RuntimeError("reference queue requires generation before pop")
        return int(self.pending.popleft())


class CleanE48ReferenceRunner:
    """Thin model runner with state isolated from the official adapter path."""

    def __init__(
        self,
        *,
        evaluator_namespace,
        model,
        tokenizer,
        image_processor,
        device,
        modules,
    ) -> None:
        self.evaluator_namespace = evaluator_namespace
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.device = device
        self.torch = modules["torch"]
        self.np = modules["numpy"]
        self.Image = modules["image"]
        self.to_numpy_array = modules["to_numpy_array"]
        self.filter_depth = modules["filter_depth"]
        self.output_ids = None
        self.past_key_values = None

    def reset(self) -> None:
        self.model.reset_for_env(0)
        self.output_ids = None
        self.past_key_values = None

    def parse_actions(self, decoded: str) -> list[int]:
        owner = object.__new__(self.evaluator_namespace.VLNEvaluator)
        owner.actions2idx = OrderedDict(
            (("STOP", [0]), ("↑", [1]), ("←", [2]), ("→", [3]))
        )
        return owner.parse_actions(decoded)

    def input_ids(self, *, source_messages, add_system: bool, conjunction: str):
        owner = object.__new__(self.evaluator_namespace.VLNEvaluator)
        owner.conjunctions = [conjunction]
        preprocess = self.evaluator_namespace.VLNEvaluator.preprocess_qwen
        result = preprocess(
            owner,
            [copy.deepcopy(source_messages)],
            self.tokenizer,
            True,
            add_system=add_system,
        )
        if not isinstance(result, tuple) or len(result) < 1:
            raise RuntimeError("clean-e48 preprocess_qwen output schema drifted")
        value = result[0]
        if self.output_ids is not None:
            value = self.torch.cat(
                [self.output_ids, value.to(self.output_ids.device)], dim=1
            )
        return value.to(self.device)

    def frames(self, frames):
        images, depths, poses, intrinsics = [], [], [], []
        for frame in frames:
            rgb = self.np.frombuffer(frame.rgb_u8, dtype=self.np.uint8).reshape(
                480, 640, 3
            )
            image = self.Image.fromarray(rgb).convert("RGB")
            images.append(
                self.image_processor.preprocess(images=image, return_tensors="pt")[
                    "pixel_values"
                ][0]
            )
            depth = self.np.frombuffer(
                frame.depth_normalized_f32le, dtype="<f4"
            ).reshape(480, 640).copy()
            depth = self.filter_depth(depth, blur_type=None) * 10000.0
            depth_image = self.Image.fromarray(
                depth.astype(self.np.uint16), mode="I;16"
            ).resize((384, 384), self.Image.NEAREST)
            depths.append(
                self.torch.from_numpy(self.to_numpy_array(depth_image) / 1000).float()
            )
            poses.append(
                self.torch.from_numpy(
                    self.np.frombuffer(frame.pose_f64le, dtype="<f8").reshape(4, 4)
                )
            )
            intrinsic = self.np.frombuffer(
                frame.intrinsics_f64le, dtype="<f8"
            ).reshape(4, 4).copy()
            intrinsic[0] /= 640 / 384
            intrinsic[1] /= 480 / 384
            intrinsic[0, 2] -= 0
            intrinsics.append(self.torch.from_numpy(intrinsic).float())
        values = {
            "images": images,
            "depths": depths,
            "poses": poses,
            "intrinsics": intrinsics,
        }
        return {
            name: self.torch.stack(value)
            .unsqueeze(0)
            .to(self.device)
            .to(self.torch.bfloat16)
            for name, value in values.items()
        }

    def generate(self, *, source_messages, add_system, conjunction, time_ids, frames):
        tensors = self.frames(frames)
        input_ids = self.input_ids(
            source_messages=source_messages,
            add_system=add_system,
            conjunction=conjunction,
        )
        model_inputs = dict(
            tensors,
            inputs=input_ids,
            env_id=0,
            time_ids=[list(time_ids)],
            task_type=[0],
        )
        generation_kwargs = {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": 10000,
            "use_cache": True,
            "return_dict_in_generate": True,
        }
        with self.torch.inference_mode():
            outputs = self.model.generate(
                **model_inputs,
                **generation_kwargs,
                past_key_values=self.past_key_values,
            )
        self.output_ids = outputs.sequences
        self.past_key_values = outputs.past_key_values
        decoded = self.tokenizer.batch_decode(
            self.output_ids, skip_special_tokens=False
        )[0].strip()
        return tensors, input_ids, decoded


def reference_source_messages(user_content: str):
    return [
        {"from": "human", "value": user_content},
        {"from": "gpt", "value": ""},
    ]


def reference_user_content(instruction: str, *, step: int) -> str:
    prompt = (
        "<video>\nYou are an autonomous navigation assistant. Your task is to "
        "<instruction>. Devise an action sequence to follow the instruction using "
        "the four actions: TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, MOVE "
        "FORWARD (↑) by 25 centimeters, or STOP."
    ).replace("<instruction>.", instruction)
    if step:
        prompt += " These are your historical observations <memory>."
    return prompt.replace("<video>\n", "")


def generation_observation(runner, *, tensors, input_ids, decoded, queue) -> dict:
    embeds = runner.model.cache[0]["inputs_embeds"]
    return {
        "output_ids": tensor_audit(runner.output_ids, runner.torch),
        "decoded": decoded,
        "parsed_queue": queue,
        "past_key_values_structure": cache_structure(
            runner.past_key_values, runner.torch
        ),
        "model_curr_t": runner.model.curr_t[0],
        "inputs_embeds_seq_len": int(embeds.shape[1]),
        "input_ids": tensor_audit(input_ids, runner.torch),
        "tensors": {
            name: tensor_audit(tensors[name], runner.torch)
            for name in REFERENCE_TENSOR_NAMES
        },
    }
