"""Dependency-injected tests for the pinned live StreamVLN adapter."""

import copy
from dataclasses import dataclass
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import random
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).parents[1]
STAGE = ROOT / "integrations" / "rlinf_v03"
sys.path.insert(0, str(STAGE))

import streamvln_p8_official_adapter as adapter_module  # noqa: E402
import streamvln_p8_official_keep as keep_module  # noqa: E402
import streamvln_p8_successor_runtime as runtime_module  # noqa: E402
from streamvln_p8_official_adapter import (  # noqa: E402
    OfficialFramePreprocessor,
    PinnedStreamVLNGenerationAdapter,
    PreparedOfficialFrames,
    build_feature_v2_depth_summary,
    build_full_content_artifact_manifest,
    verify_full_content_artifact_manifest,
)
from streamvln_p8_official_frame import (  # noqa: E402
    OFFICIAL_RGB_HEIGHT,
    OFFICIAL_RGB_WIDTH,
    OfficialFrame,
)
from streamvln_p8_official_keep import (  # noqa: E402
    DECODE_KWARGS,
    GENERATION_KWARGS,
    IMAGE_TOKEN_INDEX,
    MEMORY_TOKEN_INDEX,
    PINNED_ARTIFACT_IDS,
    PINNED_EVALUATOR_SHA256,
    PINNED_EVALUATOR_SOURCE_RELATIVE_PATH,
    PINNED_MODEL_CODE_SHA256,
    PINNED_MODEL_SOURCE_RELATIVE_PATH,
    PINNED_SOURCE_IMPORT_ROOT_RELATIVE_PATH,
    PINNED_SOURCE_ROOT_KIND,
    PINNED_STREAMVLN_COMMIT,
    STATE_AUDIT_SCOPE,
    TEST_ONLY_CONTRACT_VALIDATION_LEVEL,
    OfficialEvalSession,
    validate_official_runtime_receipt,
)


class FakeTensor:
    def __init__(self, data, *, dtype="torch.int64", shape=None, device="cpu"):
        self.data = data
        self.dtype = dtype
        self.device = device
        self._version = 0
        if shape is not None:
            self.shape = tuple(shape)
        elif isinstance(data, list) and data and isinstance(data[0], list):
            self.shape = (len(data), len(data[0]))
        elif isinstance(data, list):
            self.shape = (len(data),)
        else:
            self.shape = ()

    def to(self, value):
        if isinstance(value, str) and value.startswith("torch."):
            self.dtype = value
        else:
            self.device = value
        return self


class FakeInferenceMode:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


class FakeDevice:
    def __init__(self, value, index=None):
        if isinstance(value, FakeDevice):
            self.type = value.type
            self.index = value.index if index is None else index
            return
        if not isinstance(value, str):
            raise TypeError("fake device requires text or FakeDevice")
        parts = value.split(":", 1)
        self.type = parts[0]
        self.index = int(parts[1]) if len(parts) == 2 else index

    def __str__(self):
        if self.index is None:
            return self.type
        return f"{self.type}:{self.index}"


class FakeCuda:
    @staticmethod
    def current_device():
        return 0


class FakeTorch:
    __version__ = "2.1.2+cu121"
    Tensor = FakeTensor
    long = "torch.int64"
    bfloat16 = "torch.bfloat16"
    uint8 = "torch.uint8"
    cuda = FakeCuda()

    @staticmethod
    def device(value, index=None):
        return FakeDevice(value, index)

    @staticmethod
    def tensor(data, dtype=None):
        return FakeTensor(data, dtype=dtype)

    @staticmethod
    def cat(values, dim):
        if dim != 1 or len(values) != 2:
            raise AssertionError("official continuation must concatenate on dim 1")
        return FakeTensor(
            [list(values[0].data[0]) + list(values[1].data[0])],
            dtype=values[0].dtype,
            device=values[0].device,
        )

    @staticmethod
    def inference_mode():
        return FakeInferenceMode()


class FeatureTensor:
    def __init__(self, data, *, dtype="torch.float32", shape=None, device="cpu"):
        self.data = copy.deepcopy(data)
        self.dtype = dtype
        self.device = device
        self._version = 0
        if shape is not None:
            self.shape = tuple(shape)
        elif isinstance(data, list):
            self.shape = (len(data),)
        else:
            self.shape = ()

    def __getitem__(self, key):
        if not isinstance(key, tuple) or len(key) != 3:
            raise AssertionError("feature tensor only supports contextual row selection")
        batch, token, feature = key
        if batch != 0 or feature != slice(None):
            raise AssertionError("feature tensor contextual row selection drifted")
        row = self.data[batch][token]
        return FeatureTensor(
            row,
            dtype=self.dtype,
            shape=(len(row),),
            device=self.device,
        )

    def detach(self):
        return self

    def to(self, value=None, *, dtype=None, device=None):
        target_dtype = dtype
        target_device = self.device if device is None else device
        if value is not None:
            if isinstance(value, str) and value.startswith("torch."):
                target_dtype = value
            else:
                target_device = value
        return FeatureTensor(
            self.data,
            dtype=self.dtype if target_dtype is None else target_dtype,
            shape=self.shape,
            device=target_device,
        )

    def cpu(self):
        return FeatureTensor(
            self.data, dtype=self.dtype, shape=self.shape, device="cpu"
        )

    def contiguous(self):
        return self

    def clone(self):
        return FeatureTensor(
            self.data, dtype=self.dtype, shape=self.shape, device=self.device
        )

    def zero_(self):
        def zero(value):
            if isinstance(value, list):
                return [zero(item) for item in value]
            return 0.0

        self.data = zero(self.data)
        self._version += 1
        return self


class FeatureDepthMask:
    def __init__(self, values):
        self.values = tuple(bool(value) for value in values)

    def __and__(self, other):
        return FeatureDepthMask(
            left and right for left, right in zip(self.values, other.values)
        )


class FeatureDepthTensor:
    def __init__(self, values, *, shape=None):
        self.values = tuple(float(value) for value in values)
        self.shape = (len(self.values),) if shape is None else tuple(shape)
        self.dtype = "torch.float32"
        self.device = "cpu"
        self._version = 0

    def reshape(self, *shape):
        if math.prod(shape) != len(self.values):
            raise AssertionError("feature depth reshape drifted")
        return FeatureDepthTensor(self.values, shape=shape)

    def __getitem__(self, key):
        if isinstance(key, FeatureDepthMask):
            return FeatureDepthTensor(
                value for value, keep in zip(self.values, key.values) if keep
            )
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or key[0] != slice(None)
            or not isinstance(key[1], slice)
            or len(self.shape) != 2
        ):
            raise AssertionError("feature depth sector selection drifted")
        rows, columns = self.shape
        start, stop, step = key[1].indices(columns)
        if step != 1:
            raise AssertionError("feature depth sector stride drifted")
        selected = []
        for row in range(rows):
            offset = row * columns
            selected.extend(self.values[offset + start : offset + stop])
        return FeatureDepthTensor(selected, shape=(rows, stop - start))

    def __mul__(self, value):
        return FeatureDepthTensor(
            (item * float(value) for item in self.values), shape=self.shape
        )

    def __add__(self, value):
        return FeatureDepthTensor(
            (item + float(value) for item in self.values), shape=self.shape
        )

    def __gt__(self, value):
        return FeatureDepthMask(item > float(value) for item in self.values)

    def __le__(self, value):
        return FeatureDepthMask(item <= float(value) for item in self.values)

    def numel(self):
        return len(self.values)


class FeatureTorch(FakeTorch):
    Tensor = (FakeTensor, FeatureTensor, FeatureDepthTensor)
    float32 = "torch.float32"

    @staticmethod
    def tensor(data, dtype=None, device=None):
        if dtype == FeatureTorch.float32:
            return FeatureTensor(data, dtype=dtype, device=device or "cpu")
        return FakeTorch.tensor(data, dtype=dtype)

    @staticmethod
    def frombuffer(buffer, dtype=None):
        if dtype != FeatureTorch.float32 or len(buffer) % 4:
            raise AssertionError("feature depth buffer contract drifted")
        count = len(buffer) // 4
        return FeatureDepthTensor(struct.unpack(f"<{count}f", buffer))

    @staticmethod
    def kthvalue(value, rank):
        selected = sorted(value.values)[rank - 1]
        return SimpleNamespace(values=SimpleNamespace(item=lambda: selected))

    @staticmethod
    def isfinite(value):
        if isinstance(value, FeatureDepthTensor):
            return FeatureDepthMask(math.isfinite(item) for item in value.values)

        def flatten(item):
            if isinstance(item, list):
                for child in item:
                    yield from flatten(child)
            else:
                yield item

        finite = all(math.isfinite(float(item)) for item in flatten(value.data))
        return SimpleNamespace(all=lambda: finite)


class FakeTokenizer:
    def __init__(self, calls=None, decoded_sequence="prefix ↑"):
        self.calls = [] if calls is None else calls
        self.decoded_sequence = decoded_sequence
        self.additional_special_tokens_ids = [151643, 151645]
        self.chat_template = None
        self.token_ids = {"<image>": 700, "<memory>": 701}

    def __deepcopy__(self, memo):
        return FakeTokenizer(self.calls, self.decoded_sequence)

    def add_tokens(self, tokens, special_tokens):
        self.calls.append(("add_tokens", tuple(tokens), special_tokens))

    def convert_tokens_to_ids(self, token):
        return self.token_ids[token]

    def apply_chat_template(self, messages):
        message = dict(messages[0])
        self.calls.append(("template", message["role"], message["content"]))
        ids = [{"system": 10, "user": 20, "assistant": 30}[message["role"]]]
        if "<memory>" in message["content"]:
            ids.append(self.token_ids["<memory>"])
        if "<image>" in message["content"]:
            ids.append(self.token_ids["<image>"])
        return ids

    def batch_decode(self, values, **kwargs):
        self.calls.append(("decode", values, dict(kwargs)))
        return [self.decoded_sequence]


@dataclass
class FakeParameter:
    dtype: str = "torch.bfloat16"
    requires_grad: bool = False
    floating: bool = True
    device: str = "cuda:0"

    def is_floating_point(self):
        return self.floating


class FakeModel:
    def __init__(
        self,
        parameters=None,
        *,
        inputs_embeds_growth=1,
        curr_t_increment=1,
        omit_inputs_embeds=False,
    ):
        self.config = SimpleNamespace(_attn_implementation="flash_attention_2")
        self.model = SimpleNamespace(num_history=8)
        self.curr_t = [0]
        self.cache = [{}]
        self.calls = []
        self.generated_kv = []
        self.inputs_embeds_growth = inputs_embeds_growth
        self.curr_t_increment = curr_t_increment
        self.omit_inputs_embeds = omit_inputs_embeds
        self._parameters = tuple(parameters or (FakeParameter(), FakeParameter()))

    def parameters(self):
        return iter(self._parameters)

    def reset_for_env(self, env_id):
        self.curr_t[env_id] = 0
        self.cache[env_id] = {}

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        env_id = kwargs["env_id"]
        previous_cache = self.cache[env_id]
        previous_embeds = previous_cache.get("inputs_embeds")
        previous_length = 0 if previous_embeds is None else previous_embeds.shape[1]
        self.curr_t[env_id] += self.curr_t_increment
        if self.omit_inputs_embeds:
            self.cache[env_id] = {}
        else:
            next_length = previous_length + self.inputs_embeds_growth
            self.cache[env_id] = {
                "inputs_embeds": FakeTensor(
                    [next_length],
                    dtype="torch.bfloat16",
                    shape=(1, next_length, 4),
                    device="cuda:0",
                )
            }
        token = 400 + self.curr_t[env_id]
        kv = (
            FakeTensor(
                [token],
                dtype="torch.bfloat16",
                shape=(1, 1, 1, 1),
                device="cuda:0",
            ),
        )
        self.generated_kv.append(kv)
        return SimpleNamespace(
            sequences=FakeTensor([[token]], dtype="torch.int64", device="cuda:0"),
            past_key_values=kv,
        )


class FakeRMSNorm:
    def __init__(self):
        self._hooks = {}
        self._next_hook_id = 0
        self.register_count = 0
        self.remove_count = 0
        self.call_count = 0
        self.hook_return_values = []

    def register_forward_hook(self, hook):
        hook_id = self._next_hook_id
        self._next_hook_id += 1
        self._hooks[hook_id] = hook
        self.register_count += 1
        owner = self

        class Handle:
            def remove(self):
                if hook_id in owner._hooks:
                    del owner._hooks[hook_id]
                    owner.remove_count += 1

        return Handle()

    def __call__(self, output):
        self.call_count += 1
        result = output
        for hook in tuple(self._hooks.values()):
            replacement = hook(self, (output,), result)
            self.hook_return_values.append(replacement)
            if replacement is not None:
                result = replacement
        return result


class FeatureV2FakeModel(FakeModel):
    def __init__(self):
        super().__init__()
        self.config.hidden_size = 3584
        self.model.norm = FakeRMSNorm()
        self.fail_generation = False
        self.norm_outputs_unchanged = []

    def get_model(self):
        return self.model

    def generate(self, **kwargs):
        base = len(self.calls) + 1
        prefill_rows = [[float(base + offset)] * 3584 for offset in range(3)]
        prefill = FeatureTensor(
            [prefill_rows],
            dtype="torch.bfloat16",
            shape=(1, 3, 3584),
            device="cuda:0",
        )
        self.norm_outputs_unchanged.append(self.model.norm(prefill) is prefill)
        if self.fail_generation:
            raise RuntimeError("synthetic feature-v2 generation failure")
        decode = FeatureTensor(
            [[[float(base + 100)] * 3584]],
            dtype="torch.bfloat16",
            shape=(1, 1, 3584),
            device="cuda:0",
        )
        self.norm_outputs_unchanged.append(self.model.norm(decode) is decode)
        return super().generate(**kwargs)


RGB_BYTES = bytes(OFFICIAL_RGB_HEIGHT * OFFICIAL_RGB_WIDTH * 3)
DEPTH_BYTES = struct.pack("<f", 0.5) * (OFFICIAL_RGB_HEIGHT * OFFICIAL_RGB_WIDTH)
POSE_BYTES = struct.pack(
    "<16d",
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
)
INTRINSICS_BYTES = struct.pack(
    "<16d",
    386, 0, 319.5, 0,
    0, 386, 239.5, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
)


def make_frame(step, identity, *, depth_bytes=DEPTH_BYTES):
    return OfficialFrame(
        env_id=0,
        episode_serial=0,
        episode_generation="0" * 32,
        physical_step=step,
        episode_id=identity["episode_id"],
        scene_id=identity["scene_id"],
        instruction_sha256=identity["instruction_sha256"],
        rgb_u8=RGB_BYTES,
        depth_normalized_f32le=depth_bytes,
        pose_f64le=POSE_BYTES,
        intrinsics_f64le=INTRINSICS_BYTES,
        rgb_sha256=hashlib.sha256(RGB_BYTES).hexdigest(),
        depth_sha256=hashlib.sha256(depth_bytes).hexdigest(),
        pose_sha256=hashlib.sha256(POSE_BYTES).hexdigest(),
        intrinsics_sha256=hashlib.sha256(INTRINSICS_BYTES).hexdigest(),
    )


def fake_tensor_audit(tensor):
    payload = repr(tensor.data).encode("utf-8")
    return {
        "dtype": tensor.dtype,
        "shape": list(tensor.shape),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "nbytes": len(payload),
    }


def make_candidate_runtime_observation():
    def file_entry(path, digest):
        return {
            "relative_path": path,
            "entry_type": "regular_file",
            "link_target": None,
            "size": 1,
            "sha256": digest,
        }

    source_files = [
        file_entry("streamvln/model/helper.py", "c" * 64),
        file_entry(PINNED_MODEL_SOURCE_RELATIVE_PATH, PINNED_MODEL_CODE_SHA256),
        file_entry(
            PINNED_EVALUATOR_SOURCE_RELATIVE_PATH, PINNED_EVALUATOR_SHA256
        ),
    ]
    artifact_manifests = {
        name: {
            "artifact_id": artifact_id,
            "files": [file_entry("payload.bin", str(index + 1) * 64)],
        }
        for index, (name, artifact_id) in enumerate(PINNED_ARTIFACT_IDS.items())
    }
    release_manifest = {
        "schema_name": adapter_module.ARTIFACT_MANIFEST_SCHEMA_NAME,
        "schema_version": adapter_module.ARTIFACT_MANIFEST_SCHEMA_VERSION,
        "verification_scope": "candidate_full_content",
        "source": {
            "source_id": PINNED_STREAMVLN_COMMIT,
            "source_root_kind": PINNED_SOURCE_ROOT_KIND,
            "import_root_relative_path": PINNED_SOURCE_IMPORT_ROOT_RELATIVE_PATH,
            "python_files": source_files,
        },
        "artifacts": artifact_manifests,
    }
    manifest_sha256 = adapter_module._json_sha256(release_manifest)
    return {
        "verification_scope": "candidate_full_content",
        "manifest_sha256": manifest_sha256,
        "release_manifest": release_manifest,
        "source": {
            "source_id": PINNED_STREAMVLN_COMMIT,
            "source_root_kind": PINNED_SOURCE_ROOT_KIND,
            "source_import_root_relative_path": (
                PINNED_SOURCE_IMPORT_ROOT_RELATIVE_PATH
            ),
            "python_file_count": len(source_files),
            "python_content_manifest_sha256": adapter_module._json_sha256(
                source_files
            ),
            "model_source_relative_path": PINNED_MODEL_SOURCE_RELATIVE_PATH,
            "model_source_sha256": PINNED_MODEL_CODE_SHA256,
            "evaluator_source_relative_path": (
                PINNED_EVALUATOR_SOURCE_RELATIVE_PATH
            ),
            "evaluator_sha256": PINNED_EVALUATOR_SHA256,
            "imported_model_module_relative_path": PINNED_MODEL_SOURCE_RELATIVE_PATH,
            "imported_model_module_sha256": PINNED_MODEL_CODE_SHA256,
            "model_class_module": "model.stream_video_vln",
        },
        "artifacts": {
            name: {
                "artifact_id": record["artifact_id"],
                "file_count": len(record["files"]),
                "content_manifest_sha256": adapter_module._json_sha256(
                    record["files"]
                ),
            }
            for name, record in artifact_manifests.items()
        },
    }

class OfficialAdapterTests(unittest.TestCase):
    def setUp(self):
        self.instruction = "walk to the chair"
        self.identity = {
            "episode_id": "episode-0",
            "scene_id": "scene-0",
            "instruction_text": self.instruction,
            "instruction_sha256": hashlib.sha256(
                self.instruction.encode("utf-8")
            ).hexdigest(),
        }
        self.model = FakeModel()
        self.tokenizer = FakeTokenizer()

        def preprocess(frames):
            return PreparedOfficialFrames(
                tensors={
                    "images": FakeTensor([], dtype="torch.bfloat16", shape=(1, len(frames), 3, 384, 384), device="cuda:0"),
                    "depths": FakeTensor([], dtype="torch.bfloat16", shape=(1, len(frames), 384, 384), device="cuda:0"),
                    "poses": FakeTensor([], dtype="torch.bfloat16", shape=(1, len(frames), 4, 4), device="cuda:0"),
                    "intrinsics": FakeTensor([], dtype="torch.bfloat16", shape=(1, len(frames), 4, 4), device="cuda:0"),
                },
                frame_refs=tuple(frame.frame_ref for frame in frames),
            )

        self.preprocess = preprocess
        self.adapter = self.make_adapter()

    def make_adapter(
        self,
        *,
        model=None,
        tokenizer=None,
        preprocessor=None,
        tensor_auditor=fake_tensor_audit,
        device="cuda:0",
        torch_module=FakeTorch,
        enable_feature_v2=False,
    ):
        return PinnedStreamVLNGenerationAdapter(
            model=self.model if model is None else model,
            tokenizer=self.tokenizer if tokenizer is None else tokenizer,
            device=device,
            env_id=0,
            torch_module=torch_module,
            transformers_version="4.45.1",
            frame_preprocessor=self.preprocess if preprocessor is None else preprocessor,
            tensor_auditor=tensor_auditor,
            enable_feature_v2=enable_feature_v2,
        )

    def make_driver(self, adapter):
        def validate_test_contract(receipt):
            validate_official_runtime_receipt(
                receipt,
                validation_level=TEST_ONLY_CONTRACT_VALIDATION_LEVEL,
            )

        with mock.patch.object(
            runtime_module,
            "validate_official_runtime_receipt",
            side_effect=validate_test_contract,
        ):
            return adapter.make_driver(
                session=OfficialEvalSession(
                    env_id=0,
                    instruction=self.instruction,
                    conjunction_rng=random.Random(3),
                ),
                episode_serial=0,
                episode_identity=self.identity,
            )

    def make_source_root(self, base):
        source_root = base / "source"
        paths = {
            "model": source_root / PINNED_MODEL_SOURCE_RELATIVE_PATH,
            "evaluator": source_root / PINNED_EVALUATOR_SOURCE_RELATIVE_PATH,
            "helper": source_root / "streamvln" / "model" / "helper.py",
        }
        for name, path in paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{name}-source".encode("utf-8"))
        return source_root

    def make_artifact_roots(self, base):
        roots = {}
        for index, (name, artifact_id) in enumerate(PINNED_ARTIFACT_IDS.items()):
            if name in {"tokenizer", "vision_tower"}:
                repository = base / f"{name}-hf"
                (repository / "blobs").mkdir(parents=True)
                root = repository / "snapshots" / artifact_id
            else:
                root = base / artifact_id
            (root / "nested").mkdir(parents=True)
            (root / "config.json").write_bytes(f"{name}-config".encode("utf-8"))
            (root / "nested" / f"payload-{index}.bin").write_bytes(
                bytes((index, index + 1, index + 2))
            )
            roots[name] = root
        return roots

    def test_driver_uses_full_tokenizer_and_continuation_state(self):
        driver = self.make_driver(self.adapter)
        first = driver.prepare_shadow_action(make_frame(0, self.identity))
        self.assertEqual(first, 1)
        first_kv = self.adapter.past_key_values
        driver.acknowledge_physical_step(terminated=False, truncated=False)
        second = driver.prepare_shadow_action(make_frame(1, self.identity))
        self.assertEqual(second, 1)

        self.assertEqual(len(self.model.calls), 2)
        self.assertIsNone(self.model.calls[0]["past_key_values"])
        self.assertIs(self.model.calls[1]["past_key_values"], first_kv)
        for key, value in GENERATION_KWARGS.items():
            self.assertEqual(self.model.calls[0][key], value)
        first_ids = self.model.calls[0]["inputs"].data[0]
        second_ids = self.model.calls[1]["inputs"].data[0]
        self.assertIn(IMAGE_TOKEN_INDEX, first_ids)
        self.assertNotIn(MEMORY_TOKEN_INDEX, first_ids)
        self.assertEqual(second_ids[0], 401)
        self.assertIn(IMAGE_TOKEN_INDEX, second_ids)
        self.assertIsNone(self.tokenizer.chat_template)
        template_roles = [
            call[1] for call in self.tokenizer.calls if call[0] == "template"
        ]
        self.assertEqual(
            template_roles,
            ["system", "user", "assistant", "user", "assistant"],
        )
        self.assertEqual(self.model.curr_t[0], 2)
        self.assertIn("inputs_embeds", self.model.cache[0])
        self.assertEqual(len(self.adapter.generation_audits), 2)
        self.assertEqual(
            self.adapter.generation_audits[1]["model_curr_t_before"], 1
        )
        self.assertNotEqual(
            self.adapter.generation_audits[0]["output_ref"],
            self.adapter.generation_audits[1]["output_ref"],
        )
        for audit in self.adapter.generation_audits:
            self.assertEqual(len(audit["output_audit"]["sha256"]), 64)
            self.assertEqual(audit["output_audit"]["dtype"], "torch.int64")
            self.assertEqual(audit["state_audit_scope"], STATE_AUDIT_SCOPE)
            self.assertFalse(
                audit["state_audit_scope"]["deep_kv_content_hash_in_steady_state"]
            )
        self.assertIsNone(
            self.adapter.generation_audits[0]["inputs_embeds_seq_len_before"]
        )
        self.assertEqual(
            self.adapter.generation_audits[0]["inputs_embeds_seq_len_after"], 1
        )
        self.assertEqual(
            self.adapter.generation_audits[1]["inputs_embeds_seq_len_before"], 1
        )
        self.assertEqual(
            self.adapter.generation_audits[1]["inputs_embeds_seq_len_after"], 2
        )
        decode_call = self.tokenizer.calls[-1]
        self.assertEqual(decode_call[0], "decode")
        self.assertIs(decode_call[1], self.adapter.output_ids)
        self.assertEqual(decode_call[2], DECODE_KWARGS)
        self.assertFalse(self.adapter.enable_feature_v2)
        self.assertTrue(
            all("feature_v2" not in audit for audit in self.adapter.generation_audits)
        )
        with self.assertRaisesRegex(RuntimeError, "no feature-v2 snapshot"):
            self.adapter.latest_feature_v2_snapshot()

    def test_periodic_reset_clears_all_three_state_layers(self):
        driver = self.make_driver(self.adapter)
        driver.prepare_shadow_action(make_frame(0, self.identity))
        self.assertIsNotNone(self.adapter.output_ids)
        self.assertIsNotNone(self.adapter.past_key_values)
        self.assertEqual(self.model.curr_t[0], 1)
        self.adapter.reset(
            {
                "event": "cache_reset",
                "env_id": 0,
                "next_step": 32,
                "terminal": False,
            }
        )
        self.assertIsNone(self.adapter.output_ids)
        self.assertIsNone(self.adapter.past_key_values)
        self.assertIsNone(self.adapter.output_ref)
        self.assertIsNone(self.adapter.past_key_values_ref)
        self.assertIsNone(self.adapter.output_audit)
        self.assertIsNone(self.adapter.inputs_embeds_seq_len)
        self.assertEqual(self.model.curr_t[0], 0)
        self.assertEqual(self.model.cache[0], {})

    def test_opt_in_feature_v2_latches_first_prefill_and_current_metric_depth(self):
        row = []
        row.extend([0.0] * 128)
        row.extend([0.0] * 64 + [0.2] * 64)
        row.extend([0.1 * index for index in range(1, 11)] + [0.0] * 118)
        row.extend([0.3] * 128)
        row.extend([0.4] * 64 + [float("nan")] * 32 + [1.5] * 32)
        self.assertEqual(len(row), OFFICIAL_RGB_WIDTH)
        depth_bytes = struct.pack(f"<{OFFICIAL_RGB_WIDTH}f", *row) * OFFICIAL_RGB_HEIGHT
        current_frame = make_frame(0, self.identity, depth_bytes=depth_bytes)
        direct_depth, direct_tensor_audit, direct_source_audit = (
            build_feature_v2_depth_summary(
                current_frame,
                torch_module=FeatureTorch,
                tensor_auditor=fake_tensor_audit,
            )
        )
        self.assertEqual(direct_depth.shape, (15,))
        self.assertEqual(direct_tensor_audit["shape"], [15])
        self.assertEqual(direct_source_audit["frame_ref"], current_frame.frame_ref)
        self.assertEqual(
            direct_source_audit["depth_sha256"], current_frame.depth_sha256
        )
        self.assertEqual(direct_source_audit["physical_step"], 0)

        model = FeatureV2FakeModel()
        tokenizer = FakeTokenizer(decoded_sequence="↑ ↑")
        adapter = self.make_adapter(
            model=model,
            tokenizer=tokenizer,
            torch_module=FeatureTorch,
            enable_feature_v2=True,
        )
        driver = self.make_driver(adapter)
        self.assertEqual(
            driver.prepare_shadow_action(
                current_frame
            ),
            1,
        )
        first = adapter.latest_feature_v2_snapshot()

        self.assertEqual(first.contextual_hidden.shape, (3584,))
        self.assertEqual(first.contextual_hidden.dtype, "torch.float32")
        self.assertEqual(first.contextual_hidden.device, "cpu")
        self.assertEqual(first.contextual_hidden.data, [3.0] * 3584)
        self.assertEqual(first.prefill_length, 3)
        self.assertEqual(first.physical_step, 0)
        self.assertEqual(first.reset_ordinal, 0)
        self.assertEqual(first.generation_ordinal, 0)
        self.assertEqual(
            first.frame_ref,
            current_frame.frame_ref,
        )
        self.assertEqual(first.depth_sha256, hashlib.sha256(depth_bytes).hexdigest())
        expected_depth = [
            0.0,
            0.0,
            0.0,
            2.0,
            2.0,
            0.5,
            1.0,
            5.0,
            10 / 128,
            3.0,
            3.0,
            1.0,
            4.0,
            4.0,
            0.5,
        ]
        for observed, expected in zip(first.depth_summary.data, expected_depth):
            self.assertAlmostEqual(observed, expected, places=5)
        self.assertEqual(first.depth_source_audit["frame_ref"], first.frame_ref)
        self.assertEqual(
            first.depth_source_audit["depth_sha256"], first.depth_sha256
        )
        self.assertEqual(first.depth_source_audit["physical_step"], 0)
        self.assertEqual(
            first.state_refs_audit["past_key_values_ref"],
            first.past_key_values_ref,
        )
        self.assertEqual(
            first.state_refs_audit["inputs_embeds_ref"], first.inputs_embeds_ref
        )
        self.assertFalse(first.state_refs_audit["copies_large_kv_payload"])
        self.assertEqual(
            adapter.generation_audits[0]["feature_v2"]["prefill_length"], 3
        )
        self.assertNotIn("output_hidden_states", model.calls[0])
        self.assertEqual(model.model.norm.call_count, 2)
        self.assertTrue(all(model.norm_outputs_unchanged))
        self.assertTrue(
            all(value is None for value in model.model.norm.hook_return_values)
        )
        self.assertEqual(model.model.norm._hooks, {})

        isolated = adapter.latest_feature_v2_snapshot()
        isolated.contextual_hidden.zero_()
        self.assertEqual(
            adapter.latest_feature_v2_snapshot().contextual_hidden.data,
            [3.0] * 3584,
        )

        driver.acknowledge_physical_step(terminated=False, truncated=False)
        self.assertEqual(
            driver.prepare_shadow_action(
                make_frame(1, self.identity, depth_bytes=depth_bytes)
            ),
            1,
        )
        queued = adapter.latest_feature_v2_snapshot()
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(queued.physical_step, 0)
        self.assertEqual(queued.generation_ordinal, 0)
        self.assertEqual(
            queued.contextual_hidden_audit, first.contextual_hidden_audit
        )

        driver.acknowledge_physical_step(terminated=False, truncated=False)
        model.fail_generation = True
        with self.assertRaisesRegex(
            RuntimeError, "synthetic feature-v2 generation failure"
        ):
            driver.prepare_shadow_action(
                make_frame(2, self.identity, depth_bytes=depth_bytes)
            )
        retained = adapter.latest_feature_v2_snapshot()
        self.assertEqual(retained.physical_step, 0)
        self.assertEqual(retained.generation_ordinal, 0)
        self.assertEqual(model.model.norm.register_count, 2)
        self.assertEqual(model.model.norm.remove_count, 2)
        self.assertEqual(model.model.norm.call_count, 3)
        self.assertEqual(model.model.norm._hooks, {})
        self.assertEqual(len(model.calls), 1)

        adapter.reset(
            {
                "event": "cache_reset",
                "env_id": 0,
                "next_step": 32,
                "terminal": False,
            }
        )
        with self.assertRaisesRegex(RuntimeError, "no feature-v2 snapshot"):
            adapter.latest_feature_v2_snapshot()

    def test_feature_v2_depth_summary_rejects_non_little_endian_host(self):
        with mock.patch.object(adapter_module.sys, "byteorder", "big"):
            with self.assertRaisesRegex(RuntimeError, "little-endian host"):
                build_feature_v2_depth_summary(
                    make_frame(7, self.identity),
                    torch_module=FeatureTorch,
                    tensor_auditor=fake_tensor_audit,
                )

    def test_crop_size_allows_unrelated_processor_metadata(self):
        processor = SimpleNamespace(
            crop_size={"height": 384, "width": 384, "shortest_edge": 384}
        )
        OfficialFramePreprocessor(
            image_processor=processor,
            device="cuda:0",
            torch_module=FakeTorch,
            numpy_module=object(),
            image_module=object(),
            to_numpy_array=lambda value: value,
            filter_depth=lambda value, blur_type=None: value,
        )
        processor.crop_size["height"] = 383
        with self.assertRaisesRegex(ValueError, "crop size drifted"):
            OfficialFramePreprocessor(
                image_processor=processor,
                device="cuda:0",
                torch_module=FakeTorch,
                numpy_module=object(),
                image_module=object(),
                to_numpy_array=lambda value: value,
                filter_depth=lambda value, blur_type=None: value,
            )

    def test_missing_model_curr_t_or_cache_fails_closed(self):
        for attribute in ("curr_t", "cache"):
            with self.subTest(attribute=attribute):
                model = FakeModel()
                delattr(model, attribute)
                with self.assertRaisesRegex(
                    RuntimeError, rf"{attribute} state is missing"
                ):
                    self.make_adapter(model=model)

    def test_only_floating_parameters_are_required_to_be_bfloat16(self):
        model = FakeModel(
            parameters=(
                FakeParameter(dtype="torch.bfloat16", floating=True),
                FakeParameter(
                    dtype="torch.int64", floating=False, device="cpu"
                ),
            )
        )
        self.make_adapter(model=model)
        bad = FakeModel(
            parameters=(FakeParameter(dtype="torch.float32", floating=True),)
        )
        with self.assertRaisesRegex(RuntimeError, "floating model parameter dtype"):
            self.make_adapter(model=bad)

    def test_prepared_tensor_shape_and_dtype_remain_strict(self):
        cases = (
            ("dtype", "images tensor dtype drifted"),
            ("shape", "depths tensor shape drifted"),
        )
        for kind, message in cases:
            with self.subTest(kind=kind):
                def bad_preprocess(frames, kind=kind):
                    prepared = self.preprocess(frames)
                    tensors = dict(prepared.tensors)
                    if kind == "dtype":
                        tensors["images"] = FakeTensor(
                            [], dtype="torch.float32",
                            shape=(1, len(frames), 3, 384, 384),
                            device="cuda:0",
                        )
                    else:
                        tensors["depths"] = FakeTensor(
                            [], dtype="torch.bfloat16",
                            shape=(1, len(frames), 383, 384),
                            device="cuda:0",
                        )
                    return PreparedOfficialFrames(
                        tensors=tensors,
                        frame_refs=prepared.frame_refs,
                    )

                adapter = self.make_adapter(preprocessor=bad_preprocess)
                driver = self.make_driver(adapter)
                with self.assertRaisesRegex(RuntimeError, message):
                    driver.prepare_shadow_action(make_frame(0, self.identity))

    def test_continuation_rejects_mutated_output_tensor_hash(self):
        driver = self.make_driver(self.adapter)
        driver.prepare_shadow_action(make_frame(0, self.identity))
        driver.acknowledge_physical_step(terminated=False, truncated=False)
        self.adapter.output_ids.data[0][0] = 999
        with self.assertRaisesRegex(RuntimeError, "tensor changed after generation"):
            driver.prepare_shadow_action(make_frame(1, self.identity))
        self.assertEqual(len(self.model.calls), 1)

    def test_di_device_resolution_stays_expected_only(self):
        for requested in ("cuda", FakeDevice("cuda"), 0):
            with self.subTest(requested=str(requested)):
                model = FakeModel(
                    parameters=(FakeParameter(device="cuda:0"),)
                )
                adapter = self.make_adapter(model=model, device=requested)
                self.assertEqual(str(adapter.device), "cuda:0")
                self.assertEqual(
                    adapter.runtime_receipt["verification_scope"],
                    "expected_contract_only",
                )
                self.assertIsNone(adapter.runtime_receipt["observed_runtime"])

        cpu_model = FakeModel(parameters=(FakeParameter(device="cpu"),))
        cpu_adapter = self.make_adapter(model=cpu_model, device="cpu")
        self.assertEqual(str(cpu_adapter.device), "cpu")
        self.assertEqual(
            cpu_adapter.runtime_receipt["verification_scope"],
            "expected_contract_only",
        )

        self.assertEqual(
            adapter_module._validate_explicit_factory_cuda("cuda:0", 0),
            "cuda:0",
        )
        self.assertEqual(
            adapter_module._validate_explicit_factory_cuda(
                FakeDevice("cuda:1"), 1
            ),
            "cuda:1",
        )
        for device, local_rank in (("cuda", 0), (0, 0), ("cuda:1", 0)):
            with self.subTest(factory_device=device, local_rank=local_rank):
                with self.assertRaisesRegex(
                    ValueError, "explicit cuda:N|device/local_rank mismatch"
                ):
                    adapter_module._validate_explicit_factory_cuda(
                        device, local_rank
                    )

        with self.assertRaisesRegex(RuntimeError, "parameter device drifted"):
            self.make_adapter(
                model=FakeModel(
                    parameters=(FakeParameter(device="cuda:0"),)
                ),
                device="cuda:1",
            )

    def test_dependency_injected_adapter_has_no_production_receipt_promotion_api(self):
        self.assertFalse(hasattr(adapter_module, "_RELEASE_FACTORY_TOKEN"))
        self.assertFalse(
            hasattr(self.adapter, "_bind_verified_release_receipt")
        )
        self.assertEqual(
            self.adapter.runtime_receipt["verification_scope"],
            "expected_contract_only",
        )
        with self.assertRaisesRegex(
            ValueError, "runtime is not pinned official StreamVLN inference"
        ):
            self.adapter.make_driver(
                session=OfficialEvalSession(
                    env_id=0,
                    instruction=self.instruction,
                    conjunction_rng=random.Random(3),
                ),
                episode_serial=0,
                episode_identity=self.identity,
            )

    def test_expected_only_adapter_cannot_create_production_driver(self):
        with self.assertRaisesRegex(
            ValueError, "runtime is not pinned official StreamVLN inference"
        ):
            self.adapter.make_driver(
                session=OfficialEvalSession(
                    env_id=0,
                    instruction=self.instruction,
                    conjunction_rng=random.Random(3),
                ),
                episode_serial=0,
                episode_identity=self.identity,
            )

    def test_inputs_embeds_must_grow_and_curr_t_must_advance_once(self):
        stagnant_model = FakeModel()
        stagnant_adapter = self.make_adapter(model=stagnant_model)
        stagnant_driver = self.make_driver(stagnant_adapter)
        stagnant_driver.prepare_shadow_action(make_frame(0, self.identity))
        stagnant_driver.acknowledge_physical_step(
            terminated=False, truncated=False
        )
        stagnant_model.inputs_embeds_growth = 0
        with self.assertRaisesRegex(
            RuntimeError, "inputs_embeds cache did not grow"
        ):
            stagnant_driver.prepare_shadow_action(make_frame(1, self.identity))

        skipped_model = FakeModel(curr_t_increment=2)
        skipped_adapter = self.make_adapter(model=skipped_model)
        with self.assertRaisesRegex(RuntimeError, "advance exactly once"):
            self.make_driver(skipped_adapter).prepare_shadow_action(
                make_frame(0, self.identity)
            )

        missing_model = FakeModel(omit_inputs_embeds=True)
        missing_adapter = self.make_adapter(model=missing_model)
        with self.assertRaisesRegex(RuntimeError, "inputs_embeds cache is missing"):
            self.make_driver(missing_adapter).prepare_shadow_action(
                make_frame(0, self.identity)
            )

    def test_candidate_manifest_hashes_source_and_every_artifact_byte(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source_root = self.make_source_root(base)
            roots = self.make_artifact_roots(base / "artifacts")
            manifest = build_full_content_artifact_manifest(
                source_root=source_root,
                model_path=roots["checkpoint"],
                tokenizer_path=roots["tokenizer"],
                vision_tower_path=roots["vision_tower"],
            )
            self.assertEqual(
                manifest["verification_scope"], "candidate_full_content"
            )
            observation = verify_full_content_artifact_manifest(
                manifest,
                source_root=source_root,
                model_path=roots["checkpoint"],
                tokenizer_path=roots["tokenizer"],
                vision_tower_path=roots["vision_tower"],
            )
            self.assertEqual(
                observation["verification_scope"], "candidate_full_content"
            )
            self.assertEqual(len(observation["manifest_sha256"]), 64)
            self.assertEqual(observation["source"]["python_file_count"], 3)
            self.assertEqual(
                set(observation["artifacts"]), set(PINNED_ARTIFACT_IDS)
            )
            for record in observation["artifacts"].values():
                self.assertEqual(record["file_count"], 2)
                self.assertEqual(len(record["content_manifest_sha256"]), 64)

            bad_manifest = copy.deepcopy(manifest)
            bad_manifest["artifacts"]["checkpoint"]["files"][0][
                "relative_path"
            ] = "../escape"
            with self.assertRaisesRegex(ValueError, "path is not canonical"):
                verify_full_content_artifact_manifest(
                    bad_manifest,
                    source_root=source_root,
                    model_path=roots["checkpoint"],
                    tokenizer_path=roots["tokenizer"],
                    vision_tower_path=roots["vision_tower"],
                )

            (roots["checkpoint"] / "config.json").write_bytes(b"mutated")
            with self.assertRaisesRegex(RuntimeError, "verification failed"):
                verify_full_content_artifact_manifest(
                    manifest,
                    source_root=source_root,
                    model_path=roots["checkpoint"],
                    tokenizer_path=roots["tokenizer"],
                    vision_tower_path=roots["vision_tower"],
                )

    def test_source_python_inventory_prunes_generated_pycache_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source_root = self.make_source_root(base)
            generated_cache = source_root / "streamvln" / "model" / "__pycache__"
            generated_cache.mkdir()
            (generated_cache / "generated.py").write_bytes(b"not-release-source")
            roots = self.make_artifact_roots(base / "artifacts")
            manifest = build_full_content_artifact_manifest(
                source_root=source_root,
                model_path=roots["checkpoint"],
                tokenizer_path=roots["tokenizer"],
                vision_tower_path=roots["vision_tower"],
            )
            source_paths = {
                entry["relative_path"]
                for entry in manifest["source"]["python_files"]
            }
            self.assertEqual(len(source_paths), 3)
            self.assertFalse(
                any("__pycache__" in relative_path for relative_path in source_paths)
            )
            observation = verify_full_content_artifact_manifest(
                manifest,
                source_root=source_root,
                model_path=roots["checkpoint"],
                tokenizer_path=roots["tokenizer"],
                vision_tower_path=roots["vision_tower"],
            )
            self.assertEqual(observation["source"]["python_file_count"], 3)

    def test_only_hf_snapshot_file_symlinks_to_own_blob_root_are_allowed(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source_root = self.make_source_root(base)
            roots = self.make_artifact_roots(base / "artifacts")
            tokenizer_root = roots["tokenizer"]
            blob = tokenizer_root.parent.parent / "blobs" / "tokenizer-blob"
            blob.write_bytes(b"tokenizer blob")
            link = tokenizer_root / "linked.bin"
            link_target = "../../blobs/tokenizer-blob"
            try:
                os.symlink(link_target, link)
            except (NotImplementedError, OSError):
                synthetic = {
                    "relative_path": "linked.bin",
                    "entry_type": "symlink",
                    "link_target": link_target,
                    "size": len(blob.read_bytes()),
                    "sha256": hashlib.sha256(blob.read_bytes()).hexdigest(),
                }
                normalized = adapter_module._normalize_file_manifest(
                    [synthetic],
                    name="tokenizer",
                    allow_file_symlinks=True,
                )
                self.assertEqual(normalized[0], synthetic)
                with self.assertRaisesRegex(ValueError, "entry type is invalid"):
                    adapter_module._normalize_file_manifest(
                        [synthetic],
                        name="checkpoint",
                        allow_file_symlinks=False,
                    )
                return

            manifest = build_full_content_artifact_manifest(
                source_root=source_root,
                model_path=roots["checkpoint"],
                tokenizer_path=roots["tokenizer"],
                vision_tower_path=roots["vision_tower"],
            )
            link_entry = next(
                entry
                for entry in manifest["artifacts"]["tokenizer"]["files"]
                if entry["relative_path"] == "linked.bin"
            )
            self.assertEqual(link_entry["entry_type"], "symlink")
            self.assertEqual(link_entry["link_target"], link_target)
            self.assertEqual(
                link_entry["sha256"], hashlib.sha256(blob.read_bytes()).hexdigest()
            )
            verify_full_content_artifact_manifest(
                manifest,
                source_root=source_root,
                model_path=roots["checkpoint"],
                tokenizer_path=roots["tokenizer"],
                vision_tower_path=roots["vision_tower"],
            )

            tampered = copy.deepcopy(manifest)
            tampered_entry = next(
                entry
                for entry in tampered["artifacts"]["tokenizer"]["files"]
                if entry["relative_path"] == "linked.bin"
            )
            tampered_entry["link_target"] = "../../blobs/different-blob"
            with self.assertRaisesRegex(RuntimeError, "verification failed"):
                verify_full_content_artifact_manifest(
                    tampered,
                    source_root=source_root,
                    model_path=roots["checkpoint"],
                    tokenizer_path=roots["tokenizer"],
                    vision_tower_path=roots["vision_tower"],
                )

            checkpoint_blob = roots["checkpoint"].parent / "checkpoint-blob.bin"
            checkpoint_blob.write_bytes(b"checkpoint blob")
            checkpoint_link = roots["checkpoint"] / "linked.bin"
            os.symlink("../checkpoint-blob.bin", checkpoint_link)
            with self.assertRaisesRegex(RuntimeError, "forbids this symlink file"):
                build_full_content_artifact_manifest(
                    source_root=source_root,
                    model_path=roots["checkpoint"],
                    tokenizer_path=roots["tokenizer"],
                    vision_tower_path=roots["vision_tower"],
                )
            checkpoint_link.unlink()

            outside_blob = base / "outside-tokenizer-blob.bin"
            outside_blob.write_bytes(b"outside")
            outside_link = tokenizer_root / "outside.bin"
            outside_target = os.path.relpath(outside_blob, tokenizer_root).replace(
                "\\", "/"
            )
            os.symlink(outside_target, outside_link)
            with self.assertRaisesRegex(RuntimeError, "escaped its trusted"):
                build_full_content_artifact_manifest(
                    source_root=source_root,
                    model_path=roots["checkpoint"],
                    tokenizer_path=roots["tokenizer"],
                    vision_tower_path=roots["vision_tower"],
                )
            outside_link.unlink()

            source_link = source_root / "streamvln" / "model" / "linked.py"
            os.symlink("helper.py", source_link)
            with self.assertRaisesRegex(
                RuntimeError, "forbids symlink or reparse-point files"
            ):
                build_full_content_artifact_manifest(
                    source_root=source_root,
                    model_path=roots["checkpoint"],
                    tokenizer_path=roots["tokenizer"],
                    vision_tower_path=roots["vision_tower"],
                )

    def test_windows_junction_or_reparse_root_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            actual = os.lstat(root)
            reparse = SimpleNamespace(
                st_mode=actual.st_mode,
                st_file_attributes=adapter_module._WINDOWS_REPARSE_POINT,
            )
            with mock.patch.object(adapter_module.os, "lstat", return_value=reparse):
                with self.assertRaisesRegex(RuntimeError, "reparse points"):
                    adapter_module._ordinary_directory_root(
                        root, name="test root"
                    )
                with self.assertRaisesRegex(RuntimeError, "reparse points"):
                    adapter_module._ordinary_directory_stat(
                        root, name="test root"
                    )

    def test_factory_rejects_any_nonzero_or_multi_model_environment_before_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source_root = self.make_source_root(base)
            roots = self.make_artifact_roots(base / "artifacts")
            manifest = build_full_content_artifact_manifest(
                source_root=source_root,
                model_path=roots["checkpoint"],
                tokenizer_path=roots["tokenizer"],
                vision_tower_path=roots["vision_tower"],
            )
            for env_count, model_env_id in ((2, 0), (1, 1), (True, 0), (1, False)):
                with self.subTest(
                    env_count=env_count, model_env_id=model_env_id
                ), mock.patch.object(
                    adapter_module.importlib,
                    "import_module",
                    side_effect=AssertionError("heavy import happened too early"),
                ) as import_mock:
                    with self.assertRaisesRegex(
                        ValueError, "env_count=1 and model_env_id=0"
                    ):
                        PinnedStreamVLNGenerationAdapter.from_released_checkpoint(
                            model_path=str(roots["checkpoint"]),
                            tokenizer_path=str(roots["tokenizer"]),
                            vision_tower_path=str(roots["vision_tower"]),
                            source_root=str(source_root),
                            artifact_manifest=manifest,
                            device="cuda:0",
                            local_rank=0,
                            env_count=env_count,
                            model_env_id=model_env_id,
                        )
                    import_mock.assert_not_called()

    def test_pending_or_unpinned_candidate_fails_before_heavy_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source_root = self.make_source_root(base)
            roots = self.make_artifact_roots(base / "artifacts")
            manifest = build_full_content_artifact_manifest(
                source_root=source_root,
                model_path=roots["checkpoint"],
                tokenizer_path=roots["tokenizer"],
                vision_tower_path=roots["vision_tower"],
            )
            with mock.patch.object(
                adapter_module, "PINNED_RELEASE_MANIFEST_SHA256", None
            ), mock.patch.object(
                adapter_module.importlib,
                "import_module",
                side_effect=AssertionError("heavy import happened too early"),
            ) as import_mock:
                with self.assertRaisesRegex(RuntimeError, "PENDING"):
                    PinnedStreamVLNGenerationAdapter.from_released_checkpoint(
                        model_path=str(roots["checkpoint"]),
                        tokenizer_path=str(roots["tokenizer"]),
                        vision_tower_path=str(roots["vision_tower"]),
                        source_root=str(source_root),
                        artifact_manifest=manifest,
                        device="cuda:0",
                        local_rank=0,
                        env_count=1,
                        model_env_id=0,
                    )
                import_mock.assert_not_called()

            with mock.patch.object(
                adapter_module.importlib,
                "import_module",
                side_effect=AssertionError("heavy import happened too early"),
            ) as import_mock:
                with self.assertRaisesRegex(
                    RuntimeError, "does not match pinned release"
                ):
                    PinnedStreamVLNGenerationAdapter.from_released_checkpoint(
                        model_path=str(roots["checkpoint"]),
                        tokenizer_path=str(roots["tokenizer"]),
                        vision_tower_path=str(roots["vision_tower"]),
                        source_root=str(source_root),
                        artifact_manifest=manifest,
                        device="cuda:0",
                        local_rank=0,
                        env_count=1,
                        model_env_id=0,
                    )
                import_mock.assert_not_called()

    def test_imported_model_source_and_class_identity_are_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            source_root = Path(temporary)
            model_source = source_root / PINNED_MODEL_SOURCE_RELATIVE_PATH
            model_source.parent.mkdir(parents=True)
            model_source.write_bytes(b"released-model-source")
            digest = hashlib.sha256(model_source.read_bytes()).hexdigest()
            model_module = SimpleNamespace(__file__=str(model_source))

            class PinnedModel:
                pass

            PinnedModel.__module__ = "model.stream_video_vln"
            with mock.patch.object(
                adapter_module, "PINNED_MODEL_CODE_SHA256", digest
            ):
                observation = adapter_module._verify_imported_model_source(
                    model_module, PinnedModel, source_root
                )
                self.assertEqual(
                    observation["imported_model_module_relative_path"],
                    PINNED_MODEL_SOURCE_RELATIVE_PATH,
                )
                self.assertEqual(
                    observation["imported_model_module_sha256"], digest
                )
                PinnedModel.__module__ = "other.module"
                with self.assertRaisesRegex(RuntimeError, "class module drifted"):
                    adapter_module._verify_imported_model_source(
                        model_module, PinnedModel, source_root
                    )

    def test_clean_repository_import_root_is_installed_first(self):
        with tempfile.TemporaryDirectory() as temporary:
            source_root = Path(temporary)
            import_root = source_root / PINNED_SOURCE_IMPORT_ROOT_RELATIVE_PATH
            import_root.mkdir()
            original_path = list(sys.path)
            try:
                installed = adapter_module._install_source_import_root(source_root)
                self.assertEqual(installed, import_root.resolve())
                self.assertEqual(sys.path[0], str(import_root.resolve()))
                self.assertEqual(sys.path[1], str(source_root.resolve()))
            finally:
                sys.path[:] = original_path
    def test_runtime_receipt_is_bound_to_pinned_e48_sources(self):
        receipt = self.adapter.runtime_receipt
        expected_evaluator_sha256 = (
            "4c556b77100ed60cc40aee620947c89428494562402f7d3c5b5120ae12320abb"
        )
        expected_model_code_sha256 = (
            "e106e16a7de4a8086ed526e55d877506e480d05980ddf1c6909286439acca8dc"
        )
        expected_generation_kwargs = {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": 10000,
            "use_cache": True,
            "return_dict_in_generate": True,
        }
        expected_decode_kwargs = {"skip_special_tokens": False}
        self.assertEqual(
            receipt["source_commit"],
            "e48f6ff7e9201d93aae003e8f64b04c00cec13bc",
        )
        self.assertEqual(receipt["source_commit"], PINNED_STREAMVLN_COMMIT)
        self.assertEqual(PINNED_EVALUATOR_SHA256, expected_evaluator_sha256)
        self.assertEqual(PINNED_MODEL_CODE_SHA256, expected_model_code_sha256)
        self.assertEqual(receipt["evaluator_sha256"], PINNED_EVALUATOR_SHA256)
        self.assertEqual(receipt["model_code_sha256"], PINNED_MODEL_CODE_SHA256)
        self.assertTrue(receipt["full_vocabulary_generation"])
        self.assertEqual(receipt["sampler"], "greedy")
        self.assertEqual(GENERATION_KWARGS, expected_generation_kwargs)
        self.assertEqual(DECODE_KWARGS, expected_decode_kwargs)
        self.assertEqual(receipt["generation_kwargs"], expected_generation_kwargs)
        self.assertEqual(receipt["decode_kwargs"], expected_decode_kwargs)
        self.assertEqual(receipt["verification_scope"], "expected_contract_only")
        self.assertIsNone(receipt["observed_runtime"])

    def test_runtime_receipt_binds_tokenizer_frame_and_model_cache_contracts(self):
        receipt = self.adapter.runtime_receipt
        self.assertEqual(
            receipt["tokenizer_preprocess_contract"]["image_token_index"], -200
        )
        self.assertIn("filter_depth", receipt["frame_preprocess_contract"]["depth"])
        self.assertIn(
            "inputs_embeds", receipt["model_internal_state_contract"]["model_cache"]
        )


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "Torch unavailable")
class FrozenFeatureContractTests(unittest.TestCase):
    def _adapter(self, *, mutate_state=False):
        import torch

        class Inner(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = torch.nn.Embedding(16, 4)
                with torch.no_grad():
                    self.embed_tokens.weight.copy_(
                        torch.arange(64, dtype=torch.float32).reshape(16, 4) + 1
                    )

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = Inner()
                self.config = SimpleNamespace(hidden_size=4)
                self.curr_t = [2]
                self.cache = [{"inputs_embeds": torch.ones(1, 3, 4)}]
                self.generate_calls = 0

            def get_model(self):
                return self.inner

            def encode_images(self, pixels):
                if mutate_state:
                    self.curr_t[0] += 1
                scale = pixels.float().mean() + 1.0
                return torch.tensor(
                    [[[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]]]
                ) * scale

            def get_2dPool(self, features, stride):
                self.assert_stride = stride
                return features

            def generate(self, **_kwargs):
                self.generate_calls += 1
                raise AssertionError("frozen feature extraction must not call generate")

        class Tokenizer:
            @staticmethod
            def encode(text, add_special_tokens):
                if add_special_tokens or not text:
                    raise AssertionError("instruction tokenization contract drifted")
                return [1, 2, 3]

        def preprocess(frames):
            count = len(frames)
            return PreparedOfficialFrames(
                tensors={
                    "images": torch.ones(1, count, 3, 384, 384, dtype=torch.bfloat16),
                    "depths": torch.ones(1, count, 384, 384, dtype=torch.bfloat16),
                    "poses": torch.eye(4, dtype=torch.float32)
                    .to(dtype=torch.bfloat16)
                    .reshape(1, 1, 4, 4),
                    "intrinsics": torch.eye(4, dtype=torch.float32)
                    .to(dtype=torch.bfloat16)
                    .reshape(1, 1, 4, 4),
                },
                frame_refs=tuple(frame.frame_ref for frame in frames),
            )

        value = object.__new__(PinnedStreamVLNGenerationAdapter)
        value.model = Model()
        value.tokenizer = Tokenizer()
        value.device = torch.device("cpu")
        value.env_id = 0
        value.torch = torch
        value.frame_preprocessor = preprocess
        value.tensor_auditor = lambda tensor: adapter_module._tensor_audit(tensor, torch)
        value.output_ids = torch.tensor([[7]], dtype=torch.int64)
        value.past_key_values = (torch.ones(1, 1, 2, 4),)
        value.output_ref = "a" * 64
        value.past_key_values_ref = "b" * 64
        value.output_audit = value.tensor_auditor(value.output_ids)
        value.inputs_embeds_seq_len = 3
        value.reset_ordinal = 0
        value.generation_ordinal = 2
        value._last_frozen_feature_snapshot = None
        return value

    def test_side_effect_free_fp32_cpu_l2_feature_and_snapshot(self):
        import torch

        instruction = "walk to the chair"
        identity = {
            "episode_id": "episode-0",
            "scene_id": "scene-0",
            "instruction_sha256": hashlib.sha256(instruction.encode()).hexdigest(),
        }
        frame = make_frame(4, identity)
        adapter = self._adapter()
        before = adapter._frozen_model_state_signature()
        first = adapter.extract_frozen_feature_snapshot(
            frame=frame, instruction=instruction
        )
        second = adapter.extract_frozen_feature_snapshot(
            frame=frame, instruction=instruction
        )
        self.assertEqual(adapter._frozen_model_state_signature(), before)
        self.assertEqual(adapter.model.generate_calls, 0)
        self.assertEqual(first.feature.shape, (8,))
        self.assertEqual(first.feature.dtype, torch.float32)
        self.assertEqual(first.feature.device.type, "cpu")
        self.assertTrue(torch.equal(first.feature, second.feature))
        self.assertAlmostEqual(float(first.feature[:4].norm()), 1.0, places=6)
        self.assertAlmostEqual(float(first.feature[4:].norm()), 1.0, places=6)
        clone = adapter.latest_frozen_feature_snapshot()
        clone.feature.zero_()
        self.assertFalse(torch.equal(clone.feature, first.feature))

    def test_feature_extraction_detects_generation_state_mutation(self):
        instruction = "walk to the chair"
        frame = make_frame(
            4,
            {
                "episode_id": "episode-0",
                "scene_id": "scene-0",
                "instruction_sha256": hashlib.sha256(instruction.encode()).hexdigest(),
            },
        )
        with self.assertRaisesRegex(RuntimeError, "mutated official generation state"):
            self._adapter(mutate_state=True).extract_frozen_feature_snapshot(
                frame=frame, instruction=instruction
            )


if __name__ == "__main__":
    unittest.main()
