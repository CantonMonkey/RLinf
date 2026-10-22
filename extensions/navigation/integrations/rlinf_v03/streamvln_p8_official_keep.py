"""Pinned official StreamVLN inference semantics for the P8 KEEP branch.

This module is intentionally model-free.  It freezes the observable contract of
``streamvln/streamvln_eval.py`` at the released source revision so that the
live adapter can be checked before loading an 8B checkpoint.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
import copy
from dataclasses import dataclass
import hashlib
import json
import re
import random
from typing import Iterable


PINNED_STREAMVLN_COMMIT = "e48f6ff7e9201d93aae003e8f64b04c00cec13bc"
PINNED_SOURCE_ROOT_KIND = "repository_root"
PINNED_SOURCE_IMPORT_ROOT_RELATIVE_PATH = "streamvln"
PINNED_EVALUATOR_SOURCE_RELATIVE_PATH = "streamvln/streamvln_eval.py"
PINNED_MODEL_SOURCE_RELATIVE_PATH = "streamvln/model/stream_video_vln.py"
PINNED_MODEL_CLASS_MODULE = "model.stream_video_vln"
PINNED_EVALUATOR_SHA256 = "4c556b77100ed60cc40aee620947c89428494562402f7d3c5b5120ae12320abb"
PINNED_MODEL_CODE_SHA256 = "e106e16a7de4a8086ed526e55d877506e480d05980ddf1c6909286439acca8dc"
PINNED_ARTIFACT_IDS = {
    "checkpoint": (
        "StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3-"
        "f1f76c66083c"
    ),
    "tokenizer": "013210b3aff822f1558b166d39c1046dd109520f",
    "vision_tower": "9fdffc58afc957d1a03a25b10dba0329ab15c2a3",
}
# Frozen from the independently inventoried clean-e48 source and original
# checkpoint/tokenizer/vision trees. Candidate manifests are never accepted.
PINNED_RELEASE_MANIFEST_SHA256 = "09a344903944942210a342f5defe8708523ecd74be98ec6be9b896334da471ac"
RELEASE_MANIFEST_SCHEMA_NAME = "streamvln_p8_full_content_artifacts"
RELEASE_MANIFEST_SCHEMA_VERSION = 3
PRODUCTION_RUNTIME_VALIDATION_LEVEL = "production_verified_release"
TEST_ONLY_CONTRACT_VALIDATION_LEVEL = "test_only_contract"
ATTENTION_IMPLEMENTATION = "flash_attention_2"
MODEL_DTYPE = "torch.bfloat16"
PINNED_TRANSFORMERS_VERSION = "4.45.1"
PINNED_TORCH_VERSION = "2.1.2"
NUM_FUTURE_STEPS = 4
NUM_FRAMES = 32
NUM_HISTORY = 8
MODEL_MAX_LENGTH = 4096
SYSTEM_MESSAGE = "You are a helpful assistant."
APPLY_CHAT_TEMPLATE_ADD_GENERATION_PROMPT = False
SOURCE_MESSAGE_ROLES = ("human", "gpt")
RENDERED_MESSAGE_ROLES = ("user", "assistant")
EMPTY_ASSISTANT_CONTENT = ""
CHAT_TEMPLATE_APPLICATION = "separate_system_then_each_single_message"
IMAGE_TOKEN = "<image>"
MEMORY_TOKEN = "<memory>"
IMAGE_TOKEN_INDEX = -200
MEMORY_TOKEN_INDEX = -300

TOKENIZER_PREPROCESS_CONTRACT = {
    "copy": "copy.deepcopy(tokenizer)_per_generation",
    "added_special_tokens_in_order": [IMAGE_TOKEN, MEMORY_TOKEN],
    "image_token_index": IMAGE_TOKEN_INDEX,
    "memory_token_index": MEMORY_TOKEN_INDEX,
    "chat_template_application": CHAT_TEMPLATE_APPLICATION,
    "apply_generation_prompt": APPLY_CHAT_TEMPLATE_ADD_GENERATION_PROMPT,
    "message_order": ["optional_system", "user", "empty_assistant"],
    "negative_index_replacement_after_chat_template": True,
    "previous_output_concat": "torch.cat([output_ids,new_input_ids],dim=1)",
}

FRAME_PREPROCESS_CONTRACT = {
    "rgb": "PIL_RGB_then_checkpoint_image_processor_preprocess_pixel_values_0",
    "depth": (
        "normalized_f32_filter_depth_none_times_10m_times_1000_to_uint16_"
        "PIL_I16_nearest_384x384_divide_1000"
    ),
    "pose": "camera_to_episodic_times_axis_align_then_bfloat16",
    "intrinsics": "raw_640x480_scale_rows_to_384x384_float32_then_bfloat16",
    "stack": "per_modality_stack_then_unsqueeze_batch",
}

MODEL_INTERNAL_STATE_CONTRACT = {
    "output_ids": "replace_with_outputs.sequences_and_concat_on_continuation",
    "past_key_values": "replace_with_outputs.past_key_values_and_reuse",
    "model_cache": "model.cache[env_id].inputs_embeds_accumulates_each_generation",
    "model_curr_t": "model.curr_t[env_id]_increments_each_generation",
    "output_ids_audit": "full_content_sha256_rechecked_before_continuation",
    "past_key_values_audit": "structure_only_no_content_hash",
    "inputs_embeds_audit": (
        "shape_dtype_device_and_strictly_increasing_sequence_length"
    ),
    "reset": (
        "model.reset_for_env(env_id)_plus_clear_output_ids_and_past_key_values_"
        "after_every_32_physical_steps_including_terminal_boundary"
    ),
}

STATE_AUDIT_SCOPE = {
    "output_ids": "full_content_sha256",
    "past_key_values": "structure_only",
    "inputs_embeds": "shape_dtype_device_and_temporal_sequence_length",
    "deep_kv_content_hash_in_steady_state": False,
}

MODEL_INPUT_CONTRACT = {
    "images": {"shape": "[1,V,3,H,W]", "dtype": "torch.bfloat16"},
    "depths": {"shape": "[1,V,H,W]", "dtype": "torch.bfloat16"},
    "poses": {"shape": "[1,V,4,4]", "dtype": "torch.bfloat16"},
    "intrinsics": {"shape": "[1,V,4,4]", "dtype": "torch.bfloat16"},
    "inputs": {"shape": "[1,L]", "dtype": "torch.int64"},
    "env_id": {"shape": "scalar", "dtype": "int"},
    "time_ids": {"shape": "[list[int]]", "dtype": "python"},
    "task_type": {"value": [0], "dtype": "python"},
    "past_key_values": {"shape": "opaque_upstream_cache", "dtype": "upstream"},
}

CONJUNCTIONS = (
    "you can see ",
    "in front of you is ",
    "there is ",
    "you can spot ",
    "you are toward the ",
    "ahead of you is ",
    "in your sight is ",
)
CHAT_TEMPLATE = (
    "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + "
    "message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)

STOP_ACTION = 0
FORWARD_ACTION = 1
LEFT_ACTION = 2
RIGHT_ACTION = 3

ACTION_MARKERS = (
    ("STOP", STOP_ACTION),
    ("↑", FORWARD_ACTION),
    ("←", LEFT_ACTION),
    ("→", RIGHT_ACTION),
)

BASE_PROMPT = (
    "<video>\nYou are an autonomous navigation assistant. Your task is to "
    "<instruction>. Devise an action sequence to follow the instruction using "
    "the four actions: TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, MOVE "
    "FORWARD (↑) by 25 centimeters, or STOP."
)
HISTORY_SUFFIX = " These are your historical observations <memory>."

GENERATION_KWARGS = {
    "do_sample": False,
    "num_beams": 1,
    "max_new_tokens": 10000,
    "use_cache": True,
    "return_dict_in_generate": True,
}
DECODE_KWARGS = {"skip_special_tokens": False}


def official_prompt(instruction: str, *, step_id: int) -> str:
    """Reproduce the prompt text constructed by the pinned evaluator."""

    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("official StreamVLN instruction must be non-empty")
    if type(step_id) is not int or step_id < 0:
        raise ValueError("official StreamVLN step_id must be a nonnegative integer")
    prompt = BASE_PROMPT.replace("<instruction>.", instruction)
    if step_id:
        prompt += HISTORY_SUFFIX
    return prompt.replace("<video>\n", "")


def official_user_content(
    instruction: str,
    *,
    step_id: int,
    conjunction: str,
) -> str:
    """Include the exact random conjunction and image token added in preprocessing."""

    if conjunction not in CONJUNCTIONS:
        raise ValueError("conjunction is outside the pinned upstream choice list")
    return official_prompt(instruction, step_id=step_id) + f" {conjunction}<image>."


def official_history_frame_indices(step_id: int) -> tuple[int, ...]:
    """Return historical frame indices added at the pinned 32-step boundary."""

    if type(step_id) is not int or step_id < 0:
        raise ValueError("step_id must be a nonnegative integer")
    if step_id == 0 or step_id % NUM_FRAMES:
        return ()
    stride = step_id // NUM_HISTORY
    if stride < 1:
        raise RuntimeError("official history stride became zero")
    return tuple(range(0, step_id, stride))


def parse_official_actions(decoded_sequence: str) -> list[int]:
    """Mirror the evaluator's ordered regex scan over the decoded sequence."""

    if not isinstance(decoded_sequence, str):
        raise TypeError("decoded StreamVLN sequence must be text")
    pattern = "|".join(re.escape(marker) for marker, _ in ACTION_MARKERS)
    marker_to_action = dict(ACTION_MARKERS)
    return [marker_to_action[match] for match in re.findall(pattern, decoded_sequence)]


def parse_with_official_empty_fallback(decoded_sequence: str) -> list[int]:
    actions = parse_official_actions(decoded_sequence)
    return actions if actions else [STOP_ACTION]


def official_keep_contract() -> dict:
    """Return a JSON-stable receipt that a live KEEP adapter must match."""

    return {
        "source_commit": PINNED_STREAMVLN_COMMIT,
        "source_file": PINNED_EVALUATOR_SOURCE_RELATIVE_PATH,
        "model_source_file": PINNED_MODEL_SOURCE_RELATIVE_PATH,
        "source_import_root_relative_path": PINNED_SOURCE_IMPORT_ROOT_RELATIVE_PATH,
        "evaluator_sha256": PINNED_EVALUATOR_SHA256,
        "model_code_sha256": PINNED_MODEL_CODE_SHA256,
        "source_provenance": {
            "kind": "clean_upstream_git_checkout",
            "commit": PINNED_STREAMVLN_COMMIT,
            "source_root_kind": PINNED_SOURCE_ROOT_KIND,
            "import_root_relative_path": PINNED_SOURCE_IMPORT_ROOT_RELATIVE_PATH,
            "evaluator_relative_path": PINNED_EVALUATOR_SOURCE_RELATIVE_PATH,
            "model_relative_path": PINNED_MODEL_SOURCE_RELATIVE_PATH,
        },
        "artifact_ids": PINNED_ARTIFACT_IDS,
        "pinned_release_manifest_sha256": PINNED_RELEASE_MANIFEST_SHA256,
        "release_manifest_schema": {
            "name": RELEASE_MANIFEST_SCHEMA_NAME,
            "version": RELEASE_MANIFEST_SCHEMA_VERSION,
        },
        "attention_implementation": ATTENTION_IMPLEMENTATION,
        "model_dtype": MODEL_DTYPE,
        "generation_kwargs": dict(GENERATION_KWARGS),
        "decode_kwargs": dict(DECODE_KWARGS),
        "decoded_object": "output_ids_assigned_from_outputs.sequences_then_batch_decode",
        "action_markers": [[marker, action] for marker, action in ACTION_MARKERS],
        "empty_generation_fallback": [STOP_ACTION],
        "parsed_action_queue_bound": "max_new_tokens_10000_no_fixed_four_cap",
        "queue_semantics": "generate_only_when_action_queue_is_empty_then_pop_one_primitive",
        "history": {
            "num_future_steps": NUM_FUTURE_STEPS,
            "num_frames": NUM_FRAMES,
            "num_history": NUM_HISTORY,
            "model_max_length": MODEL_MAX_LENGTH,
            "reset_cache_every_num_frames": True,
            "history_indices_at_boundary": "range(0, step_id, step_id // num_history)",
            "always_append_current_frame": True,
        },
        "prompt_template": BASE_PROMPT,
        "instruction_replacement": "replace('<instruction>.', instruction_text)",
        "history_suffix": HISTORY_SUFFIX,
        "conjunctions": list(CONJUNCTIONS),
        "conjunction_selection": "python_random.choice_unseeded_by_streamvln_eval",
        "chat_template": CHAT_TEMPLATE,
        "system_message": SYSTEM_MESSAGE,
        "source_message_roles": list(SOURCE_MESSAGE_ROLES),
        "rendered_message_roles": list(RENDERED_MESSAGE_ROLES),
        "empty_assistant_content": EMPTY_ASSISTANT_CONTENT,
        "apply_chat_template_add_generation_prompt": (
            APPLY_CHAT_TEMPLATE_ADD_GENERATION_PROMPT
        ),
        "chat_template_application": CHAT_TEMPLATE_APPLICATION,
        "model_inputs": [
            "images",
            "depths",
            "poses",
            "intrinsics",
            "inputs",
            "env_id",
            "time_ids",
            "task_type",
            "past_key_values",
        ],
        "model_input_contract": MODEL_INPUT_CONTRACT,
        "tokenizer_preprocess_contract": TOKENIZER_PREPROCESS_CONTRACT,
        "frame_preprocess_contract": FRAME_PREPROCESS_CONTRACT,
        "model_internal_state_contract": MODEL_INTERNAL_STATE_CONTRACT,
        "state_audit_scope": STATE_AUDIT_SCOPE,
    }


def official_keep_contract_sha256() -> str:
    payload = json.dumps(
        official_keep_contract(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass
class OfficialKeepQueue:
    """Faithful action queue boundary around one upstream greedy generation."""

    _pending: deque[int]

    def __init__(self) -> None:
        self._pending = deque()

    @property
    def needs_generation(self) -> bool:
        return not self._pending

    @property
    def pending_actions(self) -> tuple[int, ...]:
        return tuple(self._pending)

    def load_decoded_sequence(self, decoded_sequence: str) -> tuple[int, ...]:
        if self._pending:
            raise RuntimeError("official KEEP generation cannot replace a non-empty action queue")
        self._pending.extend(parse_with_official_empty_fallback(decoded_sequence))
        return self.pending_actions

    def load_actions_for_test(self, actions: Iterable[int]) -> tuple[int, ...]:
        """Inject already-parsed actions for deterministic no-model contract tests."""

        if self._pending:
            raise RuntimeError("official KEEP action queue is already populated")
        values = tuple(actions)
        if not values or any(
            type(action) is not int or action not in (0, 1, 2, 3)
            for action in values
        ):
            raise ValueError("official KEEP actions must be a non-empty primitive sequence")
        self._pending.extend(values)
        return self.pending_actions

    def pop_action(self) -> int:
        if not self._pending:
            raise RuntimeError("official KEEP requires a generation before the next action")
        return int(self._pending.popleft())


@dataclass(frozen=True)
class OfficialGenerationRequest:
    physical_step: int
    env_id: int
    user_content: str
    conjunction: str
    add_system: bool
    frame_steps: tuple[int, ...]
    time_ids: tuple[int, ...]
    previous_output_ref: str | None
    past_key_values_ref: str | None
    contract_sha256: str
    task_type: tuple[int, ...] = (0,)
    system_message: str = SYSTEM_MESSAGE
    source_messages: tuple[tuple[str, str], ...] = ()
    rendered_messages: tuple[tuple[str, str], ...] = ()
    add_generation_prompt: bool = APPLY_CHAT_TEMPLATE_ADD_GENERATION_PROMPT
    chat_template_application: str = CHAT_TEMPLATE_APPLICATION

    def __post_init__(self) -> None:
        if self.contract_sha256 != official_keep_contract_sha256():
            raise ValueError("official generation request contract hash drifted")
        if self.task_type != (0,):
            raise ValueError("official StreamVLN task_type must be (0,)")
        if self.system_message != SYSTEM_MESSAGE:
            raise ValueError("official StreamVLN system message drifted")
        expected_source = (
            (SOURCE_MESSAGE_ROLES[0], self.user_content),
            (SOURCE_MESSAGE_ROLES[1], EMPTY_ASSISTANT_CONTENT),
        )
        expected_rendered = (
            (RENDERED_MESSAGE_ROLES[0], self.user_content),
            (RENDERED_MESSAGE_ROLES[1], EMPTY_ASSISTANT_CONTENT),
        )
        if self.source_messages != expected_source:
            raise ValueError("official source message sequence drifted")
        if self.rendered_messages != expected_rendered:
            raise ValueError("official rendered message sequence drifted")
        if self.add_generation_prompt is not APPLY_CHAT_TEMPLATE_ADD_GENERATION_PROMPT:
            raise ValueError("official chat-template generation-prompt flag drifted")
        if self.chat_template_application != CHAT_TEMPLATE_APPLICATION:
            raise ValueError("official chat-template application order drifted")

    def to_dict(self) -> dict:
        return {
            "physical_step": self.physical_step,
            "env_id": self.env_id,
            "user_content": self.user_content,
            "conjunction": self.conjunction,
            "add_system": self.add_system,
            "frame_steps": list(self.frame_steps),
            "time_ids": list(self.time_ids),
            "previous_output_ref": self.previous_output_ref,
            "past_key_values_ref": self.past_key_values_ref,
            "contract_sha256": self.contract_sha256,
            "task_type": list(self.task_type),
            "system_message": self.system_message,
            "source_messages": [
                {"from": role, "value": content}
                for role, content in self.source_messages
            ],
            "rendered_messages": [
                {"role": role, "content": content}
                for role, content in self.rendered_messages
            ],
            "add_generation_prompt": self.add_generation_prompt,
            "chat_template_application": self.chat_template_application,
            "model_input_contract": MODEL_INPUT_CONTRACT,
            "tokenizer_preprocess_contract": TOKENIZER_PREPROCESS_CONTRACT,
            "frame_preprocess_contract": FRAME_PREPROCESS_CONTRACT,
            "model_internal_state_contract": MODEL_INTERNAL_STATE_CONTRACT,
        }


class OfficialEvalSession:
    """Model-independent state machine matching the pinned evaluator cadence."""

    def __init__(
        self,
        *,
        env_id: int,
        instruction: str,
        conjunction_rng: random.Random,
    ) -> None:
        if type(env_id) is not int or env_id < 0:
            raise ValueError("env_id must be a nonnegative integer")
        if not isinstance(conjunction_rng, random.Random):
            raise TypeError("official parity requires an explicit replayable Python RNG")
        official_prompt(instruction, step_id=0)
        self.env_id = env_id
        self.instruction = instruction
        self.conjunction_rng = conjunction_rng
        self.physical_step = 0
        self.queue = OfficialKeepQueue()
        self.frame_refs: dict[int, str] = {}
        self.time_ids: list[int] = []
        self.previous_output_ref: str | None = None
        self.past_key_values_ref: str | None = None
        self.pending_request: OfficialGenerationRequest | None = None
        self.action_popped_for_step = False
        self.popped_action_for_step: int | None = None
        self.terminal = False
        self.terminal_reason: str | None = None
        self.reset_steps: list[int] = []
        self.queue_crossed_reset = False

    @property
    def needs_generation(self) -> bool:
        return self.queue.needs_generation

    def observe(self, frame_ref: str) -> None:
        if self.terminal:
            raise RuntimeError("official session is terminal")
        if not isinstance(frame_ref, str) or not frame_ref:
            raise ValueError("frame_ref must be a non-empty provenance handle")
        if self.physical_step in self.frame_refs:
            raise RuntimeError("official session already observed this physical step")
        self.frame_refs[self.physical_step] = frame_ref
        self.time_ids.append(self.physical_step)

    def build_generation_request(self) -> OfficialGenerationRequest:
        if self.terminal:
            raise RuntimeError("official session is terminal")
        if not self.needs_generation:
            raise RuntimeError("official evaluator generates only when its action queue is empty")
        if self.pending_request is not None:
            raise RuntimeError("official generation request is already pending")
        if self.action_popped_for_step:
            raise RuntimeError("official action already popped for this physical step")
        if self.physical_step not in self.frame_refs:
            raise RuntimeError("current physical frame must be observed before generation")
        conjunction = self.conjunction_rng.choice(CONJUNCTIONS)
        continuation = self.previous_output_ref is not None
        if continuation:
            user_content = f"{conjunction}<image>."
        else:
            user_content = official_user_content(
                self.instruction,
                step_id=self.physical_step,
                conjunction=conjunction,
            )
        history = official_history_frame_indices(self.physical_step)
        frame_steps = history + (self.physical_step,)
        missing = [step for step in frame_steps if step not in self.frame_refs]
        if missing:
            raise RuntimeError(f"official history frame provenance is missing: {missing}")
        request = OfficialGenerationRequest(
            physical_step=self.physical_step,
            env_id=self.env_id,
            user_content=user_content,
            conjunction=conjunction,
            add_system=not continuation,
            frame_steps=frame_steps,
            time_ids=tuple(self.time_ids),
            previous_output_ref=self.previous_output_ref,
            past_key_values_ref=self.past_key_values_ref,
            contract_sha256=official_keep_contract_sha256(),
            source_messages=(
                (SOURCE_MESSAGE_ROLES[0], user_content),
                (SOURCE_MESSAGE_ROLES[1], EMPTY_ASSISTANT_CONTENT),
            ),
            rendered_messages=(
                (RENDERED_MESSAGE_ROLES[0], user_content),
                (RENDERED_MESSAGE_ROLES[1], EMPTY_ASSISTANT_CONTENT),
            ),
        )
        self.pending_request = request
        return request

    def accept_generation(
        self,
        *,
        decoded_sequence: str,
        output_ref: str,
        past_key_values_ref: str,
    ) -> tuple[int, ...]:
        if self.terminal:
            raise RuntimeError("official session is terminal")
        if self.pending_request is None:
            raise RuntimeError("official generation output has no matching request")
        if not isinstance(output_ref, str) or not output_ref:
            raise ValueError("output_ref must be a non-empty provenance handle")
        if not isinstance(past_key_values_ref, str) or not past_key_values_ref:
            raise ValueError("past_key_values_ref must be a non-empty provenance handle")
        actions = self.queue.load_decoded_sequence(decoded_sequence)
        self.previous_output_ref = output_ref
        self.past_key_values_ref = past_key_values_ref
        self.pending_request = None
        return actions

    def pop_action(self) -> int:
        if self.terminal:
            raise RuntimeError("official session is terminal")
        if self.physical_step not in self.frame_refs:
            raise RuntimeError("current frame must be observed before executing an action")
        if self.pending_request is not None:
            raise RuntimeError("official generation must finish before executing an action")
        if self.action_popped_for_step:
            raise RuntimeError("official evaluator executes exactly one action per physical step")
        action = self.queue.pop_action()
        self.action_popped_for_step = True
        self.popped_action_for_step = action
        return action

    def after_env_step(self, *, terminated: bool = False, truncated: bool = False) -> dict:
        if self.terminal:
            raise RuntimeError("official session is terminal")
        if type(terminated) is not bool or type(truncated) is not bool:
            raise TypeError("terminated and truncated must be boolean")
        if terminated and truncated:
            raise ValueError("one environment step cannot terminate and truncate together")
        if self.physical_step not in self.frame_refs:
            raise RuntimeError("cannot advance before observing the current frame")
        if not self.action_popped_for_step:
            raise RuntimeError("cannot advance before executing exactly one official action")
        if self.pending_request is not None:
            raise RuntimeError("cannot advance with an unfinished generation request")
        completed_step = self.physical_step
        completed_action = self.popped_action_for_step
        self.physical_step += 1
        self.action_popped_for_step = False
        self.popped_action_for_step = None
        if truncated:
            self.terminal_reason = "truncated"
        elif terminated and completed_action != STOP_ACTION:
            self.terminal_reason = "terminated"
        elif completed_action == STOP_ACTION:
            self.terminal_reason = "stop"
        self.terminal = self.terminal_reason is not None
        # Upstream performs this reset after env.step even when that action ends
        # the episode; terminal latching only prevents the next observation.
        reset = self.physical_step % NUM_FRAMES == 0
        if reset:
            if not self.queue.needs_generation:
                self.queue_crossed_reset = True
            self.previous_output_ref = None
            self.past_key_values_ref = None
            self.time_ids = []
            self.reset_steps.append(self.physical_step)
        return {
            "completed_step": completed_step,
            "next_step": self.physical_step,
            "cache_reset": reset,
            "pending_action_count": len(self.queue.pending_actions),
            "queue_crossed_reset": self.queue_crossed_reset,
            "action_popped_for_step": self.action_popped_for_step,
            "completed_action": completed_action,
            "terminal": self.terminal,
            "terminal_reason": self.terminal_reason,
        }

    def snapshot(self) -> dict:
        return {
            "physical_step": self.physical_step,
            "pending_actions": list(self.queue.pending_actions),
            "time_ids": list(self.time_ids),
            "previous_output_ref": self.previous_output_ref,
            "past_key_values_ref": self.past_key_values_ref,
            "reset_steps": list(self.reset_steps),
            "queue_crossed_reset": self.queue_crossed_reset,
            "frame_count": len(self.frame_refs),
            "terminal": self.terminal,
            "terminal_reason": self.terminal_reason,
            "contract_sha256": official_keep_contract_sha256(),
        }


def official_runtime_receipt_template() -> dict:
    """Return an isolated expected-contract receipt with no live verification."""

    expected = {
        "source_commit": PINNED_STREAMVLN_COMMIT,
        "evaluator_sha256": PINNED_EVALUATOR_SHA256,
        "model_code_sha256": PINNED_MODEL_CODE_SHA256,
        "source_provenance": {
            "kind": "clean_upstream_git_checkout",
            "commit": PINNED_STREAMVLN_COMMIT,
            "source_root_kind": PINNED_SOURCE_ROOT_KIND,
            "import_root_relative_path": PINNED_SOURCE_IMPORT_ROOT_RELATIVE_PATH,
            "evaluator_relative_path": PINNED_EVALUATOR_SOURCE_RELATIVE_PATH,
            "model_relative_path": PINNED_MODEL_SOURCE_RELATIVE_PATH,
        },
        "artifact_ids": PINNED_ARTIFACT_IDS,
        "pinned_release_manifest_sha256": PINNED_RELEASE_MANIFEST_SHA256,
        "release_manifest_schema": {
            "name": RELEASE_MANIFEST_SCHEMA_NAME,
            "version": RELEASE_MANIFEST_SCHEMA_VERSION,
        },
        "attention_implementation": ATTENTION_IMPLEMENTATION,
        "model_dtype": MODEL_DTYPE,
        "generation_kwargs": GENERATION_KWARGS,
        "decode_kwargs": DECODE_KWARGS,
        "full_vocabulary_generation": True,
        "sampler": "greedy",
        "transformers_version": PINNED_TRANSFORMERS_VERSION,
        "torch_version": PINNED_TORCH_VERSION,
        "contract_sha256": official_keep_contract_sha256(),
        "model_input_contract": MODEL_INPUT_CONTRACT,
        "system_message": SYSTEM_MESSAGE,
        "source_message_roles": list(SOURCE_MESSAGE_ROLES),
        "rendered_message_roles": list(RENDERED_MESSAGE_ROLES),
        "empty_assistant_content": EMPTY_ASSISTANT_CONTENT,
        "add_generation_prompt": APPLY_CHAT_TEMPLATE_ADD_GENERATION_PROMPT,
        "chat_template_application": CHAT_TEMPLATE_APPLICATION,
        "tokenizer_preprocess_contract": TOKENIZER_PREPROCESS_CONTRACT,
        "frame_preprocess_contract": FRAME_PREPROCESS_CONTRACT,
        "model_internal_state_contract": MODEL_INTERNAL_STATE_CONTRACT,
        "state_audit_scope": STATE_AUDIT_SCOPE,
        "verification_scope": "expected_contract_only",
        "observed_runtime": None,
    }
    return copy.deepcopy(expected)


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_official_runtime_receipt(
    receipt: Mapping,
    *,
    validation_level: str = PRODUCTION_RUNTIME_VALIDATION_LEVEL,
) -> None:
    """Validate a production release receipt or explicit test fixture.

    Production validation has no downgrade: it requires the independently
    frozen release-manifest digest and an observed full-content receipt bound
    exactly to that digest.  The test_only_contract level exists solely for
    dependency-free state-machine tests and accepts no observed runtime claim.
    """

    if validation_level not in {
        PRODUCTION_RUNTIME_VALIDATION_LEVEL,
        TEST_ONLY_CONTRACT_VALIDATION_LEVEL,
    }:
        raise ValueError("official runtime receipt validation level is invalid")
    expected = official_runtime_receipt_template()
    if not isinstance(receipt, Mapping):
        raise TypeError("official runtime receipt must be a mapping")
    receipt = dict(receipt)
    observation_keys = {"verification_scope", "observed_runtime"}
    mismatches = {
        key: {"expected": value, "actual": receipt.get(key)}
        for key, value in expected.items()
        if key not in observation_keys
        if receipt.get(key) != value
    }
    forbidden = {
        "allowed_action_token_mask": receipt.get("allowed_action_token_mask"),
        "multinomial_sampling": receipt.get("multinomial_sampling"),
        "fixed_four_action_sampler": receipt.get("fixed_four_action_sampler"),
    }
    enabled_forbidden = [key for key, value in forbidden.items() if value]
    unexpected_keys = sorted(set(receipt) - set(expected))
    scope = receipt.get("verification_scope")
    observed = receipt.get("observed_runtime")
    observation_errors: list[str] = []
    if unexpected_keys:
        observation_errors.append(
            f"runtime receipt has unexpected keys: {unexpected_keys}"
        )

    if validation_level == TEST_ONLY_CONTRACT_VALIDATION_LEVEL:
        if scope != "expected_contract_only":
            observation_errors.append(
                "test-only contract validation requires expected-only scope"
            )
        if observed is not None:
            observation_errors.append(
                "test-only contract receipt must not claim runtime observations"
            )
    else:
        anchor = PINNED_RELEASE_MANIFEST_SHA256
        if not _valid_sha256(anchor):
            observation_errors.append(
                "pinned release manifest SHA-256 is pending"
            )
        if scope != "full_content":
            observation_errors.append(
                "production runtime requires full-content release verification"
            )
        if not isinstance(observed, Mapping):
            observation_errors.append(
                "full-content release receipt has no observed runtime"
            )
        else:
            if observed.get("release_manifest_sha256") != anchor:
                observation_errors.append(
                    "observed release manifest does not match the pinned digest"
                )
            release_manifest = observed.get("release_manifest")
            manifest_source = None
            manifest_artifacts = None
            if not isinstance(release_manifest, Mapping):
                observation_errors.append(
                    "observed canonical release manifest is missing"
                )
            else:
                if (
                    set(release_manifest)
                    != {
                        "schema_name",
                        "schema_version",
                        "verification_scope",
                        "source",
                        "artifacts",
                    }
                    or release_manifest.get("schema_name")
                    != RELEASE_MANIFEST_SCHEMA_NAME
                    or release_manifest.get("schema_version")
                    != RELEASE_MANIFEST_SCHEMA_VERSION
                    or release_manifest.get("verification_scope")
                    != "candidate_full_content"
                ):
                    observation_errors.append(
                        "canonical release manifest schema drifted"
                    )
                try:
                    observed_manifest_digest = _canonical_json_sha256(
                        release_manifest
                    )
                except (TypeError, ValueError):
                    observed_manifest_digest = None
                if observed_manifest_digest != anchor:
                    observation_errors.append(
                        "canonical release manifest does not match pinned digest"
                    )
                manifest_source = release_manifest.get("source")
                manifest_artifacts = release_manifest.get("artifacts")

            expected_observed_keys = {
                "release_manifest_sha256",
                "release_manifest",
                "source",
                "artifacts",
                "device",
            }
            if set(observed) != expected_observed_keys:
                observation_errors.append("observed runtime schema drifted")

            source = observed.get("source")
            if not isinstance(source, Mapping):
                observation_errors.append("observed source receipt is missing")
            else:
                expected_source = {
                    "source_id": PINNED_STREAMVLN_COMMIT,
                    "source_root_kind": PINNED_SOURCE_ROOT_KIND,
                    "source_import_root_relative_path": (
                        PINNED_SOURCE_IMPORT_ROOT_RELATIVE_PATH
                    ),
                    "model_source_relative_path": PINNED_MODEL_SOURCE_RELATIVE_PATH,
                    "model_source_sha256": PINNED_MODEL_CODE_SHA256,
                    "evaluator_source_relative_path": (
                        PINNED_EVALUATOR_SOURCE_RELATIVE_PATH
                    ),
                    "evaluator_sha256": PINNED_EVALUATOR_SHA256,
                    "imported_model_module_relative_path": (
                        PINNED_MODEL_SOURCE_RELATIVE_PATH
                    ),
                    "imported_model_module_sha256": PINNED_MODEL_CODE_SHA256,
                    "model_class_module": PINNED_MODEL_CLASS_MODULE,
                }
                if set(source) != set(expected_source) | {
                    "python_file_count",
                    "python_content_manifest_sha256",
                }:
                    observation_errors.append("observed source schema drifted")
                for key, value in expected_source.items():
                    if source.get(key) != value:
                        observation_errors.append(f"observed source {key} drifted")
                source_files = (
                    manifest_source.get("python_files")
                    if isinstance(manifest_source, Mapping)
                    else None
                )
                if (
                    not isinstance(manifest_source, Mapping)
                    or manifest_source.get("source_id") != PINNED_STREAMVLN_COMMIT
                    or manifest_source.get("source_root_kind")
                    != PINNED_SOURCE_ROOT_KIND
                    or manifest_source.get("import_root_relative_path")
                    != PINNED_SOURCE_IMPORT_ROOT_RELATIVE_PATH
                    or not isinstance(source_files, list)
                    or len(source_files) < 2
                ):
                    observation_errors.append(
                        "canonical source Python inventory is invalid"
                    )
                else:
                    if source.get("python_file_count") != len(source_files):
                        observation_errors.append(
                            "observed source Python file count drifted"
                        )
                    if source.get(
                        "python_content_manifest_sha256"
                    ) != _canonical_json_sha256(source_files):
                        observation_errors.append(
                            "observed source Python inventory digest drifted"
                        )
                    source_by_path = {
                        entry.get("relative_path"): entry
                        for entry in source_files
                        if isinstance(entry, Mapping)
                    }
                    required_source_hashes = {
                        PINNED_MODEL_SOURCE_RELATIVE_PATH: PINNED_MODEL_CODE_SHA256,
                        PINNED_EVALUATOR_SOURCE_RELATIVE_PATH: PINNED_EVALUATOR_SHA256,
                    }
                    for path, digest in required_source_hashes.items():
                        if source_by_path.get(path, {}).get("sha256") != digest:
                            observation_errors.append(
                                f"canonical source {path} drifted"
                            )

            artifacts = observed.get("artifacts")
            if not isinstance(artifacts, Mapping) or set(artifacts) != set(
                PINNED_ARTIFACT_IDS
            ):
                observation_errors.append("full-content artifact set drifted")
            elif (
                not isinstance(manifest_artifacts, Mapping)
                or set(manifest_artifacts) != set(PINNED_ARTIFACT_IDS)
            ):
                observation_errors.append(
                    "canonical release artifact set drifted"
                )
            else:
                for name, artifact_id in PINNED_ARTIFACT_IDS.items():
                    record = artifacts.get(name)
                    manifest_record = manifest_artifacts.get(name)
                    if not isinstance(record, Mapping):
                        observation_errors.append(
                            f"{name} artifact receipt is missing"
                        )
                        continue
                    if (
                        not isinstance(manifest_record, Mapping)
                        or manifest_record.get("artifact_id") != artifact_id
                        or not isinstance(manifest_record.get("files"), list)
                        or not manifest_record.get("files")
                    ):
                        observation_errors.append(
                            f"canonical {name} artifact manifest is invalid"
                        )
                        continue
                    files = manifest_record["files"]
                    if set(record) != {
                        "artifact_id",
                        "file_count",
                        "content_manifest_sha256",
                    }:
                        observation_errors.append(
                            f"{name} artifact receipt schema drifted"
                        )
                    if set(manifest_record) != {"artifact_id", "files"}:
                        observation_errors.append(
                            f"canonical {name} artifact schema drifted"
                        )
                    if record.get("artifact_id") != artifact_id:
                        observation_errors.append(
                            f"{name} artifact identity drifted"
                        )
                    if record.get("file_count") != len(files):
                        observation_errors.append(
                            f"{name} artifact file count drifted"
                        )
                    if record.get(
                        "content_manifest_sha256"
                    ) != _canonical_json_sha256(files):
                        observation_errors.append(
                            f"{name} artifact digest drifted"
                        )

            device = observed.get("device")
            if (
                not isinstance(device, Mapping)
                or set(device) != {"requested", "resolved", "local_rank"}
                or not isinstance(device.get("requested"), str)
                or re.fullmatch(r"cuda:[0-9]+", device.get("requested", ""))
                is None
                or not isinstance(device.get("resolved"), str)
                or re.fullmatch(r"cuda:[0-9]+", device.get("resolved", ""))
                is None
                or device.get("requested") != device.get("resolved")
                or type(device.get("local_rank")) is not int
                or device.get("resolved")
                != f"cuda:{device.get('local_rank')}"
            ):
                observation_errors.append("observed runtime CUDA binding is invalid")

    if mismatches or enabled_forbidden or observation_errors:
        raise ValueError(
            f"runtime is not pinned official StreamVLN inference; "
            f"mismatches={mismatches}, forbidden={enabled_forbidden}, "
            f"observations={observation_errors}"
        )
