"""One-shot A800 parity for clean-e48 StreamVLN versus the P8 adapter.

This module stays light until formal provenance, source, and artifact bytes have
all passed.  It never constructs Habitat or reads datasets/splits.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import struct
import sys
from typing import Callable, Mapping

try:
    from . import streamvln_p8_clean_e48_reference as reference_module
    from . import streamvln_p8_official_adapter as adapter_module
    from .streamvln_p8_official_frame import OFFICIAL_RGB_HEIGHT, OFFICIAL_RGB_WIDTH, OfficialFrame
    from .streamvln_p8_official_keep import (
        NUM_FRAMES,
        PINNED_EVALUATOR_SOURCE_RELATIVE_PATH,
        PINNED_MODEL_CLASS_MODULE,
        PINNED_SOURCE_IMPORT_ROOT_RELATIVE_PATH,
        OfficialEvalSession,
        OfficialGenerationRequest,
        OfficialKeepQueue,
    )
except ImportError:  # Direct staged-file execution.
    import streamvln_p8_clean_e48_reference as reference_module
    import streamvln_p8_official_adapter as adapter_module
    from streamvln_p8_official_frame import OFFICIAL_RGB_HEIGHT, OFFICIAL_RGB_WIDTH, OfficialFrame
    from streamvln_p8_official_keep import (
        NUM_FRAMES,
        PINNED_EVALUATOR_SOURCE_RELATIVE_PATH,
        PINNED_MODEL_CLASS_MODULE,
        PINNED_SOURCE_IMPORT_ROOT_RELATIVE_PATH,
        OfficialEvalSession,
        OfficialGenerationRequest,
        OfficialKeepQueue,
    )


SCHEMA_NAME = "streamvln_p8_a800_parity"
SCHEMA_VERSION = 1
CHECK_NAMES = ("preprocess", "initial_generation", "continuation", "history_and_reset")
TENSORS = ("images", "depths", "poses", "intrinsics")
INSTRUCTION = "walk to the synthetic parity target"
CONJUNCTIONS = (
    "you can see ", "in front of you is ", "there is ", "you can spot ",
    "you are toward the ", "ahead of you is ", "in your sight is ",
)


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _json_hash(value: object) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _read_manifest(path_value: str | Path) -> dict:
    path = Path(path_value).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError("parity manifest must be a regular file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("parity manifest root must be an object")
    return value


def preflight(args: argparse.Namespace) -> dict:
    """Perform every light verification before any heavy import or path install."""

    device = adapter_module._validate_explicit_factory_cuda(args.device, args.local_rank)
    # Formal-only: there is no flag or callable path that accepts a candidate.
    anchor = adapter_module._require_pinned_release_manifest_sha256()
    manifest_path = Path(args.manifest).expanduser().resolve(strict=True)
    manifest = _read_manifest(manifest_path)
    if adapter_module._manifest_sha256(manifest) != anchor:
        raise RuntimeError("parity manifest digest does not match pinned release")

    source_root = Path(args.source_root).expanduser().resolve(strict=True)
    import_root = Path(args.import_root).expanduser().resolve(strict=True)
    expected_import_root = (source_root / PINNED_SOURCE_IMPORT_ROOT_RELATIVE_PATH).resolve(strict=True)
    if not source_root.is_dir() or not import_root.is_dir() or import_root != expected_import_root:
        raise ValueError("parity source/import root contract drifted")
    verification = adapter_module.verify_full_content_artifact_manifest(
        manifest,
        source_root=source_root,
        model_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        vision_tower_path=args.vision,
    )
    verified_root, source = adapter_module._verify_source_tree(source_root)
    if verified_root != source_root or verification["manifest_sha256"] != anchor:
        raise RuntimeError("parity verified release provenance drifted")
    verification["source"].update(source)
    return {
        "anchor": anchor,
        "manifest_path": str(manifest_path),
        "source_root": str(source_root),
        "import_root": str(import_root),
        "device": device,
        "verification": verification,
    }


def _audit(value: object, label: str) -> dict:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} audit is not an object")
    result = dict(value)
    if set(result) != {"dtype", "shape", "sha256", "nbytes"}:
        raise RuntimeError(f"{label} audit schema drifted")
    digest = result["sha256"]
    if (not isinstance(result["dtype"], str) or not isinstance(result["shape"], list)
            or not isinstance(digest, str) or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or type(result["nbytes"]) is not int or result["nbytes"] < 1):
        raise RuntimeError(f"{label} audit value drifted")
    return result


def _validated_side(value: object, label: str) -> dict:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} side is not an object")
    side = copy.deepcopy(dict(value))
    if set(side) != set(CHECK_NAMES):
        raise RuntimeError(f"{label} check set drifted")
    preprocess = side["preprocess"]
    if not isinstance(preprocess, Mapping) or set(preprocess) != {"input_ids", "tensors"}:
        raise RuntimeError(f"{label} preprocess schema drifted")
    input_audit = _audit(preprocess["input_ids"], f"{label}.input_ids")
    if (input_audit["dtype"] != "torch.int64" or len(input_audit["shape"]) != 2
            or input_audit["shape"][0] != 1 or input_audit["shape"][1] < 1):
        raise RuntimeError(f"{label}.input_ids tensor contract drifted")
    if not isinstance(preprocess["tensors"], Mapping) or set(preprocess["tensors"]) != set(TENSORS):
        raise RuntimeError(f"{label} tensor set drifted")
    expected_preprocess_shapes = {
        "images": [1, 1, 3, 384, 384], "depths": [1, 1, 384, 384],
        "poses": [1, 1, 4, 4], "intrinsics": [1, 1, 4, 4],
    }
    for name in TENSORS:
        tensor_audit = _audit(preprocess["tensors"][name], f"{label}.{name}")
        if tensor_audit["dtype"] != "torch.bfloat16" or tensor_audit["shape"] != expected_preprocess_shapes[name]:
            raise RuntimeError(f"{label}.{name} tensor contract drifted")
    generation_keys = {
        "output_ids", "decoded", "parsed_queue", "pop_cadence",
        "past_key_values_structure", "model_curr_t", "inputs_embeds_seq_len",
    }
    for name in ("initial_generation", "continuation"):
        item = side[name]
        if not isinstance(item, Mapping) or set(item) != generation_keys:
            raise RuntimeError(f"{label}.{name} schema drifted")
        output_audit = _audit(item["output_ids"], f"{label}.{name}.output_ids")
        if (output_audit["dtype"] != "torch.int64" or len(output_audit["shape"]) != 2
                or output_audit["shape"][0] != 1 or output_audit["shape"][1] < 1):
            raise RuntimeError(f"{label}.{name} output token contract drifted")
        if not isinstance(item["decoded"], str) or item["parsed_queue"] != item["pop_cadence"]:
            raise RuntimeError(f"{label}.{name} token/queue cadence drifted")
    history = side["history_and_reset"]
    expected_history_keys = {
        "frame_count", "frame_steps", "tensors", "reset_step", "reset_state",
        "boundary_generation", "exact_queue_exhausted",
        "queue_cross_reset_state", "queue_cross_reset_rejected",
    }
    if not isinstance(history, Mapping) or set(history) != expected_history_keys:
        raise RuntimeError(f"{label}.history schema drifted")
    if (history["frame_count"] != 9
            or history["frame_steps"] != [0, 4, 8, 12, 16, 20, 24, 28, 32]
            or history["reset_step"] != 32
            or history["exact_queue_exhausted"] is not True
            or history["queue_cross_reset_rejected"] is not True):
        raise RuntimeError(f"{label}.history/reset value drifted")
    reset_state = history["reset_state"]
    if (not isinstance(reset_state, Mapping)
            or set(reset_state) != {"curr_t", "cache_nonempty", "curr_t_after", "cache_empty_after"}
            or type(reset_state["curr_t"]) is not int or reset_state["curr_t"] < 1
            or reset_state["cache_nonempty"] is not True
            or reset_state["curr_t_after"] != 0
            or reset_state["cache_empty_after"] is not True):
        raise RuntimeError(f"{label}.reset state drifted")
    if not isinstance(history["tensors"], Mapping) or set(history["tensors"]) != set(TENSORS):
        raise RuntimeError(f"{label}.history tensor set drifted")
    expected_history_shapes = {
        "images": [1, 9, 3, 384, 384], "depths": [1, 9, 384, 384],
        "poses": [1, 9, 4, 4], "intrinsics": [1, 9, 4, 4],
    }
    for name in TENSORS:
        tensor_audit = _audit(history["tensors"][name], f"{label}.history.{name}")
        if tensor_audit["dtype"] != "torch.bfloat16" or tensor_audit["shape"] != expected_history_shapes[name]:
            raise RuntimeError(f"{label}.history.{name} tensor contract drifted")
    boundary = history["boundary_generation"]
    if not isinstance(boundary, Mapping) or set(boundary) != generation_keys:
        raise RuntimeError(f"{label}.boundary generation schema drifted")
    boundary_output = _audit(boundary["output_ids"], f"{label}.boundary.output_ids")
    if (boundary_output["dtype"] != "torch.int64" or len(boundary_output["shape"]) != 2
            or boundary_output["shape"][0] != 1 or boundary_output["shape"][1] < 1
            or boundary["parsed_queue"] != boundary["pop_cadence"]):
        raise RuntimeError(f"{label}.boundary generation value drifted")
    cross_state = history["queue_cross_reset_state"]
    if (not isinstance(cross_state, Mapping)
            or cross_state.get("physical_step") != 33
            or cross_state.get("reset_step") != 32
            or cross_state.get("crossed") is not True
            or cross_state.get("residual_after_reset") != 1
            or cross_state.get("pending_after_pop33") != 0):
        raise RuntimeError(f"{label}.queue cross-reset state drifted")
    return side


def compare_parity_sides(reference: object, adapter: object) -> dict:
    expected = _validated_side(reference, "reference")
    observed = _validated_side(adapter, "adapter")
    if expected != observed:
        raise RuntimeError("clean-e48/adapter parity drift detected")
    return {"status": "pass", "check_names": list(CHECK_NAMES), "comparison_sha256": _json_hash(expected)}


def build_receipt(*, preflight_receipt: Mapping, reference: object, adapter: object) -> dict:
    expected = _validated_side(reference, "reference")
    observed = _validated_side(adapter, "adapter")
    verification = preflight_receipt["verification"]
    scenario = {
        "seed": 0,
        "synthetic_frame_count": 33,
        "first_generation_step": 0,
        "continuation_generation": True,
        "history_frame_steps": [0, 4, 8, 12, 16, 20, 24, 28, 32],
        "queue_cross_reset_probe": [1] * 33,
    }
    receipt = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "provenance": {
            "release_manifest_sha256": preflight_receipt["anchor"],
            "manifest_path": preflight_receipt["manifest_path"],
            "source_root": preflight_receipt["source_root"],
            "import_root": preflight_receipt["import_root"],
            "source_manifest_sha256": _json_hash(verification["source"]),
            "artifact_manifest_sha256": _json_hash(verification["artifacts"]),
            "evaluator_sha256": verification["source"]["evaluator_sha256"],
            "model_source_sha256": verification["source"]["model_source_sha256"],
        },
        "invocation": {"device": preflight_receipt["device"]},
        "synthetic_scenario": scenario,
        "synthetic_scenario_sha256": _json_hash(scenario),
        "checks": {
            "reference": expected,
            "adapter": observed,
            "comparison": compare_parity_sides(expected, observed),
        },
    }
    receipt["receipt_sha256"] = _json_hash(receipt)
    return receipt


def _atomic_write(path_value: str | Path, receipt: Mapping) -> None:
    path = Path(path_value).expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        temporary.write_bytes(_json_bytes(receipt) + b"\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _synthetic_frame(step: int) -> OfficialFrame:
    rgb = bytes((index + step) % 251 for index in range(OFFICIAL_RGB_HEIGHT * OFFICIAL_RGB_WIDTH * 3))
    depth = struct.pack("<f", min(0.99, 0.2 + step / 1000)) * (OFFICIAL_RGB_HEIGHT * OFFICIAL_RGB_WIDTH)
    identity = (1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.)
    pose = struct.pack("<16d", *identity)
    intrinsic = list(identity)
    intrinsic[0], intrinsic[5], intrinsic[2], intrinsic[6] = 320., 320., 320., 240.
    intrinsics = struct.pack("<16d", *intrinsic)
    digest = lambda payload: hashlib.sha256(payload).hexdigest()
    return OfficialFrame(
        env_id=0, episode_serial=0, episode_generation="0" * 32,
        physical_step=step, episode_id="p8-a800-parity", scene_id="synthetic-no-habitat",
        instruction_sha256=digest(INSTRUCTION.encode()), rgb_u8=rgb,
        depth_normalized_f32le=depth, pose_f64le=pose, intrinsics_f64le=intrinsics,
        rgb_sha256=digest(rgb), depth_sha256=digest(depth), pose_sha256=digest(pose),
        intrinsics_sha256=digest(intrinsics),
    )


def _load_runtime(preflight_receipt: Mapping) -> dict:
    """First heavy-import boundary; called only after successful preflight."""

    root = Path(preflight_receipt["source_root"])
    adapter_module._install_source_import_root(root)
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    numpy = importlib.import_module("numpy")
    image = importlib.import_module("PIL.Image")
    to_numpy_array = importlib.import_module("transformers.image_utils").to_numpy_array
    filter_depth = importlib.import_module("depth_camera_filtering").filter_depth
    evaluator = reference_module.load_verified_evaluator_namespace(
        root / PINNED_EVALUATOR_SOURCE_RELATIVE_PATH, torch_module=torch)
    model_module = importlib.import_module(PINNED_MODEL_CLASS_MODULE)
    model_class = model_module.StreamVLNForCausalLM
    adapter_module._verify_imported_model_source(model_module, model_class, root)
    return dict(torch=torch, transformers=transformers, numpy=numpy, image=image,
                to_numpy_array=to_numpy_array, filter_depth=filter_depth,
                evaluator=evaluator, model_class=model_class)


def _load_model(args, modules):
    torch, transformers = modules["torch"], modules["transformers"]
    torch.cuda.set_device(args.local_rank)
    device = torch.device(args.device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.tokenizer, model_max_length=4096, padding_side="right", local_files_only=True)
    config = transformers.AutoConfig.from_pretrained(args.checkpoint, local_files_only=True)
    config.mm_vision_tower = config.vision_tower = args.vision
    model = modules["model_class"].from_pretrained(
        args.checkpoint, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16,
        config=config, low_cpu_mem_usage=False, local_files_only=True)
    model.model.num_history = 8
    model.requires_grad_(False); model.to(device); model.eval(); model.reset(1)
    return model, tokenizer, device


def _generation_record(runner, decoded, queue, pops, auditor, cache_builder):
    return {
        "output_ids": auditor(runner.output_ids), "decoded": decoded,
        "parsed_queue": list(queue), "pop_cadence": list(pops),
        "past_key_values_structure": cache_builder(runner.past_key_values),
        "model_curr_t": runner.model.curr_t[0],
        "inputs_embeds_seq_len": int(runner.model.cache[0]["inputs_embeds"].shape[1]),
    }


def _exercise_reference(model, tokenizer, device, modules, frames) -> dict:
    torch = modules["torch"]
    runner = reference_module.CleanE48ReferenceRunner(
        evaluator_namespace=modules["evaluator"], model=model, tokenizer=tokenizer,
        image_processor=model.get_vision_tower().image_processor, device=device, modules=modules)
    runner.reset(); rng = random.Random(0)
    first_conjunction = rng.choice(CONJUNCTIONS)
    content = reference_module.reference_user_content(INSTRUCTION, step=0)
    tensors, ids, decoded = runner.generate(
        source_messages=reference_module.reference_source_messages(content), add_system=True,
        conjunction=first_conjunction, time_ids=(0,), frames=(frames[0],))
    queue = reference_module.ReferenceQueue(); values = queue.load_actions(runner.parse_actions(decoded))
    pops = [queue.pop() for _ in values]
    first = _generation_record(runner, decoded, values, pops,
        lambda x: reference_module.tensor_audit(x, torch),
        lambda x: reference_module.cache_structure(x, torch))
    continuation_conjunction = rng.choice(CONJUNCTIONS)
    _, _, decoded2 = runner.generate(
        source_messages=reference_module.reference_source_messages(""), add_system=False,
        conjunction=continuation_conjunction, time_ids=(0, 1), frames=(frames[1],))
    values2 = queue.load_actions(runner.parse_actions(decoded2)); pops2 = [queue.pop() for _ in values2]
    second = _generation_record(runner, decoded2, values2, pops2,
        lambda x: reference_module.tensor_audit(x, torch),
        lambda x: reference_module.cache_structure(x, torch))
    reset = {"curr_t": model.curr_t[0], "cache_nonempty": bool(model.cache[0])}
    runner.reset(); reset.update(curr_t_after=model.curr_t[0], cache_empty_after=model.cache[0] == {})
    history_steps = list(range(0, 32, 4)) + [32]
    exact = reference_module.ReferenceQueue(); exact.pending.extend([1] * 32)
    for _ in range(32): exact.pop()
    exact_exhausted = not exact.pending
    boundary_content = reference_module.reference_user_content(INSTRUCTION, step=32)
    history_frames = tuple(frames[index] for index in history_steps)
    history, _, decoded32 = runner.generate(
        source_messages=reference_module.reference_source_messages(boundary_content),
        add_system=True, conjunction=rng.choice(CONJUNCTIONS), time_ids=(32,),
        frames=history_frames)
    boundary_queue = reference_module.ReferenceQueue()
    boundary_values = boundary_queue.load_actions(runner.parse_actions(decoded32))
    boundary_pops = [boundary_queue.pop() for _ in boundary_values]
    boundary_generation = _generation_record(
        runner, decoded32, boundary_values, boundary_pops,
        lambda x: reference_module.tensor_audit(x, torch),
        lambda x: reference_module.cache_structure(x, torch))
    crossing = reference_module.ReferenceQueue(); crossing.pending.extend([1] * 33)
    physical_step = 0; crossed = False
    for _ in range(32):
        crossing.pop(); physical_step += 1
        if physical_step == 32 and crossing.pending: crossed = True
    residual_after_reset = len(crossing.pending)
    crossing.pop(); physical_step += 1
    try:
        if crossed: raise RuntimeError("reference queue crossed reset")
        crossing.load_actions([1])
    except RuntimeError: rejected = True
    else: rejected = False
    cross_state = {"physical_step": physical_step, "reset_step": 32, "crossed": crossed,
                   "residual_after_reset": residual_after_reset,
                   "pending_after_pop33": len(crossing.pending)}
    audit = lambda x: reference_module.tensor_audit(x, torch)
    return {
        "preprocess": {"input_ids": audit(ids), "tensors": {name: audit(tensors[name]) for name in TENSORS}},
        "initial_generation": first, "continuation": second,
        "history_and_reset": {"frame_count": 9, "frame_steps": history_steps,
            "tensors": {name: audit(history[name]) for name in TENSORS}, "reset_step": 32,
            "reset_state": reset, "exact_queue_exhausted": exact_exhausted,
            "boundary_generation": boundary_generation,
            "queue_cross_reset_state": cross_state,
            "queue_cross_reset_rejected": rejected},
    }


def _exercise_adapter(model, tokenizer, device, modules, frames) -> dict:
    torch = modules["torch"]
    preprocessor = adapter_module.OfficialFramePreprocessor(
        image_processor=model.get_vision_tower().image_processor, device=device,
        torch_module=torch, numpy_module=modules["numpy"], image_module=modules["image"],
        to_numpy_array=modules["to_numpy_array"], filter_depth=modules["filter_depth"])
    runner = adapter_module.PinnedStreamVLNGenerationAdapter(
        model=model, tokenizer=tokenizer, device=device, requested_device=str(device), env_id=0,
        torch_module=torch, transformers_version=modules["transformers"].__version__,
        frame_preprocessor=preprocessor)
    runner.reset({"event": "episode_reset", "env_id": 0, "physical_step": 0, "terminal": False})
    rng = random.Random(0)
    session = OfficialEvalSession(env_id=0, instruction=INSTRUCTION, conjunction_rng=rng)
    session.observe(frames[0].frame_ref); request = session.build_generation_request()
    prepared = preprocessor((frames[0],)).tensors
    ids = adapter_module.build_official_input_ids(request=request, tokenizer=tokenizer, torch_module=torch).to(device)
    result = runner(request, (frames[0],))
    values = list(session.accept_generation(decoded_sequence=result.decoded_sequence,
        output_ref=result.output_ref, past_key_values_ref=result.past_key_values_ref))
    pops = [session.queue.pop_action() for _ in values]
    first = _generation_record(runner, result.decoded_sequence, values, pops,
        lambda x: adapter_module._tensor_audit(x, torch),
        lambda x: adapter_module._cache_structure(x, torch))
    continuation = f"{rng.choice(CONJUNCTIONS)}<image>."
    request2 = OfficialGenerationRequest(
        physical_step=1, env_id=0, user_content=continuation,
        conjunction=continuation.removesuffix("<image>."), add_system=False,
        frame_steps=(1,), time_ids=(0, 1), previous_output_ref=result.output_ref,
        past_key_values_ref=result.past_key_values_ref,
        contract_sha256=request.contract_sha256,
        source_messages=(("human", continuation), ("gpt", "")),
        rendered_messages=(("user", continuation), ("assistant", "")))
    result2 = runner(request2, (frames[1],))
    queue2 = OfficialKeepQueue(); values2 = list(queue2.load_decoded_sequence(result2.decoded_sequence))
    pops2 = [queue2.pop_action() for _ in values2]
    second = _generation_record(runner, result2.decoded_sequence, values2, pops2,
        lambda x: adapter_module._tensor_audit(x, torch),
        lambda x: adapter_module._cache_structure(x, torch))
    reset = {"curr_t": model.curr_t[0], "cache_nonempty": bool(model.cache[0])}
    runner.reset({"event": "cache_reset", "env_id": 0, "next_step": NUM_FRAMES})
    reset.update(curr_t_after=model.curr_t[0], cache_empty_after=model.cache[0] == {})
    exact = OfficialEvalSession(env_id=0, instruction=INSTRUCTION, conjunction_rng=rng)
    exact.queue.load_actions_for_test([1] * 32)
    for step in range(32):
        exact.observe(frames[step].frame_ref); exact.pop_action(); exact.after_env_step()
    exact.observe(frames[32].frame_ref); boundary_request = exact.build_generation_request()
    history_steps = list(boundary_request.frame_steps)
    exact_exhausted = exact.queue.needs_generation
    history_frames = tuple(frames[index] for index in history_steps)
    history = preprocessor(history_frames).tensors
    result32 = runner(boundary_request, history_frames)
    boundary_queue = OfficialKeepQueue()
    boundary_values = list(boundary_queue.load_decoded_sequence(result32.decoded_sequence))
    boundary_pops = [boundary_queue.pop_action() for _ in boundary_values]
    boundary_generation = _generation_record(
        runner, result32.decoded_sequence, boundary_values, boundary_pops,
        lambda x: adapter_module._tensor_audit(x, torch),
        lambda x: adapter_module._cache_structure(x, torch))
    crossing = OfficialEvalSession(env_id=0, instruction=INSTRUCTION, conjunction_rng=random.Random(0))
    crossing.queue.load_actions_for_test([1] * 33)
    for step in range(32):
        crossing.observe(frames[step].frame_ref); crossing.pop_action(); crossing.after_env_step()
    residual_after_reset = len(crossing.queue.pending_actions)
    crossing.observe(frames[32].frame_ref); crossing.pop_action(); crossing.after_env_step()
    try:
        if crossing.queue_crossed_reset: raise RuntimeError("official queue crossed reset")
        crossing.build_generation_request()
    except RuntimeError: rejected = crossing.queue_crossed_reset
    else: rejected = False
    cross_state = {"physical_step": crossing.physical_step, "reset_step": 32,
                   "crossed": crossing.queue_crossed_reset,
                   "residual_after_reset": residual_after_reset,
                   "pending_after_pop33": len(crossing.queue.pending_actions)}
    audit = lambda x: adapter_module._tensor_audit(x, torch)
    return {
        "preprocess": {"input_ids": audit(ids), "tensors": {name: audit(prepared[name]) for name in TENSORS}},
        "initial_generation": first, "continuation": second,
        "history_and_reset": {"frame_count": 9, "frame_steps": history_steps,
            "tensors": {name: audit(history[name]) for name in TENSORS}, "reset_step": 32,
            "reset_state": reset, "exact_queue_exhausted": exact_exhausted,
            "boundary_generation": boundary_generation,
            "queue_cross_reset_state": cross_state,
            "queue_cross_reset_rejected": bool(rejected)},
    }


def _run_live(args, _preflight, modules):
    model, tokenizer, device = _load_model(args, modules)
    frames = {step: _synthetic_frame(step) for step in range(33)}
    reference = _exercise_reference(model, tokenizer, device, modules, frames)
    observed = _exercise_adapter(model, tokenizer, device, modules, frames)
    return reference, observed


def run(args: argparse.Namespace, *, heavy_loader: Callable = _load_runtime,
        live_runner: Callable = _run_live) -> dict:
    verified = preflight(args)
    modules = heavy_loader(verified)
    reference, adapter = live_runner(args, verified, modules)
    receipt = build_receipt(preflight_receipt=verified, reference=reference, adapter=adapter)
    receipt["invocation"].update(local_rank=args.local_rank, checkpoint=str(Path(args.checkpoint).absolute()),
                                  tokenizer=str(Path(args.tokenizer).absolute()), vision=str(Path(args.vision).absolute()))
    receipt["receipt_sha256"] = _json_hash({key: value for key, value in receipt.items() if key != "receipt_sha256"})
    _atomic_write(args.output, receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "import-root", "checkpoint", "tokenizer", "vision", "manifest", "device", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--local-rank", required=True, type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    try: run(build_parser().parse_args(argv))
    except Exception as exc:
        print(f"P8 A800 parity failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
