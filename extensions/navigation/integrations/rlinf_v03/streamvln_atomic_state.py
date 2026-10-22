"""Exact snapshot, hash, and restore helpers for bounded candidate updates."""

from __future__ import annotations

import copy
import hashlib
import random

import numpy as np
import torch


def clone_tree(value):
    """Clone nested state without retaining accelerator-backed aliases."""
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {copy.deepcopy(key): clone_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_tree(item) for item in value)
    return copy.deepcopy(value)


def _hash_value(digest, value) -> None:
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"torch:")
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"numpy:")
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(repr(tuple(array.shape)).encode("utf-8"))
        digest.update(array.tobytes())
    elif isinstance(value, dict):
        digest.update(b"dict{")
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            _hash_value(digest, key)
            _hash_value(digest, value[key])
        digest.update(b"}")
    elif isinstance(value, list):
        digest.update(b"list[")
        for item in value:
            _hash_value(digest, item)
        digest.update(b"]")
    elif isinstance(value, tuple):
        digest.update(b"tuple(")
        for item in value:
            _hash_value(digest, item)
        digest.update(b")")
    elif isinstance(value, bytes):
        digest.update(b"bytes:")
        digest.update(value)
    elif value is None:
        digest.update(b"none")
    else:
        digest.update(type(value).__name__.encode("utf-8"))
        digest.update(b":")
        digest.update(repr(value).encode("utf-8"))


def canonical_sha256(value) -> str:
    digest = hashlib.sha256()
    _hash_value(digest, value)
    return digest.hexdigest()


def gradients_are_clear(core) -> bool:
    return all(parameter.grad is None for parameter in core.parameters())


def capture_training_state(manager, core) -> dict:
    """Capture all mutable training state used by the candidate-update path."""
    manager.optimizer.zero_grad(set_to_none=True)
    if not gradients_are_clear(core):
        raise RuntimeError("candidate boundary retains gradients")
    return {
        "parameters": {
            name: tensor.detach().cpu().clone()
            for name, tensor in core.high_level_state().items()
        },
        "optimizer": clone_tree(manager.optimizer.state_dict()),
        "scheduler": clone_tree(manager.lr_scheduler.state_dict()),
        "grad_scaler": clone_tree(manager.grad_scaler.state_dict()),
        "optimizer_steps": int(manager.optimizer_steps),
        "actor_version": int(core.actor_version),
        "behavior_version": int(core.behavior_version),
        "rng": {
            "python": clone_tree(random.getstate()),
            "numpy": clone_tree(np.random.get_state()),
            "torch_cpu": torch.get_rng_state().detach().cpu().clone(),
            "torch_cuda": [state.detach().cpu().clone() for state in torch.cuda.get_rng_state_all()],
        },
    }


def training_state_hashes(state: dict) -> dict[str, str]:
    hashes = {
        "parameters": canonical_sha256(state["parameters"]),
        "optimizer": canonical_sha256(state["optimizer"]),
        "scheduler": canonical_sha256(state["scheduler"]),
        "grad_scaler": canonical_sha256(state["grad_scaler"]),
        "counters": canonical_sha256({
            "optimizer_steps": state["optimizer_steps"],
            "actor_version": state["actor_version"],
            "behavior_version": state["behavior_version"],
        }),
        "rng": canonical_sha256(state["rng"]),
    }
    hashes["complete"] = canonical_sha256(hashes)
    return hashes


def restore_training_state(manager, core, snapshot: dict) -> dict:
    """Restore one accepted candidate boundary and return exactness evidence."""
    # high_level_state() intentionally returns detached CPU checkpoint tensors.
    # Use the policy's checked loader to write those tensors back to the live
    # FSDP parameters rather than copying into another detached snapshot.
    core.load_high_level_state(clone_tree(snapshot["parameters"]))
    manager.optimizer.load_state_dict(clone_tree(snapshot["optimizer"]))
    manager.lr_scheduler.load_state_dict(clone_tree(snapshot["scheduler"]))
    manager.grad_scaler.load_state_dict(clone_tree(snapshot["grad_scaler"]))
    manager.optimizer_steps = int(snapshot["optimizer_steps"])
    core.actor_version = int(snapshot["actor_version"])
    core.behavior_version = int(snapshot["behavior_version"])
    random.setstate(clone_tree(snapshot["rng"]["python"]))
    np.random.set_state(clone_tree(snapshot["rng"]["numpy"]))
    torch.set_rng_state(snapshot["rng"]["torch_cpu"].clone())
    torch.cuda.set_rng_state_all([
        state.clone() for state in snapshot["rng"]["torch_cuda"]
    ])
    manager.optimizer.zero_grad(set_to_none=True)
    restored = capture_training_state(manager, core)
    before_hashes = training_state_hashes(snapshot)
    after_hashes = training_state_hashes(restored)
    return {
        "before": before_hashes,
        "after": after_hashes,
        "component_equal": {
            name: after_hashes[name] == before_hashes[name] for name in before_hashes
        },
        "exact": before_hashes == after_hashes,
    }
