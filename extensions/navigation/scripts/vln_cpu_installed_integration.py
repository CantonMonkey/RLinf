"""Bounded CPU integration smoke for the installed private RLinf R2R path.

This exercises the installed Hydra config, model and environment registries,
EnvOutput transport boundary, the real HF rollout predict implementation,
PPO replay/backward, and one action in the selected real R2R episode. It does
not initialize Ray, FSDP, CUDA, or update a checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace


PROJECT = Path("/srv/navigation")
CHECKOUT = PROJECT / "repos/RLinf-v0.3"
CONFIG_DIR = CHECKOUT / "examples/embodiment/config"
OBS_KEYS = {"rgb", "instruction_ids", "instruction_mask"}
ACTIONS = ("stop", "move_forward", "turn_left", "turn_right")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    # Hide CUDA before importing torch or the installed RLinf package.
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to the empty string")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    import torch
    from hydra import compose, initialize_config_dir
    from rlinf.config import SupportedModel
    from rlinf.data.embodied_io_struct import EnvOutput
    from rlinf.envs import get_env_cls
    from rlinf.models import get_model
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

    torch.set_num_threads(1)
    assert not torch.cuda.is_available()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.1"):
        cfg = compose(config_name="vln_r2r_ppo_smoke")
    # RLinf validate_cfg() starts a local Ray cluster as a side effect. Keep
    # this process CPU-only and validate the fields this bounded path uses.
    assert cfg.runner.task_type == "embodied"
    assert cfg.actor.training_backend == "fsdp"
    assert cfg.rollout.backend == "huggingface"
    assert cfg.env.train.env_type == "vln"
    assert cfg.env.train.dataset_split == "train"
    assert cfg.env.train.episode_index == 4852
    assert cfg.env.eval.dataset_split == "val_unseen"
    assert cfg.env.eval.episode_index == 630
    assert SupportedModel(cfg.actor.model.model_type) is SupportedModel.VLN_POLICY

    env_cls = get_env_cls(cfg.env.train.env_type, cfg.env.train)
    assert env_cls.__module__ == "rlinf.envs.vln.vln_env"
    model = get_model(cfg.actor.model)
    assert model is not None and model.__class__.__name__ == "VLNPolicy"
    assert next(model.parameters()).device.type == "cpu"

    env = env_cls(cfg.env.train, 1, 0, 1, None)
    process = env.process
    try:
        obs, identity = env.reset()
        assert set(obs) == OBS_KEYS
        assert tuple(obs["rgb"].shape) == (1, 64, 64, 3)
        assert obs["rgb"].dtype == torch.uint8
        assert tuple(obs["instruction_ids"].shape) == (1, 200)
        assert obs["instruction_ids"].dtype == torch.int64
        assert tuple(obs["instruction_mask"].shape) == (1, 200)
        assert obs["instruction_mask"].dtype == torch.bool
        assert identity["episode_id"] == env.identity["episode_id"]

        # This is the installed EnvWorker transport boundary, not a local copy.
        transported = EnvOutput(obs=obs).to_dict()
        assert set(transported["obs"]) == OBS_KEYS
        assert all(torch.equal(obs[key], transported["obs"][key]) for key in OBS_KEYS)
        assert transported["final_obs"] is None

        # Run the installed HuggingFace rollout worker's actual predict body.
        # Bypass only Worker.timer because a standalone smoke has no Ray worker
        # lifecycle or profiling fields.
        hf_worker = SimpleNamespace(
            model_cfg=cfg.actor.model,
            algorithm_cfg=cfg.algorithm,
            _train_sampling_params={},
            _eval_sampling_params={},
            hf_model=model,
            expert_model=None,
        )
        predict = inspect.unwrap(MultiStepRolloutWorker.predict)
        action, behavior = predict(hf_worker, transported["obs"], mode="train")
        assert tuple(action.shape) == (1, 1, 1)
        index = int(action.reshape(-1)[0])
        assert 0 <= index < len(ACTIONS)
        assert set(behavior["forward_inputs"]) == OBS_KEYS | {"action", "model_action"}
        assert all(
            torch.equal(obs[key], behavior["forward_inputs"][key])
            for key in OBS_KEYS
        )
        assert tuple(behavior["prev_logprobs"].shape) == (1, 1)
        assert tuple(behavior["prev_values"].shape) == (1, 1)

        model.train()
        replay = model(
            forward_inputs=behavior["forward_inputs"],
            compute_logprobs=True,
            compute_entropy=True,
            compute_values=True,
        )
        assert torch.allclose(replay["logprobs"], behavior["prev_logprobs"], atol=1e-6)
        assert torch.allclose(replay["values"], behavior["prev_values"], atol=1e-6)
        loss = -replay["logprobs"].mean() + 0.1 * replay["values"].square().mean()
        assert math.isfinite(float(loss.detach()))
        loss.backward()
        grad_count = sum(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all().item()
            and parameter.grad.abs().sum().item() > 0
            for parameter in model.parameters()
        )
        assert grad_count > 0

        step_obs, rewards, terminated, truncated, infos = env.chunk_step(action.cpu().numpy())
        assert len(step_obs) == len(infos) == 1
        assert set(step_obs[0]) == OBS_KEYS
        assert tuple(rewards.shape) == (1, 1)
        assert tuple(terminated.shape) == (1, 1)
        assert tuple(truncated.shape) == (1, 1)
        assert math.isfinite(float(rewards[0, 0]))
        transported_next = EnvOutput(obs=step_obs[0]).to_dict()
        assert set(transported_next["obs"]) == OBS_KEYS

        result = {
            "status": "installed_cpu_vln_integration_ok",
            "checks": [
                "hydra_compose_targeted_static_validation",
                "installed_vln_model_env_registry",
                "real_r2r_habitat_reset",
                "installed_envoutput_exact_observation",
                "installed_hf_rollout_predict_body",
                "behavior_logprob_value_replay",
                "finite_cpu_backward",
                "real_r2r_policy_chunk_step",
                "child_close",
            ],
            "mode": "cpu_only_no_ray_no_fsdp",
            "split": cfg.env.train.dataset_split,
            "episode_index": int(cfg.env.train.episode_index),
            "episode_id": identity["episode_id"],
            "scene_id": identity["scene_id"],
            "dataset_sha256": env.identity["dataset_sha256"],
            "observation_keys": sorted(obs),
            "rgb_shape": list(obs["rgb"].shape),
            "instruction_shape": list(obs["instruction_ids"].shape),
            "instruction_nonpad": int(obs["instruction_mask"].sum()),
            "action": index,
            "action_name": ACTIONS[index],
            "reward": float(rewards[0, 0]),
            "terminated": bool(terminated[0, 0]),
            "truncated": bool(truncated[0, 0]),
            "behavior_logprob": float(behavior["prev_logprobs"][0, 0]),
            "replayed_logprob": float(replay["logprobs"].detach()[0, 0]),
            "loss": float(loss.detach()),
            "nonzero_finite_grad_parameter_tensors": grad_count,
            "installed_sha256": {
                "rlinf/config.py": file_sha256(CHECKOUT / "rlinf/config.py"),
                "rlinf/envs/__init__.py": file_sha256(CHECKOUT / "rlinf/envs/__init__.py"),
                "rlinf/envs/vln/vln_env.py": file_sha256(CHECKOUT / "rlinf/envs/vln/vln_env.py"),
                "rlinf/models/embodiment/vln_policy.py": file_sha256(CHECKOUT / "rlinf/models/embodiment/vln_policy.py"),
                "rlinf/data/embodied_io_struct.py": file_sha256(CHECKOUT / "rlinf/data/embodied_io_struct.py"),
                "rlinf/workers/rollout/hf/huggingface_worker.py": file_sha256(
                    CHECKOUT / "rlinf/workers/rollout/hf/huggingface_worker.py"
                ),
            },
        }
    finally:
        env.close()
    assert process.poll() is not None
    result["service_exit_code"] = process.returncode
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
