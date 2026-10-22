"""Check categorical PointNav action/logprob/value shapes and gradients."""

from __future__ import annotations

import torch
from omegaconf import OmegaConf

from rlinf.models.embodiment.mlp_policy import get_model


def main():
    torch.manual_seed(7)
    policy = get_model(
        OmegaConf.create(
            {
                "obs_dim": 2,
                "action_dim": 1,
                "num_action_chunks": 1,
                "add_value_head": True,
                "discrete_n_actions": 4,
            }
        ),
        torch_dtype=torch.float32,
    )
    obs = {"states": torch.tensor([[1.0, 0.2], [0.1, -0.1]])}
    actions, sample = policy.predict_action_batch(env_obs=obs, mode="train")
    assert actions.shape == (2, 1, 1) and actions.dtype == torch.int64
    assert sample["prev_logprobs"].shape == (2, 1)
    assert sample["prev_values"].shape == (2, 1)
    assert sample["forward_inputs"]["states"].shape == (2, 2)
    out = policy(
        forward_inputs=sample["forward_inputs"],
        compute_logprobs=True,
        compute_entropy=True,
        compute_values=True,
    )
    assert out["logprobs"].shape == (2, 1)
    assert out["entropy"].shape == (2, 1)
    assert out["values"].shape == (2, 1)
    assert torch.allclose(out["logprobs"], sample["prev_logprobs"], atol=1e-6)
    advantage = torch.tensor([[1.0], [-1.0]])
    returns = torch.tensor([[1.0], [0.0]])
    loss = -(out["logprobs"] * advantage).mean() + (
        (out["values"] - returns) ** 2
    ).mean()
    loss.backward()
    assert all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in policy.parameters()
    )
    eval_actions, _ = policy.predict_action_batch(env_obs=obs, mode="eval")
    assert eval_actions.shape == (2, 1, 1)
    print("categorical_policy_ok action=(2,1,1) logprob/value=(2,1) grad=finite")


if __name__ == "__main__":
    main()
