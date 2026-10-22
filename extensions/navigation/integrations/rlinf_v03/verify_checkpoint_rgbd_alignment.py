"""Independent CPU replay of checkpoint sampled actions/logits for RGB-D run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.pop("RAY_ADDRESS", None)

import torch

from eval_checkpoint_cpu import (
    EXPECTED_CHECKPOINT_SHA256,
    canonical_json,
    choose_action,
    load_policy,
    sha256_file,
)


ROOT = Path("/srv/navigation")
PRIOR = ROOT / "runs/pointnav-ppo-v03/cpu_eval5_64/actor_step1_sampled_seed7_with_poses.jsonl"
OUTPUT = ROOT / "runs/pointnav-ppo-v03/checkpoint-rgbd-ep4-sampled-seed7"
CHECKPOINT = (
    ROOT / "runs/pointnav-ppo-v03/pointnav-ppo-smoke/checkpoints/global_step_1"
    / "actor/model_state_dict/full_weights.pt"
)


def read_steps(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["event"] == "step"
    ]


def max_delta(first, second):
    return max(abs(float(a) - float(b)) for a, b in zip(first, second))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument(
        "--expected-checkpoint-sha256", default=EXPECTED_CHECKPOINT_SHA256,
    )
    parser.add_argument("--prior-trace", type=Path, default=PRIOR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--torch-seed", type=int, default=7)
    args = parser.parse_args()
    args.expected_checkpoint_sha256 = args.expected_checkpoint_sha256.lower()
    if (
        len(args.expected_checkpoint_sha256) != 64
        or any(c not in "0123456789abcdef" for c in args.expected_checkpoint_sha256)
        or args.torch_seed < 0
    ):
        parser.error("expected checkpoint SHA256 must be 64 hex characters and seed >= 0")
    torch.set_num_threads(1)
    policy, checkpoint_sha = load_policy(
        args.checkpoint, args.expected_checkpoint_sha256
    )
    torch.manual_seed(args.torch_seed)
    prior = read_steps(args.prior_trace)
    live = read_steps(args.output_dir / "policy_trace.jsonl")
    if len(prior) != 25 or len(live) != 6:
        raise RuntimeError("unexpected 25 prior or 6 live episode actions")
    max_prior_logit_delta = 0.0
    max_prior_logprob_delta = 0.0
    max_live_logit_delta = 0.0
    max_live_logprob_delta = 0.0
    for index, row in enumerate(prior):
        action, logits, logprob = choose_action(
            policy, row["pointgoal_before"], "sampled"
        )
        if action != row["action_id"]:
            raise RuntimeError(f"prior action mismatch at step stream index {index}")
        max_prior_logit_delta = max(
            max_prior_logit_delta, max_delta(logits, row["logits"])
        )
        max_prior_logprob_delta = max(
            max_prior_logprob_delta, abs(logprob - row["selected_logprob"])
        )
        if row["episode_index"] == 4:
            visual_row = live[index - 19]
            if action != visual_row["action_id"]:
                raise RuntimeError(f"live visual action mismatch at step {index - 19}")
            max_live_logit_delta = max(
                max_live_logit_delta, max_delta(logits, visual_row["logits"])
            )
            max_live_logprob_delta = max(
                max_live_logprob_delta,
                abs(logprob - visual_row["selected_logprob"]),
            )
    max_error = max(
        max_prior_logit_delta, max_prior_logprob_delta,
        max_live_logit_delta, max_live_logprob_delta,
    )
    result = {
        "schema_version": 1,
        "mode": "cpu_checkpoint_sampling_alignment",
        "checkpoint_sha256": checkpoint_sha,
        "torch_seed": args.torch_seed,
        "prior_actions_checked": 25,
        "prior_episode_0_to_3_actions_checked": 19,
        "live_visual_episode_4_actions_checked": 6,
        "max_prior_logit_absolute_delta": max_prior_logit_delta,
        "max_prior_selected_logprob_absolute_delta": max_prior_logprob_delta,
        "max_live_logit_absolute_delta": max_live_logit_delta,
        "max_live_selected_logprob_absolute_delta": max_live_logprob_delta,
        "all_actions_matched": True,
        "logits_and_logprobs_matched_within_1e-7": max_error <= 1e-7,
        "prior_trace_sha256": sha256_file(args.prior_trace),
        "live_policy_trace_sha256": sha256_file(args.output_dir / "policy_trace.jsonl"),
        "verifier_script_sha256": sha256_file(Path(__file__)),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "ray_started": False,
    }
    if not result["logits_and_logprobs_matched_within_1e-7"]:
        raise RuntimeError(canonical_json(result))
    path = args.output_dir / "policy_alignment.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(canonical_json(result))


if __name__ == "__main__":
    main()
