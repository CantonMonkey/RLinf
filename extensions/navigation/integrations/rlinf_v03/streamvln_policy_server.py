"""Isolated StreamVLN categorical actor and replay server.

Run this file with the validated StreamVLN Python environment. RLinf talks to
it over line-delimited STREAMRPC messages and never imports the older
transformers/LLaVA stack into its own process.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np
from PIL import Image
import torch

from streamvln_context import (
    ContextLedger,
    SessionKey,
    snapshot_from_wire,
    snapshot_to_wire,
)


WIRE_PREFIX = "STREAMRPC "


def source_commit(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
    )
    if result.returncode:
        raise RuntimeError("cannot resolve StreamVLN source commit")
    return result.stdout.strip()


def rgb_from_bytes(frame: bytes, shape) -> Image.Image:
    if (
        not isinstance(shape, list)
        or len(shape) != 3
        or any(type(value) is not int or value < 1 for value in shape)
        or shape[2] != 3
        or len(frame) != shape[0] * shape[1] * shape[2]
    ):
        raise ValueError("invalid raw HWC RGB payload")
    array = np.frombuffer(frame, dtype=np.uint8).reshape(shape)
    return Image.fromarray(array, mode="RGB")


@dataclass
class RuntimeSession:
    ledger: ContextLedger
    policy: object
    generator: torch.Generator


class StreamActor:
    def __init__(self, args) -> None:
        source = Path(args.stream_source).resolve(strict=True)
        experiments = source / "experiments" / "experience_memory"
        if not experiments.is_dir():
            raise ValueError("StreamVLN source has no experience_memory module")
        sys.path.insert(0, str(experiments))
        from navigation_policy import NavigationPolicy, load_policy

        self.NavigationPolicy = NavigationPolicy
        self.source = source
        self.commit = source_commit(source)
        if args.expected_source_commit and self.commit != args.expected_source_commit:
            raise ValueError(
                f"StreamVLN source commit {self.commit} differs from "
                f"{args.expected_source_commit}"
            )
        self.template = load_policy(
            args.base,
            args.tokenizer,
            args.vision_tower,
            checkpoint=None,
            device=args.device,
        )
        self.temperature = float(args.temperature)
        self.context_tokens = int(args.context_tokens)
        self.seed = int(args.seed)
        self.sessions: dict[int, RuntimeSession] = {}
        self.runtime_sha256 = hashlib.sha256(
            (experiments / "navigation_policy.py").read_bytes()
        ).hexdigest()

    def new_policy(self):
        return self.NavigationPolicy(
            self.template.model,
            self.template.tokenizer,
            adapter=None,
            temperature=self.temperature,
            context_tokens=self.context_tokens,
        )

    def ready(self) -> dict:
        return {
            "ready": True,
            "stream_source_commit": self.commit,
            "runtime_sha256": self.runtime_sha256,
            "action_token_ids": self.template.action_ids.tolist(),
            "action_names": list(self.template.action_names),
            "context_tokens": self.context_tokens,
            "temperature": self.temperature,
            "behavior_version": "frozen-base",
            "replay_differentiable": False,
        }

    @staticmethod
    def _observation(raw: dict):
        required = {
            "env_id",
            "episode_serial",
            "scene",
            "episode_id",
            "instruction",
            "image_shape",
            "rgb_b64",
            "step",
            "episode_start",
            "previous_actions",
        }
        if not isinstance(raw, dict) or set(raw) != required:
            raise ValueError("predict observation schema mismatch")
        if type(raw["episode_start"]) is not bool:
            raise TypeError("episode_start must be boolean")
        try:
            frame = base64.b64decode(raw["rgb_b64"], validate=True)
        except Exception as error:
            raise ValueError("invalid observation RGB base64") from error
        shape = tuple(raw["image_shape"])
        rgb_from_bytes(frame, list(shape))
        key = SessionKey(
            int(raw["env_id"]),
            int(raw["episode_serial"]),
            str(raw["scene"]),
            str(raw["episode_id"]),
        )
        step = int(raw["step"])
        previous = list(raw["previous_actions"])
        if any(type(value) is not int or not 0 <= value < 4 for value in previous):
            raise ValueError("previous actions must be categorical")
        return key, str(raw["instruction"]), shape, frame, step, previous

    def _begin(self, key, instruction, shape) -> RuntimeSession:
        policy = self.new_policy()
        policy.begin_task(key.scene, instruction, carry=False)
        generator = torch.Generator(device=policy.device).manual_seed(
            self.seed + key.env_id * 1000003 + key.episode_serial
        )
        runtime = RuntimeSession(
            ContextLedger(key, instruction, shape),
            policy,
            generator,
        )
        self.sessions[key.env_id] = runtime
        return runtime

    def predict_one(self, raw: dict, *, greedy: bool) -> dict:
        key, instruction, shape, frame, step, previous = self._observation(raw)
        runtime = self.sessions.get(key.env_id)
        if raw["episode_start"]:
            if step != 0 or previous:
                raise ValueError("episode start requires step zero and no previous actions")
            if runtime is None or runtime.ledger.key != key:
                runtime = self._begin(key, instruction, shape)
        elif runtime is None or runtime.ledger.key != key:
            raise ValueError("observation has no matching active environment session")
        if runtime.ledger.instruction != instruction:
            raise ValueError("instruction changed inside an episode")

        response_key = runtime.ledger.response_key(step, frame)
        if runtime.ledger.last_response_key == response_key:
            return dict(runtime.ledger.last_response)
        if raw["episode_start"] and runtime.ledger.frames:
            raise ValueError("episode-start observation changed after first prediction")

        if step:
            runtime.ledger.acknowledge(previous)
            runtime.policy.acknowledge(
                runtime.policy_pending,
                previous,
            )
        elif previous:
            raise ValueError("step zero cannot acknowledge previous actions")

        runtime.ledger.record_observation(step, frame, shape)
        feature = runtime.policy.encode_observation(
            rgb_from_bytes(frame, list(shape))
        )
        decision = runtime.policy.sample(
            step,
            feature,
            generator=runtime.generator,
            greedy=greedy,
        )
        actions = list(decision["proposed_actions"])
        runtime.ledger.set_proposal(actions)
        runtime.policy_pending = actions
        response = {
            "env_id": key.env_id,
            "episode_serial": key.episode_serial,
            "step": step,
            "actions": actions,
            "behavior_log_probs": list(decision["behavior_log_probs"]),
            "replay": snapshot_to_wire(runtime.ledger.snapshot()),
        }
        runtime.ledger.last_response_key = response_key
        runtime.ledger.last_response = dict(response)
        return response

    def _encode_frame(self, policy, frame, shape):
        return policy.encode_observation(rgb_from_bytes(frame, shape))

    def replay_one(self, wire: dict) -> dict:
        snapshot = snapshot_from_wire(wire)
        policy = self.new_policy()
        policy.begin_task(
            snapshot["scene"],
            snapshot["instruction"],
            carry=False,
        )
        block_start = int(snapshot["block_start"])
        step = int(snapshot["step"])
        shape = snapshot["image_shape"]
        if block_start:
            history_steps = snapshot["history_steps"]
            if len(history_steps) != 8:
                raise ValueError("replay requires eight historical frames")
            policy.actions = [0] * block_start
            for history_step, frame in zip(
                history_steps,
                snapshot["history_frames"],
            ):
                policy.features[int(history_step)] = self._encode_frame(
                    policy,
                    frame,
                    shape,
                )
        block_actions = list(snapshot["block_actions"])
        block_frames = snapshot["block_frames"]
        for index, frame in enumerate(block_frames):
            decision_step = block_start + index * 4
            feature = self._encode_frame(policy, frame, shape)
            policy.prepare_observation(decision_step, feature)
            is_current = index == len(block_frames) - 1
            if is_current:
                actions = list(snapshot["current_actions"])
            else:
                offset = index * 4
                actions = block_actions[offset : offset + 4]
                if len(actions) != 4:
                    raise ValueError("prior block decision is not a full chunk")
            for action in actions:
                token = int(policy.action_ids[action])
                policy.append_action_token(token)
            policy.close_decision()
            if not is_current:
                policy.acknowledge(actions, actions)
        current, entropy = policy.block_log_probs()
        count = len(snapshot["current_actions"])
        return {
            "env_id": int(snapshot["env_id"]),
            "episode_serial": int(snapshot["episode_serial"]),
            "step": step,
            "log_probs": current[-count:].float().cpu().tolist(),
            "entropy": entropy[-count:].float().cpu().tolist(),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream-source", required=True)
    parser.add_argument("--expected-source-commit")
    parser.add_argument("--base", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--vision-tower", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--context-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()

    wire = sys.stdout
    try:
        actor = StreamActor(args)
        wire.write(WIRE_PREFIX + json.dumps(actor.ready()) + "\n")
        wire.flush()
        for line in sys.stdin:
            request = json.loads(line)
            request_id = request.get("request_id")
            try:
                operation = request["operation"]
                if operation == "predict":
                    mode = request.get("mode")
                    if mode not in ("train", "eval"):
                        raise ValueError("predict mode must be train or eval")
                    result = {
                        "request_id": request_id,
                        "results": [
                            actor.predict_one(row, greedy=mode == "eval")
                            for row in request["observations"]
                        ],
                    }
                elif operation == "replay":
                    result = {
                        "request_id": request_id,
                        "results": [
                            actor.replay_one(row)
                            for row in request["rows"]
                        ],
                    }
                elif operation == "close":
                    result = {"request_id": request_id, "closed": True}
                    wire.write(WIRE_PREFIX + json.dumps(result) + "\n")
                    wire.flush()
                    break
                else:
                    raise ValueError(f"unknown operation {operation}")
            except BaseException as error:
                result = {
                    "request_id": request_id,
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                }
            wire.write(WIRE_PREFIX + json.dumps(result) + "\n")
            wire.flush()
    except BaseException as error:
        wire.write(WIRE_PREFIX + json.dumps({
            "ready": False,
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }) + "\n")
        wire.flush()
        raise


if __name__ == "__main__":
    main()