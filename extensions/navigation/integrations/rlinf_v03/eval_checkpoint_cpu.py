# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluate one RLinf PointNav checkpoint on official val episodes on CPU.

The policy sees only PointGoal. Habitat runs in the existing Python 3.9
no-renderer stdio service. This script does not initialize Ray or CUDA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import select
import subprocess
from collections import Counter
from pathlib import Path

# Set this before importing torch or RLinf, including when called incorrectly
# from a shell with a GPU in CUDA_VISIBLE_DEVICES.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.pop("RAY_ADDRESS", None)

import torch
from omegaconf import OmegaConf

from rlinf.models.embodiment.mlp_policy import get_model

ROOT = Path("/srv/navigation")
EXPECTED_CHECKPOINT_SHA256 = (
    "80c1e8ff3296fdf6f6b4dd32a47efc4a85301152429f1ce475437f33a88cef7c"
)
RLINF_BASE_COMMIT = "0505431899574619da86f551bad70b71e0ea2177"
ACTIONS = ("stop", "move_forward", "turn_left", "turn_right")
POLICY_CONFIG = {
    "obs_dim": 2,
    "action_dim": 1,
    "num_action_chunks": 1,
    "add_value_head": True,
    "discrete_n_actions": 4,
}


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_policy(path: Path, expected_sha256: str = EXPECTED_CHECKPOINT_SHA256):
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise ValueError(f"Unexpected checkpoint SHA256: {digest}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or tuple(state["actor_mean.weight"].shape) != (4, 256):
        raise ValueError("Checkpoint is not the four-action PointNav actor")
    policy = get_model(OmegaConf.create(POLICY_CONFIG), torch_dtype=torch.float32)
    policy.load_state_dict(state, strict=True)
    policy.to("cpu").eval()
    return policy, digest


def choose_action(policy, pointgoal: list[float], mode: str) -> tuple[int, list[float], float]:
    if len(pointgoal) != 2 or not all(math.isfinite(float(x)) for x in pointgoal):
        raise ValueError("Invalid PointGoal observation")
    states = torch.tensor([pointgoal], dtype=torch.float32, device="cpu")
    with torch.inference_mode():
        actions, sample = policy.predict_action_batch(
            env_obs={"states": states}, mode="eval" if mode == "argmax" else "train"
        )
        logits = policy.actor_mean(policy.backbone(states)).float()
    if tuple(actions.shape) != (1, 1, 1) or actions.dtype != torch.int64:
        raise ValueError(f"Unexpected categorical action shape/type: {actions.shape}, {actions.dtype}")
    action = int(actions[0, 0, 0].item())
    if not 0 <= action < 4:
        raise ValueError("Policy emitted an invalid action")
    if mode == "argmax" and action != int(logits.argmax(dim=-1).item()):
        raise ValueError("Policy eval action does not match four-way argmax")
    return (
        action,
        [float(x) for x in logits[0].tolist()],
        float(sample["prev_logprobs"][0, 0].item()),
    )


class PointNavProtocol:
    def __init__(self, habitat_python: Path, service_script: Path, asset_root: Path,
                 episode_indices: list[int] | None = None, max_steps: int = 64):
        if episode_indices is None:
            episode_indices = list(range(5))
        if not episode_indices or max_steps <= 0:
            raise ValueError("PointNav protocol requires episodes and a positive action budget")
        expected_dataset = (
            asset_root.resolve() / "datasets/pointnav/habitat-test-scenes/v1/val/val.json.gz"
        )
        child_env = os.environ.copy()
        child_env.pop("PYTHONPATH", None)
        child_env.pop("RAY_ADDRESS", None)
        child_env["CUDA_VISIBLE_DEVICES"] = ""
        self.process = subprocess.Popen(
            [
                str(habitat_python),
                "-B",
                str(service_script),
                "--asset-root",
                str(asset_root),
                "--split",
                "val",
                "--max-steps",
                str(max_steps),
                "--episode-indices",
                ",".join(map(str, episode_indices)),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
            env=child_env,
        )
        try:
            ready = self._read(timeout=120)
            if (not ready.get("ready") or tuple(ready.get("actions", ())) != ACTIONS
                    or ready.get("split") != "val"
                    or Path(ready.get("dataset_path", "")).resolve() != expected_dataset
                    or ready.get("selected_indices") != episode_indices
                    or not isinstance(ready.get("episode_ids"), list)
                    or len(ready["episode_ids"]) != len(episode_indices)
                    or not ready.get("scene_id")):
                raise RuntimeError(f"PointNav service protocol mismatch: {ready}")
            self.ready_episode_ids = [str(value) for value in ready["episode_ids"]]
            self.ready_scene_id = str(ready["scene_id"])
        except BaseException:
            self.close()
            raise

    def _read(self, timeout: int = 60) -> dict:
        readable, _, _ = select.select([self.process.stdout], [], [], timeout)
        if not readable:
            raise TimeoutError("PointNav service response timed out")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"PointNav service exited: {self.process.poll()}")
        result = json.loads(line)
        if "error" in result:
            raise RuntimeError(f"PointNav service error: {result['error']}")
        return result

    def request(self, payload: dict) -> dict:
        if self.process.poll() is not None:
            raise RuntimeError("PointNav service is not running")
        self.process.stdin.write(canonical_json(payload) + "\n")
        self.process.stdin.flush()
        return self._read()

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self.process.stdin.write('{"cmd":"close"}\n')
                self.process.stdin.flush()
                self.process.wait(timeout=5)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
        self.process.stdin.close()
        self.process.stdout.close()


def load_baseline(trace_path: Path, summary_path: Path,
                  episode_start: int = 0, episodes: int = 5,
                  max_steps: int = 64) -> tuple[dict, dict]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["trace_sha256"] != sha256_file(trace_path):
        raise ValueError("Random baseline trace SHA256 mismatch")
    if (summary["episode_start"], summary["episodes"], summary["max_steps"]) != (
        episode_start, episodes, max_steps
    ):
        raise ValueError("Random baseline has a different episode or action budget")
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    if [row["episode_index"] for row in rows] != list(range(episode_start, episode_start + episodes)):
        raise ValueError("Random baseline episode indices do not match requested range")
    return summary, {row["episode_index"]: row for row in rows}


def same_initial_state(reset: dict, baseline: dict) -> bool:
    info = reset["info"]
    if str(info["episode_id"]) != str(baseline["episode_id"]):
        return False
    if str(info["scene_id"]) != str(baseline["scene_id"]):
        return False
    if not math.isclose(
        float(info["geodesic_distance"]),
        float(baseline["initial_metrics"]["distance_to_goal"]),
        abs_tol=1e-3,
    ):
        return False
    return all(
        math.isclose(float(a), float(b), abs_tol=1e-3)
        for a, b in zip(reset["obs"], baseline["initial_pointgoal"])
    )


def request_diagnostic(protocol: PointNavProtocol, episode_id: str, scene_id: str) -> dict:
    diagnostic = protocol.request({"cmd": "diagnostic"})
    if (diagnostic["episode_id"], diagnostic["scene_id"]) != (episode_id, scene_id):
        raise ValueError("PointNav diagnostic changed episode or scene")
    return diagnostic


def evaluate(args: argparse.Namespace) -> dict:
    torch.set_num_threads(1)
    revision = subprocess.check_output(
        ["git", "-C", str(args.node_root / "repos/RLinf-v0.3"), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if revision != RLINF_BASE_COMMIT:
        raise ValueError(f"RLinf checkout changed from pinned v0.3: {revision}")
    policy, checkpoint_sha = load_policy(args.checkpoint, args.expected_checkpoint_sha256)
    if args.mode == "sampled":
        # Seed after model construction/loading so initialization does not
        # consume the sampling stream for this fixed checkpoint.
        torch.manual_seed(args.torch_seed)
    baseline_summary, baseline_rows = load_baseline(
        args.baseline_trace, args.baseline_summary,
        args.episode_start, args.episodes, args.max_steps,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp_output = args.output.with_name(args.output.name + ".tmp")
    episodes = []
    episode_indices = list(range(args.episode_start, args.episode_start + args.episodes))
    protocol = PointNavProtocol(
        args.habitat_python, args.service_script, args.asset_root,
        episode_indices=episode_indices, max_steps=args.max_steps,
    )
    try:
        with temp_output.open("w", encoding="utf-8", newline="\n") as stream:
            for offset, episode_index in enumerate(episode_indices):
                reset = protocol.request({"cmd": "reset"})
                baseline = baseline_rows[episode_index]
                initial_match = same_initial_state(reset, baseline)
                episode_id = str(reset["info"]["episode_id"])
                scene_id = str(reset["info"]["scene_id"])
                if (episode_id != protocol.ready_episode_ids[offset]
                        or scene_id != protocol.ready_scene_id
                        or not initial_match):
                    raise ValueError(f"Episode or initial state mismatch at index {episode_index}")
                pointgoal = reset["obs"]
                reset_row = {
                    "event": "reset", "episode_index": episode_index,
                    "episode_id": episode_id, "scene_id": scene_id,
                    "pointgoal": pointgoal, "info": reset["info"],
                    "random_baseline_initial_state_match": initial_match,
                }
                if args.record_poses:
                    reset_row["diagnostic"] = request_diagnostic(protocol, episode_id, scene_id)
                stream.write(canonical_json(reset_row) + "\n")
                reward_sum = 0.0
                counts = Counter()
                for step in range(1, args.max_steps + 1):
                    action, logits, logprob = choose_action(policy, pointgoal, args.mode)
                    transition = protocol.request({"cmd": "step", "action": action})
                    info = transition["info"]
                    if str(info["episode_id"]) != episode_id or str(info["scene_id"]) != scene_id:
                        raise ValueError("Habitat episode changed mid-rollout")
                    terminated = bool(transition["terminated"])
                    truncated = bool(transition["truncated"])
                    if terminated != (action == 0) or (terminated and truncated):
                        raise ValueError("STOP/terminal contract changed")
                    if bool(info["success"]) and not (
                        terminated and float(info["distance_to_goal"]) < 0.2
                    ):
                        raise ValueError("Habitat success did not follow strict STOP threshold")
                    if step == args.max_steps and not (terminated or truncated):
                        raise ValueError("PointNav action budget was not enforced")
                    reward = float(transition["reward"])
                    reward_sum += reward
                    counts[ACTIONS[action]] += 1
                    step_row = {
                        "event": "step", "episode_index": episode_index,
                        "episode_id": episode_id, "scene_id": scene_id,
                        "step": step, "pointgoal_before": pointgoal,
                        "action_id": action, "action": ACTIONS[action],
                        "logits": logits, "selected_logprob": logprob,
                        "pointgoal_after": transition["obs"],
                        "reward": reward, "terminated": terminated,
                        "truncated": truncated, "info": info,
                    }
                    if args.record_poses:
                        step_row["diagnostic"] = request_diagnostic(protocol, episode_id, scene_id)
                    stream.write(canonical_json(step_row) + "\n")
                    pointgoal = transition["obs"]
                    if terminated or truncated:
                        break
                else:
                    raise ValueError("Episode did not terminate within the action budget")
                episode_result = {
                    "episode_index": episode_index,
                    "episode_id": episode_id,
                    "scene_id": scene_id,
                    "initial_pointgoal": reset["obs"],
                    "initial_geodesic_distance": float(reset["info"]["geodesic_distance"]),
                    "random_baseline_initial_state_match": initial_match,
                    "steps": step,
                    "action_counts": dict(counts),
                    "stop_called": bool(info["stop_called"]),
                    "termination_reason": info["termination_reason"],
                    "success": bool(info["success"]),
                    "spl": float(info["spl"]),
                    "final_geodesic_distance": float(info["geodesic_distance"]),
                    "final_euclidean_distance": float(info["distance_to_goal"]),
                    "path_length": float(info["path_length"]),
                    "collision_count": int(info["collision_count"]),
                    "reward_sum": reward_sum,
                    "random_baseline": baseline["outcome"],
                }
                episodes.append(episode_result)
                stream.flush()
        temp_output.replace(args.output)
    finally:
        protocol.close()
    paired = all(ep["random_baseline_initial_state_match"] for ep in episodes)
    summary = {
        "status": "ok", "evaluation_type": f"{args.mode} checkpoint diagnostic",
        "interpretation": "official PointNav val episodes; verify disjointness against the checkpoint's selected training episodes",
        "dataset": "official habitat-test-scenes PointNav v1 val",
        "simulator": "Habitat-Lab 0.2.4 / Habitat-Sim 0.2.4, no renderer",
        "episode_indices": episode_indices, "max_steps_per_episode": args.max_steps,
        "checkpoint_path": str(args.checkpoint), "checkpoint_sha256": checkpoint_sha,
        "expected_checkpoint_sha256": args.expected_checkpoint_sha256,
        "rlinf_base_commit": revision,
        "policy_config": POLICY_CONFIG,
        "policy_source_sha256": sha256_file(args.policy_source),
        "eval_script_sha256": sha256_file(Path(__file__).resolve()),
        "action_selection": (
            "MLPPolicy.predict_action_batch(mode=eval), four-logit argmax"
            if args.mode == "argmax"
            else "MLPPolicy.predict_action_batch(mode=train), seeded Categorical sample"
        ),
        "sampling_torch_seed": args.torch_seed if args.mode == "sampled" else None,
        "action_names": ACTIONS,
        "habitat_service_script": str(args.service_script),
        "habitat_service_sha256": sha256_file(args.service_script),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "ray_started": False, "renderer_created": False,
        "privileged_geometry_recorded": bool(args.record_poses),
        "privileged_geometry_used_for_action_selection": False,
        "episodes": episodes,
        "success_count": sum(int(ep["success"]) for ep in episodes),
        "mean_spl": sum(ep["spl"] for ep in episodes) / args.episodes,
        "mean_final_geodesic_distance": sum(ep["final_geodesic_distance"] for ep in episodes) / args.episodes,
        "total_steps": sum(ep["steps"] for ep in episodes),
        "stop_count": sum(int(ep["stop_called"]) for ep in episodes),
        "baseline_trace_sha256": baseline_summary["trace_sha256"],
        "baseline_summary_path": str(args.baseline_summary),
        "baseline_initial_states_all_match": paired,
        "baseline_success_count": baseline_summary["success_count"],
        "baseline_mean_spl": baseline_summary["mean_spl"],
        "baseline_mean_final_geodesic_distance": baseline_summary["mean_final_distance"],
        "trace_path": str(args.output), "trace_sha256": sha256_file(args.output),
    }
    if paired:
        summary["paired_final_geodesic_deltas_policy_minus_random"] = [
            ep["final_geodesic_distance"] - ep["random_baseline"]["distance_to_goal"]
            for ep in episodes
        ]
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(canonical_json(summary) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-root", type=Path, default=ROOT)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--expected-checkpoint-sha256", default=EXPECTED_CHECKPOINT_SHA256,
                        help="Required digest for a custom checkpoint; defaults to the original step-1 fixture")
    parser.add_argument("--baseline-trace", type=Path)
    parser.add_argument("--baseline-summary", type=Path)
    parser.add_argument("--habitat-python", type=Path, default=Path("/opt/conda/envs/streamvln/bin/python"))
    parser.add_argument("--service-script", type=Path)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--policy-source", type=Path)
    parser.add_argument("--mode", choices=("argmax", "sampled"), default="argmax")
    parser.add_argument("--torch-seed", type=int, default=7)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--record-poses", action="store_true", help="Request privileged pose/goal diagnostics only for offline rendering")
    parser.add_argument("--model-smoke-only", action="store_true")
    args = parser.parse_args()
    if not (0 <= args.episode_start < 100 and 1 <= args.episodes <= 100
            and args.episode_start + args.episodes <= 100 and 1 <= args.max_steps <= 512):
        parser.error("requested official val range or action budget is invalid")
    if (args.episode_start, args.episodes, args.max_steps) != (0, 5, 64):
        if args.baseline_trace is None or args.baseline_summary is None or args.output is None:
            parser.error("a custom val range requires explicit baseline trace, baseline summary, and output")
    args.expected_checkpoint_sha256 = args.expected_checkpoint_sha256.lower()
    if len(args.expected_checkpoint_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in args.expected_checkpoint_sha256
    ):
        parser.error("--expected-checkpoint-sha256 must be 64 hex characters")
    args.node_root = args.node_root.resolve()
    args.checkpoint = args.checkpoint or (
        args.node_root / "runs/pointnav-ppo-v03/pointnav-ppo-smoke/checkpoints/global_step_1"
        / "actor/model_state_dict/full_weights.pt"
    )
    args.baseline_trace = args.baseline_trace or args.node_root / "runs/habitat_pointnav_random_seed7_eval5_max64.jsonl"
    args.baseline_summary = args.baseline_summary or args.node_root / "runs/habitat_pointnav_random_seed7_eval5_max64.summary.json"
    args.service_script = args.service_script or args.node_root / "repos/rlinf-nav/scripts/pointnav_service.py"
    args.asset_root = args.asset_root or args.node_root / "data"
    args.policy_source = args.policy_source or args.node_root / "repos/RLinf-v0.3/rlinf/models/embodiment/mlp_policy/mlp_policy.py"
    if args.output is None:
        name = (
            "actor_step1_deterministic.jsonl"
            if args.mode == "argmax"
            else f"actor_step1_sampled_seed{args.torch_seed}.jsonl"
        )
        args.output = args.node_root / "runs/pointnav-ppo-v03/cpu_eval5_64" / name
    if args.model_smoke_only:
        torch.set_num_threads(1)
        policy, digest = load_policy(args.checkpoint, args.expected_checkpoint_sha256)
        if args.mode == "sampled":
            torch.manual_seed(args.torch_seed)
        action, logits, logprob = choose_action(
            policy, [5.643342971801758, 2.881923198699951], args.mode
        )
        print(canonical_json({"checkpoint_sha256": digest, "action": action, "logits": logits, "logprob": logprob, "cpu_only": True}))
    else:
        summary = evaluate(args)
        print(canonical_json({k: v for k, v in summary.items() if k != "episodes"}))


if __name__ == "__main__":
    main()
