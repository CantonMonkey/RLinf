"""Drive one RGB-D Habitat episode with a SHA-pinned RLinf PointGoal checkpoint.

The Torch/RLinf process is CPU-only Python 3.10; the simulator is a separate
private Habitat Python 3.9 process. Only PointGoal reaches action selection.
For a selected episode in a recorded no-render evaluation, earlier episodes
advance and verify the exact Torch sampling stream from their PointGoal
observations. The selected episode itself is stepped live in the visual
simulator.
The original step-1 checkpoint remains the default fixture; a CPU PPO
checkpoint may be supplied with its exact expected SHA256 and matching prior
no-render evaluation trace.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import select
import subprocess
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.pop("RAY_ADDRESS", None)

import torch

from eval_checkpoint_cpu import (
    ACTIONS,
    EXPECTED_CHECKPOINT_SHA256,
    POLICY_CONFIG,
    RLINF_BASE_COMMIT,
    canonical_json,
    choose_action,
    load_policy,
    sha256_file,
)


ROOT = Path("/srv/navigation")


def read_previous(trace_path: Path, summary_path: Path, episode_index: int):
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    observed_sha = sha256_file(trace_path)
    if summary["trace_sha256"] != observed_sha:
        raise ValueError("prior no-render PointNav trace hash mismatch")
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    indices = [int(i) for i in summary["episode_indices"]]
    if (not indices or indices != list(range(indices[0], indices[0] + len(indices)))
            or episode_index not in indices):
        raise ValueError("selected episode must belong to a consecutive prior val range")
    if [r["episode_index"] for r in rows if r["event"] == "reset"] != indices:
        raise ValueError("prior trace reset episodes do not match its summary")
    for r in rows:
        if r["event"] == "step" and not 0 <= r["action_id"] < 4:
            raise ValueError("prior trace has invalid action")
    return rows, summary, observed_sha


def warm_sampling_stream(policy, previous_rows, episode_index: int):
    checked = 0
    for row in previous_rows:
        if row["event"] != "step" or row["episode_index"] >= episode_index:
            continue
        action, _, _ = choose_action(policy, row["pointgoal_before"], "sampled")
        if action != row["action_id"]:
            raise RuntimeError(
                f"prior sampling stream mismatch at episode {row['episode_index']} "
                f"step {row['step']}: got {action}, expected {row['action_id']}"
            )
        checked += 1
    return checked


class VisualProtocol:
    def __init__(self, args):
        child_env = os.environ.copy()
        child_env.pop("RAY_ADDRESS", None)
        child_env["PATH"] = f"{args.habitat_env / 'bin'}:/usr/bin:/bin"
        child_env["PYTHONPATH"] = str(args.habitat_sim_source)
        child_env["LD_LIBRARY_PATH"] = str(args.habitat_env / "lib")
        child_env["CUDA_VISIBLE_DEVICES"] = ""
        child_env["LIBGL_ALWAYS_SOFTWARE"] = "1"
        child_env["EGL_PLATFORM"] = "surfaceless"
        child_env["HABITAT_SIM_SOFTWARE_EGL"] = "1"
        child_env["__EGL_VENDOR_LIBRARY_FILENAMES"] = (
            "/usr/share/glvnd/egl_vendor.d/50_mesa.json"
        )
        child_env["PYTHONDONTWRITEBYTECODE"] = "1"
        self.process = subprocess.Popen(
            [
                str(args.habitat_env / "bin/python"), "-B", str(args.service_script),
                "--asset-root", str(args.asset_root),
                "--output-dir", str(args.output_dir),
                "--episode-index", str(args.episode_index),
                "--max-steps", "64", "--fps", "6",
            ],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
            text=True, bufsize=1, env=child_env,
        )
        try:
            ready = self._read(timeout=120)
            if not ready.get("ready") or tuple(ready.get("actions", ())) != ACTIONS:
                raise RuntimeError(f"visual service protocol mismatch: {ready}")
        except BaseException:
            self.close()
            raise

    def _read(self, timeout=60):
        readable, _, _ = select.select([self.process.stdout], [], [], timeout)
        if not readable:
            raise TimeoutError("visual PointNav service response timed out")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"visual PointNav service exited: {self.process.poll()}")
        result = json.loads(line)
        if "error" in result:
            raise RuntimeError(f"visual PointNav service error: {result['error']}")
        return result

    def request(self, value, timeout=60):
        if self.process.poll() is not None:
            raise RuntimeError(f"visual service exited: {self.process.returncode}")
        self.process.stdin.write(canonical_json(value) + "\n")
        self.process.stdin.flush()
        return self._read(timeout)

    def close(self):
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


def close_float(a, b, atol=1e-5):
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=atol)


def evaluate(args):
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    torch.set_num_threads(1)
    revision = subprocess.check_output(
        ["git", "-C", str(args.node_root / "repos/RLinf-v0.3"), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if revision != RLINF_BASE_COMMIT:
        raise RuntimeError(f"unexpected RLinf v0.3 base revision: {revision}")
    policy, checkpoint_sha = load_policy(args.checkpoint, args.expected_checkpoint_sha256)
    torch.manual_seed(args.torch_seed)
    previous_rows, previous_summary, previous_sha = read_previous(
        args.previous_trace, args.previous_summary, args.episode_index
    )
    if previous_summary.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError("prior no-render evaluation used a different checkpoint")
    if previous_summary.get("sampling_torch_seed") != args.torch_seed:
        raise ValueError("prior no-render evaluation used a different sampling seed")
    warmup = warm_sampling_stream(policy, previous_rows, args.episode_index)
    expected_rows = [r for r in previous_rows if r["episode_index"] == args.episode_index]
    expected_reset = expected_rows[0]
    expected_steps = expected_rows[1:]
    if (expected_reset["event"] != "reset" or not expected_steps
            or any(row["event"] != "step" for row in expected_steps)):
        raise ValueError("unexpected prior selected episode reference")
    trace = []
    visual = VisualProtocol(args)
    try:
        reset = visual.request({"cmd": "reset"}, timeout=120)
        if tuple(reset) != ("info", "obs") and not ("obs" in reset and "info" in reset):
            raise ValueError("visual service reset contract changed")
        pointgoal = reset["obs"]
        initial_match = (
            reset["info"]["episode_id"] == expected_reset["episode_id"]
            and reset["info"]["scene_id"] == expected_reset["scene_id"]
            and all(close_float(a, b) for a, b in zip(pointgoal, expected_reset["pointgoal"]))
        )
        trace.append({
            "event": "reset", "episode_index": args.episode_index,
            "pointgoal": pointgoal, "info": reset["info"],
            "prior_no_renderer_initial_match": initial_match,
        })
        reward_sum = 0.0
        for step in range(1, 65):
            action, logits, logprob = choose_action(policy, pointgoal, "sampled")
            transition = visual.request({"cmd": "step", "action": action}, timeout=120)
            info = transition["info"]
            if bool(transition["terminated"]) != (action == 0):
                raise ValueError("STOP/terminated contract changed")
            if info["success"] and not (
                transition["terminated"] and float(info["geodesic_distance"]) < 0.2
            ):
                raise ValueError("success violates STOP/geodesic threshold")
            reward_sum += float(transition["reward"])
            row = {
                "event": "step", "episode_index": args.episode_index, "step": step,
                "pointgoal_before": pointgoal,
                "action_id": action, "action": ACTIONS[action],
                "logits": logits, "selected_logprob": logprob,
                "pointgoal_after": transition["obs"],
                "reward": transition["reward"],
                "terminated": transition["terminated"],
                "truncated": transition["truncated"], "info": info,
            }
            trace.append(row)
            pointgoal = transition["obs"]
            if transition["terminated"] or transition["truncated"]:
                break
        else:
            raise RuntimeError("visual episode exceeded 64 steps")
        finalized = visual.request({"cmd": "finalize"}, timeout=120)
        if not finalized.get("finalized"):
            raise RuntimeError("visual service did not finalize")
        manifest = finalized["manifest"]
        if manifest["outcome"]["steps"] != step:
            raise RuntimeError("service step count differs from policy trace")
    finally:
        visual.close()
    args.output_dir.mkdir(parents=False, exist_ok=True)
    trace_path = args.output_dir / "policy_trace.jsonl"
    with trace_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in trace:
            stream.write(canonical_json(row) + "\n")
    actual_steps = trace[1:]
    action_match = (
        len(actual_steps) == len(expected_steps)
        and all(a["action_id"] == b["action_id"] for a, b in zip(actual_steps, expected_steps))
    )
    pointgoal_match = (
        len(actual_steps) == len(expected_steps)
        and all(
            all(close_float(x, y) for x, y in zip(a["pointgoal_before"], b["pointgoal_before"]))
            and all(close_float(x, y) for x, y in zip(a["pointgoal_after"], b["pointgoal_after"]))
            for a, b in zip(actual_steps, expected_steps)
        )
    )
    metric_fields = ("geodesic_distance", "distance_to_goal", "path_length", "spl")
    metric_match = (
        len(actual_steps) == len(expected_steps)
        and all(
            all(close_float(a["info"][key], b["info"][key]) for key in metric_fields)
            and a["info"]["success"] == b["info"]["success"]
            and a["info"]["collision_count"] == b["info"]["collision_count"]
            and close_float(a["reward"], b["reward"])
            for a, b in zip(actual_steps, expected_steps)
        )
    )
    if not all((initial_match, action_match, pointgoal_match, metric_match)):
        raise RuntimeError(
            "visual rollout differs from the same checkpoint's no-render "
            f"reference: initial={initial_match}, actions={action_match}, "
            f"pointgoal={pointgoal_match}, metrics_reward={metric_match}"
        )
    evaluation = {
        "schema_version": 1,
        "mode": "sampled_checkpoint_drives_live_rgbd_episode",
        "interpretation": "one PPO update and one official episode; no policy-quality claim",
        "dataset": "official habitat-test-scenes PointNav v1 val",
        "episode_index": args.episode_index,
        "checkpoint_sha256": checkpoint_sha,
        "expected_checkpoint_sha256": args.expected_checkpoint_sha256,
        "rlinf_base_commit": revision,
        "policy_config": POLICY_CONFIG,
        "policy_input": "two-float PointGoal only",
        "policy_uses_rgb_depth": False,
        "policy_uses_privileged_geometry": False,
        "rendering": "private Habitat-Sim 0.2.4 CPU EGL / Mesa llvmpipe",
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "ray_started": False,
        "torch_seed": args.torch_seed,
        "sampling_warmup_actions_checked": warmup,
        "prior_trace_sha256": previous_sha,
        "prior_summary_trace_sha256": previous_summary["trace_sha256"],
        "prior_initial_state_match": initial_match,
        "prior_action_sequence_match": action_match,
        "prior_pointgoal_sequence_match": pointgoal_match,
        "prior_step_metrics_and_rewards_match": metric_match,
        "prior_episode_steps": len(expected_steps),
        "actual_steps": step,
        "action_sequence": [r["action"] for r in actual_steps],
        "reward_sum": reward_sum,
        "outcome": manifest["outcome"],
        "video": manifest["video"],
        "service_manifest_sha256": sha256_file(args.output_dir / "manifest.json"),
        "policy_trace_sha256": sha256_file(trace_path),
        "client_script_sha256": sha256_file(Path(__file__)),
        "service_script_sha256": sha256_file(args.service_script),
    }
    (args.output_dir / "evaluation.json").write_text(
        json.dumps(evaluation, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-root", type=Path, default=ROOT)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--expected-checkpoint-sha256", default=EXPECTED_CHECKPOINT_SHA256,
        help="exact digest of the selected checkpoint; defaults to the step-1 fixture",
    )
    parser.add_argument("--torch-seed", type=int, default=7)
    parser.add_argument("--episode-index", type=int, default=4)
    parser.add_argument("--previous-trace", type=Path)
    parser.add_argument("--previous-summary", type=Path)
    parser.add_argument("--habitat-env", type=Path)
    parser.add_argument("--habitat-sim-source", type=Path)
    parser.add_argument("--service-script", type=Path)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    args.expected_checkpoint_sha256 = args.expected_checkpoint_sha256.lower()
    if (
        len(args.expected_checkpoint_sha256) != 64
        or any(c not in "0123456789abcdef" for c in args.expected_checkpoint_sha256)
        or args.torch_seed < 0
        or args.episode_index < 0
    ):
        parser.error("expected checkpoint SHA256 must be 64 hex characters; seed and episode index must be >= 0")
    args.node_root = args.node_root.resolve()
    args.checkpoint = args.checkpoint or (
        args.node_root / "runs/pointnav-ppo-v03/pointnav-ppo-smoke/checkpoints/global_step_1"
        / "actor/model_state_dict/full_weights.pt"
    )
    args.previous_trace = args.previous_trace or (
        args.node_root / "runs/pointnav-ppo-v03/cpu_eval5_64"
        / "actor_step1_sampled_seed7_with_poses.jsonl"
    )
    args.previous_summary = args.previous_summary or (
        args.node_root / "runs/pointnav-ppo-v03/cpu_eval5_64"
        / "actor_step1_sampled_seed7_with_poses.summary.json"
    )
    args.habitat_env = args.habitat_env or args.node_root / "envs/habitat-cpu-egl"
    args.habitat_sim_source = args.habitat_sim_source or (
        args.node_root / "repos/habitat-sim-cpu-egl/src_python"
    )
    args.service_script = args.service_script or (
        args.node_root / "repos/rlinf-nav/scripts/habitat_rgbd_policy_service.py"
    )
    args.asset_root = args.asset_root or args.node_root / "data"
    args.output_dir = args.output_dir or (
        args.node_root / "runs/pointnav-ppo-v03"
        / f"checkpoint-rgbd-ep{args.episode_index}-sampled-seed{args.torch_seed}"
    )
    print(canonical_json(evaluate(args)))


if __name__ == "__main__":
    main()
