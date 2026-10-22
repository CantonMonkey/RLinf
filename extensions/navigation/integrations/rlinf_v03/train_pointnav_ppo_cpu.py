"""Bounded, single-environment on-policy PointNav PPO using RLinf v0.3 on CPU.

The policy receives only a two-float PointGoal from the official Habitat
PointNav *train* split. Habitat runs in a separate stdio process; the default
does not render, while --render-rgbd records camera RGB/depth on CPU Mesa.
No Ray, CUDA, privileged pose, RGB or depth enters action selection. This is
an intentionally small reference trainer, not RLinf's distributed trainer.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import inspect
import json
import math
import os
import select
import subprocess
import sys
import time
from pathlib import Path

# Set before importing Torch or RLinf, including in callers that exposed GPUs.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.pop("RAY_ADDRESS", None)

ROOT = Path("/srv/navigation")
RLINF_COMMIT = "0505431899574619da86f551bad70b71e0ea2177"
ACTIONS = ("stop", "move_forward", "turn_left", "turn_right")


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_visual_train_trace(policy_trace: Path, visual_trace: Path, manifest: dict) -> dict:
    """Bind every PPO transition to the rendered Habitat service event."""
    policy_rows = [json.loads(line) for line in policy_trace.read_text(encoding="utf-8").splitlines()]
    events = [json.loads(line) for line in visual_trace.read_text(encoding="utf-8").splitlines()]
    if not policy_rows or not events:
        raise RuntimeError("empty PPO or visual train trace")
    frames = manifest.get("video", {}).get("frames")
    if frames != len(events):
        raise RuntimeError("RGBD video must contain one frame for every reset and step")
    if manifest.get("video", {}).get("sampled_prefix_only") is not False:
        raise RuntimeError("RGBD video did not cover the complete training rollout")
    if [row.get("event_index") for row in events] != list(range(len(events))):
        raise RuntimeError("RGBD service event order changed")
    if [row.get("recorded_frame_index") for row in events] != list(range(len(events))):
        raise RuntimeError("RGBD frame sequence has a gap")
    step_index = 0
    reset_count = 0
    previous_obs = None
    require_reset = True
    for event in events:
        response = event["response"]
        if event["event"] == "reset":
            if not require_reset:
                raise RuntimeError("RGBD service reset before episode completion")
            reset_count += 1
            previous_obs = response["obs"]
            require_reset = False
            continue
        if event["event"] != "step" or require_reset or step_index >= len(policy_rows):
            raise RuntimeError("RGBD service step order changed")
        row = policy_rows[step_index]
        info = response["info"]
        matched = (
            row["global_step"] == step_index + 1
            and row["obs"] == previous_obs
            and event["action"] == row["action"]
            and info["episode_id"] == row["episode_id"]
            and info["scene_id"] == row["scene_id"]
            and response["obs"] == row["next_obs"]
            and response["reward"] == row["reward"]
            and response["terminated"] == row["terminated"]
            and response["truncated"] == row["truncated"]
            and info["success"] == row["success"]
            and info["spl"] == row["spl"]
            and info["geodesic_distance"] == row["geodesic_distance"]
        )
        if not matched:
            raise RuntimeError(f"RGBD service/PPO transition mismatch at step {step_index + 1}")
        previous_obs = response["obs"]
        require_reset = bool(response["terminated"] or response["truncated"])
        step_index += 1
    outcome = manifest.get("outcome", {})
    if (step_index != len(policy_rows)
            or outcome.get("environment_steps") != step_index
            or outcome.get("resets") != reset_count
            or outcome.get("completed_episodes") != sum(
                int(row["terminated"] or row["truncated"]) for row in policy_rows
            )
            or outcome.get("train_successes") != sum(int(row["success"]) for row in policy_rows)
            or outcome.get("reward_sum") != sum(float(row["reward"]) for row in policy_rows)):
        raise RuntimeError("RGBD manifest outcome differs from PPO train trace")
    return {
        "events_checked": len(events),
        "resets_checked": reset_count,
        "steps_checked": step_index,
        "frame_indices_contiguous": True,
        "all_ppo_transitions_match_visual_service": True,
        "visual_trace_sha256": sha256_file(visual_trace),
    }


def compute_gae(
    rewards: list[float],
    values: list[float],
    next_values: list[float],
    terminated: list[bool],
    truncated: list[bool],
    gamma: float,
    gae_lambda: float,
) -> tuple[list[float], list[float]]:
    """GAE with a value bootstrap at time limits and at rollout horizons.

    STOP is a true terminal and has zero bootstrap. A Habitat time limit is a
    truncation: its final observation is valued, but its advantage must not
    recurse into the next reset episode. The last item of a nonterminal
    rollout similarly uses its next value and ends the recursion at the batch
    edge. `next_values[t]` always belongs to the pre-reset transition.
    """
    n = len(rewards)
    if not n or any(len(field) != n for field in (values, next_values, terminated, truncated)):
        raise ValueError("GAE fields must have equal nonzero lengths")
    if not (0.0 <= gamma <= 1.0 and 0.0 <= gae_lambda <= 1.0):
        raise ValueError("gamma and lambda must be in [0, 1]")
    if any(t and u for t, u in zip(terminated, truncated)):
        raise ValueError("STOP and time-limit truncation cannot coincide")
    if not all(math.isfinite(float(x)) for field in (rewards, values, next_values) for x in field):
        raise ValueError("GAE inputs must be finite")
    advantages = [0.0] * n
    returns = [0.0] * n
    running = 0.0
    for t in range(n - 1, -1, -1):
        bootstrap = 0.0 if terminated[t] else float(next_values[t])
        delta = float(rewards[t]) + gamma * bootstrap - float(values[t])
        continue_episode = not (terminated[t] or truncated[t])
        running = delta + (gamma * gae_lambda * running if continue_episode else 0.0)
        advantages[t] = running
        returns[t] = running + float(values[t])
    return advantages, returns


def pointgoal(value: object) -> list[float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("PointGoal must be a two-element list")
    result = [float(x) for x in value]
    if not all(math.isfinite(x) for x in result):
        raise ValueError("PointGoal must be finite")
    return result


def episode_geometry_signature(episode: dict) -> str:
    """Identify the same PointNav start and goal even if its ID changes."""
    return canonical_json({
        "scene_basename": str(episode["scene_id"]).replace("\\", "/").split("/")[-1],
        "start_position": episode["start_position"],
        "start_rotation": episode["start_rotation"],
        "goals": [goal["position"] for goal in episode["goals"]],
    })


def train_episode_indices(asset_root: Path, indices: str) -> tuple[list[int], list[str], str, str]:
    data = asset_root / "datasets/pointnav/habitat-test-scenes/v1/train/train.json.gz"
    val_data = asset_root / "datasets/pointnav/habitat-test-scenes/v1/val/val.json.gz"
    with gzip.open(data, "rt", encoding="utf-8") as stream:
        episodes = json.load(stream)["episodes"]
    with gzip.open(val_data, "rt", encoding="utf-8") as stream:
        val_episodes = json.load(stream)["episodes"]
    selected = [int(item) for item in indices.split(",")]
    if not selected or len(selected) != len(set(selected)) or any(i < 0 or i >= len(episodes) for i in selected):
        raise ValueError("train episode indices must be distinct and within the official train split")
    scenes = {episodes[i]["scene_id"] for i in selected}
    if len(scenes) != 1:
        raise ValueError("one Habitat service process requires train episodes from one scene")
    # The official test dataset repeats some entries across train and val.
    # Reject both byte-equivalent episode objects and renamed identical poses.
    val_full = {canonical_json(episode) for episode in val_episodes}
    val_geometry = {episode_geometry_signature(episode) for episode in val_episodes}
    overlap = [i for i in selected if canonical_json(episodes[i]) in val_full
               or episode_geometry_signature(episodes[i]) in val_geometry]
    if overlap:
        raise ValueError(f"selected train episodes overlap official val: {overlap}")
    return selected, [str(episodes[i]["episode_id"]) for i in selected], sha256_file(data), sha256_file(val_data)


class HabitatProtocol:
    def __init__(self, python: Path, script: Path, asset_root: Path,
                 episode_indices: list[int], episode_ids: list[str], max_steps: int,
                 premature_stop_penalty: float, *, render_rgbd: bool = False,
                 visual_output_dir: Path | None = None,
                 max_recorded_frames: int = 128,
                 habitat_sim_source: Path | None = None,
                 mesa_egl_vendor_json: Path | None = None):
        self.expected_episode_ids = set(episode_ids)
        self.expected_dataset = (asset_root / "datasets/pointnav/habitat-test-scenes/v1/train/train.json.gz").resolve(strict=True)
        child_env = os.environ.copy()
        child_env.pop("PYTHONPATH", None)
        child_env.pop("RAY_ADDRESS", None)
        child_env["CUDA_VISIBLE_DEVICES"] = ""
        child_env["PYTHONDONTWRITEBYTECODE"] = "1"
        command = [
            str(python), "-B", str(script), "--asset-root", str(asset_root),
            "--split", "train", "--episode-indices", ",".join(map(str, episode_indices)),
            "--max-steps", str(max_steps),
            "--premature-stop-penalty", str(premature_stop_penalty),
        ]
        if render_rgbd:
            if visual_output_dir is None or habitat_sim_source is None or mesa_egl_vendor_json is None:
                raise ValueError("RGBD service requires output, Habitat source, and Mesa vendor paths")
            habitat_env = python.parent.parent
            child_env["PATH"] = f"{habitat_env / 'bin'}:/usr/bin:/bin"
            child_env["PYTHONPATH"] = str(habitat_sim_source)
            child_env["LD_LIBRARY_PATH"] = str(habitat_env / "lib")
            child_env["LIBGL_ALWAYS_SOFTWARE"] = "1"
            child_env["EGL_PLATFORM"] = "surfaceless"
            child_env["HABITAT_SIM_SOFTWARE_EGL"] = "1"
            child_env["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(mesa_egl_vendor_json)
            command += [
                "--output-dir", str(visual_output_dir),
                "--max-recorded-frames", str(max_recorded_frames),
            ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
            text=True, bufsize=1, env=child_env,
        )
        try:
            ready = self._read(timeout=120)
            if (ready.get("ready") is not True
                    or tuple(ready.get("actions", ())) != ACTIONS
                    or ready.get("split") != "train"
                    or Path(ready.get("dataset_path", "")).resolve() != self.expected_dataset
                    or bool(ready.get("renderer_created", False)) != render_rgbd):
                raise RuntimeError(f"Habitat train service protocol mismatch: {ready}")
        except BaseException:
            self.close()
            raise

    def _read(self, timeout: int = 60) -> dict:
        readable, _, _ = select.select([self.process.stdout], [], [], timeout)
        if not readable:
            raise TimeoutError("Habitat train service response timed out")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"Habitat train service exited: {self.process.poll()}")
        result = json.loads(line)
        if "error" in result:
            raise RuntimeError(f"Habitat train service error: {result['error']}")
        return result

    def request(self, value: dict, timeout: int = 60) -> dict:
        if self.process.poll() is not None:
            raise RuntimeError(f"Habitat train service exited: {self.process.returncode}")
        self.process.stdin.write(canonical_json(value) + "\n")
        self.process.stdin.flush()
        result = self._read(timeout)
        if value["cmd"] in ("reset", "step"):
            episode_id = str(result.get("info", {}).get("episode_id", ""))
            if episode_id not in self.expected_episode_ids:
                raise RuntimeError(f"Habitat returned an episode outside selected train IDs: {episode_id}")
        return result

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


def make_policy():
    from rlinf.models.embodiment.mlp_policy.mlp_policy import MLPPolicy

    return MLPPolicy(
        obs_dim=2, action_dim=1, num_action_chunks=1,
        add_value_head=True, add_q_head=False, discrete_n_actions=4,
    ).to("cpu")


def check_rlinf_source(checkout: Path) -> str:
    from rlinf.models.embodiment.mlp_policy.mlp_policy import MLPPolicy

    revision = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True, timeout=10
    ).strip()
    if revision != RLINF_COMMIT:
        raise ValueError(f"RLinf commit changed: {revision}")
    expected = (checkout / "rlinf/models/embodiment/mlp_policy/mlp_policy.py").resolve(strict=True)
    actual = Path(inspect.getsourcefile(MLPPolicy)).resolve(strict=True)
    if actual != expected:
        raise ValueError(f"imported RLinf policy is not the pinned checkout: {actual}")
    return sha256_file(actual)


def save_checkpoint(torch, policy, optimizer, directory: Path, update: int, steps: int, seed: int) -> dict:
    target = directory / f"update_{update:04d}"
    target.mkdir(parents=True, exist_ok=False)
    weights = target / "full_weights.pt"
    training = target / "training.pt"
    torch.save(policy.state_dict(), weights)
    torch.save({"model": policy.state_dict(), "optimizer": optimizer.state_dict(),
                "update": update, "env_steps": steps, "seed": seed}, training)
    # A fresh model must accept the inference artifact with no missing keys.
    loaded = torch.load(weights, map_location="cpu", weights_only=True)
    restored = make_policy()
    restored.load_state_dict(loaded, strict=True)
    if not all(torch.equal(value, restored.state_dict()[key]) for key, value in policy.state_dict().items()):
        raise RuntimeError("strict checkpoint round-trip changed weights")
    return {"update": update, "env_steps": steps, "full_weights": str(weights),
            "full_weights_sha256": sha256_file(weights), "training_checkpoint": str(training),
            "training_checkpoint_sha256": sha256_file(training), "strict_load": True}


def validate_args(args: argparse.Namespace) -> None:
    limits = (
        ("updates", args.updates, 1, 1000),
        ("rollout_steps", args.rollout_steps, 2, 2048),
        ("max_episode_steps", args.max_episode_steps, 2, 512),
        ("ppo_epochs", args.ppo_epochs, 1, 16),
        ("minibatch_size", args.minibatch_size, 2, args.rollout_steps),
        ("checkpoint_every", args.checkpoint_every, 1, args.updates),
        ("threads", args.threads, 1, 16),
    )
    for label, value, low, high in limits:
        if not low <= value <= high:
            raise ValueError(f"{label} must be in [{low}, {high}]")
    if args.updates * args.rollout_steps > args.max_total_steps:
        raise ValueError("planned environment steps exceed --max-total-steps")
    if args.seed < 0 or args.max_total_steps < 2:
        raise ValueError("seed must be nonnegative and max-total-steps >= 2")
    if not (0 < args.learning_rate <= 0.01 and 0 <= args.entropy_coef <= 1):
        raise ValueError("learning rate or entropy coefficient outside supported range")
    if not (0 < args.gamma <= 1 and 0 <= args.gae_lambda <= 1):
        raise ValueError("gamma or GAE lambda outside supported range")
    if not math.isfinite(args.premature_stop_penalty) or args.premature_stop_penalty < 0:
        raise ValueError("premature STOP penalty must be finite and nonnegative")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not args.output_dir.resolve().is_relative_to(args.node_root.resolve() / "runs"):
        raise ValueError("output directory must be new and below node-root/runs")
    for label, path in (("Habitat Python", args.habitat_python),
                        ("service script", args.service_script),
                        ("train dataset", args.asset_root / "datasets/pointnav/habitat-test-scenes/v1/train/train.json.gz")):
        if not path.is_file():
            raise FileNotFoundError(f"{label}: {path}")
    if args.render_rgbd:
        if args.updates * args.rollout_steps > 64:
            raise ValueError("RGBD recording is limited to 64 training steps per run")
        if args.max_recorded_frames < 2 * args.updates * args.rollout_steps:
            raise ValueError("RGBD frame budget must cover resets and all steps")
        for label, path in (("private Habitat-Sim source", args.habitat_sim_source),
                            ("Mesa EGL vendor JSON", args.mesa_egl_vendor_json)):
            if not path.exists():
                raise FileNotFoundError(f"{label}: {path}")


def train(args: argparse.Namespace) -> dict:
    validate_args(args)
    import torch
    from rlinf.algorithms.losses import compute_ppo_actor_critic_loss

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    source_sha = check_rlinf_source(args.node_root / "repos/RLinf-v0.3")
    episode_indices, episode_ids, train_dataset_sha, val_dataset_sha = train_episode_indices(args.asset_root, args.episode_indices)
    policy = make_policy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    settings = {
        "schema_version": 1, "split": "train", "episode_indices": episode_indices,
        "episode_ids": episode_ids, "train_dataset_sha256": train_dataset_sha,
        "val_dataset_sha256_for_leakage_gate": val_dataset_sha,
        "rlinf_commit": RLINF_COMMIT, "mlp_policy_sha256": source_sha,
        "trainer_sha256": sha256_file(Path(__file__)),
        "service_script_sha256": sha256_file(args.service_script),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "ray_started": False, "policy_input": "two-float PointGoal only",
        "renderer_created_expected": bool(args.render_rgbd),
        "rgb_depth_used_for_policy": False,
        "visual_output_dir": str(args.output_dir / "visual_rollout") if args.render_rgbd else None,
        "config": {name: getattr(args, name) for name in (
            "updates", "rollout_steps", "max_episode_steps", "ppo_epochs", "minibatch_size",
            "checkpoint_every", "max_total_steps", "seed", "threads", "learning_rate",
            "entropy_coef", "gamma", "gae_lambda", "normalize_advantages",
            "premature_stop_penalty", "render_rgbd", "max_recorded_frames")},
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(settings, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    run_start = time.perf_counter()
    service = HabitatProtocol(args.habitat_python, args.service_script, args.asset_root,
                              episode_indices, episode_ids, args.max_episode_steps,
                              args.premature_stop_penalty,
                              render_rgbd=args.render_rgbd,
                              visual_output_dir=(args.output_dir / "visual_rollout") if args.render_rgbd else None,
                              max_recorded_frames=args.max_recorded_frames,
                              habitat_sim_source=args.habitat_sim_source,
                              mesa_egl_vendor_json=args.mesa_egl_vendor_json)
    current = None
    episode_return = 0.0
    episode_steps = 0
    total_steps = 0
    completed = []
    # Preserve the fresh seeded model for held-out before/after evaluation.
    checkpoints = [save_checkpoint(torch, policy, optimizer,
                                   args.output_dir / "checkpoints", 0, 0, args.seed)]
    update_results = []
    visual_result = None
    try:
        with (args.output_dir / "train_trace.jsonl").open("w", encoding="utf-8", newline="\n") as trace:
            for update in range(1, args.updates + 1):
                update_start = time.perf_counter()
                rollout = []
                policy.eval()
                for slot in range(args.rollout_steps):
                    if current is None:
                        reset = service.request({"cmd": "reset"})
                        current = pointgoal(reset["obs"])
                        episode_return = 0.0
                        episode_steps = 0
                    obs = current
                    state = torch.tensor([obs], dtype=torch.float32)
                    actions, sampled = policy.predict_action_batch(env_obs={"states": state}, mode="train")
                    if tuple(actions.shape) != (1, 1, 1) or actions.dtype != torch.int64:
                        raise RuntimeError("RLinf policy action shape/type changed")
                    action = int(actions[0, 0, 0].item())
                    if action not in range(4):
                        raise RuntimeError("RLinf policy returned invalid PointNav action")
                    response = service.request({"cmd": "step", "action": action})
                    following = pointgoal(response["obs"])
                    terminal, timeout = bool(response["terminated"]), bool(response["truncated"])
                    if terminal != (action == 0) or (terminal and timeout):
                        raise RuntimeError("Habitat STOP/time-limit contract changed")
                    reward = float(response["reward"])
                    old_lp = float(sampled["prev_logprobs"][0, 0].item())
                    old_v = float(sampled["prev_values"][0, 0].item())
                    if not all(math.isfinite(x) for x in (reward, old_lp, old_v)):
                        raise RuntimeError("nonfinite policy value, logprob or reward")
                    info = response["info"]
                    if bool(info["success"]) and not terminal:
                        raise RuntimeError("Habitat success without STOP")
                    rollout.append({"obs": obs, "next_obs": following, "action": action,
                                    "old_logprob": old_lp, "old_value": old_v,
                                    "reward": reward, "terminated": terminal, "truncated": timeout})
                    total_steps += 1
                    episode_steps += 1
                    episode_return += reward
                    trace.write(canonical_json({
                        "global_step": total_steps, "update": update, "slot": slot,
                        "episode_id": info["episode_id"], "scene_id": info["scene_id"],
                        "obs": obs, "action_id": action, "action": ACTIONS[action],
                        "next_obs": following, "reward": reward, "terminated": terminal,
                        "truncated": timeout, "old_logprob": old_lp, "old_value": old_v,
                        "success": bool(info["success"]), "spl": float(info["spl"]),
                        "geodesic_distance": float(info["geodesic_distance"]),
                    }) + "\n")
                    current = None if terminal or timeout else following
                    if current is None:
                        completed.append({"episode_id": info["episode_id"],
                                          "steps": episode_steps, "return": episode_return,
                                          "success": bool(info["success"]),
                                          "spl": float(info["spl"]),
                                          "termination_reason": info["termination_reason"]})
                rollout_seconds = time.perf_counter() - update_start
                with torch.inference_mode():
                    next_states = torch.tensor([row["next_obs"] for row in rollout], dtype=torch.float32)
                    next_values = policy.value_head(next_states).reshape(-1).tolist()
                advantages, returns = compute_gae(
                    [row["reward"] for row in rollout],
                    [row["old_value"] for row in rollout], next_values,
                    [row["terminated"] for row in rollout],
                    [row["truncated"] for row in rollout], args.gamma, args.gae_lambda,
                )
                states = torch.tensor([row["obs"] for row in rollout], dtype=torch.float32)
                acts = torch.tensor([[row["action"]] for row in rollout], dtype=torch.int64)
                old_logprobs = torch.tensor([[row["old_logprob"]] for row in rollout], dtype=torch.float32)
                old_values = torch.tensor([[row["old_value"]] for row in rollout], dtype=torch.float32)
                adv = torch.tensor(advantages, dtype=torch.float32).unsqueeze(-1)
                ret = torch.tensor(returns, dtype=torch.float32).unsqueeze(-1)
                if args.normalize_advantages and len(rollout) > 1:
                    spread = adv.std(unbiased=False)
                    if float(spread) > 1e-8:
                        adv = (adv - adv.mean()) / (spread + 1e-8)
                policy.train()
                losses = []
                grad_norms = []
                optimization_start = time.perf_counter()
                for _ in range(args.ppo_epochs):
                    for ids in torch.randperm(len(rollout)).split(args.minibatch_size):
                        output = policy(
                            forward_inputs={"states": states[ids], "action": acts[ids]},
                            compute_logprobs=True, compute_entropy=True, compute_values=True,
                        )
                        loss, _ = compute_ppo_actor_critic_loss(
                            logprobs=output["logprobs"], old_logprobs=old_logprobs[ids],
                            values=output["values"], prev_values=old_values[ids],
                            advantages=adv[ids], returns=ret[ids],
                            clip_ratio_low=0.2, clip_ratio_high=0.2,
                            value_clip=1.0, huber_delta=10.0,
                            loss_mask=torch.ones_like(adv[ids], dtype=torch.bool),
                        )
                        loss = loss - args.entropy_coef * output["entropy"].mean()
                        if not bool(torch.isfinite(loss)):
                            raise RuntimeError("nonfinite RLinf PPO loss")
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            policy.parameters(), 0.5, error_if_nonfinite=True
                        )
                        grad_norms.append(float(grad_norm))
                        optimizer.step()
                        losses.append(float(loss.detach()))
                ppo_seconds = time.perf_counter() - optimization_start
                result = {"update": update, "env_steps": total_steps,
                          "mean_ppo_loss": sum(losses) / len(losses),
                          "mean_preclip_grad_norm": sum(grad_norms) / len(grad_norms),
                          "max_preclip_grad_norm": max(grad_norms),
                          "rollout_seconds": rollout_seconds,
                          "ppo_seconds": ppo_seconds,
                          "update_wall_seconds": time.perf_counter() - update_start,
                          "episodes_completed": len(completed),
                          "train_successes": sum(int(ep["success"]) for ep in completed),
                          "last_rollout_reward_sum": sum(row["reward"] for row in rollout)}
                update_results.append(result)
                print(canonical_json(result), flush=True)
                trace.flush()
                if update % args.checkpoint_every == 0 or update == args.updates:
                    checkpoints.append(save_checkpoint(torch, policy, optimizer,
                        args.output_dir / "checkpoints", update, total_steps, args.seed))
            trace.flush()
        if args.render_rgbd:
            visual_result = service.request({"cmd": "finalize"}, timeout=180)
            if visual_result.get("finalized") is not True:
                raise RuntimeError("RGBD train service did not finalize")
            visual_manifest = visual_result.get("manifest")
            if not isinstance(visual_manifest, dict):
                raise RuntimeError("RGBD train service did not return a manifest")
    finally:
        service.close()
    summary = {**settings, "status": "completed", "environment_steps": total_steps,
               "completed_train_episodes": completed, "updates": update_results,
               "checkpoints": checkpoints,
               "run_wall_seconds": time.perf_counter() - run_start,
               "trace_sha256": sha256_file(args.output_dir / "train_trace.jsonl")}
    if args.render_rgbd:
        visual_dir = args.output_dir / "visual_rollout"
        manifest_path = visual_dir / "manifest.json"
        video_path = visual_dir / "rgb_depth.mp4"
        if not manifest_path.is_file() or not video_path.is_file():
            raise RuntimeError("RGBD train service output is incomplete")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest != visual_result["manifest"]:
            raise RuntimeError("RGBD train manifest differs from service reply")
        frames = manifest.get("video", {}).get("frames")
        if not isinstance(frames, int) or frames < total_steps + 1:
            raise RuntimeError("RGBD train video has fewer frames than environment steps")
        alignment = verify_visual_train_trace(
            args.output_dir / "train_trace.jsonl", visual_dir / "trace.jsonl", manifest
        )
        if alignment["steps_checked"] != total_steps:
            raise RuntimeError("RGBD train trace step count differs from PPO updates")
        summary["visual_artifacts"] = {
            "output_dir": str(visual_dir),
            "manifest_sha256": sha256_file(manifest_path),
            "video_sha256": sha256_file(video_path),
            "frames": frames,
            "full_decode_validated": manifest.get("video", {}).get("full_decode_validated") is True,
            "alignment": alignment,
        }
        if not summary["visual_artifacts"]["full_decode_validated"]:
            raise RuntimeError("RGBD train video full decode was not verified")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-root", type=Path, default=ROOT)
    parser.add_argument("--habitat-python", type=Path, required=True)
    parser.add_argument("--service-script", type=Path)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode-indices", default="50,51")
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--rollout-steps", type=int, default=32)
    parser.add_argument("--max-episode-steps", type=int, default=64)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=16)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--max-total-steps", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--premature-stop-penalty", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--normalize-advantages", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--render-rgbd", action="store_true",
                        help="record RGB/depth in a private CPU-Mesa Habitat train service; policy remains PointGoal-only")
    parser.add_argument("--max-recorded-frames", type=int, default=128)
    parser.add_argument("--habitat-sim-source", type=Path)
    parser.add_argument("--mesa-egl-vendor-json", type=Path,
                        default=Path("/usr/share/glvnd/egl_vendor.d/50_mesa.json"))
    args = parser.parse_args()
    args.node_root = args.node_root.resolve()
    args.asset_root = args.asset_root or args.node_root / "data"
    args.service_script = args.service_script or args.node_root / "repos/rlinf-nav/scripts" / (
        "habitat_rgbd_train_service.py" if args.render_rgbd else "pointnav_service.py"
    )
    args.habitat_sim_source = args.habitat_sim_source or args.node_root / "repos/habitat-sim-cpu-egl/src_python"
    result = train(args)
    print(canonical_json({"status": result["status"], "environment_steps": result["environment_steps"],
                          "final_full_weights": result["checkpoints"][-1]["full_weights"],
                          "final_full_weights_sha256": result["checkpoints"][-1]["full_weights_sha256"]}))


if __name__ == "__main__":
    main()
