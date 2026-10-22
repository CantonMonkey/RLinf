"""Record one installed RLinf VLNPolicy/VLNEnv R2R rollout as a real RGB MP4.

This is a bounded visual engineering probe. Without --checkpoint the policy is
randomly initialized from a fixed seed, and its actions are sampled from that
fixed model. Such a video demonstrates the data/action/render chain, not a
learned VLN result. Shared R2R/MP3D assets are read only by VLNEnv's Habitat
service; all outputs stay under the private project's runs directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

# Set before Torch/RLinf imports. Habitat renders in VLNEnv's Mesa subprocess.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.pop("RAY_ADDRESS", None)
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
from hydra import compose, initialize_config_dir
from omegaconf import open_dict

from rlinf.envs import get_env_cls
from rlinf.models import get_model


ROOT = Path("/srv/navigation")
ACTIONS = ("stop", "move_forward", "turn_left", "turn_right")
OBS_KEYS = frozenset(("rgb", "instruction_ids", "instruction_mask"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _frame_bytes(obs: dict, rgb_size: int) -> bytes:
    if set(obs) != OBS_KEYS:
        raise ValueError("policy observation contains missing or privileged keys")
    rgb = obs["rgb"]
    if not isinstance(rgb, torch.Tensor) or rgb.dtype != torch.uint8:
        raise TypeError("Habitat RGB must be a uint8 tensor")
    if tuple(rgb.shape) != (1, rgb_size, rgb_size, 3):
        raise ValueError(f"unexpected Habitat RGB shape: {tuple(rgb.shape)}")
    return rgb.contiguous().cpu().numpy().tobytes()


def _probe_video(path: Path, width: int, height: int, frames: int) -> dict:
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-count_frames", "-show_entries",
            "stream=codec_name,width,height,nb_read_frames", "-of", "json", str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    streams = json.loads(probe.stdout).get("streams", [])
    if len(streams) != 1:
        raise RuntimeError("MP4 must contain exactly one video stream")
    video = streams[0]
    if (
        video.get("codec_name") != "h264"
        or video.get("width") != width
        or video.get("height") != height
        or int(video.get("nb_read_frames", -1)) != frames
    ):
        raise RuntimeError(f"MP4 stream/frame mismatch: {video}")
    decoded = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo",
            "-pix_fmt", "rgb24", "pipe:1",
        ],
        capture_output=True, check=True,
    )
    expected_bytes = frames * width * height * 3
    if len(decoded.stdout) != expected_bytes:
        raise RuntimeError(
            f"end-to-end MP4 decode yielded {len(decoded.stdout)} bytes, "
            f"expected {expected_bytes}"
        )
    return {
        "codec": video["codec_name"],
        "width": width,
        "height": height,
        "encoded_frames": frames,
        "decoded_rgb_bytes": len(decoded.stdout),
        "decoded_rgb_sha256": sha256_bytes(decoded.stdout),
    }


def _encode_video(path: Path, frames: list[bytes], width: int, height: int, fps: int) -> dict:
    if len(frames) < 2 or any(len(frame) != width * height * 3 for frame in frames):
        raise ValueError("video requires at least two complete Habitat RGB frames")
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s:v", f"{width}x{height}",
            "-framerate", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart",
            str(path),
        ],
        input=b"".join(frames), capture_output=True, check=True,
    )
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError("ffmpeg did not create a nonempty MP4")
    result = _probe_video(path, width, height, len(frames))
    result["sha256"] = sha256_file(path)
    result["bytes"] = path.stat().st_size
    result["fps"] = fps
    return result


def _load_config(args):
    config_dir = ROOT / "repos/RLinf-v0.3/examples/embodiment/config"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        cfg = compose(config_name="vln_r2r_ppo_smoke")
    with open_dict(cfg):
        cfg.env.eval.dataset_split = args.split
        cfg.env.eval.episode_index = args.episode_index
        cfg.env.eval.max_episode_steps = args.max_steps
        cfg.env.eval.max_steps_per_rollout_epoch = args.max_steps
        cfg.env.eval.video_cfg.save_video = False
    # RLinf's full validate_cfg initializes a Ray cluster. This standalone
    # visual diagnostic only needs the two installed sub-configurations.
    if cfg.env.eval.env_type != "vln" or cfg.actor.model.model_type != "vln_policy":
        raise ValueError("Hydra config does not select the installed R2R environment/policy")
    if cfg.env.eval.dataset_split != args.split or cfg.env.eval.episode_index != args.episode_index:
        raise ValueError("Hydra R2R split or episode selection changed")
    return cfg, config_dir


def _load_policy(args, model_cfg, output_dir: Path):
    torch.manual_seed(args.init_seed)
    policy = get_model(model_cfg)
    if not isinstance(policy, torch.nn.Module):
        raise RuntimeError("installed RLinf did not build a VLNPolicy")
    policy = policy.cpu()
    if args.checkpoint:
        checkpoint = args.checkpoint.expanduser().resolve(strict=True)
        actual_sha = sha256_file(checkpoint)
        if actual_sha != args.expected_checkpoint_sha256:
            raise ValueError("checkpoint SHA256 differs from the required digest")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not state or not all(
            isinstance(key, str) and isinstance(value, torch.Tensor)
            for key, value in state.items()
        ):
            raise ValueError("checkpoint must contain a plain model state dict")
        policy.load_state_dict(state, strict=True)
        source = {
            "kind": "strict_loaded_checkpoint",
            "path": str(checkpoint),
            "sha256": actual_sha,
        }
    else:
        weights = output_dir / "random_init_weights.pt"
        torch.save(policy.state_dict(), weights)
        source = {
            "kind": "untrained_fixed_seed_random_initialization",
            "init_seed": args.init_seed,
            "path": str(weights),
            "sha256": sha256_file(weights),
        }
    policy.eval()
    torch.manual_seed(args.action_seed)
    return policy, source


def _scalar(value, label):
    if not isinstance(value, torch.Tensor) or value.numel() != 1:
        raise ValueError(f"RLinf {label} must be one scalar tensor")
    return value.item()


def _source_hashes(config_dir: Path, env) -> dict:
    checkout = ROOT / "repos/RLinf-v0.3"
    paths = {
        "video_script": Path(__file__).resolve(),
        "vln_policy": checkout / "rlinf/models/embodiment/vln_policy.py",
        "vln_env": Path(env.__class__.__module__.replace(".", "/") + ".py"),
        "r2r_service": ROOT / "repos/rlinf-nav/scripts/habitat_r2r_service.py",
        "rlinf_config": config_dir / "vln_r2r_ppo_smoke.yaml",
    }
    paths["vln_env"] = checkout / paths["vln_env"]
    return {name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()}


def rollout(args):
    private_runs = (ROOT / "runs").resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    if not output_dir.is_relative_to(private_runs) or output_dir == private_runs:
        raise ValueError("output directory must be a new path under private runs/")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    cfg, config_dir = _load_config(args)
    torch.set_num_threads(1)
    output_dir.mkdir(parents=True, exist_ok=False)
    policy, weight_source = _load_policy(args, cfg.actor.model, output_dir)

    env_class = get_env_cls("vln", cfg.env.eval)
    env = env_class(cfg.env.eval, 1, 0, 1, None)
    try:
        obs, reset_info = env.reset()
        frame = _frame_bytes(obs, args.rgb_size)
        frames = [frame]
        trace = [{
            "event": "reset",
            "episode_index": args.episode_index,
            "episode_id": reset_info["episode_id"],
            "scene_id": reset_info["scene_id"],
            "rgb_sha256": sha256_bytes(frame),
        }]
        reward_sum = 0.0
        terminal = None
        for step in range(1, args.max_steps + 1):
            previous_hash = sha256_bytes(frame)
            action_tensor, policy_result = policy.predict_action_batch(
                obs,
                calculate_logprobs=True,
                calculate_values=True,
                return_obs=True,
                mode="train" if args.policy_mode == "sampled" else "eval",
            )
            action = int(_scalar(action_tensor, "action"))
            if not 0 <= action < len(ACTIONS):
                raise ValueError("VLNPolicy emitted an invalid action")
            next_observations, rewards, terminations, truncations, infos = (
                env.chunk_step(action_tensor.cpu().numpy())
            )
            if len(next_observations) != 1 or len(infos) != 1:
                raise ValueError("VLNEnv returned an unexpected chunk size")
            obs = next_observations[0]
            frame = _frame_bytes(obs, args.rgb_size)
            frames.append(frame)
            reward = float(_scalar(rewards, "reward"))
            terminated = bool(_scalar(terminations, "terminated"))
            truncated = bool(_scalar(truncations, "truncated"))
            reward_sum += reward
            item = {
                "event": "step", "step": step,
                "action_id": action, "action": ACTIONS[action],
                "behavior_logprob": float(_scalar(
                    policy_result["prev_logprobs"], "behavior_logprob"
                )),
                "value": float(_scalar(policy_result["prev_values"], "value")),
                "rgb_before_sha256": previous_hash,
                "rgb_after_sha256": sha256_bytes(frame),
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "success": float(_scalar(infos[0]["success"], "success")),
            }
            if terminated or truncated:
                episode = infos[0].get("episode")
                if not isinstance(episode, dict):
                    raise ValueError("VLNEnv omitted terminal Success/SPL metrics")
                terminal = {
                    key: float(_scalar(value, key))
                    for key, value in episode.items()
                }
                item["terminal_metrics"] = terminal
            trace.append(item)
            if terminated or truncated:
                break
        if terminal is None:
            raise RuntimeError("bounded VLN rollout did not terminate or truncate")
        trace_path = output_dir / "action_trace.jsonl"
        trace_path.write_text(
            "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                    for row in trace), encoding="utf-8"
        )
        movie = output_dir / "habitat_rgb_rollout.mp4"
        video_probe = _encode_video(movie, frames, args.rgb_size, args.rgb_size, args.fps)
        revision = subprocess.run(
            ["git", "-C", str(ROOT / "repos/RLinf-v0.3"), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        manifest = {
            "status": "ok",
            "interpretation": (
                "Untrained fixed-seed policy engineering demo; no VLN learning claim."
                if args.checkpoint is None else
                "Checkpoint policy visual rollout; evaluate learning only against a proper baseline."
            ),
            "policy_input_keys": sorted(OBS_KEYS),
            "policy_mode": args.policy_mode,
            "action_seed": args.action_seed,
            "weight_source": weight_source,
            "torch_version": torch.__version__,
            "rlinf_base_revision": revision,
            "source_sha256": _source_hashes(config_dir, env),
            "split": args.split,
            "episode_index": args.episode_index,
            "episode_id": env.identity["episode_id"],
            "scene_id": env.identity["scene_id"],
            "dataset_path": str(env.dataset_path),
            "dataset_sha256": env.identity["dataset_sha256"],
            "vocab_sha256": env.identity["vocab_sha256"],
            "instruction_sha256": env.identity["instruction_sha256"],
            "success_distance_m": env.success_distance_m,
            "steps": len(trace) - 1,
            "total_reward": reward_sum,
            "terminal_metrics": terminal,
            "action_trace": {"path": str(trace_path), "sha256": sha256_file(trace_path)},
            "video": {"path": str(movie), **video_probe},
            "renderer": "private Habitat 0.2.4 CPU Mesa llvmpipe via VLNEnv service",
        }
        _json_write(output_dir / "manifest.json", manifest)
        print(json.dumps({
            "manifest": str(output_dir / "manifest.json"),
            "video": str(movie),
            "video_sha256": video_probe["sha256"],
            "frames_decoded": video_probe["encoded_frames"],
            "steps": len(trace) - 1,
            "success": terminal["success"],
            "spl": terminal["spl"],
            "policy": weight_source["kind"],
        }, sort_keys=True))
        return manifest
    finally:
        env.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val_unseen"), default="val_unseen")
    parser.add_argument("--episode-index", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--rgb-size", type=int, default=64)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--init-seed", type=int, default=1234)
    parser.add_argument("--action-seed", type=int, default=7)
    parser.add_argument("--policy-mode", choices=("sampled", "greedy"), default="sampled")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--expected-checkpoint-sha256")
    args = parser.parse_args(argv)
    if args.episode_index is None:
        args.episode_index = 4852 if args.split == "train" else 630
    if (
        args.episode_index < 0 or not 1 <= args.max_steps <= 500
        or args.rgb_size != 64 or not 1 <= args.fps <= 60
        or args.init_seed < 0 or args.action_seed < 0
    ):
        parser.error("episode index/seeds must be nonnegative, max steps 1..500, RGB 64, fps 1..60")
    if bool(args.checkpoint) != bool(args.expected_checkpoint_sha256):
        parser.error("checkpoint and expected SHA256 must be provided together")
    if args.expected_checkpoint_sha256:
        digest = args.expected_checkpoint_sha256.lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            parser.error("expected checkpoint SHA256 must be 64 hex characters")
        args.expected_checkpoint_sha256 = digest
    return args


if __name__ == "__main__":
    rollout(parse_args())
