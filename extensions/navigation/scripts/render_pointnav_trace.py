"""Render recorded Habitat PointNav poses as an honest top-down trace video.

This draws only the agent poses and goal from a JSONL rollout. It has no scene
mesh, obstacle map, RGB camera frame, or Habitat rendering dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 880, 520
PLOT = (50, 65, 625, 460)


def load_record(path: Path, episode_index: int) -> dict:
    reset = None
    event_steps = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("episode_index") != episode_index:
                continue
            if "event" not in record:
                return record
            if record["event"] == "reset":
                if reset is not None:
                    raise ValueError("duplicate checkpoint evaluation reset")
                reset = record
            elif record["event"] == "step":
                event_steps.append(record)
            else:
                raise ValueError("unknown checkpoint evaluation event")
    if reset is None or not event_steps:
        raise ValueError(f"episode_index={episode_index} not found or incomplete in {path}")
    if not (event_steps[-1]["terminated"] or event_steps[-1]["truncated"]):
        raise ValueError("checkpoint evaluation trace has no terminal step")
    if "diagnostic" not in reset or any("diagnostic" not in step for step in event_steps):
        raise ValueError("checkpoint evaluation trace lacks poses; rerun with --record-poses")
    identity = (reset["episode_id"], reset["scene_id"])
    initial = reset["diagnostic"]
    goal = initial["goal_position_xyz"]
    steps = []
    for expected_step, event in enumerate(event_steps, start=1):
        diagnostic = event["diagnostic"]
        if event["step"] != expected_step or (
            event["episode_id"], event["scene_id"]
        ) != identity or (
            diagnostic["episode_id"], diagnostic["scene_id"]
        ) != identity or diagnostic["goal_position_xyz"] != goal:
            raise ValueError("checkpoint evaluation trace episode/step/goal mismatch")
        info = event["info"]
        steps.append({
            "step": expected_step,
            "action": event["action"],
            "pose": {
                "position_xyz": diagnostic["position_xyz"],
                "rotation_xyzw": diagnostic["rotation_xyzw"],
            },
            "metrics": {
                "distance_to_goal": info["geodesic_distance"],
                "success": info["success"],
                "spl": info["spl"],
                "collisions": {
                    "count": info["collision_count"],
                    "is_collision": info["collision"],
                },
            },
        })
    return {
        "episode_index": episode_index,
        "episode_id": reset["episode_id"],
        "scene_id": reset["scene_id"],
        "initial_pose": {
            "position_xyz": initial["position_xyz"],
            "rotation_xyzw": initial["rotation_xyzw"],
        },
        "goal_position_xyz": goal,
        "steps": steps,
        "outcome": {"termination_reason": event_steps[-1]["info"]["termination_reason"]},
        "policy": "RLinf PPO checkpoint diagnostic",
    }


def font(size: int) -> ImageFont.ImageFont:
    candidates = (
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def xz(position: list[float]) -> tuple[float, float]:
    if len(position) != 3 or not all(math.isfinite(float(x)) for x in position):
        raise ValueError("invalid position_xyz")
    return float(position[0]), float(position[2])


def forward_xz(rotation: list[float]) -> tuple[float, float]:
    """Rotate Habitat's local forward vector (0, 0, -1) by xyzw quaternion."""
    x, y, z, w = (float(v) for v in rotation)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0 or not math.isfinite(norm):
        raise ValueError("invalid rotation_xyzw")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return -2 * (x * z + y * w), -(1 - 2 * (x * x + y * y))


def draw_frame(record: dict, visible_steps: int) -> Image.Image:
    steps = record["steps"]
    initial = record["initial_pose"]
    poses = [initial] + [step["pose"] for step in steps]
    points = [xz(pose["position_xyz"]) for pose in poses]
    goal = xz(record["goal_position_xyz"])
    xs = [point[0] for point in points] + [goal[0]]
    zs = [point[1] for point in points] + [goal[1]]
    span_x = max(max(xs) - min(xs), 2.0)
    span_z = max(max(zs) - min(zs), 2.0)
    usable_w = PLOT[2] - PLOT[0] - 35
    usable_h = PLOT[3] - PLOT[1] - 35
    scale = min(usable_w / span_x, usable_h / span_z)
    mid_x = (min(xs) + max(xs)) / 2
    mid_z = (min(zs) + max(zs)) / 2
    center_px = ((PLOT[0] + PLOT[2]) / 2, (PLOT[1] + PLOT[3]) / 2)

    def pixel(point: tuple[float, float]) -> tuple[int, int]:
        return (
            round(center_px[0] + scale * (point[0] - mid_x)),
            round(center_px[1] + scale * (point[1] - mid_z)),
        )

    image = Image.new("RGB", (WIDTH, HEIGHT), "#f6f8fa")
    draw = ImageDraw.Draw(image)
    title_font, body_font, small_font = font(23), font(16), font(13)
    draw.text((35, 20), "Habitat PointNav pose trace", font=title_font, fill="#102a43")
    draw.rectangle(PLOT, fill="white", outline="#cbd5e1", width=2)

    # The trace is drawn without wall geometry. Keep a fixed scale for a stable
    # video, but draw the path only up to the current timestep.
    visible_steps = max(0, min(visible_steps, len(steps)))
    visible_points = [pixel(point) for point in points[: visible_steps + 1]]
    if len(visible_points) >= 2:
        draw.line(visible_points, fill="#2563eb", width=4, joint="curve")
    gx, gy = pixel(goal)
    draw.ellipse((gx - 9, gy - 9, gx + 9, gy + 9), fill="#dc2626", outline="white", width=2)
    sx, sy = pixel(points[0])
    draw.rectangle((sx - 6, sy - 6, sx + 6, sy + 6), fill="#16a34a", outline="white", width=2)
    cx, cy = visible_points[-1]
    draw.ellipse((cx - 8, cy - 8, cx + 8, cy + 8), fill="#f59e0b", outline="#78350f", width=2)
    heading = forward_xz(poses[visible_steps]["rotation_xyzw"])
    draw.line((cx, cy, cx + 19 * heading[0], cy + 19 * heading[1]), fill="#78350f", width=3)
    for step in steps[:visible_steps]:
        if step["metrics"]["collisions"]["is_collision"]:
            px, py = pixel(xz(step["pose"]["position_xyz"]))
            draw.line((px - 5, py - 5, px + 5, py + 5), fill="#dc2626", width=2)
            draw.line((px - 5, py + 5, px + 5, py - 5), fill="#dc2626", width=2)

    px = 650
    labels = [
        f"Episode: {record['episode_index']} / {record['episode_id']}",
        f"Step: {visible_steps}/{len(steps)}",
    ]
    if record.get("policy") == "RLinf PPO checkpoint diagnostic":
        labels.append("Policy: PPO step-1")
    if visible_steps:
        step = steps[visible_steps - 1]
        metrics = step["metrics"]
        labels += [
            f"Action: {step['action']}",
            f"Geodesic: {metrics['distance_to_goal']:.2f} m",
            f"Success: {int(metrics['success'])}",
            f"SPL: {metrics['spl']:.3f}",
            f"Collisions: {metrics['collisions']['count']}",
        ]
    else:
        labels.append("Action: reset")
    if visible_steps == len(steps):
        labels.append(f"Result: {record['outcome']['termination_reason']}")
    for offset, label in enumerate(labels):
        draw.text((px, 82 + 29 * offset), label, font=body_font, fill="#102a43")
    draw.text((px, 370), "Green  start", font=small_font, fill="#16a34a")
    draw.text((px, 390), "Red    goal", font=small_font, fill="#dc2626")
    draw.text((px, 410), "Orange agent", font=small_font, fill="#a16207")
    draw.text((35, 485), "Pose log only - no scene geometry or camera RGB. +x right, +z down.", font=small_font, fill="#475569")
    return image


def render(record: dict, output: Path, png: Path, fps: int, hold_seconds: int) -> dict:
    if fps < 1 or hold_seconds < 0:
        raise ValueError("fps must be positive and hold_seconds nonnegative")
    output.parent.mkdir(parents=True, exist_ok=True)
    png.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{WIDTH}x{HEIGHT}",
        "-framerate", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None
    assert process.stderr is not None
    last_frame = None
    try:
        for index in range(len(record["steps"]) + 1):
            last_frame = draw_frame(record, index)
            process.stdin.write(last_frame.tobytes())
        assert last_frame is not None
        last_frame.save(png)
        for _ in range(fps * hold_seconds):
            process.stdin.write(last_frame.tobytes())
        process.stdin.close()
        error = process.stderr.read().decode("utf-8", errors="replace")
        if process.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {error}")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    return {
        "episode_index": record["episode_index"],
        "frames": len(record["steps"]) + 1 + fps * hold_seconds,
        "video_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "png_sha256": hashlib.sha256(png.read_bytes()).hexdigest(),
        "video_path": str(output),
        "png_path": str(png),
        "visual_habitat_validated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--png", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--hold-seconds", type=int, default=2)
    args = parser.parse_args()
    record = load_record(args.input, args.episode_index)
    print(json.dumps(render(record, args.output, args.png, args.fps, args.hold_seconds), sort_keys=True))


if __name__ == "__main__":
    main()
