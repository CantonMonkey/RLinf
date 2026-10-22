#!/usr/bin/env python3
"""Small deterministic CPU OSMesa smoke test for MuJoCo."""
import argparse
import os
import time
from pathlib import Path

if os.environ.get("MUJOCO_GL") != "osmesa":
    raise RuntimeError("Set MUJOCO_GL=osmesa before starting Python")

import mujoco
import numpy as np
from PIL import Image

XML = r"""
<mujoco model="cpu_osmesa_smoke">
  <option timestep="0.02" gravity="0 0 -9.81"/>
  <visual>
    <global offwidth="64" offheight="64"/>
  </visual>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="1 1 1"/>
    <geom type="plane" size="2 2 0.1" rgba="0.25 0.35 0.45 1"/>
    <body name="ball" pos="0 0 0.8">
      <freejoint/>
      <geom type="sphere" size="0.14" mass="0.5" rgba="0.95 0.2 0.1 1"/>
    </body>
    <camera name="fixed" pos="0 -2 1.5" xyaxes="1 0 0 0 0.6 0.8"/>
  </worldbody>
</mujoco>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, choices=(1, 30), default=1)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    videos = root / "videos"
    frame_dir = root / "data" / "tmp" / "smoke_frames"
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=64, width=64)
    if args.frames == 30:
        frame_dir.mkdir(mode=0o750, parents=True, exist_ok=True)
    t0 = time.perf_counter()
    first = None
    last = None
    for index in range(args.frames):
        for _ in range(2):
            mujoco.mj_step(model, data)
        renderer.update_scene(data, camera="fixed")
        image = np.asarray(renderer.render()).copy()
        if image.shape != (64, 64, 3):
            raise RuntimeError(f"unexpected RGB shape: {image.shape}")
        if first is None:
            first = image
            Image.fromarray(image).save(videos / "smoke.png")
        if args.frames == 30:
            Image.fromarray(image).save(frame_dir / f"frame_{index:03d}.png")
        last = image
    elapsed = time.perf_counter() - t0
    renderer.close()
    assert first is not None and last is not None
    print(
        f"frames={args.frames} size=64x64 elapsed_s={elapsed:.3f} "
        f"render_fps={args.frames / elapsed:.2f} "
        f"first_std={float(first.std()):.2f} "
        f"last_std={float(last.std()):.2f} "
        f"mean_delta={float(np.abs(last.astype(int)-first.astype(int)).mean()):.2f}"
    )


if __name__ == "__main__":
    main()
