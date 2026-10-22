#!/usr/bin/env bash
set -euo pipefail

if [ "$(id -un)" != "navcloud" ]; then
  echo "Run this smoke test as navcloud." >&2
  exit 1
fi

root="$(cd "$(dirname "$0")/.." && pwd)"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export CUDA_VISIBLE_DEVICES=-1
export LIBGL_ALWAYS_SOFTWARE=1
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH="$root/lib/usr/lib/x86_64-linux-gnu"

"$root/envs/mujoco-cpu/bin/python" "$root/repos/mujoco_cpu_smoke.py" --frames 1
"$root/envs/mujoco-cpu/bin/python" "$root/repos/mujoco_cpu_smoke.py" --frames 30
ffmpeg -hide_banner -loglevel error -framerate 10 \
  -i "$root/data/tmp/smoke_frames/frame_%03d.png" \
  -frames:v 30 -c:v libx264 -preset ultrafast -threads 1 \
  -pix_fmt yuv420p -movflags +faststart -y "$root/videos/smoke.mp4"
ffprobe -v error -select_streams v:0 \
  -show_entries stream=codec_name,width,height,avg_frame_rate,nb_frames \
  -of default=noprint_wrappers=1 "$root/videos/smoke.mp4"
