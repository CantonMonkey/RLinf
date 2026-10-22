#!/usr/bin/env bash
set -euo pipefail

BASE=/srv/navigation
OUT=${1:?provide a new absolute output directory under $BASE/runs}
if [[ $(id -un) != navcloud || $OUT != "$BASE"/runs/* || -e $OUT ]]; then
  echo "Run as navcloud with a new output directory below $BASE/runs" >&2
  exit 2
fi

env -u PYTHONPATH -u RAY_ADDRESS \
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$BASE/envs/rlinf-v03-cu118/bin/python" -B \
  "$BASE/repos/rlinf-nav/integrations/rlinf_v03/train_pointnav_ppo_cpu.py" \
  --node-root "$BASE" \
  --habitat-python "$BASE/envs/habitat-cpu-egl/bin/python" \
  --output-dir "$OUT" \
  --episode-indices 50,51 --updates 1 --rollout-steps 32 \
  --max-episode-steps 64 --ppo-epochs 2 --minibatch-size 16 \
  --checkpoint-every 1 --max-total-steps 32 --threads 1 --seed 1234 \
  --render-rgbd --max-recorded-frames 64
