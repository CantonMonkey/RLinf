#!/usr/bin/env bash
set -euo pipefail

ENV_ROOT=/srv/navigation/envs/lightnav-habitat-v031
SOURCE_ROOT=/opt/LightNav-0
RUN_ROOT=/srv/navigation/a2_lightnav_native_20260930
DATA_ROOT="$RUN_ROOT/data"
SCENES_ROOT=/opt/InternNav/data/scene_data/mp3d_ce
READY_FILE="$RUN_ROOT/habitat_15556.ready"

test "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" = \
  a645828d81a8439651172197ca80a75dc1377977
test -z "$(git -C "$SOURCE_ROOT" status --short)"
test "$(sha256sum "$DATA_ROOT/train/train.json.gz" | cut -d' ' -f1)" = \
  4520d5e10850659b4cfc2aeabd794c035494872b15759f16a6141b768010a9dc
test "$(sha256sum "$DATA_ROOT/train/train_gt.json.gz" | cut -d' ' -f1)" = \
  cc7fa0cb21aaf63218a37eef1f0dd6b27e93b0f28d30f0d2070af17932d0be06
test "$(sha256sum "$SCENES_ROOT/mp3d/mJXqzFtmKg4/mJXqzFtmKg4.glb" | cut -d' ' -f1)" = \
  8b2103d750640b5b7417bd363441b2935f8f2ff2f6357abe31b9d9630a43273e
test "$(sha256sum "$SCENES_ROOT/mp3d/1pXnuDYAj8r/1pXnuDYAj8r.glb" | cut -d' ' -f1)" = \
  b5ac077e799b57598fc3f1fb44c01e1d89ad6c118ec57a3da4f5426293cd13c2
test ! -e "$READY_FILE"

export CUDA_VISIBLE_DEVICES=0
export HABITAT_SIM_GPU_ID=0
export NVIDIA_DRIVER_CAPABILITIES=all
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export LD_PRELOAD=/lib/x86_64-linux-gnu/libGLdispatch.so.0
export MAGNUM_LOG=quiet

exec "$ENV_ROOT/bin/python" -m lightnav_habitat.serve \
  --task vlnce \
  --config "$SOURCE_ROOT/habitat_server/configs/vlnce_r2r.yaml" \
  --port 15556 \
  --max-steps 160 \
  --image-height 270 \
  --image-width 480 \
  --split train \
  --data-path "$DATA_ROOT" \
  --scenes-dir "$SCENES_ROOT" \
  --ready-file "$READY_FILE"
