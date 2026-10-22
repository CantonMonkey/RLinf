#!/usr/bin/env bash
set -euo pipefail

ENV_ROOT=/srv/navigation/envs/lightnav-habitat-v031
SOURCE_ROOT=/srv/navigation/repos/LightNav-0-a2-l0
RUN_ROOT=/srv/navigation/a2_lightnav_native_20260930
DATA_ROOT="$RUN_ROOT/data"
SCENES_ROOT=/cache/navigation-example/scene_datasets
READY_FILE="$RUN_ROOT/habitat_15556.ready"

test "$(sha256sum "$DATA_ROOT/train/train.json.gz" | cut -d' ' -f1)" = \
  4520d5e10850659b4cfc2aeabd794c035494872b15759f16a6141b768010a9dc
test "$(sha256sum "$DATA_ROOT/train/train_gt.json.gz" | cut -d' ' -f1)" = \
  cc7fa0cb21aaf63218a37eef1f0dd6b27e93b0f28d30f0d2070af17932d0be06
test ! -e "$READY_FILE"

export CUDA_VISIBLE_DEVICES=0
export HABITAT_SIM_GPU_ID=0
export NVIDIA_DRIVER_CAPABILITIES=all
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
