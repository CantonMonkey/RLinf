#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 8 ]]; then
  echo "usage: $0 DATA_ROOT PORT READY_FILE SPLIT_MANIFEST EXPECTED_MANIFEST_SHA256 SPLIT_LABEL RUN_MANIFEST EXPECTED_RUN_MANIFEST_SHA256" >&2
  exit 2
fi

DATA_ROOT=$1
PORT=$2
READY_FILE=$3
SPLIT_MANIFEST=$4
EXPECTED_MANIFEST_SHA256=$5
SPLIT_LABEL=$6
RUN_MANIFEST=$7
EXPECTED_RUN_MANIFEST_SHA256=$8
ENV_ROOT=/srv/navigation/envs/lightnav-habitat-v031
SOURCE_ROOT=/srv/navigation/repos/LightNav-0-a3
SCENES_ROOT=/opt/InternNav/data/scene_data/mp3d_ce
EXPECTED_COMMIT=a645828d81a8439651172197ca80a75dc1377977

test "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -f "$DATA_ROOT/train/train.json.gz"
test -f "$DATA_ROOT/train/train_gt.json.gz"
test -f "$SPLIT_MANIFEST"
test -f "$RUN_MANIFEST"
test -d "$SCENES_ROOT/mp3d"
test ! -e "$READY_FILE"
test "$(sha256sum "$SPLIT_MANIFEST" | cut -d' ' -f1)" = "$EXPECTED_MANIFEST_SHA256"
test "$(sha256sum "$RUN_MANIFEST" | cut -d' ' -f1)" = "$EXPECTED_RUN_MANIFEST_SHA256"
case "$SPLIT_LABEL" in
  train_update|train_development) ;;
  *) echo "invalid split label: $SPLIT_LABEL" >&2; exit 3 ;;
esac
read -r EXPECTED_DATASET_SHA EXPECTED_GT_SHA < <(
  "$ENV_ROOT/bin/python" -c '
import json, sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
if manifest.get("schema_name") != "lightnav_a3_train_scene_split" or manifest.get("schema_version") != 1:
    raise SystemExit("invalid split manifest contract")
if manifest.get("upstream_split_read") != "train" or manifest.get("validation_split_reads") != 0:
    raise SystemExit("split manifest is not train-only")
split = manifest.get("splits", {}).get(sys.argv[2])
if not isinstance(split, dict) or split.get("habitat_split") != "train":
    raise SystemExit("split label missing from manifest")
print(split["dataset_sha256"], split["gt_sha256"])
' "$SPLIT_MANIFEST" "$SPLIT_LABEL"
)
test "$(sha256sum "$DATA_ROOT/train/train.json.gz" | cut -d' ' -f1)" = "$EXPECTED_DATASET_SHA"
test "$(sha256sum "$DATA_ROOT/train/train_gt.json.gz" | cut -d' ' -f1)" = "$EXPECTED_GT_SHA"
EXPECTED_COLLISION_DIFF_SHA=$(
  "$ENV_ROOT/bin/python" -c '
import json, sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
if manifest.get("schema_name") != "lightnav_a3_run_manifest" or manifest.get("schema_version") != 1:
    raise SystemExit("invalid outer run manifest")
print(manifest["lightnav"]["collision_diff_sha256"])
' "$RUN_MANIFEST"
)
OBSERVED_COLLISION_DIFF_SHA=$(
  git -C "$SOURCE_ROOT" diff --binary -- habitat_server/lightnav_habitat/vlnce.py | \
    sha256sum | cut -d' ' -f1
)
test "$OBSERVED_COLLISION_DIFF_SHA" = "$EXPECTED_COLLISION_DIFF_SHA"
git -C "$SOURCE_ROOT" diff --check

export CUDA_VISIBLE_DEVICES=0
export HABITAT_SIM_GPU_ID=0
export NVIDIA_DRIVER_CAPABILITIES=all
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export LD_PRELOAD=/lib/x86_64-linux-gnu/libGLdispatch.so.0
export MAGNUM_LOG=quiet
export PYTHONPATH="$SOURCE_ROOT/habitat_server${PYTHONPATH:+:$PYTHONPATH}"

exec "$ENV_ROOT/bin/python" -m lightnav_habitat.serve \
  --task vlnce \
  --config "$SOURCE_ROOT/habitat_server/configs/vlnce_r2r.yaml" \
  --port "$PORT" \
  --max-steps 160 \
  --image-height 270 \
  --image-width 480 \
  --split train \
  --data-path "$DATA_ROOT" \
  --scenes-dir "$SCENES_ROOT" \
  --ready-file "$READY_FILE"
