#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 RUN_ROOT WORKER_ID HABITAT_PORT LIGHTNAV_PORT EXPECTED_RUN_MANIFEST_SHA256 EXPECTED_PLAN_SHA256" >&2
  exit 2
fi

RUN_ROOT=$1
WORKER_ID=$2
HABITAT_PORT=$3
LIGHTNAV_PORT=$4
EXPECTED_RUN_SHA=$5
EXPECTED_PLAN_SHA=$6
RUN_MANIFEST="$RUN_ROOT/run_manifest.json"
PLAN="$RUN_ROOT/data/parallel_plan.json"
CODE_ROOT="$RUN_ROOT/code"
SOURCE_ROOT=/srv/navigation/repos/LightNav-0-navigation-rlt
SCENES_ROOT=/opt/InternNav/data/scene_data/mp3d_ce
HABITAT_ENV=/srv/navigation/envs/lightnav-habitat-v031
CLIENT_ENV=/opt/conda/envs/lightnav0
WORKER_NAME=$(printf 'worker_%03d' "$WORKER_ID")
DATA_ROOT="$RUN_ROOT/data/workers/$WORKER_NAME"
OUTPUT_DIR="$RUN_ROOT/rollout/$WORKER_NAME"
READY_FILE="$RUN_ROOT/status/habitat_${WORKER_NAME}.ready"
HABITAT_LOG="$RUN_ROOT/logs/habitat_${WORKER_NAME}.log"
ROLLOUT_LOG="$RUN_ROOT/logs/rollout_${WORKER_NAME}.log"
A3_CHECKPOINT="$RUN_ROOT/inputs/lightnav_a3_policy_v1.npz"

case "$RUN_ROOT" in
  /srv/navigation/runs/*) ;;
  *) echo "RUN_ROOT must be under the private Navigation-RLT runs root" >&2; exit 3 ;;
esac
[[ "$WORKER_ID" =~ ^[0-7]$ ]]
for PORT in "$HABITAT_PORT" "$LIGHTNAV_PORT"; do
  [[ "$PORT" =~ ^[0-9]+$ ]] && [[ "$PORT" -ge 1024 ]] && [[ "$PORT" -le 65535 ]]
done
[[ "$EXPECTED_RUN_SHA" =~ ^[0-9a-f]{64}$ ]]
[[ "$EXPECTED_PLAN_SHA" =~ ^[0-9a-f]{64}$ ]]
test -x "$HABITAT_ENV/bin/python"
test -x "$CLIENT_ENV/bin/python"
test -f "$RUN_MANIFEST"
test -f "$PLAN"
test -f "$DATA_ROOT/train/train.json.gz"
test -f "$DATA_ROOT/train/train_gt.json.gz"
test -d "$SCENES_ROOT/mp3d"
test -f "$A3_CHECKPOINT"
test -f "$CODE_ROOT/navigation_rlt_rollout.py"
test "$(sha256sum "$RUN_MANIFEST" | cut -d' ' -f1)" = "$EXPECTED_RUN_SHA"
test "$(sha256sum "$PLAN" | cut -d' ' -f1)" = "$EXPECTED_PLAN_SHA"
test ! -e "$OUTPUT_DIR"
test ! -e "$READY_FILE"
test -f "$RUN_ROOT/status/lightnav.ready"
mkdir -p "$RUN_ROOT/status" "$RUN_ROOT/logs" "$RUN_ROOT/rollout"

readarray -t CONTRACT < <(
  "$CLIENT_ENV/bin/python" - "$RUN_MANIFEST" "$WORKER_ID" <<'PY'
import json, sys
x=json.load(open(sys.argv[1], encoding='utf-8'))
assert x['schema_name']=='navigation_rlt_experiment_manifest' and x['schema_version']==1
assert x['boundaries']['upstream_split_read']=='train'
assert x['boundaries']['validation_split_reads']==0
assert x['boundaries']['lightnav_frozen'] is True
assert x['boundaries']['ttt_or_secondary_post_training'] is False
assert x['data']['worker_count']==8 and x['data']['episodes_per_worker']==32
assert int(sys.argv[2]) in range(8)
print(x['lightnav']['patched_head'])
print(x['lightnav']['context_patch_sha256'])
print(x['lightnav']['weight_sha256'])
print(x['lightnav']['tokenizer_sha256'])
print(x['lightnav']['processor_config_sha256'])
print(x['lightnav']['history_config_sha256'])
print(x['behavior']['checkpoint_sha256'])
PY
)
test "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" = "${CONTRACT[0]}"
test -z "$(git -C "$SOURCE_ROOT" status --porcelain)"
test "$(sha256sum "$A3_CHECKPOINT" | cut -d' ' -f1)" = "${CONTRACT[6]}"

export CUDA_VISIBLE_DEVICES=0
export HABITAT_SIM_GPU_ID=0
export NVIDIA_DRIVER_CAPABILITIES=all
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export LD_PRELOAD=/lib/x86_64-linux-gnu/libGLdispatch.so.0
export MAGNUM_LOG=quiet
export PYTHONPATH="$SOURCE_ROOT/habitat_server${PYTHONPATH:+:$PYTHONPATH}"

"$HABITAT_ENV/bin/python" -m lightnav_habitat.serve \
  --task vlnce \
  --config "$SOURCE_ROOT/habitat_server/configs/vlnce_r2r.yaml" \
  --port "$HABITAT_PORT" \
  --max-steps 160 \
  --image-height 270 \
  --image-width 480 \
  --split train \
  --data-path "$DATA_ROOT" \
  --scenes-dir "$SCENES_ROOT" \
  --ready-file "$READY_FILE" >"$HABITAT_LOG" 2>&1 &
HABITAT_PID=$!
printf '%s\n' "$HABITAT_PID" > "$RUN_ROOT/status/habitat_${WORKER_NAME}.pid"

cleanup() {
  if kill -0 "$HABITAT_PID" 2>/dev/null; then
    kill "$HABITAT_PID"
  fi
  wait "$HABITAT_PID" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 120); do
  [[ -e "$READY_FILE" ]] && break
  if ! kill -0 "$HABITAT_PID" 2>/dev/null; then
    echo "Habitat exited before readiness" >&2
    exit 4
  fi
  sleep 1
done
test -f "$READY_FILE"

export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$CLIENT_ENV/bin/python" "$CODE_ROOT/navigation_rlt_rollout.py" \
  --worker-plan "$PLAN" \
  --expected-worker-plan-sha256 "$EXPECTED_PLAN_SHA" \
  --worker-id "$WORKER_ID" \
  --habitat-server "tcp://127.0.0.1:$HABITAT_PORT" \
  --lightnav-server "ws://127.0.0.1:$LIGHTNAV_PORT" \
  --output-dir "$OUTPUT_DIR" \
  --behavior a3 \
  --a3-checkpoint "$A3_CHECKPOINT" \
  --expected-a3-checkpoint-sha256 "${CONTRACT[6]}" \
  --max-steps 160 \
  --seed 2026100201 \
  --gamma 0.99 \
  --client-id "navigation-rlt-${WORKER_NAME}" \
  --artifact-prefix-max-tokens 4096 \
  --policy-prefix-max-tokens 192 \
  --prefix-hidden-dim 2560 \
  --topology-max-nodes 128 \
  --lightnav-hook-sha256 "${CONTRACT[1]}" \
  --lightnav-weight-sha256 "${CONTRACT[2]}" \
  --tokenizer-sha256 "${CONTRACT[3]}" \
  --processor-sha256 "${CONTRACT[4]}" \
  --history-config-sha256 "${CONTRACT[5]}" \
  --consume-feature-artifacts \
  --feature-artifact-root "$RUN_ROOT/feature-artifacts" >"$ROLLOUT_LOG" 2>&1

test -f "$OUTPUT_DIR/worker_completion.json"
test -f "$OUTPUT_DIR/COMPLETED.json"
printf '%s\n' "$WORKER_NAME" > "$RUN_ROOT/status/last_completed_worker"
