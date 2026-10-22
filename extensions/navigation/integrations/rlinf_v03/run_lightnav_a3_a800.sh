#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 RUN_ROOT RUN_ID EXPECTED_RUN_MANIFEST_SHA256 EXPECTED_ROLLOUT_SHA256" >&2
  exit 2
fi

RUN_ROOT=$1
RUN_ID=$2
EXPECTED_RUN_MANIFEST_SHA256=$3
EXPECTED_ROLLOUT_SHA256=$4
ENV_ROOT=/srv/navigation/envs/rlinf-v03-cu118
CODE_ROOT="$RUN_ROOT/code"
INPUT_ROOT="$RUN_ROOT/input"
TRAIN_ROOT="$RUN_ROOT/training"
ROLLOUT="$INPUT_ROOT/actor_decisions.jsonl"
INITIAL="$INPUT_ROOT/lightnav_a3_initial.npz"
RUN_MANIFEST="$INPUT_ROOT/run_manifest.json"
TELEMETRY="$RUN_ROOT/gpu_telemetry.jsonl"
TELEMETRY_SUMMARY="$RUN_ROOT/gpu_telemetry_summary.json"
STOP_FILE="$RUN_ROOT/gpu_telemetry.stop"

case "$RUN_ROOT" in
  /home/navcloud/work/*) ;;
  *) echo "RUN_ROOT must be under /home/navcloud/work" >&2; exit 3 ;;
esac
test "$RUN_ID" = "$(basename "$RUN_ROOT")"
test -n "$RUN_ID"
test "$RUN_ID" != "."
test "$RUN_ID" != ".."
case "$RUN_ID" in */*|*\\*) echo "RUN_ID must be one path-safe component" >&2; exit 3 ;; esac
test -x "$ENV_ROOT/bin/python"
test -x "$ENV_ROOT/bin/torchrun"
test -f "$ROLLOUT"
test -f "$INITIAL"
test -f "$RUN_MANIFEST"
test ! -e "$TRAIN_ROOT"
test ! -e "$TELEMETRY"
test ! -e "$STOP_FILE"

AVAILABLE_BYTES=$(df --output=avail -B1 /home/navcloud/work | tail -1 | tr -d ' ')
test "$AVAILABLE_BYTES" -ge $((16 * 1024 * 1024 * 1024))

test "$(sha256sum "$ROLLOUT" | cut -d' ' -f1)" = "$EXPECTED_ROLLOUT_SHA256"
INITIAL_SHA=$(sha256sum "$INITIAL" | cut -d' ' -f1)
test "$(sha256sum "$RUN_MANIFEST" | cut -d' ' -f1)" = "$EXPECTED_RUN_MANIFEST_SHA256"
PYTHONPATH="$CODE_ROOT" "$ENV_ROOT/bin/python" \
  "$CODE_ROOT/lightnav_a3_verify_run_manifest.py" \
  --manifest "$RUN_MANIFEST" \
  --expected-manifest-sha256 "$EXPECTED_RUN_MANIFEST_SHA256" \
  --run-id "$RUN_ID" \
  --role a800 \
  --code-root "$CODE_ROOT" \
  --rlinf-source /srv/navigation/repos/RLinf-v0.3 \
  --initial-checkpoint "$INITIAL" > "$RUN_ROOT/logs/verify_a800.json"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RLINF_ALLOWED_PHYSICAL_GPUS=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4
export PYTHONPATH="$CODE_ROOT:/srv/navigation/repos/RLinf-v0.3${PYTHONPATH:+:$PYTHONPATH}"

"$ENV_ROOT/bin/python" "$CODE_ROOT/lightnav_a3_gpu_monitor.py" sample \
  --output "$TELEMETRY" \
  --stop-file "$STOP_FILE" \
  --interval 1.0 &
MONITOR_PID=$!

stop_monitor() {
  touch "$STOP_FILE"
  wait "$MONITOR_PID" || true
}
trap stop_monitor EXIT

"$ENV_ROOT/bin/torchrun" --standalone --nproc_per_node=8 \
  "$CODE_ROOT/lightnav_a3_fsdp_train.py" \
  --rollout "$ROLLOUT" \
  --expected-rollout-sha256 "$EXPECTED_ROLLOUT_SHA256" \
  --initial-checkpoint "$INITIAL" \
  --expected-initial-sha256 "$INITIAL_SHA" \
  --run-manifest "$RUN_MANIFEST" \
  --expected-run-manifest-sha256 "$EXPECTED_RUN_MANIFEST_SHA256" \
  --output-dir "$TRAIN_ROOT" \
  --epochs 4 \
  --batch-size-per-rank 128 \
  --learning-rate 3e-4 \
  --required-world-size 8

stop_monitor
trap - EXIT

"$ENV_ROOT/bin/python" "$CODE_ROOT/lightnav_a3_gpu_monitor.py" summarize \
  --source "$TELEMETRY" \
  --output "$TELEMETRY_SUMMARY"

"$ENV_ROOT/bin/python" "$CODE_ROOT/lightnav_a3_obs_archive.py" \
  --run-root "$RUN_ROOT" \
  --run-id "$RUN_ID" \
  --phase checkpoint
