#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 9 ]]; then
  echo "usage: $0 RUN_ROOT SPLIT_LABEL MODE EPISODES CHECKPOINT OUTPUT_NAME PORT EXPECTED_RUN_MANIFEST_SHA256 EXPECTED_CHECKPOINT_SHA256" >&2
  exit 2
fi

RUN_ROOT=$1
SPLIT_LABEL=$2
MODE=$3
EPISODES=$4
CHECKPOINT=$5
OUTPUT_NAME=$6
PORT=$7
EXPECTED_RUN_MANIFEST_SHA256=$8
EXPECTED_CHECKPOINT_SHA256=$9
CODE_ROOT="$RUN_ROOT/code"
DATA_BUILD_ROOT="$RUN_ROOT/data"
DATA_ROOT="$DATA_BUILD_ROOT/$SPLIT_LABEL"
SPLIT_MANIFEST="$DATA_BUILD_ROOT/split_manifest.json"
RUN_MANIFEST="$RUN_ROOT/run_manifest.json"
OUTPUT_DIR="$RUN_ROOT/$OUTPUT_NAME"
STATUS_ROOT="$RUN_ROOT/status"
READY_FILE="$STATUS_ROOT/habitat_${OUTPUT_NAME}.ready"
HABITAT_LOG="$RUN_ROOT/logs/habitat_${OUTPUT_NAME}.log"
ROLLOUT_LOG="$RUN_ROOT/logs/rollout_${OUTPUT_NAME}.log"
ENV_ROOT=/srv/navigation/envs/lightnav-habitat-v031
CLIENT_ENV_ROOT=/opt/conda/envs/lightnav0

case "$RUN_ROOT" in
  /srv/navigation/*) ;;
  *) echo "RUN_ROOT must be under the private RLinf workspace" >&2; exit 3 ;;
esac
case "$SPLIT_LABEL:$MODE" in
  train_update:train|train_development:eval) ;;
  *) echo "A3 permits train_update:train or train_development:eval" >&2; exit 3 ;;
esac
[[ "$EPISODES" =~ ^[0-9]+$ ]] && [[ "$EPISODES" -gt 0 ]]
[[ "$PORT" =~ ^[0-9]+$ ]] && [[ "$PORT" -ge 1024 ]] && [[ "$PORT" -le 65535 ]]
test -x "$ENV_ROOT/bin/python"
test -x "$CLIENT_ENV_ROOT/bin/python"
test -f "$CODE_ROOT/lightnav_a3_rollout.py"
test -f "$CODE_ROOT/run_lightnav_a3_habitat_ebk180.sh"
test -f "$SPLIT_MANIFEST"
test -f "$RUN_MANIFEST"
test -f "$CHECKPOINT"
test ! -e "$OUTPUT_DIR"
test ! -e "$READY_FILE"
mkdir -p "$STATUS_ROOT" "$RUN_ROOT/logs"

MANIFEST_SHA=$(sha256sum "$SPLIT_MANIFEST" | cut -d' ' -f1)
test "$(sha256sum "$RUN_MANIFEST" | cut -d' ' -f1)" = "$EXPECTED_RUN_MANIFEST_SHA256"
CHECKPOINT_SHA=$(sha256sum "$CHECKPOINT" | cut -d' ' -f1)
test "$CHECKPOINT_SHA" = "$EXPECTED_CHECKPOINT_SHA256"
PYTHONPATH="$CODE_ROOT" "$ENV_ROOT/bin/python" \
  "$CODE_ROOT/lightnav_a3_verify_run_manifest.py" \
  --manifest "$RUN_MANIFEST" \
  --expected-manifest-sha256 "$EXPECTED_RUN_MANIFEST_SHA256" \
  --run-id "$(basename "$RUN_ROOT")" \
  --role ebk \
  --code-root "$CODE_ROOT" \
  --rlinf-source /srv/navigation/repos/RLinf-v0.3 \
  --initial-checkpoint "$RUN_ROOT/lightnav_a3_initial.npz" \
  --lightnav-source /srv/navigation/repos/LightNav-0-a3 \
  --habitat-config /srv/navigation/repos/LightNav-0-a3/habitat_server/configs/vlnce_r2r.yaml \
  > "$RUN_ROOT/logs/verify_${OUTPUT_NAME}.json"
printf '%s\n' "$OUTPUT_NAME" > "$STATUS_ROOT/current_stage"

bash "$CODE_ROOT/run_lightnav_a3_habitat_ebk180.sh" \
  "$DATA_ROOT" "$PORT" "$READY_FILE" "$SPLIT_MANIFEST" "$MANIFEST_SHA" \
  "$SPLIT_LABEL" "$RUN_MANIFEST" "$EXPECTED_RUN_MANIFEST_SHA256" \
  >"$HABITAT_LOG" 2>&1 &
HABITAT_PID=$!
printf '%s\n' "$HABITAT_PID" > "$STATUS_ROOT/habitat_${OUTPUT_NAME}.pid"

cleanup() {
  if kill -0 "$HABITAT_PID" 2>/dev/null; then
    kill "$HABITAT_PID"
  fi
  wait "$HABITAT_PID" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 120); do
  if [[ -e "$READY_FILE" ]]; then
    break
  fi
  if ! kill -0 "$HABITAT_PID" 2>/dev/null; then
    echo "Habitat exited before readiness" >&2
    exit 4
  fi
  sleep 1
done
test -f "$READY_FILE"

export PYTHONPATH="$CODE_ROOT:/srv/navigation/repos/rlinf-nav${PYTHONPATH:+:$PYTHONPATH}"
"$CLIENT_ENV_ROOT/bin/python" "$CODE_ROOT/lightnav_a3_rollout.py" \
  --habitat-server "tcp://127.0.0.1:$PORT" \
  --lightnav-server ws://127.0.0.1:18050 \
  --checkpoint "$CHECKPOINT" \
  --expected-checkpoint-sha256 "$CHECKPOINT_SHA" \
  --output-dir "$OUTPUT_DIR" \
  --mode "$MODE" \
  --split-manifest "$SPLIT_MANIFEST" \
  --expected-split-manifest-sha256 "$MANIFEST_SHA" \
  --run-manifest "$RUN_MANIFEST" \
  --expected-run-manifest-sha256 "$EXPECTED_RUN_MANIFEST_SHA256" \
  --split-label "$SPLIT_LABEL" \
  --episodes "$EPISODES" \
  --max-steps 160 >"$ROLLOUT_LOG" 2>&1

test -f "$OUTPUT_DIR/COMPLETED.json"
printf '%s\n' "$OUTPUT_NAME" > "$STATUS_ROOT/last_completed_stage"
printf '%s\n' idle > "$STATUS_ROOT/current_stage"
