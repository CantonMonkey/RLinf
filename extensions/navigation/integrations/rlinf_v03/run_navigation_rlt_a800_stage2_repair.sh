#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

usage() {
  cat >&2 <<EOF
usage: $0 RUN_ROOT PHASE RUN_MANIFEST EXPECTED_RUN_MANIFEST_SHA256 REPLAY_MANIFEST EXPECTED_REPLAY_MANIFEST_SHA256 [STAGE1_CHECKPOINT EXPECTED_STAGE1_CHECKPOINT_SHA256]

This repair wrapper accepts only PHASE=stage2 and requires the two Stage-1
checkpoint arguments.  Every run input must resolve below
/srv/navigation/runs/.  It may run only from an
independent code-stage2-repair-<commit> sibling and under the fixed repair
launch lock.
EOF
}

if [[ $# -ne 8 ]]; then
  usage
  exit 2
fi

RUNS_ROOT=/srv/navigation/runs
ENV_ROOT=/srv/navigation/envs/rlinf-v03-cu118
RLINF_SOURCE=/srv/navigation/repos/RLinf-v0.3
RUN_ROOT_RAW=$1
PHASE=$2
RUN_MANIFEST_RAW=$3
EXPECTED_RUN_MANIFEST_SHA256=$4
REPLAY_MANIFEST_RAW=$5
EXPECTED_REPLAY_MANIFEST_SHA256=$6
STAGE1_CHECKPOINT_RAW=${7:-}
EXPECTED_STAGE1_CHECKPOINT_SHA256=${8:-}

if [[ "$PHASE" != stage2 ]]; then
  echo "Stage2 repair wrapper accepts only PHASE=stage2" >&2
  exit 2
fi

sha256_argument() {
  local value=$1
  local name=$2
  if [[ ! "$value" =~ ^[0-9a-f]{64}$ ]]; then
    echo "$name must be a lowercase SHA-256" >&2
    exit 2
  fi
}

sha256_argument "$EXPECTED_RUN_MANIFEST_SHA256" EXPECTED_RUN_MANIFEST_SHA256
sha256_argument "$EXPECTED_REPLAY_MANIFEST_SHA256" EXPECTED_REPLAY_MANIFEST_SHA256
sha256_argument "$EXPECTED_STAGE1_CHECKPOINT_SHA256" EXPECTED_STAGE1_CHECKPOINT_SHA256

if [[ -L "$RUN_ROOT_RAW" || ! -d "$RUN_ROOT_RAW" ]]; then
  echo "RUN_ROOT must be an existing non-symlink directory" >&2
  exit 3
fi
RUN_ROOT=$(realpath -e -- "$RUN_ROOT_RAW")
case "$RUN_ROOT" in
  "$RUNS_ROOT"/*) ;;
  *)
    echo "RUN_ROOT must resolve below $RUNS_ROOT/" >&2
    exit 3
    ;;
esac

contained_regular_file() {
  local raw=$1
  local name=$2
  if [[ -L "$raw" || ! -f "$raw" ]]; then
    echo "$name must be an existing non-symlink regular file" >&2
    exit 3
  fi
  local resolved
  resolved=$(realpath -e -- "$raw")
  case "$resolved" in
    "$RUNS_ROOT"/*) printf '%s\n' "$resolved" ;;
    *)
      echo "$name must resolve below $RUNS_ROOT/" >&2
      exit 3
      ;;
  esac
}

RUN_MANIFEST=$(contained_regular_file "$RUN_MANIFEST_RAW" RUN_MANIFEST)
REPLAY_MANIFEST=$(contained_regular_file "$REPLAY_MANIFEST_RAW" REPLAY_MANIFEST)
STAGE1_CHECKPOINT=$(contained_regular_file \
  "$STAGE1_CHECKPOINT_RAW" STAGE1_CHECKPOINT)

WRAPPER=$(realpath -e -- "$0")
CODE_ROOT=$(dirname -- "$WRAPPER")
if [[ -L "$CODE_ROOT" || ! -d "$CODE_ROOT" ]]; then
  echo "repair code root must be an existing non-symlink directory" >&2
  exit 3
fi
CODE_ROOT=$(realpath -e -- "$CODE_ROOT")
case "$CODE_ROOT" in
  "$RUN_ROOT"/code-stage2-repair-*) ;;
  *)
    echo "repair wrapper must run from a direct code-stage2-repair-* sibling" >&2
    exit 3
    ;;
esac
if [[ "$(dirname -- "$CODE_ROOT")" != "$RUN_ROOT" ]]; then
  echo "repair code root must be a direct child of RUN_ROOT" >&2
  exit 3
fi
if [[ "$CODE_ROOT" == "$RUN_ROOT/code" ]]; then
  echo "repair wrapper must not run from the original code root" >&2
  exit 3
fi
NEW_LOCK="$RUN_ROOT/.launch-navigation-rlt-stage2-repair-v2"
OLD_LOCK="$RUN_ROOT/.launch-navigation-rlt-stage2"
if [[ -e "$OLD_LOCK" || -L "$OLD_LOCK" ]]; then
  echo "legacy Stage2 launch lock exists; repair launch is forbidden" >&2
  exit 3
fi
if [[ -L "$NEW_LOCK" || ! -d "$NEW_LOCK" ]]; then
  echo "fixed Stage2 repair launch lock is absent or unsafe" >&2
  exit 3
fi
TRAINER="$CODE_ROOT/navigation_rlt_fsdp_train.py"
POLICY="$CODE_ROOT/navigation_rlt_policy.py"
REPLAY_CODE="$CODE_ROOT/navigation_rlt_replay.py"
MONITOR="$CODE_ROOT/lightnav_a3_gpu_monitor.py"
for source_file in "$TRAINER" "$POLICY" "$REPLAY_CODE" "$MONITOR" "$WRAPPER"; do
  if [[ -L "$source_file" || ! -f "$source_file" ]]; then
    echo "required source is absent, non-regular, or a symlink: $source_file" >&2
    exit 3
  fi
  source_file_resolved=$(realpath -e -- "$source_file")
  case "$source_file_resolved" in
    "$CODE_ROOT"/*) ;;
    *) echo "required source escapes the repair code root: $source_file" >&2; exit 3 ;;
  esac
done

PYTHON="$ENV_ROOT/bin/python"
TORCHRUN="$ENV_ROOT/bin/torchrun"
if [[ ! -x "$PYTHON" || ! -x "$TORCHRUN" ]]; then
  echo "the fixed RLinf Python environment is incomplete" >&2
  exit 3
fi
if [[ ! -d "$RLINF_SOURCE/rlinf" ]]; then
  echo "the fixed RLinf source tree is absent" >&2
  exit 3
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required" >&2
  exit 3
fi

TELEMETRY_INTERVAL_S=${NAVRLT_TELEMETRY_INTERVAL_S:-30}
"$PYTHON" - "$TELEMETRY_INTERVAL_S" <<'PY'
import math
import sys

try:
    value = float(sys.argv[1])
except ValueError as error:
    raise SystemExit("NAVRLT_TELEMETRY_INTERVAL_S must be numeric") from error
if not math.isfinite(value) or not 1.0 <= value <= 3600.0:
    raise SystemExit("NAVRLT_TELEMETRY_INTERVAL_S must be within [1, 3600]")
PY

mapfile -t GPU_INDICES < <(
  nvidia-smi --query-gpu=index --format=csv,noheader,nounits | tr -d ' '
)
if [[ ${#GPU_INDICES[@]} -ne 8 ]]; then
  echo "Navigation-RLT A800 launch requires exactly eight visible physical GPUs" >&2
  exit 3
fi
for index in 0 1 2 3 4 5 6 7; do
  if [[ "${GPU_INDICES[$index]}" != "$index" ]]; then
    echo "physical GPU indices must be exactly 0..7" >&2
    exit 3
  fi
done

AVAILABLE_BYTES=$(df --output=avail -B1 "$RUN_ROOT" | tail -1 | tr -d ' ')
if [[ ! "$AVAILABLE_BYTES" =~ ^[0-9]+$ ]] || \
   (( AVAILABLE_BYTES < 16 * 1024 * 1024 * 1024 )); then
  echo "at least 16 GiB of free run storage is required" >&2
  exit 3
fi

PREFLIGHT_JSON=$("$PYTHON" - \
  "$RUNS_ROOT" "$RUN_ROOT" "$PHASE" "$CODE_ROOT" \
  "$RUN_MANIFEST" "$EXPECTED_RUN_MANIFEST_SHA256" \
  "$REPLAY_MANIFEST" "$EXPECTED_REPLAY_MANIFEST_SHA256" \
  "$STAGE1_CHECKPOINT" "$EXPECTED_STAGE1_CHECKPOINT_SHA256" <<'PY'
import hashlib
import json
from pathlib import Path
import re
import sys

(runs_root_s, run_root_s, phase, code_root_s, run_manifest_s,
 expected_run_sha, replay_manifest_s, expected_replay_sha,
 checkpoint_s, expected_checkpoint_sha) = sys.argv[1:]
runs_root = Path(runs_root_s).resolve(strict=True)
run_root = Path(run_root_s).resolve(strict=True)
code_root = Path(code_root_s).resolve(strict=True)

def fail(message):
    raise SystemExit(message)

def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def contained_file(value, name):
    raw = Path(value)
    if raw.is_symlink():
        fail(f'{name} must not be a symlink')
    try:
        path = raw.resolve(strict=True)
        path.relative_to(runs_root)
    except (OSError, ValueError):
        fail(f'{name} escapes the fixed runs root or is missing')
    if not path.is_file():
        fail(f'{name} must be a regular file')
    return path

def reject_constant(value):
    fail(f'non-finite JSON constant is forbidden: {value}')

def load_json(path, name):
    try:
        value = json.loads(path.read_text(encoding='utf-8'), parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        fail(f'{name} is not strict JSON: {error}')
    if not isinstance(value, dict):
        fail(f'{name} must contain an object')
    return value

run_manifest = contained_file(run_manifest_s, 'run manifest')
replay_manifest = contained_file(replay_manifest_s, 'replay manifest')
if sha256(run_manifest) != expected_run_sha:
    fail('run manifest SHA-256 differs')
if sha256(replay_manifest) != expected_replay_sha:
    fail('replay manifest SHA-256 differs')
run = load_json(run_manifest, 'run manifest')
replay = load_json(replay_manifest, 'replay manifest')
if run.get('phase') != phase or run.get('required_world_size') != 8:
    fail('run manifest phase/world size differs from the A800 launch')
if not isinstance(run.get('run_id'), str) or not run['run_id']:
    fail('run manifest run_id is absent')
expected_representation = 'prefix_hidden' if phase == 'stage1' else 'frozen_z_rl'
run_replay = run.get('replay')
if not isinstance(run_replay, dict) or run_replay.get('manifest_sha256') != expected_replay_sha:
    fail('run manifest does not bind the replay manifest SHA-256')
if run_replay.get('representation_kind') != expected_representation:
    fail('run manifest replay representation differs from the phase')
if replay.get('representation_kind') != expected_representation:
    fail('replay representation differs from the phase')
if replay.get('run_id') != run['run_id']:
    fail('run and replay identifiers differ')
if replay.get('source_fingerprint') != run_replay.get('source_fingerprint'):
    fail('run and replay source fingerprints differ')
if replay.get('split_label') != 'train_update' or replay.get('validation_split_reads') != 0:
    fail('replay violates the train-only boundary')
if replay.get('expected_episode_count') != replay.get('completed_episode_count'):
    fail('replay fixed episode budget is incomplete')
if (
    replay.get('expected_episode_count') != 256
    or replay.get('behavior_policy_id') != 'navigation_rlt_warmstart_lineage_v1'
    or replay.get('behavior_policy_versions') != [100001]
):
    fail('Stage2 repair requires the fixed 256-episode A3 replay')
training = run.get('training')
expected_additions = {
    'cql_alpha': 1.0,
    'cql_temperature': 1.0,
    'minimum_behavior_gate_count': 1,
}
if not isinstance(training, dict) or any(
    training.get(name) != value for name, value in expected_additions.items()
):
    fail('Stage2 repair objective/CQL contract differs')
objective_schema = training.get('objective_schema')
if objective_schema == 'proposal_conditioned_finite_candidate_cql_v1':
    if 'behavior_kl_weight' in training:
        fail('historical Stage2 objective must not carry behavior KL')
elif objective_schema == 'proposal_conditioned_finite_candidate_cql_behavior_kl_v2':
    behavior_kl_weight = training.get('behavior_kl_weight')
    if type(behavior_kl_weight) not in (int, float) or behavior_kl_weight != 1.0:
        fail('Stage2 behavior KL weight must be exactly 1.0')
else:
    fail('Stage2 repair objective/CQL contract differs')
support = run.get('stage2_support')
support_fields = {
    'eligibility_schema', 'gate_names', 'gate_counts',
    'gate_support_mask', 'minimum_behavior_gate_count',
}
if not isinstance(support, dict) or set(support) != support_fields:
    fail('Stage2 behavior support contract is absent')
counts = support.get('gate_counts')
mask = support.get('gate_support_mask')
minimum = support.get('minimum_behavior_gate_count')
if (
    support.get('eligibility_schema')
       != 'credit_mask_true_and_requested_gate_equals_executed_gate_v1'
    or support.get('gate_names') != ['KEEP', 'ADJUST', 'STOP', 'RECOVER']
    or type(minimum) is not int
    or minimum != 1
    or not isinstance(counts, list)
    or len(counts) != 4
    or any(type(count) is not int or count < 0 for count in counts)
    or not isinstance(mask, list)
    or len(mask) != 4
    or any(type(enabled) is not bool for enabled in mask)
    or mask != [count >= minimum for count in counts]
    or not any(mask)
    or counts[3] != 0
    or mask[3] is not False
):
    fail('Stage2 repair requires RECOVER support=false from exact behavior counts')
shards = replay.get('shards')
if not isinstance(shards, list) or not shards:
    fail('replay shard inventory is empty')
paths = set()
hashes = set()
for index, receipt in enumerate(shards):
    if not isinstance(receipt, dict):
        fail(f'replay shard {index} receipt is invalid')
    path = contained_file(receipt.get('path', ''), f'replay shard {index}')
    shard_sha = receipt.get('sha256')
    if not isinstance(shard_sha, str) or re.fullmatch(r'[0-9a-f]{64}', shard_sha) is None:
        fail(f'replay shard {index} SHA-256 is invalid')
    identity = str(path)
    if identity in paths or shard_sha in hashes:
        fail('replay shard path or SHA-256 is duplicated')
    paths.add(identity)
    hashes.add(shard_sha)

implementation = run.get('implementation')
source_paths = {
    'trainer_sha256': code_root / 'navigation_rlt_fsdp_train.py',
    'policy_sha256': code_root / 'navigation_rlt_policy.py',
    'replay_sha256': code_root / 'navigation_rlt_replay.py',
}
if not isinstance(implementation, dict) or set(implementation) != set(source_paths):
    fail('run manifest implementation contract differs')
source_hashes = {}
for name, path in source_paths.items():
    path = contained_file(path, name)
    actual = sha256(path)
    if implementation.get(name) != actual:
        fail(f'{name} differs from the run manifest')
    source_hashes[name] = actual

checkpoint = None
if phase == 'stage1':
    if 'stage1_checkpoint' in run or checkpoint_s or expected_checkpoint_sha:
        fail('stage1 must not bind a Stage-1 checkpoint input')
else:
    checkpoint = contained_file(checkpoint_s, 'Stage-1 checkpoint')
    if sha256(checkpoint) != expected_checkpoint_sha:
        fail('Stage-1 checkpoint SHA-256 differs')
    binding = run.get('stage1_checkpoint')
    if binding != {'sha256': expected_checkpoint_sha}:
        fail('run manifest Stage-1 checkpoint binding differs')

result = {
    'schema_name': 'navigation_rlt_a800_preflight',
    'schema_version': 1,
    'status': 'passed',
    'run_root': str(run_root),
    'phase': phase,
    'run_id': run['run_id'],
    'run_manifest': {'path': str(run_manifest), 'sha256': expected_run_sha},
    'replay_manifest': {
        'path': str(replay_manifest),
        'sha256': expected_replay_sha,
        'representation_kind': expected_representation,
        'episode_count': replay['completed_episode_count'],
        'transition_count': replay['transition_count'],
        'shard_count': len(shards),
        'content_verification_owner': 'navigation_rlt_fsdp_train.py',
    },
    'stage1_checkpoint': None if checkpoint is None else {
        'path': str(checkpoint), 'sha256': expected_checkpoint_sha,
    },
    'stage2_support': support,
    'implementation': source_hashes,
}
print(json.dumps(result, sort_keys=True, allow_nan=False))
PY
)

STAGE_ROOT="$RUN_ROOT/navigation_rlt_$PHASE"
TRAIN_ROOT="$STAGE_ROOT/training"
TRAIN_LOG="$STAGE_ROOT/torchrun.log"
MONITOR_LOG="$STAGE_ROOT/gpu_monitor.log"
TELEMETRY="$STAGE_ROOT/gpu_telemetry.jsonl"
TELEMETRY_SUMMARY="$STAGE_ROOT/gpu_telemetry_summary.json"
STOP_FILE="$STAGE_ROOT/gpu_telemetry.stop"
PREFLIGHT="$STAGE_ROOT/preflight.json"
VERIFICATION="$STAGE_ROOT/verification.json"
STAGE_MARKER="$STAGE_ROOT/STAGE.json"

if [[ -e "$STAGE_ROOT" ]]; then
  echo "refusing to overwrite stage output: $STAGE_ROOT" >&2
  exit 4
fi
mkdir "$STAGE_ROOT"
printf '%s\n' "$PREFLIGHT_JSON" > "$PREFLIGHT"
PREFLIGHT_SHA256=$(sha256sum "$PREFLIGHT" | cut -d' ' -f1)
WRAPPER_SHA256=$(sha256sum "$WRAPPER" | cut -d' ' -f1)
MONITOR_SHA256=$(sha256sum "$MONITOR" | cut -d' ' -f1)
STARTED_UNIX_S=$(date +%s)

TRAIN_RC=-1
MONITOR_RC=-1
SUMMARY_RC=-1
MONITOR_PID=
FINALIZED=0
CHECKPOINT_PATH=
CHECKPOINT_SHA256=
VERIFICATION_SHA256=

write_stage_marker() {
  local status=$1
  local exit_code=$2
  "$PYTHON" - \
    "$STAGE_MARKER" "$status" "$exit_code" "$STARTED_UNIX_S" \
    "$RUN_ROOT" "$PHASE" "$RUN_MANIFEST" "$EXPECTED_RUN_MANIFEST_SHA256" \
    "$REPLAY_MANIFEST" "$EXPECTED_REPLAY_MANIFEST_SHA256" \
    "$STAGE1_CHECKPOINT" "$EXPECTED_STAGE1_CHECKPOINT_SHA256" \
    "$TRAIN_ROOT" "$TRAIN_LOG" "$TELEMETRY" "$TELEMETRY_SUMMARY" \
    "$PREFLIGHT" "$PREFLIGHT_SHA256" "$WRAPPER_SHA256" "$MONITOR_SHA256" \
    "$TRAIN_RC" "$MONITOR_RC" "$SUMMARY_RC" \
    "$CHECKPOINT_PATH" "$CHECKPOINT_SHA256" \
    "$VERIFICATION" "$VERIFICATION_SHA256" <<'PY'
import json
import os
from pathlib import Path
import sys
import tempfile
import time

(marker_s, status, exit_code, started, run_root, phase, run_manifest,
 run_manifest_sha, replay_manifest, replay_manifest_sha, checkpoint,
 checkpoint_sha, train_root, train_log, telemetry, telemetry_summary,
 preflight, preflight_sha, wrapper_sha, monitor_sha, train_rc, monitor_rc,
 summary_rc, output_checkpoint, output_checkpoint_sha, verification,
 verification_sha) = sys.argv[1:]

def optional_rc(value):
    parsed = int(value)
    return None if parsed < 0 else parsed

value = {
    'schema_name': 'navigation_rlt_a800_stage',
    'schema_version': 1,
    'status': status,
    'exit_code': int(exit_code),
    'phase': phase,
    'run_root': run_root,
    'run_manifest': {'path': run_manifest, 'sha256': run_manifest_sha},
    'replay_manifest': {'path': replay_manifest, 'sha256': replay_manifest_sha},
    'stage1_checkpoint': None if not checkpoint else {
        'path': checkpoint, 'sha256': checkpoint_sha,
    },
    'output': {
        'training': train_root,
        'log': train_log,
        'telemetry': telemetry,
        'telemetry_summary': telemetry_summary,
        'checkpoint': None if not output_checkpoint else {
            'path': output_checkpoint, 'sha256': output_checkpoint_sha,
        },
    },
    'preflight': {'path': preflight, 'sha256': preflight_sha},
    'verification': None if not verification_sha else {
        'path': verification, 'sha256': verification_sha,
    },
    'implementation': {
        'wrapper_sha256': wrapper_sha,
        'gpu_monitor_sha256': monitor_sha,
    },
    'process_exit_codes': {
        'torchrun': optional_rc(train_rc),
        'gpu_monitor': optional_rc(monitor_rc),
        'telemetry_summary': optional_rc(summary_rc),
    },
    'started_unix_s': int(started),
    'updated_unix_s': time.time(),
}
marker = Path(marker_s)
payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
if status == 'running':
    with marker.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(payload)
else:
    descriptor, name = tempfile.mkstemp(prefix=marker.name + '.staging-', dir=marker.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8', newline='\n') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, marker)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise
PY
}

stop_monitor() {
  if [[ -z "$MONITOR_PID" ]]; then
    return 0
  fi
  touch "$STOP_FILE"
  wait "$MONITOR_PID"
  local result=$?
  MONITOR_PID=
  return "$result"
}

summarize_telemetry() {
  if [[ ! -s "$TELEMETRY" ]]; then
    return 5
  fi
  "$PYTHON" "$MONITOR" summarize \
    --source "$TELEMETRY" \
    --output "$TELEMETRY_SUMMARY" >> "$MONITOR_LOG" 2>&1
}

on_exit() {
  local shell_rc=$?
  set +e
  if [[ -n "$MONITOR_PID" ]]; then
    stop_monitor
    MONITOR_RC=$?
  fi
  if [[ ! -f "$TELEMETRY_SUMMARY" && -s "$TELEMETRY" ]]; then
    summarize_telemetry
    SUMMARY_RC=$?
  fi
  if [[ "$FINALIZED" -ne 1 ]]; then
    write_stage_marker failed "$shell_rc" || true
  fi
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

write_stage_marker running 0

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RLINF_ALLOWED_PHYSICAL_GPUS=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTHONPATH="$CODE_ROOT:$RLINF_SOURCE${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" "$MONITOR" sample \
  --output "$TELEMETRY" \
  --stop-file "$STOP_FILE" \
  --interval "$TELEMETRY_INTERVAL_S" > "$MONITOR_LOG" 2>&1 &
MONITOR_PID=$!
sleep 1
if ! kill -0 "$MONITOR_PID" 2>/dev/null; then
  wait "$MONITOR_PID" || true
  MONITOR_PID=
  echo "GPU telemetry failed to start; see $MONITOR_LOG" >&2
  exit 5
fi

TRAIN_ARGS=(
  "$PHASE"
  --run-manifest "$RUN_MANIFEST"
  --expected-run-manifest-sha256 "$EXPECTED_RUN_MANIFEST_SHA256"
  --replay-manifest "$REPLAY_MANIFEST"
  --expected-replay-manifest-sha256 "$EXPECTED_REPLAY_MANIFEST_SHA256"
  --output-dir "$TRAIN_ROOT"
  --required-world-size 8
)
if [[ "$PHASE" == stage2 ]]; then
  TRAIN_ARGS+=(
    --stage1-checkpoint "$STAGE1_CHECKPOINT"
    --expected-stage1-checkpoint-sha256 "$EXPECTED_STAGE1_CHECKPOINT_SHA256"
  )
fi

set +e
"$TORCHRUN" --standalone --nproc-per-node=8 \
  "$TRAINER" "${TRAIN_ARGS[@]}" > "$TRAIN_LOG" 2>&1
TRAIN_RC=$?
stop_monitor
MONITOR_RC=$?
summarize_telemetry
SUMMARY_RC=$?
set -e

if [[ "$TRAIN_RC" -ne 0 || "$MONITOR_RC" -ne 0 || "$SUMMARY_RC" -ne 0 ]]; then
  echo "Navigation-RLT stage failed; see $TRAIN_LOG and $MONITOR_LOG" >&2
  write_stage_marker failed 6
  FINALIZED=1
  exit 6
fi

"$PYTHON" - \
  "$RUNS_ROOT" "$PHASE" "$TRAIN_ROOT" "$EXPECTED_RUN_MANIFEST_SHA256" \
  "$EXPECTED_REPLAY_MANIFEST_SHA256" "$TELEMETRY_SUMMARY" "$VERIFICATION" <<'PY'
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile

(runs_root_s, phase, train_root_s, run_sha, replay_sha,
 telemetry_summary_s, verification_s) = sys.argv[1:]
runs_root = Path(runs_root_s).resolve(strict=True)
train_root = Path(train_root_s).resolve(strict=True)

def fail(message):
    raise SystemExit(message)

def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def regular(value, name):
    raw = Path(value)
    if raw.is_symlink():
        fail(f'{name} must not be a symlink')
    try:
        path = raw.resolve(strict=True)
        path.relative_to(runs_root)
    except (OSError, ValueError):
        fail(f'{name} escapes the runs root or is missing')
    if not path.is_file():
        fail(f'{name} must be a regular file')
    return path

def load(path, name):
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        fail(f'{name} is not JSON: {error}')
    if not isinstance(value, dict):
        fail(f'{name} must contain an object')
    return value

completion_path = regular(train_root / 'COMPLETED.json', 'training completion')
result_path = regular(train_root / 'training_result.json', 'training result')
inventory_path = regular(train_root / 'artifact_inventory.json', 'artifact inventory')
checkpoint_path = regular(
    train_root / 'checkpoints/latest/navigation_rlt_policy.pt', 'final checkpoint'
)
telemetry_path = regular(telemetry_summary_s, 'telemetry summary')
completion = load(completion_path, 'training completion')
result = load(result_path, 'training result')
telemetry = load(telemetry_path, 'telemetry summary')
expected_completion = {
    'schema_name': 'navigation_rlt_training_completion',
    'schema_version': 1,
    'status': 'completed',
    'phase': phase,
    'training_result_sha256': sha256(result_path),
    'artifact_inventory_sha256': sha256(inventory_path),
}
if completion != expected_completion:
    fail('training completion marker differs')
if (result.get('status') != 'completed' or result.get('phase') != phase or
        result.get('world_size') != 8 or result.get('unique_gpu_uuid_count') != 8):
    fail('training result does not prove an eight-rank completion')
if result.get('run_manifest', {}).get('sha256') != run_sha:
    fail('training result run manifest SHA-256 differs')
if result.get('replay_manifest', {}).get('sha256') != replay_sha:
    fail('training result replay manifest SHA-256 differs')
checkpoint_receipt = result.get('final_checkpoint')
if not isinstance(checkpoint_receipt, dict):
    fail('training result lacks a final checkpoint receipt')
if Path(checkpoint_receipt.get('path', '')).resolve(strict=True) != checkpoint_path:
    fail('final checkpoint path differs')
checkpoint_sha = sha256(checkpoint_path)
if checkpoint_receipt.get('sha256') != checkpoint_sha:
    fail('final checkpoint SHA-256 differs')
restore = result.get('independent_restore')
if not isinstance(restore, dict) or restore.get('status') != 'passed':
    fail('independent checkpoint restore did not pass')
if restore.get('checkpoint_sha256') != checkpoint_sha:
    fail('independent restore checkpoint SHA-256 differs')
gpus = telemetry.get('gpus')
if telemetry.get('gpu_count') != 8 or not isinstance(gpus, list) or len(gpus) != 8:
    fail('telemetry summary does not contain eight GPUs')
for gpu in gpus:
    for name in ('memory_used_mean_mib', 'memory_used_max_mib',
                 'gpu_util_mean_percent', 'gpu_util_max_percent',
                 'power_mean_w', 'power_max_w'):
        value = float(gpu[name])
        if not math.isfinite(value):
            fail('telemetry summary contains non-finite values')

verification = {
    'schema_name': 'navigation_rlt_a800_verification',
    'schema_version': 1,
    'status': 'passed',
    'phase': phase,
    'world_size': 8,
    'unique_gpu_uuid_count': 8,
    'training_completion': {
        'path': str(completion_path), 'sha256': sha256(completion_path),
    },
    'training_result': {
        'path': str(result_path), 'sha256': sha256(result_path),
    },
    'checkpoint': {'path': str(checkpoint_path), 'sha256': checkpoint_sha},
    'telemetry_summary': {
        'path': str(telemetry_path), 'sha256': sha256(telemetry_path),
    },
}
output = Path(verification_s)
payload = json.dumps(verification, indent=2, sort_keys=True, allow_nan=False) + '\n'
with output.open('x', encoding='utf-8', newline='\n') as stream:
    stream.write(payload)
print(json.dumps(verification, sort_keys=True, allow_nan=False))
PY

VERIFICATION_SHA256=$(sha256sum "$VERIFICATION" | cut -d' ' -f1)
CHECKPOINT_PATH=$("$PYTHON" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"]["path"])' \
  "$VERIFICATION")
CHECKPOINT_SHA256=$("$PYTHON" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"]["sha256"])' \
  "$VERIFICATION")
write_stage_marker completed 0
FINALIZED=1
trap - EXIT INT TERM

"$PYTHON" - "$STAGE_MARKER" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
value = json.loads(path.read_text(encoding='utf-8'))
print(json.dumps({
    'status': value['status'],
    'phase': value['phase'],
    'stage_marker': str(path),
    'stage_marker_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    'checkpoint': value['output']['checkpoint'],
    'verification': value['verification'],
}, sort_keys=True, allow_nan=False))
PY
