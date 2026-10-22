#!/usr/bin/env bash
set -euo pipefail
umask 077
set -o noclobber

if [[ $# -ne 11 ]]; then
  cat >&2 <<'EOF'
usage: run_navigation_rlt_ebk_eval.sh RUN_ROOT BEHAVIOR HABITAT_PORT LIGHTNAV_PORT EXPECTED_RUN_MANIFEST_SHA256 EXPECTED_SPLIT_MANIFEST_SHA256 EXPECTED_EVAL_PLAN_SHA256 EXPECTED_EVALUATION_CODE_INVENTORY_SHA256 CHECKPOINT EXPECTED_CHECKPOINT_SHA256 EXPECTED_CONTEXT_SOURCE_SHA256

BEHAVIOR must be native, a3, or navigation_rlt.  Native requires CHECKPOINT
and EXPECTED_CHECKPOINT_SHA256 to both be '-'.
EOF
  exit 2
fi

RAW_RUN_ROOT=$1
BEHAVIOR=$2
HABITAT_PORT=$3
LIGHTNAV_PORT=$4
EXPECTED_RUN_SHA=$5
EXPECTED_SPLIT_SHA=$6
EXPECTED_PLAN_SHA=$7
EXPECTED_EVAL_INVENTORY_SHA=$8
CHECKPOINT_ARG=$9
EXPECTED_CHECKPOINT_SHA=${10}
EXPECTED_CONTEXT_SOURCE_SHA=${11}

PRIVATE_RUNS_ROOT=/srv/navigation/runs
LIGHTNAV_SOURCE=/srv/navigation/repos/LightNav-0-navigation-rlt
SCENES_ROOT=/opt/InternNav/data/scene_data/mp3d_ce
HABITAT_ENV=/srv/navigation/envs/lightnav-habitat-v031
CLIENT_ENV=/opt/conda/envs/lightnav0
MODEL_ROOT=/opt/LightNav-0/checkpoints/LightNav-0
MODEL_WEIGHT="$MODEL_ROOT/model-00001-of-00001.safetensors"
MODEL_TOKENIZER="$MODEL_ROOT/tokenizer.json"
MODEL_PROCESSOR="$MODEL_ROOT/processor_config.json"
MODEL_HISTORY_CONFIG="$MODEL_ROOT/eval_config.json"
HABITAT_CONFIG="$LIGHTNAV_SOURCE/habitat_server/configs/vlnce_r2r.yaml"

die() {
  echo "navigation-rlt ebk eval: $*" >&2
  exit 3
}

require_sha256() {
  local value=$1
  local label=$2
  [[ "$value" =~ ^[0-9a-f]{64}$ ]] || die "$label must be a lowercase SHA-256"
}

[[ -d "$RAW_RUN_ROOT" && ! -L "$RAW_RUN_ROOT" ]] || die "RUN_ROOT is absent or linked"
RUN_ROOT=$(realpath -e -- "$RAW_RUN_ROOT")
case "$RUN_ROOT" in
  "$PRIVATE_RUNS_ROOT"/*) ;;
  *) die "RUN_ROOT must be a child of $PRIVATE_RUNS_ROOT" ;;
esac

case "$BEHAVIOR" in
  native)
    EVAL_BEHAVIOR=native
    [[ "$CHECKPOINT_ARG" == - && "$EXPECTED_CHECKPOINT_SHA" == - ]] || \
      die "native must not receive a checkpoint"
    ;;
  a3)
    EVAL_BEHAVIOR=a3
    require_sha256 "$EXPECTED_CHECKPOINT_SHA" "A3 checkpoint hash"
    ;;
  navigation_rlt)
    EVAL_BEHAVIOR=navigation-rlt
    require_sha256 "$EXPECTED_CHECKPOINT_SHA" "Navigation-RLT checkpoint hash"
    ;;
  *) die "BEHAVIOR must be native, a3, or navigation_rlt" ;;
esac

for item in \
  "$EXPECTED_RUN_SHA:run manifest hash" \
  "$EXPECTED_SPLIT_SHA:split manifest hash" \
  "$EXPECTED_PLAN_SHA:evaluation plan hash" \
  "$EXPECTED_EVAL_INVENTORY_SHA:evaluation code inventory hash" \
  "$EXPECTED_CONTEXT_SOURCE_SHA:context feature source hash"; do
  require_sha256 "${item%%:*}" "${item#*:}"
done

[[ "$HABITAT_PORT" =~ ^[0-9]+$ ]] || die "Habitat port is not numeric"
[[ "$LIGHTNAV_PORT" =~ ^[0-9]+$ ]] || die "LightNav port is not numeric"
(( HABITAT_PORT >= 1024 && HABITAT_PORT <= 65535 )) || die "Habitat port is out of range"
(( LIGHTNAV_PORT >= 1024 && LIGHTNAV_PORT <= 65535 )) || die "LightNav port is out of range"
[[ "$HABITAT_PORT" != "$LIGHTNAV_PORT" ]] || die "Habitat and LightNav ports must differ"

EVAL_RUNTIME_ROOT="$RUN_ROOT/eval-runtime"
EVAL_CODE_ROOT="$EVAL_RUNTIME_ROOT/eval-code"
RUN_MANIFEST="$RUN_ROOT/run_manifest.json"
SPLIT_MANIFEST="$RUN_ROOT/data/build/split_manifest.json"
DATA_ROOT="$RUN_ROOT/data/build/train_development"
DATASET="$DATA_ROOT/train/train.json.gz"
GROUND_TRUTH="$DATA_ROOT/train/train_gt.json.gz"
EVAL_PLAN="$EVAL_RUNTIME_ROOT/eval-plan.json"
EVAL_INVENTORY="$EVAL_CODE_ROOT/evaluation_code_inventory.json"
EVAL_SCRIPT="$EVAL_CODE_ROOT/navigation_rlt_eval.py"
WRAPPER="$EVAL_CODE_ROOT/run_navigation_rlt_ebk_eval.sh"
OUTPUT_DIR="$RUN_ROOT/evaluations/$BEHAVIOR"
FEATURE_PARENT="$RUN_ROOT/eval-feature-artifacts"
FEATURE_ROOT="$FEATURE_PARENT/$BEHAVIOR"
STATUS_ROOT="$RUN_ROOT/status"
LOG_ROOT="$RUN_ROOT/logs"
HABITAT_READY="$STATUS_ROOT/eval_${BEHAVIOR}_habitat.ready"
LIGHTNAV_READY="$STATUS_ROOT/eval_${BEHAVIOR}_lightnav.ready"
HABITAT_PID_FILE="$STATUS_ROOT/eval_${BEHAVIOR}_habitat.pid"
LIGHTNAV_PID_FILE="$STATUS_ROOT/eval_${BEHAVIOR}_lightnav.pid"
HABITAT_LOG="$LOG_ROOT/eval_${BEHAVIOR}_habitat.log"
LIGHTNAV_LOG="$LOG_ROOT/eval_${BEHAVIOR}_lightnav.log"
EVAL_LOG="$LOG_ROOT/eval_${BEHAVIOR}.log"

[[ -d "$EVAL_CODE_ROOT" && ! -L "$EVAL_CODE_ROOT" ]] || die "eval-code bundle is absent or linked"
EVAL_CODE_ROOT=$(realpath -e -- "$EVAL_CODE_ROOT")
[[ "$EVAL_CODE_ROOT" == "$RUN_ROOT/eval-runtime/eval-code" ]] || die "eval-code bundle is not canonical"

for path in \
  "$RUN_MANIFEST" "$SPLIT_MANIFEST" "$DATASET" "$GROUND_TRUTH" \
  "$EVAL_PLAN" "$EVAL_INVENTORY" "$EVAL_SCRIPT" "$WRAPPER"; do
  [[ -f "$path" && ! -L "$path" ]] || die "required regular file is absent or linked: $path"
done
for path in "$RUN_ROOT/data" "$DATA_ROOT" "$DATA_ROOT/train"; do
  [[ -d "$path" && ! -L "$path" ]] || die "required data directory is absent or linked: $path"
done
for path in "$MODEL_WEIGHT" "$MODEL_TOKENIZER" "$MODEL_PROCESSOR" "$MODEL_HISTORY_CONFIG"; do
  [[ -f "$path" && ! -L "$path" ]] || die "LightNav model/config artifact is unavailable or linked: $path"
done
[[ -f "$HABITAT_CONFIG" ]] || die "Habitat configuration is unavailable"
[[ -x "$HABITAT_ENV/bin/python" ]] || die "Habitat Python is unavailable"
[[ -x "$CLIENT_ENV/bin/python" ]] || die "LightNav client Python is unavailable"
[[ -d "$SCENES_ROOT/mp3d" && ! -L "$SCENES_ROOT" ]] || die "MP3D scene root is unavailable"
[[ -d "$LIGHTNAV_SOURCE" && ! -L "$LIGHTNAV_SOURCE" ]] || die "LightNav source is unavailable or linked"
[[ "$(realpath -e -- "$0")" == "$WRAPPER" ]] || die "wrapper is not running from the eval-code bundle"

[[ "$(sha256sum "$RUN_MANIFEST" | cut -d' ' -f1)" == "$EXPECTED_RUN_SHA" ]] || \
  die "run manifest SHA-256 differs"
[[ "$(sha256sum "$SPLIT_MANIFEST" | cut -d' ' -f1)" == "$EXPECTED_SPLIT_SHA" ]] || \
  die "split manifest SHA-256 differs"
[[ "$(sha256sum "$EVAL_PLAN" | cut -d' ' -f1)" == "$EXPECTED_PLAN_SHA" ]] || \
  die "evaluation plan SHA-256 differs"
[[ "$(sha256sum "$EVAL_INVENTORY" | cut -d' ' -f1)" == "$EXPECTED_EVAL_INVENTORY_SHA" ]] || \
  die "evaluation code inventory SHA-256 differs"

CHECKPOINT=-
if [[ "$BEHAVIOR" != native ]]; then
  [[ -d "$RUN_ROOT/inputs" && ! -L "$RUN_ROOT/inputs" ]] || die "RUN_ROOT/inputs is absent or linked"
  [[ -f "$CHECKPOINT_ARG" && ! -L "$CHECKPOINT_ARG" ]] || die "checkpoint is absent or linked"
  CHECKPOINT=$(realpath -e -- "$CHECKPOINT_ARG")
  case "$CHECKPOINT" in
    "$RUN_ROOT"/inputs/*) ;;
    *) die "checkpoint must be a regular file under RUN_ROOT/inputs" ;;
  esac
  [[ "$(sha256sum "$CHECKPOINT" | cut -d' ' -f1)" == "$EXPECTED_CHECKPOINT_SHA" ]] || \
    die "checkpoint SHA-256 differs"
fi

CONTRACT_OUTPUT=$(
  "$CLIENT_ENV/bin/python" - \
    "$RUN_MANIFEST" "$EXPECTED_RUN_SHA" \
    "$SPLIT_MANIFEST" "$EXPECTED_SPLIT_SHA" "$DATASET" "$GROUND_TRUTH" \
    "$EVAL_PLAN" "$EXPECTED_PLAN_SHA" \
    "$EVAL_INVENTORY" "$EXPECTED_EVAL_INVENTORY_SHA" "$EVAL_CODE_ROOT" \
    "$BEHAVIOR" "$EXPECTED_CONTEXT_SOURCE_SHA" <<'PY'
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

(
    run_path, expected_run_sha, split_path, expected_split_sha, dataset_path,
    gt_path, plan_path, expected_plan_sha, inventory_path,
    expected_inventory_sha, code_root, behavior, expected_context_source_sha,
) = sys.argv[1:]

def fail(message):
    raise SystemExit(message)

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def digest(value, label):
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        fail(f"{label} is not a lowercase SHA-256")
    try:
        int(value, 16)
    except ValueError:
        fail(f"{label} is not a lowercase SHA-256")
    return value

def commit(value, label):
    if not isinstance(value, str) or len(value) != 40 or value != value.lower():
        fail(f"{label} is not a lowercase full Git SHA-1")
    try:
        int(value, 16)
    except ValueError:
        fail(f"{label} is not a lowercase full Git SHA-1")
    return value

def load_json(path, label):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        fail(f"{label} is absent or linked")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail(f"{label} is invalid JSON: {exc}")
    if not isinstance(value, dict):
        fail(f"{label} must contain an object")
    return value

for path, expected, label in (
    (run_path, expected_run_sha, "run manifest"),
    (split_path, expected_split_sha, "split manifest"),
    (plan_path, expected_plan_sha, "evaluation plan"),
    (inventory_path, expected_inventory_sha, "evaluation code inventory"),
):
    if sha256(path) != digest(expected, f"expected {label} hash"):
        fail(f"{label} SHA-256 differs")

inventory = load_json(inventory_path, "evaluation code inventory")
if set(inventory) != {"schema_name", "schema_version", "navigation_code_commit", "files"}:
    fail("evaluation code inventory members differ")
if inventory.get("schema_name") != "navigation_rlt_evaluation_code_inventory" or inventory.get("schema_version") != 1:
    fail("evaluation code inventory schema differs")
inventory_commit = commit(inventory.get("navigation_code_commit"), "evaluation code commit")
files = inventory.get("files")
required = {
    "build_navigation_rlt_train_split.py",
    "lightnav_a3_policy.py",
    "lightnav_a3_rollout.py",
    "lightnav_l0_native_probe.py",
    "navigation_rlt_action.py",
    "navigation_rlt_compare.py",
    "navigation_rlt_eval.py",
    "navigation_rlt_obs_archive.py",
    "navigation_rlt_parallel.py",
    "navigation_rlt_policy.py",
    "navigation_rlt_post_training.py",
    "navigation_rlt_replay.py",
    "navigation_rlt_rollout.py",
    "streamvln_executor_contract.py",
    "streamvln_topology.py",
    "navrl/__init__.py",
    "navrl/contracts.py",
    "navrl/envs/__init__.py",
    "navrl/envs/base.py",
    "navrl/envs/grid_mock.py",
    "navrl/episode.py",
    "navrl/metrics.py",
    "navrl/policies/__init__.py",
    "navrl/policies/base.py",
    "navrl/rewards.py",
    "navrl/stack/__init__.py",
    "navrl/stack/checkpoint.py",
    "navrl/stack/collector.py",
    "navrl/stack/contracts.py",
    "navrl/stack/execution.py",
    "navrl/stack/interfaces.py",
    "navrl/stack/legacy.py",
    "navrl/stack/ppo.py",
    "navrl/stack/returns.py",
    "run_navigation_rlt_ebk_eval.sh",
}
if not isinstance(files, dict) or set(files) != required:
    fail("evaluation code inventory does not exactly match the runtime dependency set")
root = Path(code_root).resolve(strict=True)
for name, expected in files.items():
    if not isinstance(name, str):
        fail("evaluation code inventory filename is invalid")
    relative = PurePosixPath(name)
    if (
        not relative.parts
        or relative.is_absolute()
        or relative.as_posix() != name
        or any(part in ("", ".", "..") for part in relative.parts)
    ):
        fail(f"evaluation code inventory filename is not canonical: {name!r}")
    expected = digest(expected, f"evaluation code inventory files[{name}]")
    candidate = root.joinpath(*relative.parts)
    cursor = candidate
    while cursor != root:
        if cursor.is_symlink():
            fail(f"evaluation code path contains a link: {name}")
        cursor = cursor.parent
    try:
        actual = candidate.resolve(strict=True)
        actual.relative_to(root)
    except (OSError, ValueError):
        fail(f"evaluation code file escapes or is absent: {name}")
    if not actual.is_file() or sha256(actual) != expected:
        fail(f"evaluation code file SHA-256 differs: {name}")

manifest = load_json(run_path, "run manifest")
if manifest.get("schema_name") != "navigation_rlt_experiment_manifest" or manifest.get("schema_version") != 1:
    fail("run manifest schema differs")
run_id = manifest.get("run_id")
if not isinstance(run_id, str) or not run_id or Path(run_id).name != run_id:
    fail("run manifest run_id is invalid")
boundaries = manifest.get("boundaries")
if not isinstance(boundaries, dict) or any((
    boundaries.get("upstream_split_read") != "train",
    boundaries.get("validation_split_reads") != 0,
    boundaries.get("lightnav_frozen") is not True,
    boundaries.get("weight_origin") != "original_release",
    boundaries.get("ttt_or_secondary_post_training") is not False,
)):
    fail("run manifest train-only/frozen boundary differs")
data = manifest.get("data")
if not isinstance(data, dict) or any((
    data.get("split_manifest_sha256") != expected_split_sha,
    data.get("development_episode_count") != 32,
)):
    fail("run manifest development data identity differs")
protocol = manifest.get("protocol")
if not isinstance(protocol, dict) or any((
    protocol.get("max_physical_steps") != 160,
    protocol.get("matched_eval_arms") != ["native_lightnav", "a3_ppo", "navigation_rlt"],
)):
    fail("run manifest matched evaluation protocol differs")
lightnav = manifest.get("lightnav")
if not isinstance(lightnav, dict) or any((
    lightnav.get("backend") != "hf",
    lightnav.get("context_request") != "contextual_prefix_hidden_v1",
    lightnav.get("artifact_prefix_max_tokens") != 4096,
    lightnav.get("policy_prefix_max_tokens") != 192,
    lightnav.get("prefix_hidden_dim") != 2560,
)):
    fail("run manifest LightNav service/context contract differs")
original_revision = commit(lightnav.get("original_release_revision"), "original LightNav revision")
patched_head = commit(lightnav.get("patched_head"), "patched LightNav revision")
context_patch_sha = digest(lightnav.get("context_patch_sha256"), "LightNav context patch hash")
weight_sha = digest(lightnav.get("weight_sha256"), "LightNav weight hash")
tokenizer_sha = digest(lightnav.get("tokenizer_sha256"), "LightNav tokenizer hash")
processor_sha = digest(lightnav.get("processor_config_sha256"), "LightNav processor hash")
history_sha = digest(lightnav.get("history_config_sha256"), "LightNav history hash")
a3_sha = digest(manifest.get("behavior", {}).get("checkpoint_sha256"), "A3 checkpoint hash")
context_source_sha = digest(
    expected_context_source_sha,
    "expected LightNav prefix feature source hash",
)

split = load_json(split_path, "split manifest")
if split.get("schema_name") != "navigation_rlt_train_scene_split" or split.get("schema_version") != 1:
    fail("split manifest schema differs")
if split.get("upstream_split_read") != "train" or split.get("validation_split_reads") != 0:
    fail("split manifest is not train-only")
development = split.get("splits", {}).get("train_development")
if not isinstance(development, dict) or any((
    development.get("habitat_split") != "train",
    development.get("episode_count") != 32,
    development.get("scene_count") != 8,
)):
    fail("split manifest development contract differs")
if sha256(dataset_path) != digest(development.get("dataset_sha256"), "development dataset hash"):
    fail("development dataset SHA-256 differs")
if sha256(gt_path) != digest(development.get("gt_sha256"), "development ground-truth hash"):
    fail("development ground-truth SHA-256 differs")
try:
    dataset = json.loads(gzip.decompress(Path(dataset_path).read_bytes()).decode("utf-8"))
    ground_truth = json.loads(gzip.decompress(Path(gt_path).read_bytes()).decode("utf-8"))
except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
    fail(f"development dataset or ground truth is invalid: {exc}")
episodes = dataset.get("episodes") if isinstance(dataset, dict) else None
if not isinstance(episodes, list) or len(episodes) != 32:
    fail("development dataset does not contain exactly 32 episodes")
episode_ids = [str(row.get("episode_id")) for row in episodes if isinstance(row, dict)]
if len(episode_ids) != 32 or len(set(episode_ids)) != 32:
    fail("development dataset episode IDs are incomplete or repeated")
if not isinstance(ground_truth, dict) or set(ground_truth) != set(episode_ids):
    fail("development ground-truth episode inventory differs")

plan = load_json(plan_path, "evaluation plan")
if any((
    plan.get("schema_name") != "navigation_rlt_parallel_rollout_plan",
    plan.get("schema_version") != 1,
    plan.get("run_id") != run_id,
    plan.get("split_label") != "train_development",
    plan.get("validation_split_reads") != 0,
    plan.get("worker_count") != 1,
    plan.get("episode_count") != 32,
)):
    fail("evaluation plan is not the single-worker fixed development contract")
split_ref = plan.get("split_manifest")
if not isinstance(split_ref, dict) or split_ref.get("sha256") != expected_split_sha:
    fail("evaluation plan split-manifest identity differs")
workers = plan.get("workers")
if not isinstance(workers, list) or len(workers) != 1:
    fail("evaluation plan worker inventory differs")
planned = workers[0].get("episodes") if isinstance(workers[0], dict) else None
if not isinstance(planned, list) or workers[0].get("worker_id") != 0 or len(planned) != 32:
    fail("evaluation plan worker differs")
expected_rows = development.get("episodes")
if not isinstance(expected_rows, list) or len(expected_rows) != 32:
    fail("split manifest development identity rows differ")
for index, (row, expected) in enumerate(zip(planned, expected_rows)):
    if not isinstance(row, dict) or not isinstance(expected, dict) or any((
        row.get("episode_order") != index,
        str(row.get("episode_id")) != str(expected.get("episode_id")),
        row.get("scene") != expected.get("scene"),
        row.get("instruction_text_sha256") != expected.get("instruction_text_sha256"),
    )):
        fail(f"evaluation plan identity differs at episode order {index}")

for value in (
    original_revision, patched_head, context_patch_sha, weight_sha, tokenizer_sha,
    processor_sha, history_sha, a3_sha, context_source_sha, inventory_commit,
):
    print(value)
PY
)
readarray -t CONTRACT <<<"$CONTRACT_OUTPUT"
[[ ${#CONTRACT[@]} -eq 10 ]] || die "preflight contract output is incomplete"

[[ "$(git -C "$LIGHTNAV_SOURCE" rev-parse HEAD)" == "${CONTRACT[1]}" ]] || \
  die "LightNav patched HEAD differs"
[[ -z "$(git -C "$LIGHTNAV_SOURCE" status --porcelain)" ]] || die "LightNav worktree is dirty"
git -C "$LIGHTNAV_SOURCE" merge-base --is-ancestor "${CONTRACT[0]}" "${CONTRACT[1]}" || \
  die "frozen LightNav release is not an ancestor of the patched service"
git -C "$LIGHTNAV_SOURCE" diff --check || die "LightNav source fails diff --check"
[[ "$(sha256sum "$MODEL_WEIGHT" | cut -d' ' -f1)" == "${CONTRACT[3]}" ]] || \
  die "LightNav model weight SHA-256 differs"
[[ "$(sha256sum "$MODEL_TOKENIZER" | cut -d' ' -f1)" == "${CONTRACT[4]}" ]] || \
  die "LightNav tokenizer SHA-256 differs"
[[ "$(sha256sum "$MODEL_PROCESSOR" | cut -d' ' -f1)" == "${CONTRACT[5]}" ]] || \
  die "LightNav processor config SHA-256 differs"
[[ "$(sha256sum "$MODEL_HISTORY_CONFIG" | cut -d' ' -f1)" == "${CONTRACT[6]}" ]] || \
  die "LightNav history config SHA-256 differs"
if [[ "$BEHAVIOR" == a3 && "$EXPECTED_CHECKPOINT_SHA" != "${CONTRACT[7]}" ]]; then
  die "A3 checkpoint differs from the outer manifest"
fi

"$CLIENT_ENV/bin/python" - "$HABITAT_PORT" "$LIGHTNAV_PORT" <<'PY'
import socket
import sys

sockets = []
try:
    for raw in sys.argv[1:]:
        value = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        value.bind(("127.0.0.1", int(raw)))
        sockets.append(value)
finally:
    for value in sockets:
        value.close()
PY

[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || \
  die "evaluation output already exists or is linked: $OUTPUT_DIR"
for path in "$RUN_ROOT/evaluations" "$STATUS_ROOT" "$LOG_ROOT" "$FEATURE_PARENT"; do
  if [[ -e "$path" || -L "$path" ]]; then
    [[ -d "$path" && ! -L "$path" ]] || die "evaluation parent path is not a regular directory: $path"
  fi
done
for path in \
  "$HABITAT_READY" "$LIGHTNAV_READY" "$HABITAT_PID_FILE" "$LIGHTNAV_PID_FILE" \
  "$HABITAT_LOG" "$LIGHTNAV_LOG" "$EVAL_LOG"; do
  [[ ! -e "$path" && ! -L "$path" ]] || die "stale evaluation path exists or is linked: $path"
done
mkdir -p -- "$RUN_ROOT/evaluations" "$STATUS_ROOT" "$LOG_ROOT" "$FEATURE_PARENT"
for path in "$RUN_ROOT/evaluations" "$STATUS_ROOT" "$LOG_ROOT" "$FEATURE_PARENT"; do
  [[ -d "$path" && ! -L "$path" ]] || die "created evaluation parent is not a regular directory: $path"
done
[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || \
  die "evaluation output appeared during parent creation: $OUTPUT_DIR"
for path in \
  "$HABITAT_READY" "$LIGHTNAV_READY" "$HABITAT_PID_FILE" "$LIGHTNAV_PID_FILE" \
  "$HABITAT_LOG" "$LIGHTNAV_LOG" "$EVAL_LOG"; do
  [[ ! -e "$path" && ! -L "$path" ]] || die "evaluation path appeared during parent creation: $path"
done
[[ ! -e "$FEATURE_ROOT" && ! -L "$FEATURE_ROOT" ]] || die "dedicated eval feature root already exists"
mkdir -- "$FEATURE_ROOT"

HABITAT_PID=
LIGHTNAV_PID=
cleanup() {
  local status=$?
  trap - EXIT INT TERM HUP
  set +e
  for pid in "$HABITAT_PID" "$LIGHTNAV_PID"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null
    fi
  done
  for _ in $(seq 1 30); do
    local alive=0
    for pid in "$HABITAT_PID" "$LIGHTNAV_PID"; do
      if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        alive=1
      fi
    done
    (( alive == 0 )) && break
    sleep 1
  done
  for pid in "$HABITAT_PID" "$LIGHTNAV_PID"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill -KILL "$pid" 2>/dev/null
    fi
    if [[ -n "$pid" ]]; then
      wait "$pid" 2>/dev/null
    fi
  done
  rm -f -- "$HABITAT_READY" "$LIGHTNAV_READY" "$HABITAT_PID_FILE" "$LIGHTNAV_PID_FILE"
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT TERM HUP

(
  export CUDA_VISIBLE_DEVICES=0
  export PYTHONPATH="$LIGHTNAV_SOURCE/src"
  export LIGHTNAV_ATTN=sdpa
  exec "$CLIENT_ENV/bin/python" -m lightnav.serving.ws_server \
    --task vln \
    --model_path "$MODEL_ROOT" \
    --backend hf \
    --device cuda \
    --num_history_frames 64 \
    --host 127.0.0.1 \
    --port "$LIGHTNAV_PORT" \
    --max_batch_size 4 \
    --max_wait_ms 8 \
    --feature_request_capability contextual_prefix_hidden_v1 \
    --feature_artifact_dir "$FEATURE_ROOT" \
    --feature_max_tokens 4096 \
    --feature_max_bytes_mib 128 \
    --ready_file "$LIGHTNAV_READY"
) >"$LIGHTNAV_LOG" 2>&1 &
LIGHTNAV_PID=$!
printf '%s\n' "$LIGHTNAV_PID" >"$LIGHTNAV_PID_FILE"

(
  export CUDA_VISIBLE_DEVICES=0
  export HABITAT_SIM_GPU_ID=0
  export NVIDIA_DRIVER_CAPABILITIES=all
  export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
  export LD_PRELOAD=/lib/x86_64-linux-gnu/libGLdispatch.so.0
  export MAGNUM_LOG=quiet
  export PYTHONPATH="$LIGHTNAV_SOURCE/habitat_server"
  exec "$HABITAT_ENV/bin/python" -m lightnav_habitat.serve \
    --task vlnce \
    --config "$HABITAT_CONFIG" \
    --port "$HABITAT_PORT" \
    --max-steps 160 \
    --image-height 270 \
    --image-width 480 \
    --split train \
    --data-path "$DATA_ROOT" \
    --scenes-dir "$SCENES_ROOT" \
    --ready-file "$HABITAT_READY"
) >"$HABITAT_LOG" 2>&1 &
HABITAT_PID=$!
printf '%s\n' "$HABITAT_PID" >"$HABITAT_PID_FILE"

wait_for_ready() {
  local ready=$1
  local pid=$2
  local label=$3
  local count
  for count in $(seq 1 600); do
    if [[ -f "$ready" && ! -L "$ready" ]]; then
      return 0
    fi
    kill -0 "$pid" 2>/dev/null || die "$label exited before readiness"
    sleep 1
  done
  die "$label did not become ready within 600 seconds"
}

wait_for_ready "$LIGHTNAV_READY" "$LIGHTNAV_PID" "LightNav server"
wait_for_ready "$HABITAT_READY" "$HABITAT_PID" "Habitat server"

EVAL_ARGS=(
  --eval-plan "$EVAL_PLAN"
  --expected-eval-plan-sha256 "$EXPECTED_PLAN_SHA"
  --expected-split-manifest-sha256 "$EXPECTED_SPLIT_SHA"
  --run-manifest "$RUN_MANIFEST"
  --expected-run-manifest-sha256 "$EXPECTED_RUN_SHA"
  --evaluation-code-inventory "$EVAL_INVENTORY"
  --expected-evaluation-code-inventory-sha256 "$EXPECTED_EVAL_INVENTORY_SHA"
  --habitat-server "tcp://127.0.0.1:$HABITAT_PORT"
  --lightnav-server "ws://127.0.0.1:$LIGHTNAV_PORT"
  --output-dir "$OUTPUT_DIR"
  --behavior "$EVAL_BEHAVIOR"
  --max-steps 160
  --seed 2026100202
  --client-id "navigation-rlt-matched-eval-$BEHAVIOR"
  --topology-max-nodes 128
  --lightnav-source-revision "${CONTRACT[0]}"
  --lightnav-hook-sha256 "${CONTRACT[2]}"
  --lightnav-weight-sha256 "${CONTRACT[3]}"
  --tokenizer-sha256 "${CONTRACT[4]}"
  --processor-sha256 "${CONTRACT[5]}"
  --history-config-sha256 "${CONTRACT[6]}"
)
case "$BEHAVIOR" in
  native) ;;
  a3)
    EVAL_ARGS+=(
      --a3-checkpoint "$CHECKPOINT"
      --expected-a3-checkpoint-sha256 "$EXPECTED_CHECKPOINT_SHA"
    )
    ;;
  navigation_rlt)
    EVAL_ARGS+=(
      --rlt-checkpoint "$CHECKPOINT"
      --expected-rlt-checkpoint-sha256 "$EXPECTED_CHECKPOINT_SHA"
      --policy-device cpu
      --artifact-prefix-max-tokens 4096
      --policy-prefix-max-tokens 192
      --prefix-hidden-dim 2560
      --expected-context-source-sha256 "${CONTRACT[8]}"
      --feature-artifact-root "$FEATURE_ROOT"
    )
    ;;
esac

PYTHONPATH="$EVAL_CODE_ROOT" "$CLIENT_ENV/bin/python" "$EVAL_SCRIPT" "${EVAL_ARGS[@]}" \
  >"$EVAL_LOG" 2>&1

EVALUATION="$OUTPUT_DIR/evaluation.json"
COMPLETED="$OUTPUT_DIR/COMPLETED.json"
[[ -f "$EVALUATION" && ! -L "$EVALUATION" ]] || die "evaluation.json is absent or linked"
[[ -f "$COMPLETED" && ! -L "$COMPLETED" ]] || die "COMPLETED.json is absent or linked"

"$CLIENT_ENV/bin/python" - \
  "$EVALUATION" "$COMPLETED" "$EVAL_BEHAVIOR" "$EXPECTED_RUN_SHA" \
  "$EXPECTED_PLAN_SHA" "$EXPECTED_EVAL_INVENTORY_SHA" "$EXPECTED_CHECKPOINT_SHA" \
  "${CONTRACT[8]}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

evaluation_path, completed_path, behavior, run_sha, plan_sha, inventory_sha, checkpoint_sha, context_sha = sys.argv[1:]

def load(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"result is not an object: {path}")
    return value

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

evaluation = load(evaluation_path)
completed = load(completed_path)
if any((
    evaluation.get("schema_name") != "navigation_rlt_matched_train_development_eval",
    evaluation.get("schema_version") != 1,
    evaluation.get("status") != "completed",
    evaluation.get("behavior") != behavior,
    evaluation.get("split_label") != "train_development",
    evaluation.get("validation_split_reads") != 0,
    evaluation.get("training_replay_written") is not False,
    len(evaluation.get("episodes", ())) != 32,
    evaluation.get("executor", {}).get("max_physical_steps") != 160,
    evaluation.get("run_manifest_sha256") != run_sha,
    evaluation.get("evaluation_plan", {}).get("sha256") != plan_sha,
    evaluation.get("evaluation_code", {}).get("inventory_sha256") != inventory_sha,
)):
    raise SystemExit("evaluation receipt differs from the fixed matched contract")
source = evaluation.get("behavior_source", {})
if behavior == "native":
    if "checkpoint_sha256" in source:
        raise SystemExit("native evaluation unexpectedly names a policy checkpoint")
elif source.get("checkpoint_sha256") != checkpoint_sha:
    raise SystemExit("evaluation checkpoint SHA-256 differs")
if behavior == "navigation-rlt" and evaluation.get("context", {}).get("source_sha256") != context_sha:
    raise SystemExit("evaluation context source SHA-256 differs")
evaluation_sha = sha256(evaluation_path)
if any((
    completed.get("schema_name") != "navigation_rlt_matched_train_development_eval",
    completed.get("schema_version") != 1,
    completed.get("status") != "completed",
    completed.get("behavior") != behavior,
    completed.get("run_manifest_sha256") != run_sha,
    completed.get("evaluation_code", {}).get("inventory_sha256") != inventory_sha,
    completed.get("evaluation_sha256") != evaluation_sha,
    completed.get("training_replay_written") is not False,
)):
    raise SystemExit("COMPLETED receipt differs from evaluation.json")
print(json.dumps({
    "behavior": behavior,
    "evaluation": {"path": str(Path(evaluation_path).resolve()), "sha256": evaluation_sha},
    "completion": {"path": str(Path(completed_path).resolve()), "sha256": sha256(completed_path)},
}, sort_keys=True))
PY
