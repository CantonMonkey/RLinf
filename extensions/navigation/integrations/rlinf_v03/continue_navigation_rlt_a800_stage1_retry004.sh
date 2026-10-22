#!/usr/bin/env bash
set -euo pipefail
umask 077

if [[ $# -ne 0 ]]; then
  echo "This run-bound continuation accepts no arguments." >&2
  exit 2
fi

RUN=/srv/navigation/runs/navigation-rlt-20261002-v2
CODE="$RUN/code"
PROVENANCE="$RUN/provenance"
PYTHON=/srv/navigation/envs/rlinf-v03-cu118/bin/python
RUN_ID=navigation-rlt-20261002-v2
SOURCE_FINGERPRINT=1741b8f23682cdc3b1dd2e781fb7c05b5f667b816f5001e4f4880c45abdc1c01
SOURCE_ROLLOUT_ROOT=/srv/navigation/runs/navigation-rlt-20261002-v2/rollout

TRANSFER="$RUN/transfer-prefix"
ORIGINAL_MANIFEST="$TRANSFER/original_prefix_manifest.json"
ORIGINAL_MANIFEST_SHA=1e0f0a109bbc89029445ce7d42d495004b696ddab62825181bca3cfced377ab2
TRANSFER_INVENTORY_SHA=9bb4843ae6986332ba9603dd780f887015506c7b67b654ee8d4e0a8e474162cf
TRANSFER_MARKER_SHA=9d7a68d837b5202b66811a730664b8e0c5c77bbe8a6c3d01891cec51673c79b8
FINALIZER="$PROVENANCE/finalize_transfer_prefix_retry004.py"

RELOCATED="$RUN/navigation_rlt_prefix_relocated"
SPLIT="$RUN/navigation_rlt_stage1_split"
PREPARED="$RUN/navigation_rlt_stage1_prepare"
STAGE="$RUN/navigation_rlt_stage1"
GATE="$RUN/navigation_rlt_stage1_gate"

die() {
  echo "$*" >&2
  exit 1
}

sha256_file() {
  sha256sum -- "$1" | cut -d' ' -f1
}

require_sha() {
  local path=$1
  local expected=$2
  [[ -f "$path" && ! -L "$path" ]] || die "Missing or unsafe regular file: $path"
  local actual
  actual=$(sha256_file "$path")
  [[ "$actual" == "$expected" ]] || die "SHA-256 mismatch: $path"
}

json_values() {
  local path=$1
  local schema=$2
  local status=$3
  local operation=$4
  shift 4
  "$PYTHON" - "$path" "$schema" "$status" "$operation" "$@" <<'PY'
import json
import math
import sys
from pathlib import Path

path, schema, status, operation, *fields = sys.argv[1:]
value = json.loads(
    Path(path).read_text(encoding="utf-8"),
    parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)),
)
if not isinstance(value, dict):
    raise SystemExit(f"JSON root is not an object: {path}")
if value.get("schema_name") != schema or value.get("status") != status:
    raise SystemExit(f"completion marker schema/status differs: {path}")
if operation != "-" and value.get("operation") != operation:
    raise SystemExit(f"completion marker operation differs: {path}")
for field in fields:
    current = value
    for component in field.split("."):
        if not isinstance(current, dict) or component not in current:
            raise SystemExit(f"missing JSON field {field}: {path}")
        current = current[component]
    if isinstance(current, (dict, list)) or current is None:
        raise SystemExit(f"JSON field {field} is not scalar: {path}")
    if isinstance(current, float) and not math.isfinite(current):
        raise SystemExit(f"JSON field {field} is non-finite: {path}")
    if isinstance(current, bool):
        print("true" if current else "false")
    else:
        print(current)
PY
}

for binding in \
  "$PROVENANCE/training_protocol.json:7ef54354df67f33ecfbb7ea897405ca349d3f236fa0f1566754ad2b1333d1ff7" \
  "$FINALIZER:bb2e0734b0970989636ae53280b38ef49cb7829b7fc4b40b61810ff3dd88a7b3" \
  "$CODE/navigation_rlt_relocate_replay.py:88fde9513ccc926d931bf90f5abbe05c7824f192472258983cb6aef3be29bc5b" \
  "$CODE/navigation_rlt_stage1_gate.py:4e198f38acf778a262fe511a7f5ad8a4f38dc9781705aa99b2c17507053d2725" \
  "$CODE/navigation_rlt_prepare_training.py:889eb745d7bd40afef65dc3a425eefaf783d4db68ba20781d85c31d97daee8db" \
  "$CODE/navigation_rlt_fsdp_train.py:22609a6071d73e3752a43cae6515e42f8de6b8e43990b4c48fdd4d3edf23f00c" \
  "$CODE/navigation_rlt_policy.py:8611bfff992344fd7d9c9a7702f62185ed32a152ad62ee729f94a1ec69dd970b" \
  "$CODE/navigation_rlt_replay.py:f8fdfd33568bbdb83047c6372181c6c71822381ee049b30d32aea267e5743c94" \
  "$CODE/lightnav_a3_gpu_monitor.py:9e890e8550a3c881a2ad5b87c112f8123b5046231f39b3fd3804653ec553903f" \
  "$CODE/run_navigation_rlt_a800.sh:ad5996d605a71a9f7a2fc2f6af099fe0ee91ce11393eb945f097110b502ae2ba"
do
  require_sha "${binding%:*}" "${binding##*:}"
done

# The retry004 finalizer verifies all 260 transferred files and resumes only
# by reusing exact staging hardlinks.  It returns already_published on reruns.
"$PYTHON" "$FINALIZER" --publish
require_sha "$TRANSFER/transfer_inventory.json" "$TRANSFER_INVENTORY_SHA"
require_sha "$TRANSFER/COMPLETED.json" "$TRANSFER_MARKER_SHA"
require_sha "$ORIGINAL_MANIFEST" "$ORIGINAL_MANIFEST_SHA"

if [[ ! -f "$RELOCATED/COMPLETED.json" ]]; then
  [[ ! -e "$RELOCATED" && ! -L "$RELOCATED" ]] || \
    die "Relocation output exists without its completion marker: $RELOCATED"
  "$PYTHON" "$CODE/navigation_rlt_relocate_replay.py" \
    --original-manifest "$ORIGINAL_MANIFEST" \
    --expected-original-manifest-sha256 "$ORIGINAL_MANIFEST_SHA" \
    --source-root "$SOURCE_ROLLOUT_ROOT" \
    --destination-replay-root "$TRANSFER/replay_workers" \
    --expected-run-id "$RUN_ID" \
    --expected-source-fingerprint "$SOURCE_FINGERPRINT" \
    --expected-episode-count 256 \
    --output-dir "$RELOCATED"
fi
mapfile -t relocation < <(json_values \
  "$RELOCATED/COMPLETED.json" navigation_rlt_replay_relocation completed - \
  run_id original_manifest_sha256 relocated_manifest_sha256)
[[ "${relocation[0]}" == "$RUN_ID" ]] || die "Relocation run_id differs"
[[ "${relocation[1]}" == "$ORIGINAL_MANIFEST_SHA" ]] || die "Relocation parent SHA differs"
RELOCATED_MANIFEST="$RELOCATED/replay_manifest.json"
RELOCATED_SHA=${relocation[2]}
require_sha "$RELOCATED_MANIFEST" "$RELOCATED_SHA"

if [[ ! -f "$SPLIT/COMPLETED.json" ]]; then
  [[ ! -e "$SPLIT" && ! -L "$SPLIT" ]] || \
    die "Stage1 split output exists without its completion marker: $SPLIT"
  "$PYTHON" "$CODE/navigation_rlt_stage1_gate.py" split \
    --replay-manifest "$RELOCATED_MANIFEST" \
    --expected-replay-manifest-sha256 "$RELOCATED_SHA" \
    --expected-source-fingerprint "$SOURCE_FINGERPRINT" \
    --output-dir "$SPLIT"
fi
mapfile -t split_marker < <(json_values \
  "$SPLIT/COMPLETED.json" navigation_rlt_stage1_gate completed split \
  receipt_sha256 train_manifest_sha256 dev_manifest_sha256)
SPLIT_RECEIPT="$SPLIT/stage1_gate_receipt.json"
require_sha "$SPLIT_RECEIPT" "${split_marker[0]}"
mapfile -t split_receipt < <(json_values \
  "$SPLIT_RECEIPT" navigation_rlt_stage1_gate completed split \
  parent_replay_manifest.sha256 parent_replay_manifest.source_fingerprint \
  subsets.representation_train.manifest_path \
  subsets.representation_train.manifest_sha256 \
  subsets.representation_train.completion_marker_path \
  subsets.representation_train.episode_count \
  subsets.representation_dev.manifest_sha256 \
  subsets.representation_dev.episode_count)
[[ "${split_receipt[0]}" == "$RELOCATED_SHA" ]] || die "Split parent SHA differs"
[[ "${split_receipt[1]}" == "$SOURCE_FINGERPRINT" ]] || die "Split source fingerprint differs"
[[ "${split_receipt[3]}" == "${split_marker[1]}" ]] || die "Train subset SHA differs"
[[ "${split_receipt[5]}" == 224 ]] || die "Stage1 train subset is not 224 episodes"
[[ "${split_receipt[6]}" == "${split_marker[2]}" ]] || die "Heldout subset SHA differs"
[[ "${split_receipt[7]}" == 32 ]] || die "Stage1 heldout subset is not 32 episodes"
TRAIN_MANIFEST=${split_receipt[2]}
TRAIN_SHA=${split_receipt[3]}
TRAIN_MARKER=${split_receipt[4]}
require_sha "$TRAIN_MANIFEST" "$TRAIN_SHA"
mapfile -t train_package < <(json_values \
  "$TRAIN_MARKER" navigation_rlt_prepare_training completed aggregate-prefix \
  artifact_sha256)
[[ "${train_package[0]}" == "$TRAIN_SHA" ]] || die "Train package marker SHA differs"

if [[ ! -f "$PREPARED/COMPLETED.json" ]]; then
  [[ ! -e "$PREPARED" && ! -L "$PREPARED" ]] || \
    die "Stage1 preparation exists without its completion marker: $PREPARED"
  "$PYTHON" "$CODE/navigation_rlt_prepare_training.py" stage1 \
    --replay-manifest "$TRAIN_MANIFEST" \
    --expected-replay-manifest-sha256 "$TRAIN_SHA" \
    --replay-completion-marker "$TRAIN_MARKER" \
    --expected-run-id "$RUN_ID" \
    --expected-episode-count 224 \
    --seed 2026100202 \
    --required-world-size 8 \
    --epochs 5 \
    --batch-size-per-rank 16 \
    --learning-rate 0.0001 \
    --weight-decay 0.0001 \
    --clip-grad 1.0 \
    --max-prefix-tokens 192 \
    --attention-heads 8 \
    --transformer-layers 2 \
    --feedforward-dim 768 \
    --critic-hidden-dim 256 \
    --dropout 0.0 \
    --residual-forward-limit-m 0.05 \
    --residual-yaw-limit-rad 0.17453292519943295 \
    --initial-log-std -1.5 \
    --min-log-std -5.0 \
    --max-log-std 1.0 \
    --prefix-mask-ratio 0.25 \
    --topology-mask-ratio 0.25 \
    --prefix-weight 1.0 \
    --topology-weight 1.0 \
    --output-dir "$PREPARED" \
    --output-manifest run_manifest.json
fi
mapfile -t prepared_marker < <(json_values \
  "$PREPARED/COMPLETED.json" navigation_rlt_prepare_training completed stage1 \
  artifact_sha256 receipt_sha256)
STAGE1_RUN_MANIFEST="$PREPARED/run_manifest.json"
STAGE1_RUN_SHA=${prepared_marker[0]}
require_sha "$STAGE1_RUN_MANIFEST" "$STAGE1_RUN_SHA"
require_sha "$PREPARED/prepare_receipt.json" "${prepared_marker[1]}"

if [[ ! -e "$STAGE" && ! -L "$STAGE" ]]; then
  "$CODE/run_navigation_rlt_a800.sh" \
    "$RUN" stage1 \
    "$STAGE1_RUN_MANIFEST" "$STAGE1_RUN_SHA" \
    "$TRAIN_MANIFEST" "$TRAIN_SHA"
elif [[ ! -f "$STAGE/STAGE.json" ]]; then
  die "Stage1 output exists without STAGE.json: $STAGE"
fi

mapfile -t stage1 < <("$PYTHON" - \
  "$STAGE/STAGE.json" "$STAGE1_RUN_SHA" "$TRAIN_SHA" <<'PY'
import json
import sys
from pathlib import Path

path, expected_run, expected_replay = sys.argv[1:]
value = json.loads(Path(path).read_text(encoding="utf-8"))
if (
    value.get("schema_name") != "navigation_rlt_a800_stage"
    or value.get("status") != "completed"
    or value.get("phase") != "stage1"
    or value.get("exit_code") != 0
    or value.get("run_manifest", {}).get("sha256") != expected_run
    or value.get("replay_manifest", {}).get("sha256") != expected_replay
):
    raise SystemExit(f"Stage1 is absent, active, or failed: {path}")
checkpoint = value.get("output", {}).get("checkpoint")
if not isinstance(checkpoint, dict):
    raise SystemExit("Stage1 marker lacks its checkpoint")
print(checkpoint["path"])
print(checkpoint["sha256"])
print(value["run_manifest"]["sha256"])
print(value["replay_manifest"]["sha256"])
PY
)
STAGE1_CHECKPOINT=${stage1[0]}
STAGE1_CHECKPOINT_SHA=${stage1[1]}
STAGE1_STAGE_RUN_SHA=${stage1[2]}
STAGE1_STAGE_REPLAY_SHA=${stage1[3]}
require_sha "$STAGE1_CHECKPOINT" "$STAGE1_CHECKPOINT_SHA"

if [[ -f "$GATE/REJECTED.json" ]]; then
  echo "Stage1 heldout gate is already rejected; Stage2 remains forbidden." >&2
  exit 3
fi
if [[ ! -f "$GATE/COMPLETED.json" ]]; then
  [[ ! -e "$GATE" && ! -L "$GATE" ]] || \
    die "Stage1 gate output exists without a terminal marker: $GATE"
  "$PYTHON" "$CODE/navigation_rlt_stage1_gate.py" evaluate \
    --split-output-dir "$SPLIT" \
    --expected-split-receipt-sha256 "${split_marker[0]}" \
    --expected-parent-replay-manifest-sha256 "$RELOCATED_SHA" \
    --expected-source-fingerprint "$SOURCE_FINGERPRINT" \
    --stage1-checkpoint "$STAGE1_CHECKPOINT" \
    --expected-stage1-checkpoint-sha256 "$STAGE1_CHECKPOINT_SHA" \
    --expected-stage1-run-manifest-sha256 "$STAGE1_STAGE_RUN_SHA" \
    --expected-stage1-replay-manifest-sha256 "$STAGE1_STAGE_REPLAY_SHA" \
    --output-dir "$GATE" \
    --batch-size 64 \
    --prefix-mask-ratio 0.25 \
    --topology-mask-ratio 0.25 \
    --seed 20261002 \
    --minimum-relative-improvement 0.0 \
    --minimum-z-variance 1e-8
fi
mapfile -t gate < <(json_values \
  "$GATE/COMPLETED.json" navigation_rlt_stage1_gate completed evaluate \
  gate_passed receipt_sha256 checkpoint_sha256)
[[ "${gate[0]}" == true ]] || die "Stage1 heldout gate did not pass"
[[ "${gate[2]}" == "$STAGE1_CHECKPOINT_SHA" ]] || die "Gate checkpoint SHA differs"
require_sha "$GATE/stage1_gate_receipt.json" "${gate[1]}"

printf 'Stage1 heldout gate passed; Stage2 was not started. marker=%s\n' \
  "$GATE/COMPLETED.json"
