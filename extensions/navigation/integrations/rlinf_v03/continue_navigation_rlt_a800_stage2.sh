#!/usr/bin/env bash
set -euo pipefail
umask 077

if [[ $# -ne 0 ]]; then
  echo "This run-bound Stage2 continuation accepts no arguments." >&2
  exit 2
fi

RUN=/srv/navigation/runs/navigation-rlt-20261002-v2
CODE="$RUN/code"
PROVENANCE="$RUN/provenance"
PYTHON=/srv/navigation/envs/rlinf-v03-cu118/bin/python
RUN_ID=navigation-rlt-20261002-v2
SOURCE_FINGERPRINT=1741b8f23682cdc3b1dd2e781fb7c05b5f667b816f5001e4f4880c45abdc1c01

STAGE1="$RUN/navigation_rlt_stage1"
GATE="$RUN/navigation_rlt_stage1_gate"
FULL_PREFIX="$RUN/navigation_rlt_prefix_relocated"
PROMOTED="$RUN/navigation_rlt_stage2_replay"
PREPARED="$RUN/navigation_rlt_stage2_prepare"
STAGE2="$RUN/navigation_rlt_stage2"

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

for binding in \
  "$PROVENANCE/training_protocol.json:7ef54354df67f33ecfbb7ea897405ca349d3f236fa0f1566754ad2b1333d1ff7" \
  "$CODE/navigation_rlt_promote_replay.py:dc955a3cf82cd5e0b9d9b12bf39235587a5c23cde26582ce61681d747c4c687d" \
  "$CODE/navigation_rlt_prepare_training.py:889eb745d7bd40afef65dc3a425eefaf783d4db68ba20781d85c31d97daee8db" \
  "$CODE/navigation_rlt_fsdp_train.py:22609a6071d73e3752a43cae6515e42f8de6b8e43990b4c48fdd4d3edf23f00c" \
  "$CODE/navigation_rlt_policy.py:8611bfff992344fd7d9c9a7702f62185ed32a152ad62ee729f94a1ec69dd970b" \
  "$CODE/navigation_rlt_replay.py:f8fdfd33568bbdb83047c6372181c6c71822381ee049b30d32aea267e5743c94" \
  "$CODE/navigation_rlt_relocate_replay.py:88fde9513ccc926d931bf90f5abbe05c7824f192472258983cb6aef3be29bc5b" \
  "$CODE/navigation_rlt_stage1_gate.py:4e198f38acf778a262fe511a7f5ad8a4f38dc9781705aa99b2c17507053d2725" \
  "$CODE/lightnav_a3_gpu_monitor.py:9e890e8550a3c881a2ad5b87c112f8123b5046231f39b3fd3804653ec553903f" \
  "$CODE/run_navigation_rlt_a800.sh:ad5996d605a71a9f7a2fc2f6af099fe0ee91ce11393eb945f097110b502ae2ba"
do
  require_sha "${binding%:*}" "${binding##*:}"
done

[[ ! -e "$GATE/REJECTED.json" && ! -L "$GATE/REJECTED.json" ]] || \
  die "Stage1 gate is REJECTED; Stage2 is forbidden."
[[ ! -e "$GATE/FAILED.json" && ! -L "$GATE/FAILED.json" ]] || \
  die "Stage1 gate has a failure marker; Stage2 is forbidden."

# Validate the full protocol and bind the gate to the exact checkpoint, run
# manifest and replay manifest recorded by the completed Stage1 wrapper.
mapfile -t stage1 < <("$PYTHON" - \
  "$RUN" "$RUN_ID" "$SOURCE_FINGERPRINT" \
  "$PROVENANCE/training_protocol.json" \
  "$STAGE1/STAGE.json" "$GATE/COMPLETED.json" \
  "$GATE/stage1_gate_receipt.json" \
  "$FULL_PREFIX/COMPLETED.json" "$FULL_PREFIX/replay_manifest.json" <<'PY'
import hashlib
import json
import math
import re
import sys
from pathlib import Path

(run_s, run_id, source_fingerprint, protocol_s, stage_s, gate_marker_s,
 gate_receipt_s, relocation_marker_s, full_manifest_s) = sys.argv[1:]
run = Path(run_s).resolve(strict=True)
sha_pattern = re.compile(r"[0-9a-f]{64}")

def fail(message):
    raise SystemExit(message)

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def regular(value, name):
    path = Path(value)
    if path.is_symlink():
        fail(f"{name} must not be a symlink")
    try:
        path = path.resolve(strict=True)
        path.relative_to(run)
    except (OSError, ValueError):
        fail(f"{name} is missing or escapes the fixed run")
    if not path.is_file():
        fail(f"{name} must be a regular file")
    return path

def load(value, name):
    path = regular(value, name)
    try:
        data = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda item: fail(f"{name} contains {item}"),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        fail(f"{name} is not strict JSON: {error}")
    if not isinstance(data, dict):
        fail(f"{name} must contain an object")
    return path, data

def digest(value, name):
    if not isinstance(value, str) or sha_pattern.fullmatch(value) is None:
        fail(f"{name} is not a lowercase SHA-256")
    return value

protocol_path, protocol = load(protocol_s, "training protocol")
expected_top = {
    "schema_name", "schema_version", "run_id", "preregistration_commit",
    "navigation_training_code_commit", "boundaries", "rollout", "model",
    "stage1", "heldout_gate", "stage2", "stage1_checkpoint_provenance_gate",
}
if set(protocol) != expected_top:
    fail("training protocol top-level fields differ")
if (
    protocol.get("schema_name") != "navigation_rlt_training_protocol"
    or protocol.get("schema_version") != 1
    or protocol.get("run_id") != run_id
    or protocol.get("navigation_training_code_commit")
       != "d2e7dc44db4865cb45374db178470e4c6b9fc61a"
):
    fail("training protocol run/code identity differs")
if protocol.get("boundaries") != {
    "lightnav_frozen": True,
    "remote_push_allowed": False,
    "ttt_allowed": False,
    "validation_splits_allowed": False,
}:
    fail("training protocol boundaries differ")
if protocol.get("stage1_checkpoint_provenance_gate") != {
    "required": True,
    "run_manifest_sha256_source":
        "navigation_rlt_stage1/STAGE.json:run_manifest.sha256",
    "replay_manifest_sha256_source":
        "navigation_rlt_stage1/STAGE.json:replay_manifest.sha256",
    "required_cli_arguments": [
        "--expected-stage1-run-manifest-sha256",
        "--expected-stage1-replay-manifest-sha256",
    ],
}:
    fail("training protocol Stage1 provenance gate differs")
expected_stage1 = {
    "seed": 2026100202,
    "epochs": 5,
    "batch_size_per_rank": 16,
    "learning_rate": 0.0001,
    "weight_decay": 0.0001,
    "clip_grad": 1.0,
    "prefix_mask_ratio": 0.25,
    "topology_mask_ratio": 0.25,
    "prefix_weight": 1.0,
    "topology_weight": 1.0,
    "required_world_size": 8,
    "expected_episode_count": 224,
}
expected_gate = {
    "seed": 20261002,
    "batch_size": 64,
    "prefix_mask_ratio": 0.25,
    "topology_mask_ratio": 0.25,
    "minimum_relative_improvement": 0.0,
    "minimum_z_rl_variance": 1e-8,
}
expected_stage2 = {
    "seed": 2026100203,
    "epochs": 10,
    "batch_size_per_rank": 64,
    "learning_rate": 0.0003,
    "weight_decay": 0.0001,
    "clip_grad": 1.0,
    "reference_gate_weight": 0.25,
    "reference_residual_weight": 0.1,
    "reference_dropout_probability": 0.2,
    "target_tau": 0.005,
    "policy_version": 200001,
    "required_world_size": 8,
    "expected_episode_count": 256,
}
expected_model = {
    "attention_heads": 8,
    "critic_hidden_dim": 256,
    "dropout": 0.0,
    "feedforward_dim": 768,
    "initial_log_std": -1.5,
    "max_log_std": 1.0,
    "max_prefix_tokens": 192,
    "min_log_std": -5.0,
    "residual_forward_limit_m": 0.05,
    "residual_yaw_limit_rad": 0.17453292519943295,
    "transformer_layers": 2,
}
if protocol.get("stage1") != expected_stage1:
    fail("Stage1 protocol fields differ")
if protocol.get("heldout_gate") != expected_gate:
    fail("heldout gate protocol fields differ")
if protocol.get("stage2") != expected_stage2:
    fail("Stage2 protocol fields differ")
if protocol.get("model") != expected_model:
    fail("model protocol fields differ")
rollout = protocol.get("rollout")
if (
    not isinstance(rollout, dict)
    or set(rollout) != {
        "behavior", "episode_count", "outer_manifest_sha256",
        "parallel_plan_sha256", "split",
    }
    or rollout.get("behavior") != "a3"
    or rollout.get("episode_count") != 256
    or rollout.get("split") != "train"
):
    fail("rollout protocol differs from the fixed train-only collection")

stage_path, stage = load(stage_s, "Stage1 stage marker")
if (
    stage.get("schema_name") != "navigation_rlt_a800_stage"
    or stage.get("schema_version") != 1
    or stage.get("status") != "completed"
    or stage.get("phase") != "stage1"
    or stage.get("exit_code") != 0
    or Path(str(stage.get("run_root"))).resolve(strict=True) != run
):
    fail("Stage1 is absent, active, partial, or failed")
stage_run = stage.get("run_manifest")
stage_replay = stage.get("replay_manifest")
output = stage.get("output")
checkpoint = output.get("checkpoint") if isinstance(output, dict) else None
verification = stage.get("verification")
if not all(isinstance(value, dict) for value in (
    stage_run, stage_replay, checkpoint, verification
)):
    fail("Stage1 marker lacks run/replay/checkpoint/verification evidence")
stage_run_sha = digest(stage_run.get("sha256"), "Stage1 run manifest SHA-256")
stage_replay_sha = digest(
    stage_replay.get("sha256"), "Stage1 replay manifest SHA-256"
)
checkpoint_sha = digest(checkpoint.get("sha256"), "Stage1 checkpoint SHA-256")
verification_sha = digest(
    verification.get("sha256"), "Stage1 verification SHA-256"
)
stage_run_path, stage_run_value = load(stage_run.get("path"), "Stage1 run manifest")
stage_replay_path, stage_replay_value = load(
    stage_replay.get("path"), "Stage1 replay manifest"
)
checkpoint_path = regular(checkpoint.get("path"), "Stage1 checkpoint")
verification_path, verification_value = load(
    verification.get("path"), "Stage1 verification"
)
for path, expected, name in (
    (stage_run_path, stage_run_sha, "Stage1 run manifest"),
    (stage_replay_path, stage_replay_sha, "Stage1 replay manifest"),
    (checkpoint_path, checkpoint_sha, "Stage1 checkpoint"),
    (verification_path, verification_sha, "Stage1 verification"),
):
    if sha256(path) != expected:
        fail(f"{name} SHA-256 differs")
stage1_training = {
    key: value for key, value in expected_stage1.items()
    if key not in {"seed", "required_world_size", "expected_episode_count"}
}
if (
    stage_run_value.get("schema_name") != "navigation_rlt_training_run_manifest"
    or stage_run_value.get("schema_version") != 1
    or stage_run_value.get("run_id") != run_id
    or stage_run_value.get("phase") != "stage1"
    or stage_run_value.get("seed") != expected_stage1["seed"]
    or stage_run_value.get("required_world_size") != 8
    or stage_run_value.get("training") != stage1_training
    or stage_run_value.get("replay", {}).get("manifest_sha256")
       != stage_replay_sha
):
    fail("Stage1 run manifest differs from the fixed protocol")
if (
    stage_replay_value.get("schema_name") != "navigation_rlt_replay_manifest"
    or stage_replay_value.get("schema_version") != 3
    or stage_replay_value.get("run_id") != run_id
    or stage_replay_value.get("representation_kind") != "prefix_hidden"
    or stage_replay_value.get("expected_episode_count") != 224
    or stage_replay_value.get("completed_episode_count") != 224
):
    fail("Stage1 replay is not the completed 224-episode train subset")
if (
    verification_value.get("schema_name") != "navigation_rlt_a800_verification"
    or verification_value.get("status") != "passed"
    or verification_value.get("phase") != "stage1"
    or verification_value.get("world_size") != 8
    or verification_value.get("unique_gpu_uuid_count") != 8
    or verification_value.get("checkpoint", {}).get("sha256") != checkpoint_sha
):
    fail("Stage1 eight-rank/checkpoint verification differs")

relocation_path, relocation = load(
    relocation_marker_s, "full prefix relocation marker"
)
full_path, full = load(full_manifest_s, "full prefix replay manifest")
full_sha = sha256(full_path)
if (
    relocation.get("schema_name") != "navigation_rlt_replay_relocation"
    or relocation.get("status") != "completed"
    or relocation.get("run_id") != run_id
    or relocation.get("relocated_manifest_sha256") != full_sha
):
    fail("full prefix relocation is absent, partial, or mismatched")
if (
    full.get("schema_name") != "navigation_rlt_replay_manifest"
    or full.get("schema_version") != 3
    or full.get("run_id") != run_id
    or full.get("representation_kind") != "prefix_hidden"
    or full.get("expected_episode_count") != 256
    or full.get("completed_episode_count") != 256
    or full.get("source_fingerprint") != source_fingerprint
):
    fail("full parent replay is not the fixed completed 256-episode package")

gate_marker_path, gate_marker = load(gate_marker_s, "Stage1 gate completion")
gate_receipt_path, gate_receipt = load(gate_receipt_s, "Stage1 gate receipt")
gate_marker_sha = sha256(gate_marker_path)
gate_receipt_sha = sha256(gate_receipt_path)
if gate_marker != {
    "schema_name": "navigation_rlt_stage1_gate",
    "schema_version": 1,
    "status": "completed",
    "operation": "evaluate",
    "gate_passed": True,
    "receipt_sha256": gate_receipt_sha,
    "checkpoint_sha256": checkpoint_sha,
}:
    fail("Stage1 gate completion is not an exact PASS for this checkpoint")
gate_checkpoint = gate_receipt.get("checkpoint")
gate_split = gate_receipt.get("split")
gate_evaluation = gate_receipt.get("evaluation_contract")
if (
    gate_receipt.get("schema_name") != "navigation_rlt_stage1_gate"
    or gate_receipt.get("schema_version") != 1
    or gate_receipt.get("status") != "passed"
    or gate_receipt.get("operation") != "evaluate"
    or gate_receipt.get("gate") != "episode-heldout-stage1-validity"
    or not isinstance(gate_checkpoint, dict)
    or gate_checkpoint.get("sha256") != checkpoint_sha
    or gate_checkpoint.get("run_manifest_sha256") != stage_run_sha
    or gate_checkpoint.get("replay_manifest_sha256") != stage_replay_sha
    or not isinstance(gate_split, dict)
    or gate_split.get("parent_replay_manifest_sha256") != full_sha
    or gate_split.get("source_fingerprint") != source_fingerprint
    or gate_split.get("train_manifest_sha256") != stage_replay_sha
    or gate_split.get("train_episode_count") != 224
    or gate_split.get("dev_episode_count") != 32
    or gate_split.get("episode_overlap_count") != 0
    or not isinstance(gate_evaluation, dict)
    or {key: gate_evaluation.get(key) for key in expected_gate} != expected_gate
):
    fail("Stage1 gate receipt does not bind the exact Stage1/full replay protocol")
checkpoint_config = gate_checkpoint.get("config")
if (
    not isinstance(checkpoint_config, dict)
    or {key: checkpoint_config.get(key) for key in expected_model} != expected_model
):
    fail("Stage1 checkpoint model config differs from the training protocol")

for value in (
    checkpoint_path, checkpoint_sha, stage_run_sha, stage_replay_sha,
    full_sha, gate_marker_sha, gate_receipt_sha,
):
    print(value)
PY
)
[[ ${#stage1[@]} -eq 7 ]] || die "Stage1/gate validation returned incomplete evidence"
STAGE1_CHECKPOINT=${stage1[0]}
STAGE1_CHECKPOINT_SHA=${stage1[1]}
STAGE1_RUN_SHA=${stage1[2]}
STAGE1_REPLAY_SHA=${stage1[3]}
FULL_PREFIX_SHA=${stage1[4]}
GATE_MARKER_SHA=${stage1[5]}
GATE_RECEIPT_SHA=${stage1[6]}

if [[ ! -f "$PROMOTED/COMPLETED.json" ]]; then
  [[ ! -e "$PROMOTED" && ! -L "$PROMOTED" ]] || \
    die "Stage2 replay output exists without a completion marker: $PROMOTED"
  "$PYTHON" "$CODE/navigation_rlt_promote_replay.py" \
    --input-manifest "$FULL_PREFIX/replay_manifest.json" \
    --expected-input-manifest-sha256 "$FULL_PREFIX_SHA" \
    --stage1-checkpoint "$STAGE1_CHECKPOINT" \
    --expected-stage1-checkpoint-sha256 "$STAGE1_CHECKPOINT_SHA" \
    --stage1-gate-completion-marker "$GATE/COMPLETED.json" \
    --expected-stage1-gate-completion-marker-sha256 "$GATE_MARKER_SHA" \
    --output-dir "$PROMOTED" \
    --device cuda:0 \
    --batch-size 128
fi

mapfile -t promotion < <("$PYTHON" - \
  "$RUN" "$RUN_ID" "$SOURCE_FINGERPRINT" "$PROMOTED/COMPLETED.json" \
  "$PROMOTED/replay_manifest.json" "$PROMOTED/promotion_receipt.json" \
  "$FULL_PREFIX_SHA" "$STAGE1_CHECKPOINT_SHA" "$STAGE1_RUN_SHA" \
  "$STAGE1_REPLAY_SHA" "$GATE_MARKER_SHA" "$GATE_RECEIPT_SHA" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path

(run_s, run_id, source_fingerprint, marker_s, manifest_s, receipt_s,
 input_sha, checkpoint_sha, stage1_run_sha, stage1_replay_sha,
 gate_marker_sha, gate_receipt_sha) = sys.argv[1:]
run = Path(run_s).resolve(strict=True)
pattern = re.compile(r"[0-9a-f]{64}")

def fail(message):
    raise SystemExit(message)

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def load(value, name):
    path = Path(value)
    if path.is_symlink():
        fail(f"{name} must not be a symlink")
    try:
        path = path.resolve(strict=True)
        path.relative_to(run)
    except (OSError, ValueError):
        fail(f"{name} is missing or escapes the fixed run")
    if not path.is_file():
        fail(f"{name} must be a regular file")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        fail(f"{name} is not JSON: {error}")
    if not isinstance(data, dict):
        fail(f"{name} must contain an object")
    return path, data

marker_path, marker = load(marker_s, "promotion completion")
manifest_path, manifest = load(manifest_s, "promoted replay manifest")
receipt_path, receipt = load(receipt_s, "promotion receipt")
manifest_sha = sha256(manifest_path)
receipt_sha = sha256(receipt_path)
if marker != {
    "schema_name": "navigation_rlt_replay_promotion",
    "schema_version": 1,
    "status": "completed",
    "promotion_receipt_sha256": receipt_sha,
    "output_manifest_sha256": manifest_sha,
    "stage1_checkpoint_sha256": checkpoint_sha,
    "stage1_gate_marker_sha256": gate_marker_sha,
    "stage1_gate_receipt_sha256": gate_receipt_sha,
}:
    fail("Stage2 replay promotion marker differs")
if (
    manifest.get("schema_name") != "navigation_rlt_replay_manifest"
    or manifest.get("schema_version") != 3
    or manifest.get("run_id") != run_id
    or manifest.get("representation_kind") != "frozen_z_rl"
    or manifest.get("expected_episode_count") != 256
    or manifest.get("completed_episode_count") != 256
    or manifest.get("behavior_policy_id")
       != "navigation_rlt_warmstart_lineage_v1"
    or manifest.get("behavior_policy_versions") != [100001]
):
    fail("promoted replay is not the completed full 256-episode frozen package")
promoted_source_fingerprint = manifest.get("source_fingerprint")
if (
    not isinstance(promoted_source_fingerprint, str)
    or pattern.fullmatch(promoted_source_fingerprint) is None
):
    fail("promoted replay source fingerprint is invalid")
gate = receipt.get("stage1_gate")
checkpoint = receipt.get("stage1_checkpoint")
source = receipt.get("input")
output = receipt.get("output")
if (
    receipt.get("schema_name") != "navigation_rlt_replay_promotion"
    or receipt.get("schema_version") != 1
    or receipt.get("status") != "completed"
    or not isinstance(gate, dict)
    or gate.get("gate_passed") is not True
    or gate.get("marker_sha256") != gate_marker_sha
    or gate.get("receipt_sha256") != gate_receipt_sha
    or gate.get("checkpoint_sha256") != checkpoint_sha
    or gate.get("run_manifest_sha256") != stage1_run_sha
    or gate.get("replay_manifest_sha256") != stage1_replay_sha
    or gate.get("parent_replay_manifest_sha256") != input_sha
    or gate.get("source_fingerprint") != source_fingerprint
    or not isinstance(checkpoint, dict)
    or checkpoint.get("sha256") != checkpoint_sha
    or not isinstance(source, dict)
    or source.get("manifest_sha256") != input_sha
    or source.get("source_fingerprint") != source_fingerprint
    or not isinstance(output, dict)
    or output.get("manifest_sha256") != manifest_sha
    or output.get("source_fingerprint") != promoted_source_fingerprint
    or output.get("episode_count") != 256
    or output.get("behavior_policy_versions") != [100001]
):
    fail("promotion does not bind the exact Stage1 checkpoint/run/replay")
if pattern.fullmatch(manifest_sha) is None:
    fail("promoted replay SHA-256 is invalid")
print(manifest_sha)
print(receipt_sha)
print(promoted_source_fingerprint)
PY
)
[[ ${#promotion[@]} -eq 3 ]] || die "Promotion validation returned incomplete evidence"
PROMOTED_SHA=${promotion[0]}
PROMOTED_SOURCE_FINGERPRINT=${promotion[2]}

if [[ ! -f "$PREPARED/COMPLETED.json" ]]; then
  [[ ! -e "$PREPARED" && ! -L "$PREPARED" ]] || \
    die "Stage2 preparation exists without a completion marker: $PREPARED"
  "$PYTHON" "$CODE/navigation_rlt_prepare_training.py" stage2 \
    --replay-manifest "$PROMOTED/replay_manifest.json" \
    --expected-replay-manifest-sha256 "$PROMOTED_SHA" \
    --replay-completion-marker "$PROMOTED/COMPLETED.json" \
    --expected-run-id "$RUN_ID" \
    --expected-episode-count 256 \
    --seed 2026100203 \
    --required-world-size 8 \
    --epochs 10 \
    --batch-size-per-rank 64 \
    --learning-rate 0.0003 \
    --weight-decay 0.0001 \
    --clip-grad 1.0 \
    --stage1-checkpoint "$STAGE1_CHECKPOINT" \
    --expected-stage1-checkpoint-sha256 "$STAGE1_CHECKPOINT_SHA" \
    --reference-gate-weight 0.25 \
    --reference-residual-weight 0.10 \
    --reference-dropout-probability 0.20 \
    --target-tau 0.005 \
    --policy-version 200001 \
    --output-dir "$PREPARED" \
    --output-manifest run_manifest.json
fi

mapfile -t prepared < <("$PYTHON" - \
  "$RUN" "$RUN_ID" "$PREPARED/COMPLETED.json" \
  "$PREPARED/run_manifest.json" "$PREPARED/prepare_receipt.json" \
  "$PROMOTED_SHA" "$PROMOTED_SOURCE_FINGERPRINT" \
  "$STAGE1_CHECKPOINT_SHA" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

(run_s, run_id, marker_s, manifest_s, receipt_s, replay_sha,
 promoted_source_fingerprint, checkpoint_sha) = sys.argv[1:]
run = Path(run_s).resolve(strict=True)

def fail(message):
    raise SystemExit(message)

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def load(value, name):
    path = Path(value)
    if path.is_symlink():
        fail(f"{name} must not be a symlink")
    try:
        path = path.resolve(strict=True)
        path.relative_to(run)
    except (OSError, ValueError):
        fail(f"{name} is missing or escapes the fixed run")
    if not path.is_file():
        fail(f"{name} must be a regular file")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        fail(f"{name} must contain an object")
    return path, data

marker_path, marker = load(marker_s, "Stage2 prepare completion")
manifest_path, manifest = load(manifest_s, "Stage2 run manifest")
receipt_path, receipt = load(receipt_s, "Stage2 prepare receipt")
manifest_sha = sha256(manifest_path)
receipt_sha = sha256(receipt_path)
expected_training = {
    "epochs": 10,
    "batch_size_per_rank": 64,
    "learning_rate": 0.0003,
    "weight_decay": 0.0001,
    "clip_grad": 1.0,
    "reference_gate_weight": 0.25,
    "reference_residual_weight": 0.1,
    "reference_dropout_probability": 0.2,
    "target_tau": 0.005,
    "policy_version": 200001,
}
if marker != {
    "schema_name": "navigation_rlt_prepare_training",
    "schema_version": 1,
    "status": "completed",
    "operation": "stage2",
    "artifact_kind": "run_manifest",
    "artifact_sha256": manifest_sha,
    "receipt_sha256": receipt_sha,
}:
    fail("Stage2 prepare completion marker differs")
if (
    manifest.get("schema_name") != "navigation_rlt_training_run_manifest"
    or manifest.get("schema_version") != 1
    or manifest.get("run_id") != run_id
    or manifest.get("phase") != "stage2"
    or manifest.get("seed") != 2026100203
    or manifest.get("required_world_size") != 8
    or manifest.get("replay", {}).get("manifest_sha256") != replay_sha
    or manifest.get("replay", {}).get("source_fingerprint")
       != promoted_source_fingerprint
    or manifest.get("stage1_checkpoint") != {"sha256": checkpoint_sha}
    or manifest.get("training") != expected_training
):
    fail("Stage2 run manifest differs from the fixed protocol")
if (
    receipt.get("schema_name") != "navigation_rlt_prepare_training"
    or receipt.get("status") != "completed"
    or receipt.get("operation") != "stage2"
    or receipt.get("run_id") != run_id
    or receipt.get("replay", {}).get("manifest_sha256") != replay_sha
    or receipt.get("replay", {}).get("source_fingerprint")
       != promoted_source_fingerprint
    or receipt.get("replay", {}).get("expected_episode_count") != 256
    or receipt.get("replay", {}).get("episode_count") != 256
    or receipt.get("stage1_checkpoint", {}).get("sha256") != checkpoint_sha
    or receipt.get("training") != expected_training
):
    fail("Stage2 prepare receipt differs from the fixed protocol")
print(manifest_sha)
PY
)
[[ ${#prepared[@]} -eq 1 ]] || die "Stage2 preparation validation returned no manifest SHA"
STAGE2_RUN_SHA=${prepared[0]}

if [[ ! -e "$STAGE2" && ! -L "$STAGE2" ]]; then
  "$CODE/run_navigation_rlt_a800.sh" \
    "$RUN" stage2 \
    "$PREPARED/run_manifest.json" "$STAGE2_RUN_SHA" \
    "$PROMOTED/replay_manifest.json" "$PROMOTED_SHA" \
    "$STAGE1_CHECKPOINT" "$STAGE1_CHECKPOINT_SHA"
elif [[ ! -f "$STAGE2/STAGE.json" || -L "$STAGE2/STAGE.json" ]]; then
  die "Stage2 output exists without a safe STAGE.json: $STAGE2"
fi

mapfile -t completed < <("$PYTHON" - \
  "$RUN" "$RUN_ID" "$STAGE2/STAGE.json" "$STAGE2_RUN_SHA" \
  "$PROMOTED_SHA" "$PROMOTED_SOURCE_FINGERPRINT" \
  "$STAGE1_CHECKPOINT_SHA" "$PREPARED/run_manifest.json" \
  "$PROMOTED/replay_manifest.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

(run_s, run_id, marker_s, run_sha, replay_sha, promoted_source_fingerprint,
 checkpoint_input_sha, run_manifest_s, replay_manifest_s) = sys.argv[1:]
run = Path(run_s).resolve(strict=True)

def fail(message):
    raise SystemExit(message)

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def regular(value, name):
    path = Path(value)
    if path.is_symlink():
        fail(f"{name} must not be a symlink")
    try:
        path = path.resolve(strict=True)
        path.relative_to(run)
    except (OSError, ValueError):
        fail(f"{name} is missing or escapes the fixed run")
    if not path.is_file():
        fail(f"{name} must be a regular file")
    return path

marker_path = regular(marker_s, "Stage2 marker")
marker = json.loads(marker_path.read_text(encoding="utf-8"))
run_manifest_path = regular(run_manifest_s, "Stage2 run manifest")
replay_manifest_path = regular(replay_manifest_s, "promoted replay manifest")
run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
replay_manifest = json.loads(replay_manifest_path.read_text(encoding="utf-8"))
checkpoint = marker.get("output", {}).get("checkpoint")
verification = marker.get("verification")
if (
    marker.get("schema_name") != "navigation_rlt_a800_stage"
    or marker.get("schema_version") != 1
    or marker.get("status") != "completed"
    or marker.get("exit_code") != 0
    or marker.get("phase") != "stage2"
    or Path(str(marker.get("run_root"))).resolve(strict=True) != run
    or marker.get("run_manifest", {}).get("sha256") != run_sha
    or marker.get("replay_manifest", {}).get("sha256") != replay_sha
    or sha256(run_manifest_path) != run_sha
    or sha256(replay_manifest_path) != replay_sha
    or run_manifest.get("replay", {}).get("source_fingerprint")
       != promoted_source_fingerprint
    or replay_manifest.get("source_fingerprint") != promoted_source_fingerprint
    or replay_manifest.get("behavior_policy_id")
       != "navigation_rlt_warmstart_lineage_v1"
    or replay_manifest.get("behavior_policy_versions") != [100001]
    or marker.get("stage1_checkpoint", {}).get("sha256") != checkpoint_input_sha
    or not isinstance(checkpoint, dict)
    or not isinstance(verification, dict)
):
    fail("Stage2 is absent, active, partial, failed, or bound to different inputs")
checkpoint_path = regular(checkpoint.get("path"), "Stage2 checkpoint")
checkpoint_sha = sha256(checkpoint_path)
if checkpoint.get("sha256") != checkpoint_sha:
    fail("Stage2 checkpoint SHA-256 differs")
verification_path = regular(verification.get("path"), "Stage2 verification")
if verification.get("sha256") != sha256(verification_path):
    fail("Stage2 verification SHA-256 differs")
verification_value = json.loads(verification_path.read_text(encoding="utf-8"))
if (
    verification_value.get("schema_name") != "navigation_rlt_a800_verification"
    or verification_value.get("status") != "passed"
    or verification_value.get("phase") != "stage2"
    or verification_value.get("world_size") != 8
    or verification_value.get("unique_gpu_uuid_count") != 8
    or verification_value.get("checkpoint", {}).get("path") != str(checkpoint_path)
    or verification_value.get("checkpoint", {}).get("sha256") != checkpoint_sha
):
    fail("Stage2 eight-rank/checkpoint verification differs")
print(checkpoint_path)
print(checkpoint_sha)
print(sha256(marker_path))
PY
)
[[ ${#completed[@]} -eq 3 ]] || die "Stage2 completion validation returned incomplete evidence"

printf 'Stage2 completed and verified; post-training work was not started. checkpoint=%s sha256=%s stage_marker_sha256=%s\n' \
  "${completed[0]}" "${completed[1]}" "${completed[2]}"
