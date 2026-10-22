#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
umask 077

if [[ $# -ne 0 ]]; then
  echo "This run-bound anti-collapse smoke launcher accepts no arguments." >&2
  exit 2
fi

RUN=/srv/navigation/runs/navigation-rlt-20261002-v2
RUN_ID=navigation-rlt-20261002-v2
RUNS_ROOT=/srv/navigation/runs
ENV_ROOT=/srv/navigation/envs/rlinf-v03-cu118
PYTHON="$ENV_ROOT/bin/python"
SMOKE_ID=stage2-anticollapse-fe6b16d-2epoch-v1
SMOKE_PARENT="$RUN/stage2-smokes"
SMOKE_ROOT="$SMOKE_PARENT/$SMOKE_ID"
CODE_ROOT="$SMOKE_ROOT/code-stage2-repair-fe6b16d6a8dd"
COMPAT_LOCK="$SMOKE_ROOT/.launch-navigation-rlt-stage2-repair-v2"
BASE_PREPARE="$RUN/navigation_rlt_stage2_prepare"
BASE_MANIFEST="$BASE_PREPARE/run_manifest.json"
BASE_COMPLETED="$BASE_PREPARE/COMPLETED.json"
BASE_RECEIPT="$BASE_PREPARE/prepare_receipt.json"
BASE_PROTOCOL="$RUN/provenance/stage2-repair-v2/stage2_repair_protocol.json"
REPLAY_ROOT="$RUN/navigation_rlt_stage2_replay"
REPLAY_MANIFEST="$REPLAY_ROOT/replay_manifest.json"
REPLAY_COMPLETED="$REPLAY_ROOT/COMPLETED.json"
REPLAY_RECEIPT="$REPLAY_ROOT/promotion_receipt.json"
STAGE1_MARKER="$RUN/navigation_rlt_stage1/STAGE.json"
STAGE1_GATE="$RUN/navigation_rlt_stage1_gate/COMPLETED.json"
STORAGE_GUARD="$RUN/code/navigation_rlt_storage_guard.py"
STORAGE_SAFETY_ROOT="$RUN/storage-safety"
PENDING_WRITE_BYTES=17179869184
TURBO_SOURCE='navigation.storage.example.invalid:/'

ANTI_COLLAPSE_COMMIT=fe6b16d6a8dd5421d9667a9dd55c65fcf234cc44
EXPECTED_STAGE1_SHA256=744f06f3541b09528d2e5c46c8b85a12d8dbc95d4787cb5e92a7c1e25761b41d
BASE_PROTOCOL_SHA256=8c9ae4586e365a94d2aeacf3467374bb3f01b795f81a288474d81d17e8c1d43e
STORAGE_GUARD_SHA256=2cc42194509297c11d31d475244147372ce6c5ae474a02c36f3932cebfd3df67
TRAINER_SHA256=da4cf14e636842b161a0705df9945267eac75bee5f41ccf6f61a7410d0047e62
POLICY_SHA256=edf4ce64433542df4f798589ac72b76acaa874cc74178b2916c8f7422262afd5
REPLAY_CODE_SHA256=f8fdfd33568bbdb83047c6372181c6c71822381ee049b30d32aea267e5743c94
MONITOR_SHA256=9e890e8550a3c881a2ad5b87c112f8123b5046231f39b3fd3804653ec553903f
RUN_WRAPPER_SHA256=f931936696d7971355880b0d292963f4329bda984a1a2149aba9d7087723d7b6

die() {
  echo "$*" >&2
  exit 1
}

sha256_file() {
  sha256sum -- "$1" | cut -d' ' -f1
}

json_get() {
  local document=$1
  local field=$2
  "$PYTHON" -B - "$document" "$field" <<'PY'
import json
import sys

value = json.loads(sys.argv[1])
for part in sys.argv[2].split('.'):
    if not isinstance(value, dict) or part not in value:
        raise SystemExit(f"missing JSON field: {sys.argv[2]}")
    value = value[part]
if isinstance(value, (dict, list)) or value is None:
    raise SystemExit(f"JSON field is not scalar: {sys.argv[2]}")
print(str(value).lower() if isinstance(value, bool) else value)
PY
}

[[ -x "$PYTHON" ]] || die "fixed RLinf Python is absent"
[[ ! -L "$0" && -f "$0" ]] || die "launcher must be a regular non-symlink file"
SOURCE_ROOT=$(dirname -- "$(realpath -e -- "$0")")
[[ -d "$SOURCE_ROOT" && ! -L "$SOURCE_ROOT" ]] || die "launcher source root is unsafe"

declare -A SOURCE_HASHES=(
  [navigation_rlt_fsdp_train.py]="$TRAINER_SHA256"
  [navigation_rlt_policy.py]="$POLICY_SHA256"
  [navigation_rlt_replay.py]="$REPLAY_CODE_SHA256"
  [lightnav_a3_gpu_monitor.py]="$MONITOR_SHA256"
  [run_navigation_rlt_a800_stage2_repair.sh]="$RUN_WRAPPER_SHA256"
)
for name in "${!SOURCE_HASHES[@]}"; do
  path="$SOURCE_ROOT/$name"
  [[ -f "$path" && ! -L "$path" ]] || die "smoke source is absent or unsafe: $name"
  [[ "$(sha256_file "$path")" == "${SOURCE_HASHES[$name]}" ]] || \
    die "smoke source SHA-256 differs: $name"
done

[[ -f "$STORAGE_GUARD" && ! -L "$STORAGE_GUARD" ]] || die "Turbo guard is absent or unsafe"
[[ "$(sha256_file "$STORAGE_GUARD")" == "$STORAGE_GUARD_SHA256" ]] || \
  die "Turbo guard SHA-256 differs"
[[ "$(stat -c '%a' -- "$STORAGE_GUARD")" == "500" ]] || die "Turbo guard mode differs"
[[ ! -e "$SMOKE_ROOT" && ! -L "$SMOKE_ROOT" ]] || die "fixed smoke root already exists"
if [[ -e "$SMOKE_PARENT" || -L "$SMOKE_PARENT" ]]; then
  [[ -d "$SMOKE_PARENT" && ! -L "$SMOKE_PARENT" ]] || die "smoke parent is unsafe"
fi

# All source artifacts are verified read-only before the first smoke write.
input_json=$("$PYTHON" -B - \
  "$RUN" "$RUN_ID" "$BASE_MANIFEST" "$BASE_COMPLETED" "$BASE_RECEIPT" \
  "$BASE_PROTOCOL" "$BASE_PROTOCOL_SHA256" \
  "$REPLAY_MANIFEST" "$REPLAY_COMPLETED" "$REPLAY_RECEIPT" \
  "$STAGE1_MARKER" "$STAGE1_GATE" "$EXPECTED_STAGE1_SHA256" <<'PY'
import hashlib
import json
import math
from pathlib import Path
import sys

(run_s, run_id, base_manifest_s, base_completed_s, base_receipt_s,
 base_protocol_s, expected_protocol_sha,
 replay_manifest_s, replay_completed_s, replay_receipt_s,
 stage1_marker_s, stage1_gate_s, expected_stage1_sha) = sys.argv[1:]
run = Path(run_s).resolve(strict=True)

def fail(message):
    raise SystemExit(message)

def reject_constant(value):
    fail(f"non-finite JSON constant is forbidden: {value}")

def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def regular(value, name):
    raw = Path(value)
    if raw.is_symlink():
        fail(f"{name} is a symlink")
    try:
        path = raw.resolve(strict=True)
        path.relative_to(run)
    except (OSError, ValueError):
        fail(f"{name} is missing or escapes the run root")
    if not path.is_file():
        fail(f"{name} is not a regular file")
    return path

def load(value, name):
    path = regular(value, name)
    try:
        data = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        fail(f"{name} is not strict JSON: {error}")
    if not isinstance(data, dict):
        fail(f"{name} must contain an object")
    return path, data

base_path, base = load(base_manifest_s, "base Stage2 run manifest")
base_completed_path, base_completed = load(base_completed_s, "base prepare completion")
base_receipt_path, base_receipt = load(base_receipt_s, "base prepare receipt")
base_protocol_path, base_protocol = load(base_protocol_s, "base Stage2 protocol")
replay_path, replay = load(replay_manifest_s, "promoted replay manifest")
replay_completed_path, replay_completed = load(replay_completed_s, "promotion completion")
replay_receipt_path, replay_receipt = load(replay_receipt_s, "promotion receipt")
stage1_marker_path, stage1 = load(stage1_marker_s, "Stage1 marker")
stage1_gate_path, stage1_gate = load(stage1_gate_s, "Stage1 gate")

base_sha = sha(base_path)
base_receipt_sha = sha(base_receipt_path)
replay_sha = sha(replay_path)
replay_receipt_sha = sha(replay_receipt_path)
if sha(base_protocol_path) != expected_protocol_sha:
    fail("base Stage2 protocol SHA-256 differs")
if base_completed != {
    "schema_name": "navigation_rlt_prepare_training",
    "schema_version": 1,
    "status": "completed",
    "operation": "stage2",
    "artifact_kind": "run_manifest",
    "artifact_sha256": base_sha,
    "receipt_sha256": base_receipt_sha,
}:
    fail("base Stage2 prepare completion differs")
checkpoint_ref = stage1.get("output", {}).get("checkpoint")
if (
    stage1.get("schema_name") != "navigation_rlt_a800_stage"
    or stage1.get("schema_version") != 1
    or stage1.get("status") != "completed"
    or stage1.get("exit_code") != 0
    or stage1.get("phase") != "stage1"
    or Path(str(stage1.get("run_root"))).resolve(strict=True) != run
    or not isinstance(checkpoint_ref, dict)
):
    fail("Stage1 marker is absent, partial, or mismatched")
checkpoint_path = regular(checkpoint_ref.get("path", ""), "Stage1 checkpoint")
checkpoint_sha = sha(checkpoint_path)
if checkpoint_sha != expected_stage1_sha or checkpoint_ref.get("sha256") != checkpoint_sha:
    fail("Stage1 checkpoint identity differs")
if (
    stage1_gate.get("schema_name") != "navigation_rlt_stage1_gate"
    or stage1_gate.get("schema_version") != 1
    or stage1_gate.get("status") != "completed"
    or stage1_gate.get("operation") != "evaluate"
    or stage1_gate.get("gate_passed") is not True
    or stage1_gate.get("checkpoint_sha256") != checkpoint_sha
):
    fail("Stage1 gate is not an exact pass")
if (
    replay_completed.get("schema_name") != "navigation_rlt_replay_promotion"
    or replay_completed.get("schema_version") != 1
    or replay_completed.get("status") != "completed"
    or replay_completed.get("promotion_receipt_sha256") != replay_receipt_sha
    or replay_completed.get("output_manifest_sha256") != replay_sha
    or replay_completed.get("stage1_checkpoint_sha256") != checkpoint_sha
):
    fail("promoted replay completion differs")
if (
    replay_receipt.get("schema_name") != "navigation_rlt_replay_promotion"
    or replay_receipt.get("schema_version") != 1
    or replay_receipt.get("status") != "completed"
    or replay_receipt.get("stage1_checkpoint", {}).get("sha256") != checkpoint_sha
    or replay_receipt.get("output", {}).get("manifest_sha256") != replay_sha
):
    fail("promoted replay receipt differs")
if (
    replay.get("schema_name") != "navigation_rlt_replay_manifest"
    or replay.get("schema_version") != 3
    or replay.get("run_id") != run_id
    or replay.get("representation_kind") != "frozen_z_rl"
    or replay.get("expected_episode_count") != 256
    or replay.get("completed_episode_count") != 256
    or replay.get("split_label") != "train_update"
    or replay.get("validation_split_reads") != 0
    or replay.get("behavior_policy_id") != "navigation_rlt_warmstart_lineage_v1"
    or replay.get("behavior_policy_versions") != [100001]
):
    fail("promoted replay train-only contract differs")
training = base.get("training")
if (
    base.get("schema_name") != "navigation_rlt_training_run_manifest"
    or base.get("schema_version") != 1
    or base.get("run_id") != run_id
    or base.get("phase") != "stage2"
    or base.get("required_world_size") != 8
    or not isinstance(training, dict)
    or training.get("epochs") != 10
    or base.get("replay", {}).get("manifest_sha256") != replay_sha
    or base.get("stage1_checkpoint") != {"sha256": checkpoint_sha}
    or base.get("boundaries", {}).get("validation_split_reads") != 0
    or base.get("boundaries", {}).get("lightnav_frozen") is not True
    or base.get("boundaries", {}).get("ttt_enabled") is not False
):
    fail("base Stage2 run manifest differs")
if (
    base_receipt.get("schema_name") != "navigation_rlt_prepare_training"
    or base_receipt.get("schema_version") != 1
    or base_receipt.get("status") != "completed"
    or base_receipt.get("operation") != "stage2"
    or base_receipt.get("run_id") != run_id
    or base_receipt.get("output", {}).get("run_manifest_sha256") != base_sha
    or base_receipt.get("replay", {}).get("manifest_sha256") != replay_sha
    or base_receipt.get("stage1_checkpoint", {}).get("sha256") != checkpoint_sha
    or base_receipt.get("training") != training
):
    fail("base Stage2 prepare receipt differs")
if (
    base_protocol.get("schema_name") != "navigation_rlt_stage2_repair_protocol"
    or base_protocol.get("schema_version") != 1
    or base_protocol.get("run_id") != run_id
    or base_protocol.get("stage1_checkpoint_sha256") != checkpoint_sha
    or base_protocol.get("stage2_replay_manifest_sha256") != replay_sha
    or base_protocol.get("stage2_run_manifest_sha256") != base_sha
    or base_protocol.get("validation_split_reads") != 0
):
    fail("base Stage2 protocol lineage differs")

print(json.dumps({
    "status": "passed",
    "base_manifest": {"path": str(base_path), "sha256": base_sha},
    "base_protocol": {"path": str(base_protocol_path), "sha256": expected_protocol_sha},
    "replay_manifest": {"path": str(replay_path), "sha256": replay_sha},
    "stage1_checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_sha},
    "base_epochs": training["epochs"],
    "validation_split_reads": 0,
}, sort_keys=True, allow_nan=False))
PY
)
[[ "$(json_get "$input_json" status)" == passed ]] || die "smoke input verification did not pass"

if [[ -e "$STORAGE_SAFETY_ROOT" || -L "$STORAGE_SAFETY_ROOT" ]]; then
  [[ -d "$STORAGE_SAFETY_ROOT" && ! -L "$STORAGE_SAFETY_ROOT" ]] || \
    die "storage-safety root is unsafe"
else
  mkdir -- "$STORAGE_SAFETY_ROOT"
fi
attempt_id=$("$PYTHON" -B -c 'import uuid; print(uuid.uuid4())')
[[ "$attempt_id" =~ ^[0-9a-f-]{36}$ ]] || die "could not create a unique Turbo attempt id"
TURBO_RECEIPT="$STORAGE_SAFETY_ROOT/turbo-stage2-anticollapse-smoke-attempt-$attempt_id.json"
[[ ! -e "$TURBO_RECEIPT" && ! -L "$TURBO_RECEIPT" ]] || die "Turbo receipt path already exists"
"$PYTHON" -B "$STORAGE_GUARD" \
  --profile turbo \
  --path "$RUN" \
  --pending-write-bytes "$PENDING_WRITE_BYTES" \
  --status-output "$TURBO_RECEIPT" >/dev/null
"$PYTHON" -B - "$TURBO_RECEIPT" "$RUN" "$PENDING_WRITE_BYTES" "$TURBO_SOURCE" <<'PY'
import json
from pathlib import Path
import sys

path, run, pending, source = sys.argv[1:]
value = json.loads(Path(path).read_text(encoding="utf-8"))
mount = value.get("mount", {})
space = value.get("ordinary_space", {})
if (
    value.get("schema_name") != "navigation_rlt_storage_guard"
    or value.get("schema_version") != 1
    or value.get("status") != "passed"
    or value.get("profile") != "turbo"
    or value.get("path") != run
    or value.get("pending_write_bytes") != int(pending)
    or value.get("blocking_reasons") != []
    or mount.get("mount_point") != "/home/navcloud/work"
    or mount.get("filesystem") not in {"nfs", "nfs4"}
    or mount.get("source") != source
    or mount.get("writable") is not True
    or space.get("projected_available_bytes", -1) < 107_374_182_400
):
    raise SystemExit("fresh Turbo guard receipt differs or is blocked")
PY
TURBO_RECEIPT_SHA256=$(sha256_file "$TURBO_RECEIPT")

if [[ ! -e "$SMOKE_PARENT" && ! -L "$SMOKE_PARENT" ]]; then
  mkdir -- "$SMOKE_PARENT"
fi
mkdir -- "$SMOKE_ROOT" || die "could not atomically create the fixed smoke root"
mkdir -- "$CODE_ROOT"
for name in navigation_rlt_fsdp_train.py navigation_rlt_policy.py \
  navigation_rlt_replay.py lightnav_a3_gpu_monitor.py; do
  install -m 400 -- "$SOURCE_ROOT/$name" "$CODE_ROOT/$name"
done
install -m 500 -- "$SOURCE_ROOT/run_navigation_rlt_a800_stage2_repair.sh" \
  "$CODE_ROOT/run_navigation_rlt_a800_stage2_repair.sh"
for name in "${!SOURCE_HASHES[@]}"; do
  [[ "$(sha256_file "$CODE_ROOT/$name")" == "${SOURCE_HASHES[$name]}" ]] || \
    die "materialized smoke source SHA-256 differs: $name"
done
chmod 500 -- "$CODE_ROOT"
mkdir -- "$COMPAT_LOCK"

CHECKPOINT=$(json_get "$input_json" stage1_checkpoint.path)
CHECKPOINT_SHA256=$(json_get "$input_json" stage1_checkpoint.sha256)
REPLAY=$(json_get "$input_json" replay_manifest.path)
REPLAY_SHA256=$(json_get "$input_json" replay_manifest.sha256)
BASE_SHA256=$(json_get "$input_json" base_manifest.sha256)
PREPARE_ROOT="$SMOKE_ROOT/prepare"

derived_json=$("$PYTHON" -B - \
  "$BASE_MANIFEST" "$BASE_SHA256" "$PREPARE_ROOT" \
  "$TRAINER_SHA256" "$POLICY_SHA256" "$REPLAY_CODE_SHA256" \
  "$ANTI_COLLAPSE_COMMIT" "$TURBO_RECEIPT" "$TURBO_RECEIPT_SHA256" <<'PY'
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

(base_s, expected_base_sha, output_s, trainer_sha, policy_sha, replay_sha,
 commit, guard_s, guard_sha) = sys.argv[1:]
base_path = Path(base_s).resolve(strict=True)
output = Path(output_s)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_new(path, value):
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    if path.read_bytes() != payload:
        raise SystemExit(f"create-only readback differs: {path}")
    return sha(path)

if sha(base_path) != expected_base_sha:
    raise SystemExit("base run manifest SHA-256 differs during derivation")
base = json.loads(base_path.read_text(encoding="utf-8"))
derived = copy.deepcopy(base)
base_training = copy.deepcopy(base["training"])
derived["training"]["epochs"] = 2
derived["implementation"] = {
    "trainer_sha256": trainer_sha,
    "policy_sha256": policy_sha,
    "replay_sha256": replay_sha,
}
if {name: value for name, value in derived["training"].items() if name != "epochs"} != {
    name: value for name, value in base_training.items() if name != "epochs"
}:
    raise SystemExit("a Stage2 training parameter other than epochs changed")
base_other = copy.deepcopy(base)
derived_other = copy.deepcopy(derived)
for value in (base_other, derived_other):
    value.pop("implementation")
    value["training"].pop("epochs")
if base_other != derived_other:
    raise SystemExit("derived manifest changed a field outside epochs/implementation")
if output.exists() or output.is_symlink():
    raise SystemExit("smoke prepare root already exists")
output.mkdir(mode=0o700)
manifest_path = output / "run_manifest.json"
manifest_sha = write_new(manifest_path, derived)
receipt = {
    "schema_name": "navigation_rlt_anticollapse_smoke_prepare",
    "schema_version": 1,
    "status": "completed",
    "anti_collapse_commit": commit,
    "base_run_manifest": {"path": str(base_path), "sha256": expected_base_sha},
    "derived_run_manifest": {"path": str(manifest_path), "sha256": manifest_sha},
    "training_changed_paths": ["training.epochs"],
    "base_epochs": base_training["epochs"],
    "smoke_epochs": 2,
    "implementation_changed_paths": [
        "implementation.policy_sha256",
        "implementation.replay_sha256",
        "implementation.trainer_sha256",
    ],
    "implementation": derived["implementation"],
    "turbo_guard": {"path": str(Path(guard_s).resolve(strict=True)), "sha256": guard_sha},
    "validation_split_reads": 0,
}
receipt_path = output / "DERIVATION.json"
receipt_sha = write_new(receipt_path, receipt)
completion_path = output / "COMPLETED.json"
completion_sha = write_new(completion_path, {
    "schema_name": "navigation_rlt_anticollapse_smoke_prepare",
    "schema_version": 1,
    "status": "completed",
    "run_manifest_sha256": manifest_sha,
    "derivation_sha256": receipt_sha,
    "validation_split_reads": 0,
})
print(json.dumps({
    "status": "completed",
    "run_manifest": {"path": str(manifest_path), "sha256": manifest_sha},
    "derivation": {"path": str(receipt_path), "sha256": receipt_sha},
    "completion": {"path": str(completion_path), "sha256": completion_sha},
}, sort_keys=True, allow_nan=False))
PY
)
DERIVED_MANIFEST=$(json_get "$derived_json" run_manifest.path)
DERIVED_SHA256=$(json_get "$derived_json" run_manifest.sha256)

export NAVRLT_TELEMETRY_INTERVAL_S=5
"$CODE_ROOT/run_navigation_rlt_a800_stage2_repair.sh" \
  "$SMOKE_ROOT" stage2 \
  "$DERIVED_MANIFEST" "$DERIVED_SHA256" \
  "$REPLAY" "$REPLAY_SHA256" \
  "$CHECKPOINT" "$CHECKPOINT_SHA256"

"$PYTHON" -B - \
  "$RUNS_ROOT" "$SMOKE_ROOT" "$BASE_MANIFEST" "$BASE_SHA256" \
  "$DERIVED_MANIFEST" "$DERIVED_SHA256" "$REPLAY_SHA256" \
  "$CHECKPOINT_SHA256" "$TURBO_RECEIPT" "$TURBO_RECEIPT_SHA256" \
  "$CODE_ROOT" "$ANTI_COLLAPSE_COMMIT" <<'PY'
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

(runs_root_s, smoke_root_s, base_s, base_sha, manifest_s, manifest_sha,
 replay_sha, stage1_sha, guard_s, guard_sha, code_root_s, commit) = sys.argv[1:]
runs_root = Path(runs_root_s).resolve(strict=True)
smoke_root = Path(smoke_root_s).resolve(strict=True)
code_root = Path(code_root_s).resolve(strict=True)

def fail(message):
    raise SystemExit(message)

def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def regular(value, name):
    raw = Path(value)
    if raw.is_symlink():
        fail(f"{name} is a symlink")
    try:
        path = raw.resolve(strict=True)
        path.relative_to(runs_root)
    except (OSError, ValueError):
        fail(f"{name} is missing or escapes the runs root")
    if not path.is_file():
        fail(f"{name} is not a regular file")
    return path

def load(value, name):
    path = regular(value, name)
    try:
        data = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda item: fail(f"non-finite constant: {item}"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        fail(f"{name} is not strict JSON: {error}")
    if not isinstance(data, dict):
        fail(f"{name} must contain an object")
    return path, data

def finite_tree(value, name="value"):
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            fail(f"{name} is non-finite")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            finite_tree(item, f"{name}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            finite_tree(item, f"{name}.{key}")
        return
    fail(f"{name} has an unsupported type")

base_path, base = load(base_s, "base run manifest")
manifest_path, manifest = load(manifest_s, "smoke run manifest")
if sha(base_path) != base_sha or sha(manifest_path) != manifest_sha:
    fail("base or smoke run manifest SHA-256 differs")
if manifest["training"].get("epochs") != 2:
    fail("smoke run manifest does not use exactly two epochs")
if {key: value for key, value in manifest["training"].items() if key != "epochs"} != {
    key: value for key, value in base["training"].items() if key != "epochs"
}:
    fail("smoke changed a Stage2 training parameter other than epochs")
base_compare = json.loads(json.dumps(base))
smoke_compare = json.loads(json.dumps(manifest))
for value in (base_compare, smoke_compare):
    value.pop("implementation")
    value["training"].pop("epochs")
if base_compare != smoke_compare:
    fail("smoke manifest changed a field outside epochs/implementation")

stage_root = smoke_root / "navigation_rlt_stage2"
stage_path, stage = load(stage_root / "STAGE.json", "smoke stage marker")
result_path, result = load(stage_root / "training/training_result.json", "smoke training result")
completion_path, completion = load(stage_root / "training/COMPLETED.json", "training completion")
verification_path, verification = load(stage_root / "verification.json", "A800 verification")
telemetry_path, telemetry = load(stage_root / "gpu_telemetry_summary.json", "GPU telemetry summary")
checkpoint_path = regular(stage_root / "training/checkpoints/latest/navigation_rlt_policy.pt", "smoke checkpoint")
checkpoint_sha = sha(checkpoint_path)

if (
    stage.get("schema_name") != "navigation_rlt_a800_stage"
    or stage.get("schema_version") != 1
    or stage.get("status") != "completed"
    or stage.get("exit_code") != 0
    or stage.get("phase") != "stage2"
    or stage.get("run_manifest", {}).get("sha256") != manifest_sha
    or stage.get("replay_manifest", {}).get("sha256") != replay_sha
    or stage.get("stage1_checkpoint", {}).get("sha256") != stage1_sha
    or stage.get("output", {}).get("checkpoint") != {"path": str(checkpoint_path), "sha256": checkpoint_sha}
):
    fail("smoke Stage2 marker differs")
if (
    completion.get("schema_name") != "navigation_rlt_training_completion"
    or completion.get("schema_version") != 1
    or completion.get("status") != "completed"
    or completion.get("phase") != "stage2"
    or completion.get("training_result_sha256") != sha(result_path)
):
    fail("smoke training completion differs")
if (
    result.get("schema_name") != "navigation_rlt_rlinf_fsdp_training"
    or result.get("schema_version") != 1
    or result.get("status") != "completed"
    or result.get("phase") != "stage2"
    or result.get("rlinf_backend") != "FSDPModelManager"
    or result.get("fsdp_wrapper") != "FullyShardedDataParallel"
    or result.get("world_size") != 8
    or result.get("unique_gpu_uuid_count") != 8
    or result.get("training") != manifest["training"]
    or result.get("run_manifest", {}).get("sha256") != manifest_sha
    or result.get("replay_manifest", {}).get("sha256") != replay_sha
    or result.get("stage1_checkpoint_input") != {"path": stage["stage1_checkpoint"]["path"], "sha256": stage1_sha}
):
    fail("smoke training identity/FSDP contract differs")

initialization = result.get("stage2_actor_initialization")
if not isinstance(initialization, dict):
    fail("Stage2 actor initialization evidence is absent")
initial_probabilities = initialization.get("gate_probabilities")
if (
    initialization.get("schema_name") != "navigation_rlt_support_aware_behavior_gate_prior"
    or initialization.get("schema_version") != 1
    or not isinstance(initial_probabilities, list)
    or len(initial_probabilities) != 4
    or any(not math.isfinite(float(value)) or float(value) < 0.0 for value in initial_probabilities)
    or not math.isclose(sum(float(value) for value in initial_probabilities), 1.0, rel_tol=0.0, abs_tol=1e-6)
    or not math.isclose(float(initialization.get("maximum_gate_probability", 1.0)), max(float(value) for value in initial_probabilities), rel_tol=0.0, abs_tol=1e-7)
    or float(initialization.get("maximum_gate_probability", 1.0)) >= 0.98
):
    fail("support-aware Stage2 actor initialization lacks collapse margin")
epochs = result.get("action_probability_epochs")
probability_names = {
    "keep_probability_mean", "adjust_probability_mean",
    "stop_probability_mean", "recover_probability_mean",
}
if not isinstance(epochs, list) or len(epochs) != 2:
    fail("smoke must contain exactly two action-probability epochs")
for index, epoch in enumerate(epochs):
    if not isinstance(epoch, dict) or set(epoch) != probability_names:
        fail(f"action-probability epoch {index} members differ")
    values = [float(epoch[name]) for name in sorted(probability_names)]
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in values):
        fail(f"action-probability epoch {index} is invalid")
    if max(values) >= 0.98:
        fail(f"action-probability epoch {index} crossed the collapse boundary")
if result.get("collapsed_epoch_streak") != 0:
    fail("smoke ended with a non-zero collapse streak")

rank_receipts = result.get("rank_receipts")
steps = result.get("optimizer_steps_per_rank")
if type(steps) is not int or steps < 2 or not isinstance(rank_receipts, list) or len(rank_receipts) != 8:
    fail("smoke rank/optimizer evidence differs")
uuids = set()
for rank, receipt in enumerate(sorted(rank_receipts, key=lambda item: item.get("rank", -1))):
    start = receipt.get("gpu_start", {})
    end = receipt.get("gpu_end", {})
    uuid = start.get("gpu_uuid")
    if (
        receipt.get("rank") != rank
        or receipt.get("local_rank") != rank
        or receipt.get("world_size") != 8
        or receipt.get("gradient_steps") != steps
        or not math.isclose(float(receipt.get("collective_sum", -1)), 28.0, rel_tol=0.0, abs_tol=1e-6)
        or not math.isclose(float(receipt.get("expected_collective_sum", -1)), 28.0, rel_tol=0.0, abs_tol=1e-6)
        or not isinstance(uuid, str)
        or not uuid
        or end.get("gpu_uuid") != uuid
    ):
        fail(f"rank {rank} FSDP/GPU receipt differs")
    uuids.add(uuid)
if len(uuids) != 8:
    fail("smoke does not prove eight unique GPU UUIDs")
if result.get("losses", {}).get("steps") != steps:
    fail("finite loss summary step count differs")
finite_tree(result.get("losses"), "losses")

checkpoint_ref = result.get("final_checkpoint")
restore = result.get("independent_restore")
components = result.get("component_parameter_sha256")
component_names = {"stage1", "actor", "q1", "q2", "target_q1", "target_q2"}
if (
    not isinstance(checkpoint_ref, dict)
    or checkpoint_ref.get("path") != str(checkpoint_path)
    or checkpoint_ref.get("sha256") != checkpoint_sha
    or not isinstance(restore, dict)
    or restore.get("schema_name") != "navigation_rlt_independent_restore"
    or restore.get("schema_version") != 1
    or restore.get("status") != "passed"
    or restore.get("checkpoint_sha256") != checkpoint_sha
    or restore.get("stage1_frozen") is not True
    or restore.get("training_step") != steps
    or not isinstance(components, dict)
    or set(components) != component_names
    or any(re.fullmatch(r"[0-9a-f]{64}", str(value)) is None for value in components.values())
):
    fail("smoke checkpoint/restore/component evidence differs")

sys.path.insert(0, str(code_root))
from navigation_rlt_policy import NavigationRLTPolicy
policy, metadata = NavigationRLTPolicy.load_checkpoint(checkpoint_path, map_location="cpu")
extra = metadata.get("extra")
initial = extra.get("initial_component_sha256") if isinstance(extra, dict) else None
if not isinstance(initial, dict) or set(initial) != component_names:
    fail("checkpoint initial component hashes are absent")
if components["stage1"] != initial["stage1"]:
    fail("smoke changed the frozen Stage1 component")
for name in ("actor", "q1", "q2", "target_q1", "target_q2"):
    if components[name] == initial[name]:
        fail(f"smoke changed no {name} parameter")
    if extra.get(f"{name}_parameter_sha256") != components[name]:
        fail(f"checkpoint {name} hash differs")
if (
    not bool(policy.stage1_frozen.item())
    or metadata.get("training_step") != steps
    or extra.get("training_phase") != "stage2"
    or extra.get("run_manifest_sha256") != manifest_sha
    or extra.get("replay_manifest_sha256") != replay_sha
    or extra.get("stage1_parameter_sha256") != components["stage1"]
):
    fail("restored checkpoint provenance differs")
if (
    verification.get("schema_name") != "navigation_rlt_a800_verification"
    or verification.get("status") != "passed"
    or verification.get("world_size") != 8
    or verification.get("unique_gpu_uuid_count") != 8
    or verification.get("checkpoint") != {"path": str(checkpoint_path), "sha256": checkpoint_sha}
    or telemetry.get("gpu_count") != 8
):
    fail("A800 verification or telemetry differs")
guard_path = regular(guard_s, "Turbo guard receipt")
if sha(guard_path) != guard_sha:
    fail("Turbo guard receipt SHA-256 differs")

completion_path = smoke_root / "SMOKE_COMPLETED.json"
value = {
    "schema_name": "navigation_rlt_stage2_anticollapse_smoke",
    "schema_version": 1,
    "status": "completed",
    "run_id": "navigation-rlt-20261002-v2",
    "smoke_id": "stage2-anticollapse-fe6b16d-2epoch-v1",
    "anti_collapse_commit": commit,
    "epochs": 2,
    "collapse_threshold": 0.98,
    "collapsed_epoch_streak": 0,
    "initial_maximum_gate_probability": initialization["maximum_gate_probability"],
    "action_probability_epochs": epochs,
    "world_size": 8,
    "unique_gpu_uuid_count": 8,
    "fsdp_wrapper": result["fsdp_wrapper"],
    "optimizer_steps_per_rank": steps,
    "component_parameter_sha256": components,
    "base_run_manifest": {"path": str(base_path), "sha256": base_sha},
    "smoke_run_manifest": {"path": str(manifest_path), "sha256": manifest_sha},
    "stage_marker": {"path": str(stage_path), "sha256": sha(stage_path)},
    "training_result": {"path": str(result_path), "sha256": sha(result_path)},
    "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_sha},
    "independent_restore": restore,
    "a800_verification": {"path": str(verification_path), "sha256": sha(verification_path)},
    "telemetry_summary": {"path": str(telemetry_path), "sha256": sha(telemetry_path)},
    "turbo_guard": {"path": str(guard_path), "sha256": guard_sha},
    "validation_split_reads": 0,
}
payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
descriptor = os.open(completion_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
with os.fdopen(descriptor, "wb") as stream:
    stream.write(payload)
    stream.flush()
    os.fsync(stream.fileno())
if completion_path.read_bytes() != payload:
    fail("smoke completion readback differs")
print(json.dumps({
    "status": "completed",
    "smoke_completion": {"path": str(completion_path), "sha256": sha(completion_path)},
    "checkpoint": value["checkpoint"],
    "action_probability_epochs": epochs,
}, sort_keys=True, allow_nan=False))
PY
