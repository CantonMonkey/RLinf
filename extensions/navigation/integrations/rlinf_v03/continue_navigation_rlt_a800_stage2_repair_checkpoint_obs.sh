#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
umask 077

if [[ $# -ne 0 ]]; then
  echo "This run-bound Stage2 repair checkpoint-archive successor accepts no arguments." >&2
  exit 2
fi

RUN=/srv/navigation/runs/navigation-rlt-20261002-v2
RUN_ID=navigation-rlt-20261002-v2
CODE="$RUN/code"
PROVENANCE="$RUN/provenance"
PYTHON=/srv/navigation/envs/rlinf-v03-cu118/bin/python
REPAIR_PROVENANCE="$PROVENANCE/stage2-repair-v2"
SUCCESSOR="$REPAIR_PROVENANCE/continue_navigation_rlt_a800_stage2_repair.sh"
HELPER="$REPAIR_PROVENANCE/navigation_rlt_stage2_repair_bundle.py"
PROTOCOL="$REPAIR_PROVENANCE/stage2_repair_protocol.json"
STORAGE_GUARD="$CODE/navigation_rlt_storage_guard.py"
STORAGE_GUARD_SHA256=2cc42194509297c11d31d475244147372ce6c5ae474a02c36f3932cebfd3df67
ARCHIVER="$CODE/navigation_rlt_checkpoint_obs_archive.py"
ARCHIVER_SHA256=c49ddbb03f83d0ff515a9171ac745b82d3f0d12acf1ad247294213b4baf7119f
STAGE1_MARKER="$RUN/navigation_rlt_stage1/STAGE.json"
STAGE_MARKER="$RUN/navigation_rlt_stage2/STAGE.json"
OBS_MOUNT=/obs
OBS_ROOT=/obs/navigation-example
EXPECTED_OBS_SOURCE='s3fs[/navigation]'
PENDING_WRITE_BYTES=$((16 * 1024 * 1024 * 1024))
NEW_LOCK="$RUN/.launch-navigation-rlt-stage2-repair-v2"

die() {
  echo "$*" >&2
  exit 1
}

sha256_file() {
  sha256sum -- "$1" | cut -d' ' -f1
}

require_immutable() {
  local path=$1
  local expected=$2
  local mode numeric_mode
  [[ "$expected" =~ ^[0-9a-f]{64}$ ]] || die "Expected SHA-256 is not bound: $path"
  [[ -f "$path" && ! -L "$path" ]] || die "Missing or unsafe file: $path"
  mode=$(stat -c '%A' -- "$path") || die "Could not read file mode: $path"
  [[ ${#mode} -eq 10 ]] || die "Unexpected file mode: $path"
  [[ "${mode:2:1}${mode:5:1}${mode:8:1}" == "---" ]] || \
    die "Run-bound file has write bits: $path"
  numeric_mode=$(stat -c '%a' -- "$path") || die "Could not read numeric file mode: $path"
  [[ "$numeric_mode" == "500" ]] || die "Immutable file mode must be exactly 500: $path"
  [[ "$(sha256_file "$path")" == "$expected" ]] || die "SHA-256 mismatch: $path"
}

require_readonly() {
  local path=$1
  local mode
  [[ -f "$path" && ! -L "$path" ]] || die "Missing or unsafe file: $path"
  mode=$(stat -c '%A' -- "$path") || die "Could not read file mode: $path"
  [[ ${#mode} -eq 10 ]] || die "Unexpected file mode: $path"
  [[ "${mode:2:1}${mode:5:1}${mode:8:1}" == "---" ]] || \
    die "Run-bound file has write bits: $path"
}

storage_guard() {
  "$PYTHON" "$STORAGE_GUARD" \
    --profile turbo \
    --path "$RUN" \
    --pending-write-bytes "$PENDING_WRITE_BYTES"
}

verify_obs_mount() {
  [[ -d "$OBS_MOUNT" && ! -L "$OBS_MOUNT" ]] || die "OBS mountpoint is missing or unsafe"
  [[ -d "$OBS_ROOT" && ! -L "$OBS_ROOT" ]] || die "Canonical OBS root is missing or unsafe"
  local mount_record target source filesystem options extra
  mount_record=$(LC_ALL=C findmnt -T "$OBS_ROOT" -n -o TARGET,SOURCE,FSTYPE,OPTIONS) || \
    die "Could not identify the OBS mount"
  [[ -n "$mount_record" && "$mount_record" != *$'\n'* ]] || \
    die "findmnt did not return exactly one OBS mount"
  read -r target source filesystem options extra <<<"$mount_record"
  [[ -z "${extra:-}" ]] || die "Unexpected OBS mount record"
  [[ "$target" == "$OBS_MOUNT" ]] || die "Longest-match OBS mountpoint differs"
  [[ "$source" == "$EXPECTED_OBS_SOURCE" ]] || die "OBS source/bucket differs"
  [[ "$filesystem" == "fuse.s3fs" ]] || die "OBS filesystem is not fuse.s3fs"
  [[ ",$options," == *,rw,* && ",$options," != *,ro,* ]] || \
    die "OBS mount is not read-write"
}

verify_stage1_archive() {
  local checkpoint_sha=$1
  "$PYTHON" "$ARCHIVER" \
    --verify-completed-phase \
    --expected-checkpoint-sha256 "$checkpoint_sha" \
    --run-id "$RUN_ID" \
    --phase stage1 \
    --obs-root "$OBS_ROOT"
}

verified_checkpoint() {
  local phase=$1
  local marker=$2
  "$PYTHON" - "$RUN" "$phase" "$marker" <<'PY'
import hashlib
import json
from pathlib import Path
import re
import sys

run_s, phase, marker_s = sys.argv[1:]
run = Path(run_s).resolve(strict=True)

def fail(message):
    raise SystemExit(message)

def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

def regular(value, name):
    raw = Path(value)
    if raw.is_symlink():
        fail(f"{name} must not be a symlink")
    try:
        path = raw.resolve(strict=True)
        path.relative_to(run)
    except (OSError, ValueError):
        fail(f"{name} is missing or escapes the fixed run")
    if not path.is_file():
        fail(f"{name} must be a regular file")
    return path

def load(path, name):
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda constant: fail(
                f"{name} contains non-finite constant {constant}"
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        fail(f"{name} is not strict JSON: {error}")
    if not isinstance(value, dict):
        fail(f"{name} must contain an object")
    return value

marker_path = regular(marker_s, "stage marker")
marker = load(marker_path, "stage marker")
checkpoint = marker.get("output", {}).get("checkpoint")
verification = marker.get("verification")
if (
    marker.get("schema_name") != "navigation_rlt_a800_stage"
    or marker.get("schema_version") != 1
    or marker.get("status") != "completed"
    or marker.get("exit_code") != 0
    or marker.get("phase") != phase
    or Path(str(marker.get("run_root"))).resolve(strict=True) != run
    or not isinstance(checkpoint, dict)
    or not isinstance(verification, dict)
):
    fail("stage marker is not an exact completed stage")
checkpoint_path = regular(checkpoint.get("path"), "checkpoint")
checkpoint_sha = checkpoint.get("sha256")
if (
    not isinstance(checkpoint_sha, str)
    or re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha) is None
    or digest(checkpoint_path) != checkpoint_sha
):
    fail("checkpoint SHA-256 differs")
verification_path = regular(verification.get("path"), "verification receipt")
if verification.get("sha256") != digest(verification_path):
    fail("verification receipt SHA-256 differs")
receipt = load(verification_path, "verification receipt")
if (
    receipt.get("schema_name") != "navigation_rlt_a800_verification"
    or receipt.get("schema_version") != 1
    or receipt.get("status") != "passed"
    or receipt.get("phase") != phase
    or receipt.get("world_size") != 8
    or receipt.get("unique_gpu_uuid_count") != 8
    or receipt.get("checkpoint")
       != {"path": str(checkpoint_path), "sha256": checkpoint_sha}
):
    fail("eight-rank checkpoint verification differs")
print(checkpoint_path)
print(checkpoint_sha)
PY
}

archive_verified_checkpoint() {
  local checkpoint_path=$1
  local checkpoint_sha=$2
  "$PYTHON" "$ARCHIVER" \
    --checkpoint "$checkpoint_path" \
    --expected-checkpoint-sha256 "$checkpoint_sha" \
    --run-id "$RUN_ID" \
    --phase stage2 \
    --obs-root "$OBS_ROOT"
  printf 'Stage2 checkpoint OBS archive completed. checkpoint_sha256=%s\n' \
    "$checkpoint_sha"
}

require_readonly "$SUCCESSOR"
require_readonly "$HELPER"
[[ -d "$NEW_LOCK" && ! -L "$NEW_LOCK" ]] || \
  die "fixed Stage2 repair v2 launch lock is absent or unsafe"
require_immutable "$STORAGE_GUARD" "$STORAGE_GUARD_SHA256"
require_immutable "$ARCHIVER" "$ARCHIVER_SHA256"
verify_obs_mount
stage1_checkpoint_output=$(verified_checkpoint stage1 "$STAGE1_MARKER")
mapfile -t stage1_checkpoint <<<"$stage1_checkpoint_output"
[[ ${#stage1_checkpoint[@]} -eq 2 ]] || \
  die "Verified Stage1 checkpoint output is incomplete"
verify_stage1_archive "${stage1_checkpoint[1]}"
guard_status=0
storage_guard || guard_status=$?
if (( guard_status != 0 )); then
  checkpoint_output=
  if checkpoint_output=$(verified_checkpoint stage2 "$STAGE_MARKER"); then
    mapfile -t checkpoint <<<"$checkpoint_output"
    [[ ${#checkpoint[@]} -eq 2 ]] || die "Verified Stage2 checkpoint output is incomplete"
    archive_verified_checkpoint "${checkpoint[0]}" "${checkpoint[1]}"
  fi
  exit "$guard_status"
fi

"$SUCCESSOR"
checkpoint_output=$(verified_checkpoint stage2 "$STAGE_MARKER")
mapfile -t checkpoint <<<"$checkpoint_output"
[[ ${#checkpoint[@]} -eq 2 ]] || die "Verified Stage2 checkpoint output is incomplete"
archive_verified_checkpoint "${checkpoint[0]}" "${checkpoint[1]}"

[[ -f "$PROTOCOL" && ! -L "$PROTOCOL" ]] || die "Stage2 repair protocol is absent"
PROTOCOL_SHA256=$(sha256_file "$PROTOCOL")
post_args_json=$("$PYTHON" -B "$HELPER" write-post-args \
  --run-root "$RUN" \
  --run-id "$RUN_ID" \
  --stage2-repair-protocol "$PROTOCOL" \
  --expected-stage2-repair-protocol-sha256 "$PROTOCOL_SHA256")
printf 'Stage2 repair post-training lineage arguments published. receipt=%s\n' \
  "$("$PYTHON" -c 'import json,sys; print(json.loads(sys.argv[1])["path"])' "$post_args_json")"
