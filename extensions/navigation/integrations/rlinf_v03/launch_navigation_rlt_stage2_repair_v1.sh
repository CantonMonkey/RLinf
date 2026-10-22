#!/usr/bin/env bash
set -euo pipefail
umask 077

if [[ $# -ne 0 ]]; then
  echo "This run-bound Stage2 repair launcher accepts no arguments." >&2
  exit 2
fi

RUN=/srv/navigation/runs/navigation-rlt-20261002-v2
RUN_ID=navigation-rlt-20261002-v2
PYTHON=/srv/navigation/envs/rlinf-v03-cu118/bin/python
REPAIR_PROVENANCE="$RUN/provenance/stage2-repair-v1"
CANONICAL_LAUNCHER="$REPAIR_PROVENANCE/launch_navigation_rlt_stage2_repair_v1.sh"
HELPER="$REPAIR_PROVENANCE/navigation_rlt_stage2_repair_bundle.py"
CONTEXT="$REPAIR_PROVENANCE/bundle_context.json"
SAFE_WRAPPER="$REPAIR_PROVENANCE/continue_navigation_rlt_a800_stage2_repair_checkpoint_obs.sh"
SUCCESSOR="$REPAIR_PROVENANCE/continue_navigation_rlt_a800_stage2_repair.sh"
STORAGE_GUARD="$RUN/code/navigation_rlt_storage_guard.py"
STORAGE_GUARD_SHA256=2cc42194509297c11d31d475244147372ce6c5ae474a02c36f3932cebfd3df67
ARCHIVER="$RUN/code/navigation_rlt_checkpoint_obs_archive.py"
ARCHIVER_SHA256=c49ddbb03f83d0ff515a9171ac745b82d3f0d12acf1ad247294213b4baf7119f
NEW_LOCK="$RUN/.launch-navigation-rlt-stage2-repair-v1"
OLD_LOCK="$RUN/.launch-navigation-rlt-stage2"
STORAGE_SAFETY_ROOT="$RUN/storage-safety"
PENDING_WRITE_BYTES=$((16 * 1024 * 1024 * 1024))

die() {
  echo "$*" >&2
  exit 1
}

# Reject copied launchers before taking the admission flock, running bundle
# admission, writing a storage-guard receipt, or creating the fixed run lock.
# The canonical file is subsequently authenticated by verify-admission through
# the create-only launcher inventory.
SELF_REALPATH=$(readlink -f -- "$0") || die "could not resolve Stage2 repair launcher path"
[[ "$SELF_REALPATH" == "$CANONICAL_LAUNCHER" ]] || \
  die "Stage2 repair launcher must run from its canonical inventory path"

sha256_file() {
  sha256sum -- "$1" | cut -d' ' -f1
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

require_immutable() {
  local path=$1
  local expected=$2
  local numeric_mode
  require_readonly "$path"
  [[ "$expected" =~ ^[0-9a-f]{64}$ ]] || die "Expected SHA-256 is not bound: $path"
  numeric_mode=$(stat -c '%a' -- "$path") || die "Could not read numeric file mode: $path"
  [[ "$numeric_mode" == "500" ]] || die "Immutable file mode must be exactly 500: $path"
  [[ "$(sha256_file "$path")" == "$expected" ]] || die "SHA-256 mismatch: $path"
}

json_get_value() {
  "$PYTHON" - "$1" "$2" <<'PY'
import json
import sys

value = json.loads(sys.argv[1])
for part in sys.argv[2].split('.'):
    if not isinstance(value, dict) or part not in value:
        raise SystemExit(f"missing JSON field: {sys.argv[2]}")
    value = value[part]
if isinstance(value, (dict, list)) or value is None:
    raise SystemExit(f"JSON field is not scalar: {sys.argv[2]}")
print(value)
PY
}

command -v flock >/dev/null 2>&1 || die "flock is required for single-launch admission"
exec 9<"$0"
flock -n 9 || die "another Stage2 repair launcher holds the admission lock"

for source in "$HELPER" "$CONTEXT" "$SAFE_WRAPPER" "$SUCCESSOR"; do
  require_readonly "$source"
done
require_immutable "$STORAGE_GUARD" "$STORAGE_GUARD_SHA256"
require_immutable "$ARCHIVER" "$ARCHIVER_SHA256"
[[ ! -e "$OLD_LOCK" && ! -L "$OLD_LOCK" ]] || die "legacy Stage2 lock exists; repair launch is forbidden"
[[ ! -e "$NEW_LOCK" && ! -L "$NEW_LOCK" ]] || die "fixed Stage2 repair lock already exists"
[[ ! -e "$RUN/navigation_rlt_stage2" && ! -L "$RUN/navigation_rlt_stage2" ]] || \
  die "canonical Stage2 output already exists before repair admission"

# This read-only command verifies the exact bundle context and its launcher
# inventory, all four provenance files, current Stage1 lineage, and every
# pre-existing promotion/prepare/protocol artifact before any lock is created.
prelock_json=$("$PYTHON" -B "$HELPER" verify-admission \
  --run-root "$RUN" --run-id "$RUN_ID")
stage1_checkpoint_sha=$(json_get_value "$prelock_json" stage1.stage1_checkpoint.sha256)
CODE_VERIFICATION=$(json_get_value "$prelock_json" repair_code.code_verification.path)
CODE_VERIFICATION_SHA=$(json_get_value "$prelock_json" repair_code.code_verification.sha256)

if [[ -e "$STORAGE_SAFETY_ROOT" || -L "$STORAGE_SAFETY_ROOT" ]]; then
  [[ -d "$STORAGE_SAFETY_ROOT" && ! -L "$STORAGE_SAFETY_ROOT" ]] || \
    die "storage-safety root is unsafe"
else
  mkdir -- "$STORAGE_SAFETY_ROOT"
fi
attempt_id=$("$PYTHON" -c 'import uuid; print(uuid.uuid4())')
[[ "$attempt_id" =~ ^[0-9a-f-]{36}$ ]] || die "could not create a unique Turbo attempt id"
TURBO_RECEIPT="$STORAGE_SAFETY_ROOT/turbo-stage2-repair-v1-attempt-$attempt_id.json"
[[ ! -e "$TURBO_RECEIPT" && ! -L "$TURBO_RECEIPT" ]] || \
  die "unique Stage2 repair Turbo receipt path already exists"

"$PYTHON" "$STORAGE_GUARD" \
  --profile turbo \
  --path "$RUN" \
  --pending-write-bytes "$PENDING_WRITE_BYTES" \
  --status-output "$TURBO_RECEIPT" >/dev/null
guard_sha=$(sha256_file "$TURBO_RECEIPT")
"$PYTHON" - "$TURBO_RECEIPT" "$RUN" "$PENDING_WRITE_BYTES" <<'PY'
import json
from pathlib import Path
import sys

path, run, pending = sys.argv[1:]
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
    or mount.get("writable") is not True
    or space.get("projected_available_bytes", -1) < 107_374_182_400
):
    raise SystemExit("fresh Turbo guard receipt differs or is blocked")
PY

mkdir -- "$NEW_LOCK" || die "could not atomically create fixed Stage2 repair lock"
admission="$NEW_LOCK/ADMISSION.json"
"$PYTHON" - \
  "$admission" "$stage1_checkpoint_sha" \
  "$CODE_VERIFICATION" "$CODE_VERIFICATION_SHA" \
  "$HELPER" "$SAFE_WRAPPER" "$SUCCESSOR" \
  "$TURBO_RECEIPT" "$guard_sha" "$prelock_json" <<'PY'
import datetime as dt
import json
import os
from pathlib import Path
import sys

(output, checkpoint_sha, verification, verification_sha, helper,
 safe_wrapper, successor, guard, guard_sha, prelock_json) = sys.argv[1:]

def sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

value = {
    "schema_name": "navigation_rlt_stage2_repair_launch_admission",
    "schema_version": 1,
    "status": "admitted",
    "run_id": "navigation-rlt-20261002-v2",
    "admitted_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "stage1_checkpoint_sha256": checkpoint_sha,
    "stage2_repair_code_verification": {
        "path": verification,
        "sha256": verification_sha,
    },
    "helper": {"path": helper, "sha256": sha(helper)},
    "safe_wrapper": {"path": safe_wrapper, "sha256": sha(safe_wrapper)},
    "successor": {"path": successor, "sha256": sha(successor)},
    "turbo_guard": {"path": guard, "sha256": guard_sha, "status": "passed"},
    "prelock_verification": json.loads(prelock_json),
    "legacy_stage2_lock_forbidden": True,
    "validation_split_reads": 0,
}
payload = (json.dumps(value, sort_keys=True) + "\n").encode("utf-8")
fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
with os.fdopen(fd, "wb") as stream:
    stream.write(payload)
    stream.flush()
    os.fsync(stream.fileno())
if Path(output).read_bytes() != payload:
    raise SystemExit("Stage2 repair admission readback differs")
PY
admission_sha=$(sha256_file "$admission")

stdout_log="$NEW_LOCK/stdout.log"
stderr_log="$NEW_LOCK/stderr.log"
pid_file="$NEW_LOCK/PID"
started="$NEW_LOCK/STARTED.json"
for path in "$stdout_log" "$stderr_log" "$pid_file" "$started"; do
  [[ ! -e "$path" && ! -L "$path" ]] || die "launch artifact already exists: $path"
done

nohup "$PYTHON" - \
  "$pid_file" "$started" "$admission" "$admission_sha" "$SAFE_WRAPPER" \
  >"$stdout_log" 2>"$stderr_log" <<'PY' &
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys

pid_file, started, admission, admission_sha, wrapper = sys.argv[1:]
pid = os.getpid()
value = {
    "schema_name": "navigation_rlt_stage2_repair_launch",
    "schema_version": 1,
    "status": "started",
    "run_id": "navigation-rlt-20261002-v2",
    "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "pid": pid,
    "wrapper": {
        "path": wrapper,
        "sha256": hashlib.sha256(Path(wrapper).read_bytes()).hexdigest(),
    },
    "admission": {"path": admission, "sha256": admission_sha},
}

def write_exclusive(path, payload):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    if Path(path).read_bytes() != payload:
        raise SystemExit(f"launch receipt readback differs: {path}")

write_exclusive(pid_file, f"{pid}\n".encode("ascii"))
write_exclusive(started, (json.dumps(value, sort_keys=True) + "\n").encode("utf-8"))
directory_fd = os.open(str(Path(started).parent), os.O_RDONLY | os.O_DIRECTORY)
try:
    os.fsync(directory_fd)
finally:
    os.close(directory_fd)
os.execv(wrapper, [wrapper])
PY
pid=$!
printf '{"status":"launch_dispatched","lock":"%s","pid":%s,"admission_sha256":"%s"}\n' \
  "$NEW_LOCK" "$pid" "$admission_sha"
