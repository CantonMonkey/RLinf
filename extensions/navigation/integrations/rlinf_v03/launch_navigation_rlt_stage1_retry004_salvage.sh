#!/usr/bin/env bash
set -euo pipefail
umask 077

if [[ $# -ne 0 ]]; then
  echo "This run-bound retry004 salvage launcher accepts no arguments." >&2
  exit 2
fi

RUN=/srv/navigation/runs/navigation-rlt-20261002-v2
RUN_ID=navigation-rlt-20261002-v2
CODE="$RUN/code"
PROVENANCE="$RUN/provenance"
PYTHON=/srv/navigation/envs/rlinf-v03-cu118/bin/python
INPUTS="$PROVENANCE/retry004-salvage-inputs"
STATE="$INPUTS/navigation_rlt_transfer_state_retry004.json"
FAILED_EXIT="$INPUTS/navigation_rlt_transfer_exit_retry004.json"
STAGE="$RUN/.transfer-stage-20261002T122413Z-retry004-844f270c/transfer-prefix"
FINAL="$RUN/transfer-prefix"
LOCK="$RUN/.launch-navigation-rlt-stage1-retry004"
TURBO_RECEIPT="$RUN/storage-safety/turbo-salvage-retry004.json"

FINALIZER="$PROVENANCE/finalize_transfer_prefix_retry004.py"
STORAGE_GUARD="$CODE/navigation_rlt_storage_guard.py"
ARCHIVER="$CODE/navigation_rlt_checkpoint_obs_archive.py"
SAFE_WRAPPER="$PROVENANCE/continue_navigation_rlt_a800_stage1_retry004_checkpoint_obs.sh"
SUCCESSOR="$PROVENANCE/continue_navigation_rlt_a800_stage1_retry004.sh"

STATE_SHA256=6da3049874731ba48a5bcab2b86328973e3c440692e44f3cf72c8e2c2a935405
FAILED_EXIT_SHA256=d78dd9986729eaa1c986061036ebe0bd3c26fdd2822864b5f311d1b59d36b872
FINALIZER_SHA256=bb2e0734b0970989636ae53280b38ef49cb7829b7fc4b40b61810ff3dd88a7b3
STORAGE_GUARD_SHA256=2cc42194509297c11d31d475244147372ce6c5ae474a02c36f3932cebfd3df67
ARCHIVER_SHA256=c49ddbb03f83d0ff515a9171ac745b82d3f0d12acf1ad247294213b4baf7119f
SAFE_WRAPPER_SHA256=5d4be2a87f4895d8e9661397e22d39f11376e1d261cd1d157c71a14bf06d85fb
SUCCESSOR_SHA256=3b136ee6f2b3b38d7dab0f9aa080bf8051cae87ff5c08def2cf3b9bf3ebe9970
PENDING_WRITE_BYTES=$((16 * 1024 * 1024 * 1024))

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
  [[ -f "$path" && ! -L "$path" ]] || die "Missing or unsafe file: $path"
  [[ "$(sha256_file "$path")" == "$expected" ]] || die "SHA-256 mismatch: $path"
}

require_mode_500() {
  local path=$1
  [[ "$(stat -c '%a' -- "$path")" == 500 ]] || die "Expected mode 500: $path"
}

command -v flock >/dev/null 2>&1 || die "flock is required for single-launch admission"
exec 9<"$0"
flock -n 9 || die "another retry004 salvage launcher holds the admission lock"

require_sha "$STATE" "$STATE_SHA256"
require_sha "$FAILED_EXIT" "$FAILED_EXIT_SHA256"
require_sha "$FINALIZER" "$FINALIZER_SHA256"
require_sha "$STORAGE_GUARD" "$STORAGE_GUARD_SHA256"
require_sha "$ARCHIVER" "$ARCHIVER_SHA256"
require_sha "$SAFE_WRAPPER" "$SAFE_WRAPPER_SHA256"
require_sha "$SUCCESSOR" "$SUCCESSOR_SHA256"
python_real=$(readlink -f -- "$PYTHON") || die "fixed Python runtime cannot be resolved"
[[ -f "$python_real" && -x "$python_real" ]] || die "fixed Python runtime is missing or unsafe"
require_mode_500 "$STORAGE_GUARD"
require_mode_500 "$ARCHIVER"
require_mode_500 "$SAFE_WRAPPER"
require_mode_500 "$SUCCESSOR"

"$PYTHON" - "$STATE" "$FAILED_EXIT" "$STAGE" "$RUN" "$FINALIZER" <<'PY'
import json
from pathlib import Path
import sys

state_path, exit_path, stage, run, finalizer = sys.argv[1:]

def load(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )

state = load(state_path)
failed = load(exit_path)
stage_parent = str(Path(stage).parent)
expected_state = {
    "schema_name": "navigation_rlt_transfer_state",
    "attempt": "retry004",
    "status": "launched",
    "destination_root": run,
    "stage_parent": stage_parent,
    "stage_payload": stage,
    "finalizer_remote": finalizer,
    "finalizer_sha256": "bb2e0734b0970989636ae53280b38ef49cb7829b7fc4b40b61810ff3dd88a7b3",
    "finalizer_executed": False,
    "expected_file_count": 260,
    "expected_directory_bytes": 28_596_789_742,
    "expected_payload_bytes": 28_596_553_632,
    "expected_shard_count": 256,
    "expected_inventory_sha256": "9bb4843ae6986332ba9603dd780f887015506c7b67b654ee8d4e0a8e474162cf",
    "expected_marker_sha256": "9d7a68d837b5202b66811a730664b8e0c5c77bbe8a6c3d01891cec51673c79b8",
    "expected_manifest_sha256": "1e0f0a109bbc89029445ce7d42d495004b696ddab62825181bca3cfced377ab2",
}
for key, value in expected_state.items():
    if state.get(key) != value:
        raise SystemExit(f"retry004 state differs at {key}")
expected_exit = {
    "schema_name": "navigation_rlt_detached_transfer_exit",
    "attempt": "retry004",
    "stage_parent": stage_parent,
    "stage_payload": stage,
    "wrapper_pid": 21172,
    "scp_pid": 68584,
    "exit_code": None,
    "succeeded": False,
}
for key, value in expected_exit.items():
    if failed.get(key) != value:
        raise SystemExit(f"retry004 failed EXIT differs at {key}")
if "launch_error" in failed:
    raise SystemExit("retry004 failed EXIT unexpectedly contains launch_error")
PY

[[ -d "$STAGE" && ! -L "$STAGE" ]] || die "retry004 staging is missing or unsafe"
"$PYTHON" - "$STAGE" <<'PY'
from pathlib import Path
import sys

needle = str(Path(sys.argv[1]).parent).encode()
for entry in Path("/proc").iterdir():
    if not entry.name.isdigit():
        continue
    try:
        arguments = (entry / "cmdline").read_bytes().split(b"\0")
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    if not arguments or not arguments[0]:
        continue
    executable = Path(arguments[0].decode(errors="replace")).name
    if executable == "scp" and needle in b" ".join(arguments):
        raise SystemExit(f"active scp still references retry004 staging: pid={entry.name}")
PY
[[ ! -e "$FINAL" && ! -L "$FINAL" ]] || die "final transfer target already exists"
[[ ! -e "$LOCK" && ! -L "$LOCK" ]] || die "fixed Stage1 lock already exists"
for path in \
  "$RUN/navigation_rlt_prefix_relocated" \
  "$RUN/navigation_rlt_stage1_split" \
  "$RUN/navigation_rlt_stage1_prepare" \
  "$RUN/navigation_rlt_stage1" \
  "$RUN/navigation_rlt_stage1_gate"
do
  [[ ! -e "$path" && ! -L "$path" ]] || die "Stage1 successor state already exists: $path"
done
[[ ! -e "$TURBO_RECEIPT" && ! -L "$TURBO_RECEIPT" ]] || \
  die "salvage Turbo receipt already exists"

verify_json=$("$PYTHON" -B "$FINALIZER" --verify-only)
"$PYTHON" - "$verify_json" "$STAGE" <<'PY'
import json
import sys

value = json.loads(sys.argv[1])
expected = {
    "status": "verified",
    "stage": sys.argv[2],
    "file_count": 260,
    "directory_bytes": 28_596_789_742,
    "payload_bytes": 28_596_553_632,
    "inventory_sha256": "9bb4843ae6986332ba9603dd780f887015506c7b67b654ee8d4e0a8e474162cf",
    "marker_sha256": "9d7a68d837b5202b66811a730664b8e0c5c77bbe8a6c3d01891cec51673c79b8",
    "manifest_sha256": "1e0f0a109bbc89029445ce7d42d495004b696ddab62825181bca3cfced377ab2",
}
if value != expected:
    raise SystemExit("retry004 verify-only summary differs")
PY
verify_stdout_sha256=$(printf '%s\n' "$verify_json" | sha256sum | cut -d' ' -f1)

"$PYTHON" "$STORAGE_GUARD" \
  --profile turbo \
  --path "$RUN" \
  --pending-write-bytes "$PENDING_WRITE_BYTES" \
  --status-output "$TURBO_RECEIPT" >/dev/null
guard_sha256=$(sha256_file "$TURBO_RECEIPT")
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
    or mount.get("source") != "navigation.storage.example.invalid:/"
    or mount.get("writable") is not True
    or space.get("projected_available_bytes", -1) < 107_374_182_400
):
    raise SystemExit("fresh Turbo guard receipt differs or is blocked")
PY

mkdir -- "$LOCK" || die "could not atomically create fixed Stage1 lock"
admission="$LOCK/SALVAGE_ADMISSION.json"
"$PYTHON" - \
  "$admission" "$verify_json" "$verify_stdout_sha256" \
  "$TURBO_RECEIPT" "$guard_sha256" <<'PY'
import datetime as dt
import json
import os
from pathlib import Path
import sys

output, verify_raw, verify_sha, guard_path, guard_sha = sys.argv[1:]
value = {
    "schema_name": "navigation_rlt_retry004_salvage_admission",
    "schema_version": 1,
    "status": "admitted",
    "run_id": "navigation-rlt-20261002-v2",
    "attempt": "retry004",
    "authorized_by": "user",
    "authorized_at_thread_resume": True,
    "admitted_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "standard_exit_gate_passed": False,
    "salvage_gate_passed": True,
    "transport_status": "indeterminate",
    "validation_split_reads": 0,
    "state": {
        "sha256": "6da3049874731ba48a5bcab2b86328973e3c440692e44f3cf72c8e2c2a935405"
    },
    "failed_exit": {
        "sha256": "d78dd9986729eaa1c986061036ebe0bd3c26fdd2822864b5f311d1b59d36b872",
        "exit_code": None,
        "succeeded": False,
    },
    "stage": "/srv/navigation/runs/navigation-rlt-20261002-v2/.transfer-stage-20261002T122413Z-retry004-844f270c/transfer-prefix",
    "source": "/srv/navigation/runs/navigation-rlt-20261002-v2/transfer-prefix",
    "source_ready_sha256": "23834c58371cb584d3fdabcc58f19ae9f3f210068d0b3423cbcd8e8ed06f0142",
    "finalizer": {
        "path": "/srv/navigation/runs/navigation-rlt-20261002-v2/provenance/finalize_transfer_prefix_retry004.py",
        "sha256": "bb2e0734b0970989636ae53280b38ef49cb7829b7fc4b40b61810ff3dd88a7b3",
    },
    "safe_wrapper": {
        "path": "/srv/navigation/runs/navigation-rlt-20261002-v2/provenance/continue_navigation_rlt_a800_stage1_retry004_checkpoint_obs.sh",
        "sha256": "5d4be2a87f4895d8e9661397e22d39f11376e1d261cd1d157c71a14bf06d85fb",
    },
    "verify_only": {
        "exit_code": 0,
        "stdout_sha256": verify_sha,
        "summary": json.loads(verify_raw),
    },
    "turbo_guard": {
        "path": guard_path,
        "sha256": guard_sha,
        "status": "passed",
    },
}
payload = (json.dumps(value, sort_keys=True) + "\n").encode("utf-8")
fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
with os.fdopen(fd, "wb") as stream:
    stream.write(payload)
    stream.flush()
    os.fsync(stream.fileno())
if Path(output).read_bytes() != payload:
    raise SystemExit("salvage admission readback differs")
PY
admission_sha256=$(sha256_file "$admission")

stdout_log="$LOCK/stdout.log"
stderr_log="$LOCK/stderr.log"
pid_file="$LOCK/PID"
started="$LOCK/STARTED.json"
for path in "$stdout_log" "$stderr_log" "$pid_file" "$started"; do
  [[ ! -e "$path" && ! -L "$path" ]] || die "launch artifact already exists: $path"
done

nohup "$PYTHON" - \
  "$pid_file" "$started" "$admission" "$admission_sha256" "$SAFE_WRAPPER" \
  >"$stdout_log" 2>"$stderr_log" <<'PY' &
import datetime as dt
import json
import os
from pathlib import Path
import sys

pid_file, started, admission, admission_sha, wrapper = sys.argv[1:]
pid = os.getpid()
value = {
    "schema_name": "navigation_rlt_stage1_salvage_launch",
    "schema_version": 1,
    "status": "started",
    "run_id": "navigation-rlt-20261002-v2",
    "attempt": "retry004",
    "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "pid": pid,
    "wrapper": {
        "path": wrapper,
        "sha256": "5d4be2a87f4895d8e9661397e22d39f11376e1d261cd1d157c71a14bf06d85fb",
    },
    "salvage_admission": {"path": admission, "sha256": admission_sha},
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
write_exclusive(
    started,
    (json.dumps(value, sort_keys=True) + "\n").encode("utf-8"),
)
directory_fd = os.open(str(Path(started).parent), os.O_RDONLY | os.O_DIRECTORY)
try:
    os.fsync(directory_fd)
finally:
    os.close(directory_fd)
os.execv(wrapper, [wrapper])
PY
pid=$!
"$PYTHON" - "$LOCK" "$pid" "$admission_sha256" <<'PY'
import json
import sys

lock, pid, admission_sha = sys.argv[1:]
print(json.dumps({
    "status": "launch_gate_dispatched",
    "lock": lock,
    "gate_pid": int(pid),
    "salvage_admission_sha256": admission_sha,
}, sort_keys=True))
PY
