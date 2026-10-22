#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
umask 077

if [[ $# -ne 0 ]]; then
  echo "This run-bound A3 training-budget receipt launcher accepts no arguments." >&2
  exit 2
fi

RUN=/srv/navigation/runs/navigation-rlt-20261002-v2
RUN_ID=navigation-rlt-20261002-v2
PYTHON=/srv/navigation/envs/rlinf-v03-cu118/bin/python
A3_RUN_ID=lightnav-a3-20260930-v1
OBS_MOUNT=/obs
OBS_ROOT=/obs/navigation-example
EXPECTED_OBS_SOURCE='s3fs[/navigation]'
FINAL_MANIFEST="$OBS_ROOT/manifests/lightnav-a3/$A3_RUN_ID.final.json"
ARCHIVE_COMPLETION="$OBS_ROOT/results/lightnav-a3/$A3_RUN_ID/COMPLETE"
PROVENANCE_PARENT="$RUN/provenance"
OUTPUT_ROOT="$PROVENANCE_PARENT/a3-training-budget-receipt-v3"
RECEIPT="$OUTPUT_ROOT/A3_TRAINING_BUDGET_RECEIPT.json"
TURBO_RECEIPT="$OUTPUT_ROOT/TURBO_GUARD.json"
COMPLETION="$OUTPUT_ROOT/COMPLETED.json"
STORAGE_GUARD="$RUN/code/navigation_rlt_storage_guard.py"
PENDING_WRITE_BYTES=17179869184
TURBO_SOURCE='navigation.storage.example.invalid:/'

GENERATOR_SHA256=a4b596ed05d6610e7ba859e53f2427108fbf16f9ca746166c66f28d745b929fe
STORAGE_GUARD_SHA256=2cc42194509297c11d31d475244147372ce6c5ae474a02c36f3932cebfd3df67

die() {
  echo "$*" >&2
  exit 1
}

sha256_file() {
  sha256sum -- "$1" | cut -d' ' -f1
}

require_readonly_file() {
  local path=$1
  local mode
  [[ -f "$path" && ! -L "$path" ]] || die "Missing or unsafe file: $path"
  mode=$(stat -c '%A' -- "$path") || die "Could not inspect file mode: $path"
  [[ ${#mode} -eq 10 ]] || die "Unexpected file mode: $path"
  [[ "${mode:2:1}${mode:5:1}${mode:8:1}" == "---" ]] || \
    die "Run-bound file has write bits: $path"
}

[[ -x "$PYTHON" ]] || die "Fixed RLinf Python is absent"
[[ -f "$0" && ! -L "$0" ]] || die "Launcher must be a regular non-symlink file"
SOURCE_ROOT=$(dirname -- "$(realpath -e -- "$0")")
[[ -d "$SOURCE_ROOT" && ! -L "$SOURCE_ROOT" ]] || die "Launcher source root is unsafe"
GENERATOR="$SOURCE_ROOT/lightnav_a3_training_budget_receipt.py"
require_readonly_file "$GENERATOR"
[[ "$(sha256_file "$GENERATOR")" == "$GENERATOR_SHA256" ]] || \
  die "A3 training-budget receipt generator SHA-256 differs"

[[ -d "$RUN" && ! -L "$RUN" ]] || die "Fixed Navigation-RLT run root is unsafe"
[[ "$(realpath -e -- "$RUN")" == "$RUN" ]] || die "Fixed run root resolves elsewhere"
[[ -d "$PROVENANCE_PARENT" && ! -L "$PROVENANCE_PARENT" ]] || \
  die "Fixed provenance parent is unsafe"
[[ ! -e "$OUTPUT_ROOT" && ! -L "$OUTPUT_ROOT" ]] || \
  die "Fixed A3 receipt provenance directory already exists"

require_readonly_file "$STORAGE_GUARD"
[[ "$(stat -c '%a' -- "$STORAGE_GUARD")" == "500" ]] || \
  die "Turbo storage guard mode differs"
[[ "$(sha256_file "$STORAGE_GUARD")" == "$STORAGE_GUARD_SHA256" ]] || \
  die "Turbo storage guard SHA-256 differs"

[[ -d "$OBS_MOUNT" && ! -L "$OBS_MOUNT" ]] || die "OBS mountpoint is unsafe"
[[ -d "$OBS_ROOT" && ! -L "$OBS_ROOT" ]] || die "Canonical OBS root is unsafe"
mount_record=$(LC_ALL=C findmnt -T "$OBS_ROOT" -n -o TARGET,SOURCE,FSTYPE,OPTIONS) || \
  die "Could not identify the live OBS mount"
[[ -n "$mount_record" && "$mount_record" != *$'\n'* ]] || \
  die "findmnt did not return exactly one OBS longest-match mount"
read -r target source filesystem options extra <<<"$mount_record"
[[ -z "${extra:-}" ]] || die "Unexpected OBS mount record"
[[ "$target" == "$OBS_MOUNT" ]] || die "Longest-match OBS mountpoint is not /obs"
[[ "$source" == "$EXPECTED_OBS_SOURCE" ]] || die "OBS source/bucket differs"
[[ "$filesystem" == "fuse.s3fs" ]] || die "OBS filesystem is not fuse.s3fs"
[[ ",$options," == *,rw,* && ",$options," != *,ro,* ]] || \
  die "OBS mount is not read-write"

[[ -f "$FINAL_MANIFEST" && ! -L "$FINAL_MANIFEST" ]] || \
  die "Fixed A3 final manifest is absent or unsafe"
[[ -f "$ARCHIVE_COMPLETION" && ! -L "$ARCHIVE_COMPLETION" ]] || \
  die "Fixed A3 archive completion is absent or unsafe"

# The guard writes nothing: its strict JSON is captured and checked before the
# fixed provenance directory is created.
guard_json=$("$PYTHON" -B "$STORAGE_GUARD" \
  --profile turbo \
  --path "$RUN" \
  --pending-write-bytes "$PENDING_WRITE_BYTES")
guard_canonical=$("$PYTHON" -B - \
  "$guard_json" "$RUN" "$PENDING_WRITE_BYTES" "$TURBO_SOURCE" <<'PY'
import json
import sys


def reject_constant(value):
    raise SystemExit(f"non-finite Turbo guard constant: {value}")


value = json.loads(sys.argv[1], parse_constant=reject_constant)
run, pending, source = sys.argv[2:]
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
print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
PY
)

mkdir -- "$OUTPUT_ROOT" || die "Could not atomically create fixed A3 provenance directory"

turbo_sha=$("$PYTHON" -B - "$TURBO_RECEIPT" "$guard_canonical" <<'PY'
import hashlib
import os
from pathlib import Path
import sys


path = Path(sys.argv[1])
payload = (sys.argv[2] + "\n").encode("utf-8")
flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
if hasattr(os, "O_NOFOLLOW"):
    flags |= os.O_NOFOLLOW
descriptor = os.open(path, flags, 0o400)
with os.fdopen(descriptor, "wb") as stream:
    stream.write(payload)
    stream.flush()
    os.fsync(stream.fileno())
if path.read_bytes() != payload:
    raise SystemExit("create-only Turbo guard readback differs")
print(hashlib.sha256(payload).hexdigest())
PY
)

# This is the only archive reader.  Its bound production contract rehashes all
# 130 archived files and recomputes replay eligibility and 8-rank A3 training
# evidence before it creates the budget receipt outside OBS. Diagnostic progress
# and trace payloads were not archived; the completion records this coverage.
generator_json=$("$PYTHON" -B "$GENERATOR" \
  --obs-root "$OBS_ROOT" \
  --final-manifest "$FINAL_MANIFEST" \
  --completion "$ARCHIVE_COMPLETION" \
  --output "$RECEIPT")
receipt_sha=$(sha256_file "$RECEIPT")

completion_json=$("$PYTHON" -B - \
  "$RUN" "$RUN_ID" "$A3_RUN_ID" "$OBS_ROOT" "$OUTPUT_ROOT" \
  "$RECEIPT" "$receipt_sha" "$TURBO_RECEIPT" "$turbo_sha" \
  "$COMPLETION" "$EXPECTED_OBS_SOURCE" "$generator_json" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys


(run_s, run_id, a3_run_id, obs_s, output_s, receipt_s, expected_receipt_sha,
 guard_s, expected_guard_sha, completion_s, obs_source, generator_s) = sys.argv[1:]


def fail(message):
    raise SystemExit(message)


def reject_constant(value):
    fail(f"non-finite JSON constant: {value}")


def sha256(path):
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
        resolved = raw.resolve(strict=True)
    except OSError:
        fail(f"{name} is missing")
    if not resolved.is_file():
        fail(f"{name} is not a regular file")
    return resolved


run = Path(run_s).resolve(strict=True)
obs = Path(obs_s).resolve(strict=True)
output = Path(output_s).resolve(strict=True)
if str(run) != run_s or output != run / "provenance" / "a3-training-budget-receipt-v3":
    fail("fixed Navigation-RLT provenance path differs")
receipt_path = regular(receipt_s, "A3 training-budget receipt")
guard_path = regular(guard_s, "Turbo guard receipt")
if receipt_path.parent != output or guard_path.parent != output:
    fail("receipt artifacts escape the fixed provenance directory")
for path in (receipt_path, guard_path, Path(completion_s).absolute()):
    try:
        path.relative_to(obs)
    except ValueError:
        pass
    else:
        fail("a receipt artifact resolves inside OBS")
if sha256(receipt_path) != expected_receipt_sha:
    fail("A3 training-budget receipt SHA-256 differs")
if sha256(guard_path) != expected_guard_sha:
    fail("Turbo guard receipt SHA-256 differs")
receipt = json.loads(
    receipt_path.read_text(encoding="utf-8"), parse_constant=reject_constant
)
scope = receipt.get("scope", {})
if (
    receipt.get("schema_name") != "navigation_rlt_a3_training_budget_receipt"
    or receipt.get("schema_version") != 1
    or receipt.get("status") != "verified"
    or receipt.get("run_id") != a3_run_id
    or scope.get("upstream_split_read") != "train"
    or scope.get("validation_split_reads") != 0
    or scope.get("train_only") is not True
):
    fail("A3 training-budget receipt contract differs")
generated = json.loads(generator_s, parse_constant=reject_constant)
if (generated.get("status") != "verified" or generated.get("path") != str(receipt_path)
        or generated.get("sha256") != expected_receipt_sha):
    fail("generator output and budget receipt differ")
coverage = generated.get("coverage")
if (not isinstance(coverage, dict)
        or coverage.get("schema_name") != "lightnav_a3_rollout_inventory_coverage"
        or coverage.get("schema_version") != 1
        or coverage.get("rollout_parent") != "input/train_rollout_receipt"
        or coverage.get("full_rollout_inventory_readback") is not False):
    fail("rollout inventory coverage differs")
budget_entries = coverage.get("budget_required_entries", {})
diagnostics = coverage.get("unarchived_diagnostic_entries", {})
if (set(budget_entries) != {"actor_decisions.jsonl", "result.json"}
        or set(diagnostics) != {"episode_progress.jsonl", "execution_trace.jsonl"}):
    fail("budget or diagnostic coverage entries differ")
expected_paths = {
    "actor_decisions.jsonl": "input/actor_decisions.jsonl",
    "result.json": "input/train_rollout_receipt/result.json",
}
for name, expected_path in expected_paths.items():
    entry = budget_entries[name]
    if (entry.get("archive_relative") != expected_path
            or entry.get("payload_readback") is not True):
        fail("budget payload readback coverage differs")
if budget_entries["actor_decisions.jsonl"].get("sha256") != receipt["provenance"]["replay_sha256"]:
    fail("coverage actor replay differs from budget receipt")
sealed = coverage.get("sealed_inventory", {})
if (sealed.get("archive_relative") != "input/train_rollout_receipt/artifact_inventory.json"
        or sealed.get("payload_readback") is not True):
    fail("sealed inventory coverage differs")
if any(entry.get("payload_readback") is not False for entry in diagnostics.values()):
    fail("unarchived diagnostic payloads must not claim readback")
completion = {
    "schema_name": "navigation_rlt_a3_training_budget_receipt_completion",
    "schema_version": 1,
    "status": "completed",
    "run_id": run_id,
    "source_run_id": a3_run_id,
    "receipt": {
        "path": str(receipt_path),
        "sha256": expected_receipt_sha,
        "schema_name": receipt["schema_name"],
        "schema_version": receipt["schema_version"],
        "status": receipt["status"],
        "run_id": receipt["run_id"],
    },
    "turbo_guard": {
        "path": str(guard_path),
        "sha256": expected_guard_sha,
        "status": "passed",
    },
    "obs_mount": {
        "mount_point": "/obs",
        "source": obs_source,
        "filesystem": "fuse.s3fs",
        "writable": True,
    },
    "validation_split_reads": 0,
    "rollout_inventory_coverage": coverage,
}
payload = (json.dumps(completion, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
completion_path = Path(completion_s)
if completion_path.parent.resolve(strict=True) != output:
    fail("completion path escapes the fixed provenance directory")
flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
if hasattr(os, "O_NOFOLLOW"):
    flags |= os.O_NOFOLLOW
descriptor = os.open(completion_path, flags, 0o400)
with os.fdopen(descriptor, "wb") as stream:
    stream.write(payload)
    stream.flush()
    os.fsync(stream.fileno())
if completion_path.read_bytes() != payload:
    fail("create-only completion readback differs")
print(json.dumps({
    "status": "completed",
    "receipt": {"path": str(receipt_path), "sha256": expected_receipt_sha},
    "completion": {
        "path": str(completion_path.resolve(strict=True)),
        "sha256": hashlib.sha256(payload).hexdigest(),
    },
    "validation_split_reads": 0,
}, sort_keys=True, allow_nan=False))
PY
)

printf '%s\n' "$completion_json"
