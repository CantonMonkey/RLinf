#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
umask 077

if [[ $# -ne 0 ]]; then
  echo "This run-bound Stage2 repair successor accepts no arguments." >&2
  exit 2
fi

RUN=/srv/navigation/runs/navigation-rlt-20261002-v2
RUN_ID=navigation-rlt-20261002-v2
PYTHON=/srv/navigation/envs/rlinf-v03-cu118/bin/python
REPAIR_PROVENANCE="$RUN/provenance/stage2-repair-v2"
HELPER="$REPAIR_PROVENANCE/navigation_rlt_stage2_repair_bundle.py"
CONTEXT="$REPAIR_PROVENANCE/bundle_context.json"
PROTOCOL="$REPAIR_PROVENANCE/stage2_repair_protocol.json"
PROMOTED="$RUN/navigation_rlt_stage2_replay"
PREPARED="$RUN/navigation_rlt_stage2_prepare"
STAGE2="$RUN/navigation_rlt_stage2"
NEW_LOCK="$RUN/.launch-navigation-rlt-stage2-repair-v2"
OLD_LOCK="$RUN/.launch-navigation-rlt-stage2"

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
  "$PYTHON" - "$document" "$field" <<'PY'
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

[[ -f "$HELPER" && ! -L "$HELPER" ]] || die "repair provenance helper is missing or unsafe"
[[ -f "$CONTEXT" && ! -L "$CONTEXT" ]] || die "repair bundle context is missing or unsafe"
[[ -d "$NEW_LOCK" && ! -L "$NEW_LOCK" ]] || die "fixed Stage2 repair launch lock is absent or unsafe"
[[ ! -e "$OLD_LOCK" && ! -L "$OLD_LOCK" ]] || die "legacy Stage2 launch lock exists"

context_json=$(cat -- "$CONTEXT")
CODE_VERIFICATION=$(json_get "$context_json" stage2_repair_code_verification.path)
CODE_VERIFICATION_SHA=$(json_get "$context_json" stage2_repair_code_verification.sha256)
code_json=$("$PYTHON" -B "$HELPER" verify-code \
  --run-root "$RUN" \
  --stage2-repair-code-verification "$CODE_VERIFICATION" \
  --expected-stage2-repair-code-verification-sha256 "$CODE_VERIFICATION_SHA")
CODE=$(json_get "$code_json" code_root)
RUN_WRAPPER="$CODE/run_navigation_rlt_a800.sh"
[[ -f "$RUN_WRAPPER" && ! -L "$RUN_WRAPPER" ]] || \
  die "repair run wrapper is missing or unsafe: $RUN_WRAPPER"

stage1_json=$("$PYTHON" -B "$HELPER" inspect-stage1 \
  --run-root "$RUN" --run-id "$RUN_ID")
STAGE1_CHECKPOINT=$(json_get "$stage1_json" stage1_checkpoint.path)
STAGE1_CHECKPOINT_SHA=$(json_get "$stage1_json" stage1_checkpoint.sha256)

[[ -d "$PROMOTED" && ! -L "$PROMOTED" ]] || \
  die "v2 recovery requires the completed v1 Stage2 replay directory"
[[ -f "$PROMOTED/COMPLETED.json" && ! -L "$PROMOTED/COMPLETED.json" ]] || \
  die "v2 recovery refuses to recompute an incomplete Stage2 replay"
promotion_json=$("$PYTHON" -B "$HELPER" verify-promotion \
  --run-root "$RUN" --run-id "$RUN_ID" --promotion-dir "$PROMOTED")
PROMOTED_MANIFEST=$(json_get "$promotion_json" manifest.path)
PROMOTED_SHA=$(json_get "$promotion_json" manifest.sha256)

[[ -d "$PREPARED" && ! -L "$PREPARED" ]] || \
  die "v2 recovery requires the completed v1 Stage2 prepare directory"
[[ -f "$PREPARED/COMPLETED.json" && ! -L "$PREPARED/COMPLETED.json" ]] || \
  die "v2 recovery refuses to recompute an incomplete Stage2 prepare"

if [[ ! -e "$PROTOCOL" && ! -L "$PROTOCOL" ]]; then
  protocol_json=$("$PYTHON" -B "$HELPER" write-protocol \
    --run-root "$RUN" --run-id "$RUN_ID" \
    --stage1-checkpoint "$STAGE1_CHECKPOINT" \
    --expected-stage1-checkpoint-sha256 "$STAGE1_CHECKPOINT_SHA" \
    --stage2-replay-manifest "$PROMOTED_MANIFEST" \
    --expected-stage2-replay-manifest-sha256 "$PROMOTED_SHA" \
    --stage2-prepare-dir "$PREPARED")
  PROTOCOL_SHA=$(json_get "$protocol_json" protocol.sha256)
else
  [[ -f "$PROTOCOL" && ! -L "$PROTOCOL" ]] || die "repair protocol path is unsafe"
  PROTOCOL_SHA=$(sha256_file "$PROTOCOL")
fi

preflight_json=$("$PYTHON" -B "$HELPER" verify-prelaunch \
  --run-root "$RUN" --run-id "$RUN_ID" \
  --stage2-repair-protocol "$PROTOCOL" \
  --expected-stage2-repair-protocol-sha256 "$PROTOCOL_SHA" \
  --stage1-checkpoint "$STAGE1_CHECKPOINT" \
  --expected-stage1-checkpoint-sha256 "$STAGE1_CHECKPOINT_SHA" \
  --stage2-replay-manifest "$PROMOTED_MANIFEST" \
  --expected-stage2-replay-manifest-sha256 "$PROMOTED_SHA" \
  --stage2-prepare-dir "$PREPARED")
[[ "$(json_get "$preflight_json" status)" == passed ]] || \
  die "Stage2 repair preflight did not pass"

STAGE2_RUN_MANIFEST="$PREPARED/run_manifest.json"
STAGE2_RUN_SHA=$(sha256_file "$STAGE2_RUN_MANIFEST")
if [[ ! -e "$STAGE2" && ! -L "$STAGE2" ]]; then
  "$RUN_WRAPPER" \
    "$RUN" stage2 \
    "$STAGE2_RUN_MANIFEST" "$STAGE2_RUN_SHA" \
    "$PROMOTED_MANIFEST" "$PROMOTED_SHA" \
    "$STAGE1_CHECKPOINT" "$STAGE1_CHECKPOINT_SHA"
elif [[ ! -f "$STAGE2/STAGE.json" || -L "$STAGE2/STAGE.json" ]]; then
  die "Stage2 output exists without a safe STAGE.json: $STAGE2"
fi

completed_json=$("$PYTHON" -B "$HELPER" verify-stage2 \
  --run-root "$RUN" --run-id "$RUN_ID" \
  --stage2-repair-protocol "$PROTOCOL" \
  --expected-stage2-repair-protocol-sha256 "$PROTOCOL_SHA" \
  --stage1-checkpoint "$STAGE1_CHECKPOINT" \
  --expected-stage1-checkpoint-sha256 "$STAGE1_CHECKPOINT_SHA" \
  --stage2-replay-manifest "$PROMOTED_MANIFEST" \
  --expected-stage2-replay-manifest-sha256 "$PROMOTED_SHA" \
  --stage2-prepare-dir "$PREPARED")
printf 'Stage2 repair completed and verified. checkpoint=%s sha256=%s protocol_sha256=%s\n' \
  "$(json_get "$completed_json" checkpoint.path)" \
  "$(json_get "$completed_json" checkpoint.sha256)" \
  "$PROTOCOL_SHA"
