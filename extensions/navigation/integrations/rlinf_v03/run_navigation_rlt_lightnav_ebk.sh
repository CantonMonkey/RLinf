#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 RUN_ROOT PORT EXPECTED_RUN_MANIFEST_SHA256" >&2
  exit 2
fi

RUN_ROOT=$1
PORT=$2
EXPECTED_RUN_SHA=$3
RUN_MANIFEST="$RUN_ROOT/run_manifest.json"
SOURCE_ROOT=/srv/navigation/repos/LightNav-0-navigation-rlt
MODEL_ROOT=/opt/LightNav-0/checkpoints/LightNav-0
ENV_ROOT=/opt/conda/envs/lightnav0
FEATURE_ROOT="$RUN_ROOT/feature-artifacts"
READY_FILE="$RUN_ROOT/status/lightnav.ready"

case "$RUN_ROOT" in
  /srv/navigation/runs/*) ;;
  *) echo "RUN_ROOT must be under the private Navigation-RLT runs root" >&2; exit 3 ;;
esac
[[ "$PORT" =~ ^[0-9]+$ ]] && [[ "$PORT" -ge 1024 ]] && [[ "$PORT" -le 65535 ]]
[[ "$EXPECTED_RUN_SHA" =~ ^[0-9a-f]{64}$ ]]
test -x "$ENV_ROOT/bin/python"
test -f "$RUN_MANIFEST"
test -f "$MODEL_ROOT/model-00001-of-00001.safetensors"
test "$(sha256sum "$RUN_MANIFEST" | cut -d' ' -f1)" = "$EXPECTED_RUN_SHA"
test ! -e "$READY_FILE"
mkdir -p "$RUN_ROOT/status" "$RUN_ROOT/logs"
if [[ -e "$FEATURE_ROOT" ]]; then
  test -d "$FEATURE_ROOT"
  test ! -L "$FEATURE_ROOT"
else
  mkdir "$FEATURE_ROOT"
fi

readarray -t IDENTITY < <(
  "$ENV_ROOT/bin/python" - "$RUN_MANIFEST" <<'PY'
import json, sys
x=json.load(open(sys.argv[1], encoding='utf-8'))
assert x['schema_name']=='navigation_rlt_experiment_manifest' and x['schema_version']==1
assert x['boundaries']=={
  'lightnav_frozen': True,
  'remote_push': False,
  'ttt_or_secondary_post_training': False,
  'upstream_split_read': 'train',
  'validation_split_reads': 0,
  'weight_origin': 'original_release',
}
assert x['lightnav']['backend']=='hf'
assert x['lightnav']['context_request']=='contextual_prefix_hidden_v1'
print(x['lightnav']['patched_head'])
print(x['lightnav']['weight_sha256'])
PY
)
test "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" = "${IDENTITY[0]}"
test -z "$(git -C "$SOURCE_ROOT" status --porcelain)"
test "$(sha256sum "$MODEL_ROOT/model-00001-of-00001.safetensors" | cut -d' ' -f1)" = "${IDENTITY[1]}"

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$SOURCE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export LIGHTNAV_ATTN=sdpa

exec "$ENV_ROOT/bin/python" -m lightnav.serving.ws_server \
  --task vln \
  --model_path "$MODEL_ROOT" \
  --backend hf \
  --device cuda \
  --num_history_frames 64 \
  --host 127.0.0.1 \
  --port "$PORT" \
  --max_batch_size 4 \
  --max_wait_ms 8 \
  --feature_request_capability contextual_prefix_hidden_v1 \
  --feature_artifact_dir "$FEATURE_ROOT" \
  --feature_max_tokens 4096 \
  --feature_max_bytes_mib 128 \
  --ready_file "$READY_FILE"
