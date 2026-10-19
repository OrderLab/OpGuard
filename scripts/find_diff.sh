#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 <trace1_path> <trace2_path> <output_dir> [filter_op] [dp_count] [worker_count]"
  echo "Example:"
  echo "  $0 /path/trace1 /path/trace2 /tmp/out"
  echo "  $0 /path/trace1 /path/trace2 /tmp/out 'meta.global_step<=299'"
  echo "  $0 /path/trace1 /path/trace2 /tmp/out 'meta.global_step<=299' 8"
  echo "  $0 /path/trace1 /path/trace2 /tmp/out 'meta.global_step<=299' 8 4"
}

if [[ $# -lt 3 || $# -gt 7 ]]; then
  usage
  exit 1
fi

TRACE1="$1"
TRACE2="$2"
OUTDIR="$3"
FILTER_OP="${4-}"       
DP_COUNT="${5-}"        
WORKER_COUNT="${6-}"
IGNORE_CONFIG="${7-}"    

echo "[OpGuard]==> Running inspector..."

INSPECT_ARGS=(
  -m opguard.inspector
  "$TRACE1" "$TRACE2"
  --out-dir "$OUTDIR"
)

if [[ -n "${FILTER_OP}" ]]; then
  INSPECT_ARGS+=( --filter "$FILTER_OP" )
fi

if [[ -n "${WORKER_COUNT}" ]]; then
  INSPECT_ARGS+=( --worker "$WORKER_COUNT" )
fi

if [[ -n "${IGNORE_CONFIG}" ]]; then
  INSPECT_ARGS+=( --ignore-config "$IGNORE_CONFIG" )
fi

set -x
python3 "${INSPECT_ARGS[@]}"
set +x

COMBINED="$OUTDIR/combined_diff_log.jsonl"

if [[ -s "$COMBINED" ]]; then
  echo "[OpGuard]==> There is diff between $TRACE1 and $TRACE2, combined_diff_log.jsonl is non-empty: $COMBINED"

  
  if [[ -f "$OUTDIR/first_diff_log.jsonl" ]]; then
    echo "[OpGuard]==> First diff log:"
    cat "$OUTDIR/first_diff_log.jsonl"
  fi

  echo "[OpGuard]==> Running build-effect-chain..."

  
  EFFECT_ARGS=(
    -m opguard.build-effect-chain
    --diff_log "$COMBINED"
    --out-dir "$OUTDIR"
  )
  if [[ -n "${DP_COUNT}" ]]; then
    EFFECT_ARGS+=( --dp "$DP_COUNT" )
  fi

  set -x
  python3 "${EFFECT_ARGS[@]}"
  set +x
else
  echo "[OpGuard]==> There is no diff between $TRACE1 and $TRACE2, combined_diff_log.jsonl not found or empty: $COMBINED"
  echo "[OpGuard]==> Skipping build-effect-chain."
fi
