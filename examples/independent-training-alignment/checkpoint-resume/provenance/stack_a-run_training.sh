#!/usr/bin/env bash
set -euo pipefail
if [[ "$#" != 2 ]]; then
    printf 'Usage: %s CONFIG OUTPUT\n' "$0" >&2
    exit 2
fi
STACK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_CONFIG="$(realpath "$1")"
TRAIN_OUTPUT="$(realpath -m "$2")"
mkdir -p "${TRAIN_OUTPUT}"
"${STACK_ROOT}/launch.sh" --config "${TRAIN_CONFIG}" --output "${TRAIN_OUTPUT}" --stop-after 32 > "${TRAIN_OUTPUT}/phase_1.log" 2>&1
cp -- "${TRAIN_OUTPUT}/result.json" "${TRAIN_OUTPUT}/phase_1_result.json"
"${STACK_ROOT}/launch.sh" --config "${TRAIN_CONFIG}" --output "${TRAIN_OUTPUT}" --resume "${TRAIN_OUTPUT}/checkpoint-32" > "${TRAIN_OUTPUT}/phase_2.log" 2>&1
