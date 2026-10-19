#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TASK_OUT=${1:?output directory required}
shift
mkdir -p "$TASK_OUT"
TASK_OUT=$(cd -- "$TASK_OUT" && pwd)
bash "$TASK_ROOT/launch.sh" "$TASK_OUT" --exit-interval 32 --save-interval 32 "$@" > "$TASK_OUT/phase_1.log" 2>&1
bash "$TASK_ROOT/launch.sh" "$TASK_OUT" --load "$TASK_OUT/checkpoints" --save-interval 32 "$@" > "$TASK_OUT/phase_2.log" 2>&1
/opt/opguard/.venv_opguard/bin/python "$TASK_ROOT/collect_run.py" "$TASK_OUT"
