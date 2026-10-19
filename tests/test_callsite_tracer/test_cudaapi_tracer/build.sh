#!/usr/bin/env bash
set -euo pipefail

# Adjust if needed
: "${CUDA_HOME:=${CUDA_HOME:-/usr/local/cuda}}"

CUPTI_INC="$CUDA_HOME/extras/CUPTI/include"
CUPTI_LIB="$CUDA_HOME/extras/CUPTI/lib64"
CUDA_LIB="$CUDA_HOME/lib64"

PY_INC=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['include'])")
PY_LDFLAGS=$(python3-config --ldflags)

g++ -std=c++17 -fPIC -shared hook.cpp -o libpylaunch_hook.so $(python3-config --includes) \
  -I"$CUPTI_INC" -I"$PY_INC" -I"$CUDA_HOME/include" \
  -L"$CUPTI_LIB" -L"$CUDA_LIB" \
  -lcupti -lcuda -lcudart -ldl -lpthread $PY_LDFLAGS

echo "Built libpylaunch_hook.so"
