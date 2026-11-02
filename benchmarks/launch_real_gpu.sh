#!/usr/bin/env bash
# Local benchmark only: no services or canonical data are modified.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT="${NAHUAL_BENCH_ARTIFACT:-/work/scratch/amunoz/nahual-real-gpu-20261102}"
RECIPE="${NAHUAL_RECIPE_ROOT:-/work/users/amunoz/deployments/JUMP_lite-remaining-preliminary-fast-v1-pipelined4-nonappose-20261021}"
RUNTIME="${NAHUAL_RUNTIME_ROOT:-/work/scratch/amunoz/broader-jump-nonlite-dl-successor-v2/runtime}"
PYTHON="$RUNTIME/model-env-v2/bin/python"
CODE_ROOT="${NAHUAL_CODE_ROOT:-$ROOT/src}"
export PYTHONPATH="$CODE_ROOT:$ROOT:$RECIPE:$RUNTIME/nahual-shm-env-v1/site-packages:/work/users/amunoz/projects/JUMP_lite-broader-jump-dl-ranking/.venv/lib/python3.12/site-packages"
export LD_LIBRARY_PATH="/run/opengl-driver/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4 POLARS_MAX_THREADS=4
exec "$PYTHON" -B -m benchmarks.run_real_gpu --manifest "$ARTIFACT/inputs/manifest.json" "$@"
