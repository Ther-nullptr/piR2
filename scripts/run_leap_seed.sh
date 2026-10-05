#!/usr/bin/env bash
set -euo pipefail
PIR2_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PIR2_ROOT"
: "${PIR2_GPU:?Select an available GPU first}"
: "${PIR2_SEED:?Specify the training seed}"
export CUDA_VISIBLE_DEVICES="$PIR2_GPU"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="$PIR2_ROOT/.cache/jax"
export OMP_NUM_THREADS=4
PIR2_PYTHON="$PIR2_ROOT/.venv-sim/bin/python"
PIR2_RUN_DIR="$PIR2_ROOT/outputs/simulation/pir2-seed$PIR2_SEED"
PIR2_STATUS="$PIR2_ROOT/artifacts/simulation/job-seed$PIR2_SEED-status.txt"
mkdir -p "$PIR2_RUN_DIR" "$PIR2_ROOT/artifacts/simulation"
exec 9> "$PIR2_RUN_DIR/run.lock"
flock -n 9 || { echo "This seed already has a running job"; exit 1; }
trap 'PIR2_EXIT=$?; date -Iseconds > "$PIR2_STATUS"; echo "failed: exit $PIR2_EXIT; inspect the log before resuming" >> "$PIR2_STATUS"; exit "$PIR2_EXIT"' ERR
date -Iseconds > "$PIR2_STATUS"
echo "training: piR2 seed $PIR2_SEED, 800 epochs" >> "$PIR2_STATUS"
PIR2_RESUME=()
if [ -f "$PIR2_RUN_DIR/latest.pt" ]; then
    PIR2_RESUME=(--resume "$PIR2_RUN_DIR/latest.pt")
fi
"$PIR2_PYTHON" -u -m simulation.train --method pir2 --seed "$PIR2_SEED" \
    --epochs 800 --output "$PIR2_RUN_DIR" "${PIR2_RESUME[@]}"
date -Iseconds > "$PIR2_STATUS"
echo 'evaluating: 100 episodes per condition' >> "$PIR2_STATUS"
"$PIR2_PYTHON" -u -m simulation.evaluate \
    --checkpoint "$PIR2_RUN_DIR/epoch-0800.pt" \
    --output "artifacts/simulation/pir2-seed$PIR2_SEED-epoch800-async.json"
"$PIR2_PYTHON" -u -m simulation.evaluate \
    --checkpoint "$PIR2_RUN_DIR/epoch-0800.pt" --no-async \
    --output "artifacts/simulation/pir2-seed$PIR2_SEED-epoch800-no-async.json"
date -Iseconds > "$PIR2_STATUS"
echo 'complete: training and evaluation' >> "$PIR2_STATUS"
