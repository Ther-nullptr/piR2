#!/usr/bin/env bash
set -euo pipefail
PIR2_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PIR2_ROOT"
PIR2_GPU="${PIR2_GPU:-3}"
PIR2_MODE="${1:-measure}"
PIR2_OUTPUT="${PIR2_OUTPUT:-$PIR2_ROOT/artifacts/coexecution/$PIR2_MODE-$(date +%Y%m%d-%H%M%S)}"
PIR2_PYTHON="$PIR2_ROOT/.venv/bin/python"
"$PIR2_PYTHON" -m coexecution.preflight --gpu "$PIR2_GPU" --output "$PIR2_OUTPUT/gpu-preflight.json"
export CUDA_VISIBLE_DEVICES="$PIR2_GPU"
export OMP_NUM_THREADS=4
export GR00T_IMAGE_DELAY_MAX=0
export NO_ALBUMENTATIONS_UPDATE=1
export GROOT_HF_LOCAL_FIRST=1
export GROOT_PATCH_MISTRAL=1
if [ "$PIR2_MODE" = measure ]; then
    exec "$PIR2_PYTHON" -u -m coexecution.run --output "$PIR2_OUTPUT" \
        --iterations 100 --repeats 3 --warmup 5
elif [ "$PIR2_MODE" = fusion ]; then
    exec "$PIR2_PYTHON" -u -m coexecution.fusion_ab --output "$PIR2_OUTPUT"
elif [ "$PIR2_MODE" = runtime ]; then
    exec "$PIR2_PYTHON" -u -m coexecution.runtime_ab --output "$PIR2_OUTPUT" \
        --periodic-seconds 20 --periodic-repeats 3
elif [ "$PIR2_MODE" = protocols ]; then
    exec "$PIR2_PYTHON" -u -m coexecution.protocol_ab --output "$PIR2_OUTPUT"
elif [ "$PIR2_MODE" = trace ] || [ "$PIR2_MODE" = operators ] || [ "$PIR2_MODE" = fusion-trace ] || [ "$PIR2_MODE" = runtime-trace ]; then
    PIR2_NSYS="${PIR2_NSYS:-nsys}"
    if [ "$PIR2_MODE" = operators ]; then
        PIR2_ENTRY=(coexecution.profile_operators --output "$PIR2_OUTPUT" --iterations 3 --warmup 1)
    elif [ "$PIR2_MODE" = fusion-trace ]; then
        PIR2_ENTRY=(coexecution.fusion_ab --output "$PIR2_OUTPUT" --variants baseline rope_rms
            --profile --iterations 6 --repeats 1 --warmup 2 --check-actions 8 --modes serial concurrent)
    elif [ "$PIR2_MODE" = runtime-trace ]; then
        PIR2_ENTRY=(coexecution.runtime_ab --output "$PIR2_OUTPUT" --variants rope combined
            --profile --iterations 6 --repeats 1 --warmup 2 --check-actions 8 --modes serial concurrent)
    else
        PIR2_ENTRY=(coexecution.run --output "$PIR2_OUTPUT" --profile --modes serial concurrent
            --iterations 6 --repeats 1 --warmup 2)
    fi
    exec "$PIR2_NSYS" profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
        --capture-range=cudaProfilerApi --capture-range-end=stop --cuda-graph-trace=node \
        -o "$PIR2_OUTPUT/nsys" "$PIR2_PYTHON" -u -m "${PIR2_ENTRY[@]}"
elif [ "$PIR2_MODE" = periodic ]; then
    exec "$PIR2_PYTHON" -u -m coexecution.periodic --output "$PIR2_OUTPUT/results.json" \
        --seconds 20 --period-ms 40 --camera-hz 30
else
    echo "Usage: bash scripts/run_single_gpu_baseline.sh [measure|trace|operators|periodic|fusion|fusion-trace|runtime|runtime-trace|protocols]" >&2
    exit 2
fi
