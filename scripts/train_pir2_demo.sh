#!/usr/bin/env bash
set -euo pipefail

PIR2_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIR2_REPO="$PIR2_ROOT/upstream/learning/Isaac-GR00T"
PIR2_PYTHON="$PIR2_ROOT/.venv/bin/python"
PIR2_STEPS="${PIR2_STEPS:-10}"
PIR2_OUTPUT="${PIR2_OUTPUT:-$PIR2_ROOT/outputs/pir2-so100-smoke}"

: "${PIR2_GPU:?Set PIR2_GPU after checking nvidia-smi.}"
export CUDA_VISIBLE_DEVICES="$PIR2_GPU"
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export GR00T_IMAGE_DELAY_MAX=5
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1
export GROOT_HF_LOCAL_FIRST=1
export GROOT_PATCH_MISTRAL=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PIR2_REPO${PYTHONPATH:+:$PYTHONPATH}"

nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
"$PIR2_PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA is required"; print("Training on:", torch.cuda.get_device_name(0))'

cd "$PIR2_REPO"
exec "$PIR2_PYTHON" gr00t/experiment/launch_finetune.py \
    --base-model-path "$PIR2_ROOT/models/GR00T-N1.7-3B" \
    --dataset-path "$PIR2_REPO/demo_data/cube_to_bowl_5" \
    --modality-config-path "$PIR2_REPO/examples/SO100/so100_config.py" \
    --embodiment-tag NEW_EMBODIMENT \
    --num-gpus 1 --global-batch-size 1 \
    --max-steps "$PIR2_STEPS" --save-steps "$PIR2_STEPS" \
    --save-total-limit 1 --save-only-model \
    --output-dir "$PIR2_OUTPUT" \
    --dataloader-num-workers 2 \
    --shard-size 1024 --num-shards-per-epoch 2 --episode-sampling-rate 0.1 \
    --streaming --streaming-constant-weight 0.2 \
    --streaming-chunk-wise-weight 0.8 --streaming-schedule-mode pir2 \
    --streaming-chunk-size-max 5 --streaming-mask-clean-end \
    --image-delay-max 5 --image-delay-embed-dim 64 \
    "$@"
