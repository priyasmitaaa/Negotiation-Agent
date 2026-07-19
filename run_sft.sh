#!/bin/bash
# run_sft.sh — Launch Qwen2.5-Omni SFT on this machine
# -------------------------------------------------------
# Usage:
#   bash run_sft.sh              # auto-select GPUs, recommended DDP mode
#   bash run_sft.sh --single     # GPU 1 only (QLoRA, if GPU 0 still busy)
#   bash run_sft.sh --dual-mp    # GPU 0+1, model-parallel QLoRA (single process)
#   bash run_sft.sh --dual-ddp   # GPU 0+1, full bf16 LoRA via DDP (fastest)
#
# GPU status at time of writing:
#   GPU 0  A100 80GB   BUSY (PID 9284, 17GB used, 65% util — someone else)
#   GPU 1  A100 80GB   FREE (22 MiB, 0% util)
#
# Model: Qwen/Qwen2.5-Omni-7B  (will be downloaded from HF Hub on first run)
# NOTE:  Set HF_TOKEN env var if the model requires authentication.
# -------------------------------------------------------

set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-}"

# ── Step 1: Fix audio paths if needed ──────────────────────────────────────
echo "[1/3] Checking dataset audio paths..."
python3 fix_dataset_paths.py

# ── Step 2: GPU availability check ─────────────────────────────────────────
echo ""
echo "[2/3] GPU status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits | \
    awk -F',' '{printf "  GPU %s  %s  used=%sMiB  free=%sMiB  util=%s%%\n",$1,$2,$3,$4,$5}'

GPU0_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0 | tr -d ' ')
GPU0_MEM=$(nvidia-smi  --query-gpu=memory.used     --format=csv,noheader,nounits -i 0 | tr -d ' ')

echo ""
if [ "$GPU0_MEM" -gt 1000 ]; then
    echo "  WARNING: GPU 0 is busy (${GPU0_MEM} MiB used). "
    echo "  Use --single to train on GPU 1 only, or wait until GPU 0 is free."
else
    echo "  GPU 0 appears free."
fi

# ── Step 3: Launch ──────────────────────────────────────────────────────────
echo ""
echo "[3/3] Launching training..."

COMMON_ARGS=(
    --model_id       "Qwen/Qwen2.5-Omni-7B"
    --dataset_dir    "dataset"
    --output_dir     "sft_output"
    --epochs         1          # 1 epoch; single-GPU gives 127 steps (same as v4, now with full audio)
    --lr             5e-5       # lowered from 2e-4 to prevent loss collapse
    --weight_decay   0.01       # L2 regularisation
    --lora_r         8          # halved from 16; less memorisation capacity
    --lora_alpha     16         # keep alpha/r = 2
    --lora_dropout   0.1        # increased from 0.05
    --max_length     3072
    --batch_size     1
    --grad_accum     16
    --save_steps     50         # finer checkpointing to catch loss early
    --eval_steps     50
    --warmup_ratio   0.1
)

case "$MODE" in

  --single)
    # GPU 1 only — QLoRA, safe when GPU 0 is occupied
    echo "  Mode: single GPU (GPU 1), QLoRA"
    CUDA_VISIBLE_DEVICES=1 python3 train_sft.py "${COMMON_ARGS[@]}"
    ;;

  --dual-mp)
    # GPU 0+1 — model-parallel QLoRA (single process, device_map=auto)
    # Model is split across GPUs layer-by-layer. Requires both GPUs free.
    echo "  Mode: dual GPU model-parallel, QLoRA"
    CUDA_VISIBLE_DEVICES=0,1 python3 train_sft.py "${COMMON_ARGS[@]}"
    ;;

  --dual-ddp)
    # GPU 0+1 — full bf16 LoRA via DDP (2 processes, one per GPU)
    # Faster throughput. Each GPU holds a full model copy (~14 GB in bf16).
    # Requires both GPUs free.
    echo "  Mode: dual GPU DDP, bf16 LoRA (no quantization)"
    CUDA_VISIBLE_DEVICES=0,1 python3 -m accelerate.commands.launch \
        --config_file accelerate_2gpu.yaml \
        train_sft.py "${COMMON_ARGS[@]}" --no_qlora
    ;;

  "")
    # Auto: pick best available mode
    if [ "$GPU0_MEM" -gt 1000 ]; then
        echo "  Auto-selected: single GPU (GPU 1), QLoRA  [GPU 0 is busy]"
        CUDA_VISIBLE_DEVICES=1 python3 train_sft.py "${COMMON_ARGS[@]}"
    else
        echo "  Auto-selected: dual GPU DDP, bf16 LoRA  [both GPUs free]"
        CUDA_VISIBLE_DEVICES=0,1 python3 -m accelerate.commands.launch \
            --config_file accelerate_2gpu.yaml \
            train_sft.py "${COMMON_ARGS[@]}" --no_qlora
    fi
    ;;

  *)
    echo "Unknown mode: $MODE"
    echo "Usage: bash run_sft.sh [--single | --dual-mp | --dual-ddp]"
    exit 1
    ;;

esac
