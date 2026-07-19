#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/mnt/storage/paritosh/hf_cache
export HUGGINGFACE_HUB_CACHE=/mnt/storage/paritosh/hf_cache
export HF_DATASETS_CACHE=/mnt/storage/paritosh/hf_cache/datasets

ROOT=/home/paritosh/priyasmita/Qwen3-tts/v2
RUN_DIR="$ROOT/sft_output_v6/20260614_010317"
SPEECH_DIR=/mnt/storage/paritosh/v6_speech_outputs_20260614

mkdir -p "$SPEECH_DIR"
cd "$ROOT"

python3 -u inference_v6.py \
  --speech \
  --n_samples 0 \
  --resume_speech_dir "$SPEECH_DIR" \
  2>&1 | tee v6_speech_inference.log

python3 -u inference_v6.py \
  --n_samples 0 \
  2>&1 | tee v6_text_inference.log

python3 -u error_analysis.py \
  --version v6 \
  2>&1 | tee v6_error_analysis.log

python3 -u judge.py \
  --version v6 \
  2>&1 | tee v6_judge.log
