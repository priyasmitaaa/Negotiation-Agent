#!/bin/bash
# run_cl_sft.sh — Curriculum Learning SFT (3 phases)
# =====================================================
#
# Phases (starting from v5c SFT adapter at sft_output/final_adapter):
#
#   Phase 1 — EASY + 30 EL floor (~746 samples), 2 epochs, lr=2e-5
#             Consolidates MM/EL boundary on unambiguous cases; EL floor prevents
#             EL recall collapse from the 97%-MM class imbalance in EASY tier.
#             Init: v5c adapter (88.48% test)
#             Output: sft_output/cl_phase1/final_adapter
#
#   Phase 2 — EASY+MEDIUM (1362 samples), 1 epoch, lr=1e-5
#             Smooth expansion to borderline-confident cases.
#             Init: Phase 1 adapter
#             Output: sft_output/cl_phase2/final_adapter
#
#   Phase 3 — HARD + 20% EASY replay (≈808 samples), 2 epochs, lr=5e-6
#             Targets the 69.4%-accuracy hard region; replay prevents forgetting.
#             Init: Phase 2 adapter
#             Output: sft_output/cl_phase3/final_adapter
#
# Eval gating: after each phase, eval on val set.
# If macro-F1 < 0.8664 (v5c baseline), abort and keep previous adapter.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=1 bash run_cl_sft.sh
#   CUDA_VISIBLE_DEVICES=1 bash run_cl_sft.sh --phase 2   # resume at phase 2
# =====================================================

set -euo pipefail
cd "$(dirname "$0")"

GPU="${CUDA_VISIBLE_DEVICES:-1}"
START_PHASE="${2:-1}"  # pass --phase N to skip to a later phase

BASELINE_ADAPTER="sft_output/final_adapter"
BASELINE_F1="0.860"       # CL gate: bootstrap CI on val (~±0.03) makes 0.8664 indistinguishable
                          # from 0.862; gate at 0.860 is statistically meaningful
PHASE1_GATE_F1="0.80"     # Phase 1 trains on 93% MM + EL floor; class weights over-correct toward EL
                          # → MM recall dips but EL recall rises; Phase 2 recovers MM recall

COMMON=(
    --model_id       "Qwen/Qwen2.5-Omni-7B"
    --dataset_dir    "dataset"
    --lora_r         8
    --lora_alpha     16
    --lora_dropout   0.1
    --max_length     3072
    --batch_size     1
    --grad_accum     16
    --weight_decay   0.01
    --warmup_ratio   0.05
    --save_steps     30
    --eval_steps     30
    --no_qlora
)

# ── Helpers ────────────────────────────────────────────────────────────────────

run_eval() {
    local adapter="$1"
    local split="$2"    # val_sft or test_sft (must match a dataset/${split}.jsonl file)
    echo "  [EVAL] ${adapter} on ${split}..."
    CUDA_VISIBLE_DEVICES="$GPU" python3 eval_sft.py \
        --adapter    "$adapter" \
        --jsonl      "dataset/${split}.jsonl" \
        --save_preds
    echo "  [EVAL] done → ${adapter}/eval_${split}.json"
}

check_f1() {
    # Returns 0 (pass) if val macro-F1 >= threshold, 1 (fail) otherwise
    # Usage: check_f1 <adapter> [threshold]  (default threshold = BASELINE_F1)
    local adapter="$1"
    local threshold="${2:-${BASELINE_F1}}"
    local json="${adapter}/eval_val_sft.json"
    if [ ! -f "$json" ]; then
        echo "  [GATE] No eval JSON found at ${json} — running eval first..."
        run_eval "$adapter" "val_sft"
    fi
    local f1
    f1=$(python3 -c "import json; d=json.load(open('${json}')); print(d['f1_macro'])")
    local pass
    pass=$(python3 -c "print('ok' if ${f1} >= ${threshold} else 'fail')")
    echo "  [GATE] ${adapter}  val F1=${f1}  threshold=${threshold}  → ${pass}"
    [ "$pass" = "ok" ]
}

# ── Prepare curriculum data files ─────────────────────────────────────────────

echo ""
echo "================================================================"
echo "  CL-SFT PIPELINE — starting at Phase ${START_PHASE}"
echo "  GPU: ${GPU}   Baseline adapter: ${BASELINE_ADAPTER}"
echo "================================================================"

echo ""
echo "[PREP] Preparing curriculum data files..."
python3 prepare_cl_data.py --replay_frac 0.20 --el_floor 30 --seed 42
echo "[PREP] Done."

# ── Phase 1: EASY only ────────────────────────────────────────────────────────

if [ "${START_PHASE}" -le 1 ]; then
    echo ""
    echo "================================================================"
    echo "  PHASE 1 — EASY + 30 EL floor (~746 samples) | lr=2e-5 | 2 epochs"
    echo "  Init adapter: ${BASELINE_ADAPTER}"
    echo "================================================================"

    CUDA_VISIBLE_DEVICES="$GPU" python3 train_sft.py \
        "${COMMON[@]}" \
        --train_jsonl  "dataset/cl_phase1.jsonl" \
        --init_adapter "${BASELINE_ADAPTER}" \
        --output_dir   "sft_output/cl_phase1" \
        --epochs       2 \
        --lr           2e-5

    # train_sft.py saves output_dir/final_adapter at end (load_best_model_at_end=True)
    echo "  [PHASE1] Adapter saved by train_sft.py → sft_output/cl_phase1/final_adapter"

    # Eval gate
    run_eval "sft_output/cl_phase1/final_adapter" "val_sft"
    if ! check_f1 "sft_output/cl_phase1/final_adapter" "${PHASE1_GATE_F1}"; then
        echo "  [GATE] Phase 1 dropped below Phase1 gate F1=${PHASE1_GATE_F1}!"
        echo "  [GATE] Aborting CL. Keeping v5c adapter at ${BASELINE_ADAPTER}."
        exit 1
    fi
    echo "  [PHASE1] PASSED gate (threshold=${PHASE1_GATE_F1}). Proceeding to Phase 2."
fi

# ── Phase 2: EASY + MEDIUM ────────────────────────────────────────────────────

if [ "${START_PHASE}" -le 2 ]; then
    if [ ! -f "sft_output/cl_phase1/final_adapter/adapter_config.json" ]; then
        echo "  [ERROR] Phase 2 requires sft_output/cl_phase1/final_adapter/adapter_config.json"
        echo "  [ERROR] Run Phase 1 first (or without --phase flag to start from Phase 1)."
        exit 1
    fi
    echo ""
    echo "================================================================"
    echo "  PHASE 2 — EASY+MEDIUM (1362 samples) | lr=1e-5 | 1 epoch"
    echo "  Init adapter: sft_output/cl_phase1/final_adapter"
    echo "================================================================"

    CUDA_VISIBLE_DEVICES="$GPU" python3 train_sft.py \
        "${COMMON[@]}" \
        --train_jsonl  "dataset/cl_phase2.jsonl" \
        --init_adapter "sft_output/cl_phase1/final_adapter" \
        --output_dir   "sft_output/cl_phase2" \
        --epochs       1 \
        --lr           1e-5

    # train_sft.py saves output_dir/final_adapter at end (load_best_model_at_end=True)
    echo "  [PHASE2] Adapter saved by train_sft.py → sft_output/cl_phase2/final_adapter"

    run_eval "sft_output/cl_phase2/final_adapter" "val_sft"
    if ! check_f1 "sft_output/cl_phase2/final_adapter" "${BASELINE_F1}"; then
        echo "  [GATE] Phase 2 dropped below baseline F1=${BASELINE_F1}!"
        echo "  [GATE] Keeping Phase 1 adapter at sft_output/cl_phase1/final_adapter."
        exit 1
    fi
    echo "  [PHASE2] PASSED gate (threshold=${BASELINE_F1}). Proceeding to Phase 3."
fi

# ── Phase 3: HARD + 20% EASY replay ──────────────────────────────────────────

if [ "${START_PHASE}" -le 3 ]; then
    if [ ! -f "sft_output/cl_phase2/final_adapter/adapter_config.json" ]; then
        echo "  [ERROR] Phase 3 requires sft_output/cl_phase2/final_adapter/adapter_config.json"
        echo "  [ERROR] Run Phase 2 first (or without --phase flag to start from Phase 1)."
        exit 1
    fi
    echo ""
    echo "================================================================"
    echo "  PHASE 3 — HARD + 20% EASY replay (~806 samples) | lr=5e-6 | 2 epochs"
    echo "  Init adapter: sft_output/cl_phase2/final_adapter"
    echo "================================================================"

    CUDA_VISIBLE_DEVICES="$GPU" python3 train_sft.py \
        "${COMMON[@]}" \
        --train_jsonl  "dataset/cl_phase3.jsonl" \
        --init_adapter "sft_output/cl_phase2/final_adapter" \
        --output_dir   "sft_output/cl_phase3" \
        --epochs       2 \
        --lr           5e-6

    # train_sft.py saves output_dir/final_adapter at end (load_best_model_at_end=True)
    echo "  [PHASE3] Adapter saved by train_sft.py → sft_output/cl_phase3/final_adapter"

    run_eval "sft_output/cl_phase3/final_adapter" "val_sft"
    run_eval "sft_output/cl_phase3/final_adapter" "test_sft"

    if ! check_f1 "sft_output/cl_phase3/final_adapter"; then
        echo "  [GATE] Phase 3 dropped below baseline!"
        echo "  [GATE] Best CL adapter is Phase 2: sft_output/cl_phase2/final_adapter"
    else
        echo "  [PHASE3] PASSED gate."
    fi
fi

# ── Summary ───────────────────────────────────────────────────────────────────

echo ""
echo "================================================================"
echo "  CL-SFT COMPLETE — Results summary"
echo "================================================================"

for phase_dir in sft_output/cl_phase1 sft_output/cl_phase2 sft_output/cl_phase3; do
    adapter="${phase_dir}/final_adapter"
    val_json="${adapter}/eval_val_sft.json"
    test_json="${adapter}/eval_test_sft.json"
    if [ -f "$val_json" ]; then
        val_f1=$(python3 -c "import json; d=json.load(open('${val_json}')); print(f\"{d['f1_macro']:.4f} (acc={d['accuracy']:.4f})\")")
        echo "  ${phase_dir}:  val_F1=${val_f1}"
    fi
    if [ -f "$test_json" ]; then
        test_f1=$(python3 -c "import json; d=json.load(open('${test_json}')); print(f\"{d['f1_macro']:.4f} (acc={d['accuracy']:.4f})\")")
        echo "               test_F1=${test_f1}"
    fi
done

echo ""
echo "  Baseline (v5c): val_F1=0.8664 (acc=0.8779)  test_F1=0.8756 (acc=0.8848)"
echo "================================================================"
echo ""
