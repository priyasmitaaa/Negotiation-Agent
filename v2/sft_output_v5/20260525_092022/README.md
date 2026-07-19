# 20260525_092022/ — v5 SFT Run (Started 2026-05-25 09:20)

The v5 training run, which adds chain-of-thought reasoning to the model's output target.

## Training config

| Param | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-Omni-7B` |
| LoRA r / α | 8 / 16 |
| Epochs | 3 |
| LR | 2e-5, warmup 5% |
| Train examples | 4,224 |
| Total steps | 792 |
| Best eval loss | 0.8095 |
| Final train loss | 10.24 (longer targets than v4) |

## Contents

| File / Folder | Description |
|---|---|
| `checkpoint-600/` | LoRA adapter at step 600 |
| `checkpoint-700/` | LoRA adapter at step 700 |
| `checkpoint-792/` | LoRA adapter at step 792 (final step) |
| `final_adapter/` | **Best checkpoint** — use this for inference |
| `sft.log` | Full training log |
| `training_log.json` | Loss curve and eval metrics |
| `run_summary.json` | Run metadata and final metrics |
| `inference_20260526_210501.json` | Early small-scale text inference (pilot, ~50 samples) |
| `inference_20260605_194041.json` | **Full text-only inference** — 540 examples |
| `judge_20260606_012143.json` | Judge scores for text inference |
| `error_analysis_20260605_235207.md` | Error analysis for text inference |
| `speech_outputs_20260612_001937/` | 🔄 **v5 speech WAVs** — currently being generated (inference running) |

## Status

Speech inference (`inference_v5.py --speech --speaker Chelsie`) is currently running on GPU 1. Expected completion ~10–11 AM IST 2026-06-12.
