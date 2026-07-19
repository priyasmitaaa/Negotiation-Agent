# 20260525_014244/ — v4 SFT Run (Started 2026-05-25 01:42)

The canonical v4 training run. All evaluation has been completed against this run.

## Training config

| Param | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-Omni-7B` |
| LoRA r / α | 8 / 16 |
| Epochs | 3 |
| LR | 2e-5, warmup 5% |
| Train examples | 4,224 |
| Total steps | 792 |
| Best eval loss | 0.4100 |

## Contents

| File / Folder | Description |
|---|---|
| `checkpoint-600/` | LoRA adapter weights at step 600 |
| `checkpoint-700/` | LoRA adapter weights at step 700 |
| `checkpoint-792/` | LoRA adapter weights at step 792 (final step) |
| `final_adapter/` | **Best checkpoint** (saved by trainer as final) — use this for inference |
| `sft.log` | Full training log |
| `training_log.json` | Structured loss curve and eval metrics per step |
| `run_summary.json` | Key run metadata (model, params, timestamps, final metrics) |
| `inference_20260604_225504.json` | **Text-only inference results** — 540 test examples, 96.9% accuracy |
| `inference_speech_20260606_035020.json` | **Full speech inference results** — 540 examples + 540 WAVs, 96.9% accuracy ← canonical |
| `speech_outputs_20260606_035020/` | **540 WAV files** — model-generated seller speech for all test turns ← canonical |
| `v4_judge_text_and_speech_20260611_220937.json` | **Combined judge scores** (text + speech) — 540/540 judged, avg 4.03/5.0 ← canonical |
| `v4_error_analysis_text_and_speech_20260611_220737.md` | **Combined error analysis** — decision errors, emotion breakdowns, GRPO reward design ← canonical |

> Files not listed above (old pilot runs, superseded judges) are kept for traceability but are not the canonical evaluation artefacts.
