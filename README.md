# Qwen3-TTS Negotiation Agent

Bias-aware negotiation agent trained on Qwen2.5-Omni-7B using SFT and GRPO reinforcement learning.

The agent acts as a second-hand electronics shop seller in India, analysing buyer emotion and price signals to decide when to LEVERAGE, MITIGATE, or remain UNDECIDED — then generating a spoken response via the model's built-in TTS.

## Project layout

```
Qwen3-tts/
├── dataset/          # v1 SFT dataset (train/val/test splits + backups)
├── preprocessed/     # Raw per-template per-range dialogue JSON files (pre-dataset-build)
├── logs/             # Eval and build logs from v1 pipeline runs
├── sft_output/       # (legacy) v1 SFT checkpoint output — superseded by v2/sft_output_v*
└── v2/               # Active workspace — all v2+ training, inference, and evaluation
```

## Versioning

| Version | Description |
|---|---|
| v1 (root) | Initial SFT with VAII bias labels, emotion in context, no reasoning target. Superseded. |
| v2 (`v2/`) | Restructured dataset (3,154 dialogues), multiple SFT variants (v4, v5), speech inference, GRPO reward design. **Active.** |

## Model

Base: `Qwen/Qwen2.5-Omni-7B` — a multimodal model with a thinker (text) + talker (TTS) head. Fine-tuned with LoRA (r=8, α=16) on 4,224 training examples across 3 epochs.
