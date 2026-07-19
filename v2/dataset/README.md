# dataset/ — Canonical v2 Negotiation Dialogue Dataset (3,154 dialogues)

This is the **source of truth** for all v2+ SFT training and evaluation. Contains restructured negotiation dialogues in JSON format, one file per dialogue.

## What each dialogue contains

Each `dialogue_XXXX.json` has:
- `seed`: Original metadata — domain, pricing (fair value, asking price, anchor range), buyer profile, provenance (generation model, template, scenario label)
- `processed`: List of conversation turns with speaker, text, emotion labels, intensity, valence, and optionally seller reasoning
- `bias_setup`: Bias configuration for LEVERAGE/MITIGATE labelling
- `cl`: Curriculum learning tier assignment (easy/medium/hard)
- `final_price`: The agreed transaction price

## Scale

- **3,154 dialogue files** covering second-hand electronics (flagship smartphones, mid-range phones, earbuds, etc.)
- Products priced from ~₹1k to ~₹50k
- Buyer profiles: college students, professionals, retirees, parents, etc.
- Scenarios generated from 20 templates × multiple price ranges

## Training/test split

Splits are defined in `../splits_v4.json` (90 test dialogues = 540 turns, 4,224 train examples).

> **This is the live dataset.** The v1 dataset in `../../dataset/` is superseded.
