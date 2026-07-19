# intermediate/ — Dialogues with Seller Reasoning Annotations

Working copies of the dialogue JSONs from `../dataset/` with seller reasoning text written in. These are produced by `../generate_reasoning.py` using a larger LLM (Qwen3-30B-A3B or similar) to annotate each seller turn with a `reasoning` field explaining the decision logic.

## Why it exists

SFT v5 trains the model to produce `<reasoning> <decision> <response>`. The reasoning field in these files is the training target for the reasoning component.

## Contents

Same structure as `../dataset/` — 3,154 `dialogue_XXXX.json` files — but with `reasoning` fields populated inside each seller turn in the `processed` list.

## Status

Reasoning generation ran as a long background job (logged in `../reasoning_generation.log`). Files here may be partially annotated — check `../reasoning_status.sh` for completion status.

> These are consumed by `../sft_data_v4.py` / `../sft_data_v5.py` to build the final SFT JSONL.
