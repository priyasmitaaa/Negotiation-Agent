# sft_output_v4/ — SFT v4 Training Run Outputs

All outputs from the v4 SFT training run on `Qwen2.5-Omni-7B` with LoRA.

**v4 objective:** Train the model to output `<decision> [LEVERAGE/MITIGATE/UNDECIDED]` + `<response>` given the conversation history with buyer emotion labels in context. No reasoning in the target.

## Run directory

There is one timestamped run directory: `20260525_014244/` (training started 2026-05-25 01:42).

## Results summary

- **Decision accuracy: 96.9%** (523/540 test examples)
- **Speech output: 540/540 WAVs** generated successfully
- **Judge score: 4.03/5.0** average overall quality
