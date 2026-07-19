# sft_output_v5/ — SFT v5 Training Run Outputs

All outputs from the v5 SFT training run on `Qwen2.5-Omni-7B` with LoRA.

**v5 objective:** Extend v4 by adding chain-of-thought reasoning to the target. The model now outputs `<reasoning> ... </reasoning> <decision> [LEVERAGE/MITIGATE/UNDECIDED] <response> ...` — making the decision process interpretable and providing a stronger training signal for GRPO.

## Run directory

One timestamped run: `20260525_092022/` (training started 2026-05-25 09:20).

## v4 vs v5 comparison

| Aspect | v4 | v5 |
|---|---|---|
| Target format | `<decision> <response>` | `<reasoning> <decision> <response>` |
| Best eval loss | 0.4100 | 0.8095 (higher due to longer target) |
| Decision accuracy | 96.9% | TBD (speech inference running) |
| Speech inference | ✅ Complete | 🔄 In progress |
