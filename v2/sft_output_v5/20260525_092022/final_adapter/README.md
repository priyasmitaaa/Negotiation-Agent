# final_adapter/ — v5 Best LoRA Adapter Weights

Production-ready LoRA adapter for SFT v5 (reasoning + decision + response). Load on top of `Qwen/Qwen2.5-Omni-7B` thinker for inference.

## Key difference from v4

v5 was trained with `<reasoning>...</reasoning> <decision>...</decision> <response>...</response>` as the target, making the model's decision process explicit and suitable as an initialisation point for GRPO.

## Files

Same structure as v4 final_adapter: `adapter_config.json`, `adapter_model.safetensors`, tokenizer files.

> See `../../../inference_v5.py` for the full inference pipeline.
