# final_adapter/ — v4 Best LoRA Adapter Weights

The production-ready LoRA adapter for SFT v4. This is what gets loaded for all inference, judging, and evaluation.

## What's here

| File | Description |
|---|---|
| `adapter_config.json` | LoRA config (r=8, α=16, target modules, base model path) |
| `adapter_model.safetensors` | The LoRA delta weights (load on top of `Qwen/Qwen2.5-Omni-7B`) |
| `tokenizer.json` / `vocab.json` / `merges.txt` | Tokenizer files (copied from base model for self-contained loading) |
| `tokenizer_config.json` / `special_tokens_map.json` / `added_tokens.json` | Tokenizer config |
| `preprocessor_config.json` / `video_preprocessor_config.json` | Multimodal preprocessor config |
| `chat_template.jinja` | Chat template used during training |
| `README.md` | This file |

## How to load

```python
from peft import PeftModel
from transformers import Qwen2_5OmniForConditionalGeneration

base = Qwen2_5OmniForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-Omni-7B")
model = PeftModel.from_pretrained(base.thinker, "path/to/final_adapter")
```

> See `../../inference_v4.py` or `../../inference_v5.py` for the full inference pipeline.