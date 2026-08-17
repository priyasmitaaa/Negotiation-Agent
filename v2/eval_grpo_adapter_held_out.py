#!/usr/bin/env python3
"""
Standalone, larger held-out decision-accuracy eval for a trained GRPO adapter.

Built to answer one question: did the 300-step run's final decision_accuracy=1.0
(n=20, the small slice used *during* training for cheap periodic checks) hold up
on a meaningfully larger, still-practical sample — or was it noise from a small
slice? Pulls from the same canonical val split (splits_v6.json via
NegotiationGRPODataset) the training-time eval used, just a larger fixed prefix
of it, so this is a scaled-up version of the same check, not a different one.

Usage:
  python3 eval_grpo_adapter_held_out.py --adapter <dir> --n 100 [--split val]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import transformers.utils.import_utils as _triu
_triu.check_torch_load_is_safe = lambda: None
import transformers.models.qwen2_5_omni.modeling_qwen2_5_omni as _qwen_model
_qwen_model.check_torch_load_is_safe = lambda: None

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from peft import PeftModel

from grpo_curriculum import NegotiationGRPODataset, build_fixed_eval_slice, evaluate_decision_accuracy

MODEL_ID = "Qwen/Qwen2.5-Omni-7B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, help="Path to the trained LoRA adapter directory")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--no_audio_input", action="store_true")
    parser.add_argument("--output_json", default=None,
                         help="Optional path to write a JSON summary (default: "
                              "<adapter>/held_out_eval_n<n>.json)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter_path = Path(args.adapter)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")

    print(f"Loading base model + adapter from {adapter_path} ...")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="eager",
        low_cpu_mem_usage=True, enable_audio_output=False,
    )
    model.thinker = PeftModel.from_pretrained(model.thinker, str(adapter_path))
    model.config.use_cache = True
    model.thinker.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "left"
    processor.pad_token_id = processor.tokenizer.pad_token_id
    processor.eos_token_id = processor.tokenizer.eos_token_id

    print(f"Building fixed eval slice: split={args.split}, n={args.n} ...")
    dataset = NegotiationGRPODataset(split=args.split, include_audio=not args.no_audio_input)
    eval_slice = build_fixed_eval_slice(dataset, n=args.n)
    print(f"Slice built: {len(eval_slice)} examples "
          f"(requested {args.n}; dataset has {len(dataset.examples)} total for this split)")

    print("Running greedy-decode eval (this is the slow part — one generate() call per example) ...")
    acc = evaluate_decision_accuracy(model.thinker, processor, eval_slice, model.thinker.device)

    result = {
        "adapter": str(adapter_path),
        "split": args.split,
        "n_requested": args.n,
        "n_actual": len(eval_slice),
        "decision_accuracy": acc,
    }
    print(json.dumps(result, indent=2))

    output_path = Path(args.output_json) if args.output_json else adapter_path / f"held_out_eval_n{len(eval_slice)}.json"
    output_path.write_text(json.dumps(result, indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
