#!/usr/bin/env python3
"""
Curriculum GRPO runner for the negotiation agent.

This is speech-conditioned GRPO: buyer speech/audio is part of the prompt, but
the rollout completion is text (`<decision>` + `<response>`). Speech waveform
generation is evaluated offline at checkpoints because generating WAVs inside
every GRPO rollout would be prohibitively slow.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from trl import GRPOConfig

# Match the existing SFT/inference scripts on this machine. The installed torch
# is 2.5.1, while recent transformers guards torch.load speaker loading behind
# a 2.6+ CVE check even when we do not train the talker.
import transformers.utils.import_utils as _triu
_triu.check_torch_load_is_safe = lambda: None
import transformers.models.qwen2_5_omni.modeling_qwen2_5_omni as _qwen_model
_qwen_model.check_torch_load_is_safe = lambda: None

from grpo_curriculum import (
    CurriculumCallback,
    NegotiationGRPODataset,
    negotiation_grpo_reward,
)


V2_ROOT = Path(__file__).parent
OMNI_R1_SRC = V2_ROOT / "Omni-R1" / "src"
if str(OMNI_R1_SRC) not in sys.path:
    sys.path.insert(0, str(OMNI_R1_SRC))

from trainer.grpo_trainer import GRPOTrainer  # noqa: E402


MODEL_ID = "Qwen/Qwen2.5-Omni-7B"
DEFAULT_ADAPTER = V2_ROOT / "sft_output_v6" / "20260614_010317" / "final_adapter"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--output_dir", default=str(V2_ROOT / "grpo_output_curriculum"))
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_prompt_length", type=int, default=4096)
    parser.add_argument("--max_completion_length", type=int, default=128)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--no_audio_input", action="store_true")
    parser.add_argument("--skip_save", action="store_true")
    return parser.parse_args()


def load_policy(adapter_path: Path):
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        enable_audio_output=False,
    )
    model.thinker = PeftModel.from_pretrained(model.thinker, str(adapter_path), is_trainable=True)
    model.config.use_cache = False
    model.train()
    gc.collect()
    return model


def save_adapter_checkpoint(model, processor, output_dir: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    policy = getattr(model, "module", model)
    policy.thinker.save_pretrained(output_path)
    processor.save_pretrained(output_path)


def main() -> None:
    args = parse_args()
    adapter_path = Path(args.adapter)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")

    dataset = NegotiationGRPODataset(
        split=args.split,
        include_audio=not args.no_audio_input,
        stage="easy",
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "left"
    processor.pad_token_id = processor.tokenizer.pad_token_id
    processor.eos_token_id = processor.tokenizer.eos_token_id

    model = load_policy(adapter_path)

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        seed=42,
        data_seed=42,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        bf16=True,
        report_to="none",
        gradient_checkpointing=False,
        max_steps=args.max_steps,
        save_strategy="no",
        save_steps=args.save_steps,
        temperature=args.temperature,
        num_generations=args.num_generations,
        run_name="negotiation-curriculum-grpo",
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[negotiation_grpo_reward],
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        processing_class=processor,
        callbacks=[CurriculumCallback(dataset)],
        attn_implementation="sdpa",
    )
    trainer.train()
    if not args.skip_save:
        save_adapter_checkpoint(trainer.model, processor, args.output_dir)


if __name__ == "__main__":
    main()
