#!/usr/bin/env python3
"""
Curriculum GRPO runner for the negotiation agent — default path (2026-08-17).

Uses AudioGRPOTrainer (trl_audio_grpo_trainer.py), a thin subclass of trl's own
stock GRPOTrainer, and runs in the shared environment's plain Python — no vendored
third-party trainer code, no isolated venv. See causal_rm_results.md sections 4.3
(plumbing/memory: 44.9GB/81.9GB peak at num_generations=4, vs. Omni-R1's ~100% OOM
at the same setting) and 4.4 (training-stability: 128 real steps with the real
causal RM, reward/entropy/grad_norm all stable) for the full evidence trail behind
why this replaced the Omni-R1 path as default.

The previous default, train_grpo_curriculum_omni_r1.py, still exists unchanged and
still works (Omni-R1's vendored GRPOTrainer, isolated .venv_grpo_pilot, trl==0.15.2)
— kept, not deleted, since removing/deprecating it outright is a separate decision
(causal_rm_results.md section 4.1's IP-cleanliness question) not yet made. Use it
only if this script hits a problem this one hasn't; the causal RM / reward
function / dataset code (grpo_curriculum.py) is shared between both.

Run with: python3 train_grpo_curriculum.py [args]
(plain system python3, NOT .venv_grpo_pilot/bin/python3 — that venv's ancient
trl==0.15.2 predates the steps_per_generation/chunking machinery this script relies
on, and is now only needed by train_grpo_curriculum_omni_r1.py)
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch
from peft import PeftModel
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, TrainerCallback
from trl import GRPOConfig

# Match the existing SFT/inference/GRPO scripts on this machine (same shim as
# train_grpo_curriculum.py — torch.load speaker-loading CVE guard fires even
# though we never train the talker).
import transformers.utils.import_utils as _triu
_triu.check_torch_load_is_safe = lambda: None
import transformers.models.qwen2_5_omni.modeling_qwen2_5_omni as _qwen_model
_qwen_model.check_torch_load_is_safe = lambda: None

from grpo_curriculum import (
    CurriculumCallback,
    NegotiationGRPODataset,
    PeriodicEvalCallback,
    build_fixed_eval_slice,
    make_negotiation_grpo_reward,
)
from trl_audio_grpo_trainer import AudioGRPOTrainer  # environment shims live here


MODEL_ID = "Qwen/Qwen2.5-Omni-7B"
DEFAULT_ADAPTER = Path(__file__).parent / "sft_output_v6" / "20260614_010317" / "final_adapter"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--output_dir", default=str(Path(__file__).parent / "grpo_output_trl_native"))
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_prompt_length", type=int, default=4096)
    parser.add_argument("--max_completion_length", type=int, default=128)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--steps_per_generation", type=int, default=None,
                         help="trl-native only: how many training steps' worth of "
                              "completions to buffer per generation call. Defaults "
                              "to gradient_accumulation_steps if unset (trl's own "
                              "default). This is the mechanism from causal_rm_results.md "
                              "section 4.2/4.3 that should let num_generations=4 avoid "
                              "the OOM Omni-R1's trainer hit.")
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=50,
                         help="Run a small held-out decision-accuracy check every N "
                              "steps (0 disables). Reward/entropy/grad_norm logging "
                              "alone can't distinguish genuine improvement from a "
                              "policy finding a soft spot in the causal RM's scalar "
                              "score — see causal_rm_results.md section 4.4's "
                              "discussion of the step-32 pattern.")
    parser.add_argument("--eval_n", type=int, default=20,
                         help="Number of fixed held-out val examples for --eval_steps.")
    parser.add_argument("--no_audio_input", action="store_true")
    parser.add_argument("--skip_save", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                         help="Off by default here, unlike the legacy "
                              "train_grpo_curriculum_omni_r1.py (which requires it) — "
                              "the whole point of the trl-native route is that "
                              "steps_per_generation buffering makes it unnecessary at "
                              "real pilot scale (verified: causal_rm_results.md "
                              "section 4.3). Pass this flag to re-enable it if memory "
                              "still needs it.")
    parser.add_argument(
        "--causal_rm_checkpoint",
        default=None,
        help="Path to a trained causal_reward_model.py checkpoint directory. Only "
             "pass a checkpoint here after it has passed causal_rm_audit.py's "
             "go/no-go check (causal_rm_architecture.md section 7) — this flag does "
             "not itself validate anything.",
    )
    parser.add_argument("--causal_rm_device", default="cuda")
    parser.add_argument("--component_log_path", default=None)
    return parser.parse_args()


def load_policy(adapter_path: Path):
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        enable_audio_output=False,
    )
    model.thinker = PeftModel.from_pretrained(model.thinker, str(adapter_path), is_trainable=True)
    model.config.use_cache = False
    model.train()
    gc.collect()
    return model


def save_adapter_checkpoint(thinker, processor, output_dir: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    policy = getattr(thinker, "module", thinker)
    policy.save_pretrained(output_path)
    processor.save_pretrained(output_path)


class PeriodicCheckpointCallback(TrainerCallback):
    """GRPOConfig is constructed with save_strategy="no" below (HF Trainer's
    generic step-based checkpointing isn't exercised/tested against a bare
    model.thinker submodule in this setup) — so --save_steps needs its own
    mechanism to actually do anything. Reuses save_adapter_checkpoint (the same,
    already-proven function used at the end of training) rather than a new,
    untested save path, so a crash mid-run doesn't lose everything."""

    def __init__(self, processor, output_dir: str, save_steps: int):
        self.processor = processor
        self.output_dir = output_dir
        self.save_steps = save_steps

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if self.save_steps <= 0 or state.global_step == 0 or state.global_step % self.save_steps != 0:
            return
        ckpt_dir = Path(self.output_dir) / f"checkpoint-{state.global_step}"
        save_adapter_checkpoint(model, self.processor, str(ckpt_dir))
        print(f"[checkpoint] saved step {state.global_step} to {ckpt_dir}")


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
    # AudioGRPOTrainer (like stock trl's GRPOTrainer) expects a plain causal-LM
    # model with a normal forward(input_ids=..., attention_mask=..., ...) signature
    # — confirmed directly (2026-08-16) that passing the full
    # Qwen2_5OmniForConditionalGeneration wrapper instead raises
    # "_forward_unimplemented() got an unexpected keyword argument 'input_ids'".
    # Pass model.thinker (the PeftModel-wrapped causal LM) directly, unlike
    # train_grpo_curriculum_omni_r1.py which passes the full wrapper to Omni-R1's
    # trainer (which has its own internal .thinker handling stock trl doesn't).
    thinker = model.thinker
    if not hasattr(thinker, "warnings_issued"):
        thinker.warnings_issued = {}

    causal_rm = None
    if args.causal_rm_checkpoint:
        from causal_reward_model import load_causal_rm

        causal_rm = load_causal_rm(args.causal_rm_checkpoint, device=args.causal_rm_device)
        print(f"Loaded causal reward model from {args.causal_rm_checkpoint} — "
              f"price_strategy/emotion/progression rewards now come from this RM.")

    component_log_path = Path(args.component_log_path) if args.component_log_path else Path(args.output_dir) / "reward_components.jsonl"
    component_log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Logging per-completion reward components to {component_log_path}")
    reward_fn = make_negotiation_grpo_reward(causal_rm=causal_rm, component_log_path=component_log_path)

    callbacks = [CurriculumCallback(dataset)]
    if args.eval_steps > 0:
        val_dataset = NegotiationGRPODataset(
            split="val",
            include_audio=not args.no_audio_input,
            stage="easy",
        )
        eval_slice = build_fixed_eval_slice(val_dataset, n=args.eval_n)
        eval_log_path = Path(args.output_dir) / "held_out_eval.jsonl"
        callbacks.append(PeriodicEvalCallback(eval_slice, eval_steps=args.eval_steps, log_path=eval_log_path))
        print(f"Periodic held-out decision-accuracy eval every {args.eval_steps} steps "
              f"(n={len(eval_slice)}), logging to {eval_log_path}")
    if args.save_steps > 0 and not args.skip_save:
        callbacks.append(PeriodicCheckpointCallback(processor, args.output_dir, args.save_steps))

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        seed=42,
        data_seed=42,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        steps_per_generation=args.steps_per_generation,
        logging_steps=args.logging_steps,
        bf16=True,
        report_to="none",
        gradient_checkpointing=args.gradient_checkpointing,
        max_steps=args.max_steps,
        save_strategy="no",
        save_steps=args.save_steps,
        temperature=args.temperature,
        num_generations=args.num_generations,
        run_name="negotiation-curriculum-grpo-trl-native",
    )

    trainer = AudioGRPOTrainer(
        model=thinker,
        reward_funcs=[reward_fn],
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        processing_class=processor,
        callbacks=callbacks,
    )
    trainer.train()
    if not args.skip_save:
        save_adapter_checkpoint(trainer.model, processor, args.output_dir)


if __name__ == "__main__":
    main()
