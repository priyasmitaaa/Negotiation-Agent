#!/usr/bin/env python3
"""
Curriculum GRPO runner for the negotiation agent — legacy Omni-R1 path.
NOT RUNNABLE AS-IS (2026-08-17): the Omni-R1/ reference folder this script
imports from has been removed from the repository, per this project's original
requirement to ship without it (everything else in the active pipeline is this
project's own code — see causal_rm_results.md section 4.1's audit, which
confirmed this script's `from trainer.grpo_trainer import GRPOTrainer` was the
*only* load-bearing dependency on that folder anywhere in the project).

Preserved as a historical/documentation record, not deleted outright, because
it's the actual code that produced the comparison this project's memory-safety
claim rests on: causal_rm_results.md section 4.3's "44.9GB/81.9GB peak memory
vs. Omni-R1's ~100% OOM at num_generations=4" is only meaningful because this
exact script (with gradient_checkpointing=True, Omni-R1's vendored GRPOTrainer)
was the thing that actually hit that OOM. Read it to understand what was being
compared against; don't expect to run it without restoring Omni-R1/ and the
isolated .venv_grpo_pilot venv (trl==0.15.2) it depended on.

Superseded by train_grpo_curriculum.py as the default (see causal_rm_results.md
sections 4.3/4.4): this script's Omni-R1 vendored GRPOTrainer requires
gradient_checkpointing=True to fit num_generations=4 in memory, and that combination
crashes on this machine's transformers/Qwen2.5-Omni audio path (section 4, the
original documented blocker). The trl-native AudioGRPOTrainer route in
train_grpo_curriculum.py avoids the conflict entirely rather than working around it,
with real measured evidence (44.9GB/81.9GB peak memory, 128 stable training steps)
that the Omni-R1 path never had.

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
    make_negotiation_grpo_reward,
)


V2_ROOT = Path(__file__).parent
OMNI_R1_SRC = V2_ROOT / "Omni-R1" / "src"
if str(OMNI_R1_SRC) not in sys.path:
    sys.path.insert(0, str(OMNI_R1_SRC))

# Environment note (2026-08-01): this vendored Omni-R1 trainer is pinned to
# trl==0.15.2 (see Omni-R1/requirements.txt), which must be run in the
# isolated .venv_grpo_pilot venv (`source .venv_grpo_pilot/bin/activate`) —
# the shared environment's system-wide trl==0.24.0 is required by
# unsloth/unsloth_zoo for other work on this machine, so it is intentionally
# NOT downgraded globally. trl==0.15.2's package-availability probe still
# tries to detect the optional `llm_blender` package (an unrelated LLM
# pairwise-judge feature we never use, imported only for
# SyncRefModelCallback's sibling code in trl.trainer.callbacks); the
# system-wide llm_blender install is itself broken against the transformers
# version Qwen2.5-Omni needs (unrelated ABI/API mismatch). Stub it out
# before importing trl so trl's own availability check short-circuits
# instead of triggering llm_blender's broken import chain — we do not use
# any LLM-judge functionality in this training script.
import importlib.machinery
import types as _types
if "llm_blender" not in sys.modules:
    _llm_blender_stub = _types.ModuleType("llm_blender")
    _llm_blender_stub.__spec__ = importlib.machinery.ModuleSpec("llm_blender", loader=None)
    sys.modules["llm_blender"] = _llm_blender_stub

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
    parser.add_argument(
        "--causal_rm_checkpoint",
        default=None,
        help=(
            "Path to a trained causal_reward_model.py checkpoint directory "
            "(rubric_head.pt + config.json). When set, price_strategy/"
            "emotion/progression reward components come from this RM "
            "instead of the keyword heuristics. Default (unset) reproduces "
            "the original heuristic-only reward exactly. Only pass a "
            "checkpoint here after it has passed causal_rm_audit.py's "
            "go/no-go check (causal_rm_architecture.md section 7) — this "
            "flag does not itself validate anything."
        ),
    )
    parser.add_argument("--causal_rm_device", default="cuda")
    parser.add_argument(
        "--no_gradient_checkpointing",
        action="store_true",
        help=(
            "Disable gradient_checkpointing (default: enabled). 2026-08-16 "
            "cheap check: the original GRPO pilot crash (key/value shape "
            "mismatch under gradient_checkpointing+audio) was hit while "
            "fighting another user's job for GPU memory, which is why "
            "checkpointing was turned on in the first place — testing "
            "whether it's actually needed now that both GPUs are free, "
            "before assuming the transformers-level bug needs a real fix."
        ),
    )
    parser.add_argument(
        "--component_log_path",
        default=None,
        help=(
            "If set, append per-completion reward component breakdowns "
            "(JSONL) to this path — needed to watch per-dimension reward "
            "trajectories during a pilot (e.g. detect overall reward "
            "climbing without genuine emotion/price_strategy/progression "
            "improvement, the reward-hacking tell). Defaults to "
            "<output_dir>/reward_components.jsonl when unset."
        ),
    )
    return parser.parse_args()


def load_policy(adapter_path: Path):
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",  # 2026-08-01: SDPA + gradient_checkpointing hits a key/value shape mismatch inside transformers' Qwen2.5-Omni decoder layer on this version; eager sidesteps it
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
    # Environment note (2026-08-01): Omni-R1's vendored GRPOTrainer.__init__
    # does `model.warnings_issued["estimate_tokens"] = True`, a pattern from
    # an older HF `PreTrainedModel` that always initialized `warnings_issued`
    # as an instance dict. The transformers version this project needs for
    # Qwen2.5-Omni support no longer sets that attribute, so GRPOTrainer's
    # own line raises AttributeError before training even starts. Pre-set it
    # here rather than patch the vendored trainer file, so this shim stays
    # visible in our code instead of silently living in a dependency.
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    causal_rm = None
    if args.causal_rm_checkpoint:
        # Deferred import: only pull in causal_reward_model.py's Qwen3-8B
        # loading when a checkpoint is actually requested, so the default
        # (heuristic-only) path never pays for it.
        from causal_reward_model import load_causal_rm

        causal_rm = load_causal_rm(args.causal_rm_checkpoint, device=args.causal_rm_device)
        print(f"Loaded causal reward model from {args.causal_rm_checkpoint} — "
              f"price_strategy/emotion/progression rewards now come from this RM.")

    component_log_path = Path(args.component_log_path) if args.component_log_path else Path(args.output_dir) / "reward_components.jsonl"
    component_log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Logging per-completion reward components to {component_log_path}")
    reward_fn = make_negotiation_grpo_reward(causal_rm=causal_rm, component_log_path=component_log_path)

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
        gradient_checkpointing=not args.no_gradient_checkpointing,
        max_steps=args.max_steps,
        save_strategy="no",
        save_steps=args.save_steps,
        temperature=args.temperature,
        num_generations=args.num_generations,
        run_name="negotiation-curriculum-grpo",
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_fn],
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        processing_class=processor,
        callbacks=[CurriculumCallback(dataset)],
        attn_implementation="eager",  # 2026-08-01: SDPA + gradient_checkpointing hits a key/value shape mismatch inside transformers' Qwen2.5-Omni decoder layer on this version; eager sidesteps it
    )
    trainer.train()
    if not args.skip_save:
        save_adapter_checkpoint(trainer.model, processor, args.output_dir)


if __name__ == "__main__":
    main()
