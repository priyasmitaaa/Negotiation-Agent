#!/usr/bin/env python3
"""
train_sft_v3.py
===============
SFT of Qwen2.5-Omni-7B (thinker only) — reasoning-free output.

Target: 3-part structured seller response only —
  <decision>LEVERAGE/MITIGATE/UNDECIDED</decision>
  <seller_vaii>0.05-0.45</seller_vaii>
  <response>...spoken seller text...</response>

Mirrors Omni-R1/src/train_sft.py structure:
  @dataclass DataTrainingArguments → HfArgumentParser → main() →
  load model → extract thinker → LoRA → processor → dataset →
  collate_fn → TrainingArguments → Trainer → train() → save_model()

No val/test — SFT is a warm-up before GRPO. Real evaluation happens
via GRPO reward functions, not SFT cross-entropy loss.

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 Qwen3-tts/v2/train_sft_v3.py

  # override epochs / LR
  CUDA_VISIBLE_DEVICES=0 python3 Qwen3-tts/v2/train_sft_v3.py --epochs 1 --lr 3e-5

  # resume from checkpoint
  CUDA_VISIBLE_DEVICES=0 python3 Qwen3-tts/v2/train_sft_v3.py \\
      --resume_from_checkpoint Qwen3-tts/v2/sft_output_v3/YYYYMMDD_HHMMSS/checkpoint-400

Outputs go to v2/sft_output_v3/{timestamp}/
"""

import datetime
import gc
import json
import logging
import os
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    HfArgumentParser,
)
from peft import LoraConfig, get_peft_model
from peft import PeftModel as _PeftModel

import transformers

from sft_data_v2 import (
    NegotiationV2Dataset,
    build_splits,
    compute_class_weights,
    TARGET_SR,
)

warnings.filterwarnings("ignore")

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_ID          = "Qwen/Qwen2.5-Omni-7B"
OUTPUT_DIR        = Path(__file__).parent / "sft_output_v3"
DEFAULT_AUDIO_DIR = Path(__file__).parent / "tts_outputs"

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
_AUDIO_MODULE_KEYS = (
    "audio_tower", "audio_encoder", "whisper",
    "conv1", "conv2", "encoder.layers", "embed_positions",
)

# v3: no <reasoning> instruction — model learns decision + vaii + response directly
SYSTEM_PROMPT_SFT_V3 = (
    "You are an AI negotiation assistant for a second-hand electronics shop in India. "
    "Guide the negotiation to within 10% of fair market value. Analyse the "
    "conversation and buyer VAII signals (< 0.50 = calm, >= 0.50 = stressed). "
    "VAII (Vocal Affective Intensity Index) reflects the buyer's emotional stress level. "
    "Produce in this exact order: <decision> [UNDECIDED/LEVERAGE/MITIGATE], "
    "<seller_vaii> [0.05-0.45], <response>. Infer everything from raw signals."
)

_RE_REASONING = re.compile(r'<reasoning>.*?</reasoning>\n?', re.DOTALL)


# ── Args dataclass — mirrors Omni-R1's DataTrainingArguments + HfArgumentParser ──

@dataclass
class DataTrainingArguments:
    """
    SFT v3 arguments. Parsed with HfArgumentParser (same pattern as Omni-R1/src/train_sft.py).
    All fields have defaults so the script runs with zero flags.
    """
    model_name_or_path: Optional[str]   = field(default=MODEL_ID,              metadata={"help": "HF model ID or local path"})
    out_dir:            Optional[str]   = field(default=str(OUTPUT_DIR),        metadata={"help": "base output directory"})
    audio_dir:          Optional[str]   = field(default=str(DEFAULT_AUDIO_DIR), metadata={"help": "tts_outputs/ dir with dialogue_XXXX/ subdirs"})
    epochs:             Optional[int]   = field(default=1,                      metadata={"help": "training epochs"})
    batch_size:         Optional[int]   = field(default=1,                      metadata={"help": "per-device train batch size"})
    grad_accum:         Optional[int]   = field(default=16,                     metadata={"help": "gradient accumulation steps"})
    lr:                 Optional[float] = field(default=5e-5,                   metadata={"help": "peak learning rate"})
    max_length:         Optional[int]   = field(default=4096,                   metadata={"help": "max token sequence length"})
    lora_r:             Optional[int]   = field(default=8,                      metadata={"help": "LoRA rank"})
    lora_alpha:         Optional[int]   = field(default=16,                     metadata={"help": "LoRA alpha"})
    lora_dropout:       Optional[float] = field(default=0.1,                    metadata={"help": "LoRA dropout"})
    save_steps:         Optional[int]   = field(default=200,                    metadata={"help": "checkpoint save interval (steps)"})
    eval_steps:         Optional[int]   = field(default=200,                    metadata={"help": "eval interval (steps)"})
    resume_from_checkpoint: Optional[str] = field(default=None,                metadata={"help": "checkpoint path to resume training from"})
    init_adapter:       Optional[str]   = field(default=None,                  metadata={"help": "existing LoRA adapter path as weight init"})


# ── Torch Dataset wrapper ─────────────────────────────────────────────────────

class _TorchDataset(Dataset):
    def __init__(self, inner: NegotiationV2Dataset):
        self._inner = inner
    def __len__(self):
        return len(self._inner)
    def __getitem__(self, idx):
        return self._inner[idx]


# ── Collator (replaces Omni-R1's inline collate_fn) ───────────────────────────
#
# How prompts reach the model — parallel to Omni-R1's collate_fn:
#
#   Omni-R1:
#     combined = f"{prompt_text}\n{solution}"
#     inputs   = processor(text=combined, audio=audio, ...)
#
#   Ours (v3):
#     messages[1]["content"] = user message  ← the prompt
#     messages[2]["content"] = assistant target (reasoning stripped)  ← the solution
#     processor.apply_chat_template(messages) → single string with <|im_start|> tokens
#     labels masked so loss fires only on assistant tokens

@dataclass
class MultimodalSFTCollatorV3:
    processor:  Qwen2_5OmniProcessor
    max_length: int = 4096

    def __post_init__(self):
        asst_header = "<|im_start|>assistant\n"
        self._asst_header_ids = torch.tensor(
            self.processor.tokenizer(asst_header, add_special_tokens=False).input_ids,
            dtype=torch.long,
        )

    def _build_labels(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Mask prompt tokens so loss fires only on assistant output tokens."""
        labels  = torch.full_like(input_ids, -100)
        seq_len = input_ids.size(0)
        hdr     = self._asst_header_ids.to(input_ids.device)
        hdr_len = hdr.size(0)
        for start in range(seq_len - hdr_len, -1, -1):
            if torch.equal(input_ids[start: start + hdr_len], hdr):
                label_start = start + hdr_len
                if label_start < seq_len:
                    labels[label_start:] = input_ids[label_start:]
                break
        else:
            warnings.warn("_build_labels: assistant header not found — zero loss for this sample")
        return labels

    def __call__(self, batch: list[dict]) -> dict:
        all_input_ids, all_attn_mask, all_labels = [], [], []
        all_audio_feats, all_feat_masks = [], []
        max_seq = 0

        processed = []
        for sample in batch:
            messages     = sample["messages"]
            audio_arrays = sample["audio_arrays"]

            def _adapt_v3(msg):
                role    = msg["role"]
                content = msg["content"]
                if role == "system":
                    content = SYSTEM_PROMPT_SFT_V3
                elif role == "assistant":
                    content = _RE_REASONING.sub('', content).strip()
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]
                return {"role": role, "content": content}

            messages_norm = [_adapt_v3(m) for m in messages]
            text = self.processor.apply_chat_template(
                messages_norm, tokenize=False, add_generation_prompt=False,
            )

            try:
                inputs = self.processor(
                    text=text,
                    audio=audio_arrays if audio_arrays else None,
                    sampling_rate=TARGET_SR,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=self.max_length,
                )
            except Exception:
                inputs = self.processor(
                    text=text,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=self.max_length,
                )

            input_ids = inputs["input_ids"].squeeze(0)
            attn_mask = inputs["attention_mask"].squeeze(0)
            labels    = self._build_labels(input_ids)

            processed.append({
                "input_ids":      input_ids,
                "attention_mask": attn_mask,
                "labels":         labels,
                "audio_features": inputs.get("input_features"),
                "feat_attn_mask": inputs.get("feature_attention_mask"),
            })
            max_seq = max(max_seq, input_ids.size(0))

        pad_id = self.processor.tokenizer.pad_token_id or 0
        for p in processed:
            pad_len = max_seq - p["input_ids"].size(0)
            if pad_len > 0:
                p["input_ids"]      = F.pad(p["input_ids"],      (0, pad_len), value=pad_id)
                p["attention_mask"] = F.pad(p["attention_mask"], (0, pad_len), value=0)
                p["labels"]         = F.pad(p["labels"],         (0, pad_len), value=-100)
            all_input_ids.append(p["input_ids"])
            all_attn_mask.append(p["attention_mask"])
            all_labels.append(p["labels"])
            if p["audio_features"] is not None:
                all_audio_feats.append(p["audio_features"])
                all_feat_masks.append(p["feat_attn_mask"])

        batch_out = {
            "input_ids":      torch.stack(all_input_ids),
            "attention_mask": torch.stack(all_attn_mask),
            "labels":         torch.stack(all_labels),
            "decision_ints":  torch.tensor(
                [s["decision_int"] for s in batch], dtype=torch.long
            ),
        }

        if all_audio_feats:
            max_T = max(f.shape[-1] for f in all_audio_feats)
            p_feats, p_masks = [], []
            for feat, mask in zip(all_audio_feats, all_feat_masks):
                pad_T = max_T - feat.shape[-1]
                if pad_T > 0:
                    feat = F.pad(feat, (0, pad_T), value=0.0)
                    if mask is not None:
                        mask = F.pad(mask, (0, pad_T), value=0)
                p_feats.append(feat)
                p_masks.append(mask)
            batch_out["input_features"] = torch.cat(p_feats, dim=0)
            if any(m is not None for m in p_masks):
                batch_out["feature_attention_mask"] = torch.cat(
                    [m for m in p_masks if m is not None], dim=0
                )
        return batch_out


# ── Weighted loss trainer ──────────────────────────────────────────────────────

class WeightedLossTrainer(Trainer):
    def __init__(self, *args, class_weights: dict = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.cw = class_weights or {"LEVERAGE": 1.0, "MITIGATE": 1.0}
        self._int_to_dec = {0: "LEVERAGE", 1: "MITIGATE", 2: "UNDECIDED"}

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items()
                           if k not in ("labels", "decision_ints")})
        logits  = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        vocab_size   = shift_logits.size(-1)

        loss_fct   = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        token_loss = loss_fct(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
        )
        B, T = shift_labels.shape
        token_loss = token_loss.view(B, T)

        sample_weights = torch.ones(B, device=token_loss.device)
        if "decision_ints" in inputs:
            for i, di in enumerate(inputs["decision_ints"]):
                dec = self._int_to_dec.get(int(di), "LEVERAGE")
                sample_weights[i] = self.cw.get(dec, 1.0)

        mask = (shift_labels != -100).float()
        loss = (token_loss * mask * sample_weights.unsqueeze(1)).sum() / mask.sum().clamp(min=1)
        return (loss, outputs) if return_outputs else loss


# ── Progress callback ──────────────────────────────────────────────────────────

class ProgressCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            step  = state.global_step
            total = state.max_steps
            loss  = logs.get("loss", logs.get("train_loss", "?"))
            pct   = 100 * step / total if total else 0
            logging.getLogger(__name__).info(
                f"Step {step}/{total} ({pct:.1f}%)  loss={loss}"
            )

    def on_save(self, args, state, control, **kwargs):
        logging.getLogger(__name__).info(
            f"Checkpoint saved at step {state.global_step}"
        )


# ── Main — mirrors Omni-R1's main() step order ────────────────────────────────

def main():
    # 1. Parse args — same pattern as Omni-R1's HfArgumentParser(DataTrainingArguments)
    parser = HfArgumentParser(DataTrainingArguments)
    args   = parser.parse_args_into_dataclasses()[0]

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    transformers.logging.set_verbosity_info()
    log = logging.getLogger(__name__)
    log.info(args)

    _run_tag   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.out_dir) / _run_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    _fh = logging.FileHandler(output_dir / "sft.log")
    _fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(_fh)
    log.info(f"Run tag: {_run_tag}  |  Output: {output_dir}")

    audio_dir = Path(args.audio_dir)
    if not audio_dir.exists():
        raise SystemExit(f"\nERROR: audio directory not found: {audio_dir}\n")

    print(f"\n{'='*65}")
    print(f"  SFT v3 — Qwen2.5-Omni-7B  |  bf16 LoRA  |  1 epoch")
    print(f"  Audio+Text → <decision> <seller_vaii> <response>")
    print(f"{'='*65}\n")

    # 2. Splits + class weights
    splits        = build_splits()
    class_weights = compute_class_weights(splits)
    print(f"  Class weights: LEVERAGE={class_weights['LEVERAGE']:.4f}  "
          f"MITIGATE={class_weights['MITIGATE']:.4f}")

    # 3. Datasets
    print("  Building datasets…")
    train_inner = NegotiationV2Dataset(splits["train"], audio_dir, "train")
    val_inner   = NegotiationV2Dataset(splits["val"],   audio_dir, "val")
    test_inner  = NegotiationV2Dataset(splits["test"],  audio_dir, "test")

    if len(train_inner) == 0:
        raise SystemExit("ERROR: 0 training examples. Check audio_dir and intermediate files.")

    train_ds = _TorchDataset(train_inner)
    val_ds   = _TorchDataset(val_inner)
    test_ds  = _TorchDataset(test_inner)
    print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")

    # 4. Processor — same as Omni-R1's Qwen2_5OmniProcessor.from_pretrained(...)
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    print(f"  Loading processor: {args.model_name_or_path}")
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_name_or_path, token=hf_token)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.pad_token_id = processor.tokenizer.pad_token_id
    processor.eos_token_id = processor.tokenizer.eos_token_id

    # 5. Load model — mirrors Omni-R1: Qwen2_5OmniForConditionalGeneration.from_pretrained(...)
    print(f"  Loading Qwen2.5-Omni in bf16…")
    _full = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        enable_audio_output=False,
        token=hf_token,
    )

    # 6. Extract thinker — verbatim from Omni-R1
    if hasattr(_full, "module") and hasattr(_full.module, "thinker"):
        model = _full.module.thinker
    elif hasattr(_full, "thinker"):
        model = _full.thinker
    print(f"  Using model component: {type(model).__name__}")
    model.config.use_cache = False
    del _full; gc.collect()

    # 7. Apply LoRA — Omni-R1 has this commented out; we keep it enabled
    if args.init_adapter:
        print(f"  Loading LoRA init from {args.init_adapter}…")
        model = _PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=None,
        )
        model = get_peft_model(model, lora_cfg)

    # Freeze audio encoder — LoRA attaches to all linear layers including audio
    frozen = 0
    for name, param in model.named_parameters():
        if any(k in name.lower() for k in _AUDIO_MODULE_KEYS):
            param.requires_grad = False
            frozen += param.numel()
    if frozen:
        print(f"  Audio encoder params re-frozen: {frozen:,}")

    # Print trainable parameters — mirrors Omni-R1's print block
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params       = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {trainable_params:,} ({100 * trainable_params / all_params:.2f}%)")

    # 8. Collator — replaces Omni-R1's inline collate_fn
    collator = MultimodalSFTCollatorV3(processor=processor, max_length=args.max_length)

    # 9. TrainingArguments — mirrors Omni-R1
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        seed=42,
        data_seed=42,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        logging_steps=1,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        report_to="tensorboard",
        run_name="SFT-v3-no-reasoning",
        optim="adamw_torch",
    )

    # 10. Trainer — mirrors Omni-R1's Trainer(...)
    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        class_weights=class_weights,
        callbacks=[ProgressCallback()],
    )

    # 11. Train — trainer.train() mirrors Omni-R1
    print(f"\n  Starting training…")
    print(f"  Effective batch : {args.batch_size * args.grad_accum}")
    print(f"  Epochs          : {args.epochs}")
    print(f"  LR              : {args.lr}\n")

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # 12. Save adapter — trainer.save_model() mirrors Omni-R1
    final = output_dir / "final_adapter"
    model.save_pretrained(final)
    processor.save_pretrained(final)
    print(f"  Adapter saved → {final}")

    # 13. Test evaluation
    print("  Evaluating on held-out test set…")
    test_results = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")
    print(f"  Test loss: {test_results.get('test_loss', 'n/a'):.4f}")

    # 14. Training log + run summary
    log_history = trainer.state.log_history
    (output_dir / "training_log.json").write_text(json.dumps(log_history, indent=2))

    train_steps = [e for e in log_history if "loss" in e and "eval_loss" not in e]
    eval_steps  = [e for e in log_history if "eval_loss" in e]
    summary = {
        "completed_at":               datetime.datetime.now().isoformat(),
        "model_id":                   args.model_name_or_path,
        "lora_r":                     args.lora_r,
        "lora_alpha":                 args.lora_alpha,
        "epochs":                     args.epochs,
        "lr":                         args.lr,
        "batch_size":                 args.batch_size,
        "grad_accum":                 args.grad_accum,
        "effective_batch":            args.batch_size * args.grad_accum,
        "train_examples":             len(train_ds),
        "skipped_no_reasoning_train": train_inner.skipped_no_reasoning,
        "skipped_no_audio_train":     train_inner.skipped_no_audio,
        "class_weights":              class_weights,
        "final_train_loss":           train_steps[-1]["loss"] if train_steps else None,
        "best_eval_loss":             min((e["eval_loss"] for e in eval_steps), default=None),
        "test_loss":                  test_results.get("test_loss"),
        "total_steps":                trainer.state.global_step,
        "adapter_path":               str(final),
        "run_tag":                    _run_tag,
        "version":                    "v3-no-reasoning",
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{'='*65}\n  DONE  →  {output_dir}\n{'='*65}\n")


if __name__ == "__main__":
    main()
