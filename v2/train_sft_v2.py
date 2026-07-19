#!/usr/bin/env python3
"""
train_sft_v2.py
===============
LoRA SFT of Qwen2.5-Omni-7B on v2 negotiation dataset.

Target: 4-part structured seller response —
  <reasoning>...</reasoning>
  <decision>LEVERAGE/MITIGATE/UNDECIDED</decision>
  <seller_vaii>0.05-0.45</seller_vaii>
  <response>...spoken seller text...</response>

Input:  v2/intermediate/dialogue_XXXX.json  (3,154 dialogues)
        v2/tts_outputs/dialogue_XXXX/turn_NN_{buyer|seller}.wav  (REQUIRED)
Output: v2/sft_output_v2/final_adapter/

Split:  80/10/10 train/val/test  (stratified by LEVERAGE/MITIGATE)
Weights: Inverse-frequency class balance (LEVERAGE upweighted ~1.45×)

Usage:
  # Single GPU bf16 LoRA (A100 80GB)
  CUDA_VISIBLE_DEVICES=1 python3 v2/train_sft_v2.py --no_qlora

  # QLoRA (smaller GPU)
  CUDA_VISIBLE_DEVICES=1 python3 v2/train_sft_v2.py

  # Resume
  CUDA_VISIBLE_DEVICES=1 python3 v2/train_sft_v2.py --no_qlora \\
      --resume_from_checkpoint v2/sft_output_v2/checkpoint-500
"""

import argparse
import gc
import json
import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft import PeftModel as _PeftModel

from sft_data_v2 import (
    NegotiationV2Dataset,
    build_splits,
    compute_class_weights,
    TARGET_SR,
)

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)
logging.getLogger("root").setLevel(logging.ERROR)

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_ID          = "Qwen/Qwen2.5-Omni-7B"
OUTPUT_DIR        = Path(__file__).parent / "sft_output_v2"
# Auto-detected from the script's location: always v2/tts_outputs/
DEFAULT_AUDIO_DIR = Path(__file__).parent / "tts_outputs"

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
_AUDIO_MODULE_KEYS = (
    "audio_tower", "audio_encoder", "whisper",
    "conv1", "conv2", "encoder.layers", "embed_positions",
)

# ── Torch Dataset wrapper (adds len + getitem) ────────────────────────────────

class _TorchDataset(Dataset):
    def __init__(self, inner: NegotiationV2Dataset):
        self._inner = inner
    def __len__(self):
        return len(self._inner)
    def __getitem__(self, idx):
        return self._inner[idx]


# ── Collator ──────────────────────────────────────────────────────────────────

@dataclass
class MultimodalSFTCollator:
    processor:  Qwen2_5OmniProcessor
    max_length: int = 4096

    def __post_init__(self):
        asst_header = "<|im_start|>assistant\n"
        self._asst_header_ids = torch.tensor(
            self.processor.tokenizer(asst_header, add_special_tokens=False).input_ids,
            dtype=torch.long,
        )

    def _build_labels(self, input_ids: torch.Tensor) -> torch.Tensor:
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
            warnings.warn(
                "_build_labels: assistant header not found — zero loss for this sample"
            )
        return labels

    def __call__(self, batch: list[dict]) -> dict:
        all_input_ids, all_attn_mask, all_labels = [], [], []
        all_audio_feats, all_feat_masks = [], []
        max_seq = 0

        processed = []
        for sample in batch:
            messages     = sample["messages"]
            audio_arrays = sample["audio_arrays"]

            def _wrap(msg):
                c = msg["content"]
                if isinstance(c, str):
                    c = [{"type": "text", "text": c}]
                return {"role": msg["role"], "content": c}

            messages_norm = [_wrap(m) for m in messages]
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


# ── Weighted loss trainer (class-balance weights, not CL weights) ──────────────

class WeightedLossTrainer(Trainer):
    def __init__(self, *args, class_weights: dict = None, **kwargs):
        super().__init__(*args, **kwargs)
        # class_weights: {"LEVERAGE": float, "MITIGATE": float}
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
            vloss = logs.get("eval_loss", "")
            pct   = 100 * step / total if total else 0
            msg   = f"Step {step}/{total} ({pct:.1f}%)  loss={loss}"
            if vloss:
                msg += f"  eval_loss={vloss}"
            log.info(msg)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            log.info(
                f"Eval @ step {state.global_step} — "
                + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, float))
            )

    def on_save(self, args, state, control, **kwargs):
        log.info(f"Checkpoint saved at step {state.global_step} → {args.output_dir}/checkpoint-{state.global_step}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir",   default=str(DEFAULT_AUDIO_DIR),
                        help=(
                            "Path to the tts_outputs/ directory containing dialogue_XXXX/ subdirs. "
                            f"Defaults to v2/tts_outputs/ (auto-detected as {DEFAULT_AUDIO_DIR}). "
                            "Override only if your TTS outputs are in a non-standard location."
                        ))
    parser.add_argument("--model_id",    default=MODEL_ID)
    parser.add_argument("--output_dir",  default=str(OUTPUT_DIR))
    parser.add_argument("--epochs",      type=int,   default=1)
    parser.add_argument("--batch_size",  type=int,   default=1)
    parser.add_argument("--grad_accum",  type=int,   default=16)
    parser.add_argument("--lr",          type=float, default=5e-5)
    parser.add_argument("--weight_decay",type=float, default=0.01)
    parser.add_argument("--max_length",  type=int,   default=4096)
    parser.add_argument("--lora_r",      type=int,   default=8)
    parser.add_argument("--lora_alpha",  type=int,   default=16)
    parser.add_argument("--lora_dropout",type=float, default=0.1)
    parser.add_argument("--save_steps",  type=int,   default=50)
    parser.add_argument("--eval_steps",  type=int,   default=50)
    parser.add_argument("--warmup_ratio",type=float, default=0.1)
    parser.add_argument("--no_qlora",    action="store_true",
                        help="Load in bf16 (DDP-compatible). Recommended on A100 80GB.")
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--init_adapter", default=None,
                        help="Load existing LoRA adapter as weight init (for future CL phases).")
    args = parser.parse_args()

    # ── Per-run directory — every file for this run lives here ───────────────
    import datetime as _dt
    _run_tag  = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / _run_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    _log_file = output_dir / "sft.log"
    _fh = logging.FileHandler(_log_file)
    _fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(_fh)
    log.info(f"Run tag    : {_run_tag}")
    log.info(f"Output dir : {output_dir}")
    log.info(f"Run log    : {_log_file}")

    audio_dir = Path(args.audio_dir)
    if not audio_dir.exists():
        raise SystemExit(
            f"\nERROR: TTS audio directory not found: {audio_dir}\n"
            f"Generate the speech dialogues first:\n"
            f"  CUDA_VISIBLE_DEVICES=X python3 v2/generate_tts_v2.py --all\n"
            f"Then re-run SFT.\n"
        )
    print(f"  Audio directory : {audio_dir}")

    mode = "bf16 LoRA (DDP-compatible)" if args.no_qlora else f"QLoRA r={args.lora_r}"
    print(f"\n{'='*65}")
    print(f"  SFT v2 — Qwen2.5-Omni-7B  |  {mode}")
    print(f"  Audio+Text → <reasoning><decision><seller_vaii><response>")
    print(f"{'='*65}\n")

    # ── Splits + class weights ─────────────────────────────────────────────────
    splits     = build_splits()
    class_weights = compute_class_weights(splits)
    print(f"  Class weights: LEVERAGE={class_weights['LEVERAGE']:.4f}  "
          f"MITIGATE={class_weights['MITIGATE']:.4f}")

    # ── Datasets ───────────────────────────────────────────────────────────────
    print("  Building datasets (loading intermediates + checking audio)…")
    train_inner = NegotiationV2Dataset(splits["train"], audio_dir, "train")
    val_inner   = NegotiationV2Dataset(splits["val"],   audio_dir, "val")
    test_inner  = NegotiationV2Dataset(splits["test"],  audio_dir, "test")

    if len(train_inner) == 0:
        raise SystemExit("ERROR: 0 training examples built. Check audio_dir and intermediate files.")

    # Only count audio-skipped examples in the completeness check.
    # Reasoning-skipped examples (Phase 2 incomplete) are expected and acceptable.
    expected_with_reasoning = (len(splits["train"]) * 6) - train_inner.skipped_no_reasoning
    if expected_with_reasoning > 0:
        missing_frac = train_inner.skipped_no_audio / expected_with_reasoning
    else:
        missing_frac = 0.0
    if missing_frac > 0.10:
        raise SystemExit(
            f"ERROR: {missing_frac:.1%} of audio-expected training examples have missing WAVs. "
            f"({train_inner.skipped_no_audio} dialogues affected — check sft_missing_audio.log). "
            f"Run generate_tts_v2.py to completion before SFT."
        )

    train_ds = _TorchDataset(train_inner)
    val_ds   = _TorchDataset(val_inner)

    print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test (held-out): {len(test_inner):,}")

    # ── Processor ─────────────────────────────────────────────────────────────
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    print(f"  Loading processor: {args.model_id}")
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_id, token=hf_token)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # ── Model loading ──────────────────────────────────────────────────────────
    if args.no_qlora:
        print(f"  Loading Qwen2.5-Omni in bf16 (enable_audio_output=False)…")
        _full = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            enable_audio_output=False,   # saves ~6GB — Talker not loaded
            token=hf_token,
        )
        model = _full.thinker
        model.config.use_cache = False
        del _full; gc.collect()
        print("  Thinker extracted (Talker/Token2Wav not loaded).")
    else:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        n_gpus     = torch.cuda.device_count()
        total_mem  = torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)
        per_gpu    = f"{max(total_mem - 10, 10)}GiB"
        max_memory = {i: per_gpu for i in range(n_gpus)}
        max_memory["cpu"] = "32GiB"
        print(f"  Loading Qwen2.5-Omni in 4-bit NF4 (QLoRA) | GPUs:{n_gpus} budget:{per_gpu}")
        _full = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            args.model_id,
            quantization_config=bnb,
            device_map="auto",
            max_memory=max_memory,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            enable_audio_output=False,
            token=hf_token,
        )
        model = _full.thinker
        model.config.use_cache = False
        del _full; gc.collect()
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # ── LoRA ───────────────────────────────────────────────────────────────────
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
            task_type=None,   # generic PeftModel — avoids forward() conflicts with Omni
        )
        model = get_peft_model(model, lora_cfg)

    # ── Fix 3: freeze audio encoder (LoRA attaches to audio layers too) ────────
    frozen = 0
    for name, param in model.named_parameters():
        if any(k in name.lower() for k in _AUDIO_MODULE_KEYS):
            param.requires_grad = False
            frozen += param.numel()
    if frozen:
        print(f"  Audio encoder params re-frozen: {frozen:,}")

    model.print_trainable_parameters()

    audio_trainable = sum(
        p.numel() for n, p in model.named_parameters()
        if any(k in n.lower() for k in _AUDIO_MODULE_KEYS) and p.requires_grad
    )
    print(f"  Trainable audio encoder params: {audio_trainable} "
          f"({'FROZEN ✓' if audio_trainable == 0 else 'WARNING: not fully frozen'})")

    # ── Collator + Training args ───────────────────────────────────────────────
    collator = MultimodalSFTCollator(processor=processor, max_length=args.max_length)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        seed=42,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=25,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        report_to="none",
        optim="adamw_torch" if args.no_qlora else "paged_adamw_8bit",
    )

    # ── Trainer ────────────────────────────────────────────────────────────────
    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        class_weights=class_weights,
        callbacks=[ProgressCallback()],
    )

    # ── Train ──────────────────────────────────────────────────────────────────
    print(f"\n  Starting training…")
    print(f"  Effective batch : {args.batch_size * args.grad_accum}")
    print(f"  Max seq length  : {args.max_length}")
    print(f"  Epochs          : {args.epochs}")
    print(f"  LR              : {args.lr}\n")

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # ── Save adapter ───────────────────────────────────────────────────────────
    final = output_dir / "final_adapter"
    model.save_pretrained(final)
    processor.save_pretrained(final)

    # ── Save test split ids ────────────────────────────────────────────────────
    (output_dir / "test_ids.json").write_text(
        json.dumps(splits["test"], indent=2)
    )

    # ── Save training log (loss curve per step) ────────────────────────────────
    log_history = trainer.state.log_history
    (output_dir / "training_log.json").write_text(json.dumps(log_history, indent=2))
    print(f"  Training log    → {output_dir / 'training_log.json'}")

    # ── Evaluate on held-out test set ─────────────────────────────────────────
    print("\n  Evaluating on test set…")
    test_ds = _TorchDataset(test_inner)
    test_results = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")
    (output_dir / "test_results.json").write_text(json.dumps(test_results, indent=2))
    print(f"  Test results    → {output_dir / 'test_results.json'}")
    print(f"  Test loss       : {test_results.get('test_loss', 'n/a'):.4f}")

    # ── Save run summary ───────────────────────────────────────────────────────
    import datetime
    train_steps  = [e for e in log_history if "loss" in e and "eval_loss" not in e]
    eval_steps   = [e for e in log_history if "eval_loss" in e]
    best_eval    = min((e["eval_loss"] for e in eval_steps), default=None)

    summary = {
        "completed_at":       datetime.datetime.now().isoformat(),
        "model_id":           args.model_id,
        "lora_r":             args.lora_r,
        "lora_alpha":         args.lora_alpha,
        "epochs":             args.epochs,
        "lr":                 args.lr,
        "batch_size":         args.batch_size,
        "grad_accum":         args.grad_accum,
        "effective_batch":    args.batch_size * args.grad_accum,
        "train_examples":     len(train_ds),
        "val_examples":       len(val_ds),
        "test_examples":      len(test_inner),
        "skipped_no_reasoning_train": train_inner.skipped_no_reasoning,
        "skipped_no_audio_train":     train_inner.skipped_no_audio,
        "class_weights":      class_weights,
        "final_train_loss":   train_steps[-1]["loss"] if train_steps else None,
        "best_eval_loss":     best_eval,
        "test_loss":          test_results.get("test_loss"),
        "total_steps":        trainer.state.global_step,
        "adapter_path":       str(final),
        "run_tag":            _run_tag,
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  Run summary     → {output_dir / 'run_summary.json'}")
    print(f"\n{'='*65}\n  DONE\n{'='*65}\n")


if __name__ == "__main__":
    main()
