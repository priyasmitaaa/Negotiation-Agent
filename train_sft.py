#!/usr/bin/env python3
"""
train_sft.py
============
LoRA fine-tuning of Qwen2.5-Omni-7B on the negotiation bias detection
dataset (MANDATORY_MITIGATION vs ETHICAL_LEVERAGE).

Both modalities are fed to the model per training step:
  SPEECH — actual .wav files loaded from paths in train_sft.jsonl, resampled
            to 16 kHz and fed to Qwen2.5-Omni's Whisper audio encoder
  TEXT   — turn transcripts + vulnerability labels + VAII summary

Two modes (controlled by --no_qlora flag):

  QLoRA (default, --no_qlora NOT set):
    Base model in 4-bit NF4 — frozen, ~4.5 GB
    LoRA adapters on LLM backbone only — trainable
    Audio encoder (Whisper-based) — frozen
    Best for: single GPU or model-parallel across 2 GPUs

  Full bf16 LoRA (--no_qlora, recommended for A100 80 GB):
    Base model in bf16 — ~14 GB per GPU
    LoRA adapters on LLM backbone only — trainable
    Compatible with DDP (accelerate launch --num_processes 2)
    Best for: 2× A100 80 GB via DDP

Class imbalance (MM:EL = 2:1) handled by weighted cross-entropy loss:
  weights loaded from dataset/class_weights.json

Usage:
  # Single GPU, QLoRA (any GPU)
  CUDA_VISIBLE_DEVICES=1 python train_sft.py

  # Two GPUs, model-parallel QLoRA (single process)
  CUDA_VISIBLE_DEVICES=0,1 python train_sft.py

  # Two GPUs, full bf16 LoRA via DDP (recommended on A100 80 GB)
  CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file accelerate_2gpu.yaml \\
      train_sft.py --no_qlora

  # Resume from checkpoint
  CUDA_VISIBLE_DEVICES=1 python train_sft.py --resume_from_checkpoint sft_output/checkpoint-500
"""

import argparse
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import gc
import numpy as np
import soundfile as sf
import librosa
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
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

warnings.filterwarnings("ignore")
# Silence Qwen2.5-Omni's "System prompt modified" warning (fires per audio turn)
import logging as _logging
_logging.getLogger("root").setLevel(_logging.ERROR)

# ── Defaults ──────────────────────────────────────────────────────────────────
MODEL_ID      = "Qwen/Qwen2.5-Omni-7B"
DATASET_DIR   = Path("dataset")
OUTPUT_DIR    = Path("sft_output")
TARGET_SR     = 16_000   # Qwen2.5-Omni audio encoder expects 16 kHz

# LoRA target modules — LLM backbone only, audio encoder stays frozen
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

LABEL_MAP = {"ETHICAL_LEVERAGE": 0, "MANDATORY_MITIGATION": 1}


# ── Dataset ───────────────────────────────────────────────────────────────────

class NegotiationSFTDataset(Dataset):
    """
    Loads train_sft.jsonl / val_sft.jsonl produced by build_dataset.py.

    Each record has:
      messages[0] — system prompt (text)
      messages[1] — user content: interleaved {"type":"audio","audio":path}
                    and {"type":"text","text":"..."} items
      messages[2] — assistant response: "MANDATORY_MITIGATION" | "ETHICAL_LEVERAGE"

    __getitem__ returns a dict ready for the custom collator:
      text_with_response  str   full chat-formatted string incl. assistant turn
      audio_arrays        list  [np.float32 arrays at TARGET_SR] one per turn
      label               int   0 or 1
    """

    def __init__(self, jsonl_path: str, max_audio_turns: int = 12):
        self.records = []
        self.max_audio_turns = max_audio_turns
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

    def __len__(self):
        return len(self.records)

    def _load_wav(self, path: str) -> Optional[np.ndarray]:
        """Load and resample a wav file to TARGET_SR. Returns None on failure."""
        try:
            audio, sr = sf.read(path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)       # stereo → mono
            if sr != TARGET_SR:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
            return audio.astype(np.float32)
        except Exception:
            return None

    def __getitem__(self, idx: int) -> dict:
        rec      = self.records[idx]
        messages = rec["messages"]
        label    = rec.get("label", LABEL_MAP.get(rec.get("label_str", ""), 0))

        # ── Collect audio arrays in turn order ────────────────────────────────
        audio_arrays = []
        for item in messages[1]["content"]:   # user message
            if item["type"] == "audio":
                arr = self._load_wav(item["audio"])
                if arr is not None:
                    audio_arrays.append(arr)
                # If load fails, we simply don't add — text still present

        return {
            "messages":     messages,
            "audio_arrays": audio_arrays,
            "label":        label,
        }


# ── Collator ──────────────────────────────────────────────────────────────────

@dataclass
class MultimodalSFTCollator:
    """
    Converts a batch of dataset items into model inputs.

    For each sample:
      1. Apply processor.apply_chat_template → formatted text string
         (includes both the user prompt AND the assistant response so the
         model can compute loss on the response tokens)
      2. Pass text + audio arrays to processor → input_ids, attention_mask,
         input_features (audio mel-spectrograms)
      3. Build labels: mask everything except assistant response tokens with -100
    """

    processor: Qwen2_5OmniProcessor
    max_length: int = 4096

    def __post_init__(self):
        # Pre-compute assistant turn header token ids once at construction time.
        # Qwen2.5 chat template always emits "<|im_start|>assistant\n" before the
        # assistant response. Searching for this header is robust: it is never
        # affected by BPE context merges between the response text and surrounding
        # tokens, and it correctly handles audio token injection (which shifts
        # absolute positions but does not move or alter the assistant header).
        asst_header = "<|im_start|>assistant\n"
        self._asst_header_ids = torch.tensor(
            self.processor.tokenizer(asst_header, add_special_tokens=False).input_ids,
            dtype=torch.long,
        )

    def _build_labels(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Return label tensor where prompt tokens → -100 (ignored),
        assistant response tokens (everything after <|im_start|>assistant\\n)
        → actual token ids so cross-entropy loss is computed on them.

        Searches right-to-left for the last occurrence of the assistant header
        so it works correctly even when the header appears in few-shot examples.
        """
        labels   = torch.full_like(input_ids, -100)
        seq_len  = input_ids.size(0)
        hdr      = self._asst_header_ids.to(input_ids.device)
        hdr_len  = hdr.size(0)

        for start in range(seq_len - hdr_len, -1, -1):
            if torch.equal(input_ids[start: start + hdr_len], hdr):
                label_start = start + hdr_len
                if label_start < seq_len:
                    labels[label_start:] = input_ids[label_start:]
                break
        else:
            import warnings as _w
            _w.warn(
                "_build_labels: assistant header not found in input_ids — "
                "all labels are -100, this sample contributes zero loss. "
                "Check that apply_chat_template is producing the expected format.",
                stacklevel=2,
            )

        return labels

    def __call__(self, batch: list[dict]) -> dict:
        all_input_ids      = []
        all_attention_mask = []
        all_labels         = []
        all_audio_features = []   # [1, n_mels, T_i] per sample, or empty
        all_feat_attn_masks = []  # [1, T_i] per sample, parallel to above
        max_seq = 0

        processed = []
        for sample in batch:
            messages     = sample["messages"]
            audio_arrays = sample["audio_arrays"]

            # ── Normalise message content for Qwen2.5-Omni processor ──────────
            # The processor's apply_chat_template expects content to always be a
            # list of typed dicts (e.g. [{"type":"text","text":"..."}]).
            # System and assistant messages in the dataset store plain strings.
            def _wrap(msg):
                c = msg["content"]
                if isinstance(c, str):
                    c = [{"type": "text", "text": c}]
                return {"role": msg["role"], "content": c}
            messages_norm = [_wrap(m) for m in messages]

            # ── Chat-formatted text (with assistant response) ─────────────────
            text = self.processor.apply_chat_template(
                messages_norm,
                tokenize=False,
                add_generation_prompt=False,
            )

            # ── Processor call ─────────────────────────────────────────────────
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
                # Fallback: text only (if audio processing fails for this sample)
                inputs = self.processor(
                    text=text,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=self.max_length,
                )

            input_ids = inputs["input_ids"].squeeze(0)
            attn_mask = inputs["attention_mask"].squeeze(0)
            # Label masking: find assistant header in input_ids, score everything
            # after it. This is exact and immune to BPE context-merge issues.
            labels    = self._build_labels(input_ids)

            processed.append({
                "input_ids":      input_ids,
                "attention_mask": attn_mask,
                "labels":         labels,
                "audio_features": inputs.get("input_features"),       # [1, n_mels, T] or None
                "feat_attn_mask": inputs.get("feature_attention_mask"), # [1, T] or None
            })
            max_seq = max(max_seq, input_ids.size(0))

        # ── Pad batch to same length ──────────────────────────────────────────
        pad_id = self.processor.tokenizer.pad_token_id or 0
        for p in processed:
            seq_len = p["input_ids"].size(0)
            pad_len = max_seq - seq_len
            if pad_len > 0:
                p["input_ids"]      = F.pad(p["input_ids"],      (0, pad_len), value=pad_id)
                p["attention_mask"] = F.pad(p["attention_mask"], (0, pad_len), value=0)
                p["labels"]         = F.pad(p["labels"],         (0, pad_len), value=-100)

            all_input_ids.append(p["input_ids"])
            all_attention_mask.append(p["attention_mask"])
            all_labels.append(p["labels"])
            if p["audio_features"] is not None:
                all_audio_features.append(p["audio_features"])
                all_feat_attn_masks.append(p["feat_attn_mask"])

        batch_out = {
            "input_ids":      torch.stack(all_input_ids),
            "attention_mask": torch.stack(all_attention_mask),
            "labels":         torch.stack(all_labels),
            "sample_labels":  torch.tensor(
                [s["label"] for s in batch], dtype=torch.long
            ),
        }

        # Stack audio features and masks; pad to same time dimension across batch.
        # feature_attention_mask [B, T] is required by Thinker's audio encoder.
        if all_audio_features:
            max_T = max(f.shape[-1] for f in all_audio_features)
            padded_feats, padded_masks = [], []
            for feat, mask in zip(all_audio_features, all_feat_attn_masks):
                pad_T = max_T - feat.shape[-1]
                if pad_T > 0:
                    feat = F.pad(feat, (0, pad_T), value=0.0)
                    if mask is not None:
                        mask = F.pad(mask, (0, pad_T), value=0)
                padded_feats.append(feat)
                padded_masks.append(mask)
            batch_out["input_features"] = torch.cat(padded_feats, dim=0)           # [B, n_mels, T]
            if any(m is not None for m in padded_masks):
                batch_out["feature_attention_mask"] = torch.cat(
                    [m for m in padded_masks if m is not None], dim=0
                )                                                                    # [B, T]

        return batch_out


# ── Weighted loss trainer ──────────────────────────────────────────────────────

class WeightedLossTrainer(Trainer):
    """
    Overrides compute_loss to apply class weights (EL upweighted, MM downweighted)
    counteracting the 2:1 MM:EL imbalance.

    Weight is applied token-wise on the response tokens (labels != -100).
    We derive the sample label from which response string appears in labels.
    """

    def __init__(self, *args, class_weights: dict = None, **kwargs):
        super().__init__(*args, **kwargs)
        # class_weights: {0: w_EL, 1: w_MM}
        self.cw = class_weights or {0: 1.0, 1: 1.0}

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels      = inputs.get("labels")
        outputs     = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits      = outputs.logits

        # Shift for causal LM: predict next token
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        vocab_size = shift_logits.size(-1)
        loss_fct   = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        token_loss = loss_fct(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
        )  # shape: [B * (T-1)]

        # ── Per-sample weighting ──────────────────────────────────────────────
        B, T = shift_labels.shape
        token_loss = token_loss.view(B, T)

        # Determine sample weight from which label string is in this sample.
        # We use the dataset label stored in inputs if available, else default 1.0
        sample_weights = torch.ones(B, device=token_loss.device)
        if "sample_labels" in inputs:
            for i, lbl in enumerate(inputs["sample_labels"]):
                sample_weights[i] = self.cw.get(int(lbl), 1.0)

        # Apply weight only to non-ignored (response) tokens
        mask = (shift_labels != -100).float()
        weighted_loss = (token_loss * mask * sample_weights.unsqueeze(1)).sum()
        denom = mask.sum().clamp(min=1)
        loss  = weighted_loss / denom

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
            msg   = f"  Step {step}/{total} ({pct:.1f}%)  loss={loss}"
            if vloss:
                msg += f"  eval_loss={vloss}"
            print(msg, flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id",       default=MODEL_ID)
    parser.add_argument("--dataset_dir",    default=str(DATASET_DIR))
    parser.add_argument("--output_dir",     default=str(OUTPUT_DIR))
    parser.add_argument("--epochs",         type=int,   default=1,
                        help="Number of training epochs. 1 is recommended to avoid "
                             "memorisation on the 2025-sample dataset.")
    parser.add_argument("--batch_size",     type=int,   default=1,
                        help="Per-device batch size. Keep at 1 for audio+text on 11GB GPU")
    parser.add_argument("--grad_accum",     type=int,   default=16,
                        help="Accumulate gradients over N steps (effective batch = 16)")
    parser.add_argument("--lr",             type=float, default=5e-5,
                        help="Peak learning rate. 5e-5 avoids the memorisation seen "
                             "at 2e-4 (loss→0 after epoch 1.5).")
    parser.add_argument("--weight_decay",   type=float, default=0.01)
    parser.add_argument("--max_length",     type=int,   default=3072)
    parser.add_argument("--lora_r",         type=int,   default=8,
                        help="LoRA rank. 8 halves memorisation capacity vs 16 "
                             "while preserving expressiveness.")
    parser.add_argument("--lora_alpha",     type=int,   default=16,
                        help="LoRA alpha. Keep ratio alpha/r = 2.")
    parser.add_argument("--lora_dropout",   type=float, default=0.1)
    parser.add_argument("--save_steps",     type=int,   default=50)
    parser.add_argument("--eval_steps",     type=int,   default=50,
                        help="Evaluate every N steps. 50 catches loss collapse early.")
    parser.add_argument("--warmup_ratio",   type=float, default=0.1)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--no_qlora", action="store_true",
                        help="Load model in bf16 without 4-bit quantization. "
                             "Required for DDP (accelerate launch --num_processes 2). "
                             "Recommended on A100 80 GB.")
    parser.add_argument("--train_jsonl", default=None,
                        help="Override the default dataset_dir/train_sft.jsonl. "
                             "Pass a specific JSONL file for curriculum learning phases "
                             "(e.g. dataset/tier_easy.jsonl).")
    parser.add_argument("--init_adapter", default=None,
                        help="Load an existing LoRA adapter as weight initialization "
                             "instead of random init. Used for CL-SFT phases: each phase "
                             "starts from the previous phase's adapter. "
                             "Example: sft_output/final_adapter")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mode = "bf16 LoRA (DDP-compatible)" if args.no_qlora else f"QLoRA r={args.lora_r}"
    print(f"\n{'='*65}")
    print(f"  SFT — Qwen2.5-Omni-7B  |  {mode}")
    print(f"  Speech (.wav) + Text → MANDATORY_MITIGATION / ETHICAL_LEVERAGE")
    print(f"{'='*65}\n")

    # ── Class weights ─────────────────────────────────────────────────────────
    cw_path = dataset_dir / "class_weights.json"
    with open(cw_path) as f:
        cw_raw = json.load(f)
    class_weights = {
        0: cw_raw["0_ETHICAL_LEVERAGE"],
        1: cw_raw["1_MANDATORY_MITIGATION"],
    }
    print(f"  Class weights: EL={class_weights[0]:.4f}  MM={class_weights[1]:.4f}")

    # ── Load processor ────────────────────────────────────────────────────────
    print(f"  Loading processor: {args.model_id}")
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_id, token=hf_token)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    if args.no_qlora:
        # ── Full bf16 LoRA — DDP-compatible, recommended on A100 80 GB ────────
        # Qwen2_5OmniForConditionalGeneration has no forward() method; it is
        # a pipeline model (Thinker + Talker + Token2Wav) designed only for
        # generate(). For training we load the full model and extract the
        # Thinker sub-model (Qwen2_5OmniThinkerForConditionalGeneration) which
        # has a proper forward(input_ids, input_features, labels, ...) and is
        # the component responsible for audio+text understanding → text output.
        print(f"  Loading full Qwen2.5-Omni in bf16 (to extract Thinker)...")
        _full = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            token=hf_token,
        )
        model = _full.thinker   # Qwen2_5OmniThinkerForConditionalGeneration
        model.config.use_cache = False
        # Free Talker & Token2Wav from CPU RAM (not needed for SFT)
        if hasattr(_full, "talker"):    del _full.talker
        if hasattr(_full, "token2wav"): del _full.token2wav
        del _full
        gc.collect()
        print(f"  Thinker extracted. Talker/Token2Wav freed from memory.")
        # Do NOT call gradient_checkpointing_enable() here — the Thinker's
        # get_input_embeddings() may not be registered at this point.
        # The Trainer handles GC via gradient_checkpointing_kwargs below.
    else:
        # ── QLoRA — 4-bit NF4, model-parallel, works on smaller GPUs ─────────
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        print(f"  Loading model in 4-bit NF4 (QLoRA)...")
        # device_map="auto" splits model across all visible GPUs (model-parallel).
        # Use ~10 GB headroom below total GPU memory.
        n_gpus = torch.cuda.device_count()
        total_mem = torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)
        per_gpu_mem = f"{max(total_mem - 10, 10)}GiB"
        max_memory = {i: per_gpu_mem for i in range(n_gpus)}
        max_memory["cpu"] = "32GiB"
        print(f"  GPUs visible: {n_gpus}  |  per-GPU budget: {per_gpu_mem}")

        _full = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            args.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            max_memory=max_memory,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            token=hf_token,
        )
        model = _full.thinker   # extract trainable Thinker submodel
        model.config.use_cache = False
        if hasattr(_full, "talker"):    del _full.talker
        if hasattr(_full, "token2wav"): del _full.token2wav
        del _full
        gc.collect()
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )

    # ── LoRA — LLM backbone only, audio encoder stays frozen ─────────────────
    if args.init_adapter:
        # CL-SFT: load previous phase's adapter as weight initialization.
        # PeftModel.from_pretrained with is_trainable=True keeps LoRA params
        # trainable so we can continue fine-tuning from this checkpoint.
        from peft import PeftModel as _PeftModel
        print(f"  Loading LoRA init from {args.init_adapter} (is_trainable=True)…")
        model = _PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
        print(f"  LoRA weights initialized from adapter. Training will update them.")
    else:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=args.lora_dropout,
            bias="none",
            # task_type=None → generic PeftModel (not PeftModelForCausalLM).
            # Qwen2.5-Omni is multimodal; PeftModelForCausalLM's forward drops
            # audio kwargs (input_features) and conflicts with the model's interface.
            task_type=None,
            modules_to_save=None,
        )
        model = get_peft_model(model, lora_config)

    # ── Explicitly freeze audio encoder (Fix 3) ───────────────────────────────
    # LoRA target module names (q_proj, k_proj, …) also match attention layers
    # inside the Whisper-based audio tower, so PEFT attached adapters there too.
    # We want ONLY the LLM backbone to be trained. Freeze every parameter whose
    # fully-qualified name contains an audio-encoder module path segment.
    _AUDIO_MODULE_KEYS = ("audio_tower", "audio_encoder", "whisper",
                          "conv1", "conv2", "encoder.layers", "embed_positions")
    frozen_audio = 0
    for name, param in model.named_parameters():
        name_lower = name.lower()
        if any(k in name_lower for k in _AUDIO_MODULE_KEYS):
            param.requires_grad = False
            frozen_audio += param.numel()
    if frozen_audio:
        print(f"  Audio encoder params re-frozen: {frozen_audio:,}")

    model.print_trainable_parameters()

    # Confirm audio encoder is fully frozen after the explicit freeze
    audio_trainable = sum(
        p.numel() for n, p in model.named_parameters()
        if any(k in n.lower() for k in _AUDIO_MODULE_KEYS) and p.requires_grad
    )
    print(f"  Trainable audio encoder params: {audio_trainable} "
          f"({'FROZEN ✓' if audio_trainable == 0 else 'WARNING: still not frozen — check module names'})")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_jsonl = Path(args.train_jsonl) if args.train_jsonl else dataset_dir / "train_sft.jsonl"
    print(f"  Loading datasets: train={train_jsonl}  val={dataset_dir}/val_sft.jsonl")
    train_dataset = NegotiationSFTDataset(train_jsonl, max_audio_turns=12)
    val_dataset   = NegotiationSFTDataset(
        dataset_dir / "val_sft.jsonl",   max_audio_turns=12)
    print(f"  Train: {len(train_dataset):,}  |  Val: {len(val_dataset):,}")

    collator = MultimodalSFTCollator(processor=processor, max_length=args.max_length)

    # ── Training arguments ────────────────────────────────────────────────────
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
        # use_reentrant=False avoids the get_input_embeddings() hook that
        # Qwen2.5-Omni doesn't implement; works with both LoRA and QLoRA.
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
        remove_unused_columns=False,   # critical — we have custom fields
        report_to="none",
        # paged_adamw_8bit requires bitsandbytes; use adamw_torch in DDP/bf16 mode
        optim="adamw_torch" if args.no_qlora else "paged_adamw_8bit",
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        class_weights=class_weights,
        callbacks=[ProgressCallback()],
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\n  Starting training...")
    print(f"  Effective batch size : {args.batch_size * args.grad_accum}")
    print(f"  Max sequence length  : {args.max_length}")
    print(f"  Epochs               : {args.epochs}")
    print(f"  LR                   : {args.lr}\n")

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # ── Save final adapter ────────────────────────────────────────────────────
    final_path = output_dir / "final_adapter"
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    print(f"\n  LoRA adapter saved → {final_path}")
    print(f"  Load for inference:")
    print(f"    from peft import PeftModel")
    print(f"    model = Qwen2_5OmniForConditionalGeneration.from_pretrained('{args.model_id}', ...)")
    print(f"    model = PeftModel.from_pretrained(model, '{final_path}')")
    print(f"\n{'='*65}\n  DONE\n{'='*65}\n")


if __name__ == "__main__":
    main()
