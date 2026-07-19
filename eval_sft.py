#!/usr/bin/env python3
"""
eval_sft.py — Test-set evaluation of the fine-tuned LoRA adapter
=================================================================
Loads the Thinker + LoRA adapter, runs logit-based inference on every
sample in a JSONL split, and prints accuracy / F1 / confusion matrix.

Inference strategy — logit comparison (fast, ~15× vs autoregressive):
  1. Format the prompt (system + user, NO assistant turn).
  2. One forward pass through the Thinker → logits at the last position.
  3. Compare logit[24958] ("MAND…") vs logit[7625] ("ETH…") — the
     distinct first tokens of MANDATORY_MITIGATION and ETHICAL_LEVERAGE.
  4. Argmax is the prediction.

Usage:
  # Val set (default — used only for sanity; not seen during training? No — val
  # WAS used for eval_loss during training, so use test.parquet for clean eval)
  CUDA_VISIBLE_DEVICES=1 python3 eval_sft.py

  # Explicit JSONL path
  CUDA_VISIBLE_DEVICES=1 python3 eval_sft.py --jsonl dataset/val_sft.jsonl

  # Different adapter
  CUDA_VISIBLE_DEVICES=1 python3 eval_sft.py --adapter sft_output/checkpoint-50

  # Quick sanity with 20 samples
  CUDA_VISIBLE_DEVICES=1 python3 eval_sft.py --max_samples 20
"""

import argparse
import gc
import json
import logging as _logging
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
)
from peft import PeftModel

warnings.filterwarnings("ignore")
_logging.getLogger("root").setLevel(_logging.ERROR)

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_ID  = "Qwen/Qwen2.5-Omni-7B"
TARGET_SR = 16_000
LABELS    = ["ETHICAL_LEVERAGE", "MANDATORY_MITIGATION"]   # index 0, 1
LABEL2ID  = {l: i for i, l in enumerate(LABELS)}


# ── Bootstrap CI ───────────────────────────────────────────────────────────────

def bootstrap_ci(
    y_true: list[int],
    y_pred: list[int],
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, tuple[float, float]]:
    """
    Compute bootstrap 95% CIs for accuracy and macro-F1.
    Returns {metric: (lo, hi)} at the requested confidence level.
    Uses numpy vectorised resampling (no sklearn dependency beyond f1_score).
    """
    rng = np.random.default_rng(seed)
    yt  = np.array(y_true)
    yp  = np.array(y_pred)
    n   = len(yt)
    accs, f1s = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        accs.append(float((yt[idx] == yp[idx]).mean()))
        f1s.append(float(f1_score(yt[idx], yp[idx], average="macro", zero_division=0)))
    alpha = (1.0 - ci) / 2.0
    return {
        "accuracy": (float(np.quantile(accs, alpha)),
                     float(np.quantile(accs, 1.0 - alpha))),
        "f1_macro": (float(np.quantile(f1s,  alpha)),
                     float(np.quantile(f1s,  1.0 - alpha))),
    }


# ── Audio loader ───────────────────────────────────────────────────────────────

def _load_wav(path: str) -> np.ndarray | None:
    try:
        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != TARGET_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        return audio.astype(np.float32)
    except Exception:
        return None


# ── Model loader ───────────────────────────────────────────────────────────────

def load_thinker(model_id: str, adapter_path: str, device: str):
    """
    Load Qwen2.5-Omni Thinker sub-model with LoRA adapter merged in.
    Talker + Token2Wav are discarded — not needed for text classification.
    merge_and_unload() bakes the LoRA deltas into the base weights so
    inference has zero adapter overhead.
    """
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    print("  Loading base model (bf16)…")
    _full = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        token=hf_token,
    )
    model = _full.thinker
    if hasattr(_full, "talker"):    del _full.talker
    if hasattr(_full, "token2wav"): del _full.token2wav
    del _full
    gc.collect()

    print(f"  Merging LoRA from {adapter_path} …")
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()  # bake weights, drop adapter overhead

    model = model.to(device)
    model.eval()
    print(f"  Ready on {device}\n")
    return model


# ── Single-sample inference ────────────────────────────────────────────────────

def predict_one(
    model,
    processor,
    record: dict,
    label_first_ids: tuple[int, int],
    device: str,
    max_audio_turns: int = 12,
) -> tuple[str, float, float]:
    """
    Predict the label for one record.

    Returns
    -------
    pred       : "MANDATORY_MITIGATION" | "ETHICAL_LEVERAGE"
    score_mm   : raw logit for MANDATORY_MITIGATION first token
    score_el   : raw logit for ETHICAL_LEVERAGE first token

    Strategy: one forward pass, compare logits at the last prompt position.
    MAND… (id=24958) and ETH… (id=7625) are the distinct first tokens of the
    two labels — confirmed by tokenizer inspection before the eval loop.
    """
    messages = record["messages"]

    def _wrap(msg):
        c = msg["content"]
        if isinstance(c, str):
            c = [{"type": "text", "text": c}]
        return {"role": msg["role"], "content": c}

    # Prompt = system + user only; add_generation_prompt appends <|im_start|>assistant\n
    prompt_msgs = [_wrap(m) for m in messages[:2]]
    text = processor.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True
    )

    # Collect audio arrays (up to max_audio_turns)
    audio_arrays = []
    for item in messages[1]["content"]:
        if item["type"] == "audio" and len(audio_arrays) < max_audio_turns:
            arr = _load_wav(item["audio"])
            if arr is not None:
                audio_arrays.append(arr)

    # Process inputs
    try:
        inputs = processor(
            text=text,
            audio=audio_arrays if audio_arrays else None,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=3072,
        )
    except Exception:
        # Audio processing failed — fall back to text-only
        inputs = processor(
            text=text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=3072,
        )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    mm_id, el_id = label_first_ids
    with torch.no_grad():
        outputs = model(**inputs)
        last_logits = outputs.logits[0, -1, :]   # [vocab_size]

    score_mm = last_logits[mm_id].item()
    score_el = last_logits[el_id].item()
    pred     = "MANDATORY_MITIGATION" if score_mm > score_el else "ETHICAL_LEVERAGE"
    return pred, score_mm, score_el


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id",    default=MODEL_ID)
    parser.add_argument("--adapter",     default="sft_output/final_adapter",
                        help="Path to the LoRA adapter directory.")
    parser.add_argument("--jsonl",       default=None,
                        help="JSONL file to evaluate. Defaults to "
                             "dataset/val_sft.jsonl (test_sft.jsonl if it exists).")
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Evaluate only the first N samples (quick sanity check).")
    parser.add_argument("--save_preds",  action="store_true",
                        help="Save per-sample predictions to <adapter>/predictions.csv")
    args = parser.parse_args()

    # ── Resolve JSONL ──────────────────────────────────────────────────────────
    if args.jsonl:
        jsonl_path = Path(args.jsonl)
    else:
        test_path = Path("dataset/test_sft.jsonl")
        val_path  = Path("dataset/val_sft.jsonl")
        jsonl_path = test_path if test_path.exists() else val_path
    assert jsonl_path.exists(), f"JSONL not found: {jsonl_path}"

    split_name = jsonl_path.stem   # e.g. "val_sft"

    print(f"\n{'='*65}")
    print(f"  Eval — Qwen2.5-Omni-7B + LoRA")
    print(f"  Adapter : {args.adapter}")
    print(f"  Data    : {jsonl_path}")
    print(f"  Device  : {args.device}")
    print(f"{'='*65}\n")

    # ── Load records ───────────────────────────────────────────────────────────
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if args.max_samples:
        records = records[: args.max_samples]
    n = len(records)
    print(f"  Loaded {n:,} records from {jsonl_path.name}\n")

    # ── Baseline stats (majority-class oracle) ─────────────────────────────────
    gt_labels  = [r.get("label_str") or LABELS[r["label"]] for r in records]
    from collections import Counter
    class_counts = Counter(gt_labels)
    majority     = class_counts.most_common(1)[0][0]
    majority_acc = class_counts[majority] / n
    print(f"  Class distribution : {dict(class_counts)}")
    print(f"  Majority baseline  : {majority_acc*100:.1f}% (always predict {majority})\n")

    # ── Load model + processor ─────────────────────────────────────────────────
    hf_token  = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_id, token=hf_token)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = load_thinker(args.model_id, args.adapter, args.device)

    # ── Pre-compute first-token IDs for each label ─────────────────────────────
    mm_first_id = processor.tokenizer(
        "MANDATORY_MITIGATION", add_special_tokens=False
    ).input_ids[0]
    el_first_id = processor.tokenizer(
        "ETHICAL_LEVERAGE", add_special_tokens=False
    ).input_ids[0]
    assert mm_first_id != el_first_id, "First-token IDs must differ for logit comparison"
    print(f"  First-token IDs — MANDATORY_MITIGATION: {mm_first_id}  "
          f"ETHICAL_LEVERAGE: {el_first_id}\n")

    label_first_ids = (mm_first_id, el_first_id)

    # ── Inference loop ─────────────────────────────────────────────────────────
    y_true, y_pred = [], []
    rows      = []   # for per-sample CSV
    n_errors  = 0

    for rec in tqdm(records, desc="Evaluating", unit="sample", ncols=80):
        gt_str = rec.get("label_str") or LABELS[rec["label"]]
        try:
            pred, score_mm, score_el = predict_one(
                model, processor, rec, label_first_ids, args.device
            )
        except Exception as e:
            n_errors += 1
            pred     = majority          # safe fallback
            score_mm = score_el = float("nan")

        y_true.append(LABEL2ID[gt_str])
        y_pred.append(LABEL2ID[pred])
        rows.append({
            "source_path": rec.get("source_path", ""),
            "gt":          gt_str,
            "pred":        pred,
            "correct":     gt_str == pred,
            "score_mm":    score_mm,
            "score_el":    score_el,
            "margin":      score_mm - score_el,   # positive → model leans MM
        })

    # ── Metrics ────────────────────────────────────────────────────────────────
    acc    = accuracy_score(y_true, y_pred)
    f1_mm  = f1_score(y_true, y_pred, pos_label=1)
    f1_el  = f1_score(y_true, y_pred, pos_label=0)
    f1_mac = f1_score(y_true, y_pred, average="macro")
    cm     = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=LABELS, digits=4)

    n_correct = sum(t == p for t, p in zip(y_true, y_pred))

    # Bootstrap 95% CIs (skip if too few samples for meaningful resampling)
    cis = bootstrap_ci(y_true, y_pred) if n >= 50 else None

    print(f"\n{'='*65}")
    print(f"  RESULTS on {split_name}  ({n:,} samples)")
    print(f"{'='*65}")
    print(f"\n  Accuracy        : {acc*100:.2f}%  ({n_correct}/{n})", end="")
    if cis:
        lo, hi = cis["accuracy"]
        print(f"  95% CI [{lo*100:.2f}%, {hi*100:.2f}%]", end="")
    print()
    print(f"  Majority base   : {majority_acc*100:.1f}%  (Δ = {(acc - majority_acc)*100:+.1f}%)")
    print(f"  F1 macro        : {f1_mac:.4f}", end="")
    if cis:
        lo, hi = cis["f1_macro"]
        print(f"  95% CI [{lo:.4f}, {hi:.4f}]", end="")
    print()
    print(f"  F1 MM           : {f1_mm:.4f}")
    print(f"  F1 EL           : {f1_el:.4f}")
    if n_errors:
        print(f"  Inference errors: {n_errors}  (fell back to majority class)")

    print(f"\n  Confusion matrix  (rows = true, cols = predicted)")
    print(f"  {'':28s}  EL pred   MM pred")
    print(f"  {'ETHICAL_LEVERAGE (true)':28s}  {cm[0,0]:>7}   {cm[0,1]:>7}")
    print(f"  {'MANDATORY_MITIGATION (true)':28s}  {cm[1,0]:>7}   {cm[1,1]:>7}")
    print(f"\n  Per-class report:\n")
    print(report)

    # ── Error analysis — worst-confidence mistakes ─────────────────────────────
    df = pd.DataFrame(rows)
    mistakes = df[~df["correct"]].copy()
    if len(mistakes):
        # Sort by smallest margin (least confident, or confidently wrong)
        mistakes["abs_margin"] = mistakes["margin"].abs()
        worst = mistakes.nlargest(5, "abs_margin")
        print(f"  Worst-confidence mistakes (top 5 by |margin|):")
        for _, r in worst.iterrows():
            print(f"    gt={r['gt']:24s}  pred={r['pred']:24s}  margin={r['margin']:+.2f}")
        print()

    # ── Save results ───────────────────────────────────────────────────────────
    adapter_dir = Path(args.adapter)
    results = {
        "adapter":          args.adapter,
        "split":            split_name,
        "n_samples":        n,
        "accuracy":         acc,
        "majority_baseline": majority_acc,
        "delta_vs_majority": acc - majority_acc,
        "f1_macro":         f1_mac,
        "f1_MANDATORY_MITIGATION": f1_mm,
        "f1_ETHICAL_LEVERAGE":     f1_el,
        "confusion_matrix": cm.tolist(),
        "n_errors":         n_errors,
        "per_class":        classification_report(
            y_true, y_pred, target_names=LABELS, output_dict=True
        ),
        "bootstrap_ci_95":  cis,   # None if n < 50
    }
    results_path = adapter_dir / f"eval_{split_name}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  JSON results  → {results_path}")

    if args.save_preds:
        preds_path = adapter_dir / f"predictions_{split_name}.csv"
        df.to_csv(preds_path, index=False)
        print(f"  Predictions   → {preds_path}")

    print(f"\n{'='*65}\n  DONE\n{'='*65}\n")


if __name__ == "__main__":
    main()
