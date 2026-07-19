#!/usr/bin/env python3
"""
score_difficulty.py
===================
Score training samples with the SFT-finetuned model to produce
difficulty tiers for GRPO reinforcement learning.

Why SFT model, not base model?
  Base model accuracy = 49.6% (coin-flip). It has no signal on this task,
  so "hard for base model" ≈ everything. SFT model (83.6%) gives a genuine
  3-way split: confident right (EASY), uncertain right (MEDIUM), wrong (HARD).

Tier thresholds — calibrated from HELD-OUT val+test predictions:
  The SFT adapter already has predictions_val_sft.csv + predictions_test_sft.csv.
  We use P33 / P66 of abs(margin) on those 868 samples as tier boundaries.
  This is unbiased (val/test never in training) and transfers to training data.
  Empirically: P33 = 1.750, P66 = 2.875

  HARD   : abs_margin < P33  (SFT model uncertain — correct or wrong)
  MEDIUM : P33 ≤ abs_margin < P66  (borderline confident)
  EASY   : abs_margin ≥ P66  (SFT model confidently correct)

Template integrity — lighter touch than v1:
  Templates spanning all 3 tiers are WARNED about but NOT forcibly snapped.
  (Snapping inflated HARD to 76% in v1.) Natural spanning is fine when the
  template has dialogues of genuinely different difficulty.

Complexity cross-check:
  HARD-tier samples are joined against the training parquet to check whether
  they cluster around genuine complexity markers (vulnerability=high,
  sophistication=low, borderline price gap, high turn count) or are randomly
  distributed (suggesting label noise rather than real ambiguity).

Outputs
-------
  dataset/difficulty_scores.jsonl      2025 train records + SFT difficulty scores
  dataset/tier_easy.jsonl              EASY tier (SFT prompt format)
  dataset/tier_medium.jsonl            MEDIUM tier
  dataset/tier_hard.jsonl              HARD tier
  dataset/difficulty_stats.json        tier stats + complexity cross-check

Usage
-----
  CUDA_VISIBLE_DEVICES=1 python3 score_difficulty.py
  CUDA_VISIBLE_DEVICES=1 python3 score_difficulty.py --adapter sft_output/final_adapter
  CUDA_VISIBLE_DEVICES=1 python3 score_difficulty.py --max_samples 30  # quick test
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
from peft import PeftModel
from tqdm import tqdm
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

warnings.filterwarnings("ignore")
_logging.getLogger("root").setLevel(_logging.ERROR)

MODEL_ID      = "Qwen/Qwen2.5-Omni-7B"
DEFAULT_ADAPTER = "sft_output/final_adapter"
TARGET_SR     = 16_000

# Calibrated from val+test held-out predictions (868 samples, unbiased)
# P33 and P66 of abs(margin) give a balanced 3-way split (~33% each)
THRESH_P33 = 1.750   # abs_margin < this  → HARD
THRESH_P66 = 2.875   # abs_margin ≥ this  → EASY
# P33 ≤ abs_margin < P66 → MEDIUM


# ── Audio loader ──────────────────────────────────────────────────────────────

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


# ── Model loader ──────────────────────────────────────────────────────────────

def load_model(model_id: str, adapter_path: str | None, device: str):
    """
    Load Qwen2.5-Omni Thinker.
    If adapter_path is given, load the SFT LoRA adapter (merged for speed).
    Otherwise load raw base model.
    """
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    label = f"SFT adapter ({adapter_path})" if adapter_path else "base (no adapter)"
    print(f"  Loading {label}…")

    _full = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        token=hf_token,
    )
    thinker = _full.thinker
    if hasattr(_full, "talker"):    del _full.talker
    if hasattr(_full, "token2wav"): del _full.token2wav
    del _full
    gc.collect()

    if adapter_path:
        thinker = PeftModel.from_pretrained(thinker, adapter_path)
        thinker = thinker.merge_and_unload()
        print("  LoRA merged.")

    thinker = thinker.to(device)
    thinker.eval()
    print(f"  Model ready on {device}\n")
    return thinker


# ── Single-sample inference ───────────────────────────────────────────────────

def predict_one(model, processor, record: dict, label_first_ids: tuple[int, int],
                device: str, max_audio_turns: int = 12) -> tuple[str, float, float]:
    messages = record["messages"]

    def _wrap(msg):
        c = msg["content"]
        if isinstance(c, str):
            c = [{"type": "text", "text": c}]
        return {"role": msg["role"], "content": c}

    prompt_msgs = [_wrap(m) for m in messages[:2]]
    text = processor.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True
    )

    audio_arrays = []
    for item in messages[1]["content"]:
        if item["type"] == "audio" and len(audio_arrays) < max_audio_turns:
            arr = _load_wav(item["audio"])
            if arr is not None:
                audio_arrays.append(arr)

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
        out = model(**inputs)
        last_logits = out.logits[0, -1, :]

    score_mm = last_logits[mm_id].item()
    score_el = last_logits[el_id].item()
    pred = "MANDATORY_MITIGATION" if score_mm > score_el else "ETHICAL_LEVERAGE"
    return pred, score_mm, score_el


# ── Tier assignment ───────────────────────────────────────────────────────────

def assign_tiers(scored_rows: list[dict],
                 thresh_p33: float = THRESH_P33,
                 thresh_p66: float = THRESH_P66) -> tuple[list[dict], pd.DataFrame]:
    """
    Assign tiers using calibrated abs_margin thresholds derived from
    held-out val+test predictions (unbiased reference distribution).

    HARD   : abs_margin < thresh_p33
    MEDIUM : thresh_p33 ≤ abs_margin < thresh_p66
    EASY   : abs_margin ≥ thresh_p66
    """
    df = pd.DataFrame([{
        "idx":         i,
        "label":       r["label"],
        "label_str":   r["label_str"],
        "margin":      r["margin"],
        "abs_margin":  abs(r["margin"]),
        "is_correct":  r["is_correct"],
        "source_path": r["source_path"],
        "template_id": r["template_id"],
    } for i, r in enumerate(scored_rows)])

    n = len(df)
    global_mm_ratio = (df.label == 1).mean()

    sft_acc = df.is_correct.mean()
    mm_wrong = ((~df.is_correct) & (df.label == 1)).sum()
    el_wrong = ((~df.is_correct) & (df.label == 0)).sum()
    print(f"\n  SFT model accuracy on training data : {sft_acc*100:.1f}%  ({df.is_correct.sum()}/{n})")
    print(f"  Wrong: MM={mm_wrong}, EL={el_wrong}  "
          f"(note: slight upward bias — model was trained on these samples)")

    # ── Threshold-based assignment ────────────────────────────────────────────
    conditions = [
        df.abs_margin < thresh_p33,
        (df.abs_margin >= thresh_p33) & (df.abs_margin < thresh_p66),
        df.abs_margin >= thresh_p66,
    ]
    df["tier"] = np.select(conditions, ["HARD", "MEDIUM", "EASY"], default="MEDIUM")

    print(f"\n  Tier assignment (thresholds from held-out val+test P33={thresh_p33}, P66={thresh_p66}):")
    for tier_name in ["EASY", "MEDIUM", "HARD"]:
        grp = df[df.tier == tier_name]
        mm_n = (grp.label == 1).sum()
        el_n = (grp.label == 0).sum()
        acc  = grp.is_correct.mean()
        print(f"    {tier_name:<8}: {len(grp):>4} ({100*len(grp)/n:.0f}%)  "
              f"MM={mm_n}({100*mm_n/max(len(grp),1):.0f}%)  "
              f"EL={el_n}({100*el_n/max(len(grp),1):.0f}%)  "
              f"sft_acc={acc*100:.1f}%")

    # ── Template span check — warn only, don't force-snap ────────────────────
    span3 = []
    for tid, grp in df.groupby("template_id"):
        if len(set(grp["tier"].unique())) == 3:
            span3.append(tid)
    if span3:
        print(f"\n  [INFO] {len(span3)} templates span all 3 tiers (natural — not snapped):")
        print(f"    {span3}")
        print(f"  These templates have dialogues of genuinely different difficulty.")
        print(f"  Forcing them into fewer tiers would lose real signal.")

    # ── Write tier back ───────────────────────────────────────────────────────
    tier_map = dict(zip(df["idx"], df["tier"]))
    for i, r in enumerate(scored_rows):
        r["tier"] = tier_map[i]

    return scored_rows, df


# ── Complexity cross-check ────────────────────────────────────────────────────

def complexity_cross_check(df_tiers: pd.DataFrame, parquet_path: str) -> dict:
    """
    Join HARD/MEDIUM/EASY tiers against the tabular training parquet.
    Check whether HARD samples cluster around genuine complexity markers:
      - vulnerability=high (buyer susceptible)
      - sophistication=low (buyer naive)
      - anchoring=weak (buyer didn't anchor well)
      - high n_turns (long, complex negotiation)
      - borderline price_gap_ratio (close to 0 — genuinely ambiguous outcome)

    If HARD samples are random across these features → label noise dominates.
    If HARD samples skew toward the above → genuine complexity, CL will help.

    Returns a dict summary for difficulty_stats.json.
    """
    try:
        parquet_cols = ["source_path", "vulnerability", "sophistication", "anchoring",
                        "n_turns", "price_gap_ratio", "vaii_score", "template_id"]
        df_tab = pd.read_parquet(parquet_path, columns=parquet_cols)
    except Exception as e:
        print(f"\n  [WARN] Could not load parquet for cross-check: {e}")
        return {}

    df_merged = df_tiers.merge(df_tab, on="source_path", how="left",
                               suffixes=("", "_tab"))

    check = {}
    print(f"\n{'─'*65}")
    print(f"  COMPLEXITY CROSS-CHECK  —  HARD vs EASY tier feature distributions")
    print(f"{'─'*65}")

    for feat in ["vulnerability", "sophistication", "anchoring"]:
        # These are 0/1 encoded in parquet (high=1, strong=1)
        col = feat if feat in df_merged.columns else feat + "_tab"
        if col not in df_merged.columns:
            continue
        easy_mean = df_merged[df_merged.tier == "EASY"][col].mean()
        hard_mean = df_merged[df_merged.tier == "HARD"][col].mean()
        med_mean  = df_merged[df_merged.tier == "MEDIUM"][col].mean()
        check[feat] = {"EASY": round(easy_mean, 3), "MEDIUM": round(med_mean, 3),
                       "HARD": round(hard_mean, 3)}
        # Expected: vulnerability=high more in HARD, sophistication=low more in HARD
        flag = ""
        if feat == "vulnerability" and hard_mean > easy_mean + 0.05:
            flag = " ← HARD has more high-vulnerability (genuine complexity)"
        if feat == "sophistication" and hard_mean < easy_mean - 0.05:
            flag = " ← HARD has more low-sophistication buyers (genuine complexity)"
        if feat == "anchoring" and hard_mean < easy_mean - 0.05:
            flag = " ← HARD has weaker anchoring (genuine complexity)"
        print(f"  {feat:<16}: EASY={easy_mean:.3f}  MED={med_mean:.3f}  HARD={hard_mean:.3f}{flag}")

    for feat, col in [("n_turns", "n_turns"), ("price_gap_ratio", "price_gap_ratio"),
                      ("vaii_score", "vaii_score")]:
        if col not in df_merged.columns:
            continue
        easy_mean = df_merged[df_merged.tier == "EASY"][col].mean()
        hard_mean = df_merged[df_merged.tier == "HARD"][col].mean()
        med_mean  = df_merged[df_merged.tier == "MEDIUM"][col].mean()
        check[feat] = {"EASY": round(easy_mean, 3), "MEDIUM": round(med_mean, 3),
                       "HARD": round(hard_mean, 3)}
        flag = ""
        if feat == "n_turns" and hard_mean > easy_mean + 1:
            flag = " ← HARD negotiations are longer"
        if feat == "price_gap_ratio":
            # Borderline: price_gap_ratio close to 0 = ambiguous outcome
            easy_pgr_abs = df_merged[df_merged.tier == "EASY"][col].abs().mean()
            hard_pgr_abs = df_merged[df_merged.tier == "HARD"][col].abs().mean()
            if hard_pgr_abs < easy_pgr_abs - 0.05:
                flag = " ← HARD has borderline price outcomes (genuine ambiguity)"
            check["price_gap_ratio_abs"] = {
                "EASY": round(easy_pgr_abs, 3), "HARD": round(hard_pgr_abs, 3)
            }
        print(f"  {feat:<16}: EASY={easy_mean:.3f}  MED={med_mean:.3f}  HARD={hard_mean:.3f}{flag}")

    # Label noise check: within HARD, what fraction of WRONG predictions have
    # genuinely borderline price_gap_ratio (|pgr| < 0.05)?
    if "price_gap_ratio" in df_merged.columns:
        hard_wrong = df_merged[(df_merged.tier == "HARD") & (~df_merged.is_correct)]
        hard_borderline = (hard_wrong["price_gap_ratio"].abs() < 0.05).mean() if len(hard_wrong) else 0
        check["hard_wrong_borderline_pgr_frac"] = round(float(hard_borderline), 3)
        print(f"\n  Within HARD wrong predictions:")
        print(f"    {len(hard_wrong)} samples wrong; {hard_borderline*100:.1f}% have "
              f"|price_gap_ratio| < 0.05 (borderline outcomes → ambiguity, not noise)")

    # Template overlap check
    hard_templates   = set(df_merged[df_merged.tier == "HARD"]["template_id"])
    easy_templates   = set(df_merged[df_merged.tier == "EASY"]["template_id"])
    shared = hard_templates & easy_templates
    check["templates_in_hard"] = len(hard_templates)
    check["templates_in_easy"] = len(easy_templates)
    check["templates_shared_hard_easy"] = len(shared)
    print(f"\n  Template distribution:")
    print(f"    HARD uses {len(hard_templates)} unique templates, "
          f"EASY uses {len(easy_templates)}")
    print(f"    {len(shared)} templates appear in BOTH hard and easy "
          f"→ within-template difficulty variation is {'real' if len(shared) > 5 else 'low'}")

    print(f"{'─'*65}")
    return check


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id",    default=MODEL_ID)
    parser.add_argument("--adapter",     default=DEFAULT_ADAPTER,
                        help="Path to SFT LoRA adapter. Pass '' to use base model.")
    parser.add_argument("--train_jsonl", default="dataset/train_sft.jsonl")
    parser.add_argument("--train_parquet", default="dataset/train.parquet")
    parser.add_argument("--val_preds",   default="sft_output/final_adapter/predictions_val_sft.csv",
                        help="Pre-computed val predictions for threshold calibration")
    parser.add_argument("--test_preds",  default="sft_output/final_adapter/predictions_test_sft.csv",
                        help="Pre-computed test predictions for threshold calibration")
    parser.add_argument("--out_dir",     default="dataset/")
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Score only first N samples (quick test)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    adapter  = args.adapter.strip() or None

    print(f"\n{'='*65}")
    print(f"  DIFFICULTY SCORING  —  SFT-calibrated tiers")
    print(f"  Model  : {args.model_id}")
    print(f"  Adapter: {adapter or 'NONE (base model)' }")
    print(f"  Input  : {args.train_jsonl}")
    print(f"  Device : {args.device}")
    print(f"{'='*65}\n")

    # ── Recompute calibration thresholds from held-out predictions ────────────
    thresh_p33, thresh_p66 = THRESH_P33, THRESH_P66
    held_out_csvs = [p for p in [args.val_preds, args.test_preds] if Path(p).exists()]
    if held_out_csvs:
        dfs = [pd.read_csv(p) for p in held_out_csvs]
        ho = pd.concat(dfs)
        ho["abs_margin"] = ho["margin"].abs()
        thresh_p33 = float(ho["abs_margin"].quantile(0.33))
        thresh_p66 = float(ho["abs_margin"].quantile(0.66))
        print(f"  Calibration from {len(ho)} held-out samples "
              f"({', '.join(Path(p).name for p in held_out_csvs)})")
        print(f"  Thresholds — P33={thresh_p33:.3f}  P66={thresh_p66:.3f}\n")
    else:
        print(f"  [WARN] No held-out CSVs found; using default thresholds "
              f"P33={thresh_p33}  P66={thresh_p66}\n")

    # ── Load training records ─────────────────────────────────────────────────
    records = []
    with open(args.train_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if args.max_samples:
        records = records[:args.max_samples]
    n = len(records)
    print(f"  Loaded {n:,} training records\n")

    df_parquet = pd.read_parquet(args.train_parquet, columns=["source_path", "template_id"])
    path_to_template = dict(zip(df_parquet["source_path"], df_parquet["template_id"]))

    # ── Load model ────────────────────────────────────────────────────────────
    hf_token  = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_id, token=hf_token)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = load_model(args.model_id, adapter, args.device)

    mm_id = processor.tokenizer("MANDATORY_MITIGATION", add_special_tokens=False).input_ids[0]
    el_id = processor.tokenizer("ETHICAL_LEVERAGE",     add_special_tokens=False).input_ids[0]
    assert mm_id != el_id
    print(f"  First-token IDs — MM: {mm_id}  EL: {el_id}\n")
    label_ids = (mm_id, el_id)

    # ── Scoring loop ──────────────────────────────────────────────────────────
    scored_rows = []
    n_errors    = 0

    desc = "Scoring (SFT)" if adapter else "Scoring (base)"
    for rec in tqdm(records, desc=desc, unit="sample", ncols=80):
        gt_str      = rec.get("label_str") or ["ETHICAL_LEVERAGE", "MANDATORY_MITIGATION"][rec["label"]]
        source_path = rec.get("source_path", "")

        try:
            pred, score_mm, score_el = predict_one(model, processor, rec, label_ids, args.device)
        except Exception:
            n_errors += 1
            # Mark as NaN — will fall into HARD tier (most conservative)
            pred     = "MANDATORY_MITIGATION"
            score_mm = score_el = float("nan")

        margin     = score_mm - score_el
        is_correct = (pred == gt_str)

        scored_rows.append({
            "messages":       rec["messages"],
            "source_path":    source_path,
            "label":          rec["label"],
            "label_str":      gt_str,
            "n_audio_turns":  rec.get("n_audio_turns", 0),
            "n_missing_audio": rec.get("n_missing_audio", 0),
            "pred":           pred,
            "sft_score_mm":   score_mm,
            "sft_score_el":   score_el,
            "margin":         margin,
            "abs_margin":     abs(margin) if not np.isnan(margin) else 0.0,
            "is_correct":     is_correct,
            "template_id":    path_to_template.get(source_path, -1),
            "tier":           None,
        })

    if n_errors:
        print(f"\n  Inference errors (NaN margin → HARD): {n_errors}")

    # ── Assign tiers ──────────────────────────────────────────────────────────
    scored_rows, df_tiers = assign_tiers(scored_rows, thresh_p33, thresh_p66)

    # ── Complexity cross-check ────────────────────────────────────────────────
    complexity = complexity_cross_check(df_tiers, args.train_parquet)

    # ── Save outputs ──────────────────────────────────────────────────────────
    scores_path = out_dir / "difficulty_scores.jsonl"
    with open(scores_path, "w") as f:
        for r in scored_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n  Saved: {scores_path}  ({len(scored_rows):,} records)")

    sft_keys = {"messages", "source_path", "label", "label_str",
                "n_audio_turns", "n_missing_audio"}
    for tier_name in ["EASY", "MEDIUM", "HARD"]:
        tier_records = [r for r in scored_rows if r["tier"] == tier_name]
        tier_sft     = [{k: v for k, v in r.items() if k in sft_keys} for r in tier_records]
        tier_path    = out_dir / f"tier_{tier_name.lower()}.jsonl"
        with open(tier_path, "w") as f:
            for r in tier_sft:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  Saved: {tier_path}  ({len(tier_sft):,} records)")

    # Stats JSON
    tier_stats = {}
    for tier_name, grp in df_tiers.groupby("tier", observed=True):
        tier_stats[tier_name] = {
            "n":                 int(len(grp)),
            "n_mm":              int((grp.label == 1).sum()),
            "n_el":              int((grp.label == 0).sum()),
            "mm_pct":            round(float((grp.label == 1).mean()), 4),
            "sft_accuracy":      round(float(grp.is_correct.mean()), 4),
            "n_wrong":           int((~grp.is_correct).sum()),
            "median_abs_margin": round(float(grp.abs_margin.median()), 4),
            "mean_abs_margin":   round(float(grp.abs_margin.mean()), 4),
            "n_templates":       int(grp.template_id.nunique()),
        }

    template_tier_spans = {
        str(tid): sorted(grp["tier"].unique().tolist())
        for tid, grp in df_tiers.groupby("template_id", observed=True)
    }

    stats = {
        "model":              args.model_id,
        "adapter":            adapter,
        "n_scored":           n,
        "n_errors":           n_errors,
        "sft_accuracy_train": round(float(df_tiers.is_correct.mean()), 4),
        "global_mm_ratio":    round(float((df_tiers.label == 1).mean()), 4),
        "thresholds":         {"p33": thresh_p33, "p66": thresh_p66},
        "calibrated_from":    held_out_csvs,
        "tiers":              tier_stats,
        "complexity_check":   complexity,
        "template_tier_spans": template_tier_spans,
        "n_templates_spanning_3_tiers": sum(
            1 for v in template_tier_spans.values() if len(v) == 3),
    }
    stats_path = out_dir / "difficulty_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved: {stats_path}")

    print(f"\n{'='*65}\n  DONE\n{'='*65}\n")


if __name__ == "__main__":
    main()
