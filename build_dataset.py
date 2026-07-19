#!/usr/bin/env python3
"""
build_dataset.py
================
Builds a multimodal SFT dataset for Qwen2.5-Omni-7B fine-tuning on
negotiation bias detection, using BOTH:

  SPEECH modality — actual .wav files per turn, fed to Qwen2.5-Omni's
                    Whisper-based audio encoder so it hears real prosody,
                    emotional arousal, pace, and vocal stress directly.

  TEXT modality   — turn transcripts, vulnerability labels, price context,
                    and aggregated acoustic features (VAII, F0, etc.) as
                    supplementary structured text.

Qwen2.5-Omni native format (per turn in the user message):
  {"type": "audio", "audio": "/abs/path/turn_01_buyer.wav"}
  {"type": "text",  "text":  "Turn 01 [Buyer]: \"I'm interested but...\""}
  ... repeated for every turn ...
  {"type": "text",  "text":  "=== CONTEXT ===\n...\n→ Classify:"}

This lets the model attend to the actual voice (prosody/emotion) AND the
words simultaneously, rather than only seeing derived numeric summaries.

Filters  : vaii_score > 0.0  (drops dialogues with missing TTS audio)
Imbalance: MANDATORY_MITIGATION 66.9% vs ETHICAL_LEVERAGE 33.1% (2:1).
           Handled via class weights saved to class_weights.json for the
           SFT trainer's weighted cross-entropy loss.

Outputs:
  dataset/full_dataset.parquet   — tabular feature rows (2,893 rows)
  dataset/train.parquet          — 70% stratified
  dataset/val.parquet            — 15% stratified
  dataset/test.parquet           — 15% stratified
  dataset/train_sft.jsonl        — Qwen2.5-Omni multimodal prompts (train)
  dataset/val_sft.jsonl          — Qwen2.5-Omni multimodal prompts (val)
  dataset/class_weights.json     — {0: w_EL, 1: w_MM} for loss weighting
  dataset/dataset_stats.json     — distribution + coverage report

Usage:
  python build_dataset.py
  python build_dataset.py --preprocessed_root preprocessed/ \\
                          --audio_root tts_outputs/ \\
                          --out_dir dataset/
  python build_dataset.py --append   # merge new rows after remaining TTS jobs
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


# ── Encoders ──────────────────────────────────────────────────────────────────
VULN_MAP  = {"low": 0, "high": 1}
SOPH_MAP  = {"low": 0, "high": 1}
ANCH_MAP  = {"weak": 0, "strong": 1}
LABEL_MAP = {"ETHICAL_LEVERAGE": 0, "MANDATORY_MITIGATION": 1}

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a negotiation bias detection agent.\n"
    "You will receive a negotiation dialogue where each turn consists of the "
    "actual speech audio followed by its transcript. Listen carefully to the "
    "vocal tone, pacing, emotional arousal, and stress in each speaker's voice, "
    "and read the words they say. Use both the acoustic and linguistic signals "
    "together to classify the negotiation as:\n"
    "  MANDATORY_MITIGATION — the buyer is being exploited; the agent MUST intervene\n"
    "  ETHICAL_LEVERAGE     — the situation is ethically acceptable\n\n"
    "Respond with exactly one of those two labels."
)


# ── Wav path resolver ─────────────────────────────────────────────────────────

def resolve_wav(audio_root: Path, json_path: Path, turn_num: int, speaker: str) -> Path | None:
    parts = json_path.parts
    try:
        idx = next(i for i, p in enumerate(parts) if p == "preprocessed")
    except StopIteration:
        return None
    wav = (audio_root
           / parts[idx + 1]          # template_XX
           / parts[idx + 2]          # range_Y-Z
           / json_path.stem          # dialogue_NNN
           / f"turn_{turn_num:02d}_{speaker.lower()}.wav")
    if wav.exists() and wav.stat().st_size > 0:
        return wav
    # Fallback: glob for any wav matching the turn number
    parent = wav.parent
    if parent.exists():
        candidates = [f for f in parent.glob(f"turn_{turn_num:02d}_*.wav")
                      if f.stat().st_size > 0]
        if candidates:
            return candidates[0]
    return None


# ── SFT prompt builder ────────────────────────────────────────────────────────

def build_qwen_omni_message(d: dict, json_path: Path, audio_root: Path) -> tuple[list, str, int, int]:
    """
    Build the Qwen2.5-Omni user message content list and assistant response.

    Each dialogue turn becomes two consecutive content items:
      1. {"type": "audio", "audio": "/abs/path/turn_XX_speaker.wav"}
         → fed to Qwen2.5-Omni's Whisper audio encoder (real speech)
      2. {"type": "text",  "text": "Turn XX [Speaker]: \"...transcript...\""}
         → fed to the LLM for linguistic understanding

    After all turns, a classification prompt is appended.
    No price context, VAII scores, or vulnerability labels are included —
    all of these directly encode or are 100%-correlated with the label.

    Returns (content_list, label_str, n_audio_turns, n_missing_audio).
    """
    meta   = d.get("metadata", {})
    labels = meta.get("labels", {})

    content = []
    n_audio = 0
    n_missing = 0

    # ── Per-turn: audio clip + transcript ─────────────────────────────────────
    for turn in d.get("trajectory", []):
        t_num   = turn.get("turn", 0)
        speaker = turn.get("speaker", "?")
        text    = (turn.get("tts_text") or turn.get("text", "")).strip()

        wav = resolve_wav(audio_root, json_path, t_num, speaker)

        if wav is not None:
            # Real speech — Qwen2.5-Omni's audio encoder will process this
            content.append({"type": "audio", "audio": str(wav)})
            n_audio += 1
        else:
            # Fallback: note that audio is unavailable for this turn
            content.append({"type": "text",
                            "text": f"[Turn {t_num:02d} audio unavailable]"})
            n_missing += 1

        # Always include the transcript alongside the audio
        content.append({
            "type": "text",
            "text": f'Turn {t_num:02d} [{speaker}]: "{text}"'
        })

    # ── Classification prompt (no price/VAII/vulnerability context — all leak the label) ──
    content.append({"type": "text",
                    "text": "→ Based on what you heard and read above, classify this negotiation:"})

    label_str = labels.get("bias_decision", "?")
    return content, label_str, n_audio, n_missing


# ── Row extractor (tabular + SFT) ─────────────────────────────────────────────

def extract_row(jp: Path, audio_root: Path) -> tuple[dict | None, dict | None]:
    try:
        with open(jp) as f:
            d = json.load(f)
    except Exception:
        return None, None

    meta   = d.get("metadata", {})
    labels = meta.get("labels", {})
    prices = meta.get("prices", {})
    mv     = meta.get("calculated_vaii_from_speech", {})

    vaii_score = mv.get("dialogue_vaii_score", 0.0)
    if vaii_score <= 0.0:
        return None, None

    required = ["vulnerability", "sophistication", "anchoring",
                "harm_direction", "bias_decision"]
    if any(k not in labels for k in required):
        return None, None

    label = LABEL_MAP.get(labels["bias_decision"], -1)
    if label == -1:
        return None, None

    fair   = prices.get("fair_price")
    b_open = prices.get("buyer_first_offer")
    final  = prices.get("final_price")

    tabular = {
        "template_id":        meta.get("template_id", -1),
        "range_id":           meta.get("range_id", ""),
        "dialogue_id":        jp.stem,
        "source_path":        str(jp),
        "vulnerability":      VULN_MAP.get(labels["vulnerability"], -1),
        "sophistication":     SOPH_MAP.get(labels["sophistication"], -1),
        "anchoring":          ANCH_MAP.get(labels["anchoring"], -1),
        # harm_direction = target rephrased — metadata only, NOT a model input
        "harm_direction_meta": labels.get("harm_direction", ""),
        "vulnerability_str":  labels["vulnerability"],
        "sophistication_str": labels["sophistication"],
        "anchoring_str":      labels["anchoring"],
        "vaii_score":         round(vaii_score, 6),
        "vaii_class":         1 if vaii_score >= 0.70 else 0,
        "price_gap_ratio":    round((final - fair) / fair, 6) if (fair and final) else 0.0,
        "buyer_anchor_off":   round((b_open - fair) / fair, 6) if (fair and b_open) else 0.0,
        "fair_price":         fair,
        "buyer_first_offer":  b_open,
        "buyer_last_offer":   prices.get("buyer_last_offer"),
        "final_price":        final,
        "n_turns":            len(d.get("trajectory", [])),
        "label":              label,
        "label_str":          labels["bias_decision"],
    }

    # ── Qwen2.5-Omni SFT record ───────────────────────────────────────────────
    try:
        user_content, label_str, n_audio, n_missing = build_qwen_omni_message(
            d, jp, audio_root
        )
        sft = {
            # Qwen2.5-Omni chat format
            "messages": [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": user_content},
                {"role": "assistant", "content": label_str},
            ],
            # Metadata for filtering / debugging
            "source_path":    str(jp),
            "label":          label,
            "label_str":      label_str,
            "n_audio_turns":  n_audio,
            "n_missing_audio": n_missing,
        }
    except Exception as e:
        sft = None

    return tabular, sft


# ── Dataset builder ───────────────────────────────────────────────────────────

def build_dataframe(preprocessed_root: Path, audio_root: Path):
    rows, sfts, skipped = [], [], 0
    missing_audio_total = 0

    for jp in sorted(preprocessed_root.rglob("*.json")):
        row, sft = extract_row(jp, audio_root)
        if row is None:
            skipped += 1
        else:
            rows.append(row)
            if sft:
                sfts.append(sft)
                missing_audio_total += sft.get("n_missing_audio", 0)

    print(f"  Extracted   : {len(rows):,} rows  ({len(sfts):,} with SFT prompts)")
    print(f"  Skipped     : {skipped:,}  (zero VAII / incomplete labels)")
    if sfts:
        total_turns = sum(s["n_audio_turns"] + s["n_missing_audio"] for s in sfts)
        audio_turns = sum(s["n_audio_turns"] for s in sfts)
        print(f"  Audio turns : {audio_turns:,}/{total_turns:,} "
              f"({100*audio_turns/total_turns:.1f}% have real .wav)")
        if missing_audio_total:
            print(f"  Missing wav : {missing_audio_total:,} turns fell back to text-only")

    return pd.DataFrame(rows), sfts


def compute_weights(labels: pd.Series) -> dict:
    classes = np.array([0, 1])
    w = compute_class_weight("balanced", classes=classes, y=labels.values)
    return {int(c): round(float(v), 6) for c, v in zip(classes, w)}


def stratified_split(df, val_size=0.15, test_size=0.15, seed=42):
    tr_val, test = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=seed)
    train, val = train_test_split(
        tr_val, test_size=val_size / (1 - test_size),
        stratify=tr_val["label"], random_state=seed)
    return (train.reset_index(drop=True),
            val.reset_index(drop=True),
            test.reset_index(drop=True))


def split_sfts(sfts, train_paths, val_paths, test_paths):
    return (
        [s for s in sfts if s["source_path"] in train_paths],
        [s for s in sfts if s["source_path"] in val_paths],
        [s for s in sfts if s["source_path"] in test_paths],
    )


def write_jsonl(records, path):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def print_split_stats(name, df):
    mm, el, n = (df["label"]==1).sum(), (df["label"]==0).sum(), len(df)
    print(f"  {name:<8}: {n:>5} rows  |  MM={mm} ({100*mm/n:.1f}%)  "
          f"EL={el} ({100*el/n:.1f}%)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed_root", default="preprocessed/")
    parser.add_argument("--audio_root",        default="tts_outputs/")
    parser.add_argument("--out_dir",           default="dataset/")
    parser.add_argument("--val_size",  type=float, default=0.15)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--append",    action="store_true",
                        help="Merge new rows after TTS jobs finish")
    args = parser.parse_args()

    preprocessed_root = Path(args.preprocessed_root).resolve()
    audio_root        = Path(args.audio_root).resolve()
    out_dir           = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  BUILD MULTIMODAL SFT DATASET  —  Qwen2.5-Omni-7B")
    print(f"  Speech (.wav per turn) + Text → interleaved audio+text prompts")
    print(f"{'='*65}")
    print(f"  Preprocessed : {preprocessed_root}")
    print(f"  Audio root   : {audio_root}")
    print(f"  Output       : {out_dir}\n")

    df, sfts = build_dataframe(preprocessed_root, audio_root)

    # ── Append mode ───────────────────────────────────────────────────────────
    full_path = out_dir / "full_dataset.parquet"
    if args.append and full_path.exists():
        existing = pd.read_parquet(full_path)
        before   = len(existing)
        df = pd.concat([existing, df], ignore_index=True).drop_duplicates(
            subset=["source_path"]).reset_index(drop=True)
        print(f"  Append: {before:,} existing + {len(df)-before:,} new = {len(df):,} total\n")

    # ── Class distribution ────────────────────────────────────────────────────
    mm, el = (df["label"]==1).sum(), (df["label"]==0).sum()
    ratio  = mm / el
    print(f"  Class distribution:")
    print(f"    MANDATORY_MITIGATION (1): {mm:>5}  ({100*mm/len(df):.1f}%)")
    print(f"    ETHICAL_LEVERAGE     (0): {el:>5}  ({100*el/len(df):.1f}%)")
    print(f"    Imbalance MM:EL          {ratio:.2f}:1")

    cw = compute_weights(df["label"])
    print(f"\n  Class weights (balanced):")
    print(f"    w[EL=0]  = {cw[0]:.4f}  ← minority upweighted")
    print(f"    w[MM=1]  = {cw[1]:.4f}  ← majority downweighted")

    # ── Splits ────────────────────────────────────────────────────────────────
    train, val, test = stratified_split(df, args.val_size, args.test_size, args.seed)
    print(f"\n  Stratified splits (seed={args.seed}):")
    print_split_stats("train", train)
    print_split_stats("val",   val)
    print_split_stats("test",  test)

    train_sft, val_sft, test_sft = split_sfts(
        sfts, set(train["source_path"]), set(val["source_path"]), set(test["source_path"]))
    print(f"\n  SFT prompt files (Qwen2.5-Omni format):")
    print(f"    train_sft.jsonl : {len(train_sft):,} prompts")
    print(f"    val_sft.jsonl   : {len(val_sft):,} prompts")
    print(f"    test_sft.jsonl  : {len(test_sft):,} prompts")

    # ── Save ──────────────────────────────────────────────────────────────────
    df.to_parquet(full_path, index=False)
    train.to_parquet(out_dir / "train.parquet", index=False)
    val.to_parquet(out_dir / "val.parquet",     index=False)
    test.to_parquet(out_dir / "test.parquet",   index=False)
    write_jsonl(train_sft, out_dir / "train_sft.jsonl")
    write_jsonl(val_sft,   out_dir / "val_sft.jsonl")
    write_jsonl(test_sft,  out_dir / "test_sft.jsonl")

    (out_dir / "class_weights.json").write_text(json.dumps({
        "0_ETHICAL_LEVERAGE":     cw[0],
        "1_MANDATORY_MITIGATION": cw[1],
        "model":  "Qwen/Qwen2.5-Omni-7B-Instruct",
        "note": (
            "Apply as loss weights in SFT trainer weighted cross-entropy. "
            "harm_direction, vulnerability, sophistication, anchoring, and VAII scores "
            "are ALL excluded from prompts — they directly encode the label."
        ),
    }, indent=2))

    feature_cols = ["vulnerability", "sophistication", "anchoring",
                    "vaii_score", "vaii_class", "price_gap_ratio",
                    "buyer_anchor_off", "n_turns"]
    stats = {
        "model":              "Qwen/Qwen2.5-Omni-7B-Instruct",
        "total_rows":         len(df),
        "train_rows":         len(train),
        "val_rows":           len(val),
        "test_rows":          len(test),
        "sft_train_prompts":  len(train_sft),
        "sft_val_prompts":    len(val_sft),
        "sft_test_prompts":   len(test_sft),
        "class_counts":       {"MANDATORY_MITIGATION": int(mm),
                               "ETHICAL_LEVERAGE": int(el)},
        "imbalance_ratio":    round(ratio, 4),
        "class_weights":      cw,
        "modalities": {
            "speech": "Per-turn .wav files passed directly to Qwen2.5-Omni "
                      "Whisper audio encoder — real prosody, emotion, arousal",
            "text":   "Turn transcripts + VAII summary + vulnerability labels "
                      "+ price context as structured text",
            "format": "Interleaved: [audio_turn1, text_turn1, audio_turn2, "
                      "text_turn2, ..., context_block]",
            "harm_direction": "EXCLUDED from prompts — target leak (100% corr with label)",
        },
        "audio_coverage":     {
            "total_turns_in_train": sum(
                s["n_audio_turns"] + s["n_missing_audio"] for s in train_sft),
            "audio_turns_in_train": sum(s["n_audio_turns"] for s in train_sft),
            "missing_turns_in_train": sum(s["n_missing_audio"] for s in train_sft),
        },
        "feature_means":      df[feature_cols].mean().round(4).to_dict(),
        "feature_stds":       df[feature_cols].std().round(4).to_dict(),
        "templates_included": sorted(df["template_id"].unique().tolist()),
    }
    (out_dir / "dataset_stats.json").write_text(json.dumps(stats, indent=2))

    print(f"\n  Saved to {out_dir}/:")
    for fname in ["full_dataset.parquet", "train.parquet", "val.parquet",
                  "test.parquet", "train_sft.jsonl", "val_sft.jsonl",
                  "test_sft.jsonl", "class_weights.json", "dataset_stats.json"]:
        size = (out_dir / fname).stat().st_size
        print(f"    {fname:<28} {size/1024:>8.1f} KB")

    # ── Sample prompt preview ─────────────────────────────────────────────────
    if train_sft:
        s = train_sft[0]
        print(f"\n{'─'*65}")
        print(f"  SAMPLE SFT RECORD (train[0] | label={s['label_str']} | "
              f"audio={s['n_audio_turns']} turns)")
        print(f"{'─'*65}")
        print(f"  [SYSTEM] {s['messages'][0]['content'][:120]}...\n")
        print(f"  [USER content items: {len(s['messages'][1]['content'])} total]")
        for item in s["messages"][1]["content"][:6]:
            if item["type"] == "audio":
                print(f"    audio → {Path(item['audio']).name}")
            else:
                print(f"    text  → {item['text'][:90]}")
        print(f"    ... ({len(s['messages'][1]['content'])-6} more items)")
        print(f"\n  [ASSISTANT] {s['messages'][2]['content']}")
        print(f"{'─'*65}")

    print(f"\n{'='*65}\n  DONE\n{'='*65}\n")


if __name__ == "__main__":
    main()
