"""
validate_emotions.py — Analyse emotion label diversity across detected emotions.

Reads all intermediate JSONs that have emotion fields and reports:
  1. Total coverage (how many turns have emotions vs total)
  2. Full label frequency distribution (are we getting diversity or just calm/neutral?)
  3. Intensity distribution
  4. Valence distribution
  5. Confidence distribution
  6. Buyer vs Seller emotion breakdown
  7. Top-N most/least common labels
  8. Monotony score — if top-3 labels cover >70% of all turns, WAVs are monotonous

Usage:
  python3 Qwen3-tts/v2/validate_emotions.py
  python3 Qwen3-tts/v2/validate_emotions.py --min_confidence 0.7
"""

import json
import argparse
from pathlib import Path
from collections import Counter

V2_ROOT     = Path(__file__).parent
INTERM_DIR  = V2_ROOT / "dataset"


def main(min_confidence: float = 0.0):
    print(f"\n{'='*65}")
    print(f"  EMOTION DIVERSITY VALIDATION")
    print(f"{'='*65}\n")

    total_turns       = 0
    turns_with_emotion = 0
    buyer_labels      = []
    seller_labels     = []
    all_labels        = []
    intensities       = []
    valences          = []
    confidences       = []
    low_confidence    = []

    for fpath in sorted(INTERM_DIR.glob("dialogue_*.json")):
        try:
            data = json.loads(fpath.read_text())
        except Exception:
            continue
        for turn in data.get("processed", []):
            total_turns += 1
            em = turn.get("emotion")
            if not em:
                continue
            conf = em.get("confidence", 0.0)
            if conf < min_confidence:
                continue
            turns_with_emotion += 1
            labels    = em.get("labels", [])
            intensity = em.get("intensity", "unknown")
            valence   = em.get("valence", "unknown")
            speaker   = turn["speaker"]

            all_labels.extend(labels)
            intensities.append(intensity)
            valences.append(valence)
            confidences.append(conf)

            if speaker == "buyer":
                buyer_labels.extend(labels)
            else:
                seller_labels.extend(labels)

            if conf < 0.5:
                low_confidence.append({
                    "turn_id": f"{fpath.stem}_t{turn['turn_index']:02d}",
                    "labels":  labels,
                    "conf":    conf,
                    "notes":   em.get("notes", ""),
                })

    if turns_with_emotion == 0:
        print("  No emotion fields found. Run detect_emotions.py first.")
        return

    # ── Per-dialogue completeness check ───────────────────────────────────────
    complete_dialogues   = 0
    incomplete_dialogues = []

    for fpath in sorted(INTERM_DIR.glob("dialogue_*.json")):
        try:
            data = json.loads(fpath.read_text())
        except Exception:
            continue
        turns = data.get("processed", [])
        if not turns:
            continue
        missing = [
            f"t{t['turn_index']:02d}_{t['speaker']}"
            for t in turns if "emotion" not in t
        ]
        if missing:
            incomplete_dialogues.append((fpath.stem, missing))
        else:
            complete_dialogues += 1

    total_dialogues = complete_dialogues + len(incomplete_dialogues)
    print(f"\n  COMPLETENESS CHECK (per-dialogue)")
    print(f"  {'─'*45}")
    print(f"  Total dialogues      : {total_dialogues}")
    print(f"  Fully complete       : {complete_dialogues}  ({100*complete_dialogues/max(total_dialogues,1):.1f}%)")
    print(f"  Incomplete           : {len(incomplete_dialogues)}")
    if incomplete_dialogues:
        print(f"\n  Incomplete dialogues (first 20):")
        for did, missing in incomplete_dialogues[:20]:
            print(f"    {did}: missing {missing}")
        if len(incomplete_dialogues) > 20:
            print(f"    ... and {len(incomplete_dialogues)-20} more")
    else:
        print(f"  ✓  All dialogues have emotions on every turn.")

    print(f"  COVERAGE")
    print(f"  {'─'*45}")
    print(f"  Total turns          : {total_turns}")
    print(f"  Turns with emotion   : {turns_with_emotion}  ({100*turns_with_emotion/total_turns:.1f}%)")
    print(f"  Min confidence filter: {min_confidence}")

    # ── Label frequency ───────────────────────────────────────────────────────
    label_counts  = Counter(all_labels)
    total_labels  = sum(label_counts.values())
    unique_labels = len(label_counts)

    print(f"\n  LABEL DIVERSITY (all speakers)")
    print(f"  {'─'*45}")
    print(f"  Unique labels        : {unique_labels}")
    print(f"  Total label instances: {total_labels}")
    print(f"  Avg labels/turn      : {total_labels/turns_with_emotion:.2f}")
    print(f"\n  Top 20 labels:")
    for label, count in label_counts.most_common(20):
        pct = 100 * count / total_labels
        bar = "█" * int(pct / 2)
        print(f"    {label:<25} {count:>5}  ({pct:5.1f}%)  {bar}")

    # Monotony score
    top3_pct = sum(c for _, c in label_counts.most_common(3)) / total_labels * 100
    top5_pct = sum(c for _, c in label_counts.most_common(5)) / total_labels * 100
    print(f"\n  MONOTONY SCORE")
    print(f"  {'─'*45}")
    print(f"  Top-3 labels cover   : {top3_pct:.1f}% of all instances")
    print(f"  Top-5 labels cover   : {top5_pct:.1f}% of all instances")
    if top3_pct > 70:
        print(f"  ⚠  HIGH MONOTONY — top-3 labels cover >70%. WAVs may be emotionally flat.")
    elif top3_pct > 50:
        print(f"  △  MODERATE MONOTONY — some diversity but WAVs lean toward a few emotions.")
    else:
        print(f"  ✓  GOOD DIVERSITY — emotions are spread across many labels.")

    # ── Buyer vs Seller ───────────────────────────────────────────────────────
    print(f"\n  BUYER EMOTIONS (top 10)")
    print(f"  {'─'*45}")
    bc = Counter(buyer_labels)
    for label, count in bc.most_common(10):
        print(f"    {label:<25} {count:>5}  ({100*count/len(buyer_labels):.1f}%)")

    print(f"\n  SELLER EMOTIONS (top 10)")
    print(f"  {'─'*45}")
    sc = Counter(seller_labels)
    for label, count in sc.most_common(10):
        print(f"    {label:<25} {count:>5}  ({100*count/len(seller_labels):.1f}%)")

    # ── Intensity ─────────────────────────────────────────────────────────────
    print(f"\n  INTENSITY DISTRIBUTION")
    print(f"  {'─'*45}")
    ic = Counter(intensities)
    for k, v in ic.most_common():
        print(f"    {k:<15} {v:>5}  ({100*v/len(intensities):.1f}%)")

    # ── Valence ───────────────────────────────────────────────────────────────
    print(f"\n  VALENCE DISTRIBUTION")
    print(f"  {'─'*45}")
    vc = Counter(valences)
    for k, v in vc.most_common():
        print(f"    {k:<15} {v:>5}  ({100*v/len(valences):.1f}%)")

    # ── Confidence ────────────────────────────────────────────────────────────
    avg_conf = sum(confidences) / len(confidences)
    low_conf_count = sum(1 for c in confidences if c < 0.5)
    print(f"\n  CONFIDENCE")
    print(f"  {'─'*45}")
    print(f"  Average confidence   : {avg_conf:.3f}")
    print(f"  Low confidence (<0.5): {low_conf_count}  ({100*low_conf_count/len(confidences):.1f}%)")

    if low_confidence:
        print(f"\n  Low-confidence examples (first 5):")
        for item in low_confidence[:5]:
            print(f"    {item['turn_id']}: {item['labels']} (conf={item['conf']:.2f}) — {item['notes'][:80]}")

    # ── Rare labels (signal of real diversity) ────────────────────────────────
    rare = [(l, c) for l, c in label_counts.items() if c <= 2]
    print(f"\n  RARE LABELS (count ≤ 2) — {len(rare)} unique labels")
    print(f"  {'─'*45}")
    if rare:
        for label, count in sorted(rare, key=lambda x: x[1]):
            print(f"    {label:<25} {count}")
    else:
        print("    None — all labels appear >2 times")

    print(f"\n{'='*65}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--min_confidence", type=float, default=0.0,
                   help="Only count turns with confidence >= this value")
    args = p.parse_args()
    main(min_confidence=args.min_confidence)
