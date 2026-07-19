#!/usr/bin/env python3
"""
error_analysis.py — Deep error analysis of SFT inference outputs.

Reads inference_*.json, analyses ALL wrong predictions in depth across:
  - Confusion direction (LEVERAGE→MITIGATE, MITIGATE→LEVERAGE, etc.)
  - Negotiation zone (pre/post crystallisation)
  - Decision flip turns
  - Buyer emotion (label type, intensity, valence)
  - Price proximity to fair value
  - Anchoring strength
  - Response quality (length, repetition across dialogue turns)
  - Format failures

Outputs a markdown report: error_analysis_{timestamp}.md
alongside the inference file.

This report is designed to directly inform GRPO reward function design.

Usage:
  python3 Qwen3-tts/v2/error_analysis.py --version v4
  python3 Qwen3-tts/v2/error_analysis.py --version v5
  python3 Qwen3-tts/v2/error_analysis.py \\
      --inference_file sft_output_v4/.../inference_*.json
"""

import argparse
import datetime
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

V2_ROOT    = Path(__file__).parent
SFT_OUT_V4 = V2_ROOT / "sft_output_v4"
SFT_OUT_V5 = V2_ROOT / "sft_output_v5"
SFT_OUT_V6 = V2_ROOT / "sft_output_v6"


# ── Args ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--version",        default="v4", choices=["v4", "v5", "v6"])
    p.add_argument("--inference_file", default=None,
                   help="Explicit text inference file. If omitted, auto-detects latest.")
    return p.parse_args()


def find_inference_files(version):
    """Returns (text_inference_path, speech_inference_path_or_None)."""
    sft_dir = {"v4": SFT_OUT_V4, "v5": SFT_OUT_V5, "v6": SFT_OUT_V6}[version]
    run_dirs = sorted([d for d in sft_dir.iterdir() if d.is_dir()],
                      key=lambda d: d.name, reverse=True)
    text_path, speech_path = None, None
    for rd in run_dirs:
        candidates = sorted(rd.glob("inference_*.json"), reverse=True)
        for c in candidates:
            if "speech" in c.name and speech_path is None:
                speech_path = c
            elif "speech" not in c.name and text_path is None:
                text_path = c
        if text_path:
            break
    if not text_path:
        sys.exit(f"No inference_*.json found in {sft_dir}/")
    return text_path, speech_path


# ── Helpers ────────────────────────────────────────────────────────────────────

def price_gap_pct(buyer_offers, fair_value):
    """Closest buyer offer as % below fair value. Negative = below fair."""
    if not buyer_offers or not fair_value:
        return None
    closest = max(buyer_offers)  # highest offer = closest to fair value
    return round(100 * (closest - fair_value) / fair_value, 1)


def response_word_count(text):
    if not text:
        return 0
    return len(text.split())


def ngram_overlap(text1, text2, n=3):
    """Fraction of n-grams in text2 that also appear in text1."""
    if not text1 or not text2:
        return 0.0
    def ngrams(t, n):
        words = t.lower().split()
        return set(tuple(words[i:i+n]) for i in range(len(words)-n+1))
    g1, g2 = ngrams(text1, n), ngrams(text2, n)
    if not g2:
        return 0.0
    return len(g1 & g2) / len(g2)


def repetition_score_for_dialogue(dialogue_responses):
    """
    For a list of seller responses in one dialogue (in order),
    return avg pairwise 3-gram overlap between consecutive turns.
    High score = repetitive.
    """
    if len(dialogue_responses) < 2:
        return 0.0
    overlaps = []
    for i in range(1, len(dialogue_responses)):
        overlaps.append(ngram_overlap(dialogue_responses[i-1], dialogue_responses[i]))
    return round(mean(overlaps), 3)


def emotion_summary(em):
    if not em:
        return "none", "none", "none"
    labels    = ", ".join(em.get("labels") or []) or "none"
    intensity = em.get("intensity") or "none"
    valence   = em.get("valence")   or "none"
    return labels, intensity, valence


# ── Main analysis ──────────────────────────────────────────────────────────────

def analyse(examples, version):
    n_total   = len(examples)
    errors    = [e for e in examples if not e["decision_correct"]]
    correct   = [e for e in examples if e["decision_correct"]]
    n_errors  = len(errors)
    n_correct = len(correct)

    lines = []
    def h(text): lines.append(f"\n## {text}\n")
    def h2(text): lines.append(f"\n### {text}\n")
    def ln(text=""): lines.append(text)
    def tbl(headers, rows):
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---"]*len(headers)) + "|")
        for r in rows:
            lines.append("| " + " | ".join(str(x) for x in r) + " |")
        lines.append("")

    # ── Header ────────────────────────────────────────────────────────────────
    lines.append(f"# Error Analysis — SFT {version.upper()}")
    lines.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    lines.append(f"**Total examples:** {n_total}  ")
    lines.append(f"**Correct:** {n_correct} ({100*n_correct/n_total:.1f}%)  ")
    lines.append(f"**Errors:** {n_errors} ({100*n_errors/n_total:.1f}%)\n")

    if n_errors == 0:
        lines.append("No errors found.")
        return "\n".join(lines)

    # ── 1. Confusion matrix ───────────────────────────────────────────────────
    h("1. Confusion Matrix")
    confusion = Counter((e["gt_decision"], e["pred_decision"] or "MISSING") for e in examples)
    decisions = ["LEVERAGE", "MITIGATE", "UNDECIDED"]
    tbl(
        ["GT \\ Pred"] + decisions + ["MISSING"],
        [
            [gt] + [confusion.get((gt, pred), 0) for pred in decisions] + [confusion.get((gt, "MISSING"), 0)]
            for gt in decisions
        ]
    )

    # Directional error breakdown
    h2("Directional breakdown of errors")
    dir_counts = Counter((e["gt_decision"], e["pred_decision"] or "MISSING") for e in errors)
    rows = sorted(dir_counts.items(), key=lambda x: -x[1])
    tbl(
        ["GT → Pred", "Count", "% of errors", "% of total"],
        [
            (f"{gt} → {pred}", cnt,
             f"{100*cnt/n_errors:.1f}%",
             f"{100*cnt/n_total:.1f}%")
            for (gt, pred), cnt in rows
        ]
    )
    ln("**GRPO reward implication:** The dominant error direction should get the heaviest penalty in the decision accuracy reward.")

    # ── 2. Zone analysis ──────────────────────────────────────────────────────
    h("2. Errors by Negotiation Zone")
    for zone in ["pre_crystallisation", "post_crystallisation", ""]:
        zone_label = zone if zone else "unknown"
        ze = [e for e in examples if e.get("zone","") == zone]
        ze_err = [e for e in ze if not e["decision_correct"]]
        if ze:
            ln(f"**{zone_label}:** {len(ze_err)}/{len(ze)} errors ({100*len(ze_err)/len(ze):.1f}%)")

    ln("")
    ln("**GRPO reward implication:** If post-crystallisation accuracy is lower, reward correct decisions in that zone more heavily.")

    # ── 3. Decision flip analysis ─────────────────────────────────────────────
    h("3. Decision Flip Turn Errors")
    flip_ex    = [e for e in examples if e.get("decision_flip")]
    flip_err   = [e for e in flip_ex   if not e["decision_correct"]]
    noflip_ex  = [e for e in examples  if not e.get("decision_flip")]
    noflip_err = [e for e in noflip_ex if not e["decision_correct"]]

    ln(f"**Flip turns:**    {len(flip_err)}/{len(flip_ex)} errors ({100*len(flip_err)/len(flip_ex):.1f}%)" if flip_ex else "No flip turns in test set.")
    ln(f"**Non-flip turns:** {len(noflip_err)}/{len(noflip_ex)} errors ({100*len(noflip_err)/len(noflip_ex):.1f}%)" if noflip_ex else "")
    ln("")

    if flip_err:
        h2("Flip error directions")
        flip_dir = Counter((e["gt_decision"], e["pred_decision"] or "MISSING") for e in flip_err)
        tbl(["GT → Pred", "Count"],
            [(f"{gt} → {pred}", cnt) for (gt, pred), cnt in sorted(flip_dir.items(), key=lambda x: -x[1])])

    ln("**GRPO reward implication:** Add a bonus reward for correctly handling flip turns — these are the highest-stakes moments in a negotiation.")

    # ── 4. Buyer emotion analysis ─────────────────────────────────────────────
    h("4. Errors by Buyer Emotion")

    h2("By emotion valence")
    valence_groups = defaultdict(list)
    for e in examples:
        _, _, valence = emotion_summary(e.get("prior_buyer_emotion"))
        valence_groups[valence].append(e)
    tbl(
        ["Valence", "Total", "Errors", "Error %"],
        sorted([
            (v, len(exs), sum(1 for x in exs if not x["decision_correct"]),
             f"{100*sum(1 for x in exs if not x['decision_correct'])/len(exs):.1f}%")
            for v, exs in valence_groups.items()
        ], key=lambda r: -int(r[2]))
    )

    h2("By emotion intensity")
    intensity_groups = defaultdict(list)
    for e in examples:
        _, intensity, _ = emotion_summary(e.get("prior_buyer_emotion"))
        intensity_groups[intensity].append(e)
    tbl(
        ["Intensity", "Total", "Errors", "Error %"],
        sorted([
            (i, len(exs), sum(1 for x in exs if not x["decision_correct"]),
             f"{100*sum(1 for x in exs if not x['decision_correct'])/len(exs):.1f}%")
            for i, exs in intensity_groups.items()
        ], key=lambda r: -int(r[2]))
    )

    h2("Most common emotion labels in error examples")
    label_counter = Counter()
    for e in errors:
        em = e.get("prior_buyer_emotion") or {}
        for lbl in (em.get("labels") or []):
            label_counter[lbl] += 1
    tbl(["Emotion label", "Appears in N errors"],
        label_counter.most_common(10))

    h2("Most common emotion labels in correct examples")
    label_counter_correct = Counter()
    for e in correct:
        em = e.get("prior_buyer_emotion") or {}
        for lbl in (em.get("labels") or []):
            label_counter_correct[lbl] += 1
    tbl(["Emotion label", "Appears in N correct"],
        label_counter_correct.most_common(10))

    ln("**GRPO reward implication:** Emotions with higher error rates are the ones the model underweights.")
    ln("Add `reasoning_mentions_emotion` reward — penalise outputs where reasoning ignores these specific emotion labels.")

    # ── 5. Price proximity analysis ───────────────────────────────────────────
    h("5. Errors by Price Proximity to Fair Value")
    ln("*(buyer's highest offer as % of fair value — negative = below fair value)*\n")

    buckets = {
        "very far below (<-20%)":  [],
        "far below (-20% to -10%)": [],
        "near below (-10% to -5%)": [],
        "boundary (-5% to 0%)":    [],
        "at or above fair (>=0%)": [],
        "unknown":                 [],
    }
    for e in examples:
        gap = price_gap_pct(e.get("buyer_offers"), e.get("fair_value"))
        if gap is None:
            buckets["unknown"].append(e)
        elif gap >= 0:
            buckets["at or above fair (>=0%)"].append(e)
        elif gap >= -5:
            buckets["boundary (-5% to 0%)"].append(e)
        elif gap >= -10:
            buckets["near below (-10% to -5%)"].append(e)
        elif gap >= -20:
            buckets["far below (-20% to -10%)"].append(e)
        else:
            buckets["very far below (<-20%)"].append(e)

    tbl(
        ["Price bucket", "Total", "Errors", "Error %"],
        [
            (bkt, len(exs), sum(1 for x in exs if not x["decision_correct"]),
             f"{100*sum(1 for x in exs if not x['decision_correct'])/len(exs):.1f}%" if exs else "—")
            for bkt, exs in buckets.items() if exs
        ]
    )
    ln("**GRPO reward implication:** High error rate near the boundary (-5% to 0%) indicates the model struggles")
    ln("when price signals are ambiguous. Consider a graded decision reward that penalises boundary errors more heavily.")

    # ── 6. Anchoring strength analysis ───────────────────────────────────────
    h("6. Errors by Anchoring Strength")
    anchor_groups = defaultdict(list)
    for e in examples:
        anchor_groups[e.get("anchoring_strength") or "none/unknown"].append(e)
    tbl(
        ["Anchoring strength", "Total", "Errors", "Error %"],
        sorted([
            (a, len(exs), sum(1 for x in exs if not x["decision_correct"]),
             f"{100*sum(1 for x in exs if not x['decision_correct'])/len(exs):.1f}%")
            for a, exs in anchor_groups.items()
        ], key=lambda r: -int(r[2]))
    )

    # ── 7. Response quality analysis ─────────────────────────────────────────
    h("7. Response Quality Analysis (all predictions)")

    h2("Response length distribution")
    pred_lengths = [response_word_count(e.get("pred_response")) for e in examples]
    gt_lengths   = [response_word_count(e.get("gt_response"))   for e in examples]
    err_lengths  = [response_word_count(e.get("pred_response")) for e in errors]

    ln(f"**Predicted responses (all):** mean={mean(pred_lengths):.1f}w, median={median(pred_lengths):.0f}w, min={min(pred_lengths)}, max={max(pred_lengths)}")
    ln(f"**Ground truth responses:**    mean={mean(gt_lengths):.1f}w, median={median(gt_lengths):.0f}w")
    if err_lengths:
        ln(f"**Error examples only:**       mean={mean(err_lengths):.1f}w, median={median(err_lengths):.0f}w")
    ln("")

    very_short = [e for e in examples if response_word_count(e.get("pred_response")) < 5]
    ln(f"Responses under 5 words: {len(very_short)} ({100*len(very_short)/n_total:.1f}%)")
    if very_short:
        for e in very_short[:5]:
            ln(f"  - `{e['dialogue_id']} T{e['turn_index']}`: \"{e.get('pred_response','')[:80]}\"")
    ln("")
    ln("**GRPO reward implication:** Penalise responses under ~8 words — they're not real negotiation turns.")

    h2("Repetition across dialogue turns")
    # Group by dialogue
    dialogue_preds = defaultdict(list)
    for e in examples:
        if e.get("pred_response"):
            dialogue_preds[e["dialogue_id"]].append(e["pred_response"])

    rep_scores = {did: repetition_score_for_dialogue(resps)
                  for did, resps in dialogue_preds.items() if len(resps) > 1}

    if rep_scores:
        avg_rep = mean(rep_scores.values())
        high_rep = {did: s for did, s in rep_scores.items() if s > 0.4}
        ln(f"**Average 3-gram overlap between consecutive seller turns:** {avg_rep:.3f}")
        ln(f"*(0 = completely different, 1 = identical)*")
        ln(f"**Dialogues with high repetition (>0.4):** {len(high_rep)}/{len(rep_scores)}")
        ln("")
        if high_rep:
            ln("Most repetitive dialogues:")
            for did, score in sorted(high_rep.items(), key=lambda x: -x[1])[:5]:
                ln(f"  - `{did}`: overlap score {score:.3f}")
        ln("")
        ln("**GRPO reward implication:** This directly maps to the `lexical_diversity` reward in our reward design.")
        ln("Penalise high n-gram overlap between the current response and the model's prior responses in the same dialogue.")

    # ── 8. Format failure analysis ────────────────────────────────────────────
    h("8. Format Failures")
    format_fails = [e for e in examples if not e.get("format_ok")]
    missing_dec  = [e for e in examples if not e.get("pred_decision")]
    missing_resp = [e for e in examples if not e.get("pred_response")]

    ln(f"**Format failures (missing any required tag):** {len(format_fails)}/{n_total} ({100*len(format_fails)/n_total:.1f}%)")
    ln(f"**Missing `<decision>` tag:** {len(missing_dec)}")
    ln(f"**Missing `<response>` tag:** {len(missing_resp)}")
    ln("")
    ln("**GRPO reward implication:** Binary format reward — 0.0 if any required tag missing, 1.0 if all present.")
    ln("This is a hard constraint, not a soft signal.")

    # ── 9. Qualitative error examples ────────────────────────────────────────
    h("9. Representative Error Examples")
    ln("*Manually review these to identify patterns not captured by metrics above.*\n")

    # Show up to 3 per error direction
    shown = defaultdict(int)
    for e in errors:
        direction = f"{e['gt_decision']}→{e.get('pred_decision') or 'MISSING'}"
        if shown[direction] >= 3:
            continue
        shown[direction] += 1
        em_labels, em_intensity, em_valence = emotion_summary(e.get("prior_buyer_emotion"))
        gap = price_gap_pct(e.get("buyer_offers"), e.get("fair_value"))

        ln(f"---")
        ln(f"**{e['dialogue_id']} T{e['turn_index']}** | Zone: `{e.get('zone','?')}` | Flip: `{e.get('decision_flip', False)}` | Price gap: `{gap}%`")
        ln(f"Buyer emotion: `{em_labels}` | intensity: `{em_intensity}` | valence: `{em_valence}`")
        ln(f"Buyer offers: `{e.get('buyer_offers')}` | Fair value: `₹{e.get('fair_value')}`")
        ln(f"**GT:** `{e['gt_decision']}` → **Pred:** `{e.get('pred_decision') or 'MISSING'}`")
        ln(f"> GT response: *\"{e.get('gt_response','')[:120]}\"*")
        ln(f"> Pred response: *\"{e.get('pred_response','')[:120]}\"*")
        ln("")

    # ── 10. Summary table for reward function design ──────────────────────────
    h("10. Reward Function Design Summary")
    ln("*Derived directly from error patterns above.*\n")

    tbl(
        ["Reward component", "Error pattern it addresses", "Suggested weight"],
        [
            ("`decision_accuracy`",         "Primary — wrong LEVERAGE/MITIGATE/UNDECIDED",        "1.0"),
            ("`format_reward`",             "Missing `<decision>` or `<response>` tags",           "0.5 (hard gate)"),
            ("`lexical_diversity`",         "Repetitive responses across dialogue turns",           "0.1"),
            ("`language_purity`",           "Non-English tokens in response",                      "0.1"),
            ("`reasoning_mentions_price`",  "Reasoning ignores price signals",                     "0.2"),
            ("`reasoning_mentions_emotion`","Reasoning ignores buyer emotion signals",              "0.2"),
            ("`response_length`",           "Responses too short to be real negotiation turns",    "0.1"),
            ("`flip_turn_bonus`",           "Correct handling of decision-flip turns",             "+0.3 bonus"),
        ]
    )

    ln("> **Note:** Weights are initial suggestions. Calibrate after first GRPO run by checking which")
    ln("> reward components have the most variance across generations — those are the ones doing useful work.")

    return "\n".join(lines)


def analyse_speech(speech_examples, text_examples, version):
    """Section 11: speech-specific stats appended to the main report."""
    lines = []
    def h(t):  lines.append(f"\n## {t}\n")
    def h2(t): lines.append(f"\n### {t}\n")
    def ln(t=""): lines.append(t)

    h("11. Speech Output Analysis")
    n_total   = len(speech_examples)
    audio_ok  = sum(1 for e in speech_examples if e.get("audio_generated"))
    audio_pct = round(100 * audio_ok / n_total, 1) if n_total else 0

    ln(f"**Total examples:** {n_total}")
    ln(f"**Audio generated successfully:** {audio_ok} ({audio_pct}%)")
    ln(f"**Audio failures (no WAV saved):** {n_total - audio_ok}")
    ln("")

    # WAV files location
    wav_files = [e.get("wav_file") for e in speech_examples if e.get("wav_file")]
    if wav_files:
        speech_dir = str(Path(wav_files[0]).parent)
        ln(f"**WAV files location:** `{speech_dir}`")
        ln("")

    # Audio failures breakdown — which examples failed to generate audio
    failed_audio = [e for e in speech_examples if not e.get("audio_generated")]
    if failed_audio:
        h2("Examples where audio generation failed")
        for e in failed_audio[:10]:
            ln(f"- `{e['dialogue_id']} T{e['turn_index']}` | zone: `{e.get('zone','?')}` | pred: `{e.get('pred_decision','?')}`")
        ln("")

    # Cross-check: confirm text decision accuracy matches speech
    speech_acc = sum(1 for e in speech_examples if e.get("decision_correct"))
    text_acc   = sum(1 for e in text_examples if e.get("decision_correct")) if text_examples else None
    h2("Decision accuracy cross-check (text vs speech)")
    ln(f"Text inference accuracy:   **{round(100*text_acc/len(text_examples),1)}%** ({text_acc}/{len(text_examples)})" if text_acc is not None else "Text inference: not available")
    ln(f"Speech inference accuracy: **{round(100*speech_acc/n_total,1)}%** ({speech_acc}/{n_total})")
    ln("")
    ln("*These should be identical — both use the same Pass 1 text generation. Any difference indicates a data mismatch.*")

    return "\n".join(lines)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.inference_file:
        text_path   = Path(args.inference_file)
        speech_path = None
        if not text_path.exists():
            sys.exit(f"File not found: {text_path}")
    else:
        text_path, speech_path = find_inference_files(args.version)

    print(f"  Text inference  : {text_path}")
    print(f"  Speech inference: {speech_path or 'not found — text-only report'}")

    text_data     = json.loads(text_path.read_text())
    text_examples = text_data["per_example"]
    version       = text_data.get("sft_version", args.version)
    print(f"  Text examples   : {len(text_examples)}  |  version: {version}")

    speech_examples = None
    if speech_path:
        speech_data     = json.loads(speech_path.read_text())
        speech_examples = speech_data["per_example"]
        print(f"  Speech examples : {len(speech_examples)}")

    report = analyse(text_examples, version)
    if speech_examples:
        report += "\n" + analyse_speech(speech_examples, text_examples, version)

    label    = "text_and_speech" if speech_examples else "text_only"
    out_path = text_path.parent / f"{version}_error_analysis_{label}_{ts}.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"  Report saved → {out_path}")


if __name__ == "__main__":
    main()
