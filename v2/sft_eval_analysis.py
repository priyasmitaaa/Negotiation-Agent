#!/usr/bin/env python3
"""
sft_eval_analysis.py — Run inference on SFT test set and identify failure modes.

Generates real model outputs, parses them, and reports:
  - Decision accuracy (LEVERAGE / MITIGATE / UNDECIDED)
  - Format compliance (all 4 tags present and in order)
  - VAII range compliance (0.05 – 0.45)
  - Reasoning quality (required elements present)
  - Response coherence (non-empty, non-degenerate)

These failures directly inform GRPO reward function design.

Usage:
  # Use latest run's adapter (auto-detected):
  CUDA_VISIBLE_DEVICES=0 python3 Qwen3-tts/v2/sft_eval_analysis.py

  # Use a specific adapter:
  CUDA_VISIBLE_DEVICES=0 python3 Qwen3-tts/v2/sft_eval_analysis.py \
      --adapter Qwen3-tts/v2/sft_output_v2/run1_20260429/final_adapter

  # Faster: only sample N examples from test set:
  CUDA_VISIBLE_DEVICES=0 python3 Qwen3-tts/v2/sft_eval_analysis.py --n_samples 50
"""

import argparse
import gc
import json
import re
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import torch
import numpy as np

warnings.filterwarnings("ignore")

V2_ROOT    = Path(__file__).parent
INTERM_DIR = V2_ROOT / "intermediate"
AUDIO_DIR  = V2_ROOT / "tts_outputs"
SFT_OUT    = V2_ROOT / "sft_output_v2"
MODEL_ID   = "Qwen/Qwen2.5-Omni-7B"

SYSTEM_PROMPT_V2 = (
    "You are an AI negotiation assistant for a second-hand electronics shop in India. "
    "Guide the negotiation to within 10% of fair market value. Analyse the "
    "conversation and buyer VAII signals (< 0.50 = calm, >= 0.50 = stressed). "
    "VAII (Vocal Affective Intensity Index) reflects the buyer's emotional stress level. "
    "Produce in this exact order: <reasoning>, <decision> [UNDECIDED/LEVERAGE/MITIGATE], "
    "<seller_vaii> [0.05-0.45], <response>. Infer everything from raw signals."
)

SYSTEM_PROMPT_V3 = (
    "You are an AI negotiation assistant for a second-hand electronics shop in India. "
    "Guide the negotiation to within 10% of fair market value. Analyse the "
    "conversation and buyer VAII signals (< 0.50 = calm, >= 0.50 = stressed). "
    "VAII (Vocal Affective Intensity Index) reflects the buyer's emotional stress level. "
    "Produce in this exact order: <decision> [UNDECIDED/LEVERAGE/MITIGATE], "
    "<seller_vaii> [0.05-0.45], <response>. Infer everything from raw signals."
)

SYSTEM_PROMPT = SYSTEM_PROMPT_V2  # default; overridden by --v3 flag

TARGET_SR = 16_000


# ── Argument parsing ──────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter",   default=None,
                   help="Path to LoRA adapter directory. Auto-detects latest run if omitted.")
    p.add_argument("--n_samples", type=int, default=100,
                   help="Number of test examples to evaluate (default 100, use 0 for all).")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--v3", action="store_true",
                   help="Use v3 system prompt (no reasoning tag). Use this for sft_output_v3 adapters.")
    return p.parse_args()


# ── Find adapter ──────────────────────────────────────────────────────────────
def find_adapter(adapter_arg):
    if adapter_arg:
        p = Path(adapter_arg)
        if not p.exists():
            sys.exit(f"Adapter not found: {p}")
        return p

    # Auto-detect: find latest completed run with final_adapter
    run_dirs = sorted(
        [d for d in SFT_OUT.iterdir() if d.is_dir()],
        key=lambda d: d.name, reverse=True
    )
    for rd in run_dirs:
        adapter = rd / "final_adapter"
        if adapter.exists():
            print(f"  Auto-detected adapter: {adapter}")
            return adapter

    sys.exit("No final_adapter found in any run. Run SFT first.")


# ── Find test IDs ─────────────────────────────────────────────────────────────
def find_test_ids():
    for rd in sorted(SFT_OUT.iterdir(), key=lambda d: d.name, reverse=True):
        f = rd / "test_ids.json"
        if f.exists():
            return json.loads(f.read_text())
    # Fall back to splits_v2.json (used by v3 runs)
    splits_file = V2_ROOT / "splits_v2.json"
    if splits_file.exists():
        splits = json.loads(splits_file.read_text())
        if "test" in splits:
            print(f"  Using test IDs from splits_v2.json ({len(splits['test'])} samples)")
            return splits["test"]
    sys.exit("No test_ids.json or splits_v2.json found.")


# ── Build test examples (same logic as sft_data_v2.py) ───────────────────────
def build_test_examples(test_ids, n_samples):
    examples = []
    for did in test_ids:
        fpath = INTERM_DIR / f"{did}.json"
        if not fpath.exists():
            continue
        data    = json.loads(fpath.read_text())
        seed    = data["seed"]
        turns   = data["processed"]
        pricing = seed["pricing"]
        domain  = seed["domain"]
        correct_decision = seed["generation_params"]["correct_decision"]

        turns_so_far = []
        for turn in turns:
            if turn["speaker"] != "seller":
                turns_so_far.append(turn)
                continue

            tidx      = turn["turn_index"]
            reasoning = (turn.get("reasoning") or "").strip()
            if not reasoning:
                turns_so_far.append(turn)
                continue

            # Check audio
            audio_paths = []
            missing = False
            for pt in turns_so_far:
                wav = AUDIO_DIR / did / f"turn_{pt['turn_index']:02d}_{pt['speaker']}.wav"
                if not wav.exists():
                    missing = True
                    break
                audio_paths.append(wav)
            if missing:
                turns_so_far.append(turn)
                continue

            context_block = (
                f"Product: {domain['product']}. "
                f"Condition: {domain['condition']}. "
                f"Asking: ${pricing['asking_price']}. "
                f"Fair value: ${pricing['fair_value']}."
            )
            conv_lines = []
            for pt in turns_so_far:
                line = f"Turn {pt['turn_index']} [{pt['speaker'].upper()}]: {pt['text']}"
                if pt["speaker"] == "buyer":
                    v = pt.get("vaii", {})
                    if v.get("raw") is not None:
                        line += f"  [VAII: {v['raw']:.2f} / {v['state']}]"
                conv_lines.append(line)

            user_text = (
                f"CONTEXT\n{context_block}\n\n"
                f"CONVERSATION SO FAR\n" + "\n".join(conv_lines) + "\n\n"
                f"TASK\nYou are the seller. It is now Turn {tidx}. "
                f"Produce <reasoning>, <decision>, <seller_vaii>, <response>."
            )

            sv  = turn["seller_vaii"]["raw"]
            dec = turn["factor_state"]["decision"]

            examples.append({
                "dialogue_id":    did,
                "turn_index":     tidx,
                "audio_paths":    audio_paths,
                "user_text":      user_text,
                "gt_decision":    dec,
                "gt_seller_vaii": sv,
                "gt_reasoning":   reasoning,
                "gt_response":    turn["text"],
                "fair_value":     pricing["fair_value"],
                "buyer_offers":   turn["factor_state"]["buyer_offers_so_far"],
                "anchoring_str":  turn["factor_state"].get("anchoring_strength"),
                "decision_flip":  turn["factor_state"].get("decision_is_flip", False),
                "zone":           turn["factor_state"].get("zone", ""),
                "correct_decision": correct_decision,
            })
            turns_so_far.append(turn)

        if n_samples and len(examples) >= n_samples:
            break

    return examples[:n_samples] if n_samples else examples


# ── Output parser ─────────────────────────────────────────────────────────────
def parse_output(text):
    result = {
        "reasoning":   None,
        "decision":    None,
        "seller_vaii": None,
        "response":    None,
        "raw":         text,
    }
    m = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
    if m:
        result["reasoning"] = m.group(1).strip()

    m = re.search(r"<decision>\s*(LEVERAGE|MITIGATE|UNDECIDED)\s*</decision>", text, re.DOTALL)
    if m:
        result["decision"] = m.group(1).strip()

    m = re.search(r"<seller_vaii>\s*([\d.]+)\s*</seller_vaii>", text)
    if m:
        try:
            result["seller_vaii"] = float(m.group(1))
        except ValueError:
            pass

    m = re.search(r"<response>(.*?)</response>", text, re.DOTALL)
    if m:
        result["response"] = m.group(1).strip()

    return result


# ── Reasoning quality checks ──────────────────────────────────────────────────
REQUIRED_REASONING_ELEMENTS = [
    ("mentions_buyer_offers",   lambda r, ex: any(f"${o}" in r or str(o) in r
                                                   for o in ex["buyer_offers"])),
    ("mentions_fair_value",     lambda r, ex: str(ex["fair_value"]) in r or
                                               f"${ex['fair_value']}" in r),
    ("mentions_vaii",           lambda r, ex: "vaii" in r.lower() or
                                               "stressed" in r.lower() or "calm" in r.lower()),
    ("mentions_harm_direction", lambda r, ex: "below" in r.lower() or "above" in r.lower() or
                                               "harm" in r.lower()),
    ("mentions_decision_word",  lambda r, ex: any(w in r.lower()
                                                   for w in ["leverage", "mitigate", "undecided"])),
    ("has_arithmetic",          lambda r, ex: "<" in r or ">" in r or "→" in r or "->" in r),
]


def check_reasoning(reasoning, example):
    if not reasoning:
        return {k: False for k, _ in REQUIRED_REASONING_ELEMENTS}
    return {
        k: fn(reasoning, example)
        for k, fn in REQUIRED_REASONING_ELEMENTS
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    global SYSTEM_PROMPT
    if args.v3:
        SYSTEM_PROMPT = SYSTEM_PROMPT_V3

    adapter_path = find_adapter(args.adapter)
    test_ids     = find_test_ids()

    print(f"\n{'='*65}")
    print(f"  SFT Failure Mode Analysis")
    print(f"{'='*65}")
    print(f"  Adapter    : {adapter_path}")
    print(f"  Test IDs   : {len(test_ids)} dialogues")
    print(f"  Samples    : {args.n_samples or 'all'}")

    print("\n  Building test examples…")
    examples = build_test_examples(test_ids, args.n_samples)
    print(f"  Built {len(examples)} test examples.")

    if not examples:
        sys.exit("No test examples built. Check audio and intermediate files.")

    # ── Load model ─────────────────────────────────────────────────────────
    print("\n  Loading model + adapter…")
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    from peft import PeftModel
    import soundfile as sf
    import librosa

    hf_token = None
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID, token=hf_token)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    _full = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        enable_audio_output=False,
        token=hf_token,
    )
    base_model = _full.thinker
    base_model.config.use_cache = True
    del _full; gc.collect()

    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print("  Model loaded.")

    def load_audio(path):
        try:
            import soundfile as sf
            audio, sr = sf.read(str(path), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != TARGET_SR:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
            return audio.astype(np.float32)
        except Exception:
            return None

    # ── Run inference ──────────────────────────────────────────────────────
    print(f"\n  Running inference on {len(examples)} examples…\n")
    results = []

    for i, ex in enumerate(examples):
        messages = [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": ex["user_text"]},
        ]

        def wrap(msg):
            c = msg["content"]
            if isinstance(c, str):
                c = [{"type": "text", "text": c}]
            return {"role": msg["role"], "content": c}

        messages_norm = [wrap(m) for m in messages]
        text = processor.apply_chat_template(
            messages_norm, tokenize=False, add_generation_prompt=True
        )

        audio_arrays = [a for p in ex["audio_paths"] if (a := load_audio(p)) is not None]

        try:
            inputs = processor(
                text=text,
                audio=audio_arrays if audio_arrays else None,
                sampling_rate=TARGET_SR,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=4096,
            ).to(device)
        except Exception:
            inputs = processor(
                text=text, return_tensors="pt",
                padding=False, truncation=True, max_length=4096,
            ).to(device)

        with torch.no_grad():
            gen_ids = model.generate(
                **{k: v for k, v in inputs.items() if k != "labels"},
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=processor.tokenizer.pad_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        output_ids = gen_ids[0][input_len:]
        output_text = processor.tokenizer.decode(output_ids, skip_special_tokens=True)

        parsed = parse_output(output_text)
        reasoning_checks = check_reasoning(parsed["reasoning"], ex)

        results.append({
            "example":          ex,
            "output":           output_text,
            "parsed":           parsed,
            "reasoning_checks": reasoning_checks,
        })

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(examples)}] processed…")

    # ── Analyse failures ───────────────────────────────────────────────────
    print(f"\n\n{'='*65}")
    print(f"  FAILURE MODE ANALYSIS  ({len(results)} examples)")
    print(f"{'='*65}")

    n = len(results)

    # 1. FORMAT failures
    no_reasoning_tag  = sum(1 for r in results if r["parsed"]["reasoning"]    is None)
    no_decision_tag   = sum(1 for r in results if r["parsed"]["decision"]     is None)
    no_vaii_tag       = sum(1 for r in results if r["parsed"]["seller_vaii"]  is None)
    no_response_tag   = sum(1 for r in results if r["parsed"]["response"]     is None)
    all_4_tags        = sum(1 for r in results if all(
        r["parsed"][k] is not None for k in ("reasoning","decision","seller_vaii","response")
    ))

    print(f"\n  FORMAT COMPLIANCE")
    print(f"  {'─'*45}")
    print(f"  All 4 tags present      : {all_4_tags}/{n}  ({100*all_4_tags/n:.1f}%)")
    print(f"  Missing <reasoning>     : {no_reasoning_tag}/{n}  ({100*no_reasoning_tag/n:.1f}%)")
    print(f"  Missing <decision>      : {no_decision_tag}/{n}  ({100*no_decision_tag/n:.1f}%)")
    print(f"  Missing <seller_vaii>   : {no_vaii_tag}/{n}  ({100*no_vaii_tag/n:.1f}%)")
    print(f"  Missing <response>      : {no_response_tag}/{n}  ({100*no_response_tag/n:.1f}%)")

    # 2. DECISION accuracy
    correct_decision = sum(
        1 for r in results
        if r["parsed"]["decision"] == r["example"]["gt_decision"]
    )
    decision_present = sum(1 for r in results if r["parsed"]["decision"] is not None)
    predicted_decisions = Counter(r["parsed"]["decision"] for r in results if r["parsed"]["decision"])
    gt_decisions        = Counter(r["example"]["gt_decision"] for r in results)

    print(f"\n  DECISION ACCURACY")
    print(f"  {'─'*45}")
    print(f"  Correct decision        : {correct_decision}/{n}  ({100*correct_decision/n:.1f}%)")
    print(f"  Decision parseable      : {decision_present}/{n}  ({100*decision_present/n:.1f}%)")
    print(f"  GT distribution         : {dict(gt_decisions)}")
    print(f"  Predicted distribution  : {dict(predicted_decisions)}")

    # Confusion breakdown
    confusion = defaultdict(int)
    for r in results:
        gt  = r["example"]["gt_decision"]
        pred = r["parsed"]["decision"] or "MISSING"
        confusion[(gt, pred)] += 1
    print(f"  Confusion (gt→pred):")
    for (gt, pred), count in sorted(confusion.items()):
        marker = "✓" if gt == pred else "✗"
        print(f"    {marker} {gt} → {pred}: {count}")

    # 3. VAII range compliance
    vaii_vals = [(r["parsed"]["seller_vaii"], r["example"]["gt_seller_vaii"])
                 for r in results if r["parsed"]["seller_vaii"] is not None]
    vaii_in_range = sum(1 for v, _ in vaii_vals if 0.05 <= v <= 0.45)
    vaii_too_high = sum(1 for v, _ in vaii_vals if v > 0.45)
    vaii_too_low  = sum(1 for v, _ in vaii_vals if v < 0.05)
    vaii_missing  = n - len(vaii_vals)

    pred_vaii_vals = [v for v, _ in vaii_vals]
    gt_vaii_vals   = [g for _, g in vaii_vals]

    print(f"\n  SELLER VAII COMPLIANCE  (valid range: 0.05 – 0.45)")
    print(f"  {'─'*45}")
    print(f"  In range [0.05–0.45]    : {vaii_in_range}/{n}  ({100*vaii_in_range/n:.1f}%)")
    print(f"  Too high (> 0.45)       : {vaii_too_high}/{n}  ({100*vaii_too_high/n:.1f}%)")
    print(f"  Too low  (< 0.05)       : {vaii_too_low}/{n}  ({100*vaii_too_low/n:.1f}%)")
    print(f"  VAII missing            : {vaii_missing}/{n}  ({100*vaii_missing/n:.1f}%)")
    if pred_vaii_vals:
        mae = np.mean([abs(p - g) for p, g in zip(pred_vaii_vals, gt_vaii_vals)])
        print(f"  Mean abs error vs GT    : {mae:.3f}")
        print(f"  Predicted VAII mean     : {np.mean(pred_vaii_vals):.3f}  (GT mean: {np.mean(gt_vaii_vals):.3f})")

    # 4. REASONING quality
    print(f"\n  REASONING QUALITY")
    print(f"  {'─'*45}")
    for element, _ in REQUIRED_REASONING_ELEMENTS:
        count = sum(1 for r in results if r["reasoning_checks"].get(element, False))
        pct   = 100 * count / n
        bar   = "▓" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"  {element:<30} [{bar}] {count}/{n} ({pct:.0f}%)")

    # 5. RESPONSE quality
    empty_response = sum(1 for r in results
                         if r["parsed"]["response"] is not None and len(r["parsed"]["response"].strip()) < 10)
    degenerate     = sum(1 for r in results
                         if r["output"] and (
                             r["output"].count("...") > 5 or
                             len(set(r["output"].split())) < 10
                         ))
    avg_response_len = np.mean([
        len(r["parsed"]["response"].split())
        for r in results if r["parsed"]["response"]
    ]) if any(r["parsed"]["response"] for r in results) else 0

    print(f"\n  RESPONSE QUALITY")
    print(f"  {'─'*45}")
    print(f"  Avg response word count : {avg_response_len:.1f}")
    print(f"  Near-empty responses    : {empty_response}/{n}")
    print(f"  Degenerate outputs      : {degenerate}/{n}")

    # 6. Failure by difficulty
    print(f"\n  FAILURE RATE BY DIFFICULTY")
    print(f"  {'─'*45}")
    for zone_name in ["pre_crystallisation", "post_crystallisation"]:
        zone_results = [r for r in results if r["example"]["zone"] == zone_name]
        if zone_results:
            z_correct = sum(1 for r in zone_results
                            if r["parsed"]["decision"] == r["example"]["gt_decision"])
            print(f"  {zone_name:<30} decision acc: {z_correct}/{len(zone_results)} ({100*z_correct/len(zone_results):.0f}%)")

    flip_results = [r for r in results if r["example"]["decision_flip"]]
    if flip_results:
        f_correct = sum(1 for r in flip_results
                        if r["parsed"]["decision"] == r["example"]["gt_decision"])
        print(f"  decision_is_flip=True          decision acc: {f_correct}/{len(flip_results)} ({100*f_correct/len(flip_results):.0f}%)")

    # 7. REWARD FUNCTION RECOMMENDATIONS
    print(f"\n{'='*65}")
    print(f"  RECOMMENDED REWARD FUNCTIONS FOR GRPO")
    print(f"{'='*65}")

    format_fail_rate   = 1 - all_4_tags / n
    decision_fail_rate = 1 - correct_decision / n
    vaii_fail_rate     = 1 - vaii_in_range / n

    recs = []

    if format_fail_rate > 0.05:
        recs.append((
            "format_reward  [CRITICAL]",
            f"  {100*format_fail_rate:.0f}% of outputs missing at least one tag.\n"
            f"  Score: +1.0 if all 4 tags present in correct order, 0.0 otherwise.\n"
            f"  GRPO will prioritise learning the output structure first."
        ))
    else:
        recs.append((
            "format_reward  [LOW PRIORITY]",
            f"  Only {100*format_fail_rate:.0f}% format failures — model already learned structure.\n"
            f"  Still include it to prevent regression during RL."
        ))

    if decision_fail_rate > 0.10:
        recs.append((
            "decision_reward  [CRITICAL]",
            f"  {100*decision_fail_rate:.0f}% wrong decisions.\n"
            f"  Score: +1.0 correct decision, 0.0 wrong. For UNDECIDED at T2 only: +0.5 (ambiguous by design).\n"
            f"  Consider weighting LEVERAGE heavier if confusion shows LEVERAGE→MITIGATE bias."
        ))
    else:
        recs.append((
            "decision_reward  [MODERATE]",
            f"  {100*decision_fail_rate:.0f}% wrong decisions — model mostly correct.\n"
            f"  Still include as primary reward signal."
        ))

    if vaii_fail_rate > 0.10:
        recs.append((
            "vaii_range_reward  [HIGH]",
            f"  {100*vaii_fail_rate:.0f}% of VAII outputs out of range or missing.\n"
            f"  Score: +1.0 if 0.05 ≤ seller_vaii ≤ 0.45, 0.0 otherwise.\n"
            f"  Or use continuous: score = 1.0 − min(1.0, |pred − gt|/0.40)"
        ))

    reasoning_fails = {k: sum(1 for r in results if not r["reasoning_checks"].get(k, False))
                       for k, _ in REQUIRED_REASONING_ELEMENTS}
    worst_reasoning = max(reasoning_fails, key=reasoning_fails.get)
    worst_count     = reasoning_fails[worst_reasoning]
    if worst_count / n > 0.20:
        recs.append((
            "reasoning_quality_reward  [MODERATE]",
            f"  {100*worst_count/n:.0f}% of reasonings missing '{worst_reasoning}'.\n"
            f"  Score: +0.2 for each of these 5 elements present:\n"
            f"    buyer offers mentioned, FV mentioned, VAII state mentioned,\n"
            f"    harm direction mentioned, decision word present.\n"
            f"  Max +1.0. Encourages substantive reasoning, not just hallucination."
        ))

    price_reward_needed = any(
        r["example"]["zone"] == "post_crystallisation" and
        r["parsed"]["decision"] != r["example"]["gt_decision"]
        for r in results
    )
    if price_reward_needed:
        recs.append((
            "price_direction_reward  [FUTURE / PHASE 2]",
            f"  Model makes wrong decisions post-crystallisation.\n"
            f"  Score: did the seller response push buyer toward FV?\n"
            f"    LEVERAGE: buyer next offer > current offer → +reward\n"
            f"    MITIGATE: buyer next offer < current offer → +reward\n"
            f"  Requires running the full dialogue — best implemented in GRPO Phase 2."
        ))

    for title, desc in recs:
        print(f"\n  ▶ {title}")
        print(desc)

    # ── Save results ───────────────────────────────────────────────────────
    out_path = adapter_path.parent / "failure_analysis.json"
    save_data = []
    for r in results:
        save_data.append({
            "dialogue_id":       r["example"]["dialogue_id"],
            "turn_index":        r["example"]["turn_index"],
            "gt_decision":       r["example"]["gt_decision"],
            "pred_decision":     r["parsed"]["decision"],
            "decision_correct":  r["parsed"]["decision"] == r["example"]["gt_decision"],
            "gt_seller_vaii":    r["example"]["gt_seller_vaii"],
            "pred_seller_vaii":  r["parsed"]["seller_vaii"],
            "vaii_in_range":     (0.05 <= r["parsed"]["seller_vaii"] <= 0.45)
                                  if r["parsed"]["seller_vaii"] is not None else False,
            "all_tags_present":  all(r["parsed"][k] is not None
                                     for k in ("reasoning","decision","seller_vaii","response")),
            "reasoning_checks":  r["reasoning_checks"],
            "zone":              r["example"]["zone"],
            "decision_flip":     r["example"]["decision_flip"],
            "anchoring_strength":r["example"]["anchoring_str"],
            "output":            r["output"],
        })
    out_path.write_text(json.dumps(save_data, indent=2, ensure_ascii=False))
    print(f"\n{'='*65}")
    print(f"  Full results saved → {out_path}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
