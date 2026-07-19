#!/usr/bin/env python3
"""
judge.py — LLM-as-a-judge evaluation for SFT v4/v5 inference outputs.

Uses Qwen3-8B locally to score each predicted response on 4 dimensions:
  1. emotion_handling    — did the seller acknowledge/respond to buyer's emotion?
  2. negotiation_quality — is the response tactically sound for the decision made?
  3. response_naturalness — does it sound like a real human seller?
  4. progression         — does it move the negotiation forward vs repeating?

Each dimension scored 1-5. Overall score = mean of 4.

Input:  inference_{timestamp}.json  (from inference_v4.py or inference_v5.py)
Output: judge_{timestamp}.json      (saved alongside input file)

Usage:
  # Auto-detect latest v4 inference result:
  CUDA_VISIBLE_DEVICES=1 python3 Qwen3-tts/v2/judge.py --version v4

  # Specific inference file:
  CUDA_VISIBLE_DEVICES=1 python3 Qwen3-tts/v2/judge.py \\
      --inference_file Qwen3-tts/v2/sft_output_v4/20260525_014244/inference_20260526_193657.json

  # Faster run on 50 examples:
  CUDA_VISIBLE_DEVICES=1 python3 Qwen3-tts/v2/judge.py --version v4 --n_samples 50
"""

import argparse
import datetime
import json
import sys
import warnings
from pathlib import Path
from statistics import mean

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")

V2_ROOT    = Path(__file__).parent
SFT_OUT_V4 = V2_ROOT / "sft_output_v4"
SFT_OUT_V5 = V2_ROOT / "sft_output_v5"
SFT_OUT_V6 = V2_ROOT / "sft_output_v6"

JUDGE_MODEL_ID = "Qwen/Qwen3-8B"

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator of AI negotiation assistants for second-hand electronics shops.

You will be shown a negotiation turn with context, and the AI seller's predicted response.
Score the response on 4 dimensions, each from 1 to 5.

Scoring guide:
1 = Very poor  2 = Poor  3 = Acceptable  4 = Good  5 = Excellent

Dimensions:
- emotion_handling: Did the seller acknowledge or appropriately respond to the buyer's emotional state?
- negotiation_quality: Is the response tactically correct given the decision (LEVERAGE/MITIGATE/UNDECIDED)?
- response_naturalness: Does it sound like a real, natural human seller — not robotic or repetitive?
- progression: Does it move the negotiation forward, or just repeat the same thing as before?

Reply ONLY with valid JSON in this exact format, no other text:
{
  "emotion_handling": <1-5>,
  "negotiation_quality": <1-5>,
  "response_naturalness": <1-5>,
  "progression": <1-5>,
  "reasoning": "<one sentence explaining the scores>"
}"""


JUDGE_USER_TEMPLATE = """NEGOTIATION CONTEXT
Product: {product}
Condition: {condition}
Asking price: ₹{asking_price}
Fair value: ₹{fair_value}
Turn: {turn_index}
Zone: {zone}

CONVERSATION SO FAR
{conversation}

BUYER EMOTION (last buyer turn)
{buyer_emotion}

AI SELLER DECISION: {pred_decision}
AI SELLER RESPONSE: {pred_response}

GROUND TRUTH DECISION: {gt_decision}
GROUND TRUTH RESPONSE: {gt_response}

Score the AI SELLER RESPONSE on the 4 dimensions."""


# ── Args ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--version",        default="v4", choices=["v4", "v5", "v6"],
                   help="Which SFT version to judge. Auto-detects latest inference file.")
    p.add_argument("--inference_file", default=None,
                   help="Explicit path to inference JSON. Overrides --version.")
    p.add_argument("--n_samples",      type=int, default=0,
                   help="Number of examples to judge (0 = all).")
    p.add_argument("--judge_model",    default=JUDGE_MODEL_ID,
                   help=f"HF model ID for judge. Default: {JUDGE_MODEL_ID}")
    p.add_argument("--max_new_tokens", type=int, default=256,
                   help="Max tokens for judge response.")
    return p.parse_args()


# ── Find inference file ────────────────────────────────────────────────────────

def find_inference_files(version: str):
    """Returns (text_path, speech_path_or_None) — latest run for the version."""
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


# ── Build judge prompt ─────────────────────────────────────────────────────────

def build_judge_prompt(ex: dict, inference_data: dict) -> str:
    """Build the user prompt for the judge from a per-example result."""
    # Reconstruct minimal conversation from raw_output context
    # We use fields stored in the per-example result
    buyer_emotion = ex.get("prior_buyer_emotion") or {}
    if buyer_emotion.get("labels"):
        emotion_str = (
            f"Labels: {', '.join(buyer_emotion['labels'])} | "
            f"Intensity: {buyer_emotion.get('intensity','?')} | "
            f"Valence: {buyer_emotion.get('valence','?')}"
        )
    else:
        emotion_str = "Not available"

    # Reconstruct product info from inference_params system prompt isn't ideal,
    # so we use what's stored per-example
    return JUDGE_USER_TEMPLATE.format(
        product       = ex.get("product", "unknown"),
        condition     = ex.get("condition", "unknown"),
        asking_price  = ex.get("asking_price", "?"),
        fair_value    = ex.get("fair_value", "?"),
        turn_index    = ex.get("turn_index", "?"),
        zone          = ex.get("zone", "?"),
        conversation  = ex.get("conversation_so_far", "(not stored)"),
        buyer_emotion = emotion_str,
        pred_decision = ex.get("pred_decision") or "MISSING",
        pred_response = ex.get("pred_response") or "(no response generated)",
        gt_decision   = ex.get("gt_decision", "?"),
        gt_response   = ex.get("gt_response", "?"),
    )


# ── Parse judge output ─────────────────────────────────────────────────────────

def parse_judge_output(text: str) -> dict:
    """Extract JSON scores from judge output. Returns None on parse failure."""
    import re
    # strip <think>...</think> if Qwen3 thinking mode is on
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # find JSON block
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        required = {"emotion_handling", "negotiation_quality", "response_naturalness", "progression"}
        if not required.issubset(obj.keys()):
            return None
        # clamp scores to 1-5
        for k in required:
            obj[k] = max(1, min(5, int(obj[k])))
        obj["overall"] = round(mean(obj[k] for k in required), 2)
        return obj
    except Exception:
        return None


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Find inference files
    if args.inference_file:
        text_path   = Path(args.inference_file)
        speech_path = None
        if not text_path.exists():
            sys.exit(f"Inference file not found: {text_path}")
    else:
        text_path, speech_path = find_inference_files(args.version)

    text_data  = json.loads(text_path.read_text())
    examples   = text_data["per_example"]
    speech_wav_map = {}   # dialogue_id+turn_index → wav_file path
    if speech_path:
        speech_data = json.loads(speech_path.read_text())
        for e in speech_data["per_example"]:
            key = f"{e['dialogue_id']}_T{e['turn_index']}"
            speech_wav_map[key] = e.get("wav_file")

    if args.n_samples and args.n_samples < len(examples):
        import random
        random.seed(42)
        examples = random.sample(examples, args.n_samples)
        print(f"  Sampled {args.n_samples} examples from {len(text_data['per_example'])} total.")

    print(f"\n{'='*65}")
    print(f"  LLM Judge — {args.judge_model}")
    print(f"{'='*65}")
    print(f"  Text inference  : {text_path}")
    print(f"  Speech inference: {speech_path or 'not found'}")
    print(f"  Examples        : {len(examples)}")
    print(f"  Judge model     : {args.judge_model}")

    # ── Load judge model ───────────────────────────────────────────────────────
    print(f"\n  Loading judge model…")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.judge_model)
    judge_model = AutoModelForCausalLM.from_pretrained(
        args.judge_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device)
    judge_model.eval()
    print(f"  Judge loaded on {device}.")

    # ── Run judgement ──────────────────────────────────────────────────────────
    print(f"\n  Judging {len(examples)} examples…\n")
    judge_results = []
    parse_failures = 0

    for i, ex in enumerate(examples):
        user_prompt = build_judge_prompt(ex, text_data)

        messages = [
            {"role": "system",  "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user",    "content": user_prompt},
        ]

        # Qwen3 supports thinking mode — disable it for structured JSON output
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            out_ids = judge_model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        input_len  = inputs["input_ids"].shape[1]
        judge_text = tokenizer.decode(out_ids[0][input_len:], skip_special_tokens=True)
        scores     = parse_judge_output(judge_text)

        if scores is None:
            parse_failures += 1
            scores = {
                "emotion_handling": None, "negotiation_quality": None,
                "response_naturalness": None, "progression": None,
                "overall": None, "reasoning": "parse_failed",
            }

        key     = f"{ex['dialogue_id']}_T{ex['turn_index']}"
        wav_file = speech_wav_map.get(key)
        judge_results.append({
            "dialogue_id":      ex["dialogue_id"],
            "turn_index":       ex["turn_index"],
            "zone":             ex["zone"],
            "gt_decision":      ex["gt_decision"],
            "pred_decision":    ex["pred_decision"],
            "decision_correct": ex["decision_correct"],
            "gt_response":      ex["gt_response"],
            "pred_response":    ex["pred_response"],
            "wav_file":         wav_file,
            "scores":           scores,
            "raw_judge_output": judge_text,
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(examples):
            valid = [r for r in judge_results if r["scores"]["overall"] is not None]
            avg   = mean(r["scores"]["overall"] for r in valid) if valid else 0
            print(f"  [{i+1}/{len(examples)}]  running avg overall: {avg:.2f}/5.0")

    # ── Aggregate ──────────────────────────────────────────────────────────────
    valid_results = [r for r in judge_results if r["scores"]["overall"] is not None]
    n_valid = len(valid_results)

    def avg_dim(dim):
        vals = [r["scores"][dim] for r in valid_results if r["scores"][dim] is not None]
        return round(mean(vals), 2) if vals else None

    agg_scores = {
        "n_judged":            len(judge_results),
        "n_valid":             n_valid,
        "parse_failures":      parse_failures,
        "avg_overall":         avg_dim("overall") if n_valid else None,
        "avg_emotion_handling":      avg_dim("emotion_handling"),
        "avg_negotiation_quality":   avg_dim("negotiation_quality"),
        "avg_response_naturalness":  avg_dim("response_naturalness"),
        "avg_progression":           avg_dim("progression"),
    }

    # Breakdown by decision correctness
    correct_results   = [r for r in valid_results if r["decision_correct"]]
    incorrect_results = [r for r in valid_results if not r["decision_correct"]]
    if correct_results:
        agg_scores["avg_overall_correct_decisions"]   = round(mean(r["scores"]["overall"] for r in correct_results), 2)
    if incorrect_results:
        agg_scores["avg_overall_incorrect_decisions"] = round(mean(r["scores"]["overall"] for r in incorrect_results), 2)

    # Breakdown by zone
    for zone in ["pre_crystallisation", "post_crystallisation"]:
        zr = [r for r in valid_results if r["zone"] == zone]
        if zr:
            agg_scores[f"avg_overall_{zone}"] = round(mean(r["scores"]["overall"] for r in zr), 2)

    # ── Print summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  JUDGE RESULTS  ({n_valid}/{len(judge_results)} valid)")
    print(f"{'='*65}")
    print(f"  Overall score         : {agg_scores['avg_overall']}/5.0")
    print(f"  Emotion handling      : {agg_scores['avg_emotion_handling']}/5.0")
    print(f"  Negotiation quality   : {agg_scores['avg_negotiation_quality']}/5.0")
    print(f"  Response naturalness  : {agg_scores['avg_response_naturalness']}/5.0")
    print(f"  Progression           : {agg_scores['avg_progression']}/5.0")
    if "avg_overall_correct_decisions" in agg_scores:
        print(f"  Avg (correct dec.)    : {agg_scores['avg_overall_correct_decisions']}/5.0")
    if "avg_overall_incorrect_decisions" in agg_scores:
        print(f"  Avg (incorrect dec.)  : {agg_scores['avg_overall_incorrect_decisions']}/5.0")
    print(f"  Parse failures        : {parse_failures}/{len(judge_results)}")

    # ── Save ───────────────────────────────────────────────────────────────────
    label    = "text_and_speech" if speech_path else "text_only"
    sft_ver  = text_data.get("sft_version", args.version)
    out_path = text_path.parent / f"{sft_ver}_judge_{label}_{run_ts}.json"
    output = {
        "sft_version":        text_data.get("sft_version", "unknown"),
        "text_inference_file":  str(text_path),
        "speech_inference_file": str(speech_path) if speech_path else None,
        "judge_model":        args.judge_model,
        "judge_timestamp":    run_ts,
        "n_samples":          args.n_samples or "all",
        "note":               "Scores are from text inference (speech uses identical text — same scores apply). wav_file field links each example to its spoken audio.",
        "aggregate_scores":   agg_scores,
        "per_example":        judge_results,
    }
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))

    print(f"\n{'='*65}")
    print(f"  Results saved → {out_path}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
