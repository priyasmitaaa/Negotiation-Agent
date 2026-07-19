#!/usr/bin/env python3
"""
Executable reward function for negotiation-agent RL.

This implements the reward design documented in rl_reward_function_v6.md, but the
script name is intentionally version-neutral so it can be reused after v6.

Usage as a library:
    from reward_function import compute_reward
    reward, components = compute_reward(example, raw_output)

Offline audit:
    python3 reward_function.py \
        --inference_file sft_output_v6/20260614_010317/inference_20260620_150244.json \
        --speech_inference_file sft_output_v6/20260614_010317/inference_speech_20260615_102532.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import statistics
import wave
from array import array
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


VALID_DECISIONS = {"UNDECIDED", "LEVERAGE", "MITIGATE"}

WEIGHTS = {
    "decision": 1.00,
    "emotion": 0.30,
    "progression": 0.25,
    "price_strategy": 0.25,
    "format": 0.15,
    "naturalness": 0.10,
    "repetition_penalty": -0.20,
    "language_impurity_penalty": -0.20,
    "audio_quality": 0.10,
    "flip": 0.20,
}

CLIP_MIN = -2.0
CLIP_MAX = 2.0

LEVERAGE_TERMS = {
    "firm", "value", "condition", "market", "worth", "cannot", "can't",
    "asking", "popular", "lowest", "fixed", "premium", "rare",
}
MITIGATE_TERMS = {
    "can do", "willing", "meet", "close", "fair", "work with", "reduce",
    "accept", "deal", "settle", "flexible", "make it",
}
UNDECIDED_TERMS = {
    "appreciate", "asking", "consider", "tell me", "offer", "thanks",
    "what", "could", "would",
}
ACK_TERMS = {
    "understand", "appreciate", "fair", "i hear", "i get", "thanks",
    "thank", "see where", "makes sense",
}
SOFTEN_TERMS = {
    "work with", "meet", "can do", "let's", "i can", "willing",
    "reduce", "flexible",
}
PROGRESSION_TERMS = {
    "can do", "i can", "meet", "settle", "close", "deal", "pickup",
    "payment", "today", "final", "counter", "offer", "accept", "reduce",
    "make it", "if you can", "would you",
}
NEGATIVE_EMOTIONS = {
    "frustrated", "skeptical", "hesitant", "pleading", "resigned",
    "disappointed", "annoyed", "concerned", "worried", "uncertain",
}
POSITIVE_EMOTIONS = {
    "eager", "hopeful", "agreeable", "appreciative", "polite",
    "interested", "enthusiastic", "excited",
}


def clamp(value: float, lo: float = CLIP_MIN, hi: float = CLIP_MAX) -> float:
    return max(lo, min(hi, value))


def normalize_text(text: Optional[str]) -> str:
    return (text or "").strip()


def lower_text(text: Optional[str]) -> str:
    return normalize_text(text).lower()


def word_count(text: Optional[str]) -> int:
    return len(re.findall(r"\b[\w']+\b", text or ""))


def contains_any(text: str, terms: Iterable[str]) -> bool:
    text_l = lower_text(text)
    return any(term in text_l for term in terms)


def parse_model_output(raw_output: Optional[str]) -> Dict[str, Optional[str]]:
    """Parse deployment-format model output."""
    raw = raw_output or ""
    dec_match = re.search(
        r"<decision>\s*(UNDECIDED|LEVERAGE|MITIGATE)\s*</decision>",
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    )
    resp_match = re.search(
        r"<response>\s*(.*?)\s*</response>",
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    )
    decision = dec_match.group(1).upper() if dec_match else None
    response = resp_match.group(1).strip() if resp_match else None
    return {"decision": decision, "response": response, "raw": raw}


def get_prediction(example: Dict[str, Any], raw_output: Optional[str]) -> Dict[str, Optional[str]]:
    parsed = parse_model_output(raw_output or example.get("raw_output") or "")
    decision = example.get("pred_decision") or parsed["decision"]
    response = example.get("pred_response") or parsed["response"]
    return {
        "decision": decision if decision in VALID_DECISIONS else None,
        "response": response,
        "raw": raw_output or example.get("raw_output") or "",
    }


def ngrams(text: str, n: int = 3) -> set:
    words = re.findall(r"\b[\w']+\b", lower_text(text))
    return {tuple(words[i : i + n]) for i in range(max(0, len(words) - n + 1))}


def ngram_overlap(candidate: str, reference: str, n: int = 3) -> float:
    cand = ngrams(candidate, n)
    ref = ngrams(reference, n)
    if not cand or not ref:
        return 0.0
    return len(cand & ref) / len(cand)


def format_reward(raw_output: str, decision: Optional[str], response: Optional[str]) -> Tuple[float, Dict[str, Any]]:
    decision_tags = len(re.findall(r"<decision>.*?</decision>", raw_output or "", flags=re.I | re.S))
    response_tags = len(re.findall(r"<response>.*?</response>", raw_output or "", flags=re.I | re.S))
    reasoning_present = bool(re.search(r"<reasoning>.*?</reasoning>", raw_output or "", flags=re.I | re.S))

    if decision not in VALID_DECISIONS:
        score = -1.0
    elif not response:
        score = -0.5
    elif decision_tags != 1 or response_tags != 1:
        score = -0.5
    elif reasoning_present:
        score = -0.5
    else:
        score = 1.0

    return score, {
        "decision_tags": decision_tags,
        "response_tags": response_tags,
        "reasoning_present": reasoning_present,
        "valid_decision": decision in VALID_DECISIONS,
        "has_response": bool(response),
    }


def decision_reward(pred_decision: Optional[str], gt_decision: Optional[str]) -> float:
    if pred_decision == gt_decision and pred_decision in VALID_DECISIONS:
        return 1.0
    if gt_decision == "MITIGATE" and pred_decision == "LEVERAGE":
        return -1.25
    if gt_decision == "LEVERAGE" and pred_decision == "MITIGATE":
        return -0.75
    return -1.0


def price_strategy_reward(example: Dict[str, Any], pred_decision: Optional[str], response: str) -> Tuple[float, Dict[str, Any]]:
    gt_decision = example.get("gt_decision")
    zone = example.get("zone") or ""
    decision_correct = pred_decision == gt_decision

    response_l = lower_text(response)
    has_leverage = contains_any(response_l, LEVERAGE_TERMS)
    has_mitigate = contains_any(response_l, MITIGATE_TERMS)
    has_undecided = contains_any(response_l, UNDECIDED_TERMS)
    mentions_price = bool(re.search(r"[$\u20b9]?\s*\d{2,}", response_l))

    if not decision_correct and zone == "post_crystallisation":
        score = -1.0
    elif not decision_correct:
        score = -0.75
    elif pred_decision == "LEVERAGE":
        score = 1.0 if (has_leverage or mentions_price) and not has_mitigate else 0.5
    elif pred_decision == "MITIGATE":
        score = 1.0 if has_mitigate or mentions_price else 0.5
    elif pred_decision == "UNDECIDED":
        score = 1.0 if has_undecided or mentions_price else 0.5
    else:
        score = -0.5

    if decision_correct and contradicts_decision(pred_decision, response_l):
        score = min(score, -0.5)

    return score, {
        "mentions_price": mentions_price,
        "has_leverage_terms": has_leverage,
        "has_mitigate_terms": has_mitigate,
        "has_undecided_terms": has_undecided,
        "zone": zone,
    }


def contradicts_decision(pred_decision: Optional[str], response_l: str) -> bool:
    if pred_decision == "LEVERAGE":
        return contains_any(response_l, {"accept", "deal", "you can have it"}) and not contains_any(response_l, {"if", "but"})
    if pred_decision == "MITIGATE":
        return contains_any(response_l, {"cannot", "can't", "firm", "fixed"}) and not contains_any(response_l, {"but", "still"})
    return False


def emotion_reward(example: Dict[str, Any], response: str) -> Tuple[float, Dict[str, Any]]:
    emotion = example.get("prior_buyer_emotion") or {}
    labels = [lower_text(x) for x in emotion.get("labels") or []]
    intensity = lower_text(emotion.get("intensity"))
    valence = lower_text(emotion.get("valence"))

    if not labels and not intensity and not valence:
        return 0.0, {"emotion_present": False}

    response_l = lower_text(response)
    acknowledges = contains_any(response_l, ACK_TERMS)
    softens = contains_any(response_l, SOFTEN_TERMS)
    closure = contains_any(response_l, {"deal", "close", "settle", "today", "pickup", "accept"})
    negative = valence == "negative" or any(label in NEGATIVE_EMOTIONS for label in labels)
    positive = valence == "positive" or any(label in POSITIVE_EMOTIONS for label in labels)
    high_intensity = intensity == "high"

    if negative and high_intensity and not (acknowledges or softens):
        score = -1.0
    elif negative and (acknowledges or softens):
        score = 1.0
    elif negative:
        score = 0.5 if not escalates_negative_emotion(response_l) else -0.5
    elif positive and closure:
        score = 1.0
    elif positive and (acknowledges or contains_any(response_l, {"great", "glad"})):
        score = 0.5
    elif high_intensity and not (acknowledges or softens):
        score = -0.5
    else:
        score = 0.5

    return score, {
        "labels": labels,
        "intensity": intensity,
        "valence": valence,
        "acknowledges": acknowledges,
        "softens": softens,
        "closure_oriented": closure,
    }


def escalates_negative_emotion(response_l: str) -> bool:
    return contains_any(response_l, {"take it or leave it", "no way", "wasting", "ridiculous"})


def progression_reward(response: str) -> Tuple[float, Dict[str, Any]]:
    wc = word_count(response)
    response_l = lower_text(response)
    has_price = bool(re.search(r"[$\u20b9]?\s*\d{2,}", response_l))
    has_next_step = contains_any(response_l, PROGRESSION_TERMS)
    is_static = contains_any(response_l, {"as i said", "like i said", "already told"})

    if wc < 5:
        score = -1.0
    elif is_static:
        score = -0.5
    elif has_price and has_next_step:
        score = 1.0
    elif has_price or has_next_step:
        score = 0.5
    else:
        score = 0.0

    return score, {"word_count": wc, "has_price": has_price, "has_next_step": has_next_step, "static_phrase": is_static}


def repetition_penalty(response: str, previous_seller_responses: Optional[List[str]] = None) -> Tuple[float, Dict[str, Any]]:
    previous = previous_seller_responses or []
    if not response or not previous:
        return 0.0, {"max_3gram_overlap": 0.0}

    max_overlap = max(ngram_overlap(response, prior, n=3) for prior in previous)
    if max_overlap >= 0.50:
        penalty = 1.0
    elif max_overlap >= 0.30:
        penalty = 0.5
    else:
        penalty = 0.0
    return penalty, {"max_3gram_overlap": round(max_overlap, 4)}


def naturalness_reward(response: str) -> Tuple[float, Dict[str, Any]]:
    wc = word_count(response)
    malformed = bool(re.search(r"</?(decision|response|reasoning)>", response or "", flags=re.I))
    repeated_punct = bool(re.search(r"([!?.,])\1{2,}", response or ""))

    if not response or wc == 0:
        score = -1.0
    elif malformed or repeated_punct:
        score = -0.5
    elif 8 <= wc <= 35:
        score = 1.0
    elif 5 <= wc < 8 or 36 <= wc <= 60:
        score = 0.5
    else:
        score = -0.5

    return score, {"word_count": wc, "malformed_inner_tags": malformed, "repeated_punctuation": repeated_punct}


def language_impurity_penalty(response: str) -> Tuple[float, Dict[str, Any]]:
    if not response:
        return 0.0, {"non_ascii_chars": []}

    allowed = {"\u20b9"}
    non_ascii = sorted({ch for ch in response if ord(ch) > 127 and ch not in allowed})
    has_cjk = bool(re.search(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", response))

    if has_cjk:
        penalty = 1.0
    elif non_ascii:
        penalty = 0.5
    else:
        penalty = 0.0

    return penalty, {"non_ascii_chars": non_ascii[:20], "has_cjk_or_korean_japanese": has_cjk}


def flip_reward(example: Dict[str, Any], pred_decision: Optional[str]) -> float:
    if not example.get("decision_flip"):
        return 0.0
    return 0.5 if pred_decision == example.get("gt_decision") else -0.5


def audio_quality_reward(
    wav_file: Optional[str],
    response: str,
    require_audio: bool = False,
) -> Tuple[float, Dict[str, Any]]:
    """Lightweight WAV-quality guardrail using only Python stdlib."""
    metrics: Dict[str, Any] = {
        "audio_exists": False,
        "audio_duration_valid": False,
        "audio_not_silent": False,
        "audio_not_clipped": False,
        "audio_no_abrupt_cutoff": False,
        "audio_smoothness_ok": False,
        "audio_intelligibility_ok": None,
    }

    if not wav_file:
        return (-1.0 if require_audio else 0.0), metrics

    path = Path(wav_file)
    if not path.exists() or path.stat().st_size == 0:
        return -1.0, metrics

    metrics["audio_exists"] = True
    try:
        samples, sample_rate, channels = read_wav_samples(path)
    except Exception as exc:
        metrics["error"] = str(exc)
        return -1.0, metrics

    if not samples or sample_rate <= 0:
        return -1.0, metrics

    duration = len(samples) / float(sample_rate * max(channels, 1))
    metrics["duration_seconds"] = round(duration, 3)
    metrics["sample_rate"] = sample_rate
    metrics["channels"] = channels

    expected_seconds = max(0.8, word_count(response) / 2.3)
    metrics["expected_seconds"] = round(expected_seconds, 3)
    metrics["audio_duration_valid"] = 0.4 * expected_seconds <= duration <= 2.5 * expected_seconds

    rms = math.sqrt(sum(x * x for x in samples) / len(samples))
    peak = max(abs(x) for x in samples)
    clip_fraction = sum(1 for x in samples if abs(x) >= 0.98) / len(samples)
    metrics["rms"] = round(rms, 5)
    metrics["peak"] = round(peak, 5)
    metrics["clip_fraction"] = round(clip_fraction, 6)

    metrics["audio_not_silent"] = rms >= 0.005 and peak >= 0.02
    metrics["audio_not_clipped"] = clip_fraction <= 0.005
    metrics["audio_no_abrupt_cutoff"] = no_abrupt_cutoff(samples, sample_rate, channels)
    metrics["audio_smoothness_ok"] = smoothness_ok(samples)

    failed = [
        name for name in (
            "audio_duration_valid",
            "audio_not_silent",
            "audio_not_clipped",
            "audio_no_abrupt_cutoff",
            "audio_smoothness_ok",
        )
        if not metrics[name]
    ]

    if not failed:
        score = 1.0
    elif len(failed) == 1 and failed[0] in {"audio_duration_valid", "audio_smoothness_ok"}:
        score = 0.5
    elif "audio_not_silent" in failed or "audio_not_clipped" in failed:
        score = -0.5
    else:
        score = 0.0

    metrics["failed_audio_checks"] = failed
    return score, metrics


def read_wav_samples(path: Path) -> Tuple[List[float], int, int]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    if sample_width == 1:
        arr = array("B")
        arr.frombytes(frames)
        samples = [(x - 128) / 128.0 for x in arr]
    elif sample_width == 2:
        arr = array("h")
        arr.frombytes(frames)
        samples = [max(-1.0, min(1.0, x / 32768.0)) for x in arr]
    elif sample_width == 4:
        arr = array("i")
        arr.frombytes(frames)
        samples = [max(-1.0, min(1.0, x / 2147483648.0)) for x in arr]
    else:
        raise ValueError(f"unsupported WAV sample width: {sample_width}")
    return samples, sample_rate, channels


def no_abrupt_cutoff(samples: List[float], sample_rate: int, channels: int) -> bool:
    window = max(1, int(0.2 * sample_rate * max(channels, 1)))
    if len(samples) <= window * 2:
        return True
    tail = samples[-window:]
    before = samples[-2 * window : -window]
    tail_rms = math.sqrt(sum(x * x for x in tail) / len(tail))
    before_rms = math.sqrt(sum(x * x for x in before) / len(before))
    if before_rms < 0.005:
        return True
    return tail_rms <= before_rms * 1.5


def smoothness_ok(samples: List[float]) -> bool:
    if len(samples) < 100:
        return False
    step = max(1, len(samples) // 20000)
    sampled = samples[::step]
    diffs = [abs(sampled[i] - sampled[i - 1]) for i in range(1, len(sampled))]
    if not diffs:
        return False
    large_jump_fraction = sum(1 for d in diffs if d > 0.75) / len(diffs)
    near_zero_runs = longest_low_motion_run(sampled)
    return large_jump_fraction < 0.01 and near_zero_runs < max(100, len(sampled) * 0.35)


def longest_low_motion_run(samples: List[float], threshold: float = 1e-5) -> int:
    longest = 0
    current = 0
    for i in range(1, len(samples)):
        if abs(samples[i] - samples[i - 1]) <= threshold:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def compute_reward(
    example: Dict[str, Any],
    raw_output: Optional[str] = None,
    previous_seller_responses: Optional[List[str]] = None,
    wav_file: Optional[str] = None,
    speech_mode: bool = False,
    require_audio: bool = False,
) -> Tuple[float, Dict[str, Any]]:
    """Compute total clipped reward and return component details."""
    pred = get_prediction(example, raw_output)
    pred_decision = pred["decision"]
    pred_response = pred["response"] or ""
    raw = pred["raw"]

    components: Dict[str, Any] = {}

    r_format, format_meta = format_reward(raw, pred_decision, pred_response)
    components["format"] = r_format
    components["format_meta"] = format_meta

    if pred_decision not in VALID_DECISIONS:
        components["decision"] = -1.0
        components["hard_gate"] = "invalid_decision"
        return CLIP_MIN, components

    components["decision"] = decision_reward(pred_decision, example.get("gt_decision"))

    r_price, price_meta = price_strategy_reward(example, pred_decision, pred_response)
    components["price_strategy"] = r_price
    components["price_strategy_meta"] = price_meta

    r_emotion, emotion_meta = emotion_reward(example, pred_response)
    components["emotion"] = r_emotion
    components["emotion_meta"] = emotion_meta

    r_progression, progression_meta = progression_reward(pred_response)
    components["progression"] = r_progression
    components["progression_meta"] = progression_meta

    r_naturalness, naturalness_meta = naturalness_reward(pred_response)
    components["naturalness"] = r_naturalness
    components["naturalness_meta"] = naturalness_meta

    p_repetition, repetition_meta = repetition_penalty(pred_response, previous_seller_responses)
    components["repetition_penalty"] = p_repetition
    components["repetition_meta"] = repetition_meta

    p_language, language_meta = language_impurity_penalty(pred_response)
    components["language_impurity_penalty"] = p_language
    components["language_impurity_meta"] = language_meta

    components["flip"] = flip_reward(example, pred_decision)

    use_wav = wav_file or example.get("wav_file")
    if speech_mode or use_wav or require_audio:
        r_audio, audio_meta = audio_quality_reward(use_wav, pred_response, require_audio=require_audio)
        components["audio_quality"] = r_audio
        components["audio_quality_meta"] = audio_meta
    else:
        components["audio_quality"] = None

    total = 0.0
    total += WEIGHTS["decision"] * components["decision"]
    total += WEIGHTS["emotion"] * components["emotion"]
    total += WEIGHTS["progression"] * components["progression"]
    total += WEIGHTS["price_strategy"] * components["price_strategy"]
    total += WEIGHTS["format"] * components["format"]
    total += WEIGHTS["naturalness"] * components["naturalness"]
    total += WEIGHTS["repetition_penalty"] * components["repetition_penalty"]
    total += WEIGHTS["language_impurity_penalty"] * components["language_impurity_penalty"]
    total += WEIGHTS["flip"] * components["flip"]

    if components["audio_quality"] is not None:
        total += WEIGHTS["audio_quality"] * components["audio_quality"]

    components["unclipped_total"] = round(total, 6)
    components["total"] = round(clamp(total), 6)
    return components["total"], components


def audit_inference_file(
    inference_file: Path,
    speech_inference_file: Optional[Path] = None,
    output_file: Optional[Path] = None,
    n_samples: int = 0,
    require_audio: bool = False,
) -> Path:
    data = json.loads(inference_file.read_text())
    examples = list(data.get("per_example") or [])
    if n_samples:
        examples = examples[:n_samples]

    speech_wavs: Dict[Tuple[str, int], str] = {}
    if speech_inference_file:
        speech_data = json.loads(speech_inference_file.read_text())
        for ex in speech_data.get("per_example") or []:
            key = (ex.get("dialogue_id"), ex.get("turn_index"))
            if ex.get("wav_file"):
                speech_wavs[key] = ex["wav_file"]

    scored = []
    seen_responses: Dict[str, List[str]] = defaultdict(list)
    for ex in sorted(examples, key=lambda e: (str(e.get("dialogue_id")), int(e.get("turn_index") or 0))):
        dialogue_id = str(ex.get("dialogue_id"))
        key = (ex.get("dialogue_id"), ex.get("turn_index"))
        wav_file = speech_wavs.get(key) or ex.get("wav_file")
        speech_mode = bool(wav_file or speech_inference_file)
        total, components = compute_reward(
            ex,
            previous_seller_responses=seen_responses[dialogue_id],
            wav_file=wav_file,
            speech_mode=speech_mode,
            require_audio=require_audio and speech_mode,
        )
        pred_response = ex.get("pred_response")
        if pred_response:
            seen_responses[dialogue_id].append(pred_response)

        scored.append({
            "dialogue_id": ex.get("dialogue_id"),
            "turn_index": ex.get("turn_index"),
            "gt_decision": ex.get("gt_decision"),
            "pred_decision": ex.get("pred_decision"),
            "decision_correct": ex.get("decision_correct"),
            "total_reward": total,
            "components": components,
        })

    summary = summarize_scores(scored)
    result = {
        "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "inference_file": str(inference_file),
        "speech_inference_file": str(speech_inference_file) if speech_inference_file else None,
        "weights": WEIGHTS,
        "clip": [CLIP_MIN, CLIP_MAX],
        "summary": summary,
        "per_example": scored,
    }

    if output_file is None:
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = inference_file.with_name(f"reward_audit_{ts}.json")
    output_file.write_text(json.dumps(result, indent=2))
    return output_file


def summarize_scores(scored: List[Dict[str, Any]]) -> Dict[str, Any]:
    totals = [x["total_reward"] for x in scored]
    summary: Dict[str, Any] = {
        "n": len(scored),
        "avg_total_reward": round(statistics.mean(totals), 4) if totals else None,
        "min_total_reward": min(totals) if totals else None,
        "max_total_reward": max(totals) if totals else None,
    }
    component_names = [
        "format", "decision", "price_strategy", "emotion", "progression",
        "naturalness", "repetition_penalty", "language_impurity_penalty",
        "flip", "audio_quality",
    ]
    for name in component_names:
        values = [
            ex["components"].get(name)
            for ex in scored
            if ex["components"].get(name) is not None
        ]
        if values:
            summary[f"avg_{name}"] = round(statistics.mean(values), 4)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute reward audit for negotiation inference outputs.")
    parser.add_argument("--inference_file", required=True, help="Text or speech inference JSON.")
    parser.add_argument("--speech_inference_file", default=None, help="Optional speech inference JSON to attach WAV paths.")
    parser.add_argument("--output_file", default=None, help="Where to save reward audit JSON.")
    parser.add_argument("--n_samples", type=int, default=0, help="Limit to first N examples for quick checks.")
    parser.add_argument("--require_audio", action="store_true", help="Penalize missing WAVs in speech-mode audits.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = audit_inference_file(
        inference_file=Path(args.inference_file),
        speech_inference_file=Path(args.speech_inference_file) if args.speech_inference_file else None,
        output_file=Path(args.output_file) if args.output_file else None,
        n_samples=args.n_samples,
        require_audio=args.require_audio,
    )
    print(f"Saved reward audit to {out}")


if __name__ == "__main__":
    main()
