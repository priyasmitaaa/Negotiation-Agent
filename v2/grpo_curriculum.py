#!/usr/bin/env python3
"""
Curriculum utilities for negotiation GRPO.

Three difficulty levels are used: easy, medium, hard. The labels are deterministic
and grounded in the v6 error analysis: post-crystallisation turns, MITIGATE cases,
negative/high-intensity emotion, price-boundary cases, and decision flips are
progressively harder than stable pre-crystallisation turns.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import TrainerCallback

from reward_function import compute_reward
from sft_data_v6 import DATASET_DIR, TARGET_SR, V2_ROOT, load_wav


AUDIO_DIR = V2_ROOT / "tts_outputs"
SPLITS_FILE = V2_ROOT / "splits_v6.json"
TEXT_INFERENCE_V6 = V2_ROOT / "sft_output_v6" / "20260614_010317" / "inference_20260620_150244.json"

SYSTEM_PROMPT = (
    "You are an AI negotiation assistant for a second-hand electronics shop in India. "
    "Guide the negotiation to within 10% of fair market value. Analyse the "
    "conversation and buyer emotional signals (emotion labels, intensity, valence). "
    "Produce in this exact order: <decision> [UNDECIDED/LEVERAGE/MITIGATE], "
    "<response>. Infer everything from raw signals."
)

CURRICULUM_WEIGHTS = {
    "easy": {"easy": 0.80, "medium": 0.20, "hard": 0.00},
    "medium": {"easy": 0.30, "medium": 0.55, "hard": 0.15},
    "hard": {"easy": 0.20, "medium": 0.35, "hard": 0.45},
}


def load_splits() -> dict:
    return json.loads(SPLITS_FILE.read_text())


def baseline_error_keys(inference_file: Optional[Path]) -> set[tuple[str, int]]:
    if not inference_file or not inference_file.exists():
        return set()
    data = json.loads(inference_file.read_text())
    return {
        (ex["dialogue_id"], int(ex["turn_index"]))
        for ex in data.get("per_example", [])
        if not ex.get("decision_correct", True)
    }


def price_gap_pct(buyer_offers: list, fair_value: Optional[float]) -> Optional[float]:
    if not buyer_offers or not fair_value:
        return None
    return 100.0 * (max(buyer_offers) - fair_value) / fair_value


def difficulty_for_example(ex: dict, baseline_errors: set[tuple[str, int]]) -> tuple[str, int, list[str]]:
    """Return (level, score, reasons)."""
    score = 0
    reasons: list[str] = []
    emotion = ex.get("prior_buyer_emotion") or {}
    intensity = (emotion.get("intensity") or "").lower()
    valence = (emotion.get("valence") or "").lower()
    gap = price_gap_pct(ex.get("buyer_offers") or [], ex.get("fair_value"))

    if ex.get("zone") == "post_crystallisation":
        score += 1
        reasons.append("post_crystallisation")
    if ex.get("gt_decision") == "MITIGATE":
        score += 1
        reasons.append("mitigate_decision")
    if intensity in {"medium", "high"}:
        score += 1
        reasons.append(f"{intensity}_emotion")
    if valence == "negative":
        score += 1
        reasons.append("negative_emotion")
    if gap is not None and abs(gap) <= 10:
        score += 1
        reasons.append("price_boundary_within_10pct")
    if ex.get("decision_flip"):
        score += 2
        reasons.append("decision_flip")
    if (ex["dialogue_id"], int(ex["turn_index"])) in baseline_errors:
        score += 2
        reasons.append("v6_baseline_error")

    if ex.get("decision_flip") or score >= 4:
        return "hard", score, reasons
    if score >= 2:
        return "medium", score, reasons
    return "easy", score, reasons


def build_grpo_examples(
    split: str = "train",
    include_audio: bool = True,
    baseline_inference_file: Optional[Path] = None,
) -> list[dict]:
    splits = load_splits()
    dialogue_ids = splits[split]
    baseline_errors = baseline_error_keys(baseline_inference_file)
    examples: list[dict] = []

    for did in dialogue_ids:
        data = json.loads((DATASET_DIR / f"{did}.json").read_text())
        seed = data["seed"]
        pricing = seed["pricing"]
        domain = seed["domain"]
        turns = data["processed"]
        turns_so_far: list[dict] = []
        previous_seller_responses: list[str] = []

        for turn in turns:
            if turn["speaker"] != "seller":
                turns_so_far.append(turn)
                continue

            prior_buyer = next((t for t in reversed(turns_so_far) if t["speaker"] == "buyer"), None)
            audio_paths = []
            if include_audio and prior_buyer:
                wav = AUDIO_DIR / did / f"turn_{prior_buyer['turn_index']:02d}_buyer.wav"
                if wav.exists():
                    audio_paths.append(str(wav))

            conversation = []
            for pt in turns_so_far:
                line = f"Turn {pt['turn_index']} [{pt['speaker'].upper()}]: {pt['text']}"
                if pt["speaker"] == "buyer":
                    em = pt.get("emotion", {})
                    if em.get("labels"):
                        line += (
                            f"  [Emotion: {', '.join(em['labels'])} | "
                            f"intensity: {em.get('intensity', '?')} | "
                            f"valence: {em.get('valence', '?')}]"
                        )
                conversation.append(line)

            user_text = (
                f"CONTEXT\n"
                f"Product: {domain['product']}. Condition: {domain['condition']}. "
                f"Asking: ₹{pricing['asking_price']}. Fair value: ₹{pricing['fair_value']}.\n\n"
                f"CONVERSATION SO FAR\n" + "\n".join(conversation) + "\n\n"
                f"TASK\nYou are the seller. It is now Turn {turn['turn_index']}. "
                f"Produce <decision>, <response>."
            )

            ex = {
                "dialogue_id": did,
                "turn_index": int(turn["turn_index"]),
                "prompt": [
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                    {
                        "role": "user",
                        "content": (
                            [{"type": "audio", "audio": ""} for _ in audio_paths]
                            + [{"type": "text", "text": user_text}]
                        ),
                    },
                ],
                "audio_paths": audio_paths,
                "gt_decision": turn["factor_state"]["decision"],
                "gt_response": turn["text"],
                "fair_value": pricing["fair_value"],
                "asking_price": pricing["asking_price"],
                "buyer_offers": turn["factor_state"].get("buyer_offers_so_far", []),
                "decision_flip": turn["factor_state"].get("decision_is_flip", False),
                "zone": turn["factor_state"].get("zone", ""),
                "prior_buyer_emotion": prior_buyer.get("emotion") if prior_buyer else None,
                "previous_seller_responses": previous_seller_responses[:],
            }
            level, score, reasons = difficulty_for_example(ex, baseline_errors)
            ex["difficulty_level"] = level
            ex["difficulty_score"] = score
            ex["difficulty_reasons"] = reasons
            examples.append(ex)

            previous_seller_responses.append(turn["text"])
            turns_so_far.append(turn)

    return examples


class NegotiationGRPODataset(Dataset):
    """Speech-conditioned GRPO dataset with three-stage curriculum sampling."""

    def __init__(
        self,
        split: str = "train",
        include_audio: bool = True,
        stage: str = "easy",
        baseline_inference_file: Optional[Path] = None,
        seed: int = 42,
    ):
        self.examples = build_grpo_examples(split, include_audio, baseline_inference_file)
        self.buckets = {
            level: [ex for ex in self.examples if ex["difficulty_level"] == level]
            for level in ("easy", "medium", "hard")
        }
        self.stage = stage
        self.rng = random.Random(seed)
        self.include_audio = include_audio

    def __len__(self) -> int:
        return len(self.examples)

    def set_stage(self, stage: str) -> None:
        if stage not in CURRICULUM_WEIGHTS:
            raise ValueError(f"Unknown curriculum stage: {stage}")
        self.stage = stage

    def set_progress(self, progress: float) -> None:
        if progress < 0.30:
            self.set_stage("easy")
        elif progress < 0.70:
            self.set_stage("medium")
        else:
            self.set_stage("hard")

    def __getitem__(self, index: int) -> dict:
        level = self._sample_level()
        bucket = self.buckets[level] or self.examples
        ex = self.rng.choice(bucket)
        audio = []
        if self.include_audio:
            for path in ex["audio_paths"]:
                arr = load_wav(Path(path))
                if arr is not None:
                    audio.append(arr)
        audio_value = audio[0] if len(audio) == 1 else audio
        return {
            "prompt": ex["prompt"],
            "audio": audio_value,
            "dialogue_id": ex["dialogue_id"],
            "turn_index": ex["turn_index"],
            "gt_decision": ex["gt_decision"],
            "gt_response": ex["gt_response"],
            "fair_value": ex["fair_value"],
            "asking_price": ex["asking_price"],
            "buyer_offers": ex["buyer_offers"],
            "decision_flip": ex["decision_flip"],
            "zone": ex["zone"],
            "prior_buyer_emotion": ex["prior_buyer_emotion"],
            "previous_seller_responses": ex["previous_seller_responses"],
            "difficulty_level": ex["difficulty_level"],
            "difficulty_score": ex["difficulty_score"],
            "difficulty_reasons": ex["difficulty_reasons"],
        }

    def _sample_level(self) -> str:
        weights = CURRICULUM_WEIGHTS[self.stage]
        levels = [level for level in ("easy", "medium", "hard") if self.buckets[level] and weights[level] > 0]
        if not levels:
            return "medium"
        probs = [weights[level] for level in levels]
        total = sum(probs)
        draw = self.rng.random() * total
        upto = 0.0
        for level, prob in zip(levels, probs):
            upto += prob
            if draw <= upto:
                return level
        return levels[-1]


class CurriculumCallback(TrainerCallback):
    def __init__(self, dataset: NegotiationGRPODataset):
        self.dataset = dataset

    def on_step_begin(self, args, state, control, **kwargs):
        max_steps = max(1, int(getattr(args, "max_steps", 0) or state.max_steps or 1))
        self.dataset.set_progress(state.global_step / max_steps)


def build_fixed_eval_slice(val_dataset: NegotiationGRPODataset, n: int = 20) -> List[Dict]:
    """A small, DETERMINISTIC (not randomly resampled) held-out slice for periodic
    in-training eval — NegotiationGRPODataset.__getitem__ samples randomly per call
    (curriculum sampling), which would make step-to-step accuracy numbers
    incomparable. Mirrors __getitem__'s audio-loading exactly, just over a fixed
    prefix of val_dataset.examples instead of a random draw."""
    slice_examples = val_dataset.examples[:n]
    out = []
    for ex in slice_examples:
        audio = []
        if val_dataset.include_audio:
            for path in ex["audio_paths"]:
                arr = load_wav(Path(path))
                if arr is not None:
                    audio.append(arr)
        out.append({
            "prompt": ex["prompt"],
            "audio": audio,
            "gt_decision": ex["gt_decision"],
        })
    return out


def evaluate_decision_accuracy(model, processor, eval_examples: List[Dict], device) -> float:
    """Greedy-decode a small held-out slice with the LIVE in-training model and
    score decision-tag accuracy against ground truth. This exists specifically to
    catch silent quality drift (a policy finding some way to game the causal RM's
    scalar reward without genuinely improving negotiation quality) that
    reward/entropy/grad_norm logging alone can't distinguish from real
    improvement — see causal_rm_results.md section 4.4's discussion of the
    step-32 pattern for why this gap matters. Deliberately minimal: greedy
    decoding, no speech, no full error-analysis.py-style breakdown — just a
    cheap accuracy number to log alongside the existing training metrics."""
    from inference_v6 import parse_output  # tiny regex parser, safe to import
                                            # (inference_v6.py's own CLI code is
                                            # guarded by `if __name__ == "__main__"`)

    was_training = model.training
    model.eval()
    correct = 0
    total = 0
    try:
        with torch.no_grad():
            for ex in eval_examples:
                text = processor.apply_chat_template(
                    ex["prompt"], tokenize=False, add_generation_prompt=True
                )
                kwargs = {"audio": ex["audio"]} if ex["audio"] else {}
                inputs = processor(text=[text], return_tensors="pt", padding=True, **kwargs)
                inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
                prompt_len = inputs["input_ids"].shape[1]
                out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
                gen_text = processor.tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
                parsed = parse_output(gen_text)
                total += 1
                if parsed["decision"] == ex["gt_decision"]:
                    correct += 1
    finally:
        if was_training:
            model.train()
    return correct / total if total else 0.0


class PeriodicEvalCallback(TrainerCallback):
    """Runs evaluate_decision_accuracy every `eval_steps` training steps and logs
    the result — both to stdout and, if given, as a JSONL line — alongside trl's
    own reward/entropy/grad_norm logging, so a training run's health signal
    includes a held-out correctness check, not just training-batch statistics."""

    def __init__(self, eval_examples: List[Dict], eval_steps: int = 50, log_path: Optional[Path] = None):
        self.eval_examples = eval_examples
        self.eval_steps = eval_steps
        self.log_path = log_path

    def on_step_end(self, args, state, control, model=None, processing_class=None, **kwargs):
        if self.eval_steps <= 0 or state.global_step == 0 or state.global_step % self.eval_steps != 0:
            return
        device = next(model.parameters()).device
        acc = evaluate_decision_accuracy(model, processing_class, self.eval_examples, device)
        print(f"[held-out eval] step={state.global_step} decision_accuracy={acc:.4f} "
              f"(n={len(self.eval_examples)})")
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a") as f:
                f.write(json.dumps({
                    "step": state.global_step,
                    "decision_accuracy": acc,
                    "n": len(self.eval_examples),
                }) + "\n")


def make_negotiation_grpo_reward(causal_rm: Optional[object] = None, component_log_path: Optional[Path] = None):
    """Build a GRPO reward function, optionally backed by a trained causal
    reward model instead of the keyword-heuristic reward components.
    `causal_rm=None` (the default) reproduces the exact original reward
    function unchanged — this is an opt-in integration point
    (causal_rubric_rl_plan.md Phase 4 / implementation step 7).

    IMPORTANT: only pass a real causal_rm here after it has passed
    causal_rm_audit.py's go/no-go check (causal_rm_architecture.md section
    7). Building this factory function does not itself activate anything —
    train_grpo_curriculum.py must be explicitly told to load and pass a
    checkpoint via --causal_rm_checkpoint.

    `component_log_path`: TRL's GRPOTrainer only consumes the scalar list
    this function returns — the per-dimension `components` dict from
    compute_reward() would otherwise be silently discarded every call,
    which means the exact reward-hacking signal we'd want to watch for
    during a GRPO pilot (does `overall` climb without genuine
    emotion/price_strategy/progression improvement) would be invisible
    until after the fact. When set, every call appends one JSON line per
    completion with its full component breakdown plus a monotonic call
    counter (proxy for training step ordering) to this file — read it back
    with summarize_component_log() below."""
    call_counter = 0

    def negotiation_grpo_reward(completions, **kwargs) -> list[float]:
        nonlocal call_counter
        rewards: list[float] = []
        log_lines: list[str] = []
        for i, completion in enumerate(completions):
            text = completion[0]["content"] if isinstance(completion, list) else str(completion)
            # trl's stock GRPOTrainer._calculate_rewards (unlike Omni-R1's vendored
            # trainer) injects reward_kwargs["trainer_state"] = self.state — a single
            # TrainerState object, not a per-example list — alongside the genuine
            # per-example dataset columns. Skip anything that isn't list/tuple-like
            # rather than assuming every kwarg is indexable per-completion; this is
            # additive (doesn't change Omni-R1's existing behavior, which never
            # injected a non-list kwarg here) and defensive against similar future
            # additions from either trainer.
            example = {
                key: values[i] for key, values in kwargs.items() if isinstance(values, (list, tuple))
            }
            reward, components = compute_reward(
                example,
                raw_output=text,
                previous_seller_responses=example.get("previous_seller_responses"),
                speech_mode=False,
                causal_rm=causal_rm,
            )
            rewards.append(float(reward))
            if component_log_path is not None:
                record = {
                    "call_index": call_counter,
                    "total": float(reward),
                    **{k: v for k, v in components.items() if not k.endswith("_meta") and k != "hard_gate"},
                }
                log_lines.append(json.dumps(record))
            call_counter += 1

        if component_log_path is not None and log_lines:
            with open(component_log_path, "a") as f:
                f.write("\n".join(log_lines) + "\n")

        return rewards

    return negotiation_grpo_reward


def summarize_component_log(log_path: Path, bucket_size: int = 20) -> List[Dict[str, float]]:
    """Read a component_log_path JSONL file back and compute per-dimension
    mean reward in buckets of `bucket_size` consecutive calls — a cheap
    proxy for "reward trajectory over training" without needing TRL-level
    step alignment. Used post-pilot to check for the reward-hacking tell:
    overall climbing without decision/emotion/price_strategy/progression
    climbing alongside it."""
    records = [json.loads(line) for line in Path(log_path).read_text().splitlines() if line.strip()]
    records.sort(key=lambda r: r["call_index"])

    buckets: List[Dict[str, float]] = []
    for i in range(0, len(records), bucket_size):
        chunk = records[i : i + bucket_size]
        keys = [k for k in chunk[0] if k != "call_index"]
        bucket = {"call_index_start": chunk[0]["call_index"], "n": len(chunk)}
        for k in keys:
            values = [r[k] for r in chunk if isinstance(r.get(k), (int, float))]
            if values:
                bucket[k] = sum(values) / len(values)
        buckets.append(bucket)
    return buckets


# Backward-compatible default (causal_rm=None): existing callers that import
# `negotiation_grpo_reward` directly keep working unchanged.
negotiation_grpo_reward = make_negotiation_grpo_reward(causal_rm=None)


def write_manifest(split: str, output: Path, include_audio: bool = True) -> None:
    examples = build_grpo_examples(split=split, include_audio=include_audio)
    counts = Counter(ex["difficulty_level"] for ex in examples)
    reason_counts = Counter(reason for ex in examples for reason in ex["difficulty_reasons"])
    payload = {
        "split": split,
        "n_examples": len(examples),
        "difficulty_counts": dict(counts),
        "difficulty_reason_counts": dict(reason_counts),
        "thresholds": {
            "easy": "score <= 1 and not decision_flip",
            "medium": "score 2-3 and not decision_flip",
            "hard": "decision_flip or score >= 4",
        },
        "score_features": [
            "post_crystallisation",
            "MITIGATE decision",
            "medium/high emotion intensity",
            "negative emotion valence",
            "buyer offer within 10% of fair value",
            "decision flip (+2)",
            "optional v6 baseline error (+2, when provided)",
        ],
        "examples": examples,
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--output", default="grpo_curriculum_manifest.json")
    parser.add_argument("--no_audio", action="store_true")
    args = parser.parse_args()
    write_manifest(args.split, Path(args.output), include_audio=not args.no_audio)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
