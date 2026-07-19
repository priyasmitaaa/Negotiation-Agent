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


def negotiation_grpo_reward(completions, **kwargs) -> list[float]:
    rewards: list[float] = []
    for i, completion in enumerate(completions):
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)
        example = {key: values[i] for key, values in kwargs.items()}
        reward, _components = compute_reward(
            example,
            raw_output=text,
            previous_seller_responses=example.get("previous_seller_responses"),
            speech_mode=False,
        )
        rewards.append(float(reward))
    return rewards


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
