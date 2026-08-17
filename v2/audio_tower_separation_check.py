#!/usr/bin/env python3
"""
One-off check (not a permanent pipeline component): does Qwen2.5-Omni's
audio_tower embedding actually separate calm vs. distressed buyer speech, or
does it mostly cluster by lexical content? Answers this before committing to
using it as the frozen audio encoder for the causal RM's emotion/flip_score
grounding (see conversation 2026-08-01 for the full reasoning).

v2 (2026-08-01): first pass used mean pooling only and found a near-zero
separation gap (0.0127) on 6v6 random (content-mismatched) clips. Retrying
with (a) multiple pooling strategies (mean, std, mean+std concat) — std
pooling is more sensitive to localized prosodic fluctuation, which mean
pooling structurally washes out — and (b) content-matched pairs (identical
template text, only the emotion/price differs) to separate "encoder can't
hear affect" from "content differences swamp everything regardless of
pooling."

Uses a forward hook on model.audio_tower during a normal thinker forward
pass, rather than hand-computing feature_lens ourselves — reuses exactly the
same internal computation transformers already does, no reimplementation risk.
"""
import json
from pathlib import Path

import numpy as np
import torch
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

MODEL_ID = "Qwen/Qwen2.5-Omni-7B"


def load_thinker():
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    model.eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID)
    return model, processor


def raw_hidden_for_clip(model, processor, wav_path: str) -> np.ndarray:
    """Returns the full (seq_len, hidden_dim) audio_tower hidden state for
    one clip, captured via a forward hook — lets us compute multiple pooling
    strategies from one forward pass instead of rerunning per strategy."""
    captured = {}

    def hook(_module, _inputs, output):
        hidden = output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]
        captured["hidden"] = hidden.detach()

    handle = model.audio_tower.register_forward_hook(hook)
    try:
        messages = [{
            "role": "user",
            "content": [{"type": "audio", "audio": wav_path}, {"type": "text", "text": "Describe."}],
        }]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        from qwen_omni_utils import process_mm_info
        audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
        inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt")
        with torch.no_grad():
            model(**inputs)
    finally:
        handle.remove()

    return captured["hidden"].float().numpy()  # (seq_len, hidden_dim)


def pool(hidden: np.ndarray, strategy: str) -> np.ndarray:
    if strategy == "mean":
        return hidden.mean(axis=0)
    if strategy == "std":
        return hidden.std(axis=0)
    if strategy == "mean_std":
        return np.concatenate([hidden.mean(axis=0), hidden.std(axis=0)])
    raise ValueError(strategy)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def separation_stats(embs_by_group: dict) -> dict:
    labels, embs = [], []
    for g, items in embs_by_group.items():
        for e in items:
            labels.append(g)
            embs.append(e)
    n = len(embs)
    within, across = [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            sim = cosine(embs[i], embs[j])
            (within if labels[i] == labels[j] else across).append(sim)
    return {
        "mean_within_class_similarity": round(float(np.mean(within)), 4),
        "mean_across_class_similarity": round(float(np.mean(across)), 4),
        "separation_gap": round(float(np.mean(within) - np.mean(across)), 4),
    }


def main():
    random_samples = json.loads(Path("/tmp/audio_check_samples.json").read_text())
    matched_pairs = json.loads(Path("/tmp/audio_check_matched_pairs.json").read_text())

    model, processor = load_thinker()

    # --- Part 1: random 6v6 clips, multiple pooling strategies ---
    print("=== Part 1: random content, multiple pooling strategies ===")
    raw_hidden = {"calm": [], "distressed": []}
    for group, items in random_samples.items():
        for item in items:
            print(f"  embedding {group}: {item['dialogue_id']} T{item['turn_index']}")
            raw_hidden[group].append(raw_hidden_for_clip(model, processor, item["wav"]))

    results = {"random_content": {}}
    for strategy in ["mean", "std", "mean_std"]:
        pooled = {g: [pool(h, strategy) for h in hs] for g, hs in raw_hidden.items()}
        results["random_content"][strategy] = separation_stats(pooled)
        print(f"  [{strategy}] {results['random_content'][strategy]}")

    # --- Part 2: content-matched pairs (calm vs distressed, near-identical text) ---
    print("\n=== Part 2: content-matched pairs, multiple pooling strategies ===")
    raw_hidden_matched = {"calm": [], "distressed": []}
    for pair in matched_pairs:
        calm_item, distressed_item = pair
        print(f"  pair template: {calm_item['norm'][:50]}")
        raw_hidden_matched["calm"].append(raw_hidden_for_clip(model, processor, calm_item["wav"]))
        raw_hidden_matched["distressed"].append(raw_hidden_for_clip(model, processor, distressed_item["wav"]))

    results["content_matched"] = {}
    for strategy in ["mean", "std", "mean_std"]:
        pooled = {g: [pool(h, strategy) for h in hs] for g, hs in raw_hidden_matched.items()}
        results["content_matched"][strategy] = separation_stats(pooled)
        print(f"  [{strategy}] {results['content_matched'][strategy]}")

    print("\n" + json.dumps(results, indent=2))
    Path("audio_tower_separation_check_result.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
