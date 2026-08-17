#!/usr/bin/env python3
"""
One-off check (not a permanent pipeline component): does
audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim — a dedicated
continuous-dimensional SER model (arousal/dominance/valence), trained on
natural (podcast) speech — separate calm vs. distressed buyer speech on our
TTS-synthesized negotiation audio? This is the same domain-transfer question
the Qwen2.5-Omni audio_tower check answered negatively for; no SER model has
published TTS-domain validation, so this check is required, not optional
(see conversation 2026-08-02).

Reuses the exact same sample clips (random 6v6 + 9 content-matched pairs)
already selected for the audio_tower check, for a directly comparable result.
Unlike that check, this model outputs the quantities of interest directly
(arousal/dominance/valence), so no pooling-strategy exploration is needed —
we just compare the raw 3D output between classes.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel

from sft_data_v6 import load_wav

MODEL_ID = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
SAMPLE_RATE = 16000


class RegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = torch.tanh(self.dense(x))
        x = self.dropout(x)
        return self.out_proj(x)


class EmotionModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()
        # Environment note (2026-08-02): the model card's __init__ predates a
        # newer transformers internal (`all_tied_weights_keys`, normally set
        # via a post-init hook this older custom-class snippet never calls).
        # This model has no tied weights (a regression head, not an
        # embedding-tied LM), so an empty dict is the correct value, not a
        # workaround masking a real tied-weight requirement.
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = {}

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)
        return hidden_states, logits


def predict(model, processor, wav_path: str) -> dict:
    signal = load_wav(Path(wav_path))
    y = processor(signal, sampling_rate=SAMPLE_RATE)
    y = torch.from_numpy(y["input_values"][0]).reshape(1, -1)
    with torch.no_grad():
        _hidden, logits = model(y)
    arousal, dominance, valence = logits.numpy()[0].tolist()
    return {"arousal": arousal, "dominance": dominance, "valence": valence}


def main():
    random_samples = json.loads(Path("/tmp/audio_check_samples.json").read_text())
    matched_pairs = json.loads(Path("/tmp/audio_check_matched_pairs.json").read_text())

    print(f"Loading {MODEL_ID}...")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = EmotionModel.from_pretrained(MODEL_ID)
    model.eval()

    results = {"random_content": {"calm": [], "distressed": []}, "content_matched": {"calm": [], "distressed": []}}

    print("=== Part 1: random content ===")
    for group, items in random_samples.items():
        for item in items:
            pred = predict(model, processor, item["wav"])
            print(f"  {group}: {item['dialogue_id']} T{item['turn_index']} -> {pred} (labels={item['labels']})")
            results["random_content"][group].append(pred)

    print("\n=== Part 2: content-matched pairs ===")
    for calm_item, distressed_item in matched_pairs:
        pred_c = predict(model, processor, calm_item["wav"])
        pred_d = predict(model, processor, distressed_item["wav"])
        print(f"  template: {calm_item['norm'][:50]}")
        print(f"    calm       -> {pred_c} (labels={calm_item['labels']})")
        print(f"    distressed -> {pred_d} (labels={distressed_item['labels']})")
        results["content_matched"]["calm"].append(pred_c)
        results["content_matched"]["distressed"].append(pred_d)

    summary = {}
    for condition, groups in results.items():
        summary[condition] = {}
        for dim in ["arousal", "dominance", "valence"]:
            calm_vals = [p[dim] for p in groups["calm"]]
            distressed_vals = [p[dim] for p in groups["distressed"]]
            summary[condition][dim] = {
                "calm_mean": round(float(np.mean(calm_vals)), 4),
                "distressed_mean": round(float(np.mean(distressed_vals)), 4),
                "diff": round(float(np.mean(distressed_vals) - np.mean(calm_vals)), 4),
                "calm_std": round(float(np.std(calm_vals)), 4),
                "distressed_std": round(float(np.std(distressed_vals)), 4),
            }

    print("\n" + json.dumps(summary, indent=2))
    Path("ser_model_separation_check_result.json").write_text(
        json.dumps({"raw": results, "summary": summary}, indent=2)
    )


if __name__ == "__main__":
    main()
