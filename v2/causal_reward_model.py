#!/usr/bin/env python3
"""
causal_reward_model.py — rubric-vector causal reward model.

Implements the architecture finalized in causal_rm_architecture.md:

    frozen Qwen3-8B backbone
        -> pooled hidden state at the final token
        -> shared MLP trunk (hidden_dim -> 512 -> 256), trainable
        -> 5 linear heads (one per rubric dimension), trainable
        -> tanh bound to [-1, 1] per dimension
        -> overall = fixed weighted sum (reward_function.py's WEIGHTS),
           NOT a learned combiner (see causal_rm_architecture.md section 4/6
           for why v1 keeps this interpretable rather than learned)

Why a shared trunk before specializing into 5 heads (rather than 5
independent linear probes straight off the backbone): the rubric dimensions
are not independent tasks. Softening a response for an upset buyer moves
emotion, progression, AND price_strategy together — a shared representation
lets the model first learn "this looks like a negotiation state" before
splitting into dimension-specific heads, standard multi-task learning
practice, and it is cheap (backbone stays frozen; only the trunk + 5 small
heads are trainable).

`decision_score` and `flip_score` are always computed and always returned
(never dropped) even though reward_function.py keeps decision_reward()/
flip_reward() rule-based and does not consume these two dimensions in
`overall` — they exist as diagnostics: agreement between the RM's own
decision_score and the rule-based decision_reward()'s verdict is evidence
the RM learned something real about causal factor C1, not just about
C2-C4 in isolation (see causal_rm_architecture.md section 3). Returning the
full 5-dimension vector unconditionally is also what makes per-dimension
training curves and audit plots possible later without re-running anything.

Usage as a library (mirrors reward_function.py::compute_reward's contract):
    from causal_reward_model import load_causal_rm, CausalRewardModel
    rm = load_causal_rm("causal_rm_checkpoint/")
    overall, components = rm.score(example, response_text)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model as _Wav2Vec2Model,
    Wav2Vec2PreTrainedModel as _Wav2Vec2PreTrainedModel,
)

from reward_function import WEIGHTS, CLIP_MIN, CLIP_MAX, clamp
from sft_data_v6 import load_wav


V2_ROOT = Path(__file__).parent

BACKBONE_MODEL_ID = "Qwen/Qwen3-8B"  # same judge model already used in judge.py

# The 5 rubric dimensions the RM predicts. Only OVERALL_DIMS feed into
# `overall`; the rest (DIAGNOSTIC_DIMS) are logged/cross-checked only, per
# causal_rm_architecture.md section 3.
RUBRIC_DIMENSIONS = ["decision_score", "price_strategy", "emotion", "progression", "flip_score"]
OVERALL_DIMS = ["price_strategy", "emotion", "progression"]
DIAGNOSTIC_DIMS = ["decision_score", "flip_score"]

TRUNK_HIDDEN = 512
TRUNK_OUT = 256


class RubricHead(nn.Module):
    """Shared trunk (hidden_dim -> 512 -> 256) + 5 linear heads (256 -> 1),
    each tanh-bounded to [-1, 1]. This is the ONLY trainable part of the
    causal reward model — the Qwen3-8B backbone stays frozen (see
    causal_rm_architecture.md section 4)."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim, TRUNK_HIDDEN),
            nn.GELU(),
            nn.Linear(TRUNK_HIDDEN, TRUNK_OUT),
            nn.GELU(),
        )
        self.heads = nn.ModuleDict({
            dim: nn.Linear(TRUNK_OUT, 1) for dim in RUBRIC_DIMENSIONS
        })

    def forward(self, pooled_hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        """pooled_hidden: (batch, hidden_dim) -> {dim: (batch,) in [-1, 1]}"""
        shared = self.trunk(pooled_hidden)
        return {dim: torch.tanh(head(shared).squeeze(-1)) for dim, head in self.heads.items()}

    def raw_logits(self, pooled_hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Pre-tanh logits, exposed for train_causal_rm.py's L_reg term
        (weight decay on head OUTPUTS, not weights — see
        causal_rm_architecture.md section 5)."""
        shared = self.trunk(pooled_hidden)
        return {dim: head(shared).squeeze(-1) for dim, head in self.heads.items()}


SER_DIM = 3  # audeering model's (arousal, dominance, valence) output — see
             # causal_rm_architecture.md section 9.5 for why these 3 final
             # dimensions were chosen over the model's richer internal embedding


class SERFusionHead(nn.Module):
    """v2 (causal_rm_architecture.md section 9): audio-grounds `emotion`
    only, via a second parallel trunk fed [text_hidden ‖ SER 3-dim vector].
    `decision_score`/`price_strategy`/`progression`/`flip_score` are
    untouched — they still come from an embedded, ordinary `RubricHead`.

    Deliberately a COMPOSITION wrapping an unmodified RubricHead, not a
    reimplementation of its trunk+heads: `self.base` is byte-for-byte the
    same class used throughout sections 1-8, which means (a) this really is
    additive — deleting the fusion trunk and emotion_head_fusion recovers
    the exact v1 architecture, not an approximation of it, and (b) a v1
    checkpoint's rubric_head.pt can be loaded directly into `self.base` to
    warm-start decision_score/price_strategy/progression/flip_score (and
    emotion's own text-only fallback path below) rather than training every
    dimension from scratch.

    Fallback behavior (no audio available for this example): `emotion`
    comes from `self.base`'s own emotion head, i.e. exactly the v1
    text-only computation — NOT a separate untrained head, and NOT routed
    through the fusion trunk with a zero-vector audio placeholder (which
    would be a different, untested distribution shift, not a real
    fallback). This is what "additive, not a breaking change" in section
    9.6 means concretely: with ser_vector=None, forward() reproduces v1
    RubricHead.forward() exactly, dimension-for-dimension."""

    def __init__(self, hidden_dim: int, ser_dim: int = SER_DIM):
        super().__init__()
        self.base = RubricHead(hidden_dim)
        self.fusion_trunk = nn.Sequential(
            nn.Linear(hidden_dim + ser_dim, TRUNK_HIDDEN),
            nn.GELU(),
            nn.Linear(TRUNK_HIDDEN, TRUNK_OUT),
            nn.GELU(),
        )
        self.emotion_head_fusion = nn.Linear(TRUNK_OUT, 1)

    def forward(
        self, pooled_hidden: torch.Tensor, ser_vector: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """pooled_hidden: (batch, hidden_dim). ser_vector: (batch, SER_DIM)
        or None. Returns {dim: (batch,) in [-1, 1]} for all 5
        RUBRIC_DIMENSIONS, same contract as RubricHead.forward()."""
        scores = self.base(pooled_hidden)  # all 5 dims via the unmodified v1 path
        if ser_vector is not None:
            fusion_input = torch.cat([pooled_hidden, ser_vector], dim=-1)
            fusion_shared = self.fusion_trunk(fusion_input)
            scores["emotion"] = torch.tanh(self.emotion_head_fusion(fusion_shared).squeeze(-1))
        return scores

    def raw_logits(
        self, pooled_hidden: torch.Tensor, ser_vector: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Pre-tanh logits, same rationale as RubricHead.raw_logits (L_reg
        term in train_causal_rm.py)."""
        logits = self.base.raw_logits(pooled_hidden)
        if ser_vector is not None:
            fusion_input = torch.cat([pooled_hidden, ser_vector], dim=-1)
            fusion_shared = self.fusion_trunk(fusion_input)
            logits["emotion"] = self.emotion_head_fusion(fusion_shared).squeeze(-1)
        return logits


def format_context_response(context: Dict[str, Any], response: str) -> str:
    """Build one text sequence from a (possibly partial) context dict + a
    candidate response, for the backbone to embed. Deliberately tolerant of
    partial context: generate_intervention_pairs.py's causal pairs carry
    different subsets of fields per factor (e.g. C1 pairs have
    zone/gt_decision/buyer_offers/fair_value, C3 pairs have only `emotion`),
    so this only renders the fields actually present rather than assuming a
    fixed schema."""
    lines = []

    if "product" in context or "condition" in context:
        lines.append(f"Product: {context.get('product', '?')}. Condition: {context.get('condition', '?')}.")
    if "asking_price" in context or "fair_value" in context:
        lines.append(f"Asking price: {context.get('asking_price', '?')}. Fair value: {context.get('fair_value', '?')}.")
    if "zone" in context:
        lines.append(f"Zone: {context['zone']}.")
    if "gt_decision" in context:
        lines.append(f"Decision context: {context['gt_decision']}.")
    if "buyer_offers" in context and context["buyer_offers"]:
        lines.append(f"Buyer offers so far: {context['buyer_offers']}.")
    if "decision_flip" in context:
        lines.append(f"Decision flip turn: {context['decision_flip']}.")
    if "emotion" in context and context["emotion"]:
        em = context["emotion"]
        lines.append(
            f"Buyer emotion: labels={em.get('labels', [])}, "
            f"intensity={em.get('intensity', '?')}, valence={em.get('valence', '?')}."
        )
    elif "prior_buyer_emotion" in context and context["prior_buyer_emotion"]:
        em = context["prior_buyer_emotion"]
        lines.append(
            f"Buyer emotion: labels={em.get('labels', [])}, "
            f"intensity={em.get('intensity', '?')}, valence={em.get('valence', '?')}."
        )

    context_str = " ".join(lines) if lines else "(no structured context provided)"
    return f"NEGOTIATION CONTEXT\n{context_str}\n\nSELLER RESPONSE\n{response}"


class CausalRewardModel:
    """Inference-time wrapper: frozen backbone + trainable RubricHead.
    `score()` mirrors reward_function.py::compute_reward's (overall,
    components) contract so it's a drop-in replacement for the heuristic
    price_strategy_reward/emotion_reward/progression_reward calls."""

    def __init__(
        self,
        backbone,
        tokenizer,
        head: RubricHead,
        device: str = "cuda",
        ser_model=None,
        ser_processor=None,
    ):
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.head = head.to(device)
        self.device = device
        # v2 (causal_rm_architecture.md section 9): only set when the loaded
        # checkpoint's config.json has audio_fusion=True (see load_causal_rm).
        # None for every v1 checkpoint, so score() below falls back to the
        # exact v1 (text-only) path automatically.
        self.ser_model = ser_model
        self.ser_processor = ser_processor
        self._ser_cache: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def _ser_vector(self, wav_path: Optional[str]) -> Optional[torch.Tensor]:
        """Mirrors train_causal_rm.py::SEREmbedder's caching (by wav_path,
        preprocessing single-sourced inside this method) but read-only/
        inference-time, so it lives here rather than importing the trainer
        module (which imports FROM this one — importing back would cycle)."""
        if self.ser_model is None or not wav_path:
            return None
        cached = self._ser_cache.get(wav_path)
        if cached is not None:
            return cached
        signal = load_wav(Path(wav_path))
        if signal is None:
            return None
        processed = self.ser_processor(signal, sampling_rate=SER_SAMPLE_RATE)
        input_values = torch.from_numpy(processed["input_values"][0]).reshape(1, -1).to(self.device)
        _hidden, logits = self.ser_model(input_values)
        vector = logits.float().clone()  # (1, SER_DIM)
        self._ser_cache[wav_path] = vector
        return vector

    @torch.no_grad()
    def pooled_hidden(self, text: str) -> torch.Tensor:
        """Embed one (context, response) text with the frozen backbone and
        return the final-token hidden state, (1, hidden_dim)."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
        out = self.backbone(**inputs, output_hidden_states=True)
        last_hidden = out.hidden_states[-1]  # (1, seq_len, hidden_dim)
        # Backbone runs in bf16 (memory); RubricHead's trunk/heads are fp32
        # (standard practice for a small trainable probe on frozen features —
        # more stable gradients than bf16 for a from-scratch-initialized head).
        # Cast here so the two never mismatch at the first Linear layer.
        return last_hidden[:, -1, :].float()

    def score(self, example: Dict[str, Any], response: str) -> Tuple[float, Dict[str, Any]]:
        """example: same dict shape used throughout the project (gt_decision,
        zone, prior_buyer_emotion, buyer_offers, fair_value, decision_flip,
        ...) — see grpo_curriculum.py::build_grpo_examples for the canonical
        set. Returns (overall, components) where components ALWAYS contains
        the full rubric vector (see causal_rm_architecture.md section 6)."""
        text = format_context_response(example, response)
        pooled = self.pooled_hidden(text)
        ser_vector = self._ser_vector(example.get("buyer_audio_path")) if isinstance(self.head, SERFusionHead) else None
        with torch.no_grad():
            scores = self.head(pooled, ser_vector=ser_vector) if isinstance(self.head, SERFusionHead) else self.head(pooled)
        scores = {dim: float(v.item()) for dim, v in scores.items()}

        overall = sum(WEIGHTS.get(_component_weight_key(dim), 0.0) * scores[dim] for dim in OVERALL_DIMS)
        overall = clamp(overall, CLIP_MIN, CLIP_MAX)

        components = dict(scores)
        components["overall"] = round(overall, 6)
        return overall, components


def _component_weight_key(rubric_dim: str) -> str:
    """Map rubric-vector dimension names to reward_function.py's WEIGHTS
    keys (price_strategy -> "price_strategy", emotion -> "emotion",
    progression -> "progression"; decision_score/flip_score have no WEIGHTS
    entry since they don't feed `overall`)."""
    return {"price_strategy": "price_strategy", "emotion": "emotion", "progression": "progression"}.get(rubric_dim, "")


def load_backbone(model_id: str = BACKBONE_MODEL_ID, device: str = "cuda"):
    """Load the frozen Qwen3-8B backbone, same loading pattern as judge.py."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, tokenizer


# ── v2 audio grounding (causal_rm_architecture.md section 9) ────────────────
# Frozen SER (speech emotion recognition) encoder feeding SERFusionHead's
# fusion trunk. Audited and selected 2026-08-02 (see architecture doc
# section 9.2): Qwen2.5-Omni's own audio_tower was tested first and
# rejected (no calm/distressed separation under any pooling strategy on
# this project's TTS negotiation speech); this model was tested and
# accepted (real, content-independent separation). License CC-BY-NC-SA-4.0
# — research-only, confirmed acceptable for this project.

SER_MODEL_ID = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
SER_SAMPLE_RATE = 16000


class _SERRegressionHead(nn.Module):
    """Verbatim from the model card's usage snippet (see architecture doc
    section 9.2) — this is the SER model's OWN classifier head, unrelated
    to and not to be confused with this project's RubricHead/SERFusionHead."""

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


class SEREmotionModel(_Wav2Vec2PreTrainedModel):
    """The audeering model's custom architecture, per its model card's
    usage snippet. Outputs (pooled_hidden_states, logits) where logits is
    (batch, 3) = (arousal, dominance, valence), each in ~[0, 1].

    __init__ sets `all_tied_weights_keys` explicitly (2026-08-02 environment
    note): the model card's original __init__ predates a `transformers`
    internal that's normally set via a post-init hook this older custom
    class never calls. This model has no tied weights (a regression head,
    not an embedding-tied LM), so an empty dict is the correct value, not a
    workaround masking a real tied-weight requirement — same fix already
    verified working in ser_model_separation_check.py's audit."""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = _Wav2Vec2Model(config)
        self.classifier = _SERRegressionHead(config)
        self.init_weights()
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = {}

    def forward(self, input_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)
        return hidden_states, logits


def load_ser_model(model_id: str = SER_MODEL_ID, device: str = "cuda"):
    """Load the frozen SER model + its processor. Same frozen-and-eval
    pattern as load_backbone — this model is never fine-tuned, only used as
    a fixed feature extractor feeding SERFusionHead's fusion trunk."""
    from transformers import Wav2Vec2Processor

    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = SEREmotionModel.from_pretrained(model_id).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, processor


def load_causal_rm(checkpoint_dir: str, device: str = "cuda") -> CausalRewardModel:
    """Load a trained RubricHead checkpoint on top of the frozen backbone.
    `checkpoint_dir` should contain `rubric_head.pt` (state_dict) and
    `config.json` (hidden_dim, backbone model id) written by
    train_causal_rm.py."""
    ckpt_dir = Path(checkpoint_dir)
    config = json.loads((ckpt_dir / "config.json").read_text())
    backbone, tokenizer = load_backbone(config.get("backbone_model_id", BACKBONE_MODEL_ID), device=device)

    # v2 (causal_rm_architecture.md section 9): auto-detect from the
    # checkpoint's own config.json rather than a caller-supplied flag, so a
    # v2 checkpoint can never accidentally get loaded as plain RubricHead
    # (which would silently drop the fusion_trunk/emotion_head_fusion
    # weights via a state_dict mismatch) — absent for every v1 checkpoint,
    # so `.get(..., False)` keeps old checkpoints loading exactly as before.
    audio_fusion = config.get("audio_fusion", False)
    head = SERFusionHead(hidden_dim=config["hidden_dim"]) if audio_fusion else RubricHead(hidden_dim=config["hidden_dim"])
    head.load_state_dict(torch.load(ckpt_dir / "rubric_head.pt", map_location=device))
    head.eval()

    ser_model = ser_processor = None
    if audio_fusion:
        ser_model, ser_processor = load_ser_model(device=device)

    return CausalRewardModel(backbone, tokenizer, head, device=device, ser_model=ser_model, ser_processor=ser_processor)
