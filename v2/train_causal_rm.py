#!/usr/bin/env python3
"""
train_causal_rm.py — train the RubricHead (shared trunk + 5 rubric heads)
on top of a frozen Qwen3-8B backbone, using the intervention pairs from
generate_intervention_pairs.py.

Implements the loss formulation from causal_rm_architecture.md section 5:

    L_total = lambda_bt * L_causal + lambda_inv * L_spurious + lambda_reg * L_reg
  - L_causal: Bradley-Terry ranking loss, routed to the ONE head matching
    each causal pair's target_factor (C1_decision_zone -> decision_score,
    C3_emotion -> emotion, C5_flip -> flip_score). Only that head gets
    gradient from a given causal pair.
  - L_spurious: squared-difference invariance penalty across ALL 5 rubric
    dimensions (not just the 3 overall-feeding ones) for every spurious
    pair (S1 length, S2 politeness, S3 formatting). Originally covered only
    price_strategy/emotion/progression; widened 2026-08-01 after a
    lambda_inv sweep showed decision_score/flip_score drifting as an
    unsupervised side-effect of the shared trunk (see spurious_pair_loss's
    docstring for the full finding) — they still don't feed into `overall`,
    they just now get their own regularization signal too.
  - L_reg: small L2 penalty on the pre-tanh head logits (not the weights),
    added purely for training stability — see causal_rm_architecture.md
    section 5 for why this is needed even though BT/invariance losses are
    themselves scale-tolerant.

Backbone stays frozen throughout (no gradient, no optimizer state) — only
the RubricHead (shared trunk + 5 small linear heads) is trained. This keeps
training cheap relative to fine-tuning the 8B backbone, consistent with the
project's "frontload cost into RM training, cheap at GRPO rollout time"
goal (see causal_rubric_rl_plan.md).

IMPORTANT — spurious pairs do not inline context (implementation note from
causal_rm_architecture.md section 5.2): generate_intervention_pairs.py only
stores dialogue_id/turn_index for spurious pairs, not a context dict, so
this script reconstructs context once per split via
grpo_curriculum.py::build_grpo_examples and indexes by (dialogue_id,
turn_index) — the exact pattern generate_intervention_pairs.py's own
build_donor_index() already uses.

Usage:
    python3 train_causal_rm.py --pairs_dir . --output_dir causal_rm_checkpoint \\
        --epochs 1 --max_pairs 500   # small prototype run first (step 6 of
                                      # the implementation order: verify
                                      # before scaling up)
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from causal_reward_model import (
    RubricHead,
    SERFusionHead,
    format_context_response,
    load_backbone,
    load_ser_model,
    RUBRIC_DIMENSIONS,
)
from grpo_curriculum import build_grpo_examples


V2_ROOT = Path(__file__).parent

# Map a pair's target_factor to the ONE rubric dimension it should train.
# C1/C3/C5 (rule-based diagnostic dimensions) covered from the start; C2/C4
# (price_strategy/progression — 2 of the 3 dimensions that feed `overall`)
# added 2026-08-01 after a lambda_inv sweep + head-correlation analysis
# showed those two heads were underdetermined without any causal anchor,
# receiving only invariance-loss gradient and never causal-ranking
# gradient. C6 (zone-boundary leniency) still not generated — it's a
# leniency *modifier* on C1/C2 rather than an independent ranking signal,
# lower priority. Any target_factor not in this map is skipped (via
# causal_pair_loss's `dim is None` check), not an error — this dict is the
# single source of truth for which causal factors currently have training
# supervision; causal_rubric_taxonomy.md section 4 tracks the same coverage
# status for the paper-facing narrative.
FACTOR_TO_DIMENSION = {
    "C1_decision_zone": "decision_score",
    "C2_price_strategy": "price_strategy",
    "C3_emotion": "emotion",
    "C4_progression": "progression",
    "C5_flip": "flip_score",
}


def build_context_index(split: str) -> Dict[Tuple[str, int], dict]:
    """(dialogue_id, turn_index) -> full example dict, for reconstructing
    spurious-pair context (see module docstring)."""
    examples = build_grpo_examples(split=split, include_audio=False)
    return {(ex["dialogue_id"], ex["turn_index"]): ex for ex in examples}


def load_pairs(pairs_dir: Path, split: str) -> List[dict]:
    data = json.loads((pairs_dir / f"causal_rm_pairs_{split}.json").read_text())
    return data["pairs"]


class Embedder:
    """Wraps the frozen backbone's pooled-hidden-state extraction so
    train/eval loops and the smoke-test path share one code path. Backbone
    forward always runs under no_grad — we only ever need it as a fixed
    feature extractor; gradients flow into the RubricHead, not the backbone.

    Caches by exact formatted text (reviewer-recommended 2026-07-20): the
    backbone is frozen and deterministic, so the same (context, response)
    text always produces the same hidden state. With ~70k pairs sharing a
    finite donor-context pool and a finite set of gt_responses, many exact
    texts repeat across pairs (e.g. the same donor context reused across
    many causal pairs, the same shared_response reused across a turn's C1/C3
    interventions) — caching avoids redundant 8B forward passes for those
    repeats. Cache is unbounded for now (fine at prototype scale); revisit
    with an LRU cap if it grows unwieldy at full-dataset training scale."""

    def __init__(self, backbone, tokenizer, device: str):
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.device = device
        self._cache: Dict[str, torch.Tensor] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    @torch.no_grad()
    def __call__(self, text: str) -> torch.Tensor:
        cached = self._cache.get(text)
        if cached is not None:
            self.cache_hits += 1
            return cached
        self.cache_misses += 1
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
        out = self.backbone(**inputs, output_hidden_states=True)
        # Backbone runs in bf16; RubricHead is fp32 (see matching comment in
        # causal_reward_model.py::pooled_hidden) — cast here too so training
        # and inference paths agree on dtype.
        hidden = out.hidden_states[-1][:, -1, :].float().clone()  # (1, hidden_dim)
        self._cache[text] = hidden
        return hidden


class SEREmbedder:
    """v2 (causal_rm_architecture.md section 9): wraps the frozen SER
    model's (arousal, dominance, valence) extraction for SERFusionHead's
    fusion trunk, mirroring Embedder's caching pattern above.

    Cache key granularity (reviewer-recommended 2026-08-02, decided
    deliberately, not just copied from Embedder): keys by wav_path alone.
    This is safe ONLY because ALL preprocessing (load_wav's resample-to-
    16kHz-mono, then the SER processor's own normalization) happens INSIDE
    this method, after the cache lookup, and nowhere else in the codebase
    computes a SER embedding by any other path. That single-sourcing is
    the actual safety property, not the wav_path string itself — if a
    second call site ever preprocessed the same file differently (e.g. a
    different resample method) before reaching this cache, identical
    wav_paths could silently share a cache entry that should have been
    distinct. Keep all SER preprocessing routed through this class."""

    def __init__(self, ser_model, ser_processor, device: str):
        self.ser_model = ser_model
        self.ser_processor = ser_processor
        self.device = device
        self._cache: Dict[str, torch.Tensor] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    @torch.no_grad()
    def __call__(self, wav_path: str) -> torch.Tensor:
        cached = self._cache.get(wav_path)
        if cached is not None:
            self.cache_hits += 1
            return cached
        self.cache_misses += 1
        from sft_data_v6 import load_wav
        from causal_reward_model import SER_SAMPLE_RATE

        signal = load_wav(Path(wav_path))
        if signal is None:
            raise FileNotFoundError(f"SEREmbedder could not load audio: {wav_path}")
        processed = self.ser_processor(signal, sampling_rate=SER_SAMPLE_RATE)
        input_values = torch.from_numpy(processed["input_values"][0]).reshape(1, -1).to(self.device)
        _hidden, logits = self.ser_model(input_values)
        vector = logits.squeeze(0).float().clone()  # (SER_DIM,) = (arousal, dominance, valence)
        self._cache[wav_path] = vector
        return vector


def embed_pair_side(embedder: Embedder, context: dict, response: str) -> torch.Tensor:
    return embedder(format_context_response(context, response))


def score_head(head, hidden: torch.Tensor, ser_vector: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """Dispatch on head type so causal_pair_loss/spurious_pair_loss/reg_loss
    don't need to know whether they're training v1's RubricHead or v2's
    SERFusionHead. RubricHead.forward() takes no ser_vector arg (v1 has no
    audio path); SERFusionHead.forward(ser_vector=None) reproduces v1's own
    RubricHead output exactly per the dry-run verification in
    causal_reward_model.py, so passing ser_vector=None through here for a
    non-C3 pair is always safe."""
    if isinstance(head, SERFusionHead):
        return head(hidden, ser_vector=ser_vector)
    return head(hidden)


def raw_logits_head(head, hidden: torch.Tensor, ser_vector: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    if isinstance(head, SERFusionHead):
        return head.raw_logits(hidden, ser_vector=ser_vector)
    return head.raw_logits(hidden)


def ser_vector_for_context(context: dict, ser_embedder: Optional["SEREmbedder"]) -> Optional[torch.Tensor]:
    """v2 (causal_rm_architecture.md section 9.6): only C3_emotion pairs'
    context dicts carry buyer_audio_path (see generate_intervention_pairs.py
    ::causal_pair_emotion) — every other pair type simply won't have the
    key, so this returns None for them and SERFusionHead's fallback path
    kicks in automatically. Also None when audio fusion isn't enabled at all
    (ser_embedder is None) or the path is missing (donor/anchor turn had no
    transferred wav)."""
    if ser_embedder is None:
        return None
    path = context.get("buyer_audio_path")
    if not path:
        return None
    return ser_embedder(path).unsqueeze(0)  # (1, SER_DIM) to match hidden's batch dim


def causal_pair_loss(
    pair: dict,
    embedder: Embedder,
    head: RubricHead,
    ser_embedder: Optional["SEREmbedder"] = None,
) -> Tuple[Optional[torch.Tensor], Optional[str], Optional[bool], Optional[Dict[str, float]]]:
    """Returns (loss, dimension, ranked_correctly, full_scores_a) or
    (None, None, None, None) if this pair's target_factor isn't covered by
    FACTOR_TO_DIMENSION. full_scores_a is the complete 5-dimension rubric
    vector for context_a (all heads, not just the targeted one) — used only
    for head_correlation monitoring (causal_rm_architecture.md section 8.3),
    not for the loss itself."""
    dim = FACTOR_TO_DIMENSION.get(pair["target_factor"])
    if dim is None:
        return None, None, None, None

    response = pair["shared_response"]
    hidden_a = embed_pair_side(embedder, pair["context_a"], response)
    hidden_b = embed_pair_side(embedder, pair["context_b"], response)
    ser_a = ser_vector_for_context(pair["context_a"], ser_embedder)
    ser_b = ser_vector_for_context(pair["context_b"], ser_embedder)

    scores_a = score_head(head, hidden_a, ser_a)
    s_a = scores_a[dim]  # (1,)
    s_b = score_head(head, hidden_b, ser_b)[dim]

    sign = 1.0 if pair["expected_relation"] == "a_better" else -1.0
    diff = sign * (s_a - s_b)
    loss = -F.logsigmoid(diff).mean()
    ranked_correctly = bool((diff > 0).item())
    full_scores_a = {k: float(v.item()) for k, v in scores_a.items()}
    return loss, dim, ranked_correctly, full_scores_a


def spurious_pair_loss(
    pair: dict,
    context: dict,
    embedder: Embedder,
    head: RubricHead,
) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, float]]:
    """Returns (loss, per_dim_sq, full_scores_a) — full_scores_a for
    head_correlation monitoring, same rationale as causal_pair_loss above.

    Invariance loss is computed over ALL 5 RUBRIC_DIMENSIONS, not just
    OVERALL_DIMS (fixed 2026-08-01, after the lambda_inv sweep found why:
    with the loss only covering price_strategy/emotion/progression,
    decision_score and flip_score received zero invariance supervision at
    any lambda_inv, so their spurious-pair invariance drifted as an
    uncontrolled side-effect of the shared trunk absorbing more invariance
    pressure for the other three dimensions — decision_score's spurious
    delta rose monotonically from 0.016 to 0.287 as lambda_inv went
    1.0->3.5 in that sweep, purely from collateral pressure, never from its
    own gradient. decision_score/flip_score still do NOT feed into
    `overall` (causal_reward_model.py's OVERALL_DIMS is unchanged, per
    causal_rm_architecture.md section 3's diagnostics-only rationale) —
    this only gives them their own regularization signal so they stop
    drifting as collateral damage while price_strategy/emotion/progression
    are pushed toward invariance."""
    hidden_a = embed_pair_side(embedder, context, pair["response_a"])
    hidden_b = embed_pair_side(embedder, context, pair["response_b"])

    # Spurious pairs share ONE context (only the response differs), so if
    # audio fusion is on and this context happens to carry buyer_audio_path,
    # the same ser_vector is used on both sides — audio isn't the varied
    # attribute here, so this can only help keep the emotion head's
    # invariance check consistent with the causal path, never introduce a
    # false asymmetry. ser_embedder is None for spurious pairs today since
    # context dicts reconstructed via build_context_index() don't carry
    # buyer_audio_path (only causal_pair_emotion's stored contexts do, per
    # generate_intervention_pairs.py) — left as an explicit no-op path
    # rather than silently wired up, so it's obvious this is unimplemented
    # scope, not a bug, if S-pair audio support is added later.
    scores_a = score_head(head, hidden_a)
    scores_b = score_head(head, hidden_b)

    per_dim_sq = {}
    total = torch.zeros(1, device=hidden_a.device)
    for dim in RUBRIC_DIMENSIONS:
        sq = (scores_a[dim] - scores_b[dim]) ** 2
        per_dim_sq[dim] = float(sq.item())
        total = total + sq
    full_scores_a = {k: float(v.item()) for k, v in scores_a.items()}
    return (total / len(RUBRIC_DIMENSIONS)).mean(), per_dim_sq, full_scores_a


def compute_head_correlation(score_buffer: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
    """Mean off-diagonal pairwise Pearson correlation between the 5 heads'
    outputs over a window of recent examples (causal_rm_architecture.md
    section 8.3 monitoring — not yet used as a training penalty, just
    observed). A high value isn't automatically bad: some cross-dimension
    correlation is expected and even desirable (e.g. softening for an upset
    buyer legitimately moves emotion + progression + price_strategy
    together, which is the whole reason for the shared trunk in the first
    place, see section 2). This is a diagnostic to look at, not a target to
    minimize by default."""
    if len(score_buffer) < 3:
        return None
    dims = RUBRIC_DIMENSIONS
    vectors = {d: [s[d] for s in score_buffer] for d in dims}
    correlations = []
    for i, d1 in enumerate(dims):
        for d2 in dims[i + 1:]:
            x, y = vectors[d1], vectors[d2]
            try:
                r = statistics.correlation(x, y)
            except statistics.StatisticsError:
                continue
            correlations.append(r)
    if not correlations:
        return None
    return {"mean_abs_off_diagonal": round(sum(abs(r) for r in correlations) / len(correlations), 4)}


def reg_loss(head: RubricHead, hidden: torch.Tensor) -> torch.Tensor:
    logits = raw_logits_head(head, hidden)
    return sum((v ** 2).mean() for v in logits.values()) / len(logits)


def train(
    pairs_dir: Path,
    output_dir: Path,
    backbone_model_id: str,
    device: str,
    epochs: int,
    lr: float,
    lambda_bt: float,
    lambda_inv: float,
    lambda_reg: float,
    max_pairs: int,
    log_every: int,
    seed: int,
    embedder_override: Optional[Embedder] = None,
    hidden_dim_override: Optional[int] = None,
    audio_fusion: bool = False,
    ser_device: Optional[str] = None,
    ser_embedder_override: Optional["SEREmbedder"] = None,
) -> Path:
    rng = random.Random(seed)
    # --seed previously only controlled pair-shuffle order via the
    # random.Random instance above; RubricHead/SERFusionHead weight init
    # goes through torch's own (unseeded) global RNG, so two "same-seed"
    # runs actually started from two different random initializations —
    # discovered 2026-08-16 when a same-scale v1-vs-v2 comparison run
    # showed spurious-invariance deltas differing by 2-20x on dimensions
    # (decision_score, progression, flip_score) that audio_fusion cannot
    # possibly touch, which is only explainable by uncontrolled init
    # variance at this small a step count. Seed both RNGs so "--seed 42
    # twice" is actually a controlled comparison, not just the same data
    # order fed to two different random heads.
    torch.manual_seed(seed)

    if embedder_override is not None:
        embedder = embedder_override
        hidden_dim = hidden_dim_override
    else:
        backbone, tokenizer = load_backbone(backbone_model_id, device=device)
        embedder = Embedder(backbone, tokenizer, device)
        hidden_dim = backbone.config.hidden_size

    # Re-seed immediately before head construction (not just once at the top
    # of train()): load_backbone()/load_ser_model() consume torch's global
    # RNG differently depending on audio_fusion (the SER model load is an
    # extra RNG-consuming step only on the audio_fusion=True path), so
    # without re-seeding here the head's actual initial weights would still
    # diverge between "same-seed" audio_fusion=True/False runs even with the
    # top-of-function seed above.
    torch.manual_seed(seed)
    ser_embedder: Optional[SEREmbedder] = None
    if audio_fusion:
        # v2 (causal_rm_architecture.md section 9): SERFusionHead wraps an
        # unmodified RubricHead as .base, so warm-starting from a v1
        # checkpoint (load_state_dict into head.base, strict=True) stays
        # available even though this run constructs a fresh head — that's
        # a separate --init_from_v1_checkpoint concern, not needed for a
        # from-scratch prototype run.
        head = SERFusionHead(hidden_dim=hidden_dim).to(device)
        if ser_embedder_override is not None:
            ser_embedder = ser_embedder_override
        else:
            ser_model, ser_processor = load_ser_model(device=ser_device or device)
            ser_embedder = SEREmbedder(ser_model, ser_processor, ser_device or device)
    else:
        head = RubricHead(hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)

    train_pairs = load_pairs(pairs_dir, "train")
    rng.shuffle(train_pairs)
    if max_pairs:
        train_pairs = train_pairs[:max_pairs]

    context_index = build_context_index("train")

    causal_pairs = [p for p in train_pairs if p["pair_family"] == "causal"]
    spurious_pairs = [p for p in train_pairs if p["pair_family"] == "spurious"]

    history: List[dict] = []
    step = 0

    for epoch in range(epochs):
        rng.shuffle(causal_pairs)
        rng.shuffle(spurious_pairs)
        n_steps = max(len(causal_pairs), len(spurious_pairs))

        bt_correct: Dict[str, int] = {}
        bt_total: Dict[str, int] = {}
        inv_sq: Dict[str, List[float]] = {dim: [] for dim in RUBRIC_DIMENSIONS}
        score_buffer: List[Dict[str, float]] = []  # for head_correlation (section 8.3)

        for i in range(n_steps):
            optimizer.zero_grad()
            total_loss = torch.zeros(1, device=device)
            reg_context: Optional[dict] = None
            reg_response: Optional[str] = None

            if causal_pairs:
                c_pair = causal_pairs[i % len(causal_pairs)]
                loss_c, dim, correct, full_scores = causal_pair_loss(c_pair, embedder, head, ser_embedder)
                if loss_c is not None:
                    total_loss = total_loss + lambda_bt * loss_c
                    bt_total[dim] = bt_total.get(dim, 0) + 1
                    bt_correct[dim] = bt_correct.get(dim, 0) + int(correct)
                    score_buffer.append(full_scores)
                reg_context, reg_response = c_pair["context_a"], c_pair["shared_response"]

            if spurious_pairs:
                s_pair = spurious_pairs[i % len(spurious_pairs)]
                key = (s_pair["dialogue_id"], s_pair["turn_index"])
                s_context = context_index.get(key, {})
                loss_s, per_dim_sq, full_scores = spurious_pair_loss(s_pair, s_context, embedder, head)
                total_loss = total_loss + lambda_inv * loss_s
                for dim, sq in per_dim_sq.items():
                    inv_sq[dim].append(sq)
                score_buffer.append(full_scores)
                if reg_context is None:
                    reg_context, reg_response = s_context, s_pair["response_a"]

            # Regularizer computed on whichever pair side we already have
            # context/response for this step (cheap, no extra backbone
            # forward needed beyond what causal/spurious losses used).
            if reg_context is not None:
                total_loss = total_loss + lambda_reg * reg_loss(head, embed_pair_side(embedder, reg_context, reg_response))

            total_loss.backward()
            optimizer.step()
            step += 1

            if step % log_every == 0:
                metrics = {
                    "epoch": epoch,
                    "step": step,
                    "loss": float(total_loss.item()),
                    "bt_accuracy_per_dimension": {
                        d: round(bt_correct.get(d, 0) / max(1, bt_total.get(d, 0)), 4) for d in bt_total
                    },
                    "invariance_mse_per_dimension": {
                        d: round(statistics.mean(v), 6) if v else None for d, v in inv_sq.items()
                    },
                    "head_correlation": compute_head_correlation(score_buffer),
                }
                if hasattr(embedder, "cache_hits"):
                    metrics["embedder_cache_hit_rate"] = round(
                        embedder.cache_hits / max(1, embedder.cache_hits + embedder.cache_misses), 4
                    )
                if ser_embedder is not None:
                    metrics["ser_embedder_cache_hit_rate"] = round(
                        ser_embedder.cache_hits / max(1, ser_embedder.cache_hits + ser_embedder.cache_misses), 4
                    )
                history.append(metrics)
                print(json.dumps(metrics))
                score_buffer.clear()  # reset the correlation window each log interval

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), output_dir / "rubric_head.pt")
    (output_dir / "config.json").write_text(json.dumps({
        "hidden_dim": hidden_dim,
        "backbone_model_id": backbone_model_id,
        "rubric_dimensions": RUBRIC_DIMENSIONS,
        "audio_fusion": audio_fusion,
    }, indent=2))
    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    print(f"Saved checkpoint to {output_dir}")
    return output_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs_dir", default=str(V2_ROOT))
    p.add_argument("--output_dir", default=str(V2_ROOT / "causal_rm_checkpoint"))
    p.add_argument("--backbone_model_id", default="Qwen/Qwen3-8B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lambda_bt", type=float, default=1.0)
    p.add_argument("--lambda_inv", type=float, default=1.0)
    p.add_argument("--lambda_reg", type=float, default=1e-3)
    p.add_argument("--max_pairs", type=int, default=0, help="0 = use all pairs. Set small (e.g. 200-500) for a first prototype run per the recommended implementation order.")
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--audio_fusion", action="store_true", help="v2 (causal_rm_architecture.md section 9): train SERFusionHead instead of RubricHead, feeding C3_emotion pairs' buyer_audio_path through the frozen audeering SER model. Default off, reproduces v1 exactly.")
    p.add_argument("--ser_device", default=None, help="Device for the frozen SER model; defaults to --device if unset.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train(
        pairs_dir=Path(args.pairs_dir),
        output_dir=Path(args.output_dir),
        backbone_model_id=args.backbone_model_id,
        device=args.device,
        epochs=args.epochs,
        lr=args.lr,
        lambda_bt=args.lambda_bt,
        lambda_inv=args.lambda_inv,
        lambda_reg=args.lambda_reg,
        max_pairs=args.max_pairs,
        log_every=args.log_every,
        seed=args.seed,
        audio_fusion=args.audio_fusion,
        ser_device=args.ser_device,
    )


if __name__ == "__main__":
    main()
