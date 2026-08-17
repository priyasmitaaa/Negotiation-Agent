#!/usr/bin/env python3
"""
causal_rm_audit.py — measure causal-factor sensitivity and spurious
invariance for a reward source, against causal_rm_pairs_{split}.json.

Works in two modes, independently:

  --backend heuristic   Uses the existing reward_function.py functions
                         (decision_reward, price_strategy_reward,
                         emotion_reward, progression_reward, flip_reward) —
                         no GPU, no trained RM needed. This is the "before"
                         baseline: how hackable/insensitive is the current
                         keyword-heuristic reward on our own intervention
                         pairs?

  --backend causal_rm    Loads a trained causal_reward_model.py checkpoint
                          (via --checkpoint) and scores the same pairs with
                          its rubric vector. This is the "after" — run once
                          implementation step 6's real-backbone training run
                          exists (causal_rm_architecture.md section 7).

Both backends report the same two things per rubric dimension, so the two
runs are directly comparable in one table:

  - causal ranking accuracy: on causal pairs, did the dimension's score for
    context_a exceed context_b (or vice versa) in the direction
    `expected_relation` implies? (chance = 50%)
  - spurious invariance: on spurious pairs, mean |score(response_a) -
    score(response_b)| holding context fixed — lower is better, and this is
    the number the paper's "our RM is less reward-hackable than the
    baseline" claim rests on (causal_rm_architecture.md section 7).

Usage:
    # GPU-free baseline, run today:
    python3 causal_rm_audit.py --backend heuristic --split val

    # After a causal RM checkpoint exists:
    python3 causal_rm_audit.py --backend causal_rm --split val \\
        --checkpoint causal_rm_checkpoint/
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from reward_function import (
    decision_reward,
    price_strategy_reward,
    emotion_reward,
    progression_reward,
    flip_reward,
)
from train_causal_rm import FACTOR_TO_DIMENSION, build_context_index
from causal_reward_model import RUBRIC_DIMENSIONS


V2_ROOT = Path(__file__).parent

ScorerFn = Callable[[str, dict, str, Optional[str]], Optional[float]]


def heuristic_dimension_score(
    dimension: str, context: dict, response: str, pinned_decision: Optional[str]
) -> Optional[float]:
    """Compute ONE rubric dimension's score using the existing heuristic
    reward_function.py functions, given a context dict and a response.
    `pinned_decision` is held fixed across a pair's two sides (see module
    docstring's "before" rationale in causal_rm_audit's causal-pair audit
    below) so that what varies is only the CONTEXT (for causal pairs) or
    only the RESPONSE (for spurious pairs), never both at once — otherwise
    a reward difference couldn't be attributed to the one thing we
    intervened on."""
    if dimension == "decision_score":
        gt = context.get("gt_decision")
        if gt is None:
            return None
        return decision_reward(pinned_decision, gt)

    if dimension == "flip_score":
        if "decision_flip" not in context and "gt_decision" not in context:
            return None
        return flip_reward(context, pinned_decision)

    if dimension == "price_strategy":
        score, _ = price_strategy_reward(context, pinned_decision, response)
        return score

    if dimension == "emotion":
        em = context.get("emotion") or context.get("prior_buyer_emotion")
        ex = {"prior_buyer_emotion": em}
        score, _ = emotion_reward(ex, response)
        return score

    if dimension == "progression":
        # NOTE (audit finding worth keeping for the paper): progression_reward()
        # takes only `response`, not `context` — it is CONTEXT-INDEPENDENT by
        # construction in the old heuristic. This means the old heuristic
        # cannot possibly be causally sensitive to C4 (progression as a
        # function of negotiation state) no matter what we intervene on;
        # any "sensitivity" measured here is a no-op / always-tied result,
        # which is itself evidence for the reward-hacking-motivation claim.
        score, _ = progression_reward(response)
        return score

    return None


def causal_rm_dimension_score_factory(checkpoint_dir: str, device: str = "cuda") -> ScorerFn:
    """Builds the causal_rm-backed ScorerFn. Caches rm.score()'s full
    5-dimension result by formatted (context, response) text, because
    audit_spurious_pairs() calls this scorer once PER RUBRIC DIMENSION for
    the same (context, response) pair — without caching that's 5 redundant
    backbone forward passes per pair-side for what is, underneath, one
    rm.score() call. This mirrors train_causal_rm.py::Embedder's caching
    rationale, just at the score-dict level instead of the hidden-state
    level (simpler here since causal_rm_audit.py doesn't need gradients)."""
    from causal_reward_model import load_causal_rm, format_context_response  # deferred: heavy import, only when needed

    rm = load_causal_rm(checkpoint_dir, device=device)
    cache: Dict[str, Dict[str, float]] = {}

    def _score(dimension: str, context: dict, response: str, pinned_decision: Optional[str]) -> Optional[float]:
        # v2 (causal_rm_architecture.md section 9.6): format_context_response
        # never renders buyer_audio_path into text, so two contexts that
        # differ ONLY in which donor wav they point to (same emotion label
        # text, different underlying audio) would otherwise collide on this
        # cache key and silently reuse the wrong SER-influenced score. Fold
        # the path into the key explicitly rather than changing the prompt
        # text itself (which must stay audio-path-free by design).
        key = format_context_response(context, response) + "|" + str(context.get("buyer_audio_path") or "")
        components = cache.get(key)
        if components is None:
            _, components = rm.score(context, response)
            cache[key] = components
        return components.get(dimension)

    return _score


def load_pairs(pairs_dir: Path, split: str) -> List[dict]:
    data = json.loads((pairs_dir / f"causal_rm_pairs_{split}.json").read_text())
    return data["pairs"]


def audit_causal_pairs(
    pairs: List[dict], scorer: ScorerFn, collect_raw: bool = False
) -> Tuple[Dict[str, Any], Dict[str, List[float]]]:
    """Returns (summary, raw_scores). raw_scores[dim] holds every s_a value
    seen for that dimension when collect_raw=True (empty dict otherwise) —
    used only for the score-distribution plots in generate_report(), not
    for the aggregate numbers themselves."""
    correct: Dict[str, int] = {}
    total: Dict[str, int] = {}
    skipped = 0
    raw: Dict[str, List[float]] = {}

    for pair in pairs:
        if pair["pair_family"] != "causal":
            continue
        dim = FACTOR_TO_DIMENSION.get(pair["target_factor"])
        if dim is None:
            skipped += 1
            continue

        response = pair["shared_response"]
        pinned_decision = pair["context_a"].get("gt_decision")

        s_a = scorer(dim, pair["context_a"], response, pinned_decision)
        s_b = scorer(dim, pair["context_b"], response, pinned_decision)
        if s_a is None or s_b is None:
            skipped += 1
            continue

        sign = 1.0 if pair["expected_relation"] == "a_better" else -1.0
        ranked_correctly = sign * (s_a - s_b) > 0
        total[dim] = total.get(dim, 0) + 1
        correct[dim] = correct.get(dim, 0) + int(ranked_correctly)
        if collect_raw:
            raw.setdefault(dim, []).extend([s_a, s_b])

    summary = {
        "ranking_accuracy_per_dimension": {
            d: round(correct.get(d, 0) / total[d], 4) for d in total
        },
        "n_pairs_per_dimension": total,
        "n_skipped": skipped,
    }
    return summary, raw


def audit_spurious_pairs(
    pairs: List[dict], scorer: ScorerFn, context_index: Dict[Tuple[str, int], dict], collect_raw: bool = False
) -> Tuple[Dict[str, Any], Dict[str, List[float]]]:
    """Returns (summary, raw_deltas). raw_deltas[dim] holds every |s_a - s_b|
    value for the invariance-distribution plot."""
    deltas: Dict[str, List[float]] = {dim: [] for dim in RUBRIC_DIMENSIONS}
    n_skipped = 0

    for pair in pairs:
        if pair["pair_family"] != "spurious":
            continue
        key = (pair["dialogue_id"], pair["turn_index"])
        context = context_index.get(key)
        if context is None:
            n_skipped += 1
            continue
        pinned_decision = context.get("gt_decision")

        for dim in RUBRIC_DIMENSIONS:
            s_a = scorer(dim, context, pair["response_a"], pinned_decision)
            s_b = scorer(dim, context, pair["response_b"], pinned_decision)
            if s_a is None or s_b is None:
                continue
            deltas[dim].append(abs(s_a - s_b))

    summary = {
        "mean_abs_delta_per_dimension": {
            d: (round(statistics.mean(v), 6) if v else None) for d, v in deltas.items()
        },
        "n_pairs_scored_per_dimension": {d: len(v) for d, v in deltas.items()},
        "n_skipped": n_skipped,
    }
    return summary, (deltas if collect_raw else {})


def run_audit(
    pairs_dir: Path,
    split: str,
    backend: str,
    checkpoint: Optional[str],
    device: str,
    collect_raw: bool = False,
    max_pairs: int = 0,
    seed: int = 42,
) -> Tuple[dict, Dict[str, List[float]], Dict[str, List[float]]]:
    """Returns (result, causal_raw_scores, spurious_raw_deltas). The raw
    dicts are empty unless collect_raw=True (only needed for
    generate_report's plots — the plain JSON audit doesn't pay for it).

    `max_pairs`: 0 = audit the full split (use for the real go/no-go
    check). For cheap iteration (e.g. a lambda_inv sweep), set this to a
    few hundred — random subsample (fixed seed, naturally preserves the
    causal:spurious ratio since both families are drawn from one shuffled
    list) trades statistical precision for speed. Not a substitute for a
    full-split audit before actually deciding to proceed to GRPO."""
    pairs = load_pairs(pairs_dir, split)
    if max_pairs and max_pairs < len(pairs):
        pairs = random.Random(seed).sample(pairs, max_pairs)
    context_index = build_context_index(split)

    if backend == "heuristic":
        scorer: ScorerFn = heuristic_dimension_score
    elif backend == "causal_rm":
        if not checkpoint:
            raise ValueError("--checkpoint is required for --backend causal_rm")
        scorer = causal_rm_dimension_score_factory(checkpoint, device=device)
    else:
        raise ValueError(f"unknown backend: {backend}")

    causal_summary, causal_raw = audit_causal_pairs(pairs, scorer, collect_raw=collect_raw)
    spurious_summary, spurious_raw = audit_spurious_pairs(pairs, scorer, context_index, collect_raw=collect_raw)

    result = {
        "backend": backend,
        "split": split,
        "checkpoint": checkpoint,
        "causal_pairs": causal_summary,
        "spurious_pairs": spurious_summary,
    }
    return result, causal_raw, spurious_raw


def generate_report(
    result: dict,
    causal_raw: Dict[str, List[float]],
    spurious_raw: Dict[str, List[float]],
    output_dir: Path,
) -> Path:
    """Write plots (ranking accuracy bar chart, invariance bar chart, and
    per-dimension score/delta distribution histograms) plus a markdown
    summary table to output_dir. Requires matplotlib — deferred import so
    the plain JSON audit path (run_audit alone) never needs it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{result['backend']}_{result['split']}"

    # --- Bar chart: causal ranking accuracy per dimension ---
    acc = result["causal_pairs"]["ranking_accuracy_per_dimension"]
    if acc:
        fig, ax = plt.subplots(figsize=(6, 4))
        dims = list(acc.keys())
        ax.bar(dims, [acc[d] for d in dims], color="#4C72B0")
        ax.axhline(0.5, color="gray", linestyle="--", label="chance (50%)")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Ranking accuracy")
        ax.set_title(f"Causal-pair ranking accuracy ({tag})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"causal_ranking_accuracy_{tag}.png", dpi=150)
        plt.close(fig)

    # --- Bar chart: spurious invariance (mean abs delta) per dimension ---
    inv = result["spurious_pairs"]["mean_abs_delta_per_dimension"]
    if inv:
        fig, ax = plt.subplots(figsize=(6, 4))
        dims = [d for d in inv if inv[d] is not None]
        ax.bar(dims, [inv[d] for d in dims], color="#DD8452")
        ax.set_ylabel("Mean |Δ score| (lower = more invariant)")
        ax.set_title(f"Spurious-pair invariance ({tag})")
        fig.tight_layout()
        fig.savefig(output_dir / f"spurious_invariance_{tag}.png", dpi=150)
        plt.close(fig)

    # --- Histograms: score distributions per dimension ---
    for source_name, raw, xlabel in [
        ("causal_scores", causal_raw, "raw dimension score"),
        ("spurious_deltas", spurious_raw, "|Δ score| between spurious pair sides"),
    ]:
        dims_with_data = [d for d, v in raw.items() if v]
        if not dims_with_data:
            continue
        fig, axes = plt.subplots(1, len(dims_with_data), figsize=(4 * len(dims_with_data), 3.5), squeeze=False)
        for ax, dim in zip(axes[0], dims_with_data):
            ax.hist(raw[dim], bins=20, color="#55A868")
            ax.set_title(dim)
            ax.set_xlabel(xlabel)
        fig.suptitle(f"{source_name} distribution ({tag})")
        fig.tight_layout()
        fig.savefig(output_dir / f"{source_name}_distribution_{tag}.png", dpi=150)
        plt.close(fig)

    # --- Markdown summary table ---
    lines = [
        f"# Causal RM Audit Report — {tag}",
        "",
        f"Backend: `{result['backend']}`  |  Split: `{result['split']}`  |  Checkpoint: `{result['checkpoint'] or 'N/A (heuristic)'}`",
        "",
        "## Causal-pair ranking accuracy (chance = 50%)",
        "",
        "| Dimension | Accuracy | N pairs |",
        "|---|---:|---:|",
    ]
    n_per_dim = result["causal_pairs"]["n_pairs_per_dimension"]
    for d, a in acc.items():
        lines.append(f"| {d} | {a:.4f} | {n_per_dim.get(d, 0)} |")
    lines += [
        "",
        "## Spurious-pair invariance (lower = better)",
        "",
        "| Dimension | Mean \\|Δ\\| | N pairs |",
        "|---|---:|---:|",
    ]
    n_scored = result["spurious_pairs"]["n_pairs_scored_per_dimension"]
    for d, v in inv.items():
        v_str = f"{v:.6f}" if v is not None else "N/A"
        lines.append(f"| {d} | {v_str} | {n_scored.get(d, 0)} |")
    lines += [
        "",
        f"![causal ranking accuracy](causal_ranking_accuracy_{tag}.png)",
        f"![spurious invariance](spurious_invariance_{tag}.png)",
    ]

    report_path = output_dir / f"report_{tag}.md"
    report_path.write_text("\n".join(lines))
    return report_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs_dir", default=str(V2_ROOT))
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--backend", default="heuristic", choices=["heuristic", "causal_rm"])
    p.add_argument("--checkpoint", default=None, help="Required for --backend causal_rm")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output_file", default=None)
    p.add_argument(
        "--report_dir",
        default=None,
        help="If set, also generate plots (ranking accuracy, invariance, score distributions) and a markdown summary into this directory. Requires matplotlib.",
    )
    p.add_argument(
        "--max_pairs",
        type=int,
        default=0,
        help="0 = audit the full split. Set to a few hundred for cheap iteration (e.g. a lambda sweep) — not a substitute for a full-split audit before the real go/no-go decision.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result, causal_raw, spurious_raw = run_audit(
        pairs_dir=Path(args.pairs_dir),
        split=args.split,
        backend=args.backend,
        checkpoint=args.checkpoint,
        device=args.device,
        collect_raw=bool(args.report_dir),
        max_pairs=args.max_pairs,
    )

    print(json.dumps(result, indent=2))

    out_path = Path(args.output_file) if args.output_file else Path(args.pairs_dir) / f"causal_rm_audit_{args.backend}_{args.split}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved audit to {out_path}")

    if args.report_dir:
        report_path = generate_report(result, causal_raw, spurious_raw, Path(args.report_dir))
        print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
