# Empirical Curriculum for GRPO — Design Proposal

**Status:** design proposal, not yet implemented. Recorded 2026-07-20.
**Relationship to the causal-factor RM work:** this is a **parallel
workstream**, not a dependency of `causal_reward_model.py`/`train_causal_rm.py`.
The RM prototype (see `causal_rm_architecture.md`) should proceed on its own
timeline; this doc captures a separate, data-driven replacement for
`grpo_curriculum.py`'s current hand-authored `difficulty_for_example()`
scoring (fixed point weights on a handful of boolean conditions) with an
empirically validated difficulty score.

## Executive Summary

Define a data-driven curriculum for GRPO by scoring each training example on
multiple difficulty signals (model confidence, SFT loss, price gap, emotion
intensity, decision flip, number of prior offers, etc.), normalizing them,
optionally combining into one score, and validating the result empirically
— rather than assuming the current hand-picked weights in
`grpo_curriculum.py::difficulty_for_example()` are correct. Validation has
two legs: (1) correlate each signal with held-out SFT model performance
(loss/accuracy), and (2) run pilot GRPO training comparing an easy→hard
curriculum against random sampling, measuring learning curves and final
accuracy.

## Candidate Difficulty Signals

| Signal | Fields & computation | Pros | Cons | Use |
|---|---|---|---|---|
| Model confidence | SFT model's predicted probability for the correct decision: `conf = model_prob(gt_decision)` | Directly correlates with ease; high confidence ⇒ clear case | Needs a model pass; may reward shallow/generic answers | Primary ranking signal (high ⇒ easy) |
| Training loss | Cross-entropy loss of the SFT model on the example: `loss = -log(conf)` | Well-studied in self-paced curriculum learning; auto-adapts to the model's own learning state | Scale-sensitive; can overweight outliers/noisy labels | Secondary; validates the confidence signal |
| Price gap % | `gap = |last_buyer_offer - fair_value| / fair_value * 100` (from `buyer_offers_so_far[-1]` and `pricing.fair_value`) | Encodes closeness to agreement — small gap (near settlement) is easier, large gap is harder | Early-turn small gaps aren't always easy (anchoring can still be live) | Weak monotonic signal |
| Decision flip | `flip = factor_state.decision_is_flip` | Rare and strategically important — likely harder; already used in `grpo_curriculum.py::difficulty_for_example` | Low frequency limits how much separation it alone provides | Mark hard when true; keep in weighted sum |
| # prior offers | `n = len(buyer_offers_so_far)` before the current turn, normalized e.g. `min(n/10, 1)` | Captures negotiation length/complexity | Early turns (small n) aren't always easy — first offers can set the anchor | Mild positive weight |
| Emotion intensity | `emotion.intensity` (low/medium/high) | High intensity, especially negative, plausibly harder | Emotion labels are LLM-derived (Gemini) and may be noisy; calm hostility exists | Weight higher intensity as harder |
| Emotion valence | `emotion.valence` (negative/neutral/positive), mapped to `(valence + 1) / 2` for a [0,1] scale | Negative valence generally signals conflict (harder), positive signals cooperation (easier) | Coarse; neutral is ambiguous | Combine with intensity: high-intensity negative = hardest |
| Zone stage | `zone == "pre_crystallisation"` vs `"post_crystallisation"` | Post-crystallisation means an agreement region has been found (easier on average); pre-zone is more open-ended | Structural, not purely a difficulty signal — already used as a difficulty component in `grpo_curriculum.py` | Treat post-zone as easier on average |
| Content complexity (optional) | Response length, rare-token count, syntactic complexity | May correlate with negotiation complexity | Confounds style with difficulty — same S1 spurious-length risk flagged in `causal_rubric_taxonomy.md`; must be checked for invariance, not trusted blindly | Use cautiously, verify it isn't just rewarding verbosity |
| Audio quality (optional, speech-mode only) | SNR / clarity metrics on the buyer wav | Poor audio makes the task genuinely harder | Out of scope for text-mode curriculum; risk of overfitting to synthetic-TTS artifacts specific to `tts_outputs/` rather than real difficulty | Speech-mode only; verify invariance |

The strongest signals are model-based (confidence/loss) and
negotiation-specific (price gap, flips, emotion). Signals like raw response
length or generic politeness are spurious to difficulty scoring for the same
reason they're spurious to reward scoring (see `causal_rubric_taxonomy.md`
S1/S2) and should not be allowed to drive curriculum difficulty either.

## Combining Signals Into a Difficulty Score

No single combination method is obviously best — plan to try more than one
and pick empirically (see Validation section):

1. **Weighted sum / logistic**: `score = sum(w_i * x_i)`, weights hand-set initially, then tuned against validation correlation with SFT loss.
2. **PCA**: first principal component of the normalized signal matrix as a latent difficulty axis — implicitly weights correlated signals, reduces noise.
3. **Learned difficulty regressor**: train a small regressor (or ranking model) predicting SFT loss/error from the signals; use its prediction as the difficulty score. Auto-weighs by predictive power.
4. **Quantile aggregation**: convert each signal to a percentile rank, average the percentiles — avoids scale/unit issues, naturally bounded.

## Clustering / Thresholding Into Buckets

- **Quantile thresholds**: e.g. bottom 20% easy, middle 60% medium, top 20% hard — balanced by construction, fractions tunable.
- **K-means / GMM (k=3)**: cluster on the difficulty signal(s), map clusters to easy/medium/hard — adapts to the data's actual distribution rather than assuming even splits.
- **Hierarchical clustering**: cut a dendrogram into 3 clusters if the data shows clear modes.
- **Direct thresholds on confidence/loss**: e.g. `confidence > 0.9` → easy, `< 0.5` → hard, calibrated on a validation split.

Before clustering, normalize/scale each signal (e.g. [0,1] or z-score) so no
single signal dominates distance calculations by virtue of its raw units.

```python
scores = compute_difficulty_scores(examples)  # one score per example
f_easy, f_hard = 0.2, 0.2
n = len(scores)
i_easy, i_hard = int(f_easy * n), int((1 - f_hard) * n)
sorted_ids = argsort(scores)  # ascending: low = easy, high = hard
easy_ids, med_ids, hard_ids = sorted_ids[:i_easy], sorted_ids[i_easy:i_hard], sorted_ids[i_hard:]
```

## Validation and Diagnostics

1. **Correlation with model performance**: Pearson/Spearman correlation between each signal (and the combined score) and SFT loss/error on held-out examples. Expect positive correlation (harder buckets → higher loss). Signals that don't correlate (e.g. spurious length) should be downweighted or dropped.
2. **Held-out performance gap**: SFT accuracy/F1 separately on easy/medium/hard buckets — expect a monotonic (or close to it) trend; large overlap between buckets signals poor separation.
3. **Ablation pilots (curriculum vs. random)**: small-scale GRPO runs (reusing `train_grpo_curriculum.py`, same pattern as the existing `grpo_pilot_*`/`grpo_smoke_*` runs) comparing an easy→hard curriculum schedule against randomized sampling, measuring learning curves. At least 5 independent runs per condition for a meaningful paired significance test (t-test or Wilcoxon signed-rank on final accuracy).
4. **Error slices**: after training, compare error rates on hard examples between curriculum-trained and randomly-trained models — checks whether curriculum actually helps the genuinely hard cases, not just aggregate accuracy.
5. **Correlation heatmap**: pairwise correlation between all signals and against loss/confidence — flags redundant signals (e.g. intensity/valence co-varying) and confirms spurious signals (response length) stay uncorrelated with the validated difficulty score.
6. **Visualization**: 2D PCA/UMAP of examples colored by difficulty bucket, to check easy/hard occupy visually distinct regions.

## Experimental Protocol

- **Splits**: use the existing `splits_v6.json` — compute difficulty signals on train, reserve val for threshold tuning, keep test untouched for final evaluation (no curriculum-design leakage into the held-out test set).
- **Pilot training**: short GRPO runs (10-20 episodes) per curriculum condition, identical hyperparameters except sampling order, metrics logged every N steps.
- **Metrics**: decision accuracy, negotiated-price-vs-fair, plus the domain-specific measures already in `error_analysis.py`.
- **Statistical tests**: paired t-test / Wilcoxon on final accuracy or learning-curve AUC between curriculum and random conditions.

## Relationship to Existing Code

- `grpo_curriculum.py::difficulty_for_example()` already implements a
  *hand-weighted* version of several of these signals (`post_crystallisation`,
  `mitigate_decision`, emotion intensity/valence, price-boundary proximity,
  `decision_flip`, and an optional baseline-error bonus) with fixed point
  weights (+1/+1/+1/+1/+1/+2/+2) and fixed thresholds (`hard` if score≥4 or
  flip; `medium` if score≥2). This proposal replaces those fixed weights/
  thresholds with empirically validated ones, using the same underlying
  fields — it's an evolution of the existing function, not a replacement of
  its signal set.
- `train_grpo_curriculum.py` already has the `CurriculumCallback`/
  `NegotiationGRPODataset.set_progress()` staging mechanism this proposal's
  pilot ablation would reuse directly — no new training-loop plumbing needed,
  only a new difficulty-scoring function feeding the existing bucket
  assignment.

## Next Steps Checklist

1. Compute difficulty signals using `grpo_curriculum.py`'s example loader (`build_grpo_examples`).
2. Normalize and combine into one score (weighted sum, PCA, or learned regressor — try more than one).
3. Split into easy/medium/hard via quantiles or k-means; save split indices.
4. Validate: confirm SFT accuracy on easy ≫ hard; adjust thresholds if not.
5. Run pilot GRPO trainings (curriculum vs. random) at small scale.
6. Build a `difficulty_diagnostics.py` script: correlation tables, heatmaps, embedding plots, learning-curve comparison.
7. Iterate on signal set/weights based on diagnostics.

**Outputs to save**: difficulty scores + cluster assignments (JSON),
learning curves, correlation tables/heatmaps — same "everything scripted
and reproducible" discipline the rest of this project already follows.
