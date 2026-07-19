# Causal-Rubric Reward Modeling for the Negotiation Agent

## Context

The project (v2/) has completed SFT (v6 is the strong baseline: 98.3%/98.4%
text/speech decision accuracy) and has a working but heuristic GRPO reward in
`reward_function.py` — keyword lists, word-count buckets, n-gram overlap. This
kind of reward is exactly the failure mode the causal-rubric RM literature
warns about: it can be satisfied by *surface* correlates (politeness words,
response length, generic phrasing) rather than actual negotiation competence.
The user wants to adopt the "causal-rubric reward modeling" idea (CRome /
CausalRM / "Reward Models Identify Consistency, Not Causality") as the
project's next research contribution, replacing/augmenting the heuristic
reward with one that is provably sensitive to causal negotiation factors and
invariant to spurious style/acoustic factors — and to make this the centerpiece
of an ACL-level paper, building on the existing `research_paper_draft_v1.md`.

Why this project is an unusually good fit: the dataset already carries
structured causal state per turn (`factor_state`: `zone`, `anchor_price`,
`buyer_offers_so_far`, `harm_direction`, `decision`, `decision_is_flip`) and
affect state (`emotion.labels/intensity/valence`). This means the "causal
rubric taxonomy" the papers call for does not need to be invented from
scratch — it can be *derived directly* from fields the dataset generator
(`restructure_dataset.py`) already produces, and `grpo_curriculum.py`'s
`difficulty_for_example()` already computes rubric-like tags
(`post_crystallisation`, `mitigate_decision`, `negative_emotion`,
`price_boundary_within_10pct`, `decision_flip`). That function is effectively
a proto-causal-rubric-taxonomy already in the repo.

Goal of this plan: turn that existing structure into (1) an explicit causal
vs. spurious rubric taxonomy, (2) an intervention-pair generation pipeline
(text + speech), (3) a causal-invariant reward model that plugs into the
existing GRPO loop as a near drop-in replacement for the soft components of
`compute_reward`, and (4) the ablations/diagnostics needed to make the ACL
paper's empirical case (reward-hacking exposure, causal-sensitivity /
spurious-invariance metrics, downstream policy comparison, human-eval
correlation).

## Causal Rubric Taxonomy (Phase 1)

Ground the taxonomy directly in existing schema fields — no new annotation
required for the base version.

**Causal factors** (must move the reward):
- Decision correctness relative to `zone`/`anchor_price`/`buyer_offers_so_far` (already `R_decision`/`price_strategy_reward` in `reward_function.py`).
- Strategic consistency: does response content match the chosen decision given `harm_direction` and offer trajectory.
- Emotion-appropriate adaptation: does response causally track `valence`/`intensity` (acknowledge negative/high-intensity, close on positive) — not just contain acknowledgement words.
- Progression: does response change negotiation state (new offer, concrete next step) vs. stall.
- Flip handling: correct behavior at `decision_is_flip` turns.

**Spurious factors** (must NOT move the reward):
- Response length / word count (S1).
- Generic politeness/warmth phrasing that doesn't track any actual offer/emotion change (S2).
- Punctuation/formatting style, tag ordering artifacts (S3).
- TTS acoustic style: pitch, speaking rate, voice identity, loudness — i.e. `voice_instruction_generator_v2.py` parameters — when content/strategy is unchanged (S4).
- Superficial LLM-judge style bias conflating fluency with strategy (S5).
- Paraphrase: different wording, identical decision/price/emotion content (S6, reviewer-recommended addition).
- Fluency rewrite: grammar/style polish, identical strategy (S7, reviewer-recommended addition).

Deliverable: `causal_rubric_taxonomy.md` documenting this table with the exact
schema fields backing each rubric item, plus the explicit list of confounds to
test against later (word count, ack-keyword count, TTS pitch/rate/loudness).

**Status (2026-07-19):** `generate_intervention_pairs.py` v1 implements only
S1/S2 (both template-verifiable, no LLM call) as a first prototype. Before
final ACL experiments this must be extended to S3 (cheap, template-level),
then S6/S7 (needs an LLM generation step + manual spot-check, since a second
generation step can itself introduce drift into what's supposed to be a
clean ground-truth invariance pair), then S4 (needs TTS regeneration — real
compute, deferred until text-mode invariance is validated). Narrower coverage
is fine for validating the pipeline end-to-end, but the full S1-S7 set is
required before the invariance claims are reportable — see
`causal_rubric_taxonomy.md` section 4 for the detailed rationale per factor.

## Intervention-Pair Generation (Phase 2)

New script: `generate_intervention_pairs.py`, built on top of the existing
per-dialogue loader logic already used in `grpo_curriculum.py::build_grpo_examples`
(reuse its context/prompt construction, `factor_state`/emotion extraction).

For each eligible seller turn, generate two pair types:

1. **Causal pairs** (vary a causal attribute, hold surface style fixed):
   - Swap `zone`/`anchor_price`/last buyer offer to flip the correct decision (using other dialogues' matched turns as donors, or template edits to `buyer_offers_so_far`), keep response phrasing template fixed → reward must change.
   - Swap buyer emotion valence/intensity (e.g. substitute a negative/high-intensity buyer turn context for a neutral one) while keeping seller response fixed → reward for an emotion-blind response must drop.
   - Use existing `sft_data_v6.py`/`sft_harmonise_reasoning.py` utilities for reading/writing dialogue turns to avoid reinventing parsing.

2. **Spurious pairs** (vary surface only, hold causal content fixed):
   - LLM-paraphrase the same response at different lengths/politeness levels while preserving decision + price content (small script call, same style as `detect_emotions.py`'s API-call pattern) → reward must stay flat.
   - Re-synthesize the same response text through `generate_tts_v2.py`/`voice_instruction_generator_v2.py` with different prosody/voice settings → audio-reward component must stay flat.

Output: `causal_rm_pairs_train.json` / `_val.json` / `_test.json`, split-aligned
with `splits_v6.json` so there's no leakage across SFT/RM/RL splits. Each
record: `{context, response_a, response_b, pair_type: causal|spurious, expected_relation}`.

## Causal-Invariant Reward Model (Phase 3)

New files: `train_causal_rm.py`, `causal_reward_model.py`.

- Base model: reuse `judge.py`'s Qwen3-8B loading pattern (already proven to run locally), fine-tuned as a **rubric-vector** reward head (or pairwise Bradley-Terry per dimension) rather than a JSON-scoring prompt or a single pooled scalar.
- **Output shape (reviewer-recommended, adopted 2026-07-19): a rubric vector, not one scalar.** `causal_reward_model.py::score(example, response)` should return the same shape as `reward_function.py::compute_reward`'s `components` dict — `decision_score` (C1), `price_strategy_score` (C2/C6), `emotion_score` (C3), `progression_score` (C4), `flip_score` (C5), plus `overall` = the same weighted sum `WEIGHTS` already used in `reward_function.py`. GRPO still consumes only `overall` as the scalar reward, so this is free from the RL-training-loop's perspective, but the paper gains: per-dimension ablations, easier debugging of any single rollout's reward, and — most importantly — a *falsifiable* per-dimension invariance claim (perturbing C3 should move only `emotion_score`, not `decision_score`; perturbing S2 should move nothing) rather than one aggregate number that can hide a mix of correct and spurious sensitivity. See `causal_rubric_taxonomy.md` section 5 for the full rationale.
- Loss = Bradley-Terry preference loss on **causal pairs** (push the matching rubric-dimension score apart, in the correct direction, routed by each pair's `target_factor`) + an invariance penalty (squared/KL difference) on **spurious pairs** (push every rubric-dimension score together, not just `overall`). This is the CRome-style objective, applied per-dimension instead of pooled.
- `causal_reward_model.py` exposes `score(example, response) -> (overall, components)` with the same input contract as `compute_reward` in `reward_function.py` (same `example` dict keys: `gt_decision`, `zone`, `prior_buyer_emotion`, `buyer_offers`, etc.) so it's swappable — `grpo_curriculum.py::negotiation_grpo_reward` and `error_analysis.py` keep working against the same `components` dict shape they already expect.
- Optional Phase 3b (later): if/when human annotations from `human_annotation_guide.md` come in as noisy observational preferences, add propensity-corrected reweighting (CausalRM-style) as a second training stage — do not build this until real annotation data exists. Note the human rubric dimensions (decision correctness, price strategy, emotion handling, negotiation progression) already line up 1:1 with the RM's rubric-vector dimensions, which is what makes this phase feasible without redesigning the annotation form.

## GRPO Integration (Phase 4) — minimal, drop-in

Modify `reward_function.py::compute_reward`:
- Keep the hard gates as-is: `format_reward`, `decision_reward` (already causal-by-construction, rule-based, low-noise — no need to learn these).
- Add a `--use_causal_rm` path that replaces the heuristic `emotion_reward`, `progression_reward`, `price_strategy_reward`, `naturalness_reward` calls with calls into `causal_reward_model.score(...)`, keeping the same component-logging structure (`components["emotion"]`, etc.) so `error_analysis.py`/audit tooling keep working unchanged.
- `grpo_curriculum.py::negotiation_grpo_reward` needs no structural change — it already just calls `compute_reward`; it will pick up the causal RM automatically once the flag/config is threaded through.
- Curriculum compatibility already exists via `CURRICULUM_WEIGHTS`/`difficulty_for_example`. Extend the curriculum philosophy (not code structure) to also stage *reward* complexity: Phase 1 GRPO steps use hard gates only (format+decision), Phase 2+ blend in the causal RM soft rewards — mirrors `rl_reward_function_v6.md`'s existing "Stage D — Curriculum GRPO" section, so this is continuing an already-planned direction, not a new mechanism.

## Diagnostics for the ACL paper (Phase 5)

New script: `causal_rm_audit.py` (sibling to `reward_function.py`'s
`audit_inference_file`, same JSON-in/JSON-out pattern):
- Reward-hacking exposure: correlate reward (old heuristic vs. naive LLM-judge vs. new causal RM) against spurious confounds (word count, ack-keyword count, TTS pitch/rate) on existing v6 inference outputs (`sft_output_v6/20260614_010317/inference_*.json`) — show causal RM has ~0 correlation, others don't. This is the "Reward Models Identify Consistency, Not Causality" diagnostic, directly reusable since we already have those inference JSONs.
- Causal sensitivity: on held-out causal pairs, does RM correctly rank them (accuracy metric, like a preference-model eval).
- Downstream comparison: run small GRPO pilots (reusing `train_grpo_curriculum.py`, same pattern as the existing `grpo_pilot_20260704*` smoke runs) with (a) current heuristic reward, (b) naive LLM-judge-as-reward, (c) causal RM — compare via existing `error_analysis.py` + `judge.py` + a slice of human eval using `human_annotation_guide.md`'s rubric, since it already maps 1:1 onto the causal rubric (decision correctness, price strategy, emotion handling, progression, fairness).

## Paper Narrative

- **Motivation figure**: current reward is keyword/length-driven → show its spurious correlations empirically (this project's own `reward_function.py` becomes the "naive baseline" strawman for free).
- **Method**: causal rubric taxonomy grounded in dataset's own structured `factor_state`/emotion fields (no hand-authored rubric needed — a nice novelty/feasibility argument), intervention generation spanning text *and* speech/prosody (fills the literature gap called out: no prior work does causal-invariant RM for spoken negotiation).
- **Results**: reward-hacking-resistance metrics + downstream GRPO policy quality (decision accuracy preserved, human-judged emotion/progression improved without length/generic-empathy inflation) + human-eval correlation validating the RM tracks true causal quality.

## Implementation Order

1. `causal_rubric_taxonomy.md` (no code, formalizes Phase 1) — quick, unblocks everything else conceptually.
2. `generate_intervention_pairs.py` + generated pair files (Phase 2).
3. `causal_reward_model.py` + `train_causal_rm.py` (Phase 3).
4. Wire into `reward_function.py` / `grpo_curriculum.py` behind a flag (Phase 4).
5. `causal_rm_audit.py` for diagnostics, run against existing v6 inference JSONs immediately (doesn't require the RM to be trained yet — can run on old reward first to establish the "before" baseline).
6. Small GRPO pilot comparison (reuse `train_grpo_curriculum.py` CLI, mirror the existing `grpo_pilot_*` smoke-test workflow) once RM is trained.

## Verification

- Step 1: sanity-run `causal_rm_audit.py` against existing heuristic reward on `sft_output_v6/20260614_010317/inference_20260620_150244.json` to confirm the spurious-correlation baseline is measurable and non-trivial (motivates the whole paper).
- Step 2: unit-check `generate_intervention_pairs.py` output on a handful of dialogues (e.g. `dialogue_0964.json`, which has a clean pre/post-crystallisation flip at turn 4→6) to confirm causal pairs actually flip the intended factor and spurious pairs actually preserve it.
- Step 3: after RM training, re-run `causal_rm_audit.py` to confirm correlation with spurious confounds drops and causal-pair ranking accuracy is high.
- Step 4: run one short GRPO pilot (few steps, mirroring `grpo_pilot_20260704b.log`'s scale) with the causal RM wired in, confirm training doesn't crash and reward/KL curves look sane, before committing to a full run.
