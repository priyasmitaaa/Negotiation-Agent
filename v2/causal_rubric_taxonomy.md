# Causal Rubric Taxonomy for the Negotiation Reward

**Terminology note (added 2026-07-20, reviewer-recommended):** "causal" here
means *intervention-based causal-factor sensitivity* — we intervene on one
structured field (via donor substitution or template edit) and check the
reward responds — not a formal structural causal model (SCM) with an
explicit causal graph and identified do-operators. Our donor-substitution
interventions typically change several correlated fields at once (e.g.
swapping `zone` also changes `gt_decision`/`buyer_offers_so_far` together,
since those co-occur in the donor turn), so this is closer to *contrastive
intervention learning* grounded in domain-structured factors than to formal
causal inference. Throughout this doc and the rest of the pipeline, prefer
saying **"intervention-based causal-factor reward model"** or
**"causal-factor-aware reward model"** over an unqualified "causal reward
model" — the qualified phrasing is accurate and much harder for a reviewer
to attack on causal-inference grounds, without weakening the actual
contribution (sensitivity to negotiation-relevant structure, invariance to
surface style is the real, defensible claim either way).

**Purpose.** Define, precisely and in terms of fields that already exist in
`dataset/dialogue_*.json` and in the GRPO example builder
(`grpo_curriculum.py::build_grpo_examples`), which factors a negotiation
reward *should* move on (causal) and which factors it *must not* move on
(spurious). This taxonomy is the foundation for:

- `generate_intervention_pairs.py` (builds causal/spurious pairs from this taxonomy)
- `train_causal_rm.py` (the training objective is: sensitive to causal factors, invariant to spurious factors)
- `causal_rm_audit.py` (measures whether a reward — old heuristic, naive LLM judge, or the new RM — actually respects this taxonomy)

No new hand-authored rubric is required for the base version: every causal
factor below already has a field in `factor_state` or `emotion` produced by
`restructure_dataset.py` / `detect_emotions.py`. `grpo_curriculum.py`'s
`difficulty_for_example()` already reads several of these fields to compute
difficulty tags — that function is the closest existing thing to a rubric
taxonomy in the repo, and the causal factors below are a superset/formalization
of its `difficulty_reasons`.

---

## 1. Causal Factors

These are attributes of the *negotiation state* that a correct seller
response must react to. Changing one of these while holding everything else
fixed should change what a good response looks like, and therefore should
change the reward assigned to a fixed response.

| # | Causal factor | Backing schema field(s) | What "reacting correctly" means | Existing heuristic proxy (to be replaced/augmented) |
|---|---|---|---|---|
| C1 | Decision correctness given state | `factor_state.decision` (gt), `factor_state.zone`, `factor_state.anchor_price`, `factor_state.buyer_offers_so_far` | Predicted `<decision>` should match what the price/zone state implies | `decision_reward()` in `reward_function.py` — already rule-based/causal, keep as-is |
| C2 | Price-strategy consistency | `factor_state.buyer_offers_so_far`, `pricing.fair_value`, `factor_state.harm_direction`, `factor_state.zone` | LEVERAGE responses hold firm near/above fair value; MITIGATE responses soften/concede without collapsing below floor; response must reference the actual offer, not a generic one | `price_strategy_reward()` — currently keyword-set based (`LEVERAGE_TERMS`, `MITIGATE_TERMS`) → spurious-prone, see S2 |
| C3 | Emotion-appropriate adaptation | `emotion.valence`, `emotion.intensity`, `emotion.labels` | Negative/high-intensity buyer emotion should be met with acknowledgement/softening, not escalation; positive emotion can be met with closure-oriented moves | `emotion_reward()` — currently keyword-set based (`ACK_TERMS`, `SOFTEN_TERMS`) → spurious-prone, see S3 |
| C4 | Negotiation progression | Sequence of `factor_state.buyer_offers_so_far` across turns, presence of a new offer/next-step in the response | Response should change negotiation state (new offer, concrete ask, closing move), not stall | `progression_reward()` — currently word-count + keyword-set → spurious-prone, see S1/S3 |
| C5 | Flip-turn handling | `factor_state.decision_is_flip`, `factor_state.previous_decision` | At a flip turn, the response must reflect the *new* decision, not carry over the previous turn's stance | `flip_reward()` — already rule-based/causal, keep as-is |
| C6 | Zone-appropriate boundary behavior | `factor_state.zone` (`pre_crystallisation` vs `post_crystallisation`), `factor_state.anchor_price` proximity to `pricing.fair_value` | Near-boundary cases (offer within ~10% of fair value) are genuinely ambiguous and should be graded more leniently than clear-cut cases | Already partially captured in `grpo_curriculum.py::price_gap_pct` / `difficulty_for_example` — extend into RM training signal, not just curriculum difficulty |
| C7 | *(proposed, not scoped)* Progression-appropriateness conditioned on buyer emotion | None worked out yet — would be some combination of C3's `emotion.*` fields and C4's `buyer_offers_so_far` sequence | Open question, not yet a defined factor: does a progression-appropriate move (new offer/closing move) also need to be *emotion*-appropriate in its manner, independent of C3 and C4 each being satisfied alone? | None — see `causal_rm_architecture.md` section 9.4.1 for why this came up (a trunk-decoupling side effect, not a designed test) |

## 2. Spurious Factors

These are attributes of the *response's surface form* that correlate with
"looks good" but do not track actual negotiation competence. A reward that
moves when these change (holding the causal factors fixed) is reward-hackable.

| # | Spurious factor | Where it currently leaks into the reward | Why it's spurious | Diagnostic confound to test in `causal_rm_audit.py` |
|---|---|---|---|---|
| S1 | Response length / word count | `progression_reward()`, `naturalness_reward()` use `word_count()` thresholds directly | A longer response is not a better negotiation move; SFT v6 responses average ~15.8 words and adding filler should not raise reward | `word_count(response)` vs. reward, Pearson r |
| S2 | Generic politeness / warmth phrasing | `price_strategy_reward()`, `emotion_reward()` reward presence of words like "understand", "appreciate", "fair" regardless of whether they're tied to an actual offer/emotion change | A seller can say "I understand" while ignoring the buyer's actual offer — the word is a spurious marker of empathy, not real adaptation | count of `ACK_TERMS`/`SOFTEN_TERMS` hits vs. reward, controlling for whether an actual offer changed |
| S3 | Punctuation/formatting artifacts, tag ordering | `naturalness_reward()` checks repeated punctuation/malformed tags — legitimately causal for *format*, but must not bleed into strategy/emotion scoring | Superficial formatting is a format-gate concern (already handled by `format_reward()`), not a strategy/emotion concern | correlation between formatting-only edits and non-format reward components — implement as a cheap template-level intervention (capitalization/punctuation swap, no LLM call needed) alongside S1/S2 in `generate_intervention_pairs.py` |
| S4 | TTS acoustic style (pitch, rate, voice identity, loudness) | Not currently in `reward_function.py`'s text-only components, but relevant once `audio_quality_reward()` / speech-mode RL is used | Two renders of the *same* strategically-correct text at different prosody/voice settings (`voice_instruction_generator_v2.py` params) should get the same non-audio reward components | re-synthesize same response with varied TTS voice/rate settings, check emotion/progression/price_strategy components are unchanged |
| S5 | Superficial LLM-judge style bias | `judge.py`'s Qwen3-8B judge scores fluency/naturalness alongside strategy in one holistic call | An LLM judge can rate a fluent-but-wrong response higher than a blunt-but-correct one; conflates naturalness with strategic correctness | correlation between `judge.py` naturalness score and negotiation_quality score across decision-correct vs decision-incorrect examples |
| S6 | Paraphrase (different wording, identical strategy) | Not yet implemented in `generate_intervention_pairs.py` v1 (added 2026-07: reviewer-recommended expansion) | An LLM-generated paraphrase of a gold response that preserves the same decision, price reference, and emotional stance should score the same as the original — reward should track *strategy*, not phrasing choice | generate paraphrases via an LLM call (same pattern as `detect_emotions.py`'s API usage), **manually spot-check a sample** that strategy/price/emotion content is unchanged before trusting the pair as ground truth, since this is now a second generation step that could itself introduce drift |
| S7 | Fluency rewrite (grammar/style polish, identical strategy) | Not yet implemented | A lightly grammar-polished or de-stiffened version of the same response (same content, smoother delivery) should not move the reward — otherwise the RM is rewarding writing quality rather than negotiation quality | same generation + manual-check discipline as S6, but constrained to surface grammar edits only (no reordering of clauses that could change emphasis/strategy) |

## 3. Mapping to Reward Components

| `reward_function.py` component | Status | Action |
|---|---|---|
| `format_reward` | Rule-based, gate | Keep as-is — already causal-by-construction (parses the deployment contract) |
| `decision_reward` | Rule-based | Keep as-is — directly implements C1 |
| `flip_reward` | Rule-based | Keep as-is — directly implements C5 |
| `price_strategy_reward` | Keyword heuristic | Replace with causal RM score for C2/C6, guided by S2 |
| `emotion_reward` | Keyword heuristic | Replace with causal RM score for C3, guided by S2/S5 |
| `progression_reward` | Word-count + keyword heuristic | Replace with causal RM score for C4, guided by S1 |
| `naturalness_reward` | Word-count + regex heuristic | Keep as light guardrail (format-adjacent), do not let it dominate — this is what S3 warns about |
| `repetition_penalty`, `language_impurity_penalty` | Rule-based | Keep as-is — legitimate guardrails, not strategy signals |
| `audio_quality_reward` | Waveform heuristic | Keep as guardrail; add S4 invariance check once speech-mode RL is active |

## 4. Intervention Pair Design Implied by This Table

- **Causal pairs** perturb exactly one of C1–C6 (e.g. swap `buyer_offers_so_far` / `zone` / `emotion.valence` context while holding the candidate response fixed) and assert the reward *must* move in the implied direction.
- **Spurious pairs** perturb exactly one of S1–S7 (length, politeness phrasing, formatting, TTS prosody, judge style, paraphrase, fluency rewrite) while holding all C1–C6 state fixed, and assert the reward *must not* move.

This table is the direct input contract for `generate_intervention_pairs.py`
(Phase 2 of the integration plan) — each generated pair records which C-number
or S-number it targets, so training and audit code can report per-factor
sensitivity/invariance rather than one aggregate number.

**Coverage status (updated 2026-08-01):** `generate_intervention_pairs.py`
now generates causal pairs for **C1, C2, C3, C4, C5** (all except C6) and
spurious pairs for **S1, S2, S3**. C2 (`causal_pair_price_strategy`) and C4
(`causal_pair_progression`) were added after a lambda_inv sweep + pairwise
head-correlation analysis on a trained RM checkpoint showed `price_strategy`
and `progression` — 2 of the 3 dimensions that actually feed `overall` —
were underdetermined: they only ever received invariance-loss gradient,
never causal-ranking gradient, because no C2/C4 pairs existed to anchor
them. Confirmed empirically rather than assumed: re-running the heuristic
baseline audit against the new C2/C4 pairs shows the OLD heuristic's
`price_strategy_reward()`/`progression_reward()` score **exactly 0%** causal
ranking accuracy on them (not noise — systematic, 1,884 and 1,570 pairs
respectively), strong independent evidence the heuristic reward is blind to
negotiation state on these two dimensions.

C4 required a real fix during implementation, not just a copy of the C1/C3/C5
pattern: `progression_stage` (offer-count bucket) was originally confounded
with `zone` (verified empirically: `pre_crystallisation` examples have
*exactly* 1 buyer offer, always, zero variance — crystallisation is
definitionally what happens once offers accumulate), so restricting C4's
donor pool to the same zone as C1 requires made it structurally impossible
to find a contrasting donor. Fixed by scoping C4 to `post_crystallisation`
only (where offer count has real 2-6 spread) with a within-zone median-ish
split (`early_post` vs `late_post` at the `offers <= 3` boundary) — an
honest scope limit (documented in `generate_intervention_pairs.py`), not a
workaround.

**Scoping clarification (2026-08-16), added after the v2 audio-fusion trunk
split removed `emotion`'s gradient-entanglement correlation with
`progression` — see `causal_rm_architecture.md` section 9.4.1 for the full
finding.** That correlation drop naturally raises "shouldn't progression
account for emotion, then?" — checked both C4's formal definition (row
above: `buyer_offers_so_far` sequence and next-step presence, nothing
about affect) and its implementation (`causal_pair_progression` only ever
varies `progression_stage`, an offer-count bucket) and confirmed: **C4 is
scoped to offer-sequence/next-step behavior only.** Buyer-emotion
sensitivity in progression would be *new scope* requiring its own taxonomy
justification (grounded in schema fields, same rigor C1-C5 each got), not
a C4 extension — logged as the still-unscoped **C7** row above rather than
folded into C4's existing pairs. Also checked C2 (price-strategy) for the
same latent pattern while this was fresh, since it's the other dimension
that feeds `overall`: C2's own definition and `causal_pair_price_strategy`
both key strictly off `buyer_offers_so_far`/`fair_value`/`harm_direction`/
`zone` — no emotion dependency, same clean scoping as C4. Nothing to fix
there.

Still not implemented:
- **C6** — zone-boundary leniency modifier rather than an independent ranking signal, lower priority than C2/C4 were.
- **C7** — proposed only, no schema-grounded definition worked out yet; see the row above and `causal_rm_architecture.md` section 9.4.1.
- **S6 (paraphrase)** and **S7 (fluency rewrite)** — require an LLM generation step plus a manual spot-check pass (same discipline as `detect_emotions.py`'s API-based annotation) to confirm the paraphrase/rewrite actually preserved decision, price reference, and emotional stance before it's trusted as a "meaning-preserving" pair.
- **S4 (prosody)** — requires TTS regeneration via `voice_instruction_generator_v2.py`; real compute cost, so this should wait until text-mode spurious invariance is validated first, consistent with the phased "text-only reward first, speech-aware RL later" approach already recommended in `rl_reward_function_v6.md` section 4.11.

Narrower spurious coverage is an acceptable prototype for validating the
pipeline end-to-end, but a reviewer would rightly ask "is invariance really
proven, or just to the two cheapest perturbations?" — so the expanded set
above is required before the ACL-facing experiments, not optional polish.

## 5. Reward Model Output Shape: Rubric Vector, Not a Single Scalar

The causal RM (`causal_reward_model.py`, Phase 3) should output a **rubric
vector** — one score per causal factor family, not a single scalar reward.
This is not new design surface: `reward_function.py::compute_reward` already
returns `(total, components)` where `components` is a dict with per-factor
entries (`decision`, `emotion`, `progression`, `price_strategy`, `format`,
`naturalness`, `flip`, plus `*_meta` diagnostic sub-fields) — the causal RM
should honor that exact same contract so it is a true drop-in replacement,
not a new interface `grpo_curriculum.py`/`error_analysis.py` need to be
rewritten around.

Recommended vector (aligned to the causal factors C1-C6):

```text
decision_score        (C1)
price_strategy_score  (C2, C6)
emotion_score         (C3)
progression_score     (C4)
flip_score            (C5)
overall               = weighted_sum(above)   # same weights as WEIGHTS in reward_function.py
```

Why this matters for the paper, beyond interpretability:
- **Per-dimension ablation** — we can report which rubric dimension actually
  benefits from the causal-invariance training (e.g. does `emotion_score`
  become invariant to S2/S6 while `decision_score` stays sensitive to C1),
  rather than one aggregate number that hides this.
- **Debugging** — when a GRPO rollout gets an unexpected reward, per-component
  logging (already the pattern `reward_function.py::audit_inference_file`
  uses) tells us which rubric dimension drove it.
- **Evidence against memorization** — if the RM has genuinely learned the
  causal rubric rather than a single blended correlation, then perturbing one
  C-factor should move *only* the corresponding rubric score and leave the
  others stable. That per-dimension isolation is itself a strong, reportable
  experimental result (a stronger invariance claim than a single scalar can
  support) and directly answers the "is this really causal, or just another
  consistency-tracking RM" critique the diagnostic companion paper raises.

`train_causal_rm.py` should therefore apply the Bradley-Terry + invariance
loss (Phase 3 of `causal_rubric_rl_plan.md`) **per rubric dimension**, using
each pair's `target_factor` (C1-C6/S1-S7) to route the loss to the matching
output head, rather than training one pooled scalar.
