# Causal-Factor-Aware Reward Model — Architecture Design

**Status:** design finalized 2026-07-20, precedes `causal_reward_model.py` /
`train_causal_rm.py` per the revised implementation order in
`causal_rubric_rl_plan.md` (design doc first, so model + training code
implement an already-settled spec rather than improvising architecture and
training logic simultaneously). **v1 implemented and dry-run verified
2026-07-20; not yet trained on the real backbone.**

**Terminology note:** see `causal_rubric_taxonomy.md`'s terminology note —
"causal" throughout this doc means intervention-based causal-factor
sensitivity, not a formal SCM. Prefer "causal-factor-aware reward model" in
writing.

This doc exists so the paper's Method section can point at one settled
architecture description, and so implementation doesn't drift from the
rubric-vector decision made in `causal_rubric_taxonomy.md` section 5.

## 1. Goal Recap

Replace the keyword-heuristic `price_strategy_reward`, `emotion_reward`,
`progression_reward` in `reward_function.py` with a learned reward model
that is (a) sensitive to the causal factors C1-C6 and (b) invariant to the
spurious factors S1-S7, per `causal_rubric_taxonomy.md`. Output a rubric
vector, not a single scalar, so per-dimension behavior is auditable.

## 2. Architecture Diagram

```text
                context (dialogue state: product, price, conversation
                so far, buyer emotion) + candidate seller response
                                  |
                                  v
                    format as one text sequence
                 (same prompt-construction pattern
                  as judge.py::build_judge_prompt /
              grpo_curriculum.py::build_grpo_examples)
                                  |
                                  v
                +---------------------------------+
                |   Qwen3-8B backbone (frozen)     |   <- reuse judge.py's
                |   AutoModelForCausalLM, bf16      |      loading pattern;
                +---------------------------------+      frozen for v1 (see
                                  |                       section 4)
                                  v
                pooled hidden state at final token
                        (hidden_dim,)
                                  |
                                  v
                +---------------------------------+
                |   Shared MLP trunk (trainable)    |   <- NEW (2026-07-20,
                |   Linear -> GELU -> Linear        |      reviewer-recom-
                |   hidden_dim -> 512 -> 256        |      mended): the 5
                +---------------------------------+      rubric dimensions
                                  |                       are correlated
                        shared repr (256,)                (e.g. softening a
                                  |                        response for an
                                  v                        angry buyer moves
                +---------------------------------+       emotion AND
                |   5 linear regression heads       |      progression AND
                |   (one per rubric dimension)      |      price_strategy
                +---------------------------------+       together) — a
                  |        |        |        |        |   shared trunk lets
                  v        v        v        v        v   the model learn
              decision  price_   emotion  progres-  flip  "this is a
              _score    strategy _score   sion_     _score negotiation
              (diag-    _score   (C3)     score     (diag- state" once
              nostic,   (C2,C6)           (C4)      nostic, before
              C1)                                   C5)    specializing,
                  |        |________|________|        |    standard
                  |             |                      |    multi-task
                  |             v                      |    learning
                  |   overall = weighted_sum(           |   practice.
                  |     price_strategy, emotion,        |
                  |     progression; WEIGHTS from        |
                  |     reward_function.py)             |
                  |             |                      |
                  +-------------+----------------------+
                                |
                                v
                   clip(overall, -2.0, +2.0)
                                |
                                v
      ALL 5 dimensions + overall are always returned as one rubric_vector
      dict, even though GRPO only consumes `overall` and decision_score/
      flip_score are diagnostics-only (see section 3). This costs nothing
      at inference time and is what makes the per-epoch per-dimension
      training curves / ACL figures possible later (section 5.3).
                                |
                                v
              fed into reward_function.py::compute_reward
              in place of the 3 heuristic sub-rewards;
              decision_score/flip_score logged for
              cross-validation against the existing
              rule-based decision_reward()/flip_reward()
              (see section 3)
```

Each per-dimension head outputs a raw logit passed through `tanh`, bounding
every dimension to `[-1, 1]` — matching the numeric range the existing
heuristic components already use in `reward_function.py` (e.g.
`emotion_reward` returns values in `{-1.0, -0.5, 0.0, 0.5, 1.0}`), so the RM
is a drop-in numeric replacement, not just a drop-in interface.

**Trainable parameter count stays small even with the shared trunk**: the
backbone is frozen (section 4), so only the MLP trunk (`hidden_dim -> 512 ->
256`, a few million params) and the 5 small linear heads (`256 -> 1` each)
are trained — nowhere near the cost of fine-tuning the 8B backbone, keeping
the "frontload cost into RM training, cheap at GRPO rollout time" property
intact.

## 3. Why 5 Heads When Only 3 Feed the Total Reward

The RM predicts all of C1-C6's associated dimensions
(`decision_score`, `price_strategy_score`, `emotion_score`,
`progression_score`, `flip_score`) even though `reward_function.py` keeps
`decision_reward()` and `flip_reward()` rule-based (per
`causal_rubric_taxonomy.md` section 3 — those two are already
low-noise/rule-derivable, no benefit to learning them). Reasons to still
predict them:

1. **We already generated C1 and C5 intervention pairs** (`causal_pair_decision_zone`,
   `causal_pair_flip` in `generate_intervention_pairs.py`) — discarding that
   training signal would waste data we already paid for.
2. **Cross-validation, not redundancy.** At audit time
   (`causal_rm_audit.py`, implementation step 6), we can check that
   `decision_score`'s sign agrees with the rule-based `decision_reward()`'s
   verdict on held-out examples. High agreement is evidence the RM has
   learned something real about C1, not just about C2-C4 in isolation —
   this directly strengthens the "genuinely causal, not just another
   consistency-tracking RM" claim from the diagnostic-paper critique the
   whole project is responding to.
3. Only `price_strategy_score`, `emotion_score`, `progression_score` are
   summed into `overall` / fed back into `compute_reward`, because those are
   the three components that were previously keyword-heuristic and are
   being replaced. `decision_score`/`flip_score` are logged as diagnostics
   only (`components["decision_score_rm_diagnostic"]`, etc.) — GRPO's actual
   reward signal for decision/flip correctness continues to come from the
   existing rule-based gates, unchanged.

## 4. Backbone: Frozen vs. Fine-Tuned

**v1 (this implementation): frozen Qwen3-8B backbone + trainable linear
heads only.** Rationale:

- Matches the project's existing "frontload compute into reward-data
  generation and RM training, keep online RL cost close to vanilla" goal
  from the original CRome-adaptation proposal — a frozen backbone means we
  compute expensive forward passes once (or cache them) rather than
  backpropagating through 8B parameters for every pair.
- A linear-probe-style head on frozen features is the cheapest thing that
  could plausibly work, and is a reasonable first thing to try before
  committing to full/LoRA fine-tuning of the backbone.
- If v1 underperforms (heads can't separate causal from spurious pairs even
  with the backbone's existing representations), the fallback is LoRA
  fine-tuning of the backbone jointly with the heads — same LoRA recipe
  already used everywhere else in this project (`train_sft_v6.py`: r=8,
  alpha=16, dropout=0.1) — not full fine-tuning. This fallback is documented
  here but only exercised if v1's audit (implementation step 6) fails.

## 5. Loss Formulation

Two pair families, each contributing a loss term, plus a small stability
regularizer, summed with tunable weights:

```text
L_total = lambda_bt * L_causal + lambda_inv * L_spurious + lambda_reg * L_reg
```

`lambda_bt = 1.0`, `lambda_inv = 1.0` as starting points (see 5.2 for the
tuning heuristic). `lambda_reg` is small (e.g. `1e-3`) — `L_reg` is L2 weight
decay on the head outputs (not the weights themselves), added purely for
training stability: unconstrained reward heads can drift to large
magnitudes over training even when the ranking/invariance losses are
satisfied, since both are shift/scale-tolerant on their own. `L_reg` is not
expected to change ranking or invariance behavior, only to keep raw logits
(pre-`tanh`) in a well-behaved range.

### 5.1 Causal pairs → Bradley-Terry ranking loss, per dimension

Each causal pair has `context_a`, `context_b`, one `shared_response`, a
`target_factor` (e.g. `C3_emotion`), and `expected_relation`
(`a_better`/`b_better`). Score the *same* response under both contexts:

```text
s_a = head[target_factor's dimension]( backbone(context_a, shared_response) )
s_b = head[target_factor's dimension]( backbone(context_b, shared_response) )

sign = +1 if expected_relation == "a_better" else -1

L_causal_i = -log( sigmoid( sign * (s_a - s_b) ) )
```

Only the head matching `target_factor` receives gradient for that pair
(`C1_decision_zone` → `decision_score` head, `C3_emotion` → `emotion_score`
head, `C5_flip` → `flip_score` head; our current C1/C3/C5 coverage — see
`causal_rubric_taxonomy.md` section 4 for why C2/C4/C6 causal pairs aren't
generated yet). This routing is why `target_factor` is recorded on every
pair in `generate_intervention_pairs.py`'s output.

### 5.2 Spurious pairs → invariance penalty, all dimensions

Each spurious pair has `response_a`, `response_b`, and (via
`dialogue_id`/`turn_index`) an implicit shared context — **implementation
note:** `generate_intervention_pairs.py`'s spurious-pair records do not
inline the context dict the way causal pairs do; `train_causal_rm.py` must
reconstruct context by re-running `grpo_curriculum.py::build_grpo_examples`
once per split and indexing by `(dialogue_id, turn_index)`, exactly the
pattern `generate_intervention_pairs.py::build_donor_index` already uses.

```text
for each dimension d in {price_strategy, emotion, progression}:
    s_a = head_d( backbone(context, response_a) )
    s_b = head_d( backbone(context, response_b) )
    L_spurious += (s_a - s_b) ** 2
```

Averaged over dimensions and pairs. `lambda_inv` starts at `1.0` (equal
weight to the causal loss); tune based on the audit in step 6 — if the RM
is invariant but insensitive (collapses to a constant), lower `lambda_inv`;
if it's sensitive but not invariant, raise it.

### 5.3 Training-Time Logging

Log every training step/epoch, per rubric dimension where applicable — these
become the paper's training-dynamics figures directly (e.g. "emotion head
sensitivity rising over epochs while staying invariant to S2"), which is
only possible because section 6 always returns the full vector rather than
just `overall`:

```text
bt_accuracy_per_dimension      # did s_a > s_b match expected_relation, per C-factor
invariance_mse_per_dimension   # mean (s_a - s_b)^2 on spurious pairs, per dimension
head_correlation                # pairwise correlation between the 5 heads' outputs
                                 # (sanity check the shared trunk isn't collapsing
                                 # all heads to the same signal)
avg_reward, reward_variance     # overall reward distribution, watch for saturation
```

## 6. Input/Output Contract

`causal_reward_model.py` exposes:

```python
def score(example: dict, response: str) -> tuple[float, dict]:
    """Same top-level signature shape as reward_function.py::compute_reward.

    `example` uses the same keys already used throughout the project
    (gt_decision, zone, prior_buyer_emotion, buyer_offers, fair_value,
    decision_flip, ...) — see grpo_curriculum.py::build_grpo_examples for
    the canonical set.

    Returns (overall, components). `components` ALWAYS contains the full
    rubric vector, not just the 3 dimensions that feed `overall` — cheap to
    return, and it's what makes per-dimension audit/plotting possible later:
      decision_score        <- always present; diagnostic only, cross-checked
                                against decision_reward()'s verdict, does NOT
                                feed into overall
      price_strategy        <- feeds into overall (replaces price_strategy_reward)
      emotion                <- feeds into overall (replaces emotion_reward)
      progression             <- feeds into overall (replaces progression_reward)
      flip_score             <- always present; diagnostic only, cross-checked
                                against flip_reward()'s verdict, does NOT feed
                                into overall
    `overall` is a fixed weighted sum (not a learned combiner — see section
    4's v1 scope note and the reviewer discussion this doc's history
    reflects: "much easier to interpret/debug/publish") of price_strategy,
    emotion, progression using the same WEIGHTS dict already defined in
    reward_function.py.
    """
```

This mirrors `compute_reward`'s existing `(total, components)` return shape
exactly (see `reward_function.py` lines ~500-570), so
`grpo_curriculum.py::negotiation_grpo_reward` and `error_analysis.py` need
no structural changes to consume it — only `reward_function.py` gets a new
branch (Phase 4, implementation step 7) that calls this instead of the
three heuristic functions it's replacing.

## 7. Verification Checklist (maps to implementation step 6)

Before wiring into GRPO:

- [ ] On held-out causal pairs (val/test split): ranking accuracy per
      dimension (did the RM score `context_a` higher than `context_b` when
      `expected_relation == "a_better"`, and vice versa) — target
      meaningfully above 50% (chance) for C1/C3/C5.
- [ ] On held-out spurious pairs: mean `|s_a - s_b|` per dimension — should
      be small and, critically, smaller than the same statistic measured
      against the **old heuristic reward** (`price_strategy_reward`,
      `emotion_reward`, `progression_reward` from `reward_function.py`) on
      the same pairs — this comparison is the headline "our RM is less
      reward-hackable than the baseline" result for the paper, and doesn't
      require a trained RM to compute the baseline side of it (can run
      today against `causal_rm_pairs_val.json`).
- [ ] `decision_score`/`flip_score` diagnostic agreement rate with
      `decision_reward()`/`flip_reward()` sign, on the same held-out pairs.
- [ ] Reward distribution sanity: not saturated at ±1 for every example
      (a collapsed/degenerate head looks "invariant" for the wrong reason).

## 8. Deferred Improvements — Required Before Final ACL Experiments, Not Before the v1 Prototype

Reviewer feedback (2026-07-20) identified several real gaps. None of them
invalidate running the v1 prototype (implementation step 6 — a small,
real-backbone training run to sanity-check the whole pipeline works and
produces a directionally sane RM); all of them should land before the
final, paper-reportable RM training run. Same "prototype cheap, then harden"
philosophy already applied to S1-S3 vs. S6/S7/S4 in
`causal_rubric_taxonomy.md` section 4.

### 8.1 Structured feature fusion, not text-only

v1 renders context as text (`format_context_response()`) and lets the
backbone's language-model representation carry all the signal. But the
project already has clean structured fields (`zone`, `emotion.valence`,
`buyer_offers_so_far`, `decision_flip`, `fair_value`) — collapsing them into
a sentence and hoping the LM attends to them correctly throws away
information for free. Planned v2 architecture:

```text
Qwen3-8B pooled hidden state  ‖  structured-feature embedding
                    (concat)
                        |
                        v
                  shared MLP trunk
```

The structured-feature embedding side is a small learned embedding table /
MLP over the same fields `grpo_curriculum.py::difficulty_for_example()`
already reads (categorical fields like `zone`/`valence` embedded, numeric
fields like `price_gap_pct`/`buyer_offers` count normalized). This should
improve both robustness (less dependent on the LM correctly "noticing" a
field mentioned in a paragraph) and interpretability (can inspect the
structured-embedding path's contribution separately from the text path's).

### 8.2 Nearest-neighbor donor matching — DONE (2026-07-27)

`generate_intervention_pairs.py::pick_donor()` used to sample randomly from
a same-zone pool and filter by a predicate only (different dialogue,
opposite valence/decision/etc.), which meant causal pairs could differ on
*more* than the one targeted factor. Fixed: `pick_donor()` now takes an
`anchor` argument and first searches (up to `tries=50`) for a donor that
also passes `donor_similarity_ok()` — same product category, asking price
within 30% relative, turn position within ±2 — before falling back to the
original similarity-free search if no match is found. All three causal-pair
functions (`causal_pair_decision_zone`, `causal_pair_emotion`,
`causal_pair_flip`) now pass `anchor=ex`.

Measured result across all three regenerated splits: **49.3% of causal-pair
donors matched strictly** (category+price+turn-position), **49.8% fell back**
to the relaxed similarity-free search, and only **0.8% found no donor at
all** (tracked via `DONOR_MATCH_STATS`, saved per-split in
`causal_rm_pairs_*.json`'s `donor_match_stats` field). The ~50/50 split
between strict/relaxed is expected — the dataset's product categories and
price ranges aren't dense enough for every anchor to have many strictly
similar candidates in the opposite zone/valence/flip-status pool — but the
near-zero failure rate confirms the constraint isn't starving pair
generation. `causal_rm_pairs_{train,val,test}.json` were regenerated with
this fix; the heuristic baseline audit was re-run against the new pairs and
produced materially the same numbers (expected — donor matching improves
pair *quality/cleanliness*, not what the heuristic itself does).

### 8.3 Head disentanglement / correlation monitoring

The shared trunk (section 2) is deliberately there to let correlated
dimensions share representation — but nothing currently *penalizes* a
causal pair's gradient from leaking into non-targeted heads (e.g. an
emotion-only intervention nudging the price_strategy head too, since both
read from the same trunk output). `train_causal_rm.py` now logs
`head_correlation` (pairwise correlation between the 5 heads' raw outputs
across a batch — see updated section 5.3) so this can be *monitored* during
the v1 prototype run without committing to a fix yet. If correlation is
high and doesn't correspond to genuine domain correlation (e.g. softening
for an angry buyer legitimately moving emotion+progression+price_strategy
together, which is fine per section 2's own rationale — vs. an emotion
intervention pair moving `decision_score` for no principled reason, which
is not fine), add an orthogonality/disentanglement penalty term
(`lambda_orth * off_diagonal(head_correlation_matrix)`) in a later training
iteration. Not added preemptively because it's easy to over-regularize away
the *legitimate* cross-dimension correlation the shared trunk exists to
capture — better to look at real correlation numbers from a real run first.

### 8.4 Additional audit metrics (for `causal_rm_audit.py`, not yet written)

- **Robustness under intervention**: for one response, generate ~20
  spurious variants (mixing S1/S2/S3, and later S6/S7), measure reward
  variance across them — lower is better, and this is a stronger aggregate
  robustness statistic than single-pair invariance checks.
- **Head consistency under single-factor interventions**: when only one
  causal factor is perturbed (e.g. emotion via a C3 pair), the *targeted*
  head should move and the *other* heads should stay stable — a per-example
  version of the aggregate `head_correlation` monitoring in 8.3, and a much
  stronger validation than checking only the `overall` scalar moved
  correctly.

## 9. Audio-Grounded emotion Fusion (v2, design finalized 2026-08-02, revised same day)

### 9.1 Why this exists, and why the scope is `emotion` alone, not `emotion`+`flip_score`

Section 1-8 describe a purely text-grounded RM: `emotion` is scored against
`emotion.labels/intensity/valence` — text annotations Gemini derived from
buyer audio during dataset construction (`detect_emotions.py`), not the
waveform itself. This was flagged directly by the project owner as
insufficient for a project whose core premise is speech-conditioned
negotiation: a reward that never touches raw audio can't distinguish a
policy that genuinely perceives the buyer's voice from one that's
pattern-matching on a pre-computed label baked into the prompt. Sections
9.2-9.6 extend the RM to consume real buyer audio for **`emotion` only**.

**`flip_score` was in scope in the first draft of this section, then
removed on review** — worth recording why, since the reasoning matters more
than the conclusion. The original justification was "flips often track
emotional escalation," which is true as a *correlation* (flip turns and
emotionally charged buyer speech co-occur) but doesn't establish that
`flip_score`'s own reward target benefits from audio. Re-checking C5's
actual definition in `causal_rubric_taxonomy.md`: "at a flip turn, the
response must reflect the *new* decision, not carry over the previous
turn's stance" — this is a check on the *seller's response text* against a
*structural* state variable (`factor_state.decision_is_flip`,
`previous_decision`), not a perceptual/affect judgment. It's the same kind
of check as C1/`decision_score` (also structural, also already
rule-implementable via `flip_reward()`, explicitly marked
"already rule-based/causal, keep as-is" in the taxonomy) — not the same
kind of check as C3/`emotion`. `flip_score` stays on the text-only trunk,
grouped with `decision_score`, its actual structural sibling.
`price_strategy`/`progression` also stay text/state-grounded for the
original reason: they score objective negotiation facts (offer amounts,
offer count), not perceptual qualities, so audio grounding doesn't add
information there.

**What was rejected first, and why it matters to remember:** an earlier
design considered hand-crafted acoustic features (pitch, jitter, shimmer,
HNR — the same category of explicit numeric descriptor used by this
project's sibling bias-aware-negotiation work's VAII pipeline). That
approach was deliberately not used here: the project's own prior research
guidance flagged explicit hand-defined acoustic numbers as a real reviewer
risk (not a theoretically grounded feature set, easy to critique as
hand-engineering rather than genuine multimodal learning). This ruled out
Option A from the 2026-08-02 investigation before any encoder was chosen.

### 9.2 Encoder selection — audited, not assumed

Two candidates were empirically compared via a reusable "embedding
separation check" (calm vs. distressed buyer speech, cosine similarity
within-class vs. across-class, on this project's actual TTS-synthesized
negotiation audio — no public SER model has published TTS-domain
validation, so this check is required for any candidate, not optional):

| Candidate | Result |
|---|---|
| Qwen2.5-Omni's own `audio_tower` (mean/std/mean+std pooling, random content and content-matched pairs) | **Rejected** — separation gap ≤0.013 across every pooling strategy and both content conditions; content-matching *shrank* the already-tiny gap further, indicating whatever signal existed was content-driven, not affect-driven. General audio-*understanding* encoders are not guaranteed to encode paralinguistic affect. |
| `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` (dedicated continuous-dimensional SER model, arousal/dominance/valence output, trained on MSP-Podcast natural speech) | **Accepted** — direction-consistent separation on all 3 output dimensions, in *both* random-content and content-matched conditions (content-matching did NOT collapse the gap, the opposite of the audio_tower result) — the signature of a model responding to delivery, not lexical content. |

License: CC-BY-NC-SA-4.0 (research-only). Confirmed acceptable — this
project is pure academic research, exactly the licensed use case. Would
need re-litigating only if the project ever acquires a commercial angle.

Model card's custom `EmotionModel` class predates a `transformers` internal
(`all_tied_weights_keys`, now expected to be set via a post-init hook the
older snippet never calls) — fixed by setting it to `{}` explicitly in
`__init__` (correct value: this model has no tied weights, a regression
head is not an embedding-tied LM) — same category of environment-drift fix
as several others hit throughout this project, documented inline where
implemented, not hidden in a requirements pin.

### 9.3 Two important caveats to carry forward, not just note once

**The arousal-direction anomaly is a property of this dataset's TTS
delivery, not a universal truth about distress, and must not be assumed
elsewhere.** In the separation check, `arousal` ran calm > distressed
(diff ≈ -0.07), which is initially counterintuitive — distress is usually
associated with *higher* arousal. The likely explanation: this project's
"distressed" emotion labels skew toward low-energy/resigned states
(`resigned`, `defeated`, `weary`, `strained`) rather than high-energy panic,
and the TTS voice rendered those as subdued rather than agitated. This is a
property of *this dataset's specific emotion-label vocabulary and TTS
rendering*, not a general fact about how distress sounds. Anyone
interpreting the RM's `emotion` head output later, or writing the paper's
discussion of what it learned, must not assume the standard
arousal-equals-distress relationship holds — check against this dataset's
actual label distribution, not intuition.

**Effect size is real but modest — set training expectations accordingly.**
The strongest separation (valence, random-content) was ≈0.7-0.8σ, not an
overwhelming gap. If Bradley-Terry training on emotion-audio causal pairs
converges more slowly or noisily than the text-only emotion training did
(section 3's causal accuracy climbed to 94-99% within a few hundred steps),
that is not necessarily a bug to chase — it may simply reflect that audio
delivery is a genuinely subtler, harder signal than a pre-extracted text
label, which is exactly *why* this fusion is worth building (a trivial
signal wouldn't need grounding in the first place) but also why patience
during training/debugging is warranted before assuming something is broken.

### 9.4 Architecture: two parallel trunks, not one shared trunk with a bigger input

Sections 1-8's single shared trunk fed all 5 heads from one text embedding.
Adding audio to *that* trunk would mean every head — including
`decision_score`/`price_strategy`/`progression`/`flip_score`, none of which
have an established need to read buyer audio (see 9.1) — receives the audio
signal and has to learn to ignore it. That relies on training data to
enforce a scoping decision we already know we want, which cuts against this
project's whole design bias toward explicit, auditable structure over
implicit learned scoping (rubric vectors instead of a single scalar, a
fixed weighted sum instead of a learned combiner, format/decision gates
kept rule-based instead of learned). So: two parallel trunks, split by
which head needs audio.

```text
Text context+response                    Buyer audio clip (same turn
        |                                 prior_buyer_emotion already
        v                                 derives its text label from)
Qwen3-8B (frozen)                                  |
pooled hidden state                                v
   (hidden_dim,)                    audeering SER model (frozen)
        |                           (arousal, dominance, valence)
        |                                    (3,) — see 9.5 for why
        |                                    3 dims, not the richer
        |                                    internal embedding
        |                                          |
        |                    +---------------------+
        |                    |
        v                    v
  +-----------+      +--------------------+
  | Text-only |      | Text + audio       |
  | trunk     |      | fusion trunk       |
  | (existing,|      | (NEW) input =      |
  |  unchanged| |    | [text_hidden (3,)] |
  |  hidden_dim->     |  hidden_dim+3      |
  |  512->256)|      |  -> 512 -> 256     |
  +-----------+      +--------------------+
        |                          |
   +----+----+----+                |
   |         |     |                |
   v         v     v                v
decision  price_ flip_score      emotion
_score    strategy (C5,           (C3)
(C1,      progression diagnostic,
diagnostic)(C4)      structural —
                      see 9.1)
```

Both trunks stay small (same `hidden_dim -> 512 -> 256` shape, the fusion
trunk's input layer is just `hidden_dim+3` instead of `hidden_dim`) — the
parameter cost of a second trunk is negligible next to the frozen 8B
backbone, consistent with the "cheap to train, backbone does the heavy
lifting" property this RM has had since section 4.

**This split is itself a design choice made now, not a proven fact, and
should be stated as one rather than an implied given.** It assumes
`decision_score`/`price_strategy`/`progression`/`flip_score` genuinely have
zero benefit from audio — plausible (they score objective facts, not
perceptual qualities) but not audited the way the `emotion` encoder choice
was. If a future audit shows one of these four plateauing at a lower causal
accuracy than its text-only-era baseline (section 3's numbers), "maybe
delivery tone affects perceived legitimacy of this dimension too" is a
reasonable hypothesis to test then — not a sign the current split was
wrong, just a sign it was a starting assumption worth re-checking with
evidence, the same way the encoder choice itself was.

#### 9.4.1 Finding (2026-08-16 prototype): the trunk split eliminates `emotion~progression`'s v1 correlation, and this is structural, not a training-budget artifact

A same-scale (300 pairs, 1 epoch, same `--seed 42`, weight-init controlled —
see the seeding fix below) `audio_fusion=True` vs `audio_fusion=False`
comparison, audited on the same 150 val causal-pair contexts, gave:

| Pair | v1 (text-only) | v2 (audio_fusion) |
|---|---:|---:|
| decision_score~price_strategy | 0.916 | 0.973 |
| decision_score~progression | 0.713 | 0.775 |
| price_strategy~progression | 0.778 | 0.787 |
| decision_score~flip_score | −0.495 | −0.620 |
| price_strategy~flip_score | −0.620 | −0.680 |
| progression~flip_score | −0.385 | −0.477 |
| decision_score~emotion | 0.413 | 0.024 |
| price_strategy~emotion | 0.593 | 0.034 |
| **emotion~progression** | **0.537** | **−0.033** |
| emotion~flip_score | −0.288 | 0.036 |

The six pairs not involving `emotion` are essentially unchanged (same
ballpark, same sign) — the split has the intended contained blast radius,
not touching the other four heads' relationships to each other. But every
pair involving `emotion` collapses to ~0, including `emotion~progression`,
which was v1's *strongest* cross-dimension correlation and the one this
section's original text flagged as plausibly legitimate ("softening for an
upset buyer legitimately moves emotion + progression + price_strategy
together").

**Traced the mechanism in code rather than guessing from the number alone.**
`SERFusionHead.forward()` computes `scores = self.base(pooled_hidden)` (all
5 dims, unmodified v1 path) and then, only when `ser_vector is not None`,
*overwrites* `scores["emotion"]` with the fusion trunk's output — the
`self.base`-computed emotion tensor is discarded before it ever reaches a
loss, so it carries no gradient. `causal_pair_loss` only backprops through
the one dimension a pair targets (`FACTOR_TO_DIMENSION[pair["target_factor"]]`),
so for a C3_emotion pair with audio present, the loss touches
`fusion_trunk`'s parameters exclusively — `self.trunk` (the shared trunk
`progression`'s head reads from) receives **zero** gradient from
`emotion`'s causal supervision under `audio_fusion=True`. It still receives
spurious-pair gradient on its own `heads["emotion"]` (since spurious
contexts never carry `buyer_audio_path`, so `ser_vector=None` there), but
that's an invariance loss — training the trunk not to move on response-only
perturbations, not training it what buyer distress looks like. Checked the
reverse direction too: `causal_pair_progression` (C4) has only ever
constructed its context from `zone`/`buyer_offers`/`progression_stage` —
`progression` has never had emotion in its own supervision, directly.

**Conclusion:** v1's `emotion~progression` correlation was not evidence that
`progression` had learned a genuine emotion-progression relationship — it
was a multi-task representation-entanglement artifact of `self.trunk` being
gradient-shaped jointly by both heads' causal losses through one shared
256-dim bottleneck. `progression` inherited emotion-correlated structure for
free, because it read off the same trunk emotion's own loss was shaping.
The v2 split severs exactly this channel by design (that is what "clean
decoupling" in 9.4's opening paragraph means, concretely): `fusion_trunk`
and `self.trunk` are disjoint parameter sets, so there is no shared weight
left for a joint-training artifact to form through. This means **more
epochs or more C3_emotion training pairs will not restore this
correlation** — it would only improve `fusion_trunk`'s own C3 ranking
quality, not reopen a channel that no longer exists structurally. Ruled
this out empirically too before accepting it as the explanation: the
seeded comparison's causal ranking numbers for `decision_score`/`emotion`
were identical between runs and `price_strategy`/`progression` differed by
only ~3-5pp (consistent with ordinary bf16 forward-pass non-determinism,
not a training-signal difference), so the correlation collapse isn't
attributable to the two runs having learned differently in some broader
sense — it's specifically the trunk-sharing channel that's gone.

**Decision (2026-08-16): accept this as intended behavior, not a bug to
design around.** The alternative would be re-introducing a shared channel
between `fusion_trunk` and `self.trunk` (e.g. feeding the SER vector or
`fusion_trunk`'s output into `self.trunk`'s input too) specifically to
preserve a correlation that was never verified to reflect a real causal
relationship rather than a bottleneck artifact — that would be solving an
unconfirmed problem by quietly undoing the clean-decoupling property this
section exists to provide. If `progression` (or another head) should
genuinely be sensitive to buyer distress, the correct fix is to give it its
own direct supervision for that — e.g. extending `causal_pair_progression`
(C4) to vary emotion context the way `causal_pair_emotion` (C3) does — which
would be real, audited, causal signal rather than inherited entanglement.
That is a separate, later decision (not made here, not implied by this
finding) and should go through the same audit-before-implement discipline
as everything else in this section, not get bundled into the trunk-split
work as an implicit fix.

**Confirmed at full dataset scale (2026-08-16, same day, `v2_audio_fusion_full_20260816_152709`):**
1 epoch over the full 59,314 causal / 45,396 spurious training pairs
(`lambda_inv=1.5`, matching v1's validated hyperparameters, `seed=42`).
Val audit (1000 pairs): `emotion` causal ranking accuracy **100.0%**,
`decision_score` 99.4%, `flip_score` 100%, `price_strategy` 94.9%,
`progression` 90.7% — every dimension comfortably above chance, `emotion`
now exceeding v1's own 98.66% text-only baseline. Re-ran the pairwise
`head_correlation` check from the prototype (same method, 300 val
causal-pair contexts) specifically to see whether `emotion~progression`
would recover with ~350x the training data the prototype had for the
fusion trunk (only 40 C3_emotion pairs there vs. thousands at full scale):
it did not — **`emotion~progression` = 0.0053** at full scale, statistically
indistinguishable from the prototype's −0.033 and from zero. This is the
confirmation the mechanistic argument above predicted: more C3 data
improved `fusion_trunk`'s own ranking quality (94.3%→100% `emotion`
accuracy from prototype to full scale) but did nothing to reopen the
severed shared-trunk channel, because there is no such channel left to
reopen. The rest of the full-scale matrix matches the prototype's pattern
too — `emotion`'s correlation with all four other heads stays near zero
(0.086, −0.043, 0.005, 0.043) while non-`emotion` pairs retain real
structure (`decision_score~price_strategy` 0.636, `price_strategy~progression`
0.500, `progression~flip_score` −0.416) — the decoupling's blast radius is
still contained to `emotion` alone at scale, not just at prototype size.
This closes the "is this training-budget or structural" question with a
second, stronger data point rather than resting on the 300-pair result
alone.

**Methodological note worth keeping for the paper's rigor section:** the
first pass at this comparison (unseeded) showed spurious-invariance deltas
differing by 2-20x between the two runs on dimensions `audio_fusion` cannot
possibly touch (`decision_score`, `progression`, `flip_score`), which traced
back to `train_causal_rm.py::train()` never having called
`torch.manual_seed()` — `--seed` only controlled pair-shuffle order, not
head weight initialization, so "same seed" runs were silently starting from
different random heads. Fixed by seeding immediately before head
construction (not just once at the top of `train()`, since
`load_backbone()`/`load_ser_model()` consume the global RNG differently
depending on `audio_fusion` and would otherwise desync the two paths even
with a top-of-function seed). Verified the fix directly: `SERFusionHead(...).base`'s
state dict is now byte-identical to a standalone `RubricHead(...)`'s, with
`load_ser_model()`'s RNG consumption happening in between, exactly as it
does in a real `audio_fusion=True` run. Same class of mistake as the first
`lambda_inv` sweep (section 8.3) looking like a clean trend before the
missing-invariance-supervision gap was found — a controlled-comparison
claim needs to actually control everything being varied, and worth
independently checking for anywhere else in this codebase two "same-seed"
runs are compared.

### 9.5 Why 3 final dimensions (arousal/dominance/valence), not the SER model's internal embedding

The SER model's `forward()` (see section 9.2's model card usage) returns
both the 3 regression outputs *and* the pooled last-transformer-layer
hidden state (1024-dim) if requested. Chose the 3-dim output as the default
fusion input, not the richer embedding, for the same reason this project
has consistently preferred interpretable structure over black-box richness:
a 3-number vector with known semantics (arousal/dominance/valence) can be
logged, plotted, and sanity-checked the same way `factor_state`/`emotion`
fields already are — "this pair's fusion input changed valence from 0.54 to
0.42" is a checkable, reportable claim; "some norm of a 1024-dim embedding
changed" is not. This mirrors the fixed-weighted-sum-over-learned-combiner
decision in section 6 and the rubric-vector-over-scalar decision in
section 5 of `causal_rubric_taxonomy.md`. If the 3-dim version proves
insufficiently informative once real training data exists (i.e. `emotion`'s
causal accuracy plateaus well below the text-only baseline's 94-99%), the
1024-dim embedding is the documented fallback — not chosen by default,
kept here as the next thing to try if 3 dims underperforms.

### 9.6 What else needs to change (implementation, not yet built)

- **`causal_reward_model.py`**: add a `ser_model`/`ser_processor` pair
  (loaded the same frozen-backbone way as `load_backbone()`), a
  `SERFusionHead` variant of `RubricHead` implementing the dual-trunk split
  above, and thread an optional `buyer_audio_path` through `score()` —
  falls back to text-only behavior (current section 1-8 architecture,
  unchanged) when no audio path is available, so this is additive, not a
  breaking change to the existing text-only RM contract.
- **`train_causal_rm.py`**: `Embedder`-style caching for SER outputs (keyed
  by wav path, mirroring the existing text-embedding cache's rationale —
  many pairs reuse the same buyer clip via donor substitution) — a
  `SEREmbedder` sibling class, not a modification to the existing one, so
  causal_rm_pairs generated before this change still work unchanged for
  training the text-only heads.
- **`generate_intervention_pairs.py`**: `causal_pair_emotion` (C3) needs
  extending; **`causal_pair_flip` (C5) needs no change at all** — stated
  explicitly here so a future diff of this doc's history reads as a
  deliberate no-op, not an oversight: `flip_score` was in scope in the
  first draft of section 9, reconsidered and moved back to the text-only
  trunk per 9.1, and `causal_pair_flip`'s generation logic was never
  touched throughout that reconsideration and stays exactly as it is today.
  Extend `causal_pair_emotion`
  to also carry the donor's real buyer-audio wav path
  (`tts_outputs/<dialogue_id>/turn_<NN>_buyer.wav` — already exists for
  every turn, no new TTS generation needed) alongside the swapped text
  label, so context_a/context_b are audio-consistent with their own
  emotion label, not just text-consistent. Spurious pairs (S1-S3) need no
  change — they already hold context (and therefore buyer audio) fixed
  while varying only the seller response surface form, which is exactly
  the invariance property we want to test for the fusion trunk too.
- **Verification**: same discipline as sections 1-8 — small prototype run
  first (a few hundred pairs) via `run_causal_rm_experiment.py`, audit
  against the existing text-only baseline in `causal_rm_results.md` before
  any scale-up, watch `emotion`'s causal accuracy trend (expect slower/
  noisier convergence per 9.3, not necessarily a bug) rather than assuming
  parity with the text-only run's speed.
- **Verification, specifically for the two-trunk split**: section 8.3's
  `head_correlation` monitoring was built to catch unintended cross-head
  bleed through *one* shared trunk; with two trunks now, it needs to answer
  a two-sided question, not just "is correlation low." (a) Confirm the
  fusion trunk's two heads (`emotion`) and the text-only trunk's four heads
  don't show *new*, unexplained coupling that wasn't there in the v1
  single-trunk run — that would suggest an implementation bug (e.g. a
  leaked reference) rather than legitimate shared structure. (b) Equally
  check that splitting trunks didn't silently *break* correlation that was
  legitimate under the v1 single trunk — e.g. `emotion` and `progression`
  were expected to move together sometimes (softening for an upset buyer
  plausibly affects both), and that's a real domain relationship, not an
  artifact of sharing a trunk; confirm it's still visible now that they're
  on separate trunks, not just assume decoupling the architecture
  automatically decoupled the underlying relationship. Compare v2's
  `head_correlation` numbers against the v1 run's (`causal_rm_results.md`
  section 3 / the checkpoint's `training_history.json`) as the baseline for
  both checks, rather than evaluating the v2 numbers in isolation.
