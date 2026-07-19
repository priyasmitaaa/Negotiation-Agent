# RL Reward Function Design for v6 Negotiation Agent

**Generated:** 2026-06-20  
**Starting policy:** `sft_output_v6/20260614_010317/final_adapter`  
**Target format:** `<decision>` + `<response>`  
**Grounding files:**

- `sft_output_v6/20260614_010317/v6_error_analysis_text_and_speech_20260620_161120.md`
- `sft_output_v6/20260614_010317/v6_judge_text_and_speech_20260620_161122.json`
- `sft_output_v6/20260614_010317/v4_v5_v6_comparison_20260620.md`

---

## 1. Do We Need Complex Mathematical Formulations?

No. For this project, complex RL math is not the bottleneck. The bottleneck is designing rewards that match the actual negotiation failure modes.

Use simple, bounded components that are easy to inspect:

```text
reward = weighted_sum(component_rewards)
reward = clip(reward, -2.0, +2.0)
```

That is enough for GRPO/PPO-style training. We do not need a dense theoretical derivation unless we are publishing the method as a new RL algorithm. What we do need is:

- clear reward components
- bounded ranges
- no unbounded LLM-judge scores
- hard format gates
- logging each component separately
- ablations to prove each component helps

The only "math" worth writing is the implementation formula for the total reward and each component.

---

## 2. What the Analysis Says

### v6 Strengths

- Text decision accuracy: **98.3%** (`1876/1908`)
- Speech decision accuracy: **98.4%** (`1877/1908`)
- Format OK: **100%**
- Audio generated: **100%**
- Judge overall: **4.12 / 5.0**
- Judge negotiation quality: **4.53 / 5.0**
- Judge response naturalness: **4.41 / 5.0**

### v6 Weaknesses

From error analysis:

- Total decision errors: **32 / 1908**
- Dominant error: `MITIGATE -> LEVERAGE`, **24 / 32 errors**
- All decision errors are post-crystallisation: **32 / 1590**
- Flip-turn errors: **7 / 184**

From judge analysis:

- Emotion handling avg: **3.52 / 5.0**
- Emotion handling <= 3: **882 / 1908** (`46.2%`)
- Progression avg: **4.04 / 5.0**
- Progression <= 3: **632 / 1908** (`33.1%`)
- Naturalness is already strong; do not over-optimize it.

Therefore RL should mostly improve:

1. `MITIGATE -> LEVERAGE` errors
2. post-crystallisation strategy
3. flip-turn decisions
4. emotion handling
5. progression / non-repetition

---

## 3. Recommended Reward Overview

```text
R_total =
    1.00 * R_decision
  + 0.35 * R_price_strategy
  + 0.30 * R_emotion
  + 0.25 * R_progression
  + 0.15 * R_format
  + 0.10 * R_naturalness
  + 0.10 * R_audio_quality
  - 0.20 * P_repetition
  - 0.20 * P_language_impurity

R_total = clip(R_total, -2.0, +2.0)
```

Decision correctness is still the anchor, but because v6 is already 98%+, reward mass should now move toward emotion and progression.

---

## 4. Component Definitions

### 4.1 Format Reward

Purpose: preserve v6's clean 100% format success.

```text
R_format =
  +1.0 if exactly one valid <decision> and one valid <response>
  -1.0 if decision tag missing or invalid
  -0.5 if response tag missing
  -0.5 if extra <reasoning> appears in deployment-format training
```

Valid decisions:

```text
UNDECIDED, LEVERAGE, MITIGATE
```

This is a gate. If format is invalid, either set total reward to a strong negative or heavily downweight other rewards.

Recommended:

```text
if decision tag invalid:
    R_total = -2.0
```

### 4.2 Decision Reward

Purpose: keep high decision accuracy and target the remaining dominant error.

```text
if pred_decision == gt_decision:
    R_decision = +1.0
elif gt_decision == "MITIGATE" and pred_decision == "LEVERAGE":
    R_decision = -1.25
elif gt_decision == "LEVERAGE" and pred_decision == "MITIGATE":
    R_decision = -0.75
elif gt_decision == "UNDECIDED":
    R_decision = -1.0
else:
    R_decision = -1.0
```

Rationale:

- `MITIGATE -> LEVERAGE` is 75% of v6's errors.
- It is riskier because it may push too hard when the seller should soften.
- `LEVERAGE -> MITIGATE` is still wrong, but less frequent and usually less harmful.

### 4.3 Price Strategy Reward

Purpose: reward the response being tactically consistent with price state.

Inputs already available in inference/error analysis:

- `buyer_offers`
- `fair_value`
- `zone`
- `anchoring_strength`
- `decision_flip`
- `gt_decision`
- `pred_decision`

Define:

```text
best_offer = max(buyer_offers) if buyer_offers else None
price_gap = (best_offer - fair_value) / fair_value
```

Reward:

```text
R_price_strategy =
  +1.0 if decision is correct and response matches the decision tactically
  +0.5 if decision is correct but response is generic
  -0.5 if decision is correct but response language contradicts it
  -1.0 if decision is wrong in post_crystallisation
```

Tactical checks:

- `LEVERAGE`: response should be firm, justify value, resist lowballing.
- `MITIGATE`: response should soften, concede, invite closure, or acknowledge fair offer.
- `UNDECIDED`: response should gather/hold position without overcommitting.

Implementation can start rule-based:

```text
LEVERAGE keywords: firm, value, condition, market, worth, cannot, asking
MITIGATE keywords: can do, willing, meet, close, fair, work with, reduce, accept
UNDECIDED keywords: appreciate, asking, consider, tell me, offer
```

Then replace or combine with a small judge later.

### 4.4 Flip-Turn Bonus

Purpose: preserve and improve v6's strong flip handling.

```text
if decision_flip and pred_decision == gt_decision:
    R_flip = +0.5
elif decision_flip and pred_decision != gt_decision:
    R_flip = -0.5
else:
    R_flip = 0
```

Fold this into `R_price_strategy`, or keep it separate in logs. If separate:

```text
R_total += 0.20 * R_flip
```

### 4.5 Emotion Reward

Purpose: improve the weakest judge dimension.

v6 judge:

- emotion handling avg: **3.52**
- emotion handling <= 3: **46.2%**

Use buyer emotion labels, intensity, and valence from context.

```text
R_emotion =
  +1.0 if response acknowledges or adapts to buyer emotion appropriately
  +0.5 if response is emotionally safe but generic
   0.0 if no emotion signal is present
  -0.5 if response ignores high-intensity emotion
  -1.0 if response escalates negative/high-intensity emotion
```

Rule-based starter:

- For negative emotions (`frustrated`, `skeptical`, `hesitant`, `pleading`, `resigned`): reward calm acknowledgement and non-dismissive tone.
- For positive emotions (`eager`, `hopeful`, `agreeable`, `appreciative`): reward closure-oriented response.
- For high intensity: require some acknowledgement or softening.

Example detector:

```text
ack_words = ["understand", "appreciate", "fair", "I hear", "I get", "thanks"]
soften_words = ["work with", "meet", "can do", "let's", "I can"]
```

Do not require explicit emotion labels in the response. We want natural selling language, not robotic "I see you are frustrated".

### 4.6 Progression Reward

Purpose: improve the second weakest area.

v6 judge:

- progression avg: **4.04**
- progression <= 3: **33.1%**

```text
R_progression =
  +1.0 if response advances negotiation with a concrete next step
  +0.5 if response adds some new useful information
   0.0 if response is acceptable but static
  -0.5 if response only restates prior seller position
  -1.0 if response repeats previous turn with no movement
```

Concrete next steps:

- new counteroffer
- acceptance condition
- request for buyer's revised offer
- closing proposal
- pickup/payment/logistics suggestion after price is close

### 4.7 Repetition Penalty

Purpose: avoid repeated "fair market value" style phrasing.

```text
P_repetition = max_3gram_overlap(current_response, previous_seller_responses)
```

Simple bounded version:

```text
if overlap >= 0.50: P_repetition = 1.0
elif overlap >= 0.30: P_repetition = 0.5
else: P_repetition = 0.0
```

Only compare against prior seller turns in the same dialogue.

### 4.8 Naturalness Reward

Purpose: protect the already-good naturalness score without overfitting to judge style.

v6 naturalness is already **4.41 / 5.0**, so keep this low-weight.

```text
R_naturalness =
  +1.0 if response is fluent, concise, and seller-like
  +0.5 if acceptable
  -0.5 if awkward, too long, too short, or unnatural
  -1.0 if incoherent
```

Cheap rule starter:

```text
8 <= word_count <= 35        -> good
word_count < 5               -> too short
word_count > 60              -> too long
contains malformed tags      -> bad
```

### 4.9 Language Purity Penalty

Purpose: avoid non-English / stray Chinese tokens observed earlier.

```text
P_language_impurity =
  1.0 if response contains unwanted non-English script
  0.5 if response contains suspicious non-ASCII artifacts
  0.0 otherwise
```

Allow Indian currency symbols if needed:

```text
allowed_non_ascii = {"₹"}
```

### 4.10 Audio Quality Reward

Purpose: keep speech generation reliable and penalize crippled, clipped, broken, or non-smooth audio.

```text
R_audio_quality =
  +1.0 if WAV generated, valid, smooth, intelligible, and duration is plausible
  +0.5 if WAV generated with minor roughness but still intelligible
   0.0 if WAV is usable but noticeably low quality
  -0.5 if WAV has clipping, stutter, abrupt cuts, long silence, or non-smooth pacing
  -1.0 if WAV missing, empty, invalid, unintelligible, or severely crippled
```

Optional duration heuristic:

```text
expected_seconds ~= word_count / 2.3
valid if 0.4 * expected_seconds <= wav_duration <= 2.5 * expected_seconds
```

Audio quality checks to log separately:

```text
audio_exists
audio_duration_valid
audio_not_silent
audio_not_clipped
audio_no_abrupt_cutoff
audio_smoothness_ok
audio_intelligibility_ok
```

Starter signal definitions:

```text
audio_not_silent:
    rms_energy above a small threshold for most voiced regions

audio_not_clipped:
    low fraction of samples near max amplitude

audio_no_abrupt_cutoff:
    no sudden truncation at the end; final 100-300 ms decays naturally or ends cleanly

audio_smoothness_ok:
    no repeated waveform stalls, large discontinuities, robotic jumps, or choppy pacing

audio_intelligibility_ok:
    ASR transcript approximately matches the generated text, or a speech-quality judge marks it understandable
```

This should remain a guardrail reward, not the main RL objective. If speech quality becomes a frequent failure, split it into explicit penalties:

```text
P_audio_missing
P_audio_clipping
P_audio_stutter
P_audio_cutoff
P_audio_unintelligible
```

This reward applies only in speech-mode RL/evaluation. For text-only RL, omit it.

### 4.11 Current Audio Caveat

The completed v6 speech inference WAVs are technically valid, but subjectively degraded compared with v4/v5. The perceived failure is "pixelated" or rough audio: the files can be played, but they are not as smooth or listenable as the v4/v5 speech outputs.

Reward audit results from 2026-06-28:

| Version | N | Avg audio quality | Automatic audio failures |
|---|---:|---:|---|
| v4 | 540 | 0.9333 | duration only |
| v5 | 540 | 0.9231 | duration only |
| v6 | 1908 | 0.9148 | duration only |

The simple waveform checks did not detect the perceptual degradation. They found no silence, clipping, abrupt-cutoff, or smoothness failures; only duration-ratio mismatches. v6 did show lower average RMS/peak than v4/v5:

| Version | Avg RMS | Avg peak |
|---|---:|---:|
| v4 | 0.0791 | 0.6369 |
| v5 | 0.0781 | 0.6309 |
| v6 | 0.0536 | 0.4724 |

Implication:

- Do not trust the current `R_audio_quality` as the only judge of perceptual speech quality.
- Before speech-inclusive GRPO, regenerate a small v6 speech sample and compare by listening against v4/v5.
- Add a stronger perceptual audio-quality signal if audio smoothness becomes part of the RL objective.
- During GRPO, audio-quality reward is expensive enough that we should cache audio metrics instead of recomputing WAV statistics every sample.

Practical recommendation:

```text
Phase 1 GRPO: text-only reward, no audio generation in the loop.
Phase 2 GRPO/evaluation: regenerate speech for selected checkpoints and score audio offline.
Phase 3 speech-aware RL: only after audio generation quality and audio reward detection are reliable.
```

---

## 5. LLM Judge Usage

Use the LLM judge sparingly. It is useful for:

- emotion handling
- progression
- naturalness
- tactical consistency

But do not let it dominate. Judge scores are noisy and can be gamed.

Recommended conversion:

```text
R_judge_dim = (score - 3) / 2
```

This maps:

```text
1 -> -1.0
2 -> -0.5
3 ->  0.0
4 -> +0.5
5 -> +1.0
```

For online GRPO, run the judge on a subset or use a smaller distilled reward model. For offline reranking/evaluation, Qwen3-8B judge is fine.

---

## 6. Recommended v6 GRPO Reward Formula

Start with this:

```text
R_total =
    1.00 * R_decision
  + 0.30 * R_emotion
  + 0.25 * R_progression
  + 0.25 * R_price_strategy
  + 0.15 * R_format
  + 0.10 * R_naturalness
  - 0.20 * P_repetition
  - 0.20 * P_language_impurity

if speech_mode:
    R_total += 0.10 * R_audio_quality

if decision_flip:
    R_total += 0.20 * R_flip

R_total = clip(R_total, -2.0, +2.0)
```

Why these weights:

- Decision is still the main correctness signal.
- Emotion and progression are the biggest quality gaps in judge analysis.
- Price strategy targets post-crystallisation failures.
- Format/naturalness/audio quality are guardrails, not the main objective.
- Repetition and language impurity are penalties because they are failure modes, not goals.

---

## 7. Training Stages

### Stage A — Offline Reward Validation

Before RL, compute reward components on existing v6 inference outputs.

Expected sanity checks:

- correct decisions should have higher `R_decision`
- incorrect decisions should have lower total reward
- low judge emotion examples should have lower `R_emotion`
- repeated responses should be penalized
- total reward should not be mostly saturated at +2

### Stage B — GRPO on Existing Dialogue Turns

Use v6 as initialization.

For each prompt:

1. sample `G` candidate responses
2. parse `<decision>` and `<response>`
3. compute reward components
4. update with GRPO using group-relative normalized rewards

Recommended:

```text
G = 4 or 8
temperature = 0.7 initially
KL penalty to v6 reference
short run first: 200-500 update steps
```

### Stage C — Hard-Case Oversampling

Oversample examples with:

- post-crystallisation zone
- decision flip
- `MITIGATE` ground truth
- high/medium emotion intensity
- anchoring strengths that appeared in v6 errors

Do not oversample easy pre-crystallisation `UNDECIDED` turns; v6 is already perfect there.

### Stage D — Curriculum GRPO

Curriculum learning should be integrated through the data sampler and reward schedule, not by changing the output format.

Recommended curriculum:

```text
Phase 0: reward audit only
Phase 1: easy/stable turns
Phase 2: post-crystallisation turns
Phase 3: high-emotion and MITIGATE turns
Phase 4: decision-flip turns and known v6 error neighborhoods
Phase 5: mixed replay over the full train split
```

The curriculum should gradually increase the fraction of hard cases:

| Phase | Sampling focus | Hard-case ratio |
|---|---|---:|
| 1 | mostly correct/easy turns, format preservation | 0-10% |
| 2 | post-crystallisation strategy | 25-40% |
| 3 | high/medium emotion, negative emotion, MITIGATE | 40-60% |
| 4 | decision flips, v6 error-like cases | 60-80% |
| 5 | full mixed replay | 30-50% |

Hard-case tags can be derived from existing fields:

```text
zone == "post_crystallisation"
decision_flip == True
gt_decision == "MITIGATE"
prior_buyer_emotion.intensity in {"medium", "high"}
prior_buyer_emotion.valence == "negative"
price_gap close to decision boundary
dialogue/turn belongs to v6 error-analysis failure set
```

Reward weights can also be scheduled:

```text
Early:  high R_format, high R_decision, low exploration
Middle: increase R_emotion, R_progression, R_price_strategy
Late:   increase hard-case sampling, keep KL to v6 reference, mixed replay to prevent forgetting
```

Implementation requirements:

- Create a GRPO dataset builder that outputs prompts plus metadata needed by `compute_reward`.
- Add curriculum stage labels to each seller-turn example.
- Add a sampler that can change stage weights by step or epoch.
- Keep v6 as the reference policy for KL.
- Log reward components per batch, not just total reward.
- Save curriculum phase, sampled difficulty bucket, and reward components with each checkpoint.
- Keep validation fixed and non-curriculum so evaluation remains comparable.

### Stage E — Optional Self-Play Rollouts

After reward validation, add buyer simulator rollouts.

Reward final negotiation outcome:

- price within 10% of fair value
- no excessive concession below fair value
- buyer emotion does not deteriorate sharply
- seller closes when fair offer appears

Keep this as stage D, not stage A. It is powerful but easier to destabilize.

---

## 8. What Not to Do

- Do not make response length a major reward; response length is already stable.
- Do not optimize only judge overall; it can reward nice-sounding but tactically wrong responses.
- Do not add public `<reasoning>` back into v6 deployment format.
- Do not use harsh binary reward for every LEVERAGE/MITIGATE confusion; distinguish harmful `MITIGATE -> LEVERAGE`.
- Do not patch bad behavior at inference time with filters; train it through rewards and keep raw outputs auditable.

---

## 9. Minimal Implementation Checklist

- [ ] Parser for `<decision>` and `<response>`
- [ ] `R_format`
- [ ] `R_decision`
- [ ] `R_price_strategy`
- [ ] `R_emotion`
- [ ] `R_progression`
- [ ] `P_repetition`
- [ ] `P_language_impurity`
- [ ] Optional `R_audio_quality`
- [ ] Component-level logging
- [ ] Reward histogram per batch
- [ ] Ablations: decision-only, decision+emotion, decision+emotion+progression, full reward

---

## 10. Final Recommendation

Use v6 as the RL starting policy and keep the clean v4/v6 output format.

The first RL version should optimize:

1. decision correctness with asymmetric `MITIGATE -> LEVERAGE` penalty
2. emotion handling
3. progression
4. post-crystallisation price strategy
5. repetition and language purity guardrails

No complex mathematical formulation is required. The reward should be simple, bounded, auditable, and directly tied to observed v6 failures.
