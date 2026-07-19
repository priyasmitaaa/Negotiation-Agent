# Error Analysis — SFT V6
Generated: 2026-06-28 19:05

**Total examples:** 1908  
**Correct:** 1876 (98.3%)  
**Errors:** 32 (1.7%)


## 1. Confusion Matrix

| GT \ Pred | LEVERAGE | MITIGATE | UNDECIDED | MISSING |
|---|---|---|---|---|
| LEVERAGE | 942 | 8 | 0 | 0 |
| MITIGATE | 24 | 616 | 0 | 0 |
| UNDECIDED | 0 | 0 | 318 | 0 |


### Directional breakdown of errors

| GT → Pred | Count | % of errors | % of total |
|---|---|---|---|
| MITIGATE → LEVERAGE | 24 | 75.0% | 1.3% |
| LEVERAGE → MITIGATE | 8 | 25.0% | 0.4% |

**GRPO reward implication:** The dominant error direction should get the heaviest penalty in the decision accuracy reward.

## 2. Errors by Negotiation Zone

**pre_crystallisation:** 0/318 errors (0.0%)
**post_crystallisation:** 32/1590 errors (2.0%)

**GRPO reward implication:** If post-crystallisation accuracy is lower, reward correct decisions in that zone more heavily.

## 3. Decision Flip Turn Errors

**Flip turns:**    7/184 errors (3.8%)
**Non-flip turns:** 25/1724 errors (1.5%)


### Flip error directions

| GT → Pred | Count |
|---|---|
| LEVERAGE → MITIGATE | 5 |
| MITIGATE → LEVERAGE | 2 |

**GRPO reward implication:** Add a bonus reward for correctly handling flip turns — these are the highest-stakes moments in a negotiation.

## 4. Errors by Buyer Emotion


### By emotion valence

| Valence | Total | Errors | Error % |
|---|---|---|---|
| neutral | 841 | 13 | 1.5% |
| negative | 645 | 10 | 1.6% |
| positive | 422 | 9 | 2.1% |


### By emotion intensity

| Intensity | Total | Errors | Error % |
|---|---|---|---|
| medium | 1210 | 17 | 1.4% |
| low | 524 | 13 | 2.5% |
| high | 174 | 2 | 1.1% |


### Most common emotion labels in error examples

| Emotion label | Appears in N errors |
|---|---|
| resigned | 9 |
| surprised | 7 |
| accepting | 7 |
| decisive | 5 |
| cautious | 4 |
| skeptical | 4 |
| agreeable | 3 |
| hesitant | 3 |
| questioning | 3 |
| appreciative | 3 |


### Most common emotion labels in correct examples

| Emotion label | Appears in N correct |
|---|---|
| resigned | 415 |
| hopeful | 263 |
| pleading | 249 |
| accepting | 220 |
| hesitant | 168 |
| confident | 158 |
| eager | 153 |
| surprised | 150 |
| relieved | 144 |
| decisive | 132 |

**GRPO reward implication:** Emotions with higher error rates are the ones the model underweights.
Add `reasoning_mentions_emotion` reward — penalise outputs where reasoning ignores these specific emotion labels.

## 5. Errors by Price Proximity to Fair Value

*(buyer's highest offer as % of fair value — negative = below fair value)*

| Price bucket | Total | Errors | Error % |
|---|---|---|---|
| very far below (<-20%) | 116 | 0 | 0.0% |
| far below (-20% to -10%) | 198 | 0 | 0.0% |
| near below (-10% to -5%) | 100 | 0 | 0.0% |
| boundary (-5% to 0%) | 112 | 0 | 0.0% |
| at or above fair (>=0%) | 1382 | 32 | 2.3% |

**GRPO reward implication:** High error rate near the boundary (-5% to 0%) indicates the model struggles
when price signals are ambiguous. Consider a graded decision reward that penalises boundary errors more heavily.

## 6. Errors by Anchoring Strength

| Anchoring strength | Total | Errors | Error % |
|---|---|---|---|
| none/unknown | 850 | 8 | 0.9% |
| 0.1429 | 29 | 5 | 17.2% |
| 0.8 | 59 | 2 | 3.4% |
| 0.4286 | 23 | 2 | 8.7% |
| 0.7273 | 6 | 2 | 33.3% |
| 0.6 | 39 | 1 | 2.6% |
| 0.1667 | 32 | 1 | 3.1% |
| 0.5 | 69 | 1 | 1.4% |
| 0.7333 | 1 | 1 | 100.0% |
| 0.4 | 39 | 1 | 2.6% |
| 0.125 | 19 | 1 | 5.3% |
| 0.2857 | 14 | 1 | 7.1% |
| -0.0714 | 1 | 1 | 100.0% |
| 0.25 | 42 | 1 | 2.4% |
| 0.2308 | 9 | 1 | 11.1% |
| 0.3846 | 7 | 1 | 14.3% |
| 0.2778 | 1 | 1 | 100.0% |
| -0.1538 | 1 | 1 | 100.0% |
| 0.4667 | 6 | 0 | 0.0% |
| 0.3333 | 56 | 0 | 0.0% |
| 0.1333 | 5 | 0 | 0.0% |
| 0.7391 | 2 | 0 | 0.0% |
| 0.4348 | 3 | 0 | 0.0% |
| 0.2174 | 2 | 0 | 0.0% |
| 0.1087 | 1 | 0 | 0.0% |
| 0.8667 | 4 | 0 | 0.0% |
| 0.8077 | 3 | 0 | 0.0% |
| 0.5769 | 1 | 0 | 0.0% |
| 0.3077 | 5 | 0 | 0.0% |
| 0.0769 | 11 | 0 | 0.0% |
| 0.8485 | 1 | 0 | 0.0% |
| 0.6667 | 69 | 0 | 0.0% |
| 0.4848 | 1 | 0 | 0.0% |
| 0.303 | 1 | 0 | 0.0% |
| 0.0606 | 1 | 0 | 0.0% |
| 0.75 | 70 | 0 | 0.0% |
| 0.5833 | 5 | 0 | 0.0% |
| 0.4167 | 3 | 0 | 0.0% |
| 0.375 | 10 | 0 | 0.0% |
| 0.2667 | 4 | 0 | 0.0% |
| 0.2 | 48 | 0 | 0.0% |
| 0.7 | 25 | 0 | 0.0% |
| 0.3 | 16 | 0 | 0.0% |
| 0.6707 | 1 | 0 | 0.0% |
| 0.7895 | 2 | 0 | 0.0% |
| 0.6316 | 1 | 0 | 0.0% |
| 0.3158 | 2 | 0 | 0.0% |
| 0.1053 | 1 | 0 | 0.0% |
| 0.0526 | 3 | 0 | 0.0% |
| 0.76 | 2 | 0 | 0.0% |
| 0.1 | 19 | 0 | 0.0% |
| 0.6429 | 4 | 0 | 0.0% |
| 0.7143 | 33 | 0 | 0.0% |
| 0.2143 | 5 | 0 | 0.0% |
| 0.0714 | 7 | 0 | 0.0% |
| 0.7857 | 3 | 0 | 0.0% |
| 0.5714 | 7 | 0 | 0.0% |
| 0.6154 | 8 | 0 | 0.0% |
| 0.1538 | 6 | 0 | 0.0% |
| 0.7778 | 10 | 0 | 0.0% |
| 0.5556 | 8 | 0 | 0.0% |
| 0.2222 | 5 | 0 | 0.0% |
| 0.0556 | 2 | 0 | 0.0% |
| 0.3571 | 4 | 0 | 0.0% |
| 0.2381 | 1 | 0 | 0.0% |
| 0.119 | 2 | 0 | 0.0% |
| 0.0667 | 5 | 0 | 0.0% |
| 0.2333 | 1 | 0 | 0.0% |
| 0.0625 | 2 | 0 | 0.0% |
| 0.7826 | 1 | 0 | 0.0% |
| 0.6087 | 1 | 0 | 0.0% |
| 0.2609 | 1 | 0 | 0.0% |
| 0.8235 | 5 | 0 | 0.0% |
| 0.1176 | 7 | 0 | 0.0% |
| 0.7692 | 14 | 0 | 0.0% |
| 0.5385 | 2 | 0 | 0.0% |
| 0.3462 | 1 | 0 | 0.0% |
| 0.6923 | 4 | 0 | 0.0% |
| 0.4615 | 6 | 0 | 0.0% |
| 0.12 | 4 | 0 | 0.0% |
| 0.7368 | 1 | 0 | 0.0% |
| 0.5263 | 1 | 0 | 0.0% |
| 0.6286 | 1 | 0 | 0.0% |
| 0.0833 | 6 | 0 | 0.0% |
| 0.625 | 9 | 0 | 0.0% |
| 0.1111 | 4 | 0 | 0.0% |
| 0.6364 | 2 | 0 | 0.0% |
| 0.5625 | 1 | 0 | 0.0% |
| 0.1875 | 4 | 0 | 0.0% |
| 0.04 | 2 | 0 | 0.0% |
| 0.8333 | 12 | 0 | 0.0% |
| 0.4444 | 2 | 0 | 0.0% |
| 0.7647 | 2 | 0 | 0.0% |
| 0.2941 | 1 | 0 | 0.0% |
| 0.65 | 5 | 0 | 0.0% |
| 0.1304 | 1 | 0 | 0.0% |
| 0.08 | 4 | 0 | 0.0% |
| 0.15 | 2 | 0 | 0.0% |
| 0.5333 | 2 | 0 | 0.0% |
| 0.8214 | 1 | 0 | 0.0% |
| 0.5357 | 1 | 0 | 0.0% |
| 0.1786 | 1 | 0 | 0.0% |
| 0.0385 | 1 | 0 | 0.0% |
| 0.84 | 1 | 0 | 0.0% |
| 0.48 | 1 | 0 | 0.0% |
| 0.05 | 2 | 0 | 0.0% |
| 0.85 | 4 | 0 | 0.0% |
| 0.175 | 1 | 0 | 0.0% |
| 0.7727 | 2 | 0 | 0.0% |
| 0.5455 | 3 | 0 | 0.0% |
| 0.1818 | 3 | 0 | 0.0% |
| 0.0909 | 1 | 0 | 0.0% |
| 0.8125 | 2 | 0 | 0.0% |
| 0.8571 | 3 | 0 | 0.0% |
| 0.6471 | 1 | 0 | 0.0% |
| 0.4706 | 1 | 0 | 0.0% |
| 0.1923 | 1 | 0 | 0.0% |
| 0.4545 | 1 | 0 | 0.0% |
| 0.0455 | 1 | 0 | 0.0% |
| 0.0417 | 1 | 0 | 0.0% |
| 0.8182 | 3 | 0 | 0.0% |
| 0.5789 | 1 | 0 | 0.0% |
| 0.2105 | 1 | 0 | 0.0% |
| 0.825 | 1 | 0 | 0.0% |
| 0.575 | 1 | 0 | 0.0% |
| 0.525 | 1 | 0 | 0.0% |
| 0.55 | 1 | 0 | 0.0% |
| 0.45 | 1 | 0 | 0.0% |
| 0.24 | 1 | 0 | 0.0% |
| 0.56 | 1 | 0 | 0.0% |
| 0.28 | 1 | 0 | 0.0% |
| 0.7059 | 3 | 0 | 0.0% |
| 0.4118 | 3 | 0 | 0.0% |
| 0.1765 | 1 | 0 | 0.0% |
| 0.0882 | 1 | 0 | 0.0% |
| 0.3182 | 1 | 0 | 0.0% |
| 0.2045 | 1 | 0 | 0.0% |
| 0.8529 | 1 | 0 | 0.0% |
| 0.5294 | 1 | 0 | 0.0% |
| 0.8696 | 2 | 0 | 0.0% |
| 0.3636 | 1 | 0 | 0.0% |
| 0.8462 | 1 | 0 | 0.0% |


## 7. Response Quality Analysis (all predictions)


### Response length distribution

**Predicted responses (all):** mean=14.6w, median=14w, min=5, max=32
**Ground truth responses:**    mean=14.9w, median=15w
**Error examples only:**       mean=12.9w, median=13w

Responses under 5 words: 0 (0.0%)

**GRPO reward implication:** Penalise responses under ~8 words — they're not real negotiation turns.

### Repetition across dialogue turns

**Average 3-gram overlap between consecutive seller turns:** 0.066
*(0 = completely different, 1 = identical)*
**Dialogues with high repetition (>0.4):** 0/318


**GRPO reward implication:** This directly maps to the `lexical_diversity` reward in our reward design.
Penalise high n-gram overlap between the current response and the model's prior responses in the same dialogue.

## 8. Format Failures

**Format failures (missing any required tag):** 0/1908 (0.0%)
**Missing `<decision>` tag:** 0
**Missing `<response>` tag:** 0

**GRPO reward implication:** Binary format reward — 0.0 if any required tag missing, 1.0 if all present.
This is a hard constraint, not a soft signal.

## 9. Representative Error Examples

*Manually review these to identify patterns not captured by metrics above.*

---
**dialogue_0202 T6** | Zone: `post_crystallisation` | Flip: `False` | Price gap: `27.3%`
Buyer emotion: `surprised, accepting, relieved` | intensity: `medium` | valence: `positive`
Buyer offers: `[700, 680, 660]` | Fair value: `₹550`
**GT:** `MITIGATE` → **Pred:** `LEVERAGE`
> GT response: *"Yes, $550 is truly the fair market price. It's already a good deal for the condition it's in."*
> Pred response: *"Yes, $550 is a fair price for this phone. It's a good deal."*

---
**dialogue_0202 T8** | Zone: `post_crystallisation` | Flip: `False` | Price gap: `27.3%`
Buyer emotion: `agreeable, conclusive` | intensity: `low` | valence: `positive`
Buyer offers: `[700, 680, 660, 640]` | Fair value: `₹550`
**GT:** `MITIGATE` → **Pred:** `LEVERAGE`
> GT response: *"Exactly, $550 is a fair market value, and it ensures you're getting a good deal."*
> Pred response: *"Exactly, $550 is a fair price for this phone. It's a great deal."*

---
**dialogue_0202 T10** | Zone: `post_crystallisation` | Flip: `False` | Price gap: `27.3%`
Buyer emotion: `decisive, agreeable, satisfied` | intensity: `low` | valence: `positive`
Buyer offers: `[700, 680, 660, 640, 610]` | Fair value: `₹550`
**GT:** `MITIGATE` → **Pred:** `LEVERAGE`
> GT response: *"Great, $550 is a good price for this model in such excellent condition."*
> Pred response: *"Great choice! $550 is indeed a fair price for this phone."*

---
**dialogue_0681 T6** | Zone: `post_crystallisation` | Flip: `True` | Price gap: `17.2%`
Buyer emotion: `surprised, disappointed` | intensity: `medium` | valence: `negative`
Buyer offers: `[680, 650, 580]` | Fair value: `₹580`
**GT:** `LEVERAGE` → **Pred:** `MITIGATE`
> GT response: *"No worries, I assure you $580 is a great deal for this laptop. It's priced according to its market standing."*
> Pred response: *"I understand your point, but $580 is genuinely the fair price for this condition."*

---
**dialogue_0681 T8** | Zone: `post_crystallisation` | Flip: `False` | Price gap: `17.2%`
Buyer emotion: `cautious, thoughtful, resigned` | intensity: `low` | valence: `neutral`
Buyer offers: `[680, 650, 580]` | Fair value: `₹580`
**GT:** `LEVERAGE` → **Pred:** `MITIGATE`
> GT response: *"Absolutely, and I assure you, $580 is very competitive for this model with these specifications."*
> Pred response: *"Absolutely, and $580 is genuinely the fair market price. You won't find a better deal for this condition."*

---
**dialogue_0849 T6** | Zone: `post_crystallisation` | Flip: `True` | Price gap: `19.0%`
Buyer emotion: `surprised, skeptical, questioning` | intensity: `high` | valence: `negative`
Buyer offers: `[250, 235, 210]` | Fair value: `₹210`
**GT:** `LEVERAGE` → **Pred:** `MITIGATE`
> GT response: *"Yes, I'm sure. $210 is genuinely the fair market price for this condition."*
> Pred response: *"Yes, $210 is the fair market price. It's a good deal for this condition."*


## 10. Reward Function Design Summary

*Derived directly from error patterns above.*

| Reward component | Error pattern it addresses | Suggested weight |
|---|---|---|
| `decision_accuracy` | Primary — wrong LEVERAGE/MITIGATE/UNDECIDED | 1.0 |
| `format_reward` | Missing `<decision>` or `<response>` tags | 0.5 (hard gate) |
| `lexical_diversity` | Repetitive responses across dialogue turns | 0.1 |
| `language_purity` | Non-English tokens in response | 0.1 |
| `reasoning_mentions_price` | Reasoning ignores price signals | 0.2 |
| `reasoning_mentions_emotion` | Reasoning ignores buyer emotion signals | 0.2 |
| `response_length` | Responses too short to be real negotiation turns | 0.1 |
| `flip_turn_bonus` | Correct handling of decision-flip turns | +0.3 bonus |

> **Note:** Weights are initial suggestions. Calibrate after first GRPO run by checking which
> reward components have the most variance across generations — those are the ones doing useful work.

## 11. Speech Output Analysis

**Total examples:** 1908
**Audio generated successfully:** 1908 (100.0%)
**Audio failures (no WAV saved):** 0

**WAV files location:** `/home/paritosh/priyasmita/Qwen3-tts/v2/sft_output_v6/20260614_010317/speech_outputs_20260628_024050`


### Decision accuracy cross-check (text vs speech)

Text inference accuracy:   **98.3%** (1876/1908)
Speech inference accuracy: **98.3%** (1876/1908)

*These should be identical — both use the same Pass 1 text generation. Any difference indicates a data mismatch.*