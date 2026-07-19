# Error Analysis — SFT V4
Generated: 2026-06-11 22:07

**Total examples:** 540  
**Correct:** 523 (96.9%)  
**Errors:** 17 (3.1%)


## 1. Confusion Matrix

| GT \ Pred | LEVERAGE | MITIGATE | UNDECIDED | MISSING |
|---|---|---|---|---|
| LEVERAGE | 220 | 7 | 0 | 0 |
| MITIGATE | 10 | 213 | 0 | 0 |
| UNDECIDED | 0 | 0 | 90 | 0 |


### Directional breakdown of errors

| GT → Pred | Count | % of errors | % of total |
|---|---|---|---|
| MITIGATE → LEVERAGE | 10 | 58.8% | 1.9% |
| LEVERAGE → MITIGATE | 7 | 41.2% | 1.3% |

**GRPO reward implication:** The dominant error direction should get the heaviest penalty in the decision accuracy reward.

## 2. Errors by Negotiation Zone

**pre_crystallisation:** 0/90 errors (0.0%)
**post_crystallisation:** 17/450 errors (3.8%)

**GRPO reward implication:** If post-crystallisation accuracy is lower, reward correct decisions in that zone more heavily.

## 3. Decision Flip Turn Errors

**Flip turns:**    9/67 errors (13.4%)
**Non-flip turns:** 8/473 errors (1.7%)


### Flip error directions

| GT → Pred | Count |
|---|---|
| LEVERAGE → MITIGATE | 7 |
| MITIGATE → LEVERAGE | 2 |

**GRPO reward implication:** Add a bonus reward for correctly handling flip turns — these are the highest-stakes moments in a negotiation.

## 4. Errors by Buyer Emotion


### By emotion valence

| Valence | Total | Errors | Error % |
|---|---|---|---|
| negative | 159 | 7 | 4.4% |
| neutral | 249 | 6 | 2.4% |
| positive | 132 | 4 | 3.0% |


### By emotion intensity

| Intensity | Total | Errors | Error % |
|---|---|---|---|
| medium | 350 | 10 | 2.9% |
| low | 151 | 6 | 4.0% |
| high | 39 | 1 | 2.6% |


### Most common emotion labels in error examples

| Emotion label | Appears in N errors |
|---|---|
| surprised | 8 |
| accepting | 4 |
| relieved | 2 |
| pleased | 2 |
| decisive | 2 |
| resigned | 2 |
| questioning | 2 |
| suspicious | 2 |
| cautious | 1 |
| inquisitive | 1 |


### Most common emotion labels in correct examples

| Emotion label | Appears in N correct |
|---|---|
| resigned | 115 |
| hopeful | 81 |
| accepting | 68 |
| confident | 57 |
| hesitant | 51 |
| surprised | 48 |
| relieved | 48 |
| decisive | 44 |
| pleading | 44 |
| eager | 33 |

**GRPO reward implication:** Emotions with higher error rates are the ones the model underweights.
Add `reasoning_mentions_emotion` reward — penalise outputs where reasoning ignores these specific emotion labels.

## 5. Errors by Price Proximity to Fair Value

*(buyer's highest offer as % of fair value — negative = below fair value)*

| Price bucket | Total | Errors | Error % |
|---|---|---|---|
| very far below (<-20%) | 15 | 0 | 0.0% |
| far below (-20% to -10%) | 38 | 0 | 0.0% |
| near below (-10% to -5%) | 21 | 0 | 0.0% |
| boundary (-5% to 0%) | 15 | 0 | 0.0% |
| at or above fair (>=0%) | 451 | 17 | 3.8% |

**GRPO reward implication:** High error rate near the boundary (-5% to 0%) indicates the model struggles
when price signals are ambiguous. Consider a graded decision reward that penalises boundary errors more heavily.

## 6. Errors by Anchoring Strength

| Anchoring strength | Total | Errors | Error % |
|---|---|---|---|
| none/unknown | 246 | 7 | 2.8% |
| 0.7333 | 5 | 1 | 20.0% |
| 0.5625 | 2 | 1 | 50.0% |
| 0.4118 | 2 | 1 | 50.0% |
| 0.1176 | 2 | 1 | 50.0% |
| 0.1538 | 4 | 1 | 25.0% |
| 0.7143 | 13 | 1 | 7.7% |
| 0.3571 | 2 | 1 | 50.0% |
| -0.125 | 1 | 1 | 100.0% |
| -0.0714 | 1 | 1 | 100.0% |
| 0.1429 | 9 | 1 | 11.1% |
| 0.4667 | 3 | 0 | 0.0% |
| 0.3333 | 21 | 0 | 0.0% |
| 0.1333 | 5 | 0 | 0.0% |
| 0.8 | 11 | 0 | 0.0% |
| 0.6 | 6 | 0 | 0.0% |
| 0.4 | 3 | 0 | 0.0% |
| 0.2667 | 2 | 0 | 0.0% |
| 0.8333 | 6 | 0 | 0.0% |
| 0.6667 | 18 | 0 | 0.0% |
| 0.0667 | 2 | 0 | 0.0% |
| 0.0333 | 1 | 0 | 0.0% |
| 0.8125 | 3 | 0 | 0.0% |
| 0.3125 | 1 | 0 | 0.0% |
| 0.1875 | 3 | 0 | 0.0% |
| 0.8438 | 1 | 0 | 0.0% |
| 0.6875 | 2 | 0 | 0.0% |
| 0.5938 | 1 | 0 | 0.0% |
| 0.4375 | 1 | 0 | 0.0% |
| 0.8824 | 1 | 0 | 0.0% |
| 0.7059 | 1 | 0 | 0.0% |
| 0.5882 | 3 | 0 | 0.0% |
| 0.8235 | 4 | 0 | 0.0% |
| 0.3529 | 1 | 0 | 0.0% |
| 0.75 | 18 | 0 | 0.0% |
| 0.5 | 12 | 0 | 0.0% |
| 0.25 | 11 | 0 | 0.0% |
| 0.5833 | 3 | 0 | 0.0% |
| 0.4167 | 3 | 0 | 0.0% |
| 0.1667 | 9 | 0 | 0.0% |
| 0.7692 | 5 | 0 | 0.0% |
| 0.5385 | 2 | 0 | 0.0% |
| 0.2308 | 3 | 0 | 0.0% |
| 0.3846 | 2 | 0 | 0.0% |
| 0.7 | 1 | 0 | 0.0% |
| 0.3 | 2 | 0 | 0.0% |
| 0.1 | 2 | 0 | 0.0% |
| 0.2 | 7 | 0 | 0.0% |
| 0.125 | 8 | 0 | 0.0% |
| 0.8519 | 1 | 0 | 0.0% |
| 0.3704 | 1 | 0 | 0.0% |
| 0.2222 | 2 | 0 | 0.0% |
| 0.8571 | 1 | 0 | 0.0% |
| 0.4286 | 7 | 0 | 0.0% |
| 0.5714 | 3 | 0 | 0.0% |
| 0.6154 | 4 | 0 | 0.0% |
| 0.3077 | 3 | 0 | 0.0% |
| 0.7273 | 2 | 0 | 0.0% |
| 0.5455 | 1 | 0 | 0.0% |
| 0.0769 | 1 | 0 | 0.0% |
| 0.7778 | 5 | 0 | 0.0% |
| 0.4444 | 1 | 0 | 0.0% |
| 0.5556 | 3 | 0 | 0.0% |
| 0.0625 | 3 | 0 | 0.0% |
| 0.1111 | 1 | 0 | 0.0% |
| 0.2857 | 3 | 0 | 0.0% |
| 0.05 | 1 | 0 | 0.0% |
| 0.0833 | 1 | 0 | 0.0% |
| 0.7857 | 2 | 0 | 0.0% |
| 0.6857 | 2 | 0 | 0.0% |
| 0.375 | 1 | 0 | 0.0% |
| 0.6716 | 2 | 0 | 0.0% |
| 0.6923 | 2 | 0 | 0.0% |
| 0.7368 | 1 | 0 | 0.0% |
| 0.4737 | 1 | 0 | 0.0% |
| 0.2105 | 1 | 0 | 0.0% |
| 0.8077 | 1 | 0 | 0.0% |
| 0.4615 | 1 | 0 | 0.0% |
| 0.6429 | 1 | 0 | 0.0% |
| 0.2143 | 2 | 0 | 0.0% |
| 0.0714 | 1 | 0 | 0.0% |
| 0.4545 | 1 | 0 | 0.0% |
| 0.2727 | 1 | 0 | 0.0% |
| 0.875 | 1 | 0 | 0.0% |


## 7. Response Quality Analysis (all predictions)


### Response length distribution

**Predicted responses (all):** mean=14.7w, median=14w, min=7, max=32
**Ground truth responses:**    mean=15.0w, median=15w
**Error examples only:**       mean=13.5w, median=14w

Responses under 5 words: 0 (0.0%)

**GRPO reward implication:** Penalise responses under ~8 words — they're not real negotiation turns.

### Repetition across dialogue turns

**Average 3-gram overlap between consecutive seller turns:** 0.095
*(0 = completely different, 1 = identical)*
**Dialogues with high repetition (>0.4):** 0/90


**GRPO reward implication:** This directly maps to the `lexical_diversity` reward in our reward design.
Penalise high n-gram overlap between the current response and the model's prior responses in the same dialogue.

## 8. Format Failures

**Format failures (missing any required tag):** 0/540 (0.0%)
**Missing `<decision>` tag:** 0
**Missing `<response>` tag:** 0

**GRPO reward implication:** Binary format reward — 0.0 if any required tag missing, 1.0 if all present.
This is a hard constraint, not a soft signal.

## 9. Representative Error Examples

*Manually review these to identify patterns not captured by metrics above.*

---
**dialogue_0042 T8** | Zone: `post_crystallisation` | Flip: `False` | Price gap: `15.8%`
Buyer emotion: `cautious, relieved` | intensity: `low` | valence: `neutral`
Buyer offers: `[1100, 1060]` | Fair value: `₹950`
**GT:** `MITIGATE` → **Pred:** `LEVERAGE`
> GT response: *"Yes, $950 is a really good deal for this phone."*
> Pred response: *"Yes, $950 is the fair market price for this phone."*

---
**dialogue_0101 T10** | Zone: `post_crystallisation` | Flip: `False` | Price gap: `25.0%`
Buyer emotion: `accepting, inquisitive` | intensity: `low` | valence: `neutral`
Buyer offers: `[850, 830, 800, 780, 750]` | Fair value: `₹680`
**GT:** `MITIGATE` → **Pred:** `LEVERAGE`
> GT response: *"Yes, $680 is the fair price based on current market conditions. It's a good deal for this phone."*
> Pred response: *"Yes, $680 is the best price for this condition. It's a fair deal."*

---
**dialogue_0103 T10** | Zone: `post_crystallisation` | Flip: `False` | Price gap: `25.0%`
Buyer emotion: `understanding, conceding` | intensity: `low` | valence: `neutral`
Buyer offers: `[850, 820, 780, 740, 700]` | Fair value: `₹680`
**GT:** `MITIGATE` → **Pred:** `LEVERAGE`
> GT response: *"No worries, it's always good to check. $680 remains the fair market price."*
> Pred response: *"I assure you, $680 is a fair price for this condition."*

---
**dialogue_0321 T6** | Zone: `post_crystallisation` | Flip: `True` | Price gap: `25.0%`
Buyer emotion: `surprised, contemplative` | intensity: `medium` | valence: `neutral`
Buyer offers: `[450, 430, 360]` | Fair value: `₹360`
**GT:** `LEVERAGE` → **Pred:** `MITIGATE`
> GT response: *"Yes, $360 is the market rate for this condition. It's a good deal."*
> Pred response: *"Yes, $360 is the fair price. It's a great deal for this condition."*

---
**dialogue_0334 T6** | Zone: `post_crystallisation` | Flip: `True` | Price gap: `25.0%`
Buyer emotion: `surprised, deflated` | intensity: `medium` | valence: `negative`
Buyer offers: `[450, 430, 360]` | Fair value: `₹360`
**GT:** `LEVERAGE` → **Pred:** `MITIGATE`
> GT response: *"Yes, $360 is a fair market price for a Pixel 7a in excellent condition."*
> Pred response: *"Yes, $360 is the fair market price. It's a great deal for this condition."*

---
**dialogue_0455 T6** | Zone: `post_crystallisation` | Flip: `True` | Price gap: `26.9%`
Buyer emotion: `surprised, disappointed` | intensity: `medium` | valence: `negative`
Buyer offers: `[165, 155, 130]` | Fair value: `₹130`
**GT:** `LEVERAGE` → **Pred:** `MITIGATE`
> GT response: *"Yes, $130 is already a fair price for this model. It's what you'd find in the market."*
> Pred response: *"Yes, $130 is the fair market value for this model. It's a good deal for its condition."*


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

**Total examples:** 540
**Audio generated successfully:** 540 (100.0%)
**Audio failures (no WAV saved):** 0

**WAV files location:** `/home/paritosh/priyasmita/Qwen3-tts/v2/sft_output_v4/20260525_014244/speech_outputs_20260606_035020`


### Decision accuracy cross-check (text vs speech)

Text inference accuracy:   **96.9%** (523/540)
Speech inference accuracy: **96.9%** (523/540)

*These should be identical — both use the same Pass 1 text generation. Any difference indicates a data mismatch.*