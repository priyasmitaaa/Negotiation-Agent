# Error Analysis — SFT V5-SPEECH
Generated: 2026-06-13 11:47

**Total examples:** 540  
**Correct:** 499 (92.4%)  
**Errors:** 41 (7.6%)


## 1. Confusion Matrix

| GT \ Pred | LEVERAGE | MITIGATE | UNDECIDED | MISSING |
|---|---|---|---|---|
| LEVERAGE | 221 | 6 | 0 | 0 |
| MITIGATE | 35 | 188 | 0 | 0 |
| UNDECIDED | 0 | 0 | 90 | 0 |


### Directional breakdown of errors

| GT → Pred | Count | % of errors | % of total |
|---|---|---|---|
| MITIGATE → LEVERAGE | 35 | 85.4% | 6.5% |
| LEVERAGE → MITIGATE | 6 | 14.6% | 1.1% |

**GRPO reward implication:** The dominant error direction should get the heaviest penalty in the decision accuracy reward.

## 2. Errors by Negotiation Zone

**pre_crystallisation:** 0/90 errors (0.0%)
**post_crystallisation:** 41/450 errors (9.1%)

**GRPO reward implication:** If post-crystallisation accuracy is lower, reward correct decisions in that zone more heavily.

## 3. Decision Flip Turn Errors

**Flip turns:**    5/67 errors (7.5%)
**Non-flip turns:** 36/473 errors (7.6%)


### Flip error directions

| GT → Pred | Count |
|---|---|
| LEVERAGE → MITIGATE | 5 |

**GRPO reward implication:** Add a bonus reward for correctly handling flip turns — these are the highest-stakes moments in a negotiation.

## 4. Errors by Buyer Emotion


### By emotion valence

| Valence | Total | Errors | Error % |
|---|---|---|---|
| negative | 159 | 19 | 11.9% |
| neutral | 249 | 11 | 4.4% |
| positive | 132 | 11 | 8.3% |


### By emotion intensity

| Intensity | Total | Errors | Error % |
|---|---|---|---|
| medium | 350 | 20 | 5.7% |
| low | 151 | 14 | 9.3% |
| high | 39 | 7 | 17.9% |


### Most common emotion labels in error examples

| Emotion label | Appears in N errors |
|---|---|
| resigned | 15 |
| hopeful | 7 |
| surprised | 6 |
| conceding | 6 |
| questioning | 5 |
| relieved | 5 |
| accepting | 5 |
| hesitant | 3 |
| urgent | 3 |
| earnest | 3 |


### Most common emotion labels in correct examples

| Emotion label | Appears in N correct |
|---|---|
| resigned | 102 |
| hopeful | 74 |
| accepting | 67 |
| confident | 58 |
| surprised | 50 |
| hesitant | 49 |
| relieved | 45 |
| decisive | 43 |
| pleading | 42 |
| eager | 31 |

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
| at or above fair (>=0%) | 451 | 41 | 9.1% |

**GRPO reward implication:** High error rate near the boundary (-5% to 0%) indicates the model struggles
when price signals are ambiguous. Consider a graded decision reward that penalises boundary errors more heavily.

## 6. Errors by Anchoring Strength

| Anchoring strength | Total | Errors | Error % |
|---|---|---|---|
| none/unknown | 246 | 6 | 2.4% |
| 0.1429 | 9 | 6 | 66.7% |
| 0.25 | 11 | 4 | 36.4% |
| 0.2 | 7 | 4 | 57.1% |
| 0.1667 | 9 | 3 | 33.3% |
| 0.125 | 8 | 3 | 37.5% |
| 0.3333 | 21 | 2 | 9.5% |
| 0.4 | 3 | 1 | 33.3% |
| 0.2667 | 2 | 1 | 50.0% |
| 0.0667 | 2 | 1 | 50.0% |
| 0.3 | 2 | 1 | 50.0% |
| 0.3571 | 2 | 1 | 50.0% |
| 0.4286 | 7 | 1 | 14.3% |
| 0.5556 | 3 | 1 | 33.3% |
| 0.0625 | 3 | 1 | 33.3% |
| 0.1111 | 1 | 1 | 100.0% |
| 0.05 | 1 | 1 | 100.0% |
| 0.2105 | 1 | 1 | 100.0% |
| 0.2143 | 2 | 1 | 50.0% |
| 0.0714 | 1 | 1 | 100.0% |
| 0.7333 | 5 | 0 | 0.0% |
| 0.4667 | 3 | 0 | 0.0% |
| 0.1333 | 5 | 0 | 0.0% |
| 0.8 | 11 | 0 | 0.0% |
| 0.6 | 6 | 0 | 0.0% |
| 0.8333 | 6 | 0 | 0.0% |
| 0.6667 | 18 | 0 | 0.0% |
| 0.0333 | 1 | 0 | 0.0% |
| 0.8125 | 3 | 0 | 0.0% |
| 0.5625 | 2 | 0 | 0.0% |
| 0.3125 | 1 | 0 | 0.0% |
| 0.1875 | 3 | 0 | 0.0% |
| 0.8438 | 1 | 0 | 0.0% |
| 0.6875 | 2 | 0 | 0.0% |
| 0.5938 | 1 | 0 | 0.0% |
| 0.4375 | 1 | 0 | 0.0% |
| 0.8824 | 1 | 0 | 0.0% |
| 0.7059 | 1 | 0 | 0.0% |
| 0.5882 | 3 | 0 | 0.0% |
| 0.4118 | 2 | 0 | 0.0% |
| 0.8235 | 4 | 0 | 0.0% |
| 0.3529 | 1 | 0 | 0.0% |
| 0.1176 | 2 | 0 | 0.0% |
| 0.75 | 18 | 0 | 0.0% |
| 0.5 | 12 | 0 | 0.0% |
| 0.5833 | 3 | 0 | 0.0% |
| 0.4167 | 3 | 0 | 0.0% |
| 0.7692 | 5 | 0 | 0.0% |
| 0.5385 | 2 | 0 | 0.0% |
| 0.2308 | 3 | 0 | 0.0% |
| 0.3846 | 2 | 0 | 0.0% |
| 0.1538 | 4 | 0 | 0.0% |
| 0.7 | 1 | 0 | 0.0% |
| 0.1 | 2 | 0 | 0.0% |
| 0.8519 | 1 | 0 | 0.0% |
| 0.3704 | 1 | 0 | 0.0% |
| 0.2222 | 2 | 0 | 0.0% |
| 0.8571 | 1 | 0 | 0.0% |
| 0.7143 | 13 | 0 | 0.0% |
| -0.125 | 1 | 0 | 0.0% |
| -0.0714 | 1 | 0 | 0.0% |
| 0.5714 | 3 | 0 | 0.0% |
| 0.6154 | 4 | 0 | 0.0% |
| 0.3077 | 3 | 0 | 0.0% |
| 0.7273 | 2 | 0 | 0.0% |
| 0.5455 | 1 | 0 | 0.0% |
| 0.0769 | 1 | 0 | 0.0% |
| 0.7778 | 5 | 0 | 0.0% |
| 0.4444 | 1 | 0 | 0.0% |
| 0.2857 | 3 | 0 | 0.0% |
| 0.0833 | 1 | 0 | 0.0% |
| 0.7857 | 2 | 0 | 0.0% |
| 0.6857 | 2 | 0 | 0.0% |
| 0.375 | 1 | 0 | 0.0% |
| 0.6716 | 2 | 0 | 0.0% |
| 0.6923 | 2 | 0 | 0.0% |
| 0.7368 | 1 | 0 | 0.0% |
| 0.4737 | 1 | 0 | 0.0% |
| 0.8077 | 1 | 0 | 0.0% |
| 0.4615 | 1 | 0 | 0.0% |
| 0.6429 | 1 | 0 | 0.0% |
| 0.4545 | 1 | 0 | 0.0% |
| 0.2727 | 1 | 0 | 0.0% |
| 0.875 | 1 | 0 | 0.0% |


## 7. Response Quality Analysis (all predictions)


### Response length distribution

**Predicted responses (all):** mean=14.6w, median=14w, min=5, max=31
**Ground truth responses:**    mean=15.0w, median=15w
**Error examples only:**       mean=14.3w, median=14w

Responses under 5 words: 0 (0.0%)

**GRPO reward implication:** Penalise responses under ~8 words — they're not real negotiation turns.

### Repetition across dialogue turns

**Average 3-gram overlap between consecutive seller turns:** 0.071
*(0 = completely different, 1 = identical)*
**Dialogues with high repetition (>0.4):** 0/90


**GRPO reward implication:** This directly maps to the `lexical_diversity` reward in our reward design.
Penalise high n-gram overlap between the current response and the model's prior responses in the same dialogue.

## 8. Format Failures

**Format failures (missing any required tag):** 240/540 (44.4%)
**Missing `<decision>` tag:** 0
**Missing `<response>` tag:** 0

**GRPO reward implication:** Binary format reward — 0.0 if any required tag missing, 1.0 if all present.
This is a hard constraint, not a soft signal.

## 9. Representative Error Examples

*Manually review these to identify patterns not captured by metrics above.*

---
**dialogue_0337 T6** | Zone: `post_crystallisation` | Flip: `False` | Price gap: `25.0%`
Buyer emotion: `hesitant, questioning` | intensity: `low` | valence: `neutral`
Buyer offers: `[450, 430, 410]` | Fair value: `₹360`
**GT:** `MITIGATE` → **Pred:** `LEVERAGE`
> GT response: *"I understand where you're coming from, but $365 is already a competitive price for this condition."*
> Pred response: *"I understand your position, but $360 is the fair market rate. I can offer it at $370."*

---
**dialogue_0337 T8** | Zone: `post_crystallisation` | Flip: `False` | Price gap: `25.0%`
Buyer emotion: `eager, urgent` | intensity: `high` | valence: `positive`
Buyer offers: `[450, 430, 410, 390]` | Fair value: `₹360`
**GT:** `MITIGATE` → **Pred:** `LEVERAGE`
> GT response: *"I can see that you're interested, but $360 is the fair market rate. It's a solid deal."*
> Pred response: *"I appreciate your enthusiasm, but $360 is the best I can offer. It's a fair price for this condition."*

---
**dialogue_0337 T10** | Zone: `post_crystallisation` | Flip: `False` | Price gap: `25.0%`
Buyer emotion: `earnest, desperate` | intensity: `high` | valence: `negative`
Buyer offers: `[450, 430, 410, 390, 370]` | Fair value: `₹360`
**GT:** `MITIGATE` → **Pred:** `LEVERAGE`
> GT response: *"I understand, but $360 is already a fair price according to the market. It's the best I can do."*
> Pred response: *"I understand your position, but $360 is the best I can offer. It's a fair price for this condition."*

---
**dialogue_0428 T6** | Zone: `post_crystallisation` | Flip: `True` | Price gap: `23.3%`
Buyer emotion: `surprised, accepting, decisive` | intensity: `medium` | valence: `neutral`
Buyer offers: `[185, 175, 150]` | Fair value: `₹150`
**GT:** `LEVERAGE` → **Pred:** `MITIGATE`
> GT response: *"Great, $150 is indeed the fair market rate, and you’re getting a good deal."*
> Pred response: *"Great, $150 it is. I'm glad we could agree on a fair price."*

---
**dialogue_0455 T6** | Zone: `post_crystallisation` | Flip: `True` | Price gap: `26.9%`
Buyer emotion: `surprised, disappointed` | intensity: `medium` | valence: `negative`
Buyer offers: `[165, 155, 130]` | Fair value: `₹130`
**GT:** `LEVERAGE` → **Pred:** `MITIGATE`
> GT response: *"Yes, $130 is already a fair price for this model. It's what you'd find in the market."*
> Pred response: *"Yes, $130 is the fair market price for this model. It's a good deal."*

---
**dialogue_0527 T6** | Zone: `post_crystallisation` | Flip: `True` | Price gap: `23.2%`
Buyer emotion: `surprised, questioning` | intensity: `medium` | valence: `negative`
Buyer offers: `[345, 330, 280]` | Fair value: `₹280`
**GT:** `LEVERAGE` → **Pred:** `MITIGATE`
> GT response: *"No, nothing wrong with it at all. It's just that $280 is the fair market price, so you don't need to pay more."*
> Pred response: *"Yes, $280 is the fair market value. It's a great deal for the condition."*


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