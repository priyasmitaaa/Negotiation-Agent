# GRPO Reward Candidates
Faults observed from SFT v4 inference outputs → encode as reward signals in GRPO.

---

## Observed Faults

### 1. Repetitive anchoring phrases
**Observed:** Model repeats near-identical phrases across consecutive turns of the same dialogue.
Example from dialogue_0007: "fair market rate" / "fair market price" used in T04, T08, T10 — verbatim or near-verbatim.
**Candidate reward:** Lexical diversity penalty — measure n-gram overlap between the current response and all prior seller responses in the same dialogue. Penalise high overlap.

---

### 2. Chinese tokens at end of TTS output
**Observed:** The talker appends Chinese speech after the English response ends (observed in turn 12 of dialogue_0007).
**Candidate reward:** Language purity reward — detect non-English tokens in the generated response text. Penalise any non-ASCII / non-English characters in the seller response.

---

## To Be Filled (as more inference faults are observed)

- [ ] Response too short / not engaging enough
- [ ] Fails to acknowledge buyer's emotional state
- [ ] Price anchoring too aggressive or too passive relative to zone
- [ ] Decision flip not handled naturally in language
- [ ] Response does not progress the negotiation (says same thing as previous turn)

---

## GRPO Strategy

**Baseline:** Start from v4 SFT adapter (clean decision + response, no reasoning noise).
Add `<reasoning>` to the generation format via system prompt — let GRPO discover reasoning from scratch.
Do NOT use Gemini-generated reasoning (v5 SFT) as supervision — it is post-hoc rationalisation, causal direction is wrong.

**Reward structure (Option B — light structural guardrails):**
```
total_reward = 1.0 × decision_accuracy          ← primary, drives everything
             + 0.2 × reasoning_mentions_price    ← model must look at price numbers
             + 0.2 × reasoning_mentions_emotion  ← model must look at buyer emotion
             + 0.1 × language_purity             ← no Chinese/non-English tokens
             + 0.1 × lexical_diversity           ← no repetitive anchoring phrases
             + 0.3 × flip_turn_bonus             ← bonus for correct flip turn decisions
```

**Graded decision reward (from error analysis):**
At boundary conditions (buyer offer >= fair value), LEVERAGE vs MITIGATE become ambiguous
— the dataset labels are noisy there. Use graded penalty instead of binary:
```
correct decision              → +1.0
LEVERAGE/MITIGATE confusion
  at boundary (offer >= FV)   → -0.3   (not -1.0, label is likely ambiguous)
UNDECIDED wrong               → -1.0   (no ambiguity, always a clear error)
missing decision tag          →  0.0   (format gate handles this separately)
```
Decision reward dominates. Reasoning rewards are guardrails only — they ensure the model
attends to the right signals without dictating what conclusions to draw. Model discovers
reasoning content on its own through RL.

---

## Notes
- Do NOT patch these at inference time (no repetition_penalty, no language filters at generation).
- Keep raw inference outputs as ground truth of current model faults.
- Each fault here maps to one reward component in the GRPO reward function.
- Reference: inference results at sft_output_v4/20260525_014244/speech_inference_*.json
