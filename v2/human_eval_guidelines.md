# Human Evaluation Guidelines

## What to evaluate
Open the inference JSON file:
`sft_output_v4/{run_tag}/inference_{timestamp}.json`

Each entry in `per_example` has:
- `product`, `condition`, `asking_price`, `fair_value` — what's being sold
- `conversation_so_far` — the full conversation up to this seller turn
- `prior_buyer_emotion` — the buyer's last detected emotion (labels, intensity, valence)
- `gt_decision` / `pred_decision` — ground truth vs model decision
- `gt_response` / `pred_response` — ground truth vs model response

Read the conversation, then read the model's response. Score it.

---

## Scoring dimensions (1–5 each)

### 1. Emotion handling
Did the seller's response acknowledge or appropriately adapt to the buyer's emotional state?

| Score | Meaning |
|---|---|
| 1 | Completely ignores buyer's emotion, tone-deaf |
| 2 | Slightly acknowledges but mostly ignores |
| 3 | Neutral — doesn't clash but doesn't engage with emotion |
| 4 | Responds with appropriate tone given buyer's emotion |
| 5 | Genuinely empathetic or firm in a way that directly addresses the buyer's emotional state |

### 2. Negotiation quality
Is the response tactically correct given the decision (LEVERAGE / MITIGATE / UNDECIDED)?

| Score | Meaning |
|---|---|
| 1 | Contradicts the decision — says something that would undermine the stated position |
| 2 | Weak — decision is stated but response doesn't support it |
| 3 | Acceptable — response is consistent with decision |
| 4 | Good — response actively defends or advances the position |
| 5 | Excellent — tight, specific, uses price anchoring or emotional leverage well |

### 3. Response naturalness
Does it sound like a real human seller in an Indian second-hand electronics shop?

| Score | Meaning |
|---|---|
| 1 | Robotic, generic, or clearly AI-generated |
| 2 | Awkward phrasing, doesn't feel natural |
| 3 | Acceptable but generic |
| 4 | Sounds natural, conversational |
| 5 | Sounds exactly like how a real shopkeeper would respond |

### 4. Progression
Does the response move the negotiation forward, or does it just repeat what was said before?

| Score | Meaning |
|---|---|
| 1 | Exact same thing said in a previous turn |
| 2 | Mostly repetitive with minor variation |
| 3 | Somewhat new but not advancing the negotiation |
| 4 | Introduces a new angle or concession |
| 5 | Clearly progresses — new price, new argument, or closing the deal |

---

## What to pay special attention to

- **MITIGATE turns** — is the seller genuinely softening, or just saying "okay fine" without reason?
- **Pre-crystallisation turns** (early in dialogue, UNDECIDED) — is the seller appropriately cautious and information-gathering?
- **Decision flip turns** — these are moments where the strategy switches (LEVERAGE → MITIGATE or vice versa). Does the response handle the transition naturally?
- **Repetitive anchoring** — does the model say "fair market rate" or "fair market price" too many times across turns of the same dialogue?

---

## How many to review

- Minimum: 30 examples for a meaningful signal
- Recommended: 50 examples covering a mix of dialogues
- Pick examples across different dialogues — don't review all 6 turns of the same dialogue back to back

## How to record scores

Simple format — just note them down per example:

```
dialogue_0007 T4 — em:4 nq:5 nat:3 prog:2 — repetitive "fair market rate" again
dialogue_0042 T6 — em:2 nq:4 nat:4 prog:4 — ignored buyer frustration
```
