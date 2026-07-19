Fair challenge. Let me stress-test both claims honestly.

---

**Claim 1: Δanchoring is computable turn-by-turn**

Yes, but with one real constraint you need to account for. The formula `Anchoring = 1 − (Actual Adjustment / Rational Adjustment)` requires three values at every turn: the customer's anchor (first offer, fixed), the fair value (fixed per scenario), and the customer's *current* offer. All three are available in your dataset — the dialogues already have price offers annotated, fair value is part of the scenario metadata, and the anchor is the first customer price mention.

So computing `anchoring_t` at each turn is straightforward. `Δanchoring = anchoring_t − anchoring_{t+2}` is just arithmetic. No second model, no ambiguity. This part holds.

The one practical wrinkle: not every dialogue turn contains a price offer. The customer might ask a question or make a non-price statement. You need a fallback — either carry forward the last known offer, or only compute `Δanchoring` on turns where a new offer is detected. This is a 10-line logic block, not a research problem.

---

**Claim 2: R_harm_penalty as a terminal reward is non-redundant**

This also holds, but only if you're running a dialogue agent through full episodes. The per-turn ethics compliance signal tells the agent "you acted correctly at this step." The terminal harm penalty tells the agent "despite acting correctly at each step, the customer still ended up overpaying." These are genuinely different failure modes.

However — and this is the honest complication — implementing a terminal reward in your training loop means your RL episodes need to run to completion before the signal arrives. If you're using PPO with short rollouts, the terminal signal gets heavily discounted and may barely influence early-turn behavior. You'd need either long rollout windows or reward shaping to propagate the terminal signal back through the episode. That's not insurmountable but it's a real implementation consideration, not just a design choice.

---

**So can this be designed without overcomplicating things?**

Yes, with one simplification: rather than treating `R_harm_penalty` as a separate term, fold it into `R_ethics_compliance` as a large bonus/penalty at the final turn. This gives you:

```
R_total = α·R_outcome + β·R_ethics_compliance + γ·Δanchoring
```

where `R_ethics_compliance` returns its normal per-turn value on intermediate turns, and on the *final turn* additionally adds `−δ` if final price > fair value. You keep the terminal signal without adding a fourth weight to tune, and the implementation is a single conditional in your reward function — check if `turn == final_turn`, then append the harm check. Programmatically this is clean, flows naturally through your existing episode loop, and doesn't require a separate tracking variable.

The flow would be:

```python
def compute_reward(turn, state, action, is_final):
    r_outcome = compute_outcome(state)
    r_ethics = check_rule_compliance(state, action)
    r_debias = compute_delta_anchoring(state)  # 0 if no new offer this turn
    
    if is_final and state.final_price > state.fair_value:
        r_ethics -= HARM_PENALTY  # fold terminal signal here
    
    return alpha*r_outcome + beta*r_ethics + gamma*r_debias
```

Three weights, clean episode logic, terminal harm signal included without a fourth term. That's the simplest version that preserves the structural correctness of Option 3.