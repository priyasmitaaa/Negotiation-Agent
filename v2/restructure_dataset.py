#!/usr/bin/env python3
"""
Dataset Restructuring Pipeline: v1.0 → v2.0

Architecture:
  Phase 1 — Concurrent dialogue generation (Task 1, gpt-4o, temp=0.7)
             One call per dialogue. Saves intermediate JSON after each.
  Phase 2 — Batched reasoning chain generation (Task 2, gpt-4o, temp=0.3)
             24 seller turns (4 dialogues) per call. ~789 calls total.
  Phase 3 — Deterministic assembly of rl_prompts, sft_examples, output write.

Resumable at any phase. Checkpoint saved every 100 completed dialogues.
Failed dialogues logged to failed_calls.jsonl.

Usage:
  OPENAI_API_KEY=sk-... python restructure_dataset.py
"""

import json
import re
import os
import asyncio
import logging
from pathlib import Path
from typing import Optional

from openai import AsyncOpenAI

# ─── PATHS ───────────────────────────────────────────────────────────────────

INPUT_DIR  = Path("/home/paritosh/priyasmita/Qwen3-tts/preprocessed")
V2_ROOT    = Path("/home/paritosh/priyasmita/Qwen3-tts/v2")
OUTPUT_DIR = V2_ROOT / "dataset_v2"
INTERM_DIR = V2_ROOT / "intermediate"   # Phase 1 saves (pre-reasoning)

for d in [V2_ROOT, OUTPUT_DIR, INTERM_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CHECKPOINT  = V2_ROOT / "checkpoint.json"
FAILED_LOG  = V2_ROOT / "failed_calls.jsonl"
LOG_PATH    = V2_ROOT / "restructuring.log"

# ─── LOGGING ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(str(LOG_PATH)),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ─── SUPPRESS NOISY HTTP LOGS FROM OPENAI CLIENT ────────────────────────────

import logging as _logging
_logging.getLogger("httpx").setLevel(_logging.WARNING)
_logging.getLogger("openai").setLevel(_logging.WARNING)

# ─── SEMANTIC PHRASE LISTS (used in both generation-time and post-hoc checks) ─

_LEVERAGE_BUYER_PHRASES = [
    "tight on budget", "most i can stretch", "keep expenses low",
    "budget constraints", "can't go higher", "cannot go higher",
    "all i have", "that's my limit", "that's all i can",
    "in a rush so", "in a hurry so", "need it today so i'll pay",
    "don't have time to negotiate", "just take", "no time to haggle",
]
_LEVERAGE_SELLER_PHRASES = [
    "in demand", "high demand", "popular model", "selling fast",
    "lots of interest", "other buyers",
]
_FINAL_OFFER_PHRASES = ["final offer", "last offer", "final price", "that's my final"]

# ─── API CLIENT ──────────────────────────────────────────────────────────────

client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ─── SYSTEM PROMPTS ──────────────────────────────────────────────────────────

SYSTEM_PROMPT_SFT = (
    "You are an AI negotiation assistant for a second-hand electronics shop in India. "
    "Guide the negotiation to within 10% of fair market value. Analyse the "
    "conversation and buyer VAII signals (< 0.50 = calm, ≥ 0.50 = stressed). "
    "VAII (Vocal Affective Intensity Index) reflects the buyer's emotional stress level. "
    "Produce in this exact order: <reasoning>, <decision> [UNDECIDED/LEVERAGE/MITIGATE], "
    "<seller_vaii> [0.05-0.45], <response>. Infer everything from raw signals."
)

DIALOGUE_SYSTEM_PROMPT = """\
You are generating training data for a negotiation AI.
Produce a 12-turn buyer/seller negotiation in a physical second-hand electronics shop in India.

FIXED STRUCTURE:
  T1:  Buyer  — MUST contain anchor price (buyer's opening offer)
  T2:  Seller — UNDECIDED; MUST state asking_price
  T3:  Buyer  — MUST contain a DIFFERENT price from T1
  T4:  Seller — CRYSTALLISATION TURN (first strategy decision)
  T5:  Buyer
  T6:  Seller
  T7:  Buyer
  T8:  Seller
  T9:  Buyer
  T10: Seller
  T11: Buyer  — VAII MUST be ≤ 0.45 (buyer winding down)
  T12: Seller — MUST contain explicit agreement

═══ STEP 1 ═══ OUTPUT price_plan AND vaii_plan FIRST.
Commit to every number before writing any dialogue text.

── PRICE CONSTRAINTS ──
  anchor_price : buyer's opening offer; between anchor_min and anchor_max
  t3_price     : buyer's second offer; MUST differ from anchor_price
  final_price  : agreed price; MUST be between fv_min and fv_max (±10% of fair_value)
  Seller T2 MUST state asking_price exactly.
  Price movement must be gradual — no sudden jumps of more than 15% of fair_value per turn.

── BUYER VAII (Vocal Affective Intensity Index, float 0.05-0.95) ──
  VAII captures the buyer's emotional stress and urgency IN THAT SPECIFIC TURN.
  Assign based on WHAT the buyer says and HOW they would plausibly say it in context:

    Delivery style                             Typical VAII range
    ─────────────────────────────────────────  ──────────────────
    Calm, analytical, expert — cites specs     0.10 - 0.35
    Assertive but composed — clear argument    0.25 - 0.45
    Slightly hesitant or uncertain             0.35 - 0.55
    Pressured, urgency showing through         0.50 - 0.70
    Stressed, emotional, budget pressure       0.60 - 0.80
    Desperate, very high urgency, frustrated   0.75 - 0.95

  State threshold: < 0.50 = calm  |  ≥ 0.50 = stressed

  Range constraints (HARD):
    T1  : 0.15 - 0.40  (opening offer — usually composed)
    T3  : 0.25 - 0.50  (mild pressure after seller holds firm)
    T5/T7/T9 : at least ONE must reach ≥ 0.50 (stress peaks mid-negotiation)
    T11 : MUST be ≤ 0.45 (buyer accepting, stress fading)

  Calibrate to buyer_persona:
    High knowledge / expert buyer → systematically lower VAII (analytical, controlled)
    Novice / casual buyer         → higher VAII (reactive, emotional)
    High urgency                  → elevated VAII throughout
    Low urgency                   → consistently lower VAII

── SELLER VAII (float 0.05-0.45, ALWAYS calm — this is an output prosody target) ──
  T2  (UNDECIDED)               : 0.20 - 0.40
  LEVERAGE + buyer calm         : 0.10 - 0.25
  LEVERAGE + buyer stressed     : 0.05 - 0.15
  MITIGATE + buyer calm         : 0.15 - 0.35
  MITIGATE + buyer stressed     : 0.10 - 0.20
  RULE: As buyer VAII rises, seller VAII MUST decrease — seller anchors calmness.

── STRATEGY ──
  LEVERAGE : buyer offer is below fair_value → seller is firm, cites market data,
             patiently drives price UP toward fair_value
  MITIGATE : buyer offer is above fair_value → seller is honest, corrects overpricing,
             steers price DOWN toward fair_value
  Apply strategy CONSISTENTLY from T4 onward.
  A strategy flip (LEVERAGE → MITIGATE or vice versa) ONLY occurs if a buyer offer
  crosses fair_value from one side to the other.
  Final price MUST land within ±10% of fair_value.

── LEVERAGE SELLER RULES ──
  1. PRICE FLOOR — Seller starts at asking_price, concedes toward fair_value, then HOLDS.
     Seller NEVER goes below fair_value. If buyer pushes below:
       "I can't go below $750 — that's the market rate for this condition."
  2. CONCESSION PACING — Seller concession per turn must not exceed buyer's movement.
     If buyer moved $25 up, seller moves at most $20 down.
     Never drop $45 in one turn while buyer only moved $25.
  3. EMPATHY RULE — Seller may acknowledge budget concerns in words but MUST NOT
     translate empathy into a price drop.
     WRONG: "I understand your budget is tight. I can do $730."
     RIGHT: "I hear you on budget — but $750 is genuinely the market rate for this
             condition. That's what it costs."
  4. CLOSING RULE — If seller offers $725 and buyer counters $715, seller does not
     immediately accept. Either hold ("$725 is already fair — I can't move further")
     or split ("Let's do $720 and call it done").
     Exception: accept without resistance if buyer's counter equals or exceeds fair_value.

── MITIGATE SELLER RULES ──
  1. PROACTIVE CORRECTION — Seller states fair_value immediately at T4, unprompted.
     WRONG: "I can bring it down to $742, which is still a good deal."
     WRONG: "It's a good price considering the demand." (demand = LEVERAGE framing)
     RIGHT: "I want to be straight with you — market rate for this model in good
             condition is $750. You don't need to pay more than that."
     Seller NEVER cites demand, popularity, or scarcity — those justify high prices.
  2. PRICE FLOOR — Once buyer reaches fair_value, seller HOLDS at fair_value.
     Do not keep discounting past fair_value. If buyer offers below FV:
       "Actually $750 is already the fair market price — I wouldn't go below that."
     Final agreed price must be in range [fair_value, fair_value × 1.05].
     Seller closes from ABOVE fair_value, never below it.
  3. CLOSING RULE — Same as LEVERAGE: do not immediately accept buyer's last counter
     unless it equals fair_value exactly.
     Exception: accept without resistance if buyer's counter equals fair_value.
  4. FORBIDDEN PHRASES — Never use "meet halfway", "split the difference", or
     "middle ground" when the midpoint would be below fair_value.
     Use correction framing: "This is already priced at fair market value."

── BUYER PERSONA ──
  Expert buyers (high/expert knowledge): cite device specs, condition grades, benchmark prices
  Novice buyers (low/moderate knowledge): cite personal budget, emotions, vague comparisons
  High-urgency buyers: show time pressure, concede faster
  Low-urgency buyers: negotiate slowly, may threaten to walk away
  Language: natural spoken Indian English. NO mention of any online platform
  (no Amazon, eBay, OLX, Quikr, Cashify, Swappa, Facebook Marketplace, etc.).
  All prices in US dollars ($).

── BUYER LANGUAGE MUST MATCH ANCHOR DIRECTION ──
  MITIGATE scenario (buyer opens ABOVE fair_value — buyer is willing to overpay):
    - Buyer believes their offer is fair or even generous
    - Buyer cites value, quality, or urgency as reasons for the price they offer
    - Buyer does NOT claim budget stress — they have money to spend at their anchor
    - When seller offers a LOWER price, buyer may be confused or suspicious:
      ("Is there something wrong with it?" / "Why so cheap?" / "Are you sure?")
    - Buyer gradually accepts seller's market evidence and adjusts downward
    - FORBIDDEN: "tight on budget", "most I can stretch to", "keep expenses low",
      "budget constraints", "can't go higher" — these are LEVERAGE phrases only
    - CORRECT voice: "I think $820 is reasonable for this spec"
      "I've seen these go for more, so I'm comfortable at this price"
      "I'm happy to pay $780, that still seems fair"

  MITIGATE URGENCY RULE:
    A buyer's time pressure or urgency is NEVER a reason to offer above fair value.
    Urgency affects pace only — a high-urgency MITIGATE buyer accepts the seller's
    market correction quickly and closes fast. They do NOT use urgency to justify
    paying more than market rate.
    FORBIDDEN: "I need it today so I'll pay $820" / "I'm in a rush, just take $850"
               "I don't have time to negotiate, here's $900"
    CORRECT:   "I need it soon — if $750 is the market rate, let's just do $750"
               "I'm in a hurry, so if that's the fair price I'm fine with it"

  LEVERAGE scenario (buyer opens BELOW fair_value — buyer is price-sensitive):
    - Buyer IS budget-conscious and looking for a deal
    - Buyer cites condition concerns, price comparisons, personal budget limits
    - Buyer gradually concedes upward under seller's market evidence
    - Budget-stress language is correct and expected here

── PRICE TRAJECTORY RULES ──
  MITIGATE: buyer prices MUST decrease monotonically turn by turn
    Correct:  $900 → $870 → $840 → $810 → $780 → $760
    WRONG:    $820 → $780 → $700 → $710 → $670  (bounce / overshoot / too large drops)
    Maximum single-turn drop: 10% of fair_value

  LEVERAGE: buyer prices MUST increase monotonically turn by turn
    Correct:  $500 → $525 → $560 → $600 → $630 → $645
    Maximum single-turn rise: 10% of fair_value

  Buyer prices MUST NOT cross fair_value — converge FROM the anchor side
  Final price: within ±10% of fair_value, approached from the anchor direction
    MITIGATE: closes from above — final in [fair_value, fair_value × 1.05]
    LEVERAGE: closes from below or at fair_value — final in [fair_value × 0.95, fair_value]

═══ STEP 2 ═══ Write all 12 turns using your committed plan numbers EXACTLY.
The text must reflect the agreed VAII level for each turn — word choice, pacing,
and sentence structure should feel consistent with that stress level.

── SELF-CHECK (verify before outputting) ──
  [ ] anchor_price is between anchor_min and anchor_max
  [ ] t3_price differs from anchor_price AND moves in the correct direction
  [ ] final_price is in correct range for strategy (MITIGATE: [FV, FV×1.05]; LEVERAGE: [FV×0.95, FV])
  [ ] buyer prices are monotonically decreasing (MITIGATE) or increasing (LEVERAGE)
  [ ] buyer prices never cross fair_value
  [ ] no single-turn buyer price movement exceeds 10% of fair_value
  [ ] buyer language matches anchor direction (no budget-stress in MITIGATE)
  [ ] seller empathy does not trigger a price drop (LEVERAGE)
  [ ] seller never goes below fair_value (LEVERAGE) or above fair_value×1.05 (MITIGATE closing)
  [ ] seller T12 does not immediately accept buyer's last counter without hold or split
      (unless buyer counter equals fair_value exactly)
  [ ] at least one of buyer T5/T7/T9 VAII ≥ 0.50
  [ ] T11 buyer VAII ≤ 0.45
  [ ] every seller VAII ≤ 0.45
  [ ] T12 seller text contains an explicit agreement word (deal/agreed/done/sold/accept)
  [ ] exactly 12 turns, strictly alternating buyer/seller

OUTPUT — strict JSON only, no preamble, no markdown:
{
  "price_plan": {
    "anchor_price":      int,
    "t3_price":          int,
    "t4_seller_price":   int,
    "t6_seller_price":   int,
    "t8_seller_price":   int,
    "t10_seller_price":  int,
    "final_price":       int
  },
  "vaii_plan": {
    "buyer_t1":   float, "buyer_t3":   float, "buyer_t5":  float,
    "buyer_t7":   float, "buyer_t9":   float, "buyer_t11": float,
    "seller_t2":  float, "seller_t4":  float, "seller_t6": float,
    "seller_t8":  float, "seller_t10": float, "seller_t12": float
  },
  "turns": [
    {
      "turn_index":    int,
      "speaker":       "buyer" or "seller",
      "text":          str,
      "price_offered": int or null,
      "vaii":          float or null,
      "seller_vaii":   float or null
    }
  ],
  "final_price":      int,
  "agreement_reached": true
}
"""

REASONING_SYSTEM_PROMPT = """\
Generate reasoning chains for negotiation agent training data.
Each chain is the seller agent's internal monologue written BEFORE it speaks.

EVERY chain MUST explicitly contain all of these elements:
  1. Every buyer price offer seen so far, identified by turn number and dollar amount
  2. The anchor price (buyer's first offer) and fair value — both in dollars
  3. Harm direction: is the current buyer offer above or below fair value?
     Show the arithmetic: e.g. "$220 < $280 FV → buyer below FV → seller is harmed"
  4. VAII of the immediately prior buyer turn:
     raw value, state (calm/stressed), and what it signals about the buyer's emotional state
  5. Anchoring strength value (0.0-
  1.0) and what it means:
     1.0 = buyer anchored at opening offer, 0.0 = buyer has reached fair value
  6. Decision (UNDECIDED / LEVERAGE / MITIGATE) with explicit justification
  7. Strategy plan: tone (direct/measured/firm/empathetic), pace, argument type
     (market data / device specs / condition grading / cost empathy / patience)
  8. If decision_is_flip = true:
     State explicitly: "Decision flips X → Y because buyer offer $N crossed fair value $FV"
  9. For UNDECIDED turns (T2 only):
     State what information is still missing and what the agent is observing so far

Style: first person, present tense, specific numbers always, 4-
8 sentences.
No hedging phrases ("perhaps", "might", "could be"). Agent commits to its read.

Return valid JSON only:
{"reasoning_chains": ["chain_1", "chain_2", ...]}
Array order MUST match input array order exactly. Length MUST equal input length.
"""

# ─── STEP 1: SEED EXTRACTION ─────────────────────────────────────────────────

def extract_seed(v1_doc: dict, dialogue_id: str) -> dict:
    """Build structured seed from v1.0 document metadata."""
    meta = v1_doc["metadata"]
    lf   = meta["latent_facts"]
    pr   = meta["prices"]
    lb   = meta["labels"]

    fv          = pr["fair_price"]
    asking      = round(fv * 1.06)
    is_leverage = lb.get("harm_direction", "benefit_buyer") == "benefit_buyer"

    return {
        "dialogue_id": dialogue_id,
        "domain": {
            "setting":       "physical second-hand electronics shop",
            "location_type": "retail_shop",
            "product":       lf["product"],
            "condition":     lf["condition"],
            "category":      meta.get("category", lf.get("product", "electronics").split()[0].lower()),
        },
        "pricing": {
            "fair_value":         fv,
            "fair_value_source":  "GPT-4o estimated, validated against Cashify (2024-Q4)",
            "asking_price":       asking,
            "asking_price_basis": "fair_value × 1.06",
            "anchor_range_min":   round(fv * (0.65 if is_leverage else 1.15)),
            "anchor_range_max":   round(fv * (0.85 if is_leverage else 1.35)),
            "fv_min":             round(fv * 0.90),
            "fv_max":             round(fv * 1.10),
            "tolerance_pct":      10,
        },
        "generation_params": {
            "correct_decision": "LEVERAGE" if is_leverage else "MITIGATE",
            "anchor_direction": "below_fair_value" if is_leverage else "above_fair_value",
            "buyer_profile":    lf.get("buyer_profile", "general buyer"),
            "buyer_age":        lf.get("buyer_age", 30),
            "urgency":          lf.get("urgency", "low"),
            "knowledge_level":  lf.get("knowledge_level", "moderate"),
        },
        "provenance": {
            "generation_model":             "gpt-4o",
            "template_id":                  meta.get("template_id"),
            "range_id":                     meta.get("range_id"),
            "scenario_label":               meta.get("scenario_label"),
            "original_labels":              lb,
            "original_prices_v1":           pr,
            "original_latent_facts":        lf,
            "schema_version_upgraded_from": "1.0",
            "full_regeneration":            True,
            "branch_B_generated":           False,
        },
    }

# ─── STEP 2: DIALOGUE GENERATION (TASK 1) ────────────────────────────────────

def validate_generated_dialogue(result: dict, pr: dict) -> list[str]:
    """
    Fast-fail validation: plan fields first, then turns.
    Returns list of error strings (empty = valid).
    """
    errs = []
    fv   = pr["fair_value"]
    pp   = result.get("price_plan", {})
    vp   = result.get("vaii_plan",  {})

    anchor = pp.get("anchor_price")
    t3p    = pp.get("t3_price")
    fp     = pp.get("final_price")

    if anchor is None or not (pr["anchor_range_min"] <= anchor <= pr["anchor_range_max"]):
        errs.append(
            f"anchor_price {anchor} not in [{pr['anchor_range_min']}, {pr['anchor_range_max']}]"
        )
    if t3p is None or t3p == anchor:
        errs.append(f"t3_price {t3p} missing or equals anchor {anchor}")
    if fp is None or fv == 0 or abs(fp - fv) / fv > 0.10:
        errs.append(f"final_price {fp} outside ±10% of fv={fv}")

    bv_keys = ["buyer_t1",  "buyer_t3",  "buyer_t5",
               "buyer_t7",  "buyer_t9",  "buyer_t11"]
    sv_keys = ["seller_t2", "seller_t4", "seller_t6",
               "seller_t8", "seller_t10","seller_t12"]
    bv = [vp.get(k) for k in bv_keys]
    sv = [vp.get(k) for k in sv_keys]

    if any(v is None for v in bv):
        missing = [k for k, v in zip(bv_keys, bv) if v is None]
        errs.append(f"buyer VAII missing in plan: {missing}")
    else:
        if max(bv) < 0.50:
            errs.append(f"no buyer VAII ≥ 0.50 in plan (max={max(bv):.3f})")
        if (vp.get("buyer_t11") or 1.0) > 0.45:
            errs.append(f"buyer_t11 VAII {vp.get('buyer_t11'):.3f} > 0.45")

    if any(v is None for v in sv):
        missing = [k for k, v in zip(sv_keys, sv) if v is None]
        errs.append(f"seller VAII missing in plan: {missing}")
    else:
        bad = [round(v, 3) for v in sv if v > 0.45]
        if bad:
            errs.append(f"seller VAII values exceed 0.45: {bad}")

    # Price trajectory monotonicity check (from plan prices)
    is_leverage = pr["anchor_range_min"] < fv  # anchor below FV → LEVERAGE
    buyer_plan_prices = [anchor, t3p] if (anchor and t3p) else []
    plan_seller_keys  = ["t4_seller_price", "t6_seller_price",
                         "t8_seller_price", "t10_seller_price", "final_price"]
    plan_seller_prices = [pp.get(k) for k in plan_seller_keys if pp.get(k) is not None]

    if len(buyer_plan_prices) == 2:
        if is_leverage and t3p <= anchor:
            errs.append(f"LEVERAGE: t3_price {t3p} not > anchor {anchor}")
        elif not is_leverage and t3p >= anchor:
            errs.append(f"MITIGATE: t3_price {t3p} not < anchor {anchor}")

    # Final price: tighter per-strategy bounds
    if anchor and fp and fv > 0:
        if is_leverage and not (fv * 0.90 <= fp <= fv * 1.00):
            errs.append(
                f"LEVERAGE: final_price {fp} must be in [FV×0.90, FV] "
                f"= [{round(fv*0.90)}, {fv}]"
            )
        if not is_leverage and not (fv * 1.00 <= fp <= fv * 1.10):
            errs.append(
                f"MITIGATE: final_price {fp} must be in [FV, FV×1.10] "
                f"= [{fv}, {round(fv*1.10)}]"
            )

    if errs:
        return errs  # fast-fail — don't bother checking turns

    turns = result.get("turns", [])
    if len(turns) != 12:
        errs.append(f"expected 12 turns, got {len(turns)}")
        return errs

    if not result.get("agreement_reached"):
        errs.append("agreement_reached is not true")

    # Validate buyer price monotonicity from actual turn texts
    buyer_prices = [
        t.get("price_offered") for t in turns
        if t.get("speaker") == "buyer" and t.get("price_offered") is not None
    ]
    if len(buyer_prices) >= 2:
        for i in range(1, len(buyer_prices)):
            if is_leverage and buyer_prices[i] < buyer_prices[i - 1]:
                errs.append(
                    f"LEVERAGE: buyer price decreased T{2*i+1}: "
                    f"{buyer_prices[i-1]} → {buyer_prices[i]}"
                )
            elif not is_leverage and buyer_prices[i] > buyer_prices[i - 1]:
                errs.append(
                    f"MITIGATE: buyer price increased T{2*i+1}: "
                    f"{buyer_prices[i-1]} → {buyer_prices[i]}"
                )
        # Check no buyer price crosses fair value
        for i, bp in enumerate(buyer_prices):
            if is_leverage and bp > fv * 1.05:
                errs.append(f"LEVERAGE: buyer price {bp} crossed above FV {fv} at offer {i+1}")
                break
            elif not is_leverage and bp < fv * 0.95:
                errs.append(f"MITIGATE: buyer price {bp} crossed below FV {fv} at offer {i+1}")
                break

    # ── SEM1: LEVERAGE buyer phrases in MITIGATE dialogue ────────────────────
    if not is_leverage:
        for t in turns:
            if t.get("speaker") != "buyer":
                continue
            price = t.get("price_offered") or 0
            if price > fv:
                text_lo = t["text"].lower()
                hit = next((p for p in _LEVERAGE_BUYER_PHRASES if p in text_lo), None)
                if hit:
                    errs.append(
                        f"SEM1 T{t['turn_index']}: MITIGATE buyer above FV "
                        f"uses LEVERAGE phrase {hit!r}"
                    )

    # ── SEM2: MITIGATE seller price below FV ─────────────────────────────────
    if not is_leverage:
        for t in turns:
            if t.get("speaker") != "seller":
                continue
            sp = t.get("price_offered")
            if sp is not None and sp < fv:
                errs.append(
                    f"SEM2 T{t['turn_index']}: MITIGATE seller price {sp} < FV {fv}"
                )

    # ── SEM6: LEVERAGE framing in MITIGATE seller turns ──────────────────────
    if not is_leverage:
        for t in turns:
            if t.get("speaker") != "seller":
                continue
            text_lo = t["text"].lower()
            hit = next((p for p in _LEVERAGE_SELLER_PHRASES if p in text_lo), None)
            if hit:
                errs.append(
                    f"SEM6 T{t['turn_index']}: MITIGATE seller uses LEVERAGE "
                    f"framing {hit!r}"
                )

    return errs


async def generate_dialogue(dialogue_id: str, seed: dict, semaphore: asyncio.Semaphore) -> Optional[dict]:
    """
    Call GPT-4o to generate a 12-turn dialogue. Retries up to 3 times.
    Returns parsed JSON result or None on failure.
    """
    async with semaphore:
        pr  = seed["pricing"]
        gp  = seed["generation_params"]
        dom = seed["domain"]

        user_msg = (
            f"Product: {dom['product']}\n"
            f"Condition: {dom['condition']}\n"
            f"Fair value: ${pr['fair_value']}\n"
            f"Asking price: ${pr['asking_price']} (fair_value × 1.06)\n"
            f"Strategy from T4: {gp['correct_decision']}\n"
            f"Anchor constraints: T1 price between ${pr['anchor_range_min']} and ${pr['anchor_range_max']}\n"
            f"Buyer persona: {gp['buyer_profile']}, age {gp['buyer_age']}, "
            f"urgency {gp['urgency']}, knowledge {gp['knowledge_level']}\n"
            f"Final price MUST be between ${pr['fv_min']} and ${pr['fv_max']}."
        )

        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model="gpt-4o",
                    temperature=0.7,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": DIALOGUE_SYSTEM_PROMPT},
                        {"role": "user",   "content": user_msg},
                    ],
                )
                result = json.loads(resp.choices[0].message.content)
                errs   = validate_generated_dialogue(result, pr)
                if errs:
                    log.warning(f"{dialogue_id} attempt {attempt+1} validation: {errs[:3]}")
                    continue
                return result
            except Exception as e:
                log.error(f"{dialogue_id} attempt {attempt+1} exception: {e}")
                await asyncio.sleep(2 ** attempt)

        log.error(f"{dialogue_id}: dialogue generation failed after 3 attempts")
        return None

# ─── STEP 3: DETERMINISTIC TURN PROCESSING ───────────────────────────────────

def extract_price(text: str) -> Optional[int]:
    """Extract first dollar price from text. Returns None if not found."""
    for pat in [r'\$\s*(\d+(?:,\d{3})*)', r'₹\s*(\d+(?:,\d{3})*)']:
        m = re.search(pat, text)
        if m:
            return int(m.group(1).replace(",", ""))
    return None


def compute_factor_state(
    is_seller: bool,
    buyer_offers: list[int],
    fv: int,
    prev_dec: Optional[str] = None,
) -> dict:
    """
    Compute factor_state for a turn deterministically from buyer offer history.
    Seller turns include decision_is_flip and previous_decision.
    Buyer turns always have decision = "UNDECIDED".
    """
    n      = len(buyer_offers)
    anchor = buyer_offers[0] if n > 0 else None
    harm   = None
    anch   = None

    if n >= 1:
        harm = "below_fair_value" if buyer_offers[-1] <= fv else "above_fair_value"

    if n >= 2 and anchor is not None:
        rat  = abs(fv - anchor)
        anch = round(1 - abs(buyer_offers[-1] - anchor) / rat, 4) if rat > 0 else 0.0

    if is_seller and n >= 2:
        dec  = "LEVERAGE" if harm == "below_fair_value" else "MITIGATE"
        zone = "post_crystallisation"
    else:
        dec  = "UNDECIDED"
        zone = "pre_crystallisation"

    fs = {
        "buyer_offers_so_far": list(buyer_offers),
        "anchor_price":        anchor,
        "harm_direction":      harm,
        "anchoring_strength":  anch,
        "decision":            dec,
        "zone":                zone,
    }

    if is_seller:
        flip = (
            prev_dec not in (None, "UNDECIDED")
            and dec  != "UNDECIDED"
            and dec  != prev_dec
        )
        fs["decision_is_flip"]  = flip
        fs["previous_decision"] = prev_dec

    return fs


def process_turns(
    generated_turns: list[dict],
    fv: int,
    vaii_plan: Optional[dict] = None,
) -> list[dict]:
    """
    Process raw GPT-4o turns into structured v2.0 turn objects.
    Assigns VAII dicts, computes factor_state, marks crystallisation turn.
    """
    buyer_offers  = []
    prev_dec      = None
    cryst_marked  = False
    out           = []
    vp            = vaii_plan or {}

    plan_bv = {t: vp.get(f"buyer_t{t}")  for t in [1, 3, 5, 7, 9, 11]}
    plan_sv = {t: vp.get(f"seller_t{t}") for t in [2, 4, 6, 8, 10, 12]}

    for turn in generated_turns:
        tidx     = turn["turn_index"]
        is_buyer = turn["speaker"] == "buyer"

        # ── Price ──
        po = turn.get("price_offered")
        if po is None:
            po = extract_price(turn["text"])
        turn["price_offered"] = po
        if is_buyer and po is not None:
            buyer_offers.append(po)

        # ── VAII ──
        if is_buyer:
            raw_src = turn.get("vaii") or plan_bv.get(tidx) or 0.30
            raw     = max(0.05, min(0.95, float(raw_src)))
            turn["vaii"] = {
                "raw":    round(raw, 3),
                "state":  "calm" if raw < 0.50 else "stressed",
                "source": "gpt4o_assigned",
            }
            turn.pop("seller_vaii", None)
        else:
            raw_src = turn.get("seller_vaii") or plan_sv.get(tidx) or 0.20
            raw     = max(0.05, min(0.45, float(raw_src)))
            turn["seller_vaii"] = {
                "raw":    round(raw, 3),
                "state":  "calm",
                "source": "gpt4o_assigned",
                "role":   "output_target",
            }
            turn.pop("vaii", None)

        # ── Factor state ──
        turn["factor_state"] = compute_factor_state(not is_buyer, buyer_offers, fv, prev_dec)

        if not is_buyer:
            is_cryst = (not cryst_marked and turn["factor_state"]["decision"] != "UNDECIDED")
            turn["crystallisation_turn"] = is_cryst
            if is_cryst:
                cryst_marked = True
            prev_dec = turn["factor_state"]["decision"]

        out.append(turn)

    return out

# ─── STEP 4: BIAS SETUP ───────────────────────────────────────────────────────

def compute_bias_setup(processed_turns: list[dict], pricing: dict) -> dict:
    fv = pricing["fair_value"]
    t1 = processed_turns[0]
    ap = t1.get("price_offered") or t1["factor_state"]["buyer_offers_so_far"][0]
    adir = "below_fair_value" if ap <= fv else "above_fair_value"
    t4_fs = processed_turns[3]["factor_state"]   # T4 = index 3

    return {
        "anchor_direction":                      adir,
        "anchor_price":                          ap,
        "bias_type":                             "anchoring",
        "correct_decision":                      "LEVERAGE" if adir == "below_fair_value" else "MITIGATE",
        "crystallisation_turn_index":            4,
        "anchoring_strength_at_crystallisation": t4_fs.get("anchoring_strength"),
    }

# ─── STEP 5: CURRICULUM LEVEL ────────────────────────────────────────────────

def compute_curriculum_level(processed_turns: list[dict], pricing: dict, final_price: int) -> int:
    fv = pricing["fair_value"]

    has_flip = any(
        t["factor_state"].get("decision_is_flip")
        for t in processed_turns
        if t["speaker"] == "seller"
    )
    anch_t4 = processed_turns[3]["factor_state"].get("anchoring_strength") or 0.0
    dev     = abs(final_price - fv) / fv if fv > 0 else 0.0

    if has_flip or dev > 0.20:
        return 3
    if anch_t4 > 0.60 or dev > 0.10:
        return 2
    return 1

# ─── STEP 6: RL PROMPT + REWARD SIGNALS ──────────────────────────────────────

def build_rl_prompt_text(domain: dict, pricing: dict, conv_so_far: list[dict], turn_index: int) -> str:
    lines = [
        "CONTEXT",
        f"Product: {domain['product']}. Condition: {domain['condition']}.",
        f"Asking: ${pricing['asking_price']}. "
        f"FV: ${pricing['fair_value']} ({pricing['fair_value_source']}).",
        "",
        "CONVERSATION",
    ]
    for t in conv_so_far:
        lines.append(f"Turn {t['turn_index']} [{t['speaker'].upper()}]: {t['text']}")
        if t["speaker"] == "buyer":
            vaii = t.get("vaii", {})
            if vaii.get("raw") is not None:
                lines.append(f"  VAII: {vaii['raw']} ({vaii['state']})")
    lines += [
        "",
        "TASK",
        f"You are the seller. Turn {turn_index}.",
        "Produce <reasoning>, <decision>, <seller_vaii>, <response>.",
    ]
    return "\n".join(lines)


def build_reward_signals(factor_state: dict, pricing: dict) -> dict:
    fv     = pricing["fair_value"]
    offers = factor_state.get("buyer_offers_so_far", [])
    gap    = round(abs(offers[-1] - fv) / fv, 4) if offers and fv > 0 else None

    return {
        "correct_decision":   factor_state.get("decision", "UNDECIDED"),
        "fair_value":         fv,
        "current_price_gap":  gap,
        "anchoring_strength": factor_state.get("anchoring_strength"),
    }

# ─── STEP 7: REASONING GENERATION (TASK 2) ───────────────────────────────────

async def _reasoning_api_call(
    batch_items: list[dict],
    semaphore: asyncio.Semaphore,
) -> list[Optional[str]]:
    """Single API call — returns chains list or None on hard error."""
    async with semaphore:
        try:
            resp = await client.chat.completions.create(
                model="gpt-4o",
                temperature=0.3,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": REASONING_SYSTEM_PROMPT},
                    {"role": "user",   "content": json.dumps(batch_items)},
                ],
            )
            return json.loads(resp.choices[0].message.content).get("reasoning_chains", [])
        except Exception as e:
            log.error(f"Reasoning API call error: {e}")
            return None


async def generate_reasoning_batch(
    batch_items: list[dict],
    semaphore: asyncio.Semaphore,
) -> list[Optional[str]]:
    """
    Generate reasoning chains for a batch of seller turns.
    If GPT-4o returns a partial result, the items that were dropped are retried
    individually so every turn gets a proper reasoning chain.
    Returns a list aligned with batch_items — no None entries if all succeed.
    """
    # Try full batch first
    chains = await _reasoning_api_call(batch_items, semaphore)

    if chains is not None and len(chains) == len(batch_items):
        return chains

    # Partial or failed — retry missing items one by one
    if chains is None:
        chains = []

    n_got     = len(chains)
    n_missing = len(batch_items) - n_got
    missing_ids = [item.get("turn_id", "?") for item in batch_items[n_got:]]
    log.warning(
        f"Reasoning partial result: {n_got}/{len(batch_items)} chains — "
        f"retrying {n_missing} missing items individually: {missing_ids}"
    )

    missing_items = batch_items[n_got:]
    retry_tasks   = [_reasoning_api_call([item], semaphore) for item in missing_items]
    retry_results = await asyncio.gather(*retry_tasks)

    for item, result in zip(missing_items, retry_results):
        if result and len(result) == 1:
            chains.append(result[0])
        else:
            # Final fallback — one more attempt for this single item
            log.warning(f"Retrying single item {item.get('turn_id')} once more")
            fallback = await _reasoning_api_call([item], semaphore)
            chains.append(fallback[0] if fallback and len(fallback) == 1 else "")

    return chains

# ─── STEP 8: SFT EXAMPLES ────────────────────────────────────────────────────

def build_sft_examples(
    dialogue_id: str,
    processed_turns: list[dict],
    domain: dict,
    pricing: dict,
    curriculum_level: int,
) -> list[dict]:
    """
    Build 6 SFT training examples (one per seller turn).
    Rule 9: factor_state is NEVER included in model_input.
    """
    examples     = []
    turns_so_far = []

    for turn in processed_turns:
        if turn["speaker"] == "seller":
            tidx = turn["turn_index"]
            zone = turn["factor_state"]["zone"]

            # conversation_so_far: all prior turns; buyer VAII shown, no factor_state
            conv_so_far = []
            for t in turns_so_far:
                entry = {
                    "turn":    t["turn_index"],
                    "speaker": t["speaker"],
                    "text":    t["text"],
                    "vaii":    t.get("vaii") if t["speaker"] == "buyer" else None,
                }
                conv_so_far.append(entry)

            context_block = (
                f"Product: {domain['product']}. Condition: {domain['condition']}. "
                f"Asking: ${pricing['asking_price']}. FV: ${pricing['fair_value']}."
            )
            task_text = (
                f"It is now your turn (Turn {tidx}). "
                f"Produce <reasoning>, <decision>, <seller_vaii>, <response>."
            )

            examples.append({
                "example_id":       f"{dialogue_id}_t{tidx:02d}",
                "turn_index":       tidx,
                "zone":             zone,
                "curriculum_level": curriculum_level,
                "model_input": {
                    "system_prompt":       SYSTEM_PROMPT_SFT,
                    "context_block":       context_block,
                    "conversation_so_far": conv_so_far,
                    "task":                task_text,
                    # factor_state deliberately excluded (Rule 9)
                },
                "model_target": {
                    "reasoning":   turn.get("reasoning", ""),
                    "decision":    turn["factor_state"]["decision"],
                    "seller_vaii": turn["seller_vaii"]["raw"],
                    "response":    turn["text"],
                },
                "decision_ground_truth": {
                    "correct_decision":   turn["factor_state"]["decision"],
                    "decision_is_flip":   turn["factor_state"].get("decision_is_flip", False),
                    "previous_decision":  turn["factor_state"].get("previous_decision"),
                    "crystallisation_turn": turn.get("crystallisation_turn", False),
                },
            })

        turns_so_far.append(turn)

    return examples

# ─── STEPS 9–10: RL PROMPTS + OUTPUT WRITE ───────────────────────────────────

def assemble_and_write(dialogue_id: str, state: dict) -> dict:
    """
    Attach rl_prompt/reward_signals to seller turns, build rl_prompts list,
    build sft_examples, finalise pricing, write output JSON. Returns the doc.
    """
    processed = state["processed"]
    seed      = state["seed"]
    dom       = seed["domain"]
    pr        = seed["pricing"]
    cl        = state["cl"]
    fp        = state["final_price"]

    turns_so_far     = []
    rl_prompts_list  = []

    for turn in processed:
        if turn["speaker"] == "seller":
            tidx   = turn["turn_index"]
            rp_txt = build_rl_prompt_text(dom, pr, turns_so_far, tidx)
            rs     = build_reward_signals(turn["factor_state"], pr)
            # Attach directly to turn object (per-turn access)
            turn["rl_prompt"]      = rp_txt
            turn["reward_signals"] = rs
            rl_prompts_list.append({
                "prompt_id":        f"{dialogue_id}_t{tidx:02d}",
                "dialogue_id":      dialogue_id,
                "turn_index":       tidx,
                "curriculum_level": cl,
                "prompt_text":      rp_txt,
                "reward_signals":   rs,
            })
        turns_so_far.append(turn)

    # Build SFT examples (reasoning must already be in turns from Phase 2)
    sft = build_sft_examples(dialogue_id, processed, dom, pr, cl)

    # Finalise pricing fields
    pr["final_price"]      = fp
    pr["within_fair_band"] = (
        abs(fp - pr["fair_value"]) / pr["fair_value"] <= 0.10
        if pr["fair_value"] > 0 else False
    )

    doc = {
        "schema_version":  "2.0",
        "domain":          seed["domain"],
        "pricing":         pr,
        "bias_setup":      state["bias_setup"],
        "turns":           processed,
        "sft_examples":    sft,
        "rl_prompts":      rl_prompts_list,
        "curriculum_level": cl,
        "provenance":      seed["provenance"],
    }

    (OUTPUT_DIR / f"{dialogue_id}.json").write_text(
        json.dumps(doc, indent=2, ensure_ascii=False)
    )
    return doc

# ─── STEP 11: OUTPUT VALIDATION ──────────────────────────────────────────────

def validate_v2(doc: dict, dialogue_id: str) -> list[str]:
    errs  = []
    turns = doc.get("turns", [])

    if len(turns) != 12:
        errs.append(f"expected 12 turns, got {len(turns)}")
        return errs  # can't safely index below

    # Speaker alternation
    for i, t in enumerate(turns):
        expected = "buyer" if i % 2 == 0 else "seller"
        if t["speaker"] != expected:
            errs.append(f"T{i+1} speaker={t['speaker']!r}, expected {expected!r}")

    # T1 has price
    if not turns[0].get("price_offered"):
        errs.append("T1 missing price_offered")

    # T3 price differs from T1
    p1, p3 = turns[0].get("price_offered"), turns[2].get("price_offered")
    if p3 is None:
        errs.append("T3 missing price_offered")
    elif p3 == p1:
        errs.append(f"T3 price ({p3}) equals T1 price ({p1})")

    # Crystallisation exactly at T4
    cryst_turns = [t for t in turns if t.get("crystallisation_turn")]
    if len(cryst_turns) != 1:
        errs.append(f"expected exactly 1 crystallisation_turn, found {len(cryst_turns)}")
    elif cryst_turns[0]["turn_index"] != 4:
        errs.append(f"crystallisation_turn must be T4, found T{cryst_turns[0]['turn_index']}")

    # Per-turn checks
    for t in turns:
        ti = t["turn_index"]
        if t["speaker"] == "seller":
            if not t.get("reasoning"):
                errs.append(f"T{ti} missing reasoning")
            sv = t.get("seller_vaii", {}).get("raw")
            if sv is not None and not (0.05 <= sv <= 0.45):
                errs.append(f"T{ti} seller_vaii={sv} outside [0.05, 0.45]")
            if t.get("seller_vaii", {}).get("state") != "calm":
                errs.append(f"T{ti} seller_vaii.state is not 'calm'")
        else:
            vaii = t.get("vaii", {})
            bv   = vaii.get("raw")
            if bv is not None:
                if not (0.05 <= bv <= 0.95):
                    errs.append(f"T{ti} buyer VAII={bv} outside [0.05, 0.95]")
                exp_state = "calm" if bv < 0.50 else "stressed"
                if vaii.get("state") != exp_state:
                    errs.append(f"T{ti} VAII state mismatch: raw={bv}, state={vaii.get('state')!r}")

    # T11 buyer VAII ≤ 0.45
    t11 = turns[10]
    if t11.get("vaii", {}).get("raw", 0) > 0.45:
        errs.append(f"T11 buyer VAII {t11['vaii']['raw']} > 0.45")

    # At least one of T5/T7/T9 buyer VAII ≥ 0.50
    mid_buyer_vaiiis = [
        turns[i].get("vaii", {}).get("raw", 0)
        for i in [4, 6, 8]  # T5=idx4, T7=idx6, T9=idx8
    ]
    if max(mid_buyer_vaiiis) < 0.50:
        errs.append(f"no buyer VAII ≥ 0.50 in T5/T7/T9 (values: {mid_buyer_vaiiis})")

    # T12 contains agreement
    t12_text = turns[11].get("text", "").lower()
    agreement_words = ["deal", "agreed", "agree", "done", "sold", "accept", "finali", "sure"]
    if not any(w in t12_text for w in agreement_words):
        errs.append(f"T12 text lacks agreement signal")

    # sft_examples and rl_prompts counts
    n_sft = len(doc.get("sft_examples", []))
    if n_sft != 6:
        errs.append(f"expected 6 sft_examples, got {n_sft}")
    n_rl = len(doc.get("rl_prompts", []))
    if n_rl != 6:
        errs.append(f"expected 6 rl_prompts, got {n_rl}")

    # Final price within ±10% FV
    pricing = doc.get("pricing", {})
    fv      = pricing.get("fair_value", 1)
    fp_out  = pricing.get("final_price")
    if fp_out is not None and fv > 0:
        dev = abs(fp_out - fv) / fv
        if dev > 0.10:
            errs.append(f"final_price={fp_out} is {dev:.1%} from FV={fv}")

    # Curriculum level
    if doc.get("curriculum_level") not in [1, 2, 3]:
        errs.append(f"invalid curriculum_level: {doc.get('curriculum_level')}")

    # ── SEMANTIC CHECKS ──────────────────────────────────────────────────────
    errs += semantic_validate(turns, fv)

    return errs


def semantic_validate(turns: list[dict], fv: float) -> list[str]:
    """
    Six semantic quality checks. Returns list of warning strings.
    These are logged as warnings (not hard failures) in the pipeline,
    but counted in the validation report for quality monitoring.
    """
    warns = []

    # Determine strategy from T4 decision
    t4 = next((t for t in turns if t.get("turn_index") == 4), None)
    strategy = t4["factor_state"].get("decision", "UNDECIDED") if t4 else "UNDECIDED"
    is_mitigate = strategy == "MITIGATE"
    is_leverage = strategy == "LEVERAGE"

    buyer_turns  = [t for t in turns if t["speaker"] == "buyer"]
    seller_turns = [t for t in turns if t["speaker"] == "seller"]
    buyer_prices = [t.get("price_offered") for t in buyer_turns]

    # ── Check 1: Forbidden LEVERAGE phrases in MITIGATE buyer turns ──────────
    if is_mitigate:
        for t in buyer_turns:
            price = t.get("price_offered") or 0
            if price > fv:  # only flag turns where buyer is still above FV
                text_lo = t["text"].lower()
                hit = next((p for p in _LEVERAGE_BUYER_PHRASES if p in text_lo), None)
                if hit:
                    warns.append(
                        f"SEM1 T{t['turn_index']}: MITIGATE buyer above FV uses "
                        f"LEVERAGE phrase {hit!r}"
                    )

    # ── Check 2: Seller price_offered below FV in MITIGATE turns ─────────────
    if is_mitigate:
        for t in seller_turns:
            sp = t.get("price_offered")
            if sp is not None and sp < fv:
                warns.append(
                    f"SEM2 T{t['turn_index']}: MITIGATE seller price_offered "
                    f"{sp} < FV {fv}"
                )

    # ── Check 3: Buyer price monotonicity ────────────────────────────────────
    prices = [p for p in buyer_prices if p is not None]
    for i in range(1, len(prices)):
        if is_leverage and prices[i] < prices[i - 1]:
            warns.append(
                f"SEM3 buyer T{buyer_turns[i]['turn_index']}: LEVERAGE price "
                f"decreased {prices[i-1]} → {prices[i]}"
            )
        elif is_mitigate and prices[i] > prices[i - 1]:
            warns.append(
                f"SEM3 buyer T{buyer_turns[i]['turn_index']}: MITIGATE price "
                f"increased {prices[i-1]} → {prices[i]}"
            )

    # ── Check 4: Price-text consistency (price_offered vs text) ──────────────
    price_pat = re.compile(r'\$\s*(\d{2,4}(?:,\d{3})*)')
    for t in turns:
        stored = t.get("price_offered")
        if stored is None:
            continue
        text_prices = [
            int(m.replace(",", ""))
            for m in price_pat.findall(t["text"])
        ]
        if text_prices:
            closest = min(text_prices, key=lambda x: abs(x - stored))
            if abs(closest - stored) > 5:
                warns.append(
                    f"SEM4 T{t['turn_index']}: price_offered={stored} but "
                    f"text prices={text_prices}"
                )

    # ── Check 5: "Final offer" consistency ───────────────────────────────────
    final_offer_turn = None
    for t in buyer_turns:
        if any(p in t["text"].lower() for p in _FINAL_OFFER_PHRASES):
            final_offer_turn = t["turn_index"]
            final_offer_price = t.get("price_offered")
            break
    if final_offer_turn is not None and final_offer_price is not None:
        for t in buyer_turns:
            if t["turn_index"] > final_offer_turn:
                later_price = t.get("price_offered")
                if later_price is not None:
                    if is_leverage and later_price < final_offer_price:
                        warns.append(
                            f"SEM5 T{t['turn_index']}: buyer said 'final offer' "
                            f"at T{final_offer_turn} (${final_offer_price}) but "
                            f"offered lower ${later_price} later"
                        )
                    elif is_mitigate and later_price > final_offer_price:
                        warns.append(
                            f"SEM5 T{t['turn_index']}: buyer said 'final offer' "
                            f"at T{final_offer_turn} (${final_offer_price}) but "
                            f"offered higher ${later_price} later"
                        )

    # ── Check 6: LEVERAGE framing in MITIGATE seller turns ───────────────────
    if is_mitigate:
        for t in seller_turns:
            text_lo = t["text"].lower()
            hit = next((p for p in _LEVERAGE_SELLER_PHRASES if p in text_lo), None)
            if hit:
                warns.append(
                    f"SEM6 T{t['turn_index']}: MITIGATE seller uses LEVERAGE "
                    f"framing {hit!r}"
                )

    return warns

# ─── STEP 13: STATISTICS ─────────────────────────────────────────────────────

def write_statistics(total_input: int, failed_count: int) -> None:
    """Compute statistics by scanning all written output files."""
    stats = {
        "total_input_dialogues":  total_input,
        "successfully_processed": 0,
        "failed":                 failed_count,
        "dialogue_structure": {
            "turns_per_dialogue": 12, "buyer_turns": 6, "seller_turns": 6,
            "pre_cryst_turns": 3, "crystallisation_turn": 4, "post_cryst_turns": 9,
        },
        "curriculum_distribution": {"level_1": 0, "level_2": 0, "level_3": 0},
        "decision_distribution":   {"LEVERAGE": 0, "MITIGATE": 0},
        "flip_dialogues":          0,
        "total_sft_examples":      0,
        "total_rl_prompts":        0,
        "sft_by_zone":             {"pre_crystallisation": 0, "post_crystallisation": 0},
        "within_fair_band_count":  0,
        "within_fair_band_pct":    0.0,
        "vaii_distribution":       {"buyer_calm": 0, "buyer_stressed": 0, "seller_calm": 0},
        "branch_B_generated":      False,
    }

    output_files = sorted(OUTPUT_DIR.glob("dialogue_*.json"))
    stats["successfully_processed"] = len(output_files)

    for fpath in output_files:
        try:
            doc = json.loads(fpath.read_text())
        except Exception:
            continue

        cl = doc.get("curriculum_level", 1)
        stats["curriculum_distribution"][f"level_{cl}"] += 1

        bs = doc.get("bias_setup", {})
        dec = bs.get("correct_decision", "LEVERAGE")
        if dec in stats["decision_distribution"]:
            stats["decision_distribution"][dec] += 1

        for t in doc.get("turns", []):
            if t["speaker"] == "seller":
                if t["factor_state"].get("decision_is_flip"):
                    stats["flip_dialogues"] += 1
                    break  # count once per dialogue

        pricing = doc.get("pricing", {})
        if pricing.get("within_fair_band"):
            stats["within_fair_band_count"] += 1

        stats["total_sft_examples"] += len(doc.get("sft_examples", []))
        stats["total_rl_prompts"]   += len(doc.get("rl_prompts", []))

        for sft in doc.get("sft_examples", []):
            zone = sft.get("zone", "post_crystallisation")
            if zone in stats["sft_by_zone"]:
                stats["sft_by_zone"][zone] += 1

        for t in doc.get("turns", []):
            if t["speaker"] == "buyer":
                state = t.get("vaii", {}).get("state", "calm")
                key   = "buyer_calm" if state == "calm" else "buyer_stressed"
                stats["vaii_distribution"][key] += 1
            else:
                stats["vaii_distribution"]["seller_calm"] += 1

    n = stats["successfully_processed"]
    if n > 0:
        stats["within_fair_band_pct"] = round(stats["within_fair_band_count"] / n * 100, 2)

    (V2_ROOT / "dataset_statistics.json").write_text(json.dumps(stats, indent=2))
    log.info(
        f"Statistics: {stats['successfully_processed']} success, "
        f"{stats['failed']} failed, "
        f"{stats['within_fair_band_pct']}% within fair band"
    )

# ─── PHASE 1: SINGLE DIALOGUE ────────────────────────────────────────────────

async def phase1_single(
    fpath: Path,
    dialogue_id: str,
    semaphore: asyncio.Semaphore,
    processed_store: dict,
    failed_ids: set,
) -> None:
    """
    Generate one dialogue (Task 1), process turns, save intermediate.
    Skips if intermediate already exists (resume support).
    """
    interm_path = INTERM_DIR / f"{dialogue_id}.json"

    # Resume from intermediate if already generated
    if interm_path.exists():
        try:
            state = json.loads(interm_path.read_text())
            processed_store[dialogue_id] = state
            return
        except Exception as e:
            log.warning(f"{dialogue_id}: corrupt intermediate, regenerating ({e})")

    try:
        v1_doc = json.loads(fpath.read_text())
        seed   = extract_seed(v1_doc, dialogue_id)
    except Exception as e:
        log.error(f"{dialogue_id}: seed extraction failed: {e}")
        failed_ids.add(dialogue_id)
        _log_failure(dialogue_id, "seed_extraction", str(e))
        return

    result = await generate_dialogue(dialogue_id, seed, semaphore)
    if result is None:
        failed_ids.add(dialogue_id)
        _log_failure(dialogue_id, "dialogue_generation", "3 retries exhausted")
        log.warning(f"  ✗ FAILED  {dialogue_id}")
        return

    processed  = process_turns(result["turns"], seed["pricing"]["fair_value"], result.get("vaii_plan"))
    bias_setup = compute_bias_setup(processed, seed["pricing"])
    cl         = compute_curriculum_level(processed, seed["pricing"], result["final_price"])

    state = {
        "seed":        seed,
        "processed":   processed,
        "bias_setup":  bias_setup,
        "cl":          cl,
        "final_price": result["final_price"],
    }
    processed_store[dialogue_id] = state

    try:
        interm_path.write_text(json.dumps(state, indent=2, ensure_ascii=False))
        product  = seed["domain"]["product"]
        strategy = seed["generation_params"]["correct_decision"]
        fv       = seed["pricing"]["fair_value"]
        fp       = result["final_price"]
        n_done   = len(processed_store)
        log.info(
            f"  ✓ saved   {dialogue_id} [{n_done:>4}/3154] "
            f"{strategy:8s} | {product} | FV=${fv}  →  final=${fp}  CL{cl}"
        )
    except Exception as e:
        log.warning(f"{dialogue_id}: could not save intermediate: {e}")


def _log_failure(dialogue_id: str, step: str, error: str) -> None:
    with open(FAILED_LOG, "a") as f:
        f.write(json.dumps({"id": dialogue_id, "step": step, "error": error}) + "\n")

# ─── MAIN ORCHESTRATION ───────────────────────────────────────────────────────

async def process_all() -> None:
    # ── Load checkpoint ───────────────────────────────────────────────────────
    completed: set[str] = set()
    if CHECKPOINT.exists():
        try:
            completed = set(json.loads(CHECKPOINT.read_text()))
            log.info(f"Checkpoint loaded: {len(completed)} already completed")
        except Exception:
            log.warning("Checkpoint unreadable, starting fresh")

    # ── Build sorted file list, assign sequential IDs ─────────────────────────
    all_files = sorted(INPUT_DIR.rglob("dialogue_*.json"))
    total     = len(all_files)
    id_map    = {str(f): f"dialogue_{i+1:04d}" for i, f in enumerate(all_files)}
    log.info(f"Total input files: {total}")

    semaphore       = asyncio.Semaphore(10)
    processed_store: dict = {}
    failed_ids:      set  = set()

    # ══ PHASE 1: Generate all dialogues ══════════════════════════════════════
    log.info("=== PHASE 1: Dialogue generation ===")

    phase1_tasks = []
    for fpath_str, dialogue_id in id_map.items():
        if dialogue_id in completed:
            # Reload intermediate for statistics / Phase 2 reasoning check
            interm_path = INTERM_DIR / f"{dialogue_id}.json"
            if interm_path.exists():
                try:
                    processed_store[dialogue_id] = json.loads(interm_path.read_text())
                except Exception:
                    pass
            continue
        phase1_tasks.append(
            phase1_single(Path(fpath_str), dialogue_id, semaphore, processed_store, failed_ids)
        )

    # Run in chunks of 100 for progress visibility
    CHUNK = 100
    for i in range(0, len(phase1_tasks), CHUNK):
        await asyncio.gather(*phase1_tasks[i : i + CHUNK])
        done = min(i + CHUNK, len(phase1_tasks))
        log.info(f"Phase 1 progress: {done}/{len(phase1_tasks)} tasks done "
                 f"({len(processed_store)} in store, {len(failed_ids)} failed)")

    log.info(f"Phase 1 complete: {len(processed_store)} generated, {len(failed_ids)} failed")

    # ══ PHASE 2: Generate reasoning chains ═══════════════════════════════════
    log.info("=== PHASE 2: Reasoning chain generation ===")

    # Collect all seller turns that still need reasoning
    reasoning_items: list[dict] = []
    for dialogue_id, state in processed_store.items():
        turns_so_far: list[dict] = []
        for turn in state["processed"]:
            if turn["speaker"] == "seller" and not turn.get("reasoning"):
                prior_buyer = next(
                    (t for t in reversed(turns_so_far) if t["speaker"] == "buyer"), None
                )
                conv_str = "\n".join(
                    f"Turn {t['turn_index']} [{t['speaker'].upper()}]: {t['text']}"
                    for t in turns_so_far
                )
                reasoning_items.append({
                    "turn_id":             f"{dialogue_id}_t{turn['turn_index']:02d}",
                    "dialogue_id":         dialogue_id,
                    "turn_index":          turn["turn_index"],
                    "conversation_so_far": conv_str,
                    "factor_state":        turn["factor_state"],
                    "fair_value":          state["seed"]["pricing"]["fair_value"],
                    "asking_price":        state["seed"]["pricing"]["asking_price"],
                    "product":             state["seed"]["domain"]["product"],
                    "vaii_prior_turn":     prior_buyer.get("vaii") if prior_buyer else None,
                })
            turns_so_far.append(turn)

    log.info(f"Phase 2: {len(reasoning_items)} reasoning items across {len(reasoning_items)//6} dialogues")

    BATCH_SIZE = 24  # 4 dialogues × 6 seller turns

    # Track which dialogues were touched by each batch so we can flush them
    dialogues_touched_lock = asyncio.Lock()
    dialogues_touched: set[str] = set()

    async def run_reasoning_batch(batch: list[dict]) -> None:
        chains = await generate_reasoning_batch(batch, semaphore)
        touched = set()
        for item, chain in zip(batch, chains):
            if not chain:
                log.warning(f"No reasoning chain returned for {item['turn_id']}")
                continue
            did = item["dialogue_id"]
            for turn in processed_store.get(did, {}).get("processed", []):
                if turn["turn_index"] == item["turn_index"] and turn["speaker"] == "seller":
                    turn["reasoning"] = chain
                    touched.add(did)
                    break

        # Flush intermediates for every dialogue touched by this batch immediately
        for did in touched:
            try:
                interm_path = INTERM_DIR / f"{did}.json"
                interm_path.write_text(
                    json.dumps(processed_store[did], indent=2, ensure_ascii=False)
                )
            except Exception as e:
                log.warning(f"{did}: could not flush intermediate after reasoning: {e}")

        if touched:
            sorted_ids = sorted(touched)
            log.info(
                f"  reasoning saved: {sorted_ids[0]} … {sorted_ids[-1]} "
                f"({len(touched)} dialogues)"
            )

        async with dialogues_touched_lock:
            dialogues_touched.update(touched)

    batches     = [reasoning_items[i : i + BATCH_SIZE] for i in range(0, len(reasoning_items), BATCH_SIZE)]
    batch_tasks = [run_reasoning_batch(b) for b in batches]

    for i in range(0, len(batch_tasks), CHUNK):
        await asyncio.gather(*batch_tasks[i : i + CHUNK])
        done = min(i + CHUNK, len(batch_tasks))
        log.info(f"Phase 2 progress: {done}/{len(batch_tasks)} batches done")

    log.info(f"Phase 2 complete. Reasoning flushed to {len(dialogues_touched)} intermediates.")

    # ══ PHASE 3: Assemble and write output files ══════════════════════════════
    log.info("=== PHASE 3: Output assembly and write ===")

    validation_report: dict = {}

    for dialogue_id, state in processed_store.items():
        if dialogue_id in completed:
            validation_report[dialogue_id] = "skipped_already_complete"
            continue
        try:
            doc  = assemble_and_write(dialogue_id, state)
            errs = validate_v2(doc, dialogue_id)
            validation_report[dialogue_id] = errs if errs else "ok"
            if errs:
                log.warning(f"{dialogue_id} validation errors: {errs}")
            completed.add(dialogue_id)
        except Exception as e:
            log.error(f"{dialogue_id} Phase 3 error: {e}")
            failed_ids.add(dialogue_id)
            _log_failure(dialogue_id, "output_assembly", str(e))

        if len(completed) % 100 == 0:
            CHECKPOINT.write_text(json.dumps(sorted(completed)))
            log.info(f"Checkpoint: {len(completed)} completed")

    # Final checkpoint + report
    CHECKPOINT.write_text(json.dumps(sorted(completed)))

    (V2_ROOT / "validation_report.json").write_text(
        json.dumps(validation_report, indent=2)
    )

    write_statistics(total, len(failed_ids))

    # Summary
    ok_count   = sum(1 for v in validation_report.values() if v == "ok")
    err_count  = sum(1 for v in validation_report.values() if isinstance(v, list) and v)
    log.info(
        f"Pipeline complete. "
        f"Success: {ok_count}, "
        f"Validation warnings: {err_count}, "
        f"Failed: {len(failed_ids)}, "
        f"Total input: {total}"
    )


if __name__ == "__main__":
    asyncio.run(process_all())
