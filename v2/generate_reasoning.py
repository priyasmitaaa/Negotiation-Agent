"""
generate_reasoning.py — Generate per-seller-turn reasoning using Gemini 2.5 Pro.

For each seller turn in each dialogue, generates a reasoning chain that the
seller would have thought BEFORE speaking, incorporating:
  - Negotiation state (price gap, anchor, harm direction, zone)
  - Buyer emotion at the prior turn (labels, intensity, valence, notes)
  - Seller emotion at prior turns (pattern so far)
  - Decision (UNDECIDED / LEVERAGE / MITIGATE) with justification
  - Strategy plan for the response

Written back to dataset/dialogue_XXXX.json under turn["reasoning"].
Fully resumable — skips seller turns that already have a reasoning field.

Batches 4 seller turns per Gemini call (across dialogues).

Usage:
  export VERTEX_API_KEY=<key>
  python3 Qwen3-tts/v2/generate_reasoning.py --test
  python3 Qwen3-tts/v2/generate_reasoning.py --sample 50
  python3 Qwen3-tts/v2/generate_reasoning.py
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types

# ── Paths ─────────────────────────────────────────────────────────────────────
V2_ROOT     = Path(__file__).parent
DATASET_DIR = V2_ROOT / "dataset"
LOG_FILE    = V2_ROOT / "reasoning_generation.log"

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
log = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google").setLevel(logging.WARNING)

# ── Gemini ────────────────────────────────────────────────────────────────────
GEMINI_MODEL       = "gemini-2.5-pro"
TURNS_PER_CALL     = 4    # seller turns per Gemini call
CONCURRENCY        = 1    # sequential to avoid file race conditions

client = genai.Client(
    vertexai=True,
    api_key=os.environ["VERTEX_API_KEY"],
)

REASONING_SYSTEM_PROMPT = """\
You are generating internal reasoning chains for an AI negotiation seller agent \
in a second-hand electronics shop in India.

Each reasoning chain is the seller's internal monologue written BEFORE they speak \
at their turn. It must reflect what the seller observes, infers, and decides.

You will receive structured context for each seller turn. Generate reasoning that:

1. States the current negotiation state in numbers:
   - Buyer's offer(s) so far and the anchor price (first offer)
   - Fair value and asking price
   - Whether buyer is above or below fair value (harm direction) with arithmetic
   - Anchoring strength (how far buyer has moved from anchor toward fair value)
   - Zone: pre_crystallisation (still probing) or post_crystallisation (decision locked)

2. Analyses the buyer's emotional state from the PRIOR buyer turn:
   - Emotion labels, intensity, valence, and vocal notes
   - What this signals about the buyer's urgency, desperation, or leverage

3. Notes the seller's own emotional pattern so far (from seller turn emotions)

4. States the decision (UNDECIDED / LEVERAGE / MITIGATE) with clear justification:
   - LEVERAGE: buyer is below fair value or has high anchoring strength → seller holds firm
   - MITIGATE: buyer is above fair value → seller should guide down to fair range
   - UNDECIDED: first turn or insufficient information yet

5. Outlines the strategy: tone (firm/empathetic/measured), argument type \
   (market data / condition / patience / urgency), pace

Style: first person, present tense, specific numbers always, 5-8 sentences.
No hedging. The seller commits to their read.

If decision_is_flip is true, explicitly state: \
"Decision flips [old] → [new] because buyer offer ₹N crossed fair value ₹FV."

Return ONLY valid JSON:
{
  "reasoning_chains": ["chain_for_item_1", "chain_for_item_2", ...]
}
Array order MUST match input order. Length MUST equal number of items sent.
"""


# ── Build context for one seller turn ────────────────────────────────────────

def _build_item(dialogue_id: str, data: dict, turn: dict, turns_so_far: list[dict]) -> dict:
    pricing  = data["seed"]["pricing"]
    gen_params = data["seed"]["generation_params"]

    # Prior buyer turn emotion
    prior_buyer = next(
        (t for t in reversed(turns_so_far) if t["speaker"] == "buyer"),
        None
    )
    buyer_emotion = prior_buyer.get("emotion") if prior_buyer else None

    # Seller emotions so far (pattern)
    seller_emotions_so_far = [
        {"turn": t["turn_index"], "labels": t["emotion"]["labels"],
         "intensity": t["emotion"]["intensity"]}
        for t in turns_so_far
        if t["speaker"] == "seller" and t.get("emotion")
    ]

    # Conversation transcript so far
    conv = "\n".join(
        f"Turn {t['turn_index']} [{t['speaker'].upper()}]: {t['text']}"
        for t in turns_so_far
    )

    fs = turn["factor_state"]

    return {
        "turn_id":              f"{dialogue_id}_t{turn['turn_index']:02d}",
        "dialogue_id":          dialogue_id,
        "turn_index":           turn["turn_index"],
        "product":              data["seed"]["domain"]["product"],
        "fair_value":           pricing["fair_value"],
        "asking_price":         pricing["asking_price"],
        "buyer_profile":        gen_params.get("buyer_profile", "unknown"),
        "correct_decision":     gen_params.get("correct_decision", "unknown"),
        "buyer_offers_so_far":  fs["buyer_offers_so_far"],
        "anchor_price":         fs["anchor_price"],
        "harm_direction":       fs["harm_direction"],
        "anchoring_strength":   fs["anchoring_strength"],
        "decision":             fs["decision"],
        "zone":                 fs["zone"],
        "decision_is_flip":     fs["decision_is_flip"],
        "previous_decision":    fs["previous_decision"],
        "buyer_emotion_prior_turn": buyer_emotion,
        "seller_emotions_so_far":   seller_emotions_so_far,
        "conversation_so_far":      conv,
    }


# ── Collect seller turns needing reasoning ────────────────────────────────────

def _collect_missing(data: dict, dialogue_id: str) -> list[dict]:
    items = []
    turns_so_far = []
    for turn in data["processed"]:
        if turn["speaker"] == "seller" and not turn.get("reasoning"):
            items.append(_build_item(dialogue_id, data, turn, turns_so_far))
        turns_so_far.append(turn)
    return items


# ── Single Gemini call ────────────────────────────────────────────────────────

async def _gemini_call(
    items: list[dict],
    semaphore: asyncio.Semaphore,
) -> Optional[list[str]]:
    async with semaphore:
        try:
            response = await client.aio.models.generate_content(
                model=GEMINI_MODEL,
                contents=json.dumps(items, ensure_ascii=False),
                config=types.GenerateContentConfig(
                    system_instruction=REASONING_SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    temperature=0.3,
                ),
            )
            return json.loads(response.text).get("reasoning_chains", [])
        except Exception as e:
            log.error(f"Gemini call failed: {e}")
            return None


# ── Process a batch of items ──────────────────────────────────────────────────

async def process_batch(
    items: list[dict],
    store: dict,
    semaphore: asyncio.Semaphore,
) -> int:
    dids = list({item["dialogue_id"] for item in items})
    log.info(f"  Calling Gemini for {dids} — {len(items)} seller turns")

    chains = await _gemini_call(items, semaphore)
    if chains is None:
        log.error(f"  Failed for {dids}")
        return 0

    if len(chains) != len(items):
        log.warning(f"  Got {len(chains)} chains for {len(items)} items — partial")

    # Pre-load dataset versions once per dialogue
    for did in dids:
        dp = DATASET_DIR / f"{did}.json"
        if dp.exists():
            try:
                store[did] = json.loads(dp.read_text())
            except Exception:
                pass

    written = 0
    touched = set()
    for item, chain in zip(items, chains):
        if not chain:
            log.warning(f"  Empty chain for {item['turn_id']}")
            continue
        did  = item["dialogue_id"]
        tidx = item["turn_index"]
        for turn in store[did]["processed"]:
            if turn["turn_index"] == tidx and turn["speaker"] == "seller":
                turn["reasoning"] = chain
                touched.add(did)
                written += 1
                break

    for did in sorted(touched):
        try:
            (DATASET_DIR / f"{did}.json").write_text(
                json.dumps(store[did], indent=2, ensure_ascii=False)
            )
        except Exception as e:
            log.warning(f"  Flush failed for {did}: {e}")

    log.info(f"  Written {written} reasoning chains across {len(touched)} dialogues")
    return written


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(sample: Optional[int] = None, test_mode: bool = False):
    log.info("=== Reasoning Generation — Gemini 2.5 Pro ===")
    log.info(f"Model: {GEMINI_MODEL}  |  Batch: {TURNS_PER_CALL} turns/call  |  Concurrency: {CONCURRENCY}")

    log.info("Loading dataset…")
    store: dict[str, dict] = {}
    for fpath in sorted(DATASET_DIR.glob("dialogue_*.json")):
        try:
            store[fpath.stem] = json.loads(fpath.read_text())
        except Exception as e:
            log.warning(f"Could not load {fpath.name}: {e}")
    log.info(f"Loaded {len(store)} dialogues.")

    # Collect all seller turns needing reasoning
    all_items = []
    for did, data in sorted(store.items()):
        missing = _collect_missing(data, did)
        all_items.extend(missing)

    log.info(f"Seller turns needing reasoning: {len(all_items)}")

    if not all_items:
        log.info("Nothing to do — all seller turns already have reasoning.")
        return

    if sample and sample < len(all_items):
        import random; random.seed(42)
        # sample whole dialogues not individual turns
        all_dids = list({item["dialogue_id"] for item in all_items})
        sampled_dids = set(random.sample(all_dids, min(sample, len(all_dids))))
        all_items = [i for i in all_items if i["dialogue_id"] in sampled_dids]
        log.info(f"Sampling {len(sampled_dids)} dialogues → {len(all_items)} turns.")

    if test_mode:
        log.info("\n=== TEST MODE — first 4 turns, no writes ===")
        semaphore = asyncio.Semaphore(1)
        test_items = all_items[:4]
        for item in test_items:
            log.info(f"  {item['turn_id']}: decision={item['decision']} harm={item['harm_direction']} buyer_emotion={item['buyer_emotion_prior_turn']}")
        chains = await _gemini_call(test_items, semaphore)
        if chains:
            for item, chain in zip(test_items, chains):
                log.info(f"\n--- {item['turn_id']} ---\n{chain}\n")
            log.info("Test OK.")
        else:
            log.error("Test FAILED.")
        return

    # Batch into groups of TURNS_PER_CALL
    batches = [all_items[i:i+TURNS_PER_CALL] for i in range(0, len(all_items), TURNS_PER_CALL)]
    log.info(f"Total batches: {len(batches)}")

    semaphore     = asyncio.Semaphore(CONCURRENCY)
    total_written = 0

    for i, batch in enumerate(batches):
        n = await process_batch(batch, store, semaphore)
        total_written += n
        if (i + 1) % 10 == 0 or (i + 1) == len(batches):
            pct = (i + 1) / len(batches) * 100
            log.info(f"Progress: {i+1}/{len(batches)} batches ({pct:.1f}%) — {total_written} chains written")

    log.info(f"=== Done. {total_written} reasoning chains written. ===")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=None,
                   help="Number of dialogues to sample")
    p.add_argument("--test", action="store_true",
                   help="Test mode: first 4 turns, no writes")
    args = p.parse_args()
    asyncio.run(main(sample=args.sample, test_mode=args.test))
