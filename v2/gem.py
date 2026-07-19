"""
gem.py — Generate reasoning chains for ~1034 missing-reasoning dialogues
using Google Gemini (Vertex AI). Fully resumable — skips turns that already
have reasoning. Flushes each intermediate file immediately after its batch.

Usage:
  pip install google-genai
  export VERTEX_API_KEY=<your-key>
  python3 Qwen3-tts/v2/gem.py

  # Dry-run test on ONE turn (no writes):
  python3 Qwen3-tts/v2/gem.py --test
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

# Identical to REASONING_SYSTEM_PROMPT in restructure_dataset.py (lines 308-337)
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

# ── Paths ─────────────────────────────────────────────────────────────────────
V2_ROOT    = Path(__file__).parent
INTERM_DIR = V2_ROOT / "intermediate"

# ── Logging ───────────────────────────────────────────────────────────────────
_LOG_FILE = V2_ROOT / "gem_reasoning.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(_LOG_FILE),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google").setLevel(logging.WARNING)

# ── Gemini client ─────────────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-2.5-pro"

client = genai.Client(
    vertexai=True,
    api_key=os.environ["VERTEX_API_KEY"],
)

BATCH_SIZE  = 24   # 4 dialogues × 6 seller turns — same as OpenAI pipeline
CONCURRENCY = 3    # max parallel Gemini calls


# ── Single Gemini API call ────────────────────────────────────────────────────

async def _gemini_call(
    batch_items: list[dict],
    semaphore: asyncio.Semaphore,
) -> Optional[list[str]]:
    async with semaphore:
        try:
            response = await client.aio.models.generate_content(
                model=GEMINI_MODEL,
                contents=json.dumps(batch_items),
                config=types.GenerateContentConfig(
                    system_instruction=REASONING_SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    temperature=0.3,
                ),
            )
            return json.loads(response.text).get("reasoning_chains", [])
        except Exception as e:
            log.error(f"Gemini API error: {e}")
            return None


# ── Batch generation with retry for partial results ───────────────────────────

async def generate_reasoning_batch(
    batch_items: list[dict],
    semaphore: asyncio.Semaphore,
) -> list[str]:
    chains = await _gemini_call(batch_items, semaphore)

    if chains is not None and len(chains) == len(batch_items):
        return chains

    if chains is None:
        chains = []

    n_got     = len(chains)
    n_missing = len(batch_items) - n_got
    missing   = batch_items[n_got:]
    log.warning(
        f"Partial result: {n_got}/{len(batch_items)} — "
        f"retrying {n_missing} items individually"
    )

    retry_results = await asyncio.gather(
        *[_gemini_call([item], semaphore) for item in missing]
    )

    for item, result in zip(missing, retry_results):
        if result and len(result) == 1:
            chains.append(result[0])
        else:
            log.warning(f"Final fallback for {item.get('turn_id')}")
            fallback = await _gemini_call([item], semaphore)
            chains.append(fallback[0] if fallback and len(fallback) == 1 else "")

    return chains


# ── Build reasoning items from a dialogue ─────────────────────────────────────

def collect_missing_items(processed_store: dict) -> list[dict]:
    items = []
    for dialogue_id, state in sorted(processed_store.items()):
        turns_so_far: list[dict] = []
        for turn in state["processed"]:
            if turn["speaker"] == "seller" and not turn.get("reasoning"):
                prior_buyer = next(
                    (t for t in reversed(turns_so_far) if t["speaker"] == "buyer"),
                    None,
                )
                conv_str = "\n".join(
                    f"Turn {t['turn_index']} [{t['speaker'].upper()}]: {t['text']}"
                    for t in turns_so_far
                )
                items.append({
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
    return items


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(test_mode: bool = False):
    log.info("=== Gemini Reasoning Generator ===")
    log.info(f"Model      : {GEMINI_MODEL}")
    log.info(f"Batch size : {BATCH_SIZE}  |  Concurrency: {CONCURRENCY}")
    log.info(f"Prompt     : imported from restructure_dataset.REASONING_SYSTEM_PROMPT")

    # Load all intermediates
    log.info("Loading intermediates…")
    processed_store: dict[str, dict] = {}
    for fpath in sorted(INTERM_DIR.glob("dialogue_*.json")):
        try:
            processed_store[fpath.stem] = json.loads(fpath.read_text())
        except Exception as e:
            log.warning(f"Could not load {fpath.name}: {e}")
    log.info(f"Loaded {len(processed_store)} intermediates.")

    reasoning_items = collect_missing_items(processed_store)
    n_dialogues = len({item["dialogue_id"] for item in reasoning_items})

    log.info(
        f"Reasoning needed: {len(reasoning_items)} seller turns across "
        f"{n_dialogues} dialogues"
    )

    if not reasoning_items:
        log.info("Nothing to do — all dialogues already have reasoning.")
        return

    # ── TEST MODE: run exactly one turn, print result, no file writes ──────────
    if test_mode:
        log.info("\n=== TEST MODE — one turn, no writes ===")
        semaphore = asyncio.Semaphore(1)
        test_item = reasoning_items[0]
        log.info(f"Testing on: {test_item['turn_id']}  product={test_item['product']}")
        log.info(f"Input:\n{json.dumps(test_item, indent=2)}")

        result = await _gemini_call([test_item], semaphore)
        if result and result[0]:
            log.info(f"\n=== GEMINI OUTPUT ===\n{result[0]}\n====================")
            log.info("Test OK — Gemini returned valid reasoning.")
        else:
            log.error("Test FAILED — no reasoning returned.")
        return

    # ── FULL RUN ───────────────────────────────────────────────────────────────
    batches   = [reasoning_items[i : i + BATCH_SIZE] for i in range(0, len(reasoning_items), BATCH_SIZE)]
    semaphore = asyncio.Semaphore(CONCURRENCY)
    completed = 0

    async def run_batch(batch_idx: int, batch: list[dict]) -> None:
        nonlocal completed
        first_id = batch[0]["dialogue_id"]
        last_id  = batch[-1]["dialogue_id"]
        log.info(
            f"Batch {batch_idx+1}/{len(batches)} — "
            f"{first_id} … {last_id} ({len(batch)} turns)"
        )

        chains  = await generate_reasoning_batch(batch, semaphore)
        touched: set[str] = set()

        for item, chain in zip(batch, chains):
            if not chain:
                log.warning(f"Empty chain for {item['turn_id']} — skipping")
                continue
            did = item["dialogue_id"]
            for turn in processed_store.get(did, {}).get("processed", []):
                if turn["turn_index"] == item["turn_index"] and turn["speaker"] == "seller":
                    turn["reasoning"] = chain
                    touched.add(did)
                    break

        # Flush intermediates immediately so Ctrl+C is safe
        for did in sorted(touched):
            try:
                (INTERM_DIR / f"{did}.json").write_text(
                    json.dumps(processed_store[did], indent=2, ensure_ascii=False)
                )
            except Exception as e:
                log.warning(f"{did}: flush failed: {e}")

        if touched:
            log.info(
                f"  reasoning saved: {sorted(touched)[0]} … {sorted(touched)[-1]} "
                f"({len(touched)} dialogues)"
            )
        completed += len([c for c in chains if c])

    for i in range(0, len(batches), CONCURRENCY):
        chunk = batches[i : i + CONCURRENCY]
        await asyncio.gather(*[run_batch(i + j, b) for j, b in enumerate(chunk)])
        pct = min(i + CONCURRENCY, len(batches)) / len(batches) * 100
        log.info(
            f"Progress: {min(i+CONCURRENCY, len(batches))}/{len(batches)} batches "
            f"({pct:.1f}%) — {completed}/{len(reasoning_items)} chains written"
        )

    log.info(f"=== Done. {completed}/{len(reasoning_items)} reasoning chains generated. ===")
    log.info(f"Log: {_LOG_FILE}")


if __name__ == "__main__":
    test = "--test" in sys.argv
    asyncio.run(main(test_mode=test))
