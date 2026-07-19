"""
detect_emotions.py — Detect emotions from audio turns using Gemini 2.5 Pro.

For each turn (buyer + seller) in each dialogue, sends the audio WAV to
Gemini and asks it to freely label emotions (no fixed taxonomy).
Results are written back into intermediate/dialogue_XXXX.json under:
  turn["emotion"] = {
      "labels":     ["hesitant", "curious"],   # free-form, 1-3 labels
      "intensity":  "medium",                  # low / medium / high
      "valence":    "negative",                # positive / negative / neutral
      "confidence": 0.82,
      "notes":      "voice trails off, rising intonation",
      "source":     "gemini-2.5-pro"
  }

Batches 2 dialogues per Gemini call (all turns of both dialogues).
Fully resumable — skips turns that already have emotion field.

Usage:
  export VERTEX_API_KEY=<your-key>
  python3 Qwen3-tts/v2/detect_emotions.py --sample 50
  python3 Qwen3-tts/v2/detect_emotions.py --sample 50 --test
  python3 Qwen3-tts/v2/detect_emotions.py          # full run
"""

import os
import sys
import json
import asyncio
import logging
import base64
import random
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types

# ── Paths ─────────────────────────────────────────────────────────────────────
V2_ROOT     = Path(__file__).parent
INTERM_DIR  = V2_ROOT / "intermediate"   # source for loading only
DATASET_DIR = V2_ROOT / "dataset"        # target for writing
AUDIO_DIR   = V2_ROOT / "tts_outputs"
LOG_FILE    = V2_ROOT / "emotion_detection.log"

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
GEMINI_MODEL = "gemini-2.5-pro"
DIALOGUES_PER_CALL = 2   # how many dialogues to batch per Gemini call
CONCURRENCY        = 1   # set to 1 to avoid race conditions on partial files

client = genai.Client(
    vertexai=True,
    api_key=os.environ["VERTEX_API_KEY"],
)

EMOTION_SYSTEM_PROMPT = """\
You are an expert audio emotion analyst. You will receive audio clips from a \
negotiation dialogue between a buyer and a seller in a second-hand electronics shop.

For each audio clip, analyse the speaker's vocal emotion freely — do NOT limit \
yourself to a fixed list. Use natural language emotion labels that best describe \
what you hear (e.g. "hesitant", "frustrated", "eager", "resigned", "suspicious", \
"relieved", "desperate", "confident", "nervous", "playful", "firm", "rushed").

You may assign 1 to 3 emotion labels per clip. Also rate:
- intensity: how strong the emotion is (low / medium / high)
- valence: overall tone (positive / negative / neutral)
- confidence: your confidence in the labels (0.0 to 1.0)
- notes: 1 sentence about the specific vocal cue that led to this label \
  (e.g. "voice trails off at end", "speaks fast with clipped words", \
  "long pause before answer")

Return ONLY valid JSON in this exact structure:
{
  "results": [
    {
      "turn_id": "<dialogue_id>_t<turn_index>",
      "labels": ["label1", "label2"],
      "intensity": "medium",
      "valence": "negative",
      "confidence": 0.85,
      "notes": "..."
    },
    ...
  ]
}
Array order MUST match input order exactly. Length MUST equal number of clips sent.
"""


# ── Audio loading ─────────────────────────────────────────────────────────────

def _load_audio_b64(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def _audio_part(b64: str) -> types.Part:
    return types.Part.from_bytes(
        data=base64.b64decode(b64),
        mime_type="audio/wav",
    )


# ── Build turn list for a dialogue ───────────────────────────────────────────

def _turns_needing_emotion(dialogue_id: str, data: dict) -> list[dict]:
    """Return turns that don't yet have an emotion field. Checks dataset/ for prior writes."""
    # If dataset/ has a newer version (already partially processed), use that
    dataset_path = DATASET_DIR / f"{dialogue_id}.json"
    if dataset_path.exists():
        try:
            data = json.loads(dataset_path.read_text())
        except Exception:
            pass

    audio_dir = AUDIO_DIR / dialogue_id
    turns = []
    for turn in data["processed"]:
        if "emotion" in turn:
            continue
        tidx     = turn["turn_index"]
        speaker  = turn["speaker"]
        wav_name = f"turn_{tidx:02d}_{speaker}.wav"
        wav_path = audio_dir / wav_name
        turns.append({
            "turn_index": tidx,
            "speaker":    speaker,
            "text":       turn["text"],
            "wav_path":   wav_path,
            "turn_id":    f"{dialogue_id}_t{tidx:02d}",
        })
    return turns


# ── Single Gemini call with audio ─────────────────────────────────────────────

async def _gemini_emotion_call(
    items: list[dict],   # list of {turn_id, wav_path, speaker, text}
    semaphore: asyncio.Semaphore,
) -> Optional[list[dict]]:
    """Send multiple audio clips in one call. Returns list of emotion dicts."""
    async with semaphore:
        # Build content: interleave text labels + audio parts
        parts = []
        valid_items = []
        for item in items:
            b64 = _load_audio_b64(item["wav_path"])
            if b64 is None:
                log.warning(f"Audio not found: {item['wav_path']} — skipping")
                continue
            parts.append(types.Part.from_text(text=(
                f"Turn ID: {item['turn_id']} | Speaker: {item['speaker'].upper()} | "
                f"Text: \"{item['text'][:100]}\""
            )))
            parts.append(_audio_part(b64))
            valid_items.append(item)

        if not parts:
            return []

        parts.append(types.Part.from_text(text=(
            f"\nAnalyse the {len(valid_items)} audio clips above. "
            f"Return JSON with {len(valid_items)} results in order."
        )))

        try:
            response = await client.aio.models.generate_content(
                model=GEMINI_MODEL,
                contents=types.Content(role="user", parts=parts),
                config=types.GenerateContentConfig(
                    system_instruction=EMOTION_SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    temperature=0.2,
                ),
            )
            parsed = json.loads(response.text)
            results = parsed.get("results", [])
            # align by turn_id
            result_map = {r["turn_id"]: r for r in results}
            return [result_map.get(item["turn_id"]) for item in valid_items]
        except Exception as e:
            log.error(f"Gemini call failed: {e}")
            return None


# ── Process a batch of dialogues ──────────────────────────────────────────────

async def process_dialogue_batch(
    dialogue_ids: list[str],
    processed_store: dict,
    semaphore: asyncio.Semaphore,
) -> int:
    """Process DIALOGUES_PER_CALL dialogues in one Gemini call. Returns count written."""
    all_items = []
    for did in dialogue_ids:
        data  = processed_store[did]
        turns = _turns_needing_emotion(did, data)
        for t in turns:
            t["dialogue_id"] = did
        all_items.extend(turns)

    if not all_items:
        log.info(f"  {dialogue_ids} — all turns already have emotions, skipping")
        return 0

    log.info(
        f"  Calling Gemini for {dialogue_ids} — {len(all_items)} turns"
    )
    results = await _gemini_emotion_call(all_items, semaphore)

    if results is None:
        log.error(f"  Failed for {dialogue_ids}")
        return 0

    # Pre-load dataset/ versions once per dialogue before iterating results
    for did in dialogue_ids:
        dataset_path = DATASET_DIR / f"{did}.json"
        if dataset_path.exists():
            try:
                processed_store[did] = json.loads(dataset_path.read_text())
            except Exception:
                pass

    written = 0
    touched_dids = set()
    for item, result in zip(all_items, results):
        if result is None:
            log.warning(f"  No result for {item['turn_id']}")
            continue
        did  = item["dialogue_id"]
        tidx = item["turn_index"]
        for turn in processed_store[did]["processed"]:
            if turn["turn_index"] == tidx:
                turn["emotion"] = {
                    "labels":     [l.lower() for l in result.get("labels", [])],
                    "intensity":  result.get("intensity", "unknown").lower(),
                    "valence":    result.get("valence", "unknown").lower(),
                    "confidence": result.get("confidence", 0.0),
                    "notes":      result.get("notes", ""),
                    "source":     GEMINI_MODEL,
                }
                touched_dids.add(did)
                written += 1
                break

    # Flush to dataset/ immediately
    for did in sorted(touched_dids):
        try:
            (DATASET_DIR / f"{did}.json").write_text(
                json.dumps(processed_store[did], indent=2, ensure_ascii=False)
            )
        except Exception as e:
            log.warning(f"  Flush failed for {did}: {e}")

    log.info(f"  Written {written} emotion fields across {len(touched_dids)} dialogues")
    return written


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(sample: Optional[int] = None, test_mode: bool = False):
    log.info("=== Emotion Detection — Gemini 2.5 Pro ===")
    log.info(f"Model: {GEMINI_MODEL}  |  Batch: {DIALOGUES_PER_CALL} dialogues/call  |  Concurrency: {CONCURRENCY}")

    # Load intermediates
    log.info("Loading intermediates…")
    processed_store: dict[str, dict] = {}
    for fpath in sorted(INTERM_DIR.glob("dialogue_*.json")):
        try:
            processed_store[fpath.stem] = json.loads(fpath.read_text())
        except Exception as e:
            log.warning(f"Could not load {fpath.name}: {e}")
    log.info(f"Loaded {len(processed_store)} intermediates.")

    # Filter to dialogues that have audio + have turns needing emotion
    dialogue_ids = []
    for did, data in sorted(processed_store.items()):
        audio_dir = AUDIO_DIR / did
        if not audio_dir.exists():
            continue
        turns_needed = _turns_needing_emotion(did, data)
        if turns_needed:
            dialogue_ids.append(did)

    log.info(f"Dialogues needing emotion detection: {len(dialogue_ids)}")

    if not dialogue_ids:
        log.info("Nothing to do — all turns already have emotions.")
        return

    # Sample subset if requested
    if sample and sample < len(dialogue_ids):
        random.seed(42)
        dialogue_ids = random.sample(dialogue_ids, sample)
        log.info(f"Sampling {sample} dialogues.")

    # Test mode: run exactly one batch, print result, no writes
    if test_mode:
        log.info("\n=== TEST MODE — 2 dialogues, no writes ===")
        test_ids = dialogue_ids[:DIALOGUES_PER_CALL]
        semaphore = asyncio.Semaphore(1)
        all_items = []
        for did in test_ids:
            turns = _turns_needing_emotion(did, processed_store[did])
            for t in turns:
                t["dialogue_id"] = did
            all_items.extend(turns[:3])  # only first 3 turns per dialogue in test

        log.info(f"Testing on: {test_ids} — {len(all_items)} turns")
        results = await _gemini_emotion_call(all_items, semaphore)
        if results:
            for item, result in zip(all_items, results):
                log.info(f"\n  {item['turn_id']} ({item['speaker']}): {result}")
            log.info("\nTest OK.")
        else:
            log.error("Test FAILED — no results.")
        return

    # Full run — batch into groups of DIALOGUES_PER_CALL
    batches = [
        dialogue_ids[i : i + DIALOGUES_PER_CALL]
        for i in range(0, len(dialogue_ids), DIALOGUES_PER_CALL)
    ]
    log.info(f"Total batches: {len(batches)}")

    semaphore  = asyncio.Semaphore(CONCURRENCY)
    total_written = 0

    async def run_batch(batch_idx: int, batch: list[str]) -> None:
        nonlocal total_written
        n = await process_dialogue_batch(batch, processed_store, semaphore)
        total_written += n

    for i in range(0, len(batches), CONCURRENCY):
        chunk = batches[i : i + CONCURRENCY]
        await asyncio.gather(*[run_batch(i + j, b) for j, b in enumerate(chunk)])
        pct = min(i + CONCURRENCY, len(batches)) / len(batches) * 100
        log.info(
            f"Progress: {min(i+CONCURRENCY,len(batches))}/{len(batches)} batches "
            f"({pct:.1f}%) — {total_written} emotion fields written"
        )

    log.info(f"=== Done. {total_written} emotion fields written. Log: {LOG_FILE} ===")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=None,
                   help="Number of dialogues to sample (default: all)")
    p.add_argument("--test",   action="store_true",
                   help="Test mode: 1 batch, no writes")
    args = p.parse_args()
    asyncio.run(main(sample=args.sample, test_mode=args.test))
