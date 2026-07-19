#!/usr/bin/env python3
"""
generate_tts_v2.py
==================
Generate speech audio for all v2 dialogues using Qwen3-TTS.

Reads:  v2/intermediate/dialogue_XXXX.json
Writes: v2/tts_outputs/dialogue_XXXX/turn_NN_{buyer|seller}.wav
        v2/tts_outputs/generation_results.json  (per-dialogue result log)
        v2/tts_generation.log

Key behaviour:
  - Resume-safe: if all 12 WAVs for a dialogue exist and are non-empty → skip
  - Per-turn VAII/seller_vaii drives every voice instruction (no static labels)
  - Prior buyer VAII passed to seller instruction for inverse-relationship delivery
  - All numbers in text converted to words before synthesis
  - Voices assigned deterministically per dialogue_id (same pair across restarts)

Usage:
  # Test first 3 dialogues only
  CUDA_VISIBLE_DEVICES=1 python3 v2/generate_tts_v2.py --test

  # Full run (all 3154 dialogues, resumes automatically)
  CUDA_VISIBLE_DEVICES=1 python3 v2/generate_tts_v2.py --all

  # Single dialogue
  CUDA_VISIBLE_DEVICES=1 python3 v2/generate_tts_v2.py --dialogue dialogue_0042
"""

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import torch

# ── Path setup: run from Qwen3-tts root or from v2/ ──────────────────────────
_HERE   = Path(__file__).parent           # v2/
_ROOT   = _HERE.parent                    # Qwen3-tts/

# Add both to sys.path so imports from v2/ and root both work
for p in [str(_HERE), str(_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Project imports ───────────────────────────────────────────────────────────
from qwen3_tts_engine import Qwen3TTSEngine
from tts_text_preprocessor import number_to_words
from voice_instruction_generator_v2 import (
    PRESET_SPEAKERS,
    get_buyer_instruction,
    get_seller_instruction,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
INTERM_DIR   = _HERE / "intermediate"
OUTPUT_DIR   = _HERE / "tts_outputs"
RESULTS_FILE = OUTPUT_DIR / "generation_results.json"
LOG_PATH     = _HERE / "tts_generation.log"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(str(LOG_PATH)),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── Result tracker ────────────────────────────────────────────────────────────

def load_results() -> dict:
    if RESULTS_FILE.exists():
        try:
            raw = RESULTS_FILE.read_text().strip()
            if not raw:
                return {}
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
            log.warning(f"{RESULTS_FILE} is not a JSON object; resetting results.")
            return {}
        except json.JSONDecodeError:
            log.warning(f"{RESULTS_FILE} is invalid JSON; resetting results.")
            return {}
    return {}

def save_results(results: dict):
    RESULTS_FILE.write_text(json.dumps(results, indent=2))


# ── Resume check ─────────────────────────────────────────────────────────────

def is_complete(dialogue_id: str) -> bool:
    """True if all 12 WAVs exist and are non-empty for this dialogue."""
    d = OUTPUT_DIR / dialogue_id
    if not d.exists():
        return False
    wavs = list(d.glob("turn_*.wav"))
    if len(wavs) < 12:
        return False
    return all(w.stat().st_size > 0 for w in wavs)


# ── Per-dialogue generation ───────────────────────────────────────────────────

def process_dialogue(
    dialogue_id: str,
    engine: Qwen3TTSEngine,
    results: dict,
    run_voice_map: dict[str, dict[str, str]],
    run_rng: random.Random,
) -> dict:
    """
    Generate all 12 turn WAVs for one dialogue.
    Returns a result dict: {status, turns_ok, turns_failed, errors}.
    """
    fpath = INTERM_DIR / f"{dialogue_id}.json"
    if not fpath.exists():
        log.warning(f"{dialogue_id}: intermediate file missing — skip")
        return {"status": "skipped", "reason": "no_intermediate"}

    data       = json.loads(fpath.read_text())
    seed       = data["seed"]
    turns      = data["processed"]
    gp         = seed["generation_params"]
    strategy   = gp["correct_decision"]       # LEVERAGE / MITIGATE
    kl         = gp.get("knowledge_level", "moderate")

    # Random per run, fixed within a dialogue for this run.
    if dialogue_id not in run_voice_map:
        buyer_voice, seller_voice = run_rng.sample(PRESET_SPEAKERS, 2)
        run_voice_map[dialogue_id] = {"buyer": buyer_voice, "seller": seller_voice}
    voices = run_voice_map[dialogue_id]

    out_dir         = OUTPUT_DIR / dialogue_id
    out_dir.mkdir(parents=True, exist_ok=True)

    prior_buyer_vaii: float | None = None
    turns_ok, turns_failed, errors = 0, 0, []

    log.info(f"  {dialogue_id} | strategy={strategy} | "
             f"buyer={voices['buyer']} seller={voices['seller']}")

    for turn in turns:
        tidx    = turn["turn_index"]
        speaker = turn["speaker"]
        text    = number_to_words(turn["text"])

        wav_path = out_dir / f"turn_{tidx:02d}_{speaker}.wav"

        # Resume: skip this turn if WAV already exists and is non-empty
        if wav_path.exists() and wav_path.stat().st_size > 0:
            log.info(f"    T{tidx:02d} [{speaker}] — already exists, skip")
            if speaker == "buyer":
                prior_buyer_vaii = turn.get("vaii", {}).get("raw")
            turns_ok += 1
            continue

        # Build voice instruction
        if speaker == "buyer":
            vaii_raw         = turn.get("vaii", {}).get("raw", 0.30)
            voice_instruction = get_buyer_instruction(
                vaii_raw=vaii_raw,
                turn_index=tidx,
                knowledge_level=kl,
                strategy=strategy,
            )
            assigned_voice    = voices["buyer"]
            prior_buyer_vaii  = vaii_raw   # store for the next seller turn

        else:  # seller
            sv_raw    = turn.get("seller_vaii", {}).get("raw", 0.25)
            decision  = turn.get("factor_state", {}).get("decision", strategy)
            voice_instruction = get_seller_instruction(
                seller_vaii_raw=sv_raw,
                turn_index=tidx,
                strategy=decision,
                prior_buyer_vaii=prior_buyer_vaii,
            )
            assigned_voice = voices["seller"]

        log.info(
            f"    T{tidx:02d} [{speaker.upper()}] "
            f"voice={assigned_voice} | {text[:60]}…"
        )

        success = engine.generate_speech(
            text=text,
            voice_instruction=voice_instruction,
            speaker=assigned_voice,
            output_path=wav_path,
            language="English",
        )

        if success:
            turns_ok += 1
        else:
            turns_failed += 1
            errors.append(f"T{tidx:02d}_{speaker}")
            log.error(f"    FAILED: {dialogue_id} T{tidx:02d} {speaker}")

    status = "complete" if turns_failed == 0 else ("partial" if turns_ok > 0 else "failed")
    result = {
        "status":        status,
        "turns_ok":      turns_ok,
        "turns_failed":  turns_failed,
        "errors":        errors,
    }
    results[dialogue_id] = result
    save_results(results)

    log.info(f"  {dialogue_id}: {status} ({turns_ok}/12 turns)")
    return result


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--all",       action="store_true",
                      help="Process all dialogues in v2/intermediate/ (resumes automatically)")
    mode.add_argument("--test",      action="store_true",
                      help="Process first 3 dialogues only (smoke test)")
    mode.add_argument("--dialogue",  type=str, metavar="DIALOGUE_ID",
                      help="Process a single dialogue by ID, e.g. --dialogue dialogue_0042")
    parser.add_argument(
        "--start-id",
        type=str,
        default=None,
        help="Optional inclusive start dialogue id, e.g. dialogue_0001",
    )
    parser.add_argument(
        "--end-id",
        type=str,
        default=None,
        help="Optional inclusive end dialogue id, e.g. dialogue_1577",
    )
    args = parser.parse_args()

    # ── Collect dialogue IDs to process ───────────────────────────────────────
    all_files = sorted(INTERM_DIR.glob("dialogue_*.json"))
    if not all_files:
        raise SystemExit(f"ERROR: No intermediate files found in {INTERM_DIR}")

    if args.dialogue:
        target = args.dialogue if args.dialogue.endswith(".json") else f"{args.dialogue}"
        target = target.replace(".json", "")
        files  = [INTERM_DIR / f"{target}.json"]
        if not files[0].exists():
            raise SystemExit(f"ERROR: {files[0]} not found")
    elif args.test:
        files = all_files[:3]
        log.info("TEST mode: processing first 3 dialogues")
    else:
        files = all_files

    dialogue_ids = [f.stem for f in files]

    def _to_num(dialogue_id: str) -> int:
        # dialogue_0123 -> 123
        try:
            return int(dialogue_id.split("_")[-1])
        except Exception:
            return -1

    if args.start_id or args.end_id:
        start_num = _to_num(args.start_id) if args.start_id else -10**9
        end_num = _to_num(args.end_id) if args.end_id else 10**9
        dialogue_ids = [d for d in dialogue_ids if start_num <= _to_num(d) <= end_num]
        if not dialogue_ids:
            raise SystemExit(
                f"ERROR: No dialogues in requested range start={args.start_id} end={args.end_id}"
            )
        log.info(
            f"Range filter applied: start={args.start_id or 'MIN'} "
            f"end={args.end_id or 'MAX'} -> {len(dialogue_ids)} dialogues"
        )

    # ── Resume: filter out already-complete dialogues ─────────────────────────
    results     = load_results()
    pending     = [d for d in dialogue_ids if not is_complete(d)]
    already_done = len(dialogue_ids) - len(pending)

    log.info(f"{'='*60}")
    log.info(f"  Qwen3-TTS v2 Generation")
    log.info(f"  Total      : {len(dialogue_ids):,} dialogues")
    log.info(f"  Already done: {already_done:,}")
    log.info(f"  Pending    : {len(pending):,}")
    log.info(f"  Output dir : {OUTPUT_DIR}")
    log.info(f"{'='*60}")

    if not pending:
        log.info("All dialogues already complete. Nothing to do.")
        return

    # ── Load TTS model ─────────────────────────────────────────────────────────
    log.info("Loading Qwen3-TTS model…")
    engine = Qwen3TTSEngine()
    engine.load_model()
    log.info("Model loaded. Starting generation.")

    # ── Generate ──────────────────────────────────────────────────────────────
    t0 = time.time()
    n_complete = n_partial = n_failed = 0
    run_rng = random.Random(time.time_ns())
    run_voice_map: dict[str, dict[str, str]] = {}

    for i, did in enumerate(pending, 1):
        log.info(f"\n[{i}/{len(pending)}] {did}")
        result = process_dialogue(did, engine, results, run_voice_map, run_rng)
        s = result.get("status", "?")
        if s == "complete":   n_complete += 1
        elif s == "partial":  n_partial  += 1
        elif s == "failed":   n_failed   += 1

        elapsed = time.time() - t0
        rate    = i / elapsed
        eta     = (len(pending) - i) / rate if rate > 0 else 0
        log.info(
            f"  Progress: {i}/{len(pending)} | "
            f"OK={n_complete} partial={n_partial} failed={n_failed} | "
            f"ETA: {eta/60:.1f} min"
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    total_time = time.time() - t0
    log.info(f"\n{'='*60}")
    log.info(f"  DONE in {total_time/60:.1f} min")
    log.info(f"  Complete : {n_complete + already_done:,}")
    log.info(f"  Partial  : {n_partial:,}  (some turns failed)")
    log.info(f"  Failed   : {n_failed:,}")
    log.info(f"  Results  : {RESULTS_FILE}")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
