"""
sft_data_v6.py — Dataset, split builder, class weights for v6 SFT.

Key differences from sft_data_v4.py:
  - Uses ALL 3154 dialogues (not just 881 with reasoning annotated)
  - Eligibility: only requires emotion on buyer turns (no reasoning filter)
  - Splits saved to splits_v6.json (independent of v4/v5)
  - use_reasoning=False always (v6 has no reasoning target)
  - Target format: <decision> + <response> only (same as v4)

Imported by train_sft_v6.py and inference_v6.py.
"""

import json
import random
import logging
import soundfile as sf
import numpy as np
from math import gcd
from pathlib import Path
from scipy.signal import resample_poly
from typing import Optional

log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
V2_ROOT      = Path(__file__).parent           # Qwen3-tts/v2/
DATASET_DIR  = V2_ROOT / "dataset"             # has emotions + reasoning
SPLITS_FILE  = V2_ROOT / "splits_v6.json"
CW_FILE      = V2_ROOT / "class_weights_v6.json"
MISSING_LOG  = V2_ROOT / "sft_missing_audio_v6.log"

TARGET_SR    = 16_000   # Qwen2.5-Omni audio encoder expects 16 kHz


# ── Split builder ──────────────────────────────────────────────────────────────

def build_splits_v6(seed: int = 42) -> dict:
    """
    Build stratified 80/10/10 train/val/test split from ALL dialogues that have:
      - ALL buyer turns with emotion field

    Unlike v4/v5, NO reasoning filter is applied — all 3154 dialogues are eligible.
    Stratified by correct_decision (LEVERAGE / MITIGATE).
    Saves to splits_v6.json (independent of v4/v5 splits).
    """
    if SPLITS_FILE.exists():
        log.info(f"Loading existing splits from {SPLITS_FILE}")
        return json.loads(SPLITS_FILE.read_text())

    files = sorted(DATASET_DIR.glob("dialogue_*.json"))
    if not files:
        raise FileNotFoundError(f"No dataset files found in {DATASET_DIR}")

    leverage_ids, mitigate_ids = [], []
    skipped = 0
    for f in files:
        try:
            data = json.loads(f.read_text())
        except Exception:
            skipped += 1
            continue

        turns = data.get("processed", [])
        if not turns:
            skipped += 1
            continue

        buyer_turns = [t for t in turns if t["speaker"] == "buyer"]
        if not buyer_turns:
            skipped += 1
            continue

        # Only require emotion on buyer turns (no reasoning filter)
        if not all("emotion" in t for t in buyer_turns):
            skipped += 1
            continue

        dec = data["seed"]["generation_params"]["correct_decision"]
        if dec == "LEVERAGE":
            leverage_ids.append(f.stem)
        else:
            mitigate_ids.append(f.stem)

    log.info(
        f"Eligible dialogues: {len(leverage_ids)+len(mitigate_ids)} "
        f"(LEVERAGE={len(leverage_ids)} MITIGATE={len(mitigate_ids)}, "
        f"skipped={skipped})"
    )

    rng = random.Random(seed)

    def stratified_split(ids):
        ids = ids[:]
        rng.shuffle(ids)
        n     = len(ids)
        n_tr  = int(n * 0.80)
        n_val = int(n * 0.10)
        return ids[:n_tr], ids[n_tr:n_tr + n_val], ids[n_tr + n_val:]

    lev_tr, lev_val, lev_te = stratified_split(leverage_ids)
    mit_tr, mit_val, mit_te = stratified_split(mitigate_ids)

    splits = {
        "train": sorted(lev_tr + mit_tr),
        "val":   sorted(lev_val + mit_val),
        "test":  sorted(lev_te + mit_te),
        "counts": {
            "total":    len(leverage_ids) + len(mitigate_ids),
            "leverage": len(leverage_ids),
            "mitigate": len(mitigate_ids),
            "skipped":  skipped,
        },
    }
    SPLITS_FILE.write_text(json.dumps(splits, indent=2))
    log.info(
        f"Splits saved — train:{len(splits['train'])} "
        f"val:{len(splits['val'])} test:{len(splits['test'])}"
    )
    return splits


def compute_class_weights_v6(splits: dict) -> dict:
    """
    Inverse-frequency class weights from TRAINING split only.
    w_class = n_total_train / (2 * n_class_train)
    """
    if CW_FILE.exists():
        cw = json.loads(CW_FILE.read_text())
        log.info(f"Class weights loaded: LEVERAGE={cw['LEVERAGE']:.4f}  MITIGATE={cw['MITIGATE']:.4f}")
        return cw

    train_ids = splits["train"]
    lev = mit = 0
    for did in train_ids:
        f    = DATASET_DIR / f"{did}.json"
        data = json.loads(f.read_text())
        dec  = data["seed"]["generation_params"]["correct_decision"]
        if dec == "LEVERAGE":
            lev += 1
        else:
            mit += 1

    total = lev + mit
    cw = {
        "LEVERAGE": round(total / (2 * lev), 6) if lev > 0 else 1.0,
        "MITIGATE": round(total / (2 * mit), 6) if mit > 0 else 1.0,
    }
    CW_FILE.write_text(json.dumps(cw, indent=2))
    log.info(f"Class weights — LEVERAGE:{cw['LEVERAGE']:.4f}  MITIGATE:{cw['MITIGATE']:.4f}")
    return cw


# ── Audio loader ───────────────────────────────────────────────────────────────

def load_wav(path: Path) -> Optional[np.ndarray]:
    """Load WAV, convert to mono float32, resample to 16 kHz."""
    try:
        audio, sr = sf.read(str(path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != TARGET_SR:
            common = gcd(sr, TARGET_SR)
            audio = resample_poly(audio, TARGET_SR // common, sr // common)
        return audio.astype(np.float32)
    except Exception as exc:
        log.warning("Could not load audio %s: %s", path, exc)
        return None


# ── Dataset ────────────────────────────────────────────────────────────────────

class NegotiationV6Dataset:
    """
    One training example per seller turn. No reasoning in target.

    Each example:
      messages[0]  system  — system prompt
      messages[1]  user    — context block + conversation up to this seller turn
                             buyer turns include emotion labels/intensity/valence
                             seller turns show text only
      messages[2]  asst    — structured target (v6, no reasoning):
                               <decision>LEVERAGE</decision>
                               <response>...text...</response>

    Audio: WAVs for all turns PRECEDING this seller turn loaded from audio_dir.
    """

    DECISION_MAP = {"LEVERAGE": 0, "MITIGATE": 1, "UNDECIDED": 2}

    def __init__(
        self,
        dialogue_ids: list,
        audio_dir: Path,
        system_prompt: str,
        split_name: str = "train",
        text_only_ratio: float = 0.15,
    ):
        self.audio_dir     = audio_dir
        self.split_name    = split_name
        self.text_only_ratio = text_only_ratio if split_name == "train" else 0.0
        self.examples: list = []
        self._missing_log  = open(MISSING_LOG, "a")

        self.skipped_no_audio = 0

        for did in dialogue_ids:
            fpath = DATASET_DIR / f"{did}.json"
            if not fpath.exists():
                log.warning(f"{did}: dataset file missing — skip")
                continue

            data     = json.loads(fpath.read_text())
            seed     = data["seed"]
            turns    = data["processed"]
            decision = seed["generation_params"]["correct_decision"]
            pricing  = seed["pricing"]
            domain   = seed["domain"]

            turns_so_far: list = []
            for turn in turns:
                if turn["speaker"] != "seller":
                    turns_so_far.append(turn)
                    continue

                tidx = turn["turn_index"]
                dec  = turn["factor_state"]["decision"]

                # Check audio for all preceding turns
                audio_paths = []
                missing = False
                for pt in turns_so_far:
                    ti  = pt["turn_index"]
                    spk = pt["speaker"]
                    wav = audio_dir / did / f"turn_{ti:02d}_{spk}.wav"
                    if not wav.exists():
                        self._missing_log.write(f"{did} T{ti} {spk}: {wav}\n")
                        self._missing_log.flush()
                        missing = True
                        break
                    audio_paths.append(wav)

                if missing:
                    self.skipped_no_audio += 1
                    turns_so_far.append(turn)
                    continue

                # User message: context + conversation with buyer emotions
                context_block = (
                    f"Product: {domain['product']}. "
                    f"Condition: {domain['condition']}. "
                    f"Asking: ₹{pricing['asking_price']}. "
                    f"Fair value: ₹{pricing['fair_value']}."
                )
                conv_lines = []
                for pt in turns_so_far:
                    line = f"Turn {pt['turn_index']} [{pt['speaker'].upper()}]: {pt['text']}"
                    if pt["speaker"] == "buyer":
                        em = pt.get("emotion", {})
                        if em.get("labels"):
                            labels_str = ", ".join(em["labels"])
                            line += (
                                f"  [Emotion: {labels_str} | "
                                f"intensity: {em.get('intensity','?')} | "
                                f"valence: {em.get('valence','?')}]"
                            )
                    conv_lines.append(line)

                user_text = (
                    f"CONTEXT\n{context_block}\n\n"
                    f"CONVERSATION SO FAR\n" + "\n".join(conv_lines) + "\n\n"
                    f"TASK\nYou are the seller. It is now Turn {tidx}. "
                    f"Produce <decision>, <response>."
                )

                # Assistant target: no reasoning
                asst_text = (
                    f"<decision>{dec}</decision>\n"
                    f"<response>\n{turn['text']}\n</response>"
                )

                self.examples.append({
                    "dialogue_id":  did,
                    "turn_index":   tidx,
                    "decision":     decision,
                    "decision_int": self.DECISION_MAP.get(decision, 0),
                    "audio_paths":  [str(p) for p in audio_paths],
                    "messages": [
                        {"role": "system",    "content": system_prompt},
                        {"role": "user",      "content": user_text},
                        {"role": "assistant", "content": asst_text},
                    ],
                })
                turns_so_far.append(turn)

        self._missing_log.close()
        log.info(
            f"[{split_name}] {len(self.examples)} examples built. "
            f"Skipped: {self.skipped_no_audio} (missing audio)."
        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        audio_arrays = []
        # Mix text-only and text+speech examples during training. This makes the
        # adapter useful in both modes and reduces reliance on one modality.
        use_audio = not (
            self.text_only_ratio > 0.0 and random.random() < self.text_only_ratio
        )
        if use_audio:
            for path in ex["audio_paths"]:
                arr = load_wav(Path(path))
                if arr is None:
                    raise RuntimeError(f"Audio validated at build time but failed to load: {path}")
                audio_arrays.append(arr)
        return {
            "messages":     ex["messages"],
            "audio_arrays": audio_arrays,
            "decision_int": ex["decision_int"],
        }
