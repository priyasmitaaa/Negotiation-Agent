"""
sft_data_v4.py — Dataset, split builder, class weights for v4/v5 SFT.

Key differences from sft_data_v2.py:
  - Reads from dataset/ (not intermediate/) — this is where emotions live
  - Uses buyer emotion (labels, intensity, valence) instead of VAII
  - Filters to dialogues where ALL turns have emotion AND all seller turns have reasoning
  - Splits saved to splits_v4.json — shared by both v4 and v5 for fair comparison
  - use_reasoning=True/False controls whether <reasoning> appears in assistant target
  - No <seller_vaii> tag (removed from dataset)

Imported by train_sft_v4.py and train_sft_v5.py.
"""

import json
import random
import logging
import soundfile as sf
import librosa
import numpy as np
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
V2_ROOT      = Path(__file__).parent           # Qwen3-tts/v2/
DATASET_DIR  = V2_ROOT / "dataset"             # has emotions + reasoning
SPLITS_FILE  = V2_ROOT / "splits_v4.json"
CW_FILE      = V2_ROOT / "class_weights_v4.json"
MISSING_LOG  = V2_ROOT / "sft_missing_audio_v4.log"

TARGET_SR    = 16_000   # Qwen2.5-Omni audio encoder expects 16 kHz


# ── Split builder ──────────────────────────────────────────────────────────────

def build_splits_v4(seed: int = 42) -> dict:
    """
    Build stratified 80/10/10 train/val/test split from dialogues that have:
      - ALL turns with emotion field
      - ALL seller turns with reasoning field

    Stratified by correct_decision (LEVERAGE / MITIGATE).
    Both v4 and v5 use the same splits_v4.json for a fair comparison.
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

        # Must have emotion on every turn
        if not all("emotion" in t for t in turns):
            skipped += 1
            continue

        # Must have reasoning on every seller turn
        seller_turns = [t for t in turns if t["speaker"] == "seller"]
        if not seller_turns or not all(t.get("reasoning") for t in seller_turns):
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


def compute_class_weights_v4(splits: dict) -> dict:
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
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        return audio.astype(np.float32)
    except Exception:
        return None


# ── Dataset ────────────────────────────────────────────────────────────────────

class NegotiationV4Dataset:
    """
    One training example per seller turn.

    Each example:
      messages[0]  system  — system prompt (v4 or v5 variant, passed in)
      messages[1]  user    — context block + conversation up to this seller turn
                             buyer turns include emotion labels/intensity/valence
                             seller turns show text only
      messages[2]  asst    — structured target:
                             v4 (use_reasoning=False):
                               <decision>LEVERAGE</decision>
                               <response>...text...</response>
                             v5 (use_reasoning=True):
                               <reasoning>...</reasoning>
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
        use_reasoning: bool,
        split_name: str = "train",
    ):
        self.audio_dir     = audio_dir
        self.split_name    = split_name
        self.use_reasoning = use_reasoning
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

                tidx      = turn["turn_index"]
                reasoning = (turn.get("reasoning") or "").strip()
                dec       = turn["factor_state"]["decision"]

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

                task_outputs = "<decision>, <response>"
                if use_reasoning:
                    task_outputs = "<reasoning>, <decision>, <response>"

                user_text = (
                    f"CONTEXT\n{context_block}\n\n"
                    f"CONVERSATION SO FAR\n" + "\n".join(conv_lines) + "\n\n"
                    f"TASK\nYou are the seller. It is now Turn {tidx}. "
                    f"Produce {task_outputs}."
                )

                # Assistant target
                if use_reasoning:
                    asst_text = (
                        f"<reasoning>\n{reasoning}\n</reasoning>\n"
                        f"<decision>{dec}</decision>\n"
                        f"<response>\n{turn['text']}\n</response>"
                    )
                else:
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
        for path in ex["audio_paths"]:
            arr = load_wav(Path(path))
            if arr is not None:
                audio_arrays.append(arr)
        return {
            "messages":     ex["messages"],
            "audio_arrays": audio_arrays,
            "decision_int": ex["decision_int"],
        }