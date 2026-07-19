"""
sft_data_v2.py — Dataset, split builder, class weights for v2 SFT.
Imported by train_sft_v2.py.
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
V2_ROOT      = Path(__file__).parent                          # Qwen3-tts/v2/
INTERM_DIR   = V2_ROOT / "intermediate"
SPLITS_FILE  = V2_ROOT / "splits_v2.json"
CW_FILE      = V2_ROOT / "class_weights_v2.json"
MISSING_LOG  = V2_ROOT / "sft_missing_audio.log"

TARGET_SR    = 16_000   # Qwen2.5-Omni audio encoder expects 16 kHz

# ── System prompt (same as restructure_dataset.py SYSTEM_PROMPT_SFT) ──────────
SYSTEM_PROMPT_SFT = (
    "You are an AI negotiation assistant for a second-hand electronics shop in India. "
    "Guide the negotiation to within 10% of fair market value. Analyse the "
    "conversation and buyer VAII signals (< 0.50 = calm, >= 0.50 = stressed). "
    "VAII (Vocal Affective Intensity Index) reflects the buyer's emotional stress level. "
    "Produce in this exact order: <reasoning>, <decision> [UNDECIDED/LEVERAGE/MITIGATE], "
    "<seller_vaii> [0.05-0.45], <response>. Infer everything from raw signals."
)

# ── Split builder ──────────────────────────────────────────────────────────────

def build_splits(seed: int = 42) -> dict:
    """
    Build stratified 80/10/10 train/val/test split at dialogue level.
    Stratified by correct_decision (LEVERAGE / MITIGATE).
    Saves splits_v2.json; returns dict with keys train/val/test → [dialogue_id, ...].
    Skips re-splitting if splits_v2.json already exists.
    """
    if SPLITS_FILE.exists():
        log.info(f"Loading existing splits from {SPLITS_FILE}")
        return json.loads(SPLITS_FILE.read_text())

    files = sorted(INTERM_DIR.glob("dialogue_*.json"))
    if not files:
        raise FileNotFoundError(f"No intermediate files found in {INTERM_DIR}")

    leverage_ids, mitigate_ids = [], []
    for f in files:
        meta = json.loads(f.read_text())
        dec  = meta["seed"]["generation_params"]["correct_decision"]
        did  = f.stem  # "dialogue_0001"
        if dec == "LEVERAGE":
            leverage_ids.append(did)
        else:
            mitigate_ids.append(did)

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
            "total":    len(files),
            "leverage": len(leverage_ids),
            "mitigate": len(mitigate_ids),
        },
    }
    SPLITS_FILE.write_text(json.dumps(splits, indent=2))
    log.info(
        f"Splits created — train:{len(splits['train'])} "
        f"val:{len(splits['val'])} test:{len(splits['test'])}"
    )
    return splits


def compute_class_weights(splits: dict) -> dict[str, float]:
    """
    Inverse-frequency class weights computed from TRAINING split only.
    w_class = n_total_train / (2 * n_class_train)
    Saves class_weights_v2.json.
    """
    if CW_FILE.exists():
        cw = json.loads(CW_FILE.read_text())
        log.info(f"Class weights loaded: LEVERAGE={cw['LEVERAGE']:.4f}  MITIGATE={cw['MITIGATE']:.4f}")
        return cw

    train_ids = splits["train"]
    lev = mit = 0
    for did in train_ids:
        f    = INTERM_DIR / f"{did}.json"
        meta = json.loads(f.read_text())
        dec  = meta["seed"]["generation_params"]["correct_decision"]
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
    except Exception as e:
        return None


# ── Dataset ────────────────────────────────────────────────────────────────────

class NegotiationV2Dataset:
    """
    One training example per seller turn (6 per dialogue, up to 18,924 total).

    Each example:
      messages[0]  system  — SYSTEM_PROMPT_SFT
      messages[1]  user    — context block + conversation up to (not including)
                             this seller turn; buyer turns show VAII, seller turns
                             show text only; NO factor_state, NO decision labels
      messages[2]  asst    — 4-part structured target:
                             <reasoning>...</reasoning>
                             <decision>LEVERAGE/MITIGATE/UNDECIDED</decision>
                             <seller_vaii>0.15</seller_vaii>
                             <response>...spoken seller text...</response>

    Audio: WAVs for all turns PRECEDING this seller turn are loaded from audio_dir.
           Naming: dialogue_XXXX/turn_NN_buyer.wav  OR  turn_NN_seller.wav
           If any WAV is missing the example is skipped (logged to sft_missing_audio.log).

    Examples with empty reasoning are silently skipped.
    """

    DECISION_MAP = {"LEVERAGE": 0, "MITIGATE": 1, "UNDECIDED": 2}

    def __init__(
        self,
        dialogue_ids: list[str],
        audio_dir: Path,
        split_name: str = "train",
    ):
        self.audio_dir  = audio_dir
        self.split_name = split_name
        self.examples: list[dict] = []
        self._missing_log = open(MISSING_LOG, "a")

        self.skipped_no_reasoning = 0
        self.skipped_no_audio     = 0
        skipped_no_reasoning = 0
        skipped_no_audio     = 0

        for did in dialogue_ids:
            fpath = INTERM_DIR / f"{did}.json"
            if not fpath.exists():
                log.warning(f"{did}: intermediate file missing — skip")
                continue
            data = json.loads(fpath.read_text())
            seed     = data["seed"]
            turns    = data["processed"]
            decision = seed["generation_params"]["correct_decision"]
            pricing  = seed["pricing"]
            domain   = seed["domain"]

            turns_so_far: list[dict] = []
            for turn in turns:
                if turn["speaker"] != "seller":
                    turns_so_far.append(turn)
                    continue

                tidx      = turn["turn_index"]
                reasoning = (turn.get("reasoning") or "").strip()
                if not reasoning:
                    skipped_no_reasoning += 1
                    self.skipped_no_reasoning += 1
                    turns_so_far.append(turn)
                    continue

                # Check audio files for all preceding turns
                audio_paths = []
                missing = False
                for pt in turns_so_far:
                    ti  = pt["turn_index"]
                    spk = pt["speaker"]
                    wav = audio_dir / did / f"turn_{ti:02d}_{spk}.wav"
                    if not wav.exists():
                        self._missing_log.write(
                            f"{did} T{ti} {spk}: {wav}\n"
                        )
                        self._missing_log.flush()
                        missing = True
                        break
                    audio_paths.append(wav)

                if missing:
                    skipped_no_audio += 1
                    self.skipped_no_audio += 1
                    turns_so_far.append(turn)
                    continue

                # Build user message: context block + conversation so far
                context_block = (
                    f"Product: {domain['product']}. "
                    f"Condition: {domain['condition']}. "
                    f"Asking: ${pricing['asking_price']}. "
                    f"Fair value: ${pricing['fair_value']}."
                )
                conv_lines = []
                for pt in turns_so_far:
                    line = f"Turn {pt['turn_index']} [{pt['speaker'].upper()}]: {pt['text']}"
                    if pt["speaker"] == "buyer":
                        v = pt.get("vaii", {})
                        if v.get("raw") is not None:
                            line += f"  [VAII: {v['raw']:.2f} / {v['state']}]"
                    conv_lines.append(line)

                user_text = (
                    f"CONTEXT\n{context_block}\n\n"
                    f"CONVERSATION SO FAR\n" + "\n".join(conv_lines) + "\n\n"
                    f"TASK\nYou are the seller. It is now Turn {tidx}. "
                    f"Produce <reasoning>, <decision>, <seller_vaii>, <response>."
                )

                # Build assistant target
                sv   = turn["seller_vaii"]["raw"]
                dec  = turn["factor_state"]["decision"]
                asst_text = (
                    f"<reasoning>\n{reasoning}\n</reasoning>\n"
                    f"<decision>{dec}</decision>\n"
                    f"<seller_vaii>{sv}</seller_vaii>\n"
                    f"<response>\n{turn['text']}\n</response>"
                )

                self.examples.append({
                    "dialogue_id":      did,
                    "turn_index":       tidx,
                    "decision":         decision,
                    "decision_int":     self.DECISION_MAP.get(decision, 0),
                    "audio_paths":      [str(p) for p in audio_paths],
                    "messages": [
                        {"role": "system",    "content": SYSTEM_PROMPT_SFT},
                        {"role": "user",      "content": user_text},
                        {"role": "assistant", "content": asst_text},
                    ],
                })
                turns_so_far.append(turn)

        log.info(
            f"[{split_name}] {len(self.examples)} examples built. "
            f"Skipped: {skipped_no_reasoning} (no reasoning), "
            f"{skipped_no_audio} (missing audio)."
        )
        self._missing_log.close()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        audio_arrays = []
        for path in ex["audio_paths"]:
            arr = load_wav(Path(path))
            if arr is None:
                log.warning(f"Could not load {path} at __getitem__ — skipping audio turn")
            else:
                audio_arrays.append(arr)
        return {
            "messages":     ex["messages"],
            "audio_arrays": audio_arrays,
            "decision_int": ex["decision_int"],
        }
