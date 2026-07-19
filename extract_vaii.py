#!/usr/bin/env python3
"""
VAII (Vocal Arousal & Instability Index) Feature Extraction
============================================================
Computes per-turn and dialogue-level VAII scores from TTS WAV files using:
  - praat-parselmouth: F0 (pitch), Jitter, Shimmer, HNR
  - librosa:           Speech rate via onset detection
  - soundfile+numpy:   RMS intensity

Results are written back into preprocessed dialogue JSONs under the key:
  `calculated_vaii_from_speech`

VAII Formula (per turn):
  VAII = 0.25*F0_cv + 0.20*Jitter_norm + 0.15*Shimmer_norm +
         0.15*HNR_inv + 0.15*Rate_dev + 0.10*RMS_norm
  → clipped to [0.0, 1.0]

Dialogue-level VAII: linear recency-weighted mean of per-turn scores
(later turns weigh more — arousal typically peaks at decision points)

Usage:
  # Single dialogue (dry-run, verbose):
  python extract_vaii.py --mode single \\
      --json preprocessed/template_01/range_1-10/dialogue_003.json \\
      --audio_root tts_outputs/ --verbose

  # Full batch across all JSONs:
  python extract_vaii.py --mode batch \\
      --preprocessed_root preprocessed/ \\
      --audio_root tts_outputs/ \\
      --workers 8
"""

import argparse
import json
import os
import re
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import librosa
import parselmouth
from parselmouth.praat import call
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─── VAII Formula Constants ──────────────────────────────────────────────────

WEIGHTS = {
    "f0_cv":          0.25,   # Pitch coefficient of variation (instability marker)
    "jitter_norm":    0.20,   # Cycle-to-cycle pitch perturbation
    "shimmer_norm":   0.15,   # Amplitude perturbation
    "hnr_inv":        0.15,   # Inverted HNR (more noise → more arousal)
    "rate_dev":       0.15,   # Deviation from neutral speech rate
    "rms_norm":       0.10,   # Normalised loudness
}

# Normalization ceilings — empirically calibrated for synthetic TTS audio
# (real speech would have higher ceilings; TTS is less extreme)
JITTER_CEIL       = 0.04    # 4% local jitter = high instability in TTS context
SHIMMER_CEIL      = 1.50    # 1.5 dB local shimmer ceiling
HNR_NEUTRAL_DB    = 28.0    # Above this → essentially clean voice
NEUTRAL_RATE      = 3.5     # Syllables per second — neutral conversational rate
RATE_MAX_DEV      = 2.5     # Maximum expected rate deviation (syl/sec)

# Minimum duration to attempt Praat analysis (very short turns fail)
MIN_DURATION_SEC  = 0.25
PRAAT_PITCH_MIN   = 75.0    # Hz  — covers both male and female voices
PRAAT_PITCH_MAX   = 600.0   # Hz


# ─── Core Feature Extractor ──────────────────────────────────────────────────

def extract_features_from_wav(wav_path: str) -> dict:
    """
    Extract all six VAII acoustic features from a mono WAV file.

    Returns a dict with raw feature values + a status key.
    status = "ok" | "too_short" | "error"
    """
    result = {
        "f0_mean_hz":          None,
        "f0_std_hz":           None,
        "f0_cv":               None,
        "jitter_local":        None,
        "shimmer_local_db":    None,
        "hnr_db":              None,
        "hnr_inverted":        None,
        "speech_rate_syl_per_sec": None,
        "rate_deviation":      None,
        "rms_energy":          None,
        "rms_normalized":      None,
        "status":              "ok",
    }

    try:
        # ── Load audio ──────────────────────────────────────────────────────
        audio, sr = sf.read(wav_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # stereo → mono (shouldn't happen)

        duration = len(audio) / sr
        if duration < MIN_DURATION_SEC:
            result["status"] = "too_short"
            return result

        # ── Parselmouth (Praat) analysis ────────────────────────────────────
        snd = parselmouth.Sound(wav_path)

        # — F0 (pitch) ———————————————————————————————————————————————————————
        pitch = snd.to_pitch_ac(
            time_step=0.01,
            pitch_floor=PRAAT_PITCH_MIN,
            pitch_ceiling=PRAAT_PITCH_MAX,
        )
        f0_values = pitch.selected_array["frequency"]
        f0_voiced = f0_values[f0_values > 0]

        if len(f0_voiced) < 5:
            # Too few voiced frames — very short or whispered audio
            result["status"] = "too_short"
            return result

        f0_mean = float(np.mean(f0_voiced))
        f0_std  = float(np.std(f0_voiced))
        f0_cv   = float(f0_std / f0_mean) if f0_mean > 0 else 0.0

        result["f0_mean_hz"] = round(f0_mean, 2)
        result["f0_std_hz"]  = round(f0_std, 2)
        result["f0_cv"]      = round(f0_cv, 5)

        # — Jitter ————————————————————————————————————————————————————————————
        point_process = call(snd, "To PointProcess (periodic, cc)", PRAAT_PITCH_MIN, PRAAT_PITCH_MAX)
        jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        jitter_local = float(jitter_local) if jitter_local is not None else 0.0
        result["jitter_local"] = round(jitter_local, 6)

        # — Shimmer ———————————————————————————————————————————————————————————
        shimmer_db = call(
            [snd, point_process],
            "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6
        )
        shimmer_db = float(shimmer_db) if shimmer_db is not None else 0.0
        result["shimmer_local_db"] = round(shimmer_db, 5)

        # — HNR ———————————————————————————————————————————————————————————————
        harmonicity = call(snd, "To Harmonicity (cc)", 0.01, PRAAT_PITCH_MIN, 0.1, 1.0)
        hnr_db = call(harmonicity, "Get mean", 0, 0)
        hnr_db = float(hnr_db) if (hnr_db is not None and not np.isnan(hnr_db)) else 0.0
        hnr_inv = float(1.0 - np.clip(hnr_db / HNR_NEUTRAL_DB, 0.0, 1.0))
        result["hnr_db"]      = round(hnr_db, 3)
        result["hnr_inverted"] = round(hnr_inv, 5)

        # ── Librosa — Speech Rate (onset density as syllable proxy) ─────────
        y_lib = audio.copy()
        if sr != 22050:
            y_lib = librosa.resample(y_lib, orig_sr=sr, target_sr=22050)
            sr_lib = 22050
        else:
            sr_lib = sr

        onset_frames = librosa.onset.onset_detect(
            y=y_lib, sr=sr_lib,
            hop_length=256,
            backtrack=True,
        )
        n_onsets = max(len(onset_frames), 1)
        speech_rate = float(n_onsets / duration)     # onsets/sec ≈ syllables/sec
        rate_dev = float(abs(speech_rate - NEUTRAL_RATE) / RATE_MAX_DEV)
        rate_dev = float(np.clip(rate_dev, 0.0, 1.0))

        result["speech_rate_syl_per_sec"] = round(speech_rate, 3)
        result["rate_deviation"]          = round(rate_dev, 5)

        # ── RMS Intensity ────────────────────────────────────────────────────
        rms_full = float(np.sqrt(np.mean(audio ** 2)))
        # Normalize by 95th-percentile RMS across short frames for robustness
        frame_size = int(sr * 0.025)
        hop_size   = int(sr * 0.010)
        frames = librosa.util.frame(audio, frame_length=frame_size, hop_length=hop_size)
        frame_rms = np.sqrt(np.mean(frames ** 2, axis=0))
        p95 = float(np.percentile(frame_rms, 95)) if len(frame_rms) > 0 else 1.0
        rms_norm = float(np.clip(rms_full / max(p95, 1e-8), 0.0, 1.0))

        result["rms_energy"]     = round(rms_full, 6)
        result["rms_normalized"] = round(rms_norm, 5)

    except Exception as e:
        result["status"] = f"error: {str(e)}"

    return result


def compute_turn_vaii(features: dict) -> float:
    """
    Compute single-turn VAII score [0.0, 1.0] from raw features dict.
    Returns 0.0 if features are incomplete (short/error turns).
    """
    if features.get("status") != "ok":
        return 0.0

    # Normalize each component to [0, 1]
    f0_cv_n      = float(np.clip(features["f0_cv"], 0.0, 1.0))
    jitter_n     = float(np.clip(features["jitter_local"] / JITTER_CEIL, 0.0, 1.0))
    shimmer_n    = float(np.clip(features["shimmer_local_db"] / SHIMMER_CEIL, 0.0, 1.0))
    hnr_inv_n    = float(np.clip(features["hnr_inverted"], 0.0, 1.0))
    rate_dev_n   = float(np.clip(features["rate_deviation"], 0.0, 1.0))
    rms_n        = float(np.clip(features["rms_normalized"], 0.0, 1.0))

    vaii = (
        WEIGHTS["f0_cv"]       * f0_cv_n    +
        WEIGHTS["jitter_norm"] * jitter_n   +
        WEIGHTS["shimmer_norm"]* shimmer_n  +
        WEIGHTS["hnr_inv"]     * hnr_inv_n  +
        WEIGHTS["rate_dev"]    * rate_dev_n +
        WEIGHTS["rms_norm"]    * rms_n
    )
    return float(round(np.clip(vaii, 0.0, 1.0), 5))


def aggregate_dialogue_vaii(turn_scores: list[float]) -> float:
    """
    Aggregate per-turn VAII scores into a single dialogue-level score.
    Uses linear recency weighting: later turns have higher weight.
    """
    n = len(turn_scores)
    if n == 0:
        return 0.0
    weights = np.linspace(1.0, float(n), n)   # [1, 2, 3, ..., n]
    weights = weights / weights.sum()
    return float(round(float(np.dot(weights, turn_scores)), 5))


# ─── WAV Path Resolver ───────────────────────────────────────────────────────

def resolve_wav_path(audio_root: Path, json_path: Path, turn_idx: int, speaker: str) -> Optional[Path]:
    """
    Derive the WAV file path for a given dialogue turn.

    Convention:
      tts_outputs/<template_XX>/<range_Y-Z>/<dialogue_NNN>/turn_<TT>_<speaker>.wav
    Mirroring:
      preprocessed/<template_XX>/<range_Y-Z>/<dialogue_NNN>.json
    """
    # json_path: .../preprocessed/template_01/range_1-10/dialogue_003.json
    parts        = json_path.parts
    try:
        pre_idx      = next(i for i, p in enumerate(parts) if p == "preprocessed")
    except StopIteration:
        return None

    template_dir = parts[pre_idx + 1]   # template_01
    range_dir    = parts[pre_idx + 2]   # range_1-10
    dialogue_dir = json_path.stem       # dialogue_003

    speaker_tag = speaker.lower()       # "buyer" or "seller"
    turn_str    = f"turn_{turn_idx:02d}_{speaker_tag}.wav"

    wav = audio_root / template_dir / range_dir / dialogue_dir / turn_str
    return wav


# ─── Per-Dialogue Processor ──────────────────────────────────────────────────

EXTRACTION_MODEL = f"praat-parselmouth=={parselmouth.__version__} + librosa=={librosa.__version__}"

def process_dialogue_json(json_path: Path, audio_root: Path, verbose: bool = False) -> dict:
    """
    Load a dialogue JSON, extract VAII for every turn, write-back, return summary.
    Returns dict with: path, turns_processed, turns_failed, dialogue_vaii_score, status
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    trajectory     = data.get("trajectory", [])
    turns_ok       = 0
    turns_failed   = 0
    per_turn_scores = []

    for entry in trajectory:
        turn_num = entry.get("turn", 0)
        speaker  = entry.get("speaker", "unknown")

        wav_path = resolve_wav_path(audio_root, json_path, turn_num, speaker)

        if wav_path is None or not wav_path.exists():
            # Try alternate filenames in case of mismatches
            alt_found = False
            if wav_path is not None:
                parent = wav_path.parent
                if parent.exists():
                    candidates = list(parent.glob(f"turn_{turn_num:02d}_*.wav"))
                    if candidates:
                        wav_path  = candidates[0]
                        alt_found = True

            if not alt_found:
                vaii_payload = {
                    "turn_vaii_score": 0.0,
                    "features": {},
                    "audio_file": str(wav_path) if wav_path else "NOT_FOUND",
                    "extraction_model": EXTRACTION_MODEL,
                    "status": "audio_file_not_found",
                }
                entry["calculated_vaii_from_speech"] = vaii_payload
                per_turn_scores.append(0.0)
                turns_failed += 1
                continue

        features         = extract_features_from_wav(str(wav_path))
        turn_vaii        = compute_turn_vaii(features)
        per_turn_scores.append(turn_vaii)

        if features["status"] == "ok":
            turns_ok += 1
        else:
            turns_failed += 1

        vaii_payload = {
            "turn_vaii_score":  turn_vaii,
            "features": {
                "f0_mean_hz":              features["f0_mean_hz"],
                "f0_std_hz":               features["f0_std_hz"],
                "f0_cv":                   features["f0_cv"],
                "jitter_local":            features["jitter_local"],
                "shimmer_local_db":        features["shimmer_local_db"],
                "hnr_db":                  features["hnr_db"],
                "hnr_inverted":            features["hnr_inverted"],
                "speech_rate_syl_per_sec": features["speech_rate_syl_per_sec"],
                "rate_deviation":          features["rate_deviation"],
                "rms_energy":              features["rms_energy"],
                "rms_normalized":          features["rms_normalized"],
            },
            "audio_file":       str(wav_path.relative_to(audio_root.parent)),
            "extraction_model": EXTRACTION_MODEL,
            "status":           features["status"],
        }
        entry["calculated_vaii_from_speech"] = vaii_payload

        if verbose:
            print(
                f"  Turn {turn_num:02d} [{speaker:6s}] | "
                f"VAII={turn_vaii:.4f} | "
                f"F0_cv={features['f0_cv']:.4f} | "
                f"Jitter={features['jitter_local']:.5f} | "
                f"Shimmer={features['shimmer_local_db']:.3f}dB | "
                f"HNR={features['hnr_db']:.1f}dB | "
                f"Rate={features['speech_rate_syl_per_sec']:.2f}syl/s | "
                f"Status={features['status']}"
            )

    # ── Dialogue-level aggregate ─────────────────────────────────────────────
    dialogue_vaii  = aggregate_dialogue_vaii(per_turn_scores)
    classification = "severe" if dialogue_vaii >= 0.70 else "stable"

    data["metadata"]["calculated_vaii_from_speech"] = {
        "dialogue_vaii_score": dialogue_vaii,
        "aggregation":         "linear_recency_weighted_mean",
        "turns_processed":     turns_ok + turns_failed,
        "turns_ok":            turns_ok,
        "turns_failed":        turns_failed,
        "vaii_classification": classification,
        "per_turn_scores":     per_turn_scores,
        "extraction_model":    EXTRACTION_MODEL,
    }
    data["metadata"]["vaii_extracted"] = True

    # Write back in-place
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    return {
        "path":               str(json_path),
        "turns_processed":    turns_ok + turns_failed,
        "turns_failed":       turns_failed,
        "dialogue_vaii_score": dialogue_vaii,
        "vaii_classification": classification,
        "status":             "ok",
    }


# ─── Worker wrapper for multiprocessing ─────────────────────────────────────

def _worker(args):
    json_path, audio_root = args
    try:
        return process_dialogue_json(Path(json_path), Path(audio_root), verbose=False)
    except Exception as e:
        return {
            "path":    str(json_path),
            "status":  f"error: {str(e)}",
            "traceback": traceback.format_exc(),
        }


# ─── CLI Entry Point ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="VAII Audio Feature Extraction — compute vocal arousal scores from TTS audio"
    )
    parser.add_argument("--mode", choices=["single", "batch"], required=True)
    parser.add_argument("--json",              help="Path to single dialogue JSON [--mode single]")
    parser.add_argument("--audio_root",        required=True,
                        help="Root of tts_outputs/ directory")
    parser.add_argument("--preprocessed_root", help="Root of preprocessed/ directory [--mode batch]")
    parser.add_argument("--workers",  type=int, default=4,
                        help="Number of parallel workers for batch mode")
    parser.add_argument("--verbose",  action="store_true",
                        help="Print per-turn details [--mode single only]")
    parser.add_argument("--skip_done", action="store_true",
                        help="Skip JSONs that already have vaii_extracted=True")
    args = parser.parse_args()

    audio_root = Path(args.audio_root).resolve()

    # ── SINGLE MODE ──────────────────────────────────────────────────────────
    if args.mode == "single":
        if not args.json:
            parser.error("--json is required for --mode single")
        json_path = Path(args.json).resolve()
        print(f"\n{'='*70}")
        print(f"  VAII Extraction — Single Dialogue")
        print(f"  JSON : {json_path}")
        print(f"  Audio: {audio_root}")
        print(f"{'='*70}\n")

        result = process_dialogue_json(json_path, audio_root, verbose=args.verbose)

        print(f"\n{'─'*70}")
        print(f"  Dialogue VAII Score : {result['dialogue_vaii_score']:.5f}")
        print(f"  Classification      : {result['vaii_classification'].upper()}")
        print(f"  Turns processed     : {result['turns_processed']}")
        print(f"  Turns failed        : {result['turns_failed']}")
        print(f"  Written to          : {result['path']}")
        print(f"{'─'*70}\n")

    # ── BATCH MODE ───────────────────────────────────────────────────────────
    elif args.mode == "batch":
        if not args.preprocessed_root:
            parser.error("--preprocessed_root is required for --mode batch")
        preprocessed_root = Path(args.preprocessed_root).resolve()

        all_jsons = sorted(preprocessed_root.rglob("*.json"))
        print(f"\n  Found {len(all_jsons)} JSON files under {preprocessed_root}")

        if args.skip_done:
            pending = []
            for jp in all_jsons:
                try:
                    with open(jp) as f:
                        d = json.load(f)
                    if not d.get("metadata", {}).get("vaii_extracted", False):
                        pending.append(jp)
                except Exception:
                    pending.append(jp)
            print(f"  Skipping {len(all_jsons) - len(pending)} already-extracted. Processing {len(pending)}.")
            all_jsons = pending
        else:
            print(f"  Processing all {len(all_jsons)} files.")

        if not all_jsons:
            print("  Nothing to process. Done.")
            return

        work_args = [(str(jp), str(audio_root)) for jp in all_jsons]

        results     = []
        errors      = []
        severe_count = 0

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_worker, a): a[0] for a in work_args}
            with tqdm(total=len(futures), desc="Extracting VAII", unit="dlg") as pbar:
                for future in as_completed(futures):
                    res = future.result()
                    results.append(res)
                    if "error" in str(res.get("status", "")):
                        errors.append(res)
                    elif res.get("vaii_classification") == "severe":
                        severe_count += 1
                    pbar.update(1)

        # ── Summary ──────────────────────────────────────────────────────────
        ok_results  = [r for r in results if "error" not in str(r.get("status", ""))]
        vaii_scores = [r["dialogue_vaii_score"] for r in ok_results if "dialogue_vaii_score" in r]

        print(f"\n{'='*70}")
        print(f"  VAII BATCH EXTRACTION COMPLETE")
        print(f"{'='*70}")
        print(f"  Total processed   : {len(results)}")
        print(f"  Successful        : {len(ok_results)}")
        print(f"  Errors            : {len(errors)}")
        if vaii_scores:
            print(f"  VAII mean         : {np.mean(vaii_scores):.4f}")
            print(f"  VAII std          : {np.std(vaii_scores):.4f}")
            print(f"  VAII min/max      : {min(vaii_scores):.4f} / {max(vaii_scores):.4f}")
            print(f"  Severe (≥0.70)    : {severe_count} ({100*severe_count/len(ok_results):.1f}%)")
            print(f"  Stable (<0.70)    : {len(ok_results)-severe_count} ({100*(len(ok_results)-severe_count)/len(ok_results):.1f}%)")
        print(f"{'='*70}\n")

        if errors:
            print(f"  [ERRORS — first 5]:")
            for e in errors[:5]:
                print(f"    {e['path']}: {e['status']}")

        # Save batch summary log
        log_path = preprocessed_root / "vaii_extraction_log.json"
        with open(log_path, "w") as f:
            json.dump({
                "total": len(results),
                "successful": len(ok_results),
                "errors": len(errors),
                "vaii_mean": float(np.mean(vaii_scores)) if vaii_scores else None,
                "vaii_std":  float(np.std(vaii_scores))  if vaii_scores else None,
                "severe_count": severe_count,
                "error_details": errors[:20],
            }, f, indent=2)
        print(f"  Log saved → {log_path}\n")


if __name__ == "__main__":
    main()
