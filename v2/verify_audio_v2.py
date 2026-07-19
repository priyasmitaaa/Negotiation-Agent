#!/usr/bin/env python3
"""
verify_audio_v2.py
==================
Validate all generated TTS WAV files in v2/tts_outputs/ before SFT.

Checks per WAV file:
  1. File exists
  2. Size > 0 bytes (not a zero-byte placeholder)
  3. Readable by soundfile (not corrupt/truncated)
  4. Sample rate = 24000 Hz (Qwen3-TTS native output)
  5. Duration >= 0.3 s (not a flash of noise / empty render)
  6. Duration <= 90 s  (not a runaway generation)
  7. RMS > silence threshold (not a silent render that slipped through)
  8. Peak-to-RMS ratio check (not pure noise / staticky output)

Checks per dialogue:
  9. All 12 turns present (no turn gaps)
  10. Both buyer and seller turns present (no speaker missing)

Summary report:
  - Counts for every check category
  - Full list of failing files
  - Exit 0 if no CRITICAL issues, Exit 1 if any CRITICAL issues

Usage:
  # From Qwen3-tts root:
  python3 v2/verify_audio_v2.py

  # Custom paths / thresholds:
  python3 v2/verify_audio_v2.py --audio_root v2/tts_outputs \\
      --report v2/audio_verify_report.json \\
      --min_dur 0.3 --max_dur 90.0 \\
      --expected_sr 24000 --silence_rms 0.001
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

# ── Defaults ───────────────────────────────────────────────────────────────────
_HERE         = Path(__file__).parent
AUDIO_ROOT    = _HERE / "tts_outputs"
INTERM_DIR    = _HERE / "intermediate"
REPORT_PATH   = _HERE / "audio_verify_report.json"
EXPECTED_SR   = 24_000   # Qwen3-TTS native sample rate
MIN_DUR_SEC   = 0.3      # shorter → empty/noise render
MAX_DUR_SEC   = 90.0     # longer  → runaway generation
SILENCE_RMS   = 0.001    # below this → silent file (failed render)
NOISE_CREST   = 3.0      # crest factor (peak/RMS) below this → pure noise / DC offset

TURNS_PER_DIALOGUE = 12


# ── Single WAV inspector ───────────────────────────────────────────────────────

def inspect_wav(wav_path: Path, expected_sr: int, min_dur: float,
                max_dur: float, silence_rms: float, noise_crest: float) -> dict:
    """
    Open and fully inspect one WAV file.
    Returns a result dict with all check outcomes and a human-readable 'issue' string
    (None if all checks pass).
    """
    r = dict(
        exists=False, size_ok=False, readable=False,
        sr_ok=False, dur_ok=False, silent=False, noisy=False,
        sr=None, duration_sec=None, rms=None, issue=None,
    )

    if not wav_path.exists():
        r["issue"] = "missing"
        return r
    r["exists"] = True

    if wav_path.stat().st_size == 0:
        r["issue"] = "zero-byte file"
        return r
    r["size_ok"] = True

    try:
        info = sf.info(str(wav_path))
        r["sr"]           = info.samplerate
        r["duration_sec"] = round(info.duration, 3)
        r["readable"]     = True
    except Exception as e:
        r["issue"] = f"unreadable: {e}"
        return r

    if info.samplerate != expected_sr:
        r["issue"] = f"wrong sr={info.samplerate} (expected {expected_sr})"
        return r
    r["sr_ok"] = True

    if not (min_dur <= info.duration <= max_dur):
        r["dur_ok"] = False
        r["issue"]  = f"bad duration {info.duration:.2f}s (expected {min_dur}–{max_dur}s)"
        return r
    r["dur_ok"] = True

    # Load audio for content checks (mono float32)
    try:
        audio, _ = sf.read(str(wav_path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
    except Exception as e:
        r["issue"] = f"audio read failed: {e}"
        return r

    rms  = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.abs(audio).max()) if len(audio) > 0 else 0.0
    r["rms"] = round(rms, 6)

    # Check 7: silence
    if rms < silence_rms:
        r["silent"] = True
        r["issue"]  = f"silent (RMS={rms:.6f} < {silence_rms})"
        return r

    # Check 8: pure noise / DC (crest factor: peak/RMS should be > noise_crest for speech)
    crest = peak / rms if rms > 0 else 0.0
    if crest < noise_crest:
        r["noisy"] = True
        r["issue"] = f"suspected noise/DC (crest={crest:.2f} < {noise_crest})"
        return r

    return r   # all checks passed → issue=None


# ── Dialogue-level completeness check ─────────────────────────────────────────

def check_dialogue_completeness(dialogue_id: str, audio_root: Path) -> list[str]:
    """
    Returns list of missing WAV names for this dialogue.
    Expected: turn_01_buyer … turn_12_seller (alternating buyer/seller).
    """
    d    = audio_root / dialogue_id
    missing = []
    # Canonical turn → speaker mapping (1=buyer,2=seller,…,11=buyer,12=seller)
    for tidx in range(1, 13):
        spk = "buyer" if tidx % 2 == 1 else "seller"
        wav = d / f"turn_{tidx:02d}_{spk}.wav"
        if not wav.exists() or wav.stat().st_size == 0:
            missing.append(wav.name)
    return missing


# ── Main verification ──────────────────────────────────────────────────────────

def run_verification(audio_root: Path, interm_dir: Path,
                     expected_sr: int, min_dur: float, max_dur: float,
                     silence_rms: float, noise_crest: float) -> dict:

    print(f"\n{'='*62}")
    print(f"  Qwen3-TTS v2 Audio Verification")
    print(f"  audio_root  : {audio_root}")
    print(f"  expected sr : {expected_sr} Hz")
    print(f"  duration    : {min_dur}–{max_dur} s")
    print(f"  silence RMS : {silence_rms}")
    print(f"  noise crest : {noise_crest}")
    print(f"{'='*62}\n")

    # Collect all dialogue IDs from intermediate/
    all_dialogue_ids = sorted(p.stem for p in interm_dir.glob("dialogue_*.json"))
    total_dialogues  = len(all_dialogue_ids)
    total_expected   = total_dialogues * TURNS_PER_DIALOGUE

    print(f"  Dialogues expected : {total_dialogues:,}")
    print(f"  WAVs expected      : {total_expected:,}")
    print()

    # Per-check counters
    missing       = []   # (dialogue_id, wav_name)
    zero_byte     = []
    unreadable    = []
    wrong_sr      = []
    bad_duration  = []
    silent_files  = []
    noisy_files   = []
    ok_count      = 0

    # Per-dialogue completeness
    incomplete_dialogues = {}   # dialogue_id → [missing wav names]

    for i, did in enumerate(all_dialogue_ids, 1):
        if i % 100 == 0 or i == total_dialogues:
            print(f"  Checking {i:,}/{total_dialogues:,} dialogues…", end="\r", flush=True)

        # Dialogue-level completeness
        missing_wavs = check_dialogue_completeness(did, audio_root)
        if missing_wavs:
            incomplete_dialogues[did] = missing_wavs

        # Per-turn WAV inspection
        d = audio_root / did
        for tidx in range(1, 13):
            spk     = "buyer" if tidx % 2 == 1 else "seller"
            wav     = d / f"turn_{tidx:02d}_{spk}.wav"
            wav_str = str(wav)

            if not wav.exists():
                missing.append((did, wav.name))
                continue

            r = inspect_wav(wav, expected_sr, min_dur, max_dur, silence_rms, noise_crest)

            if r["issue"] is None:
                ok_count += 1
            elif "zero-byte" in (r["issue"] or ""):
                zero_byte.append(wav_str)
            elif "unreadable" in (r["issue"] or "") or "read failed" in (r["issue"] or ""):
                unreadable.append({"path": wav_str, "error": r["issue"]})
            elif "wrong sr" in (r["issue"] or ""):
                wrong_sr.append({"path": wav_str, "sr": r["sr"]})
            elif "duration" in (r["issue"] or ""):
                bad_duration.append({"path": wav_str, "duration_sec": r["duration_sec"],
                                     "issue": r["issue"]})
            elif r["silent"]:
                silent_files.append({"path": wav_str, "rms": r["rms"]})
            elif r["noisy"]:
                noisy_files.append({"path": wav_str, "issue": r["issue"]})
            else:
                unreadable.append({"path": wav_str, "error": r["issue"]})

    print()  # clear \r line

    return {
        "total_dialogues":          total_dialogues,
        "total_wavs_expected":      total_expected,
        "total_wavs_ok":            ok_count,
        "pct_ok":                   round(100 * ok_count / max(total_expected, 1), 2),

        # Critical
        "missing_count":            len(missing),
        "missing":                  [{"dialogue": d, "wav": w} for d, w in missing[:200]],
        "missing_truncated":        len(missing) > 200,

        "zero_byte_count":          len(zero_byte),
        "zero_byte":                zero_byte[:100],

        "unreadable_count":         len(unreadable),
        "unreadable":               unreadable[:50],

        "wrong_sr_count":           len(wrong_sr),
        "wrong_sr":                 wrong_sr[:50],

        # Warnings
        "bad_duration_count":       len(bad_duration),
        "bad_duration":             bad_duration[:50],

        "silent_count":             len(silent_files),
        "silent":                   silent_files[:50],

        "noisy_count":              len(noisy_files),
        "noisy":                    noisy_files[:50],

        # Dialogue completeness
        "incomplete_dialogues_count": len(incomplete_dialogues),
        "incomplete_dialogues":       {k: v for k, v in
                                       list(incomplete_dialogues.items())[:100]},
    }


# ── Report printer ─────────────────────────────────────────────────────────────

def print_report(r: dict) -> bool:
    """Print human-readable report. Returns True if no critical issues."""
    print(f"\n{'='*62}")
    print(f"  V2 AUDIO VERIFICATION REPORT")
    print(f"{'='*62}")
    print(f"\n  Dialogues     : {r['total_dialogues']:,}")
    print(f"  WAVs expected : {r['total_wavs_expected']:,}")
    print(f"  WAVs OK       : {r['total_wavs_ok']:,}  ({r['pct_ok']:.1f}%)")

    critical = False

    def _section(label, count, items, is_critical=True, show_n=5):
        nonlocal critical
        tag = "[CRITICAL]" if is_critical else "[WARN]    "
        ok  = "[OK]      "
        if count:
            if is_critical:
                critical = True
            print(f"\n  {tag} {label} : {count:,}")
            for item in items[:show_n]:
                print(f"    {item}")
            if count > show_n:
                print(f"    … and {count - show_n} more (see report JSON)")
        else:
            print(f"  {ok} {label} : 0")

    print()
    _section("Missing WAVs",        r["missing_count"],
             [f"{x['dialogue']}/{x['wav']}" for x in r["missing"][:5]])
    _section("Zero-byte WAVs",      r["zero_byte_count"],
             r["zero_byte"][:5])
    _section("Unreadable/corrupt",  r["unreadable_count"],
             [f"{x['path']}  →  {x['error']}" for x in r["unreadable"][:5]])
    _section("Wrong sample rate",   r["wrong_sr_count"],
             [f"{x['path']}  sr={x['sr']}" for x in r["wrong_sr"][:5]])
    _section("Bad duration",        r["bad_duration_count"],
             [f"{x['path']}  {x['issue']}" for x in r["bad_duration"][:5]],
             is_critical=False)
    _section("Silent renders",      r["silent_count"],
             [f"{x['path']}  RMS={x['rms']}" for x in r["silent"][:5]],
             is_critical=False)
    _section("Suspected noise/DC",  r["noisy_count"],
             [f"{x['path']}  {x['issue']}" for x in r["noisy"][:5]],
             is_critical=False)

    n = r["incomplete_dialogues_count"]
    if n:
        print(f"\n  [INFO]     Dialogues with ≥1 missing turn : {n:,}")
        for did, wavs in list(r["incomplete_dialogues"].items())[:3]:
            print(f"    {did}: missing {wavs}")

    print(f"\n{'='*62}")
    if critical:
        print("  STATUS: CRITICAL ISSUES — fix before running SFT.")
        print("  Re-run: CUDA_VISIBLE_DEVICES=1 python3 v2/generate_tts_v2.py --all")
    else:
        print("  STATUS: ALL CRITICAL CHECKS PASSED — safe to run SFT.")
        if r["bad_duration_count"] or r["silent_count"] or r["noisy_count"]:
            print("  (Warnings present — review report JSON for details)")
        print("  Run: CUDA_VISIBLE_DEVICES=1 python3 v2/train_sft_v2.py --no_qlora")
    print(f"{'='*62}\n")

    return not critical


# ── Entry ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Verify v2 TTS audio files.")
    parser.add_argument("--audio_root",  default=str(AUDIO_ROOT))
    parser.add_argument("--report",      default=str(REPORT_PATH))
    parser.add_argument("--expected_sr", type=int,   default=EXPECTED_SR)
    parser.add_argument("--min_dur",     type=float, default=MIN_DUR_SEC)
    parser.add_argument("--max_dur",     type=float, default=MAX_DUR_SEC)
    parser.add_argument("--silence_rms", type=float, default=SILENCE_RMS)
    parser.add_argument("--noise_crest", type=float, default=NOISE_CREST)
    args = parser.parse_args()

    audio_root = Path(args.audio_root)
    if not audio_root.exists():
        print(f"ERROR: audio_root not found: {audio_root}")
        sys.exit(1)

    report = run_verification(
        audio_root, INTERM_DIR,
        args.expected_sr, args.min_dur, args.max_dur,
        args.silence_rms, args.noise_crest,
    )

    Path(args.report).write_text(json.dumps(report, indent=2))
    print(f"  Full report → {args.report}")

    ok = print_report(report)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
