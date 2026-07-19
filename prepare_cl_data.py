#!/usr/bin/env python3
"""
prepare_cl_data.py
==================
Prepare curriculum learning data files for CL-SFT phases.

Outputs
-------
  dataset/cl_phase1.jsonl   EASY only + optional EL floor from MEDIUM
  dataset/cl_phase2.jsonl   EASY + MEDIUM (shuffled)
  dataset/cl_phase3.jsonl   HARD + 20% EASY replay (catastrophic-forgetting prevention)

Phase 3 replay buffer (Lopez-Paz 2017 / Rebuffi et al. iCaRL):
  Sample 20% of EASY instances to replay during HARD training.
  This prevents catastrophic forgetting of the Phase 1/2 decision boundary.
  Sampling is stratified by class (EL/MM) and shuffled with a fixed seed.

EL floor (--el_floor N, default 30):
  Phase 1 is 97% MM (only 24 EL in 716 EASY samples). Without EL signal,
  Phase 1 risks collapsing EL recall. We augment Phase 1 with N EL records
  sampled from the MEDIUM tier so the model retains EL discrimination before
  seeing the full EASY+MEDIUM mix in Phase 2.

Usage
-----
  python3 prepare_cl_data.py
  python3 prepare_cl_data.py --el_floor 30 --replay_frac 0.20 --seed 42
  python3 prepare_cl_data.py --el_floor 0   # disable EL floor
"""

import argparse
import json
import random
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved {len(records):,} records → {path}")


def stratified_sample(records: list[dict], frac: float, seed: int) -> list[dict]:
    """Sample `frac` of records, stratified by label (0=EL, 1=MM)."""
    rng = random.Random(seed)
    by_label: dict[int, list] = {}
    for r in records:
        lbl = r.get("label", 0)
        by_label.setdefault(lbl, []).append(r)

    sampled = []
    for lbl, group in by_label.items():
        n = max(1, round(len(group) * frac))
        chosen = rng.sample(group, min(n, len(group)))
        sampled.extend(chosen)
    return sampled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    default="dataset")
    parser.add_argument("--replay_frac", type=float, default=0.20,
                        help="Fraction of EASY to replay in Phase 3 (default 0.20)")
    parser.add_argument("--el_floor",   type=int,   default=30,
                        help="Number of EL samples from MEDIUM to add to Phase 1. "
                             "Phase 1 has only 24 EL (3%%) without this; the floor "
                             "gives the model enough EL signal to retain recall. "
                             "Set 0 to disable. (default 30)")
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    easy   = load_jsonl(data_dir / "tier_easy.jsonl")
    medium = load_jsonl(data_dir / "tier_medium.jsonl")
    hard   = load_jsonl(data_dir / "tier_hard.jsonl")

    print(f"\n  Tier sizes:  EASY={len(easy)}  MEDIUM={len(medium)}  HARD={len(hard)}")

    rng = random.Random(args.seed)

    # Phase 1: EASY only + optional EL floor from MEDIUM
    phase1 = list(easy)
    if args.el_floor > 0:
        medium_el = [r for r in medium if r.get("label", 0) == 0]   # EL records only
        n_floor = min(args.el_floor, len(medium_el))
        el_floor_samples = rng.sample(medium_el, n_floor)
        phase1.extend(el_floor_samples)
        print(f"  EL floor: added {n_floor} EL records from MEDIUM to Phase 1")
    rng.shuffle(phase1)
    save_jsonl(phase1, data_dir / "cl_phase1.jsonl")

    # Phase 2: EASY + MEDIUM, shuffled
    phase2 = list(easy) + list(medium)
    rng.shuffle(phase2)
    save_jsonl(phase2, data_dir / "cl_phase2.jsonl")

    # Phase 3: HARD + 20% EASY replay, shuffled
    replay = stratified_sample(easy, args.replay_frac, args.seed)
    phase3 = list(hard) + replay
    rng.shuffle(phase3)
    save_jsonl(phase3, data_dir / "cl_phase3.jsonl")

    # Print class balance per phase
    print()
    for name, records in [("Phase1", phase1), ("Phase2", phase2), ("Phase3", phase3)]:
        n    = len(records)
        n_mm = sum(1 for r in records if r.get("label", 0) == 1)
        n_el = n - n_mm
        print(f"  {name}: n={n}  MM={n_mm}({100*n_mm/n:.0f}%)  EL={n_el}({100*n_el/n:.0f}%)")

    print(f"\n  Replay buffer: {len(replay)} EASY samples in Phase 3 "
          f"({args.replay_frac*100:.0f}% of EASY, stratified by class, seed={args.seed})\n")


if __name__ == "__main__":
    main()
