"""
remove_vaii.py — Strip all VAII fields and mentions from dataset/ dialogues.

Removes:
  - turn["vaii"]        (buyer turns)
  - turn["seller_vaii"] (seller turns)
  - turn["reasoning"]   (contains VAII numbers — will be regenerated with emotions)

Operates only on dataset/ — intermediate/ is untouched (VAII preserved there).
Fully idempotent — safe to run multiple times.

Usage:
  python3 Qwen3-tts/v2/remove_vaii.py --dry_run
  python3 Qwen3-tts/v2/remove_vaii.py
"""

import json
import argparse
from pathlib import Path

V2_ROOT     = Path(__file__).parent
DATASET_DIR = V2_ROOT / "dataset"

VAII_TURN_KEYS = {"vaii", "seller_vaii"}


def clean_dialogue(data: dict) -> tuple[dict, int]:
    """Remove VAII fields from all turns. Returns (cleaned_data, n_fields_removed)."""
    removed = 0
    for turn in data.get("processed", []):
        for key in VAII_TURN_KEYS:
            if key in turn:
                del turn[key]
                removed += 1
        # Remove reasoning — it contains VAII values and will be regenerated
        if "reasoning" in turn:
            del turn["reasoning"]
            removed += 1
    return data, removed


def main(dry_run: bool = False):
    tag = "[DRY RUN] " if dry_run else ""
    print(f"\n{'='*65}")
    print(f"  {tag}REMOVE VAII FROM dataset/")
    print(f"{'='*65}\n")

    if not DATASET_DIR.exists():
        print(f"  ERROR: {DATASET_DIR} does not exist. Run backup_and_copy.py first.")
        return

    files = sorted(DATASET_DIR.glob("dialogue_*.json"))
    print(f"  Files to process: {len(files)}")

    total_removed  = 0
    files_modified = 0

    for fpath in files:
        try:
            data = json.loads(fpath.read_text())
        except Exception as e:
            print(f"  WARN: could not read {fpath.name}: {e}")
            continue

        cleaned, n = clean_dialogue(data)
        if n > 0:
            total_removed  += n
            files_modified += 1
            if not dry_run:
                fpath.write_text(json.dumps(cleaned, indent=2, ensure_ascii=False))

    print(f"  Files modified      : {files_modified}/{len(files)}")
    print(f"  Fields removed      : {total_removed}")
    print(f"  Fields per file avg : {total_removed/max(files_modified,1):.1f}")

    if dry_run:
        print(f"\n  DRY RUN — no files written. Remove --dry_run to apply.")
    else:
        print(f"\n  Done. dataset/ is now VAII-free.")
        print(f"  intermediate/ still has original VAII data.")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()
    main(dry_run=args.dry_run)
