"""Reprocess only the problematic dialogues identified in validation report"""
import json
from pathlib import Path
import subprocess

# Load validation report
preprocessed_dir = Path("/home/paritosh/priyasmita/dialogue-dataset/Qwen3-tts/preprocessed")
report_path = preprocessed_dir / "validation_report.json"

with open(report_path, 'r') as f:
    report = json.load(f)

# Get all problematic dialogue paths
problematic_files = []
for issue_type, files in report['issue_details'].items():
    problematic_files.extend(files)

# Remove duplicates
problematic_files = list(set(problematic_files))

print(f"Found {len(problematic_files)} problematic dialogues to reprocess")
print(f"  - Empty trajectories: {len(report['issue_details']['Empty trajectory'])}")
print(f"  - No price mentions: {len(report['issue_details']['No price mentions found in any turn'])}")
print()

# Reprocess each problematic dialogue
source_base = Path("/home/paritosh/priyasmita/dialogue-dataset/dialogues")
success_count = 0
still_failing = []

for relative_path in problematic_files:
    # Convert preprocessed path back to source path
    source_path = source_base / relative_path
    
    if not source_path.exists():
        print(f"⚠ Source file not found: {source_path}")
        continue
    
    # Run preprocessing on this specific dialogue
    cmd = [
        "python3", "preprocess_dialogues.py",
        "--dialogue", str(source_path),
        "--source-dir", str(source_base),
        "--output-dir", "preprocessed"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd="/home/paritosh/priyasmita/dialogue-dataset/Qwen3-tts")
    
    if result.returncode == 0:
        success_count += 1
        print(f"✓ {relative_path}")
    else:
        still_failing.append(relative_path)
        print(f"✗ {relative_path}")
        if result.stderr:
            print(f"  Error: {result.stderr[:100]}")

print()
print("=" * 70)
print(f"Reprocessing complete:")
print(f"  Success: {success_count}/{len(problematic_files)}")
print(f"  Still failing: {len(still_failing)}")

if still_failing:
    print(f"\nStill failing files:")
    for f in still_failing:
        print(f"  - {f}")
