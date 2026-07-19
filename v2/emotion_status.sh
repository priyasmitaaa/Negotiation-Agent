#!/bin/bash
python3 -c "
import json
from pathlib import Path
dataset = Path('/home/paritosh/priyasmita/Qwen3-tts/v2/dataset')
complete = partial = none = corrupt = 0
missing_turns = 0
for f in sorted(dataset.glob('dialogue_*.json')):
    try:
        d = json.loads(f.read_text())
    except:
        corrupt += 1
        continue
    turns = d.get('processed', [])
    with_em = sum(1 for t in turns if 'emotion' in t)
    wo_em = len(turns) - with_em
    missing_turns += wo_em
    if with_em == len(turns): complete += 1
    elif with_em > 0: partial += 1
    else: none += 1
total = complete+partial+none+corrupt
print(f'  Fully complete : {complete}/{total}  ({100*complete/total:.1f}%)')
print(f'  Partial        : {partial}')
print(f'  Not started    : {none}')
print(f'  Corrupt        : {corrupt}')
print(f'  Missing turns  : {missing_turns}')
"
echo ""
echo "  Last progress:"
grep "Progress:" /home/paritosh/priyasmita/Qwen3-tts/v2/emotion_detection.log | tail -1
echo ""
PID=$(ps aux | grep detect_emotions | grep -v grep | awk '{print $2}')
if [ -n "$PID" ]; then echo "  Process: RUNNING (PID $PID)"; else echo "  Process: NOT RUNNING"; fi
