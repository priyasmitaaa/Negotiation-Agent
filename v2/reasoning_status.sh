#!/bin/bash
/home/paritosh/miniconda3/envs/casal/bin/python3 -c "
import json
from pathlib import Path
dataset = Path('/home/paritosh/priyasmita/Qwen3-tts/v2/dataset')
total_seller = 0
with_reasoning = 0
complete_dialogues = 0
partial_dialogues = 0
no_reasoning_dialogues = 0

for f in sorted(dataset.glob('dialogue_*.json')):
    try:
        d = json.loads(f.read_text())
    except:
        continue
    seller_turns = [t for t in d.get('processed', []) if t['speaker'] == 'seller']
    has_r = sum(1 for t in seller_turns if t.get('reasoning'))
    total_seller += len(seller_turns)
    with_reasoning += has_r
    if has_r == len(seller_turns): complete_dialogues += 1
    elif has_r > 0: partial_dialogues += 1
    else: no_reasoning_dialogues += 1

total_d = complete_dialogues + partial_dialogues + no_reasoning_dialogues
print(f'  Seller turns total       : {total_seller}')
print(f'  Reasoning written        : {with_reasoning}  ({100*with_reasoning/max(total_seller,1):.1f}%)')
print(f'  Missing                  : {total_seller - with_reasoning}')
print(f'  Dialogues fully done     : {complete_dialogues}/{total_d}  ({100*complete_dialogues/max(total_d,1):.1f}%)')
print(f'  Dialogues partial        : {partial_dialogues}')
print(f'  Dialogues not started    : {no_reasoning_dialogues}')
"
echo ""
echo "  Last progress:"
tail -1 /home/paritosh/priyasmita/Qwen3-tts/v2/reasoning_generation.log 2>/dev/null || echo "  (no log yet)"
echo ""
PID=$(ps aux | grep generate_reasoning | grep -v grep | awk '{print $2}')
if [ -n "$PID" ]; then echo "  Process: RUNNING (PID $PID)"; else echo "  Process: NOT RUNNING"; fi
