"""
Read-only: Examine the edge cases in 'fair' harm_direction:
- 18 cases where final < fair_price (BENEFIT)
- 10 cases where final = fair_price (EQUAL)
Also check what VAII scores exist in templates 1-50 markdown for these ranges.
"""
import json, os, glob

base = '/home/paritosh/priyasmita/dialogue-dataset/Qwen3-tts/preprocessed'

print("=== 'fair' cases where final < fair_price (BENEFIT) ===")
for t in range(1, 51):
    tdir = os.path.join(base, f'template_{t:02d}')
    if not os.path.exists(tdir):
        continue
    for f in sorted(glob.glob(f'{tdir}/**/*.json', recursive=True)):
        with open(f) as fp:
            d = json.load(fp)
        labels = d['metadata']['labels']
        prices = d['metadata']['prices']
        if labels.get('harm_direction') == 'fair' and prices['final_price'] < prices['fair_price']:
            print(f"  T{t:02d} {'/'.join(f.split('/')[-3:])}: final={prices['final_price']}, fair={prices['fair_price']}, vuln={labels['vulnerability']}, anch={labels['anchoring']}, bias={labels['bias_decision']}")

print("\n=== 'fair' cases where final = fair_price (EQUAL) ===")
for t in range(1, 51):
    tdir = os.path.join(base, f'template_{t:02d}')
    if not os.path.exists(tdir):
        continue
    for f in sorted(glob.glob(f'{tdir}/**/*.json', recursive=True)):
        with open(f) as fp:
            d = json.load(fp)
        labels = d['metadata']['labels']
        prices = d['metadata']['prices']
        if labels.get('harm_direction') == 'fair' and prices['final_price'] == prices['fair_price']:
            print(f"  T{t:02d} {'/'.join(f.split('/')[-3:])}: final={prices['final_price']}, fair={prices['fair_price']}, vuln={labels['vulnerability']}, anch={labels['anchoring']}, bias={labels['bias_decision']}")

# Also check: what templates have 'benefit' (not 'benefit_buyer') in 1-50?
print("\n=== Templates 1-50 with 'benefit' harm_direction (sample 10) ===")
count = 0
for t in range(1, 51):
    tdir = os.path.join(base, f'template_{t:02d}')
    if not os.path.exists(tdir):
        continue
    for f in sorted(glob.glob(f'{tdir}/**/*.json', recursive=True)):
        with open(f) as fp:
            d = json.load(fp)
        labels = d['metadata']['labels']
        prices = d['metadata']['prices']
        if labels.get('harm_direction') == 'benefit':
            print(f"  T{t:02d} {'/'.join(f.split('/')[-3:])}: final={prices['final_price']}, fair={prices['fair_price']}, bias={labels['bias_decision']}")
            count += 1
            if count >= 10:
                break
    if count >= 10:
        break

# Summary of what needs to change
print("\n=== SUMMARY: What needs to change in harm_direction ===")
changes = {
    'fair->harm_buyer (final>fair)': 0,
    'fair->benefit_buyer (final<fair)': 0,
    'fair->benefit_buyer (final=fair, <=)': 0,
    'benefit->benefit_buyer (rename)': 0,
    'overpay->overpay (keep or rename to harm_buyer?)': 0,
}
for t in range(1, 51):
    tdir = os.path.join(base, f'template_{t:02d}')
    if not os.path.exists(tdir):
        continue
    for f in sorted(glob.glob(f'{tdir}/**/*.json', recursive=True)):
        with open(f) as fp:
            d = json.load(fp)
        labels = d['metadata']['labels']
        prices = d['metadata']['prices']
        hd = labels.get('harm_direction')
        final = prices['final_price']
        fair = prices['fair_price']
        if hd == 'fair':
            if final > fair:
                changes['fair->harm_buyer (final>fair)'] += 1
            else:
                changes['fair->benefit_buyer (final<fair)'] += 1
        elif hd == 'benefit':
            changes['benefit->benefit_buyer (rename)'] += 1
        elif hd == 'overpay':
            changes['overpay->overpay (keep or rename to harm_buyer?)'] += 1

for k, v in changes.items():
    print(f"  {k}: {v}")
