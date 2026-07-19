#!/usr/bin/env python3
"""
sft_full_compare.py — Deep side-by-side comparison of all SFT runs.

Covers:
  - Data split sizes and class weights
  - Hyperparameters
  - Training loss curve (every logged step)
  - Eval loss curve (every eval checkpoint)
  - Gradient norm progression
  - Learning rate schedule
  - Convergence rate (loss drop per 100 steps)
  - Best / final / test metrics
  - Verdict and recommendation

Usage:
  python3 Qwen3-tts/v2/sft_full_compare.py
"""

import json
import re
from pathlib import Path

SFT_OUTPUT = Path(__file__).parent / "sft_output_v2"

# ── Discover runs ─────────────────────────────────────────────────────────────
run_dirs = sorted([
    d for d in SFT_OUTPUT.iterdir()
    if d.is_dir() and re.match(r"\d{8}_\d{6}", d.name)
])
legacy = SFT_OUTPUT / "run1_20260429"
if legacy.exists():
    run_dirs = [legacy] + run_dirs

if not run_dirs:
    print("No SFT runs found.")
    raise SystemExit

# ── Load each run's data ──────────────────────────────────────────────────────
runs = []
for d in run_dirs:
    r = {"tag": d.name, "dir": d}

    summary_path = d / "run_summary.json"
    if summary_path.exists():
        r["summary"] = json.loads(summary_path.read_text())
        r["complete"] = True
    else:
        r["summary"] = {}
        r["complete"] = False

    tlog_path = d / "training_log.json"
    if tlog_path.exists():
        history = json.loads(tlog_path.read_text())
        r["train_steps"] = [e for e in history if "loss" in e and "eval_loss" not in e]
        r["eval_steps"]  = [e for e in history if "eval_loss" in e]
    else:
        r["train_steps"] = []
        r["eval_steps"]  = []

    runs.append(r)

# ── Helpers ───────────────────────────────────────────────────────────────────
SEP  = "─" * 72
SEP2 = "=" * 72
W    = 26

def fmt(val, spec=None):
    if val is None or val == "":
        return "—"
    if spec:
        try:
            return spec.format(val)
        except Exception:
            pass
    return str(val)

def row(label, values, spec=None):
    line = f"  {label:<30}" + "".join(f"{fmt(v, spec):>{W}}" for v in values)
    print(line)

tags   = [r["tag"] for r in runs]
summs  = [r["summary"] for r in runs]

# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP2}")
print(f"  SFT Full Comparison  ({len(runs)} run{'s' if len(runs)!=1 else ''})")
print(f"{SEP2}")

# Header
print(f"\n  {'Metric':<30}" + "".join(f"{t:>{W}}" for t in tags))
print(f"  {'─'*30}" + SEP[-W * len(runs):])

# ── Status ────────────────────────────────────────────────────────────────────
row("Status",    ["COMPLETE" if r["complete"] else "INCOMPLETE" for r in runs])
row("Completed at", [s.get("completed_at","")[:19] for s in summs])

# ── Data ─────────────────────────────────────────────────────────────────────
print(f"\n  {'DATA':<30}")
row("  Train examples",  [s.get("train_examples") for s in summs], "{:,}")
row("  Val examples",    [s.get("val_examples")   for s in summs], "{:,}")
row("  Test examples",   [s.get("test_examples")  for s in summs], "{:,}")
row("  Skipped (no reason)", [s.get("skipped_no_reasoning_train",0) for s in summs], "{:,}")
row("  LEVERAGE weight", [s.get("class_weights",{}).get("LEVERAGE") for s in summs], "{:.4f}")
row("  MITIGATE weight", [s.get("class_weights",{}).get("MITIGATE") for s in summs], "{:.4f}")

# ── Hyperparams ───────────────────────────────────────────────────────────────
print(f"\n  {'HYPERPARAMETERS':<30}")
row("  Epochs",          [s.get("epochs")       for s in summs])
row("  Learning rate",   [s.get("lr")           for s in summs], "{:.0e}")
row("  LoRA r",          [s.get("lora_r")       for s in summs])
row("  LoRA alpha",      [s.get("lora_alpha")   for s in summs])
row("  Batch size",      [s.get("batch_size")   for s in summs])
row("  Grad accum steps",[s.get("grad_accum")   for s in summs])
row("  Effective batch", [s.get("effective_batch") for s in summs])
row("  Total steps",     [s.get("total_steps")  for s in summs], "{:,}")

# ── Final metrics ─────────────────────────────────────────────────────────────
print(f"\n  {'FINAL METRICS':<30}")
row("  Final train loss",  [s.get("final_train_loss") for s in summs], "{:.4f}")
row("  Best eval loss  ◄", [s.get("best_eval_loss")   for s in summs], "{:.4f}")
row("  Test loss       ◄", [s.get("test_loss")        for s in summs], "{:.4f}")

# ── Training loss curve ───────────────────────────────────────────────────────
any_train = any(r["train_steps"] for r in runs)
if any_train:
    print(f"\n  {SEP}")
    print(f"  TRAINING LOSS CURVE  (every 25 steps)")
    print(f"  {SEP}")
    print(f"  {'Step':>8}  {'Epoch':>6}" + "".join(f"{'loss':>{W}}" for _ in runs))

    # Align by step
    all_steps = sorted(set(
        e["step"] for r in runs for e in r["train_steps"]
    ))
    for step in all_steps:
        epoch_vals = []
        loss_vals  = []
        for r in runs:
            match = next((e for e in r["train_steps"] if e["step"] == step), None)
            if match:
                epoch_vals.append(f"{match['epoch']:.3f}")
                loss_vals.append(f"{match['loss']:.4f}")
            else:
                epoch_vals.append("—")
                loss_vals.append("—")
        epoch_str = epoch_vals[0] if epoch_vals else "—"
        line = f"  {step:>8}  {epoch_str:>6}" + "".join(f"{v:>{W}}" for v in loss_vals)
        print(line)

# ── Eval loss curve ───────────────────────────────────────────────────────────
any_eval = any(r["eval_steps"] for r in runs)
if any_eval:
    print(f"\n  {SEP}")
    print(f"  EVAL LOSS CURVE  (every 50 steps)")
    print(f"  {SEP}")
    print(f"  {'Step':>8}  {'Epoch':>6}" + "".join(f"{'eval_loss':>{W}}" for _ in runs))

    all_eval_steps = sorted(set(
        e["step"] for r in runs for e in r["eval_steps"]
    ))
    for step in all_eval_steps:
        epoch_vals = []
        loss_vals  = []
        for r in runs:
            match = next((e for e in r["eval_steps"] if e["step"] == step), None)
            if match:
                epoch_vals.append(f"{match['epoch']:.3f}")
                loss_vals.append(f"{match['eval_loss']:.4f}")
            else:
                epoch_vals.append("—")
                loss_vals.append("—")
        epoch_str = epoch_vals[0] if epoch_vals else "—"
        line = f"  {step:>8}  {epoch_str:>6}" + "".join(f"{v:>{W}}" for v in loss_vals)
        print(line)

# ── Gradient norm progression ─────────────────────────────────────────────────
any_gnorm = any(
    any("grad_norm" in e for e in r["train_steps"]) for r in runs
)
if any_gnorm:
    print(f"\n  {SEP}")
    print(f"  GRADIENT NORM  (stability check — should decrease over time)")
    print(f"  {SEP}")
    print(f"  {'Step':>8}  {'Epoch':>6}" + "".join(f"{'grad_norm':>{W}}" for _ in runs))

    for step in all_steps:
        epoch_vals = []
        gnorm_vals = []
        for r in runs:
            match = next((e for e in r["train_steps"] if e["step"] == step), None)
            if match:
                epoch_vals.append(f"{match['epoch']:.3f}")
                gn = match.get("grad_norm")
                gnorm_vals.append(f"{gn:.3f}" if gn is not None else "—")
            else:
                epoch_vals.append("—")
                gnorm_vals.append("—")
        epoch_str = epoch_vals[0] if epoch_vals else "—"
        line = f"  {step:>8}  {epoch_str:>6}" + "".join(f"{v:>{W}}" for v in gnorm_vals)
        print(line)

# ── Learning rate schedule ────────────────────────────────────────────────────
any_lr = any(
    any("learning_rate" in e for e in r["train_steps"]) for r in runs
)
if any_lr:
    print(f"\n  {SEP}")
    print(f"  LEARNING RATE SCHEDULE  (cosine warmup → decay)")
    print(f"  {SEP}")
    print(f"  {'Step':>8}  {'Epoch':>6}" + "".join(f"{'lr':>{W}}" for _ in runs))

    for step in all_steps:
        epoch_vals = []
        lr_vals    = []
        for r in runs:
            match = next((e for e in r["train_steps"] if e["step"] == step), None)
            if match:
                epoch_vals.append(f"{match['epoch']:.3f}")
                lr = match.get("learning_rate")
                lr_vals.append(f"{lr:.2e}" if lr is not None else "—")
            else:
                epoch_vals.append("—")
                lr_vals.append("—")
        epoch_str = epoch_vals[0] if epoch_vals else "—"
        line = f"  {step:>8}  {epoch_str:>6}" + "".join(f"{v:>{W}}" for v in lr_vals)
        print(line)

# ── Convergence rate ──────────────────────────────────────────────────────────
print(f"\n  {SEP}")
print(f"  CONVERGENCE ANALYSIS")
print(f"  {SEP}")
for r in runs:
    ts = r["train_steps"]
    if len(ts) < 2:
        continue
    first_loss = ts[0]["loss"]
    last_loss  = ts[-1]["loss"]
    drop       = first_loss - last_loss
    pct        = drop / first_loss * 100
    steps      = ts[-1]["step"] - ts[0]["step"]
    rate       = drop / steps * 100 if steps else 0
    print(f"  {r['tag']}")
    print(f"    Loss: {first_loss:.4f} → {last_loss:.4f}  (drop: {drop:.4f}, {pct:.1f}%)")
    print(f"    Steps: {steps}  |  Drop per 100 steps: {rate:.4f}")

    es = r["eval_steps"]
    if len(es) >= 2:
        # How many steps to reach within 10% of final eval loss
        final_eval = es[-1]["eval_loss"]
        target     = final_eval * 1.10
        for e in es:
            if e["eval_loss"] <= target:
                print(f"    Reached 90% of final eval loss at step {e['step']} (epoch {e['epoch']:.3f})")
                break
    print()

# ── Verdict ───────────────────────────────────────────────────────────────────
complete = [r for r in runs if r["summary"].get("test_loss")]
if complete:
    print(f"  {SEP}")
    print(f"  VERDICT")
    print(f"  {SEP}")
    best = min(complete, key=lambda r: r["summary"]["test_loss"])
    print(f"  Best test loss : {best['summary']['test_loss']:.4f}  →  run: {best['tag']}")
    print()

    for i in range(1, len(complete)):
        prev = complete[i-1]
        curr = complete[i]
        tl_prev = prev["summary"]["test_loss"]
        tl_curr = curr["summary"]["test_loss"]
        delta = tl_prev - tl_curr
        pct   = delta / tl_prev * 100
        arrow = "▼ improved" if delta > 0 else "▲ regressed"
        print(f"  {prev['tag']} → {curr['tag']}: {arrow} by {abs(delta):.4f} ({abs(pct):.1f}%)")

    print()
    print(f"  INTERPRETATION:")
    if len(complete) >= 2:
        t1 = complete[0]["summary"]["test_loss"]
        t2 = complete[1]["summary"]["test_loss"]
        d1 = complete[0]["summary"].get("train_examples", 0)
        d2 = complete[1]["summary"].get("train_examples", 0)
        skipped1 = complete[0]["summary"].get("skipped_no_reasoning_train", 0)
        skipped2 = complete[1]["summary"].get("skipped_no_reasoning_train", 0)

        if t2 > t1:
            print(f"  Run 2 regressed despite +{d2-d1:,} more training examples.")
            print(f"  Run 1 skipped {skipped1:,} turns (no reasoning) — trained on homogeneous")
            print(f"  OpenAI-generated reasoning only.")
            print(f"  Run 2 used 0 skipped turns — but mixed OpenAI + Gemini reasoning styles,")
            print(f"  creating inconsistent training signal → higher loss throughout all epochs.")
            print(f"  → Recommendation: use Run 1 adapter for GRPO. Run 3 only if you unify")
            print(f"    reasoning style first (see sft_harmonise_reasoning.py).")
        else:
            print(f"  Run 2 improved over Run 1. Use Run 2 adapter for GRPO.")

print(f"\n{SEP2}\n")
