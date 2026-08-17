#!/usr/bin/env python3
"""
run_causal_rm_experiment.py — single-command experiment runner for the
causal reward model: train -> audit -> report -> summary, in one call.

Rationale (reviewer-recommended, 2026-07-20): once ablations start (varying
lambda_bt/lambda_inv/lambda_reg, max_pairs, backbone choice, etc.), running
train_causal_rm.py and causal_rm_audit.py by hand for each variant is slow
and error-prone — configs drift, someone forgets to audit a checkpoint
before reporting it, results end up scattered across ad-hoc file names. This
script makes each experiment reproducible and self-contained: one config in,
one directory out, containing everything needed to know what was run and
what happened.

For each run, creates:
    causal_rm_experiments/<name>_<timestamp>/
        config.json              <- exact parameters used (train_causal_rm.py args)
        checkpoint/               <- rubric_head.pt, config.json (backbone), training_history.json
        audit/
            causal_rm_audit_causal_rm_val.json
            report_causal_rm_val.md
            *.png                 <- ranking accuracy, invariance, distributions
        summary.md                <- one-page combined summary (training + audit)

Usage:
    python3 run_causal_rm_experiment.py --name prototype_200pairs \\
        --max_pairs 200 --epochs 1 --lr 1e-4

    # Full run once the prototype passes audit:
    python3 run_causal_rm_experiment.py --name full_s1s3 \\
        --max_pairs 0 --epochs 3 --lr 1e-4
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Optional

from train_causal_rm import train
from causal_rm_audit import run_audit, generate_report


V2_ROOT = Path(__file__).parent
EXPERIMENTS_ROOT = V2_ROOT / "causal_rm_experiments"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", required=True, help="Experiment name, used as directory prefix")
    p.add_argument("--pairs_dir", default=str(V2_ROOT))
    p.add_argument("--backbone_model_id", default="Qwen/Qwen3-8B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lambda_bt", type=float, default=1.0)
    p.add_argument("--lambda_inv", type=float, default=1.0)
    p.add_argument("--lambda_reg", type=float, default=1e-3)
    p.add_argument("--max_pairs", type=int, default=0, help="0 = use all pairs. Prototype runs should set this small (e.g. 200-500).")
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--audit_split", default="val", choices=["val", "test"], help="Which split to audit the freshly trained checkpoint on. Use 'val' for iteration; only use 'test' for the final reportable run.")
    p.add_argument("--audio_fusion", action="store_true", help="v2 (causal_rm_architecture.md section 9): train SERFusionHead + audit it with SER audio scoring instead of v1's text-only RubricHead.")
    p.add_argument("--ser_device", default=None, help="Device for the frozen SER model; defaults to --device if unset.")
    p.add_argument(
        "--audit_max_pairs",
        type=int,
        default=0,
        help="0 = audit the full split (slow but precise — this is what the audit's held-out numbers are for). Set to a few hundred for a cheap sweep iteration where you mainly care about the sweep's relative trend, not a publishable per-run number.",
    )
    return p.parse_args()


def run_experiment(args: argparse.Namespace) -> Path:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = EXPERIMENTS_ROOT / f"{args.name}_{timestamp}"
    checkpoint_dir = exp_dir / "checkpoint"
    audit_dir = exp_dir / "audit"
    exp_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    (exp_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"[1/4] Experiment dir: {exp_dir}")

    print(f"[2/4] Training (max_pairs={args.max_pairs or 'all'}, epochs={args.epochs})...")
    train(
        pairs_dir=Path(args.pairs_dir),
        output_dir=checkpoint_dir,
        backbone_model_id=args.backbone_model_id,
        device=args.device,
        epochs=args.epochs,
        lr=args.lr,
        lambda_bt=args.lambda_bt,
        lambda_inv=args.lambda_inv,
        lambda_reg=args.lambda_reg,
        max_pairs=args.max_pairs,
        log_every=args.log_every,
        seed=args.seed,
        audio_fusion=args.audio_fusion,
        ser_device=args.ser_device,
    )

    print(f"[3/4] Auditing on split={args.audit_split} (max_pairs={args.audit_max_pairs or 'all'})...")
    result, causal_raw, spurious_raw = run_audit(
        pairs_dir=Path(args.pairs_dir),
        split=args.audit_split,
        backend="causal_rm",
        checkpoint=str(checkpoint_dir),
        device=args.device,
        collect_raw=True,
        max_pairs=args.audit_max_pairs,
        seed=args.seed,
    )
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / f"causal_rm_audit_causal_rm_{args.audit_split}.json").write_text(json.dumps(result, indent=2))
    report_path = generate_report(result, causal_raw, spurious_raw, audit_dir)

    print("[4/4] Writing summary...")
    summary_path = write_summary(exp_dir, config, result, report_path)
    print(f"\nDone. Summary: {summary_path}")
    return exp_dir


def write_summary(exp_dir: Path, config: dict, audit_result: dict, report_path: Path) -> Path:
    history_path = exp_dir / "checkpoint" / "training_history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    last_metrics = history[-1] if history else {}

    lines = [
        f"# Causal RM Experiment — {config['name']}",
        "",
        f"Config: `{exp_dir / 'config.json'}`",
        f"Checkpoint: `{exp_dir / 'checkpoint'}`",
        f"Full audit report: `{report_path}`",
        "",
        "## Training (final logged step)",
        "",
        "```json",
        json.dumps(last_metrics, indent=2),
        "```",
        "",
        "## Audit summary",
        "",
        "### Causal-pair ranking accuracy (chance = 50%)",
        "",
        "| Dimension | Accuracy |",
        "|---|---:|",
    ]
    for d, a in audit_result["causal_pairs"]["ranking_accuracy_per_dimension"].items():
        lines.append(f"| {d} | {a:.4f} |")
    lines += [
        "",
        "### Spurious-pair invariance (lower = better)",
        "",
        "| Dimension | Mean \\|Δ\\| |",
        "|---|---:|",
    ]
    for d, v in audit_result["spurious_pairs"]["mean_abs_delta_per_dimension"].items():
        lines.append(f"| {d} | {v if v is None else round(v, 6)} |")

    lines += [
        "",
        "## Go/No-Go (manual judgment call — see causal_rm_architecture.md section 7)",
        "",
        "- [ ] Causal ranking accuracy meaningfully above 50% for C1/C3/C5-covered dimensions",
        "- [ ] Spurious invariance better (lower) than the heuristic baseline "
        "(compare against `causal_rm_audit_heuristic_val.json` in the repo root)",
        "- [ ] Reward distribution not saturated at ±1 for every example",
        "",
        "If all pass: proceed to `train_grpo_curriculum.py --causal_rm_checkpoint "
        f"{exp_dir / 'checkpoint'}` for a small GRPO pilot. If not: iterate on "
        "lambda_bt/lambda_inv/lambda_reg or architecture before scaling up.",
    ]

    summary_path = exp_dir / "summary.md"
    summary_path.write_text("\n".join(lines))
    return summary_path


def main() -> None:
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
