# Causal RM Experiment — lambda_inv_sweep_3.5

Config: `/home/paritosh/priyasmita/Qwen3-tts/v2/causal_rm_experiments/lambda_inv_sweep_3.5_20260801_132137/config.json`
Checkpoint: `/home/paritosh/priyasmita/Qwen3-tts/v2/causal_rm_experiments/lambda_inv_sweep_3.5_20260801_132137/checkpoint`
Full audit report: `/home/paritosh/priyasmita/Qwen3-tts/v2/causal_rm_experiments/lambda_inv_sweep_3.5_20260801_132137/audit/report_causal_rm_val.md`

## Training (final logged step)

```json
{
  "epoch": 0,
  "step": 180,
  "loss": 0.6700148582458496,
  "bt_accuracy_per_dimension": {
    "decision_score": 0.8617,
    "emotion": 0.9494,
    "flip_score": 0.7143
  },
  "invariance_mse_per_dimension": {
    "price_strategy": 0.00077,
    "emotion": 0.010532,
    "progression": 0.001404
  },
  "head_correlation": {
    "mean_abs_off_diagonal": 0.3633
  },
  "embedder_cache_hit_rate": 0.3456
}
```

## Audit summary

### Causal-pair ranking accuracy (chance = 50%)

| Dimension | Accuracy |
|---|---:|
| emotion | 0.9691 |
| decision_score | 0.8400 |
| flip_score | 1.0000 |

### Spurious-pair invariance (lower = better)

| Dimension | Mean \|Δ\| |
|---|---:|
| decision_score | 0.28738 |
| price_strategy | 0.012991 |
| emotion | 0.001061 |
| progression | 0.013834 |
| flip_score | 0.048471 |

## Go/No-Go (manual judgment call — see causal_rm_architecture.md section 7)

- [ ] Causal ranking accuracy meaningfully above 50% for C1/C3/C5-covered dimensions
- [ ] Spurious invariance better (lower) than the heuristic baseline (compare against `causal_rm_audit_heuristic_val.json` in the repo root)
- [ ] Reward distribution not saturated at ±1 for every example

If all pass: proceed to `train_grpo_curriculum.py --causal_rm_checkpoint /home/paritosh/priyasmita/Qwen3-tts/v2/causal_rm_experiments/lambda_inv_sweep_3.5_20260801_132137/checkpoint` for a small GRPO pilot. If not: iterate on lambda_bt/lambda_inv/lambda_reg or architecture before scaling up.