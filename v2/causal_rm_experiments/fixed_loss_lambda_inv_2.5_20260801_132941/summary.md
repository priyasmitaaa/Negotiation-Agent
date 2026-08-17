# Causal RM Experiment — fixed_loss_lambda_inv_2.5

Config: `/home/paritosh/priyasmita/Qwen3-tts/v2/causal_rm_experiments/fixed_loss_lambda_inv_2.5_20260801_132941/config.json`
Checkpoint: `/home/paritosh/priyasmita/Qwen3-tts/v2/causal_rm_experiments/fixed_loss_lambda_inv_2.5_20260801_132941/checkpoint`
Full audit report: `/home/paritosh/priyasmita/Qwen3-tts/v2/causal_rm_experiments/fixed_loss_lambda_inv_2.5_20260801_132941/audit/report_causal_rm_val.md`

## Training (final logged step)

```json
{
  "epoch": 0,
  "step": 180,
  "loss": 0.45747336745262146,
  "bt_accuracy_per_dimension": {
    "decision_score": 0.8511,
    "emotion": 0.9241,
    "flip_score": 1.0
  },
  "invariance_mse_per_dimension": {
    "decision_score": 0.033516,
    "price_strategy": 0.001776,
    "emotion": 0.018463,
    "progression": 0.001101,
    "flip_score": 0.001275
  },
  "head_correlation": {
    "mean_abs_off_diagonal": 0.3269
  },
  "embedder_cache_hit_rate": 0.3456
}
```

## Audit summary

### Causal-pair ranking accuracy (chance = 50%)

| Dimension | Accuracy |
|---|---:|
| emotion | 0.9381 |
| decision_score | 0.8300 |
| flip_score | 1.0000 |

### Spurious-pair invariance (lower = better)

| Dimension | Mean \|Δ\| |
|---|---:|
| decision_score | 0.009984 |
| price_strategy | 0.0204 |
| emotion | 0.047536 |
| progression | 0.017815 |
| flip_score | 0.032604 |

## Go/No-Go (manual judgment call — see causal_rm_architecture.md section 7)

- [ ] Causal ranking accuracy meaningfully above 50% for C1/C3/C5-covered dimensions
- [ ] Spurious invariance better (lower) than the heuristic baseline (compare against `causal_rm_audit_heuristic_val.json` in the repo root)
- [ ] Reward distribution not saturated at ±1 for every example

If all pass: proceed to `train_grpo_curriculum.py --causal_rm_checkpoint /home/paritosh/priyasmita/Qwen3-tts/v2/causal_rm_experiments/fixed_loss_lambda_inv_2.5_20260801_132941/checkpoint` for a small GRPO pilot. If not: iterate on lambda_bt/lambda_inv/lambda_reg or architecture before scaling up.