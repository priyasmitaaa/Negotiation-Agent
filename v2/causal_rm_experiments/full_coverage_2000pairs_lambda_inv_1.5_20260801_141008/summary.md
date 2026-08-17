# Causal RM Experiment — full_coverage_2000pairs_lambda_inv_1.5

Config: `/home/paritosh/priyasmita/Qwen3-tts/v2/causal_rm_experiments/full_coverage_2000pairs_lambda_inv_1.5_20260801_141008/config.json`
Checkpoint: `/home/paritosh/priyasmita/Qwen3-tts/v2/causal_rm_experiments/full_coverage_2000pairs_lambda_inv_1.5_20260801_141008/checkpoint`
Full audit report: `/home/paritosh/priyasmita/Qwen3-tts/v2/causal_rm_experiments/full_coverage_2000pairs_lambda_inv_1.5_20260801_141008/audit/report_causal_rm_val.md`

## Training (final logged step)

```json
{
  "epoch": 0,
  "step": 1160,
  "loss": 0.19194820523262024,
  "bt_accuracy_per_dimension": {
    "emotion": 0.9677,
    "progression": 0.8228,
    "flip_score": 1.0,
    "price_strategy": 0.8576,
    "decision_score": 0.8396
  },
  "invariance_mse_per_dimension": {
    "decision_score": 0.012775,
    "price_strategy": 0.016677,
    "emotion": 0.009868,
    "progression": 0.010642,
    "flip_score": 0.001016
  },
  "head_correlation": {
    "mean_abs_off_diagonal": 0.4828
  },
  "embedder_cache_hit_rate": 0.3147
}
```

## Audit summary

### Causal-pair ranking accuracy (chance = 50%)

| Dimension | Accuracy |
|---|---:|
| price_strategy | 0.8676 |
| emotion | 0.9866 |
| decision_score | 0.8280 |
| progression | 0.8475 |
| flip_score | 1.0000 |

### Spurious-pair invariance (lower = better)

| Dimension | Mean \|Δ\| |
|---|---:|
| decision_score | 0.02647 |
| price_strategy | 0.018625 |
| emotion | 0.02103 |
| progression | 0.023863 |
| flip_score | 0.021225 |

## Go/No-Go (manual judgment call — see causal_rm_architecture.md section 7)

- [ ] Causal ranking accuracy meaningfully above 50% for C1/C3/C5-covered dimensions
- [ ] Spurious invariance better (lower) than the heuristic baseline (compare against `causal_rm_audit_heuristic_val.json` in the repo root)
- [ ] Reward distribution not saturated at ±1 for every example

If all pass: proceed to `train_grpo_curriculum.py --causal_rm_checkpoint /home/paritosh/priyasmita/Qwen3-tts/v2/causal_rm_experiments/full_coverage_2000pairs_lambda_inv_1.5_20260801_141008/checkpoint` for a small GRPO pilot. If not: iterate on lambda_bt/lambda_inv/lambda_reg or architecture before scaling up.