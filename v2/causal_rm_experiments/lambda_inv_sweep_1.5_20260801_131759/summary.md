# Causal RM Experiment — lambda_inv_sweep_1.5

Config: `/home/paritosh/priyasmita/Qwen3-tts/v2/causal_rm_experiments/lambda_inv_sweep_1.5_20260801_131759/config.json`
Checkpoint: `/home/paritosh/priyasmita/Qwen3-tts/v2/causal_rm_experiments/lambda_inv_sweep_1.5_20260801_131759/checkpoint`
Full audit report: `/home/paritosh/priyasmita/Qwen3-tts/v2/causal_rm_experiments/lambda_inv_sweep_1.5_20260801_131759/audit/report_causal_rm_val.md`

## Training (final logged step)

```json
{
  "epoch": 0,
  "step": 180,
  "loss": 0.15882915258407593,
  "bt_accuracy_per_dimension": {
    "decision_score": 0.8617,
    "emotion": 0.9747,
    "flip_score": 1.0
  },
  "invariance_mse_per_dimension": {
    "price_strategy": 0.001642,
    "emotion": 0.023497,
    "progression": 0.00161
  },
  "head_correlation": {
    "mean_abs_off_diagonal": 0.3471
  },
  "embedder_cache_hit_rate": 0.3456
}
```

## Audit summary

### Causal-pair ranking accuracy (chance = 50%)

| Dimension | Accuracy |
|---|---:|
| emotion | 1.0000 |
| decision_score | 0.8300 |
| flip_score | 1.0000 |

### Spurious-pair invariance (lower = better)

| Dimension | Mean \|Δ\| |
|---|---:|
| decision_score | 0.066702 |
| price_strategy | 0.016333 |
| emotion | 0.097357 |
| progression | 0.020235 |
| flip_score | 0.081122 |

## Go/No-Go (manual judgment call — see causal_rm_architecture.md section 7)

- [ ] Causal ranking accuracy meaningfully above 50% for C1/C3/C5-covered dimensions
- [ ] Spurious invariance better (lower) than the heuristic baseline (compare against `causal_rm_audit_heuristic_val.json` in the repo root)
- [ ] Reward distribution not saturated at ±1 for every example

If all pass: proceed to `train_grpo_curriculum.py --causal_rm_checkpoint /home/paritosh/priyasmita/Qwen3-tts/v2/causal_rm_experiments/lambda_inv_sweep_1.5_20260801_131759/checkpoint` for a small GRPO pilot. If not: iterate on lambda_bt/lambda_inv/lambda_reg or architecture before scaling up.