# Causal RM Experiment — scaleup_2000pairs_lambda_inv_1.5

Config: `/home/paritosh/priyasmita/Qwen3-tts/v2/causal_rm_experiments/scaleup_2000pairs_lambda_inv_1.5_20260801_133731/config.json`
Checkpoint: `/home/paritosh/priyasmita/Qwen3-tts/v2/causal_rm_experiments/scaleup_2000pairs_lambda_inv_1.5_20260801_133731/checkpoint`
Full audit report: `/home/paritosh/priyasmita/Qwen3-tts/v2/causal_rm_experiments/scaleup_2000pairs_lambda_inv_1.5_20260801_133731/audit/report_causal_rm_val.md`

## Training (final logged step)

```json
{
  "epoch": 0,
  "step": 1160,
  "loss": 0.4514448642730713,
  "bt_accuracy_per_dimension": {
    "emotion": 0.9879,
    "decision_score": 0.8688,
    "flip_score": 1.0
  },
  "invariance_mse_per_dimension": {
    "decision_score": 0.018939,
    "price_strategy": 0.000705,
    "emotion": 0.021282,
    "progression": 0.000943,
    "flip_score": 0.001937
  },
  "head_correlation": {
    "mean_abs_off_diagonal": 0.4545
  },
  "embedder_cache_hit_rate": 0.3153
}
```

## Audit summary

### Causal-pair ranking accuracy (chance = 50%)

| Dimension | Accuracy |
|---|---:|
| emotion | 0.9895 |
| decision_score | 0.8650 |
| flip_score | 1.0000 |

### Spurious-pair invariance (lower = better)

| Dimension | Mean \|Δ\| |
|---|---:|
| decision_score | 0.012647 |
| price_strategy | 0.018915 |
| emotion | 0.001644 |
| progression | 0.023669 |
| flip_score | 0.055814 |

## Go/No-Go (manual judgment call — see causal_rm_architecture.md section 7)

- [ ] Causal ranking accuracy meaningfully above 50% for C1/C3/C5-covered dimensions
- [ ] Spurious invariance better (lower) than the heuristic baseline (compare against `causal_rm_audit_heuristic_val.json` in the repo root)
- [ ] Reward distribution not saturated at ±1 for every example

If all pass: proceed to `train_grpo_curriculum.py --causal_rm_checkpoint /home/paritosh/priyasmita/Qwen3-tts/v2/causal_rm_experiments/scaleup_2000pairs_lambda_inv_1.5_20260801_133731/checkpoint` for a small GRPO pilot. If not: iterate on lambda_bt/lambda_inv/lambda_reg or architecture before scaling up.