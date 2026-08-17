# Causal RM Experiment — v2_audio_fusion_full

Config: `/home/paritosh/priyasmita/Qwen3-tts/v2/causal_rm_experiments/v2_audio_fusion_full_20260816_152709/config.json`
Checkpoint: `/home/paritosh/priyasmita/Qwen3-tts/v2/causal_rm_experiments/v2_audio_fusion_full_20260816_152709/checkpoint`
Full audit report: `/home/paritosh/priyasmita/Qwen3-tts/v2/causal_rm_experiments/v2_audio_fusion_full_20260816_152709/audit/report_causal_rm_val.md`

## Training (final logged step)

```json
{
  "epoch": 0,
  "step": 59300,
  "loss": 0.16534797847270966,
  "bt_accuracy_per_dimension": {
    "progression": 0.8855,
    "decision_score": 0.9775,
    "emotion": 0.9973,
    "price_strategy": 0.9188,
    "flip_score": 1.0
  },
  "invariance_mse_per_dimension": {
    "decision_score": 0.00145,
    "price_strategy": 0.001113,
    "emotion": 4.9e-05,
    "progression": 0.001457,
    "flip_score": 0.0006
  },
  "head_correlation": {
    "mean_abs_off_diagonal": 0.4638
  },
  "embedder_cache_hit_rate": 0.3963,
  "ser_embedder_cache_hit_rate": 0.4532
}
```

## Audit summary

### Causal-pair ranking accuracy (chance = 50%)

| Dimension | Accuracy |
|---|---:|
| price_strategy | 0.9485 |
| emotion | 1.0000 |
| decision_score | 0.9936 |
| progression | 0.9068 |
| flip_score | 1.0000 |

### Spurious-pair invariance (lower = better)

| Dimension | Mean \|Δ\| |
|---|---:|
| decision_score | 0.002027 |
| price_strategy | 0.006508 |
| emotion | 0.00252 |
| progression | 0.027784 |
| flip_score | 0.014972 |

## Go/No-Go (manual judgment call — see causal_rm_architecture.md section 7)

- [ ] Causal ranking accuracy meaningfully above 50% for C1/C3/C5-covered dimensions
- [ ] Spurious invariance better (lower) than the heuristic baseline (compare against `causal_rm_audit_heuristic_val.json` in the repo root)
- [ ] Reward distribution not saturated at ±1 for every example

If all pass: proceed to `train_grpo_curriculum.py --causal_rm_checkpoint /home/paritosh/priyasmita/Qwen3-tts/v2/causal_rm_experiments/v2_audio_fusion_full_20260816_152709/checkpoint` for a small GRPO pilot. If not: iterate on lambda_bt/lambda_inv/lambda_reg or architecture before scaling up.