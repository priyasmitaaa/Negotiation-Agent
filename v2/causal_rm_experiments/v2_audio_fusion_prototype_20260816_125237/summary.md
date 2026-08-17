# Causal RM Experiment — v2_audio_fusion_prototype

Config: `/home/paritosh/priyasmita/Qwen3-tts/v2/causal_rm_experiments/v2_audio_fusion_prototype_20260816_125237/config.json`
Checkpoint: `/home/paritosh/priyasmita/Qwen3-tts/v2/causal_rm_experiments/v2_audio_fusion_prototype_20260816_125237/checkpoint`
Full audit report: `/home/paritosh/priyasmita/Qwen3-tts/v2/causal_rm_experiments/v2_audio_fusion_prototype_20260816_125237/audit/report_causal_rm_val.md`

## Training (final logged step)

```json
{
  "epoch": 0,
  "step": 160,
  "loss": 0.6958970427513123,
  "bt_accuracy_per_dimension": {
    "price_strategy": 0.8421,
    "emotion": 0.9231,
    "progression": 0.7436,
    "decision_score": 0.8421,
    "flip_score": 1.0
  },
  "invariance_mse_per_dimension": {
    "decision_score": 0.025988,
    "price_strategy": 0.011988,
    "emotion": 0.000992,
    "progression": 0.018684,
    "flip_score": 0.0017
  },
  "head_correlation": {
    "mean_abs_off_diagonal": 0.5532
  },
  "embedder_cache_hit_rate": 0.2675,
  "ser_embedder_cache_hit_rate": 0.0
}
```

## Audit summary

### Causal-pair ranking accuracy (chance = 50%)

| Dimension | Accuracy |
|---|---:|
| price_strategy | 0.8571 |
| emotion | 0.9615 |
| decision_score | 0.8571 |
| progression | 0.8857 |
| flip_score | 1.0000 |

### Spurious-pair invariance (lower = better)

| Dimension | Mean \|Δ\| |
|---|---:|
| decision_score | 0.006545 |
| price_strategy | 0.009991 |
| emotion | 0.019181 |
| progression | 0.00347 |
| flip_score | 0.024562 |

## Go/No-Go (manual judgment call — see causal_rm_architecture.md section 7)

- [ ] Causal ranking accuracy meaningfully above 50% for C1/C3/C5-covered dimensions
- [ ] Spurious invariance better (lower) than the heuristic baseline (compare against `causal_rm_audit_heuristic_val.json` in the repo root)
- [ ] Reward distribution not saturated at ±1 for every example

If all pass: proceed to `train_grpo_curriculum.py --causal_rm_checkpoint /home/paritosh/priyasmita/Qwen3-tts/v2/causal_rm_experiments/v2_audio_fusion_prototype_20260816_125237/checkpoint` for a small GRPO pilot. If not: iterate on lambda_bt/lambda_inv/lambda_reg or architecture before scaling up.