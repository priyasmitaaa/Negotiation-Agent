# Causal RM Experiment — v2_audio_fusion_prototype_seeded

Config: `/home/paritosh/priyasmita/Qwen3-tts/v2/causal_rm_experiments/v2_audio_fusion_prototype_seeded_20260816_135520/config.json`
Checkpoint: `/home/paritosh/priyasmita/Qwen3-tts/v2/causal_rm_experiments/v2_audio_fusion_prototype_seeded_20260816_135520/checkpoint`
Full audit report: `/home/paritosh/priyasmita/Qwen3-tts/v2/causal_rm_experiments/v2_audio_fusion_prototype_seeded_20260816_135520/audit/report_causal_rm_val.md`

## Training (final logged step)

```json
{
  "epoch": 0,
  "step": 160,
  "loss": 0.4859614074230194,
  "bt_accuracy_per_dimension": {
    "price_strategy": 0.8684,
    "emotion": 0.9487,
    "progression": 0.8718,
    "decision_score": 0.8158,
    "flip_score": 1.0
  },
  "invariance_mse_per_dimension": {
    "decision_score": 0.037654,
    "price_strategy": 0.01718,
    "emotion": 0.001263,
    "progression": 0.023201,
    "flip_score": 0.001364
  },
  "head_correlation": {
    "mean_abs_off_diagonal": 0.32
  },
  "embedder_cache_hit_rate": 0.2675,
  "ser_embedder_cache_hit_rate": 0.0
}
```

## Audit summary

### Causal-pair ranking accuracy (chance = 50%)

| Dimension | Accuracy |
|---|---:|
| price_strategy | 0.7857 |
| emotion | 0.9615 |
| decision_score | 0.8571 |
| progression | 0.9143 |
| flip_score | 1.0000 |

### Spurious-pair invariance (lower = better)

| Dimension | Mean \|Δ\| |
|---|---:|
| decision_score | 0.042821 |
| price_strategy | 0.065195 |
| emotion | 0.01996 |
| progression | 0.036966 |
| flip_score | 0.032152 |

## Go/No-Go (manual judgment call — see causal_rm_architecture.md section 7)

- [ ] Causal ranking accuracy meaningfully above 50% for C1/C3/C5-covered dimensions
- [ ] Spurious invariance better (lower) than the heuristic baseline (compare against `causal_rm_audit_heuristic_val.json` in the repo root)
- [ ] Reward distribution not saturated at ±1 for every example

If all pass: proceed to `train_grpo_curriculum.py --causal_rm_checkpoint /home/paritosh/priyasmita/Qwen3-tts/v2/causal_rm_experiments/v2_audio_fusion_prototype_seeded_20260816_135520/checkpoint` for a small GRPO pilot. If not: iterate on lambda_bt/lambda_inv/lambda_reg or architecture before scaling up.