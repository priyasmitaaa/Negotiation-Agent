# Causal RM Experiment — v1_text_only_control_seeded

Config: `/home/paritosh/priyasmita/Qwen3-tts/v2/causal_rm_experiments/v1_text_only_control_seeded_20260816_135744/config.json`
Checkpoint: `/home/paritosh/priyasmita/Qwen3-tts/v2/causal_rm_experiments/v1_text_only_control_seeded_20260816_135744/checkpoint`
Full audit report: `/home/paritosh/priyasmita/Qwen3-tts/v2/causal_rm_experiments/v1_text_only_control_seeded_20260816_135744/audit/report_causal_rm_val.md`

## Training (final logged step)

```json
{
  "epoch": 0,
  "step": 160,
  "loss": 0.6077171564102173,
  "bt_accuracy_per_dimension": {
    "price_strategy": 0.8158,
    "emotion": 0.9487,
    "progression": 0.8462,
    "decision_score": 0.8158,
    "flip_score": 1.0
  },
  "invariance_mse_per_dimension": {
    "decision_score": 0.021344,
    "price_strategy": 0.011766,
    "emotion": 0.015405,
    "progression": 0.018369,
    "flip_score": 0.002063
  },
  "head_correlation": {
    "mean_abs_off_diagonal": 0.7262
  },
  "embedder_cache_hit_rate": 0.2675
}
```

## Audit summary

### Causal-pair ranking accuracy (chance = 50%)

| Dimension | Accuracy |
|---|---:|
| price_strategy | 0.8333 |
| emotion | 0.9615 |
| decision_score | 0.8571 |
| progression | 0.9429 |
| flip_score | 1.0000 |

### Spurious-pair invariance (lower = better)

| Dimension | Mean \|Δ\| |
|---|---:|
| decision_score | 0.036942 |
| price_strategy | 0.049158 |
| emotion | 0.075079 |
| progression | 0.062144 |
| flip_score | 0.029939 |

## Go/No-Go (manual judgment call — see causal_rm_architecture.md section 7)

- [ ] Causal ranking accuracy meaningfully above 50% for C1/C3/C5-covered dimensions
- [ ] Spurious invariance better (lower) than the heuristic baseline (compare against `causal_rm_audit_heuristic_val.json` in the repo root)
- [ ] Reward distribution not saturated at ±1 for every example

If all pass: proceed to `train_grpo_curriculum.py --causal_rm_checkpoint /home/paritosh/priyasmita/Qwen3-tts/v2/causal_rm_experiments/v1_text_only_control_seeded_20260816_135744/checkpoint` for a small GRPO pilot. If not: iterate on lambda_bt/lambda_inv/lambda_reg or architecture before scaling up.