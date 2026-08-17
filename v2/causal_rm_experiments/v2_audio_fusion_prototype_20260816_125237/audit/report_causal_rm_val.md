# Causal RM Audit Report — causal_rm_val

Backend: `causal_rm`  |  Split: `val`  |  Checkpoint: `/home/paritosh/priyasmita/Qwen3-tts/v2/causal_rm_experiments/v2_audio_fusion_prototype_20260816_125237/checkpoint`

## Causal-pair ranking accuracy (chance = 50%)

| Dimension | Accuracy | N pairs |
|---|---:|---:|
| price_strategy | 0.8571 | 42 |
| emotion | 0.9615 | 52 |
| decision_score | 0.8571 | 42 |
| progression | 0.8857 | 35 |
| flip_score | 1.0000 | 2 |

## Spurious-pair invariance (lower = better)

| Dimension | Mean \|Δ\| | N pairs |
|---|---:|---:|
| decision_score | 0.006545 | 127 |
| price_strategy | 0.009991 | 127 |
| emotion | 0.019181 | 127 |
| progression | 0.003470 | 127 |
| flip_score | 0.024562 | 127 |

![causal ranking accuracy](causal_ranking_accuracy_causal_rm_val.png)
![spurious invariance](spurious_invariance_causal_rm_val.png)