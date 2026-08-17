# Causal RM Audit Report — causal_rm_val

Backend: `causal_rm`  |  Split: `val`  |  Checkpoint: `/home/paritosh/priyasmita/Qwen3-tts/v2/causal_rm_experiments/lambda_inv_sweep_3.5_20260801_132137/checkpoint`

## Causal-pair ranking accuracy (chance = 50%)

| Dimension | Accuracy | N pairs |
|---|---:|---:|
| emotion | 0.9691 | 97 |
| decision_score | 0.8400 | 100 |
| flip_score | 1.0000 | 10 |

## Spurious-pair invariance (lower = better)

| Dimension | Mean \|Δ\| | N pairs |
|---|---:|---:|
| decision_score | 0.287380 | 293 |
| price_strategy | 0.012991 | 293 |
| emotion | 0.001061 | 293 |
| progression | 0.013834 | 293 |
| flip_score | 0.048471 | 293 |

![causal ranking accuracy](causal_ranking_accuracy_causal_rm_val.png)
![spurious invariance](spurious_invariance_causal_rm_val.png)