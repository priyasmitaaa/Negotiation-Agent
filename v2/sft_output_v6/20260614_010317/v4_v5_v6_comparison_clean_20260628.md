# SFT v4 vs v5 vs v6 — Clean v6 Speech Comparison

**Generated:** 2026-06-28  
**Base model:** `Qwen/Qwen2.5-Omni-7B` with LoRA  
**Judge:** `Qwen/Qwen3-8B`  
**Update:** v6 speech was regenerated cleanly on 2026-06-28 and this report uses the new speech inference artifacts.

---

## 0. Comparability Note

v4 and v5 use the same 881-dialogue eligible pool and the same 90-dialogue test split.

v6 uses the full 3154-dialogue emotion-annotated dataset and a different, larger test split.

| | v4 | v5 | v6 |
|---|---:|---:|---:|
| Test seller turns | 540 | 540 | 1908 |
| Dialogue pool | 881 | 881 | 3154 |
| Eligibility | emotion + reasoning | emotion + reasoning | emotion only |
| Target | decision + response | reasoning + decision + response | decision + response |

So v4 vs v5 is a strict shared-test comparison. v6 is the full-data baseline and is not perfectly apples-to-apples with v4/v5.

---

## 1. Decision Accuracy

| Mode | v4 | v5 | v6 clean |
|---|---:|---:|---:|
| Text inference | 96.9% (523/540) | 86.3% (466/540) | **98.3% (1876/1908)** |
| Speech inference | 96.9% (523/540) | 92.4% (499/540) | **98.3% (1876/1908)** |

The clean v6 speech regeneration preserves the same strong decision accuracy as the prior v6 text run.

---

## 2. Speech Confusion Matrix

| Model | LEVERAGE correct | LEVERAGE -> MITIGATE | MITIGATE -> LEVERAGE | MITIGATE correct | UNDECIDED correct | Total errors |
|---|---:|---:|---:|---:|---:|---:|
| v4 | 220 | 7 | 10 | 213 | 90 | 17 |
| v5 | 221 | 6 | 35 | 188 | 90 | 41 |
| v6 clean | 942 | 8 | 24 | 616 | 318 | **32** |

The remaining v6 error profile is still dominated by `MITIGATE -> LEVERAGE`, which should remain a specific GRPO penalty.

---

## 3. Zone and Flip Accuracy

| Zone | v4 speech | v5 speech | v6 clean speech |
|---|---:|---:|---:|
| Pre-crystallisation | 100.0% (90/90) | 100.0% (90/90) | 100.0% (318/318) |
| Post-crystallisation | 96.2% (433/450) | 90.9% (409/450) | **98.0% (1558/1590)** |

| | v4 | v5 | v6 clean |
|---|---:|---:|---:|
| Flip accuracy | 86.6% (58/67) | 92.5% (62/67) | **96.2% (177/184)** |

---

## 4. Judge Scores

| Dimension | v4 | v5 | v6 clean |
|---|---:|---:|---:|
| Overall | 4.03 | 3.88 | **4.12** |
| Emotion handling | 3.44 | 3.38 | **3.52** |
| Negotiation quality | 4.45 | 4.15 | **4.53** |
| Response naturalness | 4.34 | 4.25 | **4.41** |
| Progression | 3.88 | 3.76 | **4.04** |
| Avg correct decisions | 4.07 | 4.06 | **4.14** |
| Avg incorrect decisions | 2.90 | 2.79 | **2.91** |
| Pre-crystallisation avg | 3.18 | 3.19 | **3.26** |
| Post-crystallisation avg | 4.20 | 4.02 | **4.30** |

v6 remains strongest on every judge dimension after the clean speech regeneration.

---

## 5. Reward Audit

Clean v6 reward audit:

| Metric | Value |
|---|---:|
| Examples | 1908 |
| Avg total reward | 1.852 |
| Avg decision reward | 0.9644 |
| Avg emotion reward | 0.7623 |
| Avg progression reward | 0.8029 |
| Avg price strategy reward | 0.5967 |
| Avg audio quality reward | 0.9164 |

Audio quality breakdown:

| Audio reward | Count |
|---|---:|
| 1.0 | 1589 |
| 0.5 | 319 |

Automatic audio failures were duration-ratio only; no silence, clipping, abrupt-cutoff, or smoothness failures were flagged.

---

## 6. Format and Output Quality

| Metric | v4 | v5 speech | v6 clean speech |
|---|---:|---:|---:|
| Avg response length | 14.7 | 14.6 | 14.6 |
| Format OK | 100.0% | 55.6% | **100.0%** |
| Audio generated | 100.0% | 100.0% | 100.0% |

v6 keeps the deployment-style v4 target and avoids the v5 reasoning-format truncation issue.

---

## 7. Recommendation

Use **v6 clean speech** as the SFT baseline for GRPO.

Recommended GRPO setup:

- Start from `sft_output_v6/20260614_010317/final_adapter`.
- Use speech-conditioned prompts.
- Generate text policy outputs during GRPO: `<decision>` + `<response>`.
- Use three-level curriculum learning: easy, medium, hard.
- Evaluate checkpoints with full speech generation offline.

Primary reward pressure should target:

1. `MITIGATE -> LEVERAGE` errors.
2. post-crystallisation decisions.
3. decision flips.
4. emotion handling.
5. progression and repetition.

---

## 8. Artifact Index

| Artifact | Path |
|---|---|
| v4 speech inference | `sft_output_v4/20260525_014244/inference_speech_20260606_035020.json` |
| v4 judge | `sft_output_v4/20260525_014244/v4_judge_text_and_speech_20260611_220937.json` |
| v5 speech inference | `sft_output_v5/20260525_092022/inference_speech_20260613_030034.json` |
| v5 judge | `sft_output_v5/20260525_092022/v5_judge_text_and_speech_20260613_114947.json` |
| v6 text inference | `sft_output_v6/20260614_010317/inference_20260620_150244.json` |
| v6 clean speech inference | `sft_output_v6/20260614_010317/inference_speech_20260628_024050.json` |
| v6 clean speech WAVs | `sft_output_v6/20260614_010317/speech_outputs_20260628_024050/` |
| v6 clean error analysis | `sft_output_v6/20260614_010317/v6_error_analysis_text_and_speech_20260628_190512.md` |
| v6 clean reward audit | `sft_output_v6/20260614_010317/reward_audit_text_and_speech_clean_20260628.json` |
| v6 clean judge | `sft_output_v6/20260614_010317/v6_judge_text_and_speech_20260628_190515.json` |
| This report | `sft_output_v6/20260614_010317/v4_v5_v6_comparison_clean_20260628.md` |
