# SFT v4 vs v5 vs v6 — Full Comparison Report

**Generated:** 2026-06-20  
**Base model:** `Qwen/Qwen2.5-Omni-7B` with LoRA (r=8, alpha=16, 3 epochs)  
**Judge:** `Qwen/Qwen3-8B` (LLM-as-judge, 5-point scale)  
**Prior report:** `sft_output_v5/20260525_092022/v4_vs_v5_comparison_20260613.md`

---

## 0. Important Comparability Note

v4 and v5 use the same 881-dialogue eligible pool and the same 90-dialogue test split:

- v4/v5 test examples: **540** seller turns
- v4/v5 eligibility: emotion + reasoning annotations

v6 uses the full emotion-annotated dataset and a new split:

- v6 test examples: **1908** seller turns
- v6 eligibility: emotion only, no reasoning filter
- v6 train examples: **15132** vs **4224** for v4/v5

So v4 vs v5 is a strict same-test comparison. v6 is a larger-data/new-test comparison; its numbers are still useful, but not perfectly apples-to-apples.

---

## 1. Training Target and Dataset

| | v4 | v5 | v6 |
|---|---|---|---|
| Target format | `<decision>` + `<response>` | `<reasoning>` + `<decision>` + `<response>` | `<decision>` + `<response>` |
| Reasoning in target | No | Yes | No |
| Dialogue pool | 881 | 881 | 3154 |
| Train examples | 4224 | 4224 | 15132 |
| Test examples | 540 | 540 | 1908 |
| Best eval loss | 0.4100 | 0.8095 | **0.3676** |
| Test loss | 0.4237 | 0.8241 | **0.3716** |
| Special note | Deployment-style target | Interpretability / GRPO init | Full-data no-reasoning SFT, text+speech mix |

---

## 2. Decision Accuracy

| Mode | v4 | v5 | v6 |
|---|---:|---:|---:|
| Text-only inference | 96.9% (523/540) | 86.3% (466/540) | **98.3% (1876/1908)** |
| Speech inference | 96.9% (523/540) | 92.4% (499/540) | **98.4% (1877/1908)** |

v6 is strongest on both modes, with the caveat that it is evaluated on a new larger test split.

### Speech Confusion Matrix

| Model | LEVERAGE correct | LEVERAGE -> MITIGATE | MITIGATE -> LEVERAGE | MITIGATE correct | UNDECIDED correct | Total errors |
|---|---:|---:|---:|---:|---:|---:|
| v4 | 220 | 7 | 10 | 213 | 90 | 17 |
| v5 | 221 | 6 | 35 | 188 | 90 | 41 |
| v6 | 943 | 7 | 24 | 616 | 318 | **31** |

v5's main regression was over-predicting `LEVERAGE` on `MITIGATE` cases. v6 still has the same dominant error direction, but at much lower rate: 24 errors over 1908 examples.

---

## 3. Zone Accuracy

| Zone | v4 speech | v5 speech | v6 speech |
|---|---:|---:|---:|
| Pre-crystallisation | 100.0% (90/90) | 100.0% (90/90) | 100.0% (318/318) |
| Post-crystallisation | 96.2% (433/450) | 90.9% (409/450) | **98.1% (1559/1590)** |

All versions are perfect on early/pre-crystallisation examples. v6 improves the hard post-crystallisation region substantially.

---

## 4. Decision Flip Accuracy

| | v4 | v5 | v6 |
|---|---:|---:|---:|
| Flip accuracy | 86.6% (58/67) | 92.5% (62/67) | **96.2% (177/184)** |

v5 improved flip handling over v4, likely because explicit reasoning helped identify state changes. v6 surpasses both, suggesting that full-data coverage helps even without reasoning in the target.

---

## 5. Judge Scores

| Dimension | v4 | v5 | v6 |
|---|---:|---:|---:|
| Overall | 4.03 | 3.88 | **4.12** |
| Emotion handling | 3.44 | 3.38 | **3.52** |
| Negotiation quality | 4.45 | 4.15 | **4.53** |
| Response naturalness | 4.34 | 4.25 | **4.41** |
| Progression | 3.88 | 3.76 | **4.04** |
| Avg — correct decisions | 4.07 | 4.06 | **4.14** |
| Avg — incorrect decisions | 2.90 | 2.79 | **2.91** |
| Pre-crystallisation avg | 3.18 | 3.19 | **3.26** |
| Post-crystallisation avg | 4.20 | 4.02 | **4.30** |

v6 leads on every judge dimension. The biggest useful gains are in negotiation quality and progression, which were exactly where v5 lagged after adding reasoning.

---

## 6. Response and Format Quality

| Metric | v4 | v5 text | v5 speech | v6 text | v6 speech |
|---|---:|---:|---:|---:|---:|
| Avg response length | 14.7 | 14.5 | 14.6 | 14.6 | 14.6 |
| Format OK | 100.0% | 100.0% | 55.6% | 100.0% | 100.0% |
| Reasoning present | - | 100.0% | 55.6% | - | - |
| Avg reasoning score | - | 0.945 | 0.522 | - | - |
| Audio generated | 100.0% | - | 100.0% | - | 100.0% |

v6 keeps the clean v4-style output format and avoids the v5 speech truncation issue, where reasoning tags were missing in 44.4% of speech-mode generations.

---

## 7. Error Profile

| Error category | v4 | v5 | v6 |
|---|---:|---:|---:|
| Total examples | 540 | 540 | 1908 |
| Total errors | 17 | 74 text / 41 speech | 32 text / 31 speech |
| Dominant error | MITIGATE -> LEVERAGE | MITIGATE -> LEVERAGE | MITIGATE -> LEVERAGE |
| Post-crystallisation errors | 17/450 | 74/450 text, 41/450 speech | 32/1590 text, 31/1590 speech |
| Flip errors | 9/67 | 10/67 text, 5/67 speech | 7/184 |

v6's remaining weakness is still over-leveraging in a small number of `MITIGATE` cases. For reward design, `MITIGATE -> LEVERAGE` should remain a specific penalty.

---

## 8. Summary

| Aspect | Winner | Notes |
|---|---|---|
| Strict v4/v5 decision accuracy | v4 | v4 beats v5 on the shared split. |
| Full-data decision accuracy | v6 | 98.4% speech accuracy on 1908 examples. |
| Judge quality | v6 | Best overall and every subdimension. |
| Flip handling | v6 | 96.2% on 184 flip turns. |
| Format robustness | v4/v6 | v5 speech has reasoning truncation. |
| Interpretability | v5 | Only version with explicit reasoning target. |
| Deployment readiness | v6 | Best quality with simple `<decision>` + `<response>` target. |

### Recommendation

Use **v6** as the deployment-style SFT baseline and likely the best starting point for response-quality work. It preserves the clean v4 target format while benefiting from the full 3154-dialogue dataset.

For GRPO, there are two viable paths:

1. Start from **v6** for best decision quality and natural responses, then add rewards for the remaining `MITIGATE -> LEVERAGE` errors and flip robustness.
2. Start from **v5** only if explicit reasoning traces are required as reward inputs, but fix `max_new_tokens` and the speech truncation problem first.

Given the v6 judge and accuracy gains, the practical recommendation is:

- **Primary baseline:** v6 final adapter
- **Reward focus:** post-crystallisation decisions, flip turns, and `MITIGATE -> LEVERAGE` penalties
- **Keep output format:** no reasoning for deployment; optional private reasoning only if GRPO specifically needs it

---

## 9. File Index

| File | Location |
|---|---|
| Existing v4/v5 comparison | `sft_output_v5/20260525_092022/v4_vs_v5_comparison_20260613.md` |
| v4 text inference | `sft_output_v4/20260525_014244/inference_20260604_225504.json` |
| v4 speech inference | `sft_output_v4/20260525_014244/inference_speech_20260606_035020.json` |
| v4 judge | `sft_output_v4/20260525_014244/v4_judge_text_and_speech_20260611_220937.json` |
| v4 error analysis | `sft_output_v4/20260525_014244/v4_error_analysis_text_and_speech_20260611_220737.md` |
| v5 text inference | `sft_output_v5/20260525_092022/inference_20260605_194041.json` |
| v5 speech inference | `sft_output_v5/20260525_092022/inference_speech_20260613_030034.json` |
| v5 judge | `sft_output_v5/20260525_092022/v5_judge_text_and_speech_20260613_114947.json` |
| v5 error analysis | `sft_output_v5/20260525_092022/v5_error_analysis_text_and_speech_20260613_114905.md` |
| v6 text inference | `sft_output_v6/20260614_010317/inference_20260620_150244.json` |
| v6 speech inference | `sft_output_v6/20260614_010317/inference_speech_20260615_102532.json` |
| v6 judge | `sft_output_v6/20260614_010317/v6_judge_text_and_speech_20260620_161122.json` |
| v6 error analysis | `sft_output_v6/20260614_010317/v6_error_analysis_text_and_speech_20260620_161120.md` |
| This report | `sft_output_v6/20260614_010317/v4_v5_v6_comparison_20260620.md` |
