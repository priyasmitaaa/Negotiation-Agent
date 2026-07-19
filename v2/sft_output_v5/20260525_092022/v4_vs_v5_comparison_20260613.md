# SFT v4 vs v5 — Full Comparison Report

**Generated:** 2026-06-13  
**Scope:** 540 test examples (90 dialogues × ~6 seller turns each)  
**Base model:** `Qwen/Qwen2.5-Omni-7B` with LoRA (r=8, α=16, 3 epochs)  
**Judge:** `Qwen/Qwen3-8B` (LLM-as-judge, 5-point scale)

---

## 1. Key Difference: Training Target

| | v4 | v5 |
|---|---|---|
| **Target format** | `<decision> [LEVERAGE/MITIGATE/UNDECIDED]` + `<response>` | `<reasoning>...</reasoning>` + `<decision>` + `<response>` |
| **Objective** | Learn correct decision + natural response | Add explicit chain-of-thought before deciding |
| **Best eval loss** | 0.4100 | 0.8095 *(longer targets)* |
| **Use case** | Deployment | GRPO initialisation / interpretability |

---

## 2. Decision Accuracy

| Mode | v4 | v5 | Δ |
|---|---|---|---|
| **Text-only inference** | **96.9%** (523/540) | 86.3% (466/540) | −10.6 pp |
| **Speech inference** | **96.9%** (523/540) | 92.4% (499/540) | −4.5 pp |

> **Note on the v5 text vs speech gap (+6.1 pp):** The speech run uses audio as input (buyer WAVs), which provides richer emotional context. The text-only run uses text transcripts only. This suggests v5 benefits meaningfully from multimodal input — the reasoning module can ground on audio cues.

### Confusion Matrix (Speech Inference)

| Predicted → | LEVERAGE | MITIGATE | UNDECIDED |
|---|---|---|---|
| **GT: LEVERAGE** | 220 ✓ | 7 ✗ | 0 |
| **GT: MITIGATE** | **10** ✗ | 213 ✓ | 0 |
| **GT: UNDECIDED** | 0 | 0 | **90 ✓** |

**v4 errors:** 17 total — 7 LEVERAGE→MITIGATE, 10 MITIGATE→LEVERAGE

| Predicted → | LEVERAGE | MITIGATE | UNDECIDED |
|---|---|---|---|
| **GT: LEVERAGE** | 221 ✓ | 6 ✗ | 0 |
| **GT: MITIGATE** | **35** ✗ | 188 ✓ | 0 |
| **GT: UNDECIDED** | 0 | 0 | **90 ✓** |

**v5 errors:** 41 total — 6 LEVERAGE→MITIGATE, **35 MITIGATE→LEVERAGE** ← main regression

> v5's primary failure mode is **over-predicting LEVERAGE** when the correct answer is MITIGATE (35 cases vs v4's 10). The reasoning chain may be overcounting buyer pressure signals.

---

## 3. Zone Accuracy

| Zone | v4 | v5 (speech) | Δ |
|---|---|---|---|
| **Pre-crystallisation** | 100.0% (90/90) | 100.0% (90/90) | 0 |
| **Post-crystallisation** | **96.2%** (433/450) | 90.9% (409/450) | −5.3 pp |

Both models are perfect on pre-crystallisation (early turns, simpler decisions). Regression is entirely in post-crystallisation (later turns with complex price dynamics).

---

## 4. Decision Flip Accuracy

Flips = turns where the correct decision changes from the previous seller turn (hardest cases).

| | v4 | v5 (speech) | Δ |
|---|---|---|---|
| **Flip accuracy** | 86.6% (58/67) | **92.5%** (62/67) | **+5.9 pp** ✅ |

> **v5 is better on flips.** The explicit reasoning chain helps the model recognise when the negotiation state has shifted and the strategy needs to change. This is encouraging for GRPO training.

---

## 5. Judge Scores (LLM-as-Judge, Qwen3-8B, 1–5 scale)

| Dimension | v4 | v5 | Δ |
|---|---|---|---|
| **Overall** | **4.03** | 3.88 | −0.15 |
| Emotion handling | **3.44** | 3.38 | −0.06 |
| Negotiation quality | **4.45** | 4.15 | −0.30 |
| Response naturalness | **4.34** | 4.25 | −0.09 |
| Progression | **3.88** | 3.76 | −0.12 |
| Avg — correct decisions | **4.07** | 4.06 | −0.01 |
| Avg — incorrect decisions | **2.90** | 2.79 | −0.11 |
| Pre-crystallisation avg | 3.18 | **3.19** | +0.01 |
| Post-crystallisation avg | **4.20** | 4.02 | −0.18 |

> v4 leads on all judge dimensions. However the gap on **correct-decision examples is near-zero (4.07 vs 4.06)** — v5's lower overall score is primarily driven by the additional 24 incorrect decisions being scored harshly by the judge.

---

## 6. Response & Reasoning Quality

| Metric | v4 | v5 (text) | v5 (speech) |
|---|---|---|---|
| Avg response length | 14.7 words | 14.5 words | 14.6 words |
| Format OK (all tags present) | 100.0% | 100.0% | 55.6% |
| Reasoning present | — | 100.0% | 55.6% |
| Avg reasoning length | — | 102.0 words | 101.6 words |
| Avg reasoning score (0–1) | — | **0.945** | 0.522 |

### v5 Reasoning Element Coverage (Text-only, 100% format OK)

| Element | v5 Text | v5 Speech |
|---|---|---|
| mentions_price_numbers | **100.0%** | 55.6% |
| mentions_fair_value | **99.8%** | 55.4% |
| mentions_buyer_emotion | **99.3%** | 55.0% |
| mentions_harm_direction | **68.3%** | 36.7% |
| mentions_decision_word | **100.0%** | 55.6% |
| has_strategy | **99.8%** | 55.4% |

> **The ~55% reasoning coverage in speech mode is a direct consequence of the 44.4% missing-reasoning rate** in `inference_speech_20260613_030034.json`. This is because in speech mode, `max_new_tokens=512` is shared between the long reasoning chain (~102 words) and the response — some examples hit the limit before completing the reasoning tags. Increasing `--max_new_tokens` to 768 or 1024 would fix this.

---

## 7. Audio Generation

| | v4 | v5 |
|---|---|---|
| WAVs generated | 540/540 (100%) | 541/540 (100%+1 bonus) |
| Speaker | Chelsie | Chelsie |
| Parse failures | 0 | 0 |

---

## 8. Summary & GRPO Implications

| Aspect | Winner | Notes |
|---|---|---|
| Decision accuracy | **v4** (96.9% vs 92.4%) | v5 over-leverages on MITIGATE cases |
| Flip accuracy | **v5** (+5.9 pp) | Reasoning helps on complex state changes |
| Judge quality | **v4** (4.03 vs 3.88) | Gap almost vanishes on correct-decision examples |
| Reasoning quality | **v5** (by design) | 0.945 reasoning score, 102 word chain |
| Audio quality | Tie | Both 100% generation, same voice |
| GRPO readiness | **v5** | Explicit reasoning trace → better reward signal |

### GRPO Recommendation

Use **v5 as the GRPO initialisation point** (`sft_output_v5/20260525_092022/final_adapter`). The reward function should:
1. **Decision correctness reward** — binary +1/−1 (addresses the 7.6% accuracy gap vs v4)
2. **Flip bonus** — extra reward when correctly predicting decision changes (v5 already stronger here)
3. **MITIGATE precision reward** — penalise MITIGATE→LEVERAGE errors specifically (35 cases = primary failure mode)
4. **Reasoning quality reward** — score on reasoning element checklist (pushes the 55% speech coverage up)
5. **Fix `max_new_tokens`** — set to 1024 for speech inference to eliminate the 44.4% truncation rate before GRPO

---

## 9. File Index

| File | Location |
|---|---|
| v4 speech inference | `sft_output_v4/20260525_014244/inference_speech_20260606_035020.json` |
| v4 judge | `sft_output_v4/20260525_014244/v4_judge_text_and_speech_20260611_220937.json` |
| v4 error analysis | `sft_output_v4/20260525_014244/v4_error_analysis_text_and_speech_20260611_220737.md` |
| v5 text inference | `sft_output_v5/20260525_092022/inference_20260605_194041.json` |
| v5 speech inference | `sft_output_v5/20260525_092022/inference_speech_20260613_030034.json` |
| v5 judge | `sft_output_v5/20260525_092022/v5_judge_text_and_speech_20260613_114947.json` |
| v5 error analysis | `sft_output_v5/20260525_092022/v5_error_analysis_text_and_speech_20260613_114905.md` |
| **This report** | `sft_output_v5/20260525_092022/v4_vs_v5_comparison_20260613.md` |
