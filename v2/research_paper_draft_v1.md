# Toward Emotion-Aware Multimodal Negotiation Agents

First draft research write-up for the Qwen3-tts/v2 project.

This document summarizes the project from dataset construction through supervised fine-tuning, text and speech inference, error analysis, LLM-as-a-judge evaluation, and the current reward-function design for reinforcement learning. It is intended as a detailed internal first draft for a research paper. It should be refined later for claims, citations, notation, ablations, and final experimental framing.

## Abstract

This project studies multimodal negotiation agents that respond to a buyer in a second-hand electronics bargaining scenario using both textual dialogue context and speech-derived affective signals. The core problem is to train a seller-side assistant that can decide whether to LEVERAGE or MITIGATE the buyer's emotional state and then generate a short, natural bargaining response that advances the negotiation.

We constructed a 3,154-dialogue dataset of multi-turn buyer-seller negotiations. Each dialogue contains 12 turns, with alternating buyer and seller utterances, price offers, negotiation state metadata, and affect annotations. The full dataset contains 37,848 utterance-level speech files and 18,924 seller decision targets. Earlier supervised fine-tuning variants used only the 881 dialogues for which all seller turns also had explicit reasoning annotations. The final v6 supervised fine-tuning setup removes the reasoning requirement and trains on the full 3,154-dialogue dataset, using a target format containing only `<decision>` and `<response>`.

The project evaluates three major SFT variants. v4 trains the model without reasoning on the smaller 881-dialogue reasoning-complete subset. v5 trains on the same subset but includes explicit reasoning in the output target. v6 trains without reasoning on the full dataset. All variants use Qwen/Qwen2.5-Omni-7B in a thinker-only LoRA fine-tuning setup, with audio encoder parameters frozen and lightweight adaptation applied to language-model projection layers. v6 reaches 98.3% text decision accuracy and 98.4% speech decision accuracy on its test split, with an LLM-as-a-judge overall score of 4.12/5. The largest remaining weaknesses are not raw decision classification but emotion handling, progression, price-strategy nuance, repetition, and turn-level adaptation near decision flips.

Based on the completed error analysis and judge outputs, the next phase is reinforcement learning. The proposed reward function combines a hard decision reward with softer judge-derived and heuristic components for emotion handling, progression, price strategy, output format, naturalness, repetition avoidance, and language purity. The goal is not to introduce unnecessarily complex mathematical machinery, but to construct a bounded, auditable, multi-objective reward that directly attacks the observed post-SFT failure modes.

## 1. Research Motivation

Negotiation is not only a task of producing a correct factual answer. A good negotiator must interpret the other party's emotional state, track offer history, understand anchoring pressure, and decide when to use the buyer's affect as leverage versus when to de-escalate or accommodate it. In spoken negotiation, this becomes harder because the same text can imply different strategic opportunities depending on voice tone, confidence, frustration, or hesitation.

The project therefore focuses on an emotion-aware multimodal seller agent. Given a negotiation history and the current buyer utterance, the model must produce:

1. A strategic decision: `LEVERAGE` or `MITIGATE`.
2. A natural seller response that reflects the chosen strategy.

The decision target has a behavioral interpretation:

- `LEVERAGE`: the seller can take advantage of the buyer's emotional or strategic position, for example by holding firm, emphasizing scarcity, using anchoring, or extracting a better price.
- `MITIGATE`: the seller should reduce tension, preserve rapport, soften the offer, or respond in a way that accommodates buyer hesitation, frustration, or dissatisfaction.

This is a useful research setting because it sits between classification, dialogue modeling, speech understanding, affect recognition, and strategic generation. The model is not merely asked to identify emotion; it must use emotion as part of a negotiation policy.

## 2. Task Definition

For each seller turn, the input consists of:

- Dialogue history up to the current seller response.
- The current buyer utterance.
- Negotiation state such as buyer offers so far, anchor price, current zone, and decision flip indicators.
- Buyer emotion metadata derived from speech.
- In speech inference/training modes, the buyer audio waveform.

The target for the final v6 model is:

```text
<decision>LEVERAGE|MITIGATE</decision>
<response>seller response text</response>
```

The target deliberately excludes chain-of-thought or explicit reasoning in v4 and v6. v5 explored reasoning-augmented supervision, but the v5 results showed that adding reasoning increased formatting and truncation risk without improving the final no-reasoning policy objective enough to justify it as the main model format.

The desired output properties are:

- Correct strategic decision.
- Short and natural seller response.
- No hidden reasoning.
- Strong adherence to XML-like tags.
- Emotion-aware phrasing.
- Negotiation progression rather than generic politeness.
- Consistency across text-only and speech-conditioned inference.

## 3. Dataset Overview

The main dataset is stored under:

```text
dataset/
```

The dataset contains 3,154 dialogue JSON files:

```text
dataset/dialogue_0001.json
...
dataset/dialogue_3154.json
```

Each dialogue contains 12 turns, alternating buyer and seller. The full dataset therefore contains:

| Quantity | Count |
|---|---:|
| Dialogues | 3,154 |
| Turns per dialogue | 12 |
| Total turns | 37,848 |
| Buyer turns | 18,924 |
| Seller turns | 18,924 |
| Seller decision examples | 18,924 |

The decision distribution in the full dataset is:

| Decision | Count |
|---|---:|
| MITIGATE | 2,066 dialogues |
| LEVERAGE | 1,088 dialogues |

At the seller-turn level, the SFT builder derives multiple training examples per dialogue. For v6, the training split contains 15,132 seller-turn examples.

The dataset is organized around second-hand electronics bargaining. Products include phones and other consumer devices such as Google Pixel 7a, iPhone 13, Nothing Phone (2a), and similar items. Some older documentation describes the setting as an Indian electronics shop with rupee-style formatting, while the generated dialogue content often uses dollar signs. For the current paper framing, it is safest to describe the dataset as second-hand electronics bargaining and treat the currency symbol as a generated price token rather than a claim about a specific real-world market.

## 4. Dataset Schema

A representative dialogue file contains two major sections:

```json
{
  "seed": {...},
  "processed": [...]
}
```

The `seed` object stores dialogue-level metadata, including:

- Domain and product information.
- Pricing information such as initial price, buyer budget, seller floor, and target price.
- Generation parameters.
- Provenance fields describing when and how the dialogue was created.
- Bias setup and final price.

The `processed` list stores turn-level records. Each turn includes:

- `turn_index`
- `speaker`
- `text`
- `price_offered`
- `factor_state`
- `emotion`
- seller-side `reasoning` when available

The `factor_state` field is central to the negotiation task. It contains structured variables such as:

- Buyer offers so far.
- Anchor price.
- Harm direction.
- Anchoring strength.
- Decision label.
- Negotiation zone.
- Whether the current decision is a flip.

The `emotion` field contains affective annotations, including:

- Emotion label.
- Intensity.
- Valence.
- Confidence.
- Notes and source metadata.

Across the full dataset, emotion annotations cover all dialogues. The aggregate distributions observed from the dataset are:

| Valence | Count |
|---|---:|
| neutral | 16,160 |
| positive | 12,797 |
| negative | 8,891 |

| Intensity | Count |
|---|---:|
| medium | 23,492 |
| low | 7,784 |
| high | 6,572 |

The full audio set contains 37,848 WAV files, corresponding to every dialogue turn.

## 5. Dataset Creation Pipeline

The dataset construction pipeline is represented by several scripts in the project root.

### 5.1 Dialogue Restructuring

The main restructuring script is:

```text
restructure_dataset.py
```

This script converts earlier/raw dialogue artifacts into the current v2 dataset format. The code indicates a two-phase data generation and restructuring process:

1. Dialogue generation.
2. Reasoning generation.

The resulting format is deterministic and stores both dialogue content and structured negotiation metadata. Earlier versions included VAII-style metadata and heavier reasoning requirements. Later project phases gradually removed those from the final model target because the research objective shifted toward a cleaner decision-plus-response seller policy.

The project also includes:

```text
remove_vaii.py
```

This removes VAII and reasoning fields for variants that focus on emotion-first and decision/response-only supervision.

### 5.2 Reasoning Harmonization

The file:

```text
sft_harmonise_reasoning.py
```

normalizes seller reasoning fields. This was important for v5, which trained the model to emit reasoning as part of the output target. The harmonization step helped reduce stylistic variance in reasoning openings and made reasoning targets more consistent.

For v6, reasoning is not used in the target and is not required for eligibility. However, the presence or absence of reasoning is still historically important because it explains why v4/v5 used only 881 dialogues while v6 uses all 3,154 dialogues.

### 5.3 Speech Generation

The audio outputs are stored under:

```text
tts_outputs/
```

The README and file layout show that the project generated turn-level speech for the full dataset. The final inventory contains 37,848 WAV files, matching 3,154 dialogues times 12 turns per dialogue.

The TTS generation machinery includes:

```text
generate_tts_v2.py
voice_instruction_generator_v2.py
monitor_tts_throughput.py
```

The speech data is used in speech-conditioned inference and in multimodal SFT examples. In v6 training, a text-only dropout ratio is also used so the model does not overfit to the presence of audio and remains robust under text-only inference.

### 5.4 Emotion Detection

The emotion detection pipeline is represented by:

```text
detect_emotions.py
```

The script uses Gemini 2.5 Pro for audio emotion detection. It writes turn-level labels back into the dataset, including emotion label, intensity, valence, confidence, notes, and source. This produces the buyer-emotion signal that the negotiation model later consumes.

This design is important because it separates emotion recognition from negotiation policy learning. Rather than training the SFT model to infer all emotion structure from scratch, the supervised dataset contains explicit affect annotations derived from speech.

## 6. Versioned SFT Dataset Builders

The project contains multiple SFT data builders reflecting the evolution of the training target.

### 6.1 v2 Dataset Builder

The file:

```text
sft_data_v2.py
```

uses an earlier target format that included more structured fields:

```text
<reasoning>...</reasoning>
<decision>...</decision>
<seller_vaii>...</seller_vaii>
<response>...</response>
```

This version represents the early stage of the project, where the model was trained with explicit reasoning and VAII-style seller metadata. It is useful historically but no longer matches the final target behavior.

### 6.2 v4 Dataset Builder

The file:

```text
sft_data_v4.py
```

builds the dataset for v4 and v5. It filters to dialogues where all seller turns have both emotion and reasoning fields. This filter leaves 881 eligible dialogues.

The v4 target format removes reasoning and keeps only:

```text
<decision>...</decision>
<response>...</response>
```

v4 therefore tests whether the model can learn the final clean output format from the smaller reasoning-complete subset.

### 6.3 v5 Dataset Builder

v5 uses the same 881-dialogue split as v4, but trains with reasoning enabled. Its target includes explicit reasoning before decision and response. The motivation was to test whether supervised reasoning improves decision quality and response quality.

However, the final evaluation showed that reasoning adds fragility. In speech inference especially, v5 experienced more truncation and format problems. This made v5 less attractive as the production-style final policy even when the reasoning text looked useful to a human reader.

### 6.4 v6 Dataset Builder

The file:

```text
sft_data_v6.py
```

is the main final SFT dataset builder. It follows the v4 output format but drops the reasoning eligibility requirement. Only emotion availability is required. This expands the eligible dataset from 881 dialogues to the full 3,154-dialogue corpus.

Key v6 design choices:

- Uses all 3,154 dialogues.
- Saves splits to `splits_v6.json`.
- Saves class weights to `class_weights_v6.json`.
- Uses `use_reasoning=False`.
- Preserves the v4-style clean target with only decision and response.
- Uses buyer emotion as an explicit input feature.
- Supports speech/audio inputs using the TTS WAV files.
- Includes `text_only_ratio=0.15` during training to reduce overfitting to audio availability.
- Uses resampling utilities so WAV inputs can be aligned to model expectations.

The v6 split is:

| Split | Dialogues |
|---|---:|
| Train | 2,522 |
| Validation | 314 |
| Test | 318 |
| Total | 3,154 |

The v4/v5 split is:

| Split | Dialogues |
|---|---:|
| Train | 704 |
| Validation | 87 |
| Test | 90 |
| Total | 881 |

The v4/v5 class weights are:

| Class | Weight |
|---|---:|
| LEVERAGE | 2.496454 |
| MITIGATE | 0.625222 |

The v6 class weights are:

| Class | Weight |
|---|---:|
| LEVERAGE | 1.449425 |
| MITIGATE | 0.763317 |

The class imbalance is still present in v6, but less extreme than in the 881-dialogue subset.

## 7. Model and Training Setup

The supervised fine-tuning scripts are:

```text
train_sft_v4.py
train_sft_v5.py
train_sft_v6.py
```

All major SFT variants use:

- Base model: `Qwen/Qwen2.5-Omni-7B`
- Mode: thinker-only generation
- Adaptation: LoRA
- LoRA rank: 8
- LoRA alpha: 16
- LoRA dropout: 0.1
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- Audio encoder: frozen
- Loss: class-weighted token loss
- Batch size: 1
- Gradient accumulation: 16
- Epochs: 3
- Learning rate: 2e-5
- Intended GPU: A100 80GB on CUDA device 0

The class-weighted token loss is important because the dataset has a meaningful imbalance between MITIGATE and LEVERAGE decisions. Without weighting, the model could achieve high apparent accuracy by leaning toward the majority class while failing strategically on rarer leverage cases.

The output directories follow the versioned convention:

```text
sft_output_v4/<timestamp>/
sft_output_v5/<timestamp>/
sft_output_v6/<timestamp>/
```

This convention keeps checkpoints, inference outputs, error analyses, judge outputs, and comparison documents grouped under the corresponding SFT run.

## 8. Completed SFT Runs

The main completed runs are:

```text
sft_output_v4/20260525_014244/
sft_output_v5/20260525_092022/
sft_output_v6/20260614_010317/
```

### 8.1 v4

v4 is the no-reasoning baseline on the 881-dialogue reasoning-complete subset.

| Metric | Value |
|---|---:|
| Train dialogues | 704 |
| Validation dialogues | 87 |
| Test dialogues | 90 |
| Train examples | 4,224 |
| Total steps | 792 |
| Best eval loss | 0.4099877 |
| Test loss | 0.423740 |
| Final train loss | 4.6012 |

v4 established that the clean `<decision>` plus `<response>` format works well and can be evaluated reliably in both text and speech inference modes.

### 8.2 v5

v5 uses the same 881-dialogue split but adds reasoning to the target.

| Metric | Value |
|---|---:|
| Train examples | 4,224 |
| Total steps | 792 |
| Best eval loss | 0.809487 |
| Test loss | 0.824099 |
| Final train loss | 10.2358 |

v5 was useful as an experiment in reasoning supervision. However, its weaker decision accuracy and its speech-side formatting/truncation issues suggest that explicit reasoning is not the best final output format for this task.

### 8.3 v6

v6 is the main current SFT model. It uses all 3,154 dialogues and keeps the v4 no-reasoning output format.

| Metric | Value |
|---|---:|
| Train dialogues | 2,522 |
| Validation dialogues | 314 |
| Test dialogues | 318 |
| Train examples | 15,132 |
| Total steps | 2,838 |
| Best eval loss | 0.367639 |
| Test loss | 0.371639 |
| Final train loss | 4.9432 |
| Best checkpoint | checkpoint-2800 |
| Text-only training ratio | 0.15 |

The final adapter is stored at:

```text
sft_output_v6/20260614_010317/final_adapter/
```

v6 was explicitly designed to avoid overfitting by:

- Expanding from 881 to 3,154 dialogues.
- Using validation and test splits saved separately from v4/v5.
- Applying LoRA rather than full fine-tuning.
- Freezing the audio encoder.
- Using dropout in LoRA.
- Using `text_only_ratio=0.15`.
- Monitoring validation loss and checkpoint selection.
- Keeping output targets short and constrained.

## 9. Inference Setup

Inference scripts are:

```text
inference_v4.py
inference_v5.py
inference_v6.py
```

Each script loads the corresponding final adapter and evaluates the model on the version-specific test split. v6 automatically detects:

```text
sft_output_v6/20260614_010317/
```

The project supports both:

- Text inference.
- Speech inference.

The speech inference path uses the buyer audio WAV files. The v6 speech path was fixed to handle multimodal audio placeholders and thinker-only Qwen2.5-Omni generation compatibility.

Completed v6 inference artifacts:

```text
sft_output_v6/20260614_010317/inference_20260620_150244.json
sft_output_v6/20260614_010317/inference_speech_20260615_102532.json
```

The v6 speech WAV outputs are stored outside the run folder in:

```text
/mnt/storage/paritosh/v6_speech_outputs_20260614/
```

A future cleanup step could mirror or symlink speech outputs under `sft_output_v6/20260614_010317/` for a more self-contained artifact layout, but the completed inference JSONs are already stored under the v6 SFT output directory.

## 10. Evaluation Methods

The project evaluates the models with multiple complementary methods.

### 10.1 Decision Accuracy

The simplest metric is exact match on the `<decision>` tag:

```text
predicted decision == gold decision
```

This is computed for both text and speech inference.

Decision accuracy is necessary but not sufficient. A model can classify correctly while still producing weak negotiation responses. For this reason, the project also uses error analysis and LLM-as-a-judge evaluation.

### 10.2 Format Validity

The generated output is checked for expected XML-like tags. For v4 and v6, the expected format is:

```text
<decision>...</decision>
<response>...</response>
```

For v5, reasoning introduces additional format requirements and therefore more chances for truncation or malformed outputs.

### 10.3 Error Analysis

The main script is:

```text
error_analysis.py
```

It analyzes:

- Confusion matrix.
- Decision errors.
- Dialogue zones.
- Decision flips.
- Emotion categories.
- Price proximity.
- Anchoring.
- Repetition.
- GRPO/RL reward implications.

The v6 error analysis output is:

```text
sft_output_v6/20260614_010317/v6_error_analysis_text_and_speech_20260620_161120.md
```

### 10.4 LLM-as-a-Judge

The main script is:

```text
judge.py
```

It uses Qwen3-8B as an evaluator and scores each example on four dimensions:

- Emotion handling.
- Negotiation quality.
- Response naturalness.
- Progression.

The judge output includes aggregate scores and per-example judgments. The v6 judge output is:

```text
sft_output_v6/20260614_010317/v6_judge_text_and_speech_20260620_161122.json
```

The judge is not treated as ground truth. It is used as a structured proxy for qualitative response quality and as a useful diagnostic for reward-function design.

## 11. Main Results

The main comparison artifact is:

```text
sft_output_v6/20260614_010317/v4_v5_v6_comparison_20260620.md
```

The high-level results are:

| Version | Dataset | Target | Text Decision Accuracy | Speech Decision Accuracy | Judge Overall |
|---|---:|---|---:|---:|---:|
| v4 | 881 dialogues | decision + response | 96.9% | 96.9% | 4.03 |
| v5 | 881 dialogues | reasoning + decision + response | 86.3% | 92.4% | 3.88 |
| v6 | 3,154 dialogues | decision + response | 98.3% | 98.4% | 4.12 |

v6 is the strongest model overall. It improves over v4 while using the same clean target format, primarily because it trains on the full dataset rather than the smaller reasoning-complete subset. v5 shows that reasoning supervision is not automatically beneficial when the deployment target is concise decision and response generation.

The v6 judge dimension scores are:

| Dimension | Score |
|---|---:|
| Emotion handling | 3.52 |
| Negotiation quality | 4.53 |
| Response naturalness | 4.41 |
| Progression | 4.04 |
| Overall | 4.12 |

These scores show that v6 is already strong in negotiation quality and naturalness. The weaker dimension is emotion handling, which is directly relevant to the planned RL reward.

## 12. v6 Error Analysis

v6 achieves high decision accuracy, but the remaining errors are informative.

### 12.1 Decision Errors

On the v6 test set:

| Quantity | Count |
|---|---:|
| Total seller-turn examples | 1,908 |
| Text decision errors | 32 |
| Text decision error rate | 1.7% |

The error direction is asymmetric:

| Error Type | Count |
|---|---:|
| Gold MITIGATE predicted LEVERAGE | 24 |
| Gold LEVERAGE predicted MITIGATE | 8 |

This suggests that the model's remaining mistakes often involve being too aggressive in contexts where mitigation is expected.

### 12.2 Decision Flips

The error analysis found:

| Category | Count |
|---|---:|
| Flip-turn errors | 7 / 184 |
| Post-crystallisation errors | 32 / 1,590 |

Decision flips are important because they represent moments where the strategic policy changes. These are exactly the moments where a static classifier can fail. In RL, flip-aware rewards or curriculum sampling can increase pressure on these difficult transition states.

### 12.3 Judge-Derived Weaknesses

The judge analysis highlights that v6's biggest remaining weakness is not format or raw decision classification:

| Weakness | Count |
|---|---:|
| Emotion handling score <= 3 | 882 / 1,908 |
| Progression score <= 3 | 632 / 1,908 |

This motivates a reward function that gives explicit credit for:

- Addressing buyer emotion rather than ignoring it.
- Advancing the negotiation rather than producing generic agreeable responses.
- Choosing a strategically consistent price move.
- Avoiding repetitive or templated responses.

## 13. Interpretation of v4, v5, and v6

The project can be interpreted as a staged investigation.

### 13.1 v4: Clean Target Works

v4 showed that the model can learn the desired format and achieve strong decision accuracy using only:

```text
<decision>
<response>
```

However, v4 was limited by the 881-dialogue subset.

### 13.2 v5: Reasoning Is Not Free

v5 tested whether explicit reasoning supervision improves the policy. It did not improve the main outcomes. Instead, it introduced extra output length, more formatting constraints, and speech-side truncation risk.

This is an important negative result. Reasoning annotations may be useful for offline analysis, reward modeling, or critic training, but they should not necessarily be emitted by the deployed seller agent.

### 13.3 v6: Scale Plus Simpler Target Wins

v6 combines the best lesson from v4 with the full data scale:

- Keep the clean target.
- Drop the reasoning eligibility filter.
- Train on all emotion-annotated dialogues.
- Preserve speech support.

This produces the best decision and judge results among the three main variants.

## 14. Why SFT Is Not the End

SFT trains the model to imitate dataset responses. This is useful, but it has limits:

- It rewards matching the supervised target, not necessarily improving negotiation outcome.
- It does not directly optimize judge dimensions such as emotion handling or progression.
- It cannot easily penalize repetition unless the dataset itself strongly avoids repetition.
- It does not expose the model to sampled alternatives and compare them.
- It does not directly optimize difficult zones such as decision flips.

The v6 error analysis shows that the model is good enough that remaining improvements are likely to come from targeted preference/reward optimization rather than simply more SFT on the same target.

## 15. Reward Function Design

The reward design document is:

```text
rl_reward_function_v6.md
```

The recommended reward is a bounded multi-objective reward:

```text
R_total =
    1.00 * R_decision
  + 0.30 * R_emotion
  + 0.25 * R_progression
  + 0.25 * R_price_strategy
  + 0.15 * R_format
  + 0.10 * R_naturalness
  - 0.20 * P_repetition
  - 0.20 * P_language_impurity
```

The final reward should be clipped to a bounded interval such as:

```text
[-2, 2]
```

Optional additions include:

- Audio-consistency bonus.
- Decision-flip bonus.
- Penalty for over-aggression when gold or context suggests mitigation.
- Penalty for passivity when leverage is justified.

### 15.1 Decision Reward

The decision reward should be the strongest term because incorrect strategic direction can make even fluent responses behaviorally wrong.

A simple formulation:

```text
R_decision = +1 if predicted decision matches gold decision
R_decision = -1 otherwise
```

This is intentionally simple. The SFT model already performs well, so RL should not destabilize decision accuracy while optimizing softer dimensions.

### 15.2 Emotion Reward

The emotion reward should target the largest judge weakness. It can be derived from:

- Judge emotion-handling score.
- Presence of emotionally appropriate acknowledgments.
- Avoidance of tone-deaf responses.
- Consistency between valence/intensity and seller response style.

Example mapping:

```text
R_emotion = (judge_emotion_score - 3) / 2
```

This maps a 1-5 judge score into approximately `[-1, 1]`.

### 15.3 Progression Reward

Progression rewards responses that move the negotiation forward:

- Makes or defends a price.
- Refers to product value.
- Responds to buyer's offer.
- Avoids empty agreement.
- Avoids repeating earlier seller language.

The progression reward is motivated by the 632 / 1,908 v6 examples with judge progression score <= 3.

### 15.4 Price Strategy Reward

The price-strategy reward should use structured dataset fields such as:

- Current buyer offer.
- Seller floor.
- Anchor price.
- Zone.
- Harm direction.
- Decision.
- Whether the decision is a flip.

The exact implementation can begin as a heuristic. For example:

- LEVERAGE responses should avoid immediately conceding below strategic thresholds.
- MITIGATE responses can soften, but should not collapse below floor.
- Responses should be consistent with the current zone and offer history.

### 15.5 Format Reward

The format reward protects the clean deployment interface:

```text
<decision>...</decision>
<response>...</response>
```

This term can be binary:

```text
R_format = +1 if tags are valid and complete
R_format = -1 otherwise
```

Because v6 already has near-perfect formatting, this term should be lower weight than decision or emotion rewards.

### 15.6 Naturalness Reward

Naturalness can be derived from the judge response-naturalness score or from a lightweight critic. v6 is already strong here, so this term should be modest. Its purpose is to prevent RL from producing awkward reward-hacking responses.

### 15.7 Repetition Penalty

Repetition is a common RL failure mode and a dialogue-quality issue. The penalty can detect:

- Repeated seller phrases across turns.
- Repeated clauses within a response.
- Excessive reuse of generic templates.

This term should be included early because RL can amplify repetitive patterns if they accidentally score well on decision or format rewards.

### 15.8 Language Impurity Penalty

The project should penalize unwanted language mixing, malformed tags, leaked reasoning, and artifacts that make the response less deployable. This can be implemented as a regex or classifier-based penalty.

## 16. Do We Need Complex Mathematical Formulations?

At this stage, no. The reward function does not need complex mathematical notation to be useful. The more important properties are:

- Direct connection to observed errors.
- Bounded scale.
- Interpretability.
- Easy debugging.
- Per-component logging.
- Resistance to reward hacking.

A simple weighted sum is appropriate for the first RL experiments. Complex formulations should be introduced only if they solve a concrete problem observed during RL, such as instability, over-optimization of one component, or poor credit assignment across turns.

The paper can still include mathematical notation later, but the implementation should remain auditable. For example, the paper can define:

```text
R(x, y) = sum_i w_i r_i(x, y) - sum_j lambda_j p_j(x, y)
```

where `x` is the dialogue state, `y` is the model output, `r_i` are positive reward components, and `p_j` are penalties. This is mathematically clear without making the system unnecessarily complex.

## 17. RL Direction

The repository includes an `Omni-R1/` directory, which contains GRPO-related training scripts and reward utilities. The presence of this directory suggests the next implementation direction:

- Start from the v6 final adapter.
- Generate multiple candidate seller responses per prompt.
- Score candidates with the reward function.
- Optimize with GRPO-style reinforcement learning.

The reward should be logged component-by-component:

| Component | Purpose |
|---|---|
| `R_decision` | Preserve strategic classification |
| `R_emotion` | Improve emotion handling |
| `R_progression` | Improve negotiation movement |
| `R_price_strategy` | Improve price consistency |
| `R_format` | Preserve parseable output |
| `R_naturalness` | Preserve fluency |
| `P_repetition` | Prevent repetitive outputs |
| `P_language_impurity` | Prevent artifacts and leakage |

The first RL phase should be conservative. The SFT model is already strong, so RL should focus on improving judge and error-analysis weaknesses without damaging decision accuracy.

## 18. Reproducibility and Artifact Map

Important top-level documentation:

```text
README.md
dataset/README.md
intermediate/README.md
tts_outputs/README.md
sft_output_v4/README.md
sft_output_v5/README.md
```

Important dataset and preprocessing scripts:

```text
restructure_dataset.py
remove_vaii.py
detect_emotions.py
sft_harmonise_reasoning.py
generate_tts_v2.py
voice_instruction_generator_v2.py
monitor_tts_throughput.py
```

Important SFT scripts:

```text
sft_data_v2.py
sft_data_v4.py
sft_data_v6.py
train_sft_v2.py
train_sft_v3.py
train_sft_v4.py
train_sft_v5.py
train_sft_v6.py
```

Important inference and evaluation scripts:

```text
inference_v4.py
inference_v5.py
inference_v6.py
error_analysis.py
judge.py
sft_eval_analysis.py
sft_full_compare.py
```

Important split and class-weight files:

```text
splits_v4.json
splits_v6.json
class_weights_v2.json
class_weights_v4.json
class_weights_v6.json
```

Important completed model outputs:

```text
sft_output_v4/20260525_014244/
sft_output_v5/20260525_092022/
sft_output_v6/20260614_010317/
```

Important v6 artifacts:

```text
sft_output_v6/20260614_010317/final_adapter/
sft_output_v6/20260614_010317/inference_20260620_150244.json
sft_output_v6/20260614_010317/inference_speech_20260615_102532.json
sft_output_v6/20260614_010317/v6_error_analysis_text_and_speech_20260620_161120.md
sft_output_v6/20260614_010317/v6_judge_text_and_speech_20260620_161122.json
sft_output_v6/20260614_010317/v4_v5_v6_comparison_20260620.md
rl_reward_function_v6.md
```

## 19. Limitations

This first draft should be treated carefully. Several limitations remain:

1. The data is synthetic or semi-synthetic, so real-world generalization must be tested separately.
2. Emotion labels are produced by an external model and may contain systematic bias.
3. The judge is also an LLM, so judge scores are approximate quality signals rather than human ground truth.
4. v4/v5 and v6 use different test splits, so direct numeric comparisons must mention that v6 is evaluated on a larger and different test set.
5. Decision accuracy is high, but response-quality weaknesses remain.
6. The current reward function is designed but not yet validated through completed RL experiments.
7. Audio files are generated TTS rather than naturally recorded human negotiation speech.
8. The currency and market setting should be normalized in the final paper.

These limitations do not invalidate the project. They clarify what the paper should claim. The strongest current claim is that scaling from the reasoning-complete subset to the full emotion-annotated multimodal dataset improves a concise decision-plus-response negotiation agent, and that post-SFT evaluation identifies emotion handling and negotiation progression as the most valuable next targets for RL.

## 20. Draft Paper Framing

A possible paper framing is:

**Problem:** Build a multimodal seller agent that uses buyer speech emotion and negotiation context to choose a strategic bargaining action and produce a natural response.

**Dataset contribution:** A 3,154-dialogue, 37,848-turn multimodal negotiation dataset with utterance-level speech, affect annotations, price state, and seller strategic labels.

**Model contribution:** A Qwen2.5-Omni-7B LoRA SFT pipeline for text and speech negotiation inference, comparing no-reasoning and reasoning-augmented targets.

**Empirical finding:** The best model is not the reasoning-output model. The best model is v6: a concise no-reasoning target trained on the full emotion-annotated dataset.

**Evaluation contribution:** The project combines decision accuracy, speech/text consistency, error analysis, and LLM-as-a-judge scoring to identify remaining behavioral weaknesses.

**RL contribution in progress:** A targeted reward function derived from observed SFT errors and judge weaknesses, designed for GRPO-style optimization of emotion handling, progression, and price strategy while preserving decision accuracy and output format.

## 21. Current Project State

As of the completed v6 evaluation:

- Dataset construction is complete for the current corpus.
- Full TTS coverage exists for all 3,154 dialogues.
- Emotion annotation exists for all dialogues.
- v4, v5, and v6 SFT training are complete.
- v6 text inference is complete.
- v6 speech inference is complete.
- v6 error analysis is complete.
- v6 LLM-as-a-judge evaluation is complete.
- v4/v5/v6 comparison is saved.
- Reward-function design is written.
- RL implementation remains the next major experimental phase.

## 22. Next Steps

The immediate research and engineering next steps are:

1. Convert `rl_reward_function_v6.md` into executable reward code.
2. Add per-component reward logging.
3. Create a small offline reward audit over existing v6 generations.
4. Run a small GRPO smoke test from the v6 adapter.
5. Compare RL outputs against v6 SFT using the same text/speech inference, error analysis, and judge pipeline.
6. Add human evaluation for a sampled subset, especially cases where the judge rates emotion handling or progression poorly.
7. Normalize paper terminology around dataset currency, market setting, and synthetic generation.
8. Add citations for Qwen2.5-Omni, Qwen3, LoRA, GRPO, LLM-as-a-judge evaluation, speech emotion recognition, and negotiation dialogue modeling.

## 23. Working Title Options

Possible titles:

1. Emotion-Aware Multimodal Negotiation Agents with Speech-Conditioned Supervised Fine-Tuning
2. From Speech Emotion to Strategic Bargaining: Training a Multimodal Seller Agent
3. Scaling No-Reasoning SFT for Emotion-Aware Negotiation with Qwen2.5-Omni
4. Text and Speech Supervision for Strategic Negotiation Dialogue Agents

## 24. Short Summary for Abstract Revision

We develop a multimodal seller agent for second-hand electronics negotiation. The agent observes dialogue context, price state, and buyer affect derived from speech, then outputs a strategic `LEVERAGE` or `MITIGATE` decision and a concise seller response. We construct a 3,154-dialogue, 37,848-turn dataset with utterance-level TTS audio, emotion annotations, price metadata, and strategic seller labels. We compare supervised fine-tuning variants of Qwen2.5-Omni-7B using LoRA. A no-reasoning model trained on the full dataset, v6, outperforms both a no-reasoning smaller-subset baseline and a reasoning-augmented variant, reaching 98.3% text decision accuracy, 98.4% speech decision accuracy, and a 4.12/5 LLM-as-a-judge overall score. Error analysis shows that remaining weaknesses concentrate in emotion handling, progression, and difficult strategic flips rather than output format. We therefore design a bounded multi-component reward function for the next GRPO-based RL phase, combining decision correctness with emotion handling, negotiation progression, price strategy, naturalness, and anti-repetition penalties.

