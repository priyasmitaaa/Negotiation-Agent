# v2/ — Active Workspace (v2+ Training, Inference, Evaluation)

This is the main active workspace for the negotiation agent project. Everything in `../` (root) is v1 and superseded. All ongoing work lives here.

## What this workspace does

Trains `Qwen2.5-Omni-7B` with LoRA on second-hand electronics negotiation dialogues. The model learns to:
1. Analyse buyer emotion and price signals
2. Decide: `LEVERAGE` / `MITIGATE` / `UNDECIDED`
3. Generate a natural seller response (text + optional speech via TTS)

## Key directories

| Folder | Purpose |
|---|---|
| `dataset/` | 3,154 restructured negotiation dialogues (canonical source of truth) |
| `intermediate/` | Working copies of dialogues with reasoning annotations written in |
| `intermediate_backup_vaii/` | Backup of intermediate folder before VAII label removal |
| `tts_outputs/` | Ground-truth TTS WAV files (buyer + seller turns) for all 3,154 dialogues |
| `sft_output_v4/` | SFT v4 training run outputs (checkpoints, inference, judge, error analysis) |
| `sft_output_v5/` | SFT v5 training run outputs (with reasoning in target) |

## Key scripts

| Script | Purpose |
|---|---|
| `train_sft_v4.py` | SFT trainer for v4 (no reasoning in target) |
| `train_sft_v5.py` | SFT trainer for v5 (reasoning + decision + response) |
| `inference_v4.py` | Text-only inference for v4 adapter |
| `inference_v5.py` | Text + optional speech inference for v5 adapter |
| `judge.py` | LLM-as-judge scorer (Qwen3-8B) for response quality |
| `error_analysis.py` | Decision accuracy + GRPO reward design analysis |
| `detect_emotions.py` | Emotion detection pass over dataset |
| `generate_tts_v2.py` | TTS generation for ground-truth audio |
| `restructure_dataset.py` | Full v1→v2 dataset restructuring script |

## SFT versions

| Version | Target format | Decision accuracy | Notes |
|---|---|---|---|
| v4 | `<decision> <response>` | 96.9% (523/540) | No reasoning, emotions in context |
| v5 | `<reasoning> <decision> <response>` | TBD | Adds chain-of-thought reasoning |
| v6 | `<decision> <response>` | 98.3% text / 98.4% speech | No reasoning, full 3,154-dialogue corpus. Current best model — see `research_paper_draft_v1.md`. |

## RL / GRPO phase (current work)

After v6 SFT, the project moved to RL (GRPO) to fix the judge-identified weak
spots (emotion handling, negotiation progression). See `research_paper_draft_v1.md`
section 15+ for the full narrative. Key files:

| File | Purpose |
|---|---|
| `reward_function.py` | Executable multi-component reward (decision, emotion, progression, price strategy, format, naturalness, repetition/language penalties, audio quality) |
| `rl_reward_function_v6.md` / `grpo_reward_candidates.md` | Reward design docs the above implements |
| `grpo_curriculum.py` | Builds GRPO training examples + easy/medium/hard curriculum buckets from `factor_state`/emotion fields (currently hand-weighted thresholds — see `causal_rm_curriculum_plan.md` for a proposed empirically-validated replacement) |
| `train_grpo_curriculum.py` | GRPO training entrypoint, **default as of 2026-08-17** — trl-native `AudioGRPOTrainer` (`trl_audio_grpo_trainer.py`), no vendored trainer, no isolated venv. See `causal_rm_results.md` sections 4.3/4.4 for the memory-safety and training-stability evidence behind the switch |
| `train_grpo_curriculum_omni_r1.py` | **Not runnable** (2026-08-17) — legacy GRPO entrypoint for the now-removed `Omni-R1/` vendored trainer, preserved as a historical/documentation record only (it's the code that produced the memory comparison in `causal_rm_results.md` section 4.3). The IP-cleanliness question from section 4.1 is resolved: `Omni-R1/` has been deleted, this project ships without it |
| `causal_rm_curriculum_plan.md` | **Design proposal** (not yet implemented) for a data-driven curriculum difficulty score, validated via correlation with SFT loss and curriculum-vs-random GRPO pilots — a parallel workstream to the causal-factor RM, not a dependency |
| `human_annotation_guide.md` / `human_eval_guidelines.md` | Human eval rubric to validate RL improvements beyond automatic reward |

### Intervention-based causal-factor reward modeling (new direction, in progress)

The heuristic reward above is keyword/length-driven, which is exactly the
"reward hacking" risk that causal-rubric reward modeling (CRome / CausalRM)
targets. We are extending the reward layer to be explicitly sensitive to
negotiation-relevant causal factors and invariant to spurious surface
factors, aiming this at an ACL-level contribution. **Terminology note:** we
call this "intervention-based causal-factor" sensitivity rather than
"causal" unqualified — our interventions are donor-substitution/template
edits on structured fields, not a formal SCM with identified do-operators;
see `causal_rubric_taxonomy.md`'s terminology note for the full rationale.
See:

| File | Purpose |
|---|---|
| `causal_rubric_rl_plan.md` | Full integration plan (phases, file map, verification steps) — start here |
| `causal_rubric_taxonomy.md` | Formal causal (C1-C6) vs spurious (S1-S5) factor taxonomy, grounded in `factor_state`/`emotion` schema fields |
| `generate_intervention_pairs.py` | Builds causal/spurious intervention pairs from `dataset/` for RM training (outputs `causal_rm_pairs_{split}.json`). Covers causal C1/C3/C5 and spurious S1 (length)/S2 (politeness)/S3 (formatting) so far; S6/S7 (paraphrase/fluency, need LLM + manual check) and S4 (prosody, needs TTS regen) are documented as required-before-final-experiments in `causal_rubric_taxonomy.md` section 4 |
| `causal_rm_pairs_train.json` / `_val.json` / `_test.json` | Generated intervention pairs, split-aligned with `splits_v6.json` |
| `causal_rm_architecture.md` | Finalized rubric-vector RM design (diagram, frozen-backbone + shared-trunk + per-dimension heads, Bradley-Terry + invariance + reg loss, verification checklist) — written before model code per the revised implementation order |
| `causal_reward_model.py` | Rubric-vector scorer: frozen Qwen3-8B backbone + trainable shared MLP trunk + 5 rubric heads (`decision_score`, `price_strategy`, `emotion`, `progression`, `flip_score`, all tanh-bounded to [-1,1]). `score(example, response) -> (overall, components)` mirrors `reward_function.py::compute_reward`'s contract; `decision_score`/`flip_score` are diagnostics only, not summed into `overall` |
| `train_causal_rm.py` | Trains the `RubricHead` (trunk+heads only, backbone frozen) on `causal_rm_pairs_*.json` — Bradley-Terry loss per dimension on causal pairs, invariance loss on spurious pairs, small L2 stability regularizer, embedding caching, head-correlation monitoring. Verified end-to-end with a lightweight fake-embedder dry run (plumbing only, not a real training result) |
| `causal_rm_audit.py` | Go/no-go diagnostic: causal-pair ranking accuracy + spurious-pair invariance, works with `--backend heuristic` (no GPU/RM needed) or `--backend causal_rm --checkpoint ...` (once trained). `--report_dir <dir>` generates matplotlib plots (ranking accuracy, invariance, per-dimension score/delta distributions) + a markdown summary table alongside the JSON — directly comparable output so "before vs after" is one table |
| `run_causal_rm_experiment.py` | **Single-command experiment runner**: train → audit → report → summary in one call, into `causal_rm_experiments/<name>_<timestamp>/` (config.json, checkpoint/, audit/ with plots, summary.md with an explicit go/no-go checklist). Built so ablations (varying lambda_bt/lambda_inv/lambda_reg/max_pairs) stay reproducible and don't require remembering to manually audit each checkpoint |
| `causal_rm_curriculum_plan.md` | Design proposal (not yet implemented) for a data-driven curriculum difficulty score — parallel workstream, not a dependency of the RM work |
| **`causal_rm_results.md`** | **Start here for status.** Consolidated results doc: full experimental story (lambda_inv bug found+fixed, C2/C4 causal-coverage gap found+fixed), final go/no-go numbers for the **text-grounded v1 RM** (PASS — every causal dimension now 85-99% accuracy vs. a 0-11% heuristic baseline), the known GRPO-pilot limitation, and a pointer to the v2 audio-fusion design below |
| `audio_tower_separation_check.py`, `ser_model_separation_check.py` (+ `*_result.json`) | Audit scripts + results for the v2 audio-encoder selection: Qwen2.5-Omni's `audio_tower` tested and **rejected** (no calm/distressed separation under any pooling strategy); `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` tested and **accepted** (real, content-independent separation) — see `causal_rm_architecture.md` section 9 for the full writeup |

**Status (2026-08-02): v1 (text-grounded) RM validated and complete — see
`causal_rm_results.md`.** Recommended v1 checkpoint:
`causal_rm_experiments/full_coverage_2000pairs_lambda_inv_1.5_20260801_141008/checkpoint`.
GRPO integration hooks are built and off by default: `reward_function.py::compute_reward(..., causal_rm=None)`
and `grpo_curriculum.py::make_negotiation_grpo_reward(causal_rm=None)` both
default to the exact original heuristic behavior; `train_grpo_curriculum.py --causal_rm_checkpoint <dir>`
activates the RM path (currently blocked from a live pilot run by the
environment issue documented in `causal_rm_results.md` section 4 — the RM
itself is unaffected and usable via `causal_rm_audit.py` independent of GRPO).
**v2 (audio-grounded `emotion`) is implemented, trained at full scale, and
validated (2026-08-16)** — see `causal_rm_architecture.md` section 9 for
the full dual-trunk fusion design, encoder-selection audit, and section
9.4.1 for the trunk-decoupling finding (confirmed at both prototype and
full-dataset scale). Recommended v2 checkpoint:
`causal_rm_experiments/v2_audio_fusion_full_20260816_152709/checkpoint`
(val audit: `emotion` 100.0% causal ranking accuracy, exceeding v1's
98.66% text-only baseline; `decision_score` 99.4%, `flip_score` 100%,
`price_strategy` 94.9%, `progression` 90.7%). Train with
`train_causal_rm.py --audio_fusion` (or `run_causal_rm_experiment.py
--audio_fusion`); `causal_reward_model.py::load_causal_rm` auto-detects
audio-fusion checkpoints from their `config.json` and loads
`SERFusionHead` + the frozen SER model automatically, so
`causal_rm_audit.py` and any downstream `CausalRewardModel.score()` caller
work unchanged. GRPO integration for this checkpoint is untested (same
`train_grpo_curriculum.py --causal_rm_checkpoint` path as v1, still
blocked by the environment issue in section 4 above, unrelated to audio
fusion).
