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
| `Omni-R1/` | Reference repo for GRPO/RL training on Omni models (next phase) |

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
| `grpo_curriculum.py` | Builds GRPO training examples + easy/medium/hard curriculum buckets from `factor_state`/emotion fields |
| `train_grpo_curriculum.py` | GRPO training entrypoint (uses `Omni-R1/` trainer) |
| `human_annotation_guide.md` / `human_eval_guidelines.md` | Human eval rubric to validate RL improvements beyond automatic reward |

### Causal-rubric reward modeling (new direction, in progress)

The heuristic reward above is keyword/length-driven, which is exactly the
"reward hacking" risk that causal-rubric reward modeling (CRome / CausalRM)
targets. We are extending the reward layer to be explicitly sensitive to
*causal* negotiation factors and invariant to *spurious* surface factors,
aiming this at an ACL-level contribution. See:

| File | Purpose |
|---|---|
| `causal_rubric_rl_plan.md` | Full integration plan (phases, file map, verification steps) — start here |
| `causal_rubric_taxonomy.md` | Formal causal (C1-C6) vs spurious (S1-S5) factor taxonomy, grounded in `factor_state`/`emotion` schema fields |
| `generate_intervention_pairs.py` | Builds causal/spurious intervention pairs from `dataset/` for RM training (outputs `causal_rm_pairs_{split}.json`). Covers causal C1/C3/C5 and spurious S1 (length)/S2 (politeness)/S3 (formatting) so far; S6/S7 (paraphrase/fluency, need LLM + manual check) and S4 (prosody, needs TTS regen) are documented as required-before-final-experiments in `causal_rubric_taxonomy.md` section 4 |
| `causal_rm_pairs_train.json` / `_val.json` / `_test.json` | Generated intervention pairs, split-aligned with `splits_v6.json` |

Planned next files (not yet written): `train_causal_rm.py` (trains the
causal-invariant reward model), `causal_reward_model.py` (drop-in scorer that
plugs into `reward_function.py`/`grpo_curriculum.py` in place of the keyword
heuristics — outputs a **rubric vector** per `causal_rubric_taxonomy.md`
section 5, not a single scalar, matching `reward_function.py`'s existing
`components` dict shape), `causal_rm_audit.py` (reward-hacking /
spurious-correlation diagnostics for the paper).
