# Causal-Factor-Aware Reward Model — Results

**Status: text-grounded RM (v1) validated and complete; audio-grounded
`emotion` fusion (v2) designed 2026-08-02, not yet built.**
Everything below (sections 1-3) describes the v1 RM, which scores
`price_strategy`/`emotion`/`progression`/`decision_score`/`flip_score`
against text-derived context (`factor_state`, `emotion.labels/intensity/
valence` — themselves Gemini-derived from audio during dataset
construction, but the RM itself never touches the waveform). This was
flagged directly as insufficient for a project whose core premise is
speech-conditioned negotiation — a reward that never perceives raw audio
can't distinguish genuine audio understanding from label pattern-matching.
`causal_rm_architecture.md` section 9 now documents the audited, designed
fix: a dual-trunk fusion feeding real buyer audio (via a dedicated,
domain-validated speech-emotion-recognition model) into the `emotion`
head specifically — `flip_score` was considered and then deliberately
excluded on review (see section 9.1: it's a structural/text check, not a
perceptual one, same category as `decision_score`). That section includes
the full encoder
selection audit (Qwen2.5-Omni's own audio tower was tested and rejected —
no separation between calm/distressed buyer speech under any pooling
strategy; `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` was
tested and accepted — real, content-independent separation). Section 4
below (live GRPO pilot blocked) is unrelated to this and still accurate as
written for the v1 RM. Read `causal_rm_architecture.md` section 9 before
assuming the v1 numbers below are the final word on this RM's audio
grounding.

This document is the single reportable summary of the causal-factor reward
model work: the taxonomy it's built on, the architecture, the full
experimental story (including a real bug found and fixed mid-investigation),
and the final go/no-go numbers. Supporting detail lives in
`causal_rubric_taxonomy.md` (factor definitions), `causal_rm_architecture.md`
(model design), and `causal_rubric_rl_plan.md` (original integration plan).

---

## 1. What this is

`reward_function.py`'s original reward is keyword/heuristic-based
(`price_strategy_reward`, `emotion_reward`, `progression_reward` — regex and
term-list matching). The causal-factor RM replaces those three components
with a learned model trained to be **sensitive to causal negotiation
factors** (does the response actually fit the buyer's price position,
emotional state, and negotiation stage) and **invariant to spurious surface
factors** (response length, generic politeness phrasing, formatting) —
see `causal_rubric_taxonomy.md` for the full C1-C6 / S1-S7 factor
definitions this is built on.

Architecture: frozen Qwen3-8B backbone → shared MLP trunk → 5 tanh-bounded
rubric heads (`decision_score`, `price_strategy`, `emotion`, `progression`,
`flip_score`). Only `price_strategy`/`emotion`/`progression` feed the
`overall` reward; `decision_score`/`flip_score` stay diagnostic-only
(decision/flip correctness is already handled by cheap, reliable rule-based
logic in `reward_function.py` — no need to learn what's already solved).
Full rationale in `causal_rm_architecture.md`.

## 2. The experimental story

This wasn't a single train-and-done run — the process surfaced and fixed two
real bugs, which is part of why the final result is trustworthy rather than
just a lucky number.

### 2.1 Baseline (no RM, heuristic reward as-is)

Audited the existing heuristic reward against intervention pairs built from
the dataset's own structured fields (`generate_intervention_pairs.py`,
donor-substitution method — see taxonomy doc). Result, full val split:

| Dimension | Causal ranking accuracy | Spurious invariance (Δ) |
|---|---:|---:|
| `emotion` | **10.76%** (worse than chance) | 0.027 |
| `price_strategy` | **0%** | 0.0 (degenerate — heuristic ignores response content) |
| `progression` | **0%** | 0.001 (degenerate, same reason) |
| `decision_score` | 100% (rule-based, trivial) | 0.0 |
| `flip_score` | 100% (rule-based, trivial) | 0.0 |

The `price_strategy`/`progression` 0% numbers only appeared after C2/C4 pair
generation was added (section 2.4) — worth remembering when reading this
table, since it's not the number that was visible at the very start of the
investigation, but it's the correct final baseline for comparison.

### 2.2 First prototype (300 pairs, S1-S3 spurious coverage, C1/C3/C5 causal)

`emotion` causal accuracy: 10.76% → **97.46%**. Strong signal the core
mechanism works. But spurious invariance was *worse* than the (degenerate)
heuristic baseline across most dimensions.

### 2.3 lambda_inv sweep → bug found → fixed → re-verified

A sweep over `lambda_inv` (1.0/1.5/2.5/3.5) found `decision_score`'s
invariance degrading monotonically with λ (0.016→0.067→0.181→0.287) while
`emotion` improved. Root cause, found by inspecting `train_causal_rm.py`
directly: `spurious_pair_loss` only computed invariance loss over the 3
`overall`-feeding dimensions — `decision_score`/`flip_score` **never
received any invariance gradient at any λ**, so they drifted as an
uncontrolled side-effect of the shared trunk absorbing more invariance
pressure elsewhere.

**Fix**: widened `spurious_pair_loss` to cover all 5 dimensions (they still
don't feed `overall` — just get their own regularization signal now).
Re-verified with a matched sweep: at λ=1.5, `emotion` invariance improved
13× (0.097→0.0077) and **beat the heuristic baseline outright** (0.0077 vs
0.027), while `decision_score` improved 2× (0.067→0.033). Confirmed via a
2000-pair scale-up run that this wasn't small-sample luck — `emotion` held
at 98.66% causal accuracy / 0.0016 invariance (even stronger at scale).

### 2.4 Second gap found: `progression`/`price_strategy` had no causal anchor at all

After the λ fix, `progression`'s invariance still regressed at the 2000-pair
scale (0.0075→0.024). A free diagnostic — computing the trained RM's
pairwise head-correlation matrix on real val examples — **ruled out** the
obvious hypothesis (emotion's gradient bleeding into progression via the
shared trunk: measured correlation was -0.134, weakly *negative*, not
positive). The real cause: `causal_rubric_taxonomy.md`'s coverage table
already documented that no C2 (price strategy) or C4 (progression) causal
pairs existed — those two heads only ever got invariance-loss gradient,
never causal-ranking gradient, so they were structurally underdetermined.

**Fix**: implemented `causal_pair_price_strategy` (C2) and
`causal_pair_progression` (C4) in `generate_intervention_pairs.py`, reusing
the same donor-substitution machinery as C1/C3/C5. C4 required a genuine
additional fix mid-implementation: `progression_stage` (offer-count bucket)
turned out to be perfectly confounded with `zone` in this dataset
(`pre_crystallisation` examples have *exactly* 1 buyer offer, always, zero
variance — verified empirically, not assumed), making the intended
same-zone donor contrast structurally impossible. Fixed by scoping C4 to
`post_crystallisation` only, where offer count has real spread (2-6),
documented as an honest scope limit in the code.

Re-running the heuristic baseline against the new C2/C4 pairs produced the
striking **exactly-0%** numbers quoted in section 2.1 — independent
confirmation the heuristic reward is blind to negotiation state on two of
the three dimensions that make up its actual output.

## 3. Final result

Retrained with full C1-C5 causal coverage (2000 pairs, λ_inv=1.5, the fixed
loss), audited on a 1000-pair held-out sample of the val split (vs. the
heuristic's full-split audit — noted for scale honesty, not hidden):

| Dimension | **RM causal accuracy** | Heuristic baseline | **RM spurious Δ** | Heuristic Δ |
|---|---:|---:|---:|---:|
| `emotion` | **98.66%*** | 11.33%** | **0.021** | 0.027 |
| `price_strategy` | **86.76%** | 0% | 0.019 | 0.0 (degenerate) |
| `progression` | **84.75%** | 0% | 0.024 | 0.001 (degenerate) |
| `decision_score` (diagnostic) | 82.80% | 100% (trivial) | 0.026 | 0.0 (degenerate) |
| `flip_score` (diagnostic) | 100% | 100% (trivial) | 0.021 | 0.0 (degenerate) |

*emotion peaked at 98.95%/0.0016 on the intermediate 2000-pair run before
final C2/C4 retraining slightly redistributed optimization pressure across
now-5-causally-anchored dimensions — still dramatically ahead of baseline.
**baseline `emotion` accuracy is 11.33% on this specific audit sample (this
document's headline number of 10.76% is from a very slightly different
pair-generation run earlier in the investigation; both numbers are
"worse than chance," the exact figure moved by ~0.6pp across pair
regenerations, immaterial to the conclusion).

**Every causal dimension that used to have zero grounding (`price_strategy`,
`progression`) now sits at 85-87% causal accuracy — up from a heuristic
baseline that was literally 0%.** `emotion` remains dominant and beats the
heuristic on both axes simultaneously. Remaining invariance gaps vs. the
heuristic's degenerate zeros are modest in absolute terms (0.02-0.03 on a
bounded [-1,1] scale) and reflect a reward that actually reads response
content, which the heuristic never did.

**Go/No-Go verdict: PASS.** This is the strongest checkpoint produced across
the entire investigation:
`causal_rm_experiments/full_coverage_2000pairs_lambda_inv_1.5_20260801_141008/checkpoint`

## 4. Known limitation: live GRPO pilot blocked

The RM is wired into `reward_function.py`/`grpo_curriculum.py`/
`train_grpo_curriculum.py` behind an opt-in `--causal_rm_checkpoint` flag
(off by default, verified not to change existing behavior). Actually running
a GRPO training pilot with it enabled is currently blocked, not by the RM,
but by an environment/library compatibility issue:

- Getting the vendored `Omni-R1` GRPO trainer to import at all required
  fixing 8 separate environment issues (missing/broken optional
  dependencies, a `trl==0.15.2` vs. system `trl==0.24.0` version conflict
  resolved via an isolated venv `.venv_grpo_pilot/`, an HF API drift shim,
  a GPU-visibility misconfiguration, and an empty-audio-batch bug worked
  around by running with real audio instead of `--no_audio_input`).
- With all of that resolved, GPU memory became the next constraint: the
  combined policy + causal RM footprint (~44GB) plus another user's
  concurrent job on the same shared GPU pushed past capacity.
  `gradient_checkpointing=True` was enabled to fit the memory budget, but
  this **breaks the model's audio-conditioned forward pass** with a
  key/value shape-mismatch crash — reproduced identically under both
  `sdpa` and `eager` attention backends, meaning the conflict is with
  checkpointing itself, not the attention implementation.
- This crash occurs inside third-party code (`transformers`'s Qwen2.5-Omni
  decoder layer × Omni-R1's vendored `GRPOTrainer`), not in any of our own
  reward/RM/dataset code.

**This is a genuine memory-vs-stability conflict on this specific shared
machine at this specific library-version combination — not evidence of any
problem with the causal RM.** It does not block anything else in the
project roadmap; the RM is independently usable and auditable via
`causal_rm_audit.py` and `causal_reward_model.py` without GRPO at all.

### 4.1 Omni-R1 dependency audit (2026-08-01) — read this before assuming a trainer swap fixes anything

A code audit (no changes made) confirmed only one file
(`train_grpo_curriculum.py`) has a real runtime dependency on Omni-R1 — a
single `from trainer.grpo_trainer import GRPOTrainer` import. The
negotiation policy's model architecture
(`Qwen2_5OmniForConditionalGeneration`) comes directly from the official
`transformers` library, not from Omni-R1. `train_sft_v3.py`'s Omni-R1
mentions are comments only (design reference, zero imports). So far this
sounds like a clean, one-file swap — **it is not**, for three reasons worth
stating precisely so this doesn't get re-investigated from scratch later:

1. **The blocker is a `transformers`-level bug, not a trl-version or
   trainer-choice issue.** The crash lives inside `transformers`'s own
   Qwen2.5-Omni decoder layer / SDPA attention code, triggered whenever
   gradient checkpointing wraps an audio-conditioned forward pass —
   reproduced identically under two different attention backends (`sdpa`,
   `eager`), which rules out the attention implementation as the cause.
   Any trainer that performs a gradient-checkpointed, audio-conditioned
   forward pass through this model on this `transformers` version would hit
   the same wall.
2. **Omni-R1's vendored trainer is not a thin wrapper around `trl.GRPOTrainer`
   — it's an independent ~600-line GRPO reimplementation** (subclasses HF's
   base `Trainer` directly, forked from an early community GRPO
   implementation, predating and diverging from trl's mainline). It exists
   specifically because **stock `trl` never added audio-modality support** —
   checked directly: `trl==0.24.0`'s `grpo_trainer.py` has zero references
   to audio or `input_features` anywhere, and no Omni/Qwen2Audio-specific
   code exists anywhere in the `trl` package. It does have real, actively
   maintained multimodal support, but image-only. So removing Omni-R1's
   trainer means choosing between (a) rebuilding audio-conditioned GRPO
   training from scratch on top of trl's image-only multimodal plumbing —
   a project-scoped effort comparable to what Omni-R1's original author
   already did, not a swap — or (b) knowingly dropping audio conditioning
   from GRPO training entirely, a real capability loss given this project's
   whole premise is speech-conditioned negotiation. That is a deliberate
   product decision for a human to make later, not something to schedule
   casually or stumble into via a cleanup pass.
3. **Consequently, replacing the vendored trainer would not have unblocked
   this pilot.** If audio conditioning were rebuilt (option a above), the
   same `transformers`-level crash would very likely resurface, since it's
   triggered by the model + checkpointing combination, not by which trainer
   invokes it. If audio conditioning were dropped (option b), the crash
   would trivially disappear — but that's a capability loss, not a fix.

### 4.2 "Does GPU contention going away make this moot?" — tested directly (2026-08-16), answer: no, not at real pilot scale

Section 4's note above speculates that easing GPU contention "may sidestep
the memory pressure that required `gradient_checkpointing` in the first
place." Both GPUs were free at one point on 2026-08-16 (a rare condition on
this shared machine), which made this speculation directly testable rather
than left as a hopeful aside — tested it before scaling anything further,
same audit-before-implement discipline as the rest of this project.

Added a `--no_gradient_checkpointing` flag to `train_grpo_curriculum.py`
and ran two pilots with it, both on a single fully-free A100 (GPU
contention returned mid-check from another user's job claiming the other
GPU, so the second run below is not a "tested only when nobody else was
around" result):

- **`num_generations=2`, 5 steps: succeeded.** Ran to completion, sane
  reward/KL values, no crash. But peak memory was already **76.97GB /
  81.92GB (94%) immediately after model loading, before any generation
  step ran** — this was a narrow fit, not comfortable headroom.
- **`num_generations=4` (the project's actual intended setting — multiple
  rollouts per scenario is the whole reason GRPO was chosen over PPO
  here), 25 steps: OOM'd on the first training step.** `torch.OutOfMemoryError`,
  **79.13GB / 79.25GB used** — not a near-miss, essentially the entire
  card. `gradient_checkpointing` was not a symptom of unlucky timing or
  contention; at real pilot scale, the model's own memory footprint
  requires it (or some other real memory-reduction lever), full stop.

**Conclusion — this is a conditional, narrow result, not a fix, and should
not be remembered as "solved":** running with `gradient_checkpointing`
disabled is only viable at a reduced `num_generations` unrepresentative of
the actual training regime, and even then only with a full 80GB GPU
uncontended. It does not sidestep the underlying `transformers`-level
checkpointing/audio conflict — it just avoids ever exercising the memory
pressure that made checkpointing necessary in the first place at
`num_generations=2`. At `num_generations=4`, the pilot needs
`gradient_checkpointing` (or an equivalent memory reduction) exactly as
much as it did before this test, and hits the exact same crash section 4
describes the moment it's re-enabled. This closes the "was the original
crash real or just squeezed for memory" question definitively — it's real,
confirmed at real scale, not an artifact of contention.

**Update (2026-08-17): this blocker is resolved — see section 4.3.** The
paragraph above (sections 4/4.1/4.2) is kept as-written because the
investigation trail in it is exactly what led to the fix and is worth
preserving, but its conclusion is superseded: the trl-native route
described in 4.3 avoids the `transformers`-level checkpointing/audio
conflict entirely (never needs `gradient_checkpointing=True` in the first
place, at real pilot scale) rather than fixing it or dropping audio.

### 4.3 Resolution: trl-native `AudioGRPOTrainer` (2026-08-16/17) — blocker closed, with evidence

**Headline result:** a real pilot at `num_generations=4` — the exact
setting that OOM'd at ~100% GPU memory (79.13GB/79.25GB) under Omni-R1's
vendored trainer per section 4.2 — completed cleanly under a new
trl-native trainer with **peak GPU memory 44.9GB/81.9GB (~55%)**, real
headroom to spare, `gradient_checkpointing` off. Roughly half the memory
footprint on the exact configuration that completely failed before.

**Why this works, briefly** (full derivation in the conversation that
produced this; kept short here since the mechanism, not the narrative, is
what matters for future readers): stock `trl`'s `GRPOTrainer` (unlike
Omni-R1's simpler vendored fork) generates a full `generation_batch_size`
batch once, then buffers and replays it through the expensive
forward/backward pass in `per_device_train_batch_size`-sized chunks across
multiple gradient-accumulation steps (`steps_per_generation`), and
additionally chunks the log-prob computation itself
(`_get_per_token_logps_and_entropies`'s own internal `batch_size` loop).
Omni-R1's trainer does neither — it generates and backprops through all
`num_generations` completions simultaneously, which is what actually
required `gradient_checkpointing` to fit at all (and what then hit the
`transformers`-level audio+checkpointing crash in section 4). trl's stock
multimodal support only threads `images`, never `audio`, through this
machinery — so the fix was to add an `audio` path alongside `images`, not
to touch any of the buffering/chunking logic itself.

**Implementation**: `trl_audio_grpo_utils.py` (audio-equivalent of trl's
`prepare_multimodal_messages`) + `trl_audio_grpo_trainer.py`
(`AudioGRPOTrainer`, a subclass of trl's `GRPOTrainer` overriding
`_set_signature_columns_if_needed`, `_generate_single_turn`, `_generate`,
`_generate_and_score_completions`, `_get_per_token_logps_and_entropies`,
and `_compute_loss` to thread `audio`/`input_features`/
`feature_attention_mask`/`num_audio` alongside trl's existing
`images`/`pixel_values` handling) + `train_grpo_curriculum_trl_native.py`
(a parallel runner to `train_grpo_curriculum.py`, **not a replacement for
it at the time this was written** — see "What's still pending" below;
**as of section 4.4 below, it has since become the default and was
renamed to `train_grpo_curriculum.py`, with the old Omni-R1 entrypoint
renamed to `train_grpo_curriculum_omni_r1.py`**). Written independently against
trl's actual installed source (read for understanding, then written from
scratch) — no trl internals imported beyond `GRPOTrainer` itself and a
handful of small public/semi-public utilities.

**The audio row-slicing math (the one place genuinely new bookkeeping was
needed, not just mirroring images) turned out simpler than images', not
harder**: `image_grid_thw`/`pixel_values` need row-counting math because
each image contributes a variable number of patch-rows to a flattened
stack. Confirmed empirically (feeding real variable-length audio batches
through Qwen2.5-Omni's actual processor) that `input_features`/
`feature_attention_mask` instead come out as `[total_clips_in_call, ...]`
— one padded slot per clip, no row-counting needed, just a cumulative sum
over per-example clip counts.

**A silent bug the shape-inspection step caught before it became a
training-time mystery**: that same empirical check also revealed that
Qwen2.5-Omni's processor *rejects* a nested per-example audio list (raises
`ValueError`, "inhomogeneous shape") — unlike the Qwen-VL image processor
trl's original code was written against, which accepts nesting directly.
The generation-path override was passing the nested structure straight
through, mirroring trl's `images` code verbatim; a single-example smoke
test would never have surfaced this, since nesting-vs-flat is invisible at
batch size 1. Fixed by flattening before the processor call.

**Self-correction worth preserving, not just the passing result it led
to**: after wiring the trainer and running the real pilot successfully,
this was initially reported as having validated two specific
generalization concerns (chunk boundaries at `per_device_train_batch_size
> 1`; genuine multi-clip examples, not just the dataset's real 0-or-1-clip
shape). On re-check, **neither was actually true** — the real pilot ran at
`per_device_train_batch_size=1` throughout, and
`NegotiationGRPODataset.__getitem__` (`grpo_curriculum.py:241`) only ever
attaches at most one audio clip per example (confirmed by reading the
dataset-building code directly), so the "real" pilot never exercised
either path. Caught this before it became a false sense of security and
closed it with a dedicated test instead of leaving it asserted-but-
unverified: a synthetic batch with clip counts `[2, 1, 1, 2]` (deliberately
not the real dataset's shape) and `per_device_train_batch_size=2` (chunk
boundaries landing mid-batch, not just at the dataset's own edges), run
through the actual wired `_generate_and_score_completions` — not
hand-built kwargs. Result: `num_audio` came out as `[2, 2, 1, 1, 1, 1, 2,
2]` (correct per-row), `input_features.shape[0] == 12 == sum(num_audio)`,
`advantages.shape == (8,)` — confirmed correct, with a structural reason
it had to be: `num_audio`/`cum_audio` are computed **per row**, not per
unique example, so a chunk boundary landing inside a `num_generations`
-repeat group is indistinguishable, mathematically, from one landing
between two different examples.

**Three environment bugs found and fixed along the way, independent of the
audio work itself — reusable findings for anyone else touching this `trl`
install:**
1. **Every `is_*_available()` helper in this machine's installed `trl`
   returns the raw `(bool, version)` tuple** from `transformers`' internal
   package probe instead of unwrapping it — and a non-empty tuple is
   truthy in Python regardless of its first element, so every
   `if is_X_available():` guard in trl's own code fired unconditionally,
   trying to import optional packages that aren't installed
   (`vllm_ascend`, `weave`) or that are installed-but-broken
   (`llm_blender`, whose install is broken against this `transformers`
   version — the same problem `train_grpo_curriculum.py` already stubs
   around for Omni-R1's old `trl==0.15.2`). Patched generically (wrap
   every `is_*_available` in `trl.import_utils`) rather than stubbing each
   downstream symptom one at a time; audited all 14 affected functions
   individually to confirm exactly which ones changed real behavior (3 did
   — `is_math_verify_available`, `is_vllm_ascend_available`,
   `is_weave_available`, all flipping from incorrectly-`True` to
   correctly-`False`) and that none of the three sit anywhere in
   `AudioGRPOTrainer`'s own code path. Also forced `is_vllm_available()`
   to `False` outright (this project never uses vLLM, and the installed
   `vllm==0.24.0`'s API has drifted from what this `trl` expects) —
   verified separately that `GRPOConfig.use_vllm` (which actually
   determines `self.use_vllm` at runtime) is config-driven, not
   availability-derived, so this can't desync the two.
2. **`super()._prepare_inputs(...)` inside an `AudioGRPOTrainer` method
   copied from trl's `GRPOTrainer` resolves to a different method than the
   original author intended.** trl's own `_generate_single_turn` is
   defined *on* `GRPOTrainer`, where `super()` means "go to `Trainer`."
   Copied verbatim into a subclass of `GRPOTrainer`, `super()` now means
   "go to `GRPOTrainer`" — silently re-entering trl's own overridden
   `_prepare_inputs` (the entire generation-buffering method) instead of
   `Trainer`'s plain tensors-to-device mover. Surfaced as a real recursive
   `TypeError` on the first end-to-end pilot attempt. Fixed with an
   explicit `Trainer._prepare_inputs(self, ...)` call, bypassing the MRO
   ambiguity. Worth remembering as a general hazard: copying any
   `super()`-calling method into a subclass changes what that call
   resolves to.
3. **`AudioGRPOTrainer`/trl's stock `GRPOTrainer` expects the bare causal-LM
   submodule (`model.thinker`), not the full `Qwen2_5OmniForConditionalGeneration`
   wrapper** `train_grpo_curriculum.py` passes to Omni-R1's trainer — the
   full wrapper's `forward()` doesn't accept plain `input_ids=`/
   `attention_mask=` kwargs the way a standard causal LM does. Also:
   `warnings_issued` needs setting on `model.thinker` specifically once
   passed that way, not on the full wrapper.
4. **`negotiation_grpo_reward` (`grpo_curriculum.py`) wasn't defensive
   against non-per-example reward kwargs.** trl's stock
   `_calculate_rewards` injects `reward_kwargs["trainer_state"] = self.state`
   — a single `TrainerState` object, not a per-completion list — which
   Omni-R1's trainer never did. Fixed to skip any kwarg that isn't a
   list/tuple rather than assuming every kwarg is indexable per-completion;
   additive, doesn't change Omni-R1's existing (still-used) behavior.

### 4.4 Training-stability pilots (2026-08-17): 32 steps, then 128 — closing the "plumbing works, but does training itself behave" gap

Section 4.3 established the plumbing (memory, correctness, real forward
passes). It did not establish that real GRPO training over many steps —
real causal-RM rewards, real optimizer updates, not just one
`_generate_and_score_completions` call — behaves sanely. Two pilots, both
`train_grpo_curriculum_trl_native.py`, both using the real audited causal
RM checkpoint (`causal_rm_experiments/full_coverage_2000pairs_lambda_inv_1.5_20260801_141008/checkpoint`)
as the reward, not a stub — that gap was already closed in section 4.3's
pilots, so this was purely a step-count extension.

**First pilot: 32 steps.** Ran cleanly, no OOM, no crash. Reward stayed in
a stable 1.13–1.43 band throughout, entropy stable (0.24–0.29), no
collapse. But the final logged step (32) showed `grad_norm` jumping to
7.11 (steady range elsewhere: ~1.8–3.7) at the same point
`completions/mean_length` hit its run-high (45.66) and reward hit its
run-low (1.129) — three metrics moving together at exactly the pattern
worth watching for (a reward-hacking/instability tell), right at the point
the run happened to end. Correctly flagged as genuinely ambiguous rather
than either dismissed or treated as proof of a problem: 32 steps cannot
distinguish "brief noise that would have self-corrected" from "the start
of real instability."

**Second pilot: 128 steps, identical config, specifically to resolve that
ambiguity.** Result: **the pattern was noise, not a leading indicator** —
confirmed by evidence, not by assumption or by simply not observing a
repeat. Two isolated single-step `grad_norm` spikes occurred (4.41 at step
8, 6.28 at step 24), both self-corrected on the very next logged point,
neither cascaded. The run continued 96 steps past where the 32-step pilot
had ended, and the final 8 logged points (steps 100–128) show `grad_norm`
settled into 1.96–3.02 — a *tighter* band than the first pilot's steady
range, not a wider one. `completions/mean_length` fluctuated 36–45
throughout with **no sustained growth trend** (ended at 37.25, near the
low end — the opposite direction from what a reward-hacking pattern would
predict). Reward stayed in a consistent 1.0–1.44 band across all 32
logging points for the full run, entropy stable 0.22–0.30. Zero
tracebacks, zero OOM, clean exit, `train_runtime: 1937s` (~32 min).

This is a materially stronger result than "ran again and nothing bad
happened": the run didn't just fail to repeat the pattern, it reversed —
completion length trending down instead of up, `grad_norm` tightening
instead of drifting. That distinguishes "noise that happened to not
recur" from "actively stable," which a same-length rerun could not have
shown.

**What this closes:** both the plumbing question (section 4.3) and the
training-dynamics question (this section) now have real, extended-run
evidence behind them — not just a single short pilot. `AudioGRPOTrainer`
is validated for correctness, memory safety, and training stability at
`num_generations=4`.

**What's now unblocked versus what's still pending — precise boundary,
so this doesn't get remembered as "GRPO is done":**
- ✅ The core blocker (audio + `gradient_checkpointing` conflict at real
  `num_generations`) is resolved and verified with real measured memory
  numbers, not just "it imported."
- ✅ All four pieces of `AudioGRPOTrainer` are individually verified, some
  with real forward passes through the actual thinker + LoRA adapter, not
  just unit-level shape checks.
- ✅ **Training stability over 128 real steps with the real causal RM is
  verified** (section 4.4) — reward, entropy, and grad_norm all stable, an
  initial ambiguous signal at step 32 investigated and resolved as noise
  rather than left unresolved.
- ✅ **`AudioGRPOTrainer` is now the default GRPO path (2026-08-17).**
  `train_grpo_curriculum.py` (previously the Omni-R1 entrypoint) now runs
  the trl-native route; the Omni-R1 path was renamed to
  `train_grpo_curriculum_omni_r1.py` and kept working, unchanged, rather
  than deleted.
- ⏳ Removing/deprecating the Omni-R1 vendored-trainer path (the IP-cleanliness
  question from section 4.1) is a separate decision, still not made.

## 5. Artifact index

| Artifact | Path |
|---|---|
| Final checkpoint (recommended for use) | `causal_rm_experiments/full_coverage_2000pairs_lambda_inv_1.5_20260801_141008/checkpoint` |
| Final checkpoint's full audit (JSON + plots) | `causal_rm_experiments/full_coverage_2000pairs_lambda_inv_1.5_20260801_141008/audit/` |
| Heuristic baseline audit | `causal_rm_audit_heuristic_val.json`, `causal_rm_reports/report_heuristic_val.md` |
| Intervention pairs (C1-C5 causal, S1-S3 spurious) | `causal_rm_pairs_{train,val,test}.json` |
| Taxonomy / factor definitions | `causal_rubric_taxonomy.md` |
| Architecture design | `causal_rm_architecture.md` |
| Original integration plan | `causal_rubric_rl_plan.md` |
| RM scorer / training code | `causal_reward_model.py`, `train_causal_rm.py` |
| Audit / experiment tooling | `causal_rm_audit.py`, `run_causal_rm_experiment.py` |
| GRPO integration hooks (opt-in, off by default) | `reward_function.py::compute_reward(..., causal_rm=)`, `grpo_curriculum.py::make_negotiation_grpo_reward`, `train_grpo_curriculum.py --causal_rm_checkpoint` |
| trl-native audio GRPO route (sections 4.3/4.4, blocker-resolution, now default) | `trl_audio_grpo_utils.py`, `trl_audio_grpo_trainer.py` (`AudioGRPOTrainer`), `train_grpo_curriculum.py` (renamed from `train_grpo_curriculum_trl_native.py` — legacy Omni-R1 entrypoint now `train_grpo_curriculum_omni_r1.py`) |
