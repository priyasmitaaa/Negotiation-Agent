"""
AudioGRPOTrainer: a thin subclass of trl.GRPOTrainer that threads an `audio` modality
through the same plumbing trl already uses for `images`, so this project can train
GRPO on Qwen2.5-Omni with real audio input while keeping trl's own generation
buffering (`steps_per_generation`) and per-token-logps chunking (see
causal_rm_results.md section 4.2) fully intact and untouched.

Written independently against trl's public/private API — no trl code imported beyond
`GRPOTrainer` itself and the few small utilities trl already exports. Where a method
had to be fully overridden (Python doesn't support partial method overrides), the
body below started as a direct read of trl's own method and was then modified only at
the points marked "AUDIO:". This is the same read-for-understanding-then-write-our-own
discipline used for train_sft_v3.py against Omni-R1 (see causal_rm_results.md section
4.1) — NOT the train_grpo_curriculum_omni_r1.py pattern of importing and running
someone else's GRPOTrainer class wholesale.

STATUS (see causal_rm_results.md section 4.3 for the full plan):
  - Piece 1 (prepare_multimodal_audio_messages): DONE, trl_audio_grpo_utils.py
  - Piece 2 (this file, generation path):        DONE
  - Piece 3 (_get_per_token_logps_and_entropies
             audio row-slicing branch):          NOT YET DONE — its own checkpoint,
                                                  see NotImplementedError below.
  - Piece 4 (this file, signature columns +
             _generate_and_score_completions +
             _compute_loss audio threading):     DONE

This class will raise NotImplementedError if actually run end-to-end right now,
because _get_per_token_logps_and_entropies (inherited unmodified from trl) does not
yet accept `input_features`/`feature_attention_mask`/`num_audio` — piece 3's job.
That's intentional: wiring pieces 2+4 correctly first, in isolation, is easier to get
right and easier to review than doing all three at once.

DELIBERATELY NOT SUPPORTED (unused by this project, not silently missing):
  - vLLM generation (`use_vllm`) and paged generation (`use_transformers_paged`) are
    not given audio-aware branches in `_generate_single_turn` below — only the
    "Regular generation path" (plain `model.generate()`, what train_grpo_curriculum.py
    already uses) is overridden. If this project ever adopts vLLM for GRPO rollout
    speed, those branches need the same `audio` treatment applied to `images` there.
  - Mixing images AND audio in the same example is not handled — this project only
    ever has audio, never images, so `images` is passed through as None/unused.
"""

import importlib.machinery
import sys
import types as _types
from typing import Any, Optional, Union

import torch

# Environment shim (2026-08-16): trl.extras.vllm_client does
# `if is_vllm_ascend_available(): from vllm_ascend... import PyHcclCommunicator`.
# On this machine's trl version, is_vllm_ascend_available() returns the tuple
# (False, None) instead of a plain bool — a non-empty tuple is truthy in Python, so
# the guard always fires and tries to import a package we don't have (no Ascend/NPU
# hardware here, only NVIDIA A100s). This is a real upstream bug in trl, not a
# misconfiguration on our end. Same fix pattern as the llm_blender stub already in
# train_grpo_curriculum_omni_r1.py: stub the missing module rather than patch trl
# itself.
# Environment shim (2026-08-16), root cause: every `is_*_available()` helper in
# trl.import_utils on this machine's installed trl version returns the raw
# `(bool, version)` tuple from transformers' internal package-probe instead of
# unwrapping just the bool — e.g. is_vllm_ascend_available() returns (False, None).
# A non-empty tuple is truthy in Python regardless of its first element, so every
# `if is_X_available():` guard in trl's own code fires unconditionally, trying to
# import optional packages (vllm_ascend, weave, ...) that aren't installed and
# aren't needed by this project. Rather than stub each such package one at a time
# as it's hit (the vllm_ascend/llm_blender stubs below predate this fix and are
# kept for defense-in-depth), patch the helpers themselves once, before any trl
# submodule binds them via `from ..import_utils import is_X_available`.
import trl.import_utils as _trl_import_utils

for _name in dir(_trl_import_utils):
    if _name.startswith("is_") and _name.endswith("_available"):
        _fn = getattr(_trl_import_utils, _name)
        if callable(_fn):
            def _make_bool_wrapper(fn):
                def _wrapped(*args, **kwargs):
                    result = fn(*args, **kwargs)
                    return bool(result[0]) if isinstance(result, tuple) else bool(result)
                return _wrapped
            setattr(_trl_import_utils, _name, _make_bool_wrapper(_fn))

# Audit (2026-08-16) of every is_*_available() this patch touches, comparing the
# old buggy tuple-truthy result against the correct unwrapped bool. Only three
# actually flip value; the rest were already effectively correct (either the buggy
# tuple's first element was True anyway, or the function didn't return a tuple):
#   is_deepspeed_available    (True, None)  -> True  (unchanged)
#   is_fastapi_available      (True, None)  -> True  (unchanged)
#   is_joblib_available       (True, None)  -> True  (unchanged)
#   is_liger_kernel_available  False         -> False (unchanged, not a tuple)
#   is_llm_blender_available  (True, None)  -> True  (unchanged; the package IS
#                                              importable, its broken transformers
#                                              ABI is a separate problem handled by
#                                              the sys.modules stub below)
#   is_math_verify_available  (False, None) -> False (CHANGED, was buggy-True).
#                                              Only gates trl/rewards/accuracy_rewards.py
#                                              (an unrelated math-reward leaf module,
#                                              never imported by GRPOTrainer's core
#                                              init or by anything in this file).
#   is_mergekit_available     (True, None)  -> True  (unchanged)
#   is_pydantic_available     (True, None)  -> True  (unchanged)
#   is_requests_available     (True, None)  -> True  (unchanged)
#   is_unsloth_available      (True, None)  -> True  (unchanged)
#   is_uvicorn_available      (True, None)  -> True  (unchanged)
#   is_vllm_available          True         -> True  (unchanged here; forced to
#                                              False separately below, see next block)
#   is_vllm_ascend_available  (False, None) -> False (CHANGED, was buggy-True).
#                                              Only gates the vllm_ascend import in
#                                              trl.extras.vllm_client, already stubbed
#                                              below independently as defense-in-depth.
#   is_weave_available        (False, None) -> False (CHANGED, was buggy-True).
#                                              Only gates `import weave` /
#                                              WeaveTraceCallback in trl.trainer.callbacks
#                                              (an optional W&B logging integration this
#                                              project's GRPOConfig never enables).
# All three real changes flip an optional-integration guard from "pretends available"
# to "correctly unavailable" for features this project's training path never touches
# (no weave callback, no vllm_ascend/NPU hardware, no math-accuracy reward). None of
# the three sit inside AudioGRPOTrainer's own generation/loss/reward call path.

# is_vllm_available() now correctly evaluates to True (vllm really is installed:
# 0.24.0) — but this project's AudioGRPOTrainer deliberately never uses vLLM (see
# module docstring), and the installed vllm==0.24.0's API has drifted from what
# this trl version's vLLM-only import block expects (`GuidedDecodingParams` moved),
# an unrelated third-party version mismatch this project has no reason to chase.
# Force it False so trl skips that import block entirely, same as if vllm weren't
# installed at all.
#
# Verified this doesn't silently change other trainer behavior: GRPOTrainer.__init__
# sets `self.use_vllm = args.use_vllm` directly from GRPOConfig — it does NOT
# consult is_vllm_available() to decide self.use_vllm. GRPOConfig.use_vllm defaults
# to False and train_grpo_curriculum.py never sets it, so self.use_vllm is False
# here regardless of this patch. This patch only prevents the top-of-file
# `if is_vllm_available(): from vllm import LLM, SamplingParams; from
# vllm.sampling_params import GuidedDecodingParams` from executing (and crashing on
# the GuidedDecodingParams API drift) — it cannot desync self.use_vllm from whether
# LLM/SamplingParams got imported, because both are driven by this same forced value.
_trl_import_utils.is_vllm_available = lambda: False

if "vllm_ascend" not in sys.modules:
    for _name in (
        "vllm_ascend",
        "vllm_ascend.distributed",
        "vllm_ascend.distributed.device_communicators",
        "vllm_ascend.distributed.device_communicators.pyhccl",
    ):
        _stub = _types.ModuleType(_name)
        _stub.__spec__ = importlib.machinery.ModuleSpec(_name, loader=None)
        _stub.__path__ = []  # marks it as a package, so submodule find_spec() calls succeed
        sys.modules[_name] = _stub
    sys.modules["vllm_ascend.distributed.device_communicators.pyhccl"].PyHcclCommunicator = object

# Same broken-optional-dependency situation, already documented and stubbed in
# train_grpo_curriculum_omni_r1.py: trl.trainer.judges imports llm_blender (an LLM
# pairwise-judge feature this project never uses) purely to feed
# SyncRefModelCallback's sibling code; the system-wide llm_blender install is
# itself broken against this transformers version (unrelated ABI/API mismatch:
# `TRANSFORMERS_CACHE` was removed from transformers.utils.hub). Stub it out
# before importing trl, identically to train_grpo_curriculum_omni_r1.py.
if "llm_blender" not in sys.modules:
    _llm_blender_stub = _types.ModuleType("llm_blender")
    _llm_blender_stub.__spec__ = importlib.machinery.ModuleSpec("llm_blender", loader=None)
    sys.modules["llm_blender"] = _llm_blender_stub

from accelerate.utils import gather_object
from trl.data_utils import is_conversational, maybe_apply_chat_template
from trl.extras.profiling import profiling_context
from trl.models import unwrap_model_for_generation
from transformers import Trainer
from trl.trainer.grpo_trainer import GRPOTrainer
from trl.trainer.utils import entropy_from_logits, nanmax, nanmin, nanstd, pad, selective_log_softmax

from trl_audio_grpo_utils import prepare_multimodal_audio_messages


class AudioGRPOTrainer(GRPOTrainer):
    """GRPOTrainer with an `audio` modality path alongside (not replacing) `images`."""

    def _set_signature_columns_if_needed(self):
        # AUDIO: trl's version only allowlists "image"/"images". Add "audio" so
        # Trainer's remove_unused_columns step doesn't silently drop it before it
        # ever reaches _generate_and_score_completions.
        if self._signature_columns is None:
            self._signature_columns = ["prompt", "image", "images", "audio"]

    # ------------------------------------------------------------------
    # Piece 2: generation path
    # ------------------------------------------------------------------

    def _generate_single_turn(self, prompts: list[str], audio: Optional[list]):
        """Regular-generation-path-only counterpart of GRPOTrainer._generate_single_turn.

        AUDIO: mirrors trl's method with `images` replaced by `audio` and
        `prepare_multimodal_messages` replaced by `prepare_multimodal_audio_messages`.
        The vLLM and transformers-paged-generation branches present in trl's original
        are intentionally omitted — see module docstring "DELIBERATELY NOT SUPPORTED".
        """
        device = self.accelerator.device

        kwargs = {}
        if audio is not None:
            # AUDIO: unlike trl's `images` (a Qwen-VL processor that accepts nested
            # per-example lists directly), Qwen2.5-Omni's audio processor requires a
            # FLAT list of clips — confirmed empirically (2026-08-16): passing a
            # nested per-example list raises "ValueError: setting an array element
            # with a sequence... inhomogeneous shape". Flatten here; num_audio-based
            # bookkeeping in _generate_and_score_completions/piece 3 recovers the
            # per-example boundaries from the resulting flat input_features stack.
            flat_audio = [clip for audio_list in audio for clip in audio_list]
            kwargs = {"audio": flat_audio}
            for prompt, audio_list in zip(prompts, audio):
                if isinstance(prompt, list):  # conversational data
                    prepare_multimodal_audio_messages(prompt, num_audio=len(audio_list))

        prompts_text = [
            maybe_apply_chat_template({"prompt": prompt}, self.processing_class)["prompt"] for prompt in prompts
        ]

        if audio is not None:
            prompt_inputs = self.processing_class(text=prompts_text, padding=True, return_tensors="pt", **kwargs)
            # AUDIO: NOT super()._prepare_inputs(...) — trl's original method body
            # this was copied from is defined ON GRPOTrainer, where super() means
            # "skip to Trainer". Here, inside AudioGRPOTrainer, plain super() would
            # resolve to GRPOTrainer itself (re-entering the whole generation-batch
            # buffering machinery recursively) rather than Trainer's plain
            # tensors-to-device mover. Call Trainer's version explicitly, bypassing
            # GRPOTrainer in the MRO — confirmed necessary (2026-08-16) after this
            # exact mistake produced a real recursive-call TypeError during the
            # first end-to-end pilot run.
            prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
            # AUDIO: this dict comprehension is already modality-agnostic in trl's
            # original — it just keeps whatever keys the processor returned besides
            # input_ids/attention_mask, which for Qwen2.5-Omni's processor with
            # audio=... means input_features / feature_attention_mask show up here
            # automatically, no extra code needed.
            forward_kwargs = {k: v for k, v in prompt_inputs.items() if k not in ["input_ids", "attention_mask"]}
        else:
            forward_kwargs = {}

        if self.use_vllm or self.use_transformers_paged:
            raise NotImplementedError(
                "AudioGRPOTrainer only implements the regular (non-vLLM, non-paged) generation "
                "path. See trl_audio_grpo_trainer.py module docstring."
            )

        # Regular generation path (mirrors trl's `else:` branch in _generate_single_turn)
        generate_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            max_length=self.max_prompt_length,
            truncation=True,
            add_special_tokens=False,
            **kwargs,
        )
        generate_inputs = Trainer._prepare_inputs(self, generate_inputs)  # AUDIO: see comment above

        with (
            profiling_context(self, "transformers.generate"),
            unwrap_model_for_generation(
                self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
            ) as unwrapped_model,
            torch.no_grad(),
        ):
            prompt_completion_ids = unwrapped_model.generate(
                **generate_inputs, generation_config=self.generation_config, disable_compile=True
            )

        prompt_ids, prompt_mask = generate_inputs["input_ids"], generate_inputs["attention_mask"]
        prompt_length = prompt_ids.size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]

        is_eos = completion_ids == self.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        prompt_ids = [p[m].tolist() for p, m in zip(prompt_ids, prompt_mask.bool())]
        completion_ids = [c[m].tolist() for c, m in zip(completion_ids, completion_mask.bool())]
        logprobs = None

        return prompt_ids, completion_ids, logprobs, forward_kwargs

    def _generate(self, prompts: list[str], audio: Optional[list]):
        """AUDIO: identical to trl's _generate, `images` param renamed `audio`."""
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompt_ids, completion_ids, logprobs, forward_kwargs = self._generate_single_turn(prompts, audio)

        prompt_lengths = torch.tensor([len(ids) for ids in prompt_ids], device=device)
        completion_lengths = torch.tensor([len(ids) for ids in completion_ids], device=device)
        agg_prompt_lengths = self.accelerator.gather(prompt_lengths)
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        total_prompt_tokens = agg_prompt_lengths.sum()
        total_completion_tokens = agg_completion_lengths.sum()

        if mode == "train":
            self.state.num_input_tokens_seen += (total_prompt_tokens + total_completion_tokens).item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        eos_and_pad = [self.eos_token_id, self.pad_token_id]
        is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids], device=device)
        agg_is_truncated = self.accelerator.gather(is_truncated)
        self._metrics[mode]["completions/clipped_ratio"].append(agg_is_truncated.float().mean().item())
        term_completion_lengths = agg_completion_lengths[~agg_is_truncated]
        if len(term_completion_lengths) == 0:
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

        return prompt_ids, completion_ids, total_completion_tokens, logprobs, forward_kwargs

    # ------------------------------------------------------------------
    # Piece 4: signature/extraction threading
    # ------------------------------------------------------------------

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        """AUDIO: mirrors trl's method with an `audio` extraction branch alongside
        (not replacing) `images`, and `num_audio` threaded everywhere `num_images` is.
        """
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompts = [x["prompt"] for x in inputs]

        if "images" in inputs[0]:
            images = [example.get("images") for example in inputs]
        elif "image" in inputs[0]:
            images = [[example.get("image")] if example.get("image") is not None else None for example in inputs]
        else:
            images = None
        if images is not None and all(img_list == [] for img_list in images):
            images = None

        # AUDIO: new extraction branch, mirrors the images one above.
        #
        # AUDIO: NegotiationGRPODataset.__getitem__ (grpo_curriculum.py:241) returns
        # example["audio"] as a BARE array when there's exactly one clip (unwrapped,
        # written for Omni-R1's trainer, which assumes one clip per example — this
        # dataset only ever attaches at most one prior-buyer-turn clip, so that
        # assumption happens to hold for Omni-R1 today) and only returns a genuine
        # list for zero or multiple clips. Confirmed by reading grpo_curriculum.py
        # and Omni-R1/src/trainer/grpo_trainer.py directly (2026-08-16) rather than
        # assuming the dataset already matches AudioGRPOTrainer's list-per-example
        # convention — changing the dataset itself would break Omni-R1's still-live
        # `audios = [x["audio"] for x in inputs]` path, so normalize defensively
        # here instead. A raw numpy array also can't be compared with `== []`
        # (raises "truth value of an array is ambiguous"), so this must run before
        # any such check, not after.
        if "audio" in inputs[0]:
            audio = []
            for example in inputs:
                a = example.get("audio")
                if a is None:
                    audio.append([])
                elif isinstance(a, list):
                    audio.append(a)
                else:  # bare array (single-clip case)
                    audio.append([a])
        else:
            audio = None
        if audio is not None and all(len(a_list) == 0 for a_list in audio):
            audio = None

        if images is not None and audio is not None:
            raise NotImplementedError(
                "AudioGRPOTrainer does not support mixing images and audio in the same "
                "batch (this project never needs to)."
            )

        (
            prompt_ids_list,
            completion_ids_list,
            num_items_in_batch,
            sampling_per_token_logps_list,
            forward_kwargs,
        ) = self._generate(prompts, audio if audio is not None else images)

        prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left")
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right")
        if sampling_per_token_logps_list is not None:
            sampling_per_token_logps = [torch.tensor(logps, device=device) for logps in sampling_per_token_logps_list]
            sampling_per_token_logps = pad(sampling_per_token_logps, padding_value=0.0, padding_side="right")
        else:
            sampling_per_token_logps = None

        if self.mask_truncated_completions:
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()

        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        if "token_type_ids" in forward_kwargs:
            token_type_ids = forward_kwargs["token_type_ids"]
            forward_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids.new_zeros(completion_ids.shape)], dim=1
            )

        logits_to_keep = completion_ids.size(1)
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        num_images = [len(img_list) for img_list in images] if images is not None else None
        # AUDIO: analogous per-sample count, consumed by piece 3's row-slicing branch.
        num_audio = [len(a_list) for a_list in audio] if audio is not None else None

        with torch.no_grad():
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self.args.gradient_accumulation_steps % generate_every != 0 or (
                self.use_vllm and self.vllm_importance_sampling_correction
            ):
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    num_images=num_images,
                    num_audio=num_audio,
                    **forward_kwargs,
                )
            else:
                old_per_token_logps = None

            if self.use_vllm and self.vllm_importance_sampling_correction:
                importance_sampling_ratio = torch.exp(old_per_token_logps - sampling_per_token_logps)
                importance_sampling_ratio = torch.clamp(
                    importance_sampling_ratio, max=self.vllm_importance_sampling_cap
                )

            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                        num_images=num_images,
                        num_audio=num_audio,
                        **forward_kwargs,
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=batch_size,
                            num_images=num_images,
                            num_audio=num_audio,
                            **forward_kwargs,
                        )
            else:
                ref_per_token_logps = None

        prompts_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards

        if self.scale_rewards in ["group", "none"]:
            std_rewards = rewards.view(-1, self.num_generations).std(dim=1)
            std_rewards = std_rewards.repeat_interleave(self.num_generations, dim=0)
        elif self.scale_rewards == "batch":
            std_rewards = rewards.std().expand_as(rewards)
        else:
            raise ValueError(
                f"Invalid value for scale_rewards: {self.scale_rewards}. Must be one of 'batch', 'group', or 'none'."
            )

        is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))
        if self.scale_rewards != "none":
            advantages = advantages / (std_rewards + 1e-4)

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()
        advantages = advantages[process_slice]

        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_func_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_func_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(all_process_advantages.tolist())

        if images is not None:
            self._logs["images"].extend(gather_object(images))
        # AUDIO: no equivalent audio logging yet (trl doesn't wandb.Audio-log images
        # either without extra code) — deferred, not needed for training correctness.

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "num_items_in_batch": num_items_in_batch,
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if self.use_vllm and self.vllm_importance_sampling_correction:
            output["importance_sampling_ratio"] = importance_sampling_ratio
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        if "pixel_values" in forward_kwargs:
            output["pixel_values"] = forward_kwargs["pixel_values"]
        if "image_grid_thw" in forward_kwargs:
            output["image_grid_thw"] = forward_kwargs["image_grid_thw"]
        if "pixel_attention_mask" in forward_kwargs:
            output["pixel_attention_mask"] = forward_kwargs["pixel_attention_mask"]
        if "image_sizes" in forward_kwargs:
            output["image_sizes"] = forward_kwargs["image_sizes"]
        if "token_type_ids" in forward_kwargs:
            output["token_type_ids"] = forward_kwargs["token_type_ids"]
        # AUDIO: analogous passthrough for Qwen2.5-Omni's audio processor outputs.
        if "input_features" in forward_kwargs:
            output["input_features"] = forward_kwargs["input_features"]
        if "feature_attention_mask" in forward_kwargs:
            output["feature_attention_mask"] = forward_kwargs["feature_attention_mask"]
        if images is not None:
            output["num_images"] = num_images
        if audio is not None:
            output["num_audio"] = num_audio
        return output

    # ------------------------------------------------------------------
    # Piece 3: audio row-slicing in the chunked log-prob computation
    # ------------------------------------------------------------------

    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        pixel_values=None,
        image_grid_thw=None,
        num_images=None,
        pixel_attention_mask=None,
        image_sizes=None,
        token_type_ids=None,
        input_features=None,
        feature_attention_mask=None,
        num_audio=None,
    ):
        """AUDIO: mirrors trl's method with a new audio branch alongside the image
        branches. Confirmed empirically (2026-08-16, see shape-inspection notes
        above _generate_single_turn) that Qwen2.5-Omni's audio processor returns
        `input_features`/`feature_attention_mask` shaped [total_clips_in_call, ...],
        one slot per audio clip in flat order — NOT a variable-rows-per-item stack
        like image_grid_thw/pixel_values. This means audio slicing needs only a
        cumulative-sum over per-example clip counts (`num_audio`) to find each
        chunk's clip-index range, unlike images' extra image_grid_thw.prod(dim=-1)
        row-counting step — the audio case is the *simpler* one of the two.
        """
        batch_size = batch_size or input_ids.size(0)
        all_logps = []
        all_entropies = []
        for start in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[start : start + batch_size]
            attention_mask_batch = attention_mask[start : start + batch_size]

            model_inputs = {"input_ids": input_ids_batch, "attention_mask": attention_mask_batch}
            if image_grid_thw is not None and pixel_values is not None:
                rows_per_image = image_grid_thw.prod(dim=-1)
                rows_per_sample = torch.split(rows_per_image, num_images)
                rows_per_sample = torch.stack([s.sum() for s in rows_per_sample])
                cum_rows = torch.cat([torch.tensor([0], device=rows_per_sample.device), rows_per_sample.cumsum(0)])
                row_start, row_end = cum_rows[start].item(), cum_rows[start + batch_size].item()
                model_inputs["pixel_values"] = pixel_values[row_start:row_end]
                cum_imgs = torch.tensor([0] + num_images).cumsum(0)
                img_start, img_end = cum_imgs[start], cum_imgs[start + batch_size]
                model_inputs["image_grid_thw"] = image_grid_thw[img_start:img_end]
            elif pixel_values is not None:
                model_inputs["pixel_values"] = pixel_values[start : start + batch_size]
            if pixel_attention_mask is not None:
                model_inputs["pixel_attention_mask"] = pixel_attention_mask[start : start + batch_size]
            if image_sizes is not None:
                model_inputs["image_sizes"] = image_sizes[start : start + batch_size]
            if token_type_ids is not None:
                model_inputs["token_type_ids"] = token_type_ids[start : start + batch_size]
            # AUDIO: no row-counting needed — one slot per clip, so a direct
            # cumsum over num_audio (per-example clip counts) gives the clip-index
            # range for this chunk of examples.
            if input_features is not None and num_audio is not None:
                cum_audio = torch.tensor([0] + list(num_audio)).cumsum(0)
                audio_start, audio_end = cum_audio[start].item(), cum_audio[start + batch_size].item()
                model_inputs["input_features"] = input_features[audio_start:audio_end]
                if feature_attention_mask is not None:
                    model_inputs["feature_attention_mask"] = feature_attention_mask[audio_start:audio_end]

            if "logits_to_keep" in self.model_kwarg_keys:
                model_inputs["logits_to_keep"] = logits_to_keep + 1

            model_inputs["use_cache"] = False

            logits = model(**model_inputs).logits
            logits = logits[:, :-1, :]
            logits = logits[:, -logits_to_keep:, :]
            logits = logits / self.temperature

            completion_ids = input_ids_batch[:, -logits_to_keep:]
            logps = selective_log_softmax(logits, completion_ids)
            all_logps.append(logps)

            if compute_entropy:
                with torch.no_grad():
                    entropies = entropy_from_logits(logits)
                all_entropies.append(entropies)

        logps = torch.cat(all_logps, dim=0)
        entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
        return logps, entropies

    # ------------------------------------------------------------------
    # Piece 4 (continued): training-time forward pass
    # ------------------------------------------------------------------

    def _compute_loss(self, model, inputs):
        """AUDIO: identical to trl's _compute_loss, with input_features /
        feature_attention_mask / num_audio threaded into the single
        _get_per_token_logps_and_entropies call alongside the image kwargs.
        This is the actual gradient-carrying forward pass — without this override,
        audio would silently never reach the model during the real training step
        even if pieces 2's generation path worked, because inputs.get("input_features")
        would just be None here.
        """
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            num_images=inputs.get("num_images"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
            token_type_ids=inputs.get("token_type_ids"),
            input_features=inputs.get("input_features"),
            feature_attention_mask=inputs.get("feature_attention_mask"),
            num_audio=inputs.get("num_audio"),
        )

        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(entropies, completion_mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        advantages = inputs["advantages"]
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(
                f"Unknown importance sampling level: {self.importance_sampling_level}. Possible values are 'token' "
                "and 'sequence'."
            )

        coef_1 = torch.exp(log_importance_weights)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask

        if self.use_vllm and self.vllm_importance_sampling_correction:
            per_token_loss = per_token_loss * inputs["importance_sampling_ratio"]

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type == "grpo":
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "dapo":
            normalizer = inputs["num_items_in_batch"] / self.accelerator.num_processes
            loss = (per_token_loss * completion_mask).sum() / normalizer
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Log the metrics (identical to trl's original — copied for completeness)
        mode = "train" if self.model.training else "eval"
        completion_token_count = completion_mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:
                return x.mean()
            return (x * completion_mask).sum() / completion_token_count

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())

        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = masked_batch_mean(is_low_clipped.float())
        high_clip = masked_batch_mean(is_high_clipped.float())
        clip_ratio = masked_batch_mean(is_region_clipped.float())

        gathered_low_clip = self.accelerator.gather(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        gathered_clip_ratio = self.accelerator.gather(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())

        return loss
