"""
Audio-modality equivalent of trl's `prepare_multimodal_messages` (trl/data_utils.py),
written independently against trl's public/private API — no trl code copied, no import
of trl internals beyond what any GRPOTrainer subclass would need. Same "read for
understanding, write our own" discipline used for train_sft_v3.py against Omni-R1;
see causal_rm_results.md section 4.1 for why train_grpo_curriculum_omni_r1.py's
direct import of Omni-R1's GRPOTrainer was the thing to avoid repeating.

Why this exists: trl's GRPOTrainer (trl/trainer/grpo_trainer.py) has real, tested
memory-management machinery this project needs (`steps_per_generation` buffering,
`_get_per_token_logps_and_entropies`'s internal batch-size chunking — see
causal_rm_results.md section 4.2's investigation) but its multimodal support only
threads `images`/`pixel_values` through the trainer, never `audio`. This module
provides the audio-side counterpart to the one piece of that support that's a
self-contained utility function rather than something tangled through trainer
internals: turning a plain-string prompt into the structured chat-template content
list the Qwen2.5-Omni processor expects, with audio placeholders inserted.

This is the FIRST of three integration pieces (see causal_rm_results.md section 4.3
for the full plan); it is not yet wired into a trainer subclass.
"""

from typing import Any


def prepare_multimodal_audio_messages(messages: list[dict[str, Any]], num_audio: int) -> None:
    """
    Convert messages into a structured multimodal format if needed, inserting audio
    placeholders. Mirrors trl.data_utils.prepare_multimodal_messages exactly, with
    "image" replaced by "audio" throughout — same placeholder-insertion contract, so
    it can be called from the same call site (mirroring
    `if isinstance(prompt, list): prepare_multimodal_messages(prompt, num_images=...)`
    in trl's `_generate_single_turn`) with `num_audio=len(audio_list)` in place of
    `num_images=len(image_list)`.

    Each message's content is transformed from a raw string into a list of typed parts.
    The first user message is prefixed with `num_audio` audio placeholders, all other
    user and assistant messages are wrapped as plain text entries. Messages whose
    content is already a list (i.e., already prepared — e.g. by a caller that also
    handles images) are left untouched, exactly as trl's version does.

    Args:
        messages: Messages with "role" and "content"; content may be a raw string
            before transformation.
        num_audio: Number of audio placeholders to add to the first user message.

    Example:
        # Input
        [
            {"role": "user", "content": "What does the buyer sound like?"},
            {"role": "assistant", "content": "Frustrated but still engaged."},
        ]
        # Output (num_audio=2)
        [
            {"role": "user", "content": [
                {"type": "audio"}, {"type": "audio"},
                {"type": "text", "text": "What does the buyer sound like?"},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Frustrated but still engaged."},
            ]},
        ]
    """
    audio_included = False
    for message in messages:
        if message["role"] == "system":
            if isinstance(message["content"], str):
                message["content"] = [{"type": "text", "text": message["content"]}]
        elif message["role"] == "user":
            if isinstance(message["content"], str) and not audio_included:
                placeholders = [{"type": "audio"}] * num_audio
                message["content"] = [*placeholders, {"type": "text", "text": message["content"]}]
                audio_included = True
            elif isinstance(message["content"], str) and audio_included:
                message["content"] = [{"type": "text", "text": message["content"]}]
        elif message["role"] == "assistant":
            if isinstance(message["content"], str):
                message["content"] = [{"type": "text", "text": message["content"]}]
        else:
            raise ValueError(f"Invalid role in message: {message['role']}. Expected 'user', 'assistant', or 'system'.")
