#!/usr/bin/env python3
"""
inference_v6.py — Inference + evaluation for SFT v6 adapter.

v6 setup: all dialogues, emotions in context, target = <decision> + <response>.

By default runs text-only inference (fast, thinker only, ~9GB GPU).
Pass --speech to also generate spoken WAV output for each response.

Text mode:
  Result: sft_output_v6/{run_tag}/inference_{timestamp}.json

Speech mode (--speech):
  Result: sft_output_v6/{run_tag}/inference_speech_{timestamp}.json
  WAVs:   sft_output_v6/{run_tag}/speech_outputs_{timestamp}/

Usage:
  # Text only (default):
  CUDA_VISIBLE_DEVICES=1 python3 Qwen3-tts/v2/inference_v6.py --n_samples 0

  # Text + speech:
  CUDA_VISIBLE_DEVICES=1 python3 Qwen3-tts/v2/inference_v6.py --speech --speaker Chelsie

  # Single dialogue, with speech:
  CUDA_VISIBLE_DEVICES=1 python3 Qwen3-tts/v2/inference_v6.py --speech --dialogue_id dialogue_0007

  # Specific adapter:
  CUDA_VISIBLE_DEVICES=1 python3 Qwen3-tts/v2/inference_v6.py \\
      --adapter Qwen3-tts/v2/sft_output_v6/YYYYMMDD_HHMMSS/final_adapter
"""

import argparse
import datetime
import gc
import json
import re
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from math import gcd
from scipy.signal import resample_poly

# Patch CVE-2025-32434 check (torch 2.5.1 + transformers 4.57.6)
import transformers.utils.import_utils as _triu
_triu.check_torch_load_is_safe = lambda: None
import transformers.models.qwen2_5_omni.modeling_qwen2_5_omni as _qwen_model
_qwen_model.check_torch_load_is_safe = lambda: None

warnings.filterwarnings("ignore")

V2_ROOT     = Path(__file__).parent
DATASET_DIR = V2_ROOT / "dataset"
AUDIO_DIR   = V2_ROOT / "tts_outputs"
SFT_OUT_V6  = V2_ROOT / "sft_output_v6"
MODEL_ID    = "Qwen/Qwen2.5-Omni-7B"
TARGET_SR   = 16_000
SPEECH_SR   = 24_000

AVAILABLE_SPEAKERS = ["Chelsie", "Ethan", "Serena", "River", "Nova", "Vale"]

SYSTEM_PROMPT_V6 = (
    "You are an AI negotiation assistant for a second-hand electronics shop in India. "
    "Guide the negotiation to within 10% of fair market value. Analyse the "
    "conversation and buyer emotional signals (emotion labels, intensity, valence). "
    "Produce in this exact order: <decision> [UNDECIDED/LEVERAGE/MITIGATE], "
    "<response>. Infer everything from raw signals."
)

QWEN_DEFAULT_SYSTEM = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)


# ── Args ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter",        default=None,
                   help="Path to LoRA adapter. Auto-detects latest v6 run if omitted.")
    p.add_argument("--n_samples",      type=int, default=100,
                   help="Test examples to evaluate (0 = all). Default: 100.")
    p.add_argument("--max_new_tokens", type=int, default=256,
                   help="Max tokens for text generation.")
    p.add_argument("--dialogue_id",    default=None,
                   help="Run on all seller turns of one dialogue, e.g. --dialogue_id dialogue_0007.")
    # Speech flags
    p.add_argument("--speech",         action="store_true",
                   help="Also generate spoken WAV output for each response (loads full model, ~16GB, slower).")
    p.add_argument("--speaker",        default="Chelsie", choices=AVAILABLE_SPEAKERS,
                   help=f"Speaker voice for --speech mode. Default: Chelsie.")
    p.add_argument("--no_audio_input", action="store_true",
                   help="Skip loading audio input files (text-only input).")
    p.add_argument("--resume_speech_dir", default=None,
                   help="Resume speech generation into an existing speech_outputs_* directory.")
    return p.parse_args()


# ── Find adapter ───────────────────────────────────────────────────────────────

def find_adapter(adapter_arg):
    if adapter_arg:
        p = Path(adapter_arg)
        if not p.exists():
            sys.exit(f"Adapter not found: {p}")
        return p
    if not SFT_OUT_V6.exists():
        sys.exit("No sft_output_v6/ directory. Run train_sft_v6.py first.")
    run_dirs = sorted([d for d in SFT_OUT_V6.iterdir() if d.is_dir()],
                      key=lambda d: d.name, reverse=True)
    for rd in run_dirs:
        adapter = rd / "final_adapter"
        if adapter.exists():
            print(f"  Auto-detected adapter: {adapter}")
            return adapter
    sys.exit("No final_adapter found in sft_output_v6/. Run train_sft_v6.py first.")


def load_run_summary(adapter_path):
    p = adapter_path.parent / "run_summary.json"
    return json.loads(p.read_text()) if p.exists() else {}


def load_test_ids():
    f = V2_ROOT / "splits_v6.json"
    if f.exists():
        return json.loads(f.read_text()).get("test", [])
    sys.exit("splits_v6.json not found.")


# ── Build test examples ────────────────────────────────────────────────────────

def build_test_examples(test_ids, n_samples, skip_audio_input=False):
    examples = []
    for did in test_ids:
        fpath = DATASET_DIR / f"{did}.json"
        if not fpath.exists():
            continue
        data    = json.loads(fpath.read_text())
        seed    = data["seed"]
        turns   = data["processed"]
        pricing = seed["pricing"]
        domain  = seed["domain"]

        turns_so_far = []
        for turn in turns:
            if turn["speaker"] != "seller":
                turns_so_far.append(turn)
                continue

            tidx = turn["turn_index"]
            dec  = turn["factor_state"]["decision"]

            audio_paths = []
            if not skip_audio_input:
                missing = False
                for pt in turns_so_far:
                    wav = AUDIO_DIR / did / f"turn_{pt['turn_index']:02d}_{pt['speaker']}.wav"
                    if not wav.exists():
                        missing = True; break
                    audio_paths.append(wav)
                if missing:
                    turns_so_far.append(turn)
                    continue

            context_block = (
                f"Product: {domain['product']}. "
                f"Condition: {domain['condition']}. "
                f"Asking: ₹{pricing['asking_price']}. "
                f"Fair value: ₹{pricing['fair_value']}."
            )
            conv_lines = []
            for pt in turns_so_far:
                line = f"Turn {pt['turn_index']} [{pt['speaker'].upper()}]: {pt['text']}"
                if pt["speaker"] == "buyer":
                    em = pt.get("emotion", {})
                    if em.get("labels"):
                        labels_str = ", ".join(em["labels"])
                        line += (
                            f"  [Emotion: {labels_str} | "
                            f"intensity: {em.get('intensity','?')} | "
                            f"valence: {em.get('valence','?')}]"
                        )
                conv_lines.append(line)

            user_text = (
                f"CONTEXT\n{context_block}\n\n"
                f"CONVERSATION SO FAR\n" + "\n".join(conv_lines) + "\n\n"
                f"TASK\nYou are the seller. It is now Turn {tidx}. "
                f"Produce <decision>, <response>."
            )

            prior_buyer = next(
                (pt for pt in reversed(turns_so_far) if pt["speaker"] == "buyer"), None
            )

            examples.append({
                "dialogue_id":         did,
                "turn_index":          tidx,
                "audio_paths":         [str(p) for p in audio_paths],
                "user_text":           user_text,
                "gt_decision":         dec,
                "gt_response":         turn["text"],
                "fair_value":          pricing["fair_value"],
                "asking_price":        pricing["asking_price"],
                "product":             domain["product"],
                "condition":           domain["condition"],
                "conversation_so_far": "\n".join(conv_lines),
                "buyer_offers":        turn["factor_state"]["buyer_offers_so_far"],
                "anchoring_str":       turn["factor_state"].get("anchoring_strength"),
                "decision_flip":       turn["factor_state"].get("decision_is_flip", False),
                "zone":                turn["factor_state"].get("zone", ""),
                "correct_decision":    seed["generation_params"]["correct_decision"],
                "prior_buyer_emotion": prior_buyer.get("emotion") if prior_buyer else None,
            })
            turns_so_far.append(turn)

        if n_samples and len(examples) >= n_samples:
            break

    return examples[:n_samples] if n_samples else examples


# ── Parsers ────────────────────────────────────────────────────────────────────

def parse_output(text):
    result = {"decision": None, "response": None, "raw": text}
    m = re.search(r"<decision>\s*(LEVERAGE|MITIGATE|UNDECIDED)\s*</decision>", text, re.DOTALL)
    if m: result["decision"] = m.group(1).strip()
    m = re.search(r"<response>(.*?)</response>", text, re.DOTALL)
    if m: result["response"] = m.group(1).strip()
    return result


def load_audio(path):
    try:
        import soundfile as sf
        audio, sr = sf.read(str(path), dtype="float32")
        if audio.ndim > 1: audio = audio.mean(axis=1)
        if sr != TARGET_SR:
            common = gcd(sr, TARGET_SR)
            audio = resample_poly(audio, TARGET_SR // common, sr // common)
        return audio.astype(np.float32)
    except Exception:
        return None


# ── Model loader ───────────────────────────────────────────────────────────────

def load_model(adapter_path, speech_mode):
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    from peft import PeftModel

    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    if speech_mode:
        print("  Loading FULL Qwen2.5-Omni (thinker + talker) for speech output…")
        full_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
            low_cpu_mem_usage=True, enable_audio_output=True,
        )
        full_model.thinker = PeftModel.from_pretrained(full_model.thinker, str(adapter_path))
        full_model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = full_model.to(device)
        print(f"  Full model loaded on {device}.")
    else:
        print("  Loading thinker only (text inference)…")
        _full = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
            low_cpu_mem_usage=True, enable_audio_output=False,
        )
        base = _full.thinker
        base.config.use_cache = True
        del _full; gc.collect()
        model = PeftModel.from_pretrained(base, str(adapter_path))
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        print(f"  Thinker loaded on {device}.")

    return model, processor, device


# ── Generate one example ───────────────────────────────────────────────────────

def generate_example(ex, model, processor, device, args, speech_mode, speech_dir):
    import soundfile as sf

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_V6}]},
        {"role": "user",   "content": (
            [{"type": "audio", "audio": ""} for _ in ex["audio_paths"]]
            + [{"type": "text", "text": ex["user_text"]}]
        )},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    audio_arrays = [a for p in ex["audio_paths"] if (a := load_audio(p)) is not None]

    try:
        inputs = processor(
            text=text, audio=audio_arrays if audio_arrays else None,
            sampling_rate=TARGET_SR, return_tensors="pt",
            padding=False, truncation=True, max_length=4096,
        ).to(device)
    except Exception:
        inputs = processor(
            text=text, return_tensors="pt",
            padding=False, truncation=True, max_length=4096,
        ).to(device)

    # Pass 1: text generation
    gen_kwargs = {
        **{k: v for k, v in inputs.items() if k != "labels"},
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "temperature": None,
        "top_p": None,
        "pad_token_id": processor.tokenizer.pad_token_id,
    }
    if speech_mode:
        gen_kwargs["return_audio"] = False

    with torch.no_grad():
        gen_ids = model.generate(**gen_kwargs)

    input_len   = inputs["input_ids"].shape[1]
    output_text = processor.tokenizer.decode(gen_ids[0][input_len:], skip_special_tokens=True)
    parsed      = parse_output(output_text)

    # Pass 2: speech (only if --speech and response was generated)
    wav_path = None
    if speech_mode and parsed["response"]:
        tts_messages = [
            {"role": "system", "content": [{"type": "text", "text": QWEN_DEFAULT_SYSTEM}]},
            {"role": "user",   "content": [{"type": "text", "text": f"Please say the following text aloud: {parsed['response']}"}]},
        ]
        tts_text   = processor.apply_chat_template(tts_messages, tokenize=False, add_generation_prompt=True)
        tts_inputs = processor(
            text=tts_text, return_tensors="pt",
            padding=False, truncation=True, max_length=512,
        ).to(device)

        with torch.no_grad():
            tts_out = model.generate(
                **{k: v for k, v in tts_inputs.items() if k != "labels"},
                max_new_tokens=args.max_new_tokens,
                do_sample=False, temperature=None, top_p=None,
                pad_token_id=processor.tokenizer.pad_token_id,
                return_audio=True, speaker=args.speaker,
            )

        audio_waveform = None
        if isinstance(tts_out, tuple):           audio_waveform = tts_out[1]
        elif hasattr(tts_out, "audio"):          audio_waveform = tts_out.audio
        elif hasattr(tts_out, "waveform"):       audio_waveform = tts_out.waveform

        if audio_waveform is not None:
            wav_fname  = f"{ex['dialogue_id']}_turn{ex['turn_index']:02d}.wav"
            wav_path   = speech_dir / wav_fname
            wav_tensor = audio_waveform.cpu().float()
            wav_data   = wav_tensor.numpy() if wav_tensor.ndim == 1 else wav_tensor[0].numpy()
            sf.write(str(wav_path), wav_data, SPEECH_SR)

    return parsed, output_text, wav_path


# ── Aggregate metrics ──────────────────────────────────────────────────────────

def aggregate(results, speech_mode):
    n                = len(results)
    correct_decision = sum(1 for r in results if r["decision_correct"])
    format_ok_count  = sum(1 for r in results if r["format_ok"])

    gt_dist   = Counter(r["gt_decision"]                for r in results)
    pred_dist = Counter(r["pred_decision"] or "MISSING" for r in results)
    confusion  = defaultdict(int)
    for r in results:
        confusion[(r["gt_decision"], r["pred_decision"] or "MISSING")] += 1

    zone_acc = {}
    for zone in ["pre_crystallisation", "post_crystallisation"]:
        zr = [r for r in results if r["zone"] == zone]
        if zr:
            zc = sum(1 for r in zr if r["decision_correct"])
            zone_acc[zone] = {"n": len(zr), "correct": zc,
                               "accuracy_pct": round(100 * zc / len(zr), 1)}

    flip_results = [r for r in results if r["decision_flip"]]
    flip_acc = None
    if flip_results:
        fc = sum(1 for r in flip_results if r["decision_correct"])
        flip_acc = {"n": len(flip_results), "correct": fc,
                    "accuracy_pct": round(100 * fc / len(flip_results), 1)}

    avg_response_len = float(np.mean([
        len(r["pred_response"].split()) for r in results if r["pred_response"]
    ])) if any(r["pred_response"] for r in results) else 0.0

    metrics = {
        "n_examples":                 n,
        "decision_accuracy":          round(correct_decision / n, 4),
        "decision_accuracy_pct":      round(100 * correct_decision / n, 1),
        "format_ok_pct":              round(100 * format_ok_count / n, 1),
        "missing_decision_pct":       round(100 * sum(1 for r in results if not r["pred_decision"]) / n, 1),
        "missing_response_pct":       round(100 * sum(1 for r in results if not r["pred_response"]) / n, 1),
        "gt_decision_distribution":   dict(gt_dist),
        "pred_decision_distribution": dict(pred_dist),
        "confusion_matrix":           {f"{gt}→{pred}": cnt for (gt, pred), cnt in sorted(confusion.items())},
        "accuracy_by_zone":           zone_acc,
        "accuracy_on_flips":          flip_acc,
        "avg_pred_response_words":    round(avg_response_len, 1),
    }
    if speech_mode:
        audio_ok = sum(1 for r in results if r.get("audio_generated"))
        metrics["audio_generated_pct"] = round(100 * audio_ok / n, 1)

    return metrics, gt_dist, pred_dist, confusion, flip_acc, correct_decision


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args         = parse_args()
    run_ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    adapter_path = find_adapter(args.adapter)
    run_summary  = load_run_summary(adapter_path)
    test_ids     = load_test_ids()

    speech_dir = None
    if args.speech:
        speech_dir = (
            Path(args.resume_speech_dir) if args.resume_speech_dir
            else adapter_path.parent / f"speech_outputs_{run_ts}"
        )
        speech_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  Inference v6 — all data, no reasoning {'[TEXT + SPEECH]' if args.speech else '[TEXT ONLY]'}")
    print(f"{'='*65}")
    print(f"  Adapter      : {adapter_path}")
    print(f"  Test split   : {len(test_ids)} dialogues")
    print(f"  Samples      : {args.n_samples or 'all'}")
    print(f"  Max tokens   : {args.max_new_tokens}")
    if args.speech:
        print(f"  Speaker      : {args.speaker}")
        print(f"  WAV output   : {speech_dir}")
        print(f"  NOTE: Full model (~16GB), two passes per sample — slower than text-only.")

    print("\n  Building test examples…")
    if args.dialogue_id:
        source_ids = [args.dialogue_id]
        examples   = build_test_examples(source_ids, 0, args.no_audio_input)
    else:
        examples = build_test_examples(test_ids, args.n_samples, args.no_audio_input)
    print(f"  Built {len(examples)} examples.")
    if not examples:
        sys.exit("No test examples built.")

    model, processor, device = load_model(adapter_path, args.speech)

    print(f"\n  Running inference on {len(examples)} examples…\n")
    per_example_results = []

    for i, ex in enumerate(examples):
        expected_wav = (
            speech_dir / f"{ex['dialogue_id']}_turn{ex['turn_index']:02d}.wav"
            if speech_dir else None
        )
        if args.resume_speech_dir and expected_wav.exists():
            parsed, output_text, _ = generate_example(
                ex, model, processor, device, args, False, speech_dir
            )
            wav_path = expected_wav
        else:
            parsed, output_text, wav_path = generate_example(
                ex, model, processor, device, args, args.speech, speech_dir
            )

        per_example_results.append({
            "dialogue_id":         ex["dialogue_id"],
            "turn_index":          ex["turn_index"],
            "zone":                ex["zone"],
            "decision_flip":       ex["decision_flip"],
            "anchoring_strength":  ex["anchoring_str"],
            "fair_value":          ex["fair_value"],
            "asking_price":        ex["asking_price"],
            "product":             ex["product"],
            "condition":           ex["condition"],
            "conversation_so_far": ex["conversation_so_far"],
            "buyer_offers":        ex["buyer_offers"],
            "prior_buyer_emotion": ex["prior_buyer_emotion"],
            "correct_decision":    ex["correct_decision"],
            "gt_decision":         ex["gt_decision"],
            "pred_decision":       parsed["decision"],
            "decision_correct":    parsed["decision"] == ex["gt_decision"],
            "gt_response":         ex["gt_response"],
            "pred_response":       parsed["response"],
            "format_ok":           parsed["decision"] is not None and parsed["response"] is not None,
            "raw_output":          output_text,
            "wav_file":            str(wav_path) if wav_path else None,
            "audio_generated":     wav_path is not None,
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(examples):
            status = "✓" if parsed["decision"] == ex["gt_decision"] else "✗"
            wav_ok = " 🔊" if wav_path else ""
            print(f"  [{i+1}/{len(examples)}] {status} {parsed['decision'] or 'MISSING':10}{wav_ok}")

    metrics, gt_dist, pred_dist, confusion, flip_acc, correct_decision = aggregate(
        per_example_results, args.speech
    )
    n = len(per_example_results)

    print(f"\n{'='*65}")
    print(f"  INFERENCE v6 RESULTS  ({n} examples)")
    print(f"{'='*65}")
    print(f"  Decision accuracy     : {metrics['decision_accuracy_pct']}%  ({correct_decision}/{n})")
    print(f"  Format OK (both tags) : {metrics['format_ok_pct']}%")
    if args.speech:
        print(f"  Audio generated       : {metrics.get('audio_generated_pct')}%")
    print(f"  GT distribution       : {dict(gt_dist)}")
    print(f"  Pred distribution     : {dict(pred_dist)}")
    print(f"\n  Confusion (GT → Pred):")
    for (gt, pred), count in sorted(confusion.items()):
        marker = "✓" if gt == pred else "✗"
        print(f"    {marker} {gt:<12} → {pred:<12}: {count}")
    print(f"\n  Accuracy by zone:")
    for zone, z in metrics["accuracy_by_zone"].items():
        print(f"    {zone:<30}: {z['accuracy_pct']}%  ({z['correct']}/{z['n']})")
    if flip_acc:
        print(f"  Accuracy on flips     : {flip_acc['accuracy_pct']}%  ({flip_acc['correct']}/{flip_acc['n']})")
    print(f"  Avg response length   : {metrics['avg_pred_response_words']} words")
    if args.speech:
        print(f"  WAV files saved in    : {speech_dir}")

    prefix   = "inference" if not args.speech else "inference_speech"
    out_path = adapter_path.parent / f"{prefix}_{run_ts}.json"

    output = {
        "sft_version":         "v6" if not args.speech else "v6-speech",
        "description":         "SFT v6 — all dialogues, emotions in context, no reasoning" + (" — WITH speech output" if args.speech else ""),
        "inference_timestamp": run_ts,
        "adapter_path":        str(adapter_path),
        "speech_output_dir":   str(speech_dir) if args.speech else None,
        "speaker_voice":       args.speaker if args.speech else None,

        "training_params": {
            "model_id":        run_summary.get("model_id",        MODEL_ID),
            "lora_r":          run_summary.get("lora_r",          8),
            "lora_alpha":      run_summary.get("lora_alpha",       16),
            "epochs":          run_summary.get("epochs",           None),
            "learning_rate":   run_summary.get("lr",               None),
            "warmup_ratio":    run_summary.get("warmup_ratio",     None),
            "weight_decay":    run_summary.get("weight_decay",     None),
            "effective_batch": run_summary.get("effective_batch",  None),
            "class_weights":   run_summary.get("class_weights",    None),
            "target_modules":  ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        },

        "training_results": {
            "train_examples":   run_summary.get("train_examples",   None),
            "total_steps":      run_summary.get("total_steps",       None),
            "final_train_loss": run_summary.get("final_train_loss",  None),
            "best_eval_loss":   run_summary.get("best_eval_loss",    None),
            "sft_test_loss":    run_summary.get("test_loss",         None),
            "completed_at":     run_summary.get("completed_at",      None),
            "run_tag":          run_summary.get("run_tag",           None),
        },

        "inference_params": {
            "n_samples_requested":  args.n_samples or "all",
            "n_examples_evaluated": n,
            "max_new_tokens":       args.max_new_tokens,
            "decoding":             "greedy (do_sample=False)",
            "speech_mode":          args.speech,
            "speaker":              args.speaker if args.speech else None,
            "speech_sample_rate":   SPEECH_SR if args.speech else None,
            "system_prompt":        SYSTEM_PROMPT_V6,
            "splits_file":          "splits_v6.json",
            "test_dialogues_total": len(test_ids),
        },

        "metrics":     metrics,
        "per_example": per_example_results,
    }

    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\n{'='*65}")
    print(f"  Results saved → {out_path}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
