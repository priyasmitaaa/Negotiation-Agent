"""Qwen3 TTS Engine - Model loading and inference"""
import torch
import soundfile as sf
from pathlib import Path
import json
import numpy as np
from typing import Dict, List, Optional
from huggingface_hub import snapshot_download, login
try:
    from qwen_tts import Qwen3TTSModel
except ImportError:
    print("WARNING: qwen_tts library not installed. Install with: pip install qwen-tts")
    Qwen3TTSModel = None

from config import TTSConfig
from tts_text_preprocessor import TTSTextPreprocessor
from voice_instruction_generator import VoiceInstructionGenerator


class Qwen3TTSEngine:
    """Qwen3 TTS model interface for speech generation"""
    
    def __init__(self, model_name: Optional[str] = None, hf_token: Optional[str] = None):
        """Initialize TTS engine"""
        self.config = TTSConfig()
        self.model_name = model_name or self.config.MODEL_NAME
        self.hf_token = hf_token or self.config.HF_TOKEN
        
        if Qwen3TTSModel is None:
            raise ImportError("qwen_tts library not installed. Install with: pip install qwen-tts")
        
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"TTS Engine initialized. Device: {self.device}")
    
    def load_model(self):
        """Download and load Qwen3 TTS model"""
        print(f"Loading model: {self.model_name}")
        print("This may take a while on first run...")
        
        try:
            # Login to HuggingFace
            login(token=self.hf_token)
            
            # Download model
            model_path = snapshot_download(self.model_name)
            
            # Load Qwen3TTS model
            self.model = Qwen3TTSModel.from_pretrained(
                model_path,
                device_map=self.device,
                dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                token=self.hf_token,
            )
            
            print(f"✓ Model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise
    
    def generate_speech(
        self,
        text: str,
        voice_instruction: Optional[str],
        speaker: str,
        output_path: Path,
        language: str = "Auto"
    ) -> bool:
        """
        Generate speech audio from text using Qwen3TTS CustomVoice
        
        Args:
            text: Preprocessed text to convert to speech
            voice_instruction: Optional voice instruction for extra style control
            speaker: Speaker name (Aiden, Dylan, Eric, Ono_anna, etc.)
            output_path: Path to save output audio file
            language: Language ("Auto", "English", "Chinese", etc.)
        
        Returns:
            True if successful, False otherwise
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Normalize speaker name (lowercase, underscores)
            speaker_normalized = speaker.lower().replace(" ", "_")
            
            print(f"  Generating {speaker} ({language}): {text[:50]}...")
            
            # Generate audio using CustomVoice model
            # API: generate_custom_voice(text, language, speaker, instruct, non_streaming_mode, max_new_tokens)
            wavs, sr = self.model.generate_custom_voice(
                text=text,
                language=language,
                speaker=speaker_normalized,
                instruct=voice_instruction,  # Optional style instruction
                non_streaming_mode=True,
                max_new_tokens=2048,
            )
            
            # Save audio output
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to numpy and save as WAV.
            # Use soundfile instead of torchaudio.save to avoid TorchCodec/FFmpeg runtime dependency.
            audio_np = wavs[0].cpu().numpy() if torch.is_tensor(wavs[0]) else wavs[0]
            if isinstance(audio_np, list):
                audio_np = np.asarray(audio_np, dtype=np.float32)
            if audio_np.ndim > 1:
                # Ensure mono waveform for consistent downstream handling.
                audio_np = audio_np.squeeze()
                if audio_np.ndim > 1:
                    audio_np = np.mean(audio_np, axis=0)
            audio_np = np.asarray(audio_np, dtype=np.float32)
            sf.write(str(output_path), audio_np, int(sr))
            
            print(f"  ✓ Generated: {output_path.name}")
            return True
            
        except Exception as e:
            print(f"  ✗ Error generating speech: {e}")
            return False
    
    def process_dialogue(
        self,
        dialogue_path: Path,
        output_dir: Path,
        save_individual: bool = True,
        save_combined: bool = False
    ) -> Dict:
        """
        Process entire dialogue and generate all audio files
        
        Args:
            dialogue_path: Path to dialogue JSON file
            output_dir: Directory to save audio outputs
            save_individual: Save individual turn audio files
            save_combined: Combine all turns into single audio file
        
        Returns:
            Dictionary with processing results
        """
        # Load dialogue
        with open(dialogue_path, 'r') as f:
            dialogue = json.load(f)
        
        # Check if using preprocessed data
        use_preprocessed = getattr(self, 'use_preprocessed', False)
        is_preprocessed = dialogue.get("metadata", {}).get("tts_preprocessed", False)
        
        # Initialize preprocessor and voice generator only if needed
        if not (use_preprocessed and is_preprocessed):
            preprocessor = TTSTextPreprocessor()
            voice_gen = VoiceInstructionGenerator()
            dialogue_id = str(dialogue_path)
        else:
            preprocessor = None
            voice_gen = None
            dialogue_id = str(dialogue_path)
        
        results = {
            "dialogue_id": dialogue_id,
            "total_turns": len(dialogue["trajectory"]),
            "audio_files": [],
            "errors": []
        }
        
        print(f"\nProcessing: {dialogue_path.name}")
        if use_preprocessed and is_preprocessed:
            print("  [Using preprocessed TTS metadata]")
        
        # Process each utterance
        for utterance in dialogue["trajectory"]:
            turn = utterance["turn"]
            speaker = utterance["speaker"]
            
            # Use preprocessed data if available, otherwise process on-the-fly
            if use_preprocessed and is_preprocessed and "tts_text" in utterance:
                processed_text = utterance["tts_text"]
                voice_instruction = utterance.get("voice_instruction")
                assigned_speaker = utterance.get("voice_id", "Aiden")
            else:
                # On-the-fly preprocessing
                original_text = utterance["text"]
                processed_text, _ = preprocessor.preprocess_for_tts(
                    original_text,
                    dialogue["metadata"]["labels"],
                    turn,
                    speaker
                )
                voice_instruction, assigned_speaker = voice_gen.generate_voice_instruction(
                    utterance,
                    dialogue["metadata"],
                    {},
                    dialogue_id
                )
            
            # Generate audio
            audio_filename = f"turn_{turn:02d}_{speaker.lower()}.wav"
            audio_path = output_dir / audio_filename
            
            print(f"  Turn {turn} ({speaker}): {assigned_speaker}")
            
            success = self.generate_speech(
                processed_text,
                voice_instruction,
                assigned_speaker,
                audio_path
            )
            
            if success:
                results["audio_files"].append(str(audio_path))
            else:
                results["errors"].append(f"Turn {turn}")
        
        print(f"✓ Completed: {len(results['audio_files'])}/{results['total_turns']} turns")
        
        return results


# Example usage
if __name__ == "__main__":
    engine = Qwen3TTSEngine()
    
    # Test model loading
    print("Testing model load...")
    engine.load_model()
    
    # Test single utterance
    test_text = "I can offer 500 dollars for this item"
    test_instruction = "Speak in a confident, matter-of-fact tone"
    test_speaker = "Aiden"
    test_output = Path("test_output.wav")
    
    engine.generate_speech(test_text, test_instruction, test_speaker, test_output)
