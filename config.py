"""Configuration for Qwen3 TTS System"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class TTSConfig:
    """Configuration for TTS generation system"""
    
    # Hugging Face Authentication
    HF_TOKEN = os.getenv("HF_TOKEN", "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")
    
    # Qwen3 TTS Model - using CustomVoice for voice descriptions
    MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"  # 0.6B for faster inference
    MODEL_SIZE = "0.6B"  # Can be "0.6B" or "1.7B"
    
    # Directories
    BASE_DIR = Path(__file__).resolve().parent
    DIALOGUE_SOURCE_DIR = BASE_DIR.parent / "multi-agent-gen" / "dialogues_51_100"
    OUTPUT_DIR = BASE_DIR / "tts_outputs"
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Voice Configuration - 9 preset speakers from Qwen3-TTS
    # Using these preset speakers for maximum diversity
    PRESET_SPEAKERS = [
        "Aiden", "Dylan", "Eric", "Ono_anna", "Ryan", 
        "Serena", "Sohee", "Uncle_fu", "Vivian"
    ]
    
    # Voice assignment strategy: cycle through all 9 preset speakers
    # Each dialogue gets a unique voice combination to maximize diversity
    
    # Filler words configuration
    FILLER_WORDS = {
        "low_sophistication": ["umm", "like", "you know", "kinda", "this kind of"],
        "high_sophistication": ["perhaps", "somewhat", "maybe", "I suppose", "you see"]
    }
    
    # Filler injection rates (per dialogue) for vulnerable personas
    FILLER_RATES = {
        "early_turns": (1, 2),  # turns 1-5
        "mid_turns": (0, 1),     # turns 6-9
        "late_turns": (1, 2)     # turns 10-12
    }
    
    # Checkpoint and Progress Tracking
    PROGRESS_FILE = OUTPUT_DIR / "tts_progress.json"
    EXPECTED_TURNS_PER_DIALOGUE = 12
