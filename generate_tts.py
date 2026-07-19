"""Main script to generate TTS audio from dialogue JSON files"""
import argparse
import json
from pathlib import Path
from typing import List, Optional, Dict
from tqdm import tqdm
from datetime import datetime
from config import TTSConfig
from qwen3_tts_engine import Qwen3TTSEngine


def find_dialogue_files(
    source_dir: Path,
    template_id: Optional[int] = None,
    range_id: Optional[str] = None
) -> List[Path]:
    """
    Find dialogue JSON files to process
    
    Args:
        source_dir: Base directory containing dialogues
        template_id: Optional template ID filter (e.g., 70, 79)
        range_id: Optional range filter (e.g., "1-10", "21-30")
    
    Returns:
        List of dialogue JSON file paths
    """
    pattern = "dialogue_*.json"
    
    # Zero-pad template_id to 2 digits to match folder naming (template_01, template_02, ...)
    template_dir_name = f"template_{template_id:02d}" if template_id else None

    if template_id and range_id:
        search_dir = source_dir / template_dir_name / f"range_{range_id}"
        if not search_dir.exists():
            print(f"Warning: Directory not found: {search_dir}")
            return []
        dialogue_files = list(search_dir.glob(pattern))
    elif template_id:
        search_dir = source_dir / template_dir_name
        if not search_dir.exists():
            print(f"Warning: Directory not found: {search_dir}")
            return []
        dialogue_files = list(search_dir.glob(f"*/{pattern}"))
    else:
        # Process all dialogues
        dialogue_files = list(source_dir.glob(f"*/*/{pattern}"))
    
    return sorted(dialogue_files)


def is_dialogue_complete(output_dir: Path, expected_turns: int = 12) -> bool:
    """
    Check if a dialogue has already been fully processed.
    Requires both: enough WAV files AND each file must be non-empty (> 0 bytes).
    0-byte placeholder files from interrupted generation are treated as incomplete.

    Args:
        output_dir: Output directory for the dialogue
        expected_turns: Expected number of audio files (default: 12)

    Returns:
        True if all expected WAV files exist and are non-empty, False otherwise
    """
    if not output_dir.exists():
        return False

    wav_files = list(output_dir.glob("*.wav"))
    if len(wav_files) < expected_turns:
        return False

    # Also verify none are empty (0-byte) placeholder files
    return all(f.stat().st_size > 0 for f in wav_files)


def load_progress(progress_file: Path) -> Dict:
    """
    Load existing progress tracker
    
    Args:
        progress_file: Path to progress JSON file
    
    Returns:
        Dictionary with progress data
    """
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            return json.load(f)
    
    return {
        "last_updated": None,
        "completed_dialogues": [],
        "failed_dialogues": [],
        "total_processed": 0
    }


def save_progress(progress_file: Path, progress_data: Dict):
    """
    Save progress tracker incrementally
    
    Args:
        progress_file: Path to progress JSON file
        progress_data: Progress dictionary to save
    """
    progress_data["last_updated"] = datetime.now().isoformat()
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(progress_file, 'w') as f:
        json.dump(progress_data, f, indent=2)


def should_skip_dialogue(
    dialogue_path: Path,
    output_dir: Path,
    force_regenerate: bool,
    progress_data: Dict,
    expected_turns: int = 12
) -> bool:
    """
    Determine if a dialogue should be skipped
    
    Args:
        dialogue_path: Path to dialogue JSON
        output_dir: Output directory for the dialogue
        force_regenerate: If True, never skip
        progress_data: Progress tracking data
        expected_turns: Expected number of audio files
    
    Returns:
        True if dialogue should be skipped, False otherwise
    """
    if force_regenerate:
        return False
    
    dialogue_id = str(dialogue_path)
    
    # Check if already completed
    if dialogue_id in progress_data.get("completed_dialogues", []):
        # Verify files still exist
        if is_dialogue_complete(output_dir, expected_turns):
            return True
    
    # Check if files exist even if not in progress file
    return is_dialogue_complete(output_dir, expected_turns)


def generate_tts_for_dialogues(
    dialogue_files: List[Path],
    output_base_dir: Path,
    engine: Qwen3TTSEngine,
    source_dir: Path,
    force_regenerate: bool = False,
    progress_file: Optional[Path] = None
):
    """
    Process multiple dialogue files and generate TTS audio
    
    Args:
        dialogue_files: List of dialogue JSON paths
        output_base_dir: Base output directory
        engine: Initialized TTS engine
        source_dir: Source directory used to compute relative paths
        force_regenerate: If True, regenerate even if files exist
        progress_file: Path to progress tracker JSON
    """
    print(f"\nFound {len(dialogue_files)} dialogues to process")
    
    # Load progress tracker
    config = TTSConfig()
    if progress_file is None:
        progress_file = config.PROGRESS_FILE
    
    progress_data = load_progress(progress_file)
    
    # Load existing results if they exist
    results_file = output_base_dir / "generation_results.json"
    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)
    else:
        results = {
            "total_dialogues": len(dialogue_files),
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "details": []
        }
    
    print(f"Mode: {'FORCE REGENERATE' if force_regenerate else 'SKIP COMPLETED'}")
    if not force_regenerate:
        print(f"Progress: {progress_data['total_processed']} dialogues processed previously")
    
    for dialogue_path in tqdm(dialogue_files, desc="Generating TTS"):
        # Create output directory matching source structure
        relative_path = dialogue_path.relative_to(source_dir)
        dialogue_name = dialogue_path.stem  # e.g., "dialogue_001"
        
        # Output structure: tts_outputs/template_X/range_Y-Z/dialogue_N/
        output_dir = output_base_dir / relative_path.parent / dialogue_name
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except FileNotFoundError:
            # Race condition: another process created an intermediate dir between
            # the existence check and mkdir. Retry once.
            import time; time.sleep(0.1)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if dialogue should be skipped
        if should_skip_dialogue(
            dialogue_path,
            output_dir,
            force_regenerate,
            progress_data,
            config.EXPECTED_TURNS_PER_DIALOGUE
        ):
            tqdm.write(f"⊘ Skipping {relative_path} (already complete)")
            results["skipped"] = results.get("skipped", 0) + 1
            
            # Check if already in details
            dialogue_id = str(relative_path)
            if not any(d["dialogue"] == dialogue_id for d in results["details"]):
                results["details"].append({
                    "dialogue": dialogue_id,
                    "status": "skipped",
                    "audio_files": config.EXPECTED_TURNS_PER_DIALOGUE,
                    "errors": []
                })
            continue
        
        try:
            # Process dialogue
            result = engine.process_dialogue(
                dialogue_path,
                output_dir,
                save_individual=True,
                save_combined=False
            )
            
            results["successful"] += 1
            dialogue_id = str(dialogue_path)
            
            # Update results details
            dialogue_result = {
                "dialogue": str(relative_path),
                "status": "success",
                "audio_files": len(result["audio_files"]),
                "errors": result["errors"]
            }
            results["details"].append(dialogue_result)
            
            # Update progress tracker
            if dialogue_id not in progress_data["completed_dialogues"]:
                progress_data["completed_dialogues"].append(dialogue_id)
            progress_data["total_processed"] = len(progress_data["completed_dialogues"])
            
            # Save progress incrementally
            save_progress(progress_file, progress_data)
            
            # Save results incrementally
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
        except Exception as e:
            print(f"\n✗ Error processing {relative_path}: {e}")
            results["failed"] += 1
            dialogue_id = str(dialogue_path)
            
            results["details"].append({
                "dialogue": str(relative_path),
                "status": "failed",
                "error": str(e)
            })
            
            # Track failed dialogue
            if dialogue_id not in progress_data["failed_dialogues"]:
                progress_data["failed_dialogues"].append(dialogue_id)
            
            # Save progress even on failure
            save_progress(progress_file, progress_data)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate TTS audio from dialogue JSON files"
    )
    
    parser.add_argument(
        "--template",
        type=int,
        help="Process specific template ID (e.g., 70, 79)"
    )
    
    parser.add_argument(
        "--range",
        type=str,
        help="Process specific range (e.g., '1-10', '21-30')"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all available dialogues"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom output directory (default: tts_outputs/)"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: process only first dialogue found"
    )
    
    parser.add_argument(
        "--preprocessed",
        action="store_true",
        help="Use preprocessed dialogues (from preprocessed/ folder) instead of original"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint (skip already-completed dialogues)"
    )
    
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Force regeneration of all dialogues, even if they already exist"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.range and not args.template:
        print("Error: --range requires --template")
        return
    
    if not args.all and not args.template:
        print("Error: Must specify --all or --template")
        return
    
    if args.resume and args.force_regenerate:
        print("Error: Cannot use --resume and --force-regenerate together")
        return
    
    # Initialize config
    config = TTSConfig()
    
    # Determine source directory
    if args.preprocessed:
        source_dir = config.BASE_DIR / "preprocessed"
        print("Using PREPROCESSED dialogues from preprocessed/ folder")
    else:
        source_dir = config.DIALOGUE_SOURCE_DIR
        print("Using ORIGINAL dialogues (will preprocess on-the-fly)")
    
    # Find dialogue files
    dialogue_files = find_dialogue_files(
        source_dir,
        template_id=args.template,
        range_id=args.range
    )
    
    if not dialogue_files:
        print("No dialogue files found!")
        return
    
    # Test mode: process only first dialogue
    if args.test:
        print("TEST MODE: Processing first dialogue only")
        dialogue_files = dialogue_files[:1]
    
    # Setup output directory
    output_dir = Path(args.output) if args.output else config.OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize TTS engine
    print("Initializing TTS engine...")
    engine = Qwen3TTSEngine()
    engine.load_model()
    
    # Set preprocessed mode for engine
    engine.use_preprocessed = args.preprocessed
    
    # Use per-template progress file to avoid concurrent write corruption
    # when multiple GPU workers run simultaneously
    if args.template:
        progress_file = config.OUTPUT_DIR / f"tts_progress_template_{args.template:02d}.json"
    else:
        progress_file = config.PROGRESS_FILE

    # Generate TTS with checkpoint support
    results = generate_tts_for_dialogues(
        dialogue_files,
        output_dir,
        engine,
        source_dir,
        force_regenerate=args.force_regenerate,
        progress_file=progress_file
    )
    
    # Print summary
    print("\n" + "="*60)
    print("TTS GENERATION SUMMARY")
    print("="*60)
    print(f"Total dialogues: {results['total_dialogues']}")
    print(f"Successful: {results['successful']}")
    print(f"Skipped: {results.get('skipped', 0)}")
    print(f"Failed: {results['failed']}")
    print(f"Output directory: {output_dir}")
    print(f"Progress file: {config.PROGRESS_FILE}")
    
    # Save results to JSON
    results_file = output_dir / "generation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")


if __name__ == "__main__":
    main()
