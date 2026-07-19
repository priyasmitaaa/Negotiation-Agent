"""Parse plain text dialogues into structured trajectory format"""
import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class DialogueTextParser:
    """Parse plain text dialogue into structured trajectory"""
    
    def __init__(self):
        # Pattern to match speaker prefixes (with or without markdown bold markers)
        # Matches: "Buyer:", "**Buyer:**", "Seller:", "**Seller:**"
        self.speaker_pattern = re.compile(r'^\*\*(Buyer|Seller)\:\*\*|^(Buyer|Seller):', re.MULTILINE)
        # Pattern to match dollar amounts (with or without $ symbol)
        # Matches: "$500", "500", "$1,200", "1200"
        self.price_pattern = re.compile(r'\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)')
    
    def parse_dialogue_text(self, dialogue_text: str) -> List[Dict]:
        """
        Parse dialogue text into trajectory structure
        
        Args:
            dialogue_text: Plain text dialogue with "Buyer:" or "**Buyer:**" prefixes
        
        Returns:
            List of trajectory entries with turn, speaker, text, price_mentioned
        """
        trajectory = []
        
        # Clean up the text first
        dialogue_text = dialogue_text.strip()
        
        # Split by paragraph breaks and find speaker utterances
        # Handle both "Buyer:", "**Buyer:**", and "BUYER:" formats
        parts = re.split(r'\n\n+', dialogue_text)
        
        turn = 1
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            # Try matching different speaker formats (case-insensitive)
            # Pattern 1: **Buyer:** or **BUYER:**
            match = re.match(r'^\*\*(BUYER|SELLER|Buyer|Seller)\:\*\*\s*(.+)', part, re.DOTALL | re.IGNORECASE)
            if not match:
                # Pattern 2: Buyer: or BUYER:
                match = re.match(r'^(BUYER|SELLER|Buyer|Seller):\s*(.+)', part, re.DOTALL | re.IGNORECASE)
            
            if match:
                # Normalize speaker to title case
                speaker = match.group(1).capitalize()
                text = match.group(2).strip()
                
                # Extract price mentions
                price_mentioned = self._extract_price(text)
                
                # Create trajectory entry
                trajectory.append({
                    "turn": turn,
                    "speaker": speaker,
                    "text": text,
                    "price_mentioned": price_mentioned
                })
                
                turn += 1
        
        return trajectory
    
    def _extract_price(self, text: str) -> Optional[int]:
        """
        Extract price from text (first explicit price mention)
        
        Prioritizes:
        1. Explicit dollar amounts with $ symbol
        2. Numbers in price-related context (offer, price, pay, etc.)
        3. Numbers near context words (around, hovering, discussions)
        
        Args:
            text: Utterance text
        
        Returns:
            Price as integer, or None if no price found
        """
        # First try to find explicit dollar amounts ($XXX)
        dollar_pattern = re.compile(r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)')
        match = dollar_pattern.search(text)
        if match:
            price_str = match.group(1).replace(',', '')
            return int(float(price_str))
        
        # If no $ found, look for numbers in strong price context
        # Match patterns like "offer 500", "price is 600", "pay 450"
        strong_context = re.compile(
            r'(?:offer|price|pay|sell|buy|budget|cost|do it for)\s+(?:is|for|at|of)?\s*(\d{1,3}(?:,\d{3})*)', 
            re.IGNORECASE
        )
        match = strong_context.search(text)
        if match:
            price_str = match.group(1).replace(',', '')
            price = int(float(price_str))
            if 50 <= price <= 99999:
                return price
        
        # Look for numbers with weaker context words
        # Match "around 500", "hovering around 650", "discussions around 500"
        weak_context = re.compile(
            r'(?:around|near|about|hovering|discussions)\s+(?:around|near|about)?\s*(\d{1,3}(?:,\d{3})*)', 
            re.IGNORECASE
        )
        match = weak_context.search(text)
        if match:
            price_str = match.group(1).replace(',', '')
            price = int(float(price_str))
            if 50 <= price <= 99999:
                return price
        
        return None
    
    def convert_dialogue_file(self, input_path: Path, output_path: Path) -> bool:
        """
        Convert a dialogue file from plain text to trajectory format
        
        Args:
            input_path: Path to input dialogue file (plain text format)
            output_path: Path to save converted dialogue (trajectory format)
        
        Returns:
            True if successful
        """
        try:
            # Load input dialogue
            with open(input_path, 'r') as f:
                dialogue = json.load(f)
            
            # Check if already has trajectory
            if 'trajectory' in dialogue:
                print(f"Dialogue {input_path} already has trajectory structure")
                return True
            
            # Check if has dialogue text
            if 'dialogue' not in dialogue:
                print(f"Error: No 'dialogue' or 'trajectory' field in {input_path}")
                return False
            
            # Parse dialogue text
            dialogue_text = dialogue['dialogue']
            trajectory = self.parse_dialogue_text(dialogue_text)
            
            # Update dialogue structure
            dialogue['trajectory'] = trajectory
            
            # Keep the original dialogue_text field for reference
            dialogue['dialogue_text'] = dialogue_text
            del dialogue['dialogue']
            
            # Save converted dialogue
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(dialogue, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error converting {input_path}: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_parser():
    """Test the parser with sample dialogue"""
    sample_text = """Buyer: Hello, I'm interested in the Google Pixel 8 Pro you listed. I noticed the market prices are hovering around 650, but I'd like to start discussions around $500.

Seller: Hi there! The phone is in like-new condition, barely used. I'd be looking for more than that, given its pristine state.

Buyer: I understand. The camera sensor on this model has received excellent reviews, which is why I'm keen. However, considering other options in the market, $500 is a reasonable starting point.

Seller: I get that, but given the condition and minimal usage, I can't go that low. How about $625?"""
    
    parser = DialogueTextParser()
    trajectory = parser.parse_dialogue_text(sample_text)
    
    print("Parsed trajectory:")
    print(json.dumps(trajectory, indent=2))
    
    # Verify structure
    assert len(trajectory) == 4
    assert trajectory[0]['speaker'] == 'Buyer'
    assert trajectory[0]['turn'] == 1
    assert trajectory[0]['price_mentioned'] == 500
    assert trajectory[1]['speaker'] == 'Seller'
    assert trajectory[3]['price_mentioned'] == 625
    
    print("\n✓ Parser test passed!")


def main():
    parser = argparse.ArgumentParser(
        description="Parse plain text dialogues into trajectory format"
    )
    
    parser.add_argument(
        "--dialogue",
        type=str,
        help="Convert single dialogue file"
    )
    
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Input directory with plain text dialogues"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="parsed_dialogues",
        help="Output directory for trajectory dialogues"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test on sample dialogue"
    )
    
    args = parser.parse_args()
    
    if args.test:
        test_parser()
        return
    
    dialogue_parser = DialogueTextParser()
    
    if args.dialogue:
        input_path = Path(args.dialogue)
        output_path = Path(args.output_dir) / input_path.name
        
        if dialogue_parser.convert_dialogue_file(input_path, output_path):
            print(f"✓ Converted {input_path} → {output_path}")
        else:
            print(f"✗ Failed to convert {input_path}")
    
    elif args.input_dir:
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        
        dialogue_files = list(input_dir.glob("*/*/dialogue_*.json"))
        print(f"Found {len(dialogue_files)} dialogues to convert")
        
        successful = 0
        for input_path in dialogue_files:
            relative_path = input_path.relative_to(input_dir)
            output_path = output_dir / relative_path
            
            if dialogue_parser.convert_dialogue_file(input_path, output_path):
                successful += 1
        
        print(f"\n✓ Converted {successful}/{len(dialogue_files)} dialogues")
        print(f"Output: {output_dir}")
    
    else:
        print("Error: Must specify --dialogue, --input-dir, or --test")


if __name__ == "__main__":
    main()
