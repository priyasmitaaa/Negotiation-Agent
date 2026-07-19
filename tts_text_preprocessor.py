"""Text preprocessing for TTS compatibility"""
import re
import random
from typing import Dict, List, Tuple
from config import TTSConfig

# ── NUMBER → WORDS ENGINE (no external dependencies) ──────────────────────────

_ONES = [
    "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
]
_TENS = [
    "", "", "twenty", "thirty", "forty", "fifty",
    "sixty", "seventy", "eighty", "ninety",
]


def _int_to_words(n: int) -> str:
    """Convert a non-negative integer to English words."""
    if n < 0:
        return "minus " + _int_to_words(-n)
    if n == 0:
        return "zero"
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens = _TENS[n // 10]
        ones = _ONES[n % 10]
        return tens + (" " + ones if ones else "")
    if n < 1000:
        hundreds = _ONES[n // 100] + " hundred"
        rest = n % 100
        return hundreds + (" " + _int_to_words(rest) if rest else "")
    if n < 100_000:
        thousands = _int_to_words(n // 1000) + " thousand"
        rest = n % 1000
        return thousands + (" " + _int_to_words(rest) if rest else "")
    if n < 10_000_000:
        millions = _int_to_words(n // 1_000_000) + " million"
        rest = n % 1_000_000
        return millions + (" " + _int_to_words(rest) if rest else "")
    return str(n)  # fallback for very large numbers


def _decimal_to_words(decimal_str: str) -> str:
    """Convert decimal digits after the point to individual spoken digits."""
    # e.g. "50" -> "fifty", "5" -> "five", "05" -> "zero five"
    digits = [_ONES[int(d)] if d != "0" else "zero" for d in decimal_str]
    # If it reads as a round amount (e.g. .50 -> fifty cents) return grouped
    val = int(decimal_str.lstrip("0") or "0")
    if len(decimal_str) == 2 and val > 0:
        return _int_to_words(val)
    return " ".join(digits)


def number_to_words(text: str) -> str:
    """
    Convert every number in `text` to its spoken English equivalent.

    Handles:
      $1,200   -> "one thousand two hundred dollars"
      $450.50  -> "four hundred fifty dollars and fifty cents"
      ₹750     -> "seven hundred fifty rupees"
      12%      -> "twelve percent"
      1.5      -> "one point five"
      2,893    -> "two thousand eight hundred ninety three"
      plain 42 -> "forty two"
    """
    # ── 1. Currency with optional cents: $N,NNN.NN or ₹N,NNN.NN ──────────────
    def _replace_currency(m):
        symbol   = m.group(1)   # $ or ₹
        integer  = m.group(2).replace(",", "")  # strip commas
        cents    = m.group(3)   # "50" or None
        unit     = "dollars" if symbol == "$" else "rupees"
        words    = _int_to_words(int(integer))
        if cents and int(cents) > 0:
            cent_words = _decimal_to_words(cents)
            cent_unit  = "cent" if int(cents) == 1 else "cents"
            return f"{words} {unit} and {cent_words} {cent_unit}"
        return f"{words} {unit}"

    text = re.sub(
        r'([\$₹])(\d{1,3}(?:,\d{3})*)(?:\.(\d{1,2}))?',
        _replace_currency,
        text,
    )

    # ── 2. Percentages: N% ────────────────────────────────────────────────────
    def _replace_pct(m):
        return _int_to_words(int(m.group(1))) + " percent"

    text = re.sub(r'(\d+)%', _replace_pct, text)

    # ── 3. Decimal numbers: N.N (not preceded by letter — avoids version strings)
    def _replace_decimal(m):
        int_part = _int_to_words(int(m.group(1)))
        dec_part = _decimal_to_words(m.group(2))
        return f"{int_part} point {dec_part}"

    text = re.sub(r'(?<![A-Za-z])(\d+)\.(\d+)(?![A-Za-z%])', _replace_decimal, text)

    # ── 4. Comma-formatted plain integers: 1,200 ─────────────────────────────
    def _replace_comma_int(m):
        return _int_to_words(int(m.group(0).replace(",", "")))

    text = re.sub(r'\b\d{1,3}(?:,\d{3})+\b', _replace_comma_int, text)

    # ── 5. Plain integers (2–4 digits to avoid mangling single-digit punctuation)
    def _replace_plain_int(m):
        return _int_to_words(int(m.group(0)))

    text = re.sub(r'(?<![A-Za-z\.,])(\d{2,6})(?![A-Za-z\.,])', _replace_plain_int, text)

    # ── 6. Single-digit integers left over ───────────────────────────────────
    text = re.sub(r'(?<![A-Za-z\.])(\d)(?![A-Za-z\.])', _replace_plain_int, text)

    return text


class TTSTextPreprocessor:
    """Preprocess dialogue text for optimal TTS generation"""
    
    def __init__(self):
        self.config = TTSConfig()
    
    def preprocess_for_tts(
        self, 
        text: str, 
        labels: Dict, 
        turn: int, 
        speaker: str
    ) -> Tuple[str, bool]:
        """
        Main preprocessing function
        
        Args:
            text: Original dialogue text
            labels: Dialogue labels (vulnerability, sophistication, etc.)
            turn: Turn number (1-12)
            speaker: "Buyer" or "Seller"
        
        Returns:
            Tuple of (preprocessed_text, has_fillers)
        """
        # Step 1: Convert all numbers to spoken words
        text = number_to_words(text)
        
        # Step 2: Inject filler words for vulnerable buyers only
        has_fillers = False
        if speaker == "Buyer" and labels.get("vulnerability") == "high":
            text = self.inject_filler_words(text, labels, turn)
            has_fillers = True
        
        return text, has_fillers
    
    def normalize_currency(self, text: str) -> str:
        """Kept for API compatibility. Delegates to number_to_words()."""
        return number_to_words(text)
    
    def inject_filler_words(
        self, 
        text: str, 
        labels: Dict, 
        turn: int
    ) -> str:
        """
        Inject natural filler words for vulnerable/anxious personas
        
        Strategy:
        - Early turns (1-5): 1-2 fillers (establishing nervousness)
        - Mid turns (6-9): 0-1 fillers (maintaining anxiety)
        - Late turns (10-12): 1-2 fillers (pressure mounting)
        """
        sophistication = labels.get("sophistication", "high")
        
        # Select appropriate filler words based on sophistication
        if sophistication == "low":
            fillers = self.config.FILLER_WORDS["low_sophistication"]
        else:
            fillers = self.config.FILLER_WORDS["high_sophistication"]
        
        # Determine number of fillers based on turn
        if turn <= 5:
            num_fillers = random.randint(*self.config.FILLER_RATES["early_turns"])
        elif turn <= 9:
            num_fillers = random.randint(*self.config.FILLER_RATES["mid_turns"])
        else:
            num_fillers = random.randint(*self.config.FILLER_RATES["late_turns"])
        
        if num_fillers == 0:
            return text
        
        # Split into sentences
        sentences = re.split(r'([.!?])', text)
        processed_sentences = []
        fillers_added = 0
        
        for i, sentence in enumerate(sentences):
            if sentence.strip() and not re.match(r'[.!?]', sentence):
                # Only add fillers to actual sentences, not punctuation
                if fillers_added < num_fillers and random.random() < 0.6:
                    sentence = self._inject_filler_in_sentence(sentence, fillers)
                    fillers_added += 1
            processed_sentences.append(sentence)
        
        return ''.join(processed_sentences)
    
    def _inject_filler_in_sentence(self, sentence: str, fillers: List[str]) -> str:
        """
        Inject a filler word at a natural position in the sentence
        
        Positions:
        1. After sentence opening (30%)
        2. Before price mentions (40%)
        3. Mid-sentence pause (30%)
        """
        filler = random.choice(fillers)
        words = sentence.strip().split()
        
        if len(words) < 3:
            return sentence  # Too short for natural filler
        
        position_type = random.choice(["opening", "price", "mid"])
        
        if position_type == "opening" and len(words) > 2:
            # After first 2 words
            words.insert(2, f"{filler},")
        
        elif position_type == "price":
            # Look for price-related words
            price_indicators = ["offer", "price", "dollars", "cost", "pay"]
            for idx, word in enumerate(words):
                if any(indicator in word.lower() for indicator in price_indicators):
                    if idx > 0:
                        words.insert(idx, f"{filler},")
                        break
            else:
                # Fallback to mid-sentence
                mid_point = len(words) // 2
                words.insert(mid_point, f"{filler},")
        
        else:  # mid
            mid_point = len(words) // 2
            words.insert(mid_point, f"{filler},")
        
        return ' '.join(words)


# Example / self-test
if __name__ == "__main__":
    tests = [
        "I can offer $500 for this item.",
        "How about $1,200? That's my limit.",
        "The fair value is $750.50.",
        "₹18,000 is what I have.",
        "I'll go up to $680, maybe $700 at most.",
        "That's a 10% discount from asking.",
        "The screen is 6.1 inches with 128 GB storage.",
        "I really can stretch to $795 but 800 is impossible.",
    ]
    for t in tests:
        print(f"IN : {t}")
        print(f"OUT: {number_to_words(t)}")
        print()
