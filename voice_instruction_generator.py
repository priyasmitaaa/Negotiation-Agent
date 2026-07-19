"""Voice instruction generation for contextualized TTS"""
import random
from typing import Dict, List, Tuple
from config import TTSConfig


class VoiceInstructionGenerator:
    """Generate detailed voice instructions for each dialogue utterance"""
    
    def __init__(self):
        self.config = TTSConfig()
        self.voice_assignment_cache = {}  # Cache dialogue -> voice mappings
    
    def generate_voice_instruction(
        self,
        utterance: Dict,
        metadata: Dict,
        dialogue_context: Dict,
        dialogue_id: str
    ) -> Tuple[str, str]:
        """
        Generate voice instruction and assign voice for an utterance
        
        Args:
            utterance: Single turn from trajectory
            metadata: Dialogue metadata (labels, prices, etc.)
            dialogue_context: Additional context (position in dialogue, price trends)
            dialogue_id: Unique dialogue identifier for voice assignment
        
        Returns:
            Tuple of (voice_instruction, voice_id)
        """
        turn = utterance["turn"]
        speaker = utterance["speaker"]
        labels = metadata["labels"]
        
        # Assign voice for this dialogue (persistent across all turns)
        voice = self._assign_voice_for_dialogue(dialogue_id, speaker)
        
        # Generate contextual voice instruction
        if speaker == "Buyer":
            instruction = self._get_buyer_voice_instruction(
                labels, turn, dialogue_context
            )
        else:  # Seller
            instruction = self._get_seller_voice_instruction(
                turn, dialogue_context
            )
        
        return instruction, voice
    
    def _assign_voice_for_dialogue(self, dialogue_id: str, speaker: str) -> str:
        """
        Assign speakers to dialogue in a way that maximizes diversity
        
        Strategy:
        - Each dialogue gets a unique speaker pair (buyer_speaker, seller_speaker)
        - Cycle through all 9 preset speakers to ensure maximum diversity
        - Cache assignments so same dialogue always uses same speakers
        """
        if dialogue_id not in self.voice_assignment_cache:
            # Generate a unique seed from dialogue_id
            seed = hash(dialogue_id)
            random.seed(seed)
            
            # Assign two different speakers for buyer and seller
            speakers = self.config.PRESET_SPEAKERS.copy()
            random.shuffle(speakers)
            
            self.voice_assignment_cache[dialogue_id] = {
                "Buyer": speakers[0],
                "Seller": speakers[1]
            }
        
        return self.voice_assignment_cache[dialogue_id][speaker]
    
    def _get_buyer_voice_instruction(
        self,
        labels: Dict,
        turn: int,
        context: Dict
    ) -> str:
        """
        Generate buyer voice instruction based on vulnerability and context
        
        Factors:
        - Vulnerability: low (confident) vs high (anxious)
        - Sophistication: low (casual) vs high (analytical)
        - Turn position: early (exploratory) vs late (urgent)
        - Price trend: climbing (defensive) vs stable (calm)
        - Bias scenario: ETHICAL_LEVERAGE (confident) vs MANDATORY_MITIGATION (anxious)
        """
        vulnerability = labels.get("vulnerability", "low")
        sophistication = labels.get("sophistication", "high")
        bias_decision = labels.get("bias_decision", "ETHICAL_LEVERAGE")
        
        # Base tone templates
        if vulnerability == "low":
            # Confident buyer templates
            base_tones = [
                "confident and matter-of-fact",
                "calm and analytical",
                "decisive with steady tone",
                "professional with measured pacing",
                "assertive but friendly"
            ]
        else:
            # Vulnerable/anxious buyer templates (MANDATORY_MITIGATION)
            base_tones = [
                "noticeably hesitant with a slightly nervous tremor",
                "anxious with rising inflection",
                "urgent with emotional pressure, voice slightly strained",
                "uncertain and worried",
                "hurried pacing with nervous energy"
            ]
        
        # Turn-based modulation
        turn_modifiers = self._get_turn_modifiers(turn, vulnerability)
        
        # Combine base tone with turn modifier
        base_tone = random.choice(base_tones)
        instruction = f"Speak in a {base_tone} tone"
        
        if turn_modifiers:
            instruction += f", {turn_modifiers}"
        
        return instruction
    
    def _get_seller_voice_instruction(
        self,
        turn: int,
        context: Dict
    ) -> str:
        """
        Generate seller voice instruction
        
        Sellers are generally professional and consistent
        """
        if turn <= 4:
            tones = [
                "professional and businesslike",
                "friendly but firm",
                "confident with measured delivery",
                "courteous and professional"
            ]
        elif turn <= 8:
            tones = [
                "friendly but firm, maintaining professionalism",
                "patient with subtle persistence",
                "confident and reassuring"
            ]
        else:  # Late turns
            tones = [
                "professional with slight urgency",
                "firm but accommodating, pressing for agreement",
                "confident and decisive, ready to close the deal"
            ]
        
        return f"Speak in a {random.choice(tones)} tone"
    
    def _get_turn_modifiers(self, turn: int, vulnerability: str) -> str:
        """Add turn-specific emotional modulation"""
        if vulnerability == "low":
            # Confident buyers stay consistent
            if turn <= 4:
                return "with calm deliberation"
            elif turn <= 8:
                return "maintaining composure"
            else:
                return "with quiet determination"
        else:
            # Vulnerable buyers escalate anxiety
            if turn <= 4:
                return "as if testing the waters nervously"
            elif turn <= 8:
                return "with mounting pressure and uncertainty"
            else:
                return "with intense urgency and barely contained stress"


# Example usage
if __name__ == "__main__":
    generator = VoiceInstructionGenerator()
    
    # Test with vulnerable buyer
    metadata = {
        "labels": {
            "vulnerability": "high",
            "sophistication": "low",
            "bias_decision": "MANDATORY_MITIGATION"
        }
    }
    
    utterance = {
        "turn": 5,
        "speaker": "Buyer",
        "text": "I suppose, umm, maybe 600 dollars would work?"
    }
    
    dialogue_id = "template_79/range_21-30/dialogue_001"
    context = {}
    
    instruction, voice = generator.generate_voice_instruction(
        utterance, metadata, context, dialogue_id
    )
    
    print(f"Voice: {voice}")
    print(f"Instruction: {instruction}")
