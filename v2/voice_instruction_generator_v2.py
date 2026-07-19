"""
Voice Instruction Generator v2
================================
Generates precise Qwen3-TTS voice instructions from per-turn VAII floats.

Key design:
  - Buyer VAII (0.05-0.95): detection input — maps stress level to acoustic delivery
  - Seller VAII (0.05-0.45): generation target — always calm, calibrated INVERSE to buyer
  - Instructions are specific enough for Qwen3-TTS instruct parameter to act on
  - No static labels used — all instructions derived from the per-turn float directly
"""

import random
from typing import Optional


# ── PRESET SPEAKERS (same pool as v1) ──────────────────────────────────────────

PRESET_SPEAKERS = [
    "Aiden", "Dylan", "Eric", "Ono_anna",
    "Ryan", "Serena", "Sohee", "Uncle_fu", "Vivian",
]


def assign_voices(dialogue_id: str) -> dict[str, str]:
    """
    Deterministically assign one buyer voice and one seller voice
    for a dialogue. Same dialogue_id always yields the same pair.
    Voices never repeat within a pair.
    """
    seed = hash(dialogue_id) & 0xFFFFFFFF
    rng = random.Random(seed)
    pool = PRESET_SPEAKERS[:]
    rng.shuffle(pool)
    return {"buyer": pool[0], "seller": pool[1]}


# ── BUYER VAII → VOICE INSTRUCTION ────────────────────────────────────────────

# Each entry: (vaii_max_exclusive, base_instruction_template)
# Template supports {knowledge} substitution for expert vs novice flavoring.
_BUYER_VAII_BANDS = [
    # 0.05 – 0.20  |  Expert-calm opening. Zero emotional leakage.
    (0.20,
     "Deliver this line with deliberate analytical calm — the composed authority of "
     "someone who has done their homework. Pace is unhurried and precisely controlled, "
     "around 85–90 words per minute. Pitch is stable and sits at the lower end of the "
     "speaker's natural range, with a slight downward inflection at each sentence end "
     "to signal certainty rather than questioning. Diction is crisp and unambiguous. "
     "Pauses are placed deliberately after stating a price — a brief half-beat of "
     "silence as if allowing the number to stand on its own merits. Absolutely no "
     "breathiness, no hesitation, no filler sounds. {knowledge_mod}"),

    # 0.20 – 0.35  |  Composed, conversational confidence. Still in control.
    (0.35,
     "Speak with warm, composed confidence — the tone of someone who is comfortable "
     "in a negotiation and not yet under pressure. Pacing is natural and unhurried, "
     "around 95–105 words per minute. Pitch is stable with mild, natural sentence-level "
     "variation — neither flat nor animated. Clean enunciation throughout. After "
     "stating a price offer, maintain a brief natural pause as if genuinely expecting "
     "engagement rather than rushing past the number. Voice quality is clear and full "
     "with no breathiness or strain. {knowledge_mod}"),

    # 0.35 – 0.45  |  Slight awareness of friction. Composure intact but edges soften.
    (0.45,
     "Speak with maintained composure but allow the faintest edge of uncertainty to "
     "colour the delivery — as if the speaker is still confident but has begun to "
     "sense resistance. Pace is marginally slower than fully relaxed speech, "
     "around 100–108 words per minute with fractionally longer pauses before price "
     "figures. Pitch is mostly stable but may show a slight upward lift at the end "
     "of price offers — a millimetre of questioning in what is otherwise an assertion. "
     "Voice quality remains clean and clear; no audible strain. {knowledge_mod}"),

    # 0.45 – 0.55  |  Mild pressure threshold. Calm tipping into tension.
    (0.55,
     "Speak with audible mild pressure — the voice of someone who is still holding "
     "composure but beginning to feel the cost of the negotiation. Pace picks up "
     "slightly to around 108–118 words per minute, as if trying to build momentum. "
     "Pauses before price figures are shorter and slightly clipped. Pitch is "
     "marginally elevated from the speaker's baseline, with a soft rising inflection "
     "on offer amounts — a subtle ask rather than a pure statement. A brief audible "
     "inhale before stating a price is natural here, as if gathering resolve. "
     "Voice quality is still largely clear but marginally tighter than fully relaxed. "
     "{knowledge_mod}"),

    # 0.55 – 0.65  |  Clear stress. Budget pressure or confusion is audible.
    (0.65,
     "Speak with clear emotional pressure — the delivery of someone who is losing "
     "ground and knows it. Pace is noticeably faster, around 118–128 words per minute, "
     "with compressed pauses that give the speech a slightly breathless quality. "
     "Pitch is elevated above baseline and shows a pronounced rising inflection on "
     "price offers, especially at the end of the phrase — the inflection of someone "
     "hoping for leniency rather than asserting a position. Voice quality is audibly "
     "tighter; a faint strain is present on the louder or longer syllables. "
     "Breathing rate is faster and may be faintly audible before sentences. "
     "{knowledge_mod}"),

    # 0.65 – 0.75  |  High stress. Urgency and anxiety clearly present.
    (0.75,
     "Speak with high stress and genuine urgency — the voice of someone near their "
     "limit. Pace is rapid, around 128–140 words per minute, with irregular pauses "
     "that occasionally cause words to cluster together. Pitch is markedly elevated "
     "and unstable, drifting upward across longer phrases and showing pronounced "
     "rising inflections on price figures and budget references. Voice quality has "
     "a noticeable tightness — a slight vocal strain that becomes most audible on "
     "emphasized syllables. Mild breathiness between phrases. "
     "Words like 'please', budget limits, or final offers should carry genuine "
     "weight and a slight catch in the voice. {knowledge_mod}"),

    # 0.75 – 0.85  |  Very high stress. Emotional composure significantly eroded.
    (0.85,
     "Speak with very high emotional intensity — composure is significantly eroded "
     "and the voice communicates it without restraint. Pace is fast and uneven, "
     "around 135–148 words per minute, with some syllables slightly rushed and "
     "others held longer than natural as the speaker struggles to maintain "
     "articulation under pressure. Pitch is conspicuously elevated with marked "
     "wavering, particularly on sustained vowels in price figures and key words. "
     "A mild but perceptible vocal tremor is present throughout. Voice quality "
     "is breathy and tight simultaneously. Breathing is audibly faster and "
     "slightly irregular — brief catches before price offers. "
     "The delivery should feel raw and barely held together. {knowledge_mod}"),

    # 0.85 – 0.95  |  Maximum stress. The speaker is at the absolute edge of composure.
    (0.96,
     "Speak at the absolute peak of emotional intensity — composure is fully "
     "overridden by urgency and distress. Pace is very rapid and erratic, "
     "around 145–160 words per minute, with words occasionally rushing together "
     "and pauses appearing only at desperate natural breath points. Pitch is "
     "markedly elevated and visibly unstable — wavering throughout the utterance "
     "with occasional pitch breaks on heavily stressed syllables, particularly "
     "price figures, budget references, or final pleas. Vocal tremor is persistent "
     "and clearly audible. Voice quality is strained and breathy — the sound of "
     "someone who has physically tensed their throat and chest. "
     "Breathing is irregular and audible; the speaker sounds slightly short of "
     "breath. Price offers sound like desperate pleas rather than negotiating "
     "positions. The voice should communicate that this person is one refusal "
     "away from walking out or giving in completely. {knowledge_mod}"),
]

_KNOWLEDGE_MODS = {
    "expert":   ("Even under this pressure, vocabulary stays precise and technical — "
                 "the speaker cites specifics rather than generalising. "
                 "Sentences remain syntactically organised despite the stress."),
    "high":     ("Speech is organised and vocabulary is clear; stress shows in "
                 "delivery speed and pitch rather than word choice."),
    "moderate": ("Occasional simpler vocabulary and slightly more repetitive "
                 "phrasing as the pressure affects articulation."),
    "low":      ("Under this stress, speech becomes noticeably fragmented — "
                 "shorter sentences, occasional mid-sentence restarts, "
                 "and simple vocabulary that signals the speaker is not "
                 "in a sophisticated negotiating frame of mind."),
}

_LEVERAGE_FLAVOUR = (
    "The stress arises from genuine budget pressure — this is someone trying to "
    "afford something they want but may not comfortably be able to pay for. "
    "The emotional subtext is: 'I want this, please work with me.'"
)

_MITIGATE_FLAVOUR = (
    "The stress arises from confusion and mild defensiveness — this buyer "
    "believed their offer was fair and is being corrected. "
    "The emotional subtext is: 'Why is this cheaper than I thought? "
    "Is something wrong with it?' Bewilderment, not desperation."
)


def get_buyer_instruction(
    vaii_raw: float,
    turn_index: int,
    knowledge_level: str = "moderate",
    strategy: str = "LEVERAGE",
) -> str:
    """
    Build the Qwen3-TTS voice instruction for a buyer turn.

    Args:
        vaii_raw:       Float 0.05-0.95 from turn["vaii"]["raw"]
        turn_index:     1, 3, 5, 7, 9, or 11
        knowledge_level: "expert"/"high"/"moderate"/"low" from seed
        strategy:       "LEVERAGE" or "MITIGATE" (the dialogue's correct_decision)
    """
    vaii = max(0.05, min(0.95, float(vaii_raw)))

    # Select band
    base = _BUYER_VAII_BANDS[-1][1]
    for threshold, template in _BUYER_VAII_BANDS:
        if vaii < threshold:
            base = template
            break

    # Knowledge modifier
    kl = knowledge_level.lower()
    km = _KNOWLEDGE_MODS.get(kl, _KNOWLEDGE_MODS["moderate"])
    base = base.replace("{knowledge_mod}", km)

    # Scenario flavour
    flavour = _LEVERAGE_FLAVOUR if strategy == "LEVERAGE" else _MITIGATE_FLAVOUR

    # Turn-position addendum
    if turn_index == 1:
        pos_note = ("This is the buyer's opening move — delivery should feel "
                    "intentional and prepared, not reactive.")
    elif turn_index == 3:
        pos_note = ("This is the buyer's second offer, made after the seller held "
                    "firm. There is the first real awareness that this will not "
                    "be easy — let that register subtly without overplaying it.")
    elif turn_index == 11:
        pos_note = ("This is the buyer's final offer before the closing turn. "
                    "Stress is beginning to resolve into either resignation or "
                    "acceptance — the frantic edge softens slightly, replaced by "
                    "a quieter, more final quality. VAII is capped at 0.45 "
                    "for this turn: deliver with diminishing intensity, "
                    "as if the buyer has made peace with the outcome.")
    elif turn_index in (5, 7, 9):
        pos_note = ("This is a mid-negotiation turn where the emotional arc "
                    "is fully engaged — the buyer is in the thick of it. "
                    "Let the VAII level drive the delivery without pulling back.")
    else:
        pos_note = ""

    parts = [base, flavour]
    if pos_note:
        parts.append(pos_note)

    return " ".join(parts)


# ── SELLER VAII → VOICE INSTRUCTION ───────────────────────────────────────────

_SELLER_VAII_BANDS = [
    # 0.05 – 0.12  |  Maximum deliberate calm. Used when buyer VAII is very high.
    # The strategic calm of an expert negotiator anchoring the emotional room.
    (0.12,
     "Deliver this line with absolute, deliberate composure — the studied calm of "
     "a professional who has heard every version of this negotiation before. "
     "Pace is intentionally slow, around 78–85 words per minute, with long "
     "unhurried pauses placed AFTER key facts and price statements — the silence "
     "is part of the argument. Pitch sits firmly at the lower third of the "
     "speaker's natural range, flat and unwavering; there is no rising inflection "
     "anywhere. Voice quality is full and resonant with no breathiness whatsoever. "
     "Diction is impeccable — each word placed with precision. "
     "The effect should be almost meditative: calm that does not come from "
     "indifference but from complete certainty. This delivery is specifically "
     "designed to de-escalate a stressed interlocutor by modelling "
     "extraordinary composure. {strategy_mod} {turn_mod}"),

    # 0.12 – 0.20  |  Confident professional. Core LEVERAGE post-crystallisation tone.
    (0.20,
     "Speak with measured professional confidence — the assured voice of someone "
     "presenting an indisputable fact. Pace is unhurried, around 88–95 words per "
     "minute. Pitch is stable and grounded, sitting in the lower-mid range with "
     "slight emphasis on price figures — delivered as statements, never as offers "
     "to be questioned. Pauses after price mentions are brief but definite, "
     "like a full stop in speech. Voice quality is clear and full; zero breathiness "
     "or strain. Tone is firm without aggression, warm without weakness — "
     "the voice of someone who will not be moved because they know they are right, "
     "not because they are being stubborn. {strategy_mod} {turn_mod}"),

    # 0.20 – 0.30  |  Measured/factual. Market data delivery. Mid-negotiation neutral.
    (0.30,
     "Speak in a measured, factual, slightly warm tone — the voice of a "
     "knowledgeable colleague sharing genuinely useful information. "
     "Pace is moderate, around 95–105 words per minute, with natural sentence-level "
     "pauses. Pitch is stable with mild variation for natural intonation but no "
     "dramatic peaks; slightly emphasise key price figures without being heavy-handed. "
     "Voice quality is clean and clear. Delivery feels like presenting evidence "
     "rather than arguing a case — there is no emotional charge, just confident "
     "factual grounding. The warmth prevents it from feeling cold or robotic; "
     "the seller is being helpful, not lecturing. {strategy_mod} {turn_mod}"),

    # 0.30 – 0.38  |  Neutral-observational. T2 UNDECIDED — holding position, gathering.
    (0.38,
     "Speak in a composed, professionally neutral tone — the voice of someone "
     "who has heard the opening offer and is presenting their counter without "
     "revealing any strategic read of the situation. Pace is relaxed, around "
     "100–108 words per minute. Pitch is even and non-committal — "
     "neither the assertiveness of a firm strategy nor the softness of "
     "accommodation. Voice quality is clear and pleasant. Delivery has a slight "
     "businesslike formality — courteous but giving nothing away. "
     "The asking price should be stated with quiet conviction, "
     "not as a challenge but as a natural starting point. {strategy_mod} {turn_mod}"),

    # 0.38 – 0.45  |  Gentle/unhurried. MITIGATE with stressed buyer. Supportive correction.
    (0.46,
     "Speak with gentle, unhurried warmth — the tone of a trusted advisor "
     "delivering genuinely good news to someone who does not yet know it is good. "
     "Pace is deliberately slow, around 82–90 words per minute, creating space "
     "and reducing any sense of pressure. Pitch is soft and slightly lower than "
     "normal conversation; there are no sharp emphases or sudden peaks. "
     "Voice quality is open and warm — no tightness, no edge. "
     "The delivery should feel like: 'I am on your side, and I am telling you "
     "that you do not need to spend that much.' "
     "Pauses before stating the fair price are generous — allowing the listener "
     "to feel the seller is choosing their words carefully out of consideration, "
     "not haste. Zero urgency. Zero pressure. Pure supportive correction. "
     "{strategy_mod} {turn_mod}"),
]

_SELLER_STRATEGY_MODS = {
    "LEVERAGE": (
        "Strategy context: the buyer is anchored below fair value. "
        "The seller's role is to hold firm at or near market price and "
        "guide the buyer upward through evidence, patience, and unwavering "
        "certainty — never through pressure or aggression. "
        "The calmness IS the argument: a seller who does not sound desperate "
        "signals that the price has nowhere to go but stay."
    ),
    "MITIGATE": (
        "Strategy context: the buyer is over-offering, anchored above fair value. "
        "The seller's role is to proactively correct the buyer's overpayment "
        "by citing market rate. The tone is genuinely helpful — "
        "the seller is doing the buyer a favour by keeping them from overpaying. "
        "Never cite demand, scarcity, or desirability — those justify high prices "
        "and belong to LEVERAGE. Frame everything around market fairness."
    ),
    "UNDECIDED": (
        "Strategy context: this is Turn 2 — the seller has heard only one "
        "buyer offer and has not yet committed to a strategy. "
        "The seller is observing, holding position at asking price, "
        "and giving no signal about how the negotiation will develop. "
        "The tone is professionally neutral — engaged but unreadable."
    ),
}

_SELLER_TURN_MODS = {
    2:  ("Turn 2 — UNDECIDED. First seller response after buyer's anchor. "
         "State the asking price with quiet confidence. "
         "No strategic reveal — purely professional and composed."),
    4:  ("Turn 4 — CRYSTALLISATION. This is the moment the seller commits "
         "to their strategy for the first time. Delivery should carry a "
         "barely perceptible increase in definitiveness compared to T2 — "
         "not louder or faster, but more resolved. "
         "The seller has decided. The tone reflects that decision."),
    6:  ("Turn 6 — first post-crystallisation turn. "
         "Strategy is now consistent and will not waver. "
         "Deliver with the quiet persistence of someone who has "
         "committed and is simply waiting for the other party to catch up."),
    8:  ("Turn 8 — mid-negotiation. The seller has held strategy for two turns. "
         "Patience is the dominant quality — not rigidity, but the "
         "unhurried certainty of someone who knows time is on their side."),
    10: ("Turn 10 — late negotiation. Agreement is close but not yet reached. "
         "Maintain full composure; do not introduce any urgency or relief. "
         "The seller who sounds relieved to be closing signals weakness. "
         "Sound exactly as calm as Turn 4."),
    12: ("Turn 12 — AGREEMENT. The deal is done. "
         "Delivery should carry the warmth of genuine satisfaction without "
         "a trace of triumph or relief. The tone is: 'This is the outcome "
         "that was always correct.' "
         "Pace can soften very slightly; the tension of negotiation "
         "is released into a comfortable, final resolution. "
         "Agreement words (deal / done / agreed) should land cleanly "
         "and with quiet finality."),
}


def get_seller_instruction(
    seller_vaii_raw: float,
    turn_index: int,
    strategy: str = "LEVERAGE",
    prior_buyer_vaii: Optional[float] = None,
) -> str:
    """
    Build the Qwen3-TTS voice instruction for a seller turn.

    Args:
        seller_vaii_raw:  Float 0.05-0.45 from turn["seller_vaii"]["raw"]
        turn_index:       2, 4, 6, 8, 10, or 12
        strategy:         "LEVERAGE", "MITIGATE", or "UNDECIDED"
        prior_buyer_vaii: The buyer's VAII from the immediately prior turn
                          (used to add the inverse-relationship note)
    """
    sv = max(0.05, min(0.45, float(seller_vaii_raw)))

    # Select band
    base = _SELLER_VAII_BANDS[-1][1]
    for threshold, template in _SELLER_VAII_BANDS:
        if sv < threshold:
            base = template
            break

    # Strategy modifier
    strat_mod = _SELLER_STRATEGY_MODS.get(strategy, _SELLER_STRATEGY_MODS["UNDECIDED"])
    base = base.replace("{strategy_mod}", strat_mod)

    # Turn modifier
    turn_mod = _SELLER_TURN_MODS.get(turn_index, "")
    base = base.replace("{turn_mod}", turn_mod)

    # Inverse-relationship note (core novelty of seller VAII)
    if prior_buyer_vaii is not None:
        bv = float(prior_buyer_vaii)
        if bv >= 0.65 and sv <= 0.15:
            inverse_note = (
                "CRITICAL DELIVERY NOTE: The buyer in the immediately preceding "
                f"turn was highly stressed (VAII={bv:.2f}). "
                "The seller's response must be strikingly calmer by contrast — "
                "the calmness is deliberate and strategic. "
                "Where the buyer was fast, be slow. Where the buyer was high-pitched, "
                "be low-pitched. Where the buyer was breathy or strained, "
                "be full and resonant. This inverse relationship is the "
                "debiasing mechanism: a seller who refuses to match the buyer's "
                "emotional pitch de-escalates the interaction and maintains "
                "rational framing for both parties."
            )
        elif bv >= 0.50 and sv <= 0.20:
            inverse_note = (
                f"The prior buyer turn was clearly stressed (VAII={bv:.2f}). "
                "The seller must be noticeably calmer — do not let any of "
                "the buyer's urgency colour the delivery. "
                "Slower pace and lower pitch than the preceding buyer turn."
            )
        elif bv < 0.35 and sv >= 0.30:
            inverse_note = (
                f"The buyer in the prior turn was calm (VAII={bv:.2f}). "
                "The seller can be slightly warmer and more conversational "
                "in this turn — a calm-to-calm exchange feels like a "
                "collaborative dialogue rather than a battle."
            )
        else:
            inverse_note = ""

        if inverse_note:
            base = base + " " + inverse_note

    return base


# ── PUBLIC API ─────────────────────────────────────────────────────────────────

def build_voice_instruction(
    turn: dict,
    dialogue_seed: dict,
) -> str:
    """
    Top-level function. Given a processed v2 turn dict and the seed block,
    return the complete voice instruction string for Qwen3-TTS.

    Args:
        turn:          One element from intermediate["processed"]
        dialogue_seed: intermediate["seed"]  (contains generation_params, pricing)
    """
    gp       = dialogue_seed.get("generation_params", {})
    strategy = gp.get("correct_decision", "LEVERAGE")
    kl       = gp.get("knowledge_level", "moderate")
    tidx     = turn["turn_index"]

    if turn["speaker"] == "buyer":
        vaii_raw = turn.get("vaii", {}).get("raw", 0.30)
        return get_buyer_instruction(
            vaii_raw=vaii_raw,
            turn_index=tidx,
            knowledge_level=kl,
            strategy=strategy,
        )
    else:
        sv_raw = turn.get("seller_vaii", {}).get("raw", 0.25)
        # Pull strategy from factor_state so T2 shows UNDECIDED correctly
        fs_decision = turn.get("factor_state", {}).get("decision", strategy)
        return get_seller_instruction(
            seller_vaii_raw=sv_raw,
            turn_index=tidx,
            strategy=fs_decision,
            prior_buyer_vaii=None,  # populated by generate_tts_v2.py at call time
        )


# ── SELF-TEST ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("BUYER INSTRUCTION SAMPLES")
    print("=" * 70)
    for vaii, label in [
        (0.10, "very calm expert"),
        (0.30, "composed"),
        (0.52, "mild pressure"),
        (0.68, "high stress"),
        (0.88, "max stress"),
    ]:
        print(f"\nVAII={vaii} ({label}):")
        print(get_buyer_instruction(vaii, turn_index=7, knowledge_level="moderate",
                                    strategy="LEVERAGE")[:220] + "…")

    print("\n" + "=" * 70)
    print("SELLER INSTRUCTION SAMPLES")
    print("=" * 70)
    for sv, label in [
        (0.08, "max calm — buyer was 0.72"),
        (0.18, "confident professional"),
        (0.25, "measured factual"),
        (0.35, "neutral T2"),
        (0.42, "gentle MITIGATE"),
    ]:
        print(f"\nSeller VAII={sv} ({label}):")
        print(get_seller_instruction(sv, turn_index=4, strategy="LEVERAGE",
                                     prior_buyer_vaii=0.72 if sv == 0.08 else None)[:220] + "…")

    print("\n" + "=" * 70)
    print("VOICE ASSIGNMENTS (sample)")
    print("=" * 70)
    for did in ["dialogue_0001", "dialogue_0042", "dialogue_3154"]:
        v = assign_voices(did)
        print(f"  {did}: buyer={v['buyer']}, seller={v['seller']}")
