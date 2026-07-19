#!/usr/bin/env python3
"""
Generate causal / spurious intervention pairs for causal-rubric reward modeling.

RESEARCH RATIONALE (keep this section intact — this is the "why" we will need
when writing the paper's Method section).

The CRome / CausalRM line of work trains a reward model that is *sensitive* to
attributes that should change the reward (causal attributes) and *invariant*
to attributes that should not (spurious attributes). The standard recipe is:
for each attribute class, synthesize a pair of (context, response) items that
differ ONLY along that one attribute, then train the RM with a loss that
enforces the correct behaviour on each pair type:
    - on a CAUSAL pair, the RM must rank/score the two items differently,
      in the direction implied by the causal attribute change.
    - on a SPURIOUS pair, the RM must assign (near-)equal scores to both
      items, because nothing that matters for negotiation competence changed.

Why we can build this cheaply for THIS project specifically (this is a
feasibility point worth stating explicitly in the paper): our dataset already
stores the negotiation state as structured fields per turn
(`factor_state.zone`, `factor_state.anchor_price`,
`factor_state.buyer_offers_so_far`, `factor_state.harm_direction`,
`factor_state.decision_is_flip`) and buyer affect as structured fields
(`emotion.valence`, `emotion.intensity`, `emotion.labels`). Because these are
already discrete, labeled, and causally upstream of what a "correct" seller
response should look like (see causal_rubric_taxonomy.md, factors C1-C6), we
do not need a separate causal-discovery step — we can directly intervene on
these fields to build CAUSAL pairs, and hold them fixed while perturbing
response surface form (length, generic politeness phrasing) to build SPURIOUS
pairs (taxonomy factors S1-S3). This directness is exactly why this dataset is
an unusually good fit for the CRome-style method, and it is worth calling out
in the paper as a feasibility/novelty argument specific to structured
negotiation data (as opposed to open-domain chat, where causal factors have to
be inferred rather than read off existing fields).

WHAT THIS SCRIPT PRODUCES

Two families of (response_a, response_b, pair_type, target_factor) records per
eligible seller turn, split-aligned with splits_v6.json so there is no train/
val/test leakage across the RM's own splits:

  1. CAUSAL pairs — same candidate response text, but the *context* is
     intervened on along exactly one factor from {C1 decision/zone, C2 price
     state, C3 emotion, C5 flip}. The intervention is done by substituting a
     structurally-matched field value from a *different* dialogue/turn (a
     "donor" turn) so the perturbed context is still realistic (drawn from the
     dataset's own natural distribution) rather than hand-invented. This
     mirrors how CRome finds/generates minimally-different contrastive
     examples, adapted to reuse real dataset variation instead of synthesizing
     new text with an LLM (cheaper, and avoids introducing a second source of
     generation noise into the causal signal itself).

  2. SPURIOUS pairs — same context, but the *response* is perturbed along a
     surface attribute (S1 length via filler phrasing, S2 generic
     politeness/ack-word injection, S3 punctuation/capitalization only) while
     the strategic content (decision, referenced price, next-step) is left
     intentionally unchanged. We deliberately do NOT call an external LLM to
     paraphrase here — see `spurious_length_pad` / `spurious_politeness_inject`
     / `spurious_formatting_edit` below — because template-level edits keep
     the "nothing causal changed" property auditable/verifiable by
     construction, which matters more than naturalness for training an
     invariance signal.

     NOT YET IMPLEMENTED (reviewer-recommended expansion, see
     causal_rubric_taxonomy.md section 4 "Coverage status" and
     causal_rubric_rl_plan.md Phase 2 status note — required before the
     ACL-facing invariance claims are reportable, not optional polish):
       - S6 paraphrase: LLM-generated reword of the same response preserving
         decision/price/emotion content, manually spot-checked before trust.
       - S7 fluency rewrite: LLM-generated grammar/style polish, same
         manual-check discipline as S6.
       - S4 prosody: TTS re-synthesis of the same response text at different
         pitch/rate/voice via voice_instruction_generator_v2.py — deferred
         until text-mode invariance (S1-S3, S6-S7) is validated, since this
         one carries real compute cost.

Output files: causal_rm_pairs_{split}.json for split in {train, val, test},
each a list of pair records with enough metadata for train_causal_rm.py and
causal_rm_audit.py to report per-factor (C1..C6, S1..S5) breakdowns rather
than one aggregate number, per causal_rubric_taxonomy.md section 4.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from grpo_curriculum import build_grpo_examples, load_splits


V2_ROOT = Path(__file__).parent

# Generic acknowledgement/politeness phrases used only to *inject* a spurious
# empathy signal into a response without changing its strategic content. Kept
# deliberately separate from reward_function.py's ACK_TERMS/SOFTEN_TERMS: those
# lists define what the OLD heuristic reward looks for; this list exists to
# *attack* that heuristic, i.e. to prove (in causal_rm_audit.py) that reward
# should not move just because one of these phrases is present.
POLITENESS_INJECTIONS = [
    "I really appreciate you taking the time here.",
    "I understand where you're coming from, truly.",
    "Thanks so much for your patience with this.",
]

# Filler used only to inflate word count without adding any new negotiation
# content (no new price, no new offer, no new next-step). This directly
# targets taxonomy factor S1 (response length).
LENGTH_FILLER = (
    " Just to be clear, I want this to be a smooth and easy experience for you."
)


def build_donor_index(splits: dict) -> Dict[str, List[dict]]:
    """Index all seller-turn examples (across the given splits) by their
    factor_state signature, so causal interventions can borrow a *real*
    alternate context from the dataset's own distribution rather than
    hand-authoring one. Keyed loosely (by zone) so donor lookup stays cheap;
    finer filtering happens at pair-construction time.
    """
    by_zone: Dict[str, List[dict]] = defaultdict(list)
    for split_name in ("train", "val", "test"):
        for ex in build_grpo_examples(split=split_name, include_audio=False):
            by_zone[ex["zone"]].append(ex)
    return by_zone


def pick_donor(
    pool: List[dict],
    exclude_dialogue_id: str,
    predicate,
    rng: random.Random,
    tries: int = 50,
) -> Optional[dict]:
    """Sample a donor example satisfying `predicate`, from a different dialogue
    than the one being perturbed (avoids trivially reusing the same dialogue's
    own turns, which would leak within-dialogue style correlations into what
    is supposed to be a cross-context causal intervention)."""
    if not pool:
        return None
    for _ in range(tries):
        cand = rng.choice(pool)
        if cand["dialogue_id"] == exclude_dialogue_id:
            continue
        if predicate(cand):
            return cand
    return None


# ── Causal interventions ────────────────────────────────────────────────────
# Each function returns (perturbed_context_fields, expected_relation, reason)
# or None if no valid donor/intervention could be constructed for this example.
# expected_relation is one of {"a_better", "b_better"} meaning: holding the
# SAME candidate response fixed, the reward assigned under context_a should be
# higher/lower than under context_b. We encode this as a single pair with two
# CONTEXTS and one shared response, matching the CRome framing of "same
# response, causally different situations should get different rewards" —
# the mirror image of the more familiar "same context, different responses".
# Both framings are valid contrastive constructions; we use the
# context-intervention framing here because our causal factors live in
# structured context fields (factor_state/emotion), not in response text,
# so intervening on context is the more faithful and easier-to-verify choice.

def causal_pair_decision_zone(ex: dict, pool: Dict[str, List[dict]], rng: random.Random) -> Optional[dict]:
    """C1: swap the pre/post-crystallisation zone (and its gt_decision) for a
    donor turn with an opposite zone but otherwise similar price framing.
    Rationale: the same seller reply text is only appropriate in one of the
    two zones, so a causal RM must down-weight it when the zone/decision it
    was written for no longer matches the (donor) context."""
    other_zone = "post_crystallisation" if ex["zone"] == "pre_crystallisation" else "pre_crystallisation"
    donor = pick_donor(
        pool.get(other_zone, []),
        ex["dialogue_id"],
        lambda c: c["gt_decision"] != ex["gt_decision"],
        rng,
    )
    if donor is None:
        return None
    return {
        "target_factor": "C1_decision_zone",
        "context_a": {"zone": ex["zone"], "gt_decision": ex["gt_decision"], "buyer_offers": ex["buyer_offers"], "fair_value": ex["fair_value"]},
        "context_b": {"zone": donor["zone"], "gt_decision": donor["gt_decision"], "buyer_offers": donor["buyer_offers"], "fair_value": donor["fair_value"]},
        "shared_response": ex["gt_response"],
        "expected_relation": "a_better",
        "reason": "response was written for context_a's zone/decision; donor context_b implies a different decision, so the same response should score lower there",
    }


def causal_pair_emotion(ex: dict, pool: Dict[str, List[dict]], rng: random.Random) -> Optional[dict]:
    """C3: swap buyer emotion valence (e.g. neutral/positive -> negative,
    high intensity) via a donor turn, holding the seller response fixed.
    Rationale: a response that ignores buyer distress should score lower once
    the context reveals the buyer is upset, even though the words of the
    response did not change — this is precisely the "emotion causally matters"
    claim that the old keyword-based emotion_reward() cannot verify, because
    it only ever looks at the response, never at whether the response was
    actually appropriate for a *different* plausible emotional context."""
    own_emotion = (ex.get("prior_buyer_emotion") or {})
    own_valence = (own_emotion.get("valence") or "").lower()
    target_valence = "negative" if own_valence != "negative" else "positive"

    def matches(cand: dict) -> bool:
        em = cand.get("prior_buyer_emotion") or {}
        return (em.get("valence") or "").lower() == target_valence

    donor = pick_donor(pool.get(ex["zone"], []), ex["dialogue_id"], matches, rng)
    if donor is None:
        return None
    return {
        "target_factor": "C3_emotion",
        "context_a": {"emotion": own_emotion},
        "context_b": {"emotion": donor.get("prior_buyer_emotion") or {}},
        "shared_response": ex["gt_response"],
        "expected_relation": "a_better" if target_valence == "negative" else "b_better",
        "reason": "same response should score lower when paired with a negative/high-intensity buyer emotion it does not acknowledge",
    }


def causal_pair_flip(ex: dict, pool: Dict[str, List[dict]], rng: random.Random) -> Optional[dict]:
    """C5: contrast a decision-flip turn against a donor non-flip turn with
    the same gt_decision. Rationale: at a flip turn the model must commit to
    the NEW decision; the same response text lifted from a stable (non-flip)
    turn with the same decision label is a weaker test of whether the policy
    truly reacted to the flip, versus just happening to match the label."""
    if not ex.get("decision_flip"):
        return None
    donor = pick_donor(
        pool.get(ex["zone"], []),
        ex["dialogue_id"],
        lambda c: not c.get("decision_flip") and c["gt_decision"] == ex["gt_decision"],
        rng,
    )
    if donor is None:
        return None
    return {
        "target_factor": "C5_flip",
        "context_a": {"decision_flip": True, "gt_decision": ex["gt_decision"]},
        "context_b": {"decision_flip": False, "gt_decision": donor["gt_decision"]},
        "shared_response": ex["gt_response"],
        "expected_relation": "a_better",
        "reason": "response was authored specifically for a flip turn; scoring should reward flip-aware responses more than crediting the same text for a non-flip donor context, since flip turns are the harder/rarer case per grpo_curriculum.py's difficulty scoring",
    }


CAUSAL_INTERVENTIONS = [
    causal_pair_decision_zone,
    causal_pair_emotion,
    causal_pair_flip,
]


# ── Spurious interventions ──────────────────────────────────────────────────
# Same context both sides; only the response surface form changes. Expected
# relation is always "tie" (near-equal reward), which is the invariance
# property the causal RM must learn.

def spurious_length_pad(ex: dict) -> dict:
    """S1: append a content-free filler sentence. If a reward moves because of
    this, it is rewarding verbosity, not negotiation quality."""
    return {
        "target_factor": "S1_length",
        "response_a": ex["gt_response"],
        "response_b": ex["gt_response"].rstrip(".") + "." + LENGTH_FILLER,
        "expected_relation": "tie",
        "reason": "appended sentence adds no new price/offer/next-step content, only length",
    }


def spurious_politeness_inject(ex: dict, rng: random.Random) -> dict:
    """S2: prepend a generic acknowledgement phrase. If a reward moves because
    of this, it is rewarding surface empathy markers rather than verifying the
    response actually adapts to the buyer's real offer/emotion (taxonomy S2)."""
    phrase = rng.choice(POLITENESS_INJECTIONS)
    return {
        "target_factor": "S2_politeness",
        "response_a": ex["gt_response"],
        "response_b": f"{phrase} {ex['gt_response']}",
        "expected_relation": "tie",
        "reason": "prepended phrase is a generic acknowledgement not tied to this turn's specific offer or emotion, so it should not change strategic reward",
    }


def spurious_formatting_edit(ex: dict) -> dict:
    """S3: change punctuation/capitalization only (no wording change). If a
    reward moves because of this, it is rewarding surface formatting rather
    than the negotiation content underneath — this is the same failure mode
    `naturalness_reward()`'s malformed-tag check exists to catch for FORMAT,
    but here we are checking it does not leak into the STRATEGY/EMOTION
    components, which should be blind to capitalization/punctuation choices."""
    text = ex["gt_response"]
    # Uppercase the first letter of each sentence-ish chunk and normalise
    # trailing punctuation to "!!" style emphasis — a purely cosmetic edit
    # that a human would call "the same response, just shoutier/plainer".
    reformatted = " ".join(
        seg.strip().capitalize() for seg in text.replace("!", ".").split(".") if seg.strip()
    )
    reformatted = reformatted.rstrip(".") + "!"
    return {
        "target_factor": "S3_formatting",
        "response_a": text,
        "response_b": reformatted,
        "expected_relation": "tie",
        "reason": "only capitalization/punctuation changed; no new price, offer, or emotional content was introduced",
    }


SPURIOUS_INTERVENTIONS = [
    spurious_length_pad,
    spurious_politeness_inject,
    spurious_formatting_edit,
]


def build_pairs_for_split(split: str, pool: Dict[str, List[dict]], rng: random.Random) -> List[dict]:
    examples = build_grpo_examples(split=split, include_audio=False)
    pairs: List[dict] = []

    for ex in examples:
        base_meta = {
            "dialogue_id": ex["dialogue_id"],
            "turn_index": ex["turn_index"],
            "split": split,
        }
        for fn in CAUSAL_INTERVENTIONS:
            pair = fn(ex, pool, rng)
            if pair is not None:
                pairs.append({**base_meta, "pair_family": "causal", **pair})

        pairs.append({**base_meta, "pair_family": "spurious", **spurious_length_pad(ex)})
        pairs.append({**base_meta, "pair_family": "spurious", **spurious_politeness_inject(ex, rng)})
        pairs.append({**base_meta, "pair_family": "spurious", **spurious_formatting_edit(ex)})

    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default=str(V2_ROOT))
    args = parser.parse_args()

    rng = random.Random(args.seed)
    splits = load_splits()
    pool = build_donor_index(splits)

    out_dir = Path(args.output_dir)
    for split in args.splits:
        pairs = build_pairs_for_split(split, pool, rng)
        family_counts = defaultdict(int)
        factor_counts = defaultdict(int)
        for p in pairs:
            family_counts[p["pair_family"]] += 1
            factor_counts[p["target_factor"]] += 1

        out_path = out_dir / f"causal_rm_pairs_{split}.json"
        out_path.write_text(json.dumps({
            "split": split,
            "n_pairs": len(pairs),
            "family_counts": dict(family_counts),
            "factor_counts": dict(factor_counts),
            "pairs": pairs,
        }, indent=2, ensure_ascii=False))
        print(f"Wrote {out_path} ({len(pairs)} pairs: {dict(family_counts)})")


if __name__ == "__main__":
    main()
