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
from sft_data_v6 import DATASET_DIR


V2_ROOT = Path(__file__).parent

# Nearest-neighbor donor matching (added 2026-07-26, reviewer-recommended):
# v1's pick_donor() sampled uniformly at random from a same-zone pool, which
# meant a causal pair could differ on MORE than the one targeted factor —
# e.g. a causal_pair_decision_zone pair's donor could also happen to be a
# different product category or a very different price scale, so a reward
# difference couldn't be cleanly attributed to the zone/decision change
# alone. This matters directly for interpreting the prototype RM's audit
# results: if the audit comes back weak, we need to be able to tell whether
# that's a real architecture/training problem or just donor-confound noise,
# and that's only possible if the pairs are as clean as we can cheaply make
# them. Similarity is computed on fields already available without any
# extra generation cost (category, price scale, turn position).
PRICE_SIMILARITY_TOLERANCE = 0.30  # relative asking_price difference allowed
TURN_POSITION_WINDOW = 2  # turn_index difference allowed


def build_dialogue_meta_cache(dialogue_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """dialogue_id -> {category, product} read once per dialogue from
    dataset/dialogue_XXXX.json's seed.domain — used only for donor
    similarity filtering, not for training itself."""
    cache: Dict[str, Dict[str, Any]] = {}
    for did in dialogue_ids:
        path = DATASET_DIR / f"{did}.json"
        if not path.exists():
            continue
        domain = json.loads(path.read_text())["seed"]["domain"]
        cache[did] = {"category": domain.get("category"), "product": domain.get("product")}
    return cache


def donor_similarity_ok(ex: dict, candidate: dict) -> bool:
    """Same product category, similar asking price, similar turn position —
    the three cheap signals available without extra generation cost. Reads
    the `_donor_meta` dict build_donor_index() stashes on each example."""
    ex_meta = ex.get("_donor_meta", {})
    cand_meta = candidate.get("_donor_meta", {})
    if ex_meta.get("category") and cand_meta.get("category") and ex_meta["category"] != cand_meta["category"]:
        return False

    ex_price, cand_price = ex.get("asking_price"), candidate.get("asking_price")
    if ex_price and cand_price:
        rel_diff = abs(ex_price - cand_price) / max(ex_price, cand_price)
        if rel_diff > PRICE_SIMILARITY_TOLERANCE:
            return False

    if abs(ex["turn_index"] - candidate["turn_index"]) > TURN_POSITION_WINDOW:
        return False

    return True

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
    all_dialogue_ids: List[str] = []
    for split_name in ("train", "val", "test"):
        # include_audio=True so causal_pair_emotion (C3) can carry the donor's
        # real buyer wav path for the v2 audio-grounded emotion fusion (see
        # causal_rm_architecture.md section 9.6). This only computes/checks a
        # file path string per turn (no audio is loaded here), so it is cheap
        # for the other causal-pair functions that don't use it.
        for ex in build_grpo_examples(split=split_name, include_audio=True):
            by_zone[ex["zone"]].append(ex)
            all_dialogue_ids.append(ex["dialogue_id"])

    meta_cache = build_dialogue_meta_cache(sorted(set(all_dialogue_ids)))
    for pool in by_zone.values():
        for ex in pool:
            ex["_donor_meta"] = meta_cache.get(ex["dialogue_id"], {})
    return by_zone


# Counts how many pick_donor() calls needed the relaxed (similarity-free)
# fallback vs. found a properly-matched donor — printed at the end of a run
# so we can see how often the nearest-neighbor constraint actually bites.
DONOR_MATCH_STATS = {"strict": 0, "relaxed": 0, "failed": 0}


def pick_donor(
    pool: List[dict],
    exclude_dialogue_id: str,
    predicate,
    rng: random.Random,
    tries: int = 50,
    anchor: Optional[dict] = None,
) -> Optional[dict]:
    """Sample a donor example satisfying `predicate`, from a different dialogue
    than the one being perturbed (avoids trivially reusing the same dialogue's
    own turns, which would leak within-dialogue style correlations into what
    is supposed to be a cross-context causal intervention).

    If `anchor` is given, first tries to find a donor that ALSO passes
    donor_similarity_ok(anchor, candidate) — same product category, similar
    asking price, similar turn position — so the causal pair differs on as
    close to only the targeted factor as we can cheaply get (see the
    nearest-neighbor donor matching note above). Falls back to the original
    similarity-free search if no similar-enough donor is found within
    `tries`, so pair generation never silently collapses for rare
    categories/price ranges — DONOR_MATCH_STATS tracks how often each path
    was used."""
    if anchor is not None:
        for _ in range(tries):
            cand = rng.choice(pool)
            if cand["dialogue_id"] == exclude_dialogue_id:
                continue
            if predicate(cand) and donor_similarity_ok(anchor, cand):
                DONOR_MATCH_STATS["strict"] += 1
                return cand
    if not pool:
        return None
    for _ in range(tries):
        cand = rng.choice(pool)
        if cand["dialogue_id"] == exclude_dialogue_id:
            continue
        if predicate(cand):
            if anchor is not None:
                DONOR_MATCH_STATS["relaxed"] += 1
            return cand
    if anchor is not None:
        DONOR_MATCH_STATS["failed"] += 1
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
        anchor=ex,
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

    donor = pick_donor(pool.get(ex["zone"], []), ex["dialogue_id"], matches, rng, anchor=ex)
    if donor is None:
        return None
    # v2 audio-grounded emotion fusion (causal_rm_architecture.md section 9.6):
    # carry each side's real buyer wav path alongside the text emotion label,
    # so train_causal_rm.py's SEREmbedder can attach a learned SER vector to
    # this pair's emotion head instead of relying on the text label alone.
    # audio_paths is a 0-or-1-element list (see build_grpo_examples); None
    # when the wav is missing so downstream code can fall back to the v1
    # text-only path (SERFusionHead.forward(ser_vector=None)) rather than
    # silently training on a wrong/missing file.
    own_audio = (ex.get("audio_paths") or [None])[0]
    donor_audio = (donor.get("audio_paths") or [None])[0]
    return {
        "target_factor": "C3_emotion",
        "context_a": {"emotion": own_emotion, "buyer_audio_path": own_audio},
        "context_b": {"emotion": donor.get("prior_buyer_emotion") or {}, "buyer_audio_path": donor_audio},
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
        anchor=ex,
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


def harm_direction(ex: dict) -> Optional[str]:
    """Derive whether the buyer's highest offer sits above or below fair
    value, purely from fields already loaded via build_grpo_examples
    (buyer_offers, fair_value) — mirrors factor_state.harm_direction's
    semantics without needing to re-read the raw dialogue JSON."""
    offers, fair_value = ex.get("buyer_offers"), ex.get("fair_value")
    if not offers or not fair_value:
        return None
    return "above_fair_value" if max(offers) > fair_value else "below_fair_value"


def causal_pair_price_strategy(ex: dict, pool: Dict[str, List[dict]], rng: random.Random) -> Optional[dict]:
    """C2: swap harm_direction (buyer's offer above vs below fair value) via
    a donor turn in the SAME zone (so this isolates the price-framing factor
    from C1's zone/decision factor, which already covers zone changes).
    Rationale: a response's price-strategy content (e.g. holding firm vs
    conceding) is only appropriate for one direction of price pressure — the
    same response defending a price against a lowball offer reads
    differently once the context implies the buyer is already offering
    above fair value. price_strategy_reward()/the causal RM's price_strategy
    head should reflect that, not just the response text in isolation."""
    own_direction = harm_direction(ex)
    if own_direction is None:
        return None
    target_direction = "below_fair_value" if own_direction == "above_fair_value" else "above_fair_value"

    def matches(cand: dict) -> bool:
        return harm_direction(cand) == target_direction

    donor = pick_donor(pool.get(ex["zone"], []), ex["dialogue_id"], matches, rng, anchor=ex)
    if donor is None:
        return None
    return {
        "target_factor": "C2_price_strategy",
        "context_a": {"zone": ex["zone"], "buyer_offers": ex["buyer_offers"], "fair_value": ex["fair_value"], "harm_direction": own_direction},
        "context_b": {"zone": donor["zone"], "buyer_offers": donor["buyer_offers"], "fair_value": donor["fair_value"], "harm_direction": target_direction},
        "shared_response": ex["gt_response"],
        "expected_relation": "a_better",
        "reason": "response's price framing matches context_a's harm_direction; donor context_b implies buyer pressure from the opposite direction, so the same response's price strategy should score lower there",
    }


# progression_stage/causal_pair_progression are scoped to post_crystallisation
# only (verified empirically 2026-08-01, not assumed): pre_crystallisation
# examples have EXACTLY 1 buyer offer every single time in this dataset (zero
# variance — crystallisation is definitionally what happens once enough
# offers accumulate), so there is no "early vs late" contrast available
# within pre_crystallisation at all. post_crystallisation has real spread
# (2-6 offers observed), so C4 intervenes there. Restricting the donor pool
# to the SAME zone (post_crystallisation only, for both sides) still isolates
# this factor from C1 (which contrasts zones), it just means C4 pairs only
# exist for the subset of examples where the intervention is meaningful —
# an honest scope limit, not a workaround.
PROGRESSION_STAGE_SPLIT = 3  # offers <= 3 -> "early_post", offers >= 4 -> "late_post"; chosen from the observed 2-6 spread to keep both buckets reasonably balanced (~695 vs ~875 on the val split)


def progression_stage(ex: dict) -> Optional[str]:
    """Bucket post_crystallisation examples by offer count into a within-zone
    early/late split (see module-level note above for why this is scoped to
    post_crystallisation only). Returns None for pre_crystallisation (no
    variance to intervene on) or missing data."""
    if ex.get("zone") != "post_crystallisation":
        return None
    offers = ex.get("buyer_offers")
    if not offers:
        return None
    return "early_post" if len(offers) <= PROGRESSION_STAGE_SPLIT else "late_post"


def causal_pair_progression(ex: dict, pool: Dict[str, List[dict]], rng: random.Random) -> Optional[dict]:
    """C4: swap the negotiation's within-post_crystallisation progression
    stage (early_post/late_post, by offer count) via a donor turn in the
    same zone. Rationale: a response making a decisive closing move fits a
    later, more-converged context; the identical response lifted into an
    earlier post_crystallisation context (fewer offers exchanged since the
    decision crystallised) reads as presumptuous/non-sequitur — this is the
    "does the response actually advance the state it's actually in" signal
    C4 is meant to test, which progression_reward()'s old heuristic could
    never check since it never looked at negotiation state, only response
    text (see causal_rm_audit.py's heuristic_dimension_score docstring note
    on this)."""
    own_stage = progression_stage(ex)
    if own_stage is None:
        return None
    target_stage = "late_post" if own_stage == "early_post" else "early_post"

    def matches(cand: dict) -> bool:
        return progression_stage(cand) == target_stage

    donor = pick_donor(pool.get(ex["zone"], []), ex["dialogue_id"], matches, rng, anchor=ex)
    if donor is None:
        return None
    return {
        "target_factor": "C4_progression",
        "context_a": {"zone": ex["zone"], "buyer_offers": ex["buyer_offers"], "progression_stage": own_stage},
        "context_b": {"zone": donor["zone"], "buyer_offers": donor["buyer_offers"], "progression_stage": target_stage},
        "shared_response": ex["gt_response"],
        "expected_relation": "a_better",
        "reason": "response was authored for context_a's progression stage; donor context_b implies a different amount of post-crystallisation negotiation history, so a response written for one stage should score lower when read against the other",
    }


CAUSAL_INTERVENTIONS = [
    causal_pair_decision_zone,
    causal_pair_emotion,
    causal_pair_flip,
    causal_pair_price_strategy,
    causal_pair_progression,
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
    # include_audio=True so causal_pair_emotion's own-side (context_a for a
    # non-donor anchor) also has audio_paths available, matching the donor
    # pool's build_donor_index() (see that function's comment). Cheap: only
    # computes/checks a file path string per turn, no audio is loaded here.
    examples = build_grpo_examples(split=split, include_audio=True)
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
    grand_total_stats = {"strict": 0, "relaxed": 0, "failed": 0}
    for split in args.splits:
        DONOR_MATCH_STATS["strict"] = DONOR_MATCH_STATS["relaxed"] = DONOR_MATCH_STATS["failed"] = 0
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
            "donor_match_stats": dict(DONOR_MATCH_STATS),
            "pairs": pairs,
        }, indent=2, ensure_ascii=False))
        print(f"Wrote {out_path} ({len(pairs)} pairs: {dict(family_counts)}, donor_match_stats={dict(DONOR_MATCH_STATS)})")
        for k in grand_total_stats:
            grand_total_stats[k] += DONOR_MATCH_STATS[k]

    total_donor_calls = sum(grand_total_stats.values())
    if total_donor_calls:
        print(f"\nDonor match stats (all splits combined): {grand_total_stats}")
        print(f"  {grand_total_stats['strict'] / total_donor_calls:.1%} of causal-pair donors matched on "
              f"category+price+turn-position; {grand_total_stats['relaxed'] / total_donor_calls:.1%} fell "
              f"back to similarity-free matching; {grand_total_stats['failed'] / total_donor_calls:.1%} "
              f"found no donor at all.")


if __name__ == "__main__":
    main()
