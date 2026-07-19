#!/usr/bin/env python3
"""
sft_harmonise_reasoning.py — Normalise all non-canonical reasoning openings
to the standard "Turn N: Buyer offers $X. ..." format.

Usage:
  python3 Qwen3-tts/v2/sft_harmonise_reasoning.py --dry_run   # preview only
  python3 Qwen3-tts/v2/sft_harmonise_reasoning.py --backup    # apply + backup
  python3 Qwen3-tts/v2/sft_harmonise_reasoning.py             # apply
"""

import argparse
import json
import re
import shutil
from pathlib import Path

INTERM_DIR = Path(__file__).parent / "intermediate"

# ── Canonical check ───────────────────────────────────────────────────────────
CANONICAL = re.compile(r"^Turn \d+: Buyer offers \$", re.IGNORECASE)

# ── Split point: where the analytical body begins ────────────────────────────
_BODY_KEYWORDS = [
    "anchor price", "anchor is", "this sets the anchor",
    "fair value", "the anchor", "→", "harm direction",
]

def _body_start(text: str) -> int:
    tl = text.lower()
    best = len(text)
    for kw in _BODY_KEYWORDS:
        idx = tl.find(kw)
        if 0 < idx < best:
            best = idx
    # Walk back over any leading "The " so it is kept
    if best < len(text) and best >= 4 and text[best - 4:best].lower() == "the ":
        best -= 4
    return best

# ── Offer extraction ──────────────────────────────────────────────────────────

def _extract_offers(text: str, seller_turn: int | None = None) -> dict[int, str]:
    """
    Extract {buyer_turn: price_str} from the offer-summary portion of a reasoning.
    seller_turn is used for context-aware patterns (e.g. "latest offer is $X" → T=seller-1).
    Returns dict sorted by turn.
    """
    found: dict[int, str] = {}

    def add(t_str, p_str):
        t = int(t_str)
        p = p_str.replace(",", "")
        if t not in found:
            found[t] = p

    # ── Explicit Turn N patterns ──────────────────────────────────────────────

    # "Turn 3: Buyer offers/offered $630"  /  "Turn 3: $630"  /  "Turn 3 [BUYER]: Offer $630"
    for m in re.finditer(
        r'Turn (\d+)(?:\s+\[.*?\])?:\s+(?:Buyer (?:offers?|offered)|Offer)?\s*\$?([\d,]+)',
        text, re.I
    ):
        add(m.group(1), m.group(2))

    # "Turn 3: Buyer increased/moved/went/raised/dropped to $630"
    for m in re.finditer(
        r'Turn (\d+):\s+Buyer\s+\w+(?:\s+\w+)?\s+(?:to|at)\s+\$?([\d,]+)', text, re.I
    ):
        add(m.group(1), m.group(2))

    # "Turn 3 offer: $630" / "Turn 3 [BUYER] offer: $630"
    # "Turn 1 buyer offer is $200" / "Turn 1 offer from buyer is $200" / "Turn 1 offer of $200"
    for m in re.finditer(
        r'Turn (\d+)(?:\s+\[.*?\])?\s+(?:buyer\s+)?offer(?:\s+(?:from buyer|of|from))?\s*(?:[: ]|(?=\$))\s*(?:is\s+|was\s+)?\$?([\d,]+)',
        text, re.I
    ):
        add(m.group(1), m.group(2))

    # "Turn 1 offer was/is $500"  /  "Turn 1 was/is $500"  /  "Turn 1 - $500"
    for m in re.finditer(
        r'Turn (\d+)(?:\s+offer)?\s*(?:-+|was|is)\s*\$?([\d,]+)', text, re.I
    ):
        add(m.group(1), m.group(2))

    # "Turn 1 at $500" / "T1: $500" / "T1 $500"
    for m in re.finditer(r'T(?:urn )?(\d+)(?:[: ]+| at )\$?([\d,]+)', text, re.I):
        add(m.group(1), m.group(2))

    # "$630 (T3)" / "$630 (Turn 3)"
    for m in re.finditer(r'\$?([\d,]+)\s*\(T(?:urn )?(\d+)\)', text, re.I):
        add(m.group(2), m.group(1))

    # "$630 in T3" / "$630 in Turn 3"
    for m in re.finditer(r'\$?([\d,]+)\s+in\s+T(?:urn )?(\d+)', text, re.I):
        add(m.group(2), m.group(1))

    # "Buyer offered/offers $X in/at Turn N"
    for m in re.finditer(
        r'Buyer (?:offered|offers)\s+\$?([\d,]+)\s+(?:in |at )?T(?:urn )?(\d+)', text, re.I
    ):
        add(m.group(2), m.group(1))

    # ── T1-specific patterns ──────────────────────────────────────────────────

    # "initial/first/opening offer is/was/of $X" → T1
    for m in re.finditer(
        r"(?:buyer'?s?\s+)?(?:initial|first|opening) offer (?:is|was|of|at)\s+\$?([\d,]+)", text, re.I
    ):
        add("1", m.group(1))

    # "initial anchor of/at/is $X" → T1 (anchor = first offer)
    for m in re.finditer(r'initial anchor (?:of|at|is)\s+\$?([\d,]+)', text, re.I):
        add("1", m.group(1))

    # "starting at $X" / "started at $X" → T1
    for m in re.finditer(r'start(?:ing|ed) at \$?([\d,]+)', text, re.I):
        add("1", m.group(1))

    # "after starting at $X" → T1
    for m in re.finditer(r'after starting (?:at )?\$?([\d,]+)', text, re.I):
        add("1", m.group(1))

    # ── Chronological price lists ─────────────────────────────────────────────

    # "offers have been/are $X, $Y, and now $Z" / "has offered $X, $Y" /
    # "has made offers of $X, $Y" / "offers so far are $X"
    # → find trigger, grab all prices in that sentence → T1, T3, T5, ...
    trigger = re.search(
        r"(?:buyer'?s?\s+)?(?:"
        r"offers?\s+(?:have been|has been|so far[: ]+|are|so far are?|have progressed)\s*"
        r"|has (?:offered|made offers of)\s+"
        r")",
        text, re.I
    )
    if trigger:
        rest = text[trigger.end():]
        sentence = rest.split(".")[0]
        prices = [p.replace(",","") for p in re.findall(r'\$?([\d,]+)', sentence)]
        for i, p in enumerate(prices):
            add(str(1 + i * 2), p)

    # "with previous offers of/being $A, $B, $C" → T1, T3, T5, ...
    # "after initial offers of $A, $B" → T1, T3, ...
    for mo in re.finditer(
        r'(?:with previous offers? (?:of|being)|after initial offers? of)\s+'
        r'((?:\$?[\d,]+(?:,\s*(?:and\s+)?\$?[\d,]+)*))',
        text, re.I
    ):
        prices = [p.replace(",","") for p in re.findall(r'\$?([\d,]+)', mo.group(1))]
        for i, p in enumerate(prices):
            add(str(1 + i * 2), p)

    # "following/after (previous/prior) offers of $A, $B[, and $C]" → T1, T3, T5, ...
    for m in re.finditer(
        r'(?:following|after) (?:previous |prior )?offers?(?: of)?\s+'
        r'((?:\$?[\d,]+(?:,\s*(?:and\s+)?\$?[\d,]+)*))',
        text, re.I
    ):
        prices = [p.replace(",","") for p in re.findall(r'\$?([\d,]+)', m.group(1))]
        for i, p in enumerate(prices):
            add(str(1 + i * 2), p)

    # "has offered $A, $B, $C, and now $X" → T1=$A, T3=$B, ... ; Tcur=$X
    mo = re.search(
        r'has offered ((?:\$?[\d,]+(?:,\s+)?)+)(?:,\s+)?and now \$?([\d,]+)',
        text, re.I
    )
    if mo:
        prev_prices = [p.replace(",","") for p in re.findall(r'\$?([\d,]+)', mo.group(1))]
        for i, p in enumerate(prev_prices):
            add(str(1 + i * 2), p)
        # "now" price is at current buyer turn — added later by context-aware section

    # "following their initial $X offer[, $Y, $Z]" → T1=X, then T3, T5, ...
    m = re.search(
        r'following (?:their |an? )?(?:initial )?\$?([\d,]+) offer(?:,\s*([\$\d,\s]+))?',
        text, re.I
    )
    if m:
        add("1", m.group(1))
        if m.group(2):
            extras = [p.replace(",","") for p in re.findall(r'\$?([\d,]+)', m.group(2))]
            for i, p in enumerate(extras):
                add(str(3 + i * 2), p)

    # "moved/decreased/increased/progressed from $A to/through $B to/through $C" → T1, T3, T5,...
    # "moving from an initial $A through $B and $C" → same
    mo = re.search(
        r'(?:mov(?:ed|ing)|decreased|increased|gone|dropped|raised|reduced|progressed)\s+'
        r'(?:their (?:offer )?)?from\s+(?:an\s+initial\s+)?'
        r'(\$?[\d,]+(?:\s+(?:to|through)\s+\$?[\d,]+)+)',
        text, re.I
    )
    if mo:
        prices = [p.replace(",","") for p in re.findall(r'\$?([\d,]+)', mo.group(1))]
        for i, p in enumerate(prices):
            add(str(1 + i * 2), p)

    # "after $A and $B" / "after $A, $B" → chronological T1, T3, ... before current
    mo = re.search(r'\bafter\s+((?:\$?[\d,]+(?:,?\s+(?:and\s+)?\$?[\d,]+)+))', text, re.I)
    if mo:
        prices = [p.replace(",","") for p in re.findall(r'\$?([\d,]+)', mo.group(1))]
        for i, p in enumerate(prices):
            add(str(1 + i * 2), p)

    # ── Context-aware: seller_turn known ─────────────────────────────────────
    if seller_turn and seller_turn >= 2:
        buyer_cur_turn = seller_turn - 1

        # Single offer at seller T2 → T1=$X
        if seller_turn == 2:
            for pat in [
                r'buyer has (?:offered|made an offer of)\s+\$?([\d,]+)',
                r"buyer'?s?\s+offer is \$?([\d,]+)",
                r"buyer'?s?\s+(?:initial |first )?offer (?:of |at )?\$?([\d,]+)",
                r'buyer (?:has )?offered \$?([\d,]+)',
                r"buyer'?s?\s+(?:anchor price|opening offer|initial bid) (?:is|was|of|at)\s+\$?([\d,]+)",
                r'Offer \$?([\d,]+)\s+seen',      # "Turn 2 [BUYER]: Offer $X seen"
                r'anchor price (?:is|was|at)\s+\$?([\d,]+)',  # "buyer's anchor price is $X"
            ]:
                m = re.search(pat, text, re.I)
                if m:
                    add("1", m.group(1))
                    break

        # "revised/adjusted/reduced/increased/maintained their offer to/at $X"
        # "latest/current offer is $X"  /  "offer has decreased/increased to $X"
        # → the price is the CURRENT buyer offer at buyer_cur_turn
        cur_pat = re.search(
            r'(?:'
            r'(?:revised|adjusted|reduced|increased|maintained|settled|raised|dropped|held|now offered|finally offered) '
            r'(?:their )?(?:offer )?(?:to|at)\s+\$?([\d,]+)'
            r'|(?:has now offered|has finally offered)\s+\$?([\d,]+)'
            r'|(?:latest|current) offer is \$?([\d,]+)'
            r'|offer has (?:decreased|increased|dropped|risen) to \$?([\d,]+)'
            r'|further (?:reduced|increased) (?:their )?offer to \$?([\d,]+)'
            r'|and now \$?([\d,]+)'
            r')',
            text, re.I
        )
        if cur_pat:
            price = next(g for g in cur_pat.groups() if g)
            add(str(buyer_cur_turn), price)

    return found


# "Buyer offers $X in Turn N, following $A, $B[, $C] offers."
# Prices are chronological: T1=$A, T3=$B, T5=$C, ..., TN=$X
_PAT_FOLLOWING_LIST = re.compile(
    r'^Buyer offers \$?([\d,]+) in Turn (\d+), following ([\$\d,\s]+?)(?:\s*offers?)\.',
    re.IGNORECASE
)


def _canonicalise(reasoning: str) -> str | None:
    """
    Rewrite reasoning into canonical "Turn N: Buyer offers $X. ..." form.
    Returns rewritten string, or None if extraction fails.
    """
    reasoning = reasoning.strip()
    if CANONICAL.match(reasoning):
        return None

    # Special case: "Buyer offers $X in Turn N, following $A, $B, $C offers."
    m = _PAT_FOLLOWING_LIST.match(reasoning)
    if m:
        cur_price  = m.group(1).replace(",", "")
        cur_turn   = int(m.group(2))
        prev_prices = [p.replace(",","") for p in re.findall(r'\$?([\d,]+)', m.group(3))]
        offers: dict[int, str] = {}
        for i, p in enumerate(prev_prices):
            t = 1 + i * 2
            if t < cur_turn:
                offers[t] = p
        offers[cur_turn] = cur_price
        if 1 in offers:
            split_at = _body_start(reasoning)
            body = reasoning[split_at:].strip()
            prefix = " ".join(f"Turn {t}: Buyer offers ${p}." for t, p in sorted(offers.items()))
            return f"{prefix} {body}".strip() if body else prefix

    # Extract seller turn from "Turn N:" at start (handles "Turn N [TAG]:" too)
    seller_turn = None
    m = re.match(r'^(?:\d+\.\s+)?Turn (\d+)(?:\s+\[.*?\])?:', reasoning)
    if m:
        seller_turn = int(m.group(1))

    split_at = _body_start(reasoning)
    opening  = reasoning[:split_at]
    body     = reasoning[split_at:].strip()

    # Extract from opening; if T1 missing, also try full text
    offers = _extract_offers(opening, seller_turn)
    if not offers or 1 not in offers:
        offers = _extract_offers(reasoning, seller_turn)

    # Trust only odd-numbered buyer turns (1, 3, 5, ...)
    offers = {t: p for t, p in offers.items() if t % 2 == 1}

    # If T1 still missing but other turns found, infer it from anchor price
    # (anchor price = buyer's first offer in this negotiation domain)
    if offers and 1 not in offers:
        m = re.search(r'[Aa]nchor price (?:is|was|:)\s*\$?([\d,]+)', reasoning)
        if m:
            offers[1] = m.group(1).replace(",", "")

    if not offers or 1 not in offers:
        return None

    prefix = " ".join(f"Turn {t}: Buyer offers ${p}." for t, p in sorted(offers.items()))
    return f"{prefix} {body}".strip() if body else prefix


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true",
                        help="Report what would change without writing")
    parser.add_argument("--backup", action="store_true",
                        help="Write .json.bak before modifying each file")
    args = parser.parse_args()

    files         = sorted(INTERM_DIR.glob("dialogue_*.json"))
    total_turns   = 0
    non_canonical = 0
    converted     = 0
    unmatched     = 0
    changed_files = 0
    unmatched_ex  = []

    for fpath in files:
        data    = json.loads(fpath.read_text())
        changed = False

        for t in data["processed"]:
            if t["speaker"] != "seller":
                continue
            r = t.get("reasoning", "").strip()
            if not r:
                continue
            total_turns += 1

            if CANONICAL.match(r):
                continue
            non_canonical += 1

            new_r = _canonicalise(r)

            if new_r is None:
                unmatched += 1
                if len(unmatched_ex) < 12:
                    unmatched_ex.append((fpath.stem, t.get("turn_index"), r[:120]))
                continue

            converted += 1
            if args.dry_run and converted <= 6:
                print(f"  [{fpath.stem} T{t.get('turn_index')}]")
                print(f"    BEFORE: {r[:110]!r}")
                print(f"    AFTER : {new_r[:110]!r}")
                print()

            if not args.dry_run:
                t["reasoning"] = new_r
                changed = True

        if changed and not args.dry_run:
            if args.backup:
                shutil.copy2(fpath, fpath.with_suffix(".json.bak"))
            fpath.write_text(json.dumps(data, indent=2, ensure_ascii=False))
            changed_files += 1

    print(f"\n{'='*62}")
    print(f"  Reasoning Harmonisation  {'(DRY RUN)' if args.dry_run else ''}")
    print(f"{'='*62}")
    print(f"  Total seller turns       : {total_turns:,}")
    print(f"  Already canonical        : {total_turns - non_canonical:,}  ({(total_turns-non_canonical)/total_turns*100:.1f}%)")
    print(f"  Non-canonical detected   : {non_canonical:,}  ({non_canonical/total_turns*100:.1f}%)")
    print(f"  Successfully converted   : {converted:,}")
    print(f"  Could not convert        : {unmatched:,}  ({unmatched/total_turns*100:.1f}%)")
    if not args.dry_run:
        print(f"  Files modified           : {changed_files:,}")
    print(f"{'='*62}")

    if unmatched_ex:
        print(f"\n  Could not convert ({unmatched} total, first {len(unmatched_ex)} shown):")
        for did, ti, snippet in unmatched_ex:
            print(f"  {did} T{ti}: {snippet!r}")

    if args.dry_run:
        print(f"\n  Run without --dry_run to apply changes.")
        if unmatched == 0:
            print(f"  All {non_canonical:,} non-canonical turns can be converted cleanly.")
    elif converted > 0:
        print(f"\n  Done. {converted:,} turns normalised across {changed_files:,} files.")
        print(f"  Validate: python3.10 Qwen3-tts/v2/validate_reasoning.py")
        print(f"  Then run SFT run 3.")
    print()


if __name__ == "__main__":
    main()
