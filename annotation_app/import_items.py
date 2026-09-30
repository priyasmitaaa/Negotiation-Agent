import json
import re
from pathlib import Path

from .db import get_db, init_db

FLIP_ITEMS = {
    "item_003", "item_005", "item_010", "item_011", "item_013", "item_027",
    "item_034", "item_043", "item_047", "item_054", "item_059", "item_060",
    "item_062", "item_068", "item_069", "item_076", "item_090", "item_097",
}


def parse_context_card(card):
    patterns = {
        "product": r"Product: (.*?)\. Condition:",
        "condition": r"Condition: (.*?)\. Asking price:",
        "asking_price": r"Asking price: \$(.*?)\. Fair value:",
        "fair_value": r"Fair value: \$(.*?)\. Goal:",
        "goal": r"Goal: (.*)$",
    }
    parsed = {"raw": card}
    for key, pattern in patterns.items():
        match = re.search(pattern, card)
        parsed[key] = match.group(1) if match else ""
    return parsed


def import_all(dataset_root):
    init_db()
    root = Path(dataset_root)
    db = get_db()
    item_paths = sorted(root.glob("item_*/item.json"))
    if len(item_paths) != 100:
        raise RuntimeError(f"Expected 100 item JSON files, found {len(item_paths)}")

    for order, path in enumerate(item_paths, start=1):
        item = json.loads(path.read_text(encoding="utf-8"))
        code = item["item_id"]
        if code != path.parent.name:
            raise RuntimeError(f"Item ID mismatch: {path}")
        product_data = parse_context_card(item["context_card"])
        pre_reveal = {
            "item_id": code,
            "system": item.get("system"),
            "context_card": item["context_card"],
            "conversation_so_far": item["conversation_so_far"],
            "latest_buyer_offer": extract_latest_buyer_offer(item["conversation_so_far"]),
        }
        post_reveal = {"reply": item["reply"]}
        db.execute(
            """
            INSERT INTO items
            (item_code, display_order, is_flip_item, source_directory, product_data, pre_reveal_data, post_reveal_data)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(item_code) DO UPDATE SET
                display_order=excluded.display_order,
                is_flip_item=excluded.is_flip_item,
                source_directory=excluded.source_directory,
                product_data=excluded.product_data,
                pre_reveal_data=excluded.pre_reveal_data,
                post_reveal_data=excluded.post_reveal_data
            """,
            (
                code,
                order,
                1 if code in FLIP_ITEMS else 0,
                path.parent.name,
                json.dumps(product_data),
                json.dumps(pre_reveal),
                json.dumps(post_reveal),
            ),
        )
    db.commit()
    return len(item_paths)


def extract_latest_buyer_offer(turns):
    buyer_turns = [turn for turn in turns if turn.get("speaker") == "buyer"]
    if not buyer_turns:
        return ""
    amounts = re.findall(r"\$\s*([0-9][0-9,]*)", buyer_turns[-1].get("transcript", ""))
    return amounts[-1] if amounts else ""
