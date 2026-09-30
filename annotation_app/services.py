import csv
import io
import json
from datetime import datetime, timezone

from .db import get_db

STRATEGIES = {"Open the negotiation", "Negotiate firmly", "Protect the buyer"}
YES_NO = {"yes", "no"}
RATING_FIELDS = [f"d{i}" for i in range(1, 13)]
FLAG_FIELDS = [
    "flag_wrong_price_product",
    "flag_dishonest_manipulative",
    "flag_unsafe_offensive",
    "flag_audio_unusable",
]
EXPORT_COLUMNS = [
    "Rater ID",
    "Item",
    "D6 applies? (flip item)",
    "D0 Strategy (Step 1, before reveal)",
    "D0 Confidence 1-3",
    "D1 Decision appropriateness",
    "D2 Decision-response consistency",
    "D3 Price strategy",
    "D4 Emotional responsiveness",
    "D5 Negotiation progression",
    "D6 Adaptation to change",
    "D7 Coherence & context",
    "D8 Text naturalness",
    "D9 Speech intelligibility",
    "D10 Speech naturalness & prosody",
    "D11 Speech-text fidelity",
    "D12 Overall quality",
    "D13 Acceptable? (yes/no)",
    "Flag: wrong price/product",
    "Flag: dishonest/manipulative",
    "Flag: unsafe/offensive",
    "Flag: audio unusable",
    "Comment (optional)",
]


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_annotator(rater_code, guide_acknowledged=False):
    code = (rater_code or "").strip()
    if not code:
        raise ValueError("Rater ID is required.")
    db = get_db()
    timestamp = now()
    row = db.execute("SELECT * FROM annotators WHERE rater_code = ?", (code,)).fetchone()
    if row is None:
        db.execute(
            "INSERT INTO annotators (rater_code, guide_acknowledged_at, last_active_at) VALUES (?, ?, ?)",
            (code, timestamp if guide_acknowledged else None, timestamp),
        )
    else:
        db.execute(
            """
            UPDATE annotators
            SET guide_acknowledged_at = COALESCE(guide_acknowledged_at, ?),
                last_active_at = ?
            WHERE rater_code = ?
            """,
            (timestamp if guide_acknowledged else None, timestamp, code),
        )
    db.commit()
    return db.execute("SELECT * FROM annotators WHERE rater_code = ?", (code,)).fetchone()


def annotator_by_id(annotator_id):
    return get_db().execute("SELECT * FROM annotators WHERE id = ?", (annotator_id,)).fetchone()


def item_by_code(item_code):
    return get_db().execute("SELECT * FROM items WHERE item_code = ?", (item_code,)).fetchone()


def get_or_create_annotation(annotator_id, item_id):
    db = get_db()
    db.execute(
        "INSERT OR IGNORE INTO annotations (annotator_id, item_id) VALUES (?, ?)",
        (annotator_id, item_id),
    )
    db.commit()
    return db.execute(
        "SELECT * FROM annotations WHERE annotator_id = ? AND item_id = ?",
        (annotator_id, item_id),
    ).fetchone()


def annotation_for(annotator_id, item_id):
    return get_db().execute(
        "SELECT * FROM annotations WHERE annotator_id = ? AND item_id = ?",
        (annotator_id, item_id),
    ).fetchone()


def progress_list(annotator_id):
    rows = get_db().execute(
        """
        SELECT i.item_code, i.display_order, i.is_flip_item, a.status, a.revealed_at, a.completed_at
        FROM items i
        LEFT JOIN annotations a ON a.item_id = i.id AND a.annotator_id = ?
        ORDER BY i.display_order
        """,
        (annotator_id,),
    ).fetchall()
    return [
        {
            "item": r["item_code"],
            "order": r["display_order"],
            "is_flip_item": bool(r["is_flip_item"]),
            "status": r["status"] or "untouched",
            "revealed": bool(r["revealed_at"]),
            "completed_at": r["completed_at"],
        }
        for r in rows
    ]


def first_unresolved_order(annotator_id):
    for entry in progress_list(annotator_id):
        if entry["status"] not in {"complete", "technical_issue"}:
            return entry["order"]
    return 101


def can_open_item(annotator_id, item):
    first = first_unresolved_order(annotator_id)
    existing = annotation_for(annotator_id, item["id"])
    return item["display_order"] <= first or existing is not None


def serialize_item(item, ann=None, include_post=False):
    payload = {
        "item": item["item_code"],
        "display_order": item["display_order"],
        "is_flip_item": bool(item["is_flip_item"]),
        "product": json.loads(item["product_data"]),
        "pre_reveal": json.loads(item["pre_reveal_data"]),
        "annotation": serialize_annotation(ann),
        "reply_revealed": bool(ann and ann["revealed_at"]),
    }
    if include_post and ann and ann["revealed_at"]:
        payload["post_reveal"] = json.loads(item["post_reveal_data"])
    return payload


def serialize_annotation(ann):
    if not ann:
        return None
    keys = [
        "d0_strategy", "d0_confidence", "d0_locked_at", "revealed_at", "d1", "d2", "d3",
        "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13",
        *FLAG_FIELDS, "comment", "status", "technical_issue_description",
        "technical_issue_at", "completed_at", "updated_at",
    ]
    return {key: ann[key] for key in keys}


def save_d0(annotator_id, item, strategy, confidence):
    if submitted(annotator_id):
        raise PermissionError("This annotation set has been submitted and is read-only.")
    if strategy not in STRATEGIES:
        raise ValueError("Choose a valid D0 strategy.")
    try:
        confidence = int(confidence)
    except (TypeError, ValueError):
        raise ValueError("Choose D0 confidence.")
    if confidence not in {1, 2, 3}:
        raise ValueError("Choose D0 confidence.")
    ann = get_or_create_annotation(annotator_id, item["id"])
    if ann["d0_locked_at"]:
        raise PermissionError("D0 is locked for this item.")
    db = get_db()
    db.execute(
        """
        UPDATE annotations
        SET d0_strategy = ?, d0_confidence = ?, status = 'partial', updated_at = ?
        WHERE id = ?
        """,
        (strategy, confidence, now(), ann["id"]),
    )
    db.commit()
    return annotation_for(annotator_id, item["id"])


def reveal(annotator_id, item):
    if submitted(annotator_id):
        raise PermissionError("This annotation set has been submitted and is read-only.")
    ann = get_or_create_annotation(annotator_id, item["id"])
    if not ann["d0_strategy"] or not ann["d0_confidence"]:
        raise ValueError("D0 Strategy and Confidence are required before reveal.")
    if ann["revealed_at"]:
        return ann
    timestamp = now()
    d6_value = "N/A" if not item["is_flip_item"] else ann["d6"]
    get_db().execute(
        """
        UPDATE annotations
        SET d0_locked_at = COALESCE(d0_locked_at, ?),
            revealed_at = ?,
            d6 = ?,
            status = 'partial',
            updated_at = ?
        WHERE id = ?
        """,
        (timestamp, timestamp, d6_value, timestamp, ann["id"]),
    )
    get_db().commit()
    return annotation_for(annotator_id, item["id"])


def autosave(annotator_id, item, updates):
    ann = get_or_create_annotation(annotator_id, item["id"])
    if not ann["revealed_at"]:
        raise PermissionError("Reveal the seller reply before rating it.")
    if submitted(annotator_id):
        raise PermissionError("This annotation set has been submitted and is read-only.")
    allowed = set(RATING_FIELDS + ["d13"] + FLAG_FIELDS + ["comment"])
    assignments = {}
    for key, value in updates.items():
        if key not in allowed:
            continue
        assignments[key] = validate_field(key, value, bool(item["is_flip_item"]))
    if not item["is_flip_item"]:
        assignments["d6"] = "N/A"
    if not assignments:
        return ann
    fields = ", ".join(f"{key} = ?" for key in assignments)
    values = list(assignments.values())
    status, completed_at = calculate_status({**dict(ann), **assignments}, bool(item["is_flip_item"]))
    values.extend([status, completed_at, now(), ann["id"]])
    get_db().execute(
        f"UPDATE annotations SET {fields}, status = ?, completed_at = ?, updated_at = ? WHERE id = ?",
        values,
    )
    get_db().commit()
    return annotation_for(annotator_id, item["id"])


def validate_field(key, value, is_flip):
    if key in RATING_FIELDS:
        if key == "d6" and not is_flip:
            return "N/A"
        try:
            rating = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"{key.upper()} must be 1-5.")
        if rating < 1 or rating > 5:
            raise ValueError(f"{key.upper()} must be 1-5.")
        return str(rating) if key == "d6" else rating
    if key == "d13" or key in FLAG_FIELDS:
        if value not in YES_NO:
            raise ValueError(f"{key} must be yes or no.")
        return value
    if key == "comment":
        return (value or "").strip()
    return value


def calculate_status(data, is_flip):
    if data.get("status") == "technical_issue":
        return "technical_issue", data.get("completed_at")
    if not data.get("d0_locked_at") or not data.get("revealed_at"):
        return "partial" if data.get("d0_strategy") or data.get("d0_confidence") else "untouched", None
    required = [f"d{i}" for i in range(1, 13)] + ["d13"] + FLAG_FIELDS
    if not is_flip:
        data["d6"] = "N/A"
    complete = all(data.get(key) not in (None, "") for key in required)
    if complete and (is_flip or data.get("d6") == "N/A"):
        return "complete", data.get("completed_at") or now()
    return "partial", None


def report_technical_issue(annotator_id, item, description):
    if submitted(annotator_id):
        raise PermissionError("This annotation set has been submitted and is read-only.")
    desc = (description or "").strip()
    if not desc:
        raise ValueError("Problem description is required.")
    ann = get_or_create_annotation(annotator_id, item["id"])
    timestamp = now()
    get_db().execute(
        """
        UPDATE annotations
        SET status = 'technical_issue',
            technical_issue_description = ?,
            technical_issue_at = ?,
            updated_at = ?,
            completed_at = NULL
        WHERE id = ?
        """,
        (desc, timestamp, timestamp, ann["id"]),
    )
    get_db().commit()
    return annotation_for(annotator_id, item["id"])


def submitted(annotator_id):
    row = annotator_by_id(annotator_id)
    return bool(row and row["submitted_at"])


def submit_all(annotator_id):
    unresolved = [p for p in progress_list(annotator_id) if p["status"] not in {"complete", "technical_issue"}]
    if unresolved:
        raise ValueError(f"{len(unresolved)} items still need completion or technical-issue reporting.")
    timestamp = now()
    get_db().execute(
        "UPDATE annotators SET submitted_at = ?, submission_status = 'submitted', last_active_at = ? WHERE id = ?",
        (timestamp, timestamp, annotator_id),
    )
    get_db().commit()


def csv_export(audit=False):
    db = get_db()
    rows = db.execute(
        """
        SELECT a.*, an.rater_code, an.submission_status, an.submitted_at,
               i.item_code, i.display_order, i.is_flip_item
        FROM annotations a
        JOIN annotators an ON an.id = a.annotator_id
        JOIN items i ON i.id = a.item_id
        ORDER BY an.rater_code, i.display_order
        """
    ).fetchall()
    out = io.StringIO()
    extra = [
        "Internal annotation ID", "Reveal timestamp", "Completion timestamp", "Last update timestamp",
        "Technical-issue status", "Final-submission status", "Submission timestamp",
    ] if audit else []
    writer = csv.DictWriter(out, fieldnames=EXPORT_COLUMNS + extra)
    writer.writeheader()
    for r in rows:
        row = {
            "Rater ID": r["rater_code"],
            "Item": r["item_code"],
            "D6 applies? (flip item)": "yes" if r["is_flip_item"] else "no (enter N/A)",
            "D0 Strategy (Step 1, before reveal)": r["d0_strategy"],
            "D0 Confidence 1-3": r["d0_confidence"],
            "D1 Decision appropriateness": r["d1"],
            "D2 Decision-response consistency": r["d2"],
            "D3 Price strategy": r["d3"],
            "D4 Emotional responsiveness": r["d4"],
            "D5 Negotiation progression": r["d5"],
            "D6 Adaptation to change": r["d6"] or ("N/A" if not r["is_flip_item"] else ""),
            "D7 Coherence & context": r["d7"],
            "D8 Text naturalness": r["d8"],
            "D9 Speech intelligibility": r["d9"],
            "D10 Speech naturalness & prosody": r["d10"],
            "D11 Speech-text fidelity": r["d11"],
            "D12 Overall quality": r["d12"],
            "D13 Acceptable? (yes/no)": r["d13"],
            "Flag: wrong price/product": r["flag_wrong_price_product"],
            "Flag: dishonest/manipulative": r["flag_dishonest_manipulative"],
            "Flag: unsafe/offensive": r["flag_unsafe_offensive"],
            "Flag: audio unusable": r["flag_audio_unusable"],
            "Comment (optional)": r["comment"],
        }
        if audit:
            row.update({
                "Internal annotation ID": r["id"],
                "Reveal timestamp": r["revealed_at"],
                "Completion timestamp": r["completed_at"],
                "Last update timestamp": r["updated_at"],
                "Technical-issue status": r["status"] == "technical_issue",
                "Final-submission status": r["submission_status"],
                "Submission timestamp": r["submitted_at"],
            })
        writer.writerow(row)
    return out.getvalue()
