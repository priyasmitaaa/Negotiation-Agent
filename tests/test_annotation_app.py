import csv
import io

from annotation_app.import_items import FLIP_ITEMS
from annotation_app.services import EXPORT_COLUMNS
from conftest import start


def test_import_all_100_items_and_order(client):
    res = start(client)
    data = res.get_json()
    assert len(data["progress"]) == 100
    assert data["progress"][0]["item"] == "item_001"
    assert data["progress"][-1]["item"] == "item_100"
    assert {p["item"] for p in data["progress"] if p["is_flip_item"]} == FLIP_ITEMS


def test_start_resume_and_separate_annotators(client):
    assert start(client, "alpha").get_json()["annotator"]["rater_id"] == "alpha"
    first = client.get("/api/items/item_001").get_json()
    assert first["annotation"]["status"] == "untouched"
    assert start(client, "beta").get_json()["annotator"]["rater_id"] == "beta"
    second = client.get("/api/items/item_001").get_json()
    assert second["annotation"]["status"] == "untouched"


def test_pre_reveal_excludes_seller_reply_and_asset(client):
    start(client)
    item = client.get("/api/items/item_001").get_json()
    assert "post_reveal" not in item
    assert "reply" not in str(item["pre_reveal"])
    assert client.get("/assets/item_001/reply.wav").status_code == 404


def test_reveal_requires_valid_d0_then_locks(client):
    start(client)
    assert client.post("/api/items/item_001/reveal", json={}).status_code == 400
    assert client.post("/api/items/item_001/d0", json={"d0_strategy": "bad", "d0_confidence": 1}).status_code == 400
    ok = client.post(
        "/api/items/item_001/d0",
        json={"d0_strategy": "Open the negotiation", "d0_confidence": 3},
    )
    assert ok.status_code == 200
    revealed = client.post("/api/items/item_001/reveal", json={})
    assert revealed.status_code == 200
    data = revealed.get_json()
    assert data["annotation"]["d0_locked_at"]
    assert data["post_reveal"]["reply"]["text"]
    assert client.get("/assets/item_001/reply.wav").status_code == 200
    assert client.post(
        "/api/items/item_001/d0",
        json={"d0_strategy": "Negotiate firmly", "d0_confidence": 2},
    ).status_code == 423


def test_rating_validation_d6_and_flags(client):
    start(client)
    reveal_item(client, "item_001")
    bad = client.patch("/api/items/item_001/ratings", json={"d1": 6})
    assert bad.status_code == 400
    ok = client.patch("/api/items/item_001/ratings", json={"d1": 5, "d6": 3})
    assert ok.status_code == 200
    assert ok.get_json()["annotation"]["d6"] == "N/A"

    reveal_item(client, "item_003")
    ok = client.patch("/api/items/item_003/ratings", json={"d6": 3})
    assert ok.get_json()["annotation"]["d6"] == "3"
    assert client.patch("/api/items/item_003/ratings", json={"flag_audio_unusable": "maybe"}).status_code == 400


def test_completion_and_final_submission(client):
    start(client)
    reveal_item(client, "item_001")
    payload = complete_payload(is_flip=False)
    res = client.patch("/api/items/item_001/ratings", json=payload)
    assert res.get_json()["annotation"]["status"] == "complete"
    assert client.post("/api/submit", json={}).status_code == 400


def test_technical_issue_counts_as_resolved_exception(client):
    start(client)
    res = client.post("/api/items/item_001/technical-issue", json={"description": "audio missing in browser"})
    assert res.status_code == 200
    assert res.get_json()["annotation"]["status"] == "technical_issue"


def test_complete_or_issue_allows_final_submission_and_duplicate(client):
    start(client)
    for i in range(1, 101):
        code = f"item_{i:03d}"
        client.post(f"/api/items/{code}/technical-issue", json={"description": "batch test exception"})
    assert client.post("/api/submit", json={}).status_code == 200
    assert client.post("/api/submit", json={}).status_code == 200
    assert client.patch("/api/items/item_001/ratings", json={"d1": 5}).status_code == 423


def test_csv_export_column_order(client):
    start(client)
    reveal_item(client, "item_001")
    client.patch("/api/items/item_001/ratings", json=complete_payload(is_flip=False))
    res = client.get("/admin/export.csv?secret=test-secret")
    assert res.status_code == 200
    reader = csv.reader(io.StringIO(res.get_data(as_text=True)))
    assert next(reader) == EXPORT_COLUMNS


def reveal_item(client, code):
    client.get(f"/api/items/{code}")
    res = client.post(
        f"/api/items/{code}/d0",
        json={"d0_strategy": "Open the negotiation", "d0_confidence": 3},
    )
    assert res.status_code == 200
    res = client.post(f"/api/items/{code}/reveal", json={})
    assert res.status_code == 200


def complete_payload(is_flip):
    payload = {f"d{i}": 5 for i in range(1, 13)}
    if not is_flip:
        payload["d6"] = "N/A"
    payload.update(
        {
            "d13": "yes",
            "flag_wrong_price_product": "no",
            "flag_dishonest_manipulative": "no",
            "flag_unsafe_offensive": "no",
            "flag_audio_unusable": "no",
        }
    )
    return payload
