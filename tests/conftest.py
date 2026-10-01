import tempfile
from pathlib import Path

import pytest

from annotation_app import create_app
from annotation_app.import_items import import_all


@pytest.fixture()
def app():
    root = Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory() as tmp:
        app = create_app(
            {
                "TESTING": True,
                "DATABASE_PATH": str(Path(tmp) / "test.sqlite3"),
                "DATASET_ROOT": str(root),
                "ADMIN_EXPORT_SECRET": "test-secret",
                "SECRET_KEY": "test-secret-key",
            }
        )
        with app.app_context():
            import_all(root)
        yield app


@pytest.fixture()
def client(app):
    return app.test_client()


def start(client, rater="rater-a"):
    response = client.post(
        "/api/signup",
        json={"rater_id": rater, "password": "password123", "guide_acknowledged": True},
    )
    if response.status_code == 400:
        response = client.post(
            "/api/login",
            json={"rater_id": rater, "password": "password123"},
        )
    return response
