import os
from pathlib import Path

from flask import Flask

from .routes import bp


def create_app(test_config=None):
    root = Path(__file__).resolve().parent.parent
    app = Flask(__name__, instance_relative_config=True)
    app.config.update(
        DATABASE_PATH=os.environ.get("DATABASE_PATH", str(root / "instance" / "annotations.sqlite3")),
        DATASET_ROOT=str(root),
        ADMIN_EXPORT_SECRET=os.environ.get("ADMIN_EXPORT_SECRET", ""),
        SECRET_KEY=os.environ.get("SECRET_KEY", "dev-only-change-me"),
    )
    if test_config:
        app.config.update(test_config)

    Path(app.config["DATABASE_PATH"]).parent.mkdir(parents=True, exist_ok=True)
    app.register_blueprint(bp)

    return app
