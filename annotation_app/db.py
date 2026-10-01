import sqlite3
from flask import current_app, g

from .schema import SCHEMA


def get_db():
    if "db" not in g:
        db = sqlite3.connect(current_app.config["DATABASE_PATH"])
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        g.db = db
    return g.db


def close_db(_exc=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    db = get_db()
    db.executescript(SCHEMA)
    migrate_db(db)
    db.commit()


def migrate_db(db):
    columns = {row["name"] for row in db.execute("PRAGMA table_info(annotators)").fetchall()}
    if "password_hash" not in columns:
        db.execute("ALTER TABLE annotators ADD COLUMN password_hash TEXT")
    if "password_set_at" not in columns:
        db.execute("ALTER TABLE annotators ADD COLUMN password_set_at TEXT")


def row_to_dict(row):
    return dict(row) if row is not None else None
