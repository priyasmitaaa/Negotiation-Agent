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
    db.commit()


def row_to_dict(row):
    return dict(row) if row is not None else None
