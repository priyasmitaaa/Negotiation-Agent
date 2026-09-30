SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS annotators (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rater_code TEXT NOT NULL UNIQUE,
    guide_acknowledged_at TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_active_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    submitted_at TEXT,
    submission_status TEXT NOT NULL DEFAULT 'in_progress'
);

CREATE TABLE IF NOT EXISTS items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_code TEXT NOT NULL UNIQUE,
    display_order INTEGER NOT NULL UNIQUE,
    is_flip_item INTEGER NOT NULL CHECK (is_flip_item IN (0, 1)),
    source_directory TEXT NOT NULL,
    product_data TEXT NOT NULL,
    pre_reveal_data TEXT NOT NULL,
    post_reveal_data TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS annotations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    annotator_id INTEGER NOT NULL REFERENCES annotators(id) ON DELETE CASCADE,
    item_id INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    d0_strategy TEXT CHECK (d0_strategy IN ('Open the negotiation','Negotiate firmly','Protect the buyer')),
    d0_confidence INTEGER CHECK (d0_confidence IN (1, 2, 3)),
    d0_locked_at TEXT,
    revealed_at TEXT,
    d1 INTEGER CHECK (d1 BETWEEN 1 AND 5),
    d2 INTEGER CHECK (d2 BETWEEN 1 AND 5),
    d3 INTEGER CHECK (d3 BETWEEN 1 AND 5),
    d4 INTEGER CHECK (d4 BETWEEN 1 AND 5),
    d5 INTEGER CHECK (d5 BETWEEN 1 AND 5),
    d6 TEXT,
    d7 INTEGER CHECK (d7 BETWEEN 1 AND 5),
    d8 INTEGER CHECK (d8 BETWEEN 1 AND 5),
    d9 INTEGER CHECK (d9 BETWEEN 1 AND 5),
    d10 INTEGER CHECK (d10 BETWEEN 1 AND 5),
    d11 INTEGER CHECK (d11 BETWEEN 1 AND 5),
    d12 INTEGER CHECK (d12 BETWEEN 1 AND 5),
    d13 TEXT CHECK (d13 IN ('yes', 'no')),
    flag_wrong_price_product TEXT CHECK (flag_wrong_price_product IN ('yes', 'no')),
    flag_dishonest_manipulative TEXT CHECK (flag_dishonest_manipulative IN ('yes', 'no')),
    flag_unsafe_offensive TEXT CHECK (flag_unsafe_offensive IN ('yes', 'no')),
    flag_audio_unusable TEXT CHECK (flag_audio_unusable IN ('yes', 'no')),
    comment TEXT,
    status TEXT NOT NULL DEFAULT 'untouched' CHECK (status IN ('untouched','partial','complete','technical_issue')),
    technical_issue_description TEXT,
    technical_issue_at TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TEXT,
    UNIQUE (annotator_id, item_id)
);
"""
