from __future__ import annotations
"""
vektor.persistence.schema
-------------------------
SQLite table definitions for metadata.db.

All schema strings live here — never inline in application code.
Adding a new table means adding a string here and calling it in db.py.
"""


COLLECTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS collections (
    name              TEXT PRIMARY KEY,
    dimension         INTEGER NOT NULL,
    metric            TEXT NOT NULL,
    m                 INTEGER NOT NULL DEFAULT 16,
    ef_construction   INTEGER NOT NULL DEFAULT 200,
    created_at        TEXT NOT NULL,
    vektor_version    TEXT NOT NULL
);
"""

VECTORS_TABLE = """
CREATE TABLE IF NOT EXISTS vectors (
    id          TEXT NOT NULL,
    collection  TEXT NOT NULL,
    slot_id     INTEGER NOT NULL,
    deleted     INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL,
    metadata    TEXT,          -- JSON string, nullable
    PRIMARY KEY (id, collection),
    FOREIGN KEY (collection) REFERENCES collections(name)
);
"""

RUNS_TABLE = """
CREATE TABLE IF NOT EXISTS runs (
    run_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    take_number     INTEGER NOT NULL,
    collection      TEXT,
    operation       TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'started',
    m               INTEGER,
    ef_construction INTEGER,
    ef              INTEGER,
    dataset_name    TEXT,
    notes           TEXT,
    started_at      TEXT NOT NULL,
    ended_at        TEXT
);
"""

LOGS_TABLE = """
CREATE TABLE IF NOT EXISTS logs (
    log_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER,
    timestamp   TEXT NOT NULL,
    level       TEXT NOT NULL,
    message     TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
"""

FIXES_TABLE = """
CREATE TABLE IF NOT EXISTS fixes (
    fix_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER,
    problem     TEXT NOT NULL,
    fix_applied TEXT NOT NULL,
    fixed_at    TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
"""

ALL_TABLES = [
    COLLECTIONS_TABLE,
    VECTORS_TABLE,
    RUNS_TABLE,
    LOGS_TABLE,
    FIXES_TABLE,
]