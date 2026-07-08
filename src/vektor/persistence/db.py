from __future__ import annotations
"""
vektor.persistence.db
---------------------
SQLite connection management and all query functions for metadata.db.
"""


import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from vektor.persistence.schema import ALL_TABLES


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_connection(db_path: Path) -> sqlite3.Connection:
    """
    Open a SQLite connection safe for cross-thread access.

    check_same_thread=False: allows threads other than the creating thread
    to use this connection. Thread safety is handled by CollectionLock —
    all SQLite calls are serialised through the collection lock, so sharing
    one connection is safe.
    """
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def initialize_db(db_path: Path) -> sqlite3.Connection:
    """
    Create metadata.db and all tables if they don't exist.

    Safe to call on an existing database — uses CREATE TABLE IF NOT EXISTS.

    Args:
        db_path: Path to the metadata.db file.

    Returns:
        Configured sqlite3.Connection.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = get_connection(db_path)
    with conn:
        for table_sql in ALL_TABLES:
            conn.execute(table_sql)
    return conn


# ---------------------------------------------------------------------------
# Collections
# ---------------------------------------------------------------------------

def insert_collection(conn: sqlite3.Connection, name: str, dimension: int,
                      metric: str, m: int, ef_construction: int,
                      vektor_version: str) -> None:
    with conn:
        conn.execute(
            """INSERT INTO collections
               (name, dimension, metric, m, ef_construction, created_at, vektor_version)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (name, dimension, metric, m, ef_construction, utcnow(), vektor_version),
        )


def get_collection(conn: sqlite3.Connection, name: str) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM collections WHERE name = ?", (name,)
    ).fetchone()


def delete_collection(conn: sqlite3.Connection, name: str) -> None:
    with conn:
        conn.execute("DELETE FROM vectors WHERE collection = ?", (name,))
        conn.execute("DELETE FROM collections WHERE name = ?", (name,))


# ---------------------------------------------------------------------------
# Vectors
# ---------------------------------------------------------------------------

def insert_vector_record(conn: sqlite3.Connection, id: str,
                         collection: str, slot_id: int) -> None:
    with conn:
        conn.execute(
            """INSERT INTO vectors (id, collection, slot_id, deleted, created_at)
               VALUES (?, ?, ?, 0, ?)""",
            (id, collection, slot_id, utcnow()),
        )


def tombstone_vector(conn: sqlite3.Connection, id: str, collection: str) -> None:
    with conn:
        conn.execute(
            "UPDATE vectors SET deleted = 1 WHERE id = ? AND collection = ?",
            (id, collection),
        )


def get_vector_record(conn: sqlite3.Connection, id: str,
                      collection: str) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM vectors WHERE id = ? AND collection = ? AND deleted = 0",
        (id, collection),
    ).fetchone()


def get_all_live_vector_records(conn: sqlite3.Connection,
                                collection: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM vectors WHERE collection = ? AND deleted = 0",
        (collection,),
    ).fetchall()


def get_next_slot_id(conn: sqlite3.Connection, collection: str) -> int:
    row = conn.execute(
        "SELECT MAX(slot_id) FROM vectors WHERE collection = ?", (collection,)
    ).fetchone()
    return (row[0] or -1) + 1


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

def start_run(conn: sqlite3.Connection, collection: Optional[str],
              operation: str, **kwargs) -> int:
    take = _next_take_number(conn)
    with conn:
        cursor = conn.execute(
            """INSERT INTO runs
               (take_number, collection, operation, status, m, ef_construction,
                ef, dataset_name, notes, started_at)
               VALUES (?, ?, ?, 'started', ?, ?, ?, ?, ?, ?)""",
            (take, collection, operation,
             kwargs.get("m"), kwargs.get("ef_construction"),
             kwargs.get("ef"), kwargs.get("dataset_name"),
             kwargs.get("notes"), utcnow()),
        )
    return cursor.lastrowid


def end_run(conn: sqlite3.Connection, run_id: int,
            status: str = "completed", notes: Optional[str] = None) -> None:
    with conn:
        conn.execute(
            "UPDATE runs SET status = ?, ended_at = ?, notes = COALESCE(?, notes) WHERE run_id = ?",
            (status, utcnow(), notes, run_id),
        )


def _next_take_number(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT MAX(take_number) FROM runs").fetchone()
    return (row[0] or 0) + 1


# ---------------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------------

def write_log(conn: sqlite3.Connection, message: str,
              level: str = "INFO", run_id: Optional[int] = None) -> None:
    with conn:
        conn.execute(
            "INSERT INTO logs (run_id, timestamp, level, message) VALUES (?, ?, ?, ?)",
            (run_id, utcnow(), level, message),
        )


# ---------------------------------------------------------------------------
# Fixes
# ---------------------------------------------------------------------------

def record_fix(conn: sqlite3.Connection, run_id: Optional[int],
               problem: str, fix_applied: str) -> None:
    with conn:
        conn.execute(
            "INSERT INTO fixes (run_id, problem, fix_applied, fixed_at) VALUES (?, ?, ?, ?)",
            (run_id, problem, fix_applied, utcnow()),
        )