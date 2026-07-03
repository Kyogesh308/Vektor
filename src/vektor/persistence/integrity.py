from __future__ import annotations
"""
vektor.persistence.integrity
-----------------------------
Startup integrity check for a Vektor collection directory.

Run once per collection-open, not once globally at startup.
"""


import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from vektor.persistence.binary import (
    read_vector_bin_header,
    read_all_offsets,
)
from vektor.persistence.manifest import read_manifest, validate_manifest_against_db
from vektor.persistence.db import get_collection, write_log, tombstone_vector


@dataclass
class IntegrityReport:
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def healthy(self) -> bool:
        return len(self.errors) == 0


class IntegrityError(Exception):
    """Raised when a collection has an unrecoverable inconsistency."""


def check_collection_integrity(
    collection_dir: Path,
    conn: sqlite3.Connection,
    collection_name: str,
    run_id: int = None,
) -> IntegrityReport:
    """
    Run all three integrity checks for a collection directory.

    Check 1: collection.json matches collections table in SQLite.
    Check 2: vector.bin slot count matches offsets.bin record count.
    Check 3: every live vector in SQLite has a readable slot in vector.bin.

    Args:
        collection_dir: Directory containing the five collection files.
        conn:           Open SQLite connection to metadata.db.
        collection_name: Name of the collection to check.
        run_id:         Optional run ID for logging.

    Returns:
        IntegrityReport with any warnings or errors found.

    Raises:
        IntegrityError: On unrecoverable inconsistency (manifest/DB mismatch).
    """
    report = IntegrityReport()

    # Check 1 — manifest vs SQLite
    manifest_path = collection_dir / "collection.json"
    db_row = get_collection(conn, collection_name)

    try:
        manifest = read_manifest(manifest_path)
        validate_manifest_against_db(manifest, db_row)
    except Exception as e:
        report.errors.append(f"Manifest inconsistency: {e}")
        write_log(conn, f"[INTEGRITY] Manifest error: {e}", "ERROR", run_id)
        raise IntegrityError(str(e)) from e

    # Check 2 — vector.bin slot count vs offsets.bin record count
    vector_path = collection_dir / "vector.bin"
    offsets_path = collection_dir / "offsets.bin"

    if vector_path.exists() and offsets_path.exists():
        header = read_vector_bin_header(vector_path)
        offsets = read_all_offsets(offsets_path)
        if header["slot_count"] != len(offsets):
            msg = (f"Slot count mismatch: vector.bin has {header['slot_count']} slots, "
                   f"offsets.bin has {len(offsets)} records.")
            report.warnings.append(msg)
            write_log(conn, f"[INTEGRITY] {msg}", "WARNING", run_id)

    # Check 3 — every live SQLite record has a readable vector slot
    from vektor.persistence.db import get_all_live_vector_records
    live_records = get_all_live_vector_records(conn, collection_name)
    dimension = db_row["dimension"]

    for record in live_records:
        slot_id = record["slot_id"]
        try:
            from vektor.persistence.binary import read_vector
            read_vector(vector_path, slot_id, dimension)
        except Exception as e:
            msg = f"Vector ID '{record['id']}' slot {slot_id} unreadable: {e}"
            report.warnings.append(msg)
            write_log(conn, f"[INTEGRITY] {msg} — tombstoning.", "WARNING", run_id)
            tombstone_vector(conn, record["id"], collection_name)

    return report