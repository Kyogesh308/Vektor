"""
vektor.benchmark.results
------------------------
Writes benchmark results to:
  1. benchmarks/results/<experiment>.csv
  2. The runs table in metadata.db (if a connection is provided)
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import sqlite3

import vektor

RESULTS_DIR = Path(__file__).parent.parent.parent.parent / "benchmarks" / "results"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_benchmark_result(
    experiment_name: str,
    result: dict,
    conn: Optional[sqlite3.Connection] = None,
) -> Path:
    """
    Append a benchmark result row to a CSV file and optionally to metadata.db.

    Args:
        experiment_name: Used as the CSV filename (no extension).
        result:          Dict containing all benchmark fields.
        conn:            Optional SQLite connection for runs table logging.

    Returns:
        Path to the CSV file written.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / f"{experiment_name}.csv"

    # Inject metadata
    result.setdefault("timestamp", _utcnow())
    result.setdefault("vektor_version", vektor.__version__)

    # CSV write
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)

    # SQLite runs table
    if conn is not None:
        from vektor.persistence.db import start_run, end_run
        run_id = start_run(
            conn,
            collection=result.get("dataset_name"),
            operation="benchmark",
            m=result.get("m"),
            ef_construction=result.get("ef_construction"),
            ef=result.get("ef"),
            dataset_name=result.get("dataset_name"),
            notes=str(result),
        )
        end_run(conn, run_id, status="completed")

    return csv_path