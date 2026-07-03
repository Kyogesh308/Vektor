from __future__ import annotations
"""
vektor.persistence.logger
-------------------------
High-level logging interface wrapping the runs/logs/fixes tables.
"""


import sqlite3
from typing import Optional

from vektor.persistence.db import (
    start_run, end_run, write_log, record_fix
)


class VektorLogger:
    """
    Thin wrapper around the runs/logs/fixes tables.

    Usage:
        logger = VektorLogger(conn)
        run_id = logger.start("collection_name", "insert")
        logger.info("Inserted vector vec_001", run_id)
        logger.finish(run_id)
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def start(self, collection: Optional[str], operation: str, **kwargs) -> int:
        return start_run(self._conn, collection, operation, **kwargs)

    def finish(self, run_id: int, status: str = "completed",
               notes: Optional[str] = None) -> None:
        end_run(self._conn, run_id, status, notes)

    def debug(self, message: str, run_id: Optional[int] = None) -> None:
        write_log(self._conn, message, "DEBUG", run_id)

    def info(self, message: str, run_id: Optional[int] = None) -> None:
        write_log(self._conn, message, "INFO", run_id)

    def warning(self, message: str, run_id: Optional[int] = None) -> None:
        write_log(self._conn, message, "WARNING", run_id)

    def error(self, message: str, run_id: Optional[int] = None) -> None:
        write_log(self._conn, message, "ERROR", run_id)

    def fix(self, problem: str, fix_applied: str,
            run_id: Optional[int] = None) -> None:
        record_fix(self._conn, run_id, problem, fix_applied)