"""
vektor.filtering.tombstone
---------------------------
Tombstone set construction and entry point recovery.

Tombstoned nodes must be handled differently from pre-filter-ineligible nodes:
- Tombstoned: exclude from results, but traverse through (they exist in graph)
- Ineligible: exclude from both results and traversal (treat as non-existent)
"""

from __future__ import annotations

import sqlite3
from typing import Optional

from vektor.persistence.db import write_log


def get_tombstone_slot_ids(
    conn: sqlite3.Connection,
    collection_name: str,
) -> frozenset[int]:
    """
    Query SQLite for all tombstoned slot IDs in a collection.

    Called once per search — not once per candidate. The cost is one
    SQL query per search call, amortised across all candidates visited.

    Returns:
        frozenset of integer slot IDs marked deleted=1 in the vectors table.
    """
    rows = conn.execute(
        "SELECT slot_id FROM vectors WHERE collection = ? AND deleted = 1",
        (collection_name,),
    ).fetchall()
    return frozenset(row[0] for row in rows)


def recover_entry_point(
    conn: sqlite3.Connection,
    collection_name: str,
    current_entry_point: int,
    graph: dict,
    run_id: Optional[int] = None,
) -> Optional[int]:
    """
    Check if the current entry point is tombstoned. If so, find a replacement.

    Scans all non-deleted slot IDs and returns the one with the highest
    maximum layer in the graph. If no valid entry point exists (all vectors
    deleted), returns None.

    Args:
        conn:                Open SQLite connection.
        collection_name:     Name of the collection.
        current_entry_point: The slot ID currently stored as entry point.
        graph:               In-memory HNSW adjacency structure.
        run_id:              Optional run ID for log attribution.

    Returns:
        Valid entry point slot ID, or None if the index is now empty.
    """
    tombstones = get_tombstone_slot_ids(conn, collection_name)

    if current_entry_point not in tombstones:
        return current_entry_point  # Still valid — fast path

    write_log(
        conn,
        f"[TOMBSTONE] Entry point slot {current_entry_point} is deleted. "
        f"Scanning for replacement.",
        level="WARNING",
        run_id=run_id,
    )

    # Find the non-deleted node with the highest maximum layer
    live_rows = conn.execute(
        "SELECT slot_id FROM vectors WHERE collection = ? AND deleted = 0",
        (collection_name,),
    ).fetchall()

    if not live_rows:
        write_log(conn,
                  "[TOMBSTONE] No live vectors remain. Index is empty.",
                  level="WARNING", run_id=run_id)
        return None

    best_slot = None
    best_layer = -1

    for row in live_rows:
        slot_id = row[0]
        if slot_id in graph:
            node_max_layer = max(graph[slot_id].keys(), default=-1)
            if node_max_layer > best_layer:
                best_layer = node_max_layer
                best_slot = slot_id

    if best_slot is not None:
        write_log(
            conn,
            f"[TOMBSTONE] Recovered entry point: slot {best_slot} "
            f"at layer {best_layer}.",
            level="WARNING",
            run_id=run_id,
        )

    return best_slot