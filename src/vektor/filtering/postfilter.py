"""
vektor.filtering.postfilter
----------------------------
Post-filter search: HNSW runs unrestricted, results filtered afterward.

Best for filter selectivity < 20% of dataset (fewer graph navigation
penalties), but may return fewer than k results when filter is very
selective. Always surfaces a warning when this happens.

Default overfetch_factor=3: fetches k×3 candidates before filtering.
Minimum allowed: 2. Setting to 1 almost guarantees under-k results
for any selective filter.
"""

from __future__ import annotations

import sqlite3
from typing import Optional

import numpy as np

from vektor.filtering.parser import parse_filter
from vektor.filtering.tombstone import get_tombstone_slot_ids
from vektor.hnsw.algorithms import knn_search, NodeID, Graph
from vektor.hnsw.exceptions import EmptyIndexError, InvalidEFError
from vektor.persistence.db import write_log


MIN_OVERFETCH_FACTOR = 2


class SearchWarning:
    """A warning attached to a search result when results are incomplete."""

    def __init__(self, message: str) -> None:
        self.message = message

    def __repr__(self) -> str:
        return f"SearchWarning({self.message!r})"


class FilteredSearchResult:
    """
    Container for post-filter search results.

    Attributes:
        results:  List of (distance, slot_id) tuples.
        warnings: List of SearchWarning objects. Empty if results are complete.
    """
    __slots__ = ("results", "warnings")

    def __init__(
        self,
        results: list[tuple[float, NodeID]],
        warnings: list[SearchWarning],
    ) -> None:
        self.results = results
        self.warnings = warnings

    @property
    def is_complete(self) -> bool:
        return len(self.warnings) == 0


def search_postfilter(
    query_vector: np.ndarray,
    k: int,
    ef: int,
    filter_dict: dict,
    conn: sqlite3.Connection,
    collection_name: str,
    entry_point: Optional[NodeID],
    max_layer: int,
    graph: Graph,
    vectors: dict[NodeID, np.ndarray],
    dist_fn,
    overfetch_factor: int = 3,
    run_id: Optional[int] = None,
) -> FilteredSearchResult:
    """
    Post-filter nearest-neighbour search.

    Args:
        query_vector:      Float32 query vector.
        k:                 Desired result count.
        ef:                HNSW beam width.
        filter_dict:       User filter specification.
        conn:              Open SQLite connection.
        collection_name:   Collection to search.
        entry_point:       HNSW global entry point.
        max_layer:         HNSW max graph layer.
        graph:             In-memory HNSW graph.
        vectors:           Slot ID → vector mapping.
        dist_fn:           Distance function (lower = more similar).
        overfetch_factor:  Multiplier for candidate pool size. Default 3.
        run_id:            Optional run ID for log attribution.

    Returns:
        FilteredSearchResult with results and any warnings.
        Warnings are non-empty when fewer than k results satisfy the filter.

    Raises:
        EmptyIndexError:  Index has no live vectors.
        InvalidEFError:   ef < k.
        ValueError:       overfetch_factor < MIN_OVERFETCH_FACTOR.
    """
    if entry_point is None:
        raise EmptyIndexError("Cannot search an empty index.")
    if ef < k:
        raise InvalidEFError(f"ef ({ef}) must be >= k ({k}).")
    if overfetch_factor < MIN_OVERFETCH_FACTOR:
        raise ValueError(
            f"overfetch_factor must be >= {MIN_OVERFETCH_FACTOR}. "
            f"Got {overfetch_factor}. Setting to 1 guarantees incomplete "
            f"results for selective filters."
        )

    # Fetch k × overfetch_factor candidates, skipping only tombstones
    fetch_k = k * overfetch_factor
    tombstone_ids = get_tombstone_slot_ids(conn, collection_name)

    candidates = knn_search(
        query_vector=query_vector,
        k=fetch_k,
        ef=max(ef, fetch_k),  # ef must cover the larger fetch size
        entry_point=entry_point,
        max_layer=max_layer,
        graph=graph,
        vectors=vectors,
        dist_fn=dist_fn,
        skip_from_results=tombstone_ids,
        skip_entirely=frozenset(),
    )

    # Apply filter to candidates
    where_clause, params = parse_filter(filter_dict)
    candidate_slot_ids = [slot_id for _, slot_id in candidates]

    if where_clause and candidate_slot_ids:
        placeholders = ",".join("?" * len(candidate_slot_ids))
        sql = (
            f"SELECT slot_id FROM vectors "
            f"WHERE collection = ? AND deleted = 0 "
            f"AND slot_id IN ({placeholders}) "
            f"AND {where_clause}"
        )
        rows = conn.execute(sql, [collection_name] + candidate_slot_ids + params).fetchall()
        passing_ids = frozenset(row[0] for row in rows)
    elif candidate_slot_ids:
        # No filter — all live candidates pass
        placeholders = ",".join("?" * len(candidate_slot_ids))
        sql = (
            f"SELECT slot_id FROM vectors "
            f"WHERE collection = ? AND deleted = 0 "
            f"AND slot_id IN ({placeholders})"
        )
        rows = conn.execute(sql, [collection_name] + candidate_slot_ids).fetchall()
        passing_ids = frozenset(row[0] for row in rows)
    else:
        passing_ids = frozenset()

    # Filter candidates while preserving distance order
    filtered = [(d, sid) for d, sid in candidates if sid in passing_ids]
    final_results = filtered[:k]

    # Build warnings if result set is incomplete
    warnings = []
    if len(final_results) < k:
        msg = (
            f"Requested {k} results but only {len(final_results)} satisfied "
            f"the filter after fetching {fetch_k} candidates "
            f"(overfetch_factor={overfetch_factor}). "
            f"Consider increasing overfetch_factor or using pre-filter search."
        )
        warnings.append(SearchWarning(msg))
        write_log(conn, f"[POST-FILTER] {msg}", level="WARNING", run_id=run_id)

    return FilteredSearchResult(results=final_results, warnings=warnings)