"""
vektor.filtering.prefilter
---------------------------
Pre-filter search: eligible IDs determined before HNSW search begins.

Best for filter selectivity > 20% of dataset.
Below 20%, graph navigation degrades as eligible subgraph becomes sparse.

Strategy:
  1. Parse filter → get eligible slot IDs from SQLite
  2. Get tombstone slot IDs
  3. Build skip_entirely = all_slot_ids - eligible_ids + tombstone_ids
  4. Run knn_search with skip_entirely
  5. Return results (all guaranteed to satisfy filter)
"""

from __future__ import annotations

import sqlite3
from typing import Optional

import numpy as np

from vektor.filtering.parser import get_eligible_slot_ids
from vektor.filtering.tombstone import get_tombstone_slot_ids
from vektor.hnsw.algorithms import knn_search, NodeID, Graph
from vektor.hnsw.exceptions import EmptyIndexError, InvalidEFError


def search_prefilter(
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
) -> list[tuple[float, NodeID]]:
    """
    Pre-filter nearest-neighbour search.

    Args:
        query_vector:    Float32 query vector.
        k:               Desired result count.
        ef:              HNSW beam width (must be >= k).
        filter_dict:     User filter specification.
        conn:            Open SQLite connection.
        collection_name: Collection to search.
        entry_point:     HNSW global entry point.
        max_layer:       HNSW max graph layer.
        graph:           In-memory HNSW graph.
        vectors:         Slot ID → vector mapping.
        dist_fn:         Distance function (lower = more similar).

    Returns:
        List of (distance, slot_id) tuples, all satisfying the filter.
        Length <= k.

    Raises:
        EmptyIndexError: Index has no live vectors.
        InvalidEFError:  ef < k.
    """
    if entry_point is None:
        raise EmptyIndexError("Cannot search an empty index.")
    if ef < k:
        raise InvalidEFError(f"ef ({ef}) must be >= k ({k}).")

    # Build eligible set and tombstone set
    eligible_ids = get_eligible_slot_ids(conn, collection_name, filter_dict)
    tombstone_ids = get_tombstone_slot_ids(conn, collection_name)

    # All slot IDs in the graph
    all_slot_ids = frozenset(vectors.keys())

    # skip_entirely = nodes that don't satisfy the filter
    # (ineligible by filter OR tombstoned)
    # We keep non-tombstoned eligible nodes for traversal AND results.
    # We keep tombstoned nodes as traversal waypoints (skip_from_results only)
    # when they ARE eligible — but if tombstoned, they should not appear as results.
    skip_from_results = tombstone_ids
    skip_entirely = all_slot_ids - eligible_ids  # ineligible by filter

    return knn_search(
        query_vector=query_vector,
        k=k,
        ef=ef,
        entry_point=entry_point,
        max_layer=max_layer,
        graph=graph,
        vectors=vectors,
        dist_fn=dist_fn,
        skip_from_results=skip_from_results,
        skip_entirely=skip_entirely,
    )