"""
vektor.hnsw.index
------------------
HNSWIndex: the public interface to the HNSW graph.

Owns the graph structure, the distance function closure, the entry point,
and the threading lock. Delegates all algorithmic work to algorithms.py.
"""

from __future__ import annotations

import math
import random
import threading
from typing import Optional

import numpy as np

from vektor.collection import Collection
from vektor.distance import compute_distance, METRIC_HIGHER_IS_BETTER
from vektor.hnsw.algorithms import (
    Graph, NodeID,
    insert as _insert,
    knn_search as _knn_search,
)
from vektor.hnsw.exceptions import EmptyIndexError, InvalidEFError
from vektor.hnsw.layer import assign_layer


class HNSWIndex:
    """
    In-memory HNSW graph index for a single Vektor collection.

    One HNSWIndex instance per collection. Stores vectors internally
    alongside the graph structure. Coordinates with VectorStore for
    persistence via Phase 5 binary functions.

    Thread safety: an RLock protects insert and search. Phase 12
    (FastAPI) requires this — do not remove.

    Usage:
        index = HNSWIndex(collection)
        index.add(slot_id=0, vector=np.array([...]))
        results = index.search(query_vector, k=10, ef=100)
    """

    def __init__(
        self,
        collection: Collection,
        seed: int = 42,
    ) -> None:
        self._collection = collection
        self._M = collection.m
        self._ef_construction = collection.ef_construction
        self._seed = seed

        # Random state — fixed seed for reproducible benchmarks
        self._rng = random.Random(seed)

        # Graph: slot_id → {layer: [neighbour_slot_ids]}
        self._graph: Graph = {}

        # Vectors: slot_id → float32 ndarray
        self._vectors: dict[NodeID, np.ndarray] = {}

        # Entry point and maximum layer
        self._entry_point: Optional[NodeID] = None
        self._max_layer: int = 0

        # Thread safety
        self._lock = threading.RLock()

        # Distance function closure
        # HNSW internally always works in "lower = more similar" space.
        # For cosine and dot (higher = better), we negate the score.
        metric = collection.metric
        if METRIC_HIGHER_IS_BETTER[metric]:
            def _dist(a: np.ndarray, b: np.ndarray) -> float:
                return -compute_distance(metric, a, b)
        else:
            def _dist(a: np.ndarray, b: np.ndarray) -> float:
                return compute_distance(metric, a, b)
        self._dist_fn = _dist

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------

    def add(self, slot_id: NodeID, vector: np.ndarray) -> None:
        """
        Insert a vector into the HNSW graph.

        Args:
            slot_id: Integer slot ID matching the VectorStore's slot system.
            vector:  Float32 ndarray, pre-validated by Phase 1.

        Note: Phase 1 validation must have already run before calling add().
              This method does NOT re-validate — it trusts the caller.
        """
        new_layer = assign_layer(self._M, self._rng)

        with self._lock:
            # Register vector before insert so SEARCH-LAYER can access it
            self._vectors[slot_id] = vector

            self._entry_point, self._max_layer = _insert(
                new_id=slot_id,
                new_vector=vector,
                graph=self._graph,
                vectors=self._vectors,
                entry_point=self._entry_point,
                max_layer=self._max_layer,
                new_node_layer=new_layer,
                M=self._M,
                ef_construction=self._ef_construction,
                dist_fn=self._dist_fn,
                use_heuristic=True,
            )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: np.ndarray,
        k: int,
        ef: int,
        skip_ids: Optional[set[NodeID]] = None,
    ) -> list[tuple[float, NodeID]]:
        """
        Find the k approximate nearest neighbours of the query vector.

        Args:
            query:    Float32 query vector, pre-validated by Phase 1.
            k:        Number of results to return.
            ef:       Beam width. Must be >= k.
            skip_ids: Optional set of tombstoned slot IDs to exclude.

        Returns:
            List of (distance, slot_id) tuples, sorted ascending by distance.
            For cosine/dot collections, distance here is the negated score
            (so lower = more similar, consistent with euclidean).
            Phase 12 will convert back to the original score before returning
            to the user.

        Raises:
            EmptyIndexError: Index has no inserted vectors.
            InvalidEFError:  ef < k.
        """
        with self._lock:
            if self._entry_point is None:
                raise EmptyIndexError(
                    "Cannot search an empty HNSW index. Insert at least one vector first."
                )
            if ef < k:
                raise InvalidEFError(f"ef ({ef}) must be >= k ({k}).")

            return _knn_search(
                query_vector=query,
                k=k,
                ef=ef,
                entry_point=self._entry_point,
                max_layer=self._max_layer,
                graph=self._graph,
                vectors=self._vectors,
                dist_fn=self._dist_fn,
                skip_ids=skip_ids,
            )

    # ------------------------------------------------------------------
    # Integrity check (used by tests)
    # ------------------------------------------------------------------

    def check_integrity(self) -> dict:
        """
        Verify all structural invariants of the graph.

        Returns:
            Dict with keys: errors (list of str), warnings (list of str).
            An empty errors list means the graph is structurally correct.
        """
        errors = []
        warnings = []
        M = self._M

        for node_id, layers in self._graph.items():
            for layer, neighbours in layers.items():
                Mmax = 2 * M if layer == 0 else M

                # Degree constraint
                if len(neighbours) > Mmax:
                    errors.append(
                        f"Node {node_id} layer {layer}: degree {len(neighbours)} "
                        f"exceeds Mmax {Mmax}."
                    )

                # Bidirectionality
                for neighbour_id in neighbours:
                    if neighbour_id not in self._graph:
                        errors.append(
                            f"Node {node_id} references non-existent neighbour "
                            f"{neighbour_id} at layer {layer}."
                        )
                        continue
                    if layer not in self._graph[neighbour_id]:
                        errors.append(
                            f"Edge {node_id}→{neighbour_id} at layer {layer} "
                            f"has no reverse: {neighbour_id} has no layer {layer}."
                        )
                    elif node_id not in self._graph[neighbour_id][layer]:
                        errors.append(
                            f"Edge {node_id}→{neighbour_id} at layer {layer} "
                            f"is not bidirectional."
                        )

        # Entry point has highest layer
        if self._entry_point is not None:
            ep_max_layer = max(self._graph[self._entry_point].keys())
            if ep_max_layer != self._max_layer:
                errors.append(
                    f"Entry point {self._entry_point} max layer {ep_max_layer} "
                    f"!= recorded max_layer {self._max_layer}."
                )

        return {"errors": errors, "warnings": warnings}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of vectors in the index."""
        return len(self._vectors)

    @property
    def entry_point(self) -> Optional[NodeID]:
        return self._entry_point

    @property
    def max_layer(self) -> int:
        return self._max_layer