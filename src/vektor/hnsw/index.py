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


# src/vektor/hnsw/index.py — updated relevant sections

from vektor.concurrency.lock import CollectionLock
from vektor.concurrency.exceptions import VektorTimeoutError

class HNSWIndex:

    def __init__(
        self,
        collection: Collection,
        seed: int = 42,
        lock_timeout: float = 5.0,
    ) -> None:
        self._collection = collection
        self._M = collection.m
        self._ef_construction = collection.ef_construction
        self._seed = seed
        self._rng = random.Random(seed)
        self._graph: Graph = {}
        self._vectors: dict[NodeID, np.ndarray] = {}
        self._entry_point: Optional[NodeID] = None
        self._max_layer: int = 0

        # v1: single exclusive lock for all operations
        # v2 upgrade: replace with reader-writer lock
        self._lock = CollectionLock(timeout=lock_timeout)

        metric = collection.metric
        if METRIC_HIGHER_IS_BETTER[metric]:
            def _dist(a: np.ndarray, b: np.ndarray) -> float:
                return -compute_distance(metric, a, b)
        else:
            def _dist(a: np.ndarray, b: np.ndarray) -> float:
                return compute_distance(metric, a, b)
        self._dist_fn = _dist

    def add(
        self,
        slot_id: NodeID,
        vector: np.ndarray,
        timeout: float = None,
    ) -> None:
        """
        Insert a vector. Acquires the exclusive lock.

        Args:
            slot_id: Integer slot ID.
            vector:  Float32 ndarray, pre-validated.
            timeout: Lock acquisition timeout. Uses instance default if None.

        Raises:
            VektorTimeoutError: Lock not acquired within timeout.
        """
        new_layer = assign_layer(self._M, self._rng)

        # Lock acquired here — released in finally even if insert raises
        with self._lock.acquire(operation="insert", timeout=timeout):
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

    def search(
    self,
    query: np.ndarray,
    k: int,
    ef: int,
    skip_ids: Optional[set[NodeID]] = None,      # <-- restore
    skip_from_results: Optional[frozenset[NodeID]] = None,
    skip_entirely: Optional[frozenset[NodeID]] = None,
    timeout: float = None,
) -> list[tuple[float, NodeID]]:
        """
        Search. Acquires the exclusive lock.

        # v2 note: this should acquire a shared (read) lock.
        # In v1, reads are serialised with writes — no concurrent search.

        Raises:
            EmptyCollectionError: No live vectors in the index.
            InvalidEFError:       ef < k.
            VektorTimeoutError:   Lock not acquired within timeout.
        """
        # Lock acquired here — tombstone set must be built INSIDE the lock
        # to prevent a delete committing between tombstone build and search start
        with self._lock.acquire(operation="search", timeout=timeout):
            # ----------------------------------------------------------
            # Backward compatibility (Phase 7 -> Phase 9)
            # ----------------------------------------------------------

            if skip_ids is not None and skip_from_results is None:
                skip_from_results = frozenset(skip_ids)

            if skip_from_results is None:
                skip_from_results = frozenset()

            if skip_entirely is None:
                skip_entirely = frozenset()
                
            # Entry point guard — must be inside lock
            if self._entry_point is None:
                raise EmptyIndexError(
                    "Cannot search an empty HNSW index. "
                    "Insert at least one vector first."
                )

            if ef < k:
                from vektor.hnsw.exceptions import InvalidEFError
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
                skip_from_results=skip_from_results or frozenset(),
                skip_entirely=skip_entirely or frozenset(),
            )

    def check_integrity(self, timeout: float = None) -> dict:
        """Acquires the lock for a consistent integrity snapshot."""
        with self._lock.acquire(operation="integrity_check", timeout=timeout):
            # ... existing integrity check logic, unchanged ...
            errors = []
            warnings = []
            M = self._M
            for node_id, layers in self._graph.items():
                for layer, neighbours in layers.items():
                    Mmax = 2 * M if layer == 0 else M
                    if len(neighbours) > Mmax:
                        errors.append(
                            f"Node {node_id} layer {layer}: degree {len(neighbours)} "
                            f"exceeds Mmax {Mmax}."
                        )
                    for neighbour_id in neighbours:
                        if neighbour_id not in self._graph:
                            errors.append(
                                f"Node {node_id} references non-existent "
                                f"neighbour {neighbour_id} at layer {layer}."
                            )
                            continue
                        if layer not in self._graph[neighbour_id]:
                            errors.append(
                                f"Edge {node_id}→{neighbour_id} at layer {layer} "
                                f"has no reverse."
                            )
                        elif node_id not in self._graph[neighbour_id][layer]:
                            errors.append(
                                f"Edge {node_id}→{neighbour_id} at layer {layer} "
                                f"is not bidirectional."
                            )
            if self._entry_point is not None:
                ep_max = max(self._graph[self._entry_point].keys())
                if ep_max != self._max_layer:
                    errors.append(
                        f"Entry point max layer {ep_max} != "
                        f"recorded max_layer {self._max_layer}."
                    )
            return {"errors": errors, "warnings": warnings}

    @property
    def size(self) -> int:
        return len(self._vectors)

    @property
    def entry_point(self) -> Optional[NodeID]:
        return self._entry_point

    @property
    def max_layer(self) -> int:
        return self._max_layer

