"""
vektor.storage
--------------
In-memory vector storage and brute-force nearest-neighbor search.

Design contracts:
- All vectors are stored as float32 NumPy arrays.
- Input is validated by vektor.validator before storage.
- Search uses vektor.distance.compute_distance and METRIC_HIGHER_IS_BETTER.
- Deleted vectors are tombstoned, not physically removed.
- One VectorStore instance serves one collection.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Optional

from vektor.validator import (
    validate_vector,
    validate_id,
    validate_metadata,
)
from vektor.distance import compute_distance, METRIC_HIGHER_IS_BETTER
from vektor.collection import Collection

from vektor.concurrency.lock import CollectionLock
from vektor.concurrency.exceptions import VektorTimeoutError

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class VektorStorageError(Exception):
    """Base class for storage-layer errors."""


class VectorNotFoundError(VektorStorageError):
    """Raised when a vector ID does not exist or has been deleted."""


class DuplicateVectorIDError(VektorStorageError):
    """Raised when inserting a vector ID that already exists (non-deleted)."""


class InvalidKError(VektorStorageError):
    """Raised when k is not a positive integer in a search call."""


# ---------------------------------------------------------------------------
# Search result
# ---------------------------------------------------------------------------

class SearchResult:
    """
    A single result returned by brute-force search.

    Attributes:
        id:       The vector's ID.
        score:    The distance or similarity score.
                  Interpretation depends on the collection metric:
                  cosine/dot → higher is more similar.
                  euclidean  → lower is more similar.
        metadata: The metadata dict stored with this vector.
    """
    __slots__ = ("id", "score", "metadata")

    def __init__(self, id: str, score: float, metadata: dict) -> None:
        self.id = id
        self.score = score
        self.metadata = metadata

    def __repr__(self) -> str:
        return f"SearchResult(id={self.id!r}, score={self.score:.6f})"


# ---------------------------------------------------------------------------
# Vector Store
# ---------------------------------------------------------------------------

# src/vektor/storage.py — updated class with lock

from vektor.concurrency.lock import CollectionLock
from vektor.concurrency.exceptions import VektorTimeoutError

class VectorStore:

    def __init__(
        self,
        collection: Collection,
        lock_timeout: float = 5.0,
    ) -> None:
        self._collection = collection
        self._vectors: dict[str, np.ndarray] = {}
        self._metadata: dict[str, dict] = {}
        # Shared lock — if HNSWIndex and VectorStore serve the same collection,
        # pass in the SAME CollectionLock instance so they share coordination.
        self._lock = CollectionLock(timeout=lock_timeout)

    def insert(self, id: str, vector, metadata=None, timeout: float = None) -> None:
        if metadata is None:
            metadata = {}
        validated_id = validate_id(id)
        validated_vector = validate_vector(vector, self._collection.dimension)
        validated_metadata = validate_metadata(metadata)

        with self._lock.acquire(operation="insert", timeout=timeout):
            if validated_id in self._vectors and \
               not self._metadata[validated_id]["deleted"]:
                raise DuplicateVectorIDError(
                    f"Vector ID '{validated_id}' already exists."
                )
            self._vectors[validated_id] = validated_vector
            self._metadata[validated_id] = {
                "meta": validated_metadata,
                "deleted": False,
            }

    def get(self, id: str, timeout: float = None) -> dict:
        with self._lock.acquire(operation="get", timeout=timeout):
            self._assert_exists(id)
            return {
                "id": id,
                "vector": self._vectors[id],
                "metadata": self._metadata[id]["meta"],
            }

    def update(self, id: str, vector, metadata=None, timeout: float = None) -> None:
        validated_vector = validate_vector(vector, self._collection.dimension)
        with self._lock.acquire(operation="update", timeout=timeout):
            self._assert_exists(id)
            if metadata is not None:
                validated_metadata = validate_metadata(metadata)
            else:
                validated_metadata = self._metadata[id]["meta"]
            self._vectors[id] = validated_vector
            self._metadata[id]["meta"] = validated_metadata

    def delete(self, id: str, timeout: float = None) -> None:
        with self._lock.acquire(operation="delete", timeout=timeout):
            self._assert_exists(id)
            self._metadata[id]["deleted"] = True

    def search(self, query, k: int, timeout: float = None):
        if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
            raise InvalidKError(f"k must be a positive integer, got {k!r}.")
        validated_query = validate_vector(query, self._collection.dimension)

        with self._lock.acquire(operation="search", timeout=timeout):
            metric = self._collection.metric
            scores = []
            for vec_id, stored_vector in self._vectors.items():
                if self._metadata[vec_id]["deleted"]:
                    continue
                score = compute_distance(metric, validated_query, stored_vector)
                scores.append((vec_id, score))

            reverse = METRIC_HIGHER_IS_BETTER[metric]
            scores.sort(key=lambda x: x[1], reverse=reverse)

            return [
                SearchResult(
                    id=vec_id,
                    score=score,
                    metadata=self._metadata[vec_id]["meta"],
                )
                for vec_id, score in scores[:k]
            ]

    def count(self, timeout: float = None) -> int:
        with self._lock.acquire(operation="count", timeout=timeout):
            return sum(1 for m in self._metadata.values() if not m["deleted"])

    def count_all(self) -> int:
        return len(self._vectors)

    def ids(self, timeout: float = None) -> list[str]:
        with self._lock.acquire(operation="ids", timeout=timeout):
            return sorted(
                vid for vid, m in self._metadata.items() if not m["deleted"]
            )

    def _assert_exists(self, id: str) -> None:
        if id not in self._vectors or self._metadata[id]["deleted"]:
            raise VectorNotFoundError(
                f"Vector ID '{id}' does not exist or has been deleted."
            )
    
