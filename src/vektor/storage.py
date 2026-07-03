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

class VectorStore:
    """
    In-memory storage and brute-force search for a single collection.

    One VectorStore instance is created per collection. It does not know
    about other collections — the Collection object passed at init provides
    the dimension and metric contract this store enforces.

    Usage:
        store = VectorStore(collection)
        store.insert("vec_001", [0.1, 0.2, ...], {"source": "arxiv"})
        result = store.get("vec_001")
        results = store.search([0.1, 0.2, ...], k=5)
        store.update("vec_001", [0.3, 0.4, ...], {"source": "updated"})
        store.delete("vec_001")
    """

    def __init__(self, collection: Collection) -> None:
        self._collection = collection
        # Primary storage: id → float32 ndarray
        self._vectors: dict[str, np.ndarray] = {}
        # Metadata + tombstone flag: id → {"meta": dict, "deleted": bool}
        self._metadata: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def collection(self) -> Collection:
        return self._collection

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------

    def insert(
        self,
        id: str,
        vector: list | np.ndarray,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Validate and insert a vector into the store.

        Args:
            id:       Unique string identifier for this vector.
            vector:   The raw vector. Will be validated and stored as float32.
            metadata: Optional dict of JSON-serializable values.

        Raises:
            DuplicateVectorIDError: ID already exists and is not deleted.
            InvalidIDTypeError, EmptyIDError, ...: From Phase 1 validator.
            InvalidVectorDimensionError, NonFiniteVectorError, ...: From Phase 1.
            InvalidMetadataTypeError, NonSerializableMetadataError: From Phase 1.
        """
        if metadata is None:
            metadata = {}

        # Phase 1 validation — order matters: ID first, then vector, then metadata
        validated_id = validate_id(id)
        validated_vector = validate_vector(vector, self._collection.dimension)
        validated_metadata = validate_metadata(metadata)

        # Duplicate check: reject if ID exists and is NOT tombstoned
        if validated_id in self._vectors and not self._metadata[validated_id]["deleted"]:
            raise DuplicateVectorIDError(
                f"Vector ID '{validated_id}' already exists. "
                f"Use update() to replace it or delete() first."
            )

        self._vectors[validated_id] = validated_vector
        self._metadata[validated_id] = {
            "meta": validated_metadata,
            "deleted": False,
        }

    # ------------------------------------------------------------------
    # Get
    # ------------------------------------------------------------------

    def get(self, id: str) -> dict:
        """
        Retrieve a stored vector and its metadata by ID.

        Args:
            id: The vector ID to retrieve.

        Returns:
            Dict with keys "id", "vector" (float32 ndarray), "metadata" (dict).

        Raises:
            VectorNotFoundError: ID does not exist or has been deleted.
        """
        self._assert_exists(id)
        return {
            "id": id,
            "vector": self._vectors[id],
            "metadata": self._metadata[id]["meta"],
        }

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        id: str,
        vector: list | np.ndarray,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Replace an existing vector and its metadata.

        Args:
            id:       The vector ID to update. Must exist and not be deleted.
            vector:   New vector. Will be re-validated against collection dimension.
            metadata: New metadata. If None, existing metadata is preserved.

        Raises:
            VectorNotFoundError: ID does not exist or has been deleted.
            InvalidVectorDimensionError, NonFiniteVectorError, ...: From Phase 1.
        """
        self._assert_exists(id)

        validated_vector = validate_vector(vector, self._collection.dimension)

        if metadata is not None:
            validated_metadata = validate_metadata(metadata)
        else:
            validated_metadata = self._metadata[id]["meta"]

        self._vectors[id] = validated_vector
        self._metadata[id]["meta"] = validated_metadata

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete(self, id: str) -> None:
        """
        Tombstone a vector, making it invisible to search and retrieval.

        The vector is NOT physically removed — it is flagged as deleted.
        This is intentional: Phase 7's HNSW graph preserves deleted nodes
        as structural stubs, only filtering them from results.

        Args:
            id: The vector ID to delete.

        Raises:
            VectorNotFoundError: ID does not exist or is already deleted.
        """
        self._assert_exists(id)
        self._metadata[id]["deleted"] = True

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: list | np.ndarray,
        k: int,
    ) -> list[SearchResult]:
        """
        Brute-force nearest-neighbor search.

        Computes the distance from the query to every non-deleted stored
        vector, sorts by the collection's metric, and returns the top k.

        This is the ground truth oracle. Every HNSW recall benchmark in
        Phase 7 onward is measured against this function's output.

        Args:
            query: The query vector. Must match the collection dimension.
            k:     Number of results to return.

        Returns:
            List of SearchResult, sorted best-first. Length is min(k, live_count).

        Raises:
            InvalidKError: k is not a positive integer.
            InvalidVectorDimensionError, NonFiniteVectorError, ...: From Phase 1.
        """
        # Validate k
        if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
            raise InvalidKError(
                f"k must be a positive integer, got {k!r}."
            )

        # Validate query vector against collection dimension
        validated_query = validate_vector(query, self._collection.dimension)

        metric = self._collection.metric

        # Compute distances to all live (non-deleted) vectors
        scores: list[tuple[str, float]] = []
        for vec_id, stored_vector in self._vectors.items():
            if self._metadata[vec_id]["deleted"]:
                continue
            score = compute_distance(metric, validated_query, stored_vector)
            scores.append((vec_id, score))

        # Sort using the Phase 2 registry — never hardcode reverse=True/False
        reverse = METRIC_HIGHER_IS_BETTER[metric]
        scores.sort(key=lambda x: x[1], reverse=reverse)

        # Return top k (or fewer if fewer live vectors exist)
        return [
            SearchResult(
                id=vec_id,
                score=score,
                metadata=self._metadata[vec_id]["meta"],
            )
            for vec_id, score in scores[:k]
        ]

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return the number of live (non-deleted) vectors."""
        return sum(
            1 for m in self._metadata.values() if not m["deleted"]
        )

    def count_all(self) -> int:
        """Return total entries including tombstoned vectors."""
        return len(self._vectors)

    def ids(self) -> list[str]:
        """Return a sorted list of live (non-deleted) vector IDs."""
        return sorted(
            vec_id for vec_id, m in self._metadata.items()
            if not m["deleted"]
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assert_exists(self, id: str) -> None:
        """Raise VectorNotFoundError if ID is missing or tombstoned."""
        if id not in self._vectors or self._metadata[id]["deleted"]:
            raise VectorNotFoundError(
                f"Vector ID '{id}' does not exist or has been deleted."
            )
"""In-memory vector storage and search. Implementation begins in Phase 4."""