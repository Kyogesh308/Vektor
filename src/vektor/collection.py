from __future__ import annotations
"""
vektor.collection
-----------------
In-memory registry of named collections.

A collection is a named, typed container for vectors. It records:
- The expected vector dimension
- The distance metric for all searches
- HNSW graph parameters (inert until Phase 7)

This module does NOT store vectors. That is Phase 4's responsibility.
"""


from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class VektorCollectionError(Exception):
    """Base class for collection-layer errors."""


class CollectionAlreadyExistsError(VektorCollectionError):
    """Raised when creating a collection whose name is already registered."""


class CollectionNotFoundError(VektorCollectionError):
    """Raised when accessing a collection name that does not exist."""


class InvalidCollectionNameError(VektorCollectionError):
    """Raised when a collection name is empty or exceeds the maximum length."""


class InvalidDimensionError(VektorCollectionError):
    """Raised when the dimension is not a positive integer."""


class InvalidHNSWParameterError(VektorCollectionError):
    """Raised when M or ef_construction are out of their valid ranges."""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_METRICS = {"cosine", "euclidean", "dot"}
COLLECTION_NAME_MAX_LENGTH = 256

# HNSW parameter bounds — enforced at creation, used for real in Phase 7
M_MIN, M_MAX = 2, 128
EF_CONSTRUCTION_MIN, EF_CONSTRUCTION_MAX = 10, 2000


# ---------------------------------------------------------------------------
# Collection schema
# ---------------------------------------------------------------------------

@dataclass
class Collection:
    """
    Metadata record for a single Vektor collection.

    Instances are created by CollectionManager.create() and are read-only
    after creation. No field should be mutated directly.

    Attributes:
        name:             Unique identifier for the collection.
        dimension:        Required vector length for all inserts.
        metric:           Distance metric used for all searches.
        m:                HNSW graph connectivity parameter (Phase 7).
        ef_construction:  HNSW build-time beam width (Phase 7).
    """
    name: str
    dimension: int
    metric: str
    m: int = 16
    ef_construction: int = 200

    def to_dict(self) -> dict:
        """Serialize the collection record to a plain dict (for future persistence)."""
        return {
            "name": self.name,
            "dimension": self.dimension,
            "metric": self.metric,
            "m": self.m,
            "ef_construction": self.ef_construction,
        }

# ---------------------------------------------------------------------------
# Collection Manager
# ---------------------------------------------------------------------------

class CollectionManager:
    """
    In-memory registry of named collections.

    Provides create, get, delete, and list operations.
    All state is held in a single dict keyed by collection name.
    This class is intentionally stateful — one instance is shared across the
    system, acting as the single source of truth for collection metadata.

    Usage:
        manager = CollectionManager()
        manager.create("documents", dimension=1536, metric="cosine")
        col = manager.get("documents")
        manager.delete("documents")
        names = manager.list_collections()
    """

    def __init__(self) -> None:
        self._registry: dict[str, Collection] = {}

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    def create(
        self,
        name: str,
        dimension: int,
        metric: str,
        m: int = 16,
        ef_construction: int = 200,
    ) -> Collection:
        """
        Register a new named collection.

        Args:
            name:            Unique collection name. Non-empty, max 256 chars.
            dimension:       Required vector length. Must be a positive integer.
            metric:          Distance metric. One of "cosine", "euclidean", "dot".
            m:               HNSW connectivity. Range [2, 128]. Default 16.
            ef_construction: HNSW build beam width. Range [10, 2000]. Default 200.

        Returns:
            The newly created Collection record.

        Raises:
            InvalidCollectionNameError:  Name is empty or too long.
            InvalidDimensionError:       Dimension is not a positive integer.
            InvalidMetricError:          Metric is not in the supported set.
            InvalidHNSWParameterError:   M or ef_construction out of range.
            CollectionAlreadyExistsError: Name is already registered.
        """
        self._validate_name(name)
        self._validate_dimension(dimension)
        self._validate_metric(metric)
        self._validate_hnsw_params(m, ef_construction)

        if name in self._registry:
            raise CollectionAlreadyExistsError(
                f"Collection '{name}' already exists. "
                f"Use a different name or delete the existing collection first."
            )

        collection = Collection(
            name=name,
            dimension=dimension,
            metric=metric,
            m=m,
            ef_construction=ef_construction,
        )
        self._registry[name] = collection
        return collection

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, name: str) -> Collection:
        """
        Retrieve a collection by name.

        Args:
            name: The collection name to look up.

        Returns:
            The Collection record.

        Raises:
            CollectionNotFoundError: No collection with this name exists.
        """
        if name not in self._registry:
            raise CollectionNotFoundError(
                f"Collection '{name}' does not exist. "
                f"Available collections: {sorted(self._registry.keys()) or 'none'}."
            )
        return self._registry[name]

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete(self, name: str) -> None:
        """
        Remove a collection from the registry.

        Note: This removes the collection metadata only. In Phase 4, the
        VectorStore layer is responsible for also dropping associated vector
        data when a collection is deleted.

        Args:
            name: The collection name to delete.

        Raises:
            CollectionNotFoundError: No collection with this name exists.
        """
        if name not in self._registry:
            raise CollectionNotFoundError(
                f"Cannot delete '{name}': collection does not exist."
            )
        del self._registry[name]

    # ------------------------------------------------------------------
    # List
    # ------------------------------------------------------------------

    def list_collections(self) -> list[str]:
        """
        Return a sorted list of all registered collection names.

        Returns:
            List of collection name strings, sorted alphabetically.
            Empty list if no collections exist.
        """
        return sorted(self._registry.keys())

    def count(self) -> int:
        """Return the number of registered collections."""
        return len(self._registry)

    def exists(self, name: str) -> bool:
        """Return True if a collection with this name is registered."""
        return name in self._registry

    # ------------------------------------------------------------------
    # Internal validators
    # ------------------------------------------------------------------

    def _validate_name(self, name: str) -> None:
        if not isinstance(name, str) or len(name) == 0:
            raise InvalidCollectionNameError(
                "Collection name must be a non-empty string."
            )
        if len(name) > COLLECTION_NAME_MAX_LENGTH:
            raise InvalidCollectionNameError(
                f"Collection name length {len(name)} exceeds maximum "
                f"of {COLLECTION_NAME_MAX_LENGTH} characters."
            )

    def _validate_dimension(self, dimension: int) -> None:
        if not isinstance(dimension, int) or isinstance(dimension, bool):
            raise InvalidDimensionError(
                f"Dimension must be an integer, got {type(dimension).__name__}."
            )
        if dimension <= 0:
            raise InvalidDimensionError(
                f"Dimension must be a positive integer, got {dimension}."
            )

    def _validate_metric(self, metric: str) -> None:
        # Reuse the same exception type from Phase 1 is not appropriate here
        # because this is a collection-layer error, not a vector-layer error.
        # Raise a specific collection error with the supported set listed.
        if metric not in SUPPORTED_METRICS:
            raise VektorCollectionError(
                f"Metric '{metric}' is not supported. "
                f"Choose from: {sorted(SUPPORTED_METRICS)}."
            )

    def _validate_hnsw_params(self, m: int, ef_construction: int) -> None:
        if not isinstance(m, int) or isinstance(m, bool):
            raise InvalidHNSWParameterError(
                f"M must be an integer, got {type(m).__name__}."
            )
        if not (M_MIN <= m <= M_MAX):
            raise InvalidHNSWParameterError(
                f"M must be in range [{M_MIN}, {M_MAX}], got {m}."
            )
        if not isinstance(ef_construction, int) or isinstance(ef_construction, bool):
            raise InvalidHNSWParameterError(
                f"ef_construction must be an integer, got {type(ef_construction).__name__}."
            )
        if not (EF_CONSTRUCTION_MIN <= ef_construction <= EF_CONSTRUCTION_MAX):
            raise InvalidHNSWParameterError(
                f"ef_construction must be in range "
                f"[{EF_CONSTRUCTION_MIN}, {EF_CONSTRUCTION_MAX}], got {ef_construction}."
            )

"""Collection manager. Implementation begins in Phase 3."""