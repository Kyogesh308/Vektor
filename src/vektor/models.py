"""
vektor.models
-------------
Public data structures returned by the Vektor client API.

These are the ONLY structured types users receive from public methods.
Never return raw dicts or tuples from Collection or Vektor methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# SearchResult
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """
    A single result from a search or get operation.

    Attributes:
        id:       The user-supplied string ID of the vector.
        score:    Similarity or distance score. Meaning depends on the
                  collection's metric:
                  - cosine, dot: HIGHER score = more similar.
                  - euclidean:   LOWER score = more similar.
                  None for get() calls (no search score applies).
        vector:   The stored vector as a numpy float32 array. Only present
                  if include_vector=True was passed to search() or get().
                  None otherwise.
        metadata: The metadata dict stored with this vector. Empty dict
                  if no metadata was provided at insert time.
        rank:     1-indexed position in the returned result list.
                  None for get() calls (no ranking applies to a direct fetch).

    Equality is based on `id` only — two results with the same ID are
    considered equal regardless of score, since the ID uniquely identifies
    a vector in a collection.
    """
    id: str
    score: Optional[float]
    metadata: dict
    vector: Optional[np.ndarray] = None
    rank: Optional[int] = None

    def __eq__(self, other) -> bool:
        if not isinstance(other, SearchResult):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    def __repr__(self) -> str:
        score_str = f"{self.score:.6f}" if self.score is not None else "None"
        rank_str = f"#{self.rank}" if self.rank is not None else ""
        return f"SearchResult(id={self.id!r}, score={score_str}{' ' + rank_str if rank_str else ''})"


# ---------------------------------------------------------------------------
# VektorConfig
# ---------------------------------------------------------------------------

@dataclass
class VektorConfig:
    """
    Top-level configuration for the Vektor client.

    Attributes:
        data_dir:                 Directory where collection files are stored.
                                   Created if it doesn't exist. Default: ./vektor_data
        lock_timeout:              Seconds before a lock acquisition raises
                                   VektorTimeoutError. Default: 5.0
        log_level:                 Verbosity of internal log database entries.
                                   One of "DEBUG", "INFO", "WARNING", "ERROR".
                                   Default: "INFO"
        default_overfetch_factor:  Post-filter search over-fetch multiplier.
                                   Default: 3
    """
    data_dir: Path = field(default_factory=lambda: Path("./vektor_data"))
    lock_timeout: float = 5.0
    log_level: str = "INFO"
    default_overfetch_factor: int = 3

    def __post_init__(self):
        if not isinstance(self.data_dir, Path):
            self.data_dir = Path(self.data_dir)


# ---------------------------------------------------------------------------
# CollectionConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CollectionConfig:
    """
    Immutable configuration snapshot of a collection, set at creation time.

    Attributes:
        name:            Collection name.
        dim:             Vector dimension required for all inserts.
        metric:          Distance metric — "cosine", "euclidean", or "dot".
        M:               HNSW graph connectivity parameter.
        ef_construction: HNSW build-time beam width.
        mode:            "beginner" or "research".
        created_at:      ISO 8601 timestamp of collection creation.

    This object is frozen — attempting to set any field after construction
    raises dataclasses.FrozenInstanceError.
    """
    name: str
    dim: int
    metric: str
    M: int
    ef_construction: int
    mode: str
    created_at: str


# ---------------------------------------------------------------------------
# BatchInsertResult
# ---------------------------------------------------------------------------

@dataclass
class BatchInsertResult:
    """
    Result of a batch_insert() call.

    Attributes:
        inserted: Count of vectors successfully inserted.
        failed:   Count of vectors that failed validation or insertion.
        errors:   List of dicts, each with keys "id" and "error_message",
                  one entry per failed record. Empty list if failed == 0.

    Batch insert is NOT all-or-nothing: a bad record does not prevent
    the rest of the batch from being inserted.
    """
    inserted: int
    failed: int
    errors: list = field(default_factory=list)

    def __repr__(self) -> str:
        return f"BatchInsertResult(inserted={self.inserted}, failed={self.failed})"


# ---------------------------------------------------------------------------
# MemoryEstimate
# ---------------------------------------------------------------------------

@dataclass
class MemoryEstimate:
    """
    Estimated memory footprint for a collection at a given vector count.

    Attributes:
        graph_bytes:    Estimated bytes for the HNSW adjacency structure.
        vector_bytes:   Estimated bytes for raw vector storage.
        metadata_bytes: Rough estimate for SQLite metadata storage.
        total_bytes:    Sum of the above.
        total_mb:       total_bytes converted to megabytes, for convenience.
        warning:        Non-empty string if the estimate exceeds 80% of
                        available system memory. Empty string otherwise.

    These are LOWER-BOUND estimates. Python dict overhead makes actual
    graph memory larger than this formula predicts — treat this as a
    floor, not an exact prediction.
    """
    graph_bytes: int
    vector_bytes: int
    metadata_bytes: int
    total_bytes: int
    total_mb: float
    warning: str = ""