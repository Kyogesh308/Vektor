"""
vektor.concurrency.exceptions
------------------------------
Concurrency-specific exceptions.
"""

from __future__ import annotations


class VektorConcurrencyError(Exception):
    """Base class for concurrency-related errors."""


class VektorTimeoutError(VektorConcurrencyError):
    """
    Raised when a lock cannot be acquired within the configured timeout.

    This is not a fatal error. The caller should retry the operation
    after a short delay. If the timeout fires repeatedly, it indicates
    a writer is holding the lock for an abnormally long time.
    """


class EmptyCollectionError(VektorConcurrencyError):
    """
    Raised when search is attempted on a collection with no live vectors.

    Distinct from EmptyIndexError (Phase 7) — this is raised at the
    collection layer with lock held, not inside the HNSW algorithm.
    """