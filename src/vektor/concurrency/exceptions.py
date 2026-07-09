"""
vektor.concurrency.exceptions
------------------------------
Concurrency-specific exceptions.
"""

from __future__ import annotations

from vektor.hnsw.exceptions import EmptyIndexError


class VektorConcurrencyError(Exception):
    """Base class for concurrency-related errors."""


class VektorTimeoutError(VektorConcurrencyError):
    """Raised when a lock cannot be acquired within the configured timeout."""


class EmptyCollectionError(EmptyIndexError):
    """
    Raised when search is attempted on a collection with no live vectors.

    Subclasses EmptyIndexError (Phase 7) so existing code and tests that
    catch EmptyIndexError continue to work unchanged. This is a narrowing,
    not a replacement — EmptyCollectionError IS-AN EmptyIndexError.
    """