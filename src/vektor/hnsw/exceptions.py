"""
vektor.hnsw.exceptions
-----------------------
Named exceptions for the HNSW index layer.
"""

from __future__ import annotations


class HNSWError(Exception):
    """Base class for all HNSW errors."""


class EmptyIndexError(HNSWError):
    """Raised when search is attempted on an index with no inserted vectors."""


class InvalidEFError(HNSWError):
    """Raised when ef < k at search time."""


class InvalidMError(HNSWError):
    """Raised when M is outside the valid range [2, 128]."""


class GraphCorruptionError(HNSWError):
    """Raised when a structural invariant violation is detected during integrity check."""