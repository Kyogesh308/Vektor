"""
Vektor — a production-style vector database built from scratch.

Public API:
    from vektor import Vektor, VektorConfig

Everything else in this package is an implementation detail.
"""

from vektor.client import (
    Vektor, VektorConfigError, CollectionAlreadyExistsError,
    CollectionNotFoundError,
)
from vektor.models import (
    SearchResult, VektorConfig, CollectionConfig,
    BatchInsertResult, MemoryEstimate,
)
from vektor.collection_api import VectorNotFoundError, DuplicateIDError
from vektor.concurrency.exceptions import VektorTimeoutError, EmptyCollectionError
from vektor.hnsw.exceptions import InvalidEFError

__version__ = "0.5.0"

__all__ = [
    "Vektor", "VektorConfig", "SearchResult", "CollectionConfig",
    "BatchInsertResult", "MemoryEstimate", "VektorConfigError",
    "CollectionAlreadyExistsError", "CollectionNotFoundError",
    "VectorNotFoundError", "DuplicateIDError", "VektorTimeoutError",
    "EmptyCollectionError", "InvalidEFError",
]