"""
vektor.client
-------------
The top-level Vektor client. This is the only class most users import.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

from vektor.models import VektorConfig
from vektor.collection_api import Collection
from vektor.collection import Collection as InternalCollection


class VektorConfigError(Exception):
    """Raised for invalid configuration, e.g. beginner-mode parameter override."""


class CollectionAlreadyExistsError(Exception):
    """Raised when create_collection is called with a name already on disk."""


class CollectionNotFoundError(Exception):
    """Raised when get_collection or delete_collection references an unknown name."""


class Vektor:
    """
    Top-level entry point for the Vektor vector database.

    Usage:
        from vektor import Vektor, VektorConfig

        client = Vektor(VektorConfig(data_dir="./my_data"))
        collection = client.create_collection("documents", dim=1536, metric="cosine")
        collection.insert("doc1", [0.1, 0.2, ...], {"source": "arxiv"})
        results = collection.search([0.1, 0.2, ...], k=5)
    """

    def __init__(self, config: Optional[VektorConfig] = None) -> None:
        """
        Args:
            config: VektorConfig instance. Uses defaults if None.
        """
        self._config = config or VektorConfig()
        self._config.data_dir.mkdir(parents=True, exist_ok=True)

        # One Collection object per name per process — required for
        # Phase 10 lock guarantees (two objects would hold separate locks).
        self._open_collections: dict[str, Collection] = {}

    def create_collection(
        self,
        name: str,
        dim: int,
        metric: str,
        mode: str = "beginner",
        M: Optional[int] = None,
        ef_construction: Optional[int] = None,
    ) -> Collection:
        """
        Create a new named collection.

        Args:
            name:            Unique collection name.
            dim:             Vector dimension for all inserts.
            metric:          "cosine", "euclidean", or "dot".
            mode:            "beginner" (fixed M=16, ef_construction=200, ef=100)
                             or "research" (all parameters configurable).
                             Default "beginner".
            M:               HNSW connectivity. Only valid in research mode.
            ef_construction: HNSW build beam width. Only valid in research mode.

        Returns:
            A Collection object for immediate use.

        Raises:
            CollectionAlreadyExistsError: A collection with this name exists on disk.
            VektorConfigError:            mode is invalid, or M/ef_construction
                                          given in beginner mode.
        """
        if mode not in ("beginner", "research"):
            raise VektorConfigError(
                f"mode must be 'beginner' or 'research', got {mode!r}."
            )

        collection_dir = self._config.data_dir / name
        if collection_dir.exists():
            raise CollectionAlreadyExistsError(
                f"Collection '{name}' already exists at {collection_dir}. "
                f"Delete it first if you want to recreate it."
            )

        if mode == "beginner":
            if M is not None or ef_construction is not None:
                raise VektorConfigError(
                    f"M and ef_construction cannot be set in beginner mode. "
                    f"Use mode='research' to configure HNSW parameters directly."
                )
            M = 16
            ef_construction = 200
        else:
            M = M if M is not None else 16
            ef_construction = ef_construction if ef_construction is not None else 200

        collection = Collection._create(
            data_dir=self._config.data_dir,
            name=name, dim=dim, metric=metric,
            M=M, ef_construction=ef_construction, mode=mode,
            lock_timeout=self._config.lock_timeout,
            default_overfetch_factor=self._config.default_overfetch_factor,
        )

        self._open_collections[name] = collection
        return collection

    def get_collection(self, name: str) -> Collection:
        """
        Open an existing collection.

        If already open in this process, returns the same object.
        Otherwise loads from disk, running the startup integrity check.

        Args:
            name: Collection name.

        Returns:
            Collection object.

        Raises:
            CollectionNotFoundError: No collection with this name exists on disk.
        """
        if name in self._open_collections:
            return self._open_collections[name]

        collection_dir = self._config.data_dir / name
        if not collection_dir.exists():
            raise CollectionNotFoundError(
                f"Collection '{name}' does not exist at {collection_dir}."
            )

        collection = Collection._load(
            data_dir=self._config.data_dir, name=name,
            lock_timeout=self._config.lock_timeout,
            default_overfetch_factor=self._config.default_overfetch_factor,
        )
        self._open_collections[name] = collection
        return collection

    def delete_collection(self, name: str) -> None:
        """
        Permanently delete a collection and all its data from disk.

        Args:
            name: Collection name.

        Raises:
            CollectionNotFoundError: No collection with this name exists.
        """
        collection_dir = self._config.data_dir / name
        if not collection_dir.exists():
            raise CollectionNotFoundError(f"Collection '{name}' does not exist.")

        if name in self._open_collections:
            self._open_collections[name]._close()
            del self._open_collections[name]

        import shutil
        shutil.rmtree(collection_dir)

    def list_collections(self) -> list[str]:
        """
        List all collection names present in the data directory.

        Returns:
            Sorted list of collection names.
        """
        if not self._config.data_dir.exists():
            return []
        return sorted(p.name for p in self._config.data_dir.iterdir() if p.is_dir())

    def close(self) -> None:
        """Close all open collections and release SQLite connections."""
        for collection in self._open_collections.values():
            collection._close()
        self._open_collections.clear()

    def __enter__(self) -> "Vektor":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()