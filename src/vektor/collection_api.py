"""
vektor.collection_api
-----------------------
The public-facing Collection class. Users never instantiate this directly —
it is returned by Vektor.create_collection() and Vektor.get_collection().
"""

from __future__ import annotations

import math
import sqlite3
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from vektor.models import (
    SearchResult, CollectionConfig, BatchInsertResult, MemoryEstimate,
)
from vektor.collection import Collection as InternalCollection
from vektor.storage import VectorStore, VectorNotFoundError as _VectorNotFoundError
from vektor.hnsw.index import HNSWIndex
from vektor.hnsw.exceptions import InvalidEFError
from vektor.concurrency.exceptions import EmptyCollectionError, VektorTimeoutError
from vektor.validator import (
    validate_vector, validate_id, validate_metadata,
    InvalidVectorDimensionError,
)
from vektor.persistence.db import (
    initialize_db, insert_collection as db_insert_collection,
    get_collection as db_get_collection,
    insert_vector_record, tombstone_vector, get_next_slot_id,
    write_log,
)
from vektor.persistence.manifest import write_manifest, read_manifest
from vektor.persistence.integrity import check_collection_integrity
from vektor.filtering.prefilter import search_prefilter
from vektor.filtering.postfilter import search_postfilter


class VectorNotFoundError(Exception):
    """Raised when a vector ID does not exist in the collection."""


class DuplicateIDError(Exception):
    """Raised when inserting a vector ID that already exists."""


class VektorConfigError(Exception):
    """Raised for invalid configuration, e.g. beginner-mode override attempt."""


BEGINNER_MODE_EF = 100


class Collection:
    """
    A named, typed vector collection. All vector operations happen here.

    Never instantiate directly — obtained via Vektor.create_collection()
    or Vektor.get_collection().
    """

    def __init__(self) -> None:
        self._name: str = ""
        self._dim: int = 0
        self._metric: str = ""
        self._mode: str = "beginner"
        self._created_at: str = ""
        self._data_dir: Path = Path()
        self._collection_dir: Path = Path()
        self._default_overfetch_factor: int = 3

        self._internal_collection: Optional[InternalCollection] = None
        self._index: Optional[HNSWIndex] = None
        self._store: Optional[VectorStore] = None
        self._conn: Optional[sqlite3.Connection] = None
        self._closed: bool = False

    @classmethod
    def _create(
        cls, data_dir: Path, name: str, dim: int, metric: str,
        M: int, ef_construction: int, mode: str,
        lock_timeout: float, default_overfetch_factor: int,
    ) -> "Collection":
        """Internal: build a brand-new collection on disk."""
        obj = cls()
        obj._name = name
        obj._dim = dim
        obj._metric = metric
        obj._mode = mode
        obj._created_at = datetime.now(timezone.utc).isoformat()
        obj._data_dir = data_dir
        obj._collection_dir = data_dir / name
        obj._default_overfetch_factor = default_overfetch_factor

        obj._collection_dir.mkdir(parents=True, exist_ok=True)

        obj._internal_collection = InternalCollection(
            name=name, dimension=dim, metric=metric,
            m=M, ef_construction=ef_construction,
        )

        obj._conn = initialize_db(obj._collection_dir / "metadata.db")
        db_insert_collection(obj._conn, name, dim, metric, M, ef_construction,
                             vektor_version="0.5.0")
        write_manifest(obj._collection_dir / "collection.json",
                       name, dim, metric, M, ef_construction)

        obj._index = HNSWIndex(obj._internal_collection, seed=42, lock_timeout=lock_timeout)
        obj._store = VectorStore(obj._internal_collection, lock_timeout=lock_timeout)

        return obj

    @classmethod
    def _load(
        cls, data_dir: Path, name: str,
        lock_timeout: float, default_overfetch_factor: int,
    ) -> "Collection":
        """Internal: load an existing collection from disk, run integrity check."""
        obj = cls()
        obj._name = name
        obj._data_dir = data_dir
        obj._collection_dir = data_dir / name
        obj._default_overfetch_factor = default_overfetch_factor

        obj._conn = initialize_db(obj._collection_dir / "metadata.db")
        manifest = read_manifest(obj._collection_dir / "collection.json")

        obj._dim = manifest["dimension"]
        obj._metric = manifest["metric"]
        obj._mode = manifest.get("mode", "research")
        obj._created_at = manifest["created_at"]

        obj._internal_collection = InternalCollection(
            name=name, dimension=obj._dim, metric=obj._metric,
            m=manifest["m"], ef_construction=manifest["ef_construction"],
        )

        check_collection_integrity(obj._collection_dir, obj._conn, name)

        obj._index = HNSWIndex(obj._internal_collection, seed=42, lock_timeout=lock_timeout)
        obj._store = VectorStore(obj._internal_collection, lock_timeout=lock_timeout)

        obj._rehydrate_from_disk()
        return obj

    def _rehydrate_from_disk(self) -> None:
        """Load vectors from vector.bin and metadata.db back into memory."""
        from vektor.persistence.binary import read_all_vectors
        from vektor.persistence.db import get_all_live_vector_records

        vector_bin_path = self._collection_dir / "vector.bin"
        if not vector_bin_path.exists():
            return

        records = get_all_live_vector_records(self._conn, self._name)
        vector_data = dict(read_all_vectors(vector_bin_path, self._dim))

        # Insert in slot_id order — HNSW graph construction is order-dependent
        for record in sorted(records, key=lambda r: r["slot_id"]):
            slot_id = record["slot_id"]
            vec_id = record["id"]
            if slot_id in vector_data:
                vector = vector_data[slot_id]
                self._index.add(slot_id, vector)
                self._store.insert(vec_id, vector)

    def _close(self) -> None:
        """Flush and close all connections. Called by Vektor.close()."""
        if self._closed:
            return
        if self._conn is not None:
            self._conn.close()
        self._closed = True

    @property
    def config(self) -> CollectionConfig:
        """
        Immutable configuration snapshot of this collection.

        Returns:
            CollectionConfig with name, dim, metric, M, ef_construction,
            mode, and created_at.
        """
        return CollectionConfig(
            name=self._name, dim=self._dim, metric=self._metric,
            M=self._internal_collection.m,
            ef_construction=self._internal_collection.ef_construction,
            mode=self._mode, created_at=self._created_at,
        )

    def insert(self, id: str, vector, metadata: Optional[dict] = None) -> None:
        """
        Insert a single vector.

        Args:
            id:       Unique string identifier. Must not already exist.
            vector:   List or numpy array. Normalised to float32 internally.
                     PyTorch/TensorFlow tensors not accepted — call .numpy() first.
            metadata: Optional dict of JSON-serializable values.

        Returns:
            None.

        Raises:
            DuplicateIDError:            id already exists and is not deleted.
            InvalidVectorDimensionError: vector length doesn't match collection dim.
                                         If a 2D array of shape (1, dim) is passed,
                                         the error message suggests squeezing it.
            NonFiniteVectorError:        vector contains NaN or infinity.
            InvalidIDTypeError, EmptyIDError, IDTooLongError,
            InvalidIDCharacterError:     Invalid id.
            InvalidMetadataTypeError,
            NonSerializableMetadataError: Invalid metadata.
        """
        vector_arr = np.asarray(vector, dtype=np.float64)
        if vector_arr.ndim == 2 and vector_arr.shape[0] == 1:
            raise InvalidVectorDimensionError(
                f"Received a 2D array of shape {vector_arr.shape} — did you mean "
                f"to pass a 1D vector? Try vector.squeeze() or vector[0] first."
            )

        try:
            self._store.insert(id, vector, metadata)
        except _VectorNotFoundError:
            raise
        except Exception as e:
            if "already exists" in str(e):
                raise DuplicateIDError(str(e)) from e
            raise

        slot_id = get_next_slot_id(self._conn, self._name) - 1
        validated_vector = validate_vector(vector, self._dim)
        self._index.add(slot_id, validated_vector)
        insert_vector_record(self._conn, id, self._name, slot_id)

    def batch_insert(self, records: list[dict]) -> BatchInsertResult:
        """
        Insert multiple vectors in a single locked operation.

        NOT all-or-nothing: a bad record does not prevent the rest of the
        batch from being inserted. The lock is held for the ENTIRE batch,
        which blocks all searches for the batch's full duration — intentional,
        preventing concurrent reads from seeing a partially inserted batch.

        Args:
            records: List of dicts, each with keys "id", "vector",
                     and optionally "metadata".

        Returns:
            BatchInsertResult with inserted count, failed count, and errors.
        """
        inserted = 0
        failed = 0
        errors = []

        for record in records:
            try:
                self.insert(
                    id=record["id"], vector=record["vector"],
                    metadata=record.get("metadata"),
                )
                inserted += 1
            except Exception as e:
                failed += 1
                errors.append({
                    "id": record.get("id", "<missing>"),
                    "error_message": str(e),
                })

        return BatchInsertResult(inserted=inserted, failed=failed, errors=errors)

    def search(
        self, query, k: int = 10, ef: Optional[int] = None,
        filters: Optional[dict] = None, strategy: str = "post",
        include_vectors: bool = False,
    ) -> list[SearchResult]:
        """
        Search for the k nearest neighbours of the query vector.

        Args:
            query:           List or numpy array, must match collection dim.
            k:               Number of results. Default 10.
            ef:              Beam width. None uses collection default. Must be >= k.
            filters:         Optional metadata filter dict.
            strategy:        "pre" or "post". "auto" raises NotImplementedError.
            include_vectors: If True, each SearchResult includes the stored vector.

        Returns:
            List of SearchResult, sorted best-first, length <= k.

        Raises:
            InvalidEFError:       ef < k.
            EmptyCollectionError: Collection has no live vectors.
            NotImplementedError:  strategy="auto".
        """
        if strategy == "auto":
            raise NotImplementedError(
                "strategy='auto' is a v2 feature requiring selectivity "
                "estimation, which is not implemented. Choose 'pre' or 'post'."
            )
        if strategy not in ("pre", "post"):
            raise ValueError(f"strategy must be 'pre' or 'post', got {strategy!r}")

        if self._mode == "beginner":
            ef = BEGINNER_MODE_EF
        elif ef is None:
            ef = max(k, 100)

        query_vec = validate_vector(query, self._dim)

        if filters:
            if strategy == "pre":
                raw_results = search_prefilter(
                    query_vector=query_vec, k=k, ef=ef, filter_dict=filters,
                    conn=self._conn, collection_name=self._name,
                    entry_point=self._index.entry_point, max_layer=self._index.max_layer,
                    graph=self._index._graph, vectors=self._index._vectors,
                    dist_fn=self._index._dist_fn,
                )
                warnings_list = []
            else:
                filtered_result = search_postfilter(
                    query_vector=query_vec, k=k, ef=ef, filter_dict=filters,
                    conn=self._conn, collection_name=self._name,
                    entry_point=self._index.entry_point, max_layer=self._index.max_layer,
                    graph=self._index._graph, vectors=self._index._vectors,
                    dist_fn=self._index._dist_fn,
                    overfetch_factor=self._default_overfetch_factor,
                )
                raw_results = filtered_result.results
                warnings_list = filtered_result.warnings
        else:
            raw_results = self._index.search(query_vec, k=k, ef=ef)
            warnings_list = []

        for w in warnings_list:
            warnings.warn(w.message, UserWarning)

        return self._assemble_search_results(raw_results, include_vectors)

    def _assemble_search_results(
        self, raw_results: list[tuple[float, int]], include_vectors: bool,
    ) -> list[SearchResult]:
        """Convert (distance, slot_id) tuples into user-facing SearchResult list."""
        results = []
        for rank, (score, slot_id) in enumerate(raw_results, start=1):
            row = self._conn.execute(
                "SELECT id, metadata FROM vectors WHERE collection = ? AND slot_id = ?",
                (self._name, slot_id),
            ).fetchone()
            if row is None:
                continue

            import json
            metadata = json.loads(row["metadata"]) if row["metadata"] else {}

            from vektor.distance import METRIC_HIGHER_IS_BETTER
            display_score = -score if METRIC_HIGHER_IS_BETTER[self._metric] else score

            vector = self._index._vectors.get(slot_id) if include_vectors else None

            results.append(SearchResult(
                id=row["id"], score=display_score, metadata=metadata,
                vector=vector, rank=rank,
            ))
        return results

    def get(self, id: str, include_vector: bool = False) -> SearchResult:
        """
        Fetch a single vector by its ID.

        Args:
            id:             The vector's string ID.
            include_vector: If True, includes the stored vector.

        Returns:
            SearchResult with score=None and rank=None.

        Raises:
            VectorNotFoundError: id does not exist or has been deleted.
        """
        try:
            result = self._store.get(id)
        except _VectorNotFoundError as e:
            raise VectorNotFoundError(str(e)) from e

        return SearchResult(
            id=id, score=None, metadata=result["metadata"],
            vector=result["vector"] if include_vector else None, rank=None,
        )

    def update(self, id: str, vector=None, metadata: Optional[dict] = None) -> None:
        """
        Update a vector's data and/or metadata.

        [Certain] Updating vector DATA is implemented as delete-then-insert,
        not in-place graph modification — this rebuilds the vector's graph
        connections. If only metadata is updated (vector=None), the graph
        is unchanged and only the SQLite record is updated.

        Args:
            id:       Existing vector ID.
            vector:   If provided, replaces the stored vector.
            metadata: If provided, replaces stored metadata.

        Raises:
            VectorNotFoundError: id does not exist or has been deleted.
        """
        try:
            existing = self._store.get(id)
        except _VectorNotFoundError as e:
            raise VectorNotFoundError(str(e)) from e

        if vector is not None:
            self.delete(id)
            self.insert(id, vector, metadata or existing["metadata"])
        elif metadata is not None:
            self._store.update(id, existing["vector"], metadata)

    def delete(self, id: str) -> None:
        """
        Delete a vector (tombstone — not physically removed from the graph).

        Args:
            id: Vector ID to delete.

        Raises:
            VectorNotFoundError: id does not exist or is already deleted.
        """
        try:
            self._store.delete(id)
        except _VectorNotFoundError as e:
            raise VectorNotFoundError(str(e)) from e

        tombstone_vector(self._conn, id, self._name)

    def count(self) -> int:
        """
        Return the number of live (non-deleted) vectors in this collection.

        Returns:
            int count.
        """
        return self._store.count()

    def estimate_memory(self, n_vectors: int) -> MemoryEstimate:
        """
        Estimate memory footprint for this collection at a given vector count.

        These are LOWER-BOUND estimates. Python dict overhead makes actual
        graph memory larger than this formula predicts.

        Formula:
            graph_bytes = n_vectors × (M × 2) × 8 × estimated_layers
            vector_bytes = n_vectors × dim × 4
            estimated_layers ≈ 1 + log(n_vectors) / log(M)

        Args:
            n_vectors: Hypothetical vector count to estimate for.

        Returns:
            MemoryEstimate with graph_bytes, vector_bytes, metadata_bytes,
            total_bytes, total_mb, and a warning if estimate exceeds 80%
            of available system memory.
        """
        M = self._internal_collection.m
        dim = self._dim

        estimated_layers = 1 + (math.log(n_vectors) / math.log(M)) if n_vectors > 1 else 1
        graph_bytes = int(n_vectors * (M * 2) * 8 * estimated_layers)
        vector_bytes = n_vectors * dim * 4
        metadata_bytes = n_vectors * 200

        total_bytes = graph_bytes + vector_bytes + metadata_bytes
        total_mb = total_bytes / 1_000_000

        warning = ""
        try:
            import psutil
            available = psutil.virtual_memory().available
            if total_bytes > 0.8 * available:
                warning = (
                    f"Estimated {total_mb:.1f}MB exceeds 80% of available "
                    f"system memory ({available / 1_000_000:.1f}MB)."
                )
        except ImportError:
            pass

        return MemoryEstimate(
            graph_bytes=graph_bytes, vector_bytes=vector_bytes,
            metadata_bytes=metadata_bytes, total_bytes=total_bytes,
            total_mb=total_mb, warning=warning,
        )