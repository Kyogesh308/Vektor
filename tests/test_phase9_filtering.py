"""
Phase 9 — Metadata Filtering and Tombstone-Aware Search Tests

Test order:
  1. Parser unit tests (no index needed)
  2. Tombstone tests (no filter needed)
  3. Pre-filter correctness
  4. Post-filter correctness + warning system
  5. Integration: combined tombstone + filter
  6. Recall benchmark (selectivity sweep)

Do not run recall tests before correctness tests pass.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from vektor.collection import Collection
from vektor.filtering.parser import parse_filter, get_eligible_slot_ids, FilterParseError
from vektor.filtering.tombstone import get_tombstone_slot_ids, recover_entry_point
from vektor.filtering.prefilter import search_prefilter
from vektor.filtering.postfilter import (
    search_postfilter, FilteredSearchResult, SearchWarning,
)
from vektor.hnsw.index import HNSWIndex
from vektor.persistence.db import (
    initialize_db, insert_collection, insert_vector_record, tombstone_vector,
)
from vektor.storage import VectorStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db_conn(tmp_path):
    conn = initialize_db(tmp_path / "metadata.db")
    insert_collection(conn, "test", 8, "euclidean", 16, 200, "0.4.0")
    return conn


@pytest.fixture
def collection():
    return Collection(name="test", dimension=8, metric="euclidean",
                      m=8, ef_construction=50)


@pytest.fixture
def index(collection):
    return HNSWIndex(collection, seed=42)


def make_vec(dim=8, val=1.0) -> np.ndarray:
    return np.array([val] * dim, dtype=np.float32)


def make_random_vecs(n: int, dim: int, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(dim).astype(np.float32) for _ in range(n)]


# ---------------------------------------------------------------------------
# Filter Parser
# ---------------------------------------------------------------------------

class TestFilterParser:

    def test_empty_filter_returns_empty_clause(self):
        clause, params = parse_filter({})
        assert clause == ""
        assert params == []

    def test_equality_shorthand(self):
        clause, params = parse_filter({"year": 2024})
        assert "json_extract" in clause
        assert "= ?" in clause
        assert params == [2024]

    def test_equality_explicit(self):
        clause, params = parse_filter({"year": {"eq": 2024}})
        assert "= ?" in clause
        assert params == [2024]

    def test_greater_than(self):
        clause, params = parse_filter({"score": {"gt": 0.5}})
        assert "> ?" in clause
        assert params == [0.5]

    def test_less_than_or_equal(self):
        clause, params = parse_filter({"year": {"lte": 2023}})
        assert "<= ?" in clause
        assert params == [2023]

    def test_range_filter(self):
        clause, params = parse_filter({"year": {"gte": 2020, "lte": 2024}})
        assert ">= ?" in clause
        assert "<= ?" in clause
        assert 2020 in params
        assert 2024 in params

    def test_in_operator(self):
        clause, params = parse_filter({"tag": {"in": ["ml", "rag", "llm"]}})
        assert "IN" in clause
        assert params == ["ml", "rag", "llm"]

    def test_in_operator_empty_list_raises(self):
        with pytest.raises(FilterParseError):
            parse_filter({"tag": {"in": []}})

    def test_unsupported_operator_raises(self):
        with pytest.raises(FilterParseError):
            parse_filter({"year": {"between": [2020, 2024]}})

    def test_invalid_field_name_raises(self):
        with pytest.raises(FilterParseError):
            parse_filter({"fi/eld": "value"})

    def test_sql_injection_value_is_bound_not_interpolated(self):
        # The dangerous value must appear only in params, never in the clause string
        dangerous = "'; DROP TABLE vectors; --"
        clause, params = parse_filter({"name": dangerous})
        assert dangerous not in clause
        assert dangerous in params

    def test_multiple_fields_joined_with_and(self):
        clause, params = parse_filter({"year": 2024, "tag": "ml"})
        assert " AND " in clause

    def test_no_user_values_in_clause_string(self):
        clause, params = parse_filter({"year": 9999, "tag": "injection_attempt"})
        assert "9999" not in clause
        assert "injection_attempt" not in clause


# ---------------------------------------------------------------------------
# Tombstone
# ---------------------------------------------------------------------------

class TestTombstoneSetBuilder:

    def test_empty_collection_returns_empty_frozenset(self, db_conn):
        result = get_tombstone_slot_ids(db_conn, "test")
        assert result == frozenset()
        assert isinstance(result, frozenset)

    def test_deleted_vector_appears_in_set(self, db_conn):
        insert_vector_record(db_conn, "v1", "test", slot_id=0)
        tombstone_vector(db_conn, "v1", "test")
        result = get_tombstone_slot_ids(db_conn, "test")
        assert 0 in result

    def test_live_vector_not_in_tombstone_set(self, db_conn):
        insert_vector_record(db_conn, "v1", "test", slot_id=0)
        result = get_tombstone_slot_ids(db_conn, "test")
        assert 0 not in result

    def test_mixed_live_and_deleted(self, db_conn):
        insert_vector_record(db_conn, "v0", "test", slot_id=0)
        insert_vector_record(db_conn, "v1", "test", slot_id=1)
        tombstone_vector(db_conn, "v0", "test")
        result = get_tombstone_slot_ids(db_conn, "test")
        assert 0 in result
        assert 1 not in result


class TestEntryPointRecovery:

    def test_valid_entry_point_returned_unchanged(self, db_conn, index):
        v = make_vec()
        insert_vector_record(db_conn, "v0", "test", slot_id=0)
        index.add(0, v)
        result = recover_entry_point(db_conn, "test", 0, index._graph)
        assert result == 0

    def test_tombstoned_entry_point_replaced(self, db_conn, index):
        for i in range(10):
            v = make_random_vecs(1, 8, seed=i)[0]
            insert_vector_record(db_conn, f"v{i}", "test", slot_id=i)
            index.add(i, v)

        # Tombstone the entry point
        ep = index.entry_point
        tombstone_vector(db_conn, f"v{ep}", "test")

        new_ep = recover_entry_point(db_conn, "test", ep, index._graph)
        assert new_ep != ep
        assert new_ep is not None

    def test_all_vectors_tombstoned_returns_none(self, db_conn, index):
        v = make_vec()
        insert_vector_record(db_conn, "v0", "test", slot_id=0)
        index.add(0, v)
        tombstone_vector(db_conn, "v0", "test")
        result = recover_entry_point(db_conn, "test", 0, index._graph)
        assert result is None


# ---------------------------------------------------------------------------
# Pre-Filter Search
# ---------------------------------------------------------------------------

class TestPreFilterSearch:

    def _build_index_with_metadata(self, db_conn, n=100):
        """
        Insert n vectors. First half get partition=0, second half partition=1.
        Returns (index, vectors_dict).
        """
        col = Collection(name="test", dimension=8, metric="euclidean",
                         m=8, ef_construction=100)
        idx = HNSWIndex(col, seed=42)
        vecs = make_random_vecs(n, 8, seed=0)

        for i, v in enumerate(vecs):
            partition = 0 if i < n // 2 else 1
            meta = f'{{"partition": {partition}}}'
            # Insert with metadata stored in the vectors table
            conn_exec = db_conn.execute(
                """INSERT INTO vectors (id, collection, slot_id, deleted, created_at, metadata)
                   VALUES (?, 'test', ?, 0, datetime('now'), ?)""",
                (f"v{i}", i, meta),
            )
            db_conn.commit()
            idx.add(i, v)

        return idx, {i: v for i, v in enumerate(vecs)}

    def _dist_fn(self, a, b):
        return float(np.linalg.norm(a - b))

    def test_all_results_satisfy_filter(self, db_conn):
        # Schema update: add metadata column to vectors table
        try:
            db_conn.execute("ALTER TABLE vectors ADD COLUMN metadata TEXT")
            db_conn.commit()
        except Exception:
            pass

        idx, vecs = self._build_index_with_metadata(db_conn, n=100)
        query = make_random_vecs(1, 8, seed=99)[0]

        results = search_prefilter(
            query_vector=query,
            k=5,
            ef=50,
            filter_dict={"partition": 0},
            conn=db_conn,
            collection_name="test",
            entry_point=idx.entry_point,
            max_layer=idx.max_layer,
            graph=idx._graph,
            vectors=idx._vectors,
            dist_fn=idx._dist_fn,
        )

        assert len(results) > 0
        for _, slot_id in results:
            # slot_id < 50 means partition=0
            assert slot_id < 50, (
                f"Result slot {slot_id} is in partition=1, violates filter."
            )

    def test_deleted_vectors_absent_from_results(self, db_conn):
        try:
            db_conn.execute("ALTER TABLE vectors ADD COLUMN metadata TEXT")
            db_conn.commit()
        except Exception:
            pass

        idx, vecs = self._build_index_with_metadata(db_conn, n=50)
        # Tombstone half the eligible partition
        for i in range(10):
            tombstone_vector(db_conn, f"v{i}", "test")

        query = make_random_vecs(1, 8, seed=77)[0]
        results = search_prefilter(
            query_vector=query,
            k=5, ef=50,
            filter_dict={"partition": 0},
            conn=db_conn,
            collection_name="test",
            entry_point=idx.entry_point,
            max_layer=idx.max_layer,
            graph=idx._graph,
            vectors=idx._vectors,
            dist_fn=idx._dist_fn,
        )

        returned_ids = {slot_id for _, slot_id in results}
        for deleted_id in range(10):
            assert deleted_id not in returned_ids


# ---------------------------------------------------------------------------
# Post-Filter Search and Warning System
# ---------------------------------------------------------------------------

class TestPostFilterSearch:

    def test_warning_issued_when_results_incomplete(self, db_conn):
        try:
            db_conn.execute("ALTER TABLE vectors ADD COLUMN metadata TEXT")
            db_conn.commit()
        except Exception:
            pass

        col = Collection(name="test", dimension=8, metric="euclidean",
                         m=8, ef_construction=100)
        idx = HNSWIndex(col, seed=42)
        vecs = make_random_vecs(200, 8, seed=0)

        # Only 2 vectors get the rare tag
        for i, v in enumerate(vecs):
            tag = "rare" if i < 2 else "common"
            meta = f'{{"tag": "{tag}"}}'
            db_conn.execute(
                """INSERT INTO vectors (id, collection, slot_id, deleted, created_at, metadata)
                   VALUES (?, 'test', ?, 0, datetime('now'), ?)""",
                (f"v{i}", i, meta),
            )
            db_conn.commit()
            idx.add(i, v)

        query = make_random_vecs(1, 8, seed=55)[0]
        result = search_postfilter(
            query_vector=query,
            k=10, ef=50,
            filter_dict={"tag": "rare"},
            conn=db_conn,
            collection_name="test",
            entry_point=idx.entry_point,
            max_layer=idx.max_layer,
            graph=idx._graph,
            vectors=idx._vectors,
            dist_fn=idx._dist_fn,
            overfetch_factor=3,
        )

        # Only 2 vectors match — must warn
        assert len(result.results) <= 2
        assert not result.is_complete
        assert len(result.warnings) > 0
        assert "2" in result.warnings[0].message or "Requested" in result.warnings[0].message

    def test_no_warning_when_results_complete(self, db_conn):
        try:
            db_conn.execute("ALTER TABLE vectors ADD COLUMN metadata TEXT")
            db_conn.commit()
        except Exception:
            pass

        col = Collection(name="test", dimension=8, metric="euclidean",
                         m=8, ef_construction=100)
        idx = HNSWIndex(col, seed=42)
        vecs = make_random_vecs(100, 8, seed=0)

        for i, v in enumerate(vecs):
            meta = '{"tag": "common"}'
            db_conn.execute(
                """INSERT INTO vectors (id, collection, slot_id, deleted, created_at, metadata)
                   VALUES (?, 'test', ?, 0, datetime('now'), ?)""",
                (f"v{i}", i, meta),
            )
            db_conn.commit()
            idx.add(i, v)

        query = make_random_vecs(1, 8, seed=33)[0]
        result = search_postfilter(
            query_vector=query,
            k=5, ef=50,
            filter_dict={"tag": "common"},
            conn=db_conn,
            collection_name="test",
            entry_point=idx.entry_point,
            max_layer=idx.max_layer,
            graph=idx._graph,
            vectors=idx._vectors,
            dist_fn=idx._dist_fn,
            overfetch_factor=3,
        )

        assert len(result.results) == 5
        assert result.is_complete

    def test_overfetch_below_minimum_raises(self, db_conn):
        col = Collection(name="test", dimension=8, metric="euclidean",
                         m=8, ef_construction=50)
        idx = HNSWIndex(col, seed=42)
        v = make_vec()
        idx.add(0, v)

        with pytest.raises(ValueError, match="overfetch_factor"):
            search_postfilter(
                query_vector=v,
                k=5, ef=10,
                filter_dict={},
                conn=db_conn,
                collection_name="test",
                entry_point=idx.entry_point,
                max_layer=idx.max_layer,
                graph=idx._graph,
                vectors=idx._vectors,
                dist_fn=idx._dist_fn,
                overfetch_factor=1,
            )


# ---------------------------------------------------------------------------
# Recall benchmark — selectivity sweep
# ---------------------------------------------------------------------------

class TestFilterRecallBenchmark:

    def test_prefilter_recall_above_threshold_at_50pct_selectivity(self):
        """
        At 50% selectivity, pre-filter should achieve recall@5 >= 0.80.
        Below 0.80 indicates the eligible subgraph is too sparse for
        reliable navigation — investigate M or ef_construction values.
        """
        from vektor.benchmark.datasets import load_synthetic
        from vektor.benchmark.recall import mean_recall_at_k

        # Synthetic dataset: 1000 vectors, partition 0 or 1 (50/50)
        n = 1000
        dim = 32
        k = 5
        rng = np.random.default_rng(42)
        all_vectors = rng.standard_normal((n, dim)).astype(np.float32)
        queries = rng.standard_normal((50, dim)).astype(np.float32)

        # Ground truth: brute force among partition=0 vectors only
        eligible_mask = np.arange(n) < n // 2  # first 500 are partition=0
        eligible_vectors = all_vectors[eligible_mask]
        eligible_indices = np.where(eligible_mask)[0]

        # Compute ground truth
        gt = []
        for q in queries:
            dists = np.sum((eligible_vectors - q) ** 2, axis=1)
            top_k_in_eligible = np.argsort(dists)[:k]
            gt.append(eligible_indices[top_k_in_eligible].tolist())

        # Build index
        col = Collection(name="bench", dimension=dim, metric="euclidean",
                         m=16, ef_construction=200)
        idx = HNSWIndex(col, seed=42)
        for i, v in enumerate(all_vectors):
            idx.add(i, v)

        # Build eligible frozenset (partition=0)
        eligible_ids = frozenset(range(n // 2))
        all_ids = frozenset(range(n))
        skip_entirely = all_ids - eligible_ids

        from vektor.hnsw.algorithms import knn_search
        all_returned = []
        for q in queries:
            results = knn_search(
                query_vector=q,
                k=k, ef=100,
                entry_point=idx.entry_point,
                max_layer=idx.max_layer,
                graph=idx._graph,
                vectors=idx._vectors,
                dist_fn=idx._dist_fn,
                skip_from_results=frozenset(),
                skip_entirely=skip_entirely,
            )
            all_returned.append([r[1] for r in results])

        gt_array = np.array(gt, dtype=np.int32)
        recall = mean_recall_at_k(all_returned, gt_array, k=k)

        assert recall >= 0.80, (
            f"Pre-filter recall@{k} at 50% selectivity = {recall:.4f}. "
            f"Expected >= 0.80. Check M and ef_construction values."
        )