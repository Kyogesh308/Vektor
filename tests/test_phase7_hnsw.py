"""
Phase 7 — HNSW Tests

Test order mirrors implementation order.
Do not run recall tests before graph integrity tests pass.
A graph that fails integrity will produce undebuggable recall failures.

Fixed seed: 42 throughout. All recall thresholds are verified against
the Phase 6 brute-force baseline (recall@10 == 1.0 on the same dataset).
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from vektor.collection import Collection
from vektor.hnsw.index import HNSWIndex
from vektor.hnsw.layer import assign_layer, layer_distribution_stats
from vektor.hnsw.algorithms import (
    search_layer, select_neighbors_simple,
    select_neighbors_heuristic,
)
from vektor.hnsw.exceptions import EmptyIndexError, InvalidEFError
from vektor.benchmark.datasets import load_synthetic
from vektor.benchmark.recall import mean_recall_at_k
from vektor.collection import Collection
from vektor.storage import VectorStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cosine_collection():
    return Collection(name="test", dimension=32, metric="cosine", m=8,
                      ef_construction=100)

@pytest.fixture
def euclidean_collection():
    return Collection(name="test", dimension=32, metric="euclidean", m=8,
                      ef_construction=100)

@pytest.fixture
def cosine_index(cosine_collection):
    return HNSWIndex(cosine_collection, seed=42)

@pytest.fixture
def euclidean_index(euclidean_collection):
    return HNSWIndex(euclidean_collection, seed=42)

def make_vectors(n: int, dim: int, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(dim).astype(np.float32) for _ in range(n)]


# ---------------------------------------------------------------------------
# Layer assignment
# ---------------------------------------------------------------------------

class TestLayerAssignment:

    def test_returns_non_negative_integer(self):
        for _ in range(100):
            layer = assign_layer(M=16)
            assert isinstance(layer, int)
            assert layer >= 0

    def test_layer_zero_is_most_common(self):
        stats = layer_distribution_stats(M=16, n_samples=10_000, seed=42)
        assert stats[0] > 9_000, (
            f"Layer 0 should have ~93.75% of nodes for M=16. Got {stats[0]}/10000."
        )

    def test_layer_ratio_matches_geometric_distribution(self):
        # For M=16: fraction at layer l ≈ (1-1/M)^l × (1/M)
        # In practice: layer 0 has ~9375, layer 1 ~585, etc.
        stats = layer_distribution_stats(M=16, n_samples=100_000, seed=42)
        layer0 = stats.get(0, 0)
        layer1 = stats.get(1, 0)
        if layer0 > 0 and layer1 > 0:
            ratio = layer0 / layer1
            # Should be approximately M (16) — allow 30% tolerance
            assert 10 < ratio < 22, (
                f"Layer 0/1 ratio should be ~{16}. Got {ratio:.1f}."
            )

    def test_fixed_rng_is_reproducible(self):
        rng = random.Random(42)
        results_1 = [assign_layer(16, rng) for _ in range(100)]
        rng2 = random.Random(42)
        results_2 = [assign_layer(16, rng2) for _ in range(100)]
        assert results_1 == results_2


# ---------------------------------------------------------------------------
# Algorithm 3 — SELECT-NEIGHBORS-SIMPLE
# ---------------------------------------------------------------------------

class TestSelectNeighborsSimple:

    def test_returns_m_closest(self):
        candidates = [(1.0, 1), (2.0, 2), (3.0, 3), (4.0, 4), (5.0, 5)]
        result = select_neighbors_simple(candidates, M=3)
        assert result == [1, 2, 3]

    def test_returns_all_when_fewer_than_m(self):
        candidates = [(1.0, 1), (2.0, 2)]
        result = select_neighbors_simple(candidates, M=10)
        assert set(result) == {1, 2}

    def test_empty_candidates_returns_empty(self):
        assert select_neighbors_simple([], M=5) == []


# ---------------------------------------------------------------------------
# Graph construction — integrity tests (run before recall tests)
# ---------------------------------------------------------------------------

class TestGraphIntegrity:

    def test_single_insertion_creates_entry_point(self, euclidean_index):
        v = np.array([1.0] * 32, dtype=np.float32)
        euclidean_index.add(0, v)
        assert euclidean_index.entry_point == 0

    def test_degree_never_exceeds_mmax(self, euclidean_index):
        vectors = make_vectors(200, 32, seed=0)
        for i, v in enumerate(vectors):
            euclidean_index.add(i, v)

        report = euclidean_index.check_integrity()
        assert report["errors"] == [], (
            f"Degree violations found:\n" + "\n".join(report["errors"])
        )

    def test_all_edges_are_bidirectional(self, euclidean_index):
        vectors = make_vectors(200, 32, seed=1)
        for i, v in enumerate(vectors):
            euclidean_index.add(i, v)

        report = euclidean_index.check_integrity()
        bidirectional_errors = [e for e in report["errors"]
                                 if "bidirectional" in e or "reverse" in e]
        assert bidirectional_errors == [], (
            "Non-bidirectional edges found:\n" + "\n".join(bidirectional_errors)
        )

    def test_entry_point_has_highest_layer(self, euclidean_index):
        vectors = make_vectors(500, 32, seed=2)
        for i, v in enumerate(vectors):
            euclidean_index.add(i, v)

        report = euclidean_index.check_integrity()
        ep_errors = [e for e in report["errors"] if "Entry point" in e]
        assert ep_errors == [], "\n".join(ep_errors)

    def test_all_referenced_neighbours_exist(self, euclidean_index):
        vectors = make_vectors(200, 32, seed=3)
        for i, v in enumerate(vectors):
            euclidean_index.add(i, v)

        report = euclidean_index.check_integrity()
        existence_errors = [e for e in report["errors"] if "non-existent" in e]
        assert existence_errors == []

    def test_full_integrity_passes_on_200_nodes(self, euclidean_index):
        vectors = make_vectors(200, 32, seed=4)
        for i, v in enumerate(vectors):
            euclidean_index.add(i, v)

        report = euclidean_index.check_integrity()
        assert report["errors"] == []


# ---------------------------------------------------------------------------
# Search — edge cases
# ---------------------------------------------------------------------------

class TestSearchEdgeCases:

    def test_empty_index_raises(self, euclidean_index):
        query = np.zeros(32, dtype=np.float32)
        with pytest.raises(EmptyIndexError):
            euclidean_index.search(query, k=5, ef=10)

    def test_ef_less_than_k_raises(self, euclidean_index):
        v = np.array([1.0] * 32, dtype=np.float32)
        euclidean_index.add(0, v)
        with pytest.raises(InvalidEFError):
            euclidean_index.search(v, k=10, ef=5)

    def test_k_equals_1_returns_nearest(self, euclidean_index):
        # Insert two vectors; query should return the closer one
        v1 = np.array([1.0] + [0.0] * 31, dtype=np.float32)
        v2 = np.array([10.0] + [0.0] * 31, dtype=np.float32)
        query = np.array([1.1] + [0.0] * 31, dtype=np.float32)
        euclidean_index.add(0, v1)
        euclidean_index.add(1, v2)
        results = euclidean_index.search(query, k=1, ef=10)
        assert len(results) == 1
        assert results[0][1] == 0  # slot 0 (v1) is closer

    def test_large_ef_terminates(self, euclidean_index):
        vectors = make_vectors(100, 32, seed=5)
        for i, v in enumerate(vectors):
            euclidean_index.add(i, v)
        query = np.zeros(32, dtype=np.float32)
        # ef much larger than dataset size — must not loop infinitely
        results = euclidean_index.search(query, k=10, ef=10_000)
        assert len(results) <= 10

    def test_skip_ids_excluded_from_results(self, euclidean_index):
        vectors = make_vectors(50, 32, seed=6)
        for i, v in enumerate(vectors):
            euclidean_index.add(i, v)
        query = vectors[0]
        all_results = euclidean_index.search(query, k=5, ef=20)
        top_id = all_results[0][1]

        # Skip the top result and confirm it's absent
        results_with_skip = euclidean_index.search(query, k=5, ef=20,
                                                    skip_ids={top_id})
        returned_ids = [r[1] for r in results_with_skip]
        assert top_id not in returned_ids


# ---------------------------------------------------------------------------
# Recall validation — the primary correctness test
# ---------------------------------------------------------------------------

class TestRecallValidation:

    def _build_index_and_store(self, collection, n_train=2000, dim=32, seed=42):
        """Helper: build HNSWIndex + VectorStore with the same vectors."""
        ds = load_synthetic(n_train=n_train, n_queries=100, dim=dim,
                            k=10, metric=collection.metric, seed=seed)
        index = HNSWIndex(collection, seed=seed)
        store = VectorStore(collection)

        for i, vector in enumerate(ds.train):
            index.add(i, vector)
            store.insert(str(i), vector)

        return index, store, ds

    def test_recall_at_10_euclidean_exceeds_threshold(self):
        col = Collection(name="test", dimension=32, metric="euclidean",
                         m=16, ef_construction=200)
        index, store, ds = self._build_index_and_store(col, n_train=2000)

        all_returned = []
        for query in ds.queries:
            results = index.search(query, k=10, ef=100)
            all_returned.append([r[1] for r in results])

        recall = mean_recall_at_k(all_returned, ds.ground_truth, k=10)
        assert recall >= 0.90, (
            f"HNSW recall@10 = {recall:.4f} — below 0.90 threshold.\n"
            f"Check graph integrity first, then bidirectionality, "
            f"then heap ordering in search_layer."
        )

    def test_recall_at_10_cosine_exceeds_threshold(self):
        col = Collection(name="test", dimension=32, metric="cosine",
                         m=16, ef_construction=200)
        index, store, ds = self._build_index_and_store(col, n_train=2000,
                                                        seed=43)

        all_returned = []
        for query in ds.queries:
            results = index.search(query, k=10, ef=100)
            all_returned.append([r[1] for r in results])

        recall = mean_recall_at_k(all_returned, ds.ground_truth, k=10)
        assert recall >= 0.90, (
            f"HNSW cosine recall@10 = {recall:.4f} — below threshold."
        )

    def test_recall_monotonically_increases_with_ef(self):
        """
        The recall-vs-ef curve must be monotonically non-decreasing.
        If recall drops between two ef values, SEARCH-LAYER has a
        termination or heap bug.
        """
        col = Collection(name="test", dimension=32, metric="euclidean",
                         m=16, ef_construction=200)
        ds = load_synthetic(n_train=2000, n_queries=50, dim=32,
                            k=10, metric="euclidean", seed=42)
        index = HNSWIndex(col, seed=42)
        for i, v in enumerate(ds.train):
            index.add(i, v)

        ef_values = [10, 20, 50, 100, 200, 500]
        recalls = []
        for ef in ef_values:
            all_returned = []
            for query in ds.queries:
                results = index.search(query, k=10, ef=ef)
                all_returned.append([r[1] for r in results])
            recalls.append(mean_recall_at_k(all_returned, ds.ground_truth, k=10))

        for i in range(1, len(recalls)):
            assert recalls[i] >= recalls[i-1] - 0.01, (
                f"Recall dropped from ef={ef_values[i-1]} ({recalls[i-1]:.4f}) "
                f"to ef={ef_values[i]} ({recalls[i]:.4f}). "
                f"This is a bug in SEARCH-LAYER termination or heap management."
            )

    def test_hnsw_faster_than_brute_force_at_scale(self):
        """
        HNSW must be faster than brute-force at 2000 vectors.
        This is the point of the index.
        """
        import time
        col = Collection(name="test", dimension=32, metric="euclidean",
                         m=16, ef_construction=200)
        ds = load_synthetic(n_train=2000, n_queries=50, dim=32,
                            k=10, metric="euclidean", seed=42)

        index = HNSWIndex(col, seed=42)
        store = VectorStore(col)
        for i, v in enumerate(ds.train):
            index.add(i, v)
            store.insert(str(i), v)

        # Warm up
        for q in ds.queries[:10]:
            index.search(q, k=10, ef=50)
            store.search(q, k=10)

        # HNSW
        t0 = time.perf_counter()
        for q in ds.queries:
            index.search(q, k=10, ef=50)
        hnsw_time = time.perf_counter() - t0

        # Brute force
        t0 = time.perf_counter()
        for q in ds.queries:
            store.search(q, k=10)
        bf_time = time.perf_counter() - t0

        assert hnsw_time < bf_time, (
            f"HNSW ({hnsw_time*1000:.1f}ms) was slower than brute force "
            f"({bf_time*1000:.1f}ms) on 2000 vectors. "
            f"This suggests a graph construction bug."
        )