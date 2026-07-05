"""
Phase 6 — Benchmark Infrastructure Tests

The most important test in this file:
    TestBaselineValidation::test_brute_force_recall_is_1_on_synthetic

Brute-force must return recall@10 == 1.0 on synthetic data.
If it does not, the recall calculator or the ground truth computation
is broken — do not proceed to Phase 7 until this reads 1.0.
"""

import numpy as np
import pytest

from vektor.benchmark.datasets import load_synthetic, DatasetBundle
from vektor.benchmark.recall import recall_at_k, mean_recall_at_k
from vektor.benchmark.latency import measure_latency, measure_qps
from vektor.benchmark.memory import measure_peak_memory
from vektor.benchmark.distance_bench import run_distance_benchmark
from vektor.benchmark.results import write_benchmark_result


# ---------------------------------------------------------------------------
# recall.py
# ---------------------------------------------------------------------------

class TestRecallAtK:

    def test_perfect_recall(self):
        assert recall_at_k([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], k=5) == 1.0

    def test_zero_recall(self):
        assert recall_at_k([6, 7, 8, 9, 10], [1, 2, 3, 4, 5], k=5) == 0.0

    def test_partial_recall(self):
        # 4 of 5 correct → 0.8
        assert recall_at_k([1, 2, 3, 4, 99], [1, 2, 3, 4, 5], k=5) == pytest.approx(0.8)

    def test_k_equals_one(self):
        assert recall_at_k([1], [1], k=1) == 1.0
        assert recall_at_k([2], [1], k=1) == 0.0

    def test_order_does_not_matter(self):
        # recall@k is set-based, not order-based
        assert recall_at_k([5, 4, 3, 2, 1], [1, 2, 3, 4, 5], k=5) == 1.0

    def test_mean_recall_across_queries(self):
        all_returned = [[1, 2, 3], [4, 5, 6]]
        gt = np.array([[1, 2, 3], [4, 5, 99]])
        # Query 1: recall=1.0, Query 2: recall=2/3
        expected = (1.0 + 2/3) / 2
        assert mean_recall_at_k(all_returned, gt, k=3) == pytest.approx(expected)

    def test_mean_recall_all_perfect(self):
        all_returned = [[1, 2], [3, 4]]
        gt = np.array([[1, 2], [3, 4]])
        assert mean_recall_at_k(all_returned, gt, k=2) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# datasets.py — synthetic loader
# ---------------------------------------------------------------------------

class TestSyntheticDataset:

    def test_shapes_are_correct(self):
        ds = load_synthetic(n_train=100, n_queries=10, dim=32, k=5, seed=0)
        assert ds.train.shape == (100, 32)
        assert ds.queries.shape == (10, 32)
        assert ds.ground_truth.shape == (10, 5)

    def test_dtypes_are_float32_and_int32(self):
        ds = load_synthetic(n_train=100, n_queries=10, dim=32, k=5)
        assert ds.train.dtype == np.float32
        assert ds.queries.dtype == np.float32
        assert ds.ground_truth.dtype == np.int32

    def test_fixed_seed_is_reproducible(self):
        ds1 = load_synthetic(n_train=100, n_queries=10, dim=32, k=5, seed=42)
        ds2 = load_synthetic(n_train=100, n_queries=10, dim=32, k=5, seed=42)
        np.testing.assert_array_equal(ds1.train, ds2.train)

    def test_different_seeds_differ(self):
        ds1 = load_synthetic(n_train=100, n_queries=10, dim=32, k=5, seed=1)
        ds2 = load_synthetic(n_train=100, n_queries=10, dim=32, k=5, seed=2)
        assert not np.array_equal(ds1.train, ds2.train)

    def test_ground_truth_indices_in_valid_range(self):
        ds = load_synthetic(n_train=200, n_queries=20, dim=16, k=10)
        assert np.all(ds.ground_truth >= 0)
        assert np.all(ds.ground_truth < 200)


# ---------------------------------------------------------------------------
# latency.py
# ---------------------------------------------------------------------------

class TestLatencyHarness:

    def _fast_search(self, query, k):
        """Trivially fast search — returns fixed list instantly."""
        return list(range(k))

    def test_returns_required_keys(self):
        queries = np.random.rand(200, 4).astype(np.float32)
        result = measure_latency(self._fast_search, queries, k=5, n_warmup=10)
        assert "median_ms" in result
        assert "p95_ms" in result
        assert "p99_ms" in result
        assert "min_ms" in result
        assert "max_ms" in result

    def test_median_is_positive(self):
        queries = np.random.rand(200, 4).astype(np.float32)
        result = measure_latency(self._fast_search, queries, k=5, n_warmup=10)
        assert result["median_ms"] > 0

    def test_p99_geq_median(self):
        queries = np.random.rand(200, 4).astype(np.float32)
        result = measure_latency(self._fast_search, queries, k=5, n_warmup=10)
        assert result["p99_ms"] >= result["median_ms"]

    def test_qps_is_positive(self):
        queries = np.random.rand(100, 4).astype(np.float32)
        result = measure_qps(self._fast_search, queries, k=5, n_runs=2)
        assert result["median_qps"] > 0


# ---------------------------------------------------------------------------
# memory.py
# ---------------------------------------------------------------------------

class TestMemoryHarness:

    def test_peak_mb_is_positive_after_allocation(self):
        with measure_peak_memory() as mem:
            _ = np.zeros((1000, 1000), dtype=np.float32)
        assert mem["peak_mb"] > 0

    def test_result_dict_populated_after_block(self):
        with measure_peak_memory() as mem:
            pass
        assert "peak_mb" in mem
        assert "current_mb" in mem


# ---------------------------------------------------------------------------
# distance_bench.py
# ---------------------------------------------------------------------------

class TestDistanceBenchmark:

    def test_all_three_metrics_present(self):
        results = run_distance_benchmark()
        assert "cosine" in results
        assert "dot" in results
        assert "l2" in results

    def test_numpy_faster_than_python(self):
        results = run_distance_benchmark()
        for metric, stats in results.items():
            assert stats["speedup"] > 1.0, (
                f"{metric}: NumPy should be faster than pure Python. "
                f"Got speedup={stats['speedup']:.2f}x. "
                f"Check that pure-Python functions aren't accidentally calling NumPy."
            )


# ---------------------------------------------------------------------------
# results.py
# ---------------------------------------------------------------------------

class TestResultsWriter:

    def test_csv_created_on_first_write(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "vektor.benchmark.results.RESULTS_DIR", tmp_path
        )
        path = write_benchmark_result("test_exp", {"recall_at_k": 0.95, "k": 10})
        assert path.exists()

    def test_csv_row_is_readable(self, tmp_path, monkeypatch):
        import csv
        monkeypatch.setattr(
            "vektor.benchmark.results.RESULTS_DIR", tmp_path
        )
        write_benchmark_result("test_exp", {"recall_at_k": 0.95, "k": 10})
        with open(tmp_path / "test_exp.csv") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert float(rows[0]["recall_at_k"]) == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# Baseline Validation — the most important test in Phase 6
# ---------------------------------------------------------------------------

class TestBaselineValidation:

    def test_brute_force_recall_is_1_on_synthetic(self):
        """
        Brute-force search must return recall@10 == 1.0 on synthetic data.

        This test validates the entire benchmark framework:
        - Synthetic dataset ground truth is correct
        - Recall calculator is correct
        - ID mapping between search results and ground truth is correct

        If this fails, do not proceed to Phase 7. The framework is broken.
        """
        from vektor.collection import Collection
        from vektor.storage import VectorStore

        K = 10
        ds = load_synthetic(n_train=500, n_queries=50, dim=32, k=K,
                            metric="euclidean", seed=42)

        # Build a VectorStore and insert all training vectors
        col = Collection(name="test", dimension=32, metric="euclidean")
        store = VectorStore(col)

        slot_to_gt_index = {}
        for gt_index, vector in enumerate(ds.train):
            vec_id = str(gt_index)
            store.insert(vec_id, vector)
            slot_to_gt_index[vec_id] = gt_index

        # Run search for each query and collect returned gt indices
        all_returned = []
        for query in ds.queries:
            results = store.search(query, k=K)
            returned_gt_indices = [slot_to_gt_index[r.id] for r in results]
            all_returned.append(returned_gt_indices)

        recall = mean_recall_at_k(all_returned, ds.ground_truth, k=K)

        assert recall == pytest.approx(1.0, abs=1e-6), (
            f"Brute-force recall@{K} = {recall:.6f} — expected exactly 1.0.\n"
            f"This indicates a bug in the recall calculator, ground truth "
            f"computation, or ID mapping. Do not proceed to Phase 7."
        )