"""
vektor.benchmark.runner
------------------------
BenchmarkRunner: assembles all harnesses into a single interface.
"""

from __future__ import annotations

from typing import Callable, Optional
import sqlite3

import numpy as np

from vektor.benchmark.datasets import DatasetBundle
from vektor.benchmark.recall import mean_recall_at_k
from vektor.benchmark.latency import measure_latency, measure_qps
from vektor.benchmark.memory import measure_peak_memory
from vektor.benchmark.distance_bench import run_distance_benchmark
from vektor.benchmark.results import write_benchmark_result


class BenchmarkRunner:
    """
    Unified benchmark harness for a Vektor search function.

    Usage:
        runner = BenchmarkRunner(search_fn=store.search, k=10)
        runner.run_recall_benchmark(dataset, experiment_name="brute_force_sift")
        runner.run_latency_benchmark(dataset, experiment_name="brute_force_latency")
    """

    def __init__(
        self,
        search_fn: Callable,
        k: int = 10,
        conn: Optional[sqlite3.Connection] = None,
    ) -> None:
        """
        Args:
            search_fn: Callable(query: np.ndarray, k: int) → list of result objects
                       Each result must have a .id attribute (integer slot ID).
            k:         Number of nearest neighbours to retrieve.
            conn:      Optional SQLite connection for logging to runs table.
        """
        self._search_fn = search_fn
        self._k = k
        self._conn = conn

    def _extract_ids(self, results) -> list[int]:
        """Extract integer slot IDs from search results."""
        return [r.id for r in results]

    def run_recall_benchmark(
        self,
        dataset: DatasetBundle,
        experiment_name: str,
        ef: Optional[int] = None,
        m: Optional[int] = None,
        ef_construction: Optional[int] = None,
    ) -> float:
        """
        Run recall@k benchmark over all query vectors.

        Returns:
            Mean recall@k as a float.
        """
        all_returned = []
        for query in dataset.queries:
            results = self._search_fn(query, self._k)
            all_returned.append(self._extract_ids(results))

        recall = mean_recall_at_k(all_returned, dataset.ground_truth, self._k)

        write_benchmark_result(
            experiment_name,
            {
                "dataset_name": dataset.name,
                "dataset_size": len(dataset.train),
                "dimension": dataset.train.shape[1],
                "metric": dataset.metric,
                "k": self._k,
                "recall_at_k": round(recall, 6),
                "m": m,
                "ef_construction": ef_construction,
                "ef": ef,
            },
            conn=self._conn,
        )
        return recall

    def run_latency_benchmark(
        self,
        dataset: DatasetBundle,
        experiment_name: str,
        n_warmup: int = 100,
    ) -> dict:
        """Run latency benchmark. Returns latency stats dict."""
        def wrapped_search(query, k):
            return self._search_fn(query, k)

        stats = measure_latency(wrapped_search, dataset.queries, self._k, n_warmup)

        write_benchmark_result(
            experiment_name,
            {
                "dataset_name": dataset.name,
                "metric": dataset.metric,
                "k": self._k,
                **stats,
            },
            conn=self._conn,
        )
        return stats

    def run_qps_benchmark(
        self,
        dataset: DatasetBundle,
        experiment_name: str,
        n_runs: int = 3,
    ) -> dict:
        """Run QPS benchmark. Returns QPS stats dict."""
        def wrapped_search(query, k):
            return self._search_fn(query, k)

        stats = measure_qps(wrapped_search, dataset.queries, self._k, n_runs)

        write_benchmark_result(
            experiment_name,
            {
                "dataset_name": dataset.name,
                "metric": dataset.metric,
                "k": self._k,
                **stats,
            },
            conn=self._conn,
        )
        return stats

    def run_memory_benchmark(
        self,
        build_fn: Callable,
        experiment_name: str,
        dataset_name: str = "",
    ) -> dict:
        """
        Measure peak memory during index construction.

        Args:
            build_fn: Callable that builds the index (called once inside measurement).
        """
        with measure_peak_memory() as mem:
            build_fn()

        write_benchmark_result(
            experiment_name,
            {
                "dataset_name": dataset_name,
                "peak_memory_mb": round(mem["peak_mb"], 2),
                "current_memory_mb": round(mem["current_mb"], 2),
                "note": "tracemalloc: undercounts NumPy C-extension allocations",
            },
            conn=self._conn,
        )
        return mem

    def run_distance_comparison(self, experiment_name: str = "distance_comparison") -> dict:
        """Run Python vs NumPy distance benchmark."""
        results = run_distance_benchmark()
        for metric_name, stats in results.items():
            write_benchmark_result(experiment_name, {"metric": metric_name, **stats},
                                   conn=self._conn)
        return results