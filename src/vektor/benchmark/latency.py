"""
vektor.benchmark.latency
------------------------
Latency and QPS measurement harnesses.

Uses time.perf_counter() throughout.
[Certain] Do not use time.time() — it has lower resolution and is not
monotonic on all platforms.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np


def measure_latency(
    search_fn: Callable,
    queries: np.ndarray,
    k: int,
    n_warmup: int = 100,
) -> dict:
    """
    Measure per-query latency over a set of queries.

    Runs n_warmup queries first and discards their times.
    Reports median, p95, p99, min, max in milliseconds.

    Args:
        search_fn: Callable accepting (query_vector, k) and returning results.
        queries:   Float32 array of shape (n_queries, dim).
        k:         Number of results to request per query.
        n_warmup:  Queries to run before timing begins. Default 100.

    Returns:
        Dict with keys: median_ms, p95_ms, p99_ms, min_ms, max_ms, n_queries.
    """
    # Warm-up — discard results entirely
    for query in queries[:n_warmup]:
        search_fn(query, k)

    # Timed window
    latencies_ms = []
    for query in queries[n_warmup:]:
        t0 = time.perf_counter()
        search_fn(query, k)
        latencies_ms.append((time.perf_counter() - t0) * 1000)

    latencies = np.array(latencies_ms)
    return {
        "median_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "n_queries": len(latencies),
    }


def measure_qps(
    search_fn: Callable,
    queries: np.ndarray,
    k: int,
    n_runs: int = 3,
) -> dict:
    """
    Measure sustained queries-per-second.

    Runs the full query set n_runs times and reports median QPS.
    [Certain] Reports median, not best run — cherry-picking best run
    is a methodologically invalid performance claim.

    Args:
        search_fn: Callable accepting (query_vector, k).
        queries:   Float32 array of shape (n_queries, dim).
        k:         Number of results per query.
        n_runs:    Number of full-set repetitions. Default 3.

    Returns:
        Dict with keys: median_qps, min_qps, max_qps, n_queries_per_run.
    """
    qps_values = []
    n = len(queries)

    for _ in range(n_runs):
        t0 = time.perf_counter()
        for query in queries:
            search_fn(query, k)
        elapsed = time.perf_counter() - t0
        qps_values.append(n / elapsed)

    return {
        "median_qps": float(np.median(qps_values)),
        "min_qps": float(np.min(qps_values)),
        "max_qps": float(np.max(qps_values)),
        "n_queries_per_run": n,
    }