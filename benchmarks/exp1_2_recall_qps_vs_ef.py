"""
benchmarks/exp1_2_recall_qps_vs_ef.py
------------------------------------------
Recall@10 (Exp 1) and QPS (Exp 2) versus ef, on FULL SIFT-128 and GloVe-100.

Builds each index ONCE, reuses it for both experiments — building twice
would risk the two experiments running against subtly different graphs.

WARNING: full-scale build (1M+ vectors) may take hours in pure Python.
Run this as a background process. Check the log database for completion,
don't watch it interactively.
"""

import csv
import time
from pathlib import Path

import numpy as np

from vektor.hnsw.index import HNSWIndex
from vektor.collection import Collection
from vektor.benchmark.datasets import load_sift128, load_glove100
from vektor.benchmark.recall import mean_recall_at_k
from benchmarks.common import base_row

RESULTS_DIR = Path(__file__).parent / "results"
EF_VALUES = [10, 20, 50, 100, 200, 500]
M = 16
EF_CONSTRUCTION = 200
K = 10
SEED = 42
N_QPS_WARMUP = 1_000
N_QPS_TIMED = 1_000
N_QPS_RUNS = 3


def build_full_index(train_vectors: np.ndarray, metric: str) -> HNSWIndex:
    col = Collection(name="bench", dimension=train_vectors.shape[1],
                     metric=metric, m=M, ef_construction=EF_CONSTRUCTION)
    index = HNSWIndex(col, seed=SEED)

    print(f"  Building index on {len(train_vectors):,} vectors "
          f"(metric={metric})... this may take a while.")
    t0 = time.perf_counter()
    for i, v in enumerate(train_vectors):
        index.add(i, v)
        if (i + 1) % 100_000 == 0:
            print(f"    {i+1:,}/{len(train_vectors):,} inserted "
                  f"({time.perf_counter()-t0:.0f}s elapsed)")
    print(f"  Build complete: {time.perf_counter()-t0:.0f}s total")
    return index


def measure_qps(index: HNSWIndex, queries: np.ndarray, ef: int) -> float:
    """Median QPS across N_QPS_RUNS full passes, after warm-up."""
    for q in queries[:min(N_QPS_WARMUP, len(queries))]:
        index.search(q, k=K, ef=ef)

    qps_values = []
    timed_queries = queries[:N_QPS_TIMED] if len(queries) >= N_QPS_TIMED else queries
    for _ in range(N_QPS_RUNS):
        t0 = time.perf_counter()
        for q in timed_queries:
            index.search(q, k=K, ef=ef)
        elapsed = time.perf_counter() - t0
        qps_values.append(len(timed_queries) / elapsed)

    return float(np.median(qps_values))


def run_for_dataset(dataset_name: str, load_fn) -> None:
    print(f"\n=== {dataset_name} ===")
    ds = load_fn()
    index = build_full_index(ds.train, ds.metric)

    recall_rows = []
    qps_rows = []

    for ef in EF_VALUES:
        print(f"  ef={ef}: computing recall...")
        all_returned = []
        for query in ds.queries:
            results = index.search(query, k=K, ef=ef)
            all_returned.append([r[1] for r in results])
        recall = mean_recall_at_k(all_returned, ds.ground_truth, k=K)

        print(f"  ef={ef}: measuring QPS...")
        qps = measure_qps(index, ds.queries, ef)

        recall_rows.append(base_row(
            seed=SEED, dataset=dataset_name, M=M,
            ef_construction=EF_CONSTRUCTION, ef=ef, k=K,
            n_vectors=len(ds.train), n_queries=len(ds.queries),
            recall_at_10=round(recall, 4),
        ))
        qps_rows.append(base_row(
            seed=SEED, dataset=dataset_name, M=M,
            ef_construction=EF_CONSTRUCTION, ef=ef, k=K,
            median_qps=round(qps, 2),
        ))
        print(f"  ef={ef}: recall@10={recall:.4f}, QPS={qps:.1f}")

    return recall_rows, qps_rows


def run() -> None:
    all_recall_rows = []
    all_qps_rows = []

    for name, loader in [("sift-128-euclidean", load_sift128),
                         ("glove-100-angular", load_glove100)]:
        recall_rows, qps_rows = run_for_dataset(name, loader)
        all_recall_rows.extend(recall_rows)
        all_qps_rows.extend(qps_rows)

    _write_csv(RESULTS_DIR / "exp1_recall_vs_ef.csv", all_recall_rows)
    _write_csv(RESULTS_DIR / "exp2_qps_vs_ef.csv", all_qps_rows)
    print("\nExperiments 1 & 2 complete.")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    run()