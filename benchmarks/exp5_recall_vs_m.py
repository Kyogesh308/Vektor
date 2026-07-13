"""
benchmarks/exp5_recall_vs_m.py
---------------------------------
Recall@10 versus M on a SIFT-128 subset. Four separate graphs, one per M.
"""

import csv
import time
from pathlib import Path

from vektor.hnsw.index import HNSWIndex
from vektor.collection import Collection
from vektor.benchmark.datasets import load_sift128
from vektor.benchmark.recall import mean_recall_at_k
from vektor.benchmark.memory import measure_peak_memory
from benchmarks.common import base_row

RESULTS_PATH = Path(__file__).parent / "results" / "exp5_recall_vs_m.csv"
M_VALUES = [4, 8, 16, 32]
EF_CONSTRUCTION = 200
EF_SEARCH = 100
K = 10
SEED = 42
SUBSET_SIZE = 100_000
QUERY_SUBSET = 1_000


def run() -> None:
    ds = load_sift128(max_train=SUBSET_SIZE, max_queries=QUERY_SUBSET)
    rows = []

    for M in M_VALUES:
        col = Collection(name="bench", dimension=128, metric="euclidean",
                         m=M, ef_construction=EF_CONSTRUCTION)
        index = HNSWIndex(col, seed=SEED)

        t0 = time.perf_counter()
        with measure_peak_memory() as mem:
            for i, v in enumerate(ds.train):
                index.add(i, v)
        build_time = time.perf_counter() - t0

        all_returned = []
        for query in ds.queries:
            results = index.search(query, k=K, ef=EF_SEARCH)
            all_returned.append([r[1] for r in results])

        recall = mean_recall_at_k(all_returned, ds.ground_truth, k=K)

        rows.append(base_row(
            seed=SEED, dataset="sift-128-euclidean",
            M=M, ef_construction=EF_CONSTRUCTION, ef=EF_SEARCH, k=K,
            n_vectors=SUBSET_SIZE, n_queries=QUERY_SUBSET,
            recall_at_10=round(recall, 4),
            build_time_s=round(build_time, 2),
            peak_memory_mb=round(mem["peak_mb"], 2),
        ))
        print(f"  M={M}: recall@10={recall:.4f}, build_time={build_time:.1f}s, "
              f"memory={mem['peak_mb']:.1f}MB")

    with open(RESULTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Experiment 5 complete. Results: {RESULTS_PATH}")


if __name__ == "__main__":
    run()