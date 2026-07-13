"""
benchmarks/exp3_4_build_and_memory.py
-----------------------------------------
Build time (Experiment 3) and memory usage (Experiment 4) versus dataset size.
Measured together in the same construction pass — building twice would be
wasteful and risks subtle graph differences between the two experiments.
"""

import csv
import time
from pathlib import Path

import numpy as np

from vektor.collection import Collection
from vektor.hnsw.index import HNSWIndex
from vektor.benchmark.memory import measure_peak_memory
from benchmarks.common import base_row

RESULTS_DIR = Path(__file__).parent / "results"
DATASET_SIZES = [1_000, 5_000, 10_000, 50_000, 100_000]
DIM = 128
M = 16
EF_CONSTRUCTION = 200
SEED = 42
N_TIMING_RUNS = 3


def build_index(vectors: np.ndarray) -> tuple[HNSWIndex, float]:
    """Build an index from scratch, return (index, build_time_seconds)."""
    col = Collection(name="bench", dimension=DIM, metric="euclidean",
                     m=M, ef_construction=EF_CONSTRUCTION)
    index = HNSWIndex(col, seed=SEED)

    t0 = time.perf_counter()
    for i, v in enumerate(vectors):
        index.add(i, v)
    elapsed = time.perf_counter() - t0

    return index, elapsed


def run() -> None:
    build_time_rows = []
    memory_rows = []

    for size in DATASET_SIZES:
        rng = np.random.default_rng(SEED)
        vectors = rng.standard_normal((size, DIM)).astype(np.float32)

        # Build time: 3 runs, report median + spread
        build_times = []
        for run_idx in range(N_TIMING_RUNS):
            _, elapsed = build_index(vectors)
            build_times.append(elapsed)
            print(f"  size={size} run={run_idx+1}/{N_TIMING_RUNS}: {elapsed:.2f}s")

        median_time = float(np.median(build_times))
        min_time = float(np.min(build_times))
        max_time = float(np.max(build_times))

        build_time_rows.append(base_row(
            seed=SEED, dataset=f"synthetic_{DIM}d",
            n_vectors=size, M=M, ef_construction=EF_CONSTRUCTION,
            median_build_time_s=round(median_time, 3),
            min_build_time_s=round(min_time, 3),
            max_build_time_s=round(max_time, 3),
        ))

        # Memory: single measurement (deterministic given fixed seed)
        with measure_peak_memory() as baseline_mem:
            vectors_baseline = vectors.copy()  # force allocation baseline

        with measure_peak_memory() as full_mem:
            index, _ = build_index(vectors)

        vector_bytes_measured = baseline_mem["peak_mb"] * 1_000_000
        graph_bytes_measured = max(0, (full_mem["peak_mb"] - baseline_mem["peak_mb"]) * 1_000_000)

        # Cross-check against Phase 11 formula
        import math
        estimated_layers = 1 + (math.log(size) / math.log(M)) if size > 1 else 1
        formula_graph_bytes = size * (M * 2) * 8 * estimated_layers
        overhead_ratio = graph_bytes_measured / formula_graph_bytes if formula_graph_bytes > 0 else 0

        memory_rows.append(base_row(
            seed=SEED, dataset=f"synthetic_{DIM}d",
            n_vectors=size, M=M,
            measured_graph_mb=round(graph_bytes_measured / 1_000_000, 2),
            formula_estimate_mb=round(formula_graph_bytes / 1_000_000, 2),
            overhead_ratio=round(overhead_ratio, 2),
        ))

    _write_csv(RESULTS_DIR / "exp3_build_time_vs_size.csv", build_time_rows)
    _write_csv(RESULTS_DIR / "exp4_memory_vs_size.csv", memory_rows)

    print("Experiments 3 & 4 complete.")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    run()