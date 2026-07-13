"""
benchmarks/exp6_distance_speedup.py
--------------------------------------
Python vs NumPy distance speedup. Already implemented in Phase 6 —
re-run here so it's part of the same benchmark session as everything else.
"""

import csv
from pathlib import Path

from vektor.benchmark.distance_bench import run_distance_benchmark
from benchmarks.common import base_row

RESULTS_PATH = Path(__file__).parent / "results" / "exp6_distance_speedup.csv"


def run() -> None:
    results = run_distance_benchmark()

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", newline="") as f:
        fieldnames = ["git_commit", "seed", "dataset", "timestamp",
                     "metric", "numpy_us_per_call", "python_us_per_call", "speedup"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for metric_name, stats in results.items():
            row = base_row(seed=42, dataset="fixed_128d_vectors",
                          metric=metric_name,
                          numpy_us_per_call=round(stats["numpy_us_per_call"], 4),
                          python_us_per_call=round(stats["python_us_per_call"], 4),
                          speedup=round(stats["speedup"], 2))
            writer.writerow(row)

    print(f"Experiment 6 complete. Results: {RESULTS_PATH}")
    for metric_name, stats in results.items():
        print(f"  {metric_name}: {stats['speedup']:.1f}x speedup")


if __name__ == "__main__":
    run()