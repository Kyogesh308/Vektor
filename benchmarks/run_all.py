"""
benchmarks/run_all.py
------------------------
Runs all six experiments in sequence with fixed, documented parameters.

Order matters:
  6 first  — fast, no dataset dependency, environment sanity check
  3+4      — moderate cost, synthetic data
  5        — expensive, SIFT-128 subset, 4 full builds
  1+2      — most expensive, full SIFT-128 + GloVe-100 builds

Run this as: python -m benchmarks.run_all
Or in background: nohup python -m benchmarks.run_all > run_all.log 2>&1 &
"""

import sys
import time

from benchmarks import exp6_distance_speedup
from benchmarks import exp3_4_build_and_memory
from benchmarks import exp5_recall_vs_m
from benchmarks import exp1_2_recall_qps_vs_ef


def preflight_check() -> bool:
    """Confirm brute-force recall == 1.0 before generating any HNSW numbers."""
    import subprocess
    print("Pre-flight: verifying brute-force baseline recall == 1.0 ...")
    result = subprocess.run(
        ["pytest", "tests/test_phase6_benchmark.py::TestBaselineValidation", "-v"],
        capture_output=True, text=True,
    )
    passed = result.returncode == 0
    print(result.stdout[-500:])
    if not passed:
        print("PRE-FLIGHT FAILED. Fix the baseline before running any experiment.")
    return passed


def main() -> None:
    if not preflight_check():
        sys.exit(1)

    experiments = [
        ("Experiment 6 (distance speedup)", exp6_distance_speedup.run),
        ("Experiments 3 & 4 (build time + memory)", exp3_4_build_and_memory.run),
        ("Experiment 5 (recall vs M)", exp5_recall_vs_m.run),
        ("Experiments 1 & 2 (recall + QPS vs ef, full datasets)",
         exp1_2_recall_qps_vs_ef.run),
    ]

    for label, fn in experiments:
        print(f"\n{'='*70}\n{label}\n{'='*70}")
        t0 = time.perf_counter()
        fn()
        print(f"[{label} finished in {time.perf_counter()-t0:.0f}s]")

    print("\nAll experiments complete. Run monotonicity checks next:")
    print("  python benchmarks/check_monotonicity.py")


if __name__ == "__main__":
    main()