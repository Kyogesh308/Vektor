"""
Phase 2 Benchmark — NumPy vs Pure Python Distance Metrics

Run with:
    python benchmarks/bench_phase2_distance.py

This script is NOT part of the test suite. It produces a human-readable
performance comparison for documentation and research purposes.
"""

import time
import numpy as np
from vektor.distance import cosine_similarity, dot_product, l2_distance

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DIM = 1536          # OpenAI text-embedding-ada-002 dimension
N_TRIALS = 10_000   # Number of distance calculations per metric

# ---------------------------------------------------------------------------
# Pure Python implementations (for comparison only — not shipped code)
# ---------------------------------------------------------------------------

def cosine_python(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x ** 2 for x in a) ** 0.5
    norm_b = sum(x ** 2 for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def dot_python(a: list, b: list) -> float:
    return sum(x * y for x, y in zip(a, b))


def l2_python(a: list, b: list) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def benchmark(label: str, numpy_fn, python_fn, a_np, b_np, a_py, b_py):
    # Warm up numpy (first call includes JIT-like setup overhead)
    for _ in range(10):
        numpy_fn(a_np, b_np)

    # NumPy timing
    start = time.perf_counter()
    for _ in range(N_TRIALS):
        numpy_fn(a_np, b_np)
    numpy_time = time.perf_counter() - start

    # Pure Python timing
    start = time.perf_counter()
    for _ in range(N_TRIALS):
        python_fn(a_py, b_py)
    python_time = time.perf_counter() - start

    speedup = python_time / numpy_time

    print(f"\n{'─' * 50}")
    print(f"  Metric:      {label}")
    print(f"  Dimension:   {DIM}")
    print(f"  Trials:      {N_TRIALS:,}")
    print(f"  NumPy time:  {numpy_time * 1000:.2f} ms total  |  {numpy_time / N_TRIALS * 1e6:.2f} µs per call")
    print(f"  Python time: {python_time * 1000:.2f} ms total  |  {python_time / N_TRIALS * 1e6:.2f} µs per call")
    print(f"  Speedup:     {speedup:.1f}x  ← NumPy is faster")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    a_np = rng.standard_normal(DIM).astype(np.float32)
    b_np = rng.standard_normal(DIM).astype(np.float32)
    a_py = a_np.tolist()
    b_py = b_np.tolist()

    print(f"\nVektor Phase 2 Benchmark — NumPy vs Pure Python")
    print(f"Dimension: {DIM} | Trials: {N_TRIALS:,}")

    benchmark("Cosine Similarity", cosine_similarity, cosine_python, a_np, b_np, a_py, b_py)
    benchmark("Dot Product",       dot_product,       dot_python,    a_np, b_np, a_py, b_py)
    benchmark("L2 Distance",       l2_distance,       l2_python,     a_np, b_np, a_py, b_py)

    print(f"\n{'─' * 50}")
    print(f"\nRecord these numbers in docs/benchmarks.md for future reference.")