"""
vektor.benchmark.distance_bench
--------------------------------
Python vs NumPy distance metric speedup benchmark.

Uses fixed, deterministic vectors from a fixed seed.
Absolute times are hardware-dependent; speedup ratio is what matters.
"""

from __future__ import annotations

import time

import numpy as np

from vektor.distance import cosine_similarity, dot_product, l2_distance

N_TRIALS = 1_000_000
DIM = 128
SEED = 42


def _python_cosine(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x ** 2 for x in a) ** 0.5
    norm_b = sum(x ** 2 for x in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def _python_dot(a: list, b: list) -> float:
    return sum(x * y for x, y in zip(a, b))


def _python_l2(a: list, b: list) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def _time_fn(fn, a, b, n: int) -> float:
    """Return wall-clock seconds for n calls to fn(a, b)."""
    t0 = time.perf_counter()
    for _ in range(n):
        fn(a, b)
    return time.perf_counter() - t0


def run_distance_benchmark() -> dict:
    """
    Run Python vs NumPy comparison for all three metrics.

    Returns:
        Dict keyed by metric name, each containing:
        numpy_us_per_call, python_us_per_call, speedup.
    """
    rng = np.random.default_rng(SEED)
    a_np = rng.standard_normal(DIM).astype(np.float32)
    b_np = rng.standard_normal(DIM).astype(np.float32)
    a_py = a_np.tolist()
    b_py = b_np.tolist()

    benchmarks = {
        "cosine": (cosine_similarity, _python_cosine),
        "dot":    (dot_product,       _python_dot),
        "l2":     (l2_distance,       _python_l2),
    }

    results = {}
    for name, (np_fn, py_fn) in benchmarks.items():
        # Warm up NumPy
        for _ in range(100):
            np_fn(a_np, b_np)

        numpy_time = _time_fn(np_fn, a_np, b_np, N_TRIALS)
        python_time = _time_fn(py_fn, a_py, b_py, N_TRIALS)

        results[name] = {
            "numpy_us_per_call":  (numpy_time / N_TRIALS) * 1e6,
            "python_us_per_call": (python_time / N_TRIALS) * 1e6,
            "speedup":            python_time / numpy_time,
            "n_trials":           N_TRIALS,
            "dimension":          DIM,
        }

    return results