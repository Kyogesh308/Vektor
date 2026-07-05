"""
vektor.benchmark.memory
------------------------
Memory measurement using tracemalloc.

Limitation: tracemalloc tracks only Python allocator memory.
NumPy arrays allocated in C extensions may be undercounted.
This limitation must be documented in benchmark reports.
"""

from __future__ import annotations

import tracemalloc
from contextlib import contextmanager
from typing import Generator


@contextmanager
def measure_peak_memory() -> Generator[dict, None, None]:
    """
    Context manager that records peak memory delta during the wrapped block.

    Usage:
        with measure_peak_memory() as mem:
            build_index(vectors)
        print(mem["peak_mb"])

    The result dict is populated after the block exits.
    Keys: peak_mb (float), current_mb (float).

    Note: undercounts NumPy C-extension allocations. Document this caveat
    whenever reporting these numbers.
    """
    result = {}
    tracemalloc.start()
    try:
        yield result
    finally:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        result["peak_mb"] = peak / 1_000_000
        result["current_mb"] = current / 1_000_000