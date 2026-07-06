"""
vektor.hnsw.layer
------------------
Layer assignment for HNSW node insertion.

Uses the geometric distribution from Malkov & Yashunin (2018), Section 4.1.
"""

from __future__ import annotations

import math
import random


def assign_layer(M: int, rng: random.Random = None) -> int:
    """
    Draw a maximum layer index for a new node.

    Args:
        M:   The HNSW M parameter (maximum connections per layer).
             Controls the layer density via mL = 1/ln(M).
        rng: Optional random.Random instance for reproducibility in tests.
             If None, uses the module-level random state.

    Returns:
        Non-negative integer. 0 means the node only exists at the base layer.

    Formula:
        max_layer = floor(-ln(uniform(0,1)) × (1/ln(M)))
    """
    if rng is None:
        u = random.random()
    else:
        u = rng.random()

    # Guard against u=0 which would produce -inf
    u = max(u, 1e-10)
    mL = 1.0 / math.log(M)
    return int(-math.log(u) * mL)


def layer_distribution_stats(M: int, n_samples: int = 10_000,
                              seed: int = 42) -> dict:
    """
    Draw n_samples layer assignments and return distribution stats.

    Used to verify the geometric distribution is correct before construction.
    Expected: ~50% at layer 0, ~25% at layer 1, ~12.5% at layer 2, etc.
    (exact fractions depend on M).

    Returns:
        Dict mapping layer_number → count.
    """
    rng = random.Random(seed)
    counts: dict[int, int] = {}
    for _ in range(n_samples):
        layer = assign_layer(M, rng)
        counts[layer] = counts.get(layer, 0) + 1
    return dict(sorted(counts.items()))