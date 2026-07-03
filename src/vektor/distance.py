from __future__ import annotations
"""Distance metric implementations. Implementation begins in Phase 2."""
"""
vektor.distance
---------------
Distance and similarity metrics for vector search.

Design contracts:
- All functions accept NumPy float32 arrays of identical shape.
- Vectors are assumed to be pre-validated by vektor.validator.
  This layer does NOT re-check dimension, NaN, or infinity.
- cosine_similarity and dot_product return higher values for more similar vectors.
- l2_distance returns lower values for more similar vectors.
- METRIC_HIGHER_IS_BETTER maps each metric name to a bool for sort-direction safety.
"""



import numpy as np


# ---------------------------------------------------------------------------
# Sort direction registry
# Phase 4 brute-force search must consult this — never hardcode sort direction.
# ---------------------------------------------------------------------------

METRIC_HIGHER_IS_BETTER: dict[str, bool] = {
    "cosine": True,
    "dot": True,
    "euclidean": False,
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ZeroVectorError(Exception):
    """Raised when cosine similarity is attempted on a zero-magnitude vector."""


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Returns a value in [-1.0, 1.0].
    1.0  → identical direction
    0.0  → orthogonal
    -1.0 → opposite direction

    Args:
        a: NumPy float32 array.
        b: NumPy float32 array of identical shape.

    Returns:
        float in [-1.0, 1.0]

    Raises:
        ZeroVectorError: Either vector has zero magnitude.
    """
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))

    if norm_a == 0.0:
        raise ZeroVectorError(
            "Vector 'a' has zero magnitude. Cosine similarity is undefined for zero vectors."
        )
    if norm_b == 0.0:
        raise ZeroVectorError(
            "Vector 'b' has zero magnitude. Cosine similarity is undefined for zero vectors."
        )

    raw = float(np.dot(a, b)) / (norm_a * norm_b)

    # Clamp to [-1.0, 1.0] to correct floating-point rounding errors
    # e.g., a unit vector dotted with itself can produce 1.0000000000000002
    return float(np.clip(raw, -1.0, 1.0))


def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the dot product (inner product) of two vectors.

    No normalization is applied. Magnitude and direction both contribute.

    Args:
        a: NumPy float32 array.
        b: NumPy float32 array of identical shape.

    Returns:
        float
    """
    return float(np.dot(a, b))


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the L2 (Euclidean) distance between two vectors.

    Lower values indicate more similar vectors.
    Identical vectors return 0.0.

    Args:
        a: NumPy float32 array.
        b: NumPy float32 array of identical shape.

    Returns:
        float >= 0.0
    """
    return float(np.linalg.norm(a - b))


# ---------------------------------------------------------------------------
# Dispatch table
# Allows Phase 4 to call compute_distance(metric, a, b) without if-elif chains.
# ---------------------------------------------------------------------------

_METRIC_FUNCTIONS = {
    "cosine": cosine_similarity,
    "dot": dot_product,
    "euclidean": l2_distance,
}


def compute_distance(metric: str, a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the distance or similarity between two vectors using the named metric.

    This is the function Phase 4 and later phases call — it removes if-elif
    chains from callers and makes adding a new metric a one-line change here.

    Args:
        metric: One of "cosine", "dot", "euclidean".
        a:      NumPy float32 array.
        b:      NumPy float32 array of identical shape.

    Returns:
        float. Interpretation depends on metric — consult METRIC_HIGHER_IS_BETTER.

    Raises:
        ValueError:      Metric name is not in the supported set.
        ZeroVectorError: Cosine similarity called with a zero-magnitude vector.
    """
    if metric not in _METRIC_FUNCTIONS:
        raise ValueError(
            f"Unknown metric '{metric}'. Supported: {sorted(_METRIC_FUNCTIONS)}."
        )
    return _METRIC_FUNCTIONS[metric](a, b)