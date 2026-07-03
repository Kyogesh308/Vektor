"""
Phase 2 — Distance Engine Tests

Every expected value in this file was calculated by hand before being
written as an assertion. If a test fails, the implementation is wrong —
not the expected value.

Floating-point comparisons use pytest.approx(abs=1e-6) throughout.
Do not loosen this tolerance to make a failing test pass.
"""

import numpy as np
import pytest

from vektor.distance import (
    cosine_similarity,
    dot_product,
    l2_distance,
    compute_distance,
    METRIC_HIGHER_IS_BETTER,
    ZeroVectorError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def vec(*values) -> np.ndarray:
    """Shorthand for creating float32 test vectors."""
    return np.array(values, dtype=np.float32)


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:

    def test_identical_vectors_return_one(self):
        a = vec(1.0, 0.0)
        assert cosine_similarity(a, a) == pytest.approx(1.0, abs=1e-6)

    def test_identical_nonunit_vectors_return_one(self):
        # Direction is the same regardless of magnitude
        a = vec(3.0, 0.0)
        b = vec(7.0, 0.0)
        assert cosine_similarity(a, b) == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors_return_zero(self):
        a = vec(1.0, 0.0)
        b = vec(0.0, 1.0)
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors_return_negative_one(self):
        a = vec(1.0, 0.0)
        b = vec(-1.0, 0.0)
        assert cosine_similarity(a, b) == pytest.approx(-1.0, abs=1e-6)

    def test_known_3d_vectors(self):
        # a = [1, 2, 3], b = [4, 5, 6]
        # dot = 4+10+18 = 32
        # ||a|| = sqrt(1+4+9) = sqrt(14) ≈ 3.7417
        # ||b|| = sqrt(16+25+36) = sqrt(77) ≈ 8.7749
        # cosine = 32 / (3.7417 × 8.7749) ≈ 0.9746
        a = vec(1.0, 2.0, 3.0)
        b = vec(4.0, 5.0, 6.0)
        assert cosine_similarity(a, b) == pytest.approx(0.9746318, abs=1e-5)

    def test_result_clamped_to_valid_range(self):
        # Unit vector dotted with itself can exceed 1.0 due to float32 rounding
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        result = cosine_similarity(a, a)
        assert -1.0 <= result <= 1.0

    def test_zero_vector_a_raises(self):
        a = vec(0.0, 0.0, 0.0)
        b = vec(1.0, 2.0, 3.0)
        with pytest.raises(ZeroVectorError):
            cosine_similarity(a, b)

    def test_zero_vector_b_raises(self):
        a = vec(1.0, 2.0, 3.0)
        b = vec(0.0, 0.0, 0.0)
        with pytest.raises(ZeroVectorError):
            cosine_similarity(a, b)

    def test_negative_components_handled_correctly(self):
        a = vec(-1.0, 0.0)
        b = vec(-1.0, 0.0)
        assert cosine_similarity(a, b) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# dot_product
# ---------------------------------------------------------------------------

class TestDotProduct:

    def test_known_3d_vectors(self):
        # 1×4 + 2×5 + 3×6 = 32
        a = vec(1.0, 2.0, 3.0)
        b = vec(4.0, 5.0, 6.0)
        assert dot_product(a, b) == pytest.approx(32.0, abs=1e-6)

    def test_orthogonal_vectors_return_zero(self):
        a = vec(1.0, 0.0)
        b = vec(0.0, 1.0)
        assert dot_product(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors(self):
        a = vec(1.0, 0.0)
        b = vec(-1.0, 0.0)
        assert dot_product(a, b) == pytest.approx(-1.0, abs=1e-6)

    def test_magnitude_affects_score(self):
        # Unlike cosine, magnitude matters here
        a = vec(2.0, 0.0)
        b = vec(3.0, 0.0)
        assert dot_product(a, b) == pytest.approx(6.0, abs=1e-6)

    def test_zero_vector_returns_zero(self):
        # Unlike cosine, this is valid and does not raise
        a = vec(0.0, 0.0, 0.0)
        b = vec(1.0, 2.0, 3.0)
        assert dot_product(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_negative_dot_product(self):
        a = vec(1.0, 1.0)
        b = vec(-1.0, -1.0)
        assert dot_product(a, b) == pytest.approx(-2.0, abs=1e-6)

    def test_unit_vectors_equals_cosine(self):
        # For unit vectors, dot product and cosine similarity must be identical
        a = vec(1.0, 0.0)
        b = vec(0.0, 1.0)
        assert dot_product(a, b) == pytest.approx(cosine_similarity(a, b), abs=1e-6)


# ---------------------------------------------------------------------------
# l2_distance
# ---------------------------------------------------------------------------

class TestL2Distance:

    def test_identical_vectors_return_zero(self):
        a = vec(1.0, 2.0, 3.0)
        assert l2_distance(a, a) == pytest.approx(0.0, abs=1e-6)

    def test_known_2d_case(self):
        # 3-4-5 right triangle: sqrt(3² + 4²) = sqrt(25) = 5.0
        a = vec(0.0, 0.0)
        b = vec(3.0, 4.0)
        assert l2_distance(a, b) == pytest.approx(5.0, abs=1e-6)

    def test_unit_step_along_first_axis(self):
        a = vec(0.0, 0.0)
        b = vec(1.0, 0.0)
        assert l2_distance(a, b) == pytest.approx(1.0, abs=1e-6)

    def test_symmetry(self):
        # l2_distance must be symmetric: d(a, b) == d(b, a)
        a = vec(1.0, 2.0, 3.0)
        b = vec(4.0, 6.0, 8.0)
        assert l2_distance(a, b) == pytest.approx(l2_distance(b, a), abs=1e-6)

    def test_result_is_non_negative(self):
        a = vec(-3.0, -4.0)
        b = vec(3.0, 4.0)
        assert l2_distance(a, b) >= 0.0

    def test_known_3d_case(self):
        # sqrt((4-1)² + (6-2)² + (8-3)²) = sqrt(9+16+25) = sqrt(50) ≈ 7.0711
        a = vec(1.0, 2.0, 3.0)
        b = vec(4.0, 6.0, 8.0)
        assert l2_distance(a, b) == pytest.approx(7.07107, abs=1e-4)


# ---------------------------------------------------------------------------
# compute_distance (dispatch)
# ---------------------------------------------------------------------------

class TestComputeDistance:

    def test_dispatches_cosine(self):
        a = vec(1.0, 0.0)
        b = vec(1.0, 0.0)
        result = compute_distance("cosine", a, b)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_dispatches_dot(self):
        a = vec(1.0, 2.0, 3.0)
        b = vec(4.0, 5.0, 6.0)
        result = compute_distance("dot", a, b)
        assert result == pytest.approx(32.0, abs=1e-6)

    def test_dispatches_euclidean(self):
        a = vec(0.0, 0.0)
        b = vec(3.0, 4.0)
        result = compute_distance("euclidean", a, b)
        assert result == pytest.approx(5.0, abs=1e-6)

    def test_unknown_metric_raises_value_error(self):
        a = vec(1.0, 2.0)
        b = vec(3.0, 4.0)
        with pytest.raises(ValueError):
            compute_distance("manhattan", a, b)


# ---------------------------------------------------------------------------
# METRIC_HIGHER_IS_BETTER — sort direction registry
# ---------------------------------------------------------------------------

class TestSortDirectionRegistry:

    def test_cosine_is_higher_is_better(self):
        assert METRIC_HIGHER_IS_BETTER["cosine"] is True

    def test_dot_is_higher_is_better(self):
        assert METRIC_HIGHER_IS_BETTER["dot"] is True

    def test_euclidean_is_lower_is_better(self):
        assert METRIC_HIGHER_IS_BETTER["euclidean"] is False

    def test_all_supported_metrics_are_registered(self):
        # If a new metric is added to compute_distance, it must also be in this dict.
        # This test catches the omission.
        from vektor.distance import _METRIC_FUNCTIONS
        assert set(_METRIC_FUNCTIONS.keys()) == set(METRIC_HIGHER_IS_BETTER.keys())