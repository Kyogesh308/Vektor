"""
Phase 1 — Input Validator Tests

Coverage requirement:
- Every function has exactly one test for valid input (the happy path).
- Every function has one test per distinct failure mode.
- No test checks more than one failure at a time.
"""

import numpy as np
import pytest

from vektor.validator import (
    validate_vector,
    validate_id,
    validate_metadata,
    validate_metric,
    InvalidVectorTypeError,
    InvalidVectorDimensionError,
    EmptyVectorError,
    NonNumericVectorError,
    NonFiniteVectorError,
    InvalidIDTypeError,
    EmptyIDError,
    IDTooLongError,
    InvalidIDCharacterError,
    InvalidMetadataTypeError,
    NonSerializableMetadataError,
    InvalidMetricError,
    ID_MAX_LENGTH,
)


# ---------------------------------------------------------------------------
# validate_vector
# ---------------------------------------------------------------------------

class TestValidateVector:

    def test_valid_list_returns_float32_array(self):
        result = validate_vector([1.0, 2.0, 3.0], expected_dim=3)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (3,)

    def test_valid_numpy_array_passes(self):
        v = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        result = validate_vector(v, expected_dim=3)
        assert result.shape == (3,)

    def test_rejects_non_list_non_array(self):
        with pytest.raises(InvalidVectorTypeError):
            validate_vector("not a vector", expected_dim=3)

    def test_rejects_dict_input(self):
        with pytest.raises(InvalidVectorTypeError):
            validate_vector({"a": 1}, expected_dim=3)

    def test_rejects_integer_input(self):
        with pytest.raises(InvalidVectorTypeError):
            validate_vector(42, expected_dim=3)

    def test_rejects_empty_list(self):
        with pytest.raises(EmptyVectorError):
            validate_vector([], expected_dim=0)

    def test_rejects_wrong_dimension(self):
        with pytest.raises(InvalidVectorDimensionError):
            validate_vector([1.0, 2.0, 3.0], expected_dim=4)

    def test_rejects_nan_value(self):
        with pytest.raises(NonFiniteVectorError):
            validate_vector([1.0, float('nan'), 3.0], expected_dim=3)

    def test_rejects_positive_infinity(self):
        with pytest.raises(NonFiniteVectorError):
            validate_vector([1.0, float('inf'), 3.0], expected_dim=3)

    def test_rejects_negative_infinity(self):
        with pytest.raises(NonFiniteVectorError):
            validate_vector([1.0, float('-inf'), 3.0], expected_dim=3)

    def test_rejects_non_numeric_string_in_list(self):
        with pytest.raises(NonNumericVectorError):
            validate_vector([1.0, "two", 3.0], expected_dim=3)

    def test_rejects_2d_array(self):
        v = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(InvalidVectorTypeError):
            validate_vector(v, expected_dim=4)

    def test_integer_list_is_valid(self):
        # Integers should be accepted and coerced to float32
        result = validate_vector([1, 2, 3], expected_dim=3)
        assert result.dtype == np.float32

    def test_nan_produced_by_numpy_is_rejected(self):
        # NaN doesn't only come from float('nan') — verify np.nan is caught too
        with pytest.raises(NonFiniteVectorError):
            validate_vector([1.0, np.nan, 3.0], expected_dim=3)


# ---------------------------------------------------------------------------
# validate_id
# ---------------------------------------------------------------------------

class TestValidateID:

    def test_valid_id_returns_unchanged(self):
        result = validate_id("vec_001")
        assert result == "vec_001"

    def test_rejects_integer_id(self):
        with pytest.raises(InvalidIDTypeError):
            validate_id(42)

    def test_rejects_none(self):
        with pytest.raises(InvalidIDTypeError):
            validate_id(None)

    def test_rejects_empty_string(self):
        with pytest.raises(EmptyIDError):
            validate_id("")

    def test_rejects_id_exceeding_max_length(self):
        long_id = "a" * (ID_MAX_LENGTH + 1)
        with pytest.raises(IDTooLongError):
            validate_id(long_id)

    def test_accepts_id_at_exact_max_length(self):
        boundary_id = "a" * ID_MAX_LENGTH
        result = validate_id(boundary_id)
        assert len(result) == ID_MAX_LENGTH

    def test_rejects_id_with_forward_slash(self):
        with pytest.raises(InvalidIDCharacterError):
            validate_id("vec/001")

    def test_rejects_id_with_space(self):
        with pytest.raises(InvalidIDCharacterError):
            validate_id("vec 001")

    def test_rejects_id_with_tab(self):
        with pytest.raises(InvalidIDCharacterError):
            validate_id("vec\t001")

    def test_rejects_id_with_newline(self):
        with pytest.raises(InvalidIDCharacterError):
            validate_id("vec\n001")

    def test_rejects_id_with_null_byte(self):
        with pytest.raises(InvalidIDCharacterError):
            validate_id("vec\x00001")

    def test_hyphens_underscores_are_valid(self):
        # These are common in real IDs and must not be rejected
        result = validate_id("user-123_doc-456")
        assert result == "user-123_doc-456"

    def test_uuid_format_is_valid(self):
        result = validate_id("550e8400-e29b-41d4-a716-446655440000")
        assert result == "550e8400-e29b-41d4-a716-446655440000"


# ---------------------------------------------------------------------------
# validate_metadata
# ---------------------------------------------------------------------------

class TestValidateMetadata:

    def test_valid_empty_dict_passes(self):
        result = validate_metadata({})
        assert result == {}

    def test_valid_string_values_pass(self):
        result = validate_metadata({"source": "arxiv", "year": 2024})
        assert result["source"] == "arxiv"

    def test_valid_nested_dict_passes(self):
        result = validate_metadata({"tags": ["ml", "rag"], "meta": {"k": 1}})
        assert result["tags"] == ["ml", "rag"]

    def test_rejects_list_input(self):
        with pytest.raises(InvalidMetadataTypeError):
            validate_metadata(["key", "value"])

    def test_rejects_string_input(self):
        with pytest.raises(InvalidMetadataTypeError):
            validate_metadata("not a dict")

    def test_rejects_none_input(self):
        with pytest.raises(InvalidMetadataTypeError):
            validate_metadata(None)

    def test_rejects_non_serializable_value(self):
        # A Python set is not JSON-serializable
        with pytest.raises(NonSerializableMetadataError):
            validate_metadata({"tags": {1, 2, 3}})

    def test_rejects_custom_object_value(self):
        class Unserializable:
            pass
        with pytest.raises(NonSerializableMetadataError):
            validate_metadata({"obj": Unserializable()})

    def test_boolean_values_are_valid(self):
        result = validate_metadata({"active": True, "archived": False})
        assert result["active"] is True

    def test_none_value_in_dict_is_valid(self):
        # JSON supports null, so None values are valid
        result = validate_metadata({"label": None})
        assert result["label"] is None


# ---------------------------------------------------------------------------
# validate_metric
# ---------------------------------------------------------------------------

class TestValidateMetric:

    def test_cosine_is_valid(self):
        assert validate_metric("cosine") == "cosine"

    def test_euclidean_is_valid(self):
        assert validate_metric("euclidean") == "euclidean"

    def test_dot_is_valid(self):
        assert validate_metric("dot") == "dot"

    def test_rejects_unknown_metric(self):
        with pytest.raises(InvalidMetricError):
            validate_metric("manhattan")

    def test_rejects_uppercase_variant(self):
        # Metric names are case-sensitive
        with pytest.raises(InvalidMetricError):
            validate_metric("Cosine")

    def test_rejects_integer_metric(self):
        with pytest.raises(InvalidMetricError):
            validate_metric(1)

    def test_rejects_empty_string(self):
        with pytest.raises(InvalidMetricError):
            validate_metric("")

    def test_rejects_none(self):
        with pytest.raises(InvalidMetricError):
            validate_metric(None)