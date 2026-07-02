"""Input validation layer. Implementation begins in Phase 1."""
"""
vektor.validator
----------------
Input validation for all data entering the Vektor system.

Design contract:
- Every public function either returns None (input is valid) or raises a
  specific, named exception.
- No function attempts to repair, transform, or coerce input.
- Bad input is rejected immediately. It never reaches storage.

import numpy as np
import math

a = np.array([1.0, float('nan'), 3.0])
b = np.array([1.0, 2.0, 3.0])

# NaN propagates silently — no error raised
diff = a - b
print(diff)       # [ 0.  nan  0.]
print(np.sum(diff ** 2))   # nan — distance calculation is now poisoned

# Wrong — tells the caller nothing
raise ValueError("bad vector")

# Right — tells the caller exactly what failed and why
raise VectorDimensionError(
    f"Expected dimension 128, got 512."
)

"""

from __future__ import annotations

import json
import numpy as np
from typing import Any


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

class VektorValidationError(Exception):
    """Base class for all Vektor validation failures.
    
    Catch this if you want to handle any validation failure uniformly.
    Catch the subclasses if you need to distinguish between failure modes.
    """


class InvalidVectorTypeError(VektorValidationError):
    """Raised when a vector is not a list or NumPy array."""


class InvalidVectorDimensionError(VektorValidationError):
    """Raised when a vector's length does not match the expected dimension."""


class EmptyVectorError(VektorValidationError):
    """Raised when a vector has zero elements."""


class NonNumericVectorError(VektorValidationError):
    """Raised when a vector contains non-numeric values."""


class NonFiniteVectorError(VektorValidationError):
    """Raised when a vector contains NaN or infinity values."""


class InvalidIDTypeError(VektorValidationError):
    """Raised when an ID is not a string."""


class EmptyIDError(VektorValidationError):
    """Raised when an ID is an empty string."""


class IDTooLongError(VektorValidationError):
    """Raised when an ID exceeds the maximum allowed length."""


class InvalidIDCharacterError(VektorValidationError):
    """Raised when an ID contains forbidden characters."""


class InvalidMetadataTypeError(VektorValidationError):
    """Raised when metadata is not a dict."""


class NonSerializableMetadataError(VektorValidationError):
    """Raised when a metadata value cannot be serialized to JSON."""


class InvalidMetricError(VektorValidationError):
    """Raised when a metric name is not in the supported set."""
    
    
    
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ID_MAX_LENGTH = 512
ID_FORBIDDEN_CHARACTERS = set('/ \t\n\r\x00')
SUPPORTED_METRICS = {"cosine", "euclidean", "dot"}


# ---------------------------------------------------------------------------
# Vector validation
# ---------------------------------------------------------------------------

def validate_vector(vector: Any, expected_dim: int) -> np.ndarray:
    """
    Validate a vector input and return it as a NumPy float32 array.

    Args:
        vector:       The input to validate. Must be a list or np.ndarray.
        expected_dim: The exact number of dimensions required.

    Returns:
        A NumPy float32 array of shape (expected_dim,).

    Raises:
        InvalidVectorTypeError:      Input is not a list or np.ndarray.
        EmptyVectorError:            Input has zero elements.
        NonNumericVectorError:       Input contains non-numeric values.
        NonFiniteVectorError:        Input contains NaN or infinity.
        InvalidVectorDimensionError: Length does not match expected_dim.
    """
    # 1. Type check — must come first, before any shape or content access
    if not isinstance(vector, (list, np.ndarray)):
        raise InvalidVectorTypeError(
            f"Vector must be a list or numpy.ndarray, got {type(vector).__name__}."
        )

    # 2. Convert to NumPy array for uniform handling
    #    Use float64 during validation for precision, then cast to float32
    try:
        arr = np.array(vector, dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise NonNumericVectorError(
            f"Vector contains non-numeric values and could not be converted: {e}"
        ) from e

    # 3. Empty check — after conversion so shape is reliable
    if arr.size == 0:
        raise EmptyVectorError("Vector must not be empty.")

    # 4. Dimension check — before content checks to fail fast on obviously wrong input
    if arr.ndim != 1:
        raise InvalidVectorTypeError(
            f"Vector must be 1-dimensional, got shape {arr.shape}."
        )

    if arr.shape[0] != expected_dim:
        raise InvalidVectorDimensionError(
            f"Expected dimension {expected_dim}, got {arr.shape[0]}."
        )

    # 5. Finite check — NaN and Infinity are valid float64 values; check explicitly
    if np.any(np.isnan(arr)):
        raise NonFiniteVectorError(
            "Vector contains NaN values. Check your embedding pipeline for division by zero."
        )

    if np.any(np.isinf(arr)):
        raise NonFiniteVectorError(
            "Vector contains infinity values. Check your embedding pipeline for overflow."
        )

    # Return as float32 — halves memory usage, sufficient precision for distance math
    return arr.astype(np.float32)

# ---------------------------------------------------------------------------
# ID validation
# ---------------------------------------------------------------------------

def validate_id(id_value: Any) -> str:
    """
    Validate a vector ID.

    Args:
        id_value: The ID to validate. Must be a non-empty string.

    Returns:
        The validated ID string, unchanged.

    Raises:
        InvalidIDTypeError:      Not a string.
        EmptyIDError:            Empty string.
        IDTooLongError:          Exceeds ID_MAX_LENGTH characters.
        InvalidIDCharacterError: Contains a forbidden character.
    """
    if not isinstance(id_value, str):
        raise InvalidIDTypeError(
            f"ID must be a string, got {type(id_value).__name__}."
        )

    if len(id_value) == 0:
        raise EmptyIDError("ID must not be an empty string.")

    if len(id_value) > ID_MAX_LENGTH:
        raise IDTooLongError(
            f"ID length {len(id_value)} exceeds maximum of {ID_MAX_LENGTH} characters."
        )

    forbidden_found = ID_FORBIDDEN_CHARACTERS.intersection(id_value)
    if forbidden_found:
        printable = {repr(c) for c in forbidden_found}
        raise InvalidIDCharacterError(
            f"ID contains forbidden characters: {printable}."
        )

    return id_value


# ---------------------------------------------------------------------------
# Metadata validation
# ---------------------------------------------------------------------------

def validate_metadata(metadata: Any) -> dict:
    """
    Validate vector metadata.

    Args:
        metadata: The metadata to validate. Must be a dict with
                  JSON-serializable values.

    Returns:
        The validated metadata dict, unchanged.

    Raises:
        InvalidMetadataTypeError:   Not a dict.
        NonSerializableMetadataError: Any value is not JSON-serializable.
    """
    if not isinstance(metadata, dict):
        raise InvalidMetadataTypeError(
            f"Metadata must be a dict, got {type(metadata).__name__}."
        )

    try:
        json.dumps(metadata)
    except (TypeError, ValueError) as e:
        raise NonSerializableMetadataError(
            f"Metadata contains a value that cannot be serialized to JSON: {e}"
        ) from e

    return metadata