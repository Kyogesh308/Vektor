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