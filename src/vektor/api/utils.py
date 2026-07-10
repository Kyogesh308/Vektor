"""
vektor.api.utils
-----------------
JSON serialization helpers. Numpy types are not natively JSON-serializable —
this converts them recursively before any response leaves the API.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def to_json_safe(value: Any) -> Any:
    """
    Recursively convert numpy types to native Python types.

    Handles: ndarray, numpy scalars (float32, int64, etc.), nested dicts,
    nested lists/tuples. Passes through anything already JSON-safe unchanged.

    Args:
        value: Any value, possibly containing numpy types at any nesting depth.

    Returns:
        The same structure with all numpy types converted to native Python
        types (float, int, list).
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(v) for v in value]
    return value