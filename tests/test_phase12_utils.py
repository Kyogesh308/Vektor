"""Phase 12 — Numpy serialization tests. Test before any endpoint uses this."""

import numpy as np
import pytest

from vektor.api.utils import to_json_safe


class TestToJsonSafe:

    def test_ndarray_converts_to_list(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = to_json_safe(arr)
        assert result == [1.0, 2.0, 3.0]
        assert isinstance(result, list)

    def test_numpy_float32_scalar(self):
        val = np.float32(3.14)
        result = to_json_safe(val)
        assert isinstance(result, float)

    def test_numpy_int64_scalar(self):
        val = np.int64(42)
        result = to_json_safe(val)
        assert isinstance(result, int)

    def test_nested_dict_with_numpy_values(self):
        data = {"score": np.float32(0.95), "meta": {"count": np.int64(10)}}
        result = to_json_safe(data)
        assert isinstance(result["score"], float)
        assert isinstance(result["meta"]["count"], int)

    def test_list_of_dicts_with_numpy_arrays(self):
        data = [{"vector": np.array([1.0, 2.0])}, {"vector": np.array([3.0, 4.0])}]
        result = to_json_safe(data)
        assert result[0]["vector"] == [1.0, 2.0]
        assert result[1]["vector"] == [3.0, 4.0]

    def test_plain_python_values_pass_through(self):
        data = {"name": "test", "count": 5, "score": 0.9}
        assert to_json_safe(data) == data

    def test_deeply_nested_structure(self):
        data = {"a": [{"b": {"c": np.float32(1.5)}}]}
        result = to_json_safe(data)
        assert isinstance(result["a"][0]["b"]["c"], float)

    def test_none_passes_through(self):
        assert to_json_safe(None) is None