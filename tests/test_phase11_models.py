"""
Phase 11 — Data Structure Tests

Test these before anything uses them. If SearchResult equality or
CollectionConfig immutability is wrong, every downstream test becomes
harder to debug.
"""

import dataclasses
import numpy as np
import pytest

from vektor.models import (
    SearchResult, VektorConfig, CollectionConfig,
    BatchInsertResult, MemoryEstimate,
)


class TestSearchResult:

    def test_equality_based_on_id_only(self):
        r1 = SearchResult(id="v1", score=0.9, metadata={})
        r2 = SearchResult(id="v1", score=0.1, metadata={"different": True})
        assert r1 == r2

    def test_inequality_different_ids(self):
        r1 = SearchResult(id="v1", score=0.9, metadata={})
        r2 = SearchResult(id="v2", score=0.9, metadata={})
        assert r1 != r2

    def test_repr_includes_id_and_score(self):
        r = SearchResult(id="v1", score=0.876543, metadata={}, rank=1)
        repr_str = repr(r)
        assert "v1" in repr_str
        assert "0.876543" in repr_str

    def test_repr_handles_none_score(self):
        r = SearchResult(id="v1", score=None, metadata={})
        repr_str = repr(r)
        assert "None" in repr_str

    def test_hashable_for_set_usage(self):
        r1 = SearchResult(id="v1", score=0.9, metadata={})
        r2 = SearchResult(id="v1", score=0.1, metadata={})
        assert len({r1, r2}) == 1


class TestVektorConfig:

    def test_defaults(self):
        cfg = VektorConfig()
        assert cfg.lock_timeout == 5.0
        assert cfg.log_level == "INFO"
        assert cfg.default_overfetch_factor == 3

    def test_string_data_dir_normalised_to_path(self):
        from pathlib import Path
        cfg = VektorConfig(data_dir="/tmp/vektor_test")
        assert isinstance(cfg.data_dir, Path)


class TestCollectionConfig:

    def test_frozen_raises_on_mutation(self):
        cfg = CollectionConfig(
            name="test", dim=128, metric="cosine", M=16,
            ef_construction=200, mode="research", created_at="2025-01-01T00:00:00Z",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.dim = 256


class TestBatchInsertResult:

    def test_repr(self):
        result = BatchInsertResult(inserted=999, failed=1,
                                   errors=[{"id": "bad", "error_message": "wrong dim"}])
        assert "999" in repr(result)
        assert "1" in repr(result)

    def test_empty_errors_default(self):
        result = BatchInsertResult(inserted=10, failed=0)
        assert result.errors == []


class TestMemoryEstimate:

    def test_all_fields_present(self):
        est = MemoryEstimate(
            graph_bytes=1000, vector_bytes=2000, metadata_bytes=500,
            total_bytes=3500, total_mb=0.0035,
        )
        assert est.total_bytes == 3500
        assert est.warning == ""