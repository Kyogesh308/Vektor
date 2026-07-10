"""Phase 12 — Pydantic model validation tests."""

import pytest
from pydantic import ValidationError

from vektor.api.models import (
    CreateCollectionRequest, InsertVectorRequest, UpdateVectorRequest,
    SearchRequest,
)


class TestCreateCollectionRequest:

    def test_valid_request(self):
        req = CreateCollectionRequest(name="docs", dim=384, metric="cosine")
        assert req.mode == "beginner"

    def test_invalid_metric_raises(self):
        with pytest.raises(ValidationError):
            CreateCollectionRequest(name="docs", dim=384, metric="manhattan")

    def test_negative_dim_raises(self):
        with pytest.raises(ValidationError):
            CreateCollectionRequest(name="docs", dim=-1, metric="cosine")


class TestUpdateVectorRequest:

    def test_both_none_raises(self):
        with pytest.raises(ValidationError, match="At least one"):
            UpdateVectorRequest()

    def test_vector_only_valid(self):
        req = UpdateVectorRequest(vector=[0.1, 0.2])
        assert req.metadata is None

    def test_metadata_only_valid(self):
        req = UpdateVectorRequest(metadata={"key": "value"})
        assert req.vector is None


class TestSearchRequest:

    def test_ef_less_than_k_raises(self):
        with pytest.raises(ValidationError, match="must be >="):
            SearchRequest(query=[0.1, 0.2], k=10, ef=5)

    def test_ef_none_is_valid(self):
        req = SearchRequest(query=[0.1, 0.2], k=10)
        assert req.ef is None

    def test_ef_geq_k_is_valid(self):
        req = SearchRequest(query=[0.1, 0.2], k=5, ef=100)
        assert req.ef == 100

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValidationError):
            SearchRequest(query=[0.1], k=5, strategy="auto")