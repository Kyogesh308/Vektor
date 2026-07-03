"""
Phase 4 — Vector Storage and Brute-Force Search Tests

The most important tests in this file are:
    TestSearch::test_cosine_returns_correct_nearest_neighbor
    TestSearch::test_euclidean_returns_correct_nearest_neighbor
    TestSearch::test_dot_returns_correct_nearest_neighbor

These verify that brute-force search is mathematically correct.
Every future HNSW recall benchmark is measured against this function.
If these tests are wrong, all future recall numbers are wrong.
"""

import numpy as np
import pytest

from vektor.collection import Collection
from vektor.storage import (
    VectorStore,
    SearchResult,
    VectorNotFoundError,
    DuplicateVectorIDError,
    InvalidKError,
)
from vektor.validator import (
    InvalidVectorDimensionError,
    NonFiniteVectorError,
    InvalidIDTypeError,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cosine_collection():
    return Collection(name="test_cosine", dimension=4, metric="cosine")


@pytest.fixture
def euclidean_collection():
    return Collection(name="test_euclidean", dimension=4, metric="euclidean")


@pytest.fixture
def dot_collection():
    return Collection(name="test_dot", dimension=4, metric="dot")


@pytest.fixture
def cosine_store(cosine_collection):
    return VectorStore(cosine_collection)


@pytest.fixture
def euclidean_store(euclidean_collection):
    return VectorStore(euclidean_collection)


@pytest.fixture
def dot_store(dot_collection):
    return VectorStore(dot_collection)


def vec(*values) -> np.ndarray:
    return np.array(values, dtype=np.float32)


# ---------------------------------------------------------------------------
# insert()
# ---------------------------------------------------------------------------

class TestInsert:

    def test_insert_succeeds(self, cosine_store):
        cosine_store.insert("v1", [1.0, 0.0, 0.0, 0.0])
        assert cosine_store.count() == 1

    def test_insert_with_metadata(self, cosine_store):
        cosine_store.insert("v1", [1.0, 0.0, 0.0, 0.0], {"source": "arxiv"})
        result = cosine_store.get("v1")
        assert result["metadata"]["source"] == "arxiv"

    def test_insert_without_metadata_defaults_to_empty_dict(self, cosine_store):
        cosine_store.insert("v1", [1.0, 0.0, 0.0, 0.0])
        assert cosine_store.get("v1")["metadata"] == {}

    def test_vector_stored_as_float32(self, cosine_store):
        cosine_store.insert("v1", [1.0, 2.0, 3.0, 4.0])
        stored = cosine_store.get("v1")["vector"]
        assert stored.dtype == np.float32

    def test_duplicate_id_raises(self, cosine_store):
        cosine_store.insert("v1", [1.0, 0.0, 0.0, 0.0])
        with pytest.raises(DuplicateVectorIDError):
            cosine_store.insert("v1", [0.0, 1.0, 0.0, 0.0])

    def test_wrong_dimension_raises(self, cosine_store):
        with pytest.raises(InvalidVectorDimensionError):
            cosine_store.insert("v1", [1.0, 0.0, 0.0])  # dim 3, expects 4

    def test_nan_in_vector_raises(self, cosine_store):
        with pytest.raises(NonFiniteVectorError):
            cosine_store.insert("v1", [1.0, float("nan"), 0.0, 0.0])

    def test_invalid_id_raises(self, cosine_store):
        with pytest.raises(InvalidIDTypeError):
            cosine_store.insert(123, [1.0, 0.0, 0.0, 0.0])

    def test_deleted_id_can_be_reinserted(self, cosine_store):
        cosine_store.insert("v1", [1.0, 0.0, 0.0, 0.0])
        cosine_store.delete("v1")
        # Should succeed — tombstoned ID is available again
        cosine_store.insert("v1", [0.0, 1.0, 0.0, 0.0])
        assert cosine_store.count() == 1


# ---------------------------------------------------------------------------
# get()
# ---------------------------------------------------------------------------

class TestGet:

    def test_get_returns_correct_vector(self, cosine_store):
        cosine_store.insert("v1", [1.0, 2.0, 3.0, 4.0])
        result = cosine_store.get("v1")
        assert result["id"] == "v1"
        np.testing.assert_array_almost_equal(result["vector"], [1.0, 2.0, 3.0, 4.0])

    def test_get_missing_id_raises(self, cosine_store):
        with pytest.raises(VectorNotFoundError):
            cosine_store.get("nonexistent")

    def test_get_deleted_id_raises(self, cosine_store):
        cosine_store.insert("v1", [1.0, 0.0, 0.0, 0.0])
        cosine_store.delete("v1")
        with pytest.raises(VectorNotFoundError):
            cosine_store.get("v1")


# ---------------------------------------------------------------------------
# update()
# ---------------------------------------------------------------------------

class TestUpdate:

    def test_update_replaces_vector(self, cosine_store):
        cosine_store.insert("v1", [1.0, 0.0, 0.0, 0.0])
        cosine_store.update("v1", [0.0, 1.0, 0.0, 0.0])
        result = cosine_store.get("v1")
        np.testing.assert_array_almost_equal(result["vector"], [0.0, 1.0, 0.0, 0.0])

    def test_update_replaces_metadata(self, cosine_store):
        cosine_store.insert("v1", [1.0, 0.0, 0.0, 0.0], {"v": 1})
        cosine_store.update("v1", [1.0, 0.0, 0.0, 0.0], {"v": 2})
        assert cosine_store.get("v1")["metadata"]["v"] == 2

    def test_update_preserves_metadata_when_none(self, cosine_store):
        cosine_store.insert("v1", [1.0, 0.0, 0.0, 0.0], {"keep": True})
        cosine_store.update("v1", [0.0, 1.0, 0.0, 0.0], metadata=None)
        assert cosine_store.get("v1")["metadata"]["keep"] is True

    def test_update_nonexistent_raises(self, cosine_store):
        with pytest.raises(VectorNotFoundError):
            cosine_store.update("ghost", [1.0, 0.0, 0.0, 0.0])

    def test_update_deleted_raises(self, cosine_store):
        cosine_store.insert("v1", [1.0, 0.0, 0.0, 0.0])
        cosine_store.delete("v1")
        with pytest.raises(VectorNotFoundError):
            cosine_store.update("v1", [0.0, 1.0, 0.0, 0.0])

    def test_update_wrong_dimension_raises(self, cosine_store):
        cosine_store.insert("v1", [1.0, 0.0, 0.0, 0.0])
        with pytest.raises(InvalidVectorDimensionError):
            cosine_store.update("v1", [1.0, 0.0])


# ---------------------------------------------------------------------------
# delete()
# ---------------------------------------------------------------------------

class TestDelete:

    def test_delete_removes_from_count(self, cosine_store):
        cosine_store.insert("v1", [1.0, 0.0, 0.0, 0.0])
        cosine_store.delete("v1")
        assert cosine_store.count() == 0

    def test_delete_does_not_remove_from_count_all(self, cosine_store):
        cosine_store.insert("v1", [1.0, 0.0, 0.0, 0.0])
        cosine_store.delete("v1")
        assert cosine_store.count_all() == 1  # tombstone still exists

    def test_delete_nonexistent_raises(self, cosine_store):
        with pytest.raises(VectorNotFoundError):
            cosine_store.delete("ghost")

    def test_double_delete_raises(self, cosine_store):
        cosine_store.insert("v1", [1.0, 0.0, 0.0, 0.0])
        cosine_store.delete("v1")
        with pytest.raises(VectorNotFoundError):
            cosine_store.delete("v1")

    def test_deleted_vector_not_in_ids(self, cosine_store):
        cosine_store.insert("v1", [1.0, 0.0, 0.0, 0.0])
        cosine_store.insert("v2", [0.0, 1.0, 0.0, 0.0])
        cosine_store.delete("v1")
        assert "v1" not in cosine_store.ids()
        assert "v2" in cosine_store.ids()


# ---------------------------------------------------------------------------
# search() — the most important tests in Phase 4
# ---------------------------------------------------------------------------

class TestSearch:

    def test_cosine_returns_correct_nearest_neighbor(self, cosine_store):
        """
        Hand-verified: query=[1,0,0,0]
        v1=[1,0,0,0] → cosine=1.0 (identical direction, nearest)
        v2=[0,1,0,0] → cosine=0.0 (orthogonal)
        v3=[-1,0,0,0] → cosine=-1.0 (opposite, furthest)
        Expected order: v1, v2, v3
        """
        cosine_store.insert("v1", [1.0, 0.0, 0.0, 0.0])
        cosine_store.insert("v2", [0.0, 1.0, 0.0, 0.0])
        cosine_store.insert("v3", [-1.0, 0.0, 0.0, 0.0])

        results = cosine_store.search([1.0, 0.0, 0.0, 0.0], k=3)

        assert results[0].id == "v1"
        assert results[1].id == "v2"
        assert results[2].id == "v3"
        assert results[0].score == pytest.approx(1.0, abs=1e-5)
        assert results[1].score == pytest.approx(0.0, abs=1e-5)
        assert results[2].score == pytest.approx(-1.0, abs=1e-5)

    def test_euclidean_returns_correct_nearest_neighbor(self, euclidean_store):
        """
        Hand-verified: query=[0,0,0,0]
        v1=[1,0,0,0] → L2=1.0 (nearest)
        v2=[2,0,0,0] → L2=2.0
        v3=[3,0,0,0] → L2=3.0 (furthest)
        Expected order: v1, v2, v3
        """
        euclidean_store.insert("v1", [1.0, 0.0, 0.0, 0.0])
        euclidean_store.insert("v2", [2.0, 0.0, 0.0, 0.0])
        euclidean_store.insert("v3", [3.0, 0.0, 0.0, 0.0])

        results = euclidean_store.search([0.0, 0.0, 0.0, 0.0], k=3)

        assert results[0].id == "v1"
        assert results[1].id == "v2"
        assert results[2].id == "v3"
        assert results[0].score == pytest.approx(1.0, abs=1e-5)
        assert results[1].score == pytest.approx(2.0, abs=1e-5)
        assert results[2].score == pytest.approx(3.0, abs=1e-5)

    def test_dot_returns_correct_nearest_neighbor(self, dot_store):
        """
        Hand-verified: query=[1,0,0,0]
        v1=[5,0,0,0] → dot=5.0 (highest, nearest)
        v2=[2,0,0,0] → dot=2.0
        v3=[1,0,0,0] → dot=1.0 (lowest, furthest)
        Expected order: v1, v2, v3
        """
        dot_store.insert("v1", [5.0, 0.0, 0.0, 0.0])
        dot_store.insert("v2", [2.0, 0.0, 0.0, 0.0])
        dot_store.insert("v3", [1.0, 0.0, 0.0, 0.0])

        results = dot_store.search([1.0, 0.0, 0.0, 0.0], k=3)

        assert results[0].id == "v1"
        assert results[1].id == "v2"
        assert results[2].id == "v3"

    def test_search_respects_k(self, cosine_store):
        for i in range(10):
            cosine_store.insert(f"v{i}", [float(i), 0.0, 0.0, 0.0])
        results = cosine_store.search([1.0, 0.0, 0.0, 0.0], k=3)
        assert len(results) == 3

    def test_search_returns_fewer_than_k_when_store_is_small(self, cosine_store):
        cosine_store.insert("v1", [1.0, 0.0, 0.0, 0.0])
        results = cosine_store.search([1.0, 0.0, 0.0, 0.0], k=5)
        assert len(results) == 1

    def test_search_excludes_deleted_vectors(self, cosine_store):
        cosine_store.insert("best", [1.0, 0.0, 0.0, 0.0])
        cosine_store.insert("other", [0.0, 1.0, 0.0, 0.0])
        cosine_store.delete("best")

        results = cosine_store.search([1.0, 0.0, 0.0, 0.0], k=5)
        ids = [r.id for r in results]
        assert "best" not in ids
        assert "other" in ids

    def test_search_on_empty_store_returns_empty_list(self, cosine_store):
        results = cosine_store.search([1.0, 0.0, 0.0, 0.0], k=5)
        assert results == []

    def test_search_invalid_k_zero_raises(self, cosine_store):
        cosine_store.insert("v1", [1.0, 0.0, 0.0, 0.0])
        with pytest.raises(InvalidKError):
            cosine_store.search([1.0, 0.0, 0.0, 0.0], k=0)

    def test_search_invalid_k_negative_raises(self, cosine_store):
        with pytest.raises(InvalidKError):
            cosine_store.search([1.0, 0.0, 0.0, 0.0], k=-1)

    def test_search_invalid_k_float_raises(self, cosine_store):
        with pytest.raises(InvalidKError):
            cosine_store.search([1.0, 0.0, 0.0, 0.0], k=3.0)

    def test_search_query_wrong_dimension_raises(self, cosine_store):
        with pytest.raises(InvalidVectorDimensionError):
            cosine_store.search([1.0, 0.0], k=3)  # dim 2, expects 4

    def test_search_result_contains_metadata(self, cosine_store):
        cosine_store.insert("v1", [1.0, 0.0, 0.0, 0.0], {"label": "test"})
        results = cosine_store.search([1.0, 0.0, 0.0, 0.0], k=1)
        assert results[0].metadata["label"] == "test"


# ---------------------------------------------------------------------------
# count() and ids()
# ---------------------------------------------------------------------------

class TestUtilities:

    def test_count_zero_on_init(self, cosine_store):
        assert cosine_store.count() == 0

    def test_count_increments_on_insert(self, cosine_store):
        cosine_store.insert("v1", [1.0, 0.0, 0.0, 0.0])
        assert cosine_store.count() == 1

    def test_count_does_not_include_deleted(self, cosine_store):
        cosine_store.insert("v1", [1.0, 0.0, 0.0, 0.0])
        cosine_store.delete("v1")
        assert cosine_store.count() == 0

    def test_count_all_includes_deleted(self, cosine_store):
        cosine_store.insert("v1", [1.0, 0.0, 0.0, 0.0])
        cosine_store.delete("v1")
        assert cosine_store.count_all() == 1

    def test_ids_returns_only_live_ids(self, cosine_store):
        cosine_store.insert("v1", [1.0, 0.0, 0.0, 0.0])
        cosine_store.insert("v2", [0.0, 1.0, 0.0, 0.0])
        cosine_store.delete("v1")
        assert cosine_store.ids() == ["v2"]

    def test_ids_returns_sorted(self, cosine_store):
        cosine_store.insert("zebra", [1.0, 0.0, 0.0, 0.0])
        cosine_store.insert("apple", [0.0, 1.0, 0.0, 0.0])
        assert cosine_store.ids() == ["apple", "zebra"]