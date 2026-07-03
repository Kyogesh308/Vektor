"""
Phase 3 — Collection Manager Tests

Coverage:
- Happy path for every public method
- Every named exception, triggered by its specific failure condition
- Isolation: operations on one collection never affect another
- HNSW parameter boundary values (min, max, out-of-range)
- State consistency after sequences of operations
"""

import pytest

from vektor.collection import (
    CollectionManager,
    Collection,
    CollectionAlreadyExistsError,
    CollectionNotFoundError,
    InvalidCollectionNameError,
    InvalidDimensionError,
    InvalidHNSWParameterError,
    VektorCollectionError,
    COLLECTION_NAME_MAX_LENGTH,
    M_MIN, M_MAX,
    EF_CONSTRUCTION_MIN, EF_CONSTRUCTION_MAX,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def manager():
    """A fresh CollectionManager for each test. No shared state between tests."""
    return CollectionManager()


@pytest.fixture
def populated_manager(manager):
    """A manager with two pre-created collections."""
    manager.create("documents", dimension=1536, metric="cosine")
    manager.create("images", dimension=512, metric="euclidean")
    return manager


# ---------------------------------------------------------------------------
# create()
# ---------------------------------------------------------------------------

class TestCreate:

    def test_returns_collection_object(self, manager):
        col = manager.create("test", dimension=128, metric="cosine")
        assert isinstance(col, Collection)

    def test_collection_has_correct_fields(self, manager):
        col = manager.create("test", dimension=256, metric="dot")
        assert col.name == "test"
        assert col.dimension == 256
        assert col.metric == "dot"

    def test_default_hnsw_parameters_applied(self, manager):
        col = manager.create("test", dimension=128, metric="cosine")
        assert col.m == 16
        assert col.ef_construction == 200

    def test_custom_hnsw_parameters_stored(self, manager):
        col = manager.create("test", dimension=128, metric="cosine", m=32, ef_construction=400)
        assert col.m == 32
        assert col.ef_construction == 400

    def test_all_three_metrics_accepted(self, manager):
        manager.create("a", dimension=64, metric="cosine")
        manager.create("b", dimension=64, metric="euclidean")
        manager.create("c", dimension=64, metric="dot")
        assert manager.count() == 3

    def test_duplicate_name_raises(self, manager):
        manager.create("test", dimension=128, metric="cosine")
        with pytest.raises(CollectionAlreadyExistsError):
            manager.create("test", dimension=256, metric="dot")

    def test_duplicate_name_does_not_overwrite(self, manager):
        manager.create("test", dimension=128, metric="cosine")
        with pytest.raises(CollectionAlreadyExistsError):
            manager.create("test", dimension=999, metric="dot")
        # Original collection must be unchanged
        assert manager.get("test").dimension == 128

    def test_empty_name_raises(self, manager):
        with pytest.raises(InvalidCollectionNameError):
            manager.create("", dimension=128, metric="cosine")

    def test_name_at_max_length_is_valid(self, manager):
        long_name = "a" * COLLECTION_NAME_MAX_LENGTH
        col = manager.create(long_name, dimension=128, metric="cosine")
        assert col.name == long_name

    def test_name_exceeding_max_length_raises(self, manager):
        too_long = "a" * (COLLECTION_NAME_MAX_LENGTH + 1)
        with pytest.raises(InvalidCollectionNameError):
            manager.create(too_long, dimension=128, metric="cosine")

    def test_non_string_name_raises(self, manager):
        with pytest.raises(InvalidCollectionNameError):
            manager.create(123, dimension=128, metric="cosine")

    def test_zero_dimension_raises(self, manager):
        with pytest.raises(InvalidDimensionError):
            manager.create("test", dimension=0, metric="cosine")

    def test_negative_dimension_raises(self, manager):
        with pytest.raises(InvalidDimensionError):
            manager.create("test", dimension=-1, metric="cosine")

    def test_float_dimension_raises(self, manager):
        with pytest.raises(InvalidDimensionError):
            manager.create("test", dimension=128.0, metric="cosine")

    def test_bool_dimension_raises(self, manager):
        # bool is a subclass of int in Python — must be explicitly rejected
        with pytest.raises(InvalidDimensionError):
            manager.create("test", dimension=True, metric="cosine")

    def test_unknown_metric_raises(self, manager):
        with pytest.raises(VektorCollectionError):
            manager.create("test", dimension=128, metric="manhattan")

    def test_m_at_minimum_boundary(self, manager):
        col = manager.create("test", dimension=128, metric="cosine", m=M_MIN)
        assert col.m == M_MIN

    def test_m_at_maximum_boundary(self, manager):
        col = manager.create("test", dimension=128, metric="cosine", m=M_MAX)
        assert col.m == M_MAX

    def test_m_below_minimum_raises(self, manager):
        with pytest.raises(InvalidHNSWParameterError):
            manager.create("test", dimension=128, metric="cosine", m=M_MIN - 1)

    def test_m_above_maximum_raises(self, manager):
        with pytest.raises(InvalidHNSWParameterError):
            manager.create("test", dimension=128, metric="cosine", m=M_MAX + 1)

    def test_ef_construction_at_minimum_boundary(self, manager):
        col = manager.create("test", dimension=128, metric="cosine",
                             ef_construction=EF_CONSTRUCTION_MIN)
        assert col.ef_construction == EF_CONSTRUCTION_MIN

    def test_ef_construction_at_maximum_boundary(self, manager):
        col = manager.create("test", dimension=128, metric="cosine",
                             ef_construction=EF_CONSTRUCTION_MAX)
        assert col.ef_construction == EF_CONSTRUCTION_MAX

    def test_ef_construction_below_minimum_raises(self, manager):
        with pytest.raises(InvalidHNSWParameterError):
            manager.create("test", dimension=128, metric="cosine",
                           ef_construction=EF_CONSTRUCTION_MIN - 1)

    def test_ef_construction_above_maximum_raises(self, manager):
        with pytest.raises(InvalidHNSWParameterError):
            manager.create("test", dimension=128, metric="cosine",
                           ef_construction=EF_CONSTRUCTION_MAX + 1)


# ---------------------------------------------------------------------------
# get()
# ---------------------------------------------------------------------------

class TestGet:

    def test_returns_correct_collection(self, populated_manager):
        col = populated_manager.get("documents")
        assert col.name == "documents"
        assert col.dimension == 1536
        assert col.metric == "cosine"

    def test_returns_second_collection_correctly(self, populated_manager):
        col = populated_manager.get("images")
        assert col.dimension == 512
        assert col.metric == "euclidean"

    def test_nonexistent_name_raises(self, manager):
        with pytest.raises(CollectionNotFoundError):
            manager.get("does_not_exist")

    def test_empty_manager_raises(self, manager):
        with pytest.raises(CollectionNotFoundError):
            manager.get("anything")

    def test_get_after_delete_raises(self, manager):
        manager.create("temp", dimension=64, metric="dot")
        manager.delete("temp")
        with pytest.raises(CollectionNotFoundError):
            manager.get("temp")


# ---------------------------------------------------------------------------
# delete()
# ---------------------------------------------------------------------------

class TestDelete:

    def test_delete_removes_collection(self, manager):
        manager.create("test", dimension=128, metric="cosine")
        manager.delete("test")
        assert not manager.exists("test")

    def test_delete_nonexistent_raises(self, manager):
        with pytest.raises(CollectionNotFoundError):
            manager.delete("ghost")

    def test_delete_does_not_affect_other_collections(self, populated_manager):
        populated_manager.delete("documents")
        # images must still exist and be unaffected
        col = populated_manager.get("images")
        assert col.dimension == 512

    def test_count_decreases_after_delete(self, populated_manager):
        before = populated_manager.count()
        populated_manager.delete("documents")
        assert populated_manager.count() == before - 1

    def test_deleted_name_can_be_recreated(self, manager):
        manager.create("test", dimension=128, metric="cosine")
        manager.delete("test")
        # Same name, different dimension — must succeed
        col = manager.create("test", dimension=256, metric="dot")
        assert col.dimension == 256


# ---------------------------------------------------------------------------
# list_collections()
# ---------------------------------------------------------------------------

class TestListCollections:

    def test_empty_manager_returns_empty_list(self, manager):
        assert manager.list_collections() == []

    def test_returns_all_names(self, populated_manager):
        names = populated_manager.list_collections()
        assert "documents" in names
        assert "images" in names

    def test_returns_sorted_alphabetically(self, manager):
        manager.create("zebra", dimension=64, metric="cosine")
        manager.create("apple", dimension=64, metric="cosine")
        manager.create("mango", dimension=64, metric="cosine")
        assert manager.list_collections() == ["apple", "mango", "zebra"]

    def test_list_updates_after_delete(self, populated_manager):
        populated_manager.delete("documents")
        names = populated_manager.list_collections()
        assert "documents" not in names
        assert "images" in names


# ---------------------------------------------------------------------------
# exists() and count()
# ---------------------------------------------------------------------------

class TestExistsAndCount:

    def test_exists_returns_true_for_registered(self, manager):
        manager.create("test", dimension=128, metric="cosine")
        assert manager.exists("test") is True

    def test_exists_returns_false_for_unregistered(self, manager):
        assert manager.exists("missing") is False

    def test_count_is_zero_on_init(self, manager):
        assert manager.count() == 0

    def test_count_increments_on_create(self, manager):
        manager.create("a", dimension=64, metric="cosine")
        assert manager.count() == 1
        manager.create("b", dimension=64, metric="dot")
        assert manager.count() == 2

    def test_count_decrements_on_delete(self, populated_manager):
        populated_manager.delete("images")
        assert populated_manager.count() == 1


# ---------------------------------------------------------------------------
# Collection.to_dict()
# ---------------------------------------------------------------------------

class TestCollectionToDict:

    def test_to_dict_contains_all_fields(self, manager):
        col = manager.create("test", dimension=128, metric="cosine", m=8, ef_construction=100)
        d = col.to_dict()
        assert d["name"] == "test"
        assert d["dimension"] == 128
        assert d["metric"] == "cosine"
        assert d["m"] == 8
        assert d["ef_construction"] == 100

    def test_to_dict_returns_plain_dict(self, manager):
        col = manager.create("test", dimension=64, metric="dot")
        assert isinstance(col.to_dict(), dict)


# ---------------------------------------------------------------------------
# Isolation — cross-collection independence
# ---------------------------------------------------------------------------

class TestIsolation:

    def test_two_collections_with_same_dimension_independent(self, manager):
        manager.create("a", dimension=128, metric="cosine")
        manager.create("b", dimension=128, metric="euclidean")
        assert manager.get("a").metric == "cosine"
        assert manager.get("b").metric == "euclidean"

    def test_failed_create_does_not_corrupt_existing(self, manager):
        manager.create("existing", dimension=128, metric="cosine")
        with pytest.raises(CollectionAlreadyExistsError):
            manager.create("existing", dimension=999, metric="dot")
        # existing must be unchanged
        assert manager.get("existing").dimension == 128
        assert manager.count() == 1