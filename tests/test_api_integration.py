"""
Phase 11 — Public API Integration Tests

CRITICAL: this file must import ONLY from `vektor`. If it needs an
internal module, the public API is incomplete — fix the API, not this test.
"""

from __future__ import annotations

import numpy as np
import pytest

from vektor import (
    Vektor, VektorConfig, SearchResult,
    VectorNotFoundError, DuplicateIDError,
    CollectionAlreadyExistsError, CollectionNotFoundError,
    VektorConfigError, InvalidEFError,
)


@pytest.fixture
def client(tmp_path):
    config = VektorConfig(data_dir=tmp_path / "vektor_data")
    c = Vektor(config)
    yield c
    c.close()


def make_vecs(n: int, dim: int, seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(dim).astype(np.float32).tolist() for _ in range(n)]


class TestFullLifecycle:

    def test_create_insert_search_get_update_delete(self, client):
        col = client.create_collection("docs", dim=16, metric="cosine")
        vecs = make_vecs(100, 16, seed=0)

        for i, v in enumerate(vecs):
            col.insert(f"doc{i}", v, {"index": i})

        assert col.count() == 100

        results = col.search(vecs[0], k=5)
        assert len(results) == 5
        assert isinstance(results[0], SearchResult)
        assert results[0].id == "doc0"

        fetched = col.get("doc0")
        assert fetched.metadata["index"] == 0

        col.update("doc0", metadata={"index": 0, "updated": True})
        assert col.get("doc0").metadata["updated"] is True

        col.delete("doc0")
        with pytest.raises(VectorNotFoundError):
            col.get("doc0")


class TestBatchInsert:

    def test_batch_insert_all_valid(self, client):
        col = client.create_collection("batch_test", dim=8, metric="euclidean")
        vecs = make_vecs(1000, 8, seed=1)
        records = [{"id": f"v{i}", "vector": v} for i, v in enumerate(vecs)]

        result = col.batch_insert(records)
        assert result.inserted == 1000
        assert result.failed == 0
        assert result.errors == []

    def test_batch_insert_one_bad_record(self, client):
        col = client.create_collection("batch_bad", dim=8, metric="euclidean")
        vecs = make_vecs(999, 8, seed=2)
        records = [{"id": f"v{i}", "vector": v} for i, v in enumerate(vecs)]
        records.append({"id": "bad", "vector": [1.0, 2.0]})

        result = col.batch_insert(records)
        assert result.inserted == 999
        assert result.failed == 1
        assert result.errors[0]["id"] == "bad"


class TestFilteredSearch:

    def test_prefilter_all_results_satisfy_filter(self, client):
        col = client.create_collection("filtered", dim=8, metric="euclidean",
                                        mode="research")
        vecs = make_vecs(200, 8, seed=3)
        for i, v in enumerate(vecs):
            partition = "A" if i < 100 else "B"
            col.insert(f"v{i}", v, {"partition": partition})

        results = col.search(vecs[0], k=10, ef=50,
                             filters={"partition": "A"}, strategy="pre")
        for r in results:
            assert r.metadata["partition"] == "A"


class TestBeginnerMode:

    def test_cannot_override_m_in_beginner_mode(self, client):
        with pytest.raises(VektorConfigError):
            client.create_collection("beg", dim=8, metric="cosine",
                                     mode="beginner", M=32)

    def test_search_works_with_defaults(self, client):
        col = client.create_collection("beg2", dim=8, metric="cosine")
        vecs = make_vecs(50, 8, seed=4)
        for i, v in enumerate(vecs):
            col.insert(f"v{i}", v)
        results = col.search(vecs[0], k=5)
        assert len(results) == 5


class TestResearchMode:

    def test_ef_per_query_override(self, client):
        col = client.create_collection("res", dim=8, metric="cosine",
                                        mode="research", M=16, ef_construction=100)
        vecs = make_vecs(50, 8, seed=5)
        for i, v in enumerate(vecs):
            col.insert(f"v{i}", v)
        results = col.search(vecs[0], k=5, ef=200)
        assert len(results) == 5

    def test_ef_less_than_k_raises(self, client):
        col = client.create_collection("res2", dim=8, metric="cosine",
                                        mode="research")
        col.insert("v0", make_vecs(1, 8, seed=6)[0])
        with pytest.raises(InvalidEFError):
            col.search(make_vecs(1, 8, seed=7)[0], k=10, ef=5)


class TestPersistenceRoundTrip:

    def test_close_reopen_identical_search(self, tmp_path):
        config = VektorConfig(data_dir=tmp_path / "persist_test")

        client1 = Vektor(config)
        col1 = client1.create_collection("persist", dim=8, metric="euclidean")
        vecs = make_vecs(200, 8, seed=8)
        for i, v in enumerate(vecs):
            col1.insert(f"v{i}", v)

        query = vecs[0]
        results_before = col1.search(query, k=10)
        ids_before = [r.id for r in results_before]
        client1.close()

        client2 = Vektor(config)
        col2 = client2.get_collection("persist")
        results_after = col2.search(query, k=10)
        ids_after = [r.id for r in results_after]
        client2.close()

        assert ids_before == ids_after


class TestMemoryEstimate:

    def test_returns_positive_values(self, client):
        col = client.create_collection("mem_test", dim=768, metric="cosine")
        est = col.estimate_memory(10_000)
        assert est.graph_bytes > 0
        assert est.vector_bytes > 0
        assert est.metadata_bytes > 0
        assert est.total_bytes == est.graph_bytes + est.vector_bytes + est.metadata_bytes
        assert est.total_mb == pytest.approx(est.total_bytes / 1_000_000)


class TestErrorCases:

    def test_vector_not_found(self, client):
        col = client.create_collection("err1", dim=8, metric="cosine")
        with pytest.raises(VectorNotFoundError):
            col.get("nonexistent")

    def test_duplicate_id(self, client):
        col = client.create_collection("err2", dim=8, metric="cosine")
        v = make_vecs(1, 8, seed=9)[0]
        col.insert("v1", v)
        with pytest.raises(DuplicateIDError):
            col.insert("v1", v)

    def test_duplicate_collection(self, client):
        client.create_collection("err3", dim=8, metric="cosine")
        with pytest.raises(CollectionAlreadyExistsError):
            client.create_collection("err3", dim=16, metric="euclidean")

    def test_collection_not_found(self, client):
        with pytest.raises(CollectionNotFoundError):
            client.get_collection("does_not_exist")


class TestCollectionRegistry:

    def test_get_collection_twice_returns_same_object(self, client):
        col1 = client.create_collection("registry_test", dim=8, metric="cosine")
        col2 = client.get_collection("registry_test")
        assert col1 is col2