"""
Phase 12 — FastAPI Integration Tests

Uses TestClient with a fresh temp data_dir per test — no shared state.
Tests every endpoint: happy path, validation failure (422), and every
mapped Vektor exception (correct HTTP status + error code).
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from vektor.api.app import create_app


@pytest.fixture
def client(tmp_path):
    app = create_app(data_dir=str(tmp_path / "vektor_data"))
    with TestClient(app) as c:
        yield c


def make_vec(dim=8, val=1.0):
    return [val] * dim


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class TestHealth:

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_info_returns_200(self, client):
        resp = client.get("/v1/info")
        assert resp.status_code == 200
        assert "vektor_version" in resp.json()


# ---------------------------------------------------------------------------
# Collections
# ---------------------------------------------------------------------------

class TestCollectionEndpoints:

    def test_create_collection_201(self, client):
        resp = client.post("/v1/collections", json={
            "name": "docs", "dim": 8, "metric": "cosine",
        })
        assert resp.status_code == 201
        assert resp.json()["name"] == "docs"

    def test_create_duplicate_collection_409(self, client):
        client.post("/v1/collections", json={"name": "dup", "dim": 8, "metric": "cosine"})
        resp = client.post("/v1/collections", json={"name": "dup", "dim": 8, "metric": "cosine"})
        assert resp.status_code == 409
        assert resp.json()["error"]["code"] == "COLLECTION_ALREADY_EXISTS"

    def test_create_invalid_metric_422(self, client):
        resp = client.post("/v1/collections", json={
            "name": "bad", "dim": 8, "metric": "manhattan",
        })
        assert resp.status_code == 422

    def test_list_collections(self, client):
        client.post("/v1/collections", json={"name": "a", "dim": 8, "metric": "cosine"})
        client.post("/v1/collections", json={"name": "b", "dim": 8, "metric": "cosine"})
        resp = client.get("/v1/collections")
        assert resp.status_code == 200
        assert set(resp.json()) >= {"a", "b"}

    def test_get_collection_404(self, client):
        resp = client.get("/v1/collections/nonexistent")
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "COLLECTION_NOT_FOUND"

    def test_delete_collection_204(self, client):
        client.post("/v1/collections", json={"name": "del_me", "dim": 8, "metric": "cosine"})
        resp = client.delete("/v1/collections/del_me")
        assert resp.status_code == 204

    def test_delete_nonexistent_collection_404(self, client):
        resp = client.delete("/v1/collections/nope")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Vectors
# ---------------------------------------------------------------------------

class TestVectorEndpoints:

    def _create_collection(self, client, name="vecs", dim=8):
        client.post("/v1/collections", json={"name": name, "dim": dim, "metric": "cosine"})

    def test_insert_vector_201(self, client):
        self._create_collection(client)
        resp = client.post("/v1/collections/vecs/vectors", json={
            "id": "v1", "vector": make_vec(), "metadata": {"tag": "test"},
        })
        assert resp.status_code == 201

    def test_insert_duplicate_id_409(self, client):
        self._create_collection(client)
        client.post("/v1/collections/vecs/vectors", json={"id": "v1", "vector": make_vec()})
        resp = client.post("/v1/collections/vecs/vectors", json={"id": "v1", "vector": make_vec()})
        assert resp.status_code == 409
        assert resp.json()["error"]["code"] == "DUPLICATE_ID"

    def test_insert_wrong_dimension_422(self, client):
        self._create_collection(client, dim=8)
        resp = client.post("/v1/collections/vecs/vectors", json={
            "id": "v1", "vector": [1.0, 2.0],
        })
        assert resp.status_code == 422

    def test_get_vector_200(self, client):
        self._create_collection(client)
        client.post("/v1/collections/vecs/vectors", json={
            "id": "v1", "vector": make_vec(), "metadata": {"tag": "test"},
        })
        resp = client.get("/v1/collections/vecs/vectors/v1")
        assert resp.status_code == 200
        assert resp.json()["metadata"]["tag"] == "test"

    def test_get_vector_404(self, client):
        self._create_collection(client)
        resp = client.get("/v1/collections/vecs/vectors/nonexistent")
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "VECTOR_NOT_FOUND"

    def test_get_vector_with_include_vector(self, client):
        self._create_collection(client)
        client.post("/v1/collections/vecs/vectors", json={"id": "v1", "vector": make_vec()})
        resp = client.get("/v1/collections/vecs/vectors/v1?include_vector=true")
        assert resp.status_code == 200
        assert resp.json()["vector"] is not None
        assert isinstance(resp.json()["vector"], list)

    def test_update_vector_200(self, client):
        self._create_collection(client)
        client.post("/v1/collections/vecs/vectors", json={"id": "v1", "vector": make_vec()})
        resp = client.put("/v1/collections/vecs/vectors/v1", json={
            "metadata": {"updated": True},
        })
        assert resp.status_code == 200

    def test_update_both_none_422(self, client):
        self._create_collection(client)
        client.post("/v1/collections/vecs/vectors", json={"id": "v1", "vector": make_vec()})
        resp = client.put("/v1/collections/vecs/vectors/v1", json={})
        assert resp.status_code == 422

    def test_delete_vector_204(self, client):
        self._create_collection(client)
        client.post("/v1/collections/vecs/vectors", json={"id": "v1", "vector": make_vec()})
        resp = client.delete("/v1/collections/vecs/vectors/v1")
        assert resp.status_code == 204

    def test_delete_vector_404(self, client):
        self._create_collection(client)
        resp = client.delete("/v1/collections/vecs/vectors/nonexistent")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Batch Insert
# ---------------------------------------------------------------------------

class TestBatchInsert:

    def _create_collection(self, client, name="batch_col", dim=8):
        client.post("/v1/collections", json={"name": name, "dim": dim, "metric": "cosine"})

    def test_batch_all_valid_207(self, client):
        self._create_collection(client)
        records = [{"id": f"v{i}", "vector": make_vec()} for i in range(10)]
        resp = client.post("/v1/collections/batch_col/vectors/batch", json={"records": records})
        assert resp.status_code == 207
        assert resp.json()["inserted"] == 10
        assert resp.json()["failed"] == 0

    def test_batch_partial_failure_207(self, client):
        self._create_collection(client, dim=8)
        records = [{"id": f"v{i}", "vector": make_vec(dim=8)} for i in range(9)]
        records.append({"id": "bad", "vector": [1.0, 2.0]})  # wrong dim
        resp = client.post("/v1/collections/batch_col/vectors/batch", json={"records": records})
        assert resp.status_code == 207
        assert resp.json()["inserted"] == 9
        assert resp.json()["failed"] == 1

    def test_batch_all_invalid_207(self, client):
        self._create_collection(client, dim=8)
        records = [{"id": "bad1", "vector": [1.0]}, {"id": "bad2", "vector": [1.0]}]
        resp = client.post("/v1/collections/batch_col/vectors/batch", json={"records": records})
        assert resp.status_code == 207
        assert resp.json()["inserted"] == 0
        assert resp.json()["failed"] == 2

    def test_batch_route_not_shadowed_by_id_route(self, client):
        """
        Regression test: /vectors/batch must not be routed to /vectors/{id}
        with id='batch'. This only passes if batch is registered first.
        """
        self._create_collection(client)
        resp = client.post("/v1/collections/batch_col/vectors/batch", json={"records": []})
        assert resp.status_code == 207  # NOT 404 (which {id} lookup would give)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

class TestSearchEndpoint:

    def _setup(self, client, name="search_col", n=20, dim=8):
        client.post("/v1/collections", json={"name": name, "dim": dim, "metric": "euclidean"})
        rng = np.random.default_rng(42)
        for i in range(n):
            v = rng.standard_normal(dim).astype(np.float32).tolist()
            client.post(f"/v1/collections/{name}/vectors", json={
                "id": f"v{i}", "vector": v, "metadata": {"idx": i},
            })

    def test_search_200(self, client):
        self._setup(client)
        resp = client.post("/v1/collections/search_col/search", json={
            "query": make_vec(), "k": 5,
        })
        assert resp.status_code == 200
        assert len(resp.json()["results"]) == 5

    def test_search_ef_less_than_k_422(self, client):
        self._setup(client)
        resp = client.post("/v1/collections/search_col/search", json={
            "query": make_vec(), "k": 10, "ef": 5,
        })
        assert resp.status_code == 422

    def test_search_include_vectors_serializes(self, client):
        self._setup(client)
        resp = client.post("/v1/collections/search_col/search", json={
            "query": make_vec(), "k": 3, "include_vectors": True,
        })
        assert resp.status_code == 200
        for r in resp.json()["results"]:
            assert isinstance(r["vector"], list)

    def test_search_response_includes_metric(self, client):
        self._setup(client)
        resp = client.post("/v1/collections/search_col/search", json={
            "query": make_vec(), "k": 3,
        })
        assert resp.json()["metric"] == "euclidean"

    def test_search_postfilter_under_k_warning(self, client):
        client.post("/v1/collections", json={
            "name": "warn_col", "dim": 8, "metric": "euclidean", "mode": "research",
        })
        rng = np.random.default_rng(1)
        for i in range(50):
            v = rng.standard_normal(8).astype(np.float32).tolist()
            tag = "rare" if i < 2 else "common"
            client.post("/v1/collections/warn_col/vectors", json={
                "id": f"v{i}", "vector": v, "metadata": {"tag": tag},
            })

        resp = client.post("/v1/collections/warn_col/search", json={
            "query": make_vec(), "k": 10, "ef": 50,
            "filters": {"tag": "rare"}, "strategy": "post",
        })
        assert resp.status_code == 200
        assert len(resp.json()["warnings"]) > 0