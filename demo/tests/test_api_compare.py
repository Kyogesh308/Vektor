import pytest
from fastapi.testclient import TestClient

from vektor.api.app import create_app
from vektor.api.dependencies import get_vektor
from vektor import Vektor, VektorConfig


@pytest.fixture
def test_app(tmp_path):
    config = VektorConfig(data_dir=tmp_path / "vektor_data")
    client = Vektor(config)
    
    app = create_app(data_dir=str(tmp_path / "vektor_data"))
    app.dependency_overrides[get_vektor] = lambda: client

    with TestClient(app) as test_client:
        yield test_client, client

    client.close()


def test_compare_endpoint(test_app):
    api_client, vektor_client = test_app

    # Create a small collection
    api_client.post(
        "/v1/collections",
        json={"name": "demo_compare", "dim": 2, "metric": "euclidean"}
    )

    # Insert some vectors
    api_client.post(
        "/v1/collections/demo_compare/vectors/batch",
        json={"records": [
            {"id": "v1", "vector": [1.0, 1.0]},
            {"id": "v2", "vector": [2.0, 2.0]},
            {"id": "v3", "vector": [10.0, 10.0]},
        ]}
    )

    # Search compare
    response = api_client.post(
        "/v1/collections/demo_compare/search/compare",
        json={"query": [1.1, 1.1], "k": 2}
    )
    
    assert response.status_code == 200
    data = response.json()

    assert "brute_force_results" in data
    assert "hnsw_results" in data
    assert "recall_at_k" in data
    assert "brute_force_latency_ms" in data
    assert "hnsw_latency_ms" in data
    assert "speedup_ratio" in data

    bf_results = data["brute_force_results"]
    assert len(bf_results) == 2
    assert bf_results[0]["id"] == "v1"
    assert bf_results[1]["id"] == "v2"

    assert data["recall_at_k"] <= 1.0
    assert data["recall_at_k"] >= 0.0

    assert data["brute_force_latency_ms"] >= 0.0
    assert data["hnsw_latency_ms"] >= 0.0
