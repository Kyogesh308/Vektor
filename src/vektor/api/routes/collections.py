"""
vektor.api.routes.collections
--------------------------------
Collection management endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response

from vektor import Vektor
from vektor.api.dependencies import get_vektor
from vektor.api.models import CreateCollectionRequest, CollectionResponse

router = APIRouter(tags=["collections"])


@router.post("/collections", response_model=CollectionResponse, status_code=201)
def create_collection(
    req: CreateCollectionRequest,
    client: Vektor = Depends(get_vektor),
) -> CollectionResponse:
    """Create a new named collection."""
    col = client.create_collection(
        name=req.name, dim=req.dim, metric=req.metric,
        mode=req.mode, M=req.M, ef_construction=req.ef_construction,
    )
    cfg = col.config
    return CollectionResponse(
        name=cfg.name, dim=cfg.dim, metric=cfg.metric, M=cfg.M,
        ef_construction=cfg.ef_construction, mode=cfg.mode,
        created_at=cfg.created_at,
    )


@router.get("/collections", response_model=list[str])
def list_collections(client: Vektor = Depends(get_vektor)) -> list[str]:
    """List all collection names."""
    return client.list_collections()


@router.get("/collections/{name}", response_model=CollectionResponse)
def get_collection(name: str, client: Vektor = Depends(get_vektor)) -> CollectionResponse:
    """Get a collection's configuration."""
    col = client.get_collection(name)
    cfg = col.config
    return CollectionResponse(
        name=cfg.name, dim=cfg.dim, metric=cfg.metric, M=cfg.M,
        ef_construction=cfg.ef_construction, mode=cfg.mode,
        created_at=cfg.created_at,
    )


@router.delete("/collections/{name}", status_code=204)
def delete_collection(name: str, client: Vektor = Depends(get_vektor)) -> Response:
    """Delete a collection and all its data."""
    client.delete_collection(name)
    return Response(status_code=204)


@router.get("/collections/{name}/count")
def count_vectors(name: str, client: Vektor = Depends(get_vektor)) -> dict:
    """Count live vectors in a collection."""
    col = client.get_collection(name)
    return {"count": col.count()}


@router.get("/collections/{name}/estimate-memory")
def estimate_memory(
    name: str, n_vectors: int, client: Vektor = Depends(get_vektor),
) -> dict:
    """Estimate memory footprint for a hypothetical vector count."""
    col = client.get_collection(name)
    est = col.estimate_memory(n_vectors)
    return {
        "graph_bytes": est.graph_bytes, "vector_bytes": est.vector_bytes,
        "metadata_bytes": est.metadata_bytes, "total_bytes": est.total_bytes,
        "total_mb": est.total_mb, "warning": est.warning,
    }