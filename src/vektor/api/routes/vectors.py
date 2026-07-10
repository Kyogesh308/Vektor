"""
vektor.api.routes.vectors
----------------------------
Vector CRUD endpoints.

CRITICAL: /vectors/batch is registered BEFORE /vectors/{id}.
FastAPI matches routes in registration order. If {id} came first,
POST /vectors/batch would match with id="batch" instead of hitting
the batch handler.
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, Response

from vektor import Vektor
from vektor.api.dependencies import get_vektor
from vektor.api.models import (
    InsertVectorRequest, BatchInsertRequest, BatchInsertResponse,
    UpdateVectorRequest,
)
from vektor.api.utils import to_json_safe

router = APIRouter(tags=["vectors"])


# ---------------------------------------------------------------------------
# Batch — MUST come before /{id} routes
# ---------------------------------------------------------------------------

@router.post(
    "/collections/{name}/vectors/batch",
    response_model=BatchInsertResponse,
    status_code=207,
)
def batch_insert(
    name: str, req: BatchInsertRequest, client: Vektor = Depends(get_vektor),
) -> BatchInsertResponse:
    """
    Insert multiple vectors. Returns 207 Multi-Status — individual
    records may succeed or fail independently.
    """
    col = client.get_collection(name)
    records = [
        {"id": r.id, "vector": r.vector, "metadata": r.metadata}
        for r in req.records
    ]
    result = col.batch_insert(records)
    return BatchInsertResponse(
        inserted=result.inserted, failed=result.failed, errors=result.errors,
    )


# ---------------------------------------------------------------------------
# Single vector CRUD
# ---------------------------------------------------------------------------

@router.post("/collections/{name}/vectors", status_code=201)
def insert_vector(
    name: str, req: InsertVectorRequest, client: Vektor = Depends(get_vektor),
) -> dict:
    """Insert a single vector."""
    col = client.get_collection(name)
    col.insert(req.id, req.vector, req.metadata)
    return {"id": req.id}


@router.get("/collections/{name}/vectors/{id}")
def get_vector(
    name: str, id: str, include_vector: bool = False,
    client: Vektor = Depends(get_vektor),
) -> dict:
    """Fetch a single vector by ID."""
    col = client.get_collection(name)
    result = col.get(id, include_vector=include_vector)
    return to_json_safe({
        "id": result.id, "metadata": result.metadata,
        "vector": result.vector,
    })


@router.put("/collections/{name}/vectors/{id}")
def update_vector(
    name: str, id: str, req: UpdateVectorRequest,
    client: Vektor = Depends(get_vektor),
) -> dict:
    """Update a vector's data and/or metadata."""
    col = client.get_collection(name)
    col.update(id, vector=req.vector, metadata=req.metadata)
    return {"id": id, "updated": True}


@router.delete("/collections/{name}/vectors/{id}", status_code=204)
def delete_vector(
    name: str, id: str, client: Vektor = Depends(get_vektor),
) -> Response:
    """Delete (tombstone) a vector."""
    col = client.get_collection(name)
    col.delete(id)
    return Response(status_code=204)