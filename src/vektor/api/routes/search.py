"""
vektor.api.routes.search
---------------------------
Search endpoint. POST, never GET — query vectors are too large for a URL.
"""

from __future__ import annotations

import warnings

from fastapi import APIRouter, Depends

from vektor import Vektor
from vektor.api.dependencies import get_vektor
from vektor.api.models import SearchRequest, SearchResponse, SearchResultResponse
from vektor.api.utils import to_json_safe

router = APIRouter(tags=["search"])


@router.post("/collections/{name}/search", response_model=SearchResponse)
def search(
    name: str, req: SearchRequest, client: Vektor = Depends(get_vektor),
) -> SearchResponse:
    """
    Search for the k nearest neighbours of the query vector.

    Never GET — query vectors are too large to encode safely in a URL.
    """
    col = client.get_collection(name)

    captured_warnings = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = col.search(
            query=req.query, k=req.k, ef=req.ef, filters=req.filters,
            strategy=req.strategy, include_vectors=req.include_vectors,
        )
        captured_warnings = [str(w.message) for w in caught]

    result_responses = [
        SearchResultResponse(
            id=r.id, score=r.score, rank=r.rank, metadata=r.metadata,
            vector=to_json_safe(r.vector) if r.vector is not None else None,
        )
        for r in results
    ]

    return SearchResponse(
        results=result_responses, count=len(result_responses),
        warnings=captured_warnings, metric=col.config.metric,
    )