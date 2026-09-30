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
from vektor.api.models import SearchRequest, SearchResponse, SearchResultResponse, CompareResponse
from vektor.api.utils import to_json_safe
import time

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


@router.post("/collections/{name}/search/compare", response_model=CompareResponse)
def search_compare(
    name: str, req: SearchRequest, client: Vektor = Depends(get_vektor),
) -> CompareResponse:
    """
    Compare brute-force and HNSW search results and latencies for a single query.
    Used exclusively for the comparison demo UI.
    """
    col = client.get_collection(name)

    # Brute-force search
    t0 = time.perf_counter()
    bf_raw_results = col._store.search(query=req.query, k=req.k)
    t1 = time.perf_counter()
    bf_latency = (t1 - t0) * 1000

    bf_responses = []
    bf_ids = []
    for rank, r in enumerate(bf_raw_results, start=1):
        bf_ids.append(r.id)
        # Note: _store.search doesn't include vectors. We can fetch them if include_vectors is true.
        vector_val = None
        if req.include_vectors:
            vector_val = to_json_safe(col._store._vectors.get(r.id))
        bf_responses.append(SearchResultResponse(
            id=r.id, score=r.score, rank=rank, metadata=r.metadata, vector=vector_val
        ))

    # HNSW search
    t2 = time.perf_counter()
    hnsw_results = col.search(
        query=req.query, k=req.k, ef=req.ef, filters=req.filters,
        strategy=req.strategy, include_vectors=req.include_vectors,
    )
    t3 = time.perf_counter()
    hnsw_latency = (t3 - t2) * 1000

    hnsw_responses = [
        SearchResultResponse(
            id=r.id, score=r.score, rank=r.rank, metadata=r.metadata,
            vector=to_json_safe(r.vector) if r.vector is not None else None,
        )
        for r in hnsw_results
    ]

    hnsw_ids = [r.id for r in hnsw_results]

    if not bf_ids:
        recall = 1.0 if not hnsw_ids else 0.0
    else:
        hits = sum(1 for hid in hnsw_ids if hid in bf_ids)
        recall = hits / len(bf_ids)

    speedup = (bf_latency / hnsw_latency) if hnsw_latency > 0 else 0.0

    return CompareResponse(
        brute_force_results=bf_responses,
        hnsw_results=hnsw_responses,
        recall_at_k=recall,
        brute_force_latency_ms=bf_latency,
        hnsw_latency_ms=hnsw_latency,
        speedup_ratio=speedup
    )