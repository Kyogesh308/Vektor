"""
vektor.api.models
------------------
Pydantic request and response models. Uses Pydantic v2 syntax throughout.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class CreateCollectionRequest(BaseModel):
    name: str
    dim: int = Field(gt=0)
    metric: str = Field(pattern="^(cosine|dot|euclidean)$")
    mode: str = Field(default="beginner", pattern="^(beginner|research)$")
    M: Optional[int] = None
    ef_construction: Optional[int] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "documents", "dim": 384, "metric": "cosine",
                "mode": "beginner",
            }
        }
    }


class InsertVectorRequest(BaseModel):
    id: str
    vector: list[float]
    metadata: Optional[dict] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "doc_001",
                "vector": [0.1, 0.2, 0.3, 0.4],
                "metadata": {"source": "arxiv"},
            }
        }
    }


class BatchInsertRequest(BaseModel):
    records: list[InsertVectorRequest]


class UpdateVectorRequest(BaseModel):
    vector: Optional[list[float]] = None
    metadata: Optional[dict] = None

    @model_validator(mode="after")
    def check_at_least_one_field(self) -> "UpdateVectorRequest":
        if self.vector is None and self.metadata is None:
            raise ValueError(
                "At least one of 'vector' or 'metadata' must be provided."
            )
        return self


class SearchRequest(BaseModel):
    query: list[float]
    k: int = Field(default=10, gt=0)
    ef: Optional[int] = None
    filters: Optional[dict] = None
    strategy: str = Field(default="post", pattern="^(pre|post)$")
    include_vectors: bool = False
    overfetch_factor: Optional[int] = None

    @model_validator(mode="after")
    def check_ef_geq_k(self) -> "SearchRequest":
        if self.ef is not None and self.ef < self.k:
            raise ValueError(f"ef ({self.ef}) must be >= k ({self.k}).")
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": [0.1, 0.2, 0.3, 0.4], "k": 5,
                "filters": {"source": "arxiv"}, "strategy": "post",
            }
        }
    }


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class SearchResultResponse(BaseModel):
    id: str
    score: Optional[float]
    rank: int
    metadata: dict
    vector: Optional[list[float]] = None


class SearchResponse(BaseModel):
    results: list[SearchResultResponse]
    count: int
    warnings: list[str] = []
    metric: str


class BatchInsertResponse(BaseModel):
    inserted: int
    failed: int
    errors: list[dict] = []


class CollectionResponse(BaseModel):
    name: str
    dim: int
    metric: str
    M: int
    ef_construction: int
    mode: str
    created_at: str


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float


class InfoResponse(BaseModel):
    vektor_version: str
    python_version: str
    uptime_seconds: float
    collections_open: int


# ---------------------------------------------------------------------------
# Response Envelope
# ---------------------------------------------------------------------------

class SuccessEnvelope(BaseModel):
    status: str = "success"
    data: dict
    warnings: list[str] = []


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorEnvelope(BaseModel):
    status: str = "error"
    error: ErrorDetail