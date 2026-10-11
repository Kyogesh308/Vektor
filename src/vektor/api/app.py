"""
vektor.api.app
----------------
FastAPI application factory with lifespan management.
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from vektor import Vektor, VektorConfig
from vektor.api.exceptions import register_exception_handlers
from vektor.api.routes import collections, vectors, search, health


def create_app(data_dir: str | None = None) -> FastAPI:
    # Prefer env var (set to /data in Docker), then argument, then default
    data_dir = data_dir or os.environ.get("VEKTOR_DATA_DIR", "./vektor_data")
    """
    Application factory. Called by the CLI and by tests (with a temp data_dir).

    Args:
        data_dir: Directory for Vektor's collection storage.

    Returns:
        Configured FastAPI application.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.start_time = time.perf_counter()
        app.state.vektor_client = Vektor(VektorConfig(data_dir=Path(data_dir)))
        yield
        app.state.vektor_client.close()

    app = FastAPI(
        title="Vektor",
        description="A production-style vector database built from scratch.",
        version="0.5.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            # Streamlit UI origins (local and Docker)
            "http://localhost:8501",
            "http://127.0.0.1:8501",
            "http://vektor-ui:8501",
            # Allow all — safe behind Docker bridge; tighten for prod
            "*",
        ],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)

    app.include_router(health.router)
    app.include_router(collections.router, prefix="/v1")
    app.include_router(vectors.router, prefix="/v1")
    app.include_router(search.router, prefix="/v1")

    return app


# Default app instance for `uvicorn vektor.api.app:app`
# data_dir is resolved from VEKTOR_DATA_DIR env var inside create_app()
app = create_app()