"""
vektor.api.routes.health
--------------------------
Health and info endpoints. No Vektor work — safe to be async def.
"""

from __future__ import annotations

import sys
import time
from importlib.metadata import version, PackageNotFoundError

from fastapi import APIRouter, Request

from vektor.api.models import HealthResponse, InfoResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Liveness check for monitoring systems. No Vektor work performed."""
    uptime = time.perf_counter() - request.app.state.start_time
    return HealthResponse(status="ok", uptime_seconds=uptime)


@router.get("/v1/info", response_model=InfoResponse)
async def info(request: Request) -> InfoResponse:
    """Server and package version info."""
    try:
        vektor_version = version("vektor-db")
    except PackageNotFoundError:
        vektor_version = "0.5.0"  # dev install fallback

    uptime = time.perf_counter() - request.app.state.start_time
    client = request.app.state.vektor_client

    return InfoResponse(
        vektor_version=vektor_version,
        python_version=sys.version.split()[0],
        uptime_seconds=uptime,
        collections_open=len(client._open_collections),
    )