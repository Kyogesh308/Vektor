"""
vektor.api.dependencies
-------------------------
FastAPI dependency injection for the Vektor client.
"""

from __future__ import annotations

from fastapi import Request

from vektor import Vektor


def get_vektor(request: Request) -> Vektor:
    """
    Retrieve the Vektor client from application state.

    Every endpoint needing Vektor declares: client: Vektor = Depends(get_vektor)
    This makes the client substitutable in tests via dependency_overrides,
    with zero changes to endpoint code.
    """
    return request.app.state.vektor_client