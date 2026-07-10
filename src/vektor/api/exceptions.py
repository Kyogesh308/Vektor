"""
vektor.api.exceptions
-----------------------
Central Vektor-exception-to-HTTP-status mapping. All exception handlers
registered here, in one place, so no endpoint needs its own try/except.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from vektor import (
    CollectionNotFoundError, CollectionAlreadyExistsError,
    VectorNotFoundError, DuplicateIDError, VektorConfigError,
    VektorTimeoutError, EmptyCollectionError,
)
from vektor.validator import (
    InvalidVectorDimensionError, NonFiniteVectorError, InvalidMetricError,
)


EXCEPTION_STATUS_MAP = {
    CollectionNotFoundError: (404, "COLLECTION_NOT_FOUND"),
    CollectionAlreadyExistsError: (409, "COLLECTION_ALREADY_EXISTS"),
    VectorNotFoundError: (404, "VECTOR_NOT_FOUND"),
    DuplicateIDError: (409, "DUPLICATE_ID"),
    InvalidVectorDimensionError: (422, "DIMENSION_MISMATCH"),
    NonFiniteVectorError: (422, "INVALID_VECTOR_VALUE"),
    InvalidMetricError: (422, "INVALID_METRIC"),
    VektorConfigError: (400, "CONFIG_ERROR"),
    VektorTimeoutError: (503, "TIMEOUT"),
    EmptyCollectionError: (409, "EMPTY_COLLECTION"),
}


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"status": "error", "error": {"code": code, "message": message}},
    )


def register_exception_handlers(app: FastAPI) -> None:
    """
    Register one handler per mapped exception type, plus a catch-all
    for unhandled exceptions (returns 500, never leaks a traceback).
    """
    for exc_type, (status_code, code) in EXCEPTION_STATUS_MAP.items():

        def make_handler(status_code=status_code, code=code):
            async def handler(request: Request, exc: Exception):
                return _error_response(status_code, code, str(exc))
            return handler

        app.add_exception_handler(exc_type, make_handler())

    @app.exception_handler(Exception)
    async def catch_all_handler(request: Request, exc: Exception):
        # Never leak a traceback to the client
        return _error_response(500, "INTERNAL_ERROR", "An internal error occurred.")