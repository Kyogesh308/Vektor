"""
vektor.concurrency.lock
------------------------
Thread-safe lock wrapper for Vektor collection operations.

v1 Design: single exclusive RLock per collection.
- All operations (reads and writes) acquire exclusively.
- Concurrent searches within one collection are serialised.
- Collections are independent — different collections do not block each other.
- Lock acquisition has a configurable timeout (default: 5 seconds).
- Reentrant acquisition from the same thread is safe and immediate.

v2 Upgrade path: replace with a true reader-writer lock.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Generator

from vektor.concurrency.exceptions import VektorTimeoutError


DEFAULT_TIMEOUT_SECONDS = 5.0


class CollectionLock:
    """
    Reentrant exclusive lock for a single Vektor collection.

    One instance per collection. Shared by HNSWIndex, VectorStore,
    and all persistence operations on that collection.

    Usage:
        lock = CollectionLock()

        with lock.acquire(operation="insert"):
            # modify shared state
        # lock released here, even if exception raised inside
    """

    def __init__(self, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> None:
        self._lock = threading.RLock()
        self._default_timeout = timeout

    def _is_owner(self) -> bool:
        """
        Check whether the current thread already owns the lock.

        Accesses RLock._owner directly — this is a private CPython
        implementation detail, but it is stable across CPython 3.8–3.12
        and is the only reliable way to detect reentrant ownership before
        calling acquire(timeout=N).

        Returns False defensively if the attribute is absent (non-CPython).
        """
        try:
            return self._lock._owner == threading.get_ident()
        except AttributeError:
            return False

    @contextmanager
    def acquire(
        self,
        operation: str = "unknown",
        timeout: float = None,
    ) -> Generator[None, None, None]:
        """
        Context manager that acquires the lock with a timeout.

        Reentrant acquisition (same thread, nested `with lock.acquire(...)` blocks)
        is detected and handled without a timeout — the owning thread's reacquisition
        is always immediate for RLock regardless of CPython C/Python implementation.

        Args:
            operation: Human-readable name of the operation.
                       Used in timeout error messages.
            timeout:   Seconds to wait for lock acquisition.
                       None → use instance default (5s).
                       0   → non-blocking (raise immediately if not free).

        Raises:
            VektorTimeoutError: Lock not acquired within timeout.

        Usage:
            with lock.acquire(operation="insert"):
                ...  # lock held here
            # lock always released here, even on exception
        """
        if timeout is None:
            timeout = self._default_timeout

        # Reentrant acquisition — CPython's RLock.acquire(timeout=N) does not
        # reliably short-circuit for the owning thread when timeout is specified.
        # Detect ownership first and use blocking=True (always immediate for owner).
        if self._is_owner():
            # Always succeeds — RLock from its owner never blocks
            acquired = self._lock.acquire(blocking=True)
        else:
            acquired = self._lock.acquire(timeout=timeout)

        if not acquired:
            raise VektorTimeoutError(
                f"Operation '{operation}' timed out waiting for the collection lock "
                f"after {timeout}s. The collection may be under heavy write load. "
                f"Retry the operation after a short delay."
            )

        try:
            yield
        finally:
            self._lock.release()