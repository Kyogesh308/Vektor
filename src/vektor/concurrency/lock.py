"""
vektor.concurrency.lock
------------------------
Thread-safe lock wrapper for Vektor collection operations.

v1 Design: single exclusive RLock per collection.
- All operations (reads and writes) acquire exclusively.
- Concurrent searches within one collection are serialised.
- Collections are independent — different collections do not block each other.
- Lock acquisition has a configurable timeout (default: 5 seconds).

v2 Upgrade path: replace with a true reader-writer lock (rwlock package
or threading.Condition-based implementation) that allows concurrent reads.
This is documented here, not implemented — Phase 10 is correctness, not
performance.
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

        # Write operation
        with lock.acquire(operation="insert", timeout=5.0):
            # modify shared state here

        # Read operation (same lock in v1 — comment marks v2 upgrade point)
        with lock.acquire(operation="search", timeout=5.0):  # v2: read-lock
            # read shared state here
    """

    def __init__(self, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> None:
        self._lock = threading.Lock()
        self._default_timeout = timeout

    @contextmanager
    def acquire(
        self,
        operation: str = "unknown",
        timeout: float = None,
    ) -> Generator[None, None, None]:
        """
        Context manager that acquires the lock with a timeout.

        Args:
            operation: Human-readable name of the operation acquiring the lock.
                       Used in error messages. E.g. "insert", "search", "delete".
            timeout:   Seconds to wait. If None, uses the instance default.
                       Set to 0 for a non-blocking attempt.

        Raises:
            VektorTimeoutError: Lock not acquired within timeout.

        Usage:
            with lock.acquire(operation="insert"):
                ...  # lock held here
            # lock released here, even if exception raised inside
        """
        if timeout is None:
            timeout = self._default_timeout

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