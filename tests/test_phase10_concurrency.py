"""
Phase 10 — Concurrency Tests

The stress test (TestConcurrentStress) is the checkpoint test.
It must pass 20 consecutive runs — not just once.
Run it with:

    pytest tests/test_phase10_concurrency.py::TestConcurrentStress -v -s

If it fails on run 7 of 20, you have a race condition.
The failure message will be the only clue — study it before adding more locks.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Optional

import numpy as np
import pytest

from vektor.collection import Collection
from vektor.concurrency.lock import CollectionLock, DEFAULT_TIMEOUT_SECONDS
from vektor.concurrency.exceptions import VektorTimeoutError, EmptyCollectionError
from vektor.hnsw.index import HNSWIndex
from vektor.storage import VectorStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_collection(dim: int = 16, metric: str = "euclidean") -> Collection:
    return Collection(name="test", dimension=dim, metric=metric,
                      m=8, ef_construction=50)


def make_vecs(n: int, dim: int, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(dim).astype(np.float32) for _ in range(n)]


# ---------------------------------------------------------------------------
# CollectionLock unit tests
# ---------------------------------------------------------------------------

class TestCollectionLock:

    def test_acquires_and_releases_normally(self):
        lock = CollectionLock()
        with lock.acquire(operation="test"):
            pass  # Lock held here
        # Lock released — next acquire must succeed
        with lock.acquire(operation="test2"):
            pass

    def test_lock_is_released_on_exception(self):
        lock = CollectionLock()
        try:
            with lock.acquire(operation="test"):
                raise ValueError("intentional error")
        except ValueError:
            pass
        # Lock must be released — this must not timeout
        with lock.acquire(operation="after_exception", timeout=1.0):
            pass

    def test_timeout_zero_raises_when_lock_held(self):
        lock = CollectionLock()
        acquired = lock._lock.acquire(blocking=False)
        assert acquired

        try:
            with pytest.raises(VektorTimeoutError):
                with lock.acquire(operation="nonblocking", timeout=0):
                    pass
        finally:
            lock._lock.release()

    def test_timeout_error_message_contains_operation_name(self):
        lock = CollectionLock()
        lock._lock.acquire()
        try:
            with pytest.raises(VektorTimeoutError, match="my_operation"):
                with lock.acquire(operation="my_operation", timeout=0):
                    pass
        finally:
            lock._lock.release()

    def test_reentrant_acquisition_does_not_deadlock(self):
        """RLock allows the same thread to acquire multiple times."""
        lock = CollectionLock()
        with lock.acquire(operation="outer"):
            with lock.acquire(operation="inner"):
                pass  # Both held simultaneously — no deadlock

    def test_default_timeout_applied_when_none_passed(self):
        lock = CollectionLock(timeout=0.1)  # 100ms default
        lock._lock.acquire()
        start = time.perf_counter()
        try:
            with pytest.raises(VektorTimeoutError):
                with lock.acquire(operation="test"):
                    pass
        finally:
            lock._lock.release()
        elapsed = time.perf_counter() - start
        # Must have waited approximately the timeout, not longer
        assert elapsed < 1.0, f"Waited {elapsed:.2f}s — too long for 0.1s timeout"


# ---------------------------------------------------------------------------
# Timeout test
# ---------------------------------------------------------------------------

class TestTimeout:

    def test_writer_blocking_causes_timeout_for_second_thread(self):
        """
        Thread 1 holds the lock. Thread 2 attempts insert with short timeout.
        Thread 2 must receive VektorTimeoutError. Thread 1 releases.
        Thread 2 retries and succeeds.
        """
        col = make_collection()
        index = HNSWIndex(col, seed=42)

        # Pre-insert one vector so the index is non-empty
        index.add(0, make_vecs(1, 16, seed=0)[0])

        thread2_errors = []
        thread2_succeeded_on_retry = threading.Event()

        def thread2_fn():
            # First attempt — short timeout, will fail (thread 1 holds lock)
            try:
                index.add(999, make_vecs(1, 16, seed=999)[0], timeout=0.1)
            except VektorTimeoutError:
                thread2_errors.append("timeout_as_expected")

            # Retry after thread 1 releases
            time.sleep(0.2)
            try:
                index.add(999, make_vecs(1, 16, seed=999)[0], timeout=2.0)
                thread2_succeeded_on_retry.set()
            except Exception as e:
                thread2_errors.append(f"retry_failed: {e}")

        # Thread 1 holds lock manually for 300ms
        index._lock._lock.acquire()
        t2 = threading.Thread(target=thread2_fn)
        t2.start()
        time.sleep(0.3)
        index._lock._lock.release()
        t2.join(timeout=5.0)

        assert "timeout_as_expected" in thread2_errors
        assert thread2_succeeded_on_retry.is_set(), (
            f"Thread 2 retry failed. Errors: {thread2_errors}"
        )

    def test_exception_mid_insert_releases_lock(self):
        """
        If insert raises partway through, the lock must still release.
        The collection must remain usable.
        """
        col = make_collection()
        store = VectorStore(col)

        # Insert with wrong dimension — triggers validation error inside lock
        bad_vector = np.zeros(999, dtype=np.float32)  # wrong dim
        from vektor.validator import InvalidVectorDimensionError
        with pytest.raises(InvalidVectorDimensionError):
            store.insert("bad", bad_vector)

        # Lock must be released — this insert must succeed
        good_vector = np.zeros(16, dtype=np.float32)
        store.insert("good", good_vector)
        assert store.count() == 1


# ---------------------------------------------------------------------------
# Empty collection guard
# ---------------------------------------------------------------------------

class TestEmptyCollectionGuard:

    def test_search_on_empty_raises_empty_collection_error(self):
        col = make_collection()
        index = HNSWIndex(col, seed=42)
        query = np.zeros(16, dtype=np.float32)
        with pytest.raises(EmptyCollectionError):
            index.search(query, k=5, ef=10)

    def test_two_threads_searching_empty_collection_both_get_error(self):
        col = make_collection()
        index = HNSWIndex(col, seed=42)
        query = np.zeros(16, dtype=np.float32)
        errors = []

        def search_thread():
            try:
                index.search(query, k=5, ef=10)
            except EmptyCollectionError:
                errors.append("EmptyCollectionError")
            except Exception as e:
                errors.append(f"wrong_error: {type(e).__name__}: {e}")

        threads = [threading.Thread(target=search_thread) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert all(e == "EmptyCollectionError" for e in errors), (
            f"Expected all EmptyCollectionError, got: {errors}"
        )
        assert len(errors) == 5


# ---------------------------------------------------------------------------
# Concurrent stress test — THE checkpoint test
# Must pass 20 consecutive runs with zero exceptions
# ---------------------------------------------------------------------------

class TestConcurrentStress:

    N_RUNS = 20
    N_PRELOAD = 200
    N_WRITER_THREADS = 5
    N_INSERTS_PER_WRITER = 20
    N_READER_THREADS = 5
    N_SEARCHES_PER_READER = 30
    DIM = 16
    K = 5
    EF = 20

    def _run_once(self, run_number: int) -> dict:
        """
        One stress test run. Returns a result dict.
        Raises AssertionError on failure.
        """
        col = make_collection(dim=self.DIM)
        index = HNSWIndex(col, seed=run_number)
        store = VectorStore(col)

        # Preload vectors so the graph is non-trivial
        preload_vecs = make_vecs(self.N_PRELOAD, self.DIM, seed=run_number)
        for i, v in enumerate(preload_vecs):
            index.add(i, v)
            store.insert(str(i), v)

        # Shared state for collecting results
        thread_exceptions: list[Exception] = []
        search_result_counts: list[int] = []
        inserted_slot_ids: set[int] = set(range(self.N_PRELOAD))
        result_lock = threading.Lock()

        # Barrier ensures all threads start simultaneously
        barrier = threading.Barrier(
            self.N_WRITER_THREADS + self.N_READER_THREADS
        )

        def writer_fn(writer_id: int):
            barrier.wait()
            base_slot = self.N_PRELOAD + writer_id * self.N_INSERTS_PER_WRITER
            vecs = make_vecs(self.N_INSERTS_PER_WRITER, self.DIM,
                             seed=1000 + writer_id + run_number * 100)
            for i, v in enumerate(vecs):
                slot_id = base_slot + i
                try:
                    index.add(slot_id, v, timeout=10.0)
                    with result_lock:
                        inserted_slot_ids.add(slot_id)
                except Exception as e:
                    with result_lock:
                        thread_exceptions.append(
                            RuntimeError(
                                f"Writer {writer_id} insert {i} failed: "
                                f"{type(e).__name__}: {e}"
                            )
                        )

        def reader_fn(reader_id: int):
            barrier.wait()
            rng = np.random.default_rng(2000 + reader_id + run_number * 100)
            for _ in range(self.N_SEARCHES_PER_READER):
                query = rng.standard_normal(self.DIM).astype(np.float32)
                try:
                    results = index.search(query, k=self.K, ef=self.EF,
                                           timeout=10.0)
                    with result_lock:
                        search_result_counts.append(len(results))
                except EmptyCollectionError:
                    pass  # Acceptable if collection is being modified
                except Exception as e:
                    with result_lock:
                        thread_exceptions.append(
                            RuntimeError(
                                f"Reader {reader_id} search failed: "
                                f"{type(e).__name__}: {e}"
                            )
                        )

        # Launch all threads
        writers = [
            threading.Thread(target=writer_fn, args=(i,))
            for i in range(self.N_WRITER_THREADS)
        ]
        readers = [
            threading.Thread(target=reader_fn, args=(i,))
            for i in range(self.N_READER_THREADS)
        ]

        all_threads = writers + readers
        for t in all_threads:
            t.start()

        for t in all_threads:
            t.join(timeout=60.0)

        # Verify threads actually finished (not timed out by join)
        still_alive = [t for t in all_threads if t.is_alive()]
        assert not still_alive, (
            f"Run {run_number}: {len(still_alive)} threads still alive "
            f"after 60s timeout. Likely deadlock."
        )

        # Verify zero exceptions
        assert not thread_exceptions, (
            f"Run {run_number}: {len(thread_exceptions)} thread exception(s):\n"
            + "\n".join(str(e) for e in thread_exceptions)
        )

        # Verify vector count
        expected_count = (
            self.N_PRELOAD +
            self.N_WRITER_THREADS * self.N_INSERTS_PER_WRITER
        )
        actual_count = index.size
        assert actual_count == expected_count, (
            f"Run {run_number}: expected {expected_count} vectors, "
            f"got {actual_count}. Data was lost or double-counted."
        )

        # Verify search results don't reference non-existent slot IDs
        bad_ids = []
        for slot_id, layer_dict in index._graph.items():
            for layer, neighbours in layer_dict.items():
                for n_id in neighbours:
                    if n_id not in inserted_slot_ids:
                        bad_ids.append((slot_id, layer, n_id))

        assert not bad_ids, (
            f"Run {run_number}: graph references {len(bad_ids)} "
            f"non-existent slot IDs. Graph is corrupt."
        )

        # Verify all search result counts <= k
        oversized = [c for c in search_result_counts if c > self.K]
        assert not oversized, (
            f"Run {run_number}: {len(oversized)} searches returned "
            f"more than k={self.K} results."
        )

        return {
            "run": run_number,
            "vector_count": actual_count,
            "search_count": len(search_result_counts),
            "exceptions": len(thread_exceptions),
        }

    def test_stress_20_consecutive_runs(self):
        """
        Run the concurrent stress test 20 times consecutively.
        Concurrency bugs are intermittent — one passing run proves nothing.
        20 consecutive passes provide meaningful confidence.

        If this test fails on run N, the failure message identifies
        whether it was an exception, a count mismatch, or a graph corruption.
        """
        for run in range(1, self.N_RUNS + 1):
            result = self._run_once(run_number=run)
            print(
                f"  Run {run:02d}/{self.N_RUNS}: "
                f"vectors={result['vector_count']}, "
                f"searches={result['search_count']}, "
                f"exceptions={result['exceptions']}"
            )


# ---------------------------------------------------------------------------
# VectorStore concurrency
# ---------------------------------------------------------------------------

class TestVectorStoreConcurrency:

    def test_concurrent_inserts_no_data_loss(self):
        col = make_collection()
        store = VectorStore(col)

        exceptions = []
        n_threads = 10
        inserts_per_thread = 20
        barrier = threading.Barrier(n_threads)

        def insert_fn(thread_id: int):
            barrier.wait()
            vecs = make_vecs(inserts_per_thread, 16, seed=thread_id)
            for i, v in enumerate(vecs):
                vec_id = f"t{thread_id}_v{i}"
                try:
                    store.insert(vec_id, v, timeout=10.0)
                except Exception as e:
                    exceptions.append(f"{vec_id}: {e}")

        threads = [threading.Thread(target=insert_fn, args=(i,))
                   for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30.0)

        assert not exceptions, f"Exceptions: {exceptions}"
        expected = n_threads * inserts_per_thread
        assert store.count() == expected, (
            f"Expected {expected} vectors, got {store.count()}"
        )

    def test_concurrent_delete_and_search_no_crash(self):
        col = make_collection()
        store = VectorStore(col)
        vecs = make_vecs(100, 16, seed=0)
        for i, v in enumerate(vecs):
            store.insert(str(i), v)

        exceptions = []
        barrier = threading.Barrier(10)

        def delete_fn():
            barrier.wait()
            for i in range(0, 50, 5):
                try:
                    store.delete(str(i), timeout=5.0)
                except Exception as e:
                    if "does not exist" not in str(e):
                        exceptions.append(f"delete: {e}")

        def search_fn():
            barrier.wait()
            q = np.zeros(16, dtype=np.float32)
            for _ in range(20):
                try:
                    store.search(q, k=5, timeout=5.0)
                except Exception as e:
                    exceptions.append(f"search: {e}")

        threads = (
            [threading.Thread(target=delete_fn) for _ in range(5)] +
            [threading.Thread(target=search_fn) for _ in range(5)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30.0)

        assert not exceptions, f"Exceptions during concurrent delete+search: {exceptions}"