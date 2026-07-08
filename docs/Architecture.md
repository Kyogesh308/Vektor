# Vektor Architecture

## Concurrency Model (v1)

### Lock Design

Vektor v1 uses a single exclusive `threading.RLock` per collection,
wrapped in a `CollectionLock` with a configurable timeout (default 5s).

All operations — reads and writes — acquire this lock exclusively.
This means concurrent searches within one collection are **serialised**,
not parallelised. This is a known v1 limitation, not an oversight.

### What the Lock Protects

Per collection:
- HNSW graph adjacency structure (`_graph` dict)
- Global entry point and max layer
- In-memory vector storage (`_vectors`, `_metadata` dicts)
- SQLite connection (single connection, lock-serialised)

### What the Lock Does NOT Protect

- Operations on different collections (each collection has its own lock)
- Phase 5 binary file writes (handled by atomic rename — crash-safe without locks)
- Read-only mmaps (safe without locking — no writes through these handles)
- BenchmarkRunner (single-threaded by design — no lock needed)

### Concurrency Contract

| Guarantee | v1 |
|---|---|
| Single-threaded correctness | Yes |
| Concurrent insert safety | Yes (serialised) |
| Concurrent search safety | Yes (serialised) |
| Concurrent search parallelism | **No** — searches queue |
| Multi-process safety | **No** — single process only |
| Lock timeout | Yes — `VektorTimeoutError` after 5s default |

### Hard Constraints

**Single process only.** Two Python processes opening the same collection
directory simultaneously will corrupt the database. This is not detected
or prevented — it is a hard architectural constraint.

**No search parallelism in v1.** Under FastAPI with multiple concurrent
requests, searches queue behind each other. Throughput under load is
approximately equal to single-threaded throughput.

### v2 Upgrade Path

Replace `CollectionLock` with a true reader-writer lock. Reads acquire
a shared lock (non-blocking for other readers). Writes acquire an
exclusive lock (blocks all readers and writers). This allows concurrent
searches while still serialising inserts.

The `CollectionLock.acquire(operation="search")` calls are already
marked with `# v2: read-lock` comments in the implementation.