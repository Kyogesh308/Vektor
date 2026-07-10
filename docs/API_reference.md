# Vektor API Reference

## Base URL
`http://127.0.0.1:8000`

## Health

### GET /health
Liveness check. No Vektor work performed.

**Response 200:**
```json
{"status": "ok", "uptime_seconds": 123.45}
```

## Collections

### POST /v1/collections
Create a new collection.

**Request:**
```json
{"name": "docs", "dim": 384, "metric": "cosine", "mode": "beginner"}
```

**Responses:** 201 Created | 409 (COLLECTION_ALREADY_EXISTS) | 422 (invalid metric/dim)

**curl:**
```bash
curl -X POST http://127.0.0.1:8000/v1/collections \
  -H "Content-Type: application/json" \
  -d '{"name": "docs", "dim": 4, "metric": "cosine"}'
```

### GET /v1/collections
List all collection names. Returns 200 with `["docs", "images"]`.

### GET /v1/collections/{name}
Get collection config. 200 | 404 (COLLECTION_NOT_FOUND).

### DELETE /v1/collections/{name}
Delete a collection. 204 (empty body) | 404.

## Vectors

### POST /v1/collections/{name}/vectors
Insert one vector. 201 | 409 (DUPLICATE_ID) | 422 (DIMENSION_MISMATCH).

**curl:**
```bash
curl -X POST http://127.0.0.1:8000/v1/collections/docs/vectors \
  -H "Content-Type: application/json" \
  -d '{"id": "doc1", "vector": [0.1, 0.2, 0.3, 0.4], "metadata": {"source": "test"}}'
```

### POST /v1/collections/{name}/vectors/batch
Batch insert. Always 207 — inspect `inserted`/`failed`/`errors`.

### GET /v1/collections/{name}/vectors/{id}
Fetch a vector. `?include_vector=true` includes the raw vector. 200 | 404.

### PUT /v1/collections/{name}/vectors/{id}
Update vector data and/or metadata. 200 | 404 | 422 (both fields None).

### DELETE /v1/collections/{name}/vectors/{id}
Tombstone a vector. 204 | 404.

## Search

### POST /v1/collections/{name}/search
Nearest-neighbour search. Always POST — query vectors are too large for a URL.

**Request:**
```json
{"query": [0.1, 0.2, 0.3, 0.4], "k": 5, "filters": {"source": "test"}, "strategy": "post"}
```

**Response 200:**
```json
{
  "results": [{"id": "doc1", "score": 0.98, "rank": 1, "metadata": {}}],
  "count": 1, "warnings": [], "metric": "cosine"
}
```

**curl:**
```bash
curl -X POST http://127.0.0.1:8000/v1/collections/docs/search \
  -H "Content-Type: application/json" \
  -d '{"query": [0.1, 0.2, 0.3, 0.4], "k": 5}'
```

## Error Envelope

```json
{"status": "error", "error": {"code": "VECTOR_NOT_FOUND", "message": "..."}}
```