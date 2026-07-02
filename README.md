# Vektor

A production-style vector database built from scratch in Python.

## Status

> Phase 0 — Project foundation. CI running. No application logic yet.

## Architecture

| Layer | Module | Status |
|---|---|---|
| Input Validation | `vektor.validator` | Phase 1 |
| Distance Metrics | `vektor.distance` | Phase 2 |
| Collection Manager | `vektor.collection` | Phase 3 |
| Vector Storage + Search | `vektor.storage` | Phase 4 |

## Development

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Notes

`test_fails_intentionally` in `tests/test_phase0_sanity.py` is a sentinel test — it is supposed to fail. It exists to prove the test runner correctly reports failures. Do not fix it.
