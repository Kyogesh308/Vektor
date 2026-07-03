from __future__ import annotations
"""
vektor.persistence.manifest
---------------------------
Read and write collection.json manifest files.
"""


import json
from datetime import datetime, timezone
from pathlib import Path

import vektor


class ManifestError(Exception):
    """Raised when a manifest is missing, corrupt, or inconsistent."""


def write_manifest(path: Path, name: str, dimension: int, metric: str,
                   m: int, ef_construction: int) -> None:
    """Write collection.json for the given collection."""
    data = {
        "name": name,
        "dimension": dimension,
        "metric": metric,
        "m": m,
        "ef_construction": ef_construction,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "vektor_version": vektor.__version__,
    }
    from vektor.persistence.atomic import atomic_write
    with atomic_write(path, mode="w") as f:
        json.dump(data, f, indent=2)


def read_manifest(path: Path) -> dict:
    """
    Read and parse collection.json.

    Raises:
        ManifestError: File missing or invalid JSON.
    """
    if not path.exists():
        raise ManifestError(f"Manifest not found: {path}")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ManifestError(f"Corrupt manifest at {path}: {e}") from e


def validate_manifest_against_db(manifest: dict, db_row) -> None:
    """
    Confirm manifest fields match the collections table row.

    Raises:
        ManifestError: Any field mismatch detected.
    """
    for field in ("dimension", "metric", "m", "ef_construction"):
        if manifest[field] != db_row[field]:
            raise ManifestError(
                f"Manifest/DB mismatch on '{field}': "
                f"manifest={manifest[field]!r}, db={db_row[field]!r}"
            )