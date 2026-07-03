from __future__ import annotations
"""
vektor.persistence.archive
--------------------------
Export and import Vektor collections as .vdb archive files.

A .vdb file is a ZIP archive containing the five per-collection files:
  vector.bin, offsets.bin, graph.bin, metadata.db, collection.json
"""


import shutil
import sqlite3
import zipfile
from pathlib import Path

from vektor.persistence.manifest import read_manifest
from vektor.persistence.db import initialize_db, insert_collection


class ArchiveError(Exception):
    """Raised when export or import fails validation."""


COLLECTION_FILES = [
    "vector.bin",
    "offsets.bin",
    "graph.bin",
    "metadata.db",
    "collection.json",
]


def export_collection(collection_dir: Path, output_path: Path) -> None:
    """
    Pack a collection directory into a .vdb archive.

    Args:
        collection_dir: Directory containing the five collection files.
        output_path:    Destination path for the .vdb file.

    Raises:
        ArchiveError: If required files are missing.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    missing = [f for f in COLLECTION_FILES
               if not (collection_dir / f).exists()]
    if missing:
        raise ArchiveError(
            f"Cannot export: missing files in {collection_dir}: {missing}"
        )

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename in COLLECTION_FILES:
            zf.write(collection_dir / filename, arcname=filename)


def import_collection(vdb_path: Path, target_dir: Path,
                      conn: sqlite3.Connection) -> str:
    """
    Unpack a .vdb archive and register the collection in metadata.db.

    Args:
        vdb_path:   Path to the .vdb file.
        target_dir: Directory to extract collection files into.
        conn:       Open SQLite connection to the target metadata.db.

    Returns:
        The collection name from the manifest.

    Raises:
        ArchiveError: If the archive is invalid or the collection already exists.
    """
    vdb_path = Path(vdb_path)
    target_dir = Path(target_dir)

    if not zipfile.is_zipfile(vdb_path):
        raise ArchiveError(f"{vdb_path} is not a valid .vdb archive.")

    with zipfile.ZipFile(vdb_path, "r") as zf:
        names = zf.namelist()
        missing = [f for f in COLLECTION_FILES if f not in names]
        if missing:
            raise ArchiveError(f"Archive missing required files: {missing}")

        target_dir.mkdir(parents=True, exist_ok=True)
        zf.extractall(target_dir)

    manifest = read_manifest(target_dir / "collection.json")

    insert_collection(
        conn,
        name=manifest["name"],
        dimension=manifest["dimension"],
        metric=manifest["metric"],
        m=manifest["m"],
        ef_construction=manifest["ef_construction"],
        vektor_version=manifest["vektor_version"],
    )

    return manifest["name"]