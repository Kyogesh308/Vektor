from __future__ import annotations
"""
vektor.persistence.binary
-------------------------
Read/write for vector.bin, offsets.bin, and graph.bin.

Binary format is fixed. Do not change field widths or byte order after
data has been written — there is no migration path without re-writing
every record.

Byte order: little-endian throughout (<).
"""


import mmap
import os
import struct
from pathlib import Path
from typing import Optional

import numpy as np

from vektor.persistence.atomic import atomic_write


# ---------------------------------------------------------------------------
# vector.bin
# ---------------------------------------------------------------------------

MAGIC = b'VKTR'
FORMAT_VERSION = 1
HEADER_SIZE = 24  # magic(4) + version(4) + dimension(8) + slot_count(8)
HEADER_FMT = "<4sIQQ"  # magic, version, dimension, slot_count


def _vector_stride(dimension: int) -> int:
    return 8 + dimension * 4  # slot_id (uint64) + float32 × dim


def _record_fmt(dimension: int) -> str:
    return f"<Q{dimension}f"  # slot_id + dimension floats


def init_vector_bin(path: Path, dimension: int) -> None:
    """Create a new vector.bin with header and zero slots."""
    with atomic_write(path) as f:
        f.write(struct.pack(HEADER_FMT, MAGIC, FORMAT_VERSION, dimension, 0))


def read_vector_bin_header(path: Path) -> dict:
    """Read and validate the vector.bin header."""
    with open(path, "rb") as f:
        raw = f.read(HEADER_SIZE)
    magic, version, dimension, slot_count = struct.unpack(HEADER_FMT, raw)
    if magic != MAGIC:
        raise ValueError(f"Invalid magic bytes in {path}: {magic!r}")
    if version != FORMAT_VERSION:
        raise ValueError(f"Unsupported format version {version} in {path}")
    return {"dimension": dimension, "slot_count": slot_count}


def append_vector(path: Path, slot_id: int, vector: np.ndarray) -> None:
    """
    Append one vector record to vector.bin and update the slot count in the header.

    Uses atomic write — the entire file is rewritten via a temp file.
    This is safe but O(N) in file size. Phase 7 may switch to append-only
    writes for performance, but correctness is the priority in Phase 5.
    """
    path = Path(path)
    dimension = vector.shape[0]
    fmt = _record_fmt(dimension)

    # Read existing content
    existing = path.read_bytes() if path.exists() else b""

    if existing:
        magic, version, dim, slot_count = struct.unpack(HEADER_FMT, existing[:HEADER_SIZE])
        if dim != dimension:
            raise ValueError(f"Dimension mismatch: file has {dim}, got {dimension}")
        new_slot_count = slot_count + 1
    else:
        new_slot_count = 1

    # Pack new record
    new_record = struct.pack(fmt, slot_id, *vector.tolist())

    # Write atomically
    with atomic_write(path) as f:
        new_header = struct.pack(HEADER_FMT, MAGIC, FORMAT_VERSION,
                                 dimension, new_slot_count)
        body = existing[HEADER_SIZE:] if existing else b""
        f.write(new_header + body + new_record)


def read_vector(path: Path, slot_index: int, dimension: int) -> np.ndarray:
    """
    Read one vector from vector.bin by its sequential slot index.

    Args:
        path:       Path to vector.bin.
        slot_index: The 0-based position of the record (not the slot_id field).
        dimension:  Expected vector dimension.

    Returns:
        NumPy float32 array of shape (dimension,).
    """
    stride = _vector_stride(dimension)
    offset = HEADER_SIZE + slot_index * stride
    fmt = _record_fmt(dimension)

    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        raw = mm[offset: offset + stride]
        mm.close()

    unpacked = struct.unpack(fmt, raw)
    # unpacked[0] is slot_id, unpacked[1:] are the float values
    return np.array(unpacked[1:], dtype=np.float32)


def read_all_vectors(path: Path, dimension: int) -> list[tuple[int, np.ndarray]]:
    """
    Read all vector records from vector.bin.

    Returns:
        List of (slot_id, vector) tuples.
    """
    header = read_vector_bin_header(path)
    slot_count = header["slot_count"]
    results = []
    for i in range(slot_count):
        vec = read_vector(path, i, dimension)
        stride = _vector_stride(dimension)
        offset = HEADER_SIZE + i * stride
        fmt = _record_fmt(dimension)
        with open(path, "rb") as f:
            raw = f.read()[offset: offset + stride]
        slot_id = struct.unpack(f"<Q", raw[:8])[0]
        results.append((slot_id, vec))
    return results


# ---------------------------------------------------------------------------
# offsets.bin
# ---------------------------------------------------------------------------

OFFSET_RECORD_SIZE = 16  # node_id (uint64) + byte_offset (uint64)
OFFSET_FMT = "<QQ"


def append_offset(path: Path, node_id: int, byte_offset: int) -> None:
    """Append one node_id → byte_offset mapping to offsets.bin."""
    record = struct.pack(OFFSET_FMT, node_id, byte_offset)
    existing = path.read_bytes() if path.exists() else b""
    with atomic_write(path) as f:
        f.write(existing + record)


def read_offset(path: Path, index: int) -> tuple[int, int]:
    """
    Read one offset record by sequential index.

    Returns:
        (node_id, byte_offset) tuple.
    """
    offset = index * OFFSET_RECORD_SIZE
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        raw = mm[offset: offset + OFFSET_RECORD_SIZE]
        mm.close()
    return struct.unpack(OFFSET_FMT, raw)


def read_all_offsets(path: Path) -> list[tuple[int, int]]:
    """Read all offset records. Returns list of (node_id, byte_offset)."""
    if not path.exists() or path.stat().st_size == 0:
        return []
    data = path.read_bytes()
    count = len(data) // OFFSET_RECORD_SIZE
    return [struct.unpack(OFFSET_FMT, data[i*16:(i+1)*16]) for i in range(count)]


# ---------------------------------------------------------------------------
# graph.bin — Phase 5: stub implementation only
# ---------------------------------------------------------------------------

GRAPH_STUB_FMT = "<B"  # num_layers = 0


def write_stub_graph_node(path: Path) -> int:
    """
    Append a stub graph node to graph.bin.

    A stub node has num_layers=0 and no adjacency data.
    Returns the byte offset where this record was written (for offsets.bin).

    Returns:
        int: byte offset of the written stub record in graph.bin.
    """
    stub = struct.pack(GRAPH_STUB_FMT, 0)  # num_layers = 0
    existing = path.read_bytes() if path.exists() else b""
    byte_offset = len(existing)
    with atomic_write(path) as f:
        f.write(existing + stub)
    return byte_offset


def read_graph_node(path: Path, byte_offset: int) -> dict:
    """
    Read one graph node from graph.bin at the given byte offset.

    Returns:
        dict with "num_layers" and "layers" (list of neighbour lists).
    """
    with open(path, "rb") as f:
        f.seek(byte_offset)
        num_layers = struct.unpack("<B", f.read(1))[0]
        layers = []
        for _ in range(num_layers):
            neighbour_count = struct.unpack("<H", f.read(2))[0]
            neighbours = list(struct.unpack(f"<{neighbour_count}Q",
                                            f.read(neighbour_count * 8)))
            layers.append(neighbours)
    return {"num_layers": num_layers, "layers": layers}