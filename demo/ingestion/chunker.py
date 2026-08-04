"""
demo/ingestion/chunker.py
----------------------------
Split text into overlapping chunks for embedding.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    id: str
    text: str
    start_char: int


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 75) -> list[Chunk]:
    """
    Split text into overlapping chunks by character count.

    Args:
        text:       Full document text.
        chunk_size: Target characters per chunk.
        overlap:    Characters shared between consecutive chunks.

    Returns:
        List of Chunk objects with unique IDs.
    """
    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = start + chunk_size
        chunk_text_content = text[start:end].strip()
        if chunk_text_content:
            chunks.append(Chunk(id=f"chunk_{idx}", text=chunk_text_content, start_char=start))
            idx += 1
        start += chunk_size - overlap
    return chunks
