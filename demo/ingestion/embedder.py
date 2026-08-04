"""
demo/ingestion/embedder.py
-----------------------------
Wraps sentence-transformers for chunk embedding.
"""

from __future__ import annotations

import numpy as np


class Embedder:
    """
    Thin wrapper around sentence-transformers.

    Usage:
        embedder = Embedder()
        vectors = embedder.embed(["text one", "text two"])
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        """Returns float32 array of shape (len(texts), self.dim)."""
        return self._model.encode(texts, convert_to_numpy=True).astype(np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]
