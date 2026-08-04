"""
Tests for demo/ingestion/embedder.py

Fast tests mock SentenceTransformer entirely — no model download in CI.
The one real-model test is marked slow and skipped by default.
"""

from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from demo.ingestion.embedder import Embedder


def _create_mock_transformer(dim: int = 384):
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = dim
    mock_model.get_embedding_dimension.return_value = dim
    return mock_model


class TestEmbedderMocked:

    def test_embed_returns_correct_shape(self):
        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_model = _create_mock_transformer(384)
            mock_model.encode.return_value = np.random.rand(3, 384)
            mock_st.return_value = mock_model

            embedder = Embedder()
            result = embedder.embed(["text one", "text two", "text three"])

        assert result.shape == (3, 384)

    def test_embed_returns_float32(self):
        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_model = _create_mock_transformer(384)
            mock_model.encode.return_value = np.random.rand(2, 384).astype(np.float64)
            mock_st.return_value = mock_model

            embedder = Embedder()
            result = embedder.embed(["a", "b"])

        assert result.dtype == np.float32

    def test_embed_one_returns_single_vector_not_batch(self):
        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_model = _create_mock_transformer(384)
            mock_model.encode.return_value = np.random.rand(1, 384)
            mock_st.return_value = mock_model

            embedder = Embedder()
            result = embedder.embed_one("single text")

        assert result.shape == (384,)  # 1D, not (1, 384)

    def test_dim_attribute_matches_model(self):
        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_model = _create_mock_transformer(384)
            mock_st.return_value = mock_model

            embedder = Embedder()

        assert embedder.dim == 384


@pytest.mark.slow
class TestEmbedderRealModel:
    """
    Actually loads the real sentence-transformers model.
    Run explicitly with: pytest demo/tests/test_embedder.py -m slow
    Excluded from default test runs — downloads a model on first call.
    """

    def test_real_model_produces_384_dim_vectors(self):
        embedder = Embedder()
        result = embedder.embed_one("This is a real test sentence.")
        assert result.shape == (384,)
        assert embedder.dim == 384

    def test_similar_sentences_have_higher_cosine_similarity_than_dissimilar(self):
        """Sanity check the embedding model itself produces sensible geometry."""
        embedder = Embedder()
        v1 = embedder.embed_one("The cat sat on the mat.")
        v2 = embedder.embed_one("A feline rested on the rug.")
        v3 = embedder.embed_one("Quantum mechanics describes subatomic particles.")

        def cosine(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        sim_related = cosine(v1, v2)
        sim_unrelated = cosine(v1, v3)
        assert sim_related > sim_unrelated, (
            "Semantically related sentences should be closer than unrelated ones — "
            "if this fails, the embedding model itself is the problem, not Vektor."
        )
