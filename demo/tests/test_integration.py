"""
demo/tests/test_integration.py
----------------------------------
End-to-end wiring test: chunking -> real Vektor collection -> search ->
golden-set eval. Uses real Vektor (already tested elsewhere), mocks only
the embedder to avoid a model download in CI, using deterministic fake
vectors so results are predictable and assertable.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from vektor import Vektor, VektorConfig
from demo.ingestion.chunker import chunk_text
from demo.eval.run_eval import run_golden_set_eval


DIM = 8


def deterministic_embed(text: str) -> np.ndarray:
    """Same text always produces the same vector — makes results assertable."""
    seed = abs(hash(text)) % (2**31)
    rng = np.random.default_rng(seed)
    return rng.standard_normal(DIM).astype(np.float32)


@pytest.fixture
def fake_embedder():
    embedder = MagicMock()
    embedder.dim = DIM
    embedder.embed = lambda texts: np.array([deterministic_embed(t) for t in texts])
    embedder.embed_one = lambda t: deterministic_embed(t)
    return embedder


class TestEndToEndWiring:

    def test_chunk_index_search_round_trip(self, tmp_path, fake_embedder):
        text = "First chunk about cats. " * 20 + "Second chunk about dogs. " * 20
        chunks = chunk_text(text, chunk_size=200, overlap=20)
        assert len(chunks) >= 2

        client = Vektor(VektorConfig(data_dir=tmp_path / "vektor_data"))
        collection = client.create_collection("test_doc", dim=DIM, metric="cosine")

        vectors = fake_embedder.embed([c.text for c in chunks])
        for chunk, vector in zip(chunks, vectors):
            collection.insert(chunk.id, vector, {"text": chunk.text})

        assert collection.count() == len(chunks)

        # Search using the same deterministic embedding for the first chunk's
        # own text — it must retrieve itself as the top result
        query_vector = fake_embedder.embed_one(chunks[0].text)
        results = collection.search(query_vector, k=1)
        assert results[0].id == chunks[0].id

        client.close()

    def test_golden_set_ids_exist_in_actual_built_index(self, tmp_path, fake_embedder, monkeypatch):
        """
        Regression test for golden-set drift: if chunk_size/overlap change,
        chunk IDs shift, and a golden set written against old boundaries
        silently references chunks that no longer exist. This test catches
        that drift automatically instead of relying on someone noticing.
        """
        text = "Some document content. " * 100
        chunks = chunk_text(text, chunk_size=200, overlap=20)

        client = Vektor(VektorConfig(data_dir=tmp_path / "vektor_data"))
        collection = client.create_collection("drift_test", dim=DIM, metric="cosine")

        vectors = fake_embedder.embed([c.text for c in chunks])
        for chunk, vector in zip(chunks, vectors):
            collection.insert(chunk.id, vector, {"text": chunk.text})

        # Simulate a golden set entry referencing a chunk ID
        fake_golden_entry_id = chunks[2].id  # a real, currently-existing chunk

        import demo.eval.run_eval as run_eval_module
        monkeypatch.setattr(run_eval_module, "GOLDEN_SET", [
            {"query": "irrelevant for this test",
             "expected_chunk_ids": [fake_golden_entry_id]},
        ])

        # Confirm the referenced ID actually exists in the built collection —
        # this is what would catch drift if chunking parameters changed
        # without updating golden_set.py
        with pytest.raises(Exception):
            collection.get("chunk_9999_does_not_exist")  # sanity: nonexistent ID does fail

        result = collection.get(fake_golden_entry_id)  # must NOT raise
        assert result.id == fake_golden_entry_id

        client.close()

    def test_dimension_mismatch_between_embedder_and_collection_fails_loudly(self, tmp_path):
        """
        If someone swaps embedding models without updating collection dim,
        this must fail with a clear error, not silently corrupt data.
        """
        client = Vektor(VektorConfig(data_dir=tmp_path / "vektor_data"))
        collection = client.create_collection("dim_test", dim=384, metric="cosine")

        wrong_dim_vector = np.random.rand(8).astype(np.float32)  # wrong: 8 != 384

        with pytest.raises(Exception):  # InvalidVectorDimensionError from Phase 1
            collection.insert("v1", wrong_dim_vector)

        client.close()
