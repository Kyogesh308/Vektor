import pytest
from vektor import Vektor, VektorConfig
from demo.ingestion.pipeline import ingest_documents
from demo.ingestion.embedder import Embedder

def test_ingest_documents_pipeline(tmp_path):
    config = VektorConfig(data_dir=tmp_path / "vektor_data")
    client = Vektor(config)
    
    # We use a dummy embedder or actual Embedder for test
    # Actually, initializing sentence-transformers inside the test might be slow,
    # but it's acceptable for a single integration test on a small model.
    embedder = Embedder()
    
    collection = client.create_collection("test_ingestion", dim=embedder.dim, metric="cosine")
    
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Python is a high-level programming language.",
        "Vektor is a vector database built from scratch."
    ]
    
    inserted_count = ingest_documents(sample_texts, collection, embedder)
    
    assert inserted_count == len(sample_texts)
    assert collection.count() == len(sample_texts)
    
    # Retrieve via ordinary Collection.get()
    all_ids = collection._store.ids() # Use _store to get all ids since we generated uuids
    assert len(all_ids) == 3
    
    # Verify metadata text
    retrieved_texts = []
    for vec_id in all_ids:
        res = collection.get(vec_id)
        assert "text" in res.metadata
        retrieved_texts.append(res.metadata["text"])
        
    for text in sample_texts:
        assert text in retrieved_texts
        
    # Verify search
    query_vec = embedder.embed_one("Tell me about Python.")
    results = collection.search(query_vec, k=1)
    assert len(results) == 1
    assert "text" in results[0].metadata
    assert "Python" in results[0].metadata["text"]
    
    client.close()
