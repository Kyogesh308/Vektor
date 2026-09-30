"""
demo/ingestion/pipeline.py
--------------------------
Ingests raw text documents, computes embeddings, and inserts them 
into a Vektor collection using `batch_insert`.
"""

import uuid
from typing import List, Optional
from vektor.collection_api import Collection
from demo.ingestion.embedder import Embedder

def ingest_documents(
    texts: List[str], 
    collection: Collection, 
    embedder: Optional[Embedder] = None
) -> int:
    """
    Takes a list of text snippets, computes their embeddings, and batch inserts
    them into the provided Vektor collection. The original text is stored in the
    'text' metadata field.

    Args:
        texts: A list of text strings to ingest.
        collection: The Vektor Collection instance to insert into.
        embedder: An Embedder instance. If None, one will be initialized.

    Returns:
        The number of successfully inserted documents.
    """
    if not texts:
        return 0

    if embedder is None:
        embedder = Embedder()

    # Generate embeddings
    vectors = embedder.embed(texts)

    # Prepare records for batch insertion
    records = []
    for i, text in enumerate(texts):
        # Generate a unique ID for each vector
        doc_id = str(uuid.uuid4())
        
        records.append({
            "id": doc_id,
            "vector": vectors[i].tolist(),
            "metadata": {"text": text}
        })

    # Perform batch insert
    result = collection.batch_insert(records)
    
    return result.inserted
