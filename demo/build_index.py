"""
demo/build_index.py
----------------------
End-to-end: PDF -> chunks -> vectors -> Vektor collection.
"""

import sys
from pathlib import Path

# Ensure project root and src are in sys.path
project_root = Path(__file__).resolve().parent.parent
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from vektor import Vektor, VektorConfig
from demo.ingestion.pdf_loader import extract_text
from demo.ingestion.chunker import chunk_text
from demo.ingestion.embedder import Embedder



def build_index(pdf_path: str, collection_name: str = "demo_doc") -> None:
    text = extract_text(pdf_path)
    chunks = chunk_text(text)

    embedder = Embedder()
    vectors = embedder.embed([c.text for c in chunks])

    client = Vektor(VektorConfig(data_dir="./demo_vektor_data"))

    # Recreate cleanly each run for demo purposes
    if collection_name in client.list_collections():
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

    try:
        collection = client.create_collection(collection_name, dim=embedder.dim, metric="cosine")
    except Exception:
        collection = client.get_collection(collection_name)

    for chunk, vector in zip(chunks, vectors):
        try:
            collection.insert(chunk.id, vector, {"text": chunk.text, "start_char": chunk.start_char})
        except Exception:
            pass

    print(f"Indexed {len(chunks)} chunks into '{collection_name}'.")
    return client, collection


if __name__ == "__main__":
    build_index("demo/data/sample.pdf")
