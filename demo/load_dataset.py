"""
demo/load_dataset.py
--------------------
Downloads a public dataset of real documents (e.g. AG News) and feeds them 
into the Comparison Lab's Collection via the demo pipeline.

Usage:
  python demo/load_dataset.py --dataset ag_news --count 10000 --collection demo_compare
"""

import argparse
from datasets import load_dataset
from vektor import Vektor, VektorConfig
from demo.ingestion.pipeline import ingest_documents
from demo.ingestion.embedder import Embedder

def load_data(dataset_name: str, count: int, collection_name: str, batch_size: int = 1000):
    print(f"Loading {count} examples from {dataset_name}...")
    
    # Load dataset
    ds = load_dataset(dataset_name, split="train")
    if len(ds) > count:
        ds = ds.select(range(count))
        
    texts = []
    for item in ds:
        # AG news has 'text' column, SciQ has 'question' or 'support'
        if "text" in item:
            texts.append(item["text"])
        elif "question" in item:
            texts.append(item["question"])
        else:
            # Fallback for other datasets
            texts.append(str(list(item.values())[0]))
            
    print(f"Initializing Vektor and Embedder...")
    client = Vektor(VektorConfig(data_dir="./demo_vektor_data"))
    embedder = Embedder()
    
    if collection_name not in client.list_collections():
        print(f"Creating collection '{collection_name}'...")
        collection = client.create_collection(collection_name, dim=embedder.dim, metric="cosine")
    else:
        print(f"Using existing collection '{collection_name}'...")
        collection = client.get_collection(collection_name)
        
    print(f"Starting batch ingestion of {len(texts)} documents (this will take a while)...")
    
    total_inserted = 0
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        print(f"Ingesting batch {i//batch_size + 1} ({len(batch_texts)} documents)...")
        inserted = ingest_documents(batch_texts, collection, embedder)
        total_inserted += inserted
        
    print(f"\nSuccessfully ingested {total_inserted} documents into '{collection_name}'.")
    client.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load public datasets into Vektor for comparison.")
    parser.add_argument("--dataset", type=str, default="ag_news", help="HuggingFace dataset name")
    parser.add_argument("--count", type=int, default=10000, help="Number of documents to load")
    parser.add_argument("--collection", type=str, default="demo_compare", help="Vektor collection name")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for embeddings")
    
    args = parser.parse_args()
    load_data(args.dataset, args.count, args.collection, args.batch_size)
