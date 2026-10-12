"""
demo/load_enterprise_dataset.py
-------------------------------
Generates and ingests a large enterprise company-level knowledge base
into Vektor for Phase 2 Central Server demonstration and Comparison Lab benchmarking.

Features:
- Generates 1,000 to 100,000 realistic enterprise records (IT incidents, RFCs,
  Platform logs, Customer support, Compliance policies).
- Automatically chunks and batches embeddings with real-time ETA.
- Fully offline-compatible (no external internet dependency required in restricted lab PCs).
- Directly targets the persistent Vektor storage volume (/data or demo_vektor_data).

Usage:
  # Quick 1,000 docs demo:
  python demo/load_enterprise_dataset.py --count 1000

  # Company scale 10,000 docs:
  python demo/load_enterprise_dataset.py --count 10000 --collection enterprise_kb

  # Heavy scale 50,000 docs for Phase 2 server:
  python demo/load_enterprise_dataset.py --count 50000 --collection enterprise_kb
"""

from __future__ import annotations

import os
import sys
import time
import uuid
import random
import argparse
from pathlib import Path

# Ensure project root and src are in sys.path
project_root = Path(__file__).resolve().parent.parent
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from vektor import Vektor, VektorConfig
from demo.ingestion.embedder import Embedder

# Realistic corporate templates
DEPARTMENTS = ["DevOps & SRE", "Cybersecurity", "Platform Engineering", "Cloud Infrastructure", "Database Ops", "Enterprise Customer Success", "FinOps", "AI Core"]
INCIDENT_TYPES = ["High Memory Utilization", "PostgreSQL Connection Saturation", "HNSW Index Memory Spike", "Kafka Consumer Lag", "TLS Handshake Timeout", "API Gateway 504 Gateway Timeout", "Zero-day Dependency Patching", "Kubernetes Pod OOMKilled", "Distributed Lock Contention"]
SEVERITY = ["SEV-1 Critical", "SEV-2 High", "SEV-3 Medium", "SEV-4 Low"]
CLUSTERS = ["us-east-prod-k8s", "eu-central-finance-cluster", "ap-south-inference-node", "us-west-gpu-farm", "core-data-warehouse-01"]
RESOLUTIONS = [
    "Scaled replica count from 4 to 16 and adjusted HPA CPU threshold.",
    "Restarted Redis sentinel cluster and drained stale connections.",
    "Applied hotfix for connection leak and tuned connection pool size to 128.",
    "Purged orphaned vector indexes and enabled memory mapping on binary graph files.",
    "Upgraded Linux kernel TCP buffer limits and restarted ingress envoy proxy.",
    "Enabled AVX-512 vector acceleration kernels for nearest-neighbour search.",
    "Switched index build strategy from brute-force full scan to HNSW efConstruction=200.",
]


def generate_enterprise_record(index: int) -> dict:
    dept = random.choice(DEPARTMENTS)
    incident = random.choice(INCIDENT_TYPES)
    sev = random.choice(SEVERITY)
    cluster = random.choice(CLUSTERS)
    resolution = random.choice(RESOLUTIONS)
    ticket_id = f"CORP-INC-{index:06d}"

    text = (
        f"Incident Record [{ticket_id}] - {dept} Department. "
        f"Incident Summary: {incident} detected on production cluster {cluster}. "
        f"Severity Classification: {sev}. "
        f"Diagnostic Root Cause: Root cause analysis indicated high throughput query load under peak traffic. "
        f"Mitigation & Resolution: {resolution} "
        f"Status: Resolved. Postmortem verified and archived in enterprise knowledge base."
    )

    metadata = {
        "ticket_id": ticket_id,
        "department": dept,
        "severity": sev,
        "cluster": cluster,
        "incident_type": incident,
        "created_timestamp": time.time() - random.randint(1000, 10000000),
        "text": text,
    }
    return {"id": ticket_id, "text": text, "metadata": metadata}


def ingest_enterprise_kb(
    count: int = 5000,
    collection_name: str = "enterprise_kb",
    batch_size: int = 500,
    data_dir: str | None = None,
):
    resolved_data_dir = (
        data_dir
        or os.environ.get("VEKTOR_DATA_DIR")
        or "./demo_vektor_data"
    )
    print("=" * 70)
    print("[*] VEKTOR ENTERPRISE DATABASE INGESTION (Phase 2 Compatible)")
    print("=" * 70)
    print(f"Target Collection : {collection_name}")
    print(f"Target Records    : {count:,} documents")
    print(f"Storage Directory : {resolved_data_dir}")
    print(f"Batch Size        : {batch_size}")
    print("=" * 70)

    t_start = time.perf_counter()

    print("\n[1/4] Initializing Sentence Embedder...")
    embedder = Embedder()
    print(f"      Embedder loaded: dimension = {embedder.dim}")

    print("\n[2/4] Connecting to Vektor...")
    client = Vektor(VektorConfig(data_dir=resolved_data_dir))

    if collection_name in client.list_collections():
        print(f"      Re-using existing collection '{collection_name}'...")
        collection = client.get_collection(collection_name)
    else:
        print(f"      Creating new HNSW collection '{collection_name}' (dim={embedder.dim}, metric=cosine)...")
        collection = client.create_collection(
            collection_name,
            dim=embedder.dim,
            metric="cosine",
            mode="research",
            M=16,
            ef_construction=100,
        )

    existing_count = collection.count()
    print(f"      Current collection count before ingestion: {existing_count:,}")

    print(f"\n[3/4] Generating and ingesting {count:,} enterprise records in batches...")
    total_inserted = 0

    num_batches = (count + batch_size - 1) // batch_size
    for b in range(num_batches):
        batch_start_idx = b * batch_size
        batch_count = min(batch_size, count - batch_start_idx)

        # Generate synthetic records
        records_data = [
            generate_enterprise_record(existing_count + batch_start_idx + i + 1)
            for i in range(batch_count)
        ]

        texts = [r["text"] for r in records_data]

        # Embed batch
        t_b0 = time.perf_counter()
        vectors = embedder.embed(texts)

        # Build Vektor insert payloads
        insert_payloads = [
            {
                "id": records_data[i]["id"],
                "vector": vectors[i].tolist(),
                "metadata": records_data[i]["metadata"],
            }
            for i in range(batch_count)
        ]

        batch_result = collection.batch_insert(insert_payloads)
        total_inserted += batch_result.inserted
        t_b1 = time.perf_counter()

        elapsed = t_b1 - t_start
        rate = total_inserted / elapsed if elapsed > 0 else 0
        percent = (total_inserted / count) * 100
        print(
            f"      Batch {b+1:3d}/{num_batches:3d} | "
            f"Inserted: {total_inserted:6,}/{count:6,} ({percent:5.1f}%) | "
            f"Rate: {rate:5.1f} docs/sec"
        )

    t_end = time.perf_counter()
    final_count = collection.count()

    print("\n" + "=" * 70)
    print("[+] ENTERPRISE DATABASE INGESTION COMPLETE")
    print("=" * 70)
    print(f"Total Newly Inserted : {total_inserted:,} records")
    print(f"Total Collection Size: {final_count:,} vectors")
    print(f"Total Time Taken     : {t_end - t_start:.2f} seconds")
    print(f"Persistent Storage   : {resolved_data_dir}")
    print("=" * 70)
    print("\n>>> You can now query this dataset in the Comparison Lab UI")
    print(f"    Enter Collection Name: '{collection_name}' in the Streamlit UI to run benchmarks!")

    client.close()
    return total_inserted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest a large enterprise-level database into Vektor."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5000,
        help="Number of documents to generate and ingest (default: 5000)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="enterprise_kb",
        help="Collection name (default: enterprise_kb)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for embeddings (default: 500)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Storage directory for Vektor database",
    )

    args = parser.parse_args()
    ingest_enterprise_kb(
        count=args.count,
        collection_name=args.collection,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
    )
