"""
demo/eval/run_eval.py
------------------------
Runs golden-set and auto-generated queries against a Vektor collection,
logs structured results. Keeps the two sources separate throughout.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from demo.ingestion.embedder import Embedder
from demo.eval.golden_set import GOLDEN_SET

RESULTS_DIR = Path(__file__).parent.parent / "results"


def run_golden_set_eval(collection, embedder: Embedder, k: int = 5) -> dict:
    """
    Run the hand-authored golden set. Computes precision@k against
    human-labeled expected chunk IDs.
    """
    results = []
    total_precision = 0.0

    for item in GOLDEN_SET:
        query_vector = embedder.embed_one(item["query"])
        search_results = collection.search(query_vector, k=k)

        retrieved_ids = [r.id for r in search_results]
        expected = set(item["expected_chunk_ids"])
        hits = [rid for rid in retrieved_ids if rid in expected]
        precision = len(hits) / k

        results.append({
            "query": item["query"],
            "expected_chunk_ids": item["expected_chunk_ids"],
            "retrieved": [
                {"rank": i + 1, "id": r.id, "score": r.score,
                 "text_preview": r.metadata.get("text", "")[:150],
                 "matches_expected": r.id in expected}
                for i, r in enumerate(search_results)
            ],
            "precision_at_k": precision,
        })
        total_precision += precision

    return {
        "source": "golden_set",
        "methodology": "Hand-authored by a human reading the source document. "
                       "This is the evidentiary component for semantic accuracy claims.",
        "mean_precision_at_k": total_precision / len(GOLDEN_SET) if GOLDEN_SET else 0.0,
        "n_queries": len(GOLDEN_SET),
        "results": results,
    }


def run_auto_questionnaire_eval(collection, embedder: Embedder,
                                 questions: list[str], k: int = 5) -> dict:
    """
    Run auto-generated questions. NO ground truth exists for these —
    results are qualitative/demo only, never cited as accuracy evidence.
    """
    results = []
    for q in questions:
        query_vector = embedder.embed_one(q)
        search_results = collection.search(query_vector, k=k)
        results.append({
            "query": q,
            "retrieved": [
                {"rank": i + 1, "id": r.id, "score": r.score,
                 "text_preview": r.metadata.get("text", "")[:150]}
                for i, r in enumerate(search_results)
            ],
        })

    return {
        "source": "auto_generated_questionnaire",
        "methodology": "LLM-generated questions from the same document. "
                       "NO independent ground truth — supplementary demo output only. "
                       "Do not cite as evidence of semantic accuracy.",
        "n_queries": len(questions),
        "results": results,
    }


def save_report(golden_report: dict, auto_report: dict) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output = {
        "timestamp": timestamp,
        "golden_set_evaluation": golden_report,
        "auto_questionnaire_demo": auto_report,
    }
    path = RESULTS_DIR / f"eval_run_{timestamp}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return path
