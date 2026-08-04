"""
demo/eval/golden_set.py
--------------------------
Hand-authored query -> expected-relevant-chunk-ID pairs.

THIS is the evidence for semantic correctness, not the auto-generated
questionnaire. Written by a human reading the source document, independent
of any model's judgment.
"""

GOLDEN_SET = [
    {
        "query": "What is Retrieval-Augmented Generation and how does it work?",
        "expected_chunk_ids": ["chunk_0"],
    },
    {
        "query": "What is the main challenge in production RAG systems?",
        "expected_chunk_ids": ["chunk_0", "chunk_1"],
    },
    {
        "query": "What graph data structure does Vektor use for approximate nearest neighbor retrieval?",
        "expected_chunk_ids": ["chunk_1"],
    },
    {
        "query": "What are the key hyper-parameters governing HNSW performance?",
        "expected_chunk_ids": ["chunk_2"],
    },
    {
        "query": "How does Vektor optimize memory layout for vector distance computations?",
        "expected_chunk_ids": ["chunk_3"],
    },
    {
        "query": "Which SIMD hardware instructions accelerate distance metrics in Vektor?",
        "expected_chunk_ids": ["chunk_3"],
    },
    {
        "query": "What methods does Vektor use when vector collection sizes exceed RAM?",
        "expected_chunk_ids": ["chunk_4"],
    },
    {
        "query": "How does Product Quantization compress vectors and how much RAM is saved?",
        "expected_chunk_ids": ["chunk_4", "chunk_5"],
    },
    {
        "query": "What is the difference between mathematical recall and semantic relevance?",
        "expected_chunk_ids": ["chunk_5", "chunk_6"],
    },
    {
        "query": "How should researchers evaluate semantic accuracy of vector retrieval systems?",
        "expected_chunk_ids": ["chunk_6"],
    },
    {
        "query": "What programming languages are combined in Vektor's architecture?",
        "expected_chunk_ids": ["chunk_6", "chunk_7"],
    },
    {
        "query": "What are the recommended HNSW configuration parameters for production RAG workloads?",
        "expected_chunk_ids": ["chunk_7"],
    },
]
