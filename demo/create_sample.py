"""
Script to create sample.pdf for the demo and help author golden_set.py accurately.
"""

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def make_sample_pdf(output_path: Path):
    doc = SimpleDocTemplate(str(output_path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=20,
        spaceAfter=15
    )
    heading_style = ParagraphStyle(
        'DocHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceBefore=12,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        spaceAfter=10
    )

    story.append(Paragraph("Vektor Technical Report: RAG and Vector Search Architecture", title_style))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Section 1: Introduction to Vector Search in Modern RAG Pipelines", heading_style))
    story.append(Paragraph(
        "Retrieval-Augmented Generation (RAG) combines dense semantic retrieval with large language models to provide accurate, grounded answers. "
        "Unlike keyphrase searching, dense vector retrieval converts unstructured text chunks into high-dimensional numerical embeddings using deep neural networks. "
        "The primary challenge in production RAG systems is balancing high query throughput, low search latency, and memory efficiency across large-scale document collections.",
        body_style
    ))

    story.append(Paragraph("Section 2: Hierarchical Navigable Small World (HNSW) Graph Indexing", heading_style))
    story.append(Paragraph(
        "Vektor utilizes a Hierarchical Navigable Small World (HNSW) graph for approximate nearest neighbor (ANN) retrieval. "
        "HNSW constructs a multi-layer graph where upper layers contain long-range skip edges for logarithmic greedy routing, while the bottom layer contains a dense proximity graph. "
        "Key hyper-parameters governing HNSW performance include M (max number of bi-directional connections per node) and efConstruction (size of dynamic candidate list during build). "
        "Increasing M improves recall@10 on complex metric spaces but increases index memory footprint.",
        body_style
    ))

    story.append(Paragraph("Section 3: Memory Layout and SIMD Vector Distance Computations", heading_style))
    story.append(Paragraph(
        "To achieve maximum hardware utilization during vector distance calculations, Vektor arranges vector storage in contiguous 64-byte cache-aligned memory blocks. "
        "Distance metrics such as Euclidean distance (L2), Cosine similarity, and Inner Product (IP) are accelerated using AVX-512 and FMA (Fused Multiply-Add) vector intrinsics. "
        "By enforcing contiguous memory layout and avoiding pointer indirection during distance loops, Vektor achieves up to 4x throughput improvements over naive pointer-array implementations.",
        body_style
    ))

    story.append(Paragraph("Section 4: Vector Quantization and Product Quantization (PQ)", heading_style))
    story.append(Paragraph(
        "When collection sizes exceed system RAM, Vektor employs Scalar Quantization (SQ8) and Product Quantization (PQ) to compress high-dimensional floating-point vectors into compact byte codes. "
        "Product Quantization divides a high-dimensional vector space into lower-dimensional sub-spaces, quantizing each sub-space independently into cluster centroids. "
        "This compression reduces RAM requirements by up to 80% while retaining over 95% of top-k retrieval accuracy for similarity search operations.",
        body_style
    ))

    story.append(Paragraph("Section 5: Empirical Evaluation and Semantic Relevance Methodology", heading_style))
    story.append(Paragraph(
        "Evaluating a vector database requires separating mathematical recall from semantic relevance. "
        "Mathematical recall@k verifies that the ANN index returns the same nearest neighbors as brute-force exact search. "
        "In contrast, semantic relevance measures whether the retrieved text chunks satisfy human user queries. "
        "To evaluate semantic accuracy rigorously, researchers use human-labeled golden query sets scored with Precision@k metrics, independent of model self-evaluation.",
        body_style
    ))

    story.append(Paragraph("Section 6: Conclusion and System Recommendations", heading_style))
    story.append(Paragraph(
        "Vektor demonstrates that combining optimized C++/Rust core index execution with clean Python interfaces provides both developer agility and high retrieval performance. "
        "For production RAG workloads, configuring HNSW with M=16 and ef_search=64 offers an optimal tradeoff between sub-millisecond query latency and high precision retrieval.",
        body_style
    ))

    doc.build(story)
    print(f"Sample PDF created at {output_path}")

if __name__ == "__main__":
    out = Path(__file__).parent / "data" / "sample.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    make_sample_pdf(out)
