"""
demo/run_demo_eval.py
-----------------------
Main entrypoint to run the end-to-end semantic evaluation.
Builds vector index from sample.pdf, runs golden set eval & auto questionnaire,
and outputs formatted reports in demo/results/.
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

from demo.build_index import build_index
from demo.ingestion.embedder import Embedder
from demo.eval.run_eval import run_golden_set_eval, run_auto_questionnaire_eval, save_report
from demo.eval.report import generate_markdown_report



def main():
    sample_pdf = Path(__file__).parent / "data" / "sample.pdf"
    if not sample_pdf.exists():
        from demo.create_sample import make_sample_pdf
        make_sample_pdf(sample_pdf)

    print("Step 1: Building Vektor index from PDF...")
    client, collection = build_index(str(sample_pdf))

    embedder = Embedder()

    print("Step 2: Running hand-authored Golden Set Evaluation (Evidentiary)...")
    golden_report = run_golden_set_eval(collection, embedder, k=5)
    print(f"Mean Precision@5: {golden_report['mean_precision_at_k']:.3f}")

    print("Step 3: Running Auto-Generated Questionnaire (Demo Only)...")
    sample_auto_questions = [
        "How are text chunks converted into embeddings?",
        "What are the benefits of AVX-512 intrinsics?",
        "How much memory does Product Quantization save?",
        "What is the recommended value for M in HNSW?",
    ]
    auto_report = run_auto_questionnaire_eval(collection, embedder, sample_auto_questions, k=5)

    print("Step 4: Saving structured reports...")
    json_path = save_report(golden_report, auto_report)
    print(f"JSON Report saved to {json_path}")

    md_content = generate_markdown_report({
        "timestamp": json_path.stem.replace("eval_run_", ""),
        "golden_set_evaluation": golden_report,
        "auto_questionnaire_demo": auto_report
    })

    md_path = json_path.with_suffix(".md")
    md_path.write_text(md_content, encoding="utf-8")
    print(f"Markdown Report saved to {md_path}")
    print("\n--- Markdown Report Preview ---\n")
    print(md_content)


if __name__ == "__main__":
    main()
