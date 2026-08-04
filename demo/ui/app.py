"""
demo/ui/app.py
-----------------
Streamlit demo: upload PDF, build index, query live, view golden-set report.

Run: streamlit run demo/ui/app.py
"""

import sys
import tempfile
from pathlib import Path

# Ensure project root and src are in sys.path when running via streamlit
project_root = Path(__file__).resolve().parent.parent.parent
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st

from demo.build_index import build_index
from demo.ingestion.embedder import Embedder
from demo.eval.golden_set import GOLDEN_SET
from demo.eval.run_eval import run_golden_set_eval


st.title("Vektor — Semantic Retrieval Demo")

st.markdown(
    "**Note:** this demo shows retrieval quality on real documents, "
    "distinct from the mathematical recall@10 benchmarks in the main report."
)

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file and st.button("Build Index"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    if "client" in st.session_state:
        try:
            st.session_state["client"].close()
        except Exception:
            pass

    with st.spinner("Indexing..."):
        client, collection = build_index(tmp_path)
        st.session_state["client"] = client
        st.session_state["collection"] = collection
        st.session_state["embedder"] = Embedder()
    st.success("Index built.")

if "collection" in st.session_state:
    st.subheader("Live Query")
    query = st.text_input("Ask a question about the document")
    if query:
        embedder = st.session_state["embedder"]
        query_vector = embedder.embed_one(query)
        results = st.session_state["collection"].search(query_vector, k=5)
        for r in results:
            st.markdown(f"**Score:** {r.score:.4f} — {r.metadata.get('text', '')[:200]}...")

    st.subheader("Golden Set Evaluation")
    if st.button("Run Golden Set Eval"):
        report = run_golden_set_eval(st.session_state["collection"],
                                     st.session_state["embedder"])
        st.metric("Mean Precision@k", f"{report['mean_precision_at_k']:.3f}")
        for r in report["results"]:
            icon = "PASS" if r["precision_at_k"] > 0 else "FAIL"
            st.write(f"[{icon}] {r['query']} — precision={r['precision_at_k']:.2f}")
