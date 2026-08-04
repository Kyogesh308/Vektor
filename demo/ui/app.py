"""
demo/ui/app.py
--------------
Streamlit demo: upload PDF, index document into Vektor, and perform live semantic retrieval.

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

st.set_page_config(page_title="Vektor Semantic Search", page_icon="🔍", layout="wide")

st.title("🔍 Vektor — Semantic Document Search Demo")

st.markdown(
    "Upload any PDF document to chunk and index it into Vektor using dense vector embeddings. "
    "Ask natural language queries to retrieve the most semantically relevant text passages."
)

st.sidebar.header("Document & Index Settings")
uploaded_file = st.file_uploader("Upload a PDF Document", type="pdf")

if uploaded_file and st.button("🚀 Build Vector Index", type="primary"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    if "client" in st.session_state:
        try:
            st.session_state["client"].close()
        except Exception:
            pass

    with st.spinner("Extracting text, generating embeddings, and building Vektor index..."):
        client, collection = build_index(tmp_path)
        st.session_state["client"] = client
        st.session_state["collection"] = collection
        st.session_state["embedder"] = Embedder()
        st.session_state["doc_name"] = uploaded_file.name

    st.success(f"Successfully indexed **{collection.count()} chunks** from `{uploaded_file.name}` into Vektor!")

if "collection" in st.session_state:
    st.divider()

    col1, col2 = st.columns([3, 1])

    with col2:
        st.markdown("### Index Info")
        st.info(
            f"**Document:** `{st.session_state.get('doc_name', 'Uploaded PDF')}`\n\n"
            f"**Total Chunks:** `{st.session_state['collection'].count()}`\n\n"
            f"**Vector Dimension:** `{st.session_state['embedder'].dim}`\n\n"
            f"**Distance Metric:** `Cosine`"
        )
        k_val = st.slider("Top Results (k)", min_value=1, max_value=10, value=5)

    with col1:
        st.markdown("### 💬 Ask a Question")
        query = st.text_input(
            "Enter your query about the document:",
            placeholder="e.g. What is the name of the director? What are the key findings?"
        )

        if query:
            embedder = st.session_state["embedder"]
            query_vector = embedder.embed_one(query)
            results = st.session_state["collection"].search(query_vector, k=k_val)

            st.markdown(f"#### Top {len(results)} Retrieved Passages")

            for rank, r in enumerate(results, start=1):
                chunk_text_content = r.metadata.get("text", "")
                start_char = r.metadata.get("start_char", 0)
                score_pct = max(0.0, min(1.0, float(r.score)))

                with st.expander(
                    f"Rank #{rank} | Relevance Score: {r.score:.4f} ({score_pct:.1%}) — ID: {r.id}",
                    expanded=(rank == 1)
                ):
                    st.progress(score_pct)
                    st.markdown("**Full Chunk Content:**")
                    st.text_area(
                        label="Chunk Text",
                        value=chunk_text_content,
                        height=180,
                        key=f"chunk_text_{r.id}_{rank}"
                    )
                    st.caption(f"Chunk ID: `{r.id}` | Start Character: `{start_char}` | Length: `{len(chunk_text_content)} chars`")
