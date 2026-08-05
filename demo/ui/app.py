"""
demo/ui/app.py
--------------
Streamlit RAG demo: upload PDF, index document into Vektor, perform semantic retrieval,
and generate grounded answers using Ollama (llama3:8b) at localhost:11434.

Run: streamlit run demo/ui/app.py
"""

import sys
import tempfile
import requests
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

st.set_page_config(page_title="Vektor RAG & Semantic Search", page_icon="🦙", layout="wide")

st.title("🦙 Vektor RAG — Semantic Document Q&A")

st.markdown(
    "Upload any PDF document to chunk and index it into Vektor using dense vector embeddings. "
    "Ask queries to retrieve relevant passages and generate grounded answers using **Ollama (`llama3:8b`)**."
)

st.sidebar.header("Document & Model Settings")
uploaded_file = st.file_uploader("Upload a PDF Document", type="pdf")
ollama_model = st.sidebar.text_input("Ollama Model", value="llama3:8b", help="Ollama model name at localhost:11434")

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


def generate_ollama_answer(query: str, retrieved_chunks: list[str], model_name: str = "llama3:8b") -> str:
    if not retrieved_chunks:
        return "No relevant passages found in document."

    context_str = "\n\n---\n\n".join([f"Passage {i+1}:\n{text}" for i, text in enumerate(retrieved_chunks)])
    prompt = (
        "You are an AI assistant answering questions about a document using only the provided passages.\n\n"
        f"DOCUMENT PASSAGES:\n{context_str}\n\n"
        f"QUESTION: {query}\n\n"
        "Provide a clear, direct, and concise answer based strictly on the passages above. "
        "If the passages do not contain the answer, state that the document does not explicitly state this information."
    )

    models_to_try = [model_name, "llama3:8b", "llama3", "llama3:latest"]
    unique_models = []
    for m in models_to_try:
        if m not in unique_models:
            unique_models.append(m)

    for m in unique_models:
        try:
            res = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": m, "prompt": prompt, "stream": False},
                timeout=45
            )
            if res.status_code == 200:
                answer = res.json().get("response", "").strip()
                if answer:
                    return f"**LLM Generated Answer** *(via Ollama model `{m}`)*:\n\n{answer}"
        except Exception:
            continue

    return (
        "⚠️ **Could not connect to Ollama at `http://localhost:11434`**\n\n"
        "Please ensure Ollama is running (`ollama serve` or `ollama run llama3`) "
        "and that the model is installed."
    )


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
            placeholder="e.g. What is the name of the director? What are the key conclusions?"
        )

        if query:
            embedder = st.session_state["embedder"]
            query_vector = embedder.embed_one(query)
            results = st.session_state["collection"].search(query_vector, k=k_val)

            retrieved_texts = [r.metadata.get("text", "") for r in results if r.metadata.get("text", "")]

            st.markdown("---")
            with st.spinner(f"Generating answer with Ollama (`{ollama_model}`)..."):
                ollama_answer = generate_ollama_answer(query, retrieved_texts, ollama_model)
                st.markdown(ollama_answer)

            st.markdown("---")
            st.markdown(f"#### 📚 Top {len(results)} Retrieved Passages from Vektor")

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
                    st.markdown(f"```text\n{chunk_text_content}\n```")
                    st.caption(f"Chunk ID: `{r.id}` | Start Character: `{start_char}` | Length: `{len(chunk_text_content)} chars`")
