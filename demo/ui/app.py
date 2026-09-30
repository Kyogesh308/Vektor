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

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["RAG Demo (Q&A)", "Comparison Lab"])

st.sidebar.header("Settings")
ollama_model = st.sidebar.text_input("Ollama Model", value="llama3:8b", help="Ollama model name at localhost:11434")

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

if page == "RAG Demo (Q&A)":
    st.title("🦙 Vektor RAG — Semantic Document Q&A")

    st.markdown(
        "Upload any PDF document to chunk and index it into Vektor using dense vector embeddings. "
        "Ask queries to retrieve relevant passages and generate grounded answers using **Ollama (`llama3:8b`)**."
    )
    
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

elif page == "Comparison Lab":
    st.title("⚖️ Comparison Lab: Brute-Force vs HNSW")
    st.markdown(
        "Paste short text snippets below to ingest them into a Vektor collection. "
        "Then compare search results and latencies between brute-force exact search and HNSW approximate search."
    )
    
    st.markdown("### 📥 1. Ingest Documents (Demo Layer)")
    st.info(
        "**Honest Framing & Scale:** Brute-force search cost grows linearly. "
        "At a very small scale (e.g. fewer than 5,000 documents), the difference between brute-force exact search "
        "and HNSW approximate search is often imperceptible to a human, and brute-force may even be faster due to "
        "graph-traversal overhead. To see a visceral, human-perceptible speedup (e.g. 50x+), you need scale "
        "(10,000+ to 1,000,000+ vectors). When testing below, keep the total collection size in mind!"
    )
    
    text_input = st.text_area(
        "Paste text snippets (one per line):", 
        height=150,
        placeholder="Document 1 text...\nDocument 2 text...\nDocument 3 text..."
    )
    
    collection_name = st.text_input("Collection Name", value="demo_compare")
    
    if st.button("Ingest Documents", type="primary"):
        snippets = [t.strip() for t in text_input.split('\n') if t.strip()]
        if not snippets:
            st.warning("Please enter some text snippets.")
        else:
            try:
                from demo.ingestion.pipeline import ingest_documents
                from demo.ingestion.embedder import Embedder
                from vektor import Vektor, VektorConfig
                
                with st.spinner("Initializing embedder..."):
                    if "compare_embedder" not in st.session_state:
                        st.session_state["compare_embedder"] = Embedder()
                    embedder = st.session_state["compare_embedder"]
                
                with st.spinner("Connecting to Vektor..."):
                    if "compare_client" not in st.session_state:
                        st.session_state["compare_client"] = Vektor(VektorConfig(data_dir="./demo_vektor_data"))
                    client = st.session_state["compare_client"]
                    
                    if collection_name not in client.list_collections():
                        collection = client.create_collection(collection_name, dim=embedder.dim, metric="cosine")
                    else:
                        collection = client.get_collection(collection_name)
                
                with st.spinner(f"Embedding and batch inserting {len(snippets)} snippets..."):
                    inserted = ingest_documents(snippets, collection, embedder)
                
                st.success(f"Successfully ingested {inserted} documents into collection '{collection_name}'!")
                st.session_state["compare_collection"] = collection_name
                
            except Exception as e:
                st.error(f"Error during ingestion: {e}")

    st.markdown("---")
    st.markdown("### 🔍 2. Query & Compare")
    
    if "compare_collection" in st.session_state:
        query_text = st.text_input("Enter a search query:", placeholder="e.g. Find me a document about python...")
        
        col_k, col_ef = st.columns(2)
        with col_k:
            k_val = st.number_input("Top K results", min_value=1, max_value=20, value=3)
        with col_ef:
            ef_val = st.number_input("ef (HNSW Search parameter)", min_value=10, max_value=200, value=50)
            
        if st.button("Search & Compare", type="primary") and query_text:
            try:
                from vektor.api.models import SearchRequest
                
                embedder = st.session_state["compare_embedder"]
                client = st.session_state["compare_client"]
                col_name = st.session_state["compare_collection"]
                
                with st.spinner("Embedding query and executing comparison search..."):
                    query_vec = embedder.embed_one(query_text)
                    from vektor.api.routes.search import search_compare
                    
                    req = SearchRequest(query=query_vec.tolist(), k=k_val, ef=ef_val, include_vectors=False)
                    compare_res = search_compare(name=col_name, req=req, client=client)
                
                # Render Metrics
                st.markdown("#### 📊 Performance Metrics")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Brute-Force Recall", "100.0%", help="Brute-Force is exact, so it always finds the perfect ground-truth nearest neighbors.")
                m2.metric("HNSW Recall@K", f"{compare_res.recall_at_k:.1%}", help="HNSW trades a tiny fraction of accuracy for massive speedups.")
                m3.metric("Brute-Force Latency", f"{compare_res.brute_force_latency_ms:.2f} ms")
                m4.metric("HNSW Latency", f"{compare_res.hnsw_latency_ms:.2f} ms")
                
                # Second row of metrics for scaling insight
                s1, s2, s3, s4 = st.columns(4)
                total_docs = client.get_collection(col_name).count()
                s1.metric("Speedup", f"{compare_res.speedup_ratio:.1f}x", help="How many times faster HNSW is than Brute-Force.")
                s2.metric("Brute-Force Distances", f"{total_docs:,}", help="Brute-Force must calculate the distance to EVERY single vector in the database.")
                s3.metric("HNSW Distances", f"~ {ef_val * 2}", help="HNSW only evaluates a tiny fraction of the database by navigating the graph.")
                s4.metric("Total Collection Size", f"{total_docs:,} docs")
                
                st.markdown("#### 📄 Search Results")
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    st.markdown("##### Brute-Force (Exact / Ground Truth)")
                    for r in compare_res.brute_force_results:
                        with st.container(border=True):
                            st.markdown(f"**ID:** `{r.id}`")
                            st.markdown(f"**Score:** `{r.score:.4f}` | **Rank:** `#{r.rank}`")
                            st.info(f"{r.metadata.get('text', 'No text found in metadata')}")
                            
                with res_col2:
                    st.markdown("##### HNSW (Approximate)")
                    if not compare_res.hnsw_results:
                        st.warning("No results returned by HNSW.")
                    for r in compare_res.hnsw_results:
                        with st.container(border=True):
                            st.markdown(f"**ID:** `{r.id}`")
                            st.markdown(f"**Score:** `{r.score:.4f}` | **Rank:** `#{r.rank}`")
                            st.info(f"{r.metadata.get('text', 'No text found in metadata')}")
                            
            except Exception as e:
                st.error(f"Error during comparison search: {e}")
    else:
        st.info("Ingest some documents above to unlock the Query & Compare tool.")
