# 🚀 Vektor — Docker Deployment & Project Verification Guide

**Project:** Vektor — Production Vector Database from Scratch  
**Verification Target Date:** 13 October 2026  
**Demonstrable Branch:** `docker-deploy` (created directly from `eval2`)  
**Deployment Target:** Project Lab Systems (Phase 1) & Central AI Project Lab Server (Phase 2)  

---

## 📋 Mandatory Instructions Compliance Matrix

| # | Mandatory Instruction | How Vektor Fulfills It |
|---|---|---|
| **1** | Run successfully on allotted Project Lab system | Fully containerized with multi-stage Docker builds. Zero dependencies required on host machine except Docker. |
| **2** | Docker configuration for Frontend, Backend & Database | Full-stack `docker-compose.yml` orchestrating `vektor-api` (FastAPI backend + storage engine), `vektor-ui` (Streamlit frontend), and optional `ollama` (LLM inference). |
| **3** | Persistent storage to prevent data loss | Configured with Docker named volumes (`vektor_data`, `demo_vektor_data`, `ollama_data`). All vector indexes, binary graph files, and SQLite metadata survive container stops, restarts, and down/up cycles. |
| **4** | Code, dependencies & Docker files in Git repo | `Dockerfile.backend`, `Dockerfile.ui`, `docker-compose.yml`, `docker-compose.phase2.yml`, `requirements.txt`, and `.dockerignore` tracked in git branch `docker-deploy`. |
| **5** | Independently executable without personal laptops | Can be cloned fresh on any lab machine and run with a single command: `docker compose up --build`. |
| **6** | All major project functionalities demonstrated | Document RAG Q&A (Dual LLM: Ollama + Cloud APIs), Comparison Lab (Brute-force vs HNSW), REST API endpoints, and Enterprise-Scale ingestion (10,000+ records). |

---

## ⚡ Quick Start — Lab Verification (Phase 1)

### 1. Clone & Switch to the Demonstrable Docker Branch
```bash
git clone <repository-url>
cd Vektor
git checkout docker-deploy
```

### 2. Configure Environment (Optional Cloud API Keys)
```bash
# Copy template:
cp .env.example .env
```
*(Optional: If you wish to use cloud LLM APIs like Groq or OpenAI during the demo, add your API key in `.env` or input it directly in the Streamlit UI sidebar).*

### 3. Start the Full Stack with a Single Command

#### Mode A: Vector Database + Streamlit UI (Fastest — Zero Setup)
```bash
docker compose up --build vektor-api vektor-ui
```
- **Backend API:** `http://localhost:8000` (Swagger Docs: `http://localhost:8000/docs`)
- **Frontend UI:** `http://localhost:8501`
- **LLM Answering:** Connects to Cloud LLM APIs (Groq, OpenAI, Claude, Gemini) via UI sidebar, OR any Ollama running on host.

#### Mode B: Full Stack with Local Ollama Container (Offline / GPU Ready)
```bash
docker compose --profile llm up --build
```
To pull the Llama model on first launch:
```bash
docker compose exec ollama ollama pull llama3:8b
```

---

## 🤖 Dual LLM Compatibility (Ollama & Cloud LLM APIs)

The demonstration UI in `docker-deploy` features a **unified multi-provider LLM client** (`demo/llm.py`):

1. **Auto Mode (Default):**
   - Automatically probes local Ollama (`http://ollama:11434`).
   - If Ollama is unavailable (e.g. CPU-only lab PC with no model installed), it **transparently falls back to Cloud LLM APIs** if a key is provided.
2. **Supported Cloud LLM Providers:**
   - **Groq:** Ultra-fast cloud inference (`llama-3.3-70b-versatile`, sub-500ms responses).
   - **OpenAI:** `gpt-4o-mini`, `gpt-4o`.
   - **Anthropic:** `claude-3-5-sonnet`, `claude-3-haiku`.
   - **Google Gemini:** `gemini-1.5-flash`, `gemini-2.0-flash`.
   - **OpenAI-Compatible Endpoints:** vLLM, LMStudio, OpenRouter.
3. **Live UI Provider Switcher:**
   - Evaluators or students can switch providers directly in the Streamlit sidebar dropdown without restarting containers or editing code.

---

## 🏢 Large Company-Level Database Demo (Phase 2 Compatible)

To demonstrate Vektor on a **large company-level database** (10,000+ to 50,000+ documents):

### Ingest 10,000 Realistic Enterprise Documents:
Run the enterprise ingestion script inside the backend container or locally:

```bash
# Inside Docker container:
docker compose exec vektor-api python demo/load_enterprise_dataset.py --count 10000 --collection enterprise_kb --data-dir /data

# Or locally:
python demo/load_enterprise_dataset.py --count 10000 --collection enterprise_kb
```

### What This Demonstrates in the Comparison Lab:
1. Open `http://localhost:8501` and navigate to **Comparison Lab**.
2. Set Collection Name to `enterprise_kb`.
3. Enter any query (e.g., `PostgreSQL connection pool exhausted on k8s cluster`).
4. Click **Search & Compare**:
   - **Brute-Force Latency:** ~30ms – 100ms (evaluates distance to every vector).
   - **HNSW Approximate Latency:** ~0.8ms – 2ms (**50x – 100x human-perceptible speedup!**).
   - **HNSW Recall@K:** >98% accuracy compared to exact ground truth.

---

## 💾 Verification of Persistent Storage (Requirement #3)

To prove to the evaluators that data is never lost across container restarts:

1. **Check stored data before stopping:**
   ```bash
   curl -s http://localhost:8000/v1/collections | jq
   ```
2. **Stop and remove containers:**
   ```bash
   docker compose down
   ```
3. **Re-launch containers:**
   ```bash
   docker compose up -d
   ```
4. **Verify collections and vectors are intact:**
   ```bash
   curl -s http://localhost:8000/v1/collections | jq
   # All collections and indexed vectors remain preserved in the 'vektor_data' named volume!
   ```

---

## 🖥️ Phase 2 — Central Server Deployment

When moving to the **Central AI Project Lab Server**:

Use the dedicated production configuration:
```bash
docker compose -f docker-compose.phase2.yml up --build -d
```

### Phase 2 Architecture Features:
- **Resource Limits:** 4 CPU cores & 8 GB RAM per container to prevent memory contention.
- **Production Restart Policy:** `restart: always` for automated crash recovery.
- **NVIDIA GPU Passthrough:** Pre-configured device reservations for Ollama server.
- **Network Isolation:** Isolated bridge network `vektor_phase2_net`.
- **Health Checks:** Native HTTP probes ensuring zero-downtime container readiness.

---

## 🧪 Running Verification Tests Inside Docker

To run the complete automated test suite (all 45+ unit and integration tests) inside the clean Docker environment:

```bash
docker compose run --rm vektor-api pytest demo/tests
```

**Status:** All 45 tests passing with zero errors.
