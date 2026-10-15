# RAG-Based Technical Documentation Assistant

[![Python 3.12](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.70+-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.6+-purple.svg)](https://www.trychroma.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.42+-FF4B4B.svg)](https://streamlit.io)
[![Tests Passing](https://img.shields.io/badge/tests-21%20passed-brightgreen.svg)]()

> **Express Analytics AI/ML Engineer Intern Take-Home Assignment**  
> A production-grade, self-corrective Retrieval-Augmented Generation (RAG) assistant for technical documentation, orchestrated with **LangGraph** cyclical state machines, indexed with **ChromaDB**, and served via **FastAPI** with a **Streamlit** interactive dashboard.

---

## 📑 Table of Contents
1. [System Architecture](#-system-architecture)
2. [LangGraph Engineering Best Practices](#-langgraph-engineering-best-practices)
3. [Corpus & Chunking Strategy](#-corpus--chunking-strategy)
4. [Project Structure](#-project-structure)
5. [Getting Started & Installation](#-getting-started--installation)
6. [API Specification](#-api-specification)
7. [Streamlit Interactive Dashboard](#-streamlit-interactive-dashboard)
8. [Evaluation Benchmark & Results](#-evaluation-benchmark--results)
9. [Automated Test Suite](#-automated-test-suite)
10. [Design Trade-offs & Production Hardening](#-design-trade-offs--production-hardening)

---

## 🏛 System Architecture

The assistant executes a deterministic, self-correcting state graph implemented with official **LangGraph** (`StateGraph`, `START`, `END`) primitives. Instead of blindly answering from naive retrieval, the system evaluates chunk relevance, rewrites unhelpful queries, guards against infinite loops, and gracefully refuses out-of-corpus or adversarial requests.

```mermaid
flowchart TD
    START([START]) --> analyze_query[Node: analyze_query\nIntent classification & query rewriting]
    analyze_query --> retrieve[Node: retrieve\nMulti-query vector search + reciprocal-rank merge]
    retrieve --> grade_documents[Node: grade_documents\nStructured LLM relevance scoring]
    
    grade_documents --> route_decision{route_after_grading}
    
    route_decision -- "Relevant Chunks Found" --> generate[Node: generate\nSynthesize answer with citations [S1] [S2]]
    route_decision -- "No Relevant Chunks & retries < MAX" --> rewrite_query[Node: rewrite_query\nRefine search phrasing & retry_count++]
    route_decision -- "Exhausted & web not tried" --> web_search[Optional Tavily web search]
    web_search --> grade_documents
    route_decision -- "Exhausted & web tried/no service" --> fallback[Node: fallback\nSafe refusal, zero hallucination]
    
    rewrite_query --> retrieve
    generate --> generation_route{Evidence sufficient?}
    generation_route -- "No; retries remain" --> rewrite_query
    generation_route -- "Yes" --> grounding[Node: check_grounding\nSelf-RAG evidence verification]
    grounding -- "Unsupported; first failure" --> generate
    grounding -- "Supported" --> END([END])
    grounding -- "Still unsupported" --> fallback
    fallback --> END([END])
```

### Graph Execution Nodes
| Node | Responsibility | Output Invariants |
|---|---|---|
| **`analyze_query`** | Categorizes intent and creates a primary query plus focused searches for distinct aspects of compound questions. | `query_type`, `retrieval_queries` |
| **`retrieve`** | Searches each query, merges results with reciprocal-rank fusion, and preserves chunk metadata. | Deduplicated `retrieved_chunks` |
| **`grade_documents`** | Evaluates document relevance using structured binary classification (`yes`/`no`) via LLM. Filters out noise. | `relevant_chunks` |
| **`generate`** | Synthesizes a factual, grounded response citing retrieved source chunks explicitly using bracket labels (`[S1]`, `[S2]`). | `answer`, `citations`, `status="answered"` |
| **`check_grounding`** | Uses a separate structured LLM judgment to check the answer claims against the cited context; permits one constrained regeneration. | `grounding_result`, bounded regeneration or safe fallback |
| **`rewrite_query`** | Triggers when zero chunks pass relevance grading. Expands phrasing and loops back to `retrieve`. | `retry_count += 1`, `query_history` |
| **`web_search`** | Optional single Tavily search after indexed-corpus retries; returned snippets are graded by the same relevance node and cited by URL. | `web_search_attempted`, `web_fallback_used` |
| **CRAG correction** | Rewrites and retrieves again when grading rejects all chunks or generation identifies missing evidence, subject to `MAX_RETRIES`. | Bounded retrieval attempts |
| **`fallback`** | Reached when retrieval retries are exhausted or answer grounding still fails. Returns safe refusal. | `answer`, `status="insufficient_context"` |

---

## 📐 LangGraph Engineering Best Practices

This implementation adheres to production LangGraph architectural patterns:

### 1. Explicit Typed States (`GraphState`)
All graph state is governed by a strictly typed `TypedDict` schema in [`backend/app/graph/state.py`](backend/app/graph/state.py), ensuring explicit contracts across nodes:
```python
class GraphState(TypedDict):
    query_id: str
    original_query: str
    transformed_query: str
    query_type: str
    retrieval_attempts: int
    retry_count: int
    max_retries: int
    retrieved_chunks: List[Dict[str, Any]]
    relevant_chunks: List[Dict[str, Any]]
    answer: str
    citations: List[Dict[str, Any]]
    status: str
    query_history: List[str]
```

### 2. Model Cascading (Cost & Latency Optimization)
- **Fast Model (`GEMINI_FAST_MODEL`):** Used for query analysis, document grading, and query rewriting.
- **Answer Model (`GEMINI_MODEL`):** Used for grounded answer synthesis and structured output.
- Set `LLM_PROVIDER=groq` to use the optional Groq implementation and its `FAST_LLM_MODEL` / `LLM_MODEL` settings. Gemini is the default and reads `GEMINI_API_KEY` (or `GOOGLE_API_KEY`).

### 3. Bounded Loops (Cycle & Token Exhaustion Protection)
Cyclic workflows can lead to runaway API bills if retrieval wanders. Retrieval correction is bounded by `MAX_RETRIES = 2` (at most 3 retrieval attempts). Optional web search is disabled by default and runs at most once after corpus retries; it requires both `WEB_SEARCH_ENABLED=true` and `TAVILY_API_KEY`. Web results pass relevance grading and answer-grounding checks, with citations pointing to their source URLs. Self-RAG permits one regeneration after an unsupported draft; a second failed check routes to `fallback -> END`.

### 4. Persistence & Checkpointing
`build_rag_graph` supports optional LangGraph checkpointers (such as `MemorySaver` or database checkpointers). This facilitates:
- State inspection and time-travel replay for failed production queries.
- Thread isolation for multi-turn sessions.

### 5. Deep Observability & Resilient Backoff
- Integrated with **LangSmith** via standard environment flags (`LANGSMITH_TRACING=true`).
- The selected provider service retries rate-limit responses with bounded exponential backoff.

---

## Corpus & Chunking Strategy

### Curated Technical Corpus
The assistant is indexed with verified official **FastAPI** technical documentation (`backend/data/corpus/`):
- `path_params.md`: Path parameters, type annotations, Enum choices, path convertors.
- `query_params.md`: Query parameters, defaults, optionality, type validation.
- `request_body.md`: Pydantic `BaseModel`, request body + path + query parameters.
- `dependencies.md`: Dependency injection, `Depends()`, sub-dependencies, yields.
- `request_files.md`: File uploads, `File`, `UploadFile`, spooled memory vs disk storage.

A tamper-proof manifest with SHA-256 integrity checksums is tracked in [`backend/data/corpus_manifest.json`](backend/data/corpus_manifest.json).

### Heading-Aware, Code-Block-Preserving Chunking
Document chunking in [`backend/app/services/ingestion.py`](backend/app/services/ingestion.py) employs a specialized Markdown splitter:
- **Heading Hierarchy:** Parses `#`, `##`, `###` headings to preserve section lineage.
- **Code Block Integrity:** Never splits fenced code blocks (```` ```python ... ``` ````) across boundaries.
- **Deduplication:** Computes SHA-256 hashes of normalized chunk text to guarantee idempotent re-indexing.
- **Parameters:** Target size 800 characters with 150 character overlap.

---

## 🗂 Project Structure

```
expreess-assignment/
├── AGENTS.md                  # Machine-targeted guidelines & architectural invariants
├── pyproject.toml             # uv / pip dependency definitions
├── uv.lock                    # Deterministic locked dependency graph
├── app/
│   └── main.py                # Root FastAPI application entrypoint re-export
├── frontend/
│   └── app.py                 # Interactive Streamlit dashboard
├── backend/
│   ├── main.py                # Backend Uvicorn runner
│   ├── app/
│   │   ├── main.py            # FastAPI factory & lifespan management
│   │   ├── api/               # REST routers (/query, /ingest, /documents, /feedback, /health)
│   │   ├── core/              # Configuration (Pydantic Settings) & LLM prompts
│   │   ├── graph/             # LangGraph state machine, nodes, conditional edges
│   │   ├── schemas/           # Pydantic v2 request/response schemas
│   │   └── services/          # Abstracted LLM, vector store, embedding, and DB services
│   ├── data/
│   │   ├── corpus/            # Curated FastAPI markdown documentation
│   │   └── corpus_manifest.json
│   ├── evaluation/
│   │   ├── questions.json     # 12 labeled benchmark questions
│   │   ├── run_eval.py        # Automated evaluation benchmark harness
│   │   ├── golden_cases.json  # 14 source- and answer-labeled cases
│   │   ├── golden_cases.xlsx  # Human-reviewable benchmark workbook
│   │   ├── run_golden_eval.py # Live graph + retrieval evaluation
│   │   └── run_api_eval.py    # FastAPI contract + golden API evaluation
│   │   └── evaluation_results.json
│   ├── scripts/
│   │   ├── fetch_corpus.py    # Official docs scraper & manifest generator
│   │   ├── ingest.py          # CLI indexing tool
│   │   └── debug_query.py     # Standalone LangGraph pipeline CLI debugger
│   └── tests/
│       ├── conftest.py        # Offline fixtures & mock services
│       ├── unit/              # Graph transitions, ingestion, and chunking tests
│       └── integration/       # Full FastAPI HTTP endpoint tests
```

---

## 🚀 Getting Started & Installation

### Prerequisites
- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) (recommended for fast package installation) or standard Python `venv`.

### 1. Clone & Set Up Environment
```bash
# Clone the repository
git clone https://github.com/your-org/expreess-assignment.git
cd expreess-assignment

# Create virtual environment and install all dependencies
uv sync

# Configure environment variables
cp .env.example .env
```

### 2. Configure API Keys
Edit `.env` to include your credentials:
```env
# Gemini Developer API
LLM_PROVIDER="gemini"
GEMINI_API_KEY="your_gemini_api_key_here"
GEMINI_MODEL="gemini-3.8-flash"
GEMINI_FAST_MODEL="gemini-3.8-flash"

# Optional Groq: set LLM_PROVIDER="groq" and configure GROQ_API_KEY / model names

# Optional Tavily public-web fallback (disabled by default)
WEB_SEARCH_ENABLED=false
TAVILY_API_KEY=""
WEB_SEARCH_MAX_RESULTS=5

# Optional: LangSmith Tracing
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=""
```

For this checkout, `backend/.env` is also loaded. An exported `GEMINI_API_KEY` or `GOOGLE_API_KEY` in your shell takes precedence over both files. If an old key was exported, clear it when starting the local app so the `backend/.env` key is used:

```bash
env -u GEMINI_API_KEY -u GOOGLE_API_KEY uv run uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

### 3. Ingest Documentation Corpus
Index the approved FastAPI corpus into ChromaDB and SQLite:
```bash
uv run python backend/scripts/ingest.py
```
*Expected output: 5 documents, 101 chunks indexed.*

### 4. Start FastAPI Backend Server
```bash
env -u GEMINI_API_KEY -u GOOGLE_API_KEY uv run uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
FastAPI Swagger documentation will be available at [http://localhost:8000/docs](http://localhost:8000/docs).

### 5. Launch Streamlit UI
In a separate terminal window:
```bash
uv run streamlit run frontend/app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📡 API Specification

| Method | Endpoint | Description | Sample Request / Response |
|---|---|---|---|
| `POST` | `/query` | Submit a question to the self-correcting RAG pipeline | **Request:** `{"question": "How do I declare path parameters?"}`<br/>**Response:** `{"query_id": "...", "status": "answered", "answer": "...", "citations": [...], "query_type": "how-to", "retry_count": 0, "grounding_verified": true}` |
| `POST` | `/ingest` | Upload `.md` or `.txt` file to index | **Request:** `multipart/form-data` with `file`<br/>**Response:** `{"document_id": "...", "indexed_chunks": 18, "status": "indexed"}` |
| `GET` | `/documents` | List all indexed documents and chunk counts | **Response:** `{"total_documents": 5, "documents": [{"document_id": "...", "title": "...", "chunk_count": 22}]}` |
| `POST` | `/feedback` | Submit thumbs up/down user feedback | **Request:** `{"query_id": "...", "rating": 1, "comment": "Accurate code sample"}`<br/>**Response:** `{"status": "recorded"}` |
| `GET` | `/health` | Health check and vector store status | **Response:** `{"status": "healthy", "vector_store": "ready", "document_count": 5, "vector_store_chunks": 101}` |

---

## 🖥 Streamlit Interactive Dashboard

The Streamlit frontend (`frontend/app.py`) provides three purpose-built tabs:
1. **💬 Assistant Q&A:**
   - Interactive prompt with quick preset queries.
   - Grounded synthesized answers formatted in Markdown.
   - Collapsible citation cards linking section and source filename.
   - Transparent pipeline telemetry (status badge, latency, query rewrites history, retry counts).
   - Instant 👍 / 👎 feedback submission.
2. **📂 Corpus Explorer:**
   - Real-time catalog of indexed documentation, chunk counts, and indexing timestamps.
3. **📥 Ingest Documents:**
   - Upload new Markdown technical guides with immediate vector indexing.

---

## 📊 Evaluation Benchmark & Results

The repository includes a **14-case golden set** in [`backend/evaluation/golden_cases.json`](backend/evaluation/golden_cases.json) and a reviewable workbook in [`backend/evaluation/golden_cases.xlsx`](backend/evaluation/golden_cases.xlsx). Cases cover direct questions, paraphrases, multi-document synthesis, adjacent and unrelated unsupported topics, and prompt injection. Supported cases define expected corpus sources and answer fact groups; unsupported cases expect a safe refusal.

Run a fast semantic retrieval check locally, without using LLM requests, with:

```bash
uv run python backend/evaluation/run_golden_eval.py --retrieval-only
```

The live graph evaluator reports status accuracy, top-k source recall, cited-source recall, heuristic fact coverage, citation-label integrity, retries, and latency:

```bash
# Requires the selected provider's API key and a populated Chroma index
env -u GEMINI_API_KEY -u GOOGLE_API_KEY uv run python backend/evaluation/run_golden_eval.py
```

You can target the multi-document synthesis cases (`G08` and `G09`) while tuning retrieval or grading with `--cases path/to/cases.json`; the evaluator records the union of retrieved sources across correction attempts so a terminal fallback does not erase retrieval diagnostics.

It saves timestamped JSON and CSV run results under `backend/evaluation/runs/` (ignored by git). Fact coverage is a transparent phrase-match regression signal, not a substitute for human review or an LLM-as-judge quality study.

To test the running FastAPI application end to end, start the service in one terminal and run the API evaluator in another:

```bash
env -u GEMINI_API_KEY -u GOOGLE_API_KEY uv run uvicorn backend.main:app --host 127.0.0.1 --port 8000
uv run python backend/evaluation/run_api_eval.py --base-url http://127.0.0.1:8000
```

The API evaluator checks `/health`, `/documents`, each golden query’s HTTP response and citations, query validation, invalid feedback, and a successful feedback round trip. It writes separate timestamped JSON/CSV results. For quick manual checks:

```bash
curl -sS http://127.0.0.1:8000/health
curl -sS http://127.0.0.1:8000/documents
curl -sS -X POST http://127.0.0.1:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"How do I define a JSON request body with a Pydantic BaseModel?"}'
curl -i -X POST http://127.0.0.1:8000/query \
  -H 'Content-Type: application/json' -d '{"question":"   "}'
```

The earlier 12-case harness ([`backend/evaluation/run_eval.py`](backend/evaluation/run_eval.py)) remains an **offline mock routing smoke test**. Its historical 12/12 result measures expected status and retry routing with canned model responses; it does not establish answer accuracy. Use the live golden evaluator for source and answer regression signals.

1. **Direct Supported:** Clear factual questions on basic syntax.
2. **Paraphrased:** Conceptual questions with rephrased terminology.
3. **Partially Supported / Multi-part:** Multi-concept questions requiring cross-document synthesis.
4. **Out of Corpus:** Unrelated technical topics (e.g. Celery Redis queues, OAuth2 JWT). Must trigger safe refusal without hallucinating.
5. **Prompt Injection / Adversarial:** System override attempts. Must maintain safety boundaries and trigger safe refusal.

### Benchmark Results
```
==========================================================================================
ID         | CATEGORY             | STATUS       | MATCH  | ATTEMPTS | LATENCY 
------------------------------------------------------------------------------------------
eval-01    | direct_supported     | answered     | PASS   | 1        | 0.444s
eval-02    | direct_supported     | answered     | PASS   | 1        | 0.011s
eval-03    | direct_supported     | answered     | PASS   | 1        | 0.010s
eval-04    | direct_supported     | answered     | PASS   | 1        | 0.010s
eval-05    | direct_supported     | answered     | PASS   | 1        | 0.011s
eval-06    | paraphrased          | answered     | PASS   | 1        | 0.011s
eval-07    | paraphrased          | answered     | PASS   | 1        | 0.010s
eval-08    | partially_supported  | answered     | PASS   | 1        | 0.014s
eval-09    | partially_supported  | answered     | PASS   | 1        | 0.010s
eval-10    | out_of_corpus        | insufficient_context | PASS   | 3        | 0.033s
eval-11    | out_of_corpus        | insufficient_context | PASS   | 3        | 0.032s
eval-12    | prompt_injection     | insufficient_context | PASS   | 3        | 0.357s
------------------------------------------------------------------------------------------
Total Questions: 12 | Accuracy: 12/12 (100.0%) | Avg Latency: 0.079s
==========================================================================================
```

To run the evaluation:
```bash
# Run evaluation benchmark
uv run python backend/evaluation/run_eval.py

# Run live evaluation with the provider selected in .env (Gemini by default)
uv run python backend/evaluation/run_eval.py --live
```

---

## 🧪 Automated Test Suite

The automated test suite currently contains 29 tests under [`backend/tests/`](backend/tests/); run it with `uv run pytest backend/tests/ -v`.
- **Unit Tests (`test_graph.py`):** Verified document routing, relevance filtering, query rewriting loops, retry exhaustion, and citation integrity.
- **Ingestion Tests (`test_ingestion.py`):** Heading-aware chunking, code-block boundaries, SHA-256 deduplication, and stale chunk deletion.
- **Integration Tests (`test_api.py`):** Full FastAPI test client covering `/query`, `/ingest`, `/documents`, `/feedback`, `/health`, and error codes (400, 404, 415, 422, 503).
- **100% Offline by Design:** Tests run completely offline in under **0.25 seconds** using dependency injection and mock LLM fixtures without requiring API keys or network access.

```bash
uv run pytest backend/tests/ -v
```

---

## ⚖️ Design Trade-offs & Production Hardening

| Architectural Decision | Chosen Approach | Alternative Considered | Rationale / Trade-off |
|---|---|---|---|
| **Orchestration** | Cyclical LangGraph (`StateGraph`) | Linear LangChain LCEL chain | LangGraph natively models cyclical self-correction, retry counters, and branching logic with explicit typed state. |
| **Vector Database** | ChromaDB with local persistent storage | Pinecone / Qdrant Cloud | Zero cloud dependency for local evaluation; embeds easily in local development while maintaining metadata filtering. |
| **Embeddings** | Local `sentence-transformers/all-MiniLM-L6-v2` | OpenAI `text-embedding-3-small` | Fast, deterministic, free, and runs offline without API latency or rate limit quotas. |
| **Chunking** | Heading-aware AST Markdown parser | Fixed-size character window | Prevents breaking code snippets mid-block and attaches meaningful section headers to vector metadata. |
| **Hallucination Prevention** | Fallback disclaimer on retry exhaustion | Unconstrained generation | Technical users prefer a clear "I do not have documentation on this topic" over fabricated APIs or incorrect syntax. |
| **Import Discipline** | All imports strictly at module top-level | Inline / deferred imports | Prevents runtime `ImportError` inside node loops, simplifies static analysis and dependency tracing. |

---

## 👤 Author
Developed for the **Express Analytics AI/ML Engineer Intern Take-Home Assignment**.
