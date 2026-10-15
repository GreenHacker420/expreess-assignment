# AGENTS.md — Express Analytics RAG Assistant Guidelines

This document provides machine-targeted guidelines, architectural invariants, and execution constraints for AI coding agents working on the **Express Analytics AI/ML Engineer Intern Take-Home Assignment: RAG-Based Technical Documentation Assistant**.

---

## 1. Project Overview & Tech Stack
- **Project Goal:** Build a self-corrective Retrieval-Augmented Generation (RAG) assistant for technical documentation using LangGraph and FastAPI.
- **Runtime:** Python 3.12+ (managed with `uv` or virtualenv `python3.12 -m venv .venv`).
- **Orchestration Runtime:** LangGraph (`StateGraph`, `START`, `END`).
- **LLM Harness:** LangChain (`langchain`, `langchain-core`, model provider packages such as `langchain-openai`, `langchain-google-genai`, `langchain-groq`).
- **Vector Store:** ChromaDB (`chromadb`, `langchain-chroma`) with persistent on-disk storage.
- **Web API:** FastAPI + Uvicorn with Pydantic v2 schemas.
- **Tracing & Observability:** LangSmith (`LANGSMITH_TRACING=true`).

---

## 2. Setup & Environment Commands
All commands can be executed using `uv` (recommended) or standard `pip`/virtualenv from the repository root:

### Using `uv` (Fast & Recommended)
```bash
# 1. Sync dependencies and create .venv automatically (Python 3.12)
uv sync

# 2. Copy environment configuration
cp .env.example .env

# 3. Ingest initial technical document corpus
uv run python -m backend.ingestion.pipeline --data-dir backend/data/corpus

# 4. Run FastAPI backend server (port 8000)
uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# 5. Execute automated test suite
uv run pytest backend/tests/ -v
```

### Traditional Virtualenv Alternative
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

---

## 3. Architecture & LangGraph State Workflow
The workflow must be implemented as a deterministic `StateGraph` with explicit state schemas and conditional routing.

### 3.1 Graph Nodes
1. **Node 1: `query_analysis`**
   - Inputs: `question`, `retry_count`.
   - Action: Analyzes intent, categorizes query type (`conceptual`, `how-to`, `troubleshooting`, `api_reference`), and rewrites/expands search queries for optimized semantic matching.
   - Output: `transformed_query`, `query_type`.

2. **Node 2: `retrieve`**
   - Inputs: `transformed_query`, vector store client.
   - Action: Executes top-$k$ similarity search against ChromaDB. Attaches metadata (source URI/filename, section, chunk ID).
   - Output: `documents`.

3. **Node 3: `grade_documents`**
   - Inputs: `question`, `documents`.
   - Action: Evaluates document relevance using structured LLM output (Binary: `yes` / `no`). Filters out irrelevant chunks.
   - Output: `relevant_documents`, `has_relevant_docs: bool`.

4. **Node 4: `generate`**
   - Inputs: `question`, `relevant_documents`.
   - Action: Synthesizes a factual, grounded response citing retrieved source chunks explicitly (e.g. `[Doc 1: fastapi_intro.md]`).
   - Output: `generation`, `citations`.

5. **Fallback / Self-Correction Node: `rewrite_query`**
   - Triggered when `has_relevant_docs == False` and `retry_count < MAX_RETRIES`.
   - Action: Increments `retry_count`, refines search phrasing, and loops back to `retrieve`.
   - If `retry_count >= MAX_RETRIES`: Routes to fallback generation explaining insufficient documentation without hallucinating.

6. **Optional Bonus Nodes:**
   - `check_hallucination` (Self-RAG): Verifies answer groundedness against context before returning.
   - `web_search_fallback`: Queries Tavily search if vector retrieval fails after retry limits.

### 3.2 State Schema (`GraphState`)
The state must be strictly typed using `TypedDict` or Pydantic `BaseModel`:
```python
from typing import Annotated, List, Optional
from typing_extensions import TypedDict
from langchain_core.documents import Document

class GraphState(TypedDict):
    question: str
    transformed_query: Optional[str]
    query_type: Optional[str]
    documents: List[Document]
    relevant_documents: List[Document]
    generation: Optional[str]
    citations: List[str]
    retry_count: int
    web_fallback_used: bool
```

---

## 4. LangChain & LangGraph Rules (from docs-langchain MCP)
- **Use Official Orchestration Primitives:**
  - Import from `langgraph.graph import StateGraph, START, END`.
  - Always define explicit conditional edge functions (`def route_after_grading(state: GraphState) -> str:`).
- **Prevent Infinite Loops:**
  - Hard cap retries (`MAX_RETRIES = 2`).
  - Terminal condition must always reach `END`.
- **Structured LLM Outputs:**
  - Use Pydantic models with `.with_structured_output()` for document grading and hallucination checks to avoid unparseable free-form strings.
- **Traceability & LangSmith:**
  - Support `LANGSMITH_TRACING=true` and pass `run_name` / `tags` when compiling and invoking graphs for clean observability.
- **Document Metadata Preservation:**
  - Never drop chunk metadata (`source`, `title`, `chunk_id`, `url`) across graph node transitions.

---

## 5. FastAPI Endpoints Specification
The backend service must expose the following REST endpoints with strict Pydantic schemas:

| Method | Endpoint | Description | Request / Response Payload |
|---|---|---|---|
| `POST` | `/query` | Submit a question to the LangGraph RAG pipeline | `{ "question": str, "session_id": Optional[str] }` &rarr; `{ "answer": str, "sources": list, "citations": list, "query_type": str, "retries": int }` |
| `POST` | `/ingest` | Ingest new documents from file upload or URL | Multipart file upload or `{ "url": str, "source_type": str }` &rarr; `{ "status": "success", "indexed_chunks": int, "document_id": str }` |
| `GET` | `/documents` | List indexed documents and metadata | Returns `{ "total_documents": int, "documents": [ { "id": str, "source": str, "chunk_count": int } ] }` |
| `POST` | `/feedback` | Submit thumbs up/down user feedback | `{ "query_id": str, "rating": int, "comment": Optional[str] }` &rarr; `{ "status": "recorded" }` |
| `GET` | `/health` | Health check and vector store status | `{ "status": "healthy", "vector_store": "ready", "doc_count": int }` |

---

## 6. Document Ingestion & Chunking Strategy
- **Corpus Selection:** Curate a focused, high-quality technical corpus (e.g., LangChain / LangGraph core guides, FastAPI documentation, or Pydantic v2 guides) stored in `backend/data/corpus/`.
- **Text Splitter:** Use `RecursiveCharacterTextSplitter` configured for technical documentation:
  - `chunk_size`: 800 tokens / characters.
  - `chunk_overlap`: 150 tokens / characters.
  - `separators`: `["\n## ", "\n### ", "\n\n", "\n", "```", " ", ""]` to preserve code blocks and markdown headings intact.
- **Vector Storage:** Persist embeddings locally in `chroma_db/` with clear collection isolation (`tech_docs`).

---

## 7. Code Style & Architectural Boundaries
- **Directory Structure:**
  ```
  backend/
  ├── app/
  │   ├── api/          # FastAPI routers (/query, /ingest, /documents, /feedback)
  │   ├── core/         # Settings (pydantic-settings), logging, config
  │   ├── graph/        # LangGraph StateGraph, nodes, edges, state schema
  │   ├── ingestion/    # Document loaders, splitters, vector store indexing
  │   └── schemas/      # Request and response Pydantic schemas
  ├── data/
  │   └── corpus/       # Markdown / Text documents of the technical library
  ├── tests/            # Pytest test cases for graph nodes, ingestion, and API
  ├── main.py           # FastAPI entrypoint
  └── requirements.txt  # Pinned backend dependencies
  frontend/             # (Optional) Minimal interactive UI (Streamlit or React)
  ```
- **Boundaries:**
  - Graph logic in `backend/app/graph/` must not depend on FastAPI `Request` / `Response` objects.
  - Vector store interactions must be abstracted behind an interface in `backend/app/ingestion/vectorstore.py`.
  - Never commit `.env`, `chroma_db/`, or local credential files.

---

## 8. Definition of "Done"
Before completing any task, ensure:
1. `pytest backend/tests/` passes without errors.
2. The LangGraph graph compiles and visualizes successfully (or passes unit node validation).
3. FastAPI endpoints return proper HTTP status codes (200, 400, 422, 500) and validate payloads with Pydantic.
4. Answering unknown questions safely returns a graceful fallback instead of hallucinating.
5. All code is typed, commented with docstrings, and formatted cleanly.
