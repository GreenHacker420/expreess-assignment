# Implementation Plan: RAG-Based Technical Documentation Assistant

## Project Vision & Scope
A production-grade, self-corrective Retrieval-Augmented Generation (RAG) assistant for technical documentation built with **LangGraph**, **FastAPI**, and **Groq**, utilizing local embeddings via `sentence-transformers/all-MiniLM-L6-v2`, on-disk **ChromaDB**, and a local **SQLite** database for document registry, query tracking, and feedback.

---

## Architecture Overview

```
                      +-------------------+
                      |   POST /query     |
                      +---------+---------+
                                |
                                v
                      +-------------------+
                      |   analyze_query   |  (Query rewriting & intent classification)
                      +---------+---------+
                                |
                                v
                 +------------> +-------------------+
                 |              |     retrieve      |  (Chroma top_k=5 similarity search)
                 |              +---------+---------+
                 |                        |
                 |                        v
                 |              +-------------------+
                 |              |  grade_documents  |  (Structured LLM binary evaluation)
                 |              +---------+---------+
                 |                        |
                 |             [Conditional Routing]
                 |            /           |          \
                 |   Any relevant     None &       None &
                 |      chunks     retry < max    retry >= max
                 |        /               |            \
                 |       v                v             v
                 |  +----------+   +---------------+   +----------+
                 |  | generate |   | rewrite_query |   | fallback |
                 |  +----+-----+   +-------+-------+   +----+-----+
                 |       |                 |                |
                 |       v                 +----------------+
                 |      END                                 v
                 +----------------------------------------> END
```

---

## Core Invariants & Boundaries

1. **Deterministic Retry Semantics:**
   - `retry_count` starts at 0; `max_retries` is 2.
   - `retrieval_attempts` starts at 0 and increments exclusively in `retrieve`.
   - `retry_count` increments exclusively in `rewrite_query`.
   - Hard guarantee: At most $1 \text{ (initial)} + 2 \text{ (rewrites)} = 3 \text{ total retrieval attempts}$.
   - Fresh queries always reset state counters and previous context.

2. **Groundedness & Citation Resolution:**
   - `generate` assigns deterministic source tags `[S1]`, `[S2]`, ... in application code.
   - Structured model output requires citing exact source tags.
   - Application code resolves `[Sn]` labels back to source metadata (`document_id`, `section`, `source`).
   - Unknown labels or uncited factual claims trigger safe fallback.

3. **Untrusted Data & Prompt Safety:**
   - User inputs and retrieved documentation chunks are treated strictly as untrusted data.
   - Retrieved chunks are treated as evidence, never instructions. Prompts explicitly instruct the LLM to ignore embedded commands.

4. **Dependency Injection & Zero-Network Tests:**
   - All services (`LLMService`, `EmbeddingsService`, `VectorStoreService`, `DatabaseService`) implement interfaces.
   - Unit and integration tests run entirely offline with mock/fake implementations without requiring API keys or downloading weights.

---

## Task Checklist & Milestones

### Milestone 1: Scaffold, Config, Service Interfaces & Corpus Fetch [IN PROGRESS]
- [x] Project configuration (`pyproject.toml`, `.env.example`, `.gitignore`).
- [ ] Scaffold directories (`app/`, `app/graph/`, `app/services/`, `app/api/`, `scripts/`, `data/docs/`, `tests/`, `evaluation/`).
- [ ] Implement `app/config.py` with `pydantic-settings`.
- [ ] Define abstract protocols for services (`LLMService`, `EmbeddingsService`, `VectorStoreService`, `DatabaseService`).
- [ ] Implement `scripts/fetch_corpus.py` targeting 5 official FastAPI documentation pages (path params, query params, request body, dependencies, request files).
- [ ] Generate `data/corpus_manifest.json` with hashes, URLs, and MIT license attribution.
- [ ] Execute `fetch_corpus.py` and verify all 5 documents.

### Milestone 2: Ingestion, Embeddings, ChromaDB & SQLite Registry
- [ ] Implement `app/services/database.py` with SQLite tables (`documents`, `query_logs`, `feedback`).
- [ ] Implement `app/services/embeddings.py` wrapping `sentence-transformers/all-MiniLM-L6-v2` with token-length measurement and mock fallback.
- [ ] Implement `app/services/vector_store.py` for persistent ChromaDB interactions under `data/chroma`.
- [ ] Implement `app/services/ingestion.py`:
  - Heading-aware + token-aware chunker (~180-220 tokens, ~30-40 token overlap).
  - Explicit fenced code block preservation.
  - Scalar metadata (`chunk_id`, `document_id`, `source`, `title`, `section`, `chunk_index`, `content_hash`).
  - Stale chunk cleanup on document update and single-process file locking.
- [ ] Implement CLI `scripts/ingest.py`.
- [ ] Unit tests for chunking, idempotency, and update cleanup.

### Milestone 3: Graph State, Nodes, Structured Output & Execution Engine
- [ ] Define `app/graph/state.py` (`GraphState`, `RetrievedChunk`, `GradeResult`, `CitationRecord`).
- [ ] Implement structured prompts in `app/prompts.py`.
- [ ] Implement nodes in `app/graph/nodes.py`:
  - `analyze_query`: Intent classification & retrieval expansion.
  - `retrieve`: Vector search, clears stale grades.
  - `grade_documents`: Structured Pydantic evaluation with bounded concurrency.
  - `rewrite_query`: Iterative query transformation.
  - `generate`: Label-based citation synthesis.
  - `fallback`: Graceful insufficient-context response.
- [ ] Implement `app/graph/builder.py` compiling `StateGraph` with conditional edges.
- [ ] Create `scripts/debug_query.py` CLI runner for graph verification.

### Milestone 4: FastAPI REST API & Service Integration
- [ ] Implement `app/schemas.py` for request and response validation.
- [ ] Implement endpoints:
  - `POST /query`: Input validation, graph invocation, query logging.
  - `POST /ingest`: Multipart file upload, security validation, ingestion service call.
  - `GET /documents`: List indexed corpus with chunk counts.
  - `POST /feedback`: 201 Created feedback submission with query existence check.
  - `GET /health`: Corpus readiness without paid LLM calls.
- [ ] Implement error handling in `app/main.py` (422, 413, 415, 404, 503, 502, 504).

### Milestone 5: Test Suite, Evaluation Suite & Production README
- [ ] Implement unit tests (`tests/unit/`) with mock services.
- [ ] Implement integration tests (`tests/integration/`) testing all API endpoints and status codes.
- [ ] Create `evaluation/questions.json` with 12 labeled test questions.
- [ ] Implement `evaluation/run_eval.py` benchmarking accuracy, citations, and retries.
- [ ] Write comprehensive `README.md` documenting architecture, tradeoffs, setup, and evaluation metrics.
- [ ] Run full test suite to guarantee 100% green pass rate.
