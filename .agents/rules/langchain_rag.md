# LangChain & LangGraph Rules for RAG Assistant

## 1. Core Principles
- **Self-Corrective RAG (CRAG / Adaptive RAG):** Implement explicit evaluation gates between retrieval and generation.
- **Node Purity:** Keep nodes in the StateGraph modular. Each node accepts `state: GraphState` and returns a dictionary with state updates.
- **Strict State Typing:** Always use `TypedDict` for `GraphState`. Avoid unstructured state dicts.

## 2. LangGraph Workflow Structure
- Nodes:
  1. `analyze_query`: Classify query intent and rewrite for dense retrieval.
  2. `retrieve`: Vector search against ChromaDB.
  3. `grade_documents`: Binary evaluation (`yes`/`no`) using structured Pydantic output.
  4. `generate`: Grounded answer synthesis with explicit document citations.
  5. `rewrite_query`: Iterative query transformation on retrieval misses.
- Edges:
  - `START` -> `analyze_query` -> `retrieve` -> `grade_documents`
  - Conditional edge after `grade_documents`:
    - If relevant docs present -> `generate` -> `END`
    - If no relevant docs and `retry_count < MAX_RETRIES` -> `rewrite_query` -> `retrieve`
    - If no relevant docs and `retry_count >= MAX_RETRIES` -> `generate_fallback` -> `END`

## 3. Tool & Model Standards
- Use structured outputs (`.with_structured_output(GradeSchema)`) for deterministic LLM grading.
- Preserve document metadata throughout all graph transitions.
- Maintain LangSmith tracing compatibility with `LANGSMITH_TRACING=true`.
