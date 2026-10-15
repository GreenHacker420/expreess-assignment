"""API router for query submission."""

import logging
from fastapi import APIRouter, Depends, HTTPException, Request, status

from backend.app.graph.state import create_initial_state
from backend.app.schemas import CitationSchema, QueryRequest, QueryResponse
from backend.app.services.database import DatabaseService
from backend.app.services.llm import LLMTimeoutError, LLMUpstreamError

logger = logging.getLogger("api_query")
router = APIRouter(prefix="", tags=["query"])


def get_graph(request: Request):
    """Retrieve compiled LangGraph from app state."""
    return request.app.state.graph


def get_database(request: Request) -> DatabaseService:
    """Retrieve database service from app state."""
    return request.app.state.database


@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit a natural language question to the documentation assistant",
)
async def query_documentation(
    payload: QueryRequest,
    graph=Depends(get_graph),
    database: DatabaseService = Depends(get_database),
):
    """Submit a question to the self-corrective LangGraph RAG assistant.

    Flow:
    - Verifies document corpus is not empty (503 if empty).
    - Initializes fresh GraphState with zeroed retries.
    - Executes StateGraph (query analysis -> retrieve -> grade -> [generate | rewrite | fallback]).
    - Persists query record in SQLite database for feedback tracking.
    """
    question = payload.question.strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Question cannot be empty or purely whitespace.",
        )

    # 503 if corpus has no documents indexed
    doc_count = database.count_documents()
    if doc_count == 0:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The technical documentation corpus is empty. Ingest documents before querying.",
        )

    initial_state = create_initial_state(question)
    query_id = initial_state["query_id"]

    try:
        final_state = await graph.ainvoke(initial_state)
    except LLMTimeoutError as e:
        logger.error("[query_id=%s] Upstream LLM timeout: %s", query_id, e)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="The upstream model provider timed out while processing your request.",
        ) from e
    except LLMUpstreamError as e:
        logger.error("[query_id=%s] Upstream LLM error: %s", query_id, e)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"The upstream model provider encountered an error: {e}",
        ) from e
    except Exception as e:
        logger.exception("[query_id=%s] Unexpected error during graph execution: %s", query_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal server error occurred while executing the assistant workflow.",
        ) from e

    # Persist query record in SQLite for audit and feedback
    database.record_query(
        query_id=query_id,
        question=question,
        answer=final_state["answer"],
        status=final_state["status"],
        retry_count=final_state["retry_count"],
        retrieval_attempts=final_state["retrieval_attempts"],
        citations=final_state["citations"],
    )

    citations_out = [
        CitationSchema(
            label=c.get("label", ""),
            chunk_id=c.get("chunk_id", ""),
            title=c.get("title", ""),
            section=c.get("section", ""),
            source=c.get("source", ""),
        )
        for c in final_state.get("citations", [])
    ]

    return QueryResponse(
        query_id=query_id,
        answer=final_state["answer"],
        status=final_state["status"],
        citations=citations_out,
        retry_count=final_state["retry_count"],
        retrieval_attempts=final_state["retrieval_attempts"],
        query_type=final_state["query_type"],
        grounding_verified=final_state.get("grounding_result", {}).get("supported") is True,
        web_fallback_used=final_state.get("web_fallback_used", False),
    )
