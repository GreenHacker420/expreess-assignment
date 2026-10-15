"""Typed state definitions for the LangGraph self-corrective RAG workflow."""

import uuid
from typing import Any, Dict, List, Literal, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """Normalized retrieved chunk data."""
    chunk_id: str
    text: str
    source: str
    title: str
    section: str
    chunk_index: int
    distance: float = 0.0


class GradeRecord(BaseModel):
    """Document grading outcome."""
    chunk_id: str
    relevant: bool
    reason: str


class CitationRecord(BaseModel):
    """Resolved citation reference."""
    label: str  # e.g., "S1"
    chunk_id: str
    title: str
    section: str
    source: str


class GraphState(TypedDict):
    """Strictly typed state flowing between graph nodes.

    Important Invariants:
        - Lists (retrieved_chunks, relevant_chunks, grades) are REPLACED on each
          retrieval attempt. No append reducers are used.
        - retry_count increments ONLY in rewrite_query.
        - retrieval_attempts increments ONLY in retrieve.
        - status is strictly 'answered' or 'insufficient_context'.
    """
    query_id: str
    original_question: str
    retrieval_query: str
    retrieval_queries: List[str]
    query_type: str
    query_history: List[str]
    retrieved_chunks: List[Dict[str, Any]]
    retrieval_history: List[Dict[str, Any]]
    relevant_chunks: List[Dict[str, Any]]
    grades: List[Dict[str, Any]]
    retry_count: int
    generation_attempts: int
    grounding_feedback: str
    grounding_result: Dict[str, Any]
    labeled_context: str
    generation_needs_retrieval: bool
    web_fallback_used: bool
    web_search_attempted: bool
    max_retries: int
    retrieval_attempts: int
    answer: str
    citations: List[Dict[str, Any]]
    status: Literal["answered", "insufficient_context"]


def create_initial_state(
    question: str,
    query_id: Optional[str] = None,
    max_retries: int = 2,
    web_fallback_used: bool = False,
) -> GraphState:
    """Initialize a clean, fresh GraphState for a new user request."""
    qid = query_id or str(uuid.uuid4())
    cleaned_question = question.strip()

    return GraphState(
        query_id=qid,
        original_question=cleaned_question,
        retrieval_query=cleaned_question,
        retrieval_queries=[cleaned_question],
        query_type="conceptual",
        query_history=[],
        retrieved_chunks=[],
        retrieval_history=[],
        relevant_chunks=[],
        grades=[],
        retry_count=0,
        generation_attempts=0,
        grounding_feedback="",
        grounding_result={},
        labeled_context="",
        generation_needs_retrieval=False,
        web_fallback_used=web_fallback_used,
        web_search_attempted=False,
        max_retries=max_retries,
        retrieval_attempts=0,
        answer="",
        citations=[],
        status="insufficient_context",
    )
