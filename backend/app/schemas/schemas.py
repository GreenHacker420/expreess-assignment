"""Pydantic v2 request and response schemas for FastAPI endpoints."""

from typing import List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class CitationSchema(BaseModel):
    """Schema for individual document citations."""
    label: str = Field(description="Citation tag matching answer, e.g. 'S1'")
    chunk_id: str = Field(description="Unique identifier of the cited chunk")
    title: str = Field(description="Title of the source document")
    section: str = Field(description="Heading or section name where chunk appears")
    source: str = Field(description="URI or filename of the document")


class QueryRequest(BaseModel):
    """Schema for incoming user questions."""
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(
        min_length=1,
        max_length=1000,
        description="The natural language question to ask the documentation assistant"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session tracking identifier"
    )


class QueryResponse(BaseModel):
    """Schema for query answers."""
    query_id: str = Field(description="Unique tracking ID for this query")
    answer: str = Field(description="Grounded response synthesized from documentation")
    status: Literal["answered", "insufficient_context"] = Field(
        description="Outcome of RAG pipeline: answered or insufficient_context"
    )
    citations: List[CitationSchema] = Field(
        default_factory=list,
        description="List of verified citations supporting the answer"
    )
    retry_count: int = Field(description="Number of query rewrites executed")
    retrieval_attempts: int = Field(description="Total retrieval executions")
    query_type: str = Field(description="Classified query intent")


class DocumentIngestDetail(BaseModel):
    """Detail for a single ingested document."""
    document_id: str
    source: str
    title: str
    chunk_count: int
    status: str  # "indexed", "skipped", "updated"


class IngestResponse(BaseModel):
    """Response returned upon multipart document upload."""
    status: str = Field(description="Ingestion outcome status")
    indexed_documents: int = Field(description="Count of processed documents")
    total_chunks: int = Field(description="Total active chunks across processed documents")
    details: List[DocumentIngestDetail] = Field(default_factory=list)


class DocumentRecord(BaseModel):
    """Record representing an indexed document."""
    document_id: str
    source: str
    title: str
    chunk_count: int
    updated_at: str


class DocumentsListResponse(BaseModel):
    """Response listing indexed documents in the corpus."""
    total_documents: int
    documents: List[DocumentRecord]


class FeedbackRequest(BaseModel):
    """Request payload for recording thumbs up/down user feedback."""
    model_config = ConfigDict(str_strip_whitespace=True)

    query_id: str = Field(min_length=1, description="The query_id being evaluated")
    rating: Literal["up", "down"] = Field(description="User evaluation verdict")
    comment: Optional[str] = Field(default=None, max_length=1000, description="Optional written feedback")


class FeedbackResponse(BaseModel):
    """Response returned when feedback is recorded."""
    status: str = "recorded"
    feedback_id: str


class HealthResponse(BaseModel):
    """Health status and vector store readiness."""
    status: str = "healthy"
    vector_store: str
    document_count: int
    chunk_count: int
