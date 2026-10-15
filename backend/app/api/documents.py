"""API router for document corpus exploration."""

import logging
from fastapi import APIRouter, Depends, Request, status

from backend.app.schemas import DocumentRecord, DocumentsListResponse
from backend.app.services.database import DatabaseService

logger = logging.getLogger("api_documents")
router = APIRouter(prefix="", tags=["documents"])


def get_database(request: Request) -> DatabaseService:
    """Retrieve DatabaseService from app state."""
    return request.app.state.database


@router.get(
    "/documents",
    response_model=DocumentsListResponse,
    status_code=status.HTTP_200_OK,
    summary="List all indexed technical documentation in the corpus",
)
async def list_documents(database: DatabaseService = Depends(get_database)):
    """Return all indexed document records, their titles, chunk counts, and update timestamps."""
    raw_docs = database.list_documents()

    docs = [
        DocumentRecord(
            document_id=d["document_id"],
            source=d["source"],
            title=d["title"],
            chunk_count=d["chunk_count"],
            updated_at=d["updated_at"],
        )
        for d in raw_docs
    ]

    return DocumentsListResponse(
        total_documents=len(docs),
        documents=docs,
    )
