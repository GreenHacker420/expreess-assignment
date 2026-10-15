"""API router for user feedback submission."""

import logging
from fastapi import APIRouter, Depends, HTTPException, Request, status

from backend.app.schemas import FeedbackRequest, FeedbackResponse
from backend.app.services.database import DatabaseService

logger = logging.getLogger("api_feedback")
router = APIRouter(prefix="", tags=["feedback"])


def get_database(request: Request) -> DatabaseService:
    """Retrieve DatabaseService from app state."""
    return request.app.state.database


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit thumbs up/down user feedback for an answer",
)
async def submit_feedback(
    payload: FeedbackRequest,
    database: DatabaseService = Depends(get_database),
):
    """Record user feedback for a specific query response.

    Status Codes:
    - 201: Feedback recorded successfully.
    - 404: The specified query_id does not exist.
    - 422: Validation error (invalid rating value or missing query_id).
    """
    try:
        feedback_id = database.record_feedback(
            query_id=payload.query_id,
            rating=payload.rating,
            comment=payload.comment,
        )
        logger.info(
            "Recorded feedback %s for query_id=%s (rating=%s)",
            feedback_id, payload.query_id, payload.rating
        )
        return FeedbackResponse(status="recorded", feedback_id=feedback_id)
    except KeyError as e:
        logger.warning("Feedback rejected: %s", e)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Query '{payload.query_id}' not found.",
        ) from e
