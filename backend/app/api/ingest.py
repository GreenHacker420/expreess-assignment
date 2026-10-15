"""API router for multipart document uploads."""

import logging
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status

from backend.app.core.config import Settings, get_settings
from backend.app.schemas import DocumentIngestDetail, IngestResponse, UrlIngestRequest
from backend.app.services.ingestion import IngestionService

logger = logging.getLogger("api_ingest")
router = APIRouter(prefix="", tags=["ingest"])

ALLOWED_EXTENSIONS = {".md", ".txt"}


def get_ingestion_service(request: Request) -> IngestionService:
    """Retrieve IngestionService from app state."""
    return request.app.state.ingestion


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_200_OK,
    summary="Ingest new documents via file upload or JSON URL",
)
async def ingest_documents(
    request: Request,
    files: Optional[List[UploadFile]] = File(None, description="Multipart upload of UTF-8 .md or .txt files"),
    ingestion: IngestionService = Depends(get_ingestion_service),
    settings: Settings = Depends(get_settings),
):
    """Ingest technical documentation files or URL into vector store and SQLite registry.

    Security & Validation:
    - If Content-Type is application/json, fetches and indexes document from provided URL.
    - If multipart/form-data, accepts .md and .txt files.
    - Rejects unsupported file extensions (415).
    - Enforces maximum size limit (413).
    - Decodes as UTF-8 (422).
    - Prevents directory traversal attacks via filename sanitization.
    - Rejects empty uploads (422).
    """
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            body = await request.json()
            url_payload = UrlIngestRequest(**body)
            return await ingest_from_url(payload=url_payload, ingestion=ingestion, settings=settings)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid JSON ingestion payload: {e}",
            ) from e

    if not files:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No files provided in upload request.",
        )

    details: List[DocumentIngestDetail] = []
    total_chunks = 0

    for upload_file in files:
        raw_name = upload_file.filename or "uploaded_document.md"
        # Sanitize against path traversal attacks (e.g. ../../etc/passwd)
        if ".." in raw_name or "/" in raw_name or "\\" in raw_name:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Insecure filename detected: {raw_name}",
            )
        safe_filename = Path(raw_name).name

        suffix = Path(safe_filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported file type '{suffix}'. Only .md and .txt files are accepted.",
            )

        # Read content bytes with size limit check
        content_bytes = await upload_file.read()
        if len(content_bytes) > settings.max_upload_size_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=(
                    f"File '{safe_filename}' ({len(content_bytes)} bytes) exceeds the allowed size limit of "
                    f"{settings.max_upload_size_bytes} bytes."
                ),
            )

        if len(content_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"File '{safe_filename}' is empty.",
            )

        # Decode as UTF-8
        try:
            text_content = content_bytes.decode("utf-8")
        except UnicodeDecodeError as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"File '{safe_filename}' contains invalid UTF-8 byte sequences: {e}",
            ) from e

        title = Path(safe_filename).stem.replace("_", " ").title()
        source_uri = f"upload://{safe_filename}"

        try:
            res = ingestion.ingest_text(
                source=source_uri,
                title=title,
                raw_content=text_content,
            )
            details.append(
                DocumentIngestDetail(
                    document_id=res.document_id,
                    source=res.source,
                    title=res.title,
                    chunk_count=res.chunk_count,
                    status=res.status,
                )
            )
            total_chunks += res.chunk_count
        except Exception as e:
            logger.exception("Failed to ingest uploaded file '%s': %s", safe_filename, e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to index document '{safe_filename}': {e}",
            ) from e

    return IngestResponse(
        status="success",
        indexed_documents=len(details),
        total_chunks=total_chunks,
        indexed_chunks=total_chunks,
        document_id=details[0].document_id if details else None,
        details=details,
    )


@router.post(
    "/ingest/url",
    response_model=IngestResponse,
    status_code=status.HTTP_200_OK,
    summary="Ingest a technical document from an external URL",
)
async def ingest_from_url(
    payload: UrlIngestRequest,
    ingestion: IngestionService = Depends(get_ingestion_service),
    settings: Settings = Depends(get_settings),
):
    """Fetch and ingest a Markdown or text technical document from an external URL."""
    import httpx

    target_url = payload.url.strip()
    if not (target_url.startswith("http://") or target_url.startswith("https://")):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="URL must start with http:// or https://",
        )

    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            resp = await client.get(target_url)
            if resp.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Failed to fetch content from URL. Upstream server returned HTTP {resp.status_code}.",
                )
            content_bytes = resp.content
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error connecting to URL: {exc}",
        ) from exc

    if len(content_bytes) > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"URL content ({len(content_bytes)} bytes) exceeds the allowed size limit.",
        )

    if len(content_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="URL returned empty content.",
        )

    try:
        text_content = content_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="URL content is not valid UTF-8 text.",
        ) from exc

    doc_title = payload.title or Path(target_url.split("?")[0]).stem.replace("-", " ").replace("_", " ").title()

    try:
        res = ingestion.ingest_text(
            source=target_url,
            title=doc_title,
            raw_content=text_content,
        )
        return IngestResponse(
            status="success",
            indexed_documents=1,
            total_chunks=res.chunk_count,
            indexed_chunks=res.chunk_count,
            document_id=res.document_id,
            details=[
                DocumentIngestDetail(
                    document_id=res.document_id,
                    source=res.source,
                    title=res.title,
                    chunk_count=res.chunk_count,
                    status=res.status,
                )
            ],
        )
    except Exception as exc:
        logger.exception("Failed to ingest URL document: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to index document: {exc}",
        ) from exc

