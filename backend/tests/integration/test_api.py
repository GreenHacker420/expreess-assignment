"""Integration tests for all FastAPI REST endpoints and status codes."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_check(async_client: AsyncClient):
    response = await async_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "vector_store" in data
    assert "document_count" in data


@pytest.mark.asyncio
async def test_query_against_empty_corpus_returns_503(async_client: AsyncClient):
    # Before indexing documents, corpus is empty
    response = await async_client.post("/query", json={"question": "What is FastAPI?"})
    assert response.status_code == 503
    assert "corpus is empty" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_query_validation_422(async_client: AsyncClient):
    # Empty question
    r1 = await async_client.post("/query", json={"question": "   "})
    assert r1.status_code == 422

    # Oversized question (> 1000 chars)
    long_q = "a" * 1001
    r2 = await async_client.post("/query", json={"question": long_q})
    assert r2.status_code == 422


@pytest.mark.asyncio
async def test_ingest_and_query_success_flow(async_client: AsyncClient):
    # 1. Ingest a valid Markdown file
    md_content = b"# FastAPI Tutorial\n\n## Path Params\nPath parameters are declared with format strings."
    files = {"files": ("tutorial.md", md_content, "text/markdown")}

    ingest_res = await async_client.post("/ingest", files=files)
    assert ingest_res.status_code == 200
    ingest_data = ingest_res.json()
    assert ingest_data["status"] == "success"
    assert ingest_data["indexed_documents"] == 1
    assert ingest_data["total_chunks"] >= 1
    assert ingest_data["indexed_chunks"] == ingest_data["total_chunks"]
    assert ingest_data["document_id"] is not None

    # 2. Check GET /documents
    docs_res = await async_client.get("/documents")
    assert docs_res.status_code == 200
    docs_data = docs_res.json()
    assert docs_data["total_documents"] >= 1

    # 3. Query the assistant
    query_res = await async_client.post(
        "/query", json={"question": "How do path parameters work?"}
    )
    assert query_res.status_code == 200
    q_data = query_res.json()
    assert q_data["status"] == "answered"
    assert q_data["query_id"]
    assert q_data["answer"]
    assert q_data["citations"]
    assert q_data["grounding_verified"] is True
    assert q_data["web_fallback_used"] is False

    # 4. Submit feedback for the query
    fb_res = await async_client.post(
        "/feedback",
        json={
            "query_id": q_data["query_id"],
            "rating": "up",
            "comment": "Accurate and well cited!",
        },
    )
    assert fb_res.status_code == 201
    fb_data = fb_res.json()
    assert fb_data["status"] == "recorded"
    assert fb_data["feedback_id"]


@pytest.mark.asyncio
async def test_feedback_nonexistent_query_returns_404(async_client: AsyncClient):
    response = await async_client.post(
        "/feedback",
        json={"query_id": "non_existent_uuid_12345", "rating": "down"},
    )
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_feedback_invalid_rating_returns_422(async_client: AsyncClient):
    response = await async_client.post(
        "/feedback",
        json={"query_id": "some_id", "rating": "maybe"},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_ingest_unsupported_file_type_returns_415(async_client: AsyncClient):
    files = {"files": ("sample.json", b'{"key": "value"}', "application/json")}
    response = await async_client.post("/ingest", files=files)
    assert response.status_code == 415
    assert "unsupported file type" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_ingest_path_traversal_returns_422(async_client: AsyncClient):
    files = {"files": ("../../etc/passwd.md", b"# Malicious", "text/markdown")}
    response = await async_client.post("/ingest", files=files)
    assert response.status_code == 422
    assert "insecure filename" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_ingest_empty_file_returns_422(async_client: AsyncClient):
    files = {"files": ("empty.md", b"", "text/markdown")}
    response = await async_client.post("/ingest", files=files)
    assert response.status_code == 422
    assert "empty" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_ingest_json_invalid_url_returns_422(async_client: AsyncClient):
    response = await async_client.post(
        "/ingest",
        json={"url": "ftp://not-supported-scheme.com/doc.md", "source_type": "url"},
    )
    assert response.status_code == 422
    assert "url must start with http" in response.json()["detail"].lower()
