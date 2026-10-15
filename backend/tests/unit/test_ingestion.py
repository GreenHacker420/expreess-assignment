"""Unit tests for the ingestion service and chunking algorithms."""

import pytest
from backend.app.services.ingestion import IngestionService


def test_content_normalization(test_ingestion: IngestionService):
    raw = "def foo():\r\n    return 42    \r\n"
    normalized = test_ingestion.normalize_content(raw)
    assert "\r" not in normalized
    # Code indentation preserved
    assert "    return 42" in normalized


def test_heading_aware_chunking(test_ingestion: IngestionService):
    markdown = """# FastAPI Guide

## Path Parameters
You can declare path parameters using format string syntax.

```python
@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}
```

## Query Parameters
Other function parameters are interpreted as query parameters.
"""
    result = test_ingestion.ingest_text(
        source="test_guide.md",
        title="FastAPI Guide",
        raw_content=markdown,
        document_id="doc_guide",
    )

    assert result.status == "indexed"
    assert result.chunk_count >= 2

    # Verify chunks stored in vector store retain section metadata
    stored_chunks = list(test_ingestion.vector_store.chunks.values())
    sections = [c["metadata"]["section"] for c in stored_chunks]
    assert "Path Parameters" in sections
    assert "Query Parameters" in sections

    # Code block preserved intact
    code_found = any("@app.get(\"/items/{item_id}\")" in c["text"] for c in stored_chunks)
    assert code_found


def test_idempotent_ingestion(test_ingestion: IngestionService):
    markdown = "# Simple Doc\n\nContent paragraph here."
    res1 = test_ingestion.ingest_text("doc1.md", "Doc 1", markdown)
    assert res1.status == "indexed"
    initial_chunks = len(test_ingestion.vector_store.chunks)

    # Re-ingest exact same content
    res2 = test_ingestion.ingest_text("doc1.md", "Doc 1", markdown)
    assert res2.status == "skipped"
    # Chunk count unchanged
    assert len(test_ingestion.vector_store.chunks) == initial_chunks


def test_update_removes_stale_chunks(test_ingestion: IngestionService):
    doc_id = "doc_update_test"
    # Long version with multiple sections
    v1 = "# Long Doc\n\n## S1\nText 1\n\n## S2\nText 2\n\n## S3\nText 3"
    res1 = test_ingestion.ingest_text("update_doc.md", "Update Doc", v1, document_id=doc_id)
    assert res1.status == "indexed"
    count_v1 = res1.chunk_count

    # Short version with fewer sections
    v2 = "# Short Doc\n\n## S1\nOnly text 1 remains."
    res2 = test_ingestion.ingest_text("update_doc.md", "Short Doc", v2, document_id=doc_id)
    assert res2.status == "updated"
    assert res2.chunk_count < count_v1

    # Verify no stale chunks remain in vector store
    current_doc_chunks = [
        c for c in test_ingestion.vector_store.chunks.values()
        if c["metadata"]["document_id"] == doc_id
    ]
    assert len(current_doc_chunks) == res2.chunk_count
