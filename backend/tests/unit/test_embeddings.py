"""Tests for provider-specific Gemini retrieval embedding formatting."""

from backend.app.services.embeddings import GeminiEmbeddingsService


def test_gemini_embedding_2_formats_documents_asymmetric_to_queries():
    document = "[FastAPI Tutorial > Request Files]\n\nUploadFile uses a spooled temporary file."

    prepared_document = GeminiEmbeddingsService._format_document_for_embedding(document)
    prepared_query = GeminiEmbeddingsService._format_query_for_embedding(
        "How does UploadFile work?"
    )

    assert prepared_document == (
        "title: FastAPI Tutorial | text: section: Request Files\n"
        "UploadFile uses a spooled temporary file."
    )
    assert prepared_query == "task: question answering | query: How does UploadFile work?"
