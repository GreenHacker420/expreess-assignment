"""Pytest fixtures for offline, zero-network unit and integration testing."""

import os
from typing import AsyncGenerator
import pytest
from httpx import ASGITransport, AsyncClient

from backend.app.core.config import Settings
from backend.app.graph.builder import build_rag_graph
from backend.app.main import create_app
from backend.app.services.database import MockDatabaseService
from backend.app.services.embeddings import MockEmbeddingsService
from backend.app.services.ingestion import IngestionService
from backend.app.services.llm import MockLLMService
from backend.app.services.vector_store import MockVectorStoreService


@pytest.fixture
def mock_embeddings():
    """Deterministic mock embedding service."""
    return MockEmbeddingsService()


@pytest.fixture
def mock_vector_store():
    """In-memory mock vector store."""
    return MockVectorStoreService()


@pytest.fixture
def mock_database():
    """In-memory mock database."""
    return MockDatabaseService()


@pytest.fixture
def mock_llm():
    """Deterministic mock LLM service."""
    return MockLLMService()


@pytest.fixture
def test_ingestion(mock_embeddings, mock_vector_store, mock_database):
    """Ingestion service wired with mock dependencies."""
    return IngestionService(
        embeddings=mock_embeddings,
        vector_store=mock_vector_store,
        database=mock_database,
        target_chunk_tokens=200,
        overlap_tokens=35,
    )


@pytest.fixture
def test_graph(mock_llm, mock_embeddings, mock_vector_store):
    """Compiled LangGraph StateGraph wired with mock services."""
    return build_rag_graph(
        llm=mock_llm,
        embeddings=mock_embeddings,
        vector_store=mock_vector_store,
        top_k=5,
        grading_concurrency=3,
    )


@pytest.fixture
async def async_client(
    mock_llm, mock_embeddings, mock_vector_store, mock_database, test_ingestion, test_graph
) -> AsyncGenerator[AsyncClient, None]:
    """Async HTTP client with injected mock services for testing."""
    test_settings = Settings(
        GROQ_API_KEY="mock_key",
        CHROMA_DIR="data/test_chroma",
        DB_PATH="data/test_app.db",
        MAX_QUESTION_LENGTH=1000,
        MAX_UPLOAD_SIZE_BYTES=1024 * 1024,
    )

    app = create_app(custom_settings=test_settings)

    # Inject mock services directly into app state
    app.state.database = mock_database
    app.state.vector_store = mock_vector_store
    app.state.embeddings = mock_embeddings
    app.state.ingestion = test_ingestion
    app.state.llm = mock_llm
    app.state.graph = test_graph

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
