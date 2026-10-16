"""Unit tests for pluggable web search services (Gemini, Exa, Tavily)."""

from unittest.mock import MagicMock, patch
import pytest

from backend.app.core.config import Settings
from backend.app.services.web_search import (
    ExaWebSearchService,
    GeminiGoogleSearchService,
    TavilyWebSearchService,
    WebSearchService,
    create_web_search_service,
)


def test_create_web_search_service_gemini():
    settings = Settings(
        web_search_enabled=True,
        web_search_provider="gemini",
        gemini_api_key="test-gemini-key",
    )
    service = create_web_search_service(settings)
    assert isinstance(service, GeminiGoogleSearchService)
    assert isinstance(service, WebSearchService)


def test_create_web_search_service_exa():
    settings = Settings(
        web_search_enabled=True,
        web_search_provider="exa",
        exa_api_key="test-exa-key",
    )
    service = create_web_search_service(settings)
    assert isinstance(service, ExaWebSearchService)
    assert isinstance(service, WebSearchService)


def test_create_web_search_service_tavily():
    settings = Settings(
        web_search_enabled=True,
        web_search_provider="tavily",
        tavily_api_key="test-tavily-key",
    )
    service = create_web_search_service(settings)
    assert isinstance(service, TavilyWebSearchService)
    assert isinstance(service, WebSearchService)


def test_create_web_search_service_disabled_returns_none():
    settings = Settings(web_search_enabled=False)
    assert create_web_search_service(settings) is None


@pytest.mark.asyncio
async def test_gemini_google_search_service_formats_snippets():
    service = GeminiGoogleSearchService(api_key="fake-key", model_name="gemini-3.8-flash")
    mock_chunk = MagicMock()
    mock_chunk.web.uri = "https://fastapi.tiangolo.com/tutorial/path-params/"
    mock_chunk.web.title = "Path Parameters - FastAPI"

    mock_candidate = MagicMock()
    mock_candidate.grounding_metadata.grounding_chunks = [mock_chunk]

    mock_response = MagicMock()
    mock_response.candidates = [mock_candidate]
    mock_response.text = "In FastAPI, path parameters are declared using Python format strings."

    with patch.object(service._client.models, "generate_content", return_value=mock_response):
        results = await service.search("FastAPI path parameters", max_results=3)

    assert len(results) == 1
    assert results[0]["url"] == "https://fastapi.tiangolo.com/tutorial/path-params/"
    assert results[0]["title"] == "Path Parameters - FastAPI"
    assert "FastAPI" in results[0]["content"]


@pytest.mark.asyncio
async def test_exa_web_search_service_formats_snippets():
    service = ExaWebSearchService(api_key="fake-exa-key")
    mock_item = MagicMock()
    mock_item.url = "https://fastapi.tiangolo.com/advanced/"
    mock_item.title = "Advanced User Guide - FastAPI"
    mock_item.text = "Advanced dependencies and settings..."

    mock_response = MagicMock()
    mock_response.results = [mock_item]

    with patch.object(service._client, "search_and_contents", return_value=mock_response):
        results = await service.search("FastAPI advanced", max_results=2)

    assert len(results) == 1
    assert results[0]["url"] == "https://fastapi.tiangolo.com/advanced/"
    assert results[0]["title"] == "Advanced User Guide - FastAPI"
    assert results[0]["content"] == "Advanced dependencies and settings..."
