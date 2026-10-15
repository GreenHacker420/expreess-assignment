"""Optional Tavily web-search fallback service."""

import asyncio
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class WebSearchService(Protocol):
    """Minimal interface consumed by the RAG graph."""

    async def search(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        """Return web results containing title, URL, and extractable text."""
        ...


class TavilyWebSearchService:
    """Tavily search client, run in a worker thread to avoid blocking FastAPI."""

    def __init__(self, api_key: str, timeout_seconds: float = 15.0):
        if not api_key:
            raise ValueError("TAVILY_API_KEY is required to enable web search.")
        from tavily import TavilyClient

        self._client = TavilyClient(api_key=api_key)
        self.timeout_seconds = timeout_seconds

    async def search(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        """Fetch concise search snippets without Tavily-generated answer text."""
        response = await asyncio.to_thread(
            self._client.search,
            query=query,
            search_depth="basic",
            max_results=max_results,
            include_answer=False,
            include_raw_content=False,
            timeout=self.timeout_seconds,
        )
        results = response.get("results", []) if isinstance(response, dict) else []
        return [item for item in results if isinstance(item, dict)]
