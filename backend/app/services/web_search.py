"""Pluggable web-search fallback services (Gemini Google Search Grounding, Exa, Tavily)."""

import asyncio
import logging
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger("web_search")


@runtime_checkable
class WebSearchService(Protocol):
    """Minimal interface consumed by the RAG graph."""

    async def search(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        """Return web results containing title, URL, and extractable text."""
        ...


class GeminiGoogleSearchService:
    """Gemini native Google Search grounding service.

    Uses Gemini 3.8/2.5 with Google Search tool enabled to retrieve live web-grounded snippets.
    Zero external API key needed when GEMINI_API_KEY is configured.
    """

    def __init__(self, api_key: str, model_name: str = "gemini-3.8-flash"):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for Gemini Google Search grounding.")
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self.model_name = model_name

    async def search(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        """Fetch web snippets using Gemini Google Search grounding."""
        from google.genai import types

        logger.info("Executing Gemini Google Search grounding for query: %s", query)
        prompt = (
            f"Search Google and provide concise, factual technical documentation excerpts and reference details "
            f"to answer the question: {query}"
        )
        response = await asyncio.to_thread(
            self._client.models.generate_content,
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
            ),
        )

        results: list[dict[str, Any]] = []
        candidates = getattr(response, "candidates", []) or []
        if candidates and hasattr(candidates[0], "grounding_metadata") and candidates[0].grounding_metadata:
            gm = candidates[0].grounding_metadata
            chunks = getattr(gm, "grounding_chunks", []) or []
            for c in chunks[:max_results]:
                web = getattr(c, "web", None)
                if web and getattr(web, "uri", None):
                    results.append(
                        {
                            "url": web.uri,
                            "title": getattr(web, "title", "") or "Google Search Reference",
                            "content": response.text or "",
                        }
                    )

        if not results and response and getattr(response, "text", None):
            results.append(
                {
                    "url": f"https://www.google.com/search?q={query.replace(' ', '+')}",
                    "title": "Google Search Grounded Result",
                    "content": response.text,
                }
            )

        return results


class ExaWebSearchService:
    """Exa neural search client for developer technical documentation."""

    def __init__(self, api_key: str, timeout_seconds: float = 15.0):
        if not api_key:
            raise ValueError("EXA_API_KEY is required to enable Exa web search.")
        from exa_py import Exa

        self._client = Exa(api_key=api_key)
        self.timeout_seconds = timeout_seconds

    async def search(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        """Fetch search results and contents from Exa."""
        logger.info("Executing Exa neural search for query: %s", query)
        response = await asyncio.to_thread(
            self._client.search_and_contents,
            query=query,
            num_results=max_results,
            text={"max_characters": 3000},
        )
        results: list[dict[str, Any]] = []
        raw_items = getattr(response, "results", []) or []
        for item in raw_items:
            results.append(
                {
                    "url": getattr(item, "url", ""),
                    "title": getattr(item, "title", "") or "Exa Technical Result",
                    "content": getattr(item, "text", ""),
                }
            )
        return results


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
        logger.info("Executing Tavily web search for query: %s", query)
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


def create_web_search_service(settings: Any) -> WebSearchService | None:
    """Factory creating the configured web search provider service."""
    if not getattr(settings, "web_search_enabled", False):
        return None

    provider = getattr(settings, "web_search_provider", "gemini").lower()

    if provider == "gemini":
        api_key = getattr(settings, "gemini_api_key", None)
        if not api_key:
            logger.warning("Gemini web search enabled but GEMINI_API_KEY is missing.")
            return None
        model_name = getattr(settings, "gemini_fast_model", "gemini-3.8-flash")
        return GeminiGoogleSearchService(api_key=api_key, model_name=model_name)

    if provider == "exa":
        api_key = getattr(settings, "exa_api_key", None)
        if not api_key:
            logger.warning("Exa web search selected but EXA_API_KEY is missing.")
            return None
        return ExaWebSearchService(api_key=api_key)

    if provider == "tavily":
        api_key = getattr(settings, "tavily_api_key", None)
        if not api_key:
            logger.warning("Tavily web search selected but TAVILY_API_KEY is missing.")
            return None
        return TavilyWebSearchService(api_key=api_key)

    logger.warning("Unknown web search provider: %s", provider)
    return None
